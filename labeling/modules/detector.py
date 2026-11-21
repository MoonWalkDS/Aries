"""Stage detect: phát hiện thực thể bằng Florence-2 và cắt crop đa mức.

Mức crop:
- Layer 0: ảnh gốc, không cắt (không lưu lại file, chỉ ghi kích thước + caption toàn cảnh).
- Layer 2: từng thực thể, box nới thêm ``pad_margin`` mỗi phía để không cắt cụt mép.

Phần xử lý box (chuẩn hoá nhãn, lọc, NMS, đánh ID) là hàm thuần, test được không cần model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .schemas import BBox, Detection, DetectionRecord
from .utils import Box, box_area, clamp_box, get_logger, image_id_of, iou, pad_box

TASK_CAPTION = "<MORE_DETAILED_CAPTION>"
TASK_OD = "<OD>"
TASK_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
TASK_OPEN_VOCAB = "<OPEN_VOCABULARY_DETECTION>"

RawBox = tuple[Box, str, str]  # (box, nhãn, tác vụ nguồn)


# --------------------------------------------------------------------------- hàm thuần
def normalize_label(label: str, person_aliases: set[str]) -> tuple[str, bool]:
    """Chuẩn hoá nhãn Florence; mọi biến thể chỉ người quy về 'person'."""
    clean = " ".join(label.lower().replace("_", " ").split()).strip(" .")
    words = set(clean.split())
    if clean in person_aliases or words & person_aliases:
        return "person", True
    return clean, False


def filter_and_merge(
    raw: list[RawBox],
    width: int,
    height: int,
    person_aliases: set[str],
    min_area_frac: float,
    max_area_frac: float,
    nms_iou: float,
) -> list[tuple[Box, str, bool, str]]:
    """Kẹp biên, bỏ box quá nhỏ/quá lớn, gộp box cùng nhãn trùng nhau (giữ box gặp trước)."""
    image_area = float(width * height)
    kept: list[tuple[Box, str, bool, str]] = []
    for box, label, source in raw:
        box = clamp_box(box, width, height)
        frac = box_area(box) / image_area if image_area else 0.0
        if frac < min_area_frac or frac > max_area_frac:
            continue
        name, is_person = normalize_label(label, person_aliases)
        if not name:
            continue
        if any(k_name == name and iou(box, k_box) >= nms_iou for k_box, k_name, _, _ in kept):
            continue
        kept.append((box, name, is_person, source))
    return kept


def assign_entity_ids(items: list[tuple[Box, str, bool, str]]) -> list[tuple[str, Box, str, bool, str]]:
    """Đánh ID ổn định: người p01.. theo thứ tự trái sang phải, đồ vật o01.. tương tự."""
    ordered = sorted(items, key=lambda it: (it[0][0], it[0][1]))
    out: list[tuple[str, Box, str, bool, str]] = []
    n_person = n_object = 0
    for box, name, is_person, source in ordered:
        if is_person:
            n_person += 1
            entity_id = f"p{n_person:02d}"
        else:
            n_object += 1
            entity_id = f"o{n_object:02d}"
        out.append((entity_id, box, name, is_person, source))
    return out


def build_record(
    image_path: Path,
    width: int,
    height: int,
    caption: str,
    raw: list[RawBox],
    cfg: dict[str, Any],
) -> DetectionRecord:
    """Từ box thô của Florence dựng DetectionRecord (chưa lưu crop)."""
    det_cfg = cfg["detector"]
    merged = filter_and_merge(
        raw,
        width,
        height,
        person_aliases={a.lower() for a in det_cfg["person_aliases"]},
        min_area_frac=float(det_cfg["min_box_area_frac"]),
        max_area_frac=float(det_cfg["max_box_area_frac"]),
        nms_iou=float(det_cfg["nms_iou"]),
    )
    margin = float(cfg["thresholds"]["pad_margin"])
    entities = [
        Detection(
            entity_id=entity_id,
            label=name,
            is_person=is_person,
            box=BBox.from_seq(box),
            padded_box=BBox.from_seq(pad_box(box, margin, width, height)),
            source=source,
        )
        for entity_id, box, name, is_person, source in assign_entity_ids(merged)
    ]
    return DetectionRecord(
        image_id=image_id_of(image_path),
        file_name=image_path.name,
        width=width,
        height=height,
        global_caption=caption,
        entities=entities,
    )


def save_crops(image: Image.Image, record: DetectionRecord, crops_dir: Path, save_objects: bool) -> None:
    """Lưu crop Layer 2 vào ``output_crops/<image_id>/<entity_id>.jpg`` và ghi crop_path vào record."""
    out_dir = crops_dir / record.image_id
    for entity in record.entities:
        if not entity.is_person and not save_objects:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in entity.padded_box.as_tuple())
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        rel = f"{record.image_id}/{entity.entity_id}.jpg"
        image.crop((x1, y1, x2, y2)).save(crops_dir / rel, quality=92)
        entity.crop_path = rel


# --------------------------------------------------------------------------- model
class SceneDetector:
    """Bọc Florence-2: caption toàn cảnh + phát hiện vật thể + phát hiện theo từ vựng mở."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor

        self.cfg = cfg
        det_cfg = cfg["detector"]
        self.log = get_logger()
        self._torch = torch

        device = det_cfg.get("device", "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # float16 trên CPU rất chậm và thiếu toán tử, nên CPU luôn dùng float32.
        self.dtype = torch.float16 if (device == "cuda" and det_cfg.get("dtype") == "float16") else torch.float32

        model_id = det_cfg["model_id"]
        cache_dir = cfg["paths"].get("model_cache")
        # Hai cách nạp Florence-2:
        # - Lớp gốc của transformers (Florence2ForConditionalGeneration, có từ 4.56) với checkpoint
        #   đã chuyển đổi "florence-community/...". Đây là cách chạy được trên transformers 5.x.
        # - Mã remote của "microsoft/..." (trust_remote_code). Hỏng trên transformers 5.x với lỗi
        #   'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'.
        native_cls = getattr(transformers, "Florence2ForConditionalGeneration", None)
        use_native = native_cls is not None and not model_id.startswith("microsoft/")
        self.log.info("Nạp %s trên %s (%s, %s)", model_id, device, self.dtype, "lớp gốc" if use_native else "mã remote")
        if use_native:
            self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            self.model = native_cls.from_pretrained(model_id, dtype=self.dtype, cache_dir=cache_dir).to(device).eval()
        else:
            from transformers import AutoModelForCausalLM

            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, cache_dir=cache_dir)
            self.model = (
                AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=self.dtype, trust_remote_code=True, cache_dir=cache_dir)
                .to(device)
                .eval()
            )

    def _run(self, image: Image.Image, task: str, text: Optional[str] = None) -> Any:
        """Chạy một tác vụ Florence-2 và trả kết quả đã hậu xử lý về toạ độ pixel."""
        det_cfg = self.cfg["detector"]
        prompt = task + (text or "")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device, self.dtype)
        with self._torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=int(det_cfg["max_new_tokens"]),
                num_beams=int(det_cfg["num_beams"]),
                do_sample=False,
            )
        decoded = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(decoded, task=task, image_size=image.size)
        return parsed.get(task, parsed)

    @staticmethod
    def _boxes_from(result: Any, source: str) -> list[RawBox]:
        """Gom box + nhãn từ kết quả <OD>/<CAPTION_TO_PHRASE_GROUNDING>/<OPEN_VOCABULARY_DETECTION>."""
        if not isinstance(result, dict):
            return []
        boxes = result.get("bboxes") or []
        labels = result.get("labels") or result.get("bboxes_labels") or []
        out: list[RawBox] = []
        for box, label in zip(boxes, labels):
            if len(box) == 4 and str(label).strip():
                x1, y1, x2, y2 = (float(v) for v in box)
                out.append(((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), str(label), source))
        return out

    def detect(self, image_path: Path, crops_dir: Path) -> DetectionRecord:
        """Xử lý một ảnh: caption, phát hiện, lọc/gộp box, lưu crop. Trả DetectionRecord."""
        det_cfg = self.cfg["detector"]
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        width, height = image.size

        caption = self._run(image, TASK_CAPTION)
        caption = caption.strip() if isinstance(caption, str) else ""

        # Box từ <OD> đặt trước: khi trùng box grounding cùng nhãn thì giữ box <OD>.
        raw = self._boxes_from(self._run(image, TASK_OD), "od")
        classes: list[str] = list(det_cfg.get("grounding_classes") or [])
        if classes and det_cfg.get("open_vocab_mode", "grounding") == "per_class":
            for name in classes:
                raw += self._boxes_from(self._run(image, TASK_OPEN_VOCAB, name), "open_vocab")
        elif classes:
            raw += self._boxes_from(self._run(image, TASK_GROUNDING, ". ".join(classes) + "."), "grounding")

        record = build_record(image_path, width, height, caption, raw, self.cfg)
        save_crops(image, record, crops_dir, bool(det_cfg.get("save_object_crops", True)))
        return record

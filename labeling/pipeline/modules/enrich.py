"""Làm giàu dữ liệu theo tầng: tầng NGƯỜI và tầng VÙNG CỐ ĐỊNH (``python main.py enrich ...``).

Khác stage ``adl`` (một nhãn cho cả ảnh), ở đây đơn vị là crop:

- Tầng người: RT-DETR dò người, mỗi người một crop ở độ phân giải gốc, một vector SigLIP2, và một
  ``zone`` (đang đứng ở vùng nào) suy từ hình học vì camera cố định.
- Tầng vùng cố định: các vùng khai báo trong ``rois.yaml`` (cửa lò, bồn rửa, tủ lạnh...). Mỗi vùng
  được so với nền của giờ đó; chỉ khi vùng THAY ĐỔI mới cắt và mã hoá, nên vật nhỏ được nhìn ở độ
  phân giải gốc thay vì bị thu về 384 px cùng cả khung hình. Mỗi giờ có một "ảnh nền" (ảnh ít
  chuyển động nhất) làm đại diện cho trạng thái bình thường của mọi vùng trong giờ đó.

``build`` chỉ trích xuất. Gán nhãn (hành động của người, trạng thái của vùng) dùng lại nguyên vòng
lặp gợi ý - duyệt của stage adl qua ``--profile``:

    python main.py adl init  --profile people        # hành động của từng người
    python main.py adl round --profile people
    python main.py adl init  --profile roi-stove      # trạng thái của một vùng
    python main.py adl round --profile roi-stove

``merge`` gộp mọi thứ thành một bản ghi cho mỗi ảnh: ``output_jsons/enriched/<image_id>.json``.

Bố cục ``work/enrich/``:
    people/embeddings.npy, ids.json, people.json, crops/<image_id>_pNN.jpg
    rois/<vùng>/embeddings.npy, ids.json, labels.yaml, crops/<image_id>.jpg (và bg-<giờ>.jpg)
    rois/motion.npy, roi_names.json, image_ids.json, baseline.json
"""
from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import yaml
from PIL import Image

from .adl_embed import background_of, group_key, load_small
from .utils import get_logger, image_id_of, read_json, resolve_from_config, write_json

RelBox = tuple[float, float, float, float]
BASELINE_PREFIX = "bg-"
MANUAL_BOXES_FILE = "manual_boxes.json"  # box người dùng tự vẽ bằng label_viewer/, trong annotations/
CUSTOM_LABELS_FILE = "custom_labels.json"  # nhãn người tự gõ trên label_viewer/ khi không lớp nào khớp: {profile: {khoá: {text}}}
OBJECTS_FILE = "objects.json"  # đồ vật do `enrich objects` dò, trong work/enrich/
OWL_OBJECTS_FILE = "objects_owl.json"  # đồ vật do `enrich objects --owl` (OWLv2, theo tên) dò, trong work/enrich/


# --------------------------------------------------------------------------- vùng cố định
@dataclass(frozen=True)
class Roi:
    name: str
    name_vi: str
    box: RelBox
    zone: RelBox
    states: tuple[dict[str, Any], ...]


def _rel_box(value: Any, where: str) -> RelBox:
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: cần 4 số [x1, y1, x2, y2], nhận được {value!r}") from exc
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"{where}: toạ độ phải là tỉ lệ 0..1 với x1 < x2, y1 < y2; nhận được {value!r}")
    return (x1, y1, x2, y2)


def load_rois(path: str | Path) -> list[Roi]:
    with open(path, encoding="utf-8") as f:
        raw = (yaml.safe_load(f) or {}).get("rois") or []
    rois: list[Roi] = []
    for item in raw:
        name = str(item.get("name", "")).strip()
        if not name or any(r.name == name for r in rois) or not name.replace("_", "").isalnum():
            raise ValueError(f"{path}: tên vùng rỗng, trùng hoặc có ký tự lạ (chỉ chữ, số, gạch dưới): {name!r}")
        box = _rel_box(item.get("box"), f"{path}: {name}.box")
        zone = _rel_box(item.get("zone", item.get("box")), f"{path}: {name}.zone")
        rois.append(Roi(name, str(item.get("name_vi", "")).strip(), box, zone, tuple(item.get("states") or [])))
    if not rois:
        raise ValueError(f"{path}: không có vùng nào")
    return rois


def write_roi_labels(roi: Roi, path: Path) -> None:
    """Ghi các trạng thái của vùng thành file nhãn theo định dạng của stage adl."""
    classes = [
        {"name": s["name"], "name_vi": s.get("name_vi", ""), "group": roi.name, "descriptions": list(s.get("descriptions") or [])}
        for s in roi.states
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# Trạng thái của vùng {roi.name!r}, sinh từ rois.yaml bởi `enrich build`. Sửa ở rois.yaml, không sửa ở đây.\n"
    path.write_text(header + yaml.safe_dump({"classes": classes}, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")


def roi_pixels(box: RelBox, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    left, top = int(x1 * width), int(y1 * height)
    return left, top, max(left + 1, int(round(x2 * width))), max(top + 1, int(round(y2 * height)))


def roi_motion(
    image_paths: Sequence[Path], rois: Sequence[Roi], size: tuple[int, int], threshold: float,
    max_background_frames: int = 200, workers: int = 8,
) -> tuple[np.ndarray, dict[str, dict[str, str]]]:
    """(ma trận ảnh x vùng: tỉ lệ pixel của vùng đã đổi so với nền của giờ, {giờ: {vùng: image_id ảnh nền}}).

    Ảnh nền của một vùng trong một giờ là ảnh mà vùng đó ít thay đổi nhất, tức là gần với trạng thái
    bình thường của vùng trong giờ đó nhất (không bị người che).
    """
    with ThreadPoolExecutor(workers) as pool:
        small = np.stack(list(pool.map(lambda p: load_small(p, size), image_paths)))
    w, h = size
    boxes = [roi_pixels(r.box, w, h) for r in rois]
    groups: dict[str, list[int]] = {}
    for i, path in enumerate(image_paths):
        groups.setdefault(group_key(image_id_of(path)), []).append(i)
    motion = np.zeros((len(image_paths), len(rois)), dtype=np.float32)
    baseline: dict[str, dict[str, str]] = {}
    for key, members in groups.items():
        step = max(1, len(members) // max_background_frames)
        changed = np.abs(small[members] - background_of(small[members[::step]])) > threshold
        for r, (x1, y1, x2, y2) in enumerate(boxes):
            motion[members, r] = changed[:, y1:y2, x1:x2].mean(axis=(1, 2))
            baseline.setdefault(key, {})[rois[r].name] = image_id_of(image_paths[members[int(motion[members, r].argmin())]])
    return motion, baseline


# --------------------------------------------------------------------------- người
def pad_pixels(box: Sequence[float], margin: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * margin, (y2 - y1) * margin
    return max(0, int(x1 - px)), max(0, int(y1 - py)), min(width, int(round(x2 + px))), min(height, int(round(y2 + py)))


def assign_zone(box: Sequence[float], width: int, height: int, rois: Sequence[Roi], min_overlap: float) -> tuple[Optional[str], float]:
    """Vùng mà người này "đang ở": zone phủ nhiều diện tích box người nhất, nếu đạt ``min_overlap``."""
    x1, y1, x2, y2 = box
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    best: tuple[Optional[str], float] = (None, 0.0)
    for roi in rois:
        zx1, zy1, zx2, zy2 = roi.zone[0] * width, roi.zone[1] * height, roi.zone[2] * width, roi.zone[3] * height
        overlap = max(0.0, min(x2, zx2) - max(x1, zx1)) * max(0.0, min(y2, zy2) - max(y1, zy1)) / area
        if overlap > best[1]:
            best = (roi.name, overlap)
    return (best[0], round(best[1], 3)) if best[1] >= min_overlap else (None, round(best[1], 3))


def suppress_duplicates(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """NMS đơn giản: bỏ box trùng nhiều với một box điểm cao hơn. Trả chỉ số được giữ, theo điểm giảm dần."""
    order = np.argsort(-scores)
    keep: list[int] = []
    for i in order:
        x1, y1, x2, y2 = boxes[i]
        duplicate = False
        for j in keep:
            a1, b1, a2, b2 = boxes[j]
            inter = max(0.0, min(x2, a2) - max(x1, a1)) * max(0.0, min(y2, b2) - max(y1, b1))
            union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - inter
            if union > 0 and inter / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(int(i))
    return keep


class PersonDetector:
    """Dò người bằng RT-DETR (transformers). Trả về, cho mỗi ảnh, mảng box pixel và mảng điểm."""

    def __init__(self, model_id: str, threshold: float, device: str = "auto") -> None:
        import torch
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        self._torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.threshold = threshold
        self.processor = RTDetrImageProcessor.from_pretrained(model_id)
        self.model = RTDetrForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.person_ids = {i for i, name in self.model.config.id2label.items() if name == "person"}

    def detect(self, images: Sequence[Image.Image]) -> list[tuple[np.ndarray, np.ndarray]]:
        torch = self._torch
        with torch.inference_mode():
            inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
        sizes = torch.tensor([im.size[::-1] for im in images])
        results = self.processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=self.threshold)
        out = []
        for result in results:
            mask = [int(label) in self.person_ids for label in result["labels"]]
            out.append((result["boxes"][mask].float().cpu().numpy().reshape(-1, 4), result["scores"][mask].float().cpu().numpy()))
        return out

    def detect_all(self, images: Sequence[Image.Image]) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
        """Mọi lớp COCO: (box pixel, điểm, tên lớp) cho mỗi ảnh."""
        torch = self._torch
        with torch.inference_mode():
            inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
        sizes = torch.tensor([im.size[::-1] for im in images])
        results = self.processor.post_process_object_detection(outputs, target_sizes=sizes, threshold=self.threshold)
        names = self.model.config.id2label
        return [(r["boxes"].float().cpu().numpy().reshape(-1, 4), r["scores"].float().cpu().numpy(), [names[int(k)] for k in r["labels"]]) for r in results]


def run_owl_objects(cfg: dict[str, Any], image_paths: Sequence[Path], data_dir: Path, force: bool = False) -> dict[str, Any]:
    """Pha dò đồ vật theo tên (``enrich objects --owl``): OWLv2 tìm các vật mà lớp COCO của RT-DETR không có
    (nồi, chảo, ấm, thớt, khăn...). Ghi ``work/enrich/objects_owl.json``, mã ``<image_id>_wNN``. Chạy lại chỉ dò ảnh chưa có.
    """
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    ocfg, log = cfg["enrich"]["objects"]["owl"], get_logger()
    out_path = data_dir / OWL_OBJECTS_FILE
    cached: dict[str, list[dict[str, Any]]] = {} if force or not out_path.exists() else read_json(out_path)
    todo = [p for p in image_paths if image_id_of(p) not in cached]
    queries = list(ocfg["queries"])
    if todo:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        processor = Owlv2Processor.from_pretrained(ocfg["model_id"])
        model = Owlv2ForObjectDetection.from_pretrained(ocfg["model_id"], torch_dtype=dtype).to(device).eval()
        prompts = [f"a photo of a {q}" for q in queries]
        started = time.perf_counter()
        for n, path in enumerate(todo, start=1):
            image = Image.open(path).convert("RGB")
            inputs = processor(text=[prompts], images=image, return_tensors="pt").to(device)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            with torch.inference_mode():
                outputs = model(**inputs)
            side = max(image.size)  # OWLv2 đệm ảnh thành hình vuông: toạ độ tính theo cạnh dài
            r = processor.post_process_grounded_object_detection(outputs, threshold=float(ocfg["threshold"]), target_sizes=torch.tensor([[side, side]]))[0]
            boxes, scores = r["boxes"].float().cpu().numpy().reshape(-1, 4), r["scores"].float().cpu().numpy()
            names = [queries[int(k)] for k in r["labels"]]
            keep = []
            for label in sorted(set(names)):
                idx = [k for k, q in enumerate(names) if q == label]
                keep += [idx[j] for j in suppress_duplicates(boxes[idx], scores[idx], float(ocfg["nms_iou"]))]
            keep = sorted(sorted(keep, key=lambda k: -scores[k])[: int(ocfg["max_per_image"])], key=lambda k: (boxes[k][0], boxes[k][1]))
            image_id = image_id_of(path)
            w, h = image.size
            cached[image_id] = [{"id": f"{image_id}_w{m:02d}", "label": names[k], "box": [round(float(min(max(v, 0), lim)), 1) for v, lim in zip(boxes[k], (w, h, w, h))],
                                 "score": round(float(scores[k]), 3)} for m, k in enumerate(keep, start=1)]
            if n % 500 == 0:
                write_json(out_path, cached)
                log.info("enrich: OWLv2 %d/%d ảnh, %.2f s/ảnh", n, len(todo), (time.perf_counter() - started) / n)
        write_json(out_path, cached)
        log.info("enrich: OWLv2 xong %d ảnh sau %.1f phút", len(todo), (time.perf_counter() - started) / 60)
    from collections import Counter

    counts = Counter(o["label"] for objs in cached.values() for o in objs)
    return {"images": len(cached), "objects": sum(counts.values()), "per_label": dict(counts.most_common())}


def static_object_ids(objects: dict[str, list[dict[str, Any]]], iou: float, min_fraction: float) -> tuple[set[str], list[dict[str, Any]]]:
    """Mã các box nằm ở một chỗ cố định: cùng nhãn, IoU >= ``iou`` với box đại diện của chỗ đó, ở >= ``min_fraction`` số ảnh.

    Trả (mã box cần bỏ, danh sách chỗ cố định {label, box, images}). Gom tham lam theo box đầu tiên gặp ở mỗi chỗ.
    """
    from collections import defaultdict

    by_label: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for image_id, objs in objects.items():
        for o in objs:
            by_label[o["label"]].append((image_id, o))
    need = min_fraction * max(1, len(objects))
    drop: set[str] = set()
    spots: list[dict[str, Any]] = []
    for label, items in sorted(by_label.items()):
        reps: list[np.ndarray] = []
        images: list[set[str]] = []
        members: list[list[str]] = []
        for image_id, o in items:
            box = np.asarray(o["box"], dtype=np.float64)
            if reps:
                stack = np.asarray(reps)
                ix = np.clip(np.minimum(box[2], stack[:, 2]) - np.maximum(box[0], stack[:, 0]), 0, None)
                iy = np.clip(np.minimum(box[3], stack[:, 3]) - np.maximum(box[1], stack[:, 1]), 0, None)
                inter = ix * iy
                union = (box[2] - box[0]) * (box[3] - box[1]) + (stack[:, 2] - stack[:, 0]) * (stack[:, 3] - stack[:, 1]) - inter
                overlap = inter / np.maximum(union, 1e-9)
                k = int(overlap.argmax())
                if overlap[k] >= iou:
                    images[k].add(image_id)
                    members[k].append(o["id"])
                    continue
            reps.append(box)
            images.append({image_id})
            members.append([o["id"]])
        for k, rep in enumerate(reps):
            if len(images[k]) >= need:
                drop.update(members[k])
                spots.append({"label": label, "box": [round(float(v), 1) for v in rep], "images": len(images[k])})
    return drop, spots


def run_objects(cfg: dict[str, Any], image_paths: Sequence[Path], data_dir: Path, force: bool = False) -> dict[str, Any]:
    """Pha dò đồ vật (``enrich objects``): {image_id: [{id, label, box, score}]} trong ``work/enrich/objects.json``.

    Dùng lại model RT-DETR của pha dò người, giữ các lớp COCO trừ người và đồ cố định trong ``enrich.objects.exclude``.
    Chạy lại chỉ dò ảnh chưa có.
    """
    ecfg, ocfg, log = cfg["enrich"], cfg["enrich"]["objects"], get_logger()
    out_path = data_dir / OBJECTS_FILE
    cached: dict[str, list[dict[str, Any]]] = {} if force or not out_path.exists() else read_json(out_path)
    todo = [p for p in image_paths if image_id_of(p) not in cached]
    if todo:
        detector = PersonDetector(ecfg["person"]["model_id"], float(ocfg["threshold"]), ecfg.get("device", "auto"))
        exclude = set(ocfg["exclude"])
        batch_size, started = int(ecfg["batch_size"]), time.perf_counter()
        with ThreadPoolExecutor(int(ecfg.get("workers", 4))) as pool:
            for start in range(0, len(todo), batch_size):
                chunk = todo[start:start + batch_size]
                images = list(pool.map(lambda path: Image.open(path).convert("RGB"), chunk))
                for path, (boxes, scores, names) in zip(chunk, detector.detect_all(images)):
                    keep = []
                    for label in sorted(set(names) - exclude):
                        idx = [k for k, n in enumerate(names) if n == label]
                        keep += [idx[j] for j in suppress_duplicates(boxes[idx], scores[idx], float(ocfg["nms_iou"]))]
                    keep = sorted(sorted(keep, key=lambda k: -scores[k])[: int(ocfg["max_per_image"])], key=lambda k: (boxes[k][0], boxes[k][1]))
                    image_id = image_id_of(path)
                    cached[image_id] = [{"id": f"{image_id}_o{n:02d}", "label": names[k], "box": [round(float(v), 1) for v in boxes[k]], "score": round(float(scores[k]), 3)}
                                        for n, k in enumerate(keep, start=1)]
                if (start // batch_size) % 40 == 39:
                    write_json(out_path, cached)
                    log.info("enrich: dò đồ vật %d/%d ảnh", start + len(chunk), len(todo))
        write_json(out_path, cached)
        log.info("enrich: dò đồ vật xong %d ảnh sau %.1f phút", len(todo), (time.perf_counter() - started) / 60)
    from collections import Counter

    counts = Counter(o["label"] for objs in cached.values() for o in objs)
    return {"images": len(cached), "objects": sum(counts.values()), "per_label": dict(counts.most_common())}


# --------------------------------------------------------------------------- build
def _thumbnail(image: Image.Image, long_side: int) -> Image.Image:
    scale = long_side / max(image.size)
    return image if scale >= 1 else image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)


def _save_stats(directory: Path, cfg: dict[str, Any], count: int, dim: int) -> None:
    clip = cfg["adl"]["clip"]
    write_json(directory / "build_stats.json", {"num_indexed": count, "dim": dim, "model": clip["model"], "pretrained": clip["pretrained"]})


def detect_people(
    cfg: dict[str, Any], image_paths: Sequence[Path], cache_path: Path, detector: Optional[Any], force: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Pha dò người: {image_id: [{box, score}]}, lưu vào ``cache_path`` để lần sau không dò lại."""
    ecfg, pcfg, log = cfg["enrich"], cfg["enrich"]["person"], get_logger()
    cached: dict[str, list[dict[str, Any]]] = {} if force or not cache_path.exists() else read_json(cache_path)
    todo = [p for p in image_paths if image_id_of(p) not in cached]
    if not todo:
        return cached
    owns_detector = detector is None
    if owns_detector:
        detector = PersonDetector(pcfg["model_id"], float(pcfg["threshold"]), ecfg.get("device", "auto"))
    batch_size, started = int(ecfg["batch_size"]), time.perf_counter()
    with ThreadPoolExecutor(int(ecfg.get("workers", 4))) as pool:
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            images = list(pool.map(lambda path: Image.open(path).convert("RGB"), chunk))
            for path, image, (boxes, scores) in zip(chunk, images, detector.detect(images)):
                w, h = image.size
                keep = [k for k in suppress_duplicates(boxes, scores, float(pcfg["nms_iou"])) if (boxes[k][2] - boxes[k][0]) * (boxes[k][3] - boxes[k][1]) >= float(pcfg["min_area_frac"]) * w * h]
                keep = sorted(keep[: int(pcfg["max_per_image"])], key=lambda k: (boxes[k][0], boxes[k][1]))   # trái -> phải
                cached[image_id_of(path)] = [{"box": [round(float(v), 1) for v in boxes[k]], "score": round(float(scores[k]), 3)} for k in keep]
            if (start // batch_size) % 40 == 39:
                write_json(cache_path, cached)
                log.info("enrich: dò người %d/%d ảnh", start + len(chunk), len(todo))
    write_json(cache_path, cached)
    log.info("enrich: dò người xong %d ảnh sau %.1f phút", len(todo), (time.perf_counter() - started) / 60)
    if owns_detector:
        # Trả GPU trước khi nạp model mã hoá: hai model cùng lúc không vừa 6 GB và chậm đi hàng chục lần.
        del detector
        from .utils import free_gpu_memory

        free_gpu_memory()
    return cached


def _roi_crop_ids(image_id: str, i: int, r: int, changed: np.ndarray, baseline_hour: dict[tuple[str, str], str], name: str) -> list[str]:
    """Các id crop của vùng thứ ``r`` trên ảnh thứ ``i``: ảnh nền của giờ (nếu ảnh này là nền) và/hoặc chính ảnh (nếu vùng đổi)."""
    hour = baseline_hour.get((image_id, name))
    return ([f"{BASELINE_PREFIX}{hour}"] if hour is not None else []) + ([image_id] if changed[i, r] else [])


def _rebuild_rois(
    cfg: dict[str, Any], image_paths: Sequence[Path], rois: Sequence[Roi], selected: set[str], shards_dir: Path, rois_dir: Path,
    changed: np.ndarray, baseline_hour: dict[tuple[str, str], str], encoder: Any,
) -> int:
    """Cắt và mã hoá lại crop của các vùng ``selected`` trong mọi phần đã lưu, giữ nguyên crop người và các vùng khác.

    Zone của từng người cũng được gán lại theo rois.yaml hiện tại (không cần dò người lại). Trả số crop vùng đã mã hoá.
    """
    ecfg, log = cfg["enrich"], get_logger()
    batch_size, shard_size, thumb = int(ecfg["batch_size"]), int(ecfg["shard_size"]), int(ecfg["crop_thumb_long_side"])
    min_overlap = float(ecfg["person"]["min_zone_overlap"])
    for name in selected:
        for old in (rois_dir / name / "crops").glob("*.jpg"):
            old.unlink()
    total = 0
    for shard_start in range(0, len(image_paths), shard_size):
        path = shards_dir / f"shard_{shard_start}.npz"
        with np.load(path) as data:
            people_emb, roi_emb = data["people_emb"], data["roi_emb"]
            people_meta = yaml.safe_load(str(data["people_meta"])) or []
            roi_meta = yaml.safe_load(str(data["roi_meta"])) or []
        keep = [k for k, (name, _) in enumerate(roi_meta) if name not in selected]
        new_meta: list[list[str]] = []
        crops: list[Image.Image] = []
        size: Optional[tuple[int, int]] = None
        for i in range(shard_start, min(shard_start + shard_size, len(image_paths))):
            image_id = image_id_of(image_paths[i])
            jobs = [(r, roi, _roi_crop_ids(image_id, i, r, changed, baseline_hour, roi.name)) for r, roi in enumerate(rois) if roi.name in selected]
            if not any(ids for _, _, ids in jobs) and size is not None:
                continue
            image = Image.open(image_paths[i]).convert("RGB")
            size = image.size
            for _, roi, ids in jobs:
                crop = image.crop(roi_pixels(roi.box, *image.size))
                for crop_id in ids:
                    _thumbnail(crop, thumb).save(rois_dir / roi.name / "crops" / f"{crop_id}.jpg", quality=88)
                    crops.append(crop)
                    new_meta.append([roi.name, crop_id])
        vectors = [v for k in range(0, len(crops), batch_size) for v in encoder.encode(crops[k:k + batch_size])]
        dim = roi_emb.shape[1] if roi_emb.size else (len(vectors[0]) if vectors else 1)
        roi_emb = np.concatenate([roi_emb[keep].reshape(len(keep), dim), np.asarray(vectors, np.float32).reshape(len(vectors), dim)])
        if size is not None:
            for meta in people_meta:
                meta["zone"], meta["zone_overlap"] = assign_zone(meta["box"], size[0], size[1], rois, min_overlap)
        np.savez(path, people_emb=people_emb, roi_emb=roi_emb, people_meta=np.asarray(yaml.safe_dump(people_meta)), roi_meta=np.asarray(yaml.safe_dump([roi_meta[k] for k in keep] + new_meta)))
        total += len(new_meta)
        log.info("enrich: dựng lại %s, phần %d: %d crop", sorted(selected), shard_start, len(new_meta))
    return total


def run_build(
    cfg: dict[str, Any], image_paths: Sequence[Path], out_dir: Path, force: bool = False,
    detector: Optional[Any] = None, encoder: Optional[Any] = None, only_rois: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Trích xuất tầng người và tầng vùng cho mọi ảnh. Chạy lại sẽ tiếp tục từ các phần đã lưu.

    Ba pha: (1) so từng vùng với nền, (2) dò người, (3) cắt crop và mã hoá. Pha 2 và 3 dùng hai
    model khác nhau và không bao giờ nằm trên GPU cùng lúc. ``detector`` / ``encoder`` chỉ để test.

    ``only_rois``: sau khi sửa khung của vài vùng trong rois.yaml, chỉ cắt và mã hoá lại các vùng đó
    trên các phần đã lưu (crop người và các vùng khác giữ nguyên). Cần một lần build đầy đủ trước đó.
    """
    ecfg, log = cfg["enrich"], get_logger()
    rois = load_rois(resolve_from_config(cfg, ecfg["rois_file"]))
    if not image_paths:
        raise RuntimeError("Không có ảnh nào để xử lý.")
    people_dir, rois_dir, shards_dir = out_dir / "people", out_dir / "rois", out_dir / "_shards"
    selected = set(only_rois or ())
    if selected - {r.name for r in rois}:
        raise ValueError(f"Không có vùng {sorted(selected - {r.name for r in rois})} trong rois.yaml")
    if force and shards_dir.exists() and not selected:
        for path in shards_dir.glob("*.npz"):
            path.unlink()
    for directory in (people_dir / "crops", shards_dir, *[rois_dir / r.name / "crops" for r in rois]):
        directory.mkdir(parents=True, exist_ok=True)
    # Các phần đã lưu chỉ dùng lại được khi danh sách ảnh và cách chia phần không đổi.
    signature = {"images": [image_id_of(p) for p in image_paths], "shard_size": int(ecfg["shard_size"]), "rois": [[r.name, list(r.box)] for r in rois]}
    signature_path = shards_dir / "signature.json"
    old = read_json(signature_path) if signature_path.exists() else None
    if selected:
        shard_count = len(range(0, len(image_paths), int(ecfg["shard_size"])))
        same_rest = old is not None and old["images"] == signature["images"] and old["shard_size"] == signature["shard_size"] and [r for r in old["rois"] if r[0] not in selected] == [r for r in signature["rois"] if r[0] not in selected]
        if not same_rest or len(list(shards_dir.glob("shard_*.npz"))) < shard_count:
            raise RuntimeError("--rois cần một lần build đầy đủ với cùng danh sách ảnh và các vùng còn lại không đổi. Chạy `enrich build` không có --rois.")
    elif old is not None and old != signature:
        log.warning("enrich: danh sách ảnh hoặc rois.yaml đã đổi, bỏ các phần đã lưu và làm lại.")
        for path in shards_dir.glob("*.npz"):
            path.unlink()

    started = time.perf_counter()
    rcfg, pcfg = ecfg["roi"], ecfg["person"]
    motion, baseline = roi_motion(image_paths, rois, tuple(rcfg["size"]), float(rcfg["threshold"]))
    changed = motion >= float(rcfg["min_changed"])
    # (image_id, tên vùng) -> giờ mà ảnh này làm nền cho vùng đó
    baseline_hour = {(image_id, name): hour for hour, per_roi in baseline.items() for name, image_id in per_roi.items()}
    log.info("enrich: so nền xong sau %.0fs; %d lượt vùng thay đổi trên %d ảnh x %d vùng", time.perf_counter() - started, int(changed.sum()), len(image_paths), len(rois))

    # Dò người (nếu cần) trước khi nạp encoder: hai model không bao giờ nằm trên GPU cùng lúc.
    detections = {} if selected else detect_people(cfg, image_paths, out_dir / "_detections.json", detector, force)
    if encoder is None:
        from .adl_embed import ClipImageEncoder

        encoder = ClipImageEncoder(cfg["adl"]["clip"]["model"], cfg["adl"]["clip"]["pretrained"], ecfg.get("device", "auto"))
    if selected:
        _rebuild_rois(cfg, image_paths, rois, selected, shards_dir, rois_dir, changed, baseline_hour, encoder)
    write_json(signature_path, signature)

    batch_size, shard_size, thumb = int(ecfg["batch_size"]), int(ecfg["shard_size"]), int(ecfg["crop_thumb_long_side"])
    done = {int(p.stem.split("_")[1]) for p in shards_dir.glob("shard_*.npz")}
    encode_started, encoded_images = time.perf_counter(), 0
    with ThreadPoolExecutor(int(ecfg.get("workers", 4))) as pool:
        for shard_start in range(0, len(image_paths), shard_size):
            if shard_start in done:
                continue
            indices = range(shard_start, min(shard_start + shard_size, len(image_paths)))
            people_meta: list[dict[str, Any]] = []
            roi_meta: list[list[str]] = []                # [tên vùng, id crop]
            people_vectors: list[np.ndarray] = []
            roi_vectors: list[np.ndarray] = []
            pending: list[tuple[str, Image.Image]] = []   # ("p" | "r", crop) chờ mã hoá

            def flush() -> None:
                if not pending:
                    return
                vectors = encoder.encode([crop for _, crop in pending])
                for (kind, _), vector in zip(pending, vectors):
                    (people_vectors if kind == "p" else roi_vectors).append(vector)
                pending.clear()

            for start in range(indices.start, indices.stop, batch_size):
                chunk = range(start, min(start + batch_size, indices.stop))
                images = list(pool.map(lambda i: Image.open(image_paths[i]).convert("RGB"), chunk))
                for i, image in zip(chunk, images):
                    image_id, (w, h) = image_id_of(image_paths[i]), image.size
                    for n, det in enumerate(detections.get(image_id, []), start=1):
                        person_id = f"{image_id}_p{n:02d}"
                        crop = image.crop(pad_pixels(det["box"], float(pcfg["pad_margin"]), w, h))
                        _thumbnail(crop, thumb).save(people_dir / "crops" / f"{person_id}.jpg", quality=88)
                        zone, overlap = assign_zone(det["box"], w, h, rois, float(pcfg["min_zone_overlap"]))
                        pending.append(("p", crop))
                        people_meta.append({"id": person_id, "image_id": image_id, "box": det["box"], "score": det["score"], "zone": zone, "zone_overlap": overlap})
                    for r, roi in enumerate(rois):
                        hour = baseline_hour.get((image_id, roi.name))
                        if not changed[i, r] and hour is None:
                            continue
                        crop = image.crop(roi_pixels(roi.box, w, h))
                        # Ảnh nền được lưu dưới id riêng: nó đại diện cho mọi ảnh không đổi trong giờ đó.
                        for crop_id in ([f"{BASELINE_PREFIX}{hour}"] if hour is not None else []) + ([image_id] if changed[i, r] else []):
                            _thumbnail(crop, thumb).save(rois_dir / roi.name / "crops" / f"{crop_id}.jpg", quality=88)
                            pending.append(("r", crop))
                            roi_meta.append([roi.name, crop_id])
                    while len(pending) >= batch_size:
                        head = pending[:batch_size]
                        rest = pending[batch_size:]
                        pending[:] = head
                        flush()
                        pending[:] = rest
            flush()
            dim = len(people_vectors[0]) if people_vectors else (len(roi_vectors[0]) if roi_vectors else 1)
            np.savez(
                shards_dir / f"shard_{shard_start}.npz",
                people_emb=np.stack(people_vectors) if people_vectors else np.zeros((0, dim), np.float32),
                roi_emb=np.stack(roi_vectors) if roi_vectors else np.zeros((0, dim), np.float32),
                people_meta=np.asarray(yaml.safe_dump(people_meta)), roi_meta=np.asarray(yaml.safe_dump(roi_meta)),
            )
            encoded_images += len(indices)
            rate = (time.perf_counter() - encode_started) / encoded_images
            remaining = sum(min(shard_size, len(image_paths) - s) for s in range(indices.stop, len(image_paths), shard_size) if s not in done)
            log.info("enrich: %d/%d ảnh, phần này %d crop người + %d crop vùng, còn khoảng %.0f phút", indices.stop, len(image_paths), len(people_vectors), len(roi_vectors), rate * remaining / 60)

    # ------------------------------------------------------------------ ghép các phần
    people_emb: list[np.ndarray] = []
    people_meta_all: list[dict[str, Any]] = []
    roi_emb: dict[str, list[np.ndarray]] = {r.name: [] for r in rois}
    roi_ids: dict[str, list[str]] = {r.name: [] for r in rois}
    for shard_start in range(0, len(image_paths), shard_size):
        with np.load(shards_dir / f"shard_{shard_start}.npz") as data:
            meta = yaml.safe_load(str(data["people_meta"])) or []
            if meta:
                people_emb.append(data["people_emb"])
                people_meta_all += meta
            for vector, (name, crop_id) in zip(data["roi_emb"], yaml.safe_load(str(data["roi_meta"])) or []):
                if name in roi_emb:
                    roi_emb[name].append(vector)
                    roi_ids[name].append(crop_id)
    dim = int(people_emb[0].shape[1]) if people_emb else 0
    np.save(people_dir / "embeddings.npy", np.concatenate(people_emb).astype(np.float32) if people_emb else np.zeros((0, 1), np.float32))
    write_json(people_dir / "ids.json", [f"{m['id']}.jpg" for m in people_meta_all])
    write_json(people_dir / "people.json", {m["id"]: m for m in people_meta_all})
    _save_stats(people_dir, cfg, len(people_meta_all), dim)
    for roi in rois:
        directory = rois_dir / roi.name
        vectors = roi_emb[roi.name]
        np.save(directory / "embeddings.npy", np.stack(vectors).astype(np.float32) if vectors else np.zeros((0, 1), np.float32))
        write_json(directory / "ids.json", [f"{crop_id}.jpg" for crop_id in roi_ids[roi.name]])
        write_roi_labels(roi, directory / "labels.yaml")
        _save_stats(directory, cfg, len(vectors), int(vectors[0].shape[0]) if vectors else 0)
    np.save(rois_dir / "motion.npy", motion)
    write_json(rois_dir / "roi_names.json", [r.name for r in rois])
    write_json(rois_dir / "image_ids.json", [image_id_of(p) for p in image_paths])
    write_json(rois_dir / "baseline.json", baseline)
    return {
        "images": len(image_paths),
        "people": len(people_meta_all),
        "people_with_zone": sum(1 for m in people_meta_all if m["zone"]),
        "roi_crops": {name: len(ids) for name, ids in roi_ids.items()},
        "elapsed_min": round((time.perf_counter() - started) / 60, 1),
        "out_dir": str(out_dir),
    }


# --------------------------------------------------------------------------- merge
def _labels_of(store_root: Path, classes_file: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]]:
    """(đã duyệt, gợi ý chưa duyệt, {tên lớp: tên tiếng Việt}, nhãn tự gõ) của một profile adl; rỗng nếu chưa chạy."""
    from . import adl

    store = adl.AdlStore(store_root)
    approved = store.approved()
    suggested = {s["image_id"]: s for s in store.suggestions()["items"] if s["image_id"] not in approved}
    names = {c.name: c.name_vi for c in adl.load_labels(classes_file)} if classes_file.exists() else {}
    custom_file = store_root.parent / CUSTOM_LABELS_FILE
    profile = store_root.name.removeprefix("adl_") if store_root.name != "adl" else "image"
    custom = read_json(custom_file).get(profile, {}) if custom_file.exists() else {}
    return approved, suggested, names, custom


def _label_entry(
    item_id: str, approved: dict[str, Any], suggested: dict[str, Any], names: dict[str, str], custom: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Nhãn của một crop: nhãn người tự gõ > người đã duyệt > hạt giống > gợi ý của máy. None nếu chưa có gì."""
    if custom and item_id in custom:
        text = custom[item_id]["text"]
        return {"label": text, "label_vi": text, "source": "human", "score": None, "custom": True}
    if item_id in approved:
        record = approved[item_id]
        return {"label": record["label"], "label_vi": names.get(record["label"], ""), "source": record.get("source", "seed"), "score": record.get("score")}
    if item_id in suggested:
        record = suggested[item_id]
        return {"label": record["label"], "label_vi": names.get(record["label"], ""), "source": "suggested", "score": record.get("score")}
    return None


def fix_action(action: Optional[dict[str, Any]], pid: str, pred: dict[str, Any], rejected: dict[str, list[str]], names: dict[str, str], gap: float) -> Optional[dict[str, Any]]:
    """Nhãn máy (không phải người duyệt) được sửa khi: người duyệt đã bác đúng nhãn đó (gợi ý cũ còn sót), hoặc
    `enrich actions` chắc hơn VLM một khoảng > gap. Nhãn cũ giữ ở ``machine_label``."""
    if not action or action.get("source") == "human":
        return action
    p, bad = pred.get(pid), rejected.get(pid, [])
    new = p["label"] if p and p["label"] not in bad and (action["label"] in bad or (p["label"] != action["label"] and p["gap"] > gap)) else None
    if new:
        return {"label": new, "label_vi": names.get(new, ""), "source": "fused", "score": None, "machine_label": action["label"]}
    return None if action["label"] in bad else action


def _manual_entry(m: dict[str, Any]) -> dict[str, Any]:
    """Box tự vẽ trên trang label_viewer/ ở dạng bản ghi gộp. Nhãn do người đặt nên ``source`` luôn là human."""
    label = m.get("label")
    return {
        "id": m["id"], "kind": m.get("kind", "object"), "box": m["box"],
        "label": {"label": label, "label_vi": m.get("label_vi", ""), "source": "human", "score": None} if label else None,
    }


def run_merge(cfg: dict[str, Any], data_dir: Path, jsons_dir: Path) -> dict[str, Any]:
    """Gộp tầng người + tầng vùng + nhãn hiện có thành một bản ghi mỗi ảnh trong ``enriched/``.

    Mỗi nhãn mang ``source``: ``human`` (đã duyệt), ``seed`` (hạt giống chưa duyệt) hay ``suggested``
    (máy gợi ý, chưa duyệt). Vùng không đổi so với nền thừa hưởng nhãn của ảnh nền giờ đó.
    """
    rois = load_rois(resolve_from_config(cfg, cfg["enrich"]["rois_file"]))
    people_file, rois_dir = data_dir / "people" / "people.json", data_dir / "rois"
    if not people_file.exists():
        raise RuntimeError(f"Chưa có {people_file}. Chạy `enrich build` trước.")
    people = read_json(people_file)
    image_ids, roi_names = read_json(rois_dir / "image_ids.json"), read_json(rois_dir / "roi_names.json")
    motion = np.load(rois_dir / "motion.npy")
    min_changed = float(cfg["enrich"]["roi"]["min_changed"])
    people_labels_file = resolve_from_config(cfg, cfg["adl_profiles"]["people"]["labels_file"])
    person_labels = _labels_of(jsons_dir / "adl_people", people_labels_file)
    roi_labels = {r.name: _labels_of(jsons_dir / f"adl_roi-{r.name}", rois_dir / r.name / "labels.yaml") for r in rois}
    name_vi = {r.name: r.name_vi for r in rois}
    manual_file = jsons_dir / MANUAL_BOXES_FILE
    manual_boxes: dict[str, list[dict[str, Any]]] = read_json(manual_file) if manual_file.exists() else {}
    objects: dict[str, list[dict[str, Any]]] = {}
    for name, source in ((OBJECTS_FILE, "rtdetr"), (OWL_OBJECTS_FILE, "owlv2")):
        if (data_dir / name).exists():
            for image_id, objs in read_json(data_dir / name).items():
                objects.setdefault(image_id, []).extend({**o, "source": source} for o in objs)
    static_cfg = cfg["enrich"]["objects"].get("static")
    static_dropped = 0
    if objects and static_cfg:
        drop, spots = static_object_ids(objects, float(static_cfg["iou"]), float(static_cfg["min_fraction"]))
        write_json(data_dir / "objects_static.json", spots)
        static_dropped = len(drop)
        objects = {image_id: [o for o in objs if o["id"] not in drop] for image_id, objs in objects.items()}
    from .person_attrs import ATTRS_FILE, load_attrs

    attrs = load_attrs(jsons_dir / "adl_people" / ATTRS_FILE)
    from . import identity as ident_mod

    identity = ident_mod.load_identity(jsons_dir)
    training = ident_mod.train_predict(cfg, data_dir, jsons_dir) if any(identity[1].values()) or any(identity[2].values()) else {"skipped": "chưa đặt tên"}
    pred = ident_mod.load_pred(jsons_dir) if "skipped" not in training else {}
    min_prob = float(cfg["enrich"]["identity"]["min_prob"])
    final = {pid: ident_mod.final_identity(pid, identity, pred, min_prob) for pid in people}
    outfit_update = ident_mod.update_outfits(jsons_dir, attrs, {pid: f["name"] for pid, f in final.items() if f and f.get("name")})
    outfits = ident_mod.load_outfits(jsons_dir)

    # box người duyệt đánh dấu "không phải người" (viewer: phím X) bị bỏ khỏi dataset
    names_file = jsons_dir / ident_mod.NAMES_FILE
    not_person = (read_json(names_file).get("not_person") or {}) if names_file.exists() else {}
    from .action_fuse import load_pred as load_action_pred

    action_pred = load_action_pred(jsons_dir)   # `enrich actions`: nhãn hành động gộp nhiều bằng chứng
    rejected_actions = read_json(jsons_dir / "adl_people" / "rejected.json") if (jsons_dir / "adl_people" / "rejected.json").exists() else {}
    override_gap = float(cfg["enrich"]["actions"]["override_gap"])
    by_image: dict[str, list[dict[str, Any]]] = {}
    for person in people.values():
        if person["id"] not in not_person:
            by_image.setdefault(person["image_id"], []).append(person)
    out_dir = jsons_dir / "enriched"
    counts = {"images": 0, "people": 0, "people_labeled": 0, "people_described": 0, "people_named": 0, "roi_states": 0, "manual_boxes": 0, "objects": 0}
    records: list[dict[str, Any]] = []
    for i, image_id in enumerate(image_ids):
        persons = []
        for person in sorted(by_image.get(image_id, []), key=lambda p: p["id"]):
            action = fix_action(_label_entry(person["id"], *person_labels), person["id"], action_pred, rejected_actions, person_labels[2], override_gap)
            outfit = ident_mod.outfit_for(outfits, (final[person["id"]] or {}).get("name"), group_key(image_id))
            attributes, attributes_source = ident_mod.apply_outfit(attrs.get(person["id"]), outfit)
            persons.append({**{k: person[k] for k in ("id", "box", "score", "zone")}, "zone_vi": name_vi.get(person["zone"] or "", ""), "action": action,
                            "attributes": attributes, "attributes_source": attributes_source, "identity": final[person["id"]], "outfit": outfit})
            counts["people"] += 1
            counts["people_labeled"] += action is not None
            counts["people_described"] += attributes is not None
            counts["people_named"] += bool((persons[-1]["identity"] or {}).get("name"))
        regions = {}
        for r, name in enumerate(roi_names):
            if name not in roi_labels:
                continue
            is_changed = bool(motion[i, r] >= min_changed)
            state = _label_entry(image_id if is_changed else f"{BASELINE_PREFIX}{group_key(image_id)}", *roi_labels[name])
            regions[name] = {"name_vi": name_vi.get(name, ""), "changed": is_changed, "motion": round(float(motion[i, r]), 4), "state": state}
            counts["roi_states"] += state is not None
        manual = [_manual_entry(m) for m in manual_boxes.get(image_id, [])]
        counts["manual_boxes"] += len(manual)
        image_objects = objects.get(image_id, [])
        counts["objects"] += len(image_objects)
        record = {"image_id": image_id, "hour": group_key(image_id), "num_people": len(persons), "people": persons, "rois": regions, "manual": manual,
                  "objects": image_objects}
        write_json(out_dir / f"{image_id}.json", record)
        records.append(record)
        counts["images"] += 1
    exports = Path(cfg["paths"].get("exports") or jsons_dir / "dataset")
    write_dataset(records, exports)
    # retrieval/ chỉ đọc exports/: chép embedding SigLIP2 (ảnh, ảnh nguyên khung, crop người) sang đó
    emb_dir = exports / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    adl_emb = resolve_from_config(cfg, cfg["adl"]["embeddings"])
    for src, dst in ((adl_emb, "images.npy"), (adl_emb.with_name("embeddings_global.npy"), "images_global.npy"),
                     (resolve_from_config(cfg, cfg["adl"]["ids"]), "image_ids.json"),
                     (data_dir / "people" / "embeddings.npy", "people.npy"), (data_dir / "people" / "ids.json", "people_ids.json")):
        if Path(src).exists():
            shutil.copyfile(src, emb_dir / dst)
    return {**counts, "identity_model": training, **outfit_update, "objects_static_dropped": static_dropped, "out_dir": str(out_dir), "dataset": str(exports)}


def write_dataset(records: Sequence[dict[str, Any]], out_dir: Path) -> None:
    """Bộ dataset gộp: ``castle_dataset.jsonl`` (mỗi dòng một ảnh) + ``people.csv``, ``rois.csv`` phẳng + ``summary.json``."""
    import csv
    import json
    from collections import Counter

    out_dir.mkdir(parents=True, exist_ok=True)
    from .person_attrs import FIELDS as ATTR_FIELDS

    outfit_fields = sorted({f"outfit_{k}" for r in records for p in r["people"] for k in (p.get("outfit") or {})})

    people_rows, roi_rows, object_rows = [], [], []
    with open(out_dir / "castle_dataset.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for p in r["people"]:
                a = p.get("action") or {}
                people_rows.append({"image_id": r["image_id"], "hour": r["hour"], "person_id": p["id"], "x1": p["box"][0], "y1": p["box"][1], "x2": p["box"][2], "y2": p["box"][3],
                                    "det_score": p["score"], "zone": p.get("zone") or "", "action": a.get("label", ""), "action_vi": a.get("label_vi", ""),
                                    "source": a.get("source", ""), "custom": bool(a.get("custom")),
                                    "identity_cluster": (p.get("identity") or {}).get("cluster") or "", "identity_name": (p.get("identity") or {}).get("name") or "",
                                    "identity_source": (p.get("identity") or {}).get("source") or "", "identity_prob": ((p.get("identity") or {}).get("predicted") or {}).get("prob", ""),
                                    **{f"outfit_{k}": v for k, v in (p.get("outfit") or {}).items()},
                                    **{k: (", ".join(v) if isinstance(v, list) else v) for k, v in (p.get("attributes") or {}).items()}})
            for o in r.get("objects", []):
                object_rows.append({"image_id": r["image_id"], "hour": r["hour"], "object_id": o["id"], "label": o["label"],
                                    "x1": o["box"][0], "y1": o["box"][1], "x2": o["box"][2], "y2": o["box"][3], "score": o["score"], "source": o.get("source", "")})
            for name, roi in r["rois"].items():
                s = roi.get("state") or {}
                roi_rows.append({"image_id": r["image_id"], "hour": r["hour"], "roi": name, "changed": roi["changed"], "motion": roi["motion"],
                                 "state": s.get("label", ""), "state_vi": s.get("label_vi", ""), "source": s.get("source", ""), "custom": bool(s.get("custom"))})
    for name, rows, fields in (("people.csv", people_rows, ["image_id", "hour", "person_id", "x1", "y1", "x2", "y2", "det_score", "zone", "action", "action_vi", "source", "custom", "identity_cluster", "identity_name", "identity_source", "identity_prob", *ATTR_FIELDS, *outfit_fields]),
                               ("rois.csv", roi_rows, ["image_id", "hour", "roi", "changed", "motion", "state", "state_vi", "source", "custom"]),
                               ("objects.csv", object_rows, ["image_id", "hour", "object_id", "label", "x1", "y1", "x2", "y2", "score", "source"])):
        with open(out_dir / name, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig: Excel đọc đúng tiếng Việt
            w = csv.DictWriter(f, fieldnames=fields, restval="")
            w.writeheader()
            w.writerows(rows)
    write_json(out_dir / "summary.json", {
        "images": len(records), "people": len(people_rows), "people_labeled": sum(1 for r in people_rows if r["action"]),
        "people_described": sum(1 for r in people_rows if r.get("gender")), "people_named": sum(1 for r in people_rows if r["identity_name"]),
        "identities": dict(Counter(r["identity_name"] for r in people_rows if r["identity_name"]).most_common()), "objects": len(object_rows),
        "object_labels": dict(Counter(r["label"] for r in object_rows).most_common()),
        "roi_slots": len(roi_rows), "roi_labeled": sum(1 for r in roi_rows if r["state"]),
        "people_by_source": dict(Counter(r["source"] or "none" for r in people_rows)),
        "rois_by_source": dict(Counter(r["source"] or "none" for r in roi_rows)),
        "actions": dict(Counter(r["action"] for r in people_rows if r["action"]).most_common()),
        "roi_states": {n: dict(Counter(r["state"] for r in roi_rows if r["roi"] == n and r["state"]).most_common()) for n in dict.fromkeys(r["roi"] for r in roi_rows)},
    })

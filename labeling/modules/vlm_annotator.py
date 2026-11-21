"""Stage annotate: sinh mô tả có cấu trúc cho từng ảnh bằng VLM, theo hai lượt.

Lượt thực thể: mỗi crop người (đã nới lề, phóng to) được hỏi riêng về trang phục và hành động.
    Người ở xa trong khung 1280x720 chỉ cao vài chục pixel; hỏi riêng trên crop phóng to là
    cách chính để bắt được sọc áo, màu phần trên/phần dưới.
Lượt toàn cảnh: ảnh gốc + bảng thực thể (vị trí, danh tính đã xác minh, thuộc tính vừa có)
    để model mô tả ngữ cảnh, quan hệ, nền và viết các câu truy vấn tăng dần.

Box và danh tính luôn lấy từ detector / Re-ID. VLM chỉ điền thuộc tính theo ``entity_id``,
nên không thể bịa toạ độ hay tên người.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from PIL import Image
from pydantic import BaseModel, ValidationError

from .schemas import (
    UNKNOWN_IDENTITY,
    AnnotationMeta,
    DenseTextualCorpus,
    Detection,
    DetectionRecord,
    EntityAnnotation,
    EntityAttributes,
    IdentityFile,
    ImageAnnotation,
    Relationship,
    SceneOutput,
)
from .utils import get_logger, spatial_position
from .vlm_backends import VLMBackend, VLMResult

T = TypeVar("T", bound=BaseModel)

SYSTEM_PROMPT = (
    "Bạn là người gán nhãn ảnh cẩn thận cho một bộ dữ liệu truy xuất ảnh. "
    "Chỉ mô tả những gì nhìn thấy rõ trong ảnh. Không đoán danh tính, tên tuổi của bất kỳ ai. "
    "Khi một chi tiết không nhìn rõ, ghi \"không rõ\" thay vì đoán. "
    "Chỉ trả về đúng một đối tượng JSON hợp lệ, không kèm lời giải thích, không dùng markdown."
)

ENTITY_PROMPT = """Ảnh này là phần cắt quanh MỘT người từ camera bếp. Nếu có người khác lọt vào khung, chỉ mô tả người chiếm phần lớn khung hình.

Trả về JSON đúng cấu trúc sau, mọi giá trị viết bằng tiếng Việt:
{
  "gender_presentation": "nam | nữ | không rõ",
  "age_group": "trẻ em | thanh niên | trung niên | lớn tuổi | không rõ",
  "hair": "độ dài, màu, kiểu tóc",
  "posture": "đứng | ngồi | cúi | đi | ...",
  "action": "hành động chính đang làm, ví dụ: mở cửa tủ lạnh, rửa chén",
  "held_objects": ["vật đang cầm hoặc đang dùng"],
  "attire": {
    "upper_garment_type": "loại áo: áo len, áo thun, áo sơ mi, áo khoác, áo hoodie...",
    "upper_garment_pattern": "trơn | sọc ngang | sọc dọc | caro | có chữ | có hình in | phối màu",
    "upper_color_top": "màu PHẦN TRÊN của áo (vai, ngực)",
    "upper_color_bottom": "màu PHẦN DƯỚI của áo (bụng, gấu áo); ghi giống phần trên nếu áo một màu",
    "lower_garment_type": "quần dài | quần short | váy | không thấy",
    "lower_garment_color": "màu quần/váy",
    "footwear": "giày dép nếu thấy",
    "accessories": ["kính, mũ, đồng hồ, khẩu trang, tạp dề, tai nghe..."]
  },
  "distinguishing_details": "chi tiết giúp phân biệt người này với người khác trong ảnh",
  "uncertain": ["tên các trường ở trên mà bạn không nhìn rõ"]
}

Lưu ý:
- Nhìn kỹ áo: có sọc không, sọc ngang hay dọc, nhiều hay ít sọc; phần trên và phần dưới áo có khác màu không.
- Trường nào không nhìn rõ thì ghi "không rõ" và thêm tên trường vào "uncertain"."""

SCENE_PROMPT = """Đây là ảnh toàn cảnh từ camera bếp (kích thước {width}x{height}).

Các thực thể đã được phát hiện và kiểm tra trước. Dùng đúng các mã này, không thêm mã mới:
{entity_table}

Trả về JSON đúng cấu trúc sau. Mọi giá trị viết bằng tiếng Việt, trừ "summary_en" viết bằng tiếng Anh:
{{
  "scene_context": {{
    "environment": "mô tả không gian",
    "room_label": "tên loại phòng, ví dụ: bếp",
    "lighting": "ánh sáng: tự nhiên/đèn, sáng/tối",
    "camera_perspective": "góc máy: từ trên cao, ngang tầm mắt, góc rộng..."
  }},
  "scene_graph_relationships": [
    {{"subject": "p01", "predicate": "mở", "object": "o03"}}
  ],
  "background_clutter": {{
    "secondary_actions": ["việc những người ở phía sau/phụ đang làm, ví dụ: hai người đang rửa chén ở bồn"],
    "background_objects": ["đồ vật đáng chú ý ở nền"]
  }},
  "global_summary": "2-4 câu mô tả toàn bộ ảnh: ai (theo ngoại hình/trang phục) đang làm gì, ở đâu",
  "summary_en": "the same summary in English",
  "progressive_queries": [
    "câu 1: hành động chính, chung nhất",
    "câu 2: thêm loại trang phục",
    "câu 3: thêm hoạ tiết (sọc, caro...)",
    "câu 4: thêm màu phần trên/phần dưới",
    "câu 5: thêm người khác trong ảnh và trang phục của họ",
    "câu 6: thêm hoạt động ở nền"
  ]
}}

Lưu ý:
- "subject" và "object" dùng mã thực thể khi có (p01, o02...), nếu không thì mô tả ngắn.
- Trong câu văn, gọi người bằng tên ở cột "danh tính" nếu đã có tên; nếu là Unknown thì gọi theo ngoại hình ("một người đàn ông mặc áo len sọc"). Không tự đặt tên cho ai.
- "progressive_queries": 4 đến 6 câu, câu sau lặp lại ý câu trước và thêm một chi tiết nhỏ hơn."""

RETRY_SUFFIX = (
    "\n\nCâu trả lời trước không dùng được vì: {error}\n"
    "Hãy trả lại ĐÚNG MỘT đối tượng JSON hợp lệ theo cấu trúc trên, không thêm chữ nào khác."
)


# --------------------------------------------------------------------------- parse JSON
def extract_json(text: str) -> Any:
    """Lấy đối tượng JSON từ câu trả lời của model.

    Xử lý các kiểu sai thường gặp: bọc trong ```json ... ```, có lời dẫn trước/sau,
    dấu phẩy thừa trước ngoặc đóng. Ném ``ValueError`` nếu không cứu được.
    """
    cleaned = re.sub(r"```(?:json|JSON)?", "", text).strip()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("không tìm thấy đối tượng JSON trong câu trả lời")

    # Tìm ngoặc đóng khớp với ngoặc mở đầu tiên, bỏ qua ngoặc nằm trong chuỗi.
    depth, in_string, escaped, end = 0, False, False, -1
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        raise ValueError("JSON bị cắt giữa chừng (thiếu ngoặc đóng)")

    candidate = cleaned[start:end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)  # bỏ dấu phẩy thừa
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON không hợp lệ: {exc.msg} (dòng {exc.lineno}, cột {exc.colno})") from exc


def parse_model(text: str, model: Type[T]) -> T:
    """Parse + validate câu trả lời theo schema Pydantic. Ném ``ValueError`` kèm lý do ngắn gọn."""
    data = extract_json(text)
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        location = ".".join(str(p) for p in first["loc"]) or "(gốc)"
        raise ValueError(f"sai schema tại {location}: {first['msg']}") from exc


# --------------------------------------------------------------------------- ảnh đầu vào
def resize_long_side(image: Image.Image, max_long_side: int) -> Image.Image:
    long_side = max(image.size)
    if long_side <= max_long_side:
        return image
    scale = max_long_side / long_side
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)


def prepare_crop(image: Image.Image, entity: Detection, min_short_side: int, max_long_side: int) -> Image.Image:
    """Cắt theo box đã nới lề rồi phóng to tới cạnh ngắn tối thiểu (không vượt cạnh dài tối đa)."""
    x1, y1, x2, y2 = (int(round(v)) for v in entity.padded_box.as_tuple())
    crop = image.crop((x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)))
    short, long = min(crop.size), max(crop.size)
    scale = max(1.0, min_short_side / short)
    scale = min(scale, max_long_side / long) if long * scale > max_long_side else scale
    if abs(scale - 1.0) > 1e-3:
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
    return crop


# --------------------------------------------------------------------------- dựng văn bản
def attributes_brief(attrs: Optional[EntityAttributes]) -> str:
    """Tóm tắt một dòng thuộc tính người, dùng trong bảng thực thể và trong bm25_blob."""
    if attrs is None:
        return ""
    a = attrs.attire
    upper = " ".join(p for p in [a.upper_garment_type, a.upper_garment_pattern] if p and p != "không rõ")
    colors = ""
    if a.upper_color_top and a.upper_color_bottom and a.upper_color_top != a.upper_color_bottom:
        colors = f"phần trên áo màu {a.upper_color_top}, phần dưới áo màu {a.upper_color_bottom}"
    elif a.upper_color_top:
        colors = f"áo màu {a.upper_color_top}"
    lower = " ".join(p for p in [a.lower_garment_type, a.lower_garment_color] if p and p != "không rõ")
    parts = [
        attrs.gender_presentation,
        attrs.action,
        upper,
        colors,
        lower,
        ", ".join(a.accessories),
        ("cầm " + ", ".join(attrs.held_objects)) if attrs.held_objects else "",
        attrs.distinguishing_details,
    ]
    return "; ".join(p.strip() for p in parts if p and p.strip() and p.strip() != "không rõ")


def build_entity_table(entities: list[EntityAnnotation]) -> str:
    """Bảng thực thể dạng văn bản đưa vào prompt toàn cảnh."""
    if not entities:
        return "(không phát hiện thực thể nào)"
    lines = []
    for e in entities:
        line = f"- {e.entity_id} | {e.label} | vị trí: {e.spatial_position}"
        if e.is_person:
            line += f" | danh tính: {e.identity}"
            brief = attributes_brief(e.attributes)
            if brief:
                line += f" | {brief}"
        lines.append(line)
    return "\n".join(lines)


def build_bm25_blob(annotation: ImageAnnotation) -> str:
    """Gộp mọi trường văn bản đã validate thành một khối để đánh chỉ mục BM25.

    Dựng xác định từ dữ liệu có cấu trúc (không nhờ model viết), nên luôn khớp với các trường.
    """
    corpus = annotation.dense_textual_corpus
    parts: list[str] = [corpus.global_summary, *corpus.progressive_queries, corpus.summary_en]
    ctx = annotation.scene_context
    parts += [ctx.environment, ctx.room_label, ctx.lighting, ctx.camera_perspective]
    for e in annotation.entities:
        if e.is_person:
            who = e.identity if e.identity != UNKNOWN_IDENTITY else "một người"
            parts.append(f"{who} {e.spatial_position}: {attributes_brief(e.attributes)}")
        else:
            parts.append(f"{e.label} {e.spatial_position}")
    names = {e.entity_id: (e.identity if e.is_person and e.identity != UNKNOWN_IDENTITY else e.label) for e in annotation.entities}
    for r in annotation.scene_graph_relationships:
        parts.append(f"{names.get(r.subject, r.subject)} {r.predicate} {names.get(r.object, r.object)}")
    parts += annotation.background_clutter.secondary_actions
    parts += annotation.background_clutter.background_objects
    parts.append(annotation.detector_caption)
    if annotation.activity is not None:
        parts += [annotation.activity.label_vi, annotation.activity.label]

    seen: set[str] = set()
    unique: list[str] = []
    for p in parts:
        p = " ".join(str(p).split())
        if p and p.lower() not in seen:
            seen.add(p.lower())
            unique.append(p)
    return " \n".join(unique)


def valid_relationships(relationships: list[Relationship], entity_ids: set[str]) -> list[Relationship]:
    """Bỏ quan hệ thiếu chủ thể/vị ngữ và quan hệ trỏ tới mã thực thể không tồn tại.

    Qwen 7B từng bịa p01..p06 cho một ảnh không có người; mã dạng p01/o02/m03 mà không có trong
    danh sách thực thể thì chắc chắn là bịa. Chủ thể/đối tượng viết bằng lời thì được giữ.
    """
    def ok(ref: str) -> bool:
        return not re.fullmatch(r"[pom]\d{1,3}", ref) or ref in entity_ids

    return [r for r in relationships if r.subject and r.predicate and ok(r.subject) and ok(r.object)]


# --------------------------------------------------------------------------- annotator
class VLMStructuredAnnotator:
    """Điều phối hai lượt hỏi VLM cho một ảnh và gộp với kết quả detector / Re-ID."""

    def __init__(self, backend: VLMBackend, cfg: dict[str, Any]) -> None:
        self.backend = backend
        self.cfg = cfg
        self.vcfg = cfg["vlm"]
        self.log = get_logger()

    def _ask(self, images: list[Image.Image], prompt: str, model: Type[T], max_new_tokens: int, meta: AnnotationMeta) -> T:
        """Hỏi model và validate; JSON hỏng thì hỏi lại kèm lý do, tối đa ``vlm.retries`` lần."""
        attempts = 1 + int(self.vcfg.get("retries", 1))
        current_prompt, last_error = prompt, "không rõ"
        for _ in range(attempts):
            result: VLMResult = self.backend.generate(images, current_prompt, SYSTEM_PROMPT, max_new_tokens)
            meta.vlm_calls += 1
            meta.input_tokens += result.input_tokens
            meta.output_tokens += result.output_tokens
            meta.seconds += result.seconds
            try:
                return parse_model(result.text, model)
            except ValueError as exc:
                last_error = str(exc)
                current_prompt = prompt + RETRY_SUFFIX.format(error=last_error)
        raise ValueError(f"VLM không trả JSON hợp lệ sau {attempts} lần: {last_error}")

    def annotate_image(self, image_path: Path, record: DetectionRecord, identities: Optional[IdentityFile]) -> ImageAnnotation:
        """Sinh bản ghi đầy đủ cho một ảnh.

        Lỗi ở lượt thực thể của một người chỉ làm người đó thiếu thuộc tính (ghi vào
        ``meta.failed_entities``). Lỗi ở lượt toàn cảnh thì ném ra để ảnh không bị đánh dấu xong.
        """
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        meta = AnnotationMeta(backend=self.backend.name, model=self.backend.model_id)
        id_by_entity = identities.by_entity() if identities else {}

        entities: list[EntityAnnotation] = []
        for det in record.entities:
            match = id_by_entity.get(det.entity_id)
            entities.append(
                EntityAnnotation(
                    entity_id=det.entity_id,
                    label=det.label,
                    is_person=det.is_person,
                    box=det.box,
                    spatial_position=spatial_position(det.box.as_tuple(), record.width, record.height),
                    identity=match.identity if match else UNKNOWN_IDENTITY,
                    identity_similarity=match.similarity if match else None,
                )
            )

        # Lượt thực thể: ưu tiên người có box lớn nhất khi vượt giới hạn mỗi ảnh.
        det_by_id = {d.entity_id: d for d in record.entities}
        persons = sorted((e for e in entities if e.is_person), key=lambda e: -e.box.area)
        for entity in persons[: int(self.vcfg["max_persons_per_image"])]:
            crop = prepare_crop(
                image,
                det_by_id[entity.entity_id],
                int(self.vcfg["crop_min_short_side"]),
                int(self.vcfg["crop_max_long_side"]),
            )
            try:
                entity.attributes = self._ask(
                    [crop], ENTITY_PROMPT, EntityAttributes, int(self.vcfg["max_new_tokens_entity"]), meta
                )
            except ValueError as exc:
                entity.attributes_error = str(exc)
                meta.failed_entities.append(entity.entity_id)
                self.log.warning("%s/%s: %s", record.image_id, entity.entity_id, exc)

        # Lượt toàn cảnh.
        scene_prompt = SCENE_PROMPT.format(
            width=record.width, height=record.height, entity_table=build_entity_table(entities)
        )
        scene = self._ask(
            [resize_long_side(image, int(self.vcfg["global_max_long_side"]))],
            scene_prompt,
            SceneOutput,
            int(self.vcfg["max_new_tokens_scene"]),
            meta,
        )

        annotation = ImageAnnotation(
            image_id=record.image_id,
            file_name=record.file_name,
            width=record.width,
            height=record.height,
            scene_context=scene.scene_context,
            entities=entities,
            scene_graph_relationships=valid_relationships(scene.scene_graph_relationships, {e.entity_id for e in entities}),
            background_clutter=scene.background_clutter,
            dense_textual_corpus=DenseTextualCorpus(
                global_summary=scene.global_summary.strip(),
                summary_en=scene.summary_en.strip(),
                progressive_queries=[q.strip() for q in scene.progressive_queries if q.strip()],
            ),
            detector_caption=record.global_caption,
            meta=meta,
        )
        annotation.dense_textual_corpus.bm25_blob = build_bm25_blob(annotation)
        meta.seconds = round(meta.seconds, 2)
        return annotation

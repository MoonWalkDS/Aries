"""Tiện ích dùng chung cho các bước VLM: đọc JSON lỏng từ câu trả lời, thu nhỏ ảnh, dựng chuỗi BM25 cho một ảnh."""
from __future__ import annotations

import json
import re
from typing import Any, Optional, Type, TypeVar

from PIL import Image
from pydantic import BaseModel, ValidationError

from .schemas import UNKNOWN_IDENTITY, EntityAttributes, ImageAnnotation
from .utils import spatial_position

T = TypeVar("T", bound=BaseModel)

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


# --------------------------------------------------------------------------- annotator

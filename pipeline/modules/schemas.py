"""Schema Pydantic cho mọi dữ liệu trung gian và kết quả cuối của pipeline.

Hai nhóm:
- Dữ liệu do pipeline tự tính (box, crop, khuôn mặt, danh tính): schema chặt.
- Dữ liệu do VLM sinh (thuộc tính người, mô tả cảnh): schema "khoan dung" - thiếu trường thì
  nhận giá trị rỗng, ``null`` được đổi thành rỗng, chuỗi đơn được bọc thành danh sách. Mục đích
  là không vứt cả một ảnh chỉ vì model bỏ sót một trường phụ.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

UNKNOWN_IDENTITY = "Unknown"


# --------------------------------------------------------------------------- hình học
class BBox(BaseModel):
    """Hộp bao theo pixel của ảnh gốc, góc trên-trái (x1, y1), góc dưới-phải (x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _ordered(self) -> "BBox":
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"BBox không hợp lệ: {self.as_tuple()}")
        return self

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_seq(cls, seq: Any) -> "BBox":
        x1, y1, x2, y2 = (float(v) for v in seq)
        return cls(x1=min(x1, x2), y1=min(y1, y2), x2=max(x1, x2), y2=max(y1, y2))


# --------------------------------------------------------------------------- stage detect
class Detection(BaseModel):
    """Một thực thể do detector tìm thấy."""

    entity_id: str                       # p01, p02... cho người; o01... cho đồ vật
    label: str
    is_person: bool = False
    score: Optional[float] = None        # Florence-2 không trả điểm tin cậy
    box: BBox                            # box gốc
    padded_box: BBox                     # box đã nới pad_margin, kẹp trong biên ảnh
    crop_path: Optional[str] = None      # tương đối so với thư mục output_crops
    layer: int = 2                       # 0 = toàn ảnh, 2 = crop thực thể
    source: str = ""                     # tác vụ Florence sinh ra box này


class DetectionRecord(BaseModel):
    """Kết quả stage detect cho một ảnh (Layer 0 = chính ảnh gốc, mô tả bằng các trường dưới)."""

    image_id: str
    file_name: str
    width: int
    height: int
    global_caption: str = ""
    entities: list[Detection] = Field(default_factory=list)

    def persons(self) -> list[Detection]:
        return [e for e in self.entities if e.is_person]


# --------------------------------------------------------------------------- stage faces / reid
class FaceRecord(BaseModel):
    face_id: str
    box: BBox
    det_score: float
    entity_id: Optional[str] = None      # người chứa khuôn mặt này; None nếu không gán được
    embedding: list[float]               # 512 chiều, đã chuẩn hoá L2


class FaceFile(BaseModel):
    image_id: str
    faces: list[FaceRecord] = Field(default_factory=list)


class IdentityMatch(BaseModel):
    entity_id: Optional[str] = None
    face_id: str
    identity: str = UNKNOWN_IDENTITY
    similarity: Optional[float] = None   # cosine cao nhất với gallery (kể cả khi dưới ngưỡng)
    best_candidate: Optional[str] = None # tên có cosine cao nhất, để tra soát khi dưới ngưỡng


class IdentityFile(BaseModel):
    image_id: str
    tau_face: float
    matches: list[IdentityMatch] = Field(default_factory=list)

    def by_entity(self) -> dict[str, IdentityMatch]:
        return {m.entity_id: m for m in self.matches if m.entity_id}


# --------------------------------------------------------------------------- đầu ra VLM
class LenientModel(BaseModel):
    """Lớp nền cho dữ liệu do VLM sinh: bỏ qua khoá lạ, đổi null thành giá trị rỗng."""

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for name, field in cls.model_fields.items():
            if name not in out:
                continue
            value = out[name]
            wants_list = str(field.annotation).startswith("list[")
            if value is None:
                out[name] = [] if wants_list else ("" if field.annotation is str else value)
            elif wants_list and isinstance(value, (str, int, float)):
                out[name] = [value] if str(value).strip() else []
            elif field.annotation is str and not isinstance(value, str):
                # model đôi khi trả danh sách/số cho trường chuỗi
                out[name] = ", ".join(map(str, value)) if isinstance(value, list) else str(value)
        return out


class Attire(LenientModel):
    """Trang phục, tách riêng các chi tiết mà đề thi hay hỏi."""

    upper_garment_type: str = ""         # áo len, áo thun, áo khoác...
    upper_garment_pattern: str = ""      # trơn, sọc ngang, sọc dọc, caro, có chữ/hình...
    upper_color_top: str = ""            # màu phần TRÊN của áo (vai, ngực)
    upper_color_bottom: str = ""         # màu phần DƯỚI của áo (bụng, gấu); bằng phần trên nếu một màu
    lower_garment_type: str = ""         # quần dài, quần short, váy...
    lower_garment_color: str = ""
    footwear: str = ""
    accessories: list[str] = Field(default_factory=list)  # kính, mũ, đồng hồ, khẩu trang, tạp dề...


class EntityAttributes(LenientModel):
    """Kết quả lượt hỏi riêng trên crop của một người."""

    gender_presentation: str = ""        # nam | nữ | không rõ
    age_group: str = ""                  # trẻ em | thanh niên | trung niên | lớn tuổi | không rõ
    hair: str = ""
    posture: str = ""                    # đứng, ngồi, cúi...
    action: str = ""                     # hành động chính, ví dụ "mở cửa tủ lạnh"
    held_objects: list[str] = Field(default_factory=list)
    attire: Attire = Field(default_factory=Attire)
    distinguishing_details: str = ""
    uncertain: list[str] = Field(default_factory=list)    # tên các trường model không nhìn rõ


class SceneContext(LenientModel):
    environment: str = ""
    room_label: str = ""
    lighting: str = ""
    camera_perspective: str = ""


class Relationship(LenientModel):
    subject: str = ""
    predicate: str = ""
    object: str = ""

    @field_validator("subject", "predicate", "object")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class BackgroundClutter(LenientModel):
    secondary_actions: list[str] = Field(default_factory=list)
    background_objects: list[str] = Field(default_factory=list)


class SceneOutput(LenientModel):
    """Kết quả lượt hỏi toàn cảnh."""

    scene_context: SceneContext = Field(default_factory=SceneContext)
    scene_graph_relationships: list[Relationship] = Field(default_factory=list)
    background_clutter: BackgroundClutter = Field(default_factory=BackgroundClutter)
    global_summary: str = ""
    summary_en: str = ""
    progressive_queries: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_content(self) -> "SceneOutput":
        if not self.global_summary.strip():
            raise ValueError("thiếu global_summary")
        return self


# --------------------------------------------------------------------------- kết quả cuối
class EntityAnnotation(BaseModel):
    entity_id: str
    label: str
    is_person: bool = False
    box: BBox
    spatial_position: str = ""           # tính từ box, ví dụ "bên trái, phía dưới"
    identity: str = UNKNOWN_IDENTITY     # chỉ lấy từ Re-ID hoặc người kiểm tra, không lấy từ VLM
    identity_similarity: Optional[float] = None
    attributes: Optional[EntityAttributes] = None
    attributes_error: Optional[str] = None


class DenseTextualCorpus(BaseModel):
    global_summary: str = ""
    summary_en: str = ""
    progressive_queries: list[str] = Field(default_factory=list)
    bm25_blob: str = ""


class AnnotationMeta(BaseModel):
    backend: str = ""
    model: str = ""
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    vlm_calls: int = 0
    failed_entities: list[str] = Field(default_factory=list)


class ActivityLabel(BaseModel):
    """Nhãn hoạt động (ADL) cấp ảnh do stage ``adl`` gán."""

    label: str
    label_vi: str = ""
    group: str = ""
    source: str = ""                     # human: người đã duyệt | seed: ảnh hạt giống chưa duyệt
    score: Optional[float] = None        # cosine lúc được gợi ý
    round: int = 0


class ImageAnnotation(BaseModel):
    """Bản ghi chính cho một ảnh: output_jsons/per_image/<image_id>.json."""

    image_id: str
    file_name: str
    width: int
    height: int
    scene_context: SceneContext = Field(default_factory=SceneContext)
    entities: list[EntityAnnotation] = Field(default_factory=list)
    scene_graph_relationships: list[Relationship] = Field(default_factory=list)
    background_clutter: BackgroundClutter = Field(default_factory=BackgroundClutter)
    dense_textual_corpus: DenseTextualCorpus = Field(default_factory=DenseTextualCorpus)
    detector_caption: str = ""
    activity: Optional[ActivityLabel] = None  # chỉ do stage adl ghi, không lấy từ VLM
    verified: bool = False              # true sau khi người kiểm tra sửa qua Label Studio
    meta: AnnotationMeta = Field(default_factory=AnnotationMeta)

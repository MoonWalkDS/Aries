"""Stage render: vẽ toàn bộ nhãn của một ảnh ra một file ảnh để xem bằng mắt.

Bên trái là ảnh gốc với box (người: viền dày, mỗi người một màu; đồ vật: viền mảnh).
Bên phải là bảng chữ: tóm tắt, câu truy vấn, thuộc tính từng người (cùng màu với box),
hoạt động nền, quan hệ, danh sách đồ vật.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from .schemas import UNKNOWN_IDENTITY, EntityAnnotation, ImageAnnotation

PERSON_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#d4a017", "#f032e6", "#9a6324", "#000075"]
OBJECT_COLOR = "#00b7c7"
PANEL_WIDTH = 760
PAD = 16
# Font có dấu tiếng Việt; thử lần lượt Windows, Linux (Kaggle/Colab), macOS.
FONT_CANDIDATES = [
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]


def _fonts(size: int) -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    for regular, bold in FONT_CANDIDATES:
        if Path(regular).exists():
            bold_path = bold if Path(bold).exists() else regular
            return ImageFont.truetype(regular, size), ImageFont.truetype(bold_path, size)
    default = ImageFont.load_default()
    return default, default


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Ngắt dòng theo độ rộng pixel."""
    lines: list[str] = []
    for paragraph in str(text).split("\n"):
        current = ""
        for word in paragraph.split():
            trial = f"{current} {word}".strip()
            if draw.textlength(trial, font=font) <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _person_lines(e: EntityAnnotation) -> list[str]:
    """Mọi trường thuộc tính của một người, mỗi trường một dòng; trường rỗng bị bỏ."""
    if e.attributes is None:
        return [f"(không có thuộc tính: {e.attributes_error or 'vượt giới hạn số người mỗi ảnh'})"]
    a, t = e.attributes, e.attributes.attire
    rows = [
        ("giới / tuổi", " / ".join(v for v in (a.gender_presentation, a.age_group) if v)),
        ("tóc", a.hair),
        ("tư thế", a.posture),
        ("hành động", a.action),
        ("đang cầm", ", ".join(a.held_objects)),
        ("áo", " / ".join(v for v in (t.upper_garment_type, t.upper_garment_pattern) if v)),
        ("màu áo", f"trên: {t.upper_color_top} | dưới: {t.upper_color_bottom}" if (t.upper_color_top or t.upper_color_bottom) else ""),
        ("quần/váy", " ".join(v for v in (t.lower_garment_type, t.lower_garment_color) if v)),
        ("giày dép", t.footwear),
        ("phụ kiện", ", ".join(t.accessories)),
        ("đặc điểm", a.distinguishing_details),
        ("model không chắc", ", ".join(a.uncertain)),
    ]
    return [f"{name}: {value}" for name, value in rows if value and value.strip()]


Row = tuple[str, ImageFont.ImageFont, str, int]  # (chữ, font, màu, thụt lề)


class _Block:
    """Một khối chữ có độ rộng cố định; gom dòng trước, vẽ sau, để biết chiều cao khi dàn trang."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.rows: list[Row] = []
        self._scratch = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def add(self, text: str, font: ImageFont.ImageFont, color: str = "#111111", indent: int = 0) -> None:
        for line in _wrap(self._scratch, text, font, self.width - indent):
            self.rows.append((line, font, color, indent))

    def gap(self, font: ImageFont.ImageFont) -> None:
        self.rows.append(("", font, "#111111", 0))

    @staticmethod
    def _line_height(font: ImageFont.ImageFont) -> int:
        return int(getattr(font, "size", 14) * 1.38)

    @property
    def height(self) -> int:
        return sum(self._line_height(f) for _, f, _, _ in self.rows)

    def draw(self, draw: ImageDraw.ImageDraw, x: int, y: int) -> None:
        for line, font, color, indent in self.rows:
            draw.text((x + indent, y), line, font=font, fill=color)
            y += self._line_height(font)


def render_annotation(annotation: ImageAnnotation, image_path: Path, out_path: Path, show_objects: bool = True) -> Path:
    """Vẽ một ảnh kèm toàn bộ nhãn và lưu ra ``out_path``.

    Bố cục: ảnh + box ở trên bên trái, thông tin toàn cảnh ở bên phải ảnh, thuộc tính từng người
    xếp thành các cột bên dưới.
    """
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    font, bold = _fonts(17)
    small, small_bold = _fonts(14)
    title_font = _fonts(22)[1]

    persons = [e for e in annotation.entities if e.is_person]
    objects = [e for e in annotation.entities if not e.is_person]
    color_of = {e.entity_id: PERSON_COLORS[i % len(PERSON_COLORS)] for i, e in enumerate(persons)}
    corpus, ctx = annotation.dense_textual_corpus, annotation.scene_context

    # ----- khối bên phải ảnh: thông tin toàn cảnh -----
    side = _Block(PANEL_WIDTH - 2 * PAD)
    side.add(f"{annotation.image_id}   ({annotation.meta.model or 'chưa rõ model'})", title_font)
    status = "ĐÃ KIỂM TRA" if annotation.verified else "chưa kiểm tra"
    side.add(f"{len(persons)} người, {len(objects)} đồ vật   |   {status}", small, "#555555")
    side.gap(small)
    side.add("TÓM TẮT", bold)
    side.add(corpus.global_summary or "(trống)", font)
    if corpus.summary_en:
        side.add(corpus.summary_en, small, "#555555")
    side.gap(small)
    side.add("CÂU TRUY VẤN TĂNG DẦN", bold)
    for i, query in enumerate(corpus.progressive_queries, start=1):
        side.add(f"{i}. {query}", font)
    side.gap(small)
    side.add("BỐI CẢNH", bold)
    for name, value in (("phòng", ctx.room_label), ("không gian", ctx.environment), ("ánh sáng", ctx.lighting), ("góc máy", ctx.camera_perspective)):
        if value:
            side.add(f"{name}: {value}", small)
    side.gap(small)
    side.add("HOẠT ĐỘNG NỀN", bold)
    for action in annotation.background_clutter.secondary_actions or ["(trống)"]:
        side.add(f"- {action}", small)
    if annotation.background_clutter.background_objects:
        side.add("đồ vật nền: " + ", ".join(annotation.background_clutter.background_objects), small)
    side.gap(small)
    side.add("QUAN HỆ", bold)
    for r in annotation.scene_graph_relationships:
        side.add(f"- {r.subject}  —{r.predicate}→  {r.object}", small)
    if not annotation.scene_graph_relationships:
        side.add("(trống)", small, "#555555")
    side.gap(small)
    side.add("ĐỒ VẬT (box viền mảnh)", bold)
    side.add(", ".join(f"{e.entity_id} {e.label}" for e in objects) or "(không có)", small, "#555555")

    # ----- các khối người: xếp cột bên dưới ảnh -----
    total_width = image.width + PANEL_WIDTH
    columns = 4
    column_width = (total_width - PAD * (columns + 1)) // columns
    person_blocks: list[_Block] = []
    for e in persons:
        block = _Block(column_width)
        identity = e.identity if e.identity != UNKNOWN_IDENTITY else "Unknown"
        sim = f" ({e.identity_similarity:.2f})" if e.identity_similarity is not None else ""
        block.add(f"■ {e.entity_id}  [{identity}{sim}]", bold, color_of[e.entity_id])
        block.add(e.spatial_position, small, "#555555")
        for line in _person_lines(e):
            block.add(line, small, "#111111")
        person_blocks.append(block)
    grid_rows = [person_blocks[i:i + columns] for i in range(0, len(person_blocks), columns)]
    grid_height = sum(max(b.height for b in row) + PAD for row in grid_rows)

    top_height = max(image.height, side.height + 2 * PAD)
    header_height = (int(bold.size * 1.38) + PAD) if person_blocks else 0
    canvas = Image.new("RGB", (total_width, top_height + header_height + grid_height + PAD), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # ----- box trên ảnh: đồ vật trước, người sau để box người nằm trên -----
    def tag(x: float, y: float, text: str, f: ImageFont.ImageFont, fill: str) -> None:
        width, height = draw.textlength(text, font=f) + 8, f.size + 6
        y0 = y - height if y - height >= 0 else y  # sát mép trên thì đặt nhãn vào trong box
        x0 = min(max(0, x), image.width - width)
        draw.rectangle((x0, y0, x0 + width, y0 + height), fill=fill)
        draw.text((x0 + 4, y0 + 2), text, font=f, fill="white")

    if show_objects:
        for e in objects:
            draw.rectangle(e.box.as_tuple(), outline=OBJECT_COLOR, width=1)
            tag(e.box.x1, e.box.y1, f"{e.entity_id} {e.label}", small, OBJECT_COLOR)
    for e in persons:
        color = color_of[e.entity_id]
        draw.rectangle(e.box.as_tuple(), outline=color, width=4)
        name = e.entity_id if e.identity == UNKNOWN_IDENTITY else f"{e.entity_id} {e.identity}"
        tag(e.box.x1, e.box.y1, name, small_bold, color)

    # ----- chữ -----
    side.draw(draw, image.width + PAD, PAD)
    y = top_height
    if person_blocks:
        draw.line((PAD, y, total_width - PAD, y), fill="#cccccc", width=1)
        draw.text((PAD, y + PAD // 2), "NGƯỜI (màu tiêu đề trùng màu box)", font=bold, fill="#111111")
        y += header_height
    for row in grid_rows:
        for i, block in enumerate(row):
            block.draw(draw, PAD + i * (column_width + PAD), y)
        y += max(b.height for b in row) + PAD

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return out_path


def render_all(per_image_dir: Path, images_dir: Path, out_dir: Path, only_ids: Optional[set[str]] = None, show_objects: bool = True) -> list[Path]:
    """Render mọi annotation trong ``per_image_dir`` (hoặc chỉ các ID được chọn)."""
    from .utils import read_json

    written: list[Path] = []
    for path in sorted(per_image_dir.glob("*.json")):
        if only_ids is not None and path.stem not in only_ids:
            continue
        annotation = ImageAnnotation.model_validate(read_json(path))
        image_path = images_dir / annotation.file_name
        if not image_path.exists():
            continue
        written.append(render_annotation(annotation, image_path, out_dir / f"{annotation.image_id}.jpg", show_objects))
    return written

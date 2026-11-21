"""Cầu nối Label Studio: xuất task kèm pre-label để người kiểm tra, và nhập lại bản đã sửa.

Xuất (``ls-export``):
- ``label_studio_config.xml``: giao diện gán nhãn (box, danh tính theo từng box, thuộc tính, mô tả).
- ``label_studio_tasks.json``: mỗi ảnh một task ở định dạng import của Label Studio, phần
  ``predictions`` chứa box (đơn vị %) và giá trị do pipeline sinh.

Nhập (``ls-import``): đọc file export JSON của Label Studio, ghi các chỉnh sửa (box, nhãn,
danh tính, thuộc tính, mô tả, câu truy vấn) vào ``output_jsons/per_image/`` và đặt ``verified``.

Mỗi vùng (region) dùng ``id`` = ``entity_id`` để khớp lại khi nhập.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional
from xml.sax.saxutils import quoteattr

from pydantic import ValidationError

from .schemas import UNKNOWN_IDENTITY, BBox, EntityAnnotation, EntityAttributes, ImageAnnotation
from .utils import get_logger, read_json, spatial_position, write_json
from .vlm_annotator import build_bm25_blob

MODEL_VERSION = "castle5000-pipeline"

# Tên các control trong labeling config; dùng chung cho xuất và nhập.
BOX, IDENTITY, ATTRIBUTES, SUMMARY, QUERIES, IMAGE = "box", "identity", "attributes", "summary", "queries", "image"


# --------------------------------------------------------------------------- toạ độ
def box_to_percent(box: BBox, width: int, height: int) -> dict[str, float]:
    """Pixel -> đơn vị % của Label Studio (x, y là góc trên-trái)."""
    return {
        "x": round(box.x1 / width * 100, 4),
        "y": round(box.y1 / height * 100, 4),
        "width": round(box.width / width * 100, 4),
        "height": round(box.height / height * 100, 4),
        "rotation": 0,
    }


def percent_to_box(value: dict[str, Any], width: int, height: int) -> BBox:
    x1 = float(value["x"]) / 100 * width
    y1 = float(value["y"]) / 100 * height
    return BBox(
        x1=round(x1, 2),
        y1=round(y1, 2),
        x2=round(x1 + float(value["width"]) / 100 * width, 2),
        y2=round(y1 + float(value["height"]) / 100 * height, 2),
    )


# --------------------------------------------------------------------------- labeling config
def build_label_config(labels: Iterable[str], identities: Iterable[str]) -> str:
    """XML giao diện gán nhãn. ``labels``: nhãn box; ``identities``: tên cho ô chọn danh tính."""
    label_list = sorted({l for l in labels if l} | {"person"})
    identity_list = [UNKNOWN_IDENTITY] + sorted({i for i in identities if i and i != UNKNOWN_IDENTITY})
    lines = [
        "<View>",
        f'  <Image name="{IMAGE}" value="$image" zoom="true" zoomControl="true"/>',
        f'  <RectangleLabels name="{BOX}" toName="{IMAGE}">',
        *[f"    <Label value={quoteattr(l)}/>" for l in label_list],
        "  </RectangleLabels>",
        '  <Header value="Danh tính của box đang chọn"/>',
        f'  <Choices name="{IDENTITY}" toName="{IMAGE}" perRegion="true" choice="single" showInline="true">',
        *[f"    <Choice value={quoteattr(i)}/>" for i in identity_list],
        "  </Choices>",
        '  <Header value="Thuộc tính của box đang chọn (JSON)"/>',
        f'  <TextArea name="{ATTRIBUTES}" toName="{IMAGE}" perRegion="true" editable="true" rows="12" maxSubmissions="1"/>',
        '  <Header value="Mô tả toàn cảnh"/>',
        f'  <TextArea name="{SUMMARY}" toName="{IMAGE}" editable="true" rows="4" maxSubmissions="1"/>',
        '  <Header value="Các câu truy vấn tăng dần (mỗi dòng một câu)"/>',
        f'  <TextArea name="{QUERIES}" toName="{IMAGE}" editable="true" rows="6" maxSubmissions="1"/>',
        "</View>",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- xuất
def annotation_to_task(annotation: ImageAnnotation, image_url_prefix: str) -> dict[str, Any]:
    """Một ``ImageAnnotation`` -> một task Label Studio có ``predictions`` (pre-label)."""
    w, h = annotation.width, annotation.height
    dims = {"original_width": w, "original_height": h, "image_rotation": 0}
    result: list[dict[str, Any]] = []
    for e in annotation.entities:
        geometry = box_to_percent(e.box, w, h)
        base = {"id": e.entity_id, "to_name": IMAGE, **dims}
        result.append({**base, "from_name": BOX, "type": "rectanglelabels", "value": {**geometry, "rectanglelabels": [e.label]}})
        if e.is_person:
            result.append({**base, "from_name": IDENTITY, "type": "choices", "value": {**geometry, "choices": [e.identity]}})
            if e.attributes is not None:
                text = json.dumps(e.attributes.model_dump(), ensure_ascii=False, indent=2)
                result.append({**base, "from_name": ATTRIBUTES, "type": "textarea", "value": {**geometry, "text": [text]}})
    corpus = annotation.dense_textual_corpus
    if corpus.global_summary:
        result.append({"id": "summary", "from_name": SUMMARY, "to_name": IMAGE, "type": "textarea", "value": {"text": [corpus.global_summary]}})
    if corpus.progressive_queries:
        result.append({"id": "queries", "from_name": QUERIES, "to_name": IMAGE, "type": "textarea", "value": {"text": ["\n".join(corpus.progressive_queries)]}})
    return {
        "data": {"image": f"{image_url_prefix}{annotation.file_name}", "image_id": annotation.image_id},
        "predictions": [{"model_version": MODEL_VERSION, "result": result}],
    }


def export_tasks(cfg: dict[str, Any], per_image_dir: Path, out_dir: Path, gallery_dir: Optional[Path] = None) -> dict[str, Any]:
    """Ghi ``label_studio_tasks.json`` và ``label_studio_config.xml`` từ mọi file per-image."""
    ls_cfg = cfg["label_studio"]
    annotations = [ImageAnnotation.model_validate(read_json(p)) for p in sorted(per_image_dir.glob("*.json"))]
    if not annotations:
        raise RuntimeError(f"Không có annotation nào trong {per_image_dir}. Chạy `annotate` trước.")

    labels = {e.label for a in annotations for e in a.entities}
    identities = {e.identity for a in annotations for e in a.entities} | set(ls_cfg.get("extra_identities") or [])
    if gallery_dir and gallery_dir.is_dir():
        identities |= {p.name for p in gallery_dir.iterdir() if p.is_dir()}

    tasks = [annotation_to_task(a, ls_cfg["image_url_prefix"]) for a in annotations]
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks_path, config_path = out_dir / "label_studio_tasks.json", out_dir / "label_studio_config.xml"
    write_json(tasks_path, tasks)
    config_path.write_text(build_label_config(labels, identities), encoding="utf-8")
    return {"tasks": len(tasks), "tasks_file": str(tasks_path), "config_file": str(config_path)}


def push_to_label_studio(cfg: dict[str, Any], tasks_path: Path, config_path: Path) -> int:
    """Tạo project và import task qua ``label-studio-sdk`` (>= 1.0).

    Cần ``LABEL_STUDIO_URL`` và ``LABEL_STUDIO_API_KEY``. Trả về ID project.
    Chưa được kiểm thử với một máy chủ Label Studio thật; cách chắc chắn là import hai file
    bằng giao diện web.
    """
    from label_studio_sdk import LabelStudio

    url, key = os.getenv("LABEL_STUDIO_URL"), os.getenv("LABEL_STUDIO_API_KEY")
    if not url or not key:
        raise RuntimeError("Cần đặt LABEL_STUDIO_URL và LABEL_STUDIO_API_KEY để đẩy task qua SDK.")
    client = LabelStudio(base_url=url, api_key=key)
    project = client.projects.create(
        title=cfg["label_studio"]["project_title"],
        label_config=config_path.read_text(encoding="utf-8"),
    )
    client.projects.import_tasks(id=project.id, request=read_json(tasks_path))
    return int(project.id)


# --------------------------------------------------------------------------- nhập
def _latest_result(task: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    """Kết quả của annotation mới nhất chưa bị huỷ; None nếu task chưa được ai kiểm tra."""
    submitted = [a for a in task.get("annotations") or [] if not a.get("was_cancelled")]
    if not submitted:
        return None
    latest = max(submitted, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
    return latest.get("result") or []


def apply_review(annotation: ImageAnnotation, result: list[dict[str, Any]], person_labels: set[str]) -> ImageAnnotation:
    """Áp kết quả kiểm tra của một task lên ``ImageAnnotation``.

    Box bị xoá trong Label Studio thì thực thể bị xoá; box mới được thêm với ID ``m01``...
    Thuộc tính sửa thành JSON hỏng thì giữ thuộc tính cũ và ghi lý do vào ``attributes_error``.
    """
    w, h = annotation.width, annotation.height
    existing = {e.entity_id: e for e in annotation.entities}
    regions: dict[str, dict[str, Any]] = {}
    summary: Optional[str] = None
    queries: Optional[list[str]] = None

    for item in result:
        name, value = item.get("from_name"), item.get("value") or {}
        if name == SUMMARY:
            summary = "\n".join(value.get("text") or []).strip()
        elif name == QUERIES:
            queries = [q.strip() for q in "\n".join(value.get("text") or []).splitlines() if q.strip()]
        elif name in (BOX, IDENTITY, ATTRIBUTES) and item.get("id"):
            regions.setdefault(str(item["id"]), {})[name] = value

    entities: list[EntityAnnotation] = []
    manual = 0
    for region_id, parts in regions.items():
        if BOX not in parts:  # vùng không còn box (đã bị xoá): bỏ
            continue
        box = percent_to_box(parts[BOX], w, h)
        label = (parts[BOX].get("rectanglelabels") or ["object"])[0]
        old = existing.get(region_id)
        if old is None:
            manual += 1
            entity_id = f"m{manual:02d}"
            entity = EntityAnnotation(entity_id=entity_id, label=label, box=box)
        else:
            entity = old.model_copy(deep=True)
            entity.label, entity.box = label, box
        entity.is_person = label in person_labels
        entity.spatial_position = spatial_position(box.as_tuple(), w, h)

        if IDENTITY in parts:
            chosen = (parts[IDENTITY].get("choices") or [UNKNOWN_IDENTITY])[0]
            if chosen != entity.identity:
                entity.identity, entity.identity_similarity = chosen, None  # người kiểm tra quyết định
        if ATTRIBUTES in parts:
            text = "\n".join(parts[ATTRIBUTES].get("text") or []).strip()
            if text:
                try:
                    entity.attributes = EntityAttributes.model_validate(json.loads(text))
                    entity.attributes_error = None
                except (json.JSONDecodeError, ValidationError) as exc:
                    entity.attributes_error = f"JSON thuộc tính sửa tay không hợp lệ: {exc}"
        entities.append(entity)

    updated = annotation.model_copy(deep=True)
    updated.entities = sorted(entities, key=lambda e: e.entity_id)
    kept_ids = {e.entity_id for e in updated.entities}
    removed = set(existing) - kept_ids
    # Quan hệ trỏ tới thực thể đã bị xoá không còn nghĩa.
    updated.scene_graph_relationships = [
        r for r in updated.scene_graph_relationships if r.subject not in removed and r.object not in removed
    ]
    if summary is not None:
        updated.dense_textual_corpus.global_summary = summary
    if queries is not None:
        updated.dense_textual_corpus.progressive_queries = queries
    updated.dense_textual_corpus.bm25_blob = build_bm25_blob(updated)
    updated.verified = True
    return updated


def import_export(export_path: Path, per_image_dir: Path) -> dict[str, int]:
    """Đọc file export JSON của Label Studio và cập nhật các file per-image tương ứng."""
    log = get_logger()
    tasks = read_json(export_path)
    if not isinstance(tasks, list):
        raise ValueError("File export phải là danh sách task (định dạng JSON của Label Studio).")
    updated = skipped = missing = 0
    for task in tasks:
        data = task.get("data") or {}
        image_id = data.get("image_id") or Path(str(data.get("image", "")).split("=")[-1]).stem
        result = _latest_result(task)
        if result is None:
            skipped += 1
            continue
        path = per_image_dir / f"{image_id}.json"
        if not path.exists():
            log.warning("ls-import: không thấy %s", path)
            missing += 1
            continue
        annotation = ImageAnnotation.model_validate(read_json(path))
        write_json(path, apply_review(annotation, result, person_labels={"person"}).model_dump())
        updated += 1
    return {"updated": updated, "not_reviewed": skipped, "missing": missing}

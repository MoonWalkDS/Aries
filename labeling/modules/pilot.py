"""Stage pilot: chạy thử một tập ảnh nhỏ với nhiều model rồi xuất báo cáo so sánh.

Mục đích: quyết định dùng model nào cho cả 5000 ảnh dựa trên kết quả thật (JSON hợp lệ,
độ chi tiết trang phục, thời gian, số token) thay vì đoán.

Các model có thể chạy ở các máy khác nhau (Qwen trên Kaggle, API ở máy local): mỗi lượt ghi
vào ``output_jsons/pilot/<tên>/``; ``build_report`` gom mọi thư mục đang có thành một file HTML.
"""
from __future__ import annotations

import base64
import html
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .schemas import DetectionRecord, IdentityFile, ImageAnnotation
from .utils import append_jsonl, get_logger, image_id_of, read_json, read_jsonl, resolve_from_config, write_json
from .vlm_annotator import VLMStructuredAnnotator, attributes_brief, prepare_crop
from .vlm_backends import BackendError, create_backend


# --------------------------------------------------------------------------- chọn ảnh
def known_answer_ids(cfg: dict[str, Any]) -> list[str]:
    """ID ảnh đã nộp ĐÚNG trên DRES (đọc submissions.log) - các ảnh ta biết đề hỏi gì."""
    pcfg = cfg["pilot"]
    log_path = resolve_from_config(cfg, pcfg["submissions_log"])
    if not log_path.exists():
        return []
    prefix = pcfg.get("item_prefix", "")
    ids: list[str] = []
    for entry in read_jsonl(log_path):
        if '"CORRECT"' not in str(entry.get("response", "")):
            continue
        item = str(entry.get("item", ""))
        image_id = item[len(prefix):] if prefix and item.startswith(prefix) else item
        if image_id and image_id not in ids:
            ids.append(image_id)
    return ids


def select_pilot_ids(cfg: dict[str, Any], all_image_paths: list[Path]) -> list[str]:
    """Ảnh đáp án đã biết + phần còn lại rải đều theo giờ (tiền tố HH_ của tên file)."""
    target = int(cfg["pilot"]["num_images"])
    available = {image_id_of(p) for p in all_image_paths}
    chosen = [i for i in known_answer_ids(cfg) if i in available][:target]

    by_hour: dict[str, list[str]] = defaultdict(list)
    for image_id in sorted(available):
        by_hour[image_id.split("_")[0]].append(image_id)
    # Lấy lần lượt một ảnh ở giữa mỗi giờ, rồi các vị trí khác, tới khi đủ số lượng.
    hours = sorted(by_hour)
    offsets = [0.5, 0.25, 0.75, 0.1, 0.9, 0.4, 0.6]
    for offset in offsets:
        for hour in hours:
            if len(chosen) >= target:
                return chosen
            pool = by_hour[hour]
            candidate = pool[min(len(pool) - 1, int(len(pool) * offset))]
            if candidate not in chosen:
                chosen.append(candidate)
    return chosen


# --------------------------------------------------------------------------- chạy
def run_pilot_model(
    cfg: dict[str, Any],
    model_spec: dict[str, Any],
    image_ids: list[str],
    images_dir: Path,
    jsons_dir: Path,
) -> dict[str, Any]:
    """Chạy annotate cho các ảnh pilot bằng một model. Ảnh đã có kết quả được bỏ qua (resume)."""
    log = get_logger()
    name = model_spec["name"]
    out_dir = jsons_dir / "pilot" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = {k: v for k, v in model_spec.items() if k not in ("name", "backend")}
    backend = create_backend(cfg, model_spec["backend"], overrides)
    annotator = VLMStructuredAnnotator(backend, cfg)

    done = failed = skipped = 0
    try:
        for image_id in image_ids:
            out_path = out_dir / f"{image_id}.json"
            if out_path.exists():
                skipped += 1
                continue
            det_path = jsons_dir / "detections" / f"{image_id}.json"
            if not det_path.exists():
                log.warning("pilot %s: thiếu detection cho %s (chạy `detect` trước)", name, image_id)
                failed += 1
                continue
            record = DetectionRecord.model_validate(read_json(det_path))
            id_path = jsons_dir / "identities" / f"{image_id}.json"
            identities = IdentityFile.model_validate(read_json(id_path)) if id_path.exists() else None
            try:
                annotation = annotator.annotate_image(images_dir / record.file_name, record, identities)
            except BackendError:
                raise  # lỗi cấu hình/quota: dừng cả lượt, không cố tiếp
            except Exception as exc:  # JSON hỏng sau khi đã thử lại, ảnh lỗi...
                failed += 1
                append_jsonl(out_dir / "_failed.jsonl", {"image_id": image_id, "error": str(exc)})
                log.warning("pilot %s: %s lỗi: %s", name, image_id, exc)
                continue
            write_json(out_path, annotation.model_dump())
            done += 1
            log.info("pilot %s: %s xong (%.1fs)", name, image_id, annotation.meta.seconds)
    finally:
        backend.close()
    return {"model": name, "done": done, "skipped": skipped, "failed": failed}


# --------------------------------------------------------------------------- báo cáo
def _data_uri(image: Image.Image, max_side: int) -> str:
    image = image.copy()
    image.thumbnail((max_side, max_side), Image.LANCZOS)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _model_stats(annotations: list[ImageAnnotation], failed: int) -> dict[str, Any]:
    n = len(annotations)
    persons = sum(1 for a in annotations for e in a.entities if e.is_person)
    described = sum(1 for a in annotations for e in a.entities if e.is_person and e.attributes is not None)
    uncertain = sum(len(e.attributes.uncertain) for a in annotations for e in a.entities if e.attributes)
    total = n + failed
    return {
        "images_ok": n,
        "images_failed": failed,
        "valid_rate": round(n / total, 3) if total else 0.0,
        "persons": persons,
        "persons_described": described,
        "uncertain_fields": uncertain,
        "sec_per_image": round(sum(a.meta.seconds for a in annotations) / n, 1) if n else 0.0,
        "input_tokens_per_image": round(sum(a.meta.input_tokens for a in annotations) / n) if n else 0,
        "output_tokens_per_image": round(sum(a.meta.output_tokens for a in annotations) / n) if n else 0,
    }


def build_report(cfg: dict[str, Any], images_dir: Path, jsons_dir: Path, crops_cfg: Optional[dict[str, Any]] = None) -> Path:
    """Gom mọi thư mục ``output_jsons/pilot/<model>/`` thành ``pilot_report.html`` (tự chứa ảnh)."""
    pilot_dir = jsons_dir / "pilot"
    model_dirs = sorted(p for p in pilot_dir.iterdir() if p.is_dir()) if pilot_dir.is_dir() else []
    if not model_dirs:
        raise RuntimeError(f"Chưa có kết quả pilot nào trong {pilot_dir}")
    vcfg = cfg["vlm"]

    results: dict[str, dict[str, ImageAnnotation]] = {}
    stats: dict[str, dict[str, Any]] = {}
    for model_dir in model_dirs:
        annotations = {
            p.stem: ImageAnnotation.model_validate(read_json(p))
            for p in sorted(model_dir.glob("*.json"))
        }
        failed_path = model_dir / "_failed.jsonl"
        failed_ids = {r["image_id"] for r in read_jsonl(failed_path)} if failed_path.exists() else set()
        results[model_dir.name] = annotations
        stats[model_dir.name] = _model_stats(list(annotations.values()), len(failed_ids - set(annotations)))
    models = list(results)
    image_ids = sorted({i for annotations in results.values() for i in annotations})

    esc = html.escape
    out: list[str] = [
        "<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>Pilot report</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:16px;color:#222}"
        "table{border-collapse:collapse;width:100%;margin-bottom:24px}"
        "th,td{border:1px solid #ccc;padding:6px 8px;vertical-align:top;font-size:13px;text-align:left}"
        "th{background:#f0f0f0}img{max-width:100%}.crop{height:150px;margin:2px}"
        ".ent{margin:4px 0;padding:4px;background:#f8f8f8}.unc{color:#b45309}.err{color:#b91c1c}"
        "h2{margin-top:32px}ol{margin:4px 0;padding-left:18px}</style></head><body>",
        "<h1>Pilot: so sánh model mô tả ảnh</h1>",
        "<table><tr><th>Model</th>" + "".join(f"<th>{esc(k)}</th>" for k in next(iter(stats.values()))) + "</tr>",
    ]
    for model in models:
        out.append(f"<tr><td><b>{esc(model)}</b></td>" + "".join(f"<td>{esc(str(v))}</td>" for v in stats[model].values()) + "</tr>")
    out.append("</table>")

    for image_id in image_ids:
        first = next(results[m][image_id] for m in models if image_id in results[m])
        image_path = images_dir / first.file_name
        out.append(f"<h2>{esc(image_id)}</h2>")
        crops_html = ""
        if image_path.exists():
            with Image.open(image_path) as im:
                image = im.convert("RGB")
            out.append(f"<img src='{_data_uri(image, 900)}' alt='{esc(image_id)}'>")
            det_path = jsons_dir / "detections" / f"{image_id}.json"
            if det_path.exists():
                record = DetectionRecord.model_validate(read_json(det_path))
                for det in record.persons():
                    crop = prepare_crop(image, det, int(vcfg["crop_min_short_side"]), int(vcfg["crop_max_long_side"]))
                    crops_html += f"<figure style='display:inline-block;margin:4px'><img class='crop' src='{_data_uri(crop, 300)}'><figcaption>{esc(det.entity_id)}</figcaption></figure>"
        out.append(f"<div>{crops_html}</div>")

        out.append("<table><tr>" + "".join(f"<th>{esc(m)}</th>" for m in models) + "</tr><tr>")
        for model in models:
            annotation = results[model].get(image_id)
            if annotation is None:
                out.append("<td class='err'>không có kết quả</td>")
                continue
            cell = [f"<p>{esc(annotation.dense_textual_corpus.global_summary)}</p><ol>"]
            cell += [f"<li>{esc(q)}</li>" for q in annotation.dense_textual_corpus.progressive_queries]
            cell.append("</ol>")
            for e in annotation.entities:
                if not e.is_person:
                    continue
                if e.attributes is None:
                    cell.append(f"<div class='ent err'><b>{esc(e.entity_id)}</b>: {esc(e.attributes_error or 'không mô tả')}</div>")
                    continue
                unc = f" <span class='unc'>(không rõ: {esc(', '.join(e.attributes.uncertain))})</span>" if e.attributes.uncertain else ""
                cell.append(f"<div class='ent'><b>{esc(e.entity_id)}</b> [{esc(e.identity)}]: {esc(attributes_brief(e.attributes))}{unc}</div>")
            bg = "; ".join(annotation.background_clutter.secondary_actions)
            if bg:
                cell.append(f"<p><i>Nền:</i> {esc(bg)}</p>")
            cell.append(f"<small>{annotation.meta.seconds}s, {annotation.meta.input_tokens} in / {annotation.meta.output_tokens} out</small>")
            out.append("<td>" + "".join(cell) + "</td>")
        out.append("</tr></table>")
    out.append("</body></html>")

    report_path = pilot_dir / "pilot_report.html"
    report_path.write_text("\n".join(out), encoding="utf-8")
    write_json(pilot_dir / "pilot_stats.json", stats)
    return report_path

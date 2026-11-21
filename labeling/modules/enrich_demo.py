"""Trang demo cho kết quả làm giàu: ``python main.py enrich demo`` -> ``output_jsons/demo/index.html``.

Mỗi ảnh có: ảnh đã vẽ box người + khung vùng cố định kèm nhãn, bảng từng người (crop, vùng đang
đứng, hành động), bảng từng vùng (crop, có đổi so với nền không, trạng thái) và bản ghi JSON gốc.
Mỗi nhãn ghi rõ nguồn: người đã duyệt, hạt giống chưa duyệt, hay máy gợi ý.

Trang tham chiếu ảnh bằng đường dẫn tương đối nên chỉ mở được trên máy có dữ liệu.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image, ImageDraw

from . import adl, enrich
from .utils import get_logger, image_id_of, read_json, resolve_from_config
from .visualize import _fonts

SOURCE_VI = {"human": "người đã duyệt", "vlm": "VLM đã duyệt", "seed": "hạt giống, chưa duyệt", "suggested": "máy gợi ý, chưa duyệt"}
SOURCE_COLOR = {"human": (26, 127, 75), "vlm": (127, 90, 200), "seed": (186, 117, 23), "suggested": (47, 95, 208), None: (110, 110, 105)}
ROI_COLOR = (0, 190, 200)


def _label_text(entry: Optional[dict[str, Any]]) -> str:
    return (entry.get("label_vi") or entry["label"]) if entry else "chưa có nhãn"


def _tag(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: Any, fill: tuple[int, int, int]) -> None:
    """Nhãn chữ trắng trên nền màu, giữ trong khung ảnh."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left + 8, bottom - top + 6
    x = max(0, min(xy[0], draw.im.size[0] - w))
    y = max(0, min(xy[1], draw.im.size[1] - h))
    draw.rectangle((x, y, x + w, y + h), fill=fill)
    draw.text((x + 4, y + 3 - top), text, font=font, fill=(255, 255, 255))


def render_annotated(image_path: Path, record: dict[str, Any], rois: Sequence[enrich.Roi], out_path: Path, max_side: int = 1280) -> Path:
    """Vẽ khung vùng cố định (nét mảnh) và box người (nét dày, màu theo nguồn nhãn) lên ảnh."""
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    draw = ImageDraw.Draw(image)
    font, bold = _fonts(max(12, image.width // 85))
    w, h = image.size
    for roi in rois:
        info = record["rois"].get(roi.name)
        if info is None:
            continue
        x1, y1, x2, y2 = roi.box[0] * w, roi.box[1] * h, roi.box[2] * w, roi.box[3] * h
        draw.rectangle((x1, y1, x2, y2), outline=ROI_COLOR, width=3 if info["changed"] else 1)
        if info["state"]:
            _tag(draw, (x1, y2 - 2), _label_text(info["state"]), font, (0, 120, 128))
    for n, person in enumerate(record["people"], start=1):
        action = person.get("action")
        color = SOURCE_COLOR.get(action["source"] if action else None, SOURCE_COLOR[None])
        x1, y1, x2, y2 = (v * scale for v in person["box"])
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        _tag(draw, (x1, y1 - (getattr(bold, "size", 12) + 8)),f"p{n:02d} · {_label_text(action)}", bold, color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=88)
    return out_path


PAGE_HEAD = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Demo làm giàu dữ liệu Castle</title>
<style>
  :root { --bg:#f6f6f4; --fg:#1c1c1a; --muted:#6b6b66; --card:#fff; --line:#d9d9d4; --human:#1a7f4b; --vlm:#6b4fc0; --seed:#ba7517; --suggested:#2f5fd0; --none:#6e6e69; --roi:#007880; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161615; --fg:#ecece8; --muted:#a0a09a; --card:#222220; --line:#3a3a36; --human:#4cc38a; --vlm:#b39cf5; --seed:#f0b253; --suggested:#8fb0ff; --none:#a0a09a; --roi:#4fd0d8; }
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui, sans-serif; }
  header { padding:18px 20px 10px; border-bottom:1px solid var(--line); }
  h1 { font-size:20px; margin:0 0 6px; } h2 { font-size:16px; margin:0 0 10px; } h3 { font-size:14px; margin:16px 0 6px; }
  .summary, .legend { color:var(--muted); display:flex; flex-wrap:wrap; gap:6px 18px; }
  main { padding:16px 20px 60px; max-width:1500px; margin:0 auto; }
  section { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:22px; }
  .layout { display:grid; grid-template-columns:minmax(0, 3fr) minmax(0, 2fr); gap:18px; align-items:start; }
  @media (max-width: 1000px) { .layout { grid-template-columns:1fr; } }
  .frame img { width:100%; border-radius:6px; display:block; }
  .tablewrap { overflow-x:auto; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); vertical-align:middle; }
  th { color:var(--muted); font-weight:500; white-space:nowrap; }
  td img { height:84px; max-width:120px; object-fit:contain; border-radius:4px; background:#0002; display:block; }
  .src { font-size:12px; white-space:nowrap; } .num { font-variant-numeric:tabular-nums; }
  .human { color:var(--human); } .vlm { color:var(--vlm); } .seed { color:var(--seed); } .suggested { color:var(--suggested); } .none { color:var(--none); }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:currentColor; margin-right:5px; }
  .changed { color:var(--roi); font-weight:500; }
  details { margin-top:12px; } summary { cursor:pointer; color:var(--muted); }
  pre { background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:10px; overflow-x:auto; font-size:12px; max-height:420px; }
  a { color:var(--suggested); }
</style>
</head>
<body>
"""


def _rel(target: Path, start: Path) -> str:
    try:
        return Path(os.path.relpath(target, start)).as_posix()
    except ValueError:
        return target.resolve().as_uri()


def _source_cell(entry: Optional[dict[str, Any]]) -> str:
    if not entry:
        return '<span class="src none"><span class="dot"></span>chưa có nhãn</span>'
    score = "" if entry.get("score") is None else f' · <span class="num">{float(entry["score"]):.2f}</span>'
    return f'<span class="src {entry["source"]}"><span class="dot"></span>{html.escape(SOURCE_VI.get(entry["source"], entry["source"]))}{score}</span>'


def _label_cell(entry: Optional[dict[str, Any]]) -> str:
    if not entry:
        return "—"
    vi = html.escape(entry.get("label_vi") or "")
    return f"<strong>{vi}</strong><br>{html.escape(entry['label'])}" if vi else f"<strong>{html.escape(entry['label'])}</strong>"


def run_demo(cfg: dict[str, Any], image_paths: Sequence[Path], data_dir: Path, jsons_dir: Path, max_images: int = 60) -> dict[str, Any]:
    """Dựng ``output_jsons/demo/index.html`` cho các ảnh được chọn (tối đa ``max_images``)."""
    log = get_logger()
    enriched_dir, out_dir = jsons_dir / "enriched", jsons_dir / "demo"
    if not enriched_dir.is_dir():
        raise RuntimeError(f"Chưa có {enriched_dir}. Chạy `enrich merge` (hoặc `pipeline`) trước.")
    rois = enrich.load_rois(resolve_from_config(cfg, cfg["enrich"]["rois_file"]))
    chosen = [p for p in image_paths if (enriched_dir / f"{image_id_of(p)}.json").exists()]
    if len(chosen) > max_images:
        log.warning("demo: %d ảnh có bản ghi, chỉ dựng %d ảnh đầu. Dùng --limit / --ids-file để chọn.", len(chosen), max_images)
        chosen = chosen[:max_images]
    if not chosen:
        raise RuntimeError("Không ảnh nào được chọn có bản ghi trong enriched/.")
    image_labels = enrich._labels_of(jsons_dir / "adl", resolve_from_config(cfg, cfg["adl"]["labels_file"]))
    people_crops, roi_root = data_dir / "people" / "crops", data_dir / "rois"

    parts: list[str] = []
    totals = {"people": 0, "people_labeled": 0, "roi_changed": 0, "roi_states": 0}
    for path in chosen:
        image_id = image_id_of(path)
        record = read_json(enriched_dir / f"{image_id}.json")
        annotated = render_annotated(path, record, rois, out_dir / "annotated" / f"{image_id}.jpg")
        scene = enrich._label_entry(image_id, *image_labels)
        rows = []
        for n, person in enumerate(record["people"], start=1):
            crop = people_crops / f"{person['id']}.jpg"
            thumb = f'<img src="{html.escape(_rel(crop, out_dir))}" alt="crop người {n}" loading="lazy">' if crop.exists() else ""
            rows.append(
                f"<tr><td>{thumb}</td><td class='num'>p{n:02d}<br><span class='src none'>{person['score']:.2f}</span></td>"
                f"<td>{html.escape(person.get('zone_vi') or '—')}<br><span class='src none'>{html.escape(person.get('zone') or '')}</span></td>"
                f"<td>{_label_cell(person.get('action'))}</td><td>{_source_cell(person.get('action'))}</td></tr>"
            )
            totals["people"] += 1
            totals["people_labeled"] += person.get("action") is not None
        roi_rows = []
        for roi in rois:
            info = record["rois"].get(roi.name)
            if info is None:
                continue
            crop = roi_root / roi.name / "crops" / f"{image_id}.jpg"
            if not crop.exists():      # vùng không đổi: dùng ảnh nền của giờ đó
                crop = roi_root / roi.name / "crops" / f"{enrich.BASELINE_PREFIX}{record['hour']}.jpg"
            thumb = f'<img src="{html.escape(_rel(crop, out_dir))}" alt="vùng {html.escape(roi.name)}" loading="lazy">' if crop.exists() else ""
            changed = f'<span class="changed">đã đổi</span> <span class="num">{info["motion"]:.0%}</span>' if info["changed"] else f'như nền <span class="num">{info["motion"]:.0%}</span>'
            roi_rows.append(
                f"<tr><td>{thumb}</td><td>{html.escape(info.get('name_vi') or roi.name)}<br><span class='src none'>{html.escape(roi.name)}</span></td>"
                f"<td>{changed}</td><td>{_label_cell(info['state'])}</td><td>{_source_cell(info['state'])}</td></tr>"
            )
            totals["roi_changed"] += bool(info["changed"])
            totals["roi_states"] += info["state"] is not None
        raw = html.escape(json.dumps(record, ensure_ascii=False, indent=1))
        parts.append(f"""<section id="{html.escape(image_id)}">
<h2>{html.escape(image_id)} · giờ {html.escape(record['hour'])} · {record['num_people']} người</h2>
<div class="layout">
<div class="frame"><a href="{html.escape(_rel(path, out_dir))}" target="_blank" rel="noopener"><img src="{html.escape(_rel(annotated, out_dir))}" alt="Ảnh {html.escape(image_id)} đã vẽ box người và khung vùng" loading="lazy"></a>
<h3>Nhãn cấp ảnh</h3><div>{_label_cell(scene).replace('<br>', ' · ')} {_source_cell(scene)}</div></div>
<div>
<h3>Tầng người</h3>
<div class="tablewrap"><table><thead><tr><th>Crop</th><th>Người</th><th>Vùng đang đứng</th><th>Hành động</th><th>Nguồn nhãn</th></tr></thead><tbody>{''.join(rows) or '<tr><td colspan="5">Không dò thấy người nào.</td></tr>'}</tbody></table></div>
</div>
</div>
<h3>Tầng vùng cố định</h3>
<div class="tablewrap"><table><thead><tr><th>Crop</th><th>Vùng</th><th>So với nền</th><th>Trạng thái</th><th>Nguồn nhãn</th></tr></thead><tbody>{''.join(roi_rows)}</tbody></table></div>
<details><summary>Bản ghi JSON (output_jsons/enriched/{html.escape(image_id)}.json)</summary><pre>{raw}</pre></details>
</section>""")

    stats = adl_stats = ""
    build_stats = data_dir / "people" / "build_stats.json"
    if build_stats.exists():
        info = read_json(build_stats)
        stats = f"<span>Model embedding: {html.escape(str(info.get('model')))} ({html.escape(str(info.get('pretrained')))})</span>"
    header = f"""<header><h1>Demo làm giàu dữ liệu Castle</h1>
<div class="summary"><span>{len(chosen)} ảnh</span><span>{totals['people']} người, {totals['people_labeled']} có nhãn hành động</span>
<span>{totals['roi_changed']} lượt vùng thay đổi, {totals['roi_states']} trạng thái vùng</span>{stats}{adl_stats}</div>
<div class="legend" style="margin-top:6px"><span class="human"><span class="dot"></span>người đã duyệt</span><span class="vlm"><span class="dot"></span>VLM đã duyệt</span><span class="seed"><span class="dot"></span>hạt giống, chưa duyệt</span>
<span class="suggested"><span class="dot"></span>máy gợi ý, chưa duyệt</span><span class="none"><span class="dot"></span>chưa có nhãn</span>
<span style="color:var(--roi)">khung xanh ngọc: vùng cố định (nét dày = đã đổi so với nền)</span><span>Số sau nguồn nhãn là cosine.</span></div></header>
"""
    page = out_dir / "index.html"
    page.write_text(PAGE_HEAD + header + "<main>\n" + "\n".join(parts) + "\n</main>\n</body>\n</html>\n", encoding="utf-8")
    return {"images": len(chosen), **totals, "page": str(page)}

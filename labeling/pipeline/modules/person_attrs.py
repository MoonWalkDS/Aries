"""Mô tả ngoại hình từng người bằng VLM (``python main.py pipeline --steps attrs``).

Mỗi crop người của tầng enrich (``data/enrich/people/crops``) được hỏi MỘT lần với prompt ngắn: giới, nhóm
tuổi, tóc, áo, quần, màu, đồ đang cầm... Câu trả lời là JSON gọn (khoảng 80 token) để chạy được hàng
chục nghìn crop trên GPU Kaggle. Kết quả nằm trong ``output_jsons/adl_people/person_attrs.jsonl``, mỗi
dòng một crop, ghi ngay sau mỗi lệnh gọi: chạy lại chỉ hỏi những crop chưa có. ``enrich merge`` gắn
kết quả vào từng người trong bản ghi gộp.

Crop quá nhỏ (người ở xa, bị cắt ở mép) bị bỏ qua vì VLM chỉ đoán bừa trên đó.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .utils import append_jsonl, get_logger, read_json, read_jsonl
from .vlm_annotator import extract_json, resize_long_side
from .vlm_backends import VLMBackend

ATTRS_FILE = "person_attrs.jsonl"  # trong output_jsons/adl_people/

# Một dòng 13 trường ngăn bằng ";" thay cho JSON nhiều dòng: thử trên Kaggle T4, JSON tốn 8 s/crop vì
# model in khối ```json thụt lề (khoảng 300 ký tự); một dòng ngắn giảm còn khoảng một nửa.
PROMPT = """This is a crop of one person seen from a ceiling camera in a shared kitchen.
Describe only what you can see. Use short lowercase English words and "unknown" when something is not visible.
Reply with ONE line of 13 fields separated by ";" in this exact order, and nothing else:
gender (male/female/unknown); age (child/young adult/adult/senior/unknown); hair color and length; top garment type;
top color; top pattern (plain/striped/checked/printed/unknown); bottom garment type; bottom color; headwear (type or none);
glasses (yes/no/unknown); apron (yes/no/unknown); objects held, comma-separated, or none; view (front/back/side/top/partial)
Example: male; adult; short black; sweater; gray; plain; jeans; blue; none; yes; no; cup, phone; back"""

FIELDS = ("gender", "age", "hair", "top", "top_color", "top_pattern", "bottom", "bottom_color", "headwear", "glasses", "apron", "holding", "view")
_EMPTY = ("none", "unknown", "nothing", "")


def _clean(key: str, value: Any) -> Any:
    if key == "holding":
        items = value if isinstance(value, list) else str(value or "").split(",")
        return [t for t in (str(v).strip().lower() for v in items) if t not in _EMPTY]
    return str(value).strip().lower() or "unknown" if value is not None else "unknown"


def parse_attrs(text: str) -> Optional[dict[str, Any]]:
    """Dòng 13 trường ngăn bằng ";" (hoặc JSON cùng khoá) -> dict chữ thường; None nếu không đọc được."""
    for line in (l.strip().strip("`") for l in text.strip().splitlines()):
        parts = [part.strip() for part in line.split(";")]
        if len(parts) == len(FIELDS):
            return {key: _clean(key, value) for key, value in zip(FIELDS, parts)}
    try:
        data = extract_json(text)
    except ValueError:
        return None
    return {key: _clean(key, data.get(key)) for key in FIELDS} if isinstance(data, dict) else None


def load_attrs(path: Path) -> dict[str, dict[str, Any]]:
    """{person_id: attrs} từ file jsonl; dòng sau ghi đè dòng trước."""
    return {row["id"]: row["attrs"] for row in read_jsonl(path) if row.get("attrs")} if path.exists() else {}


def run_attrs(cfg: dict[str, Any], data_dir: Path, jsons_dir: Path, backend: VLMBackend, limit: Optional[int] = None) -> dict[str, Any]:
    """Hỏi VLM cho mọi crop người đủ lớn chưa có mô tả (tối đa ``limit`` crop mới)."""
    log = get_logger()
    acfg = cfg["enrich"]["attrs"]
    people = read_json(data_dir / "people" / "people.json")
    crops_dir = data_dir / "people" / "crops"
    out_path = jsons_dir / "adl_people" / ATTRS_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {row["id"] for row in read_jsonl(out_path)} if out_path.exists() else set()
    min_height = float(acfg["min_box_height"])
    todo = [pid for pid, person in sorted(people.items()) if pid not in done and person["box"][3] - person["box"][1] >= min_height]
    skipped_small = len(people) - len(done) - len(todo)
    if limit is not None:
        todo = todo[:limit]
    log.info("attrs: %d crop cần hỏi, %d đã có, %d quá nhỏ bỏ qua", len(todo), len(done), skipped_small)
    calls = failed = 0
    started = time.perf_counter()
    for pid in todo:
        path = crops_dir / f"{pid}.jpg"
        if not path.exists():
            log.warning("attrs: thiếu crop %s", path)
            continue
        with Image.open(path) as im:
            image = resize_long_side(im.convert("RGB"), int(acfg["max_long_side"]))
        attrs, raw = None, ""
        for _ in range(int(acfg.get("retries", 1)) + 1):
            raw = backend.generate([image], PROMPT, max_new_tokens=int(acfg["max_new_tokens"])).text
            calls += 1
            attrs = parse_attrs(raw)
            if attrs is not None:
                break
        failed += attrs is None
        append_jsonl(out_path, {"id": pid, "attrs": attrs, "raw": raw[:400], "model": backend.model_id})
        if calls % 200 == 0:
            log.info("attrs: %d lệnh gọi, %.1f s/lệnh", calls, (time.perf_counter() - started) / calls)
    seconds = time.perf_counter() - started
    return {"asked": len(todo), "already": len(done), "skipped_small": skipped_small, "unreadable": failed, "vlm_calls": calls,
            "sec_per_call": round(seconds / calls, 2) if calls else None, "model": backend.model_id}

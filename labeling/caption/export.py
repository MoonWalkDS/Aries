"""Dữ liệu đầu vào để GPT viết caption tiếng Việt cho từng ảnh.

    python labeling/caption/export.py            # ghi data/exports/caption/inputs.jsonl và prompt_vi.md

Mỗi ảnh một bản ghi gọn: từng người (mô tả trang phục, hành động + mức chắc chắn, vùng đứng, đồ đang cầm), trạng
thái các vùng có thông tin (bỏ trạng thái rỗng như "tủ lạnh đóng"), và câu mô tả tiếng Anh của thầy làm tham chiếu.
Tên định danh là tên gán nhanh của người duyệt, không phải mô tả, nên chỉ giữ trong trường ``tag`` để truy vết.
Gọi GPT bằng ``caption_gpt.py``.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]   # labeling/
EXPORTS = ROOT.parent / "data" / "exports"
OUT = EXPORTS / "caption"
TEACHER = ROOT.parent / "data" / "dataset" / "raw" / "teacher_metadata.csv"
HINTS = ROOT / "annotations" / "adl_people" / "teacher_hints.json"
ID_MIN_PROB = 0.6           # dưới mức này mô hình định danh chưa chắc: dùng thuộc tính VLM thay mô tả của tên
EMPTY_STATE = re.compile(r"không|trống|đóng|chỉ có|bị người che")   # trạng thái vùng không thêm thông tin

PROMPT = """Bạn viết caption tiếng Việt cho ảnh camera cố định trong một căn bếp (bộ dữ liệu Castle 5000).
Đầu vào là JSON của MỘT ảnh, do người gán nhãn và mô hình tạo:
- people: từng người trong ảnh. "mo_ta" là trang phục, "hanh_dong" kèm "do_chac" (chắc / khá chắc / chưa chắc),
  "vung" là chỗ đứng, "cam" là đồ đang cầm, "huong" là hướng người so với camera.
- vung_bep: trạng thái các khu trong bếp.
- thay_mo_ta: câu mô tả tiếng Anh của giáo viên cho cả ảnh: thường đúng nhưng hay thiếu người và hành động.
Nếu có ảnh đính kèm thì ảnh là nguồn đúng nhất khi mâu thuẫn với nhãn.

Yêu cầu:
1. Viết 2-4 câu tiếng Việt tự nhiên: có bao nhiêu người, mỗi người trông thế nào (theo "mo_ta") và đang làm gì ở đâu,
   rồi một câu ngắn về khung cảnh bếp nếu có chi tiết đáng nói.
2. Hành động "chắc": viết khẳng định. "khá chắc": viết bình thường nếu không mâu thuẫn với thay_mo_ta hay ảnh.
   "chưa chắc": chỉ viết khi thay_mo_ta hoặc ảnh ủng hộ, nếu không thì chỉ nói người đó đứng ở đâu.
3. Không bịa người, đồ vật hay hành động không có trong đầu vào hoặc ảnh. Không nhắc tới "tag", điểm số hay JSON.
4. Ảnh không có người: tả khung cảnh bếp từ vung_bep và thay_mo_ta.
Trả về JSON đúng dạng: {"caption": "..."}"""


def _attrs_text(a: dict[str, Any]) -> str:
    """Mô tả ngắn từ thuộc tính VLM khi chưa chắc người này là ai."""
    bits = [{"male": "nam", "female": "nữ"}.get(a.get("gender", ""), "người"),
            " ".join(x for x in (a.get("top_color"), a.get("top")) if x), " ".join(x for x in (a.get("bottom_color"), a.get("bottom")) if x)]
    if a.get("glasses") == "yes":
        bits.append("đeo kính")
    if a.get("headwear") not in (None, "", "none"):
        bits.append(a["headwear"])
    return ", ".join(b for b in bits if b)


def _action(p: dict[str, Any], hint: dict[str, str] | None) -> dict[str, str] | None:
    a = p.get("action")
    if not a or not a.get("label"):
        return None
    sure = {"human": "chắc", "custom": "chắc", "fused": "khá chắc"}.get(a.get("source") or "", "chưa chắc")
    if sure == "chưa chắc" and hint and hint.get("label") == a["label"]:
        sure = "khá chắc"      # câu của thầy tả đúng hành động này cho người này
    return {"hanh_dong": a.get("label_vi") or a["label"], "do_chac": sure}


def person_record(p: dict[str, Any], hint: dict[str, str] | None) -> dict[str, Any]:
    ident = p.get("identity") or {}
    prob = (ident.get("predicted") or {}).get("prob")
    known = ident.get("name") and (ident.get("source") != "model" or (prob or 0) >= ID_MIN_PROB)
    attrs = p.get("attributes") or {}
    rec: dict[str, Any] = {"id": p["id"], "tag": ident.get("name") if known else None,
                           "mo_ta": ((p.get("outfit") or {}).get("description") if known else None) or _attrs_text(attrs),
                           "vung": p.get("zone_vi") or None}
    act = _action(p, hint)
    if act:
        rec.update(act)
    if attrs.get("holding"):
        rec["cam"] = attrs["holding"]
    if attrs.get("view"):
        rec["huong"] = {"front": "nhìn về camera", "back": "quay lưng", "side": "nghiêng"}.get(attrs["view"], attrs["view"])
    return rec


def image_record(r: dict[str, Any], teacher: dict[str, str], hints: dict[str, dict[str, str]]) -> dict[str, Any]:
    rois = [f"{v.get('name_vi') or k}: {v['state']['label_vi']}" for k, v in r["rois"].items()
            if v.get("state") and v["state"].get("label_vi") and not EMPTY_STATE.search(v["state"]["label_vi"])]
    return {"image_id": r["image_id"], "gio": r.get("hour"), "so_nguoi": len(r["people"]),
            "people": [person_record(p, hints.get(p["id"])) for p in r["people"]], "vung_bep": rois,
            "thay_mo_ta": teacher.get(r["image_id"], "")}


def main() -> None:
    teacher = {}
    if TEACHER.exists():
        with open(TEACHER, encoding="utf-8") as f:
            teacher = {row["imageID"].replace("day1_Kitchen_", ""): row.get("description") or "" for row in csv.DictReader(f)}
    hints = json.loads(HINTS.read_text(encoding="utf-8")) if HINTS.exists() else {}
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(EXPORTS / "castle_dataset.jsonl", encoding="utf-8") as src, open(OUT / "inputs.jsonl", "w", encoding="utf-8") as dst:
        for line in src:
            dst.write(json.dumps(image_record(json.loads(line), teacher, hints), ensure_ascii=False) + "\n")
            n += 1
    (OUT / "prompt_vi.md").write_text(PROMPT + "\n", encoding="utf-8")
    print(f"{n} ảnh -> {OUT / 'inputs.jsonl'}; prompt -> {OUT / 'prompt_vi.md'}")


if __name__ == "__main__":
    main()

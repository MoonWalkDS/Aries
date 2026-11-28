"""Chuyển toàn bộ nhãn 5.000 ảnh sang định dạng metadata của Nam (metadata/metadata.json, khoá "HH_FRAME.jpg").

    python metadata/build_metadata.py            # ghi metadata/metadata.json

Mỗi ảnh giữ đúng 3 trường mô tả chính:
  so_nguoi            int, số người detector đếm (num_people)
  danh_sach_nguoi     list[str], mỗi người một câu tiếng Việt: trang phục + hành động + vị trí
  vat_the_trang_thai  list[str], tiếng Việt: vùng cố định của bếp, trạng thái vùng, vật thể
Thêm các trường có cấu trúc, và `goc` = nguyên bản ghi castle_dataset.jsonl, để không mất thông tin nào:
  so_nguoi_thay, ten_nguoi, nguoi[], vat_the[], trang_thai_vung[], caption, mo_ta_thay, thay, goc
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# bố cục main (data/exports, data/dataset) hoặc bố cục cũ (exports, dataset)
EXPORTS = next((p for p in (REPO / "data" / "exports", REPO / "exports") if p.exists()), REPO / "data" / "exports")
DATASET = next((p for p in (REPO / "data" / "dataset", REPO / "dataset") if p.exists()), REPO / "data" / "dataset")
OUT = REPO / "metadata" / "metadata.json"

OBJECT_VI = {
    "bottle": "chai", "bowl": "bát", "cup": "cốc", "cooking pot": "nồi", "tea towel": "khăn lau", "cutting board": "thớt",
    "pineapple": "quả dứa", "plate": "đĩa", "toaster": "máy nướng bánh mì", "frying pan": "chảo", "orange": "quả cam",
    "kettle": "ấm đun nước", "vase": "bình hoa", "banana": "chuối", "knife": "dao", "spoon": "thìa", "microwave": "lò vi sóng",
    "wine glass": "ly rượu", "baking tray": "khay nướng", "food container": "hộp đựng thức ăn", "apple": "quả táo",
    "book": "sách", "fork": "nĩa", "handbag": "túi xách", "cake": "bánh ngọt", "scissors": "kéo", "cell phone": "điện thoại",
    "suitcase": "vali", "dog": "con chó", "laptop": "máy tính xách tay", "sandwich": "bánh mì kẹp", "broccoli": "bông cải xanh",
    "tie": "cà vạt", "frisbee": "đĩa ném", "backpack": "ba lô", "keyboard": "bàn phím", "carrot": "cà rốt", "cat": "con mèo",
    "remote": "điều khiển",
}
# tên ngắn của vùng cố định (camera đứng yên: vùng nào cũng có trong mọi ảnh)
ROI_NOUN = {"stove": "bếp", "oven_towel": "khăn treo lò", "counter_left": "mặt bàn bếp", "sink": "bồn rửa", "island": "đảo bếp",
            "trash": "thùng rác", "fridge": "tủ lạnh", "coffee_corner": "góc pha cà phê", "living_area": "ghế sofa",
            "floor_front": "bàn thấp"}
SUBJECT = {"male": "người đàn ông", "female": "người phụ nữ"}


def person_sentence(p: dict) -> str:
    """Kiểu câu của Nam: "người đàn ông áo sơ mi xanh dương … đứng trước bồn rửa …"."""
    outfit = p.get("outfit") or {}
    gender = outfit.get("gender") or (p.get("attributes") or {}).get("gender")
    desc = (outfit.get("description") or "").strip().rstrip(".")
    for prefix in ("Nam,", "Nam ", "Nữ,", "Nữ "):          # mô tả mở đầu bằng giới tính: thay bằng chủ ngữ
        if desc.startswith(prefix):
            desc = desc[len(prefix):].strip()
            break
    parts = [SUBJECT.get(gender, "người")]
    if desc:
        parts.append(desc[0].lower() + desc[1:])
    elif (p.get("identity") or {}).get("name"):
        parts.append((p["identity"]["name"]).lower())
    action = (p.get("action") or {}).get("label_vi")
    if action:
        parts.append(action)
    if p.get("zone_vi"):
        parts.append("ở " + p["zone_vi"])
    return ", ".join(parts)


def convert(r: dict, caption: str | None = None, teacher: dict | None = None) -> dict:
    objects = Counter(o["label"] for o in r["objects"])
    rois = [(k, v) for k, v in r["rois"].items() if v.get("state")]
    vat_the = list(ROI_NOUN.get(k, v["name_vi"]) for k, v in r["rois"].items())
    vat_the += [v["state"]["label_vi"] for _, v in rois if v["state"].get("label_vi")]
    vat_the += [OBJECT_VI.get(label, label) for label in objects]
    teacher = teacher or {}
    return {
        "so_nguoi": int(r["num_people"]),
        "danh_sach_nguoi": [person_sentence(p) for p in r["people"]],
        "vat_the_trang_thai": list(dict.fromkeys(vat_the)),        # bỏ trùng, giữ thứ tự
        "so_nguoi_thay": teacher.get("so_nguoi"),
        "ten_nguoi": [n for n in ((p.get("identity") or {}).get("name") for p in r["people"]) if n],
        "nguoi": [{"id": p["id"], "ten": (p.get("identity") or {}).get("name"),
                   "gioi_tinh": (p.get("outfit") or {}).get("gender") or (p.get("attributes") or {}).get("gender"),
                   "mo_ta": (p.get("outfit") or {}).get("description"),
                   "hanh_dong": (p.get("action") or {}).get("label_vi"), "vi_tri": p.get("zone_vi"), "box": p["box"]}
                  for p in r["people"]],
        "vat_the": [{"ten": OBJECT_VI.get(k, k), "ten_en": k, "so_luong": n} for k, n in objects.most_common()],
        "trang_thai_vung": [{"vung": v["name_vi"], "trang_thai": v["state"].get("label_vi")} for _, v in rois],
        "caption": caption,
        "mo_ta_thay": teacher.get("description"),
        "thay": teacher.get("row"),
        "goc": r,
    }


def load_teacher(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_id = f'{int(row["hour"]):02d}_{row["frame"]}'   # CSV ghi "8", mã ảnh là "08_…"
            counts = json.loads(row.get("class_count") or "{}")
            out[image_id] = {"so_nguoi": counts.get("person", 0), "description": row.get("description"), "row": row}
    return out


def main() -> None:
    records = [json.loads(line) for line in open(EXPORTS / "castle_dataset.jsonl", encoding="utf-8")]
    captions = {}
    cap_file = EXPORTS / "caption" / "captions.jsonl"
    if cap_file.exists():
        for line in open(cap_file, encoding="utf-8"):
            c = json.loads(line)
            captions[c["image_id"]] = c.get("caption")
    teacher = load_teacher(DATASET / "raw" / "teacher_metadata.csv")
    meta = {f'{r["image_id"]}.jpg': convert(r, captions.get(r["image_id"]), teacher.get(r["image_id"])) for r in records}

    # tự kiểm tra: đủ ảnh, đúng kiểu 3 trường engine của Nam cần
    assert len(meta) == len(records)
    for k, m in meta.items():
        assert isinstance(m["so_nguoi"], int) and m["so_nguoi"] == len(m["danh_sach_nguoi"]), k
        assert all(isinstance(s, str) and s for s in m["danh_sach_nguoi"] + m["vat_the_trang_thai"]), k
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")   # thụt dòng như file mẫu của Nam
    print(f"{len(meta)} ảnh -> {OUT}  (caption: {sum(1 for m in meta.values() if m['caption'])}, "
          f"mô tả thầy: {sum(1 for m in meta.values() if m['mo_ta_thay'])})")


if __name__ == "__main__":
    sys.exit(main())

"""Gom metadata trong ../exports thành thư mục dataset Hugging Face (không kèm ảnh) và tuỳ chọn push.

    python hf_export.py                                  # chỉ dựng ../exports/hf/
    python hf_export.py --push TranKimHieu/castle5000-metadata [--public]

Chạy sau `python main.py enrich merge`. Repo tạo ở chế độ private trừ khi có --public.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT.parent / "exports"
OUT = EXPORTS / "hf"
LABEL_FILES = [ROOT / "labels" / "adl_people_labels.yaml", ROOT / "labels" / "rois.yaml", ROOT / "annotations" / "outfits.yaml"]
TABLES = ("people", "objects", "rois")

CARD = """---
license: mit
language: [en, vi]
pretty_name: Castle5000 metadata
size_categories: [10K<n<100K]
task_categories: [image-to-text, text-to-image]
tags: [lifelog, kitchen, adl, person-attributes, retrieval]
configs:
- config_name: images
  data_files: data/images.jsonl
  default: true
- config_name: people
  data_files: data/people.parquet
- config_name: objects
  data_files: data/objects.parquet
- config_name: rois
  data_files: data/rois.parquet
---

# Castle5000 metadata

Metadata gán nhãn bán tự động cho 5.000 ảnh camera trần cố định trong bếp chung (13 giờ, 08–20, 1280×720).
**Repo này không chứa ảnh.** Ảnh gốc lấy từ nguồn Castle5000; ghép bằng `image_id` (`<giờ>_<số thứ tự>`, ví dụ `08_1002`).

## Thống kê

{stats}

## Cấu hình

| Config | File | Một dòng là |
|---|---|---|
| `images` | `data/images.jsonl` | một ảnh: danh sách người (box, zone, hành động, thuộc tính, danh tính, trang phục), 10 vùng cố định, vật thể, box vẽ tay |
| `people` | `data/people.parquet` | một crop người (bảng phẳng của `images.people`) |
| `objects` | `data/objects.parquet` | một vật thể (RT-DETR + OWLv2, đã bỏ đồ đứng yên) |
| `rois` | `data/rois.parquet` | một lượt (ảnh × vùng): có đổi so với nền không, trạng thái |

`labels/`: bộ nhãn hành động (`adl_people_labels.yaml`), định nghĩa 10 vùng (`rois.yaml`), trang phục chuẩn theo danh tính (`outfits.yaml`).
`summary.json`: thống kê đầy đủ.

```python
from datasets import load_dataset
people = load_dataset("{repo}", "people", split="train")
```

## Trường chính

- Toạ độ `box` / `x1,y1,x2,y2`: pixel ảnh gốc 1280×720.
- `source` của nhãn: `human` (người duyệt), `vlm` (Qwen2.5-VL-7B), `seed` / `suggested` (máy, chưa duyệt). Lọc theo trường này để chọn mức tin cậy.
- `identity`: danh tính **theo trang phục**, không phải nhận diện khuôn mặt. `source`: `person` (người đặt tên), `model` (logistic regression trên SigLIP2 học từ tên người đặt, nhận khi `prob >= 0.60`), `cluster`.
- `attributes`: ngoại hình. `attributes_source = outfit` nghĩa là lấy từ `outfits.yaml` theo danh tính; `vlm` nghĩa là câu trả lời VLM cho riêng crop đó. `holding`, `view` luôn từ VLM.

## Quy trình

RT-DETR dò người → SigLIP2 SO400M embedding → vòng lặp gợi ý theo centroid (LSC-ADL, arXiv:2504.02060) → Qwen2.5-VL-7B duyệt vòng đầu → người duyệt trên web.
Danh tính: người đặt tên tay cho {named_manual} crop, mô hình đoán phần còn lại.

## Hạn chế

- Phần lớn nhãn hành động và trạng thái vùng là `source: vlm`; chưa có tập kiểm chứng gán tay độc lập.
- Danh tính theo trang phục: người đổi đồ thành hai danh tính; áo giống nhau có thể lẫn. Độ chính xác tên mô hình khi bỏ hẳn từng giờ ra kiểm tra: khoảng 0,94.
- `outfits.yaml` điền sẵn từ câu trả lời VLM gặp nhiều nhất của từng người; dòng chưa soát có thể sai.
"""


def build(repo: str) -> Path:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data").mkdir(parents=True)
    (OUT / "labels").mkdir()
    shutil.copy(EXPORTS / "castle_dataset.jsonl", OUT / "data" / "images.jsonl")
    for name in TABLES:
        df = pd.read_csv(EXPORTS / f"{name}.csv", encoding="utf-8-sig", dtype={"image_id": str, "hour": str})
        df.to_parquet(OUT / "data" / f"{name}.parquet", index=False)
    for f in LABEL_FILES:
        shutil.copy(f, OUT / "labels" / f.name)
    shutil.copy(EXPORTS / "summary.json", OUT / "summary.json")
    s = json.loads((EXPORTS / "summary.json").read_text(encoding="utf-8"))
    names = json.loads((ROOT / "annotations" / "identity_names.json").read_text(encoding="utf-8"))["people"]
    stats = "\n".join(f"- {k}: {s[k]:,}".replace(",", ".") for k in
                      ("images", "people", "people_labeled", "people_described", "people_named", "objects", "roi_slots", "roi_labeled"))
    card = CARD.replace("{stats}", stats).replace("{repo}", repo).replace("{named_manual}", f"{sum(1 for v in names.values() if v):,}".replace(",", "."))
    (OUT / "README.md").write_text(card, encoding="utf-8")
    return OUT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", metavar="REPO_ID", help="push lên dataset repo này (vd TranKimHieu/castle5000-metadata)")
    ap.add_argument("--public", action="store_true", help="tạo repo public (mặc định private)")
    args = ap.parse_args()
    out = build(args.push or "<user>/castle5000-metadata")
    print("built", out, sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()))
    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, repo_type="dataset", private=not args.public, exist_ok=True)
        info = api.upload_folder(folder_path=str(out), repo_id=args.push, repo_type="dataset", commit_message="Update Castle5000 metadata")
        print("pushed", info)


if __name__ == "__main__":
    main()

"""Dựng chỉ mục cho search_engine.py (chạy một lần; chạy lại khi caption hoặc nhãn đổi).

    tools/build_index.bat   (hoặc: .venv/Scripts/python.exe retrieval/build_search_index.py)
Đọc data/exports/castle_dataset.jsonl, data/exports/caption/captions.jsonl, data/dataset/raw/teacher_metadata.csv;
ghi data/exports/search_index/:
    images.json                  id ảnh, số người theo detector và theo thầy
    persons.json                 hồ sơ từng người (tiếng Việt, tiếng Anh, tên, giới tính, hành động, box)
    person_vectors.npy           vector SigLIP2 của hồ sơ người tiếng Anh
    scene_sentences.json         câu cảnh của từng ảnh (caption, mô tả thầy, trạng thái vùng, vật thể)
    scene_sentence_vectors.npy   vector SigLIP2 của câu cảnh tiếng Anh
    translations.jsonl           cache bản dịch caption vi -> en
Caption tiếng Việt được dịch sang tiếng Anh một lần ở đây để SigLIP2 (chỉ hiểu tiếng Anh) nhúng được.
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import numpy as np

from common import EXPORTS_DIR, REPO_ROOT
from search_engine import SEARCH_INDEX_DIR, EmbeddingModels

EMPTY_ATTRIBUTE_VALUES = {"", "no", "none", "plain", "unknown", "n/a"}


def english_profile(person: dict) -> str:
    """Hồ sơ tiếng Anh từ attributes (vốn là tiếng Anh) + hành động + vùng."""
    attributes = person.get("attributes") or {}

    def value(key: str) -> str:
        text = str(attributes.get(key) or "").strip()
        return text if text.lower() not in EMPTY_ATTRIBUTE_VALUES else ""

    parts = [value("gender"), f"{value('hair')} hair" if value("hair") else "",
             " ".join(filter(None, [value("top_color"), value("top_pattern"), value("top")])),
             " ".join(filter(None, [value("bottom_color"), value("bottom")])), value("headwear"),
             "glasses" if attributes.get("glasses") == "yes" else "", "apron" if attributes.get("apron") == "yes" else "",
             " ".join(f"holding {h}" for h in attributes.get("holding") or []),
             (person.get("action") or {}).get("label") or "", (person.get("zone") or "").replace("_", " ")]
    return ", ".join(x for x in parts if x)


def vietnamese_profile(person: dict) -> str:
    return ". ".join(filter(None, [(person.get("identity") or {}).get("name"), (person.get("outfit") or {}).get("description"),
                                   (person.get("action") or {}).get("label_vi"), person.get("zone_vi")]))


def split_scene_sentences(text: str) -> list[str]:
    """Cắt theo . ! ? ; : để mỗi đoạn vừa 64 token của SigLIP2."""
    return [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", text or "") if len(s.strip()) > 3]


def load_sources() -> tuple[list[dict], dict, dict]:
    """(bản ghi castle_dataset, caption theo ảnh, (mô tả thầy, số người thầy đếm) theo ảnh)."""
    records = [json.loads(line) for line in open(EXPORTS_DIR / "castle_dataset.jsonl", encoding="utf-8")]
    captions = {}
    for line in open(EXPORTS_DIR / "caption" / "captions.jsonl", encoding="utf-8"):
        row = json.loads(line)
        captions[row["image_id"]] = row["caption"]
    teacher = {}
    with open(REPO_ROOT / "data" / "dataset" / "raw" / "teacher_metadata.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            teacher[row["imageID"].replace("day1_Kitchen_", "")] = (row["description"], json.loads(row["class_count"] or "{}").get("person", 0))
    return records, captions, teacher


def translate_with_cache(texts: set[str], models, cache_file: Path, batch_size: int = 128) -> dict[str, str]:
    """Dịch vi→en, mỗi câu khác nhau một lần. Câu sắp theo độ dài để lô ít phần đệm; greedy (beams=1) cho nhanh.
    Ghi nối vào `cache_file` sau mỗi lô: dừng giữa chừng thì lần sau chạy tiếp."""
    translated = {}
    if cache_file.exists():
        for line in open(cache_file, encoding="utf-8"):
            row = json.loads(line)
            translated[row["vi"]] = row["en"]
    todo = sorted(texts - translated.keys(), key=len)
    started = time.time()
    with open(cache_file, "a", encoding="utf-8") as f:
        for k in range(0, len(todo), batch_size):
            chunk = todo[k:k + batch_size]
            for vi, en in zip(chunk, models.translate(chunk, beams=1)):
                translated[vi] = en
                f.write(json.dumps({"vi": vi, "en": en}, ensure_ascii=False) + "\n")
            f.flush()
            if (k // batch_size) % 10 == 0:
                print(f"dịch {k + len(chunk)}/{len(todo)} câu, {time.time() - started:.0f}s", flush=True)
    return translated


def build_index(records: list[dict], captions: dict, teacher: dict, models, out_dir: Path = SEARCH_INDEX_DIR) -> None:
    """captions: image_id -> caption; teacher: image_id -> (mô tả tiếng Anh, số người thầy đếm)."""
    image_ids = [r["image_id"] for r in records]
    missing = [i for i in image_ids if i not in captions or i not in teacher]
    if missing:
        raise SystemExit(f"Thiếu caption/mô tả thầy cho {len(missing)} ảnh, ví dụ {missing[:5]}")

    persons, scene_sentences, need_translation = [], [], []
    for image_row, record in enumerate(records):
        for person in record["people"]:
            persons.append({"image": image_row, "id": person["id"], "box": person["box"], "tag": (person.get("identity") or {}).get("name"),
                            "gender": (person.get("attributes") or {}).get("gender"),
                            "action": (person.get("action") or {}).get("label"), "vi": vietnamese_profile(person), "en": english_profile(person)})
        for sentence in split_scene_sentences(captions[record["image_id"]]):
            scene_sentences.append({"image": image_row, "src": "caption", "vi": sentence, "en": None})
            need_translation.append(len(scene_sentences) - 1)
        for sentence in split_scene_sentences(teacher[record["image_id"]][0]):
            scene_sentences.append({"image": image_row, "src": "teacher", "vi": "", "en": sentence})
        for roi in record["rois"].values():
            state = roi.get("state") or {}
            if roi.get("changed") and state.get("label_vi"):
                scene_sentences.append({"image": image_row, "src": "roi", "vi": f'{roi["name_vi"]}: {state["label_vi"]}', "en": state.get("label") or ""})
        if record["objects"]:
            scene_sentences.append({"image": image_row, "src": "objects", "vi": "",
                                    "en": "objects: " + ", ".join(sorted({o["label"] for o in record["objects"]}))})

    out_dir.mkdir(parents=True, exist_ok=True)
    translated = translate_with_cache({scene_sentences[j]["vi"] for j in need_translation}, models, out_dir / "translations.jsonl")
    for j in need_translation:
        scene_sentences[j]["en"] = translated[scene_sentences[j]["vi"]]

    np.save(out_dir / "person_vectors.npy", models.encode_many([p["en"] for p in persons]))
    np.save(out_dir / "scene_sentence_vectors.npy", models.encode_many([s["en"] for s in scene_sentences]))
    (out_dir / "persons.json").write_text(json.dumps(persons, ensure_ascii=False), encoding="utf-8")
    (out_dir / "scene_sentences.json").write_text(json.dumps(scene_sentences, ensure_ascii=False), encoding="utf-8")
    images = {"ids": image_ids, "num_people": [r["num_people"] for r in records], "teacher_people": [teacher[i][1] for i in image_ids]}
    (out_dir / "images.json").write_text(json.dumps(images, ensure_ascii=False), encoding="utf-8")
    print(f"Xong: {len(persons)} người, {len(scene_sentences)} câu cảnh -> {out_dir}")


def main() -> None:
    build_index(*load_sources(), EmbeddingModels())


if __name__ == "__main__":
    main()

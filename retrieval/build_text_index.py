"""Dựng chỉ mục text cho text_engine.py (chạy một lần; chạy lại khi caption hoặc nhãn đổi).

    build_index.bat     (hoặc: .venv/Scripts/python.exe retrieval/build_text_index.py)
Đọc data/exports/castle_dataset.jsonl, data/exports/caption/captions.jsonl, data/dataset/raw/teacher_metadata.csv;
ghi data/exports/text_index/ (meta.json, persons.json, persons_en.npy, scene.json, scene_en.npy).
Caption tiếng Việt được dịch sang tiếng Anh một lần ở đây để SigLIP2 (chỉ hiểu tiếng Anh) nhúng được.
"""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import numpy as np

from engine import EXPORTS, ROOT
from text_engine import INDEX, Models

SKIP = {"", "no", "none", "plain", "unknown", "n/a"}


def person_en(p: dict) -> str:
    """Hồ sơ tiếng Anh từ attributes (vốn là tiếng Anh) + hành động + vùng."""
    a = p.get("attributes") or {}
    v = lambda k: str(a.get(k) or "").strip() if str(a.get(k) or "").strip().lower() not in SKIP else ""  # noqa: E731
    parts = [v("gender"), f"{v('hair')} hair" if v("hair") else "",
             " ".join(filter(None, [v("top_color"), v("top_pattern"), v("top")])),
             " ".join(filter(None, [v("bottom_color"), v("bottom")])), v("headwear"),
             "glasses" if a.get("glasses") == "yes" else "", "apron" if a.get("apron") == "yes" else "",
             " ".join(f"holding {h}" for h in a.get("holding") or []),
             (p.get("action") or {}).get("label") or "", (p.get("zone") or "").replace("_", " ")]
    return ", ".join(x for x in parts if x)


def person_vi(p: dict) -> str:
    return ". ".join(filter(None, [(p.get("identity") or {}).get("name"), (p.get("outfit") or {}).get("description"),
                                   (p.get("action") or {}).get("label_vi"), p.get("zone_vi")]))


def sentences(text: str) -> list[str]:
    """Cắt theo . ! ? ; : để mỗi đoạn vừa 64 token của SigLIP2."""
    return [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", text or "") if len(s.strip()) > 3]


def load_sources() -> tuple[list[dict], dict, dict]:
    records = [json.loads(line) for line in open(EXPORTS / "castle_dataset.jsonl", encoding="utf-8")]
    captions = {}
    for line in open(EXPORTS / "caption" / "captions.jsonl", encoding="utf-8"):
        r = json.loads(line)
        captions[r["image_id"]] = r["caption"]
    teacher = {}
    with open(ROOT / "data" / "dataset" / "raw" / "teacher_metadata.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            teacher[r["imageID"].replace("day1_Kitchen_", "")] = (r["description"], json.loads(r["class_count"] or "{}").get("person", 0))
    return records, captions, teacher


def translate_cached(texts: set[str], models, cache: Path, batch: int = 128) -> dict[str, str]:
    """Dịch vi→en, mỗi câu khác nhau một lần. Câu sắp theo độ dài để lô ít phần đệm; greedy (beams=1) cho nhanh.
    Ghi nối vào `cache` sau mỗi lô: dừng giữa chừng thì lần sau chạy tiếp."""
    done = {}
    if cache.exists():
        for line in open(cache, encoding="utf-8"):
            r = json.loads(line)
            done[r["vi"]] = r["en"]
    todo = sorted(texts - done.keys(), key=len)
    t = time.time()
    with open(cache, "a", encoding="utf-8") as f:
        for k in range(0, len(todo), batch):
            chunk = todo[k:k + batch]
            for vi, en in zip(chunk, models.translate(chunk, beams=1)):
                done[vi] = en
                f.write(json.dumps({"vi": vi, "en": en}, ensure_ascii=False) + "\n")
            f.flush()
            if (k // batch) % 10 == 0:
                print(f"dịch {k + len(chunk)}/{len(todo)} câu, {time.time() - t:.0f}s", flush=True)
    return done


def build(records: list[dict], captions: dict, teacher: dict, models, out: Path = INDEX) -> None:
    """captions: image_id -> caption; teacher: image_id -> (mô tả tiếng Anh, số người thầy đếm)."""
    ids = [r["image_id"] for r in records]
    missing = [i for i in ids if i not in captions or i not in teacher]
    if missing:
        raise SystemExit(f"Thiếu caption/mô tả thầy cho {len(missing)} ảnh, ví dụ {missing[:5]}")

    persons, scene, vi_to_translate = [], [], []
    for row, r in enumerate(records):
        for p in r["people"]:
            persons.append({"image": row, "id": p["id"], "box": p["box"], "tag": (p.get("identity") or {}).get("name"),
                            "gender": (p.get("attributes") or {}).get("gender"),
                            "action": (p.get("action") or {}).get("label"), "vi": person_vi(p), "en": person_en(p)})
        for s in sentences(captions[r["image_id"]]):
            scene.append({"image": row, "src": "caption", "vi": s, "en": None})
            vi_to_translate.append(len(scene) - 1)
        for s in sentences(teacher[r["image_id"]][0]):
            scene.append({"image": row, "src": "teacher", "vi": "", "en": s})
        for roi in r["rois"].values():
            st = roi.get("state") or {}
            if roi.get("changed") and st.get("label_vi"):
                scene.append({"image": row, "src": "roi", "vi": f'{roi["name_vi"]}: {st["label_vi"]}', "en": st.get("label") or ""})
        if r["objects"]:
            scene.append({"image": row, "src": "objects", "vi": "",
                          "en": "objects: " + ", ".join(sorted({o["label"] for o in r["objects"]}))})

    out.mkdir(parents=True, exist_ok=True)
    done = translate_cached({scene[j]["vi"] for j in vi_to_translate}, models, out / "translations.jsonl")
    for j in vi_to_translate:
        scene[j]["en"] = done[scene[j]["vi"]]

    np.save(out / "persons_en.npy", models.encode_many([p["en"] for p in persons]))
    np.save(out / "scene_en.npy", models.encode_many([s["en"] for s in scene]))
    (out / "persons.json").write_text(json.dumps(persons, ensure_ascii=False), encoding="utf-8")
    (out / "scene.json").write_text(json.dumps(scene, ensure_ascii=False), encoding="utf-8")
    meta = {"ids": ids, "num_people": [r["num_people"] for r in records], "teacher_people": [teacher[i][1] for i in ids]}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"Xong: {len(persons)} người, {len(scene)} câu cảnh -> {out}")


def main() -> None:
    build(*load_sources(), Models())


if __name__ == "__main__":
    main()

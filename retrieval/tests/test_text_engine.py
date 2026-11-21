"""Chạy cả luồng TextEngine trên chỉ mục mini (40 ảnh thật) với model giả: không cần GPU, không tải model."""
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_text_index import build, load_sources  # noqa: E402
from text_engine import TextEngine  # noqa: E402
from textsearch import fold  # noqa: E402


class FakeModels:
    """Dịch = giữ nguyên; nhúng = túi từ băm vào 256 chiều (câu chung chữ thì gần nhau)."""
    device = "cpu"

    def translate(self, texts, beams=None):
        return list(texts)

    def encode(self, texts):
        v = torch.zeros(len(texts), 256)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", fold(t)):
                v[i, zlib.crc32(w.encode()) % 256] += 1
        return torch.nn.functional.normalize(v + 1e-6, dim=-1)

    def encode_many(self, texts):
        return self.encode(texts).numpy().astype(np.float16)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    from engine import EXPORTS
    if not (EXPORTS / "castle_dataset.jsonl").exists():
        pytest.skip("chưa có data/ (chạy download_data.bat)")
    records, captions, teacher = load_sources()
    records = [r for r in records if r["people"]][:40]
    out = tmp_path_factory.mktemp("text_index")
    build(records, captions, teacher, FakeModels(), out)
    e = TextEngine(device="cpu", index=out, models=FakeModels(), image_emb=None)
    e.captions, e.index = captions, out
    return e


def test_empty_query(engine):
    assert engine.search("   ")["results"] == []


def test_caption_sentence_finds_its_image(engine):
    target = engine.ids[7]
    first = re.split(r"(?<=[.!?;:])\s+", engine.captions[target])[1]
    assert engine.rank_of(first, {target}) <= 8   # 40 khung liền nhau có cùng người: nhiều ảnh khớp gần như nhau


def test_multi_person_query_layers_and_explain(engine):
    r = engine.search("Một người đàn ông mở cửa tủ lạnh. Anh ấy mặc áo len. Những người khác đang rửa chén", top=10)
    assert r["layers"][0] == 40 and 0 < r["layers"][2] <= r["layers"][1] <= 40
    assert len(r["results"]) == 10 and len({x["id"] for x in r["results"]}) == 10
    top = r["results"][0]["explain"]
    assert len(top["pairs"]) == 1 and set(top["parts"]) == {"count", "name", "scene", "image", "person", "final"}
    assert all(x["id"] in engine.records for x in r["results"])


def test_rank_of_covers_every_image(engine):
    assert len(engine.search("bếp", top=10_000)["results"]) == 40


def test_named_clause_pairs_with_that_person(engine):
    r = engine.search("Một người đàn ông mở cửa tủ lạnh. Anh ấy mặc áo len. Thầy Linh mặc áo len xanh.", top=40)
    with_linh = [x for x in r["results"] if "Linh" in engine.names_of[engine.row[x["id"]]]]
    assert with_linh and all(x["explain"]["pairs"][1]["tag"] == "Linh" for x in with_linh)


def test_image_channel_maps_rows_by_id(engine, tmp_path):
    """Embedding ảnh lưu theo thứ tự khác chỉ mục text (và có id thừa): ảnh mang vector của đề phải lên đầu."""
    import json
    fake, q = FakeModels(), "zzz quả bóng tím"
    target = engine.ids[11]
    ids = list(reversed(engine.ids)) + ["99_0001"]
    vec = fake.encode([q]).numpy()[0]
    imgs = np.random.default_rng(0).normal(size=(len(ids), 256)).astype(np.float32) * 0.01
    imgs[ids.index(target)] = vec * 5
    pids = [p["id"] for p in engine.persons][::-1]
    json.dump([f"{i}.jpg" for i in ids], open(tmp_path / "image_ids.json", "w"))
    json.dump([f"{i}.jpg" for i in pids], open(tmp_path / "people_ids.json", "w"))
    for f in ("images.npy", "images_global.npy"):
        np.save(tmp_path / f, imgs)
    np.save(tmp_path / "people.npy", np.random.default_rng(1).normal(size=(len(pids), 256)).astype(np.float32))
    e = TextEngine(device="cpu", index=engine.index, models=fake, image_emb=tmp_path)
    assert e.search(q, top=1)["results"][0]["id"] == target


def test_explicit_count_dominates(engine):
    """Đề nêu rõ số người: ảnh khớp số người (theo detector hoặc thầy) phải đứng trên ảnh lệch."""
    nums = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười"]
    fits = lambda c: sum(c in (engine.count_det[i], engine.count_teacher[i]) for i in range(len(engine.ids)))  # noqa: E731
    target = next(c for c in range(2, 11) if 3 <= fits(c) <= len(engine.ids) // 2)   # đủ ảnh khớp nhưng không phải đa số
    top = engine.search(f"Ảnh bếp có {nums[target]} người, trên đảo bếp có nhiều nồi inox, hai quả dứa, đĩa trắng và cốc sứ.", top=3)["results"]
    assert all(target in (engine.count_det[engine.row[x["id"]]], engine.count_teacher[engine.row[x["id"]]]) for x in top)

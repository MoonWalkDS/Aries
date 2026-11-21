"""Test phần xử lý ảnh của stage adl: trừ nền, dựng embedding, quy tắc bếp trống, trang duyệt, lệnh round.

Không nạp model thật: encoder được thay bằng bản giả.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as pipeline_main  # noqa: E402
from modules import adl, adl_embed  # noqa: E402
from modules.adl_review import write_review_page  # noqa: E402
from modules.utils import load_config, write_json  # noqa: E402

CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


def unit(seed: int, dim: int = 64) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- trừ nền
def test_focus_box_finds_moving_region_and_ignores_noise():
    background = np.full((90, 160), 100, dtype=np.float32)
    frame = background.copy()
    frame[40:72, 80:112] = 220                       # một "người" 32x32
    frame[3, 3] = 255                                # một pixel nhiễu lẻ không được kéo hộp ra
    mask = adl_embed.motion_mask(frame, background, threshold=25)
    box, fraction = adl_embed.focus_box(mask, min_fraction=0.004, pad=0.0)
    assert box == (80 / 160, 40 / 90, 112 / 160, 72 / 90)
    assert abs(fraction - (32 * 32 + 1) / (90 * 160)) < 1e-6
    padded, _ = adl_embed.focus_box(mask, 0.004, pad=0.5)
    assert padded[0] < box[0] and padded[3] > box[3] and all(0.0 <= v <= 1.0 for v in padded)
    # ảnh trống: không có hộp, chuyển động ~ 0
    assert adl_embed.focus_box(adl_embed.motion_mask(background, background, 25), 0.004, 0.15) == (None, 0.0)


def test_background_is_median_so_a_passing_person_disappears():
    frames = np.full((9, 4, 4), 50, dtype=np.float32)
    frames[2, 1, 1] = frames[5, 1, 1] = 250          # người che pixel này trong 2/9 khung
    assert (adl_embed.background_of(frames) == 50).all()


def _write_scene(path: Path, person_at: tuple[int, int] | None) -> None:
    image = Image.new("RGB", (320, 180), (120, 110, 100))
    if person_at:
        x, y = person_at
        ImageDraw.Draw(image).rectangle((x, y, x + 60, y + 80), fill=(250, 250, 250))
    image.save(path, quality=95)


class FakeEncoder:
    """Vector phụ thuộc màu trung bình của ảnh: ảnh cắt vào người khác hẳn ảnh toàn cảnh."""

    device = "fake"

    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []

    def encode(self, images):
        self.sizes += [im.size for im in images]
        rows = [[np.asarray(im, dtype=np.float32).mean() / 255.0, 1.0, im.size[0] / 320.0] for im in images]
        rows = np.asarray(rows, dtype=np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _make_images(tmp_path: Path) -> Path:
    images = tmp_path / "images"
    images.mkdir()
    for n in range(12):                               # nhóm giờ 08: phần lớn trống, 3 ảnh có người
        _write_scene(images / f"08_{n}.jpg", (40 + 60 * (n - 9), 50) if n >= 9 else None)
    return images


def test_run_embed_writes_combined_vectors_motion_and_boxes(tmp_path: Path):
    images = _make_images(tmp_path)
    cfg = load_config(CONFIG, input_images=str(images), output_root=str(tmp_path / "out"))
    encoder, out = FakeEncoder(), tmp_path / "emb"
    paths = sorted(images.glob("*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
    stats = adl_embed.run_embed(cfg, paths, out, encoder=encoder)
    emb, motion = np.load(out / "embeddings.npy"), np.load(out / "motion.npy")
    boxes = json.loads((out / "boxes.json").read_text(encoding="utf-8"))
    assert emb.shape == (12, 3) and np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    assert stats["with_focus_box"] == 3 and stats["model"] == cfg["adl"]["clip"]["model"]
    assert (motion[:9] < 0.002).all() and (motion[9:] > 0.05).all()
    assert boxes["08_0"] == [0.0, 0.0, 1.0, 1.0] and boxes["08_9"][2] - boxes["08_9"][0] < 0.5
    assert json.loads((out / "ids.json").read_text(encoding="utf-8"))[0] == "08_0.jpg"
    # ảnh có người được mã hoá thêm một lần trên vùng cắt nhỏ hơn cả khung
    assert len(encoder.sizes) == 15 and sum(1 for s in encoder.sizes if s != (320, 180)) == 3
    # vector cuối của ảnh có người khác vector toàn cảnh; ảnh trống thì giữ nguyên
    global_emb = np.load(out / "embeddings_global.npy")
    assert np.allclose(emb[0], global_emb[0], atol=1e-5) and not np.allclose(emb[9], global_emb[9], atol=1e-3)


def test_center_removes_shared_background_component():
    shared = unit(1)
    emb = adl.l2_normalize(np.stack([shared + 0.15 * unit(10 + k // 5) + 0.01 * unit(100 + k) for k in range(10)]))
    assert (emb @ emb.T).min() > 0.9                  # trước: mọi ảnh gần như giống nhau
    centered = adl.center(emb)
    sims = centered @ centered.T
    assert sims[:5, :5].min() > 0.8 and sims[:5, 5:].max() < -0.8   # sau: hai nhóm tách hẳn


# --------------------------------------------------------------------------- quy tắc bếp trống
def test_empty_rule_forces_label_and_blocks_it_elsewhere():
    emb = adl.l2_normalize(np.stack([unit(1), unit(1) + 0.01 * unit(2), unit(3), unit(3) + 0.01 * unit(4), unit(1) + 0.02 * unit(5)]))
    ids = ["a", "b", "c", "d", "e"]
    approved = {"a": {"label": "empty"}, "c": {"label": "cooking"}}
    centroids = adl.build_centroids(approved, ids, emb, 5, 2)
    is_empty = np.array([True, False, False, False, True])
    items = adl.suggest_labels(ids, emb, centroids, set(approved), {}, tau_sim=0.5, empty_class="empty", is_empty=is_empty)
    got = {s["image_id"]: (s["label"], s["score"]) for s in items}
    assert got["e"] == ("empty", 1.0)                 # không chuyển động: gán thẳng
    assert got["d"][0] == "cooking"
    assert "b" not in got                             # giống ảnh trống nhưng có chuyển động: không được gợi ý "empty"
    # đã bị từ chối thì không ép lại
    again = adl.suggest_labels(ids, emb, centroids, set(approved), {"e": ["empty"]}, 0.5, empty_class="empty", is_empty=is_empty)
    assert "e" not in {s["image_id"] for s in again}
    # hạt giống: ảnh trống không được vào lớp khác
    seeds = adl.select_seeds(emb, {"cooking": np.stack([unit(1)])}, per_description=2, max_per_class=5, excluded=is_empty)
    assert set(seeds) == {1, 2} or set(seeds) == {1, 3}


# --------------------------------------------------------------------------- trang duyệt
def test_review_page_embeds_items_and_escapes_script(tmp_path: Path):
    images, thumbs = tmp_path / "images", tmp_path / "thumbs"
    images.mkdir(), thumbs.mkdir()
    (thumbs / "08_1.jpg").write_bytes(b"x")
    items = [{"image_id": "08_1", "label": "eating", "score": 0.91}, {"image_id": "08_2", "label": "eating </script>", "score": None}]
    page = write_review_page(tmp_path / "adl" / "review.html", items, [("eating", "ăn"), ("eating </script>", "")], "(từ chối)", "k1", images, thumbs, {"08_1": ".jpg", "08_2": ".png"})
    html = page.read_text(encoding="utf-8")
    assert html.count("</script>") == 1               # chỉ thẻ đóng thật, dữ liệu đã được thoát
    data = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0].replace("<\\/", "</"))
    assert data["items"][0]["thumb"] == "../thumbs/08_1.jpg" and data["items"][0]["full"] == "../images/08_1.jpg"
    assert data["items"][1]["thumb"] == "../images/08_2.png"          # thiếu thumbnail: dùng ảnh gốc
    assert data["labels"][-1] == "(từ chối)" and data["names"]["eating"] == "ăn"


# --------------------------------------------------------------------------- CLI: embed -> init -> round
def test_cli_embed_init_round_with_html_review(tmp_path: Path, monkeypatch):
    images = _make_images(tmp_path)
    labels = tmp_path / "labels.yaml"
    labels.write_text(
        "classes:\n  - {name: empty kitchen, name_vi: bếp trống, descriptions: [d1]}\n  - {name: cooking, name_vi: nấu ăn, descriptions: [d2]}\n",
        encoding="utf-8",
    )
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data = tmp_path / "data"
    raw["adl"].update(labels_file=str(labels), embeddings=str(data / "embeddings.npy"), ids=str(data / "ids.json"), center=False)
    raw["adl"]["embed"].update(out_dir=str(data))
    raw["adl"]["seed"].update(per_description=1, max_per_class=2, min_similarity=-1.0)
    raw["adl"]["suggest"].update(tau_sim=0.5, max_per_round=None)
    raw["adl"]["review"] = {"thumbs_dir": None}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(adl_embed, "ClipImageEncoder", lambda *a, **k: FakeEncoder())
    monkeypatch.setattr(adl.ClipTextEncoder, "_encode_with_model", lambda self, texts: np.stack([adl.l2_normalize(np.array([1.0, 1.0, 0.3], dtype=np.float32)) for _ in texts]))
    out = tmp_path / "out"
    run = lambda *a: pipeline_main.main(["adl", *a, "--config", str(config), "--input-images", str(images), "--output-root", str(out)])  # noqa: E731
    store = adl.AdlStore(out / "output_jsons" / "adl")

    assert run("embed") == 0 and (data / "motion.npy").exists()
    assert run("init") == 0
    approved = store.approved()
    assert sum(1 for r in approved.values() if r["label"] == "empty kitchen") == 2     # từ trừ nền, tối đa max_per_class
    cooking = [i for i, r in approved.items() if r["label"] == "cooking"]
    assert len(cooking) == 1 and int(cooking[0].split("_")[1]) >= 9                    # hạt giống cooking phải là ảnh có người

    assert run("round") == 0
    suggestions = store.suggestions()["items"]
    assert {s["label"] for s in suggestions if int(s["image_id"].split("_")[1]) < 9} == {"empty kitchen"}
    assert {s["label"] for s in suggestions if int(s["image_id"].split("_")[1]) >= 9} == {"cooking"}
    page = store.root / "review_suggestions.html"
    assert page.exists() and "suggestions_round1" in page.read_text(encoding="utf-8")

    # file do trang duyệt tải về có cùng định dạng với export Label Studio
    reviewed = [
        {"data": {"image_id": s["image_id"], "suggested": s["label"], "score": s["score"]},
         "annotations": [{"updated_at": "t", "result": [{"from_name": "activity", "value": {"choices": [s["label"]]}}]}]}
        for s in suggestions
    ]
    write_json(tmp_path / "adl_review.json", reviewed)
    assert run("import", str(tmp_path / "adl_review.json")) == 0
    assert run("round") == 0 and store.suggestions()["items"] == []                    # mọi ảnh đã có nhãn
    assert run("apply") == 0
    final = json.loads(store.labels_path.read_text(encoding="utf-8"))
    assert len(final) == len(suggestions) and all(v["source"] == "human" for v in final.values())

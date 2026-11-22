"""Test tầng người + tầng vùng cố định (modules/enrich.py) và profile của stage adl.

Không nạp model thật: bộ dò người và bộ mã hoá được thay bằng bản giả.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as pipeline_main  # noqa: E402
from modules import adl, enrich  # noqa: E402
from modules.utils import load_config  # noqa: E402

CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"

ROIS_YAML = """
rois:
  - name: stove
    name_vi: bếp
    box: [0.50, 0.10, 0.90, 0.60]
    zone: [0.45, 0.05, 0.95, 0.95]
    states:
      - {name: stove idle, name_vi: bếp trống, descriptions: [idle]}
      - {name: pot on stove, name_vi: có nồi, descriptions: [pot]}
  - name: door
    name_vi: cửa
    box: [0.02, 0.10, 0.30, 0.60]
    zone: [0.00, 0.05, 0.35, 0.95]
    states:
      - {name: door closed, name_vi: cửa đóng, descriptions: [closed]}
"""


# --------------------------------------------------------------------------- logic thuần
def test_roi_vlm_hint_is_added_to_viewpoint(tmp_path: Path):
    rois = tmp_path / "rois.yaml"
    rois.write_text(ROIS_YAML.replace("  - name: door\n", "  - name: door\n    vlm_hint: The door itself is hidden.\n"), encoding="utf-8")
    cfg = load_config(CONFIG)
    cfg = {**cfg, "enrich": {**cfg["enrich"], "rois_file": str(rois)}}
    door, _ = adl.apply_profile(cfg, "roi-door")
    stove, _ = adl.apply_profile(cfg, "roi-stove")
    assert door["adl"]["viewpoint"].endswith("(door) of a shared kitchen seen from a ceiling camera. The door itself is hidden")
    assert stove["adl"]["viewpoint"].endswith("seen from a ceiling camera")


def test_real_rois_file_is_valid():
    rois = enrich.load_rois(CONFIG.parent / "label_definitions" / "rois.yaml")
    assert len({r.name for r in rois}) == len(rois) >= 6
    assert all(r.states and all(s.get("descriptions") and s.get("name_vi") for s in r.states) for r in rois)
    assert adl.load_labels(CONFIG.parent / "label_definitions" / "adl_people_labels.yaml")


def test_load_rois_rejects_bad_boxes(tmp_path: Path):
    path = tmp_path / "rois.yaml"
    path.write_text("rois:\n  - {name: a, box: [0.5, 0.1, 0.4, 0.9]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="x1 < x2"):
        enrich.load_rois(path)
    path.write_text("rois:\n  - {name: 'a/b', box: [0.1, 0.1, 0.4, 0.9]}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="ký tự lạ"):
        enrich.load_rois(path)


def test_assign_zone_picks_largest_overlap(tmp_path: Path):
    path = tmp_path / "rois.yaml"
    path.write_text(ROIS_YAML, encoding="utf-8")
    rois = enrich.load_rois(path)
    assert enrich.assign_zone([700, 100, 900, 500], 1000, 1000, rois, 0.35) == ("stove", 1.0)
    assert enrich.assign_zone([50, 100, 250, 500], 1000, 1000, rois, 0.35) == ("door", 1.0)
    zone, overlap = enrich.assign_zone([300, 100, 500, 500], 1000, 1000, rois, 0.35)      # nằm giữa hai zone
    assert zone is None and overlap == 0.25
    assert enrich.assign_zone([380, 990, 420, 1000], 1000, 1000, rois, 0.35)[0] is None  # ngoài mọi zone


def test_suppress_duplicates_keeps_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 100, 100], [200, 200, 300, 300]], dtype=np.float32)
    assert enrich.suppress_duplicates(boxes, np.array([0.6, 0.9, 0.5]), 0.7) == [1, 2]
    assert enrich.suppress_duplicates(boxes, np.array([0.6, 0.9, 0.5]), 0.95) == [1, 0, 2]


def test_apply_profile_overrides_adl_section():
    cfg = load_config(CONFIG)
    same, name = adl.apply_profile(cfg, None)
    assert same is cfg and name == "adl"
    people, name = adl.apply_profile(cfg, "people")
    assert name == "adl_people" and people["adl"]["ids"].endswith("people/ids.json")
    assert people["adl"]["clip"] == cfg["adl"]["clip"] and people["adl"]["empty"]["class"] is None
    assert cfg["adl"]["ids"] != people["adl"]["ids"]                    # config gốc không bị sửa
    roi, name = adl.apply_profile(cfg, "roi-stove")
    assert name == "adl_roi-stove" and roi["adl"]["embeddings"] == "../data/intermediate/enrich/rois/stove/embeddings.npy"
    assert roi["adl"]["review"]["images_dir"] == "../data/intermediate/enrich/rois/stove/crops"
    with pytest.raises(ValueError, match="roi-<tên vùng>"):
        adl.apply_profile(cfg, "khong-co")


# --------------------------------------------------------------------------- build + adl profile + merge
class FakeDetector:
    """Coi mọi hình chữ nhật trắng là một người."""

    def detect(self, images):
        out = []
        for image in images:
            mask = np.asarray(image.convert("L")) > 240
            if not mask.any():
                out.append((np.zeros((0, 4), np.float32), np.zeros(0, np.float32)))
                continue
            ys, xs = np.where(mask)
            box = np.array([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=np.float32)
            out.append((np.concatenate([box, box + 1]), np.array([0.9, 0.5], dtype=np.float32)))   # box thứ hai là bản trùng
        return out


class FakeEncoder:
    device = "fake"

    def encode(self, images):
        rows = []
        for image in images:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            rows.append([pixels[..., 0].mean(), pixels[..., 1].mean(), pixels[..., 2].mean(), 0.2])
        rows = np.asarray(rows, dtype=np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _scene(path: Path, person_x: int | None, pot: bool) -> None:
    image = Image.new("RGB", (400, 200), (90, 90, 90))
    draw = ImageDraw.Draw(image)
    if pot:
        draw.rectangle((230, 40, 330, 100), fill=(240, 200, 40))       # nồi vàng trong vùng stove
    if person_x is not None:
        draw.rectangle((person_x, 60, person_x + 50, 180), fill=(255, 255, 255))
    image.save(path, quality=95)


@pytest.fixture()
def project(tmp_path: Path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    # giờ 08: 8 ảnh trống, 2 ảnh có nồi, 2 ảnh có người ở zone door, 2 ảnh có người ở zone stove
    for n in range(14):
        _scene(images / f"08_{n}.jpg", person_x=(20 if n in (10, 11) else 250 if n in (12, 13) else None), pot=n in (8, 9))
    (tmp_path / "rois.yaml").write_text(ROIS_YAML, encoding="utf-8")
    people_labels = tmp_path / "people_labels.yaml"
    people_labels.write_text("classes:\n  - {name: standing, name_vi: đứng, descriptions: [white]}\n", encoding="utf-8")
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    data = tmp_path / "data"
    raw["enrich"].update(out_dir=str(data), rois_file=str(tmp_path / "rois.yaml"), shard_size=5, batch_size=3)
    raw["enrich"]["person"].update(min_area_frac=0.01)
    raw["adl"]["center"] = False
    people = raw["adl_profiles"]["people"]
    people.update(labels_file=str(people_labels), embeddings=str(data / "people" / "embeddings.npy"), ids=str(data / "people" / "ids.json"))
    people["seed"].update(per_description=2, max_per_class=2, min_similarity=-1.0)
    people["suggest"].update(tau_sim=0.5, max_per_round=None)
    people["review"] = {"images_dir": str(data / "people" / "crops"), "thumbs_dir": None}
    roi = raw["adl_profiles"]["roi"]
    for key in ("labels_file", "embeddings", "ids"):
        roi[key] = roi[key].replace("../data/intermediate/enrich", str(data).replace("\\", "/"))
    roi["review"] = {"images_dir": str(data).replace("\\", "/") + "/rois/{roi}/crops", "thumbs_dir": None}
    roi["seed"].update(per_description=1, max_per_class=1, min_similarity=-1.0)
    roi["suggest"].update(tau_sim=0.5, max_per_round=None)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(enrich, "PersonDetector", lambda *a, **k: FakeDetector())
    from modules import adl_embed

    monkeypatch.setattr(adl_embed, "ClipImageEncoder", lambda *a, **k: FakeEncoder())
    text = {"white": [1, 1, 1, 0.2], "pot": [1, 0.8, 0.2, 0.2], "idle": [0.5, 0.5, 0.5, 0.2], "closed": [0.5, 0.5, 0.5, 0.2]}
    monkeypatch.setattr(adl.ClipTextEncoder, "_encode_with_model", lambda self, texts: adl.l2_normalize(np.array([text[t] for t in texts], dtype=np.float32)))
    out = tmp_path / "out"

    def run(stage: str, *args: str) -> int:
        return pipeline_main.main([stage, *args, "--config", str(config), "--input-images", str(images), "--output-root", str(out)])

    return run, data, out / "output_jsons"


def test_enrich_build_people_rois_and_resume(project):
    run, data, _ = project
    assert run("enrich", "build") == 0
    people = json.loads((data / "people" / "people.json").read_text(encoding="utf-8"))
    assert sorted(people) == ["08_10_p01", "08_11_p01", "08_12_p01", "08_13_p01"]      # box trùng đã bị loại
    assert people["08_10_p01"]["zone"] == "door" and people["08_12_p01"]["zone"] == "stove"
    assert np.load(data / "people" / "embeddings.npy").shape == (4, 4)
    assert (data / "people" / "crops" / "08_12_p01.jpg").exists()

    stove_ids = json.loads((data / "rois" / "stove" / "ids.json").read_text(encoding="utf-8"))
    # vùng stove đổi ở 2 ảnh có nồi + 2 ảnh có người đứng trước; cộng một ảnh nền của giờ 08
    assert sorted(stove_ids) == ["08_12.jpg", "08_13.jpg", "08_8.jpg", "08_9.jpg", "bg-08.jpg"]
    assert sorted(json.loads((data / "rois" / "door" / "ids.json").read_text(encoding="utf-8"))) == ["08_10.jpg", "08_11.jpg", "bg-08.jpg"]
    motion = np.load(data / "rois" / "motion.npy")
    row = {image_id: n for n, image_id in enumerate(json.loads((data / "rois" / "image_ids.json").read_text(encoding="utf-8")))}
    assert motion.shape == (14, 2) and motion[row["08_0"]].max() < 0.01
    assert motion[row["08_8"], 0] > 0.1 and motion[row["08_8"], 1] < 0.01          # nồi chỉ làm đổi vùng stove
    assert "pot on stove" in (data / "rois" / "stove" / "labels.yaml").read_text(encoding="utf-8")

    # chạy lại: mọi phần đã lưu nên không dò, không mã hoá gì thêm
    class Boom:
        def detect(self, images):
            raise AssertionError("không được dò lại")

        def encode(self, images):
            raise AssertionError("không được mã hoá lại")

    cfg = load_config(next(data.parent.glob("config.yaml")), input_images=str(data.parent / "images"))
    paths = sorted((data.parent / "images").glob("*.jpg"))
    again = enrich.run_build(cfg, paths, data, detector=Boom(), encoder=Boom())
    assert again["people"] == 4 and again["roi_crops"] == {"stove": 5, "door": 3}


def test_rebuild_only_one_roi_after_moving_its_box(project):
    run, data, _ = project
    assert run("enrich", "build") == 0
    stove_before = np.load(data / "rois" / "stove" / "embeddings.npy")
    people_before = np.load(data / "people" / "embeddings.npy")
    rois_file = data.parent / "rois.yaml"
    rois_file.write_text(ROIS_YAML.replace("box: [0.02, 0.10, 0.30, 0.60]", "box: [0.40, 0.10, 0.70, 0.60]"), encoding="utf-8")
    cfg = load_config(next(data.parent.glob("config.yaml")), input_images=str(data.parent / "images"))
    paths = sorted((data.parent / "images").glob("*.jpg"))

    # build thường thấy rois.yaml đổi; --rois chỉ cắt lại vùng door, không dò người, không đụng vùng khác
    class Counting(FakeEncoder):
        calls = 0

        def encode(self, images):
            Counting.calls += len(images)
            return super().encode(images)

    class Boom:
        def detect(self, images):
            raise AssertionError("không được dò lại")

    with pytest.raises(ValueError):
        enrich.run_build(cfg, paths, data, detector=Boom(), encoder=Counting(), only_rois=["nope"])
    result = enrich.run_build(cfg, paths, data, detector=Boom(), encoder=Counting(), only_rois=["door"])
    door_ids = sorted(json.loads((data / "rois" / "door" / "ids.json").read_text(encoding="utf-8")))
    # khung mới phủ chỗ nồi và chỗ người đứng ở bếp, không còn phủ người đứng ở cửa
    assert door_ids == ["08_12.jpg", "08_13.jpg", "08_8.jpg", "08_9.jpg", "bg-08.jpg"]
    assert Counting.calls == 5 and result["roi_crops"] == {"stove": 5, "door": 5}
    assert sorted(p.name for p in (data / "rois" / "door" / "crops").glob("*.jpg")) == door_ids
    np.testing.assert_array_equal(np.load(data / "rois" / "stove" / "embeddings.npy"), stove_before)
    np.testing.assert_array_equal(np.load(data / "people" / "embeddings.npy"), people_before)
    # lần build thường sau đó dùng lại mọi phần đã lưu
    class NoEncode(FakeEncoder):
        def encode(self, images):
            raise AssertionError("không được mã hoá lại")

    assert enrich.run_build(cfg, paths, data, detector=Boom(), encoder=NoEncode())["roi_crops"] == {"stove": 5, "door": 5}


def test_people_and_roi_loops_then_merge(project):
    run, data, jsons = project
    assert run("enrich", "build") == 0

    assert run("adl", "init", "--profile", "people") == 0
    people_store = adl.AdlStore(jsons / "adl_people")
    assert len(people_store.approved()) == 2 and not (jsons / "adl").exists()          # trạng thái tách riêng theo profile
    assert run("adl", "round", "--profile", "people") == 0
    page = (jsons / "adl_people" / "review_suggestions.html").read_text(encoding="utf-8")
    assert "people/crops/08_1" in page and "adl_people_suggestions_round1" in page
    suggested_people = {s["image_id"] for s in people_store.suggestions()["items"]}
    assert len(suggested_people) == 2 and suggested_people.isdisjoint(people_store.approved())

    assert run("adl", "init", "--profile", "roi-stove") == 0
    stove_store = adl.AdlStore(jsons / "adl_roi-stove")
    seeds = {k: v["label"] for k, v in stove_store.approved().items()}
    assert set(seeds.values()) == {"stove idle", "pot on stove"}
    assert run("adl", "round", "--profile", "roi-stove") == 0
    assert run("adl", "init", "--profile", "roi-khongco") == 1                       # vùng chưa build

    # box tự vẽ trên label_viewer/ (định dạng của label_viewer/store.js)
    (jsons / "manual_boxes.json").write_text(json.dumps({"08_12": [
        {"id": "08_12_m01", "box": [10, 20, 60, 80], "kind": "object", "label": "chảo"},
        {"id": "08_12_m02", "box": [5, 5, 30, 90], "kind": "person"},
    ]}, ensure_ascii=False), encoding="utf-8")
    # nhãn tự gõ trên label_viewer/ thắng nhãn máy (định dạng của label_viewer/store.js)
    (jsons / "custom_labels.json").write_text(json.dumps({"roi-stove": {"08_8": {"text": "nồi bị đổ"}}}, ensure_ascii=False), encoding="utf-8")
    assert run("enrich", "merge") == 0
    rows = (jsons.parent / "exports" / "rois.csv").read_text(encoding="utf-8-sig").splitlines()
    assert rows[0].startswith("image_id,hour,roi,") and any(r.startswith("08_8,08,stove,") and "nồi bị đổ" in r and r.endswith(",human,True") for r in rows)
    assert len((jsons.parent / "exports" / "castle_dataset.jsonl").read_text(encoding="utf-8").splitlines()) == json.loads((jsons.parent / "exports" / "summary.json").read_text(encoding="utf-8"))["images"]
    custom = json.loads((jsons / "enriched" / "08_8.json").read_text(encoding="utf-8"))["rois"]["stove"]["state"]
    assert custom == {"label": "nồi bị đổ", "label_vi": "nồi bị đổ", "source": "human", "score": None, "custom": True}
    (jsons / "custom_labels.json").unlink()
    assert run("enrich", "merge") == 0
    empty = json.loads((jsons / "enriched" / "08_0.json").read_text(encoding="utf-8"))
    assert empty["num_people"] == 0 and empty["manual"] == []
    stove = empty["rois"]["stove"]                                                    # không đổi: thừa hưởng nhãn của ảnh nền bg-08
    assert (stove["name_vi"], stove["changed"], stove["motion"]) == ("bếp", False, 0.0)
    assert (stove["state"]["label"], stove["state"]["label_vi"]) == ("stove idle", "bếp trống") and stove["state"]["source"] in {"seed", "suggested"}
    assert empty["rois"]["door"]["state"] is None                                     # profile roi-door chưa chạy
    pot = json.loads((jsons / "enriched" / "08_8.json").read_text(encoding="utf-8"))
    assert pot["rois"]["stove"]["changed"] and pot["rois"]["stove"]["state"]["label"] == "pot on stove"
    person = json.loads((jsons / "enriched" / "08_12.json").read_text(encoding="utf-8"))
    assert person["num_people"] == 1 and person["people"][0]["zone"] == "stove" and person["people"][0]["zone_vi"] == "bếp"
    assert person["people"][0]["action"]["label"] == "standing" and person["people"][0]["action"]["source"] in {"seed", "suggested"}
    assert person["num_people"] == 1                                                  # box tự vẽ không cộng vào người do máy dò
    assert person["manual"] == [
        {"id": "08_12_m01", "kind": "object", "box": [10, 20, 60, 80], "label": {"label": "chảo", "label_vi": "", "source": "human", "score": None}},
        {"id": "08_12_m02", "kind": "person", "box": [5, 5, 30, 90], "label": None},
    ]



def test_fix_action_overrides_only_machine_labels_and_respects_rejections():
    names = {"walking": "đang đi", "standing idle": "đứng", "eating": "ăn"}
    vlm = {"label": "walking", "label_vi": "đang đi", "source": "vlm", "score": None}
    human = {"label": "walking", "label_vi": "đang đi", "source": "human", "score": None}
    pred = {"a": {"label": "standing idle", "gap": 2.0}, "b": {"label": "standing idle", "gap": 0.3}}
    assert enrich.fix_action(human, "a", pred, {}, names, 1.0) is human                          # người duyệt luôn thắng
    assert enrich.fix_action(vlm, "a", pred, {}, names, 1.0)["label"] == "standing idle"          # gộp chắc hơn nhiều: thay
    assert enrich.fix_action(vlm, "b", pred, {}, names, 1.0) is vlm                              # chênh ít: giữ VLM
    assert enrich.fix_action(vlm, "b", pred, {"b": ["walking"]}, names, 1.0)["label"] == "standing idle"   # nhãn đã bị bác: thay
    assert enrich.fix_action(vlm, "c", {}, {"c": ["walking"]}, names, 1.0) is None              # bị bác, không có gì thay: bỏ

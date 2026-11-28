"""Test mô tả ngoại hình người (modules/person_attrs.py) và việc gộp vào dataset. Không nạp model thật."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import enrich, person_attrs  # noqa: E402
from modules.utils import load_config  # noqa: E402
from test_adl_vlm import ChoiceBackend  # noqa: E402

ANSWER = ('Sure: {"gender": "Female", "age": "adult", "hair": "long brown", "top": "sweater", "top_color": "white", '
          '"top_pattern": "striped", "bottom": "skirt", "bottom_color": "black", "headwear": "none", "glasses": "no", '
          '"apron": "no", "holding": ["Cup", "none"], "view": "back", "extra": 1}')


def test_parse_attrs_keeps_known_fields_and_cleans_values():
    attrs = person_attrs.parse_attrs(ANSWER)
    assert attrs["gender"] == "female" and attrs["holding"] == ["cup"] and "extra" not in attrs and set(attrs) == set(person_attrs.FIELDS)
    assert person_attrs.parse_attrs('{"gender": null}')["gender"] == "unknown" and person_attrs.parse_attrs("không có json") is None
    line = person_attrs.parse_attrs("Here:\nMale; adult; short black; sweater; Gray; plain; jeans; blue; none; yes; no; cup, none; back")
    assert line["top_color"] == "gray" and line["holding"] == ["cup"] and line["view"] == "back"
    assert person_attrs.parse_attrs("male; adult; short") is None


def test_run_attrs_skips_small_crops_resumes_and_merges(tmp_path):
    cfg = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
    data_dir, jsons = tmp_path / "data", tmp_path / "output_jsons"
    crops = data_dir / "people" / "crops"
    crops.mkdir(parents=True)
    people = {"08_1_p01": {"id": "08_1_p01", "image_id": "08_1", "box": [0, 0, 50, 300], "score": 0.9, "zone": None},
              "08_1_p02": {"id": "08_1_p02", "image_id": "08_1", "box": [0, 0, 50, 100], "score": 0.9, "zone": None},   # thấp hơn 160 px
              "08_2_p01": {"id": "08_2_p01", "image_id": "08_2", "box": [0, 0, 80, 400], "score": 0.9, "zone": None}}
    (data_dir / "people" / "people.json").write_text(json.dumps(people), encoding="utf-8")
    for pid in people:
        Image.new("RGB", (40, 120), "gray").save(crops / f"{pid}.jpg")
    backend = ChoiceBackend(lambda image: ANSWER)
    first = person_attrs.run_attrs(cfg, data_dir, jsons, backend, limit=1)
    assert (first["asked"], first["skipped_small"], backend.calls) == (1, 1, 1)
    second = person_attrs.run_attrs(cfg, data_dir, jsons, backend)
    assert (second["asked"], second["already"], backend.calls) == (1, 1, 2)
    saved = person_attrs.load_attrs(jsons / "adl_people" / person_attrs.ATTRS_FILE)
    assert set(saved) == {"08_1_p01", "08_2_p01"} and saved["08_2_p01"]["top_color"] == "white"

    # gộp vào dataset: cột thuộc tính trong people.csv, đồ vật trong objects.csv
    records = [{"image_id": "08_1", "hour": "08", "num_people": 1, "rois": {}, "manual": [],
                "people": [{"id": "08_1_p01", "box": [0, 0, 50, 300], "score": 0.9, "zone": None, "action": None, "attributes": saved["08_1_p01"]}],
                "objects": [{"id": "08_1_o01", "label": "cup", "box": [1, 2, 3, 4], "score": 0.8}]}]
    enrich.write_dataset(records, tmp_path / "dataset")
    rows = (tmp_path / "dataset" / "people.csv").read_text(encoding="utf-8-sig").splitlines()
    assert rows[0].endswith(",custom,identity_cluster,identity_name,identity_source,identity_prob," + ",".join(person_attrs.FIELDS)) and ",female,adult," in rows[1] and rows[1].endswith(",cup,back")
    assert (tmp_path / "dataset" / "objects.csv").read_text(encoding="utf-8-sig").splitlines()[1] == "08_1,08,08_1_o01,cup,1,2,3,4,0.8,"
    summary = json.loads((tmp_path / "dataset" / "summary.json").read_text(encoding="utf-8"))
    assert summary["people_described"] == 1 and summary["object_labels"] == {"cup": 1}


def test_identity_name_person_override_beats_cluster(tmp_path):
    from modules import identity

    (tmp_path / identity.IDENTITIES_FILE).write_text(json.dumps({"assign": {"a_p01": {"cluster": "c01", "sim": 0.9}, "b_p01": {"cluster": "c01", "sim": 0.7}}}), encoding="utf-8")
    (tmp_path / identity.NAMES_FILE).write_text(json.dumps({"clusters": {"c01": "An"}, "people": {"b_p01": ""}}), encoding="utf-8")
    loaded = identity.load_identity(tmp_path)
    assert identity.identity_of("a_p01", *loaded) == {"cluster": "c01", "sim": 0.9, "name": "An", "source": "cluster"}
    assert identity.identity_of("b_p01", *loaded)["name"] is None and identity.identity_of("x_p01", *loaded) is None


def test_static_objects_are_dropped_moving_ones_kept():
    objects = {f"08_{n}": [{"id": f"08_{n}_o01", "label": "kettle", "box": [10, 10, 30, 40]},
                           {"id": f"08_{n}_o02", "label": "cup", "box": [100 + 7 * n, 50, 120 + 7 * n, 70]}] for n in range(10)}
    objects["08_0"].append({"id": "08_0_o03", "label": "kettle", "box": [11, 11, 31, 41]})   # cùng chỗ, cùng ảnh: vẫn tính 1 ảnh
    drop, spots = enrich.static_object_ids(objects, iou=0.5, min_fraction=0.5)
    assert drop == {f"08_{n}_o01" for n in range(10)} | {"08_0_o03"}
    assert spots == [{"label": "kettle", "box": [10.0, 10.0, 30.0, 40.0], "images": 10}]


def test_classifier_names_everyone_and_outfits_follow_the_name(tmp_path):
    """Hai người, mỗi người một vùng embedding: học từ vài crop có tên, đoán đúng cho crop chưa đặt tên; trang phục theo tên."""
    import numpy as np
    from modules import identity

    cfg = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
    cfg["enrich"]["identity"].update(min_examples=3, track=None)
    data_dir, jsons = tmp_path / "data", tmp_path / "out"
    (data_dir / "people").mkdir(parents=True)
    jsons.mkdir()
    rng = np.random.default_rng(0)
    ids = [f"08_{n}_p01" for n in range(20)]
    centers = np.zeros((2, 16), np.float32)
    centers[0, 0], centers[1, 1] = 1, 1
    vectors = np.stack([centers[n % 2] + 0.05 * rng.standard_normal(16) for n in range(20)]).astype(np.float32)
    np.save(data_dir / "people" / "embeddings.npy", vectors)
    (data_dir / "people" / "ids.json").write_text(json.dumps([f"{i}.jpg" for i in ids]), encoding="utf-8")
    (jsons / identity.IDENTITIES_FILE).write_text(json.dumps({"assign": {i: {"cluster": "c00", "sim": 0.5} for i in ids}}), encoding="utf-8")
    named = {ids[n]: ("An" if n % 2 == 0 else "Bình") for n in range(8)}            # 4 crop mỗi người có tên
    (jsons / identity.NAMES_FILE).write_text(json.dumps({"clusters": {}, "people": named}, ensure_ascii=False), encoding="utf-8")
    result = identity.train_predict(cfg, data_dir, jsons)
    assert result["names"] == ["An", "Bình"] and result["train_agree"] == 1.0
    loaded, pred = identity.load_identity(jsons), identity.load_pred(jsons)
    assert identity.final_identity(ids[12], loaded, pred, 0.6)["name"] == "An" and identity.final_identity(ids[13], loaded, pred, 0.6)["source"] == "model"
    assert identity.final_identity(ids[1], loaded, pred, 0.6)["source"] == "person"          # tên đặt riêng không bị mô hình đè

    attrs = {ids[0]: {"top": "sweater", "top_color": "white", "gender": "male"}, ids[2]: {"top": "sweater", "top_color": "white"}, ids[4]: {"top": "shirt", "top_color": "red"}}
    added = identity.update_outfits(jsons, attrs, {ids[0]: "An", ids[2]: "An", ids[4]: "An", ids[1]: "Bình"})
    assert added == {"outfits_added": ["An", "Bình"]}
    outfits = identity.load_outfits(jsons)
    assert outfits["An"]["outfits"][0]["top"] == "sweater" and outfits["An"]["outfits"][0]["top_color_upper"] == ""
    text = (jsons / identity.OUTFITS_FILE).read_text(encoding="utf-8").replace('hours: all', 'hours: [08, 12]')
    (jsons / identity.OUTFITS_FILE).write_text(text, encoding="utf-8")
    outfits = identity.load_outfits(jsons)
    assert identity.outfit_for(outfits, "An", "09")["top_color"] == "white" and identity.outfit_for(outfits, "An", "13") is None
    assert identity.update_outfits(jsons, attrs, {ids[0]: "An"}) == {"outfits_added": []}     # không ghi đè dòng đã sửa

    # người có tên: ngoại hình theo outfits.yaml, holding vẫn theo VLM; không có tên thì giữ VLM
    vlm = {"top": "shirt", "top_color": "red", "holding": ["cup"]}
    fixed, source = identity.apply_outfit(vlm, identity.outfit_for(outfits, "An", "09"))
    assert source == "outfit" and fixed["top_color"] == "white" and fixed["holding"] == ["cup"]
    assert identity.apply_outfit(vlm, None) == (vlm, "vlm") and identity.apply_outfit(None, None) == (None, None)


def test_outfit_key_normalizes_free_text():
    from modules import identity

    assert identity.outfit_key({"gender": "male", "top": "Long-sleeved shirt", "top_color": "dark blue", "top_pattern": "plain"}) == ("male", "shirt", "navy", "plain")
    assert identity.outfit_key({"gender": "male", "top": "plaid shirt", "top_color": "plaid", "top_pattern": "unknown"}) == ("male", "shirt", "unknown", "checked")
    assert identity.outfit_key({"gender": "male", "top": "sweatshirt", "top_color": "white and brown", "top_pattern": "striped"}) == ("male", "sweater", "brown-white", "striped")


def test_outfit_clusters_group_by_clothes_and_place_the_rest_by_embedding(tmp_path):
    import numpy as np
    from modules import identity

    cfg = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
    cfg["enrich"]["identity"]["min_group"] = 3
    data_dir, jsons = tmp_path / "data", tmp_path / "out"
    (data_dir / "people").mkdir(parents=True)
    (jsons / "adl_people").mkdir(parents=True)
    ids = [f"08_{n}_p01" for n in range(9)]
    v = np.zeros((9, 4), np.float32)
    v[:4, 0], v[4:8, 1], v[8, 1] = 1, 1, 0.9
    np.save(data_dir / "people" / "embeddings.npy", v)
    (data_dir / "people" / "ids.json").write_text(json.dumps([f"{i}.jpg" for i in ids]), encoding="utf-8")
    red = {"gender": "male", "top": "shirt", "top_color": "red", "top_pattern": "checked"}
    white = {"gender": "female", "top": "t-shirt", "top_color": "white", "top_pattern": "plain"}
    rows = [{"id": ids[n], "attrs": red if n < 4 else white} for n in range(8)]          # crop 8 chưa có mô tả
    (jsons / "adl_people" / person_attrs.ATTRS_FILE).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    result = identity.run_outfit_clusters(cfg, data_dir, jsons)
    assert result["clusters"] == 2 and result["by_nearest"] == 1
    doc = json.loads((jsons / identity.IDENTITIES_FILE).read_text(encoding="utf-8"))
    assert doc["assign"][ids[8]]["cluster"] == doc["assign"][ids[5]]["cluster"] and doc["assign"][ids[8]]["by"] == "nearest"
    assert {c["label"] for c in doc["clusters"].values()} == {"nam · shirt · red · caro", "nữ · t-shirt · white · trơn"}


def test_name_clusters_follow_manual_names_and_keep_strangers_apart(tmp_path):
    """Hai người có tên + một người chưa ai đặt tên: cụm theo tên cho crop chắc, người lạ vào cụm "chưa chắc"."""
    import numpy as np
    from modules import identity

    cfg = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
    cfg["enrich"]["identity"].update(min_examples=3, uncertain_clusters=1, min_prob=0.6, track=None)
    data_dir, jsons = tmp_path / "data", tmp_path / "out"
    (data_dir / "people").mkdir(parents=True)
    jsons.mkdir()
    rng = np.random.default_rng(0)
    ids = [f"08_{n}_p01" for n in range(30)]
    centers = np.zeros((3, 16), np.float32)
    centers[0, 0], centers[1, 1], centers[2, 2] = 1, 1, 1
    vectors = np.stack([centers[n % 3] + 0.05 * rng.standard_normal(16) for n in range(30)]).astype(np.float32)
    np.save(data_dir / "people" / "embeddings.npy", vectors)
    (data_dir / "people" / "ids.json").write_text(json.dumps([f"{i}.jpg" for i in ids]), encoding="utf-8")
    named = {ids[n]: ("An" if n % 3 == 0 else "Bình") for n in range(12) if n % 3 != 2}
    (jsons / identity.NAMES_FILE).write_text(json.dumps({"clusters": {}, "people": named}, ensure_ascii=False), encoding="utf-8")
    result = identity.run_name_clusters(cfg, data_dir, jsons)
    doc = json.loads((jsons / identity.IDENTITIES_FILE).read_text(encoding="utf-8"))
    assert {c["label"] for c in doc["clusters"].values() if not c["label"].startswith("chưa")} == {"An", "Bình"}
    assert doc["clusters"][doc["assign"][ids[27]]["cluster"]]["label"] == "An"          # crop An chưa đặt tên vào cụm An
    assert result["uncertain"] >= 1


def test_person_tracks_link_same_person_in_nearby_frames_only():
    """Cùng chỗ + cùng embedding ở hai ảnh liền kề thì nối; khác chỗ, khác người hay cách xa quá thì không."""
    import numpy as np
    from modules import identity

    ids = ["08_10_p01", "08_10_p02", "08_12_p01", "08_12_p02", "08_90_p01"]
    box_a, box_b = [0, 0, 100, 200], [500, 0, 600, 200]
    people = {ids[0]: {"box": box_a}, ids[1]: {"box": box_b}, ids[2]: {"box": [5, 0, 105, 200]}, ids[3]: {"box": box_a}, ids[4]: {"box": box_a}}
    vec = np.eye(4, dtype=np.float32)
    vectors = np.stack([vec[0], vec[1], vec[0], vec[2], vec[0]])   # 08_12_p02 đứng chỗ người 1 nhưng là người khác
    t = identity.person_tracks(ids, people, vectors, max_gap=21, min_iou=0.5, min_cos=0.85)
    assert t[0] == t[2] and len({t[0], t[1], t[3], t[4]}) == 4
    assert identity.box_iou(box_a, box_a) == 1.0 and identity.box_iou(box_a, box_b) == 0.0


def test_manual_person_boxes_are_encoded_once_and_named_by_the_model(tmp_path):
    """Box người tự vẽ: cắt crop + mã hoá (cache theo id và box), rồi được học và đoán tên như người do máy dò."""
    import numpy as np
    from PIL import Image
    from modules import identity

    cfg = load_config(Path(__file__).resolve().parents[2] / "config.yaml")
    cfg["enrich"]["identity"].update(min_examples=3, track=None)
    data_dir, jsons, images = tmp_path / "data", tmp_path / "out", tmp_path / "images"
    for d in (data_dir / "people", jsons, images):
        d.mkdir(parents=True)
    Image.new("RGB", (100, 100), (255, 0, 0)).save(images / "08_1.jpg")     # nửa trên đỏ, nửa dưới xanh
    img = Image.open(images / "08_1.jpg"); img.paste((0, 0, 255), (0, 50, 100, 100)); img.save(images / "08_1.jpg")

    class Encoder:
        calls = 0

        def encode(self, crops):
            Encoder.calls += len(crops)
            return [np.array([*np.asarray(c, np.float32).mean((0, 1)) / 255, 0], np.float32) for c in crops]

    ids = [f"08_{n}_p01" for n in range(10)]
    vectors = np.array([[1, 0, 0, 0] if n % 2 == 0 else [0, 0, 1, 0] for n in range(10)], np.float32)   # chẵn đỏ, lẻ xanh
    np.save(data_dir / "people" / "embeddings.npy", vectors)
    (data_dir / "people" / "ids.json").write_text(json.dumps([f"{i}.jpg" for i in ids]), encoding="utf-8")
    named = {ids[n]: ("Đỏ" if n % 2 == 0 else "Xanh") for n in range(10)}
    named["08_1_m02"] = "Xanh"                                                               # box tự vẽ có tên: dùng để học
    (jsons / identity.NAMES_FILE).write_text(json.dumps({"clusters": {}, "people": named}, ensure_ascii=False), encoding="utf-8")
    boxes = {"08_1": [{"id": "08_1_m01", "kind": "person", "box": [10, 5, 90, 40]}, {"id": "08_1_m02", "kind": "person", "box": [10, 60, 90, 95]},
                      {"id": "08_1_m03", "kind": "object", "box": [0, 0, 9, 9], "label": "cốc"}],
             "08_9": [{"id": "08_9_m01", "kind": "person", "box": [0, 0, 9, 9]}]}                # không có ảnh gốc: bỏ qua
    (jsons / enrich.MANUAL_BOXES_FILE).write_text(json.dumps(boxes), encoding="utf-8")

    extra = enrich.manual_people(cfg, data_dir, jsons, images, Encoder())
    assert extra[0] == ["08_1_m01", "08_1_m02"] and Encoder.calls == 2 and extra[1].shape == (2, 4)
    assert (data_dir / "people" / "crops" / "08_1_m01.jpg").exists()
    assert enrich.manual_people(cfg, data_dir, jsons, images, Encoder())[0] == extra[0] and Encoder.calls == 2   # cache: không mã hoá lại
    boxes["08_1"][0]["box"] = [10, 5, 90, 45]                                                # kéo lại box: mã hoá lại đúng box đó
    (jsons / enrich.MANUAL_BOXES_FILE).write_text(json.dumps(boxes), encoding="utf-8")
    extra = enrich.manual_people(cfg, data_dir, jsons, images, Encoder())
    assert Encoder.calls == 3

    result = identity.train_predict(cfg, data_dir, jsons, extra)
    assert result["train"] == 11 and result["people"] == 12
    pred = identity.load_pred(jsons)
    assert pred["08_1_m01"]["name"] == "Đỏ"
    final = identity.final_identity("08_1_m01", identity.load_identity(jsons), pred, 0.5)
    assert (final["name"], final["source"]) == ("Đỏ", "model")
    assert enrich.manual_action({"custom": "rót nước"}, {})["custom"] and enrich.manual_action({"label": "sitting"}, {"sitting": "ngồi"})["label_vi"] == "ngồi"
    assert enrich.manual_action({}, {}) is None

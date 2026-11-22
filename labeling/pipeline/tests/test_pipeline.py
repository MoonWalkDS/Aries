"""Test logic thuần của pipeline (không cần GPU, torch, insightface hay khoá API).

Chạy:  .venv/Scripts/python -m pytest tests -q     (từ thư mục labeling/)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as pipeline_main  # noqa: E402
from modules.schemas import (  # noqa: E402
    BBox,
    Detection,
    EntityAttributes,
    IdentityFile,
    IdentityMatch,
    ImageAnnotation,
    SceneOutput,
)
from modules.utils import Checkpoint, load_config, parse_shard, select_items, spatial_position, write_json  # noqa: E402
from modules.vlm_annotator import extract_json, parse_model  # noqa: E402
from modules.vlm_backends import VLMBackend, VLMResult  # noqa: E402

CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


@pytest.fixture()
def cfg(tmp_path: Path) -> dict:
    return load_config(CONFIG, input_images=str(tmp_path / "images"), output_root=str(tmp_path / "out"))


def unit(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- utils
def test_spatial_position_grid():
    assert spatial_position((0, 0, 100, 100), 1280, 720) == "bên trái, phía trên"
    assert spatial_position((600, 300, 700, 400), 1280, 720) == "ở giữa, ngang giữa"
    assert spatial_position((1100, 600, 1280, 720), 1280, 720) == "bên phải, phía dưới"


def test_checkpoint_resumes_across_instances(tmp_path: Path):
    path = tmp_path / "completed.log"
    first = Checkpoint(path)
    first.mark("08_1002")
    first.mark("08_1010")
    first.mark("08_1002")  # đánh dấu lặp không ghi trùng
    resumed = Checkpoint(path)
    assert "08_1002" in resumed and "08_1010" in resumed and "09_1" not in resumed
    assert len(resumed) == 2 and path.read_text(encoding="utf-8").count("\n") == 2
    resumed.reset()
    assert len(Checkpoint(path)) == 0


def test_shards_partition_all_items():
    items = [Path(f"{i:02d}.jpg") for i in range(10)]
    shards = [select_items(items, parse_shard(f"{i}/3")) for i in (1, 2, 3)]
    assert sorted(p for s in shards for p in s) == items
    assert select_items(items, limit=2, only_ids={"03", "05", "07"}) == [Path("03.jpg"), Path("05.jpg")]
    with pytest.raises(ValueError):
        parse_shard("4/3")


def test_load_config_resolves_paths_and_overrides(tmp_path: Path, monkeypatch):
    default = load_config(CONFIG)
    assert Path(default["paths"]["input_images"]) == (CONFIG.parent.parent / "data" / "dataset" / "images").resolve()   # resolve: ảnh có thể là junction
    monkeypatch.setenv("LABELING_OUTPUT_ROOT", str(tmp_path / "kaggle"))
    overridden = load_config(CONFIG)
    assert Path(overridden["paths"]["output_jsons"]) == tmp_path / "kaggle" / "output_jsons"
    assert Path(overridden["paths"]["logs"]) == tmp_path / "kaggle" / "logs"


# --------------------------------------------------------------------------- detector
# --------------------------------------------------------------------------- reid
def person(entity_id: str, box: tuple[float, float, float, float]) -> Detection:
    return Detection(entity_id=entity_id, label="person", is_person=True, box=BBox.from_seq(box), padded_box=BBox.from_seq(box))


# --------------------------------------------------------------------------- parse JSON của VLM
def test_extract_json_handles_fences_prose_and_trailing_commas():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Đây là kết quả: {"a": {"b": "x } y"}, "c": [1, 2,],} Hết.') == {"a": {"b": "x } y"}, "c": [1, 2]}
    with pytest.raises(ValueError, match="cắt giữa chừng"):
        extract_json('{"a": {"b": 1}')
    with pytest.raises(ValueError, match="không tìm thấy"):
        extract_json("xin lỗi, tôi không thể")


def test_lenient_schema_coerces_model_quirks():
    attrs = parse_model(
        json.dumps({"action": "mở cửa tủ lạnh", "held_objects": "ly nước", "hair": None,
                    "attire": {"upper_garment_type": "áo len", "accessories": None, "extra_key": 1}, "uncertain": None}),
        EntityAttributes,
    )
    assert attrs.held_objects == ["ly nước"] and attrs.hair == "" and attrs.uncertain == []
    assert attrs.attire.upper_garment_type == "áo len" and attrs.attire.accessories == []
    with pytest.raises(ValueError, match="global_summary"):
        parse_model('{"scene_context": {}}', SceneOutput)


# --------------------------------------------------------------------------- annotator với backend giả
ENTITY_JSON = json.dumps({
    "gender_presentation": "nam", "action": "mở cửa tủ lạnh",
    "attire": {"upper_garment_type": "áo len", "upper_garment_pattern": "sọc ngang",
               "upper_color_top": "kem", "upper_color_bottom": "đen"},
    "uncertain": ["footwear"],
}, ensure_ascii=False)
SCENE_JSON = json.dumps({
    "scene_context": {"room_label": "bếp", "lighting": "đèn trần"},
    "scene_graph_relationships": [{"subject": "p01", "predicate": "mở", "object": "o01"}, {"subject": "", "predicate": "x", "object": "y"},
                                  {"subject": "p09", "predicate": "trong", "object": "o01"}, {"subject": "p01", "predicate": "đứng cạnh", "object": "bàn bếp"}],
    "background_clutter": {"secondary_actions": ["hai người đang rửa chén"]},
    "global_summary": "Một người đàn ông mở cửa tủ lạnh.",
    "summary_en": "A man opens the fridge.",
    "progressive_queries": ["Một người đàn ông mở cửa tủ lạnh", "Một người đàn ông mở cửa tủ lạnh. Anh ấy mặc áo len."],
}, ensure_ascii=False)


class FakeBackend(VLMBackend):
    name, model_id = "fake", "fake-vlm"

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.image_sizes: list[tuple[int, int]] = []

    def generate(self, images, prompt, system=None, max_new_tokens=1024) -> VLMResult:
        self.prompts.append(prompt)
        self.image_sizes.append(images[0].size)
        return VLMResult(text=self.replies.pop(0), input_tokens=100, output_tokens=50, seconds=0.5)


def make_inputs(cfg: dict, tmp_path: Path):
    image_path = tmp_path / "10_1528.jpg"
    Image.new("RGB", (1280, 720), "gray").save(image_path)
    raw = [((400.0, 100.0, 600.0, 500.0), "man", "od"), ((900.0, 50.0, 1200.0, 650.0), "refrigerator", "od")]
    record = build_record(image_path, 1280, 720, "a kitchen", raw, cfg)
    identities = IdentityFile(image_id="10_1528", tau_face=0.65, matches=[IdentityMatch(entity_id="p01", face_id="f01", identity="Thầy Linh", similarity=0.81)])
    return image_path, record, identities


# --------------------------------------------------------------------------- Label Studio
def make_annotation(cfg: dict, tmp_path: Path) -> ImageAnnotation:
    image_path, record, identities = make_inputs(cfg, tmp_path)
    return VLMStructuredAnnotator(FakeBackend([ENTITY_JSON, SCENE_JSON]), cfg).annotate_image(image_path, record, identities)


# --------------------------------------------------------------------------- orchestrator
# --------------------------------------------------------------------------- adl
from modules import adl  # noqa: E402


def _blobs(centers: Sequence[int], per: int, noise: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(0)
    rows = [unit(c) + rng.normal(scale=noise, size=512).astype(np.float32) / np.sqrt(512) for c in centers for _ in range(per)]
    return adl.l2_normalize(np.stack(rows))


def test_adl_labels_file_is_valid():
    classes = adl.load_labels(CONFIG.parent / "label_definitions" / "adl_labels.yaml")
    assert len({c.name for c in classes}) == len(classes) >= 10
    assert all(len(c.descriptions) == 7 and c.name_vi for c in classes)


def test_adl_select_seeds_resolves_conflicts_and_caps():
    images = np.stack([unit(1), unit(2), adl.l2_normalize(unit(1) + unit(2)), unit(3)])
    text = {"a": np.stack([unit(1)]), "b": np.stack([adl.l2_normalize(unit(2) + 0.2 * unit(1))])}
    seeds = adl.select_seeds(images, text, per_description=2, max_per_class=5)
    assert seeds[0][0] == "a" and seeds[1][0] == "b"
    assert 3 not in seeds                      # không câu nào kéo ảnh 3 về
    assert seeds[2][0] in {"a", "b"}           # ảnh nằm giữa chỉ thuộc một lớp
    assert len(adl.select_seeds(images, text, per_description=2, max_per_class=1)) == 2
    assert adl.select_seeds(images, text, per_description=2, max_per_class=5, min_similarity=0.99).keys() == {0}


def test_adl_hdbscan_gives_one_centroid_per_scenario():
    emb = _blobs([10, 11, 12], per=12)         # lớp x có hai bối cảnh (10, 11), lớp y có một (12)
    ids = [f"i{n}" for n in range(len(emb))]
    approved = {ids[n]: {"label": "x" if n < 24 else "y"} for n in range(len(emb))}
    centroids = adl.build_centroids(approved, ids, emb, min_cluster_size=5, min_samples=2)
    assert centroids.summary() == {"x": {"clusters": 2, "images": 24}, "y": {"clusters": 1, "images": 12}}
    # lớp quá ít ảnh thì là một cụm
    assert adl.cluster_class(emb[:6], 5, 2).tolist() == [0] * 6


def test_adl_suggest_thresholds_and_respects_rejections():
    emb = _blobs([10, 11, 12, 13], per=6)
    ids = [f"i{n}" for n in range(len(emb))]
    approved = {ids[n]: {"label": "x"} for n in range(0, 3)} | {ids[n]: {"label": "y"} for n in range(6, 9)}
    centroids = adl.build_centroids(approved, ids, emb, 5, 2)
    items = adl.suggest_labels(ids, emb, centroids, set(approved), {}, tau_sim=0.9)
    got = {s["image_id"]: s["label"] for s in items}
    assert got == {"i3": "x", "i4": "x", "i5": "x", "i9": "y", "i10": "y", "i11": "y"}   # cụm 12, 13 dưới ngưỡng
    assert all(s["margin"] > 0.5 for s in items)
    # nhãn đã bị từ chối không được gợi ý lại; lớp còn lại dưới ngưỡng nên ảnh biến mất
    again = adl.suggest_labels(ids, emb, centroids, set(approved), {"i3": ["x"]}, tau_sim=0.9)
    assert "i3" not in {s["image_id"] for s in again}
    assert len(adl.suggest_labels(ids, emb, centroids, set(approved), {}, tau_sim=0.9, max_suggestions=2)) == 2


def test_adl_apply_reviews_approve_change_reject():
    def task(image_id: str, suggested: str, choice: Optional[str], cancelled: bool = False) -> dict:
        result = [{"from_name": adl.ACTIVITY, "value": {"choices": [choice]}}] if choice else []
        annotations = [{"result": result, "was_cancelled": cancelled, "updated_at": "1"}] if choice else []
        return {"data": {"image_id": image_id, "suggested": suggested, "score": 0.95}, "annotations": annotations}

    approved = {"s1": {"label": "x", "source": "seed", "round": 0, "score": 0.3}}
    rejected: dict[str, list[str]] = {}
    tasks = [task("a", "x", "x"), task("b", "x", "y"), task("c", "x", adl.REJECT), task("d", "x", None),
             task("e", "x", "x", cancelled=True), task("s1", "x", adl.REJECT), task("f", "x", "khong-co")]
    stats = adl.apply_reviews(tasks, approved, rejected, {"x", "y"}, round_no=1)
    assert stats == {"approved": 1, "changed": 1, "rejected": 2, "not_reviewed": 2, "invalid": 1}
    assert approved == {"a": {"label": "x", "source": "human", "round": 1, "score": 0.95},
                        "b": {"label": "y", "source": "human", "round": 1, "score": 0.95}}
    assert rejected == {"c": ["x"], "s1": ["x"]}


def test_adl_cli_full_round(cfg: dict, tmp_path: Path, monkeypatch):
    """init -> suggest -> export -> import -> update -> apply trên dữ liệu giả, không nạp CLIP."""
    emb = _blobs([10, 11], per=10)
    names = [f"08_{n}.jpg" for n in range(len(emb))]
    data = tmp_path / "data"
    data.mkdir()
    np.save(data / "embeddings.npy", emb)
    write_json(data / "ids.json", names)
    labels = tmp_path / "labels.yaml"
    labels.write_text("classes:\n  - {name: x, name_vi: lớp x, group: G, descriptions: [d10]}\n  - {name: y, name_vi: lớp y, descriptions: [d11]}\n", encoding="utf-8")
    raw = __import__("yaml").safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["adl"].update(labels_file=str(labels), embeddings=str(data / "embeddings.npy"), ids=str(data / "ids.json"))
    raw["adl"]["seed"].update(per_description=4, max_per_class=4, min_similarity=0.0)
    raw["adl"]["suggest"].update(tau_sim=0.9)
    config = tmp_path / "config.yaml"
    config.write_text(__import__("yaml").safe_dump(raw, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(adl.ClipTextEncoder, "_encode_with_model", lambda self, texts: np.stack([unit(int(t[1:])) for t in texts]))

    out = tmp_path / "out"
    run = lambda *a: pipeline_main.main(["adl", *a, "--config", str(config), "--output-root", str(out)])  # noqa: E731
    store = adl.AdlStore(out / "output_jsons" / "adl")
    assert run("init") == 0
    assert sorted(r["label"] for r in store.approved().values()) == ["x"] * 4 + ["y"] * 4
    assert run("init") == 1                    # không ghi đè golden corpus khi thiếu --force
    assert run("suggest") == 0 and len(store.suggestions()["items"]) == 12
    assert run("export") == 0
    tasks = read_json_file(store.root / "label_studio_adl_suggestions.json")
    assert tasks[0]["data"]["image"].endswith(".jpg") and tasks[0]["predictions"][0]["result"][0]["value"]["choices"]
    for t in tasks:                            # người duyệt chấp nhận mọi gợi ý
        t["annotations"] = [{"result": t["predictions"][0]["result"], "updated_at": "1"}]
    reviewed = tmp_path / "reviewed.json"
    write_json(reviewed, tasks)
    assert run("import", str(reviewed)) == 0
    assert run("update") == 0 and store.load_centroids()[2] == 1
    assert run("suggest") == 0 and store.suggestions()["items"] == []

    per_image = out / "output_jsons" / "per_image"
    write_json(per_image / "08_0.json", ImageAnnotation(image_id="08_0", file_name="08_0.jpg", width=1280, height=720).model_dump())
    assert run("apply") == 0
    final = read_json_file(store.labels_path)
    assert len(final) == 12 and all(v["source"] == "human" for v in final.values())   # hạt giống chưa duyệt bị loại
    assert run("apply", "--include-seeds") == 0 and len(read_json_file(store.labels_path)) == 20
    annotation = ImageAnnotation.model_validate(read_json_file(per_image / "08_0.json"))
    assert annotation.activity is not None and annotation.activity.label_vi in annotation.dense_textual_corpus.bm25_blob


def read_json_file(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- adl: chọn bộ nhãn
from modules import adl_label_selection as sel  # noqa: E402
from modules import vlm_backends  # noqa: E402


class ScriptedBackend(VLMBackend):
    """Trả lời theo hàm ``reply(prompt, số ảnh)``; ghi lại mọi lệnh gọi."""

    name, model_id = "scripted", "fake-llm"

    def __init__(self, reply) -> None:
        self.reply, self.calls = reply, []

    def generate(self, images, prompt, system=None, max_new_tokens=1024) -> VLMResult:
        self.calls.append((len(images), prompt))
        return VLMResult(text=self.reply(prompt, len(images)), input_tokens=10, output_tokens=5)


def test_sample_frames_keeps_temporal_order_per_hour():
    ids = [f"{h:02d}_{n}" for h in (8, 9, 10) for n in (5, 100, 20, 3, 1000, 7)]
    frames = sel.sample_frames(ids, num_groups=20, per_group=3)
    assert frames == {"08": ["08_3", "08_7", "08_1000"], "09": ["09_3", "09_7", "09_1000"], "10": ["10_3", "10_7", "10_1000"]}
    assert len(sel.sample_frames(ids, num_groups=2, per_group=3, seed=1)) == 2
    assert sel.sample_frames(["08_1", "08_2"], 20, 15) == {"08": ["08_1", "08_2"]}


def test_merge_proposals_dedupes_and_counts():
    proposals = {"08": {"present": ["Eating", "eating"], "new": ["opening the fridge"]}, "09": {"present": ["eating"], "new": ["Opening the Fridge", "washing up"]}}
    assert sel.merge_proposals(proposals, ["eating", "writing"]) == {"eating": 2, "opening the fridge": 2, "washing up": 1}
    classes = sel.build_proposed_classes({"eating": 2, "washing up": 1}, [{"name": "Eating", "name_vi": "ăn", "group": "Food & Beverage"}, {"name": "bịa", "group": "X"}])
    assert [(c["name"], c["group"], c["name_vi"]) for c in classes] == [("eating", "Food & Beverage", "ăn"), ("washing up", sel.UNGROUPED, "")]


def test_ask_json_retries_then_fails():
    replies = iter(["không phải json", '```json\n{"a": 1}\n```'])
    backend = ScriptedBackend(lambda prompt, n: next(replies))
    assert sel.ask_json(backend, "hỏi", sel.Usage()) == {"a": 1}
    assert "could not be used" in backend.calls[1][1]
    with pytest.raises(ValueError):
        sel.ask_json(ScriptedBackend(lambda prompt, n: "hỏng"), "hỏi", sel.Usage(), retries=1)


def _selection_reply(prompt: str, n_images: int) -> str:
    if n_images:
        return "Two people chop vegetables.\nOne opens the fridge."
    if '"present"' in prompt:
        return json.dumps({"present": ["eating", "taking a plane"][:1], "new": ["opening the refrigerator"]})
    if '"name_vi"' in prompt:
        return json.dumps({"labels": [{"name": "eating", "name_vi": "ăn uống", "group": "Food & Beverage"}, {"name": "opening the refrigerator", "name_vi": "mở tủ lạnh", "group": "Food & Beverage"}]})
    if '"objects"' in prompt:
        return json.dumps({"objects": ["plate", "fork"], "actions": ["sitting", "chewing"]})
    return json.dumps({"descriptions": [f"Sentence {i}." for i in range(9)] + ["Sentence 0."]})


def test_label_selection_cli_caption_propose_describe(tmp_path: Path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    for h in (8, 9):
        for n in range(4):
            Image.new("RGB", (64, 36), "gray").save(images / f"{h:02d}_{n}.jpg")
    raw = __import__("yaml").safe_load(CONFIG.read_text(encoding="utf-8"))
    labels = tmp_path / "labels.yaml"
    raw["adl"].update(labels_file=str(labels), raw_labels_file=str(CONFIG.parent / "label_definitions" / "adl_raw_labels.yaml"))
    raw["adl"]["label_selection"].update(images_per_group=2, descriptions_per_class=7)
    config = tmp_path / "config.yaml"
    config.write_text(__import__("yaml").safe_dump(raw, allow_unicode=True), encoding="utf-8")
    backend = ScriptedBackend(_selection_reply)
    monkeypatch.setattr(vlm_backends, "create_backend", lambda cfg, name=None, overrides=None: backend)
    out = tmp_path / "out"
    run = lambda *a: pipeline_main.main(["adl", *a, "--config", str(config), "--input-images", str(images), "--output-root", str(out)])  # noqa: E731
    root = out / "output_jsons" / "adl" / "label_selection"

    assert run("propose-labels") == 1                     # chưa caption
    assert run("caption") == 0
    assert read_json_file(root / "frames.json") == {"08": ["08_0", "08_3"], "09": ["09_0", "09_3"]}
    assert read_json_file(root / "descriptions.json")["08"] == "Two people chop vegetables. One opens the fridge. " * 1 + "Two people chop vegetables. One opens the fridge."
    assert sum(1 for n, _ in backend.calls if n == 1) == 4
    assert run("caption") == 0 and sum(1 for n, _ in backend.calls if n == 1) == 4   # caption đã có: không gọi lại

    assert run("propose-labels") == 0
    proposed = tmp_path / "labels.proposed.yaml"
    text = proposed.read_text(encoding="utf-8")
    assert text.startswith("# Bộ nhãn ADL do model đề xuất")
    classes = __import__("yaml").safe_load(text)["classes"]
    assert [(c["name"], c["seen_in"], c["group"]) for c in classes] == [("eating", 2, "Food & Beverage"), ("opening the refrigerator", 2, "Food & Beverage")]
    assert not labels.exists()                            # không tự ghi đè file nhãn chính
    with pytest.raises(ValueError, match="adl describe"):
        adl.load_labels(proposed)                         # chưa có câu mô tả thì init không nhận

    labels.write_text("# chú thích của người duyệt\n\n" + text.split("\n\n", 1)[1], encoding="utf-8")
    assert run("describe") == 0
    done = labels.read_text(encoding="utf-8")
    assert done.startswith("# chú thích của người duyệt")
    loaded = adl.load_labels(labels)
    assert [len(c.descriptions) for c in loaded] == [7, 7] and loaded[1].name_vi == "mở tủ lạnh"
    assert len(set(loaded[0].descriptions)) == 7
    before = len(backend.calls)
    assert run("describe") == 0 and len(backend.calls) == before      # lớp đã đủ câu: bỏ qua
    assert run("describe", "--force") == 0 and len(backend.calls) == before + 4


def test_openai_backend_request_shape_and_errors(cfg: dict, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    with pytest.raises(vlm_backends.BackendError, match="OPENAI_API_KEY"):
        vlm_backends.create_backend(cfg, "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    backend = vlm_backends.create_backend(cfg, "openai", {"model_id": "m", "reasoning_effort": "low"})
    sent: dict = {}

    class Msg:
        content, refusal = "xin chào", None

    class Resp:
        choices = [type("C", (), {"message": Msg, "finish_reason": "stop"})]
        usage = type("U", (), {"prompt_tokens": 7, "completion_tokens": 3})

    monkeypatch.setattr(backend.client.chat.completions, "create", lambda **kw: sent.update(kw) or Resp)
    result = backend.generate([Image.new("RGB", (8, 8))], "mô tả", system="hệ thống", max_new_tokens=100)
    assert (result.text, result.input_tokens, result.output_tokens) == ("xin chào", 7, 3)
    assert sent["model"] == "m" and sent["reasoning_effort"] == "low" and sent["max_completion_tokens"] == 8000
    assert sent["messages"][0] == {"role": "system", "content": "hệ thống"}
    user = sent["messages"][1]["content"]
    assert user[0]["image_url"]["url"].startswith("data:image/jpeg;base64,") and user[1] == {"type": "text", "text": "mô tả"}
    assert len(backend.generate([], "chỉ chữ").text) and len(sent["messages"][0]["content"]) == 1   # không ảnh vẫn chạy
    Msg.refusal = "không"
    with pytest.raises(vlm_backends.BackendError, match="từ chối"):
        backend.generate([], "x")

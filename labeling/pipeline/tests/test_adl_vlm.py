"""Test VLM làm người duyệt (modules/adl_vlm.py) và bước vlm của pipeline. Không nạp model thật."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules import adl, adl_vlm, cloud, vlm_backends  # noqa: E402
from modules.vlm_backends import VLMBackend, VLMResult  # noqa: E402
from test_cloud import cloud_project  # noqa: E402,F401  (fixture)

CLASSES = [adl.AdlClass("eating", "ăn", "", ("A person eating.",)), adl.AdlClass("walking", "đi", "", ("A person walking.",))]


class ChoiceBackend(VLMBackend):
    """Trả lời theo hàm ``choose(ảnh) -> văn bản``; đếm số lệnh gọi."""

    name, model_id = "fake", "fake-vlm"

    def __init__(self, choose) -> None:
        self.choose, self.calls, self.prompts = choose, 0, []

    def generate(self, images, prompt, system=None, max_new_tokens=1024) -> VLMResult:
        self.calls += 1
        self.prompts.append(prompt)
        return VLMResult(text=self.choose(images[0]), input_tokens=10, output_tokens=3, seconds=0.01)


def test_prompt_lists_classes_without_revealing_the_suggestion():
    prompt = adl_vlm.build_prompt(CLASSES, "a crop of one person", "what this person is doing")
    assert "0. none of these" in prompt and "1. eating (e.g. A person eating)" in prompt and "2. walking" in prompt
    assert "a crop of one person" in prompt and "what this person is doing" in prompt and "suggest" not in prompt.lower()


def test_parse_choice_accepts_json_bare_numbers_and_rejects_out_of_range():
    assert adl_vlm.parse_choice('{"choice": 2}', 2) == 2
    assert adl_vlm.parse_choice('```json\n{"choice": "1"}\n```', 2) == 1
    assert adl_vlm.parse_choice("0", 2) == 0 and adl_vlm.parse_choice("Option 2 fits best", 2) == 2
    assert adl_vlm.parse_choice('{"choice": 7}', 2) is None and adl_vlm.parse_choice("không biết", 2) is None
    assert adl_vlm.parse_choice('{"choice": null}', 2) is None


def test_apply_decisions_agree_change_reject_and_protect_human():
    approved = {
        "a": {"label": "eating", "source": "seed", "round": 0, "score": 0.1},
        "b": {"label": "eating", "source": "seed", "round": 0, "score": 0.1},
        "c": {"label": "eating", "source": "seed", "round": 0, "score": 0.1},
        "h": {"label": "eating", "source": "human", "round": 0, "score": 0.1},
    }
    items = [{"image_id": i, "label": "eating", "score": 0.1, "kind": "seed"} for i in ("a", "b", "c", "h", "u")]
    items.append({"image_id": "s", "label": "walking", "score": 0.7, "kind": "suggestion"})
    decisions = {("a", "eating"): "eating", ("b", "eating"): "walking", ("c", "eating"): "", ("h", "eating"): "", ("s", "walking"): "walking"}
    rejected: dict[str, list[str]] = {}
    stats = adl_vlm.apply_decisions(items, decisions, approved, rejected, round_no=2)
    assert stats == {"agree": 2, "changed": 1, "rejected": 1, "unreviewed": 1, "direct": 0, "none": 0}
    assert approved["a"] == {"label": "eating", "source": "vlm", "round": 2, "score": 0.1, "vlm": "agree"}
    assert approved["b"] == {"label": "walking", "source": "vlm", "round": 2, "score": 0.1, "vlm": "changed", "suggested": "eating"}
    assert "c" not in approved and rejected == {"c": ["eating"]}
    assert approved["h"]["source"] == "human"                       # VLM không được đụng nhãn người duyệt
    assert approved["s"]["source"] == "vlm" and "u" not in approved


def _brightness(image) -> float:
    return float(np.asarray(image.convert("L")).mean())


def test_vlm_review_cli_loop_resumes_and_exports_for_audit(cloud_project, monkeypatch):
    run, code, work = cloud_project
    assert run("pipeline") == 0
    store = adl.AdlStore(work / "output_jsons" / "adl_roi-stove")
    before = store.approved()
    assert before and all(r["source"] == "seed" for r in before.values())

    # VLM giả: crop sáng (có nồi vàng hoặc người trắng) -> "pot on stove" (lớp 2); crop tối -> "stove idle" (lớp 1)
    backend = ChoiceBackend(lambda image: json.dumps({"choice": 2 if _brightness(image) > 110 else 1}))
    monkeypatch.setattr(vlm_backends, "create_backend", lambda cfg, name=None, overrides=None: backend)
    assert run("adl", "vlm-review", "--profile", "roi-stove", "--rounds", "3") == 0
    after = store.approved()
    assert all(r["source"] == "vlm" for r in after.values()) and backend.calls >= len(before)
    assert after["bg-08"]["label"] == "stove idle"
    assert {after[i]["label"] for i in ("08_8", "08_9")} == {"pot on stove"}
    reviews = [json.loads(line) for line in (store.root / "vlm_reviews.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(reviews) == backend.calls and {r["model"] for r in reviews} == {"fake-vlm"}
    assert "the current state of this region" in backend.prompts[0] and "(stove)" in backend.prompts[0]

    # chạy lại: mọi mục đã có câu trả lời, không gọi VLM thêm lần nào
    calls = backend.calls
    assert run("adl", "vlm-review", "--profile", "roi-stove") == 0 and backend.calls == calls

    # người soát lại phần VLM đã quyết
    assert run("adl", "export", "--profile", "roi-stove", "--what", "vlm") == 0
    page = (store.root / "review_vlm.html").read_text(encoding="utf-8")
    assert page.count('"image_id"') == len(after) and "adl_roi-stove_vlm_round" in page
    # apply: mặc định chỉ lấy nhãn người; --include-vlm lấy cả nhãn VLM
    assert run("adl", "apply", "--profile", "roi-stove") == 0
    assert json.loads(store.labels_path.read_text(encoding="utf-8")) == {}
    assert run("adl", "apply", "--profile", "roi-stove", "--include-vlm") == 0
    assert len(json.loads(store.labels_path.read_text(encoding="utf-8"))) == len(after)


def test_vlm_limit_caps_calls_and_unreadable_answers_change_nothing(cloud_project, monkeypatch):
    run, code, work = cloud_project
    assert run("pipeline") == 0
    store = adl.AdlStore(work / "output_jsons" / "adl_people")
    before = store.approved()
    backend = ChoiceBackend(lambda image: "tôi không chắc")
    monkeypatch.setattr(vlm_backends, "create_backend", lambda cfg, name=None, overrides=None: backend)
    assert run("adl", "vlm-review", "--profile", "people", "--limit", "2") == 0
    assert backend.calls == 2                                       # 1 mục x (1 lần + 1 lần hỏi lại)
    assert store.approved() == before                               # trả lời không đọc được: giữ nguyên


def test_pipeline_vlm_step_reviews_people_and_rois_then_merge_shows_source(cloud_project, monkeypatch):
    run, code, work = cloud_project
    assert run("pipeline") == 0
    backend = ChoiceBackend(lambda image: '{"choice": 1}')
    monkeypatch.setattr(vlm_backends, "create_backend", lambda cfg, name=None, overrides=None: backend)
    assert run("pipeline", "--steps", "vlm,merge", "--rounds", "2") == 0
    jsons = work / "output_jsons"
    for name in ("adl_people", "adl_roi-stove", "adl_roi-door"):
        approved = adl.AdlStore(jsons / name).approved()
        assert approved and all(r["source"] == "vlm" for r in approved.values()), name
        assert (jsons / name / "review_vlm.html").exists()
    assert all(r["source"] == "seed" for r in adl.AdlStore(jsons / "adl").approved().values())    # profile cả ảnh không nằm trong bước vlm
    record = json.loads((jsons / "enriched" / "08_12.json").read_text(encoding="utf-8"))
    assert record["people"][0]["action"]["source"] == "vlm"
    # --force không được dựng lại golden corpus mà VLM đã duyệt
    assert run("pipeline", "--steps", "loops", "--force") == 0
    assert all(r["source"] == "vlm" for r in adl.AdlStore(jsons / "adl_people").approved().values())


def test_fill_step_labels_every_remaining_crop_once(cloud_project, monkeypatch):
    run, code, work = cloud_project
    assert run("pipeline") == 0
    jsons = work / "output_jsons"
    store = adl.AdlStore(jsons / "adl_roi-stove")
    ids = [Path(n).stem for n in json.loads((work / "data" / "enrich" / "rois" / "stove" / "ids.json").read_text(encoding="utf-8"))]
    # trước fill: một crop đã được người duyệt; một crop VLM từng trả lời "không lớp nào" sẽ không bị hỏi lại
    approved = store.approved()
    human_id = ids[0]
    approved[human_id] = {"label": "stove idle", "source": "human", "round": 0, "score": None}
    store.save_approved(approved)
    asked_before = next(i for i in ids if i not in approved)
    (store.root / "vlm_reviews.jsonl").write_text(json.dumps({"image_id": asked_before, "suggested": "x", "kind": "suggestion", "choice": "", "raw": "", "model": "fake-vlm", "classes": ["stove idle", "pot on stove"]}) + "\n", encoding="utf-8")

    backend = ChoiceBackend(lambda image: json.dumps({"choice": 2 if _brightness(image) > 110 else 1}))
    monkeypatch.setattr(vlm_backends, "create_backend", lambda cfg, name=None, overrides=None: backend)
    assert run("pipeline", "--steps", "fill,merge", "--profiles", "roi-stove") == 0
    after = store.approved()
    assert after[human_id]["source"] == "human"
    assert asked_before not in after                                  # "không lớp nào": vẫn không có nhãn
    assert set(after) == set(ids) - {asked_before}
    assert backend.calls == len(ids) - len(approved) - 1
    assert {r["vlm"] for i, r in after.items() if r["source"] == "vlm" and i not in approved} <= {"direct", "agree", "changed"}
    assert adl.AdlStore(jsons / "adl_people").approved() and all(r["source"] == "seed" for r in adl.AdlStore(jsons / "adl_people").approved().values())
    # chạy lại: không hỏi thêm
    calls = backend.calls
    assert run("pipeline", "--steps", "fill", "--profiles", "roi-stove") == 0 and backend.calls == calls


def test_qwen_falls_back_to_4bit_on_a_single_small_gpu(monkeypatch):
    import types

    import torch

    class Log:
        def info(self, *a):
            pass

    def fake_gpu(sizes):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(sizes))
        monkeypatch.setattr(torch.cuda, "device_count", lambda: len(sizes))
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: types.SimpleNamespace(total_memory=sizes[i] * 2**30))

    fake_gpu([15])
    assert cloud.pick_qwen_overrides(Log()) == {"quantization": "4bit"}
    fake_gpu([15, 15])
    assert cloud.pick_qwen_overrides(Log()) == {}
    fake_gpu([])
    assert cloud.pick_qwen_overrides(Log()) == {}


def test_vlm_step_runs_without_original_images_and_light_bundle_is_small(cloud_project, monkeypatch, tmp_path):
    import zipfile

    run, code, work = cloud_project
    assert run("pipeline") == 0
    backend = ChoiceBackend(lambda image: '{"choice": 1}')
    monkeypatch.setattr(vlm_backends, "create_backend", lambda cfg, name=None, overrides=None: backend)
    monkeypatch.setenv("LABELING_INPUT_IMAGES", str(tmp_path / "khong_ton_tai"))      # Kaggle chỉ có gói kết quả, không có ảnh gốc
    assert run("pipeline", "--steps", "vlm,merge", "--rounds", "1", "--vlm-limit", "3") == 0
    assert 0 < backend.calls <= 3 * 3                                                 # tối đa 3 lệnh gọi cho mỗi trong 3 profile
    assert run("pipeline", "--steps", "build") == 1                                   # bước cần ảnh thì báo lỗi rõ ràng

    light = tmp_path / "light.zip"
    assert run("bundle", "pack", str(light), "--light") == 0
    with zipfile.ZipFile(light) as archive:
        names = archive.namelist()
    assert "output_jsons/adl_people/approved.json" in names and "output_jsons/adl_people/vlm_reviews.jsonl" in names
    assert any(n.startswith("output_jsons/enriched/") for n in names)
    assert not any(n.startswith("data/") or n.endswith(".html") for n in names)

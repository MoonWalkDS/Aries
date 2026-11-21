"""Test lệnh pipeline, bundle và LABELING_DATA_ROOT (chạy trên Kaggle/Colab). Không nạp model thật."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main as pipeline_main  # noqa: E402
from modules import adl, adl_embed, cloud, enrich  # noqa: E402
from modules.utils import load_config  # noqa: E402
from test_enrich import ROIS_YAML, FakeDetector, FakeEncoder, _scene  # noqa: E402

CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"
TEXT = {"white": [1, 1, 1, 0.2], "pot": [1, 0.8, 0.2, 0.2], "idle": [0.5, 0.5, 0.5, 0.2], "closed": [0.5, 0.5, 0.5, 0.2], "gray": [0.5, 0.5, 0.5, 0.2]}


def test_data_root_rebases_only_data_paths(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LABELING_DATA_ROOT", raising=False)
    plain = load_config(CONFIG)
    assert plain["adl"]["embeddings"] == "data/work/adl/embeddings.npy" and Path(plain["_data_root"]) == CONFIG.parent / "data" / "work"
    monkeypatch.setenv("LABELING_DATA_ROOT", str(tmp_path / "d"))
    cfg = load_config(CONFIG)
    root = str((tmp_path / "d").resolve()).replace("\\", "/")
    assert cfg["adl"]["embeddings"] == f"{root}/adl/embeddings.npy" and cfg["enrich"]["out_dir"] == f"{root}/enrich"
    assert cfg["adl_profiles"]["roi"]["labels_file"] == f"{root}/enrich/rois/{{roi}}/labels.yaml"
    assert cfg["adl"]["labels_file"] == "labels/adl_labels.yaml" and cfg["enrich"]["rois_file"] == "labels/rois.yaml"   # file mã nguồn không bị dời
    assert load_config(CONFIG, data_root=str(tmp_path / "cli"))["_data_root"] == str((tmp_path / "cli").resolve())     # CLI thắng env


@pytest.fixture()
def cloud_project(tmp_path: Path, monkeypatch):
    """Mô phỏng Kaggle: mã nguồn một nơi, ảnh một nơi, mọi đầu ra dưới WORK qua biến môi trường."""
    code, images, work = tmp_path / "code", tmp_path / "input" / "images", tmp_path / "work"
    (code / "labels").mkdir(parents=True), images.mkdir(parents=True)
    for n in range(14):
        _scene(images / f"08_{n}.jpg", person_x=(20 if n in (10, 11) else 250 if n in (12, 13) else None), pot=n in (8, 9))
    (code / "labels" / "rois.yaml").write_text(ROIS_YAML, encoding="utf-8")
    (code / "labels" / "adl_people_labels.yaml").write_text("classes:\n  - {name: standing, name_vi: đứng, descriptions: [white]}\n", encoding="utf-8")
    (code / "labels" / "adl_labels.yaml").write_text("classes:\n  - {name: empty kitchen, name_vi: bếp trống, descriptions: [gray]}\n  - {name: busy, name_vi: có người, descriptions: [white]}\n", encoding="utf-8")
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["adl"]["center"] = False
    raw["adl"]["seed"].update(per_description=2, max_per_class=2, min_similarity=-1.0)
    raw["adl"]["suggest"].update(tau_sim=0.5, max_per_round=None)
    raw["enrich"].update(shard_size=5, batch_size=3)
    raw["enrich"]["person"].update(min_area_frac=0.01)
    for name in ("people", "roi"):
        raw["adl_profiles"][name]["seed"].update(per_description=1, max_per_class=2, min_similarity=-1.0)
        raw["adl_profiles"][name]["suggest"].update(tau_sim=0.5, max_per_round=None)
    (code / "config.yaml").write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    monkeypatch.setenv("LABELING_INPUT_IMAGES", str(images))
    monkeypatch.setenv("LABELING_OUTPUT_ROOT", str(work))
    monkeypatch.setenv("LABELING_DATA_ROOT", str(work / "data"))
    monkeypatch.setattr(enrich, "PersonDetector", lambda *a, **k: FakeDetector())
    monkeypatch.setattr(adl_embed, "ClipImageEncoder", lambda *a, **k: FakeEncoder())
    monkeypatch.setattr(adl.ClipTextEncoder, "_encode_with_model", lambda self, texts: adl.l2_normalize(np.array([TEXT[t] for t in texts], dtype=np.float32)))

    def run(*args: str) -> int:
        return pipeline_main.main([*args, "--config", str(code / "config.yaml")])

    return run, code, work


def test_pipeline_runs_everything_under_work_and_is_idempotent(cloud_project):
    run, code, work = cloud_project
    assert run("pipeline") == 0
    jsons, data = work / "output_jsons", work / "data"
    assert not (code / "data").exists() and not (code / "output_jsons").exists()       # không ghi gì vào thư mục mã nguồn
    assert (data / "adl" / "embeddings.npy").exists() and (data / "enrich" / "people" / "people.json").exists()
    for store in ("adl", "adl_people", "adl_roi-stove", "adl_roi-door"):
        assert (jsons / store / "approved.json").exists() and (jsons / store / "review_seeds.html").exists(), store
    assert len(list((jsons / "enriched").glob("*.json"))) == 14
    page = (jsons / "adl_people" / "review_seeds.html").read_text(encoding="utf-8")
    assert "../../data/enrich/people/crops/" in page                                   # đường dẫn tương đối: gói tải về vẫn mở được

    # người duyệt một nhãn; chạy lại pipeline (kể cả --force) không được xoá nó
    store = adl.AdlStore(jsons / "adl_people")
    approved = store.approved()
    first = sorted(approved)[0]
    approved[first] = {"label": "standing", "source": "human", "round": 0, "score": 1.0}
    store.save_approved(approved)
    stamp = (data / "adl" / "embeddings.npy").stat().st_mtime_ns
    assert run("pipeline") == 0
    assert (data / "adl" / "embeddings.npy").stat().st_mtime_ns == stamp               # embedding đã khớp: không dựng lại
    assert run("pipeline", "--steps", "loops", "--force") == 0
    assert adl.AdlStore(jsons / "adl_people").approved()[first]["source"] == "human"
    assert run("pipeline", "--steps", "khong-co") == 1


def test_bundle_pack_and_unpack_on_another_machine(cloud_project, tmp_path: Path, monkeypatch):
    run, code, work = cloud_project
    assert run("pipeline") == 0
    (work / "output_jsons" / "manual_boxes.json").write_text('{"08_0": [{"id": "08_0_m01", "box": [1, 1, 9, 9], "kind": "object"}]}', encoding="utf-8")
    bundle = tmp_path / "castle_results.zip"
    assert run("bundle", "pack", str(bundle)) == 0
    with zipfile.ZipFile(bundle) as archive:
        names = set(archive.namelist())
    assert "data/enrich/people/embeddings.npy" in names and "output_jsons/adl_people/approved.json" in names
    assert "output_jsons/manual_boxes.json" in names                                   # box tự vẽ đi theo gói
    assert "output_jsons/enriched/08_0.json" in names and not any("_shards" in n for n in names)
    assert all(n.startswith(("data/", "output_jsons/")) for n in names)

    # "máy local": mã nguồn ở chỗ khác, data/work/ và annotations/ nằm cạnh config như mặc định
    local = tmp_path / "local"
    (local / "labels").mkdir(parents=True)
    for name in ("config.yaml", "labels/rois.yaml", "labels/adl_people_labels.yaml", "labels/adl_labels.yaml"):
        (local / name).write_bytes((code / name).read_bytes())
    for var in ("LABELING_OUTPUT_ROOT", "LABELING_DATA_ROOT"):
        monkeypatch.delenv(var)
    human = adl.AdlStore(local / "annotations" / "adl_roi-stove")
    human.save_approved({"bg-08": {"label": "pot on stove", "source": "human", "round": 0, "score": 1.0}})   # công duyệt có sẵn ở local
    (local / "annotations" / "manual_boxes.json").write_text("{}", encoding="utf-8")   # local vẽ sau khi đóng gói
    assert pipeline_main.main(["bundle", "unpack", str(bundle), "--config", str(local / "config.yaml")]) == 0
    assert np.load(local / "data" / "work" / "enrich" / "people" / "embeddings.npy").shape == (4, 4)
    assert (local / "annotations" / "enriched" / "08_0.json").exists()
    assert human.approved() == {"bg-08": {"label": "pot on stove", "source": "human", "round": 0, "score": 1.0}}      # không bị gói ghi đè
    assert (local / "annotations" / "manual_boxes.json").read_text(encoding="utf-8") == "{}"                        # box local không bị gói ghi đè
    page = (local / "annotations" / "adl_people" / "review_seeds.html").read_text(encoding="utf-8")
    assert "../../data/work/enrich/people/crops/" in page and (local / "data" / "work" / "enrich" / "people" / "crops").is_dir()
    # vòng lặp chạy tiếp được ở local từ dữ liệu đã giải nén, không cần GPU
    assert pipeline_main.main(["adl", "round", "--profile", "people", "--config", str(local / "config.yaml")]) == 0


def test_unpack_rejects_paths_escaping_the_target(tmp_path: Path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr("data/../../outside.txt", "x")
    cfg = load_config(CONFIG)
    with pytest.raises(ValueError, match="không an toàn"):
        cloud.unpack_results(cfg, evil, tmp_path / "data", tmp_path / "jsons")
    assert not (tmp_path / "outside.txt").exists() and not (tmp_path.parent / "outside.txt").exists()


def test_bundle_code_contains_sources_but_no_data(tmp_path: Path):
    result = cloud.pack_code(CONFIG.parent, tmp_path / "code.zip")
    with zipfile.ZipFile(result["file"]) as archive:
        names = set(archive.namelist())
    for needed in ("castle/pipeline/main.py", "castle/config.yaml", "castle/labels/rois.yaml", "castle/pipeline/modules/enrich.py", "castle/requirements-cloud.txt", "castle/pipeline/notebooks/castle_enrich_cloud.ipynb"):
        assert needed in names, needed
    assert not any(n.startswith(("castle/data/", "castle/annotations/", "castle/.venv/", "castle/viewer/")) or n.endswith(".env") for n in names)
    assert result["size_mb"] < 5


def test_notebook_is_valid_and_points_at_real_commands():
    nb = json.loads((CONFIG.parent / "pipeline" / "notebooks" / "castle_enrich_cloud.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    source = "".join("".join(c["source"]) for c in nb["cells"])
    for needed in ('run("pipeline")', 'run("bundle", "pack"', "LABELING_INPUT_IMAGES", "LABELING_OUTPUT_ROOT", "LABELING_DATA_ROOT", "requirements-cloud.txt"):
        assert needed in source, needed
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<cell>", "exec")      # mỗi ô là Python hợp lệ

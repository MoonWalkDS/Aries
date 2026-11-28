"""Chạy trọn phần cần GPU trong một lệnh và chuyển kết quả giữa Kaggle/Colab với máy local.

``python main.py pipeline``     Mọi bước nặng theo thứ tự, bỏ qua bước đã xong:
                                embed -> enrich build -> (init + suggest + export) cho ảnh, người và
                                từng vùng -> enrich merge. Không bao giờ dựng lại golden corpus đã có.
``python main.py bundle code``  Nén mã nguồn (không kèm venv, dữ liệu, kết quả) để tải lên Kaggle/Colab.
``python main.py bundle pack``  Nén kết quả (embedding, crop, trạng thái adl, bản ghi gộp) để tải về.
``python main.py bundle unpack FILE``  Giải nén kết quả vào đúng chỗ ở máy này và dựng lại các trang
                                duyệt HTML theo đường dẫn của máy này.

Trên Kaggle/Colab đặt ba biến môi trường (notebook ``notebooks/castle_enrich_cloud.ipynb`` làm sẵn):
    LABELING_INPUT_IMAGES   thư mục ảnh (chỉ đọc)
    LABELING_OUTPUT_ROOT    nơi ghi output_jsons/, logs/
    LABELING_DATA_ROOT      nơi ghi adl/, enrich/ (thay cho data/intermediate)
"""
from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import adl
from .utils import free_gpu_memory, get_logger, image_id_of, read_json, resolve_from_config

CODE_PATTERNS = ("config.yaml", "label_definitions/*.yaml", "requirements*.txt", "README.md", "pipeline/main.py", "pipeline/modules/*.py", "pipeline/tests/*.py", "pipeline/notebooks/*.ipynb")
# Thư mục con của data root và của output_jsons được đưa vào gói kết quả.
DATA_DIRS = ("adl", "enrich")
SKIP_DIR_NAMES = {"_shards", "__pycache__"}


# --------------------------------------------------------------------------- pipeline
def _has_human_labels(store: adl.AdlStore) -> bool:
    return any(record.get("source") == "human" for record in store.approved().values())


def _has_reviewed_labels(store: adl.AdlStore) -> bool:
    """Có nhãn do người hoặc VLM duyệt: golden corpus này tốn công/tốn GPU, không được dựng lại."""
    return any(record.get("source") in ("human", "vlm") for record in store.approved().values())


def _loop_once(cfg: dict[str, Any], jsons_dir: Path, profile: Optional[str], force_init: bool, log: Any) -> dict[str, Any]:
    """init (nếu chưa có) -> suggest -> export cho một profile. Trả tóm tắt, hoặc lý do bỏ qua."""
    try:
        pcfg, store_name = adl.apply_profile(cfg, profile)
        store = adl.AdlStore(jsons_dir / store_name)
        if not resolve_from_config(pcfg, pcfg["adl"]["embeddings"]).exists():
            return {"skipped": "chưa có embedding"}
        if not store.approved() or (force_init and not _has_reviewed_labels(store)):
            init = adl.run_init(pcfg, store, force=True)
            seeds = init["seeds"]
        else:
            seeds = len(store.approved())       # đã có golden corpus (có thể đã duyệt): giữ nguyên
            if not store.centroids_path.exists():
                adl.run_update(pcfg, store)
        suggest = adl.run_suggest(pcfg, store)
        adl.run_export(pcfg, store, "seeds")
        if suggest["suggestions"]:
            adl.run_export(pcfg, store, "suggestions")
        return {"seeds": seeds, "suggestions": suggest["suggestions"]}
    except Exception as exc:  # noqa: BLE001 - một profile hỏng (vùng không có crop...) không được dừng cả pipeline
        log.error("pipeline: profile %s lỗi: %s: %s", profile or "(ảnh)", type(exc).__name__, exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_pipeline(
    cfg: dict[str, Any], image_paths: Sequence[Path], jsons_dir: Path, force: bool = False,
    steps: Optional[Sequence[str]] = None, vlm_backend: Optional[str] = None, vlm_rounds: int = 3, vlm_limit: Optional[int] = None,
    profiles: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Chạy mọi bước nặng. ``steps`` chọn trong: embed, build, loops, vlm, fill, merge.

    Mặc định chạy embed, build, loops, merge. Bước ``vlm`` (VLM duyệt nhãn của người và của từng vùng)
    phải gọi đích danh vì nó nạp một model lớn và chạy lâu. Bước ``fill`` (cũng phải gọi đích danh) cho
    VLM gán nhãn trực tiếp MỌI crop còn trống sau các vòng duyệt, mỗi crop một lệnh gọi. ``force`` dựng lại embedding/crop và golden
    corpus, TRỪ các profile đã có nhãn do người hoặc VLM duyệt. ``profiles`` giới hạn bước vlm vào
    vài profile (people, roi-<tên vùng>).
    """
    from . import adl_embed, enrich

    log = get_logger()
    wanted = set(steps or ("embed", "build", "loops", "merge"))
    unknown = wanted - {"embed", "build", "loops", "vlm", "fill", "attrs", "merge"}
    if unknown:
        raise ValueError(f"--steps không hợp lệ: {sorted(unknown)} (chọn trong embed, build, loops, vlm, fill, merge)")
    summary: dict[str, Any] = {}
    started = time.perf_counter()

    def timed(name: str, func: Callable[[], Any]) -> Any:
        t = time.perf_counter()
        log.info("pipeline: ===== %s =====", name)
        result = func()
        log.info("pipeline: %s xong sau %.1f phút", name, (time.perf_counter() - t) / 60)
        free_gpu_memory()
        return result

    if "embed" in wanted:
        emb_path, ids_path = resolve_from_config(cfg, cfg["adl"]["embeddings"]), resolve_from_config(cfg, cfg["adl"]["ids"])
        current = [p.name for p in image_paths]
        if not force and emb_path.exists() and ids_path.exists() and read_json(ids_path) == current:
            summary["embed"] = {"skipped": "embedding đã khớp danh sách ảnh"}
        else:
            out_dir = resolve_from_config(cfg, cfg["adl"]["embed"]["out_dir"])
            result = timed("adl embed", lambda: adl_embed.run_embed(cfg, image_paths, out_dir))
            summary["embed"] = {k: result[k] for k in ("num_indexed", "device", "elapsed_s")}

    data_dir = resolve_from_config(cfg, cfg["enrich"]["out_dir"])
    if "build" in wanted:
        result = timed("enrich build", lambda: enrich.run_build(cfg, image_paths, data_dir, force))
        summary["build"] = {"people": result["people"], "roi_crops": sum(result["roi_crops"].values()), "elapsed_min": result["elapsed_min"]}

    if "loops" in wanted:
        rois = enrich.load_rois(resolve_from_config(cfg, cfg["enrich"]["rois_file"]))
        profiles: list[Optional[str]] = [None, "people", *[f"{adl.ROI_PROFILE_PREFIX}{r.name}" for r in rois]]
        loops = timed("init + suggest + export", lambda: {p or "image": _loop_once(cfg, jsons_dir, p, force, log) for p in profiles})
        adl.ClipTextEncoder.release_model()
        free_gpu_memory()
        summary["loops"] = loops

    if "vlm" in wanted:
        summary["vlm"] = timed("vlm review", lambda: run_vlm_all(cfg, jsons_dir, vlm_backend, vlm_rounds, vlm_limit, profiles))

    if "fill" in wanted:
        summary["fill"] = timed("vlm fill", lambda: run_vlm_all(cfg, jsons_dir, vlm_backend, vlm_rounds, vlm_limit, profiles, fill=True))

    if "attrs" in wanted:
        summary["attrs"] = timed("vlm attrs", lambda: run_attrs_step(cfg, data_dir, jsons_dir, vlm_backend, vlm_limit))

    if "merge" in wanted:
        # Bộ chỉ chở một phần (bundle pack --profiles) có thể thiếu tầng người hoặc tầng vùng.
        if (data_dir / "people" / "people.json").exists() and (data_dir / "rois" / "image_ids.json").exists():
            result = timed("enrich merge", lambda: enrich.run_merge(cfg, data_dir, jsons_dir, Path(image_paths[0]).parent if image_paths else None))
            summary["merge"] = {k: result[k] for k in ("images", "people", "people_labeled", "roi_states")}
        else:
            summary["merge"] = {"skipped": "chưa chạy enrich build"}
    summary["total_min"] = round((time.perf_counter() - started) / 60, 1)
    return summary


def run_attrs_step(cfg: dict[str, Any], data_dir: Path, jsons_dir: Path, backend_name: Optional[str], limit: Optional[int]) -> dict[str, Any]:
    """VLM mô tả ngoại hình mọi crop người đủ lớn (modules/person_attrs.py)."""
    from .person_attrs import run_attrs
    from .vlm_backends import create_backend

    log = get_logger()
    name = backend_name or cfg["adl"]["vlm_review"]["backend"]
    backend = create_backend(cfg, name, pick_qwen_overrides(log) if name == "qwen_local" else None)
    try:
        result = run_attrs(cfg, data_dir, jsons_dir, backend, limit)
    finally:
        del backend
        free_gpu_memory()
    log.info("pipeline: attrs: %s", result)
    return result


def pick_qwen_overrides(log: Any) -> dict[str, Any]:
    """Qwen2.5-VL-7B ở float16 cần khoảng 17 GB. Một GPU nhỏ hơn thế thì chuyển sang 4-bit."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        sizes = [torch.cuda.get_device_properties(i).total_memory / 2**30 for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001 - không đọc được GPU thì để backend tự xử lý
        return {}
    if sum(sizes) < 20:
        log.info("pipeline: GPU %s GB không đủ cho float16, nạp Qwen ở 4-bit", [round(s, 1) for s in sizes])
        return {"quantization": "4bit"}
    return {}


def _pick_profiles(cfg: dict[str, Any], only: Optional[Sequence[str]]) -> list[str]:
    """Profile người và mọi vùng, hoặc chỉ các profile trong ``only`` (người, roi-<tên vùng>)."""
    from . import enrich

    rois = enrich.load_rois(resolve_from_config(cfg, cfg["enrich"]["rois_file"]))
    profiles = ["people", *[f"{adl.ROI_PROFILE_PREFIX}{r.name}" for r in rois]]
    if not only:
        return profiles
    unknown = set(only) - set(profiles)
    if unknown:
        raise ValueError(f"Profile không có: {sorted(unknown)} (chọn trong {profiles})")
    return [p for p in profiles if p in only]


def run_vlm_all(
    cfg: dict[str, Any], jsons_dir: Path, backend_name: Optional[str], rounds: int, limit: Optional[int], only: Optional[Sequence[str]] = None,
    fill: bool = False,
) -> dict[str, Any]:
    """VLM duyệt nhãn cho profile người và mọi vùng (hoặc chỉ ``only``), nạp model một lần. ``limit``: trần lệnh gọi mỗi profile."""
    from . import adl_vlm
    from .vlm_backends import create_backend

    log = get_logger()
    profiles = _pick_profiles(cfg, only)
    name = backend_name or cfg["adl"]["vlm_review"]["backend"]
    backend = create_backend(cfg, name, pick_qwen_overrides(log) if name == "qwen_local" else None)
    log.info("pipeline: VLM duyệt bằng %s (%s)", backend.name, backend.model_id)
    out: dict[str, Any] = {}
    try:
        for profile in profiles:
            try:
                pcfg, store_name = adl.apply_profile(cfg, profile)
                store = adl.AdlStore(jsons_dir / store_name)
                if not store.approved_path.exists():
                    out[profile] = {"skipped": "chưa chạy init"}
                    continue
                if fill:
                    result = adl_vlm.run_vlm_review(pcfg, store, backend, limit, fill=True)
                    out[profile] = {k: result[k] for k in ("direct", "agree", "changed", "none", "unreviewed", "vlm_calls", "sec_per_call")}
                else:
                    result = adl_vlm.run_vlm_loop(pcfg, store, backend, rounds, limit)
                    adl.run_export(pcfg, store, "vlm") if result["labeled_by_vlm"] else None
                    out[profile] = {k: result[k] for k in ("vlm_calls", "labeled_by_vlm", "seeds_left", "suggestions_left")}
                    out[profile]["sec_per_call"] = next((h["sec_per_call"] for h in result["rounds"] if h["sec_per_call"]), None)
                log.info("pipeline: vlm %s: %s", profile, result)
            except Exception as exc:  # noqa: BLE001 - một profile hỏng không dừng các profile còn lại
                log.error("pipeline: vlm %s lỗi: %s: %s", profile, type(exc).__name__, exc)
                out[profile] = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        backend.close()
    return out


# --------------------------------------------------------------------------- đóng gói
def _zip_files(zip_path: Path, entries: Sequence[tuple[Path, str]]) -> dict[str, Any]:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    # Ảnh JPEG và .npy gần như không nén được thêm: lưu nguyên (STORED) cho nhanh.
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path, name in entries:
            stored = path.suffix.lower() in {".jpg", ".jpeg", ".png", ".npy", ".npz"}
            archive.write(path, name, compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
            total += path.stat().st_size
    return {"file": str(zip_path), "files": len(entries), "size_mb": round(zip_path.stat().st_size / 1e6, 1), "source_mb": round(total / 1e6, 1)}


def pack_code(code_dir: Path, zip_path: Path) -> dict[str, Any]:
    """Nén mã nguồn pipeline dưới thư mục ``castle/`` trong file zip (thư mục labeling/, chỉ phần cần cho Kaggle/Colab)."""
    entries = [(path, f"castle/{path.relative_to(code_dir).as_posix()}") for pattern in CODE_PATTERNS for path in sorted(code_dir.glob(pattern)) if path.is_file()]
    if not entries:
        raise RuntimeError(f"Không tìm thấy mã nguồn trong {code_dir}")
    return _zip_files(zip_path, entries)


def _walk(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file() and not (SKIP_DIR_NAMES & set(p.relative_to(root).parts))]


def _profile_entries(data_root: Path, jsons_dir: Path, profiles: Sequence[str], light: bool) -> list[tuple[Path, str]]:
    """Chỉ những gì bước vlm cần cho vài profile: crop + embedding của đơn vị đó và trạng thái vòng lặp."""
    entries: list[tuple[Path, str]] = []
    for profile in profiles:
        unit = "people" if profile == "people" else f"rois/{profile[len(adl.ROI_PROFILE_PREFIX):]}"
        if not light and (data_root / "enrich" / unit).is_dir():
            entries += [(p, f"data/{p.relative_to(data_root).as_posix()}") for p in _walk(data_root / "enrich" / unit)]
        store = jsons_dir / f"adl_{profile}"
        if store.is_dir():
            entries += [(p, f"output_jsons/{p.relative_to(jsons_dir).as_posix()}") for p in _walk(store) if not (light and p.suffix == ".html")]
    return entries


def result_entries(data_root: Path, jsons_dir: Path, light: bool = False, profiles: Optional[Sequence[str]] = None) -> list[tuple[Path, str]]:
    """Các file kết quả và tên của chúng trong gói: ``data/<adl|enrich>/...`` và ``output_jsons/<adl*|enriched>/...``.

    ``light`` bỏ cả cây ``data/`` (crop, embedding) và các trang HTML: dùng khi máy nhận đã có dữ liệu
    và chỉ cần trạng thái mới của các vòng lặp. ``profiles`` chỉ lấy phần của vài profile (people,
    roi-<tên vùng>), ví dụ để chạy lại VLM cho một vùng vừa sửa khung.
    """
    if profiles:
        return _profile_entries(data_root, jsons_dir, profiles, light)
    entries: list[tuple[Path, str]] = []
    for name in DATA_DIRS:
        if not light and (data_root / name).is_dir():
            entries += [(p, f"data/{p.relative_to(data_root).as_posix()}") for p in _walk(data_root / name)]
    if jsons_dir.is_dir():
        for sub in sorted(jsons_dir.iterdir()):
            if sub.is_dir() and (sub.name in ("enriched", "adl", "demo") or sub.name.startswith("adl_")):
                entries += [(p, f"output_jsons/{p.relative_to(jsons_dir).as_posix()}") for p in _walk(sub) if not (light and (sub.name == "demo" or p.suffix == ".html"))]
        from .enrich import MANUAL_BOXES_FILE

        manual = jsons_dir / MANUAL_BOXES_FILE  # để `enrich merge` trên máy cloud giữ được box tự vẽ
        if manual.is_file():
            entries.append((manual, f"output_jsons/{MANUAL_BOXES_FILE}"))
    return entries


def pack_results(data_root: Path, jsons_dir: Path, zip_path: Path, light: bool = False, profiles: Optional[Sequence[str]] = None) -> dict[str, Any]:
    entries = result_entries(data_root, jsons_dir, light, profiles)
    if not entries:
        raise RuntimeError(f"Chưa có kết quả nào trong {data_root} hoặc {jsons_dir}. Chạy `pipeline` trước.")
    return _zip_files(zip_path, entries)


def unpack_results(cfg: dict[str, Any], zip_path: Path, data_root: Path, jsons_dir: Path, force: bool = False) -> dict[str, Any]:
    """Giải nén gói kết quả vào ``data_root`` và ``jsons_dir`` của máy này, rồi dựng lại trang duyệt.

    Profile nào ở máy này đã có nhãn người duyệt thì giữ nguyên ``approved.json`` / ``rejected.json``
    của nó (trừ khi ``force``), để gói tải về không ghi đè công duyệt.
    """
    log = get_logger()
    roots = {"data": data_root.resolve(), "output_jsons": jsons_dir.resolve()}
    from .enrich import MANUAL_BOXES_FILE

    protected = set()
    if jsons_dir.is_dir() and not force:
        for sub in jsons_dir.iterdir():
            if sub.is_dir() and _has_human_labels(adl.AdlStore(sub)):
                protected |= {f"output_jsons/{sub.name}/approved.json", f"output_jsons/{sub.name}/rejected.json"}
                log.warning("unpack: %s đã có nhãn người duyệt, giữ nguyên approved/rejected của máy này.", sub.name)
        if (jsons_dir / MANUAL_BOXES_FILE).is_file():  # box chỉ vẽ ở máy local: bản trong gói có thể cũ hơn
            protected.add(f"output_jsons/{MANUAL_BOXES_FILE}")
    written = skipped = 0
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            top, _, rest = info.filename.partition("/")
            if top not in roots or not rest:
                continue
            if info.filename in protected:
                skipped += 1
                continue
            target = (roots[top] / rest).resolve()
            if roots[top] not in target.parents:      # tên file kiểu ../ trỏ ra ngoài thư mục đích
                raise ValueError(f"Gói chứa đường dẫn không an toàn: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(target, "wb") as sink:
                while chunk := source.read(1 << 20):
                    sink.write(chunk)
            written += 1

    # Trang duyệt trong gói trỏ tới đường dẫn của máy đã chạy; dựng lại theo máy này.
    pages = 0
    from . import enrich

    try:
        rois = enrich.load_rois(resolve_from_config(cfg, cfg["enrich"]["rois_file"]))
    except (OSError, ValueError):
        rois = []
    for profile in [None, "people", *[f"{adl.ROI_PROFILE_PREFIX}{r.name}" for r in rois]]:
        try:
            pcfg, store_name = adl.apply_profile(cfg, profile)
            store = adl.AdlStore(jsons_dir / store_name)
            if not store.approved_path.exists():
                continue
            if any(r.get("source") == "seed" for r in store.approved().values()):
                adl.run_export(pcfg, store, "seeds")
                pages += 1
            if store.suggestions()["items"]:
                adl.run_export(pcfg, store, "suggestions")
                pages += 1
        except Exception as exc:  # noqa: BLE001 - thiếu file của một profile không làm hỏng việc giải nén
            log.warning("unpack: không dựng lại được trang duyệt của %s: %s", profile or "(ảnh)", exc)
    return {"written": written, "kept_local": skipped, "review_pages": pages, "data_root": str(data_root), "output_jsons": str(jsons_dir)}


def check_images(image_paths: Sequence[Path]) -> dict[str, Any]:
    """Tóm tắt thư mục ảnh để phát hiện sớm việc trỏ nhầm đường dẫn trên Kaggle/Colab."""
    groups: dict[str, int] = {}
    for path in image_paths:
        key = image_id_of(path).split("_", 1)[0]
        groups[key] = groups.get(key, 0) + 1
    return {"images": len(image_paths), "groups": dict(sorted(groups.items()))}

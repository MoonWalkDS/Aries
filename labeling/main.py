"""Orchestrator của pipeline gán nhãn Castle5000.

Mỗi stage là một lệnh con và chạy trong một tiến trình riêng, nên VRAM được trả lại giữa các
model và mỗi stage có thể chạy ở một máy khác nhau (GPU trên Kaggle, phần còn lại ở local).

    python main.py detect            # Florence-2: box + crop
    python main.py faces             # InsightFace: khuôn mặt + embedding
    python main.py cluster           # gom cụm mặt -> đặt tên trong cluster_names.yaml
    python main.py build-gallery     # cụm đã đặt tên -> identity_gallery/<Tên>/
    python main.py reid              # so mặt với gallery, gán tên (tau_face)
    python main.py annotate          # VLM: mô tả có cấu trúc
    python main.py pilot             # chạy thử 30 ảnh với nhiều model, xuất báo cáo so sánh
    python main.py merge             # gộp thành annotations.jsonl + annotations_master.json
    python main.py render            # vẽ box + nhãn lên ảnh để xem bằng mắt
    python main.py ls-export         # task + pre-label cho Label Studio
    python main.py ls-import FILE    # nhập bản đã kiểm tra từ Label Studio
    python main.py adl ACTION        # nhãn hoạt động: caption | propose-labels | describe | init | suggest |
                                     #   export | import FILE | update | apply | status
    python main.py enrich ACTION     # làm giàu theo tầng: build (crop người + vùng cố định) | merge | demo
    python main.py pipeline          # mọi bước cần GPU của adl + enrich trong một lệnh (Kaggle/Colab)
    python main.py bundle ACTION     # code | pack | unpack FILE: chuyển mã nguồn / kết quả giữa các máy
    python main.py all               # detect -> faces -> reid -> annotate -> merge

Resume: mỗi stage nặng ghi ID ảnh đã xong vào logs/completed_<stage>.log sau từng ảnh. Chạy lại
cùng lệnh sẽ bỏ qua các ảnh đó. Dùng --force để làm lại từ đầu.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm import tqdm

from modules.schemas import DetectionRecord, FaceFile, IdentityFile, ImageAnnotation
from modules.utils import (
    Checkpoint,
    append_jsonl,
    enable_utf8_console,
    free_gpu_memory,
    image_id_of,
    list_images,
    load_config,
    parse_shard,
    read_id_file,
    read_json,
    select_items,
    setup_logging,
    write_json,
    write_jsonl,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


class Context:
    """Cấu hình + các thư mục đã resolve, dùng chung cho mọi stage."""

    def __init__(self, args: argparse.Namespace, stage: str) -> None:
        self.args = args
        self.stage = stage
        self.cfg: dict[str, Any] = load_config(args.config, args.input_images, args.output_root, getattr(args, "data_root", None))
        paths = self.cfg["paths"]
        self.images_dir = Path(paths["input_images"])
        self.crops_dir = Path(paths["output_crops"])
        self.jsons_dir = Path(paths["output_jsons"])
        self.logs_dir = Path(paths["logs"])
        self.gallery_dir = Path(paths["identity_gallery"])
        self.detections_dir = self.jsons_dir / "detections"
        self.faces_dir = self.jsons_dir / "faces"
        self.identities_dir = self.jsons_dir / "identities"
        self.per_image_dir = self.jsons_dir / "per_image"
        self.clusters_dir = self.jsons_dir / "clusters"
        self.log = setup_logging(self.logs_dir, stage, args.verbose)

    def images(self) -> list[Path]:
        """Danh sách ảnh sau khi áp --ids-file, --shard, --limit."""
        only = read_id_file(self.args.ids_file) if self.args.ids_file else None
        return select_items(list_images(self.images_dir), parse_shard(self.args.shard), self.args.limit, only)

    def checkpoint(self, stage: Optional[str] = None) -> Checkpoint:
        checkpoint = Checkpoint(self.logs_dir / f"completed_{stage or self.stage}.log")
        if self.args.force:
            checkpoint.reset()
        return checkpoint


def run_loop(
    ctx: Context,
    stage: str,
    process: Callable[[Path, str], None],
    fatal: tuple[type[BaseException], ...] = (),
) -> dict[str, int]:
    """Vòng lặp chung cho các stage theo từng ảnh: bỏ qua ảnh đã xong, bắt lỗi, ghi checkpoint.

    Một ảnh lỗi không dừng cả lượt: lỗi được ghi vào logs/failed_<stage>.jsonl và ảnh đó
    KHÔNG được đánh dấu xong, nên lần chạy sau sẽ thử lại. Lỗi thuộc ``fatal`` (cấu hình sai,
    hết quota) thì dừng ngay vì mọi ảnh sau cũng sẽ lỗi như vậy.
    """
    checkpoint = ctx.checkpoint(stage)
    images = ctx.images()
    pending = [p for p in images if image_id_of(p) not in checkpoint]
    ctx.log.info("[%s] %d ảnh được chọn, %d đã xong, %d cần xử lý", stage, len(images), len(images) - len(pending), len(pending))

    done = failed = 0
    started = time.perf_counter()
    empty_every = int(ctx.cfg["vlm"].get("empty_cache_every", 10))
    for index, image_path in enumerate(tqdm(pending, desc=stage, unit="ảnh"), start=1):
        image_id = image_id_of(image_path)
        try:
            process(image_path, image_id)
        except KeyboardInterrupt:
            ctx.log.warning("[%s] Bị ngắt. Chạy lại cùng lệnh để tiếp tục từ %s.", stage, image_id)
            raise
        except fatal as exc:
            ctx.log.error("[%s] Dừng vì lỗi không khắc phục được: %s", stage, exc)
            raise
        except Exception as exc:  # noqa: BLE001 - một ảnh hỏng không được làm dừng 5000 ảnh
            failed += 1
            ctx.log.error("[%s] %s lỗi: %s: %s", stage, image_id, type(exc).__name__, exc)
            ctx.log.debug("Chi tiết lỗi %s", image_id, exc_info=True)
            append_jsonl(ctx.logs_dir / f"failed_{stage}.jsonl", {"image_id": image_id, "error": f"{type(exc).__name__}: {exc}"})
            free_gpu_memory()
            continue
        checkpoint.mark(image_id)
        done += 1
        if index % empty_every == 0:
            free_gpu_memory()
        if index % 50 == 0:
            rate = (time.perf_counter() - started) / index
            ctx.log.info("[%s] %d/%d, %.1fs/ảnh, còn khoảng %.0f phút", stage, index, len(pending), rate, rate * (len(pending) - index) / 60)

    elapsed = time.perf_counter() - started
    ctx.log.info("[%s] xong %d, lỗi %d, %.1f phút", stage, done, failed, elapsed / 60)
    return {"done": done, "failed": failed, "skipped": len(images) - len(pending)}


# --------------------------------------------------------------------------- các stage
def stage_detect(ctx: Context) -> None:
    from modules.detector import SceneDetector

    detector = SceneDetector(ctx.cfg)

    def process(image_path: Path, image_id: str) -> None:
        record = detector.detect(image_path, ctx.crops_dir)
        write_json(ctx.detections_dir / f"{image_id}.json", record.model_dump())

    run_loop(ctx, "detect", process)


def _load_detection(ctx: Context, image_id: str) -> DetectionRecord:
    path = ctx.detections_dir / f"{image_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Thiếu {path}; chạy `detect` cho ảnh này trước.")
    return DetectionRecord.model_validate(read_json(path))


def _require_detections(ctx: Context) -> None:
    """Dừng sớm, trước khi nạp model nặng, nếu chưa ảnh nào được chọn có kết quả detect."""
    if not any((ctx.detections_dir / f"{image_id_of(p)}.json").exists() for p in ctx.images()):
        raise RuntimeError(f"Chưa có kết quả detect nào trong {ctx.detections_dir} cho các ảnh được chọn. Chạy `detect` trước.")


def stage_faces(ctx: Context) -> None:
    from modules.reid import BiometricReID

    _require_detections(ctx)
    reid = BiometricReID(ctx.cfg)

    def process(image_path: Path, image_id: str) -> None:
        record = _load_detection(ctx, image_id)
        face_file = reid.extract_faces(image_path, image_id, record.persons())
        write_json(ctx.faces_dir / f"{image_id}.json", face_file.model_dump())

    run_loop(ctx, "faces", process)


def stage_cluster(ctx: Context) -> None:
    from modules.reid import cluster_faces

    summary = cluster_faces(ctx.cfg, ctx.faces_dir, ctx.images_dir, ctx.clusters_dir)
    ctx.log.info("Gom cụm: %s", summary)
    ctx.log.info("Mở các file cluster_XX.jpg trong %s, điền tên vào cluster_names.yaml, rồi chạy build-gallery.", ctx.clusters_dir)


def stage_build_gallery(ctx: Context) -> None:
    from modules.reid import build_gallery

    written = build_gallery(ctx.cfg, ctx.clusters_dir, ctx.faces_dir, ctx.images_dir, ctx.gallery_dir)
    ctx.log.info("Gallery tại %s: %s", ctx.gallery_dir, written)


def stage_reid(ctx: Context) -> None:
    """So mọi khuôn mặt đã trích với gallery. Nhanh (chỉ là phép nhân ma trận) nên luôn tính lại toàn bộ."""
    from modules.reid import BiometricReID, match_faces, summarize_similarities

    tau = float(ctx.cfg["thresholds"]["tau_face"])
    has_gallery = ctx.gallery_dir.is_dir() and any(p.is_dir() for p in ctx.gallery_dir.iterdir())
    if has_gallery:
        names, matrix = BiometricReID(ctx.cfg).load_gallery(ctx.gallery_dir)
    else:
        import numpy as np

        names, matrix = [], np.zeros((0, 512), dtype="float32")
        ctx.log.warning("Gallery rỗng (%s): mọi người sẽ là Unknown.", ctx.gallery_dir)

    wanted = {image_id_of(p) for p in ctx.images()}
    results: list[IdentityFile] = []
    for path in sorted(ctx.faces_dir.glob("*.json")):
        if path.stem not in wanted:
            continue
        identity_file = match_faces(FaceFile.model_validate(read_json(path)), names, matrix, tau)
        write_json(ctx.identities_dir / f"{path.stem}.json", identity_file.model_dump())
        results.append(identity_file)
    ctx.log.info("Re-ID cho %d ảnh, gallery có %d ảnh tham chiếu của %d người", len(results), len(names), len(set(names)))
    ctx.log.info("\n%s", summarize_similarities(results, tau))


def stage_annotate(ctx: Context) -> None:
    from modules.vlm_annotator import VLMStructuredAnnotator
    from modules.vlm_backends import BackendError, create_backend

    _require_detections(ctx)
    backend = create_backend(ctx.cfg, ctx.args.backend)
    annotator = VLMStructuredAnnotator(backend, ctx.cfg)
    ctx.log.info("Backend VLM: %s (%s)", backend.name, backend.model_id)

    def process(image_path: Path, image_id: str) -> None:
        record = _load_detection(ctx, image_id)
        id_path = ctx.identities_dir / f"{image_id}.json"
        identities = IdentityFile.model_validate(read_json(id_path)) if id_path.exists() else None
        annotation = annotator.annotate_image(image_path, record, identities)
        write_json(ctx.per_image_dir / f"{image_id}.json", annotation.model_dump())

    try:
        run_loop(ctx, "annotate", process, fatal=(BackendError,))
    finally:
        backend.close()


def stage_pilot(ctx: Context) -> None:
    from modules.pilot import build_report, run_pilot_model, select_pilot_ids

    pilot_dir = ctx.jsons_dir / "pilot"
    ids_path = pilot_dir / "pilot_ids.txt"
    if ids_path.exists() and not ctx.args.reselect:
        image_ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        image_ids = select_pilot_ids(ctx.cfg, list_images(ctx.images_dir))
        pilot_dir.mkdir(parents=True, exist_ok=True)
        ids_path.write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    ctx.log.info("Pilot: %d ảnh, danh sách tại %s", len(image_ids), ids_path)
    if ctx.args.select_only:
        ctx.log.info("Chạy detect/faces/reid cho các ảnh này bằng: --ids-file %s", ids_path)
        return

    if not ctx.args.report_only:
        specs = ctx.cfg["pilot"]["models"]
        if ctx.args.models:
            wanted = {m.strip() for m in ctx.args.models.split(",")}
            unknown = wanted - {s["name"] for s in specs}
            if unknown:
                raise SystemExit(f"--models: không có trong pilot.models của config: {sorted(unknown)}")
            specs = [s for s in specs if s["name"] in wanted]
        for spec in specs:
            ctx.log.info("Pilot: chạy %s", spec["name"])
            ctx.log.info("Pilot: %s", run_pilot_model(ctx.cfg, spec, image_ids, ctx.images_dir, ctx.jsons_dir))
            free_gpu_memory()
    ctx.log.info("Báo cáo: %s", build_report(ctx.cfg, ctx.images_dir, ctx.jsons_dir))


def stage_merge(ctx: Context) -> None:
    """Gộp mọi file per-image thành một file JSONL (mỗi dòng một ảnh) và một file JSON tổng."""
    records = [ImageAnnotation.model_validate(read_json(p)).model_dump() for p in sorted(ctx.per_image_dir.glob("*.json"))]
    if not records:
        raise SystemExit(f"Không có annotation nào trong {ctx.per_image_dir}.")
    count = write_jsonl(ctx.jsons_dir / "annotations.jsonl", records)
    write_json(ctx.jsons_dir / "annotations_master.json", {"num_images": count, "images": records})
    verified = sum(1 for r in records if r["verified"])
    ctx.log.info("Gộp %d ảnh (%d đã kiểm tra) vào %s", count, verified, ctx.jsons_dir / "annotations.jsonl")


def stage_ls_export(ctx: Context) -> None:
    from modules.label_studio import export_tasks, push_to_label_studio

    summary = export_tasks(ctx.cfg, ctx.per_image_dir, ctx.jsons_dir, ctx.gallery_dir)
    ctx.log.info("Label Studio: %s", summary)
    if ctx.args.push:
        project_id = push_to_label_studio(ctx.cfg, Path(summary["tasks_file"]), Path(summary["config_file"]))
        ctx.log.info("Đã tạo project Label Studio id=%d", project_id)


def stage_ls_import(ctx: Context) -> None:
    from modules.label_studio import import_export

    summary = import_export(Path(ctx.args.export_file), ctx.per_image_dir)
    ctx.log.info("Nhập từ Label Studio: %s. Chạy `merge` để cập nhật file tổng.", summary)


def stage_render(ctx: Context) -> None:
    """Vẽ box + toàn bộ nhãn lên ảnh để xem bằng mắt: output_jsons/rendered/<image_id>.jpg."""
    from modules.visualize import render_all

    only = {image_id_of(p) for p in ctx.images()} if (ctx.args.ids_file or ctx.args.limit or ctx.args.shard) else None
    written = render_all(ctx.per_image_dir, ctx.images_dir, ctx.jsons_dir / "rendered", only, not ctx.args.no_objects)
    ctx.log.info("Đã render %d ảnh vào %s", len(written), ctx.jsons_dir / "rendered")


def _adl_label_selection(ctx: Context, root: Path, action: str) -> dict[str, Any]:
    """Chọn bộ nhãn (mục 3.2) và sinh câu mô tả (Decompose-Expand). Các bước này gọi VLM/LLM."""
    from modules import adl_label_selection as sel
    from modules.utils import resolve_from_config
    from modules.vlm_backends import create_backend

    adl_cfg = ctx.cfg["adl"]
    labels_file = Path(ctx.args.labels_file) if ctx.args.labels_file else resolve_from_config(ctx.cfg, adl_cfg["labels_file"])
    role = "caption_backend" if action == "caption" else "llm_backend"
    backend = create_backend(ctx.cfg, ctx.args.backend or adl_cfg["label_selection"][role])
    ctx.log.info("adl %s: backend %s (%s)", action, backend.name, backend.model_id)
    store = sel.SelectionStore(root)
    try:
        if action == "caption":
            return sel.run_caption(ctx.cfg, store, backend, list_images(ctx.images_dir), ctx.args.resample)
        if action == "propose-labels":
            return sel.run_propose(ctx.cfg, store, backend, labels_file, resolve_from_config(ctx.cfg, adl_cfg["raw_labels_file"]), ctx.args.force)
        return sel.run_describe(ctx.cfg, backend, labels_file, ctx.args.force)
    finally:
        backend.close()


def stage_adl(ctx: Context) -> None:
    """Nhãn hoạt động cấp ảnh theo LSC-ADL. Một vòng: suggest -> export -> (duyệt) -> import -> update."""
    from modules import adl

    ctx.cfg, store_name = adl.apply_profile(ctx.cfg, ctx.args.profile)
    store = adl.AdlStore(ctx.jsons_dir / store_name)
    action = ctx.args.action
    if action == "embed":
        from modules.adl_embed import run_embed
        from modules.utils import resolve_from_config

        result = run_embed(ctx.cfg, list_images(ctx.images_dir), resolve_from_config(ctx.cfg, ctx.cfg["adl"]["embed"]["out_dir"]))
    elif action in ("caption", "propose-labels", "describe"):
        result = _adl_label_selection(ctx, store.root / "label_selection", action)
    elif action == "init":
        result = adl.run_init(ctx.cfg, store, ctx.args.force)
    elif action == "suggest":
        result = adl.run_suggest(ctx.cfg, store)
    elif action == "export":
        result = adl.run_export(ctx.cfg, store, ctx.args.what)
    elif action == "import":
        if not ctx.args.export_file:
            raise SystemExit("adl import cần đường dẫn file export JSON của Label Studio.")
        result = adl.run_import(ctx.cfg, store, Path(ctx.args.export_file))
    elif action == "update":
        result = adl.run_update(ctx.cfg, store)
    elif action == "round":
        result = adl.run_round(ctx.cfg, store)
    elif action == "vlm-review":
        from modules import adl_vlm
        from modules.vlm_backends import create_backend

        backend = create_backend(ctx.cfg, ctx.args.backend or ctx.cfg["adl"]["vlm_review"]["backend"])
        ctx.log.info("adl vlm-review: backend %s (%s)", backend.name, backend.model_id)
        try:
            result = adl_vlm.run_vlm_loop(ctx.cfg, store, backend, ctx.args.rounds, ctx.args.limit)
        finally:
            backend.close()
    elif action == "apply":
        result = adl.run_apply(ctx.cfg, store, ctx.per_image_dir, ctx.args.include_seeds, ctx.args.include_vlm)
    else:
        result = adl.run_status(ctx.cfg, store)
    ctx.log.info("adl %s: %s", action, result)


def stage_enrich(ctx: Context) -> None:
    """Làm giàu theo tầng: crop người + crop vùng cố định, rồi gộp thành bản ghi mỗi ảnh."""
    from modules import enrich
    from modules.utils import resolve_from_config

    data_dir = resolve_from_config(ctx.cfg, ctx.cfg["enrich"]["out_dir"])
    if ctx.args.action == "build":
        only = [s.strip() for s in ctx.args.rois.split(",") if s.strip()] if ctx.args.rois else None
        result = enrich.run_build(ctx.cfg, ctx.images(), data_dir, ctx.args.force, only_rois=only)
    elif ctx.args.action == "objects":
        run = enrich.run_owl_objects if ctx.args.owl else enrich.run_objects
        result = run(ctx.cfg, ctx.images(), data_dir, ctx.args.force)
    elif ctx.args.action == "identities":
        from modules.identity import run_identities, run_name_clusters, run_outfit_clusters

        run = {"outfit": run_outfit_clusters, "name": run_name_clusters}.get(ctx.cfg["enrich"]["identity"].get("method"), run_identities)
        result = run(ctx.cfg, data_dir, ctx.jsons_dir, ctx.args.force)
    elif ctx.args.action == "actions":
        from modules.action_fuse import run_actions

        result = run_actions(ctx.cfg, data_dir, ctx.jsons_dir)
    elif ctx.args.action == "demo":
        from modules.enrich_demo import run_demo

        result = run_demo(ctx.cfg, ctx.images(), data_dir, ctx.jsons_dir)
    else:
        result = enrich.run_merge(ctx.cfg, data_dir, ctx.jsons_dir)
    ctx.log.info("enrich %s: %s", ctx.args.action, result)


def stage_pipeline(ctx: Context) -> None:
    """Mọi bước cần GPU của adl + enrich trong một lệnh; dùng trên Kaggle/Colab."""
    from modules import cloud

    steps = [s.strip() for s in ctx.args.steps.split(",") if s.strip()] if ctx.args.steps else None
    # Chỉ embed và build đọc ảnh gốc; các bước còn lại chạy được ở nơi không có ảnh (Kaggle chỉ có gói kết quả).
    needs_images = steps is None or bool({"embed", "build"} & set(steps))
    images = ctx.images() if needs_images else []
    if needs_images:
        ctx.log.info("pipeline: ảnh từ %s: %s", ctx.images_dir, cloud.check_images(images))
    profiles = [s.strip() for s in ctx.args.profiles.split(",") if s.strip()] if ctx.args.profiles else None
    ctx.log.info("pipeline: %s", cloud.run_pipeline(ctx.cfg, images, ctx.jsons_dir, ctx.args.force, steps, ctx.args.backend, ctx.args.rounds, ctx.args.vlm_limit, profiles))


def stage_bundle(ctx: Context) -> None:
    """Đóng gói mã nguồn / kết quả để chuyển giữa máy local và Kaggle/Colab."""
    from modules import cloud

    code_dir, data_root = Path(ctx.cfg["_config_dir"]), Path(ctx.cfg["_data_root"])
    action, target = ctx.args.action, ctx.args.file
    if action == "code":
        result = cloud.pack_code(code_dir, Path(target or code_dir / "labeling_code.zip"))
    elif action == "pack":
        result = cloud.pack_results(data_root, ctx.jsons_dir, Path(target or ctx.jsons_dir.parent / "castle_results.zip"), ctx.args.light,
                                    [s.strip() for s in ctx.args.profiles.split(",") if s.strip()] if ctx.args.profiles else None)
    else:
        if not target:
            raise SystemExit("bundle unpack cần đường dẫn file zip kết quả.")
        result = cloud.unpack_results(ctx.cfg, Path(target), data_root, ctx.jsons_dir, ctx.args.force)
    ctx.log.info("bundle %s: %s", action, result)


def stage_all(ctx: Context) -> None:
    """Chạy nối tiếp trong một tiến trình (tiện cho thử nhanh); mỗi stage tự dọn GPU khi xong."""
    for name, func in (("detect", stage_detect), ("faces", stage_faces), ("reid", stage_reid), ("annotate", stage_annotate), ("merge", stage_merge)):
        ctx.log.info("===== %s =====", name)
        func(ctx)
        free_gpu_memory()


STAGES: dict[str, tuple[Callable[[Context], None], str]] = {
    "detect": (stage_detect, "Florence-2: phát hiện thực thể và cắt crop"),
    "faces": (stage_faces, "InsightFace: phát hiện mặt và trích embedding"),
    "cluster": (stage_cluster, "gom cụm khuôn mặt để đặt tên"),
    "build-gallery": (stage_build_gallery, "dựng identity_gallery từ các cụm đã đặt tên"),
    "reid": (stage_reid, "so mặt với gallery và gán danh tính"),
    "annotate": (stage_annotate, "VLM: sinh mô tả có cấu trúc"),
    "pilot": (stage_pilot, "chạy thử nhiều model trên tập ảnh nhỏ và xuất báo cáo"),
    "merge": (stage_merge, "gộp kết quả thành annotations.jsonl"),
    "render": (stage_render, "vẽ box và toàn bộ nhãn lên ảnh để xem bằng mắt"),
    "ls-export": (stage_ls_export, "xuất task + pre-label cho Label Studio"),
    "ls-import": (stage_ls_import, "nhập bản đã kiểm tra từ Label Studio"),
    "adl": (stage_adl, "nhãn hoạt động cấp ảnh: gom cụm bán tự động có người duyệt"),
    "enrich": (stage_enrich, "làm giàu theo tầng: crop người + vùng cố định"),
    "pipeline": (stage_pipeline, "mọi bước cần GPU của adl + enrich trong một lệnh (Kaggle/Colab)"),
    "bundle": (stage_bundle, "đóng gói mã nguồn / kết quả để chuyển giữa máy local và Kaggle/Colab"),
    "all": (stage_all, "detect -> faces -> reid -> annotate -> merge"),
}


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(DEFAULT_CONFIG), help="đường dẫn config.yaml")
    common.add_argument("--input-images", help="ghi đè paths.input_images")
    common.add_argument("--output-root", help="thư mục chứa output_crops/, output_jsons/, logs/, exports/")
    common.add_argument("--data-root", help="thư mục thay cho labeling/work của stage adl/enrich (hoặc LABELING_DATA_ROOT)")
    common.add_argument("--limit", type=int, help="chỉ xử lý N ảnh đầu (sau khi chia shard)")
    common.add_argument("--shard", help="i/n: chỉ xử lý phần thứ i trong n phần, ví dụ 1/4")
    common.add_argument("--ids-file", help="file mỗi dòng một image_id; chỉ xử lý các ảnh này")
    common.add_argument("--force", action="store_true", help="xoá checkpoint của stage và làm lại")
    common.add_argument("--verbose", action="store_true", help="log mức DEBUG")

    parser = argparse.ArgumentParser(description="Pipeline gán nhãn Castle5000", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True, metavar="STAGE")
    for name, (_, help_text) in STAGES.items():
        p = sub.add_parser(name, parents=[common], help=help_text)
        if name in ("annotate", "all"):
            p.add_argument("--backend", choices=["qwen_local", "api", "openai"], help="ghi đè vlm.backend")
        if name == "pilot":
            p.add_argument("--models", help="tên các model trong pilot.models, cách nhau dấu phẩy (mặc định: tất cả)")
            p.add_argument("--select-only", action="store_true", help="chỉ chọn và ghi danh sách ảnh pilot")
            p.add_argument("--report-only", action="store_true", help="chỉ dựng lại báo cáo từ kết quả đang có")
            p.add_argument("--reselect", action="store_true", help="chọn lại danh sách ảnh pilot")
        if name == "render":
            p.add_argument("--no-objects", action="store_true", help="chỉ vẽ box người, bỏ box đồ vật")
        if name == "ls-export":
            p.add_argument("--push", action="store_true", help="đẩy task lên Label Studio qua SDK")
        if name == "adl":
            p.add_argument("action", choices=["embed", "caption", "propose-labels", "describe", "init", "suggest", "export", "import", "update", "round", "vlm-review", "apply", "status"])
            p.add_argument("--backend", choices=["qwen_local", "api", "openai"], help="caption/propose-labels/describe: ghi đè backend trong adl.label_selection")
            p.add_argument("--labels-file", help="describe/propose-labels: file nhãn thay cho adl.labels_file")
            p.add_argument("--resample", action="store_true", help="caption: chọn lại ảnh mẫu")
            p.add_argument("--profile", help="chạy vòng lặp trên đơn vị khác: people (crop người) hoặc roi-<tên vùng>")
            p.add_argument("export_file", nargs="?", help="file export JSON của Label Studio (cho import)")
            p.add_argument("--what", choices=["suggestions", "seeds", "vlm"], default="suggestions", help="export: gợi ý của vòng này, ảnh hạt giống chưa duyệt, hoặc nhãn VLM đã duyệt")
            p.add_argument("--rounds", type=int, default=1, help="vlm-review: số vòng duyệt -> gom cụm -> gợi ý")
            p.add_argument("--include-vlm", action="store_true", help="apply: lấy cả nhãn VLM đã duyệt")
            p.add_argument("--include-seeds", action="store_true", help="apply: lấy cả ảnh hạt giống chưa ai duyệt")
        if name == "enrich":
            p.add_argument("action", choices=["build", "objects", "identities", "actions", "merge", "demo"])
            p.add_argument("--owl", action="store_true", help="objects: dò thêm vật theo tên bằng OWLv2 (nồi, chảo, ấm, thớt, khăn...)")
            p.add_argument("--rois", help="build: chỉ cắt và mã hoá lại các vùng này (cách nhau dấu phẩy) sau khi sửa khung trong rois.yaml")
        if name == "pipeline":
            p.add_argument("--profiles", help="bước vlm: chỉ các profile này, cách nhau dấu phẩy (people, roi-<tên vùng>)")
            p.add_argument("--steps", help="chỉ chạy các bước này, cách nhau dấu phẩy: embed,build,loops,vlm,fill,attrs,merge (vlm, fill, attrs phải gọi đích danh)")
            p.add_argument("--backend", choices=["qwen_local", "api", "openai"], help="bước vlm: ghi đè adl.vlm_review.backend")
            p.add_argument("--rounds", type=int, default=3, help="bước vlm: số vòng duyệt cho mỗi profile")
            p.add_argument("--vlm-limit", type=int, help="bước vlm: trần lệnh gọi VLM cho mỗi profile")
        if name == "bundle":
            p.add_argument("action", choices=["code", "pack", "unpack"])
            p.add_argument("file", nargs="?", help="file zip đích (code, pack) hoặc nguồn (unpack)")
            p.add_argument("--light", action="store_true", help="pack: chỉ trạng thái vòng lặp + bản ghi gộp, bỏ crop và embedding")
            p.add_argument("--profiles", help="pack: chỉ crop, embedding và trạng thái của các profile này (people, roi-<tên vùng>)")
        if name == "ls-import":
            p.add_argument("export_file", help="file export JSON của Label Studio")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    enable_utf8_console()  # trước khi argparse in --help có dấu tiếng Việt
    args = build_parser().parse_args(argv)
    if not hasattr(args, "backend"):
        args.backend = None
    if not hasattr(args, "profile"):
        args.profile = None
    ctx = Context(args, args.stage)
    try:
        STAGES[args.stage][0](ctx)
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - in lỗi gọn cho người dùng, chi tiết nằm trong log
        ctx.log.error("Stage %s thất bại: %s: %s", args.stage, type(exc).__name__, exc)
        ctx.log.debug("Chi tiết", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

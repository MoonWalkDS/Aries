"""Pipeline gán nhãn Castle5000 (chạy từ gốc repo: `python pipeline/main.py STAGE ...`).

Mỗi stage chạy trong một tiến trình riêng, nên VRAM được trả lại giữa các model và các bước nặng có thể
chạy trên Kaggle/Colab (xem pipeline/notebooks/), phần còn lại chạy ở máy local.

    python pipeline/main.py adl ACTION       # embed | caption | propose-labels | describe | init | suggest |
                                             #   export | import FILE | update | round | vlm-review | apply | status
    python pipeline/main.py enrich ACTION    # build | objects | identities | actions | merge
    python pipeline/main.py pipeline         # mọi bước cần GPU của adl + enrich trong một lệnh (Kaggle/Colab)
    python pipeline/main.py bundle ACTION    # code | pack | unpack FILE: chuyển mã nguồn / kết quả giữa các máy

`enrich merge` gộp mọi nhãn thành annotations/enriched/ và data/exports/castle_dataset.jsonl.
Dùng --force để làm lại một bước từ đầu.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from modules.utils import (
    Checkpoint,
    enable_utf8_console,
    list_images,
    load_config,
    parse_shard,
    read_id_file,
    select_items,
    setup_logging,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"   # config.yaml ở gốc repo


class Context:
    """Cấu hình + các thư mục đã resolve, dùng chung cho mọi stage."""

    def __init__(self, args: argparse.Namespace, stage: str) -> None:
        self.args = args
        self.stage = stage
        self.cfg: dict[str, Any] = load_config(args.config, args.input_images, args.output_root, getattr(args, "data_root", None))
        paths = self.cfg["paths"]
        self.images_dir = Path(paths["input_images"])
        self.jsons_dir = Path(paths["output_jsons"])
        self.logs_dir = Path(paths["logs"])
        self.per_image_dir = self.jsons_dir / "per_image"
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


# --------------------------------------------------------------------------- các stage
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
        result = cloud.pack_code(code_dir, Path(target or data_root.parent / "castle_code.zip"))
    elif action == "pack":
        result = cloud.pack_results(data_root, ctx.jsons_dir, Path(target or data_root.parent / "castle_results.zip"), ctx.args.light,
                                    [s.strip() for s in ctx.args.profiles.split(",") if s.strip()] if ctx.args.profiles else None)
    else:
        if not target:
            raise SystemExit("bundle unpack cần đường dẫn file zip kết quả.")
        result = cloud.unpack_results(ctx.cfg, Path(target), data_root, ctx.jsons_dir, ctx.args.force)
    ctx.log.info("bundle %s: %s", action, result)


STAGES: dict[str, tuple[Callable[[Context], None], str]] = {
    "adl": (stage_adl, "nhãn hoạt động cấp ảnh: gom cụm bán tự động có người duyệt"),
    "enrich": (stage_enrich, "làm giàu theo tầng: crop người + vùng cố định"),
    "pipeline": (stage_pipeline, "mọi bước cần GPU của adl + enrich trong một lệnh (Kaggle/Colab)"),
    "bundle": (stage_bundle, "đóng gói mã nguồn / kết quả để chuyển giữa máy local và Kaggle/Colab"),
}


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(DEFAULT_CONFIG), help="đường dẫn config.yaml")
    common.add_argument("--input-images", help="ghi đè paths.input_images")
    common.add_argument("--output-root", help="thư mục chứa output_crops/, output_jsons/, logs/, exports/")
    common.add_argument("--data-root", help="thư mục thay cho data/work của stage adl/enrich (hoặc LABELING_DATA_ROOT)")
    common.add_argument("--limit", type=int, help="chỉ xử lý N ảnh đầu (sau khi chia shard)")
    common.add_argument("--shard", help="i/n: chỉ xử lý phần thứ i trong n phần, ví dụ 1/4")
    common.add_argument("--ids-file", help="file mỗi dòng một image_id; chỉ xử lý các ảnh này")
    common.add_argument("--force", action="store_true", help="xoá checkpoint của stage và làm lại")
    common.add_argument("--verbose", action="store_true", help="log mức DEBUG")

    parser = argparse.ArgumentParser(description="Pipeline gán nhãn Castle5000", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True, metavar="STAGE")
    for name, (_, help_text) in STAGES.items():
        p = sub.add_parser(name, parents=[common], help=help_text)
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
            p.add_argument("action", choices=["build", "objects", "identities", "actions", "merge"])
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

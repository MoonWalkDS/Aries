"""Tiện ích dùng chung: cấu hình, logging, checkpoint, đọc/ghi file, hình học hộp bao."""
from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
Box = tuple[float, float, float, float]

# Khoá đường dẫn trong config -> biến môi trường ghi đè trực tiếp.
_PATH_ENV = {
    "input_images": "LABELING_INPUT_IMAGES",
    "identity_gallery": "LABELING_IDENTITY_GALLERY",
    "model_cache": "LABELING_MODEL_CACHE",
}
# Các mục config có đường dẫn `work/...` được dời theo LABELING_DATA_ROOT.
_DATA_SECTIONS = ("adl", "enrich", "adl_profiles")
# Các thư mục đầu ra được gom dưới LABELING_OUTPUT_ROOT nếu biến này được đặt.
_OUTPUT_KEYS = ("output_crops", "output_jsons", "logs", "exports")


# --------------------------------------------------------------------------- cấu hình
def load_config(
    config_path: str | Path,
    input_images: Optional[str] = None,
    output_root: Optional[str] = None,
    data_root: Optional[str] = None,
) -> dict[str, Any]:
    """Đọc config.yaml và đổi mọi đường dẫn trong ``paths`` thành tuyệt đối.

    Thứ tự ưu tiên: tham số CLI > biến môi trường > giá trị trong file.
    Đường dẫn tương đối được tính từ thư mục chứa file config, nên chạy từ thư mục nào cũng được.
    """
    config_path = Path(config_path).resolve()
    with open(config_path, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}
    base = config_path.parent
    paths: dict[str, Any] = cfg.setdefault("paths", {})

    output_root = output_root or os.getenv("LABELING_OUTPUT_ROOT")
    if output_root:
        for key in _OUTPUT_KEYS:
            paths[key] = str(Path(output_root) / key)
    for key, env_name in _PATH_ENV.items():
        if os.getenv(env_name):
            paths[key] = os.environ[env_name]
    if input_images:
        paths["input_images"] = input_images

    for key, value in list(paths.items()):
        if value is None:
            continue
        p = Path(str(value)).expanduser()
        paths[key] = str(p if p.is_absolute() else (base / p).resolve())

    # Dữ liệu dẫn xuất của stage adl/enrich mặc định nằm ở <config>/work/... Trên Kaggle/Colab thư mục
    # mã nguồn có thể chỉ đọc hoặc không được lưu lại, nên cho phép dời cả cây đó sang chỗ khác.
    data_root = data_root or os.getenv("LABELING_DATA_ROOT")
    cfg["_data_root"] = str(Path(data_root).expanduser().resolve()) if data_root else str((base / "work").resolve())
    if data_root:
        for section in _DATA_SECTIONS:
            if section in cfg:
                cfg[section] = _rebase_data(cfg[section], cfg["_data_root"])

    cfg["_config_dir"] = str(base)
    return cfg


def _rebase_data(value: Any, data_root: str) -> Any:
    """Đổi mọi chuỗi ``work/...`` trong một cấu trúc config lồng nhau thành ``<data_root>/...``."""
    if isinstance(value, str) and value.startswith("work/"):
        return f"{data_root.replace(os.sep, '/')}/{value[len('work/'):]}"
    if isinstance(value, dict):
        return {k: _rebase_data(v, data_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_rebase_data(v, data_root) for v in value]
    return value


def resolve_from_config(cfg: dict[str, Any], value: str) -> Path:
    """Đổi một đường dẫn tương đối nằm ngoài mục ``paths`` thành tuyệt đối."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (Path(cfg["_config_dir"]) / p).resolve()


# --------------------------------------------------------------------------- logging
def enable_utf8_console() -> None:
    """Cho phép in tiếng Việt khi console Windows đang dùng bảng mã cũ (cp1252...)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


def setup_logging(log_dir: str | Path, stage: str, verbose: bool = False) -> logging.Logger:
    """Ghi log ra console và ra ``logs/<stage>.log`` (UTF-8, nối tiếp qua các lần chạy)."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    enable_utf8_console()

    logger = logging.getLogger("labeling")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(log_dir / f"{stage}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("labeling")


# --------------------------------------------------------------------------- checkpoint
class Checkpoint:
    """Danh sách ID đã xử lý xong của một stage, lưu trong file ``completed_<stage>.log``.

    Mỗi ID một dòng, ghi nối và flush ngay sau từng ảnh. Nếu tiến trình bị ngắt, lần chạy sau
    đọc lại file và bỏ qua các ảnh đã có. Một dòng ghi dở ở cuối file (do mất điện giữa chừng)
    không khớp ID nào nên ảnh đó chỉ đơn giản được xử lý lại.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] = set()
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self._done = {line.strip() for line in f if line.strip()}

    def __contains__(self, item_id: str) -> bool:
        return item_id in self._done

    def __len__(self) -> int:
        return len(self._done)

    def mark(self, item_id: str) -> None:
        if item_id in self._done:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(item_id + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._done.add(item_id)

    def reset(self) -> None:
        self._done.clear()
        if self.path.exists():
            self.path.unlink()


# --------------------------------------------------------------------------- danh sách ảnh
def list_images(image_dir: str | Path) -> list[Path]:
    """Mọi file ảnh trong thư mục, sắp theo tên để thứ tự (và việc chia shard) ổn định."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Không thấy thư mục ảnh: {image_dir}")
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def image_id_of(path: str | Path) -> str:
    """'data/images/10_1528.jpg' -> '10_1528'."""
    return Path(path).stem


def parse_shard(spec: Optional[str]) -> Optional[tuple[int, int]]:
    """'2/4' -> (2, 4): shard thứ 2 trong 4 (đánh số từ 1)."""
    if not spec:
        return None
    try:
        index, total = (int(x) for x in spec.split("/"))
    except ValueError as exc:
        raise ValueError(f"--shard phải có dạng i/n, ví dụ 1/4; nhận được {spec!r}") from exc
    if not (1 <= index <= total):
        raise ValueError(f"--shard {spec}: cần 1 <= i <= n")
    return index, total


def select_items(
    items: Sequence[Path],
    shard: Optional[tuple[int, int]] = None,
    limit: Optional[int] = None,
    only_ids: Optional[set[str]] = None,
) -> list[Path]:
    """Lọc theo danh sách ID, rồi chia shard (xen kẽ để mỗi shard trải đều các giờ), rồi cắt limit."""
    selected = list(items)
    if only_ids is not None:
        selected = [p for p in selected if image_id_of(p) in only_ids]
    if shard:
        index, total = shard
        selected = selected[index - 1::total]
    if limit is not None:
        selected = selected[:limit]
    return selected


def read_id_file(path: str | Path) -> set[str]:
    """File văn bản mỗi dòng một image_id (chấp nhận cả tên file có đuôi)."""
    with open(path, encoding="utf-8") as f:
        return {Path(line.strip()).stem for line in f if line.strip() and not line.startswith("#")}


# --------------------------------------------------------------------------- đọc/ghi
def write_json(path: str | Path, data: Any) -> None:
    """Ghi JSON nguyên tử: ghi ra file tạm rồi đổi tên, tránh file hỏng khi bị ngắt giữa chừng."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


# --------------------------------------------------------------------------- hình học
def clamp_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def pad_box(box: Box, margin: float, width: int, height: int) -> Box:
    """Nới box thêm ``margin`` (tỉ lệ theo cạnh box) về mỗi phía rồi kẹp trong biên ảnh."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    return clamp_box((x1 - dx, y1 - dy, x2 + dx, y2 + dy), width, height)


def box_area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def contains_point(box: Box, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def spatial_position(box: Box, width: int, height: int) -> str:
    """Mô tả vị trí tâm box theo lưới 3x3, bằng tiếng Việt."""
    cx = (box[0] + box[2]) / 2 / max(width, 1)
    cy = (box[1] + box[3]) / 2 / max(height, 1)
    horizontal = "bên trái" if cx < 1 / 3 else ("ở giữa" if cx < 2 / 3 else "bên phải")
    vertical = "phía trên" if cy < 1 / 3 else ("ngang giữa" if cy < 2 / 3 else "phía dưới")
    return f"{horizontal}, {vertical}"


# --------------------------------------------------------------------------- GPU
def free_gpu_memory() -> None:
    """Thu gom rác và trả bộ nhớ GPU chưa dùng về cho CUDA. Không làm gì nếu không có torch."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # torch không cài hoặc bị chặn: bỏ qua
        pass

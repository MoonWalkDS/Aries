"""Dựng embedding ảnh cho stage ``adl`` (``python main.py adl embed``).

Mỗi ảnh cho hai vector từ cùng một model CLIP/SigLIP:

- ``global``: cả khung hình.
- ``focus``: vùng có người/chuyển động, cắt ra rồi mã hoá riêng. Camera Castle cố định nên nền của
  mọi ảnh gần như giống nhau và lấn át vector toàn cảnh; vùng chuyển động tìm bằng cách trừ nền
  (nền = trung vị theo pixel của các ảnh cùng nhóm thời gian). Ảnh không có chuyển động thì
  ``focus`` bằng ``global``.

Vector cuối = chuẩn hoá(global + focus_weight * focus). Kèm theo là ``motion``: tỉ lệ diện tích
khung hình có chuyển động, dùng để nhận ảnh bếp trống mà không cần model.

Kết quả trong thư mục ``adl.embed.out_dir``: ``embeddings.npy``, ``ids.json``, ``motion.npy``,
``boxes.json`` (hộp focus theo tỉ lệ 0..1), ``build_stats.json``.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

from .utils import get_logger, image_id_of, write_json

RelBox = tuple[float, float, float, float]   # x1, y1, x2, y2 theo tỉ lệ 0..1
FULL: RelBox = (0.0, 0.0, 1.0, 1.0)


# --------------------------------------------------------------------------- trừ nền
def group_key(image_id: str) -> str:
    return image_id.split("_", 1)[0]


def load_small(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Ảnh xám cỡ nhỏ (float32, 0..255) để trừ nền; mờ nhẹ nhờ thu nhỏ nên ít nhạy nhiễu."""
    with Image.open(path) as im:
        im.draft("L", (size[0] * 2, size[1] * 2))   # JPEG: giải mã thẳng ở cỡ nhỏ, nhanh hơn nhiều
        return np.asarray(im.convert("L").resize(size, Image.BILINEAR), dtype=np.float32)


def background_of(frames: np.ndarray) -> np.ndarray:
    """Nền = trung vị theo pixel. Người đi lại chỉ che mỗi pixel trong số ít khung nên bị loại."""
    return np.median(frames, axis=0)


def motion_mask(frame: np.ndarray, background: np.ndarray, threshold: float) -> np.ndarray:
    return np.abs(frame - background) > threshold


def focus_box(mask: np.ndarray, min_fraction: float, pad: float, grid: int = 8) -> tuple[Optional[RelBox], float]:
    """Hộp bao vùng chuyển động và tỉ lệ diện tích chuyển động.

    Mặt nạ được gộp về lưới ô thô; chỉ ô có đủ pixel đổi mới tính, nên vài pixel nhiễu lẻ không kéo
    hộp ra cả khung. Trả ``None`` khi chuyển động quá ít (ảnh không có người).
    """
    h, w = mask.shape
    fraction = float(mask.mean())
    gh, gw = h // grid, w // grid
    cells = mask[: gh * grid, : gw * grid].reshape(gh, grid, gw, grid).mean(axis=(1, 3)) > 0.25
    if fraction < min_fraction or not cells.any():
        return None, fraction
    rows, cols = np.where(cells.any(axis=1))[0], np.where(cells.any(axis=0))[0]
    x1, x2 = cols[0] * grid / w, (cols[-1] + 1) * grid / w
    y1, y2 = rows[0] * grid / h, (rows[-1] + 1) * grid / h
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return (max(0.0, x1 - px), max(0.0, y1 - py), min(1.0, x2 + px), min(1.0, y2 + py)), fraction


def compute_focus(
    image_paths: Sequence[Path],
    size: tuple[int, int],
    threshold: float,
    min_fraction: float,
    pad: float,
    max_background_frames: int = 200,
    workers: int = 8,
) -> tuple[list[RelBox], np.ndarray]:
    """Hộp focus và tỉ lệ chuyển động cho mọi ảnh; nền tính riêng cho từng nhóm thời gian."""
    with ThreadPoolExecutor(workers) as pool:
        small = np.stack(list(pool.map(lambda p: load_small(p, size), image_paths)))
    groups: dict[str, list[int]] = {}
    for i, path in enumerate(image_paths):
        groups.setdefault(group_key(image_id_of(path)), []).append(i)
    boxes: list[RelBox] = [FULL] * len(image_paths)
    motion = np.zeros(len(image_paths), dtype=np.float32)
    for members in groups.values():
        step = max(1, len(members) // max_background_frames)
        background = background_of(small[members[::step]])
        for i in members:
            box, fraction = focus_box(motion_mask(small[i], background, threshold), min_fraction, pad)
            motion[i] = fraction
            if box is not None:
                boxes[i] = box
    return boxes, motion


def crop_relative(image: Image.Image, box: RelBox) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = box
    return image.crop((int(x1 * w), int(y1 * h), max(int(x1 * w) + 1, int(x2 * w)), max(int(y1 * h) + 1, int(y2 * h))))


def combine(global_emb: np.ndarray, focus_emb: np.ndarray, focus_weight: float) -> np.ndarray:
    mixed = global_emb + focus_weight * focus_emb
    return (mixed / np.maximum(np.linalg.norm(mixed, axis=1, keepdims=True), 1e-12)).astype(np.float32)


# --------------------------------------------------------------------------- mã hoá
class ClipImageEncoder:
    """Mã hoá ảnh bằng open_clip; dùng GPU và nửa độ chính xác khi có."""

    def __init__(self, model: str, pretrained: str, device: str = "auto") -> None:
        import open_clip
        import torch

        self._torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=self.device)
        self.model.eval()
        self.half = self.device == "cuda"
        if self.half:
            self.model.half()

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        batch = torch.stack([self.preprocess(im) for im in images]).to(self.device)
        if self.half:
            batch = batch.half()
        with torch.inference_mode():
            features = self.model.encode_image(batch).float()
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()


def run_embed(cfg: dict[str, Any], image_paths: Sequence[Path], out_dir: Path, encoder: Optional[Any] = None) -> dict[str, Any]:
    """Dựng embedding cho mọi ảnh và ghi vào ``out_dir``. ``encoder`` chỉ để test thay model thật."""
    adl, emb_cfg = cfg["adl"], cfg["adl"]["embed"]
    focus_cfg, log = emb_cfg["focus"], get_logger()
    if not image_paths:
        raise RuntimeError("Không có ảnh nào để mã hoá.")
    started = time.perf_counter()
    use_focus = bool(focus_cfg.get("enabled", True))
    if use_focus:
        boxes, motion = compute_focus(
            image_paths, tuple(focus_cfg["size"]), float(focus_cfg["threshold"]),
            float(focus_cfg["min_fraction"]), float(focus_cfg["pad"]),
        )
        log.info("embed: trừ nền xong sau %.0fs, %d/%d ảnh có vùng focus", time.perf_counter() - started, sum(b != FULL for b in boxes), len(boxes))
    else:
        boxes, motion = [FULL] * len(image_paths), np.zeros(len(image_paths), dtype=np.float32)

    encoder = encoder or ClipImageEncoder(adl["clip"]["model"], adl["clip"]["pretrained"], emb_cfg.get("device", "auto"))
    batch_size = int(emb_cfg["batch_size"])
    global_parts: list[np.ndarray] = []
    focus_parts: list[np.ndarray] = []
    with ThreadPoolExecutor(int(emb_cfg.get("workers", 4))) as pool:
        for start in range(0, len(image_paths), batch_size):
            chunk = range(start, min(start + batch_size, len(image_paths)))
            images = list(pool.map(lambda i: Image.open(image_paths[i]).convert("RGB"), chunk))
            global_emb = encoder.encode(images)
            global_parts.append(global_emb)
            if use_focus:
                # Chỉ mã hoá lại ảnh có hộp focus khác cả khung; ảnh còn lại dùng luôn vector toàn cảnh.
                focus_emb = global_emb.copy()
                todo = [k for k, i in enumerate(chunk) if boxes[i] != FULL]
                if todo:
                    focus_emb[todo] = encoder.encode([crop_relative(images[k], boxes[chunk[k]]) for k in todo])
                focus_parts.append(focus_emb)
            if (start // batch_size) % 20 == 0:
                log.info("embed: %d/%d ảnh", chunk[-1] + 1, len(image_paths))
    global_emb = np.concatenate(global_parts)
    weight = float(focus_cfg.get("weight", 1.0)) if use_focus else 0.0
    final = combine(global_emb, np.concatenate(focus_parts), weight) if use_focus else global_emb.astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", final)
    np.save(out_dir / "embeddings_global.npy", global_emb.astype(np.float32))
    np.save(out_dir / "motion.npy", motion)
    write_json(out_dir / "ids.json", [p.name for p in image_paths])
    write_json(out_dir / "boxes.json", {image_id_of(p): [round(v, 4) for v in b] for p, b in zip(image_paths, boxes)})
    stats = {
        "num_indexed": len(image_paths),
        "dim": int(final.shape[1]),
        "model": adl["clip"]["model"],
        "pretrained": adl["clip"]["pretrained"],
        "device": getattr(encoder, "device", "?"),
        "focus": use_focus,
        "focus_weight": weight,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    write_json(out_dir / "build_stats.json", stats)
    return {**stats, "out_dir": str(out_dir), "with_focus_box": int(sum(b != FULL for b in boxes))}

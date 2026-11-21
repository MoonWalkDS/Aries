"""Hằng số và hàm dùng chung của phần truy vấn: tên model, đường dẫn dữ liệu, tách câu đề, khớp tên danh tính."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "data" / "exports"
CLIP_MODEL, CLIP_PRETRAINED = "ViT-SO400M-16-SigLIP2-384", "webli"   # trùng config.yaml adl.clip
MT_MODEL = "Helsinki-NLP/opus-mt-vi-en"


def split_sentences(text: str) -> list[str]:
    """Đề DRES có nhiều dòng (gợi ý tăng dần): dùng dòng cuối, dài nhất; tách thành câu."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    last = max(lines, key=len) if lines else ""
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", last) if s.strip()]


def _fold(text: str) -> str:
    """Chữ thường, bỏ dấu tiếng Việt, để khớp tên gõ thiếu dấu."""
    text = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def names_in(query: str, names: list[str]) -> list[str]:
    """Tên danh tính xuất hiện nguyên từ trong câu (không phân biệt hoa thường, dấu)."""
    q = f" {_fold(re.sub(r'[^\w]+', ' ', query))} "
    return [n for n in names if f" {_fold(re.sub(r'[^\w]+', ' ', n))} " in q]


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)

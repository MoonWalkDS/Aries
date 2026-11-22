"""Hằng số và hàm dùng chung của hệ thống truy vấn: tên model, đường dẫn dữ liệu, tách câu đề, khớp tên danh tính."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = REPO_ROOT / "data" / "exports"
SIGLIP_MODEL, SIGLIP_WEIGHTS = "ViT-SO400M-16-SigLIP2-384", "webli"   # trùng labeling/config.yaml adl.clip
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-vi-en"


def split_sentences(text: str) -> list[str]:
    """Đề DRES có nhiều dòng (gợi ý tăng dần): dùng dòng dài nhất, tách thành câu."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    longest = max(lines, key=len) if lines else ""
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+", longest) if s.strip()]


def remove_accents(text: str) -> str:
    """Chữ thường, bỏ dấu tiếng Việt ("Tủ lạnh" -> "tu lanh"), để khớp cả khi gõ thiếu dấu."""
    text = unicodedata.normalize("NFD", text.lower()).replace("đ", "d")
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def find_names(text: str, known_names: list[str]) -> list[str]:
    """Tên danh tính xuất hiện nguyên từ trong câu (không phân biệt hoa thường, dấu)."""
    padded = f" {remove_accents(re.sub(r'[^\w]+', ' ', text))} "
    return [name for name in known_names if f" {remove_accents(re.sub(r'[^\w]+', ' ', name))} " in padded]


def zscore(values: np.ndarray) -> np.ndarray:
    """Chuẩn hoá về trung bình 0, độ lệch 1, để cộng được điểm của các kênh khác thang đo."""
    return (values - values.mean()) / (values.std() + 1e-9)

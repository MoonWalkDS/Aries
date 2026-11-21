"""Tìm ảnh Castle5000 theo câu mô tả tiếng Việt (đề DRES).

Điểm của một ảnh = tổng các điểm đã chuẩn hoá (z-score trên 5.000 ảnh):
  1. cả câu (đã dịch sang tiếng Anh) so với embedding ảnh SigLIP2 (toàn khung + vùng có người);
  2. từng câu nhỏ so với embedding ảnh (vùng có người và toàn khung), lấy trung bình;
  3. từng câu nhỏ so với crop người khớp nhất trong ảnh (câu hay tả một người cụ thể);
  4. thưởng khi câu nhắc tên danh tính có trong ảnh (tên giữ nguyên, không dịch: "Linh" bị dịch thành "Ling").
Chỉ đọc dataset/ và exports/ (`python main.py enrich merge` sinh exports/embeddings/).
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"
CLIP_MODEL, CLIP_PRETRAINED = "ViT-SO400M-16-SigLIP2-384", "webli"   # trùng labeling/config.yaml adl.clip
MT_MODEL = "Helsinki-NLP/opus-mt-vi-en"
NAME_BONUS = 1.5          # cộng vào tổng z-score cho mỗi tên danh tính khớp


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


class Engine:
    def __init__(self, device: str | None = None) -> None:
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        emb = EXPORTS / "embeddings"
        self.ids = [Path(n).stem for n in json.loads((emb / "image_ids.json").read_text())]
        self.row = {i: r for r, i in enumerate(self.ids)}
        self.focus = np.load(emb / "images.npy")
        self.whole = np.load(emb / "images_global.npy")
        people = np.load(emb / "people.npy").astype(np.float32)
        self.people = people / np.linalg.norm(people, axis=1, keepdims=True)
        pids = [Path(n).stem for n in json.loads((emb / "people_ids.json").read_text())]
        self.person_image = np.array([self.row[p.rsplit("_", 1)[0]] for p in pids])
        self.records = {}
        with open(EXPORTS / "castle_dataset.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                self.records[r["image_id"]] = r
        self.names_of = {i: {(p.get("identity") or {}).get("name") for p in r["people"]} - {None} for i, r in self.records.items()}
        self.all_names = sorted(set().union(*self.names_of.values()), key=len, reverse=True)
        self._load_models()

    def _load_models(self) -> None:
        import open_clip
        from transformers import MarianMTModel, MarianTokenizer

        self.clip, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=self.device)
        self.clip.eval()
        if self.device == "cuda":
            self.clip.half()   # float32 không vừa GPU 6 GB
        self.clip_tok = open_clip.get_tokenizer(CLIP_MODEL)
        self.mt_tok = MarianTokenizer.from_pretrained(MT_MODEL)
        self.mt = MarianMTModel.from_pretrained(MT_MODEL).to(self.device).eval()

    def translate(self, sentences: list[str]) -> list[str]:
        import torch

        with torch.no_grad():
            out = self.mt.generate(**self.mt_tok(sentences, return_tensors="pt", padding=True).to(self.device), max_length=128)
        return self.mt_tok.batch_decode(out, skip_special_tokens=True)

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            f = self.clip.encode_text(self.clip_tok(texts).to(self.device)).float().cpu().numpy()
        return f / np.linalg.norm(f, axis=1, keepdims=True)

    def search(self, query: str, top: int = 200) -> dict:
        sentences = split_sentences(query)
        if not sentences:
            return {"sentences": [], "english": [], "names": [], "results": []}
        english = self.translate(sentences)
        vec = self.encode([" ".join(english)] + english)
        whole, parts = vec[0], vec[1:]
        score = _z(self.whole @ whole) + _z(self.focus @ whole) + _z((self.focus @ parts.T).mean(1)) + _z((self.whole @ parts.T).mean(1))
        best_person = np.full((len(parts), len(self.ids)), -1.0, dtype=np.float32)
        sims = parts @ self.people.T
        for k in range(len(parts)):
            np.maximum.at(best_person[k], self.person_image, sims[k])
        has_person = best_person[0] > -1
        person_score = np.where(has_person, best_person.mean(0), best_person[:, has_person].mean() if has_person.any() else 0)
        score += _z(person_score)
        names = names_in(" ".join(sentences), self.all_names)
        for n in names:
            score += NAME_BONUS * np.array([n in self.names_of.get(i, ()) for i in self.ids])
        order = np.argsort(-score)[:top]
        return {"sentences": sentences, "english": english, "names": names,
                "results": [{"id": self.ids[j], "score": round(float(score[j]), 3)} for j in order]}

    def rank_of(self, query: str, answers: set[str]) -> int:
        """Hạng (từ 1) của ảnh đúng đầu tiên; dùng cho đánh giá."""
        res = self.search(query, top=len(self.ids))["results"]
        return next(k for k, r in enumerate(res, 1) if r["id"] in answers)

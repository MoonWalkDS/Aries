"""So khớp text không cần model: tách từ, BM25 (điểm từ khoá), ghép một-một cụm-người với người trong ảnh."""
from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from common import remove_accents


def tokenize(text: str) -> list[str]:
    """Từ đã bỏ dấu cộng cặp từ liền nhau ("tu_lanh"), để cụm hai âm tiết khớp như một từ."""
    words = re.findall(r"\w+", remove_accents(text))
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class BM25:
    """Điểm từ khoá BM25 của một câu hỏi với từng tài liệu."""

    def __init__(self, documents: list[list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.num_documents = len(documents)
        lengths = np.array([len(d) for d in documents], dtype=np.float32)
        length_norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1e-9))
        postings: dict[str, tuple[list[int], list[float]]] = {}
        for doc_index, document in enumerate(documents):
            for token, count in Counter(document).items():
                rows, weights = postings.setdefault(token, ([], []))
                rows.append(doc_index)
                weights.append(count * (k1 + 1) / (count + length_norm[doc_index]))
        self.postings = {token: (np.array(rows), np.array(weights, dtype=np.float32)
                                 * math.log(1 + (self.num_documents - len(rows) + 0.5) / (len(rows) + 0.5)))
                         for token, (rows, weights) in postings.items()}

    def score(self, query_tokens: list[str]) -> np.ndarray:
        """Mảng điểm (num_documents,)."""
        scores = np.zeros(self.num_documents, dtype=np.float32)
        for token in set(query_tokens):
            if token in self.postings:
                rows, weights = self.postings[token]
                scores[rows] += weights
        return scores


@lru_cache(maxsize=None)
def _permutations(num_columns: int, num_rows: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(num_columns), num_rows)), dtype=np.int16).reshape(-1, num_rows)


def assign_one_to_one(scores: np.ndarray, unmatched_score: float, candidates_per_clause: int = 3) -> tuple[float, list[int | None]]:
    """Ghép mỗi cụm (hàng) với một người (cột) khác nhau sao cho tổng điểm cao nhất.

    Cụm không có người để ghép nhận `unmatched_score`. Mỗi cụm chỉ xét `candidates_per_clause` người khớp nhất
    (≤ 4 cụm × 3 = 12 cột), nên vét cạn mọi hoán vị vẫn nhanh. Trả (tổng điểm, chỉ số người cho từng cụm hoặc None).
    """
    num_clauses = scores.shape[0]
    if num_clauses == 0:
        return 0.0, []
    candidates = sorted({int(j) for row in scores for j in np.argsort(-row)[:candidates_per_clause]})
    # thêm num_clauses cột "không có ai", mỗi cột dùng được một lần
    table = np.hstack([scores[:, candidates], np.full((num_clauses, num_clauses), unmatched_score)])
    permutations = _permutations(table.shape[1], num_clauses)
    totals = table[np.arange(num_clauses), permutations].sum(1)
    best = permutations[int(totals.argmax())]
    return float(totals.max()), [candidates[j] if j < len(candidates) else None for j in best]

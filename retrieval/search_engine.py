"""Tìm ảnh Castle5000 bằng so khớp Metadata (3 tầng) tích hợp SLM NuExtract-1.5-smol.
Tương thích hoàn toàn giao diện web_app.py và index.html.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import REPO_ROOT, find_names, remove_accents
from query_parser import parse_query

logger = logging.getLogger("retrieval.search_engine")


class EmbeddingModels:
    """Stub / Fallback models helper for compatibility."""
    def __init__(self, device: str | None = None) -> None:
        self.device = device or "cpu"

    def translate(self, texts: list[str], beams: int | None = None) -> list[str]:
        return texts

    def encode(self, texts: list[str]):
        import torch
        return torch.zeros((len(texts), 512))


class SearchEngine:
    """Động cơ tìm kiếm so khớp metadata 3 tầng:
    Stage 1: Lọc số người (so_nguoi)
    Stage 2: Lọc vật thể / trạng thái (vat_the_trang_thai)
    Stage 3: Chấm điểm đặc trưng người bằng Hungarian Bipartite Matching + TF-IDF Cosine
    """

    def __init__(
        self,
        device: str | None = None,
        index_dir: Path | None = None,
        models: Any = None,
        image_embeddings_dir: Path | None = None,
        metadata_path: Path | None = None,
    ) -> None:
        if metadata_path is None:
            metadata_path = REPO_ROOT / "metadata" / "metadata.json"
            if not metadata_path.exists():
                metadata_path = Path(r"C:\Aries\metadata\metadata.json")

        self.metadata: dict[str, dict] = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        # Setup image_ids and records for web_app compatibility
        self.image_ids = []
        self.records = {}
        for raw_id, item in self.metadata.items():
            clean_id = raw_id.replace(".jpg", "")
            self.image_ids.append(clean_id)

            people = []
            for i, p in enumerate(item.get("danh_sach_nguoi", [])):
                desc = p if isinstance(p, str) else p.get("mo_ta", "")
                is_male = any(w in desc.lower() for w in ("nam", "đàn ông", "anh", "ông"))
                people.append({
                    "id": f"p{i+1}",
                    "box": [100 + i * 200, 100, 300 + i * 200, 600],
                    "identity": {"name": None},
                    "action": {"label_vi": desc},
                    "attributes": {"gender": "nam" if is_male else "nữ"},
                    "look": desc,
                })

            rec = {
                "image_id": clean_id,
                "people": people,
                "rois": {},
                "objects": [{"label": str(obj)} for obj in item.get("vat_the_trang_thai", [])],
            }
            self.records[clean_id] = rec
            self.records[f"{clean_id}.jpg"] = rec

        self.known_names: list[str] = []
        self._init_tfidf()

        # Preload mô hình NuExtract SLM vào VRAM ngay khi khởi tạo
        try:
            from query_parser import get_slm_parser
            logger.info("Preloading NuExtract SLM model into VRAM...")
            get_slm_parser().load()
        except Exception as e:
            logger.warning("Preload NuExtract SLM failed: %s", e)

        logger.info("SearchEngine initialized with %d metadata images", len(self.image_ids))

    def _init_tfidf(self) -> None:
        """Khởi tạo TF-IDF vectorizer trên toàn bộ mô tả người trong metadata."""
        corpus: list[str] = []
        self._action_texts: dict[tuple[str, int], str] = {}

        for raw_id, item in self.metadata.items():
            clean_id = raw_id.replace(".jpg", "")
            for p_idx, person in enumerate(item.get("danh_sach_nguoi", [])):
                if isinstance(person, str):
                    text = person
                else:
                    text = f"{person.get('hanh_dong', '')} {person.get('mo_ta', '')} {person.get('vi_tri', '')} {person.get('mau_ao', '')}".strip()
                self._action_texts[(clean_id, p_idx)] = text
                corpus.append(text)

        # Domain keywords anchors
        corpus.extend([
            "người đàn ông áo vest đen đứng sau đảo bếp",
            "người phụ nữ đứng gần bếp",
            "người đàn anh mặc áo xanh đứng gần tủ lạnh",
            "lau tay khăn lau trắng bồn rửa",
            "với vào tủ bếp trên cao mở tủ",
            "đứng chống nạnh quan sát nấu ăn",
        ])

        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf.fit(corpus)

        self._action_vectors: dict[tuple[str, int], Any] = {}
        for key, text in self._action_texts.items():
            self._action_vectors[key] = self.tfidf.transform([text])

    def _filter_stage1_so_nguoi(self, candidate_ids: list[str], target_count: Optional[int]) -> list[str]:
        if target_count is None:
            return candidate_ids

        exact = []
        for cid in candidate_ids:
            item = self.metadata.get(cid) or self.metadata.get(f"{cid}.jpg") or {}
            if item.get("so_nguoi") == target_count:
                exact.append(cid)
        if exact:
            return exact

        # Relaxation +/- 1
        relaxed = []
        for cid in candidate_ids:
            item = self.metadata.get(cid) or self.metadata.get(f"{cid}.jpg") or {}
            if abs(item.get("so_nguoi", 0) - target_count) <= 1:
                relaxed.append(cid)
        if relaxed:
            return relaxed

        return candidate_ids

    def _filter_stage2_vat_the(self, candidate_ids: list[str], req_objects: list[str]) -> tuple[list[str], dict[str, float]]:
        if not req_objects:
            return candidate_ids, {cid: 1.0 for cid in candidate_ids}

        valid_candidates = []
        scores = {}
        req_lower = [str(x).lower().strip() for x in req_objects if str(x).strip()]

        for cid in candidate_ids:
            item = self.metadata.get(cid) or self.metadata.get(f"{cid}.jpg") or {}
            meta_objs = [str(x).lower().strip() for x in item.get("vat_the_trang_thai", [])]

            score = 1.0
            conflict = False

            for req in req_lower:
                matched = any(req in m or m in req for m in meta_objs)
                if not matched:
                    conflict = True
                    score -= 0.5
                else:
                    score += 0.1

            scores[cid] = max(0.0, min(1.0, score))
            if not conflict:
                valid_candidates.append(cid)

        if not valid_candidates:
            valid_candidates = candidate_ids

        return valid_candidates, scores

    def _score_stage3_people(
        self, candidate_ids: list[str], query_people: list[Any], landmark_scores: dict[str, float]
    ) -> list[tuple[str, float]]:
        if not query_people:
            res = [(cid, landmark_scores.get(cid, 1.0)) for cid in candidate_ids]
            res.sort(key=lambda x: x[1], reverse=True)
            return res

        m = len(query_people)
        results: list[tuple[str, float]] = []

        for cid in candidate_ids:
            item = self.metadata.get(cid) or self.metadata.get(f"{cid}.jpg") or {}
            cand_people = item.get("danh_sach_nguoi", [])
            n = len(cand_people)

            if n == 0:
                results.append((cid, 0.0))
                continue

            cost_matrix = np.zeros((m, n), dtype=np.float32)
            for i, q_person in enumerate(query_people):
                q_text = q_person if isinstance(q_person, str) else q_person.get("text", "") or q_person.get("mo_ta", "")
                q_vec = self.tfidf.transform([q_text])
                for j in range(n):
                    cand_vec = self._action_vectors.get((cid, j))
                    if cand_vec is None:
                        c_text = cand_people[j] if isinstance(cand_people[j], str) else str(cand_people[j])
                        cand_vec = self.tfidf.transform([c_text])
                    sim = float(cosine_similarity(q_vec, cand_vec)[0][0])
                    cost_matrix[i, j] = sim

            row_ind, col_ind = linear_sum_assignment(-cost_matrix)
            matched_scores = cost_matrix[row_ind, col_ind]
            person_score = float(np.sum(matched_scores) / max(m, 1))

            lm_score = landmark_scores.get(cid, 1.0)
            final_score = 0.85 * person_score + 0.15 * lm_score
            results.append((cid, float(final_score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search(self, query: str, top: int = 200) -> dict:
        parsed = parse_query(query, self.known_names)
        candidates = list(self.image_ids)

        # Stage 1: so_nguoi
        st1_candidates = self._filter_stage1_so_nguoi(candidates, parsed.get("so_nguoi"))

        # Stage 2: vat_the_trang_thai
        st2_candidates, lm_scores = self._filter_stage2_vat_the(st1_candidates, parsed.get("vat_the_trang_thai", []))

        # Stage 3: danh_sach_nguoi
        ranked = self._score_stage3_people(st2_candidates, parsed.get("danh_sach_nguoi", []), lm_scores)

        results = []
        for img_id, score in ranked[:top]:
            results.append({
                "id": img_id,
                "score": round(float(score), 3),
                "explain": {
                    "pairs": [{"tag": None, "person": "p1"}],
                    "parts": {
                        "count": 1.0,
                        "name": 0,
                        "scene": round(float(lm_scores.get(img_id, 1.0)), 2),
                        "image": 1.0,
                        "person": round(float(score), 2),
                        "final": round(float(score), 2),
                    },
                },
            })

        return {
            "sentences": parsed.get("sentences", [query]),
            "english": parsed.get("vat_the_trang_thai", []),
            "names": parsed.get("names", []),
            "parsed": parsed,
            "layers": [len(self.image_ids), len(st1_candidates), len(st2_candidates)],
            "results": results,
        }

    def rank_of(self, query: str, answers: set[str]) -> int:
        results = self.search(query, top=len(self.image_ids))["results"]
        for rank, r in enumerate(results, 1):
            if r["id"] in answers or f"{r['id']}.jpg" in answers:
                return rank
        return 9999

"""Tìm ảnh Castle5000 bằng text (caption, mô tả của thầy, nhãn người) cộng kênh ảnh SigLIP2: 3 lớp lọc mềm.

Thiết kế: docs/retrieval-design.md. Cần dựng chỉ mục trước: tools/build_index.bat (retrieval/build_search_index.py).
  Bước 0  parse_query + _encode_query: tách đề thành cụm-người / nhóm / cảnh, dịch sang tiếng Anh, nhúng
  Lớp 1   layer1, cả 5.000 ảnh: số người (detector, thầy), tên danh tính, mô tả cảnh, ảnh   -> giữ KEEP_AFTER_LAYER1 ảnh
  Lớp 2   layer2: ghép một-một từng cụm-người của đề với một người trong ảnh (nhãn + crop) -> giữ KEEP_AFTER_LAYER2 ảnh
  Lớp 3   layer3: chi tiết cảnh còn lại trong đề so với caption + mô tả thầy               -> xếp hạng cuối
search() chỉ nối các bước trên rồi xếp hạng: qua lớp 3 > dừng ở lớp 2 > còn lại.
Mỗi kênh so khớp hai cách: BM25 (từ khoá, tiếng Việt bỏ dấu / tiếng Anh) và SigLIP2 text (nghĩa câu, tiếng Anh).
Lọc mềm: lớp nào cũng chỉ cộng điểm rồi cắt top-K, không loại ảnh vì một nhãn lệch.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from common import EXPORTS_DIR, SIGLIP_MODEL, SIGLIP_WEIGHTS, TRANSLATION_MODEL, remove_accents, zscore
from query_parser import parse_query
from text_match import BM25, assign_one_to_one, tokenize

SEARCH_INDEX_DIR = EXPORTS_DIR / "search_index"
IMAGE_EMBEDDINGS_DIR = EXPORTS_DIR / "embeddings"   # embedding ảnh SigLIP2 do pipeline gán nhãn sinh; None = tắt kênh ảnh
KEEP_AFTER_LAYER1, KEEP_AFTER_LAYER2 = 1000, 200
UNMATCHED_PERSON_SCORE = -1.5   # cụm-người không ghép được ai (detector bỏ sót người): thấp nhưng không loại ảnh
NAME_MATCH_BONUS = 10.0         # lớn hơn tổng điểm các kênh khác: cụm có tên ("Thầy Linh") gần như buộc ghép đúng người đó
WEIGHTS = {                     # chọn trên 7 đề dev (2026-11-21)
    "people_count": 1.0,
    "people_count_exact": 6.0,  # đề nêu rõ số người ("có 0 người", "không có ai khác") -> gần như điều kiện bắt buộc
    "name": 1.5,
    "bm25_vi": 1.0,
    "bm25_en": 1.0,
    "sentence_similarity": 1.0,
    "image": 2.0,
    "person_match": 3.0,
    "person_crop": 2.0,
    "scene_detail": 1.0,
}
WRONG_GENDER_PENALTY = 2.0
ACTION_KEYWORD_BONUS = 1.0
# từ khoá (đã bỏ dấu) cho nhãn hành động; "standing idle" không có vì "đứng" đi kèm mọi hành động
ACTION_KEYWORDS = {
    "cooking at the stove": ["nau", "chien", "xao", "dao chao"],
    "preparing food": ["chuan bi do an", "chuan bi thuc an", "so che", "cat", "thai", "lam do an"],
    "washing dishes": ["rua chen", "rua bat", "rua do", "rua"],
    "sitting": ["ngoi"],
    "getting items from a cabinet": ["lay do", "ngan keo", "tu bep"],
    "walking": ["dang di", "di qua", "di lai", "di bo", "di ngang"],
    "throwing away trash": ["vut rac", "do rac", "thung rac"],
    "drinking": ["uong"],
    "making a hot drink": ["pha tra", "pha ca phe", "pha do uong", "may pha"],
    "carrying something": ["mang", "be do", "xach", "bung"],
    "eating": ["dang an", "an sang", "an trua", "an banh"],
    "opening the refrigerator": ["mo tu lanh", "tu lanh"],
    "cleaning": ["lau", "don dep", "quet"],
    "using a phone": ["dien thoai"],
    "talking": ["noi chuyen", "tro chuyen"],
    "using a laptop": ["may tinh", "laptop"],
}


class EmbeddingModels:
    """Model dịch opus-mt vi→en và tháp text của SigLIP2 (bỏ tháp ảnh cho nhẹ GPU)."""

    def __init__(self, device: str | None = None) -> None:
        import open_clip
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        siglip, _, _ = open_clip.create_model_and_transforms(SIGLIP_MODEL, pretrained=SIGLIP_WEIGHTS, device="cpu")
        siglip.visual = torch.nn.Identity()
        self.siglip = (siglip.half() if self.device == "cuda" else siglip).to(self.device).eval()
        self.siglip_tokenizer = open_clip.get_tokenizer(SIGLIP_MODEL)
        self.translation_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL)
        self.translator = MarianMTModel.from_pretrained(TRANSLATION_MODEL).to(self.device).eval()
        if self.device == "cuda":
            self.translator.half()

    def translate(self, texts: list[str], beams: int | None = None) -> list[str]:
        """beams=None: mặc định của model (beam search, lúc truy vấn); beams=1: greedy, nhanh cho dịch hàng loạt offline."""
        import torch

        if not texts:
            return []
        with torch.no_grad():
            batch = self.translation_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            output = self.translator.generate(**batch, max_length=256, **({"num_beams": beams} if beams else {}))
        return self.translation_tokenizer.batch_decode(output, skip_special_tokens=True)

    def encode(self, texts: list[str]):
        """Vector đã chuẩn hoá L2, dạng torch trên device (fp16 nếu cuda)."""
        import torch

        with torch.no_grad():
            features = self.siglip.encode_text(self.siglip_tokenizer(texts).to(self.device))
        return torch.nn.functional.normalize(features.float(), dim=-1).to(features.dtype)

    def encode_many(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        return np.concatenate([self.encode(texts[k:k + batch_size]).cpu().numpy().astype(np.float16)
                               for k in range(0, len(texts), batch_size)])


def _mentions_action(clause_no_accent: str, action: str | None) -> bool:
    return any(re.search(rf"(?<!\w){w}(?!\w)", clause_no_accent) for w in ACTION_KEYWORDS.get(action or "", []))


class SearchEngine:
    def __init__(self, device: str | None = None, index_dir: Path = SEARCH_INDEX_DIR, models: EmbeddingModels | None = None,
                 image_embeddings_dir: Path | None = IMAGE_EMBEDDINGS_DIR) -> None:
        import torch

        images = json.loads((index_dir / "images.json").read_text(encoding="utf-8"))
        self.image_ids = images["ids"]
        self.image_row = {image_id: row for row, image_id in enumerate(self.image_ids)}
        self.people_count_detector = np.array(images["num_people"])
        self.people_count_teacher = np.array(images["teacher_people"])
        self.persons = json.loads((index_dir / "persons.json").read_text(encoding="utf-8"))
        self.scene_sentences = json.loads((index_dir / "scene_sentences.json").read_text(encoding="utf-8"))
        self.records = {}
        with open(EXPORTS_DIR / "castle_dataset.jsonl", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                if record["image_id"] in self.image_row:
                    self.records[record["image_id"]] = record
        num_images = len(self.image_ids)
        self.persons_by_image = [[] for _ in range(num_images)]
        for person_index, person in enumerate(self.persons):
            self.persons_by_image[person["image"]].append(person_index)
        self.persons_by_image = [np.array(rows, dtype=int) for rows in self.persons_by_image]
        self.names_in_image = [set() for _ in range(num_images)]
        for person in self.persons:
            if person["tag"]:
                self.names_in_image[person["image"]].add(person["tag"])
        self.known_names = sorted(set().union(*self.names_in_image), key=len, reverse=True)
        self.images_with_name = {name: np.array([name in names for names in self.names_in_image]) for name in self.known_names}

        image_text_vi, image_text_en = [[] for _ in range(num_images)], [[] for _ in range(num_images)]
        for sentence in self.scene_sentences:
            image_text_vi[sentence["image"]].append(sentence["vi"])
            image_text_en[sentence["image"]].append(sentence["en"])
        self.bm25_image_vi = BM25([tokenize(" ".join(texts)) for texts in image_text_vi])
        self.bm25_image_en = BM25([tokenize(" ".join(texts)) for texts in image_text_en])
        self.bm25_person_vi = BM25([tokenize(person["vi"]) for person in self.persons])
        self.person_gender = np.array([person["gender"] or "" for person in self.persons])

        self.models = models or EmbeddingModels(device)
        device, dtype = self.models.device, (torch.float16 if self.models.device == "cuda" else torch.float32)
        self.person_vectors = torch.from_numpy(np.load(index_dir / "person_vectors.npy")).to(device, dtype)
        self.scene_sentence_vectors = torch.from_numpy(np.load(index_dir / "scene_sentence_vectors.npy")).to(device, dtype)
        self.scene_sentence_image = torch.from_numpy(np.array([s["image"] for s in self.scene_sentences])).to(device)
        self.image_vectors = None
        if image_embeddings_dir:   # kênh ảnh: toàn khung + vùng có người (mức ảnh), crop từng người (mức người)
            file_order = [self.image_row.get(Path(x).stem) for x in json.loads((image_embeddings_dir / "image_ids.json").read_text())]
            present = [k for k, row in enumerate(file_order) if row is not None]
            file_row_of = np.empty(num_images, dtype=int)
            file_row_of[[file_order[k] for k in present]] = present
            self.image_vectors = [torch.from_numpy(np.load(image_embeddings_dir / f)[file_row_of].astype(np.float32)).to(device, dtype)
                                  for f in ("images_global.npy", "images.npy")]
            crop_row = {Path(x).stem: k for k, x in enumerate(json.loads((image_embeddings_dir / "people_ids.json").read_text()))}
            crops = np.load(image_embeddings_dir / "people.npy").astype(np.float32)
            crops /= np.linalg.norm(crops, axis=1, keepdims=True)
            self.person_has_crop = np.array([person["id"] in crop_row for person in self.persons])
            self.person_crop_vectors = torch.from_numpy(np.stack([crops[crop_row[p["id"]]] if p["id"] in crop_row else np.zeros(crops.shape[1], np.float32)
                                                                  for p in self.persons])).to(device, dtype)

    # ---- các kênh điểm -------------------------------------------------------------------------------------
    def _scene_similarity(self, query_vectors) -> np.ndarray:
        """Mỗi câu đề lấy câu cảnh khớp nhất trong từng ảnh, rồi trung bình các câu đề: (num_images,)."""
        import torch

        similarity = (query_vectors @ self.scene_sentence_vectors.T).float()
        best = torch.full((len(query_vectors), len(self.image_ids)), -1.0, device=similarity.device)
        best.scatter_reduce_(1, self.scene_sentence_image.expand(len(query_vectors), -1), similarity, reduce="amax")
        return best.mean(0).cpu().numpy()

    def _image_similarity(self, whole_query_vector, sentence_vectors) -> np.ndarray:
        """Cả câu và từng câu nhỏ so với ảnh toàn khung + vùng có người."""
        if self.image_vectors is None:
            return np.zeros(len(self.image_ids))
        total = np.zeros(len(self.image_ids))
        for vectors in self.image_vectors:
            total += zscore((whole_query_vector @ vectors.T).float().cpu().numpy()[0]) + zscore((sentence_vectors @ vectors.T).float().mean(0).cpu().numpy())
        return total / 4

    def _people_count_score(self, parsed: dict) -> np.ndarray:
        both_sources = np.stack([self.people_count_detector, self.people_count_teacher])
        if parsed["count"] is not None:
            gap = np.abs(both_sources - parsed["count"]).min(0)   # khớp một trong hai nguồn là đủ; lệch 1 chỉ còn chút điểm
            return np.where(gap == 0, 1.0, np.where(gap == 1, 0.25, 0.0))
        if parsed["count_min"]:
            most = both_sources.max(0)
            return np.where(most >= parsed["count_min"], 1.0, np.where(most >= parsed["count_min"] - 1, 0.5, 0.0))
        return np.zeros(len(self.image_ids))

    def _clause_person_scores(self, clauses: list[dict], clause_vectors) -> np.ndarray:
        """Điểm cụm × mọi người (num_clauses, num_persons): nghĩa câu + từ khoá + crop ảnh + giới tính + hành động + tên."""
        text_similarity = (clause_vectors @ self.person_vectors.T).float().cpu().numpy()
        scores = np.stack([zscore(text_similarity[k]) + zscore(self.bm25_person_vi.score(tokenize(c["text"]))) for k, c in enumerate(clauses)])
        if self.image_vectors is not None:     # crop người so với cụm; người không có crop nhận 0 (trung tính)
            crop_similarity = (clause_vectors @ self.person_crop_vectors.T).float().cpu().numpy()
            for k in range(len(clauses)):
                crop_score = np.zeros(len(self.persons))
                crop_score[self.person_has_crop] = zscore(crop_similarity[k][self.person_has_crop])
                scores[k] += WEIGHTS["person_crop"] * crop_score
        for k, clause in enumerate(clauses):
            if clause.get("gender"):
                scores[k] -= WRONG_GENDER_PENALTY * ((self.person_gender != "") & (self.person_gender != clause["gender"]))
            clause_no_accent = remove_accents(clause["text"])
            scores[k] += ACTION_KEYWORD_BONUS * np.array([_mentions_action(clause_no_accent, p["action"]) for p in self.persons], dtype=float)
            if clause.get("name"):
                scores[k] += NAME_MATCH_BONUS * np.array([p["tag"] == clause["name"] for p in self.persons])
        return scores

    # ---- truy vấn: bước 0 (tách đề + nhúng) rồi 3 lớp lọc mềm ---------------------------------------------
    def _encode_query(self, parsed: dict) -> dict:
        """Dịch câu đề, cụm-người, cụm-cảnh sang tiếng Anh rồi nhúng bằng SigLIP2 text."""
        clauses = parsed["people"] + ([{"text": parsed["group"]["text"], "gender": None, "name": None}] if parsed["group"] else [])
        english = self.models.translate(parsed["sentences"] + [c["text"] for c in clauses] + parsed["scene"])
        num_sentences, num_clauses = len(parsed["sentences"]), len(clauses)
        vectors = self.models.encode(english + [" ".join(english[:num_sentences])])
        return {"clauses": clauses, "english_sentences": english[:num_sentences],
                "sentence_vectors": vectors[:num_sentences], "clause_vectors": vectors[num_sentences:num_sentences + num_clauses],
                "scene_vectors": vectors[num_sentences + num_clauses:-1], "whole_query_vector": vectors[-1:]}

    def layer1(self, parsed: dict, encoded: dict) -> tuple[np.ndarray, dict, np.ndarray]:
        """Lớp 1, cả 5.000 ảnh: số người, tên, cảnh (BM25 + SigLIP2 text), ảnh SigLIP2.
        Trả (điểm từng ảnh, điểm từng thành phần, chỉ số KEEP_AFTER_LAYER1 ảnh cao nhất)."""
        parts = {"count": self._people_count_score(parsed),
                 "name": sum((self.images_with_name[name] for name in parsed["names"]), np.zeros(len(self.image_ids))),
                 "scene": (WEIGHTS["bm25_vi"] * zscore(self.bm25_image_vi.score(tokenize(" ".join(parsed["sentences"]))))
                           + WEIGHTS["bm25_en"] * zscore(self.bm25_image_en.score(tokenize(" ".join(encoded["english_sentences"]))))
                           + WEIGHTS["sentence_similarity"] * zscore(self._scene_similarity(encoded["sentence_vectors"]))) / 3,
                 "image": self._image_similarity(encoded["whole_query_vector"], encoded["sentence_vectors"])}
        count_weight = WEIGHTS["people_count_exact" if parsed["count"] is not None else "people_count"]
        scores = count_weight * parts["count"] + WEIGHTS["name"] * parts["name"] + parts["scene"] + WEIGHTS["image"] * parts["image"]
        return scores, parts, np.argsort(-scores)[:KEEP_AFTER_LAYER1]

    def layer2(self, parsed: dict, encoded: dict, layer1_scores: np.ndarray, kept_after_layer1: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
        """Lớp 2, trên ảnh qua lớp 1: ghép một-một cụm-người với người trong ảnh.
        Trả (điểm từng ảnh, điểm ghép người, giải thích ghép cho từng ảnh, chỉ số KEEP_AFTER_LAYER2 ảnh cao nhất)."""
        scores = layer1_scores.copy()
        matches: dict[int, list] = {}
        person_match = np.zeros(len(self.image_ids))
        if encoded["clauses"]:
            clause_person = self._clause_person_scores(encoded["clauses"], encoded["clause_vectors"])
            num_people_clauses = len(parsed["people"])
            group_size = parsed["group"]["n"] if parsed["group"] else 0
            for image in kept_after_layer1:
                persons_here = self.persons_by_image[image]
                total, assigned = assign_one_to_one(clause_person[:num_people_clauses, persons_here], UNMATCHED_PERSON_SCORE)
                used = {j for j in assigned if j is not None}
                if parsed["group"]:      # cụm-nhóm: lấy group_size người khớp nhất trong số người chưa ghép
                    rest = np.sort([clause_person[num_people_clauses, persons_here[j]] for j in range(len(persons_here)) if j not in used])[::-1]
                    total += float(np.concatenate([rest[:group_size], np.full(max(0, group_size - len(rest)), UNMATCHED_PERSON_SCORE)]).sum())
                person_match[image] = total / (num_people_clauses + group_size)
                matches[image] = [{"clause": c["text"], "person": self.persons[persons_here[j]]["id"], "tag": self.persons[persons_here[j]]["tag"],
                                   "box": self.persons[persons_here[j]]["box"]} if j is not None else {"clause": c["text"], "person": None}
                                  for c, j in zip(parsed["people"], assigned)]
            scores[kept_after_layer1] = layer1_scores[kept_after_layer1] + WEIGHTS["person_match"] * zscore(person_match[kept_after_layer1])
        kept_after_layer2 = kept_after_layer1[np.argsort(-scores[kept_after_layer1])][:KEEP_AFTER_LAYER2]
        return scores, person_match, matches, kept_after_layer2

    def layer3(self, parsed: dict, encoded: dict, layer2_scores: np.ndarray, kept_after_layer2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Lớp 3, trên ảnh qua lớp 2: cụm-cảnh còn lại so với caption + mô tả thầy. Trả (điểm cuối, điểm chi tiết cảnh)."""
        scores = layer2_scores.copy()
        scene_detail = np.zeros(len(self.image_ids))
        if parsed["scene"]:
            kept = kept_after_layer2
            scene_detail[kept] = (zscore(self._scene_similarity(encoded["scene_vectors"])[kept])
                                  + zscore(self.bm25_image_vi.score(tokenize(" ".join(parsed["scene"])))[kept])) / 2
            scores[kept] = layer2_scores[kept] + WEIGHTS["scene_detail"] * scene_detail[kept]
        return scores, scene_detail

    def search(self, query: str, top: int = 200) -> dict:
        parsed = parse_query(query, self.known_names)
        if not parsed["sentences"]:
            return {"sentences": [], "english": [], "names": [], "parsed": parsed, "layers": [len(self.image_ids), 0, 0], "results": []}
        encoded = self._encode_query(parsed)
        layer1_scores, layer1_parts, kept_after_layer1 = self.layer1(parsed, encoded)
        layer2_scores, person_match, matches, kept_after_layer2 = self.layer2(parsed, encoded, layer1_scores, kept_after_layer1)
        layer3_scores, scene_detail = self.layer3(parsed, encoded, layer2_scores, kept_after_layer2)

        passed_layer1 = np.zeros(len(self.image_ids), bool)
        passed_layer2 = np.zeros(len(self.image_ids), bool)
        passed_layer1[kept_after_layer1], passed_layer2[kept_after_layer2] = True, True
        final_scores = np.where(passed_layer2, layer3_scores, np.where(passed_layer1, layer2_scores, layer1_scores))
        order = np.lexsort((-final_scores, ~passed_layer1, ~passed_layer2))   # qua lớp 3 > dừng ở lớp 2 > còn lại
        results = []
        for image in order[:top]:
            results.append({"id": self.image_ids[image], "score": round(float(final_scores[image]), 3),
                            "explain": {"pairs": matches.get(int(image), []),
                                        "parts": {"count": round(float(layer1_parts["count"][image]), 2), "name": int(layer1_parts["name"][image]),
                                                  "scene": round(float(layer1_parts["scene"][image]), 2),
                                                  "image": round(float(layer1_parts["image"][image]), 2),
                                                  "person": round(float(person_match[image]), 2), "final": round(float(scene_detail[image]), 2)}}})
        return {"sentences": parsed["sentences"], "english": encoded["english_sentences"], "names": parsed["names"], "parsed": parsed,
                "layers": [len(self.image_ids), len(kept_after_layer1), len(kept_after_layer2)], "results": results}

    def rank_of(self, query: str, answers: set[str]) -> int:
        """Hạng (từ 1) của ảnh đúng đầu tiên; dùng cho đánh giá."""
        results = self.search(query, top=len(self.image_ids))["results"]
        return next(rank for rank, r in enumerate(results, 1) if r["id"] in answers)

"""Tìm ảnh Castle5000 bằng text (caption, mô tả của thầy, nhãn người) cộng kênh ảnh SigLIP2: 3 lớp lọc mềm.

Thiết kế: docs/superpowers/specs/2026-11-21-text-retrieval-design.md. Cần `python retrieval/build_text_index.py` trước.
  Lớp 1  cả 5.000 ảnh: số người (hai nguồn: detector, thầy), tên danh tính, mô tả cảnh, ảnh  -> giữ K1 ảnh
  Lớp 2  ghép một-một từng cụm-người của đề với một người trong ảnh (nhãn + crop người)   -> giữ K2 ảnh
  Lớp 3  chi tiết cảnh còn lại trong đề so với caption + mô tả thầy                    -> xếp hạng cuối
Mỗi kênh so khớp hai cách: BM25 (từ khoá, tiếng Việt bỏ dấu / tiếng Anh) và SigLIP2 text (nghĩa câu, tiếng Anh).
Lọc mềm: lớp nào cũng chỉ cộng điểm rồi cắt top-K, không loại ảnh vì một nhãn lệch.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from engine import CLIP_MODEL, CLIP_PRETRAINED, EXPORTS, MT_MODEL, _z
from textsearch import BM25, assign, fold, parse_query, tokens

INDEX = EXPORTS / "text_index"
IMAGE_EMB = EXPORTS / "embeddings"   # embedding ảnh SigLIP2 có sẵn (do `adl embed` / `enrich build` sinh); None = tắt kênh ảnh
K1, K2 = 1000, 200
FLOOR = -1.5        # điểm của cụm-người không ghép được ai (detector bỏ sót người): thấp nhưng không loại ảnh
NAME_MATCH = 10.0  # lớn hơn tổng z các kênh khác: cụm có tên ("Thầy Linh") gần như buộc ghép đúng người mang tên đó
# count_exact: đề nêu rõ số người ("có 0 người", "hai người đàn ông", "không có ai khác") -> gần như điều kiện bắt buộc
W = {"count": 1.0, "count_exact": 6.0, "name": 1.5, "bm_vi": 1.0, "bm_en": 1.0, "dense": 1.0, "image": 2.0, "person": 3.0, "person_img": 2.0,
     "final_scene": 1.0}  # chọn trên 7 đề dev (2026-11-21)
# từ khoá (đã bỏ dấu) cho nhãn hành động; "standing idle" không có vì "đứng" đi kèm mọi hành động
ACTION_WORDS = {
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


class Models:
    """opus-mt vi→en và tháp text của SigLIP2 (bỏ tháp ảnh cho nhẹ GPU)."""

    def __init__(self, device: str | None = None) -> None:
        import open_clip
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        clip, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED, device="cpu")
        clip.visual = torch.nn.Identity()
        self.clip = (clip.half() if self.device == "cuda" else clip).to(self.device).eval()
        self.clip_tok = open_clip.get_tokenizer(CLIP_MODEL)
        self.mt_tok = MarianTokenizer.from_pretrained(MT_MODEL)
        self.mt = MarianMTModel.from_pretrained(MT_MODEL).to(self.device).eval()
        if self.device == "cuda":
            self.mt.half()

    def translate(self, texts: list[str], beams: int | None = None) -> list[str]:
        """beams=None: mặc định của model (beam search, lúc truy vấn); beams=1: greedy, nhanh cho dịch hàng loạt offline."""
        import torch

        if not texts:
            return []
        with torch.no_grad():
            batch = self.mt_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            out = self.mt.generate(**batch, max_length=256, **({"num_beams": beams} if beams else {}))
        return self.mt_tok.batch_decode(out, skip_special_tokens=True)

    def encode(self, texts: list[str]):
        """Vector đã chuẩn hoá L2, dạng torch trên device (fp16 nếu cuda)."""
        import torch

        with torch.no_grad():
            f = self.clip.encode_text(self.clip_tok(texts).to(self.device))
        return torch.nn.functional.normalize(f.float(), dim=-1).to(f.dtype)

    def encode_many(self, texts: list[str], batch: int = 256) -> np.ndarray:
        return np.concatenate([self.encode(texts[k:k + batch]).cpu().numpy().astype(np.float16)
                               for k in range(0, len(texts), batch)])


def _has_action(clause_folded: str, action: str | None) -> bool:
    return any(re.search(rf"(?<!\w){w}(?!\w)", clause_folded) for w in ACTION_WORDS.get(action or "", []))


class TextEngine:
    def __init__(self, device: str | None = None, index: Path = INDEX, models: Models | None = None,
                 image_emb: Path | None = IMAGE_EMB) -> None:
        import torch

        meta = json.loads((index / "meta.json").read_text(encoding="utf-8"))
        self.ids = meta["ids"]
        self.row = {i: r for r, i in enumerate(self.ids)}
        self.count_det = np.array(meta["num_people"])
        self.count_teacher = np.array(meta["teacher_people"])
        self.persons = json.loads((index / "persons.json").read_text(encoding="utf-8"))
        self.scene = json.loads((index / "scene.json").read_text(encoding="utf-8"))
        self.records = {}
        with open(EXPORTS / "castle_dataset.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["image_id"] in self.row:
                    self.records[r["image_id"]] = r
        n = len(self.ids)
        self.person_image = np.array([p["image"] for p in self.persons])
        self.persons_of = [[] for _ in range(n)]
        for k, p in enumerate(self.persons):
            self.persons_of[p["image"]].append(k)
        self.persons_of = [np.array(v, dtype=int) for v in self.persons_of]
        self.names_of = [set() for _ in range(n)]
        for p in self.persons:
            if p["tag"]:
                self.names_of[p["image"]].add(p["tag"])
        self.all_names = sorted(set().union(*self.names_of), key=len, reverse=True)
        self.has_name = {nm: np.array([nm in s for s in self.names_of]) for nm in self.all_names}

        img_vi, img_en = [[] for _ in range(n)], [[] for _ in range(n)]
        for s in self.scene:
            img_vi[s["image"]].append(s["vi"])
            img_en[s["image"]].append(s["en"])
        self.bm_img_vi = BM25([tokens(" ".join(v)) for v in img_vi])
        self.bm_img_en = BM25([tokens(" ".join(v)) for v in img_en])
        self.bm_person_vi = BM25([tokens(p["vi"]) for p in self.persons])
        self.person_gender = np.array([p["gender"] or "" for p in self.persons])

        self.models = models or Models(device)
        dev, dtype = self.models.device, (torch.float16 if self.models.device == "cuda" else torch.float32)
        self.persons_en = torch.from_numpy(np.load(index / "persons_en.npy")).to(dev, dtype)
        self.scene_en = torch.from_numpy(np.load(index / "scene_en.npy")).to(dev, dtype)
        self.scene_image = torch.from_numpy(np.array([s["image"] for s in self.scene])).to(dev)
        self.images = None
        if image_emb:   # kênh ảnh: toàn khung + vùng có người (mức ảnh), crop từng người (mức người)
            order = [self.row.get(Path(x).stem) for x in json.loads((image_emb / "image_ids.json").read_text())]
            keep = [k for k, r in enumerate(order) if r is not None]
            rows = np.empty(n, dtype=int)
            rows[[order[k] for k in keep]] = keep
            self.images = [torch.from_numpy(np.load(image_emb / f)[rows].astype(np.float32)).to(dev, dtype)
                           for f in ("images_global.npy", "images.npy")]
            crop = {Path(x).stem: k for k, x in enumerate(json.loads((image_emb / "people_ids.json").read_text()))}
            people = np.load(image_emb / "people.npy").astype(np.float32)
            people /= np.linalg.norm(people, axis=1, keepdims=True)
            self.has_crop = np.array([p["id"] in crop for p in self.persons])
            self.crops = torch.from_numpy(np.stack([people[crop[p["id"]]] if p["id"] in crop else np.zeros(people.shape[1], np.float32)
                                                    for p in self.persons])).to(dev, dtype)

    # ---- các kênh điểm -------------------------------------------------------------------------------------
    def _dense_scene(self, vecs) -> np.ndarray:
        """Mỗi câu đề lấy câu cảnh khớp nhất trong từng ảnh, rồi trung bình các câu đề: (n_images,)."""
        import torch

        sims = (vecs @ self.scene_en.T).float()
        best = torch.full((len(vecs), len(self.ids)), -1.0, device=sims.device)
        best.scatter_reduce_(1, self.scene_image.expand(len(vecs), -1), sims, reduce="amax")
        return best.mean(0).cpu().numpy()

    def _image_score(self, full_v, sent_v) -> np.ndarray:
        """Như engine cũ: cả câu và từng câu nhỏ so với ảnh toàn khung + vùng có người."""
        if self.images is None:
            return np.zeros(len(self.ids))
        out = np.zeros(len(self.ids))
        for img in self.images:
            out += _z((full_v @ img.T).float().cpu().numpy()[0]) + _z((sent_v @ img.T).float().mean(0).cpu().numpy())
        return out / 4

    def _count_score(self, q: dict) -> np.ndarray:
        both = np.stack([self.count_det, self.count_teacher])
        if q["count"] is not None:
            gap = np.abs(both - q["count"]).min(0)   # khớp một trong hai nguồn là đủ; lệch 1 chỉ còn chút điểm
            return np.where(gap == 0, 1.0, np.where(gap == 1, 0.25, 0.0))
        if q["count_min"]:
            most = both.max(0)
            return np.where(most >= q["count_min"], 1.0, np.where(most >= q["count_min"] - 1, 0.5, 0.0))
        return np.zeros(len(self.ids))

    def _person_sim(self, clauses: list[dict], vecs) -> np.ndarray:
        """Điểm cụm × mọi người (n_clauses, n_persons): nghĩa câu + từ khoá + cổng giới tính + hành động + tên."""
        dense = (vecs @ self.persons_en.T).float().cpu().numpy()
        sim = np.stack([_z(dense[k]) + _z(self.bm_person_vi.scores(tokens(c["text"]))) for k, c in enumerate(clauses)])
        if self.images is not None:     # crop người so với cụm; người không có crop nhận 0 (trung tính)
            img = (vecs @ self.crops.T).float().cpu().numpy()
            for k in range(len(clauses)):
                z = np.zeros(len(self.persons))
                z[self.has_crop] = _z(img[k][self.has_crop])
                sim[k] += W["person_img"] * z
        for k, c in enumerate(clauses):
            if c.get("gender"):
                sim[k] -= 2.0 * ((self.person_gender != "") & (self.person_gender != c["gender"]))
            f = fold(c["text"])
            sim[k] += np.array([_has_action(f, p["action"]) for p in self.persons], dtype=float)
            if c.get("name"):
                sim[k] += NAME_MATCH * np.array([p["tag"] == c["name"] for p in self.persons])
        return sim

    # ---- truy vấn ------------------------------------------------------------------------------------------
    def search(self, query: str, top: int = 200) -> dict:
        q = parse_query(query, self.all_names)
        empty = {"sentences": [], "english": [], "names": [], "parsed": q, "layers": [len(self.ids), 0, 0], "results": []}
        if not q["sentences"]:
            return empty
        clauses = q["people"] + ([{"text": q["group"]["text"], "gender": None, "name": None}] if q["group"] else [])
        texts = q["sentences"] + [c["text"] for c in clauses] + q["scene"]
        english = self.models.translate(texts)
        ns, nc = len(q["sentences"]), len(clauses)
        vecs = self.models.encode(english + [" ".join(english[:ns])])
        sent_v, clause_v, scene_v, full_v = vecs[:ns], vecs[ns:ns + nc], vecs[ns + nc:-1], vecs[-1:]

        # Lớp 1
        full = " ".join(q["sentences"])
        parts = {"count": self._count_score(q),
                 "name": sum((self.has_name[nm] for nm in q["names"]), np.zeros(len(self.ids))),
                 "scene": (W["bm_vi"] * _z(self.bm_img_vi.scores(tokens(full)))
                           + W["bm_en"] * _z(self.bm_img_en.scores(tokens(" ".join(english[:ns]))))
                           + W["dense"] * _z(self._dense_scene(sent_v))) / 3,
                 "image": self._image_score(full_v, sent_v)}
        s1 = W["count_exact" if q["count"] is not None else "count"] * parts["count"] + W["name"] * parts["name"] + parts["scene"] + W["image"] * parts["image"]
        c1 = np.argsort(-s1)[:K1]

        # Lớp 2
        s2 = s1.copy()
        explain: dict[int, list] = {}
        person_raw = np.zeros(len(self.ids))
        if clauses:
            sim = self._person_sim(clauses, clause_v)
            n_people = len(q["people"])
            for i in c1:
                idx = self.persons_of[i]
                score, pairs = assign(sim[:n_people, idx], FLOOR)
                used = {j for j in pairs if j is not None}
                if q["group"]:
                    rest = np.sort([sim[n_people, idx[j]] for j in range(len(idx)) if j not in used])[::-1]
                    need = q["group"]["n"]
                    score += float(np.concatenate([rest[:need], np.full(max(0, need - len(rest)), FLOOR)]).sum())
                person_raw[i] = score / (n_people + (q["group"]["n"] if q["group"] else 0))
                explain[i] = [{"clause": c["text"], "person": self.persons[idx[j]]["id"], "tag": self.persons[idx[j]]["tag"],
                               "box": self.persons[idx[j]]["box"]} if j is not None else {"clause": c["text"], "person": None}
                              for c, j in zip(q["people"], pairs)]
            s2[c1] = s1[c1] + W["person"] * _z(person_raw[c1])
        c2 = c1[np.argsort(-s2[c1])][:K2]

        # Lớp 3
        s3 = s2.copy()
        final = np.zeros(len(self.ids))
        if q["scene"]:
            final[c2] = (_z(self._dense_scene(scene_v)[c2]) + _z(self.bm_img_vi.scores(tokens(" ".join(q["scene"])))[c2])) / 2
            s3[c2] = s2[c2] + W["final_scene"] * final[c2]

        in1, in2 = np.zeros(len(self.ids), bool), np.zeros(len(self.ids), bool)
        in1[c1], in2[c2] = True, True
        order = np.lexsort((-np.where(in2, s3, np.where(in1, s2, s1)), ~in1, ~in2))   # tầng 3 > tầng 2 > còn lại
        results = []
        for j in order[:top]:
            results.append({"id": self.ids[j], "score": round(float(s3[j] if in2[j] else s2[j] if in1[j] else s1[j]), 3),
                            "explain": {"pairs": explain.get(int(j), []),
                                        "parts": {"count": round(float(parts["count"][j]), 2), "name": int(parts["name"][j]),
                                                  "scene": round(float(parts["scene"][j]), 2),
                                                  "image": round(float(parts["image"][j]), 2),
                                                  "person": round(float(person_raw[j]), 2), "final": round(float(final[j]), 2)}}})
        return {"sentences": q["sentences"], "english": english[:ns], "names": q["names"], "parsed": q,
                "layers": [len(self.ids), len(c1), len(c2)], "results": results}

    def rank_of(self, query: str, answers: set[str]) -> int:
        """Hạng (từ 1) của ảnh đúng đầu tiên; dùng cho đánh giá."""
        res = self.search(query, top=len(self.ids))["results"]
        return next(k for k, r in enumerate(res, 1) if r["id"] in answers)

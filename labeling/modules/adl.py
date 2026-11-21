"""Gán nhãn hoạt động (ADL) cấp ảnh theo quy trình bán tự động của LSC-ADL (arXiv:2504.02060).

Bốn stage của bài báo, lặp nhiều vòng:

0. ``caption`` / ``propose-labels`` / ``describe`` - chọn bộ nhãn và sinh câu mô tả bằng VLM + LLM
                 (mục 3.2 và Decompose-Expand; xem ``adl_label_selection.py``).
1. ``init``    - Golden corpus: mỗi lớp có vài câu mô tả; mỗi câu truy vấn text->ảnh trên
                 embedding CLIP để lấy ảnh hạt giống.
2. ``suggest`` - Mỗi cụm trong mỗi lớp có một vector trung bình. Ảnh chưa gán nhận nhãn của vector
                 có cosine cao nhất; dưới ngưỡng ``tau_sim`` thì bỏ.
3. ``export`` / ``import`` - Người duyệt chấp nhận, sửa hoặc từ chối gợi ý qua Label Studio.
4. ``update``  - Thêm ảnh đã duyệt vào lớp, gom cụm lại trong từng lớp bằng HDBSCAN. Mỗi cụm là
                 một biến thể bối cảnh của hoạt động và có centroid riêng. Quay lại stage 2.

Khác bài báo: retrieval dùng thẳng CLIP (thay SnapSeek), và ảnh hạt giống mang nguồn ``seed`` để
có thể đưa qua người duyệt trước khi tin (``export --what seeds``).

Trạng thái nằm trong ``output_jsons/adl/``:
    approved.json      {image_id: {label, source: seed|human, round, score}}
    rejected.json      {image_id: [các nhãn đã bị từ chối]}
    centroids.npz      vector trung bình của từng cụm + nhãn + dấu vân tay của approved.json
    suggestions.json   gợi ý của vòng hiện tại
    labels.json        kết quả cuối (ghi bởi ``apply``)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
from xml.sax.saxutils import quoteattr

import numpy as np
import yaml

from .schemas import ActivityLabel, ImageAnnotation
from .utils import get_logger, read_json, resolve_from_config, write_json

MODEL_VERSION = "castle5000-adl"
ROI_PROFILE_PREFIX = "roi-"
REJECT = "(không lớp nào đúng)"   # lựa chọn "từ chối" trong Label Studio
ACTIVITY, IMAGE = "activity", "image"


# --------------------------------------------------------------------------- profile
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        out[key] = _deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _fill(value: Any, **fields: str) -> Any:
    """Thay ``{roi}`` trong mọi chuỗi của một cấu trúc config lồng nhau."""
    if isinstance(value, str):
        for key, text in fields.items():
            value = value.replace("{" + key + "}", text)
        return value
    if isinstance(value, dict):
        return {k: _fill(v, **fields) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, **fields) for v in value]
    return value


def _roi_vlm_hint(cfg: dict[str, Any], name: str) -> Optional[str]:
    """Câu ``vlm_hint`` của vùng ``name`` trong rois.yaml (giải thích cho VLM vùng này trông ra sao), nếu có."""
    rois_file = (cfg.get("enrich") or {}).get("rois_file")
    if not rois_file:
        return None
    path = resolve_from_config(cfg, rois_file)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        items = (yaml.safe_load(f) or {}).get("rois") or []
    return next((str(item["vlm_hint"]) for item in items if item.get("name") == name and item.get("vlm_hint")), None)


def apply_profile(cfg: dict[str, Any], profile: Optional[str]) -> tuple[dict[str, Any], str]:
    """(config đã ghi đè mục ``adl`` theo profile, tên thư mục trạng thái trong ``output_jsons``).

    Cùng một vòng lặp gợi ý - duyệt chạy trên nhiều loại đơn vị: cả ảnh (mặc định), crop người
    (``people``), crop của một vùng cố định (``roi-<tên vùng>``, lấy khuôn ``adl_profiles.roi``).
    """
    if not profile:
        return cfg, "adl"
    profiles = cfg.get("adl_profiles") or {}
    if profile.startswith(ROI_PROFILE_PREFIX) and "roi" in profiles:
        name = profile[len(ROI_PROFILE_PREFIX):]
        override = _fill(profiles["roi"], roi=name)
        hint = _roi_vlm_hint(cfg, name)
        if hint:
            viewpoint = str(override.get("viewpoint", cfg["adl"].get("viewpoint", ""))).rstrip(". ")
            override = {**override, "viewpoint": f"{viewpoint}. {hint.strip().rstrip('.')}"}
    elif profile in profiles and profile != "roi":
        override = profiles[profile]
    else:
        known = sorted(k for k in profiles if k != "roi") + ([f"{ROI_PROFILE_PREFIX}<tên vùng>"] if "roi" in profiles else [])
        raise ValueError(f"Không có profile adl {profile!r}. Có: {', '.join(known) or '(không có)'}")
    return {**cfg, "adl": _deep_merge(cfg["adl"], override)}, f"adl_{profile}"


# --------------------------------------------------------------------------- bộ nhãn
@dataclass(frozen=True)
class AdlClass:
    name: str
    name_vi: str
    group: str
    descriptions: tuple[str, ...]


def load_labels(path: str | Path) -> list[AdlClass]:
    """Đọc ``adl_labels.yaml``. Mỗi lớp cần ``name`` duy nhất và ít nhất một câu mô tả.

    Khoá phụ do bước chọn nhãn ghi (``seen_in``, ``keywords``) được bỏ qua.
    """
    with open(path, encoding="utf-8") as f:
        raw = (yaml.safe_load(f) or {}).get("classes") or []
    classes: list[AdlClass] = []
    seen: set[str] = set()
    for item in raw:
        name = str(item.get("name", "")).strip()
        descriptions = tuple(str(d).strip() for d in item.get("descriptions") or [] if str(d).strip())
        if not name or name in seen:
            raise ValueError(f"{path}: tên lớp rỗng hoặc trùng: {name!r}")
        if name == REJECT:
            raise ValueError(f"{path}: {REJECT!r} là tên dành riêng")
        if not descriptions:
            raise ValueError(f"{path}: lớp {name!r} không có câu mô tả nào. Chạy `adl describe` để sinh.")
        seen.add(name)
        classes.append(AdlClass(name, str(item.get("name_vi", "")).strip(), str(item.get("group", "")).strip(), descriptions))
    if not classes:
        raise ValueError(f"{path}: không có lớp nào")
    return classes


# --------------------------------------------------------------------------- embedding
def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    return matrix / np.maximum(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12)


def load_image_embeddings(cfg: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    """(image_id, ma trận đã chuẩn hoá L2). ``ids.json`` chứa tên file; ID là phần không đuôi."""
    adl = cfg["adl"]
    emb_path, ids_path = resolve_from_config(cfg, adl["embeddings"]), resolve_from_config(cfg, adl["ids"])
    if not emb_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"Thiếu embedding ảnh: {emb_path} / {ids_path}")
    ids = [Path(str(name)).stem for name in read_json(ids_path)]
    matrix = l2_normalize(np.load(emb_path))
    if matrix.ndim != 2 or matrix.shape[0] != len(ids):
        raise ValueError(f"embeddings {matrix.shape} không khớp {len(ids)} ID")
    stats_path = emb_path.parent / "build_stats.json"
    if stats_path.exists():  # câu mô tả phải được mã hoá bằng đúng model đã mã hoá ảnh
        stats, clip = read_json(stats_path), adl["clip"]
        built, wanted = (stats.get("model"), stats.get("pretrained")), (clip["model"], clip["pretrained"])
        if all(built) and built != wanted:
            raise ValueError(f"Embedding ảnh dựng bằng {built}, nhưng adl.clip là {wanted}")
    return ids, matrix


def center(image_emb: np.ndarray) -> np.ndarray:
    """Trừ vector trung bình của cả tập rồi chuẩn hoá lại.

    Camera cố định nên mọi ảnh chia sẻ một thành phần nền rất lớn (cosine ảnh-ảnh trung bình > 0.7).
    Bỏ thành phần chung đó thì cosine phản ánh phần khác nhau giữa các ảnh: người và hành động.
    Chỉ dùng cho so sánh ảnh-ảnh (gom cụm, gợi ý); truy vấn text->ảnh dùng embedding gốc.
    """
    return l2_normalize(image_emb - image_emb.mean(axis=0, keepdims=True))


def image_space(cfg: dict[str, Any], image_emb: np.ndarray) -> np.ndarray:
    """Embedding dùng cho so sánh ảnh-ảnh, theo ``adl.center``."""
    return center(image_emb) if cfg["adl"].get("center", True) else image_emb


def load_motion(cfg: dict[str, Any], count: int) -> Optional[np.ndarray]:
    """Tỉ lệ diện tích có chuyển động của từng ảnh (do ``adl embed`` ghi); None nếu chưa có."""
    path = resolve_from_config(cfg, cfg["adl"]["embeddings"]).parent / "motion.npy"
    if not path.exists():
        return None
    motion = np.load(path)
    return motion if len(motion) == count else None


def empty_rule(cfg: dict[str, Any], count: int) -> tuple[Optional[str], Optional[np.ndarray]]:
    """(tên lớp "không có người", mặt nạ ảnh không có chuyển động). (None, None) nếu tắt hoặc thiếu dữ liệu."""
    rule = cfg["adl"].get("empty") or {}
    motion = load_motion(cfg, count) if rule.get("class") else None
    if motion is None:
        return None, None
    return str(rule["class"]), motion <= float(rule["max_motion"])


class ClipTextEncoder:
    """Mã hoá câu mô tả bằng open_clip, có cache trên đĩa để các lần chạy sau không cần nạp model."""

    def __init__(self, model: str, pretrained: str, cache_path: Path) -> None:
        self.model, self.pretrained, self.cache_path = model, pretrained, cache_path
        self._cache: dict[str, np.ndarray] = {}
        if cache_path.exists():
            with np.load(cache_path) as data:
                self._cache = {key: data[key] for key in data.files}

    def _key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}|{self.pretrained}|{text}".encode("utf-8")).hexdigest()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if self._key(t) not in self._cache]
        if missing:
            for text, vector in zip(missing, self._encode_with_model(missing)):
                self._cache[self._key(text)] = vector
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(self.cache_path, **self._cache)
        return np.stack([self._cache[self._key(t)] for t in texts])

    _loaded: dict[tuple[str, str], Any] = {}   # model đã nạp, dùng chung giữa các encoder trong một tiến trình

    def _encode_with_model(self, texts: Sequence[str]) -> np.ndarray:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError("Stage adl cần open_clip để mã hoá câu mô tả: pip install open_clip_torch") from exc
        key = (self.model, self.pretrained)
        if key not in ClipTextEncoder._loaded:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, _ = open_clip.create_model_and_transforms(self.model, pretrained=self.pretrained, device=device)
            model.eval()
            if device == "cuda":
                model.half()   # model lớn ở float32 không vừa GPU 6 GB, tràn sang RAM dùng chung và chậm hàng chục lần
            ClipTextEncoder._loaded[key] = (model, open_clip.get_tokenizer(self.model), device)
        model, tokenizer, device = ClipTextEncoder._loaded[key]
        out = []
        with torch.no_grad():   # theo lô: hàng nghìn câu một lần không vừa GPU 6 GB
            for i in range(0, len(texts), 256):
                out.append(model.encode_text(tokenizer(list(texts[i:i + 256])).to(device)).float().cpu().numpy())
        return l2_normalize(np.concatenate(out))

    @classmethod
    def release_model(cls) -> None:
        """Bỏ model đã nạp (gọi sau khi mã hoá xong mọi profile, trước khi nạp model khác lên GPU)."""
        cls._loaded.clear()


# --------------------------------------------------------------------------- stage 1: hạt giống
def select_seeds(
    image_emb: np.ndarray,
    text_emb_by_class: dict[str, np.ndarray],
    per_description: int,
    max_per_class: int,
    min_similarity: float = 0.0,
    excluded: Optional[np.ndarray] = None,
) -> dict[int, tuple[str, float]]:
    """Chọn ảnh hạt giống: {chỉ số ảnh: (lớp, cosine)}.

    ``excluded``: mặt nạ các ảnh không được làm hạt giống (ví dụ ảnh không có người).

    Mỗi câu mô tả lấy ``per_description`` ảnh gần nhất. Điểm của ảnh với một lớp là cosine cao nhất
    qua các câu của lớp đó. Ảnh được nhiều lớp chọn thì thuộc lớp có điểm cao nhất. Mỗi lớp giữ tối
    đa ``max_per_class`` ảnh điểm cao nhất.
    """
    best: dict[int, tuple[str, float]] = {}
    for label, text_emb in text_emb_by_class.items():
        sims = image_emb @ text_emb.T                       # (số ảnh, số câu)
        if excluded is not None:
            sims = np.where(excluded[:, None], -np.inf, sims)
        k = min(per_description, sims.shape[0])
        candidates: dict[int, float] = {}
        for column in sims.T:
            for idx in np.argpartition(-column, k - 1)[:k]:
                score = float(column[idx])
                if np.isfinite(score) and score >= min_similarity:
                    candidates[int(idx)] = max(score, candidates.get(int(idx), -1.0))
        for idx, score in candidates.items():
            if idx not in best or score > best[idx][1]:
                best[idx] = (label, score)
    seeds: dict[int, tuple[str, float]] = {}
    for label in text_emb_by_class:
        mine = sorted(((s, i) for i, (l, s) in best.items() if l == label), reverse=True)[:max_per_class]
        seeds.update({i: (label, s) for s, i in mine})
    return seeds


# --------------------------------------------------------------------------- stage 4: gom cụm
def cluster_class(vectors: np.ndarray, min_cluster_size: int, min_samples: int) -> np.ndarray:
    """Nhãn cụm HDBSCAN cho ảnh của MỘT lớp; -1 là nhiễu.

    Lớp quá ít ảnh, hoặc HDBSCAN coi mọi ảnh là nhiễu, thì cả lớp là một cụm (toàn 0).
    """
    n = len(vectors)
    if n < 2 * min_cluster_size:
        return np.zeros(n, dtype=int)
    from sklearn.cluster import HDBSCAN

    # Vector đã chuẩn hoá L2 nên khoảng cách Euclid đơn điệu với khoảng cách cosine.
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, copy=True).fit_predict(vectors)
    return labels if (labels >= 0).any() else np.zeros(n, dtype=int)


@dataclass
class Centroids:
    vectors: np.ndarray        # (số cụm, chiều), đã chuẩn hoá
    labels: list[str]          # lớp ADL của từng cụm
    cluster_ids: list[int]     # số thứ tự cụm trong lớp
    sizes: list[int]           # số ảnh của từng cụm

    def summary(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for label, size in zip(self.labels, self.sizes):
            entry = out.setdefault(label, {"clusters": 0, "images": 0})
            entry["clusters"] += 1
            entry["images"] += size
        return out


def build_centroids(
    approved: dict[str, dict[str, Any]],
    ids: Sequence[str],
    image_emb: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
) -> Centroids:
    """Gom cụm trong từng lớp rồi lấy vector trung bình của mỗi cụm. Ảnh nhiễu không vào centroid."""
    index = {image_id: i for i, image_id in enumerate(ids)}
    by_label: dict[str, list[int]] = {}
    for image_id, record in approved.items():
        if image_id in index:
            by_label.setdefault(record["label"], []).append(index[image_id])
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    cluster_ids: list[int] = []
    sizes: list[int] = []
    for label in sorted(by_label):
        members = image_emb[sorted(by_label[label])]
        assignment = cluster_class(members, min_cluster_size, min_samples)
        for cluster_id in sorted(set(assignment.tolist()) - {-1}):
            chosen = members[assignment == cluster_id]
            vectors.append(l2_normalize(chosen.mean(axis=0)))
            labels.append(label)
            cluster_ids.append(int(cluster_id))
            sizes.append(int(len(chosen)))
    dim = image_emb.shape[1]
    return Centroids(np.stack(vectors) if vectors else np.zeros((0, dim), np.float32), labels, cluster_ids, sizes)


# --------------------------------------------------------------------------- stage 2: gợi ý
def suggest_labels(
    ids: Sequence[str],
    image_emb: np.ndarray,
    centroids: Centroids,
    annotated: set[str],
    rejected: dict[str, list[str]],
    tau_sim: float,
    min_margin: float = 0.0,
    max_suggestions: Optional[int] = None,
    empty_class: Optional[str] = None,
    is_empty: Optional[np.ndarray] = None,
) -> list[dict[str, Any]]:
    """Gợi ý nhãn cho ảnh chưa gán: lớp của centroid có cosine cao nhất, giữ nếu >= ``tau_sim``.

    Nhãn đã bị người duyệt từ chối cho một ảnh thì không gợi ý lại cho ảnh đó. ``margin`` là khoảng
    cách tới lớp KHÁC tốt nhất; ``min_margin`` > 0 loại các ảnh nằm giữa hai lớp.

    ``empty_class`` + ``is_empty`` (từ trừ nền): ảnh không có chuyển động được gợi ý thẳng lớp đó với
    điểm 1.0, và lớp đó không bao giờ được gợi ý cho ảnh có chuyển động.
    """
    forced: list[dict[str, Any]] = []
    if empty_class is not None and is_empty is not None:
        forced = [
            {"image_id": image_id, "label": empty_class, "score": 1.0, "cluster": -1, "runner_up": None, "margin": 1.0}
            for i, image_id in enumerate(ids)
            if is_empty[i] and image_id not in annotated and empty_class not in rejected.get(image_id, ())
        ]
    if not len(centroids.labels):
        return forced[:max_suggestions] if max_suggestions else forced
    class_names = sorted(set(centroids.labels))
    sims = image_emb @ centroids.vectors.T                   # (số ảnh, số cụm)
    # Điểm theo lớp = cụm tốt nhất của lớp đó.
    per_class = np.full((len(ids), len(class_names)), -np.inf, dtype=np.float32)
    best_cluster = np.zeros((len(ids), len(class_names)), dtype=int)
    for c, name in enumerate(class_names):
        columns = [j for j, label in enumerate(centroids.labels) if label == name]
        block = sims[:, columns]
        local = block.argmax(axis=1)
        per_class[:, c] = block[np.arange(len(ids)), local]
        best_cluster[:, c] = np.asarray(columns)[local]

    out: list[dict[str, Any]] = []
    for i, image_id in enumerate(ids):
        if image_id in annotated or (is_empty is not None and is_empty[i]):
            continue
        row = per_class[i].copy()
        if empty_class in class_names:
            row[class_names.index(empty_class)] = -np.inf
        for name in rejected.get(image_id, ()):
            if name in class_names:
                row[class_names.index(name)] = -np.inf
        order = np.argsort(-row)
        top = int(order[0])
        score = float(row[top])
        if not np.isfinite(score) or score < tau_sim:
            continue
        runner = int(order[1]) if len(order) > 1 and np.isfinite(row[order[1]]) else None
        margin = score - float(row[runner]) if runner is not None else score
        if margin < min_margin:
            continue
        out.append({
            "image_id": image_id,
            "label": class_names[top],
            "score": round(score, 4),
            "cluster": centroids.cluster_ids[int(best_cluster[i, top])],
            "runner_up": class_names[runner] if runner is not None else None,
            "margin": round(margin, 4),
        })
    out.sort(key=lambda s: -s["score"])
    if max_suggestions:
        # Ảnh không người chắc chắn nhưng nhàm: chỉ cho chiếm tối đa một phần tư mỗi vòng.
        forced = forced[: max(0, max_suggestions // 4)]
        out = out[: max_suggestions - len(forced)]
    return forced + out


# --------------------------------------------------------------------------- trạng thái trên đĩa
class AdlStore:
    """Đọc/ghi các file trạng thái trong ``output_jsons/adl/``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.approved_path = self.root / "approved.json"
        self.rejected_path = self.root / "rejected.json"
        self.centroids_path = self.root / "centroids.npz"
        self.suggestions_path = self.root / "suggestions.json"
        self.labels_path = self.root / "labels.json"
        self.text_cache_path = self.root / "text_embeddings.npz"

    def _read(self, path: Path, default: Any) -> Any:
        return read_json(path) if path.exists() else default

    def approved(self) -> dict[str, dict[str, Any]]:
        return self._read(self.approved_path, {})

    def rejected(self) -> dict[str, list[str]]:
        return self._read(self.rejected_path, {})

    def suggestions(self) -> dict[str, Any]:
        return self._read(self.suggestions_path, {"round": 0, "items": []})

    def save_approved(self, approved: dict[str, dict[str, Any]]) -> None:
        write_json(self.approved_path, dict(sorted(approved.items())))

    def save_rejected(self, rejected: dict[str, list[str]]) -> None:
        write_json(self.rejected_path, dict(sorted(rejected.items())))

    @staticmethod
    def fingerprint(approved: dict[str, dict[str, Any]]) -> str:
        pairs = sorted((image_id, record["label"]) for image_id, record in approved.items())
        return hashlib.sha1(json.dumps(pairs).encode("utf-8")).hexdigest()

    def save_centroids(self, centroids: Centroids, approved: dict[str, dict[str, Any]], round_no: int) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.savez(
            self.centroids_path,
            vectors=centroids.vectors,
            labels=np.asarray(centroids.labels, dtype=str),
            cluster_ids=np.asarray(centroids.cluster_ids, dtype=int),
            sizes=np.asarray(centroids.sizes, dtype=int),
            fingerprint=np.asarray(self.fingerprint(approved)),
            round=np.asarray(round_no),
        )

    def load_centroids(self) -> tuple[Centroids, str, int]:
        if not self.centroids_path.exists():
            raise FileNotFoundError(f"Chưa có {self.centroids_path}. Chạy `adl init` trước.")
        with np.load(self.centroids_path) as data:
            centroids = Centroids(data["vectors"], [str(x) for x in data["labels"]], data["cluster_ids"].tolist(), data["sizes"].tolist())
            return centroids, str(data["fingerprint"]), int(data["round"])


# --------------------------------------------------------------------------- các lệnh
def _cluster_cfg(cfg: dict[str, Any]) -> tuple[int, int]:
    cluster = cfg["adl"]["cluster"]
    return int(cluster["min_cluster_size"]), int(cluster["min_samples"])


def run_init(cfg: dict[str, Any], store: AdlStore, force: bool = False) -> dict[str, Any]:
    """Stage 1: dựng golden corpus từ câu mô tả, rồi dựng centroid vòng 0."""
    approved = store.approved()
    if approved and not force:
        raise RuntimeError(f"{store.approved_path} đã có {len(approved)} ảnh. Dùng --force để dựng lại từ đầu (mất nhãn đã duyệt).")
    adl = cfg["adl"]
    classes = load_labels(resolve_from_config(cfg, adl["labels_file"]))
    ids, image_emb = load_image_embeddings(cfg)
    encoder = ClipTextEncoder(adl["clip"]["model"], adl["clip"]["pretrained"], store.text_cache_path)
    encoder.encode([d for c in classes for d in c.descriptions])   # một lượt cho mọi câu: chỉ nạp model một lần
    text_emb = {c.name: encoder.encode(c.descriptions) for c in classes}
    seed_cfg = adl["seed"]
    empty_class, is_empty = empty_rule(cfg, len(ids))
    if empty_class is not None:
        # Lớp "không có người" lấy hạt giống từ trừ nền, không từ CLIP; ảnh đó cũng không vào lớp khác.
        text_emb.pop(empty_class, None)
    seeds = select_seeds(image_emb, text_emb, int(seed_cfg["per_description"]), int(seed_cfg["max_per_class"]), float(seed_cfg["min_similarity"]), is_empty)
    approved = {ids[i]: {"label": label, "source": "seed", "round": 0, "score": round(score, 4)} for i, (label, score) in seeds.items()}
    if empty_class is not None:
        for i in np.flatnonzero(is_empty)[: int(seed_cfg["max_per_class"])]:
            approved[ids[int(i)]] = {"label": empty_class, "source": "seed", "round": 0, "score": 1.0}
    store.save_approved(approved)
    store.save_rejected({})
    centroids = build_centroids(approved, ids, image_space(cfg, image_emb), *_cluster_cfg(cfg))
    store.save_centroids(centroids, approved, 0)
    return {"seeds": len(approved), "classes": centroids.summary()}


def run_update(cfg: dict[str, Any], store: AdlStore) -> dict[str, Any]:
    """Stage 4: gom cụm lại trong từng lớp với tập đã duyệt hiện tại, sang vòng mới."""
    approved = store.approved()
    if not approved:
        raise RuntimeError("Golden corpus rỗng. Chạy `adl init` trước.")
    ids, image_emb = load_image_embeddings(cfg)
    _, _, round_no = store.load_centroids()
    centroids = build_centroids(approved, ids, image_space(cfg, image_emb), *_cluster_cfg(cfg))
    store.save_centroids(centroids, approved, round_no + 1)
    return {"round": round_no + 1, "approved": len(approved), "classes": centroids.summary()}


def run_suggest(cfg: dict[str, Any], store: AdlStore) -> dict[str, Any]:
    """Stage 2: gợi ý nhãn cho ảnh chưa gán bằng centroid hiện tại."""
    approved = store.approved()
    centroids, fingerprint, round_no = store.load_centroids()
    if fingerprint != store.fingerprint(approved):
        get_logger().warning("Tập đã duyệt đã thay đổi sau lần gom cụm gần nhất. Chạy `adl update` để centroid dùng nhãn mới.")
    ids, image_emb = load_image_embeddings(cfg)
    s = cfg["adl"]["suggest"]
    empty_class, is_empty = empty_rule(cfg, len(ids))
    items = suggest_labels(
        ids, image_space(cfg, image_emb), centroids, set(approved), store.rejected(),
        float(s["tau_sim"]), float(s["min_margin"]), s.get("max_per_round"), empty_class, is_empty,
    )
    write_json(store.suggestions_path, {"round": round_no, "items": items})
    counts: dict[str, int] = {}
    for item in items:
        counts[item["label"]] = counts.get(item["label"], 0) + 1
    return {"round": round_no, "suggestions": len(items), "unlabeled": len(ids) - len(approved), "per_class": dict(sorted(counts.items()))}


def review_images_dir(cfg: dict[str, Any]) -> Path:
    """Thư mục ảnh mà trang duyệt hiển thị: ảnh gốc, hoặc thư mục crop của profile."""
    custom = (cfg["adl"].get("review") or {}).get("images_dir")
    return resolve_from_config(cfg, custom) if custom else Path(cfg["paths"]["input_images"])


def build_label_config(classes: Sequence[AdlClass]) -> str:
    """Giao diện Label Studio: một ảnh, chọn đúng một lớp hoạt động hoặc từ chối."""
    lines = [
        "<View>",
        f'  <Image name="{IMAGE}" value="$image" zoom="true" zoomControl="true"/>',
        '  <Header value="Hoạt động chính trong ảnh (gợi ý: $suggested, cosine $score)"/>',
        f'  <Choices name="{ACTIVITY}" toName="{IMAGE}" choice="single" showInline="true" required="true">',
        *[f"    <Choice value={quoteattr(c.name)} hint={quoteattr(c.name_vi)}/>" for c in classes],
        f"    <Choice value={quoteattr(REJECT)}/>",
        "  </Choices>",
        "</View>",
    ]
    return "\n".join(lines) + "\n"


def run_export(cfg: dict[str, Any], store: AdlStore, what: str = "suggestions") -> dict[str, Any]:
    """Stage 3 (xuất): task Label Studio với nhãn gợi ý đã chọn sẵn.

    ``what``: ``suggestions`` (gợi ý của vòng hiện tại), ``seeds`` (ảnh hạt giống chưa ai duyệt), hoặc
    ``vlm`` (nhãn VLM đã quyết, để người soát lại; chỗ VLM đổi nhãn xếp trước).
    """
    classes = load_labels(resolve_from_config(cfg, cfg["adl"]["labels_file"]))
    if what == "seeds":
        items = [{"image_id": i, "label": r["label"], "score": r.get("score")} for i, r in store.approved().items() if r.get("source") == "seed"]
    elif what == "vlm":
        items = [{"image_id": i, "label": r["label"], "score": r.get("score"), "changed": r.get("vlm") == "changed"} for i, r in store.approved().items() if r.get("source") == "vlm"]
        items.sort(key=lambda item: (not item["changed"], item["label"]))
    else:
        items = store.suggestions()["items"]
    if not items:
        raise RuntimeError(f"Không có gì để xuất ({what}). Chạy `adl suggest` hoặc `adl init` trước.")
    prefix = cfg["label_studio"]["image_url_prefix"]
    extension = {Path(str(name)).stem: Path(str(name)).suffix for name in read_json(resolve_from_config(cfg, cfg["adl"]["ids"]))}
    tasks = [
        {
            "data": {
                "image": f"{prefix}{item['image_id']}{extension.get(item['image_id'], '.jpg')}",
                "image_id": item["image_id"],
                "suggested": item["label"],
                "score": item.get("score"),
            },
            "predictions": [{
                "model_version": MODEL_VERSION,
                "score": item.get("score") or 0.0,
                "result": [{"id": ACTIVITY, "from_name": ACTIVITY, "to_name": IMAGE, "type": "choices", "value": {"choices": [item["label"]]}}],
            }],
        }
        for item in items
    ]
    tasks_path, config_path = store.root / f"label_studio_adl_{what}.json", store.root / "label_studio_adl_config.xml"
    write_json(tasks_path, tasks)
    config_path.write_text(build_label_config(classes), encoding="utf-8")

    from .adl_review import write_review_page

    round_no = store.load_centroids()[2] if store.centroids_path.exists() else 0
    thumbs = (cfg["adl"].get("review") or {}).get("thumbs_dir")
    page = write_review_page(
        store.root / f"review_{what}.html", items, [(c.name, c.name_vi) for c in classes], REJECT,
        f"{store.root.name}_{what}_round{round_no}", review_images_dir(cfg), resolve_from_config(cfg, thumbs) if thumbs else None, extension,
    )
    return {"tasks": len(tasks), "review_page": str(page), "tasks_file": str(tasks_path), "config_file": str(config_path)}


def apply_reviews(
    tasks: list[dict[str, Any]],
    approved: dict[str, dict[str, Any]],
    rejected: dict[str, list[str]],
    valid_labels: set[str],
    round_no: int,
) -> dict[str, int]:
    """Stage 3 (nhập): áp quyết định của người duyệt lên ``approved`` và ``rejected`` (sửa tại chỗ).

    Chọn một lớp (đúng gợi ý hay lớp khác) -> ảnh vào lớp đó với nguồn ``human``. Chọn từ chối ->
    nhãn gợi ý bị ghi vào ``rejected`` và ảnh rời khỏi golden corpus nếu đang ở đó. Task chưa ai
    nộp, hoặc đã bị huỷ, thì bỏ qua.
    """
    stats = {"approved": 0, "changed": 0, "rejected": 0, "not_reviewed": 0, "invalid": 0}
    for task in tasks:
        data = task.get("data") or {}
        image_id, suggested = data.get("image_id"), data.get("suggested")
        submitted = [a for a in task.get("annotations") or [] if not a.get("was_cancelled")]
        if not image_id or not submitted:
            stats["not_reviewed"] += 1
            continue
        latest = max(submitted, key=lambda a: a.get("updated_at") or a.get("created_at") or "")
        choices = [c for r in latest.get("result") or [] if r.get("from_name") == ACTIVITY for c in (r.get("value") or {}).get("choices") or []]
        if not choices:
            stats["not_reviewed"] += 1
            continue
        choice = choices[0]
        if choice == REJECT:
            if suggested and suggested not in rejected.setdefault(image_id, []):
                rejected[image_id].append(suggested)
            approved.pop(image_id, None)
            stats["rejected"] += 1
        elif choice in valid_labels:
            approved[image_id] = {"label": choice, "source": "human", "round": round_no, "score": data.get("score")}
            stats["approved" if choice == suggested else "changed"] += 1
        else:
            stats["invalid"] += 1
    return stats


def run_import(cfg: dict[str, Any], store: AdlStore, export_path: Path) -> dict[str, int]:
    tasks = read_json(export_path)
    if not isinstance(tasks, list):
        raise ValueError("File export phải là danh sách task (định dạng JSON của Label Studio).")
    classes = load_labels(resolve_from_config(cfg, cfg["adl"]["labels_file"]))
    approved, rejected = store.approved(), store.rejected()
    round_no = store.load_centroids()[2] if store.centroids_path.exists() else 0
    stats = apply_reviews(tasks, approved, rejected, {c.name for c in classes}, round_no)
    store.save_approved(approved)
    store.save_rejected(rejected)
    return stats


def run_apply(cfg: dict[str, Any], store: AdlStore, per_image_dir: Path, include_seeds: bool = False, include_vlm: bool = False) -> dict[str, int]:
    """Ghi nhãn cuối vào ``adl/labels.json`` và vào trường ``activity`` của các file per-image đang có.

    Mặc định chỉ lấy nhãn người đã duyệt; ``include_vlm`` lấy thêm nhãn VLM đã duyệt, ``include_seeds``
    lấy thêm ảnh hạt giống chưa duyệt.
    """
    wanted = {"human"} | ({"vlm"} if include_vlm else set()) | ({"seed"} if include_seeds else set())
    classes = {c.name: c for c in load_labels(resolve_from_config(cfg, cfg["adl"]["labels_file"]))}
    final: dict[str, dict[str, Any]] = {}
    for image_id, record in store.approved().items():
        cls = classes.get(record["label"])
        if cls is None or record.get("source", "seed") not in wanted:
            continue
        final[image_id] = ActivityLabel(
            label=cls.name, label_vi=cls.name_vi, group=cls.group,
            source=record.get("source", ""), score=record.get("score"), round=int(record.get("round", 0)),
        ).model_dump()
    write_json(store.labels_path, dict(sorted(final.items())))
    updated = 0
    for image_id, activity in final.items():
        path = per_image_dir / f"{image_id}.json"
        if not path.exists():
            continue
        from .vlm_annotator import build_bm25_blob

        annotation = ImageAnnotation.model_validate(read_json(path))
        annotation.activity = ActivityLabel.model_validate(activity)
        annotation.dense_textual_corpus.bm25_blob = build_bm25_blob(annotation)
        write_json(path, annotation.model_dump())
        updated += 1
    return {"labels": len(final), "per_image_updated": updated}


def run_round(cfg: dict[str, Any], store: AdlStore) -> dict[str, Any]:
    """Một vòng máy làm trọn: gom cụm lại (stage 4) -> gợi ý (stage 2) -> xuất trang duyệt (stage 3).

    Người duyệt xong thì ``adl import FILE`` rồi gọi lại lệnh này cho vòng kế tiếp.
    """
    update = run_update(cfg, store)
    suggest = run_suggest(cfg, store)
    if not suggest["suggestions"]:
        return {"round": update["round"], "suggestions": 0, "note": "Không còn gợi ý nào vượt ngưỡng. Hạ adl.suggest.tau_sim hoặc chạy `adl apply`."}
    export = run_export(cfg, store, "suggestions")
    return {"round": update["round"], "approved": update["approved"], "suggestions": suggest["suggestions"], "per_class": suggest["per_class"], "review_page": export["review_page"]}


def run_status(cfg: dict[str, Any], store: AdlStore) -> dict[str, Any]:
    approved = store.approved()
    per_class: dict[str, dict[str, int]] = {}
    for record in approved.values():
        entry = per_class.setdefault(record["label"], {"seed": 0, "human": 0})
        entry[record.get("source", "seed")] = entry.get(record.get("source", "seed"), 0) + 1
    round_no = store.load_centroids()[2] if store.centroids_path.exists() else None
    total = len(read_json(resolve_from_config(cfg, cfg["adl"]["ids"])))
    return {
        "round": round_no,
        "images": total,
        "labeled": len(approved),
        "human_approved": sum(1 for r in approved.values() if r.get("source") == "human"),
        "rejections": sum(len(v) for v in store.rejected().values()),
        "pending_suggestions": len(store.suggestions()["items"]),
        "per_class": dict(sorted(per_class.items())),
    }

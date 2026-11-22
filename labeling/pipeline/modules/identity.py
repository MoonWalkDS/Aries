"""Gom crop người theo ngoại hình để đặt tên "người này là ai" (``python main.py enrich identities``).

Camera trên trần ít khi thấy rõ mặt, nhưng mỗi người mặc cùng một bộ đồ suốt buổi. Embedding SigLIP của
crop người (đã có từ ``enrich build``) được gom thành nhiều cụm nhỏ bằng KMeans: cố ý chia nhiều cụm hơn
số người thật, vì một người đứng quay lưng và quay mặt thường rơi vào hai cụm. Người duyệt đặt tên cho
từng cụm trên label_viewer/; nhiều cụm cùng tên là cùng một người. Cụm lẫn nhiều người thì để trống tên và sửa
từng người.

File:
    output_jsons/identities.json      cụm do máy gom: {k, clusters: {c07: {size, examples}}, assign: {person_id: {cluster, sim}}}
    output_jsons/identity_names.json  tên do người đặt: {clusters: {c07: "An"}, people: {person_id: "An" | ""}}
                                      tên đặt cho riêng một người thắng tên của cụm; "" = không rõ là ai.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from .utils import get_logger, read_json, write_json

IDENTITIES_FILE = "identities.json"
NAMES_FILE = "identity_names.json"


def run_identities(cfg: dict[str, Any], data_dir: Path, jsons_dir: Path, force: bool = False) -> dict[str, Any]:
    """Gom cụm và ghi ``identities.json``. Không ghi đè cụm cũ (tên đã đặt gắn với mã cụm) trừ khi ``force``."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    icfg, log = cfg["enrich"]["identity"], get_logger()
    out_path = jsons_dir / IDENTITIES_FILE
    names_path = jsons_dir / NAMES_FILE
    if out_path.exists() and not force:
        raise RuntimeError(f"Đã có {out_path}. Gom lại sẽ đổi mã cụm và làm lệch tên đã đặt trong {names_path.name}; thêm --force nếu chắc chắn.")
    people = read_json(data_dir / "people" / "people.json")
    ids = [Path(str(n)).stem for n in read_json(data_dir / "people" / "ids.json")]
    vectors = np.load(data_dir / "people" / "embeddings.npy").astype(np.float32)
    big = np.array([people[i]["box"][3] - people[i]["box"][1] >= float(icfg["min_box_height"]) for i in ids])
    # Học PCA + KMeans trên crop đủ lớn (crop nhỏ chủ yếu là nền), rồi xếp MỌI crop vào cụm gần nhất.
    pca = PCA(int(icfg["pca_dims"]), random_state=int(icfg["seed"])).fit(vectors[big])

    def project(x: np.ndarray) -> np.ndarray:
        z = pca.transform(x)
        return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)

    z = project(vectors)
    k = int(icfg["clusters"])
    km = KMeans(k, n_init=4, random_state=int(icfg["seed"])).fit(z[big])
    centers = km.cluster_centers_ / np.linalg.norm(km.cluster_centers_, axis=1, keepdims=True)
    sims = z @ centers.T
    labels, best = sims.argmax(1), sims.max(1)
    clusters: dict[str, dict[str, Any]] = {}
    for c in range(k):
        members = np.where(labels == c)[0]
        order = members[np.argsort(-best[members])]
        clusters[f"c{c:02d}"] = {"size": int(len(members)), "examples": [ids[j] for j in order[: int(icfg["examples"])]]}
    assign = {pid: {"cluster": f"c{int(labels[j]):02d}", "sim": round(float(best[j]), 3)} for j, pid in enumerate(ids)}
    write_json(out_path, {"k": k, "model": "siglip+pca+kmeans", "clusters": clusters, "assign": assign})
    if force and names_path.exists():
        log.warning("identities: đã gom lại; tên theo cụm trong %s có thể không còn đúng.", names_path)
    sizes = sorted((c["size"] for c in clusters.values()), reverse=True)
    return {"clusters": k, "people": len(ids), "largest": sizes[:5], "smallest": sizes[-3:], "out": str(out_path)}


def load_identity(jsons_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, str]]:
    """(cụm của từng người, tên theo cụm, tên đặt riêng từng người); rỗng nếu chưa gom."""
    ids_path, names_path = jsons_dir / IDENTITIES_FILE, jsons_dir / NAMES_FILE
    assign = read_json(ids_path).get("assign", {}) if ids_path.exists() else {}
    names = read_json(names_path) if names_path.exists() else {}
    return assign, names.get("clusters", {}), names.get("people", {})


def identity_of(person_id: str, assign: dict[str, dict[str, Any]], cluster_names: dict[str, str], person_names: dict[str, str]) -> Optional[dict[str, Any]]:
    """{cluster, sim, name, source}: tên riêng của người > tên của cụm; None nếu người này chưa được gom."""
    a = assign.get(person_id)
    if person_id in person_names:
        return {"cluster": a["cluster"] if a else None, "sim": a["sim"] if a else None, "name": person_names[person_id] or None, "source": "person"}
    if not a:
        return None
    name = cluster_names.get(a["cluster"]) or None
    return {"cluster": a["cluster"], "sim": a["sim"], "name": name, "source": "cluster" if name else None}


# --------------------------------------------------------------------------- học tên từ crop đã đặt tên
PRED_FILE = "identity_pred.json"
OUTFITS_FILE = "outfits.yaml"
# Các trường VLM (person_attrs) dùng để điền sẵn bảng trang phục bằng giá trị gặp nhiều nhất của mỗi người.
OUTFIT_PREFILL = ("gender", "age", "hair", "top", "top_color", "top_pattern", "bottom", "bottom_color", "headwear", "glasses", "apron")
# Trường VLM không trả lời: người duyệt điền tay trong outfits.yaml.
OUTFIT_MANUAL = ("stripe_direction", "top_color_upper", "top_color_lower", "description")


def train_predict(cfg: dict[str, Any], data_dir: Path, jsons_dir: Path) -> dict[str, Any]:
    """Học "crop này là ai" từ mọi crop đã có tên (theo cụm hoặc đặt riêng), rồi đoán cho MỌI crop.

    Logistic regression trên embedding SigLIP: bộ này chỉ có ~10–15 người, mỗi người một bộ đồ, nên cố ý
    học thuộc đúng các bộ đồ đó. Crop đặt "Không rõ" ("" trong identity_names.json) không dùng để học.
    Ghi ``identity_pred.json``: {person_id: {name, prob}}. Bỏ qua (trả lý do) nếu chưa đủ tên.
    """
    from sklearn.linear_model import LogisticRegression

    icfg = cfg["enrich"]["identity"]
    assign, cluster_names, person_names = load_identity(jsons_dir)
    ids = [Path(str(n)).stem for n in read_json(data_dir / "people" / "ids.json")]
    labels = []
    for pid in ids:
        ident = identity_of(pid, assign, cluster_names, person_names)
        labels.append(ident["name"] if ident and ident["name"] else None)
    counts: dict[str, int] = {}
    for name in labels:
        if name:
            counts[name] = counts.get(name, 0) + 1
    usable = {n for n, c in counts.items() if c >= int(icfg["min_examples"])}
    if len(usable) < 2:
        return {"skipped": f"cần ít nhất 2 tên, mỗi tên >= {icfg['min_examples']} crop; đang có {counts}"}
    vectors = np.load(data_dir / "people" / "embeddings.npy").astype(np.float32)
    train = [j for j, n in enumerate(labels) if n in usable]
    model = LogisticRegression(max_iter=2000, C=float(icfg["c"]), class_weight="balanced")
    model.fit(vectors[train], [labels[j] for j in train])
    proba = model.predict_proba(vectors)
    own = proba.max(1)
    smoothed = np.zeros(len(ids), bool)
    tcfg = icfg.get("track")
    if tcfg:
        # crop phân vân lấy tên theo cả vệt (cùng người qua các ảnh liền kề): đúng 0,655 -> 0,741 trên crop phân vân
        track = person_tracks(ids, read_json(data_dir / "people" / "people.json"), vectors, int(tcfg["max_gap"]), float(tcfg["min_iou"]), float(tcfg["min_cos"]))
        logp = np.log(proba + 1e-6)
        for t in np.unique(track):
            members = np.where(track == t)[0]
            weak = members[own[members] < float(icfg["min_prob"])]
            if len(members) > 1 and len(weak):
                post = np.exp(logp[members].mean(0))
                proba[weak] = post / post.sum()
                smoothed[weak] = True
    best = proba.argmax(1)
    second = np.sort(proba, 1)[:, -2]
    pred = {pid: {"name": str(model.classes_[best[j]]), "prob": round(float(proba[j, best[j]]), 3),
                  "margin": round(float(proba[j, best[j]] - second[j]), 3), **({"track": True} if smoothed[j] else {})} for j, pid in enumerate(ids)}
    write_json(jsons_dir / PRED_FILE, pred)
    agree = sum(pred[ids[j]]["name"] == labels[j] for j in train)
    confident = sum(p["prob"] >= float(icfg["min_prob"]) for p in pred.values())
    return {"names": sorted(usable), "train": len(train), "train_agree": round(agree / len(train), 3), "confident": confident,
            "track_smoothed": int(smoothed.sum()), "people": len(ids)}


def person_tracks(ids: list[str], people: dict[str, Any], vectors: np.ndarray, max_gap: int, min_iou: float, min_cos: float) -> np.ndarray:
    """Mã vệt cho từng crop: nối người ở hai ảnh liền kề cùng giờ (cách <= max_gap khung) khi box chồng nhau
    (IoU >= min_iou) và embedding giống (cosine >= min_cos); ghép 1-1 bằng Hungarian. Nối lỏng hơn làm sai
    tăng (bếp đông người), nên ngưỡng cố ý chặt."""
    from scipy.optimize import linear_sum_assignment

    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    parent = list(range(len(ids)))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_image: dict[str, list[int]] = {}
    for j, pid in enumerate(ids):
        by_image.setdefault(pid.rsplit("_", 1)[0], []).append(j)
    frames = sorted(by_image, key=lambda i: (i.split("_")[0], int(i.split("_")[1])))
    for a, b in zip(frames, frames[1:]):
        (ha, na), (hb, nb) = a.split("_"), b.split("_")
        if ha != hb or int(nb) - int(na) > max_gap:
            continue
        A, B = by_image[a], by_image[b]
        cost = np.full((len(A), len(B)), np.inf)
        for x, i in enumerate(A):
            for y, k in enumerate(B):
                iou, cos = box_iou(people[ids[i]]["box"], people[ids[k]]["box"]), float(unit[i] @ unit[k])
                if iou >= min_iou and cos >= min_cos:
                    cost[x, y] = -(iou + cos)
        finite = np.where(np.isfinite(cost), cost, 1e6)
        for x, y in zip(*linear_sum_assignment(finite)):
            if np.isfinite(cost[x, y]):
                parent[root(A[x])] = root(B[y])
    return np.array([root(j) for j in range(len(ids))])


def box_iou(a: list[float], b: list[float]) -> float:
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_pred(jsons_dir: Path) -> dict[str, dict[str, Any]]:
    path = jsons_dir / PRED_FILE
    return read_json(path) if path.exists() else {}


def final_identity(person_id: str, loaded: tuple, pred: dict[str, dict[str, Any]], min_prob: float) -> Optional[dict[str, Any]]:
    """Tên cuối: tên đặt riêng > mô hình (prob >= min_prob) > tên của cụm. Kèm cụm và dự đoán để soát."""
    ident = identity_of(person_id, *loaded)
    p = pred.get(person_id)
    out = dict(ident) if ident else {"cluster": None, "sim": None, "name": None, "source": None}
    if p:
        out["predicted"] = p
    if out["source"] == "person":
        return out
    if p and p["prob"] >= min_prob:
        out["name"], out["source"] = p["name"], "model"
    return out if (ident or p) else None


def update_outfits(jsons_dir: Path, attrs: dict[str, dict[str, Any]], names_of: dict[str, str]) -> dict[str, Any]:
    """Thêm vào ``outfits.yaml`` một dòng cho mỗi tên mới, điền sẵn bằng giá trị VLM gặp nhiều nhất. Không sửa dòng đã có."""
    import yaml
    from collections import Counter

    path = jsons_dir / OUTFITS_FILE
    doc = (yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None) or {}
    people = doc.setdefault("people", {})
    added = []
    for name in sorted(set(names_of.values())):
        if name in people:
            continue
        rows = [attrs[pid] for pid, n in names_of.items() if n == name and pid in attrs]
        outfit: dict[str, Any] = {}
        for key in OUTFIT_PREFILL:
            values = Counter(r.get(key) for r in rows if r.get(key) not in (None, "", "unknown"))
            outfit[key] = values.most_common(1)[0][0] if values else "unknown"
        outfit.update({k: "" for k in OUTFIT_MANUAL})
        people[name] = {"crops_described": len(rows), "outfits": [{"hours": "all", **outfit}]}
        added.append(name)
    if added or not path.exists():
        header = ("# Trang phục của từng người (python main.py enrich merge điền sẵn cho tên mới, từ câu trả lời VLM gặp nhiều nhất).\n"
                  "# Sửa tự do. Điền stripe_direction (horizontal/vertical), top_color_upper / top_color_lower (màu nửa trên / nửa\n"
                  "# dưới của áo) và description (câu mô tả đầy đủ). Người thay đồ giữa buổi: thêm một mục nữa trong outfits với\n"
                  "# hours: [\"08\", \"13\"] (giờ bắt đầu, giờ kết thúc, tính cả hai đầu). \"all\" = cả buổi. Dòng đã có không bị ghi đè.\n")
        path.write_text(header + yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return {"outfits_added": added}


def outfit_for(outfits: dict[str, Any], name: Optional[str], hour: str) -> Optional[dict[str, Any]]:
    """Mục trang phục của ``name`` khớp giờ ``hour`` (mục đầu tiên khớp)."""
    if not name or name not in outfits:
        return None
    for o in outfits[name].get("outfits", []):
        hours = o.get("hours", "all")
        span = [str(h).zfill(2) for h in hours] if isinstance(hours, list) else None   # yaml đọc 08 thành số 8
        if hours == "all" or (span and len(span) == 2 and span[0] <= hour <= span[1]):
            return {k: v for k, v in o.items() if k != "hours"}
    return None


def apply_outfit(attrs: Optional[dict[str, Any]], outfit: Optional[dict[str, Any]]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """(thuộc tính cuối, nguồn). Người đã có tên lấy ngoại hình từ outfits.yaml (người duyệt soát), vì VLM đoán
    từng crop sai nhiều (kính chỉ khớp 59%, loại áo 70%); holding / view vẫn theo VLM vì đổi theo từng ảnh."""
    fixed = {k: outfit[k] for k in OUTFIT_PREFILL if outfit and outfit.get(k) not in (None, "", "unknown")}
    if not fixed:
        return attrs, "vlm" if attrs else None
    return {**(attrs or {}), **fixed}, "outfit"


def load_outfits(jsons_dir: Path) -> dict[str, Any]:
    import yaml

    path = jsons_dir / OUTFITS_FILE
    return ((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("people") or {}) if path.exists() else {}


# --------------------------------------------------------------------------- cụm theo trang phục
# Chuẩn hoá câu trả lời tự do của VLM về ít giá trị để các crop cùng bộ đồ rơi vào cùng một khoá.
_TOP_KIND = {
    "sweater": ("sweater", "sweatshirt", "hoodie", "turtleneck", "jumper", "cardigan", "fleece"),
    "jacket": ("jacket", "vest", "suit", "coat", "blazer"),
    "t-shirt": ("t-shirt", "tshirt", "tee", "tank", "crop top", "jersey"),
    "polo": ("polo",),
    "shirt": ("shirt", "blouse", "button", "flannel", "long sleeve", "long-sleeve", "long"),
}
_COLOR = {
    "dark blue": "navy", "navy": "navy", "dark": "black", "dark gray": "gray", "grey": "gray", "dark grey": "gray",
    "beige": "beige", "cream": "beige", "tan": "beige", "khaki": "beige", "nude": "beige", "off-white": "white",
    "dark green": "green", "plaid": "unknown",
}


def _top_kind(top: str) -> str:
    top = (top or "").lower()
    for kind, words in _TOP_KIND.items():
        if any(w in top for w in words):
            return kind
    return "other" if top and top != "unknown" else "unknown"


def _color(value: str) -> str:
    v = (value or "unknown").lower().strip()
    if " and " in v:                      # "brown and white" -> "brown-white" (áo nhiều màu là dấu hiệu tốt)
        return "-".join(sorted(_COLOR.get(p.strip(), p.strip()) for p in v.split(" and ")))
    return _COLOR.get(v, v)


def outfit_key(a: dict[str, Any]) -> tuple[str, str, str, str]:
    """(giới, loại áo, màu áo, hoa văn). Quần không vào khoá vì VLM không thấy quần ở ~40% crop."""
    pattern = a.get("top_pattern") or "unknown"
    color = _color(a.get("top_color", ""))
    if (a.get("top_color") or "").lower() == "plaid" or "plaid" in (a.get("top") or "").lower():
        pattern = "checked"
    return (a.get("gender") or "unknown", _top_kind(a.get("top", "")), color, pattern)


GENDER_VI = {"male": "nam", "female": "nữ", "unknown": "?"}
PATTERN_VI = {"plain": "trơn", "checked": "caro", "striped": "sọc", "printed": "in hình", "unknown": ""}


def run_outfit_clusters(cfg: dict[str, Any], data_dir: Path, jsons_dir: Path, force: bool = False) -> dict[str, Any]:
    """Gom crop người theo trang phục VLM mô tả (``enrich identities``, ``identity.method: outfit``).

    Khoá = giới + loại áo + màu áo + hoa văn (đã chuẩn hoá). Khoá có ít nhất ``min_group`` crop thành một cụm;
    nhóm nhỏ hơn và crop chưa có mô tả (crop nhỏ) được xếp vào cụm có tâm SigLIP gần nhất. Mỗi cụm có nhãn đọc
    được, ví dụ "nam · sweater · white · sọc". Tên đã đặt riêng cho từng người không bị ảnh hưởng.
    """
    from .person_attrs import ATTRS_FILE, load_attrs

    icfg, log = cfg["enrich"]["identity"], get_logger()
    out_path = jsons_dir / IDENTITIES_FILE
    if out_path.exists() and not force:
        raise RuntimeError(f"Đã có {out_path}. Thêm --force để gom lại (tên đặt theo cụm sẽ lệch; tên đặt riêng từng người giữ nguyên).")
    attrs = load_attrs(jsons_dir / "adl_people" / ATTRS_FILE)
    if not attrs:
        raise RuntimeError("Chưa có mô tả trang phục (person_attrs.jsonl). Chạy bước attrs trước.")
    ids = [Path(str(n)).stem for n in read_json(data_dir / "people" / "ids.json")]
    vectors = np.load(data_dir / "people" / "embeddings.npy").astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    keys = {pid: outfit_key(attrs[pid]) for pid in ids if pid in attrs}
    counts: dict[tuple, int] = {}
    for k in keys.values():
        counts[k] = counts.get(k, 0) + 1
    groups = sorted((k for k, n in counts.items() if n >= int(icfg["min_group"]) and "unknown" not in k[:3]), key=lambda k: -counts[k])
    index = {k: g for g, k in enumerate(groups)}
    row = {pid: j for j, pid in enumerate(ids)}
    member_rows: list[list[int]] = [[] for _ in groups]
    for pid, k in keys.items():
        if k in index:
            member_rows[index[k]].append(row[pid])
    centers = np.stack([vectors[m].mean(0) for m in member_rows])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    sims = vectors @ centers.T
    assign: dict[str, dict[str, Any]] = {}
    clusters: dict[str, dict[str, Any]] = {}
    for g, k in enumerate(groups):
        clusters[f"o{g:02d}"] = {"label": " · ".join(x for x in (GENDER_VI.get(k[0], k[0]), k[1], k[2], PATTERN_VI.get(k[3], k[3])) if x), "key": list(k)}
    for j, pid in enumerate(ids):
        k = keys.get(pid)
        g = index[k] if k in index else int(sims[j].argmax())
        assign[pid] = {"cluster": f"o{g:02d}", "sim": round(float(sims[j, g]), 3), "by": "outfit" if k in index else "nearest"}
    for c in clusters:
        members = [pid for pid, a in assign.items() if a["cluster"] == c]
        members.sort(key=lambda pid: (assign[pid]["by"] != "outfit", -assign[pid]["sim"]))
        clusters[c].update(size=len(members), examples=members[: int(icfg["examples"])])
    write_json(out_path, {"k": len(clusters), "model": "outfit-attrs", "clusters": clusters, "assign": assign})
    direct = sum(a["by"] == "outfit" for a in assign.values())
    log.info("identities: %d cụm trang phục, %d crop theo mô tả, %d crop xếp theo SigLIP", len(clusters), direct, len(assign) - direct)
    return {"clusters": len(clusters), "by_outfit": direct, "by_nearest": len(assign) - direct, "largest": [(clusters[c]["label"], clusters[c]["size"]) for c in list(clusters)[:8]]}


def _spherical_kmeans(x: np.ndarray, k: int, seed: int, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """K-means theo cosine bằng numpy (sklearn KMeans bị Application Control chặn DLL trên máy này)."""
    centers = x[np.random.default_rng(seed).choice(len(x), k, replace=False)]
    for _ in range(iters):
        labels = (x @ centers.T).argmax(1)
        centers = np.stack([x[labels == g].mean(0) if (labels == g).any() else centers[g] for g in range(k)])
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)
    return (x @ centers.T).argmax(1), centers


def run_name_clusters(cfg: dict[str, Any], data_dir: Path, jsons_dir: Path, force: bool = False) -> dict[str, Any]:
    """Gom lại theo tên người duyệt đã đặt (``enrich identities``, ``identity.method: name``).

    Cụm theo VLM gộp nhầm người (khoá trang phục do VLM đoán từng crop). Ở đây học tên từ crop đã đặt tên
    (``train_predict``), rồi: crop mô hình chắc (prob >= min_prob) vào cụm của tên đó; crop chưa chắc gom bằng
    KMeans trên SigLIP thành ``uncertain_clusters`` cụm "chưa chắc", để người mới / người đứng khuất nằm chung
    với nhau thay vì bị ép vào tên gần nhất. Tên đặt riêng từng người không đổi.
    """
    icfg, log = cfg["enrich"]["identity"], get_logger()
    out_path = jsons_dir / IDENTITIES_FILE
    if out_path.exists() and not force:
        raise RuntimeError(f"Đã có {out_path}. Thêm --force để gom lại (tên đặt theo cụm sẽ lệch; tên đặt riêng từng người giữ nguyên).")
    training = train_predict(cfg, data_dir, jsons_dir)
    if "skipped" in training:
        raise RuntimeError(f"Chưa học được tên: {training['skipped']}")
    pred = load_pred(jsons_dir)
    ids = [Path(str(n)).stem for n in read_json(data_dir / "people" / "ids.json")]
    min_prob = float(icfg["min_prob"])
    names = training["names"]
    code = {n: f"n{i:02d}" for i, n in enumerate(names)}
    clusters: dict[str, dict[str, Any]] = {code[n]: {"label": n} for n in names}
    assign: dict[str, dict[str, Any]] = {}
    unsure = [j for j, pid in enumerate(ids) if pred[pid]["prob"] < min_prob]
    for j, pid in enumerate(ids):
        p = pred[pid]
        if p["prob"] >= min_prob:
            assign[pid] = {"cluster": code[p["name"]], "sim": p["prob"], "by": "name"}
    k = min(int(icfg.get("uncertain_clusters", 20)), len(unsure))
    if k:
        vectors = np.load(data_dir / "people" / "embeddings.npy").astype(np.float32)[unsure]
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        labels, centers = _spherical_kmeans(vectors, k, int(icfg["seed"]))
        sims = (vectors * centers[labels]).sum(1)
        for g in range(k):
            near = [pred[ids[unsure[r]]]["name"] for r in np.where(labels == g)[0]] or ["?"]
            top = max(set(near), key=near.count)
            clusters[f"u{g:02d}"] = {"label": f"chưa chắc · gần {top}"}
        for r, j in enumerate(unsure):
            assign[ids[j]] = {"cluster": f"u{int(labels[r]):02d}", "sim": round(float(sims[r]), 3), "by": "uncertain"}
    for c in clusters:
        members = sorted((pid for pid, a in assign.items() if a["cluster"] == c), key=lambda pid: -assign[pid]["sim"])
        clusters[c].update(size=len(members), examples=members[: int(icfg["examples"])])
    write_json(out_path, {"k": len(clusters), "model": "names+logreg", "clusters": clusters, "assign": assign})
    log.info("identities: %d cụm theo tên, %d cụm chưa chắc (%d crop)", len(names), k, len(unsure))
    return {"names": len(names), "confident": len(ids) - len(unsure), "uncertain": len(unsure), "uncertain_clusters": k, "train_agree": training["train_agree"]}

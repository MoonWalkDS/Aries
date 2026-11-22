"""Gộp nhiều bằng chứng thành nhãn hành động cho từng người (``python main.py enrich actions``).

VLM gán hành động chỉ nhìn crop người và sai nhiều (khớp người duyệt ~36%). Ở đây mỗi cặp (người, lớp) có các
đặc trưng: VLM có chọn lớp này không, SigLIP2 crop người so với 7 câu mô tả của lớp, người có đứng đúng vùng của
lớp không (tủ lạnh, bồn rửa, bếp, góc cà phê, thùng rác), trạng thái vùng trong ảnh (vd tủ lạnh mở), câu mô tả ảnh
của thầy (dataset/raw/teacher_metadata.csv) có nhắc hành động đó không: cả ảnh, và riêng mệnh đề tả người này
(ghép mệnh đề với người theo SigLIP2 crop-câu và vùng đứng), tần suất lớp và xác suất vùng theo lớp. Đồ vật thầy
đếm (cốc, bát, laptop) đã thử: trên 145 nhãn chỉ đúng 3/171 lần bật, nên bỏ. Một logistic regression dùng chung cho mọi lớp học "cặp này đúng" từ nhãn người
duyệt; lớp điểm cao nhất là dự đoán.

Đo chéo trên nhãn người duyệt (chia theo ảnh) được ghi vào log mỗi lần chạy (``cv_acc`` so với ``vlm_acc``).
Soát mắt 16 ca đổi ngẫu nhiên thì đổi hết là quá tay (người rửa chén ở bồn bị đổi sang sơ chế), nên ``enrich merge``
chỉ thay nhãn máy khi gộp hơn VLM một khoảng > ``override_gap``.
Ghi ``adl_people/action_pred.json``: {person_id: {label, gap, vlm}}.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .utils import get_logger, read_json, resolve_from_config, write_json

PRED_FILE = "action_pred.json"
EXTRA_FILE = "extra_boxes.json"   # ứng viên box thừa theo số người của thầy
HINT_FILE = "teacher_hints.json"  # người được câu mô tả của thầy tả đang làm một hành động
# vùng bắt buộc của hành động gắn với một chỗ (trùng label_viewer/store.js ACTION_ZONE, thêm thùng rác)
PLACE = {"opening the refrigerator": "fridge", "washing dishes": "sink", "cooking at the stove": "stove",
         "making a hot drink": "coffee_corner", "throwing away trash": "trash"}
# trạng thái vùng gợi ý hành động
ROI_HINT = {"fridge open": "opening the refrigerator", "person at the sink": "washing dishes", "dishes in the sink": "washing dishes",
            "person at the stove": "cooking at the stove", "pots on the stove": "cooking at the stove", "person at the bins": "throwing away trash",
            "person at the coffee corner": "making a hot drink", "food being prepared": "preparing food", "meal served": "eating"}
# từ khoá trong câu mô tả ảnh (tiếng Anh) của thầy
CAPTION_KW = {
    "opening the refrigerator": r"fridge|refrigerator", "cooking at the stove": r"cook|stove|stir|fry|\bpan\b",
    "preparing food": r"prepar|cutting|chopp|slic|peel", "washing dishes": r"wash|dish|sink|rins", "eating": r"\beat|meal",
    "drinking": r"drink|sipping", "using a phone": r"phone", "using a laptop": r"laptop", "talking": r"talk|convers|chat",
    "walking": r"walk", "carrying something": r"carry|holding", "standing idle": r"standing|observ|waiting|watching",
    "sitting": r"sitting|seated", "getting items from a cabinet": r"cabinet|drawer|shelf|reaching", "cleaning": r"clean|wip",
    "throwing away trash": r"trash|\bbin\b|garbage", "making a hot drink": r"coffee|\btea\b|kettle|\bmug",
}
FEATURES = ("vlm", "siglip", "zone_ok", "zone_bad", "caption", "caption_person", "roi", "prior", "zone_prior")
# Luật cứng theo vùng: chỉ những ca chắc chắn. Vùng là hình chữ nhật chồng nhau, nên các hành động khác dùng
# xác suất theo vùng học từ nhãn người duyệt (zone_prior), không cấm cứng (vd pha đồ uống có cả ở đảo bếp).
FRIDGE = "opening the refrigerator"
HARD_ZONES = {"opening the refrigerator": {"fridge"}, "washing dishes": {"sink", "counter_left"}}
# Bồn rửa nằm sát camera, bàn và bếp ở phía sau: người rửa chén quay mặt về camera. Người quay lưng ở vùng bồn rửa
# đang làm ở bàn/bếp phía sau, không phải rửa chén. Quay lưng = VLM ghi view "back", hoặc view trống/không rõ mà
# SigLIP2 (crop gần câu "nhìn từ phía sau" hơn câu "nhìn thẳng") > BACK_MIN. Ngưỡng 0,03: trong 935 người VLM ghi
# "front" chỉ 0,6% vượt, trong 2.447 người "back" 56% vượt.
FACE_FRONT_ONLY = {"washing dishes"}
BACK_MIN = 0.03
BACK_TEXT = ["a photo of a person seen from behind, back turned to the camera", "the back of a person standing at a kitchen counter",
             "a person facing away from the camera"]
FRONT_TEXT = ["a photo of a person facing the camera, face visible", "a person looking down at the sink, face toward the camera",
              "a person seen from the front"]

# --- tách câu mô tả của thầy thành mệnh đề, mỗi mệnh đề tả một người ---
_SPLIT = re.compile(r"(?<=[.!?])\s+|;\s*|:\s*|,?\s+while\s+|,\s+and\s+(?=another\b|the other\b|a third\b|one\b)"
                    r"|,\s+(?=another\b|the other\b|a third\b|a fourth\b)")
_ONE = re.compile(r"\b(person|individual|man|woman|someone|figure|chef|lady|guy|another|the other|one)\b")
_GROUP = re.compile(r"\b(two|three|four|five|six|seven|eight|several|multiple|a group of)\s+(\w+\s+)?(people|individuals|persons|men|women)\b"
                    r"|\bno (one|people|person)\b|\bpeople (are|is)\b|\bthey\b|\bboth\b")
# chỗ nhắc trong mệnh đề -> vùng của mình (bàn ăn nằm trong vùng đảo bếp: VLM "ngồi" thì 488/650 ở island)
LOC_ZONES = [(r"sink", {"sink", "counter_left"}), (r"stove|oven|\bpot\b|\bpan\b", {"stove", "oven_towel"}),
             (r"fridge|refrigerator", {"fridge"}), (r"island|dining|table|stool|seated", {"island"}),
             (r"sofa|couch|living", {"living_area"}), (r"coffee|kettle", {"coffee_corner"}), (r"trash|\bbin\b|garbage", {"trash"})]
# Từ khoá hành động trong mệnh đề tả một người: chỉ động từ, không lấy chỗ đứng. Câu của thầy hay chỉ tả chỗ
# ("another is at the stove", "near the refrigerator"); coi chỗ là hành động thì ai đứng cạnh tủ lạnh cũng thành mở tủ.
CLAUSE_KW = {
    "opening the refrigerator": r"open\w*\s+(a|the|its)?\s*(\w+\s+){0,3}(fridge|refrigerator)|(fridge|refrigerator)('s)?\s+(door\s+)?(is\s+|stands\s+|wide\s+)*open\b"
                                r"|(fridge|refrigerator)[^.]*door open|reach\w* into the (fridge|refrigerator)|inside the (fridge|refrigerator)|from the (fridge|refrigerator)",
    "cooking at the stove": r"\bcook|stirr|\bfry|frying|saut", "preparing food": r"prepar|cutting|chopp|slic|peel|\bmix",
    "washing dishes": r"wash|rins|scrub", "eating": r"\beat|\bmeal", "drinking": r"drink|sipping",
    "using a phone": r"phone", "using a laptop": r"laptop|computer|typing", "talking": r"talk|convers|chat|speak",
    "walking": r"walk|moving|entering|leaving", "carrying something": r"carry", "standing idle": r"observ|waiting|watching|idle|resting",
    "sitting": r"sitting|seated|\bsits\b", "getting items from a cabinet": r"cabinet|drawer|reaching", "cleaning": r"clean|wip",
    "throwing away trash": r"throw|trash|garbage|dispos", "making a hot drink": r"coffee|\btea\b|kettle|pouring",
}
MATCH_ALPHA, MATCH_BETA = 30.0, 1.0   # độ dốc softmax: chênh cosine crop-câu, và vùng khớp / lệch


def person_clauses(text: str) -> list[str]:
    """Các mệnh đề tả đúng một người ("one person stands near the counter, possibly preparing food")."""
    return [c.strip() for c in _SPLIT.split(text.lower()) if c and _ONE.search(c) and not _GROUP.search(c)]


def clause_zone_hit(clause: str, zone: str | None) -> float:
    """+1 người đứng ở chỗ mệnh đề nhắc, -1 mệnh đề nhắc chỗ khác, 0 mệnh đề không nhắc chỗ nào."""
    places = set().union(*[z for pat, z in LOC_ZONES if re.search(pat, clause)])
    return 0.0 if not places else 1.0 if zone in places else -1.0


def clause_weights(text_sim: np.ndarray, zone_hit: np.ndarray, alpha: float = MATCH_ALPHA, beta: float = MATCH_BETA) -> np.ndarray:
    """(mệnh đề, người) -> trọng số mềm: mỗi mệnh đề chia cho những người trong ảnh, tổng theo người = 1."""
    s = alpha * (text_sim - text_sim.mean(1, keepdims=True)) + beta * zone_hit
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _teacher(path: Path) -> dict[str, dict[str, Any]]:
    """{image_id: {caption, persons}} từ bảng của thầy."""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["imageID"].replace("day1_Kitchen_", ""): {"caption": (r.get("description") or "").lower(),
                                                            "persons": json.loads(r.get("class_count") or "{}").get("person", 0)}
                for r in csv.DictReader(f)}


def _match(names: list[str], emb: np.ndarray, zones: list[str | None], clauses: list[str], clause_emb: np.ndarray,
           alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """(trọng số mệnh đề-người, mệnh đề nhắc lớp nào)."""
    hit = np.array([[bool(re.search(CLAUSE_KW[c], t)) for c in names] for t in clauses], dtype=float)
    zone_hit = np.array([[clause_zone_hit(t, z) for z in zones] for t in clauses])
    return clause_weights(clause_emb @ emb.T, zone_hit, alpha, beta), hit


def caption_person(names: list[str], people: list[str], emb: np.ndarray, zones: list[str | None], clauses: list[str],
                   clause_emb: np.ndarray, alpha: float = MATCH_ALPHA, beta: float = MATCH_BETA) -> np.ndarray:
    """(số người, số lớp): mức mệnh đề của thầy tả người này nhắc tới lớp đó (0..1)."""
    if not clauses or not people:
        return np.zeros((len(people), len(names)))
    w, hit = _match(names, emb, zones, clauses, clause_emb, alpha, beta)
    return (w[:, :, None] * hit[:, None, :]).max(0)


def teacher_hints(names: list[str], people: list[str], emb: np.ndarray, zones: list[str | None], back: set[str],
                  clauses: list[str], clause_emb: np.ndarray, alpha: float = MATCH_ALPHA, beta: float = MATCH_BETA) -> dict[str, dict[str, str]]:
    """Mỗi mệnh đề của thầy nhắc một hành động -> người khớp nhất trong số người được phép làm hành động đó
    (đúng vùng HARD_ZONES, không quay lưng với FACE_FRONT_ONLY). {person_id: {label, text}} cho hàng duyệt "Thầy tả"."""
    if not clauses or not people:
        return {}
    w, hit = _match(names, emb, zones, clauses, clause_emb, alpha, beta)
    out = {}
    for j, k in zip(*np.nonzero(hit)):
        c = names[k]
        ok = [i for i, p in enumerate(people) if zones[i] in HARD_ZONES.get(c, {zones[i]}) and not (c in FACE_FRONT_ONLY and p in back)]
        if ok:
            i = max(ok, key=lambda i: w[j, i])
            out.setdefault(people[i], {"label": c, "text": clauses[j]})
    return out


def features(names: list[str], pid: str, vlm: dict[str, str], sig_row: np.ndarray, zone: str | None,
             caption_classes: set[str], cap_person_row: np.ndarray, roi_classes: set[str],
             log_prior: np.ndarray) -> np.ndarray:
    """(số lớp, len(FEATURES) - 1) cho một người; zone_prior được nối sau vì nó học từ tập huấn luyện."""
    return np.array([[vlm.get(pid) == c, sig_row[k], float(zone == PLACE[c]) if c in PLACE else 0.5,
                      float(c in PLACE and zone != PLACE[c]), c in caption_classes, cap_person_row[k], c in roi_classes, log_prior[k]]
                     for k, c in enumerate(names)], dtype=float)


def zone_table(train: list[str], human: dict[str, str], zone: dict[str, str | None], zones: list[str], names: list[str]) -> np.ndarray:
    """log P(vùng | lớp) từ nhãn người duyệt, làm trơn +1; hàng cuối là "không có vùng".

    Không dùng P(lớp | vùng): nó kéo mọi người về lớp hay gặp ("đứng không làm gì") và gần như xoá lớp hiếm.
    Đo trên 145 nhãn (chia theo ảnh): đúng 45,5% -> 50,3%; P(lớp | vùng) đúng 52,4% nhưng độ phủ mỗi lớp thấp hơn."""
    counts = np.ones((len(zones) + 1, len(names)))
    for pid in train:
        counts[zones.index(zone[pid]) if zone.get(pid) in zones else -1, names.index(human[pid])] += 1
    return np.log(counts / counts.sum(0, keepdims=True))


def build(cfg: dict[str, Any], data_dir: Path, jsons_dir: Path, alpha: float = MATCH_ALPHA, beta: float = MATCH_BETA) -> dict[str, Any]:
    """Nạp mọi bằng chứng, trả đặc trưng gốc của từng người (chưa có zone_prior) và nhãn người duyệt."""
    from .adl import ClipTextEncoder, l2_normalize

    acfg = cfg["enrich"]["actions"]
    exports = Path(cfg["paths"]["exports"])
    labels_file = resolve_from_config(cfg, cfg["adl_profiles"]["people"]["labels_file"])
    classes = yaml.safe_load(labels_file.read_text(encoding="utf-8"))["classes"]
    names = [c["name"] for c in classes]
    store = jsons_dir / "adl_people"
    approved = read_json(store / "approved.json")
    human = {pid: e["label"] for pid, e in approved.items() if e.get("source") == "human" and e["label"] in names}
    vlm: dict[str, str] = {}
    for line in (store / "vlm_reviews.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("choice"):
            vlm[row["image_id"]] = row["choice"]
    ids = [Path(str(n)).stem for n in read_json(data_dir / "people" / "ids.json")]
    row = {pid: j for j, pid in enumerate(ids)}
    emb = l2_normalize(np.load(data_dir / "people" / "embeddings.npy").astype(np.float32))
    enc = ClipTextEncoder(cfg["adl"]["clip"]["model"], cfg["adl"]["clip"]["pretrained"], store / "text_embeddings.npz")
    sig = np.stack([(emb @ enc.encode(c["descriptions"]).T).max(1) for c in classes], 1)
    sig = (sig - sig.mean(1, keepdims=True)) / (sig.std(1, keepdims=True) + 1e-9)   # so lớp với nhau trong cùng crop
    back_score = dict(zip(ids, (emb @ enc.encode(BACK_TEXT).T).max(1) - (emb @ enc.encode(FRONT_TEXT).T).max(1)))
    zone, roi, people_of, back = {}, {}, {}, set()
    with open(exports / "castle_dataset.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            roi[r["image_id"]] = {ROI_HINT[v["state"]["label"]] for v in r["rois"].values() if v.get("state") and v["state"]["label"] in ROI_HINT}
            people_of[r["image_id"]] = [p["id"] for p in r["people"] if p["id"] in row]
            for p in r["people"]:
                zone[p["id"]] = p.get("zone")
                view = (p.get("attributes") or {}).get("view")
                if view == "back" or (view != "front" and back_score.get(p["id"], 0) > BACK_MIN):
                    back.add(p["id"])
    teacher = _teacher(resolve_from_config(cfg, acfg["teacher_metadata"]))
    clauses = {im: person_clauses(t["caption"]) for im, t in teacher.items()}
    uniq = sorted({c for cs in clauses.values() for c in cs})
    cemb = dict(zip(uniq, enc.encode(uniq))) if uniq else {}
    cap_p: dict[str, np.ndarray] = {}
    hints: dict[str, dict[str, str]] = {}
    for im, pids in people_of.items():
        cs = clauses.get(im, [])
        m = caption_person(names, pids, emb[[row[p] for p in pids]], [zone.get(p) for p in pids], cs,
                           np.stack([cemb[c] for c in cs]) if cs else np.zeros((0, emb.shape[1])), alpha, beta)
        cap_p.update(zip(pids, m))
        # câu tả cảnh không có chủ ngữ người ("the refrigerator door is open"): gợi ý người đứng ở vùng tủ lạnh
        if re.search(CLAUSE_KW[FRIDGE], teacher.get(im, {}).get("caption", "")) and not any(hints.get(p, {}).get("label") == FRIDGE for p in pids):
            hints.update({p: {"label": FRIDGE, "text": teacher[im]["caption"][:200]} for p in pids if zone.get(p) == "fridge" and p not in hints})
        if cs:
            hints.update(teacher_hints(names, pids, emb[[row[p] for p in pids]], [zone.get(p) for p in pids], back, cs,
                                       np.stack([cemb[c] for c in cs]), alpha, beta))
    prior = Counter(vlm.values())
    log_prior = np.log(np.array([prior[c] + 1 for c in names]) / (sum(prior.values()) + len(names)))
    zero = np.zeros(len(names))
    base = {}
    for pid in ids:
        if pid not in zone:      # box đã bị đánh dấu không phải người
            continue
        im = pid.rsplit("_", 1)[0]
        t = teacher.get(im, {})
        cap_cls = {c for c in names if re.search(CAPTION_KW[c], t.get("caption", ""))}
        base[pid] = features(names, pid, vlm, sig[row[pid]], zone.get(pid), cap_cls, cap_p.get(pid, zero), roi.get(im, set()), log_prior)
    return {"names": names, "human": {p: l for p, l in human.items() if p in base}, "vlm": vlm, "zone": zone, "base": base,
            "zones": sorted({z for z in zone.values() if z}), "back": back, "teacher": teacher, "hints": hints, "clauses": clauses, "cap_person": cap_p}


def _stack(ctx: dict[str, Any], pids: list[str], zt: np.ndarray, drop: tuple[str, ...] = ()) -> np.ndarray:
    zones, keep = ctx["zones"], [i for i, f in enumerate(FEATURES) if f not in drop]
    rows = [np.concatenate([ctx["base"][p], zt[zones.index(ctx["zone"][p]) if ctx["zone"].get(p) in zones else -1][:, None]], 1)
            for p in pids]
    return np.stack(rows)[:, :, keep]


def _fit(ctx: dict[str, Any], train: list[str], drop: tuple[str, ...] = ()):
    from sklearn.linear_model import LogisticRegression

    names, human = ctx["names"], ctx["human"]
    zt = zone_table(train, human, ctx["zone"], ctx["zones"], names)
    X = _stack(ctx, train, zt, drop)
    y = np.array([names.index(human[p]) for p in train])
    target = (np.arange(len(names))[None, :] == y[:, None]).reshape(-1)
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X.reshape(-1, X.shape[2]), target)
    return model, zt


def _scores(ctx: dict[str, Any], model, zt: np.ndarray, pids: list[str], drop: tuple[str, ...] = ()) -> np.ndarray:
    X = _stack(ctx, pids, zt, drop)
    s = model.decision_function(X.reshape(-1, X.shape[2])).reshape(len(pids), -1)
    names = ctx["names"]
    for j, p in enumerate(pids):
        for c, allowed in HARD_ZONES.items():
            if ctx["zone"].get(p) not in allowed:
                s[j, names.index(c)] = -np.inf
        if p in ctx["back"]:
            for c in FACE_FRONT_ONLY:
                s[j, names.index(c)] = -np.inf
    return s


def cv_accuracy(ctx: dict[str, Any], drop: tuple[str, ...] = (), folds: int = 5) -> float:
    """Độ chính xác chéo trên nhãn người duyệt, chia theo ảnh (người cùng ảnh không vừa học vừa thi)."""
    from sklearn.model_selection import GroupKFold

    pids = sorted(ctx["human"])
    y = np.array([ctx["names"].index(ctx["human"][p]) for p in pids])
    pred = np.zeros(len(pids), int)
    for tr, te in GroupKFold(folds).split(pids, y, [p.rsplit("_", 1)[0] for p in pids]):
        model, zt = _fit(ctx, [pids[i] for i in tr], drop)
        pred[te] = _scores(ctx, model, zt, [pids[i] for i in te], drop).argmax(1)
    return float((pred == y).mean())


def run_actions(cfg: dict[str, Any], data_dir: Path, jsons_dir: Path) -> dict[str, Any]:
    log, acfg = get_logger(), cfg["enrich"]["actions"]
    exports = Path(cfg["paths"]["exports"])
    store = jsons_dir / "adl_people"
    ctx = build(cfg, data_dir, jsons_dir)
    names, human, vlm = ctx["names"], ctx["human"], ctx["vlm"]
    if len(human) < int(acfg["min_human"]):
        return {"skipped": f"cần >= {acfg['min_human']} nhãn người duyệt, đang có {len(human)}"}
    train = sorted(human)
    cv = cv_accuracy(ctx)
    vlm_acc = float(np.mean([vlm.get(p) == human[p] for p in train]))
    model, zt = _fit(ctx, train)
    pids = list(ctx["base"])
    S = _scores(ctx, model, zt, pids)
    pred, changed = {}, 0
    for pid, s in zip(pids, S):
        best = int(s.argmax())
        v = names.index(vlm[pid]) if vlm.get(pid) in names else None
        gap = float(s[best] - s[v]) if v is not None and np.isfinite(s[v]) else float(s[best] - np.sort(s)[-2]) + 99 * (v is not None)
        pred[pid] = {"label": names[best], "gap": round(gap, 3), "vlm": vlm.get(pid)}
        changed += v is not None and best != v and gap > float(acfg["override_gap"])
    write_json(store / PRED_FILE, pred)
    extra = extra_box_candidates(exports / "castle_dataset.jsonl", ctx["teacher"])
    write_json(store / EXTRA_FILE, extra)
    write_json(store / HINT_FILE, ctx["hints"])
    coef = dict(zip(FEATURES, np.round(model.coef_[0], 2).tolist()))
    log.info("actions: %d nhãn người duyệt (đo chéo %.3f, VLM %.3f), %d người; sẽ thay %d nhãn VLM (gap > %s); hệ số %s",
             len(train), cv, vlm_acc, len(pred), changed, acfg["override_gap"], coef)
    return {"train": len(train), "cv_acc": round(cv, 3), "vlm_acc": round(vlm_acc, 3), "people": len(pred),
            "would_override": changed, "extra_box_candidates": len(extra), "coef": coef}


def extra_box_candidates(dataset: Path, teacher: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Ảnh mình đếm nhiều người hơn bảng của thầy: lấy (thừa) box có điểm dò thấp nhất mỗi ảnh làm ứng viên
    "không phải người" cho người duyệt soát. {person_id: {score, ours, teacher}}."""
    if not teacher:
        return {}
    out = {}
    with open(dataset, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["image_id"] not in teacher:
                continue
            n = teacher[r["image_id"]]["persons"]
            for p in sorted(r["people"], key=lambda p: p["score"])[:max(0, len(r["people"]) - n)]:
                out[p["id"]] = {"score": p["score"], "ours": len(r["people"]), "teacher": n}
    return out


def load_pred(jsons_dir: Path) -> dict[str, dict[str, Any]]:
    path = jsons_dir / "adl_people" / PRED_FILE
    return read_json(path) if path.exists() else {}

"""Phần không cần model của engine text: bỏ dấu, BM25, tách đề thành cụm-người/cảnh, ghép cụm với người.

Tách đề theo luật (không gọi LLM lúc truy vấn): câu mở đầu bằng chủ ngữ người ("một người đàn ông", "người kia",
"thầy Linh") mở một cụm-người; câu đại từ ("anh ấy…") hoặc tả tiếp đồ mặc ("áo len có…") gộp vào cụm trước;
"những người khác…" là cụm-nhóm; còn lại là cảnh.
"""
from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from functools import lru_cache

import numpy as np

from engine import _fold as fold, names_in, split_sentences  # noqa: F401  (dùng lại, không chép)

_NUM = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}
_NOUN = r"(?:nguoi|dan ong|phu nu|dan ba|chang trai|co gai|con trai|con gai|ong|anh|chi|thay|nam|nu)"
_PLURAL = re.compile(rf"^(?:(?:co|chi co)\s+)?(\d+|{'|'.join(_NUM)})\s+{_NOUN}\b")
_SINGLE = re.compile(rf"^(?:(?:co|chi co)\s+)?(?:mot\s+)?(?:nguoi (?:kia|con lai|thu \w+|khac nua)\b|{_NOUN}\b)")
_CONTINUE = re.compile(r"^(?:anh|co|ong|ba|chi|cau)\s+(?:ay|ta)\b|^(?:nguoi nay|nguoi do|ao|quan|phan|toc|mu|non|kinh|giay|tay|ho)\b")
_GROUP = re.compile(r"^(?:nhung|cac|moi)?\s*nguoi\s+(?:khac|con lai|xung quanh)\b|^moi nguoi\b")
_LOCATED = re.compile(r"^(?:phia|o|ben|gan|canh|tren|trong|sau|truoc)\b[^,]{0,40}?\bco\s+(?=(?:\d+|mot|hai|ba|bon|nam|sau|bay|tam|chin|muoi)\s+nguoi\b)")
_ONLY = re.compile(r"^chi (?:co )?mot nguoi\b")
_NOBODY_ELSE = re.compile(r"^(?:trong bep )?khong (?:co )?(?:ai|nguoi nao) khac\b")
_NUMS = rf"(\d+|{'|'.join(_NUM)})"
_COUNT_ANY = re.compile(rf"(?<!\w)co\s+{_NUMS}\s+nguoi\b")       # "ảnh bếp có 3 người" giữa câu
_ZERO = re.compile(r"(?<!\w)(?:co 0 nguoi|0 nguoi|khong co (?:ai|mot ai|nguoi nao|nguoi|bong nguoi)|khong (?:mot )?bong nguoi"
                   r"|vang (?:nguoi|ve|tanh)|khong nguoi)(?! khac| nao khac)(?!\w)")
_PHOTO = ("ảnh", "bức", "hình", "tấm", "khung", "cảnh", "căn", "bếp")   # "Ảnh bếp…" bỏ dấu thành "anh bep", không phải "anh"
_MALE = ("đàn ông", "chàng trai", "con trai", "thầy", "ông", "anh", "cậu", "nam")
_FEMALE = ("phụ nữ", "đàn bà", "cô gái", "con gái", "cô", "chị", "bà", "nữ")
_MALE_PLAIN = ("dan ong", "chang trai", "con trai", "thay", "nam")      # không dấu: bỏ "co/ba/anh" vì trùng "có/ba/ánh"
_FEMALE_PLAIN = ("phu nu", "dan ba", "co gai", "con gai", "nu")


def tokens(text: str) -> list[str]:
    """Từ đã bỏ dấu cộng cặp từ liền nhau ("tu_lanh"), để cụm hai âm tiết khớp như một từ."""
    w = re.findall(r"\w+", fold(text))
    return w + [f"{a}_{b}" for a, b in zip(w, w[1:])]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.n = len(docs)
        lens = np.array([len(d) for d in docs], dtype=np.float32)
        norm = k1 * (1 - b + b * lens / max(lens.mean(), 1e-9))
        post: dict[str, tuple[list[int], list[float]]] = {}
        for i, d in enumerate(docs):
            for t, tf in Counter(d).items():
                rows, w = post.setdefault(t, ([], []))
                rows.append(i)
                w.append(tf * (k1 + 1) / (tf + norm[i]))
        self.post = {t: (np.array(r), np.array(w, dtype=np.float32) * math.log(1 + (self.n - len(r) + 0.5) / (len(r) + 0.5)))
                     for t, (r, w) in post.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        for t in set(query):
            if t in self.post:
                rows, w = self.post[t]
                out[rows] += w
        return out


def gender_of(clause: str) -> str | None:
    head = " ".join(clause.lower().split()[:5])
    plain = head == fold(head)
    for words, g in ((_MALE_PLAIN if plain else _MALE, "male"), (_FEMALE_PLAIN if plain else _FEMALE, "female")):
        if any(re.search(rf"(?<!\w){w}(?!\w)", head) for w in words):
            return g
    return None


def _is_photo(text: str) -> bool:
    words = text.lower().split()
    return bool(words) and words[0] in _PHOTO


def _starts_person(text: str, names: list[str]) -> bool:
    if _is_photo(text):
        return False
    f = fold(text).strip()
    return bool(_SINGLE.match(f)) or any(f.startswith(fold(n)) for n in names)


def _clauses(sentence: str, names: list[str]) -> list[str]:
    """Tách "Một người mặc áo xanh, người kia mặc áo caro" thành hai vế; chỉ tách khi vế sau có chủ ngữ người."""
    parts = re.split(r"\s*[,;]\s*|\s+(?:và|còn|va)\s+", sentence)
    out = [parts[0]]
    for p in parts[1:]:
        if _starts_person(p, names) and not _GROUP.match(fold(p)):
            out.append(p)
        else:
            out[-1] = f"{out[-1]}, {p}"
    return out


def parse_query(text: str, names: list[str]) -> dict:
    sentences = split_sentences(text)
    q = {"sentences": sentences, "names": names_in(" ".join(sentences), names), "count": None, "count_min": 0,
         "people": [], "group": None, "scene": []}
    plural_gender, plural_text, exact = None, None, False
    for i, s in enumerate(sentences):
        for c in _clauses(s, names):
            f = _LOCATED.sub("", fold(c).strip())   # "phía sau có một người…" -> "một người…"
            if _ZERO.search(f) and not _NOBODY_ELSE.match(f):
                q["count"] = 0
                q["scene"].append(c)
                continue
            m = _PLURAL.match(f)
            if not m and q["count"] is None and not _starts_person(c, names) and (m2 := _COUNT_ANY.search(f)):
                q["count"] = int(m2.group(1)) if m2.group(1).isdigit() else _NUM[m2.group(1)]
                q["scene"].append(c)
                continue
            n = (int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1)]) if m else 0
            if n >= 2 and i == 0 and not q["people"]:     # câu mở đầu "Hai người đàn ông…" = tổng số người
                q["count"], plural_gender, plural_text = n, gender_of(c), c
                q["scene"].append(c)
            elif n >= 2:                                  # "phía xa có ba người…" = n người nữa, ghép như cụm-nhóm
                q["group"] = {"text": c, "n": n}
            elif _ONLY.match(f):
                if q["people"]:
                    exact = True
                else:
                    q["count"] = 1
                    q["people"].append({"text": c, "gender": gender_of(c), "name": None})
            elif _NOBODY_ELSE.match(f):
                exact = True
            elif _GROUP.match(f):
                g = q["group"] or {"text": "", "n": 1}
                q["group"] = {"text": f"{g['text']} {c}".strip(), "n": g["n"]}
            elif _CONTINUE.match(f) and q["people"]:
                q["people"][-1]["text"] += f" {c}"
            elif (_starts_person(f, names) and not _is_photo(c)) or _CONTINUE.match(f):
                q["people"].append({"text": c, "gender": gender_of(c) or plural_gender, "name": None})
            else:
                q["scene"].append(c)
    if plural_text:
        q["scene"].remove(plural_text)
        if not q["people"]:     # "Hai người đàn ông đang nấu ăn": n cụm giống nhau, ghép vào n người khác nhau
            q["people"] = [{"text": plural_text, "gender": plural_gender, "name": None} for _ in range(q["count"])]
        else:                   # hành động/vị trí của câu số nhiều áp cho từng người được tả sau đó
            for p in q["people"]:
                p["text"] += f" {plural_text}"
    if q["count"] == 0:                     # "có 0 người": mọi cụm-người là tách nhầm
        q["scene"] += [p["text"] for p in q["people"]] + ([q["group"]["text"]] if q["group"] else [])
        q["people"], q["group"] = [], None
    for p in q["people"]:
        found = names_in(p["text"], q["names"])
        p["name"] = found[0] if found else None
    q["count_min"] = len(q["people"]) + (q["group"]["n"] if q["group"] else 0)
    if exact:                               # "Không có ai khác" / "Chỉ có một người": tổng = số người đã tả
        q["count"] = q["count_min"]
    return q


@lru_cache(maxsize=None)
def _perms(n: int, c: int) -> np.ndarray:
    return np.array(list(itertools.permutations(range(n), c)), dtype=np.int16).reshape(-1, c)


def assign(sim: np.ndarray, floor: float, keep: int = 3) -> tuple[float, list[int | None]]:
    """Ghép một-một cụm (hàng) với người (cột) cho tổng điểm cao nhất; cụm không có người nhận `floor`.

    Mỗi cụm chỉ xét `keep` người khớp nhất (≤ 4 cụm × 3 = 12 cột) nên vét cạn hoán vị vẫn nhanh.
    """
    c, p = sim.shape
    if c == 0:
        return 0.0, []
    cols = sorted({int(j) for row in sim for j in np.argsort(-row)[:keep]})
    m = np.hstack([sim[:, cols], np.full((c, c), floor)])   # c cột "không có ai" dùng được một lần mỗi cột
    perms = _perms(m.shape[1], c)
    totals = m[np.arange(c), perms].sum(1)
    best = perms[int(totals.argmax())]
    return float(totals.max()), [cols[j] if j < len(cols) else None for j in best]

"""Tách đề tiếng Việt thành cấu trúc mà engine dùng: số người, tên, cụm-người, cụm-nhóm, cụm-cảnh.

Chỉ dùng luật, không gọi LLM lúc truy vấn: câu mở đầu bằng chủ ngữ người ("một người đàn ông", "người kia",
"thầy Linh") mở một cụm-người; câu đại từ ("anh ấy…") hoặc tả tiếp đồ mặc ("áo len có…") gộp vào cụm trước;
"những người khác…" là cụm-nhóm; còn lại là cảnh. Các mẫu regex chạy trên chữ đã bỏ dấu.
"""
from __future__ import annotations

import re

from common import find_names, remove_accents, split_sentences

_NUMBER_WORDS = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}
_NUMBER = rf"(\d+|{'|'.join(_NUMBER_WORDS)})"
_PERSON_NOUN = r"(?:nguoi|dan ong|phu nu|dan ba|chang trai|co gai|con trai|con gai|ong|anh|chi|thay|nam|nu)"
_PLURAL_SUBJECT = re.compile(rf"^(?:(?:co|chi co)\s+)?{_NUMBER}\s+{_PERSON_NOUN}\b")          # "hai người đàn ông…"
_SINGLE_SUBJECT = re.compile(rf"^(?:(?:co|chi co)\s+)?(?:mot\s+)?(?:nguoi (?:kia|con lai|thu \w+|khac nua)\b|{_PERSON_NOUN}\b)")
_CONTINUES_PREVIOUS = re.compile(r"^(?:anh|co|ong|ba|chi|cau)\s+(?:ay|ta)\b|^(?:nguoi nay|nguoi do|ao|quan|phan|toc|mu|non|kinh|giay|tay|ho)\b")
_OTHER_PEOPLE = re.compile(r"^(?:nhung|cac|moi)?\s*nguoi\s+(?:khac|con lai|xung quanh)\b|^moi nguoi\b")
_LOCATION_PREFIX = re.compile(rf"^(?:phia|o|ben|gan|canh|tren|trong|sau|truoc)\b[^,]{{0,40}}?\bco\s+(?={_NUMBER}\s+nguoi\b)")
_ONLY_ONE_PERSON = re.compile(r"^chi (?:co )?mot nguoi\b")
_NOBODY_ELSE = re.compile(r"^(?:trong bep )?khong (?:co )?(?:ai|nguoi nao) khac\b")
_COUNT_IN_SENTENCE = re.compile(rf"(?<!\w)co\s+{_NUMBER}\s+nguoi\b")                          # "ảnh bếp có 3 người" giữa câu
_ZERO_PEOPLE = re.compile(r"(?<!\w)(?:co 0 nguoi|0 nguoi|khong co (?:ai|mot ai|nguoi nao|nguoi|bong nguoi)|khong (?:mot )?bong nguoi"
                          r"|vang (?:nguoi|ve|tanh)|khong nguoi)(?! khac| nao khac)(?!\w)")
_PHOTO_WORDS = ("ảnh", "bức", "hình", "tấm", "khung", "cảnh", "căn", "bếp")   # "Ảnh bếp…" bỏ dấu thành "anh bep", không phải "anh"
_MALE_WORDS = ("đàn ông", "chàng trai", "con trai", "thầy", "ông", "anh", "cậu", "nam")
_FEMALE_WORDS = ("phụ nữ", "đàn bà", "cô gái", "con gái", "cô", "chị", "bà", "nữ")
_MALE_WORDS_NO_ACCENT = ("dan ong", "chang trai", "con trai", "thay", "nam")      # không dấu: bỏ "co/ba/anh" vì trùng "có/ba/ánh"
_FEMALE_WORDS_NO_ACCENT = ("phu nu", "dan ba", "co gai", "con gai", "nu")


def _to_number(word: str) -> int:
    return int(word) if word.isdigit() else _NUMBER_WORDS[word]


def detect_gender(clause: str) -> str | None:
    """"male" / "female" theo chủ ngữ (5 từ đầu), None nếu không rõ."""
    head = " ".join(clause.lower().split()[:5])
    no_accent = head == remove_accents(head)
    for words, gender in ((_MALE_WORDS_NO_ACCENT if no_accent else _MALE_WORDS, "male"),
                          (_FEMALE_WORDS_NO_ACCENT if no_accent else _FEMALE_WORDS, "female")):
        if any(re.search(rf"(?<!\w){w}(?!\w)", head) for w in words):
            return gender
    return None


def _is_photo_subject(text: str) -> bool:
    words = text.lower().split()
    return bool(words) and words[0] in _PHOTO_WORDS


def _starts_with_person(text: str, known_names: list[str]) -> bool:
    if _is_photo_subject(text):
        return False
    plain = remove_accents(text).strip()
    return bool(_SINGLE_SUBJECT.match(plain)) or any(plain.startswith(remove_accents(n)) for n in known_names)


def _split_clauses(sentence: str, known_names: list[str]) -> list[str]:
    """Tách "Một người mặc áo xanh, người kia mặc áo caro" thành hai vế; chỉ tách khi vế sau có chủ ngữ người."""
    parts = re.split(r"\s*[,;]\s*|\s+(?:và|còn|va)\s+", sentence)
    clauses = [parts[0]]
    for part in parts[1:]:
        if _starts_with_person(part, known_names) and not _OTHER_PEOPLE.match(remove_accents(part)):
            clauses.append(part)
        else:
            clauses[-1] = f"{clauses[-1]}, {part}"
    return clauses


def parse_query(text: str, known_names: list[str]) -> dict:
    """Trả dict (giữ nguyên khoá vì web hiển thị):
    sentences, names, count (số người nêu rõ hoặc None), count_min, people [{text, gender, name}],
    group ({text, n} hoặc None), scene [cụm-cảnh]."""
    sentences = split_sentences(text)
    parsed = {"sentences": sentences, "names": find_names(" ".join(sentences), known_names), "count": None, "count_min": 0,
              "people": [], "group": None, "scene": []}
    plural_gender, plural_text, total_is_exact = None, None, False
    for sentence_index, sentence in enumerate(sentences):
        for clause in _split_clauses(sentence, known_names):
            plain = _LOCATION_PREFIX.sub("", remove_accents(clause).strip())   # "phía sau có một người…" -> "một người…"
            if _ZERO_PEOPLE.search(plain) and not _NOBODY_ELSE.match(plain):
                parsed["count"] = 0
                parsed["scene"].append(clause)
                continue
            plural = _PLURAL_SUBJECT.match(plain)
            if not plural and parsed["count"] is None and not _starts_with_person(clause, known_names) and (in_sentence := _COUNT_IN_SENTENCE.search(plain)):
                parsed["count"] = _to_number(in_sentence.group(1))
                parsed["scene"].append(clause)
                continue
            plural_count = _to_number(plural.group(1)) if plural else 0
            if plural_count >= 2 and sentence_index == 0 and not parsed["people"]:   # câu mở đầu "Hai người đàn ông…" = tổng số người
                parsed["count"], plural_gender, plural_text = plural_count, detect_gender(clause), clause
                parsed["scene"].append(clause)
            elif plural_count >= 2:                                                # "phía xa có ba người…" = n người nữa
                parsed["group"] = {"text": clause, "n": plural_count}
            elif _ONLY_ONE_PERSON.match(plain):
                if parsed["people"]:
                    total_is_exact = True
                else:
                    parsed["count"] = 1
                    parsed["people"].append({"text": clause, "gender": detect_gender(clause), "name": None})
            elif _NOBODY_ELSE.match(plain):
                total_is_exact = True
            elif _OTHER_PEOPLE.match(plain):
                group = parsed["group"] or {"text": "", "n": 1}
                parsed["group"] = {"text": f"{group['text']} {clause}".strip(), "n": group["n"]}
            elif _CONTINUES_PREVIOUS.match(plain) and parsed["people"]:
                parsed["people"][-1]["text"] += f" {clause}"
            elif (_starts_with_person(plain, known_names) and not _is_photo_subject(clause)) or _CONTINUES_PREVIOUS.match(plain):
                parsed["people"].append({"text": clause, "gender": detect_gender(clause) or plural_gender, "name": None})
            else:
                parsed["scene"].append(clause)
    if plural_text:
        parsed["scene"].remove(plural_text)
        if not parsed["people"]:     # "Hai người đàn ông đang nấu ăn": n cụm giống nhau, ghép vào n người khác nhau
            parsed["people"] = [{"text": plural_text, "gender": plural_gender, "name": None} for _ in range(parsed["count"])]
        else:                        # hành động/vị trí của câu số nhiều áp cho từng người được tả sau đó
            for person in parsed["people"]:
                person["text"] += f" {plural_text}"
    if parsed["count"] == 0:         # "có 0 người": mọi cụm-người là tách nhầm
        parsed["scene"] += [p["text"] for p in parsed["people"]] + ([parsed["group"]["text"]] if parsed["group"] else [])
        parsed["people"], parsed["group"] = [], None
    for person in parsed["people"]:
        found = find_names(person["text"], parsed["names"])
        person["name"] = found[0] if found else None
    parsed["count_min"] = len(parsed["people"]) + (parsed["group"]["n"] if parsed["group"] else 0)
    if total_is_exact:               # "Không có ai khác" / "Chỉ có một người": tổng = số người đã tả
        parsed["count"] = parsed["count_min"]
    return parsed

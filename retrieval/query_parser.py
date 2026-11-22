"""Tách đề tiếng Việt thành cấu trúc mà engine dùng: số người, tên, cụm-người, cụm-nhóm, cụm-cảnh.
Hỗ trợ cả mô hình SLM NuExtract-1.5-smol và bộ luật regex tiếng Việt (fallback).
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from common import REPO_ROOT, find_names, remove_accents, split_sentences

logger = logging.getLogger("retrieval.query_parser")

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


def parse_query_rule(text: str, known_names: list[str] = []) -> dict:
    """Fallback: Tách đề bằng luật Regex (giữ nguyên cấu trúc web)."""
    sentences = split_sentences(text)
    parsed = {"sentences": sentences, "names": find_names(" ".join(sentences), known_names), "count": None, "count_min": 0,
              "people": [], "group": None, "scene": []}
    plural_gender, plural_text, total_is_exact = None, None, False
    for sentence_index, sentence in enumerate(sentences):
        for clause in _split_clauses(sentence, known_names):
            plain = _LOCATION_PREFIX.sub("", remove_accents(clause).strip())
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
            if plural_count >= 2 and sentence_index == 0 and not parsed["people"]:
                parsed["count"], plural_gender, plural_text = plural_count, detect_gender(clause), clause
                parsed["scene"].append(clause)
            elif plural_count >= 2:
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
        if not parsed["people"]:
            parsed["people"] = [{"text": plural_text, "gender": plural_gender, "name": None} for _ in range(parsed["count"])]
        else:
            for person in parsed["people"]:
                person["text"] += f" {plural_text}"
    if parsed["count"] == 0:
        parsed["scene"] += [p["text"] for p in parsed["people"]] + ([parsed["group"]["text"]] if parsed["group"] else [])
        parsed["people"], parsed["group"] = [], None
    for person in parsed["people"]:
        found = find_names(person["text"], parsed["names"])
        person["name"] = found[0] if found else None
    parsed["count_min"] = len(parsed["people"]) + (parsed["group"]["n"] if parsed["group"] else 0)
    if total_is_exact:
        parsed["count"] = parsed["count_min"]

    # Normalize fields
    parsed["so_nguoi"] = parsed["count"]
    parsed["danh_sach_nguoi"] = [p["text"] for p in parsed["people"]]
    parsed["vat_the_trang_thai"] = parsed["scene"]
    parsed["raw_query"] = text
    return parsed


class NuExtractSLM:
    """SLM Parser sử dụng mô hình NuExtract-1.5-smol."""

    _instance = None

    def __init__(self, model_dir: Path | str | None = None):
        if model_dir is None:
            model_dir = REPO_ROOT / "model" / "NuExtract-1.5-smol"
        self.model_dir = Path(model_dir)
        self.device = None
        self.tokenizer = None
        self.model = None
        self.failed = False
        self._cache: dict[str, dict] = {}

    def load(self):
        if self.model is not None or self.failed:
            return
        if not self.model_dir.exists():
            logger.warning("NuExtract model directory not found at %s", self.model_dir)
            self.failed = True
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading NuExtract from %s on %s...", self.model_dir, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_dir),
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self.model.eval()
            logger.info("NuExtract model loaded successfully!")
        except Exception as exc:
            logger.warning("Failed to load NuExtract: %s", exc)
            self.failed = True

    def extract(self, query: str) -> dict | None:
        clean_q = query.strip()
        if clean_q in self._cache:
            return self._cache[clean_q]

        self.load()
        if self.model is None or self.tokenizer is None:
            return None
        try:
            import torch

            # Prompt chỉ dẫn chi tiết theo yêu cầu của chuyên gia parsing câu truy vấn
            instruction = """Bạn là chuyên gia parsing câu truy vấn, nhiệm vụ của bạn là parsing câu truy vấn đầu vào thành 3 thành phần: (a) so_nguoi - kiểm tra số lượng người yêu cầu trong truy vấn:
- Nếu là số lượng cố định chính xác (ví dụ: "1 người", "hai người", "duy nhất") -> ghi số nguyên (ví dụ: 1, 2).
- Nếu có điều kiện cận biên, so sánh (ví dụ: "tối thiểu 2", "ít nhất 2", "từ 2 trở lên", "nhiều hơn 1") -> ghi rõ dạng toán tử: ">=2" (hoặc ">1").
- Nếu hoàn toàn không đề cập số lượng -> ghi "khong_de_cap" (b) danh_sach_nguoi - tự động truy xuất theo người, tự chia truy vấn theo người và kèm theo miêu tả vi_tri, mau_ao (c) vat_the_trang_thai - trích xuất các vật thể có trong truy vấn. Ví dụ: Với truy vấn "Một người đàn ông mặc áo vest đen đứng sau đảo bếp" thì output đầu ra là: {"so_nguoi": 1, "danh_sach_nguoi": ["người đàn ông mặc vest đen"], "vat_the_trang_thai" :["bếp"]}"""
            schema_str = '{"so_nguoi":"","danh_sach_nguoi":[],"vat_the_trang_thai":[]}'
            example_input = "Một người đàn ông mặc áo vest đen đứng sau đảo bếp"
            example_output = '{"so_nguoi": 1, "danh_sach_nguoi": ["người đàn ông mặc vest đen"], "vat_the_trang_thai": ["bếp"]}'

            prompt = (
                f"<|input|>\n"
                f"### Instruction:\n{instruction}\n"
                f"### Template:\n{schema_str}\n"
                f"### Example 1\n### Text:\n{example_input}\n### Output:\n{example_output}\n"
                f"### Text:\n{clean_q}\n<|output|>\n"
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            full_text = self.tokenizer.decode(out[0], skip_special_tokens=False)
            if "<|output|>" in full_text:
                gen_text = full_text.split("<|output|>")[-1].replace("<|endoftext|>", "").strip()
            else:
                gen_text = full_text.strip()

            res = None
            m = re.search(r"\{.*\}", gen_text, re.DOTALL)
            if m:
                try:
                    res = json.loads(m.group(0))
                except Exception:
                    pass

            # Dự phòng tự đóng ngoặc nếu chuỗi bị cắt do chạm ngưỡng 100 tokens
            if res is None and "{" in gen_text:
                s = gen_text[gen_text.find("{"):].strip()
                for suffix in ('"}', '"]}', ']}', '}'):
                    try:
                        res = json.loads(s + suffix)
                        break
                    except Exception:
                        pass

            if res is not None:
                self._cache[clean_q] = res
                return res
        except Exception as exc:
            logger.warning("NuExtract extraction failed (%s), falling back to rule parser", exc)
        return None


_SLM_PARSER: NuExtractSLM | None = None


def get_slm_parser() -> NuExtractSLM:
    global _SLM_PARSER
    if _SLM_PARSER is None:
        _SLM_PARSER = NuExtractSLM()
    return _SLM_PARSER


def parse_query(text: str, known_names: list[str] = []) -> dict:
    """Tách câu đề thành cấu trúc hoàn chỉnh bằng NuExtract SLM (fallback rule)."""
    slm = get_slm_parser()
    data = slm.extract(text)

    if data is None:
        return parse_query_rule(text, known_names)

    # Map SLM output
    raw_so_nguoi = data.get("so_nguoi")
    so_nguoi = None
    if raw_so_nguoi is not None and str(raw_so_nguoi).lower() not in ("khong_de_cap", "null", ""):
        try:
            so_nguoi = int(raw_so_nguoi)
        except (ValueError, TypeError):
            so_nguoi = None

    raw_vat_the = data.get("vat_the_trang_thai", [])
    if isinstance(raw_vat_the, list):
        vat_the_trang_thai = [str(x) for x in raw_vat_the]
    else:
        vat_the_trang_thai = []

    people_objs = []
    danh_sach_nguoi = []
    raw_people = data.get("danh_sach_nguoi", [])
    if isinstance(raw_people, list):
        for p in raw_people:
            if isinstance(p, str):
                danh_sach_nguoi.append(p)
                people_objs.append({"text": p, "gender": detect_gender(p), "name": None, "mau_ao": "", "vi_tri": ""})
            elif isinstance(p, dict):
                mo_ta = p.get("mo_ta") or p.get("hanh_dong") or ""
                danh_sach_nguoi.append(mo_ta)
                people_objs.append({
                    "text": mo_ta,
                    "gender": detect_gender(mo_ta),
                    "name": None,
                    "mau_ao": p.get("mau_ao", ""),
                    "vi_tri": p.get("vi_tri", ""),
                })

    count_val = so_nguoi if so_nguoi is not None else len(people_objs) if people_objs else None

    return {
        "sentences": split_sentences(text),
        "names": find_names(text, known_names) if known_names else [],
        "count": count_val,
        "count_min": count_val or 0,
        "people": people_objs,
        "group": None,
        "scene": vat_the_trang_thai,
        "so_nguoi": so_nguoi,
        "danh_sach_nguoi": danh_sach_nguoi,
        "vat_the_trang_thai": vat_the_trang_thai,
        "raw_query": text,
        "engine": "NuExtract-1.5-smol",
    }

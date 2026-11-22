import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import remove_accents  # noqa: E402
from query_parser import parse_query  # noqa: E402
from text_match import BM25, assign_one_to_one, tokenize  # noqa: E402

Q_09_1574 = ("Một người đàn ông mở cửa tủ lạnh. Anh ấy mặt áo len. Áo len có vài sọc ngang. Phần trên áo màu sáng "
             "trong khi phần dưới màu đen. Thầy Linh cũng có trong bức ảnh đó mặc áo len màu xanh. "
             "Những người khác như đang rửa chén")
NAMES = ["Áo len 2 màu", "Linh", "Sơ mi xanh"]


def test_fold_and_tokenize():
    assert remove_accents("Tủ lạnh Đỏ") == "tu lanh do"
    assert tokenize("Tủ lạnh") == ["tu", "lanh", "tu_lanh"]


def test_bm25_prefers_phrase_match():
    bm = BM25([tokenize("tủ bếp và lạnh"), tokenize("mở tủ lạnh"), tokenize("bồn rửa")])
    s = bm.score(tokenize("tủ lạnh"))
    assert s.shape == (3,) and s.argmax() == 1 and s[2] == 0


def test_parse_multi_person_query():
    q = parse_query(Q_09_1574, NAMES)
    assert q["names"] == ["Linh"]
    assert len(q["people"]) == 2
    assert q["people"][0]["gender"] == "male" and "sọc ngang" in q["people"][0]["text"]
    assert "Linh" in q["people"][1]["text"] and q["people"][1]["name"] == "Linh"
    assert "rửa chén" in q["group"]["text"] and q["group"]["n"] == 1
    assert q["count"] is None and q["count_min"] == 3


def test_parse_count_and_split_clauses():
    q = parse_query("Hai người đàn ông đứng cạnh nhau ở góc xa của đảo bếp. Một người mặc áo xanh, người kia mặc áo caro.", NAMES)
    assert q["count"] == 2
    assert [p["text"].split(" Hai")[0] for p in q["people"]] == ["Một người mặc áo xanh", "người kia mặc áo caro."]
    assert all(p["gender"] == "male" for p in q["people"])   # giới tính của chủ ngữ số nhiều


def test_parse_unaccented_and_scene_only():
    q = parse_query("nguoi phu nu dang rua chen. Tren dao bep co dua", NAMES)
    assert q["people"][0]["gender"] == "female" and q["scene"] == ["Tren dao bep co dua"]
    assert parse_query("Trên đảo bếp có quả dứa.", NAMES)["people"] == []
    assert parse_query("   ", NAMES)["sentences"] == []


def test_assign_one_to_one_with_floor():
    score, pairs = assign_one_to_one(np.array([[1.0, 0.0], [0.9, 0.2]]), unmatched_score=-1.0)
    assert np.isclose(score, 1.2) and pairs == [0, 1]
    score, pairs = assign_one_to_one(np.array([[0.5], [0.4], [0.3]]), unmatched_score=-1.0)
    assert np.isclose(score, 0.5 - 2.0) and pairs == [0, None, None]
    assert assign_one_to_one(np.zeros((0, 3)), unmatched_score=-1.0) == (0.0, [])


def test_parse_located_people_and_exact_count():
    q = parse_query("Một người phụ nữ mặc áo đen đang đi. Phía xa có ba người đang chuẩn bị đồ ăn", NAMES)
    assert len(q["people"]) == 1 and q["group"]["n"] == 3 and q["count_min"] == 4 and q["scene"] == []
    q = parse_query("Một người đàn ông đeo kính mở tủ lạnh. Phía sau có một người mặc áo caro. Không có ai khác trong bếp.", NAMES)
    assert [p["text"] for p in q["people"]][1] == "Phía sau có một người mặc áo caro." and q["count"] == 2
    q = parse_query("Một người đàn ông đang nấu ăn. Chỉ có một người trong bếp.", NAMES)
    assert len(q["people"]) == 1 and q["count"] == 1


def test_plural_opening_is_shared_by_each_person():
    q = parse_query("Hai người đàn ông đứng cạnh đảo bếp đang chuẩn bị đồ ăn. Một người mặc áo xanh, người kia mặc áo xám.", NAMES)
    assert all("chuẩn bị đồ ăn" in p["text"] for p in q["people"]) and q["scene"] == []


ZERO = ("Ảnh bếp có 0 người, toàn bộ các mốc cố định gồm 3 đèn thả, đảo bếp và bồn rửa, các ghế đảo, 2 thùng rác, "
        "trong khi trên đảo bếp chất nhiều nồi inox quanh bồn rửa, hai quả dứa, đĩa trắng.")


def test_zero_people_query_has_no_person_clause():
    q = parse_query(ZERO, NAMES)
    assert q["count"] == 0 and q["people"] == [] and q["group"] is None and q["count_min"] == 0
    for text in ("Không có ai trong bếp. Trên đảo có dứa.", "Bếp vắng người, trên đảo có dứa.", "Không có người nào trong ảnh."):
        assert parse_query(text, NAMES)["count"] == 0, text


def test_count_anywhere_and_photo_is_not_a_person():
    q = parse_query("Ảnh bếp có 3 người, trên đảo có dứa.", NAMES)
    assert q["count"] == 3 and q["people"] == []
    assert parse_query("Bức ảnh chụp lúc đông người, có năm người trong bếp.", NAMES)["count"] == 5
    assert parse_query("Anh ấy mở tủ lạnh.", NAMES)["people"][0]["gender"] == "male"   # "anh ấy" vẫn là người
    q = parse_query("Một người đàn ông nấu ăn. Không có ai khác trong bếp.", NAMES)
    assert q["count"] == 1 and len(q["people"]) == 1

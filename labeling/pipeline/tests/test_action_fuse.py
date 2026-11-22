import numpy as np

from modules.action_fuse import caption_person, clause_zone_hit, person_clauses


def test_person_clauses_splits_people_and_skips_groups():
    text = ("In the kitchen, three individuals are engaged in various activities. One person stands near the counter, "
            "possibly preparing food, while another is at the sink area. No people are eating.")
    assert person_clauses(text) == ["one person stands near the counter, possibly preparing food", "another is at the sink area."]


def test_clause_zone_hit():
    assert clause_zone_hit("another is at the sink area.", "sink") == 1.0
    assert clause_zone_hit("another is at the sink area.", "fridge") == -1.0
    assert clause_zone_hit("another person is cutting", "fridge") == 0.0


def test_caption_person_uses_verbs_not_places():
    names = ["washing dishes", "opening the refrigerator", "preparing food"]
    emb = np.eye(2)                       # hai người, crop trực giao
    clauses = ["one person is washing dishes at the sink", "another stands near the refrigerator"]
    m = caption_person(names, ["a", "b"], emb, ["sink", "fridge"], clauses, np.eye(2))
    assert m[0, 0] > 0.9                  # mệnh đề rửa chén về người ở bồn rửa
    assert m[1, 1] == 0                   # đứng cạnh tủ lạnh không phải mở tủ lạnh


def test_back_facing_person_cannot_be_washing_dishes():
    from modules.action_fuse import _scores

    class Flat:   # mô hình giả: mọi lớp cùng điểm
        def decision_function(self, X):
            return np.zeros(len(X))

    names = ["washing dishes", "opening the refrigerator", "preparing food"]
    base = {p: np.zeros((3, 9)) for p in ("front", "back")}
    ctx = {"names": names, "zones": ["sink"], "zone": {"front": "sink", "back": "sink"}, "base": base, "back": {"back"}}
    s = _scores(ctx, Flat(), np.zeros((2, 3)), ["front", "back"])
    assert np.isfinite(s[0, 0]) and s[1, 0] == -np.inf

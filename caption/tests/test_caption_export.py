from export import image_record, person_record


def _p(**kw):
    p = {"id": "08_1_p01", "zone_vi": "tủ lạnh", "attributes": {"gender": "male", "top": "shirt", "top_color": "blue"},
         "identity": {"name": "Sơ mi xanh", "source": "person"}, "outfit": {"description": "Nam, áo sơ mi xanh."},
         "action": {"label": "walking", "label_vi": "đang đi", "source": "vlm"}}
    p.update(kw)
    return p


def test_known_person_uses_outfit_description_and_confidence_levels():
    r = person_record(_p(), None)
    assert (r["tag"], r["mo_ta"], r["hanh_dong"], r["do_chac"]) == ("Sơ mi xanh", "Nam, áo sơ mi xanh.", "đang đi", "chưa chắc")
    assert person_record(_p(), {"label": "walking", "text": "a man walks"})["do_chac"] == "khá chắc"   # thầy tả đúng
    assert person_record(_p(action={"label": "walking", "label_vi": "đang đi", "source": "human"}), None)["do_chac"] == "chắc"


def test_unsure_identity_falls_back_to_attributes():
    r = person_record(_p(identity={"name": "Sơ mi xanh", "source": "model", "predicted": {"prob": 0.4}}), None)
    assert r["tag"] is None and r["mo_ta"] == "nam, blue shirt"


def test_image_record_drops_empty_roi_states():
    rec = {"image_id": "08_1", "hour": "08", "people": [], "rois": {
        "fridge": {"name_vi": "tủ lạnh", "state": {"label_vi": "tủ lạnh đóng"}},
        "stove": {"name_vi": "bếp", "state": {"label_vi": "có nồi chảo trên bếp"}}}}
    assert image_record(rec, {"08_1": "A kitchen."}, {})["vung_bep"] == ["bếp: có nồi chảo trên bếp"]


def test_parse_reply_keeps_only_valid_lines_of_the_batch():
    from batch import parse_reply

    text = 'Đây là kết quả:\n```\n{"image_id": "08_1", "caption": "Một người."}\n{"image_id": "99_9", "caption": "lạ"}\n{"image_id": "08_2", "caption": ""}\nbad {json\n```'
    assert parse_reply(text, {"08_1", "08_2"}) == {"08_1": "Một người."}

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_metadata import convert, person_sentence  # noqa: E402

PERSON = {"id": "14_265_p01", "box": [1, 2, 3, 4], "zone_vi": "bồn rửa trên đảo bếp",
          "action": {"label_vi": "rửa chén bát"}, "identity": {"name": "Sơ mi xanh"},
          "outfit": {"gender": "male", "description": "Nam, tóc ngắn, áo sơ mi trơn màu xanh dương."}}
RECORD = {"image_id": "14_265", "num_people": 1, "people": [PERSON], "manual": [],
          "objects": [{"label": "bottle"}, {"label": "bottle"}, {"label": "pineapple"}],
          "rois": {"sink": {"name_vi": "bồn rửa trên đảo bếp", "state": {"label_vi": "có người ở bồn rửa"}},
                   "living_area": {"name_vi": "khu ghế sofa cạnh cửa sổ", "state": None}}}


def test_person_sentence_nam_style():
    assert person_sentence(PERSON) == "người đàn ông, tóc ngắn, áo sơ mi trơn màu xanh dương, rửa chén bát, ở bồn rửa trên đảo bếp"
    assert person_sentence({"identity": {"name": "Linh"}}) == "người, linh"


def test_convert_keeps_nam_fields_and_everything_else():
    m = convert(RECORD, "caption vi", {"so_nguoi": 2, "description": "desc", "row": {"hour": "14"}})
    assert m["so_nguoi"] == 1 and len(m["danh_sach_nguoi"]) == 1
    assert m["vat_the_trang_thai"] == ["bồn rửa", "ghế sofa", "có người ở bồn rửa", "chai", "quả dứa"]
    assert m["vat_the"][0] == {"ten": "chai", "ten_en": "bottle", "so_luong": 2}
    assert m["ten_nguoi"] == ["Sơ mi xanh"] and m["so_nguoi_thay"] == 2 and m["caption"] == "caption vi"
    assert m["goc"] is RECORD

import os

import pytest

import huggingface_data


def test_patterns_include_images_only_when_asked():
    assert huggingface_data.IMAGES not in huggingface_data.patterns(with_images=False)
    assert huggingface_data.IMAGES in huggingface_data.patterns(with_images=True)
    assert all(p.startswith(("data/", "labeling/annotations/")) for p in huggingface_data.patterns(with_images=True))
    # nhãn người duyệt nằm trong git: không bao giờ tải đè từ HF
    assert not any(p.endswith(("approved.json", "rejected.json", "identity_names.json")) for p in huggingface_data.patterns(True))


def test_link_images_rejects_folder_without_jpg(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        huggingface_data.link_images(tmp_path / "empty", root=tmp_path / "repo")
    with pytest.raises(SystemExit):
        huggingface_data.link_images(tmp_path / "missing", root=tmp_path / "repo")


def test_link_images_creates_link_and_refuses_to_replace_real_folder(tmp_path):
    src = tmp_path / "my_images"
    src.mkdir()
    (src / "08_1.jpg").write_bytes(b"x")
    repo = tmp_path / "repo"
    dest = huggingface_data.link_images(src, root=repo)
    assert (dest / "08_1.jpg").read_bytes() == b"x" and os.path.realpath(dest) == os.path.realpath(src)
    huggingface_data.link_images(src, root=repo)                       # chạy lại: thay liên kết cũ, không lỗi
    real = tmp_path / "repo2" / "data" / "dataset" / "images"
    real.mkdir(parents=True)
    (real / "a.jpg").write_bytes(b"y")
    with pytest.raises(SystemExit):                            # thư mục ảnh thật đã tải: không xoá
        huggingface_data.link_images(src, root=tmp_path / "repo2")


def test_download_stops_with_login_hint_when_not_logged_in(monkeypatch, capsys):
    class NoLogin:
        def whoami(self):
            raise RuntimeError("401")

    monkeypatch.setattr(huggingface_data, "HfApi", NoLogin)
    with pytest.raises(SystemExit) as stop:
        huggingface_data.download("skip")
    assert stop.value.code == 2 and "auth login" in capsys.readouterr().out


def test_patterns_do_not_leak_into_subfolders():
    """HF lọc bằng fnmatch, nơi `*` khớp cả dấu `/`: mẫu không được kéo theo thư mục phụ."""
    import fnmatch
    hit = lambda path: any(fnmatch.fnmatch(path, p) for p in huggingface_data.patterns(with_images=False))  # noqa: E731
    assert hit("data/exports/castle_dataset.jsonl") and hit("data/intermediate/enrich/objects.json") and hit("data/intermediate/enrich/people/people.json")
    assert hit("labeling/annotations/identities.json") and not hit("labeling/annotations/identities.outfit-vlm.json")   # merge cần cụm danh tính
    assert not hit("data/exports/hf/data/captions.jsonl") and not hit("data/intermediate/enrich/_shards/people_0001.json")
    assert not hit("data/exports/caption/codex/run.json") and not hit("data/dataset/images/08_1.jpg")


def test_big_folder_goes_up_as_one_zip_and_comes_back(tmp_path):
    """HF chỉ cho tối đa 10.000 file mỗi thư mục: thư mục crop người được đẩy dưới dạng một file zip rồi giải nén khi tải."""
    import fnmatch
    hit = lambda path: any(fnmatch.fnmatch(path, p) for p in huggingface_data.patterns(with_images=True))  # noqa: E731
    crops = "data/intermediate/enrich/people/crops"
    assert not hit(f"{crops}/08_1_p01.jpg") and hit(f"{crops}.zip") and hit("data/intermediate/enrich/people/people.json")
    folder = tmp_path / crops
    folder.mkdir(parents=True)
    for n in range(3):
        (folder / f"08_{n}_p01.jpg").write_bytes(bytes([n]) * 10)
    huggingface_data.pack_archives(tmp_path)
    assert (tmp_path / f"{crops}.zip").exists()
    import shutil
    shutil.rmtree(folder)
    huggingface_data.unpack_archives(tmp_path)
    assert sorted(p.name for p in folder.iterdir()) == ["08_0_p01.jpg", "08_1_p01.jpg", "08_2_p01.jpg"]
    assert (folder / "08_2_p01.jpg").read_bytes() == bytes([2]) * 10

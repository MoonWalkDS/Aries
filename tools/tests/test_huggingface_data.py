import os

import pytest

import huggingface_data


def test_patterns_include_images_only_when_asked():
    assert "data/dataset/images.zip" not in huggingface_data.patterns(with_images=False)
    assert "data/dataset/images.zip" in huggingface_data.patterns(with_images=True)
    assert all(p.startswith("data/") for p in huggingface_data.patterns(True))
    assert not any("intermediate/" in p for p in huggingface_data.patterns(True))


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


def test_skip_does_not_unpack_existing_image_archive(tmp_path):
    import zipfile

    dataset = tmp_path / "data/dataset"
    dataset.mkdir(parents=True)
    for name in ("thumbs", "images"):
        with zipfile.ZipFile(dataset / f"{name}.zip", "w") as archive:
            archive.writestr("frame.jpg", b"frame")
    huggingface_data.unpack_archives(tmp_path, with_images=False)
    assert (dataset / "thumbs/frame.jpg").is_file()
    assert not (dataset / "images").exists()

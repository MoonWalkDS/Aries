"""Tải / đẩy dữ liệu của repo từ dataset riêng tư trên Hugging Face.

    python tools/huggingface_data.py download                     # hỏi cách lấy ảnh gốc
    python tools/huggingface_data.py download --images download   # tải cả 5.000 ảnh gốc
    python tools/huggingface_data.py download --images skip       # không lấy ảnh (web truy vấn sẽ thiếu ảnh)
    python tools/huggingface_data.py download --images D:/castle/images   # dùng thư mục ảnh có sẵn (tạo liên kết, không chép)
    python tools/huggingface_data.py upload                       # chủ dữ liệu: đẩy data/ + file máy sinh trong labeling/annotations/

Trong repo HF, file nằm đúng đường dẫn như trong repo này (data/..., labeling/annotations/...), nên tải về là đúng chỗ.
Nhãn người duyệt (approved/rejected/identity_names...) nằm trong git, không bao giờ bị tải đè.
Chạy lại `download` chỉ tải phần còn thiếu hoặc đã đổi.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]   # gốc repo
REPO = "DS314/img314"
IMAGES = "data/dataset/images/**"
# HF lọc bằng fnmatch: `*` khớp cả dấu `/`, nên file ở thư mục gốc phải ghi tên đầy đủ, chỉ dùng `/**` cho cả cây thư mục.
DATA = [
    "data/dataset/thumbs/**",
    "data/dataset/raw/teacher_metadata.csv",
    "data/exports/castle_dataset.jsonl", "data/exports/people.csv", "data/exports/objects.csv", "data/exports/rois.csv", "data/exports/summary.json",
    "data/exports/caption/inputs.jsonl", "data/exports/caption/captions.jsonl", "data/exports/caption/prompt_vi.md",
    "data/exports/embeddings/**",
    "data/exports/search_index/**",
    "data/intermediate/adl/**",
    "data/intermediate/enrich/_detections.json", "data/intermediate/enrich/objects.json", "data/intermediate/enrich/objects_owl.json", "data/intermediate/enrich/objects_static.json",
    "data/intermediate/enrich/people/people.json", "data/intermediate/enrich/people/ids.json",
    "data/intermediate/enrich/people/embeddings.npy", "data/intermediate/enrich/people/build_stats.json",
    "data/intermediate/enrich/rois/**",
    # file máy sinh nằm trong labeling/annotations/ (gitignore); nhãn người duyệt thì ở trong git
    "labeling/annotations/enriched/**",
    "labeling/annotations/adl/centroids.npz", "labeling/annotations/adl/text_embeddings.npz",
    "labeling/annotations/adl_people/centroids.npz", "labeling/annotations/adl_people/text_embeddings.npz",
    "labeling/annotations/adl_people/action_pred.json", "labeling/annotations/adl_people/extra_boxes.json", "labeling/annotations/adl_people/teacher_hints.json",
    "labeling/annotations/identity_pred.json", "labeling/annotations/identities.json",
] + [f"labeling/annotations/adl_roi-{roi}/{f}" for roi in ("coffee_corner", "counter_left", "floor_front", "fridge", "island", "living_area",
                                                   "oven_towel", "sink", "stove", "trash") for f in ("centroids.npz", "text_embeddings.npz")]


# HF chỉ cho tối đa 10.000 file mỗi thư mục: thư mục nào vượt thì đẩy dưới dạng một file <thư mục>.zip, tải về thì giải nén.
ARCHIVED_DIRS = ["data/intermediate/enrich/people/crops"]   # 12.819 crop người


def patterns(with_images: bool) -> list[str]:
    return DATA + [f"{d}.zip" for d in ARCHIVED_DIRS] + ([IMAGES] if with_images else [])


def pack_archives(root: Path = ROOT) -> None:
    """Nén mỗi thư mục trong ARCHIVED_DIRS thành <thư mục>.zip (không nén lại ảnh jpg, chỉ gom)."""
    import zipfile

    for rel in ARCHIVED_DIRS:
        folder = root / rel
        if not folder.is_dir():
            continue
        with zipfile.ZipFile(root / f"{rel}.zip", "w", zipfile.ZIP_STORED) as archive:
            for path in sorted(folder.iterdir()):
                if path.is_file():
                    archive.write(path, path.name)


def unpack_archives(root: Path = ROOT) -> None:
    """Giải nén <thư mục>.zip vừa tải về nếu thư mục chưa có đủ file. Giữ file zip để lần tải sau không tải lại."""
    import zipfile

    for rel in ARCHIVED_DIRS:
        archive_path, folder = root / f"{rel}.zip", root / rel
        if not archive_path.exists():
            continue
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if folder.is_dir() and sum(1 for _ in folder.iterdir()) >= len(names):
                continue
            folder.mkdir(parents=True, exist_ok=True)
            archive.extractall(folder)
            print(f"Đã giải nén {len(names)} file vào {rel}/")


def _login_or_exit(api: HfApi) -> None:
    try:
        api.whoami()
    except Exception:  # noqa: BLE001 - chưa đăng nhập, token sai, mất mạng: đều cần người dùng xử lý
        print(f"Chưa đăng nhập Hugging Face, hoặc token không có quyền đọc {REPO} (repo riêng tư).\n"
              "Chạy lệnh sau rồi dán token (Settings > Access Tokens trên huggingface.co):\n\n"
              "    .venv\\Scripts\\hf.exe auth login\n\n"
              "Hoặc đặt biến môi trường HF_TOKEN. Sau đó chạy lại tools\\download_data.bat.")
        sys.exit(2)


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def link_images(src: Path, root: Path = ROOT) -> Path:
    """Trỏ data/dataset/images tới thư mục ảnh có sẵn (junction trên Windows, symlink nơi khác)."""
    src = Path(src).expanduser().resolve()
    if not src.is_dir() or not any(src.glob("*.jpg")):
        sys.exit(f"Không thấy ảnh .jpg trong {src}. Cần thư mục chứa trực tiếp các file như 08_1002.jpg.")
    dest = root / "data" / "dataset" / "images"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _is_link(dest):
        os.unlink(dest) if dest.is_symlink() else os.rmdir(dest)   # chỉ gỡ liên kết, không đụng ảnh gốc
    elif dest.exists():
        if any(dest.iterdir()):
            sys.exit(f"{dest} đã có ảnh thật (tải từ HF?). Xoá thư mục đó trước nếu muốn dùng thư mục ảnh khác.")
        dest.rmdir()
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dest), str(src)], check=True, capture_output=True)
    else:
        dest.symlink_to(src, target_is_directory=True)
    return dest


def _ask_images() -> str:
    print("Ảnh gốc 5.000 ảnh (khoảng 1,2 GB):\n  1. Tải từ Hugging Face\n  2. Dùng thư mục ảnh có sẵn trên máy\n  3. Bỏ qua")
    while True:
        choice = input("Chọn 1, 2 hoặc 3: ").strip()
        if choice == "1":
            return "download"
        if choice == "3":
            return "skip"
        if choice == "2":
            return input("Đường dẫn thư mục ảnh: ").strip().strip('"')


def download(images: str, root: Path = ROOT) -> None:
    api = HfApi()
    _login_or_exit(api)
    with_images = images == "download"
    if images not in ("download", "skip"):
        link_images(Path(images), root)      # kiểm tra đường dẫn trước khi tải để sai thì báo ngay
    print(f"Tải {REPO} vào {root} ...")
    snapshot_download(REPO, repo_type="dataset", local_dir=root, allow_patterns=patterns(with_images))
    unpack_archives(root)
    if not (root / "data" / "exports" / "castle_dataset.jsonl").exists():
        sys.exit(f"{REPO} chưa có dữ liệu của repo này (chưa ai chạy tools\\upload_data.bat).")
    print("Xong. Ảnh gốc:", {"download": "đã tải", "skip": "bỏ qua"}.get(images, f"liên kết tới {images}"))


def upload(root: Path = ROOT, yes: bool = False) -> None:
    api = HfApi()
    _login_or_exit(api)
    missing = [p for p in ("data/exports/castle_dataset.jsonl", "data/exports/search_index/images.json") if not (root / p).exists()]
    if missing:
        sys.exit(f"Thiếu {missing}: chưa đủ dữ liệu để đẩy.")
    pack_archives(root)
    files = files_to_upload(root)
    images = sum(1 for rel, _ in files if rel.startswith("data/dataset/images/"))
    print(f"Sẽ đẩy {len(files)} file ({sum(size for _, size in files) / 1e9:.2f} GB), trong đó {images} ảnh gốc.")
    if not yes and input(f"Đẩy lên {REPO} (riêng tư)? Gõ yes để tiếp tục: ").strip().lower() != "yes":
        sys.exit("Đã huỷ, không đẩy gì.")
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=root, allow_patterns=patterns(with_images=True),
                      delete_patterns=[f"{d}/*" for d in ARCHIVED_DIRS],   # bỏ file lẻ của thư mục nay đã đi dạng zip
                      commit_message="Dữ liệu cho castle-retrieval (data/ + file máy sinh trong labeling/annotations/)")
    print("Đã đẩy xong.")


def files_to_upload(root: Path = ROOT) -> list[tuple[str, int]]:
    """(đường dẫn tương đối, kích thước) của các file khớp mẫu, lọc đúng như upload_folder (fnmatch)."""
    import fnmatch

    wanted = patterns(with_images=True)
    out = []
    for path in root.glob("**/*"):                    # glob của pathlib đi qua cả junction ảnh gốc
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, p) for p in wanted):
                out.append((rel, path.stat().st_size))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--images", help="download | skip | <đường dẫn thư mục ảnh có sẵn>; bỏ trống thì hỏi")
    u = sub.add_parser("upload")
    u.add_argument("--yes", action="store_true", help="không hỏi xác nhận")
    args = ap.parse_args()
    if args.cmd == "download":
        download(args.images or _ask_images())
    else:
        upload(yes=args.yes)


if __name__ == "__main__":
    main()

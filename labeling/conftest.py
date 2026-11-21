"""pytest đặt thư mục này vào sys.path (vì có conftest.py), nên test import được `modules` và `main`
dù chạy `pytest` từ gốc repo hay từ labeling/."""
import importlib.util

# Test labeling cần nhóm "Lõi" trong labeling/requirements.txt. Máy chỉ cài requirements.txt của
# backend thì bỏ qua cả thư mục, để `pytest` ở gốc repo vẫn chạy được phần backend.
# pyyaml đã có sẵn qua uvicorn[standard], nên kiểm cả các gói chỉ labeling dùng.
if any(importlib.util.find_spec(m) is None for m in ("yaml", "tqdm")):
    collect_ignore_glob = ["*"]

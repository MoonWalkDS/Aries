@echo off
chcp 65001 >nul
setlocal
rem File nằm trong tools/: mọi lệnh chạy từ gốc repo.
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Chưa cài môi trường. Chạy tools\setup.bat trước.
  exit /b 1
)
set "PY=.venv\Scripts\python.exe"
set PYTHONIOENCODING=utf-8
rem Mở web truy vấn + nộp DRES ở http://127.0.0.1:8314 (tự dựng chỉ mục nếu chưa có).
if not exist "data\exports\castle_dataset.jsonl" (
  echo Chưa có dữ liệu. Chạy tools\download_data.bat trước.
  exit /b 1
)
if not exist "data\exports\search_index\images.json" (
  echo Chưa có chỉ mục: dựng lần đầu, mất vài phút ...
  "%PY%" retrieval\build_search_index.py || exit /b 1
)
rem Preload logic: Thiết lập cờ PRELOAD_MODEL và kích hoạt nạp trước Model vào VRAM
set PRELOAD_MODEL=1
echo [Preload] Kích hoạt nạp trước Model NuExtract vào VRAM khi khởi động...
"%PY%" retrieval\web_app.py --preload %*

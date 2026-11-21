@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Chưa cài môi trường. Chạy setup.bat trước.
  exit /b 1
)
set "PY=.venv\Scripts\python.exe"
set PYTHONIOENCODING=utf-8
rem Mở web truy vấn + nộp DRES ở http://127.0.0.1:8314 (tự dựng chỉ mục nếu chưa có).
if not exist "data\exports\castle_dataset.jsonl" (
  echo Chưa có dữ liệu. Chạy download_data.bat trước.
  exit /b 1
)
if not exist "data\exports\text_index\meta.json" (
  echo Chưa có chỉ mục: dựng lần đầu, mất vài phút ...
  "%PY%" retrieval\build_text_index.py || exit /b 1
)
"%PY%" retrieval\app.py %*

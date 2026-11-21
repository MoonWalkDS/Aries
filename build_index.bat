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
rem Dựng lại chỉ mục truy vấn (data/exports/text_index/) sau khi caption hoặc nhãn đổi.
rem Bản dịch caption đã có được giữ trong cache translations.jsonl, nên chạy lại nhanh.
"%PY%" retrieval\build_text_index.py
exit /b %errorlevel%

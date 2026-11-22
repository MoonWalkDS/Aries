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
rem Chỉ dành cho chủ dữ liệu: đẩy data/ và file máy sinh trong labeling/annotations/ lên Hugging Face (repo riêng tư).
rem Script hỏi xác nhận trước khi đẩy.
"%PY%" tools\huggingface_data.py upload %*
exit /b %errorlevel%

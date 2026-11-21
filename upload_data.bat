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
rem Chỉ dành cho chủ dữ liệu: đẩy data/ và file máy sinh trong annotations/ lên Hugging Face (repo riêng tư).
rem Script hỏi xác nhận trước khi đẩy.
"%PY%" scripts\hf_data.py upload %*
exit /b %errorlevel%

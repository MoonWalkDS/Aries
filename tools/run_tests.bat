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
"%PY%" -m pytest -q
exit /b %errorlevel%

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
rem Chạy toàn bộ test: Python (pytest) và viewer (node --test).
"%PY%" -m pytest -q || exit /b 1
where npm >nul 2>nul && if exist "viewer\node_modules" (pushd viewer & call npm test --silent & popd)

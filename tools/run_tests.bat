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
rem Chạy toàn bộ test: Python (pytest) và viewer (node --test).
"%PY%" -m pytest -q || exit /b 1
where node >nul 2>nul && node --test retrieval/tests/settings.test.js || exit /b 1
where npm >nul 2>nul && if exist "labeling\label_viewer\node_modules" (pushd labeling\label_viewer & call npm test --silent & popd)

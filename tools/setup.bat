@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv || exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt || exit /b 1
if not exist ".env" copy ".env.example" ".env" >nul

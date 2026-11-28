@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Chay tools\setup.bat truoc.
  exit /b 1
)
".venv\Scripts\python.exe" -X utf8 submission\web_app.py %*
exit /b %errorlevel%

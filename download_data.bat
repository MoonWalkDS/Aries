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
rem Tải dữ liệu từ Hugging Face về data/ (và file máy sinh trong annotations/). Tham số chuyển thẳng cho script:
rem   download_data.bat                       hỏi cách lấy ảnh gốc
rem   download_data.bat --images download     tải cả ảnh gốc
rem   download_data.bat --images skip         bỏ qua ảnh gốc
rem   download_data.bat --images D:\castle\images   dùng thư mục ảnh có sẵn
"%PY%" scripts\hf_data.py download %*
exit /b %errorlevel%

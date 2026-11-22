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
rem Tải dữ liệu từ Hugging Face về data/ (và file máy sinh trong labeling/annotations/). Tham số chuyển thẳng cho script:
rem   tools\download_data.bat                       hỏi cách lấy ảnh gốc
rem   tools\download_data.bat --images download     tải cả ảnh gốc
rem   tools\download_data.bat --images skip         bỏ qua ảnh gốc
rem   tools\download_data.bat --images D:\castle\images   dùng thư mục ảnh có sẵn
"%PY%" tools\huggingface_data.py download %*
exit /b %errorlevel%

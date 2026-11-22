@echo off
chcp 65001 >nul
setlocal
rem File nằm trong tools/: mọi lệnh chạy từ gốc repo.
cd /d "%~dp0.."
rem Cài môi trường: .venv + thư viện Python (torch CUDA nếu có GPU NVIDIA) + viewer + file .env.
where python >nul 2>nul || (echo Không thấy Python. Cài Python 3.12 từ python.org, nhớ tick "Add to PATH". & exit /b 1)
if not exist ".venv\Scripts\python.exe" (
  echo Tạo .venv ...
  python -m venv .venv || exit /b 1
)
set "PY=.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip -q
where nvidia-smi >nul 2>nul && (
  echo Có GPU NVIDIA: cài torch bản CUDA ...
  "%PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -q || exit /b 1
)
echo Cài thư viện Python ...
"%PY%" -m pip install -r requirements.txt -q || exit /b 1
where npm >nul 2>nul && (
  echo Cài viewer ...
  pushd labeling\label_viewer & call npm install --silent & popd
) || echo Không thấy Node.js: bỏ qua viewer. Cài Node 18+ rồi chạy lại tools\setup.bat nếu cần viewer.
if not exist ".env" (
  copy ".env.example" ".env" >nul
  echo Đã tạo .env: điền tài khoản DRES vào đó, hoặc sửa trên trang web truy vấn.
)
echo.
echo Xong. Bước tiếp theo: tools\download_data.bat

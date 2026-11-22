@echo off
chcp 65001 >nul
setlocal
rem File nằm trong tools/: mọi lệnh chạy từ gốc repo.
cd /d "%~dp0.."
rem Mở viewer duyệt nhãn ở http://127.0.0.1:5173.
where node >nul 2>nul || (echo Không thấy Node.js. Cài Node 18+ rồi chạy tools\setup.bat. & exit /b 1)
if not exist "labeling\label_viewer\node_modules" (echo Chưa cài viewer. Chạy tools\setup.bat trước. & exit /b 1)
if not exist "labeling\annotations\enriched" (echo Chưa có labeling\annotations\enriched. Chạy tools\download_data.bat trước. & exit /b 1)
start "" http://127.0.0.1:5173
node labeling\label_viewer\server.js

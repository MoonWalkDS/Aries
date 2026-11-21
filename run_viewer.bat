@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
rem Mở viewer duyệt nhãn ở http://127.0.0.1:5173.
where node >nul 2>nul || (echo Không thấy Node.js. Cài Node 18+ rồi chạy setup.bat. & exit /b 1)
if not exist "viewer\node_modules" (echo Chưa cài viewer. Chạy setup.bat trước. & exit /b 1)
if not exist "annotations\enriched" (echo Chưa có annotations\enriched. Chạy download_data.bat trước. & exit /b 1)
start "" http://127.0.0.1:5173
node viewer\server.js

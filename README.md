# Aries v2

Bộ công cụ làm việc với Castle5000 gồm pipeline gán nhãn, viewer chỉnh nhãn, xuất metadata và CLI nộp bài DRES. Thư mục `retrieval/` dành cho hệ thống truy vấn trong tương lai.

## Chuẩn bị trên Windows

1. Cài Python 3.12 và Node.js 18 trở lên nếu dùng viewer.
2. Chạy `tools\setup.bat` để tạo môi trường, cài thư viện và tạo file `.env`.
3. Nếu cần dữ liệu, đăng nhập bằng `.venv\Scripts\hf.exe auth login` rồi chạy `tools\download_data.bat`. Tài khoản cần quyền đọc dataset riêng tư `DS314/img314`.

## Làm việc với dữ liệu

- `labeling/pipeline/` chứa các bước tạo nhãn và xuất dữ liệu. Chạy `.venv\Scripts\python.exe labeling\pipeline\main.py --help` để xem các lệnh.
- `tools\run_viewer.bat` mở viewer để kiểm tra và chỉnh nhãn. Hướng dẫn nằm trong `labeling/label_viewer/README.md`.
- `labeling/caption/` chứa công cụ tạo mô tả tiếng Việt.
- `.venv\Scripts\python.exe metadata\build_metadata.py` đọc `data/exports/castle_dataset.jsonl`, ghép caption và mô tả của thầy nếu có rồi ghi `metadata/metadata.json`.
- `tools\run_tests.bat` chạy bộ kiểm thử. Chủ dữ liệu dùng `tools\upload_data.bat` để đồng bộ các file lên Hugging Face.

## Nộp bài DRES

Điền cấu hình DRES trong `.env`. Chạy các lệnh sau từ gốc dự án khi cần lấy đề hoặc nộp mã ảnh đã chọn.

```powershell
.venv\Scripts\python.exe submission\fetch_task.py --once
.venv\Scripts\python.exe submission\submit_task.py
```

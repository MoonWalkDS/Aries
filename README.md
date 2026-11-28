# Aries v2

Nền tảng truy vấn ảnh Castle5000 và nộp kết quả lên DRES. Thư mục `retrieval/` dành cho phần cài đặt hệ thống truy vấn.

## Tải dataset trên Windows

Chạy các lệnh sau từ gốc repo với Python 3.12 và tài khoản có quyền đọc dataset riêng tư `DS314/img314`.

```bat
tools\setup.bat
.venv\Scripts\hf.exe auth login
tools\download_data.bat --images download
```

Dùng `--images skip` để chỉ lấy metadata và ảnh thu nhỏ, hoặc truyền đường dẫn thư mục ảnh có sẵn vào `--images`. Các bản ghi ảnh nằm tại `data/exports/castle_dataset.jsonl`. Để ghép caption và mô tả của thầy thành `metadata/metadata.json`, chạy lệnh sau.

```bat
.venv\Scripts\python.exe metadata\build_metadata.py
```

## Nộp DRES

Điền cấu hình DRES trong `.env`. Lệnh đầu lấy đề hiện tại. Lệnh sau mở chế độ tương tác để nộp mã ảnh.

```bat
.venv\Scripts\python.exe submission\fetch_task.py --once
.venv\Scripts\python.exe submission\submit_task.py
```

## Kiểm thử

Cài pytest bằng `.venv\Scripts\python.exe -m pip install pytest`, rồi chạy `tools\run_tests.bat`.

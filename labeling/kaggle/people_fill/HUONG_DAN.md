# Chạy VLM gán nhãn tầng người (Castle5000) trên Kaggle

Notebook cho Qwen2.5-VL-7B gán nhãn hành động cho khoảng 10.568 crop người chưa có nhãn.
Thời gian khoảng 4 giờ GPU. Không cần ảnh gốc, không cần token hay khoá API.

## Tệp trong gói

| Tệp | Nội dung |
|---|---|
| `castle_vlm_fill_people.ipynb` | notebook (đã đặt `STEPS = "fill"`, `PROFILES = "people"`) |
| `labeling_code.zip` | mã nguồn |
| `castle_results.zip` | crop người + trạng thái vòng gán nhãn (khoảng 210 MB) |

## Các bước

1. Kaggle -> **Create -> New Dataset**. Kéo thả `labeling_code.zip` và `castle_results.zip`.
   Đặt **Private**. Bấm Create, đợi tới khi dataset báo xong.
2. Kaggle -> **Create -> New Notebook** -> **File -> Import Notebook** -> chọn `castle_vlm_fill_people.ipynb`.
3. Thanh bên phải của notebook:
   - **Add Input** -> chọn dataset vừa tạo.
   - **Session options -> Accelerator: GPU T4 x2** (không chọn P100: một GPU 16 GB sẽ phải chạy 4-bit, chậm hơn).
   - **Internet: On** (để tải model Qwen, khoảng 16 GB).
   - Tài khoản cần xác minh số điện thoại thì mới bật được GPU và Internet.
4. Bấm **Save Version -> Save & Run All (Commit)**. Có thể tắt máy, Kaggle tự chạy.
5. Khoảng 4 giờ sau, mở notebook -> tab **Output** -> tải `castle_results_vlm.zip` gửi lại.

## Kiểm tra nhanh

- Log ở ô "VLM duyệt nhãn" in `vlm-review: 500 lệnh gọi, 1.3 s/lệnh`, rồi cứ 500 lệnh in một dòng.
- Dòng cuối của bước đó: `pipeline: vlm people: {... 'direct': ..., 'none': ...}`.
- Nếu ô đó báo `LỖI:`, notebook vẫn đóng gói phần đã làm. Cứ tải `castle_results_vlm.zip` gửi lại.

## Lưu ý

- Crop là ảnh người thật trong bộ dữ liệu nghiên cứu: giữ dataset và notebook ở chế độ Private, không chia sẻ lại.
- Lần chạy dùng khoảng 4 trong 30 giờ GPU miễn phí mỗi tuần của tài khoản Kaggle.

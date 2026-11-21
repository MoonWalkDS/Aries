# Caption tiếng Việt cho Castle5000 — kế hoạch và bộ chuẩn bị

Ngày kiểm tra: 2026-11-21. Mục tiêu: một caption tiếng Việt mỗi ảnh, phục vụ truy hồi văn bản.

## Đã làm

Chạy `python -X utf8 labeling/prepare_caption_run.py` từ gốc project để tái tạo bộ chuẩn bị mà không thay input, prompt hay caption đang dùng.

Sản phẩm nằm trong `exports/caption/preparation/` (được gitignore như các export khác):

- `audit.json`: thống kê, kiểm tra đầu vào và SHA256 nguồn.
- `manifest.jsonl`: toàn bộ 5.000 ID và đường dẫn ảnh.
- `inputs.jsonl`: metadata được dựng lại bằng exporter hiện tại; đã đối chiếu khớp input cũ.
- `pilot_100.jsonl`: mẫu cố định seed 314, ưu tiên bao phủ 13 giờ và nhóm số người 0, 1, 2, 3, >=4.
- `batches/001.jsonl` đến `200.jsonl`: 25 ảnh/lô.
- `review_flags.jsonl`: cờ rà soát theo heuristic; không phải kết luận dữ liệu sai.
- `prompt_metadata_vi.md`: prompt đề xuất bổ sung cho đầu vào chỉ có metadata.
- `sample_captions.jsonl`: ba caption do trợ lý viết từ metadata để minh họa; chưa kiểm chứng bằng ảnh, chưa nhập vào kết quả chính.

Kết quả: đủ 5.000 ID duy nhất, tất cả có ảnh và mô tả giáo viên, kết quả chính hiện có 0 caption. Có 1.244 ảnh detector không tìm thấy người; điều này không chứng minh ảnh vắng người. Có 4.624 ảnh mang ít nhất một cờ; các cờ gồm 3.015 lượt mô tả trang phục chứa biến thể, 7.375 lượt hành động chưa chắc, 462 lượt nấu ăn nhưng vị trí ngoài vùng bếp nấu. Cờ vị trí chỉ là gợi ý vì vùng có thể chồng lấn.

## Cách chạy đề xuất

Dùng lại `labeling/caption_gpt.py` nếu có API key và model đã chọn. Backend này gọi API trực tiếp, hỗ trợ tiếp tục và không cần mở agent phụ. Hai runner CLI khác đã có sẵn nhưng chưa được chạy trong phiên này. Hiện không có OPENAI_API_KEY trong môi trường; không đọc key từ .env.

1. Đọc prompt bổ sung và ba caption mẫu. Với các mâu thuẫn, bỏ chi tiết không chắc thay vì gán hành động cho người theo suy đoán.
2. Duyệt 100 ảnh pilot cùng metadata, ưu tiên tình huống không phát hiện người, đông người và các cờ hành động/vị trí. Chốt quy tắc đếm người, độ dài và nguồn ưu tiên.
3. Runner API hiện chỉ hỗ trợ N ảnh đầu còn thiếu, chưa hỗ trợ file pilot tùy chọn. Nếu dùng nguyên trạng, thử 10 ảnh đầu bằng lệnh dưới; chưa gọi đó là pilot phân tầng 100 ảnh. Để chạy đúng pilot cần thêm tùy chọn input/output/prompt hoặc dùng bản runner riêng trỏ vào thư mục preparation, không hoán đổi file đang dùng khi có job khác chạy.
4. Khi đầu ra thử đạt yêu cầu, chạy toàn bộ 5.000 ảnh với tối đa 2 worker ban đầu. Lưu model, prompt hash và metadata hash theo lượt; giữ caption cũ nếu đã duyệt. Không chạy nhiều runner cùng ghi một file.
5. Sau mỗi lượt, đếm ID hợp lệ duy nhất; rà JSON lỗi, caption trống, ID lạ, ID trùng. Rà caption trùng nguyên văn để tìm câu chung chung (ảnh gần nhau có thể thực sự giống nhau). Chạy lại chỉ ảnh thiếu.
6. Chỉ nhập caption vào retrieval sau khi đủ 5.000 ID hợp lệ và đã rà mẫu. So sánh baseline với caption bằng các truy vấn tiếng Việt có ảnh đúng, đo Recall@1/5/10 và MRR. Chưa khẳng định caption giúp tăng điểm khi chưa đo.

Ví dụ PowerShell, sau khi cấu hình key trong môi trường và thay MODEL_DA_CHON bằng tên model thực tế:

```powershell
python -X utf8 labeling/caption_gpt.py --model MODEL_DA_CHON --limit 10 --workers 1
python -X utf8 labeling/caption_gpt.py --model MODEL_DA_CHON --workers 2
```

Lưu ý hai lệnh dùng `exports/caption/prompt_vi.md` hiện có, chưa dùng prompt đề xuất. Không chạy lệnh thứ hai trước khi rà lượt thử. Không cần `--image` cho hướng metadata hiện tại. Nếu rà ảnh thấy detector sai số người, bổ sung kiểm chứng bằng ảnh là một lượt riêng.

## Tiêu chí duyệt

- Người, hành động, trang phục và vị trí không bị nối sai giữa các người.
- Không chuyển nhãn thiếu thành khẳng định phủ định.
- Không dùng tên tag như mô tả ngoại hình; không chọn ngẫu nhiên giữa các trang phục thay đổi theo thời điểm.
- Dịch từ tiếng Anh của thuộc tính sang tiếng Việt; không thêm mục đích hoặc quan hệ nhân quả.
- Mô tả ngắn nhưng đủ phân biệt: thường 2–4 câu, có thể dài hơn khi đông người.
- Nhãn tự động và mô tả giáo viên đều có thể sai; chưa có độ chính xác caption nếu chưa rà ảnh.

## Giới hạn hiện tại

Chưa sinh hàng loạt do chưa có backend API cấu hình. Không có ước tính tiền hay thời gian đáng tin cậy trước khi chọn model và đo lượt thử. Script chuẩn bị không gọi mạng. Bộ export bị gitignore: nếu cần chuyển máy, sao lưu cả thư mục preparation. Code, nhãn và caption đang có không bị ghi đè.

## C?p nh?t: ch?y b?ng Codex CLI

?? th? th?nh c?ng 5/5 caption b?ng t?i kho?n Codex ??ng nh?p s?n, kh?ng c?n API key. ?ang ch?y to?n b? v?i l?nh `python -X utf8 labeling/caption_codex.py --model default --batch 25 --workers 2 --timeout 900`. Sandbox ban ??u kh?ng truy c?p ???c th? m?c t?i kho?n; l?n ch?y ngo?i sandbox ?? ???c c?p quy?n. Prompt g?c sao l?u trong `exports/caption/preparation/prompt_before_codex_run.md`; l??t ch?nh th?m quy t?c x? l? metadata m?u thu?n. C?u h?nh v? hash l?u ? `exports/caption/codex_run.json`. Caption th? ??u d?ng prompt g?c; c?c l? sau d?ng prompt b? sung.

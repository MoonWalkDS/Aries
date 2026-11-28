# Người không phân biệt nguồn: box tự vẽ ngang box máy dò

Bộ dò người bỏ sót nhiều người (người ngồi, bị che, ở rìa ảnh), nên người gán nhãn phải vẽ thêm box tay. Trước đây box tự vẽ là chú thích phụ: không được đếm vào số người, không vào `people.csv` và không vào tìm ảnh. Chúng tôi quyết định mỗi box người, dù do máy dò hay tự vẽ, đều là một **Người** đầy đủ. Mỗi Người mang một trường nguồn (máy dò hoặc tự vẽ) và được đối xử như nhau ở mọi nơi: gán hành động (kể cả tự gõ), đặt tên, số người của ảnh, dataset, chỉ mục và kết quả tìm ảnh.

## Consequences

- Cấu trúc dataset đổi: `num_people` và `people.csv` có thêm Người tự vẽ, kèm cột nguồn. Mọi thứ đọc `exports/` (chỉ mục tìm ảnh, đánh giá) phải dựng lại sau thay đổi này.
- Box tự vẽ cần crop và embedding như box máy dò. Bước gộp dữ liệu tạo chúng, có cache theo box.
- Trang Ảnh chỉ còn vẽ Người. Box vật thể chỉ vẽ ở trang Đồ vật.

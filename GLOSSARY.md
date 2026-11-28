# Aries

Gán nhãn và tìm ảnh cho bộ Castle5000 (bếp nhìn từ camera trần): người gán nhãn người, vùng, đồ vật trên ảnh; hệ thống truy vấn dùng các nhãn đó để tìm ảnh khớp đề DRES.

## Gán nhãn người

**Người**:
Một người xuất hiện trong một ảnh, xác định bằng một box. Mỗi Người có một nguồn (máy dò hoặc tự vẽ) và được đối xử như nhau ở mọi nơi: gán hành động, đặt tên, đếm số người, dataset, tìm ảnh.
_Avoid_: crop (khi nói về người chứ không phải ảnh cắt), box thủ công

**Box máy dò**:
Người do bộ dò người tìm ra khi dựng dữ liệu.
_Avoid_: box detector, box gốc

**Box tự vẽ**:
Người do người gán nhãn vẽ tay, thường vì máy dò bỏ sót hoặc dò lệch.
_Avoid_: box thủ công, manual box

**Xoá box**:
Thao tác loại một Người sai khỏi mọi trang, hàng đợi, số đếm và dataset. Box máy dò chỉ bị ẩn (vẫn còn trong dữ liệu dò); box tự vẽ bị bỏ hẳn. Hoàn tác được.
_Avoid_: không phải người, not_person (tên kỹ thuật, không dùng trên giao diện)

**Sửa box**:
Chỉnh khung của một Người. Với box máy dò, đó là xoá box cũ và thay bằng một box tự vẽ mang theo hành động và tên đã gán.
_Avoid_: kéo box, chỉnh toạ độ

**Hành động**:
Việc một Người đang làm trong ảnh, chọn từ bộ lớp hành động hoặc tự gõ khi không lớp nào khớp.
_Avoid_: nhãn người, action

**Tên người**:
Người này là ai (ví dụ "Linh", "Áo len 2 màu"); nhiều Người ở nhiều ảnh mang cùng một tên. Tên giống nhau vẫn là người khác nhau.
_Avoid_: danh tính, identity, cụm

# Setting cá nhân cho trang tìm ảnh (retrieval/web_app.py)

Ngày: 2026-11-25. Áp dụng cho trang tìm ảnh + nộp DRES `retrieval/index.html`, phục vụ bởi `retrieval/web_app.py`.

## 1. Mục đích

Cả đội thi dùng chung một công cụ. Mỗi thành viên chạy app trên máy của mình, với `.env` DRES riêng; server chỉ nghe
`127.0.0.1`. Mỗi người làm việc một kiểu: có người muốn xem 1 ảnh to, có người muốn xem 8 ảnh nhỏ; có người muốn bám
level gợi ý đầu tiên, có người muốn level mới nhất. Bản này thêm một bộ setting riêng cho từng người, cộng hai tính năng
mà setting cần tới: chọn level đề, và sửa tay kết quả tách đề trước khi tìm.

- **Thành công:** mỗi thành viên chỉnh trang theo ý mình mà không phải sửa code. Setting vẫn còn sau khi tải lại trang.
  Người không chỉnh gì thấy trang chạy y như hiện tại.
- **Không làm:**
  - Chỉnh trọng số hay tham số engine (`W`, `K1`, `K2`, `FLOOR`) trên giao diện.
  - Cho người dùng tự đổi phím tắt.
  - Đồng bộ setting lên server, hoặc tạo nhiều hồ sơ setting trên cùng một trình duyệt.
  - Tự tìm lại khi có gợi ý mới.
- **Hỏng dù chạy được:**
  - Trang đang xem bị thay giữa chừng, mà người dùng không bấm gì.
  - Ảnh bị nộp mà người dùng không chủ động bấm nộp.
  - Một setting hỏng hoặc cũ làm trang trắng, không tìm được.

## 2. Quyết định của người dùng (trong buổi brainstorm)

- Ô "nhập biến" là form **sửa tay kết quả tách đề**, không phải chỉnh trọng số engine.
- Khi DRES nhả gợi ý mới, trang **chỉ báo** "có gợi ý mới". Người dùng tự bấm để tìm lại.
- Danh sách setting ở §4 được duyệt nguyên vẹn.
- Chọn cách B (§3).

## 3. Phương án đã xét

- **A. Viết hết vào `index.html`, lưu localStorage.** Không phải đổi cấu trúc. Nhưng file sẽ tăng từ khoảng 200 lên
  khoảng 650 dòng, CSS, setting, form và lưới ảnh lẫn vào nhau. Mỗi setting mới phải sửa nhiều chỗ.
- **B. Tách thành file tĩnh `index.html` + `settings.js` + `app.js`, lưu localStorage. Đã chọn.** `settings.js` có một
  bảng khai báo duy nhất. Khung setting, xuất/nhập JSON và khôi phục mặc định đều sinh ra từ bảng đó. Thêm setting mới chỉ
  cần thêm một dòng. Không cần bước build, không thêm thư viện.
- **C. Như B, nhưng lưu vào `retrieval/settings.json` qua `/api/settings`.** Đổi trình duyệt vẫn giữ setting. Đổi lại phải
  thêm một API ghi file. Mỗi người vốn chạy trên máy riêng, nên localStorage cộng xuất/nhập JSON là đủ.
- React/Vite bị loại vì repo không có bước build. Thêm vào chỉ cho một trang thì quá nặng.

## 4. Danh sách setting

Giá trị **in đậm** là mặc định. Mặc định phải giữ đúng hành vi hiện tại.

| Nhóm | Khoá | Lựa chọn | Ghi chú |
|---|---|---|---|
| Đề | `level.default` | **`latest`**, `first`, `fixed` | Level dùng khi kéo đề |
| Đề | `level.fixedN` | số nguyên ≥1, **1** | Chỉ dùng khi `fixed`. Nếu đề chưa có đủ N level thì lấy level mới nhất |
| Đề | `level.bar` | **bật**/tắt | Thanh L1…Ln để chọn level ngay trên trang |
| Đề | `poll.seconds` | **0 (tắt)**, 5, 10, 20 | Chu kỳ kiểm tra gợi ý mới |
| Đề | `poll.sound` | bật/**tắt** | Phát âm báo khi có level mới |
| Kết quả | `grid.perPage` | 1, **2**, 3, 4, 6, 8, 12 | Số ảnh trên mỗi trang |
| Kết quả | `grid.columns` | **`auto`**, 1, 2, 3, 4 | `auto` giữ CSS hiện tại: `minmax(min(100%, 420px), 1fr)` |
| Kết quả | `grid.thumbs` | bật/**tắt** | Lưới dùng `/api/image?thumb=1` |
| Kết quả | `search.top` | 100, **300**, 500 | Truyền vào `top` của `/api/search` |
| Kết quả | `grid.showPairs` | **bật**/tắt | Dòng `1:Linh · 2:?` trên ảnh |
| Kết quả | `grid.showScore` | bật/**tắt** | Hiện `score` trên ảnh. Hiện chỉ có trong `title` |
| Tách đề | `parse.showClauses` | **bật**/tắt | Người / nhóm / cảnh / số người |
| Tách đề | `parse.showEnglish` | **bật**/tắt | Dòng `EN:` |
| Tách đề | `parse.showLayers` | **bật**/tắt | Số ảnh còn lại qua 3 lớp |
| Tách đề | `parse.editor` | tắt, **`collapsed`**, `open` | Form sửa biến (§5.2) |
| Chi tiết | `detail.span` | 3, **6**, 10 | Số ảnh lân cận mỗi bên. Truyền vào `/api/detail?span=` |
| Chi tiết | `detail.boxes` | tắt, **`all`**, `matched` | Box người trong dialog |
| Chi tiết | `detail.meta` | **bật**/tắt | Danh sách người, vùng, vật |
| Nộp | `submit.confirm` | **bật**/tắt | Khi tắt thì hiện chữ đỏ cố định "Nộp không hỏi lại" ở header |
| Nộp | `submit.mark` | **bật**/tắt | Đánh dấu ảnh đã nộp (đúng/sai) trên lưới (§5.4) |
| Chung | `ui.theme` | **`system`**, `light`, `dark` | |
| Chung | `ui.fontSize` | `small` (13px), **`medium` (14px)**, `large` (16px) | |

Ngoài bảng còn có ba nút: **Xuất JSON** (tải file `castle-search-settings.json`), **Nhập JSON** và **Khôi phục mặc định**.
Thêm một bảng phím tắt chỉ để tra cứu, không sửa được:

- `←` / `→`: chuyển trang (như hiện tại).
- `1`–`9`: mở ảnh thứ k của trang đang xem.
- `Enter` khi dialog đang mở: nộp ảnh đang mở. Vẫn đi qua `submit.confirm`.
- `Esc`: đóng dialog (trình duyệt tự làm).

Phím tắt không chạy khi đang gõ trong `input`/`textarea`/`select`, giống điều kiện hiện tại.

## 5. Thiết kế

### 5.1 File và việc lưu setting

- `retrieval/index.html`: chỉ giữ HTML và CSS. Nạp `settings.js` rồi `app.js` bằng `<script src>`.
- `retrieval/settings.js`:
  - Mảng `SCHEMA`, mỗi phần tử có dạng `{key, group, label, type: 'select'|'bool'|'int', options?, default}`.
  - `load()`: đọc localStorage khoá `castle-search-settings`. Bọc trong try/catch. JSON hỏng thì dùng mặc định.
  - Áp lên mặc định, nhưng chỉ giữ khoá có trong `SCHEMA` và giá trị hợp lệ theo `options`/`type`. Khoá lạ hoặc giá trị
    sai bị bỏ qua, không làm trang hỏng.
  - `save()`, `get(key)`, `reset()`, `exportJson()`, `importJson(text)`. Nhập file cũng đi qua bước kiểm tra như `load()`.
  - `renderPanel(el, onChange)`: sinh form theo `group`. Đổi giá trị thì lưu ngay và gọi `onChange(key)`.
- `retrieval/app.js`: toàn bộ JS đang có trong `<script>` của `index.html`, đọc setting thay cho hằng số (`PER`,
  `span = 6`, `top`).
- Khung setting nằm trong một `<details>` ở header, cạnh "Cấu hình DRES". Đổi setting thì áp ngay, không cần tải lại trang:
  - Nhóm hiển thị: gọi lại `render()` hoặc `renderInfo()`.
  - `search.top`: chỉ có tác dụng từ lần tìm sau.
- `web_app.py phục vụ thêm `GET /settings.js` và `GET /app.js`, `Content-Type: text/javascript; charset=utf-8`. Dùng danh sách
  tên file cố định, không ghép đường dẫn từ URL.

### 5.2 Sửa tay kết quả tách đề

Form lấy dữ liệu từ `parsed` mà `/api/search` trả về (đầu ra của `parse_query`). Các trường:

- **Số người:** ô số, cộng chọn "đúng k" (`count`) hoặc "ít nhất k" (`count_min`, khi đó `count = null`). Để trống nghĩa
  là không ràng buộc.
- **Tên:** ô text, các tên cách nhau bằng dấu phẩy, gợi ý tên bằng `<datalist>`. Danh sách tên lấy từ endpoint mới
  `GET /api/names`, trả `SEARCH.known_names`. Tên là nhãn riêng: không gộp, không sửa tên gần giống
  (xem memory "identity names are tags").
- **Từng cụm người:** chọn giới tính (không rõ / nam / nữ), ô text mô tả, ô tên (có thể trống), nút xoá. Thêm nút "+ người".
- **Cụm nhóm:** ô text cộng ô số `n`. Xoá được.
- **Cụm cảnh:** danh sách ô text, thêm/xoá được.
- Nút **Tìm lại với bản đã sửa** và nút **Khôi phục bản tự tách** (bỏ bản sửa, tìm lại như thường).

Khi tìm lại, trang gọi `POST /api/search` (JSON, cùng kiểm tra `Content-Type` như `/api/submit`), với body
`{q, top, parsed}`. Trong đó `q` là text của level đang chọn.

`SearchEngine.search(query, top, edits=None)`:

- `edits is None`: chạy như hiện tại.
- Có `edits`: vẫn chạy `parse_query(query)` để lấy `sentences`. Các câu này phục vụ kênh cảnh, ảnh và dense của lớp 1, và
  bám theo text gốc. Sau đó **ghi đè** `names`, `count`, `people`, `group`, `scene` bằng bản đã sửa.
- Tính lại `count_min = len(people) + group.n`. Nếu người dùng nhập "ít nhất k" lớn hơn thì lấy `max`.
- Tên của từng cụm người lấy từ ô tên nếu có. Nếu ô tên trống thì dò bằng `find_names` như `parse_query`.
- Kiểm tra đầu vào ở server: chỉ nhận đúng các khoá trên, text là `str` (cắt tối đa 500 ký tự), số là `int` từ 0 đến 20,
  `gender` thuộc `{None, "male", "female"}`, tên phải có trong `known_names` (tên lạ bị bỏ). Sai kiểu thì trả 400.
- `GET /api/search` giữ nguyên để `evaluate.py` và các test cũ không đổi.

Bản sửa gắn với text của level đang tìm. Đổi level hoặc kéo đề mới thì bỏ bản sửa và tự tách lại.
Khi đang dùng bản sửa, `#info` ghi nhãn "đang dùng bản đã sửa tay".

### 5.3 Level đề và báo gợi ý mới

- Các level là các dòng không rỗng của `query` từ `/api/task`, bỏ các dòng bắt đầu bằng `[IMAGE` / `[VIDEO`. Đây là chỗ
  `hint_to_text` ghi file media. Thứ tự giữ nguyên, L1 là dòng đầu.
- Mỗi level chứa lại toàn bộ level trước, như ví dụ đề của người dùng. Vì vậy `latest` (gửi dòng cuối) cho kết quả giống
  hiện tại: `split_sentences` vốn lấy dòng dài nhất.
- Rủi ro chưa kiểm tra: nếu DRES trả cả chuỗi hint ngay từ đầu, không nhả dần, thì số level không tăng trong một task. Khi
  đó banner chỉ báo task mới. Muốn hơn thế phải đọc mốc thời gian của từng phần tử hint, việc này để sau.
- Kiểm tra định kỳ gọi DRES mỗi 5–20 giây, nên mặc định để tắt. `pull_task` chỉ ghi lại `OUT_JSON`/`OUT_TXT` khi nội dung
  đổi.
- Kéo đề ("Truy vấn lại") thì chọn level theo `level.default`, đưa text của level đó vào ô đề, rồi tìm. Gửi một dòng duy
  nhất, nên `split_sentences` ở server lấy đúng dòng đó.
- Thanh L1…Ln (`level.bar`): bấm một level để đổi ô đề và tìm lại. Level đang dùng được tô đậm.
- Ô đề vẫn sửa tay được như cũ. Gõ tay thì thanh level không tô level nào.
- Kiểm tra gợi ý mới (`poll.seconds > 0`): cứ mỗi chu kỳ gọi `/api/task`. Không gọi khi tab đang ẩn
  (`document.hidden`), và không gọi khi lần trước chưa xong.
  - So với lần kéo gần nhất: cùng `taskId` mà số level tăng, hoặc `taskId` khác.
  - Khi có thay đổi: hiện banner "Có gợi ý mới (L4)" hoặc "Có task mới". Banner có nút **Tìm** (dùng level theo setting)
    và nút **Bỏ qua**. Có thể phát âm báo theo `poll.sound`: tiếng bíp ngắn bằng WebAudio, không cần file âm thanh.
  - Lần kiểm tra này **không** đổi ô đề, kết quả hay trang đang xem.
  - Lỗi mạng khi kiểm tra: ghi nhỏ vào `#task-info` rồi thử lại ở chu kỳ sau. Không hiện banner lỗi.
- Hành vi cũ "sau khi nộp thì `pull(false)`" vẫn giữ. Lần kéo này chỉ cập nhật banner và `#task-info`, không tự tìm lại.
  Như vậy đúng quyết định "chỉ báo, không tự tìm".

### 5.4 Đánh dấu ảnh đã nộp

- Sau mỗi lần nộp, lưu `{id, ok}` vào localStorage khoá `castle-search-submitted`, xếp theo `taskId`. Khi không có `taskId`
  thì xếp vào khoá `"none"`. `ok` dựa trên `raw` có chứa `CORRECT`, như code hiện tại.
- Chỉ giữ 20 task gần nhất.
- Trên lưới: ảnh đã nộp có viền xanh (đúng) hoặc đỏ (sai), kèm nhãn "đã nộp".
- Cả hai khoá localStorage đều bọc try/catch. Lỗi đọc/ghi không làm hỏng trang.

### 5.5 Thay đổi server tóm tắt

- `GET /settings.js`, `GET /app.js`: file tĩnh, tên cố định.
- `GET /api/names`: danh sách tên danh tính.
- `GET /api/detail?id=&span=`: `span` kẹp trong khoảng 1–20, mặc định 6.
- `POST /api/search`: body `{q, top, parsed?}`.
- `SearchEngine.search(..., edits=None)` cộng `query_parser.apply_edits` để kiểm tra bản sửa.

## 6. Kiểm thử

- `retrieval/tests/test_search_engine.py`:
  - `parsed` sửa số người hoặc cụm người làm đổi thứ hạng như mong đợi.
  - `edits=None` cho kết quả y hệt trước.
  - `parsed` sai kiểu bị từ chối.
  - Tên lạ bị bỏ.
- Hàm kiểm tra setting trong `settings.js` viết thuần, không đụng DOM, để test được bằng `node --test`, giống
  `labeling/label_viewer/test/`. Kiểm tra: JSON hỏng, khoá lạ, giá trị ngoài `options` đều quay về mặc định.
- Kiểm tra trên trình duyệt (preview):
  - Đổi từng setting, tải lại trang, setting vẫn còn.
  - Xuất rồi nhập JSON.
  - Banner gợi ý mới không đổi trang đang xem.
  - Tắt `submit.confirm` thì thấy cảnh báo đỏ.

## 7. Để lại cho người làm

- Bố cục và CSS của khung setting, form sửa biến và banner. Cần dùng được ở bề rộng điện thoại.
- Chữ trên giao diện, miễn đúng tiếng Việt.
- Cách chia hàm trong `app.js`.
- Âm báo cụ thể (tần số, độ dài).

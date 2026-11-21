# Castle Labeler: hướng dẫn khởi chạy và sử dụng

Trang web chạy trên máy (localhost) để duyệt và sửa nhãn của 5000 ảnh Castle. Màn hình chỉ có một tấm ảnh.
Các box nằm trên ảnh. Bấm vào box nào thì bảng nhãn của box đó mới hiện ra.

Trang đọc và ghi trực tiếp các file trạng thái của pipeline trong `annotations/`. Đọc mục
[Dữ liệu được ghi ở đâu](#7-dữ-liệu-được-ghi-ở-đâu) trước khi sửa nhãn lần đầu.

---

## 1. Chuẩn bị

**Phần mềm:** Node.js 18 trở lên. Kiểm tra:

```bash
node -v
```

**Dữ liệu cần có** (đường dẫn tính từ gốc repo; `download_data.bat` tải đủ):

| Đường dẫn | Tạo bởi | Bắt buộc |
|---|---|---|
| `annotations/enriched/*.json` | `python pipeline/main.py enrich merge` | Có. Thiếu thì server không chạy. |
| `labels/rois.yaml`, `labels/adl_labels.yaml`, `labels/adl_people_labels.yaml` | có sẵn trong repo | Có |
| `annotations/adl/`, `annotations/adl_people/`, `annotations/adl_roi-*/` | `adl init`, `adl round`, `pipeline` | Không. Profile nào thiếu thì trang bỏ qua profile đó. |
| `data/dataset/images/*.jpg` | ảnh gốc | Có, để hiện ảnh |
| `data/work/enrich/people/crops/`, `data/work/enrich/rois/*/crops/` | `enrich build` | Không. Thiếu thì bảng nhãn không có ảnh cắt. |

Nếu kết quả được chạy trên Kaggle/Colab, giải nén về máy trước bằng
`python pipeline/main.py bundle unpack duong_dan/castle_results.zip`.

## 2. Cài đặt (một lần)

```bash
cd viewer
```

```bash
npm install
```

Lệnh này chỉ cài một thư viện (`yaml`) vào `viewer/node_modules/`.

## 3. Khởi chạy

```bash
npm start
```

Đợi khoảng 10 giây cho tới khi terminal in:

```
Nạp dữ liệu: 8.9s
5000 ảnh, 12 profile nhãn.
Mở http://localhost:5173
```

Mở trình duyệt tại **http://localhost:5173**. Dừng server bằng `Ctrl+C` trong terminal.

**Đổi cổng** khi 5173 đang bận:

PowerShell:

```powershell
$env:PORT=5180; npm start
```

Git Bash:

```bash
PORT=5180 npm start
```

**Chạy trên bản sao để tập dùng, không đụng dữ liệu thật:** chép `labels/rois.yaml`, hai file `labels/adl_*labels.yaml`
và các thư mục `annotations/enriched`, `annotations/adl*` sang một thư mục khác. Tạo liên kết `data` trỏ
về thư mục `data/` của repo, rồi chạy:

```bash
npm start -- D:/ban_sao_castle
```

Server chỉ nghe trên `127.0.0.1`, nên máy khác trong mạng không mở được trang và không ghi được file.

### Lỗi thường gặp

| Thông báo | Cách xử lý |
|---|---|
| `Chưa có ...annotations\enriched. Chạy python pipeline/main.py enrich merge trước.` | Chạy `python pipeline/main.py enrich merge` ở gốc repo. |
| `Error: listen EADDRINUSE ... 5173` | Cổng đang bận, thường do một server khác còn chạy. Tắt nó hoặc đổi cổng như trên. |
| `Cannot find module 'yaml'` | Chưa chạy `npm install` trong `viewer`. |
| Trang trắng, chỉ có chữ "Đang nạp nhãn…" | Server chưa nạp xong hoặc đã dừng. Xem terminal. |

Sau khi sửa code hoặc khi file nhãn bị pipeline Python ghi lại, dừng server rồi `npm start` lại.

---

## 4. Màn hình

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Bộ lọc │ 09_1574  549/5000 │ Người│Vùng│Cả hai │ □ Ẩn box đã duyệt │ Vẽ box B │ Cả ảnh: … │ Hoàn tác │ ? │
├────────────────────────────────────────────────────────────┬─────────────────┤
│                                                            │  Bảng nhãn      │
│                    ẢNH + CÁC BOX                           │  (chỉ hiện khi  │
│                                                            │   chọn một box) │
│                     ‹  3/7 người đã duyệt  ›               │                 │
└────────────────────────────────────────────────────────────┴─────────────────┘
```

- **Bộ lọc** mở ngăn kéo chọn hàng đợi ảnh (mục 6). Số nhỏ bên cạnh là số bộ lọc đang bật.
- **`09_1574  549/5000`** là mã ảnh và vị trí của ảnh trong hàng đợi.
- **Người / Vùng / Cả hai** chọn loại box hiện trên ảnh. Mặc định chỉ hiện box người.
- **Ẩn box đã duyệt** giấu các box bạn đã duyệt, chỉ chừa box còn phải làm.
- **Vẽ box** bật chế độ vẽ box mới (mục 8).
- **Cả ảnh: …** là nhãn hoạt động của cả ảnh. Bấm để sửa.
- **Thanh dưới ảnh:** `‹ ›` sang ảnh trước hoặc sau. Ở giữa là số box đã duyệt trong ảnh.

**Màu box:**

| Box | Nghĩa |
|---|---|
| Viền trắng nét đứt | Chưa có nhãn |
| Viền vàng | Máy gán (VLM, hạt giống hoặc gợi ý), chưa có người duyệt |
| Viền xanh lá | Người đã duyệt |
| Viền xanh dương | Box bạn tự vẽ |
| Viền đen đậm | Box đang chọn |
| Box mờ | Không khớp nhãn đang lọc |

Box của vùng có nét chấm hoặc nét gạch dài để phân biệt với box người. Rê chuột lên box để xem tên nhãn.
Phím `L` bật hoặc tắt tên nhãn trên mọi box.

---

## 5. Duyệt nhãn từng bước

1. Bấm một box, hoặc nhấn `Tab` để chọn box đầu tiên. Bảng bên phải hiện:
   - ảnh cắt của box,
   - nhãn hiện tại, nguồn của nhãn (VLM, hạt giống, gợi ý, người duyệt) và điểm số,
   - ba nút **Đúng**, **Sai**, **Đặt lại**,
   - lưới lớp 2 cột, bấm vào lớp để gán. Có ô tìm khi bộ nhãn dài hơn 20 lớp,
   - ô **Khác** ở cuối để tự gõ nhãn khi không thẻ nào khớp.
2. Chọn một trong các thao tác:

| Thao tác | Phím | Kết quả |
|---|---|---|
| **Đúng** | `A` hoặc `Enter` | Giữ nhãn hiện tại, chuyển nguồn thành người duyệt. Tự chuyển sang box chưa duyệt kế tiếp. |
| **Gán lớp khác** | bấm vào lớp | Đặt lớp đó, nguồn là người duyệt. Tự chuyển sang box kế tiếp. |
| **Sai** | `X` | Ghi nhãn hiện tại vào danh sách bị từ chối và bỏ nhãn. Vòng `adl round` sau sẽ không gợi ý lại nhãn này cho box này. |
| **Đặt lại** | `R` | Xoá quyết định đã lưu, trả về nhãn máy gợi ý (nếu có). |
| **Khác** | `O`, gõ, rồi `Enter` | Lưu mô tả tự gõ làm nhãn (tối đa 120 ký tự). Nhãn máy đang có bị đánh dấu sai. Chọn một lớp sau đó sẽ xoá mô tả tự gõ. |
| **Hoàn tác** | `Z` | Huỷ thao tác gần nhất. Bấm nhiều lần để lùi nhiều bước. |

3. Khi mọi box trong ảnh đã duyệt, trang báo "Ảnh này đã duyệt hết". Nhấn `→` để sang ảnh sau.

**Nhãn cả ảnh:** nhấn `I` hoặc bấm nút "Cả ảnh: …" trên thanh trên. Thao tác giống như với một box.

**Vùng không đổi so với nền:** khi một vùng không thay đổi so với ảnh nền của giờ đó, nhãn của vùng được
dùng chung cho mọi ảnh như vậy trong giờ. Bảng nhãn ghi "nhãn dùng chung cho N ảnh". Trang hỏi lại
trước khi sửa, vì sửa ở một ảnh sẽ đổi cho cả N ảnh.

**Bỏ chọn:** nhấn `Esc` hoặc bấm vào chỗ trống trên ảnh.

---

## 6. Bộ lọc và hàng đợi ảnh

Nhấn `/` hoặc bấm "Bộ lọc". Bộ lọc quyết định những ảnh mà `←` / `→` đi qua.

- **Lối tắt:**
  - *Tất cả ảnh*
  - *Còn nhãn máy chưa duyệt*: ảnh còn ít nhất một nhãn do VLM, hạt giống hoặc gợi ý đặt. Nên dùng cái này khi duyệt.
  - *Người chưa có nhãn*
  - *Đã sửa trong phiên*: ảnh đã sửa từ lúc server khởi động.
  - *Có box tự vẽ*
- **Từ khoá:** gõ không dấu, tìm trên mã ảnh và tên nhãn (tiếng Anh và tiếng Việt).
  - `tu lanh` tìm ảnh có đủ cả hai từ.
  - `tu lanh -mo` loại ảnh có chữ "mở".
  - `"mo tu lanh"` khớp nguyên cụm.
  - `09_` tìm các ảnh của giờ 09.
- **Theo nhãn:** chọn tab Người / Vùng / Cả ảnh rồi bấm các nhãn. Con số bên phải là số mục mang nhãn đó.
  Chọn từ 2 nhãn trở lên thì có thêm lựa chọn *Có bất kỳ* hoặc *Có tất cả*. Ví dụ "mở tủ lạnh" + "thùng có
  rác" + *Có tất cả* là ảnh vừa có người mở tủ lạnh vừa có thùng có rác.
- **Giờ:** chọn một hoặc nhiều giờ.

Bấm **Bắt đầu duyệt** để mở ảnh đầu tiên của hàng đợi. Khi đang lọc theo nhãn, box khớp được chọn sẵn và
các box khác bị làm mờ.

**Tải ids.txt** lưu danh sách mã ảnh của hàng đợi, mỗi dòng một mã. Dùng file này cho lệnh pipeline, ví dụ
`python pipeline/main.py detect --ids-file ids.txt`.

Vị trí ảnh và bộ lọc nằm trong địa chỉ trang. Tải lại trang hoặc gửi link cho người khác (trên cùng máy)
sẽ mở đúng chỗ đó.

---

## 7. Dữ liệu được ghi ở đâu

| Thao tác | File được ghi |
|---|---|
| Đúng, Sai, Đặt lại, gán lớp cho nhãn cả ảnh | `annotations/adl/approved.json`, `rejected.json` |
| … cho người | `annotations/adl_people/approved.json`, `rejected.json` |
| … cho vùng | `annotations/adl_roi-<tên vùng>/approved.json`, `rejected.json` |
| Vẽ, sửa, xoá box tự vẽ | `annotations/manual_boxes.json` |
| Tên người (tab Ai · ngoại hình) | `annotations/identity_names.json` |
| Ô **Khác** (nhãn tự gõ) | `annotations/custom_labels.json` (và đánh dấu sai nhãn máy trong `rejected.json`) |

- Định dạng giống hệt `python pipeline/main.py adl import`: nhãn người duyệt có `"source": "human"`.
- **Sao lưu tự động:** lần ghi đầu tiên vào mỗi file sau khi server khởi động sẽ tạo một bản sao
  `approved.json.bak-<thời điểm>` (tương tự cho `rejected.json` và `manual_boxes.json`). Muốn quay về thì
  dừng server, chép bản `.bak-…` đè lên file gốc, rồi chạy lại server.
- **Hoàn tác chỉ còn trong lần chạy hiện tại.** Tắt server là mất lịch sử hoàn tác, nhưng các nhãn đã lưu vẫn còn.
- **Đừng chạy pipeline cùng lúc.** Nếu `adl round`, `vlm-review` hoặc `bundle unpack` ghi vào cùng file trong lúc
  trang đang mở, server từ chối ghi để không đè mất thay đổi. Trang tự nạp lại dữ liệu và báo bạn thao tác lại.
- **Nhãn tự gõ để riêng** vì vòng lặp `adl` dựng tâm cụm theo tên lớp. `enrich merge` đọc file này và cho nó
  thắng nhãn máy, với `"source": "human", "custom": true`.
- **Sau một buổi duyệt**, bấm **Gộp dữ liệu** trên thanh trên (khoảng 15–20 giây). Nút này chạy
  `python pipeline/main.py enrich merge`: cập nhật `annotations/enriched/` và xuất lại `data/exports/`
  (`castle_dataset.jsonl`, `people.csv`, `rois.csv`, `summary.json`), rồi nạp lại trang. Sau khi gộp, lịch sử
  hoàn tác bị xoá. Server tìm Python ở `.venv` ở gốc repo, nếu không có thì dùng `python` hoặc biến môi trường `PYTHON`.

  Muốn vòng gợi ý kế tiếp học từ nhãn vừa duyệt thì chạy thêm `python pipeline/main.py adl round --profile people`
  (hoặc profile tương ứng).

---

## 7a. Người này là ai, ngoại hình, đồ vật

- **Đồ vật:** tích **Đồ vật** trên thanh trên (phím `V`) để hiện box đồ vật màu xanh ngọc do
  `python pipeline/main.py enrich objects` (RT-DETR: cup, bowl, bottle, knife...) và `enrich objects --owl` (OWLv2: nồi, chảo,
  ấm, thớt, khăn, đĩa, máy nướng bánh...) dò. Chỉ để xem, không sửa được.
- **Tab "Ai · ngoại hình"** trong bảng người (phím `N`; `H` quay về "Hành động"):
  - Cụm ngoại hình của người này (do `python pipeline/main.py enrich identities` gom), độ giống và 12 crop tiêu biểu của cụm.
  - Gõ tên rồi bấm **Cả cụm cXX** (hoặc `Enter`): mọi người trong cụm nhận tên đó. Nhiều cụm cùng tên là cùng một người.
    Không biết tên thì bấm **Số mới** để điền "Người NN" kế tiếp; cụm sau của cùng người chọn lại tên đó trong danh sách gợi ý.
  - **Chỉ người này**: đặt tên riêng, thắng tên của cụm. **Không rõ**: người này không phải người của cụm.
    **Theo cụm**: bỏ tên riêng.
  - Ngoại hình (giới, tuổi, tóc, áo, quần, mũ, kính, đồ cầm) do VLM mô tả, có sau khi chạy bước `attrs` rồi bấm **Gộp dữ liệu**.
- Tên được ghi vào `annotations/identity_names.json`.
- **Bảng định danh** (nút trên thanh trên, hoặc http://localhost:5173/board.html): sửa nhầm người hàng loạt.
  - Cột trái: các tên (kèm số crop), "Chưa có tên", "Không rõ là ai", "+ Người mới", và các cụm theo trang phục
    (vd "nam · sweater · white · sọc"), xếp theo số crop CHƯA GÁN. Crop đã gán tên (hoặc "Không rõ") rời khỏi cụm;
    cụm gán hết thì ẩn (tích "Hiện cả cụm đã gán xong" để xem lại).
  - Cụm do `python pipeline/main.py enrich identities --force` gom theo trang phục VLM mô tả (`enrich.identity.method: outfit`):
    giới + loại áo + màu áo + hoa văn; crop chưa có mô tả (crop nhỏ) xếp vào cụm gần nhất theo SigLIP.
  - Bấm một tên hoặc một cụm để xem MỌI crop của nó, xếp theo độ giống. Trong một cụm, crop đã mang tên khác được viền vàng.
  - Bấm để chọn crop, Shift+bấm để chọn một dải, "Chọn tất cả" / `Esc` bỏ chọn. Kéo các crop đã chọn thả vào một tên bên trái:
    chúng nhận tên đó (tên đặt riêng, thắng tên của cụm). Thả vào "+ Người mới": hộp thoại hỏi tên, điền sẵn "Người NN" kế tiếp, sửa tuỳ ý.
  - Đổi tên sau khi tạo: nút ✎ cạnh tên, hoặc "Đổi tên" khi đang xem một tên. Đổi ở mọi nơi (tên cụm, tên đặt riêng,
    outfits.yaml). Đổi trùng tên đã có = gộp hai người. Bấm "Gộp dữ liệu" để dataset nhận tên mới.
  - Nhấp đúp một crop để mở ảnh gốc trong tab mới. `Z` hoàn tác. "Chỉ crop lớn" ẩn crop thấp hơn 160 px.
- **Gộp dữ liệu** sau khi đặt tên:
  1. Học tên từ mọi crop đã có tên (logistic regression trên embedding SigLIP) rồi đoán tên cho mọi crop
     (`annotations/identity_pred.json`). Thứ tự ưu tiên: tên đặt riêng > mô hình đoán (độ tin >= 60%) > tên của cụm.
  2. Thêm vào `annotations/outfits.yaml` một dòng cho mỗi tên mới, điền sẵn áo, quần, màu bằng câu trả lời VLM gặp nhiều nhất.
     Mở file này điền `stripe_direction`, `top_color_upper`, `top_color_lower`, `description`, rồi bấm **Gộp dữ liệu** lần nữa.
  3. `people.csv` có cột `identity_name`, `identity_source` (person / model / cluster), `identity_prob` và các cột `outfit_*`.
- Tab "Ai · ngoại hình" hiện tên mô hình đoán và trang phục từ `outfits.yaml` (cập nhật sau mỗi lần gộp).
- Gõ tên người, màu áo hay tên đồ vật vào ô Từ khoá của bộ lọc để tìm ảnh, ví dụ `an`, `red`, `cup`.

---

## 8. Vẽ box thủ công

Dùng khi máy bỏ sót một người hoặc khi bạn muốn đánh dấu một vật thể.

1. Nhấn `B` hoặc bấm **Vẽ box**. Trên ảnh hiện dòng "Kéo để vẽ box".
2. Kéo chuột trên ảnh để vẽ hình chữ nhật. Box nhỏ hơn 8px bị bỏ qua.
3. Bảng bên phải mở ra. Chọn loại:
   - **Người:** chọn lớp hành động bằng cách bấm vào lớp.
   - **Vật thể:** gõ tên tự do, ví dụ "chảo", "cốc", "túi ni lông", rồi nhấn `Enter`. Tên đã dùng sẽ được gợi ý.
4. Box được lưu. Nhấn `B` để vẽ box tiếp. Box mới tự lấy loại của lần trước.

**Sửa box tự vẽ:** bấm chọn box để hiện 8 ô vuông ở góc và cạnh.

- Kéo một ô để đổi kích thước.
- Kéo phần thân box để dời chỗ.
- Thả chuột là lưu. `Z` hoàn tác.
- Bấm lại box để đổi loại hoặc đổi tên.
- `Delete` xoá box. `Esc` huỷ box vừa vẽ chưa lưu.

Chỉ box tự vẽ mới sửa được kích thước. Box người và vùng của máy gắn với ảnh cắt và vector của
`enrich build`, nên không sửa ở đây. Muốn chỉnh một box của máy, hãy vẽ một box mới đè lên nó.

Sau `python pipeline/main.py enrich merge`, box tự vẽ nằm trong trường `manual` của `annotations/enriched/<mã ảnh>.json`.
Chúng không được cộng vào số người. `adl round` và VLM chưa dùng các box này.

---

## 9. Phím tắt

Nhấn `?` trên trang để xem bảng này.

| Phím | Việc |
|---|---|
| `←` / `→` | Ảnh trước / sau trong hàng đợi |
| `Tab` / `Shift+Tab` | Box kế tiếp / box trước |
| `A` hoặc `Enter` | Nhãn đúng |
| `X` | Nhãn sai |
| `R` | Đặt lại về nhãn máy |
| `Z` | Hoàn tác |
| `I` | Chọn nhãn cả ảnh |
| `L` | Bật/tắt tên nhãn trên box |
| `B` | Bật/tắt chế độ vẽ box |
| `Delete` | Xoá box tự vẽ đang chọn |
| `/` | Mở bộ lọc |
| `?` | Bảng phím tắt |
| `Esc` | Huỷ box vừa vẽ / tắt chế độ vẽ / bỏ chọn / đóng cửa sổ |

Phím tắt không chạy khi con trỏ đang ở trong ô nhập chữ. Nhấn `Esc` để thoát ô nhập trước.

---

## 9a. Cho nhiều người cùng gán qua internet (Cloudflare Tunnel)

Server vẫn chạy trên máy này; `cloudflared` mở một đường HTTPS từ Cloudflare về `localhost:5173`.

1. Tạo tài khoản (mỗi người một tài khoản; mật khẩu ngẫu nhiên in ra MỘT lần, gửi riêng cho từng người):

```bash
npm run adduser -- ten_cua_ban --admin
```

```bash
npm run adduser -- an
```

   Có `viewer/users.json` thì mọi trang bắt đăng nhập. Đặt lại mật khẩu: `npm run adduser -- an --reset`.
   Xoá tài khoản: `npm run deluser -- an` (cookie cũ hết hiệu lực ngay). **Không gửi file users.json cho ai.**
2. Khởi động lại server (`npm start`). Dòng cuối phải là "Bắt đăng nhập: N tài khoản".
3. Mở tunnel (cửa sổ terminal khác), rồi gửi địa chỉ `https://….trycloudflare.com` nó in ra cho nhóm:

```bash
cloudflared tunnel --url http://localhost:5173
```

   Địa chỉ đổi mỗi lần chạy lại lệnh này. Máy phải bật, không ngủ, có mạng suốt lúc mọi người gán.

**Khi nhiều người cùng gán**

- Mỗi nhãn, tên người, nhãn tự gõ, box tự vẽ ghi thêm `by` (tài khoản) và `at` (thời điểm). Tên người: `identity_names.json` mục `by`.
- `Z` chỉ hoàn tác thao tác của chính mình. Hoàn tác trả mục đó về trạng thái trước thao tác của mình, kể cả khi người
  khác đã sửa mục đó sau.
- Hai người sửa cùng một mục thì người sửa sau thắng. Mỗi người thấy thay đổi của người khác khi tải lại ảnh / trang.
- Chỉ admin thấy và bấm được **Gộp dữ liệu** (chạy Python trên máy này) và Tải lại.
- Đăng nhập sai quá 8 lần trong 15 phút từ một IP thì bị chặn 15 phút. Server in ra mỗi lần đăng nhập thành công.
- Ảnh là ảnh người thật: chỉ gửi địa chỉ và tài khoản cho người trong nhóm nghiên cứu.

---

## 10. Cho người phát triển

```
viewer/
├── server.js        HTTP server (node:http), API JSON, phục vụ ảnh
├── store.js         nạp nhãn, gộp theo quy tắc của enrich merge, lọc, ghi file, hoàn tác
├── public/          index.html, app.js, style.css (không cần build)
└── test/            npm test
```

```bash
npm test
```

API chính: `GET /api/meta`, `GET /api/images?…` (hàng đợi), `GET /api/image/:id`, `POST /api/label`,
`POST /api/manual`, `POST /api/undo`, `POST /api/reload`, `GET /api/export?format=ids|csv|json`.

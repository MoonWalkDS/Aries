# Truy vấn Castle5000 chỉ bằng text: lọc nhiều lớp, chấm theo từng người

Ngày: 2026-11-21. Thay `retrieval/engine.py` (đang so ảnh–text bằng SigLIP2) bằng engine chỉ đọc text:
caption tiếng Việt, mô tả của thầy và nhãn có cấu trúc.

## 1. Mục đích

Đề DRES là câu tiếng Việt, gợi ý tăng dần. Đề hay tả 2–3 người cụ thể, ví dụ "người đàn ông mở tủ lạnh, áo len sọc
ngang… Thầy Linh mặc áo len xanh… người khác rửa chén". Engine phải tìm ảnh có **đủ những người đó cùng lúc**.
Sau đó engine thu hẹp dần số ảnh qua từng lớp. Người nộp bài cần thấy vì sao một ảnh đứng đầu.

- **Thành công:** R@1 và R@10 trên bộ đề dev cao hơn engine cũ (`retrieval/evaluate.py`). Một lần truy vấn mất dưới 2 giây.
- **Không làm:** dùng LLM hoặc VLM lúc truy vấn; dùng model mới. (Ban đầu không dùng embedding ảnh; xem §8.)
- **Hỏng dù chạy được:** một nhãn sai, ví dụ số người, loại hẳn ảnh đúng khỏi danh sách.

## 2. Dữ liệu đầu vào (đã đối chiếu 2026-11-21)

| Nguồn | Mức | Nội dung | Ghi chú |
|---|---|---|---|
| `exports/caption/captions.jsonl` | ảnh | Caption tiếng Việt, khoảng 100 từ | Viết từ nhãn và mô tả của thầy, không nhìn ảnh, nên không phải nguồn độc lập |
| `dataset/raw/teacher_metadata.csv` | ảnh | `description` tiếng Anh (khoảng 58 từ), `class_count` | Nguồn duy nhất không đi qua pipeline của mình |
| `exports/castle_dataset.jsonl` | người | `identity.name` (tag), `outfit.description` (mô tả tiếng Việt), `attributes` (tiếng Anh), `action.label_vi`, `zone_vi`, `view` | 1.314 người chưa có hành động |
| như trên | ảnh | `num_people`, `rois[*].state.label_vi`, `objects[*].label` | |

Số người do detector đếm (`num_people`) lệch với `class_count.person` của thầy ở 918/5.000 ảnh. Lệch 1 người: 771 ảnh.
Lệch từ 2 người trở lên: 147 ảnh. Hai nguồn không nguồn nào luôn đúng.

## 3. Phương án đã xét

- **A. Từ khoá + khớp nhãn.** Nhanh, dễ giải thích. Hỏng khi đề diễn đạt khác chữ ("rửa bát" so với "rửa chén", "áo len" so với "sweater").
- **B. Chỉ nhúng câu bằng SigLIP2 text.** Bắt được cách diễn đạt khác. Nhưng SigLIP được huấn luyện để khớp ảnh với text, nên chất lượng so text với text chưa chắc. Cách này cũng không tách được từng người.
- **C. Lai A+B, lọc nhiều lớp, chấm theo từng người. Đã chọn.** Người dùng muốn lọc theo lớp và chấm từng người, rồi giao quyết định kỹ thuật cho mình. Hai kênh A và B bù lỗi cho nhau. Lớp chấm theo người giải quyết đề nhiều người.

## 4. Kiến trúc

```
Đề tiếng Việt
  └─ Tách đề (luật, không dùng model) ─► số người k, tên, cụm-người[], cụm-cảnh[]
        └─ opus-mt vi→en từng cụm (model đã có)
Lớp 1  điều kiện cả ảnh, chạy trên 5.000 ảnh  ─► giữ top 1.000
Lớp 2  ghép cụm-người với từng người trong ảnh ─► giữ top 200
Lớp 3  xếp hạng cuối bằng caption + mô tả thầy ─► danh sách trả về, kèm giải thích
```

### 4.1 Chỉ mục dựng offline: `retrieval/build_text_index.py` → `exports/text_index/`

Chạy một lần. Chạy lại khi caption hoặc nhãn đổi.

- **Hồ sơ người** (12.819 người), mỗi người có hai bản:
  - Tiếng Việt, đã bỏ dấu: `tag + outfit.description + action.label_vi + zone_vi`.
  - Tiếng Anh: ghép từ `attributes`, ví dụ "male, short gray hair, gray plain sweater, olive brown cargo pants, walking, coffee corner". Các thuộc tính vốn là tiếng Anh nên không cần dịch.
- **Văn bản cảnh của ảnh:**
  - Tiếng Việt: caption cộng trạng thái vùng.
  - Tiếng Anh: mô tả của thầy cộng caption đã dịch.
  - Caption và mô tả của thầy được cắt thành từng câu, vì SigLIP2 chỉ nhận tối đa 64 token. Caption dịch bằng opus-mt một lần, khoảng 25.000 câu.
- **Embedding SigLIP2 text** (fp16, chuẩn hoá L2) cho hồ sơ người tiếng Anh và từng câu cảnh tiếng Anh. Lưu `.npy`, kèm chỉ số câu→ảnh và người→ảnh.
- **Bảng BM25** (tự viết bằng numpy khoảng 30 dòng; môi trường không có sklearn/scipy). Dựng cho văn bản tiếng Việt đã bỏ dấu và tiếng Anh viết thường. Token tiếng Việt gồm từ đơn và cặp từ liền nhau, để "áo len", "tủ lạnh" khớp thành cụm.
- **Số người:** lưu cả `num_people` lẫn `class_count.person`.

### 4.2 Tách đề: `parse_query`, chỉ dùng luật

- Lấy dòng cuối, dài nhất, như `split_sentences` hiện tại. Rồi tách thành câu và thành các vế "; ," có chủ ngữ mới.
- **Cụm-người mới:** câu bắt đầu bằng chủ ngữ người ("một người đàn ông", "người phụ nữ", "hai người", hoặc một tên danh tính).
  - Câu có đại từ ("anh ấy", "cô ấy") hoặc câu tiếp tục tả đồ vật vừa nói ("áo len có…") **gộp vào cụm trước**.
  - Giới tính lấy từ "đàn ông/nam/anh" và "phụ nữ/nữ/cô". Giới tính ghi vào cụm.
- **Cụm-nhóm:** "những người khác / mọi người …". Mỗi người còn lại trong ảnh phải khớp vế này.
- **Cụm-cảnh:** câu không có chủ ngữ người, ví dụ "trên đảo có dứa".
- **Số người k:** số đếm trong đề ("hai/ba… người"). Nếu không có, lấy số cụm-người làm cận dưới.
- **Tên:** dùng lại `names_in`. Tên danh tính là nhãn riêng biệt, không gộp các tên gần giống nhau.

### 4.3 Lớp 1, điều kiện cả ảnh (lọc mềm)

Điểm từng tiêu chí được chuẩn hoá z-score như engine cũ. Lớp này **cộng điểm, không loại hẳn ảnh**. Sau đó giữ top 1.000.

- **Số người:** 1 nếu k bằng một trong hai nguồn; 0,5 nếu lệch ±1; 0 nếu lệch hơn. Khi chỉ có cận dưới thì tính theo dạng "≥ k".
- **Tên:** cộng điểm cho mỗi tên có trong ảnh, như `NAME_BONUS`.
- **Cảnh:** BM25 tiếng Việt, BM25 tiếng Anh và dense (max-sim câu đề × câu cảnh) cho toàn đề.

### 4.4 Lớp 2, ghép cụm-người với người trong ảnh

Điểm của cụm c với người p:

`sim(c,p) = z(dense_en(c,p)) + z(bm25_vi(c,p)) + cổng giới tính + thưởng hành động + thưởng tên`

- **Cổng giới tính:** trừ mạnh khi lệch, 0 khi không biết giới tính.
- **Thưởng hành động:** cộng khi cụm chứa từ khoá của nhãn hành động p. Có bảng đồng nghĩa nhỏ, ví dụ "rửa bát/chén".

Mỗi cụm-người ghép với một người khác nhau, chọn cách ghép có tổng điểm cao nhất. Số cụm thường ≤ 4, số người ≤ 12. Vét cạn hoán vị đủ nhanh trên 1.000 ảnh; nếu chậm thì chuyển sang ghép tham lam. Cụm không có người để ghép, do detector bỏ sót, nhận một điểm sàn cố định chứ không bị loại. Cụm-nhóm nhận điểm trung bình trên những người chưa ghép. Giữ top 200.

### 4.5 Lớp 3, xếp hạng cuối

`final = w1·L1 + w2·L2 + w3·cụm-cảnh(dense+BM25 trên câu caption/thầy)`

Trả về kèm giải thích: mỗi cụm đã ghép với người nào (box, tag) và lớp nào cộng bao nhiêu điểm. Giao diện hiện số ảnh còn lại sau từng lớp.

### 4.6 Tích hợp

- `retrieval/text_engine.py` có cùng giao diện `search(query, top)` và `rank_of`. Trả thêm `layers` (số ảnh sau mỗi lớp) và `explain` cho từng kết quả.
- `app.py` chuyển sang engine mới. `index.html` hiện phần giải thích.
- `evaluate.py` chạy cả hai engine để so. Bản cũ chỉ còn làm mốc so sánh.
- Chỉ nạp opus-mt và SigLIP2 text. Bỏ phần nạp embedding ảnh.

## 5. Ràng buộc

- GPU 6 GB, chỉ dùng model đã có: SigLIP2 `ViT-SO400M-16-SigLIP2-384` và `Helsinki-NLP/opus-mt-vi-en`. Không thêm thư viện.
- Chỉ đọc `dataset/` và `exports/`.

## 6. Quyết định mình tự đưa ra (người dùng giao)

- Lọc mềm thay vì lọc cứng, vì số người lệch giữa hai nguồn ở 18% số ảnh.
- Tách đề bằng luật thay vì LLM, vì không gọi model lúc truy vấn, nhanh và đoán trước được kết quả.
- Top-K 1.000/200 và trọng số w1–w3 chỉnh trên bộ đề dev.
- Bỏ làm mượt theo khung lân cận. Có thể thêm sau nếu đánh giá cho thấy cần.

## 7. Để lại cho người dựng / rủi ro

- **Bộ dev chỉ có 7 đề.** Cần viết thêm khoảng 20 đề trước khi chỉnh trọng số, nếu không sẽ khớp quá sát 7 đề này.
- **Chất lượng SigLIP2 khi so text với text chưa được đo.** Bước đầu tiên là đo riêng kênh dense so với BM25 trên bộ dev. Nếu dense kém, giảm trọng số của nó.
- **Luật tách đề sẽ sai với những câu lạ.** Giao diện cho xem kết quả tách đề để người nộp bài nhận ra.

## 8. Cập nhật 2026-11-21: thêm kênh ảnh

Bản chỉ dùng text yếu hơn engine ảnh cũ trên 7 đề dev. Với đề đủ câu: R@1 2/7, R@10 2/7, R@50 4/7, so với 2/7, 3/7, 6/7 của engine cũ.
Lý do: nhãn quá thô, và cùng một người xuất hiện ở hàng trăm khung. Người dùng duyệt thêm kênh ảnh, dùng embedding SigLIP2
có sẵn trong `exports/embeddings/`, không thêm model:

- **Lớp 1:** `image` = câu đề (cả câu và từng câu) so với ảnh toàn khung và vùng có người, công thức như engine cũ. Trọng số 2.
- **Lớp 2:** `person_img` = cụm-người so với crop của từng người, cộng vào `sim(c,p)`. Trọng số 2. Người không có crop nhận 0.
- **Dịch caption offline:** dùng greedy và cache `exports/text_index/translations.jsonl`.
- **Kết quả với đề đủ câu:** R@1 3/7, R@10 4/7, R@50 5/7. Với đề 1 câu, engine cũ vẫn tốt hơn (R@10 4/7 so với 1/7).

## 9. Sửa lỗi 2026-11-21: đề nêu rõ số người

Đề "Ảnh bếp có 0 người, …" trả về ảnh có người, vì hai lỗi:

- "Ảnh" bỏ dấu thành "anh", nên bị hiểu là chủ ngữ người.
- Luật không bắt được "có 0 người", và số người nằm giữa câu thì không được nhận.

Đã sửa:

- **Tách đề:**
  - Câu mở đầu bằng "ảnh/bức/hình/…" không phải chủ ngữ người.
  - Nhận "có N người" ở bất kỳ vị trí nào trong câu.
  - Nhận số 0 qua các cụm "0 người", "không có ai/người nào", "vắng người".
  - Đề có số người bằng 0 thì bỏ mọi cụm-người.
- **Chấm điểm:** khi đề nêu rõ số người, trọng số là `count_exact = 6`. Khớp số người theo detector hoặc thầy được 1 điểm; lệch 1 chỉ được 0,25.

Đề 0 người: 19/20 ảnh đầu không có người theo cả hai nguồn. Bộ dev với đề đủ câu: R@1 3/7, R@10 3/7, R@50 5/7. Đề 09_255
("Chỉ có một người") tụt từ hạng 6 xuống 13; đề 11_3252 lên hạng 1 ngay từ câu đầu.

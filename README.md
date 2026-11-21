# Castle Retrieval

Tìm ảnh trong bộ Castle5000 (5.000 ảnh camera bếp) theo câu mô tả tiếng Việt và nộp đáp án lên DRES.
Repo gồm công cụ truy vấn cùng pipeline sinh ra dữ liệu cho nó: gán nhãn người, vùng, vật thể; danh tính;
caption tiếng Việt. Dữ liệu nằm trên Hugging Face (dataset riêng tư `DS314/img314`), không nằm trong git.

## Bắt đầu nhanh (Windows)

Cần: Python 3.12 (tick "Add to PATH" khi cài), Node.js 18+ nếu dùng viewer, tài khoản Hugging Face được cấp quyền
đọc `DS314/img314`. GPU NVIDIA không bắt buộc; có thì truy vấn nhanh hơn.

1. Cài môi trường (một lần, vài phút):

   ```powershell
   .\setup.bat
   ```

2. Đăng nhập Hugging Face, dán token tạo ở huggingface.co → Settings → Access Tokens (một lần):

   ```powershell
   .venv\Scripts\hf.exe auth login
   ```

3. Tải dữ liệu. Script hỏi cách lấy ảnh gốc: tải từ HF (khoảng 1,2 GB), dùng thư mục ảnh có sẵn trên máy, hoặc bỏ qua.

   ```powershell
   .\download_data.bat
   ```

4. Mở web truy vấn (trình duyệt tự mở http://127.0.0.1:8314 sau khoảng 1–2 phút nạp model):

   ```powershell
   .\run_app.bat
   ```

Trên trang: điền tài khoản DRES ở mục **Cấu hình DRES** (lưu vào `.env`), dán đề rồi bấm **Tìm**, hoặc bấm
**Truy vấn lại** để lấy đề đang chạy trên DRES. Dòng dưới ô nhập cho thấy engine hiểu đề thế nào (số người, từng
người, cảnh) và số ảnh còn lại sau mỗi lớp. Bấm ảnh để xem lớn; người được ghép với đề có khung xanh. **Nộp** hỏi
xác nhận trước khi gửi.

## Các file `.bat`

| File | Việc |
|---|---|
| `setup.bat` | Tạo `.venv`, cài thư viện (torch CUDA nếu có GPU NVIDIA), cài viewer, tạo `.env` từ `.env.example` |
| `download_data.bat` | Tải `data/` từ HF. Không tham số thì hỏi; hoặc `--images download`, `--images skip`, `--images D:\anh` |
| `run_app.bat` | Web truy vấn + nộp DRES. Tự dựng chỉ mục nếu chưa có. `--no-browser` để không mở trình duyệt |
| `run_viewer.bat` | Viewer duyệt và sửa nhãn ở http://127.0.0.1:5173 (xem `viewer/README.md`) |
| `build_index.bat` | Dựng lại chỉ mục truy vấn sau khi caption hoặc nhãn đổi |
| `run_tests.bat` | Chạy toàn bộ test (pytest + viewer) |
| `upload_data.bat` | Chủ dữ liệu: đẩy `data/` lên HF. Hỏi xác nhận trước khi đẩy |

Mỗi file chỉ gọi một lệnh Python/Node, nên cũng chạy tay được, ví dụ `.venv\Scripts\python.exe retrieval\app.py`.

## Cấu trúc

```
config.yaml        đường dẫn, ngưỡng, model của pipeline (đường dẫn tính từ gốc repo)
pipeline/          gán nhãn: main.py (stage adl, enrich, pipeline, bundle), modules/, notebooks/ cho Kaggle/Colab
caption/           caption tiếng Việt: export.py dựng đầu vào, codex.py gọi Codex CLI theo lô (batch.py)
retrieval/         truy vấn: tách đề, chỉ mục, engine 3 lớp, web, đánh giá
dres/              đăng nhập, kéo đề, nộp bài DRES
viewer/            web duyệt nhãn (Node)
labels/            định nghĩa nhãn (.yaml)
annotations/       nhãn người duyệt (trong git, không tái tạo được) + file máy sinh (tải từ HF)
scripts/hf_data.py tải / đẩy dữ liệu HF
data/              (không vào git) dataset/ ảnh + mô tả của thầy, work/ đầu ra trung gian, exports/ sản phẩm
```

## Truy vấn hoạt động thế nào

Thiết kế chi tiết: `docs/retrieval-design.md`. Tóm tắt:

1. **Tách đề bằng luật** (không gọi LLM): số người, tên danh tính, từng cụm tả người, cụm "những người khác", phần cảnh.
2. **Lớp 1, cả 5.000 ảnh:** số người (khớp detector hoặc mô tả của thầy; đề nêu rõ số người thì gần như bắt buộc),
   tên, BM25 + SigLIP2 trên caption và mô tả của thầy, ảnh SigLIP2. Giữ 1.000 ảnh.
3. **Lớp 2:** ghép một-một từng cụm-người với một người trong ảnh, dựa trên nhãn trang phục, hành động, vùng, giới tính
   và crop ảnh. Giữ 200 ảnh.
4. **Lớp 3:** chi tiết cảnh còn lại xếp hạng cuối.

Đánh giá: `.venv\Scripts\python.exe retrieval\evaluate.py` (7 đề trong `retrieval/dev_queries.json`).

## Chạy lại pipeline từ ảnh gốc

Chỉ cần khi có ảnh mới hoặc muốn làm lại nhãn. Các bước GPU nặng chạy trên Kaggle/Colab; máy 6 GB không đủ cho VLM.
Mọi lệnh chạy ở gốc repo bằng `.venv\Scripts\python.exe`.

| Bước | Lệnh | Ở đâu |
|---|---|---|
| 1. Đóng gói mã nguồn | `pipeline\main.py bundle code` → `data/castle_code.zip` | local |
| 2. Embedding, crop người/vùng, vòng gợi ý, gộp | notebook `pipeline/notebooks/castle_enrich_cloud.ipynb` (chạy `pipeline`, rồi `bundle pack`) | Kaggle/Colab |
| 3. Nhận kết quả | `pipeline\main.py bundle unpack <castle_results.zip>` | local |
| 4. VLM duyệt nhãn vòng đầu | notebook `pipeline/notebooks/castle_vlm_review.ipynb`, rồi `bundle unpack <castle_results_vlm.zip>` | Kaggle/Colab |
| 5. Vật thể, danh tính, gộp hành động | `pipeline\main.py enrich objects`, `enrich identities`, `enrich actions` | local (GPU nhỏ được) |
| 6. Người duyệt sửa nhãn | `run_viewer.bat` | local |
| 7. Gộp thành dataset | `pipeline\main.py enrich merge` → `annotations/enriched/`, `data/exports/castle_dataset.jsonl` | local |
| 8. Caption | `caption\export.py`, rồi `caption\codex.py --model default --limit 25` thử, sau đó bỏ `--limit` | local (cần Codex CLI đã đăng nhập) |
| 9. Chỉ mục truy vấn | `build_index.bat` | local |
| 10. Chia sẻ dữ liệu | `upload_data.bat` | local |

Chi tiết từng stage: `python pipeline\main.py --help` và docstring đầu mỗi module trong `pipeline/modules/`.

## Ghi chú

- Nhãn người duyệt trong `annotations/` (`approved.json`, `rejected.json`, `identity_names.json`, `custom_labels.json`,
  `manual_boxes.json`, `outfits.yaml`) là công sức không tái tạo được: commit sau mỗi đợt duyệt. `download_data.bat`
  không bao giờ ghi đè các file này.
- Tên danh tính là nhãn riêng biệt: hai tên gần giống nhau là hai người khác nhau.
- `.env` chứa mật khẩu DRES và token: không commit.

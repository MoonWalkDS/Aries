# Castle Retrieval

Tìm ảnh trong bộ Castle5000 (5.000 ảnh camera bếp) theo câu mô tả tiếng Việt và nộp đáp án lên DRES.
Repo chia làm ba hệ thống: **gán nhãn** (sinh dữ liệu), **truy vấn** (tìm ảnh), **nộp bài** (DRES), cùng thư mục
`tools/` chứa các file `.bat` để chạy mọi thứ. Dữ liệu nằm trên Hugging Face (dataset riêng tư `DS314/img314`),
không nằm trong git.

## Bắt đầu nhanh (Windows)

Cần: Python 3.12 (tick "Add to PATH" khi cài), Node.js 18+ nếu dùng viewer, tài khoản Hugging Face được cấp quyền
đọc `DS314/img314`. GPU NVIDIA không bắt buộc; có thì truy vấn nhanh hơn. Mọi lệnh dưới đây chạy ở gốc repo
(nhấp đúp file `.bat` trong `tools/` cũng được).

1. Cài môi trường (một lần, vài phút):

   ```powershell
   .\tools\setup.bat
   ```

2. Đăng nhập Hugging Face, dán token tạo ở huggingface.co → Settings → Access Tokens (một lần):

   ```powershell
   .venv\Scripts\hf.exe auth login
   ```

3. Tải dữ liệu. Script hỏi cách lấy ảnh gốc: tải từ HF (khoảng 1,2 GB), dùng thư mục ảnh có sẵn trên máy, hoặc bỏ qua.

   ```powershell
   .\tools\download_data.bat
   ```

4. Mở web truy vấn (trình duyệt tự mở http://127.0.0.1:8314 sau khoảng 1–2 phút nạp model):

   ```powershell
   .\tools\run_app.bat
   ```

Trên trang: điền tài khoản DRES ở mục **Cấu hình DRES** (lưu vào `.env`), dán đề rồi bấm **Tìm**, hoặc bấm
**Truy vấn lại** để lấy đề đang chạy trên DRES. Dòng dưới ô nhập cho thấy engine hiểu đề thế nào (số người, từng
người, cảnh) và số ảnh còn lại sau mỗi lớp. Bấm ảnh để xem lớn; người được ghép với đề có khung xanh. **Nộp** hỏi
xác nhận trước khi gửi.

## Cấu trúc

```
labeling/                HỆ THỐNG GÁN NHÃN: sinh dữ liệu cho truy vấn
  config.yaml              đường dẫn, ngưỡng, model (đường dẫn tính từ labeling/; dữ liệu ở ../data)
  pipeline/                main.py (stage adl, enrich, pipeline, bundle), modules/, notebooks/ cho Kaggle/Colab
  caption/                 caption tiếng Việt: export.py dựng đầu vào, codex.py gọi Codex CLI theo lô (batch.py)
  label_viewer/            web duyệt và sửa nhãn (Node)
  label_definitions/       định nghĩa các lớp nhãn: hành động, vùng cố định (.yaml)
  annotations/             nhãn người duyệt (trong git, không tái tạo được) + file máy sinh (tải từ HF)
  requirements-cloud.txt   thư viện cho Kaggle/Colab
retrieval/               HỆ THỐNG TRUY VẤN
  query_parser.py          tách đề tiếng Việt: số người, tên, cụm-người, cụm-nhóm, cảnh
  text_match.py            bỏ dấu, tách từ, BM25, ghép một-một cụm-người với người
  search_engine.py         engine 3 lớp: layer1, layer2, layer3; model dịch + SigLIP2 (EmbeddingModels)
  build_search_index.py    dựng chỉ mục data/exports/search_index/
  web_app.py, index.html   web truy vấn + nộp DRES
  evaluate.py              đánh giá trên dev_queries.json
  common.py                hằng số, đường dẫn, hàm dùng chung
submission/              HỆ THỐNG NỘP BÀI: fetch_task.py (kéo đề), submit_task.py (đăng nhập, nộp bài DRES)
tools/                   các file .bat + huggingface_data.py (tải / đẩy dữ liệu HF)
data/                    (không vào git) dataset/ ảnh + mô tả của thầy, intermediate/ đầu ra trung gian, exports/ sản phẩm
docs/                    thiết kế
requirements.txt         thư viện máy local (setup.bat cài)
```

## Các file `.bat` (trong `tools/`)

| File | Việc |
|---|---|
| `setup.bat` | Tạo `.venv`, cài thư viện (torch CUDA nếu có GPU NVIDIA), cài viewer, tạo `.env` từ `.env.example` |
| `download_data.bat` | Tải `data/` từ HF. Không tham số thì hỏi; hoặc `--images download`, `--images skip`, `--images D:\anh` |
| `run_app.bat` | Web truy vấn + nộp DRES. Tự dựng chỉ mục nếu chưa có. `--no-browser` để không mở trình duyệt |
| `run_viewer.bat` | Viewer duyệt và sửa nhãn ở http://127.0.0.1:5173 (xem `labeling/label_viewer/README.md`) |
| `build_index.bat` | Dựng lại chỉ mục truy vấn sau khi caption hoặc nhãn đổi |
| `run_tests.bat` | Chạy toàn bộ test (pytest + viewer) |
| `upload_data.bat` | Chủ dữ liệu: đẩy `data/` lên HF. Hỏi xác nhận trước khi đẩy |

Mỗi file tự chuyển về gốc repo rồi gọi một lệnh Python/Node, nên cũng chạy tay được, ví dụ
`.venv\Scripts\python.exe retrieval\web_app.py`.

## Truy vấn hoạt động thế nào

Thiết kế chi tiết: `docs/retrieval-design.md`. Code: `retrieval/search_engine.py`, mỗi lớp là một hàm.

1. **Tách đề bằng luật** (`parse_query`, không gọi LLM): số người, tên danh tính, từng cụm tả người, cụm "những người
   khác", phần cảnh.
2. **Lớp 1** (`layer1`), cả 5.000 ảnh: số người (khớp detector hoặc mô tả của thầy; đề nêu rõ số người thì gần như bắt
   buộc), tên, BM25 + SigLIP2 trên caption và mô tả của thầy, ảnh SigLIP2. Giữ 1.000 ảnh.
3. **Lớp 2** (`layer2`): ghép một-một từng cụm-người với một người trong ảnh, dựa trên nhãn trang phục, hành động,
   vùng, giới tính và crop ảnh. Giữ 200 ảnh.
4. **Lớp 3** (`layer3`): chi tiết cảnh còn lại xếp hạng cuối.

Đánh giá: `.venv\Scripts\python.exe retrieval\evaluate.py` (7 đề trong `retrieval/dev_queries.json`).

## Chạy lại pipeline gán nhãn từ ảnh gốc

Chỉ cần khi có ảnh mới hoặc muốn làm lại nhãn. Các bước GPU nặng chạy trên Kaggle/Colab; máy 6 GB không đủ cho VLM.
Mọi lệnh chạy ở gốc repo bằng `.venv\Scripts\python.exe`.

| Bước | Lệnh | Ở đâu |
|---|---|---|
| 1. Đóng gói mã nguồn | `labeling\pipeline\main.py bundle code` → `data/castle_code.zip` | local |
| 2. Embedding, crop người/vùng, vòng gợi ý, gộp | notebook `labeling/pipeline/notebooks/castle_enrich_cloud.ipynb` (chạy `pipeline`, rồi `bundle pack`) | Kaggle/Colab |
| 3. Nhận kết quả | `labeling\pipeline\main.py bundle unpack <castle_results.zip>` | local |
| 4. VLM duyệt nhãn vòng đầu | notebook `labeling/pipeline/notebooks/castle_vlm_review.ipynb`, rồi `bundle unpack <castle_results_vlm.zip>` | Kaggle/Colab |
| 5. Vật thể, danh tính, gộp hành động | `labeling\pipeline\main.py enrich objects`, `enrich identities`, `enrich actions` | local (GPU nhỏ được) |
| 6. Người duyệt sửa nhãn | `tools\run_viewer.bat` | local |
| 7. Gộp thành dataset | `labeling\pipeline\main.py enrich merge` → `labeling/annotations/enriched/`, `data/exports/castle_dataset.jsonl` | local |
| 8. Caption | `labeling\caption\export.py`, rồi `labeling\caption\codex.py --model default --limit 25` thử, sau đó bỏ `--limit` | local (cần Codex CLI đã đăng nhập) |
| 9. Chỉ mục truy vấn | `tools\build_index.bat` | local |
| 10. Chia sẻ dữ liệu | `tools\upload_data.bat` | local |

Chi tiết từng stage: `python labeling\pipeline\main.py --help` và docstring đầu mỗi module trong `labeling/pipeline/modules/`.

## Ghi chú

- Nhãn người duyệt trong `labeling/annotations/` (`approved.json`, `rejected.json`, `identity_names.json`,
  `custom_labels.json`, `manual_boxes.json`, `outfits.yaml`) là công sức không tái tạo được: commit sau mỗi đợt duyệt.
  `tools\download_data.bat` không bao giờ ghi đè các file này.
- Tên danh tính là nhãn riêng biệt: hai tên gần giống nhau là hai người khác nhau.
- `.env` chứa mật khẩu DRES và token: không commit.

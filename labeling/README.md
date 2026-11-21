# labeling - pipeline gán nhãn Castle5000

Sinh nhãn có cấu trúc cho 5000 ảnh: box + crop từng thực thể (Florence-2), danh tính theo khuôn
mặt (InsightFace), mô tả chi tiết theo schema JSON (VLM), và cầu nối Label Studio để kiểm tra lại.

## Tình trạng kiểm thử

- Máy local: 18 test logic thuần (`tests/`), CLI, chọn ảnh pilot, báo cáo pilot với backend giả.
- Kaggle (2x T4, `torch 2.11`, `transformers 5.20.2`), 5 ảnh, ngày 2026-11-11:
  `detect` → `faces` → `reid` → `annotate` (Qwen2.5-VL-7B, float16) → `merge` chạy hết, không ảnh
  nào lỗi. Kết quả lưu ở `work/debug/smoke_kaggle_qwen7b/`.
  - Florence-2: phải dùng `florence-community/Florence-2-large` (lớp gốc của transformers).
    Bản `microsoft/Florence-2-large` hỏng trên transformers 5.x.
  - InsightFace: `CUDAExecutionProvider` nạp được nhưng cuDNN báo `CUDNN_STATUS_NOT_SUPPORTED`,
    tự rơi về CPU. Vẫn chạy đúng, khoảng 1 giây/ảnh.
  - Qwen 7B: trung bình khoảng 80 giây/ảnh (155 giây cho ảnh 7 người). 5000 ảnh ước chừng
    110 giờ GPU, vượt xa quota Kaggle 30 giờ/tuần.
  - Chất lượng 7B trên ảnh `09_1574`: đúng người mở tủ lạnh và áo sọc của người đó, nhưng gán
    "sọc ngang" sai cho 4 trong 7 người (áo trơn), và câu truy vấn tăng dần không đúng dạng.
- Chưa chạy: `cluster`, `build-gallery`, `reid` với gallery thật, backend API, `pilot` với model
  thật, import vào máy chủ Label Studio thật.
- Backend API chạy từng yêu cầu một; chế độ Batch API (rẻ hơn 50%) chưa được viết.

## Cấu trúc

```
labeling/
├── config.yaml            đường dẫn, ngưỡng, model
├── main.py                orchestrator, mỗi stage một lệnh con
├── modules/
│   ├── detector.py        SceneDetector (Florence-2), crop Layer 2 nới 15%
│   ├── reid.py            BiometricReID (InsightFace), gom cụm, dựng gallery
│   ├── vlm_backends.py    QwenLocalBackend, ApiBackend
│   ├── vlm_annotator.py   VLMStructuredAnnotator: prompt hai lượt, parse/validate JSON
│   ├── pilot.py           chạy thử nhiều model + báo cáo HTML
│   ├── label_studio.py    xuất task/pre-label, nhập bản đã sửa
│   ├── schemas.py         schema Pydantic
│   └── utils.py           config, logging, checkpoint, hình học box
├── tests/
├── labels/                định nghĩa nhãn: adl_labels.yaml, adl_people_labels.yaml, adl_raw_labels.yaml, rois.yaml
├── annotations/           nhãn người duyệt + trạng thái vòng lặp adl*/, enriched/ (key `output_jsons` trong config)
├── work/                  máy sinh, xoá rồi chạy lại được
│   ├── adl/  enrich/      embedding, crop người/vùng
│   ├── clip/              index CLIP cũ (embeddings.npy, index.faiss)
│   ├── output_crops/      <image_id>/<entity_id>.jpg
│   ├── identity_gallery/  <Tên người>/*.jpg
│   ├── logs/              <stage>.log, completed_<stage>.log, failed_<stage>.jsonl
│   └── debug/             roi_check/, smoke_kaggle_qwen7b/
└── kaggle/people_fill/    gói upload Kaggle cho bước VLM điền người

Ảnh gốc ở `../dataset/images/` (5000 ảnh) và `../dataset/thumbs/`. `enrich merge` xuất `../exports/`.
Trong git chỉ có code, `labels/` và phần không tái tạo được của `annotations/` (xem `../.gitignore`).
```

## Cài đặt

Máy local (test, đặt tên cụm, Label Studio, backend API):

```bash
python -m venv --system-site-packages .venv
```

```bash
.venv/Scripts/python -m pip install pydantic pyyaml pillow numpy tqdm scikit-learn python-dotenv anthropic pytest
```

```bash
.venv/Scripts/python -m pytest tests -q
```

Kaggle/Colab (GPU), trong notebook:

```bash
pip install -q -r requirements.txt
```

Trên Kaggle đặt hai biến môi trường để đọc ảnh từ dataset và ghi kết quả vào vùng được lưu:

```python
import os
os.environ["LABELING_INPUT_IMAGES"] = "/kaggle/input/<ten-dataset>/images"
os.environ["LABELING_OUTPUT_ROOT"] = "/kaggle/working/labeling_out"
os.environ["LABELING_IDENTITY_GALLERY"] = "/kaggle/working/labeling_out/identity_gallery"
```

Với Kaggle T4, đặt trong `config.yaml`: `vlm.qwen_local.dtype: float16` (mặc định) và
`quantization: 4bit` nếu thiếu VRAM.

## Chạy

Mọi lệnh chạy từ `labeling/`. Lệnh nào cũng nhận `--limit N`, `--shard i/n`, `--ids-file FILE`,
`--force`.

### 1. Thử trên 20 ảnh

```bash
python main.py detect --limit 20
```

```bash
python main.py faces --limit 20
```

```bash
python main.py reid --limit 20
```

```bash
python main.py annotate --limit 20
```

Xem `work/output_crops/`, `annotations/per_image/` và số giây/ảnh trong `work/logs/annotate.log`.

### 2. Dựng gallery danh tính (một lần)

```bash
python main.py detect
```

```bash
python main.py faces
```

```bash
python main.py cluster
```

Mở các ảnh `annotations/clusters/cluster_XX.jpg`, điền tên vào
`annotations/clusters/cluster_names.yaml` (ví dụ `cluster_03: "Thầy Linh"`), rồi:

```bash
python main.py build-gallery
```

```bash
python main.py reid
```

`reid` in phân bố cosine. Ngưỡng `tau_face = 0.65` theo đề bài là khá chặt với ArcFace; nếu nhiều
mặt đúng người nằm trong khoảng 0.4-0.65, hạ `thresholds.tau_face` rồi chạy lại `reid` (vài giây).
Ảnh tham chiếu tự thêm vào gallery nên đặt tên `manual_*.jpg` để `build-gallery` không xoá.

### 3. Pilot: chọn model trước khi chạy cả 5000

```bash
python main.py pilot --select-only
```

Lệnh trên ghi `annotations/pilot/pilot_ids.txt` (30 ảnh, gồm ảnh đáp án các đề đã nộp đúng).
Chạy detect/faces/reid cho đúng các ảnh đó (nếu chưa chạy toàn bộ):

```bash
python main.py detect --ids-file annotations/pilot/pilot_ids.txt
```

Chạy từng model ở nơi chạy được nó, rồi dựng báo cáo:

```bash
python main.py pilot --models qwen7b
```

```bash
python main.py pilot --models claude-opus-5-5
```

```bash
python main.py pilot --report-only
```

- Backend `api` cần `ANTHROPIC_API_KEY` (đặt trong `.env` ở thư mục gốc dự án). **Tốn phí theo
  token và gửi ảnh ra dịch vụ ngoài.** 30 ảnh pilot là mức để đo; báo cáo ghi token vào/ra trung
  bình mỗi ảnh để tính chi phí cho 5000 ảnh trước khi quyết định.
- Nếu hai model chạy ở hai máy, chép thư mục `annotations/pilot/<tên>/` và `annotations/detections/`
  về cùng một máy rồi chạy `--report-only`.
- Mở `annotations/pilot/pilot_report.html`: mỗi ảnh có crop từng người và mô tả của từng model
  đặt cạnh nhau.

### 4. Chạy cả 5000 ảnh

```bash
python main.py annotate
```

Chia cho nhiều phiên Kaggle (mỗi phiên một shard; chạy lại cùng lệnh để resume sau khi bị ngắt):

```bash
python main.py annotate --shard 1/4
```

Đổi backend không cần sửa config:

```bash
python main.py annotate --backend api
```

Gộp kết quả:

```bash
python main.py merge
```

Kết quả: `annotations/annotations.jsonl` (mỗi dòng một ảnh) và `annotations_master.json`.

### 5. Kiểm tra bằng Label Studio

```bash
python main.py ls-export
```

Tạo `annotations/label_studio_config.xml` và `label_studio_tasks.json`. Trong Label Studio:
tạo project, dán nội dung file XML vào Labeling Interface, bật Local Storage với
`LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true` và
`LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<đường dẫn tới labeling/data>`, rồi import file tasks.
Sau khi sửa xong, export dạng JSON và:

```bash
python main.py ls-import duong_dan/export.json
```

```bash
python main.py merge
```

### 6. Nhãn hoạt động (ADL) cấp ảnh

Theo quy trình bán tự động của LSC-ADL (`../ref/2504.02060v1.pdf`): câu mô tả lấy ảnh hạt giống,
centroid gợi ý nhãn, người duyệt, rồi HDBSCAN gom cụm lại trong từng lớp. Stage này độc lập với các
stage khác. Bộ nhãn và câu mô tả nằm trong `labels/adl_labels.yaml`.

**6a. Xử lý ảnh (chạy một lần, cần GPU cho nhanh).**

```bash
python main.py adl embed
```

Ghi vào `work/adl/`: `embeddings.npy`, `ids.json`, `motion.npy`, `boxes.json`. Mỗi ảnh có hai vector
từ cùng một model (`adl.clip`, mặc định SigLIP2 SO400M 384px): cả khung hình, và vùng có người cắt
ra bằng trừ nền (camera cố định, nền = trung vị theo pixel của từng giờ). Vector cuối là tổng hai
vector. `motion.npy` là tỉ lệ diện tích có chuyển động; ảnh dưới `adl.empty.max_motion` được coi là
bếp trống mà không cần model. Lần đầu sẽ tải model về cache Hugging Face (vài GB).

**6b. Gán nhãn (mục 3.3 của bài báo).** Không cần khoá API.

```bash
python main.py adl init --force      # stage 1: ảnh hạt giống từ câu mô tả + centroid vòng 0
python main.py adl export --what seeds   # (nên làm) trang duyệt cho ảnh hạt giống
python main.py adl import duong_dan/adl_review_seeds_round0.json
```

Sau đó lặp mỗi vòng (bài báo lặp 2 vòng):

```bash
python main.py adl round             # stage 4 + 2 + 3: gom cụm lại, gợi ý, xuất trang duyệt
python main.py adl import duong_dan/adl_review_suggestions_roundN.json
```

`round` gộp ba lệnh `update`, `suggest`, `export`; vẫn gọi riêng từng lệnh được.

Duyệt bằng trang `annotations/adl/review_*.html`: mở thẳng bằng trình duyệt, bấm vào ảnh để đổi
"chưa duyệt → đúng → sai", chọn lớp khác nếu cần, rồi "Tải file kết quả" và đưa file đó cho
`adl import`. Ảnh chưa bấm gì vẫn là chưa duyệt. Nhãn bị từ chối không được gợi ý lại cho ảnh đó.
Ai quen Label Studio thì dùng `label_studio_adl_config.xml` + `label_studio_adl_*.json` cùng thư mục
(cấu hình Local Storage như mục 5); `adl import` đọc được cả hai loại file.

```bash
python main.py adl status            # tiến độ theo lớp
python main.py adl apply             # ghi adl/labels.json + trường activity của per_image
```

`apply` mặc định chỉ lấy nhãn người đã duyệt; thêm `--include-seeds` để lấy cả ảnh hạt giống chưa
duyệt. Các ngưỡng nằm ở mục `adl` của `config.yaml`; đổi model thì chạy lại `embed` rồi `init --force`.

**6c. Chọn bộ nhãn và sinh câu mô tả bằng GPT (mục 3.2 + Decompose-Expand). Tuỳ chọn.** Ba lệnh này
gọi VLM/LLM, mặc định là OpenAI: đặt `OPENAI_API_KEY` trong `../.env`, model ở `vlm.openai.model_id`.
Đổi backend bằng `adl.label_selection.*_backend` hoặc `--backend`. Không có khoá thì dùng luôn
`labels/adl_labels.yaml` soạn tay đang có.

```bash
python main.py adl caption           # 13 giờ x 15 ảnh theo thứ tự thời gian -> caption -> mô tả từng giờ
python main.py adl propose-labels    # nhãn thô + mô tả -> nhãn đề xuất, xếp nhóm -> labels/adl_labels.proposed.yaml
# NGƯỜI DUYỆT: sửa nhóm, gộp nhãn quá chi tiết, bỏ nhãn thừa; chép `classes` sang labels/adl_labels.yaml
python main.py adl describe          # mỗi lớp: từ khoá vật thể + hành động -> 7 câu mô tả
# NGƯỜI DUYỆT: đọc lại câu mô tả trong labels/adl_labels.yaml, rồi chạy lại 6b từ `adl init --force`
```

`caption` gửi khoảng 195 ảnh ra dịch vụ ngoài (nếu dùng openai/api). Kết quả trung gian nằm trong
`annotations/adl/label_selection/`; chạy lại không gọi lại phần đã có. `describe` chỉ sinh cho lớp
còn thiếu câu mô tả (`--force`: sinh lại tất cả) và ghi đè `labels/adl_labels.yaml`, nên sao lưu file này
nếu muốn giữ bản soạn tay. Nhãn thô lấy từ `labels/adl_raw_labels.yaml`.

### 7. Làm giàu theo tầng: người và vùng cố định

Stage `adl` ở mục 6 gán một nhãn cho cả ảnh. Stage `enrich` hạ đơn vị xuống crop, để một ảnh có năm
người làm năm việc thì có năm nhãn, và vật nhỏ được nhìn ở độ phân giải gốc.

**7a. Trích xuất (chạy một lần, cần GPU).**

```bash
python main.py enrich build
```

Ba pha: so từng vùng với nền của giờ đó, dò người bằng RT-DETR, rồi cắt crop và mã hoá bằng model
`adl.clip`. Kết quả lưu từng phần 250 ảnh nên chạy lại sẽ tiếp tục từ chỗ dở. Ghi vào `work/enrich/`:

- `people/`: mỗi người một crop, một vector và một `zone` (đang đứng ở vùng nào, suy từ box).
- `rois/<vùng>/`: crop của vùng mỗi khi nó thay đổi, cộng một ảnh nền `bg-<giờ>` cho mỗi giờ.

Các vùng và trạng thái của chúng khai báo trong `labels/rois.yaml` (toạ độ tỉ lệ 0..1). Sửa file đó để thêm
vùng, ví dụ một hộp nhỏ quanh cái khăn ở cửa lò, rồi chạy lại `enrich build`.

Chỉ sửa khung của vài vùng có sẵn thì không cần làm lại tất cả:

```bash
python main.py enrich build --rois fridge
```

Lệnh này chỉ cắt và mã hoá lại các vùng được nêu, gán lại `zone` cho người theo `labels/rois.yaml` mới, và giữ
nguyên crop người cùng các vùng khác. Vòng lặp cũ của vùng đó (`annotations/adl_roi-fridge/`) trỏ tới
crop cũ: chuyển nó đi chỗ khác rồi chạy lại `adl init` và `adl suggest` cho profile đó.

**7b. Gán nhãn.** Dùng lại nguyên vòng lặp của mục 6b, thêm `--profile`:

```bash
python main.py adl init  --profile people          # hành động của từng người (labels/adl_people_labels.yaml)
python main.py adl round --profile people
python main.py adl import duong_dan/adl_review_....json --profile people

python main.py adl init  --profile roi-stove       # trạng thái của một vùng; tên vùng lấy từ labels/rois.yaml
python main.py adl round --profile roi-stove
```

Mỗi profile có trạng thái và trang duyệt riêng ở `annotations/adl_<profile>/`. Ngưỡng của từng
profile nằm ở mục `adl_profiles` của `config.yaml`.

**7c. Gộp.**

```bash
python main.py enrich merge
```

Ghi `annotations/enriched/<image_id>.json`: danh sách người (box, vùng đang đứng, hành động) và trạng
thái từng vùng. Mỗi nhãn có `source`: `human` (đã duyệt), `seed` (hạt giống chưa duyệt) hoặc
`suggested` (máy gợi ý, chưa duyệt), để bước sau tự chọn mức tin cậy. Vùng không đổi so với nền thừa
hưởng nhãn của ảnh nền giờ đó.

Box tự vẽ bằng trang `viewer/` (mục 10, file `annotations/manual_boxes.json`) nằm ở trường `manual`:
`[{id, kind: person|object, box: [x1, y1, x2, y2], label: {label, label_vi, source: human, score: null} | null}]`.
Chúng không được cộng vào `people` hay `num_people`. `bundle pack` mang file này theo gói để `enrich merge`
trên Kaggle/Colab giữ được box; `bundle unpack` không ghi đè file này nếu máy local đã có.

Xem kết quả bằng mắt:

```bash
python main.py enrich demo --limit 20
```

Ghi `annotations/demo/index.html`: mỗi ảnh có bản đã vẽ box người và khung vùng kèm nhãn, bảng từng
người, bảng từng vùng và bản ghi JSON gốc. Chọn ảnh bằng `--limit` hoặc `--ids-file` (tối đa 60 ảnh).

### 8. Chạy trên Kaggle / Colab

Mọi bước cần GPU của mục 6 và 7 gói trong một lệnh, và kết quả chuyển về máy local bằng một file zip.

**Ở máy local**, trong `labeling/`:

```bash
python main.py bundle code
```

Tạo `labeling_code.zip` (mã nguồn, config, notebook; không kèm ảnh, venv hay kết quả).

**Trên Kaggle hoặc Colab:** mở `notebooks/castle_enrich_cloud.ipynb` và làm theo ô đầu tiên:

- Kaggle: đưa `labeling_code.zip` và thư mục ảnh lên thành hai Dataset, gắn vào notebook, bật GPU và
  Internet.
- Colab: chép `labeling_code.zip` và ảnh vào Google Drive, chọn runtime GPU.

Notebook tự tìm file zip và thư mục ảnh, cài `requirements-cloud.txt`, đặt ba biến môi trường rồi chạy:

```bash
python main.py pipeline
```

| Biến môi trường | Ý nghĩa |
|---|---|
| `LABELING_INPUT_IMAGES` | thư mục ảnh (chỉ đọc) |
| `LABELING_OUTPUT_ROOT` | nơi ghi `output_jsons/`, `logs/`, `exports/` (ở local là `annotations/`, `work/logs/`, `../exports/`) |
| `LABELING_DATA_ROOT` | nơi ghi `adl/`, `enrich/` thay cho `labeling/work` |

`pipeline` chạy `adl embed`, `enrich build`, rồi `init` + `suggest` + `export` cho ảnh, người và từng
vùng, cuối cùng `enrich merge`. Bước đã xong được bỏ qua, nên phiên bị ngắt thì chạy lại là tiếp tục.
Profile đã có nhãn người duyệt không bao giờ bị dựng lại, kể cả với `--force`. Giới hạn bước bằng
`--steps embed,build,loops,merge`; thử nhanh bằng `--limit 40`.

Cuối notebook:

```bash
python main.py bundle pack
```

Tạo `castle_results.zip` (embedding, crop, trạng thái các vòng lặp, bản ghi gộp).

**Về lại máy local:**

```bash
python main.py bundle unpack duong_dan/castle_results.zip
```

Giải nén phần `data/` của gói vào `work/` và phần `output_jsons/` vào `annotations/`, rồi dựng lại các trang `review_*.html` theo đường dẫn máy này.
Từ đây duyệt và chạy `adl round` ở local, không cần GPU.

### 9. VLM duyệt nhãn trước khi tới người

Cho một VLM (mặc định Qwen2.5-VL-7B) làm người duyệt vòng đầu của các vòng lặp ở mục 6b và 7b:

```bash
python main.py adl vlm-review --profile people --rounds 3
```

Với mỗi ảnh hạt giống và gợi ý chưa duyệt, VLM nhìn crop và tự chọn một lớp trong bộ nhãn; nó không
được cho biết nhãn máy đã gợi ý. Kết quả:

| VLM chọn | Xử lý | Ghi trong `approved.json` |
|---|---|---|
| đúng lớp máy gợi ý | nhận nhãn | `source: vlm`, `vlm: agree` |
| lớp khác | nhận lớp của VLM | `source: vlm`, `vlm: changed`, `suggested: <nhãn cũ>` |
| "không lớp nào" | loại, không gợi ý lại | nhãn vào `rejected.json` |
| trả lời không đọc được | giữ nguyên | không đổi |

Sau mỗi vòng, các cụm được gom lại và gợi ý mới được sinh ra. `--rounds N` lặp N vòng hoặc tới khi
hết gợi ý; `--limit N` là trần số lệnh gọi VLM. Mọi câu trả lời nằm trong
`annotations/adl_<profile>/vlm_reviews.jsonl`, nên chạy lại không hỏi lại. Nhãn người đã duyệt
(`source: human`) không bao giờ bị VLM đụng tới.

Chạy cho tầng người và cả 10 vùng, nạp model một lần:

```bash
python main.py pipeline --steps vlm,merge --rounds 3
```

Qwen2.5-VL-7B cần khoảng 17 GB VRAM ở float16; GPU nhỏ hơn 20 GB thì tự chuyển sang 4-bit. Máy 6 GB
không chạy được, nên dùng `notebooks/castle_vlm_review.ipynb` trên Kaggle/Colab: notebook nhận
`castle_results.zip` (tạo bằng `bundle pack`), không cần ảnh gốc, và trả về `castle_results_vlm.zip`
(gói `--light`, chỉ có trạng thái vòng lặp và bản ghi gộp). Ở local:

```bash
python main.py bundle unpack duong_dan/castle_results_vlm.zip
python main.py adl export --profile people --what vlm
```

Chỉ chạy lại VLM cho vài profile, ví dụ một vùng vừa sửa khung:

```bash
python main.py bundle pack castle_results.zip --profiles roi-fridge   # chỉ crop + trạng thái của vùng đó
```

Trong notebook, đặt `PROFILES = "roi-fridge"`. Notebook sẽ truyền `--profiles` cho `pipeline`.

Lệnh `adl export --what vlm` ở trên dựng `review_vlm.html` để người soát lại phần VLM đã quyết, chỗ VLM đổi nhãn xếp trước.
`adl apply --include-vlm` lấy cả nhãn VLM; mặc định `apply` chỉ lấy nhãn người. Trong bản ghi gộp và
trang demo, nhãn VLM mang `source: vlm`.

### 10. Xem và sửa nhãn trên web (`viewer/`)

Trang localhost để duyệt nhãn trên từng ảnh: bấm box để xác nhận, từ chối hoặc đổi lớp; vẽ và chỉnh box mới
cho người hoặc vật thể; lọc hàng đợi ảnh theo nhãn, từ khoá, giờ. Cần Node 18+.

```bash
cd viewer && npm install && npm start
```

Mở http://localhost:5173. Hướng dẫn đầy đủ (khởi chạy, lỗi thường gặp, cách duyệt, bộ lọc, vẽ box, phím tắt,
file nào bị ghi): [`viewer/README.md`](viewer/README.md).

Nhãn sửa trên web ghi vào `annotations/adl*/approved.json` + `rejected.json` (`source: "human"`, như
`adl import`). Box tự vẽ ghi vào `annotations/manual_boxes.json`. Sau khi duyệt, chạy `python main.py enrich merge`.

## Schema kết quả (một ảnh)

```
image_id, file_name, width, height
scene_context            environment, room_label, lighting, camera_perspective
entities[]               entity_id, label, is_person, box, spatial_position,
                         identity, identity_similarity,
                         attributes: gender_presentation, age_group, hair, posture, action,
                                     held_objects, distinguishing_details, uncertain,
                                     attire: upper_garment_type, upper_garment_pattern,
                                             upper_color_top, upper_color_bottom,
                                             lower_garment_type, lower_garment_color,
                                             footwear, accessories
scene_graph_relationships[]   subject, predicate, object
background_clutter       secondary_actions, background_objects
dense_textual_corpus     global_summary, summary_en, progressive_queries, bm25_blob
verified, meta           (backend, model, giây, token, thực thể lỗi)
```

- `box` và `identity` chỉ đến từ detector, Re-ID hoặc người kiểm tra; VLM không được đặt tên hay
  toạ độ.
- `bm25_blob` được dựng lại từ các trường có cấu trúc mỗi khi dữ liệu thay đổi.

## Resume và lỗi

- Mỗi ảnh xong được ghi vào `work/logs/completed_<stage>.log`; chạy lại cùng lệnh sẽ bỏ qua.
- Ảnh lỗi được ghi vào `work/logs/failed_<stage>.jsonl` và **không** bị đánh dấu xong, nên lần chạy
  sau tự thử lại.
- Lỗi backend không khắc phục được (sai khoá, hết quota, model từ chối) dừng cả stage ngay.
- `--force` xoá checkpoint của stage và làm lại từ đầu.

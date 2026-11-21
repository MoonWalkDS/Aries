"""Trang duyệt nhãn ADL tự chứa (một file HTML, mở thẳng bằng trình duyệt, không cần máy chủ).

``adl export`` ghi trang này cạnh file task Label Studio. Người duyệt bấm chấp nhận / từ chối / đổi
lớp cho từng ảnh rồi tải về ``adl_review_*.json``. File đó cùng định dạng với export của Label
Studio nên ``adl import`` đọc được cả hai.

Ảnh được tham chiếu bằng đường dẫn tương đối tới thư mục thumbnail (hoặc ảnh gốc), nên trang chỉ
dùng được trên máy có dữ liệu. Ảnh chưa bấm gì thì không có trong file tải về và vẫn là "chưa duyệt".
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

PAGE = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duyệt nhãn ADL - __TITLE__</title>
<style>
  :root { --bg:#f6f6f4; --fg:#1c1c1a; --muted:#6b6b66; --card:#fff; --line:#d9d9d4; --ok:#1a7f4b; --no:#b3261e; --accent:#2f5fd0; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#161615; --fg:#ecece8; --muted:#a0a09a; --card:#222220; --line:#3a3a36; --ok:#4cc38a; --no:#f2726a; --accent:#8fb0ff; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px 16px; align-items:center; }
  header h1 { font-size:16px; margin:0; }
  .count { color:var(--muted); font-variant-numeric:tabular-nums; }
  button, select { font:inherit; color:var(--fg); background:var(--card); border:1px solid var(--line); border-radius:6px; padding:5px 10px; cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  @media (prefers-color-scheme: dark) { button.primary { color:#10131c; } }
  button:focus-visible, select:focus-visible, .card:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
  main { padding:8px 16px 48px; }
  section h2 { font-size:15px; margin:22px 0 8px; display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; }
  section h2 small { color:var(--muted); font-weight:400; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(var(--w, 300px), 1fr)); gap:10px; }
  .card { background:var(--card); border:3px solid var(--line); border-radius:8px; overflow:hidden; }
  .card img { display:block; width:100%; aspect-ratio:var(--ratio, 16/9); object-fit:contain; cursor:pointer; background:#0003; }
  .card.accept { border-color:var(--ok); }
  .card.reject { border-color:var(--no); }
  .card.reject img { opacity:.45; }
  .meta { display:flex; gap:6px; align-items:center; padding:5px 7px; font-size:12px; color:var(--muted); }
  .meta select { flex:1; min-width:0; padding:2px 4px; font-size:12px; }
  .meta a { color:var(--accent); text-decoration:none; }
  .state { font-weight:600; min-width:5.5em; }
  .accept .state { color:var(--ok); } .reject .state { color:var(--no); }
  .help { color:var(--muted); margin:8px 0 0; }
</style>
</head>
<body>
<header>
  <h1>Duyệt nhãn ADL · __TITLE__</h1>
  <span class="count" id="count"></span>
  <label>Cỡ ảnh <input type="range" id="size" min="200" max="640" step="20" value="300"></label>
  <button id="clear">Xoá mọi quyết định</button>
  <button class="primary" id="download">Tải file kết quả</button>
</header>
<main>
  <p class="help">Bấm vào ảnh để đổi trạng thái: chưa duyệt → đúng → sai. Chọn lớp khác trong ô chọn nếu gợi ý sai nhưng ảnh thuộc lớp khác.
  Ảnh "chưa duyệt" không nằm trong file kết quả. Xong thì bấm "Tải file kết quả" rồi chạy <code>python main.py adl import &lt;file&gt;</code>.</p>
  <div id="sections"></div>
</main>
<script>
const DATA = __DATA__;
const REJECT = DATA.reject, KEY = "adl-review:" + DATA.key;
let decisions = {};
try { decisions = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(decisions)); } catch (e) {} };
const stateOf = id => !decisions[id] ? "pending" : decisions[id] === REJECT ? "reject" : "accept";
const TEXT = { pending: "chưa duyệt", accept: "đúng", reject: "sai" };
const cards = new Map();

function render(id) {
  const { card, state, select, item } = cards.get(id), s = stateOf(id);
  card.className = "card " + s;
  state.textContent = TEXT[s];
  select.value = s === "accept" ? decisions[id] : item.label;
}
function set(id, value) { if (value) decisions[id] = value; else delete decisions[id]; save(); render(id); count(); }
function cycle(id) {
  const s = stateOf(id), item = cards.get(id).item;
  set(id, s === "pending" ? item.label : s === "accept" ? REJECT : null);
}
function count() {
  const all = DATA.items.length, done = DATA.items.filter(i => decisions[i.image_id]).length;
  document.getElementById("count").textContent = done + " / " + all + " đã duyệt";
}

const groups = new Map();
for (const item of DATA.items) { if (!groups.has(item.label)) groups.set(item.label, []); groups.get(item.label).push(item); }
const root = document.getElementById("sections");
for (const [label, items] of groups) {
  const section = document.createElement("section"), h2 = document.createElement("h2");
  h2.append(label + " ");
  const small = document.createElement("small");
  small.textContent = (DATA.names[label] || "") + " · " + items.length + " ảnh";
  const all = document.createElement("button");
  all.textContent = "Đánh dấu đúng các ảnh chưa duyệt";
  all.onclick = () => items.forEach(i => { if (!decisions[i.image_id]) set(i.image_id, i.label); });
  h2.append(small, all);
  const grid = document.createElement("div");
  grid.className = "grid";
  for (const item of items) {
    const card = document.createElement("div"), img = document.createElement("img"), meta = document.createElement("div");
    const state = document.createElement("span"), select = document.createElement("select"), link = document.createElement("a");
    card.tabIndex = 0;
    img.loading = "lazy"; img.src = item.thumb; img.alt = item.image_id;
    img.onclick = () => cycle(item.image_id);
    card.onkeydown = e => { if (e.target === card && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); cycle(item.image_id); } };
    meta.className = "meta"; state.className = "state";
    for (const name of DATA.labels) { const o = document.createElement("option"); o.value = name; o.textContent = name; select.append(o); }
    select.setAttribute("aria-label", "Lớp của " + item.image_id);
    select.onchange = () => set(item.image_id, select.value);
    link.href = item.full; link.target = "_blank"; link.rel = "noopener";
    link.textContent = item.image_id + (item.score == null ? "" : " · " + Number(item.score).toFixed(2));
    link.title = "Mở ảnh gốc";
    meta.append(state, select, link);
    card.append(img, meta);
    grid.append(card);
    cards.set(item.image_id, { card, state, select, item });
    render(item.image_id);
  }
  section.append(h2, grid);
  root.append(section);
}
count();
document.getElementById("size").oninput = e => document.documentElement.style.setProperty("--w", e.target.value + "px");
document.getElementById("clear").onclick = () => { if (confirm("Xoá mọi quyết định trên trang này?")) { decisions = {}; save(); cards.forEach((_, id) => render(id)); count(); } };
document.getElementById("download").onclick = () => {
  const now = new Date().toISOString();
  const tasks = DATA.items.filter(i => decisions[i.image_id]).map(i => ({
    data: { image_id: i.image_id, suggested: i.label, score: i.score },
    annotations: [{ updated_at: now, result: [{ from_name: "activity", to_name: "image", type: "choices", value: { choices: [decisions[i.image_id]] } }] }],
  }));
  if (!tasks.length) { alert("Chưa duyệt ảnh nào."); return; }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(tasks, null, 1)], { type: "application/json" }));
  a.download = "adl_review_" + DATA.key + ".json";
  a.click();
  URL.revokeObjectURL(a.href);
};
</script>
</body>
</html>
"""


def _relative_url(target: Path, start: Path) -> str:
    """Đường dẫn tương đối kiểu URL; khác ổ đĩa (Windows) thì dùng file:// tuyệt đối."""
    try:
        return Path(os.path.relpath(target, start)).as_posix()
    except ValueError:
        return target.resolve().as_uri()


def write_review_page(
    out_path: Path,
    items: Sequence[dict[str, Any]],
    labels: Sequence[tuple[str, str]],
    reject: str,
    key: str,
    images_dir: Path,
    thumbs_dir: Path | None,
    extensions: dict[str, str],
) -> Path:
    """Ghi trang duyệt. ``items``: {image_id, label, score}; ``labels``: (tên, tên tiếng Việt)."""
    payload_items = []
    for item in items:
        name = item["image_id"] + extensions.get(item["image_id"], ".jpg")
        full = images_dir / name
        thumb = thumbs_dir / name if thumbs_dir and (thumbs_dir / name).exists() else full
        payload_items.append({
            "image_id": item["image_id"],
            "label": item["label"],
            "score": item.get("score"),
            "thumb": _relative_url(thumb, out_path.parent),
            "full": _relative_url(full, out_path.parent),
        })
    data = {"key": key, "reject": reject, "labels": [name for name, _ in labels] + [reject], "names": dict(labels), "items": payload_items}
    # "</" trong JSON có thể đóng thẻ <script> sớm; thoát ký tự để dữ liệu không phá trang.
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(PAGE.replace("__TITLE__", key).replace("__DATA__", blob), encoding="utf-8")
    return out_path

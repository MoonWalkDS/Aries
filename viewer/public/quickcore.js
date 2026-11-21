// Phần dùng chung của các trang gán nhanh (quick.html: danh tính, action.html: hành động).
'use strict';
const $ = (id) => document.getElementById(id);
const PAD = 1.2;   // nới khung cắt mỗi phía = 1.2 lần cạnh dài của box, để thấy cả vùng xung quanh

async function api(url, body) {
  const res = await fetch(url, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

// Kết quả đã gán cho người đang xem (ghi trong phiên này): quay lại bằng ← vẫn thấy mình đã chọn gì.
// x.result = {kind: 'ok' | 'bad' | 'set', text, key}; phím đã bấm được viền đậm.
function showResult(x) {
  const el = $('result');
  const r = x?.result;
  el.hidden = !r;
  if (!r) return;
  el.className = `q-result ${r.kind}`;
  el.textContent = `${r.kind === 'ok' ? '✓ Đã nhận' : r.kind === 'bad' ? '✗ Đã bác' : '✎ Đã gán'}: ${r.text}`;
  el.title = 'Bấm phím khác để đổi, Backspace để hoàn tác lần gán gần nhất';
  document.querySelector(`.q-key[data-key="${CSS.escape(r.key)}"]`)?.classList.add('chosen');
}

// Vẽ vùng quanh box người (cắt rộng từ ảnh gốc) lên canvas, viền vàng quanh box. still(): còn đúng người này không.
function drawCrop(c, x, still) {
  const g = c.getContext('2d');
  const img = new Image();
  img.onload = () => {
    if (!still()) return;   // đã chuyển người khác trong lúc tải
    const [x1, y1, x2, y2] = x.box, side = Math.max(x2 - x1, y2 - y1) * PAD;
    const sx = Math.max(0, x1 - side), sy = Math.max(0, y1 - side);
    const sw = Math.min(img.width, x2 + side) - sx, sh = Math.min(img.height, y2 + side) - sy;
    const k = Math.min(c.width / sw, c.height / sh), ox = (c.width - sw * k) / 2, oy = (c.height - sh * k) / 2;
    g.fillStyle = '#111'; g.fillRect(0, 0, c.width, c.height);
    g.drawImage(img, sx, sy, sw, sh, ox, oy, sw * k, sh * k);
    g.strokeStyle = '#ffd400'; g.lineWidth = 3;
    g.strokeRect(ox + (x1 - sx) * k, oy + (y1 - sy) * k, (x2 - x1) * k, (y2 - y1) * k);
  };
  img.src = `/img/${x.image_id}.jpg`;
}

// Xếp để hai người liên tiếp đứng gần nhau trong khung hình (đỡ hoa mắt): bắt đầu từ góc trên trái, mỗi bước
// lấy người chưa xem có tâm box gần nhất. shortcut: tham lam O(n²), đủ nhanh cho vài nghìn crop.
function byPlace(items) {
  const c = items.map((x) => [(x.box[0] + x.box[2]) / 2, (x.box[1] + x.box[3]) / 2]);
  const left = new Set(items.keys());
  let cur = [...left].reduce((a, b) => (c[b][0] + c[b][1] < c[a][0] + c[a][1] ? b : a));
  const out = [];
  while (left.size) {
    left.delete(cur); out.push(items[cur]);
    let best = -1, bd = Infinity;
    for (const j of left) { const d = (c[j][0] - c[cur][0]) ** 2 + (c[j][1] - c[cur][1]) ** 2; if (d < bd) { bd = d; best = j; } }
    cur = best;
  }
  return out;
}

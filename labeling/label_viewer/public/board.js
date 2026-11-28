'use strict';
// Bảng định danh: xem MỌI crop của một cụm hoặc một tên, kéo crop nhầm sang đúng tên (ghi identity_names.json).
const $ = (id) => document.getElementById(id);
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'class') el.className = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) el.append(k.nodeType ? k : String(k));
  return el;
};
async function api(url, body) {
  const res = await fetch(url, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
let toastT;
function toast(msg, err) { const t = $('toast'); t.textContent = msg; t.className = 'toast' + (err ? ' err' : ''); t.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => (t.hidden = true), err ? 6000 : 2200); }

const UNNAMED = { name: '' };
let SUM = null, VIEW = null, ITEMS = [], SEL = new Set(), LAST = -1;
const UNKNOWN = { unknown: true }, HARD = { hard: true };
const query = (v) => (v.hard ? { hard: '1' } : v.unknown ? { unknown: '1' } : v.cluster != null ? { cluster: v.cluster } : { name: v.name });
const fromHash = () => { const p = new URLSearchParams(location.hash.slice(1)); return p.has('hard') ? HARD : p.has('unknown') ? UNKNOWN : p.has('cluster') ? { cluster: p.get('cluster') } : p.has('name') ? { name: p.get('name') } : null; };
const toHash = (v) => history.replaceState(null, '', '#' + new URLSearchParams(query(v)));

function nextNumberName() {
  const used = new Set(SUM?.names_all || []);
  for (let n = 1; ; n++) { const name = `Người ${String(n).padStart(2, '0')}`; if (!used.has(name)) return name; }
}

// ------------------------------------------------------------ hộp thoại nhập tên (tạo người mới, đổi tên)
function askName(title, value, note) {
  const dlg = $('nameDlg'), input = $('nameDlgInput');
  $('nameDlgTitle').textContent = title;
  $('nameDlgNote').textContent = note || '';
  $('nameDlgList').replaceChildren(...(SUM?.names_all || []).map((n) => h('option', { value: n })));
  input.value = value || '';
  dlg.returnValue = '';
  dlg.showModal();
  input.select();
  return new Promise((resolve) => dlg.addEventListener('close', () => resolve(dlg.returnValue === 'ok' && input.value.trim() ? input.value.trim() : null), { once: true }));
}
async function rename(from) {
  const to = await askName(`Đổi tên "${from}"`, from, 'Đổi ở mọi nơi: tên cụm, tên đặt riêng, outfits.yaml. Trùng tên đã có thì gộp hai người.');
  if (!to || to === from) return;
  if ((SUM.names_all || []).includes(to) && !confirm(`"${to}" đã có. Gộp "${from}" vào "${to}"?`)) return;
  try {
    const r = await api('/api/identity/rename', { from, to });
    if (VIEW && VIEW.name === from) { VIEW = { name: to }; toHash(VIEW); $('viewTitle').textContent = to; }
    await refresh();
    toast(`Đã đổi "${from}" thành "${to}" (${r.renamed} mục). Bấm "Gộp dữ liệu" để cập nhật dataset.`);
  } catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------ cột trái: tên (đích thả) + cụm
function dropTarget(el, name) {
  el.addEventListener('dragover', (e) => { e.preventDefault(); el.classList.add('drop'); });
  el.addEventListener('dragleave', () => el.classList.remove('drop'));
  el.addEventListener('drop', (e) => { e.preventDefault(); el.classList.remove('drop'); moveTo(typeof name === 'function' ? name() : name); });
}
function renderSide() {
  const f = $('filter').value.trim().toLowerCase();
  const names = Object.entries(SUM.names).sort((a, b) => a[0].localeCompare(b[0], 'vi', { numeric: true }));
  const nameRows = names.filter(([n]) => !f || n.toLowerCase().includes(f)).map(([n, c]) => {
    const b = h('button', { class: 'b-item' + (VIEW && VIEW.name === n ? ' on' : ''), onclick: () => open({ name: n }), title: `Thả crop vào để gán tên "${n}"` }, h('span', {}, n), h('span', { class: 'n' }, c));
    dropTarget(b, n);
    return h('div', { class: 'b-row' }, b, h('button', { class: 'b-edit', title: `Đổi tên "${n}"`, 'aria-label': `Đổi tên ${n}`, onclick: () => rename(n) }, '✎'));
  });
  const unnamed = h('button', { class: 'b-item' + (VIEW && VIEW.name === '' ? ' on' : ''), onclick: () => open(UNNAMED) }, h('span', {}, 'Chưa có tên'), h('span', { class: 'n' }, SUM.unnamed));
  const unknown = h('button', { class: 'b-item unknown' + (VIEW?.unknown ? ' on' : ''), onclick: () => open(UNKNOWN), title: 'Thả vào: đánh dấu "Không rõ là ai" (không dùng để học)' }, h('span', {}, 'Không rõ là ai'), h('span', { class: 'n' }, SUM.unknown));
  dropTarget(unknown, '');
  const hard = h('button', { class: 'b-item' + (VIEW?.hard ? ' on' : ''), onclick: () => open(HARD), title: 'Số crop chưa ai gán; danh sách hiện 300 crop mô hình phân vân nhất. Sửa xong chạy lại enrich identities rồi duyệt tiếp.' }, h('span', {}, 'Khó trước'), h('span', { class: 'n' }, SUM.hard));
  const fresh = h('button', { class: 'b-item new', title: 'Thả vào (hoặc bấm khi đã chọn crop): tạo người mới, được sửa tên', onclick: () => newPerson() }, h('span', {}, '+ Người mới'), h('span', { class: 'n' }, nextNumberName()));
  fresh.addEventListener('dragover', (e) => { e.preventDefault(); fresh.classList.add('drop'); });
  fresh.addEventListener('dragleave', () => fresh.classList.remove('drop'));
  fresh.addEventListener('drop', (e) => { e.preventDefault(); fresh.classList.remove('drop'); newPerson(); });
  // thả vào: Xoá box (không phải người, box trùng); Del cũng xoá các crop đang chọn
  const del = h('button', { class: 'b-item unknown', title: 'Thả vào (hoặc bấm Del khi đã chọn crop): xoá box, bỏ khỏi mọi nơi. Z để hoàn tác.', onclick: () => deleteSel() }, h('span', {}, 'Xoá box'), h('span', { class: 'n' }, SUM.notPerson ?? 0));
  del.addEventListener('dragover', (e) => { e.preventDefault(); del.classList.add('drop'); });
  del.addEventListener('dragleave', () => del.classList.remove('drop'));
  del.addEventListener('drop', (e) => { e.preventDefault(); del.classList.remove('drop'); deleteSel(); });
  $('names').replaceChildren(hard, ...nameRows, unnamed, unknown, fresh, del);
  // cụm chỉ còn crop chưa gán; cụm đã gán hết bị ẩn (bật "hiện cả cụm đã xong" để xem)
  const showDone = $('showDone').checked;
  const rows = SUM.clusters.filter((c) => (showDone || c.left > 0) && (!f || c.cluster.includes(f) || `${c.label || ''} ${c.name || ''}`.toLowerCase().includes(f)))
    .sort((a, b) => b.left - a.left);
  $('clusters').replaceChildren(...rows.map((c) =>
    h('button', { class: 'b-item' + (VIEW && VIEW.cluster === c.cluster ? ' on' : '') + (c.left ? '' : ' done'), onclick: () => open({ cluster: c.cluster }), title: `${c.cluster}: còn ${c.left} / ${c.size} crop chưa gán` },
      h('span', {}, c.label || c.cluster, c.name ? h('span', { class: 'muted' }, ` · ${c.name}`) : ''), h('span', { class: 'n' }, c.left))),
    rows.length ? '' : h('div', { class: 'muted small' }, 'Không còn cụm nào có crop chưa gán.'));
}

// ------------------------------------------------------------ lưới crop
function renderGrid() {
  const big = $('bigOnly').checked;
  // chỉ ẩn trong một cụm: ở danh sách theo tên, crop máy đoán chính là nội dung cần xem
  const hideModel = $('hideModel').checked && VIEW.cluster != null;
  const items = ITEMS.filter((x) => (!big || (x.h ?? 0) >= 160) && !(hideModel && x.name && x.source === 'model'));
  $('grid').replaceChildren(...(items.length ? items.map((x, i) => {
    // trong một cụm: crop đã mang tên khác tên của cụm (đặt riêng hoặc mô hình đoán) được viền vàng
    // trong một cụm: crop mô hình đã đoán tên (chưa ai gán) được viền vàng
    const other = (VIEW.cluster != null || VIEW.hard) && x.name && x.source === 'model';
    const card = h('div', { class: 'b-card' + (SEL.has(x.id) ? ' sel' : '') + (other ? ' other' : ''), draggable: 'true', 'data-id': x.id, title: `${x.id} · cụm ${x.cluster ?? '—'} · ${x.name ?? 'chưa có tên'}${x.source ? ` (${x.source})` : ''}` },
      h('img', { src: `/crop/person/${x.id}.jpg`, alt: x.id, loading: 'lazy', onerror: (e) => cropFromImage(e.target, x) }),
      x.name && (VIEW.cluster != null || VIEW.hard) ? h('span', { class: 'tag' }, x.name) : VIEW.hard && x.guess ? h('span', { class: 'tag' }, `${x.guess}?`) : '',
      h('div', { class: 'cap' }, h('span', {}, x.image_id), h('span', {}, VIEW.hard && x.margin != null ? `±${Math.round(x.margin * 100)}` : x.sim != null ? `${Math.round(x.sim * 100)}%` : '')));
    card.addEventListener('click', (e) => select(items, i, e.shiftKey));
    card.addEventListener('dblclick', () => window.open(`/#id=${x.image_id}`, '_blank'));
    card.addEventListener('dragstart', (e) => { if (!SEL.has(x.id)) { SEL = new Set([x.id]); renderSel(); } e.dataTransfer.setData('text/plain', [...SEL].join(',')); e.dataTransfer.effectAllowed = 'move'; });
    return card;
  }) : [h('div', { class: 'b-empty' }, 'Không có crop nào.')]));
  $('viewInfo').textContent = `${items.length} crop${big ? ' lớn' : ''}`;
  renderSel();
}
function select(items, i, range) {
  const id = items[i].id;
  if (range && LAST >= 0) { const [a, b] = [Math.min(LAST, i), Math.max(LAST, i)]; for (let k = a; k <= b; k++) SEL.add(items[k].id); }
  else if (SEL.has(id)) SEL.delete(id); else SEL.add(id);
  LAST = i;
  renderSel();
}
function renderSel() {
  for (const c of $('grid').children) if (c.dataset.id) c.classList.toggle('sel', SEL.has(c.dataset.id));
  $('selCount').hidden = !SEL.size; $('selCount').textContent = `${SEL.size} đã chọn`;
}

// ------------------------------------------------------------ hành động
async function newPerson() {
  if (!SEL.size) return toast('Chọn crop trước, rồi thả vào "+ Người mới".', true);
  const name = await askName(`Người mới cho ${SEL.size} crop`, nextNumberName(), 'Sửa tên tuỳ ý. Có thể đổi lại sau bằng nút ✎.');
  if (!name) return;
  if ((SUM.names_all || []).includes(name) && !confirm(`"${name}" đã có. Gán ${SEL.size} crop cho người này?`)) return;
  await moveTo(name);
}
async function moveTo(name) {
  if (!SEL.size) return toast('Chưa chọn crop nào.', true);
  const n = SEL.size;
  try {
    await api('/api/identity', { scope: 'person', persons: [...SEL], name });
    SEL = new Set(); LAST = -1;
    await refresh();
    toast(name ? `Đã chuyển ${n} crop sang "${name}". Z để hoàn tác.` : `Đã đánh dấu ${n} crop "Không rõ là ai".`);
  } catch (e) { toast(e.message, true); }
}
// box tự vẽ chưa gộp chưa có file crop: cắt từ ảnh gốc ngay trên trình duyệt
function cropFromImage(img, x) {
  if (img.dataset.fallback || !x.box) return;
  img.dataset.fallback = '1';
  const full = new Image();
  full.onload = () => {
    const [x1, y1, x2, y2] = x.box, c = document.createElement('canvas');
    c.width = Math.max(1, Math.round(x2 - x1)); c.height = Math.max(1, Math.round(y2 - y1));
    c.getContext('2d').drawImage(full, x1, y1, x2 - x1, y2 - y1, 0, 0, c.width, c.height);
    img.src = c.toDataURL('image/jpeg', 0.85);
  };
  full.src = `/img/${x.image_id}.jpg`;
}
async function deleteSel() {
  if (!SEL.size) return toast('Chọn crop trước, rồi bấm Del hoặc thả vào "Xoá box".', true);
  const n = SEL.size;
  try {
    await api('/api/person/delete', { ids: [...SEL] });
    SEL = new Set(); LAST = -1;
    await refresh();
    toast(`Đã xoá ${n} box. Z để hoàn tác.`);
  } catch (e) { toast(e.message, true); }
}
async function open(v) {
  VIEW = v; SEL = new Set(); LAST = -1; toHash(v);
  const cl = v.cluster != null ? SUM.clusters.find((c) => c.cluster === v.cluster) : null;
  $('viewTitle').textContent = v.hard ? 'Khó trước (mô hình phân vân nhất)' : v.unknown ? 'Không rõ là ai' : cl ? `${cl.label || 'Cụm'} (${v.cluster})` : v.name ? v.name : 'Chưa có tên';
  $('renameBtn').hidden = !(v.cluster == null && !v.unknown && v.name);
  await loadItems(); renderSide();
}
async function loadItems() {
  const r = await api(`/api/identity/members?${new URLSearchParams(query(VIEW))}`);
  ITEMS = r.items; renderGrid();
}
async function refresh() { SUM = await api('/api/identity/summary'); renderSide(); if (VIEW) await loadItems(); }
async function undo() {
  try {
    const r = await api('/api/undo', {});
    toast(`${r.desc ? `Đã hoàn tác: ${r.desc}${r.page ? ` (trang ${r.page})` : ''}.` : 'Đã hoàn tác.'}${r.conflicts?.length ? ` Giữ nguyên ${r.conflicts.join(', ')}: người khác đã sửa sau.` : ''}`, !!r.conflicts?.length);
    await refresh();
  } catch (e) { toast(e.message, true); }
}

(async function init() {
  try { SUM = await api('/api/identity/summary'); } catch (e) { return toast(e.message, true); }
  $('filter').oninput = renderSide;
  $('showDone').onchange = renderSide;
  $('bigOnly').onchange = renderGrid;
  $('hideModel').onchange = renderGrid;
  $('selAll').onclick = () => { for (const c of $('grid').children) if (c.dataset.id) SEL.add(c.dataset.id); renderSel(); };
  $('selNone').onclick = () => { SEL = new Set(); renderSel(); };
  $('undo').onclick = undo;
  $('renameBtn').onclick = () => VIEW?.name && rename(VIEW.name);
  document.addEventListener('keydown', (e) => { if (/^(INPUT|TEXTAREA)$/.test(e.target.tagName) || e.ctrlKey || e.metaKey) return; if (e.key.toLowerCase() === 'z') undo(); if (e.key === 'Delete') deleteSel(); if (e.key === 'Escape') { SEL = new Set(); renderSel(); } });
  const v = fromHash();
  if (v) await open(v); else renderSide();
})();

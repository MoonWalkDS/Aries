// Gán đồ vật theo vùng. Hai chế độ: "Sửa vùng" (vẽ đa giác + chiều cao, dùng chung mọi ảnh) và
// "Gán đồ vật" (tick đồ đang có trong vùng; nhãn kế thừa sang các ảnh sau cùng giờ).
'use strict';
const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';

async function api(url, body) {
  const res = await fetch(url, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

const S = {
  mode: 'label', meta: null, img: null, region: null,   // region: tên vùng đang chọn
  selBox: -1,          // box tự vẽ đang chọn (chỉ số trong boxes)
  tool: null,          // null | 'draw' (box đồ vật) | 'place' (chấm điểm vùng) | 'height' | 'drag'
  draft: null,         // box/điểm đang vẽ
  pendingBox: null,    // box vừa vẽ, chờ đặt tên
  hl: null,            // tên đồ đang rê chuột trong danh sách
};
const W = () => S.meta.size[0], H = () => S.meta.size[1];

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  parent?.append(e);
  return e;
}
const pts = (p) => p.map(([x, y]) => `${x},${y}`).join(' ');
const up = (p, h) => p.map(([x, y]) => [x, y - h]);

function say(text, err = false, id = S.mode === 'edit' ? 'editMsg' : 'msg') {
  const m = $(id);
  m.textContent = text;
  m.className = `o-msg${err ? ' err' : ''}`;
}
async function guard(fn) {
  try { return await fn(); } catch (e) { say(e.message, true); }
}

// ---------- dữ liệu
async function loadMeta() {
  S.meta = await api('/api/objects/meta');
  if (!S.meta.regions.some((r) => r.name === S.region)) S.region = S.meta.regions[0]?.name ?? null;
  $('vocabList').replaceChildren(...S.meta.vocab.map((v) => Object.assign(document.createElement('option'), { value: v.name })));
}
async function go(id) {
  if (!id) return say('Hết ảnh theo hướng này.');
  S.img = await api(`/api/objects/image/${encodeURIComponent(id)}`);
  S.selBox = -1; S.pendingBox = null; $('newBox').hidden = true;
  $('photo').setAttribute('href', `/img/${S.img.image_id}.jpg`);
  $('imgId').value = S.img.image_id;
  $('pos').textContent = `${S.img.pos + 1}/${S.meta.images}`;
  const q = new URLSearchParams({ id: S.img.image_id, mode: S.mode, ...(S.region ? { region: S.region } : {}) });
  history.replaceState(null, '', `?${q}`);
  render();
}
const cur = () => S.img?.regions.find((r) => r.name === S.region);
const regionDef = () => S.meta.regions.find((r) => r.name === S.region);
// Vùng chưa gán mặc định bật tất cả đồ vật; nhãn đã lưu (kể cả rỗng) được giữ nguyên.
const selectedItems = (c) => c.label?.items ?? S.meta.vocab.map((v) => v.name);

// ---------- vẽ
function viewBox() {
  const r = regionDef();
  if (S.mode !== 'label' || !$('zoom').checked || !r) return [0, 0, W(), H()];
  const all = [...r.points, ...up(r.points, r.height)];
  let x1 = Math.min(...all.map((p) => p[0])), x2 = Math.max(...all.map((p) => p[0]));
  let y1 = Math.min(...all.map((p) => p[1])), y2 = Math.max(...all.map((p) => p[1]));
  const pad = Math.max(x2 - x1, y2 - y1) * 0.35 + 30;
  x1 = Math.max(0, x1 - pad); y1 = Math.max(0, y1 - pad); x2 = Math.min(W(), x2 + pad); y2 = Math.min(H(), y2 + pad);
  return [x1, y1, x2 - x1, y2 - y1];
}

function drawRegion(g, r, sel) {
  const grp = el('g', { class: `rg ${sel ? 'sel' : 'dim'}` }, g);
  el('polygon', { class: 'rg-base', points: pts(r.points) }, grp);
  if (r.height > 0) {
    const top = up(r.points, r.height);
    for (let i = 0; i < r.points.length; i++) el('line', { class: 'rg-wall', x1: r.points[i][0], y1: r.points[i][1], x2: top[i][0], y2: top[i][1] }, grp);
    el('polygon', { class: 'rg-top', points: pts(top) }, grp);
  }
  const [lx, ly] = up(r.points, r.height).reduce((a, b) => (b[1] < a[1] ? b : a));
  el('text', { class: 'rg-label', x: lx, y: ly - 6 }, grp).textContent = r.name;
  if (sel && S.mode === 'edit') r.points.forEach(([x, y], i) => el('circle', { class: 'vtx', cx: x, cy: y, r: 6, 'data-vtx': i }, grp));
  return grp;
}

function renderStage() {
  const vb = viewBox();
  $('svg').setAttribute('viewBox', vb.join(' '));
  const k = vb[2] / 1280;   // giữ cỡ chữ / chấm khi phóng to
  const g = $('layer');
  g.replaceChildren();
  g.style.fontSize = '';
  for (const r of S.meta.regions) if (r.name !== S.region) drawRegion(g, r, false);
  if (regionDef()) drawRegion(g, regionDef(), true);
  for (const t of g.querySelectorAll('text')) t.style.fontSize = `${14 * Math.max(k, 0.35)}px`;
  for (const c of g.querySelectorAll('circle')) c.setAttribute('r', 6 * Math.max(k, 0.35));
  if (S.mode === 'label') drawObjects(g, k);
  if (S.draft) drawDraft(g);
}
function render() {
  if (!S.meta) return;
  renderStage();
  S.mode === 'label' ? renderLabelSide() : renderEditSide();
}

function drawObjects(g, k) {
  const c = cur();
  if (!c) return;
  const items = new Set(selectedItems(c));
  const box = (b, cls, text, i) => {
    const o = el('g', { class: `ob ${cls}${S.hl && S.hl !== text ? ' faint' : ''}${S.hl === text ? ' hl' : ''}`, ...(i != null ? { 'data-box': i } : {}) }, g);
    el('rect', { x: b[0], y: b[1], width: b[2] - b[0], height: b[3] - b[1] }, o);
    el('text', { x: b[0] + 2, y: b[1] - 4, style: `font-size:${12 * Math.max(k, 0.35)}px` }, o).textContent = text;
  };
  for (const h of c.hints) box(h.box, `machine${items.has(h.label) ? ' on' : ''}`, h.label);
  (c.label?.boxes ?? []).forEach((b, i) => box(b.box, `user${i === S.selBox ? ' sel' : ''}`, b.label, i));
  if (S.pendingBox) box(S.pendingBox, 'user sel', '?');
}

function drawDraft(g) {
  const d = S.draft;
  if (d.kind === 'box') {
    const [x1, y1, x2, y2] = norm(d.a, d.b);
    el('rect', { class: 'draft', x: x1, y: y1, width: x2 - x1, height: y2 - y1 }, g);
  } else if (d.kind === 'poly') {
    const p = d.hover ? [...d.points, d.hover] : d.points;
    el(p.length > 2 ? 'polygon' : 'polyline', { class: 'draft', points: pts(p) }, g);
    d.points.forEach(([x, y]) => el('circle', { class: 'vtx', cx: x, cy: y, r: 5 }, g));
  }
}
const norm = (a, b) => [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];

// ---------- bảng bên: gán
function renderLabelSide() {
  const chips = $('regionChips');
  chips.replaceChildren();
  if (!S.meta.regions.length) {
    chips.textContent = 'Chưa có vùng nào. Chuyển sang "Sửa vùng" để vẽ.';
  }
  S.meta.regions.forEach((r, i) => {
    const b = document.createElement('button');
    b.className = `o-chip${r.name === S.region ? ' on' : ''}`;
    b.innerHTML = `${i < 9 ? `<kbd>${i + 1}</kbd>` : ''}<span></span><small>${S.meta.stats[r.name] || 0}</small>`;
    b.querySelector('span').textContent = r.name;
    b.title = `${S.meta.stats[r.name] || 0} ảnh có nhãn riêng`;
    b.onclick = () => selectRegion(r.name);
    chips.append(b);
  });
  const c = cur();
  const st = $('status');
  for (const id of ['confirm', 'byMachine', 'addObj', 'nextChange', 'nextTodo', 'search']) $(id).disabled = !c;
  if (!c) { st.className = 'o-status none'; st.textContent = '—'; $('list').replaceChildren(); return; }
  const lab = c.label;
  if (!lab) { st.className = 'o-status none'; st.textContent = 'Chưa gán'; }
  else if (lab.from === S.img.image_id) {
    st.className = 'o-status own';
    st.innerHTML = '<span>✓ Nhãn riêng của ảnh này</span><button class="ghost" id="clearOwn" title="Bỏ nhãn riêng, quay về kế thừa ảnh trước">Bỏ</button>';
    $('clearOwn').onclick = () => guard(async () => { await api('/api/objects/clear', { region: S.region, image_id: S.img.image_id }); await refresh(); });
  } else {
    st.className = 'o-status inherited';
    st.innerHTML = '<span></span><button class="ghost" id="jumpFrom">Mở ảnh đó</button>';
    st.firstChild.textContent = `↳ Kế thừa từ ${lab.from}`;
    $('jumpFrom').onclick = () => go(lab.from);
  }
  renderList(c);
}

function renderList(c) {
  const items = new Set(selectedItems(c));
  const hints = new Map(c.hints.map((h) => [h.label, h]));
  const userCount = new Map();
  for (const b of c.label?.boxes ?? []) userCount.set(b.label, (userCount.get(b.label) || 0) + 1);
  const q = $('search').value.trim().toLowerCase();
  const match = (n) => !q || n.toLowerCase().includes(q);
  const names = S.meta.vocab.map((v) => v.name).filter(match);
  const groups = [
    ['Đang tick', names.filter((n) => items.has(n))],
    ['Máy thấy trong vùng', names.filter((n) => !items.has(n) && hints.has(n))],
    ['Còn lại', names.filter((n) => !items.has(n) && !hints.has(n)).sort((a, b) => a.localeCompare(b, 'vi'))],
  ];
  const list = $('list');
  list.replaceChildren();
  for (const [title, ns] of groups) {
    if (!ns.length) continue;
    list.append(Object.assign(document.createElement('div'), { className: 'o-group', textContent: `${title} · ${ns.length}` }));
    for (const n of ns) {
      const row = document.createElement('label');
      row.className = `o-item${items.has(n) ? ' on' : ''}`;
      row.dataset.name = n;
      const h = hints.get(n), u = userCount.get(n);
      row.innerHTML = `<input type="checkbox"${items.has(n) ? ' checked' : ''}><span class="o-name"></span><span>${h ? `<span class="o-badge" title="Máy dò, độ tin">${h.score.toFixed(2)}</span>` : ''}${u ? ` <span class="o-badge user" title="Box tự vẽ">✎${u}</span>` : ''}</span>`;
      row.querySelector('.o-name').textContent = n;
      row.querySelector('input').onchange = (e) => toggle(n, e.target.checked);
      row.onmouseenter = () => { S.hl = n; renderStage(); };
      row.onmouseleave = () => { S.hl = null; renderStage(); };
      list.append(row);
    }
  }
  if (q && !names.length) list.append(Object.assign(document.createElement('div'), { className: 'o-group', textContent: `Không có "${q}". Enter để thêm tên mới (không cần box).` }));
}

// ---------- ghi nhãn
async function save(items, boxes) {
  const c = cur();
  const r = await api('/api/objects/label', { region: S.region, image_id: S.img.image_id, items, boxes: boxes ?? c.label?.boxes ?? [] });
  c.label = r.label;
  await loadMeta();
  render();
}
const curItems = () => [...selectedItems(cur())];
function toggle(name, on) {
  return guard(async () => {
    const items = curItems().filter((n) => n !== name);
    let boxes = cur().label?.boxes ?? [];
    if (on) items.push(name); else boxes = boxes.filter((b) => b.label !== name);   // bỏ tick = bỏ cả box tự vẽ của đồ đó
    await save(items, boxes);
    say(on ? `+ ${name}` : `− ${name}`);
  });
}
async function confirmNext() {
  await guard(async () => {
    const c = cur();
    if (c.label?.from !== S.img.image_id) await save(curItems());
    await go(S.img.next);
  });
}
async function byMachine() {
  await guard(async () => {
    const hints = cur().hints.map((h) => h.label);
    const keep = (cur().label?.boxes ?? []).map((b) => b.label);
    await save([...new Set([...hints, ...keep])]);
    say(hints.length ? `Tick theo máy: ${hints.join(', ')}` : 'Máy không thấy gì trong vùng: để trống.');
  });
}
async function jump(kind) {
  await guard(async () => {
    const r = await api(`/api/objects/next?${new URLSearchParams({ region: S.region, from: S.img.image_id, kind })}`);
    if (!r.id) return say(kind === 'change' ? 'Không còn ảnh nào máy thấy vùng đổi.' : 'Vùng này đã gán hết.');
    await go(r.id);
    say(kind === 'change' ? 'Máy thấy vùng đổi so với ảnh trước.' : 'Ảnh chưa có nhãn.');
  });
}
async function addPending() {
  const name = $('newName').value.trim();
  if (!name) return $('newName').focus();
  await guard(async () => {
    const boxes = [...(cur().label?.boxes ?? []), { label: name, box: S.pendingBox }];
    await save([...new Set([...curItems(), name])], boxes);
    S.pendingBox = null; $('newBox').hidden = true; $('newName').value = '';
    say(`Đã thêm ${name}.`);
  });
}
async function deleteUserBox() {
  if (S.selBox < 0) return;
  await guard(async () => {
    const boxes = [...cur().label.boxes];
    const [gone] = boxes.splice(S.selBox, 1);
    S.selBox = -1;
    await save(curItems(), boxes);
    say(`Đã xoá box ${gone.label} (đồ vẫn được tick, bỏ tick nếu cần).`);
  });
}
function selectRegion(name) { S.region = name; S.selBox = -1; S.pendingBox = null; $('newBox').hidden = true; go(S.img.image_id); }
async function refresh() { await loadMeta(); await go(S.img.image_id); }

// ---------- bảng bên: sửa vùng
function renderEditSide() {
  const list = $('regionList');
  list.replaceChildren();
  if (!S.meta.regions.length) list.innerHTML = '<p class="o-help">Chưa có vùng. Bấm <b>+ Vùng mới</b>.</p>';
  for (const r of S.meta.regions) {
    const row = document.createElement('div');
    row.className = `o-reg${r.name === S.region ? ' on' : ''}`;
    row.innerHTML = '<input aria-label="Tên vùng"><small></small><button class="ghost" title="Chọn">⌖</button>';
    const inp = row.querySelector('input');
    inp.value = r.name;
    inp.onfocus = () => { if (S.region !== r.name) { S.region = r.name; render(); row.querySelector('input').focus(); } };
    inp.onkeydown = (e) => { if (e.key === 'Enter') inp.blur(); if (e.key === 'Escape') { inp.value = r.name; inp.blur(); } };
    inp.onchange = () => guard(async () => {
      const to = inp.value.trim();
      if (!to || to === r.name) return;
      await saveRegions(S.meta.regions.map((x) => (x === r ? { ...x, name: to } : x)));
      S.region = to; render();
    });
    row.querySelector('small').textContent = `${r.points.length} điểm · cao ${r.height}`;
    row.querySelector('button').onclick = () => { S.region = r.name; render(); };
    list.append(row);
  }
  $('setHeight').disabled = $('delRegion').disabled = !regionDef();
}
async function saveRegions(regions) {
  S.meta = { ...S.meta, ...(await api('/api/objects/regions', { regions })) };
  say('Đã lưu vùng.');
}
function startPolygon() {
  S.tool = 'place'; S.draft = { kind: 'poly', points: [] };
  hint('Bấm các điểm quanh vùng · Enter / bấm điểm đầu: khép · Backspace: xoá điểm · Esc: huỷ');
  render();
}
async function closePolygon() {
  const p = S.draft.points;
  if (p.length < 3) return say('Cần ít nhất 3 điểm.', true);
  let n = S.meta.regions.length + 1;
  while (S.meta.regions.some((r) => r.name === `vùng ${n}`)) n++;
  S.tool = null; S.draft = null; hint('');
  await guard(async () => {
    await saveRegions([...S.meta.regions, { name: `vùng ${n}`, points: p, height: 0 }]);
    S.region = `vùng ${n}`;
    startHeight();   // chốt chiều cao xong thì ô tên được chọn sẵn để đổi tên
    S.renameAfter = true;
  });
}
function startHeight() {
  const r = regionDef();
  if (!r) return;
  S.tool = 'height'; S.draft = null;
  S.heightOf = { name: r.name, before: r.height, topY: Math.min(...r.points.map((p) => p[1])) };
  hint('Rê chuột lên để đặt chiều cao khối (tới đỉnh đồ cao nhất) · bấm để chốt · Esc: huỷ');
  render();
}
async function commitHeight() {
  S.tool = null; hint('');
  await guard(() => saveRegions(S.meta.regions));
  render();
  if (S.renameAfter) {
    S.renameAfter = false;
    const inp = $('regionList').querySelector('.o-reg.on input');
    inp?.focus(); inp?.select();
    say('Gõ tên vùng (vd: mặt bàn đảo, cửa lò) rồi Enter.');
  }
}
async function deleteRegion() {
  const r = regionDef();
  if (!r || !confirm(`Xoá vùng "${r.name}"? Nhãn đã gán của vùng vẫn giữ trong file, vẽ lại vùng cùng tên là dùng lại được. Z để hoàn tác.`)) return;
  await guard(() => saveRegions(S.meta.regions.filter((x) => x !== r)));
  S.region = S.meta.regions[0]?.name ?? null;
  render();
}

// ---------- chuột trên ảnh
function hint(t) {
  $('hint').textContent = t;
  $('stage').className = `o-stage${S.tool === 'draw' ? ' drawing' : S.tool === 'place' ? ' placing' : S.tool === 'height' ? ' heighting' : ''}`;
}
function toImg(e) {
  const p = new DOMPoint(e.clientX, e.clientY).matrixTransform($('svg').getScreenCTM().inverse());
  return [Math.round(Math.min(W(), Math.max(0, p.x)) * 10) / 10, Math.round(Math.min(H(), Math.max(0, p.y)) * 10) / 10];
}
const svg = $('svg');
svg.addEventListener('pointerdown', (e) => {
  if (!S.meta || e.button !== 0) return;
  const p = toImg(e);
  if (S.tool === 'height') return commitHeight();
  if (S.tool === 'place') {
    const first = S.draft.points[0];
    const near = first && Math.hypot(first[0] - p[0], first[1] - p[1]) < 12 * (viewBox()[2] / svg.clientWidth);
    if (near && S.draft.points.length >= 3) return closePolygon();
    S.draft.points.push(p); render(); return;
  }
  if (S.tool === 'draw') { S.draft = { kind: 'box', a: p, b: p }; svg.setPointerCapture(e.pointerId); return; }
  const vtx = e.target.closest('[data-vtx]');
  if (vtx && S.mode === 'edit') { S.tool = 'drag'; S.dragIdx = Number(vtx.dataset.vtx); svg.setPointerCapture(e.pointerId); return; }
  const ub = e.target.closest('[data-box]');
  if (S.mode === 'label') { S.selBox = ub ? Number(ub.dataset.box) : -1; render(); }
});
svg.addEventListener('pointermove', (e) => {
  if (!S.meta) return;
  const p = toImg(e);
  if (S.tool === 'place') { S.draft.hover = p; renderStage(); }
  else if (S.tool === 'draw' && S.draft) { S.draft.b = p; renderStage(); }
  else if (S.tool === 'height') { regionDef().height = Math.round(Math.max(0, S.heightOf.topY - p[1])); hint(`Chiều cao: ${regionDef().height} px · bấm để chốt`); render(); }
  else if (S.tool === 'drag') { regionDef().points[S.dragIdx] = p; renderStage(); }
});
svg.addEventListener('pointerup', (e) => {
  if (S.tool === 'drag') { S.tool = null; guard(() => saveRegions(S.meta.regions)).then(render); return; }
  if (S.tool === 'draw' && S.draft) {
    const b = norm(S.draft.a, S.draft.b);
    S.tool = null; S.draft = null; hint('');
    if (b[2] - b[0] < 6 || b[3] - b[1] < 6) { render(); return; }
    S.pendingBox = b;
    $('newBox').hidden = false;
    render();
    $('newName').focus();
  }
});
svg.addEventListener('dblclick', () => { if (S.tool === 'place') closePolygon(); });

function cancel() {
  if (S.tool === 'height') { regionDef().height = S.heightOf.before; }
  if (S.tool === 'drag') return;
  S.tool = null; S.draft = null; S.pendingBox = null; $('newBox').hidden = true; S.selBox = -1; hint('');
  if (S.mode === 'edit') loadMeta().then(render); else render();
}

// ---------- chế độ, phím
function setMode(m) {
  cancel();
  S.mode = m;
  for (const b of document.querySelectorAll('[data-mode]')) b.classList.toggle('on', b.dataset.mode === m);
  $('labelSide').hidden = m !== 'label';
  $('editSide').hidden = m !== 'edit';
  go(S.img.image_id);
}
for (const b of document.querySelectorAll('[data-mode]')) b.onclick = () => setMode(b.dataset.mode);
$('zoom').onchange = render;
$('confirm').onclick = confirmNext;
$('byMachine').onclick = byMachine;
$('addObj').onclick = () => { S.tool = 'draw'; hint('Kéo để vẽ box quanh đồ vật · Esc: huỷ'); };
$('nextChange').onclick = () => jump('change');
$('nextTodo').onclick = () => jump('unlabeled');
$('newOk').onclick = addPending;
$('newName').onkeydown = (e) => { if (e.key === 'Enter') addPending(); if (e.key === 'Escape') { e.stopPropagation(); cancel(); } };
$('newRegion').onclick = startPolygon;
$('setHeight').onclick = startHeight;
$('delRegion').onclick = deleteRegion;
$('help').onclick = () => $('helpDlg').showModal();
$('search').oninput = () => renderList(cur());
$('search').onkeydown = (e) => {
  if (e.key === 'Escape') { $('search').value = ''; $('search').blur(); renderList(cur()); }
  if (e.key !== 'Enter') return;
  const q = $('search').value.trim();
  const first = $('list').querySelector('.o-item:not(.on)');
  if (first) toggle(first.dataset.name, true);
  else if (q) toggle(q, true);   // tên mới, không box
  $('search').value = '';
};
$('imgId').onkeydown = (e) => { if (e.key === 'Enter') { guard(() => go($('imgId').value.trim())); $('imgId').blur(); } };
$('undo').onclick = undo;
async function undo() {
  await guard(async () => { await api('/api/undo', {}); await refresh(); say('Đã hoàn tác.'); });
}

document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, textarea, select') || e.ctrlKey || e.metaKey || e.altKey || !S.img) return;
  const k = e.key;
  if (k === 'Escape') return cancel();
  if (k === 'z' || k === 'Z') return undo();
  if (k === '?') return $('helpDlg').showModal();
  if (k === 'f' || k === 'F') { $('zoom').checked = !$('zoom').checked; return render(); }
  if (k === 'ArrowLeft') return go(S.img.prev);
  if (k === 'ArrowRight') return go(S.img.next);
  if (S.mode === 'edit') {
    if (S.tool === 'place' && k === 'Enter') return closePolygon();
    if (S.tool === 'place' && k === 'Backspace') { S.draft.points.pop(); return render(); }
    if (k === 'p' || k === 'P') return startPolygon();
    if (k === 'h' || k === 'H') return startHeight();
    if (k === 'Delete') return deleteRegion();
    return;
  }
  if (!cur()) return;
  if (/^[1-9]$/.test(k) && S.meta.regions[k - 1]) return selectRegion(S.meta.regions[k - 1].name);
  if (k === 'Enter') { e.preventDefault(); return confirmNext(); }
  if (k === ' ') { e.preventDefault(); return jump('change'); }
  if (k === 'u' || k === 'U') return jump('unlabeled');
  if (k === 'm' || k === 'M') return byMachine();
  if (k === 'n' || k === 'N') return $('addObj').click();
  if (k === '/') { e.preventDefault(); return $('search').focus(); }
  if (k === 'Delete' || k === 'Backspace') return deleteUserBox();
});

(async () => {
  await guard(async () => {
    const q = new URLSearchParams(location.search);
    await loadMeta();
    if (q.get('region')) S.region = q.get('region');
    if (!S.meta.regions.some((r) => r.name === S.region)) S.region = S.meta.regions[0]?.name ?? null;
    S.mode = q.get('mode') === 'edit' || !S.meta.regions.length ? 'edit' : 'label';
    for (const b of document.querySelectorAll('[data-mode]')) b.classList.toggle('on', b.dataset.mode === S.mode);
    $('labelSide').hidden = S.mode !== 'label';
    $('editSide').hidden = S.mode !== 'edit';
    await go(q.get('id') || S.meta.ids[0]);
  });
})();

'use strict';
// Giao diện gán nhãn xoay quanh MỘT tấm ảnh: box trên ảnh, bấm box mới hiện bảng nhãn.
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
const SVGNS = 'http://www.w3.org/2000/svg';
const sv = (tag, attrs = {}, text) => { const el = document.createElementNS(SVGNS, tag); for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v); if (text != null) el.textContent = text; return el; };
const NONE = '__none__';

// ---------------------------------------------------------------- state, đồng bộ URL hash
const DEFAULTS = { id: '', show: 'people', hideDone: false, labels: false, objects: false, tokens: [], mode: 'or', q: '', src: [], hours: [], edited: false, manual: false, conflict: false };
const S = { ...DEFAULTS };
let META = null, PROF = {}, QUEUE = [], D = null, SEL = '', NAT = { w: 1280, h: 720 };
// vẽ box thủ công
let DRAW = false, DRAFT = null, LASTKIND = 'person', suppressClick = false;
// đang kéo sửa box tự vẽ: {key, mode: 'move' | tổ hợp n/s/e/w, start: [x, y], orig, box}
let EDIT = null;
const editBox = (key, box) => (EDIT && EDIT.key === key ? EDIT.box : box);

function toHash() {
  const p = new URLSearchParams();
  if (S.id) p.set('id', S.id);
  if (S.show !== DEFAULTS.show) p.set('show', S.show);
  if (S.hideDone) p.set('hideDone', '1');
  if (S.labels) p.set('labels', '1');
  if (S.objects) p.set('obj', '1');
  for (const [k, v] of Object.entries(queueParams())) p.set(k, v);
  history.replaceState(null, '', '#' + p);
}
function fromHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  Object.assign(S, DEFAULTS, { tokens: [], src: [], hours: [] });
  S.id = p.get('id') || '';
  S.show = ['people', 'roi', 'all'].includes(p.get('show')) ? p.get('show') : 'people';
  S.hideDone = p.get('hideDone') === '1';
  S.labels = p.get('labels') === '1';
  S.objects = p.get('obj') === '1';
  S.tokens = (p.get('t') || '').split(';;').filter(Boolean).map((s) => { const i = s.indexOf('|'); return { profile: s.slice(0, i), label: s.slice(i + 1) }; });
  S.mode = p.get('mode') === 'and' ? 'and' : 'or';
  S.q = p.get('q') || '';
  S.src = (p.get('src') || '').split(',').filter(Boolean);
  S.hours = (p.get('hours') || '').split(',').filter(Boolean);
  S.edited = p.get('edited') === '1';
  S.manual = p.get('manual') === '1';
  S.conflict = p.get('conflict') === '1';
}
function queueParams() {
  const o = {};
  if (S.tokens.length) o.t = S.tokens.map((t) => `${t.profile}|${t.label}`).join(';;');
  if (S.tokens.length > 1) o.mode = S.mode;
  if (S.q) o.q = S.q;
  if (S.src.length) o.src = S.src.join(',');
  if (S.hours.length) o.hours = S.hours.join(',');
  if (S.edited) o.edited = '1';
  if (S.manual) o.manual = '1';
  if (S.conflict) o.conflict = '1';
  return o;
}
const activeFilters = () => S.tokens.length + (S.q ? 1 : 0) + (S.src.length ? 1 : 0) + (S.hours.length ? 1 : 0) + (S.edited ? 1 : 0) + (S.manual ? 1 : 0) + (S.conflict ? 1 : 0);

// ---------------------------------------------------------------- api
async function api(url, body) {
  const res = await fetch(url, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error(data.error || res.statusText), { status: res.status });
  return data;
}
let toastT;
function toast(msg, err) {
  const t = $('toast'); t.textContent = msg; t.className = 'toast' + (err ? ' err' : ''); t.hidden = false;
  clearTimeout(toastT); toastT = setTimeout(() => (t.hidden = true), err ? 6000 : 1800);
}
const fail = (e) => toast(e.message, true);

// ---------------------------------------------------------------- đơn vị gán nhãn của ảnh hiện tại
const statusOf = (st) => (!st ? 'todo' : st.source === 'human' ? 'human' : 'machine');
const STATUS_VI = { todo: 'Chưa có nhãn', machine: 'Máy gán', human: 'Đã duyệt' };
const SOURCE_VI = { vlm: 'VLM', seed: 'hạt giống', suggested: 'gợi ý', human: 'người duyệt' };
const uid = (u) => `${u.profile}|${u.key}`;

function units() {
  if (!D) return [];
  const out = [];
  if (PROF.image) out.push({ ...D.activity, kind: 'image', kindVi: 'Cả ảnh', title: D.image_id, crop: `/img/${D.image_id}.jpg` });
  for (const p of D.people) out.push({ ...p, kind: 'person', kindVi: 'Người', title: `Người ${p.id.split('_').pop()}`, sub: p.zone_vi, crop: `/crop/person/${p.id}.jpg`, box: p.box });
  for (const r of D.rois) {
    const [x1, y1, x2, y2] = r.box;
    out.push({ ...r, kind: 'roi', kindVi: 'Vùng', title: r.name_vi || r.name, sub: r.name, crop: `/crop/roi/${r.name}/${r.key}.jpg`, box: [x1 * NAT.w, y1 * NAT.h, x2 * NAT.w, y2 * NAT.h] });
  }
  for (const m of D.manual || []) {
    out.push({ kind: 'manual', manualKind: m.kind, profile: 'manual', key: m.id, kindVi: 'Box tự vẽ', title: `Box ${m.id.split('_').pop()}`,
      state: m.label ? { label: m.label, label_vi: m.label_vi || '', source: 'human' } : null, box: editBox(m.id, m.box) });
  }
  if (DRAFT) out.push({ kind: 'manual', draft: true, manualKind: DRAFT.kind, profile: 'manual', key: 'draft', kindVi: 'Box tự vẽ', title: 'Box mới', state: null, box: editBox('draft', DRAFT.box) });
  return out;
}
// bộ lọc "hành động lệch vùng": cùng luật với store.js ACTION_ZONE
const isConflict = (u) => S.conflict && u.kind === 'person' && u.state && u.state.source !== 'human' && !!META.actionZone?.[u.state.label] && u.zone !== META.actionZone[u.state.label];
const isMatch = (u) => isConflict(u) || S.tokens.some((t) => t.profile === u.profile && (t.label === NONE ? !u.state : u.state?.label === t.label));
function visible() {
  return units().filter((u) => {
    if (u.kind === 'image') return false;
    if (u.kind === 'manual') return u.draft || !(S.hideDone && u.state && uid(u) !== SEL);
    if (S.show === 'people' && u.kind !== 'person') return false;
    if (S.show === 'roi' && u.kind !== 'roi') return false;
    if (S.hideDone && statusOf(u.state) === 'human' && uid(u) !== SEL) return false;
    return true;
  });
}
const current = () => units().find((u) => uid(u) === SEL);

// ---------------------------------------------------------------- vẽ ảnh + box
function fit() {
  const st = $('stage'), cs = getComputedStyle(st);
  const W = st.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const H = st.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
  const k = Math.min(W / NAT.w, H / NAT.h);
  $('frame').style.width = `${Math.floor(NAT.w * k)}px`;
  $('frame').style.height = `${Math.floor(NAT.h * k)}px`;
}
function drawBoxes() {
  const svg = $('boxes');
  svg.setAttribute('viewBox', `0 0 ${NAT.w} ${NAT.h}`);
  const vis = visible().sort((a, b) => area(b) - area(a)); // box nhỏ vẽ sau, nằm trên, dễ bấm
  const hasTok = S.tokens.length > 0 || S.conflict;
  const kids = [];
  let selU = null;
  for (const u of vis) {
    const [x1, y1, x2, y2] = u.box;
    const st = statusOf(u.state);
    const sel = uid(u) === SEL;
    if (sel) selU = u;
    const g = sv('g', { class: `box ${st} ${u.kind === 'roi' || u.kind === 'manual' ? u.kind : ''} ${sel ? 'sel' : ''} ${hasTok && !isMatch(u) && !sel ? 'dim' : ''}`, 'data-id': uid(u), tabindex: -1 });
    const r = { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
    g.append(sv('rect', { ...r, class: 'halo', 'stroke-width': sel ? 7 : 5 }), sv('rect', { ...r, class: 'edge', 'stroke-width': sel ? 4 : 2.5 }), sv('rect', { ...r, class: 'hit' }));
    g.append(sv('title', {}, `${u.title}: ${u.state ? u.state.label_vi || u.state.label : 'chưa có nhãn'}`));
    g.addEventListener('click', (e) => { e.stopPropagation(); select(uid(u)); });
    if (sel && u.kind === 'manual' && !DRAW) g.addEventListener('pointerdown', (e) => startEdit(e, u, 'move'));
    g.addEventListener('mouseenter', () => showTag(u, 'hover'));
    g.addEventListener('mouseleave', () => removeTag('hover'));
    kids.push(g);
    if (S.labels && !sel) kids.push(tag(u, ''));
  }
  svg.replaceChildren(...(S.objects ? objectMarks() : []), ...kids);
  if (selU) svg.append(tag(selU, 'selected'));
  if (selU?.kind === 'manual' && !DRAW) svg.append(...handles(selU));
}
// đồ vật do `enrich objects` dò: chỉ để xem, không chọn được
function objectMarks() {
  return (D?.objects || []).map((o) => {
    const [x1, y1, x2, y2] = o.box;
    const g = sv('g', { class: 'obj', 'pointer-events': 'none' });
    g.append(sv('rect', { x: x1, y: y1, width: x2 - x1, height: y2 - y1 }), sv('text', { x: x1 + 3, y: Math.max(12, y1 - 3) }, `${o.label} ${Math.round(o.score * 100)}`));
    return g;
  });
}
// 8 tay nắm đổi cỡ, kích thước cố định trên màn hình dù ảnh co giãn
function handles(u) {
  const [x1, y1, x2, y2] = u.box, mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
  const s = (10 * NAT.w) / Math.max(1, $('boxes').clientWidth);
  return [[x1, y1, 'nw'], [mx, y1, 'n'], [x2, y1, 'ne'], [x2, my, 'e'], [x2, y2, 'se'], [mx, y2, 's'], [x1, y2, 'sw'], [x1, my, 'w']].map(([x, y, mode]) => {
    const el = sv('rect', { class: `handle h-${mode}`, x: x - s / 2, y: y - s / 2, width: s, height: s });
    el.addEventListener('pointerdown', (e) => startEdit(e, u, mode));
    el.addEventListener('click', (e) => e.stopPropagation());
    return el;
  });
}
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
function svgPt(e) {
  const r = $('boxes').getBoundingClientRect();
  return [clamp(((e.clientX - r.left) / r.width) * NAT.w, 0, NAT.w), clamp(((e.clientY - r.top) / r.height) * NAT.h, 0, NAT.h)];
}
function startEdit(e, u, mode) {
  if (e.button !== 0) return;
  e.preventDefault(); e.stopPropagation();
  $('boxes').setPointerCapture(e.pointerId);
  EDIT = { key: u.key, unit: u, mode, start: svgPt(e), orig: [...u.box], box: [...u.box] };
  removeTag('hover');
}
// box mới sau khi kéo dx, dy; giữ trong ảnh và không nhỏ hơn 8px
function draggedBox([x1, y1, x2, y2], mode, dx, dy) {
  const MIN = 8;
  if (mode === 'move') {
    dx = clamp(dx, -x1, NAT.w - x2); dy = clamp(dy, -y1, NAT.h - y2);
    return [x1 + dx, y1 + dy, x2 + dx, y2 + dy];
  }
  if (mode.includes('w')) x1 = clamp(x1 + dx, 0, x2 - MIN);
  if (mode.includes('e')) x2 = clamp(x2 + dx, x1 + MIN, NAT.w);
  if (mode.includes('n')) y1 = clamp(y1 + dy, 0, y2 - MIN);
  if (mode.includes('s')) y2 = clamp(y2 + dy, y1 + MIN, NAT.h);
  return [x1, y1, x2, y2];
}
async function finishEdit() {
  const { unit: u, orig } = EDIT;
  const box = EDIT.box.map((v) => Math.round(v * 10) / 10);
  EDIT = null;
  suppressClick = true;
  if (box.every((v, i) => Math.abs(v - orig[i]) < 0.5)) return drawBoxes(); // chỉ bấm, không kéo
  if (u.draft) { DRAFT.box = box; drawBoxes(); return renderPanel(); }
  try {
    await api('/api/manual', { action: 'update', image_id: D.image_id, id: u.key, box });
    $('undo').disabled = false;
  } catch (e) { fail(e); }
  await loadImage(S.id, true);
}
const area = (u) => (u.box[2] - u.box[0]) * (u.box[3] - u.box[1]);
function tag(u, role) {
  const text = u.state ? u.state.label_vi || u.state.label : 'chưa có nhãn';
  const g = sv('g', { class: `tag ${statusOf(u.state)}`, 'data-role': role, 'pointer-events': 'none' });
  const x = u.box[0], y = Math.max(22, u.box[1] - 6);
  const w = Math.min(420, 16 + text.length * 7.6);
  g.append(sv('rect', { x, y: y - 20, width: w, height: 24 }), sv('text', { x: x + 8, y: y - 3 }, text));
  return g;
}
function showTag(u, role) { removeTag(role); if (!S.labels && uid(u) !== SEL) $('boxes').append(tag(u, role)); }
function removeTag(role) { $('boxes').querySelector(`[data-role="${role}"]`)?.remove(); }

// ---------------------------------------------------------------- bảng nhãn (chỉ khi chọn box)
let classFilter = '';
let PTAB = 'action'; // tab của bảng người: action | who
function renderPanel() {
  const u = current();
  const panel = $('panel');
  if (!u) { panel.hidden = true; requestAnimationFrame(() => { fit(); }); return; }
  const wasHidden = panel.hidden;
  panel.hidden = false;
  if (wasHidden) requestAnimationFrame(fit);
  if (u.kind === 'manual') return renderManualPanel(u, panel);
  const p = PROF[u.profile];
  const st = statusOf(u.state);
  const all = p?.classes || [];
  const f = norm(classFilter);
  const shown = all.filter((c) => !f || norm(`${c.name} ${c.name_vi}`).includes(f));
  const filterBox = h('input', { type: 'search', placeholder: `Tìm trong ${all.length} lớp…`, value: classFilter, 'aria-label': 'Tìm lớp',
    oninput: (e) => { classFilter = e.target.value; renderPanel(); const i = $('panel').querySelector('.classes input'); i.focus(); i.setSelectionRange(i.value.length, i.value.length); } });
  const tabs = u.kind === 'person' ? h('div', { class: 'seg p-tabs', role: 'tablist' }, [['action', 'Hành động'], ['who', 'Ai · ngoại hình']].map(([k, t]) =>
    h('button', { class: PTAB === k ? 'on' : '', role: 'tab', 'aria-selected': PTAB === k, onclick: () => { PTAB = k; renderPanel(); } }, t, h('kbd', {}, k === 'action' ? 'H' : 'N')))) : '';
  if (u.kind === 'person' && PTAB === 'who') return panel.replaceChildren(headOf(u, st), tabs, whoPanel(u));
  panel.replaceChildren(
    // crop bên trái, nhãn hiện tại bên phải: chừa chỗ cho lưới thẻ để không phải cuộn
    h('div', { class: 'p-head p-row' },
      h('div', { class: 'p-crop' }, h('img', { src: u.crop, alt: `Crop ${u.title}` })),
      h('div', { class: 'p-info' },
        h('div', { class: 'p-kind' }, u.kindVi, u.sub ? ` · ${u.sub}` : ''),
        h('h2', { class: 'p-title' }, u.title),
        h('div', {}, h('span', { class: `badge ${st}` }, u.state?.custom ? 'Tự gõ' : STATUS_VI[st]), u.state && st !== 'human' ? h('span', { class: 'muted' }, `  ${SOURCE_VI[u.state.source] || u.state.source}${u.state.score != null ? ` · ${Number(u.state.score).toFixed(2)}` : ''}`) : null),
        h('div', { class: 'p-label' }, u.state ? u.state.label_vi || u.state.label : 'Chưa có nhãn'),
        u.state && !u.state.custom ? h('div', { class: 'p-en' }, u.state.label) : null)),
    tabs,
    u.inherited || u.rejected?.length ? h('div', { class: 'p-now' },
      u.inherited ? h('div', { class: 'note warn' }, `Vùng không đổi so với nền: nhãn dùng chung cho ${u.affected} ảnh giờ ${D.hour}.`) : null,
      u.rejected?.length ? h('div', { class: 'note rej' }, `Đã đánh dấu sai: ${u.rejected.join(', ')}`) : null) : '',
    h('div', { class: 'actions' },
      h('button', { class: 'btn primary', disabled: !u.state, onclick: () => act('confirm') }, 'Đúng', h('kbd', {}, 'A')),
      h('button', { class: 'btn bad', disabled: !u.state, onclick: () => act('reject') }, 'Sai', h('kbd', {}, 'X')),
      h('button', { class: 'btn', title: 'Đặt lại về nhãn của máy (R)', onclick: () => act('reset') }, 'Đặt lại', h('kbd', {}, 'R'))),
    h('div', { class: 'classes' }, all.length > 20 ? filterBox : null,
      shown.map((c, i) => h('button', { class: 'cls' + (u.state?.label === c.name && !u.state?.custom ? ' cur' : '') + ((u.rejected || []).includes(c.name) ? ' off' : ''), onclick: () => act('set', c.name), title: c.name },
        h('span', { class: 'vi' }, c.name_vi || c.name))),
      !shown.length ? h('div', { class: 'p-empty' }, 'Không có lớp khớp.') : null,
      otherBox(u)));
  panel._shown = shown;
}

// đầu bảng gọn cho tab "Ai · ngoại hình"
function headOf(u, st) {
  return h('div', { class: 'p-head p-row' },
    h('div', { class: 'p-crop' }, h('img', { src: u.crop, alt: `Crop ${u.title}` })),
    h('div', { class: 'p-info' },
      h('div', { class: 'p-kind' }, u.kindVi, u.sub ? ` · ${u.sub}` : ''),
      h('h2', { class: 'p-title' }, u.title),
      h('div', { class: 'p-label' }, u.identity?.name || 'Chưa rõ là ai'),
      h('div', { class: 'p-en' }, u.state ? u.state.label_vi || u.state.label : 'chưa có nhãn hành động')));
}

const ATTR_VI = [['gender', 'Giới'], ['age', 'Tuổi'], ['hair', 'Tóc'], ['top', 'Áo'], ['bottom', 'Quần/váy'], ['headwear', 'Mũ'], ['glasses', 'Kính'], ['apron', 'Tạp dề'], ['holding', 'Đang cầm'], ['view', 'Hướng']];
function attrText(a, k) {
  if (k === 'top') return [a.top, a.top_color, a.top_pattern !== 'plain' && a.top_pattern !== 'unknown' ? a.top_pattern : ''].filter((x) => x && x !== 'unknown').join(' · ') || 'unknown';
  if (k === 'bottom') return [a.bottom, a.bottom_color].filter((x) => x && x !== 'unknown').join(' · ') || 'unknown';
  if (k === 'holding') return a.holding?.length ? a.holding.join(', ') : '—';
  return a[k] || 'unknown';
}

// "người này là ai" (đặt tên theo cụm ngoại hình) + mô tả ngoại hình của VLM
function whoPanel(u) {
  const id = u.identity, c = id?.cluster, info = c ? META.identity?.clusters?.[c] : null;
  const input = h('input', { type: 'text', id: 'whoName', list: 'whoNames', maxlength: 60, value: id?.name || '', placeholder: 'Tên người, ví dụ: An', 'aria-label': 'Tên người',
    onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); setWho(u, c ? 'cluster' : 'person', input.value); } } });
  const where = !id ? 'Chưa gom cụm. Chạy `python main.py enrich identities`.'
    : id.source === 'person' ? `Đặt riêng cho người này (cụm ${c ?? '—'})`
      : `Cụm ${c}${info?.label ? ` (${info.label})` : ''} · ${info?.size ?? '?'} crop · giống ${id.sim != null ? Math.round(id.sim * 100) : '?'}%${id.name ? '' : ' · chưa đặt tên'}`;
  const a = u.attributes;
  return h('div', { class: 'who' },
    h('div', { class: 'p-kind' }, 'Là ai'),
    h('div', { class: 'muted small' }, where),
    h('div', { class: 'other-row' }, input, h('datalist', { id: 'whoNames' }, (META.identity?.names || []).map((n) => h('option', { value: n }))),
      h('button', { class: 'btn', title: 'Điền số người kế tiếp chưa dùng (Người 01, Người 02...)', onclick: () => { input.value = nextNumberName(); input.focus(); } }, 'Số mới')),
    h('div', { class: 'who-btns' },
      c ? h('button', { class: 'btn primary', onclick: () => setWho(u, 'cluster', input.value), title: 'Mọi crop trong cụm nhận tên này (Enter)' }, `Cả cụm ${c}`) : '',
      h('button', { class: 'btn', onclick: () => setWho(u, 'person', input.value) }, 'Chỉ người này'),
      h('button', { class: 'btn bad', onclick: () => setWho(u, 'person', '') , title: 'Người này không phải người của cụm' }, 'Không rõ'),
      id?.source === 'person' ? h('button', { class: 'btn', onclick: () => setWho(u, 'person', null) }, 'Theo cụm') : ''),
    info ? h('div', { class: 'p-kind', style: 'margin-top:10px' }, `Crop tiêu biểu của cụm ${c} · `, h('a', { href: `/board.html#cluster=${c}`, target: '_blank' }, 'xem tất cả, sửa nhầm')) : '',
    info ? h('div', { class: 'who-strip' }, info.examples.map((pid) => h('img', { src: `/crop/person/${pid}.jpg`, alt: pid, title: pid, loading: 'lazy' }))) : '',
    u.predicted ? h('div', { class: 'muted small' }, `Mô hình đoán: ${u.predicted.name} (${Math.round(u.predicted.prob * 100)}%)${u.predicted.prob < 0.6 ? ' · thấp, không dùng' : ''}`) : '',
    u.outfit ? h('div', { class: 'p-kind', style: 'margin-top:12px' }, 'Trang phục (outfits.yaml)') : '',
    u.outfit ? h('dl', { class: 'attrs' }, Object.entries(u.outfit).filter(([, v]) => v !== '' && v != null).flatMap(([k, v]) => [h('dt', {}, k), h('dd', {}, Array.isArray(v) ? v.join(', ') : String(v))])) : '',
    h('div', { class: 'p-kind', style: 'margin-top:12px' }, 'Ngoại hình (VLM)'),
    a ? h('dl', { class: 'attrs' }, ATTR_VI.flatMap(([k, t]) => [h('dt', {}, t), h('dd', {}, attrText(a, k))]))
      : h('div', { class: 'muted small' }, 'Chưa có mô tả. Bước mô tả người đang chạy trên Kaggle; bấm "Gộp dữ liệu" khi có kết quả.'));
}
// "Người NN" nhỏ nhất chưa có trong các tên đã đặt
function nextNumberName() {
  const used = new Set(META.identity?.names || []);
  for (let n = 1; ; n++) { const name = `Người ${String(n).padStart(2, '0')}`; if (!used.has(name)) return name; }
}
async function setWho(u, scope, name) {
  const body = scope === 'cluster' ? { scope, cluster: u.identity?.cluster, name } : name === null ? { scope, person: u.key, reset: true } : { scope, person: u.key, name };
  if (scope === 'cluster' && !String(name).trim() && !confirm(`Xoá tên của cụm ${u.identity?.cluster}?`)) return;
  try {
    const r = await api('/api/identity', body);
    META.identity.names = r.names;
    if (scope === 'cluster' && META.identity.clusters[u.identity.cluster]) META.identity.clusters[u.identity.cluster].name = String(name).trim() || null;
    $('undo').disabled = false;
    await loadImage(S.id, true);
    toast(scope === 'cluster' ? `Đã đặt tên cho ${r.changed.length} ảnh có người của cụm.` : 'Đã lưu.');
  } catch (e) { fail(e); }
}

// ô cuối: tự gõ nhãn khi không lớp nào khớp (lưu riêng ở custom_labels.json)
function otherBox(u) {
  const input = h('input', { type: 'text', id: 'otherText', maxlength: 120, value: u.state?.custom ? u.state.label : '', placeholder: 'Tình huống khác, ví dụ: đang rót nước', 'aria-label': 'Nhãn tự gõ',
    onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); saveOther(); } } });
  const saveOther = () => { if (!input.value.trim()) return toast('Nhập mô tả trước khi lưu.', true); act('custom', null, input.value); };
  return h('div', { class: 'other' + (u.state?.custom ? ' cur' : '') },
    h('label', { class: 'p-kind', for: 'otherText' }, 'Khác (ngoài các thẻ)', h('kbd', {}, 'O')),
    h('div', { class: 'other-row' }, input, h('button', { class: 'btn', onclick: saveOther }, 'Lưu')));
}

// ---------------------------------------------------------------- box tự vẽ
function cropCanvas(box) {
  const [x1, y1, x2, y2] = box, w = x2 - x1, hh = y2 - y1;
  const k = Math.min(300 / w, 150 / hh, 3);
  const c = h('canvas', { width: Math.max(1, Math.round(w * k)), height: Math.max(1, Math.round(hh * k)), 'aria-label': 'Vùng đã vẽ' });
  c.getContext('2d').drawImage($('photo'), x1, y1, w, hh, 0, 0, c.width, c.height);
  return c;
}
function renderManualPanel(u, panel) {
  const kind = u.manualKind;
  const seg = h('div', { class: 'seg', role: 'group', 'aria-label': 'Loại box' }, [['person', 'Người'], ['object', 'Vật thể']].map(([k, t]) =>
    h('button', { class: kind === k ? 'on' : '', onclick: () => setManualKind(u, k) }, t)));
  let body;
  if (kind === 'person') {
    const all = PROF.people?.classes || [];
    const f = norm(classFilter);
    const shown = all.filter((c) => !f || norm(`${c.name} ${c.name_vi}`).includes(f));
    panel._shown = shown;
    body = h('div', { class: 'classes' },
      all.length > 9 ? h('input', { type: 'search', placeholder: `Tìm trong ${all.length} lớp…`, value: classFilter, 'aria-label': 'Tìm lớp',
        oninput: (e) => { classFilter = e.target.value; renderPanel(); const i = $('panel').querySelector('.classes input'); i.focus(); i.setSelectionRange(i.value.length, i.value.length); } }) : null,
      shown.map((c, i) => h('button', { class: 'cls' + (u.state?.label === c.name ? ' cur' : ''), onclick: () => saveManual(u, { label: c.name }) },
        h('span', {}, h('span', { class: 'vi' }, c.name_vi || c.name), h('span', { class: 'en' }, c.name)))));
  } else {
    panel._shown = null;
    const input = h('input', { type: 'text', list: 'objLabels', value: u.state?.label || '', placeholder: 'Tên vật, ví dụ: chảo, cốc, túi ni lông', 'aria-label': 'Tên vật thể', maxlength: 80,
      onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); saveManual(u, { label: input.value }); } } });
    body = h('div', { class: 'classes' },
      h('label', { class: 'p-kind', for: 'objName' }, 'Tên vật thể'),
      Object.assign(input, { id: 'objName' }),
      h('datalist', { id: 'objLabels' }, (META.manual?.objectLabels || []).map((l) => h('option', { value: l }))),
      h('button', { class: 'btn primary', style: 'width:100%', onclick: () => saveManual(u, { label: input.value }) }, 'Lưu nhãn', h('kbd', {}, 'Enter')));
    // chỉ box vừa vẽ mới cần gõ tên ngay; box đã lưu để phím tắt (Del, Z) còn dùng được
    if (u.draft || !u.state) setTimeout(() => { if (document.activeElement === document.body) input.focus(); }, 0);
  }
  panel.replaceChildren(
    h('div', { class: 'p-head' },
      h('div', { class: 'p-kind' }, u.draft ? 'Box mới, chưa lưu' : `${u.kindVi} · ${u.key}`),
      h('h2', { class: 'p-title' }, u.draft ? 'Gán nhãn cho box vừa vẽ' : u.title),
      h('div', { class: 'p-crop' }, cropCanvas(u.box))),
    h('div', { class: 'p-now' },
      h('div', { class: 'p-kind', style: 'margin-bottom:6px' }, 'Loại'), seg,
      h('div', { class: 'p-label', style: 'margin-top:10px' }, u.state ? u.state.label_vi || u.state.label : 'Chưa có nhãn'),
      u.state?.label_vi ? h('div', { class: 'p-en' }, u.state.label) : null),
    h('div', { class: 'actions', style: 'grid-template-columns:1fr' },
      u.draft ? h('button', { class: 'btn', onclick: cancelDraft }, 'Huỷ box này', h('kbd', {}, 'Esc'))
        : h('button', { class: 'btn bad', onclick: () => deleteManual(u) }, 'Xoá box', h('kbd', {}, 'Del'))),
    body);
}
async function setManualKind(u, kind) {
  LASTKIND = kind;
  if (u.draft) { DRAFT.kind = kind; return renderPanel(); }
  await saveManual(u, { kind });
}
async function saveManual(u, fields) {
  if (fields.label !== undefined && !String(fields.label).trim() && u.draft) return toast('Nhập tên vật trước khi lưu.', true);
  try {
    const r = u.draft
      ? await api('/api/manual', { action: 'add', image_id: D.image_id, box: u.box, kind: u.manualKind, ...fields })
      : await api('/api/manual', { action: 'update', image_id: D.image_id, id: u.key, ...fields });
    DRAFT = null;
    SEL = `manual|${r.id}`;
    const l = fields.label && u.manualKind === 'object' ? String(fields.label).trim() : '';
    if (l && META.manual && !META.manual.objectLabels.includes(l)) META.manual.objectLabels.push(l);
    $('undo').disabled = false;
    await loadImage(S.id, true);
    toast(u.draft ? 'Đã lưu box mới.' : 'Đã lưu.');
  } catch (e) { fail(e); }
}
async function deleteManual(u) {
  try { await api('/api/manual', { action: 'delete', image_id: D.image_id, id: u.key }); } catch (e) { return fail(e); }
  SEL = ''; $('undo').disabled = false;
  await loadImage(S.id, true);
  toast('Đã xoá box. Bấm Z để hoàn tác.');
}
function cancelDraft() { DRAFT = null; select(''); }
function setDraw(on) {
  DRAW = on;
  $('frame').classList.toggle('drawing', on);
  $('drawBtn').classList.toggle('on', on);
  $('drawBtn').setAttribute('aria-pressed', on);
}
function wireDrawing() {
  const svg = $('boxes');
  const pt = svgPt;
  let start = null, rubber = null;
  svg.addEventListener('pointerdown', (e) => {
    suppressClick = false;
    if (!DRAW || e.button !== 0 || !D) return;
    e.preventDefault();
    svg.setPointerCapture(e.pointerId);
    start = pt(e);
    rubber = sv('rect', { class: 'rubber', x: start[0], y: start[1], width: 0, height: 0 });
    svg.append(rubber);
  });
  svg.addEventListener('pointermove', (e) => {
    if (EDIT) {
      const [x, y] = pt(e);
      EDIT.box = draggedBox(EDIT.orig, EDIT.mode, x - EDIT.start[0], y - EDIT.start[1]);
      return drawBoxes();
    }
    if (!start) return;
    const [x, y] = pt(e);
    rubber.setAttribute('x', Math.min(x, start[0])); rubber.setAttribute('y', Math.min(y, start[1]));
    rubber.setAttribute('width', Math.abs(x - start[0])); rubber.setAttribute('height', Math.abs(y - start[1]));
  });
  const end = (e) => {
    if (EDIT) return finishEdit();
    if (!start) return;
    const [x, y] = pt(e);
    const box = [Math.min(x, start[0]), Math.min(y, start[1]), Math.max(x, start[0]), Math.max(y, start[1])];
    start = null; rubber.remove(); rubber = null;
    suppressClick = true; // click sinh ra ngay sau khi nhả chuột không được bỏ chọn box mới
    if (box[2] - box[0] < 8 || box[3] - box[1] < 8) return toast('Box quá nhỏ, kéo rộng hơn.', true);
    DRAFT = { box: box.map((v) => Math.round(v * 10) / 10), kind: LASTKIND };
    setDraw(false);
    classFilter = '';
    select('manual|draft');
  };
  svg.addEventListener('pointerup', end);
  svg.addEventListener('pointercancel', () => { if (rubber) rubber.remove(); start = rubber = null; if (EDIT) { EDIT = null; drawBoxes(); } });
}
function norm(s) { return String(s || '').normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/đ/gi, 'd').toLowerCase(); }

function select(id) {
  if (SEL !== id) classFilter = '';
  if (DRAFT && id !== 'manual|draft') DRAFT = null; // bỏ box vẽ dở khi chọn box khác
  SEL = id;
  removeTag('hover');
  drawBoxes(); renderPanel(); renderTop();
}

// ---------------------------------------------------------------- sửa nhãn + tự sang box kế tiếp
async function act(action, label, text) {
  const u = current();
  if (!u || u.kind === 'manual') return;
  if ((action === 'confirm' || action === 'reject') && !u.state) return;
  if (u.inherited && !confirm(`Nhãn này dùng chung cho ${u.affected} ảnh giờ ${D.hour} (vùng không đổi so với nền). Vẫn sửa?`)) return;
  try {
    await api('/api/label', { ops: [{ profile: u.profile, key: u.key, action, label, text }] });
  } catch (e) {
    if (e.status !== 409) return fail(e);
    // file nhãn bị chương trình khác ghi đè: nạp lại để không đè mất thay đổi của họ
    await api('/api/reload', {}).catch(fail);
    await loadImage(S.id, true);
    return toast('File nhãn vừa bị chương trình khác ghi. Đã nạp lại; kiểm tra rồi thao tác lại.', true);
  }
  const before = SEL;
  await loadImage(S.id, true);
  if (action === 'confirm' || action === 'set' || action === 'custom') advance(before);
  else select(before);
  META.undo = (META.undo || 0) + 1; $('undo').disabled = false;
}
function advance(fromId) {
  const vis = visible().filter((u) => u.kind !== 'image');
  const order = sortReading(vis);
  const i = order.findIndex((u) => uid(u) === fromId);
  const rest = [...order.slice(i + 1), ...order.slice(0, Math.max(i, 0))];
  const next = rest.find((u) => statusOf(u.state) !== 'human');
  if (next) return select(uid(next));
  select(fromId);
  toast('Ảnh này đã duyệt hết. Bấm → để sang ảnh tiếp.');
}
// thứ tự đọc: trái sang phải, trên xuống dưới
const sortReading = (list) => [...list].sort((a, b) => (a.kind === b.kind ? a.box[0] - b.box[0] || a.box[1] - b.box[1] : a.kind === 'person' ? -1 : 1));
function cycle(d) {
  const order = sortReading(visible());
  if (!order.length) return;
  const i = order.findIndex((u) => uid(u) === SEL);
  select(uid(order[(i + d + order.length) % order.length]));
}
async function undo() {
  try { const r = await api('/api/undo', {}); toast(`Đã hoàn tác. Còn ${r.left} bước.`); $('undo').disabled = !r.left; await loadImage(S.id, true); } catch (e) { fail(e); }
}

// gộp nhãn đã duyệt vào enriched/ + exports/ (python main.py enrich merge)
async function mergeData() {
  if (!confirm('Gộp mọi nhãn đã duyệt vào enriched/ và xuất lại exports/? Mất khoảng 15 giây. Sau khi gộp không hoàn tác được các bước trước.')) return;
  const b = $('mergeBtn');
  b.disabled = true; b.querySelector('span').textContent = 'Đang gộp…';
  try {
    const r = await api('/api/merge', {});
    META = await api('/api/meta');
    PROF = Object.fromEntries(META.profiles.map((p) => [p.name, { ...p, roiVi: META.rois.find((x) => x.name === p.roi)?.name_vi }]));
    $('undo').disabled = true;
    await loadQueue();
    await loadImage(S.id, true);
    toast(`Đã gộp ${r.images} ảnh vào exports/.`);
  } catch (e) { fail(e); }
  b.disabled = false; b.querySelector('span').textContent = 'Gộp dữ liệu';
}

// ---------------------------------------------------------------- top bar + điều hướng
function renderTop() {
  $('imgId').textContent = D ? D.image_id : '';
  const i = QUEUE.indexOf(S.id);
  $('pos').textContent = QUEUE.length ? `${i >= 0 ? i + 1 : '–'} / ${QUEUE.length}` : '0 ảnh khớp';
  $('prev').disabled = i <= 0; $('next').disabled = i < 0 || i >= QUEUE.length - 1;
  for (const b of document.querySelectorAll('.seg button[data-show]')) b.classList.toggle('on', b.dataset.show === S.show);
  $('hideDone').checked = S.hideDone;
  const n = activeFilters();
  $('filterCount').hidden = !n; $('filterCount').textContent = n;
  const img = units().find((u) => u.kind === 'image');
  $('imageLabel').hidden = !img;
  if (img) {
    $('imageLabel').textContent = `Cả ảnh: ${img.state ? img.state.label_vi || img.state.label : 'chưa có nhãn'}`;
    $('imageLabel').classList.toggle('on', uid(img) === SEL);
  }
  const all = units().filter((u) => u.kind !== 'image' && u.kind !== 'manual' && (S.show === 'all' || (S.show === 'people') === (u.kind === 'person')));
  const done = all.filter((u) => statusOf(u.state) === 'human').length;
  $('progress').textContent = D ? `${done}/${all.length} ${S.show === 'roi' ? 'vùng' : S.show === 'people' ? 'người' : 'box'} đã duyệt` : '';
}
async function loadImage(id, keepSel) {
  if (!id) { D = null; $('photo').removeAttribute('src'); drawBoxes(); renderPanel(); renderTop(); return; }
  try { D = await api(`/api/image/${id}`); } catch (e) { return fail(e); }
  S.id = id; toHash();
  const img = $('photo');
  if (img.dataset.id !== id) {
    img.dataset.id = id;
    img.src = `/img/${id}.jpg`;
    await img.decode().catch(() => {});
    if (img.naturalWidth) NAT = { w: img.naturalWidth, h: img.naturalHeight };
  }
  if (!keepSel) {
    SEL = ''; DRAFT = null;
    // có lọc theo nhãn: chọn sẵn box khớp đầu tiên
    const m = sortReading(visible()).find(isMatch);
    if (m) SEL = uid(m);
  } else if (!units().some((u) => uid(u) === SEL)) SEL = '';
  fit(); drawBoxes(); renderPanel(); renderTop();
}
function step(d) {
  const i = QUEUE.indexOf(S.id);
  const next = QUEUE[i < 0 ? 0 : i + d];
  if (next) loadImage(next);
}
async function loadQueue() {
  const qs = new URLSearchParams({ ...queueParams(), size: 1 });
  const r = await api(`/api/images?${qs}`);
  QUEUE = r.ids;
  return r.total;
}

// ---------------------------------------------------------------- bộ lọc (ngăn kéo)
let drawerTab = 'people';
function openDrawer() { $('drawer').hidden = false; renderDrawer(); setTimeout(() => $('drawerBody').querySelector('input')?.focus(), 30); }
function closeDrawer() { $('drawer').hidden = true; }
async function applyFilters() {
  toHash();
  const total = await loadQueue().catch(fail);
  $('drawerBody').querySelector('.total') && ($('drawerBody').querySelector('.total').textContent = `${total} ảnh trong hàng đợi`);
  drawBoxes(); renderTop();
}
let qT;
function renderDrawer() {
  const profsOf = (tab) => META.profiles.filter((p) => (tab === 'roi' ? p.layer === 'roi' : p.name === tab)).map((p) => p.name);
  const chip = (on, text, fn) => h('button', { class: 'chip' + (on ? ' on' : ''), 'aria-pressed': on, onclick: async () => { fn(); renderDrawer(); await applyFilters(); } }, text);
  const preset = [
    chip(!activeFilters(), 'Tất cả ảnh', () => Object.assign(S, { tokens: [], q: '', src: [], hours: [], edited: false, manual: false, conflict: false })),
    chip(S.src.join() === 'vlm,seed,suggested', 'Còn nhãn máy chưa duyệt', () => { S.src = ['vlm', 'seed', 'suggested']; }),
    chip(S.tokens.length === 1 && S.tokens[0].label === NONE && S.tokens[0].profile === 'people', 'Người chưa có nhãn', () => { S.tokens = [{ profile: 'people', label: NONE }]; }),
    chip(S.edited, 'Đã sửa trong phiên', () => { S.edited = !S.edited; }),
    chip(S.conflict, 'Hành động lệch vùng', () => { S.conflict = !S.conflict; }),
    chip(S.manual, `Có box tự vẽ (${META.manual?.boxes ?? 0})`, () => { S.manual = !S.manual; }),
  ];
  const labelRows = [];
  for (const name of profsOf(drawerTab)) {
    const p = PROF[name];
    if (drawerTab === 'roi') labelRows.push(h('h3', { style: 'margin-top:12px' }, p.roiVi || p.roi));
    const entries = [...p.classes.map((c) => [c.name, c.name_vi]), [NONE, 'Chưa có nhãn']];
    for (const [label, vi] of entries) {
      const n = label === NONE ? p.stats.none : p.stats.byLabel[label] || 0;
      if (!n && label !== NONE) continue;
      const on = S.tokens.some((t) => t.profile === name && t.label === label);
      labelRows.push(h('button', { class: 'lrow' + (on ? ' on' : ''), 'aria-pressed': on, onclick: async () => {
        S.tokens = on ? S.tokens.filter((t) => !(t.profile === name && t.label === label)) : [...S.tokens, { profile: name, label }];
        if (name.startsWith('roi') && S.show === 'people') S.show = 'all';
        renderDrawer(); await applyFilters();
      } }, h('span', {}, vi || label), h('span', { class: 'n' }, n)));
    }
  }
  const search = h('input', { type: 'search', value: S.q, placeholder: 'Keyword không dấu: mo tu lanh, 09_…', 'aria-label': 'Keyword',
    oninput: (e) => { clearTimeout(qT); qT = setTimeout(() => { S.q = e.target.value; applyFilters(); }, 250); } });
  $('drawerBody').replaceChildren(
    h('div', { class: 'sec' }, h('h3', {}, 'Nhanh'), h('div', { class: 'chips' }, preset)),
    h('div', { class: 'sec' }, h('h3', {}, 'Từ khoá'), search),
    h('div', { class: 'sec' }, h('h3', {}, 'Theo nhãn'),
      h('div', { class: 'seg', style: 'margin-bottom:8px' }, [['people', 'Người'], ['roi', 'Vùng'], ['image', 'Cả ảnh']].filter(([k]) => profsOf(k).length).map(([k, t]) =>
        h('button', { class: drawerTab === k ? 'on' : '', onclick: () => { drawerTab = k; renderDrawer(); } }, t))),
      S.tokens.length > 1 ? h('div', { class: 'chips', style: 'margin-bottom:8px' },
        chip(S.mode === 'or', 'Có bất kỳ', () => { S.mode = 'or'; }), chip(S.mode === 'and', 'Có tất cả', () => { S.mode = 'and'; })) : null,
      labelRows),
    h('div', { class: 'sec' }, h('h3', {}, 'Giờ'), h('div', { class: 'chips' }, META.hours.map((hr) =>
      chip(S.hours.includes(hr), `${hr}h`, () => { S.hours = S.hours.includes(hr) ? S.hours.filter((x) => x !== hr) : [...S.hours, hr]; })))),
    h('div', { class: 'total' }, `${QUEUE.length} ảnh trong hàng đợi`));
  const foot = $('drawer').querySelector('.drawer-foot') || $('drawer').querySelector('.drawer-in').appendChild(h('div', { class: 'drawer-foot' }));
  foot.replaceChildren(
    h('button', { class: 'btn primary', onclick: () => { closeDrawer(); if (QUEUE.length && !QUEUE.includes(S.id)) loadImage(QUEUE[0]); else loadImage(S.id); } }, 'Bắt đầu duyệt'),
    h('span', { class: 'grow' }),
    h('a', { href: `/api/export?${new URLSearchParams({ ...queueParams(), format: 'ids' })}` }, 'Tải ids.txt'));
}

// ---------------------------------------------------------------- phím tắt
function onKey(e) {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName);
  if (e.key === 'Escape') {
    if (!$('helpBox').hidden) return ($('helpBox').hidden = true);
    if (!$('drawer').hidden) return closeDrawer();
    if (typing) return e.target.blur();
    if (DRAW) return setDraw(false);
    if (DRAFT) return cancelDraft();
    if (SEL) return select('');
  }
  if (typing || !$('drawer').hidden || !$('helpBox').hidden) return;
  const k = e.key.toLowerCase();
  if (e.key === 'ArrowLeft') return step(-1);
  if (e.key === 'ArrowRight') return step(1);
  if (e.key === 'Tab') { e.preventDefault(); return cycle(e.shiftKey ? -1 : 1); }
  if (e.key === '/') { e.preventDefault(); return openDrawer(); }
  if (e.key === '?') return ($('helpBox').hidden = false);
  if (k === 'z') return undo();
  if (k === 'b') return setDraw(!DRAW);
  if (k === 'l') { S.labels = !S.labels; toHash(); return drawBoxes(); }
  if (k === 'v') { S.objects = !S.objects; $('showObjects').checked = S.objects; toHash(); return drawBoxes(); }
  if (k === 'n' || k === 'h') { PTAB = k === 'n' ? 'who' : 'action'; return renderPanel(); }
  if (k === 'i') { const img = units().find((u) => u.kind === 'image'); if (img) select(uid(img)); return; }
  if (!SEL) return;
  const cu = current();
  if (cu?.kind === 'manual') {
    if ((e.key === 'Delete' || e.key === 'Backspace') && !cu.draft) return deleteManual(cu);
    return;
  }
  if (k === 'o') { e.preventDefault(); $('otherText')?.focus(); return; }
  if (k === 'a' || e.key === 'Enter') return act('confirm');
  if (k === 'x') return act('reject');
  if (k === 'r') return act('reset');
}

// ---------------------------------------------------------------- khởi động
(async function init() {
  fromHash();
  try {
    META = await api('/api/meta');
    PROF = Object.fromEntries(META.profiles.map((p) => [p.name, { ...p, roiVi: META.rois.find((r) => r.name === p.roi)?.name_vi }]));
    await loadQueue();
  } catch (e) { return fail(e); }
  $('undo').disabled = !META.undo;
  for (const b of document.querySelectorAll('.seg button[data-show]')) b.onclick = () => { S.show = b.dataset.show; if (SEL && !visible().some((u) => uid(u) === SEL) && current()?.kind !== 'image') SEL = ''; toHash(); drawBoxes(); renderPanel(); renderTop(); };
  $('hideDone').onchange = () => { S.hideDone = $('hideDone').checked; toHash(); drawBoxes(); };
  $('showObjects').checked = S.objects;
  $('showObjects').onchange = () => { S.objects = $('showObjects').checked; toHash(); drawBoxes(); };
  $('prev').onclick = () => step(-1);
  $('next').onclick = () => step(1);
  $('undo').onclick = undo;
  $('mergeBtn').onclick = mergeData;
  $('help').onclick = () => ($('helpBox').hidden = false);
  $('helpBox').onclick = (e) => { if (e.target === $('helpBox')) $('helpBox').hidden = true; };
  $('openDrawer').onclick = openDrawer;
  $('closeDrawer').onclick = closeDrawer;
  $('drawer').onclick = (e) => { if (e.target === $('drawer')) closeDrawer(); };
  $('imageLabel').onclick = () => { const img = units().find((u) => u.kind === 'image'); if (img) select(SEL === uid(img) ? '' : uid(img)); };
  $('stage').onclick = (e) => { if (suppressClick) { suppressClick = false; return; } if (DRAW) return; if (e.target === $('stage') || e.target === $('photo') || e.target === $('boxes')) select(''); };
  document.addEventListener('keydown', onKey);
  $('drawBtn').onclick = () => setDraw(!DRAW);
  wireDrawing();
  window.addEventListener('resize', fit);
  window.addEventListener('hashchange', async () => { const old = S.id; fromHash(); await loadQueue().catch(fail); if (S.id !== old) loadImage(S.id); else { drawBoxes(); renderTop(); } });
  await loadImage(S.id || QUEUE[0] || '');
})();

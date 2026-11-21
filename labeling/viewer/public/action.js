// Gán nhanh hành động, một người một lần. Hàng đợi:
//   Duyệt: <hành động>: người máy gán hành động đó (chưa ai duyệt). Z = đúng, X = sai (vào "Cần gán lại").
//   Cần gán lại: người đã bị bác một nhãn máy, chưa có nhãn người duyệt.  Lệch vùng: nhãn máy mâu thuẫn với vùng đứng.
//   Thầy tả: <hành động>: câu mô tả của thầy tả người này làm hành động đó; Z = nhận, X = sai.
//   Nghi sai: `enrich actions` (gộp VLM, SigLIP, vùng, câu mô tả của thầy) chọn lớp khác nhãn máy; Z = nhận đề xuất, X = bác nhãn máy.
// Phím số / chữ cố định theo hành động (KEYS); Space / →: bỏ qua; ←: người trước; Backspace: hoàn tác.
'use strict';
const KEYS = [   // [phím, lớp]: lớp hay gặp lên phím số
  ['1', 'standing idle'], ['2', 'walking'], ['3', 'preparing food'], ['4', 'cooking at the stove'], ['5', 'washing dishes'],
  ['6', 'opening the refrigerator'], ['7', 'carrying something'], ['8', 'talking'], ['9', 'making a hot drink'], ['0', 'getting items from a cabinet'],
  ['q', 'eating'], ['w', 'drinking'], ['e', 'using a phone'], ['r', 'using a laptop'], ['t', 'sitting'], ['y', 'cleaning'], ['u', 'throwing away trash'],
];
let VI = {}, QUEUE = [], I = 0, BUSY = false, MODE = '', LEFT = null;
const done = [];
const msg = (t, err) => { $('msg').textContent = t; $('msg').style.color = err ? '#c0392b' : ''; };
const vi = (label) => VI[label] || label || '';

function renderKeys(x) {
  const row = (key, label, cls, act) => {
    const b = document.createElement('button');
    b.className = `q-key ${cls || ''}`; b.dataset.key = key; b.onclick = act;
    b.innerHTML = '<kbd></kbd><span></span>'; b.firstChild.textContent = key; b.lastChild.textContent = label;
    return b;
  };
  $('keys').replaceChildren(
    row('Z', x?.label ? `Đúng: ${vi(x.label)}` : 'Đúng (chưa có nhãn)', 'ok', accept),
    row('X', MODE === 'recheck' ? 'Bỏ qua' : 'Sai: để gán lại sau', 'danger', reject),
    ...KEYS.map(([k, l]) => row(k.toUpperCase(), vi(l), l === x?.label ? 'guess' : '', () => setLabel(l, k.toUpperCase()))),
    row('␣', 'Bỏ qua (Space / →)', '', () => go(1)),
    row('⌫', 'Hoàn tác (Backspace)', '', undo));
}

const showCount = () => { const x = QUEUE[I]; $('count').textContent = x ? `${I + 1} / ${QUEUE.length}${LEFT != null ? ` · còn ${LEFT}` : ''}${x.h != null ? ` · cao ${x.h}px` : ''}` : (LEFT != null ? `còn ${LEFT}` : ''); };
function show() {
  const x = QUEUE[I];
  renderKeys(x);
  if (!x) { $('pid').textContent = QUEUE.length ? 'Hết hàng đợi' : 'Hàng đợi trống'; showCount(); $('card').replaceChildren(); showResult(null); return; }
  $('pid').textContent = x.id;
  showCount();
  renderCard(x);
  showResult(x);
  drawCrop($('view'), x, () => QUEUE[I] === x);
}

// Thẻ so sánh: nhãn máy (gạch) -> nhãn cần duyệt (chữ lớn, nền vàng), bên cạnh là vùng đứng và nhãn đã bác.
function renderCard(x) {
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const box = (cls, cap, label, hint) => {
    const b = el('div', `a-box ${cls}`);
    b.append(el('small', '', cap), el('b', '', label));
    if (hint) { const h = el('span', 'a-hint'); h.innerHTML = hint; b.append(h); }
    return b;
  };
  const parts = [];
  if (MODE === 'suspect') parts.push(box('from', 'Máy đang gán', vi(x.machine) || '—'), el('span', 'a-arrow', '→'), box('to', 'Đề xuất', vi(x.label), '<kbd>Z</kbd> nhận · <kbd>X</kbd> bác nhãn máy'));
  else if (MODE.startsWith('teacher:')) parts.push(box('to', 'Thầy tả: có phải', vi(x.label), '<kbd>Z</kbd> đúng · <kbd>X</kbd> sai'));
  else if (x.label) parts.push(box('to', 'Có phải', vi(x.label), MODE === 'recheck' ? '<kbd>Z</kbd> đúng · phím số để gán khác' : '<kbd>Z</kbd> đúng · <kbd>X</kbd> sai'));
  else parts.push(box('none', 'Chưa có nhãn', 'Chọn bằng phím số'));
  const side = el('div', 'a-side');
  if (x.zone_vi) { const c = el('span', 'a-chip'); c.append(el('span', '', 'Vùng'), x.zone_vi); side.append(c); }
  if (x.rejected?.length) { const c = el('span', 'a-chip bad'); c.append(el('span', '', 'Đã bác'), x.rejected.map(vi).join(', ')); side.append(c); }
  if (side.childElementCount) parts.push(side);
  if (x.teacher) parts.push(el('div', 'a-quote', `Thầy: “${x.teacher}”`));
  $('card').replaceChildren(...parts);
}
const go = (d) => { I = Math.max(0, Math.min(QUEUE.length, I + d)); msg(''); show(); };

function setCounts(sum) {
  for (const o of $('mode').options) {
    const l = o.value.slice(6);
    const t = o.value.startsWith('teacher:') ? o.value.slice(8) : null;
    if (t) { const n = sum.teacher[t] || 0; o.textContent = `Thầy tả: ${vi(t)} (${n})`; if (o.value === MODE) LEFT = n; continue; }
    const n = o.value === 'suspect' ? sum.suspect : o.value === 'recheck' ? sum.recheck : o.value === 'conflict' ? sum.conflict : sum.pending[l] || 0;
    o.textContent = o.value === 'suspect' ? `Nghi sai: máy gộp đề xuất khác (${n})` : o.value === 'recheck' ? `Cần gán lại (${n})` : o.value === 'conflict' ? `Lệch vùng (${n})` : `Duyệt: ${vi(l)} (${n}) · đã duyệt ${sum.human[l] || 0}`;
    if (o.value === MODE) LEFT = n;
  }
  showCount();
}
const refreshCounts = () => api('/api/action/summary').then(setCounts).catch(() => {});

async function send(op, key, text, result) {
  const x = QUEUE[I];
  if (!x || BUSY) return;
  BUSY = true;
  try {
    await api('/api/label', { ops: [{ profile: 'people', key: x.id, ...op }] });
    document.querySelector(`.q-key[data-key="${CSS.escape(key)}"]`)?.classList.add('flash');
    x.result = { ...result, key };
    done.push(I);
    msg(text);
    refreshCounts();
    setTimeout(() => go(1), 100);
  } catch (e) { msg(e.message, true); } finally { BUSY = false; }
}
const setLabel = (label, key) => send({ action: 'set', label }, key, `Đã gán "${vi(label)}".`, { kind: label === QUEUE[I]?.label ? 'ok' : 'set', text: vi(label) });
function accept() {
  const x = QUEUE[I];
  if (!x?.label) return msg('Chưa có nhãn để xác nhận: dùng phím số.', true);
  send({ action: 'set', label: x.label }, 'Z', `Đúng: "${vi(x.label)}".`, { kind: 'ok', text: vi(x.label) });   // set, không confirm: nhãn đang hiện có thể là nhãn gộp
}
function reject() {
  const x = QUEUE[I];
  if (MODE === 'recheck' || !x?.label) return go(1);
  send({ action: 'reject', label: MODE === 'suspect' ? x.machine : x.label, only: MODE.startsWith('teacher:') }, 'X', 'Sai: đã đưa vào "Cần gán lại".', { kind: 'bad', text: `${vi(MODE === 'suspect' ? x.machine : x.label)} (vào "Cần gán lại")` });
}
async function undo() {
  if (BUSY || !done.length) return msg('Không còn gì để hoàn tác.', true);
  BUSY = true;
  try { await api('/api/undo', {}); I = done.pop(); delete QUEUE[I]?.result; msg('Đã hoàn tác.'); show(); refreshCounts(); } catch (e) { msg(e.message, true); } finally { BUSY = false; }
}

async function load() {
  const sum = await api('/api/action/summary');
  VI = Object.fromEntries(sum.classes.map((c) => [c.name, c.name_vi]));
  if (!$('mode').options.length) {   // lần đầu: hàng theo từng lớp (nhiều nhãn máy trước), rồi hai hàng đặc biệt
    for (const c of [...sum.classes].sort((a, b) => (sum.pending[b.name] || 0) - (sum.pending[a.name] || 0))) $('mode').append(new Option(c.name, `label:${c.name}`));
    // hàng "Thầy tả" mỗi lớp có gợi ý, mở tủ lạnh lên đầu
    const ts = Object.keys(sum.teacher).sort((a, b) => (b === 'opening the refrigerator') - (a === 'opening the refrigerator') || sum.teacher[b] - sum.teacher[a]);
    for (const t of ts.reverse()) $('mode').prepend(new Option(t, `teacher:${t}`));
    $('mode').prepend(new Option('suspect', 'suspect'));
    $('mode').value = 'suspect';
    $('mode').append(new Option('recheck', 'recheck'), new Option('conflict', 'conflict'));
  }
  MODE = $('mode').value;
  setCounts(sum);
  const q = MODE.startsWith('teacher:') ? `teacher=${encodeURIComponent(MODE.slice(8))}` : MODE === 'suspect' ? 'suspect=1' : MODE === 'recheck' ? 'recheck=1' : MODE === 'conflict' ? 'conflict=1' : `label=${encodeURIComponent(MODE.slice(6))}`;
  const items = (await api(`/api/action/members?${q}`)).items;
  const big = $('bigOnly').checked;
  QUEUE = items.filter((x) => x.box && (!big || (x.h ?? 0) >= 160));
  if ($('byPlace').checked) QUEUE = byPlace(QUEUE);
  I = 0; done.length = 0;
  show();
}

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || ['INPUT', 'SELECT'].includes(e.target.tagName)) return;
  const k = e.key.toLowerCase();
  const hit = KEYS.find(([key]) => key === k);
  if (hit) setLabel(hit[1], hit[0].toUpperCase());
  else if (k === 'z') accept();
  else if (k === 'x') reject();
  else if (k === ' ' || k === 'arrowright') go(1);
  else if (k === 'arrowleft') go(-1);
  else if (k === 'backspace') undo();
  else return;
  e.preventDefault();
});
for (const id of ['mode', 'bigOnly', 'byPlace']) $(id).onchange = () => { $(id).blur(); load().catch((e) => msg(e.message, true)); };
load().catch((e) => msg(e.message, true));

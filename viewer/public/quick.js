// Gán nhanh danh tính, một người một lần. Ba hàng đợi:
//   Duyệt theo tên: crop máy đã gán tên đó (chưa ai xác nhận), ít chắc nhất trước. Z = đúng, X = sai (vào "Cần gán lại").
//   Khó trước: crop chưa gán, mô hình phân vân nhất. Z = nhận tên mô hình đoán.
//   Cần gán lại: crop đã bấm X, gán bằng phím số.
// Phím số cố định theo người (FIXED); tên còn lại lấy chữ q, w, e, ... K: không rõ; Delete: không phải người;
// Space / →: bỏ qua; ←: người trước; Backspace: hoàn tác.
'use strict';
const FIXED = ['Linh', 'Áo len 2 màu', 'Nữ áo trắng', 'Caro đỏ xanh', 'caro xanh trắng', 'áo khoác xám cao', 'Sơ mi xanh', 'Sơ mi xám trơn (hơi xanh xẫm)', 'Nư áo đen'];
const LETTERS = 'qwertyuiop';
let KEYMAP = [], QUEUE = [], I = 0, BUSY = false, MODE = 'hard', LEFT = null;
const done = [];   // vị trí đã xử lý, để Backspace quay lại đúng người

const msg = (t, err) => { $('msg').textContent = t; $('msg').style.color = err ? '#c0392b' : ''; };
const shown = (x) => (MODE === 'extra' ? 'Box thừa (không phải người)' : MODE.startsWith('name:') ? MODE.slice(5) : x?.guess) || null;   // tên đang hỏi đúng/sai

function renderKeys(x) {
  const row = (key, label, cls, act) => {
    const b = document.createElement('button');
    b.className = `q-key ${cls || ''}`; b.dataset.key = key; b.onclick = act;
    b.innerHTML = '<kbd></kbd><span></span>'; b.firstChild.textContent = key; b.lastChild.textContent = label;
    return b;
  };
  const cur = shown(x);
  $('keys').replaceChildren(
    row('Z', cur ? `Đúng: ${cur}` : 'Đúng (không có tên đoán)', 'ok', () => accept()),
    row('X', MODE === 'extra' ? 'Không: là người (bỏ qua)' : MODE === 'recheck' ? 'Bỏ qua' : 'Sai: để gán lại sau', 'danger', () => reject()),
    ...KEYMAP.map(([k, n]) => row(k.toUpperCase(), n, n === cur ? 'guess' : '', () => assign({ name: n }, k.toUpperCase()))),
    row('K', 'Không rõ là ai', '', () => assign({ name: '' }, 'K')),
    row('Del', 'Không phải người (xoá box)', 'danger', () => assign({ notPerson: true }, 'Del')),
    row('␣', 'Bỏ qua (Space / →)', '', () => go(1)),
    row('⌫', 'Hoàn tác (Backspace)', '', undo));
}

function show() {
  const x = QUEUE[I];
  renderKeys(x);
  if (!x) { $('pid').textContent = QUEUE.length ? 'Hết hàng đợi' : 'Hàng đợi trống'; showCount(); $('ask').textContent = ''; showResult(null); return; }
  $('pid').textContent = x.id;
  showCount();
  const cur = shown(x);
  $('ask').textContent = MODE === 'extra' ? `Box thừa? Mình đếm ${x.extra?.ours} người, thầy đếm ${x.extra?.teacher}; điểm dò ${x.extra?.score}` : MODE === 'recheck' ? `Đã bác: ${x.rejected || '?'}${x.guess ? ` · mô hình đoán: ${x.guess}` : ''}`
    : cur ? `Có phải "${cur}"?${x.prob != null ? ` (${Math.round(x.prob * 100)}%)` : ''}` : 'Mô hình chưa đoán';
  showResult(x);
  drawCrop($('view'), x, () => QUEUE[I] === x);
}
// số còn lại của mọi hàng đợi, lấy lại sau mỗi lần gán / hoàn tác
function setCounts(sum) {
  for (const o of $('mode').options) {
    const n = o.value === 'hard' ? sum.hard : o.value === 'extra' ? sum.extra : o.value === 'recheck' ? sum.recheck : sum.pending?.[o.value.slice(5)] || 0;
    o.textContent = `${o.value === 'hard' ? 'Khó trước' : o.value === 'extra' ? 'Có thể box thừa' : o.value === 'recheck' ? 'Cần gán lại' : `Duyệt: ${o.value.slice(5)}`} (${n})`;
    if (o.value === MODE) LEFT = n;
  }
  showCount();
}
const refreshCounts = () => api('/api/identity/summary').then(setCounts).catch(() => {});
const showCount = () => { const x = QUEUE[I]; $('count').textContent = x ? `${I + 1} / ${QUEUE.length}${LEFT != null ? ` · còn ${LEFT}` : ''}${x.h != null ? ` · cao ${x.h}px` : ''}` : (LEFT != null ? `còn ${LEFT}` : ''); };
const go = (d) => { I = Math.max(0, Math.min(QUEUE.length, I + d)); msg(''); show(); };

async function send(body, key, text, result) {
  const x = QUEUE[I];
  if (!x || BUSY) return;
  BUSY = true;
  try {
    await api('/api/identity', { scope: 'person', persons: [x.id], ...body });
    document.querySelector(`.q-key[data-key="${CSS.escape(key)}"]`)?.classList.add('flash');
    x.result = { ...result, key };
    done.push(I);
    msg(text);
    refreshCounts();
    setTimeout(() => go(1), 100);
  } catch (e) { msg(e.message, true); } finally { BUSY = false; }
}
const assign = (body, key, kind) => send(body, key, body.notPerson ? 'Đã đánh dấu không phải người.' : body.name ? `Đã gán "${body.name}".` : 'Đã đánh dấu không rõ.',
  { kind: kind || (body.name && body.name === shown(QUEUE[I]) ? 'ok' : 'set'), text: body.notPerson ? 'không phải người' : body.name || 'không rõ là ai' });
function accept() {
  if (MODE === 'extra') return assign({ notPerson: true }, 'Z', 'ok');   // đúng là box thừa
  const cur = shown(QUEUE[I]);
  if (!cur) return msg('Không có tên để xác nhận: dùng phím số.', true);
  send({ name: cur }, 'Z', `Đúng: "${cur}".`, { kind: 'ok', text: cur });
}
function reject() {
  if (MODE === 'recheck' || MODE === 'extra') return go(1);   // box thừa? không: là người, để đó (gán tên bằng phím số nếu muốn)
  const cur = shown(QUEUE[I]);
  send({ recheck: cur || '?' }, 'X', 'Sai: đã đưa vào "Cần gán lại".', { kind: 'bad', text: `${cur || '?'} (vào "Cần gán lại")` });
}
async function undo() {
  if (BUSY || !done.length) return msg('Không còn gì để hoàn tác.', true);
  BUSY = true;
  try { await api('/api/undo', {}); I = done.pop(); delete QUEUE[I]?.result; msg('Đã hoàn tác.'); show(); refreshCounts(); } catch (e) { msg(e.message, true); } finally { BUSY = false; }
}

async function load() {
  MODE = $('mode').value || 'hard';
  const sum = await api('/api/identity/summary');
  const all = Object.entries(sum.names).sort((a, b) => b[1] - a[1]).map(([n]) => n);
  const rest = all.filter((n) => !FIXED.includes(n));
  KEYMAP = [...FIXED.map((n, i) => [String(i + 1), n]), ...rest.slice(0, LETTERS.length).map((n, i) => [LETTERS[i], n])];   // số cố định theo người, không đổi khi số crop đổi
  if ($('mode').options.length <= 2) for (const n of all) $('mode').append(new Option(n, `name:${n}`));   // lần đầu: một mục "Duyệt: <tên>" mỗi tên
  setCounts(sum);
  const q = MODE === 'hard' ? 'hard=1' : MODE === 'extra' ? 'extra=1' : MODE === 'recheck' ? 'recheck=1' : `name=${encodeURIComponent(MODE.slice(5))}`;
  let items = (await api(`/api/identity/members?${q}`)).items;
  if (MODE.startsWith('name:')) items = items.filter((x) => x.source === 'model').sort((a, b) => (a.prob ?? 0) - (b.prob ?? 0));   // chỉ crop máy gán, ít chắc nhất trước
  const big = $('bigOnly').checked;
  QUEUE = items.filter((x) => x.box && (!big || (x.h ?? 0) >= 160));
  if ($('byPlace').checked) QUEUE = byPlace(QUEUE);
  I = 0; done.length = 0;
  show();
}

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || ['INPUT', 'SELECT'].includes(e.target.tagName)) return;
  const k = e.key.toLowerCase();
  const hit = KEYMAP.find(([key]) => key === k);
  if (hit) assign({ name: hit[1] }, hit[0].toUpperCase());
  else if (k === 'z') accept();
  else if (k === 'x') reject();
  else if (k === 'k') assign({ name: '' }, 'K');
  else if (k === 'delete') assign({ notPerson: true }, 'Del');
  else if (k === ' ' || k === 'arrowright') go(1);
  else if (k === 'arrowleft') go(-1);
  else if (k === 'backspace') undo();
  else return;
  e.preventDefault();
});
for (const id of ['mode', 'bigOnly', 'byPlace']) $(id).onchange = () => { $(id).blur(); load().catch((e) => msg(e.message, true)); };
load().catch((e) => msg(e.message, true));

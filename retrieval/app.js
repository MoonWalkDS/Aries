// Trang tìm ảnh Castle5000 + nộp DRES. Setting cá nhân: settings.js. Thiết kế: docs/search-settings-design.md
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const S = window.Settings;
S.load();

const st = {
  results: [], page: 0, why: {}, current: null,      // current = id ảnh đang mở trong dialog
  task: null, levels: [], level: null,                // level đang dùng (index) hoặc null khi gõ tay
  seen: null, pending: null,                          // seen = {taskId, n} lần kéo gần nhất; pending = task mới chưa tìm
  parsed: null, edited: false, names: [], lastQuery: '',
  timeLeft: null, tick: null, poll: null, polling: false, searching: 0,
};

// ---------------------------------------------------------------- tiện ích
async function api(url, body) {
  const opt = body === undefined ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch(url, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok && data.error) throw new Error(data.error);
  return data;
}

function toast(text, kind = '', ms = 4500) {
  const t = document.createElement('div');
  t.className = 'toast ' + kind;
  t.setAttribute('role', kind === 'bad' ? 'alert' : 'status');
  t.innerHTML = `<span>${esc(text)}</span><button aria-label="Đóng thông báo">OK</button>`;
  const kill = () => t.remove();
  t.querySelector('button').onclick = kill;
  $('toasts').append(t);
  if (ms) setTimeout(kill, ms);
}

function beep() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.value = 880; g.gain.value = 0.08;
    o.connect(g).connect(ctx.destination); o.start(); o.stop(ctx.currentTime + 0.18);
    o.onended = () => ctx.close();
  } catch { /* trình duyệt chặn âm thanh: bỏ qua */ }
}

const imgUrl = (id, thumb) => `/api/image?id=${encodeURIComponent(id)}${thumb ? '&thumb=1' : ''}`;
const perPage = () => S.get('grid.perPage');
const anyDialogOpen = () => [...document.querySelectorAll('dialog')].some((d) => d.open);

// ---------------------------------------------------------------- ảnh đã nộp (localStorage, theo task)
const SUB_KEY = 'castle-search-submitted';
function readSubmitted() {
  try { const v = JSON.parse(localStorage.getItem(SUB_KEY) || '{}'); return v && typeof v === 'object' ? v : {}; } catch { return {}; }
}
const taskKey = () => st.task?.taskId || 'none';
function submittedOf(id) {
  return (readSubmitted()[taskKey()] || []).find((x) => x.id === id);
}
function markSubmitted(id, ok) {
  const all = readSubmitted(), key = taskKey();
  all[key] = [...(all[key] || []).filter((x) => x.id !== id), { id, ok }];
  const keys = Object.keys(all);
  for (const k of keys.slice(0, Math.max(0, keys.length - 20))) delete all[k];   // giữ 20 task gần nhất
  try { localStorage.setItem(SUB_KEY, JSON.stringify(all)); } catch { /* không lưu được: chỉ mất đánh dấu */ }
}

// ---------------------------------------------------------------- áp setting lên giao diện
function applyLook() {
  const root = document.documentElement;
  const theme = S.get('ui.theme');
  if (theme === 'system') delete root.dataset.theme; else root.dataset.theme = theme;
  root.dataset.font = S.get('ui.fontSize');
  $('noconfirm').hidden = S.get('submit.confirm');
  measureHeader();
}

function measureHeader() {
  document.documentElement.style.setProperty('--header-h', $('hdr').offsetHeight + 'px');
}

function onSetting(key) {
  if (key.startsWith('ui.') || key === 'submit.confirm') applyLook();
  if (key === 'grid.perPage') st.page = Math.floor(st.page / perPage()) * perPage();
  if (key.startsWith('grid.') || key === 'submit.mark') render();
  if (key.startsWith('parse.')) renderInfo();
  if (key.startsWith('level.')) renderLevels();
  if (key.startsWith('poll.')) startPoll();
  if (key.startsWith('detail.') && $('dlg').open) openDetail(st.current);
  measureHeader();
}

// ---------------------------------------------------------------- level đề
function levelsOf(query) {
  return String(query || '').split('\n').map((l) => l.trim()).filter((l) => l && !/^\[(IMAGE|VIDEO)\b/.test(l));
}

function pickLevel(n) {
  if (!n) return null;
  const mode = S.get('level.default');
  if (mode === 'first') return 0;
  if (mode === 'fixed') return Math.min(S.get('level.fixedN'), n) - 1;
  return n - 1;
}

function renderLevels() {
  const nav = $('levels');
  nav.hidden = !S.get('level.bar') || st.levels.length < 2;
  nav.replaceChildren(...st.levels.map((text, i) => {
    const b = document.createElement('button');
    b.textContent = 'L' + (i + 1);
    b.title = text;
    b.setAttribute('aria-pressed', String(i === st.level));
    b.onclick = () => useLevel(i);
    return b;
  }));
  measureHeader();
}

function useLevel(i) {
  st.level = i;
  $('q').value = st.levels[i];
  renderLevels();
  search();
}

// ---------------------------------------------------------------- tìm
async function search(parsed) {
  const q = $('q').value.trim();
  if (!q) { $('q').focus(); return; }
  if (!parsed) { st.edited = false; }
  const my = ++st.searching;
  $('info').textContent = 'Đang tìm...';
  $('grid').innerHTML = Array.from({ length: Math.min(perPage(), 4) }, () => '<div class="skeleton"></div>').join('');
  let r;
  try {
    r = await api('/api/search', { q, top: S.get('search.top'), ...(parsed ? { parsed } : {}) });
  } catch (e) {
    if (my === st.searching) { $('info').textContent = ''; render(); toast(e.message || 'Lỗi tìm kiếm', 'bad'); }
    return;
  }
  if (my !== st.searching) return;   // đã có lần tìm mới hơn
  st.lastQuery = q;
  st.parsed = r.parsed || null; st.edited = !!parsed; st.english = r.english || []; st.layers = r.layers || [];
  st.results = r.results || []; st.page = 0;
  st.why = Object.fromEntries(st.results.map((x) => [x.id, x.explain || {}]));
  renderInfo(); renderEditor(); render();
}

function renderInfo() {
  const pq = st.parsed;
  if (!pq) return;
  const parts = [];
  if (st.edited) parts.push('<span class="edited">Đang dùng bản đã sửa tay</span>');
  if (S.get('parse.showEnglish') && st.english.length) parts.push(`<div><b>EN:</b> ${esc(st.english.join(' '))}</div>`);
  const head = [];
  if (S.get('parse.showLayers')) head.push(`<b>Lớp:</b> ${st.layers.join(' → ')} ảnh`);
  if (S.get('parse.showClauses')) {
    head.push(`<b>số người:</b> ${pq.count ?? ('≥' + (pq.count_min || 0))}`);
    if (pq.names?.length) head.push(`<b>tên:</b> ${esc(pq.names.join(', '))}`);
  }
  if (head.length) parts.push(`<div>${head.join(' · ')}</div>`);
  if (S.get('parse.showClauses')) {
    (pq.people || []).forEach((p, i) => parts.push(`<div>người ${i + 1}${p.gender ? ' (' + (p.gender === 'male' ? 'nam' : 'nữ') + ')' : ''}${p.name ? ' [' + esc(p.name) + ']' : ''}: ${esc(p.text)}</div>`));
    if (pq.group) parts.push(`<div>nhóm ×${pq.group.n}: ${esc(pq.group.text)}</div>`);
    if (pq.scene?.length) parts.push(`<div>cảnh: ${esc(pq.scene.join(' · '))}</div>`);
  }
  $('info').innerHTML = parts.join('') || `<div>${st.results.length} kết quả</div>`;
  measureHeader();
}

// ---------------------------------------------------------------- form sửa biến
function renderEditor() {
  const mode = S.get('parse.editor'), box = $('editor');
  box.hidden = mode === 'off' || !st.parsed;
  if (box.hidden) { measureHeader(); return; }
  if (mode === 'open') box.open = true;
  const pq = st.parsed;
  const g = (v) => `<option value=""${!v ? ' selected' : ''}>không rõ</option><option value="male"${v === 'male' ? ' selected' : ''}>nam</option><option value="female"${v === 'female' ? ' selected' : ''}>nữ</option>`;
  const person = (p = {}) => `<div class="ed-item person">
      <label>Giới tính<select data-f="gender">${g(p.gender)}</select></label>
      <label class="wide">Mô tả<textarea data-f="text" rows="1">${esc(p.text)}</textarea></label>
      <label>Tên<input data-f="name" list="name-list" value="${esc(p.name || '')}" placeholder="tự dò"></label>
      <button type="button" class="icon" data-del aria-label="Xoá người này"><svg class="i"><use href="#i-x"/></svg></button></div>`;
  const scene = (s = '') => `<div class="ed-item scene"><label>Cụm cảnh<textarea data-f="text" rows="1">${esc(s)}</textarea></label>
      <button type="button" class="icon" data-del aria-label="Xoá cụm cảnh"><svg class="i"><use href="#i-x"/></svg></button></div>`;
  const group = (gr) => `<div class="ed-item group"><label>Số người<input data-f="n" type="number" min="1" max="20" value="${gr.n}" inputmode="numeric"></label>
      <label class="wide">Mô tả nhóm<textarea data-f="text" rows="1">${esc(gr.text)}</textarea></label>
      <button type="button" class="icon" data-del aria-label="Xoá nhóm"><svg class="i"><use href="#i-x"/></svg></button></div>`;
  const exact = pq.count !== null && pq.count !== undefined;
  $('ed-body').innerHTML = `
    <div class="ed-grid">
      <label>Số người<input id="ed-count" type="number" min="0" max="20" inputmode="numeric" value="${exact ? pq.count : (pq.count_min || '')}" placeholder="không ràng buộc"></label>
      <label>Kiểu<select id="ed-exact"><option value="1"${exact ? ' selected' : ''}>đúng k người</option><option value="0"${!exact ? ' selected' : ''}>ít nhất k người</option></select></label>
      <label>Tên (cách nhau bằng dấu phẩy)<input id="ed-names" list="name-list" value="${esc((pq.names || []).join(', '))}"></label>
    </div>
    <datalist id="name-list">${st.names.map((n) => `<option value="${esc(n)}">`).join('')}</datalist>
    <div class="ed-section"><h4>Cụm người</h4><div id="ed-people">${(pq.people || []).map(person).join('')}</div>
      <div><button type="button" id="ed-add-person">+ người</button></div></div>
    <div class="ed-section"><h4>Cụm nhóm</h4><div id="ed-group">${pq.group ? group(pq.group) : ''}</div>
      <div><button type="button" id="ed-add-group"${pq.group ? ' hidden' : ''}>+ nhóm</button></div></div>
    <div class="ed-section"><h4>Cụm cảnh</h4><div id="ed-scene">${(pq.scene || []).map(scene).join('')}</div>
      <div><button type="button" id="ed-add-scene">+ cảnh</button></div></div>
    <div class="row"><button type="button" class="primary" id="ed-run">Tìm lại với bản đã sửa</button>
      <button type="button" id="ed-restore">Khôi phục bản tự tách</button></div>`;
  const add = (where, html) => { $(where).insertAdjacentHTML('beforeend', html); };
  $('ed-add-person').onclick = () => add('ed-people', person());
  $('ed-add-scene').onclick = () => add('ed-scene', scene());
  $('ed-add-group').onclick = () => { add('ed-group', group({ n: 1, text: '' })); $('ed-add-group').hidden = true; };
  $('ed-body').onclick = (e) => {
    const del = e.target.closest('[data-del]');
    if (!del) return;
    if (del.closest('#ed-group')) $('ed-add-group').hidden = false;
    del.closest('.ed-item').remove();
  };
  $('ed-run').onclick = () => search(readEditor());
  $('ed-restore').onclick = () => search();
  measureHeader();
}

function readEditor() {
  const val = (el, f) => el.querySelector(`[data-f="${f}"]`)?.value.trim() ?? '';
  const raw = $('ed-count').value.trim(), k = raw === '' ? null : Math.max(0, Math.min(20, parseInt(raw, 10) || 0));
  const exact = $('ed-exact').value === '1';
  const g = $('ed-group').querySelector('.ed-item');
  return {
    count: exact ? k : null, count_min: exact ? null : k,
    names: $('ed-names').value.split(',').map((s) => s.trim()).filter(Boolean),
    people: [...$('ed-people').children].map((el) => ({ text: val(el, 'text'), gender: val(el, 'gender') || null, name: val(el, 'name') || null })),
    group: g ? { text: val(g, 'text'), n: Math.max(1, Math.min(20, parseInt(val(g, 'n'), 10) || 1)) } : null,
    scene: [...$('ed-scene').children].map((el) => val(el, 'text')),
  };
}

// ---------------------------------------------------------------- lưới kết quả
function render() {
  const per = perPage(), grid = $('grid');
  const cols = S.get('grid.columns');
  if (cols === 'auto') { delete grid.dataset.cols; grid.style.removeProperty('--cols'); }
  else { grid.dataset.cols = cols; grid.style.setProperty('--cols', cols); }
  $('pager').hidden = !st.results.length;
  if (!st.results.length) {
    grid.innerHTML = st.parsed ? '<div class="empty">Không có kết quả. Thử sửa đề hoặc chọn level khác.</div>' : '';
    return;
  }
  const shown = st.results.slice(st.page, st.page + per);
  $('pos').textContent = `${st.page + 1}–${st.page + shown.length} / ${st.results.length}`;
  $('prev').disabled = st.page === 0;
  $('next').disabled = st.page + per >= st.results.length;
  const thumbs = S.get('grid.thumbs');
  grid.replaceChildren(...shown.map((x, k) => {
    const d = document.createElement('div');
    const done = S.get('submit.mark') && submittedOf(x.id);
    d.className = 'cell' + (done ? (done.ok ? ' sub-ok' : ' sub-bad') : '');
    const pairs = (x.explain?.pairs || []).map((p, i) => `${i + 1}:${p.tag || (p.person ? '?' : '—')}`).join(' · ');
    d.innerHTML = `<button class="open" aria-label="Mở ảnh ${st.page + k + 1}: ${esc(x.id)}"><img alt="" loading="lazy" decoding="async" src="${imgUrl(x.id, thumbs)}"></button>`
      + `<span class="tag id">${k < 9 ? `<kbd>${k + 1}</kbd> ` : ''}#${st.page + k + 1} · ${esc(x.id)}</span>`
      + (S.get('grid.showScore') ? `<span class="tag score">${x.score}</span>` : '')
      + (done ? `<span class="tag done ${done.ok ? 'ok' : 'bad'}">đã nộp · ${done.ok ? 'đúng' : 'sai'}</span>` : '')
      + (S.get('grid.showPairs') && pairs ? `<span class="tag why" title="${esc(pairs)}">${esc(pairs)}</span>` : '')
      + `<button class="submit" aria-label="Nộp ${esc(x.id)}"><svg class="i"><use href="#i-send"/></svg>Nộp</button>`;
    d.querySelector('.open').onclick = () => openDetail(x.id);
    d.querySelector('.submit').onclick = () => submitId(x.id);
    return d;
  }));
  st.results.slice(st.page + per, st.page + 2 * per).forEach((x) => { new Image().src = imgUrl(x.id, thumbs); });  // tải trước trang sau
}

function move(step) {
  const p = st.page + step * perPage();
  if (p < 0 || p >= st.results.length) return;
  st.page = p; render();
  window.scrollTo({ top: 0, behavior: 'instant' });
}

// ---------------------------------------------------------------- task DRES: kéo đề, đồng hồ, báo gợi ý mới
function showTask(t) {
  const chip = $('task-chip');
  clearInterval(st.tick);
  if (!t.running) {
    chip.className = 'chip';
    chip.textContent = `Không có task chạy (${t.taskStatus || '?'})${t.query ? ' · đề gần nhất' : ''}`;
    return;
  }
  st.timeLeft = { ms: t.timeLeftMs || 0, at: Date.now() };
  const draw = () => {
    const left = Math.max(0, Math.round((st.timeLeft.ms - (Date.now() - st.timeLeft.at)) / 1000));
    chip.className = 'chip live' + (left <= 30 ? ' low' : '');
    chip.textContent = `Task ${t.taskId || '?'} · còn ${Math.floor(left / 60)}:${String(left % 60).padStart(2, '0')}`;
    if (!left) clearInterval(st.tick);
  };
  draw();
  st.tick = setInterval(draw, 1000);
}

function adopt(t) {
  st.task = t;
  st.levels = levelsOf(t.query);
  st.seen = { taskId: t.taskId, n: st.levels.length };
  st.pending = null;
  $('banner').hidden = true;
  st.level = pickLevel(st.levels.length);
  $('q').value = st.level === null ? '' : st.levels[st.level];
  renderLevels();
  measureHeader();
}

async function fetchTask() {
  const t = await api('/api/task');
  if (t.error) throw new Error(t.error);
  showTask(t);
  return t;
}

async function pull() {
  $('task').disabled = true;
  try {
    const t = await fetchTask();
    if (!t.query) { toast(`Chưa có đề (task: ${t.taskStatus || '?'}).`); return; }
    adopt(t);
    await search();
  } catch (e) {
    toast(e.message || 'Lỗi kéo đề', 'bad');
  } finally {
    $('task').disabled = false;
  }
}

// Chỉ báo, không đổi ô đề hay kết quả đang xem
async function checkTask() {
  if (st.polling) return;
  st.polling = true;
  try {
    const t = await fetchTask();
    if (!t.query) return;
    const n = levelsOf(t.query).length;
    if (!st.seen) { st.seen = { taskId: t.taskId, n }; return; }
    const newTask = t.taskId !== st.seen.taskId, more = !newTask && n > st.seen.n;
    if (!newTask && !more) return;
    if (st.pending && st.pending.taskId === t.taskId && levelsOf(st.pending.query).length === n) return;
    st.pending = t;
    $('banner-text').textContent = newTask ? `Có task mới (${t.taskId || '?'}), ${n} level` : `Có gợi ý mới (L${n})`;
    $('banner').hidden = false;
    measureHeader();
    if (S.get('poll.sound')) beep();
  } catch (e) {
    $('task-chip').title = 'Lần kiểm tra gần nhất lỗi: ' + (e.message || e);
  } finally {
    st.polling = false;
  }
}

function startPoll() {
  clearInterval(st.poll);
  const sec = S.get('poll.seconds');
  if (sec) st.poll = setInterval(() => { if (!document.hidden) checkTask(); }, sec * 1000);
}

$('banner-go').onclick = () => { if (st.pending) { adopt(st.pending); search(); } };
$('banner-skip').onclick = () => {   // bỏ qua: coi như đã thấy, chỉ báo lại khi có thay đổi tiếp
  if (st.pending) st.seen = { taskId: st.pending.taskId, n: levelsOf(st.pending.query).length };
  st.pending = null; $('banner').hidden = true; measureHeader();
};

// ---------------------------------------------------------------- nộp
function askConfirm(id) {
  return new Promise((resolve) => {
    const d = $('confirm');
    $('c-img').src = imgUrl(id, true);
    $('c-img').alt = id;
    $('c-id').textContent = id;
    const prev = submittedOf(id);
    $('c-prev').textContent = prev ? `Ảnh này đã nộp trước đó (${prev.ok ? 'đúng' : 'sai'}).` : '';
    d.returnValue = '';
    d.onclose = () => resolve(d.returnValue === 'yes');
    $('c-yes').onclick = () => d.close('yes');
    d.showModal();
    $('c-yes').focus();
  });
}

async function submitId(id) {
  if (!id) return;
  if (S.get('submit.confirm') && !(await askConfirm(id))) return;
  toast(`Đang nộp ${id}...`, '', 2000);
  let r;
  try {
    r = await api('/api/submit', { id });
  } catch (e) {
    toast(`${id}: ${e.message}`, 'bad', 0);
    return;
  }
  const ok = !!(r.raw && r.raw.includes('CORRECT') && !r.raw.includes('WRONG'));
  toast(`${id}: ${r.status} · ${r.message}`, ok ? 'ok' : 'bad', ok ? 6000 : 0);
  if (r.status && r.status < 400) markSubmitted(id, ok);
  if ($('dlg').open) $('dlg').close();
  render();
  checkTask();   // chỉ cập nhật đồng hồ + banner, không tự tìm lại
}

// ---------------------------------------------------------------- ảnh chi tiết
async function openDetail(id) {
  if (!id) return;
  st.current = id;
  const r = await api(`/api/detail?id=${encodeURIComponent(id)}&span=${S.get('detail.span')}`).catch((e) => { toast(e.message, 'bad'); return null; });
  if (!r || st.current !== id) return;
  const k = st.results.findIndex((x) => x.id === id);
  $('title').textContent = id;
  $('d-pos').textContent = k >= 0 ? `#${k + 1} / ${st.results.length}` : 'ảnh lân cận';
  $('d-prev').disabled = k <= 0; $('d-next').disabled = k < 0 || k >= st.results.length - 1;
  const done = submittedOf(id);
  $('sub').innerHTML = `<svg class="i"><use href="#i-send"/></svg>${done ? 'Nộp lại' : 'Nộp'}`;
  $('strip').replaceChildren(...r.neighbors.map((n) => {
    const b = document.createElement('button');
    b.title = n; b.setAttribute('aria-current', String(n === id));
    b.innerHTML = `<img alt="${esc(n)}" loading="lazy" src="${imgUrl(n, true)}">`;
    b.onclick = () => openDetail(n);
    return b;
  }));
  // box người: toạ độ pixel ảnh gốc 1280x720; xanh = người được ghép với một cụm trong đề
  const hits = new Set(((st.why[id] || {}).pairs || []).map((p) => p.person).filter(Boolean));
  const mode = S.get('detail.boxes');
  const boxes = mode === 'off' ? [] : r.people.map((p, i) => ({ p, i })).filter(({ p }) => mode === 'all' || hits.has(p.id));
  $('big').innerHTML = `<div class="frame"><img alt="${esc(id)}" src="${imgUrl(id)}">` + boxes.map(({ p, i }) =>
    `<div class="box${hits.has(p.id) ? ' hit' : ''}" style="left:${p.box[0] / 12.8}%;top:${p.box[1] / 7.2}%;width:${(p.box[2] - p.box[0]) / 12.8}%;height:${(p.box[3] - p.box[1]) / 7.2}%"><b>p${i + 1}</b></div>`).join('') + '</div>';
  $('meta').hidden = !S.get('detail.meta');
  $('meta').innerHTML = '<ul>' + r.people.map((p, i) => `<li>p${i + 1}: <b>${esc(p.name || '?')}</b> · ${esc(p.action || '')} · ${esc(p.look)}</li>`).join('') + '</ul>'
    + (r.rois.length ? `<div>Vùng: ${esc(r.rois.join(' · '))}</div>` : '') + `<div>Vật: ${esc(r.objects.join(', '))}</div>`;
  if (!$('dlg').open) $('dlg').showModal();
  fitFrame();   // sau showModal: dialog đóng thì clientWidth = 0
  $('strip').querySelector('[aria-current="true"]')?.scrollIntoView({ block: 'nearest', inline: 'center' });
}

// Box tính theo % khung ảnh: khung phải ôm sát ảnh khi ảnh bị co theo chiều cao (object-fit: contain)
function fitFrame() {
  const frame = $('big').querySelector('.frame'), img = frame?.querySelector('img');
  if (!img) return;
  const fit = () => {
    const maxH = parseFloat(getComputedStyle(img).maxHeight) || Infinity;
    const w = Math.min($('big').clientWidth, maxH * 16 / 9);
    frame.style.width = w + 'px';
  };
  if (img.complete) fit(); else img.onload = fit;
}

function stepDetail(step) {
  const k = st.results.findIndex((x) => x.id === st.current);
  const n = st.results[k + step];
  if (k >= 0 && n) {
    openDetail(n.id);
    const page = Math.floor((k + step) / perPage()) * perPage();
    if (page !== st.page) { st.page = page; render(); }
  }
}

// ---------------------------------------------------------------- popup chung
document.querySelectorAll('dialog').forEach((d) => {
  d.addEventListener('click', (e) => {
    if (e.target.closest('[data-close]')) d.close(e.target.closest('[data-close]').value || '');
    else if (e.target === d) d.close();   // bấm nền ngoài popup
  });
});

$('open-set').onclick = () => { S.renderPanel($('set-panel'), onSetting); $('settings').showModal(); };
$('open-keys').onclick = () => $('keys').showModal();
$('open-cfg').onclick = () => { loadConfig(); $('cfg').showModal(); };
$('s-reset').onclick = () => {
  if (!confirm('Khôi phục toàn bộ setting về mặc định?')) return;
  S.reset(); S.renderPanel($('set-panel'), onSetting); S.SCHEMA.forEach((s) => onSetting(s.key));
  toast('Đã khôi phục mặc định', 'ok');
};
$('s-export').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([S.exportJson()], { type: 'application/json' }));
  a.download = 'castle-search-settings.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
$('s-import').onclick = () => $('s-file').click();
$('s-file').onchange = async () => {
  const f = $('s-file').files[0];
  $('s-file').value = '';
  if (!f) return;
  try {
    const skipped = S.importJson(await f.text());
    S.renderPanel($('set-panel'), onSetting); S.SCHEMA.forEach((s) => onSetting(s.key));
    toast(skipped ? `Đã nhập; bỏ qua ${skipped} khoá không hợp lệ` : 'Đã nhập setting', skipped ? '' : 'ok');
  } catch (e) {
    toast(e.message, 'bad');
  }
};

// ---------------------------------------------------------------- phím tắt
$('go').onclick = () => search();
$('task').onclick = () => pull();
$('prev').onclick = () => move(-1);
$('next').onclick = () => move(1);
$('d-prev').onclick = () => stepDetail(-1);
$('d-next').onclick = () => stepDetail(1);
$('sub').onclick = () => submitId(st.current);
$('q').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); search(); } });
$('q').addEventListener('input', () => { if (st.level !== null) { st.level = null; renderLevels(); } });

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey || e.target.matches?.('input, textarea, select')) return;
  if ($('dlg').open && !$('confirm').open) {
    if (e.key === 'ArrowRight') { e.preventDefault(); stepDetail(1); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); stepDetail(-1); }
    if (e.key === 'Enter' && !e.target.closest('button')) { e.preventDefault(); submitId(st.current); }
    return;
  }
  if (anyDialogOpen()) return;
  if (e.key === 'ArrowRight') move(1);
  else if (e.key === 'ArrowLeft') move(-1);
  else if (e.key === '/') { e.preventDefault(); $('q').focus(); }
  else if (e.key === '?') $('keys').showModal();
  else if (/^[1-9]$/.test(e.key)) { const x = st.results[st.page + Number(e.key) - 1]; if (x && Number(e.key) <= perPage()) openDetail(x.id); }
});

// ---------------------------------------------------------------- cấu hình DRES
async function loadConfig() {
  const c = await api('/api/config').catch(() => ({}));
  for (const el of $('cfgForm').elements) if (el.name) el.value = c[el.name] ?? '';
  $('cfgForm').elements.DRES_PASSWORD.placeholder = c.hasPassword ? 'đã lưu, để trống = giữ nguyên' : 'chưa có';
  return c;
}
$('cfgForm').onsubmit = async (e) => {
  e.preventDefault();
  const body = Object.fromEntries(new FormData($('cfgForm')));
  try {
    await api('/api/config', body);
    $('cfgMsg').textContent = 'Đã lưu vào .env';
    toast('Đã lưu cấu hình DRES', 'ok');
    loadConfig();
  } catch (err) {
    $('cfgMsg').textContent = 'Lỗi: ' + err.message;
  }
};

// ---------------------------------------------------------------- khởi động
window.addEventListener('resize', () => { measureHeader(); fitFrame(); });
applyLook();
startPoll();
api('/api/names').then((n) => { st.names = Array.isArray(n) ? n : []; }).catch(() => {});
loadConfig().then((c) => { if (!c.DRES_USERNAME || !c.hasPassword) $('cfg').showModal(); });

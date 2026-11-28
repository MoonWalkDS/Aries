// Label Viewer: `npm start` rồi mở http://localhost:5173
'use strict';
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');
const { execFile, execFileSync } = require('node:child_process');
const os = require('node:os');
const { Store, SAFE } = require('./store');
const { Auth, clientIp } = require('./auth');
const { Journal, GitSync, setJournal } = require('./journal');

// Thư mục labeling/ (mặc định thư mục cha của label_viewer/). `npm start -- <dir>` để trỏ sang bản sao khi thử.
const ROOT = path.resolve(process.argv[2] || process.env.LABELING_ROOT || path.join(__dirname, '..'));
const PORT = Number(process.env.PORT || 5173);
const HOST = '127.0.0.1'; // có API ghi file: chỉ mở cho máy này
const PUBLIC = path.join(__dirname, 'public');
const TYPES = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.jpg': 'image/jpeg' };

const auth = new Auth();

// Đồng bộ nhãn qua git (journal.js): tắt bằng LABEL_SYNC=0, hoặc tự tắt khi chạy trên bản sao (`npm start -- <dir>`).
const SYNC = process.env.LABEL_SYNC !== '0' && !process.argv[2];
const gitName = () => { try { return execFileSync('git', ['config', 'user.name'], { cwd: ROOT, encoding: 'utf8', windowsHide: true }).trim(); } catch { return ''; } };
const LABELER = process.env.LABELER || gitName() || os.userInfo().username;
let journal = null, sync = null;
if (SYNC) {
  journal = new Journal({ root: ROOT, dir: path.join(ROOT, 'annotations', 'journal'), stateFile: path.join(ROOT, '..', 'data', 'intermediate', 'journal-state.json'), user: LABELER });
  sync = new GitSync({ repoDir: ROOT, dir: journal.dir });
  console.log('Đồng bộ nhãn: đang pull…');
  const st = sync.setup();
  const r = journal.replay();
  setJournal(journal);
  console.log(`Đồng bộ nhãn (${st.mode === 'remote' ? `git ${st.branch}` : st.mode === 'local' ? 'chỉ trên máy này: chưa có git remote "origin"' : 'tắt: không phải repo git'}): `
    + `áp ${r.applied} thay đổi của người khác${r.files.length ? ` vào ${r.files.length} file` : ''}. Người gán: ${LABELER} (${journal.me}).`);
  if (st.error) console.log(`  ! ${st.error}`);
  sync.push(journal.file);   // nhãn còn sót của lần chạy trước (tắt máy, mất mạng)
}

console.time('Nạp dữ liệu');
const store = new Store(ROOT);
console.timeEnd('Nạp dữ liệu');

function send(res, status, body, type = 'application/json; charset=utf-8', extra = {}) {
  res.writeHead(status, { 'Content-Type': type, ...extra });
  res.end(typeof body === 'string' || Buffer.isBuffer(body) ? body : JSON.stringify(body));
}

function sendFile(res, file, cache) {
  fs.stat(file, (err, st) => {
    if (err || !st.isFile()) return send(res, 404, { error: 'Không có file' });
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream', 'Content-Length': st.size, 'Cache-Control': cache ? 'max-age=3600' : 'no-cache' });
    fs.createReadStream(file).pipe(res);
  });
}

function readBodyRaw(req) {
  // Chỉ nhận JSON: form của trang khác không gửi được kiểu này nếu không qua preflight CORS (chặn CSRF).
  if (!String(req.headers['content-type'] || '').startsWith('application/json')) return Promise.reject(Object.assign(new Error('Cần Content-Type: application/json'), { status: 415 }));
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (c) => { data += c; if (data.length > 5e6) reject(Object.assign(new Error('Body quá lớn'), { status: 413 })); });
    req.on('end', () => { try { resolve(data ? JSON.parse(data) : {}); } catch { reject(Object.assign(new Error('JSON không hợp lệ'), { status: 400 })); } });
  });
}

const list = (v) => (v ? v.split(',').filter(Boolean) : []);
const num = (v) => (v === null || v === '' ? null : Number(v));

function filtersFrom(q) {
  return {
    // token dạng "profile|label", nối bằng ";;"
    tokens: (q.get('t') || '').split(';;').filter(Boolean).map((s) => { const i = s.indexOf('|'); return { profile: s.slice(0, i), label: s.slice(i + 1) }; }),
    mode: q.get('mode') === 'and' ? 'and' : 'or',
    q: q.get('q') || '',
    sources: list(q.get('src')),
    hours: list(q.get('hours')),
    zone: q.get('zone') || '',
    minPeople: num(q.get('minp')),
    maxPeople: num(q.get('maxp')),
    edited: q.get('edited') === '1',
    manual: q.get('manual') === '1',
    conflict: q.get('conflict') === '1',
    sort: q.get('sort') || 'id',
  };
}

const hitOut = (h) => ({ profile: h.profile, key: h.key, kind: h.kind, id: h.id, inherited: !!h.inherited, label: h.state?.label ?? null, label_vi: h.state?.label_vi ?? '', source: h.state?.source ?? null, score: h.state?.score ?? null });

async function api(req, res, url, user) {
  // Người và trang đang thao tác: đặt lại ngay sau khi đọc body (lúc chờ đọc, request khác có thể đã đổi store.actor).
  // Trang lấy từ Referer, chỉ để ghi vào mô tả bước hoàn tác.
  const PAGES = { '/': 'Ảnh', '/index.html': 'Ảnh', '/board.html': 'Bảng định danh', '/quick.html': 'Gán nhanh danh tính', '/action.html': 'Gán nhanh hành động', '/objects.html': 'Đồ vật' };
  const bind = () => { store.actor = user?.name || LABELER; try { store.page = PAGES[new URL(req.headers.referer).pathname] || ''; } catch { store.page = ''; } };
  const readBody = async (r) => { const b = await readBodyRaw(r); bind(); return b; };
  const q = url.searchParams;
  const p = url.pathname;
  if (req.method === 'GET' && p === '/api/me') return send(res, 200, { user: user?.name ?? null, admin: !auth.enabled || !!user?.admin, auth: auth.enabled });
  if (req.method === 'POST' && p === '/api/logout') return send(res, 200, { ok: true }, undefined, { 'Set-Cookie': auth.cookie(req, '', 0) });
  // Gộp dữ liệu chạy Python trên máy này, Tải lại xoá lịch sử hoàn tác của mọi người: chỉ admin.
  if (auth.enabled && !user?.admin && req.method === 'POST' && (p === '/api/merge' || p === '/api/reload' || p === '/api/sync')) return send(res, 403, { error: 'Chỉ admin được làm việc này.' });
  if (req.method === 'GET' && p === '/api/meta') return send(res, 200, { ...store.meta(), undo: store.undoStack.length, edited: store.edited.size });
  if (req.method === 'GET' && p === '/api/images') {
    const rows = store.query(filtersFrom(q));
    const size = Math.min(Number(q.get('size') || 60), 500);
    const page = Math.max(Number(q.get('page') || 1), 1);
    const items = rows.slice((page - 1) * size, page * size).map((r) => ({ ...r, edited: store.edited.has(r.image_id), hits: r.hits.map(hitOut) }));
    return send(res, 200, { total: rows.length, page, size, ids: rows.map((r) => r.image_id), items });
  }
  if (req.method === 'GET' && p === '/api/export') {
    const rows = store.query(filtersFrom(q));
    const fmt = q.get('format') || 'ids';
    if (fmt === 'json') return send(res, 200, JSON.stringify(rows.map((r) => ({ ...r, hits: r.hits.map(hitOut) })), null, 2), 'application/json; charset=utf-8', { 'Content-Disposition': 'attachment; filename="labels_export.json"' });
    if (fmt === 'csv') {
      const esc = (s) => `"${String(s ?? '').replace(/"/g, '""')}"`;
      const lines = ['image_id,hour,num_people,hits'];
      for (const r of rows) lines.push([r.image_id, r.hour, r.num_people, esc(r.hits.map((h) => `${h.id}:${h.state?.label ?? '-'}(${h.state?.source ?? '-'})`).join('; '))].join(','));
      return send(res, 200, '﻿' + lines.join('\n'), 'text/csv; charset=utf-8', { 'Content-Disposition': 'attachment; filename="labels_export.csv"' });
    }
    return send(res, 200, rows.map((r) => r.image_id).join('\n') + '\n', 'text/plain; charset=utf-8', { 'Content-Disposition': 'attachment; filename="ids.txt"' });
  }
  let m;
  if (req.method === 'GET' && (m = p.match(/^\/api\/image\/([\w-]+)$/))) {
    const d = store.detail(m[1]);
    return d ? send(res, 200, d) : send(res, 404, { error: 'Không có ảnh' });
  }
  if (req.method === 'POST' && p === '/api/label') {
    const body = await readBody(req);
    return send(res, 200, { ...store.apply(body.ops), undo: store.undoStack.length });
  }
  if (req.method === 'POST' && p === '/api/manual') {
    const body = await readBody(req);
    return send(res, 200, store.manualOp(body));
  }
  // Xoá box (máy dò: ẩn; tự vẽ: bỏ hẳn) và sửa khung box máy dò (thay bằng box tự vẽ): docs/adr/0002
  if (req.method === 'POST' && p === '/api/person/delete') return send(res, 200, store.deletePeople((await readBody(req)).ids));
  if (req.method === 'POST' && p === '/api/person/replace') { const b = await readBody(req); return send(res, 200, store.replacePerson(b.id, b.box)); }
  if (req.method === 'GET' && p === '/api/hotkeys') return send(res, 200, store.hotkeys());
  if (req.method === 'GET' && p === '/api/action/summary') return send(res, 200, store.actionSummary());
  if (req.method === 'GET' && p === '/api/action/members') return send(res, 200, { items: store.actionMembers(q.has('suspect') ? { suspect: true } : q.has('recheck') ? { recheck: true } : q.has('conflict') ? { conflict: true } : q.has('teacher') ? { teacher: q.get('teacher') } : { label: q.get('label') || '' }) });
  if (req.method === 'GET' && p === '/api/identity/summary') return send(res, 200, { ...store.identitySummary(), names_all: store.identityNames() });
  if (req.method === 'GET' && p === '/api/identity/members') {
    const by = q.has('extra') ? { extra: true } : q.has('recheck') ? { recheck: true } : q.has('hard') ? { hard: true } : q.has('unknown') ? { unknown: true } : q.has('cluster') ? { cluster: q.get('cluster') } : { name: q.get('name') || '' };
    return send(res, 200, { by, items: store.identityMembers(by) });
  }
  if (req.method === 'POST' && p === '/api/identity/rename') {
    const body = await readBody(req);
    return send(res, 200, { ...store.renameIdentity(body), names: store.identityNames() });
  }
  if (req.method === 'POST' && p === '/api/identity') {
    const body = await readBody(req);
    return send(res, 200, { ...store.setIdentity(body), names: store.identityNames() });
  }
  if (req.method === 'GET' && p === '/api/objects/meta') return send(res, 200, store.objects.meta());
  if (req.method === 'GET' && (m = p.match(/^\/api\/objects\/image\/([\w-]+)$/))) {
    const d = store.objects.image(m[1]);
    return d ? send(res, 200, d) : send(res, 404, { error: 'Không có ảnh' });
  }
  if (req.method === 'GET' && p === '/api/objects/next') return send(res, 200, { id: store.objects.next(q.get('region') || '', q.get('from') || '', q.get('kind') || 'unlabeled', q.get('dir') === '-1' ? -1 : 1) });
  if (req.method === 'POST' && p === '/api/objects/regions') return send(res, 200, store.objects.setRegions((await readBody(req)).regions));
  if (req.method === 'POST' && p === '/api/objects/label') return send(res, 200, store.objects.setLabel(await readBody(req)));
  if (req.method === 'POST' && p === '/api/objects/clear') return send(res, 200, store.objects.clearLabel(await readBody(req)));
  if (req.method === 'GET' && p === '/api/objects/export') return send(res, 200, JSON.stringify(store.objects.exportRows(), null, 1), 'application/json; charset=utf-8', { 'Content-Disposition': 'attachment; filename="object_labels_effective.json"' });
  if (req.method === 'GET' && p === '/api/sync') return send(res, 200, syncStatus());
  if (req.method === 'POST' && p === '/api/sync') return send(res, 200, await syncNow());
  if (req.method === 'POST' && p === '/api/undo') return send(res, 200, store.undo());
  if (req.method === 'POST' && p === '/api/reload') { store.reload(); return send(res, 200, { ok: true, images: store.images.length }); }
  if (req.method === 'POST' && p === '/api/merge') return send(res, 200, await runMerge());
  return send(res, 404, { error: 'Không có API này' });
}

function syncStatus() {
  if (!journal) return { enabled: false };
  return { enabled: true, me: journal.me, user: LABELER, pending: journal.pending, ...sync.status() };
}

// Đẩy nhãn của mình lên, kéo nhãn người khác về, áp vào file rồi nạp lại (xoá lịch sử hoàn tác).
async function syncNow() {
  if (!journal) throw Object.assign(new Error('Đồng bộ đang tắt (LABEL_SYNC=0 hoặc chạy trên bản sao).'), { status: 400 });
  if (await sync.push(journal.file)) journal.pending = 0;
  sync.pull();
  const r = journal.replay();
  if (r.applied) store.reload();
  return { ...syncStatus(), applied: r.applied };
}

// Tự đẩy mỗi 5 phút khi có nhãn mới, và khi tắt server (Ctrl+C, đóng cửa sổ terminal).
if (journal) {
  setInterval(async () => { if (journal.pending && await sync.push(journal.file)) journal.pending = 0; }, 5 * 60 * 1000).unref();
  let closing = false;
  const close = (sig) => {
    if (closing) return;
    closing = true;
    if (journal.pending || sync.mode === 'remote') {
      console.log('Đang đẩy nhãn lên…');
      sync.push(journal.file, { sync: true }).then((ok) => { console.log(ok ? 'Đã đẩy nhãn.' : `Chưa đẩy được nhãn${sync.lastError ? `: ${sync.lastError}` : ''}. Lần mở sau sẽ đẩy tiếp.`); process.exit(0); });
    } else process.exit(0);
  };
  for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP', 'SIGBREAK']) process.on(sig, () => close(sig));
}

// `python labeling/pipeline/main.py enrich merge`: gộp nhãn đã duyệt vào enriched/ và xuất exports/, rồi nạp lại.
let merging = false;
function runMerge() {
  if (merging) return Promise.reject(Object.assign(new Error('Đang gộp, đợi lượt trước xong.'), { status: 409 }));
  const venv = [path.join(ROOT, '..', '.venv', 'Scripts', 'python.exe'), path.join(ROOT, '..', '.venv', 'bin', 'python')].find((f) => fs.existsSync(f));
  const python = process.env.PYTHON || venv || 'python';
  merging = true;
  return new Promise((resolve, reject) => {
    execFile(python, ['-X', 'utf8', path.join('pipeline', 'main.py'), 'enrich', 'merge'], { cwd: ROOT, timeout: 600000, maxBuffer: 1e7, env: { ...process.env, PYTHONIOENCODING: 'utf-8' } }, (err, stdout, stderr) => {
      merging = false;
      const log = `${stdout}${stderr}`.trim().split('\n').slice(-12).join('\n');
      if (err) return reject(Object.assign(new Error(`Gộp thất bại:\n${log}`), { status: 500 }));
      store.reload();
      const line = log.split('\n').reverse().find((l) => l.includes('enrich merge:')) || '';
      resolve({ ok: true, images: store.images.length, summary: line.slice(line.indexOf('enrich merge:') + 14) });
    });
  });
}

function files(res, p) {
  let m;
  const ok = (...parts) => parts.every((s) => SAFE.test(s));
  if ((m = p.match(/^\/(img|thumb)\/([^/]+)\.jpg$/)) && ok(m[2])) return sendFile(res, path.join(store.dataset, m[1] === 'img' ? 'images' : 'thumbs', `${m[2]}.jpg`), true);
  if ((m = p.match(/^\/crop\/person\/([^/]+)\.jpg$/)) && ok(m[1])) return sendFile(res, path.join(store.data, 'enrich', 'people', 'crops', `${m[1]}.jpg`), true);
  if ((m = p.match(/^\/crop\/roi\/([^/]+)\/([^/]+)\.jpg$/)) && ok(m[1], m[2])) return sendFile(res, path.join(store.data, 'enrich', 'rois', m[1], 'crops', `${m[2]}.jpg`), true);
  const name = p === '/' ? 'index.html' : p.slice(1);
  if (/^[\w.-]+$/.test(name)) return sendFile(res, path.join(PUBLIC, name), false);
  return send(res, 404, { error: 'Không có' });
}

const OPEN = new Set(['/login.html', '/style.css', '/login.js']);

async function login(req, res) {
  const ip = clientIp(req);
  if (auth.limited(ip)) return send(res, 429, { error: 'Sai quá nhiều lần. Thử lại sau 15 phút.' });
  const body = await readBodyRaw(req);
  const user = auth.check(String(body.user || ''), body.password);
  if (!user) { auth.failed(ip); return send(res, 401, { error: 'Sai tên hoặc mật khẩu.' }); }
  console.log(`Đăng nhập: ${user.name} từ ${ip}`);
  return send(res, 200, { user: user.name, admin: user.admin }, undefined, { 'Set-Cookie': auth.cookie(req, auth.issue(user.name), 14 * 86400) });
}

http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || 'localhost'}`);
  try {
    let user = null;
    if (auth.enabled) {
      if (url.pathname === '/api/login' && req.method === 'POST') return await login(req, res);
      user = auth.userOf(req);
      if (!user && !OPEN.has(url.pathname)) {
        if (url.pathname.startsWith('/api/')) return send(res, 401, { error: 'Chưa đăng nhập.' });
        res.writeHead(302, { Location: `/login.html?next=${encodeURIComponent(req.url)}` });
        return res.end();
      }
    }
    store.actor = user?.name || LABELER;
    if (url.pathname.startsWith('/api/')) return await api(req, res, url, user);
    return files(res, url.pathname);
  } catch (e) {
    if (!e.status) console.error(e);
    send(res, e.status || 500, { error: e.message });
  }
}).listen(PORT, HOST, () => {
  console.log(`${store.images.length} ảnh, ${Object.keys(store.profiles).length} profile nhãn.`);
  console.log(`Mở http://localhost:${PORT}`);
  console.log(auth.enabled ? `Bắt đăng nhập: ${Object.keys(auth.doc.users).length} tài khoản trong users.json.` : 'Không có users.json: KHÔNG bắt đăng nhập. Đừng mở ra internet khi chưa tạo tài khoản (npm run adduser -- <tên> --admin).');
});

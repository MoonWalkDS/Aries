// Đồng bộ nhãn giữa nhiều máy qua git mà không bao giờ conflict.
//
// Mỗi lần viewer ghi một file nhãn, phần thay đổi (theo khoá) được ghi thêm một dòng vào
// annotations/journal/<người>@<máy>.jsonl. Mỗi máy chỉ ghi file của chính mình, nên pull/push không đụng nhau.
// Thư mục journal/ là một git worktree của nhánh mồ côi `label-journal`: đồng bộ không chạm nhánh code.
// Lúc mở app: pull nhánh đó, áp các dòng mới của mọi máy khác vào file nhãn (ai sửa sau thắng).
// Lúc tắt app (Ctrl+C), mỗi 5 phút, hoặc khi bấm "Đồng bộ": commit file của mình rồi push.
'use strict';
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync, execFile } = require('node:child_process');

const BRANCH = 'label-journal';
const SEP = '\u0000';
// Độ sâu tách khoá khi so file: 2 = {nhóm: {khoá: giá trị}}, còn lại 1 = {khoá: giá trị}.
const DEPTH = { 'custom_labels.json': 2, 'identity_names.json': 2, 'object_labels.json': 2 };
// Các file viewer ghi với khoá cấp 1 được sắp xếp (store.js:writeJsonAtomic): áp lại cũng sắp để git diff gọn.
const UNSORTED = new Set(['identity_names.json', 'object_labels.json', 'object_regions.json']);

const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const slug = (s) => String(s || '').normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/đ/gi, 'd').replace(/[^\w.-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'nguoi-gan';

// Danh sách thay đổi từ `a` sang `b`: [{path, v}] (đặt) hoặc [{path, d: 1}] (xoá).
function diff(a, b, depth, prefix = []) {
  const out = [];
  for (const k of new Set([...Object.keys(a || {}), ...Object.keys(b || {})])) {
    const x = a?.[k], y = b?.[k], p = [...prefix, k];
    const nest = (v) => v === undefined || isObj(v);
    if (depth > 1 && nest(x) && nest(y)) out.push(...diff(x || {}, y || {}, depth - 1, p));
    else if (y === undefined) { if (x !== undefined) out.push({ path: p, d: 1 }); }
    else if (JSON.stringify(x) !== JSON.stringify(y)) out.push({ path: p, v: y });
  }
  return out;
}

function applyOp(doc, op) {
  let node = doc;
  for (const k of op.path.slice(0, -1)) node = isObj(node[k]) ? node[k] : (node[k] = {});
  const leaf = op.path[op.path.length - 1];
  if (op.d) {
    delete node[leaf];
    if (op.path.length > 1 && !Object.keys(node).length) delete doc[op.path[0]];   // như store.js: bỏ nhóm rỗng
  } else node[leaf] = op.v;
}

function readJson(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch (e) { if (e.code === 'ENOENT') return fallback; throw e; }
}
function writeJson(file, data) {
  const tmp = `${file}.${process.pid}.tmp`;
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2), 'utf8');
  fs.renameSync(tmp, file);
}
const sortKeys = (o) => Object.fromEntries(Object.entries(o).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));

class Journal {
  // root: thư mục labeling/. dir: nơi chứa các file .jsonl. state: file trạng thái riêng của máy này (không đưa lên git).
  constructor({ root, dir, stateFile, user, host = os.hostname() }) {
    this.root = root;
    this.dir = dir;
    this.stateFile = stateFile;
    this.user = user;
    this.me = `${slug(user)}@${slug(host)}`;
    this.file = path.join(dir, `${this.me}.jsonl`);
    // applied: số dòng đã áp của từng file journal; lastAt: thời điểm sửa gần nhất của từng (file, khoá).
    this.state = { applied: {}, lastAt: {}, ...readJson(stateFile, {}) };
    this.pending = 0;   // số dòng của mình chưa push
  }

  _rel(file) { return path.relative(this.root, file).split(path.sep).join('/'); }
  _saveState() { writeJson(this.stateFile, this.state); }
  _lines(file) {
    if (!fs.existsSync(file)) return [];
    return fs.readFileSync(file, 'utf8').split('\n').filter(Boolean);
  }

  // Gọi ngay trước khi viewer ghi `file` với nội dung mới `data` (JSON) hoặc `text` (file chữ như outfits.yaml).
  record(file, { data, text }) {
    const rel = this._rel(file);
    if (rel.startsWith('..') || rel.startsWith('annotations/journal/')) return;
    const at = new Date().toISOString();
    let ops;
    if (text !== undefined) {
      const old = fs.existsSync(file) ? fs.readFileSync(file, 'utf8') : null;
      ops = old === text ? [] : [{ path: [], text }];
    } else {
      ops = diff(readJson(file, {}), JSON.parse(JSON.stringify(data)), DEPTH[path.basename(file)] || 1);
    }
    if (!ops.length) return;
    fs.mkdirSync(this.dir, { recursive: true });
    fs.appendFileSync(this.file, ops.map((op) => JSON.stringify({ at, by: this.user, file: rel, ...op })).join('\n') + '\n', 'utf8');
    const name = path.basename(this.file);
    this.state.applied[name] = (this.state.applied[name] || 0) + ops.length;
    for (const op of ops) this.state.lastAt[rel + SEP + op.path.join(SEP)] = at;
    this.pending += ops.length;
    this._saveState();
  }

  // Áp các dòng chưa áp của mọi file journal (kể cả của mình ghi từ máy khác cùng tên). Trả về số thay đổi đã áp.
  replay() {
    if (!fs.existsSync(this.dir)) return { applied: 0, skipped: 0, files: [] };
    const ops = [];
    for (const name of fs.readdirSync(this.dir).filter((f) => f.endsWith('.jsonl'))) {
      const lines = this._lines(path.join(this.dir, name));
      const from = this.state.applied[name] || 0;
      for (const line of lines.slice(from)) { try { ops.push(JSON.parse(line)); } catch { /* dòng ghi dở: bỏ */ } }
      this.state.applied[name] = lines.length;
    }
    ops.sort((a, b) => (a.at < b.at ? -1 : a.at > b.at ? 1 : 0));
    const docs = new Map();
    let applied = 0, skipped = 0;
    for (const op of ops) {
      const key = op.file + SEP + op.path.join(SEP);
      if (this.state.lastAt[key] && this.state.lastAt[key] >= op.at) { skipped++; continue; }   // máy này đã có bản mới hơn
      const file = path.join(this.root, op.file);
      if (path.relative(this.root, file).startsWith('..')) continue;
      if (op.text !== undefined) docs.set(file, { text: op.text });
      else {
        if (!docs.has(file) || docs.get(file).text !== undefined) docs.set(file, { data: readJson(file, {}) });
        applyOp(docs.get(file).data, op);
      }
      this.state.lastAt[key] = op.at;
      applied++;
    }
    for (const [file, d] of docs) {
      if (d.text !== undefined) { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, d.text, 'utf8'); }
      else writeJson(file, UNSORTED.has(path.basename(file)) ? d.data : sortKeys(d.data));
    }
    this._saveState();
    return { applied, skipped, files: [...docs.keys()].map((f) => this._rel(f)) };
  }
}

// ---------- git: nhánh mồ côi label-journal, check out ở annotations/journal/
class GitSync {
  constructor({ repoDir, dir, remote = 'origin', log = console.log }) {
    this.repoDir = repoDir;
    this.dir = dir;
    this.remote = remote;
    this.log = log;
    this.mode = 'off';   // 'remote' | 'local' (chưa có remote / mất mạng) | 'off' (không phải repo git)
    this.lastPush = null;
    this.lastError = null;
  }

  _git(args, cwd = this.dir, timeout = 60000) {
    return execFileSync('git', args, { cwd, timeout, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true }).trim();
  }
  _try(args, cwd) { try { return this._git(args, cwd); } catch { return null; } }

  // Tạo worktree nếu chưa có, rồi pull. Không bao giờ ném lỗi: mất mạng thì chạy tiếp trên máy.
  setup() {
    if (this._try(['rev-parse', '--git-dir'], this.repoDir) === null) { this.mode = 'off'; return this.status(); }
    const hasRemote = (this._try(['remote'], this.repoDir) || '').split('\n').includes(this.remote);
    if (!fs.existsSync(path.join(this.dir, '.git'))) {
      if (fs.existsSync(this.dir) && fs.readdirSync(this.dir).length) {
        this.lastError = `${this.dir} đã có file nhưng không phải worktree git: không đồng bộ được.`;
        this.mode = 'local';
        return this.status();
      }
      const remoteBranch = hasRemote && this._try(['fetch', this.remote, BRANCH], this.repoDir) !== null;
      if (remoteBranch) {
        if (this._try(['rev-parse', '--verify', BRANCH], this.repoDir) === null) this._git(['branch', BRANCH, `${this.remote}/${BRANCH}`], this.repoDir);
        this._git(['worktree', 'add', this.dir, BRANCH], this.repoDir);
      } else if (this._try(['rev-parse', '--verify', BRANCH], this.repoDir) !== null) {
        this._git(['worktree', 'add', this.dir, BRANCH], this.repoDir);
      } else {
        this._git(['worktree', 'add', '--orphan', '-b', BRANCH, this.dir], this.repoDir);
      }
      this.log(`Tạo thư mục đồng bộ nhãn ${path.relative(this.repoDir, this.dir)} (nhánh ${BRANCH}).`);
    }
    this.mode = hasRemote ? 'remote' : 'local';
    this.pull();
    return this.status();
  }

  pull() {
    if (this.mode !== 'remote') return false;
    if (this._try(['ls-remote', '--exit-code', '--heads', this.remote, BRANCH], this.dir) === null) return true;   // remote chưa có nhánh: lần push đầu tạo
    try {
      this._git(['pull', '--rebase', '--quiet', this.remote, BRANCH]);
      this.lastError = null;
      return true;
    } catch (e) { this.lastError = `Không pull được: ${firstLine(e)}`; this.log(this.lastError); return false; }
  }

  // Commit file journal của mình (nếu có thay đổi) rồi push. sync=true dùng lúc tắt app.
  push(file, { sync = false } = {}) {
    if (this.mode === 'off' || !fs.existsSync(file)) return Promise.resolve(false);
    const rel = path.basename(file);
    const run = () => {
      if (!this._try(['status', '--porcelain', '--', rel])) return this.mode === 'remote' ? this._pushOnly() : false;
      this._git(['add', '--', rel]);
      this._git(['commit', '--quiet', '-m', `Nhãn từ ${rel.replace(/\.jsonl$/, '')}`, '--', rel]);
      return this.mode === 'remote' ? this._pushOnly() : true;
    };
    if (sync) { try { return Promise.resolve(run()); } catch (e) { this.lastError = firstLine(e); return Promise.resolve(false); } }
    return new Promise((resolve) => setImmediate(() => { try { resolve(run()); } catch (e) { this.lastError = firstLine(e); this.log(`Không đẩy được nhãn: ${this.lastError}`); resolve(false); } }));
  }

  _pushOnly() {
    if (this._try(['rev-list', '--count', `${this.remote}/${BRANCH}..HEAD`]) === '0') return true;
    for (let i = 0; i < 2; i++) {
      try {
        this._git(['push', '--quiet', this.remote, `HEAD:${BRANCH}`]);
        this._try(['fetch', '--quiet', this.remote, BRANCH]);
        this.lastPush = new Date().toISOString();
        this.lastError = null;
        return true;
      } catch (e) {
        this.lastError = `Không push được: ${firstLine(e)}`;
        if (i === 0) this._try(['pull', '--rebase', '--quiet', this.remote, BRANCH]);   // người khác vừa push: lấy về rồi thử lại
      }
    }
    this.log(this.lastError);
    return false;
  }

  status() { return { mode: this.mode, branch: BRANCH, lastPush: this.lastPush, error: this.lastError }; }
}

const firstLine = (e) => String(e.stderr || e.message || e).trim().split('\n').filter(Boolean).pop() || 'lỗi git';

// Móc dùng chung cho store.js / objstore.js: không có journal (test, chạy trên bản sao) thì không làm gì.
let current = null;
const setJournal = (j) => { current = j; };
const beforeWrite = (file, payload) => { if (current) current.record(file, payload); };

module.exports = { Journal, GitSync, diff, applyOp, setJournal, beforeWrite, BRANCH };

'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { Journal, GitSync, diff, applyOp } = require('../journal');

const tmp = (p) => fs.mkdtempSync(path.join(os.tmpdir(), p));
const read = (f) => JSON.parse(fs.readFileSync(f, 'utf8'));
// Ghi như viewer: báo journal trước rồi mới ghi file.
function save(j, file, data) {
  j.record(file, { data });
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(data));
}

test('diff + applyOp tái tạo đúng file, tách khoá theo độ sâu', () => {
  const a = { g1: { k1: 1, k2: 2 }, g2: { k: 'x' }, top: [1] };
  const b = { g1: { k1: 1, k2: 3, k3: 4 }, top: [1, 2] };
  const ops = diff(a, b, 2);
  assert.deepEqual(ops.map((o) => o.path.join('.')).sort(), ['g1.k2', 'g1.k3', 'g2.k', 'top']);
  const doc = structuredClone(a);
  for (const op of ops) applyOp(doc, op);
  assert.deepEqual(doc, b);   // nhóm g2 rỗng bị bỏ, như store.js làm
});

test('hai máy ghi cùng thư mục journal: áp chéo, ai sửa sau thắng', async () => {
  const shared = tmp('journal-');
  const mk = (host) => {
    const root = tmp(`root-${host}-`);
    fs.mkdirSync(path.join(root, 'annotations', 'adl'), { recursive: true });
    fs.writeFileSync(path.join(root, 'annotations', 'adl', 'approved.json'), JSON.stringify({ base: { label: 'x' } }));
    return { root, j: new Journal({ root, dir: shared, stateFile: path.join(root, 'state.json'), user: 'An Nguyễn', host }) };
  };
  const A = mk('may-a'), B = mk('may-b');
  assert.equal(A.j.me, 'An-Nguyen@may-a');
  const fa = path.join(A.root, 'annotations', 'adl', 'approved.json'), fb = path.join(B.root, 'annotations', 'adl', 'approved.json');
  save(A.j, fa, { base: { label: 'x' }, img1: { label: 'eating', source: 'human' } });
  await new Promise((r) => setTimeout(r, 5));
  save(B.j, fb, { base: { label: 'x' }, img1: { label: 'cooking', source: 'human' }, img2: { label: 'sitting' } });
  // object_labels: hai người gán hai ảnh khác nhau trong cùng vùng
  save(A.j, path.join(A.root, 'annotations', 'object_labels.json'), { bàn: { '08_1': { items: ['cup'] } } });
  save(B.j, path.join(B.root, 'annotations', 'object_labels.json'), { bàn: { '08_2': { items: ['ấm'] } } });

  assert.equal(A.j.replay().applied, 3);   // nhận img1 + img2 của B, và vùng bàn/08_2
  assert.equal(B.j.replay().skipped, 1);   // img1 của A cũ hơn bản của B: bỏ
  assert.deepEqual(read(fa), read(fb));
  assert.equal(read(fa).img1.label, 'cooking');
  assert.deepEqual(read(path.join(A.root, 'annotations', 'object_labels.json')), { bàn: { '08_1': { items: ['cup'] }, '08_2': { items: ['ấm'] } } });
  assert.equal(A.j.replay().applied, 0);   // áp lại lần nữa không đổi gì
});

test('đồng bộ thật qua git: máy A push, máy B pull rồi áp', async (t) => {
  try { execFileSync('git', ['--version']); } catch { return t.skip('không có git'); }
  const git = (cwd, ...args) => execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'] });
  const remote = tmp('remote-');
  git(remote, 'init', '--bare', '--quiet');
  const clone = (name) => {
    const dir = tmp(`${name}-`);
    git(dir, 'init', '--quiet');
    git(dir, 'config', 'user.name', name);
    git(dir, 'config', 'user.email', `${name}@example.test`);
    fs.writeFileSync(path.join(dir, 'README'), 'x');
    git(dir, 'add', '.');
    git(dir, 'commit', '--quiet', '-m', 'init');
    git(dir, 'remote', 'add', 'origin', remote);
    const j = new Journal({ root: dir, dir: path.join(dir, 'annotations', 'journal'), stateFile: path.join(dir, 'state.json'), user: name, host: 'pc' });
    const s = new GitSync({ repoDir: dir, dir: j.dir, log: () => {} });
    return { dir, j, s };
  };
  const A = clone('an');
  assert.equal(A.s.setup().mode, 'remote');
  const fa = path.join(A.dir, 'annotations', 'custom_labels.json');
  save(A.j, fa, { people: { p1: { text: 'đang lau bàn' } } });
  assert.equal(await A.s.push(A.j.file), true);

  const B = clone('binh');
  B.s.setup();                                 // remote đã có nhánh: worktree lấy từ đó
  assert.equal(B.j.replay().applied, 1);
  assert.deepEqual(read(path.join(B.dir, 'annotations', 'custom_labels.json')), { people: { p1: { text: 'đang lau bàn' } } });
  save(B.j, path.join(B.dir, 'annotations', 'custom_labels.json'), { people: { p1: { text: 'đang lau bàn' }, p2: { text: 'rửa cốc' } } });
  assert.equal(await B.s.push(B.j.file), true);
  A.s.pull();
  assert.equal(A.j.replay().applied, 1);
  assert.equal(read(fa).people.p2.text, 'rửa cốc');
  assert.equal(git(A.dir, 'status', '--porcelain').includes('journal'), false);   // journal là worktree riêng, không bẩn nhánh code
});

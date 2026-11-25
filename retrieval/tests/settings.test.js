// node --test retrieval/tests/settings.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const S = require('../settings.js');

test('mặc định giữ hành vi cũ', () => {
  const d = S.defaults();
  assert.equal(d['grid.perPage'], 2);
  assert.equal(d['search.top'], 300);
  assert.equal(d['detail.span'], 6);
  assert.equal(d['submit.confirm'], true);
  assert.equal(d['poll.seconds'], 0);
});

test('JSON hỏng, khoá lạ, giá trị sai đều về mặc định', () => {
  assert.deepEqual(S.parse('{oops'), S.defaults());
  assert.deepEqual(S.parse('[1,2]'), S.defaults());
  const v = S.sanitize({ 'grid.perPage': 7, 'grid.columns': 3, 'level.fixedN': 0, 'submit.confirm': 'no', hack: 1, 'ui.theme': 'dark' });
  assert.equal(v['grid.perPage'], 2);
  assert.equal(v['grid.columns'], 3);
  assert.equal(v['level.fixedN'], 1);
  assert.equal(v['submit.confirm'], true);
  assert.equal(v['ui.theme'], 'dark');
  assert.equal('hack' in v, false);
});

test('nhập JSON báo số khoá bị bỏ qua, từ chối file sai', () => {
  assert.equal(S.importJson(JSON.stringify({ 'grid.perPage': 8, 'x': 1 })), 1);
  assert.equal(S.get('grid.perPage'), 8);
  assert.throws(() => S.importJson('nope'));
  assert.throws(() => S.importJson('[]'));
  S.reset();
  assert.equal(S.get('grid.perPage'), 2);
});

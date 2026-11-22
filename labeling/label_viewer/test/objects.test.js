'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { ObjectLabels, inVolume, boxInRegion } = require('../objstore');

// Store giả: chỉ những trường ObjectLabels dùng.
function fakeStore() {
  const jsons = fs.mkdtempSync(path.join(os.tmpdir(), 'objects-'));
  const images = [
    { image_id: '08_1', hour: '08', objects: [{ label: 'cup', box: [110, 150, 130, 195], score: 0.6 }, { label: 'cup', box: [500, 500, 520, 540], score: 0.9 }] },
    { image_id: '08_2', hour: '08', objects: [{ label: 'cup', box: [110, 150, 130, 195], score: 0.6 }, { label: 'kettle', box: [150, 120, 190, 190], score: 0.5 }] },
    { image_id: '08_3', hour: '08', objects: [] },
    { image_id: '09_1', hour: '09', objects: [] },
  ];
  return { jsons, images, byId: new Map(images.map((im) => [im.image_id, im])), actor: 'tester', undoStack: [] };
}
// Mặt bàn hình chữ nhật 100..200 x 180..200, đồ cao tới 80 px phía trên.
const table = { name: 'bàn', points: [[100, 180], [200, 180], [200, 200], [100, 200]], height: 80 };

test('khối = mặt đáy kéo lên theo chiều cao', () => {
  assert.equal(inVolume(table.points, 80, 150, 190), true);   // trên mặt bàn
  assert.equal(inVolume(table.points, 80, 150, 120), true);   // phía trên mặt, trong chiều cao
  assert.equal(inVolume(table.points, 80, 150, 90), false);   // cao quá
  assert.equal(inVolume(table.points, 80, 150, 210), false);  // dưới mặt bàn
  assert.equal(inVolume(table.points, 80, 250, 190), false);  // ngoài cạnh
  assert.equal(boxInRegion(table, [110, 150, 130, 195]), true);  // cốc đứng trên bàn
  assert.equal(boxInRegion(table, [500, 500, 520, 540]), false);
});

test('gợi ý, kế thừa trong giờ, bộ từ, hoàn tác', () => {
  const store = fakeStore();
  const o = new ObjectLabels(store);
  assert.deepEqual(o.doc.vocab.map((v) => v.name), ['cup', 'kettle']);   // bộ từ mồi từ máy dò
  o.setRegions([table]);
  assert.deepEqual(o.image('08_2').regions[0].hints.map((h) => h.label), ['cup', 'kettle']);
  assert.equal(o.next('bàn', '', 'unlabeled'), '08_1');

  o.setLabel({ region: 'bàn', image_id: '08_1', items: ['cup'], boxes: [{ label: 'khăn', box: [100, 160, 140, 200] }] });
  assert.deepEqual(o.effective('bàn', '08_1').items, ['cup', 'khăn']);   // tên của box tự vẽ được tick luôn
  assert.ok(o.doc.vocab.some((v) => v.name === 'khăn' && v.source === 'user'));
  assert.equal(o.effective('bàn', '08_3').from, '08_1');                  // kế thừa ảnh trước cùng giờ
  assert.equal(o.effective('bàn', '09_1'), null);                         // sang giờ khác thì không
  assert.equal(o.next('bàn', '08_1', 'change'), '08_2');                  // máy thấy thêm ấm
  assert.equal(o.next('bàn', '08_1', 'unlabeled'), '09_1');

  o.setLabel({ region: 'bàn', image_id: '08_2', items: [] });
  assert.equal(o.effective('bàn', '08_3').from, '08_2');
  o.undo(store.undoStack.pop());
  assert.equal(o.effective('bàn', '08_3').from, '08_1');
  o.undo(store.undoStack.pop());
  assert.equal(o.effective('bàn', '08_1'), null);
  assert.ok(!o.doc.vocab.some((v) => v.name === 'khăn'));                 // hoàn tác cả từ mới

  const saved = JSON.parse(fs.readFileSync(path.join(store.jsons, 'object_regions.json'), 'utf8'));
  assert.equal(saved.regions[0].height, 80);
  assert.throws(() => o.setLabel({ region: 'bàn', image_id: '08_1', items: ['<script>'] }), /không hợp lệ/);
  assert.throws(() => o.setRegions([{ name: 'x', points: [[0, 0], [1, 1]] }]), /3 điểm/);
});

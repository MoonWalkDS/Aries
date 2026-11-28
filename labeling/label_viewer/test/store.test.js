'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { Store, norm } = require('../store');

// Fixture nhỏ có đúng cấu trúc repo: <tmp>/labeling (root của Store) cạnh <tmp>/data.
function fixture() {
  const root = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'viewer-')), 'labeling');
  const w = (rel, data) => {
    const f = path.join(root, rel);
    fs.mkdirSync(path.dirname(f), { recursive: true });
    fs.writeFileSync(f, typeof data === 'string' ? data : JSON.stringify(data));
  };
  w('label_definitions/rois.yaml', 'rois:\n  - name: fridge\n    name_vi: tủ lạnh\n    box: [0.6, 0.1, 0.8, 0.6]\n    zone: [0.6, 0.1, 0.8, 0.6]\n');
  w('label_definitions/adl_labels.yaml', 'classes:\n  - {name: eating, name_vi: ăn, group: Food}\n  - {name: empty kitchen, name_vi: bếp trống, group: General}\n');
  w('label_definitions/adl_people_labels.yaml', 'classes:\n  - {name: opening the refrigerator, name_vi: mở tủ lạnh, group: Food}\n  - {name: sitting, name_vi: ngồi, group: Rest}\n');
  w('../data/intermediate/enrich/rois/fridge/labels.yaml', 'classes:\n  - {name: fridge closed, name_vi: tủ lạnh đóng}\n  - {name: fridge open, name_vi: tủ lạnh mở}\n');
  const img = (id, people, changed) => w(`annotations/enriched/${id}.json`, {
    image_id: id, hour: id.slice(0, 2), num_people: people.length,
    people: people.map((zone, i) => ({ id: `${id}_p0${i + 1}`, box: [0, 0, 10, 10], score: 0.9, zone })),
    rois: { fridge: { changed, motion: changed ? 0.5 : 0 } },
  });
  img('08_1', ['fridge'], true);
  img('08_2', [], false);
  img('09_3', ['island', 'fridge'], true);
  w('annotations/adl/approved.json', { '08_2': { label: 'empty kitchen', source: 'seed', round: 0, score: 0.2 } });
  w('annotations/adl/suggestions.json', { round: 1, items: [{ image_id: '09_3', label: 'eating', score: 0.4 }, { image_id: '08_2', label: 'eating', score: 0.9 }] });
  w('annotations/adl/rejected.json', {});
  w('annotations/adl_people/approved.json', { '08_1_p01': { label: 'opening the refrigerator', source: 'vlm', round: 3, score: null } });
  w('annotations/adl_people/suggestions.json', { round: 3, items: [{ image_id: '09_3_p01', label: 'sitting', score: 0.5 }] });
  w('annotations/adl_roi-fridge/approved.json', { '08_1': { label: 'fridge open', source: 'vlm' }, 'bg-08': { label: 'fridge closed', source: 'vlm' } });
  w('annotations/adl_roi-fridge/suggestions.json', { round: 3, items: [] });
  return root;
}

test('approved thắng suggestion; vùng không đổi lấy nhãn bg-HH', () => {
  const s = new Store(fixture());
  assert.equal(s.resolve('image', '08_2').label, 'empty kitchen');
  assert.equal(s.resolve('image', '09_3').source, 'suggested');
  const d = s.detail('08_2');
  assert.equal(d.rois[0].key, 'bg-08');
  assert.equal(d.rois[0].inherited, true);
  assert.equal(d.rois[0].state.label, 'fridge closed');
  assert.equal(d.rois[0].affected, 1);
});

test('lọc OR/AND, nguồn, chưa có nhãn, keyword bỏ dấu', () => {
  const s = new Store(fixture());
  const ids = (f) => s.query(f).map((r) => r.image_id);
  const fridgeOpen = { profile: 'roi-fridge', label: 'fridge open' };
  const sitting = { profile: 'people', label: 'sitting' };
  assert.deepEqual(ids({ tokens: [fridgeOpen, sitting], mode: 'or' }), ['08_1', '09_3']);
  assert.deepEqual(ids({ tokens: [{ profile: 'people', label: 'opening the refrigerator' }, fridgeOpen], mode: 'and' }), ['08_1']);
  assert.deepEqual(ids({ tokens: [sitting], sources: ['vlm'] }), []);
  assert.deepEqual(ids({ tokens: [{ profile: 'roi-fridge', label: '__none__' }] }), ['09_3']);
  assert.deepEqual(ids({ q: 'mo tu lanh' }), ['08_1']);
  assert.deepEqual(ids({ q: 'tu lanh -mo' }), ['08_2']);
  assert.deepEqual(ids({ q: '"lanh dong"' }), ['08_2']);
  assert.deepEqual(ids({ q: '"lanh tu"' }), []);
  assert.deepEqual(ids({ q: '"mo tu lanh"' }), ['08_1']);
  assert.equal(norm('Đảo Bếp'), 'dao bep');
});

test('ghi đúng định dạng apply_reviews, hoàn tác trả lại file cũ', () => {
  const root = fixture();
  const s = new Store(root);
  const file = path.join(root, 'annotations/adl/approved.json');
  const rejFile = path.join(root, 'annotations/adl/rejected.json');
  const before = fs.readFileSync(file, 'utf8');
  s.apply([{ profile: 'image', key: '09_3', action: 'set', label: 'empty kitchen' }]);
  s.actor = 'an';
  s.apply([{ profile: 'image', key: '09_3', action: 'set', label: 'empty kitchen' }]);
  const { at, ...rec } = JSON.parse(fs.readFileSync(file, 'utf8'))['09_3'];
  assert.deepEqual(rec, { label: 'empty kitchen', source: 'human', round: 1, score: 0.4, by: 'an' });
  assert.ok(!Number.isNaN(Date.parse(at)));
  assert.equal(s.undoStack.length, 1);                       // hoàn tác riêng: 'an' chỉ thấy bước của mình
  s.undo();
  s.actor = 'local';
  assert.ok(fs.readdirSync(path.dirname(file)).some((f) => f.startsWith('approved.json.bak-')));
  s.apply([{ profile: 'image', key: '08_2', action: 'reject' }]);
  assert.equal(JSON.parse(fs.readFileSync(file, 'utf8'))['08_2'], undefined);
  assert.deepEqual(JSON.parse(fs.readFileSync(rejFile, 'utf8'))['08_2'], ['empty kitchen']);
  // gợi ý "eating" của 08_2 chưa bị từ chối nên hiện ra
  assert.equal(s.resolve('image', '08_2').label, 'eating');
  assert.ok(s.edited.has('08_2'));
  s.undo();
  s.undo();
  assert.deepEqual(JSON.parse(fs.readFileSync(file, 'utf8')), JSON.parse(before));
  assert.deepEqual(JSON.parse(fs.readFileSync(rejFile, 'utf8')), {});
});

test('sửa bg-HH ảnh hưởng mọi ảnh cùng giờ; từ chối lớp lạ và file bị ghi đè', () => {
  const root = fixture();
  const s = new Store(root);
  assert.deepEqual(s.apply([{ profile: 'roi-fridge', key: 'bg-08', action: 'confirm' }]).changed, ['08_2']);
  assert.throws(() => s.apply([{ profile: 'people', key: '08_1_p01', action: 'set', label: 'flying' }]), /không thuộc/);
  assert.throws(() => s.apply([{ profile: 'people', key: '../x', action: 'reset' }]), /Khoá/);
  const f = path.join(root, 'annotations/adl_people/approved.json');
  fs.writeFileSync(f, '{}');
  fs.utimesSync(f, new Date(), new Date(Date.now() + 5000));
  assert.throws(() => s.apply([{ profile: 'people', key: '08_1_p01', action: 'reset' }]), (e) => e.status === 409);
});

test('box thủ công: thêm, sửa nhãn, xoá, hoàn tác, kiểm lớp', () => {
  const root = fixture();
  const s = new Store(root);
  const file = path.join(root, 'annotations/manual_boxes.json');
  const { id } = s.manualOp({ action: 'add', image_id: '08_1', box: [10, 20, 110, 220], kind: 'person' });
  assert.equal(id, '08_1_m01');
  s.manualOp({ action: 'update', image_id: '08_1', id, label: 'sitting' });
  let saved = JSON.parse(fs.readFileSync(file, 'utf8'))['08_1'][0];
  assert.deepEqual([saved.box, saved.kind, saved.label, saved.label_vi], [[10, 20, 110, 220], 'person', 'sitting', 'ngồi']);
  assert.throws(() => s.manualOp({ action: 'update', image_id: '08_1', id, label: 'flying' }), /không thuộc/);
  assert.throws(() => s.manualOp({ action: 'add', image_id: '08_1', box: [0, 0, 2, 2], kind: 'object' }), /quá nhỏ/);
  s.manualOp({ action: 'update', image_id: '08_1', id, kind: 'object', label: 'cái chảo' });
  assert.deepEqual(s.query({ q: 'cai chao' }).map((r) => r.image_id), ['08_1']);
  assert.deepEqual(s.query({ manual: true }).map((r) => r.image_id), ['08_1']);
  assert.equal(s.meta().manual.objectLabels[0], 'cái chảo');
  s.manualOp({ action: 'delete', image_id: '08_1', id });
  assert.equal(JSON.parse(fs.readFileSync(file, 'utf8'))['08_1'], undefined);
  s.undo();
  assert.equal(s.detail('08_1').manual[0].label, 'cái chảo');
});

test('nhãn tự gõ: thắng nhãn máy, đánh dấu nhãn máy sai, hoàn tác được, chọn lớp thì xoá', () => {
  const root = fixture();
  const s = new Store(root);
  s.apply([{ profile: 'people', key: '08_1_p01', action: 'custom', text: '  rót   nước ' }]);
  assert.deepEqual(s.resolve('people', '08_1_p01'), { label: 'rót nước', label_vi: 'rót nước', group: '', source: 'human', score: null, vlm: null, custom: true });
  const disk = JSON.parse(fs.readFileSync(path.join(root, 'annotations/custom_labels.json'), 'utf8'));
  assert.equal(disk.people['08_1_p01'].text, 'rót nước');
  assert.deepEqual(s.profiles.people.rejected['08_1_p01'], ['opening the refrigerator']);
  assert.equal(s.profiles.people.approved['08_1_p01'], undefined);
  assert.deepEqual(s.query({ q: 'rot nuoc' }).map((r) => r.image_id), ['08_1']);
  s.apply([{ profile: 'people', key: '08_1_p01', action: 'confirm' }]);
  assert.equal(s.resolve('people', '08_1_p01').custom, true);
  s.undo(); s.undo();
  assert.equal(s.resolve('people', '08_1_p01').label, 'opening the refrigerator');
  assert.equal(new Store(root).resolve('people', '08_1_p01').custom, undefined);
  s.apply([{ profile: 'people', key: '08_1_p01', action: 'custom', text: 'x' }]);
  s.apply([{ profile: 'people', key: '08_1_p01', action: 'set', label: 'sitting' }]);
  assert.equal(s.resolve('people', '08_1_p01').label, 'sitting');
  assert.deepEqual(JSON.parse(fs.readFileSync(path.join(root, 'annotations/custom_labels.json'), 'utf8')), {});
  assert.throws(() => s.apply([{ profile: 'people', key: '08_1_p01', action: 'custom', text: '  ' }]), /Nhập mô tả/);
});

test('danh tính: đặt tên cả cụm, tên riêng thắng, hoàn tác; đồ vật vào detail và tìm kiếm', () => {
  const root = fixture();
  fs.writeFileSync(path.join(root, 'annotations/identities.json'), JSON.stringify({ clusters: { c01: { size: 2, examples: ['08_1_p01'] } }, assign: { '08_1_p01': { cluster: 'c01', sim: 0.9 }, '09_3_p02': { cluster: 'c01', sim: 0.8 } } }));
  const e = path.join(root, 'annotations/enriched/08_1.json');
  fs.writeFileSync(e, JSON.stringify({ ...JSON.parse(fs.readFileSync(e, 'utf8')), objects: [{ id: '08_1_o01', label: 'cup', box: [1, 2, 3, 4], score: 0.8 }] }));
  const s = new Store(root);
  assert.equal(s.detail('08_1').objects[0].label, 'cup');
  assert.deepEqual(s.query({ q: 'cup' }).map((r) => r.image_id), ['08_1']);
  const r = s.setIdentity({ scope: 'cluster', cluster: 'c01', name: '  An ' });
  assert.deepEqual(r.changed.sort(), ['08_1', '09_3']);
  assert.equal(s.detail('09_3').people[1].identity.name, 'An');
  s.setIdentity({ scope: 'person', person: '09_3_p02', name: '' });
  assert.equal(s.identity('09_3_p02').name, null);
  assert.deepEqual(s.query({ q: 'an' }).map((x) => x.image_id).includes('08_1'), true);
  const names = JSON.parse(fs.readFileSync(path.join(root, 'annotations/identity_names.json'), 'utf8'));
  assert.deepEqual({ clusters: names.clusters, people: names.people }, { clusters: { c01: 'An' }, people: { '09_3_p02': '' } });
  assert.deepEqual(Object.keys(names.by).sort(), ['09_3_p02', 'cluster:c01']);
  s.undo(); s.undo();
  assert.equal(s.identity('08_1_p01').name, null);
  assert.throws(() => s.setIdentity({ scope: 'cluster', cluster: 'c99', name: 'x' }), /Không có cụm/);
});

test('bảng định danh: tóm tắt, thành viên theo cụm/tên, chuyển nhiều người một bước hoàn tác', () => {
  const root = fixture();
  fs.writeFileSync(path.join(root, 'annotations/identities.json'), JSON.stringify({ clusters: { c01: { size: 3, examples: [] } }, assign: { '08_1_p01': { cluster: 'c01', sim: 0.9 }, '09_3_p01': { cluster: 'c01', sim: 0.7 }, '09_3_p02': { cluster: 'c01', sim: 0.8 } } }));
  const s = new Store(root);
  s.setIdentity({ scope: 'cluster', cluster: 'c01', name: 'Người 01' });
  assert.deepEqual(s.identityMembers({ cluster: 'c01' }), []);                      // cụm đã có tên: mọi crop đã gán, rời cụm
  s.setIdentity({ scope: 'cluster', cluster: 'c01', name: '' });
  assert.deepEqual(s.identityMembers({ cluster: 'c01' }).map((x) => x.id), ['08_1_p01', '09_3_p02', '09_3_p01']);
  s.setIdentity({ scope: 'person', persons: ['09_3_p01', '09_3_p02'], name: 'Người 02' });
  assert.deepEqual(s.identityMembers({ cluster: 'c01' }).map((x) => x.id), ['08_1_p01']);   // người đã gán rời cụm
  assert.equal(s.identitySummary().clusters[0].left, 1);
  s.setIdentity({ scope: 'person', persons: ['08_1_p01'], name: '' });
  assert.deepEqual(s.identityMembers({ unknown: true }).map((x) => x.id), ['08_1_p01']);
  s.undo(); s.undo();
  s.setIdentity({ scope: 'cluster', cluster: 'c01', name: 'Người 01' });
  s.setIdentity({ scope: 'person', persons: ['09_3_p01', '09_3_p02'], name: 'Người 02' });
  assert.deepEqual(s.identitySummary().names, { 'Người 01': 1, 'Người 02': 2 });
  assert.deepEqual(s.identityMembers({ name: 'Người 02' }).map((x) => x.id).sort(), ['09_3_p01', '09_3_p02']);
  s.undo();
  assert.deepEqual(s.identitySummary().names, { 'Người 01': 3 });
  assert.throws(() => s.setIdentity({ scope: 'person', persons: ['nope'], name: 'x' }), /Không có người/);
});

test('đổi tên: mọi nơi kể cả outfits.yaml (giữ chú thích), gộp vào tên có sẵn, hoàn tác', () => {
  const root = fixture();
  fs.writeFileSync(path.join(root, 'annotations/identities.json'), JSON.stringify({ clusters: { c01: { size: 2, examples: [] } }, assign: { '08_1_p01': { cluster: 'c01', sim: 0.9 }, '09_3_p01': { cluster: 'c01', sim: 0.7 } } }));
  const outfits = path.join(root, 'annotations/outfits.yaml');
  fs.writeFileSync(outfits, '# ghi chú\npeople:\n  Người 01:\n    outfits:\n      - hours: all\n        top: sweater\n');
  const s = new Store(root);
  s.setIdentity({ scope: 'cluster', cluster: 'c01', name: 'Người 01' });
  s.setIdentity({ scope: 'person', persons: ['09_3_p02'], name: 'Người 02' });
  const r = s.renameIdentity({ from: 'Người 01', to: 'An' });
  assert.equal(r.renamed, 1);
  assert.equal(s.identity('08_1_p01').name, 'An');
  const text = fs.readFileSync(outfits, 'utf8');
  assert.ok(text.includes('# ghi chú') && text.includes('An:') && !text.includes('Người 01'));
  s.renameIdentity({ from: 'Người 02', to: 'An' });
  assert.deepEqual(s.identitySummary().names, { An: 3 });
  s.undo(); s.undo();
  assert.equal(s.identity('08_1_p01').name, 'Người 01');
  assert.ok(fs.readFileSync(outfits, 'utf8').includes('Người 01:'));
  assert.throws(() => s.renameIdentity({ from: 'Không có', to: 'X' }), /Không có ai tên/);
});

test('hàng khó trước: crop chưa gán, mô hình phân vân nhất trước', () => {
  const root = fixture();
  const pred = (margin) => ({ name: 'An', source: 'model', predicted: { name: 'An', prob: 0.5, margin } });
  const rec = (id, margins) => fs.writeFileSync(path.join(root, `annotations/enriched/${id}.json`), JSON.stringify({
    image_id: id, hour: id.slice(0, 2), num_people: margins.length, rois: {},
    people: margins.map((m, i) => ({ id: `${id}_p0${i + 1}`, box: [0, 0, 10, 10], score: 0.9, zone: null, identity: pred(m) })),
  }));
  rec('08_1', [0.4]);
  rec('09_3', [0.05, 0.2]);
  const s = new Store(root);
  assert.deepEqual(s.identityMembers({ hard: true }).map((x) => x.id), ['09_3_p01', '09_3_p02', '08_1_p01']);
  assert.equal(s.identitySummary().hard, 3);
  s.setIdentity({ scope: 'person', persons: ['09_3_p01'], name: 'Bình' });
  assert.deepEqual(s.identityMembers({ hard: true }).map((x) => x.id), ['09_3_p02', '08_1_p01']);   // đã gán thì rời hàng
  assert.equal(s.identitySummary().hard, 2);                                                  // số đếm giảm theo
});

test('lọc hành động lệch vùng: nhãn máy mâu thuẫn với vùng đứng, nhãn người duyệt thì bỏ qua', () => {
  const root = fixture();
  const approved = path.join(root, 'annotations/adl_people/approved.json');
  fs.writeFileSync(approved, JSON.stringify({
    '08_1_p01': { label: 'opening the refrigerator', source: 'vlm' },   // đứng ở fridge: khớp
    '09_3_p01': { label: 'opening the refrigerator', source: 'vlm' },   // đứng ở island: lệch
  }));
  let s = new Store(root);
  const rows = s.query({ conflict: true });
  assert.deepEqual(rows.map((r) => r.image_id), ['09_3']);
  assert.deepEqual(rows[0].hits.map((u) => u.id), ['09_3_p01']);
  fs.writeFileSync(approved, JSON.stringify({ '09_3_p01': { label: 'opening the refrigerator', source: 'human' } }));
  s = new Store(root);
  assert.deepEqual(s.query({ conflict: true }), []);                     // người đã duyệt: không đưa lại vào hàng
});

test('không phải người: rời mọi danh sách, đếm riêng, hoàn tác được', () => {
  const root = fixture();
  fs.writeFileSync(path.join(root, 'annotations/identities.json'), JSON.stringify({ clusters: { c01: { size: 2, examples: [] } }, assign: { '08_1_p01': { cluster: 'c01', sim: 0.9 }, '09_3_p01': { cluster: 'c01', sim: 0.7 } } }));
  const s = new Store(root);
  s.setIdentity({ scope: 'person', persons: ['08_1_p01'], notPerson: true });
  assert.deepEqual(s.identityMembers({ cluster: 'c01' }).map((x) => x.id), ['09_3_p01']);
  assert.equal(s.identitySummary().notPerson, 1);
  const saved = JSON.parse(fs.readFileSync(path.join(root, 'annotations/identity_names.json'), 'utf8'));
  assert.deepEqual(saved.not_person, { '08_1_p01': true });
  s.setIdentity({ scope: 'person', persons: ['08_1_p01'], name: 'An' });                 // gán tên lại thì bỏ dấu
  assert.equal(s.identitySummary().notPerson, 0);
  s.undo();
  assert.equal(s.identitySummary().notPerson, 1);
});

test('sai, gán lại sau: vào hàng recheck, gán tên thì rời hàng, hoàn tác được', () => {
  const root = fixture();
  const s = new Store(root);
  s.setIdentity({ scope: 'person', persons: ['08_1_p01'], recheck: 'An' });
  assert.deepEqual(s.identityMembers({ recheck: true }).map((x) => [x.id, x.rejected]), [['08_1_p01', 'An']]);
  assert.equal(s.identitySummary().recheck, 1);
  assert.notEqual(s.identity('08_1_p01')?.source, 'person');                   // chưa đặt tên riêng
  s.setIdentity({ scope: 'person', persons: ['08_1_p01'], name: 'Bình' });
  assert.deepEqual(s.identityMembers({ recheck: true }), []);
  assert.equal(s.identitySummary().recheck, 0);
  s.undo();
  assert.equal(s.identityMembers({ recheck: true }).length, 1);
});

test('hàng duyệt hành động: theo lớp, bác thì sang "cần gán lại", đếm nhãn người duyệt', () => {
  const root = fixture();
  fs.writeFileSync(path.join(root, 'annotations/adl_people/approved.json'), JSON.stringify({
    '08_1_p01': { label: 'opening the refrigerator', source: 'vlm' },
    '09_3_p01': { label: 'opening the refrigerator', source: 'vlm' },
  }));
  const s = new Store(root);
  assert.deepEqual(s.actionMembers({ label: 'opening the refrigerator' }).map((x) => x.id), ['08_1_p01', '09_3_p01']);
  assert.deepEqual(s.actionMembers({ conflict: true }).map((x) => x.id), ['09_3_p01']);    // đứng ở island
  s.apply([{ profile: 'people', key: '08_1_p01', action: 'confirm' }, { profile: 'people', key: '09_3_p01', action: 'reject' }]);
  const sum = s.actionSummary();
  assert.deepEqual([sum.human, sum.recheck, sum.pending['opening the refrigerator'] || 0], [{ 'opening the refrigerator': 1 }, 1, 0]);   // 09_3_p01 giờ hiện gợi ý "sitting" của máy
  assert.deepEqual(s.actionMembers({ recheck: true }).map((x) => [x.id, x.rejected]), [['09_3_p01', ['opening the refrigerator']]]);
});

test('hàng nghi sai (gộp bằng chứng) và hàng có thể box thừa', () => {
  const root = fixture();
  fs.writeFileSync(path.join(root, 'annotations/adl_people/approved.json'), JSON.stringify({ '08_1_p01': { label: 'opening the refrigerator', source: 'vlm' } }));
  fs.writeFileSync(path.join(root, 'annotations/adl_people/action_pred.json'), JSON.stringify({ '08_1_p01': { label: 'sitting', gap: 2 } }));
  fs.writeFileSync(path.join(root, 'annotations/adl_people/extra_boxes.json'), JSON.stringify({ '09_3_p02': { score: 0.5 }, '09_3_p01': { score: 0.4 } }));
  const s = new Store(root);
  assert.deepEqual(s.actionMembers({ suspect: true }).map((x) => [x.id, x.machine, x.label]), [['08_1_p01', 'opening the refrigerator', 'sitting']]);
  assert.equal(s.actionSummary().suspect, 1);
  assert.deepEqual(s.identityMembers({ extra: true }).map((x) => x.id), ['09_3_p01', '09_3_p02']);   // điểm dò thấp trước
  s.setIdentity({ scope: 'person', persons: ['09_3_p01'], name: 'An' });                            // đã gán tên: là người thật
  assert.equal(s.identitySummary().extra, 1);
});

test('hàng duyệt hỏi theo nhãn gộp đã thay nhãn VLM; X bác cả nhãn đang hiện', () => {
  const root = fixture();
  const f = path.join(root, 'annotations/enriched/09_3.json');
  const im = JSON.parse(fs.readFileSync(f, 'utf8'));
  im.people[0].action = { label: 'sitting', source: 'fused', machine_label: 'opening the refrigerator' };
  fs.writeFileSync(f, JSON.stringify(im));
  fs.writeFileSync(path.join(root, 'annotations/adl_people/approved.json'), JSON.stringify({
    '08_1_p01': { label: 'opening the refrigerator', source: 'vlm' },
    '09_3_p01': { label: 'opening the refrigerator', source: 'vlm' },
  }));
  const s = new Store(root);
  assert.deepEqual(s.actionMembers({ label: 'opening the refrigerator' }).map((x) => x.id), ['08_1_p01']);
  assert.deepEqual(s.actionMembers({ label: 'sitting' }).map((x) => [x.id, x.source]), [['09_3_p01', 'fused']]);
  assert.equal(s.actionSummary().pending.sitting, 1);
  s.apply([{ profile: 'people', key: '09_3_p01', action: 'reject', label: 'sitting' }]);
  assert.deepEqual(s.profiles.people.rejected['09_3_p01'], ['opening the refrigerator', 'sitting']);
  assert.throws(() => s.apply([{ profile: 'people', key: '08_1_p01', action: 'reject', label: 'flying' }]), /không thuộc/);
});

test('hàng "Thầy tả": người câu của thầy tả, X chỉ bác gợi ý và giữ nhãn máy', () => {
  const root = fixture();
  fs.writeFileSync(path.join(root, 'annotations/adl_people/approved.json'), JSON.stringify({ '09_3_p02': { label: 'sitting', source: 'vlm' } }));
  fs.writeFileSync(path.join(root, 'annotations/adl_people/teacher_hints.json'), JSON.stringify({
    '09_3_p02': { label: 'opening the refrigerator', text: 'another opens the fridge' }, '08_1_p01': { label: 'sitting', text: 'a man is seated' } }));
  const s = new Store(root);
  assert.deepEqual(s.actionMembers({ teacher: 'opening the refrigerator' }).map((x) => [x.id, x.label, x.machine, x.teacher]),
    [['09_3_p02', 'opening the refrigerator', 'sitting', 'another opens the fridge']]);
  assert.deepEqual(s.actionSummary().teacher, { 'opening the refrigerator': 1, sitting: 1 });
  s.apply([{ profile: 'people', key: '09_3_p02', action: 'reject', label: 'opening the refrigerator', only: true }]);
  assert.equal(s.resolve('people', '09_3_p02').label, 'sitting');                     // nhãn máy giữ nguyên
  assert.deepEqual(s.actionMembers({ teacher: 'opening the refrigerator' }), []);
  assert.equal(s.actionSummary().teacher['opening the refrigerator'], undefined);
});

test('box người tự vẽ: đặt tên như người, mô hình đoán từ lần gộp, đổi sang vật thể hay xoá thì rời bảng định danh', () => {
  const root = fixture();
  const f = path.join(root, 'annotations/enriched/08_1.json');
  const rec = JSON.parse(fs.readFileSync(f, 'utf8'));
  rec.manual = [{ id: '08_1_m01', kind: 'person', box: [1, 2, 3, 4], label: null, identity: { name: 'An', source: 'model', predicted: { name: 'An', prob: 0.9, margin: 0.8 } } }];
  fs.writeFileSync(f, JSON.stringify(rec));
  fs.writeFileSync(path.join(root, 'annotations/manual_boxes.json'), JSON.stringify({ '08_1': [{ id: '08_1_m01', kind: 'person', box: [1, 2, 3, 4] }] }));
  const s = new Store(root);
  assert.equal(s.finalIdentity('08_1_m01').name, 'An');                               // tên mô hình đoán lúc gộp
  assert.equal(s.detail('08_1').manual[0].predicted.name, 'An');
  assert.deepEqual(s.identityMembers({ name: 'An' }).map((x) => x.id), ['08_1_m01']);
  const { id } = s.manualOp({ action: 'add', image_id: '08_1', box: [10, 20, 110, 220], kind: 'person' });
  s.setIdentity({ scope: 'person', person: id, name: 'Bình' });                         // box vừa vẽ, chưa gộp: vẫn đặt tên được
  assert.equal(s.detail('08_1').manual[1].identity.name, 'Bình');
  assert.equal(s.identitySummary().names['Bình'], 1);
  s.manualOp({ action: 'update', image_id: '08_1', id, kind: 'object', label: 'cái chảo' });
  assert.equal(s.identitySummary().names['Bình'], undefined);                          // vật thể không phải người
  s.undo();
  assert.equal(s.identitySummary().names['Bình'], 1);
  s.manualOp({ action: 'delete', image_id: '08_1', id: '08_1_m01' });
  assert.throws(() => s.setIdentity({ scope: 'person', person: '08_1_m01', name: 'x' }), /Không có người/);
});

// Gán đồ vật theo vùng: người duyệt vẽ vùng (đa giác + chiều cao) một lần cho cả bộ ảnh (camera cố định),
// rồi tick những đồ vật đang có trong từng vùng. Nhãn của một ảnh được kế thừa cho các ảnh sau cùng giờ
// cho tới ảnh kế tiếp có nhãn riêng ("tick vùng": chỉ cần tick lại khi vùng thay đổi).
//
// annotations/object_regions.json : {size: [w, h], regions: [{name, points: [[x,y]...], height}], vocab: [{name, source}]}
// annotations/object_labels.json  : {<vùng>: {<mã ảnh>: {items: [tên], boxes: [{label, box}], by, at}}}
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { beforeWrite } = require('./journal');

const NAME = /^[\p{L}\p{N} _.'()-]{1,60}$/u;

function readJson(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch (e) {
    if (e.code === 'ENOENT') return fallback;
    throw new Error(`Không đọc được ${file}: ${e.message}`);
  }
}
function writeJson(file, data) {
  beforeWrite(file, { data });   // đồng bộ sang máy khác (journal.js)
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(data, null, 2), 'utf8');
  fs.renameSync(tmp, file);
}
const mtime = (file) => { try { return fs.statSync(file).mtimeMs; } catch { return 0; } };
const bad = (msg, status = 400) => Object.assign(new Error(msg), { status });

// Các đoạn y (trên một đường thẳng đứng x) nằm trong đa giác, theo quy tắc chẵn-lẻ.
function yIntervals(points, x) {
  const ys = [];
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [xi, yi] = points[i], [xj, yj] = points[j];
    if ((xi > x) !== (xj > x)) ys.push(yi + ((x - xi) * (yj - yi)) / (xj - xi));
  }
  ys.sort((a, b) => a - b);
  const out = [];
  for (let k = 0; k + 1 < ys.length; k += 2) out.push([ys[k], ys[k + 1]]);
  return out;
}

// Điểm nằm trong "khối" = đa giác mặt đáy kéo thẳng lên `height` px: tồn tại t ∈ [0, height] để (x, y + t) trong đa giác.
function inVolume(points, height, x, y) {
  return yIntervals(points, x).some(([a, b]) => y <= b && y + height >= a);
}

// Đồ vật thuộc vùng khi chân box (giữa cạnh dưới) nằm trong khối, hoặc ít nhất một nửa box nằm trong khối
// (đồ treo như khăn trên cửa lò: chân box có thể lòi xuống dưới mặt vùng).
function boxInRegion(region, box) {
  const [x1, y1, x2, y2] = box;
  if (inVolume(region.points, region.height, (x1 + x2) / 2, y2)) return true;
  let hit = 0;
  const N = 6;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    if (inVolume(region.points, region.height, x1 + ((i + 0.5) * (x2 - x1)) / N, y1 + ((j + 0.5) * (y2 - y1)) / N)) hit++;
  }
  return hit * 2 >= N * N;
}

class ObjectLabels {
  constructor(store) {
    this.store = store;
    this.regionsFile = path.join(store.jsons, 'object_regions.json');
    this.labelsFile = path.join(store.jsons, 'object_labels.json');
    this.doc = { size: [1280, 720], regions: [], vocab: [], ...readJson(this.regionsFile, {}) };
    this.labels = readJson(this.labelsFile, {});
    this.mtimes = { regions: mtime(this.regionsFile), labels: mtime(this.labelsFile) };
    this.backedUp = { regions: false, labels: false };
    // Bộ từ ban đầu = các lớp máy dò được trong bộ ảnh, để người gán không phải gõ lại.
    const known = new Set(this.doc.vocab.map((v) => v.name));
    for (const label of [...new Set(store.images.flatMap((im) => (im.objects || []).map((o) => o.label)))].sort()) {
      if (!known.has(label)) { this.doc.vocab.push({ name: label, source: 'detector' }); known.add(label); }
    }
    for (const byImage of Object.values(this.labels)) for (const e of Object.values(byImage)) {
      for (const n of [...(e.items || []), ...(e.boxes || []).map((b) => b.label)]) if (!known.has(n)) { this.doc.vocab.push({ name: n, source: 'user' }); known.add(n); }
    }
    this.ids = store.images.map((im) => im.image_id);
    this.index = new Map(this.ids.map((id, i) => [id, i]));
  }

  region(name) {
    const r = this.doc.regions.find((x) => x.name === name);
    if (!r) throw bad(`Không có vùng "${name}".`);
    return r;
  }

  // Box máy dò rơi vào vùng, gộp theo tên lớp (giữ điểm cao nhất).
  hints(region, im) {
    const best = new Map();
    for (const o of im.objects || []) {
      if (!boxInRegion(region, o.box)) continue;
      const cur = best.get(o.label);
      if (!cur || o.score > cur.score) best.set(o.label, { label: o.label, score: o.score, box: o.box, source: o.source });
    }
    return [...best.values()].sort((a, b) => b.score - a.score);
  }

  // Nhãn hiệu lực: nhãn riêng của ảnh, không có thì kế thừa ảnh gần nhất phía trước cùng giờ.
  effective(regionName, id) {
    const own = this.labels[regionName] || {};
    if (own[id]) return { ...own[id], from: id };
    const hour = this.store.byId.get(id)?.hour;
    for (let i = this.index.get(id) - 1; i >= 0; i--) {
      const prev = this.ids[i];
      if (this.store.byId.get(prev).hour !== hour) break;
      if (own[prev]) return { ...own[prev], from: prev };
    }
    return null;
  }

  meta() {
    const stats = Object.fromEntries(this.doc.regions.map((r) => [r.name, Object.keys(this.labels[r.name] || {}).length]));
    return { ...this.doc, ids: this.ids, stats, images: this.ids.length };
  }

  image(id) {
    const im = this.store.byId.get(id);
    if (!im) return null;
    const i = this.index.get(id);
    return {
      image_id: id, hour: im.hour, pos: i, prev: this.ids[i - 1] ?? null, next: this.ids[i + 1] ?? null,
      objects: im.objects || [],
      regions: this.doc.regions.map((r) => ({ name: r.name, label: this.effective(r.name, id), hints: this.hints(r, im) })),
    };
  }

  // Ảnh kế tiếp (sau `from`) cần xem cho vùng: 'unlabeled' = chưa có nhãn hiệu lực,
  // 'change' = tập đồ máy thấy khác ảnh trước đó (gợi ý vùng vừa thay đổi), hoặc chưa có nhãn.
  next(regionName, from, kind, dir = 1) {
    const r = this.region(regionName);
    const key = (im) => this.hints(r, im).map((h) => h.label).sort().join('|');
    let i = (this.index.get(from) ?? -1) + dir;
    for (; i >= 0 && i < this.ids.length; i += dir) {
      const id = this.ids[i];
      if (!this.effective(regionName, id)) return id;
      if (kind === 'change' && i > 0) {
        const im = this.store.byId.get(id), prev = this.store.byId.get(this.ids[i - 1]);
        if (im.hour === prev.hour && !this.labels[regionName]?.[id] && key(im) !== key(prev)) return id;
      }
    }
    return null;
  }

  _fresh(k) {
    const file = k === 'regions' ? this.regionsFile : this.labelsFile;
    if (mtime(file) !== this.mtimes[k]) throw bad(`${path.basename(file)} đã bị chương trình khác ghi đè. Bấm "Tải lại" rồi sửa tiếp.`, 409);
  }

  _save(k) {
    const file = k === 'regions' ? this.regionsFile : this.labelsFile;
    if (!this.backedUp[k] && fs.existsSync(file)) fs.copyFileSync(file, `${file}.bak-${new Date().toISOString().replace(/[:.]/g, '-')}`);
    this.backedUp[k] = true;
    writeJson(file, k === 'regions' ? this.doc : this.labels);
    this.mtimes[k] = mtime(file);
  }

  _name(s) {
    const name = String(s ?? '').trim().replace(/\s+/g, ' ');
    if (!NAME.test(name)) throw bad(`Tên "${name}" không hợp lệ (1–60 ký tự, chữ, số, khoảng trắng).`);
    return name;
  }

  _addVocab(names) {
    const known = new Set(this.doc.vocab.map((v) => v.name));
    const added = names.filter((n) => !known.has(n));
    for (const n of added) this.doc.vocab.push({ name: n, source: 'user', by: this.store.actor });
    return added.length > 0;
  }

  // Ghi đè toàn bộ danh sách vùng (trang "Sửa vùng" gửi cả danh sách).
  setRegions(regions) {
    if (!Array.isArray(regions)) throw bad('regions phải là danh sách.');
    const [W, H] = this.doc.size;
    const seen = new Set();
    const clean = regions.map((r) => {
      const name = this._name(r.name);
      if (seen.has(name)) throw bad(`Trùng tên vùng "${name}".`);
      seen.add(name);
      const points = (r.points || []).map((p) => [Math.round(Math.min(W, Math.max(0, p[0])) * 10) / 10, Math.round(Math.min(H, Math.max(0, p[1])) * 10) / 10]);
      if (points.length < 3 || !points.flat().every(Number.isFinite)) throw bad(`Vùng "${name}" cần ít nhất 3 điểm.`);
      const height = Math.round(Math.min(H, Math.max(0, Number(r.height) || 0)));
      return { name, points, height };
    });
    this._fresh('regions');
    const prev = structuredClone(this.doc);
    this.doc.regions = clean;
    this._save('regions');
    this.store.undoStack.push({ obj: 'regions', prev });
    return this.meta();
  }

  // Đặt nhãn riêng cho (vùng, ảnh). items: tên đồ đang có; boxes: box tự vẽ cho đồ máy không thấy.
  setLabel({ region, image_id: id, items, boxes }) {
    this.region(region);
    if (!this.store.byId.has(id)) throw bad(`Không có ảnh ${id}.`);
    if (!Array.isArray(items)) throw bad('items phải là danh sách.');
    const names = [...new Set(items.map((s) => this._name(s)))];
    const [W, H] = this.doc.size;
    const cleanBoxes = (boxes || []).map((b) => {
      const box = (b.box || []).map(Number);
      if (box.length !== 4 || !box.every(Number.isFinite) || box[2] - box[0] < 4 || box[3] - box[1] < 4) throw bad('Box phải là [x1, y1, x2, y2] và không quá nhỏ.');
      const [x1, y1, x2, y2] = box.map((v, k) => Math.round(Math.min(k % 2 ? H : W, Math.max(0, v)) * 10) / 10);
      return { label: this._name(b.label), box: [x1, y1, x2, y2] };
    });
    for (const b of cleanBoxes) if (!names.includes(b.label)) names.push(b.label);
    this._fresh('labels');
    const vocabPrev = structuredClone(this.doc.vocab);
    const vocabChanged = this._addVocab(names);
    if (vocabChanged) { try { this._fresh('regions'); } catch (e) { this.doc.vocab = vocabPrev; throw e; } }
    const prev = this.labels[region]?.[id];
    (this.labels[region] ||= {})[id] = { items: names, boxes: cleanBoxes, by: this.store.actor, at: new Date().toISOString() };
    this._save('labels');
    if (vocabChanged) this._save('regions');
    this.store.undoStack.push({ obj: 'label', region, id, prev, vocabPrev: vocabChanged ? vocabPrev : null });
    return { label: this.effective(region, id), vocab: this.doc.vocab };
  }

  // Xoá nhãn riêng: ảnh quay về kế thừa ảnh trước.
  clearLabel({ region, image_id: id }) {
    this.region(region);
    if (!this.labels[region]?.[id]) return { label: this.effective(region, id) };
    this._fresh('labels');
    const prev = this.labels[region][id];
    delete this.labels[region][id];
    this._save('labels');
    this.store.undoStack.push({ obj: 'label', region, id, prev });
    return { label: this.effective(region, id) };
  }

  undo(step) {
    if (step.obj === 'regions') {
      this._fresh('regions');
      this.doc = step.prev;
      this._save('regions');
      return { changed: [], left: this.store.undoStack.length, objects: true };
    }
    this._fresh('labels');
    if (step.prev) (this.labels[step.region] ||= {})[step.id] = step.prev;
    else delete this.labels[step.region]?.[step.id];
    this._save('labels');
    if (step.vocabPrev) { this.doc.vocab = step.vocabPrev; this._save('regions'); }
    return { changed: [step.id], left: this.store.undoStack.length, objects: true };
  }

  // Nhãn hiệu lực của mọi (ảnh, vùng): mỗi dòng một ảnh.
  exportRows() {
    return this.ids.map((id) => ({
      image_id: id,
      regions: Object.fromEntries(this.doc.regions.map((r) => {
        const e = this.effective(r.name, id);
        return [r.name, e ? { items: e.items, boxes: e.boxes, from: e.from } : null];
      })),
    }));
  }
}

module.exports = { ObjectLabels, inVolume, boxInRegion, yIntervals };

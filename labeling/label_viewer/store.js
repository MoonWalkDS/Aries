// Nạp nhãn hiện có của pipeline vào RAM, gộp theo đúng quy tắc của pipeline/modules/enrich.py,
// lọc, và ghi lại quyết định của người duyệt theo đúng định dạng modules/adl.py:apply_reviews.
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const YAML = require('yaml');
const { ObjectLabels } = require('./objstore');
const { beforeWrite } = require('./journal');
const HARD_LIMIT = 300;   // hàng "khó trước": đủ cho một lượt duyệt, chạy lại enrich identities rồi duyệt tiếp
// Hàng "hành động lệch vùng": hành động gắn với một chỗ mà người lại đứng vùng khác thì nhiều khả năng nhãn sai
// (soát 8 ca "mở tủ lạnh" lệch vùng: 7 ca nhãn sai). Không có "vứt rác": zone trash nằm trọn trong zone fridge nên gần như không ai được gán.
const ACTION_ZONE = { 'opening the refrigerator': 'fridge', 'washing dishes': 'sink', 'cooking at the stove': 'stove', 'making a hot drink': 'coffee_corner' };

const BASELINE_PREFIX = 'bg-'; // modules/enrich.py:43
const NONE = '__none__'; // token "chưa có nhãn"
const SAFE = /^[\w-]+$/;

function readJson(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); } catch (e) {
    if (e.code === 'ENOENT') return fallback;
    throw new Error(`Không đọc được ${file}: ${e.message}`);
  }
}

function writeJsonAtomic(file, data) {
  const sorted = Object.fromEntries(Object.entries(data).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0)));
  beforeWrite(file, { data: sorted });   // ghi nhật ký để đồng bộ sang máy khác (journal.js)
  const tmp = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(tmp, JSON.stringify(sorted, null, 2), 'utf8');
  fs.renameSync(tmp, file);
}

const mtime = (file) => { try { return fs.statSync(file).mtimeMs; } catch { return 0; } };
const clone = (v) => (v === undefined ? undefined : JSON.parse(JSON.stringify(v)));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const IMAGE_SIZE = [1280, 720];   // mọi ảnh Castle5000 cùng cỡ; dùng để đặt vùng cho box tự vẽ chưa gộp
const MIN_ZONE_OVERLAP = 0.35;    // config.yaml enrich.person.min_zone_overlap
const imageOfPid = (pid) => String(pid).replace(/_[pm]\d+$/, '');
const bad = (msg, status = 400) => Object.assign(new Error(msg), { status });

function loadClasses(file) {
  if (!fs.existsSync(file)) return [];
  const doc = YAML.parse(fs.readFileSync(file, 'utf8')) || {};
  return (doc.classes || []).map((c) => ({ name: c.name, name_vi: c.name_vi || '', group: c.group || '' }));
}

// Bỏ dấu tiếng Việt + chữ thường, để "tu lanh" khớp "tủ lạnh".
function norm(s) {
  return String(s ?? '').normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/đ/g, 'd').replace(/Đ/g, 'd').toLowerCase();
}

function sortKey(id) {
  const [h, n] = id.split('_');
  return Number(h) * 1e6 + Number(n);
}

class Store {
  // Người đang thao tác: server đặt ngay trước mỗi lệnh ghi (các lệnh ghi chạy đồng bộ nên không lẫn người).
  actor = 'local';

  // Hoàn tác riêng cho từng người: Z chỉ huỷ thao tác của chính mình. Gán lại = xoá hết lịch sử (khi nạp lại).
  get undoStack() {
    if (!this._stacks) this._stacks = new Map();
    if (!this._stacks.has(this.actor)) this._stacks.set(this.actor, []);
    return this._stacks.get(this.actor);
  }
  set undoStack(_) { this._stacks = new Map(); }

  _by() { return { by: this.actor, at: new Date().toISOString() }; }

  constructor(root) {
    this.root = root; // thư mục labeling/
    this.jsons = path.join(root, 'annotations');
    this.data = path.join(root, '..', 'data', 'intermediate');
    this.dataset = path.join(root, '..', 'data', 'dataset'); // ảnh gốc + thumbs, dùng chung với retrieval/
    this.undoStack = [];
    this.edited = new Set();
    this.reload();
  }

  reload() {
    const enrichedDir = path.join(this.jsons, 'enriched');
    if (!fs.existsSync(enrichedDir)) throw new Error(`Chưa có ${enrichedDir}. Chạy \`python main.py enrich merge\` trước.`);
    this.images = fs.readdirSync(enrichedDir).filter((f) => f.endsWith('.json'))
      .map((f) => readJson(path.join(enrichedDir, f)))
      .sort((a, b) => sortKey(a.image_id) - sortKey(b.image_id));
    this.byId = new Map(this.images.map((im) => [im.image_id, im]));
    this.personToImage = new Map();
    this.personById = new Map();
    for (const im of this.images) for (const p of im.people) { this.personToImage.set(p.id, im.image_id); this.personById.set(p.id, p); }
    this.hours = [...new Set(this.images.map((im) => im.hour))].sort();

    const roiDoc = YAML.parse(fs.readFileSync(path.join(this.root, 'label_definitions', 'rois.yaml'), 'utf8')) || {};
    this.rois = (roiDoc.rois || []).map((r) => ({ name: r.name, name_vi: r.name_vi || '', box: r.box, zone: r.zone }));
    this.zoneVi = Object.fromEntries(this.rois.map((r) => [r.name, r.name_vi]));

    this.profiles = {};
    const add = (name, layer, dirName, classesFile, roi) => {
      const dir = path.join(this.jsons, dirName);
      if (!fs.existsSync(dir)) return;
      this.profiles[name] = { name, layer, roi, dir, classesFile, ...this._readState(dir), classes: loadClasses(classesFile) };
    };
    add('image', 'image', 'adl', path.join(this.root, 'label_definitions', 'adl_labels.yaml'));
    add('people', 'people', 'adl_people', path.join(this.root, 'label_definitions', 'adl_people_labels.yaml'));
    for (const r of this.rois) add(`roi-${r.name}`, 'roi', `adl_roi-${r.name}`, path.join(this.data, 'enrich', 'rois', r.name, 'labels.yaml'), r.name);
    for (const p of Object.values(this.profiles)) p.classByName = new Map(p.classes.map((c) => [c.name, c]));
    // Box người dùng tự vẽ. Để riêng, không ghi vào approved.json: pipeline Python tra khoá theo
    // embedding nên một id lạ ở đó sẽ làm hỏng `adl update`.
    this.manualFile = path.join(this.jsons, 'manual_boxes.json');
    this.manual = readJson(this.manualFile, {});
    this.manualMtime = mtime(this.manualFile);
    this.manualBackedUp = false;
    // Nhãn tự gõ khi không lớp nào khớp: {profile: {khoá: {text, updated}}}. Để riêng vì pipeline Python
    // dựng tâm cụm theo tên lớp; `enrich merge` đọc file này và cho nó thắng nhãn máy.
    this.customFile = path.join(this.jsons, 'custom_labels.json');
    this.custom = readJson(this.customFile, {});
    this.customMtime = mtime(this.customFile);
    this.customBackedUp = false;
    // "Người này là ai": cụm do `enrich identities` gom + tên người duyệt đặt (modules/identity.py).
    const idDoc = readJson(path.join(this.jsons, 'identities.json'), {});
    this.idClusters = idDoc.clusters || {};
    this.idAssign = idDoc.assign || {};
    this.namesFile = path.join(this.jsons, 'identity_names.json');
    this.names = { clusters: {}, people: {}, ...readJson(this.namesFile, {}) };
    this.namesMtime = mtime(this.namesFile);
    this.namesBackedUp = false;
    // do `enrich actions` sinh: nhãn hành động gộp nhiều bằng chứng, và box có thể thừa (so với số người của thầy)
    this.actionPred = readJson(path.join(this.jsons, 'adl_people', 'action_pred.json'), {});
    this.extraBoxes = readJson(path.join(this.jsons, 'adl_people', 'extra_boxes.json'), {});
    this.teacherHints = readJson(path.join(this.jsons, 'adl_people', 'teacher_hints.json'), {});   // {pid: {label, text}}: câu của thầy tả người này
    this.searchCache = new Map();
    this.objects = new ObjectLabels(this);   // gán đồ vật theo vùng (objects.html)
    // Người = box máy dò chưa bị xoá + box người tự vẽ (docs/adr/0001). enriched/ có sẵn người tự vẽ (source "manual")
    // của lần gộp trước: chỉ lấy dự đoán tên, ngoại hình, vùng; box và hành động đọc thẳng từ manual_boxes.json.
    // Lịch sử hoàn tác giữ qua lần nạp lại (gộp dữ liệu, đồng bộ): hoàn tác theo mục nên vẫn an toàn.
    this.manualEnriched = new Map();
    this.manualBox = new Map();
    for (const im of this.images) {
      for (const p of im.people) if (p.source === 'manual') this.manualEnriched.set(p.id, p);
      for (const m of im.manual || []) if (m.kind === 'person') this.manualEnriched.set(m.id, m);   // enriched cũ
      im.detected = im.people.filter((p) => p.source !== 'manual');
    }
    for (const im of this.images) this._refreshPeople(im.image_id);
  }

  // Dựng lại danh sách Người của một ảnh sau khi xoá / vẽ / hoàn tác.
  _refreshPeople(imageId) {
    const im = this.byId.get(imageId);
    if (!im) return;
    for (const p of im.people || []) { this.personToImage.delete(p.id); this.personById.delete(p.id); this.manualBox.delete(p.id); }
    const hidden = this.names.not_person || {};
    const manual = (this.manual[imageId] || []).filter((m) => m.kind === 'person').map((m) => {
      const e = this.manualEnriched.get(m.id);
      const predicted = e?.identity?.predicted;
      this.manualBox.set(m.id, m);
      return { id: m.id, box: m.box, score: null, zone: same(e?.box, m.box) && e?.zone !== undefined ? e.zone : this._zoneOf(m.box), source: 'manual',
        attributes: e?.attributes ?? null, outfit: e?.outfit ?? null, action: null, identity: predicted ? { predicted } : null };
    });
    im.people = [...(im.detected || []).filter((p) => !hidden[p.id]), ...manual];
    im.num_people = im.people.length;
    for (const p of im.people) { this.personToImage.set(p.id, imageId); this.personById.set(p.id, p); }
    this.searchCache.delete(imageId);
  }

  // như pipeline/modules/enrich.py:assign_zone
  _zoneOf([x1, y1, x2, y2]) {
    const [W, H] = IMAGE_SIZE, area = Math.max((x2 - x1) * (y2 - y1), 1e-6);
    let best = [null, 0];
    for (const r of this.rois) {
      if (!r.zone) continue;
      const [zx1, zy1, zx2, zy2] = [r.zone[0] * W, r.zone[1] * H, r.zone[2] * W, r.zone[3] * H];
      const o = Math.max(0, Math.min(x2, zx2) - Math.max(x1, zx1)) * Math.max(0, Math.min(y2, zy2) - Math.max(y1, zy1)) / area;
      if (o > best[1]) best = [r.name, o];
    }
    return best[1] >= MIN_ZONE_OVERLAP ? best[0] : null;
  }

  _readState(dir) {
    const files = { approved: path.join(dir, 'approved.json'), rejected: path.join(dir, 'rejected.json') };
    const sugg = readJson(path.join(dir, 'suggestions.json'), { round: 0, items: [] });
    return {
      files,
      approved: readJson(files.approved, {}),
      rejected: readJson(files.rejected, {}),
      suggestions: new Map((sugg.items || []).map((s) => [s.image_id, s])),
      round: Number(sugg.round || 0),
      mtimes: { approved: mtime(files.approved), rejected: mtime(files.rejected) },
      backedUp: false,
    };
  }

  // Ưu tiên như enrich._label_entry: đã duyệt > gợi ý của máy. Gợi ý đã bị từ chối thì không hiện lại.
  resolve(profileName, key) {
    const p = this.profiles[profileName];
    if (!p) return null;
    if (p.layer === 'people' && this.manualBox?.has(key)) {   // box tự vẽ: hành động lưu ngay trong manual_boxes.json
      const m = this.manualBox.get(key);
      if (m.custom) return { label: m.custom, label_vi: m.custom, group: '', source: 'human', score: null, vlm: null, custom: true };
      const c = m.label && p.classByName.get(m.label);
      return m.label ? { label: m.label, label_vi: c?.name_vi || m.label_vi || '', group: c?.group || '', source: 'human', score: null, vlm: null } : null;
    }
    const c = this.custom[profileName]?.[key];
    if (c) return { label: c.text, label_vi: c.text, group: '', source: 'human', score: null, vlm: null, custom: true };
    const a = p.approved[key];
    const meta = (label) => { const c = p.classByName.get(label); return { label_vi: c?.name_vi || '', group: c?.group || '' }; };
    if (a) return { label: a.label, ...meta(a.label), source: a.source || 'seed', score: a.score ?? null, vlm: a.vlm ?? null };
    const s = p.suggestions.get(key);
    if (s && !(p.rejected[key] || []).includes(s.label)) return { label: s.label, ...meta(s.label), source: 'suggested', score: s.score ?? null, vlm: null };
    return null;
  }

  roiKey(im, roi) {
    return im.rois[roi]?.changed === false ? `${BASELINE_PREFIX}${im.hour}` : im.image_id;
  }

  // Mọi đơn vị có nhãn của một ảnh: {profile, key, kind, id, state}.
  units(im) {
    const out = [];
    if (this.profiles.image) out.push({ profile: 'image', key: im.image_id, kind: 'image', id: im.image_id, state: this.resolve('image', im.image_id) });
    if (this.profiles.people) for (const p of im.people) out.push({ profile: 'people', key: p.id, kind: 'person', id: p.id, zone: p.zone, state: this.resolve('people', p.id) });
    for (const r of this.rois) {
      const prof = `roi-${r.name}`;
      if (!this.profiles[prof] || !im.rois[r.name]) continue;
      const key = this.roiKey(im, r.name);
      out.push({ profile: prof, key, kind: 'roi', id: r.name, inherited: key !== im.image_id, state: this.resolve(prof, key) });
    }
    return out;
  }

  search(im) {
    let s = this.searchCache.get(im.image_id);
    if (s === undefined) {
      // chỉ id + nhãn: tên vùng/zone có ở mọi ảnh nên khớp gần hết (đã có bộ lọc zone riêng)
      const parts = [im.image_id];
      for (const u of this.units(im)) if (u.state) parts.push(u.state.label, u.state.label_vi);
      for (const m of this.manual[im.image_id] || []) parts.push(m.label || '', m.label_vi || '');
      for (const p of im.people) {
        parts.push(this.identity(p.id)?.name || '');
        for (const v of Object.values(p.attributes || {})) parts.push(Array.isArray(v) ? v.join(' ') : v);
      }
      for (const o of im.objects || []) parts.push(o.label);
      s = norm(parts.join(' | '));
      this.searchCache.set(im.image_id, s);
    }
    return s;
  }

  // f: {tokens:[{profile,label}], mode:'or'|'and', q, sources:[], hours:[], zone, minPeople, maxPeople, edited, sort}
  query(f) {
    const tokens = f.tokens || [];
    const sources = new Set(f.sources || []);
    const hours = new Set(f.hours || []);
    // "cụm từ trong ngoặc kép" khớp nguyên cụm; còn lại tách theo khoảng trắng
    const words = (norm(f.q || '').match(/-?"[^"]+"|\S+/g) || []).map((w) => w.replace(/"/g, '')).filter((w) => w && w !== '-');
    const inc = words.filter((w) => !w.startsWith('-'));
    const exc = words.filter((w) => w.startsWith('-') && w.length > 1).map((w) => w.slice(1));
    const srcOk = (u) => !sources.size || (u.state && sources.has(u.state.source));
    const tokOk = (t, u) => u.profile === t.profile && (t.label === NONE ? !u.state : u.state?.label === t.label) && (t.label === NONE || srcOk(u));
    const out = [];
    for (const im of this.images) {
      if (hours.size && !hours.has(im.hour)) continue;
      if (f.minPeople != null && im.num_people < f.minPeople) continue;
      if (f.maxPeople != null && im.num_people > f.maxPeople) continue;
      if (f.zone && !im.people.some((p) => p.zone === f.zone)) continue;
      if (f.edited && !this.edited.has(im.image_id)) continue;
      if (f.manual && !(this.manual[im.image_id] || []).length) continue;
      if (inc.length || exc.length) {
        const s = this.search(im);
        if (!inc.every((w) => s.includes(w)) || exc.some((w) => s.includes(w))) continue;
      }
      const units = this.units(im);
      const conflicts = f.conflict ? units.filter((u) => u.kind === 'person' && u.state && u.state.source !== 'human' && ACTION_ZONE[u.state.label] && u.zone !== ACTION_ZONE[u.state.label]) : [];
      if (f.conflict && !conflicts.length) continue;
      let hits;
      if (tokens.length) {
        const per = tokens.map((t) => units.filter((u) => tokOk(t, u)));
        if (f.mode === 'and' ? per.some((h) => !h.length) : per.every((h) => !h.length)) continue;
        hits = [...new Set(per.flat())];
      } else if (sources.size) {
        hits = units.filter((u) => u.state && sources.has(u.state.source));
        if (!hits.length) continue;
      } else hits = conflicts;
      out.push({ image_id: im.image_id, hour: im.hour, num_people: im.num_people, hits });
    }
    if (f.sort === 'people') out.sort((a, b) => b.num_people - a.num_people);
    else if (f.sort === 'score') {
      const minScore = (r) => Math.min(...r.hits.map((h) => h.state?.score ?? Infinity), Infinity);
      out.sort((a, b) => minScore(a) - minScore(b));
    }
    return out;
  }

  // Đếm theo lớp / nguồn / giờ cho từng profile.
  stats() {
    const res = {};
    for (const p of Object.values(this.profiles)) res[p.name] = { byLabel: {}, bySource: {}, byHour: {}, none: 0, total: 0 };
    for (const im of this.images) {
      for (const u of this.units(im)) {
        const r = res[u.profile];
        r.total++;
        if (!u.state) { r.none++; continue; }
        const l = u.state.label;
        r.byLabel[l] = (r.byLabel[l] || 0) + 1;
        r.bySource[u.state.source] = (r.bySource[u.state.source] || 0) + 1;
        (r.byHour[l] ||= {})[im.hour] = (r.byHour[l][im.hour] || 0) + 1;
      }
    }
    return res;
  }

  // Như modules/identity.py:identity_of — tên riêng của người > tên của cụm.
  identity(pid) {
    const a = this.idAssign[pid];
    if (pid in this.names.people) return { cluster: a?.cluster ?? null, sim: a?.sim ?? null, name: this.names.people[pid] || null, source: 'person' };
    if (!a) return null;
    const name = this.names.clusters[a.cluster] || null;
    return { cluster: a.cluster, sim: a.sim, name, source: name ? 'cluster' : null };
  }

  // Tên cuối như modules/identity.py:final_identity: tên đặt riêng > mô hình (lần gộp gần nhất, prob >= minProb) > tên cụm.
  finalIdentity(pid, minProb = 0.6) {
    const live = this.identity(pid);
    const predicted = this.personById.get(pid)?.identity?.predicted || null;
    if (live?.source === 'person') return { ...live, predicted };
    if (predicted && predicted.prob >= minProb) return { cluster: live?.cluster ?? null, sim: live?.sim ?? null, name: predicted.name, source: 'model', predicted };
    return live ? { ...live, predicted } : (predicted ? { cluster: null, sim: null, name: null, source: null, predicted } : null);
  }

  // Bảng định danh: số crop theo tên cuối và theo cụm.
  // Người duyệt đã quyết định cho crop này: đặt tên riêng (kể cả "Không rõ" = "") hoặc cụm của nó đã có tên.
  assigned(pid) {
    return pid in this.names.people || !!this.names.not_person?.[pid] || !!this.names.clusters[this.idAssign[pid]?.cluster];
  }

  identitySummary() {
    const names = {}, size = {}, left = {};
    let unnamed = 0, unknown = 0, hard = 0;
    const notPerson = Object.keys(this.names.not_person || {}).length;   // box đã xoá không còn trong personToImage
    const pending = {};   // crop máy gán tên, chưa ai xác nhận: hàng "Duyệt: <tên>" của trang gán nhanh
    const recheck = Object.keys(this.names.recheck || {}).filter((pid) => !(pid in this.names.people) && !this.names.not_person?.[pid]).length;
    for (const pid of this.personToImage.keys()) {
      const f = this.finalIdentity(pid);
      if (f?.name) names[f.name] = (names[f.name] || 0) + 1;
      if (f?.source === 'model') pending[f.name] = (pending[f.name] || 0) + 1;
      else if (this.names.people[pid] === '') unknown++;
      else unnamed++;
      if (!(pid in this.names.people) && f?.predicted?.margin != null) hard++;   // còn trong hàng "Khó trước"
      const c = this.idAssign[pid]?.cluster;
      if (!c) continue;
      size[c] = (size[c] || 0) + 1;
      if (!this.assigned(pid)) left[c] = (left[c] || 0) + 1;
    }
    return { names, unnamed, unknown, hard, notPerson, pending, recheck, extra: Object.keys(this.extraBoxes).filter((pid) => this.personToImage.has(pid) && !(pid in this.names.people) && !this.names.not_person?.[pid]).length, clusters: Object.keys(this.idClusters).sort().map((c) => ({ cluster: c, size: size[c] || 0, left: left[c] || 0,
      name: this.names.clusters[c] || null, label: this.idClusters[c].label || null })) };
  }

  // Mọi crop của một cụm (by.cluster) hoặc một tên cuối (by.name; '' = chưa có tên), xếp theo độ giống với cụm.
  // by.hard: crop chưa ai gán, mô hình phân vân nhất trước (chênh xác suất tên 1 và tên 2 nhỏ nhất); sửa các crop này
  // rồi chạy lại `enrich identities` giúp mô hình nhiều nhất trên mỗi lần bấm.
  identityMembers(by) {
    const out = [];
    for (const [pid, image_id] of this.personToImage) {
      const a = this.idAssign[pid];
      const f = this.finalIdentity(pid);
      // cụm chỉ còn crop chưa gán; "Không rõ" là danh sách riêng; "Chưa có tên" bỏ crop đã đánh dấu Không rõ
      if (this.names.not_person?.[pid]) continue;
      if (by.extra) { if (!this.extraBoxes[pid] || pid in this.names.people) continue; }
      else if (by.recheck) { if (!(pid in (this.names.recheck || {})) || pid in this.names.people) continue; }
      else if (by.hard) { if (pid in this.names.people || f?.predicted?.margin == null) continue; }
      else if (by.unknown) { if (this.names.people[pid] !== '') continue; }
      else if (by.cluster != null) { if (a?.cluster !== by.cluster || this.assigned(pid)) continue; }
      else if ((f?.name || '') !== by.name || (!by.name && this.names.people[pid] === '')) continue;
      const p = this.personById.get(pid);
      out.push({ id: pid, image_id, cluster: a?.cluster ?? null, sim: a?.sim ?? null, name: f?.name ?? null, source: f?.source ?? null,
        rejected: this.names.recheck?.[pid] ?? null, extra: this.extraBoxes[pid] ?? null, box: p?.box ?? null, prob: f?.predicted?.prob ?? null, margin: f?.predicted?.margin ?? null, guess: f?.predicted?.name ?? null, h: p ? Math.round(p.box[3] - p.box[1]) : null });
    }
    if (by.hard) return out.sort((x, y) => x.margin - y.margin).slice(0, HARD_LIMIT);
    if (by.extra) return out.sort((x, y) => this.extraBoxes[x.id].score - this.extraBoxes[y.id].score);
    return out.sort((x, y) => (y.sim ?? 0) - (x.sim ?? 0));
  }

  // Trang gán nhanh hành động: người theo nhãn hành động hiện tại.
  //   by.label: nhãn máy (chưa ai duyệt) đúng lớp đó; by.recheck: đã bác (X) một nhãn máy mà chưa có nhãn người duyệt
  //   (có thể đang hiện gợi ý khác của máy);
  //   by.conflict: nhãn máy mâu thuẫn với vùng đứng (ACTION_ZONE).
  //   by.teacher: câu mô tả của thầy tả người này làm lớp đó (teacher_hints.json), chưa có nhãn người duyệt.
  // Nhãn máy đang dùng: `enrich merge` có thể đã thay nhãn VLM bằng nhãn gộp (source "fused", machine_label = nhãn VLM
  // cũ). Hàng duyệt phải hỏi theo nhãn đó, không phải nhãn VLM đã bị thay. Nhãn người duyệt luôn thắng.
  machineAction(p, st) {
    if (!st || st.source === 'human') return st;
    const a = p.action;
    return a?.source === 'fused' && a.machine_label === st.label ? { ...st, label: a.label, source: 'fused' } : st;
  }
  actionMembers(by) {
    const out = [];
    for (const im of this.images) for (const p of im.people) {
      const st = this.machineAction(p, this.resolve('people', p.id));
      const rejected = this.profiles.people?.rejected[p.id] || [];
      const pred = this.actionPred[p.id];
      const hint = this.teacherHints[p.id];
      if (by.teacher) { if (hint?.label !== by.teacher || st?.source === 'human' || rejected.includes(hint.label)) continue; }
      else if (by.suspect) { if (!st || st.source === 'human' || !pred || pred.label === st.label || rejected.includes(pred.label)) continue; }
      else if (by.recheck) { if (st?.source === 'human' || !rejected.length) continue; }
      else if (!st || st.source === 'human') continue;
      else if (by.conflict) { if (!ACTION_ZONE[st.label] || p.zone === ACTION_ZONE[st.label]) continue; }
      else if (st.label !== by.label) continue;
      out.push({ id: p.id, image_id: im.image_id, box: p.box, h: Math.round(p.box[3] - p.box[1]), zone_vi: this.zoneVi[p.zone] || '', label: by.teacher ? hint.label : by.suspect ? pred.label : st?.label ?? null, teacher: hint?.text ?? null, machine: st?.label ?? null, gap: pred?.gap ?? null, source: st?.source ?? null, rejected });
    }
    return by.suspect ? out.sort((a, b) => b.gap - a.gap) : out;
  }
  // số còn lại mỗi hàng + số nhãn người duyệt mỗi lớp (mô hình hành động cần >= 20 mỗi lớp)
  actionSummary() {
    const pending = {}, human = {};
    let recheck = 0, conflict = 0, suspect = 0;
    const teacher = {};
    for (const im of this.images) for (const p of im.people) {
      const st = this.machineAction(p, this.resolve('people', p.id));
      const hint = this.teacherHints[p.id];
      if (hint && st?.source !== 'human' && !(this.profiles.people?.rejected[p.id] || []).includes(hint.label)) teacher[hint.label] = (teacher[hint.label] || 0) + 1;
      if (st?.source === 'human') { human[st.label] = (human[st.label] || 0) + 1; continue; }
      if ((this.profiles.people?.rejected[p.id] || []).length) recheck++;
      if (!st) continue;
      pending[st.label] = (pending[st.label] || 0) + 1;
      if (ACTION_ZONE[st.label] && p.zone !== ACTION_ZONE[st.label]) conflict++;
      const pred = this.actionPred[p.id];
      if (pred && pred.label !== st.label && !(this.profiles.people?.rejected[p.id] || []).includes(pred.label)) suspect++;
    }
    return { pending, human, recheck, conflict, suspect, teacher, classes: this.profiles.people?.classes || [] };
  }
  identityNames() {
    return [...new Set([...Object.values(this.names.clusters), ...Object.values(this.names.people)].filter(Boolean))].sort();
  }

  // op: {scope: 'cluster'|'person', cluster?, person?, name} ; name '' = xoá tên (person: "không rõ là ai")
  setIdentity(op) {
    const name = String(op?.name ?? '').trim().replace(/\s+/g, ' ').slice(0, 60);
    if (op.scope === 'cluster' && !this.idClusters[op.cluster]) throw bad(`Không có cụm ${op.cluster}.`);
    // persons: nhiều người một lần (kéo thả trên bảng định danh), một bước hoàn tác
    const persons = op.scope === 'person' ? (Array.isArray(op.persons) ? op.persons : [op.person]) : [];
    for (const pid of persons) if (!this.personToImage.has(pid)) throw bad(`Không có người ${pid}.`);
    if (op.scope === 'person' && !persons.length) throw bad('Chưa chọn người nào.');
    if (!['cluster', 'person'].includes(op.scope)) throw bad('scope phải là cluster hoặc person.');
    if (op.notPerson) return this.deletePeople(persons);   // trang gán nhanh cũ: "không phải người" = Xoá box
    const who = persons.length === 1 ? persons[0] : `${persons.length} người`;
    const desc = op.scope === 'cluster' ? `đặt tên "${name}" cho cụm ${op.cluster}` : op.recheck ? `đưa ${who} vào "Cần gán lại"`
      : op.reset ? `bỏ tên riêng của ${who}` : name ? `đặt tên "${name}" cho ${who}` : `đánh dấu ${who} không rõ là ai`;
    const descs = op.scope === 'cluster' ? [{ t: 'cluster', c: op.cluster }] : persons.map((pid) => ({ t: 'name', pid }));
    return this._step(desc, descs, () => {
      // ai đặt tên nào: names.by (pipeline Python chỉ đọc clusters / people nên bỏ qua khoá này)
      const by = (this.names.by ||= {});
      const recheck = (this.names.recheck ||= {});   // "sai, gán lại sau": {pid: tên bị bác}
      if (op.scope === 'cluster') { if (name) this.names.clusters[op.cluster] = name; else delete this.names.clusters[op.cluster]; by[`cluster:${op.cluster}`] = this._by(); }
      else if (op.recheck) for (const pid of persons) { recheck[pid] = String(op.recheck); by[pid] = this._by(); }
      else for (const pid of persons) {
        delete recheck[pid];
        if (op.reset) delete this.names.people[pid];
        else this.names.people[pid] = name;
        by[pid] = this._by();
      }
    });
  }

  // Xoá box (docs/adr/0002): box máy dò bị ẩn (names.not_person, enrich merge bỏ khỏi dataset), box tự vẽ bị bỏ hẳn.
  deletePeople(ids) {
    ids = Array.isArray(ids) ? ids : [ids];
    if (!ids.length) throw bad('Chưa chọn box nào.');
    for (const pid of ids) if (!this.personToImage.has(pid)) throw bad(`Không có người ${pid}.`);
    const descs = ids.map((pid) => (this.manualBox.has(pid) ? { t: 'box', image: this.personToImage.get(pid), id: pid } : { t: 'name', pid }));
    return this._step(`xoá box ${ids.length === 1 ? ids[0] : `${ids.length} người`}`, descs, () => {
      for (const d of descs) {
        if (d.t === 'box') this.manual[d.image] = this.manual[d.image].filter((m) => m.id !== d.id);
        else {
          delete this.names.people[d.pid];
          delete this.names.recheck?.[d.pid];
          (this.names.not_person ||= {})[d.pid] = true;
          (this.names.by ||= {})[d.pid] = this._by();
        }
      }
    });
  }

  // Sửa box máy dò (docs/adr/0002): ẩn box cũ, thay bằng box tự vẽ mang theo hành động và tên do người gán.
  replacePerson(pid, box) {
    const imageId = this.personToImage.get(pid);
    if (!imageId) throw bad(`Không có người ${pid}.`);
    if (this.manualBox.has(pid)) return this.manualOp({ action: 'update', image_id: imageId, id: pid, box });
    const clean = this._cleanBox(box);
    const id = this._nextManualId(imageId);
    const st = this.resolve('people', pid);
    const ident = this.identity(pid);
    const name = ident?.source === 'person' || ident?.source === 'cluster' ? ident.name : null;
    const descs = [{ t: 'name', pid }, { t: 'box', image: imageId, id }, ...(name ? [{ t: 'name', pid: id }] : [])];
    return this._step(`sửa box ${pid} (thành ${id})`, descs, () => {
      const m = { id, kind: 'person', box: clean, created: new Date().toISOString(), by: this.actor, replaces: pid };
      if (st?.source === 'human') { if (st.custom) m.custom = st.label; else m.label = st.label; }
      this.manual[imageId] = [...(this.manual[imageId] || []), m];
      delete this.names.people[pid];
      (this.names.not_person ||= {})[pid] = true;
      if (name) this.names.people[id] = name;
      (this.names.by ||= {})[pid] = this._by();
    }, { id });
  }

  // Đổi tên `from` thành `to` ở mọi nơi: tên cụm, tên đặt riêng, tên mô hình đoán (trong RAM, lần gộp sau học lại)
  // và khoá trong annotations/outfits.yaml (giữ trang phục đã điền). `to` trùng tên đã có = gộp hai người.
  renameIdentity(op) {
    const from = String(op?.from ?? '').trim();
    const to = String(op?.to ?? '').trim().replace(/\s+/g, ' ').slice(0, 60);
    if (!from || !to) throw bad('Cần tên cũ và tên mới.');
    if (from === to) return { changed: [], renamed: 0 };
    const descs = [...Object.entries(this.names.clusters).filter(([, v]) => v === from).map(([c]) => ({ t: 'cluster', c })),
      ...Object.entries(this.names.people).filter(([, v]) => v === from).map(([pid]) => ({ t: 'name', pid }))];
    const predicted = [...this.personById.values()].filter((p) => p.identity?.predicted?.name === from).map((p) => p.id);
    if (!descs.length && !predicted.length) throw bad(`Không có ai tên "${from}".`);
    const outfitsFile = path.join(this.jsons, 'outfits.yaml');
    const outfitsPrev = fs.existsSync(outfitsFile) ? fs.readFileSync(outfitsFile, 'utf8') : null;
    const out = this._step(`đổi tên "${from}" thành "${to}"`, descs, () => {
      for (const d of descs) { if (d.t === 'cluster') this.names.clusters[d.c] = to; else this.names.people[d.pid] = to; }
      for (const pid of predicted) this.personById.get(pid).identity.predicted.name = to;
      if (outfitsPrev) {
        const doc = YAML.parseDocument(outfitsPrev);
        const people = doc.get('people');
        if (people && people.has(from)) {
          if (people.has(to)) people.delete(from); // gộp vào người đã có: giữ trang phục của người đích
          else { const pair = people.items.find((it) => String(it.key?.value ?? it.key) === from); pair.key = doc.createNode(to); }
          beforeWrite(outfitsFile, { text: doc.toString() });
          fs.writeFileSync(outfitsFile, doc.toString(), 'utf8');
        }
      }
    }, {}, { rename: { from, to, predicted }, outfits: outfitsPrev, outfitsFile });
    const ids = [...new Set([...this.personToImage].filter(([pid]) => this.finalIdentity(pid)?.name === to).map(([, img]) => img))];
    for (const id of ids) { this.searchCache.delete(id); this.edited.add(id); }
    return { changed: ids, renamed: descs.length + predicted.length };
  }
  _saveNames() {
    if (!this.namesBackedUp && fs.existsSync(this.namesFile)) fs.copyFileSync(this.namesFile, `${this.namesFile}.bak-${new Date().toISOString().replace(/[:.]/g, '-')}`);
    this.namesBackedUp = true;
    beforeWrite(this.namesFile, { data: this.names });
    fs.writeFileSync(`${this.namesFile}.tmp`, JSON.stringify(this.names, null, 2), 'utf8');
    fs.renameSync(`${this.namesFile}.tmp`, this.namesFile);
    this.namesMtime = mtime(this.namesFile);
  }

  meta() {
    const stats = this.stats();
    return {
      identity: { clusters: Object.fromEntries(Object.entries(this.idClusters).map(([c, v]) => [c, { size: v.size, examples: v.examples, label: v.label || null, name: this.names.clusters[c] || null }])), names: this.identityNames() },
      objectLabels: [...new Set(this.images.flatMap((im) => (im.objects || []).map((o) => o.label)))].sort(),
      hours: this.hours,
      actionZone: ACTION_ZONE,
      rois: this.rois,
      images: this.images.length,
      manual: {
        boxes: Object.values(this.manual).reduce((n, l) => n + l.length, 0),
        objectLabels: [...new Set(Object.values(this.manual).flat().filter((m) => m.kind === 'object' && m.label).map((m) => m.label))].sort(),
      },
      profiles: Object.values(this.profiles).map((p) => ({ name: p.name, layer: p.layer, roi: p.roi || null, round: p.round, classes: p.classes, stats: stats[p.name] })),
    };
  }

  // Số ảnh dùng chung một khoá nền bg-HH của vùng (để cảnh báo trước khi sửa).
  affected(roi, key) {
    if (!key.startsWith(BASELINE_PREFIX)) return 1;
    const hour = key.slice(BASELINE_PREFIX.length);
    return this.images.filter((im) => im.hour === hour && im.rois[roi]?.changed === false).length;
  }

  detail(id) {
    const im = this.byId.get(id);
    if (!im) return null;
    const people = im.people.map((p) => ({ ...p, manual: p.source === 'manual', zone_vi: this.zoneVi[p.zone] || '', profile: 'people', key: p.id, state: this.resolve('people', p.id), predicted: p.identity?.predicted ?? null, identity: this.identity(p.id) }));
    const rois = this.rois.filter((r) => im.rois[r.name]).map((r) => {
      const key = this.roiKey(im, r.name);
      const prof = `roi-${r.name}`;
      return { ...r, changed: im.rois[r.name].changed, motion: im.rois[r.name].motion, profile: prof, key, inherited: key !== id,
        affected: this.affected(r.name, key), state: this.profiles[prof] ? this.resolve(prof, key) : null, rejected: this.profiles[prof]?.rejected[key] || [] };
    });
    const activity = { profile: 'image', key: id, state: this.resolve('image', id), rejected: this.profiles.image?.rejected[id] || [] };
    for (const p of people) p.rejected = this.profiles.people?.rejected[p.id] || [];
    const manual = (this.manual[id] || []).filter((m) => m.kind !== 'person').map((m) => ({ ...m }));   // box vật thể cũ (nay vẽ ở trang Đồ vật)
    return { image_id: id, hour: im.hour, num_people: im.num_people, activity, people, rois, manual, objects: im.objects || [], edited: this.edited.has(id), raw: im };
  }

  _validKey(p, key) {
    if (!SAFE.test(key)) return false;
    if (p.layer === 'image') return this.byId.has(key);
    if (p.layer === 'people') return this.personToImage.has(key);
    return this.byId.has(key) || (key.startsWith(BASELINE_PREFIX) && this.hours.includes(key.slice(BASELINE_PREFIX.length)));
  }

  _imagesOf(p, key) {
    if (p.layer === 'people') return [this.personToImage.get(key)];
    if (p.layer === 'roi' && key.startsWith(BASELINE_PREFIX)) {
      const hour = key.slice(BASELINE_PREFIX.length);
      return this.images.filter((im) => im.hour === hour && im.rois[p.roi]?.changed === false).map((im) => im.image_id);
    }
    return [key];
  }


  _save(p) {
    if (!p.backedUp) {
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      for (const k of ['approved', 'rejected']) if (fs.existsSync(p.files[k])) fs.copyFileSync(p.files[k], `${p.files[k]}.bak-${stamp}`);
      p.backedUp = true;
    }
    writeJsonAtomic(p.files.approved, p.approved);
    writeJsonAtomic(p.files.rejected, p.rejected);
    p.mtimes = { approved: mtime(p.files.approved), rejected: mtime(p.files.rejected) };
  }

  _saveCustom() {
    if (!this.customBackedUp && fs.existsSync(this.customFile)) {
      fs.copyFileSync(this.customFile, `${this.customFile}.bak-${new Date().toISOString().replace(/[:.]/g, '-')}`);
    }
    this.customBackedUp = true;
    for (const k of Object.keys(this.custom)) if (!Object.keys(this.custom[k]).length) delete this.custom[k];
    writeJsonAtomic(this.customFile, this.custom);
    this.customMtime = mtime(this.customFile);
  }


  // ops: [{profile, key, action: 'set'|'confirm'|'reject'|'reset'|'custom', label?, text?}] -> một bước hoàn tác.
  // 'custom': nhãn tự gõ khi không lớp nào khớp; nhãn máy đang có bị đánh dấu sai để không quay lại vòng lặp.
  // Người tự vẽ: hành động lưu trong manual_boxes.json (label hoặc custom), không vào approved.json của pipeline.
  apply(ops) {
    if (!Array.isArray(ops) || !ops.length) throw bad('Không có thao tác nào.');
    for (const op of ops) {
      const p = this.profiles[op.profile];
      if (!p) throw bad(`Profile lạ: ${op.profile}`);
      if (!this._validKey(p, op.key)) throw bad(`Khoá không tồn tại: ${op.key}`);
      if ((op.action === 'set' || (op.action === 'reject' && op.label)) && !p.classByName.has(op.label)) throw bad(`Lớp "${op.label}" không thuộc bộ nhãn ${op.profile}`);
      if (!['set', 'confirm', 'reject', 'reset', 'custom'].includes(op.action)) throw bad(`Thao tác lạ: ${op.action}`);
      if (op.action === 'custom') {
        op.text = String(op.text ?? '').trim().replace(/\s+/g, ' ').slice(0, 120);
        if (!op.text) throw bad('Nhập mô tả trước khi lưu.');
      }
    }
    const descs = ops.map((op) => this._labelDesc(op.profile, op.key));
    const first = ops[0], st = this.resolve(first.profile, first.key);
    const what = first.action === 'custom' ? `"${first.text}"` : first.action === 'set' ? `"${this.profiles[first.profile].classByName.get(first.label)?.name_vi || first.label}"` : st ? `"${st.label_vi || st.label}"` : '';
    const verb = { set: 'gán', custom: 'gán', confirm: 'xác nhận', reject: 'bác', reset: 'đặt lại' }[first.action];
    const desc = `${verb} hành động ${what} cho ${ops.length === 1 ? first.key : `${ops.length} mục`}`.replace(/\s+/g, ' ');
    return this._step(desc, descs, () => {
      for (const op of ops) {
        const p = this.profiles[op.profile];
        if (p.layer === 'people' && this.manualBox.has(op.key)) {
          const m = this.manualBox.get(op.key);
          if (op.action === 'set') { m.label = op.label; delete m.custom; delete m.label_vi; }
          else if (op.action === 'custom') { m.custom = op.text; delete m.label; delete m.label_vi; }
          else if (op.action !== 'confirm') { delete m.label; delete m.custom; delete m.label_vi; }   // box tự vẽ không có nhãn máy để quay về
          continue;
        }
        const cur = this.resolve(op.profile, op.key);
        const own = (this.custom[op.profile] ||= {});
        if (own[op.key] && op.action !== 'confirm') delete own[op.key];
        if (op.action === 'custom') {
          if (cur && !cur.custom) {
            const list = p.rejected[op.key] || [];
            if (!list.includes(cur.label)) p.rejected[op.key] = [...list, cur.label];
          }
          delete p.approved[op.key];
          own[op.key] = { text: op.text, updated: new Date().toISOString(), by: this.actor };
        } else if (op.action === 'confirm' && cur?.custom) {
          continue; // nhãn tự gõ đã là của người duyệt
        } else if (cur?.custom && op.action === 'reject') {
          // xoá nhãn tự gõ (đã làm ở trên), không có gì để đánh dấu sai
        } else if (op.action === 'set' || op.action === 'confirm') {
          const label = op.action === 'set' ? op.label : cur?.label;
          if (!label) continue; // xác nhận một mục chưa có nhãn: bỏ qua
          p.approved[op.key] = { label, source: 'human', round: p.round, score: cur?.score ?? null, ...this._by() };
        } else if (op.action === 'reject' && op.only && op.label && op.label !== cur?.label) {
          // chỉ bác một gợi ý (vd câu của thầy), giữ nguyên nhãn máy hiện có
          const list = p.rejected[op.key] || [];
          if (!list.includes(op.label)) p.rejected[op.key] = [...list, op.label];
        } else if (op.action === 'reject') {
          if (!cur) continue;
          // op.label: nhãn đang hiện cho người duyệt (vd nhãn gộp đã thay nhãn VLM); bác cả hai
          const list = p.rejected[op.key] || [];
          p.rejected[op.key] = [...new Set([...list, cur.label, ...(op.label ? [op.label] : [])])];
          delete p.approved[op.key];
        } else delete p.approved[op.key];
      }
    });
  }

  _labelDesc(profile, key) {
    return this.profiles[profile]?.layer === 'people' && this.manualBox.has(key) ? { t: 'box', image: this.personToImage.get(key), id: key } : { t: 'label', profile, key };
  }

  // ------------------------------------------------------------ hoàn tác theo mục
  // Mỗi bước hoàn tác là danh sách mục {mô tả mục, before, after}. Hoàn tác chỉ trả lại đúng các mục đó, và bỏ qua
  // (báo xung đột) mục đã bị người khác sửa sau mình, nên không bao giờ đè thay đổi của người khác.
  //   {t: 'label', profile, key}  nhãn của một khoá (approved + rejected + tự gõ)
  //   {t: 'name', pid}            tên riêng / xoá box / cần gán lại của một người máy dò
  //   {t: 'cluster', c}           tên của một cụm
  //   {t: 'box', image, id}       một box tự vẽ (cả hành động của nó)
  _get(d) {
    if (d.t === 'label') {
      const p = this.profiles[d.profile];
      return { approved: clone(p.approved[d.key]), rejected: clone(p.rejected[d.key]), custom: clone(this.custom[d.profile]?.[d.key]) };
    }
    if (d.t === 'name') return { name: this.names.people[d.pid], hidden: this.names.not_person?.[d.pid], recheck: this.names.recheck?.[d.pid] };
    if (d.t === 'cluster') return this.names.clusters[d.c];
    return clone((this.manual[d.image] || []).find((m) => m.id === d.id));
  }

  _set(d, v, touched) {
    const put = (obj, k, val) => { if (val === undefined) delete obj[k]; else obj[k] = clone(val); };
    if (d.t === 'label') {
      const p = this.profiles[d.profile];
      put(p.approved, d.key, v.approved); put(p.rejected, d.key, v.rejected); put((this.custom[d.profile] ||= {}), d.key, v.custom);
      touched.add(p); touched.add('custom');
      for (const id of this._imagesOf(p, d.key)) touched.add(`img:${id}`);
    } else if (d.t === 'name') {
      put(this.names.people, d.pid, v.name); put((this.names.not_person ||= {}), d.pid, v.hidden); put((this.names.recheck ||= {}), d.pid, v.recheck);
      touched.add('names'); touched.add(`img:${imageOfPid(d.pid)}`);
    } else if (d.t === 'cluster') {
      put(this.names.clusters, d.c, v);
      touched.add('names');
      for (const [pid, a] of Object.entries(this.idAssign)) if (a.cluster === d.c) touched.add(`img:${imageOfPid(pid)}`);
    } else {
      const list = (this.manual[d.image] || []).filter((m) => m.id !== d.id);
      this.manual[d.image] = v === undefined ? list : [...list, clone(v)].sort((a, b) => (a.id < b.id ? -1 : 1));
      touched.add('manual'); touched.add(`img:${d.image}`);
    }
  }

  _filesOf(descs) {
    const files = new Set();
    for (const d of descs) {
      if (d.t === 'label') { const p = this.profiles[d.profile]; files.add(p.files.approved); files.add(p.files.rejected); files.add(this.customFile); }
      else if (d.t === 'box') files.add(this.manualFile);
      else files.add(this.namesFile);
    }
    return files;
  }

  _stale(descs) {
    const known = new Map([[this.customFile, this.customMtime], [this.manualFile, this.manualMtime], [this.namesFile, this.namesMtime]]);
    for (const p of Object.values(this.profiles)) { known.set(p.files.approved, p.mtimes.approved); known.set(p.files.rejected, p.mtimes.rejected); }
    return [...this._filesOf(descs)].some((f) => mtime(f) !== known.get(f));
  }

  // File vừa bị chỗ khác ghi (đồng bộ, pipeline, người khác): nạp lại. Nếu chính các mục sắp sửa đã đổi thì báo và
  // dừng để người dùng xem giá trị mới; nếu không thì làm tiếp trên dữ liệu mới.
  _fresh(descs, strict) {
    if (!this._stale(descs)) return;
    const before = descs.map((d) => this._get(d));
    this.reload();
    if (strict && descs.some((d, i) => !same(this._get(d), before[i]))) {
      throw bad('Mục này vừa được người khác sửa. Đã nạp bản mới; kiểm tra rồi làm lại.', 409);
    }
  }

  _saveTouched(touched) {
    for (const t of touched) if (typeof t === 'object') this._save(t);
    if (touched.has('custom')) this._saveCustom();
    if (touched.has('names')) this._saveNames();
    if (touched.has('manual')) this._saveManual();
    const changed = [...touched].filter((t) => typeof t === 'string' && t.startsWith('img:')).map((t) => t.slice(4));
    for (const id of changed) { this._refreshPeople(id); this.searchCache.delete(id); this.edited.add(id); }
    return changed;
  }

  // Chạy `fn` (sửa trong RAM) rồi ghi đúng các file đã đổi; lưu một bước hoàn tác gồm các mục thật sự đổi.
  _step(desc, descs, fn, result = {}, extra = {}) {
    this._fresh(descs, true);
    const before = descs.map((d) => this._get(d));
    fn();
    const items = [], touched = new Set();
    descs.forEach((d, i) => {
      const after = this._get(d);
      if (same(after, before[i])) return;
      items.push({ ...d, before: before[i], after });
      this._set(d, after, touched);   // _set ghi lại đúng giá trị: chỉ để biết file / ảnh nào cần lưu
    });
    const changed = this._saveTouched(touched);
    if (items.length || extra.rename) this.undoStack.push({ items, desc, page: this.page, ...extra });
    return { ...result, changed };
  }

  _saveManual() {
    if (!this.manualBackedUp && fs.existsSync(this.manualFile)) {
      fs.copyFileSync(this.manualFile, `${this.manualFile}.bak-${new Date().toISOString().replace(/[:.]/g, '-')}`);
    }
    this.manualBackedUp = true;
    for (const k of Object.keys(this.manual)) if (!this.manual[k].length) delete this.manual[k];
    writeJsonAtomic(this.manualFile, this.manual);
    this.manualMtime = mtime(this.manualFile);
  }

  _cleanBox(b) {
    if (!Array.isArray(b) || b.length !== 4 || !b.every(Number.isFinite)) throw bad('Box phải là [x1, y1, x2, y2].');
    const [x1, y1, x2, y2] = b.map((v) => Math.round(Math.max(0, v) * 10) / 10);
    if (x2 - x1 < 4 || y2 - y1 < 4) throw bad('Box quá nhỏ.');
    return [x1, y1, x2, y2];
  }

  _nextManualId(imageId) {
    const n = Math.max(0, ...(this.manual[imageId] || []).map((m) => Number(m.id.split('_m').pop()) || 0)) + 1;
    return `${imageId}_m${String(n).padStart(2, '0')}`;
  }

  // Box người tự vẽ (trang Ảnh chỉ vẽ Người; vật thể vẽ ở trang Đồ vật).
  // op: {action: 'add'|'update'|'delete', image_id, id?, box?}; box tính bằng pixel ảnh gốc [x1, y1, x2, y2].
  // Hành động của box gán qua apply() như mọi Người.
  manualOp(op) {
    const im = this.byId.get(op?.image_id);
    if (!im) throw bad('Ảnh không tồn tại.');
    if (op.action === 'delete') return this.deletePeople([op.id]);
    if (op.action === 'add') {
      const box = this._cleanBox(op.box);
      const id = this._nextManualId(im.image_id);
      return this._step(`vẽ box ${id}`, [{ t: 'box', image: im.image_id, id }], () => {
        this.manual[im.image_id] = [...(this.manual[im.image_id] || []), { id, kind: 'person', box, created: new Date().toISOString(), by: this.actor }];
      }, { id });
    }
    if (op.action !== 'update') throw bad(`Thao tác lạ: ${op.action}`);
    const m = (this.manual[im.image_id] || []).find((x) => x.id === op.id);
    if (!m) throw bad(`Không có box ${op.id}.`);
    const box = this._cleanBox(op.box);
    return this._step(`sửa box ${op.id}`, [{ t: 'box', image: im.image_id, id: op.id }], () => {
      const cur = this.manual[im.image_id].find((x) => x.id === op.id);
      cur.box = box;
      cur.updated = new Date().toISOString();
    }, { id: op.id });
  }

  // Phím số / chữ của trang gán nhanh: thứ tự ghim trong label_definitions/hotkeys.yaml, phần còn lại theo số crop.
  hotkeys() {
    const KEYS = [...'1234567890', ...'qwertyuiopsdfghjlcvbnm'];   // bỏ a, x, z, k: phím chung (đúng, sai, hoàn tác, không rõ)
    const doc = (() => { try { return YAML.parse(fs.readFileSync(path.join(this.root, 'label_definitions', 'hotkeys.yaml'), 'utf8')) || {}; } catch { return {}; } })();
    const order = (pinned, all, count) => {
      const rest = all.filter((x) => !pinned.includes(x)).sort((a, b) => (count[b] || 0) - (count[a] || 0));
      return [...pinned.filter((x) => all.includes(x)), ...rest].slice(0, KEYS.length).map((x, i) => [KEYS[i], x]);
    };
    const names = this.identitySummary().names;
    const actions = this.stats().people?.byLabel || {};
    return { names: order(doc.names || [], Object.keys(names), names), actions: order(doc.actions || [], (this.profiles.people?.classes || []).map((c) => c.name), actions) };
  }
  undo() {
    const step = this.undoStack.pop();
    if (!step) throw bad('Không còn gì để hoàn tác.');
    if (step.obj) { try { return this.objects.undo(step); } catch (e) { this.undoStack.push(step); throw e; } }
    try { this._fresh(step.items, false); } catch (e) { this.undoStack.push(step); throw e; }
    const touched = new Set(), conflicts = [];
    for (const it of [...step.items].reverse()) {
      if (!same(this._get(it), it.after)) { conflicts.push(it.pid || it.id || it.key || it.c); continue; }   // người khác đã sửa sau mình
      this._set(it, it.before, touched);
    }
    if (step.rename) {
      for (const pid of step.rename.predicted) { const p = this.personById.get(pid); if (p?.identity?.predicted) p.identity.predicted.name = step.rename.from; }
      if (step.outfits != null) { beforeWrite(step.outfitsFile, { text: step.outfits }); fs.writeFileSync(step.outfitsFile, step.outfits, 'utf8'); }
    }
    const changed = this._saveTouched(touched);
    return { changed, left: this.undoStack.length, desc: step.desc, page: step.page, conflicts };
  }
}

module.exports = { Store, norm, NONE, SAFE, BASELINE_PREFIX };

// Setting cá nhân của trang tìm ảnh: một bảng SCHEMA duy nhất sinh ra khung chỉnh, kiểm tra, xuất/nhập JSON.
// Lưu localStorage của trình duyệt (mỗi thành viên chạy app trên máy riêng). Thiết kế: docs/search-settings-design.md
(function (root) {
  const SCHEMA = [
    { key: 'level.default', group: 'Đề và level', label: 'Level dùng khi kéo đề', type: 'select', default: 'latest',
      options: [['latest', 'Mới nhất'], ['first', 'Level đầu'], ['fixed', 'Cố định level N']] },
    { key: 'level.fixedN', group: 'Đề và level', label: 'Level N (khi cố định)', type: 'int', min: 1, max: 20, default: 1,
      help: 'Đề chưa đủ N level thì lấy level mới nhất.' },
    { key: 'level.bar', group: 'Đề và level', label: 'Thanh chọn level L1…Ln', type: 'bool', default: true },
    { key: 'poll.seconds', group: 'Đề và level', label: 'Tự kiểm tra gợi ý mới', type: 'select', default: 0,
      options: [[0, 'Tắt'], [5, 'Mỗi 5 giây'], [10, 'Mỗi 10 giây'], [20, 'Mỗi 20 giây']],
      help: 'Chỉ báo, không tự tìm lại. Mỗi lần kiểm tra gọi DRES một lần.' },
    { key: 'poll.sound', group: 'Đề và level', label: 'Âm báo khi có gợi ý mới', type: 'bool', default: false },

    { key: 'grid.perPage', group: 'Kết quả', label: 'Số ảnh mỗi trang', type: 'select', default: 2,
      options: [1, 2, 3, 4, 6, 8, 12].map((n) => [n, String(n)]) },
    { key: 'grid.columns', group: 'Kết quả', label: 'Số cột', type: 'select', default: 'auto',
      options: [['auto', 'Tự động'], [1, '1'], [2, '2'], [3, '3'], [4, '4']], help: 'Màn hình hẹp tự giảm số cột.' },
    { key: 'grid.thumbs', group: 'Kết quả', label: 'Dùng ảnh thu nhỏ trên lưới', type: 'bool', default: false,
      help: 'Nhanh hơn khi xem nhiều ảnh một lúc.' },
    { key: 'search.top', group: 'Kết quả', label: 'Số kết quả lấy về', type: 'select', default: 300,
      options: [[100, '100'], [300, '300'], [500, '500']] },
    { key: 'grid.showPairs', group: 'Kết quả', label: 'Hiện dòng ghép người trên ảnh', type: 'bool', default: true },
    { key: 'grid.showScore', group: 'Kết quả', label: 'Hiện điểm số trên ảnh', type: 'bool', default: false },

    { key: 'parse.showClauses', group: 'Tách đề', label: 'Hiện truy vấn con (người, nhóm, cảnh)', type: 'bool', default: true },
    { key: 'parse.showEnglish', group: 'Tách đề', label: 'Hiện bản dịch EN', type: 'bool', default: true },
    { key: 'parse.showLayers', group: 'Tách đề', label: 'Hiện số ảnh qua từng lớp', type: 'bool', default: true },
    { key: 'parse.editor', group: 'Tách đề', label: 'Form sửa biến', type: 'select', default: 'collapsed',
      options: [['off', 'Tắt'], ['collapsed', 'Thu gọn'], ['open', 'Mở sẵn']] },

    { key: 'detail.span', group: 'Xem chi tiết', label: 'Số ảnh lân cận mỗi bên', type: 'select', default: 6,
      options: [[3, '3'], [6, '6'], [10, '10']] },
    { key: 'detail.boxes', group: 'Xem chi tiết', label: 'Box người', type: 'select', default: 'all',
      options: [['off', 'Tắt'], ['all', 'Tất cả'], ['matched', 'Chỉ người được ghép']] },
    { key: 'detail.meta', group: 'Xem chi tiết', label: 'Hiện thông tin người, vùng, vật', type: 'bool', default: true },

    { key: 'submit.confirm', group: 'Nộp bài', label: 'Hỏi xác nhận trước khi nộp', type: 'bool', default: true,
      help: 'Nộp sai bị trừ điểm. Tắt thì header hiện cảnh báo đỏ.' },
    { key: 'submit.mark', group: 'Nộp bài', label: 'Đánh dấu ảnh đã nộp trên lưới', type: 'bool', default: true },

    { key: 'ui.theme', group: 'Chung', label: 'Giao diện', type: 'select', default: 'system',
      options: [['system', 'Theo hệ thống'], ['light', 'Sáng'], ['dark', 'Tối']] },
    { key: 'ui.fontSize', group: 'Chung', label: 'Cỡ chữ', type: 'select', default: 'medium',
      options: [['small', 'Nhỏ'], ['medium', 'Vừa'], ['large', 'Lớn']] },
  ];
  const STORE = 'castle-search-settings';
  const BY_KEY = Object.fromEntries(SCHEMA.map((s) => [s.key, s]));

  function defaults() {
    return Object.fromEntries(SCHEMA.map((s) => [s.key, s.default]));
  }

  // Thuần, không đụng DOM: giữ khoá có trong SCHEMA với giá trị hợp lệ, còn lại về mặc định
  function sanitize(raw) {
    const out = defaults();
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return out;
    for (const [key, value] of Object.entries(raw)) {
      const s = BY_KEY[key];
      if (!s) continue;
      if (s.type === 'bool' && typeof value === 'boolean') out[key] = value;
      if (s.type === 'int' && Number.isInteger(value) && value >= s.min && value <= s.max) out[key] = value;
      if (s.type === 'select' && s.options.some(([v]) => v === value)) out[key] = value;
    }
    return out;
  }

  function parse(text) {
    try { return sanitize(JSON.parse(text)); } catch { return defaults(); }
  }

  let values = defaults();
  const storage = () => { try { return root.localStorage; } catch { return null; } };

  function load() {
    try { values = parse(storage()?.getItem(STORE) ?? 'null'); } catch { values = defaults(); }
    return values;
  }
  function save() {
    try { storage()?.setItem(STORE, JSON.stringify(values)); } catch { /* trình duyệt chặn lưu: vẫn chạy với giá trị trong bộ nhớ */ }
  }
  const get = (key) => values[key];
  function set(key, value) { values = sanitize({ ...values, [key]: value }); save(); }
  function reset() { values = defaults(); save(); }
  const exportJson = () => JSON.stringify(values, null, 2);
  // Trả về số khoá bị bỏ qua (khoá lạ / giá trị sai), để báo cho người dùng
  function importJson(text) {
    let raw;
    try { raw = JSON.parse(text); } catch { throw new Error('File không phải JSON hợp lệ'); }
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('File không phải object setting');
    values = sanitize(raw);
    save();
    return Object.keys(raw).filter((k) => values[k] !== raw[k]).length;
  }

  // Sinh khung chỉnh theo nhóm; đổi giá trị -> lưu ngay -> onChange(key)
  function renderPanel(el, onChange) {
    const doc = el.ownerDocument;
    el.replaceChildren();
    const groups = [...new Set(SCHEMA.map((s) => s.group))];
    for (const g of groups) {
      const fs = doc.createElement('fieldset');
      fs.className = 'set-group';
      fs.innerHTML = `<legend>${g}</legend>`;
      for (const s of SCHEMA.filter((x) => x.group === g)) {
        const id = 'set-' + s.key.replace('.', '-');
        const row = doc.createElement('div');
        row.className = 'set-row' + (s.type === 'bool' ? ' set-bool' : '');
        let input;
        if (s.type === 'bool') {
          input = doc.createElement('input');
          input.type = 'checkbox'; input.role = 'switch'; input.checked = values[s.key];
          input.onchange = () => { set(s.key, input.checked); onChange(s.key); };
        } else if (s.type === 'int') {
          input = doc.createElement('input');
          input.type = 'number'; input.min = s.min; input.max = s.max; input.value = values[s.key]; input.inputMode = 'numeric';
          input.onchange = () => { set(s.key, Number(input.value)); input.value = values[s.key]; onChange(s.key); };
        } else {
          input = doc.createElement('select');
          for (const [v, text] of s.options) input.add(new Option(text, String(v), false, v === values[s.key]));
          input.onchange = () => { set(s.key, s.options.find(([v]) => String(v) === input.value)[0]); onChange(s.key); };
        }
        input.id = id;
        const label = doc.createElement('label');
        label.htmlFor = id; label.textContent = s.label;
        row.append(label, input);
        if (s.help) {
          const help = doc.createElement('small');
          help.className = 'set-help'; help.id = id + '-help'; help.textContent = s.help;
          input.setAttribute('aria-describedby', help.id);
          row.append(help);
        }
        fs.append(row);
      }
      el.append(fs);
    }
  }

  const api = { SCHEMA, defaults, sanitize, parse, load, save, get, set, reset, exportJson, importJson, renderPanel };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Settings = api;
})(typeof window !== 'undefined' ? window : globalThis);

'use strict';
// Dùng chung cho index.html và board.html: hiện người đang đăng nhập + nút đăng xuất, ẩn nút chỉ dành cho admin,
// và đưa về trang đăng nhập khi phiên hết hạn (API trả 401).
(function session() {
  // Thanh trang dùng chung: mọi trang có <nav data-nav> trong header.top.
  const PAGES = [['/', 'Ảnh'], ['/objects.html', 'Đồ vật'], ['/action.html', 'Hành động'], ['/quick.html', 'Danh tính'], ['/board.html', 'Định danh']];
  const nav = document.querySelector('[data-nav]');
  if (nav) {
    const here = location.pathname === '/index.html' ? '/' : location.pathname;
    for (const [href, text] of PAGES) {
      const a = document.createElement('a');
      a.href = href; a.textContent = text;
      if (href === here) a.setAttribute('aria-current', 'page');
      nav.append(a);
    }
  }
  const origFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const res = await origFetch(...args);
    if (res.status === 401 && !String(args[0]).includes('/api/login')) location.href = `/login.html?next=${encodeURIComponent(location.pathname + location.hash)}`;
    return res;
  };
  // Nút đồng bộ nhãn (journal.js): đẩy nhãn của mình lên, kéo nhãn người khác về rồi tải lại trang.
  origFetch('/api/sync').then((r) => r.json()).then((st) => {
    const bar = document.querySelector('header.top');
    if (!st.enabled || !bar) return;
    const btn = document.createElement('button');
    btn.className = 'ghost sync';
    const paint = (s) => {
      const bad = s.error || s.mode !== 'remote';
      btn.innerHTML = `<span class="sync-dot${bad ? ' warn' : ''}"></span>Đồng bộ${s.pending ? ` <span class="count">${s.pending}</span>` : ''}`;
      btn.title = s.error ? s.error
        : s.mode === 'remote' ? `Nhánh ${s.branch}. ${s.pending ? `${s.pending} thay đổi chưa đẩy (tự đẩy mỗi 5 phút và khi tắt server).` : 'Đã đẩy hết.'}${s.lastPush ? ` Lần đẩy cuối: ${new Date(s.lastPush).toLocaleTimeString()}.` : ''}`
        : 'Chưa có git remote "origin": nhãn chỉ ghi trên máy này.';
    };
    paint(st);
    btn.onclick = async () => {
      btn.disabled = true;
      btn.textContent = 'Đang đồng bộ…';
      try {
        const res = await origFetch('/api/sync', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
        const r = await res.json();
        if (!res.ok) throw new Error(r.error);
        if (r.applied) { alert(`Đã nhận ${r.applied} thay đổi của người khác. Trang sẽ tải lại.`); location.reload(); return; }
        paint(r);
      } catch (e) { alert(`Không đồng bộ được: ${e.message}`); }
      btn.disabled = false;
    };
    const help = bar.querySelector('#help');
    help ? help.before(btn) : bar.append(btn);
    setInterval(() => origFetch('/api/sync').then((r) => r.json()).then((s) => { if (!btn.disabled) paint(s); }).catch(() => {}), 30000);
  }).catch(() => {});
  origFetch('/api/me').then((r) => r.json()).then((me) => {
    if (!me.admin) for (const el of document.querySelectorAll('[data-admin]')) el.hidden = true;
    if (!me.auth) return;
    const bar = document.querySelector('header.top');
    const who = document.createElement('span');
    who.className = 'muted small';
    who.textContent = me.user + (me.admin ? ' (admin)' : '');
    const out = document.createElement('button');
    out.className = 'ghost';
    out.textContent = 'Đăng xuất';
    out.onclick = async () => { await origFetch('/api/logout', { method: 'POST' }); location.href = '/login.html'; };
    bar.append(who, out);
  }).catch(() => {});
})();

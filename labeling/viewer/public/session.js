'use strict';
// Dùng chung cho index.html và board.html: hiện người đang đăng nhập + nút đăng xuất, ẩn nút chỉ dành cho admin,
// và đưa về trang đăng nhập khi phiên hết hạn (API trả 401).
(function session() {
  const origFetch = window.fetch.bind(window);
  window.fetch = async (...args) => {
    const res = await origFetch(...args);
    if (res.status === 401 && !String(args[0]).includes('/api/login')) location.href = `/login.html?next=${encodeURIComponent(location.pathname + location.hash)}`;
    return res;
  };
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

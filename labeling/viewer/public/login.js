'use strict';
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const err = document.getElementById('err');
  err.textContent = '';
  const res = await fetch('/api/login', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user: document.getElementById('u').value.trim(), password: document.getElementById('p').value }) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) { err.textContent = data.error || 'Không đăng nhập được.'; return; }
  const next = new URLSearchParams(location.search).get('next') || '/';
  location.href = next.startsWith('/') && !next.startsWith('//') ? next : '/';
});

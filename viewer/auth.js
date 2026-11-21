// Đăng nhập cho viewer khi mở ra ngoài (Cloudflare Tunnel...). Tài khoản nằm ở viewer/users.json:
//   {"secret": "<khoá ký cookie>", "users": {"an": {"salt", "hash", "admin"}}}
// Có file này thì mọi trang và API đều bắt đăng nhập; không có thì chạy như trước (chỉ dùng trên máy này).
//
//   npm run adduser -- an            tạo tài khoản, in mật khẩu ngẫu nhiên MỘT lần
//   npm run adduser -- an --admin    tài khoản được bấm "Gộp dữ liệu" và "Tải lại"
//   npm run adduser -- an --reset    đặt lại mật khẩu
//   npm run deluser -- an
'use strict';
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const USERS_FILE = process.env.VIEWER_USERS || path.join(__dirname, 'users.json');
const COOKIE = 'castle_sid';
const SESSION_DAYS = 14;
const NAME = /^[\w.-]{2,32}$/;

const load = () => (fs.existsSync(USERS_FILE) ? JSON.parse(fs.readFileSync(USERS_FILE, 'utf8')) : null);
function save(doc) {
  fs.writeFileSync(`${USERS_FILE}.tmp`, JSON.stringify(doc, null, 2), { mode: 0o600 });
  fs.renameSync(`${USERS_FILE}.tmp`, USERS_FILE);
}
const hashPassword = (password, salt) => crypto.scryptSync(password, salt, 64).toString('hex');

class Auth {
  constructor() {
    this.doc = load();
    this.fails = new Map(); // ip -> [thời điểm thử sai]
  }

  get enabled() { return !!this.doc && Object.keys(this.doc.users || {}).length > 0; }

  check(user, password) {
    const u = this.doc?.users?.[user];
    if (!u || typeof password !== 'string') return null;
    const a = Buffer.from(hashPassword(password, u.salt), 'hex');
    const b = Buffer.from(u.hash, 'hex');
    return a.length === b.length && crypto.timingSafeEqual(a, b) ? { name: user, admin: !!u.admin } : null;
  }

  // Tối đa 8 lần sai mỗi 15 phút cho một IP.
  limited(ip) {
    const now = Date.now();
    const list = (this.fails.get(ip) || []).filter((t) => now - t < 15 * 60 * 1000);
    this.fails.set(ip, list);
    return list.length >= 8;
  }
  failed(ip) { this.fails.set(ip, [...(this.fails.get(ip) || []), Date.now()]); }

  // Cookie = tên.hạn.chữ ký HMAC: sống qua lần khởi động lại server, không lưu phiên trong RAM.
  issue(user) {
    const exp = Date.now() + SESSION_DAYS * 864e5;
    const body = `${Buffer.from(user).toString('base64url')}.${exp}`;
    return `${body}.${crypto.createHmac('sha256', this.doc.secret).update(body).digest('base64url')}`;
  }

  verify(token) {
    const [u, exp, sig] = String(token || '').split('.');
    if (!u || !exp || !sig || Number(exp) < Date.now()) return null;
    const want = crypto.createHmac('sha256', this.doc.secret).update(`${u}.${exp}`).digest('base64url');
    if (want.length !== sig.length || !crypto.timingSafeEqual(Buffer.from(want), Buffer.from(sig))) return null;
    const name = Buffer.from(u, 'base64url').toString();
    const rec = this.doc.users[name];
    return rec ? { name, admin: !!rec.admin } : null; // tài khoản đã xoá thì cookie cũ hết hiệu lực
  }

  userOf(req) {
    const m = /(?:^|;\s*)castle_sid=([^;]+)/.exec(req.headers.cookie || '');
    return m ? this.verify(decodeURIComponent(m[1])) : null;
  }

  cookie(req, value, maxAgeSec) {
    // Sau Cloudflare Tunnel trình duyệt dùng HTTPS: thêm Secure để cookie không đi qua HTTP thường.
    const secure = req.headers['x-forwarded-proto'] === 'https' || !!req.headers['cf-connecting-ip'];
    return `${COOKIE}=${encodeURIComponent(value)}; Path=/; HttpOnly; SameSite=Lax; Max-Age=${maxAgeSec}${secure ? '; Secure' : ''}`;
  }
}

const clientIp = (req) => req.headers['cf-connecting-ip'] || String(req.headers['x-forwarded-for'] || '').split(',')[0].trim() || req.socket.remoteAddress;

function cli(argv) {
  const [cmd, name, ...flags] = argv;
  if (!['adduser', 'deluser'].includes(cmd) || !NAME.test(name || '')) {
    console.error('Dùng: node auth.js adduser <tên> [--admin] [--reset] | node auth.js deluser <tên>   (tên: 2–32 ký tự chữ, số, . _ -)');
    process.exit(1);
  }
  const doc = load() || { secret: crypto.randomBytes(32).toString('hex'), users: {} };
  if (cmd === 'deluser') {
    if (!doc.users[name]) { console.error(`Không có tài khoản ${name}.`); process.exit(1); }
    delete doc.users[name];
    save(doc);
    return console.log(`Đã xoá ${name}. Cookie đăng nhập cũ của tài khoản này hết hiệu lực ngay.`);
  }
  if (doc.users[name] && !flags.includes('--reset')) { console.error(`${name} đã có. Thêm --reset để đặt lại mật khẩu.`); process.exit(1); }
  const password = crypto.randomBytes(12).toString('base64url');
  const salt = crypto.randomBytes(16).toString('hex');
  doc.users[name] = { salt, hash: hashPassword(password, salt), admin: flags.includes('--admin') || !!doc.users[name]?.admin };
  save(doc);
  console.log(`Tài khoản: ${name}${doc.users[name].admin ? ' (admin)' : ''}\nMật khẩu:  ${password}\nMật khẩu chỉ hiện một lần. Khởi động lại server để áp dụng nếu đây là tài khoản đầu tiên.`);
}

if (require.main === module) cli(process.argv.slice(2));
module.exports = { Auth, clientIp, hashPassword, USERS_FILE };

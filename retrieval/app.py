"""Giao diện tìm ảnh + nộp DRES, chạy trên máy (chỉ nghe 127.0.0.1).

    run_app.bat   (hoặc: .venv/Scripts/python.exe retrieval/app.py [--no-browser]); trang mở ở http://127.0.0.1:8314
Cấu hình DRES (.env) sửa được ngay trên trang; nút "Truy vấn lại" kéo đề trực tiếp từ DRES.
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from engine import ROOT
from text_engine import TextEngine

sys.path.insert(0, str(ROOT / "dres"))
import requests  # noqa: E402
from dotenv import set_key  # noqa: E402
import fetch_task as ft  # noqa: E402  (get_state/get_hint/hint_to_text)
import submit_task as dres  # noqa: E402  (đọc .env, dùng lại login/submit/log)

PORT = 8314
IMAGES, THUMBS = ROOT / "data" / "dataset" / "images", ROOT / "data" / "dataset" / "thumbs"
PAGE = Path(__file__).with_name("index.html")
ENGINE: TextEngine
STATE: dict = {}
ENV_FILE = ROOT / ".env"
# khoá .env -> (tên biến trong submit_task/fetch_task, giá trị mặc định khi để trống)
CONFIG = {"DRES_URL": ("DRES_URL", "https://dres.2hbread.com/api/v2"), "DRES_USERNAME": ("USERNAME", None),
          "DRES_PASSWORD": ("PASSWORD", None), "DRES_COLLECTION": ("COLLECTION", "castle5000"),
          "DRES_ITEM_PREFIX": ("ITEM_PREFIX", "day1_Kitchen_"), "DRES_EVALUATION_ID": ("EVALUATION_ID", None),
          "DRES_SESSION_ID": ("SESSION_ID", None)}


def get_config() -> dict:
    """Giá trị đang dùng; mật khẩu không gửi ra trình duyệt, chỉ báo đã có hay chưa."""
    out = {k: getattr(dres, attr) or "" for k, (attr, _) in CONFIG.items() if k != "DRES_PASSWORD"}
    return {**out, "DRES_PASSWORD": "", "hasPassword": bool(dres.PASSWORD)}


def set_config(body: dict) -> dict:
    """Ghi vào .env và áp dụng ngay cho phiên đang chạy (không cần khởi động lại)."""
    for key, (attr, default) in CONFIG.items():
        if key not in body or (key == "DRES_PASSWORD" and not str(body[key])):  # mật khẩu trống = giữ nguyên
            continue
        value = str(body[key]).strip()
        set_key(ENV_FILE, key, value)
        live = (value.rstrip("/") if key == "DRES_URL" else value) or default
        for module in (dres, ft):
            if hasattr(module, attr):
                setattr(module, attr, live)
    STATE.clear()  # đăng nhập lại với cấu hình mới
    return get_config()


def ensure_session(fresh: bool = False) -> None:
    if fresh or "session" not in STATE:
        STATE["session"] = (not fresh and dres.SESSION_ID) or dres.login()
        STATE["evaluation_id"] = dres.EVALUATION_ID or dres.active_evaluation_id(STATE["session"])


def pull_task() -> dict:
    """Kéo đề của task đang chạy từ DRES; không có task chạy thì trả đề đã lưu gần nhất."""
    for attempt in range(2):
        ensure_session(fresh=bool(attempt))
        try:
            state = ft.get_state(STATE["session"], STATE["evaluation_id"])
            break
        except requests.HTTPError as exc:
            if attempt or exc.response is None or exc.response.status_code != 401:
                raise
    running = state.get("taskStatus") == "RUNNING"
    if running and state.get("taskTemplateId"):
        text = ft.hint_to_text(ft.get_hint(STATE["session"], STATE["evaluation_id"], state["taskTemplateId"]))
        record = {"evaluationId": STATE["evaluation_id"], "taskId": state.get("taskId"),
                  "taskTemplateId": state["taskTemplateId"], "timeLeftMs": state.get("timeLeft"), "query": text}
        ft.OUT_JSON.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        ft.OUT_TXT.write_text(text, encoding="utf-8")
        return {**record, "running": True}
    saved = json.loads(ft.OUT_JSON.read_text(encoding="utf-8")) if ft.OUT_JSON.exists() else {}
    return {**saved, "running": False, "taskStatus": state.get("taskStatus")}


def neighbors(image_id: str, span: int = 6) -> list[str]:
    hour = image_id.split("_")[0]
    same = sorted((i for i in ENGINE.ids if i.startswith(hour + "_")), key=lambda i: int(i.split("_")[1]))
    k = same.index(image_id)
    return same[max(0, k - span): k + span + 1]


def summary(image_id: str) -> dict:
    r = ENGINE.records[image_id]
    people = []
    for p in r["people"]:
        o = p.get("attributes") or {}
        people.append({"id": p["id"], "name": (p.get("identity") or {}).get("name"), "action": (p.get("action") or {}).get("label_vi"),
                       "look": " ".join(str(o.get(k, "")) for k in ("gender", "top", "top_color", "top_pattern") if o.get(k)), "box": p["box"]})
    rois = [f'{v["name_vi"]}: {(v["state"] or {}).get("label_vi") or ""}' for v in r["rois"].values() if v["changed"] and v["state"]]
    return {"id": image_id, "people": people, "rois": rois, "objects": sorted({o["label"] for o in r["objects"]})}


def submit(image_id: str) -> dict:
    """Nộp một ảnh; đăng nhập lại một lần nếu phiên hết hạn (401)."""
    if image_id not in ENGINE.records:
        return {"status": 400, "message": "Không có ảnh này trong dữ liệu"}
    item = dres.to_item_name(image_id)
    for attempt in range(2):
        ensure_session(fresh=bool(attempt))
        status, text = dres.submit(STATE["session"], STATE["evaluation_id"], item)
        if status != 401:
            break
    dres.log(item, STATE["evaluation_id"], status, text)
    return {"status": status, "item": item, "message": dres.describe(status, text), "raw": text}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, kind: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, code: int = 200) -> None:
        self._send(json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8", code)

    def do_GET(self) -> None:
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if url.path == "/api/search":
            return self._json(ENGINE.search(q.get("q", ""), int(q.get("top", 300))))
        if url.path == "/api/task":
            try:
                return self._json(pull_task())
            except Exception as exc:  # lỗi mạng / đăng nhập: báo lên giao diện
                return self._json({"error": f"Lỗi: {exc}"}, 500)
        if url.path == "/api/config":
            return self._json(get_config())
        if url.path == "/api/image" and q.get("id") in ENGINE.records:     # chỉ id có thật: không cho đọc file tuỳ ý
            folder = THUMBS if q.get("thumb") and (THUMBS / f"{q['id']}.jpg").exists() else IMAGES
            return self._send((folder / f"{q['id']}.jpg").read_bytes(), "image/jpeg")
        if url.path == "/api/detail" and q.get("id") in ENGINE.records:
            return self._json({**summary(q["id"]), "neighbors": neighbors(q["id"])})
        self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/submit", "/api/config"):
            return self._json({"error": "not found"}, 404)
        # chỉ nhận JSON: trang web lạ không gửi được form "đơn giản" tới localhost để nộp bài/đổi cấu hình
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self._json({"error": "cần Content-Type: application/json"}, 415)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            self._json(set_config(body) if path == "/api/config" else submit(str(body.get("id", ""))))
        except Exception as exc:  # lỗi mạng / đăng nhập: báo lên giao diện thay vì làm sập server
            self._json({"status": 0, "message": f"Lỗi: {exc}"}, 500)

    def log_message(self, *args) -> None:  # tắt log mỗi request
        pass


if __name__ == "__main__":
    print("Đang nạp model (khoảng 1–2 phút)...")
    ENGINE = TextEngine()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Sẵn sàng: http://127.0.0.1:{PORT}  (Ctrl+C để tắt)")
    if "--no-browser" not in sys.argv:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{PORT}")
    server.serve_forever()

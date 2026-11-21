"""Kéo đề (hint) của task đang chạy trên DRES về máy.

Chạy:  python dres/fetch_task.py          # theo dõi liên tục, in đề mới ngay khi BTC bắt đầu task
       python dres/fetch_task.py --once   # lấy đề hiện tại 1 lần rồi thoát

Đề mới nhất được ghi ra data/runtime/current_task.json (và current_task.txt) để hệ thống
tìm kiếm đọc. Đường dẫn tính theo vị trí file này nên chạy từ thư mục nào cũng được.
Cấu hình đọc từ .env ở thư mục gốc dự án.
Cần: pip install requests python-dotenv
"""
import base64
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # .env ở thư mục gốc dự án
DRES_URL = os.getenv("DRES_URL", "https://dres.2hbread.com/api/v2").rstrip("/")
USERNAME = os.getenv("DRES_USERNAME")
PASSWORD = os.getenv("DRES_PASSWORD")
EVALUATION_ID = os.getenv("DRES_EVALUATION_ID") or None
POLL_SECONDS = float(os.getenv("DRES_POLL_SECONDS", "1"))
RUNTIME_DIR = Path(__file__).resolve().parents[1] / "data" / "runtime"   # máy sinh, không vào git
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = RUNTIME_DIR / "current_task.json"
OUT_TXT = RUNTIME_DIR / "current_task.txt"


def login():
    r = requests.post(f"{DRES_URL}/login", json={"username": USERNAME, "password": PASSWORD}, timeout=10)
    r.raise_for_status()
    return r.json()["sessionId"]


def active_evaluation_id(session):
    r = requests.get(f"{DRES_URL}/client/evaluation/list", params={"session": session}, timeout=10)
    r.raise_for_status()
    active = [e for e in r.json() if e.get("status") == "ACTIVE"]
    if not active:
        raise RuntimeError("Chưa có buổi thi nào đang ACTIVE")
    return active[0]["id"]


def get_state(session, evaluation_id):
    r = requests.get(f"{DRES_URL}/evaluation/{evaluation_id}/state", params={"session": session}, timeout=10)
    r.raise_for_status()
    return r.json()


def get_hint(session, evaluation_id, template_id):
    r = requests.get(
        f"{DRES_URL}/evaluation/{evaluation_id}/template/task/{template_id}/hint",
        params={"session": session}, timeout=10,
    )
    r.raise_for_status()
    return r.json()


def hint_to_text(hint):
    """Ghép các phần TEXT của hint; phần ảnh/video được lưu ra file."""
    texts = []
    for i, el in enumerate(hint.get("sequence", [])):
        ctype = el.get("contentType")
        content = el.get("content") or ""
        if ctype == "TEXT":
            if content not in texts:  # hint có thể lặp lại cùng câu theo thời gian
                texts.append(content)
        elif content:
            ext = {"IMAGE": "jpg", "VIDEO": "mp4"}.get(ctype, "bin")
            payload = content.split(",", 1)[-1]  # bỏ tiền tố data:...;base64, nếu có
            path = RUNTIME_DIR / f"hint_{i}.{ext}"
            with open(path, "wb") as f:
                f.write(base64.b64decode(payload))
            texts.append(f"[{ctype} -> {path}]")
    return "\n".join(texts)


def main():
    once = "--once" in sys.argv
    session = login()
    evaluation_id = EVALUATION_ID or active_evaluation_id(session)
    print(f"Evaluation: {evaluation_id}")

    last_task = None
    while True:
        try:
            state = get_state(session, evaluation_id)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 401:
                session = login()
                continue
            raise

        task_id = state.get("taskId")
        template_id = state.get("taskTemplateId")
        running = state.get("taskStatus") == "RUNNING"

        if running and template_id and task_id != last_task:
            hint = get_hint(session, evaluation_id, template_id)
            text = hint_to_text(hint)
            record = {
                "evaluationId": evaluation_id,
                "taskId": task_id,
                "taskTemplateId": template_id,
                "timeLeftMs": state.get("timeLeft"),
                "query": text,
            }
            with open(OUT_JSON, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            with open(OUT_TXT, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"\n=== TASK MỚI ({task_id}) | còn {(state.get('timeLeft') or 0) // 1000}s ===\n{text}\n")
            last_task = task_id

        if once:
            if not running:
                print(f"Không có task đang chạy (taskStatus={state.get('taskStatus')})")
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

"""Nộp một ảnh lên DRES cho task đang chạy.

Chạy:  python submission/submit_task.py                        # CHẾ ĐỘ NHẬP: gõ tên ảnh, nộp, gõ tiếp...
       python submission/submit_task.py 10_1528.jpg            # nộp 1 ảnh, hỏi xác nhận trước khi nộp
       python submission/submit_task.py day1_Kitchen_10_1528   # nộp thẳng tên ảnh (imageID) cũng được
       python submission/submit_task.py 10_1528.jpg --yes      # bỏ qua bước xác nhận
       python submission/submit_task.py --check                # kiểm tra kết nối: nộp tên giả khi KHÔNG có task chạy

Đọc cấu hình từ .env ở thư mục gốc dự án (DRES_URL, DRES_USERNAME, DRES_PASSWORD, DRES_COLLECTION,
DRES_ITEM_PREFIX, DRES_EVALUATION_ID, DRES_SESSION_ID).
Evaluation ID lấy theo thứ tự: .env -> current_task.json (do fetch_task.py ghi) -> buổi thi ACTIVE.
Cần: pip install requests python-dotenv

LƯU Ý: mỗi lần nộp SAI trong lúc task đang chạy đều bị trừ điểm.
"""
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

for _stream in (sys.stdout, sys.stderr, sys.stdin):  # in/nhập tiếng Việt trên console Windows
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # .env ở thư mục gốc dự án
DRES_URL = os.getenv("DRES_URL", "https://dres.2hbread.com/api/v2").rstrip("/")
USERNAME = os.getenv("DRES_USERNAME")
PASSWORD = os.getenv("DRES_PASSWORD")
COLLECTION = os.getenv("DRES_COLLECTION") or "castle5000"
ITEM_PREFIX = os.getenv("DRES_ITEM_PREFIX") or "day1_Kitchen_"
EVALUATION_ID = os.getenv("DRES_EVALUATION_ID") or None
SESSION_ID = os.getenv("DRES_SESSION_ID") or None
RUNTIME_DIR = Path(__file__).resolve().parents[1] / "data" / "runtime"   # máy sinh, không vào git
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
TASK_JSON = RUNTIME_DIR / "current_task.json"
LOG_FILE = RUNTIME_DIR / "submissions.log"

# Ý nghĩa các mã trả về (theo tài liệu DRES_HuongDan_Castle5000.pdf, mục 7.3)
MEANING = {
    400: "Không có task đang chạy (wrong state) HOẶC sai tên ảnh/collection",
    401: "Session sai hoặc hết hạn -> đăng nhập lại",
    404: "Sai Evaluation ID hoặc tài khoản không thuộc buổi thi",
    412: "Bị từ chối: ảnh đã nộp rồi, hoặc team đã nộp đúng task này",
    500: "Lỗi máy chủ (thường do sai tên ảnh: Could not find media item)",
}


def login():
    if not USERNAME or not PASSWORD:
        raise RuntimeError("Thiếu DRES_USERNAME / DRES_PASSWORD trong .env")
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


def evaluation_from_task_file():
    try:
        with open(TASK_JSON, encoding="utf-8") as f:
            return json.load(f).get("evaluationId")
    except (OSError, ValueError):
        return None


def to_item_name(name):
    """'10_1528.jpg' / 'data/images/10_1528.jpg' -> 'day1_Kitchen_10_1528'"""
    stem = name.replace("\\", "/").rsplit("/", 1)[-1]
    if stem.lower().endswith((".jpg", ".jpeg", ".png")):
        stem = stem.rsplit(".", 1)[0]
    return stem if stem.startswith(ITEM_PREFIX) else ITEM_PREFIX + stem


def submit(session, evaluation_id, item_name):
    body = {"answerSets": [{"answers": [{
        "mediaItemName": item_name,
        "mediaItemCollectionName": COLLECTION,
    }]}]}
    r = requests.post(f"{DRES_URL}/submit/{evaluation_id}",
                      params={"session": session}, json=body, timeout=15)
    return r.status_code, r.text


def describe(status, text):
    try:
        verdict = json.loads(text).get("submission")
    except (ValueError, AttributeError):
        verdict = None
    if verdict == "CORRECT":
        return "ĐÚNG - team đã có điểm task này, không cần nộp thêm"
    if verdict == "WRONG":
        return "SAI - bị trừ điểm, tìm tiếp rồi nộp lại"
    if verdict == "INDETERMINATE":
        return "ĐANG CHỜ CHẤM - không cần làm gì"
    return MEANING.get(status, "Không rõ - chụp lại mã và nội dung gửi BTC")


def log(item_name, evaluation_id, status, text):
    task_id = None
    try:
        with open(TASK_JSON, encoding="utf-8") as f:
            task_id = json.load(f).get("taskId")
    except (OSError, ValueError):
        pass
    from datetime import datetime
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "time": datetime.now().isoformat(timespec="seconds"),
            "evaluationId": evaluation_id, "taskId": task_id,
            "item": item_name, "status": status, "response": text,
        }, ensure_ascii=False) + "\n")


def current_query():
    try:
        with open(TASK_JSON, encoding="utf-8") as f:
            return (json.load(f).get("query") or "").strip().splitlines()[-1]
    except (OSError, ValueError, IndexError):
        return None


def do_submit(state, item_name, yes, check=False):
    """Hỏi xác nhận, nộp 1 ảnh, in kết quả. state = {'session':..., 'evaluation_id':...}"""
    print(f"Nộp ảnh    : {item_name}  (collection {COLLECTION})")
    if not yes:
        if input("Xác nhận nộp? Nộp sai bị trừ điểm. [y/N] ").strip().lower() != "y":
            print("Đã huỷ.")
            return None

    status, text = submit(state["session"], state["evaluation_id"], item_name)
    if status == 401:  # session hết hạn: đăng nhập lại 1 lần (401 = DRES chưa nhận bài)
        state["session"] = login()
        status, text = submit(state["session"], state["evaluation_id"], item_name)

    if not check:
        log(item_name, state["evaluation_id"], status, text)
    print(f"HTTP {status}: {text}")
    print(f"=> {describe(status, text)}")
    return text


def interactive(state, yes):
    """Nhập tên ảnh liên tục. Enter trống hoặc 'q' để thoát."""
    print("\n=== CHẾ ĐỘ NHẬP ẢNH ===")
    print("Nhập tên ảnh: 10_1528.jpg, 10_1528 hoặc day1_Kitchen_10_1528.")
    print("Lệnh: 'q' = thoát, 'd' = xem đề hiện tại.\n")
    submitted = set()
    while True:
        try:
            raw = input("Tên ảnh> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if raw.lower() in ("", "q", "quit", "exit"):
            break
        if raw.lower() == "d":
            print(f"Đề: {current_query() or '(chưa có current_task.json)'}")
            continue
        item_name = to_item_name(raw)
        if item_name in submitted:
            print("Ảnh này đã nộp trong phiên này (DRES sẽ báo 412). Bỏ qua.")
            continue
        text = do_submit(state, item_name, yes)
        if text is not None:
            submitted.add(item_name)
            if '"CORRECT"' in text:
                print("Task này đã ĐÚNG. Chờ task tiếp theo rồi nhập tiếp.")
        print()
    print("Thoát.")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    check = "--check" in sys.argv
    yes = "--yes" in sys.argv or "-y" in sys.argv

    session = SESSION_ID or login()
    state = {
        "session": session,
        "evaluation_id": EVALUATION_ID or evaluation_from_task_file() or active_evaluation_id(session),
    }
    print(f"Evaluation : {state['evaluation_id']}")
    q = current_query()
    if q:
        print(f"Đề         : {q}")

    if check:
        print("Chế độ --check: chỉ chạy khi KHÔNG có task. Mong đợi lỗi 400 'wrong state'.")
        do_submit(state, ITEM_PREFIX + "CHECK_CONNECTION", yes, check=True)
    elif args:
        do_submit(state, to_item_name(args[0]), yes)
    else:
        interactive(state, yes)


if __name__ == "__main__":
    main()

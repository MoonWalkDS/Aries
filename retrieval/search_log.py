"""Nhật ký tìm kiếm: mỗi lần tìm và mỗi lần nộp DRES ghi một dòng JSON vào data/runtime/search_logs/searches.jsonl.

Dùng sau buổi thi: thầy đưa id ảnh đúng của từng đề, `retrieval/rank_report.py` dò xem mỗi lần tìm đã đưa ảnh đúng
lên hạng mấy. Mỗi dòng mang tên hệ thống (branch + commit, hoặc biến môi trường SEARCH_SYSTEM) để so nhiều hệ thống.

    {"time", "kind": "search", "system", "commit", "task": {evaluationId, taskId, taskTemplateId}, "query", "edited",
     "edits", "top", "names", "count", "layers", "elapsed_ms", "results": [id, ...]}
    {"time", "kind": "submit", "system", "commit", "task", "image_id" | "text" (QA), "item", "status", "message"}
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from common import REPO_ROOT

LOG_DIR = REPO_ROOT / "data" / "runtime" / "search_logs"   # máy sinh, không vào git
LOG_FILE = LOG_DIR / "searches.jsonl"
TASK_FILE = REPO_ROOT / "data" / "runtime" / "current_task.json"   # submission/fetch_task.py ghi khi kéo đề
_LOCK = threading.Lock()   # web_app chạy nhiều luồng: hai lần ghi không được chen nhau


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


COMMIT = _git("rev-parse", "--short", "HEAD")
SYSTEM = os.getenv("SEARCH_SYSTEM") or _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"


def current_task() -> Optional[dict[str, Any]]:
    """Task DRES kéo gần nhất (có thể đã hết giờ); None nếu chưa kéo đề lần nào."""
    try:
        t = json.loads(TASK_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return {k: t.get(k) for k in ("evaluationId", "taskId", "taskTemplateId")}


def write(record: dict[str, Any], path: Optional[Path] = None) -> None:
    """Ghi thêm một dòng; lỗi ghi log không bao giờ được làm hỏng việc tìm/nộp."""
    path = path or LOG_FILE
    line = json.dumps({"time": datetime.now().isoformat(timespec="milliseconds"), "system": SYSTEM, "commit": COMMIT,
                       "task": current_task(), **record}, ensure_ascii=False)
    try:
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError as exc:
        print(f"search_log: không ghi được {path}: {exc}")


def log_search(query: str, top: int, edits: Optional[dict], response: dict[str, Any], elapsed_ms: float) -> None:
    parsed = response.get("parsed") or {}
    write({"kind": "search", "query": query, "edited": edits is not None, "edits": edits, "top": top,
           "names": response.get("names"), "count": parsed.get("count"), "layers": response.get("layers"),
           "elapsed_ms": round(elapsed_ms), "results": [r["id"] for r in response.get("results", [])]})


def log_submit(image_id: Optional[str], result: dict[str, Any], text: Optional[str] = None) -> None:
    """Một lần nộp: ảnh (image_id) cho task tìm ảnh, hoặc câu trả lời chữ (text) cho task QA."""
    write({"kind": "submit", "image_id": image_id, "text": text, "item": result.get("item"), "status": result.get("status"), "message": result.get("message")})

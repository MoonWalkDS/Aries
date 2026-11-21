"""Viết caption tiếng Việt bằng Codex CLI (``codex exec``), theo lô, từ ../exports/caption/inputs.jsonl.

    python caption_codex.py --model default --limit 25          # thử một lô, model mặc định của tài khoản Codex
    python caption_codex.py --model gpt-5.5 --workers 2         # chạy hết với model chỉ định

Dùng tài khoản đã đăng nhập trong Codex (``codex login``), không cần API key riêng. Codex chạy ở sandbox chỉ đọc,
trong thư mục exports/caption/codex, đề bài và lô ảnh gửi qua stdin; câu trả lời cuối lấy bằng
``--output-last-message``. Kết quả nối vào ../exports/caption/captions.jsonl như caption_opencode.py; dừng giữa
chừng thì chạy lại lệnh để làm tiếp.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from caption_opencode import drive, parser


def run_batch(exe: str, model: str, effort: str, task: Path, batch: Path, timeout: int) -> tuple[str, str | None]:
    """(câu trả lời cuối của Codex, lỗi nếu có)."""
    last = batch.with_suffix(".reply.txt")
    cmd = [exe, "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--color", "never",
           "-C", str(batch.parent), "--output-last-message", str(last), "-c", f"model_reasoning_effort={effort}"]
    if model != "default":
        cmd += ["--model", model]
    prompt = (task.read_text(encoding="utf-8") + "\n\nFILE BATCH (JSONL):\n" + batch.read_text(encoding="utf-8"))
    proc = subprocess.run(cmd + ["-"], input=prompt, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    text = last.read_text(encoding="utf-8") if last.exists() else ""
    last.unlink(missing_ok=True)
    err = None if proc.returncode == 0 else f"codex thoát mã {proc.returncode}: {(proc.stderr or proc.stdout)[-400:]}"
    return text, err


def main() -> None:
    ap = parser('model Codex, vd gpt-5.5; "default" = model mặc định của tài khoản')
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high"], help="mức suy luận (low đỡ tốn hạn mức)")
    args = ap.parse_args()
    exe = shutil.which("codex")
    if not exe:
        sys.exit("Không tìm thấy lệnh codex. Cài lại: npm install -g @openai/codex@latest --include=optional")
    drive("codex", args, lambda task, bf: run_batch(exe, args.model, args.effort, task, bf, args.timeout))


if __name__ == "__main__":
    main()

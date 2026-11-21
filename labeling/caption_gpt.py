"""Gọi GPT viết caption tiếng Việt từ ../exports/caption/inputs.jsonl (do caption_export.py dựng).

    $env:OPENAI_API_KEY = "..."            # key chỉ đọc từ biến môi trường, không ghi ra file
    python caption_gpt.py --model <tên model> --limit 10            # thử 10 ảnh trước
    python caption_gpt.py --model <tên model> --image --workers 4   # gửi kèm ảnh (đúng hơn, tốn hơn)

Ghi ../exports/caption/captions.jsonl: {image_id, caption, model}. Chạy lại thì bỏ qua ảnh đã có caption, nên dừng
giữa chừng không mất gì. Lỗi 429 / 5xx được thử lại có giãn cách.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from caption_export import OUT, ROOT

IMAGES = ROOT.parent / "dataset" / "images"
API = "https://api.openai.com/v1/chat/completions"


def load_api_config(env_path: Path | None = None) -> tuple[str, str]:
    """Read the project .env; explicit shell variables take precedence."""
    from dotenv import load_dotenv

    load_dotenv(env_path if env_path is not None else ROOT.parent / ".env", override=False)
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
    return key, base + "/chat/completions"


def image_part(image_id: str, side: int = 768) -> dict:
    from PIL import Image

    path = next(IMAGES.rglob(f"{image_id}.*"))
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "detail": "low"}}


def ask(rec: dict, prompt: str, model: str, key: str, with_image: bool, tries: int = 5, *, api: str = API) -> str:
    content: list = [{"type": "text", "text": json.dumps({k: v for k, v in rec.items() if k != "image_id"}, ensure_ascii=False)}]
    if with_image:
        content.append(image_part(rec["image_id"]))
    body = json.dumps({"model": model, "response_format": {"type": "json_object"},
                       "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": content}]}).encode()
    for i in range(tries):
        req = urllib.request.Request(api, body, {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "Accept": "application/json", "User-Agent": "DS314-API-Test/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=120) as res:
                text = json.loads(res.read())["choices"][0]["message"]["content"]
            return json.loads(text)["caption"].strip()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == tries - 1:
                raise RuntimeError(f"{rec['image_id']}: HTTP {e.code} {e.read()[:300]!r}") from None
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
            if i == tries - 1:
                raise
        time.sleep(2 ** i)
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="tên model GPT, vd model bạn đang dùng trên tài khoản OpenAI")
    ap.add_argument("--limit", type=int, default=0, help="chỉ làm N ảnh đầu chưa có caption (0 = tất cả)")
    ap.add_argument("--image", action="store_true", help="gửi kèm ảnh (độ phân giải thấp)")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    key, api = load_api_config()
    if not key:
        sys.exit("Thiếu biến môi trường OPENAI_API_KEY.")
    prompt = (OUT / "prompt_vi.md").read_text(encoding="utf-8")
    out_path = OUT / "captions.jsonl"
    done = {json.loads(l)["image_id"] for l in out_path.read_text(encoding="utf-8").splitlines()} if out_path.exists() else set()
    todo = [r for r in map(json.loads, (OUT / "inputs.jsonl").read_text(encoding="utf-8").splitlines()) if r["image_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    lock, failed = threading.Lock(), []

    def one(rec: dict) -> None:
        try:
            cap = ask(rec, prompt, args.model, key, args.image, api=api)
        except Exception as e:   # một ảnh lỗi không dừng cả lượt; chạy lại sẽ làm tiếp ảnh đó
            failed.append(rec["image_id"])
            print("LỖI", e, file=sys.stderr)
            return
        with lock, open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"image_id": rec["image_id"], "caption": cap, "model": args.model}, ensure_ascii=False) + "\n")
            print(f"{rec['image_id']}: {cap}")

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, todo))
    print(f"Xong {len(todo) - len(failed)}/{len(todo)} ảnh; lỗi {len(failed)} (chạy lại để làm tiếp).")


if __name__ == "__main__":
    main()

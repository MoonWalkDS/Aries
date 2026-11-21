"""Phần chạy theo lô dùng chung cho runner caption (caption/codex.py): chia ảnh thành lô, gọi model, đọc câu trả lời,
nối kết quả vào data/exports/caption/captions.jsonl. Ảnh thiếu hoặc lỗi được làm lại ở lần chạy sau."""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

from export import OUT

BATCH_RULES = """

CHẾ ĐỘ LÔ (thay cho dòng "Trả về JSON" ở trên):
File batch đính kèm là JSONL, mỗi dòng là đầu vào của MỘT ảnh (có trường image_id).
Với MỖI dòng, viết caption theo yêu cầu trên và trả về đúng một dòng JSON: {"image_id": "...", "caption": "..."}.
Giữ thứ tự, đủ mọi dòng, không thêm chữ nào khác, không dùng code block. Không chạy công cụ, không đọc hay sửa file nào khác.
Đây là lượt chạy tự động, không có ai trả lời: làm ngay, không lập kế hoạch, không hỏi lại, không chờ đồng ý."""
# Lời nhắn ngắn, không dấu, không ký tự đặc biệt: đi qua cmd.exe của npm shim. Hướng dẫn chung của agent (vd AGENTS.md
# "chờ người dùng đồng ý") làm model dừng lại hỏi, nên nói rõ đây là lượt chạy tự động, làm ngay.
MESSAGE = ("Day la luot chay tu dong khong co nguoi tra loi: lam ngay, khong lap ke hoach, khong xin phep. "
           "Lam theo huong dan trong file task dinh kem va chi tra ve cac dong JSON caption cho moi dong trong file batch dinh kem.")


def parse_reply(text: str, ids: set[str]) -> dict[str, str]:
    """Dòng JSON hợp lệ có image_id thuộc lô -> caption. Bỏ qua mọi thứ khác (chữ thừa, code fence)."""
    out = {}
    for line in text.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("image_id") in ids and isinstance(row.get("caption"), str) and row["caption"].strip():
            out[row["image_id"]] = row["caption"].strip()
    return out


def parser(model_help: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=model_help)
    ap.add_argument("--batch", type=int, default=25, help="số ảnh mỗi lần gọi")
    ap.add_argument("--limit", type=int, default=0, help="chỉ làm N ảnh đầu chưa có caption (0 = tất cả)")
    ap.add_argument("--workers", type=int, default=1, help="số lô chạy song song")
    ap.add_argument("--timeout", type=int, default=900, help="giây tối đa mỗi lô")
    return ap


def drive(tool: str, args: argparse.Namespace, call) -> None:
    """Chia lô ảnh chưa có caption, gọi ``call(task, batch_file) -> (chữ trả về, lỗi)``, nối caption hợp lệ vào
    captions.jsonl. Dùng chung cho opencode và codex."""
    out_path = OUT / "captions.jsonl"
    done = {json.loads(l)["image_id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()} if out_path.exists() else set()
    todo = [r for r in map(json.loads, (OUT / "inputs.jsonl").read_text(encoding="utf-8").splitlines()) if r["image_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    work = OUT / tool
    work.mkdir(exist_ok=True)
    task = work / "task.md"
    task.write_text((OUT / "prompt_vi.md").read_text(encoding="utf-8") + BATCH_RULES + "\n", encoding="utf-8")
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    lock, stats, stop = threading.Lock(), {"ok": 0, "miss": 0}, threading.Event()
    print(f"{len(todo)} ảnh chưa có caption, {len(batches)} lô x {args.batch}; {tool} {args.model}")

    def one(k: int, recs: list[dict]) -> None:
        if stop.is_set():
            return
        bf = work / f"batch_{recs[0]['image_id']}.jsonl"
        bf.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
        try:
            text, err = call(task, bf)
        except subprocess.TimeoutExpired:
            text, err = "", f"quá {args.timeout}s"
        got = parse_reply(text, {r["image_id"] for r in recs})
        if not got and not err:
            err = "câu trả lời không có dòng caption nào: " + " ".join(text.split())[:160]
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                for r in recs:
                    if r["image_id"] in got:
                        f.write(json.dumps({"image_id": r["image_id"], "caption": got[r["image_id"]], "model": f"{tool}:{args.model}"}, ensure_ascii=False) + "\n")
            stats["ok"] += len(got)
            stats["miss"] += len(recs) - len(got)
            print(f"lô {k + 1}/{len(batches)}: {len(got)}/{len(recs)} caption" + (f" | LỖI: {err}" if err else ""), flush=True)
            if err and any(s in err.lower() for s in ("balance", "quota", "credit", "402", "401", "usage limit", "rate limit")):
                stop.set()     # hết tiền / hết hạn mức / sai tài khoản: dừng, đừng đốt thêm lô
                print("Dừng: lỗi tài khoản/hạn mức. Đổi --model hoặc chờ/nạp thêm rồi chạy lại.", flush=True)
        if not err:
            bf.unlink(missing_ok=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda kv: one(*kv), enumerate(batches)))
    print(f"Xong: {stats['ok']} caption, thiếu {stats['miss']} (chạy lại lệnh để làm tiếp). Kết quả: {out_path}")

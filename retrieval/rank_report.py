"""Sau buổi thi: dò hạng của ảnh đúng trong mọi lần tìm đã ghi ở nhật ký (search_log.py), để phân tích và viết báo cáo.

    .venv/Scripts/python.exe retrieval/rank_report.py --truth dap_an.csv [--log data/runtime/search_logs/searches.jsonl]
                                                       [--window 3] [--out data/runtime/search_logs/report]

Đáp án (thầy đưa), một trong hai dạng:
    CSV:  task,image_ids        (task = taskId hoặc taskTemplateId của DRES; image_ids cách nhau bởi dấu cách, ; hoặc ,
                                 hoặc mỗi ảnh một dòng cùng task)
    JSON: {"<task>": ["08_333", "day1_Kitchen_08_334.jpg", ...]}
Id ảnh dạng nào cũng được (08_333, day1_Kitchen_08_333, ...jpg): chỉ lấy phần <giờ>_<khung> ở cuối.

Hai thước đo: "đúng ảnh" (đúng id thầy đưa) và "cửa sổ ±window khung cùng giờ" (sự kiện kéo dài vài khung,
DRES nhận một trong các ảnh đó). Số chính để báo cáo là cửa sổ ±3. Ảnh không nằm trong danh sách kết quả đã trả về
thì hạng để trống và ghi `> top`.

Kết quả trong --out:
    searches.csv  mỗi lần tìm: thời điểm, hệ thống, task, câu truy vấn, có sửa tay không, hạng đúng ảnh, hạng ±window
    tasks.csv     mỗi task: số lần tìm, hạng lần tìm đầu / tốt nhất / cuối, đã nộp mấy ảnh, có nộp trúng không
    summary.md    bảng Recall@K theo hệ thống (lần tìm đầu và lần tìm tốt nhất của mỗi task), in ra màn hình
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from common import EXPORTS_DIR
from search_log import LOG_FILE

KS = (1, 5, 10, 20, 50, 100)
ID_RE = re.compile(r"(\d{1,2}_\d+)(?:\.\w+)?$")


def norm_id(value: str) -> str:
    """'day1_Kitchen_08_333.jpg' -> '08_333'."""
    m = ID_RE.search(str(value).strip())
    if not m:
        raise ValueError(f"Không đọc được id ảnh: {value!r}")
    hour, frame = m.group(1).split("_")
    return f"{int(hour):02d}_{int(frame)}"


def load_truth(path: Path) -> dict[str, set[str]]:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): {norm_id(i) for i in v} for k, v in raw.items()}
    truth: dict[str, set[str]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            ids = [x for x in re.split(r"[\s;,]+", row.get("image_ids") or row.get("image_id") or "") if x]
            truth.setdefault(str(row["task"]).strip(), set()).update(norm_id(i) for i in ids)
    return truth


def frames_by_hour(image_ids: Iterable[str]) -> dict[str, list[str]]:
    by_hour: dict[str, list[str]] = {}
    for i in image_ids:
        by_hour.setdefault(i.split("_")[0], []).append(i)
    for v in by_hour.values():
        v.sort(key=lambda i: int(i.split("_")[1]))
    return by_hour


def widen(answers: set[str], by_hour: dict[str, list[str]], window: int) -> set[str]:
    """Ảnh đáp án + các khung ±window cùng giờ (theo thứ tự ảnh có trong bộ dữ liệu, như retrieval/evaluate.py)."""
    out = set(answers)
    for a in answers:
        hour = by_hour.get(a.split("_")[0], [])
        if a in hour:
            k = hour.index(a)
            out.update(hour[max(0, k - window): k + window + 1])
    return out


def rank(results: list[str], answers: set[str]) -> Optional[int]:
    return next((n for n, r in enumerate(results, 1) if r in answers), None)


def read_log(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def task_key(record: dict[str, Any], truth: dict[str, set[str]]) -> Optional[str]:
    t = record.get("task") or {}
    return next((str(t[k]) for k in ("taskId", "taskTemplateId") if t.get(k) is not None and str(t[k]) in truth), None)


def analyse(log: list[dict[str, Any]], truth: dict[str, set[str]], by_hour: dict[str, list[str]], window: int) -> tuple[list[dict], list[dict]]:
    wide = {k: widen(v, by_hour, window) for k, v in truth.items()}
    searches, tasks = [], {}
    for r in log:
        key = task_key(r, truth)
        if key is None:
            continue
        t = tasks.setdefault((r.get("system"), key), {"system": r.get("system"), "task": key, "ranks": [], "submits": 0, "submit_exact": False, "submit_window": False})
        if r["kind"] == "search":
            exact, win = rank(r["results"], truth[key]), rank(r["results"], wide[key])
            searches.append({"time": r["time"], "system": r.get("system"), "commit": r.get("commit"), "task": key, "query": r["query"],
                             "edited": r.get("edited"), "top": r.get("top"), "returned": len(r["results"]), "rank_exact": exact, f"rank_window{window}": win})
            t["ranks"].append((exact, win))
        elif r["kind"] == "submit":
            image = norm_id(r["image_id"])
            t["submits"] += 1
            t["submit_exact"] |= image in truth[key]
            t["submit_window"] |= image in wide[key]
    rows = []
    for t in tasks.values():
        ranks = t["ranks"]
        best = lambda i: min((x[i] for x in ranks if x[i] is not None), default=None)  # noqa: E731
        rows.append({"system": t["system"], "task": t["task"], "searches": len(ranks),
                     "first_exact": ranks[0][0] if ranks else None, f"first_window{window}": ranks[0][1] if ranks else None,
                     "best_exact": best(0), f"best_window{window}": best(1),
                     "last_exact": ranks[-1][0] if ranks else None, f"last_window{window}": ranks[-1][1] if ranks else None,
                     "submits": t["submits"], "submit_exact": t["submit_exact"], f"submit_window{window}": t["submit_window"]})
    return searches, rows


def recall_table(tasks: list[dict], window: int) -> str:
    lines = [f"| Hệ thống | Lần tìm | Thước đo | Số task | " + " | ".join(f"R@{k}" for k in KS) + " |",
             "|---|---|---|---|" + "---|" * len(KS)]
    for system in sorted({t["system"] for t in tasks}, key=str):
        mine = [t for t in tasks if t["system"] == system and t["searches"]]
        for which, label in (("first", "đầu tiên"), ("best", "tốt nhất")):
            for metric, mlabel in ((f"window{window}", f"±{window} khung"), ("exact", "đúng ảnh")):
                ranks = [t[f"{which}_{metric}"] for t in mine]
                cells = [f"{sum(r is not None and r <= k for r in ranks)}/{len(ranks)}" for k in KS]
                lines.append(f"| {system} | {label} | {mlabel} | {len(ranks)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig: Excel đọc đúng tiếng Việt
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: ("> top" if v is None and k.startswith(("rank_", "first_", "best_", "last_")) else v) for k, v in r.items()} for r in rows)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--log", type=Path, default=LOG_FILE)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--out", type=Path, default=LOG_FILE.parent / "report")
    args = ap.parse_args(argv)
    truth = load_truth(args.truth)
    ids_file = EXPORTS_DIR / "embeddings" / "image_ids.json"
    image_ids = [norm_id(i) for i in json.loads(ids_file.read_text(encoding="utf-8"))] if ids_file.exists() else sorted({i for v in truth.values() for i in v})
    log = read_log(args.log)
    searches, tasks = analyse(log, truth, frames_by_hour(image_ids), args.window)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "searches.csv", searches)
    write_csv(args.out / "tasks.csv", tasks)
    seen = {t["task"] for t in tasks}
    unmatched = sum(1 for r in log if task_key(r, truth) is None)
    text = (f"# Hạng của ảnh đúng\n\n{len(log)} dòng nhật ký, {len(searches)} lần tìm khớp {len(seen)}/{len(truth)} task có đáp án; "
            f"{unmatched} dòng không khớp task nào.\n\n{recall_table(tasks, args.window)}\n\n"
            f"Chưa tìm lần nào: {', '.join(sorted(set(truth) - seen)) or 'không có'}\n")
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Chi tiết: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

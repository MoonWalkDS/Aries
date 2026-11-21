"""Đánh giá engine trên dev_queries.json theo kiểu gợi ý tăng dần của DRES.

    python retrieval/evaluate.py
Mỗi đề được cắt thành tiền tố 1..n câu (như BTC thêm dần gợi ý). Ảnh đúng = ảnh đáp án hoặc khung lân cận
±3 trong cùng giờ (sự kiện kéo dài vài khung; DRES nhận một trong các ảnh đúng).
"""
from __future__ import annotations

import json
from pathlib import Path

from engine import Engine, split_sentences

WINDOW = 3


def main() -> None:
    engine = Engine()
    by_hour: dict[str, list[str]] = {}
    for i in engine.ids:
        by_hour.setdefault(i.split("_")[0], []).append(i)
    for v in by_hour.values():
        v.sort(key=lambda i: int(i.split("_")[1]))
    cases = json.loads(Path(__file__).with_name("dev_queries.json").read_text(encoding="utf-8"))
    first, full = [], []
    for c in cases:
        hour = by_hour[c["answer"].split("_")[0]]
        k = hour.index(c["answer"])
        answers = set(hour[max(0, k - WINDOW): k + WINDOW + 1])
        sentences = split_sentences(c["query"])
        ranks = [engine.rank_of(" ".join(sentences[:n]), answers) for n in range(1, len(sentences) + 1)]
        first.append(ranks[0])
        full.append(ranks[-1])
        print(f'{c["answer"]:>8}  hạng theo số câu: {ranks}')
    for name, ranks in (("1 câu", first), ("đủ câu", full)):
        print(f"{name}: R@1 {sum(r <= 1 for r in ranks)}/{len(ranks)}  R@10 {sum(r <= 10 for r in ranks)}/{len(ranks)}  R@50 {sum(r <= 50 for r in ranks)}/{len(ranks)}")


if __name__ == "__main__":
    main()

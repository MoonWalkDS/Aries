"""Đo nhãn hành động VLM so với người duyệt (adl_people).

    python audit_actions.py            # độ khớp trên mọi crop người đã gán / từ chối
    python audit_actions.py --sample   # tạo annotations/action_audit_sample.json: 300 crop ngẫu nhiên (seed 0) để gán tay độc lập

Số trên crop người đã đụng tới bị lệch (người duyệt hay mở crop đáng ngờ); muốn số cho bài báo thì gán tay
đủ action_audit_sample.json rồi chạy lại: phần "sample" khi đó là ước lượng không lệch.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

DIR = Path(__file__).resolve().parent / "annotations" / "adl_people"
SAMPLE = DIR.parent / "action_audit_sample.json"   # ngoài adl_people/ để git giữ (adl*/* bị ignore)


def main() -> None:
    approved = json.loads((DIR / "approved.json").read_text(encoding="utf-8"))
    rejected = json.loads((DIR / "rejected.json").read_text(encoding="utf-8"))
    vlm = {}
    for line in (DIR / "vlm_reviews.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("choice"):
            vlm[row["image_id"]] = row["choice"]          # câu trả lời cuối cùng của VLM cho crop
    if "--sample" in sys.argv:
        ids = sorted(approved)
        random.Random(0).shuffle(ids)
        SAMPLE.write_text(json.dumps(sorted(ids[:300]), indent=1), encoding="utf-8")
        print("wrote", SAMPLE)
        return
    human = {pid: e["label"] for pid, e in approved.items() if e.get("source") == "human"}
    agree = sum(vlm.get(pid) == lab for pid, lab in human.items() if pid in vlm)
    judged = sum(pid in vlm for pid in human)
    wrong_rejected = sum(vlm.get(pid) in labs for pid, labs in rejected.items())
    confusions = Counter((vlm[pid], lab) for pid, lab in human.items() if pid in vlm and vlm[pid] != lab)
    print(f"human-labeled crops with a VLM answer: {judged}; VLM agrees on {agree} ({agree / max(judged, 1):.1%})")
    print(f"crops where reviewer rejected exactly the VLM answer: {wrong_rejected} / {len(rejected)} rejected crops")
    print("top confusions (vlm -> human):", confusions.most_common(8))
    sample_path = SAMPLE
    if sample_path.exists():
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        done = [pid for pid in sample if pid in human and pid in vlm]
        if done:
            ok = sum(vlm[pid] == human[pid] for pid in done)
            print(f"sample (unbiased): {ok}/{len(done)} = {ok / len(done):.1%}  ({len(sample) - len(done)} still unlabeled)")
        else:
            print(f"sample: 0/{len(sample)} labeled yet")


if __name__ == "__main__":
    main()

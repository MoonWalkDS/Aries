"""VLM làm người duyệt vòng đầu cho vòng lặp adl (``python main.py adl vlm-review``).

Với mỗi ảnh hạt giống và mỗi gợi ý chưa duyệt, VLM nhìn crop và tự chọn một lớp trong bộ nhãn, KHÔNG
được cho biết nhãn mà máy đã gợi ý (để không bị neo theo). So lựa chọn đó với nhãn gợi ý:

- trùng            -> nhận nhãn, ``source: vlm``, ``vlm: agree``
- chọn lớp khác    -> nhận lớp của VLM, ``source: vlm``, ``vlm: changed``, giữ ``suggested`` để soát
- "không lớp nào"  -> nhãn gợi ý vào ``rejected`` (không gợi ý lại), ảnh rời golden corpus

Nhãn người đã duyệt (``source: human``) không bao giờ bị đụng tới. Mọi câu trả lời của VLM được ghi
vào ``vlm_reviews.jsonl`` nên chạy lại không hỏi lại và có thể soát về sau. Người duyệt xem lại phần
VLM đã quyết bằng ``adl export --what vlm`` (các chỗ VLM đổi nhãn xếp trước).
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image

from . import adl
from .utils import append_jsonl, get_logger, read_json, read_jsonl, resolve_from_config
from .vlm_annotator import extract_json, resize_long_side
from .vlm_backends import VLMBackend

NONE_CHOICE = 0

PROMPT = """This image is {viewpoint}.
Choose the ONE option that best describes {question}. If none fits, or the image is too unclear to tell, choose 0.

Options:
0. none of these / cannot tell
{options}

Reply with exactly one JSON object and nothing else: {{"choice": <option number>}}"""


def build_prompt(classes: Sequence[adl.AdlClass], viewpoint: str, question: str) -> str:
    """Danh sách lớp đánh số từ 1, mỗi lớp kèm câu mô tả đầu tiên làm gợi ý nghĩa."""
    options = "\n".join(f"{n}. {c.name} (e.g. {c.descriptions[0].rstrip('.')})" for n, c in enumerate(classes, start=1))
    return PROMPT.format(viewpoint=viewpoint, question=question, options=options)


def parse_choice(text: str, num_classes: int) -> Optional[int]:
    """Số lựa chọn trong câu trả lời (0..num_classes); None nếu không đọc được."""
    value: Any = None
    try:
        data = extract_json(text)
        value = data.get("choice") if isinstance(data, dict) else None
    except ValueError:
        pass
    if value is None:
        match = re.search(r"-?\d+", text)       # model đôi khi chỉ trả một con số
        value = match.group() if match else None
    try:
        choice = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return choice if 0 <= choice <= num_classes else None


def pending_items(store: adl.AdlStore) -> list[dict[str, Any]]:
    """Ảnh hạt giống chưa ai duyệt + gợi ý của vòng hiện tại chưa có trong golden corpus."""
    approved = store.approved()
    items = [{"image_id": i, "label": r["label"], "score": r.get("score"), "kind": "seed"} for i, r in approved.items() if r.get("source") == "seed"]
    items += [{"image_id": s["image_id"], "label": s["label"], "score": s.get("score"), "kind": "suggestion"} for s in store.suggestions()["items"] if s["image_id"] not in approved]
    return items


def unlabeled_items(cfg: dict[str, Any], store: adl.AdlStore) -> list[dict[str, Any]]:
    """Mọi crop chưa có trong golden corpus. Crop đang có gợi ý giữ nhãn gợi ý (để so); còn lại ``label: None``."""
    approved = store.approved()
    suggested = {s["image_id"]: s for s in store.suggestions()["items"]}
    items: list[dict[str, Any]] = []
    for name in read_json(resolve_from_config(cfg, cfg["adl"]["ids"])):
        image_id = Path(str(name)).stem
        if image_id in approved:
            continue
        s = suggested.get(image_id)
        items.append({"image_id": image_id, "label": s["label"] if s else None, "score": s.get("score") if s else None, "kind": "suggestion" if s else "fill"})
    return items


def apply_decisions(
    items: Sequence[dict[str, Any]], decisions: dict[tuple[str, str], Optional[str]],
    approved: dict[str, dict[str, Any]], rejected: dict[str, list[str]], round_no: int,
) -> dict[str, int]:
    """Áp quyết định của VLM lên ``approved`` / ``rejected`` (sửa tại chỗ).

    ``decisions``: {(image_id, nhãn gợi ý): tên lớp VLM chọn, hoặc "" nếu VLM chọn "không lớp nào"}.
    Mục không có trong ``decisions`` (chưa hỏi, hoặc VLM trả lời không đọc được) được giữ nguyên.
    Mục không có nhãn gợi ý (``label: None``, chế độ gán trực tiếp) nhận ``vlm: direct``; nếu VLM chọn
    "không lớp nào" thì crop đó vẫn không có nhãn (câu trả lời nằm trong vlm_reviews.jsonl).
    """
    stats = {"agree": 0, "changed": 0, "rejected": 0, "unreviewed": 0, "direct": 0, "none": 0}
    for item in items:
        image_id, suggested = item["image_id"], item["label"]
        if approved.get(image_id, {}).get("source") == "human":
            continue
        choice = decisions.get((image_id, suggested))
        if choice is None:
            stats["unreviewed"] += 1
        elif suggested is None:
            if choice == "":
                stats["none"] += 1
            else:
                approved[image_id] = {"label": choice, "source": "vlm", "round": round_no, "score": None, "vlm": "direct"}
                stats["direct"] += 1
        elif choice == "":
            if suggested not in rejected.setdefault(image_id, []):
                rejected[image_id].append(suggested)
            approved.pop(image_id, None)
            stats["rejected"] += 1
        else:
            record = {"label": choice, "source": "vlm", "round": round_no, "score": item.get("score"), "vlm": "agree" if choice == suggested else "changed"}
            if choice != suggested:
                record["suggested"] = suggested
            approved[image_id] = record
            stats[record["vlm"]] += 1
    return stats


def run_vlm_review(
    cfg: dict[str, Any], store: adl.AdlStore, backend: VLMBackend, limit: Optional[int] = None, fill: bool = False,
) -> dict[str, Any]:
    """Một lượt: hỏi VLM cho mọi mục đang chờ (tối đa ``limit`` lệnh gọi mới), rồi áp quyết định.

    ``fill``: hỏi MỌI crop chưa có nhãn (kể cả crop máy không gợi ý gì), thay vì chỉ hạt giống và gợi ý.
    Không gom cụm lại sau đó: golden corpus của vòng lặp chỉ lớn lên qua các lượt duyệt thường.
    """
    log = get_logger()
    vcfg = cfg["adl"]["vlm_review"]
    classes = adl.load_labels(resolve_from_config(cfg, cfg["adl"]["labels_file"]))
    names = [c.name for c in classes]
    prompt = build_prompt(classes, cfg["adl"]["viewpoint"], vcfg["question"])
    images_dir = adl.review_images_dir(cfg)
    extension = {Path(str(n)).stem: Path(str(n)).suffix for n in read_json(resolve_from_config(cfg, cfg["adl"]["ids"]))}
    reviews_path = store.root / "vlm_reviews.jsonl"
    # Chỉ dùng lại câu trả lời cũ khi bộ nhãn (và do đó prompt) không đổi.
    known: dict[str, Optional[str]] = {}
    if reviews_path.exists():
        for row in read_jsonl(reviews_path):
            if row.get("classes") == names and row.get("model") == backend.model_id:
                known[row["image_id"]] = row.get("choice")

    items = unlabeled_items(cfg, store) if fill else pending_items(store)
    calls = failed = 0
    started = time.perf_counter()
    for item in items:
        image_id = item["image_id"]
        if image_id in known or (limit is not None and calls >= limit):
            continue
        path = images_dir / f"{image_id}{extension.get(image_id, '.jpg')}"
        if not path.exists():
            log.warning("vlm-review: thiếu ảnh %s", path)
            continue
        with Image.open(path) as im:
            image = resize_long_side(im.convert("RGB"), int(vcfg["max_long_side"]))
        choice: Optional[int] = None
        raw = ""
        for _ in range(int(vcfg.get("retries", 1)) + 1):
            result = backend.generate([image], prompt, max_new_tokens=int(vcfg["max_new_tokens"]))
            calls += 1
            raw = result.text
            choice = parse_choice(raw, len(names))
            if choice is not None:
                break
        label = None if choice is None else ("" if choice == NONE_CHOICE else names[choice - 1])
        failed += label is None
        known[image_id] = label
        append_jsonl(reviews_path, {"image_id": image_id, "suggested": item["label"], "kind": item["kind"], "choice": label, "raw": raw[:200], "model": backend.model_id, "classes": names})
        if calls % (500 if fill else 50) == 0:
            log.info("vlm-review: %d lệnh gọi, %.1f s/lệnh", calls, (time.perf_counter() - started) / calls)

    approved, rejected = store.approved(), store.rejected()
    round_no = store.load_centroids()[2] if store.centroids_path.exists() else 0
    decisions = {(i["image_id"], i["label"]): known.get(i["image_id"]) for i in items if i["image_id"] in known}
    stats = apply_decisions(items, decisions, approved, rejected, round_no)
    store.save_approved(approved)
    store.save_rejected(rejected)
    seconds = time.perf_counter() - started
    return {**stats, "pending": len(items), "vlm_calls": calls, "unreadable": failed, "sec_per_call": round(seconds / calls, 2) if calls else None, "model": backend.model_id}


def run_vlm_loop(cfg: dict[str, Any], store: adl.AdlStore, backend: VLMBackend, rounds: int = 1, limit: Optional[int] = None) -> dict[str, Any]:
    """VLM duyệt -> gom cụm lại -> gợi ý, lặp tối đa ``rounds`` vòng hoặc tới khi hết gợi ý.

    ``limit`` là trần lệnh gọi VLM cho CẢ lệnh này; hết trần thì dừng, lần chạy sau làm tiếp.
    """
    history: list[dict[str, Any]] = []
    remaining = limit
    for _ in range(max(1, rounds)):
        review = run_vlm_review(cfg, store, backend, remaining)
        if remaining is not None:
            remaining -= review["vlm_calls"]
        update = adl.run_update(cfg, store)
        suggest = adl.run_suggest(cfg, store)
        history.append({"round": update["round"], **{k: review[k] for k in ("agree", "changed", "rejected", "unreviewed", "vlm_calls", "sec_per_call")}, "next_suggestions": suggest["suggestions"]})
        if not suggest["suggestions"] or (remaining is not None and remaining <= 0):
            break
    approved = store.approved()
    return {
        "rounds": history,
        "vlm_calls": sum(h["vlm_calls"] for h in history),
        "labeled_by_vlm": sum(1 for r in approved.values() if r.get("source") == "vlm"),
        "seeds_left": sum(1 for r in approved.values() if r.get("source") == "seed"),
        "suggestions_left": history[-1]["next_suggestions"],
    }

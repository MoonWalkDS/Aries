"""Chọn bộ nhãn ADL và sinh câu mô tả, theo LSC-ADL (arXiv:2504.02060) mục 3.2 và Stage 1 mục 3.3.

Ba lệnh, chạy trước ``adl init``:

``caption``         Lấy mẫu ảnh theo nhóm thời gian (bài báo: 20 ngày x 15 ảnh; Castle chỉ có một ngày
                    nên mỗi GIỜ là một nhóm), giữ thứ tự thời gian. VLM viết caption tập trung vào
                    hành động, rồi nối caption của mỗi nhóm thành một đoạn mô tả.
``propose-labels``  Với mỗi nhóm, LLM nhận nhãn thô + đoạn mô tả, chọn nhãn thật sự xuất hiện và đề
                    xuất nhãn mới. Sau đó bỏ trùng và LLM xếp nhãn vào nhóm ngữ nghĩa. Kết quả là
                    ``<labels_file>.proposed.yaml`` để NGƯỜI duyệt: gộp nhãn quá chi tiết, bỏ nhãn thừa.
``describe``        Decompose-Expand: với mỗi lớp, LLM sinh từ khoá vật thể + hành động, rồi sinh câu
                    mô tả từ tên lớp và từ khoá. Ghi vào file nhãn để người đọc lại trước khi ``init``.

Mọi lệnh gọi model đều đi qua ``VLMBackend`` nên đổi được giữa openai / api / qwen_local.
Kết quả trung gian nằm trong ``output_jsons/adl/label_selection/`` và các lệnh chạy lại được: caption
đã có thì không gọi lại, lớp đã đủ câu mô tả thì bỏ qua.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import yaml
from PIL import Image

from .utils import get_logger, image_id_of, read_json, write_json
from .vlm_annotator import extract_json, resize_long_side
from .vlm_backends import VLMBackend

UNGROUPED = "Uncategorized"

CAPTION_PROMPT = """This image is one frame from {viewpoint}.
In one or two English sentences, describe what the people are doing: their actions and the objects \
they are using or touching. If several people do different things, mention each briefly. If nobody \
is visible, say that the room is empty.
Do not describe clothing, appearance, furniture or decoration. Reply with the sentences only."""

PROPOSE_PROMPT = """Below is a chronological description of one time period recorded by {viewpoint}. \
It was built by concatenating captions of frames sampled in temporal order.

DESCRIPTION:
{description}

CANDIDATE ACTIVITY LABELS (collected from other activity-of-daily-living datasets):
{candidates}

Tasks:
1. "present": the candidate labels whose activity clearly occurs in the description. Copy them \
exactly. Leave out candidates that do not occur.
2. "new": activities that occur in the description but are not covered by any candidate. Name each \
as a short lowercase English verb phrase in the same style as the candidates (for example \
"opening the refrigerator"). Do not invent activities that are not in the description.

Reply with exactly one JSON object: {{"present": ["..."], "new": ["..."]}}"""

GROUP_PROMPT = """These activity labels were extracted from footage of {viewpoint}. The number is \
how many time periods each label was found in.

{labels}

Categorize every label into one broader activity group. Prefer these groups and add another only \
if none fits: {groups}.
Also give each label a short natural Vietnamese name.
Keep each label's "name" exactly as given and include every label exactly once. Do not merge, \
rename or drop labels.

Reply with exactly one JSON object:
{{"labels": [{{"name": "...", "name_vi": "...", "group": "..."}}]}}"""

KEYWORDS_PROMPT = """Activity class: "{name}".
Camera: {viewpoint}.

Decompose this activity into visual keywords that could appear in an image of it from this camera.
- "objects": 5 to 10 objects, appliances or places involved.
- "actions": 5 to 10 body poses or motions of the people doing it.

Reply with exactly one JSON object: {{"objects": ["..."], "actions": ["..."]}}"""

DESCRIBE_PROMPT = """Activity class: "{name}".
Camera: {viewpoint}.
Object keywords: {objects}
Action keywords: {actions}

Write {count} different English sentences, each describing what one image of this activity looks \
like from this camera. Rules:
- one sentence each, at most 20 words, concrete and visual, third person;
- cover different scenarios of the same activity (different objects, poses, number of people);
- describe only what can be seen, no intentions or sounds;
- do not mention the camera in every sentence.

Reply with exactly one JSON object: {{"descriptions": ["..."]}}"""

RETRY_SUFFIX = "\n\nYour previous reply could not be used: {error}\nReply again with exactly one valid JSON object and nothing else."


# --------------------------------------------------------------------------- gọi model
class Usage:
    """Cộng dồn số lệnh gọi và token để báo chi phí cuối mỗi lệnh."""

    def __init__(self) -> None:
        self.calls = self.input_tokens = self.output_tokens = 0

    def add(self, result: Any) -> None:
        self.calls += 1
        self.input_tokens += int(result.input_tokens)
        self.output_tokens += int(result.output_tokens)

    def as_dict(self) -> dict[str, int]:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


def ask_json(backend: VLMBackend, prompt: str, usage: Usage, retries: int = 1, max_new_tokens: int = 2000) -> dict[str, Any]:
    """Hỏi model (không ảnh) và parse một đối tượng JSON; JSON hỏng thì hỏi lại kèm lý do."""
    error: Optional[str] = None
    for _ in range(retries + 1):
        result = backend.generate([], prompt + (RETRY_SUFFIX.format(error=error) if error else ""), max_new_tokens=max_new_tokens)
        usage.add(result)
        try:
            data = extract_json(result.text)
            if not isinstance(data, dict):
                raise ValueError("câu trả lời không phải một đối tượng JSON")
            return data
        except ValueError as exc:
            error = str(exc)
    raise ValueError(f"Model không trả JSON hợp lệ sau {retries + 1} lần: {error}")


def _strings(value: Any) -> list[str]:
    """Danh sách chuỗi sạch từ một trường JSON (bỏ phần tử rỗng hoặc không phải chuỗi)."""
    if isinstance(value, str):
        value = [value]
    return [" ".join(v.split()) for v in value or [] if isinstance(v, str) and v.strip()]


# --------------------------------------------------------------------------- lấy mẫu ảnh
def group_key(image_id: str) -> str:
    """'10_1528' -> '10': nhóm thời gian của ảnh (giờ quay)."""
    return image_id.split("_", 1)[0]


def _order_key(image_id: str) -> tuple[int, Any]:
    tail = image_id.split("_", 1)[-1]
    return (0, int(tail)) if tail.isdigit() else (1, tail)


def sample_frames(image_ids: Sequence[str], num_groups: int, per_group: int, seed: int = 0) -> dict[str, list[str]]:
    """Chọn ngẫu nhiên tối đa ``num_groups`` nhóm, mỗi nhóm ``per_group`` ảnh rải đều theo thời gian."""
    groups: dict[str, list[str]] = {}
    for image_id in image_ids:
        groups.setdefault(group_key(image_id), []).append(image_id)
    keys = sorted(groups)
    if len(keys) > num_groups:
        keys = sorted(random.Random(seed).sample(keys, num_groups))
    frames: dict[str, list[str]] = {}
    for key in keys:
        members = sorted(groups[key], key=_order_key)
        picks = np.unique(np.linspace(0, len(members) - 1, min(per_group, len(members))).round().astype(int))
        frames[key] = [members[i] for i in picks]
    return frames


# --------------------------------------------------------------------------- lưu trữ
class SelectionStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.frames_path = self.root / "frames.json"
        self.captions_path = self.root / "captions.json"
        self.descriptions_path = self.root / "descriptions.json"
        self.proposals_path = self.root / "proposals.json"
        self.grouped_path = self.root / "grouped.json"

    def read(self, path: Path, default: Any) -> Any:
        return read_json(path) if path.exists() else default


def proposed_path(labels_file: Path) -> Path:
    return labels_file.with_name(labels_file.stem + ".proposed.yaml")


def _selection_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["adl"]["label_selection"]


# --------------------------------------------------------------------------- caption
def run_caption(cfg: dict[str, Any], store: SelectionStore, backend: VLMBackend, image_paths: Sequence[Path], resample: bool = False) -> dict[str, Any]:
    """Lấy mẫu ảnh, viết caption từng ảnh, nối thành mô tả từng nhóm thời gian."""
    sel = _selection_cfg(cfg)
    by_id = {image_id_of(p): p for p in image_paths}
    frames: dict[str, list[str]] = {} if resample else store.read(store.frames_path, {})
    if not frames:
        frames = sample_frames(list(by_id), int(sel["num_groups"]), int(sel["images_per_group"]), int(sel.get("seed", 0)))
        write_json(store.frames_path, frames)
    captions: dict[str, str] = store.read(store.captions_path, {})
    prompt = CAPTION_PROMPT.format(viewpoint=cfg["adl"]["viewpoint"])
    usage, log = Usage(), get_logger()
    wanted = [i for ids in frames.values() for i in ids]
    for n, image_id in enumerate(wanted, start=1):
        if captions.get(image_id):
            continue
        if image_id not in by_id:
            raise FileNotFoundError(f"Ảnh {image_id} trong {store.frames_path} không còn trong thư mục ảnh. Chạy lại với --resample.")
        with Image.open(by_id[image_id]) as im:
            image = resize_long_side(im.convert("RGB"), int(sel["caption_max_long_side"]))
        result = backend.generate([image], prompt, max_new_tokens=int(sel["caption_max_new_tokens"]))
        usage.add(result)
        captions[image_id] = " ".join(result.text.split())
        write_json(store.captions_path, captions)     # ghi sau từng ảnh để chạy lại không mất tiền đã trả
        if n % 20 == 0:
            log.info("caption %d/%d", n, len(wanted))
    descriptions = {key: " ".join(captions[i] for i in ids if captions.get(i)) for key, ids in frames.items()}
    write_json(store.descriptions_path, descriptions)
    return {"groups": len(frames), "frames": len(wanted), "new_captions": usage.calls, "usage": usage.as_dict(), "descriptions_file": str(store.descriptions_path)}


# --------------------------------------------------------------------------- đề xuất nhãn
def load_raw_labels(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        labels = _strings((yaml.safe_load(f) or {}).get("labels"))
    if not labels:
        raise ValueError(f"{path}: không có nhãn thô nào")
    return list(dict.fromkeys(labels))


def merge_proposals(proposals: dict[str, dict[str, list[str]]], raw_labels: Sequence[str]) -> dict[str, int]:
    """Bỏ trùng (không phân biệt hoa thường) và đếm số nhóm thời gian mỗi nhãn xuất hiện.

    Nhãn ``present`` không nằm trong danh sách thô (model tự sửa chữ) được coi như nhãn mới.
    """
    canonical = {label.lower(): label for label in raw_labels}
    counts: dict[str, int] = {}
    for proposal in proposals.values():
        seen: set[str] = set()
        for label in [*proposal.get("present", []), *proposal.get("new", [])]:
            name = canonical.setdefault(label.lower(), label.lower())
            if name not in seen:
                seen.add(name)
                counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def build_proposed_classes(counts: dict[str, int], grouped: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ghép kết quả xếp nhóm của LLM với danh sách nhãn. Nhãn LLM bỏ sót vào nhóm ``Uncategorized``;
    nhãn LLM tự thêm bị bỏ."""
    info = {str(item.get("name", "")).strip().lower(): item for item in grouped if isinstance(item, dict)}
    classes = []
    for name, seen_in in counts.items():
        item = info.get(name.lower(), {})
        classes.append({
            "name": name,
            "name_vi": str(item.get("name_vi") or "").strip(),
            "group": str(item.get("group") or "").strip() or UNGROUPED,
            "seen_in": seen_in,
            "descriptions": [],
        })
    classes.sort(key=lambda c: (c["group"] == UNGROUPED, c["group"], -c["seen_in"], c["name"]))
    return classes


PROPOSED_HEADER = """# Bộ nhãn ADL do model đề xuất (python main.py adl propose-labels). CẦN NGƯỜI DUYỆT.
#
# Theo LSC-ADL mục 3.2, bước cuối là làm tay:
#   1. Sửa nhóm nào model xếp sai.
#   2. Gộp các nhãn quá chi tiết thành lớp thô hơn, bỏ nhãn hiếm hoặc trùng nghĩa.
#   3. Giữ các lớp tiêu biểu, đủ phủ các hoạt động trong dữ liệu.
# `seen_in` = số nhóm thời gian có nhãn này (trên {groups} nhóm); chỉ để tham khảo.
# Xong thì chép phần `classes` sang file nhãn chính ({labels_file}) rồi chạy:
#   python main.py adl describe      # sinh câu mô tả cho các lớp còn thiếu
#   python main.py adl init --force
"""


def write_labels_yaml(path: Path, classes: Sequence[dict[str, Any]], header: str) -> None:
    body = yaml.safe_dump({"classes": list(classes)}, allow_unicode=True, sort_keys=False, width=120)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header.rstrip("\n") + "\n\n" + body, encoding="utf-8")


def run_propose(cfg: dict[str, Any], store: SelectionStore, backend: VLMBackend, labels_file: Path, raw_labels_file: Path, force: bool = False) -> dict[str, Any]:
    """Đề xuất nhãn cho từng nhóm thời gian, bỏ trùng, xếp nhóm ngữ nghĩa, ghi file chờ duyệt."""
    sel = _selection_cfg(cfg)
    descriptions: dict[str, str] = store.read(store.descriptions_path, {})
    if not descriptions:
        raise RuntimeError(f"Chưa có {store.descriptions_path}. Chạy `adl caption` trước.")
    raw_labels = load_raw_labels(raw_labels_file)
    viewpoint, retries, usage = cfg["adl"]["viewpoint"], int(sel.get("retries", 1)), Usage()

    proposals: dict[str, dict[str, list[str]]] = {} if force else store.read(store.proposals_path, {})
    for key, description in descriptions.items():
        if key in proposals or not description.strip():
            continue
        prompt = PROPOSE_PROMPT.format(viewpoint=viewpoint, description=description, candidates="\n".join(f"- {label}" for label in raw_labels))
        data = ask_json(backend, prompt, usage, retries)
        proposals[key] = {"present": _strings(data.get("present")), "new": _strings(data.get("new"))}
        write_json(store.proposals_path, proposals)

    counts = merge_proposals(proposals, raw_labels)
    if not counts:
        raise RuntimeError("Model không tìm thấy hoạt động nào trong các đoạn mô tả.")
    prompt = GROUP_PROMPT.format(
        viewpoint=viewpoint,
        labels="\n".join(f"- {name} ({seen})" for name, seen in counts.items()),
        groups=", ".join(sel["groups"]),
    )
    grouped = ask_json(backend, prompt, usage, retries, max_new_tokens=4000).get("labels") or []
    write_json(store.grouped_path, grouped)
    classes = build_proposed_classes(counts, grouped)
    out_path = proposed_path(labels_file)
    write_labels_yaml(out_path, classes, PROPOSED_HEADER.format(groups=len(descriptions), labels_file=labels_file.name))
    return {
        "labels": len(classes),
        "from_raw": sum(1 for c in classes if c["name"].lower() in {r.lower() for r in raw_labels}),
        "ungrouped": sum(1 for c in classes if c["group"] == UNGROUPED),
        "usage": usage.as_dict(),
        "proposed_file": str(out_path),
    }


# --------------------------------------------------------------------------- Decompose-Expand
def decompose_expand(backend: VLMBackend, name: str, viewpoint: str, count: int, usage: Usage, retries: int = 1) -> dict[str, list[str]]:
    """Hai lượt hỏi cho một lớp: từ khoá vật thể + hành động, rồi ``count`` câu mô tả từ tên + từ khoá."""
    keywords = ask_json(backend, KEYWORDS_PROMPT.format(name=name, viewpoint=viewpoint), usage, retries)
    objects, actions = _strings(keywords.get("objects")), _strings(keywords.get("actions"))
    prompt = DESCRIBE_PROMPT.format(name=name, viewpoint=viewpoint, objects=", ".join(objects), actions=", ".join(actions), count=count)
    descriptions = list(dict.fromkeys(_strings(ask_json(backend, prompt, usage, retries).get("descriptions"))))[:count]
    if not descriptions:
        raise ValueError(f"Model không sinh được câu mô tả nào cho lớp {name!r}")
    return {"objects": objects, "actions": actions, "descriptions": descriptions}


def _leading_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            break
        lines.append(line)
    return "\n".join(lines).strip("\n")


def run_describe(cfg: dict[str, Any], backend: VLMBackend, labels_file: Path, force: bool = False) -> dict[str, Any]:
    """Sinh câu mô tả cho các lớp còn thiếu trong file nhãn (``force``: sinh lại mọi lớp).

    Ghi file sau từng lớp. Phần chú thích đầu file được giữ; chú thích nằm giữa các lớp thì mất.
    """
    sel = _selection_cfg(cfg)
    text = labels_file.read_text(encoding="utf-8")
    classes = (yaml.safe_load(text) or {}).get("classes") or []
    if not classes:
        raise ValueError(f"{labels_file}: không có lớp nào")
    header = _leading_comments(text)
    count, viewpoint, retries, usage = int(sel["descriptions_per_class"]), cfg["adl"]["viewpoint"], int(sel.get("retries", 1)), Usage()
    written: list[str] = []
    for item in classes:
        name = str(item.get("name", "")).strip()
        if not name or (len(item.get("descriptions") or []) >= count and not force):
            continue
        result = decompose_expand(backend, name, viewpoint, count, usage, retries)
        item["keywords"] = {"objects": result["objects"], "actions": result["actions"]}
        item["descriptions"] = result["descriptions"]
        write_labels_yaml(labels_file, classes, header)
        written.append(name)
    return {"classes": len(classes), "described": written, "usage": usage.as_dict(), "labels_file": str(labels_file)}

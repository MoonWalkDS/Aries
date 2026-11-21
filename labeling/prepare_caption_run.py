"""Prepare an offline, reproducible caption run without changing existing inputs."""
from pathlib import Path
import csv
import hashlib
import json
import random
from collections import Counter
from caption_export import EXPORTS, TEACHER, HINTS, image_record


def main():
    root = EXPORTS.parent
    out = EXPORTS / 'caption' / 'preparation'
    out.mkdir(parents=True, exist_ok=True)
    source = EXPORTS / 'castle_dataset.jsonl'
    rows = [json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if s.strip()]
    teacher = {}
    if TEACHER.exists():
        with TEACHER.open(encoding='utf-8') as f:
            teacher = {r['imageID'].replace('day1_Kitchen_', ''): r.get('description') or '' for r in csv.DictReader(f)}
    hints = json.loads(HINTS.read_text(encoding='utf-8')) if HINTS.exists() else {}
    inputs = [image_record(r, teacher, hints) for r in rows]
    ids = {r['image_id'] for r in inputs}
    assert len(inputs) == len(ids) == 5000, 'Expected exactly 5000 unique image IDs'
    images = {p.stem: p for p in (root / 'dataset' / 'images').rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    missing = sorted(ids - images.keys())
    assert not missing, f'Missing images: {missing[:10]}'
    def write(name, records):
        (out / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    write('inputs.jsonl', inputs)
    issues = []
    for r in inputs:
        flags = []
        if not r['people']:
            flags.append('no_detected_people_review_teacher')
        for p in r['people']:
            if any(w in p.get('mo_ta', '').lower() for w in ['có lúc', 'hoặc']):
                flags.append('outfit_variants:' + p['id'])
            if p.get('do_chac') == 'chưa chắc':
                flags.append('uncertain_action:' + p['id'])
            if p.get('hanh_dong') == 'nấu ăn trên bếp' and p.get('vung') and 'bếp nấu' not in p['vung']:
                flags.append('check_action_location:' + p['id'])
        if flags:
            issues.append({'image_id': r['image_id'], 'flags': flags})
    shuffled = list(inputs)
    random.Random(314).shuffle(shuffled)
    pilot, strata = [], set()
    for r in shuffled:
        key = (r.get('gio'), min(r['so_nguoi'], 4))
        if key not in strata:
            strata.add(key)
            pilot.append(r)
    chosen = {r['image_id'] for r in pilot}
    pilot.extend(r for r in shuffled if r['image_id'] not in chosen)
    write('pilot_100.jsonl', pilot[:100])
    write('review_flags.jsonl', issues)
    write('manifest.jsonl', [{'image_id': r['image_id'], 'image_path': images[r['image_id']].relative_to(root).as_posix(), 'hour': r.get('gio'), 'people_count': r['so_nguoi']} for r in inputs])
    (out / 'batches').mkdir(exist_ok=True)
    for i in range(0, len(inputs), 25):
        write(f'batches/{i // 25 + 1:03}.jsonl', inputs[i:i+25])
    existing = EXPORTS / 'caption' / 'inputs.jsonl'
    old = [json.loads(s) for s in existing.read_text(encoding='utf-8').splitlines() if s.strip()] if existing.exists() else []
    caps = EXPORTS / 'caption' / 'captions.jsonl'
    saved = [json.loads(s) for s in caps.read_text(encoding='utf-8').splitlines() if s.strip()] if caps.exists() else []
    valid = [r for r in saved if r.get('image_id') in ids and isinstance(r.get('caption'), str) and r['caption'].strip()]
    report = {'images': len(inputs), 'missing_images': missing, 'teacher_descriptions': sum(bool(r['thay_mo_ta']) for r in inputs), 'hour_counts': dict(sorted(Counter(r.get('gio') for r in inputs).items())), 'people_count_distribution': dict(sorted(Counter(r['so_nguoi'] for r in inputs).items())), 'flagged_images': len(issues), 'flag_counts': dict(Counter(f.split(':')[0] for r in issues for f in r['flags'])), 'existing_inputs_match_current_export': old == inputs, 'existing_caption_rows': len(saved), 'valid_caption_ids': len({r['image_id'] for r in valid}), 'duplicate_caption_rows': len(saved) - len({r.get('image_id') for r in saved}), 'pilot_size': 100, 'batches_of_25': 200, 'sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, TEACHER, HINTS, EXPORTS / 'caption' / 'prompt_vi.md'] if p.exists()}}
    (out / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

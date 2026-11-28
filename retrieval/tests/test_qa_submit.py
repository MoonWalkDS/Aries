import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submission"))
import rank_report  # noqa: E402
import search_log  # noqa: E402
import submit_task  # noqa: E402


def test_text_answer_sends_only_text(monkeypatch):
    sent = {}

    class Resp:
        status_code, text = 200, '{"submission": "CORRECT"}'

    def post(url, params, json, timeout):
        sent.update(url=url, params=params, json=json)
        return Resp()

    monkeypatch.setattr(submit_task.requests, "post", post)
    assert submit_task.submit_text("s", "e1", "ba người") == (200, '{"submission": "CORRECT"}')
    assert sent["json"] == {"answerSets": [{"answers": [{"text": "ba người"}]}]}   # có mediaItemName thì DRES coi là nộp ảnh
    assert sent["url"].endswith("/submit/e1") and sent["params"] == {"session": "s"}


def test_text_submit_is_logged_and_skipped_by_rank_report(tmp_path, monkeypatch):
    log = tmp_path / "searches.jsonl"
    monkeypatch.setattr(search_log, "LOG_FILE", log)
    task = tmp_path / "task.json"
    task.write_text(json.dumps({"taskId": "t1"}), encoding="utf-8")
    monkeypatch.setattr(search_log, "TASK_FILE", task)
    search_log.log_submit(None, {"item": "ba người", "status": 200, "message": "ĐÚNG"}, text="ba người")
    search_log.log_submit("08_5", {"item": "day1_Kitchen_08_5", "status": 200, "message": "SAI"})
    rows = rank_report.read_log(log)
    assert (rows[0]["text"], rows[0]["image_id"]) == ("ba người", None)
    _, tasks = rank_report.analyse(rows, {"t1": {"08_5"}}, rank_report.frames_by_hour(["08_5"]), 3)
    assert (tasks[0]["submits"], tasks[0]["submit_exact"]) == (1, True)   # chỉ lần nộp ảnh được dò

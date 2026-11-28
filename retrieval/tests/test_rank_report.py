import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rank_report  # noqa: E402
import search_log  # noqa: E402


def test_log_then_report_ranks_per_search_task_and_recall(tmp_path, monkeypatch):
    task_file = tmp_path / "current_task.json"
    monkeypatch.setattr(search_log, "TASK_FILE", task_file)
    log = tmp_path / "searches.jsonl"
    monkeypatch.setattr(search_log, "LOG_FILE", log)
    monkeypatch.setattr(search_log, "SYSTEM", "v1")

    task_file.write_text(json.dumps({"evaluationId": "e", "taskId": "t1", "taskTemplateId": "tpl1", "query": "..."}), encoding="utf-8")
    search_log.log_search("người mở tủ lạnh", 300, None, {"results": [{"id": "08_1"}, {"id": "08_7"}, {"id": "08_5"}], "parsed": {"count": 1}}, 12.3)
    search_log.log_search("người mở tủ lạnh áo len", 300, {"count": 1}, {"results": [{"id": "08_5"}, {"id": "08_9"}]}, 10)
    search_log.log_submit("08_6", {"item": "day1_Kitchen_08_6", "status": 200, "message": "ok"})
    task_file.write_text(json.dumps({"taskId": "t2", "taskTemplateId": "tpl2"}), encoding="utf-8")
    search_log.log_search("bếp trống", 300, None, {"results": [{"id": "09_1"}]}, 5)
    task_file.unlink()
    search_log.log_search("chưa kéo đề", 300, None, {"results": []}, 5)              # không có task: báo cáo bỏ qua

    rows = rank_report.read_log(log)
    assert [r["kind"] for r in rows] == ["search", "search", "submit", "search", "search"]
    assert rows[0]["task"]["taskId"] == "t1" and rows[0]["system"] == "v1" and rows[1]["edited"] and rows[-1]["task"] is None

    truth_csv = tmp_path / "truth.csv"
    truth_csv.write_text("task,image_ids\nt1,day1_Kitchen_08_9.jpg\ntpl2,10_4;10_5\n", encoding="utf-8")
    truth = rank_report.load_truth(truth_csv)
    assert truth == {"t1": {"08_9"}, "tpl2": {"10_4", "10_5"}}
    by_hour = rank_report.frames_by_hour([f"08_{n}" for n in range(1, 12)] + ["09_1", "10_4", "10_5"])
    assert rank_report.widen({"08_9"}, by_hour, 3) == {f"08_{n}" for n in range(6, 12)}

    searches, tasks = rank_report.analyse(rows, truth, by_hour, 3)
    assert [(s["rank_exact"], s["rank_window3"]) for s in searches] == [(None, 2), (2, 2), (None, None)]
    t1 = next(t for t in tasks if t["task"] == "t1")
    assert (t1["searches"], t1["first_window3"], t1["best_exact"], t1["submits"], t1["submit_exact"], t1["submit_window3"]) == (2, 2, 2, 1, False, True)
    table = rank_report.recall_table(tasks, 3)
    assert "| v1 | đầu tiên | ±3 khung | 2 | 0/2 | 1/2 |" in table

    (tmp_path / "embeddings").mkdir()
    (tmp_path / "embeddings" / "image_ids.json").write_text(json.dumps([f"{i}.jpg" for h in by_hour.values() for i in h]), encoding="utf-8")
    monkeypatch.setattr(rank_report, "EXPORTS_DIR", tmp_path)
    out = tmp_path / "report"
    assert rank_report.main(["--truth", str(truth_csv), "--log", str(log), "--out", str(out)]) == 0
    assert (out / "searches.csv").read_text(encoding="utf-8-sig").splitlines()[1].endswith("> top,2")
    assert "Chưa tìm lần nào: không có" in (out / "summary.md").read_text(encoding="utf-8")


def test_truth_json_and_odd_ids():
    assert rank_report.norm_id("day1_Kitchen_08_333.jpg") == "08_333" and rank_report.norm_id("8_0333") == "08_333"

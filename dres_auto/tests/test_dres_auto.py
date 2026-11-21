"""dres_auto chạy với tools/mock_dres.py (không mạng, không DRES thật)."""
import base64
import importlib
import json

import pytest
from fastapi.testclient import TestClient

from tools.mock_dres import EVALUATION_ID, SESSION_ID, app as mock_dres_app


@pytest.fixture
def submit_task(monkeypatch, tmp_path):
    # Đặt env trước khi import: script đọc cấu hình ở mức module, load_dotenv không ghi đè biến đã có.
    for key, value in {
        "DRES_URL": "http://mock-dres/api/v2",
        "DRES_USERNAME": "student",
        "DRES_PASSWORD": "student",
        "DRES_EVALUATION_ID": EVALUATION_ID,
        "DRES_ITEM_PREFIX": "day1_Kitchen_",
        "DRES_COLLECTION": "castle5000",
        "DRES_SESSION_ID": "",
    }.items():
        monkeypatch.setenv(key, value)
    module = importlib.reload(importlib.import_module("dres_auto.submit_task"))
    client = TestClient(mock_dres_app, base_url="http://mock-dres")
    monkeypatch.setattr(module.requests, "post", client.post)
    monkeypatch.setattr(module.requests, "get", client.get)
    monkeypatch.setattr(module, "TASK_JSON", tmp_path / "current_task.json")
    monkeypatch.setattr(module, "LOG_FILE", tmp_path / "submissions.log")
    return module


@pytest.mark.parametrize("raw", ["10_1528.jpg", "10_1528", "data/images/10_1528.jpg",
                                 "data\\images\\10_1528.JPG", "day1_Kitchen_10_1528"])
def test_to_item_name(submit_task, raw):
    assert submit_task.to_item_name(raw) == "day1_Kitchen_10_1528"


def test_login_returns_session(submit_task):
    assert submit_task.login() == SESSION_ID


def test_submit_body_and_verdicts(submit_task):
    status, text = submit_task.submit(SESSION_ID, EVALUATION_ID, "day1_Kitchen_10_1528")
    assert status == 200 and submit_task.describe(status, text).startswith("ĐÚNG")
    status, text = submit_task.submit(SESSION_ID, EVALUATION_ID, "wrong-id")
    assert submit_task.describe(status, text).startswith("SAI")
    status, text = submit_task.submit(SESSION_ID, EVALUATION_ID, "trigger-412")
    assert status == 412 and submit_task.describe(status, text) == submit_task.MEANING[412]


def test_expired_session_relogins_once_and_logs(submit_task):
    state = {"session": "expired", "evaluation_id": EVALUATION_ID}
    text = submit_task.do_submit(state, "day1_Kitchen_10_1528", yes=True)
    assert state["session"] == SESSION_ID
    assert json.loads(text)["submission"] == "CORRECT"
    entry = json.loads(submit_task.LOG_FILE.read_text(encoding="utf-8").splitlines()[-1])
    assert entry["item"] == "day1_Kitchen_10_1528" and entry["status"] == 200


def test_check_mode_does_not_log(submit_task):
    state = {"session": SESSION_ID, "evaluation_id": EVALUATION_ID}
    submit_task.do_submit(state, "day1_Kitchen_CHECK_CONNECTION", yes=True, check=True)
    assert not submit_task.LOG_FILE.exists()


def test_hint_to_text_dedups_text_and_saves_media(monkeypatch, tmp_path):
    fetch_task = importlib.import_module("dres_auto.fetch_task")
    monkeypatch.setattr(fetch_task, "RUNTIME_DIR", tmp_path)
    image = "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode()
    hint = {"sequence": [
        {"contentType": "TEXT", "content": "Một người đứng cạnh bếp."},
        {"contentType": "TEXT", "content": "Một người đứng cạnh bếp."},
        {"contentType": "IMAGE", "content": image},
    ]}
    lines = fetch_task.hint_to_text(hint).splitlines()
    assert lines[0] == "Một người đứng cạnh bếp." and len(lines) == 2
    assert (tmp_path / "hint_2.jpg").read_bytes() == b"jpeg-bytes"

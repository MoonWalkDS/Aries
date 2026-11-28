"""Local DRES web interface for the v2 retrieval foundation."""
import argparse
import json
import re
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import requests

import fetch_task
import submit_task


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, content, mime, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path != "/":
            self.send_error(404)
            return
        self.reply(200, Path(__file__).with_name("index.html").read_bytes(), "text/html; charset=utf-8")

    def do_POST(self):
        origin = f"http://127.0.0.1:{self.server.server_port}"
        if self.headers.get("Origin") != origin or self.headers.get("Host") != origin[7:]:
            self.send_error(403)
            return
        if self.path not in ("/task", "/submit"):
            self.send_error(404)
            return
        result_status, headers = 200, {}
        try:
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 4096:
                raise ValueError()
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError()
            if self.path == "/submit":
                item, answer = body.get("item"), body.get("text")
                if answer is not None:   # task QA: câu trả lời chữ
                    if not isinstance(answer, str) or not 0 < len(" ".join(answer.split())) <= 500:
                        raise ValueError()
                    answer = " ".join(answer.split())
                elif not isinstance(item, str) or not re.fullmatch(r"[\w.-]{1,150}", item):
                    raise ValueError()
            session = submit_task.SESSION_ID or submit_task.login()
            evaluation = submit_task.EVALUATION_ID or submit_task.active_evaluation_id(session)
            if self.path == "/task":
                state = fetch_task.get_state(session, evaluation)
                if state.get("taskStatus") != "RUNNING":
                    result_status = 409
                    message = "Chưa có đề đang chạy."
                else:
                    hint = fetch_task.get_hint(session, evaluation, state["taskTemplateId"])
                    message = fetch_task.hint_to_text(hint)
                    try:   # loại task để trang tự chọn nộp ảnh hay nộp QA
                        kind = fetch_task.current_task_type(session, evaluation)
                        headers["X-Task-Type"] = " ".join(v for v in (kind["taskType"], kind["taskGroup"]) if v).encode("ascii", "ignore").decode()
                    except (requests.RequestException, ValueError):
                        pass
            elif answer is not None:
                status, text = submit_task.submit_text(session, evaluation, answer)
                submit_task.log(f"text:{answer}", evaluation, status, text)
                message = f"HTTP {status} · {submit_task.describe(status, text)}"
            else:
                name = submit_task.to_item_name(item)
                status, text = submit_task.submit(session, evaluation, name)
                submit_task.log(name, evaluation, status, text)
                message = f"HTTP {status} · {submit_task.describe(status, text)}"
        except (requests.RequestException, KeyError):
            result_status = 502
            message = "Không lấy được phản hồi hợp lệ từ DRES. Kiểm tra kết nối, cấu hình và lịch sử nộp."
        except ValueError:
            self.send_error(400)
            return
        except RuntimeError as error:
            result_status = 503
            message = str(error)
        self.reply(result_status, message.encode("utf-8"), "text/plain; charset=utf-8", headers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    with HTTPServer(("127.0.0.1", 8314), Handler) as server:
        print("http://127.0.0.1:8314", flush=True)
        if not args.no_browser:
            webbrowser.open("http://127.0.0.1:8314")
        server.serve_forever()

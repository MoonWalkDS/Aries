import json
from caption_gpt import ask


def test_config_loads_project_env_without_overwriting_shell(tmp_path, monkeypatch):
    from caption_gpt import load_api_config
    env = tmp_path / '.env'
    env.write_text('OPENAI_API_KEY=file-key\nOPENAI_BASE_URL=https://api.miraiapi.com/v1/\n', encoding='utf-8')
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    assert load_api_config(env) == ('file-key', 'https://api.miraiapi.com/v1/chat/completions')
    monkeypatch.setenv('OPENAI_API_KEY', 'shell-key')
    assert load_api_config(env)[0] == 'shell-key'


def test_request_uses_configured_endpoint(monkeypatch):
    seen = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({'choices': [{'message': {'content': '{"caption":"test caption"}'}}]}).encode()
    def urlopen(req, timeout):
        assert req.get_header("User-agent") == "DS314-API-Test/1.0"
        seen.append(req.full_url)
        return Response()
    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    assert ask({'image_id': 'test'}, 'prompt', 'model', 'dummy-key', False,
               api='https://api.miraiapi.com/v1/chat/completions') == 'test caption'
    assert seen == ['https://api.miraiapi.com/v1/chat/completions']

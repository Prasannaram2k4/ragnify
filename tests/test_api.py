from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_health():
    for path in ('/health', '/api/health'):
        r = client.get(path)
        assert r.status_code == 200
        assert r.json()['status'] == 'ok'


def test_query_over_samples_without_keys(monkeypatch):
    for k in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'HF_API_TOKEN'):
        monkeypatch.delenv(k, raising=False)
    r = client.post('/api/query', json={'question': 'Which LLM providers does Ragnify support?'})
    assert r.status_code == 200
    data = r.json()
    assert data['provider'] == 'extractive'
    assert data['retrieved'] and 'OpenAI' in ' '.join(x['text'] for x in data['retrieved'])


def test_upload_then_query(monkeypatch):
    monkeypatch.delenv('HF_API_TOKEN', raising=False)
    content = b'The zorblax engine runs on purple lava and was invented in 1887 by Dr. Quimby. ' * 5
    r = client.post('/api/upload', files=[('files', ('notes.txt', content, 'text/plain'))])
    assert r.status_code == 200
    chunks = r.json()['chunks']
    assert chunks
    q = client.post('/api/query', json={'question': 'What does the zorblax engine run on?',
                                         'chunks': chunks, 'use_samples': False})
    assert q.status_code == 200
    assert q.json()['retrieved'][0]['source'] == 'notes.txt'


def test_empty_question():
    assert client.post('/query', json={'question': '  '}).status_code == 400


def test_reject_bad_extension():
    r = client.post('/upload', files=[('files', ('x.exe', b'x', 'application/octet-stream'))])
    assert r.status_code == 400

import os
import importlib


def get_client():
    # Disable API key for tests
    os.environ['API_KEY'] = ''
    # Point DATA_DIR to an empty temp dir to avoid IO
    os.environ['DATA_DIR'] = os.getcwd() + '/tests/_tmp_empty'
    os.makedirs(os.environ['DATA_DIR'], exist_ok=True)
    # Ensure module reload picks up env
    if 'backend.app' in globals():
        import backend.app as app_mod
        importlib.reload(app_mod)
        app = app_mod.app
    else:
        import backend.app as app_mod
        app = app_mod.app
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_health_ok():
    client = get_client()
    r = client.get('/health')
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get('status') == 'ok'

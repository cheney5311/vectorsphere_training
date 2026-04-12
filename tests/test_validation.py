import os
import json
import pytest


class DummyDBManager:
    def create_tables(self):
        return True
    def graceful_shutdown(self, timeout=0):
        return True


@pytest.fixture
def client(monkeypatch):
    # ensure default non-strict mode unless tests override
    monkeypatch.setenv('CHECK_API_STRICT_MODE', os.getenv('CHECK_API_STRICT_MODE', '0'))
    # avoid real DB
    monkeypatch.setenv('FORCE_PG_IMPORT', '0')
    try:
        import backend.modules.database.manager as db_manager_mod
        monkeypatch.setattr(db_manager_mod, 'get_database_manager', lambda: DummyDBManager())
    except Exception:
        pass
    try:
        import backend.modules.database.config as db_config_mod
        monkeypatch.setattr(db_config_mod, 'get_database_config', lambda: type('C', (), {'host': '127.0.0.1', 'port': 5432, 'username': 'x', 'password': 'x'}))
    except Exception:
        pass
    # avoid heavy model download env setup
    try:
        import backend.utils.model_download_config as mdl
        monkeypatch.setattr(mdl, 'setup_model_download_environment', lambda: None)
    except Exception:
        pass
    # avoid tracing heavy init
    try:
        import backend.core.tracing as tracing
        monkeypatch.setattr(tracing, 'init_tracing', lambda app=None: None)
    except Exception:
        pass

    from app import create_app
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_non_strict_allows_invalid_payload(client, monkeypatch):
    # non-strict mode should allow request to proceed even if schema validation fails
    monkeypatch.setenv('CHECK_API_STRICT_MODE', '0')
    payload = {
        # missing required `job_name`, but has dataset_id
        "dataset_id": "ds_123"
    }
    resp = client.post('/api/v1/demo/create_training', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['status'] == 'ok'
    # received should be the original payload
    assert data['received'] == payload


def test_strict_mode_rejects_invalid_payload(client, monkeypatch):
    monkeypatch.setenv('CHECK_API_STRICT_MODE', '1')
    payload = {
        "dataset_id": "ds_123"
    }
    resp = client.post('/api/v1/demo/create_training', json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data['error'] == 'schema_validation_failed'
    assert 'message' in data


def test_invalid_json_body_returns_400(client):
    # send non-json body
    resp = client.post('/api/v1/demo/create_training', data='not-a-json', content_type='text/plain')
    assert resp.status_code == 400
    data = resp.get_json()
    assert data['error'] == 'invalid_json'

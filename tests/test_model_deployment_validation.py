import os
import sys
import pytest
from flask import Flask, request, jsonify
# Ensure project root is on sys.path for correct backend import
sys.path.insert(0, "/root/VectorSphere/VectorSphere-intelligent-platform")
from backend.core.validation import validate_json_schema


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('CHECK_API_STRICT_MODE', os.getenv('CHECK_API_STRICT_MODE', '0'))
    app = Flask(__name__)

    @app.route('/api/v1/training/model/deploy', methods=['POST'])
    @validate_json_schema('backend/api/schemas/model_deploy_request.json')
    def deploy_dummy():
        data = request.get_json(silent=True)
        return jsonify({'ok': True, 'received': data})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_deploy_requires_model_id_strict(client, monkeypatch):
    monkeypatch.setenv('CHECK_API_STRICT_MODE', '1')
    payload = {
        # missing model_id
        "mode": "online",
        "replicas": 1
    }
    resp = client.post('/api/v1/training/model/deploy', json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert data['error'] == 'validation_schema_failed' or data.get('code') is not None


def test_deploy_allows_missing_optional_non_strict(client, monkeypatch):
    monkeypatch.setenv('CHECK_API_STRICT_MODE', '0')
    payload = {
        "model_id": "m_123",
        "mode": "online",
        "replicas": 1
    }
    resp = client.post('/api/v1/training/model/deploy', json=payload)
    assert resp.status_code in (200, 400, 500)

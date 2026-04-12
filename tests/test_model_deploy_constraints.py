import os
import sys
import pytest
from flask import Flask, request, jsonify
sys.path.insert(0, "/root/VectorSphere/VectorSphere-intelligent-platform")
from backend.core.validation import validate_json_schema


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('CHECK_API_STRICT_MODE', '1')
    app = Flask(__name__)

    @app.route('/deploy', methods=['POST'])
    @validate_json_schema('backend/api/schemas/model_deploy_request.json')
    def deploy():
        return jsonify({'ok': True})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_invalid_mode_enum(client):
    resp = client.post('/deploy', json={'model_id': 'm1', 'mode': 'invalid', 'replicas': 1})
    assert resp.status_code == 400


def test_invalid_release_strategy_enum(client):
    resp = client.post('/deploy', json={'model_id': 'm1', 'release_strategy': 'invalid', 'replicas': 1})
    assert resp.status_code == 400


def test_replicas_zero_invalid(client):
    resp = client.post('/deploy', json={'model_id': 'm1', 'replicas': 0})
    assert resp.status_code == 400

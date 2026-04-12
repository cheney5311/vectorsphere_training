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

    @app.route('/api/v1/training/three-stage/start', methods=['POST'])
    @validate_json_schema('backend/api/schemas/three_stage_start.json')
    def start_dummy():
        return jsonify({'ok': True})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_three_stage_start_requires_model_name(client):
    resp = client.post('/api/v1/training/three-stage/start', json={})
    assert resp.status_code == 400

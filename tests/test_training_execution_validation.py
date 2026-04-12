import os
import sys
import pytest
from flask import Flask, request, jsonify
sys.path.insert(0, "/root/VectorSphere/VectorSphere-intelligent-platform")
from backend.core.validation import validate_json_schema


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('CHECK_API_STRICT_MODE', os.getenv('CHECK_API_STRICT_MODE', '0'))
    app = Flask(__name__)

    @app.route('/api/v1/training/execution/start', methods=['POST'])
    @validate_json_schema('backend/api/schemas/training_execution_start.json')
    def start_dummy():
        return jsonify({'ok': True})

    @app.route('/api/v1/training/execution/sessions/s1/metrics', methods=['POST'])
    @validate_json_schema('backend/api/schemas/training_metrics_update.json')
    def metrics_dummy():
        return jsonify({'ok': True})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_start_requires_session_id_strict(client, monkeypatch):
    monkeypatch.setenv('CHECK_API_STRICT_MODE', '1')
    resp = client.post('/api/v1/training/execution/start', json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data['error'] == 'validation_schema_failed' or data.get('code') is not None


def test_metrics_requires_min_fields_strict(client, monkeypatch):
    monkeypatch.setenv('CHECK_API_STRICT_MODE', '1')
    # missing required fields epoch/step/loss
    resp = client.post('/api/v1/training/execution/sessions/s1/metrics', json={})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data['error'] == 'validation_schema_failed' or data.get('code') is not None

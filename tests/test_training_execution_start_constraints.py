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

    @app.route('/exec/start', methods=['POST'])
    @validate_json_schema('backend/api/schemas/training_execution_start.json')
    def start():
        return jsonify({'ok': True})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_invalid_scenario_type(client):
    resp = client.post('/exec/start', json={'session_id': 's1', 'scenario_type': 'invalid'})
    assert resp.status_code == 400


def test_valid_scenario_type(client):
    resp = client.post('/exec/start', json={'session_id': 's1', 'scenario_type': 'classification'})
    assert resp.status_code == 200

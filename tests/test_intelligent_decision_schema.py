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

    @app.route('/decision', methods=['POST'])
    @validate_json_schema('backend/api/schemas/intelligent_decision_request.json')
    def decision():
        return jsonify({'ok': True})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_missing_required_invalid(client):
    resp = client.post('/decision', json={'inputs': {}})
    assert resp.status_code == 400


def test_invalid_scenario_enum(client):
    resp = client.post('/decision', json={'scenario': 'bad', 'inputs': {}})
    assert resp.status_code == 400


def test_valid_request(client):
    resp = client.post('/decision', json={'scenario': 'classification', 'inputs': {}})
    assert resp.status_code == 200

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

    @app.route('/pipeline/create', methods=['POST'])
    @validate_json_schema('backend/api/schemas/pipeline_create.json')
    def create():
        return jsonify({'ok': True})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_step_missing_type_invalid(client):
    payload = {'name': 'p', 'steps': [{'name': 's1'}]}
    resp = client.post('/pipeline/create', json=payload)
    assert resp.status_code == 400


def test_step_with_type_valid(client):
    payload = {'name': 'p', 'steps': [{'name': 's1', 'type': 'task'}]}
    resp = client.post('/pipeline/create', json=payload)
    assert resp.status_code == 200

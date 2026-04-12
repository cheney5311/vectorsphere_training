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


def test_steps_requires_name(client):
    resp = client.post('/pipeline/create', json={'name': 'p1', 'steps': [{}]})
    assert resp.status_code == 400


def test_steps_type_enum(client):
    resp = client.post('/pipeline/create', json={'name': 'p1', 'steps': [{'name': 's1', 'type': 'invalid'}]})
    assert resp.status_code == 400


def test_steps_valid_item(client):
    resp = client.post('/pipeline/create', json={'name': 'p1', 'steps': [{'name': 's1', 'type': 'task', 'params': {'x': 1}}]})
    assert resp.status_code == 200

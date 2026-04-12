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

    @app.route('/three/start', methods=['POST'])
    @validate_json_schema('backend/api/schemas/three_stage_start.json')
    def start():
        return jsonify({'ok': True})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_model_name_with_space_invalid(client):
    resp = client.post('/three/start', json={'model_name': 'bad name'})
    assert resp.status_code == 400


def test_model_name_valid(client):
    resp = client.post('/three/start', json={'model_name': 'model-1.0'})
    assert resp.status_code == 200

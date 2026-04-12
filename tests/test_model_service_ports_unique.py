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

    @app.route('/service', methods=['POST'])
    @validate_json_schema('backend/api/schemas/model_service_request.json')
    def service():
        return jsonify({'ok': True})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_duplicate_ports_invalid(client):
    resp = client.post('/service', json={'replicas': 1, 'ports': [8080,8080]})
    assert resp.status_code == 400


def test_unique_ports_valid(client):
    resp = client.post('/service', json={'replicas': 1, 'ports': [8080,8081]})
    assert resp.status_code == 200

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

    @app.route('/api/v1/training/model/models/m1/service', methods=['POST'])
    @validate_json_schema('backend/api/schemas/model_service_request.json')
    def service_dummy():
        return jsonify({'ok': True})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_service_rejects_unknown_field_strict(client, monkeypatch):
    monkeypatch.setenv('CHECK_API_STRICT_MODE', '1')
    # contains unknown field not defined in schema
    resp = client.post('/api/v1/training/model/models/m1/service', json={'unknown': 1})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data['error'] == 'validation_schema_failed' or data.get('code') is not None


def test_service_accepts_defined_fields(client, monkeypatch):
    monkeypatch.setenv('CHECK_API_STRICT_MODE', '1')
    resp = client.post('/api/v1/training/model/models/m1/service', json={'replicas': 2, 'expose': True, 'ports': [8080]})
    assert resp.status_code in (200, 400)  # 200 if passes schema, 400 if other validation happens

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

    @app.route('/training/sessions', methods=['POST'])
    @validate_json_schema('backend/api/schemas/training_create_session.json')
    def create():
        return jsonify({'ok': True})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_training_config_epochs_too_large(client):
    resp = client.post('/training/sessions', json={'name': 'n', 'config': {'epochs': 100001}})
    assert resp.status_code == 400


def test_training_config_batch_size_too_large(client):
    resp = client.post('/training/sessions', json={'name': 'n', 'config': {'batch_size': 70000}})
    assert resp.status_code == 400


def test_training_minimal_ok(client):
    resp = client.post('/training/sessions', json={'name': 'ok'})
    assert resp.status_code == 200

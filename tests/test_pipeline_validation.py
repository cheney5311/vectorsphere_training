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

    @app.route('/api/v1/training/pipeline/create', methods=['POST'])
    @validate_json_schema('backend/api/schemas/pipeline_create.json')
    def create_dummy():
        return jsonify({'ok': True})

    @app.route('/api/v1/training/pipeline/start', methods=['POST'])
    @validate_json_schema('backend/api/schemas/pipeline_start.json')
    def start_dummy():
        return jsonify({'ok': True})

    @app.route('/api/v1/training/pipeline/pause', methods=['POST'])
    @validate_json_schema('backend/api/schemas/pipeline_session_id.json')
    def pause_dummy():
        return jsonify({'ok': True})

    @app.route('/api/v1/training/pipeline/resume', methods=['POST'])
    @validate_json_schema('backend/api/schemas/pipeline_session_id.json')
    def resume_dummy():
        return jsonify({'ok': True})

    @app.route('/api/v1/training/pipeline/rollback', methods=['POST'])
    @validate_json_schema('backend/api/schemas/pipeline_start.json')
    def rollback_dummy():
        return jsonify({'ok': True})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_pipeline_create_requires_name(client):
    resp = client.post('/api/v1/training/pipeline/create', json={})
    assert resp.status_code == 400


def test_pipeline_start_requires_name(client):
    resp = client.post('/api/v1/training/pipeline/start', json={})
    assert resp.status_code == 400


def test_pipeline_pause_requires_session_id(client):
    resp = client.post('/api/v1/training/pipeline/pause', json={})
    assert resp.status_code == 400


def test_pipeline_resume_requires_session_id(client):
    resp = client.post('/api/v1/training/pipeline/resume', json={})
    assert resp.status_code == 400


def test_pipeline_rollback_requires_name(client):
    resp = client.post('/api/v1/training/pipeline/rollback', json={})
    assert resp.status_code == 400

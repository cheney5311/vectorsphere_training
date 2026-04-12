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


def test_pretrain_epochs_invalid(client):
    payload = {
        'model_name': 'm',
        'stages': {'pretrain': {'enabled': True, 'epochs': 0}}
    }
    resp = client.post('/three/start', json=payload)
    assert resp.status_code == 400


def test_finetune_lr_invalid(client):
    payload = {
        'model_name': 'm',
        'stages': {'finetune': {'enabled': True, 'learning_rate': 0}}
    }
    resp = client.post('/three/start', json=payload)
    assert resp.status_code == 400


def test_preference_batchsize_invalid(client):
    payload = {
        'model_name': 'm',
        'stages': {'preference': {'enabled': True, 'batch_size': 0}}
    }
    resp = client.post('/three/start', json=payload)
    assert resp.status_code == 400

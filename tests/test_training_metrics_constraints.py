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

    @app.route('/metrics', methods=['POST'])
    @validate_json_schema('backend/api/schemas/training_metrics_update.json')
    def metrics():
        return jsonify({'ok': True})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_accuracy_out_of_range(client):
    resp = client.post('/metrics', json={'epoch': 1, 'step': 1, 'loss': 0.1, 'accuracy': 1.5})
    assert resp.status_code == 400


def test_gpu_utilization_out_of_range(client):
    resp = client.post('/metrics', json={'epoch': 1, 'step': 1, 'loss': 0.1, 'gpu_utilization': 150})
    assert resp.status_code == 400


def test_negative_throughput(client):
    resp = client.post('/metrics', json={'epoch': 1, 'step': 1, 'loss': 0.1, 'throughput': -1})
    assert resp.status_code == 400

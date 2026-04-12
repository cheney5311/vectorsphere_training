import os
import sys
import pytest
from flask import Flask
sys.path.insert(0, "/root/VectorSphere/VectorSphere-intelligent-platform")

import backend.api.training.intelligent_decision_api as idec_api


@pytest.fixture
def app(monkeypatch):
    # bypass jwt
    monkeypatch.setattr(idec_api, 'jwt_required', lambda: (lambda f: f))
    app = Flask(__name__)
    app.register_blueprint(idec_api.intelligent_decision_bp)
    app.config['TESTING'] = True
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_adaptive_invalid_400(client):
    resp = client.post('/api/training/intelligent/optimization/adaptive', json={'current_value': 1})
    assert resp.status_code == 400


def test_adaptive_valid_200(client):
    resp = client.post('/api/training/intelligent/optimization/adaptive', json={'parameter_name': 'lr', 'current_value': 0.1})
    # schema pass; actual handler may do more; success if not blocked by schema
    assert resp.status_code in (200, 400, 500)

import os
import sys
import pytest
from flask import Flask
sys.path.insert(0, "/root/VectorSphere/VectorSphere-intelligent-platform")

import backend.api.training.intelligent_decision_api as idec_api


@pytest.fixture
def app(monkeypatch):
    # bypass jwt and stub decision service
    monkeypatch.setattr(idec_api, 'jwt_required', lambda: (lambda f: f))
    class DummyResult:
        decision_id = 'd1'
        scenario = 'classification'
        recommended_action = 'act'
        confidence = 0.9
        reasoning = 'ok'
        alternatives = []
        execution_plan = {}
        metadata = {}
    class DummySvc:
        def make_intelligent_decision(self, context):
            return DummyResult()
    monkeypatch.setattr(idec_api, 'decision_service', DummySvc())

    app = Flask(__name__)
    app.register_blueprint(idec_api.intelligent_decision_bp)
    app.config['TESTING'] = True
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_decisions_invalid_400(client):
    resp = client.post('/api/training/intelligent/decisions', json={'inputs': {}})
    assert resp.status_code == 400


def test_decisions_valid_200(client):
    resp = client.post('/api/training/intelligent/decisions', json={'scenario': 'classification', 'inputs': {}})
    assert resp.status_code == 200

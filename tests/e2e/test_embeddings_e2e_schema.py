import os
import sys
import pytest
from flask import Flask
sys.path.insert(0, "/root/VectorSphere/VectorSphere-intelligent-platform")

# Import module after path injected
import backend.api.embeddings.api as emb_api


@pytest.fixture
def app(monkeypatch):
    # bypass jwt and stub manager
    monkeypatch.setattr(emb_api, 'jwt_required', lambda: (lambda f: f))
    class DummyMgr:
        def generate_embedding(self, text, model_type):
            import numpy as np
            return np.array([0.1, 0.2])
        def generate_batch_embeddings(self, texts, model_type):
            import numpy as np
            return np.array([[0.1, 0.2] for _ in texts])
        def calculate_similarity(self, a, b):
            return 1.0
    monkeypatch.setattr(emb_api, 'get_embedding_manager', lambda: DummyMgr())

    app = Flask(__name__)
    app.register_blueprint(emb_api.embeddings_bp)
    app.config['TESTING'] = True
    yield app


@pytest.fixture
def client(app):
    return app.test_client()


def test_generate_invalid_body_returns_400(client):
    resp = client.post('/api/v1/embeddings/generate', json={})
    assert resp.status_code == 400


def test_generate_valid_body_200(client):
    resp = client.post('/api/v1/embeddings/generate', json={'text': 'hello'})
    assert resp.status_code == 200


def test_batch_generate_invalid_body_returns_400(client):
    resp = client.post('/api/v1/embeddings/batch-generate', json={})
    assert resp.status_code == 400


def test_batch_generate_valid_body_200(client):
    resp = client.post('/api/v1/embeddings/batch-generate', json={'texts': ['a', 'b']})
    assert resp.status_code == 200

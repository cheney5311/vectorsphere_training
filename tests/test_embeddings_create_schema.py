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

    @app.route('/embeddings', methods=['POST'])
    @validate_json_schema('backend/api/schemas/embeddings_create.json')
    def create_emb():
        return jsonify({'ok': True})

    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


def test_missing_texts_invalid(client):
    resp = client.post('/embeddings', json={'model_id': 'm'})
    assert resp.status_code == 400


def test_text_valid(client):
    resp = client.post('/embeddings', json={'text': 'hello'})
    assert resp.status_code == 200


def test_texts_valid(client):
    resp = client.post('/embeddings', json={'texts': ['a', 'b']})
    assert resp.status_code == 200

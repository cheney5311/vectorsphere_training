import os
import json
import tempfile
from backend.utils.checkpoint_uploader import enqueue_upload, run_worker_once


def test_enqueue_and_worker_once(tmp_path, monkeypatch):
    qdir = tmp_path / 'queue'
    qdir.mkdir()
    # prepare a fake local file
    local = tmp_path / 'ckpt.pt'
    local.write_bytes(b'data')
    meta = {'checkpoint': 'ckpt.pt', 'sha256': 'dummy'}

    task_file = enqueue_upload(str(local), 'ckpt.pt', 'test-bucket', str(tmp_path / 'ckpt.pt.meta.json'), meta, queue_dir=str(qdir))
    assert os.path.exists(task_file)

    # monkeypatch storage adapter to always succeed
    class FakeAdapter:
        def __init__(self, bucket):
            pass
        def upload(self, local_path, object_name=None):
            return True

    monkeypatch.setitem(__import__('sys').modules, 'backend.utils.checkpoint_storage', __import__('types').SimpleNamespace(S3Adapter=FakeAdapter))

    processed = run_worker_once(str(qdir))
    assert processed == 1
    # queue dir should be empty
    assert len([f for f in os.listdir(str(qdir)) if f.endswith('.json')]) == 0

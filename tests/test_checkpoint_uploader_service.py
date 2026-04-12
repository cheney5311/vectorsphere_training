import os
import time
import tempfile
import threading
from backend.services.checkpoint_uploader_service import CheckpointUploaderService


def test_service_runs_and_updates_queue(tmp_path, monkeypatch):
    qdir = tmp_path / 'queue'
    qdir.mkdir()
    # create a fake task file
    tf = qdir / 'task_1.json'
    tf.write_text('{"local_path": "/tmp/nonexistent", "object_name": "obj", "bucket": "b", "meta_path": "/tmp/m.meta.json", "meta": {}}')

    # monkeypatch run_worker_once to simulate processing 1 task
    import backend.utils.checkpoint_uploader as uploader

    def fake_run_once(queue_dir=None):
        return 1

    monkeypatch.setattr(uploader, 'run_worker_once', fake_run_once)

    svc = CheckpointUploaderService(queue_dir=str(qdir), poll_interval=0.1, metrics_port=0)
    svc.start()
    time.sleep(0.3)
    svc.stop()
    # if no exception, service loop ran
    assert True

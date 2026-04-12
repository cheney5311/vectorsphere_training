"""Checkpoint uploader service (daemon) with Prometheus metrics endpoint.

- Runs `backend.utils.checkpoint_uploader.run_worker_once` in a loop.
- Exposes Prometheus metrics (processed, failed, retained, queue_size, last_run_timestamp).
- Configurable via ENV:
  - CHECKPOINT_QUEUE_DIR (default /tmp/checkpoint_upload_queue)
  - CHECKPOINT_UPLOADER_POLL_INTERVAL (seconds, default 5)
  - CHECKPOINT_UPLOADER_METRICS_PORT (default 8001)

Usage:
    python -m backend.services.checkpoint_uploader_service

Systemd unit example (not created):
[Unit]
Description=VectorSphere Checkpoint Uploader Service
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/VectorSphere-intelligent-platform
ExecStart=/usr/bin/env python -m backend.services.checkpoint_uploader_service
Restart=on-failure

[Install]
WantedBy=multi-user.target
"""
import os
import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

QUEUE_DIR = os.environ.get('CHECKPOINT_QUEUE_DIR', '/tmp/checkpoint_upload_queue')
POLL_INTERVAL = float(os.environ.get('CHECKPOINT_UPLOADER_POLL_INTERVAL', '5'))
METRICS_PORT = int(os.environ.get('CHECKPOINT_UPLOADER_METRICS_PORT', '8001'))

try:
    from prometheus_client import start_http_server, Counter, Gauge
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False

# metrics
_processed_counter = None
_failed_counter = None
_retained_counter = None
_queue_size_gauge = None
_last_run_gauge = None


def _init_metrics():
    global _processed_counter, _failed_counter, _retained_counter, _queue_size_gauge, _last_run_gauge
    if not _HAS_PROM:
        logger.warning('prometheus_client not available: metrics disabled')
        return
    _processed_counter = Counter('checkpoint_uploader_processed_total', 'Total processed upload tasks')
    _failed_counter = Counter('checkpoint_uploader_failed_total', 'Total failed upload tasks')
    _retained_counter = Counter('checkpoint_uploader_retained_total', 'Total requeued/retained upload tasks')
    _queue_size_gauge = Gauge('checkpoint_uploader_queue_size', 'Current number of pending tasks in queue')
    _last_run_gauge = Gauge('checkpoint_uploader_last_run_timestamp', 'Unix timestamp of last worker run')


def _update_queue_size(queue_dir: str):
    if not _HAS_PROM or _queue_size_gauge is None:
        return
    try:
        if not os.path.exists(queue_dir):
            _queue_size_gauge.set(0)
            return
        n = len([f for f in os.listdir(queue_dir) if f.endswith('.json')])
        _queue_size_gauge.set(n)
    except Exception:
        pass


class CheckpointUploaderService:
    def __init__(self, queue_dir: Optional[str] = None, poll_interval: float = POLL_INTERVAL, metrics_port: int = METRICS_PORT):
        self.queue_dir = queue_dir or QUEUE_DIR
        self.poll_interval = poll_interval
        self.metrics_port = metrics_port
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        logger.info('Starting CheckpointUploaderService')
        if _HAS_PROM:
            start_http_server(self.metrics_port)
            logger.info(f'Prometheus metrics available on port {self.metrics_port}')
        _init_metrics()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        logger.info('Stopping CheckpointUploaderService')
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self):
        from backend.utils.checkpoint_uploader import run_worker_once
        while not self._stop.is_set():
            try:
                _update_queue_size(self.queue_dir)
                processed = run_worker_once(self.queue_dir)
                # The uploader run_worker_once logs processed/failed/retained; we try to update metrics by scanning queue and assuming counts
                if _HAS_PROM and _processed_counter is not None:
                    # best-effort: increment processed counter by processed
                    try:
                        _processed_counter.inc(processed)
                    except Exception:
                        pass
                if _HAS_PROM and _last_run_gauge is not None:
                    _last_run_gauge.set(time.time())
            except Exception as e:
                logger.error(f'Uploader service loop error: {e}')
            # sleep poll interval
            time.sleep(self.poll_interval)


def main():
    svc = CheckpointUploaderService()
    try:
        svc.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        svc.stop()


if __name__ == '__main__':
    main()

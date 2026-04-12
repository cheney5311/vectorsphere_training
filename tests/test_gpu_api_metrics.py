import time
from flask import Flask
from backend.api.resources.gpu_api import gpu_bp
from backend.modules.distributed.cluster_manager import get_cluster_manager, NodeInfo, GPUInfo, NodeStatus
from backend.modules.monitoring.metrics_exporter import ALLOCATION_REQUESTS_COUNTER, ALLOCATION_FAILURES_COUNTER


def make_node(node_id: str, gpu_count: int = 0, gpu_free_mem: int = 10240):
    gpus = []
    for i in range(gpu_count):
        gpus.append(GPUInfo(
            gpu_id=i,
            name=f"GPU-{i}",
            memory_total=gpu_free_mem,
            memory_used=0,
            memory_free=gpu_free_mem,
            utilization=0.0,
            temperature=40.0,
            power_usage=50.0,
            driver_version="450",
            cuda_version="11.0",
            is_available=True,
            processes=[]
        ))
    node = NodeInfo(
        node_id=node_id,
        hostname="localhost",
        ip_address="127.0.0.1",
        port=22,
        status=NodeStatus.HEALTHY,
        last_heartbeat=None,
        cpu_count=8,
        memory_total=64000,
        memory_used=0,
        disk_total=100000,
        disk_used=0,
        gpus=gpus,
        cpu_utilization=0.0,
        memory_utilization=0.0,
        disk_utilization=0.0
    )
    return node


def test_metrics_on_allocate():
    app = Flask(__name__)
    app.register_blueprint(gpu_bp)
    client = app.test_client()

    cm = get_cluster_manager()
    cm.nodes.clear()
    cm.nodes['n1'] = make_node('n1', gpu_count=1)

    # reset counters
    # Note: prometheus_client Counters cannot be reset; we will record current samples relative to them
    before_success = list(ALLOCATION_REQUESTS_COUNTER.collect())[0].samples

    payload = {'requirement': {'cpu_cores': 1, 'memory_mb': 1024, 'gpu_count': 1, 'gpu_memory_mb': 1024, 'priority': 1}}
    resp = client.post('/api/v1/gpus/allocate', json=payload)
    assert resp.status_code == 200

    # give metrics a moment
    time.sleep(0.1)
    after_success = list(ALLOCATION_REQUESTS_COUNTER.collect())[0].samples

    # ensure at least one 'success' label increment
    success_inc = 0
    for s in after_success:
        if s.labels.get('result') == 'success':
            success_inc = s.value
    assert success_inc >= 1

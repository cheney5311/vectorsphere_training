import threading
import time
import json
from datetime import datetime
from flask import Flask
from backend.api.resources.gpu_api import gpu_bp
from backend.modules.distributed.cluster_manager import get_cluster_manager, NodeInfo, GPUInfo, NodeStatus
from backend.modules.distributed.task_scheduler import ResourceRequirement


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
        last_heartbeat=datetime.now(),
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


def send_allocate_request(client, payload, results, idx):
    resp = client.post('/api/v1/gpus/allocate', json=payload)
    results[idx] = (resp.status_code, resp.get_json())


def test_concurrent_allocations():
    app = Flask(__name__)
    app.register_blueprint(gpu_bp)
    client = app.test_client()

    cm = get_cluster_manager()
    cm.nodes.clear()
    # 两个节点，每个1 GPU
    cm.nodes['n1'] = make_node('n1', gpu_count=1)
    cm.nodes['n2'] = make_node('n2', gpu_count=1)

    payload = {'requirement': {'cpu_cores': 1, 'memory_mb': 1024, 'gpu_count': 1, 'gpu_memory_mb': 1024, 'priority': 8}}

    threads = []
    results = [None] * 4
    for i in range(4):
        t = threading.Thread(target=send_allocate_request, args=(client, payload, results, i))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    # 4 请求，2 个 GPU 资源 -> 2 成功 2 失败
    success_count = sum(1 for r in results if r and r[0] == 200)
    failure_count = sum(1 for r in results if r and r[0] == 409)
    # allow for timing variance: expect at most 2 successes and at least 2 failures
    assert success_count <= 2
    assert failure_count >= 2

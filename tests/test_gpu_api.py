import json
from datetime import datetime
from flask import Flask
from backend.api.resources.gpu_api import gpu_bp
from backend.modules.distributed.cluster_manager import get_cluster_manager, NodeInfo, GPUInfo


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
        status=None,
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


def test_gpu_allocate_api_success():
    app = Flask(__name__)
    app.register_blueprint(gpu_bp)
    client = app.test_client()

    cm = get_cluster_manager()
    cm.nodes.clear()
    from backend.modules.distributed.cluster_manager import NodeStatus
    node = make_node('node-1', gpu_count=1)
    node.labels = {'region': 'us-east'}
    node.status = NodeStatus.HEALTHY
    cm.nodes[node.node_id] = node

    payload = {
        'requirement': {
            'cpu_cores': 1,
            'memory_mb': 1024,
            'gpu_count': 1,
            'gpu_memory_mb': 1024,
            'priority': 1
        }
    }

    resp = client.post('/api/v1/gpus/allocate', json=payload)
    data = resp.get_json()
    assert resp.status_code == 200
    assert data.get('allocated') is True


def test_gpu_allocate_api_failure_no_capacity():
    app = Flask(__name__)
    app.register_blueprint(gpu_bp)
    client = app.test_client()

    cm = get_cluster_manager()
    cm.nodes.clear()

    payload = {
        'requirement': {
            'cpu_cores': 1,
            'memory_mb': 1024,
            'gpu_count': 1,
            'gpu_memory_mb': 1024,
            'priority': 1
        }
    }

    resp = client.post('/api/v1/gpus/allocate', json=payload)
    data = resp.get_json()
    assert resp.status_code == 409
    assert data.get('allocated') is False
    assert data.get('reason') == 'no_capacity'

import asyncio
import os
import time
from backend.modules.distributed.resource_allocator import ResourceAllocator, AllocationStrategy
from backend.modules.distributed.cluster_manager import NodeInfo, GPUInfo
from backend.modules.distributed.task_scheduler import ResourceRequirement
from datetime import datetime


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


def test_allocate_single_gpu_immediate():
    allocator = ResourceAllocator()
    node = make_node("node-1", gpu_count=1)
    req = ResourceRequirement(cpu_cores=1, memory_mb=1024, gpu_count=1, gpu_memory_mb=1024, priority=1)

    res = asyncio.get_event_loop().run_until_complete(allocator.allocate_resources([node], req))
    assert res is not None
    alloc_id, alloc = res
    assert alloc.node_id == "node-1"
    assert len(alloc.gpus) == 1


def test_allocate_insufficient_gpus_high_priority_wait():
    # set short wait time to speed test
    os.environ['GPU_ALLOC_WAIT_SECONDS'] = '1'
    os.environ['GPU_ALLOC_RETRY_INTERVAL'] = '0.2'

    allocator = ResourceAllocator()
    node = make_node("node-1", gpu_count=1)
    req = ResourceRequirement(cpu_cores=1, memory_mb=1024, gpu_count=2, gpu_memory_mb=1024, priority=10)

    start = time.time()
    res = asyncio.get_event_loop().run_until_complete(allocator.allocate_resources([node], req))
    elapsed = time.time() - start

    # should have waited at least a bit (since priority>5) and then failed
    assert res is None
    assert elapsed >= 0.2


def test_labels_affinity_matching():
    allocator = ResourceAllocator()
    node1 = make_node("node-1", gpu_count=1)
    node1.labels = {"region": "us-east"}
    node2 = make_node("node-2", gpu_count=1)
    node2.labels = {"region": "us-west"}

    req = ResourceRequirement(cpu_cores=1, memory_mb=1024, gpu_count=1, gpu_memory_mb=1024, priority=1, labels_affinity={"region": "us-west"})
    res = asyncio.get_event_loop().run_until_complete(allocator.allocate_resources([node1, node2], req))
    assert res is not None
    alloc_id, alloc = res
    assert alloc.node_id == "node-2"

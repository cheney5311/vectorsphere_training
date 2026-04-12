"""持久化资源释放队列与 worker

- 将待释放的 allocation 任务序列化为 JSON 文件放入 queue_dir
- 提供 `enqueue_release` 接口与 `run_worker_once` 用于处理队列（支持独立 worker 运行）
"""
import os
import json
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def enqueue_release(allocation_ref: Dict[str, Any], queue_dir: Optional[str] = None) -> str:
    """将释放任务写入磁盘队列，返回任务文件路径

    allocation_ref: { 'allocation_id': str } 或 { 'node': str, 'gpu_indices': [...] }
    """
    if queue_dir is None:
        queue_dir = os.environ.get('RESOURCE_RELEASE_QUEUE_DIR', '/tmp/resource_release_queue')
    _ensure_dir(queue_dir)
    task = {
        'allocation_ref': allocation_ref,
        'created_at': time.time()
    }
    task_id = f"release_{int(time.time()*1000)}_{os.getpid()}"
    task_file = os.path.join(queue_dir, f"{task_id}.json")
    with open(task_file, 'w') as f:
        json.dump(task, f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    logger.info(f"Enqueued release task: {task_file}")
    return task_file


def run_worker_once(queue_dir: Optional[str] = None) -> int:
    """处理队列中的任务（单次运行），返回处理数量"""
    if queue_dir is None:
        queue_dir = os.environ.get('RESOURCE_RELEASE_QUEUE_DIR', '/tmp/resource_release_queue')
    if not os.path.exists(queue_dir):
        return 0
    files = sorted([f for f in os.listdir(queue_dir) if f.endswith('.json')])
    processed = 0
    for fname in files:
        task_file = os.path.join(queue_dir, fname)
        try:
            with open(task_file, 'r') as f:
                task = json.load(f)
            ref = task.get('allocation_ref')
            if not ref:
                os.remove(task_file)
                continue
            try:
                from backend.modules.distributed.resource_allocator import get_resource_allocator
                allocator = get_resource_allocator()
                # 优先 allocation_id
                alloc_id = ref.get('allocation_id')
                if alloc_id:
                    ok = False
                    try:
                        ok = asyncio_run_release(allocator, alloc_id)
                    except Exception as e:
                        logger.error(f"Release attempt failed for {alloc_id}: {e}")
                        ok = False
                    if ok:
                        os.remove(task_file)
                        processed += 1
                        continue
                # 回退到 node/gpus 匹配释放
                node = ref.get('node')
                gpus = ref.get('gpu_indices', []) or ref.get('gpus', [])
                if node:
                    try:
                        allocation_ids = asyncio_find_and_release(allocator, node, gpus)
                        if allocation_ids:
                            os.remove(task_file)
                            processed += 1
                            continue
                    except Exception as e:
                        logger.error(f"Release by node failed for {task_file}: {e}")
                        continue
                # 如果无法释放，保留任务以便下次重试
            except Exception as e:
                logger.error(f"Failed to process release task {task_file}: {e}")
                continue
        except Exception as e:
            logger.error(f"Failed to read task file {task_file}: {e}")
            try:
                os.remove(task_file)
            except Exception:
                pass
    return processed


def asyncio_run_release(allocator, alloc_id: str) -> bool:
    """在当前进程中同步执行异步 release_resources 方法并返回结果"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 在活动 loop 中调度
            fut = asyncio.ensure_future(allocator.release_resources(alloc_id))
            # 不等待长时间结果，返回 True/False 需要外部 worker 使用 run_until_complete
            return True
        else:
            return loop.run_until_complete(allocator.release_resources(alloc_id))
    except Exception:
        try:
            loop2 = asyncio.new_event_loop()
            res = loop2.run_until_complete(allocator.release_resources(alloc_id))
            loop2.close()
            return res
        except Exception as e:
            logger.error(f"asyncio_run_release error: {e}")
            return False


def asyncio_find_and_release(allocator, node: str, gpus: list) -> list:
    """查找 allocation_ids 并释放，返回已释放的 allocation_ids 列表"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 在活动 loop 中无法同步等待，返回空以示未处理
            return []
        else:
            return loop.run_until_complete(_find_and_release_async(allocator, node, gpus))
    except Exception:
        try:
            loop2 = asyncio.new_event_loop()
            res = loop2.run_until_complete(_find_and_release_async(allocator, node, gpus))
            loop2.close()
            return res
        except Exception as e:
            logger.error(f"asyncio_find_and_release error: {e}")
            return []

async def _find_and_release_async(allocator, node: str, gpus: list) -> list:
    ids = await allocator.find_allocations_by_node_and_gpus(node, gpus)
    released = []
    for aid in ids:
        try:
            ok = await allocator.release_resources(aid)
            if ok:
                released.append(aid)
        except Exception:
            pass
    return released


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    qdir = os.environ.get('RESOURCE_RELEASE_QUEUE_DIR', '/tmp/resource_release_queue')
    while True:
        count = run_worker_once(qdir)
        if count == 0:
            time.sleep(5)
        else:
            time.sleep(1)

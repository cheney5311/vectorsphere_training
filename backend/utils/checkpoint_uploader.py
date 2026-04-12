"""持久化上传队列与后台 worker（轻量实现）
- 将上传任务序列化为 JSON 文件放入 queue_dir
- 提供 `enqueue_upload` 接口与 `run_worker_once` 用于处理队列（适合由 supervisor/cron 调度或长期 worker 运行）
- 设计目的是避免在训练进程中丢失上传任务（持久化到磁盘）
"""
import os
import json
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def enqueue_upload(local_path: str, object_name: str, bucket: str, meta_path: str, meta: Dict[str, Any], queue_dir: Optional[str] = None) -> str:
    """将上传任务写入磁盘队列，返回任务文件路径"""
    if queue_dir is None:
        queue_dir = os.environ.get('CHECKPOINT_QUEUE_DIR', '/tmp/checkpoint_upload_queue')
    _ensure_dir(queue_dir)
    task = {
        'local_path': local_path,
        'object_name': object_name,
        'bucket': bucket,
        'meta_path': meta_path,
        'meta': meta,
        'created_at': time.time()
    }
    task_id = f"task_{int(time.time()*1000)}_{os.getpid()}"
    task_file = os.path.join(queue_dir, f"{task_id}.json")
    with open(task_file, 'w') as f:
        json.dump(task, f)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    logger.info(f"Enqueued upload task: {task_file}")
    return task_file


def run_worker_once(queue_dir: Optional[str] = None) -> int:
    """处理队列中的一个或多个任务（单次运行）。
    增强特性：
    - 在处理前将任务文件重命名为 .inprogress，避免多进程竞争
    - 支持重试计数与指数退避（任务 JSON 中可包含 `retries` 字段）
    - 失败任务移动到 failed/ 目录以便人工或后续自动化处理
    - 处理成功后原子更新 meta（使用 checkpoint_metadata helpers）
    - 简单的指标日志（processed/failed/retained）
    返回已处理任务数量。
    """
    import shutil
    import math

    if queue_dir is None:
        queue_dir = os.environ.get('CHECKPOINT_QUEUE_DIR', '/tmp/checkpoint_upload_queue')
    if not os.path.exists(queue_dir):
        return 0

    failed_dir = os.path.join(queue_dir, 'failed')
    inprog_suffix = '.inprogress'
    _ensure_dir(failed_dir)

    files = sorted([f for f in os.listdir(queue_dir) if f.endswith('.json')])
    processed = 0
    failed = 0
    retained = 0

    for fname in files:
        task_file = os.path.join(queue_dir, fname)
        # Skip if it's a directory
        if os.path.isdir(task_file):
            continue
        inprog_file = task_file + inprog_suffix
        try:
            # Try to atomically claim the task by renaming
            try:
                os.replace(task_file, inprog_file)
            except Exception:
                # someone else may be processing it
                continue

            with open(inprog_file, 'r') as f:
                task = json.load(f)

            # Read retry info
            retries = int(task.get('retries', 0))
            max_retries = int(task.get('max_retries', os.environ.get('CHECKPOINT_UPLOAD_RETRIES', '3')))
            attempt = retries + 1

            backoff_base = float(task.get('backoff_base', os.environ.get('CHECKPOINT_UPLOAD_BACKOFF', '0.5')))
            backoff = backoff_base * (2 ** (retries)) if retries > 0 else 0

            try:
                from backend.utils.checkpoint_storage import S3Adapter
                adapter = S3Adapter(bucket=task['bucket'])

                # Perform upload with adapter
                ok = adapter.upload(task['local_path'], object_name=task['object_name'])

                if ok:
                    # 更新 meta file 原子替换（使用 metadata utilities if available）
                    try:
                        meta_path = task.get('meta_path')
                        if meta_path:
                            merged = dict(task.get('meta', {}))
                            merged.update({
                                'bucket': task['bucket'],
                                'object_name': task['object_name'],
                                'uploaded_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                'meta_version': merged.get('meta_version', os.environ.get('CHECKPOINT_META_VERSION', '1.1'))
                            })
                            try:
                                # normalize/migrate meta if helper exists
                                from backend.utils.checkpoint_metadata import migrate_meta, get_meta_version
                                try:
                                    merged = migrate_meta(merged, target_version=get_meta_version(merged))
                                except Exception:
                                    pass
                            except Exception:
                                pass

                            tmp_meta = meta_path + '.tmp'
                            with open(tmp_meta, 'w') as mf:
                                json.dump(merged, mf)
                                mf.flush()
                                try:
                                    os.fsync(mf.fileno())
                                except Exception:
                                    pass
                            os.replace(tmp_meta, meta_path)
                    except Exception as e:
                        logger.warning(f"Failed to update meta after background upload: {e}")

                    # 成功，移除 inprogress 文件
                    try:
                        os.remove(inprog_file)
                    except Exception:
                        pass
                    processed += 1
                else:
                    # 上传适配器返回 False -> treat as failure but retryable
                    raise RuntimeError('Upload adapter returned False')

            except Exception as e:
                logger.error(f"Upload failed for task {inprog_file}: {e}")
                # handle retry
                if attempt <= int(max_retries):
                    # update retries and put back to queue with backoff delay encoded
                    task['retries'] = attempt
                    # write updated task back
                    try:
                        with open(inprog_file, 'w') as f:
                            json.dump(task, f)
                            f.flush()
                            try:
                                os.fsync(f.fileno())
                            except Exception:
                                pass
                        # rename back to queue name for future processing
                        os.replace(inprog_file, task_file)
                        retained += 1
                        # sleep small backoff to avoid hot loop
                        try:
                            time.sleep(min(backoff, 5))
                        except Exception:
                            pass
                    except Exception as ex:
                        logger.error(f"Failed to requeue task {inprog_file}: {ex}")
                        # move to failed
                        try:
                            dest = os.path.join(failed_dir, os.path.basename(inprog_file))
                            os.replace(inprog_file, dest)
                            failed += 1
                        except Exception:
                            pass
                else:
                    # exceed retries -> move to failed
                    try:
                        dest = os.path.join(failed_dir, os.path.basename(inprog_file))
                        os.replace(inprog_file, dest)
                        failed += 1
                    except Exception as ex:
                        logger.error(f"Failed to move failed task {inprog_file}: {ex}")
            
        except Exception as e:
            logger.error(f"Failed to process task file {task_file}: {e}")
            try:
                # try to cleanup any inprogress artifact
                if os.path.exists(inprog_file):
                    dest = os.path.join(failed_dir, os.path.basename(inprog_file))
                    os.replace(inprog_file, dest)
                    failed += 1
            except Exception:
                pass
            try:
                if os.path.exists(task_file):
                    os.remove(task_file)
            except Exception:
                pass

    logger.info(f"Checkpoint uploader: processed={processed} failed={failed} retained={retained}")
    return processed


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    qdir = os.environ.get('CHECKPOINT_QUEUE_DIR', '/tmp/checkpoint_upload_queue')
    while True:
        count = run_worker_once(qdir)
        if count == 0:
            time.sleep(5)
        else:
            time.sleep(1)

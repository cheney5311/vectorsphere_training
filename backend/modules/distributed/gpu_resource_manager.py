"""模块级 GPU 资源管理器
- 周期性采集本机 GPU 指标（包装 `backend.services.gpu_resource_manager`）
- 提供内存缓存的指标查询接口
- 由 ClusterManager 调用或独立启动监控线程
"""
import threading
import time
import logging
from typing import List, Dict, Any, Optional

from backend.services import gpu_resource_manager as svc

logger = logging.getLogger(__name__)

_metrics_cache: Dict[str, Any] = {}
_metrics_lock = threading.Lock()
_monitor_thread: Optional[threading.Thread] = None
_monitor_stop = False


def collect_once() -> Dict[str, Any]:
    try:
        summary = svc.get_gpu_summary()
        with _metrics_lock:
            _metrics_cache['last'] = summary
            _metrics_cache['updated_at'] = time.time()
        return summary
    except Exception as e:
        logger.exception(f"GPU metrics collection failed: {e}")
        return {}


def get_cached_metrics() -> Dict[str, Any]:
    with _metrics_lock:
        return dict(_metrics_cache)


def _monitor_loop(interval: int):
    global _monitor_stop
    while not _monitor_stop:
        collect_once()
        time.sleep(interval)


def start_monitoring(interval: int = 10, background: bool = True) -> None:
    """启动周期性采集（可多次调用幂等）"""
    global _monitor_thread, _monitor_stop
    if _monitor_thread and _monitor_thread.is_alive():
        return
    _monitor_stop = False
    if background:
        _monitor_thread = threading.Thread(target=_monitor_loop, args=(interval,), daemon=True)
        _monitor_thread.start()
    else:
        _monitor_loop(interval)


def stop_monitoring() -> None:
    global _monitor_stop, _monitor_thread
    _monitor_stop = True
    if _monitor_thread:
        try:
            _monitor_thread.join(timeout=2.0)
        except Exception:
            pass
        _monitor_thread = None

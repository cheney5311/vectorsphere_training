"""任务注册中心抽象接口与选择器"""

import os
from typing import Optional
try:
    from .task_registry import registry as in_memory_registry
except Exception:
    in_memory_registry = None

_registry_choice = os.environ.get("TRAINING_TASK_REGISTRY", "memory").lower()
_redis_url = os.environ.get("REDIS_URL") or os.environ.get("TRAINING_REDIS_URL")

class ITaskRegistry:
    def ensure_task(self, name: str): ...
    def heartbeat(self, name: str): ...
    def pause(self, session_id: str): ...
    def resume(self, session_id: str): ...
    def cancel(self, session_id: str): ...
    def is_paused(self, session_id: str) -> bool: ...
    def is_cancelled(self, session_id: str) -> bool: ...
    def get_last_heartbeat(self, name: str): ...

def get_task_registry() -> ITaskRegistry:
    # 优先 Redis
    if _registry_choice == "redis" and _redis_url:
        try:
            from .task_registry_redis import redis_registry
            return redis_registry
        except Exception:
            # 回退到内存
            pass
    # 默认内存
    if in_memory_registry is not None:
        return in_memory_registry
    # 最后的兜底：返回一个不可用的伪实现，避免崩溃
    class _Dummy(ITaskRegistry):
        def ensure_task(self, name: str): return None
        def heartbeat(self, name: str): return None
        def pause(self, session_id: str): return None
        def resume(self, session_id: str): return None
        def cancel(self, session_id: str): return None
        def is_paused(self, session_id: str) -> bool: return False
        def is_cancelled(self, session_id: str) -> bool: return False
        def get_last_heartbeat(self, name: str): return None
    return _Dummy()
"""Redis 版任务注册中心实现"""

import os
import redis
from typing import Optional
from datetime import datetime
from .task_registry_interface import ITaskRegistry

_redis_url = os.environ.get("REDIS_URL") or os.environ.get("TRAINING_REDIS_URL") or "redis://localhost:6379/0"
_r = redis.Redis.from_url(_redis_url, decode_responses=True)

# key 结构：
# task:{name}:heartbeat -> iso8601
# task:{name}:paused -> set of session_id
# task:{name}:cancelled -> set of session_id

class RedisTaskRegistry(ITaskRegistry):
    def ensure_task(self, name: str):
        # 心跳不存在则写入当前
        key = f"task:{name}:heartbeat"
        if not _r.exists(key):
            _r.set(key, datetime.utcnow().isoformat())
        return True

    def heartbeat(self, name: str):
        _r.set(f"task:{name}:heartbeat", datetime.utcnow().isoformat())

    def pause(self, session_id: str):
        name = self._extract_task_name(session_id)
        self.ensure_task(name)
        _r.sadd(f"task:{name}:paused", session_id)

    def resume(self, session_id: str):
        name = self._extract_task_name(session_id)
        self.ensure_task(name)
        _r.srem(f"task:{name}:paused", session_id)

    def cancel(self, session_id: str):
        name = self._extract_task_name(session_id)
        self.ensure_task(name)
        _r.sadd(f"task:{name}:cancelled", session_id)

    def is_paused(self, session_id: str) -> bool:
        name = self._extract_task_name(session_id)
        return bool(_r.sismember(f"task:{name}:paused", session_id))

    def is_cancelled(self, session_id: str) -> bool:
        name = self._extract_task_name(session_id)
        return bool(_r.sismember(f"task:{name}:cancelled", session_id))

    def get_last_heartbeat(self, name: str) -> Optional[datetime]:
        val = _r.get(f"task:{name}:heartbeat")
        if not val:
            return None
        try:
            return datetime.fromisoformat(val)
        except Exception:
            return None

    @staticmethod
    def _extract_task_name(session_id: str) -> str:
        return session_id.split(":")[0]

redis_registry = RedisTaskRegistry()
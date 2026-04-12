"""任务注册中心：提供训练流水线的心跳与控制标记（暂停/取消）管理"""

from typing import Dict, Optional, Set
from threading import Lock
from datetime import datetime


class TaskInfo:
    def __init__(self, name: str):
        self.name = name
        self.last_heartbeat: datetime = datetime.utcnow()
        # 针对子会话（sess:step）的控制标记
        self.paused_sessions: Set[str] = set()
        self.cancelled_sessions: Set[str] = set()


class TaskRegistry:
    def __init__(self):
        self._lock = Lock()
        self._tasks: Dict[str, TaskInfo] = {}

    def ensure_task(self, name: str) -> TaskInfo:
        with self._lock:
            info = self._tasks.get(name)
            if not info:
                info = TaskInfo(name)
                self._tasks[name] = info
            return info

    def heartbeat(self, name: str):
        with self._lock:
            info = self._tasks.get(name)
            if info:
                info.last_heartbeat = datetime.utcnow()

    def pause(self, session_id: str):
        # session_id 形如 "sess-1001:pretrain"
        name = self._extract_task_name(session_id)
        with self._lock:
            info = self.ensure_task(name)
            info.paused_sessions.add(session_id)

    def resume(self, session_id: str):
        name = self._extract_task_name(session_id)
        with self._lock:
            info = self.ensure_task(name)
            info.paused_sessions.discard(session_id)

    def cancel(self, session_id: str):
        name = self._extract_task_name(session_id)
        with self._lock:
            info = self.ensure_task(name)
            info.cancelled_sessions.add(session_id)

    def is_paused(self, session_id: str) -> bool:
        name = self._extract_task_name(session_id)
        with self._lock:
            info = self._tasks.get(name)
            return bool(info and session_id in info.paused_sessions)

    def is_cancelled(self, session_id: str) -> bool:
        name = self._extract_task_name(session_id)
        with self._lock:
            info = self._tasks.get(name)
            return bool(info and session_id in info.cancelled_sessions)

    def get_last_heartbeat(self, name: str) -> Optional[datetime]:
        with self._lock:
            info = self._tasks.get(name)
            return info.last_heartbeat if info else None

    @staticmethod
    def _extract_task_name(session_id: str) -> str:
        # 约定主 session_id 为 "sess-1001"，子会话为 "sess-1001:step"
        # 管理统一按主会话维度存储心跳和控制标记
        return session_id.split(":")[0]


# 单例
registry = TaskRegistry()
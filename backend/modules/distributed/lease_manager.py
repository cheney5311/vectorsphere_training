"""
轻量级 Lease / Heartbeat 管理器
- 提供 lease 创建、续约（heartbeat）、查询与回收回调
- 设计目标：与上层编排器（DistributedTrainer/ClusterManager）解耦，支持异步运行与回调注册
- 不直接执行调度决定，仅负责 lease 元数据与失效通知
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any

logger = logging.getLogger(__name__)


@dataclass
class Lease:
    lease_id: str
    owner_id: str
    ttl_seconds: int
    created_at: float = field(default_factory=lambda: time.time())
    last_heartbeat: float = field(default_factory=lambda: time.time())
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def expires_at(self) -> float:
        return self.last_heartbeat + self.ttl_seconds

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class LeaseManager:
    """管理 leases，并在 lease 过期时触发回调"""

    def __init__(self, check_interval: float = 5.0):
        self._leases: Dict[str, Lease] = {}
        self._lock = asyncio.Lock()
        self._check_interval = check_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._on_expire_callback: Optional[Callable[[Lease], None]] = None

    def register_expiry_callback(self, cb: Callable[[Lease], None]):
        """注册 lease 过期回调（同步或异步函数均可）"""
        self._on_expire_callback = cb

    async def start(self):
        async with self._lock:
            if self._running:
                return
            self._running = True
            self._task = asyncio.create_task(self._monitor_loop())
            logger.info("LeaseManager started")

    async def stop(self):
        async with self._lock:
            if not self._running:
                return
            self._running = False
            if self._task:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            logger.info("LeaseManager stopped")

    async def create_lease(self, lease_id: str, owner_id: str, ttl_seconds: int, metadata: Optional[Dict[str, Any]] = None) -> Lease:
        async with self._lock:
            if metadata is None:
                metadata = {}
            lease = Lease(lease_id=lease_id, owner_id=owner_id, ttl_seconds=ttl_seconds, metadata=metadata)
            self._leases[lease_id] = lease
            logger.debug(f"Created lease {lease_id} for owner {owner_id} ttl={ttl_seconds}")
            return lease

    async def heartbeat(self, lease_id: str) -> bool:
        async with self._lock:
            lease = self._leases.get(lease_id)
            if not lease:
                logger.warning(f"Heartbeat for unknown lease: {lease_id}")
                return False
            lease.last_heartbeat = time.time()
            logger.debug(f"Heartbeat received for lease {lease_id}")
            return True

    async def renew(self, lease_id: str, ttl_seconds: Optional[int] = None) -> bool:
        async with self._lock:
            lease = self._leases.get(lease_id)
            if not lease:
                return False
            if ttl_seconds is not None:
                lease.ttl_seconds = ttl_seconds
            lease.last_heartbeat = time.time()
            logger.debug(f"Renewed lease {lease_id} ttl={lease.ttl_seconds}")
            return True

    async def revoke(self, lease_id: str) -> bool:
        async with self._lock:
            lease = self._leases.pop(lease_id, None)
            if lease:
                logger.info(f"Revoked lease {lease_id}")
                return True
            return False

    async def get_lease(self, lease_id: str) -> Optional[Lease]:
        async with self._lock:
            return self._leases.get(lease_id)

    async def list_leases(self) -> Dict[str, Lease]:
        async with self._lock:
            return dict(self._leases)

    async def _monitor_loop(self):
        try:
            while self._running:
                expired = []
                now = time.time()
                async with self._lock:
                    for lid, lease in list(self._leases.items()):
                        if lease.is_expired():
                            expired.append(lease)
                            del self._leases[lid]
                # 在锁外触发回调，允许回调执行耗时操作
                for lease in expired:
                    logger.warning(f"Lease expired: {lease.lease_id} owner={lease.owner_id}")
                    try:
                        if self._on_expire_callback:
                            res = self._on_expire_callback(lease)
                            if asyncio.iscoroutine(res):
                                await res
                    except Exception as e:
                        logger.error(f"Error in expiry callback for lease {lease.lease_id}: {e}")
                await asyncio.sleep(self._check_interval)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"LeaseManager monitor loop error: {e}")


# 提供模块级单例简易访问
_default_lease_manager: Optional[LeaseManager] = None


def get_lease_manager() -> LeaseManager:
    global _default_lease_manager
    if _default_lease_manager is None:
        _default_lease_manager = LeaseManager()
    return _default_lease_manager

"""任务调度器

负责训练任务的分配和调度。
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
import logging
import os

from .cluster_manager import NodeInfo

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"      # 待处理
    SCHEDULED = "scheduled"  # 已调度
    RUNNING = "running"      # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    CANCELLED = "cancelled"  # 已取消


class SchedulingStrategy(Enum):
    """调度策略枚举"""
    ROUND_ROBIN = "round_robin"      # 轮询
    LEAST_LOADED = "least_loaded"    # 最少负载
    GPU_FIRST = "gpu_first"          # GPU优先
    MEMORY_FIRST = "memory_first"    # 内存优先


@dataclass
class ResourceRequirement:
    """资源需求

    新增字段：
    - priority: 任务优先级（1-10，10最高），用于优先调度高优先级任务
    - labels_affinity: 可选的标签亲和（字典），优先选择匹配节点
    """
    cpu_cores: int = 1
    memory_mb: int = 1024
    gpu_count: int = 0
    gpu_memory_mb: int = 0
    disk_mb: int = 1024
    network_mbps: int = 100
    priority: int = 1
    labels_affinity: Optional[Dict[str, str]] = None


@dataclass
class TrainingTask:
    """训练任务"""
    task_id: str
    name: str
    description: str = ""
    priority: int = 1  # 1-10, 10最高优先级
    status: TaskStatus = TaskStatus.PENDING
    resource_requirements: ResourceRequirement = field(default_factory=ResourceRequirement)
    assigned_node: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "resource_requirements": {
                "cpu_cores": self.resource_requirements.cpu_cores,
                "memory_mb": self.resource_requirements.memory_mb,
                "gpu_count": self.resource_requirements.gpu_count,
                "gpu_memory_mb": self.resource_requirements.gpu_memory_mb,
                "disk_mb": self.resource_requirements.disk_mb,
                "network_mbps": self.resource_requirements.network_mbps
            },
            "assigned_node": self.assigned_node,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "progress": self.progress,
            "config": self.config,
            "dependencies": self.dependencies
        }


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, strategy: SchedulingStrategy = SchedulingStrategy.LEAST_LOADED):
        self.strategy = strategy
        self.tasks: Dict[str, TrainingTask] = {}
        self.pending_tasks: List[TrainingTask] = []
        self.scheduled_tasks: Dict[str, TrainingTask] = {}
        self.running_tasks: Dict[str, TrainingTask] = {}
        self.completed_tasks: Dict[str, TrainingTask] = {}
        self.failed_tasks: Dict[str, TrainingTask] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._scheduling_task: Optional[asyncio.Task] = None
        self._scheduling_interval = int(os.getenv('SCHEDULER_INTERVAL', '5'))  # 调度间隔(秒)
        self._monitor_interval = int(os.getenv('SCHEDULER_MONITOR_INTERVAL', '10'))  # 运行态监控间隔(秒)
        self._max_health_failures = int(os.getenv('TASK_MAX_HEALTH_FAILS', '3'))  # 健康检查失败阈值
        self.node_selector: Optional[Callable[[List[NodeInfo], ResourceRequirement], Optional[NodeInfo]]] = None
    
    async def start(self):
        """启动任务调度器"""
        async with self._lock:
            if self._running:
                return
            
            self._running = True
            self._scheduling_task = asyncio.create_task(self._scheduling_loop())
            # 启动运行态任务监控循环
            self._monitoring_task = asyncio.create_task(self._monitor_running_tasks())
            logger.info("Task scheduler started")
    
    async def stop(self):
        """停止任务调度器"""
        async with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._scheduling_task:
                self._scheduling_task.cancel()
                try:
                    await self._scheduling_task
                except asyncio.CancelledError:
                    pass
            # 停止运行态任务监控循环
            if getattr(self, '_monitoring_task', None):
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            logger.info("Task scheduler stopped")
    
    async def _scheduling_loop(self):
        """调度循环"""
        while self._running:
            try:
                await self._schedule_pending_tasks()
                await asyncio.sleep(self._scheduling_interval)
            except Exception as e:
                logger.error(f"Task scheduling loop error: {e}")
                await asyncio.sleep(5)
    
    async def _schedule_pending_tasks(self):
        """调度待处理任务"""
        async with self._lock:
            # 按优先级排序待处理任务
            self.pending_tasks.sort(key=lambda x: x.priority, reverse=True)
            
            for task in self.pending_tasks[:]:  # 创建副本以避免在迭代时修改列表
                node = await self._select_node(task.resource_requirements)
                if node:
                    # 分配任务到节点
                    task.status = TaskStatus.SCHEDULED
                    task.assigned_node = node.node_id
                    task.scheduled_at = datetime.now()

                    # 尝试使用 ResourceAllocator 为该任务创建 allocation 并记录 allocation_id 到 task
                    try:
                        from backend.modules.distributed.resource_allocator import get_resource_allocator
                        from backend.modules.distributed.task_scheduler import ResourceRequirement as RR
                        allocator = get_resource_allocator()
                        req = RR(cpu_cores=task.resource_requirements.cpu_cores,
                                 memory_mb=task.resource_requirements.memory_mb,
                                 gpu_count=task.resource_requirements.gpu_count,
                                 gpu_memory_mb=task.resource_requirements.gpu_memory_mb,
                                 disk_mb=task.resource_requirements.disk_mb,
                                 network_mbps=task.resource_requirements.network_mbps)
                        # 将 node 包装成列表调用 allocate_resources
                        alloc_res = None
                        try:
                            alloc_res = await allocator.allocate_resources([node], req)
                        except Exception:
                            # 无法分配时忽略，保持已调度但无 allocation_id
                            alloc_res = None
                        if alloc_res:
                            alloc_id, alloc_obj = alloc_res
                            # 存储 allocation_id 到 task.config
                            task.config['allocation_id'] = alloc_id
                    except Exception:
                        pass

                    # 从待处理列表移到已调度列表
                    self.pending_tasks.remove(task)
                    self.scheduled_tasks[task.task_id] = task
                    
                    logger.info(f"Task {task.task_id} scheduled to node {node.node_id}")
                # 标记运行中以便监控
                self.running_tasks[task.task_id] = task
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
    
    async def _select_node(self, requirements: ResourceRequirement) -> Optional[NodeInfo]:
        """选择合适的节点
        优先使用 ResourceAllocator 进行真实分配（如成功则返回分配的节点），否则回退到 ClusterManager 的轻量优选。
        """
        try:
            # 延迟导入以避免循环依赖
            from backend.modules.distributed.cluster_manager import ClusterManager, StaticNodeDiscovery
            from backend.modules.distributed.resource_allocator import get_resource_allocator
            # 获取全局 cluster manager（简易方式：构造临时discover）
            # Note: 上层通常会传入或维护 ClusterManager；这里若无法访问则回退到简单策略
            try:
                cluster_mgr = ClusterManager(StaticNodeDiscovery([]))
            except Exception:
                cluster_mgr = None

            # 尝试使用 ResourceAllocator
            try:
                allocator = get_resource_allocator()
                nodes = []
                if cluster_mgr:
                    nodes = await cluster_mgr.get_healthy_nodes()
                # 调用 allocator.allocate_resources(nodes, requirements)
                alloc_res = await allocator.allocate_resources(nodes, requirements)
                if alloc_res:
                    alloc_id, alloc_obj = alloc_res
                    # 返回被分配节点信息（NodeInfo 不直接可得，但alloc_obj.node_id可用）
                    # 尝试查找 NodeInfo 于 nodes 列表
                    for n in nodes:
                        if getattr(n, 'node_id', None) == alloc_obj.node_id:
                            return n
                    # 否则返回None（上层仍可使用 allocation info）
            except Exception:
                pass

            # 回退：从 cluster_mgr 获取健康节点并按简单规则选择
            if cluster_mgr:
                nodes = await cluster_mgr.get_healthy_nodes()
                if not nodes:
                    return None
                # 简单选择：按 available gpus 数量降序
                nodes.sort(key=lambda n: len(getattr(n, 'available_gpus', [])), reverse=True)
                for n in nodes:
                    # 基本检查资源是否满足
                    avail_gpus = getattr(n, 'available_gpus', [])
                    if requirements.gpu_count > 0:
                        if len(avail_gpus) < requirements.gpu_count:
                            continue
                    # 内存检查
                    mem_free = getattr(n, 'memory_total', 0) - getattr(n, 'memory_used', 0)
                    if requirements.memory_mb > mem_free:
                        continue
                    return n
        except Exception:
            pass
        return None
    
    async def submit_task(self, task: TrainingTask) -> bool:
        """提交任务"""
        async with self._lock:
            if task.task_id in self.tasks:
                logger.warning(f"Task {task.task_id} already exists")
                return False
            
            self.tasks[task.task_id] = task
            self.pending_tasks.append(task)
            logger.info(f"Task submitted: {task.task_id}")
            return True
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        async with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.CANCELLED
                task.completed_at = datetime.now()
                
                # 从相应列表中移除
                if task_id in self.pending_tasks:
                    self.pending_tasks.remove(task_id)
                if task_id in self.scheduled_tasks:
                    del self.scheduled_tasks[task_id]
                if task_id in self.running_tasks:
                    del self.running_tasks[task_id]
                
                self.completed_tasks[task_id] = task
                logger.info(f"Task cancelled: {task_id}")
                return True
            return False
    
    async def get_task(self, task_id: str) -> Optional[TrainingTask]:
        """获取任务"""
        async with self._lock:
            return self.tasks.get(task_id)
    
    async def list_tasks(self, status: Optional[TaskStatus] = None) -> List[TrainingTask]:
        """列出任务"""
        async with self._lock:
            if status:
                if status == TaskStatus.PENDING:
                    return self.pending_tasks[:]
                elif status == TaskStatus.SCHEDULED:
                    return list(self.scheduled_tasks.values())
                elif status == TaskStatus.RUNNING:
                    return list(self.running_tasks.values())
                elif status == TaskStatus.COMPLETED:
                    return list(self.completed_tasks.values())
                elif status == TaskStatus.FAILED:
                    return list(self.failed_tasks.values())
                else:
                    return [task for task in self.tasks.values() if task.status == status]
            else:
                return list(self.tasks.values())
    
    async def update_task_progress(self, task_id: str, progress: float, 
                                 status: Optional[TaskStatus] = None) -> bool:
        """更新任务进度
        进度范围 0.0-1.0
        """
        async with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.progress = max(0.0, min(1.0, progress))
                
                # 清除 stale 标志：若任务有进度更新，重置 started_at 以便监控认为活跃
                try:
                    if task.started_at is None:
                        task.started_at = datetime.now()
                except Exception:
                    pass

                if status:
                    task.status = status
                    if status == TaskStatus.RUNNING and task.started_at is None:
                        task.started_at = datetime.now()
                    elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        task.completed_at = datetime.now()
                
                logger.debug(f"Task {task_id} progress updated to {progress:.2f}")
                return True
            return False
    
    async def get_task_statistics(self) -> Dict[str, int]:
        """获取任务统计信息"""
        async with self._lock:
            return {
                "total": len(self.tasks),
                "pending": len(self.pending_tasks),
                "scheduled": len(self.scheduled_tasks),
                "running": len(self.running_tasks),
                "completed": len(self.completed_tasks),
                "failed": len(self.failed_tasks),
                "cancelled": sum(1 for task in self.tasks.values() if task.status == TaskStatus.CANCELLED)
            }

    async def _monitor_running_tasks(self):
        """
        运行态任务监控：
        - 定期检查任务的租约/心跳状态
        - 当检测到租约到期或心跳异常时，将任务标记为 FAILED 并上报 FaultToleranceManager
        - 根据任务的 retry 策略尝试重新入队
        """
        try:
            from backend.modules.distributed.lease_manager import get_lease_manager
            from backend.modules.distributed.fault_tolerance import get_fault_tolerance_manager
        except Exception:
            get_lease_manager = None
            get_fault_tolerance_manager = None

        lease_mgr = None
        ft_mgr = None
        try:
            if get_lease_manager:
                lease_mgr = get_lease_manager()
        except Exception:
            lease_mgr = None
        try:
            if get_fault_tolerance_manager:
                ft_mgr = get_fault_tolerance_manager()
        except Exception:
            ft_mgr = None

        while self._running:
            try:
                await asyncio.sleep(self._monitor_interval)
                async with self._lock:
                    now = datetime.now()
                    to_fail = []
                    for task_id, task in list(self.running_tasks.items()):
                        # 检查租约（若存在）
                        lease_expired = False
                        if lease_mgr and task.config.get('allocation_id'):
                            # allocation_id 尝试作为 lease owner_id
                            lease_id = task.config.get('lease_id') or task.config.get('allocation_id')
                            try:
                                lease = asyncio.get_event_loop().run_until_complete(lease_mgr.get_lease(lease_id)) if False else None
                                # 使用同步调用替代 run loop：直接 await via helper if available
                                try:
                                    lease = None
                                    # attempt await in async context
                                    lease = await lease_mgr.get_lease(lease_id)
                                except Exception:
                                    lease = None
                                if not lease:
                                    lease_expired = True
                                else:
                                    if lease.is_expired():
                                        lease_expired = True
                            except Exception:
                                lease_expired = True

                        # 任务长时间无进度视为健康异常
                        stale_threshold = int(os.getenv('TASK_PROGRESS_STALE_SECONDS', '120'))
                        is_stale = False
                        try:
                            if task.started_at and (now - task.started_at).total_seconds() > stale_threshold and task.progress <= 0.001:
                                is_stale = True
                        except Exception:
                            is_stale = False

                        if lease_expired or is_stale:
                            to_fail.append((task_id, lease_expired, is_stale))

                    for task_id, lease_expired, is_stale in to_fail:
                        task = self.running_tasks.pop(task_id, None)
                        if not task:
                            continue
                        task.status = TaskStatus.FAILED
                        task.error_message = 'Lease expired' if lease_expired else 'Stale progress detected'
                        task.completed_at = datetime.now()
                        self.failed_tasks[task_id] = task

                        logger.warning(f"Task {task_id} failed due to {'lease_expired' if lease_expired else 'stale_progress'}")

                        # 上报容错管理器
                        try:
                            if ft_mgr:
                                await ft_mgr.report_fault(node_id=task.assigned_node or 'unknown', task_id=task.task_id, event_type='task_failure', description=task.error_message)
                        except Exception as e:
                            logger.error(f"Failed to report fault for task {task_id}: {e}")

                        # 根据重试策略尝试重新入队
                        try:
                            max_retries = int(os.getenv('TASK_MAX_RETRIES', '3'))
                            attempts = int(task.config.get('retry_attempts', 0))
                            if attempts < max_retries:
                                task.config['retry_attempts'] = attempts + 1
                                task.status = TaskStatus.PENDING
                                task.started_at = None
                                task.completed_at = None
                                task.error_message = None
                                task.progress = 0.0
                                task.assigned_node = None
                                self.pending_tasks.append(task)
                                logger.info(f"Requeued task {task_id} (attempt {attempts+1}/{max_retries})")
                            else:
                                logger.error(f"Task {task_id} reached max retries ({max_retries}) and will not be requeued")
                        except Exception as e:
                            logger.error(f"Error handling retry for task {task_id}: {e}")

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"Running tasks monitor error: {e}")
                await asyncio.sleep(5)


# 全局任务调度器实例
_task_scheduler: Optional[TaskScheduler] = None


def get_task_scheduler() -> TaskScheduler:
    """获取全局任务调度器实例"""
    global _task_scheduler
    if _task_scheduler is None:
        _task_scheduler = TaskScheduler()
    return _task_scheduler


def set_task_scheduler(task_scheduler: TaskScheduler):
    """设置全局任务调度器实例"""
    global _task_scheduler
    _task_scheduler = task_scheduler
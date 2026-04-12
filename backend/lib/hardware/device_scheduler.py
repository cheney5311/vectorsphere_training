# -*- coding: utf-8 -*-
"""
设备调度器

管理多设备环境下的资源分配和任务调度。
"""

import logging
import threading
import queue
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Tuple
from enum import Enum
from collections import defaultdict
from contextlib import contextmanager
import time

import torch
import torch.nn as nn

from .device_types import DeviceType, DeviceInfo
from .device_manager import DeviceManager, get_device_manager

logger = logging.getLogger(__name__)


class AllocationStrategy(Enum):
    """分配策略"""
    ROUND_ROBIN = "round_robin"      # 轮询
    LEAST_LOADED = "least_loaded"    # 最少负载
    MEMORY_FIRST = "memory_first"    # 内存优先
    COMPUTE_FIRST = "compute_first"  # 计算优先
    AFFINITY = "affinity"            # 亲和性


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"      # 等待中
    RUNNING = "running"      # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    CANCELLED = "cancelled"  # 已取消


@dataclass
class DeviceAllocation:
    """设备分配"""
    task_id: str
    device: torch.device
    allocated_memory: int = 0
    start_time: float = field(default_factory=time.time)
    estimated_duration: Optional[float] = None
    priority: TaskPriority = TaskPriority.NORMAL
    
    # 新增：状态和统计
    status: TaskStatus = TaskStatus.PENDING
    end_time: Optional[float] = None
    actual_memory: int = 0
    error: Optional[str] = None
    
    @property
    def elapsed_time(self) -> float:
        """已用时间"""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def is_active(self) -> bool:
        """是否活跃"""
        return self.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
    
    @property
    def is_completed(self) -> bool:
        """是否完成"""
        return self.status == TaskStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'device': str(self.device),
            'allocated_memory': self.allocated_memory,
            'actual_memory': self.actual_memory,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'elapsed_time': self.elapsed_time,
            'estimated_duration': self.estimated_duration,
            'priority': self.priority.name,
            'status': self.status.value,
            'error': self.error,
        }
    
    def mark_running(self) -> None:
        """标记为运行中"""
        self.status = TaskStatus.RUNNING
    
    def mark_completed(self, actual_memory: Optional[int] = None) -> None:
        """标记为完成"""
        self.status = TaskStatus.COMPLETED
        self.end_time = time.time()
        if actual_memory is not None:
            self.actual_memory = actual_memory
    
    def mark_failed(self, error: str) -> None:
        """标记为失败"""
        self.status = TaskStatus.FAILED
        self.end_time = time.time()
        self.error = error
    
    def mark_cancelled(self) -> None:
        """标记为取消"""
        self.status = TaskStatus.CANCELLED
        self.end_time = time.time()


@dataclass
class SchedulerTask:
    """调度任务"""
    task_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    required_memory: int = 0  # 所需内存（字节）
    device_preference: Optional[torch.device] = None
    
    # 新增：元数据和控制
    created_time: float = field(default_factory=time.time)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 0
    callback: Optional[Callable] = None
    
    @property
    def age(self) -> float:
        """任务年龄（等待时间）"""
        return time.time() - self.created_time
    
    @property
    def can_retry(self) -> bool:
        """是否可以重试"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """增加重试计数"""
        self.retry_count += 1


@dataclass
class SchedulerStats:
    """调度器统计"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    
    total_execution_time: float = 0.0
    total_wait_time: float = 0.0
    
    avg_execution_time: float = 0.0
    avg_wait_time: float = 0.0
    
    peak_concurrent_tasks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_tasks,
            'failed_tasks': self.failed_tasks,
            'cancelled_tasks': self.cancelled_tasks,
            'success_rate': self.completed_tasks / self.total_tasks if self.total_tasks > 0 else 0.0,
            'total_execution_time': self.total_execution_time,
            'total_wait_time': self.total_wait_time,
            'avg_execution_time': self.avg_execution_time,
            'avg_wait_time': self.avg_wait_time,
            'peak_concurrent_tasks': self.peak_concurrent_tasks,
        }


class SchedulerMonitor:
    """调度器监控"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._task_history: List[DeviceAllocation] = []
        self._stats = SchedulerStats()
        self._lock = threading.Lock()
    
    def record_task(self, allocation: DeviceAllocation) -> None:
        """记录任务"""
        with self._lock:
            self._task_history.append(allocation)
            if len(self._task_history) > self.max_history:
                self._task_history.pop(0)
            
            # 更新统计
            self._update_stats(allocation)
    
    def _update_stats(self, allocation: DeviceAllocation) -> None:
        """更新统计"""
        self._stats.total_tasks += 1
        
        if allocation.status == TaskStatus.COMPLETED:
            self._stats.completed_tasks += 1
            self._stats.total_execution_time += allocation.elapsed_time
        elif allocation.status == TaskStatus.FAILED:
            self._stats.failed_tasks += 1
        elif allocation.status == TaskStatus.CANCELLED:
            self._stats.cancelled_tasks += 1
        
        # 更新平均值
        if self._stats.completed_tasks > 0:
            self._stats.avg_execution_time = (
                self._stats.total_execution_time / self._stats.completed_tasks
            )
    
    def update_concurrent_tasks(self, count: int) -> None:
        """更新并发任务数"""
        with self._lock:
            if count > self._stats.peak_concurrent_tasks:
                self._stats.peak_concurrent_tasks = count
    
    def get_stats(self) -> SchedulerStats:
        """获取统计信息"""
        with self._lock:
            return self._stats
    
    def get_recent_tasks(self, n: int = 10) -> List[DeviceAllocation]:
        """获取最近的任务"""
        with self._lock:
            return self._task_history[-n:].copy()
    
    def get_task_by_status(self, status: TaskStatus) -> List[DeviceAllocation]:
        """按状态获取任务"""
        with self._lock:
            return [a for a in self._task_history if a.status == status]
    
    def get_device_usage(self) -> Dict[str, int]:
        """获取设备使用统计"""
        with self._lock:
            usage = defaultdict(int)
            for allocation in self._task_history:
                usage[str(allocation.device)] += 1
            return dict(usage)
    
    def reset(self) -> None:
        """重置监控"""
        with self._lock:
            self._task_history.clear()
            self._stats = SchedulerStats()


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self._device_loads: Dict[str, float] = defaultdict(float)
        self._device_scores: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
    
    def update_load(self, device: torch.device, load: float) -> None:
        """更新设备负载"""
        with self._lock:
            self._device_loads[str(device)] = load
    
    def get_load(self, device: torch.device) -> float:
        """获取设备负载"""
        with self._lock:
            return self._device_loads.get(str(device), 0.0)
    
    def get_all_loads(self) -> Dict[str, float]:
        """获取所有设备负载"""
        with self._lock:
            return self._device_loads.copy()
    
    def calculate_score(
        self,
        device: torch.device,
        available_memory: int,
        task_count: int,
        device_type: str
    ) -> float:
        """
        计算设备得分
        
        综合考虑负载、内存、任务数等因素
        """
        # 基础分数
        score = 100.0
        
        # 负载惩罚
        load = self.get_load(device)
        score -= load * 50
        
        # 任务数惩罚
        score -= task_count * 10
        
        # 内存奖励
        memory_gb = available_memory / (1024**3)
        score += min(memory_gb, 10) * 5
        
        # GPU优先
        if device_type == 'cuda':
            score += 20
        
        with self._lock:
            self._device_scores[str(device)] = score
        
        return score
    
    def get_best_device(self, devices: List[torch.device]) -> Optional[torch.device]:
        """获取最佳设备"""
        with self._lock:
            if not devices:
                return None
            
            best = max(devices, key=lambda d: self._device_scores.get(str(d), 0.0))
            return best
    
    def reset(self) -> None:
        """重置负载均衡器"""
        with self._lock:
            self._device_loads.clear()
            self._device_scores.clear()


class DevicePool:
    """
    设备池
    
    管理一组设备的状态和分配。
    """
    
    def __init__(self, devices: Optional[List[torch.device]] = None):
        if devices is None:
            # 自动检测所有GPU
            if torch.cuda.is_available():
                devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
            else:
                devices = [torch.device('cpu')]
        
        self.devices = devices
        self._allocations: Dict[str, DeviceAllocation] = {}
        self._device_tasks: Dict[str, List[str]] = {str(d): [] for d in devices}
        self._lock = threading.Lock()
        
        # 新增：监控和负载均衡
        self._monitor = SchedulerMonitor()
        self._load_balancer = LoadBalancer()
        self._device_affinities: Dict[str, List[str]] = defaultdict(list)  # 任务亲和性
        
        logger.info(f"DevicePool initialized with {len(devices)} devices")
    
    def allocate(
        self, 
        task_id: str,
        required_memory: int = 0,
        strategy: AllocationStrategy = AllocationStrategy.LEAST_LOADED,
        device_preference: Optional[torch.device] = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> Optional[torch.device]:
        """
        分配设备
        
        Args:
            task_id: 任务ID
            required_memory: 所需内存
            strategy: 分配策略
            device_preference: 偏好设备
            priority: 任务优先级
        """
        with self._lock:
            # 如果指定了偏好设备且可用
            if device_preference and self._can_allocate(device_preference, required_memory):
                return self._do_allocate(task_id, device_preference, required_memory, priority)
            
            # 根据策略选择设备
            device = self._select_device(required_memory, strategy, task_id)
            
            if device is not None:
                return self._do_allocate(task_id, device, required_memory, priority)
            
            logger.warning(f"No device available for task {task_id}")
            return None
    
    def _can_allocate(self, device: torch.device, required_memory: int) -> bool:
        """检查设备是否可分配"""
        if device.type == 'cpu':
            return True  # CPU总是可用
        
        try:
            props = torch.cuda.get_device_properties(device.index)
            reserved = torch.cuda.memory_reserved(device.index)
            available = props.total_memory - reserved
            return available >= required_memory
        except Exception:
            return False
    
    def _select_device(
        self, 
        required_memory: int,
        strategy: AllocationStrategy,
        task_id: Optional[str] = None
    ) -> Optional[torch.device]:
        """根据策略选择设备"""
        candidates = []
        
        for device in self.devices:
            if self._can_allocate(device, required_memory):
                load = len(self._device_tasks[str(device)])
                memory = self._get_available_memory(device)
                
                # 计算负载均衡得分
                score = self._load_balancer.calculate_score(
                    device, memory, load, device.type
                )
                
                candidates.append((device, load, memory, score))
        
        if not candidates:
            return None
        
        # 根据策略选择
        if strategy == AllocationStrategy.ROUND_ROBIN:
            # 选择任务最少的设备
            candidates.sort(key=lambda x: x[1])
        elif strategy == AllocationStrategy.LEAST_LOADED:
            # 选择负载最低的设备（使用负载均衡得分）
            candidates.sort(key=lambda x: -x[3])  # 得分越高越好
        elif strategy == AllocationStrategy.MEMORY_FIRST:
            # 选择内存最多的设备
            candidates.sort(key=lambda x: -x[2])
        elif strategy == AllocationStrategy.COMPUTE_FIRST:
            # 优先GPU，然后考虑负载
            candidates.sort(key=lambda x: (0 if x[0].type == 'cuda' else 1, x[1]))
        elif strategy == AllocationStrategy.AFFINITY:
            # 检查亲和性
            if task_id:
                for device_str, task_list in self._device_affinities.items():
                    if task_id in task_list:
                        # 找到亲和设备
                        affinity_device = torch.device(device_str)
                        for c in candidates:
                            if c[0] == affinity_device:
                                return c[0]
            # 没有亲和性，使用负载均衡
            candidates.sort(key=lambda x: -x[3])
        
        return candidates[0][0]
    
    def _get_available_memory(self, device: torch.device) -> int:
        """获取可用内存"""
        if device.type == 'cpu':
            try:
                import psutil
                return psutil.virtual_memory().available
            except ImportError:
                return 0
        else:
            try:
                props = torch.cuda.get_device_properties(device.index)
                reserved = torch.cuda.memory_reserved(device.index)
                return props.total_memory - reserved
            except Exception:
                return 0
    
    def _do_allocate(
        self, 
        task_id: str, 
        device: torch.device,
        required_memory: int,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> torch.device:
        """执行分配"""
        allocation = DeviceAllocation(
            task_id=task_id,
            device=device,
            allocated_memory=required_memory,
            priority=priority,
            status=TaskStatus.PENDING
        )
        
        self._allocations[task_id] = allocation
        self._device_tasks[str(device)].append(task_id)
        
        # 更新负载
        load = len(self._device_tasks[str(device)]) / max(len(self.devices), 1)
        self._load_balancer.update_load(device, load)
        
        logger.debug(f"Allocated device {device} for task {task_id}")
        return device
    
    def release(self, task_id: str, status: TaskStatus = TaskStatus.COMPLETED, 
                error: Optional[str] = None):
        """释放设备"""
        with self._lock:
            if task_id not in self._allocations:
                return
            
            allocation = self._allocations.pop(task_id)
            device_key = str(allocation.device)
            
            if task_id in self._device_tasks[device_key]:
                self._device_tasks[device_key].remove(task_id)
            
            # 更新分配状态
            if status == TaskStatus.COMPLETED:
                allocation.mark_completed()
            elif status == TaskStatus.FAILED:
                allocation.mark_failed(error or "Unknown error")
            elif status == TaskStatus.CANCELLED:
                allocation.mark_cancelled()
            
            # 记录到监控
            self._monitor.record_task(allocation)
            
            # 更新负载
            load = len(self._device_tasks[device_key]) / max(len(self.devices), 1)
            self._load_balancer.update_load(allocation.device, load)
            
            logger.debug(f"Released device {allocation.device} from task {task_id}, status: {status.value}")
    
    def get_allocation(self, task_id: str) -> Optional[DeviceAllocation]:
        """获取分配信息"""
        return self._allocations.get(task_id)
    
    def get_device_tasks(self, device: torch.device) -> List[str]:
        """获取设备上的任务"""
        return self._device_tasks.get(str(device), []).copy()
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        status = {
            'devices': [],
            'total_tasks': len(self._allocations),
            'stats': self._monitor.get_stats().to_dict()
        }
        
        for device in self.devices:
            device_key = str(device)
            tasks = self._device_tasks[device_key]
            
            device_status = {
                'device': device_key,
                'tasks': len(tasks),
                'available_memory': self._get_available_memory(device),
                'load': self._load_balancer.get_load(device)
            }
            status['devices'].append(device_status)
        
        return status
    
    # ==================== 新增内部方法 ====================
    
    def _mark_task_running(self, task_id: str) -> None:
        """标记任务为运行中"""
        if task_id in self._allocations:
            self._allocations[task_id].mark_running()
    
    def _set_affinity(self, task_id: str, device: torch.device) -> None:
        """设置任务亲和性"""
        device_key = str(device)
        if task_id not in self._device_affinities[device_key]:
            self._device_affinities[device_key].append(task_id)
    
    def _get_allocation_by_id(self, task_id: str) -> Optional[DeviceAllocation]:
        """通过ID获取分配"""
        return self._allocations.get(task_id)
    
    def _get_monitor(self) -> SchedulerMonitor:
        """获取监控器"""
        return self._monitor
    
    def _get_load_balancer(self) -> LoadBalancer:
        """获取负载均衡器"""
        return self._load_balancer
    
    def _update_concurrent_count(self) -> None:
        """更新并发任务计数"""
        count = len(self._allocations)
        self._monitor.update_concurrent_tasks(count)
    
    def _get_device_usage_stats(self) -> Dict[str, Any]:
        """获取设备使用统计"""
        return {
            'device_usage': self._monitor.get_device_usage(),
            'device_loads': self._load_balancer.get_all_loads(),
        }


class DeviceScheduler:
    """
    设备调度器
    
    管理任务的设备分配和执行调度。
    """
    
    def __init__(
        self, 
        devices: Optional[List[torch.device]] = None,
        strategy: AllocationStrategy = AllocationStrategy.LEAST_LOADED,
        max_concurrent_per_device: int = 1
    ):
        self.pool = DevicePool(devices)
        self.strategy = strategy
        self.max_concurrent_per_device = max_concurrent_per_device
        
        # 任务队列
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._running_tasks: Dict[str, threading.Thread] = {}
        
        # 控制
        self._shutdown = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        
        # 新增：任务管理和状态跟踪
        self._task_results: Dict[str, Any] = {}
        self._task_lock = threading.Lock()
        self._cancelled_tasks: set = set()
        self._task_timeouts: Dict[str, float] = {}
        
        # 新增：重试机制
        self._retry_queue: queue.Queue = queue.Queue()
        self._pending_tasks: Dict[str, SchedulerTask] = {}
    
    def submit(
        self, 
        task_id: str,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        required_memory: int = 0,
        device_preference: Optional[torch.device] = None,
        **kwargs
    ) -> bool:
        """
        提交任务
        
        Args:
            task_id: 任务ID
            func: 要执行的函数
            *args: 位置参数
            priority: 优先级
            required_memory: 所需内存
            device_preference: 偏好设备
            **kwargs: 关键字参数
        """
        # 从kwargs中提取调度器特定参数
        timeout = kwargs.pop('timeout', None)
        max_retries = kwargs.pop('max_retries', 0)
        callback = kwargs.pop('callback', None)
        
        task = SchedulerTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            required_memory=required_memory,
            device_preference=device_preference,
            timeout=timeout,
            max_retries=max_retries,
            callback=callback
        )
        
        # 存储待处理任务
        with self._task_lock:
            self._pending_tasks[task_id] = task
            if timeout:
                self._task_timeouts[task_id] = time.time() + timeout
        
        # 优先级队列（负数使高优先级先出队）
        self._task_queue.put((-priority.value, time.time(), task))
        
        logger.info(f"Task {task_id} submitted with priority {priority.name}")
        return True
    
    def start(self):
        """启动调度器"""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        
        self._shutdown.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        logger.info("DeviceScheduler started")
    
    def stop(self, wait: bool = True):
        """停止调度器"""
        self._shutdown.set()
        
        if wait and self._worker_thread is not None:
            self._worker_thread.join(timeout=10)
        
        logger.info("DeviceScheduler stopped")
    
    def _worker_loop(self):
        """工作循环"""
        while not self._shutdown.is_set():
            # 处理重试队列
            self._process_retry_queue()
            
            # 检查超时任务
            self._check_timeouts()
            
            try:
                # 非阻塞获取任务
                priority, timestamp, task = self._task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # 检查任务是否被取消
            if task.task_id in self._cancelled_tasks:
                self._cancelled_tasks.remove(task.task_id)
                self.pool.release(task.task_id, TaskStatus.CANCELLED)
                continue
            
            # 尝试分配设备
            device = self.pool.allocate(
                task.task_id,
                task.required_memory,
                self.strategy,
                task.device_preference,
                task.priority
            )
            
            if device is None:
                # 放回队列稍后重试
                self._task_queue.put((priority, timestamp, task))
                time.sleep(0.1)
                continue
            
            # 更新并发任务计数
            self.pool._update_concurrent_count()
            
            # 执行任务
            thread = threading.Thread(
                target=self._execute_task,
                args=(task, device),
                daemon=True
            )
            self._running_tasks[task.task_id] = thread
            thread.start()
    
    def _execute_task(self, task: SchedulerTask, device: torch.device):
        """执行任务"""
        # 标记任务为运行中
        self.pool._mark_task_running(task.task_id)
        
        # 从待处理队列移除
        with self._task_lock:
            self._pending_tasks.pop(task.task_id, None)
        
        result = None
        error = None
        
        try:
            # 设置设备
            if device.type == 'cuda':
                torch.cuda.set_device(device)
            
            # 添加设备到kwargs
            kwargs = {**task.kwargs, 'device': device}
            
            logger.info(f"Executing task {task.task_id} on {device}")
            
            # 执行任务
            result = task.func(*task.args, **kwargs)
            
            # 存储结果
            with self._task_lock:
                self._task_results[task.task_id] = result
            
            # 调用回调
            if task.callback:
                try:
                    task.callback(result)
                except Exception as e:
                    logger.error(f"Callback failed for task {task.task_id}: {e}")
            
            logger.info(f"Task {task.task_id} completed")
            
            # 释放设备（成功）
            self.pool.release(task.task_id, TaskStatus.COMPLETED)
            
        except Exception as e:
            error = str(e)
            logger.error(f"Task {task.task_id} failed: {e}")
            
            # 检查是否可以重试
            if task.can_retry:
                logger.info(f"Retrying task {task.task_id} ({task.retry_count + 1}/{task.max_retries})")
                task.increment_retry()
                self._retry_queue.put(task)
                # 释放设备但不标记为失败
                self.pool.release(task.task_id, TaskStatus.PENDING)
            else:
                # 释放设备（失败）
                self.pool.release(task.task_id, TaskStatus.FAILED, error)
        
        finally:
            # 清理
            self._running_tasks.pop(task.task_id, None)
            with self._task_lock:
                self._task_timeouts.pop(task.task_id, None)
    
    # ==================== 新增内部方法 ====================
    
    def _process_retry_queue(self) -> None:
        """处理重试队列"""
        try:
            while not self._retry_queue.empty():
                task = self._retry_queue.get_nowait()
                # 重新提交到主队列
                self._task_queue.put((-task.priority.value, time.time(), task))
        except queue.Empty:
            pass
    
    def _check_timeouts(self) -> None:
        """检查超时任务"""
        current_time = time.time()
        
        with self._task_lock:
            timeout_tasks = []
            for task_id, timeout_time in self._task_timeouts.items():
                if current_time > timeout_time:
                    timeout_tasks.append(task_id)
            
            # 取消超时任务
            for task_id in timeout_tasks:
                if task_id in self._running_tasks:
                    logger.warning(f"Task {task_id} timed out")
                    self._cancelled_tasks.add(task_id)
                    self._task_timeouts.pop(task_id, None)
    
    def _get_task_result(self, task_id: str) -> Optional[Any]:
        """获取任务结果"""
        with self._task_lock:
            return self._task_results.get(task_id)
    
    def _clear_task_result(self, task_id: str) -> None:
        """清除任务结果"""
        with self._task_lock:
            self._task_results.pop(task_id, None)
    
    def _is_task_running(self, task_id: str) -> bool:
        """检查任务是否运行中"""
        return task_id in self._running_tasks
    
    def _get_pending_task_count(self) -> int:
        """获取待处理任务数"""
        return self._task_queue.qsize()
    
    def _get_running_task_count(self) -> int:
        """获取运行中任务数"""
        return len(self._running_tasks)
    
    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        pool_status = self.pool.get_status()
        
        with self._task_lock:
            status = {
                'pool': pool_status,
            'queue_size': self._task_queue.qsize(),
            'running_tasks': list(self._running_tasks.keys()),
                'pending_tasks': len(self._pending_tasks),
                'completed_tasks': len(self._task_results),
                'strategy': self.strategy.value,
                'cancelled_tasks': len(self._cancelled_tasks),
                'max_concurrent_per_device': self.max_concurrent_per_device,
            }
        
        return status
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> bool:
        """
        等待任务完成
        
        Returns:
            是否成功完成
        """
        thread = self._running_tasks.get(task_id)
        if thread is None:
            return True  # 已完成或不存在
        
        thread.join(timeout=timeout)
        return not thread.is_alive()
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务（仅限未开始的任务）"""
        with self._task_lock:
            # 如果任务在待处理队列中
            if task_id in self._pending_tasks:
                self._cancelled_tasks.add(task_id)
                self._pending_tasks.pop(task_id, None)
                logger.info(f"Task {task_id} marked for cancellation")
                return True
            
            # 如果任务正在运行
            if task_id in self._running_tasks:
                logger.warning(f"Cannot cancel running task {task_id}")
                return False
            
            logger.warning(f"Task {task_id} not found for cancellation")
        return False


# ==================== 便捷函数 ====================

_scheduler: Optional[DeviceScheduler] = None


def get_device_scheduler() -> DeviceScheduler:
    """获取全局设备调度器"""
    global _scheduler
    if _scheduler is None:
        _scheduler = DeviceScheduler()
        _scheduler.start()
    return _scheduler


def submit_task(
    task_id: str,
    func: Callable,
    *args,
    **kwargs
) -> bool:
    """提交任务到调度器"""
    scheduler = get_device_scheduler()
    return scheduler.submit(task_id, func, *args, **kwargs)


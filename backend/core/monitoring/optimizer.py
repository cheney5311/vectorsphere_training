"""统一资源优化器

提供智能的GPU调度、内存管理、存储优化等功能。
"""

import logging
import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

import numpy as np

from .models import SystemMetrics, GPUMetrics
from .exceptions import MonitoringError

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """优化策略枚举"""
    BALANCED = "balanced"  # 平衡策略
    PERFORMANCE_FIRST = "performance_first"  # 性能优先
    COST_EFFICIENT = "cost_efficient"  # 成本效率优先
    ENERGY_SAVING = "energy_saving"  # 节能模式


class ResourceState(Enum):
    """资源状态枚举"""
    IDLE = "idle"  # 空闲
    BUSY = "busy"  # 忙碌
    OVERLOADED = "overloaded"  # 过载
    MAINTENANCE = "maintenance"  # 维护中
    FAILED = "failed"  # 故障
    RESERVED = "reserved"  # 预留


@dataclass
class OptimizationRecommendation:
    """优化建议"""
    recommendation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)

    # 建议类型
    category: str = ""  # 类别：cpu, memory, gpu, storage, network
    priority: int = 1  # 优先级 1-5，5最高
    confidence: float = 0.0  # 置信度 0-1

    # 建议内容
    title: str = ""
    description: str = ""
    action: str = ""  # 具体操作

    # 预期效果
    expected_improvement: Dict[str, float] = field(default_factory=dict)
    estimated_cost: float = 0.0  # 预估成本
    implementation_effort: str = "low"  # low, medium, high

    # 相关资源
    affected_resources: List[str] = field(default_factory=list)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


class ResourcePredictor:
    """资源需求预测器"""

    def __init__(self, history_window_hours: int = 24):
        self.history_window_hours = history_window_hours
        self.metrics_history: deque = deque(maxlen=1000)  # 保留最近1000个数据点

    def add_metrics(self, metrics: SystemMetrics):
        """添加指标数据"""
        self.metrics_history.append(metrics)

    def predict_cpu_demand(self, horizon_minutes: int = 30) -> float:
        """预测CPU需求"""
        if len(self.metrics_history) < 10:
            return 0.5  # 默认值

        # 简单的移动平均预测
        recent_metrics = list(self.metrics_history)[-20:]  # 最近20个数据点
        avg_utilization = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)

        # 考虑趋势
        if len(recent_metrics) >= 10:
            first_half = recent_metrics[:10]
            second_half = recent_metrics[10:]

            first_avg = sum(m.cpu_percent for m in first_half) / len(first_half)
            second_avg = sum(m.cpu_percent for m in second_half) / len(second_half)

            trend = second_avg - first_avg
            predicted = avg_utilization + trend * (horizon_minutes / 30)

            return max(0.0, min(1.0, predicted))

        return avg_utilization

    def predict_memory_demand(self, horizon_minutes: int = 30) -> float:
        """预测内存需求"""
        if len(self.metrics_history) < 10:
            return 0.5

        recent_metrics = list(self.metrics_history)[-20:]
        avg_utilization = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)

        # 内存使用通常比较稳定，但需要考虑峰值
        max_utilization = max(m.memory_percent for m in recent_metrics)

        # 预测值在平均值和最大值之间
        predicted = avg_utilization * 0.7 + max_utilization * 0.3

        return max(0.0, min(1.0, predicted))

    def predict_gpu_demand(self, gpu_id: int, horizon_minutes: int = 30) -> float:
        """预测GPU需求"""
        # 简化实现，实际应用中需要更复杂的预测算法
        return 0.5


class GPUScheduler:
    """GPU智能调度器"""

    def __init__(self):
        self.gpu_allocations: Dict[int, List[str]] = defaultdict(list)  # GPU ID -> 任务ID列表
        self.task_gpu_mapping: Dict[str, List[int]] = {}  # 任务ID -> GPU ID列表
        self.gpu_metrics: Dict[int, GPUMetrics] = {}

    def allocate_gpus(self, task_id: str, required_gpus: int,
                      strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> List[int]:
        """分配GPU资源"""
        available_gpus = self._get_available_gpus()

        if len(available_gpus) < required_gpus:
            logger.warning(f"Not enough GPUs available. Required: {required_gpus}, Available: {len(available_gpus)}")
            return []

        # 根据策略选择GPU
        if strategy == OptimizationStrategy.PERFORMANCE_FIRST:
            selected_gpus = self._select_gpus_performance(available_gpus, required_gpus)
        elif strategy == OptimizationStrategy.COST_EFFICIENT:
            selected_gpus = self._select_gpus_cost_efficient(available_gpus, required_gpus)
        elif strategy == OptimizationStrategy.ENERGY_SAVING:
            selected_gpus = self._select_gpus_energy_saving(available_gpus, required_gpus)
        else:  # BALANCED or default
            selected_gpus = self._select_gpus_balanced(available_gpus, required_gpus)

        # 记录分配
        for gpu_id in selected_gpus:
            self.gpu_allocations[gpu_id].append(task_id)

        self.task_gpu_mapping[task_id] = selected_gpus

        logger.info(f"Allocated GPUs {selected_gpus} to task {task_id}")
        return selected_gpus

    def release_gpus(self, task_id: str) -> bool:
        """释放GPU资源"""
        if task_id not in self.task_gpu_mapping:
            return False

        gpu_ids = self.task_gpu_mapping[task_id]
        for gpu_id in gpu_ids:
            if task_id in self.gpu_allocations[gpu_id]:
                self.gpu_allocations[gpu_id].remove(task_id)

        del self.task_gpu_mapping[task_id]
        logger.info(f"Released GPUs {gpu_ids} from task {task_id}")
        return True

    def _get_available_gpus(self) -> List[int]:
        """获取可用GPU列表"""
        # 简化实现，实际应用中需要检查GPU状态
        return list(range(8))  # 假设有8个GPU

    def _select_gpus_performance(self, available_gpus: List[int], required_gpus: int) -> List[int]:
        """性能优先选择GPU"""
        # 选择性能最好的GPU
        return available_gpus[:required_gpus]

    def _select_gpus_cost_efficient(self, available_gpus: List[int], required_gpus: int) -> List[int]:
        """成本效率优先选择GPU"""
        # 选择成本最低的GPU
        return available_gpus[:required_gpus]

    def _select_gpus_energy_saving(self, available_gpus: List[int], required_gpus: int) -> List[int]:
        """节能模式选择GPU"""
        # 选择能耗最低的GPU
        return available_gpus[:required_gpus]

    def _select_gpus_balanced(self, available_gpus: List[int], required_gpus: int) -> List[int]:
        """平衡策略选择GPU"""
        # 选择负载最均衡的GPU
        gpu_loads = []
        for gpu_id in available_gpus:
            load = len(self.gpu_allocations.get(gpu_id, []))
            gpu_loads.append((gpu_id, load))

        # 按负载排序，选择负载最低的GPU
        gpu_loads.sort(key=lambda x: x[1])
        return [gpu_id for gpu_id, _ in gpu_loads[:required_gpus]]


class ResourceOptimizer:
    """资源优化器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化资源优化器

        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.predictor = ResourcePredictor()
        self.gpu_scheduler = GPUScheduler()

        # 优化历史
        self.optimization_history: List[Dict[str, Any]] = []

    def generate_recommendations(self, metrics: SystemMetrics,
                                 gpu_metrics: Optional[List[GPUMetrics]] = None) -> List[OptimizationRecommendation]:
        """生成优化建议

        Args:
            metrics: 系统指标
            gpu_metrics: GPU指标列表

        Returns:
            List[OptimizationRecommendation]: 优化建议列表
        """
        recommendations = []

        # 1. 基于资源预测的建议
        cpu_demand = self.predictor.predict_cpu_demand()
        memory_demand = self.predictor.predict_memory_demand()

        if cpu_demand > 0.8:
            recommendations.append(OptimizationRecommendation(
                category="cpu",
                priority=3,
                confidence=0.8,
                title="CPU资源不足预警",
                description=f"预测CPU需求将达到{cpu_demand:.1%}，建议增加CPU资源或优化任务分配",
                action="scale_up_cpu",
                expected_improvement={"performance": 0.2},
                implementation_effort="medium"
            ))

        if memory_demand > 0.85:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority=3,
                confidence=0.85,
                title="内存资源不足预警",
                description=f"预测内存需求将达到{memory_demand:.1%}，建议增加内存资源或优化内存使用",
                action="scale_up_memory",
                expected_improvement={"performance": 0.15},
                implementation_effort="medium"
            ))

        # 2. 基于GPU使用情况的建议
        # 这里可以添加更多基于GPU使用情况的建议

        return recommendations

    def add_metrics(self, metrics: SystemMetrics):
        """添加指标数据

        Args:
            metrics: 系统指标数据
        """
        self.predictor.add_metrics(metrics)

    def get_gpu_scheduler(self) -> GPUScheduler:
        """获取GPU调度器

        Returns:
            GPUScheduler: GPU调度器实例
        """
        return self.gpu_scheduler


# 全局资源优化器实例
_global_optimizer: Optional[ResourceOptimizer] = None


def get_resource_optimizer() -> ResourceOptimizer:
    """获取全局资源优化器实例

    Returns:
        ResourceOptimizer: 资源优化器实例
    """
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = ResourceOptimizer()
    return _global_optimizer
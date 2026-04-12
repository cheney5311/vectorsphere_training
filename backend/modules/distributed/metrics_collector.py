"""指标收集器

收集分布式训练的性能指标。
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
import logging

from .cluster_manager import NodeInfo, GPUInfo
from .task_scheduler import TrainingTask, TaskStatus

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """指标类型枚举"""
    SYSTEM = "system"          # 系统指标
    TRAINING = "training"      # 训练指标
    NETWORK = "network"        # 网络指标
    GPU = "gpu"                # GPU指标
    MEMORY = "memory"          # 内存指标
    DISK = "disk"              # 磁盘指标
    CUSTOM = "custom"          # 自定义指标


class AggregationType(Enum):
    """聚合类型枚举"""
    AVERAGE = "average"        # 平均值
    SUM = "sum"                # 求和
    MAX = "max"                # 最大值
    MIN = "min"                # 最小值
    COUNT = "count"            # 计数


@dataclass
class MetricData:
    """指标数据"""
    metric_name: str
    metric_type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    node_id: Optional[str] = None
    task_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    description: str = ""


@dataclass
class MetricAggregation:
    """指标聚合"""
    metric_name: str
    metric_type: MetricType
    aggregation_type: AggregationType
    value: float
    count: int = 1
    min_value: float = float('inf')
    max_value: float = float('-inf')
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, collection_interval: int = 10):
        self.collection_interval = collection_interval
        self.metrics: List[MetricData] = []
        self.aggregations: Dict[str, MetricAggregation] = {}
        self.metric_handlers: Dict[MetricType, Callable] = {}
        self._lock = asyncio.Lock()
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False
        self._max_metrics_buffer = 10000  # 最大指标缓冲区大小
    
    async def start(self):
        """启动指标收集器"""
        async with self._lock:
            if self._running:
                return
            
            self._running = True
            self._collection_task = asyncio.create_task(self._collection_loop())
            logger.info("Metrics collector started")
    
    async def stop(self):
        """停止指标收集器"""
        async with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._collection_task:
                self._collection_task.cancel()
                try:
                    await self._collection_task
                except asyncio.CancelledError:
                    pass
            logger.info("Metrics collector stopped")
    
    async def _collection_loop(self):
        """收集循环"""
        while self._running:
            try:
                # 收集系统指标
                await self._collect_system_metrics()
                
                # 收集训练指标
                await self._collect_training_metrics()
                
                # 执行自定义指标处理器
                await self._execute_custom_handlers()
                
                # 清理旧指标数据
                await self._cleanup_old_metrics()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        # 这里应该实现系统指标收集逻辑
        # 为简化起见，记录模拟指标
        logger.debug("Collecting system metrics")
    
    async def _collect_training_metrics(self):
        """收集训练指标"""
        # 这里应该实现训练指标收集逻辑
        # 为简化起见，记录模拟指标
        logger.debug("Collecting training metrics")
    
    async def _execute_custom_handlers(self):
        """执行自定义指标处理器"""
        for metric_type, handler in self.metric_handlers.items():
            try:
                await handler()
            except Exception as e:
                logger.error(f"Custom metric handler for {metric_type.value} failed: {e}")
    
    async def _cleanup_old_metrics(self):
        """清理旧指标数据"""
        async with self._lock:
            # 保留最近1小时的指标数据
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.metrics = [metric for metric in self.metrics if metric.timestamp > cutoff_time]
    
    async def record_metric(self, metric_name: str, metric_type: MetricType, value: float,
                          node_id: Optional[str] = None, task_id: Optional[str] = None,
                          tags: Optional[Dict[str, str]] = None, unit: str = "", 
                          description: str = "") -> bool:
        """记录指标"""
        async with self._lock:
            # 创建指标数据
            metric_data = MetricData(
                metric_name=metric_name,
                metric_type=metric_type,
                value=value,
                node_id=node_id,
                task_id=task_id,
                tags=tags or {},
                unit=unit,
                description=description
            )
            
            # 添加到指标列表
            self.metrics.append(metric_data)
            
            # 维护缓冲区大小
            if len(self.metrics) > self._max_metrics_buffer:
                # 移除最旧的指标
                self.metrics = self.metrics[-self._max_metrics_buffer:]
            
            # 更新聚合数据
            await self._update_aggregation(metric_data)
            
            logger.debug(f"Metric recorded: {metric_name} = {value}")
            return True
    
    async def _update_aggregation(self, metric_data: MetricData):
        """更新聚合数据"""
        # 创建聚合键
        agg_key = f"{metric_data.metric_name}_{metric_data.metric_type.value}"
        
        if agg_key in self.aggregations:
            # 更新现有聚合
            agg = self.aggregations[agg_key]
            agg.value += metric_data.value
            agg.count += 1
            agg.min_value = min(agg.min_value, metric_data.value)
            agg.max_value = max(agg.max_value, metric_data.value)
            agg.timestamp = metric_data.timestamp
        else:
            # 创建新聚合
            self.aggregations[agg_key] = MetricAggregation(
                metric_name=metric_data.metric_name,
                metric_type=metric_data.metric_type,
                aggregation_type=AggregationType.AVERAGE,
                value=metric_data.value,
                count=1,
                min_value=metric_data.value,
                max_value=metric_data.value,
                tags=metric_data.tags
            )
    
    async def record_system_metrics(self, node: NodeInfo) -> bool:
        """记录系统指标"""
        try:
            # 记录CPU指标
            await self.record_metric(
                metric_name="cpu_utilization",
                metric_type=MetricType.SYSTEM,
                value=node.cpu_utilization,
                node_id=node.node_id,
                unit="%",
                description="CPU utilization"
            )
            
            # 记录内存指标
            await self.record_metric(
                metric_name="memory_utilization",
                metric_type=MetricType.MEMORY,
                value=node.memory_utilization,
                node_id=node.node_id,
                unit="%",
                description="Memory utilization"
            )
            
            # 记录磁盘指标
            await self.record_metric(
                metric_name="disk_utilization",
                metric_type=MetricType.DISK,
                value=node.disk_utilization,
                node_id=node.node_id,
                unit="%",
                description="Disk utilization"
            )
            
            # 记录GPU指标
            for gpu in node.gpus:
                await self.record_metric(
                    metric_name=f"gpu_{gpu.gpu_id}_utilization",
                    metric_type=MetricType.GPU,
                    value=gpu.utilization,
                    node_id=node.node_id,
                    unit="%",
                    description=f"GPU {gpu.gpu_id} utilization"
                )
                
                await self.record_metric(
                    metric_name=f"gpu_{gpu.gpu_id}_memory_utilization",
                    metric_type=MetricType.GPU,
                    value=gpu.memory_utilization,
                    node_id=node.node_id,
                    unit="%",
                    description=f"GPU {gpu.gpu_id} memory utilization"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record system metrics for node {node.node_id}: {e}")
            return False
    
    async def record_training_metrics(self, task: TrainingTask, metrics: Dict[str, float]) -> bool:
        """记录训练指标"""
        try:
            for metric_name, value in metrics.items():
                await self.record_metric(
                    metric_name=metric_name,
                    metric_type=MetricType.TRAINING,
                    value=value,
                    task_id=task.task_id,
                    unit="",
                    description=f"Training metric: {metric_name}"
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record training metrics for task {task.task_id}: {e}")
            return False
    
    async def get_metrics(self, metric_type: Optional[MetricType] = None,
                         node_id: Optional[str] = None, task_id: Optional[str] = None,
                         limit: int = 100) -> List[MetricData]:
        """获取指标数据"""
        async with self._lock:
            filtered_metrics = self.metrics
            
            # 应用过滤条件
            if metric_type:
                filtered_metrics = [m for m in filtered_metrics if m.metric_type == metric_type]
            
            if node_id:
                filtered_metrics = [m for m in filtered_metrics if m.node_id == node_id]
            
            if task_id:
                filtered_metrics = [m for m in filtered_metrics if m.task_id == task_id]
            
            # 按时间排序并限制数量
            filtered_metrics.sort(key=lambda x: x.timestamp, reverse=True)
            return filtered_metrics[:limit]
    
    async def get_aggregations(self, metric_type: Optional[MetricType] = None) -> List[MetricAggregation]:
        """获取聚合数据"""
        async with self._lock:
            aggregations = list(self.aggregations.values())
            
            if metric_type:
                aggregations = [a for a in aggregations if a.metric_type == metric_type]
            
            return aggregations
    
    async def get_average_metric(self, metric_name: str, 
                               metric_type: MetricType = MetricType.SYSTEM) -> Optional[float]:
        """获取指标平均值"""
        async with self._lock:
            agg_key = f"{metric_name}_{metric_type.value}"
            if agg_key in self.aggregations:
                agg = self.aggregations[agg_key]
                return agg.value / agg.count if agg.count > 0 else None
            return None
    
    async def get_metric_statistics(self, metric_name: str,
                                  metric_type: MetricType = MetricType.SYSTEM) -> Optional[Dict[str, float]]:
        """获取指标统计信息"""
        async with self._lock:
            agg_key = f"{metric_name}_{metric_type.value}"
            if agg_key in self.aggregations:
                agg = self.aggregations[agg_key]
                return {
                    "average": agg.value / agg.count if agg.count > 0 else 0,
                    "min": agg.min_value,
                    "max": agg.max_value,
                    "count": agg.count
                }
            return None
    
    async def register_metric_handler(self, metric_type: MetricType, handler: Callable):
        """注册指标处理器"""
        self.metric_handlers[metric_type] = handler
        logger.debug(f"Registered metric handler for type: {metric_type.value}")
    
    async def unregister_metric_handler(self, metric_type: MetricType):
        """注销指标处理器"""
        if metric_type in self.metric_handlers:
            del self.metric_handlers[metric_type]
            logger.debug(f"Unregistered metric handler for type: {metric_type.value}")
    
    async def export_metrics(self, format: str = "json") -> str:
        """导出指标数据"""
        async with self._lock:
            if format.lower() == "json":
                # 转换为可序列化的格式
                metrics_data = []
                for metric in self.metrics:
                    metrics_data.append({
                        "metric_name": metric.metric_name,
                        "metric_type": metric.metric_type.value,
                        "value": metric.value,
                        "timestamp": metric.timestamp.isoformat(),
                        "node_id": metric.node_id,
                        "task_id": metric.task_id,
                        "tags": metric.tags,
                        "unit": metric.unit,
                        "description": metric.description
                    })
                
                return json.dumps(metrics_data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
    
    async def clear_metrics(self):
        """清空指标数据"""
        async with self._lock:
            self.metrics.clear()
            self.aggregations.clear()
            logger.info("Metrics data cleared")


# 全局指标收集器实例
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器实例"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def set_metrics_collector(metrics_collector: MetricsCollector):
    """设置全局指标收集器实例"""
    global _metrics_collector
    _metrics_collector = metrics_collector
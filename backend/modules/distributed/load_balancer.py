"""负载均衡器

优化训练任务的负载分布。
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
import logging

from .cluster_manager import NodeInfo, NodeStatus
from .task_scheduler import TrainingTask, TaskStatus, ResourceRequirement

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"          # 轮询
    LEAST_CONNECTIONS = "least_connections"  # 最少连接
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # 加权轮询
    LEAST_LOADED = "least_loaded"        # 最少负载
    RANDOM = "random"                    # 随机
    IP_HASH = "ip_hash"                  # IP哈希


@dataclass
class LoadMetrics:
    """负载指标"""
    node_id: str
    cpu_utilization: float = 0.0      # CPU利用率 (0-100)
    memory_utilization: float = 0.0   # 内存利用率 (0-100)
    gpu_utilization: float = 0.0      # GPU利用率 (0-100)
    disk_utilization: float = 0.0     # 磁盘利用率 (0-100)
    network_utilization: float = 0.0  # 网络利用率 (0-100)
    active_connections: int = 0       # 活跃连接数
    queued_tasks: int = 0             # 队列任务数
    timestamp: datetime = field(default_factory=datetime.now)


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED):
        self.strategy = strategy
        self.nodes: Dict[str, NodeInfo] = {}
        self.load_metrics: Dict[str, LoadMetrics] = {}
        self.node_weights: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        self._metrics_update_interval = 5  # 指标更新间隔(秒)
    
    async def start(self):
        """启动负载均衡器"""
        async with self._lock:
            if self._running:
                return
            
            self._running = True
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Load balancer started")
    
    async def stop(self):
        """停止负载均衡器"""
        async with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            logger.info("Load balancer stopped")
    
    async def _monitoring_loop(self):
        """监控循环"""
        last_metrics_update = 0
        
        while self._running:
            try:
                current_time = time.time()
                
                # 定期更新负载指标
                if current_time - last_metrics_update >= self._metrics_update_interval:
                    await self._update_load_metrics()
                    last_metrics_update = current_time
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Load balancer monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _update_load_metrics(self):
        """更新负载指标"""
        async with self._lock:
            for node_id, node in self.nodes.items():
                # 创建负载指标
                metrics = LoadMetrics(
                    node_id=node_id,
                    cpu_utilization=node.cpu_utilization,
                    memory_utilization=node.memory_utilization,
                    disk_utilization=node.disk_utilization,
                    active_connections=len(node.running_tasks),
                    queued_tasks=0  # 简化实现
                )
                
                # 计算GPU利用率
                if node.gpus:
                    total_gpu_util = sum(gpu.utilization for gpu in node.gpus)
                    metrics.gpu_utilization = total_gpu_util / len(node.gpus)
                
                # 更新指标
                self.load_metrics[node_id] = metrics
    
    async def add_node(self, node: NodeInfo, weight: float = 1.0) -> bool:
        """添加节点"""
        async with self._lock:
            self.nodes[node.node_id] = node
            self.node_weights[node.node_id] = weight
            logger.info(f"Node added to load balancer: {node.node_id} with weight {weight}")
            return True
    
    async def remove_node(self, node_id: str) -> bool:
        """移除节点"""
        async with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                if node_id in self.node_weights:
                    del self.node_weights[node_id]
                if node_id in self.load_metrics:
                    del self.load_metrics[node_id]
                logger.info(f"Node removed from load balancer: {node_id}")
                return True
            return False
    
    async def select_node(self, task: Optional[TrainingTask] = None) -> Optional[NodeInfo]:
        """选择节点"""
        async with self._lock:
            if not self.nodes:
                logger.warning("No nodes available for load balancing")
                return None
            
            # 获取健康的节点
            healthy_nodes = {nid: node for nid, node in self.nodes.items() 
                           if node.status == NodeStatus.HEALTHY}
            
            if not healthy_nodes:
                logger.warning("No healthy nodes available for load balancing")
                return None
            
            # 根据策略选择节点
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return await self._round_robin_selection(list(healthy_nodes.values()))
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return await self._least_connections_selection(list(healthy_nodes.values()))
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return await self._weighted_round_robin_selection(list(healthy_nodes.values()))
            elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
                return await self._least_loaded_selection(list(healthy_nodes.values()))
            elif self.strategy == LoadBalancingStrategy.RANDOM:
                return await self._random_selection(list(healthy_nodes.values()))
            elif self.strategy == LoadBalancingStrategy.IP_HASH:
                return await self._ip_hash_selection(list(healthy_nodes.values()), task)
            else:
                # 默认使用最少负载策略
                return await self._least_loaded_selection(list(healthy_nodes.values()))
    
    async def _round_robin_selection(self, nodes: List[NodeInfo]) -> Optional[NodeInfo]:
        """轮询选择"""
        if not nodes:
            return None
        
        # 简化实现：选择第一个节点
        return nodes[0]
    
    async def _least_connections_selection(self, nodes: List[NodeInfo]) -> Optional[NodeInfo]:
        """最少连接选择"""
        if not nodes:
            return None
        
        # 选择运行任务最少的节点
        selected_node = min(nodes, key=lambda node: len(node.running_tasks))
        return selected_node
    
    async def _weighted_round_robin_selection(self, nodes: List[NodeInfo]) -> Optional[NodeInfo]:
        """加权轮询选择"""
        if not nodes:
            return None
        
        # 根据权重选择节点
        total_weight = sum(self.node_weights.get(node.node_id, 1.0) for node in nodes)
        if total_weight <= 0:
            return nodes[0]
        
        # 简化实现：按权重比例选择
        import random
        random_value = random.uniform(0, total_weight)
        current_weight = 0
        
        for node in nodes:
            current_weight += self.node_weights.get(node.node_id, 1.0)
            if random_value <= current_weight:
                return node
        
        return nodes[0]
    
    async def _least_loaded_selection(self, nodes: List[NodeInfo]) -> Optional[NodeInfo]:
        """最少负载选择"""
        if not nodes:
            return None
        
        # 计算每个节点的负载分数
        node_scores = []
        for node in nodes:
            metrics = self.load_metrics.get(node.node_id)
            if metrics:
                # 综合负载分数 (越小越好)
                load_score = (
                    metrics.cpu_utilization * 0.3 +
                    metrics.memory_utilization * 0.2 +
                    metrics.gpu_utilization * 0.3 +
                    metrics.disk_utilization * 0.1 +
                    (metrics.active_connections / 10.0) * 0.1
                )
                node_scores.append((node, load_score))
            else:
                # 没有指标数据时使用默认分数
                node_scores.append((node, 50.0))
        
        # 选择负载分数最低的节点
        if node_scores:
            selected_node, _ = min(node_scores, key=lambda x: x[1])
            return selected_node
        
        return nodes[0]
    
    async def _random_selection(self, nodes: List[NodeInfo]) -> Optional[NodeInfo]:
        """随机选择"""
        if not nodes:
            return None
        
        import random
        return random.choice(nodes)
    
    async def _ip_hash_selection(self, nodes: List[NodeInfo], 
                               task: Optional[TrainingTask] = None) -> Optional[NodeInfo]:
        """IP哈希选择"""
        if not nodes:
            return None
        
        # 使用任务ID或随机值进行哈希
        hash_value = task.task_id if task and task.task_id else str(time.time())
        hash_int = hash(hash_value)
        
        # 选择节点
        selected_index = hash_int % len(nodes)
        return nodes[selected_index]
    
    async def update_node_weight(self, node_id: str, weight: float) -> bool:
        """更新节点权重"""
        async with self._lock:
            if node_id in self.nodes:
                self.node_weights[node_id] = weight
                logger.info(f"Node weight updated: {node_id} = {weight}")
                return True
            return False
    
    async def get_load_distribution(self) -> Dict[str, Dict[str, Any]]:
        """获取负载分布"""
        async with self._lock:
            distribution = {}
            for node_id, metrics in self.load_metrics.items():
                distribution[node_id] = {
                    "cpu_utilization": metrics.cpu_utilization,
                    "memory_utilization": metrics.memory_utilization,
                    "gpu_utilization": metrics.gpu_utilization,
                    "disk_utilization": metrics.disk_utilization,
                    "active_connections": metrics.active_connections,
                    "weight": self.node_weights.get(node_id, 1.0)
                }
            return distribution
    
    async def get_balancing_statistics(self) -> Dict[str, Any]:
        """获取负载均衡统计信息"""
        async with self._lock:
            total_nodes = len(self.nodes)
            healthy_nodes = sum(1 for node in self.nodes.values() 
                              if node.status == NodeStatus.HEALTHY)
            
            # 计算平均负载
            avg_cpu = 0.0
            avg_memory = 0.0
            avg_gpu = 0.0
            total_metrics = 0
            
            for metrics in self.load_metrics.values():
                avg_cpu += metrics.cpu_utilization
                avg_memory += metrics.memory_utilization
                avg_gpu += metrics.gpu_utilization
                total_metrics += 1
            
            if total_metrics > 0:
                avg_cpu /= total_metrics
                avg_memory /= total_metrics
                avg_gpu /= total_metrics
            
            return {
                "strategy": self.strategy.value,
                "total_nodes": total_nodes,
                "healthy_nodes": healthy_nodes,
                "average_cpu_utilization": avg_cpu,
                "average_memory_utilization": avg_memory,
                "average_gpu_utilization": avg_gpu,
                "node_weights": self.node_weights.copy()
            }


# 全局负载均衡器实例
_load_balancer: Optional[LoadBalancer] = None


def get_load_balancer() -> LoadBalancer:
    """获取全局负载均衡器实例"""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = LoadBalancer()
    return _load_balancer


def set_load_balancer(load_balancer: LoadBalancer):
    """设置全局负载均衡器实例"""
    global _load_balancer
    _load_balancer = load_balancer
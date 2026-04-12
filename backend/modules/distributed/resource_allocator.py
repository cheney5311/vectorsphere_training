"""资源分配器

负责GPU、内存等资源的智能分配。
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
from .gpu_resource_manager import get_cached_metrics

from .task_scheduler import ResourceRequirement

logger = logging.getLogger(__name__)


@dataclass
class ResourceAllocation:
    """资源分配"""
    node_id: str
    cpu_cores: int
    memory_mb: int
    gpus: List[int]  # GPU ID列表
    gpu_memory_mb: int
    disk_mb: int
    network_mbps: int
    allocation_time: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


class AllocationStrategy(Enum):
    """分配策略枚举"""
    FIRST_FIT = "first_fit"          # 首次适应
    BEST_FIT = "best_fit"            # 最佳适应
    WORST_FIT = "worst_fit"          # 最差适应
    ROUND_ROBIN = "round_robin"      # 轮询


class ResourceAllocator:
    """资源分配器

    说明：
    - 支持基于优先级与亲和规则的分配
    - 当前实现为非抢占式；将来可增加 preemptive 参数支持优先级抢占
    """

    def __init__(self, strategy: AllocationStrategy = AllocationStrategy.BEST_FIT):
        self.strategy = strategy
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.node_allocations: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()
        self._allocation_counter = 0
    
    async def allocate_resources(self, nodes: List[NodeInfo], 
                               requirements: ResourceRequirement) -> Optional[tuple]:
        """分配资源（支持基于 priority 的短时等待）
        返回 (allocation_id, ResourceAllocation) 或 None

        行为说明：
        - 在同一进程内并发调用时使用内部锁保证序列化。
        - 如果首次尝试未命中且 requirements.priority 较高（>5），会等待重试直到超时。
          等待时长由环境变量 `GPU_ALLOC_WAIT_SECONDS` 控制（默认 5s），轮询间隔由 `GPU_ALLOC_RETRY_INTERVAL` 控制（默认 0.5s）。
        - 当前不做抢占（preemptive）操作；若等待结束仍未成功则返回 None。
        """
        wait_seconds = float(__import__('os').environ.get('GPU_ALLOC_WAIT_SECONDS', '5'))
        retry_interval = float(__import__('os').environ.get('GPU_ALLOC_RETRY_INTERVAL', '0.5'))
        priority_threshold = int(getattr(requirements, 'priority', 1))

        start_ts = time.time()
        async with self._lock:
            while True:
                # 根据策略选择节点
                selected_node = await self._select_node(nodes, requirements)
                if selected_node:
                    # 分配资源
                    allocation = await self._allocate_on_node(selected_node, requirements)
                    if allocation:
                        # 记录分配
                        allocation_id = f"alloc_{self._allocation_counter}"
                        self._allocation_counter += 1
                        allocation.node_id = selected_node.node_id

                        self.allocations[allocation_id] = allocation
                        if selected_node.node_id not in self.node_allocations:
                            self.node_allocations[selected_node.node_id] = []
                        self.node_allocations[selected_node.node_id].append(allocation_id)

                        logger.info(f"Resources allocated on node {selected_node.node_id}: {allocation_id}")
                        return (allocation_id, allocation)
                    else:
                        logger.warning(f"Failed to allocate resources on node {selected_node.node_id}")
                        # 没找到合适的 GPU 或分配失败，视作未成功
                else:
                    logger.debug("No suitable node found for resource allocation on this attempt")

                # 若任务优先级高，且未超过等待时长，则等待重试
                if priority_threshold > 5:
                    elapsed = time.time() - start_ts
                    if elapsed < wait_seconds:
                        await asyncio.sleep(retry_interval)
                        # 在重试前尝试刷新 nodes 的实时信息（如果提供）
                        try:
                            # 如果 callers 传入的是 cluster manager 的节点列表引用，试图让调用方刷新
                            pass
                        except Exception:
                            pass
                        continue
                # 不重试或超时，返回 None
                return None
    
    async def _select_node(self, nodes: List[NodeInfo], 
                          requirements: ResourceRequirement) -> Optional[NodeInfo]:
        """根据策略选择节点

        优化：若 requirements.labels_affinity 存在，优先在匹配标签的节点集合中运行选择算法，保证亲和性优先。
        """
        if not nodes:
            return None

        # 优先过滤出匹配 labels_affinity 的节点（若有）
        try:
            labels_affinity = getattr(requirements, 'labels_affinity', None)
            if labels_affinity:
                preferred = [n for n in nodes if all(n.labels.get(k) == v for k, v in labels_affinity.items())]
                # 如果存在首选集合，则在首选集合内进行选择
                if preferred:
                    nodes_to_consider = preferred
                else:
                    nodes_to_consider = nodes
            else:
                nodes_to_consider = nodes
        except Exception:
            nodes_to_consider = nodes
        
        if self.strategy == AllocationStrategy.FIRST_FIT:
            for node in nodes_to_consider:
                if await self._node_meets_requirements(node, requirements):
                    return node
        elif self.strategy == AllocationStrategy.BEST_FIT:
            best_node = None
            best_fit_score = float('inf')
            
            for node in nodes_to_consider:
                if await self._node_meets_requirements(node, requirements):
                    fit_score = await self._calculate_fit_score(node, requirements)
                    if fit_score < best_fit_score:
                        best_fit_score = fit_score
                        best_node = node
            
            return best_node
        elif self.strategy == AllocationStrategy.WORST_FIT:
            worst_node = None
            worst_fit_score = -1
            
            for node in nodes_to_consider:
                if await self._node_meets_requirements(node, requirements):
                    fit_score = await self._calculate_fit_score(node, requirements)
                    if fit_score > worst_fit_score:
                        worst_fit_score = fit_score
                        worst_node = node
            
            return worst_node
        elif self.strategy == AllocationStrategy.ROUND_ROBIN:
            for node in nodes_to_consider:
                if await self._node_meets_requirements(node, requirements):
                    return node

        # 若首选集合存在且未找到合适节点，回退到全部节点再尝试一次
        if nodes_to_consider is not nodes:
            for node in nodes:
                if await self._node_meets_requirements(node, requirements):
                    return node

        return None
    
    async def _node_meets_requirements(self, node: NodeInfo, 
                                     requirements: ResourceRequirement) -> bool:
        """检查节点是否满足资源要求"""
        # 检查CPU核心数
        if requirements.cpu_cores > node.cpu_count:
            return False
        
        # 检查内存
        available_memory = node.memory_total - node.memory_used
        if requirements.memory_mb > available_memory:
            return False
        
        # 检查磁盘空间
        available_disk = node.disk_total - node.disk_used
        if requirements.disk_mb > available_disk:
            return False
        
        # 检查GPU数量和内存
        if requirements.gpu_count > 0:
            available_gpus = [gpu for gpu in node.gpus if gpu.is_available]
            if len(available_gpus) < requirements.gpu_count:
                return False
            
            # 检查GPU内存
            total_gpu_memory = sum(gpu.memory_free for gpu in available_gpus)
            if requirements.gpu_memory_mb > total_gpu_memory:
                return False
        
        return True
    
    async def _calculate_fit_score(self, node: NodeInfo, 
                                 requirements: ResourceRequirement) -> float:
        """
        计算适应度分数（越小越好）。考虑 CPU、内存、以及 GPU 实时利用率与可用内存。
        另外对有 GPU 要求的请求，会偏好可用 GPU 数量多、单卡空闲内存多、且利用率低的节点。
        """
        # CPU 利用率估计
        cpu_utilization = requirements.cpu_cores / node.cpu_count if node.cpu_count > 0 else 1.0
        # 内存利用率估计（避免除零）
        avail_mem = max(1, node.memory_total - node.memory_used)
        memory_utilization = requirements.memory_mb / avail_mem if avail_mem > 0 else 1.0

        # GPU 相关评分
        if requirements.gpu_count > 0:
            available_gpus = [gpu for gpu in node.gpus if gpu.is_available]
            if not available_gpus:
                gpu_utilization = 1.0
                gpu_memory_factor = 1.0
            else:
                # 平均 GPU 利用率（越低越好）
                avg_gpu_util = sum((gpu.utilization or 0.0) for gpu in available_gpus) / len(available_gpus)
                # 平均每卡空闲内存比例（越高越好）
                avg_gpu_mem_free = sum((gpu.memory_free or 0) for gpu in available_gpus) / len(available_gpus)
                # 期望的单卡空闲内存比例（要求越大，此项越重要）
                requested_per_gpu_mem = requirements.gpu_memory_mb / max(1, requirements.gpu_count)
                # 计算内存因子：若 avg_gpu_mem_free 足够则偏好（值范围大于0且越小越好）
                gpu_memory_factor = max(0.01, 1.0 - (avg_gpu_mem_free - requested_per_gpu_mem) / max(1, requested_per_gpu_mem))
                # 将 avg_gpu_util 映射为利用率分数（0.0 最好，1.0 最差）
                gpu_utilization = min(1.0, avg_gpu_util / 100.0)
            # 综合 GPU 分数
            gpu_util_score = (gpu_utilization + gpu_memory_factor) / 2.0
        else:
            gpu_util_score = 0.0

        # 最终 fit score：按权重合成（权重可调整）
        cpu_w = 0.4
        mem_w = 0.3
        gpu_w = 0.3
        fit_score = cpu_w * cpu_utilization + mem_w * memory_utilization + gpu_w * gpu_util_score
        return fit_score
    
    async def _allocate_on_node(self, node: NodeInfo, 
                               requirements: ResourceRequirement) -> Optional[ResourceAllocation]:
        """在节点上分配资源"""
        # 创建资源分配对象
        allocation = ResourceAllocation(
            node_id="",  # 稍后设置
            cpu_cores=requirements.cpu_cores,
            memory_mb=requirements.memory_mb,
            gpus=[],  # 稍后设置
            gpu_memory_mb=requirements.gpu_memory_mb,
            disk_mb=requirements.disk_mb,
            network_mbps=requirements.network_mbps
        )
        
        # 分配GPU：选择可用 GPU 并标记为不可用（就地修改 NodeInfo）
        if requirements.gpu_count > 0:
            available_gpus = [gpu for gpu in node.gpus if gpu.is_available]
            take = min(requirements.gpu_count, len(available_gpus))
            # 简单按顺序分配
            per_gpu_req = int(requirements.gpu_memory_mb / max(1, take)) if take > 0 else 0
            for i in range(take):
                gpu = available_gpus[i]
                allocation.gpus.append(gpu.gpu_id)
                try:
                    # 标记占用
                    gpu.is_available = False
                    # 增加已用内存与减少空闲内存
                    gpu.memory_used = int(gpu.memory_used or 0) + per_gpu_req
                    gpu.memory_free = max(0, int(gpu.memory_total or 0) - int(gpu.memory_used or 0))
                except Exception:
                    pass
        
        return allocation
    
    async def release_resources(self, allocation_id: str) -> bool:
        """释放资源"""
        async with self._lock:
            if allocation_id in self.allocations:
                allocation = self.allocations[allocation_id]
                
                # 从节点分配记录中移除
                if allocation.node_id in self.node_allocations:
                    if allocation_id in self.node_allocations[allocation.node_id]:
                        self.node_allocations[allocation.node_id].remove(allocation_id)
                
                # 删除分配记录
                del self.allocations[allocation_id]
                
                logger.info(f"Resources released: {allocation_id}")
                return True
            return False
    
    async def get_node_allocations(self, node_id: str) -> List[ResourceAllocation]:
        """获取节点的资源分配"""
        async with self._lock:
            allocation_ids = self.node_allocations.get(node_id, [])
            return [self.allocations[aid] for aid in allocation_ids if aid in self.allocations]
    
    async def get_allocation(self, allocation_id: str) -> Optional[ResourceAllocation]:
        """获取资源分配"""
        async with self._lock:
            return self.allocations.get(allocation_id)
    
    async def list_allocations(self) -> List[ResourceAllocation]:
        """列出所有资源分配"""
        async with self._lock:
            return list(self.allocations.values())
    
    async def get_allocation_statistics(self) -> Dict[str, Any]:
        """获取分配统计信息"""
        async with self._lock:
            total_allocations = len(self.allocations)
            node_allocation_counts = {node_id: len(allocation_ids) 
                                    for node_id, allocation_ids in self.node_allocations.items()}
            
            return {
                "total_allocations": total_allocations,
                "node_allocations": node_allocation_counts,
                "allocation_ids": list(self.allocations.keys())
            }

    async def find_allocations_by_node_and_gpus(self, node_id: str, gpus: List[int]) -> List[str]:
        """根据节点ID和GPU列表查找可能的 allocation_id 列表（匹配包含所有指定GPU的分配）"""
        async with self._lock:
            result = []
            allocation_ids = self.node_allocations.get(node_id, [])
            for aid in allocation_ids:
                alloc = self.allocations.get(aid)
                if not alloc:
                    continue
                # 如果 alloc.gpus 包含请求的所有 GPU，则认为匹配
                if all(g in alloc.gpus for g in gpus):
                    result.append(aid)
            return result


# 全局资源分配器实例
_resource_allocator: Optional[ResourceAllocator] = None


def get_resource_allocator() -> ResourceAllocator:
    """获取全局资源分配器实例"""
    global _resource_allocator
    if _resource_allocator is None:
        _resource_allocator = ResourceAllocator()
    return _resource_allocator


def set_resource_allocator(resource_allocator: ResourceAllocator):
    """设置全局资源分配器实例"""
    global _resource_allocator
    _resource_allocator = resource_allocator
"""集群管理器

负责GPU集群的节点发现、健康检查、资源监控等功能。
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """节点状态枚举"""
    UNKNOWN = "unknown"  # 未知状态
    HEALTHY = "healthy"  # 健康
    UNHEALTHY = "unhealthy"  # 不健康
    OFFLINE = "offline"  # 离线
    MAINTENANCE = "maintenance"  # 维护中
    DRAINING = "draining"  # 正在排空


class ClusterStatus(Enum):
    """集群状态枚举"""
    INITIALIZING = "initializing"  # 初始化中
    HEALTHY = "healthy"  # 健康
    DEGRADED = "degraded"  # 降级
    CRITICAL = "critical"  # 严重
    OFFLINE = "offline"  # 离线


@dataclass
class GPUInfo:
    """GPU信息"""
    gpu_id: int
    name: str
    memory_total: int  # MB
    memory_used: int  # MB
    memory_free: int  # MB
    utilization: float  # 0-100
    temperature: float  # 摄氏度
    power_usage: float  # 瓦特
    driver_version: str
    cuda_version: str
    is_available: bool = True
    processes: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def memory_utilization(self) -> float:
        """内存使用率"""
        return (self.memory_used / self.memory_total) * 100 if self.memory_total > 0 else 0


@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    status: NodeStatus
    last_heartbeat: datetime
    
    # 硬件信息
    cpu_count: int
    memory_total: int  # MB
    memory_used: int  # MB
    disk_total: int  # MB
    disk_used: int  # MB
    
    # GPU信息
    gpus: List[GPUInfo] = field(default_factory=list)
    
    # 性能指标
    cpu_utilization: float = 0.0  # 0-100
    memory_utilization: float = 0.0  # 0-100
    disk_utilization: float = 0.0  # 0-100
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    
    # 训练任务信息
    running_tasks: List[str] = field(default_factory=list)
    max_tasks: int = 1
    
    # 标签和注解
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    
    # 健康检查
    health_check_failures: int = 0
    max_health_check_failures: int = 3
    
    @property
    def is_healthy(self) -> bool:
        """节点是否健康"""
        return (
            self.status == NodeStatus.HEALTHY and
            self.health_check_failures < self.max_health_check_failures and
            (datetime.now() - self.last_heartbeat).total_seconds() < 60
        )
    
    @property
    def available_gpus(self) -> List[GPUInfo]:
        """可用的GPU列表"""
        return [gpu for gpu in self.gpus if gpu.is_available]
    
    @property
    def total_gpu_memory(self) -> int:
        """总GPU内存 (MB)"""
        return sum(gpu.memory_total for gpu in self.gpus)
    
    @property
    def used_gpu_memory(self) -> int:
        """已使用GPU内存 (MB)"""
        return sum(gpu.memory_used for gpu in self.gpus)
    
    @property
    def can_accept_task(self) -> bool:
        """是否可以接受新任务"""
        return (
            self.is_healthy and
            len(self.running_tasks) < self.max_tasks and
            len(self.available_gpus) > 0
        )


class NodeDiscovery(ABC):
    """节点发现接口"""
    
    @abstractmethod
    async def discover_nodes(self) -> List[NodeInfo]:
        """发现集群中的节点"""
        pass
    
    @abstractmethod
    async def register_node(self, node_info: NodeInfo) -> bool:
        """注册节点"""
        pass
    
    @abstractmethod
    async def unregister_node(self, node_id: str) -> bool:
        """注销节点"""
        pass


class StaticNodeDiscovery(NodeDiscovery):
    """静态节点发现"""
    
    def __init__(self, nodes: List[Dict[str, Any]]):
        self.nodes = nodes
        self._registered_nodes: Dict[str, NodeInfo] = {}
    
    async def discover_nodes(self) -> List[NodeInfo]:
        """发现预配置的节点"""
        discovered = []
        for node_config in self.nodes:
            node_info = NodeInfo(
                node_id=node_config['node_id'],
                hostname=node_config['hostname'],
                ip_address=node_config['ip_address'],
                port=node_config.get('port', 22),
                status=NodeStatus.UNKNOWN,
                last_heartbeat=datetime.now(),
                cpu_count=node_config.get('cpu_count', 0),
                memory_total=node_config.get('memory_total', 0),
                memory_used=0,
                disk_total=node_config.get('disk_total', 0),
                disk_used=0,
                labels=node_config.get('labels', {}),
                annotations=node_config.get('annotations', {})
            )
            discovered.append(node_info)
        return discovered
    
    async def register_node(self, node_info: NodeInfo) -> bool:
        """注册节点"""
        self._registered_nodes[node_info.node_id] = node_info
        return True
    
    async def unregister_node(self, node_id: str) -> bool:
        """注销节点"""
        if node_id in self._registered_nodes:
            del self._registered_nodes[node_id]
            return True
        return False


class KubernetesNodeDiscovery(NodeDiscovery):
    """Kubernetes节点发现"""
    
    def __init__(self, namespace: str = "default", label_selector: str = ""):
        self.namespace = namespace
        self.label_selector = label_selector
    
    async def discover_nodes(self) -> List[NodeInfo]:
        """发现Kubernetes集群中的节点"""
        # 这里应该实现Kubernetes API调用来发现节点
        # 为简化起见，返回空列表
        logger.info("Discovering nodes in Kubernetes namespace: %s", self.namespace)
        return []
    
    async def register_node(self, node_info: NodeInfo) -> bool:
        """在Kubernetes中注册节点"""
        # 这里应该实现Kubernetes API调用来注册节点
        logger.info("Registering node %s in Kubernetes", node_info.node_id)
        return True
    
    async def unregister_node(self, node_id: str) -> bool:
        """在Kubernetes中注销节点"""
        # 这里应该实现Kubernetes API调用来注销节点
        logger.info("Unregistering node %s from Kubernetes", node_id)
        return True


class ClusterManager:
    """集群管理器"""
    
    def __init__(self, node_discovery: NodeDiscovery):
        self.node_discovery = node_discovery
        self.nodes: Dict[str, NodeInfo] = {}
        self.cluster_status = ClusterStatus.INITIALIZING
        self._lock = asyncio.Lock()
        self._heartbeat_interval = 30  # 心跳间隔(秒)
        self._discovery_interval = 60  # 节点发现间隔(秒)
        self._running = False
        self._management_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动集群管理器"""
        async with self._lock:
            if self._running:
                return
            
            self._running = True
            self._management_task = asyncio.create_task(self._management_loop())
            logger.info("Cluster manager started")
    
    async def stop(self):
        """停止集群管理器"""
        async with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._management_task:
                self._management_task.cancel()
                try:
                    await self._management_task
                except asyncio.CancelledError:
                    pass
            logger.info("Cluster manager stopped")
    
    async def _management_loop(self):
        """管理循环"""
        last_discovery = 0
        last_heartbeat = 0
        
        while self._running:
            try:
                current_time = time.time()
                
                # 定期发现新节点
                if current_time - last_discovery >= self._discovery_interval:
                    await self._discover_nodes()
                    last_discovery = current_time
                
                # 定期发送心跳
                if current_time - last_heartbeat >= self._heartbeat_interval:
                    await self._send_heartbeats()
                    last_heartbeat = current_time
                
                # 更新集群状态
                await self._update_cluster_status()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Cluster management loop error: {e}")
                await asyncio.sleep(5)
    
    async def _discover_nodes(self):
        """发现节点"""
        try:
            discovered_nodes = await self.node_discovery.discover_nodes()
            
            async with self._lock:
                for node in discovered_nodes:
                    if node.node_id not in self.nodes:
                        self.nodes[node.node_id] = node
                        logger.info(f"Discovered new node: {node.node_id}")
            
            logger.debug(f"Discovered {len(discovered_nodes)} nodes")
            
        except Exception as e:
            logger.error(f"Node discovery failed: {e}")
    
    async def _send_heartbeats(self):
        """发送心跳与增强健康判定
        - 更新心跳时间
        - 本地采集 GPU 指标（若为本机），并重置/递增 health_check_failures
        - 超过阈值则标记为 UNHEALTHY 并上报 FaultToleranceManager
        """
        async with self._lock:
            for node in self.nodes.values():
                try:
                    # 尝试发送/记录心跳
                    node.last_heartbeat = datetime.now()
                    logger.debug(f"Sent heartbeat to node: {node.node_id}")

                    heartbeat_ok = True

                    # 如果节点为本地主机（hostname 简单匹配），优先使用本地采集模块的缓存指标以减少外部调用
                    if node.hostname in ("localhost", "127.0.0.1"):
                        try:
                            from backend.modules.distributed.gpu_resource_manager import get_cached_metrics, collect_once
                            # 尝试先读取缓存，若为空则立即采集一次
                            cached = get_cached_metrics()
                            if not cached or 'gpus' not in cached.get('last', {}):
                                gm_summary = collect_once()
                                gpu_metrics = gm_summary.get('gpus', []) if gm_summary else []
                            else:
                                gpu_metrics = cached.get('last', {}).get('gpus', [])
                        except Exception:
                            # 回退到服务级别实现
                            try:
                                from backend.services.gpu_resource_manager import get_gpu_metrics
                                gpu_metrics = get_gpu_metrics()
                            except Exception:
                                gpu_metrics = []

                        # 将采集到的 metrics 映射为 GPUInfo 列表（保守填充字段）
                        new_gpus = []
                        for gm in gpu_metrics:
                            try:
                                gpu = GPUInfo(
                                    gpu_id=int(gm.get('index', 0)),
                                    name=gm.get('name', 'unknown'),
                                    memory_total=int(gm.get('memory_total_mb') or gm.get('memory_total') or 0),
                                    memory_used=int(gm.get('memory_used_mb') or gm.get('memory_used') or 0),
                                    memory_free=max(0, int(gm.get('memory_total_mb') or gm.get('memory_total') or 0) - int(gm.get('memory_used_mb') or gm.get('memory_used') or 0)),
                                    utilization=float(gm.get('utilization_percent') or gm.get('utilization') or 0.0),
                                    temperature=float(gm.get('temperature') or 0.0),
                                    power_usage=float(gm.get('power_usage') or 0.0),
                                    driver_version=gm.get('driver_version', ''),
                                    cuda_version=gm.get('cuda_version', ''),
                                    is_available=(gm.get('memory_total_mb') is None) or ((int(gm.get('memory_total_mb') or gm.get('memory_total') or 0) - int(gm.get('memory_used_mb') or gm.get('memory_used') or 0)) > 0),
                                    processes=gm.get('processes', []) if isinstance(gm.get('processes', []), list) else []
                                )
                                new_gpus.append(gpu)
                            except Exception:
                                continue
                        if new_gpus:
                            node.gpus = new_gpus
                            # 导出 GPU 指标到 Prometheus（非阻塞）
                            try:
                                from backend.modules.monitoring.metrics_exporter import GPU_COUNT_GAUGE, GPU_UTIL_GAUGE, GPU_MEMORY_FREE_GAUGE
                                node_label = node.node_id
                                GPU_COUNT_GAUGE.labels(node=node_label).set(len(new_gpus))
                                for g in new_gpus:
                                    try:
                                        idx = getattr(g, 'gpu_id', None)
                                        GPU_UTIL_GAUGE.labels(node=node_label, gpu_index=str(idx)).set(float(getattr(g, 'utilization', 0.0) or 0.0))
                                        GPU_MEMORY_FREE_GAUGE.labels(node=node_label, gpu_index=str(idx)).set(float(getattr(g, 'memory_free', 0) or 0))
                                    except Exception:
                                        continue
                            except Exception:
                                pass

                    # 成功接收到心跳则重置失败计数
                    if heartbeat_ok:
                        node.health_check_failures = 0
                        node.status = NodeStatus.HEALTHY

                except Exception as e:
                    # 心跳/采集失败，递增失败计数
                    try:
                        node.health_check_failures += 1
                        logger.debug(f"Heartbeat failed for node {node.node_id}, failures={node.health_check_failures}: {e}")
                    except Exception:
                        pass

                    # 当失败次数超过阈值，标记为 UNHEALTHY 并上报 FaultToleranceManager
                    if node.health_check_failures >= node.max_health_check_failures:
                        try:
                            node.status = NodeStatus.UNHEALTHY
                            logger.warning(f"Node {node.node_id} marked UNHEALTHY after {node.health_check_failures} failures")
                            try:
                                from backend.modules.distributed.fault_tolerance import get_fault_tolerance_manager
                                ft_mgr = get_fault_tolerance_manager()
                                async def _report():
                                    await ft_mgr.report_fault(node_id=node.node_id, task_id=None, event_type='node_unhealthy', description=f'Node marked unhealthy after {node.health_check_failures} heartbeat failures')
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    asyncio.ensure_future(_report())
                                else:
                                    loop.run_until_complete(_report())
                            except Exception as e2:
                                logger.error(f"Failed to report node unhealthy: {e2}")
                        except Exception:
                            pass
    
    async def _update_cluster_status(self):
        """更新集群状态"""
        async with self._lock:
            healthy_nodes = sum(1 for node in self.nodes.values() if node.is_healthy)
            total_nodes = len(self.nodes)
            
            if total_nodes == 0:
                self.cluster_status = ClusterStatus.OFFLINE
            elif healthy_nodes == total_nodes:
                self.cluster_status = ClusterStatus.HEALTHY
            elif healthy_nodes >= total_nodes * 0.5:
                self.cluster_status = ClusterStatus.DEGRADED
            else:
                self.cluster_status = ClusterStatus.CRITICAL
    
    async def register_node(self, node_info: NodeInfo) -> bool:
        """注册节点
        扩展：在节点注册时为节点创建 lease（节点级租约），并在 lease 到期时触发容错上报。
        """
        try:
            success = await self.node_discovery.register_node(node_info)
            if success:
                async with self._lock:
                    self.nodes[node_info.node_id] = node_info
                logger.info(f"Node registered: {node_info.node_id}")

                # 尝试为节点创建 lease（TTL 可按需求配置，默认 120s）并注册回调到 FaultToleranceManager
                try:
                    from backend.modules.distributed.lease_manager import get_lease_manager
                    from backend.modules.distributed.fault_tolerance import get_fault_tolerance_manager
                    lease_mgr = get_lease_manager()
                    fault_mgr = None
                    try:
                        fault_mgr = get_fault_tolerance_manager()
                    except Exception:
                        fault_mgr = None

                    lease_id = f"node-lease-{node_info.node_id}-{int(time.time())}"
                    owner_id = f"node-{node_info.node_id}"

                    async def _create_node_lease():
                        await lease_mgr.create_lease(lease_id=lease_id, owner_id=owner_id, ttl_seconds=120, metadata={"node_id": node_info.node_id})
                    try:
                        import asyncio
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(_create_node_lease())
                        else:
                            loop.run_until_complete(_create_node_lease())
                    except Exception:
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(_create_node_lease())
                            loop.close()
                        except Exception as e:
                            logger.warning(f"Failed to create node lease synchronously: {e}")

                    # 保存节点 lease 映射
                    if not hasattr(self, '_node_leases'):
                        self._node_leases = {}
                    self._node_leases[node_info.node_id] = lease_id

                    # 注册 lease 到期回调，将其转为节点故障事件
                    def _on_lease_expired(lease):
                        try:
                            logger.warning(f"Node lease expired for node {node_info.node_id}, lease={lease.lease_id}")
                            if fault_mgr:
                                # report_fault 接口为异步，尝试以安全方式调度
                                async def _report():
                                    await fault_mgr.report_fault(node_id=node_info.node_id, task_id=None, event_type='node_lease_expired', description=f'Node lease expired: {lease.lease_id}')
                                try:
                                    loop = asyncio.get_event_loop()
                                    if loop.is_running():
                                        asyncio.ensure_future(_report())
                                    else:
                                        loop.run_until_complete(_report())
                                except Exception:
                                    try:
                                        loop = asyncio.new_event_loop()
                                        asyncio.set_event_loop(loop)
                                        loop.run_until_complete(_report())
                                        loop.close()
                                    except Exception as e:
                                        logger.error(f"Failed to report node lease expiry: {e}")
                        except Exception as e:
                            logger.error(f"Error in node lease expiry handler: {e}")

                    try:
                        lease_mgr.register_expiry_callback(_on_lease_expired)
                    except Exception as e:
                        logger.debug(f"Failed to register node lease expiry callback: {e}")

                except Exception as e:
                    logger.debug(f"Node lease creation skipped or failed: {e}")

            return success
        except Exception as e:
            logger.error(f"Node registration failed: {e}")
            return False
    
    async def unregister_node(self, node_id: str) -> bool:
        """注销节点"""
        try:
            success = await self.node_discovery.unregister_node(node_id)
            if success:
                async with self._lock:
                    if node_id in self.nodes:
                        del self.nodes[node_id]
                logger.info(f"Node unregistered: {node_id}")
            return success
        except Exception as e:
            logger.error(f"Node unregistration failed: {e}")
            return False
    
    async def update_node_status(self, node_id: str, status: NodeStatus) -> bool:
        """更新节点状态"""
        async with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = status
                logger.info(f"Node {node_id} status updated to: {status.value}")
                return True
            return False
    
    async def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """获取节点信息"""
        async with self._lock:
            return self.nodes.get(node_id)
    
    async def list_nodes(self) -> List[NodeInfo]:
        """列出所有节点"""
        async with self._lock:
            return list(self.nodes.values())
    
    async def get_cluster_status(self) -> ClusterStatus:
        """获取集群状态"""
        async with self._lock:
            return self.cluster_status
    
    async def get_healthy_nodes(self) -> List[NodeInfo]:
        """获取健康节点列表"""
        async with self._lock:
            return [node for node in self.nodes.values() if node.is_healthy]
    
    async def allocate_resources(self, required_gpus: int = 0, min_free_memory_mb: int = 0, labels_affinity: Optional[Dict[str, str]] = None, prefer_same_node: bool = True) -> Optional[List[Dict[str, Any]]]:
        """分配资源（改进版）
        - required_gpus: 需要的 GPU 数量
        - min_free_memory_mb: 每个所选 GPU 至少应有的空闲内存
        - labels_affinity: 节点标签匹配字典，优先选择匹配标签的节点
        - prefer_same_node: 优先把所有 GPU 分配在同一节点（如果可能）

        返回: List[ { 'node': NodeInfo, 'gpu_indices': [int, ...] } ] 或 None
        """
        async with self._lock:
            healthy_nodes = [node for node in self.nodes.values() if node.is_healthy]

            # 根据 labels_affinity 过滤并排序优先级
            def node_score(n: NodeInfo) -> int:
                score = 0
                if labels_affinity:
                    for k, v in labels_affinity.items():
                        if n.labels.get(k) == v:
                            score += 10
                # 更多空闲 GPU 更高分
                score += len(n.available_gpus)
                # 更多空闲内存加分
                score += int((n.total_gpu_memory - n.used_gpu_memory) / 1024)
                return score

            candidate_nodes = sorted(healthy_nodes, key=node_score, reverse=True)

            # 尝试首选把所有 GPU 放在同一节点
            allocation: List[Dict[str, Any]] = []
            if prefer_same_node:
                for node in candidate_nodes:
                    # 过滤每 GPU 的空闲内存
                    good_indices = []
                    for g in node.available_gpus:
                        if g.memory_free >= min_free_memory_mb:
                            good_indices.append(g.gpu_id)
                    if len(good_indices) >= required_gpus and required_gpus > 0:
                        allocation.append({'node': node, 'gpu_indices': good_indices[:required_gpus]})
                        return allocation
            
            # 否则跨节点分配
            remaining = required_gpus
            for node in candidate_nodes:
                if remaining <= 0:
                    break
                good_indices = []
                for g in node.available_gpus:
                    if g.memory_free >= min_free_memory_mb:
                        good_indices.append(g.gpu_id)
                if not good_indices:
                    continue
                take = min(len(good_indices), remaining)
                allocation.append({'node': node, 'gpu_indices': good_indices[:take]})
                remaining -= take

            if remaining <= 0:
                return allocation
            else:
                return None


# 全局集群管理器实例
_cluster_manager: Optional[ClusterManager] = None


def get_cluster_manager() -> ClusterManager:
    """获取全局集群管理器实例"""
    global _cluster_manager
    if _cluster_manager is None:
        # 默认使用静态节点发现
        node_discovery = StaticNodeDiscovery([])
        _cluster_manager = ClusterManager(node_discovery)
    return _cluster_manager


def set_cluster_manager(cluster_manager: ClusterManager):
    """设置全局集群管理器实例"""
    global _cluster_manager
    _cluster_manager = cluster_manager
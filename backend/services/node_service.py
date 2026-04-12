#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""节点管理服务

提供节点管理的核心业务逻辑，包括：
- 节点注册与注销
- 心跳处理
- 健康检查
- 资源调度
- 集群管理
"""

import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ==================== 配置 ====================

@dataclass
class NodeServiceConfig:
    """节点服务配置"""
    heartbeat_timeout_seconds: int = 60          # 心跳超时时间
    health_check_interval_seconds: int = 30      # 健康检查间隔
    max_health_check_failures: int = 3           # 最大健康检查失败次数
    node_offline_threshold_seconds: int = 120    # 节点离线阈值
    heartbeat_history_max: int = 1000            # 心跳历史最大数量
    event_history_max: int = 5000                # 事件历史最大数量
    cleanup_interval_hours: int = 24             # 清理间隔


# ==================== 节点服务 ====================

class NodeService:
    """节点管理服务
    
    提供完整的节点管理功能
    """
    
    def __init__(self, config: NodeServiceConfig = None, use_memory: bool = True):
        """初始化服务
        
        Args:
            config: 服务配置
            use_memory: 是否使用内存存储
        """
        self.config = config or NodeServiceConfig()
        self._use_memory = use_memory
        self._lock = threading.RLock()
        
        # 初始化仓库
        from backend.repositories.node_repository import get_node_repository
        self._repo = get_node_repository(use_memory=use_memory)
        
        # 后台任务
        self._running = False
        self._health_check_thread: Optional[threading.Thread] = None
        
        logger.info("NodeService initialized")
    
    # ==========================================================================
    # 节点注册与管理
    # ==========================================================================
    
    def register_node(
        self,
        node_id: str,
        hostname: str,
        ip_address: str,
        port: int = 22,
        node_type: str = 'worker',
        node_role: str = 'general',
        cpu_count: int = 0,
        memory_total_mb: int = 0,
        disk_total_mb: int = 0,
        gpu_info: List[Dict[str, Any]] = None,
        labels: Dict[str, str] = None,
        annotations: Dict[str, str] = None,
        tenant_id: str = None,
        max_tasks: int = 1
    ) -> Dict[str, Any]:
        """注册节点
        
        Args:
            node_id: 节点ID
            hostname: 主机名
            ip_address: IP地址
            port: 端口
            node_type: 节点类型
            node_role: 节点角色
            cpu_count: CPU核心数
            memory_total_mb: 总内存(MB)
            disk_total_mb: 总磁盘(MB)
            gpu_info: GPU信息列表
            labels: 标签
            annotations: 注解
            tenant_id: 租户ID
            max_tasks: 最大任务数
        
        Returns:
            注册结果
        """
        with self._lock:
            # 检查节点是否已存在
            existing = self._repo.nodes.get_by_node_id(node_id)
            if existing:
                # 更新已存在的节点
                return self.update_node(
                    node_id=node_id,
                    hostname=hostname,
                    ip_address=ip_address,
                    port=port,
                    cpu_count=cpu_count,
                    memory_total_mb=memory_total_mb,
                    disk_total_mb=disk_total_mb,
                    gpu_info=gpu_info,
                    labels=labels,
                    annotations=annotations
                )
            
            # 创建新节点
            node_data = {
                'id': f"node_{uuid.uuid4().hex[:12]}",
                'node_id': node_id,
                'hostname': hostname,
                'ip_address': ip_address,
                'port': port,
                'node_type': node_type,
                'node_role': node_role,
                'status': 'healthy',
                'is_schedulable': True,
                'is_ready': True,
                'cpu_count': cpu_count,
                'memory_total_mb': memory_total_mb,
                'disk_total_mb': disk_total_mb,
                'gpu_count': len(gpu_info) if gpu_info else 0,
                'gpu_info': gpu_info or [],
                'labels': labels or {},
                'annotations': annotations or {},
                'tenant_id': tenant_id,
                'max_tasks': max_tasks,
                'running_tasks': 0,
                'health_check_failures': 0,
            }
            
            node = self._repo.register_node(node_data)
            
            # 注册GPU
            if gpu_info:
                for gpu in gpu_info:
                    gpu['node_id'] = node_id
                    self._repo.gpus.update_gpu(node_id, gpu)
            
            logger.info(f"Node registered: {node_id} ({hostname})")
            
            return {
                'success': True,
                'message': 'Node registered successfully',
                'node': node
            }
    
    def update_node(
        self,
        node_id: str,
        **update_data
    ) -> Dict[str, Any]:
        """更新节点信息
        
        Args:
            node_id: 节点ID
            **update_data: 要更新的字段
        
        Returns:
            更新结果
        """
        with self._lock:
            node = self._repo.nodes.get_by_node_id(node_id)
            if not node:
                return {
                    'success': False,
                    'message': f'Node not found: {node_id}'
                }
            
            # 过滤允许更新的字段
            allowed_fields = {
                'hostname', 'ip_address', 'port', 'node_type', 'node_role',
                'cpu_count', 'memory_total_mb', 'disk_total_mb',
                'gpu_info', 'labels', 'annotations', 'is_schedulable',
                'max_tasks'
            }
            
            filtered_data = {k: v for k, v in update_data.items() if k in allowed_fields and v is not None}
            
            # 更新GPU数量
            if 'gpu_info' in filtered_data:
                filtered_data['gpu_count'] = len(filtered_data['gpu_info'])
                for gpu in filtered_data['gpu_info']:
                    gpu['node_id'] = node_id
                    self._repo.gpus.update_gpu(node_id, gpu)
            
            # 更新标签（合并）
            if 'labels' in filtered_data and node.get('labels'):
                merged_labels = node['labels'].copy()
                merged_labels.update(filtered_data['labels'])
                filtered_data['labels'] = merged_labels
            
            # 更新注解（合并）
            if 'annotations' in filtered_data and node.get('annotations'):
                merged_annotations = node['annotations'].copy()
                merged_annotations.update(filtered_data['annotations'])
                filtered_data['annotations'] = merged_annotations
            
            updated_node = self._repo.nodes.update(node_id, filtered_data)
            
            logger.info(f"Node updated: {node_id}")
            
            return {
                'success': True,
                'message': 'Node updated successfully',
                'node': updated_node
            }
    
    def unregister_node(self, node_id: str, reason: str = 'manual') -> Dict[str, Any]:
        """注销节点
        
        Args:
            node_id: 节点ID
            reason: 注销原因
        
        Returns:
            注销结果
        """
        with self._lock:
            node = self._repo.nodes.get_by_node_id(node_id)
            if not node:
                return {
                    'success': False,
                    'message': f'Node not found: {node_id}'
                }
            
            # 检查是否有运行中的任务
            if node.get('running_tasks', 0) > 0:
                return {
                    'success': False,
                    'message': f'Node has {node["running_tasks"]} running tasks, cannot unregister'
                }
            
            # 创建注销事件
            self._repo.events.create({
                'node_id': node_id,
                'event_type': 'NodeUnregistered',
                'event_reason': reason,
                'event_message': f"Node {node['hostname']} unregistered: {reason}",
                'severity': 'info',
                'source': 'node_service'
            })
            
            # 删除节点
            self._repo.nodes.delete(node_id)
            
            logger.info(f"Node unregistered: {node_id} (reason: {reason})")
            
            return {
                'success': True,
                'message': 'Node unregistered successfully'
            }
    
    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取节点详情
        
        Args:
            node_id: 节点ID
        
        Returns:
            节点信息
        """
        node = self._repo.nodes.get_by_node_id(node_id)
        if not node:
            return None
        
        # 添加GPU详细信息
        node['gpus'] = self._repo.gpus.get_gpus(node_id)
        
        # 添加最近心跳
        node['latest_heartbeat'] = self._repo.heartbeats.get_latest(node_id)
        
        # 添加最近事件
        node['recent_events'] = self._repo.events.get_recent_events(node_id, limit=10)
        
        return node
    
    def list_nodes(
        self,
        tenant_id: str = None,
        status: str = None,
        node_type: str = None,
        node_role: str = None,
        labels: Dict[str, str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """列出节点
        
        Args:
            tenant_id: 租户ID
            status: 状态过滤
            node_type: 类型过滤
            node_role: 角色过滤
            labels: 标签过滤
            limit: 返回数量
            offset: 偏移量
        
        Returns:
            节点列表
        """
        nodes, total = self._repo.nodes.list_nodes(
            tenant_id=tenant_id,
            status=status,
            node_type=node_type,
            limit=limit,
            offset=offset
        )
        
        # 标签过滤
        if labels:
            filtered = []
            for node in nodes:
                node_labels = node.get('labels', {})
                match = all(
                    node_labels.get(k) == v for k, v in labels.items()
                )
                if match:
                    filtered.append(node)
            nodes = filtered
            total = len(filtered)
        
        # 角色过滤
        if node_role:
            nodes = [n for n in nodes if n.get('node_role') == node_role]
            total = len(nodes)
        
        return {
            'nodes': nodes,
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    # ==========================================================================
    # 心跳处理
    # ==========================================================================
    
    def process_heartbeat(
        self,
        node_id: str,
        hostname: str = None,
        ip_address: str = None,
        port: int = None,
        cpu_count: int = None,
        memory_total_mb: int = None,
        disk_total_mb: int = None,
        cpu_utilization: float = None,
        memory_used_mb: int = None,
        memory_utilization: float = None,
        disk_used_mb: int = None,
        disk_utilization: float = None,
        gpu_info: List[Dict[str, Any]] = None,
        running_tasks: int = None,
        labels: Dict[str, str] = None,
        annotations: Dict[str, str] = None,
        metrics: Dict[str, Any] = None,
        tenant_id: str = None
    ) -> Dict[str, Any]:
        """处理节点心跳
        
        Args:
            node_id: 节点ID
            ... 其他参数
        
        Returns:
            处理结果
        """
        start_time = time.time()
        
        with self._lock:
            # 获取或注册节点
            node = self._repo.nodes.get_by_node_id(node_id)
            
            if not node:
                # 自动注册新节点
                if hostname:
                    result = self.register_node(
                        node_id=node_id,
                        hostname=hostname,
                        ip_address=ip_address or '',
                        port=port or 22,
                        cpu_count=cpu_count or 0,
                        memory_total_mb=memory_total_mb or 0,
                        disk_total_mb=disk_total_mb or 0,
                        gpu_info=gpu_info,
                        labels=labels,
                        annotations=annotations,
                        tenant_id=tenant_id
                    )
                    if not result['success']:
                        return result
                    node = result['node']
                else:
                    return {
                        'success': False,
                        'message': f'Node not found and hostname not provided: {node_id}'
                    }
            
            # 构建心跳指标
            heartbeat_metrics = {
                'cpu_utilization': cpu_utilization or 0,
                'memory_utilization': memory_utilization or 0,
                'disk_utilization': disk_utilization or 0,
                'memory_used_mb': memory_used_mb or 0,
                'disk_used_mb': disk_used_mb or 0,
                'gpu_utilization': [g.get('memory_utilization', 0) for g in (gpu_info or [])],
                'running_tasks': running_tasks or 0,
                'gpus': gpu_info or [],
            }
            
            if metrics:
                heartbeat_metrics.update(metrics)
            
            # 更新心跳
            self._repo.update_heartbeat(node_id, heartbeat_metrics)
            
            # 更新节点硬件信息（如果有变化）
            update_data = {}
            if hostname and hostname != node.get('hostname'):
                update_data['hostname'] = hostname
            if ip_address and ip_address != node.get('ip_address'):
                update_data['ip_address'] = ip_address
            if port and port != node.get('port'):
                update_data['port'] = port
            if cpu_count and cpu_count != node.get('cpu_count'):
                update_data['cpu_count'] = cpu_count
            if memory_total_mb and memory_total_mb != node.get('memory_total_mb'):
                update_data['memory_total_mb'] = memory_total_mb
            if disk_total_mb and disk_total_mb != node.get('disk_total_mb'):
                update_data['disk_total_mb'] = disk_total_mb
            if labels:
                merged = node.get('labels', {}).copy()
                merged.update(labels)
                update_data['labels'] = merged
            if annotations:
                merged = node.get('annotations', {}).copy()
                merged.update(annotations)
                update_data['annotations'] = merged
            
            if update_data:
                self._repo.nodes.update(node_id, update_data)
            
            latency_ms = (time.time() - start_time) * 1000
            
            logger.debug(f"Heartbeat processed for {node_id} in {latency_ms:.2f}ms")
            
            return {
                'success': True,
                'message': 'Heartbeat accepted',
                'latency_ms': latency_ms,
                'node_status': 'healthy'
            }
    
    def get_heartbeat_history(
        self,
        node_id: str,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取心跳历史
        
        Args:
            node_id: 节点ID
            start_time: 开始时间
            end_time: 结束时间
            limit: 返回数量
        
        Returns:
            心跳历史列表
        """
        return self._repo.heartbeats.get_history(
            node_id=node_id,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    # ==========================================================================
    # 健康检查
    # ==========================================================================
    
    def check_node_health(self, node_id: str) -> Dict[str, Any]:
        """检查节点健康状态
        
        Args:
            node_id: 节点ID
        
        Returns:
            健康检查结果
        """
        node = self._repo.nodes.get_by_node_id(node_id)
        if not node:
            return {
                'healthy': False,
                'reason': 'Node not found'
            }
        
        issues = []
        
        # 检查心跳
        last_heartbeat = node.get('last_heartbeat')
        if last_heartbeat:
            if isinstance(last_heartbeat, str):
                last_heartbeat = datetime.fromisoformat(last_heartbeat)
            
            seconds_since_heartbeat = (datetime.utcnow() - last_heartbeat).total_seconds()
            
            if seconds_since_heartbeat > self.config.heartbeat_timeout_seconds:
                issues.append(f'Heartbeat timeout: {seconds_since_heartbeat:.0f}s since last heartbeat')
        else:
            issues.append('No heartbeat received')
        
        # 检查资源使用
        cpu_util = node.get('cpu_utilization', 0)
        if cpu_util > 95:
            issues.append(f'High CPU utilization: {cpu_util:.1f}%')
        
        mem_util = node.get('memory_utilization', 0)
        if mem_util > 95:
            issues.append(f'High memory utilization: {mem_util:.1f}%')
        
        disk_util = node.get('disk_utilization', 0)
        if disk_util > 95:
            issues.append(f'High disk utilization: {disk_util:.1f}%')
        
        # 检查健康检查失败次数
        failures = node.get('health_check_failures', 0)
        if failures >= self.config.max_health_check_failures:
            issues.append(f'Too many health check failures: {failures}')
        
        is_healthy = len(issues) == 0
        
        return {
            'healthy': is_healthy,
            'status': 'healthy' if is_healthy else 'unhealthy',
            'issues': issues,
            'last_heartbeat': node.get('last_heartbeat'),
            'health_check_failures': failures,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def run_health_checks(self) -> Dict[str, Any]:
        """运行所有节点的健康检查
        
        Returns:
            健康检查结果摘要
        """
        nodes, _ = self._repo.nodes.list_nodes(limit=10000)
        
        results = {
            'total': len(nodes),
            'healthy': 0,
            'unhealthy': 0,
            'offline': 0,
            'checked_at': datetime.utcnow().isoformat(),
            'issues': []
        }
        
        for node in nodes:
            node_id = node['node_id']
            health = self.check_node_health(node_id)
            
            if health['healthy']:
                results['healthy'] += 1
            else:
                results['unhealthy'] += 1
                
                # 检查是否应该标记为离线
                last_heartbeat = node.get('last_heartbeat')
                if last_heartbeat:
                    if isinstance(last_heartbeat, str):
                        last_heartbeat = datetime.fromisoformat(last_heartbeat)
                    
                    if (datetime.utcnow() - last_heartbeat).total_seconds() > self.config.node_offline_threshold_seconds:
                        self._repo.mark_node_offline(node_id, 'heartbeat timeout')
                        results['offline'] += 1
                
                results['issues'].append({
                    'node_id': node_id,
                    'hostname': node.get('hostname'),
                    'issues': health['issues']
                })
        
        return results
    
    # ==========================================================================
    # 节点状态管理
    # ==========================================================================
    
    def cordon_node(self, node_id: str, reason: str = 'manual') -> Dict[str, Any]:
        """隔离节点（阻止新任务调度）
        
        Args:
            node_id: 节点ID
            reason: 隔离原因
        
        Returns:
            操作结果
        """
        node = self._repo.nodes.get_by_node_id(node_id)
        if not node:
            return {'success': False, 'message': 'Node not found'}
        
        self._repo.nodes.update(node_id, {
            'is_schedulable': False,
            'status': 'cordoned'
        })
        
        self._repo.events.create({
            'node_id': node_id,
            'event_type': 'NodeCordoned',
            'event_reason': reason,
            'event_message': f"Node {node['hostname']} cordoned: {reason}",
            'severity': 'info',
            'source': 'node_service'
        })
        
        logger.info(f"Node cordoned: {node_id}")
        
        return {'success': True, 'message': 'Node cordoned successfully'}
    
    def uncordon_node(self, node_id: str) -> Dict[str, Any]:
        """解除节点隔离
        
        Args:
            node_id: 节点ID
        
        Returns:
            操作结果
        """
        node = self._repo.nodes.get_by_node_id(node_id)
        if not node:
            return {'success': False, 'message': 'Node not found'}
        
        self._repo.nodes.update(node_id, {
            'is_schedulable': True,
            'status': 'healthy'
        })
        
        self._repo.events.create({
            'node_id': node_id,
            'event_type': 'NodeUncordoned',
            'event_message': f"Node {node['hostname']} uncordoned",
            'severity': 'info',
            'source': 'node_service'
        })
        
        logger.info(f"Node uncordoned: {node_id}")
        
        return {'success': True, 'message': 'Node uncordoned successfully'}
    
    def drain_node(self, node_id: str, force: bool = False) -> Dict[str, Any]:
        """排空节点（迁移任务并阻止调度）
        
        Args:
            node_id: 节点ID
            force: 是否强制排空
        
        Returns:
            操作结果
        """
        node = self._repo.nodes.get_by_node_id(node_id)
        if not node:
            return {'success': False, 'message': 'Node not found'}
        
        running_tasks = node.get('running_tasks', 0)
        
        if running_tasks > 0 and not force:
            return {
                'success': False,
                'message': f'Node has {running_tasks} running tasks. Use force=true to drain anyway.'
            }
        
        self._repo.nodes.update(node_id, {
            'is_schedulable': False,
            'status': 'draining'
        })
        
        self._repo.events.create({
            'node_id': node_id,
            'event_type': 'NodeDraining',
            'event_message': f"Node {node['hostname']} is being drained (force={force})",
            'severity': 'warning',
            'source': 'node_service'
        })
        
        logger.info(f"Node draining started: {node_id}")
        
        return {
            'success': True,
            'message': 'Node drain started',
            'running_tasks': running_tasks
        }
    
    # ==========================================================================
    # 集群状态
    # ==========================================================================
    
    def get_cluster_summary(self, tenant_id: str = None) -> Dict[str, Any]:
        """获取集群摘要
        
        Args:
            tenant_id: 租户ID
        
        Returns:
            集群摘要
        """
        return self._repo.get_cluster_status(tenant_id)
    
    def get_schedulable_nodes(
        self,
        tenant_id: str = None,
        required_gpus: int = 0,
        required_memory_mb: int = 0,
        labels: Dict[str, str] = None
    ) -> List[Dict[str, Any]]:
        """获取可调度节点
        
        Args:
            tenant_id: 租户ID
            required_gpus: 需要的GPU数量
            required_memory_mb: 需要的内存(MB)
            labels: 需要的标签
        
        Returns:
            可调度节点列表
        """
        nodes = self._repo.nodes.get_schedulable_nodes(tenant_id)
        
        # 过滤资源需求
        filtered = []
        for node in nodes:
            # GPU检查
            if required_gpus > 0:
                available_gpus = self._repo.gpus.get_available_gpus(node['node_id'])
                if len(available_gpus) < required_gpus:
                    continue
            
            # 内存检查
            if required_memory_mb > 0:
                available_memory = node.get('memory_total_mb', 0) - node.get('memory_used_mb', 0)
                if available_memory < required_memory_mb:
                    continue
            
            # 标签检查
            if labels:
                node_labels = node.get('labels', {})
                if not all(node_labels.get(k) == v for k, v in labels.items()):
                    continue
            
            filtered.append(node)
        
        return filtered
    
    # ==========================================================================
    # 事件管理
    # ==========================================================================
    
    def get_node_events(
        self,
        node_id: str = None,
        event_type: str = None,
        severity: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取节点事件
        
        Args:
            node_id: 节点ID
            event_type: 事件类型
            severity: 严重级别
            limit: 返回数量
            offset: 偏移量
        
        Returns:
            事件列表
        """
        events, total = self._repo.events.get_events(
            node_id=node_id,
            event_type=event_type,
            severity=severity,
            limit=limit,
            offset=offset
        )
        
        return {
            'events': events,
            'total': total,
            'limit': limit,
            'offset': offset
        }
    
    def create_event(
        self,
        node_id: str,
        event_type: str,
        event_message: str,
        severity: str = 'info',
        source: str = 'api'
    ) -> Dict[str, Any]:
        """创建事件
        
        Args:
            node_id: 节点ID
            event_type: 事件类型
            event_message: 事件消息
            severity: 严重级别
            source: 来源
        
        Returns:
            创建的事件
        """
        return self._repo.events.create({
            'node_id': node_id,
            'event_type': event_type,
            'event_message': event_message,
            'severity': severity,
            'source': source
        })
    
    # ==========================================================================
    # 服务生命周期
    # ==========================================================================
    
    def start(self):
        """启动服务"""
        if self._running:
            return
        
        self._running = True
        
        # 启动健康检查线程
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
        
        logger.info("NodeService started")
    
    def stop(self):
        """停止服务"""
        if not self._running:
            return
        
        self._running = False
        
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
        
        logger.info("NodeService stopped")
    
    def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                self.run_health_checks()
            except Exception as e:
                logger.error(f"Health check failed: {e}")
            
            time.sleep(self.config.health_check_interval_seconds)


# ==================== 单例获取 ====================

_node_service: Optional[NodeService] = None
_service_lock = threading.Lock()


def get_node_service(use_memory: bool = True) -> NodeService:
    """获取节点服务实例"""
    global _node_service
    if _node_service is None:
        with _service_lock:
            if _node_service is None:
                _node_service = NodeService(use_memory=use_memory)
    return _node_service


# ==================== 导出 ====================

__all__ = [
    'NodeService',
    'NodeServiceConfig',
    'get_node_service',
]

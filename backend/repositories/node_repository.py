#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""节点管理数据访问层

提供节点、GPU、心跳、事件的CRUD操作，支持内存存储和数据库存储。
"""

import logging
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ==================== 节点仓库基类 ====================

class BaseNodeRepository(ABC):
    """节点仓库基类"""
    
    @abstractmethod
    def create(self, node_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """创建节点"""
        pass
    
    @abstractmethod
    def get_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取节点"""
        pass
    
    @abstractmethod
    def get_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """根据节点ID获取节点"""
        pass
    
    @abstractmethod
    def update(self, node_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新节点"""
        pass
    
    @abstractmethod
    def delete(self, node_id: str) -> bool:
        """删除节点"""
        pass
    
    @abstractmethod
    def list_nodes(
        self,
        tenant_id: str = None,
        status: str = None,
        node_type: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """列出节点"""
        pass


# ==================== 内存节点仓库 ====================

class InMemoryNodeRepository(BaseNodeRepository):
    """内存节点仓库"""
    
    def __init__(self):
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def create(self, node_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            if 'id' not in node_data:
                node_data['id'] = f"node_{uuid.uuid4().hex[:12]}"
            
            node_data['created_at'] = datetime.utcnow().isoformat()
            node_data['updated_at'] = datetime.utcnow().isoformat()
            
            node_id = node_data.get('node_id', node_data['id'])
            self._nodes[node_id] = node_data
            
            logger.info(f"Node created: {node_id}")
            return node_data
    
    def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            for node in self._nodes.values():
                if node.get('id') == id:
                    return node
            return None
    
    def get_by_node_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._nodes.get(node_id)
    
    def update(self, node_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            if node_id not in self._nodes:
                return None
            
            node = self._nodes[node_id]
            for key, value in update_data.items():
                if key not in ('id', 'node_id', 'created_at'):
                    node[key] = value
            
            node['updated_at'] = datetime.utcnow().isoformat()
            
            logger.debug(f"Node updated: {node_id}")
            return node
    
    def delete(self, node_id: str) -> bool:
        with self._lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                logger.info(f"Node deleted: {node_id}")
                return True
            return False
    
    def list_nodes(
        self,
        tenant_id: str = None,
        status: str = None,
        node_type: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        with self._lock:
            nodes = list(self._nodes.values())
            
            # 过滤
            if tenant_id:
                nodes = [n for n in nodes if n.get('tenant_id') == tenant_id]
            if status:
                nodes = [n for n in nodes if n.get('status') == status]
            if node_type:
                nodes = [n for n in nodes if n.get('node_type') == node_type]
            
            total = len(nodes)
            nodes = nodes[offset:offset + limit]
            
            return nodes, total
    
    def get_healthy_nodes(self, tenant_id: str = None) -> List[Dict[str, Any]]:
        """获取健康节点"""
        with self._lock:
            nodes = list(self._nodes.values())
            
            if tenant_id:
                nodes = [n for n in nodes if n.get('tenant_id') == tenant_id]
            
            healthy = []
            for node in nodes:
                if node.get('status') == 'healthy':
                    last_heartbeat = node.get('last_heartbeat')
                    if last_heartbeat:
                        if isinstance(last_heartbeat, str):
                            last_heartbeat = datetime.fromisoformat(last_heartbeat)
                        if (datetime.utcnow() - last_heartbeat).total_seconds() < 60:
                            healthy.append(node)
            
            return healthy
    
    def get_schedulable_nodes(self, tenant_id: str = None) -> List[Dict[str, Any]]:
        """获取可调度节点"""
        healthy = self.get_healthy_nodes(tenant_id)
        return [n for n in healthy if n.get('is_schedulable', True)]
    
    def get_nodes_by_label(self, label_key: str, label_value: str) -> List[Dict[str, Any]]:
        """根据标签获取节点"""
        with self._lock:
            return [
                n for n in self._nodes.values()
                if n.get('labels', {}).get(label_key) == label_value
            ]
    
    def get_cluster_summary(self, tenant_id: str = None) -> Dict[str, Any]:
        """获取集群摘要"""
        with self._lock:
            nodes = list(self._nodes.values())
            
            if tenant_id:
                nodes = [n for n in nodes if n.get('tenant_id') == tenant_id]
            
            summary = {
                'total_nodes': len(nodes),
                'healthy_nodes': 0,
                'unhealthy_nodes': 0,
                'offline_nodes': 0,
                'total_cpus': 0,
                'total_memory_mb': 0,
                'total_disk_mb': 0,
                'total_gpus': 0,
                'used_memory_mb': 0,
                'running_tasks': 0,
                'max_tasks': 0,
                'avg_cpu_utilization': 0.0,
                'avg_memory_utilization': 0.0,
            }
            
            cpu_utils = []
            mem_utils = []
            
            for node in nodes:
                status = node.get('status', 'unknown')
                if status == 'healthy':
                    summary['healthy_nodes'] += 1
                elif status == 'offline':
                    summary['offline_nodes'] += 1
                else:
                    summary['unhealthy_nodes'] += 1
                
                summary['total_cpus'] += node.get('cpu_count', 0)
                summary['total_memory_mb'] += node.get('memory_total_mb', 0)
                summary['total_disk_mb'] += node.get('disk_total_mb', 0)
                summary['total_gpus'] += node.get('gpu_count', 0)
                summary['used_memory_mb'] += node.get('memory_used_mb', 0)
                summary['running_tasks'] += node.get('running_tasks', 0)
                summary['max_tasks'] += node.get('max_tasks', 1)
                
                cpu_utils.append(node.get('cpu_utilization', 0))
                mem_utils.append(node.get('memory_utilization', 0))
            
            if cpu_utils:
                summary['avg_cpu_utilization'] = sum(cpu_utils) / len(cpu_utils)
            if mem_utils:
                summary['avg_memory_utilization'] = sum(mem_utils) / len(mem_utils)
            
            summary['timestamp'] = datetime.utcnow().isoformat()
            
            return summary


# ==================== 心跳仓库 ====================

class InMemoryHeartbeatRepository:
    """内存心跳仓库"""
    
    def __init__(self, max_history: int = 1000):
        self._heartbeats: Dict[str, List[Dict[str, Any]]] = {}
        self._max_history = max_history
        self._lock = threading.RLock()
    
    def record(self, heartbeat: Dict[str, Any]) -> Dict[str, Any]:
        """记录心跳"""
        with self._lock:
            if 'id' not in heartbeat:
                heartbeat['id'] = f"hb_{uuid.uuid4().hex[:12]}"
            if 'timestamp' not in heartbeat:
                heartbeat['timestamp'] = datetime.utcnow().isoformat()
            
            node_id = heartbeat.get('node_id')
            if node_id not in self._heartbeats:
                self._heartbeats[node_id] = []
            
            self._heartbeats[node_id].append(heartbeat)
            
            # 限制历史记录数量
            if len(self._heartbeats[node_id]) > self._max_history:
                self._heartbeats[node_id] = self._heartbeats[node_id][-self._max_history:]
            
            return heartbeat
    
    def get_latest(self, node_id: str) -> Optional[Dict[str, Any]]:
        """获取最新心跳"""
        with self._lock:
            if node_id in self._heartbeats and self._heartbeats[node_id]:
                return self._heartbeats[node_id][-1]
            return None
    
    def get_history(
        self,
        node_id: str,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取心跳历史"""
        with self._lock:
            if node_id not in self._heartbeats:
                return []
            
            heartbeats = self._heartbeats[node_id]
            
            if start_time or end_time:
                filtered = []
                for hb in heartbeats:
                    ts = hb.get('timestamp')
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts)
                    
                    if start_time and ts < start_time:
                        continue
                    if end_time and ts > end_time:
                        continue
                    
                    filtered.append(hb)
                heartbeats = filtered
            
            return heartbeats[-limit:]
    
    def get_missed_heartbeats(self, timeout_seconds: int = 60) -> List[str]:
        """获取错过心跳的节点"""
        with self._lock:
            missed = []
            threshold = datetime.utcnow() - timedelta(seconds=timeout_seconds)
            
            for node_id, heartbeats in self._heartbeats.items():
                if not heartbeats:
                    missed.append(node_id)
                    continue
                
                last_hb = heartbeats[-1]
                ts = last_hb.get('timestamp')
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                
                if ts < threshold:
                    missed.append(node_id)
            
            return missed
    
    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """清理旧心跳记录"""
        with self._lock:
            threshold = datetime.utcnow() - timedelta(hours=max_age_hours)
            removed = 0
            
            for node_id in list(self._heartbeats.keys()):
                original_count = len(self._heartbeats[node_id])
                self._heartbeats[node_id] = [
                    hb for hb in self._heartbeats[node_id]
                    if datetime.fromisoformat(hb['timestamp']) > threshold
                ]
                removed += original_count - len(self._heartbeats[node_id])
            
            return removed


# ==================== 事件仓库 ====================

class InMemoryEventRepository:
    """内存事件仓库"""
    
    def __init__(self, max_events: int = 5000):
        self._events: Dict[str, List[Dict[str, Any]]] = {}
        self._max_events = max_events
        self._lock = threading.RLock()
    
    def create(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """创建事件"""
        with self._lock:
            if 'id' not in event:
                event['id'] = f"evt_{uuid.uuid4().hex[:12]}"
            if 'timestamp' not in event:
                event['timestamp'] = datetime.utcnow().isoformat()
            
            node_id = event.get('node_id')
            if node_id not in self._events:
                self._events[node_id] = []
            
            self._events[node_id].append(event)
            
            # 限制事件数量
            if len(self._events[node_id]) > self._max_events:
                self._events[node_id] = self._events[node_id][-self._max_events:]
            
            logger.info(f"Event created for node {node_id}: {event.get('event_type')}")
            return event
    
    def get_events(
        self,
        node_id: str = None,
        event_type: str = None,
        severity: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取事件列表"""
        with self._lock:
            if node_id:
                events = self._events.get(node_id, [])
            else:
                events = []
                for node_events in self._events.values():
                    events.extend(node_events)
            
            # 过滤
            if event_type:
                events = [e for e in events if e.get('event_type') == event_type]
            if severity:
                events = [e for e in events if e.get('severity') == severity]
            if start_time:
                events = [
                    e for e in events
                    if datetime.fromisoformat(e['timestamp']) >= start_time
                ]
            if end_time:
                events = [
                    e for e in events
                    if datetime.fromisoformat(e['timestamp']) <= end_time
                ]
            
            # 按时间倒序
            events.sort(key=lambda x: x['timestamp'], reverse=True)
            
            total = len(events)
            events = events[offset:offset + limit]
            
            return events, total
    
    def get_recent_events(self, node_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近事件"""
        with self._lock:
            if node_id not in self._events:
                return []
            
            events = self._events[node_id][-limit:]
            events.reverse()
            return events
    
    def count_by_severity(self, node_id: str = None, hours: int = 24) -> Dict[str, int]:
        """按严重级别统计事件"""
        with self._lock:
            threshold = datetime.utcnow() - timedelta(hours=hours)
            
            if node_id:
                events = self._events.get(node_id, [])
            else:
                events = []
                for node_events in self._events.values():
                    events.extend(node_events)
            
            counts = {'info': 0, 'warning': 0, 'error': 0, 'critical': 0}
            
            for event in events:
                ts = datetime.fromisoformat(event['timestamp'])
                if ts >= threshold:
                    severity = event.get('severity', 'info')
                    counts[severity] = counts.get(severity, 0) + 1
            
            return counts


# ==================== GPU仓库 ====================

class InMemoryGPURepository:
    """内存GPU仓库"""
    
    def __init__(self):
        self._gpus: Dict[str, Dict[str, Dict[str, Any]]] = {}  # node_id -> gpu_uuid -> gpu_info
        self._lock = threading.RLock()
    
    def update_gpu(self, node_id: str, gpu_info: Dict[str, Any]) -> Dict[str, Any]:
        """更新GPU信息"""
        with self._lock:
            if node_id not in self._gpus:
                self._gpus[node_id] = {}
            
            gpu_uuid = gpu_info.get('gpu_uuid')
            if not gpu_uuid:
                gpu_uuid = f"gpu_{uuid.uuid4().hex[:12]}"
                gpu_info['gpu_uuid'] = gpu_uuid
            
            gpu_info['updated_at'] = datetime.utcnow().isoformat()
            self._gpus[node_id][gpu_uuid] = gpu_info
            
            return gpu_info
    
    def get_gpus(self, node_id: str) -> List[Dict[str, Any]]:
        """获取节点的GPU列表"""
        with self._lock:
            if node_id not in self._gpus:
                return []
            return list(self._gpus[node_id].values())
    
    def get_gpu(self, node_id: str, gpu_uuid: str) -> Optional[Dict[str, Any]]:
        """获取特定GPU"""
        with self._lock:
            if node_id not in self._gpus:
                return None
            return self._gpus[node_id].get(gpu_uuid)
    
    def get_available_gpus(self, node_id: str = None) -> List[Dict[str, Any]]:
        """获取可用GPU"""
        with self._lock:
            available = []
            
            nodes = [node_id] if node_id else list(self._gpus.keys())
            
            for nid in nodes:
                if nid in self._gpus:
                    for gpu in self._gpus[nid].values():
                        if gpu.get('is_available', True) and gpu.get('is_healthy', True):
                            available.append(gpu)
            
            return available
    
    def get_cluster_gpu_summary(self) -> Dict[str, Any]:
        """获取集群GPU摘要"""
        with self._lock:
            summary = {
                'total_gpus': 0,
                'available_gpus': 0,
                'total_memory_mb': 0,
                'used_memory_mb': 0,
                'avg_utilization': 0.0,
                'avg_temperature': 0.0,
            }
            
            temps = []
            utils = []
            
            for node_gpus in self._gpus.values():
                for gpu in node_gpus.values():
                    summary['total_gpus'] += 1
                    if gpu.get('is_available', True):
                        summary['available_gpus'] += 1
                    summary['total_memory_mb'] += gpu.get('memory_total_mb', 0)
                    summary['used_memory_mb'] += gpu.get('memory_used_mb', 0)
                    
                    if 'temperature_celsius' in gpu:
                        temps.append(gpu['temperature_celsius'])
                    if 'memory_utilization' in gpu:
                        utils.append(gpu['memory_utilization'])
            
            if temps:
                summary['avg_temperature'] = sum(temps) / len(temps)
            if utils:
                summary['avg_utilization'] = sum(utils) / len(utils)
            
            return summary


# ==================== 统一节点仓库 ====================

class NodeRepository:
    """统一节点仓库
    
    提供节点、GPU、心跳、事件的统一访问接口
    """
    
    def __init__(self, use_memory: bool = True):
        self._use_memory = use_memory
        
        if use_memory:
            self._node_repo = InMemoryNodeRepository()
            self._heartbeat_repo = InMemoryHeartbeatRepository()
            self._event_repo = InMemoryEventRepository()
            self._gpu_repo = InMemoryGPURepository()
        else:
            # TODO: 实现数据库存储
            self._node_repo = InMemoryNodeRepository()
            self._heartbeat_repo = InMemoryHeartbeatRepository()
            self._event_repo = InMemoryEventRepository()
            self._gpu_repo = InMemoryGPURepository()
        
        logger.info(f"NodeRepository initialized (memory={use_memory})")
    
    # 节点操作
    @property
    def nodes(self) -> InMemoryNodeRepository:
        return self._node_repo
    
    # 心跳操作
    @property
    def heartbeats(self) -> InMemoryHeartbeatRepository:
        return self._heartbeat_repo
    
    # 事件操作
    @property
    def events(self) -> InMemoryEventRepository:
        return self._event_repo
    
    # GPU操作
    @property
    def gpus(self) -> InMemoryGPURepository:
        return self._gpu_repo
    
    # 便捷方法
    def register_node(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """注册节点"""
        node_data['status'] = 'healthy'
        node_data['registered_at'] = datetime.utcnow().isoformat()
        node_data['last_heartbeat'] = datetime.utcnow().isoformat()
        
        node = self._node_repo.create(node_data)
        
        # 创建注册事件
        self._event_repo.create({
            'node_id': node['node_id'],
            'event_type': 'NodeRegistered',
            'event_message': f"Node {node['hostname']} registered",
            'severity': 'info',
            'source': 'node_repository'
        })
        
        return node
    
    def update_heartbeat(self, node_id: str, metrics: Dict[str, Any]) -> bool:
        """更新心跳"""
        node = self._node_repo.get_by_node_id(node_id)
        if not node:
            return False
        
        # 记录心跳
        heartbeat = {
            'node_id': node_id,
            'cpu_utilization': metrics.get('cpu_utilization', 0),
            'memory_utilization': metrics.get('memory_utilization', 0),
            'disk_utilization': metrics.get('disk_utilization', 0),
            'gpu_utilization': metrics.get('gpu_utilization', []),
            'running_tasks': metrics.get('running_tasks', 0),
            'status': 'normal'
        }
        self._heartbeat_repo.record(heartbeat)
        
        # 更新节点
        update_data = {
            'last_heartbeat': datetime.utcnow().isoformat(),
            'status': 'healthy',
            'health_check_failures': 0,
            'cpu_utilization': metrics.get('cpu_utilization', node.get('cpu_utilization', 0)),
            'memory_used_mb': metrics.get('memory_used_mb', node.get('memory_used_mb', 0)),
            'memory_utilization': metrics.get('memory_utilization', node.get('memory_utilization', 0)),
            'disk_used_mb': metrics.get('disk_used_mb', node.get('disk_used_mb', 0)),
            'disk_utilization': metrics.get('disk_utilization', node.get('disk_utilization', 0)),
            'running_tasks': metrics.get('running_tasks', node.get('running_tasks', 0)),
        }
        self._node_repo.update(node_id, update_data)
        
        # 更新GPU信息
        for gpu_info in metrics.get('gpus', []):
            self._gpu_repo.update_gpu(node_id, gpu_info)
        
        return True
    
    def mark_node_offline(self, node_id: str, reason: str = 'heartbeat timeout') -> bool:
        """标记节点离线"""
        node = self._node_repo.get_by_node_id(node_id)
        if not node:
            return False
        
        self._node_repo.update(node_id, {
            'status': 'offline',
            'is_ready': False
        })
        
        self._event_repo.create({
            'node_id': node_id,
            'event_type': 'NodeOffline',
            'event_reason': reason,
            'event_message': f"Node {node['hostname']} marked as offline: {reason}",
            'severity': 'warning',
            'source': 'node_repository'
        })
        
        return True
    
    def get_cluster_status(self, tenant_id: str = None) -> Dict[str, Any]:
        """获取集群状态"""
        node_summary = self._node_repo.get_cluster_summary(tenant_id)
        gpu_summary = self._gpu_repo.get_cluster_gpu_summary()
        event_counts = self._event_repo.count_by_severity(hours=24)
        missed_heartbeats = self._heartbeat_repo.get_missed_heartbeats()
        
        return {
            'nodes': node_summary,
            'gpus': gpu_summary,
            'events_24h': event_counts,
            'nodes_with_missed_heartbeats': len(missed_heartbeats),
            'timestamp': datetime.utcnow().isoformat()
        }


# ==================== 单例获取 ====================

_node_repository: Optional[NodeRepository] = None
_repo_lock = threading.Lock()


def get_node_repository(use_memory: bool = True) -> NodeRepository:
    """获取节点仓库实例"""
    global _node_repository
    if _node_repository is None:
        with _repo_lock:
            if _node_repository is None:
                _node_repository = NodeRepository(use_memory=use_memory)
    return _node_repository


# ==================== 导出 ====================

__all__ = [
    'NodeRepository',
    'InMemoryNodeRepository',
    'InMemoryHeartbeatRepository',
    'InMemoryEventRepository',
    'InMemoryGPURepository',
    'get_node_repository',
]

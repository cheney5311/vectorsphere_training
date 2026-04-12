#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU 资源仓库层

提供 GPU 节点、设备、分配记录等数据的持久化存储。
支持内存存储和数据库存储两种模式。
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class GPUNodeRepository:
    """GPU 节点仓库"""
    
    def __init__(self, use_memory: bool = True, db_session=None):
        self._use_memory = use_memory
        self._db_session = db_session
        self._memory_store: Dict[str, Dict] = {}
    
    def _generate_id(self) -> str:
        return f"node_{uuid.uuid4().hex[:12]}"
    
    def create(self, node_data: Dict[str, Any]) -> Optional[Dict]:
        """创建节点"""
        try:
            node_id = node_data.get('id') or self._generate_id()
            now = datetime.utcnow()
            
            node = {
                'id': node_id,
                'tenant_id': node_data.get('tenant_id'),
                'hostname': node_data.get('hostname'),
                'ip_address': node_data.get('ip_address'),
                'port': node_data.get('port', 8080),
                'status': node_data.get('status', 'online'),
                'is_healthy': node_data.get('is_healthy', True),
                'last_heartbeat': node_data.get('last_heartbeat', now),
                'gpu_count': node_data.get('gpu_count', 0),
                'total_gpu_memory_mb': node_data.get('total_gpu_memory_mb', 0),
                'used_gpu_memory_mb': node_data.get('used_gpu_memory_mb', 0),
                'cpu_cores': node_data.get('cpu_cores', 0),
                'cpu_used': node_data.get('cpu_used', 0.0),
                'memory_total_mb': node_data.get('memory_total_mb', 0),
                'memory_used_mb': node_data.get('memory_used_mb', 0),
                'disk_total_mb': node_data.get('disk_total_mb', 0),
                'disk_used_mb': node_data.get('disk_used_mb', 0),
                'network_bandwidth_mbps': node_data.get('network_bandwidth_mbps', 1000),
                'labels': node_data.get('labels', {}),
                'capabilities': node_data.get('capabilities', []),
                'metadata': node_data.get('metadata', {}),
                'created_at': now,
                'updated_at': now
            }
            
            if self._use_memory:
                self._memory_store[node_id] = node
                return node
            else:
                return self._create_db(node)
                
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            return None
    
    def _create_db(self, node: Dict) -> Optional[Dict]:
        """数据库创建"""
        try:
            from backend.schemas.gpu_resource_models import GPUNodeModel, NodeStatusEnum
            
            model = GPUNodeModel(
                id=node['id'],
                tenant_id=node['tenant_id'],
                hostname=node['hostname'],
                ip_address=node['ip_address'],
                port=node['port'],
                status=NodeStatusEnum(node['status']) if node['status'] else NodeStatusEnum.ONLINE,
                is_healthy=node['is_healthy'],
                last_heartbeat=node['last_heartbeat'],
                gpu_count=node['gpu_count'],
                total_gpu_memory_mb=node['total_gpu_memory_mb'],
                used_gpu_memory_mb=node['used_gpu_memory_mb'],
                cpu_cores=node['cpu_cores'],
                cpu_used=node['cpu_used'],
                memory_total_mb=node['memory_total_mb'],
                memory_used_mb=node['memory_used_mb'],
                disk_total_mb=node['disk_total_mb'],
                disk_used_mb=node['disk_used_mb'],
                network_bandwidth_mbps=node['network_bandwidth_mbps'],
                labels=node['labels'],
                capabilities=node['capabilities'],
                node_metadata=node['metadata'],
                created_at=node['created_at'],
                updated_at=node['updated_at']
            )
            
            self._db_session.add(model)
            self._db_session.commit()
            return model.to_dict()
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB create node failed: {e}")
            return None
    
    def get_by_id(self, node_id: str, tenant_id: str = None) -> Optional[Dict]:
        """根据 ID 获取节点"""
        try:
            if self._use_memory:
                node = self._memory_store.get(node_id)
                if node and (tenant_id is None or node.get('tenant_id') == tenant_id):
                    return node
                return None
            else:
                return self._get_by_id_db(node_id, tenant_id)
                
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            return None
    
    def _get_by_id_db(self, node_id: str, tenant_id: str = None) -> Optional[Dict]:
        """数据库查询"""
        try:
            from backend.schemas.gpu_resource_models import GPUNodeModel
            
            query = self._db_session.query(GPUNodeModel).filter_by(id=node_id)
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            
            model = query.first()
            return model.to_dict() if model else None
            
        except Exception as e:
            logger.error(f"DB get node failed: {e}")
            return None
    
    def get_all(
        self,
        tenant_id: str = None,
        status: str = None,
        is_healthy: bool = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """获取节点列表"""
        try:
            if self._use_memory:
                nodes = list(self._memory_store.values())
                
                if tenant_id:
                    nodes = [n for n in nodes if n.get('tenant_id') == tenant_id]
                if status:
                    nodes = [n for n in nodes if n.get('status') == status]
                if is_healthy is not None:
                    nodes = [n for n in nodes if n.get('is_healthy') == is_healthy]
                
                nodes.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
                return nodes[offset:offset + limit]
            else:
                return self._get_all_db(tenant_id, status, is_healthy, limit, offset)
                
        except Exception as e:
            logger.error(f"Failed to get nodes: {e}")
            return []
    
    def _get_all_db(
        self,
        tenant_id: str,
        status: str,
        is_healthy: bool,
        limit: int,
        offset: int
    ) -> List[Dict]:
        """数据库查询列表"""
        try:
            from backend.schemas.gpu_resource_models import GPUNodeModel
            
            query = self._db_session.query(GPUNodeModel)
            
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            if status:
                query = query.filter_by(status=status)
            if is_healthy is not None:
                query = query.filter_by(is_healthy=is_healthy)
            
            query = query.order_by(GPUNodeModel.created_at.desc())
            query = query.offset(offset).limit(limit)
            
            return [m.to_dict() for m in query.all()]
            
        except Exception as e:
            logger.error(f"DB get nodes failed: {e}")
            return []
    
    def update(self, node_id: str, updates: Dict[str, Any], tenant_id: str = None) -> bool:
        """更新节点"""
        try:
            updates['updated_at'] = datetime.utcnow()
            
            if self._use_memory:
                if node_id not in self._memory_store:
                    return False
                node = self._memory_store[node_id]
                if tenant_id and node.get('tenant_id') != tenant_id:
                    return False
                node.update(updates)
                return True
            else:
                from backend.schemas.gpu_resource_models import GPUNodeModel
                query = self._db_session.query(GPUNodeModel).filter_by(id=node_id)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                count = query.update(updates)
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to update node {node_id}: {e}")
            return False
    
    def update_heartbeat(self, node_id: str, metrics: Dict = None) -> bool:
        """更新心跳"""
        updates = {
            'last_heartbeat': datetime.utcnow(),
            'is_healthy': True
        }
        if metrics:
            updates.update({
                'cpu_used': metrics.get('cpu_used', 0),
                'memory_used_mb': metrics.get('memory_used_mb', 0),
                'used_gpu_memory_mb': metrics.get('used_gpu_memory_mb', 0)
            })
        return self.update(node_id, updates)
    
    def delete(self, node_id: str, tenant_id: str = None) -> bool:
        """删除节点"""
        try:
            if self._use_memory:
                if node_id in self._memory_store:
                    node = self._memory_store[node_id]
                    if tenant_id and node.get('tenant_id') != tenant_id:
                        return False
                    del self._memory_store[node_id]
                    return True
                return False
            else:
                from backend.schemas.gpu_resource_models import GPUNodeModel
                query = self._db_session.query(GPUNodeModel).filter_by(id=node_id)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                count = query.delete()
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {e}")
            return False
    
    def get_healthy_nodes(self, tenant_id: str = None) -> List[Dict]:
        """获取健康节点"""
        return self.get_all(tenant_id=tenant_id, is_healthy=True, status='online')
    
    def get_statistics(self, tenant_id: str = None) -> Dict[str, Any]:
        """获取节点统计"""
        nodes = self.get_all(tenant_id=tenant_id, limit=10000)
        
        total_gpus = sum(n.get('gpu_count', 0) for n in nodes)
        total_gpu_memory = sum(n.get('total_gpu_memory_mb', 0) for n in nodes)
        used_gpu_memory = sum(n.get('used_gpu_memory_mb', 0) for n in nodes)
        
        by_status = {}
        for node in nodes:
            status = node.get('status', 'unknown')
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            'total_nodes': len(nodes),
            'healthy_nodes': sum(1 for n in nodes if n.get('is_healthy')),
            'total_gpus': total_gpus,
            'total_gpu_memory_mb': total_gpu_memory,
            'used_gpu_memory_mb': used_gpu_memory,
            'free_gpu_memory_mb': total_gpu_memory - used_gpu_memory,
            'by_status': by_status
        }


class GPUDeviceRepository:
    """GPU 设备仓库"""
    
    def __init__(self, use_memory: bool = True, db_session=None):
        self._use_memory = use_memory
        self._db_session = db_session
        self._memory_store: Dict[str, Dict] = {}
    
    def _generate_id(self) -> str:
        return f"gpu_{uuid.uuid4().hex[:12]}"
    
    def create(self, device_data: Dict[str, Any]) -> Optional[Dict]:
        """创建 GPU 设备"""
        try:
            device_id = device_data.get('id') or self._generate_id()
            now = datetime.utcnow()
            
            device = {
                'id': device_id,
                'node_id': device_data.get('node_id'),
                'tenant_id': device_data.get('tenant_id'),
                'gpu_index': device_data.get('gpu_index'),
                'uuid': device_data.get('uuid'),
                'name': device_data.get('name'),
                'status': device_data.get('status', 'available'),
                'is_available': device_data.get('is_available', True),
                'memory_total_mb': device_data.get('memory_total_mb', 0),
                'memory_used_mb': device_data.get('memory_used_mb', 0),
                'memory_free_mb': device_data.get('memory_free_mb', 0),
                'utilization_percent': device_data.get('utilization_percent', 0.0),
                'memory_utilization_percent': device_data.get('memory_utilization_percent', 0.0),
                'temperature_celsius': device_data.get('temperature_celsius'),
                'power_usage_watts': device_data.get('power_usage_watts'),
                'power_limit_watts': device_data.get('power_limit_watts'),
                'driver_version': device_data.get('driver_version'),
                'cuda_version': device_data.get('cuda_version'),
                'current_allocation_id': device_data.get('current_allocation_id'),
                'allocated_to_task_id': device_data.get('allocated_to_task_id'),
                'metadata': device_data.get('metadata', {}),
                'created_at': now,
                'updated_at': now
            }
            
            if self._use_memory:
                self._memory_store[device_id] = device
                return device
            else:
                return self._create_db(device)
                
        except Exception as e:
            logger.error(f"Failed to create GPU device: {e}")
            return None
    
    def _create_db(self, device: Dict) -> Optional[Dict]:
        """数据库创建"""
        try:
            from backend.schemas.gpu_resource_models import GPUDeviceModel, GPUStatusEnum
            
            model = GPUDeviceModel(
                id=device['id'],
                node_id=device['node_id'],
                tenant_id=device['tenant_id'],
                gpu_index=device['gpu_index'],
                uuid=device['uuid'],
                name=device['name'],
                status=GPUStatusEnum(device['status']) if device['status'] else GPUStatusEnum.AVAILABLE,
                is_available=device['is_available'],
                memory_total_mb=device['memory_total_mb'],
                memory_used_mb=device['memory_used_mb'],
                memory_free_mb=device['memory_free_mb'],
                utilization_percent=device['utilization_percent'],
                memory_utilization_percent=device['memory_utilization_percent'],
                temperature_celsius=device['temperature_celsius'],
                power_usage_watts=device['power_usage_watts'],
                driver_version=device['driver_version'],
                cuda_version=device['cuda_version'],
                current_allocation_id=device['current_allocation_id'],
                allocated_to_task_id=device['allocated_to_task_id'],
                device_metadata=device['metadata'],
                created_at=device['created_at'],
                updated_at=device['updated_at']
            )
            
            self._db_session.add(model)
            self._db_session.commit()
            return model.to_dict()
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB create GPU device failed: {e}")
            return None
    
    def get_by_id(self, device_id: str) -> Optional[Dict]:
        """根据 ID 获取设备"""
        if self._use_memory:
            return self._memory_store.get(device_id)
        else:
            try:
                from backend.schemas.gpu_resource_models import GPUDeviceModel
                model = self._db_session.query(GPUDeviceModel).filter_by(id=device_id).first()
                return model.to_dict() if model else None
            except Exception as e:
                logger.error(f"DB get device failed: {e}")
                return None
    
    def get_by_node(self, node_id: str, status: str = None) -> List[Dict]:
        """获取节点的所有 GPU 设备"""
        try:
            if self._use_memory:
                devices = [d for d in self._memory_store.values() if d.get('node_id') == node_id]
                if status:
                    devices = [d for d in devices if d.get('status') == status]
                devices.sort(key=lambda x: x.get('gpu_index', 0))
                return devices
            else:
                from backend.schemas.gpu_resource_models import GPUDeviceModel
                query = self._db_session.query(GPUDeviceModel).filter_by(node_id=node_id)
                if status:
                    query = query.filter_by(status=status)
                query = query.order_by(GPUDeviceModel.gpu_index)
                return [m.to_dict() for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get devices for node {node_id}: {e}")
            return []
    
    def get_available(self, node_id: str = None, min_memory_mb: int = 0) -> List[Dict]:
        """获取可用 GPU"""
        try:
            if self._use_memory:
                devices = list(self._memory_store.values())
                devices = [d for d in devices if d.get('is_available') and d.get('status') == 'available']
                if node_id:
                    devices = [d for d in devices if d.get('node_id') == node_id]
                if min_memory_mb > 0:
                    devices = [d for d in devices if d.get('memory_free_mb', 0) >= min_memory_mb]
                return devices
            else:
                from backend.schemas.gpu_resource_models import GPUDeviceModel, GPUStatusEnum
                query = self._db_session.query(GPUDeviceModel).filter(
                    GPUDeviceModel.is_available == True,
                    GPUDeviceModel.status == GPUStatusEnum.AVAILABLE
                )
                if node_id:
                    query = query.filter_by(node_id=node_id)
                if min_memory_mb > 0:
                    query = query.filter(GPUDeviceModel.memory_free_mb >= min_memory_mb)
                return [m.to_dict() for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get available GPUs: {e}")
            return []
    
    def update(self, device_id: str, updates: Dict[str, Any]) -> bool:
        """更新设备"""
        try:
            updates['updated_at'] = datetime.utcnow()
            
            if self._use_memory:
                if device_id in self._memory_store:
                    self._memory_store[device_id].update(updates)
                    return True
                return False
            else:
                from backend.schemas.gpu_resource_models import GPUDeviceModel
                count = self._db_session.query(GPUDeviceModel).filter_by(id=device_id).update(updates)
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to update device {device_id}: {e}")
            return False
    
    def update_metrics(self, device_id: str, metrics: Dict[str, Any]) -> bool:
        """更新设备指标"""
        updates = {
            'utilization_percent': metrics.get('utilization_percent'),
            'memory_used_mb': metrics.get('memory_used_mb'),
            'memory_free_mb': metrics.get('memory_free_mb'),
            'memory_utilization_percent': metrics.get('memory_utilization_percent'),
            'temperature_celsius': metrics.get('temperature_celsius'),
            'power_usage_watts': metrics.get('power_usage_watts')
        }
        # 移除 None 值
        updates = {k: v for k, v in updates.items() if v is not None}
        return self.update(device_id, updates)
    
    def allocate(self, device_id: str, allocation_id: str, task_id: str = None) -> bool:
        """分配 GPU"""
        return self.update(device_id, {
            'status': 'allocated',
            'is_available': False,
            'current_allocation_id': allocation_id,
            'allocated_to_task_id': task_id
        })
    
    def release(self, device_id: str) -> bool:
        """释放 GPU"""
        return self.update(device_id, {
            'status': 'available',
            'is_available': True,
            'current_allocation_id': None,
            'allocated_to_task_id': None
        })
    
    def delete(self, device_id: str) -> bool:
        """删除设备"""
        try:
            if self._use_memory:
                if device_id in self._memory_store:
                    del self._memory_store[device_id]
                    return True
                return False
            else:
                from backend.schemas.gpu_resource_models import GPUDeviceModel
                count = self._db_session.query(GPUDeviceModel).filter_by(id=device_id).delete()
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to delete device {device_id}: {e}")
            return False


class GPUAllocationRepository:
    """GPU 分配仓库"""
    
    def __init__(self, use_memory: bool = True, db_session=None):
        self._use_memory = use_memory
        self._db_session = db_session
        self._memory_store: Dict[str, Dict] = {}
    
    def _generate_id(self) -> str:
        return f"alloc_{uuid.uuid4().hex[:12]}"
    
    def create(self, allocation_data: Dict[str, Any]) -> Optional[Dict]:
        """创建分配记录"""
        try:
            alloc_id = allocation_data.get('id') or self._generate_id()
            now = datetime.utcnow()
            
            allocation = {
                'id': alloc_id,
                'tenant_id': allocation_data.get('tenant_id'),
                'node_id': allocation_data.get('node_id'),
                'task_id': allocation_data.get('task_id'),
                'user_id': allocation_data.get('user_id'),
                'gpu_indices': allocation_data.get('gpu_indices', []),
                'gpu_count': allocation_data.get('gpu_count', len(allocation_data.get('gpu_indices', []))),
                'gpu_memory_mb': allocation_data.get('gpu_memory_mb', 0),
                'cpu_cores': allocation_data.get('cpu_cores', 0),
                'memory_mb': allocation_data.get('memory_mb', 0),
                'disk_mb': allocation_data.get('disk_mb', 0),
                'network_mbps': allocation_data.get('network_mbps', 0),
                'status': allocation_data.get('status', 'pending'),
                'priority': allocation_data.get('priority', 1),
                'strategy': allocation_data.get('strategy', 'best_fit'),
                'labels_affinity': allocation_data.get('labels_affinity'),
                'requested_at': allocation_data.get('requested_at', now),
                'allocated_at': allocation_data.get('allocated_at'),
                'released_at': allocation_data.get('released_at'),
                'expires_at': allocation_data.get('expires_at'),
                'lease_id': allocation_data.get('lease_id'),
                'lease_duration_seconds': allocation_data.get('lease_duration_seconds'),
                'metadata': allocation_data.get('metadata', {}),
                'created_at': now,
                'updated_at': now
            }
            
            if self._use_memory:
                self._memory_store[alloc_id] = allocation
                return allocation
            else:
                return self._create_db(allocation)
                
        except Exception as e:
            logger.error(f"Failed to create allocation: {e}")
            return None
    
    def _create_db(self, allocation: Dict) -> Optional[Dict]:
        """数据库创建"""
        try:
            from backend.schemas.gpu_resource_models import (
                GPUAllocationModel, AllocationStatusEnum, AllocationStrategyEnum
            )
            
            model = GPUAllocationModel(
                id=allocation['id'],
                tenant_id=allocation['tenant_id'],
                node_id=allocation['node_id'],
                task_id=allocation['task_id'],
                user_id=allocation['user_id'],
                gpu_indices=allocation['gpu_indices'],
                gpu_count=allocation['gpu_count'],
                gpu_memory_mb=allocation['gpu_memory_mb'],
                cpu_cores=allocation['cpu_cores'],
                memory_mb=allocation['memory_mb'],
                disk_mb=allocation['disk_mb'],
                network_mbps=allocation['network_mbps'],
                status=AllocationStatusEnum(allocation['status']) if allocation['status'] else AllocationStatusEnum.PENDING,
                priority=allocation['priority'],
                strategy=AllocationStrategyEnum(allocation['strategy']) if allocation['strategy'] else AllocationStrategyEnum.BEST_FIT,
                labels_affinity=allocation['labels_affinity'],
                requested_at=allocation['requested_at'],
                allocated_at=allocation['allocated_at'],
                released_at=allocation['released_at'],
                expires_at=allocation['expires_at'],
                lease_id=allocation['lease_id'],
                lease_duration_seconds=allocation['lease_duration_seconds'],
                allocation_metadata=allocation['metadata'],
                created_at=allocation['created_at'],
                updated_at=allocation['updated_at']
            )
            
            self._db_session.add(model)
            self._db_session.commit()
            return model.to_dict()
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB create allocation failed: {e}")
            return None
    
    def get_by_id(self, allocation_id: str, tenant_id: str = None) -> Optional[Dict]:
        """根据 ID 获取分配"""
        try:
            if self._use_memory:
                allocation = self._memory_store.get(allocation_id)
                if allocation and (tenant_id is None or allocation.get('tenant_id') == tenant_id):
                    return allocation
                return None
            else:
                from backend.schemas.gpu_resource_models import GPUAllocationModel
                query = self._db_session.query(GPUAllocationModel).filter_by(id=allocation_id)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                model = query.first()
                return model.to_dict() if model else None
                
        except Exception as e:
            logger.error(f"Failed to get allocation {allocation_id}: {e}")
            return None
    
    def get_all(
        self,
        tenant_id: str = None,
        node_id: str = None,
        task_id: str = None,
        status: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """获取分配列表"""
        try:
            if self._use_memory:
                allocations = list(self._memory_store.values())
                
                if tenant_id:
                    allocations = [a for a in allocations if a.get('tenant_id') == tenant_id]
                if node_id:
                    allocations = [a for a in allocations if a.get('node_id') == node_id]
                if task_id:
                    allocations = [a for a in allocations if a.get('task_id') == task_id]
                if status:
                    allocations = [a for a in allocations if a.get('status') == status]
                
                allocations.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
                return allocations[offset:offset + limit]
            else:
                from backend.schemas.gpu_resource_models import GPUAllocationModel
                query = self._db_session.query(GPUAllocationModel)
                
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                if node_id:
                    query = query.filter_by(node_id=node_id)
                if task_id:
                    query = query.filter_by(task_id=task_id)
                if status:
                    query = query.filter_by(status=status)
                
                query = query.order_by(GPUAllocationModel.created_at.desc())
                query = query.offset(offset).limit(limit)
                
                return [m.to_dict() for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get allocations: {e}")
            return []
    
    def get_active(self, tenant_id: str = None, node_id: str = None) -> List[Dict]:
        """获取活跃分配"""
        return self.get_all(tenant_id=tenant_id, node_id=node_id, status='active')
    
    def update(self, allocation_id: str, updates: Dict[str, Any], tenant_id: str = None) -> bool:
        """更新分配"""
        try:
            updates['updated_at'] = datetime.utcnow()
            
            if self._use_memory:
                if allocation_id not in self._memory_store:
                    return False
                allocation = self._memory_store[allocation_id]
                if tenant_id and allocation.get('tenant_id') != tenant_id:
                    return False
                allocation.update(updates)
                return True
            else:
                from backend.schemas.gpu_resource_models import GPUAllocationModel
                query = self._db_session.query(GPUAllocationModel).filter_by(id=allocation_id)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                count = query.update(updates)
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to update allocation {allocation_id}: {e}")
            return False
    
    def activate(self, allocation_id: str, tenant_id: str = None) -> bool:
        """激活分配"""
        return self.update(allocation_id, {
            'status': 'active',
            'allocated_at': datetime.utcnow()
        }, tenant_id)
    
    def release(self, allocation_id: str, tenant_id: str = None) -> bool:
        """释放分配"""
        return self.update(allocation_id, {
            'status': 'released',
            'released_at': datetime.utcnow()
        }, tenant_id)
    
    def delete(self, allocation_id: str, tenant_id: str = None) -> bool:
        """删除分配"""
        try:
            if self._use_memory:
                if allocation_id in self._memory_store:
                    allocation = self._memory_store[allocation_id]
                    if tenant_id and allocation.get('tenant_id') != tenant_id:
                        return False
                    del self._memory_store[allocation_id]
                    return True
                return False
            else:
                from backend.schemas.gpu_resource_models import GPUAllocationModel
                query = self._db_session.query(GPUAllocationModel).filter_by(id=allocation_id)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                count = query.delete()
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to delete allocation {allocation_id}: {e}")
            return False
    
    def get_statistics(self, tenant_id: str = None) -> Dict[str, Any]:
        """获取分配统计"""
        allocations = self.get_all(tenant_id=tenant_id, limit=10000)
        
        by_status = {}
        total_gpus_allocated = 0
        total_memory_allocated = 0
        
        for alloc in allocations:
            status = alloc.get('status', 'unknown')
            by_status[status] = by_status.get(status, 0) + 1
            
            if status == 'active':
                total_gpus_allocated += alloc.get('gpu_count', 0)
                total_memory_allocated += alloc.get('gpu_memory_mb', 0)
        
        return {
            'total_allocations': len(allocations),
            'active_allocations': by_status.get('active', 0),
            'total_gpus_allocated': total_gpus_allocated,
            'total_memory_allocated_mb': total_memory_allocated,
            'by_status': by_status
        }


class GPUUsageHistoryRepository:
    """GPU 使用历史仓库"""
    
    def __init__(self, use_memory: bool = True, db_session=None):
        self._use_memory = use_memory
        self._db_session = db_session
        self._memory_store: List[Dict] = []
    
    def _generate_id(self) -> str:
        return f"usage_{uuid.uuid4().hex[:12]}"
    
    def record(self, usage_data: Dict[str, Any]) -> Optional[Dict]:
        """记录使用历史"""
        try:
            usage_id = self._generate_id()
            now = datetime.utcnow()
            
            usage = {
                'id': usage_id,
                'tenant_id': usage_data.get('tenant_id'),
                'node_id': usage_data.get('node_id'),
                'gpu_id': usage_data.get('gpu_id'),
                'gpu_index': usage_data.get('gpu_index'),
                'timestamp': usage_data.get('timestamp', now),
                'period_type': usage_data.get('period_type', 'minute'),
                'avg_utilization_percent': usage_data.get('avg_utilization_percent', 0.0),
                'max_utilization_percent': usage_data.get('max_utilization_percent', 0.0),
                'min_utilization_percent': usage_data.get('min_utilization_percent', 0.0),
                'avg_memory_used_mb': usage_data.get('avg_memory_used_mb', 0),
                'max_memory_used_mb': usage_data.get('max_memory_used_mb', 0),
                'min_memory_used_mb': usage_data.get('min_memory_used_mb', 0),
                'avg_temperature_celsius': usage_data.get('avg_temperature_celsius'),
                'max_temperature_celsius': usage_data.get('max_temperature_celsius'),
                'avg_power_watts': usage_data.get('avg_power_watts'),
                'max_power_watts': usage_data.get('max_power_watts'),
                'allocation_count': usage_data.get('allocation_count', 0),
                'active_time_seconds': usage_data.get('active_time_seconds', 0),
                'idle_time_seconds': usage_data.get('idle_time_seconds', 0),
                'created_at': now
            }
            
            if self._use_memory:
                self._memory_store.append(usage)
                # 保持最近的记录
                if len(self._memory_store) > 10000:
                    self._memory_store = self._memory_store[-10000:]
                return usage
            else:
                return self._record_db(usage)
                
        except Exception as e:
            logger.error(f"Failed to record usage: {e}")
            return None
    
    def _record_db(self, usage: Dict) -> Optional[Dict]:
        """数据库记录"""
        try:
            from backend.schemas.gpu_resource_models import GPUUsageHistoryModel
            
            model = GPUUsageHistoryModel(**usage)
            self._db_session.add(model)
            self._db_session.commit()
            return model.to_dict()
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB record usage failed: {e}")
            return None
    
    def get_history(
        self,
        node_id: str = None,
        gpu_id: str = None,
        period_type: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[Dict]:
        """获取使用历史"""
        try:
            if self._use_memory:
                history = self._memory_store.copy()
                
                if node_id:
                    history = [h for h in history if h.get('node_id') == node_id]
                if gpu_id:
                    history = [h for h in history if h.get('gpu_id') == gpu_id]
                if period_type:
                    history = [h for h in history if h.get('period_type') == period_type]
                if start_time:
                    history = [h for h in history if h.get('timestamp') and h['timestamp'] >= start_time]
                if end_time:
                    history = [h for h in history if h.get('timestamp') and h['timestamp'] <= end_time]
                
                history.sort(key=lambda x: x.get('timestamp', datetime.min), reverse=True)
                return history[:limit]
            else:
                from backend.schemas.gpu_resource_models import GPUUsageHistoryModel
                query = self._db_session.query(GPUUsageHistoryModel)
                
                if node_id:
                    query = query.filter_by(node_id=node_id)
                if gpu_id:
                    query = query.filter_by(gpu_id=gpu_id)
                if period_type:
                    query = query.filter_by(period_type=period_type)
                if start_time:
                    query = query.filter(GPUUsageHistoryModel.timestamp >= start_time)
                if end_time:
                    query = query.filter(GPUUsageHistoryModel.timestamp <= end_time)
                
                query = query.order_by(GPUUsageHistoryModel.timestamp.desc())
                query = query.limit(limit)
                
                return [m.to_dict() for m in query.all()]
                
        except Exception as e:
            logger.error(f"Failed to get usage history: {e}")
            return []


# ==================== 单例获取函数 ====================

_node_repo = None
_device_repo = None
_allocation_repo = None
_usage_repo = None


def get_gpu_node_repository(use_memory: bool = True) -> GPUNodeRepository:
    """获取节点仓库"""
    global _node_repo
    if _node_repo is None:
        _node_repo = GPUNodeRepository(use_memory=use_memory)
    return _node_repo


def get_gpu_device_repository(use_memory: bool = True) -> GPUDeviceRepository:
    """获取设备仓库"""
    global _device_repo
    if _device_repo is None:
        _device_repo = GPUDeviceRepository(use_memory=use_memory)
    return _device_repo


def get_gpu_allocation_repository(use_memory: bool = True) -> GPUAllocationRepository:
    """获取分配仓库"""
    global _allocation_repo
    if _allocation_repo is None:
        _allocation_repo = GPUAllocationRepository(use_memory=use_memory)
    return _allocation_repo


def get_gpu_usage_repository(use_memory: bool = True) -> GPUUsageHistoryRepository:
    """获取使用历史仓库"""
    global _usage_repo
    if _usage_repo is None:
        _usage_repo = GPUUsageHistoryRepository(use_memory=use_memory)
    return _usage_repo


def reset_gpu_repositories():
    """重置所有仓库（用于测试）"""
    global _node_repo, _device_repo, _allocation_repo, _usage_repo
    _node_repo = None
    _device_repo = None
    _allocation_repo = None
    _usage_repo = None

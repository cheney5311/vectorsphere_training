#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU 资源管理服务层

提供 GPU 资源的业务逻辑处理，包括资源分配、监控、统计等功能。
使用仓库层进行数据持久化，集成分布式模块进行实际资源操作。
"""

import logging
import threading
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AllocationResult:
    """分配结果"""
    success: bool
    allocation_id: str = None
    node_id: str = None
    gpu_indices: List[int] = None
    error_message: str = None


class GPUResourceService:
    """GPU 资源管理服务
    
    提供 GPU 资源的分配、释放、监控等功能。
    委托仓库层进行数据持久化。
    """
    
    def __init__(self, config: Dict[str, Any] = None, use_memory_storage: bool = True):
        self.config = config or {}
        self._use_memory_storage = use_memory_storage
        
        # 初始化仓库
        self._init_repositories()
        
        # 监控状态
        self._monitoring = False
        self._monitor_thread = None
        self._monitor_interval = self.config.get('monitor_interval', 10)
        
        # 锁
        self._lock = threading.RLock()
    
    def _init_repositories(self):
        """初始化仓库"""
        try:
            from backend.repositories.gpu_resource_repository import (
                get_gpu_node_repository,
                get_gpu_device_repository,
                get_gpu_allocation_repository,
                get_gpu_usage_repository
            )
            self._node_repo = get_gpu_node_repository(use_memory=self._use_memory_storage)
            self._device_repo = get_gpu_device_repository(use_memory=self._use_memory_storage)
            self._allocation_repo = get_gpu_allocation_repository(use_memory=self._use_memory_storage)
            self._usage_repo = get_gpu_usage_repository(use_memory=self._use_memory_storage)
        except ImportError as e:
            logger.warning(f"Failed to import repositories: {e}")
            self._node_repo = None
            self._device_repo = None
            self._allocation_repo = None
            self._usage_repo = None
    
    # ==================== 监控控制 ====================
    
    def start_monitoring(self, interval: int = None, background: bool = True) -> bool:
        """启动监控"""
        with self._lock:
            if self._monitoring:
                return True
            
            if interval:
                self._monitor_interval = interval
            
            self._monitoring = True
            
            if background:
                self._monitor_thread = threading.Thread(
                    target=self._monitoring_loop,
                    name="GPUMonitor",
                    daemon=True
                )
                self._monitor_thread.start()
            
            logger.info(f"GPU monitoring started with interval {self._monitor_interval}s")
            return True
    
    def stop_monitoring(self) -> bool:
        """停止监控"""
        with self._lock:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5)
            logger.info("GPU monitoring stopped")
            return True
    
    def _monitoring_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                self._collect_metrics()
                time.sleep(self._monitor_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(5)
    
    def _collect_metrics(self):
        """采集指标"""
        try:
            # 使用底层 GPU 资源管理器采集
            from backend.services import gpu_resource_manager as gpu_svc
            summary = gpu_svc.get_gpu_summary()
            
            # 更新缓存
            self._last_metrics = {
                'summary': summary,
                'updated_at': datetime.utcnow()
            }
            
            # 记录使用历史
            if self._usage_repo and summary.get('gpus'):
                for gpu_info in summary.get('gpus', []):
                    self._usage_repo.record({
                        'gpu_index': gpu_info.get('index'),
                        'period_type': 'minute',
                        'avg_utilization_percent': gpu_info.get('utilization_percent', 0),
                        'max_utilization_percent': gpu_info.get('utilization_percent', 0),
                        'avg_memory_used_mb': gpu_info.get('memory_used_mb', 0),
                        'max_memory_used_mb': gpu_info.get('memory_used_mb', 0)
                    })
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
    
    # ==================== 指标获取 ====================
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        try:
            # 首先尝试从模块获取缓存
            try:
                from backend.modules.distributed.gpu_resource_manager import get_cached_metrics
                cached = get_cached_metrics()
                if cached:
                    return cached
            except ImportError:
                pass
            
            # 直接采集
            from backend.services import gpu_resource_manager as gpu_svc
            summary = gpu_svc.get_gpu_summary()
            
            return {
                'last': summary,
                'updated_at': time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {'error': str(e), 'updated_at': time.time()}
    
    def get_gpu_list(self) -> List[Dict[str, Any]]:
        """获取 GPU 列表"""
        try:
            from backend.services import gpu_resource_manager as gpu_svc
            return gpu_svc.detect_gpus()
        except Exception as e:
            logger.error(f"Failed to get GPU list: {e}")
            return []
    
    def get_gpu_details(self) -> List[Dict[str, Any]]:
        """获取 GPU 详细信息"""
        try:
            from backend.services import gpu_resource_manager as gpu_svc
            return gpu_svc.get_gpu_metrics()
        except Exception as e:
            logger.error(f"Failed to get GPU details: {e}")
            return []
    
    # ==================== 节点管理 ====================
    
    def register_node(
        self,
        hostname: str,
        ip_address: str = None,
        port: int = 8080,
        tenant_id: str = None,
        labels: Dict[str, str] = None,
        capabilities: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """注册节点"""
        try:
            if not self._node_repo:
                return False, "Repository not available", None
            
            # 检测本地 GPU
            gpus = self.get_gpu_list()
            gpu_details = self.get_gpu_details()
            
            total_memory = sum(g.get('memory_total_mb', 0) for g in gpus)
            
            node_data = {
                'hostname': hostname,
                'ip_address': ip_address,
                'port': port,
                'tenant_id': tenant_id,
                'gpu_count': len(gpus),
                'total_gpu_memory_mb': total_memory,
                'labels': labels or {},
                'capabilities': capabilities or [],
                'metadata': metadata or {}
            }
            
            node = self._node_repo.create(node_data)
            if not node:
                return False, "Failed to create node", None
            
            # 注册 GPU 设备
            if self._device_repo:
                for gpu in gpus:
                    detail = next((d for d in gpu_details if d.get('index') == gpu.get('index')), {})
                    self._device_repo.create({
                        'node_id': node['id'],
                        'tenant_id': tenant_id,
                        'gpu_index': gpu.get('index'),
                        'uuid': gpu.get('uuid'),
                        'name': gpu.get('name'),
                        'memory_total_mb': gpu.get('memory_total_mb', 0),
                        'memory_used_mb': detail.get('memory_used_mb', 0),
                        'memory_free_mb': gpu.get('memory_total_mb', 0) - detail.get('memory_used_mb', 0),
                        'utilization_percent': detail.get('utilization_percent', 0)
                    })
            
            logger.info(f"Node {node['id']} registered with {len(gpus)} GPUs")
            return True, "Node registered successfully", node
            
        except Exception as e:
            logger.error(f"Failed to register node: {e}")
            return False, f"Registration failed: {str(e)}", None
    
    def get_node(self, node_id: str, tenant_id: str = None) -> Optional[Dict]:
        """获取节点"""
        if not self._node_repo:
            return None
        return self._node_repo.get_by_id(node_id, tenant_id)
    
    def list_nodes(
        self,
        tenant_id: str = None,
        status: str = None,
        is_healthy: bool = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """列出节点"""
        if not self._node_repo:
            return []
        return self._node_repo.get_all(
            tenant_id=tenant_id,
            status=status,
            is_healthy=is_healthy,
            limit=limit,
            offset=offset
        )
    
    def update_node_heartbeat(self, node_id: str, metrics: Dict = None) -> bool:
        """更新节点心跳"""
        if not self._node_repo:
            return False
        return self._node_repo.update_heartbeat(node_id, metrics)
    
    def unregister_node(self, node_id: str, tenant_id: str = None) -> Tuple[bool, str]:
        """注销节点"""
        try:
            if not self._node_repo:
                return False, "Repository not available"
            
            # 检查是否有活跃分配
            if self._allocation_repo:
                active = self._allocation_repo.get_active(node_id=node_id)
                if active:
                    return False, f"Node has {len(active)} active allocations"
            
            # 删除 GPU 设备
            if self._device_repo:
                devices = self._device_repo.get_by_node(node_id)
                for device in devices:
                    self._device_repo.delete(device['id'])
            
            # 删除节点
            success = self._node_repo.delete(node_id, tenant_id)
            if success:
                return True, "Node unregistered successfully"
            return False, "Node not found"
            
        except Exception as e:
            logger.error(f"Failed to unregister node: {e}")
            return False, f"Unregistration failed: {str(e)}"
    
    # ==================== 资源分配 ====================
    
    def allocate_resources(
        self,
        gpu_count: int = 0,
        gpu_memory_mb: int = 0,
        cpu_cores: int = 0,
        memory_mb: int = 0,
        priority: int = 1,
        labels_affinity: Dict[str, str] = None,
        prefer_same_node: bool = True,
        task_id: str = None,
        user_id: str = None,
        tenant_id: str = None,
        lease_duration_seconds: int = None,
        strategy: str = "best_fit"
    ) -> AllocationResult:
        """分配资源
        
        Args:
            gpu_count: 需要的 GPU 数量
            gpu_memory_mb: 每个 GPU 需要的内存
            cpu_cores: 需要的 CPU 核心数
            memory_mb: 需要的内存
            priority: 优先级 (1-10)
            labels_affinity: 节点标签亲和
            prefer_same_node: 是否优先同节点
            task_id: 任务 ID
            user_id: 用户 ID
            tenant_id: 租户 ID
            lease_duration_seconds: 租约时长
            strategy: 分配策略
            
        Returns:
            AllocationResult
        """
        try:
            # 首先尝试使用底层分配器
            result = self._allocate_via_module(
                gpu_count=gpu_count,
                gpu_memory_mb=gpu_memory_mb,
                cpu_cores=cpu_cores,
                memory_mb=memory_mb,
                priority=priority,
                labels_affinity=labels_affinity
            )
            
            if result.success:
                # 创建分配记录
                if self._allocation_repo:
                    allocation = self._allocation_repo.create({
                        'node_id': result.node_id,
                        'task_id': task_id,
                        'user_id': user_id,
                        'tenant_id': tenant_id,
                        'gpu_indices': result.gpu_indices,
                        'gpu_count': gpu_count,
                        'gpu_memory_mb': gpu_memory_mb,
                        'cpu_cores': cpu_cores,
                        'memory_mb': memory_mb,
                        'priority': priority,
                        'strategy': strategy,
                        'labels_affinity': labels_affinity,
                        'status': 'active',
                        'allocated_at': datetime.utcnow(),
                        'lease_duration_seconds': lease_duration_seconds,
                        'expires_at': datetime.utcnow() + timedelta(seconds=lease_duration_seconds) if lease_duration_seconds else None
                    })
                    
                    if allocation:
                        result.allocation_id = allocation['id']
                        
                        # 更新 GPU 设备状态
                        if self._device_repo:
                            for gpu_idx in result.gpu_indices:
                                devices = self._device_repo.get_by_node(result.node_id)
                                for device in devices:
                                    if device.get('gpu_index') == gpu_idx:
                                        self._device_repo.allocate(
                                            device['id'], 
                                            allocation['id'],
                                            task_id
                                        )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to allocate resources: {e}")
            return AllocationResult(success=False, error_message=str(e))
    
    def _allocate_via_module(
        self,
        gpu_count: int,
        gpu_memory_mb: int,
        cpu_cores: int,
        memory_mb: int,
        priority: int,
        labels_affinity: Dict
    ) -> AllocationResult:
        """通过分布式模块分配"""
        try:
            from backend.modules.distributed.task_scheduler import ResourceRequirement
            from backend.modules.distributed.resource_allocator import get_resource_allocator
            from backend.modules.distributed.cluster_manager import get_cluster_manager
            
            # 构建资源需求
            requirement = ResourceRequirement(
                cpu_cores=cpu_cores,
                memory_mb=memory_mb,
                gpu_count=gpu_count,
                gpu_memory_mb=gpu_memory_mb,
                priority=priority,
                labels_affinity=labels_affinity
            )
            
            # 获取集群管理器和分配器
            cluster = get_cluster_manager()
            allocator = get_resource_allocator()
            
            # 异步执行分配
            loop = asyncio.new_event_loop()
            try:
                # 获取健康节点
                nodes = loop.run_until_complete(cluster.get_healthy_nodes())
                
                # 分配资源
                result = loop.run_until_complete(
                    allocator.allocate_resources(nodes, requirement)
                )
                
                if result:
                    allocation_id, allocation = result
                    return AllocationResult(
                        success=True,
                        allocation_id=allocation_id,
                        node_id=allocation.node_id,
                        gpu_indices=allocation.gpus
                    )
                else:
                    return AllocationResult(
                        success=False,
                        error_message="No available resources"
                    )
                    
            finally:
                loop.close()
                
        except ImportError:
            logger.warning("Distributed modules not available, using local allocation")
            return self._allocate_local(gpu_count, gpu_memory_mb)
        except Exception as e:
            logger.error(f"Module allocation failed: {e}")
            return AllocationResult(success=False, error_message=str(e))
    
    def _allocate_local(self, gpu_count: int, gpu_memory_mb: int) -> AllocationResult:
        """本地分配（无分布式模块时）"""
        try:
            # 获取可用 GPU
            if not self._device_repo:
                return AllocationResult(success=False, error_message="Device repository not available")
            
            available = self._device_repo.get_available(min_memory_mb=gpu_memory_mb)
            
            if len(available) < gpu_count:
                return AllocationResult(
                    success=False,
                    error_message=f"Not enough GPUs: need {gpu_count}, available {len(available)}"
                )
            
            # 选择前 N 个 GPU
            selected = available[:gpu_count]
            gpu_indices = [g['gpu_index'] for g in selected]
            node_id = selected[0]['node_id'] if selected else None
            
            return AllocationResult(
                success=True,
                node_id=node_id,
                gpu_indices=gpu_indices
            )
            
        except Exception as e:
            logger.error(f"Local allocation failed: {e}")
            return AllocationResult(success=False, error_message=str(e))
    
    def release_allocation(
        self,
        allocation_id: str,
        tenant_id: str = None
    ) -> Tuple[bool, str]:
        """释放分配"""
        try:
            if not self._allocation_repo:
                return False, "Repository not available"
            
            allocation = self._allocation_repo.get_by_id(allocation_id, tenant_id)
            if not allocation:
                return False, "Allocation not found"
            
            if allocation.get('status') == 'released':
                return False, "Allocation already released"
            
            # 释放 GPU 设备
            if self._device_repo:
                node_id = allocation.get('node_id')
                gpu_indices = allocation.get('gpu_indices', [])
                devices = self._device_repo.get_by_node(node_id)
                for device in devices:
                    if device.get('gpu_index') in gpu_indices:
                        self._device_repo.release(device['id'])
            
            # 尝试通过底层释放
            try:
                from backend.modules.distributed.resource_allocator import get_resource_allocator
                allocator = get_resource_allocator()
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(allocator.release_resources(allocation_id))
                finally:
                    loop.close()
            except Exception:
                pass
            
            # 更新分配状态
            success = self._allocation_repo.release(allocation_id, tenant_id)
            
            if success:
                logger.info(f"Allocation {allocation_id} released")
                return True, "Allocation released successfully"
            
            return False, "Failed to release allocation"
            
        except Exception as e:
            logger.error(f"Failed to release allocation: {e}")
            return False, f"Release failed: {str(e)}"
    
    def get_allocation(self, allocation_id: str, tenant_id: str = None) -> Optional[Dict]:
        """获取分配详情"""
        if not self._allocation_repo:
            return None
        return self._allocation_repo.get_by_id(allocation_id, tenant_id)
    
    def list_allocations(
        self,
        tenant_id: str = None,
        node_id: str = None,
        task_id: str = None,
        status: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """列出分配"""
        if not self._allocation_repo:
            return []
        return self._allocation_repo.get_all(
            tenant_id=tenant_id,
            node_id=node_id,
            task_id=task_id,
            status=status,
            limit=limit,
            offset=offset
        )
    
    # ==================== 统计信息 ====================
    
    def get_statistics(self, tenant_id: str = None) -> Dict[str, Any]:
        """获取综合统计"""
        stats = {
            'nodes': {},
            'allocations': {},
            'gpus': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if self._node_repo:
            stats['nodes'] = self._node_repo.get_statistics(tenant_id)
        
        if self._allocation_repo:
            stats['allocations'] = self._allocation_repo.get_statistics(tenant_id)
        
        # GPU 统计
        if self._device_repo:
            devices = self._device_repo.get_by_node(None)  # 获取所有
            total_gpus = len(devices)
            available_gpus = sum(1 for d in devices if d.get('is_available'))
            total_memory = sum(d.get('memory_total_mb', 0) for d in devices)
            used_memory = sum(d.get('memory_used_mb', 0) for d in devices)
            
            stats['gpus'] = {
                'total': total_gpus,
                'available': available_gpus,
                'allocated': total_gpus - available_gpus,
                'total_memory_mb': total_memory,
                'used_memory_mb': used_memory,
                'free_memory_mb': total_memory - used_memory
            }
        
        return stats
    
    def get_usage_history(
        self,
        node_id: str = None,
        gpu_id: str = None,
        period_type: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[Dict]:
        """获取使用历史"""
        if not self._usage_repo:
            return []
        return self._usage_repo.get_history(
            node_id=node_id,
            gpu_id=gpu_id,
            period_type=period_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    # ==================== 健康检查 ====================
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        status = {
            'healthy': True,
            'monitoring': self._monitoring,
            'gpu_available': False,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            gpus = self.get_gpu_list()
            status['gpu_available'] = len(gpus) > 0
            status['gpu_count'] = len(gpus)
        except Exception as e:
            status['healthy'] = False
            status['error'] = str(e)
        
        return status


# ==================== 单例获取函数 ====================

_gpu_service = None
_service_lock = threading.Lock()


def get_gpu_resource_service(config: Dict[str, Any] = None, use_memory: bool = True) -> GPUResourceService:
    """获取 GPU 资源服务实例"""
    global _gpu_service
    with _service_lock:
        if _gpu_service is None:
            _gpu_service = GPUResourceService(config, use_memory_storage=use_memory)
        return _gpu_service


def reset_gpu_resource_service():
    """重置服务实例（用于测试）"""
    global _gpu_service
    with _service_lock:
        if _gpu_service:
            _gpu_service.stop_monitoring()
        _gpu_service = None

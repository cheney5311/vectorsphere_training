#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU 资源管理数据模型

定义 GPU 节点、分配记录、使用历史等 SQLAlchemy ORM 模型。
"""

import json
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

try:
    from sqlalchemy import (
        Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
        Enum as SQLEnum, ForeignKey, Index, UniqueConstraint
    )
    from sqlalchemy.orm import relationship, declarative_base
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# 基类定义
try:
    from backend.schemas.base_models import Base
except ImportError:
    if HAS_SQLALCHEMY:
        Base = declarative_base()
    else:
        Base = object


# ==================== 枚举类型 ====================

class NodeStatusEnum(str, Enum):
    """节点状态"""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"


class GPUStatusEnum(str, Enum):
    """GPU 状态"""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    RESERVED = "reserved"
    FAULTY = "faulty"
    MAINTENANCE = "maintenance"


class AllocationStatusEnum(str, Enum):
    """分配状态"""
    PENDING = "pending"
    ACTIVE = "active"
    RELEASING = "releasing"
    RELEASED = "released"
    EXPIRED = "expired"
    FAILED = "failed"


class AllocationStrategyEnum(str, Enum):
    """分配策略"""
    BEST_FIT = "best_fit"
    WORST_FIT = "worst_fit"
    FIRST_FIT = "first_fit"
    ROUND_ROBIN = "round_robin"
    PRIORITY_BASED = "priority_based"
    AFFINITY_BASED = "affinity_based"


# ==================== SQLAlchemy 模型 ====================

if HAS_SQLALCHEMY:

    class GPUNodeModel(Base):
        """GPU 节点模型"""
        __tablename__ = 'gpu_nodes'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 节点基本信息
        hostname = Column(String(256), nullable=False)
        ip_address = Column(String(64), nullable=True)
        port = Column(Integer, default=8080)
        
        # 状态
        status = Column(SQLEnum(NodeStatusEnum), default=NodeStatusEnum.ONLINE)
        is_healthy = Column(Boolean, default=True)
        last_heartbeat = Column(DateTime, nullable=True)
        
        # GPU 信息
        gpu_count = Column(Integer, default=0)
        total_gpu_memory_mb = Column(Integer, default=0)
        used_gpu_memory_mb = Column(Integer, default=0)
        
        # CPU 信息
        cpu_cores = Column(Integer, default=0)
        cpu_used = Column(Float, default=0.0)
        
        # 内存信息
        memory_total_mb = Column(Integer, default=0)
        memory_used_mb = Column(Integer, default=0)
        
        # 磁盘信息
        disk_total_mb = Column(Integer, default=0)
        disk_used_mb = Column(Integer, default=0)
        
        # 网络信息
        network_bandwidth_mbps = Column(Integer, default=1000)
        
        # 标签和元数据
        labels = Column(JSON, default=dict)
        capabilities = Column(JSON, default=list)  # 支持的功能列表
        node_metadata = Column('metadata', JSON, default=dict)
        
        # 时间戳
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # 索引
        __table_args__ = (
            Index('idx_node_status', 'status'),
            Index('idx_node_tenant', 'tenant_id', 'status'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'hostname': self.hostname,
                'ip_address': self.ip_address,
                'port': self.port,
                'status': self.status.value if self.status else None,
                'is_healthy': self.is_healthy,
                'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
                'gpu_count': self.gpu_count,
                'total_gpu_memory_mb': self.total_gpu_memory_mb,
                'used_gpu_memory_mb': self.used_gpu_memory_mb,
                'cpu_cores': self.cpu_cores,
                'cpu_used': self.cpu_used,
                'memory_total_mb': self.memory_total_mb,
                'memory_used_mb': self.memory_used_mb,
                'disk_total_mb': self.disk_total_mb,
                'disk_used_mb': self.disk_used_mb,
                'network_bandwidth_mbps': self.network_bandwidth_mbps,
                'labels': self.labels,
                'capabilities': self.capabilities,
                'metadata': self.node_metadata,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None
            }


    class GPUDeviceModel(Base):
        """GPU 设备模型"""
        __tablename__ = 'gpu_devices'
        
        id = Column(String(64), primary_key=True)
        node_id = Column(String(64), ForeignKey('gpu_nodes.id'), nullable=False, index=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # GPU 基本信息
        gpu_index = Column(Integer, nullable=False)
        uuid = Column(String(128), nullable=True, unique=True)
        name = Column(String(256), nullable=True)
        
        # 状态
        status = Column(SQLEnum(GPUStatusEnum), default=GPUStatusEnum.AVAILABLE)
        is_available = Column(Boolean, default=True)
        
        # 内存信息
        memory_total_mb = Column(Integer, default=0)
        memory_used_mb = Column(Integer, default=0)
        memory_free_mb = Column(Integer, default=0)
        
        # 利用率
        utilization_percent = Column(Float, default=0.0)
        memory_utilization_percent = Column(Float, default=0.0)
        
        # 温度和功耗
        temperature_celsius = Column(Float, nullable=True)
        power_usage_watts = Column(Float, nullable=True)
        power_limit_watts = Column(Float, nullable=True)
        
        # 驱动信息
        driver_version = Column(String(64), nullable=True)
        cuda_version = Column(String(64), nullable=True)
        
        # 当前分配信息
        current_allocation_id = Column(String(64), nullable=True)
        allocated_to_task_id = Column(String(64), nullable=True)
        
        # 元数据
        device_metadata = Column('metadata', JSON, default=dict)
        
        # 时间戳
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # 索引
        __table_args__ = (
            Index('idx_gpu_node_index', 'node_id', 'gpu_index'),
            Index('idx_gpu_status', 'status', 'is_available'),
            UniqueConstraint('node_id', 'gpu_index', name='uq_gpu_node_index'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'node_id': self.node_id,
                'tenant_id': self.tenant_id,
                'gpu_index': self.gpu_index,
                'uuid': self.uuid,
                'name': self.name,
                'status': self.status.value if self.status else None,
                'is_available': self.is_available,
                'memory_total_mb': self.memory_total_mb,
                'memory_used_mb': self.memory_used_mb,
                'memory_free_mb': self.memory_free_mb,
                'utilization_percent': self.utilization_percent,
                'memory_utilization_percent': self.memory_utilization_percent,
                'temperature_celsius': self.temperature_celsius,
                'power_usage_watts': self.power_usage_watts,
                'driver_version': self.driver_version,
                'cuda_version': self.cuda_version,
                'current_allocation_id': self.current_allocation_id,
                'allocated_to_task_id': self.allocated_to_task_id,
                'metadata': self.device_metadata,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None
            }


    class GPUAllocationModel(Base):
        """GPU 分配记录模型"""
        __tablename__ = 'gpu_allocations'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 分配信息
        node_id = Column(String(64), ForeignKey('gpu_nodes.id'), nullable=False)
        task_id = Column(String(64), nullable=True, index=True)
        user_id = Column(String(64), nullable=True)
        
        # 资源分配
        gpu_indices = Column(JSON, nullable=False)  # 分配的 GPU 索引列表
        gpu_count = Column(Integer, default=0)
        gpu_memory_mb = Column(Integer, default=0)
        cpu_cores = Column(Integer, default=0)
        memory_mb = Column(Integer, default=0)
        disk_mb = Column(Integer, default=0)
        network_mbps = Column(Integer, default=0)
        
        # 状态
        status = Column(SQLEnum(AllocationStatusEnum), default=AllocationStatusEnum.PENDING)
        priority = Column(Integer, default=1)
        
        # 配置
        strategy = Column(SQLEnum(AllocationStrategyEnum), default=AllocationStrategyEnum.BEST_FIT)
        labels_affinity = Column(JSON, nullable=True)
        
        # 时间信息
        requested_at = Column(DateTime, default=datetime.utcnow)
        allocated_at = Column(DateTime, nullable=True)
        released_at = Column(DateTime, nullable=True)
        expires_at = Column(DateTime, nullable=True)
        
        # 租约信息
        lease_id = Column(String(64), nullable=True)
        lease_duration_seconds = Column(Integer, nullable=True)
        
        # 元数据
        allocation_metadata = Column('metadata', JSON, default=dict)
        
        # 时间戳
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # 索引
        __table_args__ = (
            Index('idx_allocation_status', 'status'),
            Index('idx_allocation_task', 'task_id'),
            Index('idx_allocation_node', 'node_id', 'status'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'node_id': self.node_id,
                'task_id': self.task_id,
                'user_id': self.user_id,
                'gpu_indices': self.gpu_indices,
                'gpu_count': self.gpu_count,
                'gpu_memory_mb': self.gpu_memory_mb,
                'cpu_cores': self.cpu_cores,
                'memory_mb': self.memory_mb,
                'disk_mb': self.disk_mb,
                'network_mbps': self.network_mbps,
                'status': self.status.value if self.status else None,
                'priority': self.priority,
                'strategy': self.strategy.value if self.strategy else None,
                'labels_affinity': self.labels_affinity,
                'requested_at': self.requested_at.isoformat() if self.requested_at else None,
                'allocated_at': self.allocated_at.isoformat() if self.allocated_at else None,
                'released_at': self.released_at.isoformat() if self.released_at else None,
                'expires_at': self.expires_at.isoformat() if self.expires_at else None,
                'lease_id': self.lease_id,
                'lease_duration_seconds': self.lease_duration_seconds,
                'metadata': self.allocation_metadata,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None
            }


    class GPUUsageHistoryModel(Base):
        """GPU 使用历史模型"""
        __tablename__ = 'gpu_usage_history'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 关联信息
        node_id = Column(String(64), ForeignKey('gpu_nodes.id'), nullable=False)
        gpu_id = Column(String(64), ForeignKey('gpu_devices.id'), nullable=True)
        gpu_index = Column(Integer, nullable=True)
        
        # 时间段
        timestamp = Column(DateTime, nullable=False, index=True)
        period_type = Column(String(32), nullable=False)  # minute, hour, day
        
        # 使用指标
        avg_utilization_percent = Column(Float, default=0.0)
        max_utilization_percent = Column(Float, default=0.0)
        min_utilization_percent = Column(Float, default=0.0)
        
        avg_memory_used_mb = Column(Integer, default=0)
        max_memory_used_mb = Column(Integer, default=0)
        min_memory_used_mb = Column(Integer, default=0)
        
        avg_temperature_celsius = Column(Float, nullable=True)
        max_temperature_celsius = Column(Float, nullable=True)
        
        avg_power_watts = Column(Float, nullable=True)
        max_power_watts = Column(Float, nullable=True)
        
        # 分配统计
        allocation_count = Column(Integer, default=0)
        active_time_seconds = Column(Integer, default=0)
        idle_time_seconds = Column(Integer, default=0)
        
        # 时间戳
        created_at = Column(DateTime, default=datetime.utcnow)
        
        # 索引
        __table_args__ = (
            Index('idx_usage_timestamp', 'timestamp', 'period_type'),
            Index('idx_usage_node', 'node_id', 'timestamp'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'node_id': self.node_id,
                'gpu_id': self.gpu_id,
                'gpu_index': self.gpu_index,
                'timestamp': self.timestamp.isoformat() if self.timestamp else None,
                'period_type': self.period_type,
                'avg_utilization_percent': self.avg_utilization_percent,
                'max_utilization_percent': self.max_utilization_percent,
                'min_utilization_percent': self.min_utilization_percent,
                'avg_memory_used_mb': self.avg_memory_used_mb,
                'max_memory_used_mb': self.max_memory_used_mb,
                'min_memory_used_mb': self.min_memory_used_mb,
                'avg_temperature_celsius': self.avg_temperature_celsius,
                'max_temperature_celsius': self.max_temperature_celsius,
                'avg_power_watts': self.avg_power_watts,
                'max_power_watts': self.max_power_watts,
                'allocation_count': self.allocation_count,
                'active_time_seconds': self.active_time_seconds,
                'idle_time_seconds': self.idle_time_seconds,
                'created_at': self.created_at.isoformat() if self.created_at else None
            }


    class GPUReservationModel(Base):
        """GPU 预留模型"""
        __tablename__ = 'gpu_reservations'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 预留信息
        name = Column(String(256), nullable=False)
        description = Column(Text, nullable=True)
        
        # 资源需求
        gpu_count = Column(Integer, default=0)
        gpu_memory_mb = Column(Integer, default=0)
        cpu_cores = Column(Integer, default=0)
        memory_mb = Column(Integer, default=0)
        
        # 节点偏好
        preferred_node_ids = Column(JSON, nullable=True)
        labels_affinity = Column(JSON, nullable=True)
        
        # 时间范围
        start_time = Column(DateTime, nullable=False)
        end_time = Column(DateTime, nullable=False)
        
        # 状态
        is_active = Column(Boolean, default=True)
        is_recurring = Column(Boolean, default=False)
        recurrence_pattern = Column(String(128), nullable=True)  # cron expression
        
        # 实际分配
        actual_allocation_id = Column(String(64), nullable=True)
        actual_node_id = Column(String(64), nullable=True)
        actual_gpu_indices = Column(JSON, nullable=True)
        
        # 元数据
        reservation_metadata = Column('metadata', JSON, default=dict)
        
        # 所有者
        created_by = Column(String(64), nullable=True)
        
        # 时间戳
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # 索引
        __table_args__ = (
            Index('idx_reservation_time', 'start_time', 'end_time'),
            Index('idx_reservation_active', 'is_active', 'start_time'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'name': self.name,
                'description': self.description,
                'gpu_count': self.gpu_count,
                'gpu_memory_mb': self.gpu_memory_mb,
                'cpu_cores': self.cpu_cores,
                'memory_mb': self.memory_mb,
                'preferred_node_ids': self.preferred_node_ids,
                'labels_affinity': self.labels_affinity,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'is_active': self.is_active,
                'is_recurring': self.is_recurring,
                'recurrence_pattern': self.recurrence_pattern,
                'actual_allocation_id': self.actual_allocation_id,
                'actual_node_id': self.actual_node_id,
                'actual_gpu_indices': self.actual_gpu_indices,
                'metadata': self.reservation_metadata,
                'created_by': self.created_by,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None
            }

else:
    # 无 SQLAlchemy 时的占位类
    class GPUNodeModel: pass
    class GPUDeviceModel: pass
    class GPUAllocationModel: pass
    class GPUUsageHistoryModel: pass
    class GPUReservationModel: pass


# ==================== 数据类 ====================

@dataclass
class GPUNodeData:
    """GPU 节点数据类"""
    id: str
    hostname: str
    status: str = NodeStatusEnum.ONLINE.value
    ip_address: str = None
    port: int = 8080
    tenant_id: str = None
    is_healthy: bool = True
    last_heartbeat: datetime = None
    gpu_count: int = 0
    total_gpu_memory_mb: int = 0
    used_gpu_memory_mb: int = 0
    cpu_cores: int = 0
    cpu_used: float = 0.0
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    disk_total_mb: int = 0
    disk_used_mb: int = 0
    network_bandwidth_mbps: int = 1000
    labels: Dict[str, str] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ['last_heartbeat', 'created_at', 'updated_at']:
            if result[key] and isinstance(result[key], datetime):
                result[key] = result[key].isoformat()
        return result


@dataclass
class GPUDeviceData:
    """GPU 设备数据类"""
    id: str
    node_id: str
    gpu_index: int
    status: str = GPUStatusEnum.AVAILABLE.value
    tenant_id: str = None
    uuid: str = None
    name: str = None
    is_available: bool = True
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    memory_free_mb: int = 0
    utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    temperature_celsius: float = None
    power_usage_watts: float = None
    power_limit_watts: float = None
    driver_version: str = None
    cuda_version: str = None
    current_allocation_id: str = None
    allocated_to_task_id: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ['created_at', 'updated_at']:
            if result[key] and isinstance(result[key], datetime):
                result[key] = result[key].isoformat()
        return result


@dataclass
class GPUAllocationData:
    """GPU 分配数据类"""
    id: str
    node_id: str
    gpu_indices: List[int]
    status: str = AllocationStatusEnum.PENDING.value
    tenant_id: str = None
    task_id: str = None
    user_id: str = None
    gpu_count: int = 0
    gpu_memory_mb: int = 0
    cpu_cores: int = 0
    memory_mb: int = 0
    disk_mb: int = 0
    network_mbps: int = 0
    priority: int = 1
    strategy: str = AllocationStrategyEnum.BEST_FIT.value
    labels_affinity: Dict[str, str] = None
    requested_at: datetime = None
    allocated_at: datetime = None
    released_at: datetime = None
    expires_at: datetime = None
    lease_id: str = None
    lease_duration_seconds: int = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.requested_at is None:
            self.requested_at = datetime.utcnow()
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
        if self.gpu_count == 0:
            self.gpu_count = len(self.gpu_indices)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ['requested_at', 'allocated_at', 'released_at', 'expires_at', 'created_at', 'updated_at']:
            if result[key] and isinstance(result[key], datetime):
                result[key] = result[key].isoformat()
        return result


@dataclass
class ResourceRequirementData:
    """资源需求数据类"""
    cpu_cores: int = 1
    memory_mb: int = 1024
    gpu_count: int = 0
    gpu_memory_mb: int = 0
    disk_mb: int = 1024
    network_mbps: int = 100
    priority: int = 1
    labels_affinity: Dict[str, str] = None
    prefer_same_node: bool = True
    min_gpu_memory_mb: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

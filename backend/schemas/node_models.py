#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""节点管理数据模型

定义节点管理相关的 SQLAlchemy ORM 模型和数据类。
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
    UNKNOWN = "unknown"           # 未知
    HEALTHY = "healthy"           # 健康
    UNHEALTHY = "unhealthy"       # 不健康
    OFFLINE = "offline"           # 离线
    MAINTENANCE = "maintenance"   # 维护中
    DRAINING = "draining"         # 排空中
    CORDONED = "cordoned"         # 已隔离


class NodeTypeEnum(str, Enum):
    """节点类型"""
    MASTER = "master"             # 主节点
    WORKER = "worker"             # 工作节点
    GPU = "gpu"                   # GPU节点
    STORAGE = "storage"           # 存储节点
    EDGE = "edge"                 # 边缘节点


class NodeRoleEnum(str, Enum):
    """节点角色"""
    TRAINING = "training"         # 训练节点
    INFERENCE = "inference"       # 推理节点
    DATA = "data"                 # 数据处理节点
    GENERAL = "general"           # 通用节点


class GPUVendorEnum(str, Enum):
    """GPU厂商"""
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    OTHER = "other"


class HeartbeatStatusEnum(str, Enum):
    """心跳状态"""
    NORMAL = "normal"             # 正常
    DELAYED = "delayed"           # 延迟
    TIMEOUT = "timeout"           # 超时
    MISSED = "missed"             # 错过


# ==================== SQLAlchemy 模型 ====================

if HAS_SQLALCHEMY:
    
    class NodeModel(Base):
        """节点模型
        
        存储集群中节点的基本信息
        """
        __tablename__ = 'cluster_nodes'
        
        # 主键
        id = Column(String(64), primary_key=True)
        
        # 基本信息
        node_id = Column(String(128), unique=True, nullable=False, index=True)
        hostname = Column(String(256), nullable=False)
        ip_address = Column(String(64), nullable=False)
        port = Column(Integer, default=22)
        
        # 类型和角色
        node_type = Column(SQLEnum(NodeTypeEnum), default=NodeTypeEnum.WORKER)
        node_role = Column(SQLEnum(NodeRoleEnum), default=NodeRoleEnum.GENERAL)
        
        # 状态
        status = Column(SQLEnum(NodeStatusEnum), default=NodeStatusEnum.UNKNOWN)
        is_schedulable = Column(Boolean, default=True)
        is_ready = Column(Boolean, default=False)
        
        # 硬件信息
        cpu_count = Column(Integer, default=0)
        cpu_model = Column(String(256))
        memory_total_mb = Column(Integer, default=0)
        disk_total_mb = Column(Integer, default=0)
        
        # GPU信息（JSON存储）
        gpu_count = Column(Integer, default=0)
        gpu_info = Column(JSON, default=list)
        
        # 资源使用情况
        cpu_utilization = Column(Float, default=0.0)
        memory_used_mb = Column(Integer, default=0)
        memory_utilization = Column(Float, default=0.0)
        disk_used_mb = Column(Integer, default=0)
        disk_utilization = Column(Float, default=0.0)
        
        # 网络信息
        network_rx_bytes = Column(Integer, default=0)
        network_tx_bytes = Column(Integer, default=0)
        
        # 任务信息
        running_tasks = Column(Integer, default=0)
        max_tasks = Column(Integer, default=1)
        
        # 健康检查
        health_check_failures = Column(Integer, default=0)
        max_health_check_failures = Column(Integer, default=3)
        last_health_check = Column(DateTime)
        
        # 心跳信息
        last_heartbeat = Column(DateTime)
        heartbeat_interval_seconds = Column(Integer, default=30)
        heartbeat_status = Column(SQLEnum(HeartbeatStatusEnum), default=HeartbeatStatusEnum.NORMAL)
        
        # 标签和注解
        labels = Column(JSON, default=dict)
        annotations = Column(JSON, default=dict)
        taints = Column(JSON, default=list)
        
        # 租户信息
        tenant_id = Column(String(64), index=True)
        
        # 时间戳
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        registered_at = Column(DateTime)
        
        # 额外元数据
        metadata_ = Column('metadata', JSON, default=dict)
        
        # 索引
        __table_args__ = (
            Index('ix_nodes_status', 'status'),
            Index('ix_nodes_type', 'node_type'),
            Index('ix_nodes_tenant', 'tenant_id'),
            Index('ix_nodes_heartbeat', 'last_heartbeat'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'node_id': self.node_id,
                'hostname': self.hostname,
                'ip_address': self.ip_address,
                'port': self.port,
                'node_type': self.node_type.value if self.node_type else None,
                'node_role': self.node_role.value if self.node_role else None,
                'status': self.status.value if self.status else None,
                'is_schedulable': self.is_schedulable,
                'is_ready': self.is_ready,
                'cpu_count': self.cpu_count,
                'cpu_model': self.cpu_model,
                'memory_total_mb': self.memory_total_mb,
                'disk_total_mb': self.disk_total_mb,
                'gpu_count': self.gpu_count,
                'gpu_info': self.gpu_info,
                'cpu_utilization': self.cpu_utilization,
                'memory_used_mb': self.memory_used_mb,
                'memory_utilization': self.memory_utilization,
                'disk_used_mb': self.disk_used_mb,
                'disk_utilization': self.disk_utilization,
                'running_tasks': self.running_tasks,
                'max_tasks': self.max_tasks,
                'health_check_failures': self.health_check_failures,
                'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
                'heartbeat_status': self.heartbeat_status.value if self.heartbeat_status else None,
                'labels': self.labels,
                'annotations': self.annotations,
                'tenant_id': self.tenant_id,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            }
    
    
    class NodeGPUModel(Base):
        """节点GPU模型
        
        存储节点上每个GPU的详细信息
        """
        __tablename__ = 'node_gpus'
        
        id = Column(String(64), primary_key=True)
        node_id = Column(String(64), ForeignKey('cluster_nodes.id'), nullable=False, index=True)
        
        # GPU信息
        gpu_index = Column(Integer, nullable=False)
        gpu_uuid = Column(String(128), unique=True)
        gpu_name = Column(String(256))
        gpu_vendor = Column(SQLEnum(GPUVendorEnum), default=GPUVendorEnum.NVIDIA)
        
        # 内存信息
        memory_total_mb = Column(Integer, default=0)
        memory_used_mb = Column(Integer, default=0)
        memory_utilization = Column(Float, default=0.0)
        
        # 计算能力
        compute_capability = Column(String(32))
        cuda_cores = Column(Integer)
        tensor_cores = Column(Integer)
        
        # 温度和功耗
        temperature_celsius = Column(Float)
        power_usage_watts = Column(Float)
        power_limit_watts = Column(Float)
        
        # 状态
        is_available = Column(Boolean, default=True)
        is_healthy = Column(Boolean, default=True)
        
        # 运行任务
        running_processes = Column(JSON, default=list)
        
        # 时间戳
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'node_id': self.node_id,
                'gpu_index': self.gpu_index,
                'gpu_uuid': self.gpu_uuid,
                'gpu_name': self.gpu_name,
                'gpu_vendor': self.gpu_vendor.value if self.gpu_vendor else None,
                'memory_total_mb': self.memory_total_mb,
                'memory_used_mb': self.memory_used_mb,
                'memory_utilization': self.memory_utilization,
                'temperature_celsius': self.temperature_celsius,
                'power_usage_watts': self.power_usage_watts,
                'is_available': self.is_available,
                'is_healthy': self.is_healthy,
            }
    
    
    class NodeHeartbeatModel(Base):
        """节点心跳记录模型
        
        记录节点心跳历史
        """
        __tablename__ = 'node_heartbeats'
        
        id = Column(String(64), primary_key=True)
        node_id = Column(String(64), ForeignKey('cluster_nodes.id'), nullable=False, index=True)
        
        # 心跳信息
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        status = Column(SQLEnum(HeartbeatStatusEnum), default=HeartbeatStatusEnum.NORMAL)
        latency_ms = Column(Float)
        
        # 资源快照
        cpu_utilization = Column(Float)
        memory_utilization = Column(Float)
        disk_utilization = Column(Float)
        gpu_utilization = Column(JSON)  # 每个GPU的利用率
        
        # 网络流量
        network_rx_bytes = Column(Integer)
        network_tx_bytes = Column(Integer)
        
        # 运行任务
        running_tasks = Column(Integer)
        
        # 额外指标
        metrics = Column(JSON, default=dict)
        
        __table_args__ = (
            Index('ix_heartbeat_node_time', 'node_id', 'timestamp'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'node_id': self.node_id,
                'timestamp': self.timestamp.isoformat() if self.timestamp else None,
                'status': self.status.value if self.status else None,
                'latency_ms': self.latency_ms,
                'cpu_utilization': self.cpu_utilization,
                'memory_utilization': self.memory_utilization,
                'disk_utilization': self.disk_utilization,
                'gpu_utilization': self.gpu_utilization,
                'running_tasks': self.running_tasks,
                'metrics': self.metrics,
            }
    
    
    class NodeEventModel(Base):
        """节点事件模型
        
        记录节点相关事件
        """
        __tablename__ = 'node_events'
        
        id = Column(String(64), primary_key=True)
        node_id = Column(String(64), ForeignKey('cluster_nodes.id'), nullable=False, index=True)
        
        # 事件信息
        event_type = Column(String(64), nullable=False, index=True)
        event_reason = Column(String(256))
        event_message = Column(Text)
        
        # 事件级别
        severity = Column(String(32), default='info')  # info, warning, error, critical
        
        # 来源
        source = Column(String(128))
        
        # 时间戳
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        first_seen = Column(DateTime, default=datetime.utcnow)
        last_seen = Column(DateTime, default=datetime.utcnow)
        count = Column(Integer, default=1)
        
        # 额外数据
        metadata_ = Column('metadata', JSON, default=dict)
        
        __table_args__ = (
            Index('ix_events_node_type', 'node_id', 'event_type'),
            Index('ix_events_timestamp', 'timestamp'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'node_id': self.node_id,
                'event_type': self.event_type,
                'event_reason': self.event_reason,
                'event_message': self.event_message,
                'severity': self.severity,
                'source': self.source,
                'timestamp': self.timestamp.isoformat() if self.timestamp else None,
                'count': self.count,
                'metadata': self.metadata_,
            }


# ==================== 数据类 ====================

@dataclass
class GPUInfo:
    """GPU信息数据类"""
    gpu_index: int
    gpu_uuid: str
    gpu_name: str
    vendor: str = 'nvidia'
    memory_total_mb: int = 0
    memory_used_mb: int = 0
    temperature_celsius: float = 0.0
    power_usage_watts: float = 0.0
    is_available: bool = True
    is_healthy: bool = True
    running_processes: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def memory_utilization(self) -> float:
        if self.memory_total_mb > 0:
            return (self.memory_used_mb / self.memory_total_mb) * 100
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gpu_index': self.gpu_index,
            'gpu_uuid': self.gpu_uuid,
            'gpu_name': self.gpu_name,
            'vendor': self.vendor,
            'memory_total_mb': self.memory_total_mb,
            'memory_used_mb': self.memory_used_mb,
            'memory_utilization': self.memory_utilization,
            'temperature_celsius': self.temperature_celsius,
            'power_usage_watts': self.power_usage_watts,
            'is_available': self.is_available,
            'is_healthy': self.is_healthy,
        }


@dataclass
class NodeInfo:
    """节点信息数据类"""
    node_id: str
    hostname: str
    ip_address: str
    port: int = 22
    
    # 类型和状态
    node_type: str = 'worker'
    node_role: str = 'general'
    status: str = 'unknown'
    is_schedulable: bool = True
    is_ready: bool = False
    
    # 硬件信息
    cpu_count: int = 0
    cpu_model: Optional[str] = None
    memory_total_mb: int = 0
    disk_total_mb: int = 0
    
    # GPU信息
    gpu_count: int = 0
    gpus: List[GPUInfo] = field(default_factory=list)
    
    # 资源使用
    cpu_utilization: float = 0.0
    memory_used_mb: int = 0
    memory_utilization: float = 0.0
    disk_used_mb: int = 0
    disk_utilization: float = 0.0
    
    # 网络
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    
    # 任务
    running_tasks: int = 0
    max_tasks: int = 1
    
    # 健康检查
    health_check_failures: int = 0
    last_heartbeat: Optional[datetime] = None
    
    # 标签
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    taints: List[Dict[str, Any]] = field(default_factory=list)
    
    # 租户
    tenant_id: Optional[str] = None
    
    # 时间戳
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @property
    def is_healthy(self) -> bool:
        if self.last_heartbeat is None:
            return False
        return (
            self.status == 'healthy' and
            self.health_check_failures < 3 and
            (datetime.utcnow() - self.last_heartbeat).total_seconds() < 60
        )
    
    @property
    def can_accept_task(self) -> bool:
        return (
            self.is_healthy and
            self.is_schedulable and
            self.running_tasks < self.max_tasks
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'ip_address': self.ip_address,
            'port': self.port,
            'node_type': self.node_type,
            'node_role': self.node_role,
            'status': self.status,
            'is_schedulable': self.is_schedulable,
            'is_ready': self.is_ready,
            'is_healthy': self.is_healthy,
            'can_accept_task': self.can_accept_task,
            'cpu_count': self.cpu_count,
            'cpu_model': self.cpu_model,
            'memory_total_mb': self.memory_total_mb,
            'disk_total_mb': self.disk_total_mb,
            'gpu_count': self.gpu_count,
            'gpus': [g.to_dict() for g in self.gpus],
            'cpu_utilization': self.cpu_utilization,
            'memory_used_mb': self.memory_used_mb,
            'memory_utilization': self.memory_utilization,
            'disk_used_mb': self.disk_used_mb,
            'disk_utilization': self.disk_utilization,
            'running_tasks': self.running_tasks,
            'max_tasks': self.max_tasks,
            'labels': self.labels,
            'annotations': self.annotations,
            'tenant_id': self.tenant_id,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class NodeHeartbeat:
    """节点心跳数据类"""
    node_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: str = 'normal'
    latency_ms: float = 0.0
    
    # 资源快照
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    disk_utilization: float = 0.0
    gpu_utilization: List[float] = field(default_factory=list)
    
    # 网络
    network_rx_bytes: int = 0
    network_tx_bytes: int = 0
    
    # 任务
    running_tasks: int = 0
    
    # 额外指标
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'status': self.status,
            'latency_ms': self.latency_ms,
            'cpu_utilization': self.cpu_utilization,
            'memory_utilization': self.memory_utilization,
            'disk_utilization': self.disk_utilization,
            'gpu_utilization': self.gpu_utilization,
            'network_rx_bytes': self.network_rx_bytes,
            'network_tx_bytes': self.network_tx_bytes,
            'running_tasks': self.running_tasks,
            'metrics': self.metrics,
        }


@dataclass
class NodeEvent:
    """节点事件数据类"""
    id: str
    node_id: str
    event_type: str
    event_reason: str = ''
    event_message: str = ''
    severity: str = 'info'
    source: str = ''
    timestamp: datetime = field(default_factory=datetime.utcnow)
    count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'node_id': self.node_id,
            'event_type': self.event_type,
            'event_reason': self.event_reason,
            'event_message': self.event_message,
            'severity': self.severity,
            'source': self.source,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'count': self.count,
            'metadata': self.metadata,
        }


@dataclass
class ClusterSummary:
    """集群摘要数据类"""
    total_nodes: int = 0
    healthy_nodes: int = 0
    unhealthy_nodes: int = 0
    offline_nodes: int = 0
    
    total_cpus: int = 0
    total_memory_mb: int = 0
    total_disk_mb: int = 0
    total_gpus: int = 0
    
    used_memory_mb: int = 0
    used_disk_mb: int = 0
    
    avg_cpu_utilization: float = 0.0
    avg_memory_utilization: float = 0.0
    avg_disk_utilization: float = 0.0
    
    running_tasks: int = 0
    max_tasks: int = 0
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_nodes': self.total_nodes,
            'healthy_nodes': self.healthy_nodes,
            'unhealthy_nodes': self.unhealthy_nodes,
            'offline_nodes': self.offline_nodes,
            'total_cpus': self.total_cpus,
            'total_memory_mb': self.total_memory_mb,
            'total_disk_mb': self.total_disk_mb,
            'total_gpus': self.total_gpus,
            'used_memory_mb': self.used_memory_mb,
            'used_disk_mb': self.used_disk_mb,
            'avg_cpu_utilization': self.avg_cpu_utilization,
            'avg_memory_utilization': self.avg_memory_utilization,
            'avg_disk_utilization': self.avg_disk_utilization,
            'running_tasks': self.running_tasks,
            'max_tasks': self.max_tasks,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


# ==================== 导出 ====================

__all__ = [
    # 枚举
    'NodeStatusEnum',
    'NodeTypeEnum',
    'NodeRoleEnum',
    'GPUVendorEnum',
    'HeartbeatStatusEnum',
    
    # SQLAlchemy 模型
    'NodeModel',
    'NodeGPUModel',
    'NodeHeartbeatModel',
    'NodeEventModel',
    
    # 数据类
    'GPUInfo',
    'NodeInfo',
    'NodeHeartbeat',
    'NodeEvent',
    'ClusterSummary',
]

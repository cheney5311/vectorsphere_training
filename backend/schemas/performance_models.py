#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""性能模块数据模型

定义异步任务、性能指标、告警等相关的 SQLAlchemy ORM 模型和数据类。
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

class TaskStatusEnum(str, Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TaskPriorityEnum(str, Enum):
    """任务优先级"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class MetricTypeEnum(str, Enum):
    """指标类型"""
    SYSTEM = "system"
    GPU = "gpu"
    TRAINING = "training"
    DATABASE = "database"
    ASYNC_PROCESSOR = "async_processor"
    CUSTOM = "custom"


class AlertLevelEnum(str, Enum):
    """告警级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatusEnum(str, Enum):
    """告警状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class ResourceTypeEnum(str, Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    IO = "io"


# ==================== SQLAlchemy 模型 ====================

if HAS_SQLALCHEMY:

    class AsyncTaskModel(Base):
        """异步任务模型"""
        __tablename__ = 'async_tasks'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 任务基本信息
        name = Column(String(256), nullable=False)
        category = Column(String(64), nullable=True, index=True)
        description = Column(Text, nullable=True)
        
        # 状态
        status = Column(SQLEnum(TaskStatusEnum), default=TaskStatusEnum.PENDING, index=True)
        priority = Column(SQLEnum(TaskPriorityEnum), default=TaskPriorityEnum.NORMAL)
        
        # 参数和结果
        params = Column(JSON, default=dict)
        result = Column(JSON, nullable=True)
        error_message = Column(Text, nullable=True)
        error_traceback = Column(Text, nullable=True)
        
        # 时间信息
        created_at = Column(DateTime, default=datetime.utcnow, index=True)
        started_at = Column(DateTime, nullable=True)
        completed_at = Column(DateTime, nullable=True)
        timeout = Column(Float, nullable=True)
        
        # 执行信息
        execution_time = Column(Float, nullable=True)
        retry_count = Column(Integer, default=0)
        max_retries = Column(Integer, default=3)
        
        # 队列信息
        queue_position = Column(Integer, nullable=True)
        worker_id = Column(String(64), nullable=True)
        
        # 元数据
        task_metadata = Column('metadata', JSON, default=dict)
        
        # 所有者
        created_by = Column(String(64), nullable=True)
        
        # 索引
        __table_args__ = (
            Index('idx_task_status_created', 'status', 'created_at'),
            Index('idx_task_tenant_status', 'tenant_id', 'status'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'name': self.name,
                'category': self.category,
                'description': self.description,
                'status': self.status.value if self.status else None,
                'priority': self.priority.value if self.priority else None,
                'params': self.params,
                'result': self.result,
                'error_message': self.error_message,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'started_at': self.started_at.isoformat() if self.started_at else None,
                'completed_at': self.completed_at.isoformat() if self.completed_at else None,
                'timeout': self.timeout,
                'execution_time': self.execution_time,
                'retry_count': self.retry_count,
                'queue_position': self.queue_position,
                'worker_id': self.worker_id,
                'metadata': self.task_metadata,
                'created_by': self.created_by
            }


    class PerformanceMetricModel(Base):
        """性能指标模型"""
        __tablename__ = 'performance_metrics'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 指标信息
        metric_type = Column(SQLEnum(MetricTypeEnum), nullable=False, index=True)
        metric_name = Column(String(128), nullable=False, index=True)
        metric_value = Column(Float, nullable=False)
        metric_unit = Column(String(32), nullable=True)
        
        # 时间信息
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        period_type = Column(String(32), nullable=True)  # minute, hour, day
        
        # 资源信息
        resource_id = Column(String(64), nullable=True)
        resource_type = Column(SQLEnum(ResourceTypeEnum), nullable=True)
        
        # 统计信息
        min_value = Column(Float, nullable=True)
        max_value = Column(Float, nullable=True)
        avg_value = Column(Float, nullable=True)
        sample_count = Column(Integer, default=1)
        
        # 标签
        tags = Column(JSON, default=dict)
        
        # 索引
        __table_args__ = (
            Index('idx_metric_timestamp', 'timestamp'),
            Index('idx_metric_type_name', 'metric_type', 'metric_name'),
            Index('idx_metric_resource', 'resource_type', 'resource_id'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'metric_type': self.metric_type.value if self.metric_type else None,
                'metric_name': self.metric_name,
                'metric_value': self.metric_value,
                'metric_unit': self.metric_unit,
                'timestamp': self.timestamp.isoformat() if self.timestamp else None,
                'period_type': self.period_type,
                'resource_id': self.resource_id,
                'resource_type': self.resource_type.value if self.resource_type else None,
                'min_value': self.min_value,
                'max_value': self.max_value,
                'avg_value': self.avg_value,
                'sample_count': self.sample_count,
                'tags': self.tags
            }


    class AlertModel(Base):
        """告警模型"""
        __tablename__ = 'performance_alerts'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 告警基本信息
        name = Column(String(256), nullable=False)
        description = Column(Text, nullable=True)
        rule_id = Column(String(64), nullable=True, index=True)
        
        # 状态
        status = Column(SQLEnum(AlertStatusEnum), default=AlertStatusEnum.ACTIVE, index=True)
        level = Column(SQLEnum(AlertLevelEnum), default=AlertLevelEnum.MEDIUM)
        
        # 指标信息
        metric_type = Column(SQLEnum(MetricTypeEnum), nullable=True)
        metric_name = Column(String(128), nullable=True)
        metric_value = Column(Float, nullable=True)
        threshold = Column(Float, nullable=True)
        
        # 时间信息
        triggered_at = Column(DateTime, default=datetime.utcnow, index=True)
        acknowledged_at = Column(DateTime, nullable=True)
        resolved_at = Column(DateTime, nullable=True)
        duration_seconds = Column(Float, nullable=True)
        
        # 处理信息
        acknowledged_by = Column(String(64), nullable=True)
        resolved_by = Column(String(64), nullable=True)
        resolution_notes = Column(Text, nullable=True)
        
        # 通知状态
        notification_sent = Column(Boolean, default=False)
        notification_channels = Column(JSON, default=list)
        
        # 元数据
        alert_metadata = Column('metadata', JSON, default=dict)
        
        # 索引
        __table_args__ = (
            Index('idx_alert_status_level', 'status', 'level'),
            Index('idx_alert_triggered', 'triggered_at'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'name': self.name,
                'description': self.description,
                'rule_id': self.rule_id,
                'status': self.status.value if self.status else None,
                'level': self.level.value if self.level else None,
                'metric_type': self.metric_type.value if self.metric_type else None,
                'metric_name': self.metric_name,
                'metric_value': self.metric_value,
                'threshold': self.threshold,
                'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
                'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
                'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
                'duration_seconds': self.duration_seconds,
                'acknowledged_by': self.acknowledged_by,
                'resolved_by': self.resolved_by,
                'resolution_notes': self.resolution_notes,
                'notification_sent': self.notification_sent,
                'notification_channels': self.notification_channels,
                'metadata': self.alert_metadata
            }


    class AlertRuleModel(Base):
        """告警规则模型"""
        __tablename__ = 'alert_rules'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 规则基本信息
        name = Column(String(256), nullable=False)
        description = Column(Text, nullable=True)
        
        # 条件
        metric_type = Column(SQLEnum(MetricTypeEnum), nullable=False)
        metric_name = Column(String(128), nullable=False)
        operator = Column(String(16), nullable=False)  # >, <, >=, <=, ==, !=
        threshold = Column(Float, nullable=False)
        duration = Column(Integer, default=0)  # 持续时间（秒）
        
        # 告警级别
        severity = Column(SQLEnum(AlertLevelEnum), default=AlertLevelEnum.MEDIUM)
        
        # 状态
        enabled = Column(Boolean, default=True, index=True)
        
        # 通知配置
        notification_channels = Column(JSON, default=list)
        cooldown_seconds = Column(Integer, default=300)  # 冷却时间
        
        # 元数据
        rule_metadata = Column('metadata', JSON, default=dict)
        
        # 时间戳
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        created_by = Column(String(64), nullable=True)
        
        # 索引
        __table_args__ = (
            Index('idx_rule_enabled', 'enabled'),
            Index('idx_rule_metric', 'metric_type', 'metric_name'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'name': self.name,
                'description': self.description,
                'metric_type': self.metric_type.value if self.metric_type else None,
                'metric_name': self.metric_name,
                'operator': self.operator,
                'threshold': self.threshold,
                'duration': self.duration,
                'severity': self.severity.value if self.severity else None,
                'enabled': self.enabled,
                'notification_channels': self.notification_channels,
                'cooldown_seconds': self.cooldown_seconds,
                'metadata': self.rule_metadata,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None,
                'created_by': self.created_by
            }


    class SystemSnapshotModel(Base):
        """系统快照模型"""
        __tablename__ = 'system_snapshots'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 时间
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        
        # CPU 指标
        cpu_percent = Column(Float, nullable=True)
        cpu_count = Column(Integer, nullable=True)
        load_average_1m = Column(Float, nullable=True)
        load_average_5m = Column(Float, nullable=True)
        load_average_15m = Column(Float, nullable=True)
        
        # 内存指标
        memory_percent = Column(Float, nullable=True)
        memory_total_gb = Column(Float, nullable=True)
        memory_used_gb = Column(Float, nullable=True)
        memory_available_gb = Column(Float, nullable=True)
        
        # 磁盘指标
        disk_percent = Column(Float, nullable=True)
        disk_total_gb = Column(Float, nullable=True)
        disk_used_gb = Column(Float, nullable=True)
        disk_free_gb = Column(Float, nullable=True)
        
        # 网络指标
        network_bytes_sent = Column(Float, nullable=True)
        network_bytes_recv = Column(Float, nullable=True)
        network_connections = Column(Integer, nullable=True)
        
        # 进程信息
        process_count = Column(Integer, nullable=True)
        
        # 元数据
        snapshot_metadata = Column('metadata', JSON, default=dict)
        
        # 索引
        __table_args__ = (
            Index('idx_snapshot_timestamp', 'timestamp'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'timestamp': self.timestamp.isoformat() if self.timestamp else None,
                'cpu': {
                    'percent': self.cpu_percent,
                    'count': self.cpu_count,
                    'load_average': [self.load_average_1m, self.load_average_5m, self.load_average_15m]
                },
                'memory': {
                    'percent': self.memory_percent,
                    'total_gb': self.memory_total_gb,
                    'used_gb': self.memory_used_gb,
                    'available_gb': self.memory_available_gb
                },
                'disk': {
                    'percent': self.disk_percent,
                    'total_gb': self.disk_total_gb,
                    'used_gb': self.disk_used_gb,
                    'free_gb': self.disk_free_gb
                },
                'network': {
                    'bytes_sent': self.network_bytes_sent,
                    'bytes_recv': self.network_bytes_recv,
                    'connections': self.network_connections
                },
                'process_count': self.process_count,
                'metadata': self.snapshot_metadata
            }

else:
    # 无 SQLAlchemy 时的占位类
    class AsyncTaskModel: pass
    class PerformanceMetricModel: pass
    class AlertModel: pass
    class AlertRuleModel: pass
    class SystemSnapshotModel: pass


# ==================== 数据类 ====================

@dataclass
class AsyncTaskData:
    """异步任务数据类"""
    id: str
    name: str
    status: str = TaskStatusEnum.PENDING.value
    category: str = None
    description: str = None
    priority: str = TaskPriorityEnum.NORMAL.value
    params: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error_message: str = None
    created_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None
    timeout: float = None
    execution_time: float = None
    retry_count: int = 0
    queue_position: int = None
    worker_id: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: str = None
    tenant_id: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ['created_at', 'started_at', 'completed_at']:
            if result[key] and isinstance(result[key], datetime):
                result[key] = result[key].isoformat()
        return result


@dataclass
class PerformanceMetricData:
    """性能指标数据类"""
    id: str
    metric_type: str
    metric_name: str
    metric_value: float
    metric_unit: str = None
    timestamp: datetime = None
    period_type: str = None
    resource_id: str = None
    resource_type: str = None
    min_value: float = None
    max_value: float = None
    avg_value: float = None
    sample_count: int = 1
    tags: Dict[str, str] = field(default_factory=dict)
    tenant_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        if result['timestamp'] and isinstance(result['timestamp'], datetime):
            result['timestamp'] = result['timestamp'].isoformat()
        return result


@dataclass
class AlertData:
    """告警数据类"""
    id: str
    name: str
    level: str = AlertLevelEnum.MEDIUM.value
    status: str = AlertStatusEnum.ACTIVE.value
    description: str = None
    rule_id: str = None
    metric_type: str = None
    metric_name: str = None
    metric_value: float = None
    threshold: float = None
    triggered_at: datetime = None
    acknowledged_at: datetime = None
    resolved_at: datetime = None
    duration_seconds: float = None
    acknowledged_by: str = None
    resolved_by: str = None
    resolution_notes: str = None
    notification_sent: bool = False
    notification_channels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tenant_id: str = None
    
    def __post_init__(self):
        if self.triggered_at is None:
            self.triggered_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for key in ['triggered_at', 'acknowledged_at', 'resolved_at']:
            if result[key] and isinstance(result[key], datetime):
                result[key] = result[key].isoformat()
        return result


@dataclass
class AlertRuleData:
    """告警规则数据类"""
    id: str
    name: str
    metric_type: str
    metric_name: str
    operator: str
    threshold: float
    severity: str = AlertLevelEnum.MEDIUM.value
    description: str = None
    duration: int = 0
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    cooldown_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = None
    updated_at: datetime = None
    created_by: str = None
    tenant_id: str = None
    
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
class SystemSnapshotData:
    """系统快照数据类"""
    id: str
    timestamp: datetime = None
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_average: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    memory_percent: float = 0.0
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    disk_percent: float = 0.0
    disk_total_gb: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    network_bytes_sent: float = 0.0
    network_bytes_recv: float = 0.0
    network_connections: int = 0
    process_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tenant_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'cpu': {
                'percent': self.cpu_percent,
                'count': self.cpu_count,
                'load_average': self.load_average
            },
            'memory': {
                'percent': self.memory_percent,
                'total_gb': self.memory_total_gb,
                'used_gb': self.memory_used_gb,
                'available_gb': self.memory_available_gb
            },
            'disk': {
                'percent': self.disk_percent,
                'total_gb': self.disk_total_gb,
                'used_gb': self.disk_used_gb,
                'free_gb': self.disk_free_gb
            },
            'network': {
                'bytes_sent': self.network_bytes_sent,
                'bytes_recv': self.network_bytes_recv,
                'connections': self.network_connections
            },
            'process_count': self.process_count,
            'metadata': self.metadata
        }

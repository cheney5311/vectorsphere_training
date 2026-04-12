#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调度器数据模型

定义调度任务、模板、执行记录等SQLAlchemy ORM模型。
"""

import json
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

try:
    from sqlalchemy import (
        Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
        Enum as SQLEnum, ForeignKey, Index, UniqueConstraint
    )
    from sqlalchemy.orm import relationship, declarative_base
    from sqlalchemy.dialects.postgresql import JSONB
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
    """任务状态枚举"""
    PENDING = "pending"
    SCHEDULED = "scheduled"
    QUEUED = "queued"
    EXECUTING = "executing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


class TaskPriorityEnum(str, Enum):
    """任务优先级枚举"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class TaskTypeEnum(str, Enum):
    """任务类型枚举"""
    TRAINING = "training"
    EVALUATION = "evaluation"
    INFERENCE = "inference"
    DATA_PROCESSING = "data_processing"
    MODEL_EXPORT = "model_export"
    CLEANUP = "cleanup"
    BACKUP = "backup"
    CUSTOM = "custom"


class ScheduleTypeEnum(str, Enum):
    """调度类型枚举"""
    ONCE = "once"
    CRON = "cron"
    INTERVAL = "interval"
    DEPENDENT = "dependent"


class ExecutionModeEnum(str, Enum):
    """执行模式枚举"""
    SYNC = "sync"
    ASYNC = "async"
    DISTRIBUTED = "distributed"


# ==================== SQLAlchemy 模型 ====================

if HAS_SQLALCHEMY:

    class ScheduledTaskModel(Base):
        """调度任务模型"""
        __tablename__ = 'scheduled_tasks'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 基本信息
        name = Column(String(256), nullable=False)
        description = Column(Text, nullable=True)
        task_type = Column(SQLEnum(TaskTypeEnum), nullable=False, default=TaskTypeEnum.TRAINING)
        
        # 调度配置
        schedule_type = Column(SQLEnum(ScheduleTypeEnum), nullable=False, default=ScheduleTypeEnum.ONCE)
        schedule_time = Column(DateTime, nullable=True, index=True)
        cron_expression = Column(String(128), nullable=True)
        interval_seconds = Column(Integer, nullable=True)
        
        # 状态信息
        status = Column(SQLEnum(TaskStatusEnum), nullable=False, default=TaskStatusEnum.PENDING)
        priority = Column(SQLEnum(TaskPriorityEnum), nullable=False, default=TaskPriorityEnum.NORMAL)
        execution_mode = Column(SQLEnum(ExecutionModeEnum), nullable=False, default=ExecutionModeEnum.ASYNC)
        
        # 任务配置
        config = Column(JSON, nullable=False, default=dict)
        template_id = Column(String(64), nullable=True)
        
        # 重试配置
        max_retries = Column(Integer, default=3)
        retry_count = Column(Integer, default=0)
        retry_delay_seconds = Column(Integer, default=60)
        
        # 超时配置
        timeout_seconds = Column(Integer, nullable=True)
        
        # 依赖配置
        depends_on = Column(JSON, nullable=True)  # 依赖的任务ID列表
        
        # 结果信息
        result = Column(JSON, nullable=True)
        error_message = Column(Text, nullable=True)
        error_stack = Column(Text, nullable=True)
        
        # 执行信息
        started_at = Column(DateTime, nullable=True)
        completed_at = Column(DateTime, nullable=True)
        execution_time_seconds = Column(Float, nullable=True)
        
        # 元数据
        tags = Column(JSON, nullable=True)
        task_metadata = Column('metadata', JSON, nullable=True)
        
        # 所有者信息
        created_by = Column(String(64), nullable=True)
        updated_by = Column(String(64), nullable=True)
        
        # 时间戳
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        next_run_time = Column(DateTime, nullable=True)
        last_run_time = Column(DateTime, nullable=True)
        
        # 索引
        __table_args__ = (
            Index('idx_task_status_priority', 'status', 'priority'),
            Index('idx_task_schedule_time', 'schedule_time'),
            Index('idx_task_next_run', 'next_run_time'),
            Index('idx_task_tenant_status', 'tenant_id', 'status'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            """转换为字典"""
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'name': self.name,
                'description': self.description,
                'task_type': self.task_type.value if self.task_type else None,
                'schedule_type': self.schedule_type.value if self.schedule_type else None,
                'schedule_time': self.schedule_time.isoformat() if self.schedule_time else None,
                'cron_expression': self.cron_expression,
                'interval_seconds': self.interval_seconds,
                'status': self.status.value if self.status else None,
                'priority': self.priority.value if self.priority else None,
                'execution_mode': self.execution_mode.value if self.execution_mode else None,
                'config': self.config,
                'template_id': self.template_id,
                'max_retries': self.max_retries,
                'retry_count': self.retry_count,
                'retry_delay_seconds': self.retry_delay_seconds,
                'timeout_seconds': self.timeout_seconds,
                'depends_on': self.depends_on,
                'result': self.result,
                'error_message': self.error_message,
                'started_at': self.started_at.isoformat() if self.started_at else None,
                'completed_at': self.completed_at.isoformat() if self.completed_at else None,
                'execution_time_seconds': self.execution_time_seconds,
                'tags': self.tags,
                'metadata': self.task_metadata,
                'created_by': self.created_by,
                'updated_by': self.updated_by,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None,
                'next_run_time': self.next_run_time.isoformat() if self.next_run_time else None,
                'last_run_time': self.last_run_time.isoformat() if self.last_run_time else None
            }


    class TaskExecutionLogModel(Base):
        """任务执行日志模型"""
        __tablename__ = 'task_execution_logs'
        
        id = Column(String(64), primary_key=True)
        task_id = Column(String(64), ForeignKey('scheduled_tasks.id'), nullable=False, index=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 执行信息
        execution_number = Column(Integer, default=1)
        status = Column(SQLEnum(TaskStatusEnum), nullable=False)
        
        # 时间信息
        started_at = Column(DateTime, nullable=False)
        completed_at = Column(DateTime, nullable=True)
        execution_time_seconds = Column(Float, nullable=True)
        
        # 结果信息
        result = Column(JSON, nullable=True)
        error_message = Column(Text, nullable=True)
        error_stack = Column(Text, nullable=True)
        
        # 资源使用
        resource_usage = Column(JSON, nullable=True)
        
        # 日志信息
        log_output = Column(Text, nullable=True)
        
        # 元数据
        task_metadata = Column('metadata', JSON, nullable=True)
        
        # 时间戳
        created_at = Column(DateTime, default=datetime.utcnow)
        
        # 索引
        __table_args__ = (
            Index('idx_exec_log_task', 'task_id', 'created_at'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            """转换为字典"""
            return {
                'id': self.id,
                'task_id': self.task_id,
                'tenant_id': self.tenant_id,
                'execution_number': self.execution_number,
                'status': self.status.value if self.status else None,
                'started_at': self.started_at.isoformat() if self.started_at else None,
                'completed_at': self.completed_at.isoformat() if self.completed_at else None,
                'execution_time_seconds': self.execution_time_seconds,
                'result': self.result,
                'error_message': self.error_message,
                'resource_usage': self.resource_usage,
                'log_output': self.log_output,
                'metadata': self.task_metadata,
                'created_at': self.created_at.isoformat() if self.created_at else None
            }


    class TaskTemplateModel(Base):
        """任务模板模型"""
        __tablename__ = 'task_templates'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 基本信息
        name = Column(String(256), nullable=False)
        description = Column(Text, nullable=True)
        category = Column(String(64), nullable=True)
        task_type = Column(SQLEnum(TaskTypeEnum), nullable=False, default=TaskTypeEnum.TRAINING)
        
        # 模板配置
        config_template = Column(JSON, nullable=False, default=dict)
        default_priority = Column(SQLEnum(TaskPriorityEnum), default=TaskPriorityEnum.NORMAL)
        default_timeout_seconds = Column(Integer, nullable=True)
        default_max_retries = Column(Integer, default=3)
        
        # 参数定义
        parameters = Column(JSON, nullable=True)  # 参数定义
        
        # 状态
        is_active = Column(Boolean, default=True)
        is_system = Column(Boolean, default=False)  # 是否系统内置
        
        # 使用统计
        usage_count = Column(Integer, default=0)
        
        # 元数据
        tags = Column(JSON, nullable=True)
        task_metadata = Column('metadata', JSON, nullable=True)
        
        # 所有者
        created_by = Column(String(64), nullable=True)
        updated_by = Column(String(64), nullable=True)
        
        # 时间戳
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        # 约束
        __table_args__ = (
            UniqueConstraint('tenant_id', 'name', name='uq_template_tenant_name'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            """转换为字典"""
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'name': self.name,
                'description': self.description,
                'category': self.category,
                'task_type': self.task_type.value if self.task_type else None,
                'config_template': self.config_template,
                'default_priority': self.default_priority.value if self.default_priority else None,
                'default_timeout_seconds': self.default_timeout_seconds,
                'default_max_retries': self.default_max_retries,
                'parameters': self.parameters,
                'is_active': self.is_active,
                'is_system': self.is_system,
                'usage_count': self.usage_count,
                'tags': self.tags,
                'metadata': self.task_metadata,
                'created_by': self.created_by,
                'updated_by': self.updated_by,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None
            }


    class TaskQueueModel(Base):
        """任务队列模型"""
        __tablename__ = 'task_queues'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 队列信息
        name = Column(String(128), nullable=False)
        description = Column(Text, nullable=True)
        
        # 配置
        max_concurrent = Column(Integer, default=5)
        max_queue_size = Column(Integer, default=1000)
        priority_weight = Column(Float, default=1.0)
        
        # 状态
        is_active = Column(Boolean, default=True)
        is_paused = Column(Boolean, default=False)
        
        # 统计
        pending_count = Column(Integer, default=0)
        running_count = Column(Integer, default=0)
        completed_count = Column(Integer, default=0)
        failed_count = Column(Integer, default=0)
        
        # 元数据
        task_metadata = Column('metadata', JSON, nullable=True)
        
        # 时间戳
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        
        def to_dict(self) -> Dict[str, Any]:
            """转换为字典"""
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'name': self.name,
                'description': self.description,
                'max_concurrent': self.max_concurrent,
                'max_queue_size': self.max_queue_size,
                'priority_weight': self.priority_weight,
                'is_active': self.is_active,
                'is_paused': self.is_paused,
                'pending_count': self.pending_count,
                'running_count': self.running_count,
                'completed_count': self.completed_count,
                'failed_count': self.failed_count,
                'metadata': self.task_metadata,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None
            }


    class SchedulerMetricsModel(Base):
        """调度器指标模型"""
        __tablename__ = 'scheduler_metrics'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 时间段
        period_start = Column(DateTime, nullable=False, index=True)
        period_end = Column(DateTime, nullable=False)
        period_type = Column(String(32), nullable=False)  # minute, hour, day
        
        # 任务统计
        tasks_scheduled = Column(Integer, default=0)
        tasks_executed = Column(Integer, default=0)
        tasks_completed = Column(Integer, default=0)
        tasks_failed = Column(Integer, default=0)
        tasks_cancelled = Column(Integer, default=0)
        tasks_timeout = Column(Integer, default=0)
        
        # 时间统计
        avg_wait_time_seconds = Column(Float, nullable=True)
        avg_execution_time_seconds = Column(Float, nullable=True)
        max_execution_time_seconds = Column(Float, nullable=True)
        min_execution_time_seconds = Column(Float, nullable=True)
        
        # 资源统计
        avg_cpu_usage = Column(Float, nullable=True)
        avg_memory_usage = Column(Float, nullable=True)
        peak_concurrent_tasks = Column(Integer, nullable=True)
        
        # 时间戳
        created_at = Column(DateTime, default=datetime.utcnow)
        
        # 索引
        __table_args__ = (
            Index('idx_metrics_period', 'period_start', 'period_type'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            """转换为字典"""
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'period_start': self.period_start.isoformat() if self.period_start else None,
                'period_end': self.period_end.isoformat() if self.period_end else None,
                'period_type': self.period_type,
                'tasks_scheduled': self.tasks_scheduled,
                'tasks_executed': self.tasks_executed,
                'tasks_completed': self.tasks_completed,
                'tasks_failed': self.tasks_failed,
                'tasks_cancelled': self.tasks_cancelled,
                'tasks_timeout': self.tasks_timeout,
                'avg_wait_time_seconds': self.avg_wait_time_seconds,
                'avg_execution_time_seconds': self.avg_execution_time_seconds,
                'max_execution_time_seconds': self.max_execution_time_seconds,
                'min_execution_time_seconds': self.min_execution_time_seconds,
                'avg_cpu_usage': self.avg_cpu_usage,
                'avg_memory_usage': self.avg_memory_usage,
                'peak_concurrent_tasks': self.peak_concurrent_tasks,
                'created_at': self.created_at.isoformat() if self.created_at else None
            }

else:
    # 无 SQLAlchemy 时的占位类
    class ScheduledTaskModel:
        pass
    
    class TaskExecutionLogModel:
        pass
    
    class TaskTemplateModel:
        pass
    
    class TaskQueueModel:
        pass
    
    class SchedulerMetricsModel:
        pass


# ==================== Pydantic 验证模型 ====================

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ScheduledTaskData:
    """调度任务数据类"""
    id: str
    name: str
    config: Dict[str, Any]
    schedule_time: Optional[datetime] = None
    status: str = TaskStatusEnum.PENDING.value
    priority: str = TaskPriorityEnum.NORMAL.value
    task_type: str = TaskTypeEnum.TRAINING.value
    schedule_type: str = ScheduleTypeEnum.ONCE.value
    execution_mode: str = ExecutionModeEnum.ASYNC.value
    description: str = ""
    tenant_id: str = None
    template_id: str = None
    cron_expression: str = None
    interval_seconds: int = None
    max_retries: int = 3
    retry_count: int = 0
    retry_delay_seconds: int = 60
    timeout_seconds: int = None
    depends_on: List[str] = field(default_factory=list)
    result: Dict[str, Any] = None
    error_message: str = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: str = None
    created_at: datetime = None
    updated_at: datetime = None
    started_at: datetime = None
    completed_at: datetime = None
    execution_time_seconds: float = None
    next_run_time: datetime = None
    last_run_time: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        # 转换 datetime 为 ISO 格式
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduledTaskData':
        """从字典创建"""
        data = data.copy()
        # 转换时间字段
        for field_name in ['schedule_time', 'created_at', 'updated_at', 'started_at', 
                          'completed_at', 'next_run_time', 'last_run_time']:
            if field_name in data and data[field_name] and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name].replace('Z', '+00:00'))
        
        # 过滤未知字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**data)


@dataclass
class TaskExecutionLogData:
    """任务执行日志数据类"""
    id: str
    task_id: str
    status: str
    started_at: datetime
    execution_number: int = 1
    tenant_id: str = None
    completed_at: datetime = None
    execution_time_seconds: float = None
    result: Dict[str, Any] = None
    error_message: str = None
    error_stack: str = None
    resource_usage: Dict[str, Any] = None
    log_output: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        return result


@dataclass 
class TaskTemplateData:
    """任务模板数据类"""
    id: str
    name: str
    config_template: Dict[str, Any]
    description: str = ""
    category: str = None
    task_type: str = TaskTypeEnum.TRAINING.value
    tenant_id: str = None
    default_priority: str = TaskPriorityEnum.NORMAL.value
    default_timeout_seconds: int = None
    default_max_retries: int = 3
    parameters: Dict[str, Any] = None
    is_active: bool = True
    is_system: bool = False
    usage_count: int = 0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_by: str = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        return result

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""性能监控数据模型

定义性能监控模块使用的数据类和数据库模型：
- 系统性能指标
- 训练性能指标
- 实时监控数据
- 告警记录
- 性能统计

将 monitoring 模块的功能整合到 performance 模块中使用。
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from sqlalchemy import (
    Column, String, Text, JSON, Boolean, Integer, Float, DateTime,
    Enum as SQLEnum, ForeignKey, Index
)

from backend.schemas.base_models import Base, UUIDMixin, TimestampMixin, TenantMixin, GUID


# ==================== 枚举类型 ====================

class MetricTypeEnum(str, Enum):
    """指标类型枚举"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    TRAINING = "training"
    SYSTEM = "system"


class ResourceStatusEnum(str, Enum):
    """资源状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertLevelEnum(str, Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatusEnum(str, Enum):
    """告警状态枚举"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class CollectionStatusEnum(str, Enum):
    """采集状态枚举"""
    RUNNING = "running"
    STOPPED = "stopped"
    PAUSED = "paused"
    ERROR = "error"


# ==================== SQLAlchemy 数据库模型 ====================

class SystemMetricRecordDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """系统性能指标记录数据库模型"""
    __tablename__ = 'system_metric_records'
    
    # 指标类型
    metric_type = Column(String(50), nullable=False, index=True, comment="指标类型")
    
    # CPU指标
    cpu_percent = Column(Float, comment="CPU使用率(%)")
    cpu_count = Column(Integer, comment="CPU核心数")
    cpu_frequency = Column(Float, comment="CPU频率(MHz)")
    cpu_temperature = Column(Float, comment="CPU温度(°C)")
    
    # 内存指标
    memory_total = Column(Float, comment="总内存(bytes)")
    memory_available = Column(Float, comment="可用内存(bytes)")
    memory_used = Column(Float, comment="已用内存(bytes)")
    memory_percent = Column(Float, comment="内存使用率(%)")
    
    # 磁盘指标
    disk_total = Column(Float, comment="磁盘总容量(bytes)")
    disk_used = Column(Float, comment="磁盘已用(bytes)")
    disk_free = Column(Float, comment="磁盘空闲(bytes)")
    disk_percent = Column(Float, comment="磁盘使用率(%)")
    disk_read_speed = Column(Float, comment="磁盘读取速度(bytes/s)")
    disk_write_speed = Column(Float, comment="磁盘写入速度(bytes/s)")
    
    # 网络指标
    network_bytes_sent = Column(Float, comment="发送字节数")
    network_bytes_recv = Column(Float, comment="接收字节数")
    network_packets_sent = Column(Integer, comment="发送包数")
    network_packets_recv = Column(Integer, comment="接收包数")
    network_download_speed = Column(Float, comment="下载速度(bytes/s)")
    network_upload_speed = Column(Float, comment="上传速度(bytes/s)")
    network_latency = Column(Float, comment="网络延迟(ms)")
    
    # 记录时间
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True, comment="记录时间")
    
    # 来源信息
    source = Column(String(100), comment="数据来源")
    host_name = Column(String(200), comment="主机名")
    
    __table_args__ = (
        Index('ix_sys_metric_type_time', 'metric_type', 'recorded_at'),
        Index('ix_sys_metric_tenant_time', 'tenant_id', 'recorded_at'),
        {'extend_existing': True},
    )
    
    def __repr__(self):
        return f"<SystemMetricRecordDB(type='{self.metric_type}', time='{self.recorded_at}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'metric_type': self.metric_type,
            'cpu': {
                'percent': self.cpu_percent,
                'count': self.cpu_count,
                'frequency': self.cpu_frequency,
                'temperature': self.cpu_temperature,
            },
            'memory': {
                'total': self.memory_total,
                'available': self.memory_available,
                'used': self.memory_used,
                'percent': self.memory_percent,
            },
            'disk': {
                'total': self.disk_total,
                'used': self.disk_used,
                'free': self.disk_free,
                'percent': self.disk_percent,
                'read_speed': self.disk_read_speed,
                'write_speed': self.disk_write_speed,
            },
            'network': {
                'bytes_sent': self.network_bytes_sent,
                'bytes_recv': self.network_bytes_recv,
                'packets_sent': self.network_packets_sent,
                'packets_recv': self.network_packets_recv,
                'download_speed': self.network_download_speed,
                'upload_speed': self.network_upload_speed,
                'latency': self.network_latency,
            },
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None,
            'source': self.source,
            'host_name': self.host_name,
            'tenant_id': self.tenant_id,
        }


class GPUMetricRecordDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """GPU性能指标记录数据库模型"""
    __tablename__ = 'gpu_metric_records'
    
    # GPU标识
    gpu_id = Column(String(50), nullable=False, index=True, comment="GPU ID")
    gpu_name = Column(String(200), comment="GPU名称")
    gpu_uuid = Column(String(100), comment="GPU UUID")
    
    # 利用率指标
    utilization = Column(Float, comment="GPU利用率(%)")
    memory_utilization = Column(Float, comment="显存利用率(%)")
    
    # 显存指标
    memory_total = Column(Float, comment="总显存(MB)")
    memory_used = Column(Float, comment="已用显存(MB)")
    memory_free = Column(Float, comment="空闲显存(MB)")
    
    # 温度和功耗
    temperature = Column(Float, comment="温度(°C)")
    power_draw = Column(Float, comment="功耗(W)")
    power_limit = Column(Float, comment="功耗限制(W)")
    
    # 时钟频率
    clock_core = Column(Float, comment="核心频率(MHz)")
    clock_memory = Column(Float, comment="显存频率(MHz)")
    
    # 进程信息
    processes_count = Column(Integer, comment="运行进程数")
    
    # 记录时间
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True, comment="记录时间")
    
    # 来源信息
    host_name = Column(String(200), comment="主机名")
    
    __table_args__ = (
        Index('ix_gpu_metric_gpu_time', 'gpu_id', 'recorded_at'),
    )
    
    def __repr__(self):
        return f"<GPUMetricRecordDB(gpu='{self.gpu_id}', util={self.utilization}%)>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'gpu_id': self.gpu_id,
            'gpu_name': self.gpu_name,
            'utilization': self.utilization,
            'memory': {
                'total': self.memory_total,
                'used': self.memory_used,
                'free': self.memory_free,
                'utilization': self.memory_utilization,
            },
            'temperature': self.temperature,
            'power_draw': self.power_draw,
            'power_limit': self.power_limit,
            'clock': {
                'core': self.clock_core,
                'memory': self.clock_memory,
            },
            'processes_count': self.processes_count,
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None,
            'host_name': self.host_name,
            'tenant_id': self.tenant_id,
        }


class TrainingMetricRecordDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """训练性能指标记录数据库模型"""
    __tablename__ = 'training_metric_records'
    
    # 会话关联
    session_id = Column(String(36), nullable=False, index=True, comment="训练会话ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 训练进度
    epoch = Column(Integer, comment="当前轮次")
    step = Column(Integer, comment="当前步数")
    total_steps = Column(Integer, comment="总步数")
    
    # 核心指标
    loss = Column(Float, comment="损失值")
    accuracy = Column(Float, comment="准确率")
    learning_rate = Column(Float, comment="学习率")
    gradient_norm = Column(Float, comment="梯度范数")
    
    # 训练速度
    samples_per_second = Column(Float, comment="每秒样本数")
    tokens_per_second = Column(Float, comment="每秒token数")
    batch_size = Column(Integer, comment="批次大小")
    
    # GPU资源使用
    gpu_utilization = Column(Float, comment="GPU利用率(%)")
    gpu_memory_used = Column(Float, comment="GPU显存使用(MB)")
    gpu_memory_total = Column(Float, comment="GPU显存总量(MB)")
    gpu_temperature = Column(Float, comment="GPU温度(°C)")
    gpu_power_draw = Column(Float, comment="GPU功耗(W)")
    
    # CPU资源使用
    cpu_utilization = Column(Float, comment="CPU利用率(%)")
    cpu_memory_used = Column(Float, comment="CPU内存使用(GB)")
    cpu_memory_total = Column(Float, comment="CPU内存总量(GB)")
    cpu_temperature = Column(Float, comment="CPU温度(°C)")
    
    # 磁盘IO
    disk_read_speed = Column(Float, comment="磁盘读取速度(MB/s)")
    disk_write_speed = Column(Float, comment="磁盘写入速度(MB/s)")
    disk_utilization = Column(Float, comment="磁盘利用率(%)")
    
    # 网络IO
    network_download_speed = Column(Float, comment="下载速度(MB/s)")
    network_upload_speed = Column(Float, comment="上传速度(MB/s)")
    network_latency = Column(Float, comment="网络延迟(ms)")
    
    # 记录时间
    recorded_at = Column(DateTime, default=datetime.utcnow, index=True, comment="记录时间")
    
    # 额外指标
    custom_metrics = Column(JSON, default=dict, comment="自定义指标")
    
    __table_args__ = (
        Index('ix_train_metric_session_time', 'session_id', 'recorded_at'),
        Index('ix_train_metric_user_time', 'user_id', 'recorded_at'),
    )
    
    def __repr__(self):
        return f"<TrainingMetricRecordDB(session='{self.session_id}', epoch={self.epoch}, step={self.step})>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'session_id': self.session_id,
            'user_id': self.user_id,
            'timestamp': self.recorded_at.isoformat() if self.recorded_at else None,
            'epoch': self.epoch,
            'step': self.step,
            'total_steps': self.total_steps,
            'loss': self.loss,
            'accuracy': self.accuracy,
            'learning_rate': self.learning_rate,
            'gradient_norm': self.gradient_norm,
            'training': {
                'samples_per_second': self.samples_per_second,
                'tokens_per_second': self.tokens_per_second,
                'batch_size': self.batch_size,
            },
            'gpu': {
                'utilization': self.gpu_utilization,
                'memory': {
                    'used': self.gpu_memory_used,
                    'total': self.gpu_memory_total,
                    'utilization': (self.gpu_memory_used / self.gpu_memory_total * 100) if self.gpu_memory_total else 0,
                },
                'temperature': self.gpu_temperature,
                'power_draw': self.gpu_power_draw,
            },
            'cpu': {
                'utilization': self.cpu_utilization,
                'memory': {
                    'used': self.cpu_memory_used,
                    'total': self.cpu_memory_total,
                    'utilization': (self.cpu_memory_used / self.cpu_memory_total * 100) if self.cpu_memory_total else 0,
                },
                'temperature': self.cpu_temperature,
            },
            'disk': {
                'read_speed': self.disk_read_speed,
                'write_speed': self.disk_write_speed,
                'utilization': self.disk_utilization,
            },
            'network': {
                'download_speed': self.network_download_speed,
                'upload_speed': self.network_upload_speed,
                'latency': self.network_latency,
            },
            'custom_metrics': self.custom_metrics,
            'tenant_id': self.tenant_id,
        }


class PerformanceAlertDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """性能告警记录数据库模型"""
    __tablename__ = 'performance_alerts'
    
    # 告警基本信息
    alert_type = Column(String(50), nullable=False, index=True, comment="告警类型")
    level = Column(String(20), nullable=False, index=True, comment="告警级别")
    status = Column(String(20), default='active', index=True, comment="告警状态")
    
    # 告警内容
    title = Column(String(500), nullable=False, comment="告警标题")
    message = Column(Text, comment="告警消息")
    source = Column(String(100), comment="告警来源")
    
    # 关联信息
    resource_type = Column(String(50), comment="资源类型")
    resource_id = Column(String(100), comment="资源ID")
    session_id = Column(String(36), index=True, comment="训练会话ID")
    
    # 指标信息
    metric_name = Column(String(100), comment="指标名称")
    metric_value = Column(Float, comment="指标值")
    threshold = Column(Float, comment="阈值")
    
    # 时间信息
    triggered_at = Column(DateTime, default=datetime.utcnow, index=True, comment="触发时间")
    acknowledged_at = Column(DateTime, comment="确认时间")
    resolved_at = Column(DateTime, comment="解决时间")
    
    # 操作信息
    acknowledged_by = Column(String(36), comment="确认人ID")
    resolved_by = Column(String(36), comment="解决人ID")
    resolution_note = Column(Text, comment="解决备注")
    
    # 额外数据
    extra_data = Column(JSON, default=dict, comment="元数据")
    
    __table_args__ = (
        Index('ix_perf_alert_status_time', 'status', 'triggered_at'),
        Index('ix_perf_alert_level_status', 'level', 'status'),
        {'extend_existing': True},
    )
    
    def __repr__(self):
        return f"<PerformanceAlertDB(type='{self.alert_type}', level='{self.level}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'alert_type': self.alert_type,
            'level': self.level,
            'status': self.status,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'session_id': self.session_id,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'acknowledged_by': self.acknowledged_by,
            'resolved_by': self.resolved_by,
            'resolution_note': self.resolution_note,
            'metadata': self.metadata,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


# ==================== 数据类 ====================

@dataclass
class SystemMetrics:
    """系统性能指标数据类"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 0
    cpu_frequency: float = 0.0
    cpu_temperature: Optional[float] = None
    
    # 内存
    memory_total: int = 0
    memory_available: int = 0
    memory_used: int = 0
    memory_percent: float = 0.0
    
    # 磁盘
    disk_total: int = 0
    disk_used: int = 0
    disk_free: int = 0
    disk_percent: float = 0.0
    
    # 网络
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    network_packets_sent: int = 0
    network_packets_recv: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'cpu': {
                'percent': self.cpu_percent,
                'count': self.cpu_count,
                'frequency': self.cpu_frequency,
                'temperature': self.cpu_temperature,
            },
            'memory': {
                'total': self.memory_total,
                'available': self.memory_available,
                'used': self.memory_used,
                'percent': self.memory_percent,
            },
            'disk': {
                'total': self.disk_total,
                'used': self.disk_used,
                'free': self.disk_free,
                'percent': self.disk_percent,
            },
            'network': {
                'bytes_sent': self.network_bytes_sent,
                'bytes_recv': self.network_bytes_recv,
                'packets_sent': self.network_packets_sent,
                'packets_recv': self.network_packets_recv,
            },
        }


@dataclass
class GPUMetrics:
    """GPU性能指标数据类"""
    gpu_id: str = "0"
    gpu_name: str = ""
    utilization: float = 0.0
    memory_total: float = 0.0
    memory_used: float = 0.0
    memory_utilization: float = 0.0
    temperature: float = 0.0
    power_draw: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.gpu_id,
            'name': self.gpu_name,
            'load': self.utilization,
            'utilization': self.utilization,
            'memory_util': self.memory_utilization,
            'memory_total': self.memory_total,
            'memory_used': self.memory_used,
            'temperature': self.temperature,
            'powerDraw': self.power_draw,
        }


@dataclass
class TrainingMetrics:
    """训练性能指标数据类"""
    session_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # 训练进度
    epoch: int = 0
    step: int = 0
    loss: float = 0.0
    accuracy: float = 0.0
    learning_rate: float = 0.0
    
    # 速度指标
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    batch_size: int = 0
    gradient_norm: float = 0.0
    
    # GPU指标
    gpu_utilization: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    gpu_temperature: float = 0.0
    gpu_power_draw: float = 0.0
    
    # CPU指标
    cpu_utilization: float = 0.0
    cpu_memory_used: float = 0.0
    cpu_memory_total: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        gpu_memory_util = (self.gpu_memory_used / self.gpu_memory_total * 100) if self.gpu_memory_total > 0 else 0
        cpu_memory_util = (self.cpu_memory_used / self.cpu_memory_total * 100) if self.cpu_memory_total > 0 else 0
        
        return {
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'gpu': {
                'utilization': self.gpu_utilization,
                'memory': {
                    'used': self.gpu_memory_used,
                    'total': self.gpu_memory_total,
                    'utilization': gpu_memory_util,
                },
                'temperature': self.gpu_temperature,
                'powerDraw': self.gpu_power_draw,
            },
            'cpu': {
                'utilization': self.cpu_utilization,
                'memory': {
                    'used': self.cpu_memory_used,
                    'total': self.cpu_memory_total,
                    'utilization': cpu_memory_util,
                },
            },
            'training': {
                'samplesPerSecond': self.samples_per_second,
                'tokensPerSecond': self.tokens_per_second,
                'batchSize': self.batch_size,
                'gradientNorm': self.gradient_norm,
                'learningRate': self.learning_rate,
            },
        }


@dataclass
class TrainingStatistics:
    """训练统计数据类"""
    session_id: str = ""
    avg_gpu_utilization: float = 0.0
    avg_memory_usage: float = 0.0
    max_gpu_memory: float = 0.0
    avg_training_speed: float = 0.0
    peak_temperature: float = 0.0
    total_samples_processed: int = 0
    total_power_consumption: float = 0.0
    uptime_seconds: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'avgGpuUtilization': self.avg_gpu_utilization,
            'avgGpuUsage': self.avg_gpu_utilization,
            'avgMemoryUsage': self.avg_memory_usage,
            'maxGpuMemory': self.max_gpu_memory,
            'avgTrainingSpeed': self.avg_training_speed,
            'peakTemperature': self.peak_temperature,
            'totalSamplesProcessed': self.total_samples_processed,
            'totalPowerConsumption': self.total_power_consumption,
            'uptime': self.uptime_seconds,
        }


@dataclass
class PerformanceAlert:
    """性能告警数据类"""
    alert_id: str = ""
    alert_type: str = ""
    level: str = "warning"
    status: str = "active"
    title: str = ""
    message: str = ""
    source: str = ""
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    triggered_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.alert_id,
            'type': self.alert_type,
            'level': self.level,
            'status': self.status,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
        }


# ==================== 工具函数 ====================

def assess_resource_status(value: float, warning_threshold: float, critical_threshold: float) -> str:
    """评估资源状态
    
    Args:
        value: 当前值
        warning_threshold: 警告阈值
        critical_threshold: 严重阈值
        
    Returns:
        状态字符串: healthy/warning/critical
    """
    if value >= critical_threshold:
        return ResourceStatusEnum.CRITICAL.value
    elif value >= warning_threshold:
        return ResourceStatusEnum.WARNING.value
    return ResourceStatusEnum.HEALTHY.value


def assess_temperature_status(temp: float) -> str:
    """评估温度状态
    
    Args:
        temp: 温度值(°C)
        
    Returns:
        状态字符串
    """
    if temp >= 85:
        return ResourceStatusEnum.CRITICAL.value
    elif temp >= 75:
        return ResourceStatusEnum.WARNING.value
    return ResourceStatusEnum.HEALTHY.value


def format_bytes(bytes_value: int) -> str:
    """格式化字节数为人类可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024
    return f"{bytes_value:.2f} PB"


def format_duration(seconds: int) -> str:
    """格式化时间持续时间"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    elif seconds < 86400:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours}h {minutes}m"
    else:
        days = seconds // 86400
        hours = (seconds % 86400) // 3600
        return f"{days}d {hours}h"


# ==================== 导出 ====================

__all__ = [
    # 枚举
    'MetricTypeEnum',
    'ResourceStatusEnum',
    'AlertLevelEnum',
    'AlertStatusEnum',
    'CollectionStatusEnum',
    
    # SQLAlchemy 模型
    'SystemMetricRecordDB',
    'GPUMetricRecordDB',
    'TrainingMetricRecordDB',
    'PerformanceAlertDB',
    
    # 数据类
    'SystemMetrics',
    'GPUMetrics',
    'TrainingMetrics',
    'TrainingStatistics',
    'PerformanceAlert',
    
    # 工具函数
    'assess_resource_status',
    'assess_temperature_status',
    'format_bytes',
    'format_duration',
]

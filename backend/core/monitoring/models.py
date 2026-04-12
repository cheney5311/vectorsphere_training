"""统一监控数据模型"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class MetricType(Enum):
    """指标类型枚举"""
    SYSTEM = "system"
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    TRAINING = "training"


class AlertLevel(Enum):
    """告警级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    rule_id: str
    name: str
    description: str
    level: AlertLevel
    timestamp: datetime
    metric_value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class AlertRule:
    """告警规则"""
    id: str
    name: str
    description: str
    metric_type: MetricType
    metric_name: str
    threshold: float
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    duration: int  # 持续时间(秒)
    severity: AlertLevel
    enabled: bool = True


@dataclass
class SystemMetrics:
    """系统性能指标"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    load_average: List[float]
    process_count: int


@dataclass
class GPUMetrics:
    """GPU性能指标"""
    timestamp: float
    gpu_id: int
    gpu_name: str
    gpu_utilization: float
    memory_utilization: float
    memory_used_mb: float
    memory_total_mb: float
    temperature: float
    power_draw: float = 0.0
    power_limit: float = 0.0
    fan_speed: float = 0.0
    clock_graphics: int = 0
    clock_memory: int = 0


@dataclass
class TrainingMetrics:
    """训练性能指标"""
    timestamp: float
    session_id: str
    epoch: int
    step: int
    loss: float
    accuracy: Optional[float]
    learning_rate: float
    batch_size: int
    samples_per_second: float
    gpu_memory_usage: float
    gradient_norm: Optional[float] = None
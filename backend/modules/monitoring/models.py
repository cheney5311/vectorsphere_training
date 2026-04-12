"""监控模块数据模型"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime


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
    power_draw: float
    power_limit: float
    fan_speed: float
    clock_graphics: int
    clock_memory: int


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
    gradient_norm: Optional[float]


@dataclass
class AlertRule:
    """告警规则"""
    id: str
    name: str
    description: str
    metric_type: str  # 'system', 'gpu', 'training'
    metric_name: str
    threshold: float
    operator: str  # '>', '<', '>=', '<=', '==', '!='
    duration: int  # 持续时间(秒)
    severity: str  # 'low', 'medium', 'high', 'critical'
    enabled: bool = True


@dataclass
class Alert:
    """告警信息"""
    id: str
    rule_id: str
    rule_name: str
    message: str
    severity: str
    timestamp: float
    metric_value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[float] = None


@dataclass
class AnomalyResult:
    """异常检测结果"""
    id: str
    timestamp: float
    metric_name: str
    value: float
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    suggested_action: str


@dataclass
class AnomalyPattern:
    """异常模式"""
    id: str
    pattern_id: str
    pattern_type: str  # 'spike', 'drift', 'oscillation', 'plateau', 'drop'
    metrics_involved: List[str]
    frequency: int
    severity: str
    description: str
    first_seen: float
    last_seen: float


@dataclass
class DashboardConfig:
    """仪表板配置"""
    id: str
    name: str
    description: str
    layout: Dict[str, Any]
    charts: List[Dict[str, Any]]
    refresh_interval: int  # 刷新间隔(秒)
    enabled: bool = True


@dataclass
class MetricData:
    """指标数据"""
    id: str
    timestamp: float
    metric_name: str
    value: float
    tags: Dict[str, str]
    metadata: Dict[str, Any]


# 导出所有模型类
__all__ = [
    'SystemMetrics',
    'GPUMetrics', 
    'TrainingMetrics',
    'AlertRule',
    'Alert',
    'AnomalyResult',
    'AnomalyPattern',
    'DashboardConfig',
    'MetricData'
]
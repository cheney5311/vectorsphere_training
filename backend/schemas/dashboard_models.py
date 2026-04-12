"""仪表盘数据模型

定义仪表盘相关的数据传输对象(DTO)和响应模型。
聚合训练、模型、系统资源等多维度数据的展示模型。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class DashboardTimeRange(Enum):
    """仪表盘时间范围"""
    LAST_HOUR = "last_hour"
    LAST_24_HOURS = "last_24_hours"
    LAST_7_DAYS = "last_7_days"
    LAST_30_DAYS = "last_30_days"
    THIS_MONTH = "this_month"
    THIS_YEAR = "this_year"
    CUSTOM = "custom"


class MetricGranularity(Enum):
    """指标粒度"""
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


# =============================================================================
# 训练统计模型
# =============================================================================

@dataclass
class TrainingOverview:
    """训练概览数据
    
    Attributes:
        active_count: 活跃训练任务数量
        completed_count: 已完成训练任务数量
        failed_count: 失败训练任务数量
        pending_count: 等待中任务数量
        paused_count: 暂停中任务数量
        success_rate: 成功率 (0.0 - 1.0)
        total_training_time_hours: 总训练时长(小时)
        avg_training_time_hours: 平均训练时长(小时)
    """
    active_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    pending_count: int = 0
    paused_count: int = 0
    success_rate: float = 0.0
    total_training_time_hours: float = 0.0
    avg_training_time_hours: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'active_count': self.active_count,
            'completed_count': self.completed_count,
            'failed_count': self.failed_count,
            'pending_count': self.pending_count,
            'paused_count': self.paused_count,
            'success_rate': round(self.success_rate, 4),
            'total_training_time_hours': round(self.total_training_time_hours, 2),
            'avg_training_time_hours': round(self.avg_training_time_hours, 2)
        }


@dataclass
class TrainingTrend:
    """训练趋势数据点
    
    Attributes:
        timestamp: 时间戳
        date_label: 日期标签 (如 "2026-01-16")
        count: 任务数量
        success_count: 成功数量
        failed_count: 失败数量
        avg_duration_hours: 平均时长
    """
    timestamp: datetime
    date_label: str
    count: int = 0
    success_count: int = 0
    failed_count: int = 0
    avg_duration_hours: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'date_label': self.date_label,
            'count': self.count,
            'success_count': self.success_count,
            'failed_count': self.failed_count,
            'avg_duration_hours': round(self.avg_duration_hours, 2)
        }


@dataclass
class TrainingDetailedStats:
    """详细训练统计
    
    Attributes:
        overview: 训练概览
        trends: 趋势数据列表
        by_type: 按类型分布
        by_status: 按状态分布
        top_models: 最常训练的模型
        recent_sessions: 最近的训练会话
    """
    overview: TrainingOverview = field(default_factory=TrainingOverview)
    trends: List[TrainingTrend] = field(default_factory=list)
    by_type: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)
    top_models: List[Dict[str, Any]] = field(default_factory=list)
    recent_sessions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overview': self.overview.to_dict(),
            'trends': [t.to_dict() for t in self.trends],
            'by_type': self.by_type,
            'by_status': self.by_status,
            'top_models': self.top_models,
            'recent_sessions': self.recent_sessions
        }


# =============================================================================
# 模型统计模型
# =============================================================================

@dataclass
class ModelOverview:
    """模型概览数据
    
    Attributes:
        total_count: 模型总数
        deployed_count: 已部署模型数
        draft_count: 草稿模型数
        archived_count: 已归档模型数
        avg_accuracy: 平均准确率
        best_accuracy: 最佳准确率
        avg_f1_score: 平均F1分数
        total_size_gb: 模型总大小(GB)
    """
    total_count: int = 0
    deployed_count: int = 0
    draft_count: int = 0
    archived_count: int = 0
    avg_accuracy: float = 0.0
    best_accuracy: float = 0.0
    avg_f1_score: float = 0.0
    total_size_gb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_count': self.total_count,
            'deployed_count': self.deployed_count,
            'draft_count': self.draft_count,
            'archived_count': self.archived_count,
            'avg_accuracy': round(self.avg_accuracy, 4),
            'best_accuracy': round(self.best_accuracy, 4),
            'avg_f1_score': round(self.avg_f1_score, 4),
            'total_size_gb': round(self.total_size_gb, 2)
        }


@dataclass
class ModelDistribution:
    """模型分布数据
    
    Attributes:
        by_type: 按类型分布
        by_framework: 按框架分布
        by_status: 按状态分布
        by_category: 按分类分布
    """
    by_type: Dict[str, int] = field(default_factory=dict)
    by_framework: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'by_type': self.by_type,
            'by_framework': self.by_framework,
            'by_status': self.by_status,
            'by_category': self.by_category
        }


# =============================================================================
# 系统资源模型
# =============================================================================

@dataclass
class SystemResourceSnapshot:
    """系统资源快照
    
    Attributes:
        timestamp: 时间戳
        cpu_usage: CPU使用率(%)
        cpu_count: CPU核心数
        memory_usage: 内存使用率(%)
        memory_used_gb: 已用内存(GB)
        memory_total_gb: 总内存(GB)
        disk_usage: 磁盘使用率(%)
        disk_used_gb: 已用磁盘(GB)
        disk_total_gb: 总磁盘(GB)
        network_sent_mb: 网络发送(MB)
        network_recv_mb: 网络接收(MB)
    """
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_usage: float = 0.0
    cpu_count: int = 1
    memory_usage: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    disk_usage: float = 0.0
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage': round(self.cpu_usage, 2),
            'cpu_count': self.cpu_count,
            'memory_usage': round(self.memory_usage, 2),
            'memory_used_gb': round(self.memory_used_gb, 2),
            'memory_total_gb': round(self.memory_total_gb, 2),
            'disk_usage': round(self.disk_usage, 2),
            'disk_used_gb': round(self.disk_used_gb, 2),
            'disk_total_gb': round(self.disk_total_gb, 2),
            'network_sent_mb': round(self.network_sent_mb, 2),
            'network_recv_mb': round(self.network_recv_mb, 2)
        }


@dataclass
class GPUResourceSnapshot:
    """GPU资源快照
    
    Attributes:
        device_id: 设备ID
        name: GPU名称
        utilization: 利用率(%)
        memory_used_gb: 已用显存(GB)
        memory_total_gb: 总显存(GB)
        memory_usage: 显存使用率(%)
        temperature: 温度(°C)
        power_draw: 功耗(W)
        power_limit: 功率上限(W)
    """
    device_id: int = 0
    name: str = "Unknown"
    utilization: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    memory_usage: float = 0.0
    temperature: float = 0.0
    power_draw: float = 0.0
    power_limit: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'device_id': self.device_id,
            'name': self.name,
            'utilization': round(self.utilization, 2),
            'memory_used_gb': round(self.memory_used_gb, 2),
            'memory_total_gb': round(self.memory_total_gb, 2),
            'memory_usage': round(self.memory_usage, 2),
            'temperature': round(self.temperature, 1),
            'power_draw': round(self.power_draw, 1),
            'power_limit': round(self.power_limit, 1)
        }


@dataclass
class SystemMetricsHistory:
    """系统指标历史数据
    
    Attributes:
        cpu_usage: CPU使用率历史列表
        memory_usage: 内存使用率历史列表
        disk_usage: 磁盘使用率历史列表
        network_io: 网络IO历史列表
        gpu_usage: GPU使用率历史列表(如果有GPU)
    """
    cpu_usage: List[Dict[str, Any]] = field(default_factory=list)
    memory_usage: List[Dict[str, Any]] = field(default_factory=list)
    disk_usage: List[Dict[str, Any]] = field(default_factory=list)
    network_io: List[Dict[str, Any]] = field(default_factory=list)
    gpu_usage: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'disk_usage': self.disk_usage,
            'network_io': self.network_io,
            'gpu_usage': self.gpu_usage
        }


# =============================================================================
# 用户活动模型
# =============================================================================

@dataclass
class UserActivitySummary:
    """用户活动概要
    
    Attributes:
        total_training_count: 总训练次数
        total_models_created: 创建的模型数
        total_datasets_used: 使用的数据集数
        last_active_at: 最后活跃时间
        most_used_model_type: 最常用模型类型
        avg_training_time_hours: 平均训练时长
    """
    total_training_count: int = 0
    total_models_created: int = 0
    total_datasets_used: int = 0
    last_active_at: Optional[datetime] = None
    most_used_model_type: str = ""
    avg_training_time_hours: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_training_count': self.total_training_count,
            'total_models_created': self.total_models_created,
            'total_datasets_used': self.total_datasets_used,
            'last_active_at': self.last_active_at.isoformat() if self.last_active_at else None,
            'most_used_model_type': self.most_used_model_type,
            'avg_training_time_hours': round(self.avg_training_time_hours, 2)
        }


# =============================================================================
# 仪表盘聚合模型
# =============================================================================

@dataclass
class DashboardOverview:
    """仪表盘概览数据（聚合）
    
    Attributes:
        training: 训练概览
        models: 模型概览
        system: 系统资源快照
        gpu: GPU资源列表
        user_activity: 用户活动概要
        alerts_count: 告警数量
        last_updated: 最后更新时间
    """
    training: TrainingOverview = field(default_factory=TrainingOverview)
    models: ModelOverview = field(default_factory=ModelOverview)
    system: SystemResourceSnapshot = field(default_factory=SystemResourceSnapshot)
    gpu: List[GPUResourceSnapshot] = field(default_factory=list)
    user_activity: Optional[UserActivitySummary] = None
    alerts_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'training': self.training.to_dict(),
            'models': self.models.to_dict(),
            'system': self.system.to_dict(),
            'gpu': [g.to_dict() for g in self.gpu],
            'user_activity': self.user_activity.to_dict() if self.user_activity else None,
            'alerts_count': self.alerts_count,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class DashboardFilter:
    """仪表盘过滤条件
    
    Attributes:
        user_id: 用户ID（可选，按用户筛选）
        tenant_id: 租户ID（可选，按租户筛选）
        time_range: 时间范围
        start_date: 自定义开始时间
        end_date: 自定义结束时间
        granularity: 数据粒度
    """
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    time_range: DashboardTimeRange = DashboardTimeRange.LAST_24_HOURS
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    granularity: MetricGranularity = MetricGranularity.HOUR
    
    def get_date_range(self) -> tuple:
        """计算日期范围
        
        Returns:
            (start_date, end_date) 元组
        """
        from datetime import timedelta
        
        end_date = datetime.utcnow()
        
        if self.time_range == DashboardTimeRange.CUSTOM:
            return self.start_date or end_date, self.end_date or end_date
        
        range_mapping = {
            DashboardTimeRange.LAST_HOUR: timedelta(hours=1),
            DashboardTimeRange.LAST_24_HOURS: timedelta(hours=24),
            DashboardTimeRange.LAST_7_DAYS: timedelta(days=7),
            DashboardTimeRange.LAST_30_DAYS: timedelta(days=30),
        }
        
        if self.time_range == DashboardTimeRange.THIS_MONTH:
            start_date = end_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif self.time_range == DashboardTimeRange.THIS_YEAR:
            start_date = end_date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            delta = range_mapping.get(self.time_range, timedelta(hours=24))
            start_date = end_date - delta
        
        return start_date, end_date


# =============================================================================
# API 请求/响应模型
# =============================================================================

@dataclass
class DashboardOverviewResponse:
    """仪表盘概览响应"""
    success: bool = True
    message: str = ""
    data: Optional[DashboardOverview] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'message': self.message,
            'data': self.data.to_dict() if self.data else None
        }


@dataclass
class TrainingStatsResponse:
    """训练统计响应"""
    success: bool = True
    message: str = ""
    data: Optional[TrainingDetailedStats] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'message': self.message,
            'data': self.data.to_dict() if self.data else None
        }


@dataclass  
class SystemMetricsResponse:
    """系统指标响应"""
    success: bool = True
    message: str = ""
    data: Optional[SystemMetricsHistory] = None
    current: Optional[SystemResourceSnapshot] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'message': self.message,
            'data': self.data.to_dict() if self.data else None,
            'current': self.current.to_dict() if self.current else None
        }

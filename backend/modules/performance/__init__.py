"""性能优化模块

提供异步任务处理、数据库连接池管理和性能监控等核心功能。
现在使用统一的监控核心服务。
"""

from backend.core.monitoring.service import (
    UnifiedMonitoringService,
    get_monitoring_service as get_core_monitoring_service
)
from backend.core.monitoring.models import (
    SystemMetrics, GPUMetrics, TrainingMetrics,
    AlertRule, Alert, MetricType, AlertLevel
)
from backend.core.monitoring.exceptions import (
    MonitoringError, MetricsCollectionError, 
    AlertProcessingError, ResourceUnavailableError
)

# 为了向后兼容，重新导出核心服务的类和函数
PerformanceMonitor = UnifiedMonitoringService

get_performance_monitor = get_core_monitoring_service

__all__ = [
    # 核心服务类
    'PerformanceMonitor',
    
    # 数据模型
    'SystemMetrics',
    'GPUMetrics',
    'TrainingMetrics',
    'AlertRule',
    'Alert',
    'MetricType',
    'AlertLevel',
    
    # 异常类
    'MonitoringError',
    'MetricsCollectionError',
    'AlertProcessingError',
    'ResourceUnavailableError',
    
    # 服务获取函数
    'get_performance_monitor',
]
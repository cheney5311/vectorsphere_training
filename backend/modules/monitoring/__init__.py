"""监控模块

提供系统性能监控、指标收集、告警管理等功能。
现在使用统一的监控核心服务。
"""

from backend.core.monitoring.service import (
    UnifiedMonitoringService,
    get_monitoring_service as get_core_monitoring_service,
    start_global_monitoring as start_core_monitoring,
    stop_global_monitoring as stop_core_monitoring
)
from backend.core.monitoring.analyzer import (
    PerformanceAnalyzer,
    get_performance_analyzer as get_core_analyzer
)
from backend.core.monitoring.optimizer import (
    ResourceOptimizer,
    get_resource_optimizer as get_core_optimizer
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
PerformanceAnalysisEngine = PerformanceAnalyzer
ResourceOptimizerEngine = ResourceOptimizer

get_monitoring_service = get_core_monitoring_service
get_performance_analyzer = get_core_analyzer
get_resource_optimizer = get_core_optimizer

start_global_monitoring = start_core_monitoring
stop_global_monitoring = stop_core_monitoring

from .metrics_exporter import metrics_bp, record_training_metrics
from .training_monitor import get_training_monitor, create_progress_tracker

__all__ = [
    # 核心服务类
    # 核心服务类
    'PerformanceMonitor',
    'PerformanceAnalysisEngine',
    'ResourceOptimizerEngine',
    
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
    'get_monitoring_service',
    'get_performance_analyzer',
    'get_resource_optimizer',
    
    # 控制函数
    'start_global_monitoring',
    'stop_global_monitoring',
]
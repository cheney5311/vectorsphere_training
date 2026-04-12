"""性能监控器 - 已重构为使用统一核心服务

此文件已重构为使用统一的监控核心服务，避免重复代码。
"""

from backend.core.monitoring.service import (
    UnifiedMonitoringService,
    get_monitoring_service as get_core_monitoring_service
)

# 为了向后兼容，重新导出核心服务的类和函数
PerformanceMonitor = UnifiedMonitoringService
get_performance_monitor = get_core_monitoring_service

# 创建一个别名以保持向后兼容
create_performance_monitor = lambda collection_interval=10.0, max_history=1000: PerformanceMonitor()

__all__ = [
    'PerformanceMonitor',
    'get_performance_monitor',
    'create_performance_monitor'
]

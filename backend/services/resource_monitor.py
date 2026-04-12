"""资源监控器 - 已重构为使用统一核心服务

此文件已重构为使用统一的监控核心服务，避免重复代码。
"""

from backend.core.monitoring.service import (
    UnifiedMonitoringService,
    get_monitoring_service as get_core_monitoring_service
)

# 为了向后兼容，重新导出核心服务的类和函数
ResourceMonitor = UnifiedMonitoringService
get_resource_monitor = get_core_monitoring_service

# 创建一个别名以保持向后兼容
create_resource_monitor = lambda config=None: ResourceMonitor(config)

__all__ = [
    'ResourceMonitor',
    'get_resource_monitor',
    'create_resource_monitor'
]

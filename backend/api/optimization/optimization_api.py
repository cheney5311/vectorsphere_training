"""优化模块API接口 - 已重构为使用统一核心服务

此文件已重构为使用统一的监控API核心服务，避免重复代码。
"""

from backend.core.monitoring.api import (
    monitoring_bp,
    init_monitoring_api as init_core_monitoring_api,
    cleanup_monitoring_api as cleanup_core_monitoring_api
)

# 为了向后兼容，重新导出核心服务的函数
optimization_bp = monitoring_bp
init_optimization_api = init_core_monitoring_api
cleanup_optimization_api = cleanup_core_monitoring_api

__all__ = [
    'optimization_bp',
    'init_optimization_api',
    'cleanup_optimization_api'
]

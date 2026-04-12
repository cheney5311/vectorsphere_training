"""性能模块API接口

提供异步任务处理、数据库连接池管理、性能监控等REST API接口。
"""

from .performance_api import (
    performance_bp,
    init_performance_api,
    cleanup_performance_api,
    get_performance_api_status,
    task_registry
)

from .utils import (
    get_config,
    APIResponse,
    HealthCheckResult,
    timestamp_to_iso,
    format_duration,
    parse_time_range,
    get_resource_status,
    get_temperature_status,
    handle_api_errors
)

__all__ = [
    'performance_bp',
    'init_performance_api',
    'cleanup_performance_api',
    'get_performance_api_status',
    'task_registry',
    'get_config',
    'APIResponse',
    'HealthCheckResult',
    'timestamp_to_iso',
    'format_duration',
    'parse_time_range',
    'get_resource_status',
    'get_temperature_status',
    'handle_api_errors'
]
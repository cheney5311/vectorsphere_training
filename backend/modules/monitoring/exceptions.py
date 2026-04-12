"""监控模块异常类"""

class MonitoringError(Exception):
    """监控模块基础异常类"""
    pass


class MetricsCollectionError(MonitoringError):
    """指标收集异常"""
    pass


class AlertProcessingError(MonitoringError):
    """告警处理异常"""
    pass


class AnomalyDetectionError(MonitoringError):
    """异常检测异常"""
    pass


class DashboardError(MonitoringError):
    """仪表板异常"""
    pass


class ConfigurationError(MonitoringError):
    """配置异常"""
    pass


class ResourceUnavailableError(MonitoringError):
    """资源不可用异常"""
    pass


# 导出所有异常类
__all__ = [
    'MonitoringError',
    'MetricsCollectionError',
    'AlertProcessingError',
    'AnomalyDetectionError',
    'DashboardError',
    'ConfigurationError',
    'ResourceUnavailableError'
]
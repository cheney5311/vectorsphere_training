"""统一监控异常定义"""

class MonitoringError(Exception):
    """监控基础异常"""
    pass


class MetricsCollectionError(MonitoringError):
    """指标收集异常"""
    pass


class AlertProcessingError(MonitoringError):
    """告警处理异常"""
    pass


class ResourceUnavailableError(MonitoringError):
    """资源不可用异常"""
    pass
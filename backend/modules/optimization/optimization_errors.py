"""优化模块异常类

定义资源优化模块的异常类型。
"""

from typing import Optional


class OptimizationError(Exception):
    """优化模块基础异常类"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        """初始化异常

        Args:
            message: 异常消息
            error_code: 错误代码
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ResourceMonitorError(OptimizationError):
    """资源监控异常"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "RESOURCE_MONITOR_ERROR")


class ResourceOptimizerError(OptimizationError):
    """资源优化器异常"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "RESOURCE_OPTIMIZER_ERROR")


class PerformanceAnalyzerError(OptimizationError):
    """性能分析器异常"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "PERFORMANCE_ANALYZER_ERROR")


class PredictionError(OptimizationError):
    """预测异常"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "PREDICTION_ERROR")


class ConfigurationError(OptimizationError):
    """配置异常"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "CONFIGURATION_ERROR")


class ResourceUnavailableError(OptimizationError):
    """资源不可用异常"""

    def __init__(self, message: str, resource_type: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "RESOURCE_UNAVAILABLE_ERROR")
        self.resource_type = resource_type


class InvalidRecommendationError(OptimizationError):
    """无效优化建议异常"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "INVALID_RECOMMENDATION_ERROR")
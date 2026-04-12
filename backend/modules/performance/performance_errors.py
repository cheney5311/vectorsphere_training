"""性能模块异常类

定义性能优化模块的异常类型。
"""

from typing import Optional


class PerformanceError(Exception):
    """性能模块基础异常类"""

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


class AsyncProcessorError(PerformanceError):
    """异步处理器异常"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "ASYNC_PROCESSOR_ERROR")


class DatabasePoolError(PerformanceError):
    """数据库连接池异常"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "DATABASE_POOL_ERROR")


class PerformanceMonitorError(PerformanceError):
    """性能监控器异常"""

    def __init__(self, message: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "PERFORMANCE_MONITOR_ERROR")


class ResourceLimitExceededError(PerformanceError):
    """资源限制超出异常"""

    def __init__(self, message: str, resource_type: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "RESOURCE_LIMIT_EXCEEDED_ERROR")
        self.resource_type = resource_type


class TaskExecutionError(PerformanceError):
    """任务执行异常"""

    def __init__(self, message: str, task_id: str, error_code: Optional[str] = None):
        super().__init__(message, error_code or "TASK_EXECUTION_ERROR")
        self.task_id = task_id
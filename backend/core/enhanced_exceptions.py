"""增强的异常处理模块

提供更强大、更灵活的异常处理机制，包括重试机制、回退策略和详细的错误追踪。
"""

import logging
import time
import traceback
from typing import Optional, Dict, Any, Callable, Type, List, Union
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from contextlib import contextmanager

# 配置日志记录器
logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """恢复策略枚举"""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    FAIL_FAST = "fail_fast"


@dataclass
class ErrorContext:
    """错误上下文信息"""
    component: str = ""
    operation: str = ""
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorInfo:
    """错误信息"""
    error_type: str
    error_code: str
    message: str
    details: Dict[str, Any]
    context: ErrorContext
    severity: ErrorSeverity
    stack_trace: str = ""
    recovery_attempts: int = 0
    recovery_strategy: Optional[RecoveryStrategy] = None


class EnhancedBaseError(Exception):
    """增强的基础异常类"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None,
        context: Optional[ErrorContext] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: Optional[RecoveryStrategy] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.context = context or ErrorContext()
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.stack_trace = traceback.format_exc()
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'context': {
                'component': self.context.component,
                'operation': self.context.operation,
                'user_id': self.context.user_id,
                'tenant_id': self.context.tenant_id,
                'request_id': self.context.request_id,
                'timestamp': self.context.timestamp,
                'additional_data': self.context.additional_data
            },
            'severity': self.severity.value,
            'stack_trace': self.stack_trace,
            'recovery_strategy': self.recovery_strategy.value if self.recovery_strategy else None
        }
        
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.error_code}: {self.message}"
        
    def log(self, logger_instance: logging.Logger = None):
        """记录错误日志"""
        if logger_instance is None:
            logger_instance = logger
            
        log_method = logger_instance.error
        if self.severity == ErrorSeverity.LOW:
            log_method = logger_instance.warning
        elif self.severity == ErrorSeverity.CRITICAL:
            log_method = logger_instance.critical
            
        log_method(
            f"Error [{self.error_code}]: {self.message}",
            extra={
                'error_info': self.to_dict()
            }
        )


# 增强的异常类
class EnhancedValidationError(EnhancedBaseError):
    """增强的验证错误"""
    
    def __init__(
        self, 
        message: str = "验证失败", 
        field: Optional[str] = None, 
        value: Optional[Any] = None,
        context: Optional[ErrorContext] = None
    ):
        details = {}
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
            
        super().__init__(
            message, 
            'VALIDATION_ERROR', 
            details, 
            context, 
            ErrorSeverity.MEDIUM
        )


class EnhancedAuthenticationError(EnhancedBaseError):
    """增强的认证错误"""
    
    def __init__(
        self, 
        message: str = "认证失败", 
        reason: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        details = {}
        if reason:
            details['reason'] = reason
            
        super().__init__(
            message, 
            'AUTHENTICATION_ERROR', 
            details, 
            context, 
            ErrorSeverity.HIGH
        )


class EnhancedAuthorizationError(EnhancedBaseError):
    """增强的授权错误"""
    
    def __init__(
        self, 
        message: str = "权限不足", 
        required_permission: Optional[str] = None, 
        user_role: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        details = {}
        if required_permission:
            details['required_permission'] = required_permission
        if user_role:
            details['user_role'] = user_role
            
        super().__init__(
            message, 
            'AUTHORIZATION_ERROR', 
            details, 
            context, 
            ErrorSeverity.HIGH
        )


class EnhancedResourceNotFoundError(EnhancedBaseError):
    """增强的资源未找到错误"""
    
    def __init__(
        self, 
        message: str = "资源未找到", 
        resource_type: Optional[str] = None, 
        resource_id: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        details = {}
        if resource_type:
            details['resource_type'] = resource_type
        if resource_id:
            details['resource_id'] = resource_id
            
        super().__init__(
            message, 
            'RESOURCE_NOT_FOUND', 
            details, 
            context, 
            ErrorSeverity.MEDIUM
        )


class EnhancedSystemError(EnhancedBaseError):
    """增强的系统错误"""
    
    def __init__(
        self, 
        message: str = "系统错误", 
        component: Optional[str] = None, 
        error_code: Optional[str] = None,
        context: Optional[ErrorContext] = None
    ):
        details = {}
        if component:
            details['component'] = component
        if error_code:
            details['error_code'] = error_code
            
        super().__init__(
            message, 
            'SYSTEM_ERROR', 
            details, 
            context, 
            ErrorSeverity.HIGH
        )


class EnhancedTrainingError(EnhancedBaseError):
    """增强的训练相关错误"""
    
    def __init__(
        self, 
        message: str = "训练错误", 
        job_id: Optional[str] = None, 
        stage: Optional[str] = None,
        context: Optional[ErrorContext] = None,
        can_resume: bool = False
    ):
        details = {}
        if job_id:
            details['job_id'] = job_id
        if stage:
            details['stage'] = stage
        details['can_resume'] = can_resume
            
        super().__init__(
            message, 
            'TRAINING_ERROR', 
            details, 
            context, 
            ErrorSeverity.MEDIUM
        )


# 异常处理器
class ExceptionHandler:
    """异常处理器"""
    
    def __init__(self):
        self.error_handlers: Dict[Type[Exception], Callable] = {}
        self.fallback_handlers: Dict[str, Callable] = {}
        self.error_loggers: List[Callable] = []
        
    def register_handler(self, exception_type: Type[Exception], handler: Callable):
        """注册异常处理器"""
        self.error_handlers[exception_type] = handler
        
    def register_fallback(self, operation: str, fallback: Callable):
        """注册回退处理器"""
        self.fallback_handlers[operation] = fallback
        
    def add_error_logger(self, logger_func: Callable):
        """添加错误记录器"""
        self.error_loggers.append(logger_func)
        
    def handle_exception(self, e: Exception, context: Optional[ErrorContext] = None) -> EnhancedBaseError:
        """处理异常"""
        # 转换为增强异常
        enhanced_error = self._convert_to_enhanced_error(e, context)
        
        # 记录错误
        self._log_error(enhanced_error)
        
        # 调用特定处理器
        if type(e) in self.error_handlers:
            try:
                self.error_handlers[type(e)](enhanced_error)
            except Exception as handler_error:
                logger.error(f"异常处理器执行失败: {handler_error}")
                
        return enhanced_error
        
    def _convert_to_enhanced_error(self, e: Exception, context: Optional[ErrorContext] = None) -> EnhancedBaseError:
        """转换为增强异常"""
        if isinstance(e, EnhancedBaseError):
            if context:
                e.context = context
            return e
            
        # 根据原始异常类型转换
        if isinstance(e, ValueError):
            return EnhancedValidationError(str(e), context=context)
        elif isinstance(e, PermissionError):
            return EnhancedAuthorizationError(str(e), context=context)
        elif isinstance(e, FileNotFoundError):
            return EnhancedResourceNotFoundError(str(e), resource_type='file', context=context)
        elif isinstance(e, ConnectionError):
            return EnhancedSystemError(f"连接错误: {e}", component='network', context=context)
        else:
            return EnhancedSystemError(f"未知错误: {e}", error_code=type(e).__name__, context=context)
            
    def _log_error(self, error: EnhancedBaseError):
        """记录错误"""
        # 使用内置日志记录
        error.log()
        
        # 调用自定义错误记录器
        for logger_func in self.error_loggers:
            try:
                logger_func(error)
            except Exception as logger_error:
                logger.error(f"错误记录器执行失败: {logger_error}")


# 全局异常处理器实例
_global_exception_handler = ExceptionHandler()


def get_exception_handler() -> ExceptionHandler:
    """获取全局异常处理器实例"""
    return _global_exception_handler


# 装饰器和工具函数
def handle_exceptions(
    exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.FAIL_FAST,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    fallback_func: Optional[Callable] = None,
    context: Optional[ErrorContext] = None
):
    """异常处理装饰器
    
    Args:
        exceptions: 要处理的异常类型
        recovery_strategy: 恢复策略
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        fallback_func: 回退函数
        context: 错误上下文
    """
    if not isinstance(exceptions, list):
        exceptions = [exceptions]
    # Normalize exceptions to a tuple for except clause
    if isinstance(exceptions, list):
        exceptions_to_catch = tuple(exceptions)
    elif isinstance(exceptions, tuple):
        exceptions_to_catch = exceptions
    else:
        # Single exception class
        exceptions_to_catch = (exceptions,)
        
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions_to_catch as e:  # pylint: disable=catching-non-exception
                    last_exception = e
                    
                    # 记录重试信息
                    if attempt < max_retries:
                        logger.warning(f"函数 {func.__name__} 执行失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                        time.sleep(retry_delay * (2 ** attempt))  # 指数退避
                        continue
                        
                    # 最后一次尝试失败，处理异常
                    error_context = context or ErrorContext(
                        component=func.__module__,
                        operation=func.__name__
                    )
                    
                    enhanced_error = get_exception_handler().handle_exception(e, error_context)
                    enhanced_error.recovery_attempts = attempt
                    
                    # 根据恢复策略处理
                    if recovery_strategy == RecoveryStrategy.FAIL_FAST:
                        raise enhanced_error
                    elif recovery_strategy == RecoveryStrategy.SKIP:
                        logger.warning(f"跳过操作 {func.__name__} 由于错误: {enhanced_error}")
                        return None
                    elif recovery_strategy == RecoveryStrategy.FALLBACK and fallback_func:
                        logger.warning(f"使用回退函数处理 {func.__name__} 的错误: {enhanced_error}")
                        return fallback_func(*args, **kwargs)
                    else:
                        raise enhanced_error
                        
            # 如果所有重试都失败
            if last_exception:
                raise get_exception_handler().handle_exception(last_exception, context)
                
        return wrapper
    return decorator


@contextmanager
def error_context(component: str = "", operation: str = "", **kwargs):
    """错误上下文管理器"""
    context = ErrorContext(
        component=component,
        operation=operation,
        **kwargs
    )
    
    try:
        yield context
    except Exception as e:
        # 在上下文中处理异常
        enhanced_error = get_exception_handler().handle_exception(e, context)
        raise enhanced_error


def safe_execute(func: Callable, *args, fallback=None, **kwargs):
    """安全执行函数
    
    Args:
        func: 要执行的函数
        fallback: 回退值或函数
        *args: 函数参数
        **kwargs: 函数关键字参数
        
    Returns:
        函数执行结果或回退值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        context = ErrorContext(
            component=func.__module__ if hasattr(func, '__module__') else 'unknown',
            operation=func.__name__ if hasattr(func, '__name__') else 'unknown'
        )
        
        error = get_exception_handler().handle_exception(e, context)
        
        if callable(fallback):
            return fallback(error)
        else:
            return fallback


# 特定领域的异常处理
class TrainingExceptionHandler:
    """训练异常处理器"""
    
    def __init__(self):
        self.exception_handler = get_exception_handler()
        
    def handle_training_error(self, error: EnhancedTrainingError) -> bool:
        """处理训练错误
        
        Args:
            error: 训练错误
            
        Returns:
            bool: 是否可以恢复
        """
        logger.error(f"训练错误: {error}")
        
        # 根据错误类型决定处理策略
        if 'can_resume' in error.details and error.details['can_resume']:
            logger.info(f"训练任务 {error.details.get('job_id')} 可以恢复")
            return True
        else:
            logger.error(f"训练任务 {error.details.get('job_id')} 无法恢复")
            return False
            
    def register_training_handlers(self):
        """注册训练相关处理器"""
        self.exception_handler.register_handler(EnhancedTrainingError, self.handle_training_error)


# 初始化默认处理器
def initialize_exception_handlers():
    """初始化默认异常处理器"""
    handler = get_exception_handler()
    
    # 注册训练异常处理器
    training_handler = TrainingExceptionHandler()
    training_handler.register_training_handlers()


# 在模块导入时初始化
initialize_exception_handlers()
"""统一异常处理模块

提供平台统一的异常处理机制。
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BaseError(Exception):
    """基础异常类"""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        self.severity = severity
        super().__init__(self.message)
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'severity': self.severity.value
        }
        
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.error_code}: {self.message}"


class ValidationError(BaseError):
    """验证错误"""
    
    def __init__(self, message: str = "验证失败", field: Optional[str] = None, value: Optional[Any] = None):
        details = {}
        if field:
            details['field'] = field
        if value is not None:
            details['value'] = str(value)
            
        super().__init__(message, 'VALIDATION_ERROR', details, ErrorSeverity.MEDIUM)


class AuthenticationError(BaseError):
    """认证错误"""
    
    def __init__(self, message: str = "认证失败", reason: Optional[str] = None):
        details = {}
        if reason:
            details['reason'] = reason
            
        super().__init__(message, 'AUTHENTICATION_ERROR', details, ErrorSeverity.HIGH)


class AuthorizationError(BaseError):
    """授权错误"""
    
    def __init__(self, message: str = "权限不足", required_permission: Optional[str] = None):
        details = {}
        if required_permission:
            details['required_permission'] = required_permission
            
        super().__init__(message, 'AUTHORIZATION_ERROR', details, ErrorSeverity.HIGH)


class ResourceNotFoundError(BaseError):
    """资源未找到错误"""
    
    def __init__(self, message: str = "资源未找到", resource_type: Optional[str] = None, resource_id: Optional[str] = None):
        details = {}
        if resource_type:
            details['resource_type'] = resource_type
        if resource_id:
            details['resource_id'] = resource_id
            
        super().__init__(message, 'RESOURCE_NOT_FOUND', details, ErrorSeverity.MEDIUM)


class NotFoundError(ResourceNotFoundError):
    """未找到错误（ResourceNotFoundError的别名）"""
    pass


class BusinessLogicError(BaseError):
    """业务逻辑错误"""
    
    def __init__(self, message: str = "业务逻辑错误", operation: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        details = {}
        if operation:
            details['operation'] = operation
        if context:
            details['context'] = context
            
        super().__init__(message, 'BUSINESS_LOGIC_ERROR', details, ErrorSeverity.HIGH)


class SystemError(BaseError):
    """系统错误"""
    
    def __init__(self, message: str = "系统错误", component: Optional[str] = None, error_code: Optional[str] = None):
        details = {}
        if component:
            details['component'] = component
        if error_code:
            details['error_code'] = error_code
            
        super().__init__(message, 'SYSTEM_ERROR', details, ErrorSeverity.CRITICAL)


class DatabaseError(BaseError):
    """数据库错误"""
    
    def __init__(self, message: str = "数据库错误", operation: Optional[str] = None, table: Optional[str] = None):
        details = {}
        if operation:
            details['operation'] = operation
        if table:
            details['table'] = table
        details['component'] = 'database'
            
        super().__init__(message, 'DATABASE_ERROR', details, ErrorSeverity.CRITICAL)


class RedisError(SystemError):
    """Redis错误"""
    
    def __init__(self, message: str = "Redis错误", operation: Optional[str] = None, key: Optional[str] = None):
        details = {}
        if operation:
            details['operation'] = operation
        if key:
            details['key'] = key
            
        super().__init__(message, component='redis')
        self.error_code = 'REDIS_ERROR'
        self.details.update(details)


class ConfigurationError(SystemError):
    """配置错误"""
    
    def __init__(self, message: str = "配置错误", config_key: Optional[str] = None, config_value: Optional[str] = None):
        details = {}
        if config_key:
            details['config_key'] = config_key
        if config_value:
            details['config_value'] = config_value
            
        super().__init__(message, component='configuration')
        self.error_code = 'CONFIGURATION_ERROR'
        self.severity = ErrorSeverity.HIGH
        self.details.update(details)


class ExternalServiceError(SystemError):
    """外部服务错误"""
    
    def __init__(self, message: str = "外部服务错误", service_name: Optional[str] = None, status_code: Optional[int] = None):
        details = {}
        if service_name:
            details['service_name'] = service_name
        if status_code:
            details['status_code'] = status_code
            
        super().__init__(message, component='external_service')
        self.error_code = 'EXTERNAL_SERVICE_ERROR'
        self.severity = ErrorSeverity.HIGH
        self.details.update(details)


class RateLimitError(BaseError):
    """速率限制错误"""
    
    def __init__(self, message: str = "请求过于频繁", limit: Optional[int] = None, window: Optional[str] = None):
        details = {}
        if limit:
            details['limit'] = limit
        if window:
            details['window'] = window
            
        super().__init__(message, 'RATE_LIMIT_ERROR', details, ErrorSeverity.MEDIUM)


class TenantError(BaseError):
    """租户相关错误"""
    
    def __init__(self, message: str = "租户错误", tenant_id: Optional[str] = None, operation: Optional[str] = None):
        details = {}
        if tenant_id:
            details['tenant_id'] = tenant_id
        if operation:
            details['operation'] = operation
            
        super().__init__(message, 'TENANT_ERROR', details, ErrorSeverity.MEDIUM)


class ModelError(BaseError):
    """模型相关错误"""
    
    def __init__(self, message: str = "模型错误", model_id: Optional[str] = None, operation: Optional[str] = None):
        details = {}
        if model_id:
            details['model_id'] = model_id
        if operation:
            details['operation'] = operation
            
        super().__init__(message, 'MODEL_ERROR', details, ErrorSeverity.MEDIUM)


class TrainingError(BaseError):
    """训练相关错误"""
    
    def __init__(self, message: str = "训练错误", job_id: Optional[str] = None, stage: Optional[str] = None):
        details = {}
        if job_id:
            details['job_id'] = job_id
        if stage:
            details['stage'] = stage
            
        super().__init__(message, 'TRAINING_ERROR', details, ErrorSeverity.HIGH)


class DatasetError(BaseError):
    """数据集相关错误"""
    
    def __init__(self, message: str = "数据集错误", dataset_id: Optional[str] = None, operation: Optional[str] = None):
        details = {}
        if dataset_id:
            details['dataset_id'] = dataset_id
        if operation:
            details['operation'] = operation
            
        super().__init__(message, 'DATASET_ERROR', details, ErrorSeverity.MEDIUM)


class FileError(BaseError):
    """文件相关错误"""
    
    def __init__(self, message: str = "文件错误", file_path: Optional[str] = None, operation: Optional[str] = None):
        details = {}
        if file_path:
            details['file_path'] = file_path
        if operation:
            details['operation'] = operation
            
        super().__init__(message, 'FILE_ERROR', details, ErrorSeverity.MEDIUM)


class ResourceError(BaseError):
    """资源相关错误"""
    
    def __init__(self, message: str = "资源错误", resource_type: Optional[str] = None, resource_id: Optional[str] = None):
        details = {}
        if resource_type:
            details['resource_type'] = resource_type
        if resource_id:
            details['resource_id'] = resource_id
            
        super().__init__(message, 'RESOURCE_ERROR', details, ErrorSeverity.MEDIUM)


def handle_exception(e: Exception) -> BaseError:
    """处理异常，转换为标准异常格式
    
    Args:
        e: 原始异常
        
    Returns:
        标准异常对象
    """
    if isinstance(e, BaseError):
        return e
    
    # 根据异常类型进行转换
    if isinstance(e, ValueError):
        return ValidationError(str(e))
    elif isinstance(e, PermissionError):
        return AuthorizationError(str(e))
    elif isinstance(e, FileNotFoundError):
        return ResourceNotFoundError(str(e), resource_type='file')
    elif isinstance(e, ConnectionError):
        return ExternalServiceError(str(e))
    else:
        # 未知异常转换为系统错误
        logger.error(f"未处理的异常类型: {type(e).__name__}: {str(e)}", exc_info=True)
        return SystemError(f"未知错误: {str(e)}", error_code=type(e).__name__)


def log_exception(e: BaseError, context: Optional[Dict[str, Any]] = None) -> None:
    """记录异常日志
    
    Args:
        e: 异常对象
        context: 上下文信息
    """
    log_data = {
        'error_code': e.error_code,
        'message': e.message,
        'details': e.details
    }
    
    if context:
        log_data['context'] = context
    
    # 根据异常类型选择日志级别
    if isinstance(e, (ValidationError, AuthenticationError, AuthorizationError, ResourceNotFoundError)):
        logger.warning(f"业务异常: {log_data}")
    elif isinstance(e, (BusinessLogicError, RateLimitError, TenantError, ModelError, TrainingError, DatasetError, FileError)):
        logger.error(f"业务错误: {log_data}")
    else:
        logger.error(f"系统错误: {log_data}", exc_info=True)
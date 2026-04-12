"""核心基础设施模块

提供平台的核心基础设施组件，包括异常处理、日志记录、配置管理等。
"""

from .exceptions import (
    BaseError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    ResourceNotFoundError,
    BusinessLogicError,
    TrainingError,
    SystemError,
    DatabaseError,
    ExternalServiceError,
    handle_exception
)

from .enhanced_exceptions import (
    ErrorSeverity,
    RecoveryStrategy,
    ErrorContext,
    ErrorInfo,
    EnhancedBaseError,
    EnhancedValidationError,
    EnhancedAuthenticationError,
    EnhancedAuthorizationError,
    EnhancedResourceNotFoundError,
    EnhancedSystemError,
    EnhancedTrainingError,
    ExceptionHandler,
    get_exception_handler,
    handle_exceptions,
    error_context,
    safe_execute,
    TrainingExceptionHandler,
    initialize_exception_handlers
)

from .logging_config import (
    setup_logging,
    get_logger
)

from .config_manager import (
    ConfigSource,
    ConfigManager,
    get_config_manager,
    load_config
)

from .middleware import (
    RequestTrackingMiddleware,
    AuthMiddleware,
    RateLimitMiddleware,
    require_permissions,
    setup_middleware,
    handle_errors
)

from .redis_client import (
    init_redis,
    get_redis_client,
    MockRedisClient,
    RedisManager,
    get_redis_manager
)

__all__ = [
    # 基础异常
    'BaseError',
    'ValidationError',
    'AuthenticationError',
    'AuthorizationError',
    'ResourceNotFoundError',
    'BusinessLogicError',
    'TrainingError',
    'SystemError',
    'DatabaseError',
    'ExternalServiceError',
    'handle_exception',
    
    # 增强异常
    'ErrorSeverity',
    'RecoveryStrategy',
    'ErrorContext',
    'ErrorInfo',
    'EnhancedBaseError',
    'EnhancedValidationError',
    'EnhancedAuthenticationError',
    'EnhancedAuthorizationError',
    'EnhancedResourceNotFoundError',
    'EnhancedSystemError',
    'EnhancedTrainingError',
    'ExceptionHandler',
    'get_exception_handler',
    'handle_exceptions',
    'error_context',
    'safe_execute',
    'TrainingExceptionHandler',
    'initialize_exception_handlers',
    
    # 日志配置
    'setup_logging',
    'get_logger',
    
    # 配置管理
    'ConfigSource',
    'ConfigManager',
    'get_config_manager',
    'load_config',
    
    # 中间件
    'RequestTimingMiddleware',
    'CORSMiddleware',
    'SecurityHeadersMiddleware',
    'RequestIDMiddleware',
    'setup_middleware',
    'require_auth',
    'require_role',
    'AuthMiddleware',
    'TenantMiddleware',
    'RequestLoggingMiddleware',
    'ErrorHandlerMiddleware',
    'SecurityMiddleware',
    'handle_errors',
    
    # Redis客户端
    'init_redis',
    'get_redis_client',
    'MockRedisClient',
    'RedisManager',
    'get_redis_manager'
]
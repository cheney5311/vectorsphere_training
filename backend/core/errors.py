"""统一错误码与错误响应格式定义

定义错误码命名空间及常用错误构造器。
"""
from typing import Dict, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# 命名空间：
# 1000xx - validation
# 1001xx - auth  
# 1002xx - internal
# 1003xx - business logic
# 1004xx - resource management
# 1005xx - training specific
# 1006xx - model deployment
# 1007xx - monitoring

class ErrorCategory(Enum):
    """错误分类"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    INTERNAL = "internal"
    BUSINESS_LOGIC = "business_logic"
    RESOURCE = "resource"
    TRAINING = "training"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"

ERROR_CODES: Dict[str, int] = {
    # Validation errors (1000xx)
    'VALIDATION_SCHEMA_FAILED': 100000,
    'VALIDATION_REQUIRED_FIELD': 100001,
    'VALIDATION_INVALID_FORMAT': 100002,
    'VALIDATION_INVALID_TYPE': 100003,
    'VALIDATION_INVALID_RANGE': 100004,
    'VALIDATION_INVALID_ENUM': 100005,
    'INVALID_JSON': 100010,
    'SCHEMA_LOAD_FAILED': 100011,
    'SCHEMA_VERSION_MISMATCH': 100012,
    
    # Auth errors (1001xx)
    'AUTH_UNAUTHORIZED': 100100,
    'AUTH_FORBIDDEN': 100101,
    'AUTH_TOKEN_EXPIRED': 100102,
    'AUTH_TOKEN_INVALID': 100103,
    'AUTH_INSUFFICIENT_PERMISSIONS': 100104,
    'AUTH_USER_NOT_FOUND': 100105,
    'AUTH_INVALID_CREDENTIALS': 100106,
    
    # Internal errors (1002xx)
    'INTERNAL_ERROR': 100200,
    'DATABASE_ERROR': 100201,
    'REDIS_ERROR': 100202,
    'CONFIG_ERROR': 100203,
    'NETWORK_ERROR': 100204,
    'FILE_SYSTEM_ERROR': 100205,
    
    # Business logic errors (1003xx)
    'BUSINESS_INVALID_STATE': 100300,
    'BUSINESS_OPERATION_NOT_ALLOWED': 100301,
    'BUSINESS_RESOURCE_CONFLICT': 100302,
    'BUSINESS_QUOTA_EXCEEDED': 100303,
    
    # Resource management errors (1004xx)
    'RESOURCE_NOT_FOUND': 100400,
    'RESOURCE_ALREADY_EXISTS': 100401,
    'RESOURCE_UNAVAILABLE': 100402,
    'RESOURCE_QUOTA_EXCEEDED': 100403,
    'GPU_ALLOCATION_FAILED': 100404,
    'GPU_NOT_AVAILABLE': 100405,
    
    # Training specific errors (1005xx)
    'TRAINING_SESSION_NOT_FOUND': 100500,
    'TRAINING_ALREADY_RUNNING': 100501,
    'TRAINING_FAILED': 100502,
    'TRAINING_CANCELLED': 100503,
    'TRAINING_CONFIG_INVALID': 100504,
    'CHECKPOINT_LOAD_FAILED': 100505,
    'CHECKPOINT_SAVE_FAILED': 100506,
    
    # Model deployment errors (1006xx)
    'DEPLOYMENT_FAILED': 100600,
    'MODEL_NOT_FOUND': 100601,
    'MODEL_LOAD_FAILED': 100602,
    'SERVICE_UNAVAILABLE': 100603,
    'DEPLOYMENT_CONFIG_INVALID': 100604,
    
    # Monitoring errors (1007xx)
    'MONITORING_DATA_UNAVAILABLE': 100700,
    'METRICS_COLLECTION_FAILED': 100701,
    'ALERT_PROCESSING_FAILED': 100702,
}

# HTTP状态码映射
HTTP_STATUS_MAPPING: Dict[int, int] = {
    # Validation errors -> 400
    100000: 400, 100001: 400, 100002: 400, 100003: 400, 100004: 400, 100005: 400,
    100010: 400, 100011: 500, 100012: 400,
    
    # Auth errors -> 401/403
    100100: 401, 100101: 403, 100102: 401, 100103: 401, 100104: 403, 100105: 401, 100106: 401,
    
    # Internal errors -> 500
    100200: 500, 100201: 500, 100202: 500, 100203: 500, 100204: 500, 100205: 500,
    
    # Business logic errors -> 400/409
    100300: 400, 100301: 400, 100302: 409, 100303: 429,
    
    # Resource errors -> 404/409/503
    100400: 404, 100401: 409, 100402: 503, 100403: 429, 100404: 503, 100405: 503,
    
    # Training errors -> 400/404/500
    100500: 404, 100501: 409, 100502: 500, 100503: 400, 100504: 400, 100505: 500, 100506: 500,
    
    # Deployment errors -> 400/404/500/503
    100600: 500, 100601: 404, 100602: 500, 100603: 503, 100604: 400,
    
    # Monitoring errors -> 503/500
    100700: 503, 100701: 500, 100702: 500,
}

# 重试建议
RETRYABLE_ERRORS: set = {
    100200, 100201, 100202, 100204,  # Internal errors
    100402, 100403, 100404, 100405,  # Resource unavailable
    100502, 100505, 100506,          # Training failures
    100600, 100602, 100603,          # Deployment failures
    100700, 100701, 100702,          # Monitoring failures
}


def make_error(code_key: str, message: str, details: Optional[Dict[str, Any]] = None, 
               retryable: Optional[bool] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """创建标准化错误响应
    
    Args:
        code_key: 错误码键名
        message: 错误消息
        details: 错误详情
        retryable: 是否可重试（None时自动判断）
        context: 错误上下文信息
        
    Returns:
        标准化错误响应字典
    """
    code = ERROR_CODES.get(code_key, 100299)
    
    if retryable is None:
        retryable = code in RETRYABLE_ERRORS
    
    error_response = {
        'error': code_key.lower(),
        'code': code,
        'message': message,
        'details': details or {},
        'retryable': retryable,
        'http_status': HTTP_STATUS_MAPPING.get(code, 500)
    }
    
    if context:
        error_response['context'] = context
    
    # 记录结构化日志
    logger.warning(f"Error generated: {code_key}", extra={
        'error_code': code,
        'error_key': code_key,
        'retryable': retryable,
        'context': context or {}
    })
    
    return error_response


def get_error_category(code_key: str) -> ErrorCategory:
    """获取错误分类"""
    code = ERROR_CODES.get(code_key, 100299)
    
    if 100000 <= code < 100100:
        return ErrorCategory.VALIDATION
    elif 100100 <= code < 100200:
        return ErrorCategory.AUTHENTICATION
    elif 100200 <= code < 100300:
        return ErrorCategory.INTERNAL
    elif 100300 <= code < 100400:
        return ErrorCategory.BUSINESS_LOGIC
    elif 100400 <= code < 100500:
        return ErrorCategory.RESOURCE
    elif 100500 <= code < 100600:
        return ErrorCategory.TRAINING
    elif 100600 <= code < 100700:
        return ErrorCategory.DEPLOYMENT
    elif 100700 <= code < 100800:
        return ErrorCategory.MONITORING
    else:
        return ErrorCategory.INTERNAL


def is_retryable_error(code_key: str) -> bool:
    """判断错误是否可重试"""
    code = ERROR_CODES.get(code_key, 100299)
    return code in RETRYABLE_ERRORS

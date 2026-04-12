"""API 验证中间件

提供基于 JSON Schema 的请求验证装饰器。
"""
import json
import os
from functools import wraps
from typing import Dict, Any, Optional, Union, List
import logging

import jsonschema
from flask import request, jsonify, g

from .errors import make_error, get_error_category, ErrorCategory
from .schema_manager import get_schema_manager, validate_json_data

logger = logging.getLogger(__name__)

# Schema 缓存（保持向后兼容）
_schema_cache: Dict[str, Dict[str, Any]] = {}


class ValidationConfig:
    """验证配置类"""
    
    def __init__(self):
        self.strict_mode = os.getenv('CHECK_API_STRICT_MODE', 'false').lower() == 'true'
        self.log_validation_errors = os.getenv('LOG_VALIDATION_ERRORS', 'true').lower() == 'true'
        self.include_request_id = os.getenv('INCLUDE_REQUEST_ID', 'true').lower() == 'true'
        self.max_error_details = int(os.getenv('MAX_ERROR_DETAILS', '10'))


# 全局验证配置
validation_config = ValidationConfig()


def validate_json_schema(schema: Union[Dict[str, Any], str], 
                        strict_mode: Optional[bool] = None,
                        schema_version: str = "latest",
                        custom_error_handler: Optional[callable] = None):
    """JSON Schema 验证装饰器
    
    Args:
        schema: JSON Schema 定义或schema名称
        strict_mode: 是否启用严格模式，None 时从环境变量读取
        schema_version: schema版本（当schema为字符串时使用）
        custom_error_handler: 自定义错误处理函数
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 检查是否启用严格模式
            if strict_mode is None:
                is_strict = validation_config.strict_mode
            else:
                is_strict = strict_mode
            
            if not is_strict:
                # 非严格模式，跳过验证
                return f(*args, **kwargs)
            
            # 获取请求ID（如果存在）
            request_id = getattr(g, 'request_id', None) if validation_config.include_request_id else None
            
            # 检查请求是否包含 JSON 数据
            if not request.is_json:
                error = make_error(
                    'INVALID_JSON',
                    'Request must contain valid JSON data',
                    context={'request_id': request_id, 'endpoint': request.endpoint}
                )
                if custom_error_handler:
                    return custom_error_handler(error)
                return jsonify(error), error['http_status']
            
            try:
                data = request.get_json()
            except Exception as e:
                error = make_error(
                    'INVALID_JSON',
                    f'Invalid JSON format: {str(e)}',
                    context={'request_id': request_id, 'endpoint': request.endpoint}
                )
                if custom_error_handler:
                    return custom_error_handler(error)
                return jsonify(error), error['http_status']
            
            # 执行schema验证
            if isinstance(schema, str):
                # 使用schema管理器验证
                is_valid, validation_error = validate_json_data(data, schema, schema_version)
                if not is_valid:
                    if validation_config.log_validation_errors:
                        logger.warning(f"Validation failed for {request.endpoint}", extra={
                            'schema_name': schema,
                            'schema_version': schema_version,
                            'request_id': request_id,
                            'error': validation_error
                        })
                    
                    if custom_error_handler:
                        return custom_error_handler(validation_error)
                    return jsonify(validation_error), validation_error['http_status']
            else:
                # 直接使用提供的schema验证
                try:
                    jsonschema.validate(data, schema)
                except jsonschema.ValidationError as e:
                    error_details = {
                        'validation_path': list(e.absolute_path),
                        'failed_value': e.instance,
                        'schema_path': list(e.schema_path)
                    }
                    
                    # 限制错误详情大小
                    if len(str(error_details)) > validation_config.max_error_details * 100:
                        error_details = {
                            'validation_path': list(e.absolute_path),
                            'message': 'Error details truncated due to size'
                        }
                    
                    error = make_error(
                        'VALIDATION_SCHEMA_FAILED',
                        f'Schema validation failed: {e.message}',
                        details=error_details,
                        context={'request_id': request_id, 'endpoint': request.endpoint}
                    )
                    
                    if validation_config.log_validation_errors:
                        logger.warning(f"Schema validation failed for {request.endpoint}", extra={
                            'request_id': request_id,
                            'validation_error': e.message,
                            'validation_path': list(e.absolute_path)
                        })
                    
                    if custom_error_handler:
                        return custom_error_handler(error)
                    return jsonify(error), error['http_status']
                except Exception as e:
                    error = make_error(
                        'INTERNAL_ERROR',
                        f'Validation error: {str(e)}',
                        context={'request_id': request_id, 'endpoint': request.endpoint}
                    )
                    if custom_error_handler:
                        return custom_error_handler(error)
                    return jsonify(error), error['http_status']
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def validate_query_params(param_schema: Dict[str, Any], strict_mode: Optional[bool] = None):
    """查询参数验证装饰器
    
    Args:
        param_schema: 参数schema定义
        strict_mode: 是否启用严格模式
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if strict_mode is None:
                is_strict = validation_config.strict_mode
            else:
                is_strict = strict_mode
            
            if not is_strict:
                return f(*args, **kwargs)
            
            # 验证查询参数
            query_params = dict(request.args)
            
            try:
                jsonschema.validate(query_params, param_schema)
            except jsonschema.ValidationError as e:
                error = make_error(
                    'VALIDATION_SCHEMA_FAILED',
                    f'Query parameter validation failed: {e.message}',
                    details={
                        'validation_path': list(e.absolute_path),
                        'failed_value': e.instance
                    }
                )
                return jsonify(error), error['http_status']
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def validate_response_schema(schema: Union[Dict[str, Any], str], 
                           schema_version: str = "latest",
                           validate_in_production: bool = False):
    """响应schema验证装饰器（主要用于开发和测试）
    
    Args:
        schema: 响应schema定义或schema名称
        schema_version: schema版本
        validate_in_production: 是否在生产环境验证
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            response = f(*args, **kwargs)
            
            # 检查是否需要验证响应
            is_production = os.getenv('FLASK_ENV') == 'production'
            if is_production and not validate_in_production:
                return response
            
            # 提取响应数据
            if hasattr(response, 'get_json'):
                response_data = response.get_json()
            elif isinstance(response, tuple) and len(response) >= 1:
                response_data = response[0]
            else:
                response_data = response
            
            # 验证响应数据
            if isinstance(schema, str):
                is_valid, validation_error = validate_json_data(response_data, schema, schema_version)
                if not is_valid:
                    logger.error(f"Response validation failed for {request.endpoint}", extra={
                        'schema_name': schema,
                        'validation_error': validation_error
                    })
            else:
                try:
                    jsonschema.validate(response_data, schema)
                except jsonschema.ValidationError as e:
                    logger.error(f"Response schema validation failed for {request.endpoint}", extra={
                        'validation_error': e.message,
                        'validation_path': list(e.absolute_path)
                    })
            
            return response
        
        return decorated_function
    return decorator


def get_validation_stats() -> Dict[str, Any]:
    """获取验证统计信息"""
    schema_manager = get_schema_manager()
    available_schemas = schema_manager.get_available_schemas()
    
    return {
        'strict_mode_enabled': validation_config.strict_mode,
        'available_schemas': available_schemas,
        'total_schemas': sum(len(versions) for versions in available_schemas.values()),
        'config': {
            'log_validation_errors': validation_config.log_validation_errors,
            'include_request_id': validation_config.include_request_id,
            'max_error_details': validation_config.max_error_details
        }
    }

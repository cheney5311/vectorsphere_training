"""中间件模块

提供认证、错误处理、请求追踪等中间件功能。"""
import logging
import uuid
import time
from flask import request, jsonify, g
from functools import wraps
from typing import Optional, Dict, Any
import os

from .errors import make_error, get_error_category, ErrorCategory

logger = logging.getLogger(__name__)


class RequestTrackingMiddleware:
    """请求追踪中间件"""
    
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """初始化应用"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
    
    def before_request(self):
        """请求前处理"""
        # 生成请求ID
        g.request_id = str(uuid.uuid4())
        g.start_time = time.time()
        
        # 记录请求开始
        logger.info(f"Request started: {request.method} {request.path}", extra={
            'request_id': g.request_id,
            'method': request.method,
            'path': request.path,
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', '')
        })
    
    def after_request(self, response):
        """请求后处理"""
        if hasattr(g, 'start_time'):
            duration = time.time() - g.start_time
            
            # 记录请求完成
            logger.info(f"Request completed: {response.status_code}", extra={
                'request_id': getattr(g, 'request_id', 'unknown'),
                'status_code': response.status_code,
                'duration_ms': round(duration * 1000, 2),
                'content_length': response.content_length
            })
            
            # 添加响应头
            response.headers['X-Request-ID'] = getattr(g, 'request_id', 'unknown')
            response.headers['X-Response-Time'] = f"{round(duration * 1000, 2)}ms"
        
        return response


class AuthMiddleware:
    """认证中间件"""
    
    def __init__(self, app=None, skip_auth_endpoints=None):
        self.app = app
        self.skip_auth_endpoints = skip_auth_endpoints or ['health', 'metrics', 'docs']
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """初始化应用"""
        app.before_request(self.before_request)
    
    def before_request(self):
        """请求前处理"""
        # 跳过认证的端点
        if request.endpoint in self.skip_auth_endpoints:
            return
        
        # 开发模式下可以跳过认证
        if os.getenv('SKIP_AUTH', 'false').lower() == 'true':
            g.user_id = 'dev_user'
            g.user_roles = ['admin']
            return
        
        # 检查Authorization头
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            error = make_error(
                'AUTH_UNAUTHORIZED',
                'Missing authorization header',
                context={'request_id': getattr(g, 'request_id', None)}
            )
            return jsonify(error), error['http_status']
        
        if not auth_header.startswith('Bearer '):
            error = make_error(
                'AUTH_UNAUTHORIZED',
                'Invalid authorization header format',
                context={'request_id': getattr(g, 'request_id', None)}
            )
            return jsonify(error), error['http_status']
        
        # 提取并验证token
        token = auth_header[7:]  # 移除 'Bearer ' 前缀
        validation_result = self._validate_token(token)
        
        if not validation_result['valid']:
            error_code = validation_result.get('error_code', 'AUTH_UNAUTHORIZED')
            error = make_error(
                error_code,
                validation_result.get('message', 'Invalid token'),
                context={'request_id': getattr(g, 'request_id', None)}
            )
            return jsonify(error), error['http_status']
        
        # 存储用户信息
        g.user_id = validation_result['user_id']
        g.user_roles = validation_result.get('roles', [])
        g.user_permissions = validation_result.get('permissions', [])
    
    def _validate_token(self, token) -> Dict[str, Any]:
        """验证token"""
        try:
            # 这里应该实现实际的token验证逻辑
            # 例如：JWT验证、数据库查询等
            
            if len(token) < 10:
                return {
                    'valid': False,
                    'error_code': 'AUTH_TOKEN_INVALID',
                    'message': 'Token too short'
                }
            
            # 模拟token过期检查
            if token.startswith('expired_'):
                return {
                    'valid': False,
                    'error_code': 'AUTH_TOKEN_EXPIRED',
                    'message': 'Token has expired'
                }
            
            # 模拟成功验证
            return {
                'valid': True,
                'user_id': f"user_{token[:8]}",
                'roles': ['user'],
                'permissions': ['read', 'write']
            }
            
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return {
                'valid': False,
                'error_code': 'AUTH_TOKEN_INVALID',
                'message': 'Token validation failed'
            }


class RateLimitMiddleware:
    """速率限制中间件"""
    
    def __init__(self, app=None, default_limit="100/hour"):
        self.app = app
        self.default_limit = default_limit
        self.request_counts = {}  # 简单的内存存储，生产环境应使用Redis
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        """初始化应用"""
        app.before_request(self.before_request)
    
    def before_request(self):
        """请求前处理"""
        # 跳过某些端点的速率限制
        if request.endpoint in ['health', 'metrics']:
            return
        
        # 获取客户端标识
        client_id = self._get_client_id()
        
        # 检查速率限制
        if self._is_rate_limited(client_id):
            error = make_error(
                'BUSINESS_QUOTA_EXCEEDED',
                'Rate limit exceeded',
                details={'limit': self.default_limit},
                context={'request_id': getattr(g, 'request_id', None)}
            )
            return jsonify(error), error['http_status']
        
        # 记录请求
        self._record_request(client_id)
    
    def _get_client_id(self) -> str:
        """获取客户端标识"""
        # 优先使用用户ID，否则使用IP地址
        if hasattr(g, 'user_id'):
            return f"user:{g.user_id}"
        return f"ip:{request.remote_addr}"
    
    def _is_rate_limited(self, client_id: str) -> bool:
        """检查是否超过速率限制"""
        # 简单实现，生产环境应使用更复杂的算法
        current_time = int(time.time())
        hour_key = f"{client_id}:{current_time // 3600}"
        
        count = self.request_counts.get(hour_key, 0)
        return count >= 100  # 每小时100次请求
    
    def _record_request(self, client_id: str):
        """记录请求"""
        current_time = int(time.time())
        hour_key = f"{client_id}:{current_time // 3600}"
        
        self.request_counts[hour_key] = self.request_counts.get(hour_key, 0) + 1
        
        # 清理过期数据
        cutoff_time = current_time - 7200  # 2小时前
        keys_to_remove = [k for k in self.request_counts.keys() 
                         if int(k.split(':')[-1]) * 3600 < cutoff_time]
        for key in keys_to_remove:
            del self.request_counts[key]


def require_permissions(*required_permissions):
    """权限检查装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not hasattr(g, 'user_permissions'):
                error = make_error(
                    'AUTH_UNAUTHORIZED',
                    'Authentication required',
                    context={'request_id': getattr(g, 'request_id', None)}
                )
                return jsonify(error), error['http_status']
            
            user_permissions = set(g.user_permissions)
            required_perms = set(required_permissions)
            
            if not required_perms.issubset(user_permissions):
                missing_perms = required_perms - user_permissions
                error = make_error(
                    'AUTH_INSUFFICIENT_PERMISSIONS',
                    f'Missing required permissions: {", ".join(missing_perms)}',
                    details={'required': list(required_permissions), 'missing': list(missing_perms)},
                    context={'request_id': getattr(g, 'request_id', None)}
                )
                return jsonify(error), error['http_status']
            
            return f(*args, **kwargs)
        
        return decorated_function
    return decorator


def handle_errors(app):
    """设置全局错误处理"""
    
    @app.errorhandler(400)
    def bad_request(error):
        error_response = make_error(
            'VALIDATION_SCHEMA_FAILED',
            'Bad request',
            context={'request_id': getattr(g, 'request_id', None)}
        )
        return jsonify(error_response), error_response['http_status']
    
    @app.errorhandler(401)
    def unauthorized(error):
        error_response = make_error(
            'AUTH_UNAUTHORIZED',
            'Unauthorized',
            context={'request_id': getattr(g, 'request_id', None)}
        )
        return jsonify(error_response), error_response['http_status']
    
    @app.errorhandler(403)
    def forbidden(error):
        error_response = make_error(
            'AUTH_FORBIDDEN',
            'Forbidden',
            context={'request_id': getattr(g, 'request_id', None)}
        )
        return jsonify(error_response), error_response['http_status']
    
    @app.errorhandler(404)
    def not_found(error):
        error_response = make_error(
            'RESOURCE_NOT_FOUND',
            'Resource not found',
            context={'request_id': getattr(g, 'request_id', None)}
        )
        return jsonify(error_response), error_response['http_status']
    
    @app.errorhandler(429)
    def too_many_requests(error):
        error_response = make_error(
            'BUSINESS_QUOTA_EXCEEDED',
            'Too many requests',
            context={'request_id': getattr(g, 'request_id', None)}
        )
        return jsonify(error_response), error_response['http_status']
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}", extra={
            'request_id': getattr(g, 'request_id', None)
        })
        error_response = make_error(
            'INTERNAL_ERROR',
            'Internal server error',
            context={'request_id': getattr(g, 'request_id', None)}
        )
        return jsonify(error_response), error_response['http_status']
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.error(f"Unhandled exception: {error}", exc_info=True, extra={
            'request_id': getattr(g, 'request_id', None)
        })
        error_response = make_error(
            'INTERNAL_ERROR',
            'An unexpected error occurred',
            context={'request_id': getattr(g, 'request_id', None)}
        )
        return jsonify(error_response), error_response['http_status']


def setup_middleware(app):
    """设置所有中间件"""
    # 请求追踪中间件
    RequestTrackingMiddleware(app)
    
    # 认证中间件
    AuthMiddleware(app)
    
    # 速率限制中间件
    RateLimitMiddleware(app)
    
    # 错误处理
    handle_errors(app)
    
    logger.info("All middleware components initialized")
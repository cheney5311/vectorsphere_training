"""中间件模块

提供认证、权限、租户等中间件功能。
"""

from .auth import AuthMiddleware, get_auth_middleware, set_auth_middleware, require_auth, get_current_user, get_current_user_id
from .permissions import PermissionMiddleware, get_permission_middleware, set_permission_middleware, require_permission, Permission, Role
from .logging import RequestLoggingMiddleware, get_logging_middleware, init_logging_middleware, get_request_id, get_request_duration

__all__ = [
    # 认证中间件
    'AuthMiddleware',
    'get_auth_middleware',
    'set_auth_middleware',
    'require_auth',
    'get_current_user',
    'get_current_user_id',
    
    # 权限中间件
    'PermissionMiddleware',
    'get_permission_middleware',
    'set_permission_middleware',
    'require_permission',
    'Permission',
    'Role',
    
    # 日志中间件
    'RequestLoggingMiddleware',
    'get_logging_middleware',
    'init_logging_middleware',
    'get_request_id',
    'get_request_duration',
]
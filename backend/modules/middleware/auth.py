"""认证中间件模块

实现JWT令牌验证、用户认证和权限控制功能
"""

import logging
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from functools import wraps
from flask import Flask, request, g, abort, jsonify, current_app
from werkzeug.exceptions import Unauthorized, Forbidden

from backend.schemas.auth_models import User, UserStatus
from backend.modules.database.manager import get_database_manager

logger = logging.getLogger(__name__)


class AuthMiddleware:
    """认证中间件"""
    
    def __init__(self, app: Optional[Flask] = None):
        """初始化中间件
        
        Args:
            app: Flask应用实例
        """
        self.app = app
        self.token_cache = {}  # 简单的令牌缓存
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """初始化Flask应用
        
        Args:
            app: Flask应用实例
        """
        self.app = app
        
        # 注册中间件
        app.before_request(self.before_request)
        
        # 注册错误处理器
        app.errorhandler(401)(self.handle_unauthorized)
        app.errorhandler(403)(self.handle_forbidden)
    
    def before_request(self):
        """请求前处理"""
        try:
            # 跳过不需要认证的路径
            if self._should_skip_auth():
                return
            
            # 提取认证信息
            auth_info = self._extract_auth_info()
            if not auth_info:
                logger.warning(f"未找到认证信息: {request.url}")
                return self._handle_missing_auth()
            
            # 验证认证信息
            user = self._validate_auth(auth_info)
            if not user:
                logger.warning(f"认证验证失败: {request.url}")
                return self._handle_invalid_auth()
            
            # 检查用户状态
            if not self._validate_user_status(user):
                logger.warning(f"用户状态无效: {user.id} - {user.status}")
                return self._handle_inactive_user(user)
            
            # 设置用户上下文
            self._set_user_context(user)
            
            logger.debug(f"用户认证成功: {user.id}")
            
        except Exception as e:
            logger.error(f"认证中间件处理失败: {e}")
            return jsonify({
                "error": "auth_middleware_error",
                "message": "认证处理失败"
            }), 500
    
    def _should_skip_auth(self) -> bool:
        """检查是否应该跳过认证
        
        Returns:
            是否跳过
        """
        # 跳过的路径列表
        skip_paths = [
            '/health',
            '/metrics',
            '/docs',
            '/openapi.json',
            '/static',
            '/favicon.ico'
        ]
        
        # 跳过的路径前缀
        skip_prefixes = [
            '/api/auth/login',
            '/api/auth/register',
            '/api/auth/refresh',
            '/api/system/health',
            '/api/public'
        ]
        
        path = request.path
        
        # 检查完全匹配的路径
        if path in skip_paths:
            return True
        
        # 检查前缀匹配的路径
        for prefix in skip_prefixes:
            if path.startswith(prefix):
                return True
        
        return False
    
    def _extract_auth_info(self) -> Optional[Dict[str, Any]]:
        """提取认证信息
        
        Returns:
            认证信息字典
        """
        # 1. 从Authorization头中提取Bearer令牌
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header[7:]  # 移除'Bearer '前缀
            return {
                'type': 'bearer',
                'token': token
            }
        
        # 2. 从API Key头中提取
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return {
                'type': 'api_key',
                'key': api_key
            }
        
        # 3. 从查询参数中提取API Key
        api_key = request.args.get('api_key')
        if api_key:
            return {
                'type': 'api_key',
                'key': api_key
            }
        
        return None
    
    def _validate_auth(self, auth_info: Dict[str, Any]) -> Optional[User]:
        """验证认证信息
        
        Args:
            auth_info: 认证信息
            
        Returns:
            用户对象
        """
        try:
            if auth_info['type'] == 'bearer':
                return self._validate_jwt_token(auth_info['token'])
            elif auth_info['type'] == 'api_key':
                return self._validate_api_key(auth_info['key'])
            else:
                logger.warning(f"未知的认证类型: {auth_info['type']}")
                return None
                
        except Exception as e:
            logger.error(f"认证验证失败: {e}")
            return None
    
    def _validate_jwt_token(self, token: str) -> Optional[User]:
        """验证JWT令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            用户对象
        """
        try:
            # 先从缓存中检查
            if token in self.token_cache:
                cached_info = self.token_cache[token]
                # 检查缓存是否过期
                if datetime.now() < cached_info['expires_at']:
                    return cached_info['user']
                else:
                    # 缓存过期，删除
                    del self.token_cache[token]
            
            # 解码JWT令牌
            secret_key = current_app.config.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
            payload = jwt.decode(token, secret_key, algorithms=['HS256'])
            
            # 获取用户ID
            user_id = payload.get('sub')
            if not user_id:
                return None
            
            # 从数据库获取用户
            db_manager = get_database_manager()
            with db_manager.get_db_session() as session:
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    return None
            
            # 缓存令牌信息（1小时）
            self.token_cache[token] = {
                'user': user,
                'expires_at': datetime.now() + timedelta(hours=1)
            }
            
            return user
                
        except jwt.ExpiredSignatureError:
            logger.warning("JWT令牌已过期")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"无效的JWT令牌: {e}")
            return None
        except Exception as e:
            logger.error(f"JWT令牌验证失败: {e}")
            return None
    
    def _validate_api_key(self, api_key: str) -> Optional[User]:
        """验证API密钥
        
        Args:
            api_key: API密钥
            
        Returns:
            用户对象
        """
        try:
            # 这里应该实现API密钥验证逻辑
            # 为简化起见，返回None
            logger.debug(f"验证API密钥: {api_key}")
            return None
            
        except Exception as e:
            logger.error(f"API密钥验证失败: {e}")
            return None
    
    def _validate_user_status(self, user: User) -> bool:
        """验证用户状态
        
        Args:
            user: 用户对象
            
        Returns:
            是否有效
        """
        return user.is_active and user.status == UserStatus.ACTIVE.value
    
    def _set_user_context(self, user: User):
        """设置用户上下文
        
        Args:
            user: 用户对象
        """
        g.current_user = user
        g.current_user_id = user.id
    
    def _handle_missing_auth(self):
        """处理缺少认证信息的情况"""
        return jsonify({
            "error": "missing_auth",
            "message": "缺少认证信息"
        }), 401
    
    def _handle_invalid_auth(self):
        """处理无效认证信息的情况"""
        return jsonify({
            "error": "invalid_auth",
            "message": "认证信息无效"
        }), 401
    
    def _handle_inactive_user(self, user: User):
        """处理非活跃用户的情况"""
        return jsonify({
            "error": "inactive_user",
            "message": "用户账户未激活"
        }), 401
    
    def handle_unauthorized(self, error):
        """处理未授权错误"""
        return jsonify({
            "error": "unauthorized",
            "message": "未授权访问"
        }), 401
    
    def handle_forbidden(self, error):
        """处理禁止访问错误"""
        return jsonify({
            "error": "forbidden",
            "message": "禁止访问"
        }), 403


def require_auth(f):
    """认证装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # 检查是否有当前用户
        if not hasattr(g, 'current_user') or not g.current_user:
            abort(401)
        return f(*args, **kwargs)
    return decorated_function


def get_current_user() -> Optional[User]:
    """获取当前用户"""
    return getattr(g, 'current_user', None)


def get_current_user_id() -> Optional[str]:
    """获取当前用户ID"""
    return getattr(g, 'current_user_id', None)


# 全局认证中间件实例
_auth_middleware: Optional[AuthMiddleware] = None


def get_auth_middleware() -> AuthMiddleware:
    """获取全局认证中间件实例"""
    global _auth_middleware
    if _auth_middleware is None:
        _auth_middleware = AuthMiddleware()
    return _auth_middleware


def set_auth_middleware(auth_middleware: AuthMiddleware):
    """设置全局认证中间件实例"""
    global _auth_middleware
    _auth_middleware = auth_middleware
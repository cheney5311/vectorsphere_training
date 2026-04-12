"""认证工具

提供认证相关的工具函数。
"""

import jwt
import bcrypt
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import os

# JWT配置
def get_jwt_secret_key() -> str:
    """获取JWT密钥"""
    # 直接使用环境变量或默认值
    return os.getenv('JWT_SECRET_KEY', 'default-secret-key-change-in-production')

def get_jwt_algorithm() -> str:
    """获取JWT算法"""
    # 直接返回默认算法
    return 'HS256'

# 为了保持向后兼容性，保留这些常量
JWT_SECRET_KEY = get_jwt_secret_key()
JWT_ALGORITHM = get_jwt_algorithm()
JWT_EXPIRATION_HOURS = 24

def generate_token(user_data: Dict[str, Any], expires_in_hours: int = JWT_EXPIRATION_HOURS) -> str:
    """
    生成JWT令牌
    
    Args:
        user_data: 用户数据
        expires_in_hours: 过期时间（小时）
        
    Returns:
        JWT令牌
    """
    payload = {
        'user_id': user_data.get('user_id'),
        'username': user_data.get('username'),
        'role': user_data.get('role', 'user'),
        'exp': datetime.utcnow() + timedelta(hours=expires_in_hours),
        'iat': datetime.utcnow()
    }
    
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    验证JWT令牌
    
    Args:
        token: JWT令牌
        
    Returns:
        解码后的用户数据，验证失败返回None
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def hash_password(password: str) -> str:
    """
    哈希密码
    
    Args:
        password: 明文密码
        
    Returns:
        哈希后的密码
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed_password: str) -> bool:
    """
    验证密码
    
    Args:
        password: 明文密码
        hashed_password: 哈希密码
        
    Returns:
        密码是否匹配
    """
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

def generate_refresh_token() -> str:
    """
    生成刷新令牌
    
    Returns:
        刷新令牌
    """
    return secrets.token_urlsafe(32)

def extract_token_from_header(authorization_header: str) -> Optional[str]:
    """
    从Authorization头中提取令牌
    
    Args:
        authorization_header: Authorization头值
        
    Returns:
        提取的令牌，失败返回None
    """
    if not authorization_header:
        return None
    
    parts = authorization_header.split()
    if len(parts) != 2 or parts[0].lower() != 'bearer':
        return None
    
    return parts[1]

def check_permission(user_role: str, required_permission: str) -> bool:
    """
    检查用户权限
    
    Args:
        user_role: 用户角色
        required_permission: 需要的权限
        
    Returns:
        是否有权限
    """
    # 权限层级：admin > user > viewer
    role_hierarchy = {
        'admin': ['admin', 'user', 'viewer'],
        'user': ['user', 'viewer'],
        'viewer': ['viewer']
    }
    
    allowed_roles = role_hierarchy.get(user_role, [])
    return required_permission in allowed_roles

# 导入Flask相关模块用于装饰器
from flask import request, jsonify, g
import functools

def token_required(f):
    """
    JWT认证装饰器
    
    Args:
        f: 被装饰的函数
        
    Returns:
        装饰后的函数
    """
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': '缺少认证头'}), 401
        
        try:
            # 解析Bearer Token
            token = auth_header.split(' ')[1] if auth_header.startswith('Bearer ') else auth_header
        except IndexError:
            return jsonify({'error': '认证头格式错误'}), 401
        
        # 验证JWT Token
        user_data = verify_token(token)
        if not user_data:
            return jsonify({'error': 'Token无效或已过期'}), 401
        
        # 使用AuthService获取完整的用户对象
        try:
            from backend.auth.service import AuthService
            auth_service = AuthService()
            
            user_id = user_data.get('user_id')
            if not user_id:
                return jsonify({'error': '无效的用户ID'}), 401
            
            current_user = auth_service.get_user_by_id(int(user_id))
            
            if not current_user or not current_user.is_active:
                return jsonify({'error': '用户不存在或已被禁用'}), 401
            
            # 将用户对象添加到函数参数中
            return f(current_user, *args, **kwargs)
            
        except Exception as e:
            # 如果AuthService不可用，创建一个简单的用户对象
            class SimpleUser:
                def __init__(self, user_data):
                    self.user_id = user_data.get('user_id')
                    self.username = user_data.get('username')
                    self.role = user_data.get('role', 'user')
                    self.is_active = True
            
            current_user = SimpleUser(user_data)
            return f(current_user, *args, **kwargs)
    
    return decorated_function
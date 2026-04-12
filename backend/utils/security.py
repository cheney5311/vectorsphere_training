#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""安全工具模块
提供密码哈希、验证和令牌生成功能
"""

import os
import time
import hashlib
import hmac
import base64
import json
import logging
from typing import Dict, Tuple, Any

import jwt

logger = logging.getLogger(__name__)

# 从环境变量或配置文件读取密钥 - 统一使用相同的密钥
JWT_SECRET = os.getenv('JWT_SECRET', 'your-secret-key-change-in-production')
JWT_ALGORITHM = os.getenv('JWT_ALGORITHM', 'HS256')
ACCESS_TOKEN_EXPIRE = int(os.getenv('ACCESS_TOKEN_EXPIRE', '3600'))  # 1小时
REFRESH_TOKEN_EXPIRE = int(os.getenv('REFRESH_TOKEN_EXPIRE', '2592000'))  # 30天


def hash_password(password: str) -> str:
    """
    对密码进行哈希处理
    
    Args:
        password: 原始密码
        
    Returns:
        哈希后的密码
    """
    # 生成随机盐值
    salt = os.urandom(32)
    
    # 使用PBKDF2算法进行哈希
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )
    
    # 将盐值和哈希值一起存储
    return base64.b64encode(salt + key).decode('utf-8')


def verify_password(stored_password: str, provided_password: str) -> bool:
    """
    验证密码是否正确
    
    Args:
        stored_password: 存储的哈希密码
        provided_password: 用户提供的密码
        
    Returns:
        密码是否正确
    """
    try:
        # 解码存储的密码
        decoded = base64.b64decode(stored_password)
        
        # 提取盐值和哈希值
        salt = decoded[:32]
        key = decoded[32:]
        
        # 使用相同的盐值和算法计算提供的密码的哈希值
        new_key = hashlib.pbkdf2_hmac(
            'sha256',
            provided_password.encode('utf-8'),
            salt,
            100000
        )
        
        # 使用恒定时间比较，防止计时攻击
        return hmac.compare_digest(key, new_key)
    
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False


def generate_tokens(user_data: Dict[str, Any], remember_me: bool = False) -> Tuple[str, str]:
    """
    生成访问令牌和刷新令牌
    
    Args:
        user_data: 用户数据
        remember_me: 是否记住登录状态
        
    Returns:
        (访问令牌, 刷新令牌)
    """
    now = int(time.time())
    
    # 访问令牌有效期
    access_token_expire = now + ACCESS_TOKEN_EXPIRE
    
    # 刷新令牌有效期，如果记住登录状态则使用较长的有效期
    refresh_token_expire = now + REFRESH_TOKEN_EXPIRE if remember_me else now + (ACCESS_TOKEN_EXPIRE * 24)
    
    # 创建访问令牌
    access_token_payload = {
        'sub': user_data['id'],
        'user': {
            'id': user_data['id'],
            'username': user_data['username']
        },
        'iat': now,
        'exp': access_token_expire,
        'type': 'access'
    }
    
    # 创建刷新令牌
    refresh_token_payload = {
        'sub': user_data['id'],
        'iat': now,
        'exp': refresh_token_expire,
        'type': 'refresh'
    }
    
    # 签名令牌
    access_token = jwt.encode(access_token_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    refresh_token = jwt.encode(refresh_token_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    return access_token, refresh_token


def verify_token(token: str) -> Tuple[bool, Dict[str, Any]]:
    """
    验证令牌
    
    Args:
        token: JWT令牌
        
    Returns:
        (是否有效, 令牌数据)
    """
    try:
        # 解码令牌
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        # 检查令牌类型
        if 'type' not in payload:
            return False, {'error': 'Invalid token type'}
        
        # 检查令牌是否过期
        if 'exp' in payload and int(time.time()) > payload['exp']:
            return False, {'error': 'Token expired'}
        
        return True, payload
    
    except jwt.ExpiredSignatureError:
        return False, {'error': 'Token expired'}
    
    except jwt.InvalidTokenError:
        return False, {'error': 'Invalid token'}
    
    except Exception as e:
        logger.error(f"Token verification failed: {e}")
        return False, {'error': f'Token verification failed: {str(e)}'}


def refresh_access_token(refresh_token: str) -> Tuple[bool, Dict[str, Any]]:
    """
    使用刷新令牌生成新的访问令牌
    
    Args:
        refresh_token: 刷新令牌
        
    Returns:
        (是否成功, 新的访问令牌或错误信息)
    """
    # 验证刷新令牌
    valid, payload = verify_token(refresh_token)
    
    if not valid:
        return False, payload
    
    # 检查令牌类型
    if payload.get('type') != 'refresh':
        return False, {'error': 'Invalid token type'}
    
    try:
        # 获取用户信息
        from backend.db.database import get_db_connection
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, username, email, phone FROM users WHERE id = ?",
            (payload['sub'],)
        )
        
        user = cursor.fetchone()
        
        if not user:
            return False, {'error': 'User not found'}
        
        # 创建用户数据
        user_data = {
            'id': user[0],
            'username': user[1],
            'email': user[2],
            'phone': user[3]
        }
        
        # 生成新的访问令牌
        access_token, _ = generate_tokens(user_data)
        
        return True, {
            'access_token': access_token,
            'user': user_data
        }
    
    except Exception as e:
        logger.error(f"Failed to refresh access token: {e}")
        return False, {'error': f'Failed to refresh access token: {str(e)}'}
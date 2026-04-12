#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用户服务层
提供与backend/services/user_service.py接口一致的用户服务实现。
"""

from typing import List, Optional, Dict, Any
from sqlalchemy import and_, or_
from datetime import datetime, timedelta
import hashlib
import secrets
import re

# 添加项目根目录到Python路径
import sys
import os as os_path
sys.path.insert(0, os_path.dirname(os_path.dirname(os_path.abspath(__file__))))

from backend.core.exceptions import ValidationError, ResourceNotFoundError, AuthenticationError
from modules.database.models import User, UserSession
from modules.database.enums import UserStatus, RoleType as UserRole
from modules.database.database_manager import get_db_session

class UserService:
    """用户服务类"""
    
    def __init__(self, db_session=None):
        self.db = db_session or get_db_session()
    
    def _hash_password(self, password: str) -> str:
        """密码哈希"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, hashed_password: str) -> bool:
        """验证密码"""
        return self._hash_password(password) == hashed_password
    
    def _validate_email(self, email: str) -> bool:
        """验证邮箱格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def _validate_password(self, password: str) -> bool:
        """验证密码强度"""
        # 至少8位，包含字母和数字
        if len(password) < 8:
            return False
        if not re.search(r'[a-zA-Z]', password):
            return False
        if not re.search(r'\d', password):
            return False
        return True
    
    def create_user(self, **kwargs) -> User:
        """创建用户"""
        # 验证必需字段
        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if field not in kwargs or not kwargs[field]:
                raise ValidationError(f"Missing required field: {field}")
        
        # 验证邮箱格式
        if not self._validate_email(kwargs['email']):
            raise ValidationError("Invalid email format")
        
        # 验证密码强度
        if not self._validate_password(kwargs['password']):
            raise ValidationError("Password must be at least 8 characters and contain letters and numbers")
        
        # 检查用户名是否已存在
        existing_user = self.db.query(User).filter(
            (User.username == kwargs['username']) | (User.email == kwargs['email'])
        ).first()
        
        if existing_user:
            if existing_user.username == kwargs['username']:
                raise ValidationError("Username already exists")
            if existing_user.email == kwargs['email']:
                raise ValidationError("Email already exists")
        
        # 创建用户实例
        user = User(
            username=kwargs['username'],
            email=kwargs['email'],
            password_hash=self._hash_password(kwargs['password']),
            full_name=kwargs.get('full_name'),
            role=kwargs.get('role', UserRole.USER),
            is_email_verified=kwargs.get('is_email_verified', False)
        )
        
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        
        # 创建用户档案
        self.create_user_profile(user.id)
        
        return user
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """根据ID获取用户"""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_id_required(self, user_id: str) -> User:
        """根据ID获取用户（必须存在）"""
        user = self.get_user_by_id(user_id)
        if not user:
            raise ResourceNotFoundError(f"User not found: {user_id}")
        return user
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        return self.db.query(User).filter(User.email == email).first()
    
    def authenticate_user(self, username_or_email: str, password: str) -> Optional[User]:
        """用户认证"""
        # 尝试通过用户名或邮箱查找用户
        user = self.db.query(User).filter(
            (User.username == username_or_email) | (User.email == username_or_email)
        ).first()
        
        if not user:
            return None
        
        if user.status != UserStatus.ACTIVE:
            raise AuthenticationError("User account is not active")
        
        if not self._verify_password(password, user.password_hash):
            return None
        
        # 更新最后登录时间
        user.last_login_at = datetime.utcnow()
        user.login_count += 1
        self.db.commit()
        
        return user
    
    def update_user(self, user_id: str, **kwargs) -> User:
        """更新用户信息"""
        user = self.get_user_by_id_required(user_id)
        
        # 更新允许的字段
        updatable_fields = ['full_name', 'email', 'avatar_url', 'timezone', 'language']
        
        for field in updatable_fields:
            if field in kwargs:
                if field == 'email' and kwargs[field] != user.email:
                    # 验证新邮箱格式
                    if not self._validate_email(kwargs[field]):
                        raise ValidationError("Invalid email format")
                    # 检查邮箱是否已被使用
                    existing_user = self.db.query(User).filter(
                        and_(User.email == kwargs[field], User.id != user_id)
                    ).first()
                    if existing_user:
                        raise ValidationError("Email already exists")
                    # 重置邮箱验证状态
                    user.is_email_verified = False
                
                setattr(user, field, kwargs[field])
        
        user.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    def change_password(self, user_id: str, old_password: str, new_password: str) -> User:
        """修改密码"""
        user = self.get_user_by_id_required(user_id)
        
        # 验证旧密码
        if not self._verify_password(old_password, user.password_hash):
            raise ValidationError("Invalid old password")
        
        # 验证新密码强度
        if not self._validate_password(new_password):
            raise ValidationError("Password must be at least 8 characters and contain letters and numbers")
        
        # 更新密码
        user.password_hash = self._hash_password(new_password)
        user.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    def reset_password(self, user_id: str, new_password: str) -> User:
        """重置密码（管理员操作）"""
        user = self.get_user_by_id_required(user_id)
        
        # 验证新密码强度
        if not self._validate_password(new_password):
            raise ValidationError("Password must be at least 8 characters and contain letters and numbers")
        
        # 更新密码
        user.password_hash = self._hash_password(new_password)
        user.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    def update_user_status(self, user_id: str, status: UserStatus) -> User:
        """更新用户状态"""
        user = self.get_user_by_id_required(user_id)
        user.status = status
        user.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    def verify_email(self, user_id: str) -> User:
        """验证邮箱"""
        user = self.get_user_by_id_required(user_id)
        user.is_email_verified = True
        user.email_verified_at = datetime.utcnow()
        user.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(user)
        
        return user
    
    def delete_user(self, user_id: str) -> bool:
        """删除用户"""
        user = self.get_user_by_id_required(user_id)
        
        # 软删除：更新状态为已删除
        user.status = UserStatus.DELETED
        user.updated_at = datetime.utcnow()
        
        self.db.commit()
        
        return True
    
    def get_users(self, status: Optional[UserStatus] = None,
                 role: Optional[UserRole] = None,
                 limit: int = 100, offset: int = 0) -> List[User]:
        """获取用户列表"""
        query = self.db.query(User)
        
        if status:
            query = query.filter(User.status == status)
        if role:
            query = query.filter(User.role == role)
        
        return query.order_by(User.created_at.desc()).offset(offset).limit(limit).all()
    
    def search_users(self, query: str, limit: int = 100) -> List[User]:
        """搜索用户"""
        search_filter = or_(
            User.username.ilike(f"%{query}%"),
            User.full_name.ilike(f"%{query}%"),
            User.email.ilike(f"%{query}%")
        )
        
        return self.db.query(User).filter(
            and_(search_filter, User.status == UserStatus.ACTIVE)
        ).limit(limit).all()
    
    # 用户档案管理 - TODO: UserProfile模型不存在，需要实现
    # def create_user_profile(self, user_id: str, **kwargs):
    #     """创建用户档案"""
    #     # TODO: UserProfile模型不存在，需要实现
    #     pass
    # 
    # def get_user_profile(self, user_id: str):
    #     """获取用户档案"""
    #     # TODO: UserProfile模型不存在，需要实现
    #     return None
    # 
    # def get_user_profile_required(self, user_id: str):
    #     """获取用户档案（必须存在）"""
    #     # TODO: UserProfile模型不存在，需要实现
    #     raise ResourceNotFoundError(f"User profile not found for user: {user_id}")
    # 
    # def update_user_profile(self, user_id: str, **kwargs):
    #     """更新用户档案"""
    #     # TODO: UserProfile模型不存在，需要实现
        
    #     # 更新允许的字段
    #     updatable_fields = [
    #         'bio', 'company', 'location', 'website', 'github_url',
    #         'linkedin_url', 'twitter_url', 'skills', 'interests', 'preferences'
    #     ]
    #     
    #     for field in updatable_fields:
    #         if field in kwargs:
    #             setattr(profile, field, kwargs[field])
    #     
    #     profile.updated_at = datetime.utcnow()
    #     self.db.commit()
    #     self.db.refresh(profile)
    #     
    #     return profile
    #     pass
    
    # 用户会话管理
    def create_user_session(self, user_id: str, **kwargs) -> UserSession:
        """创建用户会话"""
        # 验证用户存在
        user = self.get_user_by_id_required(user_id)
        
        # 生成会话令牌
        session_token = secrets.token_urlsafe(32)
        
        # 创建会话实例
        session = UserSession(
            user_id=user_id,
            session_token=session_token,
            ip_address=kwargs.get('ip_address'),
            user_agent=kwargs.get('user_agent'),
            expires_at=kwargs.get('expires_at', datetime.utcnow() + timedelta(days=30))
        )
        
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        
        return session
    
    def get_user_session_by_token(self, session_token: str) -> Optional[UserSession]:
        """根据令牌获取用户会话"""
        return self.db.query(UserSession).filter(
            and_(
                UserSession.session_token == session_token,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.utcnow()
            )
        ).first()
    
    def get_user_sessions(self, user_id: str, is_active: Optional[bool] = None) -> List[UserSession]:
        """获取用户的所有会话"""
        query = self.db.query(UserSession).filter(UserSession.user_id == user_id)
        
        if is_active is not None:
            query = query.filter(UserSession.is_active == is_active)
        
        return query.order_by(UserSession.created_at.desc()).all()
    
    def update_session_activity(self, session_token: str) -> Optional[UserSession]:
        """更新会话活动时间"""
        session = self.get_user_session_by_token(session_token)
        if session:
            session.last_activity_at = datetime.utcnow()
            self.db.commit()
            self.db.refresh(session)
        return session
    
    def invalidate_user_session(self, session_token: str) -> bool:
        """使会话失效"""
        session = self.get_user_session_by_token(session_token)
        if session:
            session.is_active = False
            session.updated_at = datetime.utcnow()
            self.db.commit()
            return True
        return False
    
    def invalidate_all_user_sessions(self, user_id: str) -> int:
        """使用户的所有会话失效"""
        count = self.db.query(UserSession).filter(
            and_(
                UserSession.user_id == user_id,
                UserSession.is_active == True
            )
        ).update({'is_active': False, 'updated_at': datetime.utcnow()})
        
        self.db.commit()
        return count
    
    def cleanup_expired_sessions(self) -> int:
        """清理过期会话"""
        count = self.db.query(UserSession).filter(
            UserSession.expires_at <= datetime.utcnow()
        ).update({'is_active': False, 'updated_at': datetime.utcnow()})
        
        self.db.commit()
        return count
    
    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户统计信息"""
        user = self.get_user_by_id_required(user_id)
        
        # 统计活跃会话数
        active_sessions = self.db.query(UserSession).filter(
            and_(
                UserSession.user_id == user_id,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.utcnow()
            )
        ).count()
        
        return {
            'login_count': user.login_count,
            'last_login_at': user.last_login_at,
            'active_sessions': active_sessions,
            'account_age_days': (datetime.utcnow() - user.created_at).days,
            'is_email_verified': user.is_email_verified,
            'status': user.status.value,
            'role': user.role.value
        }


# 全局用户服务实例
_global_user_service = UserService()


def get_user_service() -> UserService:
    """获取全局用户服务实例"""
    return _global_user_service
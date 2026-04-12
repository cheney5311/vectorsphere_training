"""用户数据访问层

提供用户相关的数据库访问功能。
"""

import logging
from typing import List, Optional, Dict, Any
from sqlalchemy import and_, or_
from datetime import datetime, timedelta

from backend.core.exceptions import ValidationError, ResourceNotFoundError, AuthenticationError
from backend.schemas.base_models import User, UserSession
from backend.schemas.enums import UserStatus, RoleType as UserRole
from backend.modules.database.manager import get_database_manager

logger = logging.getLogger(__name__)


class UserRepository:
    """用户数据访问层"""
    
    def __init__(self):
        self._db_manager = get_database_manager()
    
    def create_user(self, user_data: Dict[str, Any]) -> User:
        """创建用户
        
        Args:
            user_data: 用户数据
            
        Returns:
            创建的用户
        """
        try:
            with self._db_manager.get_db_session() as db:
                user = User(**user_data)
                db.add(user)
                db.commit()
                db.refresh(user)
                return user
        except Exception as e:
            logger.error(f"创建用户失败: {e}")
            raise ValidationError(f"创建用户失败: {e}")
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """根据ID获取用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户对象，如果不存在则返回None
        """
        try:
            with self._db_manager.get_db_session() as db:
                return db.query(User).filter(User.id == user_id).first()
        except Exception as e:
            logger.error(f"获取用户失败: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户
        
        Args:
            username: 用户名
            
        Returns:
            用户对象，如果不存在则返回None
        """
        try:
            with self._db_manager.get_db_session() as db:
                return db.query(User).filter(User.username == username).first()
        except Exception as e:
            logger.error(f"根据用户名获取用户失败: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户
        
        Args:
            email: 邮箱
            
        Returns:
            用户对象，如果不存在则返回None
        """
        try:
            with self._db_manager.get_db_session() as db:
                return db.query(User).filter(User.email == email).first()
        except Exception as e:
            logger.error(f"根据邮箱获取用户失败: {e}")
            return None
    
    def get_user_by_username_or_email(self, username_or_email: str) -> Optional[User]:
        """根据用户名或邮箱获取用户
        
        Args:
            username_or_email: 用户名或邮箱
            
        Returns:
            用户对象，如果不存在则返回None
        """
        try:
            with self._db_manager.get_db_session() as db:
                return db.query(User).filter(
                    or_(User.username == username_or_email, User.email == username_or_email)
                ).first()
        except Exception as e:
            logger.error(f"根据用户名或邮箱获取用户失败: {e}")
            return None
    
    def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Optional[User]:
        """更新用户信息
        
        Args:
            user_id: 用户ID
            update_data: 更新数据
            
        Returns:
            更新后的用户对象，如果不存在则返回None
        """
        try:
            with self._db_manager.get_db_session() as db:
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    return None
                
                for key, value in update_data.items():
                    if hasattr(user, key):
                        setattr(user, key, value)
                
                db.commit()
                db.refresh(user)
                return user
        except Exception as e:
            logger.error(f"更新用户失败: {e}")
            raise ValidationError(f"更新用户失败: {e}")
    
    def delete_user(self, user_id: str) -> bool:
        """删除用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            是否删除成功
        """
        try:
            with self._db_manager.get_db_session() as db:
                user = db.query(User).filter(User.id == user_id).first()
                if not user:
                    return False
                
                db.delete(user)
                db.commit()
                return True
        except Exception as e:
            logger.error(f"删除用户失败: {e}")
            return False
    
    def get_users(
        self, 
        status: Optional[UserStatus] = None,
        role: Optional[UserRole] = None,
        limit: int = 100, 
        offset: int = 0
    ) -> List[User]:
        """获取用户列表
        
        Args:
            status: 状态过滤
            role: 角色过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            用户列表
        """
        try:
            with self._db_manager.get_db_session() as db:
                query = db.query(User)
                
                if status:
                    query = query.filter(User.status == status)
                
                if role:
                    query = query.filter(User.role == role)
                
                return query.offset(offset).limit(limit).all()
        except Exception as e:
            logger.error(f"获取用户列表失败: {e}")
            return []
    
    def search_users(self, query: str, limit: int = 100) -> List[User]:
        """搜索用户
        
        Args:
            query: 搜索关键词
            limit: 限制数量
            
        Returns:
            用户列表
        """
        try:
            with self._db_manager.get_db_session() as db:
                return db.query(User).filter(
                    or_(
                        User.username.contains(query),
                        User.email.contains(query),
                        User.full_name.contains(query)
                    )
                ).limit(limit).all()
        except Exception as e:
            logger.error(f"搜索用户失败: {e}")
            return []
    
    def create_user_session(self, session_data: Dict[str, Any]) -> UserSession:
        """创建用户会话
        
        Args:
            session_data: 会话数据
            
        Returns:
            创建的用户会话
        """
        try:
            with self._db_manager.get_db_session() as db:
                session = UserSession(**session_data)
                db.add(session)
                db.commit()
                db.refresh(session)
                return session
        except Exception as e:
            logger.error(f"创建用户会话失败: {e}")
            raise ValidationError(f"创建用户会话失败: {e}")
    
    def get_user_session_by_token(self, session_token: str) -> Optional[UserSession]:
        """根据令牌获取用户会话
        
        Args:
            session_token: 会话令牌
            
        Returns:
            用户会话，如果不存在则返回None
        """
        try:
            with self._db_manager.get_db_session() as db:
                return db.query(UserSession).filter(
                    UserSession.session_token == session_token
                ).first()
        except Exception as e:
            logger.error(f"根据令牌获取用户会话失败: {e}")
            return None
    
    def get_user_sessions(self, user_id: str, is_active: Optional[bool] = None) -> List[UserSession]:
        """获取用户的所有会话
        
        Args:
            user_id: 用户ID
            is_active: 是否活跃
            
        Returns:
            用户会话列表
        """
        try:
            with self._db_manager.get_db_session() as db:
                query = db.query(UserSession).filter(UserSession.user_id == user_id)
                
                if is_active is not None:
                    query = query.filter(UserSession.is_active == is_active)
                
                return query.order_by(UserSession.created_at.desc()).all()
        except Exception as e:
            logger.error(f"获取用户会话失败: {e}")
            return []
    
    def update_session_activity(self, session_token: str) -> Optional[UserSession]:
        """更新会话活动时间
        
        Args:
            session_token: 会话令牌
            
        Returns:
            更新后的用户会话，如果不存在则返回None
        """
        try:
            with self._db_manager.get_db_session() as db:
                session = db.query(UserSession).filter(
                    UserSession.session_token == session_token
                ).first()
                
                if session:
                    session.last_activity = datetime.utcnow()
                    db.commit()
                    db.refresh(session)
                
                return session
        except Exception as e:
            logger.error(f"更新会话活动时间失败: {e}")
            return None
    
    def invalidate_user_session(self, session_token: str) -> bool:
        """使用户会话失效
        
        Args:
            session_token: 会话令牌
            
        Returns:
            是否成功
        """
        try:
            with self._db_manager.get_db_session() as db:
                session = db.query(UserSession).filter(
                    UserSession.session_token == session_token
                ).first()
                
                if session:
                    session.is_active = False
                    db.commit()
                    return True
                
                return False
        except Exception as e:
            logger.error(f"使用户会话失效失败: {e}")
            return False
    
    def invalidate_all_user_sessions(self, user_id: str) -> int:
        """使用户的所有会话失效
        
        Args:
            user_id: 用户ID
            
        Returns:
            失效的会话数量
        """
        try:
            with self._db_manager.get_db_session() as db:
                count = db.query(UserSession).filter(
                    UserSession.user_id == user_id,
                    UserSession.is_active == True
                ).update({'is_active': False})
                
                db.commit()
                return count
        except Exception as e:
            logger.error(f"使用户所有会话失效失败: {e}")
            return 0
    
    def cleanup_expired_sessions(self) -> int:
        """清理过期会话
        
        Returns:
            清理的会话数量
        """
        try:
            with self._db_manager.get_db_session() as db:
                # 清理30天前的非活跃会话
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                count = db.query(UserSession).filter(
                    UserSession.last_activity < cutoff_date
                ).delete()
                
                db.commit()
                return count
        except Exception as e:
            logger.error(f"清理过期会话失败: {e}")
            return 0


# 全局用户仓库实例
_global_user_repository = UserRepository()


def get_user_repository() -> UserRepository:
    """获取全局用户仓库实例
    
    Returns:
        UserRepository: 用户仓库实例
    """
    return _global_user_repository
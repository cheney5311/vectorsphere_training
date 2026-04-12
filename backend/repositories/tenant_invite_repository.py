"""租户邀请数据访问层

提供租户邀请相关的数据库访问功能。
"""

import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from backend.core.exceptions import ValidationError, DatabaseError
from backend.schemas.base_models import TenantInvite

logger = logging.getLogger(__name__)


class TenantInviteRepository:
    """租户邀请数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化邀请仓库
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._invites: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("无法导入数据库服务，回退到内存存储模式")
                self._use_memory_storage = True
                self._invites: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, invite_data: Dict[str, Any]) -> TenantInvite:
        """创建邀请
        
        Args:
            invite_data: 邀请数据
            
        Returns:
            创建的邀请
        """
        try:
            invite_id = invite_data['id']
            
            if self._use_memory_storage:
                self._invites[invite_id] = invite_data
                return invite_data
            
            with self._db_manager.get_db_session() as db:
                invite = TenantInvite(
                    id=invite_id,
                    tenant_id=invite_data['tenant_id'],
                    email=invite_data['email'],
                    role=invite_data.get('role', 'member'),
                    invite_code=invite_data['invite_code'],
                    invited_by=invite_data['invited_by'],
                    status=invite_data.get('status', 'pending'),
                    expires_at=invite_data['expires_at'],
                    message=invite_data.get('message')
                )
                db.add(invite)
                db.commit()
                db.refresh(invite)
                return invite
                
        except Exception as e:
            logger.error(f"创建邀请失败: {e}")
            raise DatabaseError(f"创建邀请失败: {e}", operation="create_invite")
    
    def get_by_id(self, invite_id: str) -> Optional[TenantInvite]:
        """根据ID获取邀请
        
        Args:
            invite_id: 邀请ID
            
        Returns:
            邀请对象
        """
        try:
            if self._use_memory_storage:
                return self._invites.get(invite_id)
            
            with self._db_manager.get_db_session() as db:
                return db.query(TenantInvite).filter(
                    TenantInvite.id == invite_id
                ).first()
                
        except Exception as e:
            logger.error(f"获取邀请失败: {e}")
            return None
    
    def get_by_code(self, invite_code: str) -> Optional[TenantInvite]:
        """根据邀请码获取邀请
        
        Args:
            invite_code: 邀请码
            
        Returns:
            邀请对象
        """
        try:
            if self._use_memory_storage:
                for invite in self._invites.values():
                    if invite.get('invite_code') == invite_code:
                        return invite
                return None
            
            with self._db_manager.get_db_session() as db:
                return db.query(TenantInvite).filter(
                    TenantInvite.invite_code == invite_code
                ).first()
                
        except Exception as e:
            logger.error(f"根据邀请码获取邀请失败: {e}")
            return None
    
    def get_pending_by_email(self, tenant_id: str, email: str) -> Optional[TenantInvite]:
        """获取待处理的邀请
        
        Args:
            tenant_id: 租户ID
            email: 邮箱
            
        Returns:
            邀请对象
        """
        try:
            if self._use_memory_storage:
                for invite in self._invites.values():
                    if (invite.get('tenant_id') == tenant_id and
                        invite.get('email') == email and
                        invite.get('status') == 'pending'):
                        return invite
                return None
            
            with self._db_manager.get_db_session() as db:
                return db.query(TenantInvite).filter(
                    TenantInvite.tenant_id == tenant_id,
                    TenantInvite.email == email,
                    TenantInvite.status == 'pending'
                ).first()
                
        except Exception as e:
            logger.error(f"获取待处理邀请失败: {e}")
            return None
    
    def update_status(self, invite_id: str, status: str, 
                     accepted_by: Optional[str] = None) -> bool:
        """更新邀请状态
        
        Args:
            invite_id: 邀请ID
            status: 新状态
            accepted_by: 接受者用户ID
            
        Returns:
            是否更新成功
        """
        try:
            now = datetime.utcnow()
            
            if self._use_memory_storage:
                if invite_id in self._invites:
                    self._invites[invite_id]['status'] = status
                    if status == 'accepted':
                        self._invites[invite_id]['accepted_at'] = now
                        self._invites[invite_id]['accepted_by'] = accepted_by
                    return True
                return False
            
            with self._db_manager.get_db_session() as db:
                invite = db.query(TenantInvite).filter(
                    TenantInvite.id == invite_id
                ).first()
                
                if invite:
                    invite.status = status
                    if status == 'accepted':
                        invite.accepted_at = now
                        invite.accepted_by = accepted_by
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"更新邀请状态失败: {e}")
            return False
    
    def delete(self, invite_id: str) -> bool:
        """删除邀请
        
        Args:
            invite_id: 邀请ID
            
        Returns:
            是否删除成功
        """
        try:
            if self._use_memory_storage:
                if invite_id in self._invites:
                    del self._invites[invite_id]
                    return True
                return False
            
            with self._db_manager.get_db_session() as db:
                invite = db.query(TenantInvite).filter(
                    TenantInvite.id == invite_id
                ).first()
                
                if invite:
                    db.delete(invite)
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"删除邀请失败: {e}")
            return False
    
    def list_by_tenant(self, tenant_id: str, status: Optional[str] = None,
                      limit: int = 100, offset: int = 0) -> Tuple[List[TenantInvite], int]:
        """获取租户的邀请列表
        
        Args:
            tenant_id: 租户ID
            status: 状态过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            (邀请列表, 总数)
        """
        try:
            if self._use_memory_storage:
                invites = [i for i in self._invites.values() 
                          if i.get('tenant_id') == tenant_id]
                if status:
                    invites = [i for i in invites if i.get('status') == status]
                total = len(invites)
                return invites[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(TenantInvite).filter(
                    TenantInvite.tenant_id == tenant_id
                )
                
                if status:
                    query = query.filter(TenantInvite.status == status)
                
                total = query.count()
                invites = query.order_by(
                    TenantInvite.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return invites, total
                
        except Exception as e:
            logger.error(f"获取邀请列表失败: {e}")
            return [], 0
    
    def cleanup_expired(self) -> int:
        """清理过期邀请
        
        Returns:
            清理的数量
        """
        try:
            now = datetime.utcnow()
            
            if self._use_memory_storage:
                expired_ids = [
                    invite_id for invite_id, invite in self._invites.items()
                    if (invite.get('status') == 'pending' and 
                        invite.get('expires_at') and 
                        invite.get('expires_at') < now)
                ]
                for invite_id in expired_ids:
                    self._invites[invite_id]['status'] = 'expired'
                return len(expired_ids)
            
            with self._db_manager.get_db_session() as db:
                count = db.query(TenantInvite).filter(
                    TenantInvite.status == 'pending',
                    TenantInvite.expires_at < now
                ).update({'status': 'expired'})
                db.commit()
                return count
                
        except Exception as e:
            logger.error(f"清理过期邀请失败: {e}")
            return 0
    
    def delete_all_by_tenant(self, tenant_id: str) -> int:
        """删除租户的所有邀请
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            删除的数量
        """
        try:
            if self._use_memory_storage:
                invite_ids = [
                    invite_id for invite_id, invite in self._invites.items()
                    if invite.get('tenant_id') == tenant_id
                ]
                for invite_id in invite_ids:
                    del self._invites[invite_id]
                return len(invite_ids)
            
            with self._db_manager.get_db_session() as db:
                count = db.query(TenantInvite).filter(
                    TenantInvite.tenant_id == tenant_id
                ).delete()
                db.commit()
                return count
                
        except Exception as e:
            logger.error(f"删除租户邀请失败: {e}")
            return 0


# 全局实例
_invite_repository: Optional[TenantInviteRepository] = None


def get_tenant_invite_repository(use_memory: bool = False) -> TenantInviteRepository:
    """获取邀请仓库实例"""
    global _invite_repository
    if _invite_repository is None:
        _invite_repository = TenantInviteRepository(use_memory_storage=use_memory)
    return _invite_repository



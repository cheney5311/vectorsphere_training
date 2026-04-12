"""租户数据访问层

提供租户和租户用户相关的数据库访问功能。
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from backend.core.exceptions import ValidationError, DatabaseError
from backend.schemas.base_models import Tenant, TenantUser, TenantQuota

logger = logging.getLogger(__name__)


class TenantRepository:
    """租户数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化租户仓库
        
        Args:
            use_memory_storage: 是否使用内存存储，默认False使用数据库
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._tenants: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("无法导入数据库服务，回退到内存存储模式")
                self._use_memory_storage = True
                self._tenants: Dict[str, Dict] = {}
                self._db_manager = None
    
    # ========== 租户 CRUD ==========
    
    def create(self, tenant_data: Dict[str, Any]) -> Tenant:
        """创建租户
        
        Args:
            tenant_data: 租户数据
            
        Returns:
            创建的租户
        """
        try:
            if self._use_memory_storage:
                self._tenants[tenant_data['id']] = tenant_data
                return tenant_data
            
            with self._db_manager.get_db_session() as db:
                # 处理 settings 字段
                settings = tenant_data.get('settings', {})
                if isinstance(settings, dict):
                    settings = json.dumps(settings)
                
                tenant = Tenant(
                    id=tenant_data.get('id'),
                    name=tenant_data['name'],
                    display_name=tenant_data.get('display_name', tenant_data['name']),
                    description=tenant_data.get('description', ''),
                    status=tenant_data.get('status', 'active'),
                    settings=settings,
                    creator_user_id=tenant_data['creator_user_id']
                )
                db.add(tenant)
                db.commit()
                db.refresh(tenant)
                return tenant
                
        except Exception as e:
            logger.error(f"创建租户失败: {e}")
            raise DatabaseError(f"创建租户失败: {e}", operation="create_tenant")
    
    def get_by_id(self, tenant_id: str) -> Optional[Tenant]:
        """根据ID获取租户
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            租户对象，如果不存在则返回None
        """
        try:
            if self._use_memory_storage:
                return self._tenants.get(tenant_id)
            
            with self._db_manager.get_db_session() as db:
                return db.query(Tenant).filter(Tenant.id == tenant_id).first()
                
        except Exception as e:
            logger.error(f"获取租户失败: {e}")
            return None
    
    def get_by_name(self, name: str) -> Optional[Tenant]:
        """根据名称获取租户
        
        Args:
            name: 租户名称
            
        Returns:
            租户对象，如果不存在则返回None
        """
        try:
            if self._use_memory_storage:
                for tenant in self._tenants.values():
                    if tenant.get('name') == name:
                        return tenant
                return None
            
            with self._db_manager.get_db_session() as db:
                return db.query(Tenant).filter(Tenant.name == name).first()
                
        except Exception as e:
            logger.error(f"根据名称获取租户失败: {e}")
            return None
    
    def update(self, tenant_id: str, update_data: Dict[str, Any]) -> Optional[Tenant]:
        """更新租户
        
        Args:
            tenant_id: 租户ID
            update_data: 更新数据
            
        Returns:
            更新后的租户对象
        """
        try:
            if self._use_memory_storage:
                if tenant_id in self._tenants:
                    self._tenants[tenant_id].update(update_data)
                    self._tenants[tenant_id]['updated_at'] = datetime.utcnow()
                    return self._tenants[tenant_id]
                return None
            
            with self._db_manager.get_db_session() as db:
                tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
                if not tenant:
                    return None
                
                for key, value in update_data.items():
                    if key == 'settings' and isinstance(value, dict):
                        value = json.dumps(value)
                    if hasattr(tenant, key):
                        setattr(tenant, key, value)
                
                tenant.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(tenant)
                return tenant
                
        except Exception as e:
            logger.error(f"更新租户失败: {e}")
            raise DatabaseError(f"更新租户失败: {e}", operation="update_tenant")
    
    def delete(self, tenant_id: str) -> bool:
        """删除租户（物理删除）
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            是否删除成功
        """
        try:
            if self._use_memory_storage:
                if tenant_id in self._tenants:
                    del self._tenants[tenant_id]
                    return True
                return False
            
            with self._db_manager.get_db_session() as db:
                tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
                if not tenant:
                    return False
                
                db.delete(tenant)
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"删除租户失败: {e}")
            return False
    
    def list_all(self, status: Optional[str] = None, 
                limit: int = 100, offset: int = 0) -> Tuple[List[Tenant], int]:
        """获取租户列表
        
        Args:
            status: 状态过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            (租户列表, 总数)
        """
        try:
            if self._use_memory_storage:
                tenants = list(self._tenants.values())
                if status:
                    tenants = [t for t in tenants if t.get('status') == status]
                total = len(tenants)
                return tenants[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(Tenant)
                if status:
                    query = query.filter(Tenant.status == status)
                
                total = query.count()
                tenants = query.order_by(Tenant.created_at.desc()).offset(offset).limit(limit).all()
                return tenants, total
                
        except Exception as e:
            logger.error(f"获取租户列表失败: {e}")
            return [], 0
    
    def name_exists(self, name: str, exclude_id: Optional[str] = None) -> bool:
        """检查租户名称是否存在
        
        Args:
            name: 租户名称
            exclude_id: 排除的租户ID
            
        Returns:
            是否存在
        """
        try:
            if self._use_memory_storage:
                for tid, tenant in self._tenants.items():
                    if tenant.get('name') == name and tid != exclude_id:
                        return True
                return False
            
            with self._db_manager.get_db_session() as db:
                query = db.query(Tenant).filter(Tenant.name == name)
                if exclude_id:
                    query = query.filter(Tenant.id != exclude_id)
                return query.first() is not None
                
        except Exception as e:
            logger.error(f"检查租户名称失败: {e}")
            return False


class TenantUserRepository:
    """租户用户数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化租户用户仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._tenant_users: Dict[str, List[Dict]] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("无法导入数据库服务，回退到内存存储模式")
                self._use_memory_storage = True
                self._tenant_users: Dict[str, List[Dict]] = {}
                self._db_manager = None
    
    def create(self, tenant_user_data: Dict[str, Any]) -> TenantUser:
        """创建租户用户关联
        
        Args:
            tenant_user_data: 租户用户数据
            
        Returns:
            创建的租户用户
        """
        try:
            tenant_id = tenant_user_data['tenant_id']
            
            if self._use_memory_storage:
                if tenant_id not in self._tenant_users:
                    self._tenant_users[tenant_id] = []
                self._tenant_users[tenant_id].append(tenant_user_data)
                return tenant_user_data
            
            with self._db_manager.get_db_session() as db:
                tenant_user = TenantUser(
                    tenant_id=tenant_id,
                    user_id=tenant_user_data['user_id'],
                    role=tenant_user_data.get('role', 'member'),
                    is_active=tenant_user_data.get('is_active', True)
                )
                db.add(tenant_user)
                db.commit()
                db.refresh(tenant_user)
                return tenant_user
                
        except Exception as e:
            logger.error(f"创建租户用户失败: {e}")
            raise DatabaseError(f"创建租户用户失败: {e}", operation="create_tenant_user")
    
    def get_by_tenant_and_user(self, tenant_id: str, user_id: str) -> Optional[TenantUser]:
        """获取租户用户关联
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            租户用户对象
        """
        try:
            if self._use_memory_storage:
                if tenant_id in self._tenant_users:
                    for tu in self._tenant_users[tenant_id]:
                        if tu.get('user_id') == user_id:
                            return tu
                return None
            
            with self._db_manager.get_db_session() as db:
                return db.query(TenantUser).filter(
                    TenantUser.tenant_id == tenant_id,
                    TenantUser.user_id == user_id
                ).first()
                
        except Exception as e:
            logger.error(f"获取租户用户失败: {e}")
            return None
    
    def get_user_role(self, tenant_id: str, user_id: str) -> Optional[str]:
        """获取用户在租户中的角色
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            角色名称
        """
        tu = self.get_by_tenant_and_user(tenant_id, user_id)
        if tu:
            return tu.role if hasattr(tu, 'role') else tu.get('role')
        return None
    
    def update_role(self, tenant_id: str, user_id: str, new_role: str) -> bool:
        """更新用户角色
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            new_role: 新角色
            
        Returns:
            是否更新成功
        """
        try:
            if self._use_memory_storage:
                if tenant_id in self._tenant_users:
                    for tu in self._tenant_users[tenant_id]:
                        if tu.get('user_id') == user_id:
                            tu['role'] = new_role
                            return True
                return False
            
            with self._db_manager.get_db_session() as db:
                tenant_user = db.query(TenantUser).filter(
                    TenantUser.tenant_id == tenant_id,
                    TenantUser.user_id == user_id
                ).first()
                
                if tenant_user:
                    tenant_user.role = new_role
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"更新用户角色失败: {e}")
            return False
    
    def delete(self, tenant_id: str, user_id: str) -> bool:
        """删除租户用户关联
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            是否删除成功
        """
        try:
            if self._use_memory_storage:
                if tenant_id in self._tenant_users:
                    self._tenant_users[tenant_id] = [
                        tu for tu in self._tenant_users[tenant_id]
                        if tu.get('user_id') != user_id
                    ]
                    return True
                return False
            
            with self._db_manager.get_db_session() as db:
                tenant_user = db.query(TenantUser).filter(
                    TenantUser.tenant_id == tenant_id,
                    TenantUser.user_id == user_id
                ).first()
                
                if tenant_user:
                    db.delete(tenant_user)
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"删除租户用户失败: {e}")
            return False
    
    def list_by_tenant(self, tenant_id: str, role: Optional[str] = None,
                      is_active: Optional[bool] = None,
                      limit: int = 100, offset: int = 0) -> Tuple[List[TenantUser], int]:
        """获取租户的用户列表
        
        Args:
            tenant_id: 租户ID
            role: 角色过滤
            is_active: 是否活跃过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            (用户列表, 总数)
        """
        try:
            if self._use_memory_storage:
                users = self._tenant_users.get(tenant_id, [])
                if role:
                    users = [u for u in users if u.get('role') == role]
                if is_active is not None:
                    users = [u for u in users if u.get('is_active') == is_active]
                total = len(users)
                return users[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(TenantUser).filter(TenantUser.tenant_id == tenant_id)
                
                if role:
                    query = query.filter(TenantUser.role == role)
                if is_active is not None:
                    query = query.filter(TenantUser.is_active == is_active)
                
                total = query.count()
                users = query.order_by(TenantUser.created_at.desc()).offset(offset).limit(limit).all()
                return users, total
                
        except Exception as e:
            logger.error(f"获取租户用户列表失败: {e}")
            return [], 0
    
    def list_by_user(self, user_id: str) -> List[str]:
        """获取用户关联的租户ID列表
        
        Args:
            user_id: 用户ID
            
        Returns:
            租户ID列表
        """
        try:
            if self._use_memory_storage:
                tenant_ids = []
                for tid, users in self._tenant_users.items():
                    for u in users:
                        if u.get('user_id') == user_id and u.get('is_active', True):
                            tenant_ids.append(tid)
                            break
                return tenant_ids
            
            with self._db_manager.get_db_session() as db:
                tenant_users = db.query(TenantUser).filter(
                    TenantUser.user_id == user_id,
                    TenantUser.is_active == True
                ).all()
                return [str(tu.tenant_id) for tu in tenant_users]
                
        except Exception as e:
            logger.error(f"获取用户租户列表失败: {e}")
            return []
    
    def count_by_tenant(self, tenant_id: str, is_active: bool = True) -> int:
        """获取租户用户数量
        
        Args:
            tenant_id: 租户ID
            is_active: 是否只计算活跃用户
            
        Returns:
            用户数量
        """
        try:
            if self._use_memory_storage:
                users = self._tenant_users.get(tenant_id, [])
                if is_active:
                    users = [u for u in users if u.get('is_active', True)]
                return len(users)
            
            with self._db_manager.get_db_session() as db:
                query = db.query(TenantUser).filter(TenantUser.tenant_id == tenant_id)
                if is_active:
                    query = query.filter(TenantUser.is_active == True)
                return query.count()
                
        except Exception as e:
            logger.error(f"获取租户用户数量失败: {e}")
            return 0
    
    def count_by_role(self, tenant_id: str, role: str) -> int:
        """获取特定角色的用户数量
        
        Args:
            tenant_id: 租户ID
            role: 角色
            
        Returns:
            用户数量
        """
        try:
            if self._use_memory_storage:
                users = self._tenant_users.get(tenant_id, [])
                return sum(1 for u in users if u.get('role') == role and u.get('is_active', True))
            
            with self._db_manager.get_db_session() as db:
                return db.query(TenantUser).filter(
                    TenantUser.tenant_id == tenant_id,
                    TenantUser.role == role,
                    TenantUser.is_active == True
                ).count()
                
        except Exception as e:
            logger.error(f"获取角色用户数量失败: {e}")
            return 0
    
    def is_user_in_tenant(self, tenant_id: str, user_id: str) -> bool:
        """检查用户是否在租户中
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            是否存在
        """
        tu = self.get_by_tenant_and_user(tenant_id, user_id)
        if tu:
            is_active = tu.is_active if hasattr(tu, 'is_active') else tu.get('is_active', True)
            return is_active
        return False
    
    def delete_all_by_tenant(self, tenant_id: str) -> int:
        """删除租户的所有用户关联
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            删除的数量
        """
        try:
            if self._use_memory_storage:
                count = len(self._tenant_users.get(tenant_id, []))
                self._tenant_users.pop(tenant_id, None)
                return count
            
            with self._db_manager.get_db_session() as db:
                count = db.query(TenantUser).filter(
                    TenantUser.tenant_id == tenant_id
                ).delete()
                db.commit()
                return count
                
        except Exception as e:
            logger.error(f"删除租户用户失败: {e}")
            return 0


class TenantQuotaRepository:
    """租户配额数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化租户配额仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._quotas: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("无法导入数据库服务，回退到内存存储模式")
                self._use_memory_storage = True
                self._quotas: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, quota_data: Dict[str, Any]) -> TenantQuota:
        """创建租户配额
        
        Args:
            quota_data: 配额数据
            
        Returns:
            创建的配额
        """
        try:
            tenant_id = quota_data['tenant_id']
            
            if self._use_memory_storage:
                self._quotas[tenant_id] = quota_data
                return quota_data
            
            with self._db_manager.get_db_session() as db:
                quota = TenantQuota(**quota_data)
                db.add(quota)
                db.commit()
                db.refresh(quota)
                return quota
                
        except Exception as e:
            logger.error(f"创建租户配额失败: {e}")
            raise DatabaseError(f"创建租户配额失败: {e}", operation="create_quota")
    
    def get_by_tenant(self, tenant_id: str) -> Optional[TenantQuota]:
        """获取租户配额
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            配额对象
        """
        try:
            if self._use_memory_storage:
                return self._quotas.get(tenant_id)
            
            with self._db_manager.get_db_session() as db:
                return db.query(TenantQuota).filter(
                    TenantQuota.tenant_id == tenant_id
                ).first()
                
        except Exception as e:
            logger.error(f"获取租户配额失败: {e}")
            return None
    
    def update(self, tenant_id: str, update_data: Dict[str, Any]) -> Optional[TenantQuota]:
        """更新租户配额
        
        Args:
            tenant_id: 租户ID
            update_data: 更新数据
            
        Returns:
            更新后的配额
        """
        try:
            if self._use_memory_storage:
                if tenant_id in self._quotas:
                    self._quotas[tenant_id].update(update_data)
                    return self._quotas[tenant_id]
                return None
            
            with self._db_manager.get_db_session() as db:
                quota = db.query(TenantQuota).filter(
                    TenantQuota.tenant_id == tenant_id
                ).first()
                
                if quota:
                    for key, value in update_data.items():
                        if hasattr(quota, key):
                            setattr(quota, key, value)
                    quota.updated_at = datetime.utcnow()
                    db.commit()
                    db.refresh(quota)
                    return quota
                return None
                
        except Exception as e:
            logger.error(f"更新租户配额失败: {e}")
            raise DatabaseError(f"更新租户配额失败: {e}", operation="update_quota")
    
    def delete(self, tenant_id: str) -> bool:
        """删除租户配额
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            是否删除成功
        """
        try:
            if self._use_memory_storage:
                if tenant_id in self._quotas:
                    del self._quotas[tenant_id]
                    return True
                return False
            
            with self._db_manager.get_db_session() as db:
                quota = db.query(TenantQuota).filter(
                    TenantQuota.tenant_id == tenant_id
                ).first()
                
                if quota:
                    db.delete(quota)
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"删除租户配额失败: {e}")
            return False


# 全局实例
_tenant_repository: Optional[TenantRepository] = None
_tenant_user_repository: Optional[TenantUserRepository] = None
_tenant_quota_repository: Optional[TenantQuotaRepository] = None


def get_tenant_repository(use_memory: bool = False) -> TenantRepository:
    """获取租户仓库实例"""
    global _tenant_repository
    if _tenant_repository is None:
        _tenant_repository = TenantRepository(use_memory_storage=use_memory)
    return _tenant_repository


def get_tenant_user_repository(use_memory: bool = False) -> TenantUserRepository:
    """获取租户用户仓库实例"""
    global _tenant_user_repository
    if _tenant_user_repository is None:
        _tenant_user_repository = TenantUserRepository(use_memory_storage=use_memory)
    return _tenant_user_repository


def get_tenant_quota_repository(use_memory: bool = False) -> TenantQuotaRepository:
    """获取租户配额仓库实例"""
    global _tenant_quota_repository
    if _tenant_quota_repository is None:
        _tenant_quota_repository = TenantQuotaRepository(use_memory_storage=use_memory)
    return _tenant_quota_repository



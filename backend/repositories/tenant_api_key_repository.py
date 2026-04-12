"""租户API密钥数据访问层

提供租户API密钥相关的数据库访问功能。
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from backend.core.exceptions import ValidationError, DatabaseError
from backend.schemas.base_models import TenantApiKey

logger = logging.getLogger(__name__)


class TenantApiKeyRepository:
    """租户API密钥数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化API密钥仓库
        
        Args:
            use_memory_storage: 是否使用内存存储，默认False使用数据库
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._api_keys: Dict[str, List[Dict]] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("无法导入数据库服务，回退到内存存储模式")
                self._use_memory_storage = True
                self._api_keys: Dict[str, List[Dict]] = {}
                self._db_manager = None
    
    def create(self, api_key_data: Dict[str, Any]) -> TenantApiKey:
        """创建API密钥
        
        Args:
            api_key_data: API密钥数据
            
        Returns:
            创建的API密钥
        """
        try:
            tenant_id = api_key_data['tenant_id']
            
            if self._use_memory_storage:
                if tenant_id not in self._api_keys:
                    self._api_keys[tenant_id] = []
                self._api_keys[tenant_id].append(api_key_data)
                return api_key_data
            
            with self._db_manager.get_db_session() as db:
                # 处理 JSON 字段
                permissions = api_key_data.get('permissions', [])
                if isinstance(permissions, list):
                    permissions = json.dumps(permissions)
                
                scopes = api_key_data.get('scopes', [])
                if isinstance(scopes, list):
                    scopes = json.dumps(scopes)
                
                api_key = TenantApiKey(
                    id=api_key_data.get('id'),
                    tenant_id=tenant_id,
                    name=api_key_data['name'],
                    description=api_key_data.get('description', ''),
                    key_hash=api_key_data['key_hash'],
                    key_prefix=api_key_data['key_prefix'],
                    permissions=permissions,
                    scopes=scopes,
                    rate_limit=api_key_data.get('rate_limit', 1000),
                    created_by=api_key_data['created_by'],
                    is_active=api_key_data.get('is_active', True),
                    expires_at=api_key_data.get('expires_at')
                )
                db.add(api_key)
                db.commit()
                db.refresh(api_key)
                return api_key
                
        except Exception as e:
            logger.error(f"创建API密钥失败: {e}")
            raise DatabaseError(f"创建API密钥失败: {e}", operation="create_api_key")
    
    def get_by_id(self, key_id: str) -> Optional[TenantApiKey]:
        """根据ID获取API密钥
        
        Args:
            key_id: 密钥ID
            
        Returns:
            API密钥对象
        """
        try:
            if self._use_memory_storage:
                for keys in self._api_keys.values():
                    for key in keys:
                        if key.get('id') == key_id:
                            return key
                return None
            
            with self._db_manager.get_db_session() as db:
                return db.query(TenantApiKey).filter(
                    TenantApiKey.id == key_id
                ).first()
                
        except Exception as e:
            logger.error(f"获取API密钥失败: {e}")
            return None
    
    def get_by_tenant_and_id(self, tenant_id: str, key_id: str) -> Optional[TenantApiKey]:
        """根据租户ID和密钥ID获取API密钥
        
        Args:
            tenant_id: 租户ID
            key_id: 密钥ID
            
        Returns:
            API密钥对象
        """
        try:
            if self._use_memory_storage:
                if tenant_id in self._api_keys:
                    for key in self._api_keys[tenant_id]:
                        if key.get('id') == key_id:
                            return key
                return None
            
            with self._db_manager.get_db_session() as db:
                return db.query(TenantApiKey).filter(
                    TenantApiKey.tenant_id == tenant_id,
                    TenantApiKey.id == key_id
                ).first()
                
        except Exception as e:
            logger.error(f"获取API密钥失败: {e}")
            return None
    
    def get_by_hash(self, key_hash: str, key_prefix: str) -> Optional[TenantApiKey]:
        """根据哈希值获取API密钥
        
        Args:
            key_hash: 密钥哈希
            key_prefix: 密钥前缀
            
        Returns:
            API密钥对象
        """
        try:
            if self._use_memory_storage:
                for keys in self._api_keys.values():
                    for key in keys:
                        if (key.get('key_hash') == key_hash and 
                            key.get('key_prefix') == key_prefix and
                            key.get('is_active', True)):
                            return key
                return None
            
            with self._db_manager.get_db_session() as db:
                return db.query(TenantApiKey).filter(
                    TenantApiKey.key_hash == key_hash,
                    TenantApiKey.key_prefix == key_prefix,
                    TenantApiKey.is_active == True
                ).first()
                
        except Exception as e:
            logger.error(f"根据哈希获取API密钥失败: {e}")
            return None
    
    def update(self, key_id: str, update_data: Dict[str, Any]) -> Optional[TenantApiKey]:
        """更新API密钥
        
        Args:
            key_id: 密钥ID
            update_data: 更新数据
            
        Returns:
            更新后的API密钥
        """
        try:
            if self._use_memory_storage:
                for keys in self._api_keys.values():
                    for key in keys:
                        if key.get('id') == key_id:
                            key.update(update_data)
                            key['updated_at'] = datetime.utcnow()
                            return key
                return None
            
            with self._db_manager.get_db_session() as db:
                api_key = db.query(TenantApiKey).filter(
                    TenantApiKey.id == key_id
                ).first()
                
                if not api_key:
                    return None
                
                for key, value in update_data.items():
                    # 处理 JSON 字段
                    if key in ('permissions', 'scopes') and isinstance(value, list):
                        value = json.dumps(value)
                    if hasattr(api_key, key):
                        setattr(api_key, key, value)
                
                api_key.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(api_key)
                return api_key
                
        except Exception as e:
            logger.error(f"更新API密钥失败: {e}")
            raise DatabaseError(f"更新API密钥失败: {e}", operation="update_api_key")
    
    def revoke(self, key_id: str, revoked_by: str, reason: Optional[str] = None) -> bool:
        """撤销API密钥
        
        Args:
            key_id: 密钥ID
            revoked_by: 撤销者用户ID
            reason: 撤销原因
            
        Returns:
            是否撤销成功
        """
        try:
            now = datetime.utcnow()
            
            if self._use_memory_storage:
                for keys in self._api_keys.values():
                    for key in keys:
                        if key.get('id') == key_id:
                            key['is_active'] = False
                            key['revoked_at'] = now
                            key['revoked_by'] = revoked_by
                            key['revoke_reason'] = reason
                            key['updated_at'] = now
                            return True
                return False
            
            with self._db_manager.get_db_session() as db:
                api_key = db.query(TenantApiKey).filter(
                    TenantApiKey.id == key_id
                ).first()
                
                if api_key:
                    api_key.is_active = False
                    api_key.revoked_at = now
                    api_key.revoked_by = revoked_by
                    api_key.revoke_reason = reason
                    api_key.updated_at = now
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"撤销API密钥失败: {e}")
            return False
    
    def delete(self, key_id: str) -> bool:
        """删除API密钥
        
        Args:
            key_id: 密钥ID
            
        Returns:
            是否删除成功
        """
        try:
            if self._use_memory_storage:
                for tenant_id, keys in self._api_keys.items():
                    for i, key in enumerate(keys):
                        if key.get('id') == key_id:
                            del self._api_keys[tenant_id][i]
                            return True
                return False
            
            with self._db_manager.get_db_session() as db:
                api_key = db.query(TenantApiKey).filter(
                    TenantApiKey.id == key_id
                ).first()
                
                if api_key:
                    db.delete(api_key)
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"删除API密钥失败: {e}")
            return False
    
    def list_by_tenant(self, tenant_id: str, include_revoked: bool = False,
                      limit: int = 100, offset: int = 0) -> Tuple[List[TenantApiKey], int]:
        """获取租户的API密钥列表
        
        Args:
            tenant_id: 租户ID
            include_revoked: 是否包含已撤销的密钥
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            (API密钥列表, 总数)
        """
        try:
            if self._use_memory_storage:
                keys = self._api_keys.get(tenant_id, [])
                if not include_revoked:
                    keys = [k for k in keys if k.get('is_active', True)]
                total = len(keys)
                return keys[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(TenantApiKey).filter(
                    TenantApiKey.tenant_id == tenant_id
                )
                
                if not include_revoked:
                    query = query.filter(TenantApiKey.is_active == True)
                
                total = query.count()
                keys = query.order_by(
                    TenantApiKey.created_at.desc()
                ).offset(offset).limit(limit).all()
                
                return keys, total
                
        except Exception as e:
            logger.error(f"获取API密钥列表失败: {e}")
            return [], 0
    
    def count_by_tenant(self, tenant_id: str, is_active: bool = True) -> int:
        """获取租户API密钥数量
        
        Args:
            tenant_id: 租户ID
            is_active: 是否只计算活跃密钥
            
        Returns:
            密钥数量
        """
        try:
            if self._use_memory_storage:
                keys = self._api_keys.get(tenant_id, [])
                if is_active:
                    keys = [k for k in keys if k.get('is_active', True)]
                return len(keys)
            
            with self._db_manager.get_db_session() as db:
                query = db.query(TenantApiKey).filter(
                    TenantApiKey.tenant_id == tenant_id
                )
                if is_active:
                    query = query.filter(TenantApiKey.is_active == True)
                return query.count()
                
        except Exception as e:
            logger.error(f"获取API密钥数量失败: {e}")
            return 0
    
    def name_exists(self, tenant_id: str, name: str, 
                   exclude_key_id: Optional[str] = None) -> bool:
        """检查API密钥名称是否存在
        
        Args:
            tenant_id: 租户ID
            name: 密钥名称
            exclude_key_id: 排除的密钥ID
            
        Returns:
            是否存在
        """
        try:
            if self._use_memory_storage:
                if tenant_id in self._api_keys:
                    for key in self._api_keys[tenant_id]:
                        if (key.get('name') == name and 
                            key.get('is_active', True) and
                            key.get('id') != exclude_key_id):
                            return True
                return False
            
            with self._db_manager.get_db_session() as db:
                query = db.query(TenantApiKey).filter(
                    TenantApiKey.tenant_id == tenant_id,
                    TenantApiKey.name == name,
                    TenantApiKey.is_active == True
                )
                if exclude_key_id:
                    query = query.filter(TenantApiKey.id != exclude_key_id)
                return query.first() is not None
                
        except Exception as e:
            logger.error(f"检查API密钥名称失败: {e}")
            return False
    
    def update_usage(self, key_id: str, ip_address: Optional[str] = None) -> bool:
        """更新API密钥使用统计
        
        Args:
            key_id: 密钥ID
            ip_address: 请求IP地址
            
        Returns:
            是否更新成功
        """
        try:
            now = datetime.utcnow()
            
            if self._use_memory_storage:
                for keys in self._api_keys.values():
                    for key in keys:
                        if key.get('id') == key_id:
                            key['last_used_at'] = now
                            key['last_used_ip'] = ip_address
                            key['use_count'] = key.get('use_count', 0) + 1
                            return True
                return False
            
            with self._db_manager.get_db_session() as db:
                api_key = db.query(TenantApiKey).filter(
                    TenantApiKey.id == key_id
                ).first()
                
                if api_key:
                    api_key.last_used_at = now
                    if ip_address:
                        api_key.last_used_ip = ip_address
                    api_key.use_count = (api_key.use_count or 0) + 1
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"更新API密钥使用统计失败: {e}")
            return False
    
    def delete_all_by_tenant(self, tenant_id: str) -> int:
        """删除租户的所有API密钥
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            删除的数量
        """
        try:
            if self._use_memory_storage:
                count = len(self._api_keys.get(tenant_id, []))
                self._api_keys.pop(tenant_id, None)
                return count
            
            with self._db_manager.get_db_session() as db:
                count = db.query(TenantApiKey).filter(
                    TenantApiKey.tenant_id == tenant_id
                ).delete()
                db.commit()
                return count
                
        except Exception as e:
            logger.error(f"删除租户API密钥失败: {e}")
            return 0


# 全局实例
_api_key_repository: Optional[TenantApiKeyRepository] = None


def get_tenant_api_key_repository(use_memory: bool = False) -> TenantApiKeyRepository:
    """获取API密钥仓库实例"""
    global _api_key_repository
    if _api_key_repository is None:
        _api_key_repository = TenantApiKeyRepository(use_memory_storage=use_memory)
    return _api_key_repository


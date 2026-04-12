"""租户服务模块

提供完整的租户管理能力，包括：
- 租户 CRUD 操作
- 租户用户管理
- 资源配额管理
- 资源使用统计
- 租户环境初始化和清理
- 审计日志
"""

import uuid
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict, fields as dataclass_fields
from backend.core.exceptions import ValidationError, AuthorizationError, TenantError

import logging
logger = logging.getLogger(__name__)


# ============================================================================
# 枚举和数据类定义
# ============================================================================

class UserRole(Enum):
    """用户角色枚举"""
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class InviteStatus(Enum):
    """邀请状态枚举"""
    PENDING = "pending"
    ACCEPTED = "accepted"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class AuditAction(Enum):
    """审计操作类型"""
    TENANT_CREATED = "tenant_created"
    TENANT_UPDATED = "tenant_updated"
    TENANT_DELETED = "tenant_deleted"
    TENANT_STATUS_CHANGED = "tenant_status_changed"
    USER_ADDED = "user_added"
    USER_REMOVED = "user_removed"
    USER_ROLE_CHANGED = "user_role_changed"
    QUOTA_UPDATED = "quota_updated"
    SETTINGS_UPDATED = "settings_updated"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"


@dataclass
class TenantQuota:
    """租户配额"""
    max_users: int = 10
    max_training_sessions: int = 5
    max_concurrent_trainings: int = 2
    max_models: int = 20
    max_datasets: int = 50
    storage_limit_gb: int = 100
    compute_hours_monthly: int = 100
    api_requests_daily: int = 10000
    gpu_hours_monthly: int = 50
    
    def to_dict(self) -> Dict[str, int]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TenantQuota':
        valid_fields = {f.name for f in dataclass_fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})


@dataclass
class TenantUsage:
    """租户资源使用情况"""
    current_users: int = 0
    active_training_sessions: int = 0
    total_training_sessions: int = 0
    total_models: int = 0
    total_datasets: int = 0
    storage_used_gb: float = 0.0
    compute_hours_used: float = 0.0
    api_requests_today: int = 0
    gpu_hours_used: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TenantInvite:
    """租户邀请"""
    id: str
    tenant_id: str
    email: str
    role: str
    invite_code: str
    invited_by: str
    status: str = InviteStatus.PENDING.value
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(days=7))
    accepted_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'email': self.email,
            'role': self.role,
            'status': self.status,
            'invited_by': self.invited_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'accepted_at': self.accepted_at.isoformat() if self.accepted_at else None
        }


@dataclass
class TenantApiKeyData:
    """租户 API 密钥数据类（用于内存存储回退）"""
    id: str
    tenant_id: str
    name: str
    key_hash: str
    key_prefix: str
    permissions: List[str]
    created_by: str
    description: str = ""
    scopes: List[str] = field(default_factory=list)
    rate_limit: int = 1000
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: Optional[datetime] = None
    last_used_ip: Optional[str] = None
    use_count: int = 0
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    revoked_by: Optional[str] = None
    revoke_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'name': self.name,
            'description': self.description,
            'key_prefix': self.key_prefix,
            'permissions': self.permissions,
            'scopes': self.scopes,
            'rate_limit': self.rate_limit,
            'is_active': self.is_active,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'last_used_ip': self.last_used_ip,
            'use_count': self.use_count,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'revoked_at': self.revoked_at.isoformat() if self.revoked_at else None,
            'revoked_by': self.revoked_by
        }


@dataclass
class AuditLogEntry:
    """审计日志条目"""
    id: str
    tenant_id: str
    user_id: str
    action: str
    resource_type: str
    resource_id: Optional[str]
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'user_id': self.user_id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


# ============================================================================
# 租户服务类
# ============================================================================

class TenantService:
    """租户服务类
    
    提供完整的多租户管理能力。
    使用 Repository 模式实现数据访问和业务逻辑的解耦。
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化租户服务
        
        Args:
            use_memory_storage: 是否使用内存存储（用于测试）
        """
        self._use_memory_storage = use_memory_storage
        self._tenant_quotas: Dict[str, TenantQuota] = {}  # 租户配额缓存
        self._init_repositories()
        self._init_dependencies()
        self._init_default_quotas()
    
    def _init_repositories(self):
        """初始化数据仓库"""
        from backend.repositories import (
            get_tenant_repository,
            get_tenant_user_repository,
            get_tenant_quota_repository,
            get_tenant_api_key_repository,
            get_tenant_invite_repository,
            get_tenant_audit_log_repository
        )
        
        # 初始化各个 Repository
        self.tenant_repo = get_tenant_repository(self._use_memory_storage)
        self.tenant_user_repo = get_tenant_user_repository(self._use_memory_storage)
        self.quota_repo = get_tenant_quota_repository(self._use_memory_storage)
        self.api_key_repo = get_tenant_api_key_repository(self._use_memory_storage)
        self.invite_repo = get_tenant_invite_repository(self._use_memory_storage)
        self.audit_log_repo = get_tenant_audit_log_repository(self._use_memory_storage)
        
        # 可选的资源 Repository（用于统计）
        self.training_session_repo = None
        self.model_repo = None
        self.dataset_repo = None
        try:
            from backend.repositories.training_session_repository import get_training_session_repository
            self.training_session_repo = get_training_session_repository(self._use_memory_storage)
        except (ImportError, Exception):
            pass
        try:
            from backend.repositories.model_repository import ModelRepository
            self.model_repo = ModelRepository()
        except (ImportError, Exception):
            pass
        try:
            from backend.repositories.dataset_repository import DatasetRepository
            self.dataset_repo = DatasetRepository()
        except (ImportError, Exception):
            pass
    
    def _init_dependencies(self):
        """初始化其他依赖"""
        # Redis 客户端
        try:
            from modules.database import get_redis_client
            self.redis_client = get_redis_client()
        except ImportError:
            logger.warning("Redis client not available")
            self.redis_client = None
        
        # 认证服务
        try:
            from services.auth_service import get_auth_service
            self.auth_service = get_auth_service()
        except ImportError:
            logger.warning("Auth service not available")
            self.auth_service = None
        
        # 加密服务
        try:
            from backend.modules.security.services.encryption_service import EncryptionService
            from config import get_config
            config = get_config()
            encryption_config = {
                'key_storage_path': config.get('security.key_storage_path', 'data/keys'),
                'default_key_id': None
            }
            self.encryption_service = EncryptionService(encryption_config)
        except (ImportError, Exception) as e:
            logger.warning(f"Encryption service not available: {e}")
            self.encryption_service = None
    
    def _init_default_quotas(self):
        """初始化默认配额配置"""
        self.default_quotas = TenantQuota()
        
        # 不同套餐的配额
        self.quota_plans = {
            'free': TenantQuota(
                max_users=3,
                max_training_sessions=2,
                max_concurrent_trainings=1,
                max_models=5,
                max_datasets=10,
                storage_limit_gb=10,
                compute_hours_monthly=10,
                api_requests_daily=1000,
                gpu_hours_monthly=5
            ),
            'basic': TenantQuota(
                max_users=10,
                max_training_sessions=10,
                max_concurrent_trainings=2,
                max_models=20,
                max_datasets=50,
                storage_limit_gb=100,
                compute_hours_monthly=100,
                api_requests_daily=10000,
                gpu_hours_monthly=50
            ),
            'professional': TenantQuota(
                max_users=50,
                max_training_sessions=50,
                max_concurrent_trainings=5,
                max_models=100,
                max_datasets=200,
                storage_limit_gb=500,
                compute_hours_monthly=500,
                api_requests_daily=100000,
                gpu_hours_monthly=200
            ),
            'enterprise': TenantQuota(
                max_users=1000,
                max_training_sessions=1000,
                max_concurrent_trainings=20,
                max_models=1000,
                max_datasets=5000,
                storage_limit_gb=5000,
                compute_hours_monthly=5000,
                api_requests_daily=1000000,
                gpu_hours_monthly=1000
            )
        }
    
    # ========== 租户 CRUD 操作 ==========
    
    def create_tenant(self, tenant_data: Dict[str, Any], creator_user_id: str) -> Dict[str, Any]:
        """创建租户
        
        Args:
            tenant_data: 租户数据，包含 name, display_name, description, settings, plan 等
            creator_user_id: 创建者用户ID
            
        Returns:
            创建的租户信息
        """
        from backend.core.exceptions import ValidationError, TenantError
        
        try:
            # 验证租户数据
            self._validate_tenant_data(tenant_data)
            
            # 检查租户名称是否已存在
            if self._tenant_name_exists(tenant_data['name']):
                raise ValidationError(f"Tenant name '{tenant_data['name']}' already exists")
            
            # 生成租户ID
            tenant_id = str(uuid.uuid4())
            now = datetime.utcnow()
            
            # 创建租户记录
            tenant = {
                'id': tenant_id,
                'name': tenant_data['name'],
                'display_name': tenant_data.get('display_name', tenant_data['name']),
                'description': tenant_data.get('description', ''),
                'status': 'active',
                'plan': tenant_data.get('plan', 'basic'),
                'settings': tenant_data.get('settings', {}),
                'creator_user_id': creator_user_id,
                'created_at': now,
                'updated_at': now,
                'activated_at': now
            }
            
            # 保存租户（通过 Repository）
            self._save_tenant(tenant)
            
            # 设置资源配额
            self._create_tenant_quotas(tenant_id, tenant_data.get('plan', 'basic'))
            
            # 将创建者添加为租户所有者
            self._add_tenant_owner(tenant_id, creator_user_id)
            
            # 初始化租户环境
            self._initialize_tenant_environment(tenant_id)
            
            # 记录审计日志
            self._log_audit(
                tenant_id=tenant_id,
                user_id=creator_user_id,
                action=AuditAction.TENANT_CREATED.value,
                resource_type='tenant',
                resource_id=tenant_id,
                details={'name': tenant_data['name'], 'plan': tenant.get('plan')}
            )
            
            logger.info(f"Tenant created: {tenant_id} by user {creator_user_id}")
            
            return {
                'tenant_id': tenant_id,
                'name': tenant['name'],
                'display_name': tenant['display_name'],
                'description': tenant['description'],
                'status': tenant['status'],
                'plan': tenant['plan'],
                'created_at': tenant['created_at'].isoformat(),
                'settings': tenant['settings']
            }
            
        except (ValidationError, TenantError):
            raise
        except Exception as e:
            logger.error(f"Error creating tenant: {str(e)}")
            raise TenantError(f"Failed to create tenant: {str(e)}")
    
    def get_tenant(self, tenant_id: str, user_id: str) -> Dict[str, Any]:
        """获取租户信息
        
        Args:
            tenant_id: 租户ID
            user_id: 请求用户ID
            
        Returns:
            租户详情
        """
        from backend.core.exceptions import AuthorizationError, TenantError
        
        try:
            # 检查用户权限
            if not self._check_tenant_access(tenant_id, user_id):
                raise AuthorizationError("Access denied to tenant")
            
            tenant = self._get_tenant_by_id(tenant_id)
            if not tenant:
                raise TenantError("Tenant not found")
            
            # 获取统计信息
            stats = self._get_tenant_statistics(tenant_id)
            
            # 获取用户角色
            user_role = self._get_user_role_in_tenant(tenant_id, user_id)
            
            # 获取配额信息
            quota = self._get_tenant_quota_object(tenant_id)
            usage = self._calculate_resource_usage(tenant_id)
            
            return {
                'tenant_id': tenant_id,
                'name': tenant.get('name') or getattr(tenant, 'name', ''),
                'display_name': tenant.get('display_name') or getattr(tenant, 'display_name', ''),
                'description': tenant.get('description') or getattr(tenant, 'description', ''),
                'status': tenant.get('status') or getattr(tenant, 'status', ''),
                'plan': tenant.get('plan', 'basic'),
                'created_at': self._format_datetime(tenant.get('created_at') or getattr(tenant, 'created_at', None)),
                'updated_at': self._format_datetime(tenant.get('updated_at') or getattr(tenant, 'updated_at', None)),
                'settings': tenant.get('settings') or getattr(tenant, 'settings', {}),
                'user_role': user_role,
                'statistics': stats,
                'quota': quota.to_dict() if quota else {},
                'usage': usage.to_dict() if usage else {}
            }
            
        except (AuthorizationError, TenantError):
            raise
        except Exception as e:
            logger.error(f"Error getting tenant: {str(e)}")
            raise TenantError(f"Failed to get tenant: {str(e)}")
    
    def list_tenants(self, user_id: str, page: int = 1, page_size: int = 20, 
                    status: Optional[str] = None) -> Dict[str, Any]:
        """获取用户可访问的租户列表
        
        Args:
            user_id: 用户ID
            page: 页码
            page_size: 每页大小
            status: 状态过滤
            
        Returns:
            租户列表
        """
        from backend.core.exceptions import TenantError
        
        try:
            # 获取用户关联的租户
            tenant_ids = self._get_user_tenant_ids(user_id)
            
            if not tenant_ids:
                return {
                    'tenants': [],
                    'total': 0,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': 0
                }
            
            # 获取租户详情
            all_tenants = []
            for tid in tenant_ids:
                tenant = self._get_tenant_by_id(tid)
                if tenant:
                    # 状态过滤
                    tenant_status = tenant.get('status') or getattr(tenant, 'status', '')
                    if status and tenant_status != status:
                        continue
                    
                    user_role = self._get_user_role_in_tenant(tid, user_id)
                    user_count = self._get_tenant_user_count(tid)
                    
                    all_tenants.append({
                        'tenant_id': tid,
                        'name': tenant.get('name') or getattr(tenant, 'name', ''),
                        'display_name': tenant.get('display_name') or getattr(tenant, 'display_name', ''),
                        'description': tenant.get('description') or getattr(tenant, 'description', ''),
                        'status': tenant_status,
                        'plan': tenant.get('plan', 'basic'),
                        'created_at': self._format_datetime(tenant.get('created_at') or getattr(tenant, 'created_at', None)),
                    'user_role': user_role,
                        'user_count': user_count
                    })
            
            # 分页
            total = len(all_tenants)
            offset = (page - 1) * page_size
            tenants = all_tenants[offset:offset + page_size]
            total_pages = (total + page_size - 1) // page_size
            
            return {
                'tenants': tenants,
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': total_pages
            }
            
        except Exception as e:
            logger.error(f"Error listing tenants: {str(e)}")
            raise TenantError(f"Failed to list tenants: {str(e)}")
    
    def update_tenant(self, tenant_id: str, update_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """更新租户信息
        
        Args:
            tenant_id: 租户ID
            update_data: 更新数据
            user_id: 用户ID
            
        Returns:
            更新后的租户信息
        """
        from backend.core.exceptions import ValidationError, AuthorizationError, TenantError
        
        try:
            # 检查管理员权限
            if not self._check_tenant_admin_access(tenant_id, user_id):
                raise AuthorizationError("Admin access required")
            
            tenant = self._get_tenant_by_id(tenant_id)
            if not tenant:
                raise TenantError("Tenant not found")
            
            # 允许更新的字段
            allowed_fields = ['display_name', 'description', 'settings']
            filtered_data = {k: v for k, v in update_data.items() if k in allowed_fields}
            
            if not filtered_data:
                raise ValidationError("No valid fields to update")
            
            # 更新租户（通过 Repository）
            self._update_tenant(tenant_id, filtered_data)
            
            # 清除缓存
            self._clear_tenant_cache(tenant_id)
            
            # 记录审计日志
            self._log_audit(
                tenant_id=tenant_id,
                user_id=user_id,
                action=AuditAction.TENANT_UPDATED.value,
                resource_type='tenant',
                resource_id=tenant_id,
                details={'updated_fields': list(filtered_data.keys())}
            )
            
            logger.info(f"Tenant updated: {tenant_id} by user {user_id}")
            
            # 返回更新后的租户
            return self.get_tenant(tenant_id, user_id)
            
        except (ValidationError, AuthorizationError, TenantError):
            raise
        except Exception as e:
            logger.error(f"Error updating tenant: {str(e)}")
            raise TenantError(f"Failed to update tenant: {str(e)}")
    
    def update_tenant_status(self, tenant_id: str, status: str, user_id: str) -> Dict[str, Any]:
        """更新租户状态
        
        Args:
            tenant_id: 租户ID
            status: 新状态
            user_id: 用户ID
            
        Returns:
            更新结果
        """
        from backend.core.exceptions import ValidationError, AuthorizationError, TenantError
        from backend.schemas.enums import TenantStatus
        
        try:
            # 检查超级管理员权限
            if not self._check_super_admin_access(user_id):
                raise AuthorizationError("Super admin access required")
            
            # 验证状态值
            try:
                new_status = TenantStatus(status).value
            except ValueError:
                raise ValidationError(f"Invalid status: {status}. Valid values: {[s.value for s in TenantStatus]}")
            
            tenant = self._get_tenant_by_id(tenant_id)
            if not tenant:
                raise TenantError("Tenant not found")
            
            old_status = tenant.get('status') or getattr(tenant, 'status', '')
            now = datetime.utcnow()
            
            # 更新状态
            update_data = {
                'status': new_status,
                'updated_at': now
            }
            
            # 更新相关时间戳
            if new_status == TenantStatus.ACTIVE.value:
                update_data['activated_at'] = now
            elif new_status == TenantStatus.SUSPENDED.value:
                update_data['suspended_at'] = now
            
            # 更新状态（通过 Repository）
            self._update_tenant(tenant_id, update_data)
            
            # 根据状态变化执行相应操作
            if new_status == TenantStatus.SUSPENDED.value:
                self._suspend_tenant_resources(tenant_id)
            elif new_status == TenantStatus.ACTIVE.value and old_status == TenantStatus.SUSPENDED.value:
                self._activate_tenant_resources(tenant_id)
            
            # 清除缓存
            self._clear_tenant_cache(tenant_id)
            
            # 记录审计日志
            self._log_audit(
                tenant_id=tenant_id,
                user_id=user_id,
                action=AuditAction.TENANT_STATUS_CHANGED.value,
                resource_type='tenant',
                resource_id=tenant_id,
                details={'old_status': old_status, 'new_status': new_status}
            )
            
            logger.info(f"Tenant status updated: {tenant_id} from {old_status} to {new_status} by user {user_id}")
            
            return {
                'tenant_id': tenant_id,
                'status': new_status,
                'updated_at': now.isoformat()
            }
            
        except (ValidationError, AuthorizationError, TenantError):
            raise
        except Exception as e:
            logger.error(f"Error updating tenant status: {str(e)}")
            raise TenantError(f"Failed to update tenant status: {str(e)}")
    
    def delete_tenant(self, tenant_id: str, user_id: str, force: bool = False) -> Dict[str, Any]:
        """删除租户
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            force: 是否强制删除（忽略活跃资源检查）
            
        Returns:
            删除结果
        """
        from backend.core.exceptions import ValidationError, AuthorizationError, TenantError
        from backend.schemas.enums import TenantStatus
        
        try:
            # 检查超级管理员权限
            if not self._check_super_admin_access(user_id):
                raise AuthorizationError("Super admin access required")
            
            tenant = self._get_tenant_by_id(tenant_id)
            if not tenant:
                raise TenantError("Tenant not found")
            
            # 检查是否有活跃资源
            if not force and self._has_active_resources(tenant_id):
                raise ValidationError("Cannot delete tenant with active resources. Use force=true to override.")
            
            # 软删除租户
            now = datetime.utcnow()
            update_data = {
                'status': TenantStatus.DELETED.value if hasattr(TenantStatus, 'DELETED') else 'deleted',
                'deleted_at': now,
                'updated_at': now
            }
            
            # 软删除租户（通过 Repository）
            self._update_tenant(tenant_id, update_data)
            
            # 清理租户数据
            self._cleanup_tenant_data(tenant_id)
            
            # 记录审计日志
            self._log_audit(
                tenant_id=tenant_id,
                user_id=user_id,
                action=AuditAction.TENANT_DELETED.value,
                resource_type='tenant',
                resource_id=tenant_id,
                details={'force': force}
            )
            
            logger.info(f"Tenant deleted: {tenant_id} by user {user_id}")
            
            return {
                'tenant_id': tenant_id,
                'deleted_at': now.isoformat()
            }
            
        except (ValidationError, AuthorizationError, TenantError):
            raise
        except Exception as e:
            logger.error(f"Error deleting tenant: {str(e)}")
            raise TenantError(f"Failed to delete tenant: {str(e)}")
    
    # ========== 租户用户管理 ==========
    
    def add_tenant_user(self, tenant_id: str, user_email: str, role: str, 
                       admin_user_id: str) -> Dict[str, Any]:
        """添加租户用户
        
        Args:
            tenant_id: 租户ID
            user_email: 用户邮箱
            role: 用户角色
            admin_user_id: 管理员用户ID
            
        Returns:
            添加结果
        """
        from backend.core.exceptions import ValidationError, AuthorizationError, TenantError
        
        try:
            # 检查管理员权限
            if not self._check_tenant_admin_access(tenant_id, admin_user_id):
                raise AuthorizationError("Admin access required")
            
            # 验证角色
            try:
                user_role = UserRole(role).value
            except ValueError:
                raise ValidationError(f"Invalid role: {role}. Valid values: {[r.value for r in UserRole]}")
            
            # 获取或创建用户
            user = self._get_user_by_email(user_email)
            if not user:
                # 创建邀请而不是直接添加
                return self.create_tenant_invite(tenant_id, user_email, role, admin_user_id)
            
            user_id = user.get('id') or str(uuid.uuid4())
            
            # 检查用户是否已在租户中
            if self._is_user_in_tenant(tenant_id, user_id):
                raise ValidationError("User already exists in tenant")
            
            # 检查配额
            current_count = self._get_tenant_user_count(tenant_id)
            quota = self._get_tenant_quota_object(tenant_id)
            if quota and current_count >= quota.max_users:
                raise ValidationError(f"Tenant user limit ({quota.max_users}) exceeded")
            
            # 添加用户
            now = datetime.utcnow()
            tenant_user = {
                'id': str(uuid.uuid4()),
                'tenant_id': tenant_id,
                'user_id': user_id,
                'role': user_role,
                'is_active': True,
                'added_by': admin_user_id,
                'created_at': now
            }
            
            # 保存租户用户（通过 Repository）
            self._save_tenant_user(tenant_user)
            
            # 记录审计日志
            self._log_audit(
                tenant_id=tenant_id,
                user_id=admin_user_id,
                action=AuditAction.USER_ADDED.value,
                resource_type='tenant_user',
                resource_id=user_id,
                details={'email': user_email, 'role': user_role}
            )
            
            logger.info(f"User {user_id} added to tenant {tenant_id} with role {role}")
            
            return {
                'tenant_id': tenant_id,
                'user_id': user_id,
                'user_email': user_email,
                'role': user_role,
                'added_at': now.isoformat()
            }
            
        except (ValidationError, AuthorizationError, TenantError):
            raise
        except Exception as e:
            logger.error(f"Error adding tenant user: {str(e)}")
            raise TenantError(f"Failed to add tenant user: {str(e)}")
    
    def remove_tenant_user(self, tenant_id: str, user_id: str, admin_user_id: str) -> Dict[str, Any]:
        """移除租户用户
        
        Args:
            tenant_id: 租户ID
            user_id: 被移除的用户ID
            admin_user_id: 管理员用户ID
            
        Returns:
            移除结果
        """
        
        try:
            # 检查管理员权限
            if not self._check_tenant_admin_access(tenant_id, admin_user_id):
                raise AuthorizationError("Admin access required")
            
            # 不能移除自己
            if user_id == admin_user_id:
                raise ValidationError("Cannot remove yourself from tenant")
            
            # 检查用户是否在租户中
            user_role = self._get_user_role_in_tenant(tenant_id, user_id)
            if not user_role:
                raise ValidationError("User not found in tenant")
            
            # 不能移除所有者
            if user_role == UserRole.OWNER.value:
                raise ValidationError("Cannot remove tenant owner. Transfer ownership first.")
            
            # 检查是否是最后一个管理员
            if user_role == UserRole.ADMIN.value:
                admin_count = self._count_users_by_role(tenant_id, UserRole.ADMIN.value)
                owner_count = self._count_users_by_role(tenant_id, UserRole.OWNER.value)
                if admin_count <= 1 and owner_count == 0:
                    raise ValidationError("Cannot remove the last admin")
            
            # 移除用户（通过 Repository）
            self._remove_tenant_user(tenant_id, user_id)
            
            # 清理用户相关数据
            self._cleanup_user_tenant_data(tenant_id, user_id)
            
            # 记录审计日志
            self._log_audit(
                tenant_id=tenant_id,
                user_id=admin_user_id,
                action=AuditAction.USER_REMOVED.value,
                resource_type='tenant_user',
                resource_id=user_id,
                details={'removed_user_role': user_role}
            )
            
            logger.info(f"User {user_id} removed from tenant {tenant_id}")
            
            return {
                'tenant_id': tenant_id,
                'user_id': user_id,
                'removed_at': datetime.utcnow().isoformat()
            }
            
        except (ValidationError, AuthorizationError, TenantError):
            raise
        except Exception as e:
            logger.error(f"Error removing tenant user: {str(e)}")
            raise TenantError(f"Failed to remove tenant user: {str(e)}")
    
    def update_user_role(self, tenant_id: str, user_id: str, new_role: str, 
                        admin_user_id: str) -> Dict[str, Any]:
        """更新用户角色
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            new_role: 新角色
            admin_user_id: 管理员用户ID
            
        Returns:
            更新结果
        """
        try:
            # 检查管理员权限
            admin_role = self._get_user_role_in_tenant(tenant_id, admin_user_id)
            if admin_role not in [UserRole.OWNER.value, UserRole.ADMIN.value]:
                raise AuthorizationError("Admin access required")
            
            # 验证新角色
            try:
                new_role_value = UserRole(new_role).value
            except ValueError:
                raise ValidationError(f"Invalid role: {new_role}")
            
            # 获取当前角色
            old_role = self._get_user_role_in_tenant(tenant_id, user_id)
            if not old_role:
                raise ValidationError("User not found in tenant")
            
            # 只有所有者可以转移所有权
            if new_role_value == UserRole.OWNER.value and admin_role != UserRole.OWNER.value:
                raise AuthorizationError("Only owner can transfer ownership")
            
            # 更新角色（通过 Repository）
            self._update_tenant_user_role(tenant_id, user_id, new_role_value)
            
            # 如果是转移所有权，更新原所有者角色
            if new_role_value == UserRole.OWNER.value:
                self._demote_current_owner(tenant_id, admin_user_id)
            
            # 记录审计日志
            self._log_audit(
                tenant_id=tenant_id,
                user_id=admin_user_id,
                action=AuditAction.USER_ROLE_CHANGED.value,
                resource_type='tenant_user',
                resource_id=user_id,
                details={'old_role': old_role, 'new_role': new_role_value}
            )
            
            logger.info(f"User {user_id} role changed from {old_role} to {new_role_value} in tenant {tenant_id}")
            
            return {
                'tenant_id': tenant_id,
                'user_id': user_id,
                'old_role': old_role,
                'new_role': new_role_value,
                'updated_at': datetime.utcnow().isoformat()
            }
            
        except (ValidationError, AuthorizationError, TenantError):
            raise
        except Exception as e:
            logger.error(f"Error updating user role: {str(e)}")
            raise TenantError(f"Failed to update user role: {str(e)}")
    
    def list_tenant_users(self, tenant_id: str, user_id: str, 
                         page: int = 1, page_size: int = 20,
                         role: Optional[str] = None) -> Dict[str, Any]:
        """获取租户用户列表
        
        Args:
            tenant_id: 租户ID
            user_id: 请求用户ID
            page: 页码
            page_size: 每页大小
            role: 角色过滤
            
        Returns:
            用户列表
        """
        from backend.core.exceptions import AuthorizationError, TenantError
        
        try:
            # 检查访问权限
            if not self._check_tenant_access(tenant_id, user_id):
                raise AuthorizationError("Access denied to tenant")
            
            # 获取用户列表
            all_users = self._get_tenant_users(tenant_id)
            
            # 角色过滤
            if role:
                all_users = [u for u in all_users if u.get('role') == role]
            
            # 分页
            total = len(all_users)
            offset = (page - 1) * page_size
            users = all_users[offset:offset + page_size]
            total_pages = (total + page_size - 1) // page_size
            
            # 增强用户信息
            user_list = []
            for tu in users:
                user_info = self._get_user_info(tu.get('user_id'))
                user_list.append({
                    'user_id': tu.get('user_id'),
                    'email': user_info.get('email', ''),
                    'name': user_info.get('name', ''),
                    'role': tu.get('role'),
                    'status': 'active' if tu.get('is_active') else 'inactive',
                    'joined_at': self._format_datetime(tu.get('created_at')),
                    'last_active': user_info.get('last_active')
                })
            
            return {
                'users': user_list,
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': total_pages
            }
            
        except (AuthorizationError, TenantError):
            raise
        except Exception as e:
            logger.error(f"Error listing tenant users: {str(e)}")
            raise TenantError(f"Failed to list tenant users: {str(e)}")
    
    # ========== 邀请管理 ==========
    
    def create_tenant_invite(self, tenant_id: str, email: str, role: str, 
                            invited_by: str) -> Dict[str, Any]:
        """创建租户邀请
        
        Args:
            tenant_id: 租户ID
            email: 被邀请人邮箱
            role: 角色
            invited_by: 邀请人用户ID
            
        Returns:
            邀请信息
        """
        from backend.core.exceptions import ValidationError, AuthorizationError, TenantError
        
        try:
            # 检查权限
            if not self._check_tenant_admin_access(tenant_id, invited_by):
                raise AuthorizationError("Admin access required")
            
            # 验证角色
            try:
                user_role = UserRole(role).value
            except ValueError:
                raise ValidationError(f"Invalid role: {role}")
            
            # 检查是否已有待处理的邀请
            existing_invite = self._get_pending_invite(tenant_id, email)
            if existing_invite:
                raise ValidationError(f"Pending invite already exists for {email}")
            
            # 创建邀请
            invite = TenantInvite(
                id=str(uuid.uuid4()),
                tenant_id=tenant_id,
                email=email,
                role=user_role,
                invite_code=secrets.token_urlsafe(32),
                invited_by=invited_by
            )
            
            # 保存邀请（通过 Repository）
            self.invite_repo.create({
                'id': invite.id,
                'tenant_id': invite.tenant_id,
                'email': invite.email,
                'role': invite.role,
                'invite_code': invite.invite_code,
                'invited_by': invite.invited_by,
                'status': invite.status,
                'expires_at': invite.expires_at
            })
            
            # TODO: 发送邀请邮件
            
            logger.info(f"Invite created for {email} to tenant {tenant_id}")
            
            return {
                'invite_id': invite.id,
                'tenant_id': tenant_id,
                'email': email,
                'role': user_role,
                'invite_code': invite.invite_code,
                'expires_at': invite.expires_at.isoformat(),
                'type': 'invite'
            }
            
        except (ValidationError, AuthorizationError, TenantError):
            raise
        except Exception as e:
            logger.error(f"Error creating invite: {str(e)}")
            raise TenantError(f"Failed to create invite: {str(e)}")
    
    def accept_tenant_invite(self, invite_code: str, user_id: str) -> Dict[str, Any]:
        """接受租户邀请
        
        Args:
            invite_code: 邀请码
            user_id: 用户ID
            
        Returns:
            结果
        """
        from backend.core.exceptions import ValidationError, TenantError
        
        try:
            # 查找邀请（通过 Repository）
            invite = self.invite_repo.get_by_code(invite_code)
            
            if not invite:
                raise ValidationError("Invalid invite code")
            
            # 获取邀请状态
            invite_status = invite.status if hasattr(invite, 'status') else invite.get('status')
            invite_expires_at = invite.expires_at if hasattr(invite, 'expires_at') else invite.get('expires_at')
            invite_tenant_id = invite.tenant_id if hasattr(invite, 'tenant_id') else invite.get('tenant_id')
            invite_role = invite.role if hasattr(invite, 'role') else invite.get('role')
            invite_invited_by = invite.invited_by if hasattr(invite, 'invited_by') else invite.get('invited_by')
            invite_id = str(invite.id) if hasattr(invite, 'id') else invite.get('id')
            
            # 检查状态
            if invite_status != InviteStatus.PENDING.value:
                raise ValidationError(f"Invite is {invite_status}")
            
            # 检查是否过期
            if datetime.utcnow() > invite_expires_at:
                self.invite_repo.update_status(invite_id, InviteStatus.EXPIRED.value)
                raise ValidationError("Invite has expired")
            
            # 添加用户到租户
            now = datetime.utcnow()
            tenant_user = {
                'id': str(uuid.uuid4()),
                'tenant_id': invite_tenant_id,
                'user_id': user_id,
                'role': invite_role,
                'is_active': True,
                'added_by': invite_invited_by,
                'created_at': now
            }
            
            # 保存租户用户（通过 Repository）
            self._save_tenant_user(tenant_user)
            
            # 更新邀请状态（通过 Repository）
            self.invite_repo.update_status(invite_id, InviteStatus.ACCEPTED.value, user_id)
            
            logger.info(f"Invite accepted by user {user_id} for tenant {invite_tenant_id}")
            
            return {
                'tenant_id': invite_tenant_id,
                'user_id': user_id,
                'role': invite_role,
                'joined_at': now.isoformat()
            }
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Error accepting invite: {str(e)}")
            raise TenantError(f"Failed to accept invite: {str(e)}")
    
    # ========== 资源配额管理 ==========
    
    def get_tenant_resource_usage(self, tenant_id: str, user_id: str) -> Dict[str, Any]:
        """获取租户资源使用情况
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            资源使用情况
        """
        from backend.core.exceptions import AuthorizationError, TenantError
        
        try:
            # 检查权限
            if not self._check_tenant_access(tenant_id, user_id):
                raise AuthorizationError("Access denied to tenant")
            
            # 获取配额
            quota = self._get_tenant_quota_object(tenant_id)
            
            # 计算使用情况
            usage = self._calculate_resource_usage(tenant_id)
            
            # 计算使用率
            usage_details = {}
            quota_dict = quota.to_dict() if quota else self.default_quotas.to_dict()
            usage_dict = usage.to_dict() if usage else {}
            
            resource_mapping = {
                'max_users': 'current_users',
                'max_training_sessions': 'total_training_sessions',
                'max_models': 'total_models',
                'max_datasets': 'total_datasets',
                'storage_limit_gb': 'storage_used_gb',
                'compute_hours_monthly': 'compute_hours_used',
                'api_requests_daily': 'api_requests_today',
                'gpu_hours_monthly': 'gpu_hours_used'
            }
            
            for quota_key, usage_key in resource_mapping.items():
                limit = quota_dict.get(quota_key, 0)
                current = usage_dict.get(usage_key, 0)
                percentage = (current / limit * 100) if limit > 0 else 0
                
                usage_details[quota_key] = {
                    'current': current,
                        'limit': limit,
                    'percentage': round(percentage, 2),
                    'remaining': max(0, limit - current),
                    'status': 'critical' if percentage >= 90 else ('warning' if percentage >= 70 else 'normal')
                    }
            
            return {
                'tenant_id': tenant_id,
                'resource_usage': usage_details,
                'last_updated': datetime.utcnow().isoformat()
            }
            
        except (AuthorizationError, TenantError):
            raise
        except Exception as e:
            logger.error(f"Error getting resource usage: {str(e)}")
            raise TenantError(f"Failed to get resource usage: {str(e)}")
    
    def update_tenant_quota(self, tenant_id: str, quota_updates: Dict[str, int], 
                           admin_user_id: str) -> Dict[str, Any]:
        """更新租户配额
        
        Args:
            tenant_id: 租户ID
            quota_updates: 配额更新
            admin_user_id: 管理员用户ID
            
        Returns:
            更新后的配额
        """
        from backend.core.exceptions import ValidationError, AuthorizationError, TenantError
        
        try:
            # 检查超级管理员权限
            if not self._check_super_admin_access(admin_user_id):
                raise AuthorizationError("Super admin access required")
            
            # 获取当前配额
            current_quota = self._get_tenant_quota_object(tenant_id)
            if not current_quota:
                current_quota = TenantQuota()
            
            # 验证并更新配额
            valid_fields = {f.name for f in dataclass_fields(TenantQuota)}
            for key, value in quota_updates.items():
                if key not in valid_fields:
                    raise ValidationError(f"Invalid quota field: {key}")
                if not isinstance(value, (int, float)) or value < 0:
                    raise ValidationError(f"Invalid value for {key}: must be non-negative number")
                setattr(current_quota, key, int(value))
            
            # 保存更新后的配额
            self._tenant_quotas[tenant_id] = current_quota
            
            # 记录审计日志
            self._log_audit(
                tenant_id=tenant_id,
                user_id=admin_user_id,
                action=AuditAction.QUOTA_UPDATED.value,
                resource_type='tenant_quota',
                resource_id=tenant_id,
                details={'updated_quotas': quota_updates}
            )
            
            logger.info(f"Tenant {tenant_id} quota updated by {admin_user_id}")
            
            return {
                'tenant_id': tenant_id,
                'quota': current_quota.to_dict(),
                'updated_at': datetime.utcnow().isoformat()
            }
            
        except (ValidationError, AuthorizationError, TenantError):
            raise
        except Exception as e:
            logger.error(f"Error updating quota: {str(e)}")
            raise TenantError(f"Failed to update quota: {str(e)}")
    
    # ========== API 密钥管理 ==========
    
    def create_api_key(self, tenant_id: str, name: str, permissions: List[str],
                      user_id: str, expires_days: Optional[int] = None,
                      description: str = "", scopes: Optional[List[str]] = None,
                      rate_limit: int = 1000) -> Dict[str, Any]:
        """创建 API 密钥
        
        Args:
            tenant_id: 租户ID
            name: 密钥名称
            permissions: 权限列表
            user_id: 创建者用户ID
            expires_days: 过期天数（可选）
            description: 密钥描述
            scopes: 作用域列表
            rate_limit: 每小时请求限制
            
        Returns:
            API 密钥信息（包含明文密钥，仅在创建时返回）
        """
        from backend.core.exceptions import ValidationError, AuthorizationError, TenantError
        
        try:
            # 检查权限
            if not self._check_tenant_admin_access(tenant_id, user_id):
                raise AuthorizationError("Admin access required")
            
            # 检查配额
            current_key_count = self._get_api_key_count(tenant_id)
            quota = self._get_tenant_quota_object(tenant_id)
            max_keys = getattr(quota, 'max_api_keys', 10) if quota else 10
            if current_key_count >= max_keys:
                raise ValidationError(f"API key limit ({max_keys}) exceeded")
            
            # 验证名称唯一性
            if self._api_key_name_exists(tenant_id, name):
                raise ValidationError(f"API key with name '{name}' already exists")
            
            # 生成密钥
            raw_key = secrets.token_urlsafe(32)
            key_prefix = raw_key[:8]
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            
            key_id = str(uuid.uuid4())
            now = datetime.utcnow()
            expires_at = now + timedelta(days=expires_days) if expires_days else None
            
            # 保存 API 密钥（通过 Repository）
            api_key = self._save_api_key(
                key_id=key_id,
                tenant_id=tenant_id,
                name=name,
                description=description,
                key_hash=key_hash,
                key_prefix=key_prefix,
                permissions=permissions,
                scopes=scopes or [],
                rate_limit=rate_limit,
                created_by=user_id,
                expires_at=expires_at
            )
            
            # 记录审计日志
            self._log_audit(
                tenant_id=tenant_id,
                user_id=user_id,
                action=AuditAction.API_KEY_CREATED.value,
                resource_type='api_key',
                resource_id=key_id,
                details={'name': name, 'permissions': permissions, 'scopes': scopes}
            )
            
            logger.info(f"API key '{name}' created for tenant {tenant_id}")
            
            return {
                'key_id': key_id,
                'api_key': raw_key,  # 仅在创建时返回明文密钥
                'key_prefix': key_prefix,
                'name': name,
                'description': description,
                'permissions': permissions,
                'scopes': scopes or [],
                'rate_limit': rate_limit,
                'created_at': now.isoformat(),
                'expires_at': expires_at.isoformat() if expires_at else None
            }
            
        except (ValidationError, AuthorizationError):
            raise
        except Exception as e:
            logger.error(f"Error creating API key: {str(e)}")
            raise TenantError(f"Failed to create API key: {str(e)}")
    
    def get_api_key(self, tenant_id: str, key_id: str, user_id: str) -> Dict[str, Any]:
        """获取 API 密钥详情
        
        Args:
            tenant_id: 租户ID
            key_id: 密钥ID
            user_id: 用户ID
            
        Returns:
            API 密钥详情（不包含密钥内容）
        """
        from backend.core.exceptions import ValidationError, AuthorizationError, TenantError
        
        try:
            # 检查权限
            if not self._check_tenant_admin_access(tenant_id, user_id):
                raise AuthorizationError("Admin access required")
            
            api_key = self._get_api_key_by_id(tenant_id, key_id)
            if not api_key:
                raise ValidationError("API key not found")
            
            return self._format_api_key(api_key)
            
        except (ValidationError, AuthorizationError):
            raise
        except Exception as e:
            logger.error(f"Error getting API key: {str(e)}")
            raise TenantError(f"Failed to get API key: {str(e)}")
    
    def update_api_key(self, tenant_id: str, key_id: str, user_id: str,
                      name: Optional[str] = None, description: Optional[str] = None,
                      permissions: Optional[List[str]] = None,
                      scopes: Optional[List[str]] = None,
                      rate_limit: Optional[int] = None,
                      is_active: Optional[bool] = None) -> Dict[str, Any]:
        """更新 API 密钥
        
        Args:
            tenant_id: 租户ID
            key_id: 密钥ID
            user_id: 用户ID
            name: 新名称
            description: 新描述
            permissions: 新权限列表
            scopes: 新作用域列表
            rate_limit: 新请求限制
            is_active: 是否激活
            
        Returns:
            更新后的 API 密钥信息
        """
        from backend.core.exceptions import ValidationError, AuthorizationError, TenantError
        
        try:
            # 检查权限
            if not self._check_tenant_admin_access(tenant_id, user_id):
                raise AuthorizationError("Admin access required")
            
            api_key = self._get_api_key_by_id(tenant_id, key_id)
            if not api_key:
                raise ValidationError("API key not found")
            
            # 检查名称唯一性
            if name and name != self._get_api_key_name(api_key):
                if self._api_key_name_exists(tenant_id, name, exclude_key_id=key_id):
                    raise ValidationError(f"API key with name '{name}' already exists")
            
            # 准备更新数据
            update_data = {}
            if name is not None:
                update_data['name'] = name
            if description is not None:
                update_data['description'] = description
            if permissions is not None:
                update_data['permissions'] = permissions
            if scopes is not None:
                update_data['scopes'] = scopes
            if rate_limit is not None:
                update_data['rate_limit'] = rate_limit
            if is_active is not None:
                update_data['is_active'] = is_active
            
            if not update_data:
                raise ValidationError("No valid fields to update")
            
            # 更新 API 密钥（通过 Repository）
            self._update_api_key(key_id, update_data)
            
            # 记录审计日志
            self._log_audit(
                tenant_id=tenant_id,
                user_id=user_id,
                action='api_key_updated',
                resource_type='api_key',
                resource_id=key_id,
                details={'updated_fields': list(update_data.keys())}
            )
            
            logger.info(f"API key {key_id} updated for tenant {tenant_id}")
            
            # 返回更新后的密钥信息
            updated_key = self._get_api_key_by_id(tenant_id, key_id)
            return self._format_api_key(updated_key)
            
        except (ValidationError, AuthorizationError):
            raise
        except Exception as e:
            logger.error(f"Error updating API key: {str(e)}")
            raise TenantError(f"Failed to update API key: {str(e)}")
    
    def revoke_api_key(self, tenant_id: str, key_id: str, user_id: str,
                      reason: Optional[str] = None) -> Dict[str, Any]:
        """撤销 API 密钥
        
        Args:
            tenant_id: 租户ID
            key_id: 密钥ID
            user_id: 用户ID
            reason: 撤销原因
            
        Returns:
            结果
        """
        from backend.core.exceptions import ValidationError, AuthorizationError, TenantError
        
        try:
            # 检查权限
            if not self._check_tenant_admin_access(tenant_id, user_id):
                raise AuthorizationError("Admin access required")
            
            api_key = self._get_api_key_by_id(tenant_id, key_id)
            if not api_key:
                raise ValidationError("API key not found")
            
            # 检查是否已撤销
            is_active = api_key.is_active if hasattr(api_key, 'is_active') else api_key.get('is_active', True)
            if not is_active:
                raise ValidationError("API key is already revoked")
            
            now = datetime.utcnow()
            key_name = self._get_api_key_name(api_key)
            
            # 撤销 API 密钥（通过 Repository）
            self._revoke_api_key(key_id, user_id, reason)
            
            # 记录审计日志
            self._log_audit(
                tenant_id=tenant_id,
                user_id=user_id,
                action=AuditAction.API_KEY_REVOKED.value,
                resource_type='api_key',
                resource_id=key_id,
                details={'name': key_name, 'reason': reason}
            )
            
            logger.info(f"API key {key_id} revoked for tenant {tenant_id}")
            
            return {
                'key_id': key_id,
                'name': key_name,
                'revoked_at': now.isoformat(),
                'revoked_by': user_id,
                'reason': reason
            }
            
        except (ValidationError, AuthorizationError):
            raise
        except Exception as e:
            logger.error(f"Error revoking API key: {str(e)}")
            raise TenantError(f"Failed to revoke API key: {str(e)}")
    
    def delete_api_key(self, tenant_id: str, key_id: str, user_id: str) -> Dict[str, Any]:
        """永久删除 API 密钥
        
        Args:
            tenant_id: 租户ID
            key_id: 密钥ID
            user_id: 用户ID
            
        Returns:
            结果
        """
        from backend.core.exceptions import ValidationError, AuthorizationError, TenantError
        
        try:
            # 检查权限
            if not self._check_tenant_admin_access(tenant_id, user_id):
                raise AuthorizationError("Admin access required")
            
            api_key = self._get_api_key_by_id(tenant_id, key_id)
            if not api_key:
                raise ValidationError("API key not found")
            
            key_name = self._get_api_key_name(api_key)
            
            # 删除 API 密钥（通过 Repository）
            self._delete_api_key(key_id)
            
            # 记录审计日志
            self._log_audit(
                tenant_id=tenant_id,
                user_id=user_id,
                action='api_key_deleted',
                resource_type='api_key',
                resource_id=key_id,
                details={'name': key_name}
            )
            
            logger.info(f"API key {key_id} deleted for tenant {tenant_id}")
            
            return {
                'key_id': key_id,
                'name': key_name,
                'deleted_at': datetime.utcnow().isoformat()
            }
            
        except (ValidationError, AuthorizationError):
            raise
        except Exception as e:
            logger.error(f"Error deleting API key: {str(e)}")
            raise TenantError(f"Failed to delete API key: {str(e)}")
    
    def list_api_keys(self, tenant_id: str, user_id: str, 
                     include_revoked: bool = False,
                     page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """获取 API 密钥列表
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            include_revoked: 是否包含已撤销的密钥
            page: 页码
            page_size: 每页大小
            
        Returns:
            API 密钥列表（不包含密钥内容）
        """
        from backend.core.exceptions import AuthorizationError, TenantError
        
        try:
            # 检查权限
            if not self._check_tenant_admin_access(tenant_id, user_id):
                raise AuthorizationError("Admin access required")
            
            # 获取 API 密钥列表（通过 Repository）
            keys, total = self._list_api_keys(tenant_id, include_revoked, page, page_size)
            
            total_pages = (total + page_size - 1) // page_size
            
            return {
                'api_keys': [self._format_api_key(k) for k in keys],
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': total_pages
            }
            
        except AuthorizationError:
            raise
        except Exception as e:
            logger.error(f"Error listing API keys: {str(e)}")
            raise TenantError(f"Failed to list API keys: {str(e)}")
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """验证 API 密钥
        
        Args:
            api_key: 明文 API 密钥
            
        Returns:
            验证结果，包含租户和权限信息；如果无效则返回 None
        """
        try:
            if not api_key or len(api_key) < 8:
                return None
            
            key_prefix = api_key[:8]
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            # 查找 API 密钥（通过 Repository）
            key_record = self._find_api_key_by_hash(key_hash, key_prefix)
            
            if not key_record:
                return None
            
            # 检查是否激活
            is_active = key_record.is_active if hasattr(key_record, 'is_active') else key_record.get('is_active', True)
            if not is_active:
                return None
            
            # 检查是否过期
            expires_at = key_record.expires_at if hasattr(key_record, 'expires_at') else key_record.get('expires_at')
            if expires_at and datetime.utcnow() > expires_at:
                return None
            
            tenant_id = key_record.tenant_id if hasattr(key_record, 'tenant_id') else key_record.get('tenant_id')
            
            # 更新使用统计
            self._update_api_key_usage(key_record)
            
            # 解析权限和作用域
            permissions = key_record.permissions if hasattr(key_record, 'permissions') else key_record.get('permissions', [])
            if isinstance(permissions, str):
                permissions = json.loads(permissions) if permissions else []
            
            scopes = key_record.scopes if hasattr(key_record, 'scopes') else key_record.get('scopes', [])
            if isinstance(scopes, str):
                scopes = json.loads(scopes) if scopes else []
            
            return {
                'valid': True,
                'key_id': str(key_record.id) if hasattr(key_record, 'id') else key_record.get('id'),
                'tenant_id': tenant_id,
                'name': key_record.name if hasattr(key_record, 'name') else key_record.get('name'),
                'permissions': permissions,
                'scopes': scopes,
                'rate_limit': key_record.rate_limit if hasattr(key_record, 'rate_limit') else key_record.get('rate_limit', 1000)
            }
            
        except Exception as e:
            logger.error(f"Error validating API key: {str(e)}")
            return None
    
    def regenerate_api_key(self, tenant_id: str, key_id: str, user_id: str) -> Dict[str, Any]:
        """重新生成 API 密钥（保留设置，更换密钥值）
        
        Args:
            tenant_id: 租户ID
            key_id: 密钥ID
            user_id: 用户ID
            
        Returns:
            新的 API 密钥信息
        """
        from backend.core.exceptions import ValidationError, AuthorizationError, TenantError
        
        try:
            # 检查权限
            if not self._check_tenant_admin_access(tenant_id, user_id):
                raise AuthorizationError("Admin access required")
            
            api_key = self._get_api_key_by_id(tenant_id, key_id)
            if not api_key:
                raise ValidationError("API key not found")
            
            # 检查是否已撤销
            is_active = api_key.is_active if hasattr(api_key, 'is_active') else api_key.get('is_active', True)
            if not is_active:
                raise ValidationError("Cannot regenerate a revoked API key")
            
            # 生成新密钥
            raw_key = secrets.token_urlsafe(32)
            key_prefix = raw_key[:8]
            key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
            now = datetime.utcnow()
            
            # 更新 API 密钥（通过 Repository）
            self._update_api_key(key_id, {
                'key_hash': key_hash,
                'key_prefix': key_prefix
            })
            
            key_name = self._get_api_key_name(api_key)
            
            # 记录审计日志
            self._log_audit(
                tenant_id=tenant_id,
                user_id=user_id,
                action='api_key_regenerated',
                resource_type='api_key',
                resource_id=key_id,
                details={'name': key_name}
            )
            
            logger.info(f"API key {key_id} regenerated for tenant {tenant_id}")
            
            return {
                'key_id': key_id,
                'api_key': raw_key,  # 仅在重新生成时返回明文密钥
                'key_prefix': key_prefix,
                'name': key_name,
                'regenerated_at': now.isoformat()
            }
            
        except (ValidationError, AuthorizationError):
            raise
        except Exception as e:
            logger.error(f"Error regenerating API key: {str(e)}")
            raise TenantError(f"Failed to regenerate API key: {str(e)}")
    
    # ========== API 密钥操作（通过 Repository）==========
    
    def _save_api_key(self, key_id: str, tenant_id: str, name: str,
                     description: str, key_hash: str, key_prefix: str,
                     permissions: List[str], scopes: List[str],
                     rate_limit: int, created_by: str,
                     expires_at: Optional[datetime]) -> Any:
        """保存 API 密钥（通过 Repository）"""
        return self.api_key_repo.create({
            'id': key_id,
            'tenant_id': tenant_id,
            'name': name,
            'description': description,
            'key_hash': key_hash,
            'key_prefix': key_prefix,
            'permissions': permissions,
            'scopes': scopes,
            'rate_limit': rate_limit,
            'created_by': created_by,
            'expires_at': expires_at
        })
    
    def _get_api_key_by_id(self, tenant_id: str, key_id: str) -> Optional[Any]:
        """根据ID获取 API 密钥（通过 Repository）"""
        return self.api_key_repo.get_by_tenant_and_id(tenant_id, key_id)
    
    def _update_api_key(self, key_id: str, update_data: Dict[str, Any]) -> Optional[Any]:
        """更新 API 密钥（通过 Repository）"""
        return self.api_key_repo.update(key_id, update_data)
    
    def _revoke_api_key(self, key_id: str, revoked_by: str,
                       reason: Optional[str]) -> bool:
        """撤销 API 密钥（通过 Repository）"""
        return self.api_key_repo.revoke(key_id, revoked_by, reason)
    
    def _delete_api_key(self, key_id: str) -> bool:
        """删除 API 密钥（通过 Repository）"""
        return self.api_key_repo.delete(key_id)
    
    def _list_api_keys(self, tenant_id: str, include_revoked: bool,
                      page: int, page_size: int) -> Tuple[List[Any], int]:
        """获取 API 密钥列表（通过 Repository）"""
        return self.api_key_repo.list_by_tenant(tenant_id, include_revoked, page_size, (page - 1) * page_size)
    
    def _find_api_key_by_hash(self, key_hash: str, key_prefix: str) -> Optional[Any]:
        """根据哈希查找 API 密钥（通过 Repository）"""
        return self.api_key_repo.get_by_hash(key_hash, key_prefix)
    
    def _update_api_key_usage(self, api_key: Any):
        """更新 API 密钥使用统计（通过 Repository）"""
        try:
            key_id = str(api_key.id) if hasattr(api_key, 'id') else api_key.get('id')
            if key_id:
                self.api_key_repo.update_usage(key_id)
        except Exception as e:
            logger.debug(f"Error updating API key usage: {e}")
    
    def _get_api_key_count(self, tenant_id: str) -> int:
        """获取租户 API 密钥数量（通过 Repository）"""
        return self.api_key_repo.count_by_tenant(tenant_id, is_active=True)
    
    def _api_key_name_exists(self, tenant_id: str, name: str, 
                            exclude_key_id: Optional[str] = None) -> bool:
        """检查 API 密钥名称是否存在（通过 Repository）"""
        return self.api_key_repo.name_exists(tenant_id, name, exclude_key_id)
    
    def _get_api_key_name(self, api_key: Any) -> str:
        """获取 API 密钥名称"""
        if hasattr(api_key, 'name'):
            return api_key.name
        return api_key.get('name', '')
    
    def _format_api_key(self, api_key: Any) -> Dict[str, Any]:
        """格式化 API 密钥信息"""
        if hasattr(api_key, 'to_dict'):
            return api_key.to_dict()
        
        # 处理数据库模型
        permissions = api_key.permissions
        if isinstance(permissions, str):
            permissions = json.loads(permissions) if permissions else []
        
        scopes = api_key.scopes if hasattr(api_key, 'scopes') else None
        if isinstance(scopes, str):
            scopes = json.loads(scopes) if scopes else []
        
        return {
            'id': str(api_key.id),
            'tenant_id': api_key.tenant_id,
            'name': api_key.name,
            'description': getattr(api_key, 'description', ''),
            'key_prefix': api_key.key_prefix,
            'permissions': permissions,
            'scopes': scopes or [],
            'rate_limit': getattr(api_key, 'rate_limit', 1000),
            'is_active': api_key.is_active,
            'created_by': api_key.created_by,
            'created_at': api_key.created_at.isoformat() if api_key.created_at else None,
            'updated_at': api_key.updated_at.isoformat() if hasattr(api_key, 'updated_at') and api_key.updated_at else None,
            'last_used_at': api_key.last_used_at.isoformat() if api_key.last_used_at else None,
            'last_used_ip': getattr(api_key, 'last_used_ip', None),
            'use_count': getattr(api_key, 'use_count', 0),
            'expires_at': api_key.expires_at.isoformat() if api_key.expires_at else None,
            'revoked_at': api_key.revoked_at.isoformat() if getattr(api_key, 'revoked_at', None) else None,
            'revoked_by': getattr(api_key, 'revoked_by', None)
        }
    
    # ========== 审计日志 ==========
    
    def get_audit_logs(self, tenant_id: str, user_id: str,
                      action: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      page: int = 1, page_size: int = 50) -> Dict[str, Any]:
        """获取审计日志
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            action: 操作类型过滤
            start_date: 开始日期
            end_date: 结束日期
            page: 页码
            page_size: 每页大小
            
        Returns:
            审计日志列表
        """
        from backend.core.exceptions import AuthorizationError, TenantError
        
        try:
            # 检查管理员权限
            if not self._check_tenant_admin_access(tenant_id, user_id):
                raise AuthorizationError("Admin access required")
            
            # 通过 Repository 获取日志
            offset = (page - 1) * page_size
            logs, total = self.audit_log_repo.list_by_tenant(
                tenant_id=tenant_id,
                action=action,
                start_date=start_date,
                end_date=end_date,
                limit=page_size,
                offset=offset
            )
            
            total_pages = (total + page_size - 1) // page_size
            
            # 格式化日志
            formatted_logs = []
            for log in logs:
                if hasattr(log, 'to_dict'):
                    formatted_logs.append(log.to_dict())
                elif isinstance(log, dict):
                    formatted_logs.append(log)
                else:
                    # 数据库模型
                    formatted_logs.append({
                        'id': str(log.id),
                        'tenant_id': log.tenant_id,
                        'user_id': log.user_id,
                        'action': log.action,
                        'resource_type': log.resource_type,
                        'resource_id': log.resource_id,
                        'details': json.loads(log.details) if isinstance(log.details, str) else log.details,
                        'ip_address': log.ip_address,
                        'user_agent': log.user_agent,
                        'timestamp': log.timestamp.isoformat() if log.timestamp else None
                    })
            
            return {
                'logs': formatted_logs,
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': total_pages
            }
            
        except AuthorizationError:
            raise
        except Exception as e:
            logger.error(f"Error getting audit logs: {str(e)}")
            raise TenantError(f"Failed to get audit logs: {str(e)}")
    
    # ========== 私有辅助方法 ==========
    
    def _validate_tenant_data(self, tenant_data: Dict[str, Any]):
        """验证租户数据"""
        from backend.core.exceptions import ValidationError
        
        if 'name' not in tenant_data or not tenant_data['name']:
            raise ValidationError("Tenant name is required")
        
        name = tenant_data['name']
        if not name.replace('_', '').replace('-', '').isalnum():
            raise ValidationError("Tenant name can only contain letters, numbers, hyphens and underscores")
        
        if len(name) < 3 or len(name) > 50:
            raise ValidationError("Tenant name must be between 3 and 50 characters")
    
    def _tenant_name_exists(self, name: str) -> bool:
        """检查租户名称是否存在（通过 Repository）"""
        return self.tenant_repo.name_exists(name)
    
    def _get_tenant_by_id(self, tenant_id: str) -> Optional[Dict]:
        """根据ID获取租户（通过 Repository）"""
        tenant = self.tenant_repo.get_by_id(tenant_id)
        if tenant:
            if hasattr(tenant, 'id'):
                # 数据库模型
                return {
                    'id': str(tenant.id),
                    'name': tenant.name,
                    'display_name': tenant.display_name,
                    'description': tenant.description,
                    'status': tenant.status,
                    'settings': tenant.settings,
                    'created_at': tenant.created_at,
                    'updated_at': tenant.updated_at
                }
            return tenant  # 内存存储
        return None
    
    def _save_tenant(self, tenant: Dict):
        """保存租户（通过 Repository）"""
        self.tenant_repo.create(tenant)
    
    def _update_tenant(self, tenant_id: str, update_data: Dict):
        """更新租户（通过 Repository）"""
        self.tenant_repo.update(tenant_id, update_data)
    
    def _get_user_tenant_ids(self, user_id: str) -> List[str]:
        """获取用户关联的租户ID列表（通过 Repository）"""
        return self.tenant_user_repo.list_by_user(user_id)
    
    def _check_tenant_access(self, tenant_id: str, user_id: str) -> bool:
        """检查用户是否有租户访问权限（通过 Repository）"""
        if self._check_super_admin_access(user_id):
            return True
        return self.tenant_user_repo.is_user_in_tenant(tenant_id, user_id)
    
    def _check_tenant_admin_access(self, tenant_id: str, user_id: str) -> bool:
        """检查用户是否有租户管理员权限"""
        if self._check_super_admin_access(user_id):
            return True
        
        role = self._get_user_role_in_tenant(tenant_id, user_id)
        return role in [UserRole.OWNER.value, UserRole.ADMIN.value]
    
    def _check_super_admin_access(self, user_id: str) -> bool:
        """检查用户是否有超级管理员权限"""
        if not self.auth_service:
            return False
        try:
            user = self.auth_service.get_user(user_id)
            return bool(user and user.get('is_superuser', False))
        except Exception as e:
            logger.warning(f"Failed to check super admin access: {e}")
            return False
    
    def _get_user_role_in_tenant(self, tenant_id: str, user_id: str) -> Optional[str]:
        """获取用户在租户中的角色（通过 Repository）"""
        return self.tenant_user_repo.get_user_role(tenant_id, user_id)
    
    def _get_tenant_user_count(self, tenant_id: str) -> int:
        """获取租户用户数量（通过 Repository）"""
        return self.tenant_user_repo.count_by_tenant(tenant_id)
    
    def _get_tenant_users(self, tenant_id: str) -> List[Dict]:
        """获取租户用户列表（通过 Repository）"""
        users, _ = self.tenant_user_repo.list_by_tenant(tenant_id)
        return [{
            'user_id': str(u.user_id) if hasattr(u, 'user_id') else u.get('user_id'),
            'role': u.role if hasattr(u, 'role') else u.get('role'),
            'is_active': u.is_active if hasattr(u, 'is_active') else u.get('is_active', True),
            'created_at': u.created_at if hasattr(u, 'created_at') else u.get('created_at')
        } for u in users]
    
    def _is_user_in_tenant(self, tenant_id: str, user_id: str) -> bool:
        """检查用户是否在租户中（通过 Repository）"""
        return self.tenant_user_repo.is_user_in_tenant(tenant_id, user_id)
    
    def _count_users_by_role(self, tenant_id: str, role: str) -> int:
        """计算特定角色的用户数量（通过 Repository）"""
        return self.tenant_user_repo.count_by_role(tenant_id, role)
    
    def _get_user_by_email(self, email: str) -> Optional[Dict]:
        """根据邮箱获取用户"""
        if self.auth_service and hasattr(self.auth_service, 'get_user_by_email'):
            try:
                return self.auth_service.get_user_by_email(email)
            except Exception as e:
                logger.warning(f"Failed to get user by email: {e}")
                pass
            return None
    
    def _get_user_info(self, user_id: str) -> Dict:
        """获取用户信息"""
        if self.auth_service:
            try:
                return self.auth_service.get_user(user_id) or {}
            except Exception:
                pass
        return {}
    
    def _get_pending_invite(self, tenant_id: str, email: str) -> Optional[TenantInvite]:
        """获取待处理的邀请（通过 Repository）"""
        return self.invite_repo.get_pending_by_email(tenant_id, email)
    
    def _save_tenant_user(self, tenant_user: Dict):
        """保存租户用户（通过 Repository）"""
        self.tenant_user_repo.create(tenant_user)
    
    def _remove_tenant_user(self, tenant_id: str, user_id: str) -> bool:
        """移除租户用户（通过 Repository）"""
        return self.tenant_user_repo.delete(tenant_id, user_id)
    
    def _update_tenant_user_role(self, tenant_id: str, user_id: str, new_role: str) -> bool:
        """更新用户角色（通过 Repository）"""
        return self.tenant_user_repo.update_role(tenant_id, user_id, new_role)
    
    def _demote_current_owner(self, tenant_id: str, new_owner_id: str):
        """将当前所有者降级为管理员"""
        users = self._get_tenant_users(tenant_id)
        for user in users:
            if user['role'] == UserRole.OWNER.value and user['user_id'] != new_owner_id:
                self._update_tenant_user_role(tenant_id, user['user_id'], UserRole.ADMIN.value)
    
    def _create_tenant_quotas(self, tenant_id: str, plan: str = 'basic'):
        """创建租户配额（通过 Repository）"""
        quota_data = self.quota_plans.get(plan, self.default_quotas).to_dict()
        quota_data['tenant_id'] = tenant_id
        quota_data['plan'] = plan
        self.quota_repo.create(quota_data)
    
    def _get_tenant_quota_object(self, tenant_id: str) -> Optional[TenantQuota]:
        """获取租户配额对象（通过 Repository）"""
        quota = self.quota_repo.get_by_tenant(tenant_id)
        if quota:
            if hasattr(quota, 'max_users'):
                return TenantQuota(
                    max_users=quota.max_users,
                    max_training_sessions=quota.max_training_sessions,
                    max_concurrent_trainings=quota.max_concurrent_trainings,
                    max_models=quota.max_models,
                    max_datasets=quota.max_datasets,
                    storage_limit_gb=quota.storage_limit_gb,
                    compute_hours_monthly=quota.compute_hours_monthly,
                    api_requests_daily=quota.api_requests_daily,
                    gpu_hours_monthly=quota.gpu_hours_monthly
                )
            return TenantQuota.from_dict(quota)
        return self.default_quotas
    
    def _add_tenant_owner(self, tenant_id: str, user_id: str):
        """添加租户所有者（通过 Repository）"""
        tenant_user = {
            'id': str(uuid.uuid4()),
            'tenant_id': tenant_id,
            'user_id': user_id,
            'role': UserRole.OWNER.value,
            'is_active': True,
            'added_by': user_id,
            'created_at': datetime.utcnow()
        }
        self._save_tenant_user(tenant_user)
    
    def _initialize_tenant_environment(self, tenant_id: str):
        """初始化租户环境"""
        try:
            if self.redis_client:
                # 初始化 Redis 命名空间
                self.redis_client.set(f"tenant:{tenant_id}:initialized", "true")
                
                # 初始化配置
                tenant_config = {
                    'created_at': datetime.utcnow().isoformat(),
                    'version': '1.0',
                    'features': ['training', 'models', 'datasets', 'monitoring', 'api']
                }
                self.redis_client.set(
                    f"tenant:{tenant_id}:config",
                    json.dumps(tenant_config)
                )
                
                # 初始化计数器
                self.redis_client.set(f"tenant:{tenant_id}:api_requests_today", 0)
                
            logger.info(f"Tenant environment initialized: {tenant_id}")
            
        except Exception as e:
            logger.warning(f"Error initializing tenant environment: {e}")
    
    def _get_tenant_statistics(self, tenant_id: str) -> Dict[str, Any]:
        """获取租户统计信息"""
        usage = self._calculate_resource_usage(tenant_id)
        return {
            'user_count': usage.current_users,
            'active_training_sessions': usage.active_training_sessions,
            'total_training_sessions': usage.total_training_sessions,
            'models': usage.total_models,
            'datasets': usage.total_datasets,
            'storage_used_gb': usage.storage_used_gb
        }
    
    def _calculate_resource_usage(self, tenant_id: str) -> TenantUsage:
        """计算租户资源使用情况
        
        优先通过 Redis 缓存获取统计数据，回退到 Repository 查询
        """
        usage = TenantUsage()
        
        try:
            # 用户数量（通过 Repository）
            usage.current_users = self._get_tenant_user_count(tenant_id)
            
            # 1. 尝试从 Redis 获取统计
            redis_stats_found = False
            if self.redis_client:
                try:
                    total_sessions = self.redis_client.get(f"tenant:{tenant_id}:stats:total_training_sessions")
                    active_sessions = self.redis_client.get(f"tenant:{tenant_id}:stats:active_training_sessions")
                    total_models = self.redis_client.get(f"tenant:{tenant_id}:stats:total_models")
                    total_datasets = self.redis_client.get(f"tenant:{tenant_id}:stats:total_datasets")
                    api_requests = self.redis_client.get(f"tenant:{tenant_id}:api_requests_today")
                    
                    if all(x is not None for x in [total_sessions, active_sessions, total_models, total_datasets]):
                        usage.total_training_sessions = int(total_sessions)
                        usage.active_training_sessions = int(active_sessions)
                        usage.total_models = int(total_models)
                        usage.total_datasets = int(total_datasets)
                        usage.api_requests_today = int(api_requests) if api_requests else 0
                        redis_stats_found = True
                except Exception as e:
                    logger.debug(f"Could not get tenant stats from Redis: {e}")
            
            # 2. 如果 Redis 中没有，通过 Repository 查询
            if not redis_stats_found:
                # 训练会话统计
                if self.training_session_repo:
                    from backend.schemas.enums import TrainingStatus
                    usage.total_training_sessions = self.training_session_repo.count_by_tenant(tenant_id)
                    usage.active_training_sessions = self.training_session_repo.count_by_tenant(
                        tenant_id, status=TrainingStatus.RUNNING
                    )
                
                # 模型统计
                if self.model_repo:
                    usage.total_models = self.model_repo.count_by_tenant(tenant_id)
                
                # 数据集统计
                if self.dataset_repo:
                    usage.total_datasets = self.dataset_repo.count_by_tenant(tenant_id)
                
                # API 请求数仅从 Redis 获取（因为是临时计数器）
                if self.redis_client:
                    api_requests = self.redis_client.get(f"tenant:{tenant_id}:api_requests_today")
                    usage.api_requests_today = int(api_requests) if api_requests else 0
            
        except Exception as e:
            logger.error(f"Error calculating resource usage: {e}")
        
        return usage
    
    def _clear_tenant_cache(self, tenant_id: str):
        """清除租户缓存"""
        if self.redis_client:
            try:
                # 清除租户相关的缓存键
                pattern = f"tenant:{tenant_id}:cache:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
                logger.debug(f"Cleared cache for tenant {tenant_id}")
            except Exception as e:
                logger.warning(f"Error clearing tenant cache: {e}")
    
    def _suspend_tenant_resources(self, tenant_id: str):
        """暂停租户资源
        
        通过 Redis 标志和事件通知来暂停资源，并直接暂停数据库中的活跃会话
        """
        try:
            if self.redis_client:
                # 设置暂停标志，各服务检查此标志决定是否执行
                self.redis_client.set(f"tenant:{tenant_id}:suspended", "true")
                
                # 发布暂停事件，让订阅的服务处理具体资源暂停逻辑
                self.redis_client.publish(
                    f"tenant:events",
                    f'{{"type": "suspend", "tenant_id": "{tenant_id}"}}'
                )
            
            # 直接通过 Repository 暂停活跃的训练会话
            if self.training_session_repo:
                count = self.training_session_repo.suspend_active_sessions(tenant_id)
                if count > 0:
                    logger.info(f"Suspended {count} training sessions for tenant {tenant_id}")
            
            logger.info(f"Tenant resources suspended: {tenant_id}")
            
        except Exception as e:
            logger.error(f"Error suspending tenant resources: {e}")
    
    def _activate_tenant_resources(self, tenant_id: str):
        """激活租户资源"""
        try:
            if self.redis_client:
                # 移除暂停标志
                self.redis_client.delete(f"tenant:{tenant_id}:suspended")
            
            logger.info(f"Tenant resources activated: {tenant_id}")
            
        except Exception as e:
            logger.error(f"Error activating tenant resources: {e}")
    
    def _cleanup_user_tenant_data(self, tenant_id: str, user_id: str):
        """清理用户在租户中的数据"""
        try:
            if self.redis_client:
                # 清除用户在租户中的缓存
                pattern = f"tenant:{tenant_id}:user:{user_id}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            
            logger.debug(f"Cleaned up data for user {user_id} in tenant {tenant_id}")
            
        except Exception as e:
            logger.warning(f"Error cleaning up user tenant data: {e}")
    
    def _has_active_resources(self, tenant_id: str) -> bool:
        """检查租户是否有活跃资源
        
        优先检查 Redis 缓存，回退到 Repository 查询
        """
        try:
            if self.redis_client:
                # 从 Redis 获取活跃训练会话数
                active_sessions = self.redis_client.get(f"tenant:{tenant_id}:stats:active_training_sessions")
                if active_sessions and int(active_sessions) > 0:
                    return True
            
            # 回退到 Repository 查询
            if self.training_session_repo:
                from backend.schemas.enums import TrainingStatus
                active_count = self.training_session_repo.count_by_tenant(
                    tenant_id, status=TrainingStatus.RUNNING
                )
                if active_count > 0:
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking active resources: {e}")
            return False
            
        except Exception as e:
            logger.warning(f"Error checking active resources: {e}")
            return False
    
    def _cleanup_tenant_data(self, tenant_id: str):
        """清理租户数据（通过 Repository）"""
        try:
            if self.redis_client:
                # 清除所有租户相关的 Redis 数据
                pattern = f"tenant:{tenant_id}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    self.redis_client.delete(*keys)
            
            # 通过 Repository 清理数据
            self.quota_repo.delete(tenant_id)
            self.tenant_user_repo.delete_all_by_tenant(tenant_id)
            self.api_key_repo.delete_all_by_tenant(tenant_id)
            self.invite_repo.delete_all_by_tenant(tenant_id)
            
            # 保留审计日志用于合规
            
            logger.info(f"Cleaned up data for tenant {tenant_id}")
            
        except Exception as e:
            logger.error(f"Error cleaning up tenant data: {e}")
    
    def _log_audit(self, tenant_id: str, user_id: str, action: str,
                  resource_type: str, resource_id: Optional[str],
                  details: Dict[str, Any], ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None):
        """记录审计日志（通过 Repository）"""
        try:
            log_data = {
                'id': str(uuid.uuid4()),
                'tenant_id': tenant_id,
                'user_id': user_id,
                'action': action,
                'resource_type': resource_type,
                'resource_id': resource_id,
                'details': details,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'timestamp': datetime.utcnow()
            }
            self.audit_log_repo.create(log_data)
            
        except Exception as e:
            logger.warning(f"Error logging audit: {e}")
    
    def _format_datetime(self, dt: Optional[datetime]) -> Optional[str]:
        """格式化日期时间"""
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        return dt.isoformat()


# ============================================================================
# 模块级别函数
# ============================================================================

_tenant_service: Optional[TenantService] = None


def get_tenant_service() -> TenantService:
    """获取租户服务单例"""
    global _tenant_service
    if _tenant_service is None:
        _tenant_service = TenantService()
    return _tenant_service

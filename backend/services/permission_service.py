"""权限管理服务

提供权限管理的业务逻辑，包括：
- 权限CRUD操作
- 角色管理
- 用户角色关联
- 权限验证
- Agent智能分析和推理
- 权限建议生成
- 审计日志

该服务集成了AI Agent的长记忆推理能力，用于：
- 智能权限分配建议
- 权限使用模式分析
- 风险检测和预警
- 最小权限原则优化
"""

import logging
import time
from typing import List, Optional, Dict, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from backend.repositories.permission_repository import PermissionRepository, get_permission_repository
from backend.schemas.permission_models import (
    Permission, Role, UserRole, 
    PermissionAuditLog, PermissionAgentMemory, PermissionAgentReasoning,
    PermissionPolicy, ResourcePermission, PermissionRecommendation
)
from backend.modules.database.manager import get_database_manager

logger = logging.getLogger(__name__)


# ============================================================================
# 数据传输对象 (DTOs)
# ============================================================================

@dataclass
class PermissionCheckResult:
    """权限检查结果
    
    Attributes:
        allowed: 是否允许
        reason: 原因说明
        matched_policies: 匹配的策略
        risk_level: 风险等级
        requires_mfa: 是否需要MFA
    """
    allowed: bool
    reason: str
    matched_policies: List[Dict] = field(default_factory=list)
    risk_level: str = 'low'
    requires_mfa: bool = False


@dataclass
class PermissionAnalysis:
    """权限分析结果
    
    Attributes:
        user_id: 用户ID
        total_permissions: 总权限数
        permission_categories: 权限分类统计
        role_summary: 角色摘要
        risk_assessment: 风险评估
        recommendations: 建议列表
        usage_patterns: 使用模式
        agent_insights: Agent洞察
    """
    user_id: str
    total_permissions: int
    permission_categories: Dict[str, int] = field(default_factory=dict)
    role_summary: List[Dict] = field(default_factory=list)
    risk_assessment: Dict = field(default_factory=dict)
    recommendations: List[Dict] = field(default_factory=list)
    usage_patterns: Dict = field(default_factory=dict)
    agent_insights: Dict = field(default_factory=dict)


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


def get_permission_service():
    """获取权限服务实例
    
    Returns:
        PermissionService: 权限服务实例
    """
    return PermissionService()


class PermissionService:
    """权限管理服务
    
    提供完整的权限管理功能，包括Agent智能推理能力。
    
    Attributes:
        repository: 权限数据仓库
        db_manager: 数据库管理器
        _permission_cache: 权限缓存
        _cache_ttl: 缓存有效期（秒）
    """
    
    def __init__(self, repository: Optional[PermissionRepository] = None, db_manager=None):
        """初始化服务
        
        Args:
            repository: 权限数据仓库，如果为None则使用默认仓库
            db_manager: 数据库管理器
        """
        self.repository = repository or get_permission_repository()
        self.db_manager = db_manager or get_database_manager()
        self._permission_cache: Dict[str, Tuple[Set[str], datetime]] = {}
        self._cache_ttl = 300  # 5分钟缓存
    
    # ========================================================================
    # 权限管理
    # ========================================================================
    
    def create_permission(self,
                          name: str,
                          resource: str,
                          action: str,
                          description: Optional[str] = None,
                          is_system: bool = False,
                          scope: str = 'global',
                          conditions: Optional[Dict] = None,
                          priority: int = 0,
                          risk_level: str = 'low',
                          requires_mfa: bool = False,
                          audit_level: str = 'basic',
                          created_by: Optional[str] = None) -> Permission:
        """创建权限
        
        创建新的系统权限，支持条件权限和风险等级设置。
        
        Args:
            name: 权限名称，必须唯一
            resource: 资源类型（如 users, datasets, models）
            action: 操作类型（如 create, read, update, delete）
            description: 权限描述
            is_system: 是否为系统内置权限
            scope: 权限范围（global, tenant, personal）
            conditions: 权限条件（JSON格式的条件表达式）
            priority: 优先级（数值越大优先级越高）
            risk_level: 风险等级（low, medium, high, critical）
            requires_mfa: 是否需要MFA验证
            audit_level: 审计级别（none, basic, detailed）
            created_by: 创建者ID
            
        Returns:
            Permission: 创建的权限对象
            
        Raises:
            ValueError: 如果权限名称已存在或参数无效
            
        Example:
            >>> service = PermissionService()
            >>> perm = service.create_permission(
            ...     name="datasets:delete",
            ...     resource="datasets",
            ...     action="delete",
            ...     risk_level="high",
            ...     requires_mfa=True
            ... )
        """
        # 参数验证
        if not name or not resource or not action:
            raise ValueError("Permission name, resource, and action are required")
        
        # 创建权限
        permission = self.repository.create_permission(
            name=name,
            resource=resource,
            action=action,
            description=description,
            is_system=is_system,
            scope=scope,
            conditions=conditions,
            priority=priority,
            risk_level=risk_level,
            requires_mfa=requires_mfa,
            audit_level=audit_level
        )
        
        # 记录审计日志
        if created_by:
            self.repository.create_audit_log(
                user_id=created_by,
                action='create_permission',
                permission_id=str(permission.id),
                new_value=permission.to_dict()
            )
        
        # 创建Agent记忆
        self._create_permission_memory(
            memory_type='permission_created',
            content=f"Created permission '{name}' for {resource}:{action} with risk level {risk_level}",
            metadata={
                'permission_id': str(permission.id),
                'resource': resource,
                'action': action,
                'risk_level': risk_level
            }
        )
        
        logger.info(f"Created permission: {name}")
        return permission
    
    def get_permission(self, permission_id: str) -> Optional[Permission]:
        """获取权限详情
        
        Args:
            permission_id: 权限ID
            
        Returns:
            Optional[Permission]: 权限对象，不存在则返回None
        """
        return self.repository.get_permission_by_id(permission_id)
    
    def get_permission_by_name(self, name: str) -> Optional[Permission]:
        """根据名称获取权限
        
        Args:
            name: 权限名称
            
        Returns:
            Optional[Permission]: 权限对象
        """
        return self.repository.get_permission_by_name(name)
    
    def list_permissions(self,
                         resource: Optional[str] = None,
                         action: Optional[str] = None,
                         is_active: Optional[bool] = True,
                         risk_level: Optional[str] = None,
                         page: int = 1,
                         page_size: int = 50) -> Tuple[List[Permission], int]:
        """列出权限
        
        Args:
            resource: 过滤资源类型
            action: 过滤操作类型
            is_active: 过滤是否启用
            risk_level: 过滤风险等级
            page: 页码
            page_size: 每页大小
            
        Returns:
            Tuple[List[Permission], int]: 权限列表和总数
        """
        return self.repository.list_permissions(
            resource=resource,
            action=action,
            is_active=is_active,
            risk_level=risk_level,
            page=page,
            page_size=page_size
        )
    
    def update_permission(self,
                          permission_id: str,
                          updated_by: Optional[str] = None,
                          **kwargs) -> Optional[Permission]:
        """更新权限
        
        Args:
            permission_id: 权限ID
            updated_by: 更新者ID
            **kwargs: 要更新的字段
            
        Returns:
            Optional[Permission]: 更新后的权限对象
        """
        # 获取原始权限用于审计
        old_permission = self.repository.get_permission_by_id(permission_id)
        if not old_permission:
            return None
        
        old_value = old_permission.to_dict()
        
        # 执行更新
        permission = self.repository.update_permission(permission_id, **kwargs)
        
        if permission and updated_by:
            # 记录审计日志
            self.repository.create_audit_log(
                user_id=updated_by,
                action='update_permission',
                permission_id=str(permission_id),
                old_value=old_value,
                new_value=permission.to_dict()
            )
        
        # 清除缓存
        self._invalidate_permission_cache()
        
        return permission
    
    def delete_permission(self, permission_id: str, deleted_by: Optional[str] = None) -> bool:
        """删除权限
        
        Args:
            permission_id: 权限ID
            deleted_by: 删除者ID
            
        Returns:
            bool: 是否删除成功
        """
        # 获取权限用于审计
        permission = self.repository.get_permission_by_id(permission_id)
        if not permission:
            return False
        
        old_value = permission.to_dict()
        
        # 执行删除
        result = self.repository.delete_permission(permission_id)
        
        if result and deleted_by:
            # 记录审计日志
            self.repository.create_audit_log(
                user_id=deleted_by,
                action='delete_permission',
                permission_id=str(permission_id),
                old_value=old_value
            )
        
        # 清除缓存
        self._invalidate_permission_cache()
        
        return result
    
    # ========================================================================
    # 角色管理
    # ========================================================================
    
    def create_role(self,
                    name: str,
                    display_name: Optional[str] = None,
                    description: Optional[str] = None,
                    is_system: bool = False,
                    parent_role_id: Optional[str] = None,
                    level: int = 0,
                    max_users: Optional[int] = None,
                    metadata: Optional[Dict] = None,
                    created_by: Optional[str] = None) -> Role:
        """创建角色
        
        创建新的系统角色，支持角色继承。
        
        Args:
            name: 角色名称，必须唯一
            display_name: 显示名称
            description: 角色描述
            is_system: 是否为系统内置角色
            parent_role_id: 父角色ID（用于角色继承）
            level: 角色级别（数值越大权限越高）
            max_users: 最大用户数限制
            metadata: 元数据
            created_by: 创建者ID
            
        Returns:
            Role: 创建的角色对象
            
        Raises:
            ValueError: 如果角色名称已存在
        """
        role = self.repository.create_role(
            name=name,
            display_name=display_name,
            description=description,
            is_system=is_system,
            parent_role_id=parent_role_id,
            level=level,
            max_users=max_users,
            metadata=metadata
        )
        
        # 记录审计日志
        if created_by:
            self.repository.create_audit_log(
                user_id=created_by,
                action='create_role',
                role_id=str(role.id),
                new_value=role.to_dict()
            )
        
        logger.info(f"Created role: {name}")
        return role
    
    def get_role(self, role_id: str) -> Optional[Role]:
        """获取角色详情
        
        Args:
            role_id: 角色ID
            
        Returns:
            Optional[Role]: 角色对象
        """
        return self.repository.get_role_by_id(role_id)
    
    def get_role_by_name(self, name: str) -> Optional[Role]:
        """根据名称获取角色
        
        Args:
            name: 角色名称
            
        Returns:
            Optional[Role]: 角色对象
        """
        return self.repository.get_role_by_name(name)
    
    def list_roles(self,
                   is_active: Optional[bool] = True,
                   is_system: Optional[bool] = None,
                   page: int = 1,
                   page_size: int = 50) -> Tuple[List[Role], int]:
        """列出角色
        
        Args:
            is_active: 过滤是否启用
            is_system: 过滤是否系统内置
            page: 页码
            page_size: 每页大小
            
        Returns:
            Tuple[List[Role], int]: 角色列表和总数
        """
        return self.repository.list_roles(
            is_active=is_active,
            is_system=is_system,
            page=page,
            page_size=page_size
        )
    
    def update_role(self,
                    role_id: str,
                    updated_by: Optional[str] = None,
                    **kwargs) -> Optional[Role]:
        """更新角色
        
        Args:
            role_id: 角色ID
            updated_by: 更新者ID
            **kwargs: 要更新的字段
            
        Returns:
            Optional[Role]: 更新后的角色对象
        """
        old_role = self.repository.get_role_by_id(role_id)
        if not old_role:
            return None
        
        old_value = old_role.to_dict()
        role = self.repository.update_role(role_id, **kwargs)
        
        if role and updated_by:
            self.repository.create_audit_log(
                user_id=updated_by,
                action='update_role',
                role_id=str(role_id),
                old_value=old_value,
                new_value=role.to_dict()
            )
        
        self._invalidate_permission_cache()
        return role
    
    def delete_role(self, role_id: str, deleted_by: Optional[str] = None) -> bool:
        """删除角色
        
        Args:
            role_id: 角色ID
            deleted_by: 删除者ID
            
        Returns:
            bool: 是否删除成功
        """
        role = self.repository.get_role_by_id(role_id)
        if not role:
            return False
        
        old_value = role.to_dict()
        result = self.repository.delete_role(role_id)
        
        if result and deleted_by:
            self.repository.create_audit_log(
                user_id=deleted_by,
                action='delete_role',
                role_id=str(role_id),
                old_value=old_value
            )
        
        self._invalidate_permission_cache()
        return result
    
    # ========================================================================
    # 角色-权限关联
    # ========================================================================
    
    def assign_permission_to_role(self,
                                   role_id: str,
                                   permission_id: str,
                                   granted_by: Optional[str] = None) -> bool:
        """为角色分配权限
        
        Args:
            role_id: 角色ID
            permission_id: 权限ID
            granted_by: 授权人ID
            
        Returns:
            bool: 是否分配成功
        """
        result = self.repository.assign_permission_to_role(role_id, permission_id, granted_by)
        
        if result and granted_by:
            self.repository.create_audit_log(
                user_id=granted_by,
                action='assign_permission_to_role',
                role_id=str(role_id),
                permission_id=str(permission_id),
                new_value={'role_id': role_id, 'permission_id': permission_id}
            )
        
        self._invalidate_permission_cache()
        return result
    
    def remove_permission_from_role(self,
                                     role_id: str,
                                     permission_id: str,
                                     removed_by: Optional[str] = None) -> bool:
        """移除角色的权限
        
        Args:
            role_id: 角色ID
            permission_id: 权限ID
            removed_by: 操作人ID
            
        Returns:
            bool: 是否移除成功
        """
        result = self.repository.remove_permission_from_role(role_id, permission_id)
        
        if result and removed_by:
            self.repository.create_audit_log(
                user_id=removed_by,
                action='remove_permission_from_role',
                role_id=str(role_id),
                permission_id=str(permission_id),
                old_value={'role_id': role_id, 'permission_id': permission_id}
            )
        
        self._invalidate_permission_cache()
        return result
    
    def get_role_permissions(self, role_id: str, include_inherited: bool = True) -> List[Permission]:
        """获取角色的所有权限
        
        Args:
            role_id: 角色ID
            include_inherited: 是否包含继承的权限
            
        Returns:
            List[Permission]: 权限列表
        """
        if include_inherited:
            return self.repository.get_role_permissions_with_inherited(role_id)
        return self.repository.get_role_permissions(role_id)
    
    # ========================================================================
    # 用户-角色关联
    # ========================================================================
    
    def assign_role_to_user(self,
                            user_id: str,
                            role_id: str,
                            assigned_by: Optional[str] = None,
                            expires_at: Optional[datetime] = None,
                            conditions: Optional[Dict] = None,
                            scope: str = 'global') -> UserRole:
        """为用户分配角色
        
        Args:
            user_id: 用户ID
            role_id: 角色ID
            assigned_by: 分配者ID
            expires_at: 过期时间
            conditions: 关联条件
            scope: 权限范围
            
        Returns:
            UserRole: 创建的用户角色关联
        """
        user_role = self.repository.assign_role_to_user(
            user_id=user_id,
            role_id=role_id,
            assigned_by=assigned_by,
            expires_at=expires_at,
            conditions=conditions,
            scope=scope
        )
        
        if assigned_by:
            self.repository.create_audit_log(
                user_id=assigned_by,
                action='assign_role_to_user',
                target_user_id=user_id,
                role_id=str(role_id),
                new_value=user_role.to_dict()
            )
        
        # Agent分析：检查权限分配是否合理
        self._analyze_role_assignment(user_id, role_id, assigned_by)
        
        self._invalidate_user_permission_cache(user_id)
        return user_role
    
    def remove_role_from_user(self,
                               user_id: str,
                               role_id: str,
                               removed_by: Optional[str] = None) -> bool:
        """移除用户的角色
        
        Args:
            user_id: 用户ID
            role_id: 角色ID
            removed_by: 操作人ID
            
        Returns:
            bool: 是否移除成功
        """
        result = self.repository.remove_role_from_user(user_id, role_id)
        
        if result and removed_by:
            self.repository.create_audit_log(
                user_id=removed_by,
                action='remove_role_from_user',
                target_user_id=user_id,
                role_id=str(role_id),
                old_value={'user_id': user_id, 'role_id': role_id}
            )
        
        self._invalidate_user_permission_cache(user_id)
        return result
    
    def get_user_roles(self, user_id: str, active_only: bool = True) -> List[Role]:
        """获取用户的所有角色
        
        Args:
            user_id: 用户ID
            active_only: 是否只返回有效的角色
            
        Returns:
            List[Role]: 角色列表
        """
        return self.repository.get_user_roles(user_id, active_only)
    
    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """获取用户的所有权限
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[Permission]: 权限列表
        """
        return self.repository.get_user_permissions(user_id)
    
    def get_user_permissions_set(self, user_id: str) -> Set[str]:
        """获取用户权限集合（格式：resource:action）
        
        使用缓存提高性能。
        
        Args:
            user_id: 用户ID
            
        Returns:
            Set[str]: 权限集合
        """
        # 检查缓存
        if user_id in self._permission_cache:
            cached_set, cached_time = self._permission_cache[user_id]
            if datetime.utcnow() - cached_time < timedelta(seconds=self._cache_ttl):
                return cached_set
        
        # 获取权限
        permission_set = self.repository.get_user_permission_set(user_id)
        
        # 更新缓存
        self._permission_cache[user_id] = (permission_set, datetime.utcnow())
        
        return permission_set
    
    # ========================================================================
    # 权限验证
    # ========================================================================
    
    def check_user_permission(self,
                               user_id: str,
                               resource: str,
                               action: str,
                               ip_address: Optional[str] = None,
                               user_agent: Optional[str] = None,
                               record_access: bool = True) -> PermissionCheckResult:
        """检查用户权限（带智能分析）
        
        执行权限验证，支持策略匹配、风险评估和审计记录。
        
        Args:
            user_id: 用户ID
            resource: 资源类型
            action: 操作类型
            ip_address: 请求IP地址
            user_agent: 用户代理
            record_access: 是否记录访问日志
            
        Returns:
            PermissionCheckResult: 权限检查结果
            
        Example:
            >>> result = service.check_user_permission(
            ...     user_id="user123",
            ...     resource="datasets",
            ...     action="delete"
            ... )
            >>> if result.allowed:
            ...     # 执行操作
            ...     pass
            >>> elif result.requires_mfa:
            ...     # 需要MFA验证
            ...     pass
        """
        start_time = time.time()
        
        # 检查基本权限
        permission_set = self.get_user_permissions_set(user_id)
        permission_key = f"{resource}:{action}"
        has_basic_permission = permission_key in permission_set
        
        # 获取适用的策略
        policies = self.repository.get_applicable_policies(user_id, resource, action)
        matched_policies = [p.to_dict() for p in policies]
        
        # 评估策略
        allowed = has_basic_permission
        reason = "Permission granted" if allowed else "Permission denied"
        risk_level = 'low'
        requires_mfa = False
        
        for policy in policies:
            if policy.effect == 'deny':
                allowed = False
                reason = f"Denied by policy: {policy.name}"
                break
            elif policy.effect == 'allow' and not has_basic_permission:
                allowed = True
                reason = f"Allowed by policy: {policy.name}"
        
        # 获取权限详情检查是否需要MFA
        if allowed:
            permission = self.repository.get_permission_by_resource_action(resource, action)
            if permission:
                risk_level = permission.risk_level
                requires_mfa = permission.requires_mfa
        
        latency_ms = (time.time() - start_time) * 1000
        
        # 记录访问日志
        if record_access:
            self.repository.create_access_log(
                user_id=user_id,
                resource=resource,
                action=action,
                result=allowed,
                reason=reason,
                ip_address=ip_address,
                user_agent=user_agent,
                latency_ms=latency_ms
            )
        
        return PermissionCheckResult(
            allowed=allowed,
            reason=reason,
            matched_policies=matched_policies,
            risk_level=risk_level,
            requires_mfa=requires_mfa
        )
    
    def has_permission(self, user_id: str, required_permission: str) -> bool:
        """简单检查用户是否具有特定权限
        
        Args:
            user_id: 用户ID
            required_permission: 权限字符串（格式：resource:action）
            
        Returns:
            bool: 是否具有权限
        """
        if ":" not in required_permission:
            return False
        
        resource, action = required_permission.split(":", 1)
        result = self.check_user_permission(user_id, resource, action, record_access=False)
        return result.allowed
    
    # ========================================================================
    # 资源权限
    # ========================================================================
    
    def grant_resource_permission(self,
                                   resource_type: str,
                                   resource_id: str,
                                   user_id: str,
                                   permission_type: str = 'viewer',
                                   permissions: Optional[List[str]] = None,
                                   granted_by: Optional[str] = None,
                                   expires_at: Optional[datetime] = None) -> ResourcePermission:
        """授予资源权限
        
        Args:
            resource_type: 资源类型
            resource_id: 资源ID
            user_id: 用户ID
            permission_type: 权限类型 (owner, editor, viewer, custom)
            permissions: 具体权限列表
            granted_by: 授权人ID
            expires_at: 过期时间
            
        Returns:
            ResourcePermission: 创建的资源权限
        """
        rp = self.repository.grant_resource_permission(
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            permission_type=permission_type,
            permissions=permissions,
            granted_by=granted_by,
            expires_at=expires_at
        )
        
        if granted_by:
            self.repository.create_audit_log(
                user_id=granted_by,
                action='grant_resource_permission',
                target_user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                new_value=rp.to_dict()
            )
        
        return rp
    
    def revoke_resource_permission(self,
                                    resource_type: str,
                                    resource_id: str,
                                    user_id: str,
                                    revoked_by: Optional[str] = None) -> bool:
        """撤销资源权限
        
        Args:
            resource_type: 资源类型
            resource_id: 资源ID
            user_id: 用户ID
            revoked_by: 操作人ID
            
        Returns:
            bool: 是否撤销成功
        """
        result = self.repository.revoke_resource_permission(resource_type, resource_id, user_id)
        
        if result and revoked_by:
            self.repository.create_audit_log(
                user_id=revoked_by,
                action='revoke_resource_permission',
                target_user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id
            )
        
        return result
    
    def check_resource_permission(self,
                                   resource_type: str,
                                   resource_id: str,
                                   user_id: str,
                                   required_permission: Optional[str] = None) -> bool:
        """检查资源权限
        
        Args:
            resource_type: 资源类型
            resource_id: 资源ID
            user_id: 用户ID
            required_permission: 所需权限
            
        Returns:
            bool: 是否具有权限
        """
        return self.repository.check_resource_permission(
            resource_type, resource_id, user_id, required_permission
        )
    
    # ========================================================================
    # Agent 智能分析
    # ========================================================================
    
    def analyze_user_permissions(self, user_id: str) -> PermissionAnalysis:
        """分析用户权限（Agent智能分析）
        
        使用AI Agent分析用户的权限配置，提供优化建议。
        
        Args:
            user_id: 用户ID
            
        Returns:
            PermissionAnalysis: 分析结果
        """
        start_time = time.time()
        
        # 获取用户权限和角色
        permissions = self.get_user_permissions(user_id)
        roles = self.get_user_roles(user_id)
        
        # 分类统计
        categories: Dict[str, int] = {}
        risk_counts: Dict[str, int] = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        high_risk_permissions = []
        
        for perm in permissions:
            resource = perm.resource
            categories[resource] = categories.get(resource, 0) + 1
            risk_counts[perm.risk_level] = risk_counts.get(perm.risk_level, 0) + 1
            if perm.risk_level in ['high', 'critical']:
                high_risk_permissions.append(perm.to_dict())
        
        # 角色摘要
        role_summary = []
        for role in roles:
            role_perms = self.get_role_permissions(str(role.id))
            role_summary.append({
                'role_id': str(role.id),
                'role_name': role.name,
                'permission_count': len(role_perms),
                'level': role.level
            })
        
        # 获取使用模式
        access_stats = self.repository.get_access_statistics(
            user_id=user_id,
            start_time=datetime.utcnow() - timedelta(days=30)
        )
        
        # 风险评估
        total_risk_score = (
            risk_counts.get('low', 0) * 0.1 +
            risk_counts.get('medium', 0) * 0.3 +
            risk_counts.get('high', 0) * 0.6 +
            risk_counts.get('critical', 0) * 1.0
        )
        normalized_risk = min(1.0, total_risk_score / max(1, len(permissions)))
        
        risk_assessment = {
            'overall_risk_score': normalized_risk,
            'risk_level': self._calculate_risk_level(normalized_risk),
            'risk_distribution': risk_counts,
            'high_risk_permissions': high_risk_permissions,
            'recommendation': self._get_risk_recommendation(normalized_risk)
        }
        
        # 生成建议
        recommendations = self._generate_permission_recommendations(
            user_id, permissions, roles, access_stats
        )
        
        # Agent 洞察
        agent_insights = self._get_agent_insights(user_id, permissions, access_stats)
        
        # 记录推理过程
        latency_ms = (time.time() - start_time) * 1000
        self.repository.create_agent_reasoning(
            trigger='analyze_user_permissions',
            user_id=user_id,
            context={
                'total_permissions': len(permissions),
                'total_roles': len(roles)
            },
            reasoning_steps=[
                {'step': 'collect_permissions', 'count': len(permissions)},
                {'step': 'categorize', 'categories': len(categories)},
                {'step': 'assess_risk', 'score': normalized_risk},
                {'step': 'generate_recommendations', 'count': len(recommendations)}
            ],
            conclusion=f"Analyzed {len(permissions)} permissions with risk score {normalized_risk:.2f}",
            confidence=0.85,
            latency_ms=latency_ms
        )
        
        return PermissionAnalysis(
            user_id=user_id,
            total_permissions=len(permissions),
            permission_categories=categories,
            role_summary=role_summary,
            risk_assessment=risk_assessment,
            recommendations=recommendations,
            usage_patterns=access_stats,
            agent_insights=agent_insights
        )
    
    def get_permission_recommendations(self,
                                        user_id: str,
                                        limit: int = 10) -> List[PermissionRecommendation]:
        """获取权限建议
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            
        Returns:
            List[PermissionRecommendation]: 建议列表
        """
        recommendations, _ = self.repository.list_recommendations(
            user_id=user_id,
            status='pending',
            page_size=limit
        )
        return recommendations
    
    def review_recommendation(self,
                               recommendation_id: str,
                               status: str,
                               reviewed_by: str,
                               review_notes: Optional[str] = None,
                               apply_if_accepted: bool = True) -> Optional[PermissionRecommendation]:
        """审核权限建议
        
        Args:
            recommendation_id: 建议ID
            status: 新状态 (accepted, rejected)
            reviewed_by: 审核人ID
            review_notes: 审核备注
            apply_if_accepted: 如果接受是否自动应用
            
        Returns:
            Optional[PermissionRecommendation]: 更新后的建议
        """
        rec = self.repository.review_recommendation(
            recommendation_id, status, reviewed_by, review_notes
        )
        
        if rec and status == 'accepted' and apply_if_accepted:
            # 自动应用建议
            self._apply_recommendation(rec, reviewed_by)
        
        return rec
    
    def detect_permission_anomalies(self, user_id: str) -> List[Dict[str, Any]]:
        """检测权限异常
        
        使用Agent分析检测权限配置中的异常。
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[Dict]: 异常列表
        """
        anomalies = []
        
        # 获取用户权限
        permissions = self.get_user_permissions(user_id)
        roles = self.get_user_roles(user_id)
        
        # 检查权限过多
        if len(permissions) > 50:
            anomalies.append({
                'type': 'excessive_permissions',
                'severity': 'medium',
                'description': f"User has {len(permissions)} permissions, which may violate least privilege principle",
                'recommendation': 'Review and reduce permissions to minimum required'
            })
        
        # 检查高风险权限
        high_risk_count = sum(1 for p in permissions if p.risk_level in ['high', 'critical'])
        if high_risk_count > 5:
            anomalies.append({
                'type': 'high_risk_permissions',
                'severity': 'high',
                'description': f"User has {high_risk_count} high/critical risk permissions",
                'recommendation': 'Review necessity of high-risk permissions'
            })
        
        # 检查角色冲突
        role_levels = [r.level for r in roles]
        if len(set(role_levels)) > 1 and max(role_levels) - min(role_levels) > 5:
            anomalies.append({
                'type': 'role_level_conflict',
                'severity': 'medium',
                'description': 'User has roles with significantly different privilege levels',
                'recommendation': 'Review role assignments for consistency'
            })
        
        # 检查未使用权限（基于访问日志）
        access_stats = self.repository.get_access_statistics(
            user_id=user_id,
            start_time=datetime.utcnow() - timedelta(days=30)
        )
        
        used_resources = set(access_stats.get('by_resource', {}).keys())
        all_resources = {p.resource for p in permissions}
        unused_resources = all_resources - used_resources
        
        if unused_resources and len(unused_resources) / len(all_resources) > 0.5:
            anomalies.append({
                'type': 'unused_permissions',
                'severity': 'low',
                'description': f"Over 50% of granted permissions ({len(unused_resources)} resources) appear unused in the last 30 days",
                'recommendation': 'Consider revoking unused permissions',
                'details': {'unused_resources': list(unused_resources)}
            })
        
        # 记录异常检测
        if anomalies:
            self._create_permission_memory(
                memory_type='anomaly_detected',
                content=f"Detected {len(anomalies)} permission anomalies for user {user_id}",
                user_id=user_id,
                metadata={'anomalies': anomalies}
            )
        
        return anomalies
    
    def optimize_permissions(self, user_id: str, dry_run: bool = True) -> Dict[str, Any]:
        """优化用户权限（最小权限原则）
        
        分析用户权限使用情况，建议移除不必要的权限。
        
        Args:
            user_id: 用户ID
            dry_run: 是否仅模拟，不实际执行
            
        Returns:
            Dict: 优化结果
        """
        # 获取当前权限
        permissions = self.get_user_permissions(user_id)
        roles = self.get_user_roles(user_id)
        
        # 获取访问日志统计
        access_stats = self.repository.get_access_statistics(
            user_id=user_id,
            start_time=datetime.utcnow() - timedelta(days=90)
        )
        
        used_permissions = set()
        for resource, count in access_stats.get('by_resource', {}).items():
            if count > 0:
                for p in permissions:
                    if p.resource == resource:
                        used_permissions.add(f"{p.resource}:{p.action}")
        
        all_permissions = {f"{p.resource}:{p.action}" for p in permissions}
        unused_permissions = all_permissions - used_permissions
        
        optimization_result = {
            'user_id': user_id,
            'current_permission_count': len(permissions),
            'used_permission_count': len(used_permissions),
            'unused_permissions': list(unused_permissions),
            'recommendations': [],
            'applied': False
        }
        
        # 生成建议
        for unused in unused_permissions:
            resource, action = unused.split(':', 1)
            optimization_result['recommendations'].append({
                'action': 'revoke',
                'permission': unused,
                'reason': f"Permission {unused} has not been used in the last 90 days"
            })
            
            # 创建正式建议记录
            perm = self.repository.get_permission_by_resource_action(resource, action)
            if perm:
                self.repository.create_recommendation(
                    user_id=user_id,
                    recommendation_type='revoke',
                    reason=f"Permission '{unused}' has not been used in the last 90 days",
                    permission_id=str(perm.id),
                    confidence=0.75,
                    risk_assessment={
                        'impact': 'low',
                        'reversible': True
                    }
                )
        
        # 如果不是dry_run，执行优化
        if not dry_run and unused_permissions:
            # 实际上我们通常不会自动撤销权限，而是生成建议等待审核
            optimization_result['applied'] = False
            optimization_result['message'] = 'Recommendations created for review'
        
        # 记录优化过程
        self.repository.create_agent_reasoning(
            trigger='optimize_permissions',
            user_id=user_id,
            context={
                'total_permissions': len(permissions),
                'analysis_period_days': 90
            },
            reasoning_steps=[
                {'step': 'analyze_usage', 'used': len(used_permissions), 'unused': len(unused_permissions)},
                {'step': 'generate_recommendations', 'count': len(optimization_result['recommendations'])}
            ],
            conclusion=f"Identified {len(unused_permissions)} potentially unnecessary permissions",
            confidence=0.8 if len(permissions) > 10 else 0.6
        )
        
        return optimization_result
    
    # ========================================================================
    # 审计日志
    # ========================================================================
    
    def get_audit_logs(self,
                       user_id: Optional[str] = None,
                       target_user_id: Optional[str] = None,
                       action: Optional[str] = None,
                       status: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       page: int = 1,
                       page_size: int = 50) -> Tuple[List[PermissionAuditLog], int]:
        """获取审计日志
        
        Args:
            user_id: 过滤操作用户
            target_user_id: 过滤目标用户
            action: 过滤操作类型
            status: 过滤状态
            start_time: 开始时间
            end_time: 结束时间
            page: 页码
            page_size: 每页大小
            
        Returns:
            Tuple[List[PermissionAuditLog], int]: 日志列表和总数
        """
        return self.repository.list_audit_logs(
            user_id=user_id,
            target_user_id=target_user_id,
            action=action,
            status=status,
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size
        )
    
    def get_access_statistics(self,
                               user_id: Optional[str] = None,
                               start_time: Optional[datetime] = None,
                               end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """获取访问统计
        
        Args:
            user_id: 过滤用户ID
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            Dict: 统计信息
        """
        return self.repository.get_access_statistics(user_id, start_time, end_time)
    
    # ========================================================================
    # Agent 记忆管理
    # ========================================================================
    
    def add_agent_memory(self,
                         memory_type: str,
                         content: str,
                         user_id: Optional[str] = None,
                         importance: float = 0.5,
                         metadata: Optional[Dict] = None) -> PermissionAgentMemory:
        """添加Agent记忆
        
        Args:
            memory_type: 记忆类型
            content: 记忆内容
            user_id: 关联用户ID
            importance: 重要性分数
            metadata: 元数据
            
        Returns:
            PermissionAgentMemory: 创建的记忆
        """
        return self.repository.create_agent_memory(
            memory_type=memory_type,
            content=content,
            user_id=user_id,
            importance=importance,
            metadata=metadata,
            source='permission_service'
        )
    
    def get_agent_memories(self,
                           user_id: Optional[str] = None,
                           memory_type: Optional[str] = None,
                           min_importance: float = 0.0,
                           limit: int = 100) -> List[PermissionAgentMemory]:
        """获取Agent记忆
        
        Args:
            user_id: 过滤用户ID
            memory_type: 过滤记忆类型
            min_importance: 最小重要性
            limit: 返回数量限制
            
        Returns:
            List[PermissionAgentMemory]: 记忆列表
        """
        return self.repository.get_agent_memories(user_id, memory_type, min_importance, limit)
    
    def search_agent_memories(self,
                               query: str,
                               user_id: Optional[str] = None,
                               limit: int = 20) -> List[PermissionAgentMemory]:
        """搜索Agent记忆
        
        Args:
            query: 搜索查询
            user_id: 过滤用户ID
            limit: 返回数量限制
            
        Returns:
            List[PermissionAgentMemory]: 匹配的记忆列表
        """
        # 获取所有相关记忆
        memories = self.repository.get_agent_memories(user_id=user_id, limit=limit * 5)
        
        # 简单的关键词匹配（生产环境应使用向量搜索）
        query_lower = query.lower()
        matched = []
        for memory in memories:
            if query_lower in memory.content.lower():
                matched.append(memory)
                self.repository.update_memory_access(str(memory.id))
                if len(matched) >= limit:
                    break
        
        return matched
    
    # ========================================================================
    # 策略管理
    # ========================================================================
    
    def create_policy(self,
                      name: str,
                      resource_pattern: str,
                      action_pattern: str,
                      policy_type: str = 'allow',
                      description: Optional[str] = None,
                      conditions: Optional[Dict] = None,
                      priority: int = 0,
                      effect: str = 'allow',
                      created_by: Optional[str] = None) -> PermissionPolicy:
        """创建权限策略
        
        Args:
            name: 策略名称
            resource_pattern: 资源模式
            action_pattern: 操作模式
            policy_type: 策略类型
            description: 策略描述
            conditions: 策略条件
            priority: 优先级
            effect: 效果
            created_by: 创建者ID
            
        Returns:
            PermissionPolicy: 创建的策略
        """
        policy = self.repository.create_policy(
            name=name,
            resource_pattern=resource_pattern,
            action_pattern=action_pattern,
            policy_type=policy_type,
            description=description,
            conditions=conditions,
            priority=priority,
            effect=effect
        )
        
        if created_by:
            self.repository.create_audit_log(
                user_id=created_by,
                action='create_policy',
                new_value=policy.to_dict()
            )
        
        return policy
    
    # ========================================================================
    # 私有辅助方法
    # ========================================================================
    
    def _invalidate_permission_cache(self):
        """清除所有权限缓存"""
        self._permission_cache.clear()
    
    def _invalidate_user_permission_cache(self, user_id: str):
        """清除特定用户的权限缓存"""
        if user_id in self._permission_cache:
            del self._permission_cache[user_id]
    
    def _calculate_risk_level(self, risk_score: float) -> str:
        """计算风险等级"""
        if risk_score < 0.25:
            return 'low'
        elif risk_score < 0.5:
            return 'medium'
        elif risk_score < 0.75:
            return 'high'
        return 'critical'
    
    def _get_risk_recommendation(self, risk_score: float) -> str:
        """根据风险分数获取建议"""
        if risk_score < 0.25:
            return "Permission configuration appears appropriate."
        elif risk_score < 0.5:
            return "Consider reviewing permissions with 'medium' and above risk levels."
        elif risk_score < 0.75:
            return "High risk detected. Recommend immediate review of critical permissions."
        return "Critical risk level. Urgent review and reduction of high-risk permissions required."
    
    def _generate_permission_recommendations(self,
                                              user_id: str,
                                              permissions: List[Permission],
                                              roles: List[Role],
                                              access_stats: Dict) -> List[Dict]:
        """生成权限建议"""
        recommendations = []
        
        # 基于使用模式的建议
        used_resources = set(access_stats.get('by_resource', {}).keys())
        all_resources = {p.resource for p in permissions}
        unused = all_resources - used_resources
        
        if unused:
            recommendations.append({
                'type': 'reduce_unused',
                'priority': 'medium',
                'description': f"Consider reviewing {len(unused)} unused resource permissions",
                'resources': list(unused)
            })
        
        # 基于风险的建议
        high_risk = [p for p in permissions if p.risk_level in ['high', 'critical']]
        if high_risk:
            recommendations.append({
                'type': 'review_high_risk',
                'priority': 'high',
                'description': f"Review {len(high_risk)} high/critical risk permissions",
                'permissions': [p.name for p in high_risk]
            })
        
        return recommendations
    
    def _get_agent_insights(self,
                            user_id: str,
                            permissions: List[Permission],
                            access_stats: Dict) -> Dict:
        """获取Agent洞察"""
        # 获取相关记忆
        memories = self.get_agent_memories(user_id=user_id, limit=10)
        
        insights = {
            'permission_trend': 'stable',
            'usage_efficiency': access_stats.get('allow_rate', 0),
            'historical_context': len(memories),
            'key_observations': []
        }
        
        # 分析访问模式
        total_requests = access_stats.get('total_requests', 0)
        if total_requests > 100:
            deny_rate = access_stats.get('denied', 0) / total_requests
            if deny_rate > 0.1:
                insights['key_observations'].append(
                    f"High denial rate ({deny_rate:.1%}) may indicate misconfigured permissions"
                )
        
        return insights
    
    def _create_permission_memory(self,
                                   memory_type: str,
                                   content: str,
                                   user_id: Optional[str] = None,
                                   metadata: Optional[Dict] = None):
        """创建权限相关记忆"""
        try:
            self.repository.create_agent_memory(
                memory_type=memory_type,
                content=content,
                user_id=user_id,
                importance=0.6,
                metadata=metadata,
                source='permission_service'
            )
        except Exception as e:
            logger.warning(f"Failed to create agent memory: {e}")
    
    def _analyze_role_assignment(self,
                                  user_id: str,
                                  role_id: str,
                                  assigned_by: Optional[str]):
        """分析角色分配"""
        try:
            role = self.repository.get_role_by_id(role_id)
            if not role:
                return
            
            # 检查是否是高级别角色
            if role.level >= 8:
                self._create_permission_memory(
                    memory_type='high_level_role_assigned',
                    content=f"High-level role '{role.name}' (level {role.level}) assigned to user {user_id}",
                    user_id=user_id,
                    metadata={
                        'role_id': role_id,
                        'role_name': role.name,
                        'role_level': role.level,
                        'assigned_by': assigned_by
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to analyze role assignment: {e}")
    
    def _apply_recommendation(self, recommendation: PermissionRecommendation, applied_by: str):
        """应用权限建议"""
        try:
            if recommendation.recommendation_type == 'grant':
                if recommendation.role_id:
                    self.assign_role_to_user(
                        user_id=str(recommendation.user_id),
                        role_id=str(recommendation.role_id),
                        assigned_by=applied_by
                    )
            elif recommendation.recommendation_type == 'revoke':
                if recommendation.role_id:
                    self.remove_role_from_user(
                        user_id=str(recommendation.user_id),
                        role_id=str(recommendation.role_id),
                        removed_by=applied_by
                    )
        except Exception as e:
            logger.error(f"Failed to apply recommendation: {e}")

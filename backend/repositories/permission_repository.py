"""权限数据访问层

提供权限相关数据的CRUD操作，包括：
- 权限管理: Permission CRUD
- 角色管理: Role CRUD
- 用户角色关联: UserRole操作
- 审计日志: PermissionAuditLog记录
- Agent记忆: PermissionAgentMemory操作
- 策略管理: PermissionPolicy操作
- 资源权限: ResourcePermission操作
"""

import logging
from typing import List, Optional, Dict, Any, Set, Tuple
from datetime import datetime, timedelta
from sqlalchemy import and_, or_, func, text, desc
from sqlalchemy.orm import Session, joinedload
import uuid

from backend.modules.database.manager import get_database_manager
from backend.schemas.permission_models import (
    Permission, Role, UserRole, role_permissions,
    PermissionAuditLog, PermissionAccessLog,
    PermissionAgentMemory, PermissionAgentReasoning,
    PermissionPolicy, ResourcePermission, PermissionRecommendation
)

logger = logging.getLogger(__name__)


def get_permission_repository():
    """获取权限仓库实例
    
    Returns:
        PermissionRepository: 权限仓库实例
    """
    return PermissionRepository()


class PermissionRepository:
    """权限数据仓库
    
    处理所有权限相关的数据库操作。
    
    Attributes:
        db_manager: 数据库管理器实例
    """
    
    def __init__(self, db_manager=None):
        """初始化仓库
        
        Args:
            db_manager: 数据库管理器实例，如果为None则使用默认管理器
        """
        self.db_manager = db_manager or get_database_manager()
    
    # ========================================================================
    # Permission CRUD
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
                          audit_level: str = 'basic') -> Permission:
        """创建权限
        
        Args:
            name: 权限名称
            resource: 资源类型
            action: 操作类型
            description: 权限描述
            is_system: 是否系统内置
            scope: 权限范围
            conditions: 权限条件
            priority: 优先级
            risk_level: 风险等级
            requires_mfa: 是否需要MFA
            audit_level: 审计级别
            
        Returns:
            Permission: 创建的权限对象
            
        Raises:
            ValueError: 如果权限名称已存在
        """
        with self.db_manager.get_session() as db:
            # 检查名称是否已存在
            existing = db.query(Permission).filter(Permission.name == name).first()
            if existing:
                raise ValueError(f"Permission name '{name}' already exists")
            
            permission = Permission(
                id=str(uuid.uuid4()),
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
            db.add(permission)
            db.commit()
            db.refresh(permission)
            logger.info(f"Created permission: {name}")
            return permission
    
    def get_permission_by_id(self, permission_id: str) -> Optional[Permission]:
        """根据ID获取权限
        
        Args:
            permission_id: 权限ID
            
        Returns:
            Optional[Permission]: 权限对象，不存在则返回None
        """
        with self.db_manager.get_session() as db:
            return db.query(Permission).filter(Permission.id == permission_id).first()
    
    def get_permission_by_name(self, name: str) -> Optional[Permission]:
        """根据名称获取权限
        
        Args:
            name: 权限名称
            
        Returns:
            Optional[Permission]: 权限对象，不存在则返回None
        """
        with self.db_manager.get_session() as db:
            return db.query(Permission).filter(Permission.name == name).first()
    
    def get_permission_by_resource_action(self, resource: str, action: str) -> Optional[Permission]:
        """根据资源和操作获取权限
        
        Args:
            resource: 资源类型
            action: 操作类型
            
        Returns:
            Optional[Permission]: 权限对象，不存在则返回None
        """
        with self.db_manager.get_session() as db:
            return db.query(Permission).filter(
                and_(Permission.resource == resource, Permission.action == action)
            ).first()
    
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
        with self.db_manager.get_session() as db:
            query = db.query(Permission)
            
            if resource:
                query = query.filter(Permission.resource == resource)
            if action:
                query = query.filter(Permission.action == action)
            if is_active is not None:
                query = query.filter(Permission.is_active == is_active)
            if risk_level:
                query = query.filter(Permission.risk_level == risk_level)
            
            total = query.count()
            permissions = query.order_by(Permission.priority.desc(), Permission.name).offset(
                (page - 1) * page_size
            ).limit(page_size).all()
            
            return permissions, total
    
    def update_permission(self, permission_id: str, **kwargs) -> Optional[Permission]:
        """更新权限
        
        Args:
            permission_id: 权限ID
            **kwargs: 要更新的字段
            
        Returns:
            Optional[Permission]: 更新后的权限对象
        """
        with self.db_manager.get_session() as db:
            permission = db.query(Permission).filter(Permission.id == permission_id).first()
            if not permission:
                return None
            
            for key, value in kwargs.items():
                if hasattr(permission, key):
                    setattr(permission, key, value)
            
            permission.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(permission)
            logger.info(f"Updated permission: {permission_id}")
            return permission
    
    def delete_permission(self, permission_id: str) -> bool:
        """删除权限
        
        Args:
            permission_id: 权限ID
            
        Returns:
            bool: 是否删除成功
        """
        with self.db_manager.get_session() as db:
            permission = db.query(Permission).filter(Permission.id == permission_id).first()
            if not permission:
                return False
            
            # 检查是否为系统权限
            if permission.is_system:
                raise ValueError("Cannot delete system permission")
            
            db.delete(permission)
            db.commit()
            logger.info(f"Deleted permission: {permission_id}")
            return True
    
    # ========================================================================
    # Role CRUD
    # ========================================================================
    
    def create_role(self,
                    name: str,
                    display_name: Optional[str] = None,
                    description: Optional[str] = None,
                    is_system: bool = False,
                    parent_role_id: Optional[str] = None,
                    level: int = 0,
                    max_users: Optional[int] = None,
                    metadata: Optional[Dict] = None) -> Role:
        """创建角色
        
        Args:
            name: 角色名称
            display_name: 显示名称
            description: 角色描述
            is_system: 是否系统内置
            parent_role_id: 父角色ID
            level: 角色级别
            max_users: 最大用户数
            metadata: 元数据
            
        Returns:
            Role: 创建的角色对象
            
        Raises:
            ValueError: 如果角色名称已存在
        """
        with self.db_manager.get_session() as db:
            # 检查名称是否已存在
            existing = db.query(Role).filter(Role.name == name).first()
            if existing:
                raise ValueError(f"Role name '{name}' already exists")
            
            role = Role(
                id=str(uuid.uuid4()),
                name=name,
                display_name=display_name or name,
                description=description,
                is_system=is_system,
                parent_role_id=parent_role_id,
                level=level,
                max_users=max_users,
                extra_data=metadata
            )
            db.add(role)
            db.commit()
            db.refresh(role)
            logger.info(f"Created role: {name}")
            return role
    
    def get_role_by_id(self, role_id: str) -> Optional[Role]:
        """根据ID获取角色
        
        Args:
            role_id: 角色ID
            
        Returns:
            Optional[Role]: 角色对象，不存在则返回None
        """
        with self.db_manager.get_session() as db:
            return db.query(Role).options(
                joinedload(Role.permissions)
            ).filter(Role.id == role_id).first()
    
    def get_role_by_name(self, name: str) -> Optional[Role]:
        """根据名称获取角色
        
        Args:
            name: 角色名称
            
        Returns:
            Optional[Role]: 角色对象，不存在则返回None
        """
        with self.db_manager.get_session() as db:
            return db.query(Role).options(
                joinedload(Role.permissions)
            ).filter(Role.name == name).first()
    
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
        with self.db_manager.get_session() as db:
            query = db.query(Role)
            
            if is_active is not None:
                query = query.filter(Role.is_active == is_active)
            if is_system is not None:
                query = query.filter(Role.is_system == is_system)
            
            total = query.count()
            roles = query.order_by(Role.level.desc(), Role.name).offset(
                (page - 1) * page_size
            ).limit(page_size).all()
            
            return roles, total
    
    def update_role(self, role_id: str, **kwargs) -> Optional[Role]:
        """更新角色
        
        Args:
            role_id: 角色ID
            **kwargs: 要更新的字段
            
        Returns:
            Optional[Role]: 更新后的角色对象
        """
        with self.db_manager.get_session() as db:
            role = db.query(Role).filter(Role.id == role_id).first()
            if not role:
                return None
            
            for key, value in kwargs.items():
                if hasattr(role, key):
                    setattr(role, key, value)
            
            role.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(role)
            logger.info(f"Updated role: {role_id}")
            return role
    
    def delete_role(self, role_id: str) -> bool:
        """删除角色
        
        Args:
            role_id: 角色ID
            
        Returns:
            bool: 是否删除成功
        """
        with self.db_manager.get_session() as db:
            role = db.query(Role).filter(Role.id == role_id).first()
            if not role:
                return False
            
            # 检查是否为系统角色
            if role.is_system:
                raise ValueError("Cannot delete system role")
            
            db.delete(role)
            db.commit()
            logger.info(f"Deleted role: {role_id}")
            return True
    
    # ========================================================================
    # Role-Permission 关联
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
        with self.db_manager.get_session() as db:
            # 检查关联是否已存在
            existing = db.execute(
                role_permissions.select().where(
                    and_(
                        role_permissions.c.role_id == role_id,
                        role_permissions.c.permission_id == permission_id
                    )
                )
            ).first()
            
            if existing:
                return True  # 已存在，视为成功
            
            # 插入关联
            db.execute(role_permissions.insert().values(
                role_id=role_id,
                permission_id=permission_id,
                granted_by=granted_by,
                granted_at=datetime.utcnow()
            ))
            db.commit()
            logger.info(f"Assigned permission {permission_id} to role {role_id}")
            return True
    
    def remove_permission_from_role(self, role_id: str, permission_id: str) -> bool:
        """移除角色的权限
        
        Args:
            role_id: 角色ID
            permission_id: 权限ID
            
        Returns:
            bool: 是否移除成功
        """
        with self.db_manager.get_session() as db:
            result = db.execute(
                role_permissions.delete().where(
                    and_(
                        role_permissions.c.role_id == role_id,
                        role_permissions.c.permission_id == permission_id
                    )
                )
            )
            db.commit()
            return result.rowcount > 0
    
    def get_role_permissions(self, role_id: str) -> List[Permission]:
        """获取角色的所有权限
        
        Args:
            role_id: 角色ID
            
        Returns:
            List[Permission]: 权限列表
        """
        with self.db_manager.get_session() as db:
            return db.query(Permission).join(
                role_permissions,
                Permission.id == role_permissions.c.permission_id
            ).filter(role_permissions.c.role_id == role_id).all()
    
    def get_role_permissions_with_inherited(self, role_id: str) -> List[Permission]:
        """获取角色的所有权限（包括继承的）
        
        Args:
            role_id: 角色ID
            
        Returns:
            List[Permission]: 权限列表
        """
        with self.db_manager.get_session() as db:
            # 获取角色
            role = db.query(Role).filter(Role.id == role_id).first()
            if not role:
                return []
            
            # 收集所有角色ID（包括父角色）
            role_ids = [role_id]
            current_role = role
            while current_role.parent_role_id:
                role_ids.append(current_role.parent_role_id)
                current_role = db.query(Role).filter(
                    Role.id == current_role.parent_role_id
                ).first()
                if not current_role:
                    break
            
            # 获取所有权限
            return db.query(Permission).join(
                role_permissions,
                Permission.id == role_permissions.c.permission_id
            ).filter(role_permissions.c.role_id.in_(role_ids)).distinct().all()
    
    # ========================================================================
    # UserRole 关联
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
        with self.db_manager.get_session() as db:
            # 检查关联是否已存在
            existing = db.query(UserRole).filter(
                and_(UserRole.user_id == user_id, UserRole.role_id == role_id)
            ).first()
            
            if existing:
                # 更新现有关联
                existing.is_active = True
                existing.assigned_by = assigned_by
                existing.assigned_at = datetime.utcnow()
                existing.expires_at = expires_at
                existing.conditions = conditions
                existing.scope = scope
                db.commit()
                db.refresh(existing)
                return existing
            
            # 创建新关联
            user_role = UserRole(
                id=str(uuid.uuid4()),
                user_id=user_id,
                role_id=role_id,
                assigned_by=assigned_by,
                assigned_at=datetime.utcnow(),
                expires_at=expires_at,
                is_active=True,
                conditions=conditions,
                scope=scope
            )
            db.add(user_role)
            db.commit()
            db.refresh(user_role)
            logger.info(f"Assigned role {role_id} to user {user_id}")
            return user_role
    
    def remove_role_from_user(self, user_id: str, role_id: str) -> bool:
        """移除用户的角色
        
        Args:
            user_id: 用户ID
            role_id: 角色ID
            
        Returns:
            bool: 是否移除成功
        """
        with self.db_manager.get_session() as db:
            user_role = db.query(UserRole).filter(
                and_(UserRole.user_id == user_id, UserRole.role_id == role_id)
            ).first()
            
            if user_role:
                db.delete(user_role)
                db.commit()
                logger.info(f"Removed role {role_id} from user {user_id}")
                return True
            return False
    
    def get_user_roles(self, user_id: str, active_only: bool = True) -> List[Role]:
        """获取用户的所有角色
        
        Args:
            user_id: 用户ID
            active_only: 是否只返回有效的角色
            
        Returns:
            List[Role]: 角色列表
        """
        with self.db_manager.get_session() as db:
            query = db.query(Role).join(UserRole).filter(UserRole.user_id == user_id)
            
            if active_only:
                query = query.filter(
                    and_(
                        UserRole.is_active == True,
                        or_(
                            UserRole.expires_at.is_(None),
                            UserRole.expires_at > datetime.utcnow()
                        )
                    )
                )
            
            return query.all()
    
    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """获取用户的所有权限（通过角色）
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[Permission]: 权限列表
        """
        with self.db_manager.get_session() as db:
            # 获取用户的所有有效角色
            user_roles = db.query(UserRole).filter(
                and_(
                    UserRole.user_id == user_id,
                    UserRole.is_active == True,
                    or_(
                        UserRole.expires_at.is_(None),
                        UserRole.expires_at > datetime.utcnow()
                    )
                )
            ).all()
            
            if not user_roles:
                return []
            
            # 收集所有角色ID（包括父角色）
            role_ids = set()
            for ur in user_roles:
                role_ids.add(ur.role_id)
                # 获取父角色
                role = db.query(Role).filter(Role.id == ur.role_id).first()
                while role and role.parent_role_id:
                    role_ids.add(role.parent_role_id)
                    role = db.query(Role).filter(Role.id == role.parent_role_id).first()
            
            # 获取所有权限
            return db.query(Permission).join(
                role_permissions,
                Permission.id == role_permissions.c.permission_id
            ).filter(
                and_(
                    role_permissions.c.role_id.in_(role_ids),
                    Permission.is_active == True
                )
            ).distinct().all()
    
    def check_user_permission(self, user_id: str, resource: str, action: str) -> bool:
        """检查用户是否具有特定权限
        
        Args:
            user_id: 用户ID
            resource: 资源类型
            action: 操作类型
            
        Returns:
            bool: 是否具有权限
        """
        permissions = self.get_user_permissions(user_id)
        for perm in permissions:
            if perm.resource == resource and perm.action == action:
                return True
        return False
    
    def get_user_permission_set(self, user_id: str) -> Set[str]:
        """获取用户权限集合（格式：resource:action）
        
        Args:
            user_id: 用户ID
            
        Returns:
            Set[str]: 权限集合
        """
        permissions = self.get_user_permissions(user_id)
        return {f"{p.resource}:{p.action}" for p in permissions}
    
    # ========================================================================
    # 审计日志
    # ========================================================================
    
    def create_audit_log(self,
                         user_id: str,
                         action: str,
                         target_user_id: Optional[str] = None,
                         resource_type: Optional[str] = None,
                         resource_id: Optional[str] = None,
                         permission_id: Optional[str] = None,
                         role_id: Optional[str] = None,
                         old_value: Optional[Dict] = None,
                         new_value: Optional[Dict] = None,
                         ip_address: Optional[str] = None,
                         user_agent: Optional[str] = None,
                         status: str = 'success',
                         error_message: Optional[str] = None,
                         agent_analysis: Optional[Dict] = None) -> PermissionAuditLog:
        """创建审计日志
        
        Args:
            user_id: 操作用户ID
            action: 操作类型
            target_user_id: 目标用户ID
            resource_type: 资源类型
            resource_id: 资源ID
            permission_id: 权限ID
            role_id: 角色ID
            old_value: 变更前的值
            new_value: 变更后的值
            ip_address: IP地址
            user_agent: 用户代理
            status: 操作状态
            error_message: 错误信息
            agent_analysis: Agent分析结果
            
        Returns:
            PermissionAuditLog: 创建的审计日志
        """
        with self.db_manager.get_session() as db:
            log = PermissionAuditLog(
                id=str(uuid.uuid4()),
                user_id=user_id,
                target_user_id=target_user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                permission_id=permission_id,
                role_id=role_id,
                old_value=old_value,
                new_value=new_value,
                ip_address=ip_address,
                user_agent=user_agent,
                status=status,
                error_message=error_message,
                agent_analysis=agent_analysis
            )
            db.add(log)
            db.commit()
            db.refresh(log)
            return log
    
    def list_audit_logs(self,
                        user_id: Optional[str] = None,
                        target_user_id: Optional[str] = None,
                        action: Optional[str] = None,
                        status: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        page: int = 1,
                        page_size: int = 50) -> Tuple[List[PermissionAuditLog], int]:
        """列出审计日志
        
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
        with self.db_manager.get_session() as db:
            query = db.query(PermissionAuditLog)
            
            if user_id:
                query = query.filter(PermissionAuditLog.user_id == user_id)
            if target_user_id:
                query = query.filter(PermissionAuditLog.target_user_id == target_user_id)
            if action:
                query = query.filter(PermissionAuditLog.action == action)
            if status:
                query = query.filter(PermissionAuditLog.status == status)
            if start_time:
                query = query.filter(PermissionAuditLog.created_at >= start_time)
            if end_time:
                query = query.filter(PermissionAuditLog.created_at <= end_time)
            
            total = query.count()
            logs = query.order_by(desc(PermissionAuditLog.created_at)).offset(
                (page - 1) * page_size
            ).limit(page_size).all()
            
            return logs, total
    
    def create_access_log(self,
                          user_id: str,
                          resource: str,
                          action: str,
                          result: bool,
                          reason: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          user_agent: Optional[str] = None,
                          latency_ms: Optional[float] = None,
                          cached: bool = False) -> PermissionAccessLog:
        """创建访问日志
        
        Args:
            user_id: 用户ID
            resource: 请求的资源
            action: 请求的操作
            result: 验证结果
            reason: 原因说明
            ip_address: IP地址
            user_agent: 用户代理
            latency_ms: 延迟毫秒
            cached: 是否使用缓存
            
        Returns:
            PermissionAccessLog: 创建的访问日志
        """
        with self.db_manager.get_session() as db:
            log = PermissionAccessLog(
                id=str(uuid.uuid4()),
                user_id=user_id,
                resource=resource,
                action=action,
                result=result,
                reason=reason,
                ip_address=ip_address,
                user_agent=user_agent,
                latency_ms=latency_ms,
                cached=cached
            )
            db.add(log)
            db.commit()
            db.refresh(log)
            return log
    
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
        with self.db_manager.get_session() as db:
            query = db.query(PermissionAccessLog)
            
            if user_id:
                query = query.filter(PermissionAccessLog.user_id == user_id)
            if start_time:
                query = query.filter(PermissionAccessLog.created_at >= start_time)
            if end_time:
                query = query.filter(PermissionAccessLog.created_at <= end_time)
            
            total = query.count()
            allowed = query.filter(PermissionAccessLog.result == True).count()
            denied = query.filter(PermissionAccessLog.result == False).count()
            
            # 资源统计
            resource_stats = db.query(
                PermissionAccessLog.resource,
                func.count(PermissionAccessLog.id).label('count')
            ).group_by(PermissionAccessLog.resource).all()
            
            return {
                "total_requests": total,
                "allowed": allowed,
                "denied": denied,
                "allow_rate": allowed / total if total > 0 else 0,
                "by_resource": {r[0]: r[1] for r in resource_stats}
            }
    
    # ========================================================================
    # Agent 记忆
    # ========================================================================
    
    def create_agent_memory(self,
                            memory_type: str,
                            content: str,
                            user_id: Optional[str] = None,
                            embedding: Optional[List[float]] = None,
                            importance: float = 0.5,
                            expires_at: Optional[datetime] = None,
                            metadata: Optional[Dict] = None,
                            source: Optional[str] = None) -> PermissionAgentMemory:
        """创建Agent记忆
        
        Args:
            memory_type: 记忆类型
            content: 记忆内容
            user_id: 关联用户ID
            embedding: 向量嵌入
            importance: 重要性分数
            expires_at: 过期时间
            metadata: 元数据
            source: 记忆来源
            
        Returns:
            PermissionAgentMemory: 创建的记忆
        """
        with self.db_manager.get_session() as db:
            memory = PermissionAgentMemory(
                id=str(uuid.uuid4()),
                user_id=user_id,
                memory_type=memory_type,
                content=content,
                embedding=embedding,
                importance=importance,
                access_count=0,
                expires_at=expires_at,
                extra_data=metadata,
                source=source,
                is_active=True
            )
            db.add(memory)
            db.commit()
            db.refresh(memory)
            logger.info(f"Created permission agent memory: {memory_type}")
            return memory
    
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
        with self.db_manager.get_session() as db:
            query = db.query(PermissionAgentMemory).filter(
                and_(
                    PermissionAgentMemory.is_active == True,
                    PermissionAgentMemory.importance >= min_importance,
                    or_(
                        PermissionAgentMemory.expires_at.is_(None),
                        PermissionAgentMemory.expires_at > datetime.utcnow()
                    )
                )
            )
            
            if user_id:
                query = query.filter(
                    or_(
                        PermissionAgentMemory.user_id == user_id,
                        PermissionAgentMemory.user_id.is_(None)  # 全局记忆
                    )
                )
            if memory_type:
                query = query.filter(PermissionAgentMemory.memory_type == memory_type)
            
            return query.order_by(desc(PermissionAgentMemory.importance)).limit(limit).all()
    
    def update_memory_access(self, memory_id: str) -> None:
        """更新记忆访问信息
        
        Args:
            memory_id: 记忆ID
        """
        with self.db_manager.get_session() as db:
            memory = db.query(PermissionAgentMemory).filter(
                PermissionAgentMemory.id == memory_id
            ).first()
            if memory:
                memory.access_count += 1
                memory.last_accessed = datetime.utcnow()
                db.commit()
    
    def create_agent_reasoning(self,
                               trigger: str,
                               user_id: Optional[str] = None,
                               context: Optional[Dict] = None,
                               reasoning_steps: Optional[List[Dict]] = None,
                               conclusion: Optional[str] = None,
                               confidence: float = 0.0,
                               action_suggested: Optional[str] = None,
                               action_taken: Optional[str] = None,
                               outcome: Optional[str] = None,
                               model_used: Optional[str] = None,
                               tokens_used: int = 0,
                               latency_ms: float = 0) -> PermissionAgentReasoning:
        """创建Agent推理记录
        
        Args:
            trigger: 触发事件
            user_id: 用户ID
            context: 推理上下文
            reasoning_steps: 推理步骤
            conclusion: 结论
            confidence: 置信度
            action_suggested: 建议的行动
            action_taken: 实际采取的行动
            outcome: 结果
            model_used: 使用的模型
            tokens_used: 使用的token数
            latency_ms: 延迟毫秒
            
        Returns:
            PermissionAgentReasoning: 创建的推理记录
        """
        with self.db_manager.get_session() as db:
            reasoning = PermissionAgentReasoning(
                id=str(uuid.uuid4()),
                user_id=user_id,
                trigger=trigger,
                context=context,
                reasoning_steps=reasoning_steps,
                conclusion=conclusion,
                confidence=confidence,
                action_suggested=action_suggested,
                action_taken=action_taken,
                outcome=outcome,
                model_used=model_used,
                tokens_used=tokens_used,
                latency_ms=latency_ms
            )
            db.add(reasoning)
            db.commit()
            db.refresh(reasoning)
            return reasoning
    
    # ========================================================================
    # 权限策略
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
                      scope: str = 'global',
                      applies_to: str = 'user',
                      target_ids: Optional[List[str]] = None) -> PermissionPolicy:
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
            scope: 作用范围
            applies_to: 适用对象类型
            target_ids: 目标ID列表
            
        Returns:
            PermissionPolicy: 创建的策略
        """
        with self.db_manager.get_session() as db:
            policy = PermissionPolicy(
                id=str(uuid.uuid4()),
                name=name,
                description=description,
                policy_type=policy_type,
                resource_pattern=resource_pattern,
                action_pattern=action_pattern,
                conditions=conditions,
                priority=priority,
                is_active=True,
                effect=effect,
                scope=scope,
                applies_to=applies_to,
                target_ids=target_ids
            )
            db.add(policy)
            db.commit()
            db.refresh(policy)
            logger.info(f"Created permission policy: {name}")
            return policy
    
    def get_applicable_policies(self,
                                 user_id: str,
                                 resource: str,
                                 action: str) -> List[PermissionPolicy]:
        """获取适用的策略
        
        Args:
            user_id: 用户ID
            resource: 资源
            action: 操作
            
        Returns:
            List[PermissionPolicy]: 适用的策略列表
        """
        with self.db_manager.get_session() as db:
            # 获取所有活动策略
            policies = db.query(PermissionPolicy).filter(
                PermissionPolicy.is_active == True
            ).order_by(desc(PermissionPolicy.priority)).all()
            
            applicable = []
            for policy in policies:
                # 检查资源模式匹配
                if self._match_pattern(resource, policy.resource_pattern):
                    # 检查操作模式匹配
                    if self._match_pattern(action, policy.action_pattern):
                        # 检查是否适用于该用户
                        if self._policy_applies_to_user(policy, user_id, db):
                            applicable.append(policy)
            
            return applicable
    
    def _match_pattern(self, value: str, pattern: str) -> bool:
        """匹配模式
        
        Args:
            value: 要匹配的值
            pattern: 模式（支持*通配符）
            
        Returns:
            bool: 是否匹配
        """
        if pattern == '*':
            return True
        if '*' not in pattern:
            return value == pattern
        
        # 简单的通配符匹配
        import fnmatch
        return fnmatch.fnmatch(value, pattern)
    
    def _policy_applies_to_user(self, policy: PermissionPolicy, user_id: str, db: Session) -> bool:
        """检查策略是否适用于用户
        
        Args:
            policy: 策略
            user_id: 用户ID
            db: 数据库会话
            
        Returns:
            bool: 是否适用
        """
        if not policy.target_ids:
            return True  # 没有指定目标，适用于所有人
        
        if policy.applies_to == 'user':
            return user_id in policy.target_ids
        elif policy.applies_to == 'role':
            # 检查用户是否有目标角色
            user_roles = db.query(UserRole).filter(
                and_(
                    UserRole.user_id == user_id,
                    UserRole.role_id.in_(policy.target_ids),
                    UserRole.is_active == True
                )
            ).first()
            return user_roles is not None
        
        return False
    
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
                                   expires_at: Optional[datetime] = None,
                                   conditions: Optional[Dict] = None) -> ResourcePermission:
        """授予资源权限
        
        Args:
            resource_type: 资源类型
            resource_id: 资源ID
            user_id: 用户ID
            permission_type: 权限类型
            permissions: 具体权限列表
            granted_by: 授权人ID
            expires_at: 过期时间
            conditions: 权限条件
            
        Returns:
            ResourcePermission: 创建的资源权限
        """
        with self.db_manager.get_session() as db:
            # 检查是否已存在
            existing = db.query(ResourcePermission).filter(
                and_(
                    ResourcePermission.resource_type == resource_type,
                    ResourcePermission.resource_id == resource_id,
                    ResourcePermission.user_id == user_id
                )
            ).first()
            
            if existing:
                # 更新现有权限
                existing.permission_type = permission_type
                existing.permissions = permissions
                existing.granted_by = granted_by
                existing.expires_at = expires_at
                existing.conditions = conditions
                existing.is_active = True
                existing.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(existing)
                return existing
            
            # 创建新权限
            rp = ResourcePermission(
                id=str(uuid.uuid4()),
                resource_type=resource_type,
                resource_id=resource_id,
                user_id=user_id,
                permission_type=permission_type,
                permissions=permissions,
                granted_by=granted_by,
                expires_at=expires_at,
                is_active=True,
                conditions=conditions
            )
            db.add(rp)
            db.commit()
            db.refresh(rp)
            logger.info(f"Granted {permission_type} permission on {resource_type}:{resource_id} to user {user_id}")
            return rp
    
    def revoke_resource_permission(self,
                                    resource_type: str,
                                    resource_id: str,
                                    user_id: str) -> bool:
        """撤销资源权限
        
        Args:
            resource_type: 资源类型
            resource_id: 资源ID
            user_id: 用户ID
            
        Returns:
            bool: 是否撤销成功
        """
        with self.db_manager.get_session() as db:
            rp = db.query(ResourcePermission).filter(
                and_(
                    ResourcePermission.resource_type == resource_type,
                    ResourcePermission.resource_id == resource_id,
                    ResourcePermission.user_id == user_id
                )
            ).first()
            
            if rp:
                db.delete(rp)
                db.commit()
                logger.info(f"Revoked permission on {resource_type}:{resource_id} from user {user_id}")
                return True
            return False
    
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
        with self.db_manager.get_session() as db:
            rp = db.query(ResourcePermission).filter(
                and_(
                    ResourcePermission.resource_type == resource_type,
                    ResourcePermission.resource_id == resource_id,
                    ResourcePermission.user_id == user_id,
                    ResourcePermission.is_active == True,
                    or_(
                        ResourcePermission.expires_at.is_(None),
                        ResourcePermission.expires_at > datetime.utcnow()
                    )
                )
            ).first()
            
            if not rp:
                return False
            
            # 如果不需要特定权限，只要有记录就返回True
            if not required_permission:
                return True
            
            # 检查权限类型
            type_permissions = {
                'owner': ['read', 'write', 'delete', 'share', 'admin'],
                'editor': ['read', 'write'],
                'viewer': ['read'],
            }
            
            allowed = type_permissions.get(rp.permission_type, [])
            if required_permission in allowed:
                return True
            
            # 检查自定义权限列表
            if rp.permissions and required_permission in rp.permissions:
                return True
            
            return False
    
    # ========================================================================
    # 权限建议
    # ========================================================================
    
    def create_recommendation(self,
                               user_id: str,
                               recommendation_type: str,
                               reason: str,
                               permission_id: Optional[str] = None,
                               role_id: Optional[str] = None,
                               confidence: float = 0.0,
                               risk_assessment: Optional[Dict] = None) -> PermissionRecommendation:
        """创建权限建议
        
        Args:
            user_id: 目标用户ID
            recommendation_type: 建议类型
            reason: 建议原因
            permission_id: 相关权限ID
            role_id: 相关角色ID
            confidence: 置信度
            risk_assessment: 风险评估
            
        Returns:
            PermissionRecommendation: 创建的建议
        """
        with self.db_manager.get_session() as db:
            rec = PermissionRecommendation(
                id=str(uuid.uuid4()),
                user_id=user_id,
                recommendation_type=recommendation_type,
                permission_id=permission_id,
                role_id=role_id,
                reason=reason,
                confidence=confidence,
                risk_assessment=risk_assessment,
                status='pending'
            )
            db.add(rec)
            db.commit()
            db.refresh(rec)
            logger.info(f"Created permission recommendation for user {user_id}")
            return rec
    
    def list_recommendations(self,
                              user_id: Optional[str] = None,
                              status: Optional[str] = None,
                              page: int = 1,
                              page_size: int = 20) -> Tuple[List[PermissionRecommendation], int]:
        """列出权限建议
        
        Args:
            user_id: 过滤目标用户
            status: 过滤状态
            page: 页码
            page_size: 每页大小
            
        Returns:
            Tuple[List[PermissionRecommendation], int]: 建议列表和总数
        """
        with self.db_manager.get_session() as db:
            query = db.query(PermissionRecommendation)
            
            if user_id:
                query = query.filter(PermissionRecommendation.user_id == user_id)
            if status:
                query = query.filter(PermissionRecommendation.status == status)
            
            total = query.count()
            recs = query.order_by(desc(PermissionRecommendation.created_at)).offset(
                (page - 1) * page_size
            ).limit(page_size).all()
            
            return recs, total
    
    def review_recommendation(self,
                               recommendation_id: str,
                               status: str,
                               reviewed_by: str,
                               review_notes: Optional[str] = None) -> Optional[PermissionRecommendation]:
        """审核权限建议
        
        Args:
            recommendation_id: 建议ID
            status: 新状态
            reviewed_by: 审核人ID
            review_notes: 审核备注
            
        Returns:
            Optional[PermissionRecommendation]: 更新后的建议
        """
        with self.db_manager.get_session() as db:
            rec = db.query(PermissionRecommendation).filter(
                PermissionRecommendation.id == recommendation_id
            ).first()
            
            if not rec:
                return None
            
            rec.status = status
            rec.reviewed_by = reviewed_by
            rec.reviewed_at = datetime.utcnow()
            rec.review_notes = review_notes
            db.commit()
            db.refresh(rec)
            logger.info(f"Reviewed recommendation {recommendation_id}: {status}")
            return rec

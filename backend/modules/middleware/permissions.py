"""权限中间件模块

实现基于角色的访问控制(RBAC)和基于属性的访问控制(ABAC)
"""

import logging
from typing import Optional, Dict, Any, List, Callable, Set
from functools import wraps
from flask import Flask, request, g, abort, jsonify
from enum import Enum

logger = logging.getLogger(__name__)


class Permission(Enum):
    """权限枚举"""
    # 训练相关权限
    TRAINING_CREATE = "training:create"
    TRAINING_READ = "training:read"
    TRAINING_UPDATE = "training:update"
    TRAINING_DELETE = "training:delete"
    TRAINING_EXECUTE = "training:execute"
    TRAINING_STOP = "training:stop"
    
    # 模型相关权限
    MODEL_CREATE = "model:create"
    MODEL_READ = "model:read"
    MODEL_UPDATE = "model:update"
    MODEL_DELETE = "model:delete"
    MODEL_DEPLOY = "model:deploy"
    MODEL_DOWNLOAD = "model:download"
    
    # 数据相关权限
    DATA_CREATE = "data:create"
    DATA_READ = "data:read"
    DATA_UPDATE = "data:update"
    DATA_DELETE = "data:delete"
    DATA_UPLOAD = "data:upload"
    
    # 系统管理权限
    SYSTEM_ADMIN = "system:admin"
    USER_MANAGE = "user:manage"
    RESOURCE_MANAGE = "resource:manage"
    AUDIT_READ = "audit:read"
    
    # 成本管理权限
    COST_READ = "cost:read"
    COST_MANAGE = "cost:manage"
    BUDGET_MANAGE = "budget:manage"


class Role(Enum):
    """角色枚举"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MANAGER = "manager"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"


class PermissionManager:
    """权限管理器"""
    
    def __init__(self):
        # 角色权限映射
        self.role_permissions = self._init_role_permissions()
        
        # 用户角色映射
        self.user_roles: Dict[str, Set[Role]] = {}
    
    def _init_role_permissions(self) -> Dict[Role, Set[Permission]]:
        """初始化角色权限映射"""
        role_permissions = {
            Role.SUPER_ADMIN: set(Permission),
            Role.ADMIN: {
                Permission.TRAINING_CREATE, Permission.TRAINING_READ, Permission.TRAINING_UPDATE, Permission.TRAINING_DELETE,
                Permission.MODEL_CREATE, Permission.MODEL_READ, Permission.MODEL_UPDATE, Permission.MODEL_DELETE,
                Permission.DATA_CREATE, Permission.DATA_READ, Permission.DATA_UPDATE, Permission.DATA_DELETE,
                Permission.USER_MANAGE, Permission.RESOURCE_MANAGE, Permission.AUDIT_READ,
                Permission.COST_READ, Permission.COST_MANAGE
            },
            Role.MANAGER: {
                Permission.TRAINING_CREATE, Permission.TRAINING_READ, Permission.TRAINING_UPDATE, Permission.TRAINING_EXECUTE,
                Permission.MODEL_CREATE, Permission.MODEL_READ, Permission.MODEL_UPDATE, Permission.MODEL_DEPLOY,
                Permission.DATA_CREATE, Permission.DATA_READ, Permission.DATA_UPDATE,
                Permission.COST_READ, Permission.BUDGET_MANAGE
            },
            Role.DEVELOPER: {
                Permission.TRAINING_CREATE, Permission.TRAINING_READ, Permission.TRAINING_EXECUTE, Permission.TRAINING_STOP,
                Permission.MODEL_CREATE, Permission.MODEL_READ, Permission.MODEL_UPDATE, Permission.MODEL_DEPLOY,
                Permission.DATA_CREATE, Permission.DATA_READ, Permission.DATA_UPDATE, Permission.DATA_UPLOAD
            },
            Role.ANALYST: {
                Permission.TRAINING_READ,
                Permission.MODEL_READ,
                Permission.DATA_READ
            },
            Role.VIEWER: {
                Permission.TRAINING_READ,
                Permission.MODEL_READ,
                Permission.DATA_READ
            },
            Role.GUEST: set()
        }
        return role_permissions
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """检查用户是否具有指定权限
        
        Args:
            user_id: 用户ID
            permission: 权限
            
        Returns:
            是否具有权限
        """
        try:
            # 获取用户角色
            user_roles = self.user_roles.get(user_id, set())
            
            # 检查每个角色是否具有该权限
            for role in user_roles:
                if role in self.role_permissions:
                    role_perms = self.role_permissions[role]
                    if permission in role_perms:
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"权限检查失败: {e}")
            return False
    
    def assign_role(self, user_id: str, role: Role) -> bool:
        """分配角色
        
        Args:
            user_id: 用户ID
            role: 角色
            
        Returns:
            是否成功
        """
        try:
            if user_id not in self.user_roles:
                self.user_roles[user_id] = set()
            
            self.user_roles[user_id].add(role)
            logger.info(f"为用户 {user_id} 分配角色 {role.value}")
            return True
            
        except Exception as e:
            logger.error(f"分配角色失败: {e}")
            return False
    
    def revoke_role(self, user_id: str, role: Role) -> bool:
        """撤销角色
        
        Args:
            user_id: 用户ID
            role: 角色
            
        Returns:
            是否成功
        """
        try:
            if user_id in self.user_roles and role in self.user_roles[user_id]:
                self.user_roles[user_id].remove(role)
                logger.info(f"为用户 {user_id} 撤销角色 {role.value}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"撤销角色失败: {e}")
            return False
    
    def get_user_roles(self, user_id: str) -> List[Role]:
        """获取用户角色
        
        Args:
            user_id: 用户ID
            
        Returns:
            角色列表
        """
        return list(self.user_roles.get(user_id, set()))
    
    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """获取用户权限
        
        Args:
            user_id: 用户ID
            
        Returns:
            权限列表
        """
        permissions = set()
        
        # 获取用户角色
        user_roles = self.user_roles.get(user_id, set())
        
        # 收集所有角色的权限
        for role in user_roles:
            if role in self.role_permissions:
                permissions.update(self.role_permissions[role])
        
        return list(permissions)


class PermissionMiddleware:
    """权限中间件"""
    
    def __init__(self, app: Optional[Flask] = None):
        """初始化权限中间件
        
        Args:
            app: Flask应用实例
        """
        self.app = app
        self.permission_manager = PermissionManager()
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """初始化Flask应用
        
        Args:
            app: Flask应用实例
        """
        self.app = app
        # 这里可以注册全局权限检查钩子
        pass
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """检查权限
        
        Args:
            user_id: 用户ID
            permission: 权限
            
        Returns:
            是否具有权限
        """
        return self.permission_manager.check_permission(user_id, permission)
    
    def assign_role(self, user_id: str, role: Role) -> bool:
        """分配角色
        
        Args:
            user_id: 用户ID
            role: 角色
            
        Returns:
            是否成功
        """
        return self.permission_manager.assign_role(user_id, role)
    
    def revoke_role(self, user_id: str, role: Role) -> bool:
        """撤销角色
        
        Args:
            user_id: 用户ID
            role: 角色
            
        Returns:
            是否成功
        """
        return self.permission_manager.revoke_role(user_id, role)


def require_permission(permission: Permission):
    """权限检查装饰器
    
    Args:
        permission: 需要的权限
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # 获取当前用户
            current_user = getattr(g, 'current_user', None)
            if not current_user:
                abort(401)
            
            # 获取权限中间件
            permission_middleware = get_permission_middleware()
            
            # 检查权限
            if not permission_middleware.check_permission(current_user.id, permission):
                logger.warning(f"用户 {current_user.id} 缺少权限: {permission.value}")
                abort(403)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# 全局权限中间件实例
_permission_middleware: Optional[PermissionMiddleware] = None


def get_permission_middleware() -> PermissionMiddleware:
    """获取全局权限中间件实例"""
    global _permission_middleware
    if _permission_middleware is None:
        _permission_middleware = PermissionMiddleware()
    return _permission_middleware


def set_permission_middleware(permission_middleware: PermissionMiddleware):
    """设置全局权限中间件实例"""
    global _permission_middleware
    _permission_middleware = permission_middleware
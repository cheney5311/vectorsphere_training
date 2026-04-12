"""权限管理API接口

提供权限管理的RESTful API，包括：
- 权限CRUD: 创建、读取、更新、删除权限
- 角色管理: 角色CRUD、角色-权限关联
- 用户角色: 用户角色分配和管理
- 资源权限: 细粒度资源级别权限控制
- 权限验证: 权限检查接口
- Agent分析: 智能权限分析和建议
- 审计日志: 权限操作审计

所有API端点都需要认证，部分端点需要管理员权限。
"""

from flask import Blueprint, request
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from functools import wraps

from backend.utils.response import success_response, error_response
from backend.utils.validation import validate_json, validate_required_fields
from backend.services.permission_service import PermissionService, get_permission_service
from backend.services.auth_service import AuthService, get_auth_service
from backend.modules.auth.auth_exceptions import (
    AuthException, AuthenticationError, AuthorizationError, 
    PermissionDeniedError, UserNotFoundError
)

logger = logging.getLogger(__name__)

# 创建蓝图
permission_api_bp = Blueprint('permission_api', __name__, url_prefix='/api/v2/permissions')

# 初始化服务
permission_service = PermissionService()
auth_service = get_auth_service()


# ============================================================================
# 辅助函数和装饰器
# ============================================================================

def verify_token_and_get_user_id() -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """验证令牌并获取用户ID
    
    从请求头中提取Authorization令牌，验证并返回用户ID。
    
    Returns:
        Tuple[Optional[str], Optional[Dict]]: (用户ID, 错误信息)
        - 成功: (user_id, None)
        - 失败: (None, {"error": "错误信息"})
    """
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return None, {"error": "缺少访问令牌"}
    
    access_token = auth_header.split(' ', 1)[1]
    try:
        payload = auth_service.verify_token(access_token)
        user_id = payload.get('user_id')
        if not user_id:
            return None, {"error": "无效的访问令牌"}
        return user_id, None
    except Exception as e:
        return None, {"error": str(e)}


def get_request_context() -> Dict[str, Any]:
    """获取请求上下文
    
    返回请求的IP地址和用户代理信息。
    
    Returns:
        Dict: 包含 ip_address 和 user_agent 的字典
    """
    return {
        'ip_address': request.remote_addr,
        'user_agent': request.headers.get('User-Agent', '')
    }


def check_permission(user_id: str, resource: str, action: str) -> bool:
    """检查用户权限
    
    Args:
        user_id: 用户ID
        resource: 资源类型
        action: 操作类型
        
    Returns:
        bool: 是否具有权限
    """
    result = permission_service.check_user_permission(user_id, resource, action)
    return result.allowed


def require_permission(resource: str, action: str):
    """权限检查装饰器
    
    装饰API端点，自动检查权限。
    
    Args:
        resource: 资源类型
        action: 操作类型
        
    Returns:
        装饰器函数
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            user_id, error = verify_token_and_get_user_id()
            if error:
                return error_response("无效的访问令牌", 401, "InvalidToken", error)
            
            if not check_permission(user_id, resource, action):
                return error_response("权限不足", 403, "PermissionDenied", {
                    "required": f"{resource}:{action}"
                })
            
            # 将用户ID注入kwargs
            kwargs['current_user_id'] = user_id
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ============================================================================
# 权限管理端点
# ============================================================================

@permission_api_bp.route('/permissions', methods=['POST'])
@require_permission('permissions', 'create')
def create_permission(current_user_id: str):
    """创建权限
    
    创建新的系统权限。
    
    ---
    请求体:
        {
            "name": "string",           # 必填，权限名称，唯一
            "resource": "string",       # 必填，资源类型
            "action": "string",         # 必填，操作类型
            "description": "string",    # 可选，权限描述
            "scope": "string",          # 可选，权限范围 (global/tenant/personal)
            "conditions": {},           # 可选，权限条件
            "priority": 0,              # 可选，优先级
            "risk_level": "string",     # 可选，风险等级 (low/medium/high/critical)
            "requires_mfa": false,      # 可选，是否需要MFA
            "audit_level": "string"     # 可选，审计级别 (none/basic/detailed)
        }
    
    响应:
        200: 权限创建成功
        400: 参数验证失败
        403: 权限不足
        409: 权限名称已存在
    """
    try:
        data = validate_json(request)
        validate_required_fields(data, ['name', 'resource', 'action'])
        
        permission = permission_service.create_permission(
            name=data['name'],
            resource=data['resource'],
            action=data['action'],
            description=data.get('description'),
            scope=data.get('scope', 'global'),
            conditions=data.get('conditions'),
            priority=data.get('priority', 0),
            risk_level=data.get('risk_level', 'low'),
            requires_mfa=data.get('requires_mfa', False),
            audit_level=data.get('audit_level', 'basic'),
            created_by=current_user_id
        )
        
        return success_response({
            "permission": permission.to_dict()
        }, "权限创建成功")
        
    except ValueError as e:
        if 'already exists' in str(e):
            return error_response(str(e), 409, "PermissionExists")
        return error_response(str(e), 400, "ValidationError")
    except Exception as e:
        logger.error(f"Create permission failed: {str(e)}")
        return error_response("权限创建失败", 500, "CreatePermissionError", {"error": str(e)})


@permission_api_bp.route('/permissions', methods=['GET'])
@require_permission('permissions', 'read')
def list_permissions(current_user_id: str):
    """获取权限列表
    
    获取系统权限列表，支持分页和过滤。
    
    ---
    查询参数:
        resource: 过滤资源类型
        action: 过滤操作类型
        risk_level: 过滤风险等级
        is_active: 过滤是否启用 (true/false)
        page: 页码，默认1
        page_size: 每页大小，默认50
    
    响应:
        200: 返回权限列表
    """
    try:
        resource = request.args.get('resource')
        action = request.args.get('action')
        risk_level = request.args.get('risk_level')
        is_active = request.args.get('is_active')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        # 转换布尔值
        if is_active is not None:
            is_active = is_active.lower() == 'true'
        
        permissions, total = permission_service.list_permissions(
            resource=resource,
            action=action,
            risk_level=risk_level,
            is_active=is_active,
            page=page,
            page_size=page_size
        )
        
        return success_response({
            "permissions": [p.to_dict() for p in permissions],
            "total": total,
            "page": page,
            "page_size": page_size
        }, "获取权限列表成功")
        
    except Exception as e:
        logger.error(f"List permissions failed: {str(e)}")
        return error_response("获取权限列表失败", 500, "ListPermissionsError", {"error": str(e)})


@permission_api_bp.route('/permissions/<permission_id>', methods=['GET'])
@require_permission('permissions', 'read')
def get_permission(permission_id: str, current_user_id: str):
    """获取权限详情
    
    根据ID获取权限详细信息。
    
    ---
    路径参数:
        permission_id: 权限ID
    
    响应:
        200: 返回权限详情
        404: 权限不存在
    """
    try:
        permission = permission_service.get_permission(permission_id)
        if not permission:
            return error_response("权限不存在", 404, "PermissionNotFound")
        
        return success_response({
            "permission": permission.to_dict()
        }, "获取权限详情成功")
        
    except Exception as e:
        logger.error(f"Get permission failed: {str(e)}")
        return error_response("获取权限详情失败", 500, "GetPermissionError", {"error": str(e)})


@permission_api_bp.route('/permissions/<permission_id>', methods=['PUT'])
@require_permission('permissions', 'update')
def update_permission(permission_id: str, current_user_id: str):
    """更新权限
    
    更新权限信息。
    
    ---
    路径参数:
        permission_id: 权限ID
    
    请求体:
        {
            "description": "string",
            "scope": "string",
            "conditions": {},
            "priority": 0,
            "risk_level": "string",
            "requires_mfa": false,
            "audit_level": "string",
            "is_active": true
        }
    
    响应:
        200: 权限更新成功
        404: 权限不存在
    """
    try:
        data = validate_json(request)
        
        # 允许更新的字段
        allowed_fields = [
            'description', 'scope', 'conditions', 'priority',
            'risk_level', 'requires_mfa', 'audit_level', 'is_active'
        ]
        update_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        permission = permission_service.update_permission(
            permission_id=permission_id,
            updated_by=current_user_id,
            **update_data
        )
        
        if not permission:
            return error_response("权限不存在", 404, "PermissionNotFound")
        
        return success_response({
            "permission": permission.to_dict()
        }, "权限更新成功")
        
    except ValueError as e:
        return error_response(str(e), 400, "ValidationError")
    except Exception as e:
        logger.error(f"Update permission failed: {str(e)}")
        return error_response("权限更新失败", 500, "UpdatePermissionError", {"error": str(e)})


@permission_api_bp.route('/permissions/<permission_id>', methods=['DELETE'])
@require_permission('permissions', 'delete')
def delete_permission(permission_id: str, current_user_id: str):
    """删除权限
    
    删除指定权限（系统内置权限不可删除）。
    
    ---
    路径参数:
        permission_id: 权限ID
    
    响应:
        200: 权限删除成功
        400: 无法删除系统权限
        404: 权限不存在
    """
    try:
        result = permission_service.delete_permission(permission_id, deleted_by=current_user_id)
        
        if not result:
            return error_response("权限不存在", 404, "PermissionNotFound")
        
        return success_response({"deleted": True}, "权限删除成功")
        
    except ValueError as e:
        return error_response(str(e), 400, "CannotDeleteSystemPermission")
    except Exception as e:
        logger.error(f"Delete permission failed: {str(e)}")
        return error_response("权限删除失败", 500, "DeletePermissionError", {"error": str(e)})


# ============================================================================
# 角色管理端点
# ============================================================================

@permission_api_bp.route('/roles', methods=['POST'])
@require_permission('roles', 'create')
def create_role(current_user_id: str):
    """创建角色
    
    创建新的系统角色。
    
    ---
    请求体:
        {
            "name": "string",           # 必填，角色名称，唯一
            "display_name": "string",   # 可选，显示名称
            "description": "string",    # 可选，角色描述
            "parent_role_id": "string", # 可选，父角色ID
            "level": 0,                 # 可选，角色级别
            "max_users": null,          # 可选，最大用户数
            "metadata": {}              # 可选，元数据
        }
    
    响应:
        200: 角色创建成功
        400: 参数验证失败
        409: 角色名称已存在
    """
    try:
        data = validate_json(request)
        validate_required_fields(data, ['name'])
        
        role = permission_service.create_role(
            name=data['name'],
            display_name=data.get('display_name'),
            description=data.get('description'),
            parent_role_id=data.get('parent_role_id'),
            level=data.get('level', 0),
            max_users=data.get('max_users'),
            metadata=data.get('metadata'),
            created_by=current_user_id
        )
        
        return success_response({
            "role": role.to_dict()
        }, "角色创建成功")
        
    except ValueError as e:
        if 'already exists' in str(e):
            return error_response(str(e), 409, "RoleExists")
        return error_response(str(e), 400, "ValidationError")
    except Exception as e:
        logger.error(f"Create role failed: {str(e)}")
        return error_response("角色创建失败", 500, "CreateRoleError", {"error": str(e)})


@permission_api_bp.route('/roles', methods=['GET'])
@require_permission('roles', 'read')
def list_roles(current_user_id: str):
    """获取角色列表
    
    获取系统角色列表，支持分页和过滤。
    
    ---
    查询参数:
        is_active: 过滤是否启用
        is_system: 过滤是否系统内置
        page: 页码，默认1
        page_size: 每页大小，默认50
    
    响应:
        200: 返回角色列表
    """
    try:
        is_active = request.args.get('is_active')
        is_system = request.args.get('is_system')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        if is_active is not None:
            is_active = is_active.lower() == 'true'
        if is_system is not None:
            is_system = is_system.lower() == 'true'
        
        roles, total = permission_service.list_roles(
            is_active=is_active,
            is_system=is_system,
            page=page,
            page_size=page_size
        )
        
        return success_response({
            "roles": [r.to_dict() for r in roles],
            "total": total,
            "page": page,
            "page_size": page_size
        }, "获取角色列表成功")
        
    except Exception as e:
        logger.error(f"List roles failed: {str(e)}")
        return error_response("获取角色列表失败", 500, "ListRolesError", {"error": str(e)})


@permission_api_bp.route('/roles/<role_id>', methods=['GET'])
@require_permission('roles', 'read')
def get_role(role_id: str, current_user_id: str):
    """获取角色详情
    
    获取角色详情，包括关联的权限列表。
    
    ---
    路径参数:
        role_id: 角色ID
    
    查询参数:
        include_permissions: 是否包含权限列表 (true/false)
        include_inherited: 是否包含继承的权限 (true/false)
    
    响应:
        200: 返回角色详情
        404: 角色不存在
    """
    try:
        role = permission_service.get_role(role_id)
        if not role:
            return error_response("角色不存在", 404, "RoleNotFound")
        
        result = {"role": role.to_dict()}
        
        # 可选包含权限
        include_permissions = request.args.get('include_permissions', 'false').lower() == 'true'
        if include_permissions:
            include_inherited = request.args.get('include_inherited', 'true').lower() == 'true'
            permissions = permission_service.get_role_permissions(role_id, include_inherited)
            result['permissions'] = [p.to_dict() for p in permissions]
        
        return success_response(result, "获取角色详情成功")
        
    except Exception as e:
        logger.error(f"Get role failed: {str(e)}")
        return error_response("获取角色详情失败", 500, "GetRoleError", {"error": str(e)})


@permission_api_bp.route('/roles/<role_id>', methods=['PUT'])
@require_permission('roles', 'update')
def update_role(role_id: str, current_user_id: str):
    """更新角色
    
    更新角色信息。
    
    ---
    路径参数:
        role_id: 角色ID
    
    请求体:
        {
            "display_name": "string",
            "description": "string",
            "parent_role_id": "string",
            "level": 0,
            "max_users": null,
            "metadata": {},
            "is_active": true
        }
    
    响应:
        200: 角色更新成功
        404: 角色不存在
    """
    try:
        data = validate_json(request)
        
        allowed_fields = [
            'display_name', 'description', 'parent_role_id',
            'level', 'max_users', 'metadata', 'is_active'
        ]
        update_data = {k: v for k, v in data.items() if k in allowed_fields}
        
        role = permission_service.update_role(
            role_id=role_id,
            updated_by=current_user_id,
            **update_data
        )
        
        if not role:
            return error_response("角色不存在", 404, "RoleNotFound")
        
        return success_response({
            "role": role.to_dict()
        }, "角色更新成功")
        
    except ValueError as e:
        return error_response(str(e), 400, "ValidationError")
    except Exception as e:
        logger.error(f"Update role failed: {str(e)}")
        return error_response("角色更新失败", 500, "UpdateRoleError", {"error": str(e)})


@permission_api_bp.route('/roles/<role_id>', methods=['DELETE'])
@require_permission('roles', 'delete')
def delete_role(role_id: str, current_user_id: str):
    """删除角色
    
    删除指定角色（系统内置角色不可删除）。
    
    ---
    路径参数:
        role_id: 角色ID
    
    响应:
        200: 角色删除成功
        400: 无法删除系统角色
        404: 角色不存在
    """
    try:
        result = permission_service.delete_role(role_id, deleted_by=current_user_id)
        
        if not result:
            return error_response("角色不存在", 404, "RoleNotFound")
        
        return success_response({"deleted": True}, "角色删除成功")
        
    except ValueError as e:
        return error_response(str(e), 400, "CannotDeleteSystemRole")
    except Exception as e:
        logger.error(f"Delete role failed: {str(e)}")
        return error_response("角色删除失败", 500, "DeleteRoleError", {"error": str(e)})


# ============================================================================
# 角色-权限关联端点
# ============================================================================

@permission_api_bp.route('/roles/<role_id>/permissions', methods=['GET'])
@require_permission('roles', 'read')
def get_role_permissions(role_id: str, current_user_id: str):
    """获取角色权限
    
    获取角色的所有权限。
    
    ---
    路径参数:
        role_id: 角色ID
    
    查询参数:
        include_inherited: 是否包含继承的权限 (true/false)
    
    响应:
        200: 返回权限列表
    """
    try:
        include_inherited = request.args.get('include_inherited', 'true').lower() == 'true'
        permissions = permission_service.get_role_permissions(role_id, include_inherited)
        
        return success_response({
            "permissions": [p.to_dict() for p in permissions]
        }, "获取角色权限成功")
        
    except Exception as e:
        logger.error(f"Get role permissions failed: {str(e)}")
        return error_response("获取角色权限失败", 500, "GetRolePermissionsError", {"error": str(e)})


@permission_api_bp.route('/roles/<role_id>/permissions', methods=['POST'])
@require_permission('roles', 'manage')
def assign_permission_to_role(role_id: str, current_user_id: str):
    """为角色分配权限
    
    为角色添加新的权限。
    
    ---
    路径参数:
        role_id: 角色ID
    
    请求体:
        {
            "permission_id": "string"   # 必填，权限ID
        }
    
    响应:
        200: 权限分配成功
    """
    try:
        data = validate_json(request)
        validate_required_fields(data, ['permission_id'])
        
        result = permission_service.assign_permission_to_role(
            role_id=role_id,
            permission_id=data['permission_id'],
            granted_by=current_user_id
        )
        
        return success_response({
            "assigned": result
        }, "权限分配成功")
        
    except Exception as e:
        logger.error(f"Assign permission to role failed: {str(e)}")
        return error_response("权限分配失败", 500, "AssignPermissionError", {"error": str(e)})


@permission_api_bp.route('/roles/<role_id>/permissions/<permission_id>', methods=['DELETE'])
@require_permission('roles', 'manage')
def remove_permission_from_role(role_id: str, permission_id: str, current_user_id: str):
    """移除角色权限
    
    从角色中移除指定权限。
    
    ---
    路径参数:
        role_id: 角色ID
        permission_id: 权限ID
    
    响应:
        200: 权限移除成功
        404: 关联不存在
    """
    try:
        result = permission_service.remove_permission_from_role(
            role_id=role_id,
            permission_id=permission_id,
            removed_by=current_user_id
        )
        
        if not result:
            return error_response("角色-权限关联不存在", 404, "RolePermissionNotFound")
        
        return success_response({"removed": True}, "权限移除成功")
        
    except Exception as e:
        logger.error(f"Remove permission from role failed: {str(e)}")
        return error_response("权限移除失败", 500, "RemovePermissionError", {"error": str(e)})


# ============================================================================
# 用户角色管理端点
# ============================================================================

@permission_api_bp.route('/users/<user_id>/roles', methods=['GET'])
def get_user_roles(user_id: str):
    """获取用户角色
    
    获取指定用户的所有角色。
    
    ---
    路径参数:
        user_id: 用户ID
    
    查询参数:
        active_only: 是否只返回有效角色 (true/false)
    
    响应:
        200: 返回角色列表
    """
    current_user_id, error = verify_token_and_get_user_id()
    if error:
        return error_response("无效的访问令牌", 401, "InvalidToken", error)
    
    # 用户可以查看自己的角色，或具有查看权限的管理员
    if current_user_id != user_id and not check_permission(current_user_id, "users", "read"):
        return error_response("权限不足", 403, "PermissionDenied")
    
    try:
        active_only = request.args.get('active_only', 'true').lower() == 'true'
        roles = permission_service.get_user_roles(user_id, active_only)
        
        return success_response({
            "roles": [r.to_dict() for r in roles]
        }, "获取用户角色成功")
        
    except Exception as e:
        logger.error(f"Get user roles failed: {str(e)}")
        return error_response("获取用户角色失败", 500, "GetUserRolesError", {"error": str(e)})


@permission_api_bp.route('/users/<user_id>/roles', methods=['POST'])
@require_permission('users', 'manage')
def assign_role_to_user(user_id: str, current_user_id: str):
    """为用户分配角色
    
    为用户添加新的角色。
    
    ---
    路径参数:
        user_id: 用户ID
    
    请求体:
        {
            "role_id": "string",        # 必填，角色ID
            "expires_at": "datetime",   # 可选，过期时间 (ISO8601格式)
            "conditions": {},           # 可选，关联条件
            "scope": "string"           # 可选，权限范围
        }
    
    响应:
        200: 角色分配成功
    """
    try:
        data = validate_json(request)
        validate_required_fields(data, ['role_id'])
        
        expires_at = None
        if data.get('expires_at'):
            expires_at = datetime.fromisoformat(data['expires_at'].replace('Z', '+00:00'))
        
        user_role = permission_service.assign_role_to_user(
            user_id=user_id,
            role_id=data['role_id'],
            assigned_by=current_user_id,
            expires_at=expires_at,
            conditions=data.get('conditions'),
            scope=data.get('scope', 'global')
        )
        
        return success_response({
            "user_role": user_role.to_dict()
        }, "角色分配成功")
        
    except Exception as e:
        logger.error(f"Assign role to user failed: {str(e)}")
        return error_response("角色分配失败", 500, "AssignRoleError", {"error": str(e)})


@permission_api_bp.route('/users/<user_id>/roles/<role_id>', methods=['DELETE'])
@require_permission('users', 'manage')
def remove_role_from_user(user_id: str, role_id: str, current_user_id: str):
    """移除用户角色
    
    从用户中移除指定角色。
    
    ---
    路径参数:
        user_id: 用户ID
        role_id: 角色ID
    
    响应:
        200: 角色移除成功
        404: 关联不存在
    """
    try:
        result = permission_service.remove_role_from_user(
            user_id=user_id,
            role_id=role_id,
            removed_by=current_user_id
        )
        
        if not result:
            return error_response("用户-角色关联不存在", 404, "UserRoleNotFound")
        
        return success_response({"removed": True}, "角色移除成功")
        
    except Exception as e:
        logger.error(f"Remove role from user failed: {str(e)}")
        return error_response("角色移除失败", 500, "RemoveRoleError", {"error": str(e)})


@permission_api_bp.route('/users/<user_id>/permissions', methods=['GET'])
def get_user_permissions(user_id: str):
    """获取用户权限
    
    获取用户的所有权限（通过角色）。
    
    ---
    路径参数:
        user_id: 用户ID
    
    响应:
        200: 返回权限列表
    """
    current_user_id, error = verify_token_and_get_user_id()
    if error:
        return error_response("无效的访问令牌", 401, "InvalidToken", error)
    
    if current_user_id != user_id and not check_permission(current_user_id, "users", "read"):
        return error_response("权限不足", 403, "PermissionDenied")
    
    try:
        permissions = permission_service.get_user_permissions(user_id)
        
        return success_response({
            "permissions": [p.to_dict() for p in permissions]
        }, "获取用户权限成功")
        
    except Exception as e:
        logger.error(f"Get user permissions failed: {str(e)}")
        return error_response("获取用户权限失败", 500, "GetUserPermissionsError", {"error": str(e)})


# ============================================================================
# 权限验证端点
# ============================================================================

@permission_api_bp.route('/check', methods=['POST'])
def check_permission_api():
    """检查权限
    
    验证当前用户是否具有特定权限。
    
    ---
    请求体:
        {
            "resource": "string",   # 必填，资源类型
            "action": "string"      # 必填，操作类型
        }
    
    响应:
        200: 返回权限检查结果
            {
                "allowed": true/false,
                "reason": "string",
                "risk_level": "string",
                "requires_mfa": true/false
            }
    """
    user_id, error = verify_token_and_get_user_id()
    if error:
        return error_response("无效的访问令牌", 401, "InvalidToken", error)
    
    try:
        data = validate_json(request)
        validate_required_fields(data, ['resource', 'action'])
        
        ctx = get_request_context()
        result = permission_service.check_user_permission(
            user_id=user_id,
            resource=data['resource'],
            action=data['action'],
            ip_address=ctx['ip_address'],
            user_agent=ctx['user_agent']
        )
        
        return success_response({
            "allowed": result.allowed,
            "reason": result.reason,
            "risk_level": result.risk_level,
            "requires_mfa": result.requires_mfa,
            "matched_policies": result.matched_policies
        }, "权限检查完成")
        
    except Exception as e:
        logger.error(f"Check permission failed: {str(e)}")
        return error_response("权限检查失败", 500, "CheckPermissionError", {"error": str(e)})


@permission_api_bp.route('/users/<user_id>/check', methods=['POST'])
@require_permission('users', 'read')
def check_user_permission_api(user_id: str, current_user_id: str):
    """检查指定用户权限
    
    验证指定用户是否具有特定权限（需要管理员权限）。
    
    ---
    路径参数:
        user_id: 要检查的用户ID
    
    请求体:
        {
            "resource": "string",   # 必填，资源类型
            "action": "string"      # 必填，操作类型
        }
    
    响应:
        200: 返回权限检查结果
    """
    try:
        data = validate_json(request)
        validate_required_fields(data, ['resource', 'action'])
        
        result = permission_service.check_user_permission(
            user_id=user_id,
            resource=data['resource'],
            action=data['action'],
            record_access=False  # 检查其他用户时不记录访问日志
        )
        
        return success_response({
            "user_id": user_id,
            "allowed": result.allowed,
            "reason": result.reason,
            "risk_level": result.risk_level,
            "requires_mfa": result.requires_mfa
        }, "权限检查完成")
        
    except Exception as e:
        logger.error(f"Check user permission failed: {str(e)}")
        return error_response("权限检查失败", 500, "CheckPermissionError", {"error": str(e)})


# ============================================================================
# 资源权限端点
# ============================================================================

@permission_api_bp.route('/resources/<resource_type>/<resource_id>/permissions', methods=['GET'])
def get_resource_permissions(resource_type: str, resource_id: str):
    """获取资源权限
    
    获取特定资源的权限配置。
    
    ---
    路径参数:
        resource_type: 资源类型
        resource_id: 资源ID
    
    响应:
        200: 返回资源权限列表
    """
    user_id, error = verify_token_and_get_user_id()
    if error:
        return error_response("无效的访问令牌", 401, "InvalidToken", error)
    
    # 检查是否有权限查看资源权限
    if not permission_service.check_resource_permission(resource_type, resource_id, user_id, 'admin'):
        if not check_permission(user_id, 'resources', 'read'):
            return error_response("权限不足", 403, "PermissionDenied")
    
    try:
        # 注意：这里需要在repository中实现获取资源所有权限的方法
        # 暂时返回空列表
        return success_response({
            "resource_type": resource_type,
            "resource_id": resource_id,
            "permissions": []
        }, "获取资源权限成功")
        
    except Exception as e:
        logger.error(f"Get resource permissions failed: {str(e)}")
        return error_response("获取资源权限失败", 500, "GetResourcePermissionsError", {"error": str(e)})


@permission_api_bp.route('/resources/<resource_type>/<resource_id>/permissions', methods=['POST'])
def grant_resource_permission(resource_type: str, resource_id: str):
    """授予资源权限
    
    为用户授予特定资源的权限。
    
    ---
    路径参数:
        resource_type: 资源类型
        resource_id: 资源ID
    
    请求体:
        {
            "user_id": "string",            # 必填，目标用户ID
            "permission_type": "string",    # 可选，权限类型 (owner/editor/viewer/custom)
            "permissions": ["string"],      # 可选，具体权限列表
            "expires_at": "datetime"        # 可选，过期时间
        }
    
    响应:
        200: 权限授予成功
    """
    current_user_id, error = verify_token_and_get_user_id()
    if error:
        return error_response("无效的访问令牌", 401, "InvalidToken", error)
    
    # 检查是否有权限管理资源权限
    if not permission_service.check_resource_permission(resource_type, resource_id, current_user_id, 'admin'):
        if not check_permission(current_user_id, 'resources', 'manage'):
            return error_response("权限不足", 403, "PermissionDenied")
    
    try:
        data = validate_json(request)
        validate_required_fields(data, ['user_id'])
        
        expires_at = None
        if data.get('expires_at'):
            expires_at = datetime.fromisoformat(data['expires_at'].replace('Z', '+00:00'))
        
        rp = permission_service.grant_resource_permission(
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=data['user_id'],
            permission_type=data.get('permission_type', 'viewer'),
            permissions=data.get('permissions'),
            granted_by=current_user_id,
            expires_at=expires_at
        )
        
        return success_response({
            "resource_permission": rp.to_dict()
        }, "资源权限授予成功")
        
    except Exception as e:
        logger.error(f"Grant resource permission failed: {str(e)}")
        return error_response("资源权限授予失败", 500, "GrantResourcePermissionError", {"error": str(e)})


@permission_api_bp.route('/resources/<resource_type>/<resource_id>/permissions/<user_id>', methods=['DELETE'])
def revoke_resource_permission(resource_type: str, resource_id: str, user_id: str):
    """撤销资源权限
    
    撤销用户对特定资源的权限。
    
    ---
    路径参数:
        resource_type: 资源类型
        resource_id: 资源ID
        user_id: 目标用户ID
    
    响应:
        200: 权限撤销成功
        404: 权限不存在
    """
    current_user_id, error = verify_token_and_get_user_id()
    if error:
        return error_response("无效的访问令牌", 401, "InvalidToken", error)
    
    if not permission_service.check_resource_permission(resource_type, resource_id, current_user_id, 'admin'):
        if not check_permission(current_user_id, 'resources', 'manage'):
            return error_response("权限不足", 403, "PermissionDenied")
    
    try:
        result = permission_service.revoke_resource_permission(
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            revoked_by=current_user_id
        )
        
        if not result:
            return error_response("资源权限不存在", 404, "ResourcePermissionNotFound")
        
        return success_response({"revoked": True}, "资源权限撤销成功")
        
    except Exception as e:
        logger.error(f"Revoke resource permission failed: {str(e)}")
        return error_response("资源权限撤销失败", 500, "RevokeResourcePermissionError", {"error": str(e)})


@permission_api_bp.route('/resources/<resource_type>/<resource_id>/check', methods=['POST'])
def check_resource_permission_api(resource_type: str, resource_id: str):
    """检查资源权限
    
    检查当前用户对特定资源的权限。
    
    ---
    路径参数:
        resource_type: 资源类型
        resource_id: 资源ID
    
    请求体:
        {
            "permission": "string"  # 可选，所需权限 (read/write/delete/share/admin)
        }
    
    响应:
        200: 返回权限检查结果
    """
    user_id, error = verify_token_and_get_user_id()
    if error:
        return error_response("无效的访问令牌", 401, "InvalidToken", error)
    
    try:
        data = request.get_json() or {}
        required_permission = data.get('permission')
        
        result = permission_service.check_resource_permission(
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            required_permission=required_permission
        )
        
        return success_response({
            "resource_type": resource_type,
            "resource_id": resource_id,
            "allowed": result,
            "permission_checked": required_permission
        }, "资源权限检查完成")
        
    except Exception as e:
        logger.error(f"Check resource permission failed: {str(e)}")
        return error_response("资源权限检查失败", 500, "CheckResourcePermissionError", {"error": str(e)})


# ============================================================================
# Agent分析端点
# ============================================================================

@permission_api_bp.route('/users/<user_id>/analysis', methods=['GET'])
def analyze_user_permissions_api(user_id: str):
    """分析用户权限
    
    使用AI Agent分析用户的权限配置，提供优化建议。
    
    ---
    路径参数:
        user_id: 用户ID
    
    响应:
        200: 返回权限分析结果
            {
                "user_id": "string",
                "total_permissions": 0,
                "permission_categories": {},
                "role_summary": [],
                "risk_assessment": {},
                "recommendations": [],
                "usage_patterns": {},
                "agent_insights": {}
            }
    """
    current_user_id, error = verify_token_and_get_user_id()
    if error:
        return error_response("无效的访问令牌", 401, "InvalidToken", error)
    
    if current_user_id != user_id and not check_permission(current_user_id, "users", "analyze"):
        return error_response("权限不足", 403, "PermissionDenied")
    
    try:
        analysis = permission_service.analyze_user_permissions(user_id)
        
        return success_response({
            "user_id": analysis.user_id,
            "total_permissions": analysis.total_permissions,
            "permission_categories": analysis.permission_categories,
            "role_summary": analysis.role_summary,
            "risk_assessment": analysis.risk_assessment,
            "recommendations": analysis.recommendations,
            "usage_patterns": analysis.usage_patterns,
            "agent_insights": analysis.agent_insights
        }, "权限分析完成")
        
    except Exception as e:
        logger.error(f"Analyze user permissions failed: {str(e)}")
        return error_response("权限分析失败", 500, "AnalyzePermissionsError", {"error": str(e)})


@permission_api_bp.route('/users/<user_id>/anomalies', methods=['GET'])
@require_permission('users', 'analyze')
def detect_permission_anomalies_api(user_id: str, current_user_id: str):
    """检测权限异常
    
    使用Agent检测用户权限配置中的异常。
    
    ---
    路径参数:
        user_id: 用户ID
    
    响应:
        200: 返回异常列表
    """
    try:
        anomalies = permission_service.detect_permission_anomalies(user_id)
        
        return success_response({
            "user_id": user_id,
            "anomalies": anomalies,
            "anomaly_count": len(anomalies)
        }, "异常检测完成")
        
    except Exception as e:
        logger.error(f"Detect permission anomalies failed: {str(e)}")
        return error_response("异常检测失败", 500, "DetectAnomaliesError", {"error": str(e)})


@permission_api_bp.route('/users/<user_id>/optimize', methods=['POST'])
@require_permission('users', 'manage')
def optimize_user_permissions_api(user_id: str, current_user_id: str):
    """优化用户权限
    
    基于最小权限原则优化用户权限配置。
    
    ---
    路径参数:
        user_id: 用户ID
    
    请求体:
        {
            "dry_run": true     # 可选，是否仅模拟不实际执行
        }
    
    响应:
        200: 返回优化结果和建议
    """
    try:
        data = request.get_json() or {}
        dry_run = data.get('dry_run', True)
        
        result = permission_service.optimize_permissions(user_id, dry_run)
        
        return success_response(result, "权限优化分析完成")
        
    except Exception as e:
        logger.error(f"Optimize user permissions failed: {str(e)}")
        return error_response("权限优化失败", 500, "OptimizePermissionsError", {"error": str(e)})


@permission_api_bp.route('/recommendations', methods=['GET'])
def list_recommendations():
    """获取权限建议列表
    
    获取待处理的权限建议。
    
    ---
    查询参数:
        user_id: 过滤目标用户
        status: 过滤状态 (pending/accepted/rejected)
        page: 页码
        page_size: 每页大小
    
    响应:
        200: 返回建议列表
    """
    current_user_id, error = verify_token_and_get_user_id()
    if error:
        return error_response("无效的访问令牌", 401, "InvalidToken", error)
    
    if not check_permission(current_user_id, "recommendations", "read"):
        return error_response("权限不足", 403, "PermissionDenied")
    
    try:
        user_id = request.args.get('user_id')
        status = request.args.get('status')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 20))
        
        recommendations, total = permission_service.repository.list_recommendations(
            user_id=user_id,
            status=status,
            page=page,
            page_size=page_size
        )
        
        return success_response({
            "recommendations": [r.to_dict() for r in recommendations],
            "total": total,
            "page": page,
            "page_size": page_size
        }, "获取建议列表成功")
        
    except Exception as e:
        logger.error(f"List recommendations failed: {str(e)}")
        return error_response("获取建议列表失败", 500, "ListRecommendationsError", {"error": str(e)})


@permission_api_bp.route('/recommendations/<recommendation_id>/review', methods=['POST'])
@require_permission('recommendations', 'manage')
def review_recommendation_api(recommendation_id: str, current_user_id: str):
    """审核权限建议
    
    审核并处理权限建议。
    
    ---
    路径参数:
        recommendation_id: 建议ID
    
    请求体:
        {
            "status": "string",         # 必填，新状态 (accepted/rejected)
            "review_notes": "string",   # 可选，审核备注
            "apply": true               # 可选，如果接受是否自动应用
        }
    
    响应:
        200: 审核完成
        404: 建议不存在
    """
    try:
        data = validate_json(request)
        validate_required_fields(data, ['status'])
        
        if data['status'] not in ['accepted', 'rejected']:
            return error_response("无效的状态值", 400, "InvalidStatus")
        
        rec = permission_service.review_recommendation(
            recommendation_id=recommendation_id,
            status=data['status'],
            reviewed_by=current_user_id,
            review_notes=data.get('review_notes'),
            apply_if_accepted=data.get('apply', True)
        )
        
        if not rec:
            return error_response("建议不存在", 404, "RecommendationNotFound")
        
        return success_response({
            "recommendation": rec.to_dict()
        }, "建议审核完成")
        
    except Exception as e:
        logger.error(f"Review recommendation failed: {str(e)}")
        return error_response("建议审核失败", 500, "ReviewRecommendationError", {"error": str(e)})


# ============================================================================
# Agent记忆端点
# ============================================================================

@permission_api_bp.route('/agent/memories', methods=['GET'])
@require_permission('agent', 'read')
def get_agent_memories(current_user_id: str):
    """获取Agent记忆
    
    获取权限分析相关的Agent记忆。
    
    ---
    查询参数:
        user_id: 过滤用户ID
        memory_type: 过滤记忆类型
        min_importance: 最小重要性 (0.0-1.0)
        limit: 返回数量限制
    
    响应:
        200: 返回记忆列表
    """
    try:
        user_id = request.args.get('user_id')
        memory_type = request.args.get('memory_type')
        min_importance = float(request.args.get('min_importance', 0.0))
        limit = int(request.args.get('limit', 100))
        
        memories = permission_service.get_agent_memories(
            user_id=user_id,
            memory_type=memory_type,
            min_importance=min_importance,
            limit=limit
        )
        
        return success_response({
            "memories": [m.to_dict() for m in memories]
        }, "获取Agent记忆成功")
        
    except Exception as e:
        logger.error(f"Get agent memories failed: {str(e)}")
        return error_response("获取Agent记忆失败", 500, "GetAgentMemoriesError", {"error": str(e)})


@permission_api_bp.route('/agent/memories', methods=['POST'])
@require_permission('agent', 'write')
def add_agent_memory(current_user_id: str):
    """添加Agent记忆
    
    添加新的权限分析相关记忆。
    
    ---
    请求体:
        {
            "memory_type": "string",    # 必填，记忆类型
            "content": "string",        # 必填，记忆内容
            "user_id": "string",        # 可选，关联用户ID
            "importance": 0.5,          # 可选，重要性 (0.0-1.0)
            "metadata": {}              # 可选，元数据
        }
    
    响应:
        200: 记忆添加成功
    """
    try:
        data = validate_json(request)
        validate_required_fields(data, ['memory_type', 'content'])
        
        memory = permission_service.add_agent_memory(
            memory_type=data['memory_type'],
            content=data['content'],
            user_id=data.get('user_id'),
            importance=data.get('importance', 0.5),
            metadata=data.get('metadata')
        )
        
        return success_response({
            "memory": memory.to_dict()
        }, "Agent记忆添加成功")
        
    except Exception as e:
        logger.error(f"Add agent memory failed: {str(e)}")
        return error_response("Agent记忆添加失败", 500, "AddAgentMemoryError", {"error": str(e)})


@permission_api_bp.route('/agent/memories/search', methods=['POST'])
@require_permission('agent', 'read')
def search_agent_memories(current_user_id: str):
    """搜索Agent记忆
    
    搜索权限相关的Agent记忆。
    
    ---
    请求体:
        {
            "query": "string",      # 必填，搜索查询
            "user_id": "string",    # 可选，过滤用户ID
            "limit": 20             # 可选，返回数量限制
        }
    
    响应:
        200: 返回匹配的记忆列表
    """
    try:
        data = validate_json(request)
        validate_required_fields(data, ['query'])
        
        memories = permission_service.search_agent_memories(
            query=data['query'],
            user_id=data.get('user_id'),
            limit=data.get('limit', 20)
        )
        
        return success_response({
            "memories": [m.to_dict() for m in memories],
            "count": len(memories)
        }, "Agent记忆搜索完成")
        
    except Exception as e:
        logger.error(f"Search agent memories failed: {str(e)}")
        return error_response("Agent记忆搜索失败", 500, "SearchAgentMemoriesError", {"error": str(e)})


# ============================================================================
# 审计日志端点
# ============================================================================

@permission_api_bp.route('/audit-logs', methods=['GET'])
@require_permission('audit', 'read')
def get_audit_logs(current_user_id: str):
    """获取审计日志
    
    获取权限操作的审计日志。
    
    ---
    查询参数:
        user_id: 过滤操作用户
        target_user_id: 过滤目标用户
        action: 过滤操作类型
        status: 过滤状态
        start_time: 开始时间 (ISO8601格式)
        end_time: 结束时间 (ISO8601格式)
        page: 页码
        page_size: 每页大小
    
    响应:
        200: 返回审计日志列表
    """
    try:
        user_id = request.args.get('user_id')
        target_user_id = request.args.get('target_user_id')
        action = request.args.get('action')
        status = request.args.get('status')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('page_size', 50))
        
        if start_time:
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if end_time:
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        logs, total = permission_service.get_audit_logs(
            user_id=user_id,
            target_user_id=target_user_id,
            action=action,
            status=status,
            start_time=start_time,
            end_time=end_time,
            page=page,
            page_size=page_size
        )
        
        return success_response({
            "logs": [log.to_dict() for log in logs],
            "total": total,
            "page": page,
            "page_size": page_size
        }, "获取审计日志成功")
        
    except Exception as e:
        logger.error(f"Get audit logs failed: {str(e)}")
        return error_response("获取审计日志失败", 500, "GetAuditLogsError", {"error": str(e)})


@permission_api_bp.route('/statistics', methods=['GET'])
@require_permission('statistics', 'read')
def get_access_statistics(current_user_id: str):
    """获取访问统计
    
    获取权限访问的统计信息。
    
    ---
    查询参数:
        user_id: 过滤用户ID
        start_time: 开始时间 (ISO8601格式)
        end_time: 结束时间 (ISO8601格式)
    
    响应:
        200: 返回统计信息
    """
    try:
        user_id = request.args.get('user_id')
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        
        if start_time:
            start_time = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        if end_time:
            end_time = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
        
        stats = permission_service.get_access_statistics(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time
        )
        
        return success_response(stats, "获取访问统计成功")
        
    except Exception as e:
        logger.error(f"Get access statistics failed: {str(e)}")
        return error_response("获取访问统计失败", 500, "GetStatisticsError", {"error": str(e)})


# ============================================================================
# 策略管理端点
# ============================================================================

@permission_api_bp.route('/policies', methods=['POST'])
@require_permission('policies', 'create')
def create_policy(current_user_id: str):
    """创建权限策略
    
    创建新的权限策略规则。
    
    ---
    请求体:
        {
            "name": "string",               # 必填，策略名称
            "resource_pattern": "string",   # 必填，资源模式（支持*通配符）
            "action_pattern": "string",     # 必填，操作模式（支持*通配符）
            "policy_type": "string",        # 可选，策略类型 (allow/deny/conditional)
            "description": "string",        # 可选，策略描述
            "conditions": {},               # 可选，策略条件
            "priority": 0,                  # 可选，优先级
            "effect": "string"              # 可选，效果 (allow/deny)
        }
    
    响应:
        200: 策略创建成功
    """
    try:
        data = validate_json(request)
        validate_required_fields(data, ['name', 'resource_pattern', 'action_pattern'])
        
        policy = permission_service.create_policy(
            name=data['name'],
            resource_pattern=data['resource_pattern'],
            action_pattern=data['action_pattern'],
            policy_type=data.get('policy_type', 'allow'),
            description=data.get('description'),
            conditions=data.get('conditions'),
            priority=data.get('priority', 0),
            effect=data.get('effect', 'allow'),
            created_by=current_user_id
        )
        
        return success_response({
            "policy": policy.to_dict()
        }, "策略创建成功")
        
    except Exception as e:
        logger.error(f"Create policy failed: {str(e)}")
        return error_response("策略创建失败", 500, "CreatePolicyError", {"error": str(e)})


# ============================================================================
# 健康检查端点
# ============================================================================

@permission_api_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查
    
    检查权限服务的健康状态。
    
    响应:
        200: 服务正常
    """
    try:
        # 简单检查数据库连接
        permission_service.list_permissions(page_size=1)
        
        return success_response({
            "status": "healthy",
            "service": "permission_api",
            "timestamp": datetime.utcnow().isoformat()
        }, "权限服务正常")
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return error_response("权限服务异常", 503, "ServiceUnavailable", {"error": str(e)})

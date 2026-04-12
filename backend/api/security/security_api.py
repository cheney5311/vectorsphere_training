# -*- coding: utf-8 -*-
"""
Security API接口

提供统一的安全服务接口，包括：
- 用户认证（登录/登出/MFA）
- 访问控制（权限检查/角色管理）
- 审计日志（记录/查询）
- 加密服务（加密/解密）
- 合规检查（合规性检查/数据处理记录）
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from flask import Blueprint, request, jsonify, g
from pydantic import BaseModel, Field
from functools import wraps
import logging

logger = logging.getLogger(__name__)

# 创建蓝图
security_bp = Blueprint('security', __name__, url_prefix='/api/v1/security')

# ==================== 全局服务实例 ====================

_security_service = None


def get_security_service():
    """获取安全服务实例（单例）"""
    global _security_service
    if _security_service is None:
        try:
            from backend.services.security_service import get_security_service as _get_service
            _security_service = _get_service()
        except ImportError:
            from backend.services.security_service import SecurityService
            _security_service = SecurityService()
    return _security_service


# ==================== 辅助函数 ====================

def _get_current_user() -> Optional[Dict]:
    """获取当前用户"""
    return getattr(g, 'current_user', None)


def _get_tenant_id() -> Optional[str]:
    """获取当前租户ID"""
    # 优先从 g 对象获取
    if hasattr(g, 'tenant_id'):
        return g.tenant_id
    # 从请求头获取
    return request.headers.get('X-Tenant-ID')


def _get_client_info() -> Dict[str, str]:
    """获取客户端信息"""
    return {
        'ip_address': request.remote_addr or '',
        'user_agent': request.headers.get('User-Agent', ''),
        'device_fingerprint': request.headers.get('X-Device-Fingerprint', '')
    }


def require_auth(f):
    """认证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({
                'success': False,
                'error': 'Missing or invalid authorization header'
            }), 401
        
        token = auth_header[7:]
        service = get_security_service()
        result = service.auth.validate_token(token)
        
        if not result.get('valid'):
            return jsonify({
                'success': False,
                'error': result.get('error', 'Invalid token')
            }), 401
        
        g.current_user = {
            'user_id': result.get('user_id'),
            'tenant_id': result.get('tenant_id')
        }
        g.tenant_id = result.get('tenant_id')
        
        return f(*args, **kwargs)
    return decorated


# ==================== 请求模型 ====================

class AuthRequest(BaseModel):
    """认证请求模型"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    ip_address: Optional[str] = Field(None, description="IP地址")
    user_agent: Optional[str] = Field(None, description="用户代理")
    device_fingerprint: Optional[str] = Field(None, description="设备指纹")


class MFAVerifyRequest(BaseModel):
    """MFA验证请求模型"""
    temp_token: str = Field(..., description="临时令牌")
    mfa_code: str = Field(..., description="MFA验证码")
    mfa_method: str = Field(default="totp", description="MFA方法")


class MFASetupRequest(BaseModel):
    """MFA设置请求模型"""
    user_id: str = Field(..., description="用户ID")
    mfa_method: str = Field(default="totp", description="MFA方法")


class MFAConfirmRequest(BaseModel):
    """MFA确认请求模型"""
    user_id: str = Field(..., description="用户ID")
    verification_code: str = Field(..., description="验证代码")


class RoleAssignmentRequest(BaseModel):
    """角色分配请求模型"""
    user_id: str = Field(..., description="用户ID")
    role: str = Field(..., description="角色")
    expires_at: Optional[str] = Field(None, description="过期时间")


class RoleRevocationRequest(BaseModel):
    """角色撤销请求模型"""
    user_id: str = Field(..., description="用户ID")
    role: str = Field(..., description="角色")


class AccessCheckRequest(BaseModel):
    """访问检查请求模型"""
    user_id: str = Field(..., description="用户ID")
    resource: str = Field(..., description="资源")
    action: str = Field(..., description="操作")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")


class PolicyCreateRequest(BaseModel):
    """策略创建请求模型"""
    name: str = Field(..., description="策略名称")
    description: Optional[str] = Field(None, description="策略描述")
    effect: str = Field(default="allow", description="效果: allow/deny")
    principals: List[str] = Field(default=[], description="主体列表")
    resources: List[str] = Field(..., description="资源列表")
    actions: List[str] = Field(..., description="操作列表")
    conditions: Optional[Dict[str, Any]] = Field(None, description="条件")
    priority: int = Field(default=100, description="优先级")


class AuditLogRequest(BaseModel):
    """审计日志请求模型"""
    event_type: str = Field(..., description="事件类型")
    message: str = Field(..., description="消息")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    source_ip: Optional[str] = Field(None, description="源IP")
    user_agent: Optional[str] = Field(None, description="用户代理")
    resource: Optional[str] = Field(None, description="资源")
    action: Optional[str] = Field(None, description="操作")
    result: str = Field(default="success", description="结果")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")
    level: str = Field(default="info", description="级别")
    tags: Optional[List[str]] = Field(None, description="标签")


class AuditQueryRequest(BaseModel):
    """审计查询请求模型"""
    start_time: Optional[str] = Field(None, description="开始时间")
    end_time: Optional[str] = Field(None, description="结束时间")
    event_types: Optional[List[str]] = Field(None, description="事件类型列表")
    levels: Optional[List[str]] = Field(None, description="级别列表")
    user_ids: Optional[List[str]] = Field(None, description="用户ID列表")
    resources: Optional[List[str]] = Field(None, description="资源列表")
    results: Optional[List[str]] = Field(None, description="结果列表")
    min_risk_score: Optional[float] = Field(None, description="最小风险分数")
    max_risk_score: Optional[float] = Field(None, description="最大风险分数")
    tags: Optional[List[str]] = Field(None, description="标签列表")
    limit: int = Field(default=100, description="限制数量")
    offset: int = Field(default=0, description="偏移量")


class EncryptionRequest(BaseModel):
    """加密请求模型"""
    data: str = Field(..., description="要加密的数据")
    key_id: Optional[str] = Field(None, description="密钥ID")
    algorithm: str = Field(default="aes-256-gcm", description="算法")


class DecryptionRequest(BaseModel):
    """解密请求模型"""
    encrypted_data: str = Field(..., description="加密数据（十六进制）")
    key_id: str = Field(..., description="密钥ID")
    iv: Optional[str] = Field(None, description="初始化向量（十六进制）")
    tag: Optional[str] = Field(None, description="认证标签（十六进制）")


class KeyGenerationRequest(BaseModel):
    """密钥生成请求模型"""
    name: str = Field(..., description="密钥名称")
    algorithm: str = Field(default="aes-256-gcm", description="算法")
    description: Optional[str] = Field(None, description="描述")
    expires_days: Optional[int] = Field(None, description="过期天数")


class ComplianceCheckRequest(BaseModel):
    """合规检查请求模型"""
    standard: str = Field(..., description="合规标准")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")


class DataProcessingRecordRequest(BaseModel):
    """数据处理记录请求模型"""
    data_subject_id: str = Field(..., description="数据主体ID")
    data_categories: List[str] = Field(..., description="数据类别")
    processing_purposes: List[str] = Field(..., description="处理目的")
    legal_basis: str = Field(..., description="法律基础")
    consent_given: bool = Field(default=False, description="是否同意")
    retention_period_days: Optional[int] = Field(None, description="保留期限（天）")
    processing_location: str = Field(..., description="处理位置")
    third_party_sharing: bool = Field(default=False, description="第三方共享")
    third_parties: List[str] = Field(default=[], description="第三方")


# ==================== 认证API ====================

@security_bp.route('/auth/login', methods=['POST'])
def login():
    """用户登录
    
    接受用户凭证，返回JWT令牌或MFA挑战。
    """
    try:
        data = request.json or {}
        req = AuthRequest(**data)
        client_info = _get_client_info()
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.auth.authenticate_user(
            username=req.username,
            password=req.password,
            ip_address=req.ip_address or client_info['ip_address'],
            user_agent=req.user_agent or client_info['user_agent'],
            device_fingerprint=req.device_fingerprint or client_info['device_fingerprint'],
            tenant_id=tenant_id
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'data': {
                    'token': result.get('token'),
                    'expires_at': result.get('expires_at'),
                    'user_info': result.get('user_info')
                }
            }), 200
        elif result.get('requires_mfa'):
            return jsonify({
                'success': False,
                'requires_mfa': True,
                'temp_token': result.get('temp_token'),
                'mfa_methods': result.get('mfa_methods', ['totp'])
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Authentication failed')
            }), 401
            
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({
            'success': False,
            'error': f'Login failed: {str(e)}'
        }), 500


@security_bp.route('/auth/logout', methods=['POST'])
@require_auth
def logout():
    """用户登出"""
    try:
        auth_header = request.headers.get('Authorization', '')
        token = auth_header[7:] if auth_header.startswith('Bearer ') else ''
        
        service = get_security_service()
        result = service.auth.logout(token)
        
        return jsonify({
            'success': result.get('success', True),
            'message': result.get('message', 'Logged out successfully')
        }), 200
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({
            'success': False,
            'error': f'Logout failed: {str(e)}'
        }), 500


@security_bp.route('/auth/token/validate', methods=['POST'])
def validate_token():
    """验证令牌"""
    try:
        data = request.json or {}
        token = data.get('token', '')
        
        if not token:
            return jsonify({
                'success': False,
                'valid': False,
                'error': 'Token is required'
            }), 400
        
        service = get_security_service()
        result = service.auth.validate_token(token)
        
        return jsonify({
            'success': True,
            'valid': result.get('valid', False),
            'user_id': result.get('user_id'),
            'tenant_id': result.get('tenant_id'),
            'expires_at': result.get('expires_at'),
            'error': result.get('error')
        }), 200
        
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        return jsonify({
            'success': False,
            'valid': False,
            'error': f'Validation failed: {str(e)}'
        }), 500


@security_bp.route('/auth/token/refresh', methods=['POST'])
@require_auth
def refresh_token():
    """刷新令牌"""
    try:
        auth_header = request.headers.get('Authorization', '')
        token = auth_header[7:] if auth_header.startswith('Bearer ') else ''
        
        service = get_security_service()
        result = service.auth.refresh_token(token)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'data': {
                    'token': result.get('token'),
                    'expires_at': result.get('expires_at')
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Token refresh failed')
            }), 401
            
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        return jsonify({
            'success': False,
            'error': f'Token refresh failed: {str(e)}'
        }), 500


@security_bp.route('/auth/mfa/verify', methods=['POST'])
def verify_mfa():
    """验证MFA"""
    try:
        data = request.json or {}
        req = MFAVerifyRequest(**data)
        client_info = _get_client_info()
        
        service = get_security_service()
        result = service.auth.verify_mfa(
            temp_token=req.temp_token,
            mfa_code=req.mfa_code,
            mfa_method=req.mfa_method,
            ip_address=client_info['ip_address'],
            user_agent=client_info['user_agent'],
            device_fingerprint=client_info['device_fingerprint']
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'data': {
                    'token': result.get('token'),
                    'expires_at': result.get('expires_at')
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'MFA verification failed')
            }), 401
            
    except Exception as e:
        logger.error(f"MFA verification error: {e}")
        return jsonify({
            'success': False,
            'error': f'MFA verification failed: {str(e)}'
        }), 500


@security_bp.route('/auth/mfa/setup', methods=['POST'])
@require_auth
def setup_mfa():
    """设置MFA"""
    try:
        data = request.json or {}
        req = MFASetupRequest(**data)
        
        service = get_security_service()
        result = service.auth.setup_mfa(
            user_id=req.user_id,
            mfa_method=req.mfa_method
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'data': {
                    'secret': result.get('secret'),
                    'provisioning_uri': result.get('provisioning_uri'),
                    'qr_code': result.get('qr_code')
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'MFA setup failed')
            }), 400
            
    except Exception as e:
        logger.error(f"MFA setup error: {e}")
        return jsonify({
            'success': False,
            'error': f'MFA setup failed: {str(e)}'
        }), 500


@security_bp.route('/auth/mfa/confirm', methods=['POST'])
@require_auth
def confirm_mfa():
    """确认MFA设置"""
    try:
        data = request.json or {}
        req = MFAConfirmRequest(**data)
        
        service = get_security_service()
        result = service.auth.confirm_mfa_setup(
            user_id=req.user_id,
            verification_code=req.verification_code
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': result.get('message', 'MFA enabled successfully')
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'MFA confirmation failed')
            }), 400
            
    except Exception as e:
        logger.error(f"MFA confirmation error: {e}")
        return jsonify({
            'success': False,
            'error': f'MFA confirmation failed: {str(e)}'
        }), 500


@security_bp.route('/auth/mfa/status/<user_id>', methods=['GET'])
@require_auth
def get_mfa_status(user_id: str):
    """获取MFA状态"""
    try:
        service = get_security_service()
        result = service.auth.get_mfa_status(user_id)
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"Get MFA status error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get MFA status: {str(e)}'
        }), 500


@security_bp.route('/auth/mfa/disable', methods=['POST'])
@require_auth
def disable_mfa():
    """禁用MFA"""
    try:
        data = request.json or {}
        user_id = data.get('user_id')
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'User ID is required'
            }), 400
        
        service = get_security_service()
        result = service.auth.disable_mfa(user_id)
        
        return jsonify({
            'success': result.get('success', True),
            'message': result.get('message', 'MFA disabled')
        }), 200
        
    except Exception as e:
        logger.error(f"Disable MFA error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to disable MFA: {str(e)}'
        }), 500


@security_bp.route('/auth/sessions', methods=['GET'])
@require_auth
def get_user_sessions():
    """获取用户会话列表"""
    try:
        user = _get_current_user()
        tenant_id = _get_tenant_id()
        user_id = request.args.get('user_id', user.get('user_id') if user else None)
        
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'User ID is required'
            }), 400
        
        service = get_security_service()
        sessions = service.auth.get_user_sessions(user_id, tenant_id)
        
        return jsonify({
            'success': True,
            'data': sessions
        }), 200
        
    except Exception as e:
        logger.error(f"Get sessions error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get sessions: {str(e)}'
        }), 500


@security_bp.route('/auth/sessions/<session_id>', methods=['DELETE'])
@require_auth
def revoke_session(session_id: str):
    """撤销会话"""
    try:
        service = get_security_service()
        result = service.auth.revoke_session(session_id)
        
        return jsonify({
            'success': result.get('success', True),
            'message': result.get('message', 'Session revoked')
        }), 200
        
    except Exception as e:
        logger.error(f"Revoke session error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to revoke session: {str(e)}'
        }), 500


# ==================== 访问控制API ====================

@security_bp.route('/access/check', methods=['POST'])
@require_auth
def check_access():
    """检查访问权限"""
    try:
        data = request.json or {}
        req = AccessCheckRequest(**data)
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.access.check_permission(
            user_id=req.user_id,
            resource=req.resource,
            action=req.action,
            context=req.context,
            tenant_id=tenant_id
        )
        
        return jsonify({
            'success': True,
            'data': {
                'allowed': result.get('allowed', False),
                'reason': result.get('reason', ''),
                'matched_policies': result.get('matched_policies', []),
                'risk_score': result.get('risk_score', 0.0)
            }
        }), 200
            
    except Exception as e:
        logger.error(f"Access check error: {e}")
        return jsonify({
            'success': False,
            'error': f'Access check failed: {str(e)}'
        }), 500


@security_bp.route('/access/roles', methods=['POST'])
@require_auth
def assign_role():
    """分配角色"""
    try:
        data = request.json or {}
        req = RoleAssignmentRequest(**data)
        tenant_id = _get_tenant_id()
        user = _get_current_user()
        assigned_by = user.get('user_id') if user else None
        
        # 解析过期时间
        expires_at = None
        if req.expires_at:
            expires_at = datetime.fromisoformat(req.expires_at)
        
        service = get_security_service()
        result = service.access.assign_role(
            user_id=req.user_id,
            role=req.role,
            tenant_id=tenant_id,
            assigned_by=assigned_by,
            expires_at=expires_at
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': f'Role {req.role} assigned to user {req.user_id}',
                'role_id': result.get('role_id')
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to assign role')
            }), 500
            
    except Exception as e:
        logger.error(f"Role assignment error: {e}")
        return jsonify({
            'success': False,
            'error': f'Role assignment failed: {str(e)}'
        }), 500


@security_bp.route('/access/roles/revoke', methods=['POST'])
@require_auth
def revoke_role():
    """撤销角色"""
    try:
        data = request.json or {}
        req = RoleRevocationRequest(**data)
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.access.revoke_role(
            user_id=req.user_id,
            role=req.role,
            tenant_id=tenant_id
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': f'Role {req.role} revoked from user {req.user_id}'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to revoke role')
            }), 500
            
    except Exception as e:
        logger.error(f"Role revocation error: {e}")
        return jsonify({
            'success': False,
            'error': f'Role revocation failed: {str(e)}'
        }), 500


@security_bp.route('/access/roles/<user_id>', methods=['GET'])
@require_auth
def get_user_roles(user_id: str):
    """获取用户角色"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        roles = service.access.get_user_roles(user_id, tenant_id)
        
        return jsonify({
            'success': True,
            'data': roles
        }), 200
            
    except Exception as e:
        logger.error(f"Get user roles error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get user roles: {str(e)}'
        }), 500


@security_bp.route('/access/policies', methods=['POST'])
@require_auth
def create_policy():
    """创建访问策略"""
    try:
        data = request.json or {}
        req = PolicyCreateRequest(**data)
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.access.add_policy(
            name=req.name,
            description=req.description,
            effect=req.effect,
            principals=req.principals,
            resources=req.resources,
            actions=req.actions,
            conditions=req.conditions,
            priority=req.priority,
            tenant_id=tenant_id
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'policy_id': result.get('policy_id'),
                'message': 'Policy created successfully'
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to create policy')
            }), 500
            
    except Exception as e:
        logger.error(f"Create policy error: {e}")
        return jsonify({
            'success': False,
            'error': f'Policy creation failed: {str(e)}'
        }), 500


@security_bp.route('/access/policies', methods=['GET'])
@require_auth
def list_policies():
    """列出访问策略"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        policies = service.access.list_policies(tenant_id)
        
        return jsonify({
            'success': True,
            'data': policies
        }), 200
            
    except Exception as e:
        logger.error(f"List policies error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to list policies: {str(e)}'
        }), 500


@security_bp.route('/access/policies/<policy_id>', methods=['GET'])
@require_auth
def get_policy(policy_id: str):
    """获取策略详情"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        policy = service.access.get_policy(policy_id, tenant_id)
        
        if policy:
            return jsonify({
                'success': True,
                'data': policy
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Policy not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Get policy error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get policy: {str(e)}'
        }), 500


@security_bp.route('/access/policies/<policy_id>', methods=['PUT'])
@require_auth
def update_policy(policy_id: str):
    """更新访问策略"""
    try:
        data = request.json or {}
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.access.update_policy(policy_id, data, tenant_id)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': 'Policy updated successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to update policy')
            }), 500
            
    except Exception as e:
        logger.error(f"Update policy error: {e}")
        return jsonify({
            'success': False,
            'error': f'Policy update failed: {str(e)}'
        }), 500


@security_bp.route('/access/policies/<policy_id>', methods=['DELETE'])
@require_auth
def delete_policy(policy_id: str):
    """删除访问策略"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.access.delete_policy(policy_id, tenant_id)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': 'Policy deleted successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to delete policy')
            }), 500
            
    except Exception as e:
        logger.error(f"Delete policy error: {e}")
        return jsonify({
            'success': False,
            'error': f'Policy deletion failed: {str(e)}'
        }), 500


# ==================== 审计日志API ====================

@security_bp.route('/audit/log', methods=['POST'])
@require_auth
def log_audit_event():
    """记录审计事件"""
    try:
        data = request.json or {}
        req = AuditLogRequest(**data)
        tenant_id = _get_tenant_id()
        client_info = _get_client_info()
        
        service = get_security_service()
        result = service.audit.log_event(
            tenant_id=tenant_id,
            event_type=req.event_type,
            message=req.message,
            user_id=req.user_id,
            session_id=req.session_id,
            source_ip=req.source_ip or client_info['ip_address'],
            user_agent=req.user_agent or client_info['user_agent'],
            resource=req.resource,
            action=req.action,
            result=req.result,
            details=req.details,
            level=req.level,
            tags=req.tags
        )

        if isinstance(result, str):
            return jsonify({
                'success': True,
                'event_id': result,
                'message': 'Audit event logged successfully'
            }), 201
        
        # 兼容旧代码或其他返回格式
        if isinstance(result, dict) and result.get('success', False):
            return jsonify({
                'success': True,
                'event_id': result.get('event_id'),
                'message': 'Audit event logged successfully'
            }), 201

    except Exception as e:
        logger.error(f"Audit logging error: {e}")
        return jsonify({
            'success': False,
            'error': f'Audit logging failed: {str(e)}'
        }), 500


@security_bp.route('/audit/query', methods=['POST'])
@require_auth
def query_audit_events():
    """查询审计事件"""
    try:
        data = request.json or {}
        req = AuditQueryRequest(**data)
        tenant_id = _get_tenant_id()
        
        # 解析时间
        start_time = None
        end_time = None
        if req.start_time:
            start_time = datetime.fromisoformat(req.start_time)
        if req.end_time:
            end_time = datetime.fromisoformat(req.end_time)
        
        service = get_security_service()
        events = service.audit.query_events(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
            event_types=req.event_types,
            levels=req.levels,
            user_ids=req.user_ids,
            resources=req.resources,
            results=req.results,
            min_risk_score=req.min_risk_score,
            max_risk_score=req.max_risk_score,
            tags=req.tags,
            limit=req.limit,
            offset=req.offset
        )
        
        return jsonify({
            'success': True,
            'data': events,
            'count': len(events)
        }), 200
            
    except Exception as e:
        logger.error(f"Audit query error: {e}")
        return jsonify({
            'success': False,
            'error': f'Audit query failed: {str(e)}'
        }), 500


@security_bp.route('/audit/statistics', methods=['GET'])
@require_auth
def get_audit_statistics():
    """获取审计统计"""
    try:
        tenant_id = _get_tenant_id()
        start_time = request.args.get('start_time')
        end_time = request.args.get('end_time')
        
        if start_time:
            start_time = datetime.fromisoformat(start_time)
        if end_time:
            end_time = datetime.fromisoformat(end_time)
        
        service = get_security_service()
        stats = service.audit.get_statistics(tenant_id, start_time, end_time)
        
        return jsonify({
            'success': True,
            'data': stats
        }), 200
            
    except Exception as e:
        logger.error(f"Get audit statistics error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get statistics: {str(e)}'
        }), 500


@security_bp.route('/audit/cleanup', methods=['POST'])
@require_auth
def cleanup_audit_logs():
    """清理旧审计日志"""
    try:
        data = request.json or {}
        retention_days = data.get('retention_days', 90)
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.audit.cleanup_old_events(tenant_id, retention_days)
        
        return jsonify({
            'success': True,
            'deleted_count': result.get('deleted_count', 0),
            'message': f'Cleaned up audit logs older than {retention_days} days'
        }), 200
            
    except Exception as e:
        logger.error(f"Audit cleanup error: {e}")
        return jsonify({
            'success': False,
            'error': f'Audit cleanup failed: {str(e)}'
        }), 500


# ==================== 加密服务API ====================

@security_bp.route('/encryption/encrypt', methods=['POST'])
@require_auth
def encrypt_data():
    """加密数据"""
    try:
        data = request.json or {}
        req = EncryptionRequest(**data)
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.encryption.encrypt(
            data=req.data,
            key_id=req.key_id,
            algorithm=req.algorithm,
            tenant_id=tenant_id
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'data': {
                    'encrypted_data': result.get('encrypted_data'),
                    'key_id': result.get('key_id'),
                    'algorithm': result.get('algorithm'),
                    'iv': result.get('iv'),
                    'tag': result.get('tag')
                }
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Encryption failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Encryption error: {e}")
        return jsonify({
            'success': False,
            'error': f'Encryption failed: {str(e)}'
        }), 500


@security_bp.route('/encryption/decrypt', methods=['POST'])
@require_auth
def decrypt_data():
    """解密数据"""
    try:
        data = request.json or {}
        req = DecryptionRequest(**data)
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.encryption.decrypt(
            encrypted_data=req.encrypted_data,
            key_id=req.key_id,
            iv=req.iv,
            tag=req.tag,
            tenant_id=tenant_id
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'data': result.get('decrypted_data')
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Decryption failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Decryption error: {e}")
        return jsonify({
            'success': False,
            'error': f'Decryption failed: {str(e)}'
        }), 500


@security_bp.route('/encryption/keys', methods=['POST'])
@require_auth
def generate_key():
    """生成加密密钥"""
    try:
        data = request.json or {}
        req = KeyGenerationRequest(**data)
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.encryption.generate_key(
            name=req.name,
            algorithm=req.algorithm,
            description=req.description,
            expires_days=req.expires_days,
            tenant_id=tenant_id
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'key_id': result.get('key_id'),
                'message': 'Key generated successfully'
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Key generation failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Key generation error: {e}")
        return jsonify({
            'success': False,
            'error': f'Key generation failed: {str(e)}'
        }), 500


@security_bp.route('/encryption/keys', methods=['GET'])
@require_auth
def list_keys():
    """列出加密密钥"""
    try:
        tenant_id = _get_tenant_id()
        status = request.args.get('status', 'active')
        
        service = get_security_service()
        keys = service.encryption.list_keys(tenant_id, status)
        
        return jsonify({
            'success': True,
            'data': keys
        }), 200
            
    except Exception as e:
        logger.error(f"List keys error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to list keys: {str(e)}'
        }), 500


@security_bp.route('/encryption/keys/<key_id>', methods=['GET'])
@require_auth
def get_key_info(key_id: str):
    """获取密钥信息"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        key_info = service.encryption.get_key_info(key_id, tenant_id)
        
        if key_info:
            return jsonify({
                'success': True,
                'data': key_info
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Key not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Get key info error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get key info: {str(e)}'
        }), 500


@security_bp.route('/encryption/keys/<key_id>/rotate', methods=['POST'])
@require_auth
def rotate_key(key_id: str):
    """轮换密钥"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.encryption.rotate_key(key_id, tenant_id)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'new_key_id': result.get('new_key_id'),
                'message': 'Key rotated successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Key rotation failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Key rotation error: {e}")
        return jsonify({
            'success': False,
            'error': f'Key rotation failed: {str(e)}'
        }), 500


@security_bp.route('/encryption/keys/<key_id>', methods=['DELETE'])
@require_auth
def delete_key(key_id: str):
    """删除密钥"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.encryption.delete_key(key_id, tenant_id)
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': 'Key deleted successfully'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Key deletion failed')
            }), 500
            
    except Exception as e:
        logger.error(f"Key deletion error: {e}")
        return jsonify({
            'success': False,
            'error': f'Key deletion failed: {str(e)}'
        }), 500


# ==================== 合规检查API ====================

@security_bp.route('/compliance/check', methods=['POST'])
@require_auth
def check_compliance():
    """检查合规性"""
    try:
        data = request.json or {}
        req = ComplianceCheckRequest(**data)
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.compliance.check_compliance(
            standard=req.standard,
            context=req.context,
            tenant_id=tenant_id
        )
        
        return jsonify({
            'success': True,
            'data': {
                'report_id': result.get('report_id'),
                'standard': result.get('standard'),
                'overall_level': result.get('overall_level'),
                'score': result.get('score'),
                'total_rules': result.get('total_rules'),
                'compliant_rules': result.get('compliant_rules'),
                'non_compliant_rules': result.get('non_compliant_rules'),
                'violations': result.get('violations', []),
                'recommendations': result.get('recommendations', [])
            }
        }), 200
            
    except Exception as e:
        logger.error(f"Compliance check error: {e}")
        return jsonify({
            'success': False,
            'error': f'Compliance check failed: {str(e)}'
        }), 500


@security_bp.route('/compliance/records', methods=['POST'])
@require_auth
def record_data_processing():
    """记录数据处理活动"""
    try:
        data = request.json or {}
        req = DataProcessingRecordRequest(**data)
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        result = service.compliance.record_data_processing(
            tenant_id=tenant_id,
            data_subject_id=req.data_subject_id,
            data_categories=req.data_categories,
            processing_purposes=req.processing_purposes,
            legal_basis=req.legal_basis,
            consent_given=req.consent_given,
            retention_period_days=req.retention_period_days,
            processing_location=req.processing_location,
            third_party_sharing=req.third_party_sharing,
            third_parties=req.third_parties
        )
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'record_id': result.get('record_id'),
                'message': 'Data processing record created successfully'
            }), 201
        else:
            return jsonify({
                'success': False,
                'error': result.get('error', 'Failed to create record')
            }), 500
            
    except Exception as e:
        logger.error(f"Data processing recording error: {e}")
        return jsonify({
            'success': False,
            'error': f'Data processing recording failed: {str(e)}'
        }), 500


@security_bp.route('/compliance/records', methods=['GET'])
@require_auth
def list_data_processing_records():
    """列出数据处理记录"""
    try:
        tenant_id = _get_tenant_id()
        data_subject_id = request.args.get('data_subject_id')
        status = request.args.get('status')
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        service = get_security_service()
        records = service.compliance.get_processing_records(
            tenant_id=tenant_id,
            data_subject_id=data_subject_id,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': records,
            'count': len(records)
        }), 200
            
    except Exception as e:
        logger.error(f"List records error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to list records: {str(e)}'
        }), 500


@security_bp.route('/compliance/reports', methods=['GET'])
@require_auth
def list_compliance_reports():
    """列出合规报告"""
    try:
        tenant_id = _get_tenant_id()
        standard = request.args.get('standard')
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        service = get_security_service()
        reports = service.compliance.get_compliance_reports(
            tenant_id=tenant_id,
            standard=standard,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': reports,
            'count': len(reports)
        }), 200
            
    except Exception as e:
        logger.error(f"List reports error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to list reports: {str(e)}'
        }), 500


@security_bp.route('/compliance/reports/<report_id>', methods=['GET'])
@require_auth
def get_compliance_report(report_id: str):
    """获取合规报告详情"""
    try:
        tenant_id = _get_tenant_id()
        
        service = get_security_service()
        report = service.compliance.get_compliance_report(report_id, tenant_id)
        
        if report:
            return jsonify({
                'success': True,
                'data': report
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Report not found'
            }), 404
            
    except Exception as e:
        logger.error(f"Get report error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to get report: {str(e)}'
        }), 500


@security_bp.route('/compliance/standards', methods=['GET'])
def list_compliance_standards():
    """列出支持的合规标准"""
    try:
        service = get_security_service()
        standards = service.compliance.get_supported_standards()
        
        return jsonify({
            'success': True,
            'data': standards
        }), 200
            
    except Exception as e:
        logger.error(f"List standards error: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to list standards: {str(e)}'
        }), 500


# ==================== 健康检查 ====================

@security_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    try:
        service = get_security_service()
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'auth': 'ok',
                'access': 'ok',
                'audit': 'ok',
                'encryption': 'ok',
                'compliance': 'ok'
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500

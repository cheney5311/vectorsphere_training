# -*- coding: utf-8 -*-
"""安全服务层

整合认证、访问控制、审计日志、加密和合规检查功能，
提供统一的安全服务接口。
"""

import logging
import secrets
import hashlib
import jwt
import pyotp
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from dataclasses import asdict

logger = logging.getLogger(__name__)


class SecurityService:
    """统一安全服务
    
    整合所有安全相关功能，包括：
    - 用户认证 (AuthenticationService)
    - 访问控制 (AccessControlService)
    - 审计日志 (AuditService)
    - 加密服务 (EncryptionService)
    - 合规检查 (ComplianceService)
    """
    
    def __init__(self, config: Dict[str, Any] = None, use_memory_storage: bool = False):
        self.config = config or {}
        self._use_memory_storage = use_memory_storage
        
        # 初始化子服务
        self._auth_service = AuthenticationService(config, use_memory_storage)
        self._access_service = AccessControlService(config, use_memory_storage)
        self._audit_service = AuditService(config, use_memory_storage)
        self._encryption_service = EncryptionServiceWrapper(config, use_memory_storage)
        self._compliance_service = ComplianceService(config, use_memory_storage)
    
    @property
    def auth(self) -> 'AuthenticationService':
        return self._auth_service
    
    @property
    def access(self) -> 'AccessControlService':
        return self._access_service
    
    @property
    def audit(self) -> 'AuditService':
        return self._audit_service
    
    @property
    def encryption(self) -> 'EncryptionServiceWrapper':
        return self._encryption_service
    
    @property
    def compliance(self) -> 'ComplianceService':
        return self._compliance_service


class AuthenticationService:
    """认证服务
    
    处理用户认证、MFA、会话管理等功能
    """
    
    def __init__(self, config: Dict[str, Any] = None, use_memory_storage: bool = False):
        self.config = config or {}
        self._use_memory_storage = use_memory_storage
        
        # JWT配置
        self.jwt_secret = self.config.get('jwt_secret', secrets.token_urlsafe(32))
        self.jwt_algorithm = self.config.get('jwt_algorithm', 'HS256')
        self.session_timeout = self.config.get('session_timeout', 3600)
        self.mfa_issuer = self.config.get('mfa_issuer', 'VectorSphere')
        
        # 初始化仓库
        self._init_repositories()
        
        # 内存存储（备用）
        self._temp_tokens: Dict[str, Dict] = {}
        self._mfa_configs: Dict[str, Dict] = {}
    
    def _init_repositories(self):
        """初始化仓库"""
        try:
            from backend.repositories.security_repository import (
                get_session_repository,
                get_user_role_repository
            )
            self._session_repo = get_session_repository(use_memory=self._use_memory_storage)
            self._role_repo = get_user_role_repository(use_memory=self._use_memory_storage)
        except ImportError as e:
            logger.warning(f"Failed to import repositories: {e}")
            self._session_repo = None
            self._role_repo = None
    
    def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str,
        device_fingerprint: str,
        tenant_id: str = None
    ) -> Dict[str, Any]:
        """用户认证
        
        Args:
            username: 用户名
            password: 密码
            ip_address: IP地址
            user_agent: 用户代理
            device_fingerprint: 设备指纹
            tenant_id: 租户ID
            
        Returns:
            认证结果
        """
        try:
            # 验证密码
            user_info = self._verify_credentials(username, password, tenant_id)
            if not user_info:
                self._log_auth_failure(username, ip_address, 'invalid_credentials')
                return {
                    'success': False,
                    'error': 'Invalid credentials',
                    'requires_mfa': False
                }
            
            # 检查用户状态
            if user_info.get('status') != 'active':
                self._log_auth_failure(username, ip_address, 'account_inactive')
                return {
                    'success': False,
                    'error': 'Account is not active',
                    'requires_mfa': False
                }
            
            # 风险评估
            risk_score = self._calculate_risk_score(
                user_info['id'], ip_address, user_agent, device_fingerprint
            )
            
            # 检查是否需要MFA
            mfa_config = self._mfa_configs.get(user_info['id'])
            requires_mfa = (mfa_config and mfa_config.get('enabled')) or risk_score > 0.7
            
            if requires_mfa:
                temp_token = self._generate_temp_token(user_info['id'])
                return {
                    'success': False,
                    'requires_mfa': True,
                    'temp_token': temp_token,
                    'mfa_methods': self._get_available_mfa_methods(user_info['id'])
                }
            
            # 创建会话
            session = self._create_session(
                user_id=user_info['id'],
                tenant_id=tenant_id,
                ip_address=ip_address,
                user_agent=user_agent,
                device_fingerprint=device_fingerprint,
                risk_score=risk_score
            )
            
            self._log_auth_success(user_info['id'], ip_address)
            
            return {
                'success': True,
                'token': session['token'],
                'expires_at': session['expires_at'],
                'user_info': user_info
            }
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {
                'success': False,
                'error': f'Authentication failed: {str(e)}',
                'requires_mfa': False
            }
    
    def verify_mfa(
        self,
        temp_token: str,
        mfa_code: str,
        mfa_method: str,
        ip_address: str = None,
        user_agent: str = None,
        device_fingerprint: str = None
    ) -> Dict[str, Any]:
        """验证MFA
        
        Args:
            temp_token: 临时令牌
            mfa_code: MFA验证码
            mfa_method: MFA方法
            ip_address: IP地址
            user_agent: 用户代理
            device_fingerprint: 设备指纹
            
        Returns:
            验证结果
        """
        try:
            # 验证临时令牌
            try:
                payload = jwt.decode(
                    temp_token, self.jwt_secret,
                    algorithms=[self.jwt_algorithm]
                )
            except jwt.ExpiredSignatureError:
                return {'success': False, 'error': 'Token expired'}
            except jwt.InvalidTokenError:
                return {'success': False, 'error': 'Invalid token'}
            
            user_id = payload.get('user_id')
            if not user_id:
                return {'success': False, 'error': 'Invalid token payload'}
            
            # 验证MFA代码
            mfa_config = self._mfa_configs.get(user_id)
            if not mfa_config:
                return {'success': False, 'error': 'MFA not configured'}
            
            if mfa_method == 'totp':
                totp = pyotp.TOTP(mfa_config.get('secret'))
                if not totp.verify(mfa_code):
                    return {'success': False, 'error': 'Invalid MFA code'}
            else:
                return {'success': False, 'error': f'Unsupported MFA method: {mfa_method}'}
            
            # 创建会话
            session = self._create_session(
                user_id=user_id,
                tenant_id=payload.get('tenant_id'),
                ip_address=ip_address or '',
                user_agent=user_agent or '',
                device_fingerprint=device_fingerprint or '',
                risk_score=0.0,
                mfa_verified=True
            )
            
            return {
                'success': True,
                'token': session['token'],
                'expires_at': session['expires_at']
            }
            
        except Exception as e:
            logger.error(f"MFA verification error: {e}")
            return {'success': False, 'error': f'MFA verification failed: {str(e)}'}
    
    def setup_mfa(self, user_id: str, mfa_method: str = 'totp') -> Dict[str, Any]:
        """设置MFA
        
        Args:
            user_id: 用户ID
            mfa_method: MFA方法
            
        Returns:
            设置信息
        """
        try:
            if mfa_method == 'totp':
                secret = pyotp.random_base32()
                totp = pyotp.TOTP(secret)
                provisioning_uri = totp.provisioning_uri(
                    name=user_id,
                    issuer_name=self.mfa_issuer
                )
                
                # 存储临时配置（需要确认后才启用）
                self._mfa_configs[user_id] = {
                    'method': mfa_method,
                    'secret': secret,
                    'enabled': False,
                    'created_at': datetime.utcnow().isoformat()
                }
                
                return {
                    'success': True,
                    'secret': secret,
                    'provisioning_uri': provisioning_uri,
                    'qr_code': self._generate_qr_code(provisioning_uri)
                }
            else:
                return {'success': False, 'error': f'Unsupported MFA method: {mfa_method}'}
                
        except Exception as e:
            logger.error(f"MFA setup error: {e}")
            return {'success': False, 'error': f'MFA setup failed: {str(e)}'}
    
    def confirm_mfa_setup(self, user_id: str, verification_code: str) -> Dict[str, Any]:
        """确认MFA设置
        
        Args:
            user_id: 用户ID
            verification_code: 验证码
            
        Returns:
            确认结果
        """
        try:
            mfa_config = self._mfa_configs.get(user_id)
            if not mfa_config:
                return {'success': False, 'error': 'MFA not configured'}
            
            if mfa_config.get('enabled'):
                return {'success': False, 'error': 'MFA already enabled'}
            
            # 验证代码
            totp = pyotp.TOTP(mfa_config['secret'])
            if not totp.verify(verification_code):
                return {'success': False, 'error': 'Invalid verification code'}
            
            # 启用MFA
            mfa_config['enabled'] = True
            mfa_config['enabled_at'] = datetime.utcnow().isoformat()
            
            return {
                'success': True,
                'message': 'MFA enabled successfully'
            }
            
        except Exception as e:
            logger.error(f"MFA confirmation error: {e}")
            return {'success': False, 'error': f'MFA confirmation failed: {str(e)}'}
    
    def logout(self, token: str) -> Dict[str, Any]:
        """登出
        
        Args:
            token: 会话令牌
            
        Returns:
            登出结果
        """
        try:
            if self._session_repo:
                session = self._session_repo.get_by_token(token)
                if session:
                    self._session_repo.update_status(session['id'], 'logged_out')
            
            return {'success': True, 'message': 'Logged out successfully'}
            
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return {'success': False, 'error': str(e)}

    def refresh_token(self, token: str) -> Dict[str, Any]:
        """刷新令牌
        
        Args:
            token: 旧令牌
            
        Returns:
            新令牌信息
        """
        try:
            # 验证旧令牌（允许过期但签名必须有效）
            try:
                payload = jwt.decode(
                    token, self.jwt_secret,
                    algorithms=[self.jwt_algorithm],
                    options={'verify_exp': False}
                )
            except jwt.InvalidTokenError:
                return {'success': False, 'error': 'Invalid token'}
            
            user_id = payload.get('user_id')
            tenant_id = payload.get('tenant_id')
            
            # 检查会话状态
            if self._session_repo:
                session = self._session_repo.get_by_token(token)
                if not session or session.get('status') != 'active':
                    return {'success': False, 'error': 'Session not active'}
            
            # 生成新令牌
            expires_at = datetime.utcnow() + timedelta(seconds=self.session_timeout)
            new_payload = {
                'user_id': user_id,
                'tenant_id': tenant_id,
                'exp': expires_at
            }
            new_token = jwt.encode(new_payload, self.jwt_secret, algorithm=self.jwt_algorithm)
            
            # 更新会话
            if self._session_repo and session:
                self._session_repo.update(session['id'], {'session_token': new_token, 'expires_at': expires_at})
            
            return {
                'success': True,
                'token': new_token,
                'expires_at': expires_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Refresh token error: {e}")
            return {'success': False, 'error': str(e)}

    def revoke_session(self, session_id: str) -> Dict[str, Any]:
        """撤销会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            操作结果
        """
        try:
            if self._session_repo:
                self._session_repo.update_status(session_id, 'revoked')
            return {'success': True, 'message': 'Session revoked'}
        except Exception as e:
            logger.error(f"Revoke session error: {e}")
            return {'success': False, 'error': str(e)}
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """验证令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            验证结果
        """
        try:
            payload = jwt.decode(
                token, self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )
            
            # 检查会话是否有效
            if self._session_repo:
                session = self._session_repo.get_by_token(token)
                if not session or session.get('status') != 'active':
                    return {'valid': False, 'error': 'Session not active'}
            
            return {
                'valid': True,
                'user_id': payload.get('user_id'),
                'tenant_id': payload.get('tenant_id'),
                'expires_at': payload.get('exp')
            }
            
        except jwt.ExpiredSignatureError:
            return {'valid': False, 'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return {'valid': False, 'error': 'Invalid token'}
    
    def get_user_sessions(self, user_id: str, tenant_id: str = None) -> List[Dict]:
        """获取用户会话列表
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            会话列表
        """
        if self._session_repo:
            return self._session_repo.get_by_user(user_id, tenant_id, status='active')
        return []
    
    def get_mfa_status(self, user_id: str) -> Dict[str, Any]:
        """获取MFA状态
        
        Args:
            user_id: 用户ID
            
        Returns:
            MFA状态
        """
        mfa_config = self._mfa_configs.get(user_id)
        if mfa_config:
            return {
                'enabled': mfa_config.get('enabled', False),
                'method': mfa_config.get('method'),
                'enabled_at': mfa_config.get('enabled_at')
            }
        return {'enabled': False}
    
    def disable_mfa(self, user_id: str, verification_code: str = None) -> Dict[str, Any]:
        """禁用MFA
        
        Args:
            user_id: 用户ID
            verification_code: 验证码 (可选)
            
        Returns:
            操作结果
        """
        try:
            mfa_config = self._mfa_configs.get(user_id)
            if not mfa_config or not mfa_config.get('enabled'):
                return {'success': False, 'error': 'MFA not enabled'}
            
            # 验证代码 (如果提供了)
            if verification_code:
                totp = pyotp.TOTP(mfa_config['secret'])
                if not totp.verify(verification_code):
                    return {'success': False, 'error': 'Invalid verification code'}
            
            # 禁用MFA
            del self._mfa_configs[user_id]
            
            return {'success': True, 'message': 'MFA disabled successfully'}
            
        except Exception as e:
            logger.error(f"MFA disable error: {e}")
            return {'success': False, 'error': str(e)}
    
    # ==================== 私有方法 ====================
    
    def _verify_credentials(
        self, username: str, password: str, tenant_id: str = None
    ) -> Optional[Dict]:
        """验证凭证"""
        # TODO: 实现从数据库验证用户凭证
        # 这里返回模拟数据
        if username and password:
            return {
                'id': f'user_{hashlib.md5(username.encode()).hexdigest()[:8]}',
                'username': username,
                'email': f'{username}@example.com',
                'status': 'active',
                'tenant_id': tenant_id
            }
        return None
    
    def _calculate_risk_score(
        self, user_id: str, ip_address: str, 
        user_agent: str, device_fingerprint: str
    ) -> float:
        """计算风险分数"""
        risk_score = 0.0
        
        # 检查已知设备
        if self._session_repo:
            sessions = self._session_repo.get_by_user(user_id)
            known_fingerprints = {s.get('device_fingerprint') for s in sessions}
            if device_fingerprint not in known_fingerprints:
                risk_score += 0.3
        
        # 可疑IP检查（简化实现）
        if ip_address.startswith('10.') or ip_address.startswith('192.168.'):
            pass  # 内网IP，风险较低
        else:
            risk_score += 0.1
        
        return min(risk_score, 1.0)
    
    def _generate_temp_token(self, user_id: str, tenant_id: str = None) -> str:
        """生成临时令牌"""
        payload = {
            'user_id': user_id,
            'tenant_id': tenant_id,
            'type': 'mfa_pending',
            'exp': datetime.utcnow() + timedelta(minutes=5)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _get_available_mfa_methods(self, user_id: str) -> List[str]:
        """获取可用的MFA方法"""
        mfa_config = self._mfa_configs.get(user_id)
        if mfa_config and mfa_config.get('enabled'):
            return [mfa_config.get('method', 'totp')]
        return ['totp']  # 默认可用TOTP
    
    def _create_session(
        self,
        user_id: str,
        tenant_id: str,
        ip_address: str,
        user_agent: str,
        device_fingerprint: str,
        risk_score: float,
        mfa_verified: bool = False
    ) -> Dict[str, Any]:
        """创建会话"""
        expires_at = datetime.utcnow() + timedelta(seconds=self.session_timeout)
        
        payload = {
            'user_id': user_id,
            'tenant_id': tenant_id,
            'mfa_verified': mfa_verified,
            'exp': expires_at
        }
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        session_data = {
            'session_token': token,
            'user_id': user_id,
            'tenant_id': tenant_id,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'device_fingerprint': device_fingerprint,
            'risk_score': risk_score,
            'mfa_verified': mfa_verified,
            'status': 'active',
            'expires_at': expires_at
        }
        
        if self._session_repo:
            self._session_repo.create(session_data)
        
        return {
            'token': token,
            'expires_at': expires_at.isoformat()
        }
    
    def _generate_qr_code(self, data: str) -> str:
        """生成QR码的Base64字符串"""
        try:
            import qrcode
            from io import BytesIO
            import base64
            
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(data)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color='black', back_color='white')
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except ImportError:
            return ''
    
    def _log_auth_success(self, user_id: str, ip_address: str):
        """记录认证成功"""
        logger.info(f"Authentication success: user={user_id}, ip={ip_address}")
    
    def _log_auth_failure(self, username: str, ip_address: str, reason: str):
        """记录认证失败"""
        logger.warning(f"Authentication failure: user={username}, ip={ip_address}, reason={reason}")


class AccessControlService:
    """访问控制服务
    
    实现RBAC和ABAC访问控制
    """
    
    def __init__(self, config: Dict[str, Any] = None, use_memory_storage: bool = False):
        self.config = config or {}
        self._use_memory_storage = use_memory_storage
        
        # 初始化仓库
        self._init_repositories()
        
        # 角色权限映射
        self._role_permissions = self._init_role_permissions()
    
    def _init_repositories(self):
        """初始化仓库"""
        try:
            from backend.repositories.security_repository import (
                get_user_role_repository,
                get_access_policy_repository
            )
            self._role_repo = get_user_role_repository(use_memory=self._use_memory_storage)
            self._policy_repo = get_access_policy_repository(use_memory=self._use_memory_storage)
        except ImportError as e:
            logger.warning(f"Failed to import repositories: {e}")
            self._role_repo = None
            self._policy_repo = None
    
    def _init_role_permissions(self) -> Dict[str, List[str]]:
        """初始化角色权限"""
        return {
            'admin': [
                'read', 'write', 'delete', 'admin',
                'user:manage', 'role:manage', 'tenant:manage',
                'training:create', 'training:read', 'training:update', 'training:delete',
                'model:create', 'model:read', 'model:update', 'model:delete'
            ],
            'operator': [
                'read', 'write',
                'training:create', 'training:read', 'training:update',
                'model:read', 'model:update'
            ],
            'viewer': [
                'read',
                'training:read',
                'model:read'
            ],
            'data_scientist': [
                'read', 'write',
                'training:create', 'training:read', 'training:update',
                'model:create', 'model:read', 'model:update'
            ]
        }
    
    def check_permission(
        self,
        user_id: str,
        resource: str,
        action: str,
        tenant_id: str = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """检查权限
        
        Args:
            user_id: 用户ID
            resource: 资源
            action: 操作
            tenant_id: 租户ID
            context: 上下文信息
            
        Returns:
            权限检查结果
        """
        context = context or {}
        
        try:
            # 1. 检查RBAC权限
            rbac_allowed, rbac_reason = self._check_rbac(user_id, resource, action, tenant_id)
            
            # 2. 检查ABAC策略
            abac_allowed, abac_reason = self._check_abac(
                user_id, resource, action, tenant_id, context
            )
            
            # 3. 合并结果（ABAC可以覆盖RBAC）
            allowed = rbac_allowed or abac_allowed
            reason = abac_reason if abac_allowed else rbac_reason
            
            # 4. 计算风险分数
            risk_score = self._calculate_access_risk(user_id, resource, action, context)
            
            return {
                'allowed': allowed,
                'reason': reason,
                'risk_score': risk_score,
                'rbac_result': rbac_allowed,
                'abac_result': abac_allowed
            }
            
        except Exception as e:
            logger.error(f"Permission check error: {e}")
            return {
                'allowed': False,
                'reason': f'Permission check failed: {str(e)}',
                'risk_score': 1.0
            }
    
    def assign_role(
        self,
        user_id: str,
        role: str,
        tenant_id: str,
        assigned_by: str = None,
        expires_at: datetime = None
    ) -> Dict[str, Any]:
        """分配角色
        
        Args:
            user_id: 用户ID
            role: 角色名称
            tenant_id: 租户ID
            assigned_by: 分配者
            expires_at: 过期时间
            
        Returns:
            操作结果
        """
        try:
            if role not in self._role_permissions:
                return {'success': False, 'error': f'Unknown role: {role}'}
            
            if self._role_repo:
                role_id = self._role_repo.assign_role(
                    user_id, role, tenant_id, assigned_by, expires_at
                )
                if role_id:
                    return {
                        'success': True,
                        'role_id': role_id,
                        'message': f'Role {role} assigned to user {user_id}'
                    }
            
            return {'success': False, 'error': 'Failed to assign role'}
            
        except Exception as e:
            logger.error(f"Role assignment error: {e}")
            return {'success': False, 'error': str(e)}
    
    def revoke_role(
        self,
        user_id: str,
        role: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """撤销角色
        
        Args:
            user_id: 用户ID
            role: 角色名称
            tenant_id: 租户ID
            
        Returns:
            操作结果
        """
        try:
            if self._role_repo:
                success = self._role_repo.revoke_role(user_id, role, tenant_id)
                if success:
                    return {
                        'success': True,
                        'message': f'Role {role} revoked from user {user_id}'
                    }
            
            return {'success': False, 'error': 'Role not found or already revoked'}
            
        except Exception as e:
            logger.error(f"Role revocation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_user_roles(self, user_id: str, tenant_id: str = None) -> List[str]:
        """获取用户角色"""
        if self._role_repo:
            return self._role_repo.get_user_roles(user_id, tenant_id)
        return []
    
    def get_user_permissions(self, user_id: str, tenant_id: str = None) -> List[str]:
        """获取用户权限"""
        roles = self.get_user_roles(user_id, tenant_id)
        permissions = set()
        for role in roles:
            role_perms = self._role_permissions.get(role, [])
            permissions.update(role_perms)
        return list(permissions)
    
    def add_policy(self, *args, **kwargs):
        """添加访问策略 (create_policy 别名)"""
        return self.create_policy(*args, **kwargs)

    def create_policy(self, policy_data: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """创建访问策略
        
        Args:
            policy_data: 策略数据
            
        Returns:
            操作结果
        """
        # 处理可能的参数差异
        if 'name' in kwargs:
            policy_data = {
                'name': kwargs.get('name'),
                'description': kwargs.get('description'),
                'effect': kwargs.get('effect'),
                'principals': kwargs.get('principals'),
                'resources': kwargs.get('resources'),
                'actions': kwargs.get('actions'),
                'conditions': kwargs.get('conditions'),
                'priority': kwargs.get('priority'),
                'tenant_id': kwargs.get('tenant_id')
            }
            
        try:
            if self._policy_repo:
                policy_id = self._policy_repo.create(policy_data)
                if policy_id:
                    return {
                        'success': True,
                        'policy_id': policy_id,
                        'message': 'Policy created successfully'
                    }
            
            return {'success': False, 'error': 'Failed to create policy'}
            
        except Exception as e:
            logger.error(f"Policy creation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_policy(self, policy_id: str, tenant_id: str = None) -> Optional[Dict]:
        """获取策略详情"""
        if self._policy_repo:
            return self._policy_repo.get_by_id(policy_id, tenant_id)
        return None

    def update_policy(
        self,
        policy_id: str,
        update_data: Dict[str, Any],
        tenant_id: str = None
    ) -> Dict[str, Any]:
        """更新访问策略"""
        try:
            if self._policy_repo:
                success = self._policy_repo.update(policy_id, update_data, tenant_id)
                if success:
                    return {'success': True, 'message': 'Policy updated'}
            
            return {'success': False, 'error': 'Policy not found'}
            
        except Exception as e:
            logger.error(f"Policy update error: {e}")
            return {'success': False, 'error': str(e)}
    
    def delete_policy(self, policy_id: str, tenant_id: str = None) -> Dict[str, Any]:
        """删除访问策略"""
        try:
            if self._policy_repo:
                success = self._policy_repo.delete(policy_id, tenant_id)
                if success:
                    return {'success': True, 'message': 'Policy deleted'}
            
            return {'success': False, 'error': 'Policy not found'}
            
        except Exception as e:
            logger.error(f"Policy deletion error: {e}")
            return {'success': False, 'error': str(e)}
    
    def list_policies(self, tenant_id: str, is_active: bool = True) -> List[Dict]:
        """列出访问策略"""
        if self._policy_repo:
            return self._policy_repo.list_by_tenant(tenant_id, is_active)
        return []
    
    # ==================== 私有方法 ====================
    
    def _check_rbac(
        self,
        user_id: str,
        resource: str,
        action: str,
        tenant_id: str
    ) -> tuple:
        """检查RBAC权限"""
        roles = self.get_user_roles(user_id, tenant_id)
        
        if not roles:
            return False, 'No roles assigned'
        
        # 检查是否有管理员角色
        if 'admin' in roles:
            return True, 'Admin access granted'
        
        # 检查具体权限
        permissions = self.get_user_permissions(user_id, tenant_id)
        
        # 构建所需权限
        required_permission = f'{resource}:{action}'
        
        if required_permission in permissions:
            return True, f'Permission {required_permission} granted'
        
        if action in permissions:
            return True, f'Action {action} granted'
        
        return False, 'Permission denied'
    
    def _check_abac(
        self,
        user_id: str,
        resource: str,
        action: str,
        tenant_id: str,
        context: Dict[str, Any]
    ) -> tuple:
        """检查ABAC策略"""
        if not self._policy_repo:
            return False, 'No policies configured'
        
        policies = self._policy_repo.list_by_tenant(tenant_id, is_active=True)
        
        for policy in policies:
            # 检查策略是否匹配
            if self._policy_matches(policy, user_id, resource, action, context):
                effect = policy.get('effect', 'allow')
                if effect == 'allow':
                    return True, f'Policy {policy.get("name")} allows access'
                else:
                    return False, f'Policy {policy.get("name")} denies access'
        
        return False, 'No matching policy found'
    
    def _policy_matches(
        self,
        policy: Dict,
        user_id: str,
        resource: str,
        action: str,
        context: Dict[str, Any]
    ) -> bool:
        """检查策略是否匹配"""
        # 检查主体
        principals = policy.get('principals', [])
        if principals and user_id not in principals and '*' not in principals:
            return False
        
        # 检查资源
        resources = policy.get('resources', [])
        if resources:
            resource_match = False
            for res_pattern in resources:
                if res_pattern == '*' or resource.startswith(res_pattern.rstrip('*')):
                    resource_match = True
                    break
            if not resource_match:
                return False
        
        # 检查动作
        actions = policy.get('actions', [])
        if actions and action not in actions and '*' not in actions:
            return False
        
        # 检查条件
        conditions = policy.get('conditions', {})
        for key, expected_value in conditions.items():
            actual_value = context.get(key)
            if actual_value != expected_value:
                return False
        
        return True
    
    def _calculate_access_risk(
        self,
        user_id: str,
        resource: str,
        action: str,
        context: Dict[str, Any]
    ) -> float:
        """计算访问风险分数"""
        risk_score = 0.0
        
        # 敏感资源
        sensitive_resources = ['admin', 'user', 'role', 'tenant', 'secret']
        for sr in sensitive_resources:
            if sr in resource.lower():
                risk_score += 0.2
                break
        
        # 敏感操作
        sensitive_actions = ['delete', 'admin', 'manage']
        if action in sensitive_actions:
            risk_score += 0.3
        
        return min(risk_score, 1.0)


class AuditService:
    """审计日志服务
    
    记录和查询安全审计事件
    """
    
    def __init__(self, config: Dict[str, Any] = None, use_memory_storage: bool = False):
        self.config = config or {}
        self._use_memory_storage = use_memory_storage
        
        # 初始化仓库
        self._init_repository()
    
    def _init_repository(self):
        """初始化仓库"""
        try:
            from backend.repositories.security_repository import get_audit_log_repository
            self._audit_repo = get_audit_log_repository(use_memory=self._use_memory_storage)
        except ImportError as e:
            logger.warning(f"Failed to import audit repository: {e}")
            self._audit_repo = None
    
    def log_event(
        self,
        tenant_id: str,
        event_type: str,
        message: str,
        user_id: str = None,
        session_id: str = None,
        source_ip: str = None,
        user_agent: str = None,
        resource: str = None,
        action: str = None,
        result: str = 'success',
        details: Dict[str, Any] = None,
        level: str = 'info',
        tags: List[str] = None
    ) -> Optional[str]:
        """记录审计事件
        
        Args:
            tenant_id: 租户ID
            event_type: 事件类型
            message: 消息
            user_id: 用户ID
            session_id: 会话ID
            source_ip: 源IP
            user_agent: 用户代理
            resource: 资源
            action: 操作
            result: 结果
            details: 详细信息
            level: 级别
            tags: 标签
            
        Returns:
            事件ID
        """
        try:
            risk_score = self._calculate_risk_score(event_type, result, details)
            
            log_data = {
                'tenant_id': tenant_id,
                'event_type': event_type,
                'event_level': level,
                'user_id': user_id,
                'session_id': session_id,
                'source_ip': source_ip,
                'user_agent': user_agent,
                'resource': resource,
                'action': action,
                'result': result,
                'message': message,
                'details': details,
                'risk_score': risk_score,
                'tags': tags
            }
            
            if self._audit_repo:
                return self._audit_repo.create(log_data)
            
            # 至少记录到日志
            logger.info(f"Audit event: {event_type} - {message}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return None
    
    def query_events(
        self,
        tenant_id: str,
        start_time: datetime = None,
        end_time: datetime = None,
        event_types: List[str] = None,
        event_levels: List[str] = None,
        levels: List[str] = None,  # Alias for event_levels
        user_ids: List[str] = None,
        resources: List[str] = None,
        results: List[str] = None,
        min_risk_score: float = None,
        max_risk_score: float = None,
        tags: List[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """查询审计事件
        
        Args:
            tenant_id: 租户ID
            start_time: 开始时间
            end_time: 结束时间
            event_types: 事件类型列表
            event_levels: 事件级别列表
            levels: 事件级别列表 (别名)
            user_ids: 用户ID列表
            resources: 资源列表
            results: 结果列表
            min_risk_score: 最小风险分数
            max_risk_score: 最大风险分数
            tags: 标签列表
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            事件列表
        """
        # Handle alias
        if levels and not event_levels:
            event_levels = levels
            
        if self._audit_repo:
            return self._audit_repo.query(
                tenant_id=tenant_id,
                start_time=start_time,
                end_time=end_time,
                event_types=event_types,
                event_levels=event_levels,
                user_ids=user_ids,
                resources=resources,
                results=results,
                min_risk_score=min_risk_score,
                max_risk_score=max_risk_score,
                tags=tags,
                limit=limit,
                offset=offset
            )
        return []
    
    def get_statistics(
        self,
        tenant_id: str,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> Dict[str, Any]:
        """获取审计统计
        
        Args:
            tenant_id: 租户ID
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            统计信息
        """
        if self._audit_repo:
            return self._audit_repo.get_statistics(tenant_id, start_time, end_time)
        return {'total': 0, 'by_event_type': {}, 'by_level': {}, 'high_risk_count': 0}
    
    def cleanup_old_events(self, *args, **kwargs):
        """清理旧事件 (cleanup_old_logs 别名)"""
        return self.cleanup_old_logs(*args, **kwargs)

    def cleanup_old_logs(self, tenant_id: str, retention_days: int) -> int:
        """清理旧日志
        
        Args:
            tenant_id: 租户ID
            retention_days: 保留天数
            
        Returns:
            删除数量
        """
        if self._audit_repo:
            return self._audit_repo.cleanup_old_logs(tenant_id, retention_days)
        return 0
    
    def _calculate_risk_score(
        self,
        event_type: str,
        result: str,
        details: Dict[str, Any] = None
    ) -> float:
        """计算风险分数"""
        risk_score = 0.0
        
        # 失败事件风险更高
        if result == 'failure':
            risk_score += 0.3
        
        # 高风险事件类型
        high_risk_types = [
            'login_failure', 'permission_denied', 'unauthorized_access',
            'data_breach', 'security_violation', 'suspicious_activity'
        ]
        if event_type in high_risk_types:
            risk_score += 0.4
        
        # 检查详情中的风险标记
        if details:
            if details.get('is_suspicious'):
                risk_score += 0.3
            if details.get('repeated_failures', 0) > 3:
                risk_score += 0.2
        
        return min(risk_score, 1.0)


class EncryptionServiceWrapper:
    """加密服务包装器
    
    提供数据加密和解密功能
    """
    
    def __init__(self, config: Dict[str, Any] = None, use_memory_storage: bool = False):
        self.config = config or {}
        self._use_memory_storage = use_memory_storage
        
        # 初始化仓库
        self._init_repository()
        
        # 初始化加密后端
        self._init_crypto()
    
    def _init_repository(self):
        """初始化仓库"""
        try:
            from backend.repositories.security_repository import get_encryption_key_repository
            self._key_repo = get_encryption_key_repository(use_memory=self._use_memory_storage)
        except ImportError as e:
            logger.warning(f"Failed to import encryption key repository: {e}")
            self._key_repo = None
    
    def _init_crypto(self):
        """初始化加密后端"""
        try:
            from cryptography.fernet import Fernet
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            self._fernet_available = True
            self._aesgcm_available = True
        except ImportError:
            logger.warning("Cryptography library not available")
            self._fernet_available = False
            self._aesgcm_available = False
        
        # 默认密钥（生产环境应从安全存储获取）
        self._default_key = self.config.get('encryption_key') or secrets.token_bytes(32)
    
    def encrypt(
        self,
        data: str,
        key_id: str = None,
        algorithm: str = 'aes_gcm',
        tenant_id: str = None
    ) -> Dict[str, Any]:
        """加密数据
        
        Args:
            data: 要加密的数据
            key_id: 密钥ID
            algorithm: 加密算法
            tenant_id: 租户ID
            
        Returns:
            加密结果
        """
        try:
            # 获取或生成密钥
            key = self._get_key(key_id, tenant_id)
            
            data_bytes = data.encode('utf-8')
            
            if algorithm == 'aes_gcm' and self._aesgcm_available:
                result = self._encrypt_aes_gcm(data_bytes, key)
            elif algorithm == 'fernet' and self._fernet_available:
                result = self._encrypt_fernet(data_bytes, key)
            else:
                # 简单的XOR加密（不安全，仅用于演示）
                result = self._encrypt_simple(data_bytes, key)
            
            # 记录使用
            if key_id and self._key_repo:
                self._key_repo.increment_usage(key_id)
            
            return {
                'success': True,
                'encrypted_data': result['ciphertext'],
                'iv': result.get('iv'),
                'tag': result.get('tag'),
                'key_id': key_id,
                'algorithm': algorithm
            }
            
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return {'success': False, 'error': str(e)}
    
    def decrypt(
        self,
        encrypted_data: str,
        key_id: str = None,
        iv: str = None,
        tag: str = None,
        algorithm: str = 'aes_gcm',
        tenant_id: str = None
    ) -> Dict[str, Any]:
        """解密数据
        
        Args:
            encrypted_data: 加密数据（十六进制字符串）
            key_id: 密钥ID
            iv: 初始化向量
            tag: 认证标签
            algorithm: 加密算法
            tenant_id: 租户ID
            
        Returns:
            解密结果
        """
        try:
            # 获取密钥
            key = self._get_key(key_id, tenant_id)
            
            ciphertext = bytes.fromhex(encrypted_data)
            iv_bytes = bytes.fromhex(iv) if iv else None
            tag_bytes = bytes.fromhex(tag) if tag else None
            
            if algorithm == 'aes_gcm' and self._aesgcm_available:
                plaintext = self._decrypt_aes_gcm(ciphertext, key, iv_bytes, tag_bytes)
            elif algorithm == 'fernet' and self._fernet_available:
                plaintext = self._decrypt_fernet(ciphertext, key)
            else:
                plaintext = self._decrypt_simple(ciphertext, key)
            
            return {
                'success': True,
                'decrypted_data': plaintext.decode('utf-8')
            }
            
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_key(
        self,
        name: str,
        tenant_id: str,
        algorithm: str = 'aes_gcm',
        key_size: int = 256,
        description: str = None,
        expires_at: datetime = None,
        expires_days: int = None
    ) -> Dict[str, Any]:
        """生成新密钥
        
        Args:
            name: 密钥名称
            tenant_id: 租户ID
            algorithm: 加密算法
            key_size: 密钥大小（位）
            description: 描述
            expires_at: 过期时间
            expires_days: 过期天数 (可选)
            
        Returns:
            密钥信息
        """
        try:
            # Handle expires_days if provided
            if expires_days and not expires_at:
                expires_at = datetime.utcnow() + timedelta(days=expires_days)
                
            # 生成密钥
            key_bytes = secrets.token_bytes(key_size // 8)
            
            # 加密存储密钥
            encrypted_key = self._encrypt_key_for_storage(key_bytes)
            
            key_data = {
                'name': name,
                'description': description,
                'tenant_id': tenant_id,
                'key_type': 'symmetric',
                'algorithm': algorithm,
                'key_size': key_size,
                'encrypted_key_data': encrypted_key,
                'key_checksum': hashlib.sha256(key_bytes).hexdigest()[:16],
                'status': 'active',
                'expires_at': expires_at
            }
            
            if self._key_repo:
                key_id = self._key_repo.create(key_data)
                return {
                    'success': True,
                    'key_id': key_id,
                    'name': name,
                    'algorithm': algorithm,
                    'key_size': key_size
                }
            
            return {'success': False, 'error': 'Key repository not available'}
            
        except Exception as e:
            logger.error(f"Key generation error: {e}")
            return {'success': False, 'error': str(e)}

    def delete_key(self, key_id: str, tenant_id: str) -> Dict[str, Any]:
        """删除密钥
        
        Args:
            key_id: 密钥ID
            tenant_id: 租户ID
            
        Returns:
            操作结果
        """
        try:
            if self._key_repo:
                # 检查密钥是否存在且属于该租户
                key = self._key_repo.get_by_id(key_id, tenant_id)
                if not key:
                    return {'success': False, 'error': 'Key not found'}
                
                success = self._key_repo.delete(key_id, tenant_id)
                if success:
                    return {'success': True, 'message': 'Key deleted successfully'}
            
            return {'success': False, 'error': 'Key not found or repository unavailable'}
            
        except Exception as e:
            logger.error(f"Key deletion error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_key_info(self, key_id: str, tenant_id: str = None) -> Optional[Dict]:
        """获取密钥信息"""
        if self._key_repo:
            key = self._key_repo.get_by_id(key_id, tenant_id)
            if key:
                # 移除敏感字段
                safe_key = {k: v for k, v in key.items() 
                           if k not in ['encrypted_key_data']}
                return safe_key
        return None
    
    def rotate_key(self, key_id: str, tenant_id: str) -> Dict[str, Any]:
        """轮换密钥"""
        try:
            old_key = self.get_key_info(key_id, tenant_id)
            if not old_key:
                return {'success': False, 'error': 'Key not found'}
            
            # 生成新密钥
            result = self.generate_key(
                name=f"{old_key['name']}_rotated",
                tenant_id=tenant_id,
                algorithm=old_key.get('algorithm', 'aes_gcm'),
                key_size=old_key.get('key_size', 256)
            )
            
            if result['success']:
                # 禁用旧密钥
                if self._key_repo:
                    self._key_repo.update_status(key_id, 'rotated', tenant_id)
                
                return {
                    'success': True,
                    'old_key_id': key_id,
                    'new_key_id': result['key_id'],
                    'message': 'Key rotated successfully'
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Key rotation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def list_keys(self, tenant_id: str, status: str = None) -> List[Dict]:
        """列出密钥"""
        if self._key_repo:
            keys = self._key_repo.list_by_tenant(tenant_id, status)
            # 移除敏感字段
            return [{k: v for k, v in key.items() 
                    if k not in ['encrypted_key_data']} for key in keys]
        return []
    
    # ==================== 私有方法 ====================
    
    def _get_key(self, key_id: str = None, tenant_id: str = None) -> bytes:
        """获取密钥"""
        if key_id and self._key_repo:
            key_data = self._key_repo.get_by_id(key_id, tenant_id)
            if key_data and key_data.get('encrypted_key_data'):
                return self._decrypt_key_from_storage(key_data['encrypted_key_data'])
        return self._default_key
    
    def _encrypt_key_for_storage(self, key: bytes) -> str:
        """加密密钥用于存储"""
        # 使用主密钥加密（简化实现）
        import base64
        return base64.b64encode(key).decode('utf-8')
    
    def _decrypt_key_from_storage(self, encrypted_key: str) -> bytes:
        """从存储解密密钥"""
        import base64
        return base64.b64decode(encrypted_key.encode('utf-8'))
    
    def _encrypt_aes_gcm(self, data: bytes, key: bytes) -> Dict:
        """使用AES-GCM加密"""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        iv = secrets.token_bytes(12)
        aesgcm = AESGCM(key[:32])  # 使用前32字节
        ciphertext = aesgcm.encrypt(iv, data, None)
        
        return {
            'ciphertext': ciphertext.hex(),
            'iv': iv.hex()
        }
    
    def _decrypt_aes_gcm(self, ciphertext: bytes, key: bytes, 
                        iv: bytes, tag: bytes = None) -> bytes:
        """使用AES-GCM解密"""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        
        aesgcm = AESGCM(key[:32])
        return aesgcm.decrypt(iv, ciphertext, None)
    
    def _encrypt_fernet(self, data: bytes, key: bytes) -> Dict:
        """使用Fernet加密"""
        from cryptography.fernet import Fernet
        import base64
        
        fernet_key = base64.urlsafe_b64encode(key[:32])
        f = Fernet(fernet_key)
        ciphertext = f.encrypt(data)
        
        return {'ciphertext': ciphertext.hex()}
    
    def _decrypt_fernet(self, ciphertext: bytes, key: bytes) -> bytes:
        """使用Fernet解密"""
        from cryptography.fernet import Fernet
        import base64
        
        fernet_key = base64.urlsafe_b64encode(key[:32])
        f = Fernet(fernet_key)
        return f.decrypt(ciphertext)
    
    def _encrypt_simple(self, data: bytes, key: bytes) -> Dict:
        """简单加密（XOR，不安全）"""
        key_repeated = (key * (len(data) // len(key) + 1))[:len(data)]
        ciphertext = bytes(a ^ b for a, b in zip(data, key_repeated))
        return {'ciphertext': ciphertext.hex()}
    
    def _decrypt_simple(self, ciphertext: bytes, key: bytes) -> bytes:
        """简单解密（XOR）"""
        key_repeated = (key * (len(ciphertext) // len(key) + 1))[:len(ciphertext)]
        return bytes(a ^ b for a, b in zip(ciphertext, key_repeated))


class ComplianceService:
    """合规服务
    
    提供合规检查和数据处理记录功能
    """
    
    def __init__(self, config: Dict[str, Any] = None, use_memory_storage: bool = False):
        self.config = config or {}
        self._use_memory_storage = use_memory_storage
        
        # 初始化仓库
        self._init_repositories()
        
        # 初始化规则
        self._rules = self._init_rules()
    
    def _init_repositories(self):
        """初始化仓库"""
        try:
            from backend.repositories.security_repository import (
                get_data_processing_repository,
                get_compliance_report_repository
            )
            self._processing_repo = get_data_processing_repository(
                use_memory=self._use_memory_storage
            )
            self._report_repo = get_compliance_report_repository(
                use_memory=self._use_memory_storage
            )
        except ImportError as e:
            logger.warning(f"Failed to import compliance repositories: {e}")
            self._processing_repo = None
            self._report_repo = None
    
    def _init_rules(self) -> Dict[str, List[Dict]]:
        """初始化合规规则"""
        return {
            'gdpr': [
                {
                    'id': 'gdpr_consent',
                    'name': 'Consent Required',
                    'description': 'Personal data processing requires valid consent',
                    'check': lambda ctx: ctx.get('consent_given', False)
                },
                {
                    'id': 'gdpr_purpose',
                    'name': 'Purpose Limitation',
                    'description': 'Data must be collected for specified purposes',
                    'check': lambda ctx: bool(ctx.get('processing_purposes'))
                },
                {
                    'id': 'gdpr_minimization',
                    'name': 'Data Minimization',
                    'description': 'Only necessary data should be collected',
                    'check': lambda ctx: len(ctx.get('data_categories', [])) <= 5
                },
                {
                    'id': 'gdpr_retention',
                    'name': 'Storage Limitation',
                    'description': 'Data retention period must be defined',
                    'check': lambda ctx: ctx.get('retention_period_days') is not None
                },
                {
                    'id': 'gdpr_security',
                    'name': 'Security Measures',
                    'description': 'Appropriate security measures must be in place',
                    'check': lambda ctx: ctx.get('encryption_enabled', True)
                }
            ],
            'ccpa': [
                {
                    'id': 'ccpa_notice',
                    'name': 'Notice at Collection',
                    'description': 'Consumers must be notified at data collection',
                    'check': lambda ctx: ctx.get('notice_provided', True)
                },
                {
                    'id': 'ccpa_optout',
                    'name': 'Right to Opt-Out',
                    'description': 'Consumers can opt out of data sale',
                    'check': lambda ctx: not ctx.get('data_sold', False)
                }
            ],
            'soc2': [
                {
                    'id': 'soc2_access',
                    'name': 'Access Control',
                    'description': 'Access control mechanisms must be in place',
                    'check': lambda ctx: ctx.get('access_control_enabled', True)
                },
                {
                    'id': 'soc2_audit',
                    'name': 'Audit Logging',
                    'description': 'Audit logging must be enabled',
                    'check': lambda ctx: ctx.get('audit_logging_enabled', True)
                }
            ]
        }
    
    def check_compliance(
        self,
        tenant_id: str,
        standard: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """检查合规性
        
        Args:
            tenant_id: 租户ID
            standard: 合规标准
            context: 上下文信息
            
        Returns:
            合规报告
        """
        context = context or {}
        
        try:
            rules = self._rules.get(standard.lower(), [])
            if not rules:
                return {
                    'success': False,
                    'error': f'Unknown compliance standard: {standard}'
                }
            
            violations = []
            compliant_count = 0
            
            for rule in rules:
                try:
                    if rule['check'](context):
                        compliant_count += 1
                    else:
                        violations.append({
                            'rule_id': rule['id'],
                            'rule_name': rule['name'],
                            'description': rule['description'],
                            'severity': 'high'
                        })
                except Exception as e:
                    violations.append({
                        'rule_id': rule['id'],
                        'rule_name': rule['name'],
                        'description': f'Rule check failed: {str(e)}',
                        'severity': 'medium'
                    })
            
            total_rules = len(rules)
            score = (compliant_count / total_rules * 100) if total_rules > 0 else 0
            
            # 确定合规级别
            if score >= 90:
                level = 'compliant'
            elif score >= 70:
                level = 'partial'
            else:
                level = 'non_compliant'
            
            # 生成建议
            recommendations = self._generate_recommendations(violations, standard)
            
            report_data = {
                'tenant_id': tenant_id,
                'standard': standard,
                'overall_level': level,
                'score': score,
                'total_rules': total_rules,
                'compliant_rules': compliant_count,
                'non_compliant_rules': len(violations),
                'violations': violations,
                'recommendations': recommendations,
                'period_start': datetime.utcnow(),
                'period_end': datetime.utcnow()
            }
            
            # 保存报告
            report_id = None
            if self._report_repo:
                report_id = self._report_repo.create(report_data)
            
            return {
                'success': True,
                'report_id': report_id,
                'standard': standard,
                'level': level,
                'score': score,
                'total_rules': total_rules,
                'compliant_rules': compliant_count,
                'violations': len(violations),
                'violation_details': violations,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Compliance check error: {e}")
            return {'success': False, 'error': str(e)}
    
    def record_data_processing(
        self,
        tenant_id: str,
        data_subject_id: str,
        data_categories: List[str],
        processing_purposes: List[str],
        legal_basis: str,
        consent_given: bool = False,
        retention_period_days: int = None,
        processing_location: str = None,
        third_party_sharing: bool = False,
        third_parties: List[str] = None
    ) -> Dict[str, Any]:
        """记录数据处理活动
        
        Args:
            tenant_id: 租户ID
            data_subject_id: 数据主体ID
            data_categories: 数据类别
            processing_purposes: 处理目的
            legal_basis: 法律基础
            consent_given: 是否同意
            retention_period_days: 保留期限
            processing_location: 处理位置
            third_party_sharing: 第三方共享
            third_parties: 第三方列表
            
        Returns:
            操作结果
        """
        try:
            record_data = {
                'tenant_id': tenant_id,
                'data_subject_id': data_subject_id,
                'data_categories': data_categories,
                'processing_purposes': processing_purposes,
                'legal_basis': legal_basis,
                'consent_given': consent_given,
                'consent_timestamp': datetime.utcnow() if consent_given else None,
                'retention_period_days': retention_period_days,
                'processing_location': processing_location,
                'third_party_sharing': third_party_sharing,
                'third_parties': third_parties or [],
                'status': 'active'
            }
            
            if self._processing_repo:
                record_id = self._processing_repo.create(record_data)
                return {
                    'success': True,
                    'record_id': record_id,
                    'message': 'Data processing record created'
                }
            
            return {'success': False, 'error': 'Repository not available'}
            
        except Exception as e:
            logger.error(f"Data processing record error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_processing_records(
        self,
        tenant_id: str,
        data_subject_id: str = None,
        status: str = None,  # Added status parameter
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """获取数据处理记录
        
        Args:
            tenant_id: 租户ID
            data_subject_id: 数据主体ID
            status: 状态过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            记录列表
        """
        if not self._processing_repo:
            return []
        
        if data_subject_id:
            # Note: repository method might not support status filtering, passing it for now
            return self._processing_repo.get_by_subject(data_subject_id, tenant_id)
        else:
            return self._processing_repo.list_by_tenant(tenant_id, limit=limit, offset=offset)
    
    def get_compliance_reports(self, *args, **kwargs):
        """获取合规报告 (list_compliance_reports 别名)"""
        return self.list_compliance_reports(*args, **kwargs)

    def get_supported_standards(self) -> List[Dict]:
        """获取支持的合规标准"""
        standards = []
        for std_id, rules in self._rules.items():
            standards.append({
                'id': std_id,
                'name': std_id.upper(),
                'rule_count': len(rules)
            })
        return standards

    def get_compliance_report(
        self,
        report_id: str,
        tenant_id: str = None
    ) -> Optional[Dict]:
        """获取合规报告"""
        if self._report_repo:
            return self._report_repo.get_by_id(report_id, tenant_id)
        return None
    
    def list_compliance_reports(
        self,
        tenant_id: str,
        standard: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """列出合规报告"""
        if self._report_repo:
            return self._report_repo.list_by_tenant(tenant_id, standard, limit, offset)
        return []
    
    def get_latest_report(self, tenant_id: str, standard: str) -> Optional[Dict]:
        """获取最新合规报告"""
        if self._report_repo:
            return self._report_repo.get_latest_by_standard(tenant_id, standard)
        return None
    
    def _generate_recommendations(
        self,
        violations: List[Dict],
        standard: str
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        for violation in violations:
            rule_id = violation.get('rule_id', '')
            
            if 'consent' in rule_id:
                recommendations.append(
                    'Implement consent management system to collect and track user consent'
                )
            elif 'purpose' in rule_id:
                recommendations.append(
                    'Define and document specific purposes for data processing'
                )
            elif 'minimization' in rule_id:
                recommendations.append(
                    'Review data collection practices to ensure only necessary data is collected'
                )
            elif 'retention' in rule_id:
                recommendations.append(
                    'Establish data retention policies and implement automatic deletion'
                )
            elif 'security' in rule_id:
                recommendations.append(
                    'Implement encryption and access controls for personal data'
                )
            elif 'access' in rule_id:
                recommendations.append(
                    'Review and strengthen access control mechanisms'
                )
            elif 'audit' in rule_id:
                recommendations.append(
                    'Enable comprehensive audit logging for all security events'
                )
        
        # 去重
        return list(set(recommendations))


# ==================== 全局实例管理 ====================

_security_service: Optional[SecurityService] = None


def get_security_service(
    config: Dict[str, Any] = None,
    use_memory_storage: bool = False
) -> SecurityService:
    """获取安全服务实例"""
    global _security_service
    if _security_service is None:
        _security_service = SecurityService(config, use_memory_storage)
    return _security_service


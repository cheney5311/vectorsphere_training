# -*- coding: utf-8 -*-
"""
认证管理器服务
"""

import base64
import secrets
from datetime import datetime, timedelta
from io import BytesIO
from typing import Dict, Optional, List, Any

import jwt
import pyotp
import qrcode

from backend.modules.security.models import AuthContext, MFAConfig, AuthMethod


class AuthManager:
    """增强的认证管理器
    
    实现零信任安全架构，支持多因子认证、设备信任、行为分析等功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.jwt_secret = config.get('jwt_secret', secrets.token_urlsafe(32))
        self.jwt_algorithm = config.get('jwt_algorithm', 'HS256')
        self.session_timeout = config.get('session_timeout', 3600)  # 1小时
        self.max_concurrent_sessions = config.get('max_concurrent_sessions', 5)
        self.mfa_issuer = config.get('mfa_issuer', 'VectorSphere')
        
        # 活跃会话存储
        self.active_sessions: Dict[str, AuthContext] = {}
        # 设备信任存储
        self.trusted_devices: Dict[str, Dict] = {}
        # 用户MFA配置
        self.user_mfa_configs: Dict[str, MFAConfig] = {}
        
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str, user_agent: str,
                         device_fingerprint: str) -> Dict[str, Any]:
        """用户认证
        
        Args:
            username: 用户名
            password: 密码
            ip_address: IP地址
            user_agent: 用户代理
            device_fingerprint: 设备指纹
            
        Returns:
            认证结果
        """
        try:
            # 基础密码验证
            if not self._verify_password(username, password):
                return {
                    'success': False,
                    'error': 'Invalid credentials',
                    'requires_mfa': False
                }
            
            # 获取用户信息
            user_info = self._get_user_info(username)
            if not user_info:
                return {
                    'success': False,
                    'error': 'User not found',
                    'requires_mfa': False
                }
            
            # 检查用户状态
            if user_info.get('status') != 'active':
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
            mfa_config = self.user_mfa_configs.get(user_info['id'])
            requires_mfa = (
                mfa_config and mfa_config.enabled
            ) or risk_score > 0.7  # 高风险需要MFA
            
            if requires_mfa:
                # 生成临时认证令牌
                temp_token = self._generate_temp_token(user_info['id'])
                return {
                    'success': False,
                    'requires_mfa': True,
                    'temp_token': temp_token,
                    'mfa_methods': self._get_available_mfa_methods(user_info['id'])
                }
            
            # 创建会话
            session = self._create_session(
                user_info['id'], ip_address, user_agent, 
                device_fingerprint, risk_score, [AuthMethod.PASSWORD]
            )
            
            return {
                'success': True,
                'token': session['token'],
                'expires_at': session['expires_at'].isoformat(),
                'user_info': user_info
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Authentication failed: {str(e)}',
                'requires_mfa': False
            }
    
    def verify_mfa(self, temp_token: str, mfa_code: str, 
                   mfa_method: str) -> Dict[str, Any]:
        """验证多因子认证
        
        Args:
            temp_token: 临时认证令牌
            mfa_code: MFA验证码
            mfa_method: MFA方法
            
        Returns:
            验证结果
        """
        try:
            # 验证临时令牌
            temp_payload = jwt.decode(
                temp_token, self.jwt_secret, 
                algorithms=[self.jwt_algorithm]
            )
            
            user_id = temp_payload.get('user_id')
            if not user_id:
                return {'success': False, 'error': 'Invalid temp token'}
            
            # 验证MFA代码
            if mfa_method == 'totp':
                if not self._verify_totp(user_id, mfa_code):
                    return {'success': False, 'error': 'Invalid TOTP code'}
            elif mfa_method == 'backup':
                if not self._verify_backup_code(user_id, mfa_code):
                    return {'success': False, 'error': 'Invalid backup code'}
            else:
                return {'success': False, 'error': 'Unsupported MFA method'}
            
            # 获取认证上下文
            auth_context = temp_payload.get('auth_context', {})
            
            # 创建完整会话
            session = self._create_session(
                user_id,
                auth_context.get('ip_address', ''),
                auth_context.get('user_agent', ''),
                auth_context.get('device_fingerprint', ''),
                auth_context.get('risk_score', 0.0),
                [AuthMethod.PASSWORD, AuthMethod.MFA]
            )
            
            return {
                'success': True,
                'token': session['token'],
                'expires_at': session['expires_at'].isoformat()
            }
            
        except jwt.ExpiredSignatureError:
            return {'success': False, 'error': 'Temp token expired'}
        except jwt.InvalidTokenError:
            return {'success': False, 'error': 'Invalid temp token'}
        except Exception as e:
            return {'success': False, 'error': f'MFA verification failed: {str(e)}'}
    
    def validate_session(self, token: str) -> Optional[AuthContext]:
        """验证会话令牌
        
        Args:
            token: JWT令牌
            
        Returns:
            认证上下文或None
        """
        try:
            payload = jwt.decode(
                token, self.jwt_secret, 
                algorithms=[self.jwt_algorithm]
            )
            
            session_id = payload.get('session_id')
            if not session_id or session_id not in self.active_sessions:
                return None
            
            auth_context = self.active_sessions[session_id]
            
            # 检查会话是否过期
            if datetime.now() > auth_context.expires_at:
                self._revoke_session(session_id)
                return None
            
            # 更新最后活动时间
            auth_context.last_activity = datetime.now()
            
            return auth_context
            
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None
    
    def setup_mfa(self, user_id: str) -> Dict[str, Any]:
        """设置多因子认证
        
        Args:
            user_id: 用户ID
            
        Returns:
            MFA设置信息
        """
        try:
            # 生成密钥
            secret_key = pyotp.random_base32()
            
            # 生成备份码
            backup_codes = [secrets.token_hex(4).upper() for _ in range(10)]
            
            # 创建MFA配置
            mfa_config = MFAConfig(
                enabled=False,  # 需要用户确认后启用
                secret_key=secret_key,
                backup_codes=backup_codes,
                recovery_email='',
                recovery_phone=''
            )
            
            # 生成QR码
            user_info = self._get_user_info_by_id(user_id)
            totp_uri = pyotp.totp.TOTP(secret_key).provisioning_uri(
                name=user_info.get('email', user_id),
                issuer_name=self.mfa_issuer
            )
            
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(totp_uri)
            qr.make(fit=True)
            
            qr_img = qr.make_image(fill_color="black", back_color="white")
            qr_buffer = BytesIO()
            qr_img.save(qr_buffer, format='PNG')
            qr_code_base64 = base64.b64encode(qr_buffer.getvalue()).decode()
            
            # 临时存储配置（等待确认）
            self.user_mfa_configs[f"{user_id}_temp"] = mfa_config
            
            return {
                'success': True,
                'secret_key': secret_key,
                'qr_code': qr_code_base64,
                'backup_codes': backup_codes,
                'manual_entry_key': secret_key
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to setup MFA: {str(e)}'
            }
    
    def confirm_mfa_setup(self, user_id: str, verification_code: str) -> Dict[str, Any]:
        """确认MFA设置
        
        Args:
            user_id: 用户ID
            verification_code: 验证码
            
        Returns:
            确认结果
        """
        try:
            temp_config = self.user_mfa_configs.get(f"{user_id}_temp")
            if not temp_config:
                return {'success': False, 'error': 'No pending MFA setup'}
            
            # 验证代码
            totp = pyotp.TOTP(temp_config.secret_key)
            if not totp.verify(verification_code):
                return {'success': False, 'error': 'Invalid verification code'}
            
            # 启用MFA
            temp_config.enabled = True
            self.user_mfa_configs[user_id] = temp_config
            
            # 删除临时配置
            del self.user_mfa_configs[f"{user_id}_temp"]
            
            return {
                'success': True,
                'message': 'MFA enabled successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to confirm MFA: {str(e)}'
            }
    
    def revoke_session(self, session_id: str) -> bool:
        """撤销会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功
        """
        return self._revoke_session(session_id)
    
    def revoke_all_sessions(self, user_id: str) -> int:
        """撤销用户所有会话
        
        Args:
            user_id: 用户ID
            
        Returns:
            撤销的会话数量
        """
        revoked_count = 0
        sessions_to_revoke = []
        
        for session_id, auth_context in self.active_sessions.items():
            if auth_context.user_id == user_id:
                sessions_to_revoke.append(session_id)
        
        for session_id in sessions_to_revoke:
            if self._revoke_session(session_id):
                revoked_count += 1
        
        return revoked_count
    
    def get_active_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户活跃会话
        
        Args:
            user_id: 用户ID
            
        Returns:
            活跃会话列表
        """
        sessions = []
        for session_id, auth_context in self.active_sessions.items():
            if auth_context.user_id == user_id:
                sessions.append({
                    'session_id': session_id,
                    'ip_address': auth_context.ip_address,
                    'user_agent': auth_context.user_agent,
                    'created_at': auth_context.created_at.isoformat(),
                    'last_activity': auth_context.last_activity.isoformat(),
                    'expires_at': auth_context.expires_at.isoformat(),
                    'risk_score': auth_context.risk_score
                })
        return sessions
    
    def _verify_password(self, username: str, password: str) -> bool:
        """验证密码"""
        # 这里应该连接到实际的用户数据库
        # 暂时返回True用于演示
        return True
    
    def _get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """获取用户信息"""
        # 这里应该从数据库获取用户信息
        # 暂时返回模拟数据
        return {
            'id': f'user_{username}',
            'username': username,
            'email': f'{username}@example.com',
            'status': 'active',
            'role': 'user'
        }
    
    def _get_user_info_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取用户信息"""
        # 这里应该从数据库获取用户信息
        return {
            'id': user_id,
            'email': f'{user_id}@example.com'
        }
    
    def _calculate_risk_score(self, user_id: str, ip_address: str, 
                             user_agent: str, device_fingerprint: str) -> float:
        """计算风险评分"""
        risk_score = 0.0
        
        # 检查设备是否可信
        if device_fingerprint not in self.trusted_devices.get(user_id, {}):
            risk_score += 0.3
        
        # 检查IP地址变化
        # 这里可以添加更复杂的地理位置和历史IP检查
        risk_score += 0.1
        
        # 检查用户代理变化
        # 这里可以添加设备类型和浏览器检查
        risk_score += 0.1
        
        return min(risk_score, 1.0)
    
    def _get_available_mfa_methods(self, user_id: str) -> List[str]:
        """获取可用的MFA方法"""
        methods = []
        mfa_config = self.user_mfa_configs.get(user_id)
        
        if mfa_config and mfa_config.enabled:
            methods.append('totp')
            if mfa_config.backup_codes:
                methods.append('backup')
        
        return methods
    
    def _generate_temp_token(self, user_id: str) -> str:
        """生成临时认证令牌"""
        payload = {
            'user_id': user_id,
            'type': 'temp_auth',
            'exp': datetime.utcnow() + timedelta(minutes=5),  # 5分钟有效期
            'iat': datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def _create_session(self, user_id: str, ip_address: str, user_agent: str,
                       device_fingerprint: str, risk_score: float,
                       auth_methods: List[AuthMethod]) -> Dict[str, Any]:
        """创建会话"""
        session_id = secrets.token_urlsafe(32)
        now = datetime.now()
        expires_at = now + timedelta(seconds=self.session_timeout)
        
        # 创建认证上下文
        auth_context = AuthContext(
            user_id=user_id,
            session_id=session_id,
            auth_methods=auth_methods,
            ip_address=ip_address,
            user_agent=user_agent,
            device_fingerprint=device_fingerprint,
            risk_score=risk_score,
            created_at=now,
            expires_at=expires_at,
            last_activity=now
        )
        
        # 检查并发会话限制
        self._enforce_session_limit(user_id)
        
        # 存储会话
        self.active_sessions[session_id] = auth_context
        
        # 生成JWT令牌
        payload = {
            'user_id': user_id,
            'session_id': session_id,
            'auth_methods': [method.value for method in auth_methods],
            'exp': expires_at,
            'iat': now
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        return {
            'token': token,
            'session_id': session_id,
            'expires_at': expires_at
        }
    
    def _enforce_session_limit(self, user_id: str):
        """强制执行会话限制"""
        user_sessions = [
            (session_id, auth_context) 
            for session_id, auth_context in self.active_sessions.items()
            if auth_context.user_id == user_id
        ]
        
        if len(user_sessions) >= self.max_concurrent_sessions:
            # 删除最旧的会话
            oldest_session = min(user_sessions, key=lambda x: x[1].created_at)
            self._revoke_session(oldest_session[0])
    
    def _revoke_session(self, session_id: str) -> bool:
        """撤销会话"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            return True
        return False
    
    def _verify_totp(self, user_id: str, code: str) -> bool:
        """验证TOTP代码"""
        mfa_config = self.user_mfa_configs.get(user_id)
        if not mfa_config or not mfa_config.enabled:
            return False
        
        totp = pyotp.TOTP(mfa_config.secret_key)
        return totp.verify(code)
    
    def _verify_backup_code(self, user_id: str, code: str) -> bool:
        """验证备份代码"""
        mfa_config = self.user_mfa_configs.get(user_id)
        if not mfa_config or not mfa_config.enabled:
            return False
        
        if code.upper() in mfa_config.backup_codes:
            # 使用后删除备份码
            mfa_config.backup_codes.remove(code.upper())
            return True
        
        return False
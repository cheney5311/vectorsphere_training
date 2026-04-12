"""认证服务模块

提供生产级的认证服务，集成Agent长记忆推理能力:
- 用户认证: 注册、登录、登出、令牌管理
- 安全分析: 风险评估、异常检测、行为分析
- Agent推理: 智能风险检测、安全建议、模式学习
- 会话管理: 多设备会话、会话安全监控
- MFA支持: 双因素认证、设备管理
"""

import hashlib
import logging
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

from werkzeug.security import check_password_hash, generate_password_hash

from backend.core.exceptions import AuthenticationError, ValidationError
from backend.modules.auth.auth_exceptions import (
    UserNotFoundError, InvalidCredentialsError, UserAlreadyExistsError,
    InvalidTokenError, ExpiredTokenError
)
from backend.repositories.auth_repository import AuthRepository, get_auth_repository
from backend.schemas.auth_models import (
    User, SecurityEventType, RiskLevel, MemoryType
)

logger = logging.getLogger(__name__)


# ============================================================================
# DTOs 数据传输对象
# ============================================================================

@dataclass
class LoginRequest:
    """登录请求 DTO"""
    identifier: str
    password: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_fingerprint: Optional[str] = None
    location: Optional[str] = None
    mfa_code: Optional[str] = None


@dataclass
class LoginResult:
    """登录结果 DTO"""
    user: Dict[str, Any]
    tokens: Dict[str, Any]
    session_id: str
    requires_mfa: bool = False
    risk_assessment: Optional[Dict[str, Any]] = None


@dataclass
class RiskAssessmentResult:
    """风险评估结果 DTO"""
    risk_level: str
    risk_score: float
    risk_factors: List[str]
    recommendations: List[str]
    should_block: bool
    requires_mfa: bool
    reasoning_chain: List[str]
    confidence: float


@dataclass
class SecurityAnalysis:
    """安全分析结果 DTO"""
    trust_score: float
    risk_level: str
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]
    agent_insights: Dict[str, Any]


def get_auth_service() -> 'AuthService':
    """获取认证服务实例
    
    Returns:
        AuthService: 认证服务实例
    """
    repository = get_auth_repository()
    return AuthService(repository)


class AuthService:
    """认证服务类
    
    提供完整的认证功能，集成Agent智能推理。
    
    功能模块:
    - 用户管理: 注册、认证、资料管理
    - 会话管理: 创建、验证、撤销
    - 安全分析: 风险评估、异常检测
    - Agent推理: 智能风险检测、行为学习
    """
    
    # 配置常量
    TOKEN_EXPIRY_HOURS = 24
    REFRESH_TOKEN_EXPIRY_DAYS = 30
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_MINUTES = 30
    MIN_PASSWORD_LENGTH = 8
    HIGH_RISK_THRESHOLD = 0.7
    CRITICAL_RISK_THRESHOLD = 0.9
    
    def __init__(self, repository: AuthRepository):
        """初始化认证服务
        
        Args:
            repository: 认证仓库实例
        """
        self.repository = repository
        self.token_expiry = timedelta(hours=self.TOKEN_EXPIRY_HOURS)
        self.refresh_token_expiry = timedelta(days=self.REFRESH_TOKEN_EXPIRY_DAYS)
        
        # 懒加载的服务
        self._agent_service = None
        self._knowledge_engine = None
        self._inference_service = None
    
    @property
    def agent_service(self):
        """懒加载Agent服务"""
        if self._agent_service is None:
            try:
                from backend.services.agent_service import AgentService, get_agent_service
                self._agent_service = get_agent_service()
            except Exception as e:
                logger.warning(f"Failed to load agent service: {e}")
        return self._agent_service
    
    @property
    def knowledge_engine(self):
        """懒加载知识推理引擎"""
        if self._knowledge_engine is None:
            try:
                from backend.algo.knowledge_reasoning import KnowledgeReasoningEngine
                self._knowledge_engine = KnowledgeReasoningEngine()
            except Exception as e:
                logger.warning(f"Failed to load knowledge engine: {e}")
        return self._knowledge_engine
    
    @property
    def inference_service(self):
        """懒加载LLM推理服务"""
        if self._inference_service is None:
            try:
                from backend.services.langchain_inference_service import LangChainInferenceService
                self._inference_service = LangChainInferenceService()
            except Exception as e:
                logger.warning(f"Failed to load inference service: {e}")
        return self._inference_service
    
    # ============================================================================
    # 用户注册与认证
    # ============================================================================
    
    def register_user(self, 
                      username: str, 
                      email: str, 
                      password: str, 
                      full_name: str = "",
                      **kwargs) -> User:
        """用户注册
        
        注册新用户，包括密码验证、唯一性检查和安全初始化。
        
        Args:
            username: 用户名（3-50字符）
            email: 邮箱地址
            password: 密码（至少8字符，包含大小写字母和数字）
            full_name: 全名（可选）
            **kwargs: 其他用户属性
            
        Returns:
            User: 创建的用户实体
            
        Raises:
            ValidationError: 输入验证失败
            UserAlreadyExistsError: 用户名或邮箱已存在
        """
        # 验证密码强度
        self._validate_password(password)
        
        # 检查用户名是否已存在
        if self.repository.get_user_by_username(username):
            raise UserAlreadyExistsError("Username already exists")
        
        # 检查邮箱是否已存在
        if self.repository.get_user_by_email(email):
            raise UserAlreadyExistsError("Email already exists")
        
        # 创建用户
        user_data = {
            'username': username,
            'email': email.lower(),
            'password_hash': generate_password_hash(password),
            'full_name': full_name,
            'status': 'active',
            'trust_score': 0.5,  # 初始信任分数
            **kwargs
        }
        
        user = self.repository.create_user(user_data)
        
        # 创建安全画像
        self.repository.get_or_create_security_profile(str(user.id))
        
        # 记录安全事件
        self.repository.record_security_event({
            'user_id': str(user.id),
            'event_type': 'registration',
            'severity': 'low',
            'description': f'User {username} registered'
        })
        
        logger.info(f"User registered successfully: {username}")
        return user
    
    def authenticate_user(self, 
                          identifier: str, 
                          password: str,
                          ip_address: Optional[str] = None,
                          user_agent: Optional[str] = None,
                          device_fingerprint: Optional[str] = None,
                          location: Optional[str] = None,
                          tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """用户认证
        
        验证用户凭证并进行风险评估。
        
        Args:
            identifier: 用户名或邮箱
            password: 密码
            ip_address: IP地址（可选）
            user_agent: 用户代理（可选）
            device_fingerprint: 设备指纹（可选）
            location: 地理位置（可选）
            tenant_id: 租户ID（可选）
            
        Returns:
            Dict: 包含用户信息、令牌和会话ID的字典
            
        Raises:
            AuthenticationError: 认证失败
            InvalidCredentialsError: 凭证无效
            UserNotFoundError: 用户不存在
        """
        start_time = time.time()
        
        # 查找用户
        user = self.repository.get_user_by_identifier(identifier)
        
        # 进行风险评估
        risk_assessment = self._assess_login_risk(
            user=user,
            identifier=identifier,
            ip_address=ip_address,
            user_agent=user_agent,
            device_fingerprint=device_fingerprint,
            location=location
        )
        
        # 记录登录尝试的公共数据
        attempt_data = {
            'username': identifier,
            'ip_address': ip_address or 'unknown',
            'user_agent': user_agent,
            'device_fingerprint': device_fingerprint,
            'location': location,
            'risk_score': risk_assessment.risk_score,
            'risk_factors': risk_assessment.risk_factors
        }
        
        if not user:
            # 记录失败的登录尝试
            attempt_data['success'] = False
            attempt_data['failure_reason'] = 'user_not_found'
            self.repository.record_login_attempt(attempt_data)
            
            logger.warning(f"Login attempt with invalid username: {identifier}")
            raise InvalidCredentialsError("Invalid username or password")
        
        # 检查用户是否被锁定
        if user.is_locked():
            attempt_data['user_id'] = str(user.id)
            attempt_data['success'] = False
            attempt_data['failure_reason'] = 'account_locked'
            self.repository.record_login_attempt(attempt_data)
            
            raise AuthenticationError("Account is locked. Please try again later.")
        
        # 检查风险是否需要阻止
        if risk_assessment.should_block:
            attempt_data['user_id'] = str(user.id)
            attempt_data['success'] = False
            attempt_data['failure_reason'] = 'high_risk_blocked'
            self.repository.record_login_attempt(attempt_data)
            
            # 记录安全事件
            self.repository.record_security_event({
                'user_id': str(user.id),
                'event_type': SecurityEventType.SUSPICIOUS_ACTIVITY.value,
                'severity': 'high',
                'description': f'High risk login attempt blocked',
                'ip_address': ip_address,
                'metadata': {
                    'risk_score': risk_assessment.risk_score,
                    'risk_factors': risk_assessment.risk_factors
                }
            })
            
            raise AuthenticationError("Login blocked due to security concerns. Please contact support.")
        
        # 验证密码
        if not check_password_hash(str(user.password_hash), password):
            # 增加失败计数
            failed_count = self.repository.increment_failed_login(str(user.id))
            
            attempt_data['user_id'] = str(user.id)
            attempt_data['success'] = False
            attempt_data['failure_reason'] = 'invalid_password'
            self.repository.record_login_attempt(attempt_data)
            
            # 检查是否需要锁定账户
            if failed_count >= self.MAX_FAILED_ATTEMPTS:
                self.repository.lock_user(str(user.id), self.LOCKOUT_MINUTES)
                
                self.repository.record_security_event({
                    'user_id': str(user.id),
                    'event_type': SecurityEventType.ACCOUNT_LOCKED.value,
                    'severity': 'medium',
                    'description': f'Account locked after {failed_count} failed attempts',
                    'ip_address': ip_address
                })
            
            logger.warning(f"Invalid password for user: {identifier}")
            raise InvalidCredentialsError("Invalid username or password")
        
        # 检查用户状态
        if user.status != 'active':
            attempt_data['user_id'] = str(user.id)
            attempt_data['success'] = False
            attempt_data['failure_reason'] = 'account_inactive'
            self.repository.record_login_attempt(attempt_data)
            
            raise AuthenticationError("Account is not active")
        
        # 登录成功，重置失败计数
        self.repository.reset_failed_login(str(user.id))
        
        # 生成令牌
        access_token = self._generate_access_token(str(user.id), tenant_id)
        refresh_token = self._generate_refresh_token(str(user.id))
        
        # 创建会话
        session_data = {
            'user_id': str(user.id),
            'session_token': access_token,
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expires_at': datetime.utcnow() + self.token_expiry,
            'ip_address': ip_address,
            'user_agent': user_agent,
            'device_fingerprint': device_fingerprint,
            'location': location,
            'risk_score': risk_assessment.risk_score,
            'tenant_id': tenant_id
        }
        
        session = self.repository.create_session(session_data)
        
        # 记录成功的登录尝试
        attempt_data['user_id'] = str(user.id)
        attempt_data['success'] = True
        attempt_data['session_id'] = str(session.id)
        self.repository.record_login_attempt(attempt_data)
        
        # 更新最后登录时间
        self.repository.update_last_login(str(user.id))
        
        # 添加已知IP
        if ip_address:
            self.repository.add_known_ip(str(user.id), ip_address)
        
        # 更新用户行为模式
        self._update_login_patterns(str(user.id), ip_address, user_agent, device_fingerprint, location)
        
        # 记录安全事件
        self.repository.record_security_event({
            'user_id': str(user.id),
            'event_type': SecurityEventType.LOGIN_SUCCESS.value,
            'severity': 'low',
            'description': 'Successful login',
            'ip_address': ip_address,
            'metadata': {
                'risk_score': risk_assessment.risk_score,
                'latency_ms': (time.time() - start_time) * 1000
            }
        })
        
        # 记录风险评估
        self.repository.record_risk_assessment({
            'user_id': str(user.id),
            'session_id': str(session.id),
            'assessment_type': 'login',
            'risk_level': risk_assessment.risk_level,
            'risk_score': risk_assessment.risk_score,
            'risk_factors': risk_assessment.risk_factors,
            'recommendations': risk_assessment.recommendations,
            'action_taken': 'allowed',
            'reasoning_chain': risk_assessment.reasoning_chain,
            'confidence': risk_assessment.confidence
        })
        
        # 添加Agent记忆
        self._add_login_memory(str(user.id), risk_assessment, ip_address, location)
        
        logger.info(f"User {identifier} authenticated successfully")
        
        return {
            'user': {
                'id': str(user.id),
                'username': user.username,
                'email': user.email,
                'full_name': user.full_name,
                'role': user.role,
                'tenant_id': tenant_id
            },
            'tokens': {
                'access_token': access_token,
                'refresh_token': refresh_token,
                'expires_in': int(self.token_expiry.total_seconds())
            },
            'session_id': str(session.id),
            'risk_assessment': {
                'risk_level': risk_assessment.risk_level,
                'risk_score': risk_assessment.risk_score,
                'requires_mfa': risk_assessment.requires_mfa
            }
        }
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """验证访问令牌
        
        Args:
            token: 访问令牌
            
        Returns:
            Dict: 令牌载荷
            
        Raises:
            InvalidTokenError: 令牌无效
            ExpiredTokenError: 令牌已过期
        """
        session = self.repository.get_session_by_token(token)
        
        if not session:
            raise InvalidTokenError("Invalid token")
        
        if session.is_expired():
            raise ExpiredTokenError("Token has expired")
        
        if not session.is_active:
            raise InvalidTokenError("Token has been revoked")
        
        return {
            'user_id': str(session.user_id),
            'session_id': str(session.id),
            'tenant_id': session.tenant_id
        }
    
    def refresh_token(self, refresh_token_value: str) -> Tuple[str, str]:
        """刷新访问令牌
        
        Args:
            refresh_token_value: 刷新令牌
            
        Returns:
            Tuple[str, str]: 新的访问令牌和刷新令牌
            
        Raises:
            InvalidTokenError: 刷新令牌无效
            ExpiredTokenError: 刷新令牌已过期
        """
        session = self.repository.get_session_by_token(refresh_token_value)
        
        if not session:
            raise InvalidTokenError("Invalid refresh token")
        
        if not session.is_active:
            raise InvalidTokenError("Refresh token has been revoked")
        
        # 生成新令牌
        new_access_token = self._generate_access_token(str(session.user_id), session.tenant_id)
        new_refresh_token = self._generate_refresh_token(str(session.user_id))
        
        # 更新会话
        self.repository.create_session({
            'user_id': str(session.user_id),
            'session_token': new_access_token,
            'access_token': new_access_token,
            'refresh_token': new_refresh_token,
            'expires_at': datetime.utcnow() + self.token_expiry,
            'ip_address': session.ip_address,
            'user_agent': session.user_agent,
            'device_fingerprint': session.device_fingerprint,
            'location': session.location,
            'tenant_id': session.tenant_id
        })
        
        # 使旧会话失效
        self.repository.invalidate_session(str(session.id))
        
        logger.info(f"Token refreshed for user {session.user_id}")
        return new_access_token, new_refresh_token
    
    def revoke_token(self, token: str) -> bool:
        """撤销令牌
        
        Args:
            token: 访问令牌或刷新令牌
            
        Returns:
            bool: 是否成功撤销
        """
        session = self.repository.get_session_by_token(token)
        
        if session:
            self.repository.invalidate_session(str(session.id))
            
            self.repository.record_security_event({
                'user_id': str(session.user_id),
                'event_type': SecurityEventType.LOGOUT.value,
                'severity': 'low',
                'description': 'User logged out'
            })
            
            logger.info(f"Token revoked for user {session.user_id}")
            return True
        
        return False
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """根据ID获取用户
        
        Args:
            user_id: 用户ID
            
        Returns:
            User: 用户实体或None
        """
        return self.repository.get_user_by_id(user_id)
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict: 用户信息字典或None
        """
        user = self.repository.get_user_by_id(user_id)
        if user:
            return user.to_dict()
        return None
    
    def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Optional[User]:
        """更新用户信息
        
        Args:
            user_id: 用户ID
            update_data: 更新数据
            
        Returns:
            User: 更新后的用户实体或None
        """
        # 不允许直接更新敏感字段
        sensitive_fields = ['password_hash', 'trust_score', 'failed_login_count', 'lockout_until']
        for field in sensitive_fields:
            update_data.pop(field, None)
        
        return self.repository.update_user(user_id, update_data)
    
    def change_password(self, 
                        user_id: str, 
                        old_password: str, 
                        new_password: str) -> bool:
        """修改密码
        
        Args:
            user_id: 用户ID
            old_password: 旧密码
            new_password: 新密码
            
        Returns:
            bool: 是否成功
            
        Raises:
            ValidationError: 密码验证失败
            InvalidCredentialsError: 旧密码错误
        """
        user = self.repository.get_user_by_id(user_id)
        if not user:
            raise UserNotFoundError("User not found")
        
        # 验证旧密码
        if not check_password_hash(str(user.password_hash), old_password):
            raise InvalidCredentialsError("Invalid current password")
        
        # 验证新密码
        self._validate_password(new_password)
        
        # 更新密码
        self.repository.update_user(user_id, {
            'password_hash': generate_password_hash(new_password)
        })
        
        # 使所有会话失效
        self.repository.invalidate_user_sessions(user_id)
        
        # 记录安全事件
        self.repository.record_security_event({
            'user_id': user_id,
            'event_type': SecurityEventType.PASSWORD_CHANGE.value,
            'severity': 'medium',
            'description': 'Password changed by user'
        })
        
        logger.info(f"Password changed for user {user_id}")
        return True
    
    # ============================================================================
    # 会话管理
    # ============================================================================
    
    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的所有活跃会话
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[Dict]: 会话列表
        """
        sessions = self.repository.get_user_sessions(user_id, active_only=True)
        return [session.to_dict() for session in sessions]
    
    def validate_session(self, session_id: str, user_id: str) -> bool:
        """验证会话有效性
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            bool: 会话是否有效
        """
        sessions = self.repository.get_user_sessions(user_id, active_only=True)
        for session in sessions:
            if str(session.id) == session_id and not session.is_expired():
                return True
        return False
    
    def logout_user(self, user_id: str, session_id: Optional[str] = None) -> bool:
        """用户登出
        
        Args:
            user_id: 用户ID
            session_id: 会话ID（可选，不提供则登出所有会话）
            
        Returns:
            bool: 是否成功
        """
        if session_id:
            result = self.repository.invalidate_session(session_id)
        else:
            count = self.repository.invalidate_user_sessions(user_id)
            result = count > 0
        
        self.repository.record_security_event({
            'user_id': user_id,
            'event_type': SecurityEventType.LOGOUT.value,
            'severity': 'low',
            'description': 'User logged out'
        })
        
        return result
    
    def logout_all_devices(self, user_id: str, exclude_session_id: Optional[str] = None) -> int:
        """登出所有设备
        
        Args:
            user_id: 用户ID
            exclude_session_id: 排除的会话ID（可选）
            
        Returns:
            int: 登出的会话数量
        """
        count = self.repository.invalidate_user_sessions(user_id, exclude_session_id)
        
        self.repository.record_security_event({
            'user_id': user_id,
            'event_type': SecurityEventType.LOGOUT.value,
            'severity': 'low',
            'description': f'Logged out from all devices ({count} sessions)'
        })
        
        return count
    
    # ============================================================================
    # Agent 智能风险评估
    # ============================================================================
    
    def _assess_login_risk(self,
                           user: Optional[User],
                           identifier: str,
                           ip_address: Optional[str],
                           user_agent: Optional[str],
                           device_fingerprint: Optional[str],
                           location: Optional[str]) -> RiskAssessmentResult:
        """评估登录风险
        
        使用Agent智能推理进行风险评估。
        
        Args:
            user: 用户实体（可能为None）
            identifier: 登录标识
            ip_address: IP地址
            user_agent: 用户代理
            device_fingerprint: 设备指纹
            location: 地理位置
            
        Returns:
            RiskAssessmentResult: 风险评估结果
        """
        risk_factors = []
        reasoning_chain = []
        risk_score = 0.0
        
        # 1. 基础风险因素
        
        # IP地址检查
        if ip_address:
            # 检查IP失败次数
            ip_failed_count = self.repository.get_failed_login_count(ip_address=ip_address, minutes=30)
            if ip_failed_count > 3:
                risk_score += 0.2
                risk_factors.append(f"IP address has {ip_failed_count} failed attempts in last 30 minutes")
                reasoning_chain.append(f"检测到IP {ip_address} 在过去30分钟内有 {ip_failed_count} 次失败尝试")
            
            # 检查是否为已知IP
            if user:
                is_known = self.repository.is_known_ip(str(user.id), ip_address)
                if not is_known:
                    risk_score += 0.15
                    risk_factors.append("Unknown IP address")
                    reasoning_chain.append(f"IP地址 {ip_address} 不在用户的已知IP列表中")
                else:
                    reasoning_chain.append(f"IP地址 {ip_address} 是用户的已知IP")
        
        # 2. 用户相关风险
        if user:
            # 检查用户失败次数
            user_failed_count = self.repository.get_failed_login_count(user_id=str(user.id), minutes=30)
            if user_failed_count > 0:
                risk_score += min(0.3, user_failed_count * 0.1)
                risk_factors.append(f"User has {user_failed_count} recent failed attempts")
                reasoning_chain.append(f"用户在过去30分钟内有 {user_failed_count} 次失败尝试")
            
            # 检查用户信任分数
            if user.trust_score < 0.3:
                risk_score += 0.2
                risk_factors.append("Low trust score")
                reasoning_chain.append(f"用户信任分数较低: {user.trust_score}")
            
            # 获取用户行为模式
            patterns = self.repository.get_user_behavior_patterns(str(user.id))
            for pattern in patterns:
                if pattern.pattern_type == 'login_time':
                    # 检查登录时间是否异常
                    current_hour = datetime.utcnow().hour
                    typical_hours = pattern.pattern_data.get('typical_hours', [])
                    if typical_hours and current_hour not in typical_hours:
                        risk_score += 0.1
                        risk_factors.append("Unusual login time")
                        reasoning_chain.append(f"当前登录时间 {current_hour}:00 不在用户常用登录时间段内")
                
                elif pattern.pattern_type == 'device':
                    # 检查设备是否已知
                    if device_fingerprint:
                        known_devices = pattern.pattern_data.get('known_devices', [])
                        if known_devices and device_fingerprint not in known_devices:
                            risk_score += 0.15
                            risk_factors.append("Unknown device")
                            reasoning_chain.append("设备指纹不在用户的已知设备列表中")
                
                elif pattern.pattern_type == 'location':
                    # 检查位置是否异常
                    if location:
                        typical_locations = pattern.pattern_data.get('typical_locations', [])
                        if typical_locations and location not in typical_locations:
                            risk_score += 0.1
                            risk_factors.append("Unusual location")
                            reasoning_chain.append(f"位置 {location} 不在用户常用位置列表中")
        else:
            # 用户不存在的风险
            risk_score += 0.1
            reasoning_chain.append(f"用户名 {identifier} 不存在")
        
        # 3. 使用Agent进行智能分析
        agent_analysis = self._get_agent_risk_analysis(
            user=user,
            identifier=identifier,
            ip_address=ip_address,
            device_fingerprint=device_fingerprint,
            location=location,
            existing_factors=risk_factors
        )
        
        if agent_analysis:
            risk_score += agent_analysis.get('additional_risk', 0)
            risk_factors.extend(agent_analysis.get('factors', []))
            reasoning_chain.extend(agent_analysis.get('reasoning', []))
        
        # 4. 确定风险等级
        risk_score = min(1.0, risk_score)
        
        if risk_score >= self.CRITICAL_RISK_THRESHOLD:
            risk_level = RiskLevel.CRITICAL.value
        elif risk_score >= self.HIGH_RISK_THRESHOLD:
            risk_level = RiskLevel.HIGH.value
        elif risk_score >= 0.4:
            risk_level = RiskLevel.MEDIUM.value
        else:
            risk_level = RiskLevel.LOW.value
        
        # 5. 生成建议
        recommendations = self._generate_security_recommendations(
            risk_level=risk_level,
            risk_factors=risk_factors,
            user=user
        )
        
        # 6. 决定是否阻止或需要MFA
        should_block = risk_level == RiskLevel.CRITICAL.value
        requires_mfa = risk_level in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]
        
        if user and user.mfa_enabled:
            requires_mfa = True
        
        return RiskAssessmentResult(
            risk_level=risk_level,
            risk_score=risk_score,
            risk_factors=risk_factors,
            recommendations=recommendations,
            should_block=should_block,
            requires_mfa=requires_mfa,
            reasoning_chain=reasoning_chain,
            confidence=0.85 if agent_analysis else 0.7
        )
    
    def _get_agent_risk_analysis(self,
                                  user: Optional[User],
                                  identifier: str,
                                  ip_address: Optional[str],
                                  device_fingerprint: Optional[str],
                                  location: Optional[str],
                                  existing_factors: List[str]) -> Optional[Dict[str, Any]]:
        """使用Agent进行智能风险分析
        
        Args:
            user: 用户实体
            identifier: 登录标识
            ip_address: IP地址
            device_fingerprint: 设备指纹
            location: 地理位置
            existing_factors: 已有的风险因素
            
        Returns:
            Dict: Agent分析结果或None
        """
        try:
            if not self.knowledge_engine:
                return None
            
            # 获取相关记忆
            memories = []
            if user:
                memories = self.repository.get_auth_memories(
                    user_id=str(user.id),
                    memory_type=MemoryType.ANOMALY_PATTERN.value,
                    min_importance=0.3,
                    limit=5
                )
            
            # 构建查询上下文
            query_context = {
                'type': 'login_risk_analysis',
                'user_exists': user is not None,
                'ip_address': ip_address,
                'location': location,
                'device_fingerprint': device_fingerprint,
                'existing_risk_factors': existing_factors,
                'relevant_memories': [m.content for m in memories]
            }
            
            # 使用知识引擎进行分析
            result = self.knowledge_engine.query_knowledge(
                f"Analyze login risk for user with IP {ip_address} from location {location}"
            )
            
            additional_factors = []
            additional_reasoning = []
            additional_risk = 0.0
            
            # 从知识引擎结果提取洞察
            if result.get('recommendations'):
                additional_reasoning.extend(result['recommendations'])
            
            # 检查是否有匹配的异常模式
            for memory in memories:
                if 'suspicious' in memory.content.lower() or 'anomaly' in memory.content.lower():
                    additional_risk += 0.1 * memory.importance
                    additional_factors.append(f"Historical anomaly pattern: {memory.content[:100]}")
                    additional_reasoning.append(f"历史记忆显示类似行为模式曾被标记为异常")
            
            return {
                'additional_risk': additional_risk,
                'factors': additional_factors,
                'reasoning': additional_reasoning
            }
            
        except Exception as e:
            logger.warning(f"Agent risk analysis failed: {e}")
            return None
    
    def _generate_security_recommendations(self,
                                            risk_level: str,
                                            risk_factors: List[str],
                                            user: Optional[User]) -> List[str]:
        """生成安全建议
        
        Args:
            risk_level: 风险等级
            risk_factors: 风险因素
            user: 用户实体
            
        Returns:
            List[str]: 安全建议列表
        """
        recommendations = []
        
        if risk_level in [RiskLevel.HIGH.value, RiskLevel.CRITICAL.value]:
            recommendations.append("Consider enabling multi-factor authentication")
            recommendations.append("Review recent account activity")
        
        if "Unknown IP address" in risk_factors:
            recommendations.append("Verify this login if you don't recognize the location")
        
        if "Unknown device" in risk_factors:
            recommendations.append("Add this device to trusted devices if it's yours")
        
        if "Unusual login time" in risk_factors:
            recommendations.append("Update your usual login time preferences if needed")
        
        if user and not user.mfa_enabled:
            recommendations.append("Enable two-factor authentication for better security")
        
        if "Low trust score" in risk_factors:
            recommendations.append("Complete profile verification to improve trust score")
        
        return recommendations
    
    def _update_login_patterns(self,
                                user_id: str,
                                ip_address: Optional[str],
                                user_agent: Optional[str],
                                device_fingerprint: Optional[str],
                                location: Optional[str]) -> None:
        """更新用户登录模式
        
        Args:
            user_id: 用户ID
            ip_address: IP地址
            user_agent: 用户代理
            device_fingerprint: 设备指纹
            location: 地理位置
        """
        try:
            current_hour = datetime.utcnow().hour
            
            # 更新登录时间模式
            time_pattern = self.repository.get_or_create_behavior_pattern(user_id, 'login_time')
            time_data = time_pattern.pattern_data or {'typical_hours': [], 'hour_counts': {}}
            
            hour_counts = time_data.get('hour_counts', {})
            hour_counts[str(current_hour)] = hour_counts.get(str(current_hour), 0) + 1
            
            # 计算最常用的登录时间
            sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
            typical_hours = [int(h) for h, _ in sorted_hours[:6]]
            
            time_data['typical_hours'] = typical_hours
            time_data['hour_counts'] = hour_counts
            
            self.repository.update_behavior_pattern(user_id, 'login_time', time_data)
            
            # 更新设备模式
            if device_fingerprint:
                device_pattern = self.repository.get_or_create_behavior_pattern(user_id, 'device')
                device_data = device_pattern.pattern_data or {'known_devices': []}
                
                known_devices = device_data.get('known_devices', [])
                if device_fingerprint not in known_devices:
                    known_devices.append(device_fingerprint)
                    # 保留最近的10个设备
                    device_data['known_devices'] = known_devices[-10:]
                    self.repository.update_behavior_pattern(user_id, 'device', device_data)
            
            # 更新位置模式
            if location:
                location_pattern = self.repository.get_or_create_behavior_pattern(user_id, 'location')
                location_data = location_pattern.pattern_data or {'typical_locations': [], 'location_counts': {}}
                
                location_counts = location_data.get('location_counts', {})
                location_counts[location] = location_counts.get(location, 0) + 1
                
                sorted_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
                typical_locations = [loc for loc, _ in sorted_locations[:5]]
                
                location_data['typical_locations'] = typical_locations
                location_data['location_counts'] = location_counts
                
                self.repository.update_behavior_pattern(user_id, 'location', location_data)
                
        except Exception as e:
            logger.warning(f"Failed to update login patterns: {e}")
    
    def _add_login_memory(self,
                          user_id: str,
                          risk_assessment: RiskAssessmentResult,
                          ip_address: Optional[str],
                          location: Optional[str]) -> None:
        """添加登录相关的Agent记忆
        
        Args:
            user_id: 用户ID
            risk_assessment: 风险评估结果
            ip_address: IP地址
            location: 地理位置
        """
        try:
            # 如果风险较高，记录为异常模式
            if risk_assessment.risk_score >= 0.5:
                content = f"Login with elevated risk ({risk_assessment.risk_level}): "
                content += f"Risk factors: {', '.join(risk_assessment.risk_factors[:3])}"
                if location:
                    content += f" from {location}"
                
                self.repository.add_auth_memory({
                    'user_id': user_id,
                    'memory_type': MemoryType.ANOMALY_PATTERN.value,
                    'content': content,
                    'importance': min(1.0, risk_assessment.risk_score),
                    'metadata': {
                        'ip_address': ip_address,
                        'location': location,
                        'risk_score': risk_assessment.risk_score,
                        'risk_factors': risk_assessment.risk_factors
                    },
                    'source': 'login_risk_assessment'
                })
            
            # 记录成功登录的信任指标
            if risk_assessment.risk_level == RiskLevel.LOW.value:
                self.repository.add_auth_memory({
                    'user_id': user_id,
                    'memory_type': MemoryType.TRUST_INDICATOR.value,
                    'content': f"Normal login from {location or 'unknown location'} at {datetime.utcnow().strftime('%H:%M')}",
                    'importance': 0.3,
                    'metadata': {
                        'ip_address': ip_address,
                        'location': location
                    },
                    'source': 'login_success',
                    'expires_at': datetime.utcnow() + timedelta(days=90)
                })
                
        except Exception as e:
            logger.warning(f"Failed to add login memory: {e}")
    
    # ============================================================================
    # 安全分析
    # ============================================================================
    
    def analyze_user_security(self, user_id: str) -> SecurityAnalysis:
        """分析用户安全状况
        
        使用Agent进行综合安全分析。
        
        Args:
            user_id: 用户ID
            
        Returns:
            SecurityAnalysis: 安全分析结果
        """
        user = self.repository.get_user_by_id(user_id)
        if not user:
            raise UserNotFoundError("User not found")
        
        # 获取用户安全画像
        profile = self.repository.get_or_create_security_profile(user_id)
        
        # 获取最近的登录统计
        login_stats = self.repository.get_login_statistics(user_id=user_id, days=30)
        
        # 获取最近的安全事件
        security_events, _ = self.repository.get_security_events(
            user_id=user_id,
            days=30,
            limit=20
        )
        
        # 获取风险评估历史
        risk_assessments = self.repository.get_user_risk_assessments(user_id, days=30)
        avg_risk_score = self.repository.get_average_risk_score(user_id)
        
        # 检测异常
        anomalies = []
        
        # 检查失败登录率
        if login_stats['total_attempts'] > 0:
            failure_rate = login_stats['failed'] / login_stats['total_attempts']
            if failure_rate > 0.3:
                anomalies.append({
                    'type': 'high_failure_rate',
                    'severity': 'medium',
                    'description': f'High login failure rate: {failure_rate:.1%}',
                    'metric': failure_rate
                })
        
        # 检查高风险事件
        high_risk_events = [e for e in security_events if e.severity in ['high', 'critical']]
        if high_risk_events:
            anomalies.append({
                'type': 'security_events',
                'severity': 'high',
                'description': f'{len(high_risk_events)} high-risk security events in last 30 days',
                'events': [e.to_dict() for e in high_risk_events[:5]]
            })
        
        # 检查未解决的安全事件
        unresolved_events = [e for e in security_events if not e.is_resolved]
        if unresolved_events:
            anomalies.append({
                'type': 'unresolved_events',
                'severity': 'medium',
                'description': f'{len(unresolved_events)} unresolved security events',
                'count': len(unresolved_events)
            })
        
        # 计算综合信任分数
        trust_score = self._calculate_trust_score(
            user=user,
            profile=profile,
            login_stats=login_stats,
            avg_risk_score=avg_risk_score,
            anomalies=anomalies
        )
        
        # 更新用户信任分数
        self.repository.update_trust_score(user_id, trust_score)
        
        # 确定风险等级
        if trust_score >= 0.7:
            risk_level = RiskLevel.LOW.value
        elif trust_score >= 0.4:
            risk_level = RiskLevel.MEDIUM.value
        elif trust_score >= 0.2:
            risk_level = RiskLevel.HIGH.value
        else:
            risk_level = RiskLevel.CRITICAL.value
        
        # 生成建议
        recommendations = []
        
        if not user.mfa_enabled:
            recommendations.append("Enable multi-factor authentication for enhanced security")
        
        if anomalies:
            recommendations.append("Review and address detected security anomalies")
        
        if avg_risk_score > 0.5:
            recommendations.append("Consider reviewing recent account activity")
        
        if profile.trust_score < 0.5:
            recommendations.append("Complete profile verification to improve trust score")
        
        # Agent洞察
        agent_insights = self._get_agent_security_insights(
            user=user,
            profile=profile,
            anomalies=anomalies,
            login_stats=login_stats
        )
        
        # 更新安全画像
        self.repository.update_security_profile(user_id, {
            'trust_score': trust_score,
            'risk_level': risk_level,
            'last_security_review': datetime.utcnow(),
            'agent_insights': agent_insights
        })
        
        return SecurityAnalysis(
            trust_score=trust_score,
            risk_level=risk_level,
            anomalies=anomalies,
            recommendations=recommendations,
            agent_insights=agent_insights
        )
    
    def _calculate_trust_score(self,
                                user: User,
                                profile: Any,
                                login_stats: Dict[str, Any],
                                avg_risk_score: float,
                                anomalies: List[Dict[str, Any]]) -> float:
        """计算用户信任分数
        
        Args:
            user: 用户实体
            profile: 安全画像
            login_stats: 登录统计
            avg_risk_score: 平均风险分数
            anomalies: 异常列表
            
        Returns:
            float: 信任分数 (0.0-1.0)
        """
        score = 0.5  # 基础分数
        
        # 账户年龄加分
        if user.created_at:
            account_age_days = (datetime.utcnow() - user.created_at).days
            if account_age_days > 365:
                score += 0.1
            elif account_age_days > 90:
                score += 0.05
        
        # MFA启用加分
        if user.mfa_enabled:
            score += 0.15
        
        # 登录成功率加分
        if login_stats['total_attempts'] > 0:
            success_rate = login_stats['successful'] / login_stats['total_attempts']
            score += 0.1 * success_rate
        
        # 平均风险分数扣分
        score -= 0.2 * avg_risk_score
        
        # 异常扣分
        for anomaly in anomalies:
            if anomaly['severity'] == 'critical':
                score -= 0.2
            elif anomaly['severity'] == 'high':
                score -= 0.1
            elif anomaly['severity'] == 'medium':
                score -= 0.05
        
        # 确保分数在有效范围内
        return max(0.0, min(1.0, score))
    
    def _get_agent_security_insights(self,
                                      user: User,
                                      profile: Any,
                                      anomalies: List[Dict[str, Any]],
                                      login_stats: Dict[str, Any]) -> Dict[str, Any]:
        """使用Agent获取安全洞察
        
        Args:
            user: 用户实体
            profile: 安全画像
            anomalies: 异常列表
            login_stats: 登录统计
            
        Returns:
            Dict: Agent洞察
        """
        try:
            # 获取相关记忆
            memories = self.repository.get_auth_memories(
                user_id=str(user.id),
                min_importance=0.5,
                limit=10
            )
            
            insights = {
                'generated_at': datetime.utcnow().isoformat(),
                'key_observations': [],
                'risk_assessment': None,
                'behavioral_patterns': [],
                'recommendations_priority': []
            }
            
            # 分析记忆中的模式
            anomaly_memories = [m for m in memories if m.memory_type == MemoryType.ANOMALY_PATTERN.value]
            trust_memories = [m for m in memories if m.memory_type == MemoryType.TRUST_INDICATOR.value]
            
            if len(anomaly_memories) > len(trust_memories):
                insights['key_observations'].append("User shows higher frequency of anomalous behavior patterns")
            else:
                insights['key_observations'].append("User maintains consistent normal behavior patterns")
            
            # 分析登录模式
            if login_stats['top_ips']:
                insights['behavioral_patterns'].append({
                    'type': 'ip_diversity',
                    'observation': f"User logs in from {len(login_stats['top_ips'])} different IP addresses"
                })
            
            # 根据异常生成优先建议
            if anomalies:
                critical_anomalies = [a for a in anomalies if a['severity'] == 'critical']
                high_anomalies = [a for a in anomalies if a['severity'] == 'high']
                
                if critical_anomalies:
                    insights['recommendations_priority'].append("Immediate attention required for critical security issues")
                if high_anomalies:
                    insights['recommendations_priority'].append("Review high-severity security events")
            
            return insights
            
        except Exception as e:
            logger.warning(f"Failed to get agent security insights: {e}")
            return {}
    
    def get_security_events(self,
                            user_id: Optional[str] = None,
                            event_type: Optional[str] = None,
                            severity: Optional[str] = None,
                            days: int = 30,
                            offset: int = 0,
                            limit: int = 100) -> Tuple[List[Dict[str, Any]], int]:
        """获取安全事件
        
        Args:
            user_id: 用户ID（可选）
            event_type: 事件类型（可选）
            severity: 严重程度（可选）
            days: 时间范围（天）
            offset: 偏移量
            limit: 限制数量
            
        Returns:
            Tuple[List[Dict], int]: 事件列表和总数
        """
        events, total = self.repository.get_security_events(
            user_id=user_id,
            event_type=event_type,
            severity=severity,
            days=days,
            offset=offset,
            limit=limit
        )
        
        return [e.to_dict() for e in events], total
    
    def get_login_statistics(self, 
                              user_id: Optional[str] = None,
                              days: int = 30) -> Dict[str, Any]:
        """获取登录统计
        
        Args:
            user_id: 用户ID（可选）
            days: 统计天数
            
        Returns:
            Dict: 统计数据
        """
        return self.repository.get_login_statistics(user_id=user_id, days=days)
    
    def get_security_statistics(self, days: int = 30) -> Dict[str, Any]:
        """获取安全统计
        
        Args:
            days: 统计天数
            
        Returns:
            Dict: 统计数据
        """
        return self.repository.get_security_statistics(days=days)
    
    # ============================================================================
    # Agent 记忆管理
    # ============================================================================
    
    def add_security_memory(self,
                            user_id: Optional[str],
                            memory_type: str,
                            content: str,
                            importance: float = 0.5,
                            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """添加安全相关的Agent记忆
        
        Args:
            user_id: 用户ID（可选，全局记忆为None）
            memory_type: 记忆类型
            content: 记忆内容
            importance: 重要性 (0.0-1.0)
            metadata: 元数据
            
        Returns:
            Dict: 记忆记录
        """
        memory = self.repository.add_auth_memory({
            'user_id': user_id,
            'memory_type': memory_type,
            'content': content,
            'importance': importance,
            'metadata': metadata,
            'source': 'manual'
        })
        
        return memory.to_dict()
    
    def search_security_memories(self,
                                  query: str,
                                  user_id: Optional[str] = None,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """搜索安全相关的Agent记忆
        
        Args:
            query: 搜索查询
            user_id: 用户ID（可选）
            limit: 限制数量
            
        Returns:
            List[Dict]: 匹配的记忆列表
        """
        memories = self.repository.search_auth_memories(query, user_id, limit)
        return [m.to_dict() for m in memories]
    
    def get_user_memories(self,
                          user_id: str,
                          memory_type: Optional[str] = None,
                          limit: int = 50) -> List[Dict[str, Any]]:
        """获取用户的Agent记忆
        
        Args:
            user_id: 用户ID
            memory_type: 记忆类型（可选）
            limit: 限制数量
            
        Returns:
            List[Dict]: 记忆列表
        """
        memories = self.repository.get_auth_memories(
            user_id=user_id,
            memory_type=memory_type,
            limit=limit
        )
        return [m.to_dict() for m in memories]
    
    # ============================================================================
    # 辅助方法
    # ============================================================================
    
    def _generate_access_token(self, user_id: str, tenant_id: Optional[str] = None) -> str:
        """生成访问令牌
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            
        Returns:
            str: 访问令牌
        """
        token_data = f"{user_id}:{tenant_id}:{datetime.utcnow().timestamp()}:{secrets.token_hex(16)}"
        return hashlib.sha256(token_data.encode()).hexdigest()
    
    def _generate_refresh_token(self, user_id: str) -> str:
        """生成刷新令牌
        
        Args:
            user_id: 用户ID
            
        Returns:
            str: 刷新令牌
        """
        token_data = f"{user_id}:{datetime.utcnow().timestamp()}:{secrets.token_hex(32)}"
        return hashlib.sha256(token_data.encode()).hexdigest()
    
    def _validate_password(self, password: str) -> None:
        """验证密码强度
        
        Args:
            password: 密码
            
        Raises:
            ValidationError: 密码不符合要求
        """
        if not password or len(password) < self.MIN_PASSWORD_LENGTH:
            raise ValidationError(f"Password must be at least {self.MIN_PASSWORD_LENGTH} characters long")
        
        if not any(c.isupper() for c in password):
            raise ValidationError("Password must contain at least one uppercase letter")
        
        if not any(c.islower() for c in password):
            raise ValidationError("Password must contain at least one lowercase letter")
        
        if not any(c.isdigit() for c in password):
            raise ValidationError("Password must contain at least one digit")


# ============================================================================
# 全局实例
# ============================================================================

_global_auth_service = None


def get_global_auth_service() -> AuthService:
    """获取全局认证服务实例"""
    global _global_auth_service
    if _global_auth_service is None:
        _global_auth_service = get_auth_service()
    return _global_auth_service

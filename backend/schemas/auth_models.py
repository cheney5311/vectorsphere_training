"""认证相关数据模型

定义用户、角色、权限等认证相关数据模型。
支持生产级 agent 长记忆推理的安全认证系统。

功能模块:
- 用户管理: 用户、角色、权限
- 会话管理: 用户会话、令牌
- 安全分析: 登录尝试、安全事件、风险评估
- 行为分析: 用户行为模式、安全画像
- Agent 记忆: 认证相关的智能记忆和推理
- 双因素认证: 2FA 设备和验证码
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum

from backend.schemas.base_models import Base, UUIDMixin, TimestampMixin, GUID


# ============================================================================
# 枚举类型定义
# ============================================================================

class UserStatus(str, Enum):
    """用户状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    LOCKED = "locked"
    PENDING = "pending"


class SecurityEventType(str, Enum):
    """安全事件类型枚举"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    PASSWORD_RESET = "password_reset"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    SESSION_HIJACK_ATTEMPT = "session_hijack_attempt"
    BRUTE_FORCE_DETECTED = "brute_force_detected"
    LOCATION_ANOMALY = "location_anomaly"
    DEVICE_ANOMALY = "device_anomaly"
    API_ABUSE = "api_abuse"


class RiskLevel(str, Enum):
    """风险等级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryType(str, Enum):
    """认证记忆类型枚举"""
    USER_PATTERN = "user_pattern"
    DEVICE_FINGERPRINT = "device_fingerprint"
    LOCATION_HISTORY = "location_history"
    SECURITY_RULE = "security_rule"
    ANOMALY_PATTERN = "anomaly_pattern"
    TRUST_INDICATOR = "trust_indicator"


# ============================================================================
# 核心用户模型
# ============================================================================

class User(Base, UUIDMixin, TimestampMixin):
    """用户模型
    
    存储用户基本信息和认证相关属性。
    支持多租户、角色权限和安全配置。
    
    Attributes:
        username: 唯一用户名
        email: 唯一电子邮件
        password_hash: 加密后的密码哈希
        first_name: 名
        last_name: 姓
        full_name: 全名
        phone: 手机号码
        role: 用户角色
        status: 用户状态 (active, inactive, suspended, locked, pending)
        is_active: 是否激活
        is_superuser: 是否超级用户
        last_login: 最后登录时间
        avatar_url: 头像URL
        mfa_enabled: 是否启用双因素认证
        mfa_secret: 双因素认证密钥
        failed_login_count: 连续登录失败次数
        lockout_until: 账户锁定截止时间
        trust_score: 用户信任分数 (0.0-1.0)
        security_level: 安全等级
        preferences: 用户偏好设置 (JSON)
    """
    __tablename__ = 'users'
    
    username = Column(String(80), unique=True, nullable=False, index=True, comment="用户名")
    email = Column(String(120), unique=True, nullable=False, index=True, comment="邮箱")
    password_hash = Column(String(255), nullable=False, comment="密码哈希")
    first_name = Column(String(50), comment="名")
    last_name = Column(String(50), comment="姓")
    full_name = Column(String(100), comment="全名")
    phone = Column(String(20), comment="手机号码")
    role = Column(String(50), default='user', comment="用户角色")
    status = Column(String(20), default='active', comment="用户状态")
    is_active = Column(Boolean, default=True, comment="是否激活")
    is_superuser = Column(Boolean, default=False, comment="是否超级用户")
    last_login = Column(DateTime, comment="最后登录时间")
    avatar_url = Column(Text, comment="头像URL")
    
    # 安全相关字段
    mfa_enabled = Column(Boolean, default=False, comment="是否启用双因素认证")
    mfa_secret = Column(String(255), comment="双因素认证密钥")
    failed_login_count = Column(Integer, default=0, comment="连续登录失败次数")
    lockout_until = Column(DateTime, comment="账户锁定截止时间")
    trust_score = Column(Float, default=0.5, comment="用户信任分数")
    security_level = Column(String(20), default='standard', comment="安全等级")
    preferences = Column(JSON, comment="用户偏好设置")
    
    # 关系
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    roles = relationship("UserRole", back_populates="user", cascade="all, delete-orphan")
    login_attempts = relationship("LoginAttempt", back_populates="user", cascade="all, delete-orphan")
    security_events = relationship("SecurityEvent", back_populates="user", cascade="all, delete-orphan")
    behavior_patterns = relationship("UserBehaviorPattern", back_populates="user", cascade="all, delete-orphan")
    security_profile = relationship("UserSecurityProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    mfa_devices = relationship("MFADevice", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id='{self.id}', username='{self.username}', email='{self.email}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "full_name": self.full_name,
            "phone": self.phone,
            "role": self.role,
            "status": self.status,
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
            "mfa_enabled": self.mfa_enabled,
            "trust_score": self.trust_score,
            "security_level": self.security_level,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def is_locked(self) -> bool:
        """检查账户是否被锁定"""
        if self.lockout_until and self.lockout_until > datetime.utcnow():
            return True
        return self.status == UserStatus.LOCKED.value


class UserSession(Base, UUIDMixin, TimestampMixin):
    """用户会话模型
    
    管理用户的认证会话，支持多设备登录和会话追踪。
    
    Attributes:
        user_id: 关联的用户ID
        session_token: 会话令牌
        access_token: 访问令牌
        refresh_token: 刷新令牌
        expires_at: 过期时间
        ip_address: 客户端IP地址
        user_agent: 客户端用户代理
        device_fingerprint: 设备指纹
        location: 地理位置
        is_active: 是否激活
        status: 会话状态
        risk_score: 会话风险分数
        tenant_id: 租户ID
    """
    __tablename__ = 'user_sessions'
    
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False, index=True, comment="用户ID")
    session_token = Column(String(255), unique=True, nullable=False, index=True, comment="会话令牌")
    access_token = Column(String(512), comment="访问令牌")
    refresh_token = Column(String(512), comment="刷新令牌")
    expires_at = Column(DateTime, nullable=False, comment="过期时间")
    ip_address = Column(String(45), comment="IP地址")
    user_agent = Column(Text, comment="用户代理")
    device_fingerprint = Column(String(255), comment="设备指纹")
    location = Column(String(255), comment="地理位置")
    is_active = Column(Boolean, default=True, comment="是否激活")
    status = Column(String(20), default='active', comment="会话状态")
    risk_score = Column(Float, default=0.0, comment="会话风险分数")
    tenant_id = Column(String(36), comment="租户ID")
    
    # 关系
    user = relationship("User", back_populates="sessions")
    
    # 索引
    __table_args__ = (
        Index('ix_user_sessions_user_expires', 'user_id', 'expires_at'),
        Index('ix_user_sessions_token_active', 'session_token', 'is_active'),
    )
    
    def __repr__(self):
        return f"<UserSession(id='{self.id}', user_id='{self.user_id}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "ip_address": self.ip_address,
            "device_fingerprint": self.device_fingerprint,
            "location": self.location,
            "status": self.status,
            "risk_score": self.risk_score,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
    
    def is_expired(self) -> bool:
        """检查会话是否已过期"""
        return datetime.utcnow() > self.expires_at if self.expires_at else True


class ApiKey(Base, UUIDMixin, TimestampMixin):
    """API密钥模型
    
    管理用户的API访问密钥。
    
    Attributes:
        user_id: 关联的用户ID
        name: 密钥名称
        key_hash: 密钥哈希
        key_prefix: 密钥前缀（用于显示）
        expires_at: 过期时间
        is_active: 是否激活
        permissions: 权限列表
        rate_limit: 速率限制
        last_used_at: 最后使用时间
        usage_count: 使用次数
    """
    __tablename__ = 'api_keys'
    
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False, index=True, comment="用户ID")
    name = Column(String(100), nullable=False, comment="密钥名称")
    key_hash = Column(String(255), nullable=False, comment="密钥哈希")
    key_prefix = Column(String(10), comment="密钥前缀")
    expires_at = Column(DateTime, comment="过期时间")
    is_active = Column(Boolean, default=True, comment="是否激活")
    permissions = Column(JSON, comment="权限列表")
    rate_limit = Column(Integer, default=1000, comment="每小时速率限制")
    last_used_at = Column(DateTime, comment="最后使用时间")
    usage_count = Column(Integer, default=0, comment="使用次数")
    
    # 关系
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<ApiKey(id='{self.id}', user_id='{self.user_id}', name='{self.name}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "name": self.name,
            "key_prefix": self.key_prefix,
            "is_active": self.is_active,
            "permissions": self.permissions,
            "rate_limit": self.rate_limit,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "usage_count": self.usage_count,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# ============================================================================
# 安全与审计模型
# ============================================================================

class LoginAttempt(Base, UUIDMixin, TimestampMixin):
    """登录尝试记录模型
    
    记录所有登录尝试，用于安全分析和异常检测。
    
    Attributes:
        user_id: 尝试登录的用户ID（可能为空，如用户不存在）
        username: 尝试的用户名
        ip_address: 客户端IP地址
        user_agent: 客户端用户代理
        device_fingerprint: 设备指纹
        location: 地理位置
        success: 是否成功
        failure_reason: 失败原因
        risk_score: 风险分数
        risk_factors: 风险因素列表
        session_id: 成功登录后的会话ID
    """
    __tablename__ = 'login_attempts'
    
    user_id = Column(GUID(), ForeignKey('users.id'), index=True, comment="用户ID")
    username = Column(String(80), nullable=False, index=True, comment="尝试的用户名")
    ip_address = Column(String(45), nullable=False, index=True, comment="IP地址")
    user_agent = Column(Text, comment="用户代理")
    device_fingerprint = Column(String(255), comment="设备指纹")
    location = Column(String(255), comment="地理位置")
    country_code = Column(String(3), comment="国家代码")
    city = Column(String(100), comment="城市")
    success = Column(Boolean, nullable=False, default=False, comment="是否成功")
    failure_reason = Column(String(255), comment="失败原因")
    risk_score = Column(Float, default=0.0, comment="风险分数")
    risk_factors = Column(JSON, comment="风险因素")
    session_id = Column(GUID(), comment="创建的会话ID")
    
    # 关系
    user = relationship("User", back_populates="login_attempts")
    
    # 索引
    __table_args__ = (
        Index('ix_login_attempts_ip_time', 'ip_address', 'created_at'),
        Index('ix_login_attempts_user_time', 'user_id', 'created_at'),
        Index('ix_login_attempts_success_time', 'success', 'created_at'),
    )
    
    def __repr__(self):
        return f"<LoginAttempt(id='{self.id}', username='{self.username}', success={self.success})>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "username": self.username,
            "ip_address": self.ip_address,
            "location": self.location,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "risk_score": self.risk_score,
            "risk_factors": self.risk_factors,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class SecurityEvent(Base, UUIDMixin, TimestampMixin):
    """安全事件模型
    
    记录所有安全相关事件，供Agent分析和学习。
    
    Attributes:
        user_id: 关联的用户ID
        event_type: 事件类型
        severity: 严重程度 (low, medium, high, critical)
        description: 事件描述
        ip_address: IP地址
        user_agent: 用户代理
        device_fingerprint: 设备指纹
        location: 地理位置
        metadata: 事件元数据
        is_resolved: 是否已解决
        resolved_at: 解决时间
        resolved_by: 解决人
        resolution_notes: 解决说明
        agent_analysis: Agent分析结果
        agent_recommendation: Agent建议
    """
    __tablename__ = 'security_events'
    
    user_id = Column(GUID(), ForeignKey('users.id'), index=True, comment="用户ID")
    event_type = Column(String(50), nullable=False, index=True, comment="事件类型")
    severity = Column(String(20), nullable=False, default='medium', index=True, comment="严重程度")
    description = Column(Text, comment="事件描述")
    ip_address = Column(String(45), comment="IP地址")
    user_agent = Column(Text, comment="用户代理")
    device_fingerprint = Column(String(255), comment="设备指纹")
    location = Column(String(255), comment="地理位置")
    event_metadata = Column(JSON, comment="事件元数据")
    is_resolved = Column(Boolean, default=False, comment="是否已解决")
    resolved_at = Column(DateTime, comment="解决时间")
    resolved_by = Column(String(36), comment="解决人ID")
    resolution_notes = Column(Text, comment="解决说明")
    agent_analysis = Column(JSON, comment="Agent分析结果")
    agent_recommendation = Column(Text, comment="Agent建议")
    
    # 关系
    user = relationship("User", back_populates="security_events")
    
    # 索引
    __table_args__ = (
        Index('ix_security_events_type_time', 'event_type', 'created_at'),
        Index('ix_security_events_severity_resolved', 'severity', 'is_resolved'),
        Index('ix_security_events_user_type', 'user_id', 'event_type'),
    )
    
    def __repr__(self):
        return f"<SecurityEvent(id='{self.id}', type='{self.event_type}', severity='{self.severity}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "event_type": self.event_type,
            "severity": self.severity,
            "description": self.description,
            "ip_address": self.ip_address,
            "location": self.location,
            "metadata": self.event_metadata,
            "is_resolved": self.is_resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "agent_analysis": self.agent_analysis,
            "agent_recommendation": self.agent_recommendation,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class RiskAssessment(Base, UUIDMixin, TimestampMixin):
    """风险评估记录模型
    
    记录Agent对登录和操作的风险评估。
    
    Attributes:
        user_id: 关联的用户ID
        session_id: 关联的会话ID
        assessment_type: 评估类型 (login, operation, session)
        risk_level: 风险等级
        risk_score: 风险分数 (0.0-1.0)
        risk_factors: 风险因素列表
        recommendations: Agent建议
        action_taken: 采取的行动
        context_data: 上下文数据
        model_version: 使用的模型版本
        reasoning_chain: 推理链
        confidence: 置信度
    """
    __tablename__ = 'risk_assessments'
    
    user_id = Column(GUID(), ForeignKey('users.id'), index=True, comment="用户ID")
    session_id = Column(GUID(), comment="会话ID")
    assessment_type = Column(String(50), nullable=False, comment="评估类型")
    risk_level = Column(String(20), nullable=False, comment="风险等级")
    risk_score = Column(Float, nullable=False, comment="风险分数")
    risk_factors = Column(JSON, comment="风险因素")
    recommendations = Column(JSON, comment="Agent建议")
    action_taken = Column(String(100), comment="采取的行动")
    context_data = Column(JSON, comment="上下文数据")
    model_version = Column(String(50), comment="模型版本")
    reasoning_chain = Column(JSON, comment="推理链")
    confidence = Column(Float, default=0.0, comment="置信度")
    
    # 索引
    __table_args__ = (
        Index('ix_risk_assessments_user_time', 'user_id', 'created_at'),
        Index('ix_risk_assessments_level_time', 'risk_level', 'created_at'),
    )
    
    def __repr__(self):
        return f"<RiskAssessment(id='{self.id}', level='{self.risk_level}', score={self.risk_score})>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "assessment_type": self.assessment_type,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "risk_factors": self.risk_factors,
            "recommendations": self.recommendations,
            "action_taken": self.action_taken,
            "reasoning_chain": self.reasoning_chain,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# ============================================================================
# 用户行为与画像模型
# ============================================================================

class UserBehaviorPattern(Base, UUIDMixin, TimestampMixin):
    """用户行为模式模型
    
    记录和学习用户的行为模式，用于异常检测。
    
    Attributes:
        user_id: 关联的用户ID
        pattern_type: 模式类型 (login_time, location, device, activity)
        pattern_data: 模式数据
        confidence: 置信度
        sample_count: 样本数量
        last_observed: 最后观察时间
        is_active: 是否有效
        anomaly_threshold: 异常阈值
    """
    __tablename__ = 'user_behavior_patterns'
    
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False, index=True, comment="用户ID")
    pattern_type = Column(String(50), nullable=False, index=True, comment="模式类型")
    pattern_data = Column(JSON, nullable=False, comment="模式数据")
    confidence = Column(Float, default=0.5, comment="置信度")
    sample_count = Column(Integer, default=0, comment="样本数量")
    last_observed = Column(DateTime, comment="最后观察时间")
    is_active = Column(Boolean, default=True, comment="是否有效")
    anomaly_threshold = Column(Float, default=0.7, comment="异常阈值")
    
    # 关系
    user = relationship("User", back_populates="behavior_patterns")
    
    # 索引
    __table_args__ = (
        Index('ix_behavior_patterns_user_type', 'user_id', 'pattern_type'),
    )
    
    def __repr__(self):
        return f"<UserBehaviorPattern(id='{self.id}', user_id='{self.user_id}', type='{self.pattern_type}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "pattern_type": self.pattern_type,
            "pattern_data": self.pattern_data,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "last_observed": self.last_observed.isoformat() if self.last_observed else None,
            "is_active": self.is_active,
            "anomaly_threshold": self.anomaly_threshold
        }


class UserSecurityProfile(Base, UUIDMixin, TimestampMixin):
    """用户安全画像模型
    
    存储用户的综合安全画像，由Agent维护和更新。
    
    Attributes:
        user_id: 关联的用户ID
        trust_score: 信任分数 (0.0-1.0)
        risk_level: 风险等级
        typical_locations: 常用地理位置
        typical_devices: 常用设备
        typical_login_hours: 常用登录时间段
        known_ips: 已知IP地址
        security_questions_set: 是否设置安全问题
        mfa_methods: 启用的MFA方法
        last_security_review: 最后安全审查时间
        flags: 安全标记
        notes: 备注
        agent_insights: Agent洞察
    """
    __tablename__ = 'user_security_profiles'
    
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False, unique=True, comment="用户ID")
    trust_score = Column(Float, default=0.5, comment="信任分数")
    risk_level = Column(String(20), default='medium', comment="风险等级")
    typical_locations = Column(JSON, comment="常用地理位置")
    typical_devices = Column(JSON, comment="常用设备")
    typical_login_hours = Column(JSON, comment="常用登录时间段")
    known_ips = Column(JSON, comment="已知IP地址")
    security_questions_set = Column(Boolean, default=False, comment="是否设置安全问题")
    mfa_methods = Column(JSON, comment="启用的MFA方法")
    last_security_review = Column(DateTime, comment="最后安全审查时间")
    flags = Column(JSON, comment="安全标记")
    notes = Column(Text, comment="备注")
    agent_insights = Column(JSON, comment="Agent洞察")
    
    # 关系
    user = relationship("User", back_populates="security_profile")
    
    def __repr__(self):
        return f"<UserSecurityProfile(user_id='{self.user_id}', trust_score={self.trust_score})>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "trust_score": self.trust_score,
            "risk_level": self.risk_level,
            "typical_locations": self.typical_locations,
            "typical_devices": self.typical_devices,
            "typical_login_hours": self.typical_login_hours,
            "known_ips": self.known_ips,
            "mfa_methods": self.mfa_methods,
            "last_security_review": self.last_security_review.isoformat() if self.last_security_review else None,
            "flags": self.flags,
            "agent_insights": self.agent_insights
        }


# ============================================================================
# Agent 记忆与推理模型
# ============================================================================

class AuthAgentMemory(Base, UUIDMixin, TimestampMixin):
    """认证Agent记忆模型
    
    存储认证相关的Agent长期记忆。
    
    Attributes:
        user_id: 关联的用户ID（可选，全局记忆为空）
        memory_type: 记忆类型
        content: 记忆内容
        embedding: 向量嵌入
        importance: 重要性分数
        access_count: 访问次数
        last_accessed: 最后访问时间
        expires_at: 过期时间
        metadata: 元数据
        source: 记忆来源
        is_active: 是否有效
    """
    __tablename__ = 'auth_agent_memories'
    
    user_id = Column(GUID(), ForeignKey('users.id'), index=True, comment="用户ID")
    memory_type = Column(String(50), nullable=False, index=True, comment="记忆类型")
    content = Column(Text, nullable=False, comment="记忆内容")
    embedding = Column(JSON, comment="向量嵌入")
    importance = Column(Float, default=0.5, comment="重要性分数")
    access_count = Column(Integer, default=0, comment="访问次数")
    last_accessed = Column(DateTime, comment="最后访问时间")
    expires_at = Column(DateTime, comment="过期时间")
    auth_memory_metadata = Column(JSON, comment="元数据")
    source = Column(String(100), comment="记忆来源")
    is_active = Column(Boolean, default=True, comment="是否有效")
    
    # 索引
    __table_args__ = (
        Index('ix_auth_memories_user_type', 'user_id', 'memory_type'),
        Index('ix_auth_memories_importance', 'importance'),
        Index('ix_auth_memories_active_type', 'is_active', 'memory_type'),
    )
    
    def __repr__(self):
        return f"<AuthAgentMemory(id='{self.id}', type='{self.memory_type}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "memory_type": self.memory_type,
            "content": self.content,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.auth_memory_metadata,
            "source": self.source,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class AuthAgentReasoning(Base, UUIDMixin, TimestampMixin):
    """认证Agent推理记录模型
    
    记录Agent的推理过程和决策。
    
    Attributes:
        user_id: 关联的用户ID
        session_id: 关联的会话ID
        trigger: 触发推理的事件
        context: 推理上下文
        reasoning_steps: 推理步骤
        conclusion: 结论
        confidence: 置信度
        action_suggested: 建议的行动
        action_taken: 实际采取的行动
        outcome: 结果
        feedback: 反馈
        model_used: 使用的模型
        tokens_used: 使用的token数
        latency_ms: 延迟毫秒
    """
    __tablename__ = 'auth_agent_reasoning'
    
    user_id = Column(GUID(), ForeignKey('users.id'), index=True, comment="用户ID")
    session_id = Column(GUID(), comment="会话ID")
    trigger = Column(String(100), nullable=False, comment="触发事件")
    context = Column(JSON, comment="推理上下文")
    reasoning_steps = Column(JSON, comment="推理步骤")
    conclusion = Column(Text, comment="结论")
    confidence = Column(Float, default=0.0, comment="置信度")
    action_suggested = Column(String(100), comment="建议的行动")
    action_taken = Column(String(100), comment="实际采取的行动")
    outcome = Column(String(50), comment="结果")
    feedback = Column(JSON, comment="反馈")
    model_used = Column(String(100), comment="使用的模型")
    tokens_used = Column(Integer, default=0, comment="使用的token数")
    latency_ms = Column(Float, default=0, comment="延迟毫秒")
    
    # 索引
    __table_args__ = (
        Index('ix_auth_reasoning_user_time', 'user_id', 'created_at'),
        Index('ix_auth_reasoning_trigger', 'trigger'),
    )
    
    def __repr__(self):
        return f"<AuthAgentReasoning(id='{self.id}', trigger='{self.trigger}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "trigger": self.trigger,
            "reasoning_steps": self.reasoning_steps,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "action_suggested": self.action_suggested,
            "action_taken": self.action_taken,
            "outcome": self.outcome,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# ============================================================================
# 双因素认证模型
# ============================================================================

class MFADevice(Base, UUIDMixin, TimestampMixin):
    """多因素认证设备模型
    
    管理用户的MFA设备。
    
    Attributes:
        user_id: 关联的用户ID
        device_type: 设备类型 (totp, sms, email, webauthn, backup_code)
        device_name: 设备名称
        secret: 设备密钥
        is_primary: 是否为主设备
        is_verified: 是否已验证
        verified_at: 验证时间
        last_used_at: 最后使用时间
        is_active: 是否有效
        metadata: 元数据
    """
    __tablename__ = 'mfa_devices'
    
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False, index=True, comment="用户ID")
    device_type = Column(String(50), nullable=False, comment="设备类型")
    device_name = Column(String(100), comment="设备名称")
    secret = Column(String(255), comment="设备密钥")
    is_primary = Column(Boolean, default=False, comment="是否为主设备")
    is_verified = Column(Boolean, default=False, comment="是否已验证")
    verified_at = Column(DateTime, comment="验证时间")
    last_used_at = Column(DateTime, comment="最后使用时间")
    is_active = Column(Boolean, default=True, comment="是否有效")
    device_metadata = Column(JSON, comment="元数据")
    
    # 关系
    user = relationship("User", back_populates="mfa_devices")
    
    # 索引
    __table_args__ = (
        Index('ix_mfa_devices_user_type', 'user_id', 'device_type'),
    )
    
    def __repr__(self):
        return f"<MFADevice(id='{self.id}', user_id='{self.user_id}', type='{self.device_type}')>"
    
    def to_dict(self) -> dict:
        """转换为字典（不包含密钥）"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "device_type": self.device_type,
            "device_name": self.device_name,
            "is_primary": self.is_primary,
            "is_verified": self.is_verified,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class MFAVerification(Base, UUIDMixin, TimestampMixin):
    """MFA验证记录模型
    
    记录MFA验证尝试。
    
    Attributes:
        user_id: 关联的用户ID
        device_id: 关联的设备ID
        verification_code: 验证码（哈希后存储）
        is_successful: 是否成功
        failure_reason: 失败原因
        ip_address: IP地址
        user_agent: 用户代理
        expires_at: 过期时间
    """
    __tablename__ = 'mfa_verifications'
    
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False, index=True, comment="用户ID")
    device_id = Column(GUID(), ForeignKey('mfa_devices.id'), comment="设备ID")
    verification_code = Column(String(255), comment="验证码哈希")
    is_successful = Column(Boolean, default=False, comment="是否成功")
    failure_reason = Column(String(255), comment="失败原因")
    ip_address = Column(String(45), comment="IP地址")
    user_agent = Column(Text, comment="用户代理")
    expires_at = Column(DateTime, comment="过期时间")
    
    # 索引
    __table_args__ = (
        Index('ix_mfa_verifications_user_time', 'user_id', 'created_at'),
    )
    
    def __repr__(self):
        return f"<MFAVerification(id='{self.id}', user_id='{self.user_id}', successful={self.is_successful})>"


# ============================================================================
# 角色与权限模型
# ============================================================================

# 注意: Role 和 UserRole 模型已移至 backend/schemas/permission_models.py
# 为了避免重复定义，这里不再重复定义
# 如需使用，请从 permission_models 导入
# from backend.schemas.permission_models import Role, UserRole
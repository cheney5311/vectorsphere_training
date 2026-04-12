# -*- coding: utf-8 -*-
"""安全相关数据模型

定义安全模块的 SQLAlchemy 数据模型，包括：
- 用户会话
- 审计日志
- 访问控制
- 加密密钥
- 合规记录
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, Index, JSON
from datetime import datetime
import uuid
from enum import Enum

from backend.schemas.base_models import Base, UUIDMixin, TimestampMixin, TenantMixin, GUID


# ==================== 枚举定义 ====================

class AuthMethodEnum(str, Enum):
    """认证方式枚举"""
    PASSWORD = "password"
    MFA = "mfa"
    BIOMETRIC = "biometric"
    CERTIFICATE = "certificate"
    SSO = "sso"
    API_KEY = "api_key"
    OAUTH = "oauth"


class SessionStatusEnum(str, Enum):
    """会话状态枚举"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


class AuditEventTypeEnum(str, Enum):
    """审计事件类型"""
    # 认证事件
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    PASSWORD_CHANGE = "auth.password.change"
    MFA_ENABLE = "auth.mfa.enable"
    MFA_DISABLE = "auth.mfa.disable"
    MFA_VERIFY = "auth.mfa.verify"
    
    # 访问控制事件
    ACCESS_GRANTED = "access.granted"
    ACCESS_DENIED = "access.denied"
    PERMISSION_CHANGE = "access.permission.change"
    ROLE_ASSIGN = "access.role.assign"
    ROLE_REVOKE = "access.role.revoke"
    
    # 数据操作事件
    DATA_CREATE = "data.create"
    DATA_READ = "data.read"
    DATA_UPDATE = "data.update"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"
    DATA_ENCRYPT = "data.encrypt"
    DATA_DECRYPT = "data.decrypt"
    
    # 安全事件
    SECURITY_VIOLATION = "security.violation"
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    KEY_GENERATE = "security.key.generate"
    KEY_ROTATE = "security.key.rotate"
    COMPLIANCE_CHECK = "security.compliance.check"


class AuditLevelEnum(str, Enum):
    """审计级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RoleEnum(str, Enum):
    """角色枚举"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MANAGER = "manager"
    DEVELOPER = "developer"
    ANALYST = "analyst"
    VIEWER = "viewer"
    GUEST = "guest"


class EncryptionAlgorithmEnum(str, Enum):
    """加密算法枚举"""
    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    FERNET = "fernet"
    RSA_2048 = "rsa-2048"
    RSA_4096 = "rsa-4096"


class KeyTypeEnum(str, Enum):
    """密钥类型枚举"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC_PUBLIC = "asymmetric_public"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    DERIVED = "derived"


class KeyStatusEnum(str, Enum):
    """密钥状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING_ROTATION = "pending_rotation"


class ComplianceStandardEnum(str, Enum):
    """合规标准枚举"""
    GDPR = "gdpr"
    CCPA = "ccpa"
    SOC2 = "soc2"
    ISO27001 = "iso27001"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"


class ComplianceLevelEnum(str, Enum):
    """合规级别"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNKNOWN = "unknown"


class DataCategoryEnum(str, Enum):
    """数据类别"""
    PERSONAL_DATA = "personal_data"
    SENSITIVE_DATA = "sensitive_data"
    FINANCIAL_DATA = "financial_data"
    HEALTH_DATA = "health_data"
    BIOMETRIC_DATA = "biometric_data"
    LOCATION_DATA = "location_data"


# ==================== 数据模型 ====================

class SecurityUserSession(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """安全模块用户会话模型"""
    __tablename__ = 'security_user_sessions'
    
    # 会话标识
    session_token = Column(String(500), unique=True, nullable=False, index=True, comment="会话令牌")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 认证信息
    auth_method = Column(String(30), default=AuthMethodEnum.PASSWORD.value, comment="认证方式")
    mfa_verified = Column(Boolean, default=False, comment="MFA是否已验证")
    
    # 状态
    status = Column(String(20), default=SessionStatusEnum.ACTIVE.value, index=True, comment="会话状态")
    
    # 设备信息
    ip_address = Column(String(50), comment="IP地址")
    user_agent = Column(Text, comment="用户代理")
    device_fingerprint = Column(String(200), comment="设备指纹")
    device_type = Column(String(50), comment="设备类型")
    
    # 安全信息
    risk_score = Column(Float, default=0.0, comment="风险评分")
    is_trusted_device = Column(Boolean, default=False, comment="是否信任设备")
    
    # 时间信息
    expires_at = Column(DateTime, nullable=False, comment="过期时间")
    last_activity_at = Column(DateTime, comment="最后活动时间")
    
    # 元数据
    extra_metadata = Column('metadata', JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<UserSession(id='{self.id}', user_id='{self.user_id}', status='{self.status}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'session_token': self.session_token[:20] + '...' if self.session_token else None,
            'user_id': self.user_id,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'auth_method': self.auth_method,
            'mfa_verified': self.mfa_verified,
            'status': self.status,
            'ip_address': self.ip_address,
            'device_type': self.device_type,
            'risk_score': self.risk_score,
            'is_trusted_device': self.is_trusted_device,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_activity_at': self.last_activity_at.isoformat() if self.last_activity_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class SecurityAuditLog(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """安全审计日志模型"""
    __tablename__ = 'security_audit_logs'
    
    # 事件信息
    event_type = Column(String(50), nullable=False, index=True, comment="事件类型")
    event_level = Column(String(20), default=AuditLevelEnum.INFO.value, index=True, comment="事件级别")
    
    # 操作者信息
    user_id = Column(String(36), index=True, comment="用户ID")
    session_id = Column(String(36), index=True, comment="会话ID")
    
    # 来源信息
    source_ip = Column(String(50), comment="源IP地址")
    user_agent = Column(Text, comment="用户代理")
    
    # 操作信息
    resource = Column(String(200), comment="操作资源")
    action = Column(String(100), comment="操作动作")
    result = Column(String(20), default='success', comment="操作结果")
    
    # 内容
    message = Column(Text, nullable=False, comment="日志消息")
    details = Column(JSON, comment="详细信息")
    
    # 安全信息
    risk_score = Column(Float, default=0.0, comment="风险评分")
    
    # 标签
    tags = Column(JSON, comment="标签列表")
    
    def __repr__(self):
        return f"<SecurityAuditLog(id='{self.id}', event_type='{self.event_type}', user_id='{self.user_id}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'event_type': self.event_type,
            'event_level': self.event_level,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'source_ip': self.source_ip,
            'resource': self.resource,
            'action': self.action,
            'result': self.result,
            'message': self.message,
            'details': self.details,
            'risk_score': self.risk_score,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class UserRole(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """用户角色模型"""
    __tablename__ = 'user_roles'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    role = Column(String(50), nullable=False, index=True, comment="角色")
    
    # 分配信息
    assigned_by = Column(String(36), comment="分配者ID")
    assigned_at = Column(DateTime, default=datetime.utcnow, comment="分配时间")
    
    # 有效期
    expires_at = Column(DateTime, comment="过期时间")
    
    # 状态
    is_active = Column(Boolean, default=True, comment="是否激活")
    
    # 元数据
    extra_metadata = Column('metadata', JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<UserRole(user_id='{self.user_id}', role='{self.role}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'role': self.role,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'assigned_by': self.assigned_by,
            'assigned_at': self.assigned_at.isoformat() if self.assigned_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class AccessPolicy(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """访问策略模型"""
    __tablename__ = 'access_policies'
    
    # 策略标识
    policy_id = Column(String(64), unique=True, nullable=False, index=True, comment="策略ID")
    name = Column(String(200), nullable=False, comment="策略名称")
    description = Column(Text, comment="策略描述")
    
    # 策略内容
    effect = Column(String(10), default='allow', comment="效果(allow/deny)")
    principals = Column(JSON, comment="主体列表")
    resources = Column(JSON, comment="资源模式列表")
    actions = Column(JSON, comment="操作列表")
    conditions = Column(JSON, comment="条件")
    
    # 优先级
    priority = Column(Integer, default=100, comment="优先级")
    
    # 状态
    is_active = Column(Boolean, default=True, comment="是否激活")
    
    def __repr__(self):
        return f"<AccessPolicy(policy_id='{self.policy_id}', name='{self.name}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'policy_id': self.policy_id,
            'name': self.name,
            'description': self.description,
            'effect': self.effect,
            'principals': self.principals,
            'resources': self.resources,
            'actions': self.actions,
            'conditions': self.conditions,
            'priority': self.priority,
            'is_active': self.is_active,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class EncryptionKey(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """加密密钥模型"""
    __tablename__ = 'encryption_keys'
    
    # 密钥标识
    key_id = Column(String(64), unique=True, nullable=False, index=True, comment="密钥ID")
    name = Column(String(200), nullable=False, comment="密钥名称")
    description = Column(Text, comment="密钥描述")
    
    # 密钥信息
    key_type = Column(String(30), nullable=False, comment="密钥类型")
    algorithm = Column(String(30), nullable=False, comment="加密算法")
    key_size = Column(Integer, comment="密钥长度(bits)")
    
    # 密钥数据（加密存储）
    encrypted_key_data = Column(Text, comment="加密的密钥数据")
    key_checksum = Column(String(64), comment="密钥校验和")
    
    # 状态
    status = Column(String(20), default=KeyStatusEnum.ACTIVE.value, index=True, comment="密钥状态")
    
    # 时间
    expires_at = Column(DateTime, comment="过期时间")
    last_used_at = Column(DateTime, comment="最后使用时间")
    rotated_from = Column(String(64), comment="轮换来源密钥ID")
    
    # 使用统计
    usage_count = Column(Integer, default=0, comment="使用次数")
    
    # 元数据
    extra_metadata = Column('metadata', JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<EncryptionKey(key_id='{self.key_id}', name='{self.name}', status='{self.status}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'key_id': self.key_id,
            'name': self.name,
            'description': self.description,
            'key_type': self.key_type,
            'algorithm': self.algorithm,
            'key_size': self.key_size,
            'status': self.status,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'usage_count': self.usage_count,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class DataProcessingRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """数据处理记录模型（用于合规追踪）"""
    __tablename__ = 'data_processing_records'
    
    # 记录标识
    record_id = Column(String(64), unique=True, nullable=False, index=True, comment="记录ID")
    
    # 数据主体
    data_subject_id = Column(String(100), nullable=False, index=True, comment="数据主体ID")
    
    # 数据类别和处理目的
    data_categories = Column(JSON, comment="数据类别列表")
    processing_purposes = Column(JSON, comment="处理目的列表")
    
    # 法律依据
    legal_basis = Column(String(100), comment="法律依据")
    
    # 同意信息
    consent_given = Column(Boolean, default=False, comment="是否同意")
    consent_timestamp = Column(DateTime, comment="同意时间")
    consent_scope = Column(Text, comment="同意范围")
    
    # 保留期限
    retention_period_days = Column(Integer, comment="保留期限(天)")
    scheduled_deletion_at = Column(DateTime, comment="计划删除时间")
    
    # 处理位置
    processing_location = Column(String(100), comment="处理位置")
    
    # 第三方共享
    third_party_sharing = Column(Boolean, default=False, comment="是否第三方共享")
    third_parties = Column(JSON, comment="第三方列表")
    
    # 状态
    status = Column(String(20), default='active', comment="记录状态")
    
    # 元数据
    extra_metadata = Column('metadata', JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<DataProcessingRecord(record_id='{self.record_id}', data_subject_id='{self.data_subject_id}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'record_id': self.record_id,
            'data_subject_id': self.data_subject_id,
            'data_categories': self.data_categories,
            'processing_purposes': self.processing_purposes,
            'legal_basis': self.legal_basis,
            'consent_given': self.consent_given,
            'consent_timestamp': self.consent_timestamp.isoformat() if self.consent_timestamp else None,
            'retention_period_days': self.retention_period_days,
            'processing_location': self.processing_location,
            'third_party_sharing': self.third_party_sharing,
            'third_parties': self.third_parties,
            'status': self.status,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class ComplianceReport(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """合规报告模型"""
    __tablename__ = 'compliance_reports'
    
    # 报告标识
    report_id = Column(String(64), unique=True, nullable=False, index=True, comment="报告ID")
    
    # 合规标准
    standard = Column(String(30), nullable=False, index=True, comment="合规标准")
    
    # 评估结果
    overall_level = Column(String(30), nullable=False, comment="整体合规级别")
    score = Column(Float, comment="合规评分(0-100)")
    
    # 规则统计
    total_rules = Column(Integer, default=0, comment="总规则数")
    compliant_rules = Column(Integer, default=0, comment="合规规则数")
    non_compliant_rules = Column(Integer, default=0, comment="不合规规则数")
    
    # 违规列表
    violations = Column(JSON, comment="违规列表")
    
    # 建议
    recommendations = Column(JSON, comment="改进建议")
    
    # 评估时间范围
    period_start = Column(DateTime, comment="评估开始时间")
    period_end = Column(DateTime, comment="评估结束时间")
    
    # 元数据
    extra_metadata = Column('metadata', JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<ComplianceReport(report_id='{self.report_id}', standard='{self.standard}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'report_id': self.report_id,
            'standard': self.standard,
            'overall_level': self.overall_level,
            'score': self.score,
            'total_rules': self.total_rules,
            'compliant_rules': self.compliant_rules,
            'non_compliant_rules': self.non_compliant_rules,
            'violations': self.violations,
            'recommendations': self.recommendations,
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class MFAConfig(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """MFA配置模型"""
    __tablename__ = 'mfa_configs'
    
    user_id = Column(String(36), nullable=False, unique=True, index=True, comment="用户ID")
    
    # MFA状态
    is_enabled = Column(Boolean, default=False, comment="是否启用")
    mfa_type = Column(String(30), default='totp', comment="MFA类型(totp/sms/email)")
    
    # TOTP配置（加密存储）
    encrypted_secret = Column(Text, comment="加密的TOTP密钥")
    backup_codes = Column(JSON, comment="备用码列表")
    
    # 恢复选项
    recovery_email = Column(String(200), comment="恢复邮箱")
    recovery_phone = Column(String(50), comment="恢复电话")
    
    # 验证信息
    verified_at = Column(DateTime, comment="验证时间")
    last_used_at = Column(DateTime, comment="最后使用时间")
    
    def __repr__(self):
        return f"<MFAConfig(user_id='{self.user_id}', is_enabled={self.is_enabled})>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'is_enabled': self.is_enabled,
            'mfa_type': self.mfa_type,
            'has_backup_codes': bool(self.backup_codes),
            'recovery_email': self.recovery_email[:3] + '***' if self.recovery_email else None,
            'verified_at': self.verified_at.isoformat() if self.verified_at else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


# ==================== 索引定义 ====================

Index('idx_session_user_tenant', UserSession.user_id, UserSession.tenant_id)
Index('idx_session_status_expires', UserSession.status, UserSession.expires_at)
Index('idx_audit_tenant_time', SecurityAuditLog.tenant_id, SecurityAuditLog.created_at)
Index('idx_audit_user_event', SecurityAuditLog.user_id, SecurityAuditLog.event_type)
Index('idx_role_user_tenant', UserRole.user_id, UserRole.tenant_id)
Index('idx_key_tenant_status', EncryptionKey.tenant_id, EncryptionKey.status)
Index('idx_processing_subject', DataProcessingRecord.data_subject_id, DataProcessingRecord.tenant_id)
Index('idx_compliance_tenant_standard', ComplianceReport.tenant_id, ComplianceReport.standard)


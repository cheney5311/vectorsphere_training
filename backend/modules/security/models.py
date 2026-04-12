# -*- coding: utf-8 -*-
"""
Security模块数据模型
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class AuthMethod(Enum):
    """认证方式枚举"""
    PASSWORD = "password"
    MFA = "mfa"
    BIOMETRIC = "biometric"
    CERTIFICATE = "certificate"
    SSO = "sso"


class SessionStatus(Enum):
    """会话状态枚举"""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


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


class AuditEventType(Enum):
    """审计事件类型"""
    # 认证事件
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    PASSWORD_CHANGE = "auth.password.change"
    MFA_ENABLE = "auth.mfa.enable"
    MFA_DISABLE = "auth.mfa.disable"
    
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
    DATA_IMPORT = "data.import"
    
    # 训练操作事件
    TRAINING_START = "training.start"
    TRAINING_STOP = "training.stop"
    TRAINING_COMPLETE = "training.complete"
    TRAINING_FAIL = "training.fail"
    
    # 模型操作事件
    MODEL_CREATE = "model.create"
    MODEL_UPDATE = "model.update"
    MODEL_DELETE = "model.delete"
    MODEL_DEPLOY = "model.deploy"
    MODEL_DOWNLOAD = "model.download"
    
    # 系统事件
    SYSTEM_START = "system.start"
    SYSTEM_STOP = "system.stop"
    CONFIG_CHANGE = "system.config.change"
    BACKUP_CREATE = "system.backup.create"
    BACKUP_RESTORE = "system.backup.restore"
    
    # 安全事件
    SECURITY_VIOLATION = "security.violation"
    SUSPICIOUS_ACTIVITY = "security.suspicious"
    POLICY_VIOLATION = "security.policy.violation"
    INTRUSION_ATTEMPT = "security.intrusion.attempt"


class AuditLevel(Enum):
    """审计级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EncryptionAlgorithm(Enum):
    """加密算法枚举"""
    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    FERNET = "fernet"
    RSA_2048 = "rsa-2048"
    RSA_4096 = "rsa-4096"


class KeyType(Enum):
    """密钥类型枚举"""
    SYMMETRIC = "symmetric"
    ASYMMETRIC_PUBLIC = "asymmetric_public"
    ASYMMETRIC_PRIVATE = "asymmetric_private"
    DERIVED = "derived"


class ComplianceStandard(Enum):
    """合规标准枚举"""
    GDPR = "gdpr"  # 欧盟通用数据保护条例
    CCPA = "ccpa"  # 加州消费者隐私法
    SOC2 = "soc2"  # SOC 2 Type II
    ISO27001 = "iso27001"  # ISO 27001
    HIPAA = "hipaa"  # 健康保险便携性和责任法案
    PCI_DSS = "pci_dss"  # 支付卡行业数据安全标准


class ComplianceLevel(Enum):
    """合规级别"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNKNOWN = "unknown"


class DataCategory(Enum):
    """数据类别"""
    PERSONAL_DATA = "personal_data"  # 个人数据
    SENSITIVE_DATA = "sensitive_data"  # 敏感数据
    FINANCIAL_DATA = "financial_data"  # 金融数据
    HEALTH_DATA = "health_data"  # 健康数据
    BIOMETRIC_DATA = "biometric_data"  # 生物识别数据
    LOCATION_DATA = "location_data"  # 位置数据
    BEHAVIORAL_DATA = "behavioral_data"  # 行为数据
    TECHNICAL_DATA = "technical_data"  # 技术数据


class ProcessingPurpose(Enum):
    """处理目的"""
    CONSENT = "consent"  # 同意
    CONTRACT = "contract"  # 合同履行
    LEGAL_OBLIGATION = "legal_obligation"  # 法律义务
    VITAL_INTERESTS = "vital_interests"  # 重要利益
    PUBLIC_TASK = "public_task"  # 公共任务
    LEGITIMATE_INTERESTS = "legitimate_interests"  # 合法利益


@dataclass
class AuthContext:
    """认证上下文"""
    user_id: str
    session_id: str
    auth_methods: List[AuthMethod]
    ip_address: str
    user_agent: str
    device_fingerprint: str
    risk_score: float
    created_at: datetime
    expires_at: datetime
    last_activity: datetime


@dataclass
class MFAConfig:
    """多因子认证配置"""
    enabled: bool
    secret_key: str
    backup_codes: List[str]
    recovery_email: str
    recovery_phone: str


@dataclass
class AccessPolicy:
    """访问策略"""
    id: str
    name: str
    description: str
    effect: str  # "allow" or "deny"
    principals: List[str]  # 用户或角色
    resources: List[str]  # 资源模式
    actions: List[str]  # 操作
    conditions: Dict[str, Any]  # 条件
    priority: int  # 优先级
    created_at: datetime
    updated_at: datetime


@dataclass
class AccessRequest:
    """访问请求"""
    user_id: str
    resource: str
    action: str
    context: Dict[str, Any]
    timestamp: datetime


@dataclass
class AccessResult:
    """访问结果"""
    allowed: bool
    reason: str
    matched_policies: List[str]
    conditions_met: bool
    risk_score: float


@dataclass
class AuditEvent:
    """审计事件"""
    id: str
    timestamp: datetime
    event_type: AuditEventType
    level: AuditLevel
    user_id: Optional[str]
    session_id: Optional[str]
    source_ip: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    action: Optional[str]
    result: str  # "success", "failure", "error"
    message: str
    details: Dict[str, Any]
    risk_score: float
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        from dataclasses import asdict
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['event_type'] = self.event_type.value
        data['level'] = self.level.value
        return data


@dataclass
class AuditQuery:
    """审计查询条件"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    levels: Optional[List[AuditLevel]] = None
    user_ids: Optional[List[str]] = None
    resources: Optional[List[str]] = None
    results: Optional[List[str]] = None
    min_risk_score: Optional[float] = None
    max_risk_score: Optional[float] = None
    tags: Optional[List[str]] = None
    limit: int = 1000
    offset: int = 0


@dataclass
class EncryptionKey:
    """加密密钥"""
    id: str
    name: str
    key_type: KeyType
    algorithm: EncryptionAlgorithm
    key_data: bytes
    created_at: datetime
    expires_at: Optional[datetime]
    metadata: Dict[str, Any]
    
    def is_expired(self) -> bool:
        """检查密钥是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class EncryptionResult:
    """加密结果"""
    encrypted_data: bytes
    key_id: str
    algorithm: EncryptionAlgorithm
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DataProcessingRecord:
    """数据处理记录"""
    id: str
    data_subject_id: str
    data_categories: List[DataCategory]
    processing_purposes: List[ProcessingPurpose]
    legal_basis: str
    consent_given: bool
    consent_timestamp: Optional[datetime]
    retention_period: Optional[int]  # 保留期限（天）
    processing_location: str
    third_party_sharing: bool
    third_parties: List[str]
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceViolation:
    """合规违规"""
    id: str
    standard: ComplianceStandard
    rule_id: str
    rule_name: str
    severity: str  # "low", "medium", "high", "critical"
    description: str
    affected_data: List[str]
    remediation_steps: List[str]
    detected_at: datetime
    resolved_at: Optional[datetime]
    status: str  # "open", "in_progress", "resolved", "false_positive"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """合规报告"""
    id: str
    standard: ComplianceStandard
    overall_level: ComplianceLevel
    score: float  # 0-100
    total_rules: int
    compliant_rules: int
    violations: List[ComplianceViolation]
    recommendations: List[str]
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
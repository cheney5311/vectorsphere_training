# -*- coding: utf-8 -*-
"""
Security模块异常处理
"""

class SecurityError(Exception):
    """安全模块基础异常"""


class AuthenticationError(SecurityError):
    """认证异常"""


class InvalidCredentialsError(AuthenticationError):
    """无效凭证异常"""


class SessionExpiredError(AuthenticationError):
    """会话过期异常"""


class MFARequiredError(AuthenticationError):
    """需要多因子认证异常"""


class AuthorizationError(SecurityError):
    """授权异常"""

class PermissionDeniedError(AuthorizationError):
    """权限拒绝异常"""


class RoleNotFoundError(AuthorizationError):
    """角色未找到异常"""


class AuditError(SecurityError):
    """审计异常"""


class EncryptionError(SecurityError):
    """加密异常"""


class KeyNotFoundError(EncryptionError):
    """密钥未找到异常"""


class KeyExpiredError(EncryptionError):
    """密钥过期异常"""


class ComplianceError(SecurityError):
    """合规异常"""


class ComplianceViolationError(ComplianceError):
    """合规违规异常"""


class InvalidTokenError(SecurityError):
    """无效令牌异常"""


class InvalidSignatureError(SecurityError):
    """无效签名异常"""


class DataIntegrityError(SecurityError):
    """数据完整性异常"""
"""认证异常定义

定义认证模块的所有异常类，包括:
- 认证错误: 登录失败、凭证无效
- 授权错误: 权限不足、访问拒绝
- 令牌错误: 无效令牌、过期令牌
- 用户错误: 用户不存在、用户已存在
- 安全错误: 风险检测、异常行为
- MFA错误: 双因素认证相关
"""

from typing import Optional, Dict, Any, List


class AuthException(Exception):
    """认证基础异常
    
    所有认证相关异常的基类。
    
    Attributes:
        message: 错误消息
        error_code: 错误代码
        details: 详细信息字典
    """
    
    def __init__(self, 
                 message: str, 
                 error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """初始化异常
        
        Args:
            message: 错误消息
            error_code: 错误代码
            details: 详细信息
        """
        self.message = message
        self.error_code = error_code or "AUTH_ERROR"
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict: 异常信息字典
        """
        return {
            'message': self.message,
            'error_code': self.error_code,
            'details': self.details
        }


# ============================================================================
# 认证错误
# ============================================================================

class AuthenticationError(AuthException):
    """认证错误
    
    通用认证失败错误。
    """
    
    def __init__(self, 
                 message: str = "认证失败", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AUTHENTICATION_ERROR", details)


class InvalidCredentialsError(AuthException):
    """无效凭证错误
    
    用户名或密码错误。
    """
    
    def __init__(self, 
                 message: str = "用户名或密码错误", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "INVALID_CREDENTIALS", details)


class LoginBlockedError(AuthException):
    """登录被阻止错误
    
    由于安全原因登录被阻止。
    """
    
    def __init__(self, 
                 message: str = "登录已被阻止", 
                 reason: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if reason:
            details['reason'] = reason
        super().__init__(message, "LOGIN_BLOCKED", details)


class AccountLockedError(AuthException):
    """账户锁定错误
    
    账户已被锁定。
    """
    
    def __init__(self, 
                 message: str = "账户已被锁定", 
                 lockout_until: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if lockout_until:
            details['lockout_until'] = lockout_until
        super().__init__(message, "ACCOUNT_LOCKED", details)


class AccountInactiveError(AuthException):
    """账户未激活错误
    
    账户未激活或已停用。
    """
    
    def __init__(self, 
                 message: str = "账户未激活", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "ACCOUNT_INACTIVE", details)


class AccountSuspendedError(AuthException):
    """账户已暂停错误
    
    账户已被暂停。
    """
    
    def __init__(self, 
                 message: str = "账户已暂停", 
                 reason: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if reason:
            details['reason'] = reason
        super().__init__(message, "ACCOUNT_SUSPENDED", details)


# ============================================================================
# 授权错误
# ============================================================================

class AuthorizationError(AuthException):
    """授权错误
    
    通用权限不足错误。
    """
    
    def __init__(self, 
                 message: str = "权限不足", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "AUTHORIZATION_ERROR", details)


class PermissionDeniedError(AuthException):
    """权限拒绝错误
    
    用户没有执行操作的权限。
    """
    
    def __init__(self, 
                 message: str = "权限被拒绝", 
                 required_permission: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if required_permission:
            details['required_permission'] = required_permission
        super().__init__(message, "PERMISSION_DENIED", details)


class InsufficientPrivilegesError(AuthException):
    """权限不足错误
    
    用户权限级别不够。
    """
    
    def __init__(self, 
                 message: str = "权限级别不足", 
                 required_level: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if required_level:
            details['required_level'] = required_level
        super().__init__(message, "INSUFFICIENT_PRIVILEGES", details)


class RoleRequiredError(AuthException):
    """角色要求错误
    
    需要特定角色才能执行操作。
    """
    
    def __init__(self, 
                 message: str = "需要特定角色", 
                 required_roles: Optional[List[str]] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if required_roles:
            details['required_roles'] = required_roles
        super().__init__(message, "ROLE_REQUIRED", details)


# ============================================================================
# 令牌错误
# ============================================================================

class TokenError(AuthException):
    """令牌基础错误
    
    所有令牌相关错误的基类。
    """
    
    def __init__(self, 
                 message: str = "令牌错误", 
                 error_code: str = "TOKEN_ERROR",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class InvalidTokenError(TokenError):
    """无效令牌错误
    
    令牌格式无效或已被撤销。
    """
    
    def __init__(self, 
                 message: str = "无效令牌", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "INVALID_TOKEN", details)


class ExpiredTokenError(TokenError):
    """过期令牌错误
    
    令牌已过期。
    """
    
    def __init__(self, 
                 message: str = "令牌已过期", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "EXPIRED_TOKEN", details)


class RevokedTokenError(TokenError):
    """已撤销令牌错误
    
    令牌已被撤销。
    """
    
    def __init__(self, 
                 message: str = "令牌已被撤销", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "REVOKED_TOKEN", details)


class MissingTokenError(TokenError):
    """缺少令牌错误
    
    请求中缺少必要的令牌。
    """
    
    def __init__(self, 
                 message: str = "缺少访问令牌", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MISSING_TOKEN", details)


class RefreshTokenError(TokenError):
    """刷新令牌错误
    
    刷新令牌相关错误。
    """
    
    def __init__(self, 
                 message: str = "刷新令牌错误", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "REFRESH_TOKEN_ERROR", details)


# ============================================================================
# 用户错误
# ============================================================================

class UserError(AuthException):
    """用户基础错误
    
    所有用户相关错误的基类。
    """
    
    def __init__(self, 
                 message: str = "用户错误", 
                 error_code: str = "USER_ERROR",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class UserNotFoundError(UserError):
    """用户不存在错误
    
    指定的用户不存在。
    """
    
    def __init__(self, 
                 message: str = "用户不存在", 
                 user_identifier: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if user_identifier:
            details['user_identifier'] = user_identifier
        super().__init__(message, "USER_NOT_FOUND", details)


class UserAlreadyExistsError(UserError):
    """用户已存在错误
    
    用户名或邮箱已被使用。
    """
    
    def __init__(self, 
                 message: str = "用户已存在", 
                 conflict_field: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if conflict_field:
            details['conflict_field'] = conflict_field
        super().__init__(message, "USER_ALREADY_EXISTS", details)


class UserDisabledError(UserError):
    """用户已禁用错误
    
    用户账户已被禁用。
    """
    
    def __init__(self, 
                 message: str = "用户已禁用", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "USER_DISABLED", details)


class UserVerificationRequiredError(UserError):
    """需要用户验证错误
    
    用户需要完成验证。
    """
    
    def __init__(self, 
                 message: str = "需要完成用户验证", 
                 verification_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if verification_type:
            details['verification_type'] = verification_type
        super().__init__(message, "USER_VERIFICATION_REQUIRED", details)


# ============================================================================
# 安全错误
# ============================================================================

class SecurityError(AuthException):
    """安全基础错误
    
    所有安全相关错误的基类。
    """
    
    def __init__(self, 
                 message: str = "安全错误", 
                 error_code: str = "SECURITY_ERROR",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class HighRiskDetectedError(SecurityError):
    """高风险检测错误
    
    检测到高风险行为。
    """
    
    def __init__(self, 
                 message: str = "检测到高风险行为", 
                 risk_score: Optional[float] = None,
                 risk_factors: Optional[List[str]] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if risk_score is not None:
            details['risk_score'] = risk_score
        if risk_factors:
            details['risk_factors'] = risk_factors
        super().__init__(message, "HIGH_RISK_DETECTED", details)


class SuspiciousActivityError(SecurityError):
    """可疑活动错误
    
    检测到可疑活动。
    """
    
    def __init__(self, 
                 message: str = "检测到可疑活动", 
                 activity_type: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if activity_type:
            details['activity_type'] = activity_type
        super().__init__(message, "SUSPICIOUS_ACTIVITY", details)


class BruteForceDetectedError(SecurityError):
    """暴力破解检测错误
    
    检测到暴力破解尝试。
    """
    
    def __init__(self, 
                 message: str = "检测到暴力破解尝试", 
                 attempt_count: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if attempt_count is not None:
            details['attempt_count'] = attempt_count
        super().__init__(message, "BRUTE_FORCE_DETECTED", details)


class SessionHijackAttemptError(SecurityError):
    """会话劫持尝试错误
    
    检测到会话劫持尝试。
    """
    
    def __init__(self, 
                 message: str = "检测到会话劫持尝试", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "SESSION_HIJACK_ATTEMPT", details)


class LocationAnomalyError(SecurityError):
    """位置异常错误
    
    检测到位置异常（如不可能的地理移动）。
    """
    
    def __init__(self, 
                 message: str = "检测到位置异常", 
                 current_location: Optional[str] = None,
                 previous_location: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if current_location:
            details['current_location'] = current_location
        if previous_location:
            details['previous_location'] = previous_location
        super().__init__(message, "LOCATION_ANOMALY", details)


class DeviceAnomalyError(SecurityError):
    """设备异常错误
    
    检测到设备异常。
    """
    
    def __init__(self, 
                 message: str = "检测到设备异常", 
                 device_fingerprint: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if device_fingerprint:
            details['device_fingerprint'] = device_fingerprint
        super().__init__(message, "DEVICE_ANOMALY", details)


class RateLimitExceededError(SecurityError):
    """速率限制超出错误
    
    请求频率超出限制。
    """
    
    def __init__(self, 
                 message: str = "请求频率过高", 
                 retry_after: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if retry_after is not None:
            details['retry_after'] = retry_after
        super().__init__(message, "RATE_LIMIT_EXCEEDED", details)


# ============================================================================
# MFA 错误
# ============================================================================

class MFAError(AuthException):
    """MFA基础错误
    
    所有MFA相关错误的基类。
    """
    
    def __init__(self, 
                 message: str = "MFA错误", 
                 error_code: str = "MFA_ERROR",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class MFARequiredError(MFAError):
    """需要MFA验证错误
    
    需要进行MFA验证。
    """
    
    def __init__(self, 
                 message: str = "需要进行双因素认证", 
                 available_methods: Optional[List[str]] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if available_methods:
            details['available_methods'] = available_methods
        super().__init__(message, "MFA_REQUIRED", details)


class MFAInvalidCodeError(MFAError):
    """MFA验证码无效错误
    
    MFA验证码无效。
    """
    
    def __init__(self, 
                 message: str = "MFA验证码无效", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MFA_INVALID_CODE", details)


class MFACodeExpiredError(MFAError):
    """MFA验证码过期错误
    
    MFA验证码已过期。
    """
    
    def __init__(self, 
                 message: str = "MFA验证码已过期", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MFA_CODE_EXPIRED", details)


class MFANotSetupError(MFAError):
    """MFA未设置错误
    
    用户尚未设置MFA。
    """
    
    def __init__(self, 
                 message: str = "尚未设置双因素认证", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "MFA_NOT_SETUP", details)


class MFADeviceNotFoundError(MFAError):
    """MFA设备未找到错误
    
    MFA设备不存在。
    """
    
    def __init__(self, 
                 message: str = "MFA设备不存在", 
                 device_id: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if device_id:
            details['device_id'] = device_id
        super().__init__(message, "MFA_DEVICE_NOT_FOUND", details)


class MFAMaxAttemptsError(MFAError):
    """MFA最大尝试次数错误
    
    MFA验证尝试次数过多。
    """
    
    def __init__(self, 
                 message: str = "MFA验证尝试次数过多", 
                 lockout_minutes: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if lockout_minutes is not None:
            details['lockout_minutes'] = lockout_minutes
        super().__init__(message, "MFA_MAX_ATTEMPTS", details)


# ============================================================================
# 密码错误
# ============================================================================

class PasswordError(AuthException):
    """密码基础错误
    
    所有密码相关错误的基类。
    """
    
    def __init__(self, 
                 message: str = "密码错误", 
                 error_code: str = "PASSWORD_ERROR",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class WeakPasswordError(PasswordError):
    """弱密码错误
    
    密码强度不足。
    """
    
    def __init__(self, 
                 message: str = "密码强度不足", 
                 requirements: Optional[List[str]] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if requirements:
            details['requirements'] = requirements
        super().__init__(message, "WEAK_PASSWORD", details)


class PasswordReusedError(PasswordError):
    """密码重用错误
    
    新密码与近期使用过的密码相同。
    """
    
    def __init__(self, 
                 message: str = "不能使用最近使用过的密码", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "PASSWORD_REUSED", details)


class PasswordExpiredError(PasswordError):
    """密码过期错误
    
    密码已过期，需要修改。
    """
    
    def __init__(self, 
                 message: str = "密码已过期，请修改密码", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "PASSWORD_EXPIRED", details)


# ============================================================================
# API密钥错误
# ============================================================================

class ApiKeyError(AuthException):
    """API密钥基础错误
    
    所有API密钥相关错误的基类。
    """
    
    def __init__(self, 
                 message: str = "API密钥错误", 
                 error_code: str = "API_KEY_ERROR",
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details)


class InvalidApiKeyError(ApiKeyError):
    """无效API密钥错误
    
    API密钥无效。
    """
    
    def __init__(self, 
                 message: str = "无效的API密钥", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "INVALID_API_KEY", details)


class ExpiredApiKeyError(ApiKeyError):
    """API密钥过期错误
    
    API密钥已过期。
    """
    
    def __init__(self, 
                 message: str = "API密钥已过期", 
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "EXPIRED_API_KEY", details)


class ApiKeyRateLimitError(ApiKeyError):
    """API密钥速率限制错误
    
    API密钥请求超出速率限制。
    """
    
    def __init__(self, 
                 message: str = "API密钥请求过于频繁", 
                 retry_after: Optional[int] = None,
                 details: Optional[Dict[str, Any]] = None):
        details = details or {}
        if retry_after is not None:
            details['retry_after'] = retry_after
        super().__init__(message, "API_KEY_RATE_LIMIT", details)

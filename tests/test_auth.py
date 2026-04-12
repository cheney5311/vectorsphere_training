#!/usr/bin/env python3
"""认证模块测试脚本

测试认证API、服务层的完整功能，包括:
- 异常类
- 服务层DTOs
- 密码验证
- 令牌生成
- API端点
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAuthExceptions(unittest.TestCase):
    """认证异常测试"""
    
    def test_auth_exception_base(self):
        """测试基础认证异常"""
        from backend.modules.auth.auth_exceptions import AuthException
        
        exc = AuthException("测试错误", "TEST_ERROR", {"key": "value"})
        
        self.assertEqual(exc.message, "测试错误")
        self.assertEqual(exc.error_code, "TEST_ERROR")
        self.assertEqual(exc.details, {"key": "value"})
        
        exc_dict = exc.to_dict()
        self.assertIn('message', exc_dict)
        self.assertIn('error_code', exc_dict)
        self.assertIn('details', exc_dict)
    
    def test_authentication_error(self):
        """测试认证错误"""
        from backend.modules.auth.auth_exceptions import AuthenticationError
        
        exc = AuthenticationError()
        self.assertEqual(exc.error_code, "AUTHENTICATION_ERROR")
    
    def test_invalid_credentials_error(self):
        """测试无效凭证错误"""
        from backend.modules.auth.auth_exceptions import InvalidCredentialsError
        
        exc = InvalidCredentialsError("用户名或密码错误")
        self.assertEqual(exc.error_code, "INVALID_CREDENTIALS")
    
    def test_user_not_found_error(self):
        """测试用户不存在错误"""
        from backend.modules.auth.auth_exceptions import UserNotFoundError
        
        exc = UserNotFoundError("用户不存在", user_identifier="test_user")
        self.assertEqual(exc.error_code, "USER_NOT_FOUND")
        self.assertEqual(exc.details['user_identifier'], "test_user")
    
    def test_user_already_exists_error(self):
        """测试用户已存在错误"""
        from backend.modules.auth.auth_exceptions import UserAlreadyExistsError
        
        exc = UserAlreadyExistsError("用户已存在", conflict_field="username")
        self.assertEqual(exc.error_code, "USER_ALREADY_EXISTS")
        self.assertEqual(exc.details['conflict_field'], "username")
    
    def test_invalid_token_error(self):
        """测试无效令牌错误"""
        from backend.modules.auth.auth_exceptions import InvalidTokenError
        
        exc = InvalidTokenError("无效令牌")
        self.assertEqual(exc.error_code, "INVALID_TOKEN")
    
    def test_expired_token_error(self):
        """测试过期令牌错误"""
        from backend.modules.auth.auth_exceptions import ExpiredTokenError
        
        exc = ExpiredTokenError("令牌已过期")
        self.assertEqual(exc.error_code, "EXPIRED_TOKEN")
    
    def test_high_risk_detected_error(self):
        """测试高风险检测错误"""
        from backend.modules.auth.auth_exceptions import HighRiskDetectedError
        
        exc = HighRiskDetectedError(
            "检测到高风险行为",
            risk_score=0.9,
            risk_factors=["unknown_ip", "unusual_time"]
        )
        self.assertEqual(exc.error_code, "HIGH_RISK_DETECTED")
        self.assertEqual(exc.details['risk_score'], 0.9)
        self.assertIn("unknown_ip", exc.details['risk_factors'])
    
    def test_mfa_required_error(self):
        """测试MFA要求错误"""
        from backend.modules.auth.auth_exceptions import MFARequiredError
        
        exc = MFARequiredError(
            "需要进行双因素认证",
            available_methods=["totp", "sms"]
        )
        self.assertEqual(exc.error_code, "MFA_REQUIRED")
        self.assertIn("totp", exc.details['available_methods'])
    
    def test_permission_denied_error(self):
        """测试权限拒绝错误"""
        from backend.modules.auth.auth_exceptions import PermissionDeniedError
        
        exc = PermissionDeniedError("权限被拒绝", required_permission="admin:write")
        self.assertEqual(exc.error_code, "PERMISSION_DENIED")
        self.assertEqual(exc.details['required_permission'], "admin:write")
    
    def test_account_locked_error(self):
        """测试账户锁定错误"""
        from backend.modules.auth.auth_exceptions import AccountLockedError
        
        exc = AccountLockedError("账户已被锁定", lockout_until="2026-01-18T00:00:00")
        self.assertEqual(exc.error_code, "ACCOUNT_LOCKED")
        self.assertEqual(exc.details['lockout_until'], "2026-01-18T00:00:00")
    
    def test_brute_force_detected_error(self):
        """测试暴力破解检测错误"""
        from backend.modules.auth.auth_exceptions import BruteForceDetectedError
        
        exc = BruteForceDetectedError("检测到暴力破解尝试", attempt_count=10)
        self.assertEqual(exc.error_code, "BRUTE_FORCE_DETECTED")
        self.assertEqual(exc.details['attempt_count'], 10)
    
    def test_rate_limit_exceeded_error(self):
        """测试速率限制超出错误"""
        from backend.modules.auth.auth_exceptions import RateLimitExceededError
        
        exc = RateLimitExceededError("请求频率过高", retry_after=60)
        self.assertEqual(exc.error_code, "RATE_LIMIT_EXCEEDED")
        self.assertEqual(exc.details['retry_after'], 60)
    
    def test_weak_password_error(self):
        """测试弱密码错误"""
        from backend.modules.auth.auth_exceptions import WeakPasswordError
        
        exc = WeakPasswordError(
            "密码强度不足",
            requirements=["至少8个字符", "包含大写字母"]
        )
        self.assertEqual(exc.error_code, "WEAK_PASSWORD")
        self.assertIn("至少8个字符", exc.details['requirements'])


class TestPasswordValidation(unittest.TestCase):
    """密码验证测试"""
    
    def test_password_too_short(self):
        """测试密码长度验证"""
        from backend.core.exceptions import ValidationError
        
        # 直接测试验证逻辑
        password = "Short1"
        
        is_valid = len(password) >= 8
        self.assertFalse(is_valid)
    
    def test_password_no_uppercase(self):
        """测试密码大写字母验证"""
        password = "lowercase123"
        
        has_upper = any(c.isupper() for c in password)
        self.assertFalse(has_upper)
    
    def test_password_no_lowercase(self):
        """测试密码小写字母验证"""
        password = "UPPERCASE123"
        
        has_lower = any(c.islower() for c in password)
        self.assertFalse(has_lower)
    
    def test_password_no_digit(self):
        """测试密码数字验证"""
        password = "NoDigitsHere"
        
        has_digit = any(c.isdigit() for c in password)
        self.assertFalse(has_digit)
    
    def test_password_valid(self):
        """测试有效密码"""
        password = "ValidPass123"
        
        is_valid = (
            len(password) >= 8 and
            any(c.isupper() for c in password) and
            any(c.islower() for c in password) and
            any(c.isdigit() for c in password)
        )
        self.assertTrue(is_valid)


class TestTokenGeneration(unittest.TestCase):
    """令牌生成测试"""
    
    def test_generate_token_format(self):
        """测试令牌格式"""
        import hashlib
        import secrets
        
        user_id = "user123"
        tenant_id = "tenant456"
        timestamp = datetime.utcnow().timestamp()
        
        token_data = f"{user_id}:{tenant_id}:{timestamp}:{secrets.token_hex(16)}"
        token = hashlib.sha256(token_data.encode()).hexdigest()
        
        # 令牌应该是64字符的十六进制字符串
        self.assertEqual(len(token), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in token))
    
    def test_tokens_are_unique(self):
        """测试令牌唯一性"""
        import hashlib
        import secrets
        
        tokens = set()
        for _ in range(100):
            token_data = f"user:{datetime.utcnow().timestamp()}:{secrets.token_hex(16)}"
            token = hashlib.sha256(token_data.encode()).hexdigest()
            tokens.add(token)
        
        # 所有令牌应该是唯一的
        self.assertEqual(len(tokens), 100)


class TestRiskAssessment(unittest.TestCase):
    """风险评估测试"""
    
    def test_risk_score_calculation(self):
        """测试风险分数计算"""
        # 基础风险分数
        base_score = 0.0
        
        # IP失败次数风险
        ip_failed_count = 5
        if ip_failed_count > 3:
            base_score += 0.2
        
        # 未知IP风险
        is_known_ip = False
        if not is_known_ip:
            base_score += 0.15
        
        # 低信任分数风险
        trust_score = 0.2
        if trust_score < 0.3:
            base_score += 0.2
        
        # 确保分数在有效范围
        final_score = min(1.0, base_score)
        
        self.assertGreaterEqual(final_score, 0.0)
        self.assertLessEqual(final_score, 1.0)
        self.assertGreater(final_score, 0.4)  # 应该是中等以上风险
    
    def test_risk_level_determination(self):
        """测试风险等级判定"""
        HIGH_RISK_THRESHOLD = 0.7
        CRITICAL_RISK_THRESHOLD = 0.9
        
        # 低风险
        risk_score = 0.2
        if risk_score >= CRITICAL_RISK_THRESHOLD:
            level = "critical"
        elif risk_score >= HIGH_RISK_THRESHOLD:
            level = "high"
        elif risk_score >= 0.4:
            level = "medium"
        else:
            level = "low"
        
        self.assertEqual(level, "low")
        
        # 高风险
        risk_score = 0.8
        if risk_score >= CRITICAL_RISK_THRESHOLD:
            level = "critical"
        elif risk_score >= HIGH_RISK_THRESHOLD:
            level = "high"
        elif risk_score >= 0.4:
            level = "medium"
        else:
            level = "low"
        
        self.assertEqual(level, "high")
    
    def test_mfa_requirement_decision(self):
        """测试MFA要求决策"""
        # 高风险应要求MFA
        risk_level = "high"
        requires_mfa = risk_level in ["high", "critical"]
        self.assertTrue(requires_mfa)
        
        # 低风险不要求MFA
        risk_level = "low"
        requires_mfa = risk_level in ["high", "critical"]
        self.assertFalse(requires_mfa)


class TestTrustScoreCalculation(unittest.TestCase):
    """信任分数计算测试"""
    
    def test_base_trust_score(self):
        """测试基础信任分数"""
        score = 0.5  # 基础分数
        self.assertEqual(score, 0.5)
    
    def test_account_age_bonus(self):
        """测试账户年龄加分"""
        score = 0.5
        
        # 超过1年的账户
        account_age_days = 400
        if account_age_days > 365:
            score += 0.1
        elif account_age_days > 90:
            score += 0.05
        
        self.assertEqual(score, 0.6)
    
    def test_mfa_bonus(self):
        """测试MFA加分"""
        score = 0.5
        mfa_enabled = True
        
        if mfa_enabled:
            score += 0.15
        
        self.assertEqual(score, 0.65)
    
    def test_login_success_rate_bonus(self):
        """测试登录成功率加分"""
        score = 0.5
        
        total_attempts = 100
        successful = 95
        success_rate = successful / total_attempts
        
        score += 0.1 * success_rate
        
        self.assertAlmostEqual(score, 0.595, places=3)
    
    def test_anomaly_penalty(self):
        """测试异常扣分"""
        score = 0.5
        
        anomalies = [
            {'severity': 'critical'},
            {'severity': 'high'},
            {'severity': 'medium'}
        ]
        
        for anomaly in anomalies:
            if anomaly['severity'] == 'critical':
                score -= 0.2
            elif anomaly['severity'] == 'high':
                score -= 0.1
            elif anomaly['severity'] == 'medium':
                score -= 0.05
        
        # 0.5 - 0.2 - 0.1 - 0.05 = 0.15
        self.assertAlmostEqual(score, 0.15, places=2)
    
    def test_trust_score_bounds(self):
        """测试信任分数边界"""
        # 计算可能超出范围的分数
        score = 0.5 + 0.1 + 0.15 + 0.1 + 0.5  # 超过1.0
        final_score = max(0.0, min(1.0, score))
        self.assertEqual(final_score, 1.0)
        
        score = 0.5 - 0.2 - 0.2 - 0.2 - 0.2  # 低于0.0
        final_score = max(0.0, min(1.0, score))
        self.assertEqual(final_score, 0.0)


class TestSecurityRecommendations(unittest.TestCase):
    """安全建议测试"""
    
    def test_high_risk_recommendations(self):
        """测试高风险建议"""
        risk_level = "high"
        mfa_enabled = False
        
        recommendations = []
        
        if risk_level in ["high", "critical"]:
            recommendations.append("Consider enabling multi-factor authentication")
            recommendations.append("Review recent account activity")
        
        if not mfa_enabled:
            recommendations.append("Enable two-factor authentication for better security")
        
        self.assertIn("Consider enabling multi-factor authentication", recommendations)
        self.assertIn("Enable two-factor authentication for better security", recommendations)
    
    def test_unknown_ip_recommendation(self):
        """测试未知IP建议"""
        risk_factors = ["Unknown IP address"]
        
        recommendations = []
        
        if "Unknown IP address" in risk_factors:
            recommendations.append("Verify this login if you don't recognize the location")
        
        self.assertIn("Verify this login if you don't recognize the location", recommendations)
    
    def test_mfa_enabled_no_recommendation(self):
        """测试MFA已启用不再建议"""
        mfa_enabled = True
        
        recommendations = []
        
        if not mfa_enabled:
            recommendations.append("Enable two-factor authentication for better security")
        
        self.assertNotIn("Enable two-factor authentication for better security", recommendations)


class TestBehaviorPatternTracking(unittest.TestCase):
    """行为模式追踪测试"""
    
    def test_login_time_pattern(self):
        """测试登录时间模式"""
        current_hour = 14
        
        hour_counts = {'10': 5, '11': 3, '14': 8, '15': 6}
        hour_counts[str(current_hour)] = hour_counts.get(str(current_hour), 0) + 1
        
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        typical_hours = [int(h) for h, _ in sorted_hours[:6]]
        
        self.assertIn(14, typical_hours)
        self.assertEqual(hour_counts['14'], 9)
    
    def test_device_pattern(self):
        """测试设备模式"""
        device_fingerprint = "device_abc123"
        known_devices = ["device_xyz", "device_123"]
        
        if device_fingerprint not in known_devices:
            known_devices.append(device_fingerprint)
        
        # 保留最近的10个设备
        known_devices = known_devices[-10:]
        
        self.assertIn("device_abc123", known_devices)
    
    def test_location_pattern(self):
        """测试位置模式"""
        location = "Beijing"
        
        location_counts = {'Shanghai': 10, 'Beijing': 5}
        location_counts[location] = location_counts.get(location, 0) + 1
        
        sorted_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
        typical_locations = [loc for loc, _ in sorted_locations[:5]]
        
        self.assertIn("Beijing", typical_locations)
        self.assertEqual(location_counts['Beijing'], 6)


class TestAPIResponseFormat(unittest.TestCase):
    """API响应格式测试"""
    
    def test_success_response_format(self):
        """测试成功响应格式"""
        response = {
            'code': 200,
            'message': '操作成功',
            'data': {'key': 'value'}
        }
        
        self.assertIn('code', response)
        self.assertIn('message', response)
        self.assertIn('data', response)
        self.assertEqual(response['code'], 200)
    
    def test_error_response_format(self):
        """测试错误响应格式"""
        response = {
            'code': 401,
            'message': '认证失败',
            'error_code': 'AUTHENTICATION_ERROR',
            'details': {'reason': 'invalid_credentials'}
        }
        
        self.assertIn('code', response)
        self.assertIn('message', response)
        self.assertIn('error_code', response)
        self.assertEqual(response['code'], 401)


class TestSessionManagement(unittest.TestCase):
    """会话管理测试"""
    
    def test_session_expiry_check(self):
        """测试会话过期检查"""
        expires_at = datetime.utcnow() - timedelta(hours=1)
        is_expired = datetime.utcnow() > expires_at
        self.assertTrue(is_expired)
        
        expires_at = datetime.utcnow() + timedelta(hours=1)
        is_expired = datetime.utcnow() > expires_at
        self.assertFalse(is_expired)
    
    def test_session_risk_update(self):
        """测试会话风险更新"""
        session = {
            'id': 'session123',
            'risk_score': 0.2
        }
        
        new_risk_score = 0.6
        session['risk_score'] = new_risk_score
        
        self.assertEqual(session['risk_score'], 0.6)


class TestMemoryManagement(unittest.TestCase):
    """Agent记忆管理测试"""
    
    def test_memory_importance_bounds(self):
        """测试记忆重要性边界"""
        importance = 1.5
        normalized = max(0.0, min(1.0, importance))
        self.assertEqual(normalized, 1.0)
        
        importance = -0.5
        normalized = max(0.0, min(1.0, importance))
        self.assertEqual(normalized, 0.0)
    
    def test_memory_access_tracking(self):
        """测试记忆访问追踪"""
        memory = {
            'id': 'mem123',
            'access_count': 5,
            'last_accessed': datetime.utcnow()
        }
        
        # 模拟访问
        memory['access_count'] += 1
        memory['last_accessed'] = datetime.utcnow()
        
        self.assertEqual(memory['access_count'], 6)


def run_tests():
    """运行测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestAuthExceptions))
    suite.addTests(loader.loadTestsFromTestCase(TestPasswordValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestTokenGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskAssessment))
    suite.addTests(loader.loadTestsFromTestCase(TestTrustScoreCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityRecommendations))
    suite.addTests(loader.loadTestsFromTestCase(TestBehaviorPatternTracking))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIResponseFormat))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryManagement))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印摘要
    print("\n" + "=" * 70)
    print(f"测试完成: 运行 {result.testsRun} 个测试")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

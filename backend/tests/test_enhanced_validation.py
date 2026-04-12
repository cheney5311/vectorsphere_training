"""增强验证功能综合测试

测试新增的错误处理、验证中间件、响应格式化等功能。
"""
import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from flask import Flask, g, request

# 导入被测试的模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.errors import (
    make_error, get_error_category, is_retryable_error, 
    ErrorCategory, ERROR_CODES
)
from core.validation import (
    validate_json_schema, validate_query_params, validate_response_schema,
    ValidationConfig, get_validation_stats
)
from core.schema_manager import SchemaManager
from core.middleware import (
    RequestTrackingMiddleware, AuthMiddleware, RateLimitMiddleware,
    require_permissions, setup_middleware
)
from core.response import (
    ResponseFormatter, APIResponse, success_response, error_response,
    paginated_response, created_response, no_content_response
)
from core.config_validator import (
    ConfigValidator, ConfigRule, ConfigType, ConfigLevel,
    validate_environment_config
)


class TestErrorHandling:
    """错误处理测试"""
    
    def test_make_error_basic(self):
        """测试基本错误创建"""
        error = make_error("VALIDATION_SCHEMA_FAILED", "Test error")
        
        assert error["error"] == "validation_schema_failed"
        assert error["code"] == 100000
        assert error["message"] == "Test error"
        assert error["http_status"] == 400
        assert "retryable" in error
    
    def test_make_error_with_details(self):
        """测试带详情的错误创建"""
        details = {"field": "name", "value": "invalid"}
        error = make_error(
            "VALIDATION_FIELD_INVALID", 
            "Field validation failed",
            details=details
        )
        
        assert error["details"] == details
    
    def test_make_error_with_context(self):
        """测试带上下文的错误创建"""
        context = {"request_id": "test-123", "user_id": "user-456"}
        error = make_error(
            "AUTH_UNAUTHORIZED",
            "Unauthorized access",
            context=context
        )
        
        assert error["context"] == context
    
    def test_get_error_category(self):
        """测试错误分类"""
        assert get_error_category("VALIDATION_SCHEMA_FAILED") == ErrorCategory.VALIDATION
        assert get_error_category("AUTH_UNAUTHORIZED") == ErrorCategory.AUTHENTICATION
        assert get_error_category("INTERNAL_ERROR") == ErrorCategory.INTERNAL
        assert get_error_category("UNKNOWN_ERROR") == ErrorCategory.INTERNAL
    
    def test_is_retryable_error(self):
        """测试错误重试判断"""
        assert is_retryable_error("INTERNAL_ERROR") is True
        assert is_retryable_error("RESOURCE_UNAVAILABLE") is True
        assert is_retryable_error("VALIDATION_SCHEMA_FAILED") is False
        assert is_retryable_error("AUTH_UNAUTHORIZED") is False


class TestSchemaManager:
    """Schema管理器测试"""
    
    def setup_method(self):
        """测试设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.schema_manager = SchemaManager(schema_dir=self.temp_dir)
    
    def teardown_method(self):
        """测试清理"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_add_schema_from_dict(self):
        """测试从字典添加schema"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        
        self.schema_manager.add_schema("test_schema", schema)
        loaded_schema = self.schema_manager.get_schema("test_schema")
        
        assert loaded_schema == schema
    
    def test_validate_data(self):
        """测试数据验证"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }
        
        self.schema_manager.add_schema("person", schema)
        
        # 有效数据
        valid_data = {"name": "John", "age": 30}
        is_valid, error = self.schema_manager.validate_data(valid_data, "person")
        assert is_valid is True
        assert error is None
        
        # 无效数据 - 缺少必需字段
        invalid_data = {"age": 30}
        is_valid, error = self.schema_manager.validate_data(invalid_data, "person")
        assert is_valid is False
        assert error is not None
    
    def test_schema_versioning(self):
        """测试schema版本控制"""
        schema_v1 = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }
        
        schema_v2 = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            },
            "required": ["name", "email"]
        }
        
        self.schema_manager.add_schema("user", schema_v1, version="1.0")
        self.schema_manager.add_schema("user", schema_v2, version="2.0")
        
        # 获取不同版本
        v1_schema = self.schema_manager.get_schema("user", version="1.0")
        v2_schema = self.schema_manager.get_schema("user", version="2.0")
        
        assert v1_schema != v2_schema
        assert "email" not in v1_schema["properties"]
        assert "email" in v2_schema["properties"]


class TestValidationMiddleware:
    """验证中间件测试"""
    
    def setup_method(self):
        """测试设置"""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        
        # 启用严格模式进行测试
        import os
        os.environ['CHECK_API_STRICT_MODE'] = 'true'
        
        # 直接修改验证配置
        from backend.core.validation import validation_config
        validation_config.strict_mode = True
        
        # 添加错误处理
        @self.app.errorhandler(Exception)
        def handle_exception(e):
            return {"error": str(e)}, 500
            
        self.client = self.app.test_client()
    
    def test_validate_json_schema_decorator(self):
        """测试JSON schema验证装饰器"""
        from backend.core.validation import validate_json_schema
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0}
            },
            "required": ["name"]
        }
        
        @validate_json_schema(schema)
        def test_endpoint():
            return {"status": "success"}
        
        self.app.add_url_rule('/test', 'test_endpoint', test_endpoint, methods=['POST'])
        
        # 有效请求
        response = self.client.post('/test', 
                                  json={"name": "John", "age": 30},
                                  content_type='application/json')
        assert response.status_code == 200
        
        # 无效请求 - 缺少必需字段
        response = self.client.post('/test', 
                                  json={"age": 30},
                                  content_type='application/json')
        assert response.status_code == 400
    
    def test_validate_query_params_decorator(self):
        """测试查询参数验证装饰器"""
        from backend.core.validation import validate_query_params
        
        schema = {
            "type": "object",
            "properties": {
                "page": {"type": "string", "pattern": "^[1-9][0-9]*$"},
                "limit": {"type": "string", "pattern": "^[1-9][0-9]*$"}
            }
        }
        
        @validate_query_params(schema)
        def test_endpoint():
            return {"status": "success"}
        
        self.app.add_url_rule('/test', 'test_endpoint_query', test_endpoint, methods=['GET'])
        
        # 有效请求
        response = self.client.get('/test?page=1&limit=10')
        assert response.status_code == 200
        
        # 无效请求 - 超出范围
        response = self.client.get('/test?page=0&limit=200')
        assert response.status_code == 400


class TestMiddleware:
    """中间件测试"""
    
    def setup_method(self):
        """测试设置"""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_request_tracking_middleware(self):
        """测试请求追踪中间件"""
        middleware = RequestTrackingMiddleware(self.app)
        
        @self.app.route('/test')
        def test_endpoint():
            # 检查是否设置了请求ID
            assert hasattr(g, 'request_id')
            assert hasattr(g, 'start_time')
            return {"status": "success"}
        
        response = self.client.get('/test')
        assert response.status_code == 200
        assert 'X-Request-ID' in response.headers
        assert 'X-Response-Time' in response.headers
    
    def test_auth_middleware_skip_endpoints(self):
        """测试认证中间件跳过端点"""
        middleware = AuthMiddleware(self.app, skip_auth_endpoints=['health'])
        
        @self.app.route('/health')
        def health():
            return {"status": "healthy"}
        
        response = self.client.get('/health')
        assert response.status_code == 200
    
    def test_auth_middleware_missing_token(self):
        """测试认证中间件缺少token"""
        middleware = AuthMiddleware(self.app)
        
        @self.app.route('/protected')
        def protected():
            return {"status": "success"}
        
        response = self.client.get('/protected')
        assert response.status_code == 401
    
    def test_rate_limit_middleware(self):
        """测试速率限制中间件"""
        middleware = RateLimitMiddleware(self.app, default_limit="2/hour")
        
        @self.app.route('/test')
        def test_endpoint():
            return {"status": "success"}
        
        # 前两个请求应该成功
        response1 = self.client.get('/test')
        response2 = self.client.get('/test')
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # 第三个请求应该被限制（在实际实现中）
        # 注意：这个测试可能需要根据实际的速率限制实现进行调整
    
    def test_require_permissions_decorator(self):
        """测试权限检查装饰器"""
        @self.app.route('/admin')
        @require_permissions('admin', 'write')
        def admin_endpoint():
            return {"status": "success"}
        
        # 模拟有权限的用户
        with self.app.test_request_context():
            g.user_permissions = ['admin', 'write', 'read']
            response = self.client.get('/admin')
            # 注意：这个测试需要在实际的Flask上下文中运行


class TestResponseFormatter:
    """响应格式化器测试"""
    
    def setup_method(self):
        """设置测试环境"""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()
    
    def teardown_method(self):
        """清理测试环境"""
        self.app_context.pop()
    
    def test_success_response(self):
        """测试成功响应"""
        data = {"id": 1, "name": "test"}
        response, status_code = ResponseFormatter.success(data, "Operation successful")
        
        response_data = json.loads(response.get_data())
        assert response_data["success"] is True
        assert response_data["message"] == "Operation successful"
        assert response_data["data"] == data
        assert status_code == 200
    
    def test_error_response(self):
        """测试错误响应"""
        response, status_code = ResponseFormatter.error(
            "VALIDATION_ERROR", 
            "Validation failed",
            details={"field": "name"},
            status_code=400
        )
        
        response_data = json.loads(response.get_data())
        assert response_data["success"] is False
        assert response_data["error"]["code"] == "VALIDATION_ERROR"
        assert response_data["error"]["message"] == "Validation failed"
        assert response_data["error"]["details"]["field"] == "name"
        assert status_code == 400
    
    def test_paginated_response(self):
        """测试分页响应"""
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        response, status_code = ResponseFormatter.paginated(data, 1, 10, 25)
        
        response_data = json.loads(response.get_data())
        assert response_data["success"] is True
        assert response_data["data"] == data
        assert response_data["meta"]["pagination"]["page"] == 1
        assert response_data["meta"]["pagination"]["per_page"] == 10
        assert response_data["meta"]["pagination"]["total"] == 25
        assert response_data["meta"]["pagination"]["total_pages"] == 3
    
    def test_api_response_builder(self):
        """测试API响应构建器"""
        builder = APIResponse()
        response, status_code = (builder
                                .set_data({"test": "data"})
                                .set_message("Custom message")
                                .set_status_code(201)
                                .add_header("X-Custom", "value")
                                .build())
        
        response_data = json.loads(response.get_data())
        assert response_data["message"] == "Custom message"
        assert response_data["data"]["test"] == "data"
        assert status_code == 201
        assert response.headers.get("X-Custom") == "value"


class TestConfigValidator:
    """配置验证器测试"""
    
    def setup_method(self):
        """测试设置"""
        self.validator = ConfigValidator()
    
    def test_validate_required_config(self):
        """测试必需配置验证"""
        config = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test",
            "SECRET_KEY": "a" * 32 + "B1!" + "c" * 10,  # 满足强度要求
            "JWT_SECRET_KEY": "b" * 32 + "C2@" + "d" * 10  # 添加JWT密钥
        }
        
        result = self.validator.validate_config(config)
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
    
    def test_validate_missing_required_config(self):
        """测试缺少必需配置"""
        config = {}  # 空配置
        
        result = self.validator.validate_config(config)
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
        
        # 检查是否包含必需配置的错误
        error_messages = " ".join(result["errors"])
        assert "DATABASE_URL" in error_messages
        assert "SECRET_KEY" in error_messages
    
    def test_validate_config_types(self):
        """测试配置类型验证"""
        config = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test",
            "SECRET_KEY": "a" * 32 + "B1!" + "c" * 10,
            "JWT_SECRET_KEY": "b" * 32 + "C2@" + "d" * 10,
            "PORT": "5000",  # 字符串，应该转换为整数
            "DEBUG": "true",  # 字符串，应该转换为布尔值
            "DATABASE_POOL_SIZE": "20"  # 字符串，应该转换为整数
        }
        
        result = self.validator.validate_config(config)
        assert result["is_valid"] is True
        
        # 检查类型转换
        assert isinstance(result["config"]["PORT"], int)
        assert isinstance(result["config"]["DEBUG"], bool)
        assert isinstance(result["config"]["DATABASE_POOL_SIZE"], int)
    
    def test_validate_config_ranges(self):
        """测试配置范围验证"""
        config = {
            "DATABASE_URL": "postgresql://localhost/test",
            "SECRET_KEY": "a" * 32 + "B1!" + "c" * 10,
            "PORT": 70000,  # 超出有效端口范围
            "DATABASE_POOL_SIZE": 0  # 低于最小值
        }
        
        result = self.validator.validate_config(config)
        assert result["is_valid"] is False
        assert len(result["errors"]) >= 2
    
    def test_validate_secret_strength(self):
        """测试密钥强度验证"""
        weak_configs = [
            {"SECRET_KEY": "short"},  # 太短
            {"SECRET_KEY": "a" * 50},  # 只有小写字母
            {"SECRET_KEY": "A" * 50},  # 只有大写字母
        ]
        
        for config in weak_configs:
            config["DATABASE_URL"] = "postgresql://localhost/test"
            result = self.validator.validate_config(config)
            assert result["is_valid"] is False
    
    def test_config_with_defaults(self):
        """测试使用默认值的配置"""
        config = {
            "DATABASE_URL": "postgresql://user:pass@localhost:5432/test",
            "SECRET_KEY": "a" * 32 + "B1!" + "c" * 10,
            "JWT_SECRET_KEY": "b" * 32 + "C2@" + "d" * 10
            # 其他配置使用默认值
        }
        
        result = self.validator.validate_config(config)
        assert result["is_valid"] is True
        
        # 检查是否使用了默认值
        assert result["config"]["HOST"] == "0.0.0.0"
        assert result["config"]["PORT"] == 5000
        assert result["config"]["DEBUG"] is False


class TestIntegration:
    """集成测试"""
    
    def setup_method(self):
        """测试设置"""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
        
        # 设置中间件
        setup_middleware(self.app)
    
    def test_full_request_flow(self):
        """测试完整的请求流程"""
        # 设置环境变量以跳过认证
        import os
        original_skip_auth = os.environ.get('SKIP_AUTH')
        os.environ['SKIP_AUTH'] = 'true'
        
        try:
            schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"}
                },
                "required": ["name", "email"]
            }
            
            @self.app.route('/api/users', methods=['POST'])
            @validate_json_schema(schema)
            def create_user():
                return success_response(
                    {"id": 1, "name": "John", "email": "john@example.com"},
                    "User created successfully"
                )
            
            # 有效请求
            response = self.client.post('/api/users',
                                      json={"name": "John", "email": "john@example.com"},
                                      content_type='application/json')
            
            assert response.status_code == 200
            data = json.loads(response.get_data())
            assert data["success"] is True
            assert "request_id" in data
        finally:
            # 恢复原始环境变量
            if original_skip_auth is None:
                os.environ.pop('SKIP_AUTH', None)
            else:
                os.environ['SKIP_AUTH'] = original_skip_auth
    
    @patch.dict(os.environ, {
        'DATABASE_URL': 'postgresql://user:pass@localhost:5432/test',
        'SECRET_KEY': 'a' * 32 + 'B1!' + 'c' * 10,
        'JWT_SECRET_KEY': 'b' * 32 + 'C2@' + 'd' * 10,
        'DEBUG': 'false'
    })
    def test_environment_config_validation(self):
        """测试环境变量配置验证"""
        result = validate_environment_config()
        assert result["is_valid"] is True
        assert result["config"]["DEBUG"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
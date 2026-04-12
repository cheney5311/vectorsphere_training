"""性能验证测试

测试新增功能的性能表现，包括验证中间件、错误处理等的性能影响。
"""
import pytest
import time
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
from flask import Flask

from backend.core.validation import validate_json_schema, get_validation_stats
from backend.core.schema_manager import SchemaManager
from backend.core.middleware import RequestTrackingMiddleware, RateLimitMiddleware
from backend.core.response import ResponseFormatter
from backend.core.errors import make_error


class TestValidationPerformance:
    """验证性能测试"""
    
    def setup_method(self):
        """测试设置"""
        self.schema_manager = SchemaManager()
        self.test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "email": {"type": "string", "format": "email"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 10
                }
            },
            "required": ["name", "email"]
        }
        self.schema_manager.add_schema("user", self.test_schema)
    
    def test_schema_validation_performance(self):
        """测试schema验证性能"""
        test_data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
            "tags": ["developer", "python", "flask"]
        }
        
        # 预热
        for _ in range(10):
            self.schema_manager.validate_data(test_data, "user")
        
        # 性能测试
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            is_valid, _ = self.schema_manager.validate_data(test_data, "user")
            assert is_valid is True
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        print(f"Schema validation performance:")
        print(f"Total time for {iterations} validations: {total_time:.4f}s")
        print(f"Average time per validation: {avg_time:.6f}s")
        print(f"Validations per second: {iterations/total_time:.2f}")
        
        # 性能断言 - 每次验证应该在1ms以内
        assert avg_time < 0.001, f"Validation too slow: {avg_time:.6f}s per validation"
    
    def test_schema_caching_performance(self):
        """测试schema缓存性能"""
        # 测试缓存命中性能
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            schema = self.schema_manager.get_schema("user")
            assert schema is not None
        
        end_time = time.time()
        cached_time = end_time - start_time
        
        # 测试无缓存性能（重新加载）
        self.schema_manager._schema_cache.clear()
        
        start_time = time.time()
        for _ in range(iterations):
            self.schema_manager.add_schema("user_temp", self.test_schema)
            schema = self.schema_manager.get_schema("user_temp")
            assert schema is not None
        
        end_time = time.time()
        uncached_time = end_time - start_time
        
        print(f"Schema caching performance:")
        print(f"Cached access time: {cached_time:.4f}s for {iterations} accesses")
        print(f"Uncached access time: {uncached_time:.4f}s for {iterations} accesses")
        print(f"Cache speedup: {uncached_time/cached_time:.2f}x")
        
        # 缓存应该显著提高性能
        assert cached_time < uncached_time / 5, "Cache not providing sufficient speedup"
    
    def test_concurrent_validation_performance(self):
        """测试并发验证性能"""
        test_data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
            "tags": ["developer", "python"]
        }
        
        def validate_data():
            return self.schema_manager.validate_data(test_data, "user")
        
        # 并发测试
        num_threads = 10
        iterations_per_thread = 100
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for _ in range(num_threads):
                for _ in range(iterations_per_thread):
                    future = executor.submit(validate_data)
                    futures.append(future)
            
            # 等待所有任务完成
            for future in as_completed(futures):
                is_valid, _ = future.result()
                assert is_valid is True
        
        end_time = time.time()
        total_time = end_time - start_time
        total_validations = num_threads * iterations_per_thread
        
        print(f"Concurrent validation performance:")
        print(f"Total validations: {total_validations}")
        print(f"Total time: {total_time:.4f}s")
        print(f"Validations per second: {total_validations/total_time:.2f}")
        
        # 并发性能应该合理
        assert total_time < 10, f"Concurrent validation too slow: {total_time:.4f}s"


class TestMiddlewarePerformance:
    """中间件性能测试"""
    
    def setup_method(self):
        """测试设置"""
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_request_tracking_middleware_performance(self):
        """测试请求追踪中间件性能"""
        middleware = RequestTrackingMiddleware(self.app)
        
        @self.app.route('/test')
        def test_endpoint():
            return {"status": "success"}
        
        # 预热
        for _ in range(10):
            self.client.get('/test')
        
        # 性能测试
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            response = self.client.get('/test')
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        print(f"Request tracking middleware performance:")
        print(f"Total time for {iterations} requests: {total_time:.4f}s")
        print(f"Average time per request: {avg_time:.6f}s")
        print(f"Requests per second: {iterations/total_time:.2f}")
        
        # 中间件开销应该很小
        assert avg_time < 0.01, f"Middleware overhead too high: {avg_time:.6f}s per request"
    
    def test_rate_limit_middleware_performance(self):
        """测试速率限制中间件性能"""
        # 设置较高的限制以避免触发限制
        middleware = RateLimitMiddleware(self.app, default_limit="10000/hour")
        
        @self.app.route('/test')
        def test_endpoint():
            return {"status": "success"}
        
        # 性能测试
        start_time = time.time()
        iterations = 100  # 较少的迭代次数，因为速率限制检查可能较慢
        
        for _ in range(iterations):
            response = self.client.get('/test')
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        print(f"Rate limit middleware performance:")
        print(f"Total time for {iterations} requests: {total_time:.4f}s")
        print(f"Average time per request: {avg_time:.6f}s")
        print(f"Requests per second: {iterations/total_time:.2f}")
        
        # 速率限制中间件开销应该合理
        assert avg_time < 0.05, f"Rate limit middleware too slow: {avg_time:.6f}s per request"


class TestErrorHandlingPerformance:
    """错误处理性能测试"""
    
    def test_make_error_performance(self):
        """测试错误创建性能"""
        # 预热
        for _ in range(10):
            make_error("VALIDATION_SCHEMA_FAILED", "Test error")
        
        # 性能测试
        start_time = time.time()
        iterations = 10000
        
        for _ in range(iterations):
            error = make_error(
                "VALIDATION_SCHEMA_FAILED",
                "Test error message",
                details={"field": "name", "value": "invalid"},
                context={"request_id": "test-123"}
            )
            assert error["error"] == "validation_schema_failed"
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        print(f"Error creation performance:")
        print(f"Total time for {iterations} errors: {total_time:.4f}s")
        print(f"Average time per error: {avg_time:.6f}s")
        print(f"Errors per second: {iterations/total_time:.2f}")
        
        # 错误创建应该很快
        assert avg_time < 0.0005, f"Error creation too slow: {avg_time:.6f}s per error"
    
    def test_error_handling_with_large_details(self):
        """测试大量详情的错误处理性能"""
        # 创建大量详情数据
        large_details = {
            f"field_{i}": f"error_message_{i}" * 10
            for i in range(100)
        }
        
        start_time = time.time()
        iterations = 1000
        
        for _ in range(iterations):
            error = make_error(
                "VALIDATION_SCHEMA_FAILED",
                "Test error with large details",
                details=large_details
            )
            assert error["error"] == "validation_schema_failed"
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        print(f"Large error details performance:")
        print(f"Total time for {iterations} errors: {total_time:.4f}s")
        print(f"Average time per error: {avg_time:.6f}s")
        
        # 即使有大量详情，错误创建也应该合理快速
        assert avg_time < 0.001, f"Large error creation too slow: {avg_time:.6f}s per error"


class TestResponseFormatterPerformance:
    """响应格式化器性能测试"""

    def setup_method(self):
        self.app = Flask(__name__)
        self.app.config['TESTING'] = True
    
    def test_success_response_performance(self):
        """测试成功响应格式化性能"""
        test_data = {
            "id": 1,
            "name": "Test User",
            "email": "test@example.com",
            "metadata": {
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z"
            }
        }
        
        with self.app.app_context():
            # 预热
            for _ in range(10):
                ResponseFormatter.success(test_data, "Success message")
            
            # 性能测试
            start_time = time.time()
            iterations = 10000
            
            for _ in range(iterations):
                response, status_code = ResponseFormatter.success(test_data, "Success message")
                assert status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        print(f"Success response formatting performance:")
        print(f"Total time for {iterations} responses: {total_time:.4f}s")
        print(f"Average time per response: {avg_time:.6f}s")
        print(f"Responses per second: {iterations/total_time:.2f}")
        
        # 响应格式化应该很快
        assert avg_time < 0.0001, f"Response formatting too slow: {avg_time:.6f}s per response"
    
    def test_paginated_response_performance(self):
        """测试分页响应性能"""
        # 创建大量数据
        large_data = [{"id": i, "name": f"Item {i}"} for i in range(1000)]
        
        with self.app.app_context():
            start_time = time.time()
            iterations = 1000
            
            for _ in range(iterations):
                response, status_code = ResponseFormatter.paginated(
                    large_data, page=1, per_page=100, total=10000
                )
                assert status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        print(f"Paginated response performance:")
        print(f"Total time for {iterations} responses: {total_time:.4f}s")
        print(f"Average time per response: {avg_time:.6f}s")
        
        # 分页响应应该合理快速
        assert avg_time < 0.005, f"Paginated response too slow: {avg_time:.6f}s per response"


class TestMemoryUsage:
    """内存使用测试"""
    
    def test_schema_cache_memory_usage(self):
        """测试schema缓存内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        schema_manager = SchemaManager()
        
        # 加载大量schema
        for i in range(100):
            schema = {
                "type": "object",
                "properties": {
                    f"field_{j}": {"type": "string"}
                    for j in range(50)  # 每个schema 50个字段
                },
                "required": [f"field_{j}" for j in range(10)]
            }
            schema_manager.add_schema(f"schema_{i}", schema)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        memory_per_schema = memory_increase / 100
        
        print(f"Schema cache memory usage:")
        print(f"Initial memory: {initial_memory / 1024 / 1024:.2f} MB")
        print(f"Final memory: {final_memory / 1024 / 1024:.2f} MB")
        print(f"Memory increase: {memory_increase / 1024 / 1024:.2f} MB")
        print(f"Memory per schema: {memory_per_schema / 1024:.2f} KB")
        
        # 内存使用应该合理
        assert memory_increase < 50 * 1024 * 1024, "Schema cache using too much memory"
    
    def test_validation_stats_memory(self):
        """测试验证统计内存使用"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 执行大量验证以生成统计数据
        schema_manager = SchemaManager()
        test_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }
        schema_manager.add_schema("test", test_schema)
        
        for _ in range(10000):
            schema_manager.validate_data({"name": "test"}, "test")
        
        # 获取统计信息
        stats = get_validation_stats()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        print(f"Validation stats memory usage:")
        print(f"Memory increase: {memory_increase / 1024 / 1024:.2f} MB")
        print(f"Total validations: {stats.get('total_validations', 0)}")
        
        # 统计数据不应该占用过多内存
        assert memory_increase < 10 * 1024 * 1024, "Validation stats using too much memory"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s 显示print输出
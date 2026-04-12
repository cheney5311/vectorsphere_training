"""API性能优化工具"""

import time
import hashlib
import json
import functools
from typing import Dict, Any, Optional, Callable
from flask import request, jsonify, g
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class APICache:
    """API响应缓存"""

    def __init__(self, default_ttl: int = 300):  # 5分钟默认缓存
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.last_cleanup = time.time()

    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any], user_id: str = None) -> str:
        """生成缓存键"""
        # 包含端点、参数和用户ID（如果需要用户特定缓存）
        key_data = {
            'endpoint': endpoint,
            'params': sorted(params.items()) if params else [],
            'user_id': user_id
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, endpoint: str, params: Dict[str, Any] = None, user_id: str = None) -> Optional[Any]:
        """获取缓存数据"""
        self._cleanup_expired()

        cache_key = self._generate_cache_key(endpoint, params, user_id)
        cached_data = self.cache.get(cache_key)

        if cached_data and cached_data['expires'] > time.time():
            logger.debug(f"Cache hit for {endpoint}")
            return cached_data['data']

        logger.debug(f"Cache miss for {endpoint}")
        return None

    def set(self, endpoint: str, data: Any, params: Dict[str, Any] = None,
            user_id: str = None, ttl: int = None) -> None:
        """设置缓存数据"""
        cache_key = self._generate_cache_key(endpoint, params, user_id)
        ttl = ttl or self.default_ttl

        self.cache[cache_key] = {
            'data': data,
            'expires': time.time() + ttl,
            'created': time.time()
        }

        logger.debug(f"Cached data for {endpoint} (TTL: {ttl}s)")

    def invalidate(self, endpoint: str = None, user_id: str = None) -> None:
        """使缓存失效"""
        if endpoint is None and user_id is None:
            # 清空所有缓存
            self.cache.clear()
            logger.info("All cache cleared")
            return

        # 删除匹配的缓存项
        keys_to_delete = []
        for key, cached_data in self.cache.items():
            should_delete = False

            if endpoint and endpoint in key:
                should_delete = True
            if user_id and user_id in key:
                should_delete = True

            if should_delete:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self.cache[key]

        logger.info(f"Invalidated {len(keys_to_delete)} cache entries")

    def _cleanup_expired(self) -> None:
        """清理过期缓存"""
        now = time.time()
        if now - self.last_cleanup < 60:  # 每分钟清理一次
            return

        expired_keys = [
            key for key, data in self.cache.items()
            if data['expires'] <= now
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        self.last_cleanup = now

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        now = time.time()
        active_entries = sum(1 for data in self.cache.values() if data['expires'] > now)
        expired_entries = len(self.cache) - active_entries

        return {
            'total_entries': len(self.cache),
            'active_entries': active_entries,
            'expired_entries': expired_entries,
            'memory_usage_mb': len(str(self.cache)) / (1024 * 1024)
        }


class PerformanceMonitor:
    """API性能监控"""

    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.slow_queries: list = []
        self.error_count: Dict[str, int] = {}

    def record_request(self, endpoint: str, duration: float, status_code: int) -> None:
        """记录请求性能"""
        if endpoint not in self.metrics:
            self.metrics[endpoint] = []

        self.metrics[endpoint].append({
            'duration': duration,
            'status_code': status_code,
            'timestamp': time.time()
        })

        # 记录慢查询（超过1秒）
        if duration > 1.0:
            self.slow_queries.append({
                'endpoint': endpoint,
                'duration': duration,
                'timestamp': time.time()
            })

        # 记录错误
        if status_code >= 400:
            self.error_count[endpoint] = self.error_count.get(endpoint, 0) + 1

        # 保持最近1000条记录
        if len(self.metrics[endpoint]) > 1000:
            self.metrics[endpoint] = self.metrics[endpoint][-1000:]

    def get_endpoint_stats(self, endpoint: str) -> Dict[str, Any]:
        """获取端点统计信息"""
        if endpoint not in self.metrics:
            return {}

        durations = [m['duration'] for m in self.metrics[endpoint]]
        if not durations:
            return {}

        return {
            'total_requests': len(durations),
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'error_count': self.error_count.get(endpoint, 0),
            'error_rate': self.error_count.get(endpoint, 0) / len(durations)
        }

    def get_slow_queries(self, limit: int = 10) -> list:
        """获取慢查询列表"""
        return sorted(self.slow_queries, key=lambda x: x['duration'], reverse=True)[:limit]

    def get_overall_stats(self) -> Dict[str, Any]:
        """获取整体统计信息"""
        total_requests = sum(len(metrics) for metrics in self.metrics.values())
        total_errors = sum(self.error_count.values())

        all_durations = []
        for metrics in self.metrics.values():
            all_durations.extend([m['duration'] for m in metrics])

        if not all_durations:
            return {'total_requests': 0}

        return {
            'total_requests': total_requests,
            'total_errors': total_errors,
            'error_rate': total_errors / total_requests if total_requests > 0 else 0,
            'avg_response_time': sum(all_durations) / len(all_durations),
            'slow_queries_count': len(self.slow_queries),
            'endpoints_count': len(self.metrics)
        }


# 全局实例
api_cache = APICache()
performance_monitor = PerformanceMonitor()


def cached_response(ttl: int = 300):
    """缓存响应装饰器"""
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{request.endpoint}:{hash(str(request.args))}"

            # 尝试从缓存获取
            cached_result = api_cache.get(request.endpoint, dict(request.args))
            if cached_result:
                return cached_result

            # 执行函数
            start_time = time.time()
            result = f(*args, **kwargs)
            duration = time.time() - start_time

            # 缓存结果
            api_cache.set(request.endpoint, result, dict(request.args), ttl=ttl)

            # 记录性能
            performance_monitor.record_request(request.endpoint, duration, result[1] if isinstance(result, tuple) else 200)

            return result
        return decorated_function
    return decorator


def monitor_performance():
    """性能监控装饰器"""
    def decorator(f):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            start_time = time.time()
            try:
                result = f(*args, **kwargs)
                duration = time.time() - start_time
                status_code = result[1] if isinstance(result, tuple) else 200
                performance_monitor.record_request(request.endpoint, duration, status_code)
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_monitor.record_request(request.endpoint, duration, 500)
                raise
        return decorated_function
    return decorator
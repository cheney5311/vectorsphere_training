"""性能API公共工具模块

提供配置加载、响应构建、状态评估、数据验证、缓存、限流等公共功能。
生产级实现，支持高并发和大规模数据处理。
"""

from flask import jsonify, request, g
from typing import Dict, Any, Optional, List, Tuple, Callable, Union, TypeVar
from functools import wraps
from datetime import datetime, timedelta
from collections import OrderedDict
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import logging
import time
import os
import yaml
import uuid
import hashlib
import json
import re

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')


# ============================================================================
# 配置加载器
# ============================================================================

class PerformanceAPIConfig:
    """性能API配置管理器（单例模式）
    
    支持功能：
    - YAML配置文件加载
    - 环境变量覆盖
    - 动态配置刷新
    - 配置验证
    """
    
    _instance = None
    _lock = threading.Lock()
    _config = None
    _last_load_time = None
    _config_ttl = 300  # 配置缓存TTL（秒）
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """加载配置文件"""
        config_paths = [
            os.path.join(os.path.dirname(__file__), '../../../config/performance_api.yaml'),
            '/root/VectorSphere/VectorSphere-intelligent-platform/config/performance_api.yaml',
            'config/performance_api.yaml'
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self._config = yaml.safe_load(f)
                    logger.info(f"Performance API config loaded from: {config_path}")
                    self._last_load_time = time.time()
                    self._apply_env_overrides()
                    return
                except Exception as e:
                    logger.warning(f"Failed to load config from {config_path}: {e}")
        
        # 使用默认配置
        logger.warning("Using default performance API configuration")
        self._config = self._get_default_config()
        self._last_load_time = time.time()
    
    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        env_mappings = {
            'PERF_CPU_WARNING': 'resource_thresholds.cpu.warning',
            'PERF_CPU_CRITICAL': 'resource_thresholds.cpu.critical',
            'PERF_MEMORY_WARNING': 'resource_thresholds.memory.warning',
            'PERF_MEMORY_CRITICAL': 'resource_thresholds.memory.critical',
            'PERF_DISK_WARNING': 'resource_thresholds.disk.warning',
            'PERF_DISK_CRITICAL': 'resource_thresholds.disk.critical',
            'PERF_ASYNC_MAX_WORKERS': 'async_processor.workers.max_count',
            'PERF_ASYNC_QUEUE_SIZE': 'async_processor.queue.max_size',
            'PERF_DB_POOL_WARNING': 'database_pool.utilization.warning',
            'PERF_DB_POOL_CRITICAL': 'database_pool.utilization.critical',
            'PERF_PAGINATION_MAX_LIMIT': 'pagination.max_limit',
            'PERF_RATE_LIMIT_REQUESTS': 'rate_limiting.requests_per_minute',
            'PERF_CACHE_TTL': 'caching.default_ttl',
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    self._set_nested(config_path, self._parse_env_value(value))
                    logger.debug(f"Config override from env: {env_var} -> {config_path}")
                except Exception as e:
                    logger.warning(f"Failed to apply env override {env_var}: {e}")
    
    def _parse_env_value(self, value: str) -> Any:
        """解析环境变量值"""
        # 尝试解析为数字
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
        
        # 尝试解析为布尔
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # 尝试解析为JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        return value
    
    def _set_nested(self, path: str, value: Any):
        """设置嵌套配置值"""
        keys = path.split('.')
        current = self._config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'resource_thresholds': {
                'cpu': {'warning': 80, 'critical': 90},
                'memory': {'warning': 75, 'critical': 85},
                'disk': {'warning': 80, 'critical': 90},
                'gpu_utilization': {'warning': 80, 'critical': 95},
                'gpu_temperature': {'good': 60, 'normal': 75, 'warning': 85, 'critical': 85}
            },
            'async_processor': {
                'queue': {'warning_utilization': 90, 'max_size': 1000},
                'workers': {'max_count': 10, 'health_threshold': 0.5},
                'tasks': {'failure_rate_warning': 10, 'default_timeout': 300}
            },
            'database_pool': {
                'utilization': {'warning': 70, 'critical': 90},
                'overflow': {'warning_ratio': 0.5},
                'latency': {'excellent': 50, 'good': 100, 'moderate': 500}
            },
            'health_check': {
                'timeout': {'default': 5, 'database': 5, 'connection': 10},
                'data_freshness': {'stale_threshold': 60},
                'latency_test': {'sample_count': 3}
            },
            'pagination': {
                'default_limit': 100, 
                'max_limit': 1000, 
                'default_offset': 0
            },
            'time_ranges': {
                '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '6h': 21600, '24h': 86400,
                '7d': 604800, '30d': 2592000,
                'default': '1h'
            },
            'alerts': {
                'default_duration_minutes': 60, 
                'max_recent_alerts': 10, 
                'max_alerts_per_page': 100,
                'auto_resolve_timeout': 3600
            },
            'cleanup': {
                'shutdown_timeout': 10, 
                'tasks': {'default_max_age_days': 30, 'batch_size': 1000}
            },
            'rate_limiting': {
                'enabled': True,
                'requests_per_minute': 1000,
                'burst_size': 100,
                'window_seconds': 60
            },
            'caching': {
                'enabled': True,
                'default_ttl': 60,
                'max_entries': 10000,
                'metrics_ttl': 30,
                'config_ttl': 300
            },
            'logging': {
                'request_logging': True,
                'slow_request_threshold_ms': 1000,
                'log_request_body': False,
                'log_response_body': False
            },
            'validation': {
                'max_batch_size': 1000,
                'max_string_length': 10000,
                'allowed_metric_types': ['system', 'gpu', 'training', 'database', 'async_processor', 'custom'],
                'allowed_alert_levels': ['low', 'medium', 'high', 'critical']
            },
            'default_alert_rules': [],
            'initialization': {
                'auto_start_monitoring': True,
                'setup_default_alerts': True,
                'init_async_processor': True,
                'init_database_pool': True,
                'register_cleanup': True
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点分隔的路径"""
        # 检查是否需要刷新配置
        self._check_refresh()
        
        if not self._config:
            return default
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def _check_refresh(self):
        """检查是否需要刷新配置"""
        if self._last_load_time and (time.time() - self._last_load_time) > self._config_ttl:
            with self._lock:
                if (time.time() - self._last_load_time) > self._config_ttl:
                    self._load_config()
    
    def get_threshold(self, resource: str, level: str) -> float:
        """获取资源阈值"""
        return self.get(f'resource_thresholds.{resource}.{level}', 0)
    
    def get_time_range_seconds(self, time_range: str) -> int:
        """获取时间范围对应的秒数"""
        default_range = self.get('time_ranges.default', '1h')
        return self.get(f'time_ranges.{time_range}', self.get(f'time_ranges.{default_range}', 3600))
    
    def reload(self):
        """重新加载配置"""
        with self._lock:
            self._load_config()
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        self._check_refresh()
        return self._config.copy() if self._config else {}
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置有效性"""
        errors = []
        
        # 验证阈值
        for resource in ['cpu', 'memory', 'disk']:
            warning = self.get_threshold(resource, 'warning')
            critical = self.get_threshold(resource, 'critical')
            if warning >= critical:
                errors.append(f"{resource}: warning threshold ({warning}) should be less than critical ({critical})")
        
        # 验证分页配置
        max_limit = self.get('pagination.max_limit', 1000)
        default_limit = self.get('pagination.default_limit', 100)
        if default_limit > max_limit:
            errors.append(f"pagination: default_limit ({default_limit}) exceeds max_limit ({max_limit})")
        
        return len(errors) == 0, errors


# 全局配置实例
_config = None

def get_config() -> PerformanceAPIConfig:
    """获取配置实例"""
    global _config
    if _config is None:
        _config = PerformanceAPIConfig()
    return _config


# ============================================================================
# 响应构建器
# ============================================================================

class APIResponse:
    """API响应构建器
    
    提供标准化的响应格式，支持：
    - 成功/错误响应
    - 分页响应
    - 健康检查响应
    - 批量操作响应
    """
    
    @staticmethod
    def success(data: Any = None, message: str = None, status_code: int = 200) -> Tuple:
        """构建成功响应"""
        response = {'success': True}
        if data is not None:
            response['data'] = data
        if message:
            response['message'] = message
        response['timestamp'] = datetime.utcnow().isoformat()
        return jsonify(response), status_code
    
    @staticmethod
    def created(data: Any = None, message: str = None, resource_id: str = None) -> Tuple:
        """构建201创建成功响应"""
        response = {'success': True}
        if data is not None:
            response['data'] = data
        if message:
            response['message'] = message
        if resource_id:
            response['resource_id'] = resource_id
        response['timestamp'] = datetime.utcnow().isoformat()
        return jsonify(response), 201
    
    @staticmethod
    def accepted(task_id: str = None, message: str = None, data: Any = None) -> Tuple:
        """构建202异步任务接受响应"""
        response = {
            'success': True,
            'message': message or 'Request accepted for processing'
        }
        if task_id:
            response['task_id'] = task_id
        if data:
            response['data'] = data
        response['timestamp'] = datetime.utcnow().isoformat()
        return jsonify(response), 202
    
    @staticmethod
    def no_content() -> Tuple:
        """构建204无内容响应"""
        return '', 204
    
    @staticmethod
    def error(
        error: str,
        error_code: str = 'ERROR',
        status_code: int = 500,
        data: Dict = None,
        hint: str = None,
        request_id: str = None
    ) -> Tuple:
        """构建错误响应"""
        response = {
            'success': False,
            'error': error,
            'error_code': error_code,
            'timestamp': datetime.utcnow().isoformat()
        }
        if data:
            response['data'] = data
        if hint:
            response['hint'] = hint
        if request_id:
            response['request_id'] = request_id
        return jsonify(response), status_code
    
    @staticmethod
    def validation_error(errors: Union[str, List[str], Dict[str, str]], hint: str = None) -> Tuple:
        """构建验证错误响应"""
        if isinstance(errors, str):
            errors = [errors]
        return APIResponse.error(
            error='Validation failed',
            error_code='VALIDATION_ERROR',
            status_code=400,
            data={'validation_errors': errors},
            hint=hint
        )
    
    @staticmethod
    def not_found(resource: str, resource_id: str = None, hint: str = None) -> Tuple:
        """构建404响应"""
        error_msg = f'{resource} not found'
        if resource_id:
            error_msg = f'{resource} {resource_id} not found'
        return APIResponse.error(
            error=error_msg,
            error_code=f'{resource.upper().replace(" ", "_")}_NOT_FOUND',
            status_code=404,
            hint=hint
        )
    
    @staticmethod
    def conflict(error: str, error_code: str = 'CONFLICT', data: Dict = None) -> Tuple:
        """构建409冲突响应"""
        return APIResponse.error(
            error=error,
            error_code=error_code,
            status_code=409,
            data=data
        )
    
    @staticmethod
    def service_unavailable(service: str, hint: str = None) -> Tuple:
        """构建503响应"""
        return APIResponse.error(
            error=f'{service} is not available',
            error_code=f'{service.upper().replace(" ", "_")}_UNAVAILABLE',
            status_code=503,
            hint=hint
        )
    
    @staticmethod
    def too_many_requests(retry_after: int = None, hint: str = None) -> Tuple:
        """构建429限流响应"""
        response = {
            'success': False,
            'error': 'Too many requests',
            'error_code': 'RATE_LIMIT_EXCEEDED',
            'timestamp': datetime.utcnow().isoformat()
        }
        if hint:
            response['hint'] = hint
        if retry_after:
            response['retry_after'] = retry_after
        
        resp = jsonify(response)
        if retry_after:
            resp.headers['Retry-After'] = str(retry_after)
        return resp, 429
    
    @staticmethod
    def bad_request(error: str, error_code: str = 'BAD_REQUEST', data: Dict = None) -> Tuple:
        """构建400响应"""
        return APIResponse.error(
            error=error,
            error_code=error_code,
            status_code=400,
            data=data
        )
    
    @staticmethod
    def paginated(
        items: List,
        total: int,
        page: int = 1,
        page_size: int = 100,
        extra_data: Dict = None
    ) -> Tuple:
        """构建分页响应"""
        total_pages = (total + page_size - 1) // page_size if page_size > 0 else 0
        has_next = page < total_pages
        has_prev = page > 1
        
        response = {
            'success': True,
            'data': {
                'items': items,
                'pagination': {
                    'total': total,
                    'page': page,
                    'page_size': page_size,
                    'total_pages': total_pages,
                    'has_next': has_next,
                    'has_prev': has_prev
                }
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if extra_data:
            response['data'].update(extra_data)
        
        return jsonify(response), 200
    
    @staticmethod
    def batch_result(
        total: int,
        success_count: int,
        failed_count: int,
        results: List[Dict] = None,
        errors: List[Dict] = None
    ) -> Tuple:
        """构建批量操作响应"""
        response = {
            'success': failed_count == 0,
            'data': {
                'total': total,
                'success_count': success_count,
                'failed_count': failed_count,
                'success_rate': round(success_count / total * 100, 2) if total > 0 else 0
            },
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if results:
            response['data']['results'] = results
        if errors:
            response['data']['errors'] = errors
        
        status_code = 200 if failed_count == 0 else 207  # 207 Multi-Status
        return jsonify(response), status_code
    
    @staticmethod
    def health_response(
        is_healthy: bool,
        status: str,
        message: str,
        checks: Dict = None,
        issues: List = None,
        recommendations: List = None,
        duration_ms: float = None,
        version: str = None
    ) -> Tuple:
        """构建健康检查响应"""
        data = {
            'is_healthy': is_healthy,
            'status': status,
            'message': message,
            'timestamp': time.time(),
            'timestamp_iso': timestamp_to_iso(time.time())
        }
        
        if version:
            data['version'] = version
        
        if duration_ms is not None:
            data['check_duration_ms'] = round(duration_ms, 2)
        
        if checks:
            # 计算摘要
            passed = sum(1 for c in checks.values() if c.get('passed', False))
            warnings = sum(1 for c in checks.values() if c.get('status') == 'warning')
            failed = len(checks) - passed
            
            data['summary'] = {
                'total_checks': len(checks),
                'passed': passed,
                'warnings': warnings,
                'failed': failed
            }
            data['checks'] = checks
        
        if issues:
            data['issues'] = issues
        if recommendations:
            data['recommendations'] = list(set(recommendations))
        
        http_status = 200 if is_healthy else 503
        return jsonify({'success': True, 'data': data}), http_status


# ============================================================================
# 时间工具
# ============================================================================

def timestamp_to_iso(timestamp: Optional[float]) -> Optional[str]:
    """将Unix时间戳转换为ISO格式字符串"""
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp).isoformat()
    except (ValueError, OSError, OverflowError):
        return None


def iso_to_timestamp(iso_string: str) -> Optional[float]:
    """将ISO格式字符串转换为Unix时间戳"""
    if not iso_string:
        return None
    try:
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return dt.timestamp()
    except (ValueError, AttributeError):
        return None


def format_duration(seconds: float) -> str:
    """格式化持续时间为人类可读格式"""
    if seconds is None:
        return 'N/A'
    if seconds < 0:
        return 'Invalid'
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
    else:
        days = int(seconds / 86400)
        hours = int((seconds % 86400) / 3600)
        return f"{days}d {hours}h"


def parse_time_range(time_range: str) -> int:
    """解析时间范围字符串为秒数
    
    支持格式：
    - 预定义：5m, 15m, 30m, 1h, 6h, 24h, 7d, 30d
    - 自定义：10m, 2h, 3d
    """
    # 首先尝试预定义值
    config = get_config()
    predefined = config.get(f'time_ranges.{time_range}')
    if predefined:
        return predefined
    
    # 解析自定义格式
    match = re.match(r'^(\d+)([smhd])$', time_range.lower())
    if match:
        value = int(match.group(1))
        unit = match.group(2)
        multipliers = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
        return value * multipliers.get(unit, 1)
    
    # 默认返回1小时
    return config.get_time_range_seconds('default')


def get_time_bucket(timestamp: datetime, interval: str) -> str:
    """获取时间桶标识（用于聚合）"""
    if interval in ('1m', '1min', 'minute'):
        return timestamp.strftime('%Y-%m-%d %H:%M:00')
    elif interval in ('5m', '5min'):
        minute = (timestamp.minute // 5) * 5
        return timestamp.strftime(f'%Y-%m-%d %H:{minute:02d}:00')
    elif interval in ('15m', '15min'):
        minute = (timestamp.minute // 15) * 15
        return timestamp.strftime(f'%Y-%m-%d %H:{minute:02d}:00')
    elif interval in ('1h', '1hour', 'hour', 'hourly'):
        return timestamp.strftime('%Y-%m-%d %H:00:00')
    elif interval in ('1d', '1day', 'day', 'daily'):
        return timestamp.strftime('%Y-%m-%d 00:00:00')
    else:
        return timestamp.isoformat()


# ============================================================================
# 状态评估工具
# ============================================================================

def get_resource_status(
    value: float,
    resource_type: str = None,
    warning_threshold: float = None,
    critical_threshold: float = None
) -> str:
    """获取资源状态
    
    Args:
        value: 当前值
        resource_type: 资源类型（cpu/memory/disk/gpu_utilization）
        warning_threshold: 自定义警告阈值
        critical_threshold: 自定义严重阈值
    
    Returns:
        状态字符串：good/warning/critical
    """
    config = get_config()
    
    # 使用自定义阈值或从配置获取
    if warning_threshold is None and resource_type:
        warning_threshold = config.get_threshold(resource_type, 'warning')
    if critical_threshold is None and resource_type:
        critical_threshold = config.get_threshold(resource_type, 'critical')
    
    # 默认阈值
    warning_threshold = warning_threshold or 80
    critical_threshold = critical_threshold or 90
    
    if value >= critical_threshold:
        return 'critical'
    elif value >= warning_threshold:
        return 'warning'
    return 'good'


def get_temperature_status(temperature: float) -> str:
    """获取温度状态"""
    config = get_config()
    
    critical = config.get_threshold('gpu_temperature', 'critical') or 85
    warning = config.get_threshold('gpu_temperature', 'warning') or 85
    normal = config.get_threshold('gpu_temperature', 'normal') or 75
    
    if temperature >= critical:
        return 'critical'
    elif temperature >= warning:
        return 'warning'
    elif temperature >= normal:
        return 'normal'
    return 'good'


def assess_utilization_status(
    utilization_percent: float,
    overflow_percent: float = 0,
    resource_name: str = 'resource'
) -> Dict[str, Any]:
    """评估使用率状态"""
    config = get_config()
    
    warning_threshold = config.get('database_pool.utilization.warning', 70)
    critical_threshold = config.get('database_pool.utilization.critical', 90)
    
    if utilization_percent >= critical_threshold:
        level = 'critical'
        status = 'overloaded'
        message = f'{resource_name} is overloaded, consider increasing capacity'
    elif utilization_percent >= warning_threshold or overflow_percent > 0:
        level = 'warning'
        status = 'high_usage'
        message = f'{resource_name} usage is high, monitor closely'
    elif utilization_percent >= 50:
        level = 'normal'
        status = 'moderate_usage'
        message = f'{resource_name} usage is moderate'
    else:
        level = 'good'
        status = 'healthy'
        message = f'{resource_name} is operating normally'
    
    return {
        'level': level,
        'status': status,
        'message': message,
        'utilization_percent': utilization_percent,
        'overflow_percent': overflow_percent
    }


def calculate_health_score(checks: Dict[str, Dict]) -> float:
    """计算健康分数（0-100）"""
    if not checks:
        return 100.0
    
    total_weight = 0
    weighted_score = 0
    
    # 权重配置
    weights = {
        'cpu': 1.0,
        'memory': 1.0,
        'disk': 0.8,
        'database': 1.5,
        'async_processor': 1.2,
        'gpu': 0.8,
        'network': 0.5
    }
    
    for check_name, check_result in checks.items():
        # 获取权重
        weight = 1.0
        for key, w in weights.items():
            if key in check_name.lower():
                weight = w
                break
        
        total_weight += weight
        
        # 计算分数
        if check_result.get('passed', False):
            if check_result.get('status') == 'warning':
                weighted_score += weight * 70
            else:
                weighted_score += weight * 100
        else:
            weighted_score += weight * 0
    
    return round(weighted_score / total_weight, 2) if total_weight > 0 else 100.0


# ============================================================================
# 健康检查工具
# ============================================================================

class HealthCheckResult:
    """健康检查结果构建器"""
    
    @staticmethod
    def passed(message: str, details: Dict = None, latency_ms: float = None) -> Dict[str, Any]:
        """构建通过的检查结果"""
        result = {
            'passed': True,
            'status': 'healthy',
            'message': message
        }
        if details:
            result['details'] = details
        if latency_ms is not None:
            result['latency_ms'] = round(latency_ms, 2)
        return result
    
    @staticmethod
    def warning(message: str, recommendation: str = None, details: Dict = None, latency_ms: float = None) -> Dict[str, Any]:
        """构建警告的检查结果"""
        result = {
            'passed': True,
            'status': 'warning',
            'message': message
        }
        if recommendation:
            result['recommendation'] = recommendation
        if details:
            result['details'] = details
        if latency_ms is not None:
            result['latency_ms'] = round(latency_ms, 2)
        return result
    
    @staticmethod
    def failed(message: str, recommendation: str = None, details: Dict = None, latency_ms: float = None) -> Dict[str, Any]:
        """构建失败的检查结果"""
        result = {
            'passed': False,
            'status': 'failed',
            'message': message
        }
        if recommendation:
            result['recommendation'] = recommendation
        if details:
            result['details'] = details
        if latency_ms is not None:
            result['latency_ms'] = round(latency_ms, 2)
        return result
    
    @staticmethod
    def error(message: str, recommendation: str = None, error_type: str = None) -> Dict[str, Any]:
        """构建错误的检查结果"""
        result = {
            'passed': False,
            'status': 'error',
            'message': message
        }
        if recommendation:
            result['recommendation'] = recommendation
        if error_type:
            result['error_type'] = error_type
        return result
    
    @staticmethod
    def timeout(timeout_seconds: float, recommendation: str = None) -> Dict[str, Any]:
        """构建超时的检查结果"""
        return {
            'passed': False,
            'status': 'timeout',
            'message': f'Health check timed out after {timeout_seconds}s',
            'recommendation': recommendation or 'Check service responsiveness'
        }


def calculate_health_summary(checks: Dict[str, Dict]) -> Tuple[bool, str, str]:
    """计算健康检查总体状态
    
    Args:
        checks: 检查结果字典
        
    Returns:
        (is_healthy, status, message)
    """
    passed_count = sum(1 for c in checks.values() if c.get('passed', False))
    total_count = len(checks)
    warning_count = sum(1 for c in checks.values() if c.get('status') == 'warning')
    failed_count = total_count - passed_count
    
    if failed_count == 0 and warning_count == 0:
        return True, 'healthy', 'All checks passed'
    elif failed_count == 0 and warning_count > 0:
        return True, 'degraded', f'{warning_count} component(s) have warnings'
    elif failed_count <= 1:
        return False, 'degraded', f'{failed_count} component(s) failed, system partially operational'
    else:
        return False, 'unhealthy', f'{failed_count} component(s) failed'


# ============================================================================
# 请求参数工具
# ============================================================================

def get_bool_param(request_obj, param_name: str, default: bool = True) -> bool:
    """获取布尔类型的请求参数"""
    value = request_obj.args.get(param_name, str(default).lower())
    return value.lower() in ('true', '1', 'yes', 'on')


def get_int_param(request_obj, param_name: str, default: int, min_val: int = None, max_val: int = None) -> int:
    """获取整数类型的请求参数，带范围限制"""
    try:
        value = request_obj.args.get(param_name, default, type=int)
    except (ValueError, TypeError):
        value = default
    
    if min_val is not None and value < min_val:
        value = min_val
    if max_val is not None and value > max_val:
        value = max_val
    
    return value


def get_float_param(request_obj, param_name: str, default: float, min_val: float = None, max_val: float = None) -> float:
    """获取浮点数类型的请求参数，带范围限制"""
    try:
        value = request_obj.args.get(param_name, default, type=float)
    except (ValueError, TypeError):
        value = default
    
    if min_val is not None and value < min_val:
        value = min_val
    if max_val is not None and value > max_val:
        value = max_val
    
    return value


def get_list_param(request_obj, param_name: str, separator: str = ',', lowercase: bool = True) -> List[str]:
    """获取列表类型的请求参数"""
    value = request_obj.args.get(param_name, '')
    if not value:
        return []
    items = [item.strip() for item in value.split(separator) if item.strip()]
    if lowercase:
        items = [item.lower() for item in items]
    return items


def get_datetime_param(request_obj, param_name: str, default: datetime = None) -> Optional[datetime]:
    """获取日期时间类型的请求参数"""
    value = request_obj.args.get(param_name)
    if not value:
        return default
    
    # 尝试多种格式
    formats = [
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d'
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(value.replace('+00:00', 'Z'), fmt)
        except ValueError:
            continue
    
    return default


def get_enum_param(request_obj, param_name: str, enum_class: type, default: Any = None) -> Optional[Any]:
    """获取枚举类型的请求参数"""
    value = request_obj.args.get(param_name)
    if not value:
        return default
    
    try:
        return enum_class(value.lower())
    except (ValueError, KeyError):
        return default


def get_pagination_params(request_obj, default_limit: int = None, max_limit: int = None) -> Tuple[int, int, int]:
    """获取分页参数
    
    Returns:
        (page, page_size, offset)
    """
    config = get_config()
    default_limit = default_limit or config.get('pagination.default_limit', 100)
    max_limit = max_limit or config.get('pagination.max_limit', 1000)
    
    page = get_int_param(request_obj, 'page', 1, min_val=1)
    page_size = get_int_param(request_obj, 'page_size', default_limit, min_val=1, max_val=max_limit)
    
    # 也支持 limit/offset 模式
    if 'limit' in request_obj.args:
        page_size = get_int_param(request_obj, 'limit', default_limit, min_val=1, max_val=max_limit)
    if 'offset' in request_obj.args:
        offset = get_int_param(request_obj, 'offset', 0, min_val=0)
        page = (offset // page_size) + 1 if page_size > 0 else 1
    else:
        offset = (page - 1) * page_size
    
    return page, page_size, offset


def get_sort_params(request_obj, allowed_fields: List[str] = None, default_field: str = 'created_at') -> Tuple[str, bool]:
    """获取排序参数
    
    Returns:
        (sort_field, descending)
    """
    sort_field = request_obj.args.get('sort_by', default_field)
    sort_order = request_obj.args.get('sort_order', 'desc').lower()
    
    # 验证排序字段
    if allowed_fields and sort_field not in allowed_fields:
        sort_field = default_field
    
    descending = sort_order in ('desc', 'descending', '-1')
    
    return sort_field, descending


# ============================================================================
# 数据验证工具
# ============================================================================

class ValidationError(Exception):
    """验证错误"""
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(message)


def validate_required(data: Dict, required_fields: List[str]) -> List[str]:
    """验证必填字段"""
    errors = []
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.append(f"Field '{field}' is required")
    return errors


def validate_string_length(value: str, field_name: str, min_length: int = None, max_length: int = None) -> Optional[str]:
    """验证字符串长度"""
    if value is None:
        return None
    
    if min_length is not None and len(value) < min_length:
        return f"'{field_name}' must be at least {min_length} characters"
    
    config = get_config()
    max_length = max_length or config.get('validation.max_string_length', 10000)
    if len(value) > max_length:
        return f"'{field_name}' exceeds maximum length of {max_length} characters"
    
    return None


def validate_enum_value(value: str, allowed_values: List[str], field_name: str) -> Optional[str]:
    """验证枚举值"""
    if value is None:
        return None
    
    if value.lower() not in [v.lower() for v in allowed_values]:
        return f"'{field_name}' must be one of: {', '.join(allowed_values)}"
    
    return None


def validate_numeric_range(value: float, field_name: str, min_val: float = None, max_val: float = None) -> Optional[str]:
    """验证数值范围"""
    if value is None:
        return None
    
    if min_val is not None and value < min_val:
        return f"'{field_name}' must be at least {min_val}"
    
    if max_val is not None and value > max_val:
        return f"'{field_name}' must be at most {max_val}"
    
    return None


def validate_uuid(value: str, field_name: str) -> Optional[str]:
    """验证UUID格式"""
    if value is None:
        return None
    
    try:
        uuid.UUID(value)
        return None
    except ValueError:
        return f"'{field_name}' must be a valid UUID"


def validate_email(value: str, field_name: str = 'email') -> Optional[str]:
    """验证邮箱格式"""
    if value is None:
        return None
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, value):
        return f"'{field_name}' must be a valid email address"
    
    return None


def validate_request_body(
    data: Dict,
    required_fields: List[str] = None,
    optional_fields: List[str] = None,
    validators: Dict[str, Callable] = None
) -> Tuple[bool, List[str]]:
    """验证请求体
    
    Args:
        data: 请求数据
        required_fields: 必填字段列表
        optional_fields: 可选字段列表
        validators: 字段验证器字典
    
    Returns:
        (is_valid, errors)
    """
    errors = []
    
    # 验证必填字段
    if required_fields:
        errors.extend(validate_required(data, required_fields))
    
    # 运行自定义验证器
    if validators:
        for field, validator in validators.items():
            if field in data:
                try:
                    error = validator(data[field], field)
                    if error:
                        errors.append(error)
                except Exception as e:
                    errors.append(f"Validation error for '{field}': {str(e)}")
    
    return len(errors) == 0, errors


# ============================================================================
# 缓存工具
# ============================================================================

class LRUCache:
    """线程安全的LRU缓存"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 60):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict = OrderedDict()
        self._expiry: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._stats = {'hits': 0, 'misses': 0, 'evictions': 0}
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            # 检查过期
            if self._expiry.get(key, 0) < time.time():
                self._remove(key)
                self._stats['misses'] += 1
                return None
            
            # 移动到末尾（最近使用）
            self._cache.move_to_end(key)
            self._stats['hits'] += 1
            return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: int = None):
        """设置缓存值"""
        ttl = ttl or self.default_ttl
        
        with self._lock:
            # 如果键存在，删除旧值
            if key in self._cache:
                self._cache.move_to_end(key)
            
            self._cache[key] = value
            self._expiry[key] = time.time() + ttl
            
            # 驱逐超出容量的条目
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                self._remove(oldest_key)
                self._stats['evictions'] += 1
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self._lock:
            return self._remove(key)
    
    def _remove(self, key: str) -> bool:
        """内部删除方法"""
        if key in self._cache:
            del self._cache[key]
            self._expiry.pop(key, None)
            return True
        return False
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._expiry.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests * 100 if total_requests > 0 else 0
            
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'evictions': self._stats['evictions'],
                'hit_rate': round(hit_rate, 2)
            }
    
    def cleanup_expired(self) -> int:
        """清理过期条目"""
        cleaned = 0
        current_time = time.time()
        
        with self._lock:
            expired_keys = [
                key for key, expiry in self._expiry.items()
                if expiry < current_time
            ]
            
            for key in expired_keys:
                self._remove(key)
                cleaned += 1
        
        return cleaned


# 全局缓存实例
_api_cache = None
_cache_lock = threading.Lock()

def get_api_cache() -> LRUCache:
    """获取API缓存实例"""
    global _api_cache
    if _api_cache is None:
        with _cache_lock:
            if _api_cache is None:
                config = get_config()
                _api_cache = LRUCache(
                    max_size=config.get('caching.max_entries', 10000),
                    default_ttl=config.get('caching.default_ttl', 60)
                )
    return _api_cache


def cached(ttl: int = None, key_prefix: str = None):
    """缓存装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = get_config()
            if not config.get('caching.enabled', True):
                return func(*args, **kwargs)
            
            # 生成缓存键
            cache_key = f"{key_prefix or func.__name__}:{hashlib.md5(str(args).encode() + str(kwargs).encode()).hexdigest()[:16]}"
            
            cache = get_api_cache()
            cached_value = cache.get(cache_key)
            
            if cached_value is not None:
                return cached_value
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator


# ============================================================================
# 限流工具
# ============================================================================

class RateLimiter:
    """滑动窗口限流器"""
    
    def __init__(self, requests_per_minute: int = 1000, burst_size: int = 100):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.window_seconds = 60
        self._requests: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
    
    def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """检查是否允许请求
        
        Returns:
            (allowed, rate_info)
        """
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        with self._lock:
            # 获取或初始化客户端请求列表
            if client_id not in self._requests:
                self._requests[client_id] = []
            
            # 清理过期请求
            self._requests[client_id] = [
                t for t in self._requests[client_id]
                if t > window_start
            ]
            
            request_count = len(self._requests[client_id])
            
            # 检查是否超过限制
            if request_count >= self.requests_per_minute:
                # 计算重试时间
                oldest_request = min(self._requests[client_id])
                retry_after = int(oldest_request + self.window_seconds - current_time) + 1
                
                return False, {
                    'allowed': False,
                    'current_requests': request_count,
                    'limit': self.requests_per_minute,
                    'window_seconds': self.window_seconds,
                    'retry_after': max(retry_after, 1)
                }
            
            # 记录请求
            self._requests[client_id].append(current_time)
            
            return True, {
                'allowed': True,
                'current_requests': request_count + 1,
                'limit': self.requests_per_minute,
                'remaining': self.requests_per_minute - request_count - 1,
                'window_seconds': self.window_seconds
            }
    
    def get_client_id(self) -> str:
        """从请求获取客户端标识"""
        # 优先使用 API Key
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return f"api:{api_key[:16]}"
        
        # 其次使用用户ID
        user_id = getattr(g, 'user_id', None)
        if user_id:
            return f"user:{user_id}"
        
        # 最后使用IP
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        if client_ip:
            return f"ip:{client_ip.split(',')[0].strip()}"
        
        return 'unknown'
    
    def cleanup(self):
        """清理过期数据"""
        current_time = time.time()
        window_start = current_time - self.window_seconds
        
        with self._lock:
            empty_clients = []
            for client_id, requests in self._requests.items():
                self._requests[client_id] = [t for t in requests if t > window_start]
                if not self._requests[client_id]:
                    empty_clients.append(client_id)
            
            for client_id in empty_clients:
                del self._requests[client_id]


# 全局限流器
_rate_limiter = None
_rate_limiter_lock = threading.Lock()

def get_rate_limiter() -> RateLimiter:
    """获取限流器实例"""
    global _rate_limiter
    if _rate_limiter is None:
        with _rate_limiter_lock:
            if _rate_limiter is None:
                config = get_config()
                _rate_limiter = RateLimiter(
                    requests_per_minute=config.get('rate_limiting.requests_per_minute', 1000),
                    burst_size=config.get('rate_limiting.burst_size', 100)
                )
    return _rate_limiter


def rate_limit(requests_per_minute: int = None):
    """限流装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            config = get_config()
            if not config.get('rate_limiting.enabled', True):
                return func(*args, **kwargs)
            
            limiter = get_rate_limiter()
            client_id = limiter.get_client_id()
            
            allowed, rate_info = limiter.is_allowed(client_id)
            
            if not allowed:
                return APIResponse.too_many_requests(
                    retry_after=rate_info.get('retry_after', 60),
                    hint=f"Rate limit exceeded: {rate_info.get('current_requests')}/{rate_info.get('limit')} requests"
                )
            
            response = func(*args, **kwargs)
            
            # 添加限流头
            if isinstance(response, tuple) and len(response) >= 1:
                resp = response[0]
                if hasattr(resp, 'headers'):
                    resp.headers['X-RateLimit-Limit'] = str(rate_info.get('limit', 0))
                    resp.headers['X-RateLimit-Remaining'] = str(rate_info.get('remaining', 0))
            
            return response
        return wrapper
    return decorator


# ============================================================================
# 装饰器工具
# ============================================================================

def handle_api_errors(func: Callable) -> Callable:
    """API错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        request_id = str(uuid.uuid4())[:8]
        
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            logger.warning(f"[{request_id}] Validation error in {func.__name__}: {e.message}")
            return APIResponse.validation_error(e.message)
        except Exception as e:
            # 尝试导入 PerformanceError
            try:
                from backend.modules.performance.performance_errors import PerformanceError
                if isinstance(e, PerformanceError):
                    logger.error(f"[{request_id}] Performance error in {func.__name__}: {e}")
                    return APIResponse.error(
                        error=str(e),
                        error_code=e.error_code if hasattr(e, 'error_code') else 'PERFORMANCE_ERROR',
                        status_code=500,
                        request_id=request_id
                    )
            except ImportError:
                pass
            
            logger.error(f"[{request_id}] Unexpected error in {func.__name__}: {e}", exc_info=True)
            return APIResponse.error(
                error=f'Internal error: {str(e)}',
                error_code='INTERNAL_ERROR',
                status_code=500,
                request_id=request_id
            )
    
    return wrapper


def timed_execution(func: Callable) -> Callable:
    """执行时间记录装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = (time.time() - start_time) * 1000
        
        config = get_config()
        slow_threshold = config.get('logging.slow_request_threshold_ms', 1000)
        
        if duration > slow_threshold:
            logger.warning(f"Slow request: {func.__name__} took {duration:.2f}ms")
        else:
            logger.debug(f"{func.__name__} executed in {duration:.2f}ms")
        
        return result
    return wrapper


def log_request(include_body: bool = False, include_response: bool = False):
    """请求日志装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            request_id = str(uuid.uuid4())[:8]
            
            # 记录请求
            log_data = {
                'request_id': request_id,
                'method': request.method,
                'path': request.path,
                'client_ip': request.headers.get('X-Forwarded-For', request.remote_addr)
            }
            
            if include_body and request.is_json:
                log_data['body'] = request.get_json(silent=True)
            
            logger.info(f"[{request_id}] Request: {log_data}")
            
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            
            # 记录响应
            response_log = {
                'request_id': request_id,
                'duration_ms': round(duration, 2),
                'status_code': result[1] if isinstance(result, tuple) and len(result) > 1 else 200
            }
            
            logger.info(f"[{request_id}] Response: {response_log}")
            
            return result
        return wrapper
    return decorator


def require_json(func: Callable) -> Callable:
    """要求JSON请求体装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return APIResponse.bad_request(
                error='Request body must be JSON',
                error_code='INVALID_CONTENT_TYPE'
            )
        return func(*args, **kwargs)
    return wrapper


def validate_params(required: List[str] = None, validators: Dict[str, Callable] = None):
    """参数验证装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            data = request.get_json(silent=True) or {}
            
            is_valid, errors = validate_request_body(
                data=data,
                required_fields=required,
                validators=validators
            )
            
            if not is_valid:
                return APIResponse.validation_error(errors)
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# 指标格式化工具
# ============================================================================

def format_system_metrics(system_metrics, config: PerformanceAPIConfig = None) -> Dict[str, Any]:
    """格式化系统指标"""
    config = config or get_config()
    
    cpu_percent = getattr(system_metrics, 'cpu_percent', 0) or 0
    memory_percent = getattr(system_metrics, 'memory_percent', 0) or 0
    disk_percent = getattr(system_metrics, 'disk_percent', 0) or 0
    
    return {
        'cpu': {
            'percent': round(cpu_percent, 2),
            'status': get_resource_status(cpu_percent, 'cpu')
        },
        'memory': {
            'percent': round(memory_percent, 2),
            'used_gb': round(getattr(system_metrics, 'memory_used_gb', 0) or 0, 2),
            'total_gb': round(getattr(system_metrics, 'memory_total_gb', 0) or 0, 2),
            'available_gb': round(
                (getattr(system_metrics, 'memory_total_gb', 0) or 0) - (getattr(system_metrics, 'memory_used_gb', 0) or 0), 2
            ),
            'status': get_resource_status(memory_percent, 'memory')
        },
        'disk': {
            'percent': round(disk_percent, 2),
            'io_read_mb': round(getattr(system_metrics, 'disk_io_read_mb', 0) or 0, 2),
            'io_write_mb': round(getattr(system_metrics, 'disk_io_write_mb', 0) or 0, 2),
            'status': get_resource_status(disk_percent, 'disk')
        },
        'network': {
            'sent_mb': round(getattr(system_metrics, 'network_sent_mb', 0) or 0, 2),
            'recv_mb': round(getattr(system_metrics, 'network_recv_mb', 0) or 0, 2)
        },
        'load_average': getattr(system_metrics, 'load_average', [0, 0, 0]) or [0, 0, 0],
        'process_count': getattr(system_metrics, 'process_count', 0) or 0,
        'timestamp': getattr(system_metrics, 'timestamp', time.time())
    }


def format_gpu_metrics(gpu_metrics, config: PerformanceAPIConfig = None) -> Dict[str, Any]:
    """格式化GPU指标"""
    config = config or get_config()
    
    utilization = getattr(gpu_metrics, 'gpu_utilization', 0) or 0
    memory_util = getattr(gpu_metrics, 'memory_utilization', 0) or 0
    temperature = getattr(gpu_metrics, 'temperature', 0) or 0
    
    return {
        'id': getattr(gpu_metrics, 'gpu_id', 0),
        'name': getattr(gpu_metrics, 'gpu_name', 'Unknown'),
        'utilization': {
            'percent': round(utilization, 2),
            'status': get_resource_status(utilization, 'gpu_utilization')
        },
        'memory': {
            'utilization_percent': round(memory_util, 2),
            'used_mb': round(getattr(gpu_metrics, 'memory_used_mb', 0) or 0, 2),
            'total_mb': round(getattr(gpu_metrics, 'memory_total_mb', 0) or 0, 2),
            'status': get_resource_status(memory_util, 'gpu_utilization')
        },
        'temperature': {
            'celsius': temperature,
            'status': get_temperature_status(temperature)
        },
        'power': {
            'watts': round(getattr(gpu_metrics, 'power_watts', 0) or 0, 2),
            'limit_watts': round(getattr(gpu_metrics, 'power_limit_watts', 0) or 0, 2)
        },
        'timestamp': getattr(gpu_metrics, 'timestamp', time.time())
    }


def format_training_metrics(training_metrics) -> Dict[str, Any]:
    """格式化训练指标"""
    return {
        'session_id': getattr(training_metrics, 'session_id', None),
        'epoch': getattr(training_metrics, 'epoch', 0),
        'step': getattr(training_metrics, 'step', 0),
        'loss': round(getattr(training_metrics, 'loss', 0) or 0, 6),
        'accuracy': round(getattr(training_metrics, 'accuracy', 0) or 0, 4) if getattr(training_metrics, 'accuracy', None) else None,
        'learning_rate': getattr(training_metrics, 'learning_rate', None),
        'samples_per_second': round(getattr(training_metrics, 'samples_per_second', 0) or 0, 2),
        'gpu_utilization': round(getattr(training_metrics, 'gpu_utilization', 0) or 0, 2),
        'memory_usage_gb': round(getattr(training_metrics, 'memory_usage_gb', 0) or 0, 2),
        'timestamp': getattr(training_metrics, 'timestamp', time.time())
    }


def format_task(task: Dict, include_result: bool = False) -> Dict[str, Any]:
    """格式化任务对象"""
    formatted = {
        'id': task.get('id'),
        'name': task.get('name'),
        'category': task.get('category'),
        'status': task.get('status'),
        'priority': task.get('priority'),
        'created_at': task.get('created_at'),
        'started_at': task.get('started_at'),
        'completed_at': task.get('completed_at'),
        'execution_time': task.get('execution_time'),
        'retry_count': task.get('retry_count', 0)
    }
    
    if include_result:
        formatted['result'] = task.get('result')
        formatted['error_message'] = task.get('error_message')
    
    # 计算等待时间
    if formatted['created_at'] and formatted['started_at']:
        try:
            created = datetime.fromisoformat(str(formatted['created_at']).replace('Z', ''))
            started = datetime.fromisoformat(str(formatted['started_at']).replace('Z', ''))
            formatted['wait_time_seconds'] = (started - created).total_seconds()
        except (ValueError, TypeError):
            pass
    
    return formatted


def format_alert(alert) -> Dict[str, Any]:
    """格式化告警对象"""
    # 获取时间戳
    if hasattr(alert, 'timestamp') and alert.timestamp:
        if hasattr(alert.timestamp, 'isoformat'):
            timestamp_iso = alert.timestamp.isoformat()
            timestamp = alert.timestamp.timestamp() if hasattr(alert.timestamp, 'timestamp') else time.time()
        else:
            timestamp = alert.timestamp if alert.timestamp else time.time()
            timestamp_iso = timestamp_to_iso(timestamp)
    else:
        timestamp = time.time()
        timestamp_iso = timestamp_to_iso(timestamp)
    
    # 获取解除时间
    resolved_at = None
    if getattr(alert, 'resolved', False) and getattr(alert, 'resolved_at', None):
        if hasattr(alert.resolved_at, 'isoformat'):
            resolved_at = alert.resolved_at.isoformat()
        else:
            resolved_at = timestamp_to_iso(alert.resolved_at)
    
    # 计算持续时间
    duration_seconds = None
    if getattr(alert, 'resolved', False) and resolved_at:
        try:
            if hasattr(alert.resolved_at, 'timestamp') and hasattr(alert.timestamp, 'timestamp'):
                duration_seconds = alert.resolved_at.timestamp() - alert.timestamp.timestamp()
        except Exception:
            pass
    elif not getattr(alert, 'resolved', False):
        try:
            if hasattr(alert.timestamp, 'timestamp'):
                duration_seconds = time.time() - alert.timestamp.timestamp()
            else:
                duration_seconds = time.time() - timestamp
        except Exception:
            pass
    
    return {
        'id': getattr(alert, 'alert_id', '') or getattr(alert, 'id', ''),
        'rule_id': getattr(alert, 'rule_id', ''),
        'name': getattr(alert, 'name', ''),
        'description': getattr(alert, 'description', ''),
        'level': alert.level.value if hasattr(alert, 'level') and hasattr(alert.level, 'value') else str(getattr(alert, 'level', 'unknown')),
        'metric_type': getattr(alert, 'metric_type', ''),
        'metric_name': getattr(alert, 'metric_name', ''),
        'metric_value': getattr(alert, 'metric_value', None),
        'threshold': getattr(alert, 'threshold', None),
        'timestamp': timestamp_iso,
        'timestamp_unix': timestamp,
        'resolved': getattr(alert, 'resolved', False),
        'resolved_at': resolved_at,
        'resolved_by': getattr(alert, 'resolved_by', None),
        'acknowledged': getattr(alert, 'acknowledged', False),
        'acknowledged_by': getattr(alert, 'acknowledged_by', None),
        'duration_seconds': round(duration_seconds, 2) if duration_seconds else None,
        'duration_human': format_duration(duration_seconds) if duration_seconds else None
    }


def format_alert_rule(rule) -> Dict[str, Any]:
    """格式化告警规则对象"""
    return {
        'id': getattr(rule, 'id', '') or rule.get('id', ''),
        'name': getattr(rule, 'name', '') or rule.get('name', ''),
        'description': getattr(rule, 'description', '') or rule.get('description', ''),
        'metric_type': getattr(rule, 'metric_type', '') or rule.get('metric_type', ''),
        'metric_name': getattr(rule, 'metric_name', '') or rule.get('metric_name', ''),
        'operator': getattr(rule, 'operator', '') or rule.get('operator', ''),
        'threshold': getattr(rule, 'threshold', 0) or rule.get('threshold', 0),
        'severity': getattr(rule, 'severity', 'medium') or rule.get('severity', 'medium'),
        'enabled': getattr(rule, 'enabled', True) if hasattr(rule, 'enabled') else rule.get('enabled', True),
        'duration': getattr(rule, 'duration', 0) or rule.get('duration', 0),
        'notification_channels': getattr(rule, 'notification_channels', []) or rule.get('notification_channels', []),
        'created_at': getattr(rule, 'created_at', None) or rule.get('created_at'),
        'updated_at': getattr(rule, 'updated_at', None) or rule.get('updated_at')
    }


# ============================================================================
# 指标聚合工具
# ============================================================================

def aggregate_metrics(
    metrics: List[Dict],
    group_by: str = None,
    aggregation: str = 'avg'
) -> Dict[str, Any]:
    """聚合指标数据
    
    Args:
        metrics: 指标数据列表
        group_by: 分组字段
        aggregation: 聚合方式 (avg/sum/min/max/count)
    
    Returns:
        聚合结果
    """
    if not metrics:
        return {'count': 0, 'aggregation': aggregation, 'result': None}
    
    # 提取数值
    values = []
    for m in metrics:
        value = m.get('metric_value') or m.get('value')
        if value is not None:
            try:
                values.append(float(value))
            except (ValueError, TypeError):
                pass
    
    if not values:
        return {'count': 0, 'aggregation': aggregation, 'result': None}
    
    # 计算聚合
    if aggregation == 'avg':
        result = sum(values) / len(values)
    elif aggregation == 'sum':
        result = sum(values)
    elif aggregation == 'min':
        result = min(values)
    elif aggregation == 'max':
        result = max(values)
    elif aggregation == 'count':
        result = len(values)
    elif aggregation == 'std':
        avg = sum(values) / len(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        result = variance ** 0.5
    else:
        result = sum(values) / len(values)
    
    return {
        'count': len(values),
        'aggregation': aggregation,
        'result': round(result, 4),
        'min': round(min(values), 4),
        'max': round(max(values), 4)
    }


def calculate_percentiles(values: List[float], percentiles: List[int] = None) -> Dict[str, float]:
    """计算百分位数"""
    if not values:
        return {}
    
    percentiles = percentiles or [50, 90, 95, 99]
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    result = {}
    for p in percentiles:
        idx = (n - 1) * p / 100
        lower = int(idx)
        upper = lower + 1
        weight = idx - lower
        
        if upper >= n:
            result[f'p{p}'] = round(sorted_values[-1], 4)
        else:
            result[f'p{p}'] = round(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight, 4)
    
    return result


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 配置
    'PerformanceAPIConfig',
    'get_config',
    
    # 响应
    'APIResponse',
    
    # 时间工具
    'timestamp_to_iso',
    'iso_to_timestamp',
    'format_duration',
    'parse_time_range',
    'get_time_bucket',
    
    # 状态评估
    'get_resource_status',
    'get_temperature_status',
    'assess_utilization_status',
    'calculate_health_score',
    
    # 健康检查
    'HealthCheckResult',
    'calculate_health_summary',
    
    # 请求参数
    'get_bool_param',
    'get_int_param',
    'get_float_param',
    'get_list_param',
    'get_datetime_param',
    'get_enum_param',
    'get_pagination_params',
    'get_sort_params',
    
    # 验证
    'ValidationError',
    'validate_required',
    'validate_string_length',
    'validate_enum_value',
    'validate_numeric_range',
    'validate_uuid',
    'validate_email',
    'validate_request_body',
    
    # 缓存
    'LRUCache',
    'get_api_cache',
    'cached',
    
    # 限流
    'RateLimiter',
    'get_rate_limiter',
    'rate_limit',
    
    # 装饰器
    'handle_api_errors',
    'timed_execution',
    'log_request',
    'require_json',
    'validate_params',
    
    # 格式化
    'format_system_metrics',
    'format_gpu_metrics',
    'format_training_metrics',
    'format_task',
    'format_alert',
    'format_alert_rule',
    
    # 聚合
    'aggregate_metrics',
    'calculate_percentiles',
]

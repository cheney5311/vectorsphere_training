"""
工具定义和管理 - 生产级实现

提供工具注册、调用和执行的完整机制：
- Tool: 工具基类（支持验证、缓存、重试、限流）
- ToolRegistry: 工具注册表（支持分组、搜索、版本管理）
- ToolExecutor: 工具执行器（支持并行、优先级、监控）
- ToolManager: 工具管理器（整合注册和执行）
- 装饰器: @tool, @async_tool, @cached_tool, @rate_limited_tool, @validated_tool
"""

import json
import time
import asyncio
import inspect
import logging
import hashlib
import threading
import re
import uuid
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Union, Callable, Set, Tuple,
    TypeVar, Awaitable, get_type_hints, Pattern
)
from dataclasses import dataclass, field, asdict
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta
from collections import defaultdict, deque
from abc import ABC, abstractmethod

from .state import ToolCall, ToolResult, ErrorType, PriorityLevel

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


# =============================================================================
# 枚举类型
# =============================================================================

class ToolCategory(Enum):
    """工具类别"""
    SEARCH = "search"
    RETRIEVAL = "retrieval"
    COMPUTATION = "computation"
    CODE = "code"
    DATA = "data"
    API = "api"
    SYSTEM = "system"
    FILE = "file"
    NETWORK = "network"
    DATABASE = "database"
    MEMORY = "memory"
    WORKFLOW = "workflow"
    CUSTOM = "custom"


class ToolStatus(Enum):
    """工具状态"""
    ACTIVE = "active"
    DISABLED = "disabled"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"


class ExecutionMode(Enum):
    """执行模式"""
    SYNC = "sync"
    ASYNC = "async"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


class RetryStrategy(Enum):
    """重试策略"""
    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


class CacheStrategy(Enum):
    """缓存策略"""
    NONE = "none"
    MEMORY = "memory"
    LRU = "lru"
    TTL = "ttl"


class ValidationLevel(Enum):
    """验证级别"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"


# =============================================================================
# 工具指标
# =============================================================================

@dataclass
class ToolMetrics:
    """工具执行指标"""
    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_execution_time_ms: float = 0.0
    min_execution_time_ms: float = float('inf')
    max_execution_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    retry_count: int = 0
    rate_limit_hits: int = 0
    last_called: Optional[datetime] = None
    last_error: Optional[str] = None
    error_counts: Dict[str, int] = field(default_factory=dict)
    
    def record_call(self, success: bool, execution_time_ms: float, 
                    error: str = None, from_cache: bool = False) -> None:
        """记录调用"""
        self.total_calls += 1
        self.last_called = datetime.utcnow()
        
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            if error:
                self.last_error = error
                error_type = error.split(':')[0] if ':' in error else 'Unknown'
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        if from_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            self.total_execution_time_ms += execution_time_ms
            self.min_execution_time_ms = min(self.min_execution_time_ms, execution_time_ms)
            self.max_execution_time_ms = max(self.max_execution_time_ms, execution_time_ms)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        return self.successful_calls / self.total_calls if self.total_calls > 0 else 0.0
    
    @property
    def avg_execution_time_ms(self) -> float:
        """平均执行时间"""
        actual_calls = self.total_calls - self.cache_hits
        return self.total_execution_time_ms / actual_calls if actual_calls > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率"""
        return self.cache_hits / self.total_calls if self.total_calls > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.success_rate,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "min_execution_time_ms": self.min_execution_time_ms if self.min_execution_time_ms != float('inf') else 0,
            "max_execution_time_ms": self.max_execution_time_ms,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "retry_count": self.retry_count,
            "rate_limit_hits": self.rate_limit_hits,
            "last_called": self.last_called.isoformat() if self.last_called else None,
            "last_error": self.last_error,
            "error_counts": self.error_counts
        }


# =============================================================================
# 工具参数
# =============================================================================

@dataclass
class ToolParameter:
    """工具参数定义 - 增强版"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: List[Any] = field(default_factory=list)
    # 增强字段
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None  # 正则表达式
    examples: List[Any] = field(default_factory=list)
    
    def to_json_schema(self) -> Dict[str, Any]:
        """转换为 JSON Schema"""
        schema = {
            "type": self._python_type_to_json(self.type),
            "description": self.description
        }
        
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        if self.min_value is not None:
            schema["minimum"] = self.min_value
        if self.max_value is not None:
            schema["maximum"] = self.max_value
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.pattern:
            schema["pattern"] = self.pattern
        if self.examples:
            schema["examples"] = self.examples
            
        return schema
    
    @staticmethod
    def _python_type_to_json(python_type: str) -> str:
        """Python 类型转 JSON Schema 类型"""
        type_map = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "List": "array",
            "Dict": "object",
            "Any": "string",
            "Optional": "string",
            "Union": "string"
        }
        # 处理泛型类型
        base_type = python_type.split('[')[0]
        return type_map.get(base_type, "string")
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """验证参数值"""
        # 必填检查
        if value is None:
            if self.required:
                return False, f"参数 '{self.name}' 是必填项"
            return True, None
        
        # 枚举检查
        if self.enum and value not in self.enum:
            return False, f"参数 '{self.name}' 的值必须是 {self.enum} 之一"
        
        # 数值范围检查
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                return False, f"参数 '{self.name}' 的值不能小于 {self.min_value}"
            if self.max_value is not None and value > self.max_value:
                return False, f"参数 '{self.name}' 的值不能大于 {self.max_value}"
        
        # 字符串长度检查
        if isinstance(value, str):
            if self.min_length is not None and len(value) < self.min_length:
                return False, f"参数 '{self.name}' 的长度不能小于 {self.min_length}"
            if self.max_length is not None and len(value) > self.max_length:
                return False, f"参数 '{self.name}' 的长度不能大于 {self.max_length}"
            if self.pattern and not re.match(self.pattern, value):
                return False, f"参数 '{self.name}' 的格式不正确"
        
        return True, None


# =============================================================================
# 工具缓存
# =============================================================================

class ToolCache:
    """工具结果缓存"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: int = 300,
                 strategy: CacheStrategy = CacheStrategy.LRU):
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._access_order: deque = deque()
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """生成缓存键"""
        args_str = json.dumps(arguments, sort_keys=True, default=str)
        return hashlib.md5(f"{tool_name}:{args_str}".encode()).hexdigest()
    
    def get(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Any]:
        """获取缓存"""
        if self.strategy == CacheStrategy.NONE:
            self._misses += 1
            return False, None
        
        key = self._make_key(tool_name, arguments)
        
        with self._lock:
            if key in self._cache:
                value, expire_time = self._cache[key]
                
                # 检查是否过期
                if self.strategy == CacheStrategy.TTL and datetime.utcnow() > expire_time:
                    del self._cache[key]
                    self._misses += 1
                    return False, None
                
                # LRU: 更新访问顺序
                if self.strategy == CacheStrategy.LRU:
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._access_order.append(key)
                
                self._hits += 1
                return True, value
            
            self._misses += 1
            return False, None
    
    def set(self, tool_name: str, arguments: Dict[str, Any], 
            value: Any, ttl: int = None) -> None:
        """设置缓存"""
        if self.strategy == CacheStrategy.NONE:
            return
        
        key = self._make_key(tool_name, arguments)
        ttl = ttl or self.default_ttl
        expire_time = datetime.utcnow() + timedelta(seconds=ttl)
        
        with self._lock:
            # 检查容量
            if len(self._cache) >= self.max_size:
                self._evict()
            
            self._cache[key] = (value, expire_time)
            
            if self.strategy == CacheStrategy.LRU:
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
    
    def _evict(self) -> None:
        """驱逐缓存项"""
        if self.strategy == CacheStrategy.LRU and self._access_order:
            oldest_key = self._access_order.popleft()
            self._cache.pop(oldest_key, None)
        elif self._cache:
            # 移除最早的项
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
    
    def invalidate(self, tool_name: str = None) -> int:
        """使缓存失效"""
        with self._lock:
            if tool_name:
                # 使特定工具的缓存失效
                keys_to_remove = [k for k in self._cache if k.startswith(tool_name)]
                for key in keys_to_remove:
                    del self._cache[key]
                return len(keys_to_remove)
            else:
                # 清空所有缓存
                count = len(self._cache)
                self._cache.clear()
                self._access_order.clear()
                return count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "strategy": self.strategy.value
            }


# =============================================================================
# 工具限流器
# =============================================================================

class ToolRateLimiter:
    """工具限流器 - 令牌桶算法"""
    
    def __init__(self,
                 rate: float = 10.0,  # 每秒请求数
                 burst: int = 20,     # 突发容量
                 per_tool: bool = True):
        self.rate = rate
        self.burst = burst
        self.per_tool = per_tool
        
        self._tokens: Dict[str, float] = defaultdict(lambda: float(burst))
        self._last_update: Dict[str, float] = defaultdict(time.time)
        self._lock = threading.Lock()
    
    def _get_key(self, tool_name: str) -> str:
        return tool_name if self.per_tool else "_global_"
    
    def acquire(self, tool_name: str, tokens: int = 1) -> bool:
        """尝试获取令牌"""
        key = self._get_key(tool_name)
        
        with self._lock:
            now = time.time()
            time_passed = now - self._last_update[key]
            self._last_update[key] = now
            
            # 添加新令牌
            self._tokens[key] = min(
                self.burst,
                self._tokens[key] + time_passed * self.rate
            )
            
            if self._tokens[key] >= tokens:
                self._tokens[key] -= tokens
                return True
            
            return False
    
    def wait_and_acquire(self, tool_name: str, 
                         tokens: int = 1,
                         timeout: float = None) -> bool:
        """等待并获取令牌"""
        start_time = time.time()
        
        while True:
            if self.acquire(tool_name, tokens):
                return True
            
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            # 等待一小段时间
            time.sleep(0.01)
    
    async def async_acquire(self, tool_name: str, tokens: int = 1) -> bool:
        """异步获取令牌"""
        key = self._get_key(tool_name)
        
        while not self.acquire(tool_name, tokens):
            await asyncio.sleep(0.01)
        
        return True
    
    def get_available_tokens(self, tool_name: str) -> float:
        """获取可用令牌数"""
        key = self._get_key(tool_name)
        
        with self._lock:
            now = time.time()
            time_passed = now - self._last_update[key]
            return min(
                self.burst,
                self._tokens[key] + time_passed * self.rate
            )


# =============================================================================
# 重试处理器
# =============================================================================

class RetryHandler:
    """重试处理器"""
    
    def __init__(self,
                 strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 retryable_errors: Set[str] = None):
        self.strategy = strategy
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retryable_errors = retryable_errors or {
            'timeout', 'connection', 'rate_limit', 'temporary'
        }
    
    def should_retry(self, error: str, attempt: int) -> bool:
        """是否应该重试"""
        if self.strategy == RetryStrategy.NONE:
            return False
        
        if attempt >= self.max_retries:
            return False
        
        # 检查是否是可重试的错误
        error_lower = error.lower()
        for retryable in self.retryable_errors:
            if retryable in error_lower:
                return True
        
        return False
    
    def get_delay(self, attempt: int) -> float:
        """获取重试延迟"""
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (2 ** attempt)
        else:
            delay = 0
        
        return min(delay, self.max_delay)
    
    async def async_delay(self, attempt: int) -> None:
        """异步等待"""
        delay = self.get_delay(attempt)
        if delay > 0:
            await asyncio.sleep(delay)
    
    def sync_delay(self, attempt: int) -> None:
        """同步等待"""
        delay = self.get_delay(attempt)
        if delay > 0:
            time.sleep(delay)


# =============================================================================
# 工具钩子
# =============================================================================

@dataclass
class ToolHooks:
    """工具执行钩子"""
    before_execute: List[Callable] = field(default_factory=list)
    after_execute: List[Callable] = field(default_factory=list)
    on_error: List[Callable] = field(default_factory=list)
    on_retry: List[Callable] = field(default_factory=list)
    on_cache_hit: List[Callable] = field(default_factory=list)
    
    def add_before(self, callback: Callable) -> None:
        self.before_execute.append(callback)
    
    def add_after(self, callback: Callable) -> None:
        self.after_execute.append(callback)
    
    def add_on_error(self, callback: Callable) -> None:
        self.on_error.append(callback)
    
    def add_on_retry(self, callback: Callable) -> None:
        self.on_retry.append(callback)
    
    def trigger_before(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        for callback in self.before_execute:
            try:
                callback(tool_name, arguments)
            except Exception as e:
                logger.error(f"Before hook error: {e}")
    
    def trigger_after(self, tool_name: str, result: Any, 
                      execution_time: float) -> None:
        for callback in self.after_execute:
            try:
                callback(tool_name, result, execution_time)
            except Exception as e:
                logger.error(f"After hook error: {e}")
    
    def trigger_error(self, tool_name: str, error: Exception) -> None:
        for callback in self.on_error:
            try:
                callback(tool_name, error)
            except Exception as e:
                logger.error(f"Error hook error: {e}")
    
    def trigger_retry(self, tool_name: str, attempt: int, error: str) -> None:
        for callback in self.on_retry:
            try:
                callback(tool_name, attempt, error)
            except Exception as e:
                logger.error(f"Retry hook error: {e}")


# =============================================================================
# 工具定义
# =============================================================================

@dataclass
class Tool:
    """工具定义 - 生产级实现
    
    封装一个可被 Agent 调用的工具，支持：
    - 参数验证
    - 结果缓存
    - 执行重试
    - 限流控制
    - 执行钩子
    - 版本管理
    """
    name: str
    description: str
    func: Callable
    parameters: List[ToolParameter] = field(default_factory=list)
    category: ToolCategory = ToolCategory.CUSTOM
    is_async: bool = False
    timeout: float = 30.0
    return_direct: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 增强字段
    status: ToolStatus = ToolStatus.ACTIVE
    version: str = "1.0.0"
    author: str = ""
    tags: Set[str] = field(default_factory=set)
    
    # 缓存配置
    cache_enabled: bool = False
    cache_ttl: int = 300
    
    # 重试配置
    retry_enabled: bool = False
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    
    # 限流配置
    rate_limit_enabled: bool = False
    rate_limit: float = 10.0
    rate_limit_burst: int = 20
    
    # 验证配置
    validation_level: ValidationLevel = ValidationLevel.BASIC
    
    # 权限
    required_permissions: Set[str] = field(default_factory=set)
    
    # 依赖
    dependencies: List[str] = field(default_factory=list)
    
    # 钩子
    hooks: ToolHooks = field(default_factory=ToolHooks)
    
    # 指标
    _metrics: Optional[ToolMetrics] = field(default=None, repr=False)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.parameters:
            self.parameters = self._infer_parameters()
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
        if self._metrics is None:
            self._metrics = ToolMetrics(tool_name=self.name)
    
    @property
    def metrics(self) -> ToolMetrics:
        if self._metrics is None:
            self._metrics = ToolMetrics(tool_name=self.name)
        return self._metrics
    
    def _infer_parameters(self) -> List[ToolParameter]:
        """从函数签名推断参数"""
        parameters = []
        try:
            sig = inspect.signature(self.func)
            type_hints = get_type_hints(self.func) if hasattr(self.func, '__annotations__') else {}
        except (ValueError, TypeError):
            return parameters
        
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls', 'state', 'context', 'kwargs', 'args'):
                continue
            
            param_type = type_hints.get(param_name, Any)
            type_name = getattr(param_type, '__name__', str(param_type))
            
            required = param.default == inspect.Parameter.empty
            default = None if required else param.default
            
            # 尝试从 docstring 获取描述
            description = f"参数 {param_name}"
            if self.func.__doc__:
                doc_lines = self.func.__doc__.split('\n')
                for line in doc_lines:
                    if param_name in line and ':' in line:
                        description = line.split(':', 1)[-1].strip()
                        break
            
            parameters.append(ToolParameter(
                name=param_name,
                type=type_name,
                description=description,
                required=required,
                default=default
            ))
        
        return parameters
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证参数"""
        if self.validation_level == ValidationLevel.NONE:
            return True, []
        
        errors = []
        
        # 检查必填参数
        for param in self.parameters:
            if param.required and param.name not in arguments:
                errors.append(f"缺少必填参数: {param.name}")
            elif param.name in arguments:
                valid, error = param.validate(arguments[param.name])
                if not valid:
                    errors.append(error)
        
        # 严格模式：检查未知参数
        if self.validation_level == ValidationLevel.STRICT:
            param_names = {p.name for p in self.parameters}
            for arg_name in arguments:
                if arg_name not in param_names:
                    errors.append(f"未知参数: {arg_name}")
        
        return len(errors) == 0, errors
    
    def is_available(self) -> bool:
        """检查工具是否可用"""
        return self.status == ToolStatus.ACTIVE
    
    def to_openai_function(self) -> Dict[str, Any]:
        """转换为 OpenAI Function Calling 格式"""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    def to_anthropic_tool(self) -> Dict[str, Any]:
        """转换为 Anthropic Tool 格式"""
        input_schema = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param in self.parameters:
            input_schema["properties"][param.name] = param.to_json_schema()
            if param.required:
                input_schema["required"].append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": input_schema
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "status": self.status.value,
            "version": self.version,
            "is_async": self.is_async,
            "timeout": self.timeout,
            "return_direct": self.return_direct,
            "parameters": [asdict(p) for p in self.parameters],
            "tags": list(self.tags),
            "cache_enabled": self.cache_enabled,
            "retry_enabled": self.retry_enabled,
            "rate_limit_enabled": self.rate_limit_enabled,
            "metadata": self.metadata
        }
    
    def __call__(self, **kwargs) -> Any:
        """调用工具"""
        return self.func(**kwargs)
    
    async def acall(self, **kwargs) -> Any:
        """异步调用工具"""
        if self.is_async:
            return await self.func(**kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.func(**kwargs))

    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Tool):
            return self.name == other.name
        return False


# =============================================================================
# 工具注册表
# =============================================================================

class ToolRegistry:
    """工具注册表 - 生产级实现
    
    管理所有可用的工具，支持：
    - 分组管理
    - 版本管理
    - 工具搜索
    - 权限检查
    - 依赖解析
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {cat: [] for cat in ToolCategory}
        self._tags: Dict[str, Set[str]] = defaultdict(set)
        self._versions: Dict[str, Dict[str, Tool]] = defaultdict(dict)
        self._aliases: Dict[str, str] = {}
        self._lock = threading.RLock()
    
    def register(self, tool: Tool, alias: str = None) -> None:
        """注册工具"""
        with self._lock:
            if tool.name in self._tools:
                logger.warning(f"Tool '{tool.name}' already exists, overwriting")
                self._unregister_internal(tool.name)
            
            self._tools[tool.name] = tool
            
            if tool.name not in self._categories[tool.category]:
                self._categories[tool.category].append(tool.name)
            
            for tag in tool.tags:
                self._tags[tag].add(tool.name)
            
            # 版本管理
            self._versions[tool.name][tool.version] = tool
            
            # 别名
            if alias:
                self._aliases[alias] = tool.name
            
            logger.debug(f"Registered tool: {tool.name} v{tool.version}")
    
    def register_many(self, tools: List[Tool]) -> int:
        """批量注册工具"""
        count = 0
        for tool in tools:
            try:
                self.register(tool)
                count += 1
            except Exception as e:
                logger.error(f"Failed to register tool {tool.name}: {e}")
        return count
    
    def _unregister_internal(self, name: str) -> None:
        """内部注销工具（不加锁）"""
        if name in self._tools:
            tool = self._tools.pop(name)
            if name in self._categories[tool.category]:
                self._categories[tool.category].remove(name)
            for tag in tool.tags:
                self._tags[tag].discard(name)
            # 移除别名
            aliases_to_remove = [a for a, t in self._aliases.items() if t == name]
            for alias in aliases_to_remove:
                del self._aliases[alias]
    
    def unregister(self, name: str) -> bool:
        """注销工具"""
        with self._lock:
            if name in self._tools:
                self._unregister_internal(name)
            return True
        return False
    
    def get(self, name: str, version: str = None) -> Optional[Tool]:
        """获取工具"""
        with self._lock:
            # 检查别名
            actual_name = self._aliases.get(name, name)
            
            if version:
                return self._versions.get(actual_name, {}).get(version)
            return self._tools.get(actual_name)
    
    def get_by_category(self, category: ToolCategory) -> List[Tool]:
        """按类别获取工具"""
        with self._lock:
            return [self._tools[name] for name in self._categories.get(category, [])
                    if name in self._tools]
    
    def get_by_tag(self, tag: str) -> List[Tool]:
        """按标签获取工具"""
        with self._lock:
            return [self._tools[name] for name in self._tags.get(tag, set())
                    if name in self._tools]
    
    def get_active_tools(self) -> List[Tool]:
        """获取所有活跃的工具"""
        with self._lock:
            return [t for t in self._tools.values() if t.is_available()]
    
    def search(self, 
               query: str = None,
               category: ToolCategory = None,
               tags: Set[str] = None,
               status: ToolStatus = None) -> List[Tool]:
        """搜索工具"""
        with self._lock:
            results = list(self._tools.values())
            
            if query:
                query_lower = query.lower()
                results = [t for t in results 
                          if query_lower in t.name.lower() 
                          or query_lower in t.description.lower()]
            
            if category:
                results = [t for t in results if t.category == category]
            
            if tags:
                results = [t for t in results if tags.issubset(t.tags)]
            
            if status:
                results = [t for t in results if t.status == status]
            
            return results
    
    def list_tools(self, category: ToolCategory = None, 
                   include_disabled: bool = False) -> List[Tool]:
        """列出工具"""
        with self._lock:
            if category:
                tools = [self._tools[name] for name in self._categories.get(category, [])
                        if name in self._tools]
            else:
                tools = list(self._tools.values())
            
            if not include_disabled:
                tools = [t for t in tools if t.is_available()]
            
            return tools
    
    def get_tool_schemas(self, format: str = "openai", 
                         tools: List[str] = None) -> List[Dict[str, Any]]:
        """获取工具 schema"""
        with self._lock:
            schemas = []
            target_tools = [self._tools[t] for t in tools if t in self._tools] \
                          if tools else self._tools.values()
            
            for tool in target_tools:
                if not tool.is_available():
                    continue
                
                if format == "openai":
                    schemas.append(tool.to_openai_function())
                elif format == "anthropic":
                    schemas.append(tool.to_anthropic_tool())
                elif format == "dict":
                    schemas.append(tool.to_dict())
            
            return schemas
    
    def resolve_dependencies(self, tool_name: str) -> List[str]:
        """解析工具依赖"""
        with self._lock:
            tool = self._tools.get(tool_name)
            if not tool:
                return []
            
            resolved = []
            visited = set()
            
            def resolve(name: str):
                if name in visited:
                    return
                visited.add(name)
                
                t = self._tools.get(name)
                if t:
                    for dep in t.dependencies:
                        resolve(dep)
                    resolved.append(name)
            
            resolve(tool_name)
            return resolved
    
    def check_permissions(self, tool_name: str, 
                          user_permissions: Set[str]) -> bool:
        """检查权限"""
        with self._lock:
            tool = self._tools.get(tool_name)
            if not tool:
                return False
            
            if not tool.required_permissions:
                return True
            
            return tool.required_permissions.issubset(user_permissions)
    
    def set_tool_status(self, name: str, status: ToolStatus) -> bool:
        """设置工具状态"""
        with self._lock:
            tool = self._tools.get(name)
            if tool:
                tool.status = status
                return True
            return False
    
    def add_alias(self, alias: str, tool_name: str) -> bool:
        """添加别名"""
        with self._lock:
            if tool_name in self._tools:
                self._aliases[alias] = tool_name
                return True
            return False
    
    def get_all_tags(self) -> List[str]:
        """获取所有标签"""
        with self._lock:
            return list(self._tags.keys())
    
    def get_summary(self) -> Dict[str, Any]:
        """获取注册表摘要"""
        with self._lock:
            return {
                "total_tools": len(self._tools),
                "active_tools": len([t for t in self._tools.values() if t.is_available()]),
                "categories": {cat.value: len(tools) for cat, tools in self._categories.items() if tools},
                "tags": list(self._tags.keys()),
                "aliases": len(self._aliases)
            }
    
    def export_tools(self) -> List[Dict[str, Any]]:
        """导出所有工具配置"""
        with self._lock:
            return [tool.to_dict() for tool in self._tools.values()]
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools or name in self._aliases
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __iter__(self):
        return iter(self._tools.values())


# 全局工具注册表
_global_registry = ToolRegistry()


def get_global_registry() -> ToolRegistry:
    """获取全局工具注册表"""
    return _global_registry


# =============================================================================
# 工具执行器
# =============================================================================

class ToolExecutor:
    """工具执行器 - 生产级实现
    
    负责执行工具调用，支持：
    - 并行执行
    - 重试机制
    - 缓存管理
    - 限流控制
    - 执行监控
    - 优先级调度
    """
    
    def __init__(self, 
                 registry: ToolRegistry = None, 
                 max_workers: int = 8,
                 cache: ToolCache = None,
                 rate_limiter: ToolRateLimiter = None):
        self.registry = registry or get_global_registry()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = cache or ToolCache()
        self.rate_limiter = rate_limiter or ToolRateLimiter()
        
        self._execution_history: deque = deque(maxlen=1000)
        self._retry_handler = RetryHandler()
        self._global_hooks = ToolHooks()
        self._lock = threading.Lock()
        
        # 执行统计
        self._total_executions = 0
        self._successful_executions = 0
        self._failed_executions = 0
    
    def execute(self, tool_call: ToolCall, 
                skip_cache: bool = False,
                skip_rate_limit: bool = False,
                user_permissions: Set[str] = None) -> ToolResult:
        """执行单个工具调用"""
        start_time = time.time()
        tool = self.registry.get(tool_call.name)
        
        if not tool:
            return self._create_error_result(
                tool_call, f"Tool '{tool_call.name}' not found",
                time.time() - start_time
            )
        
        # 检查工具状态
        if not tool.is_available():
            return self._create_error_result(
                tool_call, f"Tool '{tool_call.name}' is not available (status: {tool.status.value})",
                time.time() - start_time
            )
        
        # 权限检查
        if user_permissions is not None:
            if not self.registry.check_permissions(tool_call.name, user_permissions):
                return self._create_error_result(
                    tool_call, f"Permission denied for tool '{tool_call.name}'",
                    time.time() - start_time
                )
        
        # 参数验证
        valid, errors = tool.validate_arguments(tool_call.arguments)
        if not valid:
            return self._create_error_result(
                tool_call, f"Validation failed: {'; '.join(errors)}",
                time.time() - start_time
            )
        
        # 检查缓存
        if tool.cache_enabled and not skip_cache:
            hit, cached_result = self.cache.get(tool.name, tool_call.arguments)
            if hit:
                tool.metrics.record_call(True, 0, from_cache=True)
                for cb in tool.hooks.on_cache_hit:
                    try:
                        cb(tool.name, cached_result)
                    except Exception:
                        pass
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                    result=cached_result,
                    success=True,
                    execution_time=time.time() - start_time,
                    cached=True
                )
        
        # 限流检查
        if tool.rate_limit_enabled and not skip_rate_limit:
            if not self.rate_limiter.acquire(tool.name):
                tool.metrics.rate_limit_hits += 1
                return self._create_error_result(
                    tool_call, f"Rate limit exceeded for tool '{tool_call.name}'",
                    time.time() - start_time
                )
        
        # 触发前置钩子
        tool.hooks.trigger_before(tool.name, tool_call.arguments)
        self._global_hooks.trigger_before(tool.name, tool_call.arguments)
        
        # 执行（带重试）
        result = self._execute_with_retry(tool, tool_call, start_time)
        
        # 缓存结果
        if result.success and tool.cache_enabled:
            self.cache.set(tool.name, tool_call.arguments, result.result, tool.cache_ttl)
        
        # 触发后置钩子
        execution_time = time.time() - start_time
        tool.hooks.trigger_after(tool.name, result.result, execution_time)
        self._global_hooks.trigger_after(tool.name, result.result, execution_time)
        
        # 记录历史
        with self._lock:
            self._execution_history.append(result)
            self._total_executions += 1
            if result.success:
                self._successful_executions += 1
            else:
                self._failed_executions += 1
        
        return result
    
    def _execute_with_retry(self, tool: Tool, tool_call: ToolCall, 
                            start_time: float) -> ToolResult:
        """带重试的执行"""
        max_attempts = tool.max_retries + 1 if tool.retry_enabled else 1
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                future = self.executor.submit(tool, **tool_call.arguments)
                result = future.result(timeout=tool.timeout)
                
                execution_time = time.time() - start_time
                tool.metrics.record_call(True, execution_time * 1000)
                
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result=result,
                    success=True,
                    execution_time=execution_time,
                    retries_used=attempt
                )
            
            except FuturesTimeoutError:
                last_error = f"Timeout after {tool.timeout}s"
            
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Tool execution error (attempt {attempt + 1}): {e}")
            
            # 检查是否应该重试
            if attempt < max_attempts - 1:
                if tool.retry_enabled and self._retry_handler.should_retry(last_error, attempt):
                    tool.metrics.retry_count += 1
                    tool.hooks.trigger_retry(tool.name, attempt + 1, last_error)
                    self._retry_handler.sync_delay(attempt)
                else:
                    break
        
        # 最终失败
        execution_time = time.time() - start_time
        tool.metrics.record_call(False, execution_time * 1000, error=last_error)
        tool.hooks.trigger_error(tool.name, Exception(last_error))
        self._global_hooks.trigger_error(tool.name, Exception(last_error))
        
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            result=None,
            success=False,
            error=last_error,
            execution_time=execution_time
        )
    
    async def aexecute(self, tool_call: ToolCall,
                       skip_cache: bool = False,
                       skip_rate_limit: bool = False,
                       user_permissions: Set[str] = None) -> ToolResult:
        """异步执行单个工具调用"""
        start_time = time.time()
        tool = self.registry.get(tool_call.name)
        
        if not tool:
            return self._create_error_result(
                tool_call, f"Tool '{tool_call.name}' not found",
                time.time() - start_time
            )
        
        if not tool.is_available():
            return self._create_error_result(
                tool_call, f"Tool '{tool_call.name}' is not available",
                time.time() - start_time
            )
        
        # 权限检查
        if user_permissions is not None:
            if not self.registry.check_permissions(tool_call.name, user_permissions):
                return self._create_error_result(
                    tool_call, "Permission denied",
                    time.time() - start_time
                )
        
        # 参数验证
        valid, errors = tool.validate_arguments(tool_call.arguments)
        if not valid:
            return self._create_error_result(
                tool_call, f"Validation failed: {'; '.join(errors)}",
                time.time() - start_time
            )
        
        # 检查缓存
        if tool.cache_enabled and not skip_cache:
            hit, cached_result = self.cache.get(tool.name, tool_call.arguments)
            if hit:
                tool.metrics.record_call(True, 0, from_cache=True)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                    result=cached_result,
                    success=True,
                    execution_time=time.time() - start_time,
                    cached=True
                )
        
        # 限流
        if tool.rate_limit_enabled and not skip_rate_limit:
            await self.rate_limiter.async_acquire(tool.name)
        
        # 执行
        result = await self._aexecute_with_retry(tool, tool_call, start_time)
        
        # 缓存
        if result.success and tool.cache_enabled:
            self.cache.set(tool.name, tool_call.arguments, result.result, tool.cache_ttl)
        
        # 记录
        with self._lock:
            self._execution_history.append(result)
            self._total_executions += 1
            if result.success:
                self._successful_executions += 1
            else:
                self._failed_executions += 1
        
        return result
    
    async def _aexecute_with_retry(self, tool: Tool, tool_call: ToolCall,
                                    start_time: float) -> ToolResult:
        """异步带重试的执行"""
        max_attempts = tool.max_retries + 1 if tool.retry_enabled else 1
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                result = await asyncio.wait_for(
                    tool.acall(**tool_call.arguments),
                    timeout=tool.timeout
                )
                
                execution_time = time.time() - start_time
                tool.metrics.record_call(True, execution_time * 1000)
                
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result=result,
                    success=True,
                    execution_time=execution_time,
                    retries_used=attempt
                )
            
            except asyncio.TimeoutError:
                last_error = f"Timeout after {tool.timeout}s"
            
            except Exception as e:
                last_error = str(e)
            
            if attempt < max_attempts - 1:
                if tool.retry_enabled and self._retry_handler.should_retry(last_error, attempt):
                    tool.metrics.retry_count += 1
                    await self._retry_handler.async_delay(attempt)
                else:
                    break
        
        execution_time = time.time() - start_time
        tool.metrics.record_call(False, execution_time * 1000, error=last_error)
        
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            result=None,
            success=False,
            error=last_error,
            execution_time=execution_time
        )
    
    def _create_error_result(self, tool_call: ToolCall, 
                             error: str, execution_time: float) -> ToolResult:
        """创建错误结果"""
        return ToolResult(
            tool_call_id=tool_call.id,
            name=tool_call.name,
            result=None,
            success=False,
            error=error,
            execution_time=execution_time
        )
    
    def execute_batch(self, tool_calls: List[ToolCall],
                      parallel: bool = False,
                      user_permissions: Set[str] = None) -> List[ToolResult]:
        """批量执行工具调用"""
        if parallel:
            futures = [
                self.executor.submit(self.execute, tc, user_permissions=user_permissions)
                for tc in tool_calls
            ]
            return [f.result() for f in futures]
        else:
            return [self.execute(tc, user_permissions=user_permissions) for tc in tool_calls]
    
    async def aexecute_batch(self, tool_calls: List[ToolCall], 
                             parallel: bool = True,
                             user_permissions: Set[str] = None) -> List[ToolResult]:
        """异步批量执行工具调用"""
        if parallel:
            tasks = [self.aexecute(tc, user_permissions=user_permissions) for tc in tool_calls]
            return await asyncio.gather(*tasks)
        else:
            results = []
            for tc in tool_calls:
                results.append(await self.aexecute(tc, user_permissions=user_permissions))
            return results
    
    def execute_with_priority(self, tool_calls: List[ToolCall],
                              user_permissions: Set[str] = None) -> List[ToolResult]:
        """按优先级执行工具调用"""
        # 按优先级排序
        sorted_calls = sorted(tool_calls, key=lambda tc: tc.priority.value)
        return self.execute_batch(sorted_calls, user_permissions=user_permissions)
    
    def add_global_hook(self, hook_type: str, callback: Callable) -> None:
        """添加全局钩子"""
        if hook_type == "before":
            self._global_hooks.add_before(callback)
        elif hook_type == "after":
            self._global_hooks.add_after(callback)
        elif hook_type == "error":
            self._global_hooks.add_on_error(callback)
        elif hook_type == "retry":
            self._global_hooks.add_on_retry(callback)
    
    def get_history(self, limit: int = None, 
                    tool_name: str = None) -> List[ToolResult]:
        """获取执行历史"""
        with self._lock:
            history = list(self._execution_history)
        
        if tool_name:
            history = [r for r in history if r.name == tool_name]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def clear_history(self) -> None:
        """清除执行历史"""
        with self._lock:
            self._execution_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        with self._lock:
            return {
                "total_executions": self._total_executions,
                "successful_executions": self._successful_executions,
                "failed_executions": self._failed_executions,
                "success_rate": self._successful_executions / self._total_executions 
                               if self._total_executions > 0 else 0.0,
                "cache_stats": self.cache.get_stats(),
                "history_size": len(self._execution_history)
            }
    
    def get_tool_metrics(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """获取工具指标"""
        tool = self.registry.get(tool_name)
        return tool.metrics.to_dict() if tool else None
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有工具指标"""
        return {
            tool.name: tool.metrics.to_dict()
            for tool in self.registry
        }
    
    def invalidate_cache(self, tool_name: str = None) -> int:
        """使缓存失效"""
        return self.cache.invalidate(tool_name)
    
    def shutdown(self) -> None:
        """关闭执行器"""
        self.executor.shutdown(wait=True)


# =============================================================================
# 工具管理器
# =============================================================================

class ToolManager:
    """工具管理器 - 整合注册和执行
    
    提供统一的工具管理接口。
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.registry = get_global_registry()
        self.executor = ToolExecutor(self.registry)
        self._initialized = True
    
    def register(self, tool: Tool) -> None:
        """注册工具"""
        self.registry.register(tool)
    
    def execute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        """执行工具"""
        return self.executor.execute(tool_call, **kwargs)
    
    async def aexecute(self, tool_call: ToolCall, **kwargs) -> ToolResult:
        """异步执行工具"""
        return await self.executor.aexecute(tool_call, **kwargs)
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self.registry.get(name)
    
    def list_tools(self, **kwargs) -> List[Tool]:
        """列出工具"""
        return self.registry.list_tools(**kwargs)
    
    def get_schemas(self, format: str = "openai") -> List[Dict[str, Any]]:
        """获取工具 schema"""
        return self.registry.get_tool_schemas(format)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "registry": self.registry.get_summary(),
            "executor": self.executor.get_stats()
        }


def get_tool_manager() -> ToolManager:
    """获取工具管理器单例"""
    return ToolManager()


# =============================================================================
# 工具装饰器
# =============================================================================

def tool(
    name: str = None,
    description: str = None,
    category: ToolCategory = ToolCategory.CUSTOM,
    timeout: float = 30.0,
    return_direct: bool = False,
    register: bool = True,
    # 增强参数
    tags: Set[str] = None,
    version: str = "1.0.0",
    cache_enabled: bool = False,
    cache_ttl: int = 300,
    retry_enabled: bool = False,
    max_retries: int = 3,
    rate_limit_enabled: bool = False,
    rate_limit: float = 10.0,
    validation_level: ValidationLevel = ValidationLevel.BASIC,
    required_permissions: Set[str] = None
) -> Callable[[F], F]:
    """工具装饰器 - 生产级
    
    将函数转换为可被 Agent 调用的工具。
    
    示例:
        @tool(
            name="search",
            description="搜索网页",
            cache_enabled=True,
            retry_enabled=True
        )
        def search(query: str) -> str:
            return f"搜索结果: {query}"
    """
    def decorator(func: F) -> F:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"执行 {tool_name}"
        # 清理描述中的空白
        if tool_desc:
            tool_desc = ' '.join(tool_desc.split())
        
        tool_obj = Tool(
            name=tool_name,
            description=tool_desc,
            func=func,
            category=category,
            is_async=asyncio.iscoroutinefunction(func),
            timeout=timeout,
            return_direct=return_direct,
            tags=tags or set(),
            version=version,
            cache_enabled=cache_enabled,
            cache_ttl=cache_ttl,
            retry_enabled=retry_enabled,
            max_retries=max_retries,
            rate_limit_enabled=rate_limit_enabled,
            rate_limit=rate_limit,
            validation_level=validation_level,
            required_permissions=required_permissions or set()
        )
        
        if register:
            get_global_registry().register(tool_obj)
        
        func._tool = tool_obj
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        wrapper._tool = tool_obj
        return wrapper
    
    return decorator


def async_tool(
    name: str = None,
    description: str = None,
    category: ToolCategory = ToolCategory.CUSTOM,
    timeout: float = 30.0,
    return_direct: bool = False,
    register: bool = True,
    **kwargs
) -> Callable[[F], F]:
    """异步工具装饰器"""
    def decorator(func: F) -> F:
        if not asyncio.iscoroutinefunction(func):
            raise ValueError(f"Function {func.__name__} must be async")
        
        return tool(
            name=name,
            description=description,
            category=category,
            timeout=timeout,
            return_direct=return_direct,
            register=register,
            **kwargs
        )(func)
    
    return decorator


def cached_tool(
    ttl: int = 300,
    **kwargs
) -> Callable[[F], F]:
    """带缓存的工具装饰器"""
    kwargs['cache_enabled'] = True
    kwargs['cache_ttl'] = ttl
    return tool(**kwargs)


def rate_limited_tool(
    rate: float = 10.0,
    burst: int = 20,
    **kwargs
) -> Callable[[F], F]:
    """限流工具装饰器"""
    kwargs['rate_limit_enabled'] = True
    kwargs['rate_limit'] = rate
    return tool(**kwargs)


def validated_tool(
    level: ValidationLevel = ValidationLevel.STRICT,
    **kwargs
) -> Callable[[F], F]:
    """带验证的工具装饰器"""
    kwargs['validation_level'] = level
    return tool(**kwargs)


def retry_tool(
    max_retries: int = 3,
    **kwargs
) -> Callable[[F], F]:
    """自动重试工具装饰器"""
    kwargs['retry_enabled'] = True
    kwargs['max_retries'] = max_retries
    return tool(**kwargs)


# =============================================================================
# 预置工具
# =============================================================================

@tool(
    name="calculator",
    description="执行数学计算。支持基本运算和数学函数如 sqrt, log, sin, cos, tan, pi, e。",
    category=ToolCategory.COMPUTATION,
    tags={"math", "calculation"},
    cache_enabled=True,
    cache_ttl=3600
)
def calculator(expression: str) -> str:
    """
    执行数学计算
    
    Args:
        expression: 数学表达式，如 "2 + 2" 或 "sqrt(16)"
    """
    import math
    
    safe_dict = {
        'abs': abs, 'round': round,
        'min': min, 'max': max,
        'sum': sum, 'pow': pow,
        'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
        'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
        'exp': math.exp, 'floor': math.floor, 'ceil': math.ceil,
        'pi': math.pi, 'e': math.e,
        'radians': math.radians, 'degrees': math.degrees
    }
    
    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool(
    name="current_time",
    description="获取当前日期和时间",
    category=ToolCategory.SYSTEM,
    tags={"time", "datetime"}
)
def current_time(format: str = "%Y-%m-%d %H:%M:%S", timezone: str = "local") -> str:
    """
    获取当前时间
    
    Args:
        format: 时间格式字符串
        timezone: 时区（local 或 UTC）
    """
    from datetime import datetime, timezone as tz
    
    if timezone.upper() == "UTC":
        now = datetime.now(tz.utc)
    else:
        now = datetime.now()
    
    return now.strftime(format)


@tool(
    name="json_tool",
    description="JSON 数据处理工具：解析、格式化、验证、查询",
    category=ToolCategory.DATA,
    tags={"json", "data"}
)
def json_tool(data: str, operation: str = "parse", path: str = "") -> str:
    """
    JSON 数据处理
    
    Args:
        data: JSON 字符串
        operation: 操作类型 (parse, format, validate, query, minify)
        path: JSON 路径（用于 query 操作，如 "user.name"）
    """
    try:
        if operation == "validate":
            json.loads(data)
            return "JSON 格式有效"
        
        parsed = json.loads(data)
        
        if operation == "parse" or operation == "format":
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        elif operation == "minify":
            return json.dumps(parsed, separators=(',', ':'), ensure_ascii=False)
        elif operation == "query":
            result = parsed
            for key in path.split('.'):
                if key:
                    if isinstance(result, dict):
                        result = result.get(key)
                    elif isinstance(result, list) and key.isdigit():
                        result = result[int(key)]
                    else:
                        return f"无法访问路径: {path}"
            return json.dumps(result, indent=2, ensure_ascii=False)
        else:
            return f"未知操作: {operation}"
    except json.JSONDecodeError as e:
        return f"JSON 解析错误: {str(e)}"


@tool(
    name="string_tool",
    description="字符串处理工具",
    category=ToolCategory.DATA,
    tags={"string", "text"}
)
def string_tool(text: str, operation: str, params: str = "") -> str:
    """
    字符串处理
    
    Args:
        text: 输入文本
        operation: 操作 (upper, lower, title, split, replace, count, trim, reverse, repeat, truncate)
        params: 操作参数
    """
    try:
        if operation == "upper":
            return text.upper()
        elif operation == "lower":
            return text.lower()
        elif operation == "title":
            return text.title()
        elif operation == "split":
            sep = params or " "
            return json.dumps(text.split(sep), ensure_ascii=False)
        elif operation == "replace":
            parts = params.split("|", 1)
            if len(parts) == 2:
                return text.replace(parts[0], parts[1])
            return "参数格式: old|new"
        elif operation == "count":
            if params:
                return str(text.count(params))
            return str(len(text))
        elif operation == "trim":
            return text.strip()
        elif operation == "reverse":
            return text[::-1]
        elif operation == "repeat":
            n = int(params) if params else 2
            return text * n
        elif operation == "truncate":
            n = int(params) if params else 100
            return text[:n] + ("..." if len(text) > n else "")
        else:
            return f"未知操作: {operation}"
    except Exception as e:
        return f"处理错误: {str(e)}"


@tool(
    name="list_tool",
    description="列表/数组处理工具",
    category=ToolCategory.DATA,
    tags={"list", "array"}
)
def list_tool(items: str, operation: str, params: str = "") -> str:
    """
    列表处理
    
    Args:
        items: 逗号分隔的列表或 JSON 数组
        operation: 操作 (sort, reverse, unique, count, sum, avg, min, max, join, filter, slice)
        params: 操作参数
    """
    try:
        # 尝试解析 JSON 数组
        try:
            item_list = json.loads(items)
            if not isinstance(item_list, list):
                item_list = [item_list]
        except json.JSONDecodeError:
            item_list = [x.strip() for x in items.split(",")]
        
        if operation == "sort":
            try:
                sorted_list = sorted(item_list, key=float)
            except (ValueError, TypeError):
                sorted_list = sorted(item_list)
            return json.dumps(sorted_list, ensure_ascii=False)
        elif operation == "reverse":
            return json.dumps(list(reversed(item_list)), ensure_ascii=False)
        elif operation == "unique":
            seen = []
            for item in item_list:
                if item not in seen:
                    seen.append(item)
            return json.dumps(seen, ensure_ascii=False)
        elif operation == "count":
            return str(len(item_list))
        elif operation == "sum":
            return str(sum(float(x) for x in item_list))
        elif operation == "avg":
            nums = [float(x) for x in item_list]
            return str(sum(nums) / len(nums))
        elif operation == "min":
            try:
                return str(min(float(x) for x in item_list))
            except ValueError:
                return str(min(item_list))
        elif operation == "max":
            try:
                return str(max(float(x) for x in item_list))
            except ValueError:
                return str(max(item_list))
        elif operation == "join":
            sep = params or ", "
            return sep.join(str(x) for x in item_list)
        elif operation == "filter":
            # 简单过滤：保留包含 params 的项
            filtered = [x for x in item_list if params in str(x)]
            return json.dumps(filtered, ensure_ascii=False)
        elif operation == "slice":
            parts = params.split(":")
            start = int(parts[0]) if parts[0] else None
            end = int(parts[1]) if len(parts) > 1 and parts[1] else None
            return json.dumps(item_list[start:end], ensure_ascii=False)
        else:
            return f"未知操作: {operation}"
    except Exception as e:
        return f"处理错误: {str(e)}"


@tool(
    name="hash_tool",
    description="计算文本的哈希值",
    category=ToolCategory.COMPUTATION,
    tags={"hash", "crypto"}
)
def hash_tool(text: str, algorithm: str = "md5") -> str:
    """
    计算哈希值
    
    Args:
        text: 输入文本
        algorithm: 哈希算法 (md5, sha1, sha256, sha512)
    """
    import hashlib
    
    algorithms = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha256": hashlib.sha256,
        "sha512": hashlib.sha512
    }
    
    if algorithm not in algorithms:
        return f"不支持的算法: {algorithm}。支持: {', '.join(algorithms.keys())}"
    
    hash_func = algorithms[algorithm]
    result = hash_func(text.encode()).hexdigest()
    return f"{algorithm.upper()}: {result}"


@tool(
    name="uuid_gen",
    description="生成 UUID",
    category=ToolCategory.SYSTEM,
    tags={"uuid", "id"}
)
def uuid_gen(uuid_version: int = 4, num_uuids: int = 1) -> str:
    """
    生成 UUID
    
    Args:
        uuid_version: UUID 版本 (1 或 4)
        num_uuids: 生成数量
    """
    results = []
    for _ in range(min(num_uuids, 10)):
        if uuid_version == 1:
            results.append(str(uuid.uuid1()))
        else:
            results.append(str(uuid.uuid4()))
    
    return "\n".join(results) if num_uuids > 1 else results[0]


@tool(
    name="base64_tool",
    description="Base64 编码/解码",
    category=ToolCategory.DATA,
    tags={"encoding", "base64"}
)
def base64_tool(data: str, operation: str = "encode") -> str:
    """
    Base64 编码/解码
    
    Args:
        data: 输入数据
        operation: 操作 (encode, decode)
    """
    import base64
    
    try:
        if operation == "encode":
            result = base64.b64encode(data.encode()).decode()
            return f"Base64 编码: {result}"
        elif operation == "decode":
            result = base64.b64decode(data).decode()
            return f"Base64 解码: {result}"
        else:
            return f"未知操作: {operation}"
    except Exception as e:
        return f"处理错误: {str(e)}"


@tool(
    name="url_tool",
    description="URL 编码/解码和解析",
    category=ToolCategory.DATA,
    tags={"url", "encoding"}
)
def url_tool(url: str, operation: str = "parse") -> str:
    """
    URL 处理
    
    Args:
        url: URL 字符串
        operation: 操作 (parse, encode, decode)
    """
    from urllib.parse import urlparse, quote, unquote, parse_qs
    
    try:
        if operation == "parse":
            parsed = urlparse(url)
            result = {
                "scheme": parsed.scheme,
                "netloc": parsed.netloc,
                "path": parsed.path,
                "params": parsed.params,
                "query": dict(parse_qs(parsed.query)),
                "fragment": parsed.fragment
            }
            return json.dumps(result, indent=2, ensure_ascii=False)
        elif operation == "encode":
            return quote(url, safe='')
        elif operation == "decode":
            return unquote(url)
        else:
            return f"未知操作: {operation}"
    except Exception as e:
        return f"处理错误: {str(e)}"


@tool(
    name="regex_tool",
    description="正则表达式匹配和替换",
    category=ToolCategory.DATA,
    tags={"regex", "pattern"}
)
def regex_tool(text: str, pattern: str, operation: str = "match", 
               replacement: str = "") -> str:
    """
    正则表达式处理
    
    Args:
        text: 输入文本
        pattern: 正则表达式
        operation: 操作 (match, search, findall, replace, split)
        replacement: 替换文本（用于 replace 操作）
    """
    try:
        if operation == "match":
            match = re.match(pattern, text)
            return f"匹配: {match.group() if match else '无匹配'}"
        elif operation == "search":
            match = re.search(pattern, text)
            return f"搜索: {match.group() if match else '无匹配'}"
        elif operation == "findall":
            matches = re.findall(pattern, text)
            return json.dumps(matches, ensure_ascii=False)
        elif operation == "replace":
            result = re.sub(pattern, replacement, text)
            return result
        elif operation == "split":
            parts = re.split(pattern, text)
            return json.dumps(parts, ensure_ascii=False)
        else:
            return f"未知操作: {operation}"
    except re.error as e:
        return f"正则表达式错误: {str(e)}"


@tool(
    name="format_number",
    description="数字格式化",
    category=ToolCategory.DATA,
    tags={"number", "format"}
)
def format_number(number: str, format_type: str = "comma", 
                  precision: int = 2) -> str:
    """
    数字格式化
    
    Args:
        number: 数字字符串
        format_type: 格式类型 (comma, percent, currency, scientific, binary, hex, octal)
        precision: 小数精度
    """
    try:
        num = float(number)
        
        if format_type == "comma":
            return f"{num:,.{precision}f}"
        elif format_type == "percent":
            return f"{num * 100:.{precision}f}%"
        elif format_type == "currency":
            return f"${num:,.{precision}f}"
        elif format_type == "scientific":
            return f"{num:.{precision}e}"
        elif format_type == "binary":
            return bin(int(num))
        elif format_type == "hex":
            return hex(int(num))
        elif format_type == "octal":
            return oct(int(num))
        else:
            return f"未知格式: {format_type}"
    except Exception as e:
        return f"格式化错误: {str(e)}"


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 枚举
    'ToolCategory',
    'ToolStatus',
    'ExecutionMode',
    'RetryStrategy',
    'CacheStrategy',
    'ValidationLevel',
    
    # 数据类
    'ToolMetrics',
    'ToolParameter',
    'Tool',
    'ToolHooks',
    
    # 缓存和限流
    'ToolCache',
    'ToolRateLimiter',
    'RetryHandler',
    
    # 注册表和执行器
    'ToolRegistry',
    'ToolExecutor',
    'ToolManager',
    
    # 工厂函数
    'get_global_registry',
    'get_tool_manager',
    
    # 装饰器
    'tool',
    'async_tool',
    'cached_tool',
    'rate_limited_tool',
    'validated_tool',
    'retry_tool',
    
    # 预置工具
    'calculator',
    'current_time',
    'json_tool',
    'string_tool',
    'list_tool',
    'hash_tool',
    'uuid_gen',
    'base64_tool',
    'url_tool',
    'regex_tool',
    'format_number',
]

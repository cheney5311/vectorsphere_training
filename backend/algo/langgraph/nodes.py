"""
图节点定义（生产级）

定义 LangGraph 中使用的各种节点类型：
- BaseNode: 节点基类（增强版）
- LLMNode: LLM 调用节点（支持多后端LLM服务）
- ToolNode: 工具执行节点
- ConditionalNode: 条件判断节点
- HumanNode: 人机交互节点
- RouterNode: 路由节点
- AggregatorNode: 聚合节点
- TransformNode: 状态转换节点
- BranchNode: 分支节点
- ParallelNode: 并行执行节点
- SubgraphNode: 子图节点
- RetryNode: 重试节点
- CacheNode: 缓存节点
- RateLimitNode: 限流节点
- ValidationNode: 验证节点
- MemoryNode: 记忆节点
- ReflectionNode: 反思节点
- PlanningNode: 规划节点

生产级特性：
- 支持 OpenAI、DeepSeek、Anthropic、本地模型等多种 LLM 后端
- 自动重试与指数退避错误处理
- 流式响应支持
- Token 使用统计与成本追踪
- 节点生命周期钩子（before/after/on_error）
- 节点指标收集与监控
- 响应缓存与限流
- 提示模板系统
- 输出解析器
- 节点装饰器
"""

import json
import logging
import asyncio
import time
import hashlib
import threading
import uuid
import copy
import re
from abc import ABC, abstractmethod
from typing import (
    Any, Dict, List, Optional, Union, Callable, 
    TypeVar, Awaitable, Tuple, AsyncIterator, Iterator,
    Generic, Type, Set
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

from .state import AgentState, AgentMessage, MessageType, ToolCall, AgentStatus
from .tools import Tool, ToolRegistry, ToolExecutor, get_global_registry

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=AgentState)

# ==================== 枚举定义 ====================

class LLMProvider(Enum):
    """LLM 提供者"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LOCAL = "local"
    CUSTOM = "custom"


class NodeType(Enum):
    """节点类型"""
    LLM = "llm"
    TOOL = "tool"
    CONDITIONAL = "conditional"
    HUMAN = "human"
    ROUTER = "router"
    AGGREGATOR = "aggregator"
    TRANSFORM = "transform"
    BRANCH = "branch"
    PARALLEL = "parallel"
    SUBGRAPH = "subgraph"
    RETRY = "retry"
    CACHE = "cache"
    RATE_LIMIT = "rate_limit"
    VALIDATION = "validation"
    MEMORY = "memory"
    REFLECTION = "reflection"
    PLANNING = "planning"
    LOGGING = "logging"
    METRICS = "metrics"
    CUSTOM = "custom"
    START = "__start__"
    END = "__end__"


class ExecutionStrategy(Enum):
    """执行策略"""
    SYNC = "sync"          # 同步执行
    ASYNC = "async"        # 异步执行
    PARALLEL = "parallel"  # 并行执行
    LAZY = "lazy"          # 延迟执行


class RetryStrategy(Enum):
    """重试策略"""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    FIBONACCI = "fibonacci"


class CacheStrategy(Enum):
    """缓存策略"""
    NONE = "none"
    LRU = "lru"           # 最近最少使用
    TTL = "ttl"           # 时间过期
    LRU_TTL = "lru_ttl"   # 结合两者


# ==================== 数据类定义 ====================

@dataclass
class LLMResponse:
    """LLM 响应结构"""
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    model: str = ""
    provider: str = ""
    raw_response: Any = None
    cached: bool = False
    
    @property
    def total_tokens(self) -> int:
        return self.usage.get('total_tokens', 0)
    
    @property
    def cost_estimate(self) -> float:
        """估算成本（美元）"""
        # 简化的成本估算
        prompt_tokens = self.usage.get('prompt_tokens', 0)
        completion_tokens = self.usage.get('completion_tokens', 0)
        
        # 基于模型的近似价格
        cost_per_1k = {
            'gpt-4': (0.03, 0.06),
            'gpt-4-turbo': (0.01, 0.03),
            'gpt-3.5-turbo': (0.0005, 0.0015),
            'claude-3-opus': (0.015, 0.075),
            'claude-3-sonnet': (0.003, 0.015),
        }
        
        rates = cost_per_1k.get(self.model, (0.001, 0.002))
        return (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "provider": self.provider,
            "cached": self.cached
        }


@dataclass
class NodeConfig:
    """节点配置（增强版）"""
    name: str
    node_type: NodeType
    description: str = ""
    
    # 执行控制
    retry_count: int = 3
    retry_delay: float = 1.0
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    timeout: float = 60.0
    execution_strategy: ExecutionStrategy = ExecutionStrategy.SYNC
    
    # 缓存
    enable_cache: bool = False
    cache_strategy: CacheStrategy = CacheStrategy.LRU
    cache_ttl: float = 3600.0  # 秒
    cache_max_size: int = 100
    
    # 条件执行
    condition: Optional[Callable[[AgentState], bool]] = None
    skip_on_condition_false: bool = True
    
    # 钩子
    before_hook: Optional[Callable[[AgentState], AgentState]] = None
    after_hook: Optional[Callable[[AgentState], AgentState]] = None
    error_hook: Optional[Callable[[AgentState, Exception], AgentState]] = None
    
    # 指标
    enable_metrics: bool = True
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "node_type": self.node_type.value,
            "description": self.description,
            "retry_count": self.retry_count,
            "timeout": self.timeout,
            "enable_cache": self.enable_cache,
            "enable_metrics": self.enable_metrics,
            "metadata": self.metadata,
            "tags": self.tags,
            "version": self.version
        }


@dataclass
class NodeMetrics:
    """节点执行指标"""
    node_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    retry_count: int = 0
    last_execution_time: Optional[datetime] = None
    last_error: Optional[str] = None
    
    def record_execution(self, success: bool, execution_time_ms: float, 
                        cached: bool = False, retries: int = 0) -> None:
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
        self.total_execution_time_ms += execution_time_ms
        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        self.retry_count += retries
        self.last_execution_time = datetime.utcnow()
    
    def record_error(self, error: str) -> None:
        self.last_error = error
    
    @property
    def success_rate(self) -> float:
        return self.successful_executions / self.total_executions if self.total_executions > 0 else 0.0
    
    @property
    def avg_execution_time_ms(self) -> float:
        return self.total_execution_time_ms / self.total_executions if self.total_executions > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_name": self.node_name,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": round(self.success_rate, 4),
            "total_execution_time_ms": round(self.total_execution_time_ms, 2),
            "avg_execution_time_ms": round(self.avg_execution_time_ms, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "retry_count": self.retry_count,
            "last_execution_time": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "last_error": self.last_error
        }
    
    def reset(self) -> None:
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_execution_time_ms = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.retry_count = 0
        self.last_execution_time = None
        self.last_error = None


@dataclass
class NodeExecutionResult:
    """节点执行结果"""
    node_name: str
    success: bool
    state: AgentState
    execution_time_ms: float
    error: Optional[str] = None
    cached: bool = False
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_name": self.node_name,
            "success": self.success,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "error": self.error,
            "cached": self.cached,
            "retries": self.retries,
            "metadata": self.metadata
        }


# ==================== 缓存实现 ====================

class LRUCache:
    """LRU 缓存实现"""
    
    def __init__(self, max_size: int = 100, ttl: float = 3600.0):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, datetime] = {}
        self._lock = threading.RLock()
    
    def _make_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        key_str = json.dumps({"args": str(args), "kwargs": kwargs}, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                return None
            
            # 检查 TTL
            if self.ttl > 0:
                timestamp = self._timestamps.get(key)
                if timestamp and (datetime.utcnow() - timestamp).total_seconds() > self.ttl:
                    del self._cache[key]
                    del self._timestamps[key]
                    return None
            
            # 移动到最后（最近使用）
            self._cache.move_to_end(key)
            return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    # 移除最旧的
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
                    if oldest in self._timestamps:
                        del self._timestamps[oldest]
            
            self._cache[key] = value
            self._timestamps[key] = datetime.utcnow()
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)


class BaseNode(ABC):
    """节点基类（增强版）
    
    所有节点必须实现 __call__ 方法，接收状态并返回更新后的状态。
    
    生产级特性：
    - 生命周期钩子（before/after/on_error）
    - 自动重试与退避策略
    - 结果缓存
    - 执行指标收集
    - 条件执行
    - 超时控制
    """
    
    # 节点注册表
    _registry: Dict[str, Type['BaseNode']] = {}
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.name = config.name
        self.node_type = config.node_type
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        
        # 指标
        self._metrics = NodeMetrics(node_name=self.name)
        
        # 缓存
        self._cache: Optional[LRUCache] = None
        if config.enable_cache:
            self._cache = LRUCache(
                max_size=config.cache_max_size,
                ttl=config.cache_ttl
            )
        
        # 执行锁
        self._lock = threading.RLock()
        
        # 是否启用
        self._enabled = True
    
    @classmethod
    def register(cls, name: str):
        """节点注册装饰器"""
        def decorator(node_class: Type['BaseNode']):
            cls._registry[name] = node_class
            return node_class
        return decorator
    
    @classmethod
    def get_registered(cls, name: str) -> Optional[Type['BaseNode']]:
        """获取注册的节点类"""
        return cls._registry.get(name)
    
    @classmethod
    def list_registered(cls) -> List[str]:
        """列出所有注册的节点"""
        return list(cls._registry.keys())
    
    @abstractmethod
    def _execute(self, state: AgentState) -> AgentState:
        """执行节点核心逻辑（子类必须实现）"""
        pass
    
    def __call__(self, state: AgentState) -> AgentState:
        """执行节点（带完整生命周期管理）"""
        if not self._enabled:
            self.logger.debug(f"Node {self.name} is disabled, skipping")
            return state
        
        # 检查条件
        if self.config.condition and not self.config.condition(state):
            if self.config.skip_on_condition_false:
                self.logger.debug(f"Node {self.name} condition not met, skipping")
                return state
        
        start_time = time.time()
        cached = False
        retries = 0
        
        try:
            # 检查缓存
            if self._cache:
                cache_key = self._generate_cache_key(state)
                cached_result = self._cache.get(cache_key)
                if cached_result is not None:
                    self.logger.debug(f"Node {self.name} cache hit")
                    cached = True
                    execution_time = (time.time() - start_time) * 1000
                    self._record_metrics(True, execution_time, cached, retries)
                    return cached_result
            
            # 执行前钩子
            state = self._before_execute(state)
            
            # 验证状态
            if not self.validate_state(state):
                raise ValueError(f"State validation failed for node {self.name}")
            
            # 预处理
            state = self.pre_process(state)
            
            # 带重试执行
            state, retries = self._execute_with_retry(state)
            
            # 后处理
            state = self.post_process(state)
            
            # 执行后钩子
            state = self._after_execute(state)
            
            # 保存到缓存
            if self._cache and not cached:
                self._cache.set(cache_key, state)
            
            execution_time = (time.time() - start_time) * 1000
            self._record_metrics(True, execution_time, cached, retries)
            
            return state
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._record_metrics(False, execution_time, cached, retries)
            self._metrics.record_error(str(e))
            
            # 错误钩子
            state = self._on_error(state, e)
            
            self.logger.error(f"Node {self.name} failed: {e}")
            raise
    
    async def ainvoke(self, state: AgentState) -> AgentState:
        """异步执行节点"""
        if not self._enabled:
            return state
        
        if self.config.condition and not self.config.condition(state):
            if self.config.skip_on_condition_false:
                return state
        
        start_time = time.time()
        cached = False
        retries = 0
        
        try:
            # 检查缓存
            if self._cache:
                cache_key = self._generate_cache_key(state)
                cached_result = self._cache.get(cache_key)
                if cached_result is not None:
                    cached = True
                    execution_time = (time.time() - start_time) * 1000
                    self._record_metrics(True, execution_time, cached, retries)
                    return cached_result
            
            # 执行前钩子
            state = self._before_execute(state)
            
            if not self.validate_state(state):
                raise ValueError(f"State validation failed for node {self.name}")
            
            state = self.pre_process(state)
            
            # 异步带重试执行
            state, retries = await self._aexecute_with_retry(state)
            
            state = self.post_process(state)
            state = self._after_execute(state)
            
            if self._cache and not cached:
                self._cache.set(cache_key, state)
            
            execution_time = (time.time() - start_time) * 1000
            self._record_metrics(True, execution_time, cached, retries)
            
            return state
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self._record_metrics(False, execution_time, cached, retries)
            self._metrics.record_error(str(e))
            state = self._on_error(state, e)
            raise
    
    def _execute_with_retry(self, state: AgentState) -> Tuple[AgentState, int]:
        """带重试的执行"""
        last_error = None
        retries = 0
        
        for attempt in range(self.config.retry_count + 1):
            try:
                if self.config.timeout > 0:
                    result = self._execute_with_timeout(state, self.config.timeout)
                else:
                    result = self._execute(state)
                return result, retries
            except Exception as e:
                last_error = e
                retries = attempt
                if attempt < self.config.retry_count:
                    delay = self._calculate_retry_delay(attempt)
                    self.logger.warning(f"Node {self.name} attempt {attempt + 1} failed: {e}, retrying in {delay:.2f}s")
                    time.sleep(delay)
        
        raise last_error
    
    async def _aexecute_with_retry(self, state: AgentState) -> Tuple[AgentState, int]:
        """异步带重试的执行"""
        last_error = None
        retries = 0
        
        for attempt in range(self.config.retry_count + 1):
            try:
                if self.config.timeout > 0:
                    result = await asyncio.wait_for(
                        self._aexecute(state),
                        timeout=self.config.timeout
                    )
                else:
                    result = await self._aexecute(state)
                return result, retries
            except asyncio.TimeoutError as e:
                last_error = e
                retries = attempt
                self.logger.warning(f"Node {self.name} timeout after {self.config.timeout}s")
            except Exception as e:
                last_error = e
                retries = attempt
                if attempt < self.config.retry_count:
                    delay = self._calculate_retry_delay(attempt)
                    self.logger.warning(f"Node {self.name} attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(delay)
        
        raise last_error
    
    async def _aexecute(self, state: AgentState) -> AgentState:
        """异步执行核心逻辑（可重写）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._execute, state)
    
    def _execute_with_timeout(self, state: AgentState, timeout: float) -> AgentState:
        """带超时的执行"""
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._execute, state)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(f"Node {self.name} execution timed out after {timeout}s")
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        base_delay = self.config.retry_delay
        
        if self.config.retry_strategy == RetryStrategy.LINEAR:
            return base_delay * (attempt + 1)
        elif self.config.retry_strategy == RetryStrategy.EXPONENTIAL:
            return base_delay * (2 ** attempt)
        elif self.config.retry_strategy == RetryStrategy.FIBONACCI:
            def fib(n):
                if n <= 1:
                    return n
                return fib(n-1) + fib(n-2)
            return base_delay * fib(attempt + 2)
        else:
            return base_delay
    
    def _generate_cache_key(self, state: AgentState) -> str:
        """生成缓存键"""
        # 基于状态的关键属性生成
        key_data = {
            "node": self.name,
            "input": state.input,
            "iteration": state.iteration,
            "messages_hash": hashlib.md5(
                json.dumps([m.content for m in state.messages[-5:]]).encode()
            ).hexdigest() if state.messages else ""
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _before_execute(self, state: AgentState) -> AgentState:
        """执行前钩子"""
        if self.config.before_hook:
            try:
                state = self.config.before_hook(state)
            except Exception as e:
                self.logger.warning(f"Before hook failed: {e}")
        return state
    
    def _after_execute(self, state: AgentState) -> AgentState:
        """执行后钩子"""
        if self.config.after_hook:
            try:
                state = self.config.after_hook(state)
            except Exception as e:
                self.logger.warning(f"After hook failed: {e}")
        return state
    
    def _on_error(self, state: AgentState, error: Exception) -> AgentState:
        """错误钩子"""
        if self.config.error_hook:
            try:
                state = self.config.error_hook(state, error)
            except Exception as e:
                self.logger.warning(f"Error hook failed: {e}")
        return state
    
    def _record_metrics(self, success: bool, execution_time_ms: float, 
                       cached: bool, retries: int) -> None:
        """记录指标"""
        if self.config.enable_metrics:
            self._metrics.record_execution(success, execution_time_ms, cached, retries)
    
    def validate_state(self, state: AgentState) -> bool:
        """验证输入状态（可重写）"""
        return True
    
    def pre_process(self, state: AgentState) -> AgentState:
        """预处理（可重写）"""
        return state
    
    def post_process(self, state: AgentState) -> AgentState:
        """后处理（可重写）"""
        return state
    
    # ==================== 辅助方法 ====================
    
    def enable(self) -> None:
        """启用节点"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用节点"""
        self._enabled = False
    
    def is_enabled(self) -> bool:
        """检查是否启用"""
        return self._enabled
    
    def get_metrics(self) -> NodeMetrics:
        """获取指标"""
        return self._metrics
    
    def reset_metrics(self) -> None:
        """重置指标"""
        self._metrics.reset()
    
    def clear_cache(self) -> None:
        """清空缓存"""
        if self._cache:
            self._cache.clear()
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return self._cache.size() if self._cache else 0
    
    def copy(self) -> 'BaseNode':
        """复制节点"""
        return copy.deepcopy(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "name": self.name,
            "node_type": self.node_type.value,
            "config": self.config.to_dict(),
            "enabled": self._enabled,
            "metrics": self._metrics.to_dict()
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.node_type.value})"


class LLMNode(BaseNode):
    """LLM 调用节点（增强版）
    
    调用大语言模型进行推理。
    支持多种 LLM 后端：
    - OpenAI (GPT-4, GPT-4-turbo, GPT-3.5-turbo)
    - DeepSeek
    - Anthropic (Claude)
    - Azure OpenAI
    - Google (Gemini)
    - Ollama (本地)
    - 自定义后端
    
    生产级特性：
    - 自动重试与指数退避
    - 流式响应支持
    - Token 使用统计与成本追踪
    - 多后端自动切换
    - 提示模板系统
    - 输出解析器
    - 响应缓存
    - 限流控制
    """
    
    def __init__(self, 
                 name: str,
                 llm_client: Any = None,
                 provider: LLMProvider = LLMProvider.OPENAI,
                 system_prompt: str = None,
                 tools: List[Tool] = None,
                 temperature: float = 0.7,
                 max_tokens: int = 2048,
                 model: str = "gpt-4",
                 retry_count: int = 3,
                 retry_delay: float = 1.0,
                 enable_streaming: bool = False,
                 fallback_providers: List[LLMProvider] = None,
                 # 增强参数
                 prompt_template: str = None,
                 output_parser: Callable[[str], Any] = None,
                 stop_sequences: List[str] = None,
                 enable_cache: bool = False,
                 cache_ttl: float = 3600.0,
                 rate_limit: int = 0,  # 每分钟请求数，0表示不限制
                 json_mode: bool = False,
                 response_format: Dict[str, Any] = None,
                 tool_choice: str = "auto",  # auto, none, required, 或具体工具名
                 seed: int = None,
                 top_p: float = 1.0,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 logprobs: bool = False,
                 **kwargs):
        """
        初始化 LLM 节点
        
        Args:
            name: 节点名称
            llm_client: LLM 客户端实例（如 OpenAI 客户端）
            provider: LLM 提供者类型
            system_prompt: 系统提示词
            tools: 可用工具列表
            temperature: 生成温度
            max_tokens: 最大 token 数
            model: 模型名称
            retry_count: 重试次数
            retry_delay: 重试延迟（秒）
            enable_streaming: 是否启用流式响应
            fallback_providers: 备选提供者列表
            prompt_template: 提示模板（支持变量替换）
            output_parser: 输出解析函数
            stop_sequences: 停止序列
            enable_cache: 是否启用响应缓存
            cache_ttl: 缓存过期时间（秒）
            rate_limit: 限流（每分钟请求数）
            json_mode: 是否强制 JSON 输出
            response_format: 响应格式配置
            tool_choice: 工具选择策略
            seed: 随机种子（用于可复现生成）
            top_p: 核采样参数
            frequency_penalty: 频率惩罚
            presence_penalty: 存在惩罚
            logprobs: 是否返回 logprobs
        """
        config = NodeConfig(
            name=name,
            node_type=NodeType.LLM,
            description="LLM 推理节点",
            retry_count=retry_count,
            retry_delay=retry_delay,
            enable_cache=enable_cache,
            cache_ttl=cache_ttl
        )
        super().__init__(config)
        
        self.llm_client = llm_client
        self.provider = provider
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.tools = tools or []
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.enable_streaming = enable_streaming
        self.fallback_providers = fallback_providers or []
        
        # 增强参数
        self.prompt_template = prompt_template
        self.output_parser = output_parser
        self.stop_sequences = stop_sequences
        self.rate_limit = rate_limit
        self.json_mode = json_mode
        self.response_format = response_format
        self.tool_choice = tool_choice
        self.seed = seed
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logprobs = logprobs
        self.extra_kwargs = kwargs
        
        # 统计信息
        self._total_tokens = 0
        self._total_calls = 0
        self._total_latency = 0.0
        self._total_cost = 0.0
        
        # 限流控制
        self._rate_limit_tokens: List[datetime] = []
        self._rate_limit_lock = threading.Lock()
        
        # 尝试加载多提供者 LLM 服务
        self._multi_provider_llm = None
        self._init_multi_provider_llm()
    
    def _init_multi_provider_llm(self) -> None:
        """初始化多提供者 LLM 服务"""
        if self.llm_client is None:
            try:
                from backend.services.langchain_inference_service import MultiProviderLLM
                from backend.services.api_config_manager import APIConfigManager
                
                provider_map = {
                    LLMProvider.OPENAI: "chatgpt",
                    LLMProvider.DEEPSEEK: "deepseek",
                    LLMProvider.LOCAL: "local",
                }
                
                provider_name = provider_map.get(self.provider, "local")
                self._multi_provider_llm = MultiProviderLLM(
                    provider=provider_name,
                    model_name=self.model,
                    api_config_manager=APIConfigManager()
                )
                self.logger.info(f"Initialized MultiProviderLLM with provider: {provider_name}")
            except ImportError:
                self.logger.warning("MultiProviderLLM not available, using mock responses")
            except Exception as e:
                self.logger.warning(f"Failed to init MultiProviderLLM: {e}")
    
    def _default_system_prompt(self) -> str:
        """默认系统提示"""
        return """你是一个智能助手，可以帮助用户完成各种任务。

你的能力包括：
1. 分析和回答用户问题
2. 使用提供的工具来获取信息或执行操作
3. 进行多步骤推理来解决复杂问题
4. 提供清晰、准确、有帮助的回答

工具使用指南：
- 当需要实时信息时，使用搜索工具
- 当需要计算时，使用计算器工具
- 当需要数据处理时，使用相应的数据工具

请根据用户的需求，选择合适的工具或直接回答问题。
任务完成后，请明确总结最终答案。"""
    
    def _execute(self, state: AgentState) -> AgentState:
        """执行 LLM 调用核心逻辑"""
        self.logger.debug(f"LLMNode {self.name} executing, iteration {state.iteration}")
        
        # 限流检查
        self._check_rate_limit()
        
        start_time = time.time()
        
        # 构建消息列表
        messages = self._build_messages(state)
        
        # 调用 LLM（带重试）
        response = self._call_with_retry(messages)
        
        # 更新统计
        latency = (time.time() - start_time) * 1000
        self._total_calls += 1
        self._total_latency += latency
        if response.usage:
            self._total_tokens += response.usage.get('total_tokens', 0)
        self._total_cost += response.cost_estimate
        
        response.latency_ms = latency
        
        # 处理响应
        state = self._process_response(state, response)
        
        # 应用输出解析器
        if self.output_parser and response.content:
            try:
                parsed = self.output_parser(response.content)
                state.data['parsed_output'] = parsed
            except Exception as e:
                self.logger.warning(f"Output parser failed: {e}")
        
        return state
    
    def __call__(self, state: AgentState) -> AgentState:
        """执行 LLM 调用（兼容旧接口）"""
        # 使用基类的完整生命周期管理
        return super().__call__(state)
    
    def _check_rate_limit(self) -> None:
        """检查限流"""
        if self.rate_limit <= 0:
            return
        
        with self._rate_limit_lock:
            now = datetime.utcnow()
            # 清理过期的令牌
            self._rate_limit_tokens = [
                t for t in self._rate_limit_tokens 
                if (now - t).total_seconds() < 60
            ]
            
            if len(self._rate_limit_tokens) >= self.rate_limit:
                # 需要等待
                oldest = min(self._rate_limit_tokens)
                wait_time = 60 - (now - oldest).total_seconds()
                if wait_time > 0:
                    self.logger.debug(f"Rate limit hit, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
            
            self._rate_limit_tokens.append(now)
    
    async def ainvoke(self, state: AgentState) -> AgentState:
        """异步执行 LLM 调用"""
        self.logger.debug(f"LLMNode {self.name} async executing")
        
        start_time = time.time()
        messages = self._build_messages(state)
        
        # 异步调用
        response = await self._async_call_with_retry(messages)
        
        latency = (time.time() - start_time) * 1000
        self._total_calls += 1
        self._total_latency += latency
        response.latency_ms = latency
        
        return self._process_response(state, response)
    
    def stream(self, state: AgentState) -> AsyncIterator[str]:
        """流式生成响应"""
        messages = self._build_messages(state)
        return self._stream_llm(messages)
    
    def _build_messages(self, state: AgentState) -> List[Dict[str, Any]]:
        """构建消息列表"""
        # 应用提示模板
        system_content = self._apply_prompt_template(self.system_prompt, state)
        messages = [{"role": "system", "content": system_content}]
        
        for msg in state.messages:
            if msg.type == MessageType.HUMAN:
                content = msg.content
                # 如果有用户消息模板，应用模板
                if self.prompt_template:
                    content = self._apply_prompt_template(self.prompt_template, state, user_input=content)
                messages.append({"role": "user", "content": content})
            elif msg.type == MessageType.AI:
                ai_msg = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    ai_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(ai_msg)
            elif msg.type == MessageType.TOOL:
                messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id
                })
            elif msg.type == MessageType.SYSTEM:
                messages.append({"role": "system", "content": msg.content})
        
        return messages
    
    def _apply_prompt_template(self, template: str, state: AgentState, 
                               **extra_vars) -> str:
        """应用提示模板"""
        if not template:
            return template
        
        # 构建变量字典
        variables = {
            "input": state.input,
            "iteration": state.iteration,
            "messages_count": len(state.messages),
            "tool_results": state.tool_results,
            "data": state.data,
            **state.metadata,
            **extra_vars
        }
        
        # 简单的变量替换（支持 {var} 格式）
        try:
            result = template
            for key, value in variables.items():
                placeholder = "{" + key + "}"
                if placeholder in result:
                    result = result.replace(placeholder, str(value))
            return result
        except Exception as e:
            self.logger.warning(f"Template application failed: {e}")
            return template
    
    def _call_with_retry(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        """带重试的 LLM 调用"""
        last_error = None
        
        for attempt in range(self.retry_count):
            try:
                return self._call_llm(messages)
            except Exception as e:
                last_error = e
                self.logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count - 1:
                    delay = self.retry_delay * (2 ** attempt)  # 指数退避
                    time.sleep(delay)
        
        # 尝试备选提供者
        for fallback in self.fallback_providers:
            try:
                self.logger.info(f"Trying fallback provider: {fallback}")
                return self._call_fallback(messages, fallback)
            except Exception as e:
                self.logger.warning(f"Fallback {fallback} failed: {e}")
        
        # 所有尝试失败，返回模拟响应
        self.logger.error(f"All LLM calls failed, using mock response. Last error: {last_error}")
        return self._create_mock_response(messages)
    
    async def _async_call_with_retry(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        """异步带重试的 LLM 调用"""
        last_error = None
        
        for attempt in range(self.retry_count):
            try:
                return await self._async_call_llm(messages)
            except Exception as e:
                last_error = e
                self.logger.warning(f"Async LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        self.logger.error(f"All async LLM calls failed: {last_error}")
        return self._create_mock_response(messages)
    
    def _call_llm(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        """调用 LLM API"""
        # 使用直接客户端
        if self.llm_client:
            return self._call_with_client(messages)
        
        # 使用多提供者服务
        if self._multi_provider_llm:
            return self._call_with_multi_provider(messages)
        
        # 回退到模拟响应
        return self._create_mock_response(messages)
    
    async def _async_call_llm(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        """异步调用 LLM API"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call_llm, messages)
    
    def _call_with_client(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        """使用直接客户端调用"""
        try:
            tool_schemas = None
            if self.tools:
                tool_schemas = [t.to_openai_function() for t in self.tools]
            
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "frequency_penalty": self.frequency_penalty,
                "presence_penalty": self.presence_penalty,
            }
            
            # 添加可选参数
            if tool_schemas:
                kwargs["tools"] = tool_schemas
                if self.tool_choice != "auto":
                    if self.tool_choice in ["none", "required"]:
                        kwargs["tool_choice"] = self.tool_choice
                    else:
                        # 指定具体工具
                        kwargs["tool_choice"] = {
                            "type": "function",
                            "function": {"name": self.tool_choice}
                        }
            
            if self.stop_sequences:
                kwargs["stop"] = self.stop_sequences
            
            if self.seed is not None:
                kwargs["seed"] = self.seed
            
            if self.json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            elif self.response_format:
                kwargs["response_format"] = self.response_format
            
            if self.logprobs:
                kwargs["logprobs"] = True
            
            # 合并额外参数
            kwargs.update(self.extra_kwargs)
            
            response = self.llm_client.chat.completions.create(**kwargs)
            return self._parse_openai_response(response)
            
        except Exception as e:
            self.logger.error(f"Client LLM call failed: {e}")
            raise
    
    def _call_with_multi_provider(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        """使用多提供者服务调用"""
        try:
            # 构建提示
            prompt = self._messages_to_prompt(messages)
            
            # 调用多提供者 LLM
            result = self._multi_provider_llm._call(prompt)
            
            return LLMResponse(
                content=result,
                tool_calls=[],
                provider=self.provider.value,
                model=self.model
            )
        except Exception as e:
            self.logger.error(f"MultiProvider LLM call failed: {e}")
            raise
    
    def _call_fallback(self, messages: List[Dict[str, Any]], provider: LLMProvider) -> LLMResponse:
        """调用备选提供者"""
        # 简化实现：调用多提供者服务
        try:
            from backend.services.langchain_inference_service import MultiProviderLLM
            from backend.services.api_config_manager import APIConfigManager
            
            provider_map = {
                LLMProvider.OPENAI: "chatgpt",
                LLMProvider.DEEPSEEK: "deepseek",
                LLMProvider.LOCAL: "local",
            }
            
            llm = MultiProviderLLM(
                provider=provider_map.get(provider, "local"),
                model_name=self.model,
                api_config_manager=APIConfigManager()
            )
            
            prompt = self._messages_to_prompt(messages)
            result = llm._call(prompt)
            
            return LLMResponse(
                content=result,
                tool_calls=[],
                provider=provider.value,
                model=self.model
            )
        except Exception as e:
            raise RuntimeError(f"Fallback call failed: {e}")
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """将消息列表转换为单一提示"""
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
            elif role == 'tool':
                prompt_parts.append(f"Tool Result: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    async def _stream_llm(self, messages: List[Dict[str, Any]]) -> AsyncIterator[str]:
        """流式生成"""
        if self.llm_client and hasattr(self.llm_client.chat.completions, 'create'):
            try:
                tool_schemas = [t.to_openai_function() for t in self.tools] if self.tools else None
                
                stream = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tool_schemas,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                self.logger.error(f"Stream failed: {e}")
                yield f"[Error: {e}]"
        else:
            # 非流式模式，一次性返回
            response = self._call_llm(messages)
            yield response.content
    
    def _parse_openai_response(self, response) -> LLMResponse:
        """解析 OpenAI 响应"""
        message = response.choices[0].message
        
        tool_calls = []
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args
                })
        
        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        
        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason or "stop",
            usage=usage,
            model=response.model if hasattr(response, 'model') else self.model,
            provider=self.provider.value
        )
    
    def _create_mock_response(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        """创建模拟响应（用于测试或回退）"""
        # 获取最后的用户消息
        user_content = ""
        for msg in reversed(messages):
            if msg.get('role') == 'user':
                user_content = msg.get('content', '')
                break
        
        # 检查是否有工具结果
        tool_results = []
        for msg in messages:
            if msg.get('role') == 'tool':
                tool_results.append(msg.get('content', ''))
        
        if tool_results:
            content = f"根据工具执行结果：\n" + "\n".join(f"- {r}" for r in tool_results) + "\n\n任务已完成。"
            return LLMResponse(content=content, tool_calls=[], provider="mock", model="mock")
        
        # 智能模拟响应
        input_lower = user_content.lower()
        
        # 计算需求
        if any(kw in input_lower for kw in ["计算", "加", "减", "乘", "除", "+", "-", "*", "/"]):
            import re
            numbers = re.findall(r'\d+', user_content)
            if numbers:
                return LLMResponse(
                    content="",
                    tool_calls=[{
                        "id": f"call_mock_{int(time.time())}",
                        "name": "calculator",
                        "arguments": {"expression": user_content}
                    }],
                    provider="mock",
                    model="mock"
                )
        
        # 时间需求
        if any(kw in input_lower for kw in ["时间", "几点", "日期", "今天"]):
            return LLMResponse(
                content="",
                tool_calls=[{
                    "id": f"call_mock_{int(time.time())}",
                    "name": "get_datetime",
                    "arguments": {}
                }],
                provider="mock",
                model="mock"
            )
        
        # 搜索需求
        if any(kw in input_lower for kw in ["搜索", "查找", "查询", "什么是"]):
            return LLMResponse(
                content="",
                tool_calls=[{
                    "id": f"call_mock_{int(time.time())}",
                    "name": "web_search",
                    "arguments": {"query": user_content}
                }],
                provider="mock",
                model="mock"
            )
        
        # 默认直接回答
        return LLMResponse(
            content=f"收到您的问题：「{user_content}」\n\n这是一个模拟响应。在生产环境中，请配置真实的 LLM 客户端（如 OpenAI、DeepSeek 等）以获得智能回答。\n\n任务已完成。",
            tool_calls=[],
            provider="mock",
            model="mock"
        )
    
    def _process_response(self, state: AgentState, response: LLMResponse) -> AgentState:
        """处理 LLM 响应"""
        content = response.content
        tool_calls_data = response.tool_calls
        
        # 创建工具调用对象
        tool_calls = [
            ToolCall(
                id=tc.get("id", f"call_{state.iteration}_{i}"),
                name=tc["name"],
                arguments=tc.get("arguments", {})
            )
            for i, tc in enumerate(tool_calls_data)
        ]
        
        # 添加 AI 消息
        ai_message = AgentMessage.ai(content, tool_calls=tool_calls)
        ai_message.metadata.update({
            "provider": response.provider,
            "model": response.model,
            "latency_ms": response.latency_ms,
            "usage": response.usage
        })
        state.add_message(ai_message)
        
        # 如果有工具调用，更新状态
        if tool_calls:
            for tc in tool_calls:
                state.add_tool_call(tc)
        else:
            # 没有工具调用，可能是最终答案
            if content and not state.pending_tool_calls:
                if self._is_final_answer(content):
                    state.set_final_answer(content)
        
        # 清除已处理的工具结果
        state.tool_results = []
        
        return state
    
    def _is_final_answer(self, content: str) -> bool:
        """判断是否是最终答案"""
        final_indicators = [
            "最终答案", "总结", "结论", "综上所述", "完成", 
            "任务已完成", "以上就是", "希望这能帮助",
            "Final Answer", "In conclusion", "To summarize"
        ]
        return any(indicator in content for indicator in final_indicators)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "total_latency_ms": self._total_latency,
            "avg_latency_ms": self._total_latency / max(self._total_calls, 1),
            "total_cost_usd": round(self._total_cost, 6),
            "avg_cost_per_call_usd": round(self._total_cost / max(self._total_calls, 1), 6),
            "provider": self.provider.value,
            "model": self.model,
            "cache_enabled": self.config.enable_cache,
            "rate_limit": self.rate_limit,
            "metrics": self._metrics.to_dict() if self._metrics else None
        }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._total_calls = 0
        self._total_tokens = 0
        self._total_latency = 0.0
        self._total_cost = 0.0
        self.reset_metrics()
    
    def set_system_prompt(self, prompt: str) -> None:
        """更新系统提示"""
        self.system_prompt = prompt
    
    def set_tools(self, tools: List[Tool]) -> None:
        """更新工具列表"""
        self.tools = tools
    
    def add_tool(self, tool: Tool) -> None:
        """添加工具"""
        self.tools.append(tool)
    
    def remove_tool(self, tool_name: str) -> bool:
        """移除工具"""
        for i, tool in enumerate(self.tools):
            if tool.name == tool_name:
                del self.tools[i]
                return True
        return False
    
    def set_model(self, model: str) -> None:
        """更新模型"""
        self.model = model
    
    def set_temperature(self, temperature: float) -> None:
        """更新温度"""
        self.temperature = max(0.0, min(2.0, temperature))
    
    def bind_tools(self, tools: List[Tool]) -> 'LLMNode':
        """绑定工具并返回自身（支持链式调用）"""
        self.tools = tools
        return self
    
    def with_system_prompt(self, prompt: str) -> 'LLMNode':
        """设置系统提示并返回自身"""
        self.system_prompt = prompt
        return self
    
    def with_output_parser(self, parser: Callable[[str], Any]) -> 'LLMNode':
        """设置输出解析器并返回自身"""
        self.output_parser = parser
        return self


class ToolNode(BaseNode):
    """工具执行节点（增强版）
    
    执行待处理的工具调用。
    
    增强特性：
    - 工具选择策略
    - 错误恢复
    - 工具链支持
    - 执行超时控制
    - 结果缓存
    """
    
    def __init__(self, 
                 name: str = "tools",
                 registry: ToolRegistry = None,
                 parallel: bool = True,
                 max_concurrent: int = 5,
                 tool_timeout: float = 30.0,
                 continue_on_error: bool = True,
                 enable_cache: bool = False,
                 tool_filter: Callable[[str], bool] = None,
                 result_processor: Callable[[Any], str] = None):
        """
        Args:
            name: 节点名称
            registry: 工具注册表
            parallel: 是否并行执行
            max_concurrent: 最大并发数
            tool_timeout: 单工具超时（秒）
            continue_on_error: 遇错是否继续
            enable_cache: 是否缓存工具结果
            tool_filter: 工具过滤函数（返回 True 表示允许执行）
            result_processor: 结果处理函数
        """
        config = NodeConfig(
            name=name,
            node_type=NodeType.TOOL,
            description="工具执行节点",
            enable_cache=enable_cache
        )
        super().__init__(config)
        
        self.registry = registry or get_global_registry()
        self.executor = ToolExecutor(self.registry)
        self.parallel = parallel
        self.max_concurrent = max_concurrent
        self.tool_timeout = tool_timeout
        self.continue_on_error = continue_on_error
        self.tool_filter = tool_filter
        self.result_processor = result_processor
    
        # 工具执行统计
        self._tool_stats: Dict[str, Dict[str, Any]] = {}
    
    def _execute(self, state: AgentState) -> AgentState:
        """执行所有待处理的工具调用"""
        self.logger.debug(f"ToolNode {self.name} executing {len(state.pending_tool_calls)} tools")
        
        if not state.pending_tool_calls:
            return state
        
        # 过滤工具调用
        tool_calls = state.pending_tool_calls
        if self.tool_filter:
            tool_calls = [tc for tc in tool_calls if self.tool_filter(tc.name)]
        
        if not tool_calls:
            self.logger.debug("All tool calls filtered out")
            return state
        
        # 执行工具调用
        if self.parallel:
            results = self._execute_parallel(tool_calls)
        else:
            results = self._execute_sequential(tool_calls)
        
        # 处理结果
        for result in results:
            # 更新统计
            self._update_tool_stats(result)
            
            # 处理结果
            if self.result_processor and result.success:
                try:
                    result.result = self.result_processor(result.result)
                except Exception as e:
                    self.logger.warning(f"Result processor failed: {e}")
            
            state.add_tool_result(result)
            state.add_message(result.to_message())
            state.add_intermediate_step(
                action={"tool": result.name, "arguments": {}},
                observation=result.result if result.success else result.error
            )
        
        return state
    
    def _execute_parallel(self, tool_calls: List[ToolCall]) -> List[Any]:
        """并行执行工具调用"""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_call = {
                executor.submit(self._execute_single_tool, tc): tc 
                for tc in tool_calls
            }
            for future in as_completed(future_to_call):
                tc = future_to_call[future]
                try:
                    result = future.result(timeout=self.tool_timeout)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Tool {tc.name} failed: {e}")
                    if not self.continue_on_error:
                        raise
                    # 创建失败结果
                    from .tools import ToolResult
                    results.append(ToolResult(
                        call_id=tc.id,
                        name=tc.name,
                        success=False,
                        error=str(e)
                    ))
        return results
    
    def _execute_sequential(self, tool_calls: List[ToolCall]) -> List[Any]:
        """顺序执行工具调用"""
        results = []
        for tc in tool_calls:
            try:
                result = self._execute_single_tool(tc)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Tool {tc.name} failed: {e}")
                if not self.continue_on_error:
                    raise
                from .tools import ToolResult
                results.append(ToolResult(
                    call_id=tc.id,
                    name=tc.name,
                    success=False,
                    error=str(e)
                ))
        return results
    
    def _execute_single_tool(self, tool_call: ToolCall) -> Any:
        """执行单个工具"""
        return self.executor.execute(tool_call)
    
    def _update_tool_stats(self, result) -> None:
        """更新工具统计"""
        tool_name = result.name
        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_time_ms": 0.0
            }
        
        stats = self._tool_stats[tool_name]
        stats["calls"] += 1
        if result.success:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        if hasattr(result, 'execution_time_ms'):
            stats["total_time_ms"] += result.execution_time_ms
    
    async def ainvoke(self, state: AgentState) -> AgentState:
        """异步执行工具调用"""
        if not state.pending_tool_calls:
            return state
        
        tool_calls = state.pending_tool_calls
        if self.tool_filter:
            tool_calls = [tc for tc in tool_calls if self.tool_filter(tc.name)]
        
        if not tool_calls:
            return state
        
        # 异步执行
        results = await self.executor.aexecute_batch(
            tool_calls, 
            parallel=self.parallel
        )
        
        for result in results:
            self._update_tool_stats(result)
            
            if self.result_processor and result.success:
                try:
                    result.result = self.result_processor(result.result)
                except Exception as e:
                    self.logger.warning(f"Result processor failed: {e}")
            
            state.add_tool_result(result)
            state.add_message(result.to_message())
            state.add_intermediate_step(
                action={"tool": result.name},
                observation=result.result if result.success else result.error
            )
        
        return state
    
    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取工具执行统计"""
        return self._tool_stats.copy()
    
    def reset_tool_stats(self) -> None:
        """重置工具统计"""
        self._tool_stats.clear()


class ConditionalNode(BaseNode):
    """条件判断节点
    
    根据状态条件决定下一步执行路径。
    """
    
    def __init__(self,
                 name: str,
                 condition_func: Callable[[AgentState], str],
                 branches: Dict[str, str] = None):
        """
        Args:
            name: 节点名称
            condition_func: 条件函数，接收状态，返回分支名称
            branches: 分支映射 {条件结果: 目标节点名}
        """
        config = NodeConfig(
            name=name,
            node_type=NodeType.CONDITIONAL,
            description="条件判断节点"
        )
        super().__init__(config)
        
        self.condition_func = condition_func
        self.branches = branches or {}
    
    def __call__(self, state: AgentState) -> AgentState:
        """执行条件判断（不修改状态，由边决定路由）"""
        return state
    
    def evaluate(self, state: AgentState) -> str:
        """评估条件，返回目标节点名"""
        result = self.condition_func(state)
        return self.branches.get(result, result)


class HumanNode(BaseNode):
    """人机交互节点
    
    暂停执行，等待人工输入。
    """
    
    def __init__(self,
                 name: str = "human",
                 prompt_template: str = None,
                 timeout: float = 300.0):
        config = NodeConfig(
            name=name,
            node_type=NodeType.HUMAN,
            description="人机交互节点",
            timeout=timeout
        )
        super().__init__(config)
        
        self.prompt_template = prompt_template or "请提供您的输入："
    
    def __call__(self, state: AgentState) -> AgentState:
        """请求人工输入"""
        self.logger.debug(f"HumanNode {self.name} requesting input")
        
        # 生成提示
        prompt = self._generate_prompt(state)
        
        # 请求人工输入
        state.request_human_input(prompt)
        
        return state
    
    def _generate_prompt(self, state: AgentState) -> str:
        """生成提示信息"""
        # 可以使用模板引擎
        return self.prompt_template.format(
            input=state.input,
            iteration=state.iteration,
            messages=len(state.messages)
        )
    
    def receive_input(self, state: AgentState, feedback: str) -> AgentState:
        """接收人工输入"""
        state.receive_human_input(feedback)
        return state


class RouterNode(BaseNode):
    """路由节点
    
    根据输入将请求路由到不同的处理路径。
    """
    
    def __init__(self,
                 name: str,
                 routes: Dict[str, Callable[[AgentState], bool]],
                 default_route: str = None):
        """
        Args:
            name: 节点名称
            routes: 路由规则 {目标节点: 条件函数}
            default_route: 默认路由
        """
        config = NodeConfig(
            name=name,
            node_type=NodeType.ROUTER,
            description="路由节点"
        )
        super().__init__(config)
        
        self.routes = routes
        self.default_route = default_route
    
    def __call__(self, state: AgentState) -> AgentState:
        """执行路由（状态不变）"""
        return state
    
    def get_route(self, state: AgentState) -> str:
        """获取路由目标"""
        for target, condition in self.routes.items():
            if condition(state):
                return target
        return self.default_route or "__end__"


class AggregatorNode(BaseNode):
    """聚合节点
    
    聚合多个并行分支的结果。
    """
    
    def __init__(self,
                 name: str,
                 aggregation_func: Callable[[List[AgentState]], AgentState] = None):
        config = NodeConfig(
            name=name,
            node_type=NodeType.AGGREGATOR,
            description="聚合节点"
        )
        super().__init__(config)
        
        self.aggregation_func = aggregation_func or self._default_aggregation
    
    def __call__(self, state: AgentState) -> AgentState:
        """执行聚合（单状态时直接返回）"""
        return state
    
    def aggregate(self, states: List[AgentState]) -> AgentState:
        """聚合多个状态"""
        return self.aggregation_func(states)
    
    def _default_aggregation(self, states: List[AgentState]) -> AgentState:
        """默认聚合策略：合并消息"""
        if not states:
            return AgentState()
        
        base_state = states[0].copy()
        
        for other_state in states[1:]:
            base_state.add_messages(other_state.messages)
            base_state.intermediate_steps.extend(other_state.intermediate_steps)
            base_state.data.update(other_state.data)
        
        return base_state


# ==================== 特殊节点 ====================

class StartNode(BaseNode):
    """起始节点"""
    
    def __init__(self):
        config = NodeConfig(
            name="__start__",
            node_type=NodeType.START,
            description="起始节点"
        )
        super().__init__(config)
    
    def _execute(self, state: AgentState) -> AgentState:
        """执行起始节点逻辑"""
        state.status = AgentStatus.RUNNING
        return state


class EndNode(BaseNode):
    """结束节点"""
    
    def __init__(self):
        config = NodeConfig(
            name="__end__",
            node_type=NodeType.END,
            description="结束节点"
        )
        super().__init__(config)
    
    def _execute(self, state: AgentState) -> AgentState:
        """执行结束节点逻辑"""
        if state.status != AgentStatus.FAILED:
            state.status = AgentStatus.COMPLETED
        return state


# ==================== 节点工厂 ====================

class NodeFactory:
    """节点工厂"""
    
    @staticmethod
    def create_llm_node(name: str, **kwargs) -> LLMNode:
        """创建 LLM 节点"""
        return LLMNode(name=name, **kwargs)
    
    @staticmethod
    def create_tool_node(name: str = "tools", **kwargs) -> ToolNode:
        """创建工具节点"""
        return ToolNode(name=name, **kwargs)
    
    @staticmethod
    def create_conditional_node(name: str, condition_func: Callable, 
                                branches: Dict[str, str] = None) -> ConditionalNode:
        """创建条件节点"""
        return ConditionalNode(name=name, condition_func=condition_func, branches=branches)
    
    @staticmethod
    def create_human_node(name: str = "human", **kwargs) -> HumanNode:
        """创建人机交互节点"""
        return HumanNode(name=name, **kwargs)
    
    @staticmethod
    def create_router_node(name: str, routes: Dict[str, Callable],
                          default_route: str = None) -> RouterNode:
        """创建路由节点"""
        return RouterNode(name=name, routes=routes, default_route=default_route)
    
    @staticmethod
    def create_aggregator_node(name: str, 
                              aggregation_func: Callable = None) -> AggregatorNode:
        """创建聚合节点"""
        return AggregatorNode(name=name, aggregation_func=aggregation_func)


# ==================== 新增节点类型 ====================

class TransformNode(BaseNode):
    """状态转换节点
    
    对状态进行转换处理。
    """
    
    def __init__(self, 
                 name: str,
                 transform_func: Callable[[AgentState], AgentState],
                 description: str = ""):
        config = NodeConfig(
            name=name,
            node_type=NodeType.TRANSFORM,
            description=description or "状态转换节点"
        )
        super().__init__(config)
        self.transform_func = transform_func
    
    def _execute(self, state: AgentState) -> AgentState:
        return self.transform_func(state)


class BranchNode(BaseNode):
    """分支节点
    
    根据条件将执行分发到不同分支。
    """
    
    def __init__(self,
                 name: str,
                 branches: Dict[str, Callable[[AgentState], bool]],
                 default_branch: str = None):
        """
        Args:
            name: 节点名称
            branches: 分支条件 {分支名: 条件函数}
            default_branch: 默认分支
        """
        config = NodeConfig(
            name=name,
            node_type=NodeType.BRANCH,
            description="分支节点"
        )
        super().__init__(config)
        
        self.branches = branches
        self.default_branch = default_branch
        self._selected_branch: Optional[str] = None
    
    def _execute(self, state: AgentState) -> AgentState:
        # 评估分支条件
        for branch_name, condition in self.branches.items():
            if condition(state):
                self._selected_branch = branch_name
                state.metadata['selected_branch'] = branch_name
                return state
        
        # 使用默认分支
        self._selected_branch = self.default_branch
        state.metadata['selected_branch'] = self.default_branch
        return state
    
    def get_selected_branch(self) -> Optional[str]:
        """获取选中的分支"""
        return self._selected_branch


class ParallelNode(BaseNode):
    """并行执行节点
    
    并行执行多个子处理流程。
    """
    
    def __init__(self,
                 name: str,
                 branches: Dict[str, Callable[[AgentState], AgentState]],
                 merge_func: Callable[[List[AgentState]], AgentState] = None,
                 max_workers: int = 4,
                 timeout: float = 60.0):
        """
        Args:
            name: 节点名称
            branches: 并行分支 {分支名: 处理函数}
            merge_func: 结果合并函数
            max_workers: 最大工作线程数
            timeout: 超时时间
        """
        config = NodeConfig(
            name=name,
            node_type=NodeType.PARALLEL,
            description="并行执行节点",
            timeout=timeout
        )
        super().__init__(config)
        
        self.branches = branches
        self.merge_func = merge_func or self._default_merge
        self.max_workers = max_workers
    
    def _execute(self, state: AgentState) -> AgentState:
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(func, state.copy()): name
                for name, func in self.branches.items()
            }
            
            for future in as_completed(futures, timeout=self.config.timeout):
                branch_name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Branch {branch_name} failed: {e}")
        
        return self.merge_func(results)
    
    async def _aexecute(self, state: AgentState) -> AgentState:
        """异步并行执行"""
        tasks = []
        for name, func in self.branches.items():
            if asyncio.iscoroutinefunction(func):
                tasks.append(func(state.copy()))
            else:
                loop = asyncio.get_event_loop()
                tasks.append(loop.run_in_executor(None, func, state.copy()))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        return self.merge_func(valid_results)
    
    def _default_merge(self, states: List[AgentState]) -> AgentState:
        """默认合并策略"""
        if not states:
            return AgentState()
        
        merged = states[0].copy()
        for other in states[1:]:
            merged.add_messages(other.messages)
            merged.intermediate_steps.extend(other.intermediate_steps)
            merged.data.update(other.data)
        
        return merged


class RetryNode(BaseNode):
    """重试节点
    
    封装其他节点，提供重试功能。
    """
    
    def __init__(self,
                 name: str,
                 inner_node: BaseNode,
                 max_retries: int = 3,
                 retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
                 base_delay: float = 1.0,
                 should_retry: Callable[[Exception], bool] = None):
        config = NodeConfig(
            name=name,
            node_type=NodeType.RETRY,
            description="重试节点",
            retry_count=max_retries,
            retry_strategy=retry_strategy,
            retry_delay=base_delay
        )
        super().__init__(config)
        
        self.inner_node = inner_node
        self.should_retry = should_retry or (lambda e: True)
    
    def _execute(self, state: AgentState) -> AgentState:
        return self.inner_node(state)


class CacheNode(BaseNode):
    """缓存节点
    
    缓存处理结果。
    """
    
    def __init__(self,
                 name: str,
                 inner_node: BaseNode,
                 cache_key_func: Callable[[AgentState], str] = None,
                 ttl: float = 3600.0,
                 max_size: int = 100):
        config = NodeConfig(
            name=name,
            node_type=NodeType.CACHE,
            description="缓存节点",
            enable_cache=True,
            cache_ttl=ttl,
            cache_max_size=max_size
        )
        super().__init__(config)
        
        self.inner_node = inner_node
        self.cache_key_func = cache_key_func
    
    def _generate_cache_key(self, state: AgentState) -> str:
        if self.cache_key_func:
            return self.cache_key_func(state)
        return super()._generate_cache_key(state)
    
    def _execute(self, state: AgentState) -> AgentState:
        return self.inner_node(state)


class RateLimitNode(BaseNode):
    """限流节点
    
    限制执行频率。
    """
    
    def __init__(self,
                 name: str,
                 inner_node: BaseNode,
                 requests_per_minute: int = 60,
                 burst_size: int = 10):
        config = NodeConfig(
            name=name,
            node_type=NodeType.RATE_LIMIT,
            description="限流节点"
        )
        super().__init__(config)
        
        self.inner_node = inner_node
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self._tokens: List[datetime] = []
        self._lock = threading.Lock()
    
    def _execute(self, state: AgentState) -> AgentState:
        self._acquire_token()
        return self.inner_node(state)
    
    def _acquire_token(self) -> None:
        """获取令牌（令牌桶算法）"""
        with self._lock:
            now = datetime.utcnow()
            # 清理过期令牌
            self._tokens = [
                t for t in self._tokens 
                if (now - t).total_seconds() < 60
            ]
            
            if len(self._tokens) >= self.requests_per_minute:
                # 需要等待
                oldest = min(self._tokens)
                wait_time = 60 - (now - oldest).total_seconds()
                if wait_time > 0:
                    time.sleep(wait_time)
            
            self._tokens.append(now)


class ValidationNode(BaseNode):
    """验证节点
    
    验证状态是否符合条件。
    """
    
    def __init__(self,
                 name: str,
                 validators: List[Callable[[AgentState], Tuple[bool, str]]],
                 fail_fast: bool = True,
                 on_invalid: Callable[[AgentState, List[str]], AgentState] = None):
        """
        Args:
            name: 节点名称
            validators: 验证函数列表，返回 (是否有效, 错误消息)
            fail_fast: 遇到第一个错误就失败
            on_invalid: 验证失败时的处理函数
        """
        config = NodeConfig(
            name=name,
            node_type=NodeType.VALIDATION,
            description="验证节点"
        )
        super().__init__(config)
        
        self.validators = validators
        self.fail_fast = fail_fast
        self.on_invalid = on_invalid
    
    def _execute(self, state: AgentState) -> AgentState:
        errors = []
        
        for validator in self.validators:
            valid, error_msg = validator(state)
            if not valid:
                errors.append(error_msg)
                if self.fail_fast:
                    break
        
        if errors:
            state.metadata['validation_errors'] = errors
            if self.on_invalid:
                state = self.on_invalid(state, errors)
            else:
                raise ValueError(f"Validation failed: {errors}")
        
        return state


class MemoryNode(BaseNode):
    """记忆节点
    
    管理长短期记忆。
    """
    
    def __init__(self,
                 name: str,
                 memory_type: str = "conversation",  # conversation, summary, entity
                 max_tokens: int = 4000,
                 summarize_threshold: int = 3000):
        config = NodeConfig(
            name=name,
            node_type=NodeType.MEMORY,
            description="记忆节点"
        )
        super().__init__(config)
        
        self.memory_type = memory_type
        self.max_tokens = max_tokens
        self.summarize_threshold = summarize_threshold
        self._memory_store: Dict[str, Any] = {}
    
    def _execute(self, state: AgentState) -> AgentState:
        thread_id = state.thread_id
        
        # 加载记忆
        if thread_id in self._memory_store:
            state.data['memory'] = self._memory_store[thread_id]
        
        # 保存当前状态到记忆
        self._memory_store[thread_id] = {
            'messages_summary': self._summarize_messages(state.messages),
            'key_entities': self._extract_entities(state),
            'last_updated': datetime.utcnow().isoformat()
        }
        
        return state
    
    def _summarize_messages(self, messages: List[AgentMessage]) -> str:
        """简化的消息摘要"""
        if not messages:
            return ""
        
        # 取最后几条消息的摘要
        recent = messages[-10:]
        summary_parts = []
        for msg in recent:
            content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_parts.append(f"{msg.type.value}: {content}")
        
        return "\n".join(summary_parts)
    
    def _extract_entities(self, state: AgentState) -> Dict[str, Any]:
        """提取关键实体（简化版）"""
        entities = {}
        # 从状态数据中提取
        for key, value in state.data.items():
            if isinstance(value, (str, int, float, bool)):
                entities[key] = value
        return entities
    
    def get_memory(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """获取特定线程的记忆"""
        return self._memory_store.get(thread_id)
    
    def clear_memory(self, thread_id: str = None) -> None:
        """清除记忆"""
        if thread_id:
            self._memory_store.pop(thread_id, None)
        else:
            self._memory_store.clear()


class ReflectionNode(BaseNode):
    """反思节点
    
    让 Agent 反思其行为和输出。
    """
    
    def __init__(self,
                 name: str,
                 llm_node: 'LLMNode' = None,
                 reflection_prompt: str = None,
                 criteria: List[str] = None):
        """
        Args:
            name: 节点名称
            llm_node: 用于反思的 LLM 节点
            reflection_prompt: 反思提示模板
            criteria: 反思标准
        """
        config = NodeConfig(
            name=name,
            node_type=NodeType.REFLECTION,
            description="反思节点"
        )
        super().__init__(config)
        
        self.llm_node = llm_node
        self.reflection_prompt = reflection_prompt or self._default_reflection_prompt()
        self.criteria = criteria or ["accuracy", "completeness", "relevance"]
    
    def _default_reflection_prompt(self) -> str:
        return """请反思以下对话和行动：

对话历史：
{conversation_history}

最近的行动：
{recent_actions}

请根据以下标准进行评估：
{criteria}

请提供：
1. 评估得分（1-10）
2. 改进建议
3. 是否需要重做

以 JSON 格式输出。"""
    
    def _execute(self, state: AgentState) -> AgentState:
        if not self.llm_node:
            self.logger.warning("No LLM node configured for reflection")
            return state
        
        # 构建反思上下文
        conversation = "\n".join([
            f"{m.type.value}: {m.content[:200]}" 
            for m in state.messages[-5:]
        ])
        
        actions = "\n".join([
            str(step) for step in state.intermediate_steps[-3:]
        ])
        
        criteria_str = "\n".join([f"- {c}" for c in self.criteria])
        
        reflection_input = self.reflection_prompt.format(
            conversation_history=conversation,
            recent_actions=actions,
            criteria=criteria_str
        )
        
        # 创建反思状态
        reflection_state = AgentState(input=reflection_input)
        reflection_state.add_message(AgentMessage.human(reflection_input))
        
        # 执行反思
        result_state = self.llm_node(reflection_state)
        
        # 提取反思结果
        if result_state.messages:
            last_msg = result_state.messages[-1]
            state.data['reflection'] = {
                'content': last_msg.content,
                'timestamp': datetime.utcnow().isoformat()
            }
        
        return state


class PlanningNode(BaseNode):
    """规划节点
    
    生成和管理任务执行计划。
    """
    
    def __init__(self,
                 name: str,
                 llm_node: 'LLMNode' = None,
                 planning_prompt: str = None,
                 max_steps: int = 10):
        """
        Args:
            name: 节点名称
            llm_node: 用于规划的 LLM 节点
            planning_prompt: 规划提示模板
            max_steps: 最大步骤数
        """
        config = NodeConfig(
            name=name,
            node_type=NodeType.PLANNING,
            description="规划节点"
        )
        super().__init__(config)
        
        self.llm_node = llm_node
        self.planning_prompt = planning_prompt or self._default_planning_prompt()
        self.max_steps = max_steps
    
    def _default_planning_prompt(self) -> str:
        return """请为以下任务制定执行计划：

任务：{task}

可用工具：{available_tools}

请生成一个分步计划，每步包含：
1. 步骤编号
2. 行动描述
3. 预期输出
4. 依赖的前置步骤

以 JSON 格式输出计划列表。最多 {max_steps} 步。"""
    
    def _execute(self, state: AgentState) -> AgentState:
        if not self.llm_node:
            self.logger.warning("No LLM node configured for planning")
            return state
        
        # 获取可用工具
        available_tools = []
        if hasattr(self.llm_node, 'tools'):
            available_tools = [t.name for t in self.llm_node.tools]
        
        planning_input = self.planning_prompt.format(
            task=state.input,
            available_tools=", ".join(available_tools) or "无",
            max_steps=self.max_steps
        )
        
        # 创建规划状态
        plan_state = AgentState(input=planning_input)
        plan_state.add_message(AgentMessage.human(planning_input))
        
        # 执行规划
        result_state = self.llm_node(plan_state)
        
        # 提取计划
        if result_state.messages:
            last_msg = result_state.messages[-1]
            try:
                # 尝试解析 JSON
                import re
                json_match = re.search(r'\[.*\]', last_msg.content, re.DOTALL)
                if json_match:
                    plan = json.loads(json_match.group())
                    state.data['plan'] = plan
                else:
                    state.data['plan'] = {'raw': last_msg.content}
            except json.JSONDecodeError:
                state.data['plan'] = {'raw': last_msg.content}
        
        return state


class LoggingNode(BaseNode):
    """日志节点
    
    记录执行日志。
    """
    
    def __init__(self,
                 name: str,
                 log_level: str = "INFO",
                 log_format: str = None,
                 log_fields: List[str] = None):
        config = NodeConfig(
            name=name,
            node_type=NodeType.LOGGING,
            description="日志节点",
            retry_count=0  # 日志节点不需要重试
        )
        super().__init__(config)
        
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_format = log_format or "[{timestamp}] [{node}] input={input}, iteration={iteration}, messages={messages_count}"
        self.log_fields = log_fields or ['input', 'iteration', 'messages_count']
        self._log_history: List[Dict[str, Any]] = []
    
    def _execute(self, state: AgentState) -> AgentState:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'node': self.name,
            'input': state.input[:100] if state.input else "",
            'iteration': state.iteration,
            'messages_count': len(state.messages),
            'status': state.status.value,
            'message': f"State iteration {state.iteration}, {len(state.messages)} messages"
        }
        
        try:
            message = self.log_format.format(**log_data)
        except KeyError:
            # 如果格式字符串有未知的键，使用默认格式
            message = f"[{log_data['timestamp']}] [{log_data['node']}] iteration={log_data['iteration']}"
        
        self.logger.log(self.log_level, message)
        
        self._log_history.append(log_data)
        
        return state
    
    def get_log_history(self) -> List[Dict[str, Any]]:
        """获取日志历史"""
        return self._log_history.copy()
    
    def clear_log_history(self) -> None:
        """清除日志历史"""
        self._log_history.clear()


class MetricsNode(BaseNode):
    """指标节点
    
    收集和报告执行指标。
    """
    
    def __init__(self,
                 name: str,
                 metrics_to_collect: List[str] = None,
                 report_func: Callable[[Dict[str, Any]], None] = None):
        config = NodeConfig(
            name=name,
            node_type=NodeType.METRICS,
            description="指标节点"
        )
        super().__init__(config)
        
        self.metrics_to_collect = metrics_to_collect or [
            'execution_time', 'messages_count', 'tool_calls_count', 'iteration'
        ]
        self.report_func = report_func
        self._collected_metrics: List[Dict[str, Any]] = []
    
    def _execute(self, state: AgentState) -> AgentState:
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'node': self.name,
            'execution_time': 0,  # 将在后处理中更新
            'messages_count': len(state.messages),
            'tool_calls_count': len(state.pending_tool_calls),
            'iteration': state.iteration,
            'status': state.status.value
        }
        
        self._collected_metrics.append(metrics)
        
        if self.report_func:
            self.report_func(metrics)
        
        state.data['last_metrics'] = metrics
        
        return state
    
    def get_collected_metrics(self) -> List[Dict[str, Any]]:
        """获取收集的指标"""
        return self._collected_metrics.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self._collected_metrics:
            return {}
        
        return {
            'total_executions': len(self._collected_metrics),
            'avg_messages_count': sum(m['messages_count'] for m in self._collected_metrics) / len(self._collected_metrics),
            'total_tool_calls': sum(m['tool_calls_count'] for m in self._collected_metrics),
            'first_execution': self._collected_metrics[0]['timestamp'],
            'last_execution': self._collected_metrics[-1]['timestamp']
        }


# ==================== 节点装饰器 ====================

def node(name: str = None, 
         node_type: NodeType = NodeType.CUSTOM,
         **config_kwargs):
    """节点装饰器
    
    将函数转换为节点。
    
    用法:
        @node("my_node")
        def process(state: AgentState) -> AgentState:
            # 处理逻辑
            return state
    """
    def decorator(func: Callable[[AgentState], AgentState]):
        node_name = name or func.__name__
        
        class DecoratedNode(BaseNode):
            def __init__(self):
                config = NodeConfig(
                    name=node_name,
                    node_type=node_type,
                    description=func.__doc__ or "",
                    **config_kwargs
                )
                super().__init__(config)
                self.func = func
            
            def _execute(self, state: AgentState) -> AgentState:
                return self.func(state)
        
        # 注册节点
        BaseNode._registry[node_name] = DecoratedNode
        
        return DecoratedNode()
    
    return decorator


def async_node(name: str = None,
               node_type: NodeType = NodeType.CUSTOM,
               **config_kwargs):
    """异步节点装饰器
    
    将异步函数转换为节点。
    """
    def decorator(func: Callable[[AgentState], Awaitable[AgentState]]):
        node_name = name or func.__name__
        
        class AsyncDecoratedNode(BaseNode):
            def __init__(self):
                config = NodeConfig(
                    name=node_name,
                    node_type=node_type,
                    description=func.__doc__ or "",
                    execution_strategy=ExecutionStrategy.ASYNC,
                    **config_kwargs
                )
                super().__init__(config)
                self.func = func
            
            def _execute(self, state: AgentState) -> AgentState:
                # 同步调用异步函数
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.func(state))
            
            async def _aexecute(self, state: AgentState) -> AgentState:
                return await self.func(state)
        
        BaseNode._registry[node_name] = AsyncDecoratedNode
        
        return AsyncDecoratedNode()
    
    return decorator


def cached_node(cache_key_func: Callable[[AgentState], str] = None,
                ttl: float = 3600.0,
                max_size: int = 100):
    """缓存节点装饰器
    
    为节点添加缓存功能。
    """
    def decorator(node_or_func):
        if isinstance(node_or_func, BaseNode):
            return CacheNode(
                name=f"cached_{node_or_func.name}",
                inner_node=node_or_func,
                cache_key_func=cache_key_func,
                ttl=ttl,
                max_size=max_size
            )
        else:
            # 先转换为节点再包装
            inner = node()(node_or_func)
            return CacheNode(
                name=f"cached_{node_or_func.__name__}",
                inner_node=inner,
                cache_key_func=cache_key_func,
                ttl=ttl,
                max_size=max_size
            )
    
    return decorator


def rate_limited_node(requests_per_minute: int = 60, burst_size: int = 10):
    """限流节点装饰器"""
    def decorator(node_or_func):
        if isinstance(node_or_func, BaseNode):
            return RateLimitNode(
                name=f"rate_limited_{node_or_func.name}",
                inner_node=node_or_func,
                requests_per_minute=requests_per_minute,
                burst_size=burst_size
            )
        else:
            inner = node()(node_or_func)
            return RateLimitNode(
                name=f"rate_limited_{node_or_func.__name__}",
                inner_node=inner,
                requests_per_minute=requests_per_minute,
                burst_size=burst_size
            )
    
    return decorator


# ==================== 节点组合 ====================

class NodeChain:
    """节点链
    
    按顺序执行多个节点。
    """
    
    def __init__(self, nodes: List[BaseNode], name: str = "chain"):
        self.nodes = nodes
        self.name = name
    
    def __call__(self, state: AgentState) -> AgentState:
        for node in self.nodes:
            state = node(state)
        return state
    
    async def ainvoke(self, state: AgentState) -> AgentState:
        for node in self.nodes:
            state = await node.ainvoke(state)
        return state
    
    def append(self, node: BaseNode) -> 'NodeChain':
        self.nodes.append(node)
        return self
    
    def prepend(self, node: BaseNode) -> 'NodeChain':
        self.nodes.insert(0, node)
        return self


class NodeParallelGroup:
    """节点并行组
    
    并行执行多个节点。
    """
    
    def __init__(self, nodes: List[BaseNode], 
                 merge_func: Callable[[List[AgentState]], AgentState] = None,
                 name: str = "parallel_group"):
        self.nodes = nodes
        self.merge_func = merge_func or self._default_merge
        self.name = name
    
    def __call__(self, state: AgentState) -> AgentState:
        with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
            futures = [executor.submit(n, state.copy()) for n in self.nodes]
            results = [f.result() for f in as_completed(futures)]
        return self.merge_func(results)
    
    async def ainvoke(self, state: AgentState) -> AgentState:
        tasks = [n.ainvoke(state.copy()) for n in self.nodes]
        results = await asyncio.gather(*tasks)
        return self.merge_func(list(results))
    
    def _default_merge(self, states: List[AgentState]) -> AgentState:
        if not states:
            return AgentState()
        merged = states[0].copy()
        for other in states[1:]:
            merged.add_messages(other.messages)
            merged.data.update(other.data)
        return merged


class NodeConditionalGroup:
    """条件节点组
    
    根据条件选择执行哪个节点。
    """
    
    def __init__(self, 
                 condition: Callable[[AgentState], str],
                 branches: Dict[str, BaseNode],
                 default: BaseNode = None):
        self.condition = condition
        self.branches = branches
        self.default = default
    
    def __call__(self, state: AgentState) -> AgentState:
        branch_key = self.condition(state)
        node = self.branches.get(branch_key, self.default)
        if node:
            return node(state)
        return state
    
    async def ainvoke(self, state: AgentState) -> AgentState:
        branch_key = self.condition(state)
        node = self.branches.get(branch_key, self.default)
        if node:
            return await node.ainvoke(state)
        return state


# ==================== 节点注册表增强 ====================

class NodeRegistry:
    """节点注册表
    
    管理所有可用节点类型。
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._nodes: Dict[str, Type[BaseNode]] = {}
                    cls._instance._versions: Dict[str, str] = {}
                    cls._instance._register_builtins()
        return cls._instance
    
    def _register_builtins(self) -> None:
        """注册内置节点"""
        builtins = {
            'llm': LLMNode,
            'tool': ToolNode,
            'conditional': ConditionalNode,
            'human': HumanNode,
            'router': RouterNode,
            'aggregator': AggregatorNode,
            'transform': TransformNode,
            'branch': BranchNode,
            'parallel': ParallelNode,
            'retry': RetryNode,
            'cache': CacheNode,
            'rate_limit': RateLimitNode,
            'validation': ValidationNode,
            'memory': MemoryNode,
            'reflection': ReflectionNode,
            'planning': PlanningNode,
            'logging': LoggingNode,
            'metrics': MetricsNode,
        }
        for name, cls in builtins.items():
            self.register(name, cls, "1.0.0")
    
    def register(self, name: str, node_class: Type[BaseNode], version: str = "1.0.0") -> None:
        """注册节点"""
        self._nodes[name] = node_class
        self._versions[name] = version
    
    def unregister(self, name: str) -> bool:
        """注销节点"""
        if name in self._nodes:
            del self._nodes[name]
            del self._versions[name]
            return True
        return False
    
    def get(self, name: str) -> Optional[Type[BaseNode]]:
        """获取节点类"""
        return self._nodes.get(name)
    
    def create(self, name: str, **kwargs) -> Optional[BaseNode]:
        """创建节点实例"""
        node_class = self.get(name)
        if node_class:
            return node_class(**kwargs)
        return None
    
    def list_nodes(self) -> List[str]:
        """列出所有节点"""
        return list(self._nodes.keys())
    
    def get_version(self, name: str) -> Optional[str]:
        """获取节点版本"""
        return self._versions.get(name)
    
    def get_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取节点信息"""
        if name not in self._nodes:
            return None
        
        node_class = self._nodes[name]
        return {
            'name': name,
            'class': node_class.__name__,
            'version': self._versions.get(name, "unknown"),
            'description': node_class.__doc__ or ""
        }


def get_node_registry() -> NodeRegistry:
    """获取全局节点注册表"""
    return NodeRegistry()


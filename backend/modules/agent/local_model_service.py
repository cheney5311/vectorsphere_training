"""
本地模型管理服务

该模块提供本地模型的管理和调用功能，支持多种模型类型：
- Ollama 本地模型
- Transformers 模型
- OpenAI API 兼容模型
- vLLM 高性能推理
- TensorRT-LLM 加速推理

扩展功能：
- 与 LangGraph 工具系统集成
- 支持工具调用
- 支持流式输出
- 支持模型指标收集
- 模型池化和负载均衡
- 提示词模板管理
- 高级缓存策略
- 熔断器和重试机制
- 模型健康检查
- 批量和并行生成
"""

import asyncio
import hashlib
import json
import os
import threading
import time
import uuid
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Dict, List, Optional, Any, TypeVar, AsyncIterator
)

# 可选依赖导入
import ollama
import openai
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM

# LangGraph 工具集成
from backend.algo.langgraph.tools import (
    Tool, ToolRegistry, ToolExecutor, ToolManager,
    get_global_registry, get_tool_manager
)
from backend.algo.langgraph.builtin_tools import (
    get_builtin_tools, get_tool_by_name
)
from backend.algo.langgraph.state import (
    ToolCall, ToolResult
)

T = TypeVar('T')
ModelT = TypeVar('ModelT')


# ==================== 枚举定义 ====================

class ModelType(Enum):
    """模型类型枚举"""
    OLLAMA = "ollama"
    TRANSFORMERS = "transformers"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    VLLM = "vllm"
    TENSORRT = "tensorrt"
    GGML = "ggml"
    MOCK = "mock"


class ModelStatus(Enum):
    """模型状态枚举"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    WARMING_UP = "warming_up"
    READY = "ready"
    ERROR = "error"
    UNLOADING = "unloading"


class GenerationStatus(Enum):
    """生成状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class CacheStrategy(Enum):
    """缓存策略枚举"""
    NONE = "none"
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


class LoadBalanceStrategy(Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    RANDOM = "random"
    LATENCY_BASED = "latency_based"


class RetryStrategy(Enum):
    """重试策略枚举"""
    NONE = "none"
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


# ==================== 异常定义 ====================

class ModelServiceError(Exception):
    """模型服务基础异常"""


class ModelNotFoundError(ModelServiceError):
    """模型不存在异常"""


class ModelLoadError(ModelServiceError):
    """模型加载异常"""


class GenerationError(ModelServiceError):
    """生成异常"""


class GenerationTimeoutError(GenerationError):
    """生成超时异常"""


class CircuitBreakerOpenError(ModelServiceError):
    """熔断器开启异常"""


class RateLimitExceededError(ModelServiceError):
    """速率限制异常"""


class ValidationError(ModelServiceError):
    """验证异常"""


# ==================== 数据类定义 ====================

@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    model_type: ModelType
    model_path: str
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    parameters: Optional[Dict[str, Any]] = None
    supports_tools: bool = True
    supports_streaming: bool = True
    supports_vision: bool = False
    batch_size: int = 1
    max_batch_size: int = 8
    cache_enabled: bool = True
    cache_ttl: int = 3600
    gpu_memory_utilization: float = 0.9
    max_concurrent_requests: int = 10
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    fallback_model: Optional[str] = None
    enable_fallback: bool = True
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "model_path": self.model_path,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "supports_tools": self.supports_tools,
            "supports_streaming": self.supports_streaming,
            "cache_enabled": self.cache_enabled,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "fallback_model": self.fallback_model,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        return cls(
            name=data["name"],
            model_type=ModelType(data["model_type"]),
            model_path=data["model_path"],
            max_tokens=data.get("max_tokens", 2048),
            temperature=data.get("temperature", 0.7),
            top_p=data.get("top_p", 0.9),
            supports_tools=data.get("supports_tools", True),
            supports_streaming=data.get("supports_streaming", True),
            cache_enabled=data.get("cache_enabled", True),
            timeout=data.get("timeout", 60.0),
            max_retries=data.get("max_retries", 3),
            fallback_model=data.get("fallback_model"),
        )


@dataclass
class GenerationMetrics:
    """生成指标"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    time_to_first_token_ms: float = 0.0
    tokens_per_second: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    status: GenerationStatus = GenerationStatus.COMPLETED
    success: bool = True
    error: Optional[str] = None
    retry_count: int = 0
    used_fallback: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "tokens_per_second": self.tokens_per_second,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class GenerationRequest:
    """生成请求"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    model_name: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: List[str] = field(default_factory=list)
    tools: Optional[List[str]] = None
    system_prompt: Optional[str] = None
    stream: bool = False
    priority: int = 0
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class GenerationResponse:
    """生成响应"""
    request_id: str
    content: str
    model_name: str
    finish_reason: str = "stop"
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Optional[GenerationMetrics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "content": self.content,
            "model_name": self.model_name,
            "finish_reason": self.finish_reason,
            "tool_calls": self.tool_calls,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


@dataclass
class ModelInstance:
    """模型实例"""
    config: ModelConfig
    status: ModelStatus = ModelStatus.UNLOADED
    instance: Any = None
    loaded_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    request_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    is_healthy: bool = True
    current_requests: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "status": self.status.value,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "request_count": self.request_count,
            "is_healthy": self.is_healthy,
            "current_requests": self.current_requests,
        }


# ==================== 缓存系统 ====================

class CacheEntry:
    """缓存条目"""
    def __init__(self, value: str, ttl: int = 3600):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = time.time()
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache:
    """LRU 缓存"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            entry = self._cache[key]
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            entry.access()
            self._hits += 1
            return entry.value
    
    def set(self, key: str, value: str, ttl: Optional[int] = None):
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            while len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            self._cache[key] = CacheEntry(value, ttl or self.default_ttl)
    
    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0,
            }


# ==================== 熔断器 ====================

class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"
        self._lock = threading.Lock()
    
    @property
    def state(self) -> str:
        with self._lock:
            if self._state == "open":
                if self._last_failure_time and \
                   time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = "half_open"
            return self._state
    
    def record_success(self):
        with self._lock:
            if self._state == "half_open":
                self._state = "closed"
            self._failure_count = 0
    
    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.failure_threshold:
                self._state = "open"
    
    def can_execute(self) -> bool:
        return self.state != "open"
    
    def reset(self):
        with self._lock:
            self._failure_count = 0
            self._state = "closed"
            self._last_failure_time = None


# ==================== 速率限制器 ====================

class RateLimiter:
    """速率限制器"""
    
    def __init__(self, rate: float = 10.0, burst: int = 20):
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_update = time.time()
        self._lock = threading.Lock()
    
    def _refill(self):
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_update = now
    
    def acquire(self, tokens: int = 1) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False
    
    async def acquire_async(self, tokens: int = 1, timeout: float = 10.0) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if self.acquire(tokens):
                return True
            await asyncio.sleep(0.1)
        return False


# ==================== 提示词模板管理 ====================

@dataclass
class PromptTemplate:
    """提示词模板"""
    name: str
    template: str
    description: str = ""
    variables: List[str] = field(default_factory=list)
    
    def render(self, **kwargs) -> str:
        result = self.template
        for var in self.variables:
            placeholder = f"{{{var}}}"
            value = kwargs.get(var, "")
            result = result.replace(placeholder, str(value))
        return result


class PromptTemplateManager:
    """提示词模板管理器"""
    
    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        self._templates["default"] = PromptTemplate(
            name="default",
            template="{prompt}",
            variables=["prompt"]
        )
        self._templates["chat"] = PromptTemplate(
            name="chat",
            template="System: {system_prompt}\n\nUser: {user_message}",
            variables=["system_prompt", "user_message"]
        )
        self._templates["instruct"] = PromptTemplate(
            name="instruct",
            template="### Instruction:\n{instruction}\n\n### Response:",
            variables=["instruction"]
        )
        self._templates["qa"] = PromptTemplate(
            name="qa",
            template="Question: {question}\n\nContext: {context}\n\nAnswer:",
            variables=["question", "context"]
        )
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        return self._templates.get(name)
    
    def add_template(self, template: PromptTemplate):
        self._templates[template.name] = template
    
    def render(self, template_name: str, **kwargs) -> str:
        template = self.get_template(template_name)
        if template:
            return template.render(**kwargs)
        return kwargs.get("prompt", "")
    
    def list_templates(self) -> List[str]:
        return list(self._templates.keys())


# ==================== 模型配置构建器 ====================

class ModelConfigBuilder:
    """模型配置构建器"""
    
    def __init__(self, name: str, model_type: ModelType, model_path: str):
        self._name = name
        self._model_type = model_type
        self._model_path = model_path
        self._max_tokens = 2048
        self._temperature = 0.7
        self._top_p = 0.9
        self._supports_tools = True
        self._supports_streaming = True
        self._cache_enabled = True
        self._timeout = 60.0
        self._max_retries = 3
        self._fallback_model = None
        self._parameters: Dict[str, Any] = {}
    
    def with_max_tokens(self, max_tokens: int) -> 'ModelConfigBuilder':
        self._max_tokens = max_tokens
        return self
    
    def with_temperature(self, temperature: float) -> 'ModelConfigBuilder':
        self._temperature = temperature
        return self
    
    def with_top_p(self, top_p: float) -> 'ModelConfigBuilder':
        self._top_p = top_p
        return self
    
    def with_tools_support(self, enabled: bool = True) -> 'ModelConfigBuilder':
        self._supports_tools = enabled
        return self
    
    def with_streaming(self, enabled: bool = True) -> 'ModelConfigBuilder':
        self._supports_streaming = enabled
        return self
    
    def with_cache(self, enabled: bool = True) -> 'ModelConfigBuilder':
        self._cache_enabled = enabled
        return self
    
    def with_timeout(self, timeout: float) -> 'ModelConfigBuilder':
        self._timeout = timeout
        return self
    
    def with_retries(self, max_retries: int) -> 'ModelConfigBuilder':
        self._max_retries = max_retries
        return self
    
    def with_fallback(self, fallback_model: str) -> 'ModelConfigBuilder':
        self._fallback_model = fallback_model
        return self
    
    def with_parameters(self, **kwargs) -> 'ModelConfigBuilder':
        self._parameters.update(kwargs)
        return self
    
    def build(self) -> ModelConfig:
        return ModelConfig(
            name=self._name,
            model_type=self._model_type,
            model_path=self._model_path,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            supports_tools=self._supports_tools,
            supports_streaming=self._supports_streaming,
            cache_enabled=self._cache_enabled,
            timeout=self._timeout,
            max_retries=self._max_retries,
            fallback_model=self._fallback_model,
            parameters=self._parameters,
        )


# ==================== 模型池 ====================

class ModelPool:
    """模型池"""
    
    def __init__(self, 
                 load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN):
        self._instances: Dict[str, List[ModelInstance]] = {}
        self._strategy = load_balance_strategy
        self._round_robin_index: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def add_instance(self, model_name: str, instance: ModelInstance):
        with self._lock:
            if model_name not in self._instances:
                self._instances[model_name] = []
                self._round_robin_index[model_name] = 0
            self._instances[model_name].append(instance)
    
    def remove_instance(self, model_name: str, instance: ModelInstance):
        with self._lock:
            if model_name in self._instances:
                self._instances[model_name] = [
                    i for i in self._instances[model_name] if i != instance
                ]
    
    def get_instance(self, model_name: str) -> Optional[ModelInstance]:
        with self._lock:
            instances = self._instances.get(model_name, [])
            healthy_instances = [i for i in instances if i.is_healthy and i.status == ModelStatus.READY]
            
            if not healthy_instances:
                return None
            
            if self._strategy == LoadBalanceStrategy.ROUND_ROBIN:
                idx = self._round_robin_index.get(model_name, 0)
                instance = healthy_instances[idx % len(healthy_instances)]
                self._round_robin_index[model_name] = idx + 1
                return instance
            
            elif self._strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
                return min(healthy_instances, key=lambda i: i.current_requests)
            
            elif self._strategy == LoadBalanceStrategy.RANDOM:
                import random
                return random.choice(healthy_instances)
            
            return healthy_instances[0] if healthy_instances else None
    
    def get_all_instances(self, model_name: str) -> List[ModelInstance]:
        with self._lock:
            return list(self._instances.get(model_name, []))
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = {}
            for model_name, instances in self._instances.items():
                stats[model_name] = {
                    "total_instances": len(instances),
                    "healthy_instances": sum(1 for i in instances if i.is_healthy),
                    "total_requests": sum(i.request_count for i in instances),
                }
            return stats


# ==================== 主服务类 ====================

class LocalModelService:
    """本地模型服务
    
    集成 LangGraph 工具系统的生产级本地模型服务。
    """
    
    def __init__(self, 
                 enable_tools: bool = True,
                 cache_strategy: CacheStrategy = CacheStrategy.LRU,
                 max_cache_size: int = 1000,
                 enable_circuit_breaker: bool = True,
                 enable_rate_limiting: bool = True):
        """初始化本地模型服务"""
        self.models: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, Any] = {}
        self._model_instances: Dict[str, ModelInstance] = {}
        self.default_model = "default"
        
        # 工具系统
        self.enable_tools = enable_tools
        self._tool_registry: Optional[ToolRegistry] = None
        self._tool_executor: Optional[ToolExecutor] = None
        self._tool_manager: Optional[ToolManager] = None
        
        # 缓存系统
        self._cache_strategy = cache_strategy
        self._cache = LRUCache(max_size=max_cache_size) if cache_strategy != CacheStrategy.NONE else None
        
        # 熔断器
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._enable_circuit_breaker = enable_circuit_breaker
        
        # 速率限制
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._enable_rate_limiting = enable_rate_limiting
        
        # 模型池
        self._model_pool = ModelPool()
        
        # 提示词模板
        self._prompt_manager = PromptTemplateManager()
        
        # 指标收集
        self._metrics: List[GenerationMetrics] = []
        self._metrics_lock = threading.Lock()
        
        # 健康检查
        self._health_check_task: Optional[asyncio.Task] = None
        self._health_check_interval = 30
        
        # 初始化
        self._initialize_default_models()
        if self.enable_tools:
            self._initialize_tool_system()
        
        logger.info("LocalModelService initialized")
    
    def _initialize_default_models(self):
        """初始化默认模型配置"""
        self.models["default"] = ModelConfig(
            name="default",
            model_type=ModelType.MOCK,
            model_path="mock://default",
            max_tokens=1024,
            temperature=0.7
        )
        self._model_instances["default"] = ModelInstance(config=self.models["default"])

        try:
            available_models = ollama.list()
            for model in available_models.get('models', []):
                model_name = model['name']
                self.models[model_name] = ModelConfig(
                    name=model_name,
                    model_type=ModelType.OLLAMA,
                    model_path=model_name,
                    max_tokens=2048,
                    temperature=0.7
                )
                self._model_instances[model_name] = ModelInstance(config=self.models[model_name])
                logger.info(f"Found Ollama model: {model_name}")
        except Exception as e:
            logger.warning(f"Unable to get Ollama model list: {str(e)}")
        
        if os.getenv("OPENAI_API_KEY"):
            for model_name in ["gpt-3.5-turbo", "gpt-4"]:
                self.models[model_name] = ModelConfig(
                    name=model_name,
                    model_type=ModelType.OPENAI,
                    model_path=model_name,
                    max_tokens=4096 if "3.5" in model_name else 8192,
                    temperature=0.7,
                    supports_tools=True,
                    supports_streaming=True
                )
                self._model_instances[model_name] = ModelInstance(config=self.models[model_name])
    
    def _initialize_tool_system(self):
        """初始化工具系统"""
        try:
            self._tool_registry = get_global_registry()
            self._tool_manager = get_tool_manager()
            self._tool_executor = ToolExecutor(self._tool_registry)
            
            builtin_tools = get_builtin_tools()
            for tool_instance in builtin_tools:
                if not self._tool_registry.get(tool_instance.name):
                    self._tool_registry.register(tool_instance)
            
            logger.info(f"Tool system initialized with {len(builtin_tools)} builtin tools")
        except Exception as e:
            logger.error(f"Failed to initialize tool system: {str(e)}")
            self.enable_tools = False
    
    def _get_circuit_breaker(self, model_name: str) -> CircuitBreaker:
        """获取模型的熔断器"""
        if model_name not in self._circuit_breakers:
            self._circuit_breakers[model_name] = CircuitBreaker()
        return self._circuit_breakers[model_name]
    
    def _get_rate_limiter(self, model_name: str) -> RateLimiter:
        """获取模型的速率限制器"""
        if model_name not in self._rate_limiters:
            config = self.models.get(model_name)
            rate = config.max_concurrent_requests if config else 10
            self._rate_limiters[model_name] = RateLimiter(rate=rate, burst=rate * 2)
        return self._rate_limiters[model_name]
    
    # ==================== 模型管理 ====================
    
    def add_model(self, config: ModelConfig):
        """添加模型配置"""
        self.models[config.name] = config
        self._model_instances[config.name] = ModelInstance(config=config)
        logger.info(f"Added model config: {config.name}")
    
    def remove_model(self, model_name: str) -> bool:
        """移除模型配置"""
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
            if model_name in self._model_instances:
                del self._model_instances[model_name]
            logger.info(f"Removed model config: {model_name}")
            return True
        return False
    
    def list_models(self) -> List[str]:
        """列出所有可用模型"""
        return list(self.models.keys())
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        return self.models.get(model_name)
    
    def set_default_model(self, model_name: str) -> bool:
        """设置默认模型"""
        if model_name in self.models:
            self.default_model = model_name
            logger.info(f"Set default model: {model_name}")
            return True
        return False
    
    async def load_model(self, model_name: str) -> bool:
        """加载模型"""
        if model_name not in self.models:
            logger.error(f"Model config not found: {model_name}")
            return False
        
        instance = self._model_instances.get(model_name)
        if instance and instance.status == ModelStatus.READY:
            logger.info(f"Model already loaded: {model_name}")
            return True
        
        if instance:
            instance.status = ModelStatus.LOADING
        
        config = self.models[model_name]
        
        try:
            if config.model_type == ModelType.OLLAMA:
                self.loaded_models[model_name] = "ollama_client"
            elif config.model_type == ModelType.TRANSFORMERS:
                tokenizer = AutoTokenizer.from_pretrained(config.model_path)
                model = AutoModelForCausalLM.from_pretrained(config.model_path)
                self.loaded_models[model_name] = {"tokenizer": tokenizer, "model": model}
            elif config.model_type == ModelType.OPENAI:
                self.loaded_models[model_name] = "openai_client"
            elif config.model_type == ModelType.MOCK:
                self.loaded_models[model_name] = "mock_model"
            
            if instance:
                instance.status = ModelStatus.READY
                instance.loaded_at = datetime.now()
                instance.is_healthy = True
            
            logger.info(f"Model loaded successfully: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Model load failed {model_name}: {str(e)}")
            if instance:
                instance.status = ModelStatus.ERROR
            return False
    
    async def unload_model(self, model_name: str) -> bool:
        """卸载模型"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            instance = self._model_instances.get(model_name)
            if instance:
                instance.status = ModelStatus.UNLOADED
                instance.instance = None
            logger.info(f"Model unloaded: {model_name}")
            return True
        return False
    
    async def warmup_model(self, model_name: str, warmup_prompts: List[str] = None) -> bool:
        """预热模型"""
        if not await self.load_model(model_name):
            return False
        
        warmup_prompts = warmup_prompts or ["Hello", "How are you?"]
        
        instance = self._model_instances.get(model_name)
        if instance:
            instance.status = ModelStatus.WARMING_UP
        
        try:
            for prompt in warmup_prompts:
                await self.generate_response(prompt, model_name)
            
            if instance:
                instance.status = ModelStatus.READY
            
            logger.info(f"Model warmed up: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Model warmup failed: {str(e)}")
            return False
    
    # ==================== 响应生成 ====================
    
    async def generate_response(self, 
                              prompt: str, 
                              model_name: Optional[str] = None,
                              tools: Optional[List[str]] = None,
                              **kwargs) -> str:
        """生成响应"""
        model_name = model_name or self.default_model
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        if model_name not in self.models:
            raise ModelNotFoundError(f"Model not found: {model_name}")
        
        config = self.models[model_name]
        
        # 检查熔断器
        if self._enable_circuit_breaker:
            breaker = self._get_circuit_breaker(model_name)
            if not breaker.can_execute():
                if config.enable_fallback and config.fallback_model:
                    logger.warning(f"Circuit breaker open for {model_name}, using fallback")
                    return await self.generate_response(prompt, config.fallback_model, tools, **kwargs)
                raise CircuitBreakerOpenError(f"Circuit breaker open for {model_name}")
        
        # 检查速率限制
        if self._enable_rate_limiting:
            limiter = self._get_rate_limiter(model_name)
            if not await limiter.acquire_async():
                raise RateLimitExceededError(f"Rate limit exceeded for {model_name}")
        
        # 确保模型已加载
        if not await self.load_model(model_name):
            raise ModelLoadError(f"Failed to load model: {model_name}")
        
        # 检查缓存
        cache_key = None
        if config.cache_enabled and self._cache:
            cache_key = self._get_cache_key(prompt, model_name, **kwargs)
            cached = self._cache.get(cache_key)
            if cached:
                return cached
        
        # 更新实例状态
        instance = self._model_instances.get(model_name)
        if instance:
            instance.current_requests += 1
        
        try:
            response = await self._generate_with_retry(prompt, model_name, config, tools, **kwargs)
            
            # 记录成功
            if self._enable_circuit_breaker:
                self._get_circuit_breaker(model_name).record_success()
            
            # 缓存响应
            if cache_key and self._cache:
                self._cache.set(cache_key, response, config.cache_ttl)
            
            # 记录指标
            latency_ms = (time.time() - start_time) * 1000
            self._record_metrics(GenerationMetrics(
                request_id=request_id,
                model_name=model_name,
                latency_ms=latency_ms,
                success=True
            ))
            
            if instance:
                instance.request_count += 1
                instance.last_used = datetime.now()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed for {model_name}: {str(e)}")
            
            if self._enable_circuit_breaker:
                self._get_circuit_breaker(model_name).record_failure()
            
            if instance:
                instance.error_count += 1
            
            self._record_metrics(GenerationMetrics(
                request_id=request_id,
                model_name=model_name,
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e),
                status=GenerationStatus.FAILED
            ))
            
            # 尝试降级
            if config.enable_fallback and config.fallback_model:
                logger.warning(f"Attempting fallback to {config.fallback_model}")
                return await self.generate_response(prompt, config.fallback_model, tools, **kwargs)
            
            raise GenerationError(str(e))
        
        finally:
            if instance:
                instance.current_requests -= 1
    
    async def _generate_with_retry(self, 
                                  prompt: str, 
                                  model_name: str,
                                  config: ModelConfig,
                                  tools: Optional[List[str]] = None,
                                  **kwargs) -> str:
        """带重试的生成"""
        last_error = None
        
        for attempt in range(config.max_retries + 1):
            try:
                return await self._do_generate(prompt, model_name, config, tools, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < config.max_retries:
                    delay = self._calculate_retry_delay(config, attempt)
                    logger.warning(f"Generation attempt {attempt + 1} failed, retrying in {delay}s")
                    await asyncio.sleep(delay)
        
        raise last_error
    
    def _calculate_retry_delay(self, config: ModelConfig, attempt: int) -> float:
        """计算重试延迟"""
        if config.retry_strategy == RetryStrategy.FIXED:
            return config.retry_delay
        elif config.retry_strategy == RetryStrategy.EXPONENTIAL:
            return config.retry_delay * (2 ** attempt)
        elif config.retry_strategy == RetryStrategy.LINEAR:
            return config.retry_delay * (attempt + 1)
        return config.retry_delay
    
    async def _do_generate(self, 
                          prompt: str, 
                          model_name: str,
                          config: ModelConfig,
                          tools: Optional[List[str]] = None,
                          **kwargs) -> str:
        """执行生成"""
        if config.model_type == ModelType.OLLAMA:
            return await self._generate_ollama_response(prompt, config, tools, **kwargs)
        elif config.model_type == ModelType.TRANSFORMERS:
            return await self._generate_transformers_response(prompt, config, **kwargs)
        elif config.model_type == ModelType.OPENAI:
            return await self._generate_openai_response(prompt, config, tools, **kwargs)
        elif config.model_type == ModelType.MOCK:
            return await self._generate_mock_response(prompt, config, tools, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
    
    async def generate_with_tools(self,
                                 prompt: str,
                                 model_name: Optional[str] = None,
                                 tool_names: Optional[List[str]] = None,
                                 auto_execute: bool = True,
                                 **kwargs) -> Dict[str, Any]:
        """使用工具生成响应"""
        if not self.enable_tools:
            response = await self.generate_response(prompt, model_name, **kwargs)
            return {"response": response, "tool_calls": [], "tool_results": []}
        
        model_name = model_name or self.default_model
        config = self.models.get(model_name)
        
        if not config or not config.supports_tools:
            response = await self.generate_response(prompt, model_name, **kwargs)
            return {"response": response, "tool_calls": [], "tool_results": []}
        
        tools = self._get_tools(tool_names)
        response, tool_calls = await self._generate_with_tool_calls(prompt, model_name, tools, **kwargs)
        
        tool_results = []
        if auto_execute and tool_calls and self._tool_executor:
            for tool_call in tool_calls:
                try:
                    result = await self._tool_executor.execute(tool_call.name, tool_call.arguments)
                    tool_results.append(ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        result=result,
                        success=True
                    ))
                except Exception as e:
                    tool_results.append(ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        result=str(e),
                        success=False,
                        error=str(e)
                    ))
        
        return {
            "response": response,
            "tool_calls": [tc.to_dict() if hasattr(tc, 'to_dict') else vars(tc) for tc in tool_calls],
            "tool_results": [tr.to_dict() if hasattr(tr, 'to_dict') else vars(tr) for tr in tool_results]
        }
    
    async def stream_response(self,
                             prompt: str,
                             model_name: Optional[str] = None,
                             **kwargs) -> AsyncIterator[str]:
        """流式生成响应"""
        model_name = model_name or self.default_model
        config = self.models.get(model_name)
        
        if not config:
            raise ModelNotFoundError(f"Model not found: {model_name}")
        
        if not config.supports_streaming:
            response = await self.generate_response(prompt, model_name, **kwargs)
            yield response
            return
        
        if not await self.load_model(model_name):
            raise ModelLoadError(f"Failed to load model: {model_name}")
        
        try:
            if config.model_type == ModelType.OPENAI:
                async for chunk in self._stream_openai_response(prompt, config, **kwargs):
                    yield chunk
            elif config.model_type == ModelType.OLLAMA:
                async for chunk in self._stream_ollama_response(prompt, config, **kwargs):
                    yield chunk
            else:
                response = await self.generate_response(prompt, model_name, **kwargs)
                yield response
        except Exception as e:
            logger.error(f"Stream generation failed: {str(e)}")
            raise
    
    async def batch_generate(self,
                            prompts: List[str],
                            model_name: Optional[str] = None,
                            **kwargs) -> List[str]:
        """批量生成响应"""
        tasks = [self.generate_response(prompt, model_name, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    # ==================== 私有方法 ====================
    
    async def _generate_ollama_response(self, prompt: str, config: ModelConfig,
                                       tools: Optional[List[str]] = None, **kwargs) -> str:
        try:
            response = ollama.generate(
                model=config.model_path,
                prompt=prompt,
                options={
                    "temperature": kwargs.get("temperature", config.temperature),
                    "top_p": kwargs.get("top_p", config.top_p),
                    "num_predict": kwargs.get("max_tokens", config.max_tokens)
                }
            )
            return response.get("response", "")
        except Exception as e:
            logger.error(f"Ollama generation failed: {str(e)}")
            raise
    
    async def _generate_transformers_response(self, prompt: str, config: ModelConfig, **kwargs) -> str:
        try:
            model_data = self.loaded_models[config.name]
            tokenizer = model_data["tokenizer"]
            model = model_data["model"]
            
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=kwargs.get("max_tokens", config.max_tokens),
                    temperature=kwargs.get("temperature", config.temperature),
                    top_p=kwargs.get("top_p", config.top_p),
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
        except Exception as e:
            logger.error(f"Transformers generation failed: {str(e)}")
            raise
    
    async def _generate_openai_response(self, prompt: str, config: ModelConfig,
                                       tools: Optional[List[str]] = None, **kwargs) -> str:
        try:
            client = openai.OpenAI()
            messages = [{"role": "user", "content": prompt}]
            
            system_prompt = kwargs.get("system_prompt")
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            openai_tools = None
            if tools and self.enable_tools and config.supports_tools:
                openai_tools = self._get_openai_tool_schemas(tools)
            
            create_params = {
                "model": config.model_path,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", config.max_tokens),
                "temperature": kwargs.get("temperature", config.temperature),
                "top_p": kwargs.get("top_p", config.top_p)
            }
            
            if openai_tools:
                create_params["tools"] = openai_tools
            
            response = client.chat.completions.create(**create_params)
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise
    
    async def _generate_mock_response(self, prompt: str, config: ModelConfig,
                                     tools: Optional[List[str]] = None, **kwargs) -> str:
        if "你好" in prompt or "hello" in prompt.lower():
            return "您好！我是训练助手，很高兴为您服务！"
        elif "训练" in prompt:
            return "关于训练功能，我可以帮助您创建训练任务、监控训练进度、下载训练好的模型等。"
        elif "帮助" in prompt or "help" in prompt.lower():
            return "我可以帮助您使用训练平台的各种功能。请告诉我您需要什么帮助。"
        elif tools:
            return f"我可以使用以下工具来帮助您：{', '.join(tools)}。"
        else:
            return f"我理解您的问题是关于：{prompt[:100]}{'...' if len(prompt) > 100 else ''}。"
    
    async def _generate_with_tool_calls(self, prompt: str, model_name: str,
                                       tools: List[Tool], **kwargs) -> tuple:
        config = self.models[model_name]
        
        if config.model_type == ModelType.OPENAI:
            return await self._generate_openai_with_tools(prompt, config, tools, **kwargs)
        else:
            response = await self.generate_response(prompt, model_name, **kwargs)
            return response, []
    
    async def _generate_openai_with_tools(self, prompt: str, config: ModelConfig,
                                         tools: List[Tool], **kwargs) -> tuple:
        client = openai.OpenAI()
        messages = [{"role": "user", "content": prompt}]
        
        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        openai_tools = [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.get_parameters_schema() if hasattr(t, 'get_parameters_schema') else {}
            }
        } for t in tools]
        
        response = client.chat.completions.create(
            model=config.model_path,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            max_tokens=kwargs.get("max_tokens", config.max_tokens),
            temperature=kwargs.get("temperature", config.temperature),
        )
        
        message = response.choices[0].message
        content = message.content or ""
        
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments) if tc.function.arguments else {}
                ))
        
        return content, tool_calls
    
    async def _stream_openai_response(self, prompt: str, config: ModelConfig, **kwargs) -> AsyncIterator[str]:
        client = openai.OpenAI()
        messages = [{"role": "user", "content": prompt}]
        
        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        stream = client.chat.completions.create(
            model=config.model_path,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", config.max_tokens),
            temperature=kwargs.get("temperature", config.temperature),
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def _stream_ollama_response(self, prompt: str, config: ModelConfig, **kwargs) -> AsyncIterator[str]:
        stream = ollama.generate(
            model=config.model_path,
            prompt=prompt,
            stream=True,
            options={
                "temperature": kwargs.get("temperature", config.temperature),
                "top_p": kwargs.get("top_p", config.top_p),
                "num_predict": kwargs.get("max_tokens", config.max_tokens)
            }
        )
        
        for chunk in stream:
            if "response" in chunk:
                yield chunk["response"]
    
    # ==================== 工具方法 ====================
    
    def _get_tools(self, tool_names: Optional[List[str]] = None) -> List[Tool]:
        if not self.enable_tools or not self._tool_registry:
            return []
        
        if tool_names:
            tools = []
            for name in tool_names:
                tool_instance = self._tool_registry.get(name) or get_tool_by_name(name)
                if tool_instance:
                    tools.append(tool_instance)
            return tools
        
        return self._tool_registry.list_tools(include_disabled=True)
    
    def _get_openai_tool_schemas(self, tool_names: List[str]) -> List[Dict[str, Any]]:
        tools = self._get_tools(tool_names)
        return [{
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.get_parameters_schema() if hasattr(t, 'get_parameters_schema') else {}
            }
        } for t in tools]
    
    def register_tool(self, tool_instance: Tool) -> bool:
        if not self.enable_tools or not self._tool_registry:
            return False
        try:
            self._tool_registry.register(tool_instance)
            logger.info(f"Registered tool: {tool_instance.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register tool: {str(e)}")
            return False
    
    def get_available_tools(self) -> List[str]:
        if not self.enable_tools or not self._tool_registry:
            return []
        return [t.name for t in self._tool_registry.list_tools(include_disabled=True)]
    
    def get_tool_registry(self) -> Optional[ToolRegistry]:
        return self._tool_registry
    
    def get_tool_executor(self) -> Optional[ToolExecutor]:
        return self._tool_executor
    
    # ==================== 缓存方法 ====================
    
    def _get_cache_key(self, prompt: str, model_name: str, **kwargs) -> str:
        key_data = f"{prompt}:{model_name}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def clear_cache(self):
        if self._cache:
            self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        if self._cache:
            return self._cache.get_stats()
        return {}
    
    # ==================== 指标方法 ====================
    
    def _record_metrics(self, metrics: GenerationMetrics):
        with self._metrics_lock:
            self._metrics.append(metrics)
            if len(self._metrics) > 1000:
                self._metrics = self._metrics[-1000:]
    
    def get_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._metrics_lock:
            return [m.to_dict() for m in self._metrics[-limit:]]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        with self._metrics_lock:
            if not self._metrics:
                return {}
            
            total = len(self._metrics)
            success_count = sum(1 for m in self._metrics if m.success)
            total_latency = sum(m.latency_ms for m in self._metrics)
            
            return {
                "total_requests": total,
                "success_count": success_count,
                "error_count": total - success_count,
                "success_rate": success_count / total if total > 0 else 0,
                "average_latency_ms": total_latency / total if total > 0 else 0,
            }
    
    # ==================== 健康检查 ====================
    
    async def health_check(self, model_name: str = None) -> Dict[str, Any]:
        """健康检查"""
        if model_name:
            return await self._check_model_health(model_name)
        
        results = {}
        for name in self.models.keys():
            results[name] = await self._check_model_health(name)
        return results
    
    async def _check_model_health(self, model_name: str) -> Dict[str, Any]:
        instance = self._model_instances.get(model_name)
        if not instance:
            return {"healthy": False, "error": "Model not found"}
        
        try:
            await self.generate_response("test", model_name, max_tokens=10)
            instance.is_healthy = True
            return {"healthy": True, "status": instance.status.value}
        except Exception as e:
            instance.is_healthy = False
            return {"healthy": False, "error": str(e)}
    
    # ==================== 提示词模板 ====================
    
    def get_prompt_template(self, name: str) -> Optional[PromptTemplate]:
        return self._prompt_manager.get_template(name)
    
    def add_prompt_template(self, template: PromptTemplate):
        self._prompt_manager.add_template(template)
    
    def render_prompt(self, template_name: str, **kwargs) -> str:
        return self._prompt_manager.render(template_name, **kwargs)
    
    def list_prompt_templates(self) -> List[str]:
        return self._prompt_manager.list_templates()
    
    # ==================== 状态和信息 ====================
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        if model_name not in self.models:
            return {}
        
        config = self.models[model_name]
        instance = self._model_instances.get(model_name)
        
        return {
            "name": config.name,
            "type": config.model_type.value,
            "path": config.model_path,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "is_loaded": model_name in self.loaded_models,
            "status": instance.status.value if instance else "unknown",
            "is_healthy": instance.is_healthy if instance else False,
            "request_count": instance.request_count if instance else 0,
            "supports_tools": config.supports_tools,
            "supports_streaming": config.supports_streaming,
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        return {
            "total_models": len(self.models),
            "loaded_models": len(self.loaded_models),
            "default_model": self.default_model,
            "tools_enabled": self.enable_tools,
            "available_tools": len(self.get_available_tools()) if self.enable_tools else 0,
            "cache_enabled": self._cache is not None,
            "cache_stats": self.get_cache_stats(),
            "circuit_breaker_enabled": self._enable_circuit_breaker,
            "rate_limiting_enabled": self._enable_rate_limiting,
            "available_types": {
            },
            "models": {name: self.get_model_info(name) for name in self.models.keys()},
            "metrics_summary": self.get_metrics_summary()
        }


# ==================== 工厂函数 ====================

def create_model_config(name: str, model_type: ModelType, model_path: str) -> ModelConfigBuilder:
    """创建模型配置构建器"""
    return ModelConfigBuilder(name, model_type, model_path)


# ==================== 全局实例 ====================

_local_model_service: Optional[LocalModelService] = None


def get_local_model_service() -> LocalModelService:
    """获取本地模型服务实例"""
    global _local_model_service
    if _local_model_service is None:
        _local_model_service = LocalModelService()
    return _local_model_service


def set_local_model_service(service: LocalModelService):
    """设置本地模型服务实例"""
    global _local_model_service
    _local_model_service = service

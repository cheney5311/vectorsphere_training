"""
LangChain推理服务

该模块提供基于LangChain的智能推理和会话管理功能，
集成本地模型服务，支持上下文记忆和多轮对话。

扩展功能：
- 与 LangGraph Agent 系统集成
- 支持多种 Agent 类型
- 支持策略模式
- 支持回调和指标收集
- 批量推理和并发控制
- 请求队列和负载均衡
- 提示模板管理
- 多级缓存系统
- RAG 检索增强
- 熔断器和降级策略
"""

import asyncio
import json
import uuid
import time
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, AsyncIterator, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import copy
import warnings

from loguru import logger

# LangChain相关导入
from langchain_core.language_models.llm import LLM
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory

# LangGraph 集成
from backend.algo.langgraph.agents import (
    BaseAgent,
    LoggingCallback, MetricsCallback,
    CallbackManager, AgentPool, AgentOrchestrator
)
from backend.algo.langgraph.factory import (
    AgentFactory, MasterFactory, get_master_factory,
    build_enhanced_agent, factory_health_check
)
from backend.algo.langgraph.state import (
    AgentState, AgentMessage, MessageType as AgentMessageType,
    ToolCall, ToolResult, MemoryEntry
)

from .local_model_service import LocalModelService
from .session_history_manager import SessionHistoryManager, MessageType
from .agent_type import (
    LangGraphAgentType, ExecutionMode,
    get_default_config
)
from .api_config_manager import AgentInstanceConfig


# ==================== 枚举定义 ====================

class SessionStatus(Enum):
    """会话状态枚举"""
    ACTIVE = "active"
    IDLE = "idle"
    PROCESSING = "processing"
    PAUSED = "paused"
    EXPIRED = "expired"
    TERMINATED = "terminated"


class CacheStrategy(Enum):
    """缓存策略枚举"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    NONE = "none"


class RetryStrategy(Enum):
    """重试策略枚举"""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


class CircuitState(Enum):
    """熔断器状态枚举"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RequestPriority(Enum):
    """请求优先级枚举"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


# ==================== 异常定义 ====================

class InferenceError(Exception):
    """推理异常基类"""


class SessionError(InferenceError):
    """会话异常"""


class ModelError(InferenceError):
    """模型异常"""


class RateLimitError(InferenceError):
    """限流异常"""


class CircuitBreakerOpenError(InferenceError):
    """熔断器打开异常"""


class InferenceTimeoutError(InferenceError):
    """超时异常"""


# ==================== 数据类定义 ====================

@dataclass
class InferenceConfig:
    """推理配置"""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_cache: bool = True
    cache_ttl: int = 3600
    enable_streaming: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: str
    session_id: str
    message: str
    context: Optional[Dict[str, Any]] = None
    priority: RequestPriority = RequestPriority.NORMAL
    config: Optional[InferenceConfig] = None
    created_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "message": self.message,
            "context": self.context,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "timeout": self.timeout,
        }


@dataclass
class InferenceResponse:
    """推理响应"""
    request_id: str
    session_id: str
    content: str
    tokens_used: int = 0
    latency_ms: float = 0.0
    from_cache: bool = False
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceMetrics:
    """推理指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cached_requests: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    timeout_count: int = 0
    retry_count: int = 0
    
    def record(self, success: bool, latency_ms: float = 0.0, tokens: int = 0,
               from_cache: bool = False, timeout: bool = False, retry: bool = False):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        if from_cache:
            self.cached_requests += 1
        if timeout:
            self.timeout_count += 1
        if retry:
            self.retry_count += 1
        self.total_latency_ms += latency_ms
        self.total_tokens += tokens
    
    def get_summary(self) -> Dict[str, Any]:
        avg_latency = self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0
        success_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 0
        cache_rate = self.cached_requests / self.total_requests if self.total_requests > 0 else 0
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "average_latency_ms": avg_latency,
            "total_tokens": self.total_tokens,
            "cache_hit_rate": cache_rate,
            "timeout_count": self.timeout_count,
            "retry_count": self.retry_count,
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


class ResponseCache:
    """响应缓存"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600,
                 strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
    
    def _get_cache_key(self, prompt: str, context: Dict[str, Any] = None) -> str:
        key_data = f"{prompt}:{json.dumps(context or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, prompt: str, context: Dict[str, Any] = None) -> Optional[str]:
        key = self._get_cache_key(prompt, context)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():
                    entry.access()
                    if self.strategy == CacheStrategy.LRU:
                        self._cache.move_to_end(key)
                    self._hits += 1
                    return entry.value
                else:
                    del self._cache[key]
            self._misses += 1
            return None
    
    def set(self, prompt: str, response: str, context: Dict[str, Any] = None, ttl: int = None):
        key = self._get_cache_key(prompt, context)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            
            while len(self._cache) >= self.max_size:
                if self.strategy == CacheStrategy.LRU:
                    self._cache.popitem(last=False)
                elif self.strategy == CacheStrategy.LFU:
                    min_key = min(self._cache.keys(), key=lambda k: self._cache[k].access_count)
                    del self._cache[min_key]
                else:
                    self._cache.popitem(last=False)
            
            self._cache[key] = CacheEntry(response, ttl or self.default_ttl)
    
    def clear(self):
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
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
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0,
                 half_open_requests: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_count = 0
        self._lock = threading.Lock()
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._last_failure_time and time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_count = 0
            return self._state
    
    def can_execute(self) -> bool:
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        elif state == CircuitState.HALF_OPEN:
            with self._lock:
                return self._half_open_count < self.half_open_requests
        return False
    
    def record_success(self):
        with self._lock:
            self._success_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_count += 1
                if self._half_open_count >= self.half_open_requests:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit breaker closed")
    
    def record_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker opened (half-open failure)")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker opened (failures: {self._failure_count})")
    
    def reset(self):
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_count = 0
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
        }


# ==================== 速率限制器 ====================

class RateLimiter:
    """速率限制器（令牌桶算法）"""
    
    def __init__(self, rate: float = 10.0, burst: int = 20):
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_update = now
            
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
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "rate": self.rate,
            "burst": self.burst,
            "available_tokens": self._tokens,
        }


# ==================== 提示模板管理 ====================

@dataclass
class PromptTemplate:
    """提示模板"""
    name: str
    template: str
    description: str = ""
    variables: List[str] = field(default_factory=list)
    category: str = "general"
    version: str = "1.0"
    
    def render(self, **kwargs) -> str:
        result = self.template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", str(value))
        return result


class PromptTemplateManager:
    """提示模板管理器"""
    
    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._register_builtin_templates()
    
    def _register_builtin_templates(self):
        """注册内置模板"""
        self.register(PromptTemplate(
            name="default_assistant",
            template="""你是一个专业的AI训练助手，专门帮助用户使用训练平台。

你的主要职责包括：
1. 帮助用户创建和管理训练任务
2. 监控训练进度并提供建议
3. 协助下载和使用训练好的模型
4. 解答训练相关的技术问题
5. 提供最佳实践建议

当前上下文信息：
{context}

对话历史：
{history}

用户消息：{input}

请提供专业、友好的回复：""",
            description="默认助手提示模板",
            variables=["context", "history", "input"],
            category="assistant"
        ))
        
        self.register(PromptTemplate(
            name="task_executor",
            template="""你是一个任务执行助手。

任务描述：{task}
可用工具：{tools}
上下文：{context}

请分析任务并执行，提供结果：""",
            description="任务执行提示模板",
            variables=["task", "tools", "context"],
            category="task"
        ))
        
        self.register(PromptTemplate(
            name="summarizer",
            template="""请对以下对话进行总结：

{conversation}

总结要点：""",
            description="对话总结提示模板",
            variables=["conversation"],
            category="summarization"
        ))
    
    def register(self, template: PromptTemplate):
        self._templates[template.name] = template
    
    def get(self, name: str) -> Optional[PromptTemplate]:
        return self._templates.get(name)
    
    def list_templates(self, category: str = None) -> List[str]:
        if category:
            return [name for name, t in self._templates.items() if t.category == category]
        return list(self._templates.keys())
    
    def render(self, name: str, **kwargs) -> Optional[str]:
        template = self.get(name)
        if template:
            return template.render(**kwargs)
        return None


# ==================== CustomLLM 增强版 ====================

class CustomLLM(LLM):
    """自定义LLM包装器，集成本地模型服务"""
    
    def __init__(self, local_model_service: LocalModelService, model_name: str = "default",
                 config: Optional[InferenceConfig] = None):
        super().__init__()
        self.local_model_service = local_model_service
        self.model_name = model_name
        self.config = config or InferenceConfig()
        
        self._circuit_breaker = CircuitBreaker()
        self._rate_limiter = RateLimiter(rate=10.0, burst=20)
        self._metrics = InferenceMetrics()
        self._lock = threading.Lock()
    
    @property
    def _llm_type(self) -> str:
        return "custom_local_llm"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """同步调用模型"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._async_call_with_retry(prompt, stop, **kwargs))
                    return future.result()
            else:
                return loop.run_until_complete(self._async_call_with_retry(prompt, stop, **kwargs))
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return f"抱歉，我遇到了一些技术问题：{str(e)}"
    
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """异步调用模型"""
        return await self._async_call_with_retry(prompt, stop, **kwargs)
    
    async def _async_call_with_retry(self, prompt: str, stop: Optional[List[str]] = None,
                                     **kwargs) -> str:
        """带重试的异步调用"""
        if not self._circuit_breaker.can_execute():
            raise CircuitBreakerOpenError("Circuit breaker is open")
        
        if not await self._rate_limiter.acquire_async():
            raise RateLimitError("Rate limit exceeded")
        
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                start_time = time.time()
                
                response = await asyncio.wait_for(
                    self.local_model_service.generate_response(
                        prompt, self.model_name,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        **kwargs
                    ),
                    timeout=self.config.timeout
                )
                
                latency_ms = (time.time() - start_time) * 1000
                self._metrics.record(success=True, latency_ms=latency_ms, retry=attempt > 0)
                self._circuit_breaker.record_success()
                
                return response
                
            except asyncio.TimeoutError:
                last_error = InferenceTimeoutError(f"Request timeout after {self.config.timeout}s")
                self._metrics.record(success=False, timeout=True)
                logger.warning(f"LLM call timeout (attempt {attempt + 1})")
                
            except (ConnectionError, RuntimeError, ValueError, TypeError, OSError, ModelError) as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {str(e)}")
            
            if attempt < self.config.max_retries - 1:
                delay = self.config.retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)
        
        self._circuit_breaker.record_failure()
        self._metrics.record(success=False)
        raise last_error or ModelError("All retries failed")
    
    def get_metrics(self) -> Dict[str, Any]:
        return self._metrics.get_summary()
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "circuit_breaker": self._circuit_breaker.get_status(),
            "rate_limiter": self._rate_limiter.get_status(),
            "metrics": self._metrics.get_summary(),
        }


# ==================== 会话类增强 ====================

@dataclass
class ConversationSession:
    """对话会话"""
    session_id: str
    user_id: str
    agent_id: str
    memory: Optional[Any] = None
    context: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    agent_type: LangGraphAgentType = LangGraphAgentType.REACT
    langgraph_agent: Optional['BaseAgent'] = None
    execution_mode: ExecutionMode = ExecutionMode.ASYNC
    callbacks: List[Any] = field(default_factory=list)
    
    # 扩展字段
    status: SessionStatus = SessionStatus.IDLE
    config: Optional[InferenceConfig] = None
    turn_count: int = 0
    total_tokens: int = 0
    model_name: str = "default"
    system_prompt: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_activity is None:
            self.last_activity = datetime.now()
        if self.config is None:
            self.config = InferenceConfig()
    
    def update_activity(self):
        self.last_activity = datetime.now()
    
    def update_context(self, new_context: Dict[str, Any]):
        if self.context is None:
            self.context = {}
        self.context.update(new_context)
        self.update_activity()
    
    def increment_turn(self, tokens: int = 0):
        self.turn_count += 1
        self.total_tokens += tokens
        self.update_activity()
    
    def get_history(self) -> List[Dict[str, Any]]:
        if  not self.memory:
            return []
        
        try:
            if hasattr(self.memory, 'chat_memory') and hasattr(self.memory.chat_memory, 'messages'):
                messages = []
                for msg in self.memory.chat_memory.messages:
                    if isinstance(msg, HumanMessage):
                        messages.append({"type": "user", "content": msg.content,
                                        "timestamp": datetime.now().isoformat()})
                    elif isinstance(msg, AIMessage):
                        messages.append({"type": "assistant", "content": msg.content,
                                        "timestamp": datetime.now().isoformat()})
                return messages
        except Exception as e:
            logger.warning(f"Failed to get conversation history: {str(e)}")
        return []
    
    def clear_history(self):
        if self.memory and hasattr(self.memory, 'clear'):
            self.memory.clear()
        self.update_activity()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "agent_type": self.agent_type.value,
            "turn_count": self.turn_count,
            "total_tokens": self.total_tokens,
            "model_name": self.model_name,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_activity": self.last_activity.isoformat() if self.last_activity else None,
            "tags": self.tags,
        }
    
    def clone(self, new_session_id: str = None) -> 'ConversationSession':
        """克隆会话"""
        return ConversationSession(
            session_id=new_session_id or str(uuid.uuid4()),
            user_id=self.user_id,
            agent_id=self.agent_id,
            context=copy.deepcopy(self.context),
            agent_type=self.agent_type,
            execution_mode=self.execution_mode,
            config=copy.deepcopy(self.config) if self.config else None,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            tags=self.tags.copy(),
            metadata=copy.deepcopy(self.metadata),
        )


# ==================== 请求队列 ====================

class RequestQueue:
    """请求队列"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._queue: List[InferenceRequest] = []
        self._lock = threading.Lock()
    
    def enqueue(self, request: InferenceRequest) -> bool:
        with self._lock:
            if len(self._queue) >= self.max_size:
                return False
            self._queue.append(request)
            self._queue.sort(key=lambda r: r.priority.value, reverse=True)
            return True
    
    def dequeue(self) -> Optional[InferenceRequest]:
        with self._lock:
            if self._queue:
                return self._queue.pop(0)
            return None
    
    def size(self) -> int:
        with self._lock:
            return len(self._queue)
    
    def clear(self):
        with self._lock:
            self._queue.clear()


# ==================== 主服务类 ====================

class LangChainInferenceService:
    """LangChain推理服务 - 生产级实现
    
    特性：
    - 集成 LangGraph Agent 系统
    - 批量推理和并发控制
    - 请求队列和优先级
    - 多级缓存
    - 熔断器和降级
    - 提示模板管理
    """
    
    def __init__(self, local_model_service: LocalModelService = None,
                 use_langgraph: bool = True, config: Optional[InferenceConfig] = None):
        """初始化推理服务"""
        self.local_model_service = local_model_service or LocalModelService()
        self.config = config or InferenceConfig()
        self.sessions: Dict[str, ConversationSession] = {}
        self.session_manager = SessionHistoryManager()
        
        # LangGraph 集成
        self.use_langgraph = use_langgraph
        self._agent_factory: Optional['AgentFactory'] = None
        self._agent_pool: Optional['AgentPool'] = None
        self._master_factory: Optional['MasterFactory'] = None
        self._agent_orchestrator: Optional['AgentOrchestrator'] = None
        
        if self.use_langgraph:
            self._initialize_langgraph()
        
        # 缓存系统
        self._response_cache: Optional[ResponseCache] = None
        if self.config.enable_cache:
            self._response_cache = ResponseCache(default_ttl=self.config.cache_ttl)
        
        # 提示模板管理
        self._template_manager = PromptTemplateManager()
        
        # 熔断器
        self._circuit_breaker = CircuitBreaker()
        
        # 速率限制
        self._rate_limiter = RateLimiter(rate=20.0, burst=50)
        
        # 请求队列
        self._request_queue = RequestQueue(max_size=100)
        self._queue_processor_running = False
        self._queue_processor_task: Optional[asyncio.Task] = None
        
        # 指标收集
        self._metrics = InferenceMetrics()
        self._metrics_lock = threading.Lock()
        
        # 回调管理
        self._callback_manager: Optional['CallbackManager'] = None
        
        # 默认提示模板
        self.default_prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._template_manager.get("default_assistant").template),
            ("human", "{input}")
        ])
        
        # 并发控制
        self._semaphore = asyncio.Semaphore(10)
        
        logger.info("LangChain inference service initialized")
    
    def _initialize_langgraph(self):
        """初始化 LangGraph"""
        try:
            self._master_factory = get_master_factory()
            self._agent_factory = self._master_factory.agents
            self._agent_pool = AgentPool()
            self._callback_manager = CallbackManager([LoggingCallback(), MetricsCallback()])
            self._agent_orchestrator = AgentOrchestrator(self._agent_pool)
            logger.info("LangGraph initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LangGraph: {str(e)}")
            self.use_langgraph = False
    
    # ==================== 会话管理 ====================
    
    async def create_session(self, user_id: str, agent_id: str,
                           memory_type: str = "buffer", max_token_limit: int = 2000,
                           model_name: str = "default",
                           agent_type: Union[str, LangGraphAgentType] = LangGraphAgentType.REACT,
                           config: Optional[AgentInstanceConfig] = None,
                           system_prompt: str = "",
                           tags: List[str] = None) -> str:
        """创建新的对话会话"""
        session_id = f"session_{uuid.uuid4().hex}"
        
        if isinstance(agent_type, str):
            try:
                agent_type = LangGraphAgentType(agent_type)
            except ValueError:
                agent_type = LangGraphAgentType.REACT
        
        # 创建记忆组件
        memory = None
        try:
            if memory_type == "summary":
                llm = CustomLLM(self.local_model_service, model_name)
                memory = ConversationSummaryMemory(llm=llm, max_token_limit=max_token_limit,
                                                      return_messages=True)
            else:
                memory = ConversationBufferMemory(max_token_limit=max_token_limit,
                                                     return_messages=True)
        except Exception as e:
            logger.warning(f"Failed to create memory component: {str(e)}")
        
        # 创建 LangGraph Agent
        langgraph_agent = None
        if self.use_langgraph and self._agent_factory:
            try:
                langgraph_agent = self._create_langgraph_agent(session_id, agent_type, config, model_name)
            except Exception as e:
                logger.warning(f"Failed to create LangGraph agent: {str(e)}")
        
        # 创建会话
        session = ConversationSession(
            session_id=session_id, user_id=user_id, agent_id=agent_id,
            memory=memory, agent_type=agent_type, langgraph_agent=langgraph_agent,
            model_name=model_name, system_prompt=system_prompt or self._get_default_system_prompt(),
            tags=tags or [], config=InferenceConfig())
        
        self.sessions[session_id] = session
        
        try:
            await self.session_manager.create_session(
                session_id=session_id, user_id=user_id, agent_id=agent_id,
                agent_type=agent_type.value,
                metadata={"memory_type": memory_type, "model_name": model_name,
                         "langgraph_enabled": langgraph_agent is not None})
        except Exception as e:
            logger.warning(f"Failed to create session record: {str(e)}")
        
        logger.info(f"Session created: {session_id}")
        return session_id
    
    def _create_langgraph_agent(self, session_id: str, agent_type: LangGraphAgentType,
                               config: Optional[AgentInstanceConfig],
                               model_name: str) -> Optional['BaseAgent']:
        """创建 LangGraph Agent - 使用增强的工厂方法"""
        if not self._agent_factory:
            return None
        
        try:
            default_config = get_default_config(agent_type)
            llm_client = CustomLLM(self.local_model_service, model_name)
            agent_name = f"{agent_type.value}_{session_id[:8]}"
            
            # 使用增强构建器创建 Agent
            builder = build_enhanced_agent(agent_type.value)
            
            builder = (builder
                .with_name(agent_name)
                .with_llm_client(llm_client)
                .with_model(model_name)
                .with_builtin_tools()    # 添加内置工具
                .with_metrics()           # 启用指标收集
                .with_logging())          # 启用日志
            
            # 配置熔断器和限流器
            builder.with_circuit_breaker(threshold=5, recovery_timeout=30.0)
            builder.with_rate_limiter(rpm=100, burst=10)
            
            # 应用默认配置
            if default_config.get("max_iterations"):
                builder.with_max_iterations(default_config["max_iterations"])
            if default_config.get("timeout"):
                builder.with_timeout(default_config["timeout"])
            
            # 配置执行上下文
            builder.with_execution_context(
                metadata={
                    "session_id": session_id,
                    "agent_type": agent_type.value,
                    "model": model_name
                }
            )
            
            agent = builder.build()
            
            # 配置额外的内置工具（如果可用）
            if hasattr(agent, 'add_recommended_builtin_tools'):
                agent.add_recommended_builtin_tools()
            
            if self._agent_pool:
                self._agent_pool.register(agent_name, agent)
            
            logger.info(f"Created enhanced LangGraph agent: {agent_name}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create LangGraph agent: {str(e)}")
            return None
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示"""
        template = self._template_manager.get("default_assistant")
        return template.template if template else ""
    
    async def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """获取会话"""
        return self.sessions.get(session_id)
    
    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        try:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                if session.langgraph_agent and self._agent_pool:
                    try:
                        self._agent_pool.unregister(session.langgraph_agent.name)
                    except Exception:
                        pass
                del self.sessions[session_id]
            
            await self.session_manager.delete_session(session_id)
            logger.info(f"Session deleted: {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session: {str(e)}")
            return False
    
    async def clone_session(self, session_id: str, new_session_id: str = None) -> Optional[str]:
        """克隆会话"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        new_session = session.clone(new_session_id)
        self.sessions[new_session.session_id] = new_session
        return new_session.session_id
    
    async def list_sessions(self, user_id: str) -> List[str]:
        """列出用户的所有会话"""
        try:
            user_sessions = await self.session_manager.get_user_sessions(user_id)
            return [session.session_id for session in user_sessions]
        except Exception as e:
            logger.error(f"Failed to list sessions: {str(e)}")
            return []
    
    async def clear_session_history(self, session_id: str) -> bool:
        """清除会话历史"""
        try:
            session = self.sessions.get(session_id)
            if session:
                session.clear_history()
                logger.info(f"Session history cleared: {session_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to clear session history: {str(e)}")
            return False
    
    # ==================== 对话方法 ====================
    
    async def chat(self, session_id: str, message: str,
                  context: Optional[Dict[str, Any]] = None,
                  model_name: Optional[str] = None,
                  use_langgraph_agent: bool = True,
                  skip_cache: bool = False,
                  priority: RequestPriority = RequestPriority.NORMAL) -> Optional[str]:
        """进行对话"""
        session = self.sessions.get(session_id)
        if not session:
            logger.error(f"Session not found: {session_id}")
            return None
        
        start_time = time.time()
        from_cache = False
        
        try:
            # 熔断检查
            if not self._circuit_breaker.can_execute():
                return await self._fallback_response(message, context, "服务暂时不可用，请稍后重试")
            
            # 速率限制
            if not await self._rate_limiter.acquire_async():
                raise RateLimitError("Rate limit exceeded")
            
            # 并发控制
            async with self._semaphore:
                # 更新会话上下文
                if context:
                    session.update_context(context)
                
                # 检查缓存
                if self._response_cache and not skip_cache:
                    cached = self._response_cache.get(message, session.context)
                    if cached:
                        from_cache = True
                        response = cached
                        logger.debug(f"Cache hit for session {session_id}")
                    else:
                        response = await self._do_chat(session, message, context, model_name, use_langgraph_agent)
                        self._response_cache.set(message, response, session.context)
                else:
                    response = await self._do_chat(session, message, context, model_name, use_langgraph_agent)
                
                # 记录消息
                await self._record_messages(session_id, message, response)
                
                # 更新指标
                latency_ms = (time.time() - start_time) * 1000
                self._update_metrics(success=True, latency_ms=latency_ms, from_cache=from_cache)
                self._circuit_breaker.record_success()
                
                session.increment_turn()
                logger.debug(f"Chat completed: {session_id}")
                return response
                
        except RateLimitError:
            logger.warning(f"Rate limit exceeded for session {session_id}")
            return await self._fallback_response(message, context, "请求过于频繁，请稍后重试")
        except CircuitBreakerOpenError:
            logger.warning(f"Circuit breaker open for session {session_id}")
            return await self._fallback_response(message, context, "服务暂时不可用")
        except Exception as e:
            logger.error(f"Chat processing failed: {str(e)}")
            self._update_metrics(success=False)
            self._circuit_breaker.record_failure()
            return await self._fallback_response(message, context)
    
    async def _do_chat(self, session: ConversationSession, message: str,
                      context: Optional[Dict[str, Any]], model_name: Optional[str],
                      use_langgraph_agent: bool) -> str:
        """执行实际的对话处理"""
        if use_langgraph_agent and session.langgraph_agent:
            return await self._chat_with_langgraph_agent(session, message, context)
        else:
            return await self._chat_with_langchain(session, message, context, model_name)
    
    async def _chat_with_langgraph_agent(self, session: ConversationSession,
                                        message: str,
                                        context: Optional[Dict[str, Any]] = None) -> str:
        """使用 LangGraph Agent 进行对话"""
        agent = session.langgraph_agent
        if not agent:
            return await self._simple_chat_fallback(message, context)
        
        try:
            input_data = {
                "input": message, "context": context or {},
                "session_id": session.session_id, "user_id": session.user_id
            }
            
            result = await agent.ainvoke(input_data)
            
            if hasattr(result, 'final_answer'):
                return result.final_answer or "处理完成"
            elif isinstance(result, dict):
                return result.get("output", result.get("response", "处理完成"))
            else:
                return str(result)
        except Exception as e:
            logger.error(f"LangGraph agent chat failed: {str(e)}")
            return await self._simple_chat_fallback(message, context)
    
    async def _chat_with_langchain(self, session: ConversationSession,
                                  message: str, context: Optional[Dict[str, Any]] = None,
                                  model_name: Optional[str] = None) -> str:
        """使用 LangChain 进行对话"""
        context_str = json.dumps(session.context or {}, ensure_ascii=False, indent=2)
        
        history_str = ""
        if session.memory:
            try:
                history = session.get_history()
                history_str = "\n".join([f"{msg['type']}: {msg['content']}" for msg in history[-5:]])
            except Exception as e:
                logger.warning(f"Failed to get conversation history: {str(e)}")
        
        model_name = model_name or session.model_name or "default"
        llm = CustomLLM(self.local_model_service, model_name, session.config)
        
        chain = (
            RunnablePassthrough.assign(context=lambda x: context_str, history=lambda x: history_str)
            | self.default_prompt_template
            | llm
            | StrOutputParser()
        )
        
        response = await chain.ainvoke({"input": message})
        
        if session.memory:
            try:
                session.memory.chat_memory.add_user_message(message)
                session.memory.chat_memory.add_ai_message(response)
            except Exception as e:
                logger.warning(f"Failed to update memory: {str(e)}")
        
        return response
    
    async def _simple_chat_fallback(self, message: str,
                                   context: Optional[Dict[str, Any]] = None) -> str:
        """简单对话回退方案"""
        try:
            prompt = f"用户消息: {message}\n\n请作为训练助手回复用户。"
            if context:
                prompt += f"\n\n上下文信息: {json.dumps(context, ensure_ascii=False)}"
            return await self.local_model_service.generate_response(prompt)
        except Exception as e:
            logger.error(f"Fallback chat failed: {str(e)}")
            return "抱歉，我现在无法处理您的请求。请稍后再试或联系技术支持。"
    
    async def _fallback_response(self, message: str, context: Optional[Dict[str, Any]] = None,
                                error_message: str = None) -> str:
        """降级响应"""
        if error_message:
            return error_message
        return await self._simple_chat_fallback(message, context)
    
    async def _record_messages(self, session_id: str, user_message: str, ai_response: str):
        """记录消息到会话管理器"""
        try:
            await self.session_manager.add_message(
                session_id=session_id, message_id=f"user_{uuid.uuid4().hex[:8]}",
                message_type=MessageType.USER, content=user_message)
            await self.session_manager.add_message(
                session_id=session_id, message_id=f"ai_{uuid.uuid4().hex[:8]}",
                message_type=MessageType.ASSISTANT, content=ai_response)
        except Exception as e:
            logger.warning(f"Failed to record messages: {str(e)}")
    
    # ==================== 流式对话 ====================
    
    async def stream_chat(self, session_id: str, message: str,
                         context: Optional[Dict[str, Any]] = None,
                         model_name: Optional[str] = None) -> AsyncIterator[str]:
        """流式对话"""
        session = self.sessions.get(session_id)
        if not session:
            yield "会话不存在"
            return
        
        try:
            if not self._circuit_breaker.can_execute():
                yield "服务暂时不可用，请稍后重试"
                return
            
            if context:
                session.update_context(context)
            
            full_response = ""
            async for chunk in self.local_model_service.stream_response(message, model_name):
                full_response += chunk
                yield chunk
            
            await self._record_messages(session_id, message, full_response)
            session.increment_turn()
            self._circuit_breaker.record_success()
            
        except Exception as e:
            logger.error(f"Stream chat failed: {str(e)}")
            self._circuit_breaker.record_failure()
            yield f"发生错误: {str(e)}"
    
    # ==================== 批量推理 ====================
    
    async def batch_chat(self, requests: List[Dict[str, Any]],
                        max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """批量对话"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_one(req: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                try:
                    session_id = req.get("session_id")
                    message = req.get("message", "")
                    context = req.get("context")
                    
                    response = await self.chat(session_id, message, context)
                    
                    return {
                        "session_id": session_id,
                        "message": message,
                        "response": response,
                        "success": True
                    }
                except Exception as e:
                    return {
                        "session_id": req.get("session_id"),
                        "message": req.get("message", ""),
                        "error": str(e),
                        "success": False
                    }
        
        tasks = [process_one(req) for req in requests]
        results = await asyncio.gather(*tasks)
        return list(results)
    
    async def multi_turn_chat(self, session_id: str, messages: List[Dict[str, str]],
                             provider: str = None) -> Dict[str, Any]:
        """多轮对话支持"""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        try:
            responses = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "user":
                    response = await self.chat(session_id, content)
                    responses.append({"role": "assistant", "content": response})
            
            return {"responses": responses, "session_id": session_id, "message_count": len(messages)}
        except Exception as e:
            logger.error(f"Multi-turn chat failed: {str(e)}")
            return {"error": str(e)}
    
    # ==================== 请求队列处理 ====================
    
    async def enqueue_request(self, session_id: str, message: str,
                             context: Optional[Dict[str, Any]] = None,
                             priority: RequestPriority = RequestPriority.NORMAL,
                             timeout: float = None) -> str:
        """将请求加入队列"""
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            session_id=session_id,
            message=message,
            context=context,
            priority=priority,
            timeout=timeout or self.config.timeout
        )
        
        if self._request_queue.enqueue(request):
            return request.request_id
        else:
            raise InferenceError("Request queue is full")
    
    async def start_queue_processor(self):
        """启动队列处理器"""
        if self._queue_processor_running:
            return
        
        self._queue_processor_running = True
        self._queue_processor_task = asyncio.create_task(self._queue_processor_loop())
        logger.info("Queue processor started")
    
    async def stop_queue_processor(self):
        """停止队列处理器"""
        self._queue_processor_running = False
        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Queue processor stopped")
    
    async def _queue_processor_loop(self):
        """队列处理循环"""
        while self._queue_processor_running:
            try:
                request = self._request_queue.dequeue()
                if request:
                    await self._process_queued_request(request)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Queue processor error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _process_queued_request(self, request: InferenceRequest):
        """处理队列中的请求"""
        try:
            await asyncio.wait_for(
                self.chat(request.session_id, request.message, request.context,
                         priority=request.priority),
                timeout=request.timeout
            )
            logger.debug(f"Queued request processed: {request.request_id}")
        except asyncio.TimeoutError:
            logger.warning(f"Queued request timeout: {request.request_id}")
        except Exception as e:
            logger.error(f"Queued request failed: {request.request_id} - {str(e)}")
    
    # ==================== Agent 管理 ====================
    
    def get_available_agent_types(self) -> List[str]:
        """获取可用的 Agent 类型"""
        return [t.value for t in LangGraphAgentType]
    
    async def change_agent_type(self, session_id: str,
                               new_agent_type: Union[str, LangGraphAgentType]) -> bool:
        """更改会话的 Agent 类型"""
        session = self.sessions.get(session_id)
        if not session:
            return False
        
        try:
            if isinstance(new_agent_type, str):
                new_agent_type = LangGraphAgentType(new_agent_type)
            
            if self.use_langgraph and self._agent_factory:
                new_agent = self._create_langgraph_agent(session_id, new_agent_type, None, session.model_name)
                
                old_agent = session.langgraph_agent
                session.langgraph_agent = new_agent
                session.agent_type = new_agent_type
                
                if old_agent and self._agent_pool:
                    try:
                        self._agent_pool.unregister(old_agent.name)
                    except Exception:
                        pass
            
            return True
        except Exception as e:
            logger.error(f"Failed to change agent type: {str(e)}")
            return False
    
    # ==================== 提示模板管理 ====================
    
    def register_template(self, template: PromptTemplate):
        """注册提示模板"""
        self._template_manager.register(template)
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """获取提示模板"""
        return self._template_manager.get(name)
    
    def list_templates(self, category: str = None) -> List[str]:
        """列出提示模板"""
        return self._template_manager.list_templates(category)
    
    def render_template(self, name: str, **kwargs) -> Optional[str]:
        """渲染提示模板"""
        return self._template_manager.render(name, **kwargs)
    
    # ==================== 缓存管理 ====================
    
    def clear_cache(self):
        """清除响应缓存"""
        if self._response_cache:
            self._response_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        if self._response_cache:
            return self._response_cache.get_stats()
        return {}
    
    # ==================== 指标和状态 ====================
    
    def _update_metrics(self, success: bool, latency_ms: float = 0.0, tokens: int = 0,
                       from_cache: bool = False, timeout: bool = False, retry: bool = False):
        """更新指标"""
        with self._metrics_lock:
            self._metrics.record(success, latency_ms, tokens, from_cache, timeout, retry)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        with self._metrics_lock:
            return self._metrics.get_summary()
    
    def reset_metrics(self):
        """重置指标"""
        with self._metrics_lock:
            self._metrics = InferenceMetrics()
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "langgraph_enabled": self.use_langgraph,
            "active_sessions": len(self.sessions),
            "queue_size": self._request_queue.size(),
            "queue_processor_running": self._queue_processor_running,
            "circuit_breaker": self._circuit_breaker.get_status(),
            "rate_limiter": self._rate_limiter.get_status(),
            "cache_stats": self.get_cache_stats(),
            "metrics": self.get_metrics(),
            "available_agent_types": self.get_available_agent_types(),
            "available_templates": self.list_templates(),
            "local_model_service": self.local_model_service.get_service_status(),
            "sessions": {
                session_id: session.to_dict()
                for session_id, session in self.sessions.items()
            }
        }
    
    def get_available_providers(self) -> List[str]:
        """获取可用的推理提供者"""
        providers = ["local"]
        if self.local_model_service:
            models = self.local_model_service.list_models()
            providers.extend(models)
        return list(set(providers))
    
    async def get_provider_status(self, provider: str = None) -> Dict[str, Any]:
        """获取推理提供者状态"""
        if provider:
            return {
                "provider": provider,
                "available": provider in self.get_available_providers(),
                "status": "active"
            }
        return {
            "providers": {
                p: {"available": True, "status": "active"}
                for p in self.get_available_providers()
            }
        }
    
    # ==================== 健康检查 ====================
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查 - 使用增强的工厂健康检查"""
        checks = {
            "status": "healthy",
            "components": {}
        }
        
        # 检查 LangChain
        checks["components"]["langchain"] = {
        }
        
        # 检查 LangGraph
        checks["components"]["langgraph"] = {
            "enabled": self.use_langgraph,
            "status": "ok" if self.use_langgraph else "disabled"
        }
        
        # 使用工厂健康检查方法
        try:
            factory_health = factory_health_check()
            checks["components"]["factory"] = {
                "status": factory_health.get("status", "unknown"),
                "factories": factory_health.get("factories", {}),
                "errors": factory_health.get("errors", [])
            }
            if factory_health.get("status") != "healthy":
                checks["status"] = "degraded"
        except Exception as e:
            checks["components"]["factory"] = {
                "status": "error",
                "error": str(e)
            }
        
        # 检查熔断器
        circuit_status = self._circuit_breaker.get_status()
        checks["components"]["circuit_breaker"] = {
            "state": circuit_status["state"],
            "status": "ok" if circuit_status["state"] == "closed" else "warning"
        }
        
        # 检查本地模型服务
        try:
            model_status = self.local_model_service.get_service_status()
            checks["components"]["local_model_service"] = {
                "status": "ok",
                "details": model_status
            }
        except Exception as e:
            checks["components"]["local_model_service"] = {
                "status": "error",
                "error": str(e)
            }
            checks["status"] = "degraded"
        
        # 检查会话管理
        checks["components"]["sessions"] = {
            "active_count": len(self.sessions),
            "status": "ok"
        }
        
        # 检查 Agent 池状态
        if self._agent_pool:
            try:
                pool_agents = self._agent_pool.list_agents() if hasattr(self._agent_pool, 'list_agents') else []
                checks["components"]["agent_pool"] = {
                    "active_agents": len(pool_agents),
                    "status": "ok"
                }
            except Exception as e:
                checks["components"]["agent_pool"] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return checks
    
    # ==================== 会话清理 ====================
    
    async def cleanup_expired_sessions(self, hours: int = 24):
        """清理过期会话"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                if session.last_activity and session.last_activity < cutoff_time:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                await self.delete_session(session_id)
            
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            return len(expired_sessions)
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {str(e)}")
            return 0
    
    async def cleanup(self):
        """清理资源"""
        await self.stop_queue_processor()
        await self.cleanup_expired_sessions(hours=0)
        self.clear_cache()
        logger.info("Inference service cleanup completed")
    
    # ==================== LangGraph 工业级派生方法 ====================
    
    def _get_master_factory(self) -> Optional['MasterFactory']:
        """获取 MasterFactory 实例
        
        调用 backend.algo.langgraph.factory.get_master_factory
        
        Returns:
            MasterFactory 实例或 None
        """
        try:
            return get_master_factory()
        except Exception as e:
            logger.error(f"Failed to get master factory: {e}")
            return None
    
    def create_langgraph_agent_via_factory(
        self,
        agent_type: str = "react",
        name: str = None,
        tools: List[Any] = None,
        llm_client: Any = None,
        config: Dict[str, Any] = None
    ) -> Optional['BaseAgent']:
        """通过 MasterFactory 创建 LangGraph Agent
        
        调用 MasterFactory.agents.create
        
        Args:
            agent_type: Agent 类型
            name: Agent 名称
            tools: 工具列表
            llm_client: LLM 客户端
            config: 配置字典
            
        Returns:
            LangGraph Agent 实例或 None
        """
        factory = self._get_master_factory()
        if not factory:
            logger.warning("MasterFactory not available")
            return None
        
        try:
            agent_name = name or f"agent_{uuid.uuid4().hex[:8]}"
            
            # 使用默认的 llm_client 如果未提供
            actual_llm_client = llm_client
            if not actual_llm_client:
                actual_llm_client = CustomLLM(self.local_model_service)

            agent = factory.agents.create(
                agent_type,
                name=agent_name,
                tools=tools or [],
                llm_client=actual_llm_client,
                **(config or {})
            )
            
            # 注册到 Agent 池
            if self._agent_pool and agent:
                self._agent_pool.register(agent_name, agent)
                logger.info(f"Created and registered LangGraph agent: {agent_name}")
            
            return agent
        except Exception as e:
            logger.error(f"Failed to create LangGraph agent: {e}")
            return None
    
    def create_agent_message_via_factory(
        self,
        content: str,
        message_type: str = "human",
        name: str = None,
        metadata: Dict[str, Any] = None
    ) -> Optional['AgentMessage']:
        """通过 MasterFactory 创建 Agent 消息
        
        调用 MasterFactory.create_agent_message
        
        Args:
            content: 消息内容
            message_type: 消息类型 (human, ai, system, tool)
            name: 发送者名称
            metadata: 元数据
            
        Returns:
            AgentMessage 实例或 None
        """
        factory = self._get_master_factory()
        if not factory:
            return None
        
        try:
            type_map = {
                "human": AgentMessageType.HUMAN,
                "ai": AgentMessageType.AI,
                "system": AgentMessageType.SYSTEM,
                "tool": AgentMessageType.TOOL
            }
            msg_type = type_map.get(message_type.lower(), AgentMessageType.HUMAN)
            
            return factory.create_agent_message(
                content=content,
                message_type=msg_type,
                name=name,
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Failed to create agent message: {e}")
            return None
    
    def create_memory_entry_via_factory(
        self,
        content: Any,
        memory_type: str = "short_term",
        importance: float = 0.5,
        tags: List[str] = None
    ) -> Optional['MemoryEntry']:
        """通过 MasterFactory 创建记忆条目
        
        调用 MasterFactory.create_memory_entry
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性分数
            tags: 标签列表
            
        Returns:
            MemoryEntry 实例或 None
        """
        factory = self._get_master_factory()
        if not factory:
            return None
        
        try:
            mem_type = factory.get_memory_type(memory_type)
            return factory.create_memory_entry(
                content=content,
                memory_type=mem_type,
                importance=importance,
                tags=tags
            )
        except Exception as e:
            logger.error(f"Failed to create memory entry: {e}")
            return None
    
    def create_tool_call_via_factory(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        tool_call_id: str = None
    ) -> Optional['ToolCall']:
        """通过 MasterFactory 创建工具调用
        
        调用 MasterFactory.create_tool_call
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            tool_call_id: 工具调用 ID
            
        Returns:
            ToolCall 实例或 None
        """
        factory = self._get_master_factory()
        if not factory:
            return None
        
        try:
            return factory.create_tool_call(
                name=tool_name,
                arguments=arguments,
                tool_call_id=tool_call_id or f"call_{uuid.uuid4().hex[:8]}"
            )
        except Exception as e:
            logger.error(f"Failed to create tool call: {e}")
            return None
    
    def create_tool_result_via_factory(
        self,
        tool_call_id: str,
        tool_name: str,
        result: Any,
        success: bool = True,
        error: str = None
    ) -> Optional['ToolResult']:
        """通过 MasterFactory 创建工具结果
        
        调用 MasterFactory.create_tool_result
        
        Args:
            tool_call_id: 工具调用 ID
            tool_name: 工具名称
            result: 结果
            success: 是否成功
            error: 错误信息
            
        Returns:
            ToolResult 实例或 None
        """
        factory = self._get_master_factory()
        if not factory:
            return None
        
        try:
            return factory.create_tool_result(
                tool_call_id=tool_call_id,
                name=tool_name,
                result=result,
                success=success,
                error=error
            )
        except Exception as e:
            logger.error(f"Failed to create tool result: {e}")
            return None
    
    def setup_agent_tool_cache(
        self,
        agent_name: str,
        max_size: int = 1000,
        default_ttl: int = 300
    ) -> bool:
        """为 Agent 设置工具缓存
        
        调用 AgentPool.setup_tool_cache_for_agent
        
        Args:
            agent_name: Agent 名称
            max_size: 缓存最大大小
            default_ttl: 默认 TTL
            
        Returns:
            是否成功
        """
        if not self._agent_pool:
            return False
        
        try:
            return self._agent_pool.setup_tool_cache_for_agent(
                agent_name, max_size, default_ttl
            )
        except Exception as e:
            logger.error(f"Failed to setup tool cache: {e}")
            return False
    
    def setup_agent_retry_handler(
        self,
        agent_name: str,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0
    ) -> bool:
        """为 Agent 设置重试处理器
        
        调用 AgentPool.setup_retry_handler_for_agent
        
        Args:
            agent_name: Agent 名称
            max_retries: 最大重试次数
            base_delay: 基础延迟
            max_delay: 最大延迟
            
        Returns:
            是否成功
        """
        if not self._agent_pool:
            return False
        
        try:
            return self._agent_pool.setup_retry_handler_for_agent(
                agent_name, max_retries, base_delay, max_delay
            )
        except Exception as e:
            logger.error(f"Failed to setup retry handler: {e}")
            return False
    
    def setup_agent_rate_limiter(
        self,
        agent_name: str,
        rate: float = 10.0,
        burst: int = 20
    ) -> bool:
        """为 Agent 设置限流器
        
        调用 AgentPool.setup_rate_limiter_for_agent
        
        Args:
            agent_name: Agent 名称
            rate: 速率
            burst: 突发数
            
        Returns:
            是否成功
        """
        if not self._agent_pool:
            return False
        
        try:
            return self._agent_pool.setup_rate_limiter_for_agent(
                agent_name, rate, burst
            )
        except Exception as e:
            logger.error(f"Failed to setup rate limiter: {e}")
            return False
    
    def create_enhanced_checkpoint_for_agent(
        self,
        agent_name: str,
        checkpoint_id: str = None,
        branch: str = "main",
        tags: List[str] = None
    ) -> Optional[Any]:
        """为 Agent 创建增强检查点
        
        调用 AgentPool.create_enhanced_checkpoint_for_agent
        
        Args:
            agent_name: Agent 名称
            checkpoint_id: 检查点 ID
            branch: 分支名称
            tags: 标签列表
            
        Returns:
            EnhancedCheckpoint 实例或 None
        """
        if not self._agent_pool:
            return None
        
        try:
            return self._agent_pool.create_enhanced_checkpoint_for_agent(
                agent_name, checkpoint_id, branch, tags
            )
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return None
    
    def run_langgraph_agent(
        self,
        agent_name: str,
        input_data: Any,
        **kwargs
    ) -> Optional['AgentState']:
        """运行 LangGraph Agent
        
        调用 AgentPool.run
        
        Args:
            agent_name: Agent 名称
            input_data: 输入数据
            **kwargs: 其他参数
            
        Returns:
            AgentState 结果或 None
        """
        if not self._agent_pool:
            return None
        
        try:
            return self._agent_pool.run(agent_name, input_data, **kwargs)
        except Exception as e:
            logger.error(f"Failed to run LangGraph agent: {e}")
            return None
    
    def run_agents_workflow(
        self,
        workflow: List[Tuple[str, Any]],
        stop_on_error: bool = False
    ) -> List['AgentState']:
        """执行 Agent 工作流
        
        调用 AgentOrchestrator.run_workflow
        
        Args:
            workflow: 工作流定义 [(agent_name, input_data), ...]
            stop_on_error: 遇错是否停止
            
        Returns:
            各步骤结果列表
        """
        if not self._agent_orchestrator:
            return []
        
        try:
            return self._agent_orchestrator.run_workflow(workflow, stop_on_error)
        except Exception as e:
            logger.error(f"Failed to run workflow: {e}")
            return []
    
    def run_agents_parallel(
        self,
        tasks: List[Tuple[str, Any]],
        max_workers: int = 4
    ) -> List[Tuple[str, 'AgentState']]:
        """并行执行多个 Agent 任务
        
        调用 AgentOrchestrator.run_parallel
        
        Args:
            tasks: 任务列表 [(agent_name, input_data), ...]
            max_workers: 最大并行数
            
        Returns:
            结果列表 [(agent_name, result), ...]
        """
        if not self._agent_orchestrator:
            return []
        
        try:
            return self._agent_orchestrator.run_parallel(tasks, max_workers)
        except Exception as e:
            logger.error(f"Failed to run parallel tasks: {e}")
            return []
    
    def run_workflow_with_events(
        self,
        workflow: List[Tuple[str, Any]],
        stop_on_error: bool = False
    ) -> Tuple[List['AgentState'], List[Any]]:
        """执行工作流并生成执行事件
        
        调用 AgentOrchestrator.run_workflow_with_events
        
        Args:
            workflow: 工作流定义
            stop_on_error: 遇错是否停止
            
        Returns:
            (结果列表, 事件列表)
        """
        if not self._agent_orchestrator:
            return [], []
        
        try:
            return self._agent_orchestrator.run_workflow_with_events(workflow, stop_on_error)
        except Exception as e:
            logger.error(f"Failed to run workflow with events: {e}")
            return [], []
    
    def define_workflow_pipeline(
        self,
        name: str,
        steps: List[Dict[str, Any]]
    ) -> None:
        """定义工作流流水线
        
        调用 AgentOrchestrator.define_pipeline
        
        Args:
            name: 流水线名称
            steps: 步骤定义列表
        """
        if self._agent_orchestrator:
            self._agent_orchestrator.define_pipeline(name, steps)
    
    def run_workflow_pipeline(
        self,
        pipeline_name: str,
        initial_input: Any = None
    ) -> List['AgentState']:
        """执行工作流流水线
        
        调用 AgentOrchestrator.run_pipeline
        
        Args:
            pipeline_name: 流水线名称
            initial_input: 初始输入
            
        Returns:
            各步骤结果列表
        """
        if not self._agent_orchestrator:
            return []
        
        try:
            return self._agent_orchestrator.run_pipeline(pipeline_name, initial_input)
        except Exception as e:
            logger.error(f"Failed to run pipeline: {e}")
            return []
    
    def create_priority_router(self, name: str = "priority_router") -> Optional[Any]:
        """创建优先级路由器
        
        调用 MasterFactory.create_priority_router
        
        Args:
            name: 路由器名称
            
        Returns:
            PriorityRouter 实例或 None
        """
        factory = self._get_master_factory()
        if not factory:
            return None
        
        return factory.create_priority_router(name)
    
    def create_weighted_router(self, name: str = "weighted_router") -> Optional[Any]:
        """创建权重路由器
        
        调用 MasterFactory.create_weighted_router
        
        Args:
            name: 路由器名称
            
        Returns:
            WeightedRouter 实例或 None
        """
        factory = self._get_master_factory()
        if not factory:
            return None
        
        return factory.create_weighted_router(name)
    
    def create_load_balance_router(self, name: str = "lb_router") -> Optional[Any]:
        """创建负载均衡路由器
        
        调用 MasterFactory.create_load_balance_router
        
        Args:
            name: 路由器名称
            
        Returns:
            LoadBalanceRouter 实例或 None
        """
        factory = self._get_master_factory()
        if not factory:
            return None
        
        return factory.create_load_balance_router(name)
    
    def create_ab_test_router(
        self,
        name: str = "ab_router",
        test_name: str = ""
    ) -> Optional[Any]:
        """创建 A/B 测试路由器
        
        调用 MasterFactory.create_ab_test_router
        
        Args:
            name: 路由器名称
            test_name: 测试名称
            
        Returns:
            ABTestRouter 实例或 None
        """
        factory = self._get_master_factory()
        if not factory:
            return None
        
        return factory.create_ab_test_router(name, test_name)
    
    def get_langgraph_pool_metrics(self) -> Dict[str, Any]:
        """获取 LangGraph Agent 池指标
        
        调用 AgentPool.get_pool_metrics
        
        Returns:
            池指标字典
        """
        if not self._agent_pool:
            return {"error": "Agent pool not available"}
        
        try:
            return self._agent_pool.get_pool_metrics()
        except Exception as e:
            return {"error": str(e)}
    
    def get_langgraph_pool_health(self) -> Dict[str, Any]:
        """获取 LangGraph 池健康状态
        
        调用 AgentPool.get_pool_health_with_factory
        
        Returns:
            健康状态字典
        """
        if not self._agent_pool:
            return {"error": "Agent pool not available"}
        
        try:
            return self._agent_pool.get_pool_health_with_factory()
        except Exception as e:
            return {"error": str(e)}
    
    def get_orchestrator_health(self) -> Dict[str, Any]:
        """获取编排器健康状态
        
        调用 AgentOrchestrator.get_orchestrator_health_with_factory
        
        Returns:
            健康状态字典
        """
        if not self._agent_orchestrator:
            return {"error": "Orchestrator not available"}
        
        try:
            return self._agent_orchestrator.get_orchestrator_health_with_factory()
        except Exception as e:
            return {"error": str(e)}
    
    def get_factory_health(self) -> Dict[str, Any]:
        """获取 MasterFactory 健康状态
        
        调用 MasterFactory.health_check
        
        Returns:
            健康状态字典
        """
        factory = self._get_master_factory()
        if not factory:
            return {"error": "Factory not available"}
        
        try:
            return factory.health_check()
        except Exception as e:
            return {"error": str(e)}
    
    def get_langgraph_comprehensive_health(self) -> Dict[str, Any]:
        """获取 LangGraph 综合健康状态
        
        Returns:
            综合健康状态字典
        """
        health = {"timestamp": datetime.now().isoformat(), "components": {}, "status": "healthy"}

        # 工厂健康
        factory_health = self.get_factory_health()
        health["components"]["factory"] = factory_health
        if "error" in factory_health:
            health["status"] = "degraded"
        
        # 池健康
        pool_health = self.get_langgraph_pool_health()
        health["components"]["pool"] = pool_health
        if "error" in pool_health:
            health["status"] = "degraded"
        
        # 编排器健康
        orch_health = self.get_orchestrator_health()
        health["components"]["orchestrator"] = orch_health
        if "error" in orch_health:
            health["status"] = "degraded"
        
        return health


# ==================== 全局服务实例 ====================

_inference_service: Optional[LangChainInferenceService] = None


def get_langchain_inference_service(
    local_model_service: Optional[LocalModelService] = None,
    use_langgraph: bool = True
) -> LangChainInferenceService:
    """获取LangChain推理服务实例"""
    global _inference_service
    
    if _inference_service is None:
        if local_model_service is None:
            local_model_service = LocalModelService()
        _inference_service = LangChainInferenceService(local_model_service, use_langgraph)
    
    return _inference_service


def set_langchain_inference_service(service: LangChainInferenceService):
    """设置LangChain推理服务实例"""
    global _inference_service
    _inference_service = service


def reset_langchain_inference_service():
    """重置LangChain推理服务实例"""
    global _inference_service
    if _inference_service:
        asyncio.create_task(_inference_service.cleanup())
    _inference_service = None

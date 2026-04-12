"""训练助手智能体

该模块实现了训练助手智能体，用于与训练平台交互，
集成 LangGraph Agent 系统，提供智能对话和训练管理功能。

特性：
- 基于 LangGraph 的生产级 Agent 实现
- 支持多种 Agent 类型（ReAct、Plan&Execute、Reflexion 等）
- 支持策略模式和回调系统
- 支持工具调用和状态管理
- 支持检查点和恢复
- 批量训练任务管理
- 训练参数优化建议
- 事件通知系统
- 性能监控和指标收集
"""

from typing import Dict, Any, List, Optional, AsyncIterator, TYPE_CHECKING
import logging
import asyncio
import uuid
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod

from backend.schemas.agent import Agent
from backend.services.training_service import get_training_service
from backend.services.training_history_service import get_training_history_service
from backend.services.model_service import ModelService
from backend.repositories.model_repository import ModelRepository

from .local_model_service import LocalModelService
from .langchain_inference_service import get_langchain_inference_service
from .session_history_manager import SessionHistoryManager
from .agent_type import LangGraphAgentType, ExecutionMode

# LangGraph 集成
from backend.algo.langgraph.agents import (
    BaseAgent, LoggingCallback, MetricsCallback,
    CallbackManager, ExecutionTracer, StrategyManager, get_strategy_manager
)
from backend.algo.langgraph.factory import (
    MasterFactory, get_master_factory,
    build_enhanced_agent, get_agent_diagnostics, factory_health_check
)
from backend.algo.langgraph.tools import (
    Tool, ToolRegistry, get_global_registry, tool
)
from backend.algo.langgraph.builtin_tools import get_tools_for_agent
from backend.algo.langgraph.checkpointer import (
    Checkpointer, create_memory_checkpointer
)
from backend.algo.langgraph.state import AgentState

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class AssistantStatus(Enum):
    """助手状态枚举"""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    PAUSED = "paused"
    TERMINATED = "terminated"


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class EventType(Enum):
    """事件类型枚举"""
    AGENT_STARTED = "agent_started"
    AGENT_STOPPED = "agent_stopped"
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TRAINING_STARTED = "training_started"
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"
    MODEL_DOWNLOADED = "model_downloaded"
    ERROR_OCCURRED = "error_occurred"


class ConversationMode(Enum):
    """对话模式枚举"""
    NORMAL = "normal"
    GUIDED = "guided"
    EXPERT = "expert"
    DEBUG = "debug"


# ==================== 异常定义 ====================

class TrainingAssistantError(Exception):
    """训练助手基础异常"""


class TaskExecutionError(TrainingAssistantError):
    """任务执行异常"""


class SessionError(TrainingAssistantError):
    """会话异常"""
    pass


class PermissionDeniedError(TrainingAssistantError):
    """权限异常"""


# ==================== 数据类定义 ====================

@dataclass
class TrainingAssistantConfig:
    """训练助手配置"""
    name: str = "training_assistant"
    agent_type: LangGraphAgentType = LangGraphAgentType.PLAN_EXECUTE
    
    # 执行配置
    max_iterations: int = 15
    timeout: float = 300.0
    execution_mode: ExecutionMode = ExecutionMode.ASYNC
    
    # LLM 配置
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: Optional[str] = None
    
    # 功能开关
    enable_inference: bool = True
    enable_tools: bool = True
    enable_memory: bool = True
    enable_checkpointing: bool = True
    enable_callbacks: bool = True
    enable_strategies: bool = True
    enable_batch_processing: bool = True
    enable_auto_optimization: bool = True
    enable_event_notifications: bool = True
    
    # 高级配置
    enable_replanning: bool = True
    max_reflections: int = 3
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 批量处理配置
    max_concurrent_tasks: int = 5
    task_queue_size: int = 100
    task_timeout: float = 600.0
    
    # 缓存配置
    enable_response_cache: bool = True
    cache_ttl: int = 3600
    
    # 监控配置
    metrics_enabled: bool = True
    metrics_interval: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingAssistantConfig':
        if "agent_type" in data and isinstance(data["agent_type"], str):
            data["agent_type"] = LangGraphAgentType(data["agent_type"])
        if "execution_mode" in data and isinstance(data["execution_mode"], str):
            data["execution_mode"] = ExecutionMode(data["execution_mode"])
        # Fix: Replace __dataclass_fields__ with fields from dataclasses module
        from dataclasses import fields
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in field_names})


@dataclass
class TrainingTask:
    """训练任务"""
    task_id: str
    user_id: str
    task_type: str
    parameters: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
        }


@dataclass
class ExecutionMetrics:
    """执行指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    total_tokens_used: int = 0
    tool_calls_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def record_request(self, success: bool, latency_ms: float, tokens: int = 0):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.total_latency_ms += latency_ms
        self.total_tokens_used += tokens
    
    def get_summary(self) -> Dict[str, Any]:
        avg_latency = self.total_latency_ms / self.total_requests if self.total_requests > 0 else 0
        success_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 0
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": success_rate,
            "average_latency_ms": avg_latency,
            "total_tokens_used": self.total_tokens_used,
            "tool_calls_count": self.tool_calls_count,
            "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) 
                             if (self.cache_hits + self.cache_misses) > 0 else 0,
        }


@dataclass
class AssistantEvent:
    """助手事件"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }


@dataclass
class ConversationContext:
    """对话上下文"""
    session_id: str
    user_id: str
    mode: ConversationMode = ConversationMode.NORMAL
    turn_count: int = 0
    last_intent: Optional[str] = None
    pending_actions: List[Dict[str, Any]] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history_summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==================== 事件系统 ====================

class EventListener(ABC):
    """事件监听器抽象基类"""
    @abstractmethod
    async def on_event(self, event: AssistantEvent):
        pass


class LoggingEventListener(EventListener):
    """日志事件监听器"""
    async def on_event(self, event: AssistantEvent):
        """记录事件到日志"""
        logger.info("Assistant event: %s - %s", event.event_type.value, event.data)


class WebhookEventListener(EventListener):
    """Webhook 事件监听器"""
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def on_event(self, event: AssistantEvent):
        """处理事件并发送到 webhook"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                await session.post(self.webhook_url, json=event.to_dict())
        except (OSError, ConnectionError, RuntimeError) as e:
            logger.error("Webhook notification failed: %s", str(e))


class AssistantEventManager:
    """助手事件管理器"""
    
    def __init__(self):
        self._listeners: Dict[EventType, List[EventListener]] = {}
        self._all_listeners: List[EventListener] = []
        self._event_history: List[AssistantEvent] = []
        self._max_history = 1000
        self._lock = threading.Lock()
    
    def add_listener(self, listener: EventListener, event_types: List[EventType] = None):
        with self._lock:
            if event_types:
                for event_type in event_types:
                    if event_type not in self._listeners:
                        self._listeners[event_type] = []
                    self._listeners[event_type].append(listener)
            else:
                self._all_listeners.append(listener)
    
    def remove_listener(self, listener: EventListener):
        with self._lock:
            self._all_listeners = [l for l in self._all_listeners if l != listener]
            for event_type in self._listeners:
                self._listeners[event_type] = [l for l in self._listeners[event_type] if l != listener]
    
    async def emit(self, event: AssistantEvent):
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
            listeners = list(self._all_listeners)
            if event.event_type in self._listeners:
                listeners.extend(self._listeners[event.event_type])
        
        for listener in listeners:
            try:
                await listener.on_event(event)
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.error("Event listener error: %s", str(e))
    
    def get_history(self, limit: int = 100, event_types: List[EventType] = None) -> List[AssistantEvent]:
        with self._lock:
            events = self._event_history[-limit:]
            if event_types:
                events = [e for e in events if e.event_type in event_types]
            return events


# ==================== 任务队列 ====================

class TaskQueue:
    """任务队列"""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._tasks: Dict[str, TrainingTask] = {}
        self._pending_queue: List[str] = []
        self._lock = threading.Lock()
    
    def add_task(self, task: TrainingTask) -> bool:
        with self._lock:
            if len(self._tasks) >= self.max_size:
                return False
            self._tasks[task.task_id] = task
            self._pending_queue.append(task.task_id)
            self._pending_queue.sort(key=lambda tid: self._tasks[tid].priority.value, reverse=True)
            return True
    
    def get_next_task(self) -> Optional[TrainingTask]:
        with self._lock:
            if not self._pending_queue:
                return None
            task_id = self._pending_queue.pop(0)
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()
            return task
    
    def complete_task(self, task_id: str, result: Dict[str, Any] = None, error: str = None):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.completed_at = datetime.now()
                if error:
                    task.status = TaskStatus.FAILED
                    task.error = error
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
    
    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        with self._lock:
            return self._tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status in [TaskStatus.PENDING, TaskStatus.QUEUED]:
                task.status = TaskStatus.CANCELLED
                if task_id in self._pending_queue:
                    self._pending_queue.remove(task_id)
                return True
            return False
    
    def get_pending_count(self) -> int:
        with self._lock:
            return len(self._pending_queue)
    
    def get_all_tasks(self, user_id: str = None, status: TaskStatus = None) -> List[TrainingTask]:
        with self._lock:
            tasks = list(self._tasks.values())
            if user_id:
                tasks = [t for t in tasks if t.user_id == user_id]
            if status:
                tasks = [t for t in tasks if t.status == status]
            return tasks
    
    def cleanup_completed(self, older_than_hours: int = 24):
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=older_than_hours)
            to_remove = [
                tid for tid, task in self._tasks.items()
                if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
                and task.completed_at and task.completed_at < cutoff
            ]
            for tid in to_remove:
                del self._tasks[tid]


# ==================== 响应缓存 ====================

class ResponseCache:
    """响应缓存"""
    
    def __init__(self, max_size: int = 500, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def _get_cache_key(self, prompt: str, context_hash: str = "") -> str:
        import hashlib
        key_data = f"{prompt}:{context_hash}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, prompt: str, context_hash: str = "") -> Optional[str]:
        key = self._get_cache_key(prompt, context_hash)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() < entry["expires_at"]:
                    return entry["response"]
                else:
                    del self._cache[key]
            return None
    
    def set(self, prompt: str, response: str, context_hash: str = "", ttl: int = None):
        key = self._get_cache_key(prompt, context_hash)
        with self._lock:
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["created_at"])
                del self._cache[oldest_key]
            
            self._cache[key] = {
                "response": response,
                "created_at": time.time(),
                "expires_at": time.time() + (ttl or self.default_ttl)
            }
    
    def clear(self):
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"size": len(self._cache), "max_size": self.max_size}


# ==================== 意图检测器 ====================

class IntentDetector:
    """意图检测器"""
    
    INTENT_PATTERNS = {
        "create_training": ["创建训练", "开始训练", "新建训练", "启动训练", "发起训练"],
        "check_progress": ["查看进度", "训练进度", "进度如何", "训练状态", "进展情况"],
        "download_model": ["下载模型", "模型下载", "导出模型", "获取模型"],
        "view_history": ["训练历史", "历史记录", "过往训练", "以前的训练"],
        "view_statistics": ["训练统计", "统计信息", "数据统计", "汇总数据"],
        "stop_training": ["停止训练", "终止训练", "取消训练", "暂停训练"],
        "resume_training": ["继续训练", "恢复训练", "重启训练"],
        "optimize_params": ["优化参数", "参数建议", "最佳参数", "调参建议"],
        "compare_models": ["比较模型", "模型对比", "对比分析"],
        "help": ["帮助", "help", "怎么用", "使用说明"],
        "greeting": ["你好", "hello", "hi", "嗨"],
    }
    
    @classmethod
    def detect(cls, text: str) -> Optional[str]:
        text_lower = text.lower()
        for intent, patterns in cls.INTENT_PATTERNS.items():
            if any(pattern in text_lower for pattern in patterns):
                return intent
        return None
    
    @classmethod
    def get_confidence(cls, text: str, intent: str) -> float:
        if intent not in cls.INTENT_PATTERNS:
            return 0.0
        text_lower = text.lower()
        patterns = cls.INTENT_PATTERNS[intent]
        matches = sum(1 for p in patterns if p in text_lower)
        return min(1.0, matches / len(patterns) * 2)


# ==================== 训练参数优化器 ====================

class TrainingParamOptimizer:
    """训练参数优化器"""
    
    DEFAULT_PARAMS = {
        "epochs": 3,
        "batch_size": 8,
        "learning_rate": 0.001,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 1,
    }
    
    OPTIMIZATION_RULES = {
        "small_dataset": {"epochs": 5, "batch_size": 4, "learning_rate": 0.0005},
        "large_dataset": {"epochs": 2, "batch_size": 16, "learning_rate": 0.002},
        "fine_tuning": {"epochs": 3, "learning_rate": 0.00005, "warmup_steps": 200},
        "quick_training": {"epochs": 1, "batch_size": 16},
        "high_quality": {"epochs": 10, "batch_size": 4, "learning_rate": 0.0001},
    }
    
    @classmethod
    def get_optimized_params(cls, dataset_size: int = None, model_type: str = None,
                            training_goal: str = None, user_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """获取优化后的训练参数"""
        params = cls.DEFAULT_PARAMS.copy()
        
        if dataset_size:
            if dataset_size < 1000:
                params.update(cls.OPTIMIZATION_RULES["small_dataset"])
            elif dataset_size > 10000:
                params.update(cls.OPTIMIZATION_RULES["large_dataset"])
        
        if training_goal:
            if training_goal in cls.OPTIMIZATION_RULES:
                params.update(cls.OPTIMIZATION_RULES[training_goal])
        
        if user_params:
            params.update(user_params)
        
        return params
    
    @classmethod
    def get_recommendations(cls, current_params: Dict[str, Any],
                           training_history: List[Dict[str, Any]] = None) -> List[str]:
        """获取参数优化建议"""
        recommendations = []
        
        if current_params.get("learning_rate", 0) > 0.01:
            recommendations.append("学习率较高，可能导致训练不稳定，建议降低到 0.001 左右")
        
        if current_params.get("batch_size", 0) < 4:
            recommendations.append("批量大小较小，可能导致梯度估计不准确，建议增加到 8 或更大")
        
        if current_params.get("epochs", 0) > 10:
            recommendations.append("训练轮数较多，注意观察是否过拟合")
        
        if training_history:
            last_loss = training_history[-1].get("loss") if training_history else None
            if last_loss and last_loss > 1.0:
                recommendations.append("训练损失较高，建议检查数据质量或调整学习率")
        
        return recommendations


# ==================== 主类 ====================

class TrainingAssistantAgent:
    """训练助手智能体 - 基于 LangGraph 的生产级实现
    
    特性：
    - 集成 LangGraph Agent 系统
    - 支持多种 Agent 类型
    - 支持策略模式
    - 支持工具调用
    - 支持检查点和恢复
    - 支持回调和指标收集
    - 批量任务处理
    - 训练参数优化
    - 事件通知系统
    """
    
    def __init__(self, agent_model: Agent, config: Optional[TrainingAssistantConfig] = None):
        """初始化训练助手智能体"""
        # 基础属性
        self.agent_model = agent_model
        self.agent_id = agent_model.agent_id
        self.name = agent_model.name
        self.description = agent_model.description
        self.capabilities = getattr(agent_model, "capabilities", [])
        self.status = AssistantStatus.IDLE
        self.role = "training_assistant"
        
        # 配置
        self.config = config or TrainingAssistantConfig()
        
        # 推理提供者配置
        self.default_provider = "local"
        self.available_providers = ["local", "chatgpt", "deepseek"]
        
        # 系统提示词
        self.system_prompt = self.config.system_prompt or self._get_default_system_prompt()
        
        # 初始化相关服务
        self.training_service = get_training_service()
        self.training_history_service = get_training_history_service()
        self.model_repository = ModelRepository()
        self.model_service = ModelService(self.model_repository)
        
        # 初始化本地服务
        self.local_model_service = LocalModelService(enable_tools=self.config.enable_tools)
        self.inference_service = get_langchain_inference_service(
            self.local_model_service, use_langgraph=True)
        self.session_manager = SessionHistoryManager()
        
        # LangGraph 组件
        self._langgraph_agent: Optional['BaseAgent'] = None
        self._master_factory: Optional['MasterFactory'] = None
        self._tool_registry: Optional['ToolRegistry'] = None
        self._checkpointer: Optional['Checkpointer'] = None
        self._callback_manager: Optional['CallbackManager'] = None
        self._strategy_manager: Optional['StrategyManager'] = None
        self._execution_tracer: Optional['ExecutionTracer'] = None
        
        # 会话管理
        self.current_session_id: Optional[str] = None
        self._active_sessions: Dict[str, str] = {}
        self._conversation_contexts: Dict[str, ConversationContext] = {}
        
        # 任务队列
        self._task_queue = TaskQueue(max_size=self.config.task_queue_size)
        self._task_executor_running = False
        self._task_executor_task: Optional[asyncio.Task] = None
        
        # 响应缓存
        self._response_cache: Optional[ResponseCache] = None
        if self.config.enable_response_cache:
            self._response_cache = ResponseCache(default_ttl=self.config.cache_ttl)
        
        # 事件管理
        self._event_manager: Optional[AssistantEventManager] = None
        if self.config.enable_event_notifications:
            self._event_manager = AssistantEventManager()
            self._event_manager.add_listener(LoggingEventListener())
        
        # 指标收集
        self._metrics = ExecutionMetrics()
        
        # 意图检测器
        self._intent_detector = IntentDetector()
        
        # 参数优化器
        self._param_optimizer = TrainingParamOptimizer()
        
        # 初始化 LangGraph 组件
        self._initialize_langgraph()
        
        logger.info("Training assistant agent initialized: %s", self.agent_id)
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return """你是一个智能训练助手，专门帮助用户管理和优化机器学习模型训练任务。

你的核心能力包括：
1.创建和管理训练会话
2.监控训练进度和性能指标
3.下载训练完成的模型
4.查看训练历史和统计信息
5.提供智能训练优化建议
6.自然语言交互和问答
7.使用各种工具完成任务
8.批量处理训练任务
9.自动优化训练参数

你可以理解用户的自然语言请求，并提供智能化的回答和建议。
在需要执行具体操作时，你会使用相应的工具来完成任务。
请始终保持专业、友好和有帮助的态度。"""
    
    def _initialize_langgraph(self):
        """初始化 LangGraph 组件"""
        try:
            self._master_factory = get_master_factory()
            self._tool_registry = get_global_registry()
            self._register_training_tools()
            
            if self.config.enable_checkpointing:
                self._checkpointer = create_memory_checkpointer()
            
            if self.config.enable_callbacks:
                self._callback_manager = CallbackManager([LoggingCallback(), MetricsCallback()])
            
            if self.config.enable_strategies:
                self._strategy_manager = get_strategy_manager()
            
            self._execution_tracer = ExecutionTracer()
            self._create_langgraph_agent()
            
            logger.info("LangGraph components initialized successfully")
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.error("Failed to initialize LangGraph: %s", str(e))
    
    def _register_training_tools(self):
        """注册训练相关工具"""
        if not self._tool_registry:
            return
        
        # 创建训练会话工具
        @tool(name="create_training_session", description="创建新的训练会话", category="training")
        async def create_training_session_tool(user_id: str, name: str,
                                              config: Dict[str, Any] = None) -> Dict[str, Any]:
            return await self._create_training_session_impl(user_id, name, config or {})
        
        # 获取训练进度工具
        @tool(name="get_training_progress", description="获取训练进度", category="training")
        async def get_training_progress_tool(session_id: str, user_id: str = "") -> Dict[str, Any]:
            return await self._get_training_progress_impl(session_id, user_id)
        
        # 下载模型工具
        @tool(name="download_trained_model", description="下载训练完成的模型", category="training")
        async def download_model_tool(session_id: str, user_id: str) -> Dict[str, Any]:
            return await self._download_model_impl(session_id, user_id)
        
        # 获取训练历史工具
        @tool(name="get_training_history", description="获取训练历史", category="training")
        async def get_training_history_tool(user_id: str, limit: int = 10) -> Dict[str, Any]:
            return await self._get_training_history_impl(user_id, limit)
        
        # 获取训练统计工具
        @tool(name="get_training_statistics", description="获取训练统计信息", category="training")
        async def get_training_statistics_tool(user_id: str) -> Dict[str, Any]:
            return await self._get_training_statistics_impl(user_id)
        
        # 停止训练工具
        @tool(name="stop_training", description="停止正在进行的训练", category="training")
        async def stop_training_tool(session_id: str) -> Dict[str, Any]:
            return await self._stop_training_impl(session_id)
        
        # 获取优化参数建议工具
        @tool(name="get_optimization_suggestions", description="获取训练参数优化建议", category="training")
        async def get_optimization_tool(dataset_size: int = None, model_type: str = None,
                                       training_goal: str = None) -> Dict[str, Any]:
            return self._get_optimization_suggestions_impl(dataset_size, model_type, training_goal)
        
        # 批量创建训练任务工具
        @tool(name="batch_create_training", description="批量创建训练任务", category="training")
        async def batch_create_tool(user_id: str, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
            return await self._batch_create_training_impl(user_id, tasks)
        
        tools = [
            create_training_session_tool, get_training_progress_tool, download_model_tool,
            get_training_history_tool, get_training_statistics_tool, stop_training_tool,
            get_optimization_tool, batch_create_tool
        ]
        
        for t in tools:
            if not self._tool_registry.get(t.name):
                self._tool_registry.register(t)
        
        logger.info("Registered %d training tools", len(tools))
    
    def _create_langgraph_agent(self):
        """创建 LangGraph Agent - 使用增强的工厂方法"""
        if not self._master_factory:
            return
        
        try:
            tools = self._get_tools_for_agent()
            
            # 使用增强的 Agent 构建器
            builder = build_enhanced_agent(self.config.agent_type.value)
            
            builder = (builder
                .with_name(self.config.name)
                .with_tools(tools)
                .with_model(self.config.model)
                .with_builtin_tools()  # 添加内置工具
                .with_metrics()        # 启用指标收集
                .with_logging())       # 启用日志
            
            # 配置执行上下文
            builder.with_execution_context()

            # 配置熔断器和限流器
            if self.config.max_retries > 0:
                builder.with_circuit_breaker(
                    threshold=self.config.max_retries,
                    recovery_timeout=30.0
                )

            builder.with_rate_limiter(
                rpm=100,
                burst=10
            )
            
            # 配置检查点器
            if self._checkpointer:
                builder._checkpointer = self._checkpointer
            elif self.config.enable_checkpointing:
                builder.with_memory_checkpointer()
            
            # 针对特定 Agent 类型的配置（通过 config 设置）
            if self.config.agent_type == LangGraphAgentType.PLAN_EXECUTE:
                builder._config.enable_replanning = True
                builder._config.max_replans = 3
            elif self.config.agent_type == LangGraphAgentType.REFLEXION:
                builder._config.max_reflections = 3
                builder._config.quality_threshold = 0.8
            
            self._langgraph_agent = builder.build()
            
            # 配置额外的内置工具
            if hasattr(self._langgraph_agent, 'add_recommended_builtin_tools'):
                self._langgraph_agent.add_recommended_builtin_tools()
            
            logger.info(f"Created enhanced LangGraph agent: {self.config.agent_type.value}")
        except Exception as e:
            logger.error(f"Failed to create LangGraph agent: {str(e)}")
    
    def _get_tools_for_agent(self) -> List['Tool']:
        """获取 Agent 使用的工具"""
        if not self._tool_registry:
            return []

        # ToolRegistry 使用迭代器接口
        tools = list(self._tool_registry)
        
        try:
            builtin_tools = get_tools_for_agent("training_assistant")
            for t in builtin_tools:
                if t not in tools:
                    tools.append(t)
        except (RuntimeError, ImportError, AttributeError) as e:
            logger.warning("Failed to get builtin tools: %s", str(e))
        
        return tools
    
    # ==================== 状态管理 ====================
    
    def update_status(self, status: AssistantStatus):
        """更新智能体状态"""
        old_status = self.status
        self.status = status
        logger.debug("Status updated: %s -> %s", old_status.value, status.value)
    
    async def _emit_event(self, event_type: EventType, data: Dict[str, Any] = None,
                         user_id: str = None, session_id: str = None):
        """发送事件"""
        if self._event_manager:
            event = AssistantEvent(
                event_id=str(uuid.uuid4()), event_type=event_type,
                timestamp=datetime.now(), data=data or {},
                user_id=user_id, session_id=session_id)
            await self._event_manager.emit(event)
    
    # ==================== 主处理方法 ====================
    
    async def process(self, input_data: Dict[str, Any],
                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理训练助手请求"""
        self.update_status(AssistantStatus.PROCESSING)
        start_time = time.time()
        
        try:
            user_request = input_data.get("user_request", "")
            user_id = input_data.get("user_id", "")
            session_id = input_data.get("session_id")
            use_inference = input_data.get("use_inference", self.config.enable_inference)
            use_langgraph = input_data.get("use_langgraph", True)
            conversation_mode = input_data.get("mode", ConversationMode.NORMAL.value)
            
            if not user_request:
                return self._error_response("用户请求不能为空")
            
            # 检查缓存
            if self._response_cache and not input_data.get("skip_cache", False):
                cached = self._response_cache.get(user_request, user_id)
                if cached:
                    self._metrics.cache_hits += 1
                    return {"status": "success", "result": {"response": cached, "from_cache": True},
                           "agent_id": self.agent_id}
                self._metrics.cache_misses += 1
            
            # 选择处理方式
            if use_langgraph and self._langgraph_agent:
                result = await self._process_with_langgraph(user_request, user_id, session_id, context or {})
            elif use_inference:
                result = await self._handle_intelligent_conversation(
                    user_request, user_id, session_id, context or {})
            else:
                result = await self._handle_training_request(user_request, user_id, context or {})
            
            self.update_status(AssistantStatus.IDLE)
            
            # 计算执行时间
            execution_time = (time.time() - start_time) * 1000
            self._metrics.record_request(True, execution_time)
            
            # 缓存响应
            response_text = result.get("response", "")
            if self._response_cache and response_text and not input_data.get("skip_cache", False):
                self._response_cache.set(user_request, response_text, user_id)
            
            return {
                "status": "success",
                "result": result,
                "agent_id": self.agent_id,
                "confidence": 0.95,
                "context_used": context or {},
                "session_id": self.current_session_id,
                "execution_time_ms": execution_time,
                "langgraph_used": use_langgraph and self._langgraph_agent is not None
            }
            
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError) as e:
            logger.error("Processing failed: %s", str(e))
            self.update_status(AssistantStatus.ERROR)
            execution_time = (time.time() - start_time) * 1000
            self._metrics.record_request(False, execution_time)
            await self._emit_event(EventType.ERROR_OCCURRED, {"error": str(e)})
            return self._error_response(str(e))
    
    async def _process_with_langgraph(self, user_request: str, user_id: str,
                                     session_id: Optional[str],
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """使用 LangGraph Agent 处理请求"""
        if not self._langgraph_agent:
            return await self._handle_intelligent_conversation(user_request, user_id, session_id, context)
        
        try:
            if not session_id:
                session_id = await self._get_or_create_session(user_id)
            self.current_session_id = session_id
            
            enhanced_context = await self._prepare_context(user_request, user_id, context)
            
            agent_input = {
                "input": user_request, "user_id": user_id,
                "session_id": session_id, "context": enhanced_context
            }
            
            result = await self._langgraph_agent.ainvoke(agent_input)
            response = self._extract_response(result)
            
            if self._checkpointer and self.config.enable_checkpointing:
                try:
                    # 创建 AgentState 对象用于检查点保存
                    checkpoint_state = AgentState(thread_id=session_id, output=response)
                    self._checkpointer.save(checkpoint_state)
                except (RuntimeError, IOError, AttributeError) as e:
                    logger.warning("Failed to save checkpoint: %s", str(e))
            
            await self._emit_event(EventType.TASK_COMPLETED, 
                                  {"response_length": len(response)}, user_id, session_id)
            
            return {
                "action": "langgraph_processing", "response": response,
                "session_id": session_id, "context_used": enhanced_context,
                "agent_type": self.config.agent_type.value
            }
        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            logger.error("LangGraph processing failed: %s", str(e))
            return await self._handle_intelligent_conversation(user_request, user_id, session_id, context)
    
    def _extract_response(self, result: Any) -> str:
        """从 Agent 结果中提取响应"""
        if hasattr(result, 'final_answer') and result.final_answer:
            return result.final_answer
        elif hasattr(result, 'output'):
            return result.output
        elif isinstance(result, dict):
            return result.get("output", result.get("response", result.get("final_answer", str(result))))
        elif isinstance(result, str):
            return result
        else:
            return str(result)
    
    async def _get_or_create_session(self, user_id: str) -> str:
        """获取或创建会话"""
        if user_id in self._active_sessions:
            return self._active_sessions[user_id]
        
        session_id = await self.inference_service.create_session(
            user_id=user_id, agent_id=self.agent_id,
            memory_type="summary", max_token_limit=2000,
            agent_type=self.config.agent_type)
        
        self._active_sessions[user_id] = session_id
        self._conversation_contexts[session_id] = ConversationContext(
            session_id=session_id, user_id=user_id)
        
        await self._emit_event(EventType.AGENT_STARTED, {"session_id": session_id}, user_id, session_id)
        return session_id
    
    async def _prepare_context(self, user_request: str, user_id: str,
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """准备增强的上下文信息"""
        enhanced_context = context.copy()
        
        try:
            training_stats = self.training_history_service.get_training_statistics(user_id)
            if training_stats:
                enhanced_context["user_training_stats"] = training_stats
            
            recent_sessions = self.training_service.list_training_sessions(user_id, limit=3)
            if recent_sessions:
                enhanced_context["recent_training_sessions"] = [
                    {"id": s.id, "name": s.name, "status": s.status,
                     "created_at": s.created_at.isoformat() if s.created_at else None}
                    for s in recent_sessions]
            
            available_models = self.local_model_service.list_models()
            enhanced_context["available_models"] = available_models
            
            if self._tool_registry:
                # ToolRegistry 使用迭代器接口
                enhanced_context["available_tools"] = [t.name for t in self._tool_registry]
            
            # 检测意图
            intent = IntentDetector.detect(user_request)
            if intent:
                enhanced_context["detected_intent"] = intent
                enhanced_context["intent_confidence"] = IntentDetector.get_confidence(user_request, intent)
            
            enhanced_context["system_info"] = {
                "agent_id": self.agent_id, "agent_name": self.name,
                "agent_type": self.config.agent_type.value,
                "capabilities": self.capabilities,
                "timestamp": datetime.now().isoformat(),
                "langgraph_enabled": self._langgraph_agent is not None
            }
        except (RuntimeError, ValueError, KeyError, AttributeError) as e:
            logger.warning("Failed to prepare context: %s", str(e))
        
        return enhanced_context
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """生成错误响应"""
        return {"status": "error", "error": error_message, "agent_id": self.agent_id}
    
    # ==================== 智能对话 ====================
    
    async def _handle_intelligent_conversation(self, user_request: str, user_id: str,
                                             session_id: Optional[str],
                                             context: Dict[str, Any],
                                             provider: str = "local") -> Dict[str, Any]:
        """处理智能对话"""
        try:
            if provider not in self.available_providers:
                provider = self.default_provider
            
            if not session_id:
                session_id = await self._get_or_create_session(user_id)
            self.current_session_id = session_id
            
            # 更新对话上下文
            conv_ctx = self._conversation_contexts.get(session_id)
            if conv_ctx:
                conv_ctx.turn_count += 1
                intent = IntentDetector.detect(user_request)
                if intent:
                    conv_ctx.last_intent = intent
            
            enhanced_context = await self._prepare_context(user_request, user_id, context)
            enhanced_context["provider"] = provider
            
            # 检测并执行训练操作
            action_result = await self._detect_and_execute_training_actions(user_request, user_id, context)
            if action_result:
                enhanced_context["action_result"] = action_result
            
            response = await self.inference_service.chat(
                session_id=session_id, message=user_request,
                context=enhanced_context, model_name=None)
            
            if not response:
                return await self._handle_training_request(user_request, user_id, context)
            
            return {
                "action": "intelligent_conversation", "response": response,
                "session_id": session_id, "context_used": enhanced_context,
                "action_executed": action_result is not None, "provider": provider
            }
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error("Intelligent conversation failed: %s", str(e))
            return await self._handle_training_request(user_request, user_id, context)
    
    async def _detect_and_execute_training_actions(self, user_request: str, user_id: str,
                                                 context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """检测并执行训练相关操作"""
        intent = IntentDetector.detect(user_request)
        
        if not intent:
            return None
        
        try:
            intent_handlers = {
                "create_training": self._create_training_session,
                "check_progress": self._get_training_progress,
                "download_model": self._download_trained_model,
                "view_history": self._get_training_history,
                "view_statistics": self._get_training_statistics,
                "stop_training": self._stop_training,
                "optimize_params": self._get_optimization_suggestions,
                "help": self._provide_general_assistance,
            }
            
            handler = intent_handlers.get(intent)
            if handler:
                return await handler(user_request, user_id, context)
        except (RuntimeError, ValueError, KeyError) as e:
            logger.error("Failed to execute action for intent %s: %s", intent, str(e))
        
        return None
    
    # ==================== 训练操作实现 ====================
    
    async def _create_training_session_impl(self, user_id: str, name: str,
                                           config: Dict[str, Any]) -> Dict[str, Any]:
        """创建训练会话实现"""
        try:
            # 优化参数
            if self.config.enable_auto_optimization:
                optimized_config = TrainingParamOptimizer.get_optimized_params(
                    dataset_size=config.get("dataset_size"),
                    model_type=config.get("model_type"),
                    training_goal=config.get("training_goal"),
                    user_params=config
                )
                config.update(optimized_config)
            
            session = self.training_service.create_training_session(
                user_id=user_id, name=name,
                description="通过训练助手创建的训练任务",
                config=config)
            
            await self._emit_event(EventType.TRAINING_STARTED, 
                                  {"session_id": session.id, "config": config}, user_id)
            
            return {
                "action": "create_training_session",
                "session_id": session.id,
                "message": f"训练会话已创建，ID: {session.id}",
                "session_info": session.to_dict(),
                "optimized_params": config if self.config.enable_auto_optimization else None
            }
        except Exception as e:
            await self._emit_event(EventType.TRAINING_FAILED, {"error": str(e)}, user_id)
            return {"action": "create_training_session", "error": f"创建训练会话失败: {str(e)}"}
    
    async def _get_training_progress_impl(self, session_id: str, user_id: str = "") -> Dict[str, Any]:
        """获取训练进度实现"""
        try:
            from backend.services.training_progress_service import get_training_progress_service
            progress_service = get_training_progress_service()
            progress = progress_service.get_progress(session_id, user_id=user_id)
            
            if progress:
                await self._emit_event(EventType.TRAINING_PROGRESS,
                                      {"session_id": session_id, "progress": progress.get("progress_percentage", 0)},
                                      user_id, session_id)
                return {
                    "action": "get_training_progress",
                    "session_id": session_id,
                    "progress": {
                        "status": progress.get("status"),
                        "progress": progress.get("progress_percentage", 0),
                        "current_epoch": progress.get("current_epoch", 0),
                        "current_step": progress.get("current_step", 0),
                        "total_steps": progress.get("total_steps", 0),
                        "train_loss": progress.get("loss", 0.0),
                        "eval_loss": progress.get("loss", 0.0),
                    }
                }
            return {"action": "get_training_progress", "error": "训练进度不可用"}
        except Exception as e:
            return {"action": "get_training_progress", "error": f"获取训练进度失败: {str(e)}"}
    
    async def _download_model_impl(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """下载模型实现"""
        try:
            download_url = self.training_history_service.download_training_model(session_id, user_id)
            if download_url:
                await self._emit_event(EventType.MODEL_DOWNLOADED,
                                      {"session_id": session_id}, user_id)
                return {
                    "action": "download_trained_model",
                    "session_id": session_id,
                    "download_url": download_url,
                    "message": "模型下载链接已生成"
                }
            return {"action": "download_trained_model", "error": "无法生成下载链接"}
        except Exception as e:
            return {"action": "download_trained_model", "error": f"生成下载链接失败: {str(e)}"}
    
    async def _get_training_history_impl(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        """获取训练历史实现"""
        try:
            history = self.training_history_service.get_training_history(
                user_id=user_id, page=1, limit=limit)
            return {"action": "get_training_history", "history": history}
        except Exception as e:
            return {"action": "get_training_history", "error": f"获取训练历史失败: {str(e)}"}
    
    async def _get_training_statistics_impl(self, user_id: str) -> Dict[str, Any]:
        """获取训练统计实现"""
        try:
            statistics = self.training_history_service.get_training_statistics(user_id)
            return {"action": "get_training_statistics", "statistics": statistics}
        except Exception as e:
            return {"action": "get_training_statistics", "error": f"获取训练统计信息失败: {str(e)}"}
    
    async def _stop_training_impl(self, session_id: str) -> Dict[str, Any]:
        """停止训练实现"""
        try:
            # 使用 fail_training_session 来停止训练（标记为用户取消）
            self.training_service.fail_training_session(session_id, "用户手动停止训练")
            return {"action": "stop_training", "session_id": session_id, "message": "训练已停止"}
        except (ValueError, RuntimeError, KeyError) as e:
            return {"action": "stop_training", "error": f"停止训练失败: {str(e)}"}
    
    def _get_optimization_suggestions_impl(self, dataset_size: int = None,
                                          model_type: str = None,
                                          training_goal: str = None) -> Dict[str, Any]:
        """获取优化建议实现"""
        params = TrainingParamOptimizer.get_optimized_params(
            dataset_size=dataset_size, model_type=model_type, training_goal=training_goal)
        recommendations = TrainingParamOptimizer.get_recommendations(params)
        
        return {
            "action": "get_optimization_suggestions",
            "optimized_params": params,
            "recommendations": recommendations
        }
    
    async def _batch_create_training_impl(self, user_id: str,
                                         tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量创建训练任务实现"""
        if not self.config.enable_batch_processing:
            return {"action": "batch_create_training", "error": "批量处理未启用"}
        
        created_tasks = []
        failed_tasks = []
        
        for task_config in tasks:
            task = TrainingTask(
                task_id=str(uuid.uuid4()),
                user_id=user_id,
                task_type="training",
                parameters=task_config,
                priority=TaskPriority(task_config.get("priority", TaskPriority.NORMAL.value)))
            
            if self._task_queue.add_task(task):
                created_tasks.append(task.task_id)
                await self._emit_event(EventType.TASK_CREATED, {"task_id": task.task_id}, user_id)
            else:
                failed_tasks.append({"config": task_config, "error": "队列已满"})
        
        return {
            "action": "batch_create_training",
            "created_tasks": created_tasks,
            "failed_tasks": failed_tasks,
            "queue_size": self._task_queue.get_pending_count()
        }
    
    # ==================== 传统处理方法 ====================
    
    async def _handle_training_request(self, user_request: str, user_id: str,
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        """处理训练相关请求（传统方式）"""
        intent = IntentDetector.detect(user_request)
        
        handlers = {
            "create_training": self._create_training_session,
            "check_progress": self._get_training_progress,
            "download_model": self._download_trained_model,
            "view_history": self._get_training_history,
            "view_statistics": self._get_training_statistics,
            "stop_training": self._stop_training,
            "optimize_params": self._get_optimization_suggestions,
        }
        
        handler = handlers.get(intent)
        if handler:
            return await handler(user_request, user_id, context)
        return await self._provide_general_assistance(user_request, user_id, context)
    
    async def _create_training_session(self, user_request: str, user_id: str,
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        name = f"训练任务 {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return await self._create_training_session_impl(user_id, name, {})
    
    async def _get_training_progress(self, user_request: str, user_id: str,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        session_id = context.get("session_id") or self._extract_session_id(user_request)
        if not session_id:
            sessions = self.training_service.list_training_sessions(user_id, limit=1)
            if sessions:
                session_id = sessions[0].id
            else:
                return {"action": "get_training_progress", "message": "没有找到训练会话"}
        return await self._get_training_progress_impl(session_id, user_id)
    
    async def _download_trained_model(self, user_request: str, user_id: str,
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        session_id = context.get("session_id") or self._extract_session_id(user_request)
        if not session_id:
            return {"action": "download_trained_model", "error": "请提供训练会话ID"}
        return await self._download_model_impl(session_id, user_id)
    
    async def _get_training_history(self, user_request: str, user_id: str,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        limit = context.get("limit", 10)
        return await self._get_training_history_impl(user_id, limit)
    
    async def _get_training_statistics(self, user_request: str, user_id: str,
                                       context: Dict[str, Any]) -> Dict[str, Any]:
        return await self._get_training_statistics_impl(user_id)
    
    async def _stop_training(self, user_request: str, user_id: str,
                            context: Dict[str, Any]) -> Dict[str, Any]:
        session_id = context.get("session_id") or self._extract_session_id(user_request)
        if not session_id:
            return {"action": "stop_training", "error": "请提供训练会话ID"}
        return await self._stop_training_impl(session_id)
    
    async def _get_optimization_suggestions(self, user_request: str, user_id: str,
                                            context: Dict[str, Any]) -> Dict[str, Any]:
        return self._get_optimization_suggestions_impl(
            dataset_size=context.get("dataset_size"),
            model_type=context.get("model_type"),
            training_goal=context.get("training_goal"))
    
    async def _provide_general_assistance(self, user_request: str, user_id: str,
                                         context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action": "general_assistance",
            "message": """我是训练助手，可以帮助您管理模型训练任务。您可以询问以下内容：
1. 创建训练会话 - "创建一个新的训练任务"
2. 查看训练进度 - "查看训练进度"
3. 下载训练完成的模型 - "下载模型"
4. 查看训练历史 - "查看训练历史"
5. 获取训练统计信息 - "显示训练统计"
6. 停止训练 - "停止训练"
7. 获取参数优化建议 - "优化训练参数"
8. 批量创建训练任务 - "批量训练"

请告诉我您需要什么帮助？"""
        }
    
    def _extract_session_id(self, user_request: str) -> Optional[str]:
        """从用户请求中提取会话ID"""
        import re
        match = re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', user_request)
        return match.group(0) if match else None
    
    # ==================== 会话管理 ====================
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话历史"""
        try:
            if self.inference_service:
                session = await self.inference_service.get_session(session_id)
                if session:
                    return session.get_history()
            return []
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("Failed to get session history: %s", str(e))
            return []
    
    async def clear_session(self, session_id: str) -> bool:
        """清除会话"""
        try:
            if self.inference_service:
                await self.inference_service.delete_session(session_id)
                if self.current_session_id == session_id:
                    self.current_session_id = None
                
                for user_id, sid in list(self._active_sessions.items()):
                    if sid == session_id:
                        del self._active_sessions[user_id]
                
                self._conversation_contexts.pop(session_id, None)
                await self._emit_event(EventType.AGENT_STOPPED, {"session_id": session_id})
                return True
            return False
        except (RuntimeError, ValueError, KeyError) as e:
            logger.error("Failed to clear session: %s", str(e))
            return False
    
    async def update_session_context(self, session_id: str, context: Dict[str, Any]) -> bool:
        """更新会话上下文"""
        try:
            if self.inference_service:
                session = await self.inference_service.get_session(session_id)
                if session:
                    session.update_context(context)
                    return True
            return False
        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("Failed to update session context: %s", str(e))
            return False
    
    def get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """获取对话上下文"""
        return self._conversation_contexts.get(session_id)
    
    def set_conversation_mode(self, session_id: str, mode: ConversationMode) -> bool:
        """设置对话模式"""
        ctx = self._conversation_contexts.get(session_id)
        if ctx:
            ctx.mode = mode
            return True
        return False
    
    # ==================== 任务队列处理 ====================
    
    async def start_task_executor(self):
        """启动任务执行器"""
        if self._task_executor_running:
            return
        
        self._task_executor_running = True
        self._task_executor_task = asyncio.create_task(self._task_executor_loop())
        logger.info("Task executor started")
    
    async def stop_task_executor(self):
        """停止任务执行器"""
        self._task_executor_running = False
        if self._task_executor_task:
            self._task_executor_task.cancel()
            try:
                await self._task_executor_task
            except asyncio.CancelledError:
                pass
        logger.info("Task executor stopped")
    
    async def _task_executor_loop(self):
        """任务执行循环"""
        while self._task_executor_running:
            try:
                task = self._task_queue.get_next_task()
                if task:
                    await self._execute_task(task)
                else:
                    await asyncio.sleep(1)
            except (RuntimeError, ValueError, asyncio.CancelledError) as e:
                logger.error("Task executor error: %s", str(e))
                await asyncio.sleep(5)
    
    async def _execute_task(self, task: TrainingTask):
        """执行单个任务"""
        try:
            await self._emit_event(EventType.TASK_STARTED, {"task_id": task.task_id}, task.user_id)
            
            if task.task_type == "training":
                result = await self._create_training_session_impl(
                    task.user_id,
                    task.parameters.get("name", f"批量任务_{task.task_id[:8]}"),
                    task.parameters)
                self._task_queue.complete_task(task.task_id, result=result)
            else:
                self._task_queue.complete_task(task.task_id, error=f"未知任务类型: {task.task_type}")
            
            await self._emit_event(EventType.TASK_COMPLETED, {"task_id": task.task_id}, task.user_id)
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            logger.error("Task execution failed: %s - %s", task.task_id, str(e))
            
            task.retry_count += 1
            if task.retry_count < self.config.max_retries:
                task.status = TaskStatus.PENDING
                self._task_queue._pending_queue.append(task.task_id)
                await asyncio.sleep(self.config.retry_delay)
            else:
                self._task_queue.complete_task(task.task_id, error=str(e))
                await self._emit_event(EventType.TASK_FAILED, 
                                      {"task_id": task.task_id, "error": str(e)}, task.user_id)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self._task_queue.get_task(task_id)
        return task.to_dict() if task else None
    
    def get_user_tasks(self, user_id: str, status: TaskStatus = None) -> List[Dict[str, Any]]:
        """获取用户的所有任务"""
        tasks = self._task_queue.get_all_tasks(user_id=user_id, status=status)
        return [t.to_dict() for t in tasks]
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        return self._task_queue.cancel_task(task_id)
    
    # ==================== 推理配置 ====================
    
    def enable_inference(self, enabled: bool = True):
        """启用或禁用推理功能"""
        self.config.enable_inference = enabled
        logger.info(f"Inference {'enabled' if enabled else 'disabled'}")
    
    def is_inference_enabled(self) -> bool:
        """检查推理功能是否启用"""
        return self.config.enable_inference and self.inference_service is not None
    
    def get_available_providers(self) -> List[str]:
        """获取可用的推理提供者列表"""
        return self.available_providers
    
    def set_default_provider(self, provider: str) -> bool:
        """设置默认推理提供者"""
        if provider in self.available_providers:
            self.default_provider = provider
            logger.info(f"Default provider set to: {provider}")
            return True
        return False
    
    def get_default_provider(self) -> str:
        """获取当前默认推理提供者"""
        return self.default_provider
    
    async def get_provider_status(self, provider: str = None) -> Dict[str, Any]:
        """获取推理提供者状态"""
        if self.inference_service:
            try:
                return await self.inference_service.get_provider_status(provider)
            except (RuntimeError, ValueError, AttributeError) as e:
                logger.error("Failed to get provider status: %s", str(e))
        
        if provider:
            return {"provider": provider, "available": provider in self.available_providers, "status": "unknown"}
        return {
            "providers": {p: {"available": True, "status": "unknown"} for p in self.available_providers},
            "default_provider": self.default_provider
        }
    
    # ==================== 流式对话 ====================
    
    async def stream_chat(self, session_id: str, message: str, user_id: str,
                         provider: str = None) -> AsyncIterator[Dict[str, Any]]:
        """流式对话支持"""
        try:
            selected_provider = provider or self.default_provider
            
            if self.inference_service:
                async for chunk in self.inference_service.stream_chat(
                    session_id=session_id, message=message, provider=selected_provider):
                    yield {"chunk": chunk, "session_id": session_id, "provider": selected_provider}
            else:
                yield {"error": "推理服务未启用", "session_id": session_id}
        except (RuntimeError, ValueError, ConnectionError) as e:
            logger.error("Stream chat failed: %s", str(e))
            yield {"error": str(e), "session_id": session_id}
    
    # ==================== 状态和指标 ====================
    
    def get_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """获取训练助手提示词"""
        return self.system_prompt
    
    def get_agent_status(self) -> Dict[str, Any]:
        """获取 Agent 状态 - 使用增强的工厂诊断方法"""
        status = {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "agent_type": self.config.agent_type.value,
            "langgraph_enabled": self._langgraph_agent is not None,
            "inference_enabled": self.config.enable_inference,
            "tools_enabled": self.config.enable_tools,
            "memory_enabled": self.config.enable_memory,
            "checkpointing_enabled": self.config.enable_checkpointing,
            "batch_processing_enabled": self.config.enable_batch_processing,
            "current_session_id": self.current_session_id,
            "active_sessions": len(self._active_sessions),
            "pending_tasks": self._task_queue.get_pending_count(),
            "available_providers": self.available_providers,
            "default_provider": self.default_provider,
            "metrics": self._metrics.get_summary(),
            "cache_stats": self._response_cache.get_stats() if self._response_cache else {},
            "config": self.config.to_dict()
        }
        
        # 使用工厂方法获取增强诊断信息
        if self._langgraph_agent:
            try:
                diagnostics = get_agent_diagnostics(self._langgraph_agent)
                status["langgraph_diagnostics"] = diagnostics
            except (RuntimeError, AttributeError) as e:
                status["langgraph_diagnostics_error"] = str(e)
        
        # 获取工厂健康状态
        try:
            health = factory_health_check()
            status["factory_health"] = health.get("status", "unknown")
        except (RuntimeError, AttributeError):
            status["factory_health"] = "unknown"
        
        return status
    
    def get_execution_trace(self) -> Optional[Dict[str, Any]]:
        """获取执行追踪"""
        if self._execution_tracer:
            traces = self._execution_tracer.get_all_traces()
            if traces:
                # 返回最新的追踪
                latest_trace = traces[-1]
                return latest_trace.to_dict() if hasattr(latest_trace, 'to_dict') else {}
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取执行指标 - 使用增强的工厂方法"""
        metrics = self._metrics.get_summary()
        
        # 获取 LangGraph Agent 的详细指标
        if self._langgraph_agent and self._master_factory:
            try:
                agent_factory = self._master_factory.agents
                agent_metrics = agent_factory.get_agent_metrics(self._langgraph_agent)
                metrics["langgraph_agent"] = agent_metrics
                
                # 获取工具指标
                tool_metrics = agent_factory.get_agent_tool_metrics(self._langgraph_agent)
                metrics["tools"] = tool_metrics
            except (RuntimeError, AttributeError) as e:
                metrics["langgraph_error"] = str(e)
        
        return metrics
    
    def reset_metrics(self):
        """重置指标"""
        self._metrics = ExecutionMetrics()
    
    def get_event_history(self, limit: int = 100, event_types: List[EventType] = None) -> List[Dict[str, Any]]:
        """获取事件历史"""
        if self._event_manager:
            events = self._event_manager.get_history(limit, event_types)
            return [e.to_dict() for e in events]
        return []
    
    # ==================== 事件监听 ====================
    
    def add_event_listener(self, listener: EventListener, event_types: List[EventType] = None):
        """添加事件监听器"""
        if self._event_manager:
            self._event_manager.add_listener(listener, event_types)
    
    def remove_event_listener(self, listener: EventListener):
        """移除事件监听器"""
        if self._event_manager:
            self._event_manager.remove_listener(listener)
    
    # ==================== Agent 类型切换 ====================
    
    def switch_agent_type(self, agent_type: LangGraphAgentType) -> bool:
        """切换 Agent 类型"""
        try:
            self.config.agent_type = agent_type
            self._create_langgraph_agent()
            logger.info(f"Agent type switched to: {agent_type.value}")
            return True
        except (RuntimeError, ValueError, TypeError) as e:
            logger.error("Failed to switch agent type: %s", str(e))
            return False
    
    def get_supported_agent_types(self) -> List[str]:
        """获取支持的 Agent 类型"""
        return [t.value for t in LangGraphAgentType]
    
    # ==================== 清理方法 ====================
    
    async def cleanup(self):
        """清理资源"""
        await self.stop_task_executor()
        self._task_queue.cleanup_completed()
        
        for session_id in list(self._active_sessions.values()):
            try:
                await self.clear_session(session_id)
            except (RuntimeError, ValueError) as e:
                logger.warning("Failed to clear session %s: %s", session_id, str(e))
        
        if self._response_cache:
            self._response_cache.clear()
        
        logger.info("Training assistant cleanup completed")

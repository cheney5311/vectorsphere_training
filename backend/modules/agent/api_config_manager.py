"""
API 配置管理模块

该模块提供对各种 AI API 服务的配置管理，包括 ChatGPT、DeepSeek 等。
支持安全的 token 存储、配置验证和动态配置更新。

扩展功能：
- LangGraph Agent 配置
- 检查点器配置
- 策略配置
- 回调配置
- 配置验证器
- 配置构建器
- 配置模板系统
- 配置版本管理
- 配置热重载
- 配置审计日志
"""

import copy
import hashlib
import json
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any, List, Union, Callable, TypeVar, Generic

from cryptography.fernet import Fernet
from loguru import logger

# 导入 LangGraph 相关类型
from backend.algo.langgraph.agents import AgentConfig as LangGraphAgentConfig
from backend.algo.langgraph.factory import (
    create_production_agent, create_enhanced_agent,
    get_agent_diagnostics, factory_health_check
)
from .agent_type import (
    AgentType, LangGraphAgentType, CollaborationMode,
    ExecutionMode, get_langgraph_type, get_default_config
)

# 类型变量
T = TypeVar('T')
ConfigType = TypeVar('ConfigType')


# ==================== 枚举定义 ====================

class APIProvider(Enum):
    """API 提供商枚举"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    OLLAMA = "ollama"
    AZURE = "azure"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"


class CheckpointerType(Enum):
    """检查点器类型"""
    MEMORY = "memory"
    REDIS = "redis"
    SQLITE = "sqlite"
    FILE = "file"
    POSTGRES = "postgres"
    MONGODB = "mongodb"


class StrategyType(Enum):
    """策略类型"""
    STATE = "state"
    NODE = "node"
    EDGE = "edge"
    GRAPH = "graph"
    TOOL = "tool"
    CHECKPOINT = "checkpoint"
    MEMORY = "memory"
    ROUTING = "routing"


class ConfigEventType(Enum):
    """配置事件类型"""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    VALIDATED = "validated"
    LOADED = "loaded"
    SAVED = "saved"
    ERROR = "error"


class ValidationLevel(Enum):
    """验证级别"""
    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    FULL = "full"


# ==================== 基础配置类 ====================

@dataclass
class BaseConfig(ABC):
    """配置基类
    
    所有配置类的抽象基类，提供通用的序列化和验证能力。
    """
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseConfig':
        """从字典创建实例"""
        pass
    
    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BaseConfig':
        """从 JSON 字符串创建实例"""
        return cls.from_dict(json.loads(json_str))
    
    def clone(self) -> 'BaseConfig':
        """深拷贝"""
        return self.__class__.from_dict(copy.deepcopy(self.to_dict()))
    
    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """合并配置"""
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        merged = {**self_dict, **{k: v for k, v in other_dict.items() if v is not None}}
        return self.__class__.from_dict(merged)
    
    def diff(self, other: 'BaseConfig') -> Dict[str, Any]:
        """比较差异"""
        self_dict = self.to_dict()
        other_dict = other.to_dict()
        diff = {}
        for key in set(self_dict.keys()) | set(other_dict.keys()):
            if self_dict.get(key) != other_dict.get(key):
                diff[key] = {"old": self_dict.get(key), "new": other_dict.get(key)}
        return diff
    
    def get_hash(self) -> str:
        """获取配置哈希值"""
        return hashlib.md5(self.to_json().encode()).hexdigest()


@dataclass
class APIConfig(BaseConfig):
    """API 配置数据类"""
    provider: APIProvider
    api_key: str
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30
    max_retries: int = 3
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None
    
    # 扩展字段
    rate_limit: Optional[int] = None  # 请求速率限制
    daily_quota: Optional[int] = None  # 每日配额
    priority: int = 0  # 优先级
    fallback_provider: Optional[str] = None  # 降级提供商
    headers: Optional[Dict[str, str]] = None  # 自定义请求头
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "provider": self.provider.value,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "enabled": self.enabled,
            "metadata": self.metadata or {},
            "rate_limit": self.rate_limit,
            "daily_quota": self.daily_quota,
            "priority": self.priority,
            "fallback_provider": self.fallback_provider,
            "headers": self.headers,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIConfig':
        """从字典创建实例"""
        return cls(
            provider=APIProvider(data["provider"]),
            api_key=data["api_key"],
            base_url=data.get("base_url"),
            model_name=data.get("model_name"),
            max_tokens=data.get("max_tokens", 4096),
            temperature=data.get("temperature", 0.7),
            timeout=data.get("timeout", 30),
            max_retries=data.get("max_retries", 3),
            enabled=data.get("enabled", True),
            metadata=data.get("metadata"),
            rate_limit=data.get("rate_limit"),
            daily_quota=data.get("daily_quota"),
            priority=data.get("priority", 0),
            fallback_provider=data.get("fallback_provider"),
            headers=data.get("headers"),
        )


@dataclass
class CheckpointerConfig(BaseConfig):
    """检查点器配置"""
    type: CheckpointerType = CheckpointerType.MEMORY
    enabled: bool = True
    
    # 通用配置
    max_checkpoints: int = 100
    compression: str = "gzip"  # none, gzip, lz4
    auto_cleanup: bool = True
    cleanup_interval: int = 3600
    
    # Redis 配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    redis_prefix: str = "checkpoint:"
    redis_ssl: bool = False
    redis_cluster: bool = False
    
    # SQLite 配置
    sqlite_db_path: str = "checkpoints.db"
    sqlite_journal_mode: str = "WAL"
    
    # 文件配置
    file_base_dir: str = "./checkpoints"
    file_format: str = "json"  # json, pickle, msgpack
    
    # Postgres 配置
    postgres_dsn: Optional[str] = None
    postgres_table: str = "checkpoints"
    
    # MongoDB 配置
    mongodb_uri: Optional[str] = None
    mongodb_db: str = "checkpoints"
    mongodb_collection: str = "states"
    
    # 高级配置
    enable_caching: bool = True
    cache_ttl: int = 3600
    cache_max_size: int = 1000
    enable_branching: bool = False
    enable_tagging: bool = True
    enable_versioning: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "enabled": self.enabled,
            "max_checkpoints": self.max_checkpoints,
            "compression": self.compression,
            "auto_cleanup": self.auto_cleanup,
            "cleanup_interval": self.cleanup_interval,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,
            "redis_prefix": self.redis_prefix,
            "redis_ssl": self.redis_ssl,
            "redis_cluster": self.redis_cluster,
            "sqlite_db_path": self.sqlite_db_path,
            "sqlite_journal_mode": self.sqlite_journal_mode,
            "file_base_dir": self.file_base_dir,
            "file_format": self.file_format,
            "postgres_table": self.postgres_table,
            "mongodb_db": self.mongodb_db,
            "mongodb_collection": self.mongodb_collection,
            "enable_caching": self.enable_caching,
            "cache_ttl": self.cache_ttl,
            "cache_max_size": self.cache_max_size,
            "enable_branching": self.enable_branching,
            "enable_tagging": self.enable_tagging,
            "enable_versioning": self.enable_versioning,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointerConfig':
        return cls(
            type=CheckpointerType(data.get("type", "memory")),
            enabled=data.get("enabled", True),
            max_checkpoints=data.get("max_checkpoints", 100),
            compression=data.get("compression", "gzip"),
            auto_cleanup=data.get("auto_cleanup", True),
            cleanup_interval=data.get("cleanup_interval", 3600),
            redis_host=data.get("redis_host", "localhost"),
            redis_port=data.get("redis_port", 6379),
            redis_password=data.get("redis_password"),
            redis_db=data.get("redis_db", 0),
            redis_prefix=data.get("redis_prefix", "checkpoint:"),
            redis_ssl=data.get("redis_ssl", False),
            redis_cluster=data.get("redis_cluster", False),
            sqlite_db_path=data.get("sqlite_db_path", "checkpoints.db"),
            sqlite_journal_mode=data.get("sqlite_journal_mode", "WAL"),
            file_base_dir=data.get("file_base_dir", "./checkpoints"),
            file_format=data.get("file_format", "json"),
            postgres_dsn=data.get("postgres_dsn"),
            postgres_table=data.get("postgres_table", "checkpoints"),
            mongodb_uri=data.get("mongodb_uri"),
            mongodb_db=data.get("mongodb_db", "checkpoints"),
            mongodb_collection=data.get("mongodb_collection", "states"),
            enable_caching=data.get("enable_caching", True),
            cache_ttl=data.get("cache_ttl", 3600),
            cache_max_size=data.get("cache_max_size", 1000),
            enable_branching=data.get("enable_branching", False),
            enable_tagging=data.get("enable_tagging", True),
            enable_versioning=data.get("enable_versioning", True),
        )


@dataclass
class StrategyConfig(BaseConfig):
    """策略配置"""
    enabled: bool = True
    
    # 各策略类型的配置
    state_strategy: Dict[str, Any] = field(default_factory=dict)
    node_strategy: Dict[str, Any] = field(default_factory=dict)
    edge_strategy: Dict[str, Any] = field(default_factory=dict)
    graph_strategy: Dict[str, Any] = field(default_factory=dict)
    tool_strategy: Dict[str, Any] = field(default_factory=dict)
    checkpoint_strategy: Dict[str, Any] = field(default_factory=dict)
    memory_strategy: Dict[str, Any] = field(default_factory=dict)
    routing_strategy: Dict[str, Any] = field(default_factory=dict)
    
    # 高级配置
    fallback_enabled: bool = True
    retry_strategy: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 3,
        "backoff_factor": 2.0,
        "max_delay": 60
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "state_strategy": self.state_strategy,
            "node_strategy": self.node_strategy,
            "edge_strategy": self.edge_strategy,
            "graph_strategy": self.graph_strategy,
            "tool_strategy": self.tool_strategy,
            "checkpoint_strategy": self.checkpoint_strategy,
            "memory_strategy": self.memory_strategy,
            "routing_strategy": self.routing_strategy,
            "fallback_enabled": self.fallback_enabled,
            "retry_strategy": self.retry_strategy,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyConfig':
        return cls(
            enabled=data.get("enabled", True),
            state_strategy=data.get("state_strategy", {}),
            node_strategy=data.get("node_strategy", {}),
            edge_strategy=data.get("edge_strategy", {}),
            graph_strategy=data.get("graph_strategy", {}),
            tool_strategy=data.get("tool_strategy", {}),
            checkpoint_strategy=data.get("checkpoint_strategy", {}),
            memory_strategy=data.get("memory_strategy", {}),
            routing_strategy=data.get("routing_strategy", {}),
            fallback_enabled=data.get("fallback_enabled", True),
            retry_strategy=data.get("retry_strategy", {
                "max_retries": 3,
                "backoff_factor": 2.0,
                "max_delay": 60
            }),
        )


@dataclass
class CallbackConfig(BaseConfig):
    """回调配置"""
    enabled: bool = True
    
    # 回调类型开关
    enable_logging: bool = True
    enable_metrics: bool = True
    enable_streaming: bool = False
    enable_webhook: bool = False
    enable_events: bool = True
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    log_rotation: str = "1 day"
    log_retention: str = "7 days"
    
    # 指标配置
    metrics_backend: str = "memory"  # memory, prometheus, statsd, datadog
    metrics_prefix: str = "agent"
    metrics_labels: Dict[str, str] = field(default_factory=dict)
    
    # Webhook 配置
    webhook_url: Optional[str] = None
    webhook_events: List[str] = field(default_factory=list)
    webhook_timeout: int = 10
    webhook_retry: int = 3
    webhook_headers: Dict[str, str] = field(default_factory=dict)
    
    # 事件配置
    event_queue_size: int = 1000
    event_batch_size: int = 100
    event_flush_interval: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "enable_logging": self.enable_logging,
            "enable_metrics": self.enable_metrics,
            "enable_streaming": self.enable_streaming,
            "enable_webhook": self.enable_webhook,
            "enable_events": self.enable_events,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "log_file": self.log_file,
            "log_rotation": self.log_rotation,
            "log_retention": self.log_retention,
            "metrics_backend": self.metrics_backend,
            "metrics_prefix": self.metrics_prefix,
            "metrics_labels": self.metrics_labels,
            "webhook_url": self.webhook_url,
            "webhook_events": self.webhook_events,
            "webhook_timeout": self.webhook_timeout,
            "webhook_retry": self.webhook_retry,
            "webhook_headers": self.webhook_headers,
            "event_queue_size": self.event_queue_size,
            "event_batch_size": self.event_batch_size,
            "event_flush_interval": self.event_flush_interval,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CallbackConfig':
        return cls(
            enabled=data.get("enabled", True),
            enable_logging=data.get("enable_logging", True),
            enable_metrics=data.get("enable_metrics", True),
            enable_streaming=data.get("enable_streaming", False),
            enable_webhook=data.get("enable_webhook", False),
            enable_events=data.get("enable_events", True),
            log_level=data.get("log_level", "INFO"),
            log_format=data.get("log_format", "json"),
            log_file=data.get("log_file"),
            log_rotation=data.get("log_rotation", "1 day"),
            log_retention=data.get("log_retention", "7 days"),
            metrics_backend=data.get("metrics_backend", "memory"),
            metrics_prefix=data.get("metrics_prefix", "agent"),
            metrics_labels=data.get("metrics_labels", {}),
            webhook_url=data.get("webhook_url"),
            webhook_events=data.get("webhook_events", []),
            webhook_timeout=data.get("webhook_timeout", 10),
            webhook_retry=data.get("webhook_retry", 3),
            webhook_headers=data.get("webhook_headers", {}),
            event_queue_size=data.get("event_queue_size", 1000),
            event_batch_size=data.get("event_batch_size", 100),
            event_flush_interval=data.get("event_flush_interval", 5),
        )


@dataclass
class AgentInstanceConfig(BaseConfig):
    """Agent 实例配置
    
    完整的 Agent 配置，包括所有子配置。
    """
    # 基本信息
    name: str
    agent_type: AgentType = AgentType.BASIC
    langgraph_type: LangGraphAgentType = LangGraphAgentType.REACT
    description: str = ""
    version: str = "1.0.0"
    
    # 执行配置
    max_iterations: int = 10
    timeout: float = 300.0
    execution_mode: ExecutionMode = ExecutionMode.ASYNC
    
    # LLM 配置
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: Optional[str] = None
    
    # 工具配置
    enabled_tools: List[str] = field(default_factory=list)
    auto_load_builtin_tools: bool = True
    tool_timeout: float = 30.0
    parallel_tool_calls: bool = True
    
    # 记忆配置
    enable_memory: bool = True
    memory_type: str = "short_term"  # short_term, long_term, episodic
    memory_size: int = 1000
    
    # 子配置
    checkpointer: CheckpointerConfig = field(default_factory=CheckpointerConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)
    
    # 多 Agent 配置
    collaboration_mode: CollaborationMode = CollaborationMode.SUPERVISOR
    max_workers: int = 3
    worker_timeout: float = 60.0
    
    # 高级配置
    enable_human_in_loop: bool = False
    retry_on_failure: bool = True
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    
    # 安全配置
    enable_input_validation: bool = True
    enable_output_sanitization: bool = True
    max_input_length: int = 10000
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # 自动设置 langgraph_type
        if self.langgraph_type == LangGraphAgentType.REACT:
            self.langgraph_type = get_langgraph_type(self.agent_type)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "agent_type": self.agent_type.value,
            "langgraph_type": self.langgraph_type.value,
            "description": self.description,
            "version": self.version,
            "max_iterations": self.max_iterations,
            "timeout": self.timeout,
            "execution_mode": self.execution_mode.value,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
            "enabled_tools": self.enabled_tools,
            "auto_load_builtin_tools": self.auto_load_builtin_tools,
            "tool_timeout": self.tool_timeout,
            "parallel_tool_calls": self.parallel_tool_calls,
            "enable_memory": self.enable_memory,
            "memory_type": self.memory_type,
            "memory_size": self.memory_size,
            "checkpointer": self.checkpointer.to_dict(),
            "strategies": self.strategies.to_dict(),
            "callbacks": self.callbacks.to_dict(),
            "collaboration_mode": self.collaboration_mode.value,
            "max_workers": self.max_workers,
            "worker_timeout": self.worker_timeout,
            "enable_human_in_loop": self.enable_human_in_loop,
            "retry_on_failure": self.retry_on_failure,
            "max_retries": self.max_retries,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_timeout": self.circuit_breaker_timeout,
            "enable_input_validation": self.enable_input_validation,
            "enable_output_sanitization": self.enable_output_sanitization,
            "max_input_length": self.max_input_length,
            "metadata": self.metadata,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentInstanceConfig':
        return cls(
            name=data["name"],
            agent_type=AgentType(data.get("agent_type", "basic")),
            langgraph_type=LangGraphAgentType(data.get("langgraph_type", "react")),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            max_iterations=data.get("max_iterations", 10),
            timeout=data.get("timeout", 300.0),
            execution_mode=ExecutionMode(data.get("execution_mode", "async")),
            model=data.get("model", "gpt-4"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 4096),
            system_prompt=data.get("system_prompt"),
            enabled_tools=data.get("enabled_tools", []),
            auto_load_builtin_tools=data.get("auto_load_builtin_tools", True),
            tool_timeout=data.get("tool_timeout", 30.0),
            parallel_tool_calls=data.get("parallel_tool_calls", True),
            enable_memory=data.get("enable_memory", True),
            memory_type=data.get("memory_type", "short_term"),
            memory_size=data.get("memory_size", 1000),
            checkpointer=CheckpointerConfig.from_dict(data.get("checkpointer", {})),
            strategies=StrategyConfig.from_dict(data.get("strategies", {})),
            callbacks=CallbackConfig.from_dict(data.get("callbacks", {})),
            collaboration_mode=CollaborationMode(data.get("collaboration_mode", "supervisor")),
            max_workers=data.get("max_workers", 3),
            worker_timeout=data.get("worker_timeout", 60.0),
            enable_human_in_loop=data.get("enable_human_in_loop", False),
            retry_on_failure=data.get("retry_on_failure", True),
            max_retries=data.get("max_retries", 3),
            circuit_breaker_threshold=data.get("circuit_breaker_threshold", 5),
            circuit_breaker_timeout=data.get("circuit_breaker_timeout", 60.0),
            enable_input_validation=data.get("enable_input_validation", True),
            enable_output_sanitization=data.get("enable_output_sanitization", True),
            max_input_length=data.get("max_input_length", 10000),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )
    
    def to_langgraph_config(self) -> Optional['LangGraphAgentConfig']:
        """转换为 LangGraph AgentConfig"""
        return LangGraphAgentConfig(
            name=self.name,
            max_iterations=self.max_iterations,
            timeout=self.timeout,
            model=self.model,
            temperature=self.temperature,
            system_prompt=self.system_prompt,
            enable_memory=self.enable_memory,
            enable_checkpointing=self.checkpointer.enabled,
            enable_streaming=self.callbacks.enable_streaming,
            max_retries=self.max_retries,
        )


# ==================== 配置验证器 ====================

class ConfigValidator(ABC):
    """配置验证器基类"""
    
    @abstractmethod
    def validate(self, config: BaseConfig) -> List[str]:
        """验证配置，返回错误列表"""
        pass
    
    def is_valid(self, config: BaseConfig) -> bool:
        """检查配置是否有效"""
        return len(self.validate(config)) == 0


class APIConfigValidator(ConfigValidator):
    """API 配置验证器"""
    
    def validate(self, config: APIConfig) -> List[str]:
        errors = []
        
        # 验证 API key
        if config.enabled and not config.api_key:
            if config.provider not in [APIProvider.LOCAL, APIProvider.OLLAMA]:
                errors.append(f"API key is required for provider {config.provider.value}")
        
        # 验证 base_url
        if config.base_url and not config.base_url.startswith(("http://", "https://")):
            errors.append(f"Invalid base_url format: {config.base_url}")
        
        # 验证温度
        if not 0.0 <= config.temperature <= 2.0:
            errors.append(f"Temperature must be between 0.0 and 2.0, got {config.temperature}")
        
        # 验证 max_tokens
        if config.max_tokens <= 0:
            errors.append(f"max_tokens must be positive, got {config.max_tokens}")
        
        # 验证超时
        if config.timeout <= 0:
            errors.append(f"timeout must be positive, got {config.timeout}")
        
        return errors


class CheckpointerConfigValidator(ConfigValidator):
    """检查点器配置验证器"""
    
    def validate(self, config: CheckpointerConfig) -> List[str]:
        errors = []
        
        # 验证 Redis 配置
        if config.type == CheckpointerType.REDIS:
            if config.redis_port <= 0 or config.redis_port > 65535:
                errors.append(f"Invalid redis_port: {config.redis_port}")
        
        # 验证 SQLite 配置
        if config.type == CheckpointerType.SQLITE:
            if not config.sqlite_db_path:
                errors.append("sqlite_db_path is required for SQLite checkpointer")
        
        # 验证文件配置
        if config.type == CheckpointerType.FILE:
            if not config.file_base_dir:
                errors.append("file_base_dir is required for file checkpointer")
        
        # 验证 Postgres 配置
        if config.type == CheckpointerType.POSTGRES:
            if not config.postgres_dsn:
                errors.append("postgres_dsn is required for Postgres checkpointer")
        
        # 验证通用配置
        if config.max_checkpoints <= 0:
            errors.append(f"max_checkpoints must be positive, got {config.max_checkpoints}")
        
        return errors


class AgentConfigValidator(ConfigValidator):
    """Agent 配置验证器"""
    
    def __init__(self):
        self.api_validator = APIConfigValidator()
        self.checkpointer_validator = CheckpointerConfigValidator()
    
    def validate(self, config: AgentInstanceConfig) -> List[str]:
        errors = []
        
        # 验证名称
        if not config.name or not config.name.strip():
            errors.append("Agent name is required")
        
        # 验证迭代次数
        if config.max_iterations <= 0:
            errors.append(f"max_iterations must be positive, got {config.max_iterations}")
        
        # 验证超时
        if config.timeout <= 0:
            errors.append(f"timeout must be positive, got {config.timeout}")
        
        # 验证温度
        if not 0.0 <= config.temperature <= 2.0:
            errors.append(f"Temperature must be between 0.0 and 2.0, got {config.temperature}")
        
        # 验证子配置
        errors.extend(self.checkpointer_validator.validate(config.checkpointer))
        
        return errors


class CompositeValidator(ConfigValidator):
    """组合验证器"""
    
    def __init__(self, validators: List[ConfigValidator] = None):
        self.validators = validators or []
    
    def add_validator(self, validator: ConfigValidator):
        self.validators.append(validator)
    
    def validate(self, config: BaseConfig) -> List[str]:
        errors = []
        for validator in self.validators:
            errors.extend(validator.validate(config))
        return errors


# ==================== 配置构建器 ====================

class ConfigBuilder(Generic[ConfigType], ABC):
    """配置构建器基类"""
    
    @abstractmethod
    def build(self) -> ConfigType:
        """构建配置"""
        pass
    
    @abstractmethod
    def reset(self) -> 'ConfigBuilder[ConfigType]':
        """重置构建器"""
        pass


class APIConfigBuilder(ConfigBuilder[APIConfig]):
    """API 配置构建器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'APIConfigBuilder':
        self._provider: APIProvider = APIProvider.OPENAI
        self._api_key: str = ""
        self._base_url: Optional[str] = None
        self._model_name: Optional[str] = None
        self._max_tokens: int = 4096
        self._temperature: float = 0.7
        self._timeout: int = 30
        self._max_retries: int = 3
        self._enabled: bool = True
        self._metadata: Dict[str, Any] = {}
        self._rate_limit: Optional[int] = None
        self._daily_quota: Optional[int] = None
        self._priority: int = 0
        self._fallback_provider: Optional[str] = None
        self._headers: Optional[Dict[str, str]] = None
        return self
    
    def with_provider(self, provider: Union[str, APIProvider]) -> 'APIConfigBuilder':
        self._provider = APIProvider(provider) if isinstance(provider, str) else provider
        return self
    
    def with_api_key(self, api_key: str) -> 'APIConfigBuilder':
        self._api_key = api_key
        return self
    
    def with_base_url(self, base_url: str) -> 'APIConfigBuilder':
        self._base_url = base_url
        return self
    
    def with_model(self, model_name: str) -> 'APIConfigBuilder':
        self._model_name = model_name
        return self
    
    def with_max_tokens(self, max_tokens: int) -> 'APIConfigBuilder':
        self._max_tokens = max_tokens
        return self
    
    def with_temperature(self, temperature: float) -> 'APIConfigBuilder':
        self._temperature = temperature
        return self
    
    def with_timeout(self, timeout: int) -> 'APIConfigBuilder':
        self._timeout = timeout
        return self
    
    def with_max_retries(self, max_retries: int) -> 'APIConfigBuilder':
        self._max_retries = max_retries
        return self
    
    def with_rate_limit(self, rate_limit: int) -> 'APIConfigBuilder':
        self._rate_limit = rate_limit
        return self
    
    def with_fallback(self, fallback_provider: str) -> 'APIConfigBuilder':
        self._fallback_provider = fallback_provider
        return self
    
    def enabled(self, enabled: bool = True) -> 'APIConfigBuilder':
        self._enabled = enabled
        return self
    
    def build(self) -> APIConfig:
        return APIConfig(
            provider=self._provider,
            api_key=self._api_key,
            base_url=self._base_url,
            model_name=self._model_name,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            timeout=self._timeout,
            max_retries=self._max_retries,
            enabled=self._enabled,
            metadata=self._metadata,
            rate_limit=self._rate_limit,
            daily_quota=self._daily_quota,
            priority=self._priority,
            fallback_provider=self._fallback_provider,
            headers=self._headers,
        )


class AgentConfigBuilder(ConfigBuilder[AgentInstanceConfig]):
    """Agent 配置构建器"""
    
    def __init__(self, name: str):
        self._name = name
        self.reset()
    
    def reset(self) -> 'AgentConfigBuilder':
        self._agent_type = AgentType.BASIC
        self._langgraph_type = LangGraphAgentType.REACT
        self._description = ""
        self._max_iterations = 10
        self._timeout = 300.0
        self._model = "gpt-4"
        self._temperature = 0.7
        self._max_tokens = 4096
        self._system_prompt = None
        self._enabled_tools = []
        self._checkpointer = CheckpointerConfig()
        self._strategies = StrategyConfig()
        self._callbacks = CallbackConfig()
        self._metadata = {}
        self._tags = []
        return self
    
    def with_type(self, agent_type: Union[str, AgentType]) -> 'AgentConfigBuilder':
        self._agent_type = AgentType(agent_type) if isinstance(agent_type, str) else agent_type
        self._langgraph_type = get_langgraph_type(self._agent_type)
        return self
    
    def with_langgraph_type(self, langgraph_type: Union[str, LangGraphAgentType]) -> 'AgentConfigBuilder':
        self._langgraph_type = LangGraphAgentType(langgraph_type) if isinstance(langgraph_type, str) else langgraph_type
        return self
    
    def with_description(self, description: str) -> 'AgentConfigBuilder':
        self._description = description
        return self
    
    def with_max_iterations(self, max_iterations: int) -> 'AgentConfigBuilder':
        self._max_iterations = max_iterations
        return self
    
    def with_timeout(self, timeout: float) -> 'AgentConfigBuilder':
        self._timeout = timeout
        return self
    
    def with_model(self, model: str) -> 'AgentConfigBuilder':
        self._model = model
        return self
    
    def with_temperature(self, temperature: float) -> 'AgentConfigBuilder':
        self._temperature = temperature
        return self
    
    def with_system_prompt(self, system_prompt: str) -> 'AgentConfigBuilder':
        self._system_prompt = system_prompt
        return self
    
    def with_tools(self, tools: List[str]) -> 'AgentConfigBuilder':
        self._enabled_tools = tools
        return self
    
    def with_checkpointer(self, checkpointer: CheckpointerConfig) -> 'AgentConfigBuilder':
        self._checkpointer = checkpointer
        return self
    
    def with_strategies(self, strategies: StrategyConfig) -> 'AgentConfigBuilder':
        self._strategies = strategies
        return self
    
    def with_callbacks(self, callbacks: CallbackConfig) -> 'AgentConfigBuilder':
        self._callbacks = callbacks
        return self
    
    def with_tags(self, tags: List[str]) -> 'AgentConfigBuilder':
        self._tags = tags
        return self
    
    def build(self) -> AgentInstanceConfig:
        return AgentInstanceConfig(
            name=self._name,
            agent_type=self._agent_type,
            langgraph_type=self._langgraph_type,
            description=self._description,
            max_iterations=self._max_iterations,
            timeout=self._timeout,
            model=self._model,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            system_prompt=self._system_prompt,
            enabled_tools=self._enabled_tools,
            checkpointer=self._checkpointer,
            strategies=self._strategies,
            callbacks=self._callbacks,
            metadata=self._metadata,
            tags=self._tags,
        )


# ==================== 配置观察者 ====================

class ConfigObserver(ABC):
    """配置观察者基类"""
    
    @abstractmethod
    def on_config_change(self, event_type: ConfigEventType, 
                        config_name: str, 
                        old_value: Optional[BaseConfig],
                        new_value: Optional[BaseConfig]):
        """配置变更回调"""
        pass


class LoggingConfigObserver(ConfigObserver):
    """日志配置观察者"""
    
    def on_config_change(self, event_type: ConfigEventType,
                        config_name: str,
                        old_value: Optional[BaseConfig],
                        new_value: Optional[BaseConfig]):
        logger.info(f"Config changed: {config_name}, event={event_type.value}")
        if old_value and new_value:
            diff = old_value.diff(new_value)
            if diff:
                logger.debug(f"Config diff: {diff}")


class AuditConfigObserver(ConfigObserver):
    """审计配置观察者"""
    
    def __init__(self, audit_file: str = "config_audit.log"):
        self.audit_file = audit_file
    
    def on_config_change(self, event_type: ConfigEventType,
                        config_name: str,
                        old_value: Optional[BaseConfig],
                        new_value: Optional[BaseConfig]):
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type.value,
            "config_name": config_name,
            "old_hash": old_value.get_hash() if old_value else None,
            "new_hash": new_value.get_hash() if new_value else None,
        }
        
        try:
            with open(self.audit_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {str(e)}")


class WebhookConfigObserver(ConfigObserver):
    """Webhook 配置观察者"""
    
    def __init__(self, webhook_url: str, timeout: int = 10):
        self.webhook_url = webhook_url
        self.timeout = timeout
    
    def on_config_change(self, event_type: ConfigEventType,
                        config_name: str,
                        old_value: Optional[BaseConfig],
                        new_value: Optional[BaseConfig]):
        try:
            import requests
            payload = {
                "event_type": event_type.value,
                "config_name": config_name,
                "timestamp": datetime.now().isoformat(),
            }
            requests.post(self.webhook_url, json=payload, timeout=self.timeout)
        except Exception as e:
            logger.error(f"Failed to send webhook: {str(e)}")


# ==================== 配置模板 ====================

@dataclass
class ConfigTemplate:
    """配置模板"""
    name: str
    description: str
    config_type: str
    config_data: Dict[str, Any]
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "config_type": self.config_type,
            "config_data": self.config_data,
            "version": self.version,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigTemplate':
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            config_type=data["config_type"],
            config_data=data["config_data"],
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
        )


class ConfigTemplateManager:
    """配置模板管理器"""
    
    # 预定义模板
    BUILTIN_TEMPLATES: Dict[str, ConfigTemplate] = {}
    
    def __init__(self, template_dir: str = None):
        self.template_dir = template_dir or os.path.join(
            os.path.dirname(__file__), "../../config/templates"
        )
        self.templates: Dict[str, ConfigTemplate] = {}
        self._load_builtin_templates()
        self._load_templates_from_dir()
    
    def _load_builtin_templates(self):
        """加载内置模板"""
        # 基础 Agent 模板
        self.templates["basic_agent"] = ConfigTemplate(
            name="basic_agent",
            description="基础 ReAct Agent 模板",
            config_type="agent",
            config_data={
                "agent_type": "basic",
                "langgraph_type": "react",
                "max_iterations": 10,
                "timeout": 300.0,
                "model": "gpt-4",
                "temperature": 0.7,
            }
        )
        
        # 训练助手模板
        self.templates["training_assistant"] = ConfigTemplate(
            name="training_assistant",
            description="训练助手 Agent 模板",
            config_type="agent",
            config_data={
                "agent_type": "training_assistant",
                "langgraph_type": "plan_execute",
                "max_iterations": 15,
                "timeout": 600.0,
                "model": "gpt-4",
                "temperature": 0.7,
                "enabled_tools": [
                    "create_training_session",
                    "get_training_progress",
                    "download_trained_model",
                    "get_training_history",
                    "get_training_statistics"
                ],
            },
            tags=["training", "assistant"]
        )
        
        # 代码助手模板
        self.templates["code_assistant"] = ConfigTemplate(
            name="code_assistant",
            description="代码助手 Agent 模板",
            config_type="agent",
            config_data={
                "agent_type": "code_assistant",
                "langgraph_type": "chain_of_thought",
                "max_iterations": 12,
                "timeout": 300.0,
                "model": "gpt-4",
                "temperature": 0.3,
                "enabled_tools": ["code_execute", "file_read", "file_write"],
            },
            tags=["code", "assistant"]
        )
        
        # 数据分析师模板
        self.templates["data_analyst"] = ConfigTemplate(
            name="data_analyst",
            description="数据分析师 Agent 模板",
            config_type="agent",
            config_data={
                "agent_type": "data_analyst",
                "langgraph_type": "plan_execute",
                "max_iterations": 15,
                "timeout": 600.0,
                "model": "gpt-4",
                "temperature": 0.5,
                "enabled_tools": ["data_query", "data_visualize", "calculate"],
            },
            tags=["data", "analysis"]
        )
        
        # Redis 检查点器模板
        self.templates["redis_checkpointer"] = ConfigTemplate(
            name="redis_checkpointer",
            description="Redis 检查点器配置模板",
            config_type="checkpointer",
            config_data={
                "type": "redis",
                "enabled": True,
                "max_checkpoints": 100,
                "compression": "gzip",
                "redis_host": "localhost",
                "redis_port": 6379,
                "redis_prefix": "checkpoint:",
            }
        )
        
        # 生产环境回调模板
        self.templates["production_callbacks"] = ConfigTemplate(
            name="production_callbacks",
            description="生产环境回调配置模板",
            config_type="callback",
            config_data={
                "enabled": True,
                "enable_logging": True,
                "enable_metrics": True,
                "enable_streaming": True,
                "log_level": "INFO",
                "log_format": "json",
                "metrics_backend": "prometheus",
            },
            tags=["production"]
        )
    
    def _load_templates_from_dir(self):
        """从目录加载模板"""
        if not os.path.exists(self.template_dir):
            return
        
        try:
            for filename in os.listdir(self.template_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.template_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        template = ConfigTemplate.from_dict(data)
                        self.templates[template.name] = template
        except Exception as e:
            logger.error(f"Failed to load templates: {str(e)}")
    
    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """获取模板"""
        return self.templates.get(name)
    
    def list_templates(self, config_type: str = None, tags: List[str] = None) -> List[ConfigTemplate]:
        """列出模板"""
        templates = list(self.templates.values())
        
        if config_type:
            templates = [t for t in templates if t.config_type == config_type]
        
        if tags:
            templates = [t for t in templates if any(tag in t.tags for tag in tags)]
        
        return templates
    
    def save_template(self, template: ConfigTemplate) -> bool:
        """保存模板"""
        try:
            self.templates[template.name] = template
            
            # 保存到文件
            os.makedirs(self.template_dir, exist_ok=True)
            filepath = os.path.join(self.template_dir, f"{template.name}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(template.to_dict(), f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save template: {str(e)}")
            return False
    
    def delete_template(self, name: str) -> bool:
        """删除模板"""
        if name in self.templates:
            del self.templates[name]
            
            filepath = os.path.join(self.template_dir, f"{name}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return True
        return False
    
    def create_config_from_template(self, template_name: str, **overrides) -> Optional[BaseConfig]:
        """从模板创建配置"""
        template = self.get_template(template_name)
        if not template:
            return None
        
        # 合并覆盖配置
        config_data = {**template.config_data, **overrides}
        
        # 根据类型创建配置
        if template.config_type == "agent":
            return AgentInstanceConfig.from_dict(config_data)
        elif template.config_type == "checkpointer":
            return CheckpointerConfig.from_dict(config_data)
        elif template.config_type == "callback":
            return CallbackConfig.from_dict(config_data)
        elif template.config_type == "strategy":
            return StrategyConfig.from_dict(config_data)
        elif template.config_type == "api":
            return APIConfig.from_dict(config_data)
        
        return None


# ==================== 配置版本管理 ====================

@dataclass
class ConfigVersion:
    """配置版本"""
    version_id: str
    config_name: str
    config_data: Dict[str, Any]
    created_at: datetime
    created_by: str = "system"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "config_name": self.config_name,
            "config_data": self.config_data,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
        }


class ConfigVersionManager:
    """配置版本管理器"""
    
    def __init__(self, max_versions: int = 50):
        self.max_versions = max_versions
        self._versions: Dict[str, List[ConfigVersion]] = {}
        self._lock = threading.Lock()
    
    def save_version(self, config_name: str, config: BaseConfig, 
                    description: str = "", created_by: str = "system") -> str:
        """保存配置版本"""
        version_id = str(uuid.uuid4())
        version = ConfigVersion(
            version_id=version_id,
            config_name=config_name,
            config_data=config.to_dict(),
            created_at=datetime.now(),
            created_by=created_by,
            description=description,
        )
        
        with self._lock:
            if config_name not in self._versions:
                self._versions[config_name] = []
            
            self._versions[config_name].append(version)
            
            # 清理旧版本
            if len(self._versions[config_name]) > self.max_versions:
                self._versions[config_name] = self._versions[config_name][-self.max_versions:]
        
        return version_id
    
    def get_version(self, config_name: str, version_id: str) -> Optional[ConfigVersion]:
        """获取指定版本"""
        with self._lock:
            versions = self._versions.get(config_name, [])
            for v in versions:
                if v.version_id == version_id:
                    return v
        return None
    
    def list_versions(self, config_name: str, limit: int = 10) -> List[ConfigVersion]:
        """列出版本"""
        with self._lock:
            versions = self._versions.get(config_name, [])
            return versions[-limit:][::-1]  # 最新的在前
    
    def rollback_to_version(self, config_name: str, version_id: str) -> Optional[Dict[str, Any]]:
        """回滚到指定版本"""
        version = self.get_version(config_name, version_id)
        if version:
            return version.config_data
        return None
    
    def compare_versions(self, config_name: str, 
                        version_id_1: str, 
                        version_id_2: str) -> Dict[str, Any]:
        """比较两个版本"""
        v1 = self.get_version(config_name, version_id_1)
        v2 = self.get_version(config_name, version_id_2)
        
        if not v1 or not v2:
            return {"error": "Version not found"}
        
        diff = {}
        all_keys = set(v1.config_data.keys()) | set(v2.config_data.keys())
        for key in all_keys:
            val1 = v1.config_data.get(key)
            val2 = v2.config_data.get(key)
            if val1 != val2:
                diff[key] = {"v1": val1, "v2": val2}
        
        return {
            "version_1": version_id_1,
            "version_2": version_id_2,
            "diff": diff,
        }


# ==================== 配置热重载 ====================

class ConfigHotReloader:
    """配置热重载器"""
    
    def __init__(self, config_file: str, reload_interval: int = 5):
        self.config_file = config_file
        self.reload_interval = reload_interval
        self._last_modified: float = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """注册重载回调"""
        self._callbacks.append(callback)
    
    def start(self):
        """启动热重载"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"Config hot reloader started for {self.config_file}")
    
    def stop(self):
        """停止热重载"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        logger.info("Config hot reloader stopped")
    
    def _watch_loop(self):
        """监视循环"""
        while self._running:
            try:
                if os.path.exists(self.config_file):
                    modified = os.path.getmtime(self.config_file)
                    if modified > self._last_modified:
                        self._last_modified = modified
                        self._trigger_reload()
            except Exception as e:
                logger.error(f"Hot reload error: {str(e)}")
            
            time.sleep(self.reload_interval)
    
    def _trigger_reload(self):
        """触发重载"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for callback in self._callbacks:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Reload callback error: {str(e)}")
            
            logger.info(f"Config reloaded from {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to reload config: {str(e)}")


# ==================== 主配置管理器 ====================

class APIConfigManager:
    """API 配置管理器
    
    管理 API 配置、Agent 配置、检查点配置和策略配置。
    支持配置验证、版本管理、热重载和观察者模式。
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None, 
                 encrypt_tokens: bool = True,
                 enable_hot_reload: bool = False,
                 enable_versioning: bool = True):
        """
        初始化 API 配置管理器
        
        Args:
            config_file: 配置文件路径
            encrypt_tokens: 是否加密存储 token
            enable_hot_reload: 是否启用热重载
            enable_versioning: 是否启用版本管理
        """
        self.config_file = config_file or os.path.join(
            os.path.dirname(__file__), "../../config/api_configs.json"
        )
        self.encrypt_tokens = encrypt_tokens
        self.configs: Dict[str, APIConfig] = {}
        self.agent_configs: Dict[str, AgentInstanceConfig] = {}
        self.global_checkpointer_config: CheckpointerConfig = CheckpointerConfig()
        self.global_strategy_config: StrategyConfig = StrategyConfig()
        self.global_callback_config: CallbackConfig = CallbackConfig()
        
        # 初始化加密密钥
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key) if encrypt_tokens else None
        
        # 验证器
        self._api_validator = APIConfigValidator()
        self._checkpointer_validator = CheckpointerConfigValidator()
        self._agent_validator = AgentConfigValidator()
        
        # 观察者
        self._observers: List[ConfigObserver] = [LoggingConfigObserver()]
        
        # 版本管理
        self._version_manager: Optional[ConfigVersionManager] = None
        if enable_versioning:
            self._version_manager = ConfigVersionManager()
        
        # 模板管理
        self._template_manager = ConfigTemplateManager()
        
        # 热重载
        self._hot_reloader: Optional[ConfigHotReloader] = None
        if enable_hot_reload:
            self._hot_reloader = ConfigHotReloader(self.config_file)
            self._hot_reloader.register_callback(self._on_config_reloaded)
            self._hot_reloader.start()
        
        # 缓存
        self._cache: Dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        
        # 加载配置
        self._load_configs()
        
        # 设置默认配置
        self._setup_default_configs()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """获取或创建加密密钥"""
        key_file = os.path.join(os.path.dirname(self.config_file), ".api_key")
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            os.makedirs(os.path.dirname(key_file), exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def _encrypt_token(self, token: str) -> str:
        """加密 token"""
        if not self.cipher or not token:
            return token
        return self.cipher.encrypt(token.encode()).decode()
    
    def _decrypt_token(self, encrypted_token: str) -> str:
        """解密 token"""
        if not self.cipher or not encrypted_token:
            return encrypted_token
        try:
            return self.cipher.decrypt(encrypted_token.encode()).decode()
        except Exception:
            return encrypted_token
    
    def _notify_observers(self, event_type: ConfigEventType, config_name: str,
                         old_value: Optional[BaseConfig], new_value: Optional[BaseConfig]):
        """通知观察者"""
        for observer in self._observers:
            try:
                observer.on_config_change(event_type, config_name, old_value, new_value)
            except Exception as e:
                logger.error(f"Observer notification failed: {str(e)}")
    
    def add_observer(self, observer: ConfigObserver):
        """添加观察者"""
        self._observers.append(observer)
    
    def remove_observer(self, observer: ConfigObserver):
        """移除观察者"""
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _on_config_reloaded(self, data: Dict[str, Any]):
        """配置重载回调"""
        try:
            # 重新加载 API 配置
            for provider_name, config_data in data.get("api_configs", {}).items():
                if self.encrypt_tokens and config_data.get("api_key"):
                    config_data["api_key"] = self._decrypt_token(config_data["api_key"])
                self.configs[provider_name] = APIConfig.from_dict(config_data)
            
            # 重新加载 Agent 配置
            for agent_name, agent_data in data.get("agent_configs", {}).items():
                self.agent_configs[agent_name] = AgentInstanceConfig.from_dict(agent_data)
            
            self._notify_observers(ConfigEventType.LOADED, "all", None, None)
        except Exception as e:
            logger.error(f"Config reload failed: {str(e)}")
    
    def _load_configs(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # 加载 API 配置
                for provider_name, config_data in data.get("api_configs", {}).items():
                    if self.encrypt_tokens and config_data.get("api_key"):
                        config_data["api_key"] = self._decrypt_token(config_data["api_key"])
                    self.configs[provider_name] = APIConfig.from_dict(config_data)
                    
                # 加载 Agent 配置
                for agent_name, agent_data in data.get("agent_configs", {}).items():
                    self.agent_configs[agent_name] = AgentInstanceConfig.from_dict(agent_data)
                
                # 加载全局配置
                if "global_checkpointer" in data:
                    self.global_checkpointer_config = CheckpointerConfig.from_dict(data["global_checkpointer"])
                if "global_strategy" in data:
                    self.global_strategy_config = StrategyConfig.from_dict(data["global_strategy"])
                if "global_callback" in data:
                    self.global_callback_config = CallbackConfig.from_dict(data["global_callback"])
                    
                logger.info(f"Loaded {len(self.configs)} API configs, {len(self.agent_configs)} agent configs")
                self._notify_observers(ConfigEventType.LOADED, "all", None, None)
        except Exception as e:
            logger.error(f"Failed to load API config: {str(e)}")
    
    def _save_configs(self):
        """保存配置文件"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            data = {
                "api_configs": {},
                "agent_configs": {},
                "global_checkpointer": self.global_checkpointer_config.to_dict(),
                "global_strategy": self.global_strategy_config.to_dict(),
                "global_callback": self.global_callback_config.to_dict(),
            }
            
            for provider_name, config in self.configs.items():
                config_dict = config.to_dict()
                if self.encrypt_tokens and config_dict.get("api_key"):
                    config_dict["api_key"] = self._encrypt_token(config_dict["api_key"])
                data["api_configs"][provider_name] = config_dict
                
            for agent_name, agent_config in self.agent_configs.items():
                data["agent_configs"][agent_name] = agent_config.to_dict()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info("API config saved successfully")
            self._notify_observers(ConfigEventType.SAVED, "all", None, None)
        except Exception as e:
            logger.error(f"Failed to save API config: {str(e)}")
    
    def _setup_default_configs(self):
        """设置默认配置"""
        default_configs = {
            APIProvider.OPENAI: {
                "provider": APIProvider.OPENAI,
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "base_url": "https://api.openai.com/v1",
                "model_name": "gpt-3.5-turbo",
                "max_tokens": 4096,
                "temperature": 0.7
            },
            APIProvider.DEEPSEEK: {
                "provider": APIProvider.DEEPSEEK,
                "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
                "base_url": "https://api.deepseek.com/v1",
                "model_name": "deepseek-chat",
                "max_tokens": 4096,
                "temperature": 0.7
            },
            APIProvider.LOCAL: {
                "provider": APIProvider.LOCAL,
                "api_key": "",
                "base_url": "http://localhost:11434",
                "model_name": "llama2",
                "max_tokens": 4096,
                "temperature": 0.7
            },
            APIProvider.ANTHROPIC: {
                "provider": APIProvider.ANTHROPIC,
                "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                "base_url": "https://api.anthropic.com/v1",
                "model_name": "claude-3-sonnet-20240229",
                "max_tokens": 4096,
                "temperature": 0.7
            }
        }
        
        for provider, config_data in default_configs.items():
            provider_name = provider.value
            if provider_name not in self.configs:
                self.configs[provider_name] = APIConfig(**config_data)
    
    # ==================== API 配置方法 ====================
    
    def add_config(self, provider: APIProvider, api_key: str, 
                  validate: bool = True, **kwargs) -> bool:
        """添加 API 配置"""
        try:
            config = APIConfig(provider=provider, api_key=api_key, **kwargs)
            
            if validate:
                errors = self._api_validator.validate(config)
                if errors:
                    logger.error(f"API config validation failed: {errors}")
                    return False
            
            old_config = self.configs.get(provider.value)
            self.configs[provider.value] = config
            self._save_configs()
            
            # 保存版本
            if self._version_manager:
                self._version_manager.save_version(f"api:{provider.value}", config)
            
            self._notify_observers(
                ConfigEventType.CREATED if old_config is None else ConfigEventType.UPDATED,
                f"api:{provider.value}", old_config, config
            )
            
            logger.info(f"Added {provider.value} API config successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to add API config: {str(e)}")
            return False
    
    def get_config(self, provider: APIProvider) -> Optional[APIConfig]:
        """获取 API 配置"""
        return self.configs.get(provider.value)
    
    def update_config(self, provider: APIProvider, validate: bool = True, **kwargs) -> bool:
        """更新 API 配置"""
        try:
            provider_name = provider.value
            if provider_name not in self.configs:
                logger.error(f"API config not found: {provider_name}")
                return False
            
            old_config = self.configs[provider_name].clone()
            config = self.configs[provider_name]
            
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            if validate:
                errors = self._api_validator.validate(config)
                if errors:
                    logger.error(f"API config validation failed: {errors}")
                    self.configs[provider_name] = old_config
                    return False
            
            self._save_configs()
            
            if self._version_manager:
                self._version_manager.save_version(f"api:{provider_name}", config)
            
            self._notify_observers(ConfigEventType.UPDATED, f"api:{provider_name}", old_config, config)
            
            logger.info(f"Updated {provider_name} API config successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to update API config: {str(e)}")
            return False
    
    def remove_config(self, provider: APIProvider) -> bool:
        """删除 API 配置"""
        try:
            provider_name = provider.value
            if provider_name in self.configs:
                old_config = self.configs[provider_name]
                del self.configs[provider_name]
                self._save_configs()
                
                self._notify_observers(ConfigEventType.DELETED, f"api:{provider_name}", old_config, None)
                
                logger.info(f"Removed {provider_name} API config successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove API config: {str(e)}")
            return False
    
    def list_configs(self, include_disabled: bool = False) -> List[Dict[str, Any]]:
        """列出所有 API 配置"""
        configs = []
        for provider_name, config in self.configs.items():
            if not include_disabled and not config.enabled:
                continue
            
            config_dict = config.to_dict()
            if config_dict.get("api_key"):
                config_dict["api_key"] = "***" + config_dict["api_key"][-4:] if len(config_dict["api_key"]) > 4 else "***"
            configs.append(config_dict)
        return configs
    
    def validate_config(self, provider: APIProvider) -> bool:
        """验证 API 配置"""
        config = self.get_config(provider)
        if not config:
            return False
        
        if not config.enabled:
            return False
        
        errors = self._api_validator.validate(config)
        return len(errors) == 0
    
    def get_available_providers(self) -> List[APIProvider]:
        """获取可用的 API 提供商"""
        return [APIProvider(p) for p in self.configs.keys() if self.validate_config(APIProvider(p))]
    
    def get_fallback_provider(self, provider: APIProvider) -> Optional[APIProvider]:
        """获取降级提供商"""
        config = self.get_config(provider)
        if config and config.fallback_provider:
            return APIProvider(config.fallback_provider)
        return None
    
    # ==================== Agent 配置方法 ====================
    
    def add_agent_config(self, config: AgentInstanceConfig, validate: bool = True) -> bool:
        """添加 Agent 配置"""
        try:
            if validate:
                errors = self._agent_validator.validate(config)
                if errors:
                    logger.error(f"Agent config validation failed: {errors}")
                    return False
            
            old_config = self.agent_configs.get(config.name)
            self.agent_configs[config.name] = config
            self._save_configs()
            
            if self._version_manager:
                self._version_manager.save_version(f"agent:{config.name}", config)
            
            self._notify_observers(
                ConfigEventType.CREATED if old_config is None else ConfigEventType.UPDATED,
                f"agent:{config.name}", old_config, config
            )
            
            logger.info(f"Added agent config: {config.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add agent config: {str(e)}")
            return False
    
    def get_agent_config(self, name: str) -> Optional[AgentInstanceConfig]:
        """获取 Agent 配置"""
        return self.agent_configs.get(name)
    
    def update_agent_config(self, name: str, validate: bool = True, **kwargs) -> bool:
        """更新 Agent 配置"""
        try:
            if name not in self.agent_configs:
                logger.error(f"Agent config not found: {name}")
                return False
            
            old_config = self.agent_configs[name].clone()
            config = self.agent_configs[name]
            
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            if validate:
                errors = self._agent_validator.validate(config)
                if errors:
                    logger.error(f"Agent config validation failed: {errors}")
                    self.agent_configs[name] = old_config
                    return False
            
            self._save_configs()
            
            if self._version_manager:
                self._version_manager.save_version(f"agent:{name}", config)
            
            self._notify_observers(ConfigEventType.UPDATED, f"agent:{name}", old_config, config)
            
            logger.info(f"Updated agent config: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update agent config: {str(e)}")
            return False
    
    def remove_agent_config(self, name: str) -> bool:
        """删除 Agent 配置"""
        try:
            if name in self.agent_configs:
                old_config = self.agent_configs[name]
                del self.agent_configs[name]
                self._save_configs()
                
                self._notify_observers(ConfigEventType.DELETED, f"agent:{name}", old_config, None)
                
                logger.info(f"Removed agent config: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove agent config: {str(e)}")
            return False
    
    def list_agent_configs(self, tags: List[str] = None) -> List[Dict[str, Any]]:
        """列出所有 Agent 配置"""
        configs = []
        for config in self.agent_configs.values():
            if tags:
                if not any(tag in config.tags for tag in tags):
                    continue
            configs.append(config.to_dict())
        return configs
    
    def create_agent_config_from_template(self, template_name: str, 
                                         agent_name: str, **overrides) -> Optional[AgentInstanceConfig]:
        """从模板创建 Agent 配置"""
        config = self._template_manager.create_config_from_template(template_name, name=agent_name, **overrides)
        if config and isinstance(config, AgentInstanceConfig):
            return config
        return None
    
    def create_default_agent_config(self, name: str, 
                                   agent_type: AgentType = AgentType.BASIC) -> AgentInstanceConfig:
        """创建默认 Agent 配置"""
        langgraph_type = get_langgraph_type(agent_type)
        default_config = get_default_config(langgraph_type)
        
        return AgentInstanceConfig(
            name=name,
            agent_type=agent_type,
            langgraph_type=langgraph_type,
            **default_config
        )
    
    # ==================== 全局配置方法 ====================
    
    def get_global_checkpointer_config(self) -> CheckpointerConfig:
        """获取全局检查点配置"""
        return self.global_checkpointer_config
    
    def set_global_checkpointer_config(self, config: CheckpointerConfig, validate: bool = True) -> bool:
        """设置全局检查点配置"""
        try:
            if validate:
                errors = self._checkpointer_validator.validate(config)
                if errors:
                    logger.error(f"Checkpointer config validation failed: {errors}")
                    return False
            
            old_config = self.global_checkpointer_config
            self.global_checkpointer_config = config
            self._save_configs()
            
            self._notify_observers(ConfigEventType.UPDATED, "global_checkpointer", old_config, config)
            
            logger.info("Updated global checkpointer config")
            return True
        except Exception as e:
            logger.error(f"Failed to set global checkpointer config: {str(e)}")
            return False
    
    def get_global_strategy_config(self) -> StrategyConfig:
        """获取全局策略配置"""
        return self.global_strategy_config
    
    def set_global_strategy_config(self, config: StrategyConfig) -> bool:
        """设置全局策略配置"""
        try:
            old_config = self.global_strategy_config
            self.global_strategy_config = config
            self._save_configs()
            
            self._notify_observers(ConfigEventType.UPDATED, "global_strategy", old_config, config)
            
            logger.info("Updated global strategy config")
            return True
        except Exception as e:
            logger.error(f"Failed to set global strategy config: {str(e)}")
            return False
    
    def get_global_callback_config(self) -> CallbackConfig:
        """获取全局回调配置"""
        return self.global_callback_config
    
    def set_global_callback_config(self, config: CallbackConfig) -> bool:
        """设置全局回调配置"""
        try:
            old_config = self.global_callback_config
            self.global_callback_config = config
            self._save_configs()
            
            self._notify_observers(ConfigEventType.UPDATED, "global_callback", old_config, config)
            
            logger.info("Updated global callback config")
            return True
        except Exception as e:
            logger.error(f"Failed to set global callback config: {str(e)}")
            return False
    
    # ==================== 版本管理方法 ====================
    
    def list_versions(self, config_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """列出配置版本"""
        if not self._version_manager:
            return []
        return [v.to_dict() for v in self._version_manager.list_versions(config_name, limit)]
    
    def rollback_config(self, config_name: str, version_id: str) -> bool:
        """回滚配置到指定版本"""
        if not self._version_manager:
            return False
        
        config_data = self._version_manager.rollback_to_version(config_name, version_id)
        if not config_data:
            return False
        
        try:
            if config_name.startswith("api:"):
                provider = config_name.split(":")[1]
                self.configs[provider] = APIConfig.from_dict(config_data)
            elif config_name.startswith("agent:"):
                name = config_name.split(":")[1]
                self.agent_configs[name] = AgentInstanceConfig.from_dict(config_data)
            
            self._save_configs()
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return False
    
    # ==================== 模板方法 ====================
    
    def list_templates(self, config_type: str = None) -> List[Dict[str, Any]]:
        """列出配置模板"""
        templates = self._template_manager.list_templates(config_type)
        return [t.to_dict() for t in templates]
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """获取配置模板"""
        template = self._template_manager.get_template(name)
        return template.to_dict() if template else None
    
    # ==================== 导入导出方法 ====================
    
    def export_configs(self, filepath: str = None, include_secrets: bool = False) -> str:
        """导出配置"""
        data = {
            "api_configs": {},
            "agent_configs": {},
            "global_checkpointer": self.global_checkpointer_config.to_dict(),
            "global_strategy": self.global_strategy_config.to_dict(),
            "global_callback": self.global_callback_config.to_dict(),
            "exported_at": datetime.now().isoformat(),
        }
        
        for provider_name, config in self.configs.items():
            config_dict = config.to_dict()
            if not include_secrets:
                config_dict["api_key"] = "***REDACTED***"
            data["api_configs"][provider_name] = config_dict
        
        for agent_name, config in self.agent_configs.items():
            data["agent_configs"][agent_name] = config.to_dict()
        
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
        
        return json_str
    
    def import_configs(self, filepath: str = None, json_str: str = None, 
                      merge: bool = True) -> bool:
        """导入配置"""
        try:
            if filepath:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif json_str:
                data = json.loads(json_str)
            else:
                return False
            
            if not merge:
                self.configs.clear()
                self.agent_configs.clear()
            
            for provider_name, config_data in data.get("api_configs", {}).items():
                if config_data.get("api_key") == "***REDACTED***":
                    existing = self.configs.get(provider_name)
                    if existing:
                        config_data["api_key"] = existing.api_key
                    else:
                        config_data["api_key"] = ""
                
                self.configs[provider_name] = APIConfig.from_dict(config_data)
            
            for agent_name, agent_data in data.get("agent_configs", {}).items():
                self.agent_configs[agent_name] = AgentInstanceConfig.from_dict(agent_data)
            
            if "global_checkpointer" in data:
                self.global_checkpointer_config = CheckpointerConfig.from_dict(data["global_checkpointer"])
            if "global_strategy" in data:
                self.global_strategy_config = StrategyConfig.from_dict(data["global_strategy"])
            if "global_callback" in data:
                self.global_callback_config = CallbackConfig.from_dict(data["global_callback"])
            
            self._save_configs()
            return True
            
        except Exception as e:
            logger.error(f"Import failed: {str(e)}")
            return False
    
    # ==================== 工具方法 ====================
    
    def get_status(self) -> Dict[str, Any]:
        """获取配置管理器状态 - 使用增强的工厂健康检查"""
        status = {
            "api_configs_count": len(self.configs),
            "agent_configs_count": len(self.agent_configs),
            "available_providers": [p.value for p in self.get_available_providers()],
            "hot_reload_enabled": self._hot_reloader is not None,
            "versioning_enabled": self._version_manager is not None,
            "observers_count": len(self._observers),
            "templates_count": len(self._template_manager.templates),
        }
        
        # 添加工厂健康状态
        try:
            factory_health = factory_health_check()
            status["factory_health"] = factory_health.get("status", "unknown")
            status["factory_details"] = factory_health.get("factories", {})
        except Exception as e:
            status["factory_health"] = "error"
            status["factory_error"] = str(e)
        
        return status
    
    def create_production_agent_from_config(self, config_name: str,
                                           llm_client: Any = None) -> Optional[Any]:
        """从配置创建生产级 Agent - 使用增强的工厂方法
        
        Args:
            config_name: 配置名称
            llm_client: LLM 客户端
            
        Returns:
            Agent 实例
        """
        config = self.get_agent_config(config_name)
        if not config:
            logger.error(f"Agent config not found: {config_name}")
            return None
        
        try:
            # 使用工厂创建生产级 Agent
            agent = create_production_agent(
                agent_type=config.langgraph_type.value,
                llm_client=llm_client,
                name=config.name,
                model=config.model,
                checkpointer_type=config.checkpointer.checkpointer_type.value if config.checkpointer else "memory"
            )
            
            logger.info(f"Created production agent from config: {config_name}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create production agent: {str(e)}")
            return None
    
    def create_enhanced_agent_from_config(self, config_name: str,
                                         enable_builtin_tools: bool = True,
                                         enable_metrics: bool = True,
                                         enable_logging: bool = True) -> Optional[Any]:
        """从配置创建增强的 Agent - 使用增强的工厂方法
        
        Args:
            config_name: 配置名称
            enable_builtin_tools: 是否启用内置工具
            enable_metrics: 是否启用指标
            enable_logging: 是否启用日志
        
        Returns:
            Agent 实例
        """
        config = self.get_agent_config(config_name)
        if not config:
            logger.error(f"Agent config not found: {config_name}")
            return None
        
        try:
            # 使用工厂创建增强 Agent
            agent = create_enhanced_agent(
                agent_type=config.langgraph_type.value,
                enable_builtin_tools=enable_builtin_tools,
                enable_metrics=enable_metrics,
                enable_logging=enable_logging,
                checkpointer_type=config.checkpointer.checkpointer_type.value if config.checkpointer else "memory",
                name=config.name,
                model=config.model
            )
            
            logger.info(f"Created enhanced agent from config: {config_name}")
            return agent
        except Exception as e:
            logger.error(f"Failed to create enhanced agent: {str(e)}")
            return None
    
    def get_agent_diagnostics_by_name(self, config_name: str, agent: Any) -> Dict[str, Any]:
        """获取 Agent 诊断信息
        
        Args:
            config_name: 配置名称
            agent: Agent 实例
            
        Returns:
            诊断信息字典
        """
        try:
            diagnostics = get_agent_diagnostics(agent)
            diagnostics["config_name"] = config_name
            
            # 添加配置信息
            config = self.get_agent_config(config_name)
            if config:
                diagnostics["config"] = config.to_dict()
            
            return diagnostics
        except Exception as e:
            return {"error": str(e), "config_name": config_name}
    
    def shutdown(self):
        """关闭配置管理器"""
        if self._hot_reloader:
            self._hot_reloader.stop()
        logger.info("Config manager shutdown complete")


# ==================== 工厂函数 ====================

def build_api_config() -> APIConfigBuilder:
    """创建 API 配置构建器"""
    return APIConfigBuilder()


def build_agent_config(name: str) -> AgentConfigBuilder:
    """创建 Agent 配置构建器"""
    return AgentConfigBuilder(name)


# ==================== 全局实例 ====================

_api_config_manager: Optional[APIConfigManager] = None


def get_api_config_manager() -> APIConfigManager:
    """获取全局 API 配置管理器实例"""
    global _api_config_manager
    if _api_config_manager is None:
        _api_config_manager = APIConfigManager()
    return _api_config_manager


def set_api_config_manager(manager: APIConfigManager):
    """设置全局 API 配置管理器实例"""
    global _api_config_manager
    _api_config_manager = manager


# 兼容旧代码
api_config_manager = get_api_config_manager()

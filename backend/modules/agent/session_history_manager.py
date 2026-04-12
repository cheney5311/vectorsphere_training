"""
会话历史管理模块

该模块提供会话历史的持久化存储、查询和管理功能，
支持多种存储后端（Redis、数据库等）。

扩展功能：
- 与 LangGraph 检查点系统集成
- 支持状态快照和恢复
- 支持分支和标签
- 支持差异比较
- 消息搜索和过滤
- 会话分支和合并
- 多级缓存策略
- 事件通知系统
- 数据导出和迁移
- 审计日志
"""

import json
import asyncio
import uuid
import hashlib
import threading
import time
import gzip
import base64
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import (
    Dict, List, Optional, Any, Union, Callable, 
    TypeVar, Generic, Set, Tuple, AsyncIterator
)
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import OrderedDict
from functools import wraps
import copy

import redis
from loguru import logger

from backend.config.config import Config

# LangGraph 检查点集成
from backend.algo.langgraph.checkpointer import (
    Checkpointer, MemoryCheckpointer, RedisCheckpointer,
    SQLiteCheckpointer, FileCheckpointer,
    EnhancedCheckpoint, CheckpointTag, CheckpointDiff,
    CompressionType, get_checkpointer,
    create_memory_checkpointer, create_redis_checkpointer,
    create_sqlite_checkpointer
)
from backend.algo.langgraph.state import (
    AgentState, AgentMessage, MessageType as AgentMessageType,
    StateCheckpoint, StateManager
)


# ==================== 枚举定义 ====================

class MessageType(Enum):
    """消息类型枚举"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    INFO = "info"


class SessionStatus(Enum):
    """会话状态枚举"""
    ACTIVE = "active"
    PAUSED = "paused"
    ARCHIVED = "archived"
    DELETED = "deleted"
    EXPIRED = "expired"


class StorageBackend(Enum):
    """存储后端枚举"""
    REDIS = "redis"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MEMORY = "memory"
    FILE = "file"


class CacheLevel(Enum):
    """缓存级别枚举"""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DATABASE = "l3_database"


class EventType(Enum):
    """事件类型枚举"""
    SESSION_CREATED = "session_created"
    SESSION_UPDATED = "session_updated"
    SESSION_DELETED = "session_deleted"
    SESSION_ARCHIVED = "session_archived"
    MESSAGE_ADDED = "message_added"
    MESSAGE_DELETED = "message_deleted"
    MESSAGE_EDITED = "message_edited"
    CHECKPOINT_CREATED = "checkpoint_created"
    CHECKPOINT_RESTORED = "checkpoint_restored"
    SESSION_FORKED = "session_forked"


# ==================== 异常定义 ====================

class SessionError(Exception):
    """会话基础异常"""
    pass


class SessionNotFoundError(SessionError):
    """会话不存在异常"""
    pass


class MessageNotFoundError(SessionError):
    """消息不存在异常"""
    pass


class CheckpointNotFoundError(SessionError):
    """检查点不存在异常"""
    pass


class StorageError(SessionError):
    """存储异常"""
    pass


class ValidationError(SessionError):
    """验证异常"""
    pass


# ==================== 数据类定义 ====================

@dataclass
class ChatMessage:
    """聊天消息数据类"""
    id: str
    session_id: str
    message_type: MessageType
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    tokens_used: int = 0
    
    # 扩展字段
    parent_id: Optional[str] = None  # 父消息ID（用于分支）
    is_edited: bool = False
    edit_history: List[Dict[str, Any]] = field(default_factory=list)
    is_pinned: bool = False
    is_deleted: bool = False
    reactions: Dict[str, int] = field(default_factory=dict)
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    embedding: Optional[List[float]] = None  # 用于语义搜索
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "message_type": self.message_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {},
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "tokens_used": self.tokens_used,
            "parent_id": self.parent_id,
            "is_edited": self.is_edited,
            "edit_history": self.edit_history,
            "is_pinned": self.is_pinned,
            "is_deleted": self.is_deleted,
            "reactions": self.reactions,
            "attachments": self.attachments,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        return cls(
            id=data["id"],
            session_id=data["session_id"],
            message_type=MessageType(data["message_type"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata"),
            tool_calls=data.get("tool_calls"),
            tool_results=data.get("tool_results"),
            tokens_used=data.get("tokens_used", 0),
            parent_id=data.get("parent_id"),
            is_edited=data.get("is_edited", False),
            edit_history=data.get("edit_history", []),
            is_pinned=data.get("is_pinned", False),
            is_deleted=data.get("is_deleted", False),
            reactions=data.get("reactions", {}),
            attachments=data.get("attachments", []),
        )
    
    def to_agent_message(self) -> Optional['AgentMessage']:
        """转换为 LangGraph AgentMessage"""
        type_mapping = {
            MessageType.USER: AgentMessageType.HUMAN,
            MessageType.ASSISTANT: AgentMessageType.AI,
            MessageType.SYSTEM: AgentMessageType.SYSTEM,
            MessageType.FUNCTION: AgentMessageType.FUNCTION,
            MessageType.TOOL: AgentMessageType.TOOL,
            MessageType.TOOL_RESULT: AgentMessageType.TOOL,
        }
        
        return AgentMessage(
            type=type_mapping.get(self.message_type, AgentMessageType.HUMAN),
            content=self.content,
            metadata=self.metadata
        )
    
    def clone(self, new_id: Optional[str] = None) -> 'ChatMessage':
        """克隆消息"""
        data = self.to_dict()
        data["id"] = new_id or str(uuid.uuid4())
        return ChatMessage.from_dict(data)


@dataclass
class SessionInfo:
    """会话信息数据类"""
    session_id: str
    user_id: str
    agent_id: str
    created_at: datetime
    last_activity: datetime
    message_count: int
    metadata: Optional[Dict[str, Any]] = None
    agent_type: str = "default"
    status: SessionStatus = SessionStatus.ACTIVE
    total_tokens: int = 0
    checkpoint_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # 扩展字段
    title: str = ""
    description: str = ""
    parent_session_id: Optional[str] = None  # 父会话（用于分支）
    branch_count: int = 0
    is_pinned: bool = False
    is_archived: bool = False
    model_name: Optional[str] = None
    system_prompt: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    version: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": self.message_count,
            "metadata": self.metadata or {},
            "agent_type": self.agent_type,
            "status": self.status.value if isinstance(self.status, SessionStatus) else self.status,
            "total_tokens": self.total_tokens,
            "checkpoint_id": self.checkpoint_id,
            "tags": self.tags,
            "title": self.title,
            "description": self.description,
            "parent_session_id": self.parent_session_id,
            "branch_count": self.branch_count,
            "is_pinned": self.is_pinned,
            "is_archived": self.is_archived,
            "model_name": self.model_name,
            "system_prompt": self.system_prompt,
            "settings": self.settings,
            "version": self.version,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionInfo':
        status = data.get("status", "active")
        if isinstance(status, str):
            status = SessionStatus(status)
        
        return cls(
            session_id=data["session_id"],
            user_id=data["user_id"],
            agent_id=data["agent_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            message_count=data["message_count"],
            metadata=data.get("metadata"),
            agent_type=data.get("agent_type", "default"),
            status=status,
            total_tokens=data.get("total_tokens", 0),
            checkpoint_id=data.get("checkpoint_id"),
            tags=data.get("tags", []),
            title=data.get("title", ""),
            description=data.get("description", ""),
            parent_session_id=data.get("parent_session_id"),
            branch_count=data.get("branch_count", 0),
            is_pinned=data.get("is_pinned", False),
            is_archived=data.get("is_archived", False),
            model_name=data.get("model_name"),
            system_prompt=data.get("system_prompt"),
            settings=data.get("settings", {}),
            version=data.get("version", 1),
        )


@dataclass
class SessionCheckpoint:
    """会话检查点"""
    checkpoint_id: str
    session_id: str
    created_at: datetime
    state_data: Dict[str, Any]
    message_index: int
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    # 扩展字段
    parent_checkpoint_id: Optional[str] = None
    is_auto: bool = False
    compression: CompressionType = CompressionType.NONE
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "state_data": self.state_data,
            "message_index": self.message_index,
            "tags": self.tags,
            "description": self.description,
            "parent_checkpoint_id": self.parent_checkpoint_id,
            "is_auto": self.is_auto,
            "compression": self.compression.value,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionCheckpoint':
        compression = data.get("compression", "none")
        if isinstance(compression, str):
            compression = CompressionType(compression)
        
        return cls(
            checkpoint_id=data["checkpoint_id"],
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            state_data=data["state_data"],
            message_index=data["message_index"],
            tags=data.get("tags", []),
            description=data.get("description", ""),
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            is_auto=data.get("is_auto", False),
            compression=compression,
            size_bytes=data.get("size_bytes", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SessionEvent:
    """会话事件"""
    event_id: str
    event_type: EventType
    session_id: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "user_id": self.user_id,
        }


@dataclass
class SearchQuery:
    """搜索查询"""
    query: str = ""
    session_ids: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None
    message_types: Optional[List[MessageType]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    tags: Optional[List[str]] = None
    limit: int = 50
    offset: int = 0
    sort_by: str = "timestamp"
    sort_order: str = "desc"


@dataclass
class SearchResult:
    """搜索结果"""
    messages: List[ChatMessage]
    total_count: int
    query: SearchQuery
    took_ms: float


@dataclass 
class SessionStatistics:
    """会话统计"""
    session_id: str
    message_count: int
    user_message_count: int
    assistant_message_count: int
    tool_call_count: int
    total_tokens: int
    average_response_time_ms: float
    total_content_length: int
    duration_seconds: float
    checkpoint_count: int
    branch_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==================== 缓存系统 ====================

class CacheEntry:
    """缓存条目"""
    def __init__(self, value: Any, ttl: int = 3600):
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


class MultiLevelCache:
    """多级缓存"""
    
    def __init__(self, 
                 l1_max_size: int = 1000,
                 l1_ttl: int = 300,
                 redis_client: Optional[redis.Redis] = None,
                 l2_ttl: int = 3600):
        self._l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l1_max_size = l1_max_size
        self._l1_ttl = l1_ttl
        self._l1_lock = threading.Lock()
        
        self._redis = redis_client
        self._l2_ttl = l2_ttl
        self._l2_prefix = "session_cache:"
        
        self._hits = {"l1": 0, "l2": 0}
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        # L1 缓存查询
        with self._l1_lock:
            if key in self._l1_cache:
                entry = self._l1_cache[key]
                if not entry.is_expired():
                    self._l1_cache.move_to_end(key)
                    entry.access()
                    self._hits["l1"] += 1
                    return entry.value
                else:
                    del self._l1_cache[key]
        
        # L2 Redis 缓存查询
        if self._redis:
            try:
                data = self._redis.get(f"{self._l2_prefix}{key}")
                if data:
                    value = json.loads(data)
                    self._set_l1(key, value)
                    self._hits["l2"] += 1
                    return value
            except Exception as e:
                logger.warning(f"L2 cache get error: {str(e)}")
        
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        self._set_l1(key, value, ttl)
        
        if self._redis:
            try:
                self._redis.setex(
                    f"{self._l2_prefix}{key}",
                    ttl or self._l2_ttl,
                    json.dumps(value, ensure_ascii=False, default=str)
                )
            except Exception as e:
                logger.warning(f"L2 cache set error: {str(e)}")
    
    def _set_l1(self, key: str, value: Any, ttl: Optional[int] = None):
        with self._l1_lock:
            if key in self._l1_cache:
                del self._l1_cache[key]
            
            while len(self._l1_cache) >= self._l1_max_size:
                self._l1_cache.popitem(last=False)
            
            self._l1_cache[key] = CacheEntry(value, ttl or self._l1_ttl)
    
    def delete(self, key: str):
        with self._l1_lock:
            self._l1_cache.pop(key, None)
        
        if self._redis:
            try:
                self._redis.delete(f"{self._l2_prefix}{key}")
            except Exception as e:
                logger.warning(f"L2 cache delete error: {str(e)}")
    
    def clear(self):
        with self._l1_lock:
            self._l1_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        total = self._hits["l1"] + self._hits["l2"] + self._misses
        return {
            "l1_size": len(self._l1_cache),
            "l1_hits": self._hits["l1"],
            "l2_hits": self._hits["l2"],
            "misses": self._misses,
            "hit_rate": (self._hits["l1"] + self._hits["l2"]) / total if total > 0 else 0,
        }


# ==================== 事件系统 ====================

class EventListener(ABC):
    """事件监听器抽象基类"""
    
    @abstractmethod
    async def on_event(self, event: SessionEvent):
        pass


class LoggingEventListener(EventListener):
    """日志事件监听器"""
    
    async def on_event(self, event: SessionEvent):
        logger.info(f"Session event: {event.event_type.value} - {event.session_id}")


class WebhookEventListener(EventListener):
    """Webhook 事件监听器"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def on_event(self, event: SessionEvent):
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                await session.post(self.webhook_url, json=event.to_dict())
        except Exception as e:
            logger.error(f"Webhook event error: {str(e)}")


class EventManager:
    """事件管理器"""
    
    def __init__(self):
        self._listeners: Dict[EventType, List[EventListener]] = {}
        self._all_listeners: List[EventListener] = []
        self._event_history: List[SessionEvent] = []
        self._max_history = 1000
        self._lock = threading.Lock()
    
    def add_listener(self, listener: EventListener, event_types: Optional[List[EventType]] = None):
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
                self._listeners[event_type] = [
                    l for l in self._listeners[event_type] if l != listener
                ]
    
    async def emit(self, event: SessionEvent):
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
            except Exception as e:
                logger.error(f"Event listener error: {str(e)}")
    
    def get_history(self, limit: int = 100) -> List[SessionEvent]:
        with self._lock:
            return list(self._event_history[-limit:])


# ==================== 数据压缩 ====================

class DataCompressor:
    """数据压缩器"""
    
    @staticmethod
    def compress(data: str, compression: CompressionType = CompressionType.GZIP) -> bytes:
        if compression == CompressionType.NONE:
            return data.encode('utf-8')
        elif compression == CompressionType.GZIP:
            return gzip.compress(data.encode('utf-8'))
        else:
            return data.encode('utf-8')
    
    @staticmethod
    def decompress(data: bytes, compression: CompressionType = CompressionType.GZIP) -> str:
        if compression == CompressionType.NONE:
            return data.decode('utf-8')
        elif compression == CompressionType.GZIP:
            return gzip.decompress(data).decode('utf-8')
        else:
            return data.decode('utf-8')


# ==================== 会话构建器 ====================

class SessionBuilder:
    """会话构建器"""
    
    def __init__(self, user_id: str, agent_id: str):
        self._user_id = user_id
        self._agent_id = agent_id
        self._session_id = str(uuid.uuid4())
        self._agent_type = "default"
        self._title = ""
        self._description = ""
        self._tags: List[str] = []
        self._metadata: Dict[str, Any] = {}
        self._model_name: Optional[str] = None
        self._system_prompt: Optional[str] = None
        self._settings: Dict[str, Any] = {}
    
    def with_session_id(self, session_id: str) -> 'SessionBuilder':
        self._session_id = session_id
        return self
    
    def with_agent_type(self, agent_type: str) -> 'SessionBuilder':
        self._agent_type = agent_type
        return self
    
    def with_title(self, title: str) -> 'SessionBuilder':
        self._title = title
        return self
    
    def with_description(self, description: str) -> 'SessionBuilder':
        self._description = description
        return self
    
    def with_tags(self, tags: List[str]) -> 'SessionBuilder':
        self._tags = tags
        return self
    
    def with_metadata(self, metadata: Dict[str, Any]) -> 'SessionBuilder':
        self._metadata = metadata
        return self
    
    def with_model(self, model_name: str) -> 'SessionBuilder':
        self._model_name = model_name
        return self
    
    def with_system_prompt(self, prompt: str) -> 'SessionBuilder':
        self._system_prompt = prompt
        return self
    
    def with_settings(self, settings: Dict[str, Any]) -> 'SessionBuilder':
        self._settings = settings
        return self
    
    def build(self) -> SessionInfo:
        now = datetime.now()
        return SessionInfo(
            session_id=self._session_id,
            user_id=self._user_id,
            agent_id=self._agent_id,
            created_at=now,
            last_activity=now,
            message_count=0,
            metadata=self._metadata,
            agent_type=self._agent_type,
            status=SessionStatus.ACTIVE,
            title=self._title,
            description=self._description,
            tags=self._tags,
            model_name=self._model_name,
            system_prompt=self._system_prompt,
            settings=self._settings,
        )


# ==================== 主服务类 ====================

class SessionHistoryManager:
    """会话历史管理器
    
    集成 LangGraph 检查点系统的生产级会话历史管理器。
    """
    
    def __init__(self, 
                 redis_client: Optional[redis.Redis] = None,
                 use_langgraph_checkpointer: bool = True,
                 enable_cache: bool = True,
                 enable_events: bool = True,
                 auto_checkpoint_interval: int = 10):
        """初始化会话历史管理器"""
        self.redis_client = redis_client or self._create_redis_client()
        
        # Redis 键前缀
        self.session_prefix = "session_history:"
        self.message_prefix = "session_messages:"
        self.session_info_prefix = "session_info:"
        self.user_sessions_prefix = "user_sessions:"
        self.checkpoint_prefix = "session_checkpoint:"
        self.message_index_prefix = "message_index:"
        self.tag_index_prefix = "tag_index:"
        
        # 默认过期时间（7天）
        self.default_ttl = 7 * 24 * 3600
        
        # LangGraph 检查点器
        self.use_langgraph = use_langgraph_checkpointer
        self._checkpointer: Optional[Checkpointer] = None
        self._state_manager: Optional[StateManager] = None
        
        if self.use_langgraph:
            self._initialize_langgraph_checkpointer()
        
        # 缓存系统
        self._cache: Optional[MultiLevelCache] = None
        if enable_cache:
            self._cache = MultiLevelCache(redis_client=self.redis_client)
        
        # 事件系统
        self._event_manager: Optional[EventManager] = None
        if enable_events:
            self._event_manager = EventManager()
            self._event_manager.add_listener(LoggingEventListener())
        
        # 自动检查点
        self._auto_checkpoint_interval = auto_checkpoint_interval
        self._message_counters: Dict[str, int] = {}
        
        # 压缩器
        self._compressor = DataCompressor()
        
        # 锁
        self._session_locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        
        logger.info("SessionHistoryManager initialized")
    
    def _create_redis_client(self) -> redis.Redis:
        """创建Redis客户端"""
        try:
            config = Config()
            return redis.Redis(
                host=config.redis.host,
                port=config.redis.port,
                password=config.redis.password,
                db=config.redis.db,
                decode_responses=True
            )
        except Exception as e:
            logger.error(f"Failed to create Redis client: {str(e)}")
            return redis.Redis(host='localhost', port=6379, decode_responses=True)
    
    def _initialize_langgraph_checkpointer(self):
        """初始化 LangGraph 检查点器"""
        try:
            self._checkpointer = create_redis_checkpointer(
                host=self.redis_client.connection_pool.connection_kwargs.get('host', 'localhost'),
                port=self.redis_client.connection_pool.connection_kwargs.get('port', 6379),
                prefix="langgraph_checkpoint:"
            )
            self._state_manager = StateManager()
            logger.info("LangGraph checkpointer initialized with Redis backend")
        except Exception as e:
            logger.warning(f"Failed to create Redis checkpointer: {str(e)}")
            try:
                self._checkpointer = create_memory_checkpointer()
                self._state_manager = StateManager()
                logger.info("LangGraph checkpointer initialized with memory backend")
            except Exception as e2:
                logger.error(f"Failed to initialize LangGraph checkpointer: {str(e2)}")
                self.use_langgraph = False
    
    def _get_session_lock(self, session_id: str) -> threading.Lock:
        """获取会话锁"""
        with self._locks_lock:
            if session_id not in self._session_locks:
                self._session_locks[session_id] = threading.Lock()
            return self._session_locks[session_id]
    
    async def _emit_event(self, event_type: EventType, session_id: str, 
                         data: Dict[str, Any] = None, user_id: str = None):
        """发送事件"""
        if self._event_manager:
            event = SessionEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                session_id=session_id,
                timestamp=datetime.now(),
                data=data or {},
                user_id=user_id
            )
            await self._event_manager.emit(event)
    
    # ==================== 会话管理 ====================
    
    async def create_session(self, 
                           session_id: str = None,
                           user_id: str = None,
                           agent_id: str = None,
                           agent_type: str = "default",
                           metadata: Optional[Dict[str, Any]] = None,
                           title: str = "",
                           model_name: Optional[str] = None,
                           system_prompt: Optional[str] = None) -> SessionInfo:
        """创建新会话"""
        session_id = session_id or str(uuid.uuid4())
        now = datetime.now()
        
        session_info = SessionInfo(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            created_at=now,
            last_activity=now,
            message_count=0,
            metadata=metadata,
            agent_type=agent_type,
            status=SessionStatus.ACTIVE,
            title=title,
            model_name=model_name,
            system_prompt=system_prompt,
        )
        
        try:
            session_key = f"{self.session_info_prefix}{session_id}"
            self.redis_client.setex(
                session_key,
                self.default_ttl,
                json.dumps(session_info.to_dict(), ensure_ascii=False)
            )
            
            if user_id:
                user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
                self.redis_client.sadd(user_sessions_key, session_id)
                self.redis_client.expire(user_sessions_key, self.default_ttl)
            
            if self._cache:
                self._cache.set(f"session:{session_id}", session_info.to_dict())
            
            if self.use_langgraph and self._checkpointer:
                initial_state = AgentState(
                    thread_id=session_id,
                    agent_id=agent_id,
                    created_at=now
                )
                if user_id:
                    initial_state.metadata["user_id"] = user_id
                    
                try:
                    checkpoint_id = self._checkpointer.save(state=initial_state)
                    session_info.checkpoint_id = checkpoint_id
                    self.redis_client.setex(
                        session_key,
                        self.default_ttl,
                        json.dumps(session_info.to_dict(), ensure_ascii=False)
                    )
                except Exception as e:
                    logger.warning(f"Failed to create initial checkpoint: {str(e)}")
            
            await self._emit_event(EventType.SESSION_CREATED, session_id, 
                                  {"user_id": user_id}, user_id)
            
            logger.info(f"Session created: {session_id}")
            return session_info
            
        except Exception as e:
            logger.error(f"Failed to create session: {str(e)}")
            raise StorageError(f"Failed to create session: {str(e)}")
    
    async def create_session_with_builder(self, builder: SessionBuilder) -> SessionInfo:
        """使用构建器创建会话"""
        session_info = builder.build()
        
        session_key = f"{self.session_info_prefix}{session_info.session_id}"
        self.redis_client.setex(
            session_key,
            self.default_ttl,
            json.dumps(session_info.to_dict(), ensure_ascii=False)
        )
        
        if session_info.user_id:
            user_sessions_key = f"{self.user_sessions_prefix}{session_info.user_id}"
            self.redis_client.sadd(user_sessions_key, session_info.session_id)
            self.redis_client.expire(user_sessions_key, self.default_ttl)
        
        if self._cache:
            self._cache.set(f"session:{session_info.session_id}", session_info.to_dict())
        
        await self._emit_event(EventType.SESSION_CREATED, session_info.session_id,
                              {"user_id": session_info.user_id}, session_info.user_id)
        
        return session_info
    
    async def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """获取会话信息"""
        if self._cache:
            cached = self._cache.get(f"session:{session_id}")
            if cached:
                return SessionInfo.from_dict(cached)
        
        try:
            session_key = f"{self.session_info_prefix}{session_id}"
            data = self.redis_client.get(session_key)
            
            if data:
                session_dict = json.loads(data)
                session_info = SessionInfo.from_dict(session_dict)
                
                if self._cache:
                    self._cache.set(f"session:{session_id}", session_dict)
                
                return session_info
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session info: {str(e)}")
            return None
    
    async def update_session(self, session_id: str, **kwargs) -> Optional[SessionInfo]:
        """更新会话信息"""
        session_info = await self.get_session_info(session_id)
        if not session_info:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        
        for key, value in kwargs.items():
            if hasattr(session_info, key):
                setattr(session_info, key, value)
        
        session_info.last_activity = datetime.now()
        session_info.version += 1
        
        session_key = f"{self.session_info_prefix}{session_id}"
        self.redis_client.setex(
            session_key,
            self.default_ttl,
            json.dumps(session_info.to_dict(), ensure_ascii=False)
        )
        
        if self._cache:
            self._cache.set(f"session:{session_id}", session_info.to_dict())
        
        await self._emit_event(EventType.SESSION_UPDATED, session_id, kwargs)
        
        return session_info
    
    async def delete_session(self, session_id: str, hard_delete: bool = False) -> bool:
        """删除会话"""
        try:
            session_info = await self.get_session_info(session_id)
            
            if hard_delete:
                keys_to_delete = [
                    f"{self.session_info_prefix}{session_id}",
                    f"{self.message_prefix}{session_id}",
                    f"{self.session_prefix}{session_id}",
                ]
                
                checkpoint_keys = self.redis_client.keys(f"{self.checkpoint_prefix}{session_id}:*")
                keys_to_delete.extend(checkpoint_keys)
                
                if keys_to_delete:
                    self.redis_client.delete(*keys_to_delete)
                
                if session_info and session_info.user_id:
                    user_sessions_key = f"{self.user_sessions_prefix}{session_info.user_id}"
                    self.redis_client.srem(user_sessions_key, session_id)
            else:
                await self.update_session(session_id, status=SessionStatus.DELETED)
            
            if self._cache:
                self._cache.delete(f"session:{session_id}")
            
            await self._emit_event(EventType.SESSION_DELETED, session_id,
                                  {"hard_delete": hard_delete})
            
            logger.info(f"Session deleted: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete session: {str(e)}")
            return False
    

    async def archive_session(self, session_id: str) -> bool:
        """归档会话"""
        try:
            await self.update_session(
                session_id, 
                status=SessionStatus.ARCHIVED,
                is_archived=True
            )
            await self._emit_event(EventType.SESSION_ARCHIVED, session_id)
            return True
        except Exception as e:
            logger.error(f"Failed to archive session: {str(e)}")
            return False
    
    async def fork_session(self, 
                          session_id: str, 
                          fork_from_message_index: Optional[int] = None,
                          new_session_id: Optional[str] = None) -> Optional[SessionInfo]:
        """分支会话"""
        try:
            original_session = await self.get_session_info(session_id)
            if not original_session:
                raise SessionNotFoundError(f"Session not found: {session_id}")
            
            messages = await self.get_session_messages(session_id)
            
            if fork_from_message_index is not None:
                messages = messages[:fork_from_message_index + 1]
            
            new_session_id = new_session_id or str(uuid.uuid4())
            now = datetime.now()
            
            forked_session = SessionInfo(
                session_id=new_session_id,
                user_id=original_session.user_id,
                agent_id=original_session.agent_id,
                created_at=now,
                last_activity=now,
                message_count=len(messages),
                metadata=copy.deepcopy(original_session.metadata),
                agent_type=original_session.agent_type,
                status=SessionStatus.ACTIVE,
                title=f"{original_session.title} (Fork)",
                parent_session_id=session_id,
                model_name=original_session.model_name,
                system_prompt=original_session.system_prompt,
                settings=copy.deepcopy(original_session.settings),
            )
            
            session_key = f"{self.session_info_prefix}{new_session_id}"
            self.redis_client.setex(
                session_key,
                self.default_ttl,
                json.dumps(forked_session.to_dict(), ensure_ascii=False)
            )
            
            message_key = f"{self.message_prefix}{new_session_id}"
            for message in reversed(messages):
                new_message = message.clone()
                new_message.session_id = new_session_id
                self.redis_client.lpush(
                    message_key,
                    json.dumps(new_message.to_dict(), ensure_ascii=False)
                )
            self.redis_client.expire(message_key, self.default_ttl)
            
            original_session.branch_count += 1
            await self.update_session(session_id, branch_count=original_session.branch_count)
            
            await self._emit_event(EventType.SESSION_FORKED, new_session_id,
                                  {"parent_session_id": session_id})
            
            logger.info(f"Session forked: {session_id} -> {new_session_id}")
            return forked_session
            
        except Exception as e:
            logger.error(f"Failed to fork session: {str(e)}")
            return None
    
    async def get_user_sessions(self, 
                              user_id: str, 
                              limit: Optional[int] = None,
                              status: Optional[SessionStatus] = None,
                              include_archived: bool = False) -> List[SessionInfo]:
        """获取用户的所有会话"""
        try:
            user_sessions_key = f"{self.user_sessions_prefix}{user_id}"
            session_ids = self.redis_client.smembers(user_sessions_key)
            
            sessions = []
            for session_id in session_ids:
                session_info = await self.get_session_info(session_id)
                if session_info:
                    if status and session_info.status != status:
                        continue
                    if not include_archived and session_info.is_archived:
                        continue
                    sessions.append(session_info)
            
            sessions.sort(key=lambda x: x.last_activity, reverse=True)
            
            if limit:
                sessions = sessions[:limit]
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get user sessions: {str(e)}")
            return []
    
    # ==================== 消息管理 ====================
    
    async def add_message(self, 
                        session_id: str, 
                        message_id: str = None,
                        message_type: MessageType = MessageType.USER,
                        content: str = "",
                        metadata: Optional[Dict[str, Any]] = None,
                        tool_calls: Optional[List[Dict[str, Any]]] = None,
                        tool_results: Optional[List[Dict[str, Any]]] = None,
                        tokens_used: int = 0,
                        parent_id: Optional[str] = None) -> ChatMessage:
        """添加消息到会话"""
        message_id = message_id or str(uuid.uuid4())
        now = datetime.now()
        
        message = ChatMessage(
            id=message_id,
            session_id=session_id,
            message_type=message_type,
            content=content,
            timestamp=now,
            metadata=metadata,
            tool_calls=tool_calls,
            tool_results=tool_results,
            tokens_used=tokens_used,
            parent_id=parent_id,
        )
        
        with self._get_session_lock(session_id):
            try:
                message_key = f"{self.message_prefix}{session_id}"
                self.redis_client.lpush(
                    message_key,
                    json.dumps(message.to_dict(), ensure_ascii=False)
                )
                self.redis_client.expire(message_key, self.default_ttl)
                
                await self._update_session_activity(session_id, tokens_used)
                
                if self.use_langgraph and self._checkpointer:
                    await self._update_checkpoint_with_message(session_id, message)
                
                self._message_counters[session_id] = self._message_counters.get(session_id, 0) + 1
                if self._auto_checkpoint_interval > 0:
                    if self._message_counters[session_id] >= self._auto_checkpoint_interval:
                        await self.create_checkpoint(session_id, description="Auto checkpoint", is_auto=True)
                        self._message_counters[session_id] = 0
                
                await self._emit_event(EventType.MESSAGE_ADDED, session_id,
                                      {"message_id": message_id, "message_type": message_type.value})
                
                logger.debug(f"Message added: {session_id}/{message_id}")
                return message
                
            except Exception as e:
                logger.error(f"Failed to add message: {str(e)}")
                raise StorageError(f"Failed to add message: {str(e)}")
    
    async def edit_message(self, 
                          session_id: str, 
                          message_id: str,
                          new_content: str) -> Optional[ChatMessage]:
        """编辑消息"""
        messages = await self.get_session_messages(session_id)
        
        for i, msg in enumerate(messages):
            if msg.id == message_id:
                msg.edit_history.append({
                    "content": msg.content,
                    "edited_at": datetime.now().isoformat()
                })
                msg.content = new_content
                msg.is_edited = True
                
                message_key = f"{self.message_prefix}{session_id}"
                self.redis_client.delete(message_key)
                
                for m in reversed(messages):
                    self.redis_client.lpush(
                        message_key,
                        json.dumps(m.to_dict(), ensure_ascii=False)
                    )
                self.redis_client.expire(message_key, self.default_ttl)
                
                await self._emit_event(EventType.MESSAGE_EDITED, session_id,
                                      {"message_id": message_id})
                
                return msg
        
        raise MessageNotFoundError(f"Message not found: {message_id}")
    
    async def delete_message(self, 
                            session_id: str, 
                            message_id: str,
                            soft_delete: bool = True) -> bool:
        """删除消息"""
        messages = await self.get_session_messages(session_id)
        
        for i, msg in enumerate(messages):
            if msg.id == message_id:
                if soft_delete:
                    msg.is_deleted = True
                else:
                    messages.pop(i)
                
                message_key = f"{self.message_prefix}{session_id}"
                self.redis_client.delete(message_key)
                
                for m in reversed(messages):
                    self.redis_client.lpush(
                        message_key,
                        json.dumps(m.to_dict(), ensure_ascii=False)
                    )
                self.redis_client.expire(message_key, self.default_ttl)
                
                await self._emit_event(EventType.MESSAGE_DELETED, session_id,
                                      {"message_id": message_id, "soft_delete": soft_delete})
                
                return True
        
        return False
    
    async def pin_message(self, session_id: str, message_id: str, pinned: bool = True) -> bool:
        """置顶/取消置顶消息"""
        messages = await self.get_session_messages(session_id)
        
        for msg in messages:
            if msg.id == message_id:
                msg.is_pinned = pinned
                
                message_key = f"{self.message_prefix}{session_id}"
                self.redis_client.delete(message_key)
                
                for m in reversed(messages):
                    self.redis_client.lpush(
                        message_key,
                        json.dumps(m.to_dict(), ensure_ascii=False)
                    )
                self.redis_client.expire(message_key, self.default_ttl)
                
                return True
        
        return False
    
    async def get_session_messages(self, 
                                 session_id: str, 
                                 limit: Optional[int] = None,
                                 offset: int = 0,
                                 message_types: Optional[List[MessageType]] = None,
                                 include_deleted: bool = False) -> List[ChatMessage]:
        """获取会话消息"""
        try:
            message_key = f"{self.message_prefix}{session_id}"
            
            start = offset
            end = offset + limit - 1 if limit else -1
            
            message_data = self.redis_client.lrange(message_key, start, end)
            
            messages = []
            for data in reversed(message_data):
                try:
                    message_dict = json.loads(data)
                    message = ChatMessage.from_dict(message_dict)
                    
                    if not include_deleted and message.is_deleted:
                        continue
                    
                    if message_types is None or message.message_type in message_types:
                        messages.append(message)
                except Exception as e:
                    logger.warning(f"Failed to parse message: {str(e)}")
                    continue
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get session messages: {str(e)}")
            return []
    
    async def get_pinned_messages(self, session_id: str) -> List[ChatMessage]:
        """获取置顶消息"""
        messages = await self.get_session_messages(session_id)
        return [m for m in messages if m.is_pinned]
    
    async def get_message_by_id(self, session_id: str, message_id: str) -> Optional[ChatMessage]:
        """根据ID获取消息"""
        messages = await self.get_session_messages(session_id, include_deleted=True)
        for msg in messages:
            if msg.id == message_id:
                return msg
        return None
    
    # ==================== 搜索功能 ====================
    
    async def search_messages(self, query: SearchQuery) -> SearchResult:
        """搜索消息"""
        start_time = time.time()
        
        all_messages = []
        session_ids = query.session_ids or []
        
        if query.user_ids:
            for user_id in query.user_ids:
                sessions = await self.get_user_sessions(user_id, include_archived=True)
                session_ids.extend([s.session_id for s in sessions])
        
        session_ids = list(set(session_ids))
        
        for session_id in session_ids:
            messages = await self.get_session_messages(
                session_id,
                message_types=query.message_types
            )
            
            for msg in messages:
                if query.date_from and msg.timestamp < query.date_from:
                    continue
                if query.date_to and msg.timestamp > query.date_to:
                    continue
                
                if query.query:
                    if query.query.lower() not in msg.content.lower():
                        continue
                
                all_messages.append(msg)
        
        total_count = len(all_messages)
        
        if query.sort_order == "desc":
            all_messages.sort(key=lambda m: getattr(m, query.sort_by), reverse=True)
        else:
            all_messages.sort(key=lambda m: getattr(m, query.sort_by))
        
        all_messages = all_messages[query.offset:query.offset + query.limit]
        
        took_ms = (time.time() - start_time) * 1000
        
        return SearchResult(
            messages=all_messages,
            total_count=total_count,
            query=query,
            took_ms=took_ms
        )
    
    # ==================== 检查点管理 ====================
    
    async def create_checkpoint(self, 
                               session_id: str,
                               description: str = "",
                               tags: Optional[List[str]] = None,
                               is_auto: bool = False,
                               compression: CompressionType = CompressionType.NONE) -> Optional[str]:
        """创建会话检查点"""
        try:
            session_info = await self.get_session_info(session_id)
            if not session_info:
                raise SessionNotFoundError(f"Session not found: {session_id}")
            
            messages = await self.get_session_messages(session_id)
            
            state_data = {
                "session_info": session_info.to_dict(),
                "messages": [m.to_dict() for m in messages]
            }
            
            state_json = json.dumps(state_data, ensure_ascii=False)
            size_bytes = len(state_json.encode('utf-8'))
            
            checkpoint = SessionCheckpoint(
                checkpoint_id=str(uuid.uuid4()),
                session_id=session_id,
                created_at=datetime.now(),
                state_data=state_data,
                message_index=len(messages),
                tags=tags or [],
                description=description,
                parent_checkpoint_id=session_info.checkpoint_id,
                is_auto=is_auto,
                compression=compression,
                size_bytes=size_bytes,
            )
            
            checkpoint_key = f"{self.checkpoint_prefix}{session_id}:{checkpoint.checkpoint_id}"
            self.redis_client.setex(
                checkpoint_key,
                self.default_ttl * 4,
                json.dumps(checkpoint.to_dict(), ensure_ascii=False)
            )
            
            session_info.checkpoint_id = checkpoint.checkpoint_id
            session_key = f"{self.session_info_prefix}{session_id}"
            self.redis_client.setex(
                session_key,
                self.default_ttl,
                json.dumps(session_info.to_dict(), ensure_ascii=False)
            )
            
            if self.use_langgraph and self._checkpointer:
                try:
                    # 将 SessionCheckpoint 数据转换为 AgentState
                    agent_state = AgentState(
                        thread_id=session_id,
                        data=state_data
                    )
                    self._checkpointer.save(
                        state=agent_state,
                        tags=tags
                    )
                except Exception as e:
                    logger.warning(f"Failed to save LangGraph checkpoint: {str(e)}")
            
            await self._emit_event(EventType.CHECKPOINT_CREATED, session_id,
                                  {"checkpoint_id": checkpoint.checkpoint_id})
            
            logger.info(f"Checkpoint created: {checkpoint.checkpoint_id}")
            return checkpoint.checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {str(e)}")
            return None
    
    async def restore_checkpoint(self, 
                                session_id: str,
                                checkpoint_id: str) -> bool:
        """恢复会话检查点"""
        try:
            checkpoint_key = f"{self.checkpoint_prefix}{session_id}:{checkpoint_id}"
            data = self.redis_client.get(checkpoint_key)
            
            if not data:
                raise CheckpointNotFoundError(f"Checkpoint not found: {checkpoint_id}")
            
            checkpoint = SessionCheckpoint.from_dict(json.loads(data))
            
            session_info = SessionInfo.from_dict(checkpoint.state_data["session_info"])
            session_info.checkpoint_id = checkpoint_id
            session_info.last_activity = datetime.now()
            
            session_key = f"{self.session_info_prefix}{session_id}"
            self.redis_client.setex(
                session_key,
                self.default_ttl,
                json.dumps(session_info.to_dict(), ensure_ascii=False)
            )
            
            message_key = f"{self.message_prefix}{session_id}"
            self.redis_client.delete(message_key)
            
            for msg_data in reversed(checkpoint.state_data["messages"]):
                self.redis_client.lpush(
                    message_key,
                    json.dumps(msg_data, ensure_ascii=False)
                )
            self.redis_client.expire(message_key, self.default_ttl)
            
            if self._cache:
                self._cache.delete(f"session:{session_id}")
            
            await self._emit_event(EventType.CHECKPOINT_RESTORED, session_id,
                                  {"checkpoint_id": checkpoint_id})
            
            logger.info(f"Checkpoint restored: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {str(e)}")
            return False
    
    async def list_checkpoints(self, 
                              session_id: str,
                              include_auto: bool = True) -> List[SessionCheckpoint]:
        """列出会话的所有检查点"""
        try:
            pattern = f"{self.checkpoint_prefix}{session_id}:*"
            keys = self.redis_client.keys(pattern)
            
            checkpoints = []
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    checkpoint = SessionCheckpoint.from_dict(json.loads(data))
                    if include_auto or not checkpoint.is_auto:
                        checkpoints.append(checkpoint)
            
            checkpoints.sort(key=lambda x: x.created_at, reverse=True)
            
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {str(e)}")
            return []
    
    async def delete_checkpoint(self, session_id: str, checkpoint_id: str) -> bool:
        """删除检查点"""
        try:
            checkpoint_key = f"{self.checkpoint_prefix}{session_id}:{checkpoint_id}"
            self.redis_client.delete(checkpoint_key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {str(e)}")
            return False
    
    async def compare_checkpoints(self, 
                                 session_id: str,
                                 checkpoint_id_1: str,
                                 checkpoint_id_2: str) -> Dict[str, Any]:
        """比较两个检查点"""
        checkpoints = await self.list_checkpoints(session_id)
        
        cp1 = None
        cp2 = None
        for cp in checkpoints:
            if cp.checkpoint_id == checkpoint_id_1:
                cp1 = cp
            if cp.checkpoint_id == checkpoint_id_2:
                cp2 = cp
        
        if not cp1 or not cp2:
            raise CheckpointNotFoundError("Checkpoint not found")
        
        messages_1 = cp1.state_data.get("messages", [])
        messages_2 = cp2.state_data.get("messages", [])
        
        ids_1 = {m["id"] for m in messages_1}
        ids_2 = {m["id"] for m in messages_2}
        
        return {
            "checkpoint_1": {
                "id": checkpoint_id_1,
                "message_count": len(messages_1),
                "created_at": cp1.created_at.isoformat(),
            },
            "checkpoint_2": {
                "id": checkpoint_id_2,
                "message_count": len(messages_2),
                "created_at": cp2.created_at.isoformat(),
            },
            "messages_added": len(ids_2 - ids_1),
            "messages_removed": len(ids_1 - ids_2),
            "messages_common": len(ids_1 & ids_2),
        }
    
    async def _update_session_activity(self, session_id: str, tokens_used: int = 0):
        """更新会话活动时间"""
        try:
            session_info = await self.get_session_info(session_id)
            if session_info:
                session_info.last_activity = datetime.now()
                session_info.message_count += 1
                session_info.total_tokens += tokens_used
                
                session_key = f"{self.session_info_prefix}{session_id}"
                self.redis_client.setex(
                    session_key,
                    self.default_ttl,
                    json.dumps(session_info.to_dict(), ensure_ascii=False)
                )
                
                if self._cache:
                    self._cache.set(f"session:{session_id}", session_info.to_dict())
                
        except Exception as e:
            logger.warning(f"Failed to update session activity: {str(e)}")
    
    async def _update_checkpoint_with_message(self, session_id: str, message: ChatMessage):
        """用新消息更新检查点"""
        if not self._checkpointer:
            return
        
        try:
            current_state = self._checkpointer.get_latest(thread_id=session_id)
            
            if current_state:
                agent_message = message.to_agent_message()
                if agent_message:
                    current_state.add_message(agent_message)
                    self._checkpointer.save(state=current_state)
        except Exception as e:
            logger.warning(f"Failed to update checkpoint: {str(e)}")
    
    # ==================== 状态管理 ====================
    
    async def get_agent_state(self, session_id: str) -> Optional['AgentState']:
        """获取会话的 Agent 状态"""
        if not self.use_langgraph or not self._checkpointer:
            return None
        
        try:
            state = self._checkpointer.get_latest(thread_id=session_id)
            return state
        except Exception as e:
            logger.error(f"Failed to get agent state: {str(e)}")
            return None
    
    async def save_agent_state(self, session_id: str, state: 'AgentState') -> Optional[str]:
        """保存 Agent 状态"""
        if not self.use_langgraph or not self._checkpointer:
            return None
        
        try:
            # 确保 thread_id 一致
            if state.thread_id != session_id:
                state.thread_id = session_id
                
            checkpoint_id = self._checkpointer.save(state=state)
            return checkpoint_id
        except Exception as e:
            logger.error(f"Failed to save agent state: {str(e)}")
            return None
    
    # ==================== 数据导出 ====================
    
    async def export_session(self, 
                            session_id: str,
                            format: str = "json",
                            include_checkpoints: bool = False) -> Dict[str, Any]:
        """导出会话数据"""
        session_info = await self.get_session_info(session_id)
        if not session_info:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        
        messages = await self.get_session_messages(session_id)
        
        export_data = {
            "session": session_info.to_dict(),
            "messages": [m.to_dict() for m in messages],
            "exported_at": datetime.now().isoformat(),
        }
        
        if include_checkpoints:
            checkpoints = await self.list_checkpoints(session_id)
            export_data["checkpoints"] = [cp.to_dict() for cp in checkpoints]
        
        return export_data
    
    async def import_session(self, data: Dict[str, Any], new_session_id: Optional[str] = None) -> SessionInfo:
        """导入会话数据"""
        session_data = data["session"]
        
        if new_session_id:
            session_data["session_id"] = new_session_id
        
        session_info = SessionInfo.from_dict(session_data)
        session_info.created_at = datetime.now()
        session_info.last_activity = datetime.now()
        
        session_key = f"{self.session_info_prefix}{session_info.session_id}"
        self.redis_client.setex(
            session_key,
            self.default_ttl,
            json.dumps(session_info.to_dict(), ensure_ascii=False)
        )
        
        if "messages" in data:
            message_key = f"{self.message_prefix}{session_info.session_id}"
            for msg_data in reversed(data["messages"]):
                if new_session_id:
                    msg_data["session_id"] = new_session_id
                self.redis_client.lpush(
                    message_key,
                    json.dumps(msg_data, ensure_ascii=False)
                )
            self.redis_client.expire(message_key, self.default_ttl)
        
        return session_info
    
    # ==================== 统计和分析 ====================
    
    async def get_session_statistics(self, session_id: str) -> SessionStatistics:
        """获取会话统计信息"""
        session_info = await self.get_session_info(session_id)
        if not session_info:
            raise SessionNotFoundError(f"Session not found: {session_id}")
        
        messages = await self.get_session_messages(session_id)
        
        user_count = sum(1 for m in messages if m.message_type == MessageType.USER)
        assistant_count = sum(1 for m in messages if m.message_type == MessageType.ASSISTANT)
        tool_count = sum(1 for m in messages if m.message_type in [MessageType.TOOL, MessageType.TOOL_RESULT])
        
        total_content = sum(len(m.content) for m in messages)
        
        duration = (session_info.last_activity - session_info.created_at).total_seconds()
        
        checkpoints = await self.list_checkpoints(session_id)
        
        return SessionStatistics(
            session_id=session_id,
            message_count=len(messages),
            user_message_count=user_count,
            assistant_message_count=assistant_count,
            tool_call_count=tool_count,
            total_tokens=session_info.total_tokens,
            average_response_time_ms=0,
            total_content_length=total_content,
            duration_seconds=duration,
            checkpoint_count=len(checkpoints),
            branch_count=session_info.branch_count,
        )
    
    async def cleanup_expired_sessions(self, days: int = 7) -> int:
        """清理过期会话"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            cleaned_count = 0
            
            pattern = f"{self.session_info_prefix}*"
            keys = self.redis_client.keys(pattern)
            
            for key in keys:
                try:
                    data = self.redis_client.get(key)
                    if data:
                        session_dict = json.loads(data)
                        last_activity = datetime.fromisoformat(session_dict["last_activity"])
                        
                        if last_activity < cutoff_time:
                            session_id = session_dict["session_id"]
                            await self.delete_session(session_id, hard_delete=True)
                            cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Error checking session {key}: {str(e)}")
                    continue
            
            logger.info(f"Cleaned up {cleaned_count} expired sessions")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {str(e)}")
            return 0
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        try:
            redis_connected = self.redis_client.ping()
        except:
            redis_connected = False
        
        return {
            "redis_connected": redis_connected,
            "langgraph_enabled": self.use_langgraph,
            "checkpointer_type": type(self._checkpointer).__name__ if self._checkpointer else None,
            "cache_enabled": self._cache is not None,
            "cache_stats": self._cache.get_stats() if self._cache else {},
            "events_enabled": self._event_manager is not None,
            "auto_checkpoint_interval": self._auto_checkpoint_interval,
        }
    
    # ==================== 事件管理 ====================
    
    def add_event_listener(self, listener: EventListener, event_types: List[EventType] = None):
        """添加事件监听器"""
        if self._event_manager:
            self._event_manager.add_listener(listener, event_types)
    
    def remove_event_listener(self, listener: EventListener):
        """移除事件监听器"""
        if self._event_manager:
            self._event_manager.remove_listener(listener)
    
    def get_event_history(self, limit: int = 100) -> List[SessionEvent]:
        """获取事件历史"""
        if self._event_manager:
            return self._event_manager.get_history(limit)
        return []


# ==================== 工厂函数 ====================

def create_session_builder(user_id: str, agent_id: str) -> SessionBuilder:
    """创建会话构建器"""
    return SessionBuilder(user_id, agent_id)


# ==================== 全局实例 ====================

_session_history_manager: Optional[SessionHistoryManager] = None


def get_session_history_manager() -> SessionHistoryManager:
    """获取会话历史管理器实例"""
    global _session_history_manager
    if _session_history_manager is None:
        _session_history_manager = SessionHistoryManager()
    return _session_history_manager


def set_session_history_manager(manager: SessionHistoryManager):
    """设置会话历史管理器实例"""
    global _session_history_manager
    _session_history_manager = manager
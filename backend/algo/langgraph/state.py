"""
Agent 状态定义 - 生产级实现

定义 LangGraph 中使用的各种状态结构，包括：
- AgentState: 核心状态类（支持版本控制、分支、回滚）
- AgentMessage: 消息结构（支持优先级、过期、标签）
- ToolCall/ToolResult: 工具调用相关
- StateCheckpoint: 状态检查点
- ExecutionContext: 执行上下文
- AgentMemory: Agent 记忆管理
- StateManager: 状态管理器
- StateChannel: 状态通道
"""

import uuid
import json
import copy
import hashlib
import logging
import threading
import zlib
import base64
import inspect
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Union, Callable, Tuple,
    TypeVar, Generic, Sequence, Annotated, Set, Iterator
)
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from functools import wraps
import re

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar('T')
S = TypeVar('S', bound='AgentState')


# =============================================================================
# 枚举类型
# =============================================================================

class MessageType(Enum):
    """消息类型"""
    SYSTEM = "system"
    HUMAN = "human"
    AI = "ai"
    TOOL = "tool"
    FUNCTION = "function"
    ERROR = "error"
    DEBUG = "debug"
    INTERNAL = "internal"


class AgentStatus(Enum):
    """Agent 执行状态"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    WAITING_HUMAN = "waiting_human"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ExecutionMode(Enum):
    """执行模式"""
    SYNC = "sync"
    ASYNC = "async"
    STREAMING = "streaming"
    BATCH = "batch"


class InterruptType(Enum):
    """中断类型"""
    NONE = "none"
    BEFORE_NODE = "before_node"
    AFTER_NODE = "after_node"
    ON_ERROR = "on_error"
    ON_TOOL_CALL = "on_tool_call"
    ON_HUMAN_REQUEST = "on_human_request"
    ON_APPROVAL_NEEDED = "on_approval_needed"
    MANUAL = "manual"


class ErrorType(Enum):
    """错误类型"""
    NONE = "none"
    VALIDATION = "validation"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    RESOURCE = "resource"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"
    TOOL_ERROR = "tool_error"
    LLM_ERROR = "llm_error"
    NETWORK = "network"
    INTERNAL = "internal"
    USER_CANCELLED = "user_cancelled"


class ContextType(Enum):
    """上下文类型"""
    CONVERSATION = "conversation"
    TASK = "task"
    SESSION = "session"
    WORKFLOW = "workflow"
    AGENT = "agent"


class MemoryType(Enum):
    """记忆类型"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class PriorityLevel(Enum):
    """优先级"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class ChannelType(Enum):
    """通道类型"""
    LAST_VALUE = "last_value"
    APPEND = "append"
    BINARY_OPERATOR = "binary_operator"
    CONTEXT = "context"


# =============================================================================
# 工具调用相关
# =============================================================================

@dataclass
class ToolCall:
    """工具调用
    
    表示 LLM 请求调用某个工具的信息。
    """
    id: str
    name: str
    arguments: Dict[str, Any]
    # 增强字段
    priority: PriorityLevel = PriorityLevel.NORMAL
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "priority": self.priority.value,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        """从字典创建"""
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            name=data.get('name', ''),
            arguments=data.get('arguments', {}),
            priority=PriorityLevel(data.get('priority', 2)),
            timeout=data.get('timeout'),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.utcnow(),
            metadata=data.get('metadata', {})
        )
    
    def can_retry(self) -> bool:
        """是否可以重试"""
        return self.retry_count < self.max_retries
    
    def increment_retry(self) -> None:
        """增加重试计数"""
        self.retry_count += 1


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_call_id: str
    name: str
    result: Any
    success: bool = True
    error: Optional[str] = None
    error_type: ErrorType = ErrorType.NONE
    execution_time: float = 0.0
    # 增强字段
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries_used: int = 0
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "result": self.result,
            "success": self.success,
            "error": self.error,
            "error_type": self.error_type.value,
            "execution_time": self.execution_time,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retries_used": self.retries_used,
            "cached": self.cached,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolResult':
        """从字典创建"""
        return cls(
            tool_call_id=data['tool_call_id'],
            name=data['name'],
            result=data['result'],
            success=data.get('success', True),
            error=data.get('error'),
            error_type=ErrorType(data.get('error_type', 'none')),
            execution_time=data.get('execution_time', 0.0),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            retries_used=data.get('retries_used', 0),
            cached=data.get('cached', False),
            metadata=data.get('metadata', {})
        )
    
    def to_message(self) -> 'AgentMessage':
        """转换为消息"""
        content = json.dumps(self.result) if not isinstance(self.result, str) else self.result
        if not self.success:
            content = f"Error ({self.error_type.value}): {self.error}"
        return AgentMessage.tool(
            content=content,
            tool_call_id=self.tool_call_id,
            name=self.name,
            metadata={
                "execution_time": self.execution_time,
                "cached": self.cached,
                "retries_used": self.retries_used
            }
        )


# =============================================================================
# 消息相关
# =============================================================================

@dataclass
class AgentMessage:
    """Agent 消息 - 增强版
    
    统一的消息格式，用于 Agent 之间以及 Agent 与用户之间的通信。
    支持优先级、过期、标签、搜索等功能。
    """
    content: str
    type: MessageType = MessageType.AI
    name: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 工具调用相关
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None
    
    # 增强字段
    priority: PriorityLevel = PriorityLevel.NORMAL
    expires_at: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    parent_id: Optional[str] = None  # 用于消息线程
    reply_to: Optional[str] = None
    
    # Token 使用统计
    token_count: Optional[int] = None
    
    # 编辑历史
    edited: bool = False
    edit_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        # 确保 tags 是 set
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "priority": self.priority.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "tags": list(self.tags),
            "parent_id": self.parent_id,
            "reply_to": self.reply_to,
            "token_count": self.token_count,
            "edited": self.edited,
            "edit_history": self.edit_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """从字典创建"""
        tool_calls = [ToolCall.from_dict(tc) for tc in data.get('tool_calls', [])]
        return cls(
            id=data.get('id', str(uuid.uuid4())),
            type=MessageType(data.get('type', 'ai')),
            content=data.get('content', ''),
            name=data.get('name'),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.utcnow(),
            metadata=data.get('metadata', {}),
            tool_calls=tool_calls,
            tool_call_id=data.get('tool_call_id'),
            priority=PriorityLevel(data.get('priority', 2)),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            tags=set(data.get('tags', [])),
            parent_id=data.get('parent_id'),
            reply_to=data.get('reply_to'),
            token_count=data.get('token_count'),
            edited=data.get('edited', False),
            edit_history=data.get('edit_history', [])
        )
    
    @classmethod
    def system(cls, content: str, **kwargs) -> 'AgentMessage':
        """创建系统消息"""
        return cls(content=content, type=MessageType.SYSTEM, **kwargs)
    
    @classmethod
    def human(cls, content: str, **kwargs) -> 'AgentMessage':
        """创建用户消息"""
        return cls(content=content, type=MessageType.HUMAN, **kwargs)
    
    @classmethod
    def ai(cls, content: str, tool_calls: List[ToolCall] = None, **kwargs) -> 'AgentMessage':
        """创建 AI 消息"""
        return cls(
            content=content, 
            type=MessageType.AI, 
            tool_calls=tool_calls or [],
            **kwargs
        )
    
    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str = None, **kwargs) -> 'AgentMessage':
        """创建工具消息"""
        return cls(
            content=content, 
            type=MessageType.TOOL, 
            tool_call_id=tool_call_id,
            name=name,
            **kwargs
        )

    @classmethod
    def error(cls, content: str, error_type: ErrorType = ErrorType.INTERNAL, **kwargs) -> 'AgentMessage':
        """创建错误消息"""
        kwargs.setdefault('metadata', {})['error_type'] = error_type.value
        return cls(content=content, type=MessageType.ERROR, **kwargs)
    
    @classmethod
    def debug(cls, content: str, **kwargs) -> 'AgentMessage':
        """创建调试消息"""
        return cls(content=content, type=MessageType.DEBUG, **kwargs)
    
    def is_expired(self) -> bool:
        """检查消息是否过期"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def set_expiry(self, seconds: int) -> None:
        """设置过期时间"""
        self.expires_at = datetime.utcnow() + timedelta(seconds=seconds)
    
    def add_tag(self, tag: str) -> None:
        """添加标签"""
        self.tags.add(tag)
    
    def remove_tag(self, tag: str) -> None:
        """移除标签"""
        self.tags.discard(tag)
    
    def has_tag(self, tag: str) -> bool:
        """检查是否有标签"""
        return tag in self.tags
    
    def edit(self, new_content: str) -> None:
        """编辑消息内容"""
        self.edit_history.append({
            'content': self.content,
            'timestamp': datetime.utcnow().isoformat()
        })
        self.content = new_content
        self.edited = True
    
    def matches(self, pattern: str) -> bool:
        """检查内容是否匹配正则表达式"""
        return bool(re.search(pattern, self.content, re.IGNORECASE))
    
    def get_content_hash(self) -> str:
        """获取内容哈希"""
        return hashlib.md5(self.content.encode()).hexdigest()
    
    def truncate(self, max_length: int = 100) -> str:
        """截断内容"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length - 3] + "..."
    
    def to_openai_format(self) -> Dict[str, Any]:
        """转换为 OpenAI 格式"""
        role_map = {
            MessageType.SYSTEM: "system",
            MessageType.HUMAN: "user",
            MessageType.AI: "assistant",
            MessageType.TOOL: "tool",
            MessageType.FUNCTION: "function",
        }
        
        msg = {
            "role": role_map.get(self.type, "user"),
            "content": self.content
        }
        
        if self.name:
            msg["name"] = self.name
        
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments)
                    }
                }
                for tc in self.tool_calls
            ]
        
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        
        return msg
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """转换为 Anthropic 格式"""
        role_map = {
            MessageType.HUMAN: "user",
            MessageType.AI: "assistant",
        }
        
        return {
            "role": role_map.get(self.type, "user"),
            "content": self.content
        }


class MessageBuffer:
    """消息缓冲区
    
    支持滑动窗口、优先级队列等功能。
    """
    
    def __init__(self, 
                 max_size: int = 100,
                 max_tokens: Optional[int] = None,
                 auto_summarize: bool = False):
        self._messages: deque = deque(maxlen=max_size)
        self.max_size = max_size
        self.max_tokens = max_tokens
        self.auto_summarize = auto_summarize
        self._total_tokens = 0
        self._lock = threading.Lock()
    
    def add(self, message: AgentMessage) -> None:
        """添加消息"""
        with self._lock:
            if self.max_tokens and message.token_count:
                self._total_tokens += message.token_count
                # 移除旧消息以满足 token 限制
                while self._total_tokens > self.max_tokens and self._messages:
                    removed = self._messages.popleft()
                    if removed.token_count:
                        self._total_tokens -= removed.token_count
            
            self._messages.append(message)
    
    def get_all(self) -> List[AgentMessage]:
        """获取所有消息"""
        with self._lock:
            return list(self._messages)
    
    def get_recent(self, n: int) -> List[AgentMessage]:
        """获取最近 n 条消息"""
        with self._lock:
            return list(self._messages)[-n:]
    
    def get_by_type(self, msg_type: MessageType) -> List[AgentMessage]:
        """按类型获取消息"""
        with self._lock:
            return [m for m in self._messages if m.type == msg_type]
    
    def get_by_tag(self, tag: str) -> List[AgentMessage]:
        """按标签获取消息"""
        with self._lock:
            return [m for m in self._messages if tag in m.tags]
    
    def search(self, pattern: str) -> List[AgentMessage]:
        """搜索消息"""
        with self._lock:
            return [m for m in self._messages if m.matches(pattern)]
    
    def filter_expired(self) -> List[AgentMessage]:
        """过滤过期消息"""
        with self._lock:
            valid = [m for m in self._messages if not m.is_expired()]
            self._messages = deque(valid, maxlen=self.max_size)
            return valid
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._messages.clear()
            self._total_tokens = 0
    
    def __len__(self) -> int:
        return len(self._messages)
    
    @property
    def total_tokens(self) -> int:
        return self._total_tokens


# =============================================================================
# 检查点和版本控制
# =============================================================================

@dataclass 
class StateCheckpoint:
    """状态检查点 - 增强版
    
    用于保存和恢复 Agent 状态，支持分支和版本控制。
    """
    checkpoint_id: str
    thread_id: str
    state: Dict[str, Any]
    parent_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 增强字段
    version: int = 1
    branch: str = "main"
    tags: List[str] = field(default_factory=list)
    description: str = ""
    compressed: bool = False
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.checksum is None:
            self.checksum = self._compute_checksum()
    
    def _compute_checksum(self) -> str:
        """计算状态校验和"""
        state_str = json.dumps(self.state, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]
    
    def verify_integrity(self) -> bool:
        """验证完整性"""
        return self.checksum == self._compute_checksum()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "thread_id": self.thread_id,
            "state": self.state,
            "parent_id": self.parent_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "version": self.version,
            "branch": self.branch,
            "tags": self.tags,
            "description": self.description,
            "compressed": self.compressed,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateCheckpoint':
        """从字典创建"""
        return cls(
            checkpoint_id=data['checkpoint_id'],
            thread_id=data['thread_id'],
            state=data['state'],
            parent_id=data.get('parent_id'),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.utcnow(),
            metadata=data.get('metadata', {}),
            version=data.get('version', 1),
            branch=data.get('branch', 'main'),
            tags=data.get('tags', []),
            description=data.get('description', ''),
            compressed=data.get('compressed', False),
            checksum=data.get('checksum')
        )
    
    def compress(self) -> 'StateCheckpoint':
        """压缩状态"""
        if self.compressed:
            return self
        
        state_bytes = json.dumps(self.state, default=str).encode()
        compressed_bytes = zlib.compress(state_bytes)
        compressed_state = base64.b64encode(compressed_bytes).decode()
        
        return StateCheckpoint(
            checkpoint_id=self.checkpoint_id,
            thread_id=self.thread_id,
            state={"_compressed": compressed_state},
            parent_id=self.parent_id,
            timestamp=self.timestamp,
            metadata=self.metadata,
            version=self.version,
            branch=self.branch,
            tags=self.tags,
            description=self.description,
            compressed=True,
            checksum=self.checksum
        )
    
    def decompress(self) -> 'StateCheckpoint':
        """解压缩状态"""
        if not self.compressed:
            return self
        
        compressed_state = self.state.get("_compressed", "")
        compressed_bytes = base64.b64decode(compressed_state)
        state_bytes = zlib.decompress(compressed_bytes)
        state = json.loads(state_bytes.decode())
        
        return StateCheckpoint(
            checkpoint_id=self.checkpoint_id,
            thread_id=self.thread_id,
            state=state,
            parent_id=self.parent_id,
            timestamp=self.timestamp,
            metadata=self.metadata,
            version=self.version,
            branch=self.branch,
            tags=self.tags,
            description=self.description,
            compressed=False,
            checksum=self.checksum
        )


@dataclass
class StateDiff:
    """状态差异
    
    记录两个状态之间的差异。
    """
    from_version: int
    to_version: int
    changes: Dict[str, Dict[str, Any]]  # {field: {old: ..., new: ...}}
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_version": self.from_version,
            "to_version": self.to_version,
            "changes": self.changes,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def compute(cls, old_state: Dict[str, Any], 
                new_state: Dict[str, Any],
                from_version: int,
                to_version: int) -> 'StateDiff':
        """计算两个状态之间的差异"""
        changes = {}
        
        all_keys = set(old_state.keys()) | set(new_state.keys())
        
        for key in all_keys:
            old_val = old_state.get(key)
            new_val = new_state.get(key)
            
            if old_val != new_val:
                changes[key] = {
                    "old": old_val,
                    "new": new_val
                }
        
        return cls(
            from_version=from_version,
            to_version=to_version,
            changes=changes
        )
    
    def apply(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """应用差异到状态"""
        result = copy.deepcopy(state)
        for key, change in self.changes.items():
            result[key] = change["new"]
        return result
    
    def reverse(self) -> 'StateDiff':
        """反转差异"""
        reversed_changes = {}
        for key, change in self.changes.items():
            reversed_changes[key] = {
                "old": change["new"],
                "new": change["old"]
            }
        return StateDiff(
            from_version=self.to_version,
            to_version=self.from_version,
            changes=reversed_changes
        )


@dataclass
class StateTransition:
    """状态转换
    
    记录状态转换的详细信息。
    """
    transition_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_node: str = ""
    to_node: str = ""
    trigger: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "trigger": self.trigger,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata
        }


@dataclass
class StateBranch:
    """状态分支
    
    用于管理状态的分支历史。
    """
    branch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "main"
    parent_branch: Optional[str] = None
    base_checkpoint_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "name": self.name,
            "parent_branch": self.parent_branch,
            "base_checkpoint_id": self.base_checkpoint_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }


# =============================================================================
# 执行上下文
# =============================================================================

@dataclass
class ExecutionMetrics:
    """执行指标"""
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0.0
    
    @property
    def success_rate(self) -> float:
        total = self.completed_nodes + self.failed_nodes
        return self.completed_nodes / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "total_tool_calls": self.total_tool_calls,
            "successful_tool_calls": self.successful_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "success_rate": self.success_rate
        }


@dataclass
class ExecutionContext:
    """执行上下文
    
    管理单次执行的上下文信息。
    """
    context_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context_type: ContextType = ContextType.TASK
    
    # 执行配置
    mode: ExecutionMode = ExecutionMode.SYNC
    timeout: Optional[float] = None
    max_iterations: int = 10
    max_tokens: Optional[int] = None
    
    # 中断控制
    interrupt_before: List[str] = field(default_factory=list)
    interrupt_after: List[str] = field(default_factory=list)
    current_interrupt: Optional[InterruptType] = None
    
    # 执行指标
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    
    # 回调配置
    callbacks: Dict[str, List[Callable]] = field(default_factory=dict)
    
    # 配置
    config: Dict[str, Any] = field(default_factory=dict)
    
    # 递归限制
    recursion_limit: int = 25
    current_recursion: int = 0
    
    # 流式输出
    stream_mode: str = "values"  # values, updates, messages
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """注册回调"""
        if event not in self.callbacks:
            self.callbacks[event] = []
        self.callbacks[event].append(callback)
    
    def trigger_callbacks(self, event: str, *args, **kwargs) -> None:
        """触发回调"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Callback error for event {event}: {e}")
    
    def should_interrupt(self, node_name: str, phase: str = "before") -> bool:
        """检查是否应该中断"""
        if phase == "before" and node_name in self.interrupt_before:
            return True
        if phase == "after" and node_name in self.interrupt_after:
            return True
        return False
    
    def check_recursion_limit(self) -> bool:
        """检查递归限制"""
        return self.current_recursion < self.recursion_limit
    
    def increment_recursion(self) -> None:
        """增加递归计数"""
        self.current_recursion += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "context_id": self.context_id,
            "context_type": self.context_type.value,
            "mode": self.mode.value,
            "timeout": self.timeout,
            "max_iterations": self.max_iterations,
            "max_tokens": self.max_tokens,
            "interrupt_before": self.interrupt_before,
            "interrupt_after": self.interrupt_after,
            "current_interrupt": self.current_interrupt.value if self.current_interrupt else None,
            "metrics": self.metrics.to_dict(),
            "config": self.config,
            "recursion_limit": self.recursion_limit,
            "current_recursion": self.current_recursion,
            "stream_mode": self.stream_mode
        }


# =============================================================================
# Agent 记忆
# =============================================================================

@dataclass
class MemoryEntry:
    """记忆条目"""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    memory_type: MemoryType = MemoryType.SHORT_TERM
    importance: float = 0.5  # 0-1
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    expires_at: Optional[datetime] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None  # 用于向量检索
    
    def __post_init__(self):
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
    
    def access(self) -> None:
        """访问记忆"""
        self.accessed_at = datetime.utcnow()
        self.access_count += 1
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat(),
            "access_count": self.access_count,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "tags": list(self.tags),
            "metadata": self.metadata,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        return cls(
            entry_id=data.get('entry_id', str(uuid.uuid4())),
            content=data.get('content'),
            memory_type=MemoryType(data.get('memory_type', 'short_term')),
            importance=data.get('importance', 0.5),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.utcnow(),
            accessed_at=datetime.fromisoformat(data['accessed_at']) if 'accessed_at' in data else datetime.utcnow(),
            access_count=data.get('access_count', 0),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            tags=set(data.get('tags', [])),
            metadata=data.get('metadata', {}),
            embedding=data.get('embedding')
        )


class AgentMemory:
    """Agent 记忆管理
    
    管理短期、长期、情景、语义等多种类型的记忆。
    """
    
    def __init__(self,
                 short_term_capacity: int = 50,
                 long_term_capacity: int = 1000,
                 importance_threshold: float = 0.7):
        self._memories: Dict[str, MemoryEntry] = {}
        self._by_type: Dict[MemoryType, List[str]] = defaultdict(list)
        self._by_tag: Dict[str, List[str]] = defaultdict(list)
        
        self.short_term_capacity = short_term_capacity
        self.long_term_capacity = long_term_capacity
        self.importance_threshold = importance_threshold
        
        self._lock = threading.Lock()
    
    def add(self, content: Any, 
            memory_type: MemoryType = MemoryType.SHORT_TERM,
            importance: float = 0.5,
            tags: Set[str] = None,
            ttl_seconds: Optional[int] = None,
            metadata: Dict[str, Any] = None) -> str:
        """添加记忆"""
        with self._lock:
            entry = MemoryEntry(
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags or set(),
                metadata=metadata or {}
            )
            
            if ttl_seconds:
                entry.expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            
            self._memories[entry.entry_id] = entry
            self._by_type[memory_type].append(entry.entry_id)
            
            for tag in entry.tags:
                self._by_tag[tag].append(entry.entry_id)
            
            # 检查容量并进行清理
            self._cleanup()
            
            return entry.entry_id
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """获取记忆"""
        with self._lock:
            entry = self._memories.get(entry_id)
            if entry and not entry.is_expired():
                entry.access()
                return entry
            return None
    
    def search(self, 
               query: str = None,
               memory_type: MemoryType = None,
               tags: Set[str] = None,
               min_importance: float = 0.0,
               limit: int = 10) -> List[MemoryEntry]:
        """搜索记忆"""
        with self._lock:
            candidates = []
            
            if memory_type:
                entry_ids = self._by_type.get(memory_type, [])
            else:
                entry_ids = list(self._memories.keys())
            
            for entry_id in entry_ids:
                entry = self._memories.get(entry_id)
                if not entry or entry.is_expired():
                    continue
                
                if entry.importance < min_importance:
                    continue
                
                if tags and not tags.issubset(entry.tags):
                    continue
                
                if query:
                    content_str = str(entry.content)
                    if query.lower() not in content_str.lower():
                        continue
                
                candidates.append(entry)
            
            # 按重要性和访问时间排序
            candidates.sort(
                key=lambda e: (e.importance, e.accessed_at),
                reverse=True
            )
            
            # 更新访问记录
            for entry in candidates[:limit]:
                entry.access()
            
            return candidates[:limit]
    
    def consolidate(self) -> int:
        """整合记忆
        
        将重要的短期记忆转移到长期记忆。
        返回转移的数量。
        """
        with self._lock:
            count = 0
            for entry_id in list(self._by_type[MemoryType.SHORT_TERM]):
                entry = self._memories.get(entry_id)
                if entry and entry.importance >= self.importance_threshold:
                    # 转移到长期记忆
                    self._by_type[MemoryType.SHORT_TERM].remove(entry_id)
                    self._by_type[MemoryType.LONG_TERM].append(entry_id)
                    entry.memory_type = MemoryType.LONG_TERM
                    count += 1
            return count
    
    def forget(self, entry_id: str) -> bool:
        """遗忘记忆"""
        with self._lock:
            entry = self._memories.pop(entry_id, None)
            if entry:
                self._by_type[entry.memory_type].remove(entry_id)
                for tag in entry.tags:
                    if entry_id in self._by_tag[tag]:
                        self._by_tag[tag].remove(entry_id)
                return True
            return False
    
    def _cleanup(self) -> None:
        """清理过期和超出容量的记忆"""
        # 清理过期记忆
        expired = [
            entry_id for entry_id, entry in self._memories.items()
            if entry.is_expired()
        ]
        for entry_id in expired:
            self.forget(entry_id)
        
        # 清理超出容量的短期记忆
        short_term_ids = self._by_type[MemoryType.SHORT_TERM]
        if len(short_term_ids) > self.short_term_capacity:
            # 按重要性和访问时间排序，移除最不重要的
            entries = [
                (entry_id, self._memories[entry_id])
                for entry_id in short_term_ids
                if entry_id in self._memories
            ]
            entries.sort(key=lambda x: (x[1].importance, x[1].accessed_at))
            
            to_remove = len(entries) - self.short_term_capacity
            for entry_id, _ in entries[:to_remove]:
                self.forget(entry_id)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取记忆摘要"""
        with self._lock:
            return {
                "total": len(self._memories),
                "by_type": {
                    mt.value: len(ids) for mt, ids in self._by_type.items()
                },
                "tags": list(self._by_tag.keys())
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化"""
        with self._lock:
            return {
                "memories": [entry.to_dict() for entry in self._memories.values()],
                "short_term_capacity": self.short_term_capacity,
                "long_term_capacity": self.long_term_capacity,
                "importance_threshold": self.importance_threshold
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMemory':
        """反序列化"""
        memory = cls(
            short_term_capacity=data.get('short_term_capacity', 50),
            long_term_capacity=data.get('long_term_capacity', 1000),
            importance_threshold=data.get('importance_threshold', 0.7)
        )
        
        for entry_data in data.get('memories', []):
            entry = MemoryEntry.from_dict(entry_data)
            memory._memories[entry.entry_id] = entry
            memory._by_type[entry.memory_type].append(entry.entry_id)
            for tag in entry.tags:
                memory._by_tag[tag].append(entry.entry_id)
        
        return memory


# =============================================================================
# 计划和反思
# =============================================================================

@dataclass
class PlanStep:
    """计划步骤"""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    action: str = ""
    dependencies: List[str] = field(default_factory=list)  # 依赖的步骤ID
    status: str = "pending"  # pending, running, completed, failed, skipped
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start(self) -> None:
        """开始执行"""
        self.status = "running"
        self.started_at = datetime.utcnow()
    
    def complete(self, result: Any = None) -> None:
        """完成执行"""
        self.status = "completed"
        self.result = result
        self.completed_at = datetime.utcnow()
    
    def fail(self, error: str) -> None:
        """执行失败"""
        self.status = "failed"
        self.error = error
        self.completed_at = datetime.utcnow()
    
    def skip(self, reason: str = "") -> None:
        """跳过执行"""
        self.status = "skipped"
        self.metadata["skip_reason"] = reason
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "action": self.action,
            "dependencies": self.dependencies,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanStep':
        return cls(
            step_id=data.get('step_id', str(uuid.uuid4())),
            description=data.get('description', ''),
            action=data.get('action', ''),
            dependencies=data.get('dependencies', []),
            status=data.get('status', 'pending'),
            result=data.get('result'),
            error=data.get('error'),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            metadata=data.get('metadata', {})
        )


@dataclass
class AgentPlan:
    """Agent 计划
    
    管理任务执行计划。
    """
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    steps: List[PlanStep] = field(default_factory=list)
    current_step_index: int = 0
    status: str = "pending"  # pending, running, completed, failed, cancelled
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, description: str, action: str = "", 
                 dependencies: List[str] = None) -> PlanStep:
        """添加步骤"""
        step = PlanStep(
            description=description,
            action=action,
            dependencies=dependencies or []
        )
        self.steps.append(step)
        self.updated_at = datetime.utcnow()
        return step
    
    def get_current_step(self) -> Optional[PlanStep]:
        """获取当前步骤"""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def advance(self) -> bool:
        """前进到下一步"""
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            self.updated_at = datetime.utcnow()
            return True
        return False
    
    def get_ready_steps(self) -> List[PlanStep]:
        """获取可以执行的步骤（依赖已完成）"""
        ready = []
        completed_ids = {s.step_id for s in self.steps if s.status == "completed"}
        
        for step in self.steps:
            if step.status != "pending":
                continue
            if all(dep in completed_ids for dep in step.dependencies):
                ready.append(step)
        
        return ready
    
    @property
    def progress(self) -> float:
        """计算进度"""
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status in ["completed", "skipped"])
        return completed / len(self.steps)
    
    @property
    def is_completed(self) -> bool:
        """检查是否完成"""
        return all(s.status in ["completed", "skipped"] for s in self.steps)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "goal": self.goal,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_index": self.current_step_index,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "progress": self.progress,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentPlan':
        return cls(
            plan_id=data.get('plan_id', str(uuid.uuid4())),
            goal=data.get('goal', ''),
            steps=[PlanStep.from_dict(s) for s in data.get('steps', [])],
            current_step_index=data.get('current_step_index', 0),
            status=data.get('status', 'pending'),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data['updated_at']) if 'updated_at' in data else datetime.utcnow(),
            metadata=data.get('metadata', {})
        )


@dataclass
class Reflection:
    """反思记录"""
    reflection_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    reflection_type: str = "general"  # general, error, success, learning
    trigger: str = ""  # 触发反思的事件
    insights: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: datetime = field(default_factory=datetime.utcnow)
    related_messages: List[str] = field(default_factory=list)  # 相关消息ID
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "reflection_id": self.reflection_id,
            "content": self.content,
            "reflection_type": self.reflection_type,
            "trigger": self.trigger,
            "insights": self.insights,
            "action_items": self.action_items,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "related_messages": self.related_messages,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Reflection':
        return cls(
            reflection_id=data.get('reflection_id', str(uuid.uuid4())),
            content=data.get('content', ''),
            reflection_type=data.get('reflection_type', 'general'),
            trigger=data.get('trigger', ''),
            insights=data.get('insights', []),
            action_items=data.get('action_items', []),
            confidence=data.get('confidence', 0.5),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.utcnow(),
            related_messages=data.get('related_messages', []),
            metadata=data.get('metadata', {})
        )


# =============================================================================
# Reducer 函数
# =============================================================================

def add_messages(left: List[AgentMessage], 
                 right: Union[AgentMessage, List[AgentMessage]]) -> List[AgentMessage]:
    """消息合并函数（Reducer）
    
    用于状态更新时合并消息列表。
    """
    if isinstance(right, AgentMessage):
        right = [right]
    return left + right


def replace_value(left: T, right: T) -> T:
    """替换值（Reducer）"""
    return right


def merge_dict(left: Dict, right: Dict) -> Dict:
    """合并字典（Reducer）"""
    result = copy.deepcopy(left)
    result.update(right)
    return result


def increment(left: int, right: int) -> int:
    """增量（Reducer）"""
    return left + right


def union_set(left: Set, right: Set) -> Set:
    """集合并集（Reducer）"""
    return left | right


# =============================================================================
# 核心状态类
# =============================================================================

@dataclass
class AgentState:
    """Agent 状态 - 生产级实现
    
    核心状态类，管理 Agent 执行过程中的所有状态信息。
    支持：
    - 消息历史与管理
    - 工具调用与结果
    - 版本控制与分支
    - 计划与反思
    - 记忆管理
    - 执行上下文
    - 事件系统
    - 并发控制
    """
    # 基础信息
    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    status: AgentStatus = AgentStatus.IDLE
    
    # 版本控制
    version: int = 1
    branch: str = "main"
    
    # 消息历史
    messages: List[AgentMessage] = field(default_factory=list)
    
    # 当前输入
    input: str = ""
    
    # 工具相关
    pending_tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    
    # 循环控制
    iteration: int = 0
    max_iterations: int = 10
    
    # 中间结果
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # 最终输出
    output: Optional[str] = None
    final_answer: Optional[str] = None
    
    # 错误信息
    error: Optional[str] = None
    error_type: ErrorType = ErrorType.NONE
    
    # 人机协作
    waiting_for_human: bool = False
    human_feedback: Optional[str] = None
    approval_required: bool = False
    approval_reason: Optional[str] = None
    
    # 中断控制
    interrupt_type: InterruptType = InterruptType.NONE
    interrupted_at_node: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # 计划（用于 Plan-and-Execute）
    plan: Optional[AgentPlan] = None
    current_step: int = 0
    step_results: Dict[int, Any] = field(default_factory=dict)
    
    # 反思（用于 Reflexion）
    reflections: List[Reflection] = field(default_factory=list)
    
    # 自定义数据
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 执行上下文
    context: Optional[ExecutionContext] = None
    
    # 记忆（延迟初始化）
    _memory: Optional[AgentMemory] = field(default=None, repr=False)
    
    # 状态历史（用于回滚）
    _history: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _max_history: int = field(default=50, repr=False)
    
    # 事件订阅者
    _subscribers: Dict[str, List[Callable]] = field(default_factory=dict, repr=False)
    
    # 并发锁
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    
    # 转换历史
    transitions: List[StateTransition] = field(default_factory=list)
    
    # 当前节点
    current_node: Optional[str] = None
    previous_node: Optional[str] = None
    
    # 令牌使用统计
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    # 成本追踪
    total_cost: float = 0.0
    
    @property
    def memory(self) -> AgentMemory:
        """获取记忆管理器（延迟初始化）"""
        if self._memory is None:
            self._memory = AgentMemory()
        return self._memory
    
    def _record_history(self) -> None:
        """记录状态历史"""
        if len(self._history) >= self._max_history:
            self._history.pop(0)
        
        snapshot = {
            'version': self.version,
            'timestamp': datetime.utcnow().isoformat(),
            'status': self.status.value,
            'iteration': self.iteration,
            'messages_count': len(self.messages),
            'output': self.output
        }
        self._history.append(snapshot)
    
    def _emit_event(self, event: str, data: Any = None) -> None:
        """发射事件"""
        for subscriber in self._subscribers.get(event, []):
            try:
                # 支持不同签名的回调函数
                sig = inspect.signature(subscriber)
                param_count = len(sig.parameters)
                
                if param_count == 0:
                    subscriber()
                elif param_count == 1:
                    subscriber(data)
                elif param_count == 2:
                    subscriber(event, data)
                else:
                    subscriber(self, event, data)
            except Exception as e:
                logger.error(f"Event subscriber error for {event}: {e}")
    
    def subscribe(self, event: str, callback: Callable) -> None:
        """订阅事件"""
        if event not in self._subscribers:
            self._subscribers[event] = []
        self._subscribers[event].append(callback)
    
    def unsubscribe(self, event: str, callback: Callable) -> None:
        """取消订阅"""
        if event in self._subscribers:
            self._subscribers[event] = [
                cb for cb in self._subscribers[event] if cb != callback
            ]
    
    # ==========================================================================
    # 消息管理
    # ==========================================================================
    
    def add_message(self, message: AgentMessage) -> None:
        """添加消息"""
        with self._lock:
            self.messages.append(message)
            self.updated_at = datetime.utcnow()
            self._emit_event('message_added', message)
    
    def add_messages(self, messages: List[AgentMessage]) -> None:
        """批量添加消息"""
        with self._lock:
            self.messages.extend(messages)
            self.updated_at = datetime.utcnow()
            self._emit_event('messages_added', messages)
    
    def get_last_message(self) -> Optional[AgentMessage]:
        """获取最后一条消息"""
        return self.messages[-1] if self.messages else None
    
    def get_messages_by_type(self, msg_type: MessageType) -> List[AgentMessage]:
        """获取指定类型的消息"""
        return [m for m in self.messages if m.type == msg_type]
    
    def get_conversation_history(self, 
                                  limit: int = None,
                                  include_system: bool = True,
                                  include_tool: bool = True) -> List[AgentMessage]:
        """获取对话历史"""
        messages = self.messages
        
        if not include_system:
            messages = [m for m in messages if m.type != MessageType.SYSTEM]
        if not include_tool:
            messages = [m for m in messages if m.type != MessageType.TOOL]
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def get_messages_for_llm(self, 
                             format: str = "openai",
                             max_tokens: int = None) -> List[Dict[str, Any]]:
        """获取用于 LLM 的消息格式"""
        messages = []
        
        for msg in self.messages:
            if msg.is_expired():
                continue
            
            if format == "openai":
                messages.append(msg.to_openai_format())
            elif format == "anthropic":
                messages.append(msg.to_anthropic_format())
            else:
                messages.append(msg.to_dict())
        
        return messages
    
    def clear_messages(self, keep_system: bool = True) -> None:
        """清空消息"""
        with self._lock:
            if keep_system:
                self.messages = [m for m in self.messages if m.type == MessageType.SYSTEM]
            else:
                self.messages = []
            self.updated_at = datetime.utcnow()
    
    def summarize_messages(self, max_length: int = 500) -> str:
        """摘要消息历史"""
        summaries = []
        for msg in self.messages[-10:]:  # 最近10条
            prefix = {
                MessageType.HUMAN: "User",
                MessageType.AI: "Assistant",
                MessageType.TOOL: "Tool",
                MessageType.SYSTEM: "System"
            }.get(msg.type, "Unknown")
            
            summaries.append(f"{prefix}: {msg.truncate(100)}")
        
        summary = "\n".join(summaries)
        if len(summary) > max_length:
            summary = summary[:max_length - 3] + "..."
        
        return summary
    
    # ==========================================================================
    # 工具调用管理
    # ==========================================================================
    
    def add_tool_call(self, tool_call: ToolCall) -> None:
        """添加待执行的工具调用"""
        with self._lock:
            self.pending_tool_calls.append(tool_call)
            self.status = AgentStatus.WAITING_TOOL
            self.updated_at = datetime.utcnow()
            self._emit_event('tool_call_added', tool_call)
    
    def add_tool_result(self, result: ToolResult) -> None:
        """添加工具执行结果"""
        with self._lock:
            self.tool_results.append(result)
            
        # 移除已完成的工具调用
        self.pending_tool_calls = [
            tc for tc in self.pending_tool_calls 
            if tc.id != result.tool_call_id
        ]
            
        if not self.pending_tool_calls:
            self.status = AgentStatus.RUNNING
            
            self.updated_at = datetime.utcnow()
            self._emit_event('tool_result_added', result)
            
            # 更新上下文指标
            if self.context:
                self.context.metrics.total_tool_calls += 1
                if result.success:
                    self.context.metrics.successful_tool_calls += 1
                else:
                    self.context.metrics.failed_tool_calls += 1
    
    def get_pending_tool_calls(self, 
                                priority: PriorityLevel = None) -> List[ToolCall]:
        """获取待执行的工具调用"""
        calls = self.pending_tool_calls
        if priority:
            calls = [tc for tc in calls if tc.priority == priority]
        return sorted(calls, key=lambda tc: tc.priority.value)
    
    def get_tool_result(self, tool_call_id: str) -> Optional[ToolResult]:
        """获取工具执行结果"""
        for result in self.tool_results:
            if result.tool_call_id == tool_call_id:
                return result
        return None
    
    def has_pending_tool_calls(self) -> bool:
        """是否有待执行的工具调用"""
        return len(self.pending_tool_calls) > 0
    
    # ==========================================================================
    # 迭代控制
    # ==========================================================================
    
    def add_intermediate_step(self, action: Dict[str, Any], observation: Any) -> None:
        """添加中间步骤"""
        with self._lock:
            self.intermediate_steps.append({
                "action": action,
                "observation": observation,
                "iteration": self.iteration,
                "timestamp": datetime.utcnow().isoformat()
            })
            self.updated_at = datetime.utcnow()
    
    def increment_iteration(self) -> bool:
        """增加迭代计数，返回是否可以继续"""
        with self._lock:
            self._record_history()
            self.iteration += 1
            self.version += 1
            self.updated_at = datetime.utcnow()
            self._emit_event('iteration_incremented', self.iteration)
        return self.iteration < self.max_iterations
    
    def should_continue(self) -> bool:
        """判断是否应该继续执行"""
        if self.status in [AgentStatus.COMPLETED, AgentStatus.FAILED, 
                          AgentStatus.CANCELLED, AgentStatus.TIMEOUT]:
            return False
        if self.iteration >= self.max_iterations:
            return False
        if self.final_answer is not None:
            return False
        if self.interrupt_type != InterruptType.NONE:
            return False
        return True
    
    # ==========================================================================
    # 状态设置
    # ==========================================================================
    
    def set_final_answer(self, answer: str) -> None:
        """设置最终答案"""
        with self._lock:
            self._record_history()
            self.final_answer = answer
            self.output = answer
            self.status = AgentStatus.COMPLETED
            self.updated_at = datetime.utcnow()
            self._emit_event('completed', answer)
    
    def set_error(self, error: str, error_type: ErrorType = ErrorType.INTERNAL) -> None:
        """设置错误"""
        with self._lock:
            self._record_history()
            self.error = error
            self.error_type = error_type
            self.status = AgentStatus.FAILED
            self.updated_at = datetime.utcnow()
            self._emit_event('error', {'error': error, 'type': error_type})
    
    def set_timeout(self) -> None:
        """设置超时"""
        with self._lock:
            self.status = AgentStatus.TIMEOUT
            self.error_type = ErrorType.TIMEOUT
            self.error = "Execution timed out"
            self.updated_at = datetime.utcnow()
            self._emit_event('timeout', None)
    
    def cancel(self, reason: str = "") -> None:
        """取消执行"""
        with self._lock:
            self._record_history()
            self.status = AgentStatus.CANCELLED
            self.error_type = ErrorType.USER_CANCELLED
            self.error = reason or "Cancelled by user"
            self.updated_at = datetime.utcnow()
            self._emit_event('cancelled', reason)
    
    def pause(self) -> None:
        """暂停执行"""
        with self._lock:
            self.status = AgentStatus.PAUSED
            self.updated_at = datetime.utcnow()
            self._emit_event('paused', None)
    
    def resume(self) -> None:
        """恢复执行"""
        with self._lock:
            if self.status == AgentStatus.PAUSED:
                self.status = AgentStatus.RUNNING
                self.updated_at = datetime.utcnow()
                self._emit_event('resumed', None)
    
    # ==========================================================================
    # 人机协作
    # ==========================================================================
    
    def request_human_input(self, prompt: str = None) -> None:
        """请求人工输入"""
        with self._lock:
            self.waiting_for_human = True
            self.status = AgentStatus.WAITING_HUMAN
            self.interrupt_type = InterruptType.ON_HUMAN_REQUEST
        if prompt:
            self.add_message(AgentMessage.ai(prompt))
            self.updated_at = datetime.utcnow()
            self._emit_event('human_input_requested', prompt)
    
    def receive_human_input(self, feedback: str) -> None:
        """接收人工输入"""
        with self._lock:
            self.human_feedback = feedback
            self.waiting_for_human = False
            self.status = AgentStatus.RUNNING
            self.interrupt_type = InterruptType.NONE
            self.add_message(AgentMessage.human(feedback))
            self.updated_at = datetime.utcnow()
            self._emit_event('human_input_received', feedback)
    
    def request_approval(self, reason: str, action: Dict[str, Any] = None) -> None:
        """请求审批"""
        with self._lock:
            self.approval_required = True
            self.approval_reason = reason
            self.status = AgentStatus.WAITING_APPROVAL
            self.interrupt_type = InterruptType.ON_APPROVAL_NEEDED
            self.metadata['pending_action'] = action
            self.updated_at = datetime.utcnow()
            self._emit_event('approval_requested', {'reason': reason, 'action': action})
    
    def approve(self) -> Dict[str, Any]:
        """批准并返回待执行的动作"""
        with self._lock:
            action = self.metadata.pop('pending_action', None)
            self.approval_required = False
            self.approval_reason = None
            self.status = AgentStatus.RUNNING
            self.interrupt_type = InterruptType.NONE
            self.updated_at = datetime.utcnow()
            self._emit_event('approved', action)
            return action
    
    def reject(self, reason: str = "") -> None:
        """拒绝"""
        with self._lock:
            self.metadata.pop('pending_action', None)
            self.approval_required = False
            self.approval_reason = None
            self.status = AgentStatus.RUNNING
            self.interrupt_type = InterruptType.NONE
            self.updated_at = datetime.utcnow()
            self._emit_event('rejected', reason)
    
    # ==========================================================================
    # 中断控制
    # ==========================================================================
    
    def set_interrupt(self, interrupt_type: InterruptType, node: str = None) -> None:
        """设置中断"""
        with self._lock:
            self.interrupt_type = interrupt_type
            self.interrupted_at_node = node
            self.updated_at = datetime.utcnow()
            self._emit_event('interrupted', {'type': interrupt_type, 'node': node})
    
    def clear_interrupt(self) -> None:
        """清除中断"""
        with self._lock:
            self.interrupt_type = InterruptType.NONE
            self.interrupted_at_node = None
            self.updated_at = datetime.utcnow()
    
    def is_interrupted(self) -> bool:
        """检查是否被中断"""
        return self.interrupt_type != InterruptType.NONE
    
    # ==========================================================================
    # 节点转换
    # ==========================================================================
    
    def transition_to(self, node: str, trigger: str = "") -> None:
        """转换到新节点"""
        with self._lock:
            transition = StateTransition(
                from_node=self.current_node or "",
                to_node=node,
                trigger=trigger
            )
            self.transitions.append(transition)
            self.previous_node = self.current_node
            self.current_node = node
            self.updated_at = datetime.utcnow()
            self._emit_event('node_transition', transition)
    
    def get_transition_history(self, limit: int = None) -> List[StateTransition]:
        """获取转换历史"""
        transitions = self.transitions
        if limit:
            transitions = transitions[-limit:]
        return transitions
    
    # ==========================================================================
    # 计划管理
    # ==========================================================================
    
    def create_plan(self, goal: str, steps: List[str] = None) -> AgentPlan:
        """创建计划"""
        with self._lock:
            self.plan = AgentPlan(goal=goal)
            if steps:
                for step_desc in steps:
                    self.plan.add_step(step_desc)
            self.updated_at = datetime.utcnow()
            self._emit_event('plan_created', self.plan)
            return self.plan
    
    def advance_plan(self) -> bool:
        """推进计划"""
        if self.plan:
            return self.plan.advance()
        return False
    
    def get_current_plan_step(self) -> Optional[PlanStep]:
        """获取当前计划步骤"""
        if self.plan:
            return self.plan.get_current_step()
        return None
    
    # ==========================================================================
    # 反思管理
    # ==========================================================================
    
    def add_reflection(self, content: str, 
                       reflection_type: str = "general",
                       trigger: str = "",
                       insights: List[str] = None,
                       action_items: List[str] = None) -> Reflection:
        """添加反思"""
        with self._lock:
            reflection = Reflection(
                content=content,
                reflection_type=reflection_type,
                trigger=trigger,
                insights=insights or [],
                action_items=action_items or []
            )
            self.reflections.append(reflection)
            self.updated_at = datetime.utcnow()
            self._emit_event('reflection_added', reflection)
            return reflection
    
    def get_recent_reflections(self, limit: int = 5) -> List[Reflection]:
        """获取最近的反思"""
        return self.reflections[-limit:]
    
    # ==========================================================================
    # 令牌和成本追踪
    # ==========================================================================
    
    def add_token_usage(self, prompt: int, completion: int, cost: float = 0.0) -> None:
        """添加令牌使用"""
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += prompt + completion
            self.total_cost += cost
            self.updated_at = datetime.utcnow()
    
    def get_token_summary(self) -> Dict[str, Any]:
        """获取令牌使用摘要"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost
        }
    
    # ==========================================================================
    # 版本控制
    # ==========================================================================
    
    def create_checkpoint(self, description: str = "") -> StateCheckpoint:
        """创建检查点"""
        with self._lock:
            checkpoint = StateCheckpoint(
                checkpoint_id=str(uuid.uuid4()),
                thread_id=self.thread_id,
                state=self.to_dict(),
                version=self.version,
                branch=self.branch,
                description=description
            )
            return checkpoint
    
    def restore_from_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """从检查点恢复"""
        with self._lock:
            if checkpoint.compressed:
                checkpoint = checkpoint.decompress()
            
            restored = AgentState.from_dict(checkpoint.state)
            
            # 复制所有字段
            for field_name in vars(restored):
                if not field_name.startswith('_'):
                    setattr(self, field_name, getattr(restored, field_name))

            self._emit_event('checkpoint_restored', checkpoint)

    
    def create_branch(self, branch_name: str) -> 'AgentState':
        """创建分支"""
        with self._lock:
            new_state = self.copy()
            new_state.branch = branch_name
            new_state.version = 1
            new_state._emit_event('branch_created', branch_name)
            return new_state
    
    def rollback(self, steps: int = 1) -> bool:
        """回滚到之前的状态"""
        with self._lock:
            if steps > len(self._history):
                return False
            
            # 获取目标历史记录
            target = self._history[-(steps + 1)] if steps < len(self._history) else self._history[0]
            
            # 这里只回滚基本信息
            self.version = target.get('version', self.version - steps)
            self.iteration = target.get('iteration', self.iteration - steps)
            
            self._emit_event('rolled_back', steps)
            return True
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取状态历史"""
        return list(self._history)
    
    # ==========================================================================
    # 序列化
    # ==========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "thread_id": self.thread_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "version": self.version,
            "branch": self.branch,
            "messages": [m.to_dict() for m in self.messages],
            "input": self.input,
            "pending_tool_calls": [tc.to_dict() for tc in self.pending_tool_calls],
            "tool_results": [tr.to_dict() for tr in self.tool_results],
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "intermediate_steps": self.intermediate_steps,
            "output": self.output,
            "final_answer": self.final_answer,
            "error": self.error,
            "error_type": self.error_type.value,
            "waiting_for_human": self.waiting_for_human,
            "human_feedback": self.human_feedback,
            "approval_required": self.approval_required,
            "approval_reason": self.approval_reason,
            "interrupt_type": self.interrupt_type.value,
            "interrupted_at_node": self.interrupted_at_node,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "plan": self.plan.to_dict() if self.plan else None,
            "current_step": self.current_step,
            "step_results": self.step_results,
            "reflections": [r.to_dict() for r in self.reflections],
            "data": self.data,
            "context": self.context.to_dict() if self.context else None,
            "memory": self._memory.to_dict() if self._memory else None,
            "transitions": [t.to_dict() for t in self.transitions],
            "current_node": self.current_node,
            "previous_node": self.previous_node,
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost": self.total_cost
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        """从字典创建"""
        messages = [AgentMessage.from_dict(m) for m in data.get('messages', [])]
        pending_tool_calls = [ToolCall.from_dict(tc) for tc in data.get('pending_tool_calls', [])]
        tool_results = [ToolResult.from_dict(tr) for tr in data.get('tool_results', [])]
        reflections = [Reflection.from_dict(r) for r in data.get('reflections', [])]
        transitions = [
            StateTransition(
                transition_id=t.get('transition_id', str(uuid.uuid4())),
                from_node=t.get('from_node', ''),
                to_node=t.get('to_node', ''),
                trigger=t.get('trigger', ''),
                timestamp=datetime.fromisoformat(t['timestamp']) if 'timestamp' in t else datetime.utcnow(),
                duration_ms=t.get('duration_ms', 0.0),
                metadata=t.get('metadata', {})
            )
            for t in data.get('transitions', [])
        ]
        
        state = cls(
            thread_id=data.get('thread_id', str(uuid.uuid4())),
            agent_id=data.get('agent_id', ''),
            status=AgentStatus(data.get('status', 'idle')),
            version=data.get('version', 1),
            branch=data.get('branch', 'main'),
            messages=messages,
            input=data.get('input', ''),
            pending_tool_calls=pending_tool_calls,
            tool_results=tool_results,
            iteration=data.get('iteration', 0),
            max_iterations=data.get('max_iterations', 10),
            intermediate_steps=data.get('intermediate_steps', []),
            output=data.get('output'),
            final_answer=data.get('final_answer'),
            error=data.get('error'),
            error_type=ErrorType(data.get('error_type', 'none')),
            waiting_for_human=data.get('waiting_for_human', False),
            human_feedback=data.get('human_feedback'),
            approval_required=data.get('approval_required', False),
            approval_reason=data.get('approval_reason'),
            interrupt_type=InterruptType(data.get('interrupt_type', 'none')),
            interrupted_at_node=data.get('interrupted_at_node'),
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.utcnow(),
            updated_at=datetime.fromisoformat(data['updated_at']) if 'updated_at' in data else datetime.utcnow(),
            plan=AgentPlan.from_dict(data['plan']) if data.get('plan') else None,
            current_step=data.get('current_step', 0),
            step_results=data.get('step_results', {}),
            reflections=reflections,
            data=data.get('data', {}),
            transitions=transitions,
            current_node=data.get('current_node'),
            previous_node=data.get('previous_node'),
            total_tokens=data.get('total_tokens', 0),
            prompt_tokens=data.get('prompt_tokens', 0),
            completion_tokens=data.get('completion_tokens', 0),
            total_cost=data.get('total_cost', 0.0)
        )
        
        # 恢复记忆
        if data.get('memory'):
            state._memory = AgentMemory.from_dict(data['memory'])
        
        return state
    
    def copy(self) -> 'AgentState':
        """深拷贝"""
        return AgentState.from_dict(self.to_dict())
    
    def update(self, **kwargs) -> 'AgentState':
        """更新状态并返回新实例"""
        new_state = self.copy()
        for key, value in kwargs.items():
            if hasattr(new_state, key):
                setattr(new_state, key, value)
        new_state.updated_at = datetime.utcnow()
        return new_state

    def __str__(self) -> str:
        return (f"AgentState(thread_id={self.thread_id}, "
                f"status={self.status.value}, "
                f"iteration={self.iteration}/{self.max_iterations}, "
                f"messages={len(self.messages)})")
    
    def __repr__(self) -> str:
        return self.__str__()


# =============================================================================
# 状态归约器
# =============================================================================

class StateReducer:
    """状态归约器
    
    定义如何合并状态更新。
    """
    
    @staticmethod
    def messages_reducer(current: List[AgentMessage], 
                        update: Union[AgentMessage, List[AgentMessage]]) -> List[AgentMessage]:
        """消息列表归约器：追加新消息"""
        if isinstance(update, AgentMessage):
            update = [update]
        return current + update
    
    @staticmethod
    def replace_reducer(current: Any, update: Any) -> Any:
        """替换归约器：直接替换"""
        return update
    
    @staticmethod
    def merge_dict_reducer(current: Dict, update: Dict) -> Dict:
        """字典合并归约器"""
        result = current.copy()
        result.update(update)
        return result
    
    @staticmethod
    def increment_reducer(current: int, update: int) -> int:
        """增量归约器"""
        return current + update

    @staticmethod
    def union_reducer(current: Set, update: Set) -> Set:
        """集合并集归约器"""
        return current | update
    
    @staticmethod
    def max_reducer(current: Any, update: Any) -> Any:
        """取最大值归约器"""
        return max(current, update)
    
    @staticmethod
    def min_reducer(current: Any, update: Any) -> Any:
        """取最小值归约器"""
        return min(current, update)
    
    @staticmethod
    def first_reducer(current: Any, update: Any) -> Any:
        """保留第一个值归约器"""
        return current if current is not None else update
    
    @staticmethod
    def last_reducer(current: Any, update: Any) -> Any:
        """保留最后一个值归约器"""
        return update if update is not None else current


# =============================================================================
# 状态通道
# =============================================================================

class Channel(ABC, Generic[T]):
    """状态通道基类
    
    定义状态字段如何被更新。
    """
    
    @abstractmethod
    def update(self, current: T, new: T) -> T:
        """更新通道值"""
        pass
    
    @abstractmethod
    def get_default(self) -> T:
        """获取默认值"""
        pass


class LastValue(Channel[T]):
    """最后值通道
    
    总是使用最新的值。
    """
    
    def __init__(self, default: T = None):
        self._default = default
    
    def update(self, current: T, new: T) -> T:
        return new if new is not None else current
    
    def get_default(self) -> T:
        return self._default


class AppendChannel(Channel[List[T]]):
    """追加通道
    
    将新值追加到列表。
    """
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size
    
    def update(self, current: List[T], new: Union[T, List[T]]) -> List[T]:
        if not isinstance(new, list):
            new = [new]
        result = current + new
        if self.max_size and len(result) > self.max_size:
            result = result[-self.max_size:]
        return result
    
    def get_default(self) -> List[T]:
        return []


class BinaryOperatorChannel(Channel[T]):
    """二元操作通道
    
    使用自定义二元操作符合并值。
    """
    
    def __init__(self, operator: Callable[[T, T], T], default: T = None):
        self.operator = operator
        self._default = default
    
    def update(self, current: T, new: T) -> T:
        if current is None:
            return new
        if new is None:
            return current
        return self.operator(current, new)
    
    def get_default(self) -> T:
        return self._default


class ContextChannel(Channel[Dict[str, Any]]):
    """上下文通道
    
    深度合并字典。
    """
    
    def __init__(self, default: Dict[str, Any] = None):
        self._default = default or {}
    
    def update(self, current: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        result = copy.deepcopy(current)
        self._deep_merge(result, new)
        return result
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """深度合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
    
    def get_default(self) -> Dict[str, Any]:
        return copy.deepcopy(self._default)


# =============================================================================
# 状态管理器
# =============================================================================

class StateManager:
    """状态管理器
    
    提供状态的高级管理功能，包括：
    - 版本控制
    - 分支管理
    - 回滚
    - 压缩
    - 验证
    """
    
    def __init__(self, max_checkpoints: int = 100):
        self._checkpoints: Dict[str, StateCheckpoint] = {}
        self._branches: Dict[str, StateBranch] = {"main": StateBranch(name="main")}
        self._current_branch = "main"
        self.max_checkpoints = max_checkpoints
        self._lock = threading.Lock()
    
    def save_checkpoint(self, state: AgentState, 
                        description: str = "",
                        tags: List[str] = None) -> StateCheckpoint:
        """保存检查点"""
        with self._lock:
            checkpoint = state.create_checkpoint(description)
            checkpoint.tags = tags or []
            checkpoint.branch = self._current_branch
            
            # 设置父检查点
            branch = self._branches[self._current_branch]
            if branch.base_checkpoint_id:
                checkpoint.parent_id = branch.base_checkpoint_id
            
            self._checkpoints[checkpoint.checkpoint_id] = checkpoint
            branch.base_checkpoint_id = checkpoint.checkpoint_id
            
            # 清理旧检查点
            self._cleanup_checkpoints()
            
            logger.debug(f"Saved checkpoint: {checkpoint.checkpoint_id}")
            return checkpoint
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[StateCheckpoint]:
        """加载检查点"""
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if checkpoint and checkpoint.compressed:
                return checkpoint.decompress()
            return checkpoint
    
    def get_checkpoint_by_tag(self, tag: str) -> Optional[StateCheckpoint]:
        """按标签获取检查点"""
        with self._lock:
            for checkpoint in self._checkpoints.values():
                if tag in checkpoint.tags:
                    if checkpoint.compressed:
                        return checkpoint.decompress()
                    return checkpoint
            return None
    
    def list_checkpoints(self, branch: str = None, 
                         limit: int = 10) -> List[StateCheckpoint]:
        """列出检查点"""
        with self._lock:
            checkpoints = list(self._checkpoints.values())
            
            if branch:
                checkpoints = [c for c in checkpoints if c.branch == branch]
            
            checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
            return checkpoints[:limit]
    
    def create_branch(self, name: str, 
                      base_checkpoint_id: str = None) -> StateBranch:
        """创建分支"""
        with self._lock:
            if name in self._branches:
                raise ValueError(f"Branch '{name}' already exists")
            
            branch = StateBranch(
                name=name,
                parent_branch=self._current_branch,
                base_checkpoint_id=base_checkpoint_id or self._branches[self._current_branch].base_checkpoint_id
            )
            self._branches[name] = branch
            
            logger.debug(f"Created branch: {name}")
            return branch
    
    def switch_branch(self, name: str) -> StateBranch:
        """切换分支"""
        with self._lock:
            if name not in self._branches:
                raise ValueError(f"Branch '{name}' does not exist")
            
            self._current_branch = name
            return self._branches[name]
    
    def merge_branches(self, source: str, target: str,
                       strategy: str = "replace") -> bool:
        """合并分支"""
        with self._lock:
            if source not in self._branches or target not in self._branches:
                return False
            
            source_branch = self._branches[source]
            target_branch = self._branches[target]
            
            if not source_branch.base_checkpoint_id:
                return False
            
            source_checkpoint = self._checkpoints.get(source_branch.base_checkpoint_id)
            
            if strategy == "replace":
                target_branch.base_checkpoint_id = source_checkpoint.checkpoint_id
            
            logger.debug(f"Merged branch '{source}' into '{target}'")
            return True
    
    def delete_branch(self, name: str) -> bool:
        """删除分支"""
        with self._lock:
            if name == "main":
                return False
            if name not in self._branches:
                return False
            if name == self._current_branch:
                self._current_branch = "main"
            
            del self._branches[name]
            return True
    
    def get_diff(self, checkpoint_id_1: str, 
                 checkpoint_id_2: str) -> Optional[StateDiff]:
        """获取两个检查点之间的差异"""
        with self._lock:
            cp1 = self._checkpoints.get(checkpoint_id_1)
            cp2 = self._checkpoints.get(checkpoint_id_2)
            
            if not cp1 or not cp2:
                return None
            
            if cp1.compressed:
                cp1 = cp1.decompress()
            if cp2.compressed:
                cp2 = cp2.decompress()
            
            return StateDiff.compute(
                cp1.state, cp2.state,
                cp1.version, cp2.version
            )
    
    def compress_checkpoint(self, checkpoint_id: str) -> bool:
        """压缩检查点"""
        with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if checkpoint and not checkpoint.compressed:
                self._checkpoints[checkpoint_id] = checkpoint.compress()
                return True
            return False
    
    def compress_all(self) -> int:
        """压缩所有检查点"""
        count = 0
        with self._lock:
            for checkpoint_id in list(self._checkpoints.keys()):
                if self.compress_checkpoint(checkpoint_id):
                    count += 1
        return count
    
    def _cleanup_checkpoints(self) -> None:
        """清理旧检查点"""
        if len(self._checkpoints) <= self.max_checkpoints:
            return
        
        # 按时间排序，保留最新的
        checkpoints = sorted(
            self._checkpoints.values(),
            key=lambda c: c.timestamp
        )
        
        to_remove = len(checkpoints) - self.max_checkpoints
        for checkpoint in checkpoints[:to_remove]:
            # 不删除当前分支的基础检查点
            is_base = any(
                b.base_checkpoint_id == checkpoint.checkpoint_id
                for b in self._branches.values()
            )
            if not is_base:
                del self._checkpoints[checkpoint.checkpoint_id]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取管理器摘要"""
        with self._lock:
            return {
                "total_checkpoints": len(self._checkpoints),
                "branches": list(self._branches.keys()),
                "current_branch": self._current_branch,
                "compressed_count": sum(1 for c in self._checkpoints.values() if c.compressed)
            }


# =============================================================================
# 状态验证器
# =============================================================================

class StateValidator:
    """状态验证器
    
    验证状态的有效性。
    """
    
    def __init__(self):
        self._validators: List[Callable[[AgentState], Tuple[bool, str]]] = []
    
    def add_validator(self, validator: Callable[[AgentState], Tuple[bool, str]]) -> None:
        """添加验证器"""
        self._validators.append(validator)
    
    def validate(self, state: AgentState) -> Tuple[bool, List[str]]:
        """验证状态"""
        errors = []
        
        for validator in self._validators:
            try:
                valid, message = validator(state)
                if not valid:
                    errors.append(message)
            except Exception as e:
                errors.append(f"Validator error: {str(e)}")
        
        # 内置验证
        builtin_errors = self._builtin_validation(state)
        errors.extend(builtin_errors)
        
        return len(errors) == 0, errors
    
    def _builtin_validation(self, state: AgentState) -> List[str]:
        """内置验证"""
        errors = []
        
        # 验证迭代次数
        if state.iteration < 0:
            errors.append("Iteration count cannot be negative")
        
        if state.iteration > state.max_iterations:
            errors.append(f"Iteration {state.iteration} exceeds max {state.max_iterations}")
        
        # 验证状态一致性
        if state.status == AgentStatus.COMPLETED and state.final_answer is None:
            errors.append("Completed state must have final answer")
        
        if state.status == AgentStatus.FAILED and state.error is None:
            errors.append("Failed state must have error message")
        
        # 验证工具调用一致性
        if state.status == AgentStatus.WAITING_TOOL and not state.pending_tool_calls:
            errors.append("Waiting for tool but no pending tool calls")
        
        return errors


# =============================================================================
# 状态序列化器
# =============================================================================

class StateSerializer:
    """状态序列化器
    
    提供多种序列化格式支持。
    """
    
    @staticmethod
    def to_json(state: AgentState, indent: int = None) -> str:
        """序列化为 JSON"""
        return json.dumps(state.to_dict(), indent=indent, default=str)
    
    @staticmethod
    def from_json(json_str: str) -> AgentState:
        """从 JSON 反序列化"""
        data = json.loads(json_str)
        return AgentState.from_dict(data)
    
    @staticmethod
    def to_compressed(state: AgentState) -> bytes:
        """序列化并压缩"""
        json_str = StateSerializer.to_json(state)
        return zlib.compress(json_str.encode())
    
    @staticmethod
    def from_compressed(data: bytes) -> AgentState:
        """从压缩数据反序列化"""
        json_str = zlib.decompress(data).decode()
        return StateSerializer.from_json(json_str)
    
    @staticmethod
    def to_base64(state: AgentState) -> str:
        """序列化为 Base64"""
        compressed = StateSerializer.to_compressed(state)
        return base64.b64encode(compressed).decode()
    
    @staticmethod
    def from_base64(data: str) -> AgentState:
        """从 Base64 反序列化"""
        compressed = base64.b64decode(data)
        return StateSerializer.from_compressed(compressed)


# =============================================================================
# 状态事件发射器
# =============================================================================

class StateEventEmitter:
    """状态事件发射器
    
    管理状态变化的事件通知。
    """
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def on(self, event: str, callback: Callable) -> None:
        """注册事件监听器"""
        with self._lock:
            self._listeners[event].append(callback)
    
    def off(self, event: str, callback: Callable = None) -> None:
        """移除事件监听器"""
        with self._lock:
            if callback:
                self._listeners[event] = [
                    cb for cb in self._listeners[event] if cb != callback
                ]
            else:
                self._listeners[event] = []
    
    def emit(self, event: str, data: Any = None) -> None:
        """发射事件"""
        with self._lock:
            listeners = list(self._listeners.get(event, []))
        
        for listener in listeners:
            try:
                listener(data)
            except Exception as e:
                logger.error(f"Event listener error for '{event}': {e}")
    
    def once(self, event: str, callback: Callable) -> None:
        """注册一次性事件监听器"""
        def wrapper(data):
            callback(data)
            self.off(event, wrapper)
        self.on(event, wrapper)


# =============================================================================
# 状态类型注解
# =============================================================================

# 带 Reducer 的消息列表类型
Messages = Annotated[List[AgentMessage], add_messages]

# 带 Reducer 的字典类型
MergedDict = Annotated[Dict[str, Any], merge_dict]

# 带 Reducer 的计数器类型
Counter = Annotated[int, increment]


# =============================================================================
# 工厂函数
# =============================================================================

def create_state(input: str = "",
                 agent_id: str = "",
                 max_iterations: int = 10,
                 system_prompt: str = None,
                 metadata: Dict[str, Any] = None) -> AgentState:
    """创建新状态的工厂函数"""
    state = AgentState(
        agent_id=agent_id,
        input=input,
        max_iterations=max_iterations,
        metadata=metadata or {}
    )
    
    if system_prompt:
        state.add_message(AgentMessage.system(system_prompt))
    
    if input:
        state.add_message(AgentMessage.human(input))
    
    return state


def create_execution_context(mode: ExecutionMode = ExecutionMode.SYNC,
                              timeout: float = None,
                              max_iterations: int = 10,
                              interrupt_before: List[str] = None,
                              interrupt_after: List[str] = None,
                              recursion_limit: int = 25,
                              config: Dict[str, Any] = None) -> ExecutionContext:
    """创建执行上下文的工厂函数"""
    return ExecutionContext(
        mode=mode,
        timeout=timeout,
        max_iterations=max_iterations,
        interrupt_before=interrupt_before or [],
        interrupt_after=interrupt_after or [],
        recursion_limit=recursion_limit,
        config=config or {}
    )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 枚举
    'MessageType',
    'AgentStatus',
    'ExecutionMode',
    'InterruptType',
    'ErrorType',
    'ContextType',
    'MemoryType',
    'PriorityLevel',
    'ChannelType',
    
    # 核心数据类
    'AgentMessage',
    'ToolCall',
    'ToolResult',
    'StateCheckpoint',
    'StateDiff',
    'StateTransition',
    'StateBranch',
    
    # 执行相关
    'ExecutionMetrics',
    'ExecutionContext',
    
    # 记忆相关
    'MemoryEntry',
    'AgentMemory',
    
    # 计划和反思
    'PlanStep',
    'AgentPlan',
    'Reflection',
    
    # 核心状态类
    'AgentState',
    
    # 归约器
    'StateReducer',
    'add_messages',
    'replace_value',
    'merge_dict',
    'increment',
    'union_set',
    
    # 通道
    'Channel',
    'LastValue',
    'AppendChannel',
    'BinaryOperatorChannel',
    'ContextChannel',
    
    # 管理工具
    'StateManager',
    'StateValidator',
    'StateSerializer',
    'StateEventEmitter',
    'MessageBuffer',
    
    # 类型注解
    'Messages',
    'MergedDict',
    'Counter',
    
    # 工厂函数
    'create_state',
    'create_execution_context',
]

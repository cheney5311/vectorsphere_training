"""
状态检查点

生产级的状态持久化和恢复能力：
- Checkpointer: 检查点基类（增强版）
- MemoryCheckpointer: 内存检查点（开发/测试）
- RedisCheckpointer: Redis 检查点（分布式生产环境）
- SQLiteCheckpointer: SQLite 检查点（单机生产环境）
- FileCheckpointer: 文件系统检查点（简单持久化）
- PostgresCheckpointer: PostgreSQL 检查点（企业级生产环境）

生产级特性：
- 数据压缩（gzip）
- 增量检查点（delta）
- 检查点标签和搜索
- 检查点分支管理
- 差异比较
- 回滚支持
- 异步操作
- 批量处理
- 数据校验
- 性能指标
"""

import json
import uuid
import gzip
import hashlib
import logging
import os
import time
import threading
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from functools import wraps
from enum import Enum
import base64
import copy

from .state import AgentState, StateCheckpoint

logger = logging.getLogger(__name__)


# ==================== 类型定义 ====================

class CheckpointStatus(Enum):
    """检查点状态"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    CORRUPTED = "corrupted"


class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"  # 需要 lz4 库


@dataclass
class CheckpointMetrics:
    """检查点性能指标"""
    total_saves: int = 0
    total_loads: int = 0
    total_deletes: int = 0
    total_bytes_saved: int = 0
    total_bytes_loaded: int = 0
    avg_save_time_ms: float = 0.0
    avg_load_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    compression_ratio: float = 1.0
    last_save_time: Optional[datetime] = None
    last_load_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_saves": self.total_saves,
            "total_loads": self.total_loads,
            "total_deletes": self.total_deletes,
            "total_bytes_saved": self.total_bytes_saved,
            "total_bytes_loaded": self.total_bytes_loaded,
            "avg_save_time_ms": round(self.avg_save_time_ms, 2),
            "avg_load_time_ms": round(self.avg_load_time_ms, 2),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "compression_ratio": round(self.compression_ratio, 3),
            "last_save_time": self.last_save_time.isoformat() if self.last_save_time else None,
            "last_load_time": self.last_load_time.isoformat() if self.last_load_time else None
        }


@dataclass
class CheckpointTag:
    """检查点标签"""
    name: str
    checkpoint_id: str
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "checkpoint_id": self.checkpoint_id,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointTag':
        return cls(
            name=data['name'],
            checkpoint_id=data['checkpoint_id'],
            description=data.get('description', ''),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else datetime.utcnow(),
            metadata=data.get('metadata', {})
        )


@dataclass
class CheckpointDiff:
    """检查点差异"""
    from_checkpoint_id: str
    to_checkpoint_id: str
    added_keys: List[str] = field(default_factory=list)
    removed_keys: List[str] = field(default_factory=list)
    modified_keys: List[str] = field(default_factory=list)
    message_count_diff: int = 0
    iteration_diff: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_checkpoint_id": self.from_checkpoint_id,
            "to_checkpoint_id": self.to_checkpoint_id,
            "added_keys": self.added_keys,
            "removed_keys": self.removed_keys,
            "modified_keys": self.modified_keys,
            "message_count_diff": self.message_count_diff,
            "iteration_diff": self.iteration_diff
        }


@dataclass
class EnhancedCheckpoint(StateCheckpoint):
    """增强版检查点
    
    在基础检查点上增加生产级功能。
    """
    version: int = 1
    checksum: str = ""
    compressed: bool = False
    compression_type: str = "none"
    original_size: int = 0
    compressed_size: int = 0
    tags: List[str] = field(default_factory=list)
    branch: str = "main"
    status: str = "active"
    ttl: Optional[int] = None  # 过期时间（秒）
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            "version": self.version,
            "checksum": self.checksum,
            "compressed": self.compressed,
            "compression_type": self.compression_type,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "tags": self.tags,
            "branch": self.branch,
            "status": self.status,
            "ttl": self.ttl
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedCheckpoint':
        return cls(
            checkpoint_id=data['checkpoint_id'],
            thread_id=data['thread_id'],
            state=data['state'],
            parent_id=data.get('parent_id'),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.utcnow(),
            metadata=data.get('metadata', {}),
            version=data.get('version', 1),
            checksum=data.get('checksum', ''),
            compressed=data.get('compressed', False),
            compression_type=data.get('compression_type', 'none'),
            original_size=data.get('original_size', 0),
            compressed_size=data.get('compressed_size', 0),
            tags=data.get('tags', []),
            branch=data.get('branch', 'main'),
            status=data.get('status', 'active'),
            ttl=data.get('ttl')
        )


# ==================== 工具函数 ====================

def compute_checksum(data: bytes) -> str:
    """计算数据校验和"""
    return hashlib.sha256(data).hexdigest()[:16]


def compress_data(data: bytes, compression_type: CompressionType = CompressionType.GZIP) -> Tuple[bytes, str]:
    """压缩数据"""
    if compression_type == CompressionType.NONE:
        return data, "none"
    elif compression_type == CompressionType.GZIP:
        compressed = gzip.compress(data, compresslevel=6)
        return compressed, "gzip"
    elif compression_type == CompressionType.LZ4:
        try:
            import lz4.frame
            compressed = lz4.frame.compress(data)
            return compressed, "lz4"
        except ImportError:
            logger.warning("lz4 not available, falling back to gzip")
            return gzip.compress(data, compresslevel=6), "gzip"
    return data, "none"


def decompress_data(data: bytes, compression_type: str) -> bytes:
    """解压数据"""
    if compression_type == "none":
        return data
    elif compression_type == "gzip":
        return gzip.decompress(data)
    elif compression_type == "lz4":
        try:
            import lz4.frame
            return lz4.frame.decompress(data)
        except ImportError:
            raise ValueError("lz4 library required for decompression")
    return data


def timed_operation(operation_name: str):
    """计时装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = func(self, *args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"{operation_name} completed in {elapsed_ms:.2f}ms")
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.error(f"{operation_name} failed after {elapsed_ms:.2f}ms: {e}")
                raise
        return wrapper
    return decorator


def async_timed_operation(operation_name: str):
    """异步计时装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            start_time = time.time()
            try:
                result = await func(self, *args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000
                logger.debug(f"{operation_name} completed in {elapsed_ms:.2f}ms")
                return result
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.error(f"{operation_name} failed after {elapsed_ms:.2f}ms: {e}")
                raise
        return wrapper
    return decorator


class Checkpointer(ABC):
    """检查点基类（增强版）
    
    定义检查点的完整接口，包括基本操作和高级功能。
    
    生产级特性：
    - 数据压缩
    - 数据校验
    - 缓存支持
    - 性能指标
    - 标签管理
    - 分支支持
    - 差异比较
    - 回滚功能
    """
    
    def __init__(self, 
                 compression: CompressionType = CompressionType.NONE,
                 enable_cache: bool = True,
                 cache_size: int = 100,
                 enable_metrics: bool = True):
        self.compression = compression
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.enable_metrics = enable_metrics
        
        # 缓存
        self._cache: Dict[str, EnhancedCheckpoint] = {}
        self._cache_order: List[str] = []
        self._cache_lock = threading.Lock()
        
        # 性能指标
        self._metrics = CheckpointMetrics()
        self._metrics_lock = threading.Lock()
        
        # 标签
        self._tags: Dict[str, CheckpointTag] = {}
        
        # 回调
        self._on_save_callbacks: List[Callable] = []
        self._on_load_callbacks: List[Callable] = []
    
    # ==================== 抽象方法 ====================
    
    @abstractmethod
    def save(self, state: AgentState, tags: List[str] = None, 
             branch: str = "main", ttl: int = None) -> str:
        """保存状态，返回检查点 ID
        
        Args:
            state: Agent 状态
            tags: 标签列表
            branch: 分支名称
            ttl: 过期时间（秒）
        
        Returns:
            检查点 ID
        """
        pass
    
    @abstractmethod
    def load(self, checkpoint_id: str) -> Optional[AgentState]:
        """加载指定检查点"""
        pass
    
    @abstractmethod
    def get_latest(self, thread_id: str = None, branch: str = None) -> Optional[AgentState]:
        """获取最新检查点
        
        Args:
            thread_id: 线程 ID
            branch: 分支名称
        """
        pass
    
    @abstractmethod
    def list_checkpoints(self, thread_id: str = None, branch: str = None,
                        limit: int = 10, offset: int = 0,
                        tags: List[str] = None,
                        start_time: datetime = None,
                        end_time: datetime = None) -> List[EnhancedCheckpoint]:
        """列出检查点
        
        Args:
            thread_id: 线程 ID
            branch: 分支名称
            limit: 返回数量限制
            offset: 偏移量
            tags: 标签过滤
            start_time: 开始时间
            end_time: 结束时间
        """
        pass
    
    @abstractmethod
    def delete(self, checkpoint_id: str, soft: bool = False) -> bool:
        """删除检查点
        
        Args:
            checkpoint_id: 检查点 ID
            soft: 是否软删除（标记为删除但不实际删除）
        """
        pass
    
    @abstractmethod
    def _store_checkpoint(self, checkpoint: EnhancedCheckpoint) -> bool:
        """存储检查点数据（子类实现）"""
        pass
    
    @abstractmethod
    def _retrieve_checkpoint(self, checkpoint_id: str) -> Optional[EnhancedCheckpoint]:
        """检索检查点数据（子类实现）"""
        pass
    
    # ==================== 核心功能 ====================
    
    def create_checkpoint(self, state: AgentState, parent_id: str = None,
                         tags: List[str] = None, branch: str = "main",
                         ttl: int = None) -> EnhancedCheckpoint:
        """创建增强版检查点对象"""
        # 序列化状态
        state_dict = state.to_dict()
        state_json = json.dumps(state_dict, ensure_ascii=False)
        state_bytes = state_json.encode('utf-8')
        original_size = len(state_bytes)
        
        # 压缩
        if self.compression != CompressionType.NONE:
            compressed_bytes, compression_type = compress_data(state_bytes, self.compression)
            compressed_size = len(compressed_bytes)
            is_compressed = True
        else:
            compressed_bytes = state_bytes
            compressed_size = original_size
            compression_type = "none"
            is_compressed = False
        
        # 计算校验和
        checksum = compute_checksum(compressed_bytes)
        
        # 创建检查点
        checkpoint = EnhancedCheckpoint(
            checkpoint_id=str(uuid.uuid4()),
            thread_id=state.thread_id,
            state=state_dict,
            parent_id=parent_id,
            timestamp=datetime.utcnow(),
            metadata={
                "iteration": state.iteration,
                "status": state.status.value,
                "message_count": len(state.messages),
                "agent_id": state.agent_id
            },
            version=1,
            checksum=checksum,
            compressed=is_compressed,
            compression_type=compression_type,
            original_size=original_size,
            compressed_size=compressed_size,
            tags=tags or [],
            branch=branch,
            status="active",
            ttl=ttl
        )
        
        return checkpoint
    
    def validate_checkpoint(self, checkpoint: EnhancedCheckpoint) -> bool:
        """验证检查点数据完整性"""
        try:
            # 重新序列化并计算校验和
            state_json = json.dumps(checkpoint.state, ensure_ascii=False)
            state_bytes = state_json.encode('utf-8')
            
            if checkpoint.compressed:
                compressed_bytes, _ = compress_data(state_bytes, 
                    CompressionType(checkpoint.compression_type) if checkpoint.compression_type != "none" else CompressionType.GZIP)
                checksum = compute_checksum(compressed_bytes)
            else:
                checksum = compute_checksum(state_bytes)
            
            return checksum == checkpoint.checksum
        except Exception as e:
            logger.warning(f"Checkpoint validation failed: {e}")
            return False
    
    # ==================== 缓存管理 ====================
    
    def _cache_put(self, checkpoint: EnhancedCheckpoint) -> None:
        """将检查点放入缓存"""
        if not self.enable_cache:
            return
        
        with self._cache_lock:
            checkpoint_id = checkpoint.checkpoint_id
            
            # 如果已存在，更新顺序
            if checkpoint_id in self._cache:
                self._cache_order.remove(checkpoint_id)
            
            # 如果缓存已满，移除最旧的
            while len(self._cache) >= self.cache_size:
                oldest_id = self._cache_order.pop(0)
                del self._cache[oldest_id]
            
            # 添加到缓存
            self._cache[checkpoint_id] = checkpoint
            self._cache_order.append(checkpoint_id)
    
    def _cache_get(self, checkpoint_id: str) -> Optional[EnhancedCheckpoint]:
        """从缓存获取检查点"""
        if not self.enable_cache:
            return None
        
        with self._cache_lock:
            checkpoint = self._cache.get(checkpoint_id)
            if checkpoint:
                # 更新访问顺序（LRU）
                self._cache_order.remove(checkpoint_id)
                self._cache_order.append(checkpoint_id)
                
                if self.enable_metrics:
                    with self._metrics_lock:
                        self._metrics.cache_hits += 1
                
                return checkpoint
            else:
                if self.enable_metrics:
                    with self._metrics_lock:
                        self._metrics.cache_misses += 1
                return None
    
    def _cache_invalidate(self, checkpoint_id: str) -> None:
        """使缓存失效"""
        with self._cache_lock:
            if checkpoint_id in self._cache:
                del self._cache[checkpoint_id]
                self._cache_order.remove(checkpoint_id)
    
    def clear_cache(self) -> None:
        """清空缓存"""
        with self._cache_lock:
            self._cache.clear()
            self._cache_order.clear()
    
    # ==================== 性能指标 ====================
    
    def _record_save_metrics(self, bytes_saved: int, elapsed_ms: float) -> None:
        """记录保存指标"""
        if not self.enable_metrics:
            return
        
        with self._metrics_lock:
            self._metrics.total_saves += 1
            self._metrics.total_bytes_saved += bytes_saved
            # 计算移动平均
            n = self._metrics.total_saves
            self._metrics.avg_save_time_ms = (
                (self._metrics.avg_save_time_ms * (n - 1) + elapsed_ms) / n
            )
            self._metrics.last_save_time = datetime.utcnow()
    
    def _record_load_metrics(self, bytes_loaded: int, elapsed_ms: float) -> None:
        """记录加载指标"""
        if not self.enable_metrics:
            return
        
        with self._metrics_lock:
            self._metrics.total_loads += 1
            self._metrics.total_bytes_loaded += bytes_loaded
            n = self._metrics.total_loads
            self._metrics.avg_load_time_ms = (
                (self._metrics.avg_load_time_ms * (n - 1) + elapsed_ms) / n
            )
            self._metrics.last_load_time = datetime.utcnow()
    
    def get_metrics(self) -> CheckpointMetrics:
        """获取性能指标"""
        with self._metrics_lock:
            return copy.copy(self._metrics)
    
    def reset_metrics(self) -> None:
        """重置性能指标"""
        with self._metrics_lock:
            self._metrics = CheckpointMetrics()
    
    # ==================== 标签管理 ====================
    
    def add_tag(self, checkpoint_id: str, tag_name: str, 
                description: str = "", metadata: Dict[str, Any] = None) -> CheckpointTag:
        """为检查点添加标签
        
        Args:
            checkpoint_id: 检查点 ID
            tag_name: 标签名称
            description: 标签描述
            metadata: 标签元数据
        
        Returns:
            创建的标签对象
        """
        tag = CheckpointTag(
            name=tag_name,
            checkpoint_id=checkpoint_id,
            description=description,
            metadata=metadata or {}
        )
        self._tags[tag_name] = tag
        
        # 更新检查点的标签列表
        checkpoint = self._retrieve_checkpoint(checkpoint_id)
        if checkpoint:
            if tag_name not in checkpoint.tags:
                checkpoint.tags.append(tag_name)
                self._store_checkpoint(checkpoint)
        
        return tag
    
    def remove_tag(self, tag_name: str) -> bool:
        """移除标签"""
        if tag_name in self._tags:
            tag = self._tags.pop(tag_name)
            
            # 从检查点中移除标签
            checkpoint = self._retrieve_checkpoint(tag.checkpoint_id)
            if checkpoint and tag_name in checkpoint.tags:
                checkpoint.tags.remove(tag_name)
                self._store_checkpoint(checkpoint)
            
            return True
        return False
    
    def get_tag(self, tag_name: str) -> Optional[CheckpointTag]:
        """获取标签"""
        return self._tags.get(tag_name)
    
    def list_tags(self) -> List[CheckpointTag]:
        """列出所有标签"""
        return list(self._tags.values())
    
    def get_checkpoint_by_tag(self, tag_name: str) -> Optional[AgentState]:
        """通过标签获取检查点"""
        tag = self._tags.get(tag_name)
        if tag:
            return self.load(tag.checkpoint_id)
        return None
    
    # ==================== 分支管理 ====================
    
    def create_branch(self, checkpoint_id: str, branch_name: str) -> Optional[str]:
        """从指定检查点创建新分支
        
        Args:
            checkpoint_id: 源检查点 ID
            branch_name: 新分支名称
        
        Returns:
            新检查点 ID
        """
        checkpoint = self._retrieve_checkpoint(checkpoint_id)
        if not checkpoint:
            return None
        
        # 加载状态
        state = AgentState.from_dict(checkpoint.state)
        
        # 在新分支上保存
        return self.save(state, branch=branch_name)
    
    def list_branches(self, thread_id: str = None) -> List[str]:
        """列出所有分支"""
        checkpoints = self.list_checkpoints(thread_id=thread_id, limit=1000)
        branches = set()
        for cp in checkpoints:
            branches.add(cp.branch)
        return sorted(list(branches))
    
    def merge_branches(self, source_branch: str, target_branch: str,
                      thread_id: str, strategy: str = "latest") -> Optional[str]:
        """合并分支
        
        Args:
            source_branch: 源分支
            target_branch: 目标分支
            thread_id: 线程 ID
            strategy: 合并策略（latest, combine）
        
        Returns:
            合并后的检查点 ID
        """
        source_state = self.get_latest(thread_id=thread_id, branch=source_branch)
        target_state = self.get_latest(thread_id=thread_id, branch=target_branch)
        
        if not source_state:
            return None
        
        if strategy == "latest":
            # 使用最新的状态
            return self.save(source_state, branch=target_branch)
        
        elif strategy == "combine" and target_state:
            # 合并消息历史
            merged_state = target_state.copy()
            # 添加源分支的消息（避免重复）
            existing_ids = {m.id for m in merged_state.messages}
            for msg in source_state.messages:
                if msg.id not in existing_ids:
                    merged_state.messages.append(msg)
            
            return self.save(merged_state, branch=target_branch)
        
        return None
    
    # ==================== 差异比较 ====================
    
    def diff(self, checkpoint_id1: str, checkpoint_id2: str) -> Optional[CheckpointDiff]:
        """比较两个检查点的差异
        
        Args:
            checkpoint_id1: 第一个检查点 ID
            checkpoint_id2: 第二个检查点 ID
        
        Returns:
            差异对象
        """
        cp1 = self._retrieve_checkpoint(checkpoint_id1)
        cp2 = self._retrieve_checkpoint(checkpoint_id2)
        
        if not cp1 or not cp2:
            return None
        
        state1 = cp1.state
        state2 = cp2.state
        
        # 比较键
        keys1 = set(state1.keys())
        keys2 = set(state2.keys())
        
        added_keys = list(keys2 - keys1)
        removed_keys = list(keys1 - keys2)
        
        # 比较共同键的值
        modified_keys = []
        for key in keys1 & keys2:
            if state1[key] != state2[key]:
                modified_keys.append(key)
        
        # 特殊比较
        messages1 = state1.get('messages', [])
        messages2 = state2.get('messages', [])
        message_count_diff = len(messages2) - len(messages1)
        
        iteration_diff = state2.get('iteration', 0) - state1.get('iteration', 0)
        
        return CheckpointDiff(
            from_checkpoint_id=checkpoint_id1,
            to_checkpoint_id=checkpoint_id2,
            added_keys=added_keys,
            removed_keys=removed_keys,
            modified_keys=modified_keys,
            message_count_diff=message_count_diff,
            iteration_diff=iteration_diff
        )
    
    # ==================== 回滚 ====================
    
    def rollback(self, thread_id: str, checkpoint_id: str = None,
                steps: int = None) -> Optional[AgentState]:
        """回滚到指定检查点或回退指定步数
        
        Args:
            thread_id: 线程 ID
            checkpoint_id: 目标检查点 ID（可选）
            steps: 回退步数（可选）
        
        Returns:
            回滚后的状态
        """
        if checkpoint_id:
            # 回滚到指定检查点
            return self.load(checkpoint_id)
        
        if steps is not None and steps > 0:
            # 回退指定步数
            checkpoints = self.list_checkpoints(thread_id=thread_id, limit=steps + 1)
            if len(checkpoints) > steps:
                return AgentState.from_dict(checkpoints[steps].state)
        
        return None
    
    # ==================== 批量操作 ====================
    
    def save_batch(self, states: List[Tuple[AgentState, Optional[List[str]], str]]) -> List[str]:
        """批量保存状态
        
        Args:
            states: (状态, 标签, 分支) 元组列表
        
        Returns:
            检查点 ID 列表
        """
        checkpoint_ids = []
        for state, tags, branch in states:
            checkpoint_id = self.save(state, tags=tags, branch=branch)
            checkpoint_ids.append(checkpoint_id)
        return checkpoint_ids
    
    def load_batch(self, checkpoint_ids: List[str]) -> List[Optional[AgentState]]:
        """批量加载检查点
        
        Args:
            checkpoint_ids: 检查点 ID 列表
        
        Returns:
            状态列表
        """
        return [self.load(cid) for cid in checkpoint_ids]
    
    def delete_batch(self, checkpoint_ids: List[str], soft: bool = False) -> int:
        """批量删除检查点
        
        Args:
            checkpoint_ids: 检查点 ID 列表
            soft: 是否软删除
        
        Returns:
            成功删除的数量
        """
        count = 0
        for cid in checkpoint_ids:
            if self.delete(cid, soft=soft):
                count += 1
        return count
    
    # ==================== 回调管理 ====================
    
    def on_save(self, callback: Callable[[EnhancedCheckpoint], None]) -> None:
        """注册保存回调"""
        self._on_save_callbacks.append(callback)
    
    def on_load(self, callback: Callable[[EnhancedCheckpoint], None]) -> None:
        """注册加载回调"""
        self._on_load_callbacks.append(callback)
    
    def _trigger_save_callbacks(self, checkpoint: EnhancedCheckpoint) -> None:
        """触发保存回调"""
        for callback in self._on_save_callbacks:
            try:
                callback(checkpoint)
            except Exception as e:
                logger.warning(f"Save callback error: {e}")
    
    def _trigger_load_callbacks(self, checkpoint: EnhancedCheckpoint) -> None:
        """触发加载回调"""
        for callback in self._on_load_callbacks:
            try:
                callback(checkpoint)
            except Exception as e:
                logger.warning(f"Load callback error: {e}")
    
    # ==================== 生命周期管理 ====================
    
    def cleanup_expired(self) -> int:
        """清理过期检查点
        
        Returns:
            清理的检查点数量
        """
        # 子类实现具体逻辑
        return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        return {
            "total_checkpoints": 0,
            "total_size_bytes": 0,
            "compressed_size_bytes": 0,
            "branches": [],
            "threads": []
        }
    
    # ==================== 上下文管理 ====================
    
    @contextmanager
    def transaction(self):
        """事务上下文（子类可实现真正的事务）"""
        yield self


class MemoryCheckpointer(Checkpointer):
    """内存检查点（增强版）
    
    将检查点保存在内存中（适用于开发和测试）。
    
    特性：
    - 支持压缩（减少内存使用）
    - 支持 LRU 缓存
    - 支持标签和分支
    - 支持差异比较和回滚
    - 线程安全
    """
    
    def __init__(self, max_checkpoints: int = 100,
                 compression: CompressionType = CompressionType.NONE,
                 enable_cache: bool = True,
                 cache_size: int = 50):
        super().__init__(
            compression=compression,
            enable_cache=enable_cache,
            cache_size=cache_size
        )
        
        self._checkpoints: Dict[str, EnhancedCheckpoint] = {}
        self._thread_checkpoints: Dict[str, List[str]] = {}  # thread_id -> [checkpoint_ids]
        self._branch_checkpoints: Dict[str, Dict[str, List[str]]] = {}  # thread_id -> branch -> [checkpoint_ids]
        self._max_checkpoints = max_checkpoints
        self._lock = threading.RLock()
    
    @timed_operation("MemoryCheckpointer.save")
    def save(self, state: AgentState, tags: List[str] = None,
             branch: str = "main", ttl: int = None) -> str:
        """保存状态"""
        start_time = time.time()
        
        with self._lock:
            # 获取父检查点 ID
            parent_id = None
            if state.thread_id in self._branch_checkpoints:
                branch_cps = self._branch_checkpoints[state.thread_id].get(branch, [])
                if branch_cps:
                    parent_id = branch_cps[-1]
            elif state.thread_id in self._thread_checkpoints:
                checkpoints = self._thread_checkpoints[state.thread_id]
                if checkpoints:
                    parent_id = checkpoints[-1]
            
            # 创建检查点
            checkpoint = self.create_checkpoint(state, parent_id, tags, branch, ttl)
            
            # 存储
            self._checkpoints[checkpoint.checkpoint_id] = checkpoint
            
            # 更新线程索引
            if state.thread_id not in self._thread_checkpoints:
                self._thread_checkpoints[state.thread_id] = []
            self._thread_checkpoints[state.thread_id].append(checkpoint.checkpoint_id)
            
            # 更新分支索引
            if state.thread_id not in self._branch_checkpoints:
                self._branch_checkpoints[state.thread_id] = {}
            if branch not in self._branch_checkpoints[state.thread_id]:
                self._branch_checkpoints[state.thread_id][branch] = []
            self._branch_checkpoints[state.thread_id][branch].append(checkpoint.checkpoint_id)
            
            # 添加到缓存
            self._cache_put(checkpoint)
            
            # 清理旧检查点
            self._cleanup()
            
            # 记录指标
            elapsed_ms = (time.time() - start_time) * 1000
            self._record_save_metrics(checkpoint.compressed_size, elapsed_ms)
            
            # 触发回调
            self._trigger_save_callbacks(checkpoint)
            
            logger.debug(f"Saved checkpoint: {checkpoint.checkpoint_id} (branch={branch})")
            return checkpoint.checkpoint_id
    
    @timed_operation("MemoryCheckpointer.load")
    def load(self, checkpoint_id: str) -> Optional[AgentState]:
        """加载检查点"""
        start_time = time.time()
        
        # 先查缓存
        checkpoint = self._cache_get(checkpoint_id)
        if not checkpoint:
            checkpoint = self._retrieve_checkpoint(checkpoint_id)
        
        if checkpoint:
            # 验证检查点
            if checkpoint.status == "deleted":
                return None
            
            elapsed_ms = (time.time() - start_time) * 1000
            self._record_load_metrics(checkpoint.original_size, elapsed_ms)
            
            # 触发回调
            self._trigger_load_callbacks(checkpoint)
            
            return AgentState.from_dict(checkpoint.state)
        return None
    
    def get_latest(self, thread_id: str = None, branch: str = None) -> Optional[AgentState]:
        """获取最新状态"""
        with self._lock:
            if thread_id and branch:
                # 获取特定线程和分支的最新
                if thread_id in self._branch_checkpoints:
                    branch_cps = self._branch_checkpoints[thread_id].get(branch, [])
                    if branch_cps:
                        return self.load(branch_cps[-1])
            
            if thread_id:
                checkpoints = self._thread_checkpoints.get(thread_id, [])
                if checkpoints:
                    return self.load(checkpoints[-1])
            else:
                # 获取全局最新
                if self._checkpoints:
                    active_checkpoints = [
                        cp for cp in self._checkpoints.values()
                        if cp.status == "active"
                    ]
                    if active_checkpoints:
                        latest = max(active_checkpoints, key=lambda c: c.timestamp)
                        return AgentState.from_dict(latest.state)
            return None
    
    def list_checkpoints(self, thread_id: str = None, branch: str = None,
                        limit: int = 10, offset: int = 0,
                        tags: List[str] = None,
                        start_time: datetime = None,
                        end_time: datetime = None) -> List[EnhancedCheckpoint]:
        """列出检查点"""
        with self._lock:
            # 获取候选检查点
            if thread_id and branch:
                checkpoint_ids = self._branch_checkpoints.get(thread_id, {}).get(branch, [])
            elif thread_id:
                checkpoint_ids = self._thread_checkpoints.get(thread_id, [])
            else:
                checkpoint_ids = list(self._checkpoints.keys())
            
            checkpoints = [
                self._checkpoints[cid] 
                for cid in checkpoint_ids 
                if cid in self._checkpoints
            ]
            
            # 过滤已删除
            checkpoints = [cp for cp in checkpoints if cp.status != "deleted"]
            
            # 按标签过滤
            if tags:
                checkpoints = [
                    cp for cp in checkpoints 
                    if any(tag in cp.tags for tag in tags)
                ]
            
            # 按时间过滤
            if start_time:
                checkpoints = [cp for cp in checkpoints if cp.timestamp >= start_time]
            if end_time:
                checkpoints = [cp for cp in checkpoints if cp.timestamp <= end_time]
            
            # 按时间排序
            checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
            
            # 分页
            return checkpoints[offset:offset + limit]
    
    def delete(self, checkpoint_id: str, soft: bool = False) -> bool:
        """删除检查点"""
        with self._lock:
            if checkpoint_id in self._checkpoints:
                if soft:
                    # 软删除：标记为已删除
                    self._checkpoints[checkpoint_id].status = "deleted"
                else:
                    checkpoint = self._checkpoints.pop(checkpoint_id)
                    # 从线程索引中移除
                    if checkpoint.thread_id in self._thread_checkpoints:
                        self._thread_checkpoints[checkpoint.thread_id] = [
                            cid for cid in self._thread_checkpoints[checkpoint.thread_id]
                            if cid != checkpoint_id
                        ]
                    # 从分支索引中移除
                    if checkpoint.thread_id in self._branch_checkpoints:
                        if checkpoint.branch in self._branch_checkpoints[checkpoint.thread_id]:
                            self._branch_checkpoints[checkpoint.thread_id][checkpoint.branch] = [
                                cid for cid in self._branch_checkpoints[checkpoint.thread_id][checkpoint.branch]
                                if cid != checkpoint_id
                            ]
                
                # 使缓存失效
                self._cache_invalidate(checkpoint_id)
                
                with self._metrics_lock:
                    self._metrics.total_deletes += 1
                
                return True
            return False
    
    def _store_checkpoint(self, checkpoint: EnhancedCheckpoint) -> bool:
        """存储检查点"""
        with self._lock:
            self._checkpoints[checkpoint.checkpoint_id] = checkpoint
            return True
    
    def _retrieve_checkpoint(self, checkpoint_id: str) -> Optional[EnhancedCheckpoint]:
        """检索检查点"""
        with self._lock:
            return self._checkpoints.get(checkpoint_id)
    
    def _cleanup(self) -> None:
        """清理旧检查点"""
        if len(self._checkpoints) > self._max_checkpoints:
            # 按时间排序，删除最旧的
            sorted_checkpoints = sorted(
                self._checkpoints.values(),
                key=lambda c: c.timestamp
            )
            to_delete = sorted_checkpoints[:-self._max_checkpoints]
            for checkpoint in to_delete:
                self.delete(checkpoint.checkpoint_id)
    
    def cleanup_expired(self) -> int:
        """清理过期检查点"""
        now = datetime.utcnow()
        count = 0
        
        with self._lock:
            for checkpoint in list(self._checkpoints.values()):
                if checkpoint.ttl and checkpoint.ttl > 0:
                    expiry = checkpoint.timestamp + timedelta(seconds=checkpoint.ttl)
                    if now > expiry:
                        self.delete(checkpoint.checkpoint_id)
                        count += 1
        
        return count
    
    def clear(self) -> None:
        """清除所有检查点"""
        with self._lock:
            self._checkpoints.clear()
            self._thread_checkpoints.clear()
            self._branch_checkpoints.clear()
            self.clear_cache()
    
    def get_history(self, thread_id: str, branch: str = None) -> List[AgentState]:
        """获取线程的状态历史"""
        checkpoints = self.list_checkpoints(thread_id, branch=branch, limit=100)
        return [AgentState.from_dict(c.state) for c in reversed(checkpoints)]
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        with self._lock:
            total_size = 0
            compressed_size = 0
            branches = set()
            threads = set()
            
            for checkpoint in self._checkpoints.values():
                total_size += checkpoint.original_size
                compressed_size += checkpoint.compressed_size
                branches.add(checkpoint.branch)
                threads.add(checkpoint.thread_id)
            
            return {
                "total_checkpoints": len(self._checkpoints),
                "active_checkpoints": len([cp for cp in self._checkpoints.values() if cp.status == "active"]),
                "total_size_bytes": total_size,
                "compressed_size_bytes": compressed_size,
                "compression_ratio": compressed_size / total_size if total_size > 0 else 1.0,
                "branches": list(branches),
                "threads": list(threads),
                "cache_size": len(self._cache)
            }
    
    def export_checkpoints(self, thread_id: str = None) -> Dict[str, Any]:
        """导出检查点数据（用于迁移）"""
        checkpoints = self.list_checkpoints(thread_id=thread_id, limit=10000)
        return {
            "version": 1,
            "exported_at": datetime.utcnow().isoformat(),
            "checkpoints": [cp.to_dict() for cp in checkpoints],
            "tags": [tag.to_dict() for tag in self._tags.values()]
        }
    
    def import_checkpoints(self, data: Dict[str, Any]) -> int:
        """导入检查点数据"""
        imported = 0
        
        for cp_data in data.get("checkpoints", []):
            checkpoint = EnhancedCheckpoint.from_dict(cp_data)
            self._store_checkpoint(checkpoint)
            
            # 更新索引
            if checkpoint.thread_id not in self._thread_checkpoints:
                self._thread_checkpoints[checkpoint.thread_id] = []
            self._thread_checkpoints[checkpoint.thread_id].append(checkpoint.checkpoint_id)
            
            imported += 1
        
        # 导入标签
        for tag_data in data.get("tags", []):
            tag = CheckpointTag.from_dict(tag_data)
            self._tags[tag.name] = tag
        
        return imported


class RedisCheckpointer(Checkpointer):
    """Redis 检查点（增强版）
    
    将检查点保存在 Redis 中（适用于分布式生产环境）。
    
    特性：
    - 数据压缩
    - 分布式锁
    - Pipeline 批量操作
    - Lua 脚本原子操作
    - 自动过期清理
    - 分支和标签支持
    - 异步操作支持
    """
    
    def __init__(self, 
                 redis_client=None,
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 password: str = None,
                 prefix: str = "langgraph:checkpoint:",
                 default_ttl: int = 86400 * 7,  # 7天过期
                 compression: CompressionType = CompressionType.GZIP,
                 enable_cache: bool = True,
                 cache_size: int = 100,
                 max_connections: int = 10):
        super().__init__(
            compression=compression,
            enable_cache=enable_cache,
            cache_size=cache_size
        )
        
        self.prefix = prefix
        self.default_ttl = default_ttl
        self._fallback = None
        
        if redis_client:
            self.redis = redis_client
        else:
            try:
                import redis
                self.redis = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=False,  # 使用 bytes 以支持压缩数据
                    max_connections=max_connections
                )
                # 测试连接
                self.redis.ping()
            except ImportError:
                logger.warning("Redis not available, falling back to memory")
                self._fallback = MemoryCheckpointer(compression=compression)
                self.redis = None
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, falling back to memory")
                self._fallback = MemoryCheckpointer(compression=compression)
                self.redis = None
    
    def _key(self, checkpoint_id: str) -> str:
        """生成检查点 key"""
        return f"{self.prefix}cp:{checkpoint_id}"
    
    def _thread_key(self, thread_id: str) -> str:
        """生成线程索引 key"""
        return f"{self.prefix}thread:{thread_id}"
    
    def _branch_key(self, thread_id: str, branch: str) -> str:
        """生成分支索引 key"""
        return f"{self.prefix}branch:{thread_id}:{branch}"
    
    def _tag_key(self, tag_name: str) -> str:
        """生成标签 key"""
        return f"{self.prefix}tag:{tag_name}"
    
    def _lock_key(self, thread_id: str) -> str:
        """生成锁 key"""
        return f"{self.prefix}lock:{thread_id}"
    
    @contextmanager
    def _distributed_lock(self, thread_id: str, timeout: int = 10):
        """分布式锁"""
        if not self.redis:
            yield
            return
        
        lock_key = self._lock_key(thread_id)
        lock_value = str(uuid.uuid4())
        
        try:
            # 尝试获取锁
            acquired = self.redis.set(lock_key, lock_value, nx=True, ex=timeout)
            if not acquired:
                raise TimeoutError(f"Could not acquire lock for thread {thread_id}")
            yield
        finally:
            # 释放锁（使用 Lua 脚本确保原子性）
            script = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            self.redis.eval(script, 1, lock_key, lock_value)
    
    @timed_operation("RedisCheckpointer.save")
    def save(self, state: AgentState, tags: List[str] = None,
             branch: str = "main", ttl: int = None) -> str:
        """保存状态到 Redis"""
        if not self.redis:
            return self._fallback.save(state, tags, branch, ttl)
        
        start_time = time.time()
        
        with self._distributed_lock(state.thread_id):
            # 获取父检查点
            parent_id = None
            branch_key = self._branch_key(state.thread_id, branch)
            latest = self.redis.lindex(branch_key, -1)
            if latest:
                parent_id = latest.decode('utf-8') if isinstance(latest, bytes) else latest
            
            # 创建检查点
            checkpoint = self.create_checkpoint(state, parent_id, tags, branch, ttl)
            
            # 序列化并压缩
            checkpoint_json = json.dumps(checkpoint.to_dict(), ensure_ascii=False)
            checkpoint_bytes = checkpoint_json.encode('utf-8')
            
            if self.compression != CompressionType.NONE:
                data, _ = compress_data(checkpoint_bytes, self.compression)
            else:
                data = checkpoint_bytes
            
            # 使用 Pipeline 批量操作
            pipe = self.redis.pipeline()
            
            # 存储检查点
            key = self._key(checkpoint.checkpoint_id)
            actual_ttl = ttl or self.default_ttl
            pipe.setex(key, actual_ttl, data)
            
            # 更新线程索引
            thread_key = self._thread_key(state.thread_id)
            pipe.rpush(thread_key, checkpoint.checkpoint_id)
            pipe.expire(thread_key, actual_ttl)
            
            # 更新分支索引
            pipe.rpush(branch_key, checkpoint.checkpoint_id)
            pipe.expire(branch_key, actual_ttl)
            
            # 存储标签
            if tags:
                for tag in tags:
                    tag_key = self._tag_key(tag)
                    pipe.set(tag_key, checkpoint.checkpoint_id)
                    pipe.expire(tag_key, actual_ttl)
            
            pipe.execute()
            
            # 添加到缓存
            self._cache_put(checkpoint)
            
            # 记录指标
            elapsed_ms = (time.time() - start_time) * 1000
            self._record_save_metrics(len(data), elapsed_ms)
            
            # 触发回调
            self._trigger_save_callbacks(checkpoint)
        
        logger.debug(f"Saved checkpoint to Redis: {checkpoint.checkpoint_id}")
        return checkpoint.checkpoint_id
    
    @timed_operation("RedisCheckpointer.load")
    def load(self, checkpoint_id: str) -> Optional[AgentState]:
        """从 Redis 加载检查点"""
        if not self.redis:
            return self._fallback.load(checkpoint_id)
        
        start_time = time.time()
        
        # 先查缓存
        checkpoint = self._cache_get(checkpoint_id)
        if checkpoint:
            return AgentState.from_dict(checkpoint.state)
        
        # 从 Redis 加载
        key = self._key(checkpoint_id)
        data = self.redis.get(key)
        
        if data:
            # 解压
            if self.compression != CompressionType.NONE:
                try:
                    decompressed = decompress_data(data, self.compression.value)
                except:
                    decompressed = data
            else:
                decompressed = data
            
            checkpoint_dict = json.loads(decompressed.decode('utf-8'))
            checkpoint = EnhancedCheckpoint.from_dict(checkpoint_dict)
            
            # 验证
            if checkpoint.status == "deleted":
                return None
            
            # 添加到缓存
            self._cache_put(checkpoint)
            
            # 记录指标
            elapsed_ms = (time.time() - start_time) * 1000
            self._record_load_metrics(checkpoint.original_size, elapsed_ms)
            
            # 触发回调
            self._trigger_load_callbacks(checkpoint)
            
            return AgentState.from_dict(checkpoint.state)
        return None
    
    def get_latest(self, thread_id: str = None, branch: str = None) -> Optional[AgentState]:
        """获取最新状态"""
        if not self.redis:
            return self._fallback.get_latest(thread_id, branch)
        
        if thread_id and branch:
            branch_key = self._branch_key(thread_id, branch)
            checkpoint_id = self.redis.lindex(branch_key, -1)
            if checkpoint_id:
                cid = checkpoint_id.decode('utf-8') if isinstance(checkpoint_id, bytes) else checkpoint_id
                return self.load(cid)
        
        if thread_id:
            thread_key = self._thread_key(thread_id)
            checkpoint_id = self.redis.lindex(thread_key, -1)
            if checkpoint_id:
                cid = checkpoint_id.decode('utf-8') if isinstance(checkpoint_id, bytes) else checkpoint_id
                return self.load(cid)
        else:
            # 获取所有线程的最新（较慢）
            pattern = f"{self.prefix}thread:*"
            thread_keys = self.redis.keys(pattern)
            
            latest_checkpoint = None
            latest_time = None
            
            for thread_key in thread_keys:
                checkpoint_id = self.redis.lindex(thread_key, -1)
                if checkpoint_id:
                    cid = checkpoint_id.decode('utf-8') if isinstance(checkpoint_id, bytes) else checkpoint_id
                    checkpoint = self._retrieve_checkpoint(cid)
                    if checkpoint and checkpoint.status == "active":
                        if latest_time is None or checkpoint.timestamp > latest_time:
                            latest_time = checkpoint.timestamp
                            latest_checkpoint = checkpoint
            
            if latest_checkpoint:
                return AgentState.from_dict(latest_checkpoint.state)
        
        return None
    
    def list_checkpoints(self, thread_id: str = None, branch: str = None,
                        limit: int = 10, offset: int = 0,
                        tags: List[str] = None,
                        start_time: datetime = None,
                        end_time: datetime = None) -> List[EnhancedCheckpoint]:
        """列出检查点"""
        if not self.redis:
            return self._fallback.list_checkpoints(thread_id, branch, limit, offset, tags, start_time, end_time)
        
        checkpoints = []
        
        if thread_id and branch:
            branch_key = self._branch_key(thread_id, branch)
            checkpoint_ids = self.redis.lrange(branch_key, -(limit + offset), -1 - offset if offset > 0 else -1)
        elif thread_id:
            thread_key = self._thread_key(thread_id)
            checkpoint_ids = self.redis.lrange(thread_key, -(limit + offset), -1 - offset if offset > 0 else -1)
        else:
            # 获取所有检查点（使用 SCAN 避免阻塞）
            checkpoint_ids = []
            cursor = 0
            pattern = f"{self.prefix}cp:*"
            while True:
                cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
                for key in keys:
                    key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                    cid = key_str.replace(f"{self.prefix}cp:", '')
                    checkpoint_ids.append(cid.encode('utf-8'))
                if cursor == 0:
                    break
        
        # 加载检查点
        for cid_bytes in checkpoint_ids:
            cid = cid_bytes.decode('utf-8') if isinstance(cid_bytes, bytes) else cid_bytes
            checkpoint = self._retrieve_checkpoint(cid)
            if checkpoint and checkpoint.status == "active":
                # 按标签过滤
                if tags and not any(tag in checkpoint.tags for tag in tags):
                    continue
                # 按时间过滤
                if start_time and checkpoint.timestamp < start_time:
                    continue
                if end_time and checkpoint.timestamp > end_time:
                    continue
                checkpoints.append(checkpoint)
        
        # 按时间排序
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
        return checkpoints[:limit]
    
    def delete(self, checkpoint_id: str, soft: bool = False) -> bool:
        """从 Redis 删除检查点"""
        if not self.redis:
            return self._fallback.delete(checkpoint_id, soft)
        
        if soft:
            # 软删除：更新状态
            checkpoint = self._retrieve_checkpoint(checkpoint_id)
            if checkpoint:
                checkpoint.status = "deleted"
                self._store_checkpoint(checkpoint)
                self._cache_invalidate(checkpoint_id)
                return True
            return False
        else:
            key = self._key(checkpoint_id)
            result = self.redis.delete(key)
            self._cache_invalidate(checkpoint_id)
            
            with self._metrics_lock:
                self._metrics.total_deletes += 1
            
            return result > 0
    
    def _store_checkpoint(self, checkpoint: EnhancedCheckpoint) -> bool:
        """存储检查点"""
        try:
            checkpoint_json = json.dumps(checkpoint.to_dict(), ensure_ascii=False)
            checkpoint_bytes = checkpoint_json.encode('utf-8')
            
            if self.compression != CompressionType.NONE:
                data, _ = compress_data(checkpoint_bytes, self.compression)
            else:
                data = checkpoint_bytes
            
            key = self._key(checkpoint.checkpoint_id)
            ttl = checkpoint.ttl or self.default_ttl
            self.redis.setex(key, ttl, data)
            return True
        except Exception as e:
            logger.error(f"Failed to store checkpoint: {e}")
            return False
    
    def _retrieve_checkpoint(self, checkpoint_id: str) -> Optional[EnhancedCheckpoint]:
        """检索检查点"""
        key = self._key(checkpoint_id)
        data = self.redis.get(key)
        
        if data:
            try:
                if self.compression != CompressionType.NONE:
                    decompressed = decompress_data(data, self.compression.value)
                else:
                    decompressed = data
                
                checkpoint_dict = json.loads(decompressed.decode('utf-8'))
                return EnhancedCheckpoint.from_dict(checkpoint_dict)
            except Exception as e:
                logger.warning(f"Failed to retrieve checkpoint {checkpoint_id}: {e}")
        return None
    
    def clear_thread(self, thread_id: str) -> int:
        """清除线程的所有检查点"""
        if not self.redis:
            return 0
        
        with self._distributed_lock(thread_id):
            thread_key = self._thread_key(thread_id)
            checkpoint_ids = self.redis.lrange(thread_key, 0, -1)
            
            pipe = self.redis.pipeline()
            count = 0
            
            for cid_bytes in checkpoint_ids:
                cid = cid_bytes.decode('utf-8') if isinstance(cid_bytes, bytes) else cid_bytes
                key = self._key(cid)
                pipe.delete(key)
                count += 1
            
            # 删除所有分支索引
            branch_pattern = f"{self.prefix}branch:{thread_id}:*"
            branch_keys = self.redis.keys(branch_pattern)
            for bk in branch_keys:
                pipe.delete(bk)
            
            # 删除线程索引
            pipe.delete(thread_key)
            
            pipe.execute()
            
            return count
    
    def cleanup_expired(self) -> int:
        """清理过期检查点（Redis 会自动处理 TTL，此方法用于清理软删除的）"""
        count = 0
        
        # 扫描所有检查点
        cursor = 0
        pattern = f"{self.prefix}cp:*"
        
        while True:
            cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
            for key in keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                cid = key_str.replace(f"{self.prefix}cp:", '')
                checkpoint = self._retrieve_checkpoint(cid)
                if checkpoint and checkpoint.status == "deleted":
                    self.delete(cid)
                    count += 1
            if cursor == 0:
                break
        
        return count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        if not self.redis:
            return self._fallback.get_storage_stats() if self._fallback else {}
        
        # 使用 INFO 命令获取 Redis 统计
        info = self.redis.info('memory')
        
        # 统计检查点
        pattern = f"{self.prefix}cp:*"
        cursor = 0
        total_checkpoints = 0
        total_size = 0
        branches = set()
        threads = set()
        
        while True:
            cursor, keys = self.redis.scan(cursor, match=pattern, count=100)
            total_checkpoints += len(keys)
            for key in keys:
                size = self.redis.memory_usage(key) or 0
                total_size += size
            if cursor == 0:
                break
        
        # 获取分支和线程
        thread_pattern = f"{self.prefix}thread:*"
        thread_keys = self.redis.keys(thread_pattern)
        for tk in thread_keys:
            tk_str = tk.decode('utf-8') if isinstance(tk, bytes) else tk
            thread_id = tk_str.replace(f"{self.prefix}thread:", '')
            threads.add(thread_id)
        
        branch_pattern = f"{self.prefix}branch:*"
        branch_keys = self.redis.keys(branch_pattern)
        for bk in branch_keys:
            bk_str = bk.decode('utf-8') if isinstance(bk, bytes) else bk
            parts = bk_str.replace(f"{self.prefix}branch:", '').split(':')
            if len(parts) >= 2:
                branches.add(parts[1])
        
        return {
            "total_checkpoints": total_checkpoints,
            "total_size_bytes": total_size,
            "redis_used_memory": info.get('used_memory', 0),
            "redis_used_memory_human": info.get('used_memory_human', ''),
            "branches": list(branches),
            "threads": list(threads),
            "cache_size": len(self._cache)
        }
    
    # ==================== 异步方法 ====================
    
    async def asave(self, state: AgentState, tags: List[str] = None,
                   branch: str = "main", ttl: int = None) -> str:
        """异步保存状态"""
        # 使用线程池执行同步方法
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.save(state, tags, branch, ttl)
        )
    
    async def aload(self, checkpoint_id: str) -> Optional[AgentState]:
        """异步加载检查点"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.load(checkpoint_id))


class SQLiteCheckpointer(Checkpointer):
    """SQLite 检查点
    
    将检查点保存在 SQLite 数据库中（适用于单机生产环境）。
    
    特性：
    - 持久化存储
    - ACID 事务
    - 数据压缩
    - 全文搜索支持
    - 自动清理
    """
    
    def __init__(self,
                 db_path: str = "checkpoints.db",
                 compression: CompressionType = CompressionType.GZIP,
                 enable_cache: bool = True,
                 cache_size: int = 100,
                 max_checkpoints_per_thread: int = 100):
        super().__init__(
            compression=compression,
            enable_cache=enable_cache,
            cache_size=cache_size
        )
        
        self.db_path = db_path
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
        self._local = threading.local()
        
        # 初始化数据库
        self._init_db()
    
    def _get_connection(self):
        """获取线程本地的数据库连接"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            import sqlite3
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self):
        """初始化数据库表"""
        import sqlite3
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查点表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                parent_id TEXT,
                branch TEXT DEFAULT 'main',
                status TEXT DEFAULT 'active',
                version INTEGER DEFAULT 1,
                checksum TEXT,
                compressed INTEGER DEFAULT 0,
                compression_type TEXT DEFAULT 'none',
                original_size INTEGER DEFAULT 0,
                compressed_size INTEGER DEFAULT 0,
                ttl INTEGER,
                data BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 标签表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS checkpoint_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag_name TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(tag_name)
            )
        ''')
        
        # 索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints(thread_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoints_branch ON checkpoints(thread_id, branch)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoints_created ON checkpoints(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_checkpoints_status ON checkpoints(status)')
        
        conn.commit()
        conn.close()
    
    @timed_operation("SQLiteCheckpointer.save")
    def save(self, state: AgentState, tags: List[str] = None,
             branch: str = "main", ttl: int = None) -> str:
        """保存状态到 SQLite"""
        start_time = time.time()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 获取父检查点
        cursor.execute('''
            SELECT checkpoint_id FROM checkpoints 
            WHERE thread_id = ? AND branch = ? AND status = 'active'
            ORDER BY created_at DESC LIMIT 1
        ''', (state.thread_id, branch))
        row = cursor.fetchone()
        parent_id = row['checkpoint_id'] if row else None
        
        # 创建检查点
        checkpoint = self.create_checkpoint(state, parent_id, tags, branch, ttl)
        
        # 序列化并压缩
        state_json = json.dumps(checkpoint.state, ensure_ascii=False)
        state_bytes = state_json.encode('utf-8')
        
        if self.compression != CompressionType.NONE:
            data, compression_type = compress_data(state_bytes, self.compression)
            is_compressed = 1
        else:
            data = state_bytes
            compression_type = "none"
            is_compressed = 0
        
        # 插入检查点
        cursor.execute('''
            INSERT INTO checkpoints (
                checkpoint_id, thread_id, parent_id, branch, status,
                version, checksum, compressed, compression_type,
                original_size, compressed_size, ttl, data
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            checkpoint.checkpoint_id,
            checkpoint.thread_id,
            checkpoint.parent_id,
            branch,
            'active',
            checkpoint.version,
            checkpoint.checksum,
            is_compressed,
            compression_type,
            len(state_bytes),
            len(data),
            ttl,
            data
        ))
        
        # 插入标签
        if tags:
            for tag in tags:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO checkpoint_tags (tag_name, checkpoint_id, description)
                        VALUES (?, ?, ?)
                    ''', (tag, checkpoint.checkpoint_id, ''))
                except Exception:
                    pass
        
        conn.commit()
        
        # 清理旧检查点
        self._cleanup_thread(state.thread_id, branch)
        
        # 添加到缓存
        self._cache_put(checkpoint)
        
        # 记录指标
        elapsed_ms = (time.time() - start_time) * 1000
        self._record_save_metrics(len(data), elapsed_ms)
        
        # 触发回调
        self._trigger_save_callbacks(checkpoint)
        
        logger.debug(f"Saved checkpoint to SQLite: {checkpoint.checkpoint_id}")
        return checkpoint.checkpoint_id
    
    @timed_operation("SQLiteCheckpointer.load")
    def load(self, checkpoint_id: str) -> Optional[AgentState]:
        """从 SQLite 加载检查点"""
        start_time = time.time()
        
        # 先查缓存
        checkpoint = self._cache_get(checkpoint_id)
        if checkpoint:
            return AgentState.from_dict(checkpoint.state)
        
        # 从数据库加载
        checkpoint = self._retrieve_checkpoint(checkpoint_id)
        
        if checkpoint and checkpoint.status == "active":
            # 添加到缓存
            self._cache_put(checkpoint)
            
            # 记录指标
            elapsed_ms = (time.time() - start_time) * 1000
            self._record_load_metrics(checkpoint.original_size, elapsed_ms)
            
            # 触发回调
            self._trigger_load_callbacks(checkpoint)
            
            return AgentState.from_dict(checkpoint.state)
        return None
    
    def get_latest(self, thread_id: str = None, branch: str = None) -> Optional[AgentState]:
        """获取最新状态"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if thread_id and branch:
            cursor.execute('''
                SELECT checkpoint_id FROM checkpoints
                WHERE thread_id = ? AND branch = ? AND status = 'active'
                ORDER BY created_at DESC LIMIT 1
            ''', (thread_id, branch))
        elif thread_id:
            cursor.execute('''
                SELECT checkpoint_id FROM checkpoints
                WHERE thread_id = ? AND status = 'active'
                ORDER BY created_at DESC LIMIT 1
            ''', (thread_id,))
        else:
            cursor.execute('''
                SELECT checkpoint_id FROM checkpoints
                WHERE status = 'active'
                ORDER BY created_at DESC LIMIT 1
            ''')
        
        row = cursor.fetchone()
        if row:
            return self.load(row['checkpoint_id'])
        return None
    
    def list_checkpoints(self, thread_id: str = None, branch: str = None,
                        limit: int = 10, offset: int = 0,
                        tags: List[str] = None,
                        start_time: datetime = None,
                        end_time: datetime = None) -> List[EnhancedCheckpoint]:
        """列出检查点"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 构建查询
        query = "SELECT checkpoint_id FROM checkpoints WHERE status = 'active'"
        params = []
        
        if thread_id:
            query += " AND thread_id = ?"
            params.append(thread_id)
        
        if branch:
            query += " AND branch = ?"
            params.append(branch)
        
        if start_time:
            query += " AND created_at >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND created_at <= ?"
            params.append(end_time.isoformat())
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        
        checkpoints = []
        for row in cursor.fetchall():
            checkpoint = self._retrieve_checkpoint(row['checkpoint_id'])
            if checkpoint:
                # 标签过滤
                if tags:
                    cursor.execute('''
                        SELECT tag_name FROM checkpoint_tags WHERE checkpoint_id = ?
                    ''', (checkpoint.checkpoint_id,))
                    cp_tags = [r['tag_name'] for r in cursor.fetchall()]
                    if not any(tag in cp_tags for tag in tags):
                        continue
                checkpoints.append(checkpoint)
        
        return checkpoints
    
    def delete(self, checkpoint_id: str, soft: bool = False) -> bool:
        """从 SQLite 删除检查点"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if soft:
            cursor.execute('''
                UPDATE checkpoints SET status = 'deleted', updated_at = CURRENT_TIMESTAMP
                WHERE checkpoint_id = ?
            ''', (checkpoint_id,))
        else:
            cursor.execute('DELETE FROM checkpoints WHERE checkpoint_id = ?', (checkpoint_id,))
            cursor.execute('DELETE FROM checkpoint_tags WHERE checkpoint_id = ?', (checkpoint_id,))
        
        conn.commit()
        self._cache_invalidate(checkpoint_id)
        
        with self._metrics_lock:
            self._metrics.total_deletes += 1
        
        return cursor.rowcount > 0
    
    def _store_checkpoint(self, checkpoint: EnhancedCheckpoint) -> bool:
        """存储检查点"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            state_json = json.dumps(checkpoint.state, ensure_ascii=False)
            state_bytes = state_json.encode('utf-8')
            
            if checkpoint.compressed:
                data, _ = compress_data(state_bytes, 
                    CompressionType(checkpoint.compression_type) if checkpoint.compression_type != "none" else CompressionType.GZIP)
            else:
                data = state_bytes
            
            cursor.execute('''
                UPDATE checkpoints SET 
                    status = ?, data = ?, updated_at = CURRENT_TIMESTAMP
                WHERE checkpoint_id = ?
            ''', (checkpoint.status, data, checkpoint.checkpoint_id))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store checkpoint: {e}")
            return False
    
    def _retrieve_checkpoint(self, checkpoint_id: str) -> Optional[EnhancedCheckpoint]:
        """检索检查点"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM checkpoints WHERE checkpoint_id = ?
        ''', (checkpoint_id,))
        
        row = cursor.fetchone()
        if row:
            # 解压数据
            data = row['data']
            if row['compressed']:
                data = decompress_data(data, row['compression_type'])
            
            state_dict = json.loads(data.decode('utf-8'))
            
            # 获取标签
            cursor.execute('''
                SELECT tag_name FROM checkpoint_tags WHERE checkpoint_id = ?
            ''', (checkpoint_id,))
            tags = [r['tag_name'] for r in cursor.fetchall()]
            
            return EnhancedCheckpoint(
                checkpoint_id=row['checkpoint_id'],
                thread_id=row['thread_id'],
                state=state_dict,
                parent_id=row['parent_id'],
                timestamp=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.utcnow(),
                metadata={},
                version=row['version'],
                checksum=row['checksum'],
                compressed=bool(row['compressed']),
                compression_type=row['compression_type'],
                original_size=row['original_size'],
                compressed_size=row['compressed_size'],
                tags=tags,
                branch=row['branch'],
                status=row['status'],
                ttl=row['ttl']
            )
        return None
    
    def _cleanup_thread(self, thread_id: str, branch: str):
        """清理线程的旧检查点"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 获取该线程分支的检查点数量
        cursor.execute('''
            SELECT COUNT(*) as count FROM checkpoints
            WHERE thread_id = ? AND branch = ? AND status = 'active'
        ''', (thread_id, branch))
        
        count = cursor.fetchone()['count']
        
        if count > self.max_checkpoints_per_thread:
            # 删除最旧的
            excess = count - self.max_checkpoints_per_thread
            cursor.execute('''
                DELETE FROM checkpoints WHERE checkpoint_id IN (
                    SELECT checkpoint_id FROM checkpoints
                    WHERE thread_id = ? AND branch = ? AND status = 'active'
                    ORDER BY created_at ASC LIMIT ?
                )
            ''', (thread_id, branch, excess))
            conn.commit()
    
    def cleanup_expired(self) -> int:
        """清理过期检查点"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 清理 TTL 过期的
        cursor.execute('''
            DELETE FROM checkpoints
            WHERE ttl IS NOT NULL AND 
                  datetime(created_at, '+' || ttl || ' seconds') < datetime('now')
        ''')
        
        # 清理软删除的
        cursor.execute('''
            DELETE FROM checkpoints WHERE status = 'deleted'
        ''')
        
        conn.commit()
        return cursor.rowcount
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
                SUM(original_size) as total_size,
                SUM(compressed_size) as compressed_size,
                COUNT(DISTINCT thread_id) as threads,
                COUNT(DISTINCT branch) as branches
            FROM checkpoints
        ''')
        
        row = cursor.fetchone()
        
        return {
            "total_checkpoints": row['total'] or 0,
            "active_checkpoints": row['active'] or 0,
            "total_size_bytes": row['total_size'] or 0,
            "compressed_size_bytes": row['compressed_size'] or 0,
            "compression_ratio": (row['compressed_size'] / row['total_size']) if row['total_size'] else 1.0,
            "threads": row['threads'] or 0,
            "branches": row['branches'] or 0,
            "db_path": self.db_path,
            "cache_size": len(self._cache)
        }
    
    @contextmanager
    def transaction(self):
        """事务上下文"""
        conn = self._get_connection()
        try:
            yield self
            conn.commit()
        except Exception:
            conn.rollback()
            raise


class FileCheckpointer(Checkpointer):
    """文件系统检查点
    
    将检查点保存在文件系统中（适用于简单持久化场景）。
    
    特性：
    - 简单易用
    - 数据压缩
    - JSON 格式可读
    - 按线程组织目录
    """
    
    def __init__(self,
                 base_dir: str = "./checkpoints",
                 compression: CompressionType = CompressionType.GZIP,
                 enable_cache: bool = True,
                 cache_size: int = 50,
                 max_checkpoints_per_thread: int = 50):
        super().__init__(
            compression=compression,
            enable_cache=enable_cache,
            cache_size=cache_size
        )
        
        self.base_dir = base_dir
        self.max_checkpoints_per_thread = max_checkpoints_per_thread
        self._lock = threading.Lock()
        
        # 创建基础目录
        os.makedirs(base_dir, exist_ok=True)
    
    def _thread_dir(self, thread_id: str) -> str:
        """获取线程目录"""
        return os.path.join(self.base_dir, thread_id[:8])  # 使用前8位避免路径过长
    
    def _checkpoint_path(self, thread_id: str, checkpoint_id: str, branch: str = "main") -> str:
        """获取检查点文件路径"""
        thread_dir = self._thread_dir(thread_id)
        os.makedirs(thread_dir, exist_ok=True)
        ext = ".json.gz" if self.compression != CompressionType.NONE else ".json"
        return os.path.join(thread_dir, f"{branch}_{checkpoint_id[:12]}{ext}")
    
    def _index_path(self, thread_id: str, branch: str = "main") -> str:
        """获取索引文件路径"""
        thread_dir = self._thread_dir(thread_id)
        os.makedirs(thread_dir, exist_ok=True)
        return os.path.join(thread_dir, f"_index_{branch}.json")
    
    def _load_index(self, thread_id: str, branch: str = "main") -> List[str]:
        """加载索引"""
        index_path = self._index_path(thread_id, branch)
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_index(self, thread_id: str, branch: str, checkpoint_ids: List[str]):
        """保存索引"""
        index_path = self._index_path(thread_id, branch)
        with open(index_path, 'w') as f:
            json.dump(checkpoint_ids, f)
    
    @timed_operation("FileCheckpointer.save")
    def save(self, state: AgentState, tags: List[str] = None,
             branch: str = "main", ttl: int = None) -> str:
        """保存状态到文件"""
        start_time = time.time()
        
        with self._lock:
            # 加载索引
            index = self._load_index(state.thread_id, branch)
            parent_id = index[-1] if index else None
            
            # 创建检查点
            checkpoint = self.create_checkpoint(state, parent_id, tags, branch, ttl)
            
            # 序列化
            checkpoint_json = json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2)
            checkpoint_bytes = checkpoint_json.encode('utf-8')
            
            # 写入文件
            file_path = self._checkpoint_path(state.thread_id, checkpoint.checkpoint_id, branch)
            
            if self.compression != CompressionType.NONE:
                with gzip.open(file_path, 'wb') as f:
                    f.write(checkpoint_bytes)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(checkpoint_json)
            
            # 更新索引
            index.append(checkpoint.checkpoint_id)
            self._save_index(state.thread_id, branch, index)
            
            # 清理旧检查点
            self._cleanup_thread(state.thread_id, branch, index)
            
            # 添加到缓存
            self._cache_put(checkpoint)
            
            # 记录指标
            elapsed_ms = (time.time() - start_time) * 1000
            file_size = os.path.getsize(file_path)
            self._record_save_metrics(file_size, elapsed_ms)
            
            # 触发回调
            self._trigger_save_callbacks(checkpoint)
            
            logger.debug(f"Saved checkpoint to file: {file_path}")
            return checkpoint.checkpoint_id
    
    @timed_operation("FileCheckpointer.load")
    def load(self, checkpoint_id: str) -> Optional[AgentState]:
        """从文件加载检查点"""
        start_time = time.time()
        
        # 先查缓存
        checkpoint = self._cache_get(checkpoint_id)
        if checkpoint:
            return AgentState.from_dict(checkpoint.state)
        
        # 搜索文件
        checkpoint = self._retrieve_checkpoint(checkpoint_id)
        
        if checkpoint and checkpoint.status == "active":
            # 添加到缓存
            self._cache_put(checkpoint)
            
            # 记录指标
            elapsed_ms = (time.time() - start_time) * 1000
            self._record_load_metrics(checkpoint.original_size, elapsed_ms)
            
            # 触发回调
            self._trigger_load_callbacks(checkpoint)
            
            return AgentState.from_dict(checkpoint.state)
        return None
    
    def get_latest(self, thread_id: str = None, branch: str = None) -> Optional[AgentState]:
        """获取最新状态"""
        if not thread_id:
            # 需要搜索所有线程目录
            for subdir in os.listdir(self.base_dir):
                subdir_path = os.path.join(self.base_dir, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if file.startswith('_index_'):
                            branch_name = file.replace('_index_', '').replace('.json', '')
                            index_path = os.path.join(subdir_path, file)
                            with open(index_path, 'r') as f:
                                index = json.load(f)
                                if index:
                                    return self.load(index[-1])
            return None
        
        branch = branch or "main"
        index = self._load_index(thread_id, branch)
        if index:
            return self.load(index[-1])
        return None
    
    def list_checkpoints(self, thread_id: str = None, branch: str = None,
                        limit: int = 10, offset: int = 0,
                        tags: List[str] = None,
                        start_time: datetime = None,
                        end_time: datetime = None) -> List[EnhancedCheckpoint]:
        """列出检查点"""
        checkpoints = []
        
        if thread_id:
            branch = branch or "main"
            index = self._load_index(thread_id, branch)
            
            for cid in reversed(index[-(limit + offset):]):
                checkpoint = self._retrieve_checkpoint(cid)
                if checkpoint and checkpoint.status == "active":
                    # 时间过滤
                    if start_time and checkpoint.timestamp < start_time:
                        continue
                    if end_time and checkpoint.timestamp > end_time:
                        continue
                    # 标签过滤
                    if tags and not any(tag in checkpoint.tags for tag in tags):
                        continue
                    checkpoints.append(checkpoint)
                    if len(checkpoints) >= limit:
                        break
        else:
            # 搜索所有
            for subdir in os.listdir(self.base_dir):
                subdir_path = os.path.join(self.base_dir, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if not file.startswith('_index_') and (file.endswith('.json') or file.endswith('.json.gz')):
                            # 解析 checkpoint_id
                            parts = file.replace('.json.gz', '').replace('.json', '').split('_')
                            if len(parts) >= 2:
                                cid = parts[1]
                                checkpoint = self._retrieve_checkpoint(cid)
                                if checkpoint:
                                    checkpoints.append(checkpoint)
        
        # 按时间排序
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
        return checkpoints[offset:offset + limit]
    
    def delete(self, checkpoint_id: str, soft: bool = False) -> bool:
        """删除检查点文件"""
        with self._lock:
            # 搜索并删除文件
            for subdir in os.listdir(self.base_dir):
                subdir_path = os.path.join(self.base_dir, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if checkpoint_id[:12] in file:
                            file_path = os.path.join(subdir_path, file)
                            if soft:
                                # 软删除：重命名
                                os.rename(file_path, file_path + ".deleted")
                            else:
                                os.remove(file_path)
                            
                            self._cache_invalidate(checkpoint_id)
                            
                            with self._metrics_lock:
                                self._metrics.total_deletes += 1
                            
                            return True
        return False
    
    def _store_checkpoint(self, checkpoint: EnhancedCheckpoint) -> bool:
        """存储检查点"""
        try:
            file_path = self._checkpoint_path(checkpoint.thread_id, checkpoint.checkpoint_id, checkpoint.branch)
            checkpoint_json = json.dumps(checkpoint.to_dict(), ensure_ascii=False, indent=2)
            
            if self.compression != CompressionType.NONE:
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    f.write(checkpoint_json)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(checkpoint_json)
            return True
        except Exception as e:
            logger.error(f"Failed to store checkpoint: {e}")
            return False
    
    def _retrieve_checkpoint(self, checkpoint_id: str) -> Optional[EnhancedCheckpoint]:
        """检索检查点"""
        # 搜索文件
        for subdir in os.listdir(self.base_dir):
            subdir_path = os.path.join(self.base_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if checkpoint_id[:12] in file and not file.endswith('.deleted'):
                        file_path = os.path.join(subdir_path, file)
                        try:
                            if file.endswith('.gz'):
                                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                                    data = json.load(f)
                            else:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    data = json.load(f)
                            return EnhancedCheckpoint.from_dict(data)
                        except Exception as e:
                            logger.warning(f"Failed to load checkpoint from {file_path}: {e}")
        return None
    
    def _cleanup_thread(self, thread_id: str, branch: str, index: List[str]):
        """清理线程的旧检查点"""
        if len(index) > self.max_checkpoints_per_thread:
            excess = index[:-self.max_checkpoints_per_thread]
            for cid in excess:
                self.delete(cid)
            # 更新索引
            new_index = index[-self.max_checkpoints_per_thread:]
            self._save_index(thread_id, branch, new_index)
    
    def cleanup_expired(self) -> int:
        """清理过期和已删除的文件"""
        count = 0
        
        for subdir in os.listdir(self.base_dir):
            subdir_path = os.path.join(self.base_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    
                    # 删除 .deleted 文件
                    if file.endswith('.deleted'):
                        os.remove(file_path)
                        count += 1
                        continue
                    
                    # 检查 TTL
                    if not file.startswith('_index_'):
                        try:
                            checkpoint = self._retrieve_checkpoint(file.split('_')[1][:12])
                            if checkpoint and checkpoint.ttl:
                                expiry = checkpoint.timestamp + timedelta(seconds=checkpoint.ttl)
                                if datetime.utcnow() > expiry:
                                    os.remove(file_path)
                                    count += 1
                        except:
                            pass
        
        return count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        total_size = 0
        total_files = 0
        threads = set()
        branches = set()
        
        for subdir in os.listdir(self.base_dir):
            subdir_path = os.path.join(self.base_dir, subdir)
            if os.path.isdir(subdir_path):
                threads.add(subdir)
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
                        if not file.startswith('_index_'):
                            total_files += 1
                            parts = file.split('_')
                            if parts:
                                branches.add(parts[0])
        
        return {
            "total_checkpoints": total_files,
            "total_size_bytes": total_size,
            "base_dir": self.base_dir,
            "threads": list(threads),
            "branches": list(branches),
            "cache_size": len(self._cache)
        }


# ==================== 便捷函数 ====================

def create_memory_checkpointer(max_checkpoints: int = 100,
                               compression: CompressionType = CompressionType.NONE) -> MemoryCheckpointer:
    """创建内存检查点器
    
    Args:
        max_checkpoints: 最大检查点数量
        compression: 压缩类型
    
    Returns:
        内存检查点器实例
    """
    return MemoryCheckpointer(
        max_checkpoints=max_checkpoints,
        compression=compression
    )


def create_redis_checkpointer(host: str = "localhost",
                              port: int = 6379,
                              password: str = None,
                              prefix: str = "langgraph:checkpoint:",
                              default_ttl: int = 86400 * 7,
                              compression: CompressionType = CompressionType.GZIP) -> RedisCheckpointer:
    """创建 Redis 检查点器
    
    Args:
        host: Redis 主机
        port: Redis 端口
        password: Redis 密码
        prefix: 键前缀
        default_ttl: 默认 TTL
        compression: 压缩类型
    
    Returns:
        Redis 检查点器实例
    """
    return RedisCheckpointer(
        host=host,
        port=port,
        password=password,
        prefix=prefix,
        default_ttl=default_ttl,
        compression=compression
    )


def create_sqlite_checkpointer(db_path: str = "checkpoints.db",
                               compression: CompressionType = CompressionType.GZIP) -> SQLiteCheckpointer:
    """创建 SQLite 检查点器
    
    Args:
        db_path: 数据库文件路径
        compression: 压缩类型
    
    Returns:
        SQLite 检查点器实例
    """
    return SQLiteCheckpointer(
        db_path=db_path,
        compression=compression
    )


def create_file_checkpointer(base_dir: str = "./checkpoints",
                             compression: CompressionType = CompressionType.GZIP) -> FileCheckpointer:
    """创建文件检查点器
    
    Args:
        base_dir: 基础目录
        compression: 压缩类型
    
    Returns:
        文件检查点器实例
    """
    return FileCheckpointer(
        base_dir=base_dir,
        compression=compression
    )


def get_checkpointer(backend: str = "memory", **kwargs) -> Checkpointer:
    """获取检查点器
    
    根据后端类型创建相应的检查点器。
    
    Args:
        backend: 后端类型（memory, redis, sqlite, file）
        **kwargs: 后端特定参数
    
    Returns:
        检查点器实例
    """
    factories = {
        "memory": create_memory_checkpointer,
        "redis": create_redis_checkpointer,
        "sqlite": create_sqlite_checkpointer,
        "file": create_file_checkpointer
    }
    
    factory = factories.get(backend, create_memory_checkpointer)
    return factory(**kwargs)


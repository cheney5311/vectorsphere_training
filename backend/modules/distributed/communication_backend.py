"""通信后端

支持NCCL、Gloo、MPI等通信协议。
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
import logging

from .cluster_manager import NodeInfo
from .task_scheduler import TrainingTask

logger = logging.getLogger(__name__)


class CommunicationProtocol(Enum):
    """通信协议枚举"""
    TCP = "tcp"
    UDP = "udp"
    RDMA = "rdma"
    INFINIBAND = "infiniband"


class CommunicationBackendType(Enum):
    """通信后端类型枚举"""
    NCCL = "nccl"      # NVIDIA Collective Communications Library
    GLOO = "gloo"      # Facebook的Gloo库
    MPI = "mpi"        # Message Passing Interface
    UCX = "ucx"        # Unified Communication X


@dataclass
class CommunicationConfig:
    """通信配置"""
    backend: CommunicationBackendType = CommunicationBackendType.NCCL
    protocol: CommunicationProtocol = CommunicationProtocol.TCP
    master_addr: str = "localhost"
    master_port: int = 12355
    timeout_seconds: int = 30
    buffer_size: int = 1024 * 1024  # 1MB
    max_retries: int = 3
    compression: bool = False
    encryption: bool = False


class CommunicationBackend:
    """通信后端"""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.is_initialized = False
        self.nodes: List[NodeInfo] = []
        self.rank_mappings: Dict[str, int] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._message_handlers: Dict[str, Callable] = {}
        self._background_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self, nodes: List[NodeInfo], rank_mappings: Dict[str, int]) -> bool:
        """初始化通信后端"""
        try:
            self.nodes = nodes
            self.rank_mappings = rank_mappings
            
            # 根据后端类型初始化
            if self.config.backend == CommunicationBackendType.NCCL:
                await self._init_nccl()
            elif self.config.backend == CommunicationBackendType.GLOO:
                await self._init_gloo()
            elif self.config.backend == CommunicationBackendType.MPI:
                await self._init_mpi()
            elif self.config.backend == CommunicationBackendType.UCX:
                await self._init_ucx()
            
            # 启动后台任务
            self._running = True
            self._background_task = asyncio.create_task(self._message_loop())
            
            self.is_initialized = True
            logger.info(f"Communication backend {self.config.backend.value} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize communication backend: {e}")
            return False
    
    async def _init_nccl(self):
        """初始化NCCL后端"""
        logger.info("Initializing NCCL communication backend")
        # 这里应该实现NCCL初始化逻辑
        # 为简化起见，使用模拟实现
        await asyncio.sleep(0.1)
    
    async def _init_gloo(self):
        """初始化Gloo后端"""
        logger.info("Initializing Gloo communication backend")
        # 这里应该实现Gloo初始化逻辑
        # 为简化起见，使用模拟实现
        await asyncio.sleep(0.1)
    
    async def _init_mpi(self):
        """初始化MPI后端"""
        logger.info("Initializing MPI communication backend")
        # 这里应该实现MPI初始化逻辑
        # 为简化起见，使用模拟实现
        await asyncio.sleep(0.1)
    
    async def _init_ucx(self):
        """初始化UCX后端"""
        logger.info("Initializing UCX communication backend")
        # 这里应该实现UCX初始化逻辑
        # 为简化起见，使用模拟实现
        await asyncio.sleep(0.1)
    
    async def _message_loop(self):
        """消息处理循环"""
        while self._running:
            try:
                # 处理消息队列
                if not self._message_queue.empty():
                    message = await self._message_queue.get()
                    await self._process_message(message)
                
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Message loop error: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message: Dict[str, Any]):
        """处理消息"""
        message_type = message.get("type")
        if message_type in self._message_handlers:
            try:
                await self._message_handlers[message_type](message)
            except Exception as e:
                logger.error(f"Error processing message type {message_type}: {e}")
        else:
            logger.warning(f"No handler for message type: {message_type}")
    
    async def send_message(self, target_rank: int, message: Dict[str, Any]) -> bool:
        """发送消息"""
        if not self.is_initialized:
            logger.error("Communication backend not initialized")
            return False
        
        try:
            # 添加时间戳和发送者信息
            message_with_metadata = {
                "timestamp": time.time(),
                "sender_rank": self.rank_mappings.get("local", 0),
                "target_rank": target_rank,
                **message
            }
            
            # 这里应该实现实际的消息发送逻辑
            # 为简化起见，将消息放入队列模拟发送
            await self._message_queue.put(message_with_metadata)
            
            logger.debug(f"Message sent to rank {target_rank}: {message.get('type', 'unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to rank {target_rank}: {e}")
            return False
    
    async def broadcast_message(self, message: Dict[str, Any]) -> bool:
        """广播消息"""
        if not self.is_initialized:
            logger.error("Communication backend not initialized")
            return False
        
        try:
            success_count = 0
            for node in self.nodes:
                node_rank = self.rank_mappings.get(node.node_id, -1)
                if node_rank >= 0 and node_rank != self.rank_mappings.get("local", 0):
                    if await self.send_message(node_rank, message):
                        success_count += 1
            
            logger.debug(f"Broadcast message to {success_count}/{len(self.nodes)} nodes")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return False
    
    async def register_message_handler(self, message_type: str, handler: Callable):
        """注册消息处理器"""
        self._message_handlers[message_type] = handler
        logger.debug(f"Registered message handler for type: {message_type}")
    
    async def unregister_message_handler(self, message_type: str):
        """注销消息处理器"""
        if message_type in self._message_handlers:
            del self._message_handlers[message_type]
            logger.debug(f"Unregistered message handler for type: {message_type}")
    
    async def all_reduce(self, data: Any, operation: str = "sum") -> Any:
        """全规约操作"""
        if not self.is_initialized:
            logger.error("Communication backend not initialized")
            return None
        
        try:
            # 这里应该实现全规约操作
            # 为简化起见，返回原始数据
            logger.debug(f"All-reduce operation: {operation}")
            return data
            
        except Exception as e:
            logger.error(f"All-reduce operation failed: {e}")
            return None
    
    async def all_gather(self, data: Any) -> List[Any]:
        """全收集操作"""
        if not self.is_initialized:
            logger.error("Communication backend not initialized")
            return []
        
        try:
            # 这里应该实现全收集操作
            # 为简化起见，返回包含原始数据的列表
            logger.debug("All-gather operation")
            return [data] * len(self.nodes)
            
        except Exception as e:
            logger.error(f"All-gather operation failed: {e}")
            return []
    
    async def barrier(self) -> bool:
        """屏障同步"""
        if not self.is_initialized:
            logger.error("Communication backend not initialized")
            return False
        
        try:
            # 这里应该实现屏障同步
            # 为简化起见，直接返回成功
            logger.debug("Barrier synchronization")
            return True
            
        except Exception as e:
            logger.error(f"Barrier synchronization failed: {e}")
            return False
    
    async def get_communication_stats(self) -> Dict[str, Any]:
        """获取通信统计信息"""
        return {
            "backend": self.config.backend.value,
            "protocol": self.config.protocol.value,
            "initialized": self.is_initialized,
            "nodes_count": len(self.nodes),
            "message_queue_size": self._message_queue.qsize(),
            "registered_handlers": list(self._message_handlers.keys()),
            "config": {
                "master_addr": self.config.master_addr,
                "master_port": self.config.master_port,
                "timeout_seconds": self.config.timeout_seconds,
                "buffer_size": self.config.buffer_size,
                "compression": self.config.compression,
                "encryption": self.config.encryption
            }
        }
    
    async def cleanup(self):
        """清理资源"""
        try:
            self._running = False
            
            if self._background_task:
                self._background_task.cancel()
                try:
                    await self._background_task
                except asyncio.CancelledError:
                    pass
            
            self._message_handlers.clear()
            self.nodes.clear()
            self.rank_mappings.clear()
            
            self.is_initialized = False
            logger.info("Communication backend cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup communication backend: {e}")


# 全局通信后端管理器
class CommunicationBackendManager:
    """通信后端管理器"""
    
    def __init__(self):
        self.backends: Dict[str, CommunicationBackend] = {}
        self._lock = asyncio.Lock()
    
    async def create_backend(self, task_id: str, config: CommunicationConfig,
                           nodes: List[NodeInfo], rank_mappings: Dict[str, int]) -> Optional[CommunicationBackend]:
        """创建通信后端"""
        async with self._lock:
            if task_id in self.backends:
                logger.warning(f"Communication backend for task {task_id} already exists")
                return self.backends[task_id]
            
            backend = CommunicationBackend(config)
            success = await backend.initialize(nodes, rank_mappings)
            
            if success:
                self.backends[task_id] = backend
                logger.info(f"Created communication backend for task: {task_id}")
                return backend
            else:
                logger.error(f"Failed to initialize communication backend for task: {task_id}")
                return None
    
    async def get_backend(self, task_id: str) -> Optional[CommunicationBackend]:
        """获取通信后端"""
        async with self._lock:
            return self.backends.get(task_id)
    
    async def remove_backend(self, task_id: str) -> bool:
        """移除通信后端"""
        async with self._lock:
            if task_id in self.backends:
                backend = self.backends[task_id]
                await backend.cleanup()
                del self.backends[task_id]
                logger.info(f"Removed communication backend for task: {task_id}")
                return True
            return False
    
    async def list_backends(self) -> List[str]:
        """列出所有通信后端"""
        async with self._lock:
            return list(self.backends.keys())


# 全局通信后端管理器实例
_communication_backend_manager: Optional[CommunicationBackendManager] = None


def get_communication_backend_manager() -> CommunicationBackendManager:
    """获取全局通信后端管理器实例"""
    global _communication_backend_manager
    if _communication_backend_manager is None:
        _communication_backend_manager = CommunicationBackendManager()
    return _communication_backend_manager


def set_communication_backend_manager(manager: CommunicationBackendManager):
    """设置全局通信后端管理器实例"""
    global _communication_backend_manager
    _communication_backend_manager = manager
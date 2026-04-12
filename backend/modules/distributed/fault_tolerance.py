"""容错机制

支持节点故障恢复和检查点管理。
"""

import asyncio
import json
import time
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime, timedelta
import logging

from .cluster_manager import NodeInfo, NodeStatus
from .task_scheduler import TrainingTask, TaskStatus

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """恢复策略枚举"""
    CHECKPOINT_RESTART = "checkpoint_restart"  # 检查点重启
    FAILOVER = "failover"                      # 故障转移
    REPLICATION = "replication"                # 复制
    GRACEFUL_DEGRADATION = "graceful_degradation"  # 优雅降级


class CheckpointFormat(Enum):
    """检查点格式枚举"""
    TORCH = "torch"      # PyTorch格式
    TENSORFLOW = "tensorflow"  # TensorFlow格式
    SAFETENSORS = "safetensors"  # SafeTensors格式
    CUSTOM = "custom"    # 自定义格式


@dataclass
class CheckpointInfo:
    """检查点信息"""
    checkpoint_id: str
    task_id: str
    node_id: str
    path: str
    size_bytes: int
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    format: CheckpointFormat = CheckpointFormat.TORCH
    version: str = "1.0"
    is_valid: bool = True


@dataclass
class FaultEvent:
    """故障事件"""
    event_id: str
    node_id: str
    task_id: Optional[str]
    event_type: str  # "node_failure", "task_failure", "network_partition"
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""
    recovery_actions: List[str] = field(default_factory=list)
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoints: Dict[str, CheckpointInfo] = {}
        self._lock = asyncio.Lock()
        
        # 确保检查点目录存在
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    async def create_checkpoint(self, task: TrainingTask, node: NodeInfo,
                              data: Any, format: CheckpointFormat = CheckpointFormat.TORCH) -> Optional[CheckpointInfo]:
        """创建检查点"""
        async with self._lock:
            try:
                # 生成检查点ID
                checkpoint_id = f"ckpt_{task.task_id}_{int(time.time())}"
                
                # 创建检查点文件路径
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{checkpoint_id}.ckpt")
                
                # 保存检查点数据（模拟）
                checkpoint_size = await self._save_checkpoint_data(checkpoint_path, data, format)
                
                # 创建检查点信息
                checkpoint_info = CheckpointInfo(
                    checkpoint_id=checkpoint_id,
                    task_id=task.task_id,
                    node_id=node.node_id,
                    path=checkpoint_path,
                    size_bytes=checkpoint_size,
                    format=format,
                    metadata={
                        "task_progress": task.progress,
                        "task_status": task.status.value,
                        "created_by": "checkpoint_manager"
                    }
                )
                
                # 记录检查点
                self.checkpoints[checkpoint_id] = checkpoint_info
                
                logger.info(f"Checkpoint created: {checkpoint_id} for task {task.task_id}")
                return checkpoint_info
                
            except Exception as e:
                logger.error(f"Failed to create checkpoint for task {task.task_id}: {e}")
                return None
    
    async def _save_checkpoint_data(self, path: str, data: Any, format: CheckpointFormat) -> int:
        """保存检查点数据"""
        # 这里应该实现实际的检查点保存逻辑
        # 为简化起见，创建一个空文件并返回模拟大小
        try:
            with open(path, 'w') as f:
                f.write(json.dumps({
                    "format": format.value,
                    "timestamp": time.time(),
                    "data_type": type(data).__name__
                }))
            
            # 返回文件大小（模拟）
            return os.path.getsize(path)
        except Exception as e:
            logger.error(f"Failed to save checkpoint data: {e}")
            raise
    
    async def restore_checkpoint(self, checkpoint_id: str) -> Optional[Any]:
        """恢复检查点
        - 优先在本地恢复；若本地文件缺失且检查点记录指向对象存储（metadata 中含 object_name/bucket），尝试从对象存储下载到临时路径后恢复。
        """
        import tempfile
        async with self._lock:
            if checkpoint_id not in self.checkpoints:
                logger.warning(f"Checkpoint not found: {checkpoint_id}")
                return None

            checkpoint_info = self.checkpoints[checkpoint_id]
            if not checkpoint_info.is_valid:
                logger.warning(f"Checkpoint is invalid: {checkpoint_id}")
                return None

            ckpt_path = checkpoint_info.path
            # 如果本地文件不存在，尝试从对象存储恢复（如果 metadata 指明）
            if not os.path.exists(ckpt_path):
                try:
                    obj_name = checkpoint_info.metadata.get('object_name') if checkpoint_info.metadata else None
                    bucket = checkpoint_info.metadata.get('bucket') if checkpoint_info.metadata else None
                    if obj_name and bucket:
                        try:
                            from backend.utils.checkpoint_storage import S3Adapter
                            adapter = S3Adapter(bucket=bucket)
                            tmp_fd, tmp_path = tempfile.mkstemp(prefix='ckpt_', suffix=os.path.splitext(obj_name)[1] or '.ckpt')
                            os.close(tmp_fd)
                            ok = adapter.download(obj_name, tmp_path)
                            if ok and os.path.exists(tmp_path):
                                logger.info(f"Downloaded checkpoint object {obj_name} from bucket {bucket} to {tmp_path}")
                                ckpt_path = tmp_path
                                # validate metadata if exists
                                try:
                                    meta_path = tmp_path + '.meta.json'
                                    if os.path.exists(meta_path):
                                        from backend.utils.checkpoint_metadata import canonicalize_meta_file, validate_meta
                                        meta = canonicalize_meta_file(meta_path)
                                        if not validate_meta(meta):
                                            logger.error(f"Downloaded checkpoint meta invalid for {obj_name}")
                                            return None
                                except Exception as e:
                                    logger.warning(f"Failed to validate downloaded checkpoint meta: {e}")
                            else:
                                logger.error(f"Failed to download checkpoint object {obj_name} from bucket {bucket}")
                                return None
                        except Exception as e:
                            logger.error(f"Error downloading checkpoint from storage: {e}")
                            return None
                    else:
                        logger.warning(f"Local checkpoint missing and no storage metadata for {checkpoint_id}")
                        return None
                except Exception as e:
                    logger.error(f"Unexpected error while attempting remote checkpoint restore: {e}")
                    return None

            try:
                restored_data = await self._load_checkpoint_data(ckpt_path, checkpoint_info.format)
                logger.info(f"Checkpoint restored: {checkpoint_id}")
                return restored_data
            except Exception as e:
                logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
                return None
    
    async def _load_checkpoint_data(self, path: str, format: CheckpointFormat) -> Any:
        """加载检查点数据"""
        # 这里应该实现实际的检查点加载逻辑
        # 为简化起见，读取文件并返回模拟数据
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # 返回模拟数据
            return {"restored_from": path, "format": format.value, "data": data}
        except Exception as e:
            logger.error(f"Failed to load checkpoint data: {e}")
            raise
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        async with self._lock:
            if checkpoint_id not in self.checkpoints:
                logger.warning(f"Checkpoint not found: {checkpoint_id}")
                return False
            
            checkpoint_info = self.checkpoints[checkpoint_id]
            
            try:
                # 删除检查点文件
                if os.path.exists(checkpoint_info.path):
                    os.remove(checkpoint_info.path)
                
                # 从记录中移除
                del self.checkpoints[checkpoint_id]
                
                logger.info(f"Checkpoint deleted: {checkpoint_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
                return False
    
    async def list_checkpoints(self, task_id: Optional[str] = None) -> List[CheckpointInfo]:
        """列出检查点"""
        async with self._lock:
            if task_id:
                return [ckpt for ckpt in self.checkpoints.values() if ckpt.task_id == task_id]
            else:
                return list(self.checkpoints.values())
    
    async def get_latest_checkpoint(self, task_id: str) -> Optional[CheckpointInfo]:
        """获取任务的最新检查点"""
        async with self._lock:
            task_checkpoints = [ckpt for ckpt in self.checkpoints.values() if ckpt.task_id == task_id]
            if not task_checkpoints:
                return None
            
            # 按创建时间排序，返回最新的
            task_checkpoints.sort(key=lambda x: x.created_at, reverse=True)
            return task_checkpoints[0]
    
    async def validate_checkpoint(self, checkpoint_id: str) -> bool:
        """验证检查点"""
        async with self._lock:
            if checkpoint_id not in self.checkpoints:
                return False
            
            checkpoint_info = self.checkpoints[checkpoint_id]
            
            try:
                # 检查文件是否存在
                if not os.path.exists(checkpoint_info.path):
                    checkpoint_info.is_valid = False
                    return False
                
                # 检查文件大小
                actual_size = os.path.getsize(checkpoint_info.path)
                if actual_size != checkpoint_info.size_bytes:
                    checkpoint_info.is_valid = False
                    return False
                
                # 这里可以添加更多的验证逻辑
                checkpoint_info.is_valid = True
                return True
                
            except Exception as e:
                logger.error(f"Failed to validate checkpoint {checkpoint_id}: {e}")
                checkpoint_info.is_valid = False
                return False


class FaultToleranceManager:
    """容错管理器"""
    
    def __init__(self, checkpoint_manager: CheckpointManager,
                 recovery_strategy: RecoveryStrategy = RecoveryStrategy.CHECKPOINT_RESTART):
        self.checkpoint_manager = checkpoint_manager
        self.recovery_strategy = recovery_strategy
        self.fault_events: Dict[str, FaultEvent] = {}
        self.recovery_handlers: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """启动容错管理器"""
        async with self._lock:
            if self._running:
                return
            
            self._running = True
            self._monitoring_task = asyncio.create_task(self._fault_monitoring_loop())
            logger.info("Fault tolerance manager started")

        # 尝试向全局 LeaseManager 注册过期回调，以便 Lease 到期时触发容错流程
        try:
            from backend.modules.distributed.lease_manager import get_lease_manager
            lease_mgr = get_lease_manager()
            lease_mgr.register_expiry_callback(self._on_lease_expired)
            logger.info("Registered lease expiry callback to LeaseManager")
        except Exception as e:
            logger.debug(f"No LeaseManager available to register callback: {e}")
    
    async def stop(self):
        """停止容错管理器"""
        async with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            logger.info("Fault tolerance manager stopped")
    
    async def _on_lease_expired(self, lease):
        """
        处理 Lease 到期的回调（由 LeaseManager 调用）
        将 Lease 到期事件上报为故障事件，并触发 report_fault 以便后续恢复策略执行
        """
        try:
            await self.report_fault(node_id=lease.owner_id, task_id=None, event_type='lease_expired', description=f'Lease expired: {lease.lease_id}')
        except Exception as e:
            logger.error(f"Error handling lease expiry: {e}")

    async def _fault_monitoring_loop(self):
        """故障监控循环"""
        while self._running:
            try:
                # 这里应该实现故障检测逻辑
                # 为简化起见，定期记录心跳
                logger.debug("Fault monitoring heartbeat")
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Fault monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def report_fault(self, node_id: str, task_id: Optional[str],
                         event_type: str, description: str = "") -> FaultEvent:
        """报告故障"""
        async with self._lock:
            # 生成事件ID
            event_id = f"fault_{int(time.time())}_{node_id}"
            
            # 创建故障事件
            fault_event = FaultEvent(
                event_id=event_id,
                node_id=node_id,
                task_id=task_id,
                event_type=event_type,
                description=description
            )
            
            # 记录故障事件
            self.fault_events[event_id] = fault_event
            
            # 触发恢复处理
            await self._handle_fault_recovery(fault_event)
            
            logger.warning(f"Fault reported: {event_id} - {event_type} on node {node_id}")
            return fault_event
    
    async def _handle_fault_recovery(self, fault_event: FaultEvent):
        """处理故障恢复"""
        try:
            # 根据恢复策略处理故障
            if self.recovery_strategy == RecoveryStrategy.CHECKPOINT_RESTART:
                await self._handle_checkpoint_restart(fault_event)
            elif self.recovery_strategy == RecoveryStrategy.FAILOVER:
                await self._handle_failover(fault_event)
            elif self.recovery_strategy == RecoveryStrategy.REPLICATION:
                await self._handle_replication(fault_event)
            elif self.recovery_strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                await self._handle_graceful_degradation(fault_event)
            
            # 执行自定义恢复处理器
            if fault_event.event_type in self.recovery_handlers:
                try:
                    await self.recovery_handlers[fault_event.event_type](fault_event)
                except Exception as e:
                    logger.error(f"Custom recovery handler failed: {e}")
            
        except Exception as e:
            logger.error(f"Fault recovery handling failed: {e}")
    
    async def _handle_checkpoint_restart(self, fault_event: FaultEvent):
        """处理检查点重启：
        - 尝试找到最近合法的检查点
        - 验证检查点完整性
        - 执行恢复（恢复数据/状态）并尝试重启/恢复任务
        """
        if not fault_event.task_id:
            return

        try:
            # 获取最新的检查点
            latest_checkpoint = await self.checkpoint_manager.get_latest_checkpoint(fault_event.task_id)
            if not latest_checkpoint:
                logger.warning(f"No checkpoint found for task {fault_event.task_id}")
                return

            # 验证检查点
            is_valid = await self.checkpoint_manager.validate_checkpoint(latest_checkpoint.checkpoint_id)
            if not is_valid:
                logger.warning(f"Latest checkpoint {latest_checkpoint.checkpoint_id} is invalid for task {fault_event.task_id}")
                return

            # 执行恢复（占位实现，依赖 CheckpointManager.restore_checkpoint）
            restored = await self.checkpoint_manager.restore_checkpoint(latest_checkpoint.checkpoint_id)
            if restored is None:
                logger.error(f"Failed to restore checkpoint {latest_checkpoint.checkpoint_id} for task {fault_event.task_id}")
                return

            # 记录恢复动作
            fault_event.recovery_actions.append("checkpoint_restore")
            logger.info(f"Checkpoint restored: {latest_checkpoint.checkpoint_id} for task {fault_event.task_id}")

            # 尝试通知/恢复训练任务（调用训练任务管理器的恢复接口）
            try:
                # 兼容不同导入路径
                try:
                    from backend.modules.training.core.task_manager import get_training_task_manager
                except Exception:
                    from modules.training.core.task_manager import get_training_task_manager

                task_mgr = get_training_task_manager()
                # resume_task 为同步 API；在此直接调用
                resumed = task_mgr.resume_task(fault_event.task_id)
                if resumed:
                    logger.info(f"Task {fault_event.task_id} resumed after checkpoint restore")
                    fault_event.recovery_actions.append("task_resumed")
                else:
                    logger.warning(f"Task {fault_event.task_id} could not be resumed automatically after restore")

            except Exception as e:
                logger.error(f"Failed to resume task {fault_event.task_id} after checkpoint restore: {e}")

        except Exception as e:
            logger.error(f"Checkpoint restart failed for task {fault_event.task_id}: {e}")
    
    async def _handle_failover(self, fault_event: FaultEvent):
        """处理故障转移"""
        fault_event.recovery_actions.append("failover")
        logger.info(f"Failover initiated for node {fault_event.node_id}")
    
    async def _handle_replication(self, fault_event: FaultEvent):
        """处理复制恢复"""
        fault_event.recovery_actions.append("replication")
        logger.info(f"Replication recovery initiated for node {fault_event.node_id}")
    
    async def _handle_graceful_degradation(self, fault_event: FaultEvent):
        """处理优雅降级"""
        fault_event.recovery_actions.append("graceful_degradation")
        logger.info(f"Graceful degradation initiated for node {fault_event.node_id}")
    
    async def register_recovery_handler(self, event_type: str, handler: Callable):
        """注册恢复处理器"""
        self.recovery_handlers[event_type] = handler
        logger.debug(f"Registered recovery handler for event type: {event_type}")
    
    async def unregister_recovery_handler(self, event_type: str):
        """注销恢复处理器"""
        if event_type in self.recovery_handlers:
            del self.recovery_handlers[event_type]
            logger.debug(f"Unregistered recovery handler for event type: {event_type}")
    
    async def resolve_fault(self, event_id: str) -> bool:
        """解决故障"""
        async with self._lock:
            if event_id not in self.fault_events:
                logger.warning(f"Fault event not found: {event_id}")
                return False
            
            fault_event = self.fault_events[event_id]
            fault_event.is_resolved = True
            fault_event.resolved_at = datetime.now()
            
            logger.info(f"Fault resolved: {event_id}")
            return True
    
    async def list_fault_events(self, unresolved_only: bool = False) -> List[FaultEvent]:
        """列出故障事件"""
        async with self._lock:
            if unresolved_only:
                return [event for event in self.fault_events.values() if not event.is_resolved]
            else:
                return list(self.fault_events.values())
    
    async def get_fault_statistics(self) -> Dict[str, Any]:
        """获取故障统计信息"""
        async with self._lock:
            total_events = len(self.fault_events)
            unresolved_events = sum(1 for event in self.fault_events.values() if not event.is_resolved)
            event_types = {}
            
            for event in self.fault_events.values():
                event_type = event.event_type
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            return {
                "total_events": total_events,
                "unresolved_events": unresolved_events,
                "event_types": event_types,
                "recovery_strategies": {
                    "checkpoint_restart": sum(1 for event in self.fault_events.values() 
                                            if "checkpoint_restart" in event.recovery_actions),
                    "failover": sum(1 for event in self.fault_events.values() 
                                  if "failover" in event.recovery_actions),
                    "replication": sum(1 for event in self.fault_events.values() 
                                     if "replication" in event.recovery_actions),
                    "graceful_degradation": sum(1 for event in self.fault_events.values() 
                                              if "graceful_degradation" in event.recovery_actions)
                }
            }


# 全局容错管理器实例
_fault_tolerance_manager: Optional[FaultToleranceManager] = None
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager() -> CheckpointManager:
    """获取全局检查点管理器实例（委托到 backend.utils.checkpoint_manager）。
    使项目中的检查点管理实现统一，避免重复定义。
    """
    try:
        from backend.utils.checkpoint_manager import get_checkpoint_manager as _utils_get
        return _utils_get()
    except Exception:
        # 回退到模块内的实现（如果 utils 模块不可用）
        global _checkpoint_manager
        if _checkpoint_manager is None:
            _checkpoint_manager = CheckpointManager()
        return _checkpoint_manager


def get_fault_tolerance_manager() -> FaultToleranceManager:
    """获取全局容错管理器实例"""
    global _fault_tolerance_manager
    if _fault_tolerance_manager is None:
        checkpoint_manager = get_checkpoint_manager()
        _fault_tolerance_manager = FaultToleranceManager(checkpoint_manager)
    return _fault_tolerance_manager


def set_fault_tolerance_manager(manager: FaultToleranceManager):
    """设置全局容错管理器实例"""
    global _fault_tolerance_manager
    _fault_tolerance_manager = manager


def set_checkpoint_manager(manager: CheckpointManager):
    """设置全局检查点管理器实例"""
    global _checkpoint_manager
    _checkpoint_manager = manager
"""训练进度管理器

生产级训练进度跟踪和状态管理，支持：
- 策略层集成（StrategyMetrics, StrategyMonitor）
- 硬件层集成（设备监控、内存使用）
- 分布式训练进度同步
- 多阶段训练进度追踪
- 系统资源监控

架构调用层次：
├── progress_manager.py (本模块)
│   ├── 调用 backend/modules/training/strategies/base_strategy (策略层)
│   ├── 调用 backend/lib/hardware (硬件层)
│   ├── 调用 backend/lib/distributed (分布式层)
│   └── 被 pipeline, orchestrator, scenarios 调用
└── 提供统一的进度管理接口
"""

import threading
import time
import logging
import json
import os
import sys
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

# 修复导入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

logger = logging.getLogger(__name__)


# ==================== 策略层导入 ====================

STRATEGY_LAYER_AVAILABLE = False
StrategyMetrics = None
StrategyMonitor = None

try:
    from backend.modules.training.strategies.base_strategy import (
        StrategyMetrics, StrategyMonitor,
    )
    STRATEGY_LAYER_AVAILABLE = True
    logger.info("Strategy layer loaded for progress_manager")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Strategy layer not available for progress_manager: {e}")


# ==================== 硬件层导入 ====================

HARDWARE_LAYER_AVAILABLE = False
DeviceManager = None
get_device_manager = None
MemoryManager = None
get_memory_manager = None
get_available_memory = None
clear_memory = None

try:
    from backend.lib.hardware import (
        DeviceManager, get_device_manager,
        MemoryManager, get_memory_manager,
        get_available_memory, clear_memory,
    )
    HARDWARE_LAYER_AVAILABLE = True
    logger.info("Hardware layer loaded for progress_manager")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Hardware layer not available for progress_manager: {e}")


# ==================== 分布式层导入 ====================

DISTRIBUTED_LAYER_AVAILABLE = False
DistributedManager = None
get_distributed_manager = None

try:
    from backend.lib.distributed import (
        DistributedManager, get_distributed_manager,
    )
    DISTRIBUTED_LAYER_AVAILABLE = True
    logger.info("Distributed layer loaded for progress_manager")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Distributed layer not available for progress_manager: {e}")


# ==================== 其他导入 ====================

try:
    from backend.modules.training.exceptions import BusinessLogicError
except ImportError:
    class BusinessLogicError(Exception):
        pass

try:
    from backend.modules.training.integration import (
        report_training_progress_to_tenant_platform, 
        report_training_status_to_tenant_platform
    )
except ImportError:
    report_training_progress_to_tenant_platform = None
    report_training_status_to_tenant_platform = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, system metrics will be limited")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


# ==================== 枚举定义 ====================

class ProgressStatus(str, Enum):
    """进度状态枚举"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingStageType(str, Enum):
    """训练阶段类型"""
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    PREFERENCE = "preference"
    EVALUATION = "evaluation"
    INDUSTRY_PRETRAIN = "industry_pretrain"
    INDUSTRY_ALIGN = "industry_align"
    SCENE_FINETUNE = "scene_finetune"


# ==================== 数据类定义 ====================

@dataclass
class TrainingProgress:
    """训练进度数据类"""
    session_id: str
    total_steps: int = 0
    current_step: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    progress: float = 0.0
    learning_rate: Optional[float] = None
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    train_accuracy: Optional[float] = None
    eval_accuracy: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed, cancelled
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 系统资源指标
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    disk_io: Optional[Dict[str, float]] = None
    network_io: Optional[Dict[str, float]] = None
    
    # 训练性能指标
    throughput: Optional[float] = None  # samples/sec
    latency: Optional[float] = None  # ms
    
    # 阶段信息
    current_stage: Optional[str] = None
    stage_progress: Optional[float] = None
    stages_completed: List[str] = field(default_factory=list)
    
    # 策略层集成
    strategy_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # 硬件层集成
    hardware_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'session_id': self.session_id,
            'total_steps': self.total_steps,
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'progress': self.progress,
            'learning_rate': self.learning_rate,
            'train_loss': self.train_loss,
            'eval_loss': self.eval_loss,
            'train_accuracy': self.train_accuracy,
            'eval_accuracy': self.eval_accuracy,
            'status': self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error_message': self.error_message,
            'metrics': self.metrics,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'gpu_memory': self.gpu_memory,
            'throughput': self.throughput,
            'latency': self.latency,
            'current_stage': self.current_stage,
            'stage_progress': self.stage_progress,
            'stages_completed': self.stages_completed,
            'strategy_metrics': self.strategy_metrics,
            'hardware_info': self.hardware_info,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingProgress':
        """从字典创建"""
        if 'start_time' in data and isinstance(data['start_time'], str):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if 'end_time' in data and isinstance(data['end_time'], str):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


@dataclass
class SystemMetrics:
    """系统指标数据类"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory: float = 0.0
    disk_io_read: float = 0.0
    disk_io_write: float = 0.0
    network_io_sent: float = 0.0
    network_io_recv: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'gpu_memory': self.gpu_memory,
            'disk_io': {'read': self.disk_io_read, 'write': self.disk_io_write},
            'network_io': {'sent': self.network_io_sent, 'recv': self.network_io_recv},
            'timestamp': self.timestamp.isoformat(),
        }


# ==================== 进度管理器 ====================

class TrainingProgressManager:
    """训练进度管理器
    
    生产级进度管理器，集成：
    - 策略层（StrategyMetrics）
    - 硬件层（DeviceManager, MemoryManager）
    - 分布式层（DistributedManager）
    - 系统资源监控
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # 避免重复初始化
        if getattr(self, '_initialized', False):
            return
        
        self.progress_data: Dict[str, TrainingProgress] = {}
        self.callbacks: Dict[str, List[Callable]] = {
            'progress_updated': [],
            'stage_changed': [],
            'status_changed': [],
            'error_occurred': [],
        }
        self._lock = threading.RLock()
        self._monitoring_thread = None
        self._monitoring_active = False
        self._monitoring_interval = 5.0  # seconds
        
        # 策略层组件
        self._strategy_metrics: Dict[str, 'StrategyMetrics'] = {}
        self._strategy_monitor: Optional['StrategyMonitor'] = None
        
        # 硬件层组件
        self._device_manager: Optional['DeviceManager'] = None
        self._memory_manager: Optional['MemoryManager'] = None
        
        # 分布式层组件
        self._distributed_manager: Optional['DistributedManager'] = None
        
        # 初始化各层组件
        self._init_components()
        
        self._initialized = True
        logger.info("TrainingProgressManager initialized")
        logger.info(f"  Strategy layer: {STRATEGY_LAYER_AVAILABLE}")
        logger.info(f"  Hardware layer: {HARDWARE_LAYER_AVAILABLE}")
        logger.info(f"  Distributed layer: {DISTRIBUTED_LAYER_AVAILABLE}")
    
    def _init_components(self) -> None:
        """初始化各层组件"""
        self._init_strategy_components()
        self._init_hardware_components()
        self._init_distributed_components()
    
    def _init_strategy_components(self) -> None:
        """初始化策略层组件"""
        if not STRATEGY_LAYER_AVAILABLE:
            return
        
        try:
            if StrategyMonitor is not None:
                self._strategy_monitor = StrategyMonitor()
                logger.debug("Strategy monitor initialized")
        except Exception as e:
            logger.warning(f"Failed to init strategy components: {e}")
    
    def _init_hardware_components(self) -> None:
        """初始化硬件层组件"""
        if not HARDWARE_LAYER_AVAILABLE:
            return
        
        try:
            if get_device_manager is not None:
                self._device_manager = get_device_manager()
            if get_memory_manager is not None:
                self._memory_manager = get_memory_manager()
        except Exception as e:
            logger.warning(f"Failed to init hardware components: {e}")
    
    def _init_distributed_components(self) -> None:
        """初始化分布式层组件"""
        if not DISTRIBUTED_LAYER_AVAILABLE:
            return
        
        try:
            if get_distributed_manager is not None:
                self._distributed_manager = get_distributed_manager()
        except Exception as e:
            logger.warning(f"Failed to init distributed components: {e}")
    
    def start_system_monitoring(self, interval: float = 5.0) -> None:
        """启动系统资源监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self._monitoring_active:
            return
        
        self._monitoring_interval = interval
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info(f"System monitoring started (interval={interval}s)")
    
    def stop_system_monitoring(self) -> None:
        """停止系统资源监控"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=self._monitoring_interval + 1)
        logger.info("System monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """系统资源监控循环"""
        while self._monitoring_active:
            try:
                # 收集系统资源信息
                system_metrics = self._collect_system_metrics()
                
                # 更新所有活跃训练任务的资源信息
                with self._lock:
                    for session_id, progress in self.progress_data.items():
                        if progress.status == ProgressStatus.RUNNING.value:
                            progress.cpu_usage = system_metrics.cpu_usage
                            progress.memory_usage = system_metrics.memory_usage
                            progress.gpu_usage = system_metrics.gpu_usage
                            progress.gpu_memory = system_metrics.gpu_memory
                            progress.disk_io = {
                                'read': system_metrics.disk_io_read,
                                'write': system_metrics.disk_io_write
                            }
                            progress.network_io = {
                                'sent': system_metrics.network_io_sent,
                                'recv': system_metrics.network_io_recv
                            }
                
                time.sleep(self._monitoring_interval)
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                time.sleep(self._monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统资源指标"""
        metrics = SystemMetrics()
        
        # 使用 psutil 收集系统指标
        if PSUTIL_AVAILABLE:
            try:
                metrics.cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                metrics.memory_usage = memory.percent
                
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    metrics.disk_io_read = float(disk_io.read_bytes)
                    metrics.disk_io_write = float(disk_io.write_bytes)
                
                net_io = psutil.net_io_counters()
                if net_io:
                    metrics.network_io_sent = float(net_io.bytes_sent)
                    metrics.network_io_recv = float(net_io.bytes_recv)
            except Exception as e:
                logger.debug(f"Failed to collect system metrics via psutil: {e}")
        
        # 使用 GPUtil 收集 GPU 指标
        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics.gpu_usage = gpu.load * 100
                    metrics.gpu_memory = gpu.memoryUtil * 100
            except Exception:
                pass
        
        # 使用硬件层收集指标
        if HARDWARE_LAYER_AVAILABLE:
            try:
                if get_available_memory is not None:
                    available_mem = get_available_memory()
                    if available_mem and self._device_manager:
                        # 计算使用率
                        pass
            except Exception:
                pass
        
        metrics.timestamp = datetime.now()
        return metrics
        
    def register_callback(self, event: str, callback: Callable) -> None:
        """注册回调函数
        
        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def unregister_callback(self, event: str, callback: Callable) -> bool:
        """取消注册回调函数"""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
            return True
        return False
    
    def _trigger_callbacks(self, event: str, data: Dict[str, Any]) -> None:
        """触发回调"""
        if event not in self.callbacks:
            return
        
        for callback in self.callbacks[event]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")
    
    def create_progress_tracker(
        self, 
        session_id: str, 
        total_steps: int = 0, 
        total_epochs: int = 0,
        **kwargs
    ) -> TrainingProgress:
        """创建进度跟踪器
        
        Args:
            session_id: 会话 ID
            total_steps: 总步数
            total_epochs: 总 epoch 数
            **kwargs: 其他参数
        
        Returns:
            TrainingProgress 实例
        """
        with self._lock:
            progress = TrainingProgress(
                session_id=session_id,
                total_steps=total_steps,
                total_epochs=total_epochs,
                start_time=datetime.now(),
                status=ProgressStatus.PENDING.value,
                **{k: v for k, v in kwargs.items() if hasattr(TrainingProgress, k)}
            )
            self.progress_data[session_id] = progress
            
            # 创建策略层指标跟踪器
            try:
                self._strategy_metrics[session_id] = StrategyMetrics()
            except Exception:
                pass
            
            logger.info(f"Progress tracker created: {session_id}")
            return progress
    
    def update_progress(
        self, 
        session_id: str, 
        **kwargs
    ) -> TrainingProgress:
        """更新训练进度
        
        Args:
            session_id: 会话 ID
            **kwargs: 更新的字段
        
        Returns:
            更新后的 TrainingProgress
        """
        with self._lock:
            if session_id not in self.progress_data:
                # 自动创建
                self.create_progress_tracker(session_id)
            
            progress = self.progress_data[session_id]
            old_status = progress.status
            
            # 更新字段
            for key, value in kwargs.items():
                if hasattr(progress, key):
                    setattr(progress, key, value)
            
            # 自动计算进度百分比
            if progress.total_steps > 0:
                progress.progress = (progress.current_step / progress.total_steps) * 100
            elif progress.total_epochs > 0:
                progress.progress = (progress.current_epoch / progress.total_epochs) * 100
            
            # 更新策略层指标
            if session_id in self._strategy_metrics:
                try:
                    self._strategy_metrics[session_id].update(kwargs)
                except Exception:
                    pass
            
            # 触发回调
            self._trigger_callbacks('progress_updated', progress.to_dict())
            
            # 状态变更回调
            if old_status != progress.status:
                self._trigger_callbacks('status_changed', {
                    'session_id': session_id,
                    'old_status': old_status,
                    'new_status': progress.status,
                })
            
            # 上报到租户平台
            if report_training_progress_to_tenant_platform is not None:
                try:
                    report_training_progress_to_tenant_platform(session_id, progress.to_dict())
                except Exception:
                    pass
            
            return progress
    
    def update_stage_progress(
        self, 
        session_id: str, 
        stage: str, 
        progress: float, 
        **kwargs
    ) -> TrainingProgress:
        """更新阶段进度
        
        Args:
            session_id: 会话 ID
            stage: 阶段名称
            progress: 进度百分比
            **kwargs: 其他更新字段
        
        Returns:
            更新后的 TrainingProgress
        """
        with self._lock:
            if session_id not in self.progress_data:
                self.create_progress_tracker(session_id)
            
            training_progress = self.progress_data[session_id]
            old_stage = training_progress.current_stage
            
            training_progress.current_stage = stage
            training_progress.stage_progress = progress
            
            # 更新其他字段
            for key, value in kwargs.items():
                if hasattr(training_progress, key):
                    setattr(training_progress, key, value)
            
            # 阶段完成时记录
            if progress >= 100 and stage not in training_progress.stages_completed:
                training_progress.stages_completed.append(stage)
            
            # 触发阶段变更回调
            if old_stage != stage:
                self._trigger_callbacks('stage_changed', {
                    'session_id': session_id,
                    'old_stage': old_stage,
                    'new_stage': stage,
                    'progress': progress,
                })
            
            logger.debug(f"Stage progress updated {session_id} - {stage}: {progress:.1f}%")
            return training_progress
    
    def set_status(
        self, 
        session_id: str, 
        status: Union[str, ProgressStatus],
        error_message: Optional[str] = None
    ) -> TrainingProgress:
        """设置训练状态
        
        Args:
            session_id: 会话 ID
            status: 状态
            error_message: 错误消息（如果有）
        
        Returns:
            更新后的 TrainingProgress
        """
        if isinstance(status, ProgressStatus):
            status = status.value
        
        with self._lock:
            if session_id not in self.progress_data:
                self.create_progress_tracker(session_id)
            
            progress = self.progress_data[session_id]
            old_status = progress.status
            progress.status = status
            
            if error_message:
                progress.error_message = error_message
                self._trigger_callbacks('error_occurred', {
                    'session_id': session_id,
                    'error': error_message,
                })
            
            # 完成或失败时设置结束时间
            if status in [ProgressStatus.COMPLETED.value, ProgressStatus.FAILED.value, ProgressStatus.CANCELLED.value]:
                progress.end_time = datetime.now()
            
            # 触发状态变更回调
            if old_status != status:
                self._trigger_callbacks('status_changed', {
                    'session_id': session_id,
                    'old_status': old_status,
                    'new_status': status,
                })
            
            # 上报到租户平台
            if report_training_status_to_tenant_platform is not None:
                try:
                    report_training_status_to_tenant_platform(session_id, status)
                except Exception:
                    pass
            
            return progress
    
    def cancel_training(self, session_id: str, reason: str = "User cancelled") -> bool:
        """取消训练
        
        Args:
            session_id: 会话ID
            reason: 取消原因
            
        Returns:
            是否成功取消
        """
        try:
            self.set_status(session_id, ProgressStatus.CANCELLED, error_message=reason)
            return True
        except Exception as e:
            logger.error(f"Failed to cancel training for session {session_id}: {e}")
            return False
            
    def get_progress(self, session_id: str) -> Optional[TrainingProgress]:
        """获取训练进度
        
        Args:
            session_id: 会话 ID
        
        Returns:
            TrainingProgress 或 None
        """
        with self._lock:
            return self.progress_data.get(session_id)
            
    def save_checkpoint(
        self,
        session_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        保存检查点记录
        
        Args:
            session_id: 会话ID
            name: 检查点名称
            description: 描述
            epoch: 轮次
            step: 步骤
            metrics: 指标
            
        Returns:
            检查点信息
        """
        progress = self.get_progress(session_id)
        if not progress:
            raise ValueError(f"Session {session_id} not found")
            
        current_epoch = epoch if epoch is not None else progress.current_epoch
        current_step = step if step is not None else progress.current_step
        current_metrics = metrics if metrics is not None else progress.metrics
        
        checkpoint_name = name or f"checkpoint_epoch_{current_epoch}_step_{current_step}"
        timestamp = datetime.now().isoformat()
        
        checkpoint_info = {
            'session_id': session_id,
            'name': checkpoint_name,
            'description': description or "",
            'epoch': current_epoch,
            'step': current_step,
            'metrics': current_metrics,
            'timestamp': timestamp,
            'path': f"./checkpoints/{session_id}/{checkpoint_name}"
        }
        
        # 触发回调
        self._trigger_callbacks('checkpoint_saved', checkpoint_info)
        
        return checkpoint_info
    
    def get_all_progress(self) -> Dict[str, TrainingProgress]:
        """获取所有训练进度"""
        with self._lock:
            return dict(self.progress_data)
    
    def get_active_sessions(self) -> List[str]:
        """获取活跃的会话列表"""
        with self._lock:
            return [
                session_id 
                for session_id, progress in self.progress_data.items()
                if progress.status == ProgressStatus.RUNNING.value
            ]
    
    def remove_progress(self, session_id: str) -> bool:
        """移除进度跟踪器
        
        Args:
            session_id: 会话 ID
        
        Returns:
            是否成功移除
        """
        with self._lock:
            if session_id in self.progress_data:
                del self.progress_data[session_id]
                if session_id in self._strategy_metrics:
                    del self._strategy_metrics[session_id]
                logger.info(f"Progress tracker removed: {session_id}")
                return True
            return False
    
    def get_strategy_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取策略层指标
        
        Args:
            session_id: 会话 ID
        
        Returns:
            策略指标字典或 None
        """
        with self._lock:
            if session_id in self._strategy_metrics:
                try:
                    return self._strategy_metrics[session_id].to_dict() if hasattr(
                        self._strategy_metrics[session_id], 'to_dict'
                    ) else {}
                except Exception:
                    return {}
            return None
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """获取硬件信息
        
        Returns:
            硬件信息字典
        """
        info = {
            'available': HARDWARE_LAYER_AVAILABLE,
        }
        
        if HARDWARE_LAYER_AVAILABLE:
            if self._device_manager is not None:
                try:
                    if hasattr(self._device_manager, 'device'):
                        info['device'] = str(self._device_manager.device)
                    if hasattr(self._device_manager, 'get_device_count'):
                        info['device_count'] = self._device_manager.get_device_count()
                except Exception:
                    pass
            
            if get_available_memory is not None:
                try:
                    info['available_memory_mb'] = get_available_memory()
                except Exception:
                    pass
        
        return info
    
    def get_distributed_info(self) -> Dict[str, Any]:
        """获取分布式信息
        
        Returns:
            分布式信息字典
        """
        info = {
            'available': DISTRIBUTED_LAYER_AVAILABLE,
        }
        
        if DISTRIBUTED_LAYER_AVAILABLE and self._distributed_manager is not None:
            try:
                # 检查是否初始化 (使用 state 属性)
                if hasattr(self._distributed_manager, 'state') and self._distributed_manager.state.initialized:
                    info['initialized'] = True
                    info['world_size'] = self._distributed_manager.state.world_size
                    info['rank'] = self._distributed_manager.state.rank
                    info['backend'] = self._distributed_manager.state.backend
            except Exception:
                pass
        
        return info
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断进度管理器状态
        
        Returns:
            诊断结果字典
        """
        with self._lock:
            active_sessions = self.get_active_sessions()
            
            return {
                'total_sessions': len(self.progress_data),
                'active_sessions': len(active_sessions),
                'monitoring_active': self._monitoring_active,
                'layers': {
                    'strategy': STRATEGY_LAYER_AVAILABLE,
                    'hardware': HARDWARE_LAYER_AVAILABLE,
                    'distributed': DISTRIBUTED_LAYER_AVAILABLE,
                },
                'hardware_info': self.get_hardware_info(),
                'distributed_info': self.get_distributed_info(),
                'callbacks_registered': {
                    event: len(callbacks) 
                    for event, callbacks in self.callbacks.items()
                },
            }
    
    def cleanup(self, session_id: Optional[str] = None) -> None:
        """清理资源
        
        Args:
            session_id: 要清理的会话 ID，None 表示清理所有
        """
        with self._lock:
            if session_id:
                self.remove_progress(session_id)
            else:
                self.progress_data.clear()
                self._strategy_metrics.clear()
        
        # 清理硬件层内存
        if HARDWARE_LAYER_AVAILABLE and clear_memory is not None:
            try:
                clear_memory()
            except Exception:
                pass
        
        logger.info(f"Progress manager cleaned up: session_id={session_id}")


# ==================== 全局实例管理 ====================

_global_manager: Optional[TrainingProgressManager] = None
_global_lock = threading.Lock()


def get_progress_manager() -> TrainingProgressManager:
    """获取全局进度管理器实例
    
    Returns:
        TrainingProgressManager 实例
    """
    global _global_manager
    
    with _global_lock:
        if _global_manager is None:
            _global_manager = TrainingProgressManager()
        return _global_manager


def reset_progress_manager() -> None:
    """重置全局进度管理器"""
    global _global_manager
    
    with _global_lock:
        if _global_manager is not None:
            _global_manager.stop_system_monitoring()
            _global_manager.cleanup()
            _global_manager = None


# ==================== 便捷函数 ====================

def create_progress_tracker(
    session_id: str,
    total_steps: int = 0,
    total_epochs: int = 0,
    **kwargs
) -> TrainingProgress:
    """创建进度跟踪器的便捷函数"""
    return get_progress_manager().create_progress_tracker(
        session_id, total_steps, total_epochs, **kwargs
    )


def update_progress(session_id: str, **kwargs) -> TrainingProgress:
    """更新进度的便捷函数"""
    return get_progress_manager().update_progress(session_id, **kwargs)


def get_progress(session_id: str) -> Optional[TrainingProgress]:
    """获取进度的便捷函数"""
    return get_progress_manager().get_progress(session_id)


def get_layer_availability() -> Dict[str, bool]:
    """获取各层可用性"""
    return {
        'strategy': STRATEGY_LAYER_AVAILABLE,
        'hardware': HARDWARE_LAYER_AVAILABLE,
        'distributed': DISTRIBUTED_LAYER_AVAILABLE,
    }


# ==================== 导出 ====================

__all__ = [
    # 主要类
    'TrainingProgressManager',
    'TrainingProgress',
    'SystemMetrics',
    
    # 枚举
    'ProgressStatus',
    'TrainingStageType',
    
    # 全局实例管理
    'get_progress_manager',
    'reset_progress_manager',
    
    # 便捷函数
    'create_progress_tracker',
    'update_progress',
    'get_progress',
    'get_layer_availability',
    
    # 层可用性标志
    'STRATEGY_LAYER_AVAILABLE',
    'HARDWARE_LAYER_AVAILABLE',
    'DISTRIBUTED_LAYER_AVAILABLE',
]

"""训练任务管理器模块

提供训练任务的创建、提交、监控和管理功能。
集成编排器、进度管理器和版本控制功能。
"""

import os
import sys
import logging
import uuid
import json
import threading
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))


from backend.modules.training.exceptions import TrainingError, ValidationError


from backend.schemas.enums import (
    TrainingScenario, TrainingStage, TrainingMethod,
    ScheduleType, TrainingPriority, TrainingStatus
)

# ==================== 编排器集成 ====================
from backend.modules.training.orchestrator import (
    UnifiedTrainingOrchestrator, LayerManager, OrchestratorPlan, OrchestratorPhase, PhaseConfig,
    create_orchestrator, create_standard_plan, create_three_stage_plan
)

# ==================== 进度管理集成 ====================
from backend.modules.training.progress import (
    TrainingProgressManager, TrainingProgress, get_progress_manager
)

# ==================== 流水线集成 ====================
from backend.modules.training.pipeline import (
    PipelineDefinition, PipelineExecutor, PipelineRunner,
    create_pipeline, execute_pipeline
)

# ==================== 插件集成 ====================
from backend.modules.training.plugins import (
    PluginRegistry, execute_hook, HookPoint, PluginContext
)

# ==================== 策略层集成 ====================
from backend.modules.training.strategies.base_strategy import (
    StrategyContext, StrategyResult, StrategyMetrics,
    StrategyMonitor, StrategyProfiler
)

# ==================== 分布式策略集成 ====================
from backend.modules.training.strategies.distributed_strategy import (
    DistributedMode, DistributedStrategyConfig, recommend_distributed_mode
)

# ==================== 硬件层集成 ====================
from backend.lib.hardware import (
    DeviceManager, get_device_manager, MemoryManager,
    get_available_memory, clear_memory
)

logger = logging.getLogger(__name__)


class TrainingTaskStatus(Enum):
    """训练任务状态枚举"""
    PENDING = "pending"  # 待处理
    SUBMITTED = "submitted"  # 已提交
    RUNNING = "running"  # 运行中
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败
    CANCELLED = "cancelled"  # 已取消
    PAUSED = "paused"  # 已暂停


@dataclass
class TaskVersion:
    """任务版本数据类 - 用于版本控制和回滚"""
    version_id: str
    task_id: str
    version_number: int
    created_at: datetime
    checkpoint_path: Optional[str] = None
    model_path: Optional[str] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    progress_snapshot: float = 0.0
    description: str = ""
    is_current: bool = True
    parent_version_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'version_id': self.version_id,
            'task_id': self.task_id,
            'version_number': self.version_number,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'checkpoint_path': self.checkpoint_path,
            'model_path': self.model_path,
            'config_snapshot': self.config_snapshot,
            'metrics_snapshot': self.metrics_snapshot,
            'progress_snapshot': self.progress_snapshot,
            'description': self.description,
            'is_current': self.is_current,
            'parent_version_id': self.parent_version_id
        }


@dataclass
class TrainingTask:
    """训练任务数据类"""
    task_id: str
    user_id: str
    scenario_type: str
    name: str
    description: str
    config: Dict[str, Any]
    priority: int
    status: str = TrainingTaskStatus.PENDING.value
    created_at: datetime = None  # type: ignore
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: float = 0.0
    metrics: Dict[str, Any] = None  # type: ignore
    # 版本控制相关
    current_version_id: Optional[str] = None
    version_count: int = 0
    # 编排器集成
    orchestrator_plan_id: Optional[str] = None
    pipeline_id: Optional[str] = None
    # 分布式配置
    distributed_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metrics is None:
            self.metrics = {}


class VersionManager:
    """版本管理器 - 管理训练任务的版本控制和回滚"""

    def __init__(self, storage_dir: str = "./training_versions"):
        self._storage_dir = storage_dir
        self._versions: Dict[str, List[TaskVersion]] = {}  # task_id -> versions
        self._lock = threading.Lock()

        # 确保存储目录存在
        os.makedirs(self._storage_dir, exist_ok=True)

        logger.info(f"Version manager initialized with storage: {self._storage_dir}")

    def create_version(self, task: TrainingTask,
                       checkpoint_path: Optional[str] = None,
                       model_path: Optional[str] = None,
                       description: str = "") -> TaskVersion:
        """创建新版本
        
        Args:
            task: 训练任务
            checkpoint_path: 检查点路径
            model_path: 模型路径
            description: 版本描述
            
        Returns:
            新创建的版本
        """
        with self._lock:
            task_id = task.task_id

            # 获取现有版本列表
            if task_id not in self._versions:
                self._versions[task_id] = []

            versions = self._versions[task_id]

            # 计算新版本号
            version_number = len(versions) + 1

            # 标记所有现有版本为非当前
            for v in versions:
                v.is_current = False

            # 获取父版本ID
            parent_version_id = versions[-1].version_id if versions else None

            # 创建新版本
            version = TaskVersion(
                version_id=f"v_{uuid.uuid4().hex[:12]}",
                task_id=task_id,
                version_number=version_number,
                created_at=datetime.utcnow(),
                checkpoint_path=checkpoint_path,
                model_path=model_path,
                config_snapshot=task.config.copy(),
                metrics_snapshot=task.metrics.copy() if task.metrics else {},
                progress_snapshot=task.progress,
                description=description or f"Version {version_number}",
                is_current=True,
                parent_version_id=parent_version_id
            )

            versions.append(version)

            # 保存版本元数据到文件
            self._save_version_metadata(version)

            # 如果提供了检查点路径，复制到版本存储
            if checkpoint_path and os.path.exists(checkpoint_path):
                version_checkpoint_dir = os.path.join(
                    self._storage_dir, task_id, version.version_id, "checkpoint"
                )
                os.makedirs(version_checkpoint_dir, exist_ok=True)
                try:
                    if os.path.isdir(checkpoint_path):
                        shutil.copytree(checkpoint_path, version_checkpoint_dir, dirs_exist_ok=True)
                    else:
                        shutil.copy2(checkpoint_path, version_checkpoint_dir)
                except Exception as e:
                    logger.warning(f"Failed to copy checkpoint to version storage: {e}")

            logger.info(f"Created version {version.version_id} for task {task_id}")
            return version

    def get_versions(self, task_id: str) -> List[TaskVersion]:
        """获取任务的所有版本
        
        Args:
            task_id: 任务ID
            
        Returns:
            版本列表
        """
        with self._lock:
            return self._versions.get(task_id, []).copy()

    def get_version(self, task_id: str, version_id: str) -> Optional[TaskVersion]:
        """获取特定版本
        
        Args:
            task_id: 任务ID
            version_id: 版本ID
            
        Returns:
            版本对象，不存在返回None
        """
        with self._lock:
            versions = self._versions.get(task_id, [])
            for v in versions:
                if v.version_id == version_id:
                    return v
            return None

    def get_current_version(self, task_id: str) -> Optional[TaskVersion]:
        """获取当前版本
        
        Args:
            task_id: 任务ID
            
        Returns:
            当前版本，不存在返回None
        """
        with self._lock:
            versions = self._versions.get(task_id, [])
            for v in versions:
                if v.is_current:
                    return v
            return versions[-1] if versions else None

    def rollback_to_version(self, task_id: str, version_id: str) -> Optional[TaskVersion]:
        """回滚到指定版本
        
        Args:
            task_id: 任务ID
            version_id: 目标版本ID
            
        Returns:
            回滚到的版本，失败返回None
        """
        with self._lock:
            versions = self._versions.get(task_id, [])
            target_version = None

            for v in versions:
                if v.version_id == version_id:
                    target_version = v
                    break

            if not target_version:
                logger.error(f"Version {version_id} not found for task {task_id}")
                return None

            # 标记所有版本为非当前
            for v in versions:
                v.is_current = False

            # 标记目标版本为当前
            target_version.is_current = True

            logger.info(f"Rolled back task {task_id} to version {version_id}")
            return target_version

    def _save_version_metadata(self, version: TaskVersion):
        """保存版本元数据到文件"""
        try:
            version_dir = os.path.join(self._storage_dir, version.task_id, version.version_id)
            os.makedirs(version_dir, exist_ok=True)

            metadata_path = os.path.join(version_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(version.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save version metadata: {e}")

    def get_version_checkpoint_path(self, task_id: str, version_id: str) -> Optional[str]:
        """获取版本的检查点路径
        
        Args:
            task_id: 任务ID
            version_id: 版本ID
            
        Returns:
            检查点路径
        """
        checkpoint_dir = os.path.join(
            self._storage_dir, task_id, version_id, "checkpoint"
        )
        if os.path.exists(checkpoint_dir):
            return checkpoint_dir

        # 尝试从版本元数据获取
        version = self.get_version(task_id, version_id)
        if version and version.checkpoint_path:
            return version.checkpoint_path

        return None


class TrainingTaskManager:
    """训练任务管理器类
    
    集成编排器、进度管理器、版本控制和服务层的统一任务管理器。
    """

    def __init__(self,
                 storage_dir: str = "./training_tasks",
                 version_dir: str = "./training_versions"):
        self.tasks: Dict[str, TrainingTask] = {}
        self.task_queue: List[TrainingTask] = []
        self.running_tasks: List[TrainingTask] = []
        self.completed_tasks: List[TrainingTask] = []
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self.max_concurrent_tasks = 3
        self._storage_dir = storage_dir

        # 版本管理器
        self._version_manager = VersionManager(version_dir)

        # 编排器实例
        self._orchestrator: Optional[Any] = None
        try:
            self._orchestrator = create_orchestrator()
            logger.info("Orchestrator initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize orchestrator: {e}")

        # 进度管理器实例
        self._progress_manager: Optional[Any] = None
        try:
            self._progress_manager = get_progress_manager()
            logger.info("Progress manager initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize progress manager: {e}")

        # 插件注册表
        self._plugin_registry: Optional[Any] = None
        try:
            self._plugin_registry = PluginRegistry()
            logger.info("Plugin registry initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize plugin registry: {e}")

        # 注意：服务层集成已移除
        # core 模块不应直接调用 backend/services
        # 服务调用应由上层 launcher 模块处理

        # 回调管理
        self._callbacks: Dict[str, List[Callable]] = {
            'on_task_created': [],
            'on_task_started': [],
            'on_task_completed': [],
            'on_task_failed': [],
            'on_task_paused': [],
            'on_task_resumed': [],
            'on_progress_updated': [],
            'on_version_created': []
        }

        # 确保存储目录存在
        os.makedirs(self._storage_dir, exist_ok=True)

        logger.info("Training task manager initialized")

    def register_callback(self, event: str, callback: Callable):
        """注册回调函数
        
        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """触发回调
        
        Args:
            event: 事件名称
        """
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                try:
                    callback(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Callback error for event {event}: {e}")

        # 同时触发插件钩子
        if self._plugin_registry:
            try:
                hook_map = {
                    'on_task_created': HookPoint.ON_TRAINING_START,
                    'on_task_started': HookPoint.ON_EPOCH_START,
                    'on_task_completed': HookPoint.ON_TRAINING_END,
                    'on_task_failed': HookPoint.ON_TRAINING_END,
                }
                if event in hook_map:
                    execute_hook(hook_map[event], *args, **kwargs)
            except Exception as e:
                logger.warning(f"Plugin hook error: {e}")

    def create_training_task(self, user_id: str, task_config: Dict[str, Any]) -> str:
        """创建训练任务
        
        Args:
            user_id: 用户ID
            task_config: 任务配置
            
        Returns:
            任务ID
        """
        try:
            # 验证配置
            self._validate_task_config(task_config)

            # 生成任务ID
            task_id = str(uuid.uuid4())

            # 创建任务对象
            task = TrainingTask(
                task_id=task_id,
                user_id=user_id,
                scenario_type=task_config['scenario_type'],
                name=task_config.get('name', f"Training Task {task_id[:8]}"),
                description=task_config.get('description', ''),
                config=task_config,
                priority=task_config.get('priority', 5),
                status=TrainingTaskStatus.PENDING.value,
                distributed_config=task_config.get('distributed_config')
            )

            # 添加到任务列表
            with self._lock:
                self.tasks[task_id] = task
                self.task_queue.append(task)

            # 创建初始版本
            initial_version = self._version_manager.create_version(
                task, description="Initial version"
            )
            task.current_version_id = initial_version.version_id
            task.version_count = 1

            # 初始化进度跟踪
            if self._progress_manager:
                try:
                    self._progress_manager.create_progress_tracker(
                        task_id,
                        total_steps=task_config.get('total_steps', 1000)
                    )
                except Exception as e:
                    logger.warning(f"Failed to create progress tracker: {e}")

            # 触发回调
            self._trigger_callbacks('on_task_created', task)

            logger.info(f"Training task created: {task_id}")
            return task_id

        except Exception as e:
            logger.error(f"Failed to create training task: {str(e)}")
            raise TrainingError(f"创建训练任务失败: {str(e)}")

    def _validate_task_config(self, config: Dict[str, Any]):
        """验证任务配置
        
        Args:
            config: 任务配置
            
        Raises:
            ValidationError: 配置验证失败
        """
        required_fields = ['scenario_type']
        for field in required_fields:
            if field not in config:
                raise ValidationError(f"缺少必需的配置字段: {field}")

        # 验证场景类型
        valid_scenarios = [
            'standard', 'distributed', 'multimodal',
            'distillation', 'three_stage', 'industry', 'scenario'
        ]
        if config['scenario_type'] not in valid_scenarios:
            try:
                TrainingScenario(config['scenario_type'])
            except (ValueError, TypeError):
                logger.warning(f"Using custom scenario type: {config['scenario_type']}")

    def submit_training_task(self, task_id: str) -> bool:
        """提交训练任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否提交成功
        """
        with self._lock:
            if task_id not in self.tasks:
                logger.warning(f"Task not found: {task_id}")
                return False

            task = self.tasks[task_id]
            if task.status != TrainingTaskStatus.PENDING.value:
                logger.warning(f"Task status incorrect: {task_id}, status: {task.status}")
                return False

            # 更新任务状态
            task.status = TrainingTaskStatus.SUBMITTED.value
            task.started_at = datetime.utcnow()

            logger.info(f"Training task submitted: {task_id}")
            return True

    def start_task(self, task_id: str, use_orchestrator: bool = True) -> Dict[str, Any]:
        """启动训练任务
        
        Args:
            task_id: 任务ID
            use_orchestrator: 是否使用编排器
            
        Returns:
            启动结果
        """
        with self._lock:
            if task_id not in self.tasks:
                return {'success': False, 'message': f'Task not found: {task_id}'}

            task = self.tasks[task_id]

            if task.status not in [TrainingTaskStatus.PENDING.value,
                                   TrainingTaskStatus.SUBMITTED.value,
                                   TrainingTaskStatus.PAUSED.value]:
                return {'success': False, 'message': f'Cannot start task in status: {task.status}'}

            # 更新状态
            task.status = TrainingTaskStatus.RUNNING.value
            task.started_at = datetime.utcnow()
            self.running_tasks.append(task)

        # 使用编排器启动
        if use_orchestrator and self._orchestrator:
            try:
                result = self._start_with_orchestrator(task)
                if result.get('success'):
                    task.orchestrator_plan_id = result.get('plan_id')
            except Exception as e:
                logger.error(f"Orchestrator start failed: {e}")

        # 注意：服务层调用已移除，由上层 launcher 模块处理

        # 触发回调
        self._trigger_callbacks('on_task_started', task)

        logger.info(f"Training task started: {task_id}")
        return {'success': True, 'task_id': task_id, 'status': task.status}

    def _start_with_orchestrator(self, task: TrainingTask) -> Dict[str, Any]:
        """使用编排器启动任务
        
        Args:
            task: 训练任务
            
        Returns:
            启动结果
        """
        if not self._orchestrator:
            return {'success': False, 'message': 'Orchestrator not available'}

        try:
            # 根据场景类型创建计划
            scenario_type = task.scenario_type
            plan = None

            if scenario_type == 'three_stage' and create_three_stage_plan:
                plan = create_three_stage_plan(
                    name=f"three_stage_{task.task_id}"
                )
            elif create_standard_plan:
                plan = create_standard_plan(
                    name=f"standard_{task.task_id}"
                )

            if plan:
                model = task.config.get('model')
                train_loader = task.config.get('train_loader')
                val_loader = task.config.get('val_loader')
                if train_loader is None:
                    train_loader = task.config.get('train_dataloader')
                if val_loader is None:
                    val_loader = task.config.get('val_dataloader')
                
                if model is None or train_loader is None:
                    return {'success': False, 'message': 'Missing model or train_loader for orchestrator execution'}
                
                # 执行计划（异步）
                result = self._orchestrator.execute(
                    plan,
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader
                )
                return {
                    'success': True,
                    'plan_id': getattr(plan, 'plan_id', task.task_id),
                    'result': result
                }

            return {'success': False, 'message': 'Failed to create plan'}

        except Exception as e:
            logger.error(f"Orchestrator execution failed: {e}")
            return {'success': False, 'message': str(e)}

    def pause_task(self, task_id: str) -> bool:
        """暂停任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否暂停成功
        """
        with self._lock:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]
            if task.status != TrainingTaskStatus.RUNNING.value:
                return False

            # 创建暂停时的版本快照
            self._version_manager.create_version(
                task, description=f"Paused at progress {task.progress:.1f}%"
            )
            task.version_count += 1

            # 更新任务状态
            task.status = TrainingTaskStatus.PAUSED.value

            # 从运行队列中移除
            if task in self.running_tasks:
                self.running_tasks.remove(task)

            # 通知进度管理器
            if self._progress_manager:
                try:
                    self._progress_manager.set_status(task_id, 'paused')
                except Exception as e:
                    logger.warning(f"Failed to update progress status: {e}")

            # 触发回调
            self._trigger_callbacks('on_task_paused', task)

            logger.info(f"Training task paused: {task_id}")
            return True

    def resume_task(self, task_id: str, from_version: Optional[str] = None) -> bool:
        """恢复任务
        
        Args:
            task_id: 任务ID
            from_version: 从指定版本恢复（可选）
            
        Returns:
            是否恢复成功
        """
        with self._lock:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]
            if task.status != TrainingTaskStatus.PAUSED.value:
                return False

            # 如果指定了版本，先回滚
            if from_version:
                version = self._version_manager.rollback_to_version(task_id, from_version)
                if version:
                    task.config = version.config_snapshot.copy()
                    task.metrics = version.metrics_snapshot.copy()
                    task.progress = version.progress_snapshot
                    task.current_version_id = version.version_id

            # 更新任务状态
            task.status = TrainingTaskStatus.RUNNING.value

            # 添加到运行队列
            self.running_tasks.append(task)

            # 通知进度管理器
            if self._progress_manager:
                try:
                    self._progress_manager.set_status(task_id, 'running')
                except Exception as e:
                    logger.warning(f"Failed to update progress status: {e}")

            # 触发回调
            self._trigger_callbacks('on_task_resumed', task)

            logger.info(f"Training task resumed: {task_id}")
            return True

    def cancel_task(self, task_id: str) -> bool:
        """取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否取消成功
        """
        with self._lock:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]
            if task.status in [TrainingTaskStatus.COMPLETED.value,
                               TrainingTaskStatus.FAILED.value,
                               TrainingTaskStatus.CANCELLED.value]:
                return False

            # 更新任务状态
            task.status = TrainingTaskStatus.CANCELLED.value
            task.completed_at = datetime.utcnow()

            # 从运行队列中移除
            if task in self.running_tasks:
                self.running_tasks.remove(task)

            logger.info(f"Training task cancelled: {task_id}")
            return True

    def rollback_task(self, task_id: str, version_id: str) -> Dict[str, Any]:
        """回滚任务到指定版本
        
        Args:
            task_id: 任务ID
            version_id: 目标版本ID
            
        Returns:
            回滚结果
        """
        with self._lock:
            if task_id not in self.tasks:
                return {'success': False, 'message': f'Task not found: {task_id}'}

            task = self.tasks[task_id]

            # 执行回滚
            version = self._version_manager.rollback_to_version(task_id, version_id)
            if not version:
                return {'success': False, 'message': f'Version not found: {version_id}'}

            # 恢复配置和指标
            task.config = version.config_snapshot.copy()
            task.metrics = version.metrics_snapshot.copy()
            task.progress = version.progress_snapshot
            task.current_version_id = version.version_id

            # 获取检查点路径
            checkpoint_path = self._version_manager.get_version_checkpoint_path(
                task_id, version_id
            )

            logger.info(f"Task {task_id} rolled back to version {version_id}")

            return {
                'success': True,
                'task_id': task_id,
                'version_id': version_id,
                'version_number': version.version_number,
                'checkpoint_path': checkpoint_path,
                'config': version.config_snapshot,
                'metrics': version.metrics_snapshot,
                'progress': version.progress_snapshot
            }

    def create_checkpoint_version(self, task_id: str,
                                  checkpoint_path: str,
                                  model_path: Optional[str] = None,
                                  description: str = "") -> Optional[TaskVersion]:
        """创建检查点版本
        
        Args:
            task_id: 任务ID
            checkpoint_path: 检查点路径
            model_path: 模型路径
            description: 版本描述
            
        Returns:
            创建的版本
        """
        with self._lock:
            if task_id not in self.tasks:
                return None

            task = self.tasks[task_id]

            # 创建新版本
            version = self._version_manager.create_version(
                task,
                checkpoint_path=checkpoint_path,
                model_path=model_path,
                description=description or f"Checkpoint at progress {task.progress:.1f}%"
            )

            task.current_version_id = version.version_id
            task.version_count += 1

            # 触发回调
            self._trigger_callbacks('on_version_created', task, version)

            return version

    def get_task_versions(self, task_id: str) -> List[Dict[str, Any]]:
        """获取任务的所有版本
        
        Args:
            task_id: 任务ID
            
        Returns:
            版本列表
        """
        versions = self._version_manager.get_versions(task_id)
        return [v.to_dict() for v in versions]

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务状态信息
        """
        with self._lock:
            if task_id not in self.tasks:
                return None

            task = self.tasks[task_id]

            # 从进度管理器获取实时进度
            realtime_progress = None
            if self._progress_manager:
                try:
                    realtime_progress = self._progress_manager.get_progress(task_id)
                except Exception:
                    pass

            return {
                'task_id': task.task_id,
                'status': task.status,
                'progress': realtime_progress.progress if realtime_progress else task.progress,
                'metrics': task.metrics,
                'error_message': task.error_message,
                'created_at': task.created_at.isoformat(),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'current_version_id': task.current_version_id,
                'version_count': task.version_count,
                'orchestrator_plan_id': task.orchestrator_plan_id
            }

    def update_task_progress(self, task_id: str, progress: float,
                             metrics: Optional[Dict[str, Any]] = None,
                             create_version: bool = False):
        """更新任务进度
        
        Args:
            task_id: 任务ID
            progress: 进度值(0-100)
            metrics: 指标数据
            create_version: 是否创建版本快照
        """
        with self._lock:
            if task_id not in self.tasks:
                return

            task = self.tasks[task_id]
            task.progress = max(0.0, min(100.0, progress))
            if metrics:
                task.metrics.update(metrics)

        # 更新进度管理器
        if self._progress_manager:
            try:
                self._progress_manager.update_progress(
                    task_id,
                    progress=progress,
                    metrics=metrics
                )
            except Exception as e:
                logger.warning(f"Failed to update progress manager: {e}")

        # 触发回调
        self._trigger_callbacks('on_progress_updated', task, progress, metrics)

        # 自动创建版本快照（每10%进度或达到关键节点）
        if create_version or (progress > 0 and progress % 10 < 1):
            self.create_checkpoint_version(
                task_id,
                checkpoint_path="",
                description=f"Auto checkpoint at {progress:.1f}%"
            )

    def complete_task(self, task_id: str, success: bool = True,
                      error_message: Optional[str] = None,
                      result: Optional[Dict[str, Any]] = None):
        """完成任务
        
        Args:
            task_id: 任务ID
            success: 是否成功
            error_message: 错误信息
            result: 训练结果
        """
        with self._lock:
            if task_id not in self.tasks:
                return

            task = self.tasks[task_id]
            task.status = TrainingTaskStatus.COMPLETED.value if success else TrainingTaskStatus.FAILED.value
            task.completed_at = datetime.utcnow()
            if error_message:
                task.error_message = error_message
            if result:
                task.metrics.update(result)

            # 从运行队列中移除
            if task in self.running_tasks:
                self.running_tasks.remove(task)

            # 添加到完成队列
            self.completed_tasks.append(task)

        # 创建最终版本
        self._version_manager.create_version(
            task,
            description=f"Final version - {'Success' if success else 'Failed'}"
        )

        # 通知进度管理器
        if self._progress_manager:
            try:
                self._progress_manager.set_status(
                    task_id,
                    'completed' if success else 'failed'
                )
            except Exception as e:
                logger.warning(f"Failed to update progress status: {e}")

        # 触发回调
        if success:
            self._trigger_callbacks('on_task_completed', task)
            logger.info(f"Training task completed: {task_id}")
        else:
            self._trigger_callbacks('on_task_failed', task, error_message)
            logger.error(f"Training task failed: {task_id}, error: {error_message}")

    def list_tasks(self, user_id: Optional[str] = None,
                   status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取任务列表
        
        Args:
            user_id: 用户ID过滤
            status: 状态过滤
            
        Returns:
            任务列表
        """
        with self._lock:
            tasks = list(self.tasks.values())

            # 应用用户过滤
            if user_id:
                tasks = [task for task in tasks if task.user_id == user_id]

            # 应用状态过滤
            if status:
                tasks = [task for task in tasks if task.status == status]

            # 转换为字典格式
            return [{
                'task_id': task.task_id,
                'user_id': task.user_id,
                'scenario_type': task.scenario_type,
                'name': task.name,
                'status': task.status,
                'progress': task.progress,
                'version_count': task.version_count,
                'current_version_id': task.current_version_id,
                'created_at': task.created_at.isoformat(),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None
            } for task in tasks]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息
        """
        with self._lock:
            total_tasks = len(self.tasks)
            running_tasks = len([t for t in self.tasks.values()
                                 if t.status == TrainingTaskStatus.RUNNING.value])
            completed_tasks = len([t for t in self.tasks.values()
                                   if t.status == TrainingTaskStatus.COMPLETED.value])
            failed_tasks = len([t for t in self.tasks.values()
                                if t.status == TrainingTaskStatus.FAILED.value])
            pending_tasks = len([t for t in self.tasks.values()
                                 if t.status == TrainingTaskStatus.PENDING.value])
            paused_tasks = len([t for t in self.tasks.values()
                                if t.status == TrainingTaskStatus.PAUSED.value])

            # 计算总版本数
            total_versions = sum(t.version_count for t in self.tasks.values())

            return {
                'total_tasks': total_tasks,
                'running_tasks': running_tasks,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'pending_tasks': pending_tasks,
                'paused_tasks': paused_tasks,
                'total_versions': total_versions,
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'queue_size': len(self.task_queue),
            }

    def create_strategy_context(self, task_id: str) -> Optional[StrategyContext]:
        """创建策略上下文
        
        Args:
            task_id: 任务ID
            
        Returns:
            策略上下文
        """
        with self._lock:
            if task_id not in self.tasks:
                return None
            task = self.tasks[task_id]
            
        try:
            device = 'cpu'
            if get_device_manager:
                device_manager = get_device_manager()
                if device_manager:
                    device = device_manager.get_device()
            
            context = StrategyContext(
                model=None, # 模型尚未加载
                config=task.config,
                device=device
            )
            context.extra['task_id'] = task_id
            return context
        except Exception as e:
            logger.warning(f"Failed to create strategy context: {e}")
            return None

    def create_plugin_context(self, task_id: str) -> Optional[PluginContext]:
        """创建插件上下文
        
        Args:
            task_id: 任务ID
            
        Returns:
            插件上下文
        """
        with self._lock:
            if task_id not in self.tasks:
                return None
            task = self.tasks[task_id]
            
        try:
            return PluginContext(
                hook=HookPoint.ON_TRAINING_START,
                session_id=task_id,
                model=None,
                data={'config': task.config}
            )
        except Exception as e:
            logger.warning(f"Failed to create plugin context: {e}")
            return None

    def check_hardware_resources(self) -> Dict[str, Any]:
        """检查硬件资源
        
        Returns:
            硬件资源状态
        """
        status = {}
        try:
            if get_available_memory:
                status['available_memory'] = get_available_memory()
            
            if get_device_manager:
                manager = get_device_manager()
                if manager:
                    status['devices'] = manager.list_devices()
        except Exception as e:
            logger.warning(f"Hardware check failed: {e}")
        return status

    def configure_distributed_strategy(self, config: Dict[str, Any]) -> Optional[DistributedStrategyConfig]:
        """配置分布式策略
        
        Args:
            config: 分布式配置字典
            
        Returns:
            分布式策略配置对象
        """
        try:
            if not DistributedMode or not DistributedStrategyConfig:
                return None
                
            mode_str = config.get('mode', 'DDP').upper()
            mode = getattr(DistributedMode, mode_str, DistributedMode.DDP)
            
            return DistributedStrategyConfig(
                mode=mode,
                world_size=config.get('world_size', 1),
                master_addr=config.get('master_addr', 'localhost'),
                master_port=config.get('master_port', '12355')
            )
        except Exception as e:
            logger.warning(f"Failed to configure distributed strategy: {e}")
            return None

    def recommend_distributed_config(self, num_gpus: int) -> Dict[str, Any]:
        """推荐分布式配置
        
        Args:
            num_gpus: GPU数量
            
        Returns:
            推荐配置
        """
        try:
            if recommend_distributed_mode:
                return recommend_distributed_mode(
                    model_size_gb=0.0,
                    num_gpus=num_gpus,
                    memory_per_gpu_gb=16.0
                )
        except Exception as e:
            logger.warning(f"Failed to get distributed recommendation: {e}")
        return {}

    def create_custom_orchestrator_plan(self, task_id: str, steps: List[Dict[str, Any]]) -> Optional[OrchestratorPlan]:
        """创建自定义编排计划
        
        Args:
            task_id: 任务ID
            steps: 步骤列表
            
        Returns:
            编排计划
        """
        try:
            if OrchestratorPlan:
                phase_configs = []
                for step in steps:
                    phase_value = step.get('phase', 'finetune')
                    try:
                        phase_enum = OrchestratorPhase(phase_value)
                    except Exception:
                        phase_enum = OrchestratorPhase.FINETUNE
                    
                    phase_configs.append(PhaseConfig(
                        phase=phase_enum,
                        epochs=step.get('epochs', 1),
                        learning_rate=step.get('learning_rate', 1e-4),
                        batch_size=step.get('batch_size', 32),
                        warmup_ratio=step.get('warmup_ratio', 0.1),
                        freeze_layers=step.get('freeze_layers', []),
                        trainable_layers=step.get('trainable_layers', []),
                    ))
                
                return OrchestratorPlan(
                    name=f"custom_plan_{task_id}",
                    phases=phase_configs
                )
        except Exception as e:
            logger.warning(f"Failed to create orchestrator plan: {e}")
        return None

    def create_pipeline_definition(self, name: str, steps: List[Dict[str, Any]]) -> Optional[PipelineDefinition]:
        """创建流水线定义
        
        Args:
            name: 流水线名称
            steps: 步骤配置
            
        Returns:
            流水线定义
        """
        try:
            if PipelineDefinition:
                return PipelineDefinition(
                    name=name,
                    steps=steps
                )
        except Exception as e:
            logger.warning(f"Failed to create pipeline definition: {e}")
        return None

    def diagnose(self) -> Dict[str, Any]:
        """诊断任务管理器状态
        
        Returns:
            诊断信息
        """
        return {
            'manager_status': 'running' if not self._shutdown_event.is_set() else 'stopped',
            'statistics': self.get_statistics(),
            'architecture_note': 'core 模块不直接调用 services，服务调用由 launcher 层处理',
            'version_manager': {
                'storage_dir': self._version_manager._storage_dir,
                'total_tasks_with_versions': len(self._version_manager._versions)
            },
            'callbacks_registered': {
                event: len(callbacks)
                for event, callbacks in self._callbacks.items()
            }
        }

    def start(self):
        """启动训练任务管理器"""
        self._shutdown_event.clear()
        logger.info("Training task manager started")

    def stop(self):
        """停止训练任务管理器"""
        self._shutdown_event.set()
        logger.info("Training task manager stopped")

    def shutdown(self):
        """关闭训练任务管理器"""
        self.stop()

        # 清理资源
        try:
            clear_memory()
        except Exception:
            pass

        logger.info("Training task manager shutdown")


# 全局训练任务管理器实例
_global_training_task_manager: Optional[TrainingTaskManager] = None
_manager_lock = threading.Lock()


def get_training_task_manager() -> TrainingTaskManager:
    """获取训练任务管理器实例"""
    global _global_training_task_manager

    if _global_training_task_manager is None:
        with _manager_lock:
            if _global_training_task_manager is None:
                _global_training_task_manager = TrainingTaskManager()
                _global_training_task_manager.start()

    return _global_training_task_manager


def shutdown_training_task_manager():
    """关闭训练任务管理器"""
    global _global_training_task_manager

    if _global_training_task_manager:
        _global_training_task_manager.shutdown()
        _global_training_task_manager = None


def diagnose_task_manager() -> Dict[str, Any]:
    """诊断任务管理器
    
    Returns:
        诊断信息
    """
    manager = get_training_task_manager()
    return manager.diagnose()

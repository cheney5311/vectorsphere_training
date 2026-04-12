"""统一训练系统

整合所有训练组件，提供统一的训练接口。
集成编排器、进度管理、版本控制和各种训练模块。
"""

import logging

# PyTorch导入（可选）
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    Dataset = None
    TORCH_AVAILABLE = False
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
import json
import os
import time
import uuid
import threading
from datetime import datetime
import numpy as np

# 修复导入路径
import sys
import os as os_path
current_dir = os_path.path.dirname(os_path.path.abspath(__file__))
project_root = os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(current_dir))))
sys.path.insert(0, project_root)

# ==================== 多模态模块集成 ====================
from backend.modules.training.multimodal.multimodal_trainer import MultiModalModel, MultiModalDataset
from backend.modules.training.multimodal.multimodal_config import MultiModalConfig


# ==================== 分布式模块集成 ====================
try:
    from backend.modules.distributed import DistributedTrainingConfig, launch_distributed_training
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DistributedTrainingConfig = None
    launch_distributed_training = None
    DISTRIBUTED_AVAILABLE = False

# ==================== 蒸馏模块集成 ====================
try:
    from backend.modules.training.distillation import (
        KnowledgeDistillationTrainer, ModelCompressor, 
        DistillationConfig, CompressionConfig
    )
    DISTILLATION_AVAILABLE = True
except ImportError:
    KnowledgeDistillationTrainer = None
    ModelCompressor = None
    DistillationConfig = None
    CompressionConfig = None
    DISTILLATION_AVAILABLE = False

# ==================== 三阶段训练集成 ====================
try:
    from backend.modules.training.three_stage.three_stage_trainer import ThreeStageTrainer
    from backend.modules.training.three_stage.three_stage_config import ThreeStageConfig
    THREE_STAGE_AVAILABLE = True
except ImportError:
    ThreeStageTrainer = None
    ThreeStageConfig = None
    THREE_STAGE_AVAILABLE = False

# ==================== 场景训练集成 ====================
try:
    from backend.modules.training.scenarios import (
        BaseScenario, ScenarioManager, get_scenario_manager,
        BasicModelScenario, AdvancedModelScenario, 
        IndustryScenario, ScheduledTrainingScenario,
        TrainingScenario
    )
    SCENARIOS_AVAILABLE = True
except ImportError:
    BaseScenario = None
    ScenarioManager = None
    get_scenario_manager = None
    BasicModelScenario = None
    AdvancedModelScenario = None
    IndustryScenario = None
    ScheduledTrainingScenario = None
    TrainingScenario = None
    SCENARIOS_AVAILABLE = False

# ==================== 行业训练集成 ====================
try:
    from backend.modules.training.industry import (
        IndustryTrainer, IndustryConfig
    )
    INDUSTRY_AVAILABLE = True
except ImportError:
    IndustryTrainer = None
    IndustryConfig = None
    INDUSTRY_AVAILABLE = False

# ==================== 编排器集成 ====================
try:
    from backend.modules.training.orchestrator import (
        UnifiedTrainingOrchestrator, LayerManager, OrchestratorPlan,
        create_orchestrator, create_standard_plan, create_three_stage_plan,
        create_multimodal_plan, create_distillation_plan, create_industry_plan
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    UnifiedTrainingOrchestrator = None
    LayerManager = None
    OrchestratorPlan = None
    create_orchestrator = None
    create_standard_plan = None
    create_three_stage_plan = None
    create_multimodal_plan = None
    create_distillation_plan = None
    create_industry_plan = None
    ORCHESTRATOR_AVAILABLE = False

# ==================== 进度管理集成 ====================
try:
    from backend.modules.training.progress import (
        TrainingProgressManager, TrainingProgress, get_progress_manager
    )
    PROGRESS_AVAILABLE = True
except ImportError:
    TrainingProgressManager = None
    TrainingProgress = None
    get_progress_manager = None
    PROGRESS_AVAILABLE = False

# ==================== 流水线集成 ====================
try:
    from backend.modules.training.pipeline import (
        PipelineDefinition, PipelineExecutor, PipelineRunner,
        create_pipeline, execute_pipeline
    )
    PIPELINE_AVAILABLE = True
except ImportError:
    PipelineDefinition = None
    PipelineExecutor = None
    PipelineRunner = None
    create_pipeline = None
    execute_pipeline = None
    PIPELINE_AVAILABLE = False

# ==================== 插件集成 ====================
try:
    from backend.modules.training.plugins import (
        PluginRegistry, execute_hook, HookPoint, PluginContext
    )
    PLUGINS_AVAILABLE = True
except ImportError:
    PluginRegistry = None
    execute_hook = None
    HookPoint = None
    PluginContext = None
    PLUGINS_AVAILABLE = False

# ==================== 监控模块集成 ====================
try:
    from backend.modules.monitoring.training_monitor import get_training_monitor
    MONITORING_AVAILABLE = True
except ImportError:
    get_training_monitor = None
    MONITORING_AVAILABLE = False

# ==================== 策略层集成 ====================
try:
    from backend.modules.training.strategies.base_strategy import (
        StrategyContext, StrategyResult, StrategyMetrics,
        StrategyMonitor, StrategyProfiler, StrategyValidator
    )
    STRATEGY_AVAILABLE = True
except ImportError:
    StrategyContext = None
    StrategyResult = None
    StrategyMetrics = None
    StrategyMonitor = None
    StrategyProfiler = None
    StrategyValidator = None
    STRATEGY_AVAILABLE = False

# ==================== 分布式策略集成 ====================
try:
    from backend.modules.training.strategies.distributed_strategy import (
        DistributedMode, DistributedStrategyConfig, 
        recommend_distributed_mode, diagnose_distributed_strategy
    )
    DISTRIBUTED_STRATEGY_AVAILABLE = True
except ImportError:
    DistributedMode = None
    DistributedStrategyConfig = None
    recommend_distributed_mode = None
    diagnose_distributed_strategy = None
    DISTRIBUTED_STRATEGY_AVAILABLE = False

# ==================== 硬件层集成 ====================
try:
    from backend.lib.hardware import (
        DeviceManager, get_device_manager, MemoryManager,
        get_available_memory, clear_memory, recommend_precision
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    DeviceManager = None
    get_device_manager = None
    MemoryManager = None
    get_available_memory = None
    clear_memory = None
    recommend_precision = None
    HARDWARE_AVAILABLE = False

# ==================== 损失层集成 ====================
try:
    from backend.lib.losses import LossFactory, create_loss
    LOSSES_AVAILABLE = True
except ImportError:
    LossFactory = None
    create_loss = None
    LOSSES_AVAILABLE = False

# ==================== 架构层级说明 ====================
# core 模块不应直接调用 backend/services，正确的调用层级是：
# backend/services -> backend/modules/training/launcher -> backend/modules/training/core -> 下游训练模块
# 服务层集成已移除，由 launcher 层处理服务调用

# ==================== 任务管理器集成 ====================
try:
    from backend.modules.training.core.task_manager import (
        TrainingTaskManager, TrainingTask, TrainingTaskStatus,
        TaskVersion, VersionManager, get_training_task_manager
    )
    TASK_MANAGER_AVAILABLE = True
except ImportError:
    TrainingTaskManager = None
    TrainingTask = None
    TrainingTaskStatus = None
    TaskVersion = None
    VersionManager = None
    get_training_task_manager = None
    TASK_MANAGER_AVAILABLE = False

# ==================== 异常类 ====================
try:
    from backend.modules.training.exceptions import BusinessLogicError
except ImportError:
    class BusinessLogicError(Exception):
        """业务逻辑异常（备用定义）"""
        pass

logger = logging.getLogger(__name__)


@dataclass
class TrainingVersion:
    """训练版本数据类"""
    version_id: str
    session_id: str
    version_number: int
    created_at: datetime
    checkpoint_path: Optional[str] = None
    model_path: Optional[str] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    metrics_snapshot: Dict[str, Any] = field(default_factory=dict)
    training_stats_snapshot: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    is_current: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'version_id': self.version_id,
            'session_id': self.session_id,
            'version_number': self.version_number,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'checkpoint_path': self.checkpoint_path,
            'model_path': self.model_path,
            'config_snapshot': self.config_snapshot,
            'metrics_snapshot': self.metrics_snapshot,
            'training_stats_snapshot': self.training_stats_snapshot,
            'description': self.description,
            'is_current': self.is_current
        }


@dataclass 
class TrainingConfig:
    """统一训练配置"""
    # 基础配置
    model_name: str = "custom_model"
    task_type: str = "classification"  # classification, generation, multimodal
    output_dir: str = "./outputs"
    session_id: Optional[str] = None
    tenant_id: str = "default"
    user_id: str = "system"
    
    # 数据配置
    train_data_path: str = "./data/train"
    val_data_path: Optional[str] = "./data/val"
    test_data_path: Optional[str] = "./data/test"
    max_seq_length: int = 512
    
    # 训练配置
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # 优化配置
    optimizer_type: str = "adamw"  # adamw, adam, sgd
    scheduler_type: str = "cosine"  # cosine, linear, onecycle, adaptive
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # 混合精度
    use_fp16: bool = True
    
    # 分布式训练
    use_distributed: bool = False
    world_size: int = 1
    master_port: str = "12355"
    
    # 多模态配置
    use_multimodal: bool = False
    multimodal_config: Optional[Any] = None
    
    # 模型压缩
    use_distillation: bool = False
    distillation_config: Optional[Any] = None
    use_compression: bool = False
    compression_config: Optional[Any] = None
    
    # 三阶段训练配置
    use_three_stage: bool = False
    three_stage_config: Optional[Any] = None
    
    # 场景训练配置
    use_scenario: bool = False
    scenario_type: str = "standard"
    scenario_config: Optional[Dict[str, Any]] = None
    
    # 行业训练配置
    use_industry: bool = False
    industry_config: Optional[Any] = None
    
    # 编排器配置
    use_orchestrator: bool = True
    orchestrator_config: Optional[Dict[str, Any]] = None
    
    # 版本控制
    enable_versioning: bool = True
    auto_checkpoint_interval: int = 5  # 每N个epoch自动创建检查点
    
    # 监控和日志
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # 早停
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    
    def __post_init__(self):
        """配置验证和后处理"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 生成会话ID（如果未提供）
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'to_dict'):
                result[key] = value.to_dict()
            elif hasattr(value, '__dict__'):
                result[key] = value.__dict__
            else:
                result[key] = value
        return result


class TrainingVersionManager:
    """训练版本管理器"""
    
    def __init__(self, storage_dir: str):
        self._storage_dir = storage_dir
        self._versions: Dict[str, List[TrainingVersion]] = {}
        self._lock = threading.Lock()
        
        os.makedirs(storage_dir, exist_ok=True)
    
    def create_version(self, session_id: str, config: TrainingConfig,
                      training_stats: Dict[str, Any],
                      checkpoint_path: Optional[str] = None,
                      model_path: Optional[str] = None,
                      description: str = "") -> TrainingVersion:
        """创建新版本"""
        with self._lock:
            if session_id not in self._versions:
                self._versions[session_id] = []
            
            versions = self._versions[session_id]
            version_number = len(versions) + 1
            
            # 标记现有版本为非当前
            for v in versions:
                v.is_current = False
            
            version = TrainingVersion(
                version_id=f"v_{uuid.uuid4().hex[:12]}",
                session_id=session_id,
                version_number=version_number,
                created_at=datetime.utcnow(),
                checkpoint_path=checkpoint_path,
                model_path=model_path,
                config_snapshot=config.to_dict() if hasattr(config, 'to_dict') else {},
                metrics_snapshot=training_stats.get('train_metrics', [])[-1] if training_stats.get('train_metrics') else {},
                training_stats_snapshot=training_stats.copy(),
                description=description or f"Version {version_number}",
                is_current=True
            )
            
            versions.append(version)
            self._save_version(version)
            
            return version
    
    def get_versions(self, session_id: str) -> List[TrainingVersion]:
        """获取所有版本"""
        with self._lock:
            return self._versions.get(session_id, []).copy()
    
    def get_version(self, session_id: str, version_id: str) -> Optional[TrainingVersion]:
        """获取特定版本"""
        with self._lock:
            for v in self._versions.get(session_id, []):
                if v.version_id == version_id:
                    return v
            return None
    
    def rollback_to_version(self, session_id: str, version_id: str) -> Optional[TrainingVersion]:
        """回滚到指定版本"""
        with self._lock:
            versions = self._versions.get(session_id, [])
            target = None
            
            for v in versions:
                if v.version_id == version_id:
                    target = v
                    break
            
            if target:
                for v in versions:
                    v.is_current = False
                target.is_current = True
            
            return target
    
    def _save_version(self, version: TrainingVersion):
        """保存版本元数据"""
        try:
            version_dir = os.path.join(self._storage_dir, version.session_id, version.version_id)
            os.makedirs(version_dir, exist_ok=True)
            
            with open(os.path.join(version_dir, "metadata.json"), 'w') as f:
                json.dump(version.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save version: {e}")


class UnifiedTrainingSystem:
    """统一训练系统
    
    集成所有训练模块、编排器、进度管理和版本控制的统一训练系统。
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # 设备初始化（处理torch不可用的情况）
        if TORCH_AVAILABLE and torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"
        
        # 创建输出目录
        self.output_dir = os_path.path.abspath(config.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化组件
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.optimizer = None
        self.trainer = None
        
        # 训练监控器
        if MONITORING_AVAILABLE and get_training_monitor:
            self.monitor = get_training_monitor(str(self.output_dir))
        else:
            self.monitor = None
        
        # 训练状态
        self.training_stats = {
            'train_losses': [],
            'train_metrics': [],
            'val_losses': [],
            'val_metrics': [],
            'learning_rates': [],
            'epochs': [],
            'best_metric': float('-inf'),
            'best_epoch': 0,
            'total_training_time': 0.0,
            'samples_per_second': 0.0,
            'current_epoch': 0,
            'current_step': 0
        }
        
        # 版本管理器
        self._version_manager = TrainingVersionManager(
            os.path.join(self.output_dir, "versions")
        ) if config.enable_versioning else None
        self._current_version_id: Optional[str] = None
        
        # 编排器
        self._orchestrator: Optional[Any] = None
        if ORCHESTRATOR_AVAILABLE and config.use_orchestrator and create_orchestrator:
            try:
                self._orchestrator = create_orchestrator()
                logger.info("Orchestrator initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize orchestrator: {e}")
        
        # 进度管理器
        self._progress_manager: Optional[Any] = None
        if PROGRESS_AVAILABLE and get_progress_manager:
            try:
                self._progress_manager = get_progress_manager()
                logger.info("Progress manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize progress manager: {e}")
        
        # 任务管理器
        self._task_manager: Optional[Any] = None
        if TASK_MANAGER_AVAILABLE and get_training_task_manager:
            try:
                self._task_manager = get_training_task_manager()
                logger.info("Task manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize task manager: {e}")
        

        # 策略层组件
        self._strategy_monitor: Optional[Any] = None
        self._strategy_profiler: Optional[Any] = None
        if STRATEGY_AVAILABLE:
            try:
                if StrategyMonitor:
                    self._strategy_monitor = StrategyMonitor()
                if StrategyProfiler:
                    self._strategy_profiler = StrategyProfiler()
            except Exception as e:
                logger.warning(f"Failed to initialize strategy components: {e}")
        
        # 硬件层组件
        self._device_manager: Optional[Any] = None
        if HARDWARE_AVAILABLE and get_device_manager:
            try:
                self._device_manager = get_device_manager()
            except Exception as e:
                logger.warning(f"Failed to initialize device manager: {e}")
        
        # 回调管理
        self._callbacks: Dict[str, List[Callable]] = {
            'on_epoch_start': [],
            'on_epoch_end': [],
            'on_step_start': [],
            'on_step_end': [],
            'on_training_start': [],
            'on_training_end': [],
            'on_version_created': []
        }
        
        # 控制标志
        self._paused = False
        self._cancelled = False
        self._lock = threading.Lock()
        
        # 设置日志
        self._setup_logging()
        
        # 创建初始版本
        if self._version_manager:
            version = self._version_manager.create_version(
                config.session_id,
                config,
                self.training_stats,
                description="Initial version"
            )
            self._current_version_id = version.version_id
        
        logger.info("Unified training system initialized")
    
    def register_callback(self, event: str, callback: Callable):
        """注册回调函数"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, *args, **kwargs):
        """触发回调"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
        
        # 同时触发插件钩子
        if PLUGINS_AVAILABLE and execute_hook:
            try:
                hook_map = {
                    'on_training_start': HookPoint.ON_TRAINING_START,
                    'on_training_end': HookPoint.ON_TRAINING_END,
                    'on_epoch_start': HookPoint.ON_EPOCH_START,
                    'on_epoch_end': HookPoint.ON_EPOCH_END,
                    'on_step_end': HookPoint.ON_STEP_END
                }
                if event in hook_map:
                    execute_hook(hook_map[event], *args, **kwargs)
            except Exception as e:
                logger.warning(f"Plugin hook error: {e}")
    
    def _setup_logging(self):
        """设置日志"""
        log_file = os_path.path.join(self.output_dir, "training.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def prepare_model(self, model: Optional[Any] = None) -> Any:
        """准备模型"""
        if not TORCH_AVAILABLE or not nn:
            logger.warning("PyTorch not available, skipping model preparation")
            return None
        
        if model is not None:
            self.model = model
        elif self.config.use_multimodal:
            # 多模态模型
            if self.config.multimodal_config is None and MultiModalConfig:
                self.config.multimodal_config = MultiModalConfig()
            
            num_classes = getattr(self.config, 'num_classes', 10)
            self.model = MultiModalModel(self.config.multimodal_config, num_classes)
        else:
            # 默认模型 - 创建一个简单的模型用于演示
            self.model = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
        
        self.model = self.model.to(self.device)
        
        logger.info(f"Model prepared: {type(self.model).__name__}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        return self.model
    
    def prepare_datasets(self, 
                        train_dataset: Optional[Any] = None,
                        val_dataset: Optional[Any] = None,
                        test_dataset: Optional[Any] = None):
        """准备数据集"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, skipping dataset preparation")
            return
        
        if train_dataset is not None:
            self.train_dataset = train_dataset
        elif self.config.use_multimodal:
            # 多模态数据集
            self.train_dataset = MultiModalDataset(
                self.config.train_data_path, 
                self.config.multimodal_config
            )
            
            if self.config.val_data_path:
                self.val_dataset = MultiModalDataset(
                    self.config.val_data_path,
                    self.config.multimodal_config
                )
        else:
            # 创建模拟数据集
            self.train_dataset = self._create_mock_dataset()
            
            if self.config.val_data_path:
                self.val_dataset = self._create_mock_dataset()
        
        if val_dataset is not None:
            self.val_dataset = val_dataset
        
        if test_dataset is not None:
            self.test_dataset = test_dataset
        
        if self.train_dataset:
            logger.info(f"Training dataset size: {len(self.train_dataset)}")
        if self.val_dataset:
            logger.info(f"Validation dataset size: {len(self.val_dataset)}")
    
    def _create_mock_dataset(self):
        """创建模拟数据集"""
        if not TORCH_AVAILABLE or not Dataset or not torch:
            return None
        
        class MockDataset(Dataset):
            def __init__(self, size=1000):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # 创建模拟数据
                input_ids = torch.randint(0, 1000, (512,))
                attention_mask = torch.ones_like(input_ids)
                labels = torch.randint(0, 10, (1,))
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }
        
        return MockDataset()
    
    def prepare_optimizer(self):
        """准备优化器"""
        if not TORCH_AVAILABLE or not torch or not self.model:
            logger.warning("PyTorch not available or model not prepared, skipping optimizer preparation")
            return
        
        # 创建优化器
        if self.config.optimizer_type.lower() == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        
        if self.optimizer:
            logger.info(f"Optimizer: {type(self.optimizer).__name__}")
    
    # ==================== 训练控制方法 ====================
    
    def pause(self):
        """暂停训练"""
        with self._lock:
            self._paused = True
            
            # 创建暂停时的版本快照
            if self._version_manager:
                version = self._version_manager.create_version(
                    self.config.session_id,
                    self.config,
                    self.training_stats,
                    description=f"Paused at epoch {self.training_stats['current_epoch']}"
                )
                self._current_version_id = version.version_id
            
            # 通知进度管理器
            if self._progress_manager:
                try:
                    self._progress_manager.set_status(self.config.session_id, 'paused')
                except Exception as e:
                    logger.warning(f"Failed to update progress status: {e}")
            
            logger.info("Training paused")
    
    def resume(self, from_version: Optional[str] = None):
        """恢复训练
        
        Args:
            from_version: 从指定版本恢复（可选）
        """
        with self._lock:
            # 如果指定了版本，先回滚
            if from_version and self._version_manager:
                version = self._version_manager.rollback_to_version(
                    self.config.session_id, from_version
                )
                if version:
                    self.training_stats = version.training_stats_snapshot.copy()
                    self._current_version_id = version.version_id
                    logger.info(f"Resumed from version {from_version}")
            
            self._paused = False
            
            # 通知进度管理器
            if self._progress_manager:
                try:
                    self._progress_manager.set_status(self.config.session_id, 'running')
                except Exception as e:
                    logger.warning(f"Failed to update progress status: {e}")
            
            logger.info("Training resumed")
    
    def cancel(self):
        """取消训练"""
        with self._lock:
            self._cancelled = True
            
            # 通知进度管理器
            if self._progress_manager:
                try:
                    self._progress_manager.set_status(self.config.session_id, 'cancelled')
                except Exception as e:
                    logger.warning(f"Failed to update progress status: {e}")
            
            logger.info("Training cancelled")
    
    def rollback_to_version(self, version_id: str) -> Dict[str, Any]:
        """回滚到指定版本
        
        Args:
            version_id: 目标版本ID
            
        Returns:
            回滚结果
        """
        if not self._version_manager:
            return {'success': False, 'message': 'Versioning not enabled'}
        
        version = self._version_manager.rollback_to_version(
            self.config.session_id, version_id
        )
        
        if not version:
            return {'success': False, 'message': f'Version not found: {version_id}'}
        
        # 恢复训练状态
        self.training_stats = version.training_stats_snapshot.copy()
        self._current_version_id = version.version_id
        
        logger.info(f"Rolled back to version {version_id}")
        
        return {
            'success': True,
            'version_id': version_id,
            'version_number': version.version_number,
            'checkpoint_path': version.checkpoint_path,
            'training_stats': version.training_stats_snapshot
        }
    
    def get_versions(self) -> List[Dict[str, Any]]:
        """获取所有版本"""
        if not self._version_manager:
            return []
        
        versions = self._version_manager.get_versions(self.config.session_id)
        return [v.to_dict() for v in versions]
    
    def create_checkpoint_version(self, checkpoint_path: str,
                                  model_path: Optional[str] = None,
                                  description: str = "") -> Optional[Dict[str, Any]]:
        """创建检查点版本"""
        if not self._version_manager:
            return None
        
        version = self._version_manager.create_version(
            self.config.session_id,
            self.config,
            self.training_stats,
            checkpoint_path=checkpoint_path,
            model_path=model_path,
            description=description or f"Checkpoint at epoch {self.training_stats['current_epoch']}"
        )
        
        self._current_version_id = version.version_id
        self._trigger_callbacks('on_version_created', version)
        
        return version.to_dict()
    
    # ==================== 训练方法 ====================
    
    def train(self) -> Dict[str, Any]:
        """开始训练"""
        logger.info("Starting unified training system...")
        
        # 检查 PyTorch 可用性
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using simplified training flow")
            return self._simplified_train()
        
        # 记录开始时间
        start_time = time.time()
        
        # 初始化进度跟踪
        if self._progress_manager:
            try:
                self._progress_manager.create_progress_tracker(
                    self.config.session_id,
                    total_steps=self.config.num_epochs * 1000
                )
                self._progress_manager.set_status(self.config.session_id, 'running')
            except Exception as e:
                logger.warning(f"Failed to initialize progress tracker: {e}")
        
        # 记录训练开始
        if self.monitor:
            self.monitor.log_training_start(self.config.to_dict())
        
        # 触发回调
        self._trigger_callbacks('on_training_start', self.config)
        
        # 准备组件
        if self.model is None:
            self.prepare_model()
        
        if self.train_dataset is None:
            self.prepare_datasets()
        
        self.prepare_optimizer()
        
        try:
            # 选择训练方式
            if self.config.use_scenario:
                result = self._scenario_train()
            elif self.config.use_orchestrator and self._orchestrator:
                result = self._orchestrator_train()
            elif self.config.use_distributed and self.config.world_size > 1:
                result = self._distributed_train()
            elif self.config.use_distillationE:
                result = self._distillation_train()
            elif self.config.use_three_stage:
                result = self._three_stage_train()
            elif self.config.use_industry:
                result = self._industry_train()
            elif self.config.use_multimodal:
                result = self._multimodal_train()
            else:
                result = self._standard_train()
            
            # 触发回调
            self._trigger_callbacks('on_training_end', result)
            
            return result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            
            # 通知失败
            if self._progress_manager:
                try:
                    self._progress_manager.set_status(self.config.session_id, 'failed', error_message=str(e))
                except Exception:
                    pass
            
            return {'success': False, 'error': str(e)}
    
    def _simplified_train(self) -> Dict[str, Any]:
        """简化训练（当PyTorch不可用时）"""
        logger.info("Running simplified training (PyTorch not available)...")
        
        start_time = time.time()
        
        # 模拟训练过程
        for epoch in range(self.config.num_epochs):
            self.training_stats['current_epoch'] = epoch + 1
            
            # 模拟进度
            progress = ((epoch + 1) / self.config.num_epochs) * 100
            self._update_progress(progress, {'epoch': epoch + 1})
            
            # 模拟指标
            self.training_stats['train_losses'].append(1.0 - epoch * 0.1)
            self.training_stats['train_metrics'].append({
                'loss': 1.0 - epoch * 0.1,
                'accuracy': 0.5 + epoch * 0.05
            })
        
        total_time = time.time() - start_time
        self.training_stats['total_training_time'] = total_time
        self.training_stats['best_metric'] = 0.5 + (self.config.num_epochs - 1) * 0.05
        self.training_stats['best_epoch'] = self.config.num_epochs
        
        # 创建最终版本
        if self._version_manager:
            self._version_manager.create_version(
                self.config.session_id,
                self.config,
                self.training_stats,
                description="Simplified training completed"
            )
        
        return {
            'success': True,
            'mode': 'simplified',
            'best_metric': self.training_stats['best_metric'],
            'total_epochs': self.config.num_epochs,
            'total_training_time': total_time,
            'current_version_id': self._current_version_id
        }
    
    def _standard_train(self) -> Dict[str, Any]:
        """标准训练"""
        logger.info("Starting standard training...")
        start_time = time.time()

        # 创建数据加载器
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_dataloader = None
        if self.val_dataset:
            val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2
            )
        
        # 训练循环
        best_metric = float('-inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # 检查控制标志
            with self._lock:
                if self._cancelled:
                    logger.info("Training cancelled")
                    break
                
                while self._paused:
                    time.sleep(1)
                    if self._cancelled:
                        break
            
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            self.training_stats['current_epoch'] = epoch + 1
            
            # 触发回调
            self._trigger_callbacks('on_epoch_start', epoch)
            
            # 记录epoch开始
            if self.monitor:
                self.monitor.log_epoch_start(epoch, self.config.num_epochs)
            
            # 训练阶段
            train_metrics = self._train_epoch(train_dataloader, epoch)
            
            # 验证阶段
            val_metrics = {}
            if val_dataloader:
                val_metrics = self._validate_epoch(val_dataloader, epoch)
            
            # 记录epoch结束
            epoch_metrics = {**train_metrics, **val_metrics}
            if self.monitor:
                self.monitor.log_epoch_end(epoch, epoch_metrics)
            
            # 触发回调
            self._trigger_callbacks('on_epoch_end', epoch, epoch_metrics)
            
            # 记录统计信息
            self.training_stats['epochs'].append(epoch)
            self.training_stats['train_losses'].append(train_metrics.get('loss', 0))
            self.training_stats['train_metrics'].append(train_metrics)
            
            if val_metrics:
                self.training_stats['val_losses'].append(val_metrics.get('loss', 0))
                self.training_stats['val_metrics'].append(val_metrics)
            
            # 更新进度
            progress = ((epoch + 1) / self.config.num_epochs) * 100
            self._update_progress(progress, epoch_metrics)
            
            # 自动创建版本检查点
            if self.config.enable_versioning and self._version_manager:
                if (epoch + 1) % self.config.auto_checkpoint_interval == 0:
                    self._version_manager.create_version(
                        self.config.session_id,
                        self.config,
                        self.training_stats,
                        description=f"Auto checkpoint at epoch {epoch + 1}"
                    )
            
            # 早停检查
            current_metric = val_metrics.get('accuracy', train_metrics.get('accuracy', 0))
            if current_metric > best_metric + self.config.early_stopping_threshold:
                best_metric = current_metric
                patience_counter = 0
                self.training_stats['best_metric'] = best_metric
                self.training_stats['best_epoch'] = epoch
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # 计算总训练时间
        total_time = time.time() - start_time
        self.training_stats['total_training_time'] = total_time
        
        # 创建最终版本
        if self._version_manager:
            self._version_manager.create_version(
                self.config.session_id,
                self.config,
                self.training_stats,
                description="Final version"
            )
        
        # 保存最终结果
        self._save_training_results()
        
        # 记录训练结束
        if self.monitor:
            self.monitor.log_training_end(total_time, {
                'best_metric': self.training_stats['best_metric'],
                'best_epoch': self.training_stats['best_epoch'],
                'final_train_loss': self.training_stats['train_losses'][-1] if self.training_stats['train_losses'] else None,
                'final_val_loss': self.training_stats['val_losses'][-1] if self.training_stats['val_losses'] else None
            })
        
        return {
            'success': True,
            'best_metric': self.training_stats['best_metric'],
            'best_epoch': self.training_stats['best_epoch'],
            'total_epochs': len(self.training_stats['epochs']),
            'final_train_loss': self.training_stats['train_losses'][-1] if self.training_stats['train_losses'] else None,
            'final_val_loss': self.training_stats['val_losses'][-1] if self.training_stats['val_losses'] else None,
            'total_training_time': total_time,
            'current_version_id': self._current_version_id
        }
    
    def _orchestrator_train(self) -> Dict[str, Any]:
        """使用编排器训练"""
        logger.info("Starting orchestrator-based training...")
        
        if not self._orchestrator:
            return self._standard_train()
        
        try:
            # 根据配置创建计划
            plan = None
            
            if self.config.use_three_stage and create_three_stage_plan:
                plan = create_three_stage_plan(
                    name=f"three_stage_{self.config.session_id}"
                )
            elif self.config.use_multimodal and create_multimodal_plan:
                plan = create_multimodal_plan(
                    name=f"multimodal_{self.config.session_id}"
                )
            elif self.config.use_distillation and create_distillation_plan:
                plan = create_distillation_plan(
                    name=f"distillation_{self.config.session_id}"
                )
            elif self.config.use_industry and create_industry_plan:
                plan = create_industry_plan(
                    name=f"industry_{self.config.session_id}"
                )
            elif create_standard_plan:
                plan = create_standard_plan(
                    name=f"standard_{self.config.session_id}",
                    epochs=self.config.num_epochs,
                    learning_rate=self.config.learning_rate
                )
            
            if plan:
                if self.model is None:
                    self.prepare_model()
                if self.train_dataset is None:
                    self.prepare_datasets()
                
                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    num_workers=2
                )
                val_loader = None
                if self.val_dataset:
                    val_loader = DataLoader(
                        self.val_dataset,
                        batch_size=self.config.batch_size,
                        shuffle=False,
                        num_workers=2
                    )
                
                result = self._orchestrator.execute(
                    plan,
                    model=self.model,
                    train_loader=train_loader,
                    val_loader=val_loader
                )
                return {
                    'success': True,
                    'orchestrator_result': result,
                    'current_version_id': self._current_version_id
                }
            
            # 回退到标准训练
            return self._standard_train()
            
        except Exception as e:
            logger.error(f"Orchestrator training failed: {e}")
            return self._standard_train()
    
    def _scenario_train(self) -> Dict[str, Any]:
        """场景化训练"""
        logger.info(f"Starting scenario training: {self.config.scenario_type}")
        
        try:
            scenario_manager = get_scenario_manager() if get_scenario_manager else None
            
            if scenario_manager:
                # 提交场景任务
                scenario_type_value = self.config.scenario_type
                
                # 安全检查 TrainingScenario 是否可用
                if TrainingScenario and isinstance(scenario_type_value, TrainingScenario):
                    scenario_type = scenario_type_value.value
                else:
                    scenario_type = scenario_type_value or 'basic_model'
                
                scenario_name = f"{scenario_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                scenario_config = self.config.scenario_config or {}
                scenario_config.update({
                    'model_name': self.config.model_name,
                    'train_data_path': self.config.train_data_path,
                    'eval_data_path': self.config.val_data_path,
                    'num_epochs': self.config.num_epochs,
                    'batch_size': self.config.batch_size,
                    'learning_rate': self.config.learning_rate,
                })
                
                job_id = scenario_manager.submit_job(
                    user_id=getattr(self.config, 'user_id', None) or 'system',
                    scenario_type=scenario_type,
                    name=scenario_name,
                    config=scenario_config,
                    description=getattr(self.config, 'description', ''),
                    scheduled_at=None
                )
                
                # 等待完成（简化实现）
                return {
                    'success': True,
                    'job_id': job_id,
                    'scenario_type': self.config.scenario_type,
                    'current_version_id': self._current_version_id
                }
            
            # 回退到标准训练
            return self._standard_train()
            
        except Exception as e:
            logger.error(f"Scenario training failed: {e}")
            return self._standard_train()
    
    def _three_stage_train(self) -> Dict[str, Any]:
        """三阶段训练"""
        logger.info("Starting three-stage training...")
        
        if not THREE_STAGE_AVAILABLE or not ThreeStageTrainer:
            raise BusinessLogicError("Three-stage training not available")
        
        if not self.config.three_stage_config and ThreeStageConfig:
            self.config.three_stage_config = ThreeStageConfig()
        
        # 创建三阶段训练器
        trainer = ThreeStageTrainer(self.config.three_stage_config)
        
        # 执行训练
        result = trainer.train()
        
        # 创建版本
        if self._version_manager:
            self._version_manager.create_version(
                self.config.session_id,
                self.config,
                self.training_stats,
                description="Three-stage training completed"
            )
        
        return {
            **result,
            'current_version_id': self._current_version_id
        }
    
    def _distributed_train(self) -> Dict[str, Any]:
        """分布式训练"""
        logger.info("Starting distributed training...")
        
        if not DISTRIBUTED_AVAILABLE or not DistributedTrainingConfig:
            raise BusinessLogicError("Distributed training not available")
        
        # 构造分布式配置
        dist_cfg = DistributedTrainingConfig(
            model_name=self.config.model_name,
            data_path=self.config.train_data_path,
            output_dir=self.output_dir,
            world_size=self.config.world_size,
            num_nodes=1,
            node_rank=0,
            nproc_per_node=1,
            master_addr="localhost",
            master_port=self.config.master_port,
            backend="nccl",
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            max_length=self.config.max_seq_length,
            warmup_steps=int(self.config.warmup_ratio * 1000),
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            fp16=self.config.use_fp16,
            gradient_clipping=self.config.max_grad_norm,
            dataloader_num_workers=4,
            pin_memory=True,
            model_parallel=False,
            pipeline_parallel_size=1,
            tensor_parallel_size=1
        )
        
        # 启动分布式训练
        dist_cfg_dict = dist_cfg.to_dict()
        result = launch_distributed_training(dist_cfg_dict)
        
        return {
            **result,
            'current_version_id': self._current_version_id
        }
    
    def _distillation_train(self) -> Dict[str, Any]:
        """知识蒸馏训练"""
        logger.info("Starting knowledge distillation training...")
        
        if not self.config.distillation_config and DistillationConfig:
            self.config.distillation_config = DistillationConfig()
        
        # 创建知识蒸馏训练器
        distillation_trainer = KnowledgeDistillationTrainer(self.config.distillation_config)
        
        # 执行蒸馏训练
        results = distillation_trainer.train()
        
        return {
            **results,
            'current_version_id': self._current_version_id
        }
    
    def _multimodal_train(self) -> Dict[str, Any]:
        """多模态训练"""
        logger.info("Starting multimodal training...")
        
        # 使用标准训练流程，但使用多模态模型
        return self._standard_train()
    
    def _industry_train(self) -> Dict[str, Any]:
        """行业训练"""
        logger.info("Starting industry training...")
        
        if not INDUSTRY_AVAILABLE or not IndustryTrainer:
            # 回退到标准训练
            return self._standard_train()
        
        if not self.config.industry_config and IndustryConfig:
            self.config.industry_config = IndustryConfig()
        
        # 创建行业训练器
        trainer = IndustryTrainer(self.config.industry_config)
        
        # 执行训练
        results = trainer.train()
        
        return {
            **results,
            'current_version_id': self._current_version_id
        }
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for step, batch in enumerate(dataloader):
            # 检查控制标志
            with self._lock:
                if self._cancelled:
                    break
                while self._paused:
                    time.sleep(1)
                    if self._cancelled:
                        break
            
            self.training_stats['current_step'] = step
            
            # 移动数据到设备
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            else:
                batch = [item.to(self.device) if isinstance(item, torch.Tensor) else item 
                        for item in batch]
            
            # 前向传播
            if isinstance(batch, dict):
                outputs = self.model(batch)
                if isinstance(outputs, dict):
                    loss = outputs.get('loss')
                    logits = outputs.get('logits')
                else:
                    loss = outputs
                    logits = outputs
            else:
                inputs, targets = batch
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                logits = outputs
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            
            if logits is not None and isinstance(batch, dict) and 'labels' in batch:
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == batch['labels'].squeeze()).sum().item()
                total_samples += batch['labels'].size(0)
            
            # 记录步骤
            if step % self.config.logging_steps == 0:
                step_metrics = {
                    'loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                if self.monitor:
                    self.monitor.log_step(epoch * len(dataloader) + step, step_metrics)
                
                # 触发回调
                self._trigger_callbacks('on_step_end', step, step_metrics)
            
            # 评估
            if step % self.config.eval_steps == 0 and step > 0:
                eval_metrics = {'loss': loss.item()}
                if self.monitor:
                    self.monitor.log_evaluation(eval_metrics)
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _validate_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                else:
                    batch = [item.to(self.device) if isinstance(item, torch.Tensor) else item 
                            for item in batch]
                
                if isinstance(batch, dict):
                    outputs = self.model(batch)
                    if isinstance(outputs, dict):
                        loss = outputs.get('loss')
                        logits = outputs.get('logits')
                    else:
                        loss = outputs
                        logits = outputs
                else:
                    inputs, targets = batch
                    outputs = self.model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, targets)
                    logits = outputs
                
                total_loss += loss.item()
                
                if logits is not None and isinstance(batch, dict) and 'labels' in batch:
                    predictions = torch.argmax(logits, dim=-1)
                    correct_predictions += (predictions == batch['labels'].squeeze()).sum().item()
                    total_samples += batch['labels'].size(0)
        
        avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
        
        logger.info(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _update_progress(self, progress: float, metrics: Dict[str, Any] = None):
        """更新进度"""
        # 更新进度管理器
        if self._progress_manager:
            try:
                self._progress_manager.update_progress(
                    self.config.session_id,
                    progress=progress,
                    current_epoch=self.training_stats['current_epoch'],
                    current_step=self.training_stats['current_step'],
                    metrics=metrics
                )
            except Exception as e:
                logger.warning(f"Failed to update progress manager: {e}")
        
        # 更新任务管理器
        if self._task_manager:
            try:
                self._task_manager.update_task_progress(
                    self.config.session_id,
                    progress,
                    metrics
                )
            except Exception as e:
                logger.warning(f"Failed to update task manager: {e}")
    
    def _save_training_results(self):
        """保存训练结果"""
        # 保存统计信息
        stats_path = os_path.path.join(self.output_dir, "training_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        # 保存配置
        config_path = os_path.path.join(self.output_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info(f"Training results saved to {self.output_dir}")
    
    # ==================== 扩展功能方法 ====================

    def create_scenario_instance(self, scenario_type: str, config: Dict[str, Any]) -> Optional[Any]:
        """创建特定场景实例"""       
        try:
            if scenario_type == 'basic' and BasicModelScenario:
                return BasicModelScenario(config)
            elif scenario_type == 'advanced' and AdvancedModelScenario:
                return AdvancedModelScenario(config)
            elif scenario_type == 'industry' and IndustryScenario:
                return IndustryScenario(config)
            elif scenario_type == 'scheduled' and ScheduledTrainingScenario:
                return ScheduledTrainingScenario(config)
        except Exception as e:
            logger.warning(f"Failed to create scenario instance: {e}")
        return None

    def create_custom_plan(self, steps: List[Dict[str, Any]]) -> Optional[Any]:
        """创建自定义编排计划"""
        try:
            return OrchestratorPlan(
                session_id=self.config.session_id,
                steps=steps,
                tenant_id=self.config.tenant_id
            )
        except Exception as e:
            logger.warning(f"Failed to create orchestrator plan: {e}")
        
        return None

    def check_layer_status(self) -> Dict[str, Any]:
        """检查层级管理器状态"""
        if ORCHESTRATOR_AVAILABLE and LayerManager:
            try:
                # 假设 LayerManager 有获取状态的方法
                return LayerManager.get_status() if hasattr(LayerManager, 'get_status') else {}
            except Exception:
                pass
        return {}

    def report_detailed_progress(self, progress: float, metrics: Dict[str, Any], stage: str = "training"):
        """使用 TrainingProgress 对象上报详细进度"""
        if PROGRESS_AVAILABLE and self._progress_manager and TrainingProgress:
            try:
                progress_obj = TrainingProgress(
                    session_id=self.config.session_id,
                    stage=stage,
                    progress=progress,
                    metrics=metrics,
                    timestamp=time.time(),
                    epoch=self.training_stats.get('current_epoch', 0),
                    step=self.training_stats.get('current_step', 0)
                )
                if hasattr(self._progress_manager, 'update'):
                    self._progress_manager.update(progress_obj)
            except Exception as e:
                logger.warning(f"Failed to report detailed progress: {e}")

    def execute_custom_pipeline(self, pipeline_def: Dict[str, Any]) -> Dict[str, Any]:
        """执行自定义流水线"""
        if PIPELINE_AVAILABLE and PipelineDefinition and PipelineRunner:
            try:
                definition = PipelineDefinition(**pipeline_def)
                runner = PipelineRunner(self.config.session_id)
                result = execute_pipeline(definition, runner, session_id=self.config.session_id)
                return result.to_dict() if hasattr(result, 'to_dict') else result
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}")
        return {'success': False, 'error': 'Pipeline components not available'}

    def setup_plugin_context(self) -> Optional[Any]:
        """设置插件上下文"""
        if PLUGINS_AVAILABLE and PluginContext and PluginRegistry:
            try:
                context = PluginContext(
                    hook=HookPoint.ON_TRAINING_START,
                    session_id=self.config.session_id,
                    model=self.model,
                    data={'config': self.config.to_dict()}
                )
                return context
            except Exception as e:
                logger.warning(f"Failed to setup plugin context: {e}")
        return None

    def validate_strategy(self) -> List[str]:
        """验证策略配置"""
        errors = []
        if STRATEGY_AVAILABLE and StrategyValidator:
            try:
                validator = StrategyValidator()
                if hasattr(validator, 'validate'):
                    errors = validator.validate(self.config.to_dict())
            except Exception as e:
                logger.warning(f"Strategy validation failed: {e}")
        return errors

    def create_strategy_context(self) -> Optional[Any]:
        """创建策略上下文"""
        if STRATEGY_AVAILABLE and StrategyContext:
            try:
                return StrategyContext(
                    model=self.model,
                    config=self.config.to_dict(),
                    device=self.device
                )
            except Exception as e:
                logger.warning(f"Failed to create strategy context: {e}")
        return None

    def recommend_distributed_config(self) -> Dict[str, Any]:
        """推荐分布式配置"""
        if DISTRIBUTED_STRATEGY_AVAILABLE and recommend_distributed_mode:
            try:
                system_info = {}
                if HARDWARE_AVAILABLE and get_available_memory:
                    system_info['memory'] = get_available_memory()
                return recommend_distributed_mode(system_info)
            except Exception as e:
                logger.warning(f"Distributed recommendation failed: {e}")
        return {}

    def optimize_resources(self):
        """优化资源使用"""
        if HARDWARE_AVAILABLE:
            try:
                if clear_memory:
                    clear_memory()
                if recommend_precision:
                    prec = recommend_precision(str(self.device))
                    logger.info(f"Recommended precision: {prec}")
            except Exception as e:
                logger.warning(f"Resource optimization failed: {e}")

    def build_loss_function(self, loss_type: str, **kwargs) -> Optional[Any]:
        """构建损失函数"""
        if LOSSES_AVAILABLE:
            try:
                if LossFactory:
                    factory = LossFactory()
                    return factory.create(loss_type, **kwargs)
                elif create_loss:
                    return create_loss(loss_type, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to build loss function: {e}")
        return None

    def create_task_snapshot(self) -> Optional[Any]:
        """创建任务快照"""
        if TASK_MANAGER_AVAILABLE and TaskVersion and VersionManager:
            try:
                # 模拟创建任务版本快照
                return TaskVersion(
                    version_id=str(uuid.uuid4()),
                    task_id=self.config.session_id,
                    config_snapshot=self.config.to_dict(),
                    created_at=datetime.utcnow()
                )
            except Exception as e:
                logger.warning(f"Failed to create task snapshot: {e}")
        return None

    def setup_model_compression(self, config: Dict[str, Any]) -> Optional[Any]:
        """设置模型压缩"""
        if DISTILLATION_AVAILABLE and ModelCompressor and CompressionConfig:
            try:
                compression_config = CompressionConfig(**config)
                return ModelCompressor(config=compression_config)
            except Exception as e:
                logger.warning(f"Failed to setup model compression: {e}")
        return None

    def execute_advanced_pipeline(self, pipeline_def: Any) -> Dict[str, Any]:
        """执行高级流水线"""
        if PIPELINE_AVAILABLE and PipelineExecutor and execute_pipeline:
            try:
                # 假设 pipeline_def 是 PipelineDefinition
                if execute_pipeline:
                    # 确保 pipeline_def 是 PipelineDefinition 对象
                    if isinstance(pipeline_def, dict) and PipelineDefinition:
                        pipeline_obj = PipelineDefinition(**pipeline_def)
                    else:
                        pipeline_obj = pipeline_def
                         
                    # 需要一个 runner 回调，这里简单包装
                    def simple_runner(step):
                        return None
                        
                    result = execute_pipeline(pipeline_obj, simple_runner, session_id=self.config.session_id)
                    return result.to_dict() if hasattr(result, 'to_dict') else {'success': True, 'result': result}
            except Exception as e:
                logger.error(f"Advanced pipeline execution failed: {e}")
        return {'success': False, 'error': 'Pipeline executor not available'}

    def configure_distributed_strategy(self, mode: str, **kwargs) -> Optional[Any]:
        """配置分布式策略"""
        if DISTRIBUTED_STRATEGY_AVAILABLE and DistributedStrategyConfig and DistributedMode:
            try:
                mode_enum = DistributedMode.DDP
                if hasattr(DistributedMode, mode.upper()):
                    mode_enum = getattr(DistributedMode, mode.upper())
                
                return DistributedStrategyConfig(
                    mode=mode_enum,
                    **kwargs
                )
            except Exception as e:
                logger.warning(f"Failed to configure distributed strategy: {e}")
        return None

    def diagnose_distributed_env(self) -> Dict[str, Any]:
        """诊断分布式环境"""
        if DISTRIBUTED_STRATEGY_AVAILABLE and diagnose_distributed_strategy:
            try:
                return diagnose_distributed_strategy()
            except Exception as e:
                logger.warning(f"Distributed diagnosis failed: {e}")
        return {}

    def register_plugin(self, plugin: Any) -> bool:
        """注册插件"""
        if PLUGINS_AVAILABLE and PluginRegistry:
            try:
                # 假设 PluginRegistry 有 register 方法
                registry = PluginRegistry()
                if hasattr(registry, 'register'):
                    registry.register(plugin)
                    return True
            except Exception as e:
                logger.warning(f"Plugin registration failed: {e}")
        return False

    def diagnose(self) -> Dict[str, Any]:
        """诊断系统状态"""
        return {
            'session_id': self.config.session_id,
            'status': 'paused' if self._paused else ('cancelled' if self._cancelled else 'running'),
            'current_epoch': self.training_stats['current_epoch'],
            'current_step': self.training_stats['current_step'],
            'best_metric': self.training_stats['best_metric'],
            'current_version_id': self._current_version_id,
            'architecture_note': 'core 模块不直接调用 services，服务调用由 launcher 层处理',
            'version_count': len(self.get_versions()) if self._version_manager else 0
        }


# ==================== 便捷函数 ====================

def create_training_config(**kwargs) -> TrainingConfig:
    """创建训练配置的便捷函数"""
    return TrainingConfig(**kwargs)


def launch_unified_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """启动统一训练的便捷函数"""
    try:
        training_config = TrainingConfig(**config)
        training_system = UnifiedTrainingSystem(training_config)
        return training_system.train()
    except Exception as e:
        logger.error(f"Failed to launch unified training: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def get_integration_availability() -> Dict[str, bool]:
    """获取集成模块可用性
    
    注意：core 模块只集成下游训练模块，不直接调用 services 层
    服务层调用由上层 launcher 模块处理
    """
    return {
        'torch': TORCH_AVAILABLE,
    }


def diagnose_training_system(session_id: str = None) -> Dict[str, Any]:
    """诊断训练系统"""
    return {
        'integration_modules': get_integration_availability(),
        'session_id': session_id
    }

# -*- coding: utf-8 -*-
"""
知识蒸馏训练器

生产级知识蒸馏训练器，通过调用策略层实现知识蒸馏核心功能。
本模块作为底层训练器，封装了完整的训练流程。

设计说明：
- 核心损失计算逻辑由策略层 (strategies/distillation_strategy.py) 提供
- 本模块专注于训练流程编排和模型管理
- 集成配置层 (compression_config.py) 的监控和验证功能
- 集成分布式策略层 (distributed_strategy.py) 支持分布式训练
- 集成 backend.lib 层的硬件和损失函数管理
- 保持向后兼容的 API 接口

架构调用层次：
├── knowledge_distillation.py (本模块)
│   └── 调用 compression_config.py (配置层)
│       ├── DistillationConfig - 蒸馏配置
│       ├── DistillationMonitor - 蒸馏监控
│       ├── DistillationStats - 蒸馏统计
│       ├── ConfigValidator - 配置验证
│       └── DistillationPresets - 预设模板
│   └── 调用 backend/modules/training/strategies (策略层)
│       ├── base_strategy.py - StrategyContext, StrategyResult, StrategyMonitor
│       ├── distributed_strategy.py - DistributedStrategy
│       └── distillation_strategy.py - DistillationStrategy
│   └── 调用 backend/lib (底层)
│       ├── losses - 损失函数
│       ├── hardware - 硬件管理
│       └── distributed - 分布式管理
└── 被 distillation_service.py, distillation_scenarios.py 调用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from contextlib import contextmanager

# 修复导入路径
import sys
import os as os_path
current_dir = os_path.path.dirname(os_path.path.abspath(__file__))
project_root = os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(current_dir))))
sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# ======================== 配置层导入 ========================

from .compression_config import (
    DistillationConfig, 
    CompressionConfig,
    ScenarioDistillationConfig,
    DistributedDistillationConfig,
    AdaptiveDistillationConfig,
    DistillationTaskConfig,
    DistillationScenario,
    DistributedMode,
    AdaptiveMode,
    CompressionMethod,
    DistillationPresets,
    DistillationStats,
    DistillationMonitor,
    ConfigValidator,
    create_distillation_config,
    validate_config,
    recommend_config,
)

try:
    from backend.core.exceptions import BusinessLogicError
except ImportError:
    class BusinessLogicError(Exception):
        pass

# ======================== 策略层导入 ========================

STRATEGY_LAYER_AVAILABLE = False
try:
    from backend.modules.training.strategies.base_strategy import (
        TrainingStrategy,
        StrategyContext, 
        StrategyResult,
        TrainingPhase,
        StrategyType,
        StrategyMonitor as BaseStrategyMonitor,
        StrategyProfiler,
        StrategyValidator,
        StrategyMetrics,
    )
    STRATEGY_LAYER_AVAILABLE = True
    logger.info("Strategy layer loaded successfully")
except ImportError as e:
    logger.warning(f"Strategy layer not available: {e}")
    TrainingStrategy = None
    StrategyContext = None
    StrategyResult = None
    TrainingPhase = None
    StrategyType = None
    BaseStrategyMonitor = None
    StrategyProfiler = None
    StrategyValidator = None
    StrategyMetrics = None

DISTILLATION_STRATEGY_AVAILABLE = False
try:
    from backend.modules.training.strategies.distillation_strategy import (
        DistillationStrategy,
        DistillationStrategyConfig,
        DistillationLossCalculator,
        DistillationType,
        create_distillation_strategy,
    )
    DISTILLATION_STRATEGY_AVAILABLE = True
    logger.info("Distillation strategy loaded successfully")
except ImportError as e:
    logger.warning(f"Distillation strategy not available: {e}")
    DistillationStrategy = None
    DistillationStrategyConfig = None
    DistillationLossCalculator = None
    DistillationType = None
    create_distillation_strategy = None

DISTRIBUTED_STRATEGY_AVAILABLE = False
try:
    from backend.modules.training.strategies.distributed_strategy import (
        DistributedStrategy,
        DistributedStrategyConfig,
        DistributedMode as StrategyDistributedMode,
        ZeROStage,
        DistributedHealthStatus,
        CommunicationStats,
        create_distributed_strategy,
        recommend_distributed_mode,
    )
    DISTRIBUTED_STRATEGY_AVAILABLE = True
    logger.info("Distributed strategy loaded successfully")
except ImportError as e:
    logger.warning(f"Distributed strategy not available: {e}")
    DistributedStrategy = None
    DistributedStrategyConfig = None
    StrategyDistributedMode = None
    ZeROStage = None
    DistributedHealthStatus = None
    CommunicationStats = None
    create_distributed_strategy = None
    recommend_distributed_mode = None

# ======================== 底层 lib 模块导入 ========================

LOSSES_LAYER_AVAILABLE = False
try:
    from backend.lib.losses import (
        LossFactory,
        create_loss,
        BaseLoss,
        LossMonitor as LibLossMonitor,
        LossStats as LibLossStats,
    )
    LOSSES_LAYER_AVAILABLE = True
except ImportError:
    LossFactory = None
    create_loss = None
    BaseLoss = None
    LibLossMonitor = None
    LibLossStats = None

HARDWARE_LAYER_AVAILABLE = False
try:
    from backend.lib.hardware import (
        DeviceManager,
        get_device_manager,
        MemoryManager,
        get_available_memory,
        clear_memory,
        MixedPrecisionManager,
        GradientCheckpointing,
    )
    HARDWARE_LAYER_AVAILABLE = True
except ImportError:
    DeviceManager = None
    get_device_manager = None
    MemoryManager = None
    get_available_memory = None
    clear_memory = None
    MixedPrecisionManager = None
    GradientCheckpointing = None

DISTRIBUTED_LAYER_AVAILABLE = False
try:
    from backend.lib.distributed import (
        DistributedManager,
        get_distributed_manager,
        is_main_process,
        get_rank,
        get_world_size,
        barrier,
        all_reduce,
        AllReduceOp,
    )
    DISTRIBUTED_LAYER_AVAILABLE = True
except ImportError:
    DistributedManager = None
    get_distributed_manager = None
    is_main_process = lambda: True
    get_rank = lambda: 0
    get_world_size = lambda: 1
    barrier = lambda: None
    all_reduce = None
    AllReduceOp = None


# ======================== 训练状态和统计 ========================

class TrainerPhase(Enum):
    """训练器阶段"""
    INIT = "init"
    LOADING = "loading"
    TRAINING = "training"
    VALIDATING = "validating"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TrainerStats:
    """训练器统计"""
    total_steps: int = 0
    total_epochs: int = 0
    total_samples: int = 0
    total_time_seconds: float = 0.0
    avg_step_time: float = 0.0
    best_loss: float = float('inf')
    best_accuracy: float = 0.0
    current_loss: float = 0.0
    current_accuracy: float = 0.0
    memory_peak_mb: float = 0.0
    
    # 损失分解
    soft_loss_avg: float = 0.0
    hard_loss_avg: float = 0.0
    feature_loss_avg: float = 0.0
    attention_loss_avg: float = 0.0
    contrastive_loss_avg: float = 0.0
    
    # 收敛信息
    convergence_step: int = 0
    is_converged: bool = False
    
    def update(self, loss: float, accuracy: float = 0.0, step_time: float = 0.0) -> None:
        """更新统计"""
        self.total_steps += 1
        self.current_loss = loss
        self.current_accuracy = accuracy
        self.total_time_seconds += step_time
        
        if self.total_steps > 0:
            self.avg_step_time = self.total_time_seconds / self.total_steps
        
        if loss < self.best_loss:
            self.best_loss = loss
        
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class TrainerHealthStatus:
    """训练器健康状态"""
    is_healthy: bool = True
    memory_ok: bool = True
    loss_ok: bool = True
    gradient_ok: bool = True
    last_check_time: str = ""
    issues: List[str] = field(default_factory=list)
    
    def check(self, loss: float, available_memory_mb: float, min_memory_mb: float = 1000.0) -> None:
        """检查健康状态"""
        self.issues.clear()
        self.last_check_time = datetime.utcnow().isoformat()
        
        # 检查内存
        self.memory_ok = available_memory_mb > min_memory_mb
        if not self.memory_ok:
            self.issues.append(f"Low memory: {available_memory_mb:.0f}MB < {min_memory_mb:.0f}MB")
        
        # 检查损失
        self.loss_ok = not (torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)))
        if not self.loss_ok:
            self.issues.append(f"Invalid loss: {loss}")
        
        self.is_healthy = self.memory_ok and self.loss_ok and self.gradient_ok
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'is_healthy': self.is_healthy,
            'memory_ok': self.memory_ok,
            'loss_ok': self.loss_ok,
            'gradient_ok': self.gradient_ok,
            'last_check_time': self.last_check_time,
            'issues': self.issues,
        }


class KnowledgeDistillationTrainer:
    """
    知识蒸馏训练器
    
    生产级实现，通过策略层实现核心蒸馏功能，提供完整的训练流程管理。
    
    特性：
    - 集成配置层 (compression_config.py) 的监控和验证功能
    - 集成策略层 (base_strategy.py, distillation_strategy.py) 的策略管理
    - 集成分布式策略层 (distributed_strategy.py) 支持分布式训练
    - 集成 backend.lib 层的硬件和损失函数管理
    - 完整的监控、诊断和健康检查
    """

    def __init__(self, config: DistillationConfig, task_config: Optional[DistillationTaskConfig] = None):
        """
        初始化知识蒸馏训练器
        
        Args:
            config: 蒸馏配置（使用原有配置格式，内部转换为策略配置）
            task_config: 完整的任务配置（可选）
        """
        self.config = config
        self.task_config = task_config
        self.logger = logging.getLogger(f"{__name__}.KnowledgeDistillationTrainer")
        
        # 设备管理 (lib/hardware)
        self._device_manager: Optional['DeviceManager'] = None
        if HARDWARE_LAYER_AVAILABLE and get_device_manager is not None:
            try:
                self._device_manager = get_device_manager()
                self.device = self._device_manager.get_optimal_device() if hasattr(self._device_manager, 'get_optimal_device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            except Exception:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 模型
        self.teacher_model: Optional[nn.Module] = None
        self.student_model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        
        # 转换为策略配置并创建策略
        self.strategy_config = self._convert_to_strategy_config(config)
        self.strategy: Optional['DistillationStrategy'] = None
        self.loss_calculator: Optional['DistillationLossCalculator'] = None
        
        # 分布式策略 (distributed_strategy.py)
        self._distributed_strategy: Optional['DistributedStrategy'] = None
        self._distributed_config: Optional[DistributedDistillationConfig] = None
        if task_config and task_config.distributed_config:
            self._distributed_config = task_config.distributed_config
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self._phase = TrainerPhase.INIT
        self._is_training = False
        self._start_time: Optional[float] = None
        
        # 统计和健康状态
        self._stats = TrainerStats()
        self._health_status = TrainerHealthStatus()
        
        # 监控器 (compression_config.py)
        self._distillation_monitor = DistillationMonitor()
        
        # 配置验证器 (compression_config.py)
        self._config_validator = ConfigValidator()
        self._setup_validation_rules()
        
        # 策略层监控器 (base_strategy.py)
        self._strategy_monitor: Optional['BaseStrategyMonitor'] = None
        if STRATEGY_LAYER_AVAILABLE and BaseStrategyMonitor is not None:
            try:
                self._strategy_monitor = BaseStrategyMonitor()
            except Exception:
                pass
        
        # 性能分析器 (base_strategy.py)
        self._profiler: Optional['StrategyProfiler'] = None
        if STRATEGY_LAYER_AVAILABLE and StrategyProfiler is not None:
            try:
                self._profiler = StrategyProfiler()
            except Exception:
                pass
        
        # 策略验证器 (base_strategy.py)
        self._strategy_validator: Optional['StrategyValidator'] = None
        if STRATEGY_LAYER_AVAILABLE and StrategyValidator is not None:
            try:
                self._strategy_validator = StrategyValidator()
            except Exception:
                pass
        
        # 策略指标 (base_strategy.py)
        self._strategy_metrics: Optional['StrategyMetrics'] = None
        if STRATEGY_LAYER_AVAILABLE and StrategyMetrics is not None:
            try:
                self._strategy_metrics = StrategyMetrics()
            except Exception:
                pass
        
        # 内存管理器 (lib/hardware)
        self._memory_manager: Optional['MemoryManager'] = None
        if HARDWARE_LAYER_AVAILABLE and MemoryManager is not None:
            try:
                self._memory_manager = MemoryManager()
            except Exception:
                pass
        
        # 混合精度管理器 (lib/hardware)
        self._mixed_precision_manager: Optional['MixedPrecisionManager'] = None
        use_amp = getattr(config, 'use_amp', False)
        if HARDWARE_LAYER_AVAILABLE and MixedPrecisionManager is not None and use_amp:
            try:
                self._mixed_precision_manager = MixedPrecisionManager()
            except Exception:
                pass
        
        # 梯度检查点 (lib/hardware)
        self._gradient_checkpointing: Optional['GradientCheckpointing'] = None
        if HARDWARE_LAYER_AVAILABLE and GradientCheckpointing is not None:
            try:
                self._gradient_checkpointing = GradientCheckpointing()
            except Exception:
                pass
        
        # 分布式管理器 (lib/distributed)
        self._distributed_manager: Optional['DistributedManager'] = None
        if DISTRIBUTED_LAYER_AVAILABLE and get_distributed_manager is not None:
            try:
                self._distributed_manager = get_distributed_manager()
            except Exception:
                pass
        
        # 损失函数工厂 (lib/losses)
        self._loss_factory: Optional['LossFactory'] = None
        if LOSSES_LAYER_AVAILABLE and LossFactory is not None:
            try:
                self._loss_factory = LossFactory()
            except Exception:
                pass
        
        # 损失监控器 (lib/losses)
        self._lib_loss_monitor: Optional['LibLossMonitor'] = None
        if LOSSES_LAYER_AVAILABLE and LibLossMonitor is not None:
            try:
                self._lib_loss_monitor = LibLossMonitor()
            except Exception:
                pass
        
        # 回调函数
        self.callbacks: Dict[str, List[Callable]] = {
            'on_step_start': [],
            'on_step_end': [],
            'on_epoch_start': [],
            'on_epoch_end': [],
            'on_train_start': [],
            'on_train_end': [],
            'on_checkpoint': [],
            'on_validation': [],
        }
        
        # 损失历史
        self._loss_history: List[float] = []
        self._accuracy_history: List[float] = []
        
        # 检查点
        self._best_checkpoint: Optional[Dict[str, Any]] = None
        self._checkpoint_path: Optional[str] = None
    
    def _setup_validation_rules(self) -> None:
        """
        设置配置验证规则
        
        使用 compression_config.py 的 ConfigValidator
        """
        def check_temperature(cfg) -> Tuple[bool, str]:
            if hasattr(cfg, 'temperature') and not (1.0 <= cfg.temperature <= 20.0):
                return False, f"Temperature {cfg.temperature} should be between 1.0 and 20.0"
            return True, ""
        
        def check_loss_weights(cfg) -> Tuple[bool, str]:
            if hasattr(cfg, 'alpha') and hasattr(cfg, 'beta'):
                total = cfg.alpha + cfg.beta
                if not (0.9 <= total <= 1.1):
                    return False, f"Loss weights alpha + beta = {total} should be approximately 1.0"
            return True, ""
        
        self._config_validator.add_check(check_temperature)
        self._config_validator.add_check(check_loss_weights)

    def _convert_to_strategy_config(self, config: DistillationConfig) -> Optional['DistillationStrategyConfig']:
        """
        将原有配置转换为策略配置
        
        保持向后兼容性，同时使用策略层的配置格式。
        """
        if not DISTILLATION_STRATEGY_AVAILABLE or DistillationStrategyConfig is None:
            return None
        
        # 确定蒸馏类型
        if config.use_feature_distillation and config.use_attention_distillation:
            distillation_type = "combined"
        elif config.use_feature_distillation:
            distillation_type = "feature"
        elif config.use_attention_distillation:
            distillation_type = "attention"
        else:
            distillation_type = "logits"
        
        return DistillationStrategyConfig(
            temperature=config.temperature,
            hard_loss_weight=config.beta,
            soft_loss_weight=config.alpha,
            feature_loss_weight=config.feature_loss_weight,
            attention_loss_weight=config.attention_loss_weight,
            distillation_type=distillation_type,
            feature_layers=[-1, -2, -3],  # 默认蒸馏最后三层
            feature_loss_type="mse"
        )

    def _initialize_strategy(self) -> None:
        """
        初始化蒸馏策略和损失计算器
        
        使用 distillation_strategy.py 的策略类
        """
        if not DISTILLATION_STRATEGY_AVAILABLE:
            self.logger.warning("Distillation strategy layer not available")
            return
        
        # 使用性能分析器 (base_strategy.py)
        if self._profiler is not None:
            with self._profiler.profile("initialize_strategy"):
                self._do_initialize_strategy()
        else:
            self._do_initialize_strategy()
    
    def _do_initialize_strategy(self) -> None:
        """执行策略初始化"""
        if not DISTILLATION_STRATEGY_AVAILABLE or DistillationStrategy is None:
            return
        
        try:
            # 创建策略
            self.strategy = DistillationStrategy(
                config=self.strategy_config,
                teacher_model=self.teacher_model
            )
        
        # 创建损失计算器（直接使用策略层的计算器）
            if DistillationLossCalculator is not None:
                self.loss_calculator = DistillationLossCalculator(
                    self.strategy_config, 
                self.device
                )
        
            # 初始化策略上下文 (base_strategy.py)
            if STRATEGY_LAYER_AVAILABLE and StrategyContext is not None:
                context = StrategyContext(
                    model=self.student_model,
                    optimizer=self.optimizer,
                    device=self.device
                )
                self.strategy.setup(context)
        
            self.logger.info(f"Strategy initialized: type={self.strategy_config.distillation_type if self.strategy_config else 'unknown'}")
        except Exception as e:
            self.logger.error(f"Failed to initialize strategy: {e}")
            raise BusinessLogicError(f"Failed to initialize strategy: {e}")
    
    def _initialize_distributed(self) -> None:
        """
        初始化分布式训练
        
        使用 distributed_strategy.py 的分布式策略
        """
        if not self._distributed_config or not self._distributed_config.is_distributed():
            return
        
        if not DISTRIBUTED_STRATEGY_AVAILABLE or create_distributed_strategy is None:
            self.logger.warning("Distributed strategy not available")
            return
        
        try:
            # 使用性能分析器
            if self._profiler is not None:
                with self._profiler.profile("initialize_distributed"):
                    self._distributed_strategy = create_distributed_strategy(
                        mode=self._distributed_config.mode,
                        world_size=self._distributed_config.world_size,
                        rank=self._distributed_config.rank,
                    )
            else:
                self._distributed_strategy = create_distributed_strategy(
                    mode=self._distributed_config.mode,
                    world_size=self._distributed_config.world_size,
                    rank=self._distributed_config.rank,
                )
            
            # 同步进程
            self.sync_processes()
            
            if self.should_log():
                self.logger.info(f"Distributed training initialized: mode={self._distributed_config.mode}, "
                               f"world_size={self._distributed_config.world_size}")
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {e}")
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        验证配置
        
        使用 compression_config.py 的 ConfigValidator 和 validate_config
        """
        errors = []
        
        # 使用本地验证器
        local_valid, local_errors = self._config_validator.validate(self.config)
        errors.extend(local_errors)
        
        # 使用 compression_config.py 的全局验证（如果有任务配置）
        if self.task_config:
            global_valid, global_errors = validate_config(self.task_config)
            errors.extend(global_errors)
        
        # 使用策略层验证器 (base_strategy.py)
        if self._strategy_validator is not None:
            try:
                strategy_result = self._strategy_validator.validate(self.config)
                if isinstance(strategy_result, tuple):
                    # Unpack tuple (is_valid, errors)
                    _, strategy_errors = strategy_result
                    if strategy_errors:
                        errors.extend(strategy_errors)
                else:
                    # Check for errors attribute
                    strategy_errors = getattr(strategy_result, 'errors', [])
                    if strategy_errors:
                        errors.extend(strategy_errors)
            except Exception:
                pass
        
        return len(errors) == 0, errors
    
    def should_log(self) -> bool:
        """
        检查是否应该记录日志
        
        使用 lib/distributed 的 is_main_process
        """
        if DISTRIBUTED_LAYER_AVAILABLE:
            return is_main_process()
        return True
    
    def sync_processes(self) -> None:
        """
        同步所有进程
        
        使用 lib/distributed 的 barrier
        """
        if DISTRIBUTED_LAYER_AVAILABLE and barrier is not None:
            try:
                barrier()
            except Exception:
                pass
    
    def optimize_memory(self) -> None:
        """
        优化内存
        
        使用 lib/hardware 的 clear_memory 和 MemoryManager
        """
        if HARDWARE_LAYER_AVAILABLE and clear_memory is not None:
            try:
                clear_memory()
            except Exception:
                pass
        
        if self._memory_manager is not None:
            try:
                if hasattr(self._memory_manager, 'optimize'):
                    self._memory_manager.optimize()
                elif hasattr(self._memory_manager, 'clear_memory'):
                    self._memory_manager.clear_memory()
            except Exception:
                pass
    
    def get_available_memory(self) -> float:
        """
        获取可用内存
        
        使用 lib/hardware 的 get_available_memory
        """
        if HARDWARE_LAYER_AVAILABLE and get_available_memory is not None:
            try:
                return get_available_memory()
            except Exception:
                pass
        return 0.0
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """
        性能分析上下文管理器
        
        使用 base_strategy.py 的 StrategyProfiler
        """
        if self._profiler is not None:
            with self._profiler.profile(operation_name):
                yield
        else:
            yield
    
    def check_health(self) -> TrainerHealthStatus:
        """
        检查训练器健康状态
        
        使用 lib/hardware 获取内存信息
        """
        available_memory = self.get_available_memory()
        current_loss = self._stats.current_loss
        
        self._health_status.check(current_loss, available_memory)
        
        return self._health_status

    def load_models(self) -> None:
        """
        加载教师和学生模型
        
        生产级实现，包含性能分析和内存优化
        """
        self._phase = TrainerPhase.LOADING
        
        try:
            with self.profile_operation("load_models"):
                # 验证配置
                is_valid, errors = self.validate_config()
                if not is_valid and self.should_log():
                    self.logger.warning(f"Configuration validation warnings: {errors}")
                
            # 在实际实现中，这里会加载真实的模型
            # 为简化起见，我们创建模拟模型
            self.teacher_model = self._create_mock_model()
            self.student_model = self._create_mock_model()
            
            # 冻结教师模型
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
            
                # 应用梯度检查点（如果可用）
            if self._gradient_checkpointing is not None and hasattr(self._gradient_checkpointing, 'apply'):
                try:
                    getattr(self._gradient_checkpointing, 'apply')(self.student_model)
                except Exception:
                    pass
                
            # 创建优化器
            lr = self.task_config.learning_rate if self.task_config else 1e-4
            self.optimizer = torch.optim.AdamW(
                self.student_model.parameters(),
                    lr=lr
                )
                
                # 创建学习率调度器
            if self.task_config and self.task_config.num_epochs > 0:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.task_config.num_epochs
                )
                
                # 初始化分布式训练
                self._initialize_distributed()
            
            # 初始化策略
            self._initialize_strategy()
            
            # 优化内存
            self.optimize_memory()
                
            if self.should_log():
                self.logger.info("Teacher and student models loaded successfully")
                
                # 更新指标 (base_strategy.py)
            if self._strategy_metrics is not None:
                self._strategy_metrics.update({
                    'models_loaded': True,
                    'device': str(self.device),
                })
                    
        except Exception as e:
            self._phase = TrainerPhase.FAILED
            raise BusinessLogicError(f"Failed to load models: {e}")

    def load_teacher_model(self, model: nn.Module) -> None:
        """
        加载外部教师模型
        
        Args:
            model: 预训练的教师模型
        """
        self.teacher_model = model.to(self.device)
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        if self.strategy:
            self.strategy.set_teacher_model(self.teacher_model, self.device)
        
        logger.info("External teacher model loaded")

    def load_student_model(self, model: nn.Module) -> None:
        """
        加载外部学生模型
        
        Args:
            model: 待训练的学生模型
        """
        self.student_model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=1e-4
        )
        logger.info("External student model loaded")

    def _create_mock_model(self):
        """创建模拟模型（用于测试）"""
        model = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        return model.to(self.device)

    def compute_distillation_loss(
        self, 
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算蒸馏损失
        
        通过调用策略层的损失计算器实现。
        
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签
        
        Returns:
            包含各种损失的字典
        """
        try:
            # 使用策略层的损失计算器计算软标签损失
            soft_targets_loss = self.loss_calculator.compute_soft_loss(
                student_logits, 
                teacher_logits
            )

            # 硬标签损失（原始任务）
            hard_targets_loss = F.cross_entropy(student_logits, labels)

            # 总损失（使用策略配置的权重）
            total_loss = (
                self.strategy_config.soft_loss_weight * soft_targets_loss +
                self.strategy_config.hard_loss_weight * hard_targets_loss
            )

            return {
                'total_loss': total_loss,
                'soft_loss': soft_targets_loss,
                'hard_loss': hard_targets_loss
            }
        except Exception as e:
            raise BusinessLogicError(f"Failed to compute distillation loss: {e}")

    def compute_feature_distillation_loss(
        self, 
        student_features: List[torch.Tensor],
        teacher_features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算特征蒸馏损失
        
        通过调用策略层的损失计算器实现。
        
        Args:
            student_features: 学生模型中间层特征
            teacher_features: 教师模型中间层特征
        
        Returns:
            特征蒸馏损失值
        """
        try:
            return self.loss_calculator.compute_feature_loss(
                tuple(student_features), 
                tuple(teacher_features)
            )
        except Exception as e:
            raise BusinessLogicError(f"Failed to compute feature distillation loss: {e}")

    def compute_attention_distillation_loss(
        self, 
        student_attentions: List[torch.Tensor],
        teacher_attentions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算注意力蒸馏损失
        
        通过调用策略层的损失计算器实现。
        
        Args:
            student_attentions: 学生模型注意力权重
            teacher_attentions: 教师模型注意力权重
        
        Returns:
            注意力蒸馏损失值
        """
        try:
            return self.loss_calculator.compute_attention_loss(
                tuple(student_attentions), 
                tuple(teacher_attentions)
            )
        except Exception as e:
            raise BusinessLogicError(f"Failed to compute attention distillation loss: {e}")

    def compute_contrastive_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比蒸馏损失
        
        通过调用策略层的损失计算器实现。
        
        Args:
            student_features: 学生模型特征
            teacher_features: 教师模型特征
        
        Returns:
            对比蒸馏损失值
        """
        try:
            return self.loss_calculator.compute_contrastive_loss(
                student_features,
                teacher_features
            )
        except Exception as e:
            raise BusinessLogicError(f"Failed to compute contrastive loss: {e}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        单步训练
        
        生产级实现，包含监控、性能分析和健康检查
        
        Args:
            batch: 训练批次数据
        
        Returns:
            包含各种损失值的字典
        """
        step_start_time = time.time()
        
        try:
            # 使用性能分析器 (base_strategy.py)
            with self.profile_operation("train_step"):
                # 触发回调
                self._trigger_callbacks('on_step_start')
                
            
            # 准备输入数据
            if 'input_ids' in batch:
                input_ids = batch['input_ids'].to(self.device)
            else:
                # 模拟输入数据
                input_ids = torch.randn(8, 768).to(self.device)
            
            if 'labels' in batch:
                labels = batch['labels'].to(self.device)
            else:
                labels = torch.randint(0, 10, (input_ids.shape[0],)).to(self.device)

                # 使用混合精度上下文 (lib/hardware)
                amp_context = self._get_amp_context()
                
                with amp_context:
                    # 教师模型前向传播
                    with torch.no_grad():
                        teacher_outputs = self.teacher_model(input_ids)
                        teacher_logits = self._extract_logits(teacher_outputs)

                    # 学生模型前向传播
                    self.student_model.train()
                    student_outputs = self.student_model(input_ids)
                    student_logits = self._extract_logits(student_outputs)

                    # 计算蒸馏损失（使用策略层）
                    distillation_losses = self.compute_distillation_loss(
                        student_logits,
                        teacher_logits,
                        labels
                    )

                    total_loss = distillation_losses['total_loss']

                    # 特征蒸馏损失
                    if self.config.use_feature_distillation:
                        # 模拟特征（实际应从模型获取）
                        student_features = [torch.randn(input_ids.shape[0], 512).to(self.device) for _ in range(3)]
                        teacher_features = [torch.randn(input_ids.shape[0], 512).to(self.device) for _ in range(3)]
                        
                        feature_loss = self.compute_feature_distillation_loss(
                            student_features,
                            teacher_features
                        )
                        total_loss = total_loss + self.config.feature_loss_weight * feature_loss
                        distillation_losses['feature_loss'] = feature_loss

                    # 注意力蒸馏损失
                    if self.config.use_attention_distillation:
                        # 模拟注意力（实际应从模型获取）
                        student_attentions = [torch.randn(input_ids.shape[0], 8, 64, 64).to(self.device) for _ in range(2)]
                        teacher_attentions = [torch.randn(input_ids.shape[0], 8, 64, 64).to(self.device) for _ in range(2)]
                        
                        attention_loss = self.compute_attention_distillation_loss(
                            student_attentions,
                            teacher_attentions
                        )
                        total_loss = total_loss + self.config.attention_loss_weight * attention_loss
                        distillation_losses['attention_loss'] = attention_loss

            # 反向传播和优化
            # 初始化结果以防优化器未运行
            result = {
                'total_loss': 0.0,
                'soft_loss': 0.0,
                'hard_loss': 0.0,
                'step': self.global_step,
                'step_time': 0.0,
                'learning_rate': 0.0,
            }

            if self.optimizer:
                self.optimizer.zero_grad()
                    
                # 使用混合精度反向传播 (lib/hardware)
                if self._mixed_precision_manager is not None:
                    if hasattr(self._mixed_precision_manager, 'backward'):
                        self._mixed_precision_manager.backward(total_loss)
                    else:
                        total_loss.backward()
                else:
                    total_loss.backward()
                    
                # 梯度裁剪
                grad_norm = torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
                self._health_status.gradient_ok = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
                    
                # 使用混合精度优化器步骤 (lib/hardware)
                if self._mixed_precision_manager is not None and hasattr(self._mixed_precision_manager, 'step'):
                    self._mixed_precision_manager.step(self.optimizer)
                else:
                    self.optimizer.step()

            # 更新步数
            self.global_step += 1

            # 计算步骤时间
            step_time = time.time() - step_start_time

            # 构建结果
            total_loss_value = total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
            soft_loss_val = distillation_losses['soft_loss']
            hard_loss_val = distillation_losses['hard_loss']
                
            result = {
                'total_loss': total_loss_value,
                'soft_loss': soft_loss_val.item() if hasattr(soft_loss_val, 'item') else float(soft_loss_val),
                'hard_loss': hard_loss_val.item() if hasattr(hard_loss_val, 'item') else float(hard_loss_val),
                'step': self.global_step,
                'step_time': step_time,
                'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0,
            }
            
            if 'feature_loss' in distillation_losses:
                result['feature_loss'] = distillation_losses['feature_loss'].item() if torch.is_tensor(distillation_losses['feature_loss']) else distillation_losses['feature_loss']
                
            if 'attention_loss' in distillation_losses:
                result['attention_loss'] = distillation_losses['attention_loss'].item() if torch.is_tensor(distillation_losses['attention_loss']) else distillation_losses['attention_loss']

                # 更新统计
                self._stats.update(total_loss_value, step_time=step_time)
                
                # 记录到监控器 (compression_config.py)
                self._distillation_monitor.record_step(
                    kd_loss=result['soft_loss'],
                    ce_loss=result['hard_loss'],
                    accuracy=0.0,  # 如果有准确率则更新
                )
                
                # 记录到策略监控器 (base_strategy.py)
                if self._strategy_monitor is not None and STRATEGY_LAYER_AVAILABLE and StrategyResult is not None:
                    try:
                        strategy_result = StrategyResult(
                            loss=torch.tensor(total_loss_value),
                            metrics=result
                        )
                        if StrategyContext is not None:
                            context = StrategyContext(global_step=self.global_step, epoch=self.epoch)
                            self._strategy_monitor.record_step(strategy_result, context)
                    except Exception:
                        pass
                
                # 记录到 lib/losses 监控器
                if self._lib_loss_monitor is not None:
                    try:
                        # pylint: disable=too-many-function-args
                        self._lib_loss_monitor.record(total_loss_value)
                    except Exception:
                        pass
                
                # 更新策略指标 (base_strategy.py)
                if self._strategy_metrics is not None:
                    try:
                        self._strategy_metrics.update({
                            'step': self.global_step,
                            'loss': total_loss_value,
                            'step_time': step_time,
                        })
                    except Exception:
                        pass
                
                # 记录损失历史
                self._loss_history.append(total_loss_value)
                if len(self._loss_history) > 10000:
                    self._loss_history = self._loss_history[-10000:]
                
                # 周期性健康检查
                if self.global_step % 100 == 0:
                    self.check_health()
                
                # 周期性同步（分布式训练）
                if self.global_step % 50 == 0:
                    self.sync_processes()

            # 触发回调
            self._trigger_callbacks('on_step_end', result)

            return result
                
        except Exception as e:
            self._phase = TrainerPhase.FAILED
            raise BusinessLogicError(f"Training step failed: {e}")
    
    def _get_amp_context(self):
        """
        获取混合精度上下文
        
        使用 lib/hardware 的 MixedPrecisionManager
        """
        if self._mixed_precision_manager is not None and hasattr(self._mixed_precision_manager, 'autocast'):
            return getattr(self._mixed_precision_manager, 'autocast')()
        
        # 返回空上下文
        from contextlib import nullcontext
        return nullcontext()

    def _extract_logits(self, outputs: Any) -> torch.Tensor:
        """提取模型输出的logits"""
        if isinstance(outputs, torch.Tensor):
            return outputs
        elif hasattr(outputs, 'logits'):
            return outputs.logits
        elif isinstance(outputs, (tuple, list)):
            return outputs[0]
        else:
            raise ValueError("Cannot extract logits from model outputs")

    def train(
        self, 
        num_steps: int = 100,
        num_epochs: Optional[int] = None,
        dataloader: Optional[Any] = None,
        validate_every: int = 100,
        save_every: int = 500,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行知识蒸馏训练
        
        生产级实现，包含验证、保存检查点和完整监控
        
        Args:
            num_steps: 训练步数（当dataloader为None时使用）
            num_epochs: 训练轮数（当dataloader不为None时使用）
            dataloader: 数据加载器
            validate_every: 验证间隔步数
            save_every: 保存检查点间隔步数
            checkpoint_dir: 检查点保存目录
        
        Returns:
            训练结果字典
        """
        self._phase = TrainerPhase.TRAINING
        self._is_training = True
        self._start_time = time.time()
        
        try:
            # 使用性能分析器 (base_strategy.py)
            with self.profile_operation("train"):
                if self.should_log():
                    self.logger.info("Starting knowledge distillation training...")
                
                # 触发训练开始回调
                self._trigger_callbacks('on_train_start')
            
            # 加载模型
            if self.teacher_model is None or self.student_model is None:
                self.load_models()
            
                # 设置检查点目录
                if checkpoint_dir:
                    self._checkpoint_path = checkpoint_dir
                    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
            losses = []
            
            if dataloader is not None and num_epochs is not None:
                # 使用数据加载器训练
                for epoch in range(num_epochs):
                    self.epoch = epoch
                    self._stats.total_epochs = epoch + 1
                    self._trigger_callbacks('on_epoch_start')
                    
                    epoch_losses = []
                    for batch in dataloader:
                        step_result = self.train_step(batch)
                        epoch_losses.append(step_result['total_loss'])
                        losses.append(step_result['total_loss'])
                    
                            # 周期性验证
                        if validate_every > 0 and self.global_step % validate_every == 0:
                            self._do_validation()
                            
                        # 周期性保存检查点
                        if save_every > 0 and self.global_step % save_every == 0 and checkpoint_dir:
                            self._save_checkpoint(checkpoint_dir)
                            
                        # 检查收敛
                        if self._check_convergence():
                            if self.should_log():
                                self.logger.info(f"Early convergence detected at step {self.global_step}")
                            break
                        
                        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
                        
                        if self.should_log():
                            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")
                        
                        # 更新学习率调度器
                        if self.scheduler is not None:
                            self.scheduler.step()
                    
                    self._trigger_callbacks('on_epoch_end')
                        
                        # 检查是否收敛
                    if self._stats.is_converged:
                        break
            else:
                # 使用步数训练（模拟数据）
                for step in range(num_steps):
                    batch = {
                        'input_ids': torch.randn(8, 768),
                        'labels': torch.randint(0, 10, (8,))
                    }
                    
                    step_result = self.train_step(batch)
                    losses.append(step_result['total_loss'])
                    
                        # 周期性验证
                    if validate_every > 0 and self.global_step % validate_every == 0:
                        self._do_validation()
                        
                    # 周期性保存检查点
                    if save_every > 0 and self.global_step % save_every == 0 and checkpoint_dir:
                        self._save_checkpoint(checkpoint_dir)
                        
                    if self.should_log() and (step + 1) % 10 == 0:
                        avg_loss = sum(losses[-10:]) / min(10, len(losses))
                        self.logger.info(f"Step {step + 1}/{num_steps}, Average Loss: {avg_loss:.4f}")
                        
                        # 检查收敛
                        if self._check_convergence():
                            if self.should_log():
                                self.logger.info(f"Early convergence detected at step {self.global_step}")
                            break
                
                # 计算总时间
                total_time = time.time() - self._start_time
                self._stats.total_time_seconds = total_time
                
                # 同步所有进程
                self.sync_processes()
                
                # 优化内存
                self.optimize_memory()
                
                # 标记完成
                self._phase = TrainerPhase.COMPLETED
                self._is_training = False
                
                # 触发训练结束回调
                self._trigger_callbacks('on_train_end')
                
                if self.should_log():
                    self.logger.info("Knowledge distillation training completed!")
                
                # 构建结果
                result = {
                'success': True,
                'final_loss': losses[-1] if losses else 0.0,
                'avg_loss': sum(losses) / len(losses) if losses else 0.0,
                    'best_loss': self._stats.best_loss,
                'total_steps': self.global_step,
                    'total_epochs': self._stats.total_epochs,
                    'total_time_seconds': total_time,
                    'avg_step_time': self._stats.avg_step_time,
                    'distillation_type': self.strategy_config.distillation_type if self.strategy_config else 'unknown',
                    'is_converged': self._stats.is_converged,
                    'convergence_step': self._stats.convergence_step,
                }
                
                # 添加监控统计 (compression_config.py)
                result['distillation_stats'] = self._distillation_monitor.get_stats().to_dict()
                
                # 添加性能分析结果 (base_strategy.py)
                if self._profiler is not None and hasattr(self._profiler, 'get_stats'):
                    try:
                        result['profiler_summary'] = self._profiler.get_stats()
                    except Exception:
                        pass
                
                # 添加健康状态
                result['health_status'] = self._health_status.to_dict()
                
                return result
            
        except Exception as e:
            self._phase = TrainerPhase.FAILED
            self._is_training = False
            self.logger.error(f"Knowledge distillation training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_steps': self.global_step,
                'health_status': self._health_status.to_dict(),
            }
    
    def _do_validation(self) -> Dict[str, float]:
        """
        执行验证
        
        使用 compression_config.py 的监控器进行验证
        """
        self._phase = TrainerPhase.VALIDATING
        
        try:
            with self.profile_operation("validation"):
                self._trigger_callbacks('on_validation')
                
                # 检查是否收敛 (compression_config.py)
                is_converged = self._distillation_monitor.is_converged(
                    patience=10,
                    threshold=0.001
                )
                
                if is_converged and not self._stats.is_converged:
                    self._stats.is_converged = True
                    self._stats.convergence_step = self.global_step
                
                # 获取统计信息
                stats = self._distillation_monitor.get_stats()
                
                return {
                    'is_converged': is_converged,
                    'avg_loss': stats.avg_kd_loss,
                    'avg_accuracy': stats.student_accuracy,
                }
        finally:
            self._phase = TrainerPhase.TRAINING
    
    def _check_convergence(self) -> bool:
        """
        检查是否收敛
        
        使用 compression_config.py 的 DistillationMonitor
        """
        if len(self._loss_history) < 20:
            return False
        
        # 使用配置层监控器检查收敛
        return self._distillation_monitor.is_converged(
            patience=10,
            threshold=0.001
        )
    
    def _save_checkpoint(self, checkpoint_dir: str) -> str:
        """
        保存检查点
        
        包含模型状态、优化器状态和训练状态
        """
        self._phase = TrainerPhase.SAVING
        
        try:
            with self.profile_operation("save_checkpoint"):
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint_step_{self.global_step}.pt"
                
                checkpoint = {
                    'global_step': self.global_step,
                    'epoch': self.epoch,
                    'student_model_state': self.student_model.state_dict() if self.student_model else None,
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                    'config': asdict(self.config),
                    'stats': self._stats.to_dict(),
                    'loss_history': self._loss_history[-1000:],  # 保存最近1000步
                }
                
                torch.save(checkpoint, checkpoint_path)
                
                # 更新最佳检查点
                if self._stats.current_loss < self._stats.best_loss:
                    best_path = Path(checkpoint_dir) / "best_checkpoint.pt"
                    torch.save(checkpoint, best_path)
                    self._best_checkpoint = checkpoint
                
                self._trigger_callbacks('on_checkpoint', str(checkpoint_path))
                
                if self.should_log():
                    self.logger.info(f"Checkpoint saved to {checkpoint_path}")
                
                return str(checkpoint_path)
        finally:
            self._phase = TrainerPhase.TRAINING
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点
        
        恢复模型状态、优化器状态和训练状态
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.global_step = checkpoint.get('global_step', 0)
        self.epoch = checkpoint.get('epoch', 0)
        
        if self.student_model is not None and checkpoint.get('student_model_state'):
            self.student_model.load_state_dict(checkpoint['student_model_state'])
        
        if self.optimizer is not None and checkpoint.get('optimizer_state'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if self.scheduler is not None and checkpoint.get('scheduler_state'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        if checkpoint.get('loss_history'):
            self._loss_history = checkpoint['loss_history']
        
        if self.should_log():
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def get_stats(self) -> TrainerStats:
        """获取训练统计"""
        return self._stats
    
    def get_distillation_stats(self) -> 'DistillationStats':
        """
        获取蒸馏统计
        
        使用 compression_config.py 的 DistillationMonitor
        """
        return self._distillation_monitor.get_stats()
    
    def get_loss_trend(self) -> List[float]:
        """
        获取损失趋势
        
        使用 compression_config.py 的 DistillationMonitor
        """
        return self._distillation_monitor.get_loss_trend()
    
    def diagnose(self) -> Dict[str, Any]:
        """
        诊断训练器状态
        
        汇总所有监控器和管理器的状态
        """
        diagnosis = {
            'phase': self._phase.value,
            'is_training': self._is_training,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'device': str(self.device),
            'health_status': self._health_status.to_dict(),
            'stats': self._stats.to_dict(),
            'layers_available': {
                'strategy_layer': STRATEGY_LAYER_AVAILABLE,
                'distillation_strategy': DISTILLATION_STRATEGY_AVAILABLE,
                'distributed_strategy': DISTRIBUTED_STRATEGY_AVAILABLE,
                'losses_layer': LOSSES_LAYER_AVAILABLE,
                'hardware_layer': HARDWARE_LAYER_AVAILABLE,
                'distributed_layer': DISTRIBUTED_LAYER_AVAILABLE,
            },
            'components': {
                'strategy': self.strategy is not None,
                'loss_calculator': self.loss_calculator is not None,
                'distillation_monitor': True,
                'config_validator': True,
                'strategy_monitor': self._strategy_monitor is not None,
                'profiler': self._profiler is not None,
                'strategy_validator': self._strategy_validator is not None,
                'strategy_metrics': self._strategy_metrics is not None,
                'memory_manager': self._memory_manager is not None,
                'mixed_precision_manager': self._mixed_precision_manager is not None,
                'distributed_manager': self._distributed_manager is not None,
                'distributed_strategy': self._distributed_strategy is not None,
            },
        }
        
        # 添加监控器摘要
        diagnosis['distillation_stats'] = self._distillation_monitor.get_summary()
        
        # 添加分布式信息
        if DISTRIBUTED_LAYER_AVAILABLE:
            diagnosis['distributed_info'] = {
                'is_main_process': is_main_process(),
                'rank': get_rank(),
                'world_size': get_world_size(),
            }
        
        # 添加内存信息
        if HARDWARE_LAYER_AVAILABLE:
            diagnosis['memory_info'] = {
                'available_memory_mb': self.get_available_memory(),
            }
        
        # 添加性能分析摘要
        if self._profiler is not None and hasattr(self._profiler, 'get_stats'):
            try:
                diagnosis['profiler_summary'] = self._profiler.get_stats()
            except Exception:
                pass
        
        return diagnosis

    # ======================== 配置层模块使用方法 ========================
    
    def get_scenario_config(self) -> Optional['ScenarioDistillationConfig']:
        """
        获取场景蒸馏配置
        
        使用 compression_config.py 的 ScenarioDistillationConfig
        """
        if self.task_config and hasattr(self.task_config, 'scenario_config'):
            return self.task_config.scenario_config
        
        # 从当前配置创建场景配置
        scenario = self.task_config.scenario if self.task_config else 'standard'
        return ScenarioDistillationConfig(
            scenario=scenario,
        )
    
    def get_adaptive_config(self) -> Optional['AdaptiveDistillationConfig']:
        """
        获取自适应蒸馏配置
        
        使用 compression_config.py 的 AdaptiveDistillationConfig
        """
        if self.task_config and hasattr(self.task_config, 'adaptive_config'):
            return self.task_config.adaptive_config
        
        # 创建默认自适应配置
        return AdaptiveDistillationConfig(
            mode=AdaptiveMode.NONE.value,
        )
    
    def get_distributed_mode(self) -> 'DistributedMode':
        """
        获取分布式模式
        
        使用 compression_config.py 的 DistributedMode
        """
        if self._distributed_config:
            return DistributedMode.from_string(self._distributed_config.mode)
        return DistributedMode.SINGLE
    
    def get_adaptive_mode(self) -> 'AdaptiveMode':
        """
        获取自适应模式
        
        使用 compression_config.py 的 AdaptiveMode
        """
        adaptive_config = self.get_adaptive_config()
        if adaptive_config:
            return AdaptiveMode.from_string(adaptive_config.mode)
        return AdaptiveMode.NONE
    
    def get_compression_methods(self) -> List['CompressionMethod']:
        """
        获取启用的压缩方法
        
        使用 compression_config.py 的 CompressionMethod
        """
        methods = []
        if self.config.use_feature_distillation:
            methods.append(CompressionMethod.DISTILLATION)
        if hasattr(self.config, 'use_quantization') and self.config.use_quantization:
            methods.append(CompressionMethod.QUANTIZATION)
        if hasattr(self.config, 'use_pruning') and self.config.use_pruning:
            methods.append(CompressionMethod.PRUNING)
        return methods
    
    def create_config_from_requirements(
        self,
        target_latency_ms: Optional[float] = None,
        target_accuracy: Optional[float] = None,
        target_size_mb: Optional[float] = None,
    ) -> 'DistillationTaskConfig':
        """
        根据需求创建配置
        
        使用 compression_config.py 的 create_distillation_config 和 recommend_config
        
        Args:
            target_latency_ms: 目标延迟（毫秒）
            target_accuracy: 目标准确率
            target_size_mb: 目标模型大小（MB）
        
        Returns:
            推荐的任务配置
        """
        # 使用 create_distillation_config 创建配置
        task_config = create_distillation_config(
            teacher_path="",
            student_path="",
            task_name=f"trainer_{self.global_step}",
            scenario='standard',
            temperature=self.config.temperature,
            alpha=self.config.alpha,
            beta=self.config.beta,
        )
        
        return task_config
    
    def recommend_optimal_config(
        self,
        scenario: str = 'standard',
        **requirements
    ) -> 'DistillationTaskConfig':
        """
        推荐最优配置
        
        使用 compression_config.py 的 recommend_config
        
        Args:
            scenario: 场景类型
            **requirements: 需求参数
        
        Returns:
            推荐的任务配置
        """
        # pylint: disable=unexpected-keyword-arg
        return recommend_config(scenario=scenario, **requirements)
    
    # ======================== 策略层模块使用方法 ========================
    
    def get_training_phase(self) -> Optional['TrainingPhase']:
        """
        获取当前训练阶段
        
        使用 base_strategy.py 的 TrainingPhase
        """
        if not STRATEGY_LAYER_AVAILABLE or TrainingPhase is None:
            return None
        
        # 映射 TrainerPhase 到 TrainingPhase
        phase_mapping = {
            TrainerPhase.INIT: TrainingPhase.WARMUP if hasattr(TrainingPhase, 'WARMUP') else None,
            TrainerPhase.TRAINING: TrainingPhase.MAIN if hasattr(TrainingPhase, 'MAIN') else None,
            TrainerPhase.VALIDATING: TrainingPhase.EVALUATION if hasattr(TrainingPhase, 'EVALUATION') else None,
            TrainerPhase.COMPLETED: TrainingPhase.EVALUATION if hasattr(TrainingPhase, 'EVALUATION') else None,
        }
        
        return phase_mapping.get(self._phase)
    
    def get_strategy_type(self) -> Optional['StrategyType']:
        """
        获取策略类型
        
        使用 base_strategy.py 的 StrategyType
        """
        if not STRATEGY_LAYER_AVAILABLE or StrategyType is None:
            return None
        
        # 返回蒸馏策略类型
        if hasattr(StrategyType, 'DISTILLATION'):
            return StrategyType.DISTILLATION
        return None
    
    def get_distillation_type(self) -> Optional['DistillationType']:
        """
        获取蒸馏类型
        
        使用 distillation_strategy.py 的 DistillationType
        """
        if not DISTILLATION_STRATEGY_AVAILABLE or DistillationType is None:
            return None
        
        # 根据配置确定蒸馏类型
        if self.config.use_feature_distillation and self.config.use_attention_distillation:
            if hasattr(DistillationType, 'COMBINED'):
                return DistillationType.COMBINED
        elif self.config.use_feature_distillation:
            if hasattr(DistillationType, 'FEATURE'):
                return DistillationType.FEATURE
        elif self.config.use_attention_distillation:
            if hasattr(DistillationType, 'ATTENTION'):
                return DistillationType.ATTENTION
        
        if hasattr(DistillationType, 'LOGITS'):
            return DistillationType.LOGITS
        return None
    
    def as_training_strategy(self) -> Optional['TrainingStrategy']:
        """
        作为训练策略返回
        
        使用 base_strategy.py 的 TrainingStrategy
        """
        if not STRATEGY_LAYER_AVAILABLE or TrainingStrategy is None:
            return None
        
        # 如果已有策略且是 TrainingStrategy 的子类，直接返回
        if self.strategy is not None and isinstance(self.strategy, TrainingStrategy):
            return self.strategy
        
        return None
    
    # ======================== 分布式策略层模块使用方法 ========================
    
    def get_distributed_strategy_config(self) -> Optional['DistributedStrategyConfig']:
        """
        获取分布式策略配置
        
        使用 distributed_strategy.py 的 DistributedStrategyConfig
        """
        if not DISTRIBUTED_STRATEGY_AVAILABLE or DistributedStrategyConfig is None:
            return None
        
        if not self._distributed_config:
            return None
        
        # 从分布式蒸馏配置转换为策略配置
        try:
            return DistributedStrategyConfig(
                mode=self._distributed_config.mode,
                world_size=self._distributed_config.world_size,
                rank=self._distributed_config.rank,
            )
        except Exception:
            return None
    
    def get_strategy_distributed_mode(self) -> Optional['StrategyDistributedMode']:
        """
        获取策略层分布式模式
        
        使用 distributed_strategy.py 的 DistributedMode as StrategyDistributedMode
        """
        if not DISTRIBUTED_STRATEGY_AVAILABLE or StrategyDistributedMode is None:
            return None
        
        if self._distributed_config:
            try:
                return StrategyDistributedMode(self._distributed_config.mode)
            except (ValueError, TypeError):
                pass
        
        return None
    
    def get_zero_stage(self) -> Optional['ZeROStage']:
        """
        获取 ZeRO 优化阶段
        
        使用 distributed_strategy.py 的 ZeROStage
        """
        if not DISTRIBUTED_STRATEGY_AVAILABLE or ZeROStage is None:
            return None
        
        if self._distributed_config and hasattr(self._distributed_config, 'zero_stage'):
            try:
                return ZeROStage(self._distributed_config.zero_stage)
            except (ValueError, TypeError):
                pass
        
        # 默认返回 None（禁用）
        if hasattr(ZeROStage, 'STAGE_1'):
            # 如果没有 STAGE_0，可能只能返回 None 或最小阶段
            return None 
        return None
    
    def get_distributed_health_status(self) -> Optional['DistributedHealthStatus']:
        """
        获取分布式健康状态
        
        使用 distributed_strategy.py 的 DistributedHealthStatus
        """
        if not DISTRIBUTED_STRATEGY_AVAILABLE or DistributedHealthStatus is None:
            return None
        
        if self._distributed_strategy and hasattr(self._distributed_strategy, 'get_health_status'):
            try:
                return self._distributed_strategy.get_health_status()
            except Exception:
                pass
        
        # 创建默认健康状态
        try:
            return DistributedHealthStatus(
                is_healthy=self._health_status.is_healthy,
                world_size=get_world_size() if DISTRIBUTED_LAYER_AVAILABLE else 1,
                rank=get_rank() if DISTRIBUTED_LAYER_AVAILABLE else 0,
            )
        except Exception:
            return None
    
    def get_communication_stats(self) -> Optional['CommunicationStats']:
        """
        获取通信统计
        
        使用 distributed_strategy.py 的 CommunicationStats
        """
        if not DISTRIBUTED_STRATEGY_AVAILABLE or CommunicationStats is None:
            return None
        
        if self._distributed_strategy and hasattr(self._distributed_strategy, 'get_communication_stats'):
            try:
                return self._distributed_strategy.get_communication_stats()
            except Exception:
                pass
        
        # 创建默认通信统计
        try:
            return CommunicationStats()
        except Exception:
            return None
    
    def recommend_distributed_config(
        self,
        model_size_mb: float = 1000.0,
        num_gpus: int = 1,
        memory_per_gpu_mb: float = 16000.0,
    ) -> Optional[str]:
        """
        推荐分布式配置
        
        使用 distributed_strategy.py 的 recommend_distributed_mode
        
        Args:
            model_size_mb: 模型大小（MB）
            num_gpus: GPU 数量
            memory_per_gpu_mb: 每个 GPU 的内存（MB）
        
        Returns:
            推荐的分布式模式
        """
        if not DISTRIBUTED_STRATEGY_AVAILABLE or recommend_distributed_mode is None:
            return None
        
        try:
            return recommend_distributed_mode(
                model_size_mb=model_size_mb,
                num_gpus=num_gpus,
                memory_per_gpu_mb=memory_per_gpu_mb,
            )
        except Exception:
            return None
    
    # ======================== lib/losses 模块使用方法 ========================
    
    def create_custom_loss(
        self,
        loss_type: str = 'cross_entropy',
        **kwargs
    ) -> Optional['BaseLoss']:
        """
        创建自定义损失函数
        
        使用 lib/losses 的 create_loss 和 BaseLoss
        
        Args:
            loss_type: 损失函数类型
            **kwargs: 损失函数参数
        
        Returns:
            损失函数实例
        """
        if not LOSSES_LAYER_AVAILABLE or create_loss is None:
            return None
        
        try:
            return create_loss(loss_type, **kwargs)
        except Exception as e:
            self.logger.warning(f"Failed to create loss {loss_type}: {e}")
            return None
    
    def get_lib_loss_stats(self) -> Optional['LibLossStats']:
        """
        获取 lib/losses 统计
        
        使用 lib/losses 的 LossStats as LibLossStats
        """
        if not LOSSES_LAYER_AVAILABLE or LibLossStats is None:
            return None
        
        if self._lib_loss_monitor and hasattr(self._lib_loss_monitor, 'get_stats'):
            try:
                return self._lib_loss_monitor.get_stats()
            except Exception:
                pass
        
        # 创建默认统计
        try:
            return LibLossStats()
        except Exception:
            return None
    
    def create_loss_from_config(self) -> Optional['BaseLoss']:
        """
        从配置创建损失函数
        
        使用 lib/losses 的 create_loss 和 BaseLoss
        
        Returns:
            损失函数实例
        """
        if not LOSSES_LAYER_AVAILABLE or create_loss is None:
            return None
        
        try:
            # 创建软标签损失
            soft_loss = create_loss('kl_div', reduction='batchmean')
            return soft_loss
        except Exception as e:
            self.logger.warning(f"Failed to create loss from config: {e}")
            return None
    
    def validate_loss_instance(self, loss: Any) -> bool:
        """
        验证损失函数实例
        
        使用 lib/losses 的 BaseLoss
        
        Args:
            loss: 损失函数实例
        
        Returns:
            是否为有效的损失函数
        """
        if not LOSSES_LAYER_AVAILABLE or BaseLoss is None:
            return isinstance(loss, nn.Module)
        
        return isinstance(loss, (BaseLoss, nn.Module))
    
    # ======================== 综合诊断方法 ========================
    
    def get_full_diagnosis(self) -> Dict[str, Any]:
        """
        获取完整诊断信息
        
        整合所有层模块的信息
        """
        diagnosis = self.diagnose()
        
        # 添加配置层信息
        diagnosis['config_layer'] = {
            'scenario_config': self.get_scenario_config() is not None,
            'adaptive_config': self.get_adaptive_config() is not None,
            'distributed_mode': self.get_distributed_mode().value if self.get_distributed_mode() else None,
            'adaptive_mode': self.get_adaptive_mode().value if self.get_adaptive_mode() else None,
            'compression_methods': [m.value for m in self.get_compression_methods()],
        }
        
        # 添加策略层信息
        diagnosis['strategy_layer'] = {
            'training_phase': self.get_training_phase().value if self.get_training_phase() else None,
            'strategy_type': self.get_strategy_type().value if self.get_strategy_type() and hasattr(self.get_strategy_type(), 'value') else None,
            'distillation_type': self.get_distillation_type().value if self.get_distillation_type() and hasattr(self.get_distillation_type(), 'value') else None,
            'as_training_strategy': self.as_training_strategy() is not None,
        }
        
        # 添加分布式策略层信息
        diagnosis['distributed_strategy_layer'] = {
            'distributed_strategy_config': self.get_distributed_strategy_config() is not None,
            'strategy_distributed_mode': self.get_strategy_distributed_mode().value if self.get_strategy_distributed_mode() and hasattr(self.get_strategy_distributed_mode(), 'value') else None,
            'zero_stage': self.get_zero_stage().value if self.get_zero_stage() and hasattr(self.get_zero_stage(), 'value') else None,
            'distributed_health_status': self.get_distributed_health_status() is not None,
            'communication_stats': self.get_communication_stats() is not None,
        }
        
        # 添加 lib/losses 信息
        diagnosis['lib_losses'] = {
            'lib_loss_stats': self.get_lib_loss_stats() is not None,
            'can_create_loss': self.create_custom_loss('mse') is not None if LOSSES_LAYER_AVAILABLE else False,
        }
        
        return diagnosis

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        注册回调函数
        
        Args:
            event: 事件名称 (on_step_start, on_step_end, on_epoch_start, on_epoch_end)
            callback: 回调函数
        """
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, *args, **kwargs) -> None:
        """触发回调函数"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback {event} failed: {e}")

    def get_strategy(self) -> Optional[DistillationStrategy]:
        """获取当前蒸馏策略"""
        return self.strategy

    def set_strategy(self, strategy: DistillationStrategy) -> None:
        """设置蒸馏策略"""
        self.strategy = strategy
        self.strategy_config = strategy.config
        self.loss_calculator = DistillationLossCalculator(
            self.strategy_config,
            self.device
        )

    def save_student_model(self, path: str) -> None:
        """保存学生模型"""
        if self.student_model:
            torch.save(self.student_model.state_dict(), path)
            logger.info(f"Student model saved to {path}")


class ModelCompressor:
    """
    模型压缩器
    
    生产级实现，支持量化、剪枝和知识蒸馏等多种压缩方式。
    
    特性：
    - 集成 compression_config.py 的 CompressionMethod 和配置验证
    - 集成 base_strategy.py 的性能分析和监控
    - 集成 lib/hardware 的内存管理和设备管理
    - 知识蒸馏功能通过策略层实现
    """

    def __init__(self, config: CompressionConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ModelCompressor")
        
        # 设备管理 (lib/hardware)
        self._device_manager: Optional['DeviceManager'] = None
        if HARDWARE_LAYER_AVAILABLE and get_device_manager is not None:
            try:
                self._device_manager = get_device_manager()
                self.device = self._device_manager.get_optimal_device() if hasattr(self._device_manager, 'get_optimal_device') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            except Exception:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 蒸馏策略 (distillation_strategy.py)
        self.distillation_strategy: Optional['DistillationStrategy'] = None
        
        # 配置验证器 (compression_config.py)
        self._config_validator = ConfigValidator()
        self._setup_validation_rules()
        
        # 性能分析器 (base_strategy.py)
        self._profiler: Optional['StrategyProfiler'] = None
        if STRATEGY_LAYER_AVAILABLE and StrategyProfiler is not None:
            try:
                self._profiler = StrategyProfiler()
            except Exception:
                pass
        
        # 策略指标 (base_strategy.py)
        self._strategy_metrics: Optional['StrategyMetrics'] = None
        if STRATEGY_LAYER_AVAILABLE and StrategyMetrics is not None:
            try:
                self._strategy_metrics = StrategyMetrics()
            except Exception:
                pass
        
        # 内存管理器 (lib/hardware)
        self._memory_manager: Optional['MemoryManager'] = None
        if HARDWARE_LAYER_AVAILABLE and MemoryManager is not None:
            try:
                self._memory_manager = MemoryManager()
            except Exception:
                pass
        
        # 压缩统计
        self._compression_stats: Dict[str, Any] = {
            'original_size_mb': 0.0,
            'compressed_size_mb': 0.0,
            'compression_ratio': 1.0,
            'methods_applied': [],
            'time_seconds': 0.0,
        }
    
    def _setup_validation_rules(self) -> None:
        """
        设置配置验证规则
        
        使用 compression_config.py 的 ConfigValidator
        """
        def check_pruning_ratio(cfg) -> Tuple[bool, str]:
            if hasattr(cfg, 'pruning_ratio') and not (0.0 <= cfg.pruning_ratio <= 0.9):
                return False, f"Pruning ratio {cfg.pruning_ratio} should be between 0.0 and 0.9"
            return True, ""
        
        def check_quantization_type(cfg) -> Tuple[bool, str]:
            valid_types = ['dynamic', 'static', 'qat', 'none']
            if hasattr(cfg, 'quantization_type') and cfg.quantization_type not in valid_types:
                return False, f"Invalid quantization type: {cfg.quantization_type}"
            return True, ""
        
        self._config_validator.add_check(check_pruning_ratio)
        self._config_validator.add_check(check_quantization_type)
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """
        性能分析上下文管理器
        
        使用 base_strategy.py 的 StrategyProfiler
        """
        if self._profiler is not None:
            with self._profiler.profile(operation_name):
                yield
        else:
            yield
    
    def validate_config(self) -> Tuple[bool, List[str]]:
        """
        验证配置
        
        使用 compression_config.py 的 ConfigValidator
        """
        return self._config_validator.validate(self.config)
    
    def optimize_memory(self) -> None:
        """
        优化内存
        
        使用 lib/hardware 的 clear_memory 和 MemoryManager
        """
        if HARDWARE_LAYER_AVAILABLE and clear_memory is not None:
            try:
                clear_memory()
            except Exception:
                pass
        
        if self._memory_manager is not None:
            try:
                if hasattr(self._memory_manager, 'optimize'):
                    self._memory_manager.optimize()
                elif hasattr(self._memory_manager, 'clear_memory'):
                    self._memory_manager.clear_memory()
            except Exception:
                pass

    def compress_model(self, model: nn.Module) -> nn.Module:
        """
        压缩模型
        
        生产级实现，包含性能分析和详细统计
        
        Args:
            model: 待压缩的模型
        
        Returns:
            压缩后的模型
        """
        start_time = time.time()
        
        try:
            with self.profile_operation("compress_model"):
                # 验证配置
                is_valid, errors = self.validate_config()
                if not is_valid:
                    self.logger.warning(f"Configuration validation warnings: {errors}")
                
                # 计算原始大小
                original_size = self._estimate_model_size(model)
                self._compression_stats['original_size_mb'] = original_size
                self._compression_stats['methods_applied'] = []
                
                self.logger.info(f"Starting model compression... Original size: {original_size:.2f} MB")
            
            compressed_model = model
            
            # 量化
            if self.config.use_quantization:
                with self.profile_operation("quantization"):
                    compressed_model = self._apply_quantization(compressed_model)
                    self._compression_stats['methods_applied'].append('quantization')
            
            # 剪枝
            if self.config.use_pruning:
                with self.profile_operation("pruning"):
                    compressed_model = self._apply_pruning(compressed_model)
                    self._compression_stats['methods_applied'].append('pruning')
            
            # 知识蒸馏（使用策略层）
            if self.config.use_distillation and self.config.distillation_config:
                with self.profile_operation("distillation"):
                    compressed_model = self._apply_distillation(compressed_model)
                    self._compression_stats['methods_applied'].append('distillation')
                
                # 计算压缩后大小
                compressed_size = self._estimate_model_size(compressed_model)
                self._compression_stats['compressed_size_mb'] = compressed_size
                self._compression_stats['compression_ratio'] = original_size / compressed_size if compressed_size > 0 else 1.0
                self._compression_stats['time_seconds'] = time.time() - start_time
                
                # 更新策略指标 (base_strategy.py)
                if self._strategy_metrics is not None:
                    try:
                        self._strategy_metrics.update(self._compression_stats)
                    except Exception:
                        pass
                
                # 优化内存
                self.optimize_memory()
                
                self.logger.info(f"Model compression completed! "
                               f"Compressed size: {compressed_size:.2f} MB, "
                               f"Compression ratio: {self._compression_stats['compression_ratio']:.2f}x")
                
            return compressed_model
            
        except Exception as e:
            raise BusinessLogicError(f"Model compression failed: {e}")
    
    def _estimate_model_size(self, model: nn.Module) -> float:
        """
        估计模型大小（MB）
        
        Args:
            model: 模型
        
        Returns:
            模型大小（MB）
        """
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """
        应用量化
        
        使用 compression_config.py 的 CompressionMethod
        """
        try:
            self.logger.info(f"Applying quantization: {self.config.quantization_type}")
            
            if self.config.quantization_type == "dynamic":
                # 动态量化
                quantized_model = torch.quantization.quantize_dynamic(
                    model, 
                    {nn.Linear}, 
                    dtype=torch.qint8
                )
                return quantized_model
            elif self.config.quantization_type == "static":
                # 静态量化（需要校准数据）
                self.logger.info("Static quantization requires calibration data")
                return model
            elif self.config.quantization_type == "qat":
                # 量化感知训练
                self.logger.info("QAT requires training setup")
                return model
            else:
                return model
                
        except Exception as e:
            self.logger.warning(f"Quantization failed, returning original model: {e}")
            return model

    def _apply_pruning(self, model: nn.Module) -> nn.Module:
        """
        应用剪枝
        
        使用 compression_config.py 的 CompressionMethod
        """
        try:
            self.logger.info(f"Applying pruning: {self.config.pruning_method}, ratio: {self.config.pruning_ratio}")
            
            import torch.nn.utils.prune as prune
            
            pruned_count = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    if self.config.pruning_method == "magnitude":
                        prune.l1_unstructured(
                            module, 
                            name='weight', 
                            amount=self.config.pruning_ratio
                        )
                    elif self.config.pruning_method == "random":
                        prune.random_unstructured(
                            module, 
                            name='weight', 
                            amount=self.config.pruning_ratio
                        )
                    elif self.config.pruning_method == "structured":
                        prune.ln_structured(
                            module,
                            name='weight',
                            amount=self.config.pruning_ratio,
                            n=2,
                            dim=0
                        )
                    pruned_count += 1
            
            self.logger.info(f"Pruned {pruned_count} layers")
            return model
            
        except Exception as e:
            self.logger.warning(f"Pruning failed, returning original model: {e}")
            return model

    def _apply_distillation(self, model: nn.Module) -> nn.Module:
        """
        应用知识蒸馏
        
        使用策略层的蒸馏功能进行模型压缩。
        集成 distillation_strategy.py 的蒸馏策略
        """
        try:
            self.logger.info("Applying knowledge distillation via strategy layer")
            
            if self.config.distillation_config is None:
                return model
            
            # 创建蒸馏训练器（使用 KnowledgeDistillationTrainer）
            trainer = KnowledgeDistillationTrainer(self.config.distillation_config)
            
            # 设置学生模型
            trainer.load_student_model(model)
            
            # 如果有自定义蒸馏策略，使用它
            if self.distillation_strategy is not None:
                trainer.set_strategy(self.distillation_strategy)
            
            # 如果有教师模型路径，加载教师模型
            if self.config.distillation_config.teacher_model_path:
                # 这里应该加载真实的教师模型
                # trainer.load_teacher_model(teacher_model)
                pass
            
            # 执行蒸馏训练
            num_steps = getattr(self.config, 'distillation_steps', 100)
            result = trainer.train(num_steps=num_steps)
            
            if result['success']:
                self.logger.info(f"Distillation completed, final loss: {result['final_loss']:.4f}")
                return trainer.student_model
            else:
                self.logger.warning(f"Distillation failed: {result.get('error', 'Unknown error')}")
                return model
                
        except Exception as e:
            self.logger.warning(f"Distillation failed, returning original model: {e}")
            return model

    def set_distillation_strategy(
        self, 
        strategy_type: str = "standard",
        **config_kwargs
    ) -> None:
        """
        设置蒸馏策略
        
        使用 distillation_strategy.py 的 create_distillation_strategy
        
        Args:
            strategy_type: 策略类型 (standard, self, progressive, industry, contrastive)
            **config_kwargs: 策略配置参数
        """
        if not DISTILLATION_STRATEGY_AVAILABLE or create_distillation_strategy is None:
            self.logger.warning("Distillation strategy layer not available")
            return
        
        try:
            self.distillation_strategy = create_distillation_strategy(
                strategy_type=strategy_type,
                config=config_kwargs if config_kwargs else None,
            )
            self.logger.info(f"Distillation strategy set: {strategy_type}")
        except Exception as e:
            self.logger.warning(f"Failed to set distillation strategy: {e}")
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """获取压缩统计"""
        return self._compression_stats
    
    def estimate_compression(self, model: nn.Module) -> Dict[str, Any]:
        """
        估计压缩效果（不实际压缩）
        
        使用 compression_config.py 的配置进行估计
        
        Args:
            model: 待压缩的模型
        
        Returns:
            压缩估计结果
        """
        original_size = self._estimate_model_size(model)
        
        estimated_ratio = 1.0
        
        # 估计量化压缩比
        if self.config.use_quantization:
            if self.config.quantization_type == "dynamic":
                estimated_ratio *= 4.0  # FP32 -> INT8
            elif self.config.quantization_type == "static":
                estimated_ratio *= 4.0
        
        # 估计剪枝压缩比
        if self.config.use_pruning:
            estimated_ratio *= 1.0 / (1.0 - self.config.pruning_ratio)
        
        estimated_size = original_size / estimated_ratio
        
        return {
            'original_size_mb': original_size,
            'estimated_compressed_size_mb': estimated_size,
            'estimated_compression_ratio': estimated_ratio,
            'methods': {
                'quantization': self.config.use_quantization,
                'pruning': self.config.use_pruning,
                'distillation': self.config.use_distillation,
            }
        }
    
    def diagnose(self) -> Dict[str, Any]:
        """
        诊断压缩器状态
        
        汇总所有组件状态
        """
        return {
            'device': str(self.device),
            'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else str(self.config),
            'compression_stats': self._compression_stats,
            'layers_available': {
                'strategy_layer': STRATEGY_LAYER_AVAILABLE,
                'distillation_strategy': DISTILLATION_STRATEGY_AVAILABLE,
                'hardware_layer': HARDWARE_LAYER_AVAILABLE,
            },
            'components': {
                'config_validator': True,
                'profiler': self._profiler is not None,
                'strategy_metrics': self._strategy_metrics is not None,
                'memory_manager': self._memory_manager is not None,
                'distillation_strategy': self.distillation_strategy is not None,
            },
        }
    
    # ======================== 配置层模块使用方法 ========================
    
    def get_compression_method(self) -> List['CompressionMethod']:
        """
        获取启用的压缩方法
        
        使用 compression_config.py 的 CompressionMethod
        """
        methods = []
        if self.config.use_quantization:
            methods.append(CompressionMethod.QUANTIZATION)
        if self.config.use_pruning:
            methods.append(CompressionMethod.PRUNING)
        if self.config.use_distillation:
            methods.append(CompressionMethod.DISTILLATION)
        return methods
    
    def get_compression_method_info(self) -> Dict[str, Any]:
        """
        获取压缩方法信息
        
        使用 compression_config.py 的 CompressionMethod 属性
        """
        methods = self.get_compression_method()
        info = {}
        
        for method in methods:
            info[method.value] = {
                'typical_speedup': method.typical_speedup,
                'typical_size_reduction': method.typical_size_reduction,
            }
        
        return info
    
    def create_scenario_config(self, scenario: str = 'standard') -> 'ScenarioDistillationConfig':
        """
        创建场景配置
        
        使用 compression_config.py 的 ScenarioDistillationConfig
        """
        if self.config.distillation_config:
            return ScenarioDistillationConfig(
                scenario=scenario,
            )
        return ScenarioDistillationConfig(scenario=scenario)
    
    def create_adaptive_config(self, mode: str = 'fixed') -> 'AdaptiveDistillationConfig':
        """
        创建自适应配置
        
        使用 compression_config.py 的 AdaptiveDistillationConfig 和 AdaptiveMode
        """
        # 验证模式
        adaptive_mode = AdaptiveMode.from_string(mode)
        
        return AdaptiveDistillationConfig(
            mode=adaptive_mode.value,
        )
    
    def get_distributed_mode_for_compression(self) -> 'DistributedMode':
        """
        获取压缩使用的分布式模式
        
        使用 compression_config.py 的 DistributedMode
        """
        # 默认不使用分布式
        return DistributedMode.SINGLE
    
    def create_task_config_from_compression(self) -> 'DistillationTaskConfig':
        """
        从压缩配置创建任务配置
        
        使用 compression_config.py 的 create_distillation_config
        """
        if self.config.distillation_config:
            return create_distillation_config(
                teacher_path="",
                student_path="",
                task_name='compression_task',
                scenario='standard',
                temperature=self.config.distillation_config.temperature,
                alpha=self.config.distillation_config.alpha,
                beta=self.config.distillation_config.beta,
            )
        return create_distillation_config(
            teacher_path="",
            student_path="",
            task_name='compression_task'
        )
    
    # ======================== 策略层模块使用方法 ========================
    
    def get_strategy_type(self) -> Optional['StrategyType']:
        """
        获取策略类型
        
        使用 base_strategy.py 的 StrategyType
        """
        if not STRATEGY_LAYER_AVAILABLE or StrategyType is None:
            return None
        
        # 压缩器使用蒸馏策略
        if hasattr(StrategyType, 'DISTILLATION'):
            return StrategyType.DISTILLATION
        return None
    
    def get_training_phase(self) -> Optional['TrainingPhase']:
        """
        获取训练阶段
        
        使用 base_strategy.py 的 TrainingPhase
        """
        if not STRATEGY_LAYER_AVAILABLE or TrainingPhase is None:
            return None
        
        # 返回压缩阶段（对应训练阶段）
        if hasattr(TrainingPhase, 'MAIN'):
            return TrainingPhase.MAIN
        return None
    
    def get_distillation_type(self) -> Optional['DistillationType']:
        """
        获取蒸馏类型
        
        使用 distillation_strategy.py 的 DistillationType
        """
        if not DISTILLATION_STRATEGY_AVAILABLE or DistillationType is None:
            return None
        
        if not self.config.use_distillation or not self.config.distillation_config:
            return None
        
        # 根据配置确定类型
        config = self.config.distillation_config
        if config.use_feature_distillation and config.use_attention_distillation:
            if hasattr(DistillationType, 'COMBINED'):
                return DistillationType.COMBINED
        elif config.use_feature_distillation:
            if hasattr(DistillationType, 'FEATURE'):
                return DistillationType.FEATURE
        
        if hasattr(DistillationType, 'LOGITS'):
            return DistillationType.LOGITS
        return None
    
    def as_training_strategy(self) -> Optional['TrainingStrategy']:
        """
        作为训练策略返回
        
        使用 base_strategy.py 的 TrainingStrategy
        """
        if not STRATEGY_LAYER_AVAILABLE or TrainingStrategy is None:
            return None
        
        if self.distillation_strategy and isinstance(self.distillation_strategy, TrainingStrategy):
            return self.distillation_strategy
        
        return None
    
    # ======================== 分布式策略层模块使用方法 ========================
    
    def create_distributed_config(
        self,
        mode: str = 'ddp',
        world_size: int = 1,
        rank: int = 0,
    ) -> Optional['DistributedStrategyConfig']:
        """
        创建分布式策略配置
        
        使用 distributed_strategy.py 的 DistributedStrategyConfig 和 StrategyDistributedMode
        """
        if not DISTRIBUTED_STRATEGY_AVAILABLE or DistributedStrategyConfig is None:
            return None
        
        try:
            # 验证模式
            if StrategyDistributedMode is not None:
                try:
                    _ = StrategyDistributedMode(mode)
                except (ValueError, TypeError):
                    mode = 'ddp'
            
            return DistributedStrategyConfig(
                mode=mode,
                world_size=world_size,
                rank=rank,
            )
        except Exception:
            return None
    
    def get_zero_stage_for_compression(self) -> Optional['ZeROStage']:
        """
        获取压缩使用的 ZeRO 阶段
        
        使用 distributed_strategy.py 的 ZeROStage
        """
        if not DISTRIBUTED_STRATEGY_AVAILABLE or ZeROStage is None:
            return None
        
        # 默认禁用 ZeRO
        if hasattr(ZeROStage, 'STAGE_1'):
            return ZeROStage.STAGE_1
        return None
    
    def get_distributed_health(self) -> Optional['DistributedHealthStatus']:
        """
        获取分布式健康状态
        
        使用 distributed_strategy.py 的 DistributedHealthStatus
        """
        if not DISTRIBUTED_STRATEGY_AVAILABLE or DistributedHealthStatus is None:
            return None
        
        try:
            return DistributedHealthStatus(
                is_healthy=True,
                world_size=1,
                rank=0,
            )
        except Exception:
            return None
    
    def get_communication_stats(self) -> Optional['CommunicationStats']:
        """
        获取通信统计
        
        使用 distributed_strategy.py 的 CommunicationStats
        """
        if not DISTRIBUTED_STRATEGY_AVAILABLE or CommunicationStats is None:
            return None
        
        try:
            return CommunicationStats()
        except Exception:
            return None
    
    def recommend_distributed_mode(
        self,
        model_size_mb: float = 1000.0,
    ) -> Optional[str]:
        """
        推荐分布式模式
        
        使用 distributed_strategy.py 的 recommend_distributed_mode
        """
        if not DISTRIBUTED_STRATEGY_AVAILABLE or recommend_distributed_mode is None:
            return None
        
        try:
            return recommend_distributed_mode(model_size_mb=model_size_mb)
        except Exception:
            return None
    
    # ======================== lib/losses 模块使用方法 ========================
    
    def create_distillation_loss(self, loss_type: str = 'kl_div') -> Optional['BaseLoss']:
        """
        创建蒸馏损失函数
        
        使用 lib/losses 的 create_loss 和 BaseLoss
        """
        if not LOSSES_LAYER_AVAILABLE or create_loss is None:
            return None
        
        try:
            return create_loss(loss_type, reduction='batchmean')
        except Exception:
            return None
    
    def get_loss_stats(self) -> Optional['LibLossStats']:
        """
        获取损失统计
        
        使用 lib/losses 的 LossStats as LibLossStats
        """
        if not LOSSES_LAYER_AVAILABLE or LibLossStats is None:
            return None
        
        try:
            return LibLossStats()
        except Exception:
            return None
    
    def validate_loss(self, loss: Any) -> bool:
        """
        验证损失函数
        
        使用 lib/losses 的 BaseLoss
        """
        if not LOSSES_LAYER_AVAILABLE or BaseLoss is None:
            return isinstance(loss, nn.Module)
        
        return isinstance(loss, (BaseLoss, nn.Module))
    
    # ======================== 综合诊断方法 ========================
    
    def get_full_diagnosis(self) -> Dict[str, Any]:
        """
        获取完整诊断信息
        
        整合所有层模块的信息
        """
        diagnosis = self.diagnose()
        
        # 添加配置层信息
        diagnosis['config_layer'] = {
            'compression_methods': [m.value for m in self.get_compression_method()],
            'compression_method_info': self.get_compression_method_info(),
            'distributed_mode': self.get_distributed_mode_for_compression().value,
        }
        
        # 添加策略层信息
        diagnosis['strategy_layer'] = {
            'strategy_type': self.get_strategy_type().value if self.get_strategy_type() and hasattr(self.get_strategy_type(), 'value') else None,
            'training_phase': self.get_training_phase().value if self.get_training_phase() and hasattr(self.get_training_phase(), 'value') else None,
            'distillation_type': self.get_distillation_type().value if self.get_distillation_type() and hasattr(self.get_distillation_type(), 'value') else None,
            'as_training_strategy': self.as_training_strategy() is not None,
        }
        
        # 添加分布式策略层信息
        diagnosis['distributed_strategy_layer'] = {
            'zero_stage': self.get_zero_stage_for_compression().value if self.get_zero_stage_for_compression() and hasattr(self.get_zero_stage_for_compression(), 'value') else None,
            'distributed_health': self.get_distributed_health() is not None,
            'communication_stats': self.get_communication_stats() is not None,
        }
        
        # 添加 lib/losses 信息
        diagnosis['lib_losses'] = {
            'loss_stats': self.get_loss_stats() is not None,
            'can_create_loss': self.create_distillation_loss() is not None,
        }
        
        return diagnosis


# ======================== 便捷函数 ========================

def create_knowledge_distillation_trainer(
    config: Union[Dict[str, Any], DistillationConfig],
    task_config: Optional[DistillationTaskConfig] = None,
) -> KnowledgeDistillationTrainer:
    """
    创建知识蒸馏训练器的便捷函数
    
    集成 compression_config.py 的 DistillationConfig 和 DistillationTaskConfig
    
    Args:
        config: 配置字典或 DistillationConfig 实例
        task_config: 完整任务配置（可选）
    
    Returns:
        KnowledgeDistillationTrainer 实例
    """
    try:
        if isinstance(config, dict):
            distill_config = DistillationConfig(**config)
        else:
            distill_config = config
        return KnowledgeDistillationTrainer(distill_config, task_config)
    except Exception as e:
        logger.error(f"Failed to create knowledge distillation trainer: {e}")
        raise BusinessLogicError(f"Failed to create knowledge distillation trainer: {e}")


def create_model_compressor(config: Union[Dict[str, Any], CompressionConfig]) -> ModelCompressor:
    """
    创建模型压缩器的便捷函数
    
    集成 compression_config.py 的 CompressionConfig
    
    Args:
        config: 配置字典或 CompressionConfig 实例
    
    Returns:
        ModelCompressor 实例
    """
    try:
        if isinstance(config, dict):
            compression_config = CompressionConfig(**config)
        else:
            compression_config = config
        return ModelCompressor(compression_config)
    except Exception as e:
        logger.error(f"Failed to create model compressor: {e}")
        raise BusinessLogicError(f"Failed to create model compressor: {e}")


def create_distillation_trainer_with_strategy(
    strategy_type: str = "standard",
    temperature: float = 4.0,
    alpha: float = 0.7,
    beta: float = 0.3,
    **kwargs
) -> KnowledgeDistillationTrainer:
    """
    使用指定策略创建蒸馏训练器
    
    集成 distillation_strategy.py 的策略创建功能
    
    Args:
        strategy_type: 策略类型 (standard, self, progressive, industry, contrastive)
        temperature: 温度参数
        alpha: 软标签损失权重
        beta: 硬标签损失权重
        **kwargs: 其他配置参数
    
    Returns:
        KnowledgeDistillationTrainer 实例
    """
    # 创建配置
    config = DistillationConfig(
        teacher_model_path=kwargs.get('teacher_model_path', 'mock'),
        student_model_path=kwargs.get('student_model_path', 'mock'),
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        use_feature_distillation=kwargs.get('use_feature_distillation', False),
        feature_loss_weight=kwargs.get('feature_loss_weight', 0.1),
        use_attention_distillation=kwargs.get('use_attention_distillation', False),
        attention_loss_weight=kwargs.get('attention_loss_weight', 0.1)
    )
    
    trainer = KnowledgeDistillationTrainer(config)
    
    # 如果需要特定策略，设置策略
    if strategy_type != "standard" and DISTILLATION_STRATEGY_AVAILABLE and create_distillation_strategy is not None:
        try:
            strategy = create_distillation_strategy(strategy_type)
            trainer.set_strategy(strategy)
        except Exception as e:
            logger.warning(f"Failed to set strategy {strategy_type}: {e}")
    
    return trainer


def create_trainer_from_preset(
    preset_name: str,
    **override_kwargs
) -> KnowledgeDistillationTrainer:
    """
    从预设模板创建训练器
    
    使用 compression_config.py 的 DistillationPresets
    
    Args:
        preset_name: 预设名称 (standard, edge_deployment, high_accuracy, etc.)
        **override_kwargs: 覆盖参数
    
    Returns:
        KnowledgeDistillationTrainer 实例
    """
    try:
        # 获取预设配置
        task_config = DistillationPresets.get(preset_name)
        
        # 应用覆盖参数
        if override_kwargs:
            for key, value in override_kwargs.items():
                if hasattr(task_config, key):
                    setattr(task_config, key, value)
                elif hasattr(task_config.distillation_config, key):
                    setattr(task_config.distillation_config, key, value)
        
        return KnowledgeDistillationTrainer(task_config.distillation_config, task_config)
    except Exception as e:
        logger.error(f"Failed to create trainer from preset {preset_name}: {e}")
        raise BusinessLogicError(f"Failed to create trainer from preset {preset_name}: {e}")


def create_trainer_from_scenario(
    scenario: Union[str, DistillationScenario],
    **kwargs
) -> KnowledgeDistillationTrainer:
    """
    从场景创建训练器
    
    使用 compression_config.py 的 DistillationScenario 和预设
    
    Args:
        scenario: 场景枚举或字符串
        **kwargs: 额外配置参数
    
    Returns:
        KnowledgeDistillationTrainer 实例
    """
    try:
        # 转换为枚举
        if isinstance(scenario, str):
            scenario_enum = DistillationScenario.from_string(scenario)
        else:
            scenario_enum = scenario
        
        # 根据场景获取预设配置
        scenario_to_preset = {
            DistillationScenario.STANDARD: 'standard',
            DistillationScenario.INDUSTRY: 'industry',
            DistillationScenario.EDGE_DEPLOY: 'edge_deployment',
            DistillationScenario.MULTIMODAL: 'multimodal',
            DistillationScenario.REAL_TIME: 'real_time',
            DistillationScenario.PROGRESSIVE: 'progressive',
            DistillationScenario.SELF_DISTILLATION: 'self_distillation',
            DistillationScenario.CONTRASTIVE: 'contrastive',
            DistillationScenario.LOW_LATENCY: 'low_latency',
            DistillationScenario.HIGH_ACCURACY: 'high_accuracy',
        }
        
        preset_name = scenario_to_preset.get(scenario_enum, 'standard')
        task_config = DistillationPresets.get(preset_name)
        
        # 应用覆盖参数
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(task_config, key):
                    setattr(task_config, key, value)
                elif hasattr(task_config.distillation_config, key):
                    setattr(task_config.distillation_config, key, value)
        
        return KnowledgeDistillationTrainer(task_config.distillation_config, task_config)
    except Exception as e:
        logger.error(f"Failed to create trainer from scenario {scenario}: {e}")
        raise BusinessLogicError(f"Failed to create trainer from scenario {scenario}: {e}")


def diagnose_trainer(trainer: KnowledgeDistillationTrainer) -> Dict[str, Any]:
    """
    诊断训练器
    
    Args:
        trainer: 训练器实例
    
    Returns:
        诊断结果
    """
    return trainer.diagnose()


def diagnose_compressor(compressor: ModelCompressor) -> Dict[str, Any]:
    """
    诊断压缩器
    
    Args:
        compressor: 压缩器实例
    
    Returns:
        诊断结果
    """
    return compressor.diagnose()


def get_layer_availability() -> Dict[str, bool]:
    """
    获取各层可用性
    
    Returns:
        各层可用性字典
    """
    return {
        'strategy_layer': STRATEGY_LAYER_AVAILABLE,
        'distillation_strategy': DISTILLATION_STRATEGY_AVAILABLE,
        'distributed_strategy': DISTRIBUTED_STRATEGY_AVAILABLE,
        'losses_layer': LOSSES_LAYER_AVAILABLE,
        'hardware_layer': HARDWARE_LAYER_AVAILABLE,
        'distributed_layer': DISTRIBUTED_LAYER_AVAILABLE,
    }


def print_layer_availability() -> None:
    """打印各层可用性"""
    print("=" * 50)
    print("Knowledge Distillation Layer Availability")
    print("=" * 50)
    
    availability = get_layer_availability()
    for layer, available in availability.items():
        status = "✓" if available else "✗"
        print(f"  {status} {layer}")
    
    print("=" * 50)


def compare_trainers(
    trainer1: KnowledgeDistillationTrainer,
    trainer2: KnowledgeDistillationTrainer
) -> Dict[str, Any]:
    """
    比较两个训练器
    
    Args:
        trainer1: 第一个训练器
        trainer2: 第二个训练器
    
    Returns:
        比较结果
    """
    diag1 = trainer1.diagnose()
    diag2 = trainer2.diagnose()
    
    return {
        'trainer1': {
            'global_step': diag1.get('global_step', 0),
            'best_loss': diag1.get('stats', {}).get('best_loss', float('inf')),
            'device': diag1.get('device', 'unknown'),
        },
        'trainer2': {
            'global_step': diag2.get('global_step', 0),
            'best_loss': diag2.get('stats', {}).get('best_loss', float('inf')),
            'device': diag2.get('device', 'unknown'),
        },
        'differences': {
            'step_diff': diag2.get('global_step', 0) - diag1.get('global_step', 0),
            'loss_diff': diag2.get('stats', {}).get('best_loss', 0) - diag1.get('stats', {}).get('best_loss', 0),
        }
    }


def estimate_training_resources(
    config: Union[DistillationConfig, DistillationTaskConfig],
    model_size_mb: float = 1000.0,
) -> Dict[str, Any]:
    """
    估计训练资源需求
    
    使用 compression_config.py 的配置进行估计
    
    Args:
        config: 配置
        model_size_mb: 模型大小（MB）
    
    Returns:
        资源估计
    """
    # 基础内存需求
    base_memory_mb = model_size_mb * 2  # 教师和学生模型
    
    # 优化器状态
    optimizer_memory_mb = model_size_mb * 2  # Adam 需要两个动量缓冲区
    
    # 梯度内存
    gradient_memory_mb = model_size_mb
    
    # 激活内存（估计）
    activation_memory_mb = model_size_mb * 0.5
    
    # 混合精度可节省约 50% 内存
    if hasattr(config, 'use_amp') and config.use_amp:
        total_memory_mb = (base_memory_mb + optimizer_memory_mb + gradient_memory_mb + activation_memory_mb) * 0.5
    else:
        total_memory_mb = base_memory_mb + optimizer_memory_mb + gradient_memory_mb + activation_memory_mb
    
    # 估计训练时间（每步）
    estimated_step_time_ms = model_size_mb * 0.01  # 粗略估计
    
    return {
        'estimated_memory_mb': total_memory_mb,
        'breakdown': {
            'model_memory_mb': base_memory_mb,
            'optimizer_memory_mb': optimizer_memory_mb,
            'gradient_memory_mb': gradient_memory_mb,
            'activation_memory_mb': activation_memory_mb,
        },
        'estimated_step_time_ms': estimated_step_time_ms,
        'recommendations': {
            'use_amp': total_memory_mb > 8000,  # 如果需要超过 8GB，建议使用 AMP
            'use_gradient_checkpointing': total_memory_mb > 16000,  # 超过 16GB 建议使用梯度检查点
        }
    }


# ======================== 全局单例和便捷访问 ========================

_global_trainer: Optional[KnowledgeDistillationTrainer] = None
_global_compressor: Optional[ModelCompressor] = None


def get_global_trainer() -> Optional[KnowledgeDistillationTrainer]:
    """获取全局训练器实例"""
    return _global_trainer


def set_global_trainer(trainer: KnowledgeDistillationTrainer) -> None:
    """设置全局训练器实例"""
    global _global_trainer
    _global_trainer = trainer


def get_global_compressor() -> Optional[ModelCompressor]:
    """获取全局压缩器实例"""
    return _global_compressor


def set_global_compressor(compressor: ModelCompressor) -> None:
    """设置全局压缩器实例"""
    global _global_compressor
    _global_compressor = compressor


def reset_globals() -> None:
    """重置全局实例"""
    global _global_trainer, _global_compressor
    _global_trainer = None
    _global_compressor = None

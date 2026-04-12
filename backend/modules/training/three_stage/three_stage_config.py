"""三阶段训练配置模型

生产级三阶段训练配置，支持：
- 预训练（Pretrain）、微调（Finetune）、偏好优化（Preference）
- 策略层集成（base_strategy, distributed_strategy, three_stage_strategy）
- 硬件层集成（device_manager, memory_manager, mixed_precision）
- 配置验证、序列化、预设工厂

架构调用层次：
├── three_stage_config.py (本模块)
│   └── 调用 backend/modules/training/strategies (策略层)
│   └── 调用 backend/lib/hardware (硬件层)
│   └── 调用 backend/lib/distributed (分布式层)
└── 被 three_stage_trainer.py 调用
"""

# 修复导入路径
import sys
import os as os_path
import json
import hashlib
import logging
from copy import deepcopy
from pathlib import Path

current_dir = os_path.path.dirname(os_path.path.abspath(__file__))
project_root = os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(current_dir))))
sys.path.insert(0, project_root)

from dataclasses import dataclass, field, fields, MISSING
from typing import Dict, Any, Optional, List, Union
from enum import Enum

logger = logging.getLogger(__name__)


# ==================== 异常导入 ====================

try:
    from backend.modules.training.exceptions import ValidationError
except ImportError:
    class ValidationError(Exception):
        """验证异常（备用定义）"""
        def __init__(self, message: str, field: str = None):
            super().__init__(message)
            self.field = field


# ==================== 策略层导入 ====================

STRATEGY_LAYER_AVAILABLE = False
try:
    from backend.modules.training.strategies.base_strategy import (
        StrategyType,
        TrainingPhase,
        StrategyMonitor,
        StrategyProfiler,
        StrategyValidator,
        StrategyMetrics,
        StrategyContext,
        StrategyResult,
    )
    STRATEGY_LAYER_AVAILABLE = True
    logger.info("Strategy layer (base) loaded for three_stage_config")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Strategy layer (base) not available: {e}")
    StrategyType = None
    TrainingPhase = None
    StrategyMonitor = None
    StrategyProfiler = None
    StrategyValidator = None
    StrategyMetrics = None
    StrategyContext = None
    StrategyResult = None

DISTRIBUTED_STRATEGY_AVAILABLE = False
try:
    from backend.modules.training.strategies.distributed_strategy import (
        DistributedMode,
        ZeROStage,
        DistributedStrategyConfig,
        recommend_distributed_mode,
        diagnose_distributed_strategy,
    )
    DISTRIBUTED_STRATEGY_AVAILABLE = True
    logger.info("Strategy layer (distributed) loaded for three_stage_config")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Strategy layer (distributed) not available: {e}")
    DistributedMode = None
    ZeROStage = None
    DistributedStrategyConfig = None
    recommend_distributed_mode = None
    diagnose_distributed_strategy = None

THREE_STAGE_STRATEGY_AVAILABLE = False
try:
    from backend.modules.training.strategies.three_stage_strategy import (
        ThreeStageStrategy,
        ThreeStageStrategyConfig,
        ThreeStagePhase,
        create_three_stage_strategy,
        diagnose_three_stage_strategy,
    )
    THREE_STAGE_STRATEGY_AVAILABLE = True
    logger.info("Strategy layer (three_stage) loaded for three_stage_config")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Strategy layer (three_stage) not available: {e}")
    ThreeStageStrategy = None
    ThreeStageStrategyConfig = None
    ThreeStagePhase = None
    create_three_stage_strategy = None
    diagnose_three_stage_strategy = None


# ==================== 硬件层导入 ====================

HARDWARE_LAYER_AVAILABLE = False
try:
    from backend.lib.hardware import (
        DeviceManager,
        get_device_manager,
        MemoryManager,
        MixedPrecisionManager,
        AmpConfig,
        PrecisionMode,
        get_available_memory,
        clear_memory,
        estimate_model_memory,
        recommend_precision,
        recommend_batch_size,
        DeviceType,
        DeviceInfo,
    )
    HARDWARE_LAYER_AVAILABLE = True
    logger.info("Hardware layer loaded for three_stage_config")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Hardware layer not available: {e}")
    DeviceManager = None
    get_device_manager = None
    MemoryManager = None
    MixedPrecisionManager = None
    AmpConfig = None
    PrecisionMode = None
    get_available_memory = None
    clear_memory = None
    estimate_model_memory = None
    recommend_precision = None
    recommend_batch_size = None
    DeviceType = None
    DeviceInfo = None


# ==================== 分布式层导入 ====================

from backend.lib.distributed import (
    DistributedManager,
    get_distributed_manager,
    DDPWrapper,
    FSDPWrapper,
)


# ==================== 进度管理导入 ====================

from backend.modules.training.progress.progress_manager import (
    TrainingProgressManager,
    TrainingProgress,
    get_progress_manager,
)


# ==================== 枚举定义 ====================

class TrainingStage(str, Enum):
    """训练阶段枚举"""
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    PREFERENCE = "preference"

    @classmethod
    def from_string(cls, value: str) -> 'TrainingStage':
        """从字符串创建"""
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown training stage: {value}")
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        names = {
            TrainingStage.PRETRAIN: "预训练",
            TrainingStage.FINETUNE: "监督微调",
            TrainingStage.PREFERENCE: "偏好优化",
        }
        return names.get(self, self.value)
    
    @property
    def default_epochs(self) -> int:
        """获取默认训练轮数"""
        defaults = {
            TrainingStage.PRETRAIN: 1,
            TrainingStage.FINETUNE: 3,
            TrainingStage.PREFERENCE: 2,
        }
        return defaults.get(self, 3)
    
    @property
    def default_learning_rate(self) -> float:
        """获取默认学习率"""
        defaults = {
            TrainingStage.PRETRAIN: 1e-4,
            TrainingStage.FINETUNE: 2e-5,
            TrainingStage.PREFERENCE: 1e-5,
        }
        return defaults.get(self, 5e-5)
    
    def to_strategy_phase(self) -> Optional['ThreeStagePhase']:
        """转换为策略层阶段枚举"""
        if THREE_STAGE_STRATEGY_AVAILABLE and ThreeStagePhase is not None:
            try:
                return ThreeStagePhase(self.value)
            except Exception:
                pass
        return None
    
    def to_training_phase(self) -> Optional['TrainingPhase']:
        """转换为策略层训练阶段枚举"""
        if STRATEGY_LAYER_AVAILABLE and TrainingPhase is not None:
            try:
                phase_mapping = {
                    TrainingStage.PRETRAIN: TrainingPhase.WARMUP,
                    TrainingStage.FINETUNE: TrainingPhase.MAIN,
                    TrainingStage.PREFERENCE: TrainingPhase.COOLDOWN,
                }
                return phase_mapping.get(self, TrainingPhase.MAIN)
            except Exception:
                pass
        return None


class OptimizerType(str, Enum):
    """优化器类型"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    LAMB = "lamb"
    ADAFACTOR = "adafactor"


class SchedulerType(str, Enum):
    """学习率调度器类型"""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    INVERSE_SQRT = "inverse_sqrt"


# ==================== 配置验证器 ====================

class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_stage_config(config: 'StageConfig') -> List[str]:
        """验证阶段配置"""
        errors = []
        
        if config.epochs <= 0:
            errors.append(f"epochs must be positive, got {config.epochs}")
        
        if config.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {config.learning_rate}")
        
        if config.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {config.batch_size}")
        
        if config.num_workers is not None and config.num_workers < 0:
            errors.append(f"num_workers must be non-negative, got {config.num_workers}")
        
        if config.validation_split < 0 or config.validation_split > 1:
            errors.append(f"validation_split must be in [0, 1], got {config.validation_split}")
        
        if config.gradient_clipping < 0:
            errors.append(f"gradient_clipping must be non-negative, got {config.gradient_clipping}")
        
        if config.gradient_accumulation_steps < 1:
            errors.append(f"gradient_accumulation_steps must be at least 1, got {config.gradient_accumulation_steps}")
        
        return errors
    
    @staticmethod
    def validate_three_stage_config(config: 'ThreeStageConfig') -> List[str]:
        """验证三阶段配置"""
        errors = []
        
        if not config.base_model_path:
            errors.append("base_model_path cannot be empty")
        
        if not config.output_dir:
            errors.append("output_dir cannot be empty")
        
        if config.default_num_workers is not None and config.default_num_workers < 0:
            errors.append(f"default_num_workers must be non-negative, got {config.default_num_workers}")
        
        # 验证各阶段配置
        stage_errors = ConfigValidator.validate_stage_config(config.pretrain)
        errors.extend([f"pretrain.{e}" for e in stage_errors])
        
        stage_errors = ConfigValidator.validate_stage_config(config.finetune)
        errors.extend([f"finetune.{e}" for e in stage_errors])
        
        stage_errors = ConfigValidator.validate_stage_config(config.preference)
        errors.extend([f"preference.{e}" for e in stage_errors])
        
        # 使用策略层验证器（如果可用）
        # 注意：StrategyValidator 用于验证 StrategyResult，不用于验证配置
        # 配置验证应该使用 ConfigValidator 的静态方法
        # 如果需要策略层验证，应该在训练过程中使用 StrategyValidator.validate(result)
        
        return errors
    
    @staticmethod
    def validate_distributed_config(config: Dict[str, Any]) -> List[str]:
        """验证分布式配置"""
        errors = []
        
        world_size = config.get('world_size', 1)
        if world_size < 1:
            errors.append(f"world_size must be at least 1, got {world_size}")
        
        # 使用分布式策略诊断（如果可用）
        if DISTRIBUTED_STRATEGY_AVAILABLE and diagnose_distributed_strategy is not None:
            try:
                diagnosis = diagnose_distributed_strategy()
                if diagnosis.get('status') == 'error':
                    errors.append(diagnosis.get('message', 'Distributed configuration error'))
            except Exception as e:
                logger.warning(f"Distributed strategy diagnosis failed: {e}")
        
        return errors


# ==================== 配置序列化器 ====================

class ConfigSerializer:
    """配置序列化器"""
    
    @staticmethod
    def to_json(config: Union['StageConfig', 'ThreeStageConfig'], indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(config.to_dict(), indent=indent, ensure_ascii=False)
    
    @staticmethod
    def from_json(json_str: str, config_class: type) -> Union['StageConfig', 'ThreeStageConfig']:
        """从 JSON 字符串创建配置"""
        data = json.loads(json_str)
        return config_class.from_dict(data)
    
    @staticmethod
    def to_file(config: Union['StageConfig', 'ThreeStageConfig'], file_path: str) -> None:
        """保存配置到文件"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Config saved to {file_path}")
    
    @staticmethod
    def from_file(file_path: str, config_class: type) -> Union['StageConfig', 'ThreeStageConfig']:
        """从文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return config_class.from_dict(data)
    
    @staticmethod
    def get_config_hash(config: Union['StageConfig', 'ThreeStageConfig']) -> str:
        """获取配置的哈希值"""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]


# ==================== 阶段配置 ====================

@dataclass
class StageConfig:
    """阶段配置"""
    # 基础配置
    enabled: bool = True
    epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_workers: Optional[int] = None
    warmup_steps: int = 500
    warmup_ratio: float = 0.0  # 如果设置，会覆盖 warmup_steps
    
    # 数据配置
    dataset_path: Optional[str] = None
    validation_split: float = 0.1
    max_length: int = 512
    
    # 模型配置
    model_path: Optional[str] = None
    
    # 优化配置
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    gradient_accumulation_steps: int = 1
    stats_window_size: Optional[int] = None
    
    # 保存配置
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # 早停配置
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    
    def __post_init__(self):
        """初始化后验证"""
        errors = ConfigValidator.validate_stage_config(self)
        if errors:
            raise ValidationError(f"Invalid stage config: {'; '.join(errors)}", field="stage_config")
    
    def validate(self) -> None:
        """验证配置参数"""
        errors = ConfigValidator.validate_stage_config(self)
        if errors:
            raise ValidationError(f"Invalid stage config: {'; '.join(errors)}", field="stage_config")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'enabled': self.enabled,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'warmup_steps': self.warmup_steps,
            'warmup_ratio': self.warmup_ratio,
            'dataset_path': self.dataset_path,
            'validation_split': self.validation_split,
            'max_length': self.max_length,
            'model_path': self.model_path,
            'optimizer_type': self.optimizer_type,
            'scheduler_type': self.scheduler_type,
            'weight_decay': self.weight_decay,
            'gradient_clipping': self.gradient_clipping,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'stats_window_size': self.stats_window_size,
            'save_steps': self.save_steps,
            'eval_steps': self.eval_steps,
            'logging_steps': self.logging_steps,
            'early_stopping': self.early_stopping,
            'early_stopping_patience': self.early_stopping_patience,
            'early_stopping_threshold': self.early_stopping_threshold,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StageConfig':
        """从字典创建"""
        # 创建实例时绕过 __post_init__ 验证
        config = object.__new__(cls)
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        # 设置默认值
        # 使用 fields() 函数获取 dataclass 字段
        field_dict = {f.name: f for f in fields(cls)}
        for key in ['enabled', 'epochs', 'learning_rate', 'batch_size', 'warmup_steps', 
                    'warmup_ratio', 'validation_split', 'max_length', 'optimizer_type',
                    'scheduler_type', 'weight_decay', 'gradient_clipping', 
                    'gradient_accumulation_steps', 'save_steps', 'eval_steps',
                    'logging_steps', 'early_stopping', 'early_stopping_patience',
                    'early_stopping_threshold']:
            if not hasattr(config, key) or getattr(config, key) is None:
                if key in field_dict:
                    default = field_dict[key].default
                    if default is not None and default != MISSING:
                        setattr(config, key, default)
        return config
    
    def get_optimizer_type_enum(self) -> OptimizerType:
        """获取优化器类型枚举"""
        return OptimizerType(self.optimizer_type)
    
    def get_scheduler_type_enum(self) -> SchedulerType:
        """获取调度器类型枚举"""
        return SchedulerType(self.scheduler_type)
    
    def estimate_memory_mb(self, model_size_params: int = 0) -> float:
        """估算内存占用（MB）"""
        # 基础估算
        batch_memory = self.batch_size * self.max_length * 4 / (1024 * 1024)  # float32
        
        # 使用硬件层估算（如果可用）
        if HARDWARE_LAYER_AVAILABLE and estimate_model_memory is not None and model_size_params > 0:
            try:
                estimated = estimate_model_memory(model_size_params)
                if estimated > 0:
                    return estimated + batch_memory
            except Exception as e:
                logger.warning(f"Hardware memory estimation failed: {e}")
        
        return batch_memory
    
    def clone(self) -> 'StageConfig':
        """克隆配置"""
        return StageConfig.from_dict(self.to_dict())


# ==================== 三阶段配置 ====================

@dataclass
class ThreeStageConfig:
    """三阶段训练配置"""
    # 阶段配置
    pretrain: StageConfig = field(default_factory=StageConfig)
    finetune: StageConfig = field(default_factory=StageConfig)
    preference: StageConfig = field(default_factory=StageConfig)
    
    # 通用配置
    base_model_path: str = "gpt2"
    output_dir: str = "./outputs/three_stage"
    seed: int = 42
    use_fp16: bool = True
    fp16_opt_level: str = "O1"
    default_num_workers: Optional[int] = 4
    
    # 阶段间配置
    pass_model_between_stages: bool = True
    save_intermediate_models: bool = True
    
    # 分布式配置
    use_distributed: bool = False
    distributed_mode: str = "ddp"
    world_size: int = 1
    local_rank: int = -1
    
    # 硬件配置
    device: str = "cuda"
    pin_memory: bool = True
    
    # 策略配置
    use_strategy: bool = True
    strategy_type: str = "three_stage"
    
    # 内部状态
    _strategy_monitor: Optional['StrategyMonitor'] = field(default=None, repr=False)
    _strategy_profiler: Optional['StrategyProfiler'] = field(default=None, repr=False)
    _strategy_validator: Optional['StrategyValidator'] = field(default=None, repr=False)
    
    def __post_init__(self):
        """初始化后验证"""
        self._set_stage_defaults()
        self.validate()
        self._init_strategy_integration()
        self._init_hardware_optimization()
    
    def _set_stage_defaults(self):
        """设置各阶段的默认值"""
        # 预训练默认值
        if self.pretrain.epochs == 3:
            self.pretrain.epochs = TrainingStage.PRETRAIN.default_epochs
            self.pretrain.learning_rate = TrainingStage.PRETRAIN.default_learning_rate
        
        # 微调默认值
        if self.finetune.epochs == 3:
            self.finetune.epochs = TrainingStage.FINETUNE.default_epochs
            self.finetune.learning_rate = TrainingStage.FINETUNE.default_learning_rate
        
        # 偏好优化默认值
        if self.preference.epochs == 3:
            self.preference.epochs = TrainingStage.PREFERENCE.default_epochs
            self.preference.learning_rate = TrainingStage.PREFERENCE.default_learning_rate
    
    def _init_strategy_integration(self):
        """初始化策略层集成"""
        # 创建策略监控器（如果可用）
        if STRATEGY_LAYER_AVAILABLE and StrategyMonitor is not None:
            try:
                self._strategy_monitor = StrategyMonitor()
                logger.debug("Strategy monitor initialized for three_stage config")
            except Exception as e:
                logger.warning(f"Failed to create strategy monitor: {e}")
                self._strategy_monitor = None
        else:
            self._strategy_monitor = None
        
        # 创建策略分析器（如果可用）
        if STRATEGY_LAYER_AVAILABLE and StrategyProfiler is not None:
            try:
                self._strategy_profiler = StrategyProfiler()
                logger.debug("Strategy profiler initialized for three_stage config")
            except Exception as e:
                logger.warning(f"Failed to create strategy profiler: {e}")
                self._strategy_profiler = None
        else:
            self._strategy_profiler = None
        
        # 创建策略验证器（如果可用）
        if STRATEGY_LAYER_AVAILABLE and StrategyValidator is not None:
            try:
                self._strategy_validator = StrategyValidator()
                logger.debug("Strategy validator initialized for three_stage config")
            except Exception as e:
                logger.warning(f"Failed to create strategy validator: {e}")
                self._strategy_validator = None
        else:
            self._strategy_validator = None
    
    def _init_hardware_optimization(self):
        """初始化硬件优化"""
        if HARDWARE_LAYER_AVAILABLE:
            try:
                # 获取设备管理器
                device_manager = get_device_manager()
                if device_manager is not None:
                    # 检测最佳设备
                    best_device = device_manager.get_best_device()
                    if best_device:
                        self.device = str(best_device)
                        logger.debug(f"Using best device: {self.device}")
                
                # 推荐精度模式
                if recommend_precision is not None:
                    try:
                        recommended_precision = recommend_precision(self.device)
                        if recommended_precision:
                            self.use_fp16 = recommended_precision in ['fp16', 'mixed']
                            logger.debug(f"Recommended precision: {recommended_precision}, use_fp16={self.use_fp16}")
                    except Exception as e:
                        logger.warning(f"Failed to recommend precision: {e}")
                
                # 推荐 batch size
                if recommend_batch_size is not None and get_available_memory is not None:
                    try:
                        available_mem = get_available_memory()
                        if available_mem > 0:
                            for stage_config in [self.pretrain, self.finetune, self.preference]:
                                estimated_mem = stage_config.estimate_memory_mb()
                                if estimated_mem > 0:
                                    recommended = recommend_batch_size(
                                        model_memory_mb=estimated_mem * stage_config.batch_size,
                                        available_memory_mb=available_mem
                                    )
                                    if recommended and recommended > 0 and recommended < stage_config.batch_size:
                                        logger.info(f"Adjusting batch_size from {stage_config.batch_size} to {recommended}")
                                        stage_config.batch_size = recommended
                    except Exception as e:
                        logger.warning(f"Failed to recommend batch size: {e}")
                        
            except Exception as e:
                logger.warning(f"Hardware optimization initialization failed: {e}")
        
    def validate(self) -> None:
        """验证配置参数"""
        errors = ConfigValidator.validate_three_stage_config(self)
        if errors:
            raise ValidationError(f"Invalid config: {'; '.join(errors)}", field="three_stage_config")
    
    def get_enabled_stages(self) -> List[TrainingStage]:
        """获取启用的阶段"""
        stages = []
        if self.pretrain.enabled:
            stages.append(TrainingStage.PRETRAIN)
        if self.finetune.enabled:
            stages.append(TrainingStage.FINETUNE)
        if self.preference.enabled:
            stages.append(TrainingStage.PREFERENCE)
        return stages
    
    def get_stage_config(self, stage: TrainingStage) -> Optional[StageConfig]:
        """获取阶段配置"""
        if stage == TrainingStage.PRETRAIN:
            return self.pretrain
        elif stage == TrainingStage.FINETUNE:
            return self.finetune
        elif stage == TrainingStage.PREFERENCE:
            return self.preference
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'pretrain': self.pretrain.to_dict(),
            'finetune': self.finetune.to_dict(),
            'preference': self.preference.to_dict(),
            'base_model_path': self.base_model_path,
            'output_dir': self.output_dir,
            'seed': self.seed,
            'use_fp16': self.use_fp16,
            'fp16_opt_level': self.fp16_opt_level,
            'default_num_workers': self.default_num_workers,
            'pass_model_between_stages': self.pass_model_between_stages,
            'save_intermediate_models': self.save_intermediate_models,
            'use_distributed': self.use_distributed,
            'distributed_mode': self.distributed_mode,
            'world_size': self.world_size,
            'local_rank': self.local_rank,
            'device': self.device,
            'pin_memory': self.pin_memory,
            'use_strategy': self.use_strategy,
            'strategy_type': self.strategy_type,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThreeStageConfig':
        """从字典创建配置"""
        # 处理阶段配置
        pretrain = StageConfig.from_dict(data.get('pretrain', {})) if 'pretrain' in data else StageConfig()
        finetune = StageConfig.from_dict(data.get('finetune', {})) if 'finetune' in data else StageConfig()
        preference = StageConfig.from_dict(data.get('preference', {})) if 'preference' in data else StageConfig()
        
        # 创建配置实例
        config = object.__new__(cls)
        config.pretrain = pretrain
        config.finetune = finetune
        config.preference = preference
        
        # 设置其他字段
        # 使用 fields() 函数获取 dataclass 字段
        field_dict = {f.name: f for f in fields(cls)}
        for key in ['base_model_path', 'output_dir', 'seed', 'use_fp16', 'fp16_opt_level',
                    'default_num_workers', 'pass_model_between_stages', 'save_intermediate_models',
                    'use_distributed', 'distributed_mode', 'world_size', 'local_rank',
                    'device', 'pin_memory', 'use_strategy', 'strategy_type']:
            if key in data:
                setattr(config, key, data[key])
            else:
                if key in field_dict:
                    default = field_dict[key].default
                    if default is not None and default != MISSING:
                        setattr(config, key, default)
        
        # 初始化内部状态
        config._strategy_monitor = None
        config._strategy_profiler = None
        config._strategy_validator = None
        
        # 手动调用初始化方法
        config._init_strategy_integration()
        config._init_hardware_optimization()
        
        return config
    
    def get_strategy_type_enum(self) -> Optional['StrategyType']:
        """获取策略类型枚举"""
        if STRATEGY_LAYER_AVAILABLE and StrategyType is not None:
            try:
                return StrategyType.from_string(self.strategy_type)
            except Exception:
                return StrategyType.THREE_STAGE if hasattr(StrategyType, 'THREE_STAGE') else None
        return None
    
    def get_distributed_mode_enum(self) -> Optional['DistributedMode']:
        """获取分布式模式枚举"""
        if DISTRIBUTED_STRATEGY_AVAILABLE and DistributedMode is not None:
            try:
                return DistributedMode.from_string(self.distributed_mode)
            except Exception:
                return DistributedMode.DDP
        return None
    
    def get_zero_stage(self) -> Optional['ZeROStage']:
        """获取 ZeRO 优化阶段"""
        if DISTRIBUTED_STRATEGY_AVAILABLE and ZeROStage is not None and self.use_distributed:
            try:
                if self.world_size >= 8:
                    return ZeROStage.STAGE_3
                elif self.world_size >= 4:
                    return ZeROStage.STAGE_2
                else:
                    return ZeROStage.STAGE_1
            except Exception as e:
                logger.warning(f"Failed to get ZeRO stage: {e}")
        return None
    
    def create_strategy_context(self, model=None) -> Optional['StrategyContext']:
        """创建策略上下文"""
        if STRATEGY_LAYER_AVAILABLE and StrategyContext is not None:
            try:
                return StrategyContext(
                    model=model,
                    device=self.device,
                    config=self.to_dict(),
                    metadata={
                        'enabled_stages': [s.value for s in self.get_enabled_stages()],
                        'use_fp16': self.use_fp16,
                        'use_distributed': self.use_distributed,
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to create strategy context: {e}")
        return None
    
    def create_strategy_metrics(self) -> Optional['StrategyMetrics']:
        """创建策略指标跟踪器"""
        if STRATEGY_LAYER_AVAILABLE and StrategyMetrics is not None:
            try:
                return StrategyMetrics()
            except Exception as e:
                logger.warning(f"Failed to create strategy metrics: {e}")
        return None
    
    def create_distributed_config(self) -> Optional['DistributedStrategyConfig']:
        """创建分布式策略配置"""
        if DISTRIBUTED_STRATEGY_AVAILABLE and DistributedStrategyConfig is not None and self.use_distributed:
            try:
                mode = self.get_distributed_mode_enum()
                return DistributedStrategyConfig(
                    distributed_mode=mode,
                    world_size=self.world_size,
                    gradient_accumulation_steps=self.pretrain.gradient_accumulation_steps,
                    fp16=self.use_fp16,
                )
            except Exception as e:
                logger.warning(f"Failed to create distributed config: {e}")
        return None
    
    def create_three_stage_strategy(self) -> Optional['ThreeStageStrategy']:
        """创建三阶段策略（使用策略层）"""
        if THREE_STAGE_STRATEGY_AVAILABLE and create_three_stage_strategy is not None:
            try:
                strategy_config = ThreeStageStrategyConfig(
                    device=self.device,
                    precision='fp16' if self.use_fp16 else 'fp32',
                    enable_amp=self.use_fp16,
                    gradient_accumulation_steps=self.pretrain.gradient_accumulation_steps,
                    gradient_clipping=self.pretrain.gradient_clipping,
                    weight_decay=self.pretrain.weight_decay,
                    pretrain_learning_rate=self.pretrain.learning_rate,
                    pretrain_epochs=self.pretrain.epochs,
                    pretrain_warmup_steps=self.pretrain.warmup_steps,
                    finetune_learning_rate=self.finetune.learning_rate,
                    finetune_epochs=self.finetune.epochs,
                    finetune_warmup_steps=self.finetune.warmup_steps,
                    preference_learning_rate=self.preference.learning_rate,
                    preference_epochs=self.preference.epochs,
                    preference_warmup_steps=self.preference.warmup_steps,
                    enabled_stages=[stage.value for stage in self.get_enabled_stages()],
                    pass_model_between_stages=self.pass_model_between_stages,
                ) if ThreeStageStrategyConfig else None
                
                return create_three_stage_strategy(strategy_config)
            except Exception as e:
                logger.warning(f"Failed to create three stage strategy: {e}")
        return None
    
    def recommend_distributed_settings(self) -> Dict[str, Any]:
        """获取推荐的分布式设置"""
        recommendations = {
            'use_distributed': False,
            'distributed_mode': 'ddp',
            'world_size': 1,
        }
        
        if DISTRIBUTED_STRATEGY_AVAILABLE and recommend_distributed_mode is not None:
            try:
                estimated_memory = self.estimate_total_memory_mb()
                requirements = {
                    'model_size_gb': estimated_memory / 1024,
                    'world_size': self.world_size,
                }
                
                rec = recommend_distributed_mode(requirements)
                if rec:
                    recommendations.update(rec)
                    recommendations['use_distributed'] = True
                    
            except Exception as e:
                logger.warning(f"Failed to get distributed recommendations: {e}")
        
        return recommendations
    
    def estimate_total_memory_mb(self) -> float:
        """估算总内存占用（MB）"""
        total_memory = 0.0
        
        for stage_config in [self.pretrain, self.finetune, self.preference]:
            if stage_config.enabled:
                total_memory = max(total_memory, stage_config.estimate_memory_mb())
        
        # 考虑 fp16 的节省
        if self.use_fp16:
            total_memory *= 0.5
        
        return total_memory
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断配置"""
        diagnosis = {
            'config_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
        }
        
        # 验证配置
        errors = ConfigValidator.validate_three_stage_config(self)
        if errors:
            diagnosis['config_valid'] = False
            diagnosis['errors'] = errors
        
        # 检查内存
        estimated_memory = self.estimate_total_memory_mb()
        if HARDWARE_LAYER_AVAILABLE and get_available_memory is not None:
            try:
                available = get_available_memory()
                if estimated_memory > available * 0.8:
                    diagnosis['warnings'].append(
                        f"Estimated memory ({estimated_memory:.0f}MB) exceeds 80% of available ({available:.0f}MB)"
                    )
                    diagnosis['recommendations'].append("Consider reducing batch_size or enabling fp16")
            except Exception:
                pass
        
        # 使用 DeviceManager 获取详细设备信息
        if HARDWARE_LAYER_AVAILABLE and get_device_manager is not None:
            try:
                device_mgr = get_device_manager()
                if device_mgr:
                    diagnosis['device_info'] = str(device_mgr.get_device_info(self.device) if hasattr(device_mgr, 'get_device_info') else 'unknown')
            except Exception as e:
                diagnosis['warnings'].append(f"Device info retrieval failed: {e}")

        # 检查分布式设置
        if self.use_distributed:
            dist_errors = ConfigValidator.validate_distributed_config({
                'world_size': self.world_size,
                'distributed_mode': self.distributed_mode,
            })
            if dist_errors:
                diagnosis['warnings'].extend(dist_errors)
            
            # 使用 DistributedManager 检查分布式环境
            try:
                dist_mgr = get_distributed_manager()
                if dist_mgr:
                    # is_initialized 是属性，不是方法
                    is_init = dist_mgr.is_initialized if hasattr(dist_mgr, 'is_initialized') else False
                    diagnosis['distributed_status'] = {
                        'initialized': is_init,
                        'backend': dist_mgr.backend if hasattr(dist_mgr, 'backend') else 'unknown'
                    }
            except Exception as e:
                diagnosis['warnings'].append(f"Distributed manager check failed: {e}")
        
        # 使用 ProgressManager 检查进度跟踪状态
        try:
            prog_mgr = get_progress_manager()
            if prog_mgr:
                diagnosis['progress_tracking_active'] = True
        except Exception:
                pass

        # 使用策略层诊断
        try:
            strategy_diag = diagnose_three_stage_strategy()
            diagnosis['strategy_diagnosis'] = strategy_diag
        except Exception as e:
            diagnosis['warnings'].append(f"Strategy diagnosis failed: {e}")
        
        return diagnosis

    def optimize_memory_usage(self):
        """
        主动优化内存使用
        
        调用硬件层 clear_memory 和 MemoryManager 进行清理
        """
        if HARDWARE_LAYER_AVAILABLE:
            try:
                if clear_memory is not None:
                    clear_memory()
                if MemoryManager is not None:
                    mem_mgr = MemoryManager()
                    if hasattr(mem_mgr, 'optimize'):
                        mem_mgr.optimize()
                logger.info("Memory optimization triggered from config")
            except Exception as e:
                logger.warning(f"Memory optimization failed: {e}")

    def get_progress_tracker(self) -> Optional['TrainingProgressManager']:
        """获取进度跟踪器"""
        try:
            return get_progress_manager()
        except Exception as e:
            logger.warning(f"Failed to get progress manager: {e}")
        
        return None
    
    def clone(self) -> 'ThreeStageConfig':
        """克隆配置"""
        return ThreeStageConfig.from_dict(deepcopy(self.to_dict()))
    
    def get_config_hash(self) -> str:
        """获取配置哈希值"""
        return ConfigSerializer.get_config_hash(self)
    
    def save(self, file_path: str) -> None:
        """保存配置到文件"""
        ConfigSerializer.to_file(self, file_path)
    
    @classmethod
    def load(cls, file_path: str) -> 'ThreeStageConfig':
        """从文件加载配置"""
        return ConfigSerializer.from_file(file_path, cls)
    
    def summary(self) -> str:
        """获取配置摘要"""
        enabled = [s.value for s in self.get_enabled_stages()]
        return (
            f"ThreeStageConfig["
            f"stages={enabled}, "
            f"model={self.base_model_path}, "
            f"fp16={self.use_fp16}, "
            f"distributed={self.use_distributed}]"
        )


# ==================== 预设配置 ====================

class ThreeStagePresets:
    """三阶段训练预设配置"""
    
    @staticmethod
    def standard() -> ThreeStageConfig:
        """标准配置"""
        return ThreeStageConfig(
            pretrain=StageConfig(enabled=True, epochs=1, learning_rate=1e-4, batch_size=8),
            finetune=StageConfig(enabled=True, epochs=3, learning_rate=2e-5, batch_size=4),
            preference=StageConfig(enabled=True, epochs=2, learning_rate=1e-5, batch_size=2),
        )
    
    @staticmethod
    def pretrain_only() -> ThreeStageConfig:
        """仅预训练"""
        return ThreeStageConfig(
            pretrain=StageConfig(enabled=True, epochs=3, learning_rate=1e-4, batch_size=16),
            finetune=StageConfig(enabled=False),
            preference=StageConfig(enabled=False),
        )
    
    @staticmethod
    def finetune_only() -> ThreeStageConfig:
        """仅微调"""
        return ThreeStageConfig(
            pretrain=StageConfig(enabled=False),
            finetune=StageConfig(enabled=True, epochs=5, learning_rate=2e-5, batch_size=8),
            preference=StageConfig(enabled=False),
        )
    
    @staticmethod
    def rlhf() -> ThreeStageConfig:
        """RLHF配置（微调+偏好优化）"""
        return ThreeStageConfig(
            pretrain=StageConfig(enabled=False),
            finetune=StageConfig(enabled=True, epochs=3, learning_rate=2e-5, batch_size=4),
            preference=StageConfig(enabled=True, epochs=3, learning_rate=5e-6, batch_size=2),
        )
    
    @staticmethod
    def memory_efficient() -> ThreeStageConfig:
        """内存高效配置"""
        return ThreeStageConfig(
            pretrain=StageConfig(enabled=True, epochs=1, learning_rate=1e-4, batch_size=2, gradient_accumulation_steps=4),
            finetune=StageConfig(enabled=True, epochs=3, learning_rate=2e-5, batch_size=1, gradient_accumulation_steps=8),
            preference=StageConfig(enabled=True, epochs=2, learning_rate=1e-5, batch_size=1, gradient_accumulation_steps=8),
            use_fp16=True,
        )
    
    @staticmethod
    def distributed_large_scale() -> ThreeStageConfig:
        """分布式大规模训练"""
        config = ThreeStageConfig(
            pretrain=StageConfig(enabled=True, epochs=1, learning_rate=1e-4, batch_size=32, gradient_accumulation_steps=4),
            finetune=StageConfig(enabled=True, epochs=3, learning_rate=2e-5, batch_size=16, gradient_accumulation_steps=4),
            preference=StageConfig(enabled=True, epochs=2, learning_rate=1e-5, batch_size=8, gradient_accumulation_steps=4),
            use_fp16=True,
            use_distributed=True,
            distributed_mode="fsdp",
            world_size=8,
        )
        
        # 使用分布式策略推荐
        if DISTRIBUTED_STRATEGY_AVAILABLE and recommend_distributed_mode is not None:
            try:
                recommendations = config.recommend_distributed_settings()
                if recommendations.get('distributed_mode'):
                    config.distributed_mode = recommendations['distributed_mode']
            except Exception as e:
                logger.warning(f"Failed to get distributed recommendations: {e}")
        
        return config
    
    @staticmethod
    def from_preset(name: str) -> ThreeStageConfig:
        """根据名称获取预设配置"""
        presets = {
            'standard': ThreeStagePresets.standard,
            'pretrain_only': ThreeStagePresets.pretrain_only,
            'finetune_only': ThreeStagePresets.finetune_only,
            'rlhf': ThreeStagePresets.rlhf,
            'memory_efficient': ThreeStagePresets.memory_efficient,
            'distributed': ThreeStagePresets.distributed_large_scale,
        }
        
        factory = presets.get(name)
        if factory:
            return factory()
        
        raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
    
    @staticmethod
    def list_presets() -> List[str]:
        """列出所有可用的预设"""
        return [
            'standard',
            'pretrain_only',
            'finetune_only',
            'rlhf',
            'memory_efficient',
            'distributed',
        ]


# ==================== 便捷函数 ====================

def get_layer_availability() -> Dict[str, bool]:
    """获取各层可用性"""
    return {
    }


def create_config_from_requirements(
    stages: List[str],
    base_model: str = "gpt2",
    use_distributed: bool = False,
    **kwargs
) -> ThreeStageConfig:
    """根据需求创建配置
    
    Args:
        stages: 启用的阶段列表 ["pretrain", "finetune", "preference"]
        base_model: 基础模型路径
        use_distributed: 是否使用分布式
        **kwargs: 其他配置参数
    
    Returns:
        配置实例
    """
    pretrain_enabled = "pretrain" in stages
    finetune_enabled = "finetune" in stages
    preference_enabled = "preference" in stages
    
    config = ThreeStageConfig(
        pretrain=StageConfig(enabled=pretrain_enabled),
        finetune=StageConfig(enabled=finetune_enabled),
        preference=StageConfig(enabled=preference_enabled),
        base_model_path=base_model,
        use_distributed=use_distributed,
    )
    
    # 应用额外配置
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def diagnose_config(config: ThreeStageConfig) -> Dict[str, Any]:
    """诊断配置"""
    return config.diagnose()


def optimize_config_for_hardware(config: ThreeStageConfig) -> ThreeStageConfig:
    """根据硬件优化配置
    
    Args:
        config: 原始配置
    
    Returns:
        优化后的配置
    """
    optimized = config.clone()
    
    if HARDWARE_LAYER_AVAILABLE:
        try:
            # 获取可用内存
            if get_available_memory is not None:
                available_mem = get_available_memory()
                estimated_mem = optimized.estimate_total_memory_mb()
                
                # 如果估算内存超过可用内存的80%，调整batch_size
                for stage_config in [optimized.pretrain, optimized.finetune, optimized.preference]:
                    while estimated_mem > available_mem * 0.8 and stage_config.batch_size > 1:
                        stage_config.batch_size = max(1, stage_config.batch_size // 2)
                        stage_config.gradient_accumulation_steps *= 2
                        estimated_mem = optimized.estimate_total_memory_mb()
                        logger.info(f"Reduced batch_size to {stage_config.batch_size} due to memory constraints")
            
            # 推荐精度
            if recommend_precision is not None:
                recommended = recommend_precision(optimized.device)
                if recommended in ['fp16', 'mixed']:
                    optimized.use_fp16 = True
                    logger.info("Enabled fp16 based on hardware recommendation")
                    
        except Exception as e:
            logger.warning(f"Hardware optimization failed: {e}")
    
    return optimized

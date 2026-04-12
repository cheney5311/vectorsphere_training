"""多模态训练配置模型

提供生产级多模态训练配置支持：
- 多种模态（text, image, time_series, table, audio）
- 多种融合方法（concat, attention, cross_attention, gated, late_fusion）
- 策略层集成（base_strategy, distributed_strategy, multimodal_strategy）
- 硬件层集成（device_manager, memory_manager, mixed_precision）
- 配置验证、序列化、预设工厂

架构调用层次：
├── multimodal_config.py (本模块)
│   └── 调用 backend/modules/training/strategies (策略层)
│       ├── base_strategy.py - StrategyType, TrainingPhase, StrategyMonitor
│       ├── distributed_strategy.py - DistributedMode, DistributedStrategyConfig
│       └── multimodal_strategy.py - MultiModalStrategyConfig
│   └── 调用 backend/lib/hardware (硬件层)
│       ├── DeviceManager - 设备管理
│       ├── MemoryManager - 内存管理
│       └── MixedPrecisionManager - 混合精度
│   └── 调用 backend/lib/multimodal (底层多模态库)
└── 被 multimodal_trainer.py 调用
"""

import sys
import os as os_path
import json
import logging
import hashlib
from enum import Enum
from copy import deepcopy
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable, Tuple

# 修复导入路径
current_dir = os_path.path.dirname(os_path.path.abspath(__file__))
project_root = os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(current_dir))))
sys.path.insert(0, project_root)

from dataclasses import dataclass, field, asdict
from backend.core.exceptions import ValidationError

logger = logging.getLogger(__name__)


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
except (ImportError, SyntaxError, IndentationError) as e:
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
    logger.info("Strategy layer (distributed) loaded successfully for multimodal config")
except (ImportError, SyntaxError, IndentationError) as e:
    DistributedMode = None
    ZeROStage = None
    DistributedStrategyConfig = None
    recommend_distributed_mode = None
    diagnose_distributed_strategy = None

MULTIMODAL_STRATEGY_AVAILABLE = False
try:
    from backend.modules.training.strategies.multimodal_strategy import (
        MultiModalStrategyConfig,
        FusionType,
        create_multimodal_strategy,
        diagnose_multimodal_strategy,
    )
    MULTIMODAL_STRATEGY_AVAILABLE = True
except (ImportError, SyntaxError, IndentationError) as e:
    MultiModalStrategyConfig = None
    FusionType = None
    create_multimodal_strategy = None
    diagnose_multimodal_strategy = None


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
except (ImportError, SyntaxError, IndentationError) as e:
    DeviceManager = None
    get_device_manager = None
    MemoryManager = None
    get_memory_manager = None
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


# ==================== 底层多模态库导入 ====================

LIB_MULTIMODAL_AVAILABLE = False
try:
    from backend.lib.multimodal import (
        ModalityType as LibModalityType,
        EncoderType as LibEncoderType,
        AlignmentMethod as LibAlignmentMethod,
        FusionMethod as LibFusionMethod,
        FusionStage,
        TrainingStage,
        ModalityEncoderFactory,
        CrossModalAligner,
        MultiModalFuser,
    )
    LIB_MULTIMODAL_AVAILABLE = True
    logger.info("Lib multimodal loaded successfully for multimodal config")
except ImportError as e:
    logger.warning(f"Lib multimodal not available: {e}")
    LibModalityType = None
    LibEncoderType = None
    LibAlignmentMethod = None
    LibFusionMethod = None
    FusionStage = None
    TrainingStage = None
    ModalityEncoderFactory = None
    CrossModalAligner = None
    MultiModalFuser = None


# ==================== 枚举定义 ====================

class ModalityType(str, Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    TIME_SERIES = "time_series"
    TABLE = "table"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    
    @classmethod
    def from_string(cls, value: str) -> 'ModalityType':
        """从字符串创建"""
        value = value.lower().replace('-', '_')
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown modality type: {value}")
    
    @property
    def is_visual(self) -> bool:
        """是否是视觉模态"""
        return self in (ModalityType.IMAGE, ModalityType.VIDEO)
    
    @property
    def is_sequential(self) -> bool:
        """是否是序列模态"""
        return self in (ModalityType.TEXT, ModalityType.AUDIO, ModalityType.TIME_SERIES)
    
    @property
    def default_encoder(self) -> str:
        """获取默认编码器类型"""
        mapping = {
            ModalityType.TEXT: "transformer",
            ModalityType.IMAGE: "cnn",
            ModalityType.TIME_SERIES: "lstm",
            ModalityType.TABLE: "mlp",
            ModalityType.AUDIO: "cnn1d",
            ModalityType.VIDEO: "cnn3d",
            ModalityType.DOCUMENT: "transformer",
        }
        return mapping.get(self, "linear")
    
    @property
    def default_embedding_dim(self) -> int:
        """获取默认嵌入维度"""
        dim_mapping = {
            ModalityType.TEXT: 768,
            ModalityType.IMAGE: 2048,
            ModalityType.TIME_SERIES: 256,
            ModalityType.TABLE: 256,
            ModalityType.AUDIO: 512,
            ModalityType.VIDEO: 1024,
            ModalityType.DOCUMENT: 768,
        }
        return dim_mapping.get(self, 512)
    
    def to_lib_modality(self) -> Optional['LibModalityType']:
        """转换为底层库的模态类型"""
        if LIB_MULTIMODAL_AVAILABLE and LibModalityType is not None:
            try:
                return LibModalityType.from_string(self.value)
            except Exception:
                pass
        return None
    
    def get_lib_encoder_type(self) -> Optional['LibEncoderType']:
        """获取底层库的编码器类型（使用 LibEncoderType）"""
        if LIB_MULTIMODAL_AVAILABLE and LibEncoderType is not None:
            try:
                encoder_mapping = {
                    ModalityType.TEXT: "transformer",
                    ModalityType.IMAGE: "cnn",
                    ModalityType.TIME_SERIES: "lstm",
                    ModalityType.TABLE: "mlp",
                    ModalityType.AUDIO: "cnn1d",
                    ModalityType.VIDEO: "cnn3d",
                    ModalityType.DOCUMENT: "transformer",
                }
                encoder_str = encoder_mapping.get(self, "linear")
                return LibEncoderType.from_string(encoder_str)
            except Exception as e:
                logger.warning(f"Failed to get lib encoder type: {e}")
        return None
    
    def get_alignment_method(self) -> Optional['LibAlignmentMethod']:
        """获取推荐的对齐方法（使用 LibAlignmentMethod）"""
        if LIB_MULTIMODAL_AVAILABLE and LibAlignmentMethod is not None:
            try:
                # 根据模态类型推荐对齐方法
                alignment_mapping = {
                    ModalityType.TEXT: "contrastive",
                    ModalityType.IMAGE: "contrastive",
                    ModalityType.AUDIO: "attention",
                    ModalityType.VIDEO: "cross_attention",
                    ModalityType.TIME_SERIES: "projection",
                    ModalityType.TABLE: "projection",
                }
                align_str = alignment_mapping.get(self, "contrastive")
                return LibAlignmentMethod.from_string(align_str)
            except Exception as e:
                logger.warning(f"Failed to get alignment method: {e}")
        return None


class FusionMethod(str, Enum):
    """融合方法"""
    CONCAT = "concat"
    ATTENTION = "attention"
    CROSS_ATTENTION = "cross_attention"
    GATED = "gated"
    LATE_FUSION = "late_fusion"
    EARLY_FUSION = "early_fusion"
    Q_FORMER = "q_former"
    PERCEIVER = "perceiver"
    
    @classmethod
    def from_string(cls, value: str) -> 'FusionMethod':
        """从字符串创建"""
        value = value.lower().replace('-', '_')
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown fusion method: {value}")
    
    @property
    def requires_attention(self) -> bool:
        """是否需要注意力机制"""
        return self in (
            FusionMethod.ATTENTION,
            FusionMethod.CROSS_ATTENTION,
            FusionMethod.Q_FORMER,
            FusionMethod.PERCEIVER,
        )
    
    @property
    def memory_intensity(self) -> str:
        """内存占用级别"""
        high_memory = (FusionMethod.CROSS_ATTENTION, FusionMethod.Q_FORMER, FusionMethod.PERCEIVER)
        medium_memory = (FusionMethod.ATTENTION, FusionMethod.GATED)
        if self in high_memory:
            return "high"
        elif self in medium_memory:
            return "medium"
        return "low"
    
    def to_lib_fusion(self) -> Optional['LibFusionMethod']:
        """转换为底层库的融合方法"""
        if LIB_MULTIMODAL_AVAILABLE and LibFusionMethod is not None:
            try:
                return LibFusionMethod.from_string(self.value)
            except Exception:
                pass
        return None
    
    def get_fusion_stage(self) -> Optional['FusionStage']:
        """获取融合阶段（使用 FusionStage）"""
        if LIB_MULTIMODAL_AVAILABLE and FusionStage is not None:
            try:
                # 根据融合方法确定融合阶段
                stage_mapping = {
                    FusionMethod.EARLY_FUSION: "early",
                    FusionMethod.CONCAT: "middle",
                    FusionMethod.ATTENTION: "middle",
                    FusionMethod.CROSS_ATTENTION: "middle",
                    FusionMethod.GATED: "middle",
                    FusionMethod.Q_FORMER: "middle",
                    FusionMethod.PERCEIVER: "middle",
                    FusionMethod.LATE_FUSION: "late",
                }
                stage_str = stage_mapping.get(self, "middle")
                return FusionStage.from_string(stage_str)
            except Exception as e:
                logger.warning(f"Failed to get fusion stage: {e}")
        return None
    
    def to_strategy_fusion_type(self) -> Optional['FusionType']:
        """转换为策略层的融合类型（使用 FusionType）"""
        if MULTIMODAL_STRATEGY_AVAILABLE and FusionType is not None:
            try:
                return FusionType.from_string(self.value)
            except Exception as e:
                logger.warning(f"Failed to convert to strategy fusion type: {e}")
        return None


class MultiModalScenario(str, Enum):
    """多模态场景"""
    STANDARD = "standard"
    INDUSTRY = "industry"
    MEDICAL = "medical"
    FINANCE = "finance"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    AUTONOMOUS = "autonomous"
    ROBOTICS = "robotics"
    
    @classmethod
    def from_string(cls, value: str) -> 'MultiModalScenario':
        """从字符串创建"""
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown scenario: {value}")
    
    @property
    def recommended_modalities(self) -> List[str]:
        """推荐的模态组合"""
        recommendations = {
            MultiModalScenario.STANDARD: ["text", "image"],
            MultiModalScenario.INDUSTRY: ["text", "table", "time_series", "image"],
            MultiModalScenario.MEDICAL: ["text", "image", "table"],
            MultiModalScenario.FINANCE: ["text", "table", "time_series"],
            MultiModalScenario.MANUFACTURING: ["text", "table", "time_series", "image"],
            MultiModalScenario.RETAIL: ["text", "image", "table"],
            MultiModalScenario.AUTONOMOUS: ["image", "video", "time_series"],
            MultiModalScenario.ROBOTICS: ["image", "time_series", "audio"],
        }
        return recommendations.get(self, ["text", "image"])
    
    @property
    def recommended_fusion(self) -> str:
        """推荐的融合方法"""
        fusion_recommendations = {
            MultiModalScenario.STANDARD: "concat",
            MultiModalScenario.INDUSTRY: "attention",
            MultiModalScenario.MEDICAL: "cross_attention",
            MultiModalScenario.FINANCE: "attention",
            MultiModalScenario.MANUFACTURING: "gated",
            MultiModalScenario.RETAIL: "concat",
            MultiModalScenario.AUTONOMOUS: "cross_attention",
            MultiModalScenario.ROBOTICS: "attention",
        }
        return fusion_recommendations.get(self, "concat")


# ==================== 配置验证器 ====================

class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_modality_config(config: 'ModalityConfig') -> List[str]:
        """验证模态配置"""
        errors = []
        
        if config.input_dim <= 0:
            errors.append(f"input_dim must be positive, got {config.input_dim}")
        
        if config.output_dim <= 0:
            errors.append(f"output_dim must be positive, got {config.output_dim}")
        
        if config.dropout < 0 or config.dropout >= 1:
            errors.append(f"dropout must be in [0, 1), got {config.dropout}")
        
        valid_encoder_types = ["linear", "transformer", "cnn", "lstm", "mlp", "cnn1d", "cnn3d"]
        if config.encoder_type not in valid_encoder_types:
            errors.append(f"encoder_type must be one of {valid_encoder_types}, got {config.encoder_type}")
        
        return errors
    
    @staticmethod
    def validate_multimodal_config(config: 'MultiModalConfig') -> List[str]:
        """验证多模态配置"""
        errors = []
        
        if not config.modalities:
            errors.append("modalities cannot be empty")
        
        if config.batch_size <= 0:
            errors.append(f"batch_size must be positive, got {config.batch_size}")
        
        if config.num_epochs <= 0:
            errors.append(f"num_epochs must be positive, got {config.num_epochs}")
        
        if config.learning_rate <= 0:
            errors.append(f"learning_rate must be positive, got {config.learning_rate}")
        
        if config.fusion_dim <= 0:
            errors.append(f"fusion_dim must be positive, got {config.fusion_dim}")
        
        # 验证融合方法
        valid_fusion_methods = [f.value for f in FusionMethod]
        if config.fusion_method not in valid_fusion_methods:
            errors.append(f"fusion_method must be one of {valid_fusion_methods}, got {config.fusion_method}")

        
        return errors
    
    @staticmethod
    def validate_distributed_config(config: Dict[str, Any]) -> List[str]:
        """验证分布式配置"""
        errors = []
        
        world_size = config.get('world_size', 1)
        if world_size < 1:
            errors.append(f"world_size must be at least 1, got {world_size}")
        
        # 使用分布式策略诊断（如果可用）
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
    def to_json(config: Union['ModalityConfig', 'MultiModalConfig'], indent: int = 2) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(config.to_dict(), indent=indent, ensure_ascii=False)
    
    @staticmethod
    def from_json(json_str: str, config_class: type) -> Union['ModalityConfig', 'MultiModalConfig']:
        """从 JSON 字符串创建配置"""
        data = json.loads(json_str)
        return config_class.from_dict(data)
    
    @staticmethod
    def to_file(config: Union['ModalityConfig', 'MultiModalConfig'], file_path: str) -> None:
        """保存配置到文件"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Config saved to {file_path}")
    
    @staticmethod
    def from_file(file_path: str, config_class: type) -> Union['ModalityConfig', 'MultiModalConfig']:
        """从文件加载配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return config_class.from_dict(data)
    
    @staticmethod
    def get_config_hash(config: Union['ModalityConfig', 'MultiModalConfig']) -> str:
        """获取配置的哈希值"""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]


# ==================== 模态配置 ====================

@dataclass
class ModalityConfig:
    """单个模态的配置"""
    name: str
    input_dim: int
    output_dim: int = 768
    encoder_type: str = "linear"
    pretrained_model: Optional[str] = None
    trainable: bool = True
    dropout: float = 0.1
    use_adapter: bool = False
    adapter_dim: int = 64
    pooling_type: str = "mean"  # mean, max, cls, attention
    
    def __post_init__(self):
        """初始化后处理"""
        # 验证配置
        errors = ConfigValidator.validate_modality_config(self)
        if errors:
            raise ValidationError(f"Invalid modality config: {'; '.join(errors)}", field="modality_config")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'encoder_type': self.encoder_type,
            'pretrained_model': self.pretrained_model,
            'trainable': self.trainable,
            'dropout': self.dropout,
            'use_adapter': self.use_adapter,
            'adapter_dim': self.adapter_dim,
            'pooling_type': self.pooling_type,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModalityConfig':
        """从字典创建"""
        return cls(
            name=data.get('name', 'unknown'),
            input_dim=data.get('input_dim', 768),
            output_dim=data.get('output_dim', 768),
            encoder_type=data.get('encoder_type', 'linear'),
            pretrained_model=data.get('pretrained_model'),
            trainable=data.get('trainable', True),
            dropout=data.get('dropout', 0.1),
            use_adapter=data.get('use_adapter', False),
            adapter_dim=data.get('adapter_dim', 64),
            pooling_type=data.get('pooling_type', 'mean'),
        )
    
    def get_modality_type(self) -> ModalityType:
        """获取模态类型枚举"""
        return ModalityType.from_string(self.name)
    
    def estimate_memory_mb(self) -> float:
        """估算内存占用（MB）"""
        # 基础估算：参数数量 * 4 bytes (float32)
        param_estimate = self.input_dim * self.output_dim
        if self.use_adapter:
            param_estimate += self.output_dim * self.adapter_dim * 2
        
        memory_mb = (param_estimate * 4) / (1024 * 1024)
        
        # 使用硬件层估算（如果可用）
        if HARDWARE_LAYER_AVAILABLE and estimate_model_memory is not None:
            try:
                estimated = estimate_model_memory(param_estimate)
                if estimated > 0:
                    memory_mb = estimated
            except Exception as e:
                logger.warning(f"Hardware memory estimation failed: {e}")
        
        return memory_mb
    
    def create_encoder(self) -> Optional[Any]:
        """创建编码器（使用底层库）"""
        if LIB_MULTIMODAL_AVAILABLE and ModalityEncoderFactory is not None:
            try:
                lib_modality = self.get_modality_type().to_lib_modality()
                if lib_modality is not None:
                    # ModalityEncoderFactory 使用 create_encoder 方法
                    encoder_config = {
                        'input_dim': self.input_dim,
                        'output_dim': self.output_dim,
                        'pretrained_model': self.pretrained_model,
                        'dropout': self.dropout,
                    }
                    encoder = ModalityEncoderFactory.create_encoder(
                        modality=lib_modality,
                        config=encoder_config,
                    )
                    return encoder
            except Exception as e:
                logger.warning(f"Failed to create encoder from lib: {e}")
        return None
    
    def get_strategy_config(self) -> Optional[Dict[str, Any]]:
        """获取策略层配置"""
        if MULTIMODAL_STRATEGY_AVAILABLE and MultiModalStrategyConfig is not None:
            return {
                'modality': self.name,
                'encoder_type': self.encoder_type,
                'hidden_dim': self.output_dim,
                'use_adapter': self.use_adapter,
            }
        return None
    
    def clone(self) -> 'ModalityConfig':
        """克隆配置"""
        return ModalityConfig.from_dict(self.to_dict())
    
    def get_lib_encoder_type(self) -> Optional['LibEncoderType']:
        """获取底层库的编码器类型（使用 LibEncoderType）"""
        if LIB_MULTIMODAL_AVAILABLE and LibEncoderType is not None:
            try:
                return LibEncoderType.from_string(self.encoder_type)
            except Exception as e:
                logger.warning(f"Failed to get lib encoder type: {e}")
        return None
    
    def get_training_stage(self) -> Optional['TrainingStage']:
        """获取推荐的训练阶段（使用 TrainingStage）"""
        if LIB_MULTIMODAL_AVAILABLE and TrainingStage is not None:
            try:
                # 根据模态配置推荐训练阶段
                if self.pretrained_model:
                    return TrainingStage.from_string("finetune")
                else:
                    return TrainingStage.from_string("pretrain")
            except Exception as e:
                logger.warning(f"Failed to get training stage: {e}")
        return None


# ==================== 多模态配置 ====================

@dataclass 
class MultiModalConfig:
    """多模态配置"""
    # 模态配置
    modalities: List[str] = field(default_factory=lambda: ["text", "image"])
    modality_dims: Dict[str, int] = field(default_factory=lambda: {"text": 768, "image": 2048})
    modality_configs: Dict[str, ModalityConfig] = field(default_factory=dict)
    
    # 场景配置
    scenario: str = "standard"
    
    # 文本配置
    text_model_name: str = "bert-base-uncased"
    max_text_length: int = 512
    text_embedding_dim: int = 768
    
    # 图像配置
    image_model_name: str = "resnet50"
    image_size: int = 224
    num_image_channels: int = 3
    image_embedding_dim: int = 2048
    
    # 时序配置
    time_series_length: int = 128
    time_series_channels: int = 16
    time_series_embedding_dim: int = 256
    
    # 表格配置
    table_num_features: int = 64
    table_embedding_dim: int = 256
    
    # 音频配置
    audio_sample_rate: int = 16000
    audio_max_length: int = 160000
    audio_embedding_dim: int = 512
    
    # 融合配置
    fusion_method: str = "concat"
    fusion_dim: int = 1024
    num_fusion_layers: int = 2
    fusion_dropout: float = 0.1
    
    # 对齐配置
    use_alignment: bool = True
    alignment_temperature: float = 0.07
    alignment_dim: int = 512
    alignment_method: str = "contrastive"
    
    # 对比学习配置
    use_contrastive: bool = False
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 0.1
    
    # 训练配置
    learning_rate: float = 5e-5
    batch_size: int = 16
    num_epochs: int = 10
    warmup_steps: int = 500
    warmup_ratio: float = 0.1
    
    # 优化配置
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    use_fp16: bool = True
    gradient_accumulation_steps: int = 1
    
    # 损失权重配置
    task_loss_weight: float = 1.0
    align_loss_weight: float = 0.5
    contrastive_loss_weight: float = 0.1
    
    # 模态dropout
    modality_dropout: float = 0.1
    
    # 数据配置
    train_data_path: str = "./data/multimodal/train"
    val_data_path: Optional[str] = "./data/multimodal/val"
    test_data_path: Optional[str] = "./data/multimodal/test"
    
    # 输出配置
    output_dir: str = "./outputs/multimodal"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # 策略配置
    use_strategy: bool = True
    strategy_type: str = "multimodal"
    
    # 分布式配置
    use_distributed: bool = False
    distributed_mode: str = "ddp"
    world_size: int = 1
    local_rank: int = -1
    
    # 硬件配置
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        """初始化后验证"""
        self.validate()
        self._init_modality_configs()
        self._init_strategy_integration()
        self._init_hardware_optimization()
        
    def _init_modality_configs(self):
        """初始化各模态配置"""
        for modality in self.modalities:
            if modality not in self.modality_configs:
                modality_type = ModalityType.from_string(modality)
                
                if modality == "text":
                    self.modality_configs[modality] = ModalityConfig(
                        name=modality,
                        input_dim=self.max_text_length,
                        output_dim=self.text_embedding_dim,
                        encoder_type="transformer",
                        pretrained_model=self.text_model_name
                    )
                elif modality == "image":
                    self.modality_configs[modality] = ModalityConfig(
                        name=modality,
                        input_dim=self.image_size * self.image_size * self.num_image_channels,
                        output_dim=self.image_embedding_dim,
                        encoder_type="cnn",
                        pretrained_model=self.image_model_name
                    )
                elif modality == "time_series":
                    self.modality_configs[modality] = ModalityConfig(
                        name=modality,
                        input_dim=self.time_series_length * self.time_series_channels,
                        output_dim=self.time_series_embedding_dim,
                        encoder_type="lstm"
                    )
                elif modality == "table":
                    self.modality_configs[modality] = ModalityConfig(
                        name=modality,
                        input_dim=self.table_num_features,
                        output_dim=self.table_embedding_dim,
                        encoder_type="mlp"
                    )
                elif modality == "audio":
                    self.modality_configs[modality] = ModalityConfig(
                        name=modality,
                        input_dim=self.audio_max_length,
                        output_dim=self.audio_embedding_dim,
                        encoder_type="cnn1d"
                    )
                else:
                    # 使用默认配置
                    self.modality_configs[modality] = ModalityConfig(
                        name=modality,
                        input_dim=modality_type.default_embedding_dim,
                        output_dim=modality_type.default_embedding_dim,
                        encoder_type=modality_type.default_encoder
                    )
    
    def _init_strategy_integration(self):
        """初始化策略层集成"""
        # 创建策略监控器（如果可用）
        try:
            self._strategy_monitor = StrategyMonitor()
            logger.debug("Strategy monitor initialized for multimodal config")
        except Exception as e:
            logger.warning(f"Failed to create strategy monitor: {e}")
            self._strategy_monitor = None
        
        # 创建策略分析器（如果可用）
        try:
            self._strategy_profiler = StrategyProfiler()
            logger.debug("Strategy profiler initialized for multimodal config")
        except Exception as e:
            logger.warning(f"Failed to create strategy profiler: {e}")
            self._strategy_profiler = None
        
        # 创建策略验证器（如果可用）
        try:
            self._strategy_validator = StrategyValidator()
            logger.debug("Strategy validator initialized for multimodal config")
        except Exception as e:
            logger.warning(f"Failed to create strategy validator: {e}")
            self._strategy_validator = None
    
    def _init_hardware_optimization(self):
        """初始化硬件优化"""
        # 初始化硬件管理器引用
        self._device_manager: Optional['DeviceManager'] = None
        self._memory_manager: Optional['MemoryManager'] = None
        self._mixed_precision_manager: Optional['MixedPrecisionManager'] = None
        
        try:
            # 获取设备管理器（使用 DeviceManager）
            if get_device_manager is not None:
                self._device_manager = get_device_manager()
                if self._device_manager is not None:
                    # 检测最佳设备 - 使用 get_device 方法
                    best_device = self._device_manager.get_device()
                    if best_device:
                        self.device = str(best_device)
                        logger.debug("Using best device from DeviceManager: %s", self.device)
                
            # 创建内存管理器
            if MemoryManager is not None:
                try:
                    self._memory_manager = MemoryManager()
                    logger.debug("MemoryManager initialized for config")
                except Exception as e:
                    logger.warning(f"Failed to create MemoryManager: {e}")
                
            # 创建混合精度管理器（使用 MixedPrecisionManager 和 AmpConfig）
            if MixedPrecisionManager is not None and AmpConfig is not None and self.use_fp16:
                try:
                    amp_config = AmpConfig(
                        enabled=True,
                        dtype='float16',
                        init_scale=65536.0,
                        growth_factor=2.0,
                        backoff_factor=0.5,
                        growth_interval=2000,
                    )

                    self._mixed_precision_manager = MixedPrecisionManager(amp_config)
                    logger.debug("MixedPrecisionManager initialized for config")
                except Exception as e:
                    logger.warning(f"Failed to create MixedPrecisionManager: {e}")
                
                # 获取内存管理器并推荐batch size
            if recommend_batch_size is not None and get_available_memory is not None:
                try:
                    available_mem = get_available_memory()
                    estimated_mem = self.estimate_memory_per_sample()
                    if estimated_mem > 0 and available_mem > 0:
                        recommended = recommend_batch_size(
                            model_memory_mb=estimated_mem * self.batch_size,
                            available_memory_mb=available_mem
                        )
                        if recommended and recommended > 0:
                            # 只有当推荐值更小时才调整
                            if recommended < self.batch_size:
                                logger.info(f"Adjusting batch_size from {self.batch_size} to {recommended} based on available memory")
                                self.batch_size = recommended
                except Exception as e:
                    logger.warning(f"Failed to recommend batch size: {e}")
        
                # 推荐精度模式（使用 PrecisionMode）
            if recommend_precision is not None:
                try:
                    recommended_precision = recommend_precision(self.device)
                    if recommended_precision:
                        self.use_fp16 = recommended_precision in ['fp16', 'mixed']
                        # 使用 PrecisionMode 枚举记录推荐的精度
                        if PrecisionMode is not None:
                            try:
                                self._recommended_precision_mode = PrecisionMode.from_string(recommended_precision)
                                logger.debug(f"Recommended PrecisionMode: {self._recommended_precision_mode}")
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Failed to recommend precision: {e}")
                        
        except Exception as e:
            logger.warning(f"Hardware optimization initialization failed: {e}")
        
    def validate(self) -> None:
        """验证配置参数"""
        errors = ConfigValidator.validate_multimodal_config(self)
        if errors:
            raise ValidationError(f"Invalid config: {'; '.join(errors)}", field="multimodal_config")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'modalities': self.modalities,
            'modality_dims': self.modality_dims,
            'modality_configs': {k: v.to_dict() for k, v in self.modality_configs.items()},
            'scenario': self.scenario,
            'text_model_name': self.text_model_name,
            'max_text_length': self.max_text_length,
            'text_embedding_dim': self.text_embedding_dim,
            'image_model_name': self.image_model_name,
            'image_size': self.image_size,
            'num_image_channels': self.num_image_channels,
            'image_embedding_dim': self.image_embedding_dim,
            'time_series_length': self.time_series_length,
            'time_series_channels': self.time_series_channels,
            'time_series_embedding_dim': self.time_series_embedding_dim,
            'table_num_features': self.table_num_features,
            'table_embedding_dim': self.table_embedding_dim,
            'audio_sample_rate': self.audio_sample_rate,
            'audio_max_length': self.audio_max_length,
            'audio_embedding_dim': self.audio_embedding_dim,
            'fusion_method': self.fusion_method,
            'fusion_dim': self.fusion_dim,
            'num_fusion_layers': self.num_fusion_layers,
            'fusion_dropout': self.fusion_dropout,
            'use_alignment': self.use_alignment,
            'alignment_temperature': self.alignment_temperature,
            'alignment_dim': self.alignment_dim,
            'alignment_method': self.alignment_method,
            'use_contrastive': self.use_contrastive,
            'contrastive_temperature': self.contrastive_temperature,
            'contrastive_weight': self.contrastive_weight,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'warmup_steps': self.warmup_steps,
            'warmup_ratio': self.warmup_ratio,
            'weight_decay': self.weight_decay,
            'gradient_clipping': self.gradient_clipping,
            'use_fp16': self.use_fp16,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'task_loss_weight': self.task_loss_weight,
            'align_loss_weight': self.align_loss_weight,
            'contrastive_loss_weight': self.contrastive_loss_weight,
            'modality_dropout': self.modality_dropout,
            'train_data_path': self.train_data_path,
            'val_data_path': self.val_data_path,
            'test_data_path': self.test_data_path,
            'output_dir': self.output_dir,
            'save_steps': self.save_steps,
            'eval_steps': self.eval_steps,
            'logging_steps': self.logging_steps,
            'use_strategy': self.use_strategy,
            'strategy_type': self.strategy_type,
            'use_distributed': self.use_distributed,
            'distributed_mode': self.distributed_mode,
            'world_size': self.world_size,
            'local_rank': self.local_rank,
            'device': self.device,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiModalConfig':
        """从字典创建配置"""
        # 处理 modality_configs
        modality_configs = {}
        if 'modality_configs' in data:
            for k, v in data['modality_configs'].items():
                if isinstance(v, dict):
                    modality_configs[k] = ModalityConfig.from_dict(v)
                else:
                    modality_configs[k] = v
        
        # 创建配置实例（需要临时绕过 __post_init__）
        config = object.__new__(cls)
        
        # 设置所有字段
        for f in data:
            if f == 'modality_configs':
                continue
            if hasattr(config, f):
                setattr(config, f, data[f])
        
        config.modality_configs = modality_configs
        
        # 手动调用初始化方法
        config._init_modality_configs()
        config._init_strategy_integration()
        config._init_hardware_optimization()
        
        return config
    
    def estimate_memory_per_sample(self) -> float:
        """估算每个样本的内存占用（MB）"""
        total_memory = 0.0
        
        for modality, config in self.modality_configs.items():
            total_memory += config.estimate_memory_mb()
        
        # 融合层内存
        fusion_memory = (self.fusion_dim * self.num_fusion_layers * 4) / (1024 * 1024)
        total_memory += fusion_memory
        
        return total_memory
    
    def estimate_total_memory_mb(self) -> float:
        """估算总内存占用（MB）"""
        per_sample = self.estimate_memory_per_sample()
        
        # 考虑batch size和梯度
        multiplier = 3.0 if self.use_fp16 else 4.0  # 模型 + 梯度 + 优化器状态
        
        total = per_sample * self.batch_size * multiplier
        
        # 使用硬件层估算（如果可用）
        if HARDWARE_LAYER_AVAILABLE and estimate_model_memory is not None:
            try:
                param_count = sum(
                    config.input_dim * config.output_dim 
                    for config in self.modality_configs.values()
                )
                estimated = estimate_model_memory(param_count)
                if estimated > 0:
                    total = estimated * self.batch_size * multiplier
            except Exception as e:
                logger.warning(f"Hardware memory estimation failed: {e}")
        
        return total
    
    def get_device_info(self) -> Optional[Dict[str, Any]]:
        """获取设备信息（使用 DeviceManager 和 DeviceInfo）"""
        if HARDWARE_LAYER_AVAILABLE and self._device_manager is not None:
            try:
                device_info = self._device_manager.get_device_info(self.device)
                if device_info is not None and DeviceInfo is not None:
                    return {
                        'device': self.device,
                        'device_type': device_info.device_type.value if hasattr(device_info.device_type, 'value') else str(device_info.device_type),
                        'memory_total': device_info.memory_total if hasattr(device_info, 'memory_total') else None,
                        'memory_available': device_info.memory_available if hasattr(device_info, 'memory_available') else None,
                        'compute_capability': device_info.compute_capability if hasattr(device_info, 'compute_capability') else None,
                    }
            except Exception as e:
                logger.warning(f"Failed to get device info: {e}")
        return {'device': self.device, 'device_type': 'unknown'}
    
    def get_device_type(self) -> Optional['DeviceType']:
        """获取设备类型枚举（使用 DeviceType）"""
        if HARDWARE_LAYER_AVAILABLE and DeviceType is not None:
            try:
                if 'cuda' in self.device.lower():
                    return DeviceType.GPU  # CUDA 对应 GPU 枚举
                elif 'cpu' in self.device.lower():
                    return DeviceType.CPU
                elif 'mps' in self.device.lower():
                    return DeviceType.MPS
                else:
                    return DeviceType.from_string(self.device)
            except Exception as e:
                logger.warning(f"Failed to get device type: {e}")
        return None
    
    def clear_device_memory(self) -> bool:
        """清理设备内存（使用 clear_memory）"""
        if HARDWARE_LAYER_AVAILABLE and clear_memory is not None:
            try:
                clear_memory()
                logger.debug("Device memory cleared")
                return True
            except Exception as e:
                logger.warning(f"Failed to clear memory: {e}")
        return False
    
    def get_amp_config(self) -> Optional['AmpConfig']:
        """获取混合精度配置（使用 AmpConfig）"""
        if HARDWARE_LAYER_AVAILABLE and AmpConfig is not None and self.use_fp16:
            try:
                return AmpConfig(
                    enabled=True,
                    dtype='float16',
                    init_scale=65536.0,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                )
            except Exception as e:
                logger.warning(f"Failed to create AmpConfig: {e}")
        return None
    
    def get_precision_mode(self) -> Optional['PrecisionMode']:
        """获取精度模式（使用 PrecisionMode）"""
        if HARDWARE_LAYER_AVAILABLE and PrecisionMode is not None:
            try:
                if self.use_fp16:
                    return PrecisionMode.FP16
                else:
                    return PrecisionMode.FP32
            except Exception as e:
                logger.warning(f"Failed to get precision mode: {e}")
        return None
    
    def get_zero_stage(self) -> Optional['ZeROStage']:
        """获取 ZeRO 优化阶段（使用 ZeROStage）"""
        try:
            # 根据配置推荐 ZeRO 阶段
            if self.world_size >= 8:
                return ZeROStage.STAGE_3  # 大规模分布式使用 Stage 3
            elif self.world_size >= 4:
                return ZeROStage.STAGE_2  # 中等规模使用 Stage 2
            else:
                return ZeROStage.STAGE_1  # 小规模使用 Stage 1
        except Exception as e:
            logger.warning(f"Failed to get ZeRO stage: {e}")
        return None
    
    def create_strategy_result(self, loss: float = 0.0, metrics: Optional[Dict[str, float]] = None) -> Optional['StrategyResult']:
        """创建策略结果（使用 StrategyResult）"""
        try:
            return StrategyResult(
                loss=loss,
                metrics=metrics or {},
                metadata={
                    'config_hash': self.get_config_hash() if hasattr(self, 'get_config_hash') else None,
                    'scenario': self.scenario,
                    'modalities': self.modalities,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to create strategy result: {e}")
        return None
    
    def get_training_stages(self) -> List[Optional['TrainingStage']]:
        """获取所有模态的训练阶段（使用 TrainingStage）"""
        stages = []
    
        for modality, config in self.modality_configs.items():
            stage = config.get_training_stage()
            if stage is not None:
                stages.append(stage)
        return stages
    
    def get_fusion_stage(self) -> Optional['FusionStage']:
        """获取融合阶段（使用 FusionStage）"""
        try:
            fusion_method = FusionMethod.from_string(self.fusion_method)
            return fusion_method.get_fusion_stage()
        except Exception as e:
            logger.warning(f"Failed to get fusion stage: {e}")
        return None
    
    def get_strategy_fusion_type(self) -> Optional['FusionType']:
        """获取策略层融合类型（使用 FusionType）"""
        try:
            fusion_method = FusionMethod.from_string(self.fusion_method)
            return fusion_method.to_strategy_fusion_type()
        except Exception as e:
            logger.warning(f"Failed to get strategy fusion type: {e}")
        return None
    
    def get_config_hash(self) -> str:
        """获取配置哈希值"""
        return ConfigSerializer.get_config_hash(self)
    
    def get_strategy_type_enum(self) -> Optional['StrategyType']:
        """获取策略类型枚举"""
        try:
            return StrategyType.from_string(self.strategy_type)
        except Exception:
            return StrategyType.MULTIMODAL
    
    def get_training_phase(self) -> Optional['TrainingPhase']:
        """获取训练阶段"""
        if TrainingPhase is not None:
            return TrainingPhase.MAIN
        return None
    
    def get_distributed_mode_enum(self) -> Optional['DistributedMode']:
        """获取分布式模式枚举"""
        try:
            return DistributedMode.from_string(self.distributed_mode)
        except Exception:
            return DistributedMode.DDP
    
    def create_strategy_context(self) -> Optional['StrategyContext']:
        """创建策略上下文"""
        try:
            return StrategyContext(
                phase=self.get_training_phase(),
                strategy_type=self.get_strategy_type_enum(),
                config=self.to_dict(),
                metadata={
                    'scenario': self.scenario,
                    'modalities': self.modalities,
                    'fusion_method': self.fusion_method,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to create strategy context: {e}")
        return None
    
    def create_strategy_metrics(self) -> Optional['StrategyMetrics']:
        """创建策略指标跟踪器"""
    
        try:
            return StrategyMetrics()
        except Exception as e:
            logger.warning(f"Failed to create strategy metrics: {e}")
        return None
    
    def create_distributed_config(self) -> Optional['DistributedStrategyConfig']:
        """创建分布式策略配置"""
    
        try:

            mode = self.get_distributed_mode_enum()
            return DistributedStrategyConfig(
                distributed_mode=mode,
                world_size=self.world_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                fp16=self.use_fp16,
            )
        except Exception as e:
            logger.warning(f"Failed to create distributed config: {e}")
        return None
    
    def recommend_distributed_settings(self) -> Dict[str, Any]:
        """获取推荐的分布式设置"""
        recommendations = {
            'use_distributed': False,
            'distributed_mode': 'ddp',
            'world_size': 1,
        }
        
        
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
    
    def create_multimodal_strategy(self) -> Optional[Any]:
        """创建多模态策略（使用策略层）"""
       
        try:
            strategy_config = MultiModalStrategyConfig(
                modalities=self.modalities,
                fusion_method=self.fusion_method,
                use_alignment=self.use_alignment,
                alignment_method=self.alignment_method,
            ) if MultiModalStrategyConfig else None
                
            return create_multimodal_strategy(strategy_config)
        except Exception as e:
            logger.warning(f"Failed to create multimodal strategy: {e}")
        return None
    
    def create_fusion_module(self) -> Optional[Any]:
        """创建融合模块（使用底层库）"""
       
        try:
            return MultiModalFuser(
                input_dims=self.modality_dims,
                fusion_dim=self.fusion_dim,
                fusion_method=self.fusion_method,
                num_layers=self.num_fusion_layers,
                dropout=self.fusion_dropout,
            )
        except Exception as e:
            logger.warning(f"Failed to create fusion module: {e}")
        return None
    
    def create_alignment_module(self) -> Optional[Any]:
        """创建对齐模块（使用底层库）"""
    
        try:
            return CrossModalAligner(
                input_dim=self.alignment_dim,
                alignment_method=self.alignment_method,
                temperature=self.alignment_temperature,
            )
        except Exception as e:
            logger.warning(f"Failed to create alignment module: {e}")
        return None
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断配置"""
        diagnosis = {
            'config_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
        }
        
        # 验证配置
        errors = ConfigValidator.validate_multimodal_config(self)
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
        
        # 检查分布式设置
        if self.use_distributed:
            dist_errors = ConfigValidator.validate_distributed_config({
                'world_size': self.world_size,
                'distributed_mode': self.distributed_mode,
            })
            if dist_errors:
                diagnosis['warnings'].extend(dist_errors)
        
        # 使用策略层诊断
        if MULTIMODAL_STRATEGY_AVAILABLE and diagnose_multimodal_strategy is not None:
            try:
                strategy_diag = diagnose_multimodal_strategy()
                diagnosis['strategy_diagnosis'] = strategy_diag
            except Exception as e:
                diagnosis['warnings'].append(f"Strategy diagnosis failed: {e}")
        
        # 层可用性
        diagnosis['layer_availability'] = {
            'strategy_layer': STRATEGY_LAYER_AVAILABLE,
            'distributed_strategy': DISTRIBUTED_STRATEGY_AVAILABLE,
            'multimodal_strategy': MULTIMODAL_STRATEGY_AVAILABLE,
            'hardware_layer': HARDWARE_LAYER_AVAILABLE,
            'lib_multimodal': LIB_MULTIMODAL_AVAILABLE,
        }
        
        return diagnosis
    
    def clone(self) -> 'MultiModalConfig':
        """克隆配置"""
        return MultiModalConfig.from_dict(deepcopy(self.to_dict()))
    
    def save(self, file_path: str) -> None:
        """保存配置到文件"""
        ConfigSerializer.to_file(self, file_path)
    
    @classmethod
    def load(cls, file_path: str) -> 'MultiModalConfig':
        """从文件加载配置"""
        return ConfigSerializer.from_file(file_path, cls)
    
    def summary(self) -> str:
        """获取配置摘要"""
        return (
            f"MultiModalConfig["
            f"modalities={self.modalities}, "
            f"scenario={self.scenario}, "
            f"fusion={self.fusion_method}, "
            f"batch_size={self.batch_size}, "
            f"epochs={self.num_epochs}, "
            f"distributed={self.use_distributed}]"
        )


# ==================== 行业多模态配置 ====================

@dataclass
class IndustryMultiModalConfig(MultiModalConfig):
    """行业多模态配置"""
    
    # 行业特定模态
    modalities: List[str] = field(default_factory=lambda: ["text", "table", "time_series", "image"])
    
    # 行业配置
    industry_type: str = "manufacturing"
    
    # 时序信号配置（传感器、PLC、SCADA）
    sensor_channels: int = 32
    plc_signals: int = 16
    scada_features: int = 8
    
    # 工艺文档配置
    document_max_length: int = 1024
    
    # BOM表格配置
    bom_features: int = 128
    
    # 缺陷检测图像配置
    defect_image_size: int = 512
    
    # 行业特定的融合配置
    fusion_method: str = "attention"
    use_alignment: bool = True
    align_loss_weight: float = 0.3
    contrastive_loss_weight: float = 0.2
    
    # 行业特定的训练配置
    use_domain_adaptation: bool = True
    domain_adaptation_weight: float = 0.1
    
    def __post_init__(self):
        """初始化行业配置"""
        # 设置行业特定的模态维度
        self.modality_dims = {
            "text": 768,
            "table": 256,
            "time_series": 256,
            "image": 2048
        }
        
        # 调用父类初始化
        super().__post_init__()
        
        # 设置策略类型
        self.strategy_type = "industry_multimodal"

        # 行业特定初始化
        self._init_industry_specific()
    
    def _init_industry_specific(self):
        """初始化行业特定配置"""
        # 获取场景推荐
        try:
            scenario = MultiModalScenario.from_string(self.industry_type)
            recommended_modalities = scenario.recommended_modalities
            
            # 确保包含推荐的模态
            for modality in recommended_modalities:
                if modality not in self.modalities:
                    self.modalities.append(modality)
                    logger.info(f"Added recommended modality for {self.industry_type}: {modality}")
        except Exception:
            pass
        
        # 使用策略层获取行业配置
    
        try:
            self._industry_strategy_type = StrategyType.INDUSTRY
            logger.debug(f"Set industry strategy type for {self.industry_type}")
        except Exception as e:
            logger.warning(f"Failed to set industry strategy type: {e}")
    
    def get_industry_scenario(self) -> MultiModalScenario:
        """获取行业场景"""
        try:
            return MultiModalScenario.from_string(self.industry_type)
        except Exception:
            return MultiModalScenario.INDUSTRY
    
    def get_sensor_config(self) -> Dict[str, Any]:
        """获取传感器配置"""
        return {
            'sensor_channels': self.sensor_channels,
            'plc_signals': self.plc_signals,
            'scada_features': self.scada_features,
            'total_features': self.sensor_channels + self.plc_signals + self.scada_features,
        }
    
    def create_industry_strategy_context(self) -> Optional['StrategyContext']:
        """创建行业策略上下文"""
        context = self.create_strategy_context()
        if context is not None:
            # StrategyContext 使用 extra 字段而非 metadata
            context.extra['industry_type'] = self.industry_type
            context.extra['sensor_config'] = self.get_sensor_config()
            context.extra['use_domain_adaptation'] = self.use_domain_adaptation
        return context
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = super().to_dict()
        base_dict.update({
            'industry_type': self.industry_type,
            'sensor_channels': self.sensor_channels,
            'plc_signals': self.plc_signals,
            'scada_features': self.scada_features,
            'document_max_length': self.document_max_length,
            'bom_features': self.bom_features,
            'defect_image_size': self.defect_image_size,
            'use_domain_adaptation': self.use_domain_adaptation,
            'domain_adaptation_weight': self.domain_adaptation_weight,
        })
        return base_dict


# ==================== 多模态预设配置 ====================

class MultiModalPresets:
    """多模态预设配置"""
    
    @staticmethod
    def standard_classification() -> MultiModalConfig:
        """标准分类任务"""
        return MultiModalConfig(
            modalities=["text", "image"],
            fusion_method="concat",
            use_alignment=False,
            use_contrastive=False,
            scenario="standard"
        )
    
    @staticmethod
    def multimodal_alignment() -> MultiModalConfig:
        """多模态对齐任务"""
        return MultiModalConfig(
            modalities=["text", "image"],
            fusion_method="attention",
            use_alignment=True,
            use_contrastive=True,
            align_loss_weight=0.5,
            contrastive_loss_weight=0.2,
            scenario="standard"
        )
    
    @staticmethod
    def manufacturing_scenario() -> IndustryMultiModalConfig:
        """制造业场景"""
        return IndustryMultiModalConfig(
            industry_type="manufacturing",
            modalities=["text", "table", "time_series", "image"],
            scenario="manufacturing"
        )
    
    @staticmethod
    def finance_scenario() -> IndustryMultiModalConfig:
        """金融场景"""
        return IndustryMultiModalConfig(
            industry_type="finance",
            modalities=["text", "table", "time_series"],
            scenario="finance",
            time_series_length=256,
            table_num_features=128
        )
    
    @staticmethod
    def medical_scenario() -> IndustryMultiModalConfig:
        """医疗场景"""
        return IndustryMultiModalConfig(
            industry_type="medical",
            modalities=["text", "image", "table"],
            scenario="medical",
            image_size=512,
            document_max_length=2048
        )
    
    @staticmethod
    def retail_scenario() -> IndustryMultiModalConfig:
        """零售场景"""
        return IndustryMultiModalConfig(
            industry_type="retail",
            modalities=["text", "image", "table"],
            scenario="retail"
        )

    @staticmethod
    def autonomous_driving() -> MultiModalConfig:
        """自动驾驶场景"""
        return MultiModalConfig(
            modalities=["image", "time_series"],
            fusion_method="cross_attention",
            use_alignment=True,
            use_contrastive=True,
            image_size=640,
            time_series_length=64,
            scenario="autonomous"
        )
    
    @staticmethod
    def video_understanding() -> MultiModalConfig:
        """视频理解场景"""
        return MultiModalConfig(
            modalities=["text", "image"],  # 将视频拆分为帧
            fusion_method="cross_attention",
            use_alignment=True,
            image_size=224,
            max_text_length=256,
            scenario="standard"
        )
    
    @staticmethod
    def distributed_large_scale() -> MultiModalConfig:
        """分布式大规模训练"""
        config = MultiModalConfig(
            modalities=["text", "image"],
            fusion_method="q_former",
            use_alignment=True,
            use_contrastive=True,
            use_distributed=True,
            distributed_mode="fsdp",
            world_size=8,
            batch_size=32,
            gradient_accumulation_steps=4,
            use_fp16=True,
        )
        
        # 使用分布式策略推荐（如果可用）
    
        try:
            recommendations = config.recommend_distributed_settings()
            if recommendations.get('distributed_mode'):
                config.distributed_mode = recommendations['distributed_mode']
        except Exception as e:
            logger.warning(f"Failed to get distributed recommendations: {e}")
        
        return config
    
    @staticmethod
    def from_scenario(scenario: str) -> MultiModalConfig:
        """根据场景获取预设配置"""
        scenario_mapping = {
            'standard': MultiModalPresets.standard_classification,
            'alignment': MultiModalPresets.multimodal_alignment,
            'manufacturing': MultiModalPresets.manufacturing_scenario,
            'finance': MultiModalPresets.finance_scenario,
            'medical': MultiModalPresets.medical_scenario,
            'retail': MultiModalPresets.retail_scenario,
            'autonomous': MultiModalPresets.autonomous_driving,
            'video': MultiModalPresets.video_understanding,
            'distributed': MultiModalPresets.distributed_large_scale,
        }
        
        factory = scenario_mapping.get(scenario)
        if factory:
            return factory()
        
        # 尝试使用 MultiModalScenario 枚举
        try:
            scenario_enum = MultiModalScenario.from_string(scenario)
            return MultiModalConfig(
                modalities=scenario_enum.recommended_modalities,
                fusion_method=scenario_enum.recommended_fusion,
                scenario=scenario
            )
        except Exception:
            pass
        
        # 默认返回标准配置
        return MultiModalPresets.standard_classification()
    
    @staticmethod
    def list_presets() -> List[str]:
        """列出所有可用的预设"""
        return [
            'standard_classification',
            'multimodal_alignment',
            'manufacturing_scenario',
            'finance_scenario',
            'medical_scenario',
            'retail_scenario',
            'autonomous_driving',
            'video_understanding',
            'distributed_large_scale',
        ]
    
    @staticmethod
    def get_preset(name: str) -> MultiModalConfig:
        """根据名称获取预设"""
        preset_mapping = {
            'standard_classification': MultiModalPresets.standard_classification,
            'multimodal_alignment': MultiModalPresets.multimodal_alignment,
            'manufacturing_scenario': MultiModalPresets.manufacturing_scenario,
            'finance_scenario': MultiModalPresets.finance_scenario,
            'medical_scenario': MultiModalPresets.medical_scenario,
            'retail_scenario': MultiModalPresets.retail_scenario,
            'autonomous_driving': MultiModalPresets.autonomous_driving,
            'video_understanding': MultiModalPresets.video_understanding,
            'distributed_large_scale': MultiModalPresets.distributed_large_scale,
        }
        
        factory = preset_mapping.get(name)
        if factory:
            return factory()
        
        raise ValueError(f"Unknown preset: {name}. Available: {list(preset_mapping.keys())}")


# ==================== 便捷函数 ====================

def get_layer_availability() -> Dict[str, bool]:
    """获取各层可用性"""
    return {
        'strategy_layer': STRATEGY_LAYER_AVAILABLE,
        'distributed_strategy': DISTRIBUTED_STRATEGY_AVAILABLE,
        'multimodal_strategy': MULTIMODAL_STRATEGY_AVAILABLE,
        'hardware_layer': HARDWARE_LAYER_AVAILABLE,
        'lib_multimodal': LIB_MULTIMODAL_AVAILABLE,
    }


def create_config_from_requirements(
    modalities: List[str],
    scenario: str = "standard",
    use_distributed: bool = False,
    **kwargs
) -> MultiModalConfig:
    """根据需求创建配置
    
    Args:
        modalities: 模态列表
        scenario: 场景类型
        use_distributed: 是否使用分布式
        **kwargs: 其他配置参数
    
    Returns:
        配置实例
    """
    # 尝试使用场景预设
    try:
        scenario_enum = MultiModalScenario.from_string(scenario)
        fusion_method = scenario_enum.recommended_fusion
    except Exception:
        fusion_method = "concat"
    
    # 基础配置
    config_kwargs = {
        'modalities': modalities,
        'scenario': scenario,
        'fusion_method': fusion_method,
        'use_distributed': use_distributed,
    }
    config_kwargs.update(kwargs)
    
    # 行业场景使用 IndustryMultiModalConfig
    industry_scenarios = ['manufacturing', 'finance', 'medical', 'retail']
    if scenario in industry_scenarios:
        return IndustryMultiModalConfig(industry_type=scenario, **config_kwargs)
    
    return MultiModalConfig(**config_kwargs)


def diagnose_config(config: MultiModalConfig) -> Dict[str, Any]:
    """诊断配置"""
    return config.diagnose()


def optimize_config_for_hardware(config: MultiModalConfig) -> MultiModalConfig:
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
                while estimated_mem > available_mem * 0.8 and optimized.batch_size > 1:
                    optimized.batch_size = max(1, optimized.batch_size // 2)
                    optimized.gradient_accumulation_steps *= 2
                    estimated_mem = optimized.estimate_total_memory_mb()
                    logger.info(f"Reduced batch_size to {optimized.batch_size} due to memory constraints")
            
            # 推荐精度
            if recommend_precision is not None:
                recommended = recommend_precision(optimized.device)
                if recommended in ['fp16', 'mixed']:
                    optimized.use_fp16 = True
                    logger.info("Enabled fp16 based on hardware recommendation")
                    
        except Exception as e:
            logger.warning(f"Hardware optimization failed: {e}")
    
    return optimized
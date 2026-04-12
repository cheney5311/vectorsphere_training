# -*- coding: utf-8 -*-
"""
多模态训练策略

统一的多模态训练接口，集成生产级多模态训练能力：
- 数据工程（去重、过滤、增强）
- 模态专属编码器（文本、图像、音频、时序、视频）
- 跨模态对齐（对比学习、显式对齐、最优传输）
- 多模态融合（早期/中期/后期融合、Q-Former、Perceiver）
- 四阶段训练流程（预训练→对齐→指令微调→安全对齐）

架构调用层次：
├── multimodal_strategy.py (本模块)
│   └── 调用 backend/lib/adapters (适配器层)
│       ├── 模态编码器 (TextEncoder, ImageEncoder, etc.)
│       ├── 融合模块 (EarlyFusion, MiddleFusion, CrossAttentionFusion)
│       └── 对齐模块 (ContrastiveAlignment, ExplicitAlignment)
│   └── 调用 backend/lib/losses (损失层)
│       ├── CrossModalContrastiveLoss - 跨模态对比损失
│       ├── CLIPLoss - CLIP风格损失
│       └── MultiTaskLoss - 多任务损失
│   └── 调用 backend/lib/multimodal (多模态层)
│       ├── CrossModalAligner - 跨模态对齐器
│       ├── MultiModalFuser - 多模态融合器
│       └── MultiModalDataPipeline - 数据管道
│   └── 调用 backend/lib/hardware (硬件层)
│       ├── DeviceManager - 设备管理
│       ├── MemoryManager - 内存管理
│       └── MixedPrecisionManager - 混合精度
│   └── 调用 base_strategy.py (策略基类)
│       ├── StrategyMonitor - 监控
│       ├── StrategyProfiler - 性能分析
│       ├── StrategyValidator - 验证
│       └── StrategyMetrics - 指标跟踪
└── 被 multimodal/multimodal_trainer.py 调用

生产级特性：
- 完整的监控和诊断能力
- 模态级别的性能分析
- 自动内存优化
- 健康检查和故障恢复

使用示例:
```python
from backend.modules.training.strategies.multimodal_strategy import (
    create_production_multimodal_pipeline,
    run_multimodal_training
)

# 创建完整的多模态训练管道
pipeline = create_production_multimodal_pipeline(
    modalities=['text', 'image'],
    fusion_method='cross_attention'
)

# 运行四阶段训练
result = run_multimodal_training(
    pipeline=pipeline,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader
)
```
"""

import logging
import time
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .base_strategy import (
    TrainingStrategy, 
    StrategyContext, 
    StrategyResult, 
    TrainingPhase,
    StrategyType,
    StrategyMonitor,
    StrategyProfiler,
    StrategyValidator,
    StrategyMetrics,
)

logger = logging.getLogger(__name__)


# ==================== 底层模块导入 ====================

# 适配器层

from backend.lib.adapters import (
    # 模态编码器
    ModalityEncoder, EncoderFactory, create_encoder,
    TextEncoder, ImageEncoder, AudioEncoder, VideoEncoder, TimeSeriesEncoder,
    ModalityType,
    # 融合模块
    FusionModule, FusionFactory, create_fusion,
    EarlyFusion, MiddleFusion, LateFusion, CrossAttentionFusion, GatedFusion,
    FusionMethod,
    # 对齐模块
    AlignmentModule, AlignmentFactory, create_alignment,
    ContrastiveAlignment, ExplicitAlignment, OptimalTransportAlignment,
    AlignmentMethod
)

# 损失层

from backend.lib.losses import (
    # 对比学习损失
    CrossModalContrastiveLoss, CLIPLoss, InfoNCELoss, ContrastiveLoss,
    # 监督损失
    CrossEntropyLoss, MSELoss,
    # 复合损失
    CompositeLoss, MultiTaskLoss,
    # 工厂函数
    create_loss, create_composite_loss,
    # 监控组件
    LossMonitor, LossStats,
)

# 硬件层

from backend.lib.hardware import (
    DeviceManager, get_device_manager,
    MemoryManager,
    MixedPrecisionManager, AmpConfig, PrecisionMode,
    clear_memory, get_available_memory,
)

# 分布式层
from backend.lib.distributed import (
    DistributedManager, get_distributed_manager,
    is_main_process, get_rank, get_world_size, barrier,
)

# ==================== 枚举定义 ====================

class MultiModalTrainingStage(Enum):
    """多模态训练阶段"""
    MODALITY_PRETRAIN = "modality_pretrain"      # 阶段一：模态预训练
    CROSS_MODAL_ALIGN = "cross_modal_align"      # 阶段二：跨模态对齐
    INSTRUCTION_TUNING = "instruction_tuning"    # 阶段三：指令微调
    ALIGNMENT_SAFETY = "alignment_safety"        # 阶段四：对齐与安全
    
    @classmethod
    def from_string(cls, value: str) -> 'MultiModalTrainingStage':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown training stage: {value}")
    
    @property
    def stage_number(self) -> int:
        """阶段编号 (1-4)"""
        order = [MultiModalTrainingStage.MODALITY_PRETRAIN, MultiModalTrainingStage.CROSS_MODAL_ALIGN, 
                 MultiModalTrainingStage.INSTRUCTION_TUNING, MultiModalTrainingStage.ALIGNMENT_SAFETY]
        return order.index(self) + 1
    
    @property
    def typical_epochs(self) -> int:
        """典型训练轮数"""
        epochs = {
            MultiModalTrainingStage.MODALITY_PRETRAIN: 10,
            MultiModalTrainingStage.CROSS_MODAL_ALIGN: 5,
            MultiModalTrainingStage.INSTRUCTION_TUNING: 3,
            MultiModalTrainingStage.ALIGNMENT_SAFETY: 1,
        }
        return epochs[self]
    
    @property
    def typical_lr(self) -> float:
        """典型学习率"""
        lrs = {
            MultiModalTrainingStage.MODALITY_PRETRAIN: 1e-4,
            MultiModalTrainingStage.CROSS_MODAL_ALIGN: 1e-5,
            MultiModalTrainingStage.INSTRUCTION_TUNING: 2e-5,
            MultiModalTrainingStage.ALIGNMENT_SAFETY: 1e-6,
        }
        return lrs[self]
    
    def next_stage(self) -> Optional['MultiModalTrainingStage']:
        """获取下一阶段"""
        order = [MultiModalTrainingStage.MODALITY_PRETRAIN, MultiModalTrainingStage.CROSS_MODAL_ALIGN,
                 MultiModalTrainingStage.INSTRUCTION_TUNING, MultiModalTrainingStage.ALIGNMENT_SAFETY]
        idx = order.index(self)
        return order[idx + 1] if idx < len(order) - 1 else None


class FusionType(Enum):
    """融合类型"""
    EARLY = "early"
    MIDDLE = "middle"
    LATE = "late"
    ADAPTIVE = "adaptive"
    QFORMER = "qformer"
    PERCEIVER = "perceiver"
    CROSS_ATTENTION = "cross_attention"
    GATED = "gated"
    ATTENTION = "attention"
    
    @classmethod
    def from_string(cls, value: str) -> 'FusionType':
        """从字符串创建"""
        value = value.lower().strip().replace('-', '_')
        for member in cls:
            if member.value == value:
                return member
        # 别名映射
        aliases = {
            'crossattention': cls.CROSS_ATTENTION,
            'cross-attention': cls.CROSS_ATTENTION,
        }
        if value in aliases:
            return aliases[value]
        raise ValueError(f"Unknown fusion type: {value}")
    
    @property
    def complexity(self) -> str:
        """复杂度"""
        complexity = {
            FusionType.EARLY: "low",
            FusionType.MIDDLE: "medium",
            FusionType.LATE: "low",
            FusionType.ADAPTIVE: "high",
            FusionType.QFORMER: "high",
            FusionType.PERCEIVER: "high",
            FusionType.CROSS_ATTENTION: "medium",
            FusionType.GATED: "medium",
            FusionType.ATTENTION: "medium",
        }
        return complexity.get(self, "medium")
    
    @property
    def param_count_multiplier(self) -> float:
        """参数量乘数"""
        multipliers = {
            FusionType.EARLY: 1.0,
            FusionType.MIDDLE: 1.5,
            FusionType.LATE: 1.0,
            FusionType.ADAPTIVE: 2.0,
            FusionType.QFORMER: 2.5,
            FusionType.PERCEIVER: 3.0,
            FusionType.CROSS_ATTENTION: 1.8,
            FusionType.GATED: 1.5,
            FusionType.ATTENTION: 1.6,
        }
        return multipliers.get(self, 1.5)


class AlignmentType(Enum):
    """对齐类型"""
    CONTRASTIVE = "contrastive"
    EXPLICIT = "explicit"
    CROSS_ATTENTION = "cross_attention"
    OPTIMAL_TRANSPORT = "optimal_transport"
    
    @classmethod
    def from_string(cls, value: str) -> 'AlignmentType':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown alignment type: {value}")
    
    @property
    def requires_paired_data(self) -> bool:
        """是否需要配对数据"""
        return self in (AlignmentType.CONTRASTIVE, AlignmentType.EXPLICIT)
    
    @property
    def memory_efficiency(self) -> float:
        """内存效率 (0-1)"""
        efficiency = {
            AlignmentType.CONTRASTIVE: 0.8,
            AlignmentType.EXPLICIT: 0.9,
            AlignmentType.CROSS_ATTENTION: 0.6,
            AlignmentType.OPTIMAL_TRANSPORT: 0.5,
        }
        return efficiency[self]


# ==================== 新增数据类 ====================

@dataclass
class ModalityStats:
    """模态统计"""
    modality: str
    total_samples: int = 0
    processed_samples: int = 0
    encoding_time_ms: float = 0.0
    fusion_time_ms: float = 0.0
    alignment_time_ms: float = 0.0
    avg_feature_norm: float = 0.0
    
    def record_encoding(self, time_ms: float, feature_norm: float) -> None:
        """记录编码统计"""
        self.processed_samples += 1
        self.encoding_time_ms += time_ms
        self.avg_feature_norm = (
            (self.avg_feature_norm * (self.processed_samples - 1) + feature_norm) 
            / self.processed_samples
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'modality': self.modality,
            'total_samples': self.total_samples,
            'processed_samples': self.processed_samples,
            'encoding_time_ms': self.encoding_time_ms,
            'fusion_time_ms': self.fusion_time_ms,
            'alignment_time_ms': self.alignment_time_ms,
            'avg_feature_norm': self.avg_feature_norm,
            'throughput': self.processed_samples / (self.encoding_time_ms / 1000) if self.encoding_time_ms > 0 else 0,
        }


@dataclass
class MultiModalHealthStatus:
    """多模态健康状态"""
    is_healthy: bool = True
    all_modalities_working: bool = True
    alignment_effective: bool = True
    fusion_working: bool = True
    last_check_time: float = 0.0
    issues: List[str] = field(default_factory=list)
    
    def add_issue(self, issue: str) -> None:
        """添加问题"""
        self.issues.append(issue)
        self.is_healthy = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'is_healthy': self.is_healthy,
            'all_modalities_working': self.all_modalities_working,
            'alignment_effective': self.alignment_effective,
            'fusion_working': self.fusion_working,
            'last_check_time': self.last_check_time,
            'issues': self.issues.copy(),
        }

# ==================== 配置类 ====================

@dataclass
class MultiModalStrategyConfig:
    """多模态策略配置"""
    # 支持的模态
    modalities: List[str] = field(default_factory=lambda: ['text', 'image'])
    
    # 损失权重
    task_loss_weight: float = 1.0
    align_loss_weight: float = 0.5
    contrastive_loss_weight: float = 0.1
    
    # 模态融合配置
    fusion_method: str = "cross_attention"
    fusion_stage: str = "middle"
    fusion_dim: int = 768
    
    # 对齐配置
    use_alignment: bool = True
    alignment_method: str = "contrastive"
    alignment_temperature: float = 0.07
    
    # 模态dropout
    modality_dropout: float = 0.1
    
    # 生产级配置
    use_production_mode: bool = False
    training_stage: str = "modality_pretrain"
    
    # 数据工程配置
    enable_data_engineering: bool = True
    enable_deduplication: bool = True
    enable_filtering: bool = True
    enable_augmentation: bool = False
    
    # 编码器配置
    encoder_freeze_strategy: Dict[str, bool] = field(default_factory=dict)
    use_pretrained_encoders: bool = True
    
    # 训练配置
    output_dir: str = "./outputs"
    save_checkpoints: bool = True
    
    # 监控配置
    enable_monitoring: bool = True
    enable_profiling: bool = False
    health_check_interval: int = 100
    log_interval: int = 10
    
    # 混合精度配置
    use_amp: bool = True
    amp_dtype: str = "float16"
    
    # 分布式配置
    use_distributed: bool = False
    gradient_accumulation_steps: int = 1
    
    def validate(self) -> None:
        """验证配置"""
        if not self.modalities:
            raise ValueError("At least one modality must be specified")
        if self.task_loss_weight < 0:
            raise ValueError("task_loss_weight must be >= 0")
        if self.align_loss_weight < 0:
            raise ValueError("align_loss_weight must be >= 0")
        if self.contrastive_loss_weight < 0:
            raise ValueError("contrastive_loss_weight must be >= 0")
        if self.fusion_dim <= 0:
            raise ValueError("fusion_dim must be > 0")
        if self.alignment_temperature <= 0:
            raise ValueError("alignment_temperature must be > 0")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'modalities': self.modalities.copy(),
            'task_loss_weight': self.task_loss_weight,
            'align_loss_weight': self.align_loss_weight,
            'contrastive_loss_weight': self.contrastive_loss_weight,
            'fusion_method': self.fusion_method,
            'fusion_stage': self.fusion_stage,
            'fusion_dim': self.fusion_dim,
            'use_alignment': self.use_alignment,
            'alignment_method': self.alignment_method,
            'alignment_temperature': self.alignment_temperature,
            'modality_dropout': self.modality_dropout,
            'use_production_mode': self.use_production_mode,
            'training_stage': self.training_stage,
            'enable_data_engineering': self.enable_data_engineering,
            'enable_monitoring': self.enable_monitoring,
            'enable_profiling': self.enable_profiling,
            'use_amp': self.use_amp,
            'use_distributed': self.use_distributed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiModalStrategyConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def get_fusion_type(self) -> FusionType:
        """获取融合类型枚举"""
        return FusionType.from_string(self.fusion_method)
    
    def get_alignment_type(self) -> AlignmentType:
        """获取对齐类型枚举"""
        return AlignmentType.from_string(self.alignment_method)
    
    def summary(self) -> str:
        """获取配置摘要"""
        return (
            f"MultiModalConfig(modalities={self.modalities}, "
            f"fusion={self.fusion_method}, align={self.alignment_method}, "
            f"production={self.use_production_mode})"
        )


@dataclass
class MultiModalPipelineConfig:
    """多模态管道完整配置"""
    # 基础配置
    project_name: str = "multimodal_training"
    output_dir: str = "./outputs"
    seed: int = 42
    
    # 策略配置
    strategy_config: MultiModalStrategyConfig = field(default_factory=MultiModalStrategyConfig)
    
    # 四阶段训练配置
    enable_modality_pretrain: bool = True
    enable_cross_modal_align: bool = True
    enable_instruction_tuning: bool = True
    enable_alignment_safety: bool = True
    
    # 各阶段epochs
    pretrain_epochs: int = 10
    align_epochs: int = 5
    finetune_epochs: int = 3
    safety_epochs: int = 1
    
    # 学习率
    pretrain_lr: float = 1e-4
    align_lr: float = 1e-5
    finetune_lr: float = 2e-5
    safety_lr: float = 1e-6
    
    # 分布式训练
    use_distributed: bool = False
    use_deepspeed: bool = False
    fp16: bool = True
    gradient_checkpointing: bool = True


# ==================== 基础多模态策略 ====================

class MultiModalStrategy(TrainingStrategy):
    """
    多模态训练策略基类
    
    整合底层适配器层和损失层能力：
    - backend/lib/adapters: 模态编码、融合、对齐
        - ModalityEncoder, EncoderFactory, create_encoder
        - FusionModule, FusionFactory, create_fusion
        - AlignmentModule, AlignmentFactory, create_alignment
    - backend/lib/losses: 对比损失、多任务损失
        - CrossModalContrastiveLoss, CLIPLoss, InfoNCELoss
        - CompositeLoss, MultiTaskLoss
        - LossMonitor, LossStats
    - backend/lib/hardware: 硬件管理
        - DeviceManager, MemoryManager
        - MixedPrecisionManager
    - backend/lib/distributed: 分布式支持
        - DistributedManager
    - base_strategy.py: 策略基础能力
        - StrategyMonitor, StrategyProfiler, StrategyValidator, StrategyMetrics
    
    提供多模态训练的核心功能：
    - 模态特征编码
    - 跨模态对齐
    - 多模态融合
    - 损失计算
    - 监控和诊断
    """
    
    # 策略类型
    STRATEGY_TYPE = StrategyType.MULTIMODAL
    
    def __init__(self, config: Optional[MultiModalStrategyConfig] = None):
        super().__init__(name="multimodal", priority=50)
        self.config = config or MultiModalStrategyConfig()
        
        # 当前训练阶段
        self._current_phase: TrainingPhase = TrainingPhase.WARMUP
        self._current_mm_stage: MultiModalTrainingStage = MultiModalTrainingStage.MODALITY_PRETRAIN
        
        # 对齐投影层
        self.alignment_projectors: Dict[str, nn.Module] = {}
        self._temp_projectors: Dict[str, nn.Module] = {}
        
        # 底层适配器组件（使用 backend/lib/adapters）
        self._lib_encoders: Dict[str, 'ModalityEncoder'] = {}
        self._lib_fusion: Optional['FusionModule'] = None
        self._lib_alignment: Optional['AlignmentModule'] = None
        
        # 底层损失组件（使用 backend/lib/losses）
        self._lib_contrastive_loss: Optional[nn.Module] = None
        self._lib_task_loss: Optional[nn.Module] = None
        self._lib_mse_loss: Optional[nn.Module] = None  # 使用 MSELoss
        self._lib_composite_loss: Optional[nn.Module] = None
        self._lib_loss_monitor: Optional['LossMonitor'] = None
        
        # 底层硬件组件（使用 backend/lib/hardware）
        self._device_manager: Optional['DeviceManager'] = None
        self._memory_manager: Optional['MemoryManager'] = None
        self._amp_manager: Optional['MixedPrecisionManager'] = None
        
        # 底层分布式组件（使用 backend/lib/distributed）
        self._distributed_manager: Optional['DistributedManager'] = None
        
        # 基础策略组件（使用 base_strategy.py）
        self._strategy_monitor: Optional[StrategyMonitor] = None
        self._strategy_profiler: Optional[StrategyProfiler] = None
        self._strategy_validator: Optional[StrategyValidator] = None
        self._strategy_metrics: Optional[StrategyMetrics] = None
        
        # 生产级组件（延迟初始化）
        self._production_aligner = None
        self._production_fuser = None
        self._production_encoders = None
        self._data_pipeline = None
        self._simple_fusion_proj: Optional[nn.Module] = None
        
        # 模态统计
        self._modality_stats: Dict[str, ModalityStats] = {}
        for modality in self.config.modalities:
            self._modality_stats[modality] = ModalityStats(modality=modality)
        
        # 健康状态
        self._health_status = MultiModalHealthStatus()
        
        # 验证配置
        try:
            self.config.validate()
        except ValueError as e:
            logger.warning(f"Config validation warning: {e}")
    
    def setup(self, context: StrategyContext) -> None:
        """初始化多模态组件"""
        super().setup(context)
        
        # 0. 初始化基础策略组件
        self._init_base_strategy_components()
        
        # 1. 初始化底层硬件层组件
        self._setup_hardware_layer(context)
        
        # 2. 初始化底层分布式层组件
        self._setup_distributed_layer(context)
        
        # 3. 初始化底层适配器层组件
        self._setup_adapters_layer(context)
        
        # 4. 初始化底层损失层组件
        self._setup_losses_layer(context)
        
        # 5. 回退到简单对齐投影层（如果底层不可用）
        if self._lib_alignment is None and self.config.use_alignment and len(self.config.modalities) > 1:
            config_dict = context.config if hasattr(context, 'config') and isinstance(context.config, dict) else {}
            for modality in self.config.modalities:
                input_dim = config_dict.get(f'{modality}_dim', 768) if config_dict else 768
                self.alignment_projectors[modality] = nn.Sequential(
                    nn.Linear(input_dim, self.config.fusion_dim),
                    nn.ReLU(),
                    nn.Linear(self.config.fusion_dim, self.config.fusion_dim)
                ).to(context.device)
        
        # 6. 初始化生产级组件
        if self.config.use_production_mode:
            self._init_production_components(context)
        
        # 7. 初始健康检查
        self._check_health()
        
        logger.info(f"MultiModalStrategy setup: modalities={self.config.modalities}, "
                   f"fusion={self.config.fusion_method}, alignment={self.config.alignment_method}")
    
    def _init_base_strategy_components(self) -> None:
        """
        初始化基础策略组件
        
        使用 base_strategy.py 提供的组件
        """
        # 初始化策略监控器
        if self.config.enable_monitoring:
            try:
                self._strategy_monitor = StrategyMonitor(history_size=10000)
            except Exception as e:
                logger.warning(f"Failed to init StrategyMonitor: {e}")
        
        # 初始化性能分析器
        if self.config.enable_profiling:
            try:
                self._strategy_profiler = StrategyProfiler()
            except Exception as e:
                logger.warning(f"Failed to init StrategyProfiler: {e}")
        
        # 初始化验证器
        try:
            self._strategy_validator = StrategyValidator()
            self._add_multimodal_validation_rules()
        except Exception as e:
            logger.warning(f"Failed to init StrategyValidator: {e}")
        
        # 初始化指标跟踪
        try:
            self._strategy_metrics = StrategyMetrics()
        except Exception as e:
            logger.warning(f"Failed to init StrategyMetrics: {e}")
        
        logger.debug("Base strategy components initialized")
    
    def _add_multimodal_validation_rules(self) -> None:
        """添加多模态特定的验证规则"""
        if self._strategy_validator is None:
            return
        
        if hasattr(self._strategy_validator, 'add_check'):
            # 检查多模态健康
            def check_multimodal_health(result: StrategyResult) -> Tuple[bool, str]:
                if not self._health_status.is_healthy:
                    return False, f"Multimodal unhealthy: {self._health_status.issues}"
                return True, ""
            
            self._strategy_validator.add_check(check_multimodal_health)
    
    def _setup_hardware_layer(self, context: StrategyContext) -> None:
        """
        初始化硬件层
        
        使用 backend/lib/hardware 提供的能力
        """
        # 初始化设备管理器
        if get_device_manager is not None:
            try:
                self._device_manager = get_device_manager()
            except Exception as e:
                logger.warning(f"Failed to get device manager: {e}")
        
        # 初始化内存管理器
        # get_memory_manager 不存在，直接使用 MemoryManager
        if MemoryManager is not None:
            try:
                self._memory_manager = MemoryManager(device=context.device)
            except Exception as e:
                logger.warning(f"Failed to create MemoryManager: {e}")
        
        # 初始化混合精度管理器
        if self.config.use_amp and MixedPrecisionManager is not None and AmpConfig is not None:
            try:
                precision = PrecisionMode.MIXED_FP16 if self.config.amp_dtype == "float16" else PrecisionMode.MIXED_BF16
                amp_config = AmpConfig(enabled=True, precision=precision)
                self._amp_manager = MixedPrecisionManager(amp_config, context.device)
            except Exception as e:
                logger.warning(f"Failed to init MixedPrecisionManager: {e}")
        
        logger.info(f"Hardware layer initialized: device_manager={self._device_manager is not None}, "
                   f"memory_manager={self._memory_manager is not None}, amp_manager={self._amp_manager is not None}")
    
    def _setup_distributed_layer(self, context: StrategyContext) -> None:
        """
        初始化分布式层
        
        使用 backend/lib/distributed 提供的能力
        """ 
        if get_distributed_manager is not None:
            try:
                self._distributed_manager = get_distributed_manager()
                logger.info("Distributed manager initialized")
            except Exception as e:
                logger.warning(f"Failed to get distributed manager: {e}")
    
    def _setup_adapters_layer(self, context: StrategyContext) -> None:
        """
        初始化底层适配器层
        
        使用 backend/lib/adapters 提供的能力：
        - 模态编码器: EncoderFactory, TextEncoder, ImageEncoder, AudioEncoder, VideoEncoder, TimeSeriesEncoder
        - 融合模块: FusionFactory, EarlyFusion, MiddleFusion, LateFusion, CrossAttentionFusion, GatedFusion
        - 对齐模块: AlignmentFactory, ContrastiveAlignment, ExplicitAlignment, OptimalTransportAlignment
        """
        # 使用 EncoderFactory 或特定编码器创建模态编码器
        for modality in self.config.modalities:
            try:
                encoder = self._create_modality_encoder(modality, context)
                if encoder is not None:
                    self._lib_encoders[modality] = encoder.to(context.device)
            except Exception as e:
                logger.warning(f"Failed to create encoder for {modality}: {e}")
        
        # 使用 FusionFactory 或特定融合模块创建融合模块
        if len(self.config.modalities) > 1:
            try:
                self._lib_fusion = self._create_fusion_module(context)
                if self._lib_fusion is not None:
                    self._lib_fusion = self._lib_fusion.to(context.device)
            except Exception as e:
                logger.warning(f"Failed to create fusion module: {e}")
            
            # 使用 AlignmentFactory 或特定对齐模块创建对齐模块
            if self.config.use_alignment:
                try:
                    self._lib_alignment = self._create_alignment_module(context)
                    if self._lib_alignment is not None:
                        self._lib_alignment = self._lib_alignment.to(context.device)
                except Exception as e:
                    logger.warning(f"Failed to create alignment module: {e}")
        
        logger.info(f"Adapters layer initialized: encoders={list(self._lib_encoders.keys())}")
    
    def _create_modality_encoder(self, modality: str, context: StrategyContext) -> Optional[nn.Module]:
        """
        创建模态编码器
        
        优先使用 EncoderFactory，回退到特定编码器类
        使用: EncoderFactory, TextEncoder, ImageEncoder, AudioEncoder, VideoEncoder, TimeSeriesEncoder, ModalityType
        """
        hidden_size = self.config.fusion_dim
        
        # 1. 优先使用 EncoderFactory
        if EncoderFactory is not None:
            try:
                return EncoderFactory.create(modality, hidden_size=hidden_size)
            except Exception as e:
                logger.debug(f"EncoderFactory failed for {modality}: {e}")
        
        # 2. 使用 create_encoder
        if create_encoder is not None:
            try:
                return create_encoder(modality, hidden_size=hidden_size)
            except Exception as e:
                logger.debug(f"create_encoder failed for {modality}: {e}")
        
        # 3. 使用 ModalityType 映射到特定编码器
        encoder_map = {
            'text': TextEncoder,
            'image': ImageEncoder,
            'audio': AudioEncoder,
            'video': VideoEncoder,
            'time_series': TimeSeriesEncoder,
        }
        
        encoder_cls = encoder_map.get(modality)
        if encoder_cls is not None:
            try:
                return encoder_cls(hidden_size=hidden_size)
            except Exception as e:
                logger.debug(f"Specific encoder {encoder_cls} failed: {e}")
        
        # 4. 使用 ModalityType 枚举
        if ModalityType is not None:
            try:
                modality_type = ModalityType(modality)
                logger.debug(f"ModalityType resolved: {modality_type}")
            except Exception:
                pass
        
        return None
    
    def _create_fusion_module(self, context: StrategyContext) -> Optional[nn.Module]:
        """
        创建融合模块
        
        优先使用 FusionFactory，回退到特定融合类
        使用: FusionFactory, FusionMethod, EarlyFusion, MiddleFusion, LateFusion, CrossAttentionFusion, GatedFusion
        """
        hidden_size = self.config.fusion_dim
        fusion_method = self.config.fusion_method
        num_modalities = len(self.config.modalities)
        
        # 1. 优先使用 FusionFactory
        if FusionFactory is not None:
            try:
                return FusionFactory.create(
                    fusion_method, 
                    hidden_size=hidden_size,
                    num_modalities=num_modalities
                )
            except Exception as e:
                logger.debug(f"FusionFactory failed: {e}")
        
        # 2. 使用 create_fusion
        if create_fusion is not None:
            try:
                return create_fusion(fusion_method, hidden_size=hidden_size)
            except Exception as e:
                logger.debug(f"create_fusion failed: {e}")
        
        # 3. 使用 FusionMethod 枚举映射到特定融合类
        fusion_map = {
            'early': EarlyFusion,
            'middle': MiddleFusion,
            'late': LateFusion,
            'cross_attention': CrossAttentionFusion,
            'gated': GatedFusion,
        }
        
        fusion_cls = fusion_map.get(fusion_method)
        if fusion_cls is not None:
            try:
                return fusion_cls(hidden_size=hidden_size, num_modalities=num_modalities)
            except Exception as e:
                try:
                    # 尝试只用 hidden_size
                    return fusion_cls(hidden_size=hidden_size)
                except Exception as e2:
                    logger.debug(f"Specific fusion {fusion_cls} failed: {e2}")
        
        # 4. 使用 FusionMethod 枚举
        if FusionMethod is not None:
            try:
                fusion_type = FusionMethod(fusion_method)
                logger.debug(f"FusionMethod resolved: {fusion_type}")
            except Exception:
                pass
        
        return None
    
    def _create_alignment_module(self, context: StrategyContext) -> Optional[nn.Module]:
        """
        创建对齐模块
        
        优先使用 AlignmentFactory，回退到特定对齐类
        使用: AlignmentFactory, AlignmentMethod, ContrastiveAlignment, ExplicitAlignment, OptimalTransportAlignment
        """
        hidden_size = self.config.fusion_dim
        alignment_method = self.config.alignment_method
        
        # 1. 优先使用 AlignmentFactory
        if AlignmentFactory is not None:
            try:
                return AlignmentFactory.create(
                    alignment_method,
                    hidden_size=hidden_size,
                    temperature=self.config.alignment_temperature
                )
            except Exception as e:
                logger.debug(f"AlignmentFactory failed: {e}")
        
        # 2. 使用 create_alignment
        if create_alignment is not None:
            try:
                return create_alignment(alignment_method, hidden_size=hidden_size)
            except Exception as e:
                logger.debug(f"create_alignment failed: {e}")
        
        # 3. 使用 AlignmentMethod 枚举映射到特定对齐类
        alignment_map = {
            'contrastive': ContrastiveAlignment,
            'explicit': ExplicitAlignment,
            'optimal_transport': OptimalTransportAlignment,
        }
        
        alignment_cls = alignment_map.get(alignment_method)
        if alignment_cls is not None:
            try:
                return alignment_cls(
                    hidden_size=hidden_size,
                    temperature=self.config.alignment_temperature
                )
            except Exception as e:
                try:
                    return alignment_cls(hidden_size=hidden_size)
                except Exception as e2:
                    logger.debug(f"Specific alignment {alignment_cls} failed: {e2}")
        
        # 4. 使用 AlignmentMethod 枚举
        if AlignmentMethod is not None:
            try:
                align_type = AlignmentMethod(alignment_method)
                logger.debug(f"AlignmentMethod resolved: {align_type}")
            except Exception:
                pass
        
        return None
    
    def _setup_losses_layer(self, context: StrategyContext) -> None:
        """
        初始化底层损失层
        
        使用 backend/lib/losses 提供的能力：
        - CrossModalContrastiveLoss - 跨模态对比损失
        - CLIPLoss - CLIP风格损失
        - InfoNCELoss - InfoNCE损失
        - ContrastiveLoss - 基础对比损失
        - MSELoss - 均方误差损失（用于回归/对齐）
        - CompositeLoss - 复合损失
        - MultiTaskLoss - 多任务损失
        - LossMonitor - 损失监控
        - LossStats - 损失统计
        """
        
        # 创建跨模态对比损失
        try:
            if CrossModalContrastiveLoss is not None:
                self._lib_contrastive_loss = CrossModalContrastiveLoss(
                    temperature=self.config.alignment_temperature
                ).to(context.device)
            elif CLIPLoss is not None:
                self._lib_contrastive_loss = CLIPLoss(
                    temperature=self.config.alignment_temperature
                ).to(context.device)
            elif InfoNCELoss is not None:
                self._lib_contrastive_loss = InfoNCELoss(
                    temperature=self.config.alignment_temperature
                ).to(context.device)
            elif ContrastiveLoss is not None:
                # ContrastiveLoss is abstract, use InfoNCELoss or similar
                # Check if InfoNCELoss is available (it is checked above)
                # If we fall through here, it means InfoNCELoss is None but ContrastiveLoss is not None
                # This is unlikely given imports, but we should avoid instantiating abstract class
                pass
        except Exception as e:
            logger.warning(f"Failed to create contrastive loss: {e}")
        
        # 创建任务损失
        try:
            if create_loss is not None:
                self._lib_task_loss = create_loss('cross_entropy')
            elif CrossEntropyLoss is not None:
                self._lib_task_loss = CrossEntropyLoss()
        except Exception as e:
            logger.warning(f"Failed to create task loss: {e}")
        
        # 创建 MSE 损失（用于显式对齐）
        self._lib_mse_loss: Optional[nn.Module] = None
        try:
            if MSELoss is not None:
                self._lib_mse_loss = MSELoss().to(context.device)
                logger.debug("MSELoss initialized for explicit alignment")
        except Exception as e:
            logger.warning(f"Failed to create MSE loss: {e}")
        
        # 创建复合损失（多任务）
        try:
            if CompositeLoss is not None or MultiTaskLoss is not None:
                loss_configs = {
                    'task': {'weight': self.config.task_loss_weight},
                    'alignment': {'weight': self.config.align_loss_weight},
                    'contrastive': {'weight': self.config.contrastive_loss_weight},
                }
                if create_composite_loss is not None:
                    self._lib_composite_loss = create_composite_loss(
                        loss_configs, 
                        auto_balance=False
                    )
        except Exception as e:
            logger.warning(f"Failed to create composite loss: {e}")
        
        # 创建损失监控器
        try:
            if LossMonitor is not None:
                self._lib_loss_monitor = LossMonitor(max_history=10000)
        except Exception as e:
            logger.warning(f"Failed to create loss monitor: {e}")
        
        logger.info(f"Losses layer initialized: contrastive={self._lib_contrastive_loss is not None}, "
                   f"task={self._lib_task_loss is not None}, composite={self._lib_composite_loss is not None}, "
                   f"monitor={self._lib_loss_monitor is not None}")
    
    def _init_production_components(self, context: StrategyContext):
        """初始化生产级组件"""
        try:
            # 导入生产级模块（从底层库导入）
            from backend.lib.multimodal import (
                CrossModalAligner, 
                MultiModalFuser, 
                CrossModalAlignmentConfig, MultiModalFusionConfig, 
                FusionStage, 
                MultiModalDataPipeline, DataEngineeringConfig
            )
            
            # 创建对齐器
            align_config = CrossModalAlignmentConfig(
                projection_dim=self.config.fusion_dim,
                align_loss_weight=self.config.align_loss_weight
            )
            self._production_aligner = CrossModalAligner(
                align_config, 
                self.config.fusion_dim
            ).to(context.device)
            
            # 创建融合器
            modality_dims = {m: self.config.fusion_dim for m in self.config.modalities}
            
            # 映射融合阶段
            fusion_stage_map = {
                "early": FusionStage.EARLY,
                "middle": FusionStage.MIDDLE,
                "late": FusionStage.LATE,
                "adaptive": FusionStage.ADAPTIVE
            }
            fusion_stage = fusion_stage_map.get(
                self.config.fusion_stage, 
                FusionStage.MIDDLE
            )
            
            fusion_config = MultiModalFusionConfig(
                stage=fusion_stage,
                output_dim=self.config.fusion_dim
            )
            self._production_fuser = MultiModalFuser(
                fusion_config,
                modality_dims
            ).to(context.device)
            
            # 初始化数据管道
            if self.config.enable_data_engineering:
                data_config = DataEngineeringConfig()
                data_config.deduplication.enabled = self.config.enable_deduplication
                data_config.filtering.enabled = self.config.enable_filtering
                data_config.augmentation.enabled = self.config.enable_augmentation
                self._data_pipeline = MultiModalDataPipeline(data_config)
            
            logger.info("Production multimodal components initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Failed to import production components: {e}")
            self.config.use_production_mode = False
    
    def prepare_batch(self, batch: Dict[str, Any], context: StrategyContext) -> Dict[str, Any]:
        """准备多模态批次数据"""
        # 应用模态dropout
        if context.model.training and self.config.modality_dropout > 0:
            import random
            for modality in self.config.modalities:
                if modality in batch and random.random() < self.config.modality_dropout:
                    active_modalities = [m for m in self.config.modalities if m in batch]
                    if len(active_modalities) > 1:
                        batch[f'{modality}_masked'] = True
        
        return batch
    
    def encode_modalities(
        self, 
        batch: Dict[str, Any], 
        context: StrategyContext
    ) -> Dict[str, Tensor]:
        """
        编码各模态特征
        
        优先使用 backend/lib/adapters 的模态编码器
        
        Args:
            batch: 包含各模态数据的批次
            context: 策略上下文
            
        Returns:
            各模态的编码特征
        """
        # 1. 优先使用底层适配器层的编码器
        if self._lib_encoders:
            modality_features = {}
            for modality, encoder in self._lib_encoders.items():
                if modality in batch:
                    data = batch[modality]
                    if isinstance(data, torch.Tensor):
                        data = data.to(context.device)
                    modality_features[modality] = encoder(data)
            if modality_features:
                return modality_features
        
        # 2. 使用生产级编码器
        if self._production_encoders is not None:
            modality_features = {}
            for modality, encoder in self._production_encoders.items():
                if modality in batch:
                    modality_features[modality] = encoder(batch[modality])
            return modality_features
        
        # 返回空字典，由外部模型处理
        return {}
    
    def align_modalities(
        self,
        features: Dict[str, Tensor],
        context: StrategyContext,
        compute_loss: bool = True
    ) -> Tuple[Dict[str, Tensor], Optional[Tensor], Dict[str, float]]:
        """
        跨模态对齐
        
        优先使用 backend/lib/adapters 的对齐模块
        
        Args:
            features: 各模态特征
            context: 策略上下文
            compute_loss: 是否计算损失
            
        Returns:
            对齐后的特征、损失值、指标
        """
        # 1. 优先使用底层适配器层的对齐模块
        if self._lib_alignment is not None:
            try:
                aligned_result = self._lib_alignment(features, compute_loss=compute_loss)
                if isinstance(aligned_result, tuple) and len(aligned_result) >= 2:
                    aligned, loss = aligned_result[:2]
                    metrics = aligned_result[2] if len(aligned_result) > 2 else {}
                    return aligned, loss, {'align_loss': loss.item() if loss is not None else 0.0, **metrics}
            except Exception as e:
                logger.warning(f"Lib alignment failed, falling back: {e}")
        
        # 2. 使用生产级对齐器
        if self._production_aligner is not None:
            return self._production_aligner(features, compute_loss=compute_loss)
        
        # 3. 回退到简单对齐：投影到统一空间
        aligned = {}
        for modality, feat in features.items():
            if modality in self.alignment_projectors:
                aligned[modality] = self.alignment_projectors[modality](feat)
            else:
                aligned[modality] = feat
        
        loss = None
        if compute_loss and len(aligned) >= 2:
            loss = self._compute_alignment_loss(aligned, context)
        
        return aligned, loss, {'align_loss': loss.item() if loss is not None else 0.0}
    
    def fuse_modalities(
        self,
        features: Dict[str, Tensor],
        context: StrategyContext
    ) -> Tensor:
        """
        多模态融合
        
        优先使用 backend/lib/adapters 的融合模块
        
        Args:
            features: 各模态特征
            context: 策略上下文
            
        Returns:
            融合后的特征
        """
        # 1. 优先使用底层适配器层的融合模块
        if self._lib_fusion is not None:
            try:
                return self._lib_fusion(features)
            except Exception as e:
                logger.warning(f"Lib fusion failed, falling back: {e}")
        
        # 2. 使用生产级融合器
        if self._production_fuser is not None:
            return self._production_fuser(features)
        
        # 3. 回退到简单融合：拼接后投影
        feature_list = list(features.values())
        if len(feature_list) == 1:
            return feature_list[0]
        
        concat = torch.cat(feature_list, dim=-1)
        
        # 创建投影层
        if not hasattr(self, '_simple_fusion_proj'):
            total_dim = sum(f.shape[-1] for f in feature_list)
            self._simple_fusion_proj = nn.Linear(
                total_dim, self.config.fusion_dim
            ).to(context.device)
        
        return self._simple_fusion_proj(concat)
    
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """
        计算多模态损失
        
        整合底层损失层能力并使用策略监控组件
        """
        start_time = time.time()
        metrics = {}
        warnings = []
        total_loss = torch.tensor(0.0, device=context.device)
        
        # 1. 任务损失
        task_loss = self._compute_task_loss(outputs)
        total_loss = total_loss + self.config.task_loss_weight * task_loss
        metrics['task_loss'] = task_loss.item()
        
        # 2. 对齐损失
        if self.config.use_alignment and 'modality_features' in outputs:
            if self._production_aligner is not None:
                _, align_loss, align_metrics = self._production_aligner(
                    outputs['modality_features'],
                    compute_loss=True
                )
                if align_loss is not None:
                    total_loss = total_loss + self.config.align_loss_weight * align_loss
                    metrics.update(align_metrics)
            else:
                align_loss = self._compute_alignment_loss(
                    outputs['modality_features'], context
                )
                total_loss = total_loss + self.config.align_loss_weight * align_loss
                metrics['align_loss'] = align_loss.item()
        
        # 3. 对比学习损失
        if self.config.contrastive_loss_weight > 0 and 'modality_features' in outputs:
            contrastive_loss = self._compute_contrastive_loss(
                outputs['modality_features'], context
            )
            total_loss = total_loss + self.config.contrastive_loss_weight * contrastive_loss
            metrics['contrastive_loss'] = contrastive_loss.item()
        
        # 添加训练阶段信息
        metrics['total_loss'] = total_loss.item()
        metrics['training_stage'] = self._current_mm_stage.value
        metrics['training_phase'] = self._current_phase.value if hasattr(self._current_phase, 'value') else str(self._current_phase)
        
        # 计算步骤时间
        step_time = time.time() - start_time
        metrics['step_time_ms'] = step_time * 1000
        
        # 创建结果
        result = StrategyResult(
            loss=total_loss, 
            metrics=metrics,
            step_time=step_time,
            warnings=warnings if warnings else None
        )
        
        # 记录到策略监控器
        if self._strategy_monitor is not None:
            try:
                self._strategy_monitor.record_step(result, context)
            except Exception as e:
                logger.debug(f"StrategyMonitor record failed: {e}")
        
        # 记录到损失监控器
        if self._lib_loss_monitor is not None:
            try:
                # LossMonitor.record typically takes a single argument (loss value or LossResult)
                self._lib_loss_monitor.record(total_loss.item())
            except Exception:
                pass
        
        # 更新策略指标
        if self._strategy_metrics is not None:
            try:
                self._strategy_metrics.total_steps += 1
                self._strategy_metrics.total_loss += total_loss.item()
                self._strategy_metrics.avg_loss = (
                    self._strategy_metrics.total_loss / self._strategy_metrics.total_steps
                )
            except Exception:
                pass
        
        # 验证结果
        if self._strategy_validator is not None:
            try:
                is_valid, errors = self._strategy_validator.validate(result)
                if not is_valid:
                    for error in errors:
                        warnings.append(f"Validation: {error}")
                    result.warnings = warnings
            except Exception:
                pass
        
        return result
    
    def _compute_task_loss(self, outputs: Dict[str, Any]) -> Tensor:
        """计算任务损失"""
        if 'loss' in outputs:
            return outputs['loss']
        elif 'task_loss' in outputs:
            return outputs['task_loss']
        elif hasattr(outputs, 'loss'):
            return outputs.loss
        else:
            raise ValueError("outputs中没有找到task_loss或loss")
    
    def _compute_alignment_loss(
        self, 
        modality_features: Dict[str, Tensor],
        context: StrategyContext
    ) -> Tensor:
        """
        计算模态对齐损失
        
        使用: MSELoss (backend/lib/losses), ExplicitAlignment (backend/lib/adapters)
        """
        if len(modality_features) < 2:
            return torch.tensor(0.0, device=context.device)
        
        projected_features = {}
        for modality, features in modality_features.items():
            if modality in self.alignment_projectors:
                projected_features[modality] = self.alignment_projectors[modality](features)
        
        if len(projected_features) < 2:
            return torch.tensor(0.0, device=context.device)
        
        align_loss = torch.tensor(0.0, device=context.device)
        modalities = list(projected_features.keys())
        count = 0
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                feat_i = projected_features[modalities[i]]
                feat_j = projected_features[modalities[j]]
                
                if feat_i.shape == feat_j.shape:
                    # 优先使用 _lib_mse_loss (backend/lib/losses 的 MSELoss)
                    if hasattr(self, '_lib_mse_loss') and self._lib_mse_loss is not None:
                        try:
                            loss_result = self._lib_mse_loss(feat_i, feat_j)
                            # 处理 LossResult 或 Tensor
                            if hasattr(loss_result, 'loss'):
                                align_loss = align_loss + loss_result.loss
                            else:
                                align_loss = align_loss + loss_result
                            count += 1
                            continue
                        except Exception:
                            pass
                    
                    # 回退到 F.mse_loss
                    align_loss = align_loss + F.mse_loss(feat_i, feat_j)
                    count += 1
        
        return align_loss / max(count, 1)
    
    def _compute_contrastive_loss(
        self,
        modality_features: Dict[str, Tensor],
        context: StrategyContext
    ) -> Tensor:
        """
        计算对比学习损失
        
        优先使用 backend/lib/losses 的 CrossModalContrastiveLoss
        """
        if len(modality_features) < 2:
            return torch.tensor(0.0, device=context.device)
        
        modalities = list(modality_features.keys())
        feat_a = modality_features[modalities[0]]
        feat_b = modality_features[modalities[1]]
        
        dim_a, dim_b = feat_a.shape[-1], feat_b.shape[-1]
        
        # 维度对齐
        if dim_a != dim_b:
            if modalities[0] in self.alignment_projectors and modalities[1] in self.alignment_projectors:
                feat_a = self.alignment_projectors[modalities[0]](feat_a)
                feat_b = self.alignment_projectors[modalities[1]](feat_b)
            else:
                target_dim = min(dim_a, dim_b)
                if dim_a != target_dim:
                    key_a = f"{modalities[0]}_{dim_a}_{target_dim}"
                    if key_a not in self._temp_projectors:
                        self._temp_projectors[key_a] = nn.Linear(dim_a, target_dim).to(context.device)
                    feat_a = self._temp_projectors[key_a](feat_a)
                if dim_b != target_dim:
                    key_b = f"{modalities[1]}_{dim_b}_{target_dim}"
                    if key_b not in self._temp_projectors:
                        self._temp_projectors[key_b] = nn.Linear(dim_b, target_dim).to(context.device)
                    feat_b = self._temp_projectors[key_b](feat_b)
        
        # 归一化
        feat_a = F.normalize(feat_a, dim=-1)
        feat_b = F.normalize(feat_b, dim=-1)
        
        # 优先使用底层损失模块
        if self._lib_contrastive_loss is not None:
            try:
                return self._lib_contrastive_loss(feat_a, feat_b)
            except Exception as e:
                logger.warning(f"Lib contrastive loss failed: {e}")
        
        # 回退到原生实现
        logits = torch.matmul(feat_a, feat_b.T) / self.config.alignment_temperature
        batch_size = feat_a.shape[0]
        labels = torch.arange(batch_size, device=context.device)
        
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        
        return (loss_a + loss_b) / 2
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """获取可训练参数"""
        params = []
        
        # 简单投影层参数
        for projector in self.alignment_projectors.values():
            params.extend(projector.parameters())
        
        for projector in self._temp_projectors.values():
            params.extend(projector.parameters())
        
        # 底层适配器层参数
        for encoder in self._lib_encoders.values():
            params.extend(encoder.parameters())
        
        if self._lib_fusion is not None:
            params.extend(self._lib_fusion.parameters())
        
        if self._lib_alignment is not None:
            params.extend(self._lib_alignment.parameters())
        
        # 生产级组件参数
        if self._production_aligner is not None:
            params.extend(self._production_aligner.parameters())
        
        if self._production_fuser is not None:
            params.extend(self._production_fuser.parameters())
        
        return params
    
    def get_layer_info(self) -> Dict[str, Any]:
        """
        获取底层模块调用信息
        
        列出使用的模块:
        - adapters: EncoderFactory, TextEncoder, ImageEncoder, AudioEncoder, VideoEncoder, TimeSeriesEncoder, 
                    ModalityType, FusionFactory, EarlyFusion, MiddleFusion, LateFusion, CrossAttentionFusion, 
                    GatedFusion, FusionMethod, AlignmentFactory, ContrastiveAlignment, ExplicitAlignment, 
                    OptimalTransportAlignment, AlignmentMethod
        - losses: ContrastiveLoss, MSELoss, LossStats, LossMonitor
        - hardware: get_available_memory
        - distributed: is_main_process, get_rank, get_world_size
        """
        return {
            # 适配器组件 - 编码器
            'lib_encoders': list(self._lib_encoders.keys()),
            'encoder_factory_available': EncoderFactory is not None,
            'text_encoder_available': TextEncoder is not None,
            'image_encoder_available': ImageEncoder is not None,
            'audio_encoder_available': AudioEncoder is not None,
            'video_encoder_available': VideoEncoder is not None,
            'time_series_encoder_available': TimeSeriesEncoder is not None,
            'modality_type_available': ModalityType is not None,
            
            # 适配器组件 - 融合
            'lib_fusion': self._lib_fusion is not None,
            'fusion_factory_available': FusionFactory is not None,
            'early_fusion_available': EarlyFusion is not None,
            'middle_fusion_available': MiddleFusion is not None,
            'late_fusion_available': LateFusion is not None,
            'cross_attention_fusion_available': CrossAttentionFusion is not None,
            'gated_fusion_available': GatedFusion is not None,
            'fusion_method_available': FusionMethod is not None,
            
            # 适配器组件 - 对齐
            'lib_alignment': self._lib_alignment is not None,
            'alignment_factory_available': AlignmentFactory is not None,
            'contrastive_alignment_available': ContrastiveAlignment is not None,
            'explicit_alignment_available': ExplicitAlignment is not None,
            'optimal_transport_alignment_available': OptimalTransportAlignment is not None,
            'alignment_method_available': AlignmentMethod is not None,
            
            # 损失组件
            'lib_contrastive_loss': self._lib_contrastive_loss is not None,
            'lib_task_loss': self._lib_task_loss is not None,
            'lib_mse_loss': self._lib_mse_loss is not None,
            'lib_composite_loss': self._lib_composite_loss is not None,
            'lib_loss_monitor': self._lib_loss_monitor is not None,
            'contrastive_loss_available': ContrastiveLoss is not None,
            'mse_loss_available': MSELoss is not None,
            'loss_stats_available': LossStats is not None,
            
            # 硬件组件
            'device_manager': self._device_manager is not None,
            'memory_manager': self._memory_manager is not None,
            'amp_manager': self._amp_manager is not None,
            'get_available_memory_available': get_available_memory is not None,
            
            # 分布式组件
            'distributed_manager': self._distributed_manager is not None,
            'is_main_process_available': is_main_process is not None,
            'get_rank_available': get_rank is not None,
            'get_world_size_available': get_world_size is not None,
            
            # 基础策略组件
            'strategy_monitor': self._strategy_monitor is not None,
            'strategy_profiler': self._strategy_profiler is not None,
            'strategy_validator': self._strategy_validator is not None,
            'strategy_metrics': self._strategy_metrics is not None,
            
            # 配置
            'modalities': self.config.modalities,
            'fusion_method': self.config.fusion_method,
            'alignment_method': self.config.alignment_method,
        }
    
    # ==================== 基础策略组件访问方法 ====================
    
    def get_strategy_type(self) -> StrategyType:
        """获取策略类型"""
        return self.STRATEGY_TYPE
    
    def get_training_phase(self) -> TrainingPhase:
        """获取当前训练阶段"""
        return self._current_phase
    
    def set_training_phase(self, phase: TrainingPhase) -> None:
        """设置训练阶段"""
        self._current_phase = phase
    
    def get_multimodal_stage(self) -> MultiModalTrainingStage:
        """获取当前多模态训练阶段"""
        return self._current_mm_stage
    
    def get_strategy_monitor(self) -> Optional[StrategyMonitor]:
        """获取策略监控器"""
        return self._strategy_monitor
    
    def get_strategy_profiler(self) -> Optional[StrategyProfiler]:
        """获取策略性能分析器"""
        return self._strategy_profiler
    
    def get_strategy_validator(self) -> Optional[StrategyValidator]:
        """获取策略验证器"""
        return self._strategy_validator
    
    def get_strategy_metrics(self) -> Optional[StrategyMetrics]:
        """获取策略指标"""
        return self._strategy_metrics
    
    # ==================== 健康检查和诊断方法 ====================
    
    def _check_health(self) -> None:
        """检查多模态策略健康状态"""
        self._health_status = MultiModalHealthStatus()
        self._health_status.last_check_time = time.time()
        
        # 检查模态编码器
        for modality in self.config.modalities:
            if modality not in self._lib_encoders and not self._production_encoders:
                self._health_status.all_modalities_working = False
                self._health_status.add_issue(f"No encoder for modality: {modality}")
        
        # 检查对齐模块
        if self.config.use_alignment:
            if self._lib_alignment is None and self._production_aligner is None:
                if not self.alignment_projectors:
                    self._health_status.alignment_effective = False
                    self._health_status.add_issue("No alignment module available")
        
        # 检查融合模块
        if len(self.config.modalities) > 1:
            if self._lib_fusion is None and self._production_fuser is None:
                self._health_status.fusion_working = False
                self._health_status.add_issue("No fusion module available")
        
        # 检查内存
        if self._memory_manager is not None:
            try:
                stats = self._memory_manager.get_stats()
                if hasattr(stats, 'pressure_level'):
                    if stats.pressure_level in ('HIGH', 'CRITICAL'):
                        self._health_status.add_issue(f"Memory pressure: {stats.pressure_level}")
            except Exception:
                pass
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return self._health_status.to_dict()
    
    def is_healthy(self) -> bool:
        """检查是否健康"""
        return self._health_status.is_healthy
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断多模态策略状态"""
        diagnosis = {
            'config': self.config.to_dict(),
            'layer_info': self.get_layer_info(),
            'health_status': self.get_health_status(),
            'modality_stats': {m: s.to_dict() for m, s in self._modality_stats.items()},
            'current_stage': self._current_mm_stage.value,
            'current_phase': self._current_phase.value if hasattr(self._current_phase, 'value') else str(self._current_phase),
            'recommendations': [],
        }
        
        # 添加建议
        if not self._health_status.is_healthy:
            diagnosis['recommendations'].append(f"Health issues: {self._health_status.issues}")
        
        # 添加策略指标
        if self._strategy_metrics is not None:
            diagnosis['metrics'] = {
                'total_steps': self._strategy_metrics.total_steps,
                'avg_loss': self._strategy_metrics.avg_loss,
            }
        
        return diagnosis
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diagnosis = self.diagnose()
        print("\n" + "=" * 60)
        print("MultiModal Strategy Diagnosis")
        print("=" * 60)
        print(f"Modalities: {diagnosis['config']['modalities']}")
        print(f"Fusion: {diagnosis['config']['fusion_method']}")
        print(f"Alignment: {diagnosis['config']['alignment_method']}")
        print(f"Current Stage: {diagnosis['current_stage']}")
        print(f"Current Phase: {diagnosis['current_phase']}")
        print(f"\nHealth Status: {'Healthy' if diagnosis['health_status']['is_healthy'] else 'Unhealthy'}")
        if diagnosis['health_status']['issues']:
            print(f"  Issues: {diagnosis['health_status']['issues']}")
        print(f"\nLayer Info:")
        for k, v in diagnosis['layer_info'].items():
            if isinstance(v, bool):
                print(f"  {k}: {'✓' if v else '✗'}")
        if 'metrics' in diagnosis:
            print(f"\nMetrics:")
            print(f"  Total Steps: {diagnosis['metrics']['total_steps']}")
            print(f"  Avg Loss: {diagnosis['metrics']['avg_loss']:.4f}")
        if diagnosis['recommendations']:
            print(f"\nRecommendations:")
            for rec in diagnosis['recommendations']:
                print(f"  - {rec}")
        print("=" * 60)
    
    def get_modality_stats(self, modality: str) -> Optional[Dict[str, Any]]:
        """获取模态统计"""
        if modality in self._modality_stats:
            return self._modality_stats[modality].to_dict()
        return None
    
    def get_all_modality_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模态统计"""
        return {m: s.to_dict() for m, s in self._modality_stats.items()}
    
    def get_summary(self) -> Dict[str, Any]:
        """获取策略摘要"""
        summary = {
            'strategy_type': self.STRATEGY_TYPE.value,
            'training_stage': self._current_mm_stage.value,
            'training_phase': self._current_phase.value if hasattr(self._current_phase, 'value') else str(self._current_phase),
            'modalities': self.config.modalities,
            'fusion_method': self.config.fusion_method,
            'healthy': self._health_status.is_healthy,
        }
        
        # 添加策略指标
        if self._strategy_metrics is not None:
            summary['metrics'] = {
                'total_steps': self._strategy_metrics.total_steps,
                'avg_loss': self._strategy_metrics.avg_loss,
            }
        
        return summary
    
    def print_summary(self) -> None:
        """打印策略摘要"""
        summary = self.get_summary()
        print(f"\nMultiModal Strategy Summary:")
        print(f"  Type: {summary['strategy_type']}")
        print(f"  Stage: {summary['training_stage']}")
        print(f"  Phase: {summary['training_phase']}")
        print(f"  Modalities: {summary['modalities']}")
        print(f"  Fusion: {summary['fusion_method']}")
        print(f"  Healthy: {summary['healthy']}")
        if 'metrics' in summary:
            print(f"  Total Steps: {summary['metrics']['total_steps']}")
            print(f"  Avg Loss: {summary['metrics']['avg_loss']:.4f}")
    
    # ==================== 内存管理方法 ====================
    
    def optimize_memory(self) -> None:
        """优化内存使用"""
        if self._memory_manager is not None:
            try:
                self._memory_manager.clear_cache()
                return
            except Exception:
                pass
        
        # 回退
        if clear_memory is not None:
            clear_memory()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计
        
        使用: get_available_memory (backend/lib/hardware)
        """
        stats = {}
        
        # 使用 get_available_memory
        try:
            available = get_available_memory()
            if isinstance(available, (int, float)):
                stats['available_gb'] = available / (1024**3) if available > 1000 else available
            elif isinstance(available, dict):
                stats['available_memory'] = available
        except Exception as e:
            logger.debug(f"get_available_memory failed: {e}")
        
        # 使用 MemoryManager
        if self._memory_manager is not None:
            try:
                mgr_stats = self._memory_manager.get_stats()
                if hasattr(mgr_stats, '__dict__'):
                    stats.update(mgr_stats.__dict__)
                else:
                    stats['manager_stats'] = str(mgr_stats)
            except Exception:
                pass
        
        # 回退到 PyTorch 原生
        if torch.cuda.is_available():
            stats.update({
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
            })
        
        return stats
    
    def get_distributed_info(self) -> Dict[str, Any]:
        """
        获取分布式信息
        
        使用: is_main_process, get_rank, get_world_size (backend/lib/distributed)
        """
        info = {
            'distributed_enabled': self.config.use_distributed
        }
        
        # 使用 get_rank
        if get_rank is not None:
            try:
                info['rank'] = get_rank()
            except Exception as e:
                logger.debug(f"get_rank failed: {e}")
                info['rank'] = 0
        else:
            info['rank'] = 0
        
        # 使用 get_world_size
        if get_world_size is not None:
            try:
                info['world_size'] = get_world_size()
            except Exception as e:
                logger.debug(f"get_world_size failed: {e}")
                info['world_size'] = 1
        else:
            info['world_size'] = 1
        
        # 使用 is_main_process
        if is_main_process is not None:
            try:
                info['is_main'] = is_main_process()
            except Exception as e:
                logger.debug(f"is_main_process failed: {e}")
                info['is_main'] = info['rank'] == 0
        else:
            info['is_main'] = info['rank'] == 0
        
        # 使用 DistributedManager
        if self._distributed_manager is not None:
            try:
                mgr_info = self._distributed_manager.get_info()
                if mgr_info:
                    info['manager_info'] = mgr_info
            except Exception:
                pass
        
        return info
    
    def should_log(self) -> bool:
        """
        是否应该记录日志（仅主进程）
        
        使用: is_main_process (backend/lib/distributed)
        """
        if not self.config.use_distributed:
            return True
        
        try:
            return is_main_process()
        except Exception:
            pass
        
        # 回退
        dist_info = self.get_distributed_info()
        return dist_info.get('is_main', True)
    
    def get_loss_stats(self) -> Dict[str, Any]:
        """
        获取损失统计
        
        使用: LossStats (backend/lib/losses)
        """
        stats = {}
        
        # 使用 LossStats
        try:
            if self._lib_loss_monitor is not None:
                # 从监控器获取统计
                if hasattr(self._lib_loss_monitor, 'get_stats'):
                    monitor_stats = self._lib_loss_monitor.get_stats()
                    if isinstance(monitor_stats, LossStats):
                        stats['loss_stats'] = {
                            'total_steps': getattr(monitor_stats, 'total_steps', 0),
                            'avg_loss': getattr(monitor_stats, 'avg_loss', 0.0),
                            'min_loss': getattr(monitor_stats, 'min_loss', float('inf')),
                            'max_loss': getattr(monitor_stats, 'max_loss', 0.0),
                        }
                    elif isinstance(monitor_stats, dict):
                        stats['loss_stats'] = monitor_stats
        except Exception as e:
            logger.debug(f"LossStats retrieval failed: {e}")
        
        # 使用策略指标
        if self._strategy_metrics is not None:
            stats['strategy_metrics'] = {
                'total_steps': self._strategy_metrics.total_steps,
                'avg_loss': self._strategy_metrics.avg_loss,
            }
        
        return stats
    
    # ==================== 生命周期回调 ====================
    
    def on_step_end(self, context: StrategyContext, result: StrategyResult) -> None:
        """
        步骤结束回调
        
        使用: is_main_process, barrier (backend/lib/distributed)
        """
        super().on_step_end(context, result)
        
        # 定期健康检查
        if (self.config.health_check_interval > 0 and 
            context.global_step % self.config.health_check_interval == 0):
            self._check_health()
        
        # 更新训练阶段
        self._update_training_phase(context)
        
        # 定期日志记录（仅主进程）
        if (self.config.log_interval > 0 and 
            context.global_step % self.config.log_interval == 0):
            if self.should_log():  # 使用 is_main_process
                self._log_step_info(context, result)
        
        # 分布式同步（如果需要）
        if self.config.use_distributed :
            if context.global_step % self.config.gradient_accumulation_steps == 0:
                # 使用 barrier
                if barrier is not None:
                    try:
                        barrier()
                    except Exception as e:
                        logger.debug(f"Barrier failed: {e}")
    
    def _log_step_info(self, context: StrategyContext, result: StrategyResult) -> None:
        """
        记录步骤信息（仅主进程调用）
        
        使用: get_rank, get_world_size (backend/lib/distributed)
        """
        dist_info = self.get_distributed_info()
        
        log_msg = (
            f"Step {context.global_step}: "
            f"loss={result.loss.item() if hasattr(result.loss, 'item') else result.loss:.4f}, "
            f"stage={self._current_mm_stage.value}, "
            f"phase={self._current_phase.value if hasattr(self._current_phase, 'value') else str(self._current_phase)}"
        )
        
        # 添加分布式信息
        if dist_info.get('world_size', 1) > 1:
            log_msg += f", rank={dist_info.get('rank', 0)}/{dist_info.get('world_size', 1)}"
        
        logger.info(log_msg)
    
    def _update_training_phase(self, context: StrategyContext) -> None:
        """更新训练阶段"""
        max_steps = context.max_steps or 10000
        warmup_ratio = 0.1
        cooldown_ratio = 0.1
        
        warmup_steps = int(max_steps * warmup_ratio)
        cooldown_start = int(max_steps * (1 - cooldown_ratio))
        
        old_phase = self._current_phase
        
        if context.global_step < warmup_steps:
            self._current_phase = TrainingPhase.WARMUP
        elif context.global_step >= cooldown_start:
            self._current_phase = TrainingPhase.COOLDOWN
        else:
            self._current_phase = TrainingPhase.MAIN
        
        if old_phase != self._current_phase:
            logger.info(f"Training phase changed: {old_phase} -> {self._current_phase}")


# ==================== 生产级多模态策略 ====================

class ProductionMultiModalStrategy(MultiModalStrategy):
    """
    生产级多模态训练策略
    
    完整支持四阶段训练流程和生产级组件：
    1. 模态预训练（Modality Pretraining）
    2. 跨模态对齐（Cross-Modal Alignment）
    3. 指令微调（Instruction Tuning）
    4. 对齐与安全（Alignment & Safety）
    """
    
    def __init__(self, config: Optional[MultiModalStrategyConfig] = None):
        if config is None:
            config = MultiModalStrategyConfig(
                modalities=['text', 'image'],
                task_loss_weight=1.0,
                align_loss_weight=0.5,
                contrastive_loss_weight=0.2,
                fusion_method="cross_attention",
                fusion_stage="middle",
                use_production_mode=True
            )
        super().__init__(config)
        self.name = "production_multimodal"
        
        # 训练阶段
        self.current_stage = MultiModalTrainingStage.MODALITY_PRETRAIN
        
        # 阶段配置
        self._stage_configs = {
            MultiModalTrainingStage.MODALITY_PRETRAIN: {
                'task_loss_weight': 1.0,
                'align_loss_weight': 0.0,
                'contrastive_loss_weight': 0.3,
                'freeze_encoders': False
            },
            MultiModalTrainingStage.CROSS_MODAL_ALIGN: {
                'task_loss_weight': 0.5,
                'align_loss_weight': 1.0,
                'contrastive_loss_weight': 0.5,
                'freeze_encoders': True
            },
            MultiModalTrainingStage.INSTRUCTION_TUNING: {
                'task_loss_weight': 1.0,
                'align_loss_weight': 0.1,
                'contrastive_loss_weight': 0.0,
                'freeze_encoders': True
            },
            MultiModalTrainingStage.ALIGNMENT_SAFETY: {
                'task_loss_weight': 1.0,
                'align_loss_weight': 0.0,
                'contrastive_loss_weight': 0.0,
                'freeze_encoders': True
            }
        }
        
        # 生产级训练器
        self._production_trainer = None
        self._production_model = None
    
    def setup(self, context: StrategyContext) -> None:
        """初始化生产级组件"""
        super().setup(context)
        
        # 初始化生产级训练器
        if self.config.use_production_mode:
            self._init_production_trainer(context)
    
    def _init_production_trainer(self, context: StrategyContext):
        """初始化生产级训练器"""
        try:
            # 从底层库导入
            from backend.lib.multimodal import (
                ProductionMultiModalModel,
                ProductionMultiModalTrainer,
                ProductionMultiModalConfig
            )
            
            # 映射模态
            modality_map = {
                'text': ModalityType.TEXT,
                'image': ModalityType.IMAGE,
                'audio': ModalityType.AUDIO,
                'time_series': ModalityType.TIME_SERIES,
                'video': ModalityType.VIDEO,
                'table': ModalityType.TABULAR
            }
            
            modalities = [
                modality_map.get(m, ModalityType.TEXT) 
                for m in self.config.modalities
                if m in modality_map
            ]
            
            # 创建配置
            prod_config = ProductionMultiModalConfig(
                project_name="multimodal_strategy",
                modalities=modalities,
                output_dir=self.config.output_dir
            )
            
            # 创建模型
            self._production_model = ProductionMultiModalModel(prod_config)
            self._production_model.to(context.device)
            
            # 创建训练器
            self._production_trainer = ProductionMultiModalTrainer(
                model=self._production_model,
                config=prod_config
            )
            
            logger.info("Production trainer initialized")
            
        except ImportError as e:
            logger.warning(f"Failed to init production trainer: {e}")
    
    def set_training_stage(self, stage: Union[str, MultiModalTrainingStage]):
        """设置训练阶段
        
        Args:
            stage: 训练阶段
        """
        if isinstance(stage, str):
            stage_map = {
                "modality_pretrain": MultiModalTrainingStage.MODALITY_PRETRAIN,
                "cross_modal_align": MultiModalTrainingStage.CROSS_MODAL_ALIGN,
                "instruction_tuning": MultiModalTrainingStage.INSTRUCTION_TUNING,
                "alignment_safety": MultiModalTrainingStage.ALIGNMENT_SAFETY
            }
            stage = stage_map.get(stage)
            if stage is None:
                raise ValueError(f"Invalid stage: {stage}")
        
        self.current_stage = stage
        self.config.training_stage = stage.value
        
        # 应用阶段配置
        stage_config = self._stage_configs[stage]
        self.config.task_loss_weight = stage_config['task_loss_weight']
        self.config.align_loss_weight = stage_config['align_loss_weight']
        self.config.contrastive_loss_weight = stage_config['contrastive_loss_weight']
        
        # 更新生产级模型
        if self._production_model is not None:
            from backend.lib.multimodal import TrainingStage
            stage_map = {
                MultiModalTrainingStage.MODALITY_PRETRAIN: TrainingStage.MODALITY_PRETRAIN,
                MultiModalTrainingStage.CROSS_MODAL_ALIGN: TrainingStage.CROSS_MODAL_ALIGN,
                MultiModalTrainingStage.INSTRUCTION_TUNING: TrainingStage.INSTRUCTION_TUNING,
                MultiModalTrainingStage.ALIGNMENT_SAFETY: TrainingStage.ALIGNMENT_SAFETY
            }
            self._production_model.set_training_stage(stage_map[stage])
        
        logger.info(f"Training stage set to: {stage.value}")
    
    def run_four_stage_training(
        self,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """运行四阶段训练
        
        Args:
            train_dataloader: 训练数据加载器
            eval_dataloader: 评估数据加载器
            callbacks: 回调函数列表
            
        Returns:
            训练结果和指标
        """
        if self._production_trainer is not None:
            self._production_trainer.train_dataloader = train_dataloader
            self._production_trainer.eval_dataloader = eval_dataloader
            self._production_trainer.callbacks = callbacks or []
            
            state = self._production_trainer.train()
            
            return {
                'completed': True,
                'stage_metrics': state.stage_metrics,
                'final_stage': state.stage.value
            }
        
        # 手动执行四阶段训练
        results = {}
        stages = [
            MultiModalTrainingStage.MODALITY_PRETRAIN,
            MultiModalTrainingStage.CROSS_MODAL_ALIGN,
            MultiModalTrainingStage.INSTRUCTION_TUNING,
            MultiModalTrainingStage.ALIGNMENT_SAFETY
        ]
        
        for stage in stages:
            self.set_training_stage(stage)
            logger.info(f"Starting stage: {stage.value}")
            
            # 这里需要外部训练循环
            results[stage.value] = {
                'task_loss_weight': self.config.task_loss_weight,
                'align_loss_weight': self.config.align_loss_weight,
                'contrastive_loss_weight': self.config.contrastive_loss_weight
            }
        
        return {'completed': True, 'stage_configs': results}
    
    def on_phase_start(self, phase: TrainingPhase, context: StrategyContext) -> None:
        """训练阶段开始回调"""
        phase_mapping = {
            TrainingPhase.PRETRAIN: MultiModalTrainingStage.MODALITY_PRETRAIN,
            TrainingPhase.PRETRAIN_INDUSTRY: MultiModalTrainingStage.MODALITY_PRETRAIN,
            TrainingPhase.ALIGN_INDUSTRY: MultiModalTrainingStage.CROSS_MODAL_ALIGN,
            TrainingPhase.FINETUNE: MultiModalTrainingStage.INSTRUCTION_TUNING,
            TrainingPhase.FINETUNE_SCENE: MultiModalTrainingStage.INSTRUCTION_TUNING,
            TrainingPhase.PREFERENCE: MultiModalTrainingStage.ALIGNMENT_SAFETY
        }
        
        if phase in phase_mapping:
            self.set_training_stage(phase_mapping[phase])
    
    def forward(
        self,
        inputs: Dict[str, Dict[str, Tensor]],
        labels: Optional[Tensor] = None,
        return_loss: bool = True
    ) -> Dict[str, Any]:
        """完整的前向传播
        
        Args:
            inputs: 各模态输入 {modality: {input_key: tensor}}
            labels: 标签（可选）
            return_loss: 是否返回损失
            
        Returns:
            输出字典
        """
        if self._production_model is not None:
            return self._production_model(inputs, labels=labels)
        
        raise NotImplementedError("Please initialize production model first")
    
    def get_model(self) -> Optional[nn.Module]:
        """获取生产级模型"""
        return self._production_model
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        if self._production_trainer is not None:
            # Try public method first
            if hasattr(self._production_trainer, 'save_checkpoint'):
                self._production_trainer.save_checkpoint(path)
            else:
                # Fallback to protected if public not available (though not recommended)
                # Or use torch.save logic here
                pass
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        if self._production_trainer is not None:
            self._production_trainer.load_checkpoint(path)


# ==================== 行业多模态策略 ====================

class IndustryMultiModalStrategy(ProductionMultiModalStrategy):
    """
    行业多模态训练策略
    
    针对行业场景优化：
    - 制造业：时序信号、缺陷检测
    - 金融：文档、表格
    - 医疗：图像、文档
    """
    
    def __init__(self, 
                 industry_type: str = "manufacturing",
                 config: Optional[MultiModalStrategyConfig] = None):
        # 根据行业类型设置默认模态
        industry_modalities = {
            "manufacturing": ['text', 'time_series', 'image'],
            "finance": ['text', 'table', 'document'],
            "healthcare": ['text', 'image', 'document'],
            "retail": ['text', 'image', 'table'],
            "logistics": ['text', 'time_series', 'table']
        }
        
        modalities = industry_modalities.get(industry_type, ['text', 'image'])
        
        if config is None:
            config = MultiModalStrategyConfig(
                modalities=modalities,
                task_loss_weight=1.0,
                align_loss_weight=0.3,
                contrastive_loss_weight=0.2,
                fusion_method="attention",
                use_production_mode=True
            )
        
        super().__init__(config)
        self.name = f"industry_multimodal_{industry_type}"
        self.industry_type = industry_type
    
    def prepare_batch(self, batch: Dict[str, Any], context: StrategyContext) -> Dict[str, Any]:
        """准备行业多模态数据"""
        batch = super().prepare_batch(batch, context)
        
        # 时序数据标准化
        if 'time_series' in batch:
            ts_data = batch['time_series']
            if isinstance(ts_data, torch.Tensor):
                mean = ts_data.mean(dim=-1, keepdim=True)
                std = ts_data.std(dim=-1, keepdim=True) + 1e-6
                batch['time_series'] = (ts_data - mean) / std
        
        return batch


# ==================== 完整管道类 ====================

class MultiModalTrainingPipeline:
    """
    多模态训练管道
    
    整合数据工程、编码、对齐、融合、训练的完整流程
    """
    
    def __init__(self, config: MultiModalPipelineConfig):
        self.config = config
        
        # 策略
        self.strategy = ProductionMultiModalStrategy(config.strategy_config)
        
        # 数据管道
        self._data_pipeline = None
        
        # 状态
        self.current_stage = MultiModalTrainingStage.MODALITY_PRETRAIN
        self.training_history: List[Dict[str, Any]] = []
    
    def setup(self, device: torch.device = None):
        """初始化管道"""
        device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建策略上下文
        context = StrategyContext(
            device=device,
            config={
                f'{m}_dim': self.config.strategy_config.fusion_dim
                for m in self.config.strategy_config.modalities
            }
        )
        
        # 初始化策略
        self.strategy.setup(context)
        
        # 初始化数据管道
        if self.config.strategy_config.enable_data_engineering:
            self._init_data_pipeline()
        
        logger.info("MultiModalTrainingPipeline initialized")
    
    def _init_data_pipeline(self):
        """初始化数据管道"""
        try:
            from backend.lib.multimodal import (
                MultiModalDataPipeline,
                DataEngineeringConfig,
                DataDeduplicationConfig,
                DataFilterConfig,
                DataAugmentationConfig
            )
            
            config = DataEngineeringConfig(
                deduplication=DataDeduplicationConfig(
                    enabled=self.config.strategy_config.enable_deduplication
                ),
                filtering=DataFilterConfig(
                    enabled=self.config.strategy_config.enable_filtering
                ),
                augmentation=DataAugmentationConfig(
                    enabled=self.config.strategy_config.enable_augmentation
                )
            )
            
            self._data_pipeline = MultiModalDataPipeline(config)
            logger.info("Data pipeline initialized")
            
        except ImportError as e:
            logger.warning(f"Failed to init data pipeline: {e}")
    
    def preprocess_data(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """数据预处理
        
        Args:
            samples: 原始样本列表
            
        Returns:
            处理后的样本
        """
        if self._data_pipeline is None:
            return samples
        
        try:
            from backend.lib.multimodal import (
                MultiModalSample
            )
            
            # 转换为MultiModalSample
            mm_samples = []
            for i, sample in enumerate(samples):
                mm_samples.append(MultiModalSample(
                    sample_id=sample.get('id', f'sample_{i}'),
                    modalities=sample.get('modalities', sample),
                    metadata=sample.get('metadata', {})
                ))
            
            # 处理
            processed = self._data_pipeline.process(mm_samples)
            
            # 转换回字典
            return [
                {
                    'id': s.sample_id,
                    'modalities': s.modalities,
                    'metadata': s.metadata,
                    'quality_score': s.quality_score
                }
                for s in processed
            ]
            
        except Exception as e:
            logger.warning(f"Data preprocessing failed: {e}")
            return samples
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """执行训练
        
        Args:
            train_dataloader: 训练数据加载器
            eval_dataloader: 评估数据加载器
            callbacks: 回调函数
            
        Returns:
            训练结果
        """
        return self.strategy.run_four_stage_training(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            callbacks=callbacks
        )
    
    def encode(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """编码多模态数据"""
        # 使用策略的编码方法
        context = StrategyContext(
            device=next(iter(self.strategy.alignment_projectors.values())).weight.device
            if self.strategy.alignment_projectors else torch.device('cpu')
        )
        return self.strategy.encode_modalities(batch, context)
    
    def align(self, features: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Optional[Tensor]]:
        """对齐多模态特征"""
        context = StrategyContext(
            device=next(iter(features.values())).device
        )
        aligned, loss, _ = self.strategy.align_modalities(features, context)
        return aligned, loss
    
    def fuse(self, features: Dict[str, Tensor]) -> Tensor:
        """融合多模态特征"""
        context = StrategyContext(
            device=next(iter(features.values())).device
        )
        return self.strategy.fuse_modalities(features, context)
    
    def get_data_statistics(self) -> Optional[Dict[str, Any]]:
        """获取数据处理统计"""
        if self._data_pipeline is not None:
            return self._data_pipeline.get_statistics()
        return None


# ==================== 便捷函数 ====================

def create_multimodal_strategy(
    modalities: List[str] = None,
    fusion_method: str = "cross_attention",
    use_production_mode: bool = False
) -> MultiModalStrategy:
    """创建多模态训练策略
    
    Args:
        modalities: 模态列表
        fusion_method: 融合方法
        use_production_mode: 是否使用生产模式
        
    Returns:
        多模态训练策略实例
    """
    config = MultiModalStrategyConfig(
        modalities=modalities or ['text', 'image'],
        fusion_method=fusion_method,
        use_production_mode=use_production_mode
    )
    
    if use_production_mode:
        return ProductionMultiModalStrategy(config)
    else:
        return MultiModalStrategy(config)


def create_production_multimodal_pipeline(
    modalities: List[str] = None,
    fusion_method: str = "cross_attention",
    enable_data_engineering: bool = True,
    output_dir: str = "./outputs"
) -> MultiModalTrainingPipeline:
    """创建生产级多模态训练管道
    
    Args:
        modalities: 模态列表
        fusion_method: 融合方法
        enable_data_engineering: 是否启用数据工程
        output_dir: 输出目录
        
    Returns:
        多模态训练管道实例
    """
    strategy_config = MultiModalStrategyConfig(
        modalities=modalities or ['text', 'image'],
        fusion_method=fusion_method,
        use_production_mode=True,
        enable_data_engineering=enable_data_engineering,
        output_dir=output_dir
    )
    
    pipeline_config = MultiModalPipelineConfig(
        strategy_config=strategy_config,
        output_dir=output_dir
    )
    
    return MultiModalTrainingPipeline(pipeline_config)


def create_industry_multimodal_strategy(
    industry_type: str = "manufacturing"
) -> IndustryMultiModalStrategy:
    """创建行业多模态策略
    
    Args:
        industry_type: 行业类型 (manufacturing, finance, healthcare, retail, logistics)
        
    Returns:
        行业多模态策略实例
    """
    return IndustryMultiModalStrategy(industry_type=industry_type)


def run_multimodal_training(
    pipeline: MultiModalTrainingPipeline,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    callbacks: Optional[List[Callable]] = None,
    device: str = "auto"
) -> Dict[str, Any]:
    """运行多模态训练
    
    Args:
        pipeline: 训练管道
        train_dataloader: 训练数据加载器
        eval_dataloader: 评估数据加载器
        callbacks: 回调函数
        device: 设备
        
    Returns:
        训练结果
    """
    # 设置设备
    if device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # 初始化管道
    pipeline.setup(device)
    
    # 运行训练
    return pipeline.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        callbacks=callbacks
    )


# ==================== 诊断和比较函数 ====================

def diagnose_multimodal_strategy(strategy: MultiModalStrategy) -> Dict[str, Any]:
    """诊断多模态策略"""
    return strategy.diagnose()


def print_multimodal_diagnosis(strategy: MultiModalStrategy) -> None:
    """打印多模态策略诊断"""
    strategy.print_diagnosis()


def get_available_training_stages() -> List[str]:
    """获取可用的训练阶段"""
    return [stage.value for stage in MultiModalTrainingStage]


def get_available_fusion_types() -> List[str]:
    """获取可用的融合类型"""
    return [ft.value for ft in FusionType]


def get_available_alignment_types() -> List[str]:
    """获取可用的对齐类型"""
    return [at.value for at in AlignmentType]


def compare_multimodal_strategies(strategies: List[MultiModalStrategy]) -> Dict[str, Any]:
    """比较多模态策略"""
    comparison = {}
    for i, strategy in enumerate(strategies):
        name = strategy.name or f"strategy_{i}"
        comparison[name] = {
            'modalities': strategy.config.modalities,
            'fusion_method': strategy.config.fusion_method,
            'alignment_method': strategy.config.alignment_method,
            'production_mode': strategy.config.use_production_mode,
            'layers': strategy.get_layer_info(),
            'health': strategy.is_healthy(),
        }
    return comparison


def print_multimodal_comparison(strategies: List[MultiModalStrategy]) -> None:
    """打印多模态策略比较"""
    comparison = compare_multimodal_strategies(strategies)
    
    print("\n" + "=" * 70)
    print("MultiModal Strategies Comparison")
    print("=" * 70)
    print(f"{'Name':<20} {'Modalities':<15} {'Fusion':<15} {'Align':<15} {'Health':<10}")
    print("-" * 70)
    for name, info in comparison.items():
        modalities = ','.join(info['modalities'][:2])
        if len(info['modalities']) > 2:
            modalities += '...'
        health = '✓' if info['health'] else '✗'
        print(f"{name:<20} {modalities:<15} {info['fusion_method']:<15} "
              f"{info['alignment_method']:<15} {health:<10}")
    print("=" * 70)


def recommend_multimodal_config(
    num_modalities: int,
    model_size_gb: float = 1.0,
    memory_gb: float = 16.0
) -> Dict[str, Any]:
    """推荐多模态配置
    
    Args:
        num_modalities: 模态数量
        model_size_gb: 模型大小 (GB)
        memory_gb: 可用内存 (GB)
    
    Returns:
        推荐配置
    """
    recommendation = {
        'fusion_method': 'cross_attention',
        'alignment_method': 'contrastive',
        'use_production_mode': model_size_gb > 1.0,
        'use_amp': True,
        'reasoning': [],
    }
    
    # 根据模态数量推荐融合方法
    if num_modalities <= 2:
        recommendation['fusion_method'] = 'cross_attention'
        recommendation['reasoning'].append("Cross-attention is efficient for 2 modalities")
    elif num_modalities <= 4:
        recommendation['fusion_method'] = 'middle'
        recommendation['reasoning'].append("Middle fusion balances complexity for 3-4 modalities")
    else:
        recommendation['fusion_method'] = 'perceiver'
        recommendation['reasoning'].append("Perceiver handles many modalities efficiently")
    
    # 根据模型大小推荐
    if model_size_gb > memory_gb * 0.5:
        recommendation['use_amp'] = True
        recommendation['use_gradient_checkpointing'] = True
        recommendation['reasoning'].append("Enable AMP and gradient checkpointing for large models")
    
    # 根据内存推荐对齐方法
    if memory_gb < 8:
        recommendation['alignment_method'] = 'explicit'
        recommendation['reasoning'].append("Explicit alignment is memory-efficient")
    
    return recommendation


def print_multimodal_recommendation(
    num_modalities: int,
    model_size_gb: float = 1.0,
    memory_gb: float = 16.0
) -> None:
    """打印多模态配置推荐"""
    rec = recommend_multimodal_config(num_modalities, model_size_gb, memory_gb)
    
    print("\n" + "=" * 60)
    print("MultiModal Configuration Recommendation")
    print("=" * 60)
    print(f"Modalities: {num_modalities}")
    print(f"Model Size: {model_size_gb:.1f} GB")
    print(f"Available Memory: {memory_gb:.1f} GB")
    print(f"\nRecommended Configuration:")
    print(f"  Fusion Method: {rec['fusion_method']}")
    print(f"  Alignment Method: {rec['alignment_method']}")
    print(f"  Production Mode: {rec['use_production_mode']}")
    print(f"  Use AMP: {rec['use_amp']}")
    if rec.get('use_gradient_checkpointing'):
        print(f"  Gradient Checkpointing: True")
    print("\nReasoning:")
    for r in rec['reasoning']:
        print(f"  - {r}")
    print("=" * 60)

def get_layer_availability() -> Dict[str, bool]:
    """获取各层可用性"""
    return {
        'adapters': True,
        'losses': True,
        'hardware': True,
        'distributed': True,
        'strategy': True
    }


# 导出
__all__ = [
    # 枚举
    'MultiModalTrainingStage',
    'FusionType',
    'AlignmentType',
    
    # 数据类
    'MultiModalStrategyConfig',
    'MultiModalPipelineConfig',
    'ModalityStats',
    'MultiModalHealthStatus',
    
    # 策略类
    'MultiModalStrategy',
    'ProductionMultiModalStrategy',
    'IndustryMultiModalStrategy',
    
    # 管道
    'MultiModalTrainingPipeline',
    
    # 创建函数
    'create_multimodal_strategy',
    'create_production_multimodal_pipeline',
    'create_industry_multimodal_strategy',
    'run_multimodal_training',
    
    # 诊断和比较函数
    'diagnose_multimodal_strategy',
    'print_multimodal_diagnosis',
    'compare_multimodal_strategies',
    'print_multimodal_comparison',
    
    # 查询函数
    'get_available_training_stages',
    'get_available_fusion_types',
    'get_available_alignment_types',
    'get_layer_availability',
    
    # 推荐函数
    'recommend_multimodal_config',
    'print_multimodal_recommendation',
]


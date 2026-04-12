# -*- coding: utf-8 -*-
"""
多模态底层算法库

提供生产级多模态训练的底层能力：
- 配置管理（multimodal_config）
- 模态编码器（encoders）
- 跨模态对齐（alignment）
- 多模态融合（fusion）
- 四阶段训练器（trainer）
- 数据工程（data_engineering）

使用示例:
```python
from backend.lib.multimodal import (
    # 配置
    MultiModalConfig,
    MultiModalPresets,
    ModalityType,
    TrainingStage,
    FusionStage,
    FusionMethod,
    AlignmentMethod,
    
    # 编码器
    TextEncoder,
    ImageEncoder,
    AudioEncoder,
    TimeSeriesEncoder,
    VideoEncoder,
    ModalityEncoderFactory,
    
    # 对齐
    CrossModalAligner,
    ContrastiveLearningAlignment,
    ExplicitAlignment,
    OptimalTransportAlignment,
    
    # 融合
    MultiModalFuser,
    EarlyFusion,
    MiddleFusion,
    LateFusion,
    QFormer,
    PerceiverFusion,
    
    # 训练
    MultiModalModel,
    MultiModalTrainer,
    TrainingState,
    run_four_stage_training,
    
    # 数据工程
    MultiModalDataPipeline,
    MultiModalSample,
    DataDeduplicator,
    DataFilter,
    DataAugmentor
)
```
"""

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# ==================== 配置类 ====================

from .multimodal_config import (
    # 枚举
    ModalityType,
    EncoderType,
    AlignmentMethod,
    FusionStage,
    FusionMethod,
    TrainingStage,
    DataSourceType,
    
    # 数据工程配置
    DataDeduplicationConfig,
    DataFilterConfig,
    DataAugmentationConfig,
    DataEngineeringConfig,
    
    # 编码器配置
    TextEncoderConfig,
    ImageEncoderConfig,
    AudioEncoderConfig,
    TimeSeriesEncoderConfig,
    VideoEncoderConfig,
    ModalEncodersConfig,
    
    # 对齐配置
    ContrastiveLearningConfig,
    ExplicitAlignConfig,
    CrossModalAlignmentConfig,
    
    # 融合配置
    EarlyFusionConfig,
    MiddleFusionConfig,
    LateFusionConfig,
    QFormerConfig,
    PerceiverConfig,
    MultiModalFusionConfig,
    
    # 训练配置
    ModalityPretrainConfig,
    CrossModalAlignTrainConfig,
    InstructionTuningConfig,
    AlignmentSafetyConfig,
    FourStageTrainingConfig,
    
    # 分布式配置
    DistributedConfig,
    
    # 推理配置
    InferenceConfig,
    
    # 风险配置
    RiskMitigationConfig,
    
    # 总配置
    MultiModalConfig,
    MultiModalPresets,
    
    # 工具函数
    create_config,
    validate_config,
    print_config_summary,
    compare_configs,
    print_config_comparison,
    recommend_config,
    estimate_training_resources,
    print_resource_estimate,
)

# ==================== 编码器 ====================

from .encoders import (
    # 枚举和数据类
    EncoderStatus,
    EncoderMetrics,
    EncoderMonitor,
    EncoderProfiler,
    
    # 基类
    BaseModalityEncoder,
    
    # 具体编码器
    TextEncoder,
    ImageEncoder,
    AudioEncoder,
    TimeSeriesEncoder,
    VideoEncoder,
    
    # 辅助模块
    AdapterModule,
    MeanPooler,
    MaxPooler,
    PositionalEncoding,
    SimpleViT,
    Simple3DViT,
    
    # 工厂
    ModalityEncoderFactory,
    
    # 投影层
    UnifiedProjection,
    
    # 工具函数
    create_encoder,
    create_encoders_from_config,
    freeze_encoders,
    unfreeze_encoders,
    get_encoders_summary,
    print_encoders_summary,
    count_total_parameters,
    estimate_encoders_memory,
    reset_all_encoders_stats,
    diagnose_encoders,
    print_diagnosis,
    compare_encoders_performance,
    print_performance_comparison,
    create_unified_projection_from_encoders,
    encode_multimodal_inputs,
)

# ==================== 对齐 ====================

from .alignment import (
    # 枚举和数据类
    NegativeMiningStrategy,
    AlignmentStats,
    
    # 监控组件
    AlignmentMonitor,
    TemperatureScheduler,
    
    # 对比学习
    ContrastiveLearningAlignment,
    HardNegativeMining,
    
    # 显式对齐
    ExplicitAlignment,
    CrossModalAttention,
    
    # 最优传输
    OptimalTransportAlignment,
    
    # 统一对齐器
    CrossModalAligner,
    
    # 损失函数
    AlignmentLoss,
    
    # 工具函数
    create_aligner,
    create_contrastive_loss,
    compute_alignment_metrics,
    analyze_alignment_quality,
    print_alignment_analysis,
    recommend_alignment_method,
)

# ==================== 融合 ====================

from .fusion import (
    # 枚举和数据类
    FusionStatus,
    FusionMetrics,
    FusionMonitor,
    FusionProfiler,
    
    # 早期融合
    EarlyFusion,
    
    # 中期融合
    MiddleFusion,
    CrossAttentionFusionLayer,
    
    # 后期融合
    LateFusion,
    ModalityAttentionFusion,
    GatedFusion,
    
    # Q-Former
    QFormer,
    QFormerLayer,
    
    # Perceiver
    PerceiverFusion,
    PerceiverCrossAttention,
    PerceiverSelfAttention,
    
    # 统一融合器
    MultiModalFuser,
    
    # 工具函数
    create_fusion_module,
    estimate_fusion_memory,
    recommend_fusion_strategy,
    compare_fusion_strategies,
    print_fusion_comparison,
    analyze_modality_contribution,
    print_modality_contribution,
    diagnose_fusion_module,
    print_fusion_diagnosis,
)

# ==================== 训练器 ====================

from .trainer import (
    # 状态和监控
    TrainingState,
    TrainingMetrics,
    TrainingMonitor,
    CheckpointManager,
    
    # 模型
    MultiModalModel,
    
    # 训练器
    MultiModalTrainer,
    
    # 便捷函数
    create_multimodal_trainer,
    run_four_stage_training,
    diagnose_trainer,
    print_trainer_diagnosis,
    estimate_training_time,
    compare_training_configs,
    print_training_config_comparison,
)

# ==================== 数据工程 ====================

from .data_engineering import (
    # 数据样本
    MultiModalSample,
    
    # 去重
    DataDeduplicator,
    
    # 过滤
    DataFilter,
    
    # 增强
    DataAugmentor,
    
    # 管道
    MultiModalDataPipeline,
    
    # 数据集构建器
    MultiModalDatasetBuilder
)


# ==================== 导出列表 ====================

__all__ = [
    # 配置枚举
    'ModalityType',
    'EncoderType',
    'AlignmentMethod',
    'FusionStage',
    'FusionMethod',
    'TrainingStage',
    'DataSourceType',
    
    # 数据工程配置
    'DataDeduplicationConfig',
    'DataFilterConfig',
    'DataAugmentationConfig',
    'DataEngineeringConfig',
    
    # 编码器配置
    'TextEncoderConfig',
    'ImageEncoderConfig',
    'AudioEncoderConfig',
    'TimeSeriesEncoderConfig',
    'VideoEncoderConfig',
    'ModalEncodersConfig',
    
    # 对齐配置
    'ContrastiveLearningConfig',
    'ExplicitAlignConfig',
    'CrossModalAlignmentConfig',
    
    # 融合配置
    'EarlyFusionConfig',
    'MiddleFusionConfig',
    'LateFusionConfig',
    'QFormerConfig',
    'PerceiverConfig',
    'MultiModalFusionConfig',
    
    # 训练配置
    'ModalityPretrainConfig',
    'CrossModalAlignTrainConfig',
    'InstructionTuningConfig',
    'AlignmentSafetyConfig',
    'FourStageTrainingConfig',
    
    # 其他配置
    'DistributedConfig',
    'InferenceConfig',
    'RiskMitigationConfig',
    'MultiModalConfig',
    'MultiModalPresets',
    
    # 配置工具函数
    'create_config',
    'validate_config',
    'print_config_summary',
    'compare_configs',
    'print_config_comparison',
    'recommend_config',
    'estimate_training_resources',
    'print_resource_estimate',
    
    # 编码器枚举和数据类
    'EncoderStatus',
    'EncoderMetrics',
    'EncoderMonitor',
    'EncoderProfiler',
    
    # 编码器
    'BaseModalityEncoder',
    'TextEncoder',
    'ImageEncoder',
    'AudioEncoder',
    'TimeSeriesEncoder',
    'VideoEncoder',
    'AdapterModule',
    'MeanPooler',
    'MaxPooler',
    'PositionalEncoding',
    'SimpleViT',
    'Simple3DViT',
    'ModalityEncoderFactory',
    'UnifiedProjection',
    
    # 编码器工具函数
    'create_encoder',
    'create_encoders_from_config',
    'freeze_encoders',
    'unfreeze_encoders',
    'get_encoders_summary',
    'print_encoders_summary',
    'count_total_parameters',
    'estimate_encoders_memory',
    'reset_all_encoders_stats',
    'diagnose_encoders',
    'print_diagnosis',
    'compare_encoders_performance',
    'print_performance_comparison',
    'create_unified_projection_from_encoders',
    'encode_multimodal_inputs',
    
    # 对齐枚举和数据类
    'NegativeMiningStrategy',
    'AlignmentStats',
    
    # 对齐监控组件
    'AlignmentMonitor',
    'TemperatureScheduler',
    
    # 对齐模块
    'ContrastiveLearningAlignment',
    'HardNegativeMining',
    'ExplicitAlignment',
    'CrossModalAttention',
    'OptimalTransportAlignment',
    'CrossModalAligner',
    'AlignmentLoss',
    
    # 对齐工具函数
    'create_aligner',
    'create_contrastive_loss',
    'compute_alignment_metrics',
    'analyze_alignment_quality',
    'print_alignment_analysis',
    'recommend_alignment_method',
    
    # 融合枚举和数据类
    'FusionStatus',
    'FusionMetrics',
    'FusionMonitor',
    'FusionProfiler',
    
    # 融合
    'EarlyFusion',
    'MiddleFusion',
    'CrossAttentionFusionLayer',
    'LateFusion',
    'ModalityAttentionFusion',
    'GatedFusion',
    'QFormer',
    'QFormerLayer',
    'PerceiverFusion',
    'PerceiverCrossAttention',
    'PerceiverSelfAttention',
    'MultiModalFuser',
    
    # 融合工具函数
    'create_fusion_module',
    'estimate_fusion_memory',
    'recommend_fusion_strategy',
    'compare_fusion_strategies',
    'print_fusion_comparison',
    'analyze_modality_contribution',
    'print_modality_contribution',
    'diagnose_fusion_module',
    'print_fusion_diagnosis',
    
    # 训练器状态和监控
    'TrainingState',
    'TrainingMetrics',
    'TrainingMonitor',
    'CheckpointManager',
    
    # 训练器模型和训练类
    'MultiModalModel',
    'MultiModalTrainer',
    
    # 训练器工具函数
    'create_multimodal_trainer',
    'run_four_stage_training',
    'diagnose_trainer',
    'print_trainer_diagnosis',
    'estimate_training_time',
    'compare_training_configs',
    'print_training_config_comparison',
    
    # 数据工程
    'MultiModalSample',
    'DataDeduplicator',
    'DataFilter',
    'DataAugmentor',
    'MultiModalDataPipeline',
    'MultiModalDatasetBuilder'
]


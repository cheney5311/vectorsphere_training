# -*- coding: utf-8 -*-
"""
多模态训练模块

业务层模块，整合底层算法能力和业务逻辑：
- 底层算法能力来自 backend.lib.multimodal
- 业务配置来自 multimodal_config.py
- 业务训练器来自 multimodal_trainer.py

架构层次：
├── backend/lib/hardware            # 硬件抽象层
├── backend/lib/distributed         # 分布式训练内核层
├── backend/lib/adapters            # 模型/模态适配器层
├── backend/lib/losses              # 目标函数层
├── backend/lib/multimodal          # 底层算法库（本模块依赖）
│   ├── multimodal_config.py  # 配置定义
│   ├── encoders.py           # 模态编码器
│   ├── alignment.py          # 跨模态对齐
│   ├── fusion.py             # 多模态融合
│   ├── trainer.py            # 四阶段训练器
│   └── data_engineering.py   # 数据工程
├── backend/modules/training/multimodal  # 业务训练模块（本模块）
│   ├── multimodal_config.py             # 业务配置
│   ├── multimodal_trainer.py            # 业务训练器（调用策略层）
│   └── __init__.py                      # 统一导出
└── backend/modules/training/strategies  # 策略层（本模块调用）
    ├── base.py               # 生产级策略基类
    └── multimodal_strategy.py           # 多模态训练策略

调用关系：
本模块 -> strategies/multimodal_strategy.py -> backend/lib/* (底层六层架构)
"""

import logging

logger = logging.getLogger(__name__)

# ==================== 从底层库导入生产级组件 ====================

try:
    from backend.lib.multimodal import (
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
        
        # 其他配置
        DistributedConfig,
        InferenceConfig,
        RiskMitigationConfig,
        MultiModalConfig,
        MultiModalPresets,
        
        # 编码器
        BaseModalityEncoder,
        TextEncoder,
        ImageEncoder,
        AudioEncoder,
        TimeSeriesEncoder,
        VideoEncoder,
        AdapterModule,
        MeanPooler,
        MaxPooler,
        PositionalEncoding,
        SimpleViT,
        Simple3DViT,
        ModalityEncoderFactory,
        UnifiedProjection,
        
        # 对齐
        ContrastiveLearningAlignment,
        HardNegativeMining,
        ExplicitAlignment,
        CrossModalAttention,
        OptimalTransportAlignment,
        CrossModalAligner,
        AlignmentLoss,
        
        # 融合
        EarlyFusion,
        MiddleFusion,
        CrossAttentionFusionLayer,
        LateFusion,
        ModalityAttentionFusion,
        GatedFusion,
        QFormer,
        QFormerLayer,
        PerceiverFusion,
        PerceiverCrossAttention,
        PerceiverSelfAttention,
        MultiModalFuser,
        
        # 训练器
        TrainingState,
        MultiModalModel,
        MultiModalTrainer,
        create_multimodal_trainer,
        run_four_stage_training,
        
        # 数据工程
        MultiModalSample,
        DataDeduplicator,
        DataFilter,
        DataAugmentor,
        MultiModalDataPipeline,
        MultiModalDatasetBuilder
    )
    _LIB_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Failed to import from backend.lib.multimodal: {e}")
    _LIB_AVAILABLE = False
    
    # 定义占位符
    ModalityType = None
    EncoderType = None
    AlignmentMethod = None
    FusionStage = None
    FusionMethod = None
    TrainingStage = None
    DataSourceType = None
    MultiModalConfig = None
    MultiModalPresets = None
    TextEncoder = None
    ImageEncoder = None
    AudioEncoder = None
    TimeSeriesEncoder = None
    VideoEncoder = None
    ModalityEncoderFactory = None
    UnifiedProjection = None
    ContrastiveLearningAlignment = None
    ExplicitAlignment = None
    CrossModalAttention = None
    OptimalTransportAlignment = None
    CrossModalAligner = None
    AlignmentLoss = None
    EarlyFusion = None
    MiddleFusion = None
    LateFusion = None
    QFormer = None
    PerceiverFusion = None
    MultiModalFuser = None
    MultiModalModel = None
    MultiModalTrainer = None
    TrainingState = None
    create_multimodal_trainer = None
    run_four_stage_training = None
    MultiModalSample = None
    DataDeduplicator = None
    DataFilter = None
    DataAugmentor = None
    MultiModalDataPipeline = None
    MultiModalDatasetBuilder = None

# ==================== 业务层模块 ====================

# 业务配置（可能有外部依赖）
try:
    from .multimodal_config import MultiModalConfig
except ImportError:
    MultiModalConfig = None

# 业务训练器（可能有外部依赖）
try:
    from .multimodal_trainer import MultiModalTrainer
except ImportError:
    MultiModalTrainer = None


# ==================== 便捷函数 ====================

def get_multimodal_config(**kwargs) -> 'MultiModalConfig':
    """获取生产级多模态配置
    
    Args:
        **kwargs: 配置参数
        
    Returns:
        生产级多模态配置实例
    """
    if MultiModalConfig is None:
        raise ImportError("Production multimodal config not available")
    return MultiModalConfig(**kwargs)


def get_preset_config(preset_name: str) -> 'MultiModalConfig':
    """获取预设配置
    
    Args:
        preset_name: 预设名称 (image_text_base, video_understanding, 
                    industrial_multimodal, large_scale_training)
        
    Returns:
        预设的配置实例
    """
    if MultiModalPresets is None:
        raise ImportError("Production multimodal presets not available")
    
    presets = {
        'image_text_base': MultiModalPresets.image_text_base,
        'video_understanding': MultiModalPresets.video_understanding,
        'industrial_multimodal': MultiModalPresets.industrial_multimodal,
        'large_scale_training': MultiModalPresets.large_scale_training
    }
    
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
    
    return presets[preset_name]()


def create_production_trainer(
    config: 'MultiModalConfig' = None,
    train_dataloader=None,
    eval_dataloader=None
) -> 'MultiModalTrainer':
    """创建生产级训练器
    
    Args:
        config: 配置，如果为None则使用默认配置
        train_dataloader: 训练数据加载器
        eval_dataloader: 评估数据加载器
        
    Returns:
        训练器实例
    """
    if create_multimodal_trainer is None:
        raise ImportError("Production multimodal trainer not available")
    
    if config is None:
        config = MultiModalConfig()
    
    return create_multimodal_trainer(config, train_dataloader, eval_dataloader)


def is_lib_available() -> bool:
    """检查底层库是否可用"""
    return _LIB_AVAILABLE


# ==================== 导出列表 ====================

__all__ = [
    # 可用性检查
    'is_lib_available',
    
    # 便捷函数
    'get_production_multimodal_config',
    'get_preset_config',
    'create_production_trainer',
    
    # 业务模块
    'MultiModalConfig',
    'MultiModalTrainer',
    
    # 枚举
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
    'DistributedConfig',
    'InferenceConfig',
    'RiskMitigationConfig',
    'MultiModalConfig',
    'MultiModalPresets',
    
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
    
    # 对齐
    'ContrastiveLearningAlignment',
    'HardNegativeMining',
    'ExplicitAlignment',
    'CrossModalAttention',
    'OptimalTransportAlignment',
    'CrossModalAligner',
    'AlignmentLoss',
    
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
    
    # 训练器
    'TrainingState',
    'MultiModalModel',
    'MultiModalTrainer',
    'create_multimodal_trainer',
    'run_four_stage_training',
    
    # 数据工程
    'MultiModalSample',
    'DataDeduplicator',
    'DataFilter',
    'DataAugmentor',
    'MultiModalDataPipeline',
    'MultiModalDatasetBuilder'
]

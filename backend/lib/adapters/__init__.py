# -*- coding: utf-8 -*-
"""
Model & Modality Adapter Layer（模型/模态适配器层）

统一管理各种模型架构和多模态适配，包括：
- 模态编码器（Text / Image / Audio / Video / Time-Series）
- 模态融合（Early / Middle / Late Fusion）
- 跨模态对齐（Contrastive / Explicit / Optimal Transport）
- 模型适配器（Backbone Adapter / Task Head）

架构位置：
┌──────────────────────────────────────┐
│          Training Orchestrator       │
├──────────────────────────────────────┤
│       Training Strategy Abstraction  │
├──────────────────────────────────────┤
│      Loss & Objective Composition    │
├──────────────────────────────────────┤
│ >>> Model & Modality Adapter <<< │  ← 当前层
│     (Text / Image / Audio / Fusion)  │
├──────────────────────────────────────┤
│        Distributed Training Core     │
├──────────────────────────────────────┤
│          Hardware Abstraction        │
└──────────────────────────────────────┘

使用示例:
```python
from backend.modules.training.adapters import (
    # 适配器管理
    AdapterManager,
    get_adapter_manager,
    
    # 模态编码器
    ModalityEncoder,
    TextEncoder,
    ImageEncoder,
    AudioEncoder,
    VideoEncoder,
    TimeSeriesEncoder,
    
    # 融合模块
    FusionModule,
    EarlyFusion,
    MiddleFusion,
    LateFusion,
    CrossAttentionFusion,
    
    # 对齐模块
    AlignmentModule,
    ContrastiveAlignment,
    ExplicitAlignment,
    
    # 模型适配器
    ModelAdapter,
    BackboneAdapter,
    TaskHeadAdapter,
    LoRAAdapter
)

# 创建适配器管理器
manager = get_adapter_manager()

# 获取编码器
text_encoder = manager.get_encoder('text', hidden_size=768)
image_encoder = manager.get_encoder('image', hidden_size=768)

# 创建融合模块
fusion = manager.create_fusion('cross_attention', hidden_size=768)
```
"""

import logging

logger = logging.getLogger(__name__)

# ==================== 模态编码器 ====================

from .modality_encoders import (
    # 枚举
    ModalityType,
    EncoderStatus,
    PoolingMethod,
    AugmentationType,
    NormalizationType,
    
    # 数据类
    EncoderMetrics,
    EncoderConfig,
    
    # 基类
    ModalityEncoder,
    
    # 编码器实现
    TextEncoder,
    ImageEncoder,
    AudioEncoder,
    VideoEncoder,
    TimeSeriesEncoder,
    TabularEncoder,
    GraphEncoder,
    PointCloudEncoder,
    GraphAttentionLayer,
    
    # 多模态编码器
    MultiModalEncoder,
    
    # 质量分析器
    EncoderQualityAnalyzer,
    
    # 编码器工厂
    EncoderFactory,
    create_encoder,
    
    # 配置构建器
    EncoderConfigBuilder,
    build_encoder_config,
    
    # 便捷函数
    create_multimodal_encoder,
    encoder_factory_health_check
)

# ==================== 融合模块 ====================

from .fusion_modules import (
    # 枚举
    FusionMethod,
    FusionStatus,
    PoolingType,
    NormType,
    
    # 数据类
    FusionMetrics,
    FusionConfig,
    
    # 融合模块
    FusionModule,
    RMSNorm,
    EarlyFusion,
    MiddleFusion,
    LateFusion,
    CrossAttentionFusion,
    GatedFusion,
    QFormerFusion,
    PerceiverFusion,
    TensorFusion,
    BilinearFusion,
    HybridFusion,
    
    # 质量分析器
    FusionQualityAnalyzer,
    
    # 融合工厂
    FusionFactory,
    create_fusion,
    create_hybrid_fusion,
    
    # 配置构建器
    FusionConfigBuilder,
    build_fusion_config
)

# ==================== 对齐模块 ====================

from .alignment_modules import (
    # 枚举
    AlignmentMethod,
    AlignmentStatus,
    PoolingMethod,
    LossType,
    AugmentationType,
    QualityMetric,
    
    # 数据类
    AlignmentMetrics,
    AlignmentConfig,
    
    # 对齐模块
    AlignmentModule,
    ContrastiveAlignment,
    ExplicitAlignment,
    OptimalTransportAlignment,
    CrossModalAttentionAlignment,
    CCAAlignment,
    HybridAlignment,
    
    # 对齐工厂
    AlignmentFactory,
    create_alignment,
    
    # 配置构建器
    AlignmentConfigBuilder,
    build_alignment_config
)

# ==================== 模型适配器 ====================

from .model_adapters import (
    # 枚举
    AdapterType,
    AdapterStatus,
    MergeStrategy,
    InitStrategy,
    
    # 数据类
    AdapterMetrics,
    AdapterConfig,
    
    # 基类
    ModelAdapter,
    
    # 适配器实现
    BackboneAdapter,
    TaskHeadAdapter,
    LoRALayer,
    LoRAAdapter,
    PrefixAdapter,
    PromptAdapter,
    AdapterLayerModule,
    AdapterLayersAdapter,
    BitFitAdapter,
    IA3Layer,
    IA3Adapter,
    CompacterLayer,
    CompacterAdapter,
    
    # 适配器融合
    AdapterFusion,
    AdapterMerger,
    
    # 质量分析器
    AdapterQualityAnalyzer,
    
    # 适配器工厂
    AdapterFactory,
    create_adapter,
    
    # 配置构建器
    AdapterConfigBuilder,
    build_adapter_config,
    
    # 便捷函数
    create_lora_adapter,
    create_adapter_fusion,
    merge_adapters,
    adapter_factory_health_check,
    get_adapter_summary
)

# ==================== 适配器管理器 ====================

from .adapter_manager import (
    # 核心类
    AdapterManager,
    AdapterManagerConfig,
    
    # 枚举
    ComponentType,
    ManagerStatus,
    HealthStatus,
    
    # 数据类
    ComponentMetrics,
    PipelineMetrics,
    
    # 全局函数
    get_adapter_manager,
    reset_adapter_manager,
    
    # 管理器便捷函数 - 基础
    get_managed_encoder,
    get_managed_fusion,
    get_managed_alignment,
    get_managed_adapter,
    create_multimodal_pipeline,
    run_multimodal_pipeline,
    adapter_manager_health_check,
    adapter_manager_metrics,
    
    # 配置构建器
    AdapterManagerConfigBuilder,
    build_adapter_manager_config,
    
    # 高级便捷函数 - 分析
    analyze_encoder,
    analyze_fusion,
    analyze_alignment,
    
    # 高级便捷函数 - 适配器融合
    create_managed_adapter_fusion,
    merge_managed_adapters,
    compare_managed_adapters,
    
    # 高级便捷函数 - 管道
    create_managed_multimodal_encoder,
    create_hybrid_pipeline,
    create_advanced_pipeline,
    
    # 高级便捷函数 - 诊断
    run_diagnostics,
    get_component_health_status,
    optimize_manager_cache
)


__all__ = [
    # 模态枚举
    'ModalityType',
    'EncoderStatus',
    'PoolingMethod',
    'AugmentationType',
    'NormalizationType',
    
    # 编码器数据类
    'EncoderMetrics',
    'EncoderConfig',
    
    # 模态编码器
    'ModalityEncoder',
    'TextEncoder',
    'ImageEncoder',
    'AudioEncoder',
    'VideoEncoder',
    'TimeSeriesEncoder',
    'TabularEncoder',
    'GraphEncoder',
    'PointCloudEncoder',
    'GraphAttentionLayer',
    
    # 多模态编码器
    'MultiModalEncoder',
    
    # 质量分析器
    'EncoderQualityAnalyzer',
    
    # 编码器工厂
    'EncoderFactory',
    'create_encoder',
    
    # 编码器配置构建器
    'EncoderConfigBuilder',
    'build_encoder_config',
    
    # 编码器便捷函数
    'create_multimodal_encoder',
    'encoder_factory_health_check',
    
    # 融合枚举
    'FusionMethod',
    'FusionStatus',
    'PoolingType',
    'NormType',
    
    # 融合数据类
    'FusionMetrics',
    'FusionConfig',
    
    # 融合模块
    'FusionModule',
    'RMSNorm',
    'EarlyFusion',
    'MiddleFusion',
    'LateFusion',
    'CrossAttentionFusion',
    'GatedFusion',
    'QFormerFusion',
    'PerceiverFusion',
    'TensorFusion',
    'BilinearFusion',
    'HybridFusion',
    
    # 融合质量分析
    'FusionQualityAnalyzer',
    
    # 融合工厂
    'FusionFactory',
    'create_fusion',
    'create_hybrid_fusion',
    'FusionConfigBuilder',
    'build_fusion_config',
    
    # 对齐枚举
    'AlignmentMethod',
    'AlignmentStatus',
    'PoolingMethod',
    'LossType',
    'AugmentationType',
    'QualityMetric',
    
    # 对齐数据类
    'AlignmentMetrics',
    'AlignmentConfig',
    
    # 对齐模块
    'AlignmentModule',
    'ContrastiveAlignment',
    'ExplicitAlignment',
    'OptimalTransportAlignment',
    'CrossModalAttentionAlignment',
    'CCAAlignment',
    'HybridAlignment',
    'AlignmentFactory',
    'create_alignment',
    'AlignmentConfigBuilder',
    'build_alignment_config',
    
    # 适配器枚举
    'AdapterType',
    'AdapterStatus',
    'MergeStrategy',
    'InitStrategy',
    
    # 适配器数据类
    'AdapterMetrics',
    'AdapterConfig',
    
    # 模型适配器
    'ModelAdapter',
    'BackboneAdapter',
    'TaskHeadAdapter',
    'LoRALayer',
    'LoRAAdapter',
    'PrefixAdapter',
    'PromptAdapter',
    'AdapterLayerModule',
    'AdapterLayersAdapter',
    'BitFitAdapter',
    'IA3Layer',
    'IA3Adapter',
    'CompacterLayer',
    'CompacterAdapter',
    
    # 适配器融合
    'AdapterFusion',
    'AdapterMerger',
    
    # 适配器质量分析
    'AdapterQualityAnalyzer',
    
    # 适配器工厂
    'AdapterFactory',
    'create_adapter',
    
    # 适配器配置构建器
    'AdapterConfigBuilder',
    'build_adapter_config',
    
    # 适配器便捷函数
    'create_lora_adapter',
    'create_adapter_fusion',
    'merge_adapters',
    'adapter_factory_health_check',
    'get_adapter_summary',
    
    # 管理器
    'AdapterManager',
    'AdapterManagerConfig',
    'get_adapter_manager',
    'reset_adapter_manager',
    
    # 管理器枚举和数据类
    'ComponentType',
    'ManagerStatus',
    'HealthStatus',
    'ComponentMetrics',
    'PipelineMetrics',
    
    # 管理器便捷函数 - 基础
    'get_managed_encoder',
    'get_managed_fusion',
    'get_managed_alignment',
    'get_managed_adapter',
    'create_multimodal_pipeline',
    'run_multimodal_pipeline',
    'adapter_manager_health_check',
    'adapter_manager_metrics',
    
    # 管理器配置构建器
    'AdapterManagerConfigBuilder',
    'build_adapter_manager_config',
    
    # 高级便捷函数 - 分析
    'analyze_encoder',
    'analyze_fusion',
    'analyze_alignment',
    
    # 高级便捷函数 - 适配器融合
    'create_managed_adapter_fusion',
    'merge_managed_adapters',
    'compare_managed_adapters',
    
    # 高级便捷函数 - 管道
    'create_managed_multimodal_encoder',
    'create_hybrid_pipeline',
    'create_advanced_pipeline',
    
    # 高级便捷函数 - 诊断
    'run_diagnostics',
    'get_component_health_status',
    'optimize_manager_cache'
]


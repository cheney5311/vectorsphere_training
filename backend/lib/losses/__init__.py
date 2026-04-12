# -*- coding: utf-8 -*-
"""
Loss & Objective Composition Layer（目标函数层）

统一管理训练中的各种损失函数和目标函数组合，包括：
- 监督学习损失（分类、回归、分割）
- 知识蒸馏损失（软标签、特征、注意力）
- 对比学习损失（InfoNCE、NT-Xent）
- 正则化损失（L1、L2、Dropout）
- 自定义复合损失

架构位置：
┌──────────────────────────────────────┐
│          Training Orchestrator       │
├──────────────────────────────────────┤
│       Training Strategy Abstraction  │
├──────────────────────────────────────┤
│   >>> Loss & Objective Composition <<│  ← 当前层
├──────────────────────────────────────┤
│      Model & Modality Adapter Layer  │
├──────────────────────────────────────┤
│        Distributed Training Core     │
├──────────────────────────────────────┤
│          Hardware Abstraction        │
└──────────────────────────────────────┘

使用示例:
```python
from backend.modules.training.losses import (
    # 损失工厂
    LossFactory,
    create_loss,
    
    # 监督损失
    SupervisedLoss,
    ClassificationLoss,
    RegressionLoss,
    SegmentationLoss,
    
    # 蒸馏损失
    DistillationLossModule,
    SoftLabelLoss,
    FeatureDistillationLoss,
    AttentionDistillationLoss,
    
    # 对比学习损失
    ContrastiveLoss,
    InfoNCELoss,
    NTXentLoss,
    
    # 复合损失
    CompositeLoss,
    MultiTaskLoss
)

# 创建损失函数
loss_fn = create_loss('cross_entropy', num_classes=10)

# 创建复合损失
composite = CompositeLoss([
    ('task', CrossEntropyLoss(), 1.0),
    ('kd', SoftLabelLoss(temperature=4.0), 0.5),
    ('contrastive', InfoNCELoss(), 0.1)
])
```
"""

import logging

logger = logging.getLogger(__name__)

# ==================== 基础损失 ====================

from .base_loss import (
    # 基础类
    BaseLoss,
    LossResult,
    LossConfig,
    LossRegistry,
    
    # 枚举
    LossType,
    
    # 数据类
    LossStats,
    
    # 组件类
    LossMonitor,
    LossValidator,
    
    # 工具函数
    register_loss,
    reduce_loss,
    weighted_loss,
    combine_losses,
    validate_loss_tensor,
    create_loss_from_config,
    
    # 调度函数
    linear_weight_schedule,
    cosine_weight_schedule,
    exponential_weight_schedule,
    
    # 上下文管理器
    loss_computation_context,
    temporary_weight,
    
    # 辅助函数
    print_loss_comparison,
    aggregate_loss_stats,
)

# ==================== 监督学习损失 ====================

from .supervised_loss import (
    # 基类
    SupervisedLoss,
    ClassificationLoss,
    RegressionLoss,
    SegmentationLoss,
    
    # 具体实现
    CrossEntropyLoss,
    FocalLoss,
    LabelSmoothingLoss,
    MSELoss,
    MAELoss,
    HuberLoss,
    DiceLoss,
    IoULoss,
    
    # 监控和统计组件
    ClassificationStats,
    RegressionStats,
    SegmentationStats,
    SupervisedMonitor,
    ClassWeightCalculator,
    
    # 工具函数
    create_supervised_loss,
    compare_supervised_losses,
    compute_class_weights,
    print_confusion_matrix,
    analyze_classification_results,
    print_classification_analysis,
    recommend_loss_function,
    print_loss_recommendation,
)

# ==================== 蒸馏损失 ====================

from .distillation_loss import (
    # 基类
    DistillationLossModule,
    
    # 具体损失
    SoftLabelLoss,
    FeatureDistillationLoss,
    AttentionDistillationLoss,
    RelationalDistillationLoss,
    
    # 组合蒸馏
    CombinedDistillationLoss,
    
    # 监控和统计组件
    DistillationStats,
    DistillationMonitor,
    TemperatureScheduler as DistillationTemperatureScheduler,
    LayerWeightManager,
    
    # 工具函数
    create_distillation_loss,
    compare_distillation_losses,
    analyze_distillation_progress,
    print_distillation_progress,
    recommend_temperature,
    estimate_distillation_quality,
    print_distillation_quality,
)

# ==================== 对比学习损失 ====================

from .contrastive_loss import (
    # 基类
    ContrastiveLoss,
    
    # 具体损失
    InfoNCELoss,
    NTXentLoss,
    TripletLoss,
    CenterLoss,
    CrossModalContrastiveLoss,
    CLIPLoss,
    
    # 监控和统计组件
    ContrastiveStats,
    ContrastiveMonitor,
    HardNegativeMiner,
    TemperatureScheduler,
    
    # 工具函数
    create_contrastive_loss,
    compare_contrastive_losses,
    analyze_feature_distribution,
    print_feature_analysis,
    compute_retrieval_metrics,
    print_retrieval_metrics,
)

# ==================== 复合损失 ====================

from .composite_loss import (
    # 基础复合损失
    CompositeLoss,
    MultiTaskLoss,
    DynamicWeightedLoss,
    UncertaintyWeightedLoss,
    GradNormLoss,
    
    # 监控和平衡组件
    WeightStats,
    WeightMonitor,
    TaskBalancer,
    
    # 工具函数
    create_composite_loss,
    create_multitask_loss,
    compare_composite_losses,
    analyze_weight_evolution,
    print_weight_evolution,
    balance_weights_manually,
    export_composite_config,
    export_multitask_config,
)

# ==================== 正则化损失 ====================

from .regularization_loss import (
    # 基类
    RegularizationLoss,
    
    # 具体损失
    L1Regularization,
    L2Regularization,
    ElasticNetRegularization,
    ConsistencyRegularization,
    EntropyRegularization,
    FeatureRegularization,
    SpectralRegularization,
    MixedRegularization,
    
    # 监控和统计组件
    RegularizationStats,
    RegularizationMonitor,
    LambdaScheduler,
    SparsityAnalyzer,
    
    # 工具函数
    create_regularization_loss,
    compare_regularization_losses,
    analyze_model_weights,
    print_weight_analysis,
    recommend_regularization,
    print_regularization_recommendation,
)

# ==================== 工厂函数 ====================

from .loss_factory import (
    # 核心工厂
    LossFactory,
    LossMetadata,
    FactoryStats,
    LOSS_MAPPING,
    CATEGORY_MAPPING,
    TASK_MAPPING,
    LOSS_METADATA,
    
    # 便捷创建函数
    create_loss,
    create_composite_loss as factory_create_composite_loss,
    create_distillation_loss as factory_create_distillation_loss,
    create_regularization_losses,
    
    # 查询函数
    get_loss_registry,
    get_loss_metadata,
    list_losses_by_category,
    list_losses_by_task,
    
    # 打印函数
    print_loss_info,
    print_all_losses,
    print_loss_comparison as factory_print_loss_comparison,
    
    # 比较和选择函数
    compare_losses,
    auto_select_loss,
    
    # 预设
    LossPresets,
    
    # 配置构建器
    LossConfigBuilder,
    loss_builder,
)


__all__ = [
    # 基础
    'BaseLoss',
    'LossResult',
    'LossConfig',
    'LossRegistry',
    'LossType',
    'LossStats',
    'LossMonitor',
    'LossValidator',
    
    # 工具函数
    'register_loss',
    'reduce_loss',
    'weighted_loss',
    'combine_losses',
    'validate_loss_tensor',
    'create_loss_from_config',
    
    # 调度函数
    'linear_weight_schedule',
    'cosine_weight_schedule',
    'exponential_weight_schedule',
    
    # 上下文管理器
    'loss_computation_context',
    'temporary_weight',
    
    # 辅助函数
    'print_loss_comparison',
    'aggregate_loss_stats',
    
    # 监督学习
    'SupervisedLoss',
    'ClassificationLoss',
    'RegressionLoss',
    'SegmentationLoss',
    'CrossEntropyLoss',
    'FocalLoss',
    'LabelSmoothingLoss',
    'MSELoss',
    'MAELoss',
    'HuberLoss',
    'DiceLoss',
    'IoULoss',
    
    # 监督学习组件
    'ClassificationStats',
    'RegressionStats',
    'SegmentationStats',
    'SupervisedMonitor',
    'ClassWeightCalculator',
    
    # 监督学习工具函数
    'create_supervised_loss',
    'compare_supervised_losses',
    'compute_class_weights',
    'print_confusion_matrix',
    'analyze_classification_results',
    'print_classification_analysis',
    'recommend_loss_function',
    'print_loss_recommendation',
    
    # 蒸馏
    'DistillationLossModule',
    'SoftLabelLoss',
    'FeatureDistillationLoss',
    'AttentionDistillationLoss',
    'RelationalDistillationLoss',
    'CombinedDistillationLoss',
    
    # 蒸馏组件
    'DistillationStats',
    'DistillationMonitor',
    'DistillationTemperatureScheduler',
    'LayerWeightManager',
    
    # 蒸馏工具函数
    'create_distillation_loss',
    'compare_distillation_losses',
    'analyze_distillation_progress',
    'print_distillation_progress',
    'recommend_temperature',
    'estimate_distillation_quality',
    'print_distillation_quality',
    
    # 对比学习
    'ContrastiveLoss',
    'InfoNCELoss',
    'NTXentLoss',
    'TripletLoss',
    'CenterLoss',
    'CrossModalContrastiveLoss',
    'CLIPLoss',
    
    # 对比学习组件
    'ContrastiveStats',
    'ContrastiveMonitor',
    'HardNegativeMiner',
    'TemperatureScheduler',
    
    # 对比学习工具函数
    'create_contrastive_loss',
    'compare_contrastive_losses',
    'analyze_feature_distribution',
    'print_feature_analysis',
    'compute_retrieval_metrics',
    'print_retrieval_metrics',
    
    # 复合
    'CompositeLoss',
    'MultiTaskLoss',
    'DynamicWeightedLoss',
    'UncertaintyWeightedLoss',
    'GradNormLoss',
    
    # 复合损失组件
    'WeightStats',
    'WeightMonitor',
    'TaskBalancer',
    
    # 复合损失工具函数
    'create_composite_loss',
    'create_multitask_loss',
    'compare_composite_losses',
    'analyze_weight_evolution',
    'print_weight_evolution',
    'balance_weights_manually',
    'export_composite_config',
    'export_multitask_config',
    
    # 正则化
    'RegularizationLoss',
    'L1Regularization',
    'L2Regularization',
    'ElasticNetRegularization',
    'ConsistencyRegularization',
    'EntropyRegularization',
    'FeatureRegularization',
    'SpectralRegularization',
    'MixedRegularization',
    
    # 正则化组件
    'RegularizationStats',
    'RegularizationMonitor',
    'LambdaScheduler',
    'SparsityAnalyzer',
    
    # 正则化工具函数
    'create_regularization_loss',
    'compare_regularization_losses',
    'analyze_model_weights',
    'print_weight_analysis',
    'recommend_regularization',
    'print_regularization_recommendation',
    
    # 工厂核心
    'LossFactory',
    'LossMetadata',
    'FactoryStats',
    'LOSS_MAPPING',
    'CATEGORY_MAPPING',
    'TASK_MAPPING',
    'LOSS_METADATA',
    
    # 工厂创建函数
    'create_loss',
    'create_composite_loss',
    'create_distillation_loss',
    'create_regularization_losses',
    
    # 工厂查询函数
    'get_loss_registry',
    'get_loss_metadata',
    'list_losses_by_category',
    'list_losses_by_task',
    
    # 工厂打印函数
    'print_loss_info',
    'print_all_losses',
    
    # 工厂比较和选择
    'compare_losses',
    'auto_select_loss',
    
    # 预设配置
    'LossPresets',
    
    # 配置构建器
    'LossConfigBuilder',
    'loss_builder',
]


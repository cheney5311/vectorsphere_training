# -*- coding: utf-8 -*-
"""
训练策略模块

提供可组合的训练策略层，支持多种训练范式：
- 标准训练 (StandardTrainingStrategy) → 策略层
- 多模态训练 (MultiModalStrategy) → 适配器层 + 损失层
- 知识蒸馏 (DistillationStrategy) → 策略层 + 损失层
- 场景化训练 (ScenarioStrategy) → 策略层 + 损失层
- 分布式训练 (DistributedStrategy) → 分布式层 + 硬件层
- 行业模型训练 (IndustryScenarioStrategy) → 全部六层

架构调用关系：
┌─────────────────────────────────────────────────────────────────┐
│ strategies/* (策略层)                                            │
│   ├── IndustryScenarioStrategy → 全部六层                        │
│   ├── ScenarioStrategy → 策略层 + 损失层                         │
│   ├── DistributedStrategy → 分布式层 + 硬件层                    │
│   ├── DistillationStrategy → 策略层 + 损失层                     │
│   ├── MultiModalStrategy → 适配器层 + 损失层                     │
│   └── StandardTrainingStrategy → 策略层                          │
├─────────────────────────────────────────────────────────────────┤
│ backend/lib/losses (损失层)                                      │
│ backend/lib/adapters (适配器层)                                  │
│ backend/lib/distributed (分布式层)                               │
│ backend/lib/hardware (硬件层)                                    │
└─────────────────────────────────────────────────────────────────┘

详细文档请参考: STRATEGIES_README.md

基于公司级行业模型训练平台技术方案设计。
"""

from .base_strategy import (
    # 枚举
    StrategyType,
    TrainingPhase,
    
    # 数据类
    StrategyContext,
    StrategyResult,
    StrategyMetrics,
    
    # 监控和诊断组件
    StrategyMonitor,
    StrategyProfiler,
    StrategyValidator,
    
    # 策略基类
    TrainingStrategy,
    StandardTrainingStrategy,
    CompositeStrategy,
    
    # 工具函数
    create_standard_strategy,
    create_composite_strategy,
    create_context,
    diagnose_strategy,
    print_strategy_diagnosis,
    compare_strategies,
    print_strategy_comparison,
    get_available_strategy_types,
    get_available_training_phases,
    validate_strategy,
    strategy_context,
    save_strategy_state,
    load_strategy_state,
    print_strategy_info,
    print_all_strategy_types,
    print_all_training_phases,
    
    # 底层能力可用性标志
    LOSSES_AVAILABLE as BASE_LOSSES_AVAILABLE,
    HARDWARE_AVAILABLE as BASE_HARDWARE_AVAILABLE,
    DISTRIBUTED_AVAILABLE as BASE_DISTRIBUTED_AVAILABLE,
)

from .multimodal_strategy import (
    # 枚举
    MultiModalTrainingStage,
    FusionType,
    AlignmentType,
    # 数据类
    MultiModalStrategyConfig,
    MultiModalPipelineConfig,
    ModalityStats,
    MultiModalHealthStatus,
    # 策略类
    MultiModalStrategy,
    ProductionMultiModalStrategy,
    IndustryMultiModalStrategy,
    # 管道
    MultiModalTrainingPipeline,
    # 创建函数
    create_multimodal_strategy,
    create_production_multimodal_pipeline,
    create_industry_multimodal_strategy,
    run_multimodal_training,
    # 诊断和比较函数
    diagnose_multimodal_strategy,
    print_multimodal_diagnosis,
    compare_multimodal_strategies,
    print_multimodal_comparison,
    # 查询函数
    get_available_training_stages,
    get_available_fusion_types,
    get_available_alignment_types,
    get_layer_availability,
    # 推荐函数
    recommend_multimodal_config,
    print_multimodal_recommendation,
    # 常量
    ADAPTERS_LAYER_AVAILABLE,
    LOSSES_LAYER_AVAILABLE as MM_LOSSES_LAYER_AVAILABLE,
    HARDWARE_LAYER_AVAILABLE as MM_HARDWARE_LAYER_AVAILABLE,
    DISTRIBUTED_LAYER_AVAILABLE as MM_DISTRIBUTED_LAYER_AVAILABLE,
)

from .distillation_strategy import (
    # 主策略类
    DistillationStrategy,
    IndustryDistillationStrategy,
    
    # 策略变体
    SelfDistillationStrategy,
    ProgressiveDistillationStrategy,
    ContrastiveDistillationStrategy,
    
    # 配置类
    DistillationStrategyConfig,
    
    # 损失计算器
    DistillationLossCalculator,
    
    # 监控组件
    DistillationStats,
    DistillationStrategyMonitor,
    TemperatureScheduler as DistillationTemperatureScheduler,
    
    # 枚举
    DistillationType,
    FeatureLossType,
    AttentionLossType,
    TemperatureSchedule,
    TemperatureScheduleType,
    ProgressiveScheduleType,
    MultiTeacherEnsembleType,
    
    # 便捷函数
    create_distillation_strategy,
    create_distillation_from_trainer_config,
    create_multi_teacher_distillation_strategy,
    diagnose_distillation_strategy,
    print_distillation_strategy_diagnosis,
    compare_distillation_strategies,
    print_distillation_strategy_comparison,
    recommend_distillation_strategy,
    get_available_distillation_types,
    get_available_strategy_variants,
    
    # 底层能力标志
    LOSSES_LAYER_AVAILABLE as DISTILLATION_LOSSES_AVAILABLE,
)

from .scenario_strategy import (
    # 枚举
    ScenarioType,
    
    # 数据类
    ScenarioStrategyConfig,
    ScenarioHealthStatus,
    ScenarioStats,
    
    # 组件
    ScenarioRouter,
    
    # 策略类
    ScenarioStrategy,
    IndustryScenarioStrategy,
    
    # 创建函数
    create_scenario_strategy,
    create_industry_scenario_strategy,
    
    # 查询和信息函数
    get_available_scenarios,
    get_scenario_info,
    print_scenario_info,
    
    # 诊断函数
    diagnose_scenario_strategy,
    print_scenario_diagnosis,
    compare_scenario_strategies,
    
    # 推荐函数
    recommend_scenario,
    
    # 常量
    LOSSES_LAYER_AVAILABLE as SCENARIO_LOSSES_AVAILABLE,
)

from .distributed_strategy import (
    # 枚举
    DistributedMode,
    ZeROStage,
    # 数据类
    DistributedStrategyConfig,
    CommunicationStats,
    DistributedHealthStatus,
    # 策略类
    DistributedStrategy,
    IndustryDistributedStrategy,
    # 工具函数
    create_distributed_strategy,
    create_industry_distributed_strategy,
    recommend_distributed_mode,
    print_distributed_recommendation,
    diagnose_distributed_strategy,
    print_distributed_diagnosis,
    get_available_distributed_modes,
    get_available_zero_stages,
    compare_distributed_modes,
    print_distributed_modes_comparison,
    # 常量
    DISTRIBUTED_LAYER_AVAILABLE,
    HARDWARE_LAYER_AVAILABLE,
)

# 生产级策略基类（整合六层架构底层能力）
from .production_base import (
    # 数据类
    ProductionStrategyConfig,
    ProductionHealthStatus,
    WrapperStats,
    
    # 策略类
    ProductionTrainingStrategy,
    ProductionTrainingContext,
    
    # 创建函数
    create_production_strategy,
    create_production_context,
    create_composite_production_strategy,
    
    # 信息和诊断函数
    get_available_layers,
    get_layer_details,
    print_layer_info,
    diagnose_production_base,
    print_production_base_diagnosis,
    
    # 底层能力可用性标志
    HARDWARE_AVAILABLE,
    DISTRIBUTED_AVAILABLE,
    ADAPTERS_AVAILABLE,
    LOSSES_AVAILABLE
)

# 三阶段训练策略（整合 StandardStrategy + Orchestrator + backend/lib）
from .three_stage_strategy import (
    # 枚举
    ThreeStagePhase,
    
    # 数据类
    PhaseStats,
    ThreeStageStats,
    ThreeStageHealthStatus,
    ThreeStageStrategyConfig,
    
    # 监控组件
    ThreeStageMonitor,
    PhaseTracker,
    DPOLossCalculator,
    
    # 辅助类
    GradientAccumulator,
    ConvergenceDetector,
    SimpleMixedPrecisionManager,
    
    # 策略类
    ThreeStageStrategy,
    
    # 便捷函数
    create_three_stage_strategy,
    get_three_stage_phases,
    get_phase_info,
    print_phase_info,
    diagnose_three_stage_strategy,
    print_three_stage_diagnosis,
    compare_dpo_loss_types,
    print_dpo_loss_comparison,
    estimate_training_time,
    print_training_time_estimate,
    get_layer_availability as get_three_stage_layer_availability,
    print_layer_availability as print_three_stage_layer_availability,
)

__all__ = [
    # 枚举
    'StrategyType',
    'TrainingPhase',
    
    # 数据类
    'StrategyContext',
    'StrategyResult',
    'StrategyMetrics',
    
    # 监控和诊断组件
    'StrategyMonitor',
    'StrategyProfiler',
    'StrategyValidator',
    
    # 基础策略
    'TrainingStrategy',
    'StandardTrainingStrategy',
    'CompositeStrategy',
    
    # 基础策略工具函数
    'create_standard_strategy',
    'create_composite_strategy',
    'create_context',
    'diagnose_strategy',
    'print_strategy_diagnosis',
    'compare_strategies',
    'print_strategy_comparison',
    'get_available_strategy_types',
    'get_available_training_phases',
    'validate_strategy',
    'strategy_context',
    'save_strategy_state',
    'load_strategy_state',
    'print_strategy_info',
    'print_all_strategy_types',
    'print_all_training_phases',
    
    # 底层能力可用性标志（基础）
    'BASE_LOSSES_AVAILABLE',
    'BASE_HARDWARE_AVAILABLE',
    'BASE_DISTRIBUTED_AVAILABLE',
    
    # 多模态策略 - 枚举
    'MultiModalTrainingStage',
    'FusionType',
    'AlignmentType',
    
    # 多模态策略 - 数据类
    'MultiModalStrategyConfig',
    'MultiModalPipelineConfig',
    'ModalityStats',
    'MultiModalHealthStatus',
    
    # 多模态策略 - 策略类
    'MultiModalStrategy',
    'ProductionMultiModalStrategy',
    'IndustryMultiModalStrategy',
    'MultiModalTrainingPipeline',
    
    # 多模态策略 - 创建函数
    'create_multimodal_strategy',
    'create_production_multimodal_pipeline',
    'create_industry_multimodal_strategy',
    'run_multimodal_training',
    
    # 多模态策略 - 诊断和比较函数
    'diagnose_multimodal_strategy',
    'print_multimodal_diagnosis',
    'compare_multimodal_strategies',
    'print_multimodal_comparison',
    
    # 多模态策略 - 查询函数
    'get_available_training_stages',
    'get_available_fusion_types',
    'get_available_alignment_types',
    'get_layer_availability',
    'print_layer_availability',
    
    # 多模态策略 - 推荐函数
    'recommend_multimodal_config',
    'print_multimodal_recommendation',
    
    # 多模态策略 - 常量
    'ADAPTERS_LAYER_AVAILABLE',
    'MM_LOSSES_LAYER_AVAILABLE',
    'MM_HARDWARE_LAYER_AVAILABLE',
    'MM_DISTRIBUTED_LAYER_AVAILABLE',
    
    # 蒸馏策略 - 主类和变体
    'DistillationStrategy',
    'IndustryDistillationStrategy',
    'SelfDistillationStrategy',
    'ProgressiveDistillationStrategy',
    'ContrastiveDistillationStrategy',
    
    # 蒸馏策略 - 配置和组件
    'DistillationStrategyConfig',
    'DistillationLossCalculator',
    'DistillationStats',
    'DistillationStrategyMonitor',
    'DistillationTemperatureScheduler',
    
    # 蒸馏策略 - 枚举
    'DistillationType',
    'FeatureLossType',
    'AttentionLossType',
    'TemperatureSchedule',
    'TemperatureScheduleType',
    'ProgressiveScheduleType',
    'MultiTeacherEnsembleType',
    
    # 蒸馏策略 - 工具函数
    'create_distillation_strategy',
    'create_distillation_from_trainer_config',
    'create_multi_teacher_distillation_strategy',
    'diagnose_distillation_strategy',
    'print_distillation_strategy_diagnosis',
    'compare_distillation_strategies',
    'print_distillation_strategy_comparison',
    'recommend_distillation_strategy',
    'get_available_distillation_types',
    'get_available_strategy_variants',
    'DISTILLATION_LOSSES_AVAILABLE',
    
    # 场景策略 - 枚举
    'ScenarioType',
    
    # 场景策略 - 数据类
    'ScenarioStrategyConfig',
    'ScenarioHealthStatus',
    'ScenarioStats',
    
    # 场景策略 - 组件
    'ScenarioRouter',
    
    # 场景策略 - 策略类
    'ScenarioStrategy',
    'IndustryScenarioStrategy',
    
    # 场景策略 - 创建函数
    'create_scenario_strategy',
    'create_industry_scenario_strategy',
    
    # 场景策略 - 查询和信息函数
    'get_available_scenarios',
    'get_scenario_info',
    'print_scenario_info',
    
    # 场景策略 - 诊断函数
    'diagnose_scenario_strategy',
    'print_scenario_diagnosis',
    'compare_scenario_strategies',
    
    # 场景策略 - 推荐函数
    'recommend_scenario',
    
    # 场景策略 - 常量
    'SCENARIO_LOSSES_AVAILABLE',
    
    # 分布式策略 - 枚举
    'DistributedMode',
    'ZeROStage',
    
    # 分布式策略 - 数据类
    'DistributedStrategyConfig',
    'CommunicationStats',
    'DistributedHealthStatus',
    
    # 分布式策略 - 策略类
    'DistributedStrategy',
    'IndustryDistributedStrategy',
    
    # 分布式策略 - 工具函数
    'create_distributed_strategy',
    'create_industry_distributed_strategy',
    'recommend_distributed_mode',
    'print_distributed_recommendation',
    'diagnose_distributed_strategy',
    'print_distributed_diagnosis',
    'get_available_distributed_modes',
    'get_available_zero_stages',
    'compare_distributed_modes',
    'print_distributed_modes_comparison',
    
    # 分布式策略 - 常量
    'DISTRIBUTED_LAYER_AVAILABLE',
    'HARDWARE_LAYER_AVAILABLE',
    
    # 生产级策略 - 数据类
    'ProductionStrategyConfig',
    'ProductionHealthStatus',
    'WrapperStats',
    
    # 生产级策略 - 策略类
    'ProductionTrainingStrategy',
    'ProductionTrainingContext',
    
    # 生产级策略 - 创建函数
    'create_production_strategy',
    'create_production_context',
    'create_composite_production_strategy',
    
    # 生产级策略 - 信息和诊断函数
    'get_available_layers',
    'get_layer_details',
    'print_layer_info',
    'diagnose_production_base',
    'print_production_base_diagnosis',
    
    # 生产级策略 - 常量
    'HARDWARE_AVAILABLE',
    'DISTRIBUTED_AVAILABLE',
    'ADAPTERS_AVAILABLE',
    'LOSSES_AVAILABLE',
    
    # 三阶段训练策略 - 枚举
    'ThreeStagePhase',
    
    # 三阶段训练策略 - 数据类
    'PhaseStats',
    'ThreeStageStats',
    'ThreeStageHealthStatus',
    'ThreeStageStrategyConfig',
    
    # 三阶段训练策略 - 监控组件
    'ThreeStageMonitor',
    'PhaseTracker',
    'DPOLossCalculator',
    
    # 三阶段训练策略 - 辅助类
    'GradientAccumulator',
    'ConvergenceDetector',
    'SimpleMixedPrecisionManager',
    
    # 三阶段训练策略 - 策略类
    'ThreeStageStrategy',
    
    # 三阶段训练策略 - 便捷函数
    'create_three_stage_strategy',
    'get_three_stage_phases',
    'get_phase_info',
    'print_phase_info',
    'diagnose_three_stage_strategy',
    'print_three_stage_diagnosis',
    'compare_dpo_loss_types',
    'print_dpo_loss_comparison',
    'estimate_training_time',
    'print_training_time_estimate',
    'get_three_stage_layer_availability',
    'print_three_stage_layer_availability',
]


def create_strategy(strategy_type: str, **kwargs):
    """
    策略工厂函数
    
    Args:
        strategy_type: 策略类型
            - standard: 标准训练
            - production: 生产级训练（整合六层架构）
            - multimodal: 多模态训练
            - production_multimodal: 生产级多模态
            - industry_multimodal: 行业多模态
            - distillation: 知识蒸馏
            - industry_distillation: 行业蒸馏
            - self_distillation: 自蒸馏
            - progressive_distillation: 渐进式蒸馏
            - contrastive_distillation: 对比蒸馏
            - scenario: 场景化训练
            - industry_scenario: 行业场景训练
            - distributed: 分布式训练
            - industry_distributed: 行业分布式
        **kwargs: 策略配置参数
    
    Returns:
        策略实例
    """
    strategy_map = {
        'standard': StandardTrainingStrategy,
        'production': ProductionTrainingStrategy,
        'multimodal': MultiModalStrategy,
        'production_multimodal': ProductionMultiModalStrategy,
        'industry_multimodal': IndustryMultiModalStrategy,
        'distillation': DistillationStrategy,
        'industry_distillation': IndustryDistillationStrategy,
        'self_distillation': SelfDistillationStrategy,
        'progressive_distillation': ProgressiveDistillationStrategy,
        'contrastive_distillation': ContrastiveDistillationStrategy,
        'scenario': ScenarioStrategy,
        'industry_scenario': IndustryScenarioStrategy,
        'distributed': DistributedStrategy,
        'industry_distributed': IndustryDistributedStrategy,
        'three_stage': ThreeStageStrategy,
    }
    
    if strategy_type not in strategy_map:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    strategy_class = strategy_map[strategy_type]
    
    # 创建配置
    config_map = {
        'production': ProductionStrategyConfig,
        'multimodal': MultiModalStrategyConfig,
        'production_multimodal': MultiModalStrategyConfig,
        'industry_multimodal': MultiModalStrategyConfig,
        'distillation': DistillationStrategyConfig,
        'industry_distillation': DistillationStrategyConfig,
        'self_distillation': DistillationStrategyConfig,
        'progressive_distillation': DistillationStrategyConfig,
        'contrastive_distillation': DistillationStrategyConfig,
        'scenario': ScenarioStrategyConfig,
        'industry_scenario': ScenarioStrategyConfig,
        'distributed': DistributedStrategyConfig,
        'industry_distributed': DistributedStrategyConfig,
        'three_stage': ThreeStageStrategyConfig,
    }
    
    if strategy_type in config_map:
        config_class = config_map[strategy_type]
        config = config_class(**kwargs) if kwargs else None
        return strategy_class(config=config)
    else:
        return strategy_class()


def create_composite_strategy(
    strategy_types: list,
    weights: list = None,
    **kwargs
):
    """
    创建组合策略
    
    Args:
        strategy_types: 策略类型列表
        weights: 权重列表
        **kwargs: 各策略的配置参数
    
    Returns:
        组合策略实例
    """
    strategies = []
    for st in strategy_types:
        strategy_config = kwargs.get(st, {})
        strategy = create_strategy(st, **strategy_config)
        strategies.append(strategy)
    
    return CompositeStrategy(strategies, weights)


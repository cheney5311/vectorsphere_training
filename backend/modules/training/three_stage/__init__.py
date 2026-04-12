# -*- coding: utf-8 -*-
"""
三阶段训练模块

提供完整的三阶段训练功能：
- 预训练（Pretrain）
- 监督微调（SFT - Supervised Fine-Tuning）
- 偏好优化（DPO - Direct Preference Optimization）

核心组件：
- ThreeStageTrainer: 三阶段训练器（业务层）
- ThreeStageStrategy: 三阶段训练策略（策略层）
- ThreeStageConfig: 训练配置
- TrainingLoop: 优化后的训练循环
- OptimizerUtils: 优化器工具

架构图：
┌──────────────────────────────────────┐
│   ThreeStageTrainer (业务层)         │
│   (数据加载、配置管理、进度回调)      │
├──────────────────────────────────────┤
│   ThreeStageStrategy (策略层)        │
│   (调用 backend/lib 实现生产级训练)   │
├──────────────────────────────────────┤
│   backend/lib/* (底层能力)           │
│   (hardware/distributed/losses/...)  │
└──────────────────────────────────────┘
"""

from .three_stage_config import ThreeStageConfig, TrainingStage, StageConfig
from .three_stage_trainer import ThreeStageTrainer, create_three_stage_trainer
from .runtime import setup_model_and_tokenizer, build_dataloaders
from .datasets import TextDataset, SFTDataset, DPODataset

# 优化模块
from .optimizer_utils import (
    OptimizerConfig,
    OptimizerType,
    SchedulerType,
    InitializationType,
    TrainingState,
    create_optimizer,
    create_scheduler,
    initialize_weights,
    clip_gradients,
    compute_gradient_norm,
    GradientAccumulator,
    ConvergenceDetector,
    MixedPrecisionManager
)

from .training_loop import (
    TrainingLoop,
    TrainingLoopConfig,
    TrainingMetrics,
    create_training_loop
)

# 策略层导入（整合 backend/lib 模块）
try:
    from backend.modules.training.strategies.three_stage_strategy import (
        ThreeStageStrategy,
        ThreeStageStrategyConfig,
        ThreeStagePhase,
        DPOLossCalculator,
        create_three_stage_strategy,
        get_three_stage_phases
    )
    STRATEGY_AVAILABLE = True
except ImportError:
    STRATEGY_AVAILABLE = False
    # 提供占位符以避免导入错误
    ThreeStageStrategy = None
    ThreeStageStrategyConfig = None
    ThreeStagePhase = None
    DPOLossCalculator = None
    create_three_stage_strategy = None
    get_three_stage_phases = None

__all__ = [
    # 核心训练器（业务层）
    'ThreeStageTrainer',
    'create_three_stage_trainer',
    
    # 训练策略（策略层）
    'ThreeStageStrategy',
    'ThreeStageStrategyConfig',
    'ThreeStagePhase',
    'DPOLossCalculator',
    'create_three_stage_strategy',
    'get_three_stage_phases',
    'STRATEGY_AVAILABLE',
    
    # 配置
    'ThreeStageConfig',
    'TrainingStage',
    'StageConfig',
    
    # 运行时
    'setup_model_and_tokenizer',
    'build_dataloaders',
    
    # 数据集
    'TextDataset',
    'SFTDataset',
    'DPODataset',
    
    # 优化器工具
    'OptimizerConfig',
    'OptimizerType',
    'SchedulerType',
    'InitializationType',
    'TrainingState',
    'create_optimizer',
    'create_scheduler',
    'initialize_weights',
    'clip_gradients',
    'compute_gradient_norm',
    'GradientAccumulator',
    'ConvergenceDetector',
    'MixedPrecisionManager',
    
    # 训练循环
    'TrainingLoop',
    'TrainingLoopConfig',
    'TrainingMetrics',
    'create_training_loop',
]

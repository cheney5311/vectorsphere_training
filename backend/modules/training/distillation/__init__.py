# -*- coding: utf-8 -*-
"""
知识蒸馏模块

提供完整的知识蒸馏功能：
- 配置管理：多场景配置支持
- 场景管理：标准、行业、边缘、多模态等场景
- 蒸馏服务：租户级任务管理、监控、调度
- 模型压缩：量化、剪枝、蒸馏组合

模块架构：
├── compression_config.py    - 配置类定义
├── knowledge_distillation.py - 核心训练器（调用策略层）
├── distillation_scenarios.py - 场景管理器
└── distillation_service.py   - 蒸馏服务

调用关系：
本模块 -> backend/modules/training/strategies (策略层)
       -> backend/lib/losses (损失函数层)
       -> backend/lib/distributed (分布式层)
       -> backend/lib/hardware (硬件层)
"""

# 配置类
from .compression_config import (
    # 核心配置类
    DistillationConfig,
    CompressionConfig,
    ScenarioDistillationConfig,
    DistributedDistillationConfig,
    AdaptiveDistillationConfig,
    DistillationTaskConfig,
    
    # 枚举类
    DistillationScenario,
    DistributedMode,
    AdaptiveMode,
    CompressionMethod,
    
    # 预设和模板
    DistillationPresets,
    
    # 监控和统计
    DistillationStats,
    DistillationMonitor,
    ConfigValidator,
    
    # 工具函数
    create_distillation_config,
    validate_config,
    compare_configs,
    recommend_config,
    print_config_summary,
)

# 核心训练器
from .knowledge_distillation import (
    # 枚举和数据类
    TrainerPhase,
    TrainerStats,
    TrainerHealthStatus,
    
    # 训练器和压缩器
    KnowledgeDistillationTrainer,
    ModelCompressor,
    
    # 工厂函数
    create_knowledge_distillation_trainer,
    create_model_compressor,
    create_distillation_trainer_with_strategy,
    create_trainer_from_preset,
    create_trainer_from_scenario,
    
    # 诊断和工具函数
    diagnose_trainer,
    diagnose_compressor,
    compare_trainers,
    estimate_training_resources,
    
    # 全局实例管理
    get_global_trainer,
    set_global_trainer,
    get_global_compressor,
    set_global_compressor,
    reset_globals,
)

# 场景管理器
from .distillation_scenarios import (
    # 统计和监控类
    ScenarioExecutionStats,
    ScenarioMonitor,
    
    # 场景处理器基类和具体实现
    DistillationScenarioHandler,
    StandardScenarioHandler,
    IndustryScenarioHandler,
    EdgeDeployScenarioHandler,
    MultimodalScenarioHandler,
    RealTimeScenarioHandler,
    ProgressiveScenarioHandler,
    SelfDistillationScenarioHandler,
    ContrastiveScenarioHandler,
    
    # 场景管理器
    DistillationScenarioManager,
    
    # 工具函数
    get_scenario_manager,
    reset_scenario_manager,
    create_scenario_handler,
    list_available_scenarios,
    get_scenario_description,
    recommend_distillation_scenario,
    prepare_scenario,
    get_strategy_for_scenario,
    post_process_model,
    create_distillation_task,
    get_task_status,
    update_task_progress,
    complete_task,
    fail_task,
    get_global_stats,
    get_scenario_stats,
    get_all_scenario_stats,
    diagnose_scenarios,
    print_scenario_summary,
    print_diagnosis,
    compare_scenarios,
    estimate_scenario_requirements,
)

# 蒸馏服务
from .distillation_service import (
    DistillationTaskStatus,
    DistillationTask,
    DistillationMetrics,
    DistillationReport,
    DistillationService,
    get_distillation_service,
)

__all__ = [
    # 核心配置类
    'DistillationConfig',
    'CompressionConfig',
    'ScenarioDistillationConfig',
    'DistributedDistillationConfig',
    'AdaptiveDistillationConfig',
    'DistillationTaskConfig',
    
    # 枚举类
    'DistillationScenario',
    'DistributedMode',
    'AdaptiveMode',
    'CompressionMethod',
    
    # 预设和模板
    'DistillationPresets',
    
    # 监控和统计
    'DistillationStats',
    'DistillationMonitor',
    'ConfigValidator',
    
    # 工具函数（配置层）
    'create_distillation_config',
    'validate_config',
    'compare_configs',
    'recommend_config',
    'print_config_summary',
    
    # 训练器枚举和数据类
    'TrainerPhase',
    'TrainerStats',
    'TrainerHealthStatus',
    
    # 核心训练器
    'KnowledgeDistillationTrainer',
    'ModelCompressor',
    
    # 工厂函数
    'create_knowledge_distillation_trainer',
    'create_model_compressor',
    'create_distillation_trainer_with_strategy',
    'create_trainer_from_preset',
    'create_trainer_from_scenario',
    
    # 诊断和工具函数
    'diagnose_trainer',
    'diagnose_compressor',
    'compare_trainers',
    'estimate_training_resources',
    
    # 全局实例管理
    'get_global_trainer',
    'set_global_trainer',
    'get_global_compressor',
    'set_global_compressor',
    'reset_globals',
    
    # 统计和监控类
    'ScenarioExecutionStats',
    'ScenarioMonitor',
    
    # 场景处理器
    'DistillationScenarioHandler',
    'StandardScenarioHandler',
    'IndustryScenarioHandler',
    'EdgeDeployScenarioHandler',
    'MultimodalScenarioHandler',
    'RealTimeScenarioHandler',
    'ProgressiveScenarioHandler',
    'SelfDistillationScenarioHandler',
    'ContrastiveScenarioHandler',
    'DistillationScenarioManager',
    
    # 场景管理工具函数
    'get_scenario_manager',
    'reset_scenario_manager',
    'create_scenario_handler',
    'list_available_scenarios',
    'get_scenario_description',
    'recommend_distillation_scenario',
    'prepare_scenario',
    'get_strategy_for_scenario',
    'post_process_model',
    'create_distillation_task',
    'get_task_status',
    'update_task_progress',
    'complete_task',
    'fail_task',
    'get_global_stats',
    'get_scenario_stats',
    'get_all_scenario_stats',
    'diagnose_scenarios',
    'print_scenario_summary',
    'print_diagnosis',
    'compare_scenarios',
    'estimate_scenario_requirements',
    
    # 蒸馏服务
    'DistillationTaskStatus',
    'DistillationTask',
    'DistillationMetrics',
    'DistillationReport',
    'DistillationService',
    'get_distillation_service',
]

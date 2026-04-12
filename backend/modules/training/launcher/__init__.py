"""训练启动器模块

统一训练入口，整合六层训练架构的所有能力：
- 行业模型训练
- 场景化训练  
- 分布式训练
- 知识蒸馏训练（多场景）
- 多模态训练
- 三阶段训练
- 标准训练

架构调用关系：
┌─────────────────────────────────────────────────────────────┐
│  launcher (本模块 - 统一入口)                                │
│  ├── ProductionTrainingLauncher (生产级入口)                 │
│  │   └── DistributedTrainingManager (分布式管理)             │
│  │       ├── orchestrator (编排层)                          │
│  │       ├── pipeline (流水线)                              │
│  │       └── progress (进度管理)                            │
│  └── TrainingSystemLauncher (基础入口)                       │
│      ├── _create_industry_trainer()                         │
│      │   └── orchestrator/TrainingOrchestrator              │
│      │       └── strategies/*                               │
│      ├── _create_scenario_manager()                         │
│      │   └── scenarios/IndustryScenarioTrainer              │
│      │       └── strategies/ScenarioStrategy                │
│      ├── _create_distributed_trainer()                      │
│      │   └── strategies/DistributedStrategy                 │
│      ├── _create_distillation_trainer()                     │
│      │   └── distillation/KnowledgeDistillationTrainer      │
│      │       └── strategies/DistillationStrategy            │
│      ├── _create_multimodal_trainer()                       │
│      │   └── multimodal/MultiModalTrainer                   │
│      │       └── strategies/MultiModalStrategy              │
│      └── _create_three_stage_trainer()                      │
│          └── three_stage/ThreeStageTrainer                  │
│                                                             │
│  所有策略层(strategies/*) 调用:                              │
│  └── backend/lib/*                                          │
│      ├── hardware    (硬件抽象层)                           │
│      ├── distributed (分布式训练内核层)                     │
│      ├── adapters    (模型/模态适配器层)                    │
│      └── losses      (目标函数层)                           │
└─────────────────────────────────────────────────────────────┘

使用示例：

1. 基础训练：
```python
from backend.modules.training.launcher import launch_training_system

config = {
    'model': {'name': 'gpt2'},
    'training': {'num_epochs': 10, 'batch_size': 16}
}
result = launch_training_system(config)
```

2. 生产级训练：
```python
from backend.modules.training.launcher import (
    ProductionTrainingLauncher,
    create_production_training_config
)

config = create_production_training_config(
    training_type='three_stage',
    use_orchestrator=True,
    enable_checkpoint=True
)
launcher = ProductionTrainingLauncher(config)
result = launcher.launch_training(model, train_loader)
```

3. 使用流水线：
```python
from backend.modules.training.launcher import launch_production_training

config = create_production_training_config(
    training_type='industry',
    pipeline_steps=[
        {'name': 'pretrain', 'type': 'pretrain', 'params': {'epochs': 3}},
        {'name': 'finetune', 'type': 'finetune', 'params': {'epochs': 5}},
    ]
)
result = launch_production_training(config)
```
"""

from .training_launcher import (
    # 基础训练启动器
    TrainingSystemLauncher,
    launch_training_system,
    
    # 生产级训练启动器
    ProductionTrainingLauncher,
    launch_production_training,
    
    # 分布式训练管理器
    DistributedTrainingManager,
    
    # 配置创建函数
    create_industry_training_config,
    create_distillation_training_config,
    create_production_training_config,
    create_scenario_training_config,
    create_pipeline_training_config,
    create_orchestrator_training_config,
    
    # 诊断和信息函数
    get_module_availability,
    diagnose_launcher_module,
    diagnose_all_modules,
    get_all_training_modes,
    
    # 快速任务创建函数（调用所有模块）
    create_quick_training_task,
    execute_quick_pipeline,
    setup_training_progress,
    setup_plugin_system,
    create_training_strategies,
    setup_distributed_environment,
    setup_loss_functions,
    create_industry_training_task,
    create_three_stage_training_task,
    create_multimodal_training_task,
    create_distillation_training_task,
    get_hardware_status,
)

__all__ = [
    # 基础训练启动器
    'TrainingSystemLauncher',
    'launch_training_system',
    
    # 生产级训练启动器
    'ProductionTrainingLauncher',
    'launch_production_training',
    
    # 分布式训练管理器
    'DistributedTrainingManager',
    
    # 配置创建函数
    'create_industry_training_config',
    'create_distillation_training_config',
    'create_production_training_config',
    'create_scenario_training_config',
    'create_pipeline_training_config',
    'create_orchestrator_training_config',
    
    # 诊断和信息函数
    'get_module_availability',
    'diagnose_launcher_module',
    'diagnose_all_modules',
    'get_all_training_modes',
    
    # 快速任务创建函数（调用所有模块）
    'create_quick_training_task',
    'execute_quick_pipeline',
    'setup_training_progress',
    'setup_plugin_system',
    'create_training_strategies',
    'setup_distributed_environment',
    'setup_loss_functions',
    'create_industry_training_task',
    'create_three_stage_training_task',
    'create_multimodal_training_task',
    'create_distillation_training_task',
    'get_hardware_status',
]
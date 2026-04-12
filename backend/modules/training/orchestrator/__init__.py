"""训练编排器模块

生产级训练编排，提供：
- 统一训练编排器（六层架构整合）
- 传统训练编排器（多阶段训练）
- 训练计划创建和管理
- 层管理和协调

架构位置：
├── orchestrator/ (本模块)
│   ├── unified_orchestrator.py - 六层架构统一编排
│   ├── training_orchestrator.py - 传统训练编排
│   └── 集成 strategies/, lib/, pipeline/
├── 作为训练平台的核心编排层
└── 被 services, launcher 调用

使用示例：
from backend.modules.training.orchestrator import (
        create_orchestrator,
        create_quick_plan,
    UnifiedTrainingOrchestrator,
)

# 创建编排器
    orchestrator = create_orchestrator(output_dir="./outputs")

# 创建训练计划
plan = orchestrator.create_three_stage_plan(
    name="my_training",
    pretrain_epochs=3,
    finetune_epochs=5,
)

# 执行训练
results = orchestrator.execute(plan, model, train_loader)
"""

# 统一编排器
from .unified_orchestrator import (
    # 主类
    UnifiedTrainingOrchestrator,
    LayerManager,
    
    # 配置类
    LayerConfig,
    PhaseConfig,
    OrchestratorPlan,
    PhaseResult,
    
    # 枚举
    OrchestratorPhase,
    OrchestratorStatus,
    
    # 工厂函数
    create_orchestrator,
    create_quick_plan,
    diagnose_orchestrator_module,
)

# 传统编排器
try:
    from .training_orchestrator import (
        TrainingOrchestrator,
        TrainingPlan,
        TrainingPhase as TrainingOrchestratorPhase,
        TrainingStatus as TrainingOrchestratorStatus,
    )
    TRAINING_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    TrainingOrchestrator = None
    TrainingPlan = None
    TrainingOrchestratorPhase = None
    TrainingOrchestratorStatus = None
    TRAINING_ORCHESTRATOR_AVAILABLE = False


# ==================== 便捷函数 ====================

def create_standard_plan(
    name: str = "standard_training",
    epochs: int = 10,
    learning_rate: float = 1e-4,
) -> OrchestratorPlan:
    """创建标准训练计划"""
    orchestrator = UnifiedTrainingOrchestrator()
    return orchestrator.create_standard_plan(
        name=name,
        epochs=epochs,
        learning_rate=learning_rate,
    )


def create_three_stage_plan(
    name: str = "three_stage_training",
    pretrain_epochs: int = 3,
    finetune_epochs: int = 5,
    preference_epochs: int = 2,
) -> OrchestratorPlan:
    """创建三阶段训练计划"""
    orchestrator = UnifiedTrainingOrchestrator()
    return orchestrator.create_three_stage_plan(
        name=name,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=finetune_epochs,
        preference_epochs=preference_epochs,
    )


def create_multimodal_plan(
    name: str = "multimodal_training",
    modalities: list = None,
) -> OrchestratorPlan:
    """创建多模态训练计划"""
    orchestrator = UnifiedTrainingOrchestrator()
    return orchestrator.create_multimodal_plan(
        name=name,
        modalities=modalities,
    )


def create_distillation_plan(
    name: str = "distillation_training",
    distillation_epochs: int = 10,
) -> OrchestratorPlan:
    """创建知识蒸馏计划"""
    orchestrator = UnifiedTrainingOrchestrator()
    return orchestrator.create_distillation_plan(
        name=name,
        distillation_epochs=distillation_epochs,
    )


def create_industry_plan(
    name: str = "industry_training",
    include_pretrain: bool = True,
    include_align: bool = True,
    include_finetune: bool = True,
) -> OrchestratorPlan:
    """创建行业模型训练计划"""
    orchestrator = UnifiedTrainingOrchestrator()
    return orchestrator.create_industry_plan(
        name=name,
        include_pretrain=include_pretrain,
        include_align=include_align,
        include_finetune=include_finetune,
    )


def get_orchestrator_layer_availability() -> dict:
    """获取编排器模块层可用性"""
    availability = {
        'unified_orchestrator': True,
        'training_orchestrator': TRAINING_ORCHESTRATOR_AVAILABLE,
    }
    
    # 获取统一编排器的层可用性
    try:
        from .unified_orchestrator import diagnose_orchestrator_module
        diagnosis = diagnose_orchestrator_module()
        availability['layers'] = diagnosis.get('layer_availability', {})
    except Exception:
        availability['layers'] = {}
    
    return availability


# ==================== 导出 ====================

__all__ = [
    # 统一编排器
    'UnifiedTrainingOrchestrator',
    'LayerManager',
    'LayerConfig',
    'PhaseConfig',
    'OrchestratorPlan',
    'PhaseResult',
    'OrchestratorPhase',
    'OrchestratorStatus',
    'create_orchestrator',
    'create_quick_plan',
    'diagnose_orchestrator_module',
    
    # 传统编排器
    'TrainingOrchestrator',
    'TrainingPlan',
    'TrainingOrchestratorPhase',
    'TrainingOrchestratorStatus',
    
    # 便捷函数
    'create_standard_plan',
    'create_three_stage_plan',
    'create_multimodal_plan',
    'create_distillation_plan',
    'create_industry_plan',
    'get_orchestrator_layer_availability',
    
    # 可用性标志
    'TRAINING_ORCHESTRATOR_AVAILABLE',
]

"""训练核心模块初始化文件

提供统一训练系统、任务管理和版本控制功能。

架构说明:
- core 模块不直接调用 backend/services，正确的调用层级是:
  backend/services -> backend/modules/training/launcher -> backend/modules/training/core -> 下游训练模块
- 服务层调用由 launcher 层处理

集成模块：
- UnifiedTrainingSystem: 统一训练系统，整合所有训练组件
- TrainingTaskManager: 任务管理器，管理训练任务生命周期
- VersionManager: 版本管理器，支持训练结果回滚
- 各种训练配置和数据类
"""

from .unified_training_system import (
    # 主类
    UnifiedTrainingSystem,
    TrainingConfig,
    TrainingVersion,
    TrainingVersionManager,
    # 便捷函数
    create_training_config,
    launch_unified_training,
    get_integration_availability as get_system_integration_availability,
    diagnose_training_system,
    # 可用性标志
    TORCH_AVAILABLE,
    MULTIMODAL_AVAILABLE,
    DISTRIBUTED_AVAILABLE,
    DISTILLATION_AVAILABLE,
    THREE_STAGE_AVAILABLE,
    SCENARIOS_AVAILABLE,
    INDUSTRY_AVAILABLE,
    ORCHESTRATOR_AVAILABLE,
    PROGRESS_AVAILABLE as SYSTEM_PROGRESS_AVAILABLE,
    PIPELINE_AVAILABLE as SYSTEM_PIPELINE_AVAILABLE,
    PLUGINS_AVAILABLE as SYSTEM_PLUGINS_AVAILABLE,
    MONITORING_AVAILABLE,
    STRATEGY_AVAILABLE,
    DISTRIBUTED_STRATEGY_AVAILABLE,
    HARDWARE_AVAILABLE,
    LOSSES_AVAILABLE,
    TASK_MANAGER_AVAILABLE
)

from .task_manager import (
    # 主类
    TrainingTaskManager,
    TrainingTask,
    TrainingTaskStatus,
    TaskVersion,
    VersionManager,
    # 便捷函数
    get_training_task_manager,
    shutdown_training_task_manager,
    diagnose_task_manager,
)


def diagnose_core_module() -> dict:
    """诊断核心模块状态
    
    Returns:
        诊断信息
    """
    return {
        'unified_training_system': diagnose_training_system(),
        'task_manager': diagnose_task_manager(),
    }


__all__ = [
    # UnifiedTrainingSystem 相关
    'UnifiedTrainingSystem',
    'TrainingConfig',
    'TrainingVersion',
    'TrainingVersionManager',
    'create_training_config',
    'launch_unified_training',
    'get_system_integration_availability',
    'diagnose_training_system',

    # TaskManager 相关
    'TrainingTaskManager',
    'TrainingTask',
    'TrainingTaskStatus',
    'TaskVersion',
    'VersionManager',
    'get_training_task_manager',
    'shutdown_training_task_manager',
    'diagnose_task_manager',

    # 综合功能
    'diagnose_core_module',

    # 可用性标志 - 系统级（下游训练模块）
    'TORCH_AVAILABLE',
    'MULTIMODAL_AVAILABLE',
    'DISTRIBUTED_AVAILABLE',
    'DISTILLATION_AVAILABLE',
    'THREE_STAGE_AVAILABLE',
    'SCENARIOS_AVAILABLE',
    'INDUSTRY_AVAILABLE',
    'ORCHESTRATOR_AVAILABLE',
    'SYSTEM_PROGRESS_AVAILABLE',
    'SYSTEM_PIPELINE_AVAILABLE',
    'SYSTEM_PLUGINS_AVAILABLE',
    'MONITORING_AVAILABLE',
    'STRATEGY_AVAILABLE',
    'DISTRIBUTED_STRATEGY_AVAILABLE',
    'HARDWARE_AVAILABLE',
    'LOSSES_AVAILABLE',
    'TASK_MANAGER_AVAILABLE',
]

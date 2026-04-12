"""训练场景模块

提供生产级多种训练场景支持：
- 基础模型训练场景
- 高级模型训练场景
- 定时训练场景
- 行业特定场景

架构层次（调用关系）：
┌─────────────────────────────────────────────────────┐
│  scenarios (本模块)                                  │
│  ├── BaseScenario: 基础场景抽象类                    │
│  │   ├── 调用 strategies/base_strategy (策略层)      │
│  │   ├── 调用 backend/lib/hardware (硬件层)          │
│  │   ├── 调用 backend/lib/distributed (分布式层)     │
│  │   ├── 调用 progress/progress_manager (进度管理)   │
│  │   ├── 调用 orchestrator/* (编排层)                │
│  │   ├── 调用 pipeline/* (流水线层)                  │
│  │   └── 调用 plugins/* (插件层)                     │
│  │                                                   │
│  ├── BasicModelScenario: 基础模型训练                │
│  │   └── 支持 orchestrator/pipeline/plugins 集成     │
│  ├── AdvancedModelScenario: 高级模型训练             │
│  │   └── 支持 orchestrator/pipeline/plugins 集成     │
│  ├── ScheduledTrainingScenario: 定时训练             │
│  │   └── 支持 orchestrator/pipeline/plugins 集成     │
│  │                                                   │
│  └── IndustryScenario: 行业场景训练                  │
│      └── IndustryScenarioTrainer                     │
│          ├── strategies/scenario_strategy (场景策略) │
│          └── 支持 orchestrator/pipeline/plugins 集成 │
└─────────────────────────────────────────────────────┘

调用关系：
本模块 -> orchestrator/* (编排层)
       -> pipeline/* (流水线层)
       -> plugins/* (插件层)
       -> strategies/* (策略层) -> backend/lib/* (底层六层架构)
"""

# ==================== 基础场景 ====================

from .base_scenario import (
    # 基础类
    BaseScenario,
    
    # 枚举
    TrainingStage,
    TrainingScenario,
    ScenarioStatus,
    
    # 配置和结果类
    ScenarioConfigBase,
    ScenarioResult,
    
    # 便捷函数
    get_layer_availability,
    get_available_scenarios,
    get_available_stages,
    create_scenario_result,
    
    # 层可用性标志
    STRATEGY_LAYER_AVAILABLE,
    DISTRIBUTED_STRATEGY_AVAILABLE,
    SCENARIO_STRATEGY_AVAILABLE,
    HARDWARE_LAYER_AVAILABLE,
    DISTRIBUTED_LAYER_AVAILABLE,
    PROGRESS_MANAGER_AVAILABLE,
)

# 编排器、流水线、插件层可用性
try:
    from .base_scenario import (
        ORCHESTRATOR_AVAILABLE,
        PIPELINE_AVAILABLE,
        PLUGINS_AVAILABLE,
    )
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    PIPELINE_AVAILABLE = False
    PLUGINS_AVAILABLE = False

# ==================== 场景管理器 ====================

from .scenario_manager import (
    # 管理器
    ScenarioManager,
    get_scenario_manager,
    shutdown_scenario_manager,
    
    # 数据类
    ScenarioConfig,
    TrainingJob,
    TrainingJobWrapper,
    
    # 枚举
    JobStatus,
)

# ==================== 基础模型场景 ====================

from .basic_model_scenario import (
    BasicModelScenario,
    BasicModelConfig,
    create_basic_scenario,
    get_basic_scenario_presets,
)

# ==================== 高级模型场景 ====================

from .advanced_model_scenario import (
    AdvancedModelScenario,
    AdvancedModelConfig,
    AdvancedModelType,
    AdvancedTrainingMode,
    create_advanced_scenario,
    get_advanced_scenario_presets,
)

# ==================== 定时训练场景 ====================

from .scheduled_training_scenario import (
    ScheduledTrainingScenario,
    ScheduledTrainingConfig,
    ScheduleConfig,
    ScheduleType,
    TriggerCondition,
    create_scheduled_scenario,
    get_scheduled_scenario_presets,
)

# ==================== 行业场景 ====================

from .industry_scenario import (
    # 场景类
    IndustryScenario,
    IndustryScenarioTrainer,
    
    # 配置类
    IndustryScenarioConfig,
    
    # 枚举
    IndustryScenarioType,
    
    # 模型
    TimeSeriesModel,
    ImageModel,
    TabularModel,
    
    # 预设
    IndustryScenarioPresets,
    
    # 便捷函数
    create_industry_scenario,
    get_preset_scenario,
    diagnose_industry_scenario,
)

# ==================== 从 backend.schemas.enums 导入枚举 ====================

try:
    from backend.schemas.enums import TrainingPriority
except ImportError:
    TrainingPriority = None


# ==================== 诊断函数 ====================

def diagnose_scenarios() -> dict:
    """诊断场景模块状态
    
    Returns:
        诊断结果字典
    """
    return {
        'layer_availability': get_layer_availability(),
        'available_scenarios': get_available_scenarios(),
        'available_stages': get_available_stages(),
        'industry_diagnosis': diagnose_industry_scenario(),
        'basic_presets': list(get_basic_scenario_presets().keys()),
        'advanced_presets': list(get_advanced_scenario_presets().keys()),
        'scheduled_presets': list(get_scheduled_scenario_presets().keys()),
        'industry_presets': list(IndustryScenarioPresets.get_all_presets().keys()),
        # 集成模块可用性
        'integration_modules': {
            'orchestrator': ORCHESTRATOR_AVAILABLE,
            'pipeline': PIPELINE_AVAILABLE,
            'plugins': PLUGINS_AVAILABLE,
        },
    }


def get_integration_availability() -> dict:
    """获取集成模块可用性
    
    Returns:
        各集成模块的可用状态
    """
    return {
        'orchestrator': ORCHESTRATOR_AVAILABLE,
        'pipeline': PIPELINE_AVAILABLE,
        'plugins': PLUGINS_AVAILABLE,
        'strategy_layer': STRATEGY_LAYER_AVAILABLE,
        'distributed_strategy': DISTRIBUTED_STRATEGY_AVAILABLE,
        'scenario_strategy': SCENARIO_STRATEGY_AVAILABLE,
        'hardware_layer': HARDWARE_LAYER_AVAILABLE,
        'distributed_layer': DISTRIBUTED_LAYER_AVAILABLE,
        'progress_manager': PROGRESS_MANAGER_AVAILABLE,
    }


# ==================== 导出 ====================

__all__ = [
    # ===== 基础场景 =====
    'BaseScenario',
    'TrainingStage',
    'TrainingScenario',
    'ScenarioStatus',
    'ScenarioConfigBase',
    'ScenarioResult',
    
    # ===== 场景管理器 =====
    'ScenarioManager',
    'ScenarioConfig',
    'TrainingJob',
    'TrainingJobWrapper',
    'JobStatus',
    'get_scenario_manager',
    'shutdown_scenario_manager',
    
    # ===== 基础模型场景 =====
    'BasicModelScenario',
    'BasicModelConfig',
    'create_basic_scenario',
    'get_basic_scenario_presets',
    
    # ===== 高级模型场景 =====
    'AdvancedModelScenario',
    'AdvancedModelConfig',
    'AdvancedModelType',
    'AdvancedTrainingMode',
    'create_advanced_scenario',
    'get_advanced_scenario_presets',
    
    # ===== 定时训练场景 =====
    'ScheduledTrainingScenario',
    'ScheduledTrainingConfig',
    'ScheduleConfig',
    'ScheduleType',
    'TriggerCondition',
    'create_scheduled_scenario',
    'get_scheduled_scenario_presets',
    
    # ===== 行业场景 =====
    'IndustryScenario',
    'IndustryScenarioConfig',
    'IndustryScenarioType',
    'IndustryScenarioTrainer',
    'IndustryScenarioPresets',
    'TimeSeriesModel',
    'ImageModel',
    'TabularModel',
    'create_industry_scenario',
    'get_preset_scenario',
    'diagnose_industry_scenario',
    
    # ===== 便捷函数 =====
    'get_layer_availability',
    'get_available_scenarios',
    'get_available_stages',
    'create_scenario_result',
    'diagnose_scenarios',
    'get_integration_availability',
    
    # ===== 枚举 =====
    'TrainingPriority',
    
    # ===== 层可用性标志 =====
    'STRATEGY_LAYER_AVAILABLE',
    'DISTRIBUTED_STRATEGY_AVAILABLE',
    'SCENARIO_STRATEGY_AVAILABLE',
    'HARDWARE_LAYER_AVAILABLE',
    'DISTRIBUTED_LAYER_AVAILABLE',
    'PROGRESS_MANAGER_AVAILABLE',
    
    # ===== 集成模块可用性标志 =====
    'ORCHESTRATOR_AVAILABLE',
    'PIPELINE_AVAILABLE',
    'PLUGINS_AVAILABLE',
]

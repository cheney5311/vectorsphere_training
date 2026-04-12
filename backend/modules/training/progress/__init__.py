"""训练进度管理模块

提供训练进度跟踪和状态管理功能：
- TrainingProgress: 进度数据类
- TrainingProgressManager: 进度管理器
- 系统资源监控
- 策略层和硬件层集成

架构位置：
┌─────────────────────────────────────────────────────────────┐
│  orchestrator / pipeline / scenarios                         │
│    └── >>> progress (本模块) <<<                             │
│        ├── TrainingProgressManager (进度管理)                │
│        ├── TrainingProgress (进度数据)                       │
│        └── SystemMetrics (系统指标)                          │
│            └── strategies/* (策略层)                         │
│                └── backend/lib/* (底层)                      │
└─────────────────────────────────────────────────────────────┘
"""

from .progress_manager import (
    # 主要类
    TrainingProgressManager,
    TrainingProgress,
    SystemMetrics,
    
    # 枚举
    ProgressStatus,
    TrainingStageType,
    
    # 全局实例管理
    get_progress_manager,
    reset_progress_manager,
    
    # 便捷函数
    create_progress_tracker,
    update_progress,
    get_progress,
    get_layer_availability,
    
    # 层可用性标志
    STRATEGY_LAYER_AVAILABLE,
    HARDWARE_LAYER_AVAILABLE,
    DISTRIBUTED_LAYER_AVAILABLE,
)

__all__ = [
    # 主要类
    'TrainingProgressManager',
    'TrainingProgress',
    'SystemMetrics',
    
    # 枚举
    'ProgressStatus',
    'TrainingStageType',
    
    # 全局实例管理
    'get_progress_manager',
    'reset_progress_manager',
    
    # 便捷函数
    'create_progress_tracker',
    'update_progress',
    'get_progress',
    'get_layer_availability',
    
    # 层可用性标志
    'STRATEGY_LAYER_AVAILABLE',
    'HARDWARE_LAYER_AVAILABLE',
    'DISTRIBUTED_LAYER_AVAILABLE',
]

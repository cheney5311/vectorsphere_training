# -*- coding: utf-8 -*-
"""
Distributed Training Core Layer（分布式训练内核层）

提供分布式训练的核心功能，包括：
- DDP (Data Distributed Parallel)
- FSDP (Fully Sharded Data Parallel)
- Pipeline Parallel
- ZeRO (DeepSpeed)
- Tensor Parallel
- 混合并行策略

架构位置：
┌──────────────────────────────────────┐
│          Training Orchestrator       │
├──────────────────────────────────────┤
│       Training Strategy Abstraction  │
├──────────────────────────────────────┤
│      Loss & Objective Composition    │
├──────────────────────────────────────┤
│      Model & Modality Adapter Layer  │
├──────────────────────────────────────┤
│  >>> Distributed Training Core <<<   │  ← 当前层
├──────────────────────────────────────┤
│          Hardware Abstraction        │
└──────────────────────────────────────┘

使用示例:
```python
from backend.modules.training.distributed import (
    # 分布式管理器
    DistributedManager,
    get_distributed_manager,
    
    # 并行模式
    ParallelMode,
    DDPWrapper,
    FSDPWrapper,
    PipelineWrapper,
    ZeROWrapper,
    
    # 通信
    CommunicationBackend,
    AllReduceOp,
    
    # 配置
    DistributedConfig,
    DDPConfig,
    FSDPConfig,
    PipelineConfig,
    ZeROConfig
)

# 初始化分布式
manager = get_distributed_manager()
manager.initialize(world_size=8, backend='nccl')

# 包装模型
model = manager.wrap_model(model, mode=ParallelMode.FSDP)

# 同步梯度
manager.sync_gradients(model)
```
"""

import logging

logger = logging.getLogger(__name__)

# ==================== 分布式模式和配置 ====================

from .parallel_modes import (
    # 枚举类型
    ParallelMode,
    CommunicationBackend,
    ShardingStrategy,
    
    # 配置类
    DistributedConfig,
    DDPConfig,
    FSDPConfig,
    PipelineConfig,
    ZeROConfig,
    TensorParallelConfig,
    HybridParallelConfig,
    
    # 异常
    ConfigValidationError,
    
    # 工厂函数
    create_distributed_config,
    auto_select_parallel_mode,
    get_optimal_config,
    
    # 便捷函数
    print_config_summary,
    validate_all_configs,
)

# ==================== DDP 数据并行 ====================

from .ddp_wrapper import (
    # 包装器
    DDPWrapper,
    DDPContext,
    
    # 枚举和配置
    DDPSyncMode,
    DDPReduceOp,
    DDPState,
    
    # 组件
    DDPMemoryMonitor,
    DDPProfiler,
    DDPCommunicationAnalyzer,
    
    # 便捷函数
    create_ddp_model,
    ddp_reduce_gradients,
    ddp_context,
    auto_configure_ddp,
    estimate_ddp_memory,
    print_ddp_info,
    compare_ddp_configs,
    
    # 工具函数
    get_ddp_rank,
    get_ddp_world_size,
    get_ddp_local_rank,
    is_ddp_main_process,
    ddp_barrier,
    ddp_all_reduce,
    ddp_broadcast,
    ddp_all_gather_object,
    average_gradients,
)

# ==================== FSDP 全分片数据并行 ====================

from .fsdp_wrapper import (
    # 包装器
    FSDPWrapper,
    FSDPContext,
    
    # 枚举和配置
    FSDPShardingStrategy,
    FSDPMixedPrecisionConfig,
    FSDPCheckpointConfig,
    FSDPMemoryConfig,
    
    # 组件
    FSDPMemoryMonitor,
    FSDPProfiler,
    FSDPActivationCheckpointing,
    
    # 便捷函数
    create_fsdp_model,
    fsdp_context,
    apply_fsdp_activation_checkpointing,
    get_fsdp_memory_stats,
    estimate_fsdp_memory,
    auto_configure_fsdp,
)

# ==================== Pipeline 流水线并行 ====================

from .pipeline_wrapper import (
    # 包装器
    PipelineWrapper,
    
    # 枚举和配置
    PipelineSchedule,
    PipelineStageConfig,
    MicroBatchState,
    
    # 调度器
    BaseScheduler,
    GPipeSchedule,
    OneFOneBSchedule,
    InterleavedSchedule,
    
    # 组件
    PipelineCommunicator,
    PipelineMemoryManager,
    PipelineProfiler,
    
    # 便捷函数
    create_pipeline_model,
    pipeline_context,
    estimate_pipeline_efficiency,
    recommend_pipeline_config,
    visualize_pipeline_schedule,
)

# ==================== ZeRO 优化 ====================

from .zero_wrapper import (
    # 包装器
    ZeROWrapper,
    
    # 枚举和配置
    ZeROStage,
    OffloadDevice,
    ZeROOffloadConfig,
    ZeROCommunicationConfig,
    ZeROMixedPrecisionConfig,
    ZeROConfig as ZeROOptConfig,
    
    # 组件
    ZeROMemoryMonitor,
    ZeROProfiler,
    ZeROTrainingState,
    
    # 便捷函数
    create_zero_optimizer,
    zero_context,
    auto_configure_zero,
    estimate_zero_memory,
    get_zero_stage_description,
    compare_zero_stages,
    print_zero_comparison,
)

# ==================== 分布式管理器 ====================

from .distributed_manager import (
    # 管理器
    DistributedManager,
    DistributedState,
    
    # 枚举和配置
    AllReduceOp,
    HealthStatus,
    ProcessGroupConfig,
    
    # 组件
    DistributedHealthMonitor,
    CommunicationProfiler,
    
    # 全局函数
    get_distributed_manager,
    reset_distributed_manager,
    init_distributed,
    cleanup_distributed,
    is_distributed_initialized,
    get_rank,
    get_world_size,
    get_local_rank,
    is_main_process,
    
    # 通信操作
    broadcast,
    all_reduce,
    all_gather,
    reduce_scatter,
    barrier,
    synchronize,
    gather_metrics,
    average_metrics,
    all_gather_object,
    
    # 装饰器
    main_process_only,
    synchronized,
    
    # 上下文管理器
    distributed_context,
    managed_distributed,
    only_on_main_process,
    
    # 工具函数
    print_on_main,
    save_on_main,
    auto_select_backend,
    estimate_communication_cost,
    get_optimal_bucket_size,
    create_custom_process_group,
    log_distributed_metrics,
    check_distributed_consistency,
    synchronize_random_seed,
    get_distributed_sampler,
)


__all__ = [
    # 枚举类型
    'ParallelMode',
    'CommunicationBackend',
    'ShardingStrategy',
    
    # 配置类
    'DistributedConfig',
    'DDPConfig',
    'FSDPConfig',
    'PipelineConfig',
    'ZeROConfig',
    'TensorParallelConfig',
    'HybridParallelConfig',
    
    # 异常
    'ConfigValidationError',
    
    # 工厂函数
    'create_distributed_config',
    'auto_select_parallel_mode',
    'get_optimal_config',
    
    # 便捷函数
    'print_config_summary',
    'validate_all_configs',
    
    # DDP
    'DDPWrapper',
    'DDPContext',
    'DDPSyncMode',
    'DDPReduceOp',
    'DDPState',
    'DDPMemoryMonitor',
    'DDPProfiler',
    'DDPCommunicationAnalyzer',
    'create_ddp_model',
    'ddp_reduce_gradients',
    'ddp_context',
    'auto_configure_ddp',
    'estimate_ddp_memory',
    'print_ddp_info',
    'compare_ddp_configs',
    'get_ddp_rank',
    'get_ddp_world_size',
    'get_ddp_local_rank',
    'is_ddp_main_process',
    'ddp_barrier',
    'ddp_all_reduce',
    'ddp_broadcast',
    'ddp_all_gather_object',
    'average_gradients',
    
    # FSDP
    'FSDPWrapper',
    'FSDPContext',
    'FSDPShardingStrategy',
    'FSDPMixedPrecisionConfig',
    'FSDPCheckpointConfig',
    'FSDPMemoryConfig',
    'FSDPMemoryMonitor',
    'FSDPProfiler',
    'FSDPActivationCheckpointing',
    'create_fsdp_model',
    'fsdp_context',
    'apply_fsdp_activation_checkpointing',
    'get_fsdp_memory_stats',
    'estimate_fsdp_memory',
    'auto_configure_fsdp',
    
    # Pipeline
    'PipelineWrapper',
    'PipelineSchedule',
    'PipelineStageConfig',
    'MicroBatchState',
    'BaseScheduler',
    'GPipeSchedule',
    'OneFOneBSchedule',
    'InterleavedSchedule',
    'PipelineCommunicator',
    'PipelineMemoryManager',
    'PipelineProfiler',
    'create_pipeline_model',
    'pipeline_context',
    'estimate_pipeline_efficiency',
    'recommend_pipeline_config',
    'visualize_pipeline_schedule',
    
    # ZeRO
    'ZeROWrapper',
    'ZeROStage',
    'OffloadDevice',
    'ZeROOffloadConfig',
    'ZeROCommunicationConfig',
    'ZeROMixedPrecisionConfig',
    'ZeROOptConfig',
    'ZeROMemoryMonitor',
    'ZeROProfiler',
    'ZeROTrainingState',
    'create_zero_optimizer',
    'zero_context',
    'auto_configure_zero',
    'estimate_zero_memory',
    'get_zero_stage_description',
    'compare_zero_stages',
    'print_zero_comparison',
    
    # 管理器
    'DistributedManager',
    'DistributedState',
    'get_distributed_manager',
    'reset_distributed_manager',
    'init_distributed',
    'cleanup_distributed',
    
    # 枚举和配置
    'HealthStatus',
    'ProcessGroupConfig',
    
    # 组件
    'DistributedHealthMonitor',
    'CommunicationProfiler',
    
    # 全局函数
    'is_distributed_initialized',
    'get_rank',
    'get_world_size',
    'get_local_rank',
    'is_main_process',
    
    # 通信操作
    'AllReduceOp',
    'broadcast',
    'all_reduce',
    'all_gather',
    'reduce_scatter',
    'barrier',
    'synchronize',
    'gather_metrics',
    'average_metrics',
    'all_gather_object',
    
    # 装饰器
    'main_process_only',
    'synchronized',
    
    # 上下文管理器
    'distributed_context',
    'managed_distributed',
    'only_on_main_process',
    
    # 工具函数
    'print_on_main',
    'save_on_main',
    'auto_select_backend',
    'estimate_communication_cost',
    'get_optimal_bucket_size',
    'create_custom_process_group',
    'log_distributed_metrics',
    'check_distributed_consistency',
    'synchronize_random_seed',
    'get_distributed_sampler',
]


# -*- coding: utf-8 -*-
"""
Hardware Abstraction Layer（硬件抽象层）

统一管理训练所需的硬件资源，包括：
- 设备检测（GPU/NPU/TPU/CPU）
- 内存管理
- 混合精度训练
- 设备调度
- 多设备协调

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
│        Distributed Training Core     │
├──────────────────────────────────────┤
│   >>> Hardware Abstraction <<<       │  ← 当前层
└──────────────────────────────────────┘

使用示例:
```python
from backend.modules.training.hardware import (
    # 设备管理
    DeviceManager,
    get_device_manager,
    
    # 设备类型
    DeviceType,
    DeviceInfo,
    
    # 内存管理
    MemoryManager,
    MemoryOptimizer,
    
    # 混合精度
    MixedPrecisionManager,
    AmpContext,
    
    # 设备调度
    DeviceScheduler,
    DevicePool
)

# 获取最佳设备
device_manager = get_device_manager()
device = device_manager.get_best_device()

# 检查GPU内存
memory_info = device_manager.get_memory_info()

# 混合精度上下文
with AmpContext() as amp:
    output = model(input)
    loss = criterion(output, target)
    amp.backward(loss)
```
"""

import logging

logger = logging.getLogger(__name__)

# ==================== 设备类型和信息 ====================

from .device_types import (
    # 枚举类型
    DeviceType,
    PrecisionType,
    
    # 数据类
    DeviceInfo,
    DeviceCapabilities,
    HardwareConfig,
    
    # 工具类
    DeviceSelector,
    DeviceComparator,
    
    # 工具函数
    parse_device_string,
    create_device_string,
    estimate_model_memory,
    recommend_precision,
    validate_device_config,
    print_device_comparison,
    get_optimal_device_allocation,
)

# ==================== 设备管理 ====================

from .device_manager import (
    # 管理器
    DeviceManager,
    
    # 枚举
    DeviceStatus,
    AllocationStrategy,
    
    # 数据类和组件
    DeviceMetrics,
    DeviceMonitor,
    DeviceBenchmark,
    
    # 全局函数
    get_device_manager,
    detect_devices,
    select_device,
    to_device,
    
    # 工具函数
    get_optimal_device,
    get_all_available_devices,
    check_device_compatibility,
    estimate_device_capacity,
    print_device_info,
    clear_all_caches,
    reset_all_devices,
    synchronize_all_devices,
    get_device_count,
    is_device_available,
    
    # 上下文管理器
    managed_device,
    auto_device_allocation,
    device_monitoring,
)

# ==================== 内存管理 ====================

from .memory_manager import (
    # 管理器
    MemoryManager,
    MemoryOptimizer,
    GradientCheckpointing,
    
    # 数据类和枚举
    MemoryStats,
    MemoryEvent,
    MemoryPressure,
    OptimizationStrategy,
    
    # 组件
    MemoryMonitor,
    MemoryProfiler,
    
    # 便捷函数
    clear_memory,
    get_memory_summary,
    
    # 工具函数
    get_available_memory,
    estimate_tensor_memory,
    compare_memory_usage,
    track_memory,
    managed_memory,
    optimize_model_memory,
    print_memory_report,
    recommend_batch_size,
    check_memory_health,
    emergency_memory_cleanup,
)

# ==================== 混合精度 ====================

from .mixed_precision import (
    # 管理器
    MixedPrecisionManager,
    AmpContext,
    get_amp_context,
    
    # 枚举和配置
    PrecisionMode,
    AmpConfig,
    
    # 组件
    PrecisionStats,
    PrecisionMonitor,
    PrecisionProfiler,
    
    # 便捷函数
    amp_autocast,
    cast_model_to_precision,
    
    # 工具函数
    convert_tensor_precision,
    analyze_model_precision,
    recommend_precision_mode,
    estimate_precision_speedup,
    create_precision_optimizer,
    compare_precision_modes,
    validate_precision_config,
    print_precision_info,
    auto_select_precision,
    managed_precision_training,
)

# ==================== 设备调度 ====================

from .device_scheduler import (
    DeviceScheduler,
    DevicePool,
    DeviceAllocation,
    AllocationStrategy
)


__all__ = [
    # 设备类型
    'DeviceType',
    'PrecisionType',
    'DeviceInfo',
    'DeviceCapabilities',
    'HardwareConfig',
    
    # 工具类
    'DeviceSelector',
    'DeviceComparator',
    
    # 工具函数
    'parse_device_string',
    'create_device_string',
    'estimate_model_memory',
    'recommend_precision',
    'validate_device_config',
    'print_device_comparison',
    'get_optimal_device_allocation',
    
    # 设备管理
    'DeviceManager',
    'DeviceStatus',
    'AllocationStrategy',
    'DeviceMetrics',
    'DeviceMonitor',
    'DeviceBenchmark',
    'get_device_manager',
    'detect_devices',
    'select_device',
    'to_device',
    'get_optimal_device',
    'get_all_available_devices',
    'check_device_compatibility',
    'estimate_device_capacity',
    'print_device_info',
    'clear_all_caches',
    'reset_all_devices',
    'synchronize_all_devices',
    'get_device_count',
    'is_device_available',
    'managed_device',
    'auto_device_allocation',
    'device_monitoring',
    
    # 内存管理
    'MemoryManager',
    'MemoryOptimizer',
    'MemoryStats',
    'MemoryEvent',
    'MemoryPressure',
    'OptimizationStrategy',
    'MemoryMonitor',
    'MemoryProfiler',
    'GradientCheckpointing',
    'clear_memory',
    'get_memory_summary',
    'get_available_memory',
    'estimate_tensor_memory',
    'compare_memory_usage',
    'track_memory',
    'managed_memory',
    'optimize_model_memory',
    'print_memory_report',
    'recommend_batch_size',
    'check_memory_health',
    'emergency_memory_cleanup',
    
    # 混合精度
    'MixedPrecisionManager',
    'AmpContext',
    'get_amp_context',
    'PrecisionMode',
    'AmpConfig',
    'PrecisionStats',
    'PrecisionMonitor',
    'PrecisionProfiler',
    'amp_autocast',
    'cast_model_to_precision',
    'convert_tensor_precision',
    'analyze_model_precision',
    'recommend_precision_mode',
    'estimate_precision_speedup',
    'create_precision_optimizer',
    'compare_precision_modes',
    'validate_precision_config',
    'print_precision_info',
    'auto_select_precision',
    'managed_precision_training',
    
    # 设备调度
    'DeviceScheduler',
    'DevicePool',
    'DeviceAllocation',
    'AllocationStrategy'
]


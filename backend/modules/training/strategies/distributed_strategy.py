# -*- coding: utf-8 -*-
"""
分布式训练策略

支持多种分布式训练方式：DDP、FSDP、ZeRO、Pipeline并行等。
基于技术方案中的分布式策略设计。

架构调用层次：
├── distributed_strategy.py (本模块)
│   └── 调用 backend/lib/distributed (分布式层)
│       ├── DistributedManager - 分布式管理器
│       ├── DDPWrapper/FSDPWrapper/ZeROWrapper - 模型包装器
│       ├── DDPConfig/FSDPConfig/ZeROConfig - 配置类
│       └── barrier/all_reduce/all_gather - 通信操作
│   └── 调用 backend/lib/hardware (硬件层)
│       ├── DeviceManager - 设备管理
│       ├── MixedPrecisionManager - 混合精度
│       ├── MemoryManager - 内存管理
│       └── GradientCheckpointing - 梯度检查点
│   └── 调用 base_strategy.py (策略基类)
│       ├── StrategyMonitor - 策略监控
│       ├── StrategyProfiler - 性能分析
│       ├── StrategyValidator - 结果验证
│       └── StrategyMetrics - 指标跟踪
└── 被 launcher/training_launcher.py 调用

生产级特性：
- 完整的分布式监控和诊断
- 多种并行模式的无缝切换
- 自动故障检测和恢复
- 通信性能分析和优化
"""

import logging
import time
import os
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import torch
import torch.nn as nn

from .base_strategy import (
    TrainingStrategy, 
    StrategyContext, 
    StrategyResult, 
    TrainingPhase,
    StrategyType,
    StrategyMonitor,
    StrategyProfiler,
    StrategyValidator,
    StrategyMetrics,
)

logger = logging.getLogger(__name__)


# ==================== 底层模块导入 ====================

# 分布式训练内核层

from backend.lib.distributed import (
    # 核心管理器
    DistributedManager, get_distributed_manager, ParallelMode,
    # 模型包装器
    DDPWrapper, FSDPWrapper, ZeROWrapper, PipelineWrapper,
    # 配置类
    DDPConfig, FSDPConfig, ZeROConfig, PipelineConfig,
    # 通信操作
    barrier, all_reduce, all_gather, AllReduceOp,
    # 初始化/清理
    init_distributed, cleanup_distributed,
    # 工具函数
    is_main_process, get_rank, get_world_size, synchronize,
)

# 硬件抽象层

from backend.lib.hardware import (
    # 设备管理
    DeviceManager, get_device_manager,
    # 混合精度
    MixedPrecisionManager, AmpConfig, PrecisionMode,
    # 内存管理
    MemoryManager, GradientCheckpointing, clear_memory,
    # 设备类型
    DeviceType, DeviceInfo, HardwareConfig,
    # 工具函数
    get_available_memory, estimate_tensor_memory,
)



class DistributedMode(Enum):
    """分布式模式枚举"""
    DDP = "ddp"                    # Data Distributed Parallel
    FSDP = "fsdp"                  # Fully Sharded Data Parallel
    ZERO = "zero"                  # DeepSpeed ZeRO
    PIPELINE = "pipeline"          # Pipeline Parallel
    TENSOR = "tensor"              # Tensor Parallel
    HYBRID = "hybrid"              # 混合并行
    
    @classmethod
    def from_string(cls, value: str) -> 'DistributedMode':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown distributed mode: {value}")
    
    @property
    def requires_multiple_gpus(self) -> bool:
        """是否需要多GPU"""
        return self in (DistributedMode.DDP, DistributedMode.FSDP, DistributedMode.ZERO, 
                        DistributedMode.PIPELINE, DistributedMode.TENSOR, DistributedMode.HYBRID)
    
    @property
    def supports_cpu_offload(self) -> bool:
        """是否支持CPU卸载"""
        return self in (DistributedMode.FSDP, DistributedMode.ZERO)
    
    @property
    def memory_efficiency(self) -> float:
        """内存效率评分 (0-1)"""
        scores = {
            DistributedMode.DDP: 0.3,
            DistributedMode.FSDP: 0.8,
            DistributedMode.ZERO: 0.9,
            DistributedMode.PIPELINE: 0.7,
            DistributedMode.TENSOR: 0.6,
            DistributedMode.HYBRID: 0.85
        }
        return scores.get(self, 0.5)
    
    def to_parallel_mode(self) -> Optional['ParallelMode']:
        """转换为底层 ParallelMode"""
        mode_map = {
            DistributedMode.DDP: ParallelMode.DDP,
            DistributedMode.FSDP: ParallelMode.FSDP,
            DistributedMode.ZERO: ParallelMode.ZERO_2,
            DistributedMode.PIPELINE: ParallelMode.PIPELINE,
            DistributedMode.TENSOR: ParallelMode.TENSOR,
            DistributedMode.HYBRID: ParallelMode.HYBRID
        }
        return mode_map.get(self)
    
    @classmethod
    def recommend(cls, model_size_gb: float, num_gpus: int, memory_per_gpu_gb: float) -> 'DistributedMode':
        """根据模型大小和资源推荐分布式模式"""
        if num_gpus <= 1:
            return DistributedMode.DDP  # 单GPU使用DDP
        
        # 估算每GPU需要的内存
        # memory_needed_per_gpu = model_size_gb / num_gpus
        
        if model_size_gb > memory_per_gpu_gb * num_gpus * 0.5:
            # 模型很大，使用ZeRO-3或FSDP
            return DistributedMode.ZERO
        elif model_size_gb > memory_per_gpu_gb:
            # 模型较大，使用FSDP
            return DistributedMode.FSDP
        else:
            # 模型可以放入单GPU，使用DDP
            return DistributedMode.DDP


class ZeROStage(Enum):
    """ZeRO优化阶段"""
    STAGE_1 = 1  # 优化器状态分片
    STAGE_2 = 2  # 优化器状态 + 梯度分片
    STAGE_3 = 3  # 优化器状态 + 梯度 + 参数分片
    
    @classmethod
    def from_int(cls, value: int) -> 'ZeROStage':
        """从整数创建"""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown ZeRO stage: {value}")
    
    @property
    def memory_efficiency(self) -> float:
        """内存效率"""
        return {ZeROStage.STAGE_1: 0.4, ZeROStage.STAGE_2: 0.7, ZeROStage.STAGE_3: 0.95}[self]
    
    @property
    def communication_overhead(self) -> float:
        """通信开销"""
        return {ZeROStage.STAGE_1: 0.1, ZeROStage.STAGE_2: 0.3, ZeROStage.STAGE_3: 0.5}[self]
    
    @property
    def description(self) -> str:
        """阶段描述"""
        descriptions = {
            ZeROStage.STAGE_1: "Optimizer State Partitioning",
            ZeROStage.STAGE_2: "Optimizer State + Gradient Partitioning",
            ZeROStage.STAGE_3: "Full Parameter Partitioning"
        }
        return descriptions[self]


@dataclass
class CommunicationStats:
    """通信统计"""
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_operations: int = 0
    total_time_ms: float = 0.0
    barrier_count: int = 0
    all_reduce_count: int = 0
    all_gather_count: int = 0
    broadcast_count: int = 0
    
    def record_operation(self, op_type: str, bytes_count: int, time_ms: float) -> None:
        """记录一次通信操作"""
        self.total_operations += 1
        self.total_bytes_sent += bytes_count
        self.total_time_ms += time_ms
        
        if op_type == 'barrier':
            self.barrier_count += 1
        elif op_type == 'all_reduce':
            self.all_reduce_count += 1
        elif op_type == 'all_gather':
            self.all_gather_count += 1
        elif op_type == 'broadcast':
            self.broadcast_count += 1
    
    def get_bandwidth_gbps(self) -> float:
        """获取带宽 (GB/s)"""
        if self.total_time_ms <= 0:
            return 0.0
        total_gb = self.total_bytes_sent / (1024 ** 3)
        time_seconds = self.total_time_ms / 1000.0
        return total_gb / time_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_bytes_sent': self.total_bytes_sent,
            'total_bytes_received': self.total_bytes_received,
            'total_operations': self.total_operations,
            'total_time_ms': self.total_time_ms,
            'barrier_count': self.barrier_count,
            'all_reduce_count': self.all_reduce_count,
            'all_gather_count': self.all_gather_count,
            'broadcast_count': self.broadcast_count,
            'bandwidth_gbps': self.get_bandwidth_gbps(),
        }


@dataclass
class DistributedHealthStatus:
    """分布式健康状态"""
    is_healthy: bool = True
    all_ranks_responsive: bool = True
    communication_working: bool = True
    memory_sufficient: bool = True
    last_check_time: float = 0.0
    issues: List[str] = field(default_factory=list)
    
    def add_issue(self, issue: str) -> None:
        """添加问题"""
        self.issues.append(issue)
        self.is_healthy = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'is_healthy': self.is_healthy,
            'all_ranks_responsive': self.all_ranks_responsive,
            'communication_working': self.communication_working,
            'memory_sufficient': self.memory_sufficient,
            'last_check_time': self.last_check_time,
            'issues': self.issues.copy(),
        }


@dataclass
class DistributedStrategyConfig:
    """分布式策略配置"""
    # 分布式模式
    mode: DistributedMode = DistributedMode.DDP
    
    # 基础配置
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    backend: str = "nccl"  # nccl, gloo
    timeout_minutes: int = 30
    
    # DDP配置
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    static_graph: bool = False
    
    # FSDP配置
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    cpu_offload: bool = False
    backward_prefetch: str = "BACKWARD_PRE"
    mixed_precision_fsdp: bool = False
    
    # ZeRO配置
    zero_stage: ZeROStage = ZeROStage.STAGE_2
    zero_offload: bool = False
    zero_offload_optimizer: bool = False
    
    # Pipeline配置
    num_pipeline_stages: int = 1
    pipeline_chunks: int = 1
    
    # 梯度配置
    gradient_accumulation_steps: int = 1
    sync_bn: bool = True
    gradient_clip_norm: Optional[float] = None
    
    # 通信配置
    overlap_comm: bool = True
    bucket_cap_mb: int = 25
    
    # 监控配置
    enable_monitoring: bool = True
    enable_profiling: bool = False
    health_check_interval: int = 100  # 每N步检查一次健康状态
    log_interval: int = 10
    
    # 容错配置
    auto_recovery: bool = True
    max_recovery_attempts: int = 3
    checkpoint_on_error: bool = True
    
    def validate(self) -> None:
        """验证配置"""
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size})")
        if self.local_rank < 0:
            raise ValueError("local_rank must be >= 0")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.backend not in ('nccl', 'gloo', 'mpi'):
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'mode': self.mode.value,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'master_addr': self.master_addr,
            'master_port': self.master_port,
            'backend': self.backend,
            'timeout_minutes': self.timeout_minutes,
            'find_unused_parameters': self.find_unused_parameters,
            'broadcast_buffers': self.broadcast_buffers,
            'static_graph': self.static_graph,
            'sharding_strategy': self.sharding_strategy,
            'cpu_offload': self.cpu_offload,
            'backward_prefetch': self.backward_prefetch,
            'mixed_precision_fsdp': self.mixed_precision_fsdp,
            'zero_stage': self.zero_stage.value,
            'zero_offload': self.zero_offload,
            'zero_offload_optimizer': self.zero_offload_optimizer,
            'num_pipeline_stages': self.num_pipeline_stages,
            'pipeline_chunks': self.pipeline_chunks,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'sync_bn': self.sync_bn,
            'gradient_clip_norm': self.gradient_clip_norm,
            'overlap_comm': self.overlap_comm,
            'bucket_cap_mb': self.bucket_cap_mb,
            'enable_monitoring': self.enable_monitoring,
            'enable_profiling': self.enable_profiling,
            'health_check_interval': self.health_check_interval,
            'auto_recovery': self.auto_recovery,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedStrategyConfig':
        """从字典创建"""
        if 'mode' in data and isinstance(data['mode'], str):
            data['mode'] = DistributedMode.from_string(data['mode'])
        if 'zero_stage' in data and isinstance(data['zero_stage'], int):
            data['zero_stage'] = ZeROStage.from_int(data['zero_stage'])
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    @classmethod
    def from_env(cls) -> 'DistributedStrategyConfig':
        """从环境变量创建"""
        return cls(
            world_size=int(os.environ.get('WORLD_SIZE', '1')),
            rank=int(os.environ.get('RANK', '0')),
            local_rank=int(os.environ.get('LOCAL_RANK', '0')),
            master_addr=os.environ.get('MASTER_ADDR', 'localhost'),
            master_port=os.environ.get('MASTER_PORT', '12355'),
        )
    
    def to_ddp_config(self) -> Optional['DDPConfig']:
        """转换为底层 DDPConfig"""
        return DDPConfig(
            find_unused_parameters=self.find_unused_parameters,
            broadcast_buffers=self.broadcast_buffers,
            bucket_cap_mb=self.bucket_cap_mb,
            static_graph=self.static_graph,
        )
    
    def to_fsdp_config(self) -> Optional['FSDPConfig']:
        """转换为底层 FSDPConfig"""
        return FSDPConfig(
            sharding_strategy=self.sharding_strategy,
            cpu_offload=self.cpu_offload,
            backward_prefetch=self.backward_prefetch,
            mixed_precision=self.mixed_precision_fsdp,
        )
    
    def to_zero_config(self) -> Optional['ZeROConfig']:
        """转换为底层 ZeROConfig"""
        return ZeROConfig(
            stage=self.zero_stage.value,
            offload_optimizer=self.zero_offload_optimizer,
            offload_param=self.zero_offload,
            overlap_comm=self.overlap_comm,
        )
    
    def to_pipeline_config(self) -> Optional['PipelineConfig']:
        """转换为底层 PipelineConfig"""
        return PipelineConfig(
            num_stages=self.num_pipeline_stages,
            num_micro_batches=self.pipeline_chunks,
        )
    
    def summary(self) -> str:
        """获取配置摘要"""
        return (
            f"DistributedConfig(mode={self.mode.value}, "
            f"world_size={self.world_size}, rank={self.rank}, "
            f"backend={self.backend}, grad_accum={self.gradient_accumulation_steps})"
        )


class DistributedStrategy(TrainingStrategy):
    """
    分布式训练策略
    
    整合底层分布式层和硬件层能力：
    - backend/lib/distributed: 分布式训练管理
        - DistributedManager: 统一管理分布式环境
        - DDPWrapper/FSDPWrapper/ZeROWrapper/PipelineWrapper: 模型包装器
        - DDPConfig/FSDPConfig/ZeROConfig/PipelineConfig: 配置类
        - barrier/all_reduce/all_gather: 通信操作
    - backend/lib/hardware: 设备管理、混合精度
        - DeviceManager: 设备检测和选择
        - MixedPrecisionManager: 混合精度训练
        - MemoryManager: 内存管理和优化
        - GradientCheckpointing: 梯度检查点
    - base_strategy.py: 策略基础能力
        - StrategyMonitor: 监控
        - StrategyProfiler: 性能分析
        - StrategyValidator: 验证
        - StrategyMetrics: 指标跟踪
    
    实现多种分布式训练方式：
    - DDP: 数据并行，适合大多数场景
    - FSDP: 全分片数据并行，适合大模型
    - ZeRO: DeepSpeed优化，支持三个阶段
    - Pipeline: 流水线并行，适合超长序列
    """
    
    # 策略类型
    STRATEGY_TYPE = StrategyType.DISTRIBUTED
    
    def __init__(self, config: Optional[DistributedStrategyConfig] = None):
        super().__init__(name="distributed", priority=10)
        self.config = config or DistributedStrategyConfig()
        
        # 分布式状态
        self._initialized = False
        self._process_group = None
        
        # 当前训练阶段
        self._current_phase: TrainingPhase = TrainingPhase.WARMUP
        
        # 底层管理器 (使用 backend/lib/distributed)
        self._distributed_manager: Optional['DistributedManager'] = None
        self._ddp_wrapper: Optional['DDPWrapper'] = None
        self._fsdp_wrapper: Optional['FSDPWrapper'] = None
        self._zero_wrapper: Optional['ZeROWrapper'] = None
        self._pipeline_wrapper: Optional['PipelineWrapper'] = None
        
        # 底层管理器 (使用 backend/lib/hardware)
        self._device_manager: Optional['DeviceManager'] = None
        self._amp_manager: Optional['MixedPrecisionManager'] = None
        self._memory_manager: Optional['MemoryManager'] = None
        self._gradient_checkpointing: Optional['GradientCheckpointing'] = None
        
        # 基础策略组件 (使用 base_strategy.py)
        self._strategy_monitor: Optional[StrategyMonitor] = None
        self._strategy_profiler: Optional[StrategyProfiler] = None
        self._strategy_validator: Optional[StrategyValidator] = None
        self._strategy_metrics: Optional[StrategyMetrics] = None
        
        # 通信统计
        self._comm_stats = CommunicationStats()
        
        # 健康状态
        self._health_status = DistributedHealthStatus()
        
        # 设备
        self._device: torch.device = torch.device('cpu')
        self._device_info: Optional['DeviceInfo'] = None
        
        # 验证配置
        try:
            self.config.validate()
        except ValueError as e:
            logger.warning("Config validation warning: %s", e)
    
    def setup(self, context: StrategyContext) -> None:
        """初始化分布式环境"""
        super().setup(context)
        
        # 初始化基础策略组件
        self._init_base_strategy_components()
        
        # 设置初始训练阶段
        self._current_phase = TrainingPhase.WARMUP
        
        if self.config.world_size <= 1:
            logger.info("World size <= 1, skipping distributed setup")
            # 即使单机也初始化硬件层
            self._setup_hardware_layer(context)
            return
        
        try:
            # 1. 初始化硬件层
            self._setup_hardware_layer(context)
            
            # 2. 初始化分布式层
            self._setup_distributed_layer()
            
            # 3. 包装模型
            self._wrap_model(context)
            
            # 4. 初始健康检查
            self._check_health()
            
            self._initialized = True
            
            logger.info("DistributedStrategy setup: mode=%s, world_size=%s, rank=%s",
                        self.config.mode.value, self.config.world_size, self.config.rank)
        except Exception as e:
            logger.error("Failed to setup distributed: %s", e)
            self._health_status.add_issue(f"Setup failed: {e}")
            raise
    
    def _init_base_strategy_components(self) -> None:
        """
        初始化基础策略组件
        
        使用 base_strategy.py 提供的组件
        """
        # 初始化策略监控器
        if self.config.enable_monitoring:
            try:
                self._strategy_monitor = StrategyMonitor(history_size=10000)
            except Exception as e:
                logger.warning("Failed to init StrategyMonitor: %s", e)
        
        # 初始化性能分析器
        if self.config.enable_profiling:
            try:
                self._strategy_profiler = StrategyProfiler()
            except Exception as e:
                logger.warning("Failed to init StrategyProfiler: %s", e)
        
        # 初始化验证器
        try:
            self._strategy_validator = StrategyValidator()
            self._add_distributed_validation_rules()
        except Exception as e:
            logger.warning("Failed to init StrategyValidator: %s", e)
        
        # 初始化指标跟踪
        try:
            self._strategy_metrics = StrategyMetrics()
        except Exception as e:
            logger.warning("Failed to init StrategyMetrics: %s", e)
        
        logger.debug("Base strategy components initialized")
    
    def _add_distributed_validation_rules(self) -> None:
        """添加分布式特定的验证规则"""
        if self._strategy_validator is None:
            return
        
        if hasattr(self._strategy_validator, 'add_check'):
            # 检查分布式环境健康
            def check_distributed_health(result: StrategyResult) -> Tuple[bool, str]:
                if not self._health_status.is_healthy:
                    return False, f"Distributed unhealthy: {self._health_status.issues}"
                return True, ""
            
            self._strategy_validator.add_check(check_distributed_health)
    
    def _setup_hardware_layer(self, context: StrategyContext) -> None:
        """
        初始化硬件层
        
        使用 backend/lib/hardware 提供的能力：
        - DeviceManager: 设备检测和选择
        - MixedPrecisionManager: 混合精度训练
        - MemoryManager: 内存管理
        - GradientCheckpointing: 梯度检查点
        - DeviceInfo: 设备信息
        """
        # 使用底层硬件管理器
        self._device_manager = get_device_manager()
        self._device = self._device_manager.get_device()
            
        # 获取设备信息
        if hasattr(self._device_manager, 'get_device_info'):
            self._device_info = self._device_manager.get_device_info(self._device)
            
        # 设置混合精度（分布式训练通常需要）
        if AmpConfig is not None and MixedPrecisionManager is not None:
            amp_config = AmpConfig(enabled=True, precision=PrecisionMode.MIXED_FP16)
            self._amp_manager = MixedPrecisionManager(amp_config, self._device)
            
        # 初始化内存管理器
        self._memory_manager = MemoryManager(device=self._device)

        # 初始化梯度检查点管理器
        if GradientCheckpointing is not None and context.model is not None:
            self._gradient_checkpointing = GradientCheckpointing(context.model)

            
        # 记录可用内存
        if get_available_memory is not None:
            try:
                available_mem = get_available_memory(self._device)
                logger.info("Available memory: %.2f GB", available_mem)
            except Exception:
                pass
            
        logger.info("Hardware layer initialized: device=%s, device_manager=%s, amp_manager=%s, memory_manager=%s",
                    self._device, self._device_manager is not None, self._amp_manager is not None, self._memory_manager is not None)

    
    def _setup_distributed_layer(self) -> None:
        """
        初始化分布式层
        
        使用 backend/lib/distributed 提供的能力：
        - DistributedManager: 分布式环境管理
        - init_distributed: 初始化函数
        - is_main_process/get_rank/get_world_size: 工具函数
        """
        if get_distributed_manager is not None:
            # 使用底层分布式管理器
            self._distributed_manager = get_distributed_manager()
            
            # 使用 init_distributed 如果可用
            if init_distributed is not None:
                try:
                    init_distributed(
                        backend=self.config.backend,
                        world_size=self.config.world_size,
                        rank=self.config.rank,
                    )
                except Exception as e:
                    logger.warning("init_distributed failed: %s, trying manager.initialize", e)
                
                # 设置环境变量以供 initialize 使用
                os.environ['MASTER_ADDR'] = self.config.master_addr
                os.environ['MASTER_PORT'] = self.config.master_port
                
                self._distributed_manager.initialize(
                    backend=self.config.backend,
                    world_size=self.config.world_size,
                    rank=self.config.rank
                )
            else:
                # 设置环境变量以供 initialize 使用
                os.environ['MASTER_ADDR'] = self.config.master_addr
                os.environ['MASTER_PORT'] = self.config.master_port
                
                self._distributed_manager.initialize(
                    backend=self.config.backend,
                    world_size=self.config.world_size,
                    rank=self.config.rank
                )
            
            # 使用 torch.distributed.group.WORLD 作为默认进程组
            if torch.distributed.is_initialized():
                self._process_group = torch.distributed.group.WORLD
            
            # 使用工具函数验证初始化
            if is_main_process is not None:
                is_main = is_main_process()
                logger.info("Is main process: %s", is_main)
            if get_rank is not None:
                actual_rank = get_rank()
                if actual_rank != self.config.rank:
                    logger.warning("Rank mismatch: config=%s, actual=%s", self.config.rank, actual_rank)
            if get_world_size is not None:
                actual_world_size = get_world_size()
                if actual_world_size != self.config.world_size:
                    logger.warning("World size mismatch: config=%s, actual=%s", self.config.world_size, actual_world_size)
            
            logger.info("Distributed layer initialized via DistributedManager")
        else:
            # 回退到原生 PyTorch 分布式
            self._init_distributed_native()
    
    def _init_distributed_native(self) -> None:
        """初始化分布式进程组（原生PyTorch方式）"""
        # 设置环境变量
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = self.config.master_port
        os.environ['WORLD_SIZE'] = str(self.config.world_size)
        os.environ['RANK'] = str(self.config.rank)
        
        # 初始化进程组
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=self.config.backend,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
        
        self._process_group = torch.distributed.group.WORLD
        
        logger.info("Distributed initialized (native): rank %s/%s", self.config.rank, self.config.world_size)
    
    def _wrap_model(self, context: StrategyContext) -> None:
        """
        包装模型以支持分布式训练
        
        优先使用 backend/lib/distributed 的包装器：
        - DDPWrapper: DDP模式
        - FSDPWrapper: FSDP模式
        - ZeROWrapper: ZeRO模式
        - PipelineWrapper: Pipeline模式
        """
        if context.model is None:
            return
        
        device = self._device if self._device else context.device
        
        # 优先使用底层包装器

        wrapped = self._wrap_model_via_wrappers(context.model, device)
        if wrapped is not None:
            context.model = wrapped
            return
        
        # 次选：使用分布式管理器
        if self._distributed_manager:
            context.model = self._wrap_model_via_manager(context.model, device)
            return
        
        # 回退到原生实现
        if self.config.mode == DistributedMode.DDP:
            context.model = self._wrap_ddp(context.model, device)
        elif self.config.mode == DistributedMode.FSDP:
            context.model = self._wrap_fsdp(context.model, device)
        elif self.config.mode == DistributedMode.ZERO:
            context.model = self._wrap_zero(context.model, device)
        elif self.config.mode == DistributedMode.PIPELINE:
            context.model = self._wrap_pipeline(context.model, device)
    
    def _wrap_model_via_wrappers(self, model: nn.Module, device: torch.device) -> Optional[nn.Module]:
        """
        使用底层包装器包装模型
        
        使用 backend/lib/distributed 的具体包装器类
        """
        model = model.to(device)
        
        # 同步批归一化
        if self.config.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        try:
            if self.config.mode == DistributedMode.DDP and DDPWrapper is not None:
                # 使用 DDPWrapper
                ddp_config = self.config.to_ddp_config()
                self._ddp_wrapper = DDPWrapper(ddp_config)
                self._ddp_wrapper.wrap(model)
                logger.info("Model wrapped via DDPWrapper")
                return self._ddp_wrapper.model if hasattr(self._ddp_wrapper, 'model') else model
            
            elif self.config.mode == DistributedMode.FSDP and FSDPWrapper is not None:
                # 使用 FSDPWrapper
                fsdp_config = self.config.to_fsdp_config()
                self._fsdp_wrapper = FSDPWrapper(fsdp_config)
                self._fsdp_wrapper.wrap(model)
                logger.info("Model wrapped via FSDPWrapper")
                return self._fsdp_wrapper.model if hasattr(self._fsdp_wrapper, 'model') else model
            
            elif self.config.mode == DistributedMode.ZERO and ZeROWrapper is not None:
                # 使用 ZeROWrapper
                zero_config = self.config.to_zero_config()
                self._zero_wrapper = ZeROWrapper(zero_config)
                # ZeROWrapper wrap returns (engine, optimizer, ...)
                engine, _, _, _ = self._zero_wrapper.wrap(model)
                logger.info("Model wrapped via ZeROWrapper")
                return engine
            
            elif self.config.mode == DistributedMode.PIPELINE and PipelineWrapper is not None:
                # 使用 PipelineWrapper
                pipeline_config = self.config.to_pipeline_config()
                self._pipeline_wrapper = PipelineWrapper(pipeline_config)
                
                # PipelineWrapper usage matches DistributedManager
                self._pipeline_wrapper.split_model(model)
                logger.info("Model wrapped via PipelineWrapper")
                return self._pipeline_wrapper.get_current_stage()
        
        except Exception as e:
            logger.warning("Failed to wrap via specific wrapper: %s", e)
        
        return None
    
    def _wrap_model_via_manager(self, model: nn.Module, device: torch.device) -> nn.Module:
        """
        使用底层分布式管理器包装模型
        
        通过 backend/lib/distributed 的 DistributedManager 包装模型
        """
        # 映射分布式模式到 ParallelMode
        mode_map = {
            DistributedMode.DDP: ParallelMode.DDP,
            DistributedMode.FSDP: ParallelMode.FSDP,
            DistributedMode.ZERO: ParallelMode.ZERO_2,
            DistributedMode.PIPELINE: ParallelMode.PIPELINE,
            DistributedMode.TENSOR: ParallelMode.TENSOR,
            DistributedMode.HYBRID: ParallelMode.HYBRID
        }
        parallel_mode = mode_map.get(self.config.mode, ParallelMode.DDP)
        
        # 移动模型到设备
        model = model.to(device)
        
        # 同步批归一化
        if self.config.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # 使用分布式管理器包装
        wrapped_model = self._distributed_manager.wrap_model(
            model,
            mode=parallel_mode,
            find_unused_parameters=self.config.find_unused_parameters
        )
        
        logger.info(f"Model wrapped via DistributedManager: mode={parallel_mode.value}")
        return wrapped_model
    
    def _wrap_ddp(self, model: nn.Module, device: torch.device) -> nn.Module:
        """使用DDP包装模型"""
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        model = model.to(device)
        
        # 同步批归一化
        if self.config.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        wrapped_model = DDP(
            model,
            device_ids=[self.config.local_rank] if device.type == 'cuda' else None,
            output_device=self.config.local_rank if device.type == 'cuda' else None,
            find_unused_parameters=self.config.find_unused_parameters,
            broadcast_buffers=self.config.broadcast_buffers
        )
        
        logger.info("Model wrapped with DDP")
        return wrapped_model
    
    def _wrap_fsdp(self, model: nn.Module, device: torch.device) -> nn.Module:
        """使用FSDP包装模型"""
        try:
            from torch.distributed.fsdp import (
                FullyShardedDataParallel as FSDP,
                ShardingStrategy,
                CPUOffload,
                BackwardPrefetch
            )
            
            # 解析配置
            sharding_map = {
                'FULL_SHARD': ShardingStrategy.FULL_SHARD,
                'SHARD_GRAD_OP': ShardingStrategy.SHARD_GRAD_OP,
                'NO_SHARD': ShardingStrategy.NO_SHARD
            }
            sharding_strategy = sharding_map.get(
                self.config.sharding_strategy, 
                ShardingStrategy.FULL_SHARD
            )
            
            prefetch_map = {
                'BACKWARD_PRE': BackwardPrefetch.BACKWARD_PRE,
                'BACKWARD_POST': BackwardPrefetch.BACKWARD_POST
            }
            backward_prefetch = prefetch_map.get(
                self.config.backward_prefetch,
                BackwardPrefetch.BACKWARD_PRE
            )
            
            wrapped_model = FSDP(
                model,
                sharding_strategy=sharding_strategy,
                cpu_offload=CPUOffload(offload_params=self.config.cpu_offload),
                backward_prefetch=backward_prefetch,
                device_id=device
            )
            
            logger.info(f"Model wrapped with FSDP: {self.config.sharding_strategy}")
            return wrapped_model
            
        except ImportError:
            logger.warning("FSDP not available, falling back to DDP")
            return self._wrap_ddp(model, device)
    
    def _wrap_zero(self, model: nn.Module, device: torch.device) -> nn.Module:
        """使用DeepSpeed ZeRO包装模型"""
        try:
            import deepspeed
            
            # DeepSpeed配置
            ds_config = {
                'train_batch_size': 'auto',
                'gradient_accumulation_steps': self.config.gradient_accumulation_steps,
                'optimizer': {
                    'type': 'AdamW',
                    'params': {
                        'lr': 'auto',
                        'betas': 'auto',
                        'eps': 'auto'
                    }
                },
                'zero_optimization': {
                    'stage': self.config.zero_stage.value,
                    'offload_optimizer': {
                        'device': 'cpu' if self.config.zero_offload else 'none'
                    },
                    'offload_param': {
                        'device': 'cpu' if self.config.zero_offload else 'none'
                    },
                    'overlap_comm': self.config.overlap_comm,
                    'contiguous_gradients': True,
                    'reduce_bucket_size': self.config.bucket_cap_mb * 1024 * 1024
                }
            }
            
            model_engine, _, _, _ = deepspeed.initialize(
                model=model,
                config=ds_config
            )
            
            logger.info(f"Model wrapped with DeepSpeed ZeRO Stage {self.config.zero_stage.value}")
            return model_engine
            
        except ImportError:
            logger.warning("DeepSpeed not available, falling back to DDP")
            return self._wrap_ddp(model, device)
    
    def _wrap_pipeline(self, model: nn.Module, device: torch.device) -> nn.Module:
        """使用Pipeline Parallel包装模型"""
        try:
            import importlib
            # 使用 importlib 动态导入以避免静态分析错误
            pipeline_sync = importlib.import_module("torch.distributed.pipeline.sync")
            Pipe = pipeline_sync.Pipe
            
            # 将模型分割成多个阶段
            # 注意：这需要模型已经被适当地分割
            if hasattr(model, 'get_pipeline_stages'):
                stages = model.get_pipeline_stages()
            else:
                # 简单分割
                stages = self._simple_split_model(model)
            
            wrapped_model = Pipe(
                nn.Sequential(*stages),
                chunks=self.config.pipeline_chunks
            )
            
            logger.info(f"Model wrapped with Pipeline Parallel: {len(stages)} stages")
            return wrapped_model
            
        except (ImportError, Exception) as e:
            logger.warning(f"Pipeline parallel not available: {e}, falling back to DDP")
            return self._wrap_ddp(model, device)
    
    def _simple_split_model(self, model: nn.Module) -> List[nn.Module]:
        """简单的模型分割"""
        # 将模型的子模块分成多个阶段
        children = list(model.children())
        num_stages = self.config.num_pipeline_stages
        
        if len(children) < num_stages:
            return [model]
        
        # 均匀分割
        chunk_size = len(children) // num_stages
        stages = []
        for i in range(num_stages):
            start = i * chunk_size
            end = start + chunk_size if i < num_stages - 1 else len(children)
            stages.append(nn.Sequential(*children[start:end]))
        
        return stages
    
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """
        计算损失（分布式环境下的处理）
        
        整合硬件层的混合精度能力和基础策略组件
        """
        start_time = time.time()
        warnings = []
        
        # 使用AMP上下文（如果可用）
        amp_ctx = self._amp_manager.autocast_context() if self._amp_manager else None
        
        with amp_ctx if amp_ctx else torch.no_grad():
            # 获取损失
            if 'loss' in outputs:
                loss = outputs['loss']
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                raise ValueError("outputs中没有找到loss")
        
        metrics = {
            'loss': loss.item(),
            'training_phase': self._current_phase.value if hasattr(self._current_phase, 'value') else str(self._current_phase),
        }
        
        # 在分布式环境下，可能需要对损失进行缩放
        if self._initialized and self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
            metrics['scaled_loss'] = loss.item()
        
        # 添加分布式信息
        if self._initialized:
            metrics['rank'] = self.config.rank
            metrics['world_size'] = self.config.world_size
            metrics['mode'] = self.config.mode.value
        
        # 计算步骤时间
        step_time = time.time() - start_time
        metrics['step_time_ms'] = step_time * 1000
        
        # 创建结果
        result = StrategyResult(
            loss=loss, 
            metrics=metrics,
            step_time=step_time,
            warnings=warnings if warnings else None
        )
        
        # 记录到策略监控器
        if self._strategy_monitor is not None:
            try:
                self._strategy_monitor.record_step(result, context)
            except Exception as e:
                logger.debug(f"StrategyMonitor record failed: {e}")
        
        # 更新策略指标
        if self._strategy_metrics is not None:
            try:
                self._strategy_metrics.total_steps += 1
                self._strategy_metrics.total_loss += loss.item()
                self._strategy_metrics.avg_loss = (
                    self._strategy_metrics.total_loss / self._strategy_metrics.total_steps
                )
            except Exception:
                pass
        
        # 验证结果
        if self._strategy_validator is not None:
            try:
                is_valid, errors = self._strategy_validator.validate(result)
                if not is_valid:
                    for error in errors:
                        warnings.append(f"Validation: {error}")
                    result.warnings = warnings
            except Exception:
                pass
        
        return result
    
    def backward(self, loss: torch.Tensor) -> None:
        """
        反向传播
        
        使用硬件层的混合精度后向传播
        """
        if self._amp_manager:
            self._amp_manager.backward(loss)
        else:
            loss.backward()
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """
        优化器步进
        
        使用硬件层的混合精度优化器步进
        """
        if self._amp_manager:
            self._amp_manager.step(optimizer)
        else:
            optimizer.step()
    
    def on_step_end(self, context: StrategyContext, result: StrategyResult) -> None:
        """步骤结束时的同步操作"""
        super().on_step_end(context, result)
        
        if not self._initialized:
            return
        
        # 同步梯度（如果使用梯度累积）
        if context.global_step % self.config.gradient_accumulation_steps == 0:
            self._sync_gradients()
        
        # 定期健康检查
        if (self.config.health_check_interval > 0 and 
            context.global_step % self.config.health_check_interval == 0):
            self._check_health()
        
        # 更新训练阶段
        self._update_training_phase(context)
    
    def _update_training_phase(self, context: StrategyContext) -> None:
        """
        更新训练阶段
        
        使用 base_strategy.py 的 TrainingPhase
        """
        max_steps = context.max_steps or 10000
        warmup_ratio = 0.1
        cooldown_ratio = 0.1
        
        warmup_steps = int(max_steps * warmup_ratio)
        cooldown_start = int(max_steps * (1 - cooldown_ratio))
        
        old_phase = self._current_phase
        
        if context.global_step < warmup_steps:
            self._current_phase = TrainingPhase.WARMUP
        elif context.global_step >= cooldown_start:
            self._current_phase = TrainingPhase.COOLDOWN
        else:
            self._current_phase = TrainingPhase.MAIN
        
        if old_phase != self._current_phase:
            logger.info(f"Training phase changed: {old_phase} -> {self._current_phase}")
    
    def _check_health(self) -> None:
        """检查分布式环境健康状态"""
        self._health_status = DistributedHealthStatus()
        self._health_status.last_check_time = time.time()
        
        # 检查通信是否工作
        try:
            self._sync_gradients()
            self._health_status.communication_working = True
        except Exception as e:
            self._health_status.add_issue(f"Communication failed: {e}")
            self._health_status.communication_working = False
        
        # 检查内存是否充足
        if self._memory_manager is not None:
            try:
                stats = self._memory_manager.get_stats()
                if hasattr(stats, 'pressure_level'):
                    if stats.pressure_level in ('HIGH', 'CRITICAL'):
                        self._health_status.add_issue(f"Memory pressure: {stats.pressure_level}")
                        self._health_status.memory_sufficient = False
            except Exception:
                pass
        
        # 使用底层工具函数检查
        if is_main_process is not None:
            try:
                # 主进程可以执行更多检查
                if is_main_process():
                    pass  # 可以添加主进程特有的检查
            except Exception:
                pass
    
    def _sync_gradients(self) -> None:
        """
        同步梯度
        
        优先使用底层分布式层的 barrier 和 synchronize
        """
        start_time = time.time()
        

        if synchronize is not None:
            try:
                synchronize()
                elapsed = (time.time() - start_time) * 1000
                self._comm_stats.record_operation('barrier', 0, elapsed)
                return
            except Exception:
                pass
            
        if barrier is not None:
            try:
                barrier()
                elapsed = (time.time() - start_time) * 1000
                self._comm_stats.record_operation('barrier', 0, elapsed)
                return
            except Exception:
                pass
        
        # 回退到原生
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            elapsed = (time.time() - start_time) * 1000
            self._comm_stats.record_operation('barrier', 0, elapsed)
    
    def all_reduce_tensor(self, tensor: torch.Tensor, op: str = 'mean') -> torch.Tensor:
        """
        跨设备聚合tensor
        
        使用底层分布式层的 all_reduce
        """
        if not self._initialized:
            return tensor
        
        start_time = time.time()
        bytes_count = tensor.numel() * tensor.element_size()
        
        if all_reduce is not None:
            try:
                # Custom AllReduceOp does not support AVG, so we use SUM and divide manually for mean
                reduce_op = AllReduceOp.SUM
                result = all_reduce(tensor, op=reduce_op)
                
                if op == 'mean':
                    result = result / self.config.world_size
                
                elapsed = (time.time() - start_time) * 1000
                self._comm_stats.record_operation('all_reduce', bytes_count, elapsed)
                return result
            except Exception as e:
                logger.warning("all_reduce failed: %s", e)
        
        # 回退
        result = self.reduce_tensor(tensor, self.config.world_size)
        elapsed = (time.time() - start_time) * 1000
        self._comm_stats.record_operation('all_reduce', bytes_count, elapsed)
        return result
    
    def all_gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        收集所有设备的tensor
        
        使用底层分布式层的 all_gather
        """
        if not self._initialized:
            return [tensor]
        
        start_time = time.time()
        bytes_count = tensor.numel() * tensor.element_size()
        
        if all_gather is not None:
            try:
                result = all_gather(tensor)
                elapsed = (time.time() - start_time) * 1000
                self._comm_stats.record_operation('all_gather', bytes_count, elapsed)
                return result
            except Exception as e:
                logger.warning(f"all_gather failed: {e}")
        
        # 回退
        result = self.gather_tensor(tensor, self.config.world_size)
        elapsed = (time.time() - start_time) * 1000
        self._comm_stats.record_operation('all_gather', bytes_count, elapsed)
        return result
    
    def cleanup(self) -> None:
        """清理分布式资源"""
        super().cleanup()
        
        # 优先使用底层分布式管理器清理

        if cleanup_distributed is not None:
            try:
                cleanup_distributed()
                self._initialized = False
                logger.info("Distributed resources cleaned up via cleanup_distributed")
            except Exception as e:
                logger.warning(f"cleanup_distributed failed: {e}")
        elif self._distributed_manager is not None:
            try:
                self._distributed_manager.cleanup()
                self._initialized = False
                logger.info("Distributed resources cleaned up via DistributedManager")
            except Exception as e:
                logger.warning(f"DistributedManager cleanup failed: {e}")
        
        if self._initialized and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            self._initialized = False
            logger.info("Distributed resources cleaned up (native)")
        
        # 清理硬件层资源
        try:
            clear_memory()
            logger.info("Memory cleared via hardware layer")
        except Exception:
            pass
        
        # 清理包装器
        self._ddp_wrapper = None
        self._fsdp_wrapper = None
        self._zero_wrapper = None
        self._pipeline_wrapper = None
    
    def get_layer_info(self) -> Dict[str, Any]:
        """获取底层模块调用信息"""
        return {
            # 管理器状态
            'distributed_manager': self._distributed_manager is not None,
            'device_manager': self._device_manager is not None,
            'amp_manager': self._amp_manager is not None,
            'memory_manager': self._memory_manager is not None,
            'gradient_checkpointing': self._gradient_checkpointing is not None,
            # 包装器状态
            'ddp_wrapper': self._ddp_wrapper is not None,
            'fsdp_wrapper': self._fsdp_wrapper is not None,
            'zero_wrapper': self._zero_wrapper is not None,
            'pipeline_wrapper': self._pipeline_wrapper is not None,
            # 基础策略组件
            'strategy_monitor': self._strategy_monitor is not None,
            'strategy_profiler': self._strategy_profiler is not None,
            'strategy_validator': self._strategy_validator is not None,
            'strategy_metrics': self._strategy_metrics is not None,
            # 设备信息
            'device': str(self._device),
            'device_info': self._device_info.__dict__ if self._device_info else None,
            # 配置
            'mode': self.config.mode.value if self.config else None,
            'world_size': self.config.world_size if self.config else 1,
            'rank': self.config.rank if self.config else 0,
        }
    
    # ==================== 基础策略组件访问方法 ====================
    
    def get_strategy_type(self) -> StrategyType:
        """获取策略类型"""
        return self.STRATEGY_TYPE
    
    def get_training_phase(self) -> TrainingPhase:
        """获取当前训练阶段"""
        return self._current_phase
    
    def set_training_phase(self, phase: TrainingPhase) -> None:
        """设置训练阶段"""
        self._current_phase = phase
    
    def get_strategy_monitor(self) -> Optional[StrategyMonitor]:
        """获取策略监控器"""
        return self._strategy_monitor
    
    def get_strategy_profiler(self) -> Optional[StrategyProfiler]:
        """获取策略性能分析器"""
        return self._strategy_profiler
    
    def get_strategy_validator(self) -> Optional[StrategyValidator]:
        """获取策略验证器"""
        return self._strategy_validator
    
    def get_strategy_metrics(self) -> Optional[StrategyMetrics]:
        """获取策略指标"""
        return self._strategy_metrics
    
    # ==================== 通信和健康状态方法 ====================
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """获取通信统计"""
        return self._comm_stats.to_dict()
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return self._health_status.to_dict()
    
    def is_healthy(self) -> bool:
        """检查是否健康"""
        return self._health_status.is_healthy
    
    # ==================== 诊断方法 ====================
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断分布式策略状态"""
        diagnosis = {
            'initialized': self._initialized,
            'config': self.config.to_dict() if self.config else {},
            'layer_info': self.get_layer_info(),
            'health_status': self.get_health_status(),
            'communication_stats': self.get_communication_stats(),
            'recommendations': [],
        }
        
        # 添加建议
        if not self._initialized and self.config.world_size > 1:
            diagnosis['recommendations'].append("Distributed not initialized despite world_size > 1")
        
        if not self._health_status.is_healthy:
            diagnosis['recommendations'].append(f"Health issues: {self._health_status.issues}")
        
        if self._comm_stats.total_operations > 0:
            bandwidth = self._comm_stats.get_bandwidth_gbps()
            if bandwidth < 1.0:
                diagnosis['recommendations'].append(f"Low communication bandwidth: {bandwidth:.2f} GB/s")
        
        # 检查内存
        if self._memory_manager is not None:
            try:
                stats = self._memory_manager.get_stats()
                diagnosis['memory_stats'] = stats.__dict__ if hasattr(stats, '__dict__') else str(stats)
            except Exception:
                pass
        
        return diagnosis
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diagnosis = self.diagnose()
        print("\n" + "=" * 60)
        print("Distributed Strategy Diagnosis")
        print("=" * 60)
        print(f"Initialized: {diagnosis['initialized']}")
        print(f"Mode: {diagnosis['config'].get('mode', 'N/A')}")
        print(f"World Size: {diagnosis['config'].get('world_size', 1)}")
        print(f"Rank: {diagnosis['config'].get('rank', 0)}")
        print(f"\nHealth Status: {'Healthy' if diagnosis['health_status']['is_healthy'] else 'Unhealthy'}")
        if diagnosis['health_status']['issues']:
            print(f"  Issues: {diagnosis['health_status']['issues']}")
        print(f"\nCommunication Stats:")
        for k, v in diagnosis['communication_stats'].items():
            print(f"  {k}: {v}")
        if diagnosis['recommendations']:
            print(f"\nRecommendations:")
            for rec in diagnosis['recommendations']:
                print(f"  - {rec}")
        print("=" * 60)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取策略摘要"""
        summary = {
            'strategy_type': self.STRATEGY_TYPE.value,
            'training_phase': self._current_phase.value if hasattr(self._current_phase, 'value') else str(self._current_phase),
            'initialized': self._initialized,
            'mode': self.config.mode.value if self.config else 'unknown',
            'world_size': self.config.world_size if self.config else 1,
            'healthy': self._health_status.is_healthy,
        }
        
        # 添加策略指标
        if self._strategy_metrics is not None:
            summary['metrics'] = {
                'total_steps': self._strategy_metrics.total_steps,
                'avg_loss': self._strategy_metrics.avg_loss,
            }
        
        return summary
    
    def print_summary(self) -> None:
        """打印策略摘要"""
        summary = self.get_summary()
        print(f"\nDistributed Strategy Summary:")
        print(f"  Type: {summary['strategy_type']}")
        print(f"  Phase: {summary['training_phase']}")
        print(f"  Mode: {summary['mode']}")
        print(f"  World Size: {summary['world_size']}")
        print(f"  Initialized: {summary['initialized']}")
        print(f"  Healthy: {summary['healthy']}")
        if 'metrics' in summary:
            print(f"  Total Steps: {summary['metrics']['total_steps']}")
            print(f"  Avg Loss: {summary['metrics']['avg_loss']:.4f}")
    
    # ==================== 梯度检查点方法 ====================
    
    def enable_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """
        启用梯度检查点
        
        使用 backend/lib/hardware 的 GradientCheckpointing
        """
        if self._gradient_checkpointing is not None:
            try:
                self._gradient_checkpointing.enable()
                return model
            except Exception as e:
                logger.warning("GradientCheckpointing failed: %s", e)
        
        # 回退到原生实现（如果模型支持）
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        return model
    
    # ==================== 内存管理方法 ====================
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        if self._memory_manager is not None:
            try:
                stats = self._memory_manager.get_stats()
                return stats.__dict__ if hasattr(stats, '__dict__') else {'raw': str(stats)}
            except Exception:
                pass
        
        # 回退到 PyTorch 原生
        if torch.cuda.is_available():
            return {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
            }
        
        return {}
    
    def optimize_memory(self) -> None:
        """优化内存使用"""
        if self._memory_manager is not None:
            try:
                self._memory_manager.clear_cache()
                return
            except Exception:
                pass
        
        # 回退
        if clear_memory is not None:
            clear_memory()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        """跨设备聚合tensor"""
        if world_size <= 1:
            return tensor
        
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= world_size
        return rt
    
    @staticmethod
    def gather_tensor(tensor: torch.Tensor, world_size: int) -> List[torch.Tensor]:
        """收集所有设备的tensor"""
        if world_size <= 1:
            return [tensor]
        
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, tensor)
        return gathered


class IndustryDistributedStrategy(DistributedStrategy):
    """
    行业分布式训练策略
    
    针对行业模型训练优化的分布式策略：
    - 支持异构算力环境
    - 支持私有化部署
    - 增强的容错和恢复机制
    - 自动资源优化
    
    使用底层能力：
    - backend/lib/distributed: 分布式训练管理
        - DistributedManager, DDPWrapper, FSDPWrapper, ZeROWrapper
    - backend/lib/hardware: 设备管理、内存优化
        - DeviceManager, MemoryManager, GradientCheckpointing
        - DeviceType, DeviceInfo, HardwareConfig
    - base_strategy.py: 策略基础能力
        - StrategyMonitor, StrategyProfiler, StrategyValidator
    """
    
    # 策略类型
    STRATEGY_TYPE = StrategyType.INDUSTRY
    
    def __init__(self, config: Optional[DistributedStrategyConfig] = None):
        if config is None:
            config = DistributedStrategyConfig(
                mode=DistributedMode.FSDP,  # 默认使用FSDP以支持大模型
                zero_stage=ZeROStage.STAGE_2,
                sync_bn=True,
                overlap_comm=True,
                enable_monitoring=True,
                auto_recovery=True,
            )
        super().__init__(config)
        self.name = "industry_distributed"
        
        # 检查点管理
        self._checkpoint_path = None
        self._checkpoint_interval = 1000
        self._last_checkpoint_step = 0
        
        # 恢复状态
        self._recovery_attempts = 0
        self._last_successful_step = 0
        
        # 设备类型跟踪（使用 DeviceType）
        self._device_type: Optional['DeviceType'] = None
        
        # 硬件配置
        self._hardware_config: Optional['HardwareConfig'] = None
    
    def setup(self, context: StrategyContext) -> None:
        """初始化行业分布式环境"""
        super().setup(context)
        
        # 设置检查点路径
        self._checkpoint_path = context.config.get(
            'checkpoint_path', 
            './checkpoints/distributed'
        ) if hasattr(context, 'config') and context.config else './checkpoints/distributed'
        
        # 创建检查点目录
        if self.config.rank == 0:
            os.makedirs(self._checkpoint_path, exist_ok=True)
        
        # 检测设备类型
        if DeviceType is not None:
            self._detect_device_type()
        
        # 初始化硬件配置
        if HardwareConfig is not None:
            try:
                self._hardware_config = HardwareConfig()
                if hasattr(self._hardware_config, 'apply'):
                    self._hardware_config.apply()
            except Exception as e:
                logger.warning(f"Failed to apply HardwareConfig: {e}")
        
        # 初始化或使用父类的内存管理器
        if self._memory_manager is None:
            try:
                self._memory_manager = MemoryManager(device=self._device)
                logger.info("Memory manager initialized for industry distributed training")
            except Exception as e:
                logger.warning(f"Failed to init MemoryManager: {e}")
    
    def _detect_device_type(self) -> None:
        """检测设备类型"""
        if self._device_info is not None and hasattr(self._device_info, 'device_type'):
            self._device_type = self._device_info.device_type
        elif self._device.type == 'cuda':
            self._device_type = DeviceType.GPU if DeviceType is not None else None
        elif self._device.type == 'cpu':
            self._device_type = DeviceType.CPU if DeviceType is not None else None
    
    def prepare_batch(self, batch: Dict[str, Any], context: StrategyContext) -> Dict[str, Any]:
        """
        准备批次数据
        
        行业分布式场景下的数据准备，支持内存估算
        """
        prepared = {}
        total_bytes = 0
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self._device)
                total_bytes += value.numel() * value.element_size()
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                prepared[key] = {
                    k: v.to(self._device) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            elif isinstance(value, (list, tuple)):
                # 处理列表/元组
                prepared[key] = type(value)(
                    v.to(self._device) if isinstance(v, torch.Tensor) else v
                    for v in value
                )
            else:
                prepared[key] = value
        
        # 使用 estimate_tensor_memory 估算内存（如果可用）
        try:
            for key, value in prepared.items():
                if isinstance(value, torch.Tensor):
                    # estimated = estimate_tensor_memory(value)
                    # 可以记录到日志或指标
                    pass
        except Exception:
            pass
        
        return prepared
    
    def on_step_end(self, context: StrategyContext, result: StrategyResult) -> None:
        """步骤结束回调"""
        super().on_step_end(context, result)
        
        # 记录成功步骤
        self._last_successful_step = context.global_step
        
        # 自动保存检查点
        if (self._checkpoint_interval > 0 and 
            context.global_step - self._last_checkpoint_step >= self._checkpoint_interval):
            if context.model is not None:
                checkpoint_path = os.path.join(
                    self._checkpoint_path,
                    f"checkpoint_step_{context.global_step}.pt"
                )
                self.save_checkpoint(context.model, checkpoint_path)
                self._last_checkpoint_step = context.global_step
    
    def save_checkpoint(self, model: nn.Module, path: str) -> None:
        """
        保存分布式检查点
        
        使用底层分布式管理器的检查点保存能力
        仅在主节点保存（使用 is_main_process）
        """
        # 使用底层工具判断是否是主节点
        is_main = False
        try:
            is_main = is_main_process()
        except Exception:
            is_main = self.config.rank == 0

        
        if not is_main:
            return
        
        # 使用FSDP/ZeRO包装器的保存方法
        if self._fsdp_wrapper is not None and hasattr(self._fsdp_wrapper, 'save_checkpoint'):
            try:
                self._fsdp_wrapper.save_checkpoint(path)
                logger.info(f"Checkpoint saved via FSDPWrapper to {path}")
                return
            except Exception as e:
                logger.warning(f"FSDPWrapper save failed: {e}")
        
        if self._zero_wrapper is not None and hasattr(self._zero_wrapper, 'save_checkpoint'):
            try:
                self._zero_wrapper.save_checkpoint(path)
                logger.info(f"Checkpoint saved via ZeROWrapper to {path}")
                return
            except Exception as e:
                logger.warning(f"ZeROWrapper save failed: {e}")
        
        # 使用分布式管理器
        if self._distributed_manager is not None:
            try:
                # DistributedManager 没有 save_checkpoint 方法，回退到原生保存
                # 但主进程检查已经通过，所以可以安全保存
                pass 
            except Exception as e:
                logger.warning("DistributedManager check failed: %s", e)
        
        # 回退到原生保存
        try:
            state_dict = model.state_dict()
            torch.save(state_dict, path)
            logger.info("Distributed checkpoint saved to %s", path)
        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)
    
    def load_checkpoint(self, model: nn.Module, path: str) -> nn.Module:
        """
        加载分布式检查点
        
        使用底层分布式管理器的检查点加载能力
        """
        # 使用包装器加载
        if self._fsdp_wrapper is not None and hasattr(self._fsdp_wrapper, 'load_checkpoint'):
            try:
                self._fsdp_wrapper.load_checkpoint(path)
                logger.info(f"Checkpoint loaded via FSDPWrapper from {path}")
                return model
            except Exception as e:
                logger.warning(f"FSDPWrapper load failed: {e}")
        
        if self._zero_wrapper is not None and hasattr(self._zero_wrapper, 'load_checkpoint'):
            try:
                self._zero_wrapper.load_checkpoint(path)
                logger.info(f"Checkpoint loaded via ZeROWrapper from {path}")
                return model
            except Exception as e:
                logger.warning(f"ZeROWrapper load failed: {e}")
        
        # 使用分布式管理器
        if self._distributed_manager is not None:
            try:
                # DistributedManager 没有 load_checkpoint 方法，回退到原生加载
                pass
            except Exception as e:
                logger.warning("DistributedManager check failed: %s", e)
        
        # 回退到原生加载
        map_location = f'cuda:{self.config.local_rank}' if self._device.type == 'cuda' else 'cpu'
        state_dict = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(state_dict)
        logger.info("Distributed checkpoint loaded from %s", path)
        return model
    
    def recover_from_error(self, context: StrategyContext, error: Exception) -> bool:
        """
        从错误中恢复
        
        行业场景下的容错恢复
        """
        # pylint: disable=unused-argument
        if not self.config.auto_recovery:
            return False
        
        if self._recovery_attempts >= self.config.max_recovery_attempts:
            logger.error(f"Max recovery attempts ({self.config.max_recovery_attempts}) reached")
            return False
        
        self._recovery_attempts += 1
        logger.warning(f"Attempting recovery ({self._recovery_attempts}/{self.config.max_recovery_attempts}): {error}")
        
        try:
            # 尝试清理内存
            self.optimize_memory()
            
            # 尝试同步
            self._sync_gradients()
            
            # 检查健康状态
            self._check_health()
            
            if self._health_status.is_healthy:
                logger.info("Recovery successful")
                return True
            else:
                logger.warning(f"Recovery incomplete: {self._health_status.issues}")
                return False
                
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return False
    
    def get_device_type(self) -> Optional['DeviceType']:
        """获取设备类型"""
        return self._device_type
    
    def get_hardware_config(self) -> Optional['HardwareConfig']:
        """获取硬件配置"""
        return self._hardware_config


# ==================== 工具函数 ====================

def create_distributed_strategy(
    mode: Union[str, DistributedMode] = 'ddp',
    world_size: int = 1,
    rank: int = 0,
    **kwargs
) -> DistributedStrategy:
    """
    创建分布式策略
    
    Args:
        mode: 分布式模式
        world_size: 世界大小
        rank: 当前进程排名
        **kwargs: 其他配置参数
    
    Returns:
        DistributedStrategy 实例
    """
    if isinstance(mode, str):
        mode = DistributedMode.from_string(mode)
    
    config = DistributedStrategyConfig(
        mode=mode,
        world_size=world_size,
        rank=rank,
        **{k: v for k, v in kwargs.items() if hasattr(DistributedStrategyConfig, k)}
    )
    
    return DistributedStrategy(config)


def create_industry_distributed_strategy(
    mode: Union[str, DistributedMode] = 'fsdp',
    world_size: int = 1,
    rank: int = 0,
    **kwargs
) -> IndustryDistributedStrategy:
    """
    创建行业分布式策略
    
    Args:
        mode: 分布式模式（默认FSDP）
        world_size: 世界大小
        rank: 当前进程排名
        **kwargs: 其他配置参数
    
    Returns:
        IndustryDistributedStrategy 实例
    """
    if isinstance(mode, str):
        mode = DistributedMode.from_string(mode)
    
    config = DistributedStrategyConfig(
        mode=mode,
        world_size=world_size,
        rank=rank,
        enable_monitoring=True,
        auto_recovery=True,
        **{k: v for k, v in kwargs.items() if hasattr(DistributedStrategyConfig, k)}
    )
    
    return IndustryDistributedStrategy(config)


def recommend_distributed_mode(
    model_size_gb: float,
    num_gpus: int,
    memory_per_gpu_gb: float = 16.0
) -> Dict[str, Any]:
    """
    推荐分布式模式
    
    Args:
        model_size_gb: 模型大小 (GB)
        num_gpus: GPU数量
        memory_per_gpu_gb: 每GPU内存 (GB)
    
    Returns:
        推荐配置
    """
    recommended_mode = DistributedMode.recommend(model_size_gb, num_gpus, memory_per_gpu_gb)
    
    recommendation = {
        'mode': recommended_mode.value,
        'memory_efficiency': recommended_mode.memory_efficiency,
        'supports_cpu_offload': recommended_mode.supports_cpu_offload,
        'reasoning': [],
    }
    
    # 添加推理说明
    if model_size_gb > memory_per_gpu_gb * num_gpus:
        recommendation['reasoning'].append(
            f"Model ({model_size_gb:.1f}GB) exceeds total GPU memory ({memory_per_gpu_gb * num_gpus:.1f}GB)"
        )
        recommendation['reasoning'].append("Consider using ZeRO-3 or FSDP with CPU offload")
    elif model_size_gb > memory_per_gpu_gb:
        recommendation['reasoning'].append(
            f"Model ({model_size_gb:.1f}GB) exceeds single GPU memory ({memory_per_gpu_gb:.1f}GB)"
        )
        recommendation['reasoning'].append("FSDP or ZeRO-2 recommended for memory efficiency")
    else:
        recommendation['reasoning'].append("Model fits in single GPU memory")
        recommendation['reasoning'].append("DDP is sufficient for best throughput")
    
    # ZeRO阶段推荐
    if recommended_mode == DistributedMode.ZERO:
        if model_size_gb > memory_per_gpu_gb * num_gpus * 0.8:
            recommendation['zero_stage'] = ZeROStage.STAGE_3.value
            recommendation['reasoning'].append("ZeRO Stage 3 for maximum memory efficiency")
        else:
            recommendation['zero_stage'] = ZeROStage.STAGE_2.value
            recommendation['reasoning'].append("ZeRO Stage 2 for good balance")
    
    return recommendation


def print_distributed_recommendation(
    model_size_gb: float,
    num_gpus: int,
    memory_per_gpu_gb: float = 16.0
) -> None:
    """打印分布式推荐"""
    rec = recommend_distributed_mode(model_size_gb, num_gpus, memory_per_gpu_gb)
    
    print("\n" + "=" * 60)
    print("Distributed Training Recommendation")
    print("=" * 60)
    print(f"Model Size: {model_size_gb:.1f} GB")
    print(f"GPUs: {num_gpus}")
    print(f"Memory per GPU: {memory_per_gpu_gb:.1f} GB")
    print(f"\nRecommended Mode: {rec['mode']}")
    print(f"Memory Efficiency: {rec['memory_efficiency']:.0%}")
    if 'zero_stage' in rec:
        print(f"ZeRO Stage: {rec['zero_stage']}")
    print("\nReasoning:")
    for r in rec['reasoning']:
        print(f"  - {r}")
    print("=" * 60)


def diagnose_distributed_strategy(strategy: DistributedStrategy) -> Dict[str, Any]:
    """诊断分布式策略"""
    return strategy.diagnose()


def print_distributed_diagnosis(strategy: DistributedStrategy) -> None:
    """打印分布式策略诊断"""
    strategy.print_diagnosis()


def get_available_distributed_modes() -> List[str]:
    """获取可用的分布式模式"""
    return [mode.value for mode in DistributedMode]


def get_available_zero_stages() -> List[int]:
    """获取可用的ZeRO阶段"""
    return [stage.value for stage in ZeROStage]


def compare_distributed_modes() -> Dict[str, Dict[str, Any]]:
    """比较分布式模式"""
    comparison = {}
    for mode in DistributedMode:
        comparison[mode.value] = {
            'memory_efficiency': mode.memory_efficiency,
            'supports_cpu_offload': mode.supports_cpu_offload,
            'requires_multiple_gpus': mode.requires_multiple_gpus,
        }
    return comparison


def print_distributed_modes_comparison() -> None:
    """打印分布式模式比较"""
    comparison = compare_distributed_modes()
    
    print("\n" + "=" * 70)
    print("Distributed Modes Comparison")
    print("=" * 70)
    print(f"{'Mode':<12} {'Memory Eff.':<15} {'CPU Offload':<15} {'Multi-GPU':<12}")
    print("-" * 70)
    for mode, info in comparison.items():
        print(f"{mode:<12} {info['memory_efficiency']:.0%}{'':>10} "
              f"{'Yes' if info['supports_cpu_offload'] else 'No':<15} "
              f"{'Yes' if info['requires_multiple_gpus'] else 'No':<12}")
    print("=" * 70)


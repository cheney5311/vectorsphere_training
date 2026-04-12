# -*- coding: utf-8 -*-
"""
FSDP (Fully Sharded Data Parallel) 包装器

提供PyTorch FSDP的封装和管理，支持大模型训练的生产级功能，包括：
- 智能分片策略选择
- 内存监控和优化
- 梯度同步和通信优化
- 检查点管理（完整/分片/流式）
- 性能分析和诊断
- 自动调优
"""

import os
import gc
import time
import logging
import threading
from typing import Optional, List, Any, Dict, Union, Type, Callable, Iterator, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import partial, wraps
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)

# 尝试导入FSDP
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        BackwardPrefetch,
        CPUOffload,
        StateDictType,
        FullStateDictConfig,
        LocalStateDictConfig,
        ShardedStateDictConfig,
        OptimStateDictConfig,
    )
    from torch.distributed.fsdp.wrap import (
        transformer_auto_wrap_policy,
        size_based_auto_wrap_policy,
        ModuleWrapPolicy,
        enable_wrap,
        wrap,
    )
    from torch.distributed.fsdp.api import FullOptimStateDictConfig, LocalOptimStateDictConfig
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False
    logger.warning("FSDP not available, using fallback")

# 尝试导入检查点相关
try:
    from torch.distributed.checkpoint import (
        save_state_dict,
        load_state_dict,
        FileSystemReader,
        FileSystemWriter,
    )
    DISTRIBUTED_CHECKPOINT_AVAILABLE = True
except ImportError:
    DISTRIBUTED_CHECKPOINT_AVAILABLE = False


# ==================== 枚举和配置类 ====================

class FSDPShardingStrategy(Enum):
    """
    FSDP分片策略
    
    提供不同的内存-通信权衡选项。
    """
    FULL_SHARD = "full_shard"      # 完全分片：最省内存，通信最多
    SHARD_GRAD_OP = "shard_grad_op"  # 梯度和优化器分片：中等
    NO_SHARD = "no_shard"          # 不分片：类似DDP
    HYBRID_SHARD = "hybrid_shard"  # 混合分片：节点内分片

    @property
    def memory_efficiency(self) -> float:
        """
        内存效率系数 (0-1)
        
        值越高表示越省内存。
        """
        efficiency_map = {
            self.FULL_SHARD: 1.0,
            self.HYBRID_SHARD: 0.8,
            self.SHARD_GRAD_OP: 0.6,
            self.NO_SHARD: 0.0,
        }
        return efficiency_map.get(self, 0.0)
    
    @property
    def communication_overhead(self) -> str:
        """通信开销等级"""
        overhead_map = {
            self.FULL_SHARD: "high",
            self.HYBRID_SHARD: "medium-high",
            self.SHARD_GRAD_OP: "medium",
            self.NO_SHARD: "low",
        }
        return overhead_map.get(self, "unknown")
    
    @classmethod
    def from_memory_constraint(cls, available_memory_gb: float, model_memory_gb: float) -> 'FSDPShardingStrategy':
        """
        根据内存约束自动选择策略
        
        Args:
            available_memory_gb: 可用GPU内存
            model_memory_gb: 模型所需内存
            
        Returns:
            推荐的分片策略
        """
        ratio = model_memory_gb / available_memory_gb
        
        if ratio <= 0.5:
            return cls.NO_SHARD
        elif ratio <= 0.7:
            return cls.SHARD_GRAD_OP
        elif ratio <= 0.9:
            return cls.HYBRID_SHARD
        else:
            return cls.FULL_SHARD
    
    def to_pytorch_strategy(self) -> 'ShardingStrategy':
        """转换为PyTorch ShardingStrategy"""
        if not FSDP_AVAILABLE:
            raise RuntimeError("FSDP not available")
        
        strategy_map = {
            self.FULL_SHARD: ShardingStrategy.FULL_SHARD,
            self.SHARD_GRAD_OP: ShardingStrategy.SHARD_GRAD_OP,
            self.NO_SHARD: ShardingStrategy.NO_SHARD,
            self.HYBRID_SHARD: ShardingStrategy.HYBRID_SHARD,
        }
        return strategy_map[self]


@dataclass
class FSDPMixedPrecisionConfig:
    """
    混合精度配置
    
    控制FSDP中各种数据类型的使用。
    """
    param_dtype: str = "fp32"  # fp32, fp16, bf16
    reduce_dtype: str = "fp32"
    buffer_dtype: str = "fp32"

    # 是否在CPU上保持fp32主副本
    keep_low_precision_grads: bool = False
    
    # 是否cast forward输入
    cast_forward_inputs: bool = True
    
    @classmethod
    def bf16_mixed(cls) -> 'FSDPMixedPrecisionConfig':
        """BF16混合精度配置（推荐用于A100/H100）"""
        return cls(
            param_dtype="bf16",
            reduce_dtype="bf16",
            buffer_dtype="bf16",
        )
    
    @classmethod
    def fp16_mixed(cls) -> 'FSDPMixedPrecisionConfig':
        """FP16混合精度配置"""
        return cls(
            param_dtype="fp16",
            reduce_dtype="fp16",
            buffer_dtype="fp16",
        )
    
    @classmethod
    def fp32_reduce(cls) -> 'FSDPMixedPrecisionConfig':
        """FP32归约配置（更高精度的梯度累积）"""
        return cls(
            param_dtype="bf16",
            reduce_dtype="fp32",
            buffer_dtype="fp32",
        )
    
    def to_pytorch_mixed_precision(self) -> Optional['MixedPrecision']:
        """转换为PyTorch MixedPrecision"""
        if not FSDP_AVAILABLE:
            return None
        
        dtype_map = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
        }
        
        return MixedPrecision(
            param_dtype=dtype_map.get(self.param_dtype, torch.float32),
            reduce_dtype=dtype_map.get(self.reduce_dtype, torch.float32),
            buffer_dtype=dtype_map.get(self.buffer_dtype, torch.float32),
        )
    
    def validate(self) -> List[str]:
        """验证配置"""
        warnings = []
        valid_dtypes = {'fp32', 'fp16', 'bf16'}
        
        for name in ['param_dtype', 'reduce_dtype', 'buffer_dtype']:
            dtype = getattr(self, name)
            if dtype not in valid_dtypes:
                raise ValueError(f"{name} must be one of {valid_dtypes}, got '{dtype}'")
        
        # BF16需要特定硬件支持
        if 'bf16' in [self.param_dtype, self.reduce_dtype, self.buffer_dtype]:
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                warnings.append("BF16 may not be fully supported on this GPU")
        
        return warnings


@dataclass
class FSDPCheckpointConfig:
    """
    检查点配置
    
    控制FSDP模型的保存和加载行为。
    """
    # 状态字典类型
    state_dict_type: str = "full"  # full, local, sharded
    
    # 是否只在rank 0保存
    rank0_only: bool = True
    
    # 是否使用分布式检查点（torch.distributed.checkpoint）
    use_distributed_checkpoint: bool = False
    
    # 异步保存
    async_save: bool = False
    
    # 压缩
    compression: Optional[str] = None  # None, "gzip", "lz4"
    
    # 流式保存（减少内存峰值）
    streaming: bool = False
    
    def get_state_dict_type(self) -> 'StateDictType':
        """获取PyTorch StateDictType"""
        if not FSDP_AVAILABLE:
            raise RuntimeError("FSDP not available")
        
        type_map = {
            'full': StateDictType.FULL_STATE_DICT,
            'local': StateDictType.LOCAL_STATE_DICT,
            'sharded': StateDictType.SHARDED_STATE_DICT,
        }
        return type_map.get(self.state_dict_type, StateDictType.FULL_STATE_DICT)


@dataclass
class FSDPMemoryConfig:
    """
    内存优化配置
    
    控制FSDP的内存使用优化。
    """
    # CPU卸载
    cpu_offload: bool = False
    offload_params: bool = False
    
    # 内存限制（GB）
    memory_limit_gb: Optional[float] = None
    
    # 预取策略
    backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST, None
    forward_prefetch: bool = False
    
    # 限制all_gathers
    limit_all_gathers: bool = True
    
    # 激活检查点
    activation_checkpointing: bool = False
    checkpoint_every_n_layers: int = 1
    
    # 梯度累积优化
    sync_module_states: bool = True
    
    def get_cpu_offload(self) -> Optional['CPUOffload']:
        """获取CPU卸载配置"""
        if not FSDP_AVAILABLE or not self.cpu_offload:
            return None
        return CPUOffload(offload_params=self.offload_params)
    
    def get_backward_prefetch(self) -> Optional['BackwardPrefetch']:
        """获取反向预取配置"""
        if not FSDP_AVAILABLE or self.backward_prefetch == "None":
            return None
        
        prefetch_map = {
            'BACKWARD_PRE': BackwardPrefetch.BACKWARD_PRE,
            'BACKWARD_POST': BackwardPrefetch.BACKWARD_POST,
        }
        return prefetch_map.get(self.backward_prefetch, BackwardPrefetch.BACKWARD_PRE)


# ==================== 内存监控 ====================

class FSDPMemoryMonitor:
    """
    FSDP内存监控器
    
    监控GPU和CPU内存使用，提供优化建议。
    """
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._history: List[Dict[str, float]] = []
        self._peak_memory: float = 0.0
        self._monitoring: bool = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        获取当前内存统计
        
        Returns:
            内存统计字典（单位：GB）
        """
        if not torch.cuda.is_available():
            return {'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 0, 'total_gb': 0, 'utilization': 0}
        
        torch.cuda.synchronize(self.device_id)
        
        allocated = torch.cuda.memory_allocated(self.device_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device_id) / (1024**3)
        total = torch.cuda.get_device_properties(self.device_id).total_memory / (1024**3)
        free = total - reserved
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
            'total_gb': total,
            'utilization': allocated / total if total > 0 else 0,
        }
    
    def get_peak_memory(self) -> float:
        """获取峰值内存使用（GB）"""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated(self.device_id) / (1024**3)
    
    def reset_peak_memory(self) -> None:
        """重置峰值内存统计"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
        self._peak_memory = 0.0
    
    def record(self, tag: str = "") -> Dict[str, float]:
        """
        记录当前内存状态
        
        Args:
            tag: 标记名称
            
        Returns:
            记录的内存统计
        """
        stats = self.get_memory_stats()
        stats['tag'] = tag
        stats['timestamp'] = time.time()
        self._history.append(stats)
        
        peak = self.get_peak_memory()
        if peak > self._peak_memory:
            self._peak_memory = peak
        
        return stats
    
    def start_monitoring(self, interval: float = 1.0) -> None:
        """
        开始后台监控
        
        Args:
            interval: 监控间隔（秒）
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        
        def monitor_loop():
            while self._monitoring:
                self.record("auto")
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.debug("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """停止后台监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            self._monitor_thread = None
        logger.debug("Memory monitoring stopped")
    
    def get_history(self) -> List[Dict[str, float]]:
        """获取内存历史"""
        return self._history.copy()
    
    def clear_history(self) -> None:
        """清除历史记录"""
        self._history.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取内存使用摘要
        
        Returns:
            摘要统计
        """
        if not self._history:
            return {'message': 'No memory records'}
        
        allocated_values = [h['allocated_gb'] for h in self._history]
        
        return {
            'peak_memory_gb': self._peak_memory,
            'avg_allocated_gb': sum(allocated_values) / len(allocated_values),
            'max_allocated_gb': max(allocated_values),
            'min_allocated_gb': min(allocated_values),
            'num_records': len(self._history),
        }
    
    def suggest_optimizations(self) -> List[str]:
        """
        根据内存使用情况提供优化建议
        
        Returns:
            优化建议列表
        """
        suggestions = []
        stats = self.get_memory_stats()
        
        if stats['utilization'] > 0.9:
            suggestions.append("Memory utilization > 90%: Consider enabling CPU offload or activation checkpointing")
        
        if stats['utilization'] > 0.95:
            suggestions.append("Critical memory usage: Reduce batch size or enable FULL_SHARD strategy")
        
        if self._peak_memory > stats['total_gb'] * 0.85:
            suggestions.append("Peak memory is high: Enable gradient checkpointing to reduce activation memory")
        
        reserved_waste = stats['reserved_gb'] - stats['allocated_gb']
        if reserved_waste > 2.0:  # 超过2GB的保留但未分配内存
            suggestions.append(f"Reserved but unused memory: {reserved_waste:.2f}GB. Consider torch.cuda.empty_cache()")
        
        return suggestions
    
    @contextmanager
    def track_memory(self, tag: str):
        """
        内存追踪上下文管理器
        
        Args:
            tag: 追踪标签
        """
        start_stats = self.record(f"{tag}_start")
        start_peak = self.get_peak_memory()
        
        try:
            yield
        finally:
            end_stats = self.record(f"{tag}_end")
            end_peak = self.get_peak_memory()
            
            delta_allocated = end_stats['allocated_gb'] - start_stats['allocated_gb']
            delta_peak = end_peak - start_peak
            
            logger.debug(f"Memory [{tag}]: allocated delta={delta_allocated:.3f}GB, peak delta={delta_peak:.3f}GB")


# ==================== 性能分析器 ====================

class FSDPProfiler:
    """
    FSDP性能分析器
    
    分析FSDP训练的性能瓶颈。
    """
    
    def __init__(self):
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._comm_stats: Dict[str, Any] = {}
        self._enabled: bool = False
    
    def enable(self) -> None:
        """启用分析"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用分析"""
        self._enabled = False
    
    @contextmanager
    def profile_region(self, name: str):
        """
        分析代码区域
        
        Args:
            name: 区域名称
        """
        if not self._enabled:
            yield
            return
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start_time
            self._timings[name].append(elapsed)
    
    def record_timing(self, name: str, duration: float) -> None:
        """记录时间"""
        if self._enabled:
            self._timings[name].append(duration)
    
    def get_timing_stats(self, name: str) -> Dict[str, float]:
        """
        获取特定区域的时间统计
        
        Args:
            name: 区域名称
            
        Returns:
            统计信息
        """
        timings = self._timings.get(name, [])
        if not timings:
            return {}
        
        return {
            'count': len(timings),
            'total_ms': sum(timings) * 1000,
            'avg_ms': sum(timings) / len(timings) * 1000,
            'min_ms': min(timings) * 1000,
            'max_ms': max(timings) * 1000,
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有统计"""
        return {name: self.get_timing_stats(name) for name in self._timings}
    
    def reset(self) -> None:
        """重置统计"""
        self._timings.clear()
        self._comm_stats.clear()
    
    def print_summary(self) -> None:
        """打印性能摘要"""
        print("\n=== FSDP Performance Summary ===")
        
        stats = self.get_all_stats()
        if not stats:
            print("No profiling data collected")
            return
        
        # 按总时间排序
        sorted_stats = sorted(stats.items(), key=lambda x: x[1].get('total_ms', 0), reverse=True)
        
        total_time = sum(s.get('total_ms', 0) for _, s in sorted_stats)
        
        print(f"{'Region':<30} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'%':>8}")
        print("-" * 70)
        
        for name, stat in sorted_stats:
            if stat:
                pct = stat['total_ms'] / total_time * 100 if total_time > 0 else 0
                print(f"{name:<30} {stat['count']:>8} {stat['total_ms']:>12.2f} {stat['avg_ms']:>10.2f} {pct:>7.1f}%")
        
        print("-" * 70)
        print(f"{'Total':<30} {'':<8} {total_time:>12.2f}")


# ==================== FSDP上下文管理器 ====================

class FSDPContext:
    """
    FSDP上下文管理器
    
    管理FSDP的初始化、状态和清理。
    """
    
    def __init__(self, config):
        from .parallel_modes import FSDPConfig
        self.config: FSDPConfig = config
        self._initialized = False
        self._memory_monitor: Optional[FSDPMemoryMonitor] = None
        self._profiler: Optional[FSDPProfiler] = None
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
    
    def initialize(self) -> None:
        """初始化FSDP环境"""
        if self._initialized:
            return
        
        if not dist.is_initialized():
            self._init_process_group()
        
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
        
        # 初始化内存监控
        self._memory_monitor = FSDPMemoryMonitor(self.config.local_rank)
        
        # 初始化性能分析器
        self._profiler = FSDPProfiler()
        
        self._initialized = True
        logger.info(f"FSDP context initialized: rank={self.config.rank}/{self.config.world_size}")
    
    def _init_process_group(self) -> None:
        """初始化进程组"""
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = str(self.config.master_port)
        
        backend = self.config.backend
        if hasattr(backend, 'value'):
            backend = backend.value
        
        dist.init_process_group(
            backend=backend,
            world_size=self.config.world_size,
            rank=self.config.rank
        )
    
    def cleanup(self) -> None:
        """清理FSDP资源"""
        # 停止内存监控
        if self._memory_monitor:
            self._memory_monitor.stop_monitoring()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # 销毁进程组
        if dist.is_initialized():
            dist.destroy_process_group()
        
        self._initialized = False
        logger.info("FSDP context cleaned up")
    
    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self._initialized
    
    @property
    def is_main_process(self) -> bool:
        """是否是主进程"""
        return self.config.rank == 0
    
    @property
    def device(self) -> torch.device:
        """当前设备"""
        if torch.cuda.is_available():
            return torch.device(f'cuda:{self.config.local_rank}')
        return torch.device('cpu')
    
    @property
    def memory_monitor(self) -> Optional[FSDPMemoryMonitor]:
        """内存监控器"""
        return self._memory_monitor
    
    @property
    def profiler(self) -> Optional[FSDPProfiler]:
        """性能分析器"""
        return self._profiler
    
    def barrier(self) -> None:
        """同步屏障"""
        if dist.is_initialized():
            dist.barrier()
    
    def get_state(self) -> Dict[str, Any]:
        """获取上下文状态"""
        return {
            'initialized': self._initialized,
            'rank': self.config.rank,
            'world_size': self.config.world_size,
            'local_rank': self.config.local_rank,
            'device': str(self.device),
            'memory_stats': self._memory_monitor.get_memory_stats() if self._memory_monitor else {},
        }


# ==================== FSDP包装器 ====================

class FSDPWrapper:
    """
    FSDP模型包装器
    
    封装PyTorch FSDP的创建和管理，提供生产级功能。
    """
    
    def __init__(self, config=None):
        from .parallel_modes import FSDPConfig
        self.config: FSDPConfig = config or FSDPConfig()
        
        self._model: Optional[nn.Module] = None
        self._fsdp_model: Optional[nn.Module] = None
        self._context: Optional[FSDPContext] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        
        # 组件
        self._memory_monitor: Optional[FSDPMemoryMonitor] = None
        self._profiler: FSDPProfiler = FSDPProfiler()
        self._checkpoint_config: FSDPCheckpointConfig = FSDPCheckpointConfig()
        
        # 状态
        self._is_wrapped: bool = False
        self._grad_sync_enabled: bool = True
        self._step_count: int = 0
    
    # ==================== 模型包装 ====================
    
    def wrap(
        self, 
        model: nn.Module,
        auto_wrap_policy: Optional[Callable] = None,
        transformer_layer_cls: Optional[List[Type[nn.Module]]] = None,
        ignored_modules: Optional[List[nn.Module]] = None,
    ) -> nn.Module:
        """
        将模型包装为FSDP模型
        
        Args:
            model: 原始模型
            auto_wrap_policy: 自动包装策略
            transformer_layer_cls: Transformer层类列表
            ignored_modules: 忽略的模块列表
            
        Returns:
            FSDP包装后的模型
        """
        if not FSDP_AVAILABLE:
            logger.warning("FSDP not available, returning original model")
            self._model = model
            return model
        
        self._model = model
        
        # 确保环境已初始化
        self._ensure_initialized()
        
        # 记录包装前内存
        if self._memory_monitor:
            self._memory_monitor.record("pre_wrap")
        
        # 分析模型
        model_info = self._analyze_model(model)
        logger.info(f"Wrapping model: {model_info['num_params'] / 1e6:.2f}M params, "
                   f"{model_info['num_layers']} layers")
        
        # 配置包装策略
        fsdp_kwargs = self._build_fsdp_kwargs(
            auto_wrap_policy, transformer_layer_cls, ignored_modules
        )
        
        # 创建FSDP模型
        with self._profiler.profile_region("fsdp_wrap"):
            self._fsdp_model = FSDP(model, **fsdp_kwargs)
        
        self._is_wrapped = True
        
        # 记录包装后内存
        if self._memory_monitor:
            self._memory_monitor.record("post_wrap")
        
        # 验证包装
        self._validate_wrapped_model()
        
        logger.info(f"Model wrapped with FSDP: strategy={self.config.sharding_strategy}")
        return self._fsdp_model
    
    def _ensure_initialized(self) -> None:
        """确保环境已初始化"""
        if not dist.is_initialized():
            self._context = FSDPContext(self.config)
            self._context.initialize()
        
        # 初始化内存监控
        if self._memory_monitor is None:
            self._memory_monitor = FSDPMemoryMonitor(self.config.local_rank)
    
    def _analyze_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        分析模型结构
        
        Args:
            model: 模型
            
        Returns:
            模型分析信息
        """
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_layers = len(list(model.modules()))
        
        # 统计各类型层
        layer_types = defaultdict(int)
        for name, module in model.named_modules():
            layer_types[type(module).__name__] += 1
        
        return {
            'num_params': num_params,
            'num_trainable_params': num_trainable,
            'num_layers': num_layers,
            'layer_types': dict(layer_types),
            'param_memory_mb': num_params * 4 / (1024**2),  # fp32
        }
    
    def _build_fsdp_kwargs(
        self,
        auto_wrap_policy: Optional[Callable],
        transformer_layer_cls: Optional[List[Type[nn.Module]]],
        ignored_modules: Optional[List[nn.Module]],
    ) -> Dict[str, Any]:
        """构建FSDP参数"""
        kwargs = {
            'sharding_strategy': self._get_sharding_strategy(),
            'mixed_precision': self._get_mixed_precision(),
            'cpu_offload': self._get_cpu_offload(),
            'backward_prefetch': self._get_backward_prefetch(),
            'device_id': self.config.local_rank if torch.cuda.is_available() else None,
            'use_orig_params': True,
            'limit_all_gathers': True,
            'forward_prefetch': self.config.forward_prefetch,
        }
        
        # 自动包装策略
        if auto_wrap_policy is None:
            auto_wrap_policy = self._get_auto_wrap_policy(transformer_layer_cls)
        
        if auto_wrap_policy is not None:
            kwargs['auto_wrap_policy'] = auto_wrap_policy
        
        # 忽略模块
        if ignored_modules:
            kwargs['ignored_modules'] = ignored_modules
        
        return kwargs
    
    def _validate_wrapped_model(self) -> None:
        """验证包装后的模型"""
        if self._fsdp_model is None:
            return
        
        # 检查参数是否正确分片
        total_params = sum(p.numel() for p in self._fsdp_model.parameters())
        
        # 简单的健全性检查
        if total_params == 0:
            logger.warning("Wrapped model has 0 parameters, this may indicate an issue")
    
    def _get_sharding_strategy(self):
        """获取分片策略"""
        if not FSDP_AVAILABLE:
            return None
        
        strategy_map = {
            'full_shard': ShardingStrategy.FULL_SHARD,
            'shard_grad_op': ShardingStrategy.SHARD_GRAD_OP,
            'no_shard': ShardingStrategy.NO_SHARD,
            'hybrid_shard': ShardingStrategy.HYBRID_SHARD
        }
        
        strategy_value = self.config.sharding_strategy
        if hasattr(strategy_value, 'value'):
            strategy_value = strategy_value.value
        
        return strategy_map.get(strategy_value, ShardingStrategy.FULL_SHARD)
    
    def _get_mixed_precision(self) -> Optional[MixedPrecision]:
        """获取混合精度配置"""
        if not FSDP_AVAILABLE or not self.config.mixed_precision:
            return None
        
        dtype_map = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16
        }
        
        return MixedPrecision(
            param_dtype=dtype_map.get(self.config.param_dtype, torch.float32),
            reduce_dtype=dtype_map.get(self.config.reduce_dtype, torch.float32),
            buffer_dtype=dtype_map.get(self.config.buffer_dtype, torch.float32)
        )
    
    def _get_cpu_offload(self) -> Optional[CPUOffload]:
        """获取CPU Offload配置"""
        if not FSDP_AVAILABLE or not self.config.cpu_offload:
            return None
        
        return CPUOffload(offload_params=self.config.offload_params)
    
    def _get_backward_prefetch(self) -> Optional[BackwardPrefetch]:
        """获取反向预取配置"""
        if not FSDP_AVAILABLE:
            return None
        
        prefetch_map = {
            'BACKWARD_PRE': BackwardPrefetch.BACKWARD_PRE,
            'BACKWARD_POST': BackwardPrefetch.BACKWARD_POST
        }
        return prefetch_map.get(self.config.backward_prefetch, BackwardPrefetch.BACKWARD_PRE)
    
    def _get_auto_wrap_policy(
        self, 
        transformer_layer_cls: Optional[List[Type[nn.Module]]] = None
    ) -> Optional[Callable]:
        """获取自动包装策略"""
        if not FSDP_AVAILABLE:
            return None
        
        if self.config.auto_wrap_policy == "transformer_auto_wrap":
            if transformer_layer_cls:
                return partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=set(transformer_layer_cls)
                )
        elif self.config.auto_wrap_policy == "size_based":
            return partial(
                size_based_auto_wrap_policy,
                min_num_params=self.config.min_num_params
            )
        
        return None
    
    # ==================== 模型访问 ====================
    
    def unwrap(self) -> nn.Module:
        """解包FSDP模型"""
        if self._fsdp_model is not None:
            return self._fsdp_model.module
        return self._model
    
    @property
    def module(self) -> nn.Module:
        """获取内部模型"""
        return self.unwrap()
    
    @property
    def fsdp_model(self) -> Optional[nn.Module]:
        """获取FSDP包装后的模型"""
        return self._fsdp_model
    
    @property
    def is_wrapped(self) -> bool:
        """是否已包装"""
        return self._is_wrapped
    
    # ==================== 训练控制 ====================
    
    def forward(self, *args, **kwargs) -> Any:
        """前向传播"""
        model = self._fsdp_model or self._model
        if model is None:
            raise RuntimeError("No model available")
        
        with self._profiler.profile_region("forward"):
            return model(*args, **kwargs)
    
    def backward(self, loss: torch.Tensor) -> None:
        """反向传播"""
        with self._profiler.profile_region("backward"):
            loss.backward()
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """优化器步进"""
        with self._profiler.profile_region("optimizer_step"):
            optimizer.step()
        
        self._step_count += 1
    
    def zero_grad(self, optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """清零梯度"""
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        elif self._optimizer:
            self._optimizer.zero_grad(set_to_none=True)
        elif self._fsdp_model:
            self._fsdp_model.zero_grad(set_to_none=True)
    
    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """设置优化器"""
        self._optimizer = optimizer
    
    # ==================== 梯度同步控制 ====================
    
    def enable_grad_sync(self) -> None:
        """启用梯度同步"""
        self._grad_sync_enabled = True
    
    def disable_grad_sync(self) -> None:
        """禁用梯度同步（用于梯度累积）"""
        self._grad_sync_enabled = False
    
    @contextmanager
    def no_sync(self):
        """
        无同步上下文（用于梯度累积）
        
        在此上下文中，梯度不会跨进程同步。
        """
        if self._fsdp_model is None:
            yield
            return
        
        # FSDP的no_sync上下文
        with self._fsdp_model.no_sync():
            yield
    
    def sync_gradients(self) -> None:
        """显式同步梯度"""
        if self._fsdp_model and dist.is_initialized():
            # FSDP自动同步，但可以在这里添加额外逻辑
            pass
    
    # ==================== 梯度裁剪 ====================
    
    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
        """
        梯度裁剪
        
        FSDP需要特殊处理，使用内置方法。
        
        Args:
            max_norm: 最大梯度范数
            norm_type: 范数类型
            
        Returns:
            总梯度范数
        """
        if not FSDP_AVAILABLE or self._fsdp_model is None:
            return torch.nn.utils.clip_grad_norm_(
                self._model.parameters() if self._model else [],
                max_norm,
                norm_type=norm_type
            )
        
        with self._profiler.profile_region("clip_grad_norm"):
            return self._fsdp_model.clip_grad_norm_(max_norm, norm_type=norm_type)
    
    def get_grad_norm(self, norm_type: float = 2.0) -> torch.Tensor:
        """
        获取梯度范数
        
        Args:
            norm_type: 范数类型
            
        Returns:
            梯度范数
        """
        model = self._fsdp_model or self._model
        if model is None:
            return torch.tensor(0.0)
        
        total_norm = torch.tensor(0.0, device=self.config.get_device())
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm ** norm_type
        
        total_norm = total_norm ** (1.0 / norm_type)
        
        # 跨进程聚合
        if dist.is_initialized():
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX)
        
        return total_norm
    
    # ==================== 检查点管理 ====================
    
    def get_full_state_dict(self, cpu: bool = True) -> Dict[str, Any]:
        """
        获取完整状态字典
        
        收集所有分片的状态字典。
        
        Args:
            cpu: 是否移到CPU
            
        Returns:
            完整状态字典
        """
        if not FSDP_AVAILABLE or self._fsdp_model is None:
            return self._model.state_dict() if self._model else {}
        
        with self._profiler.profile_region("get_full_state_dict"):
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=cpu,
                rank0_only=self._checkpoint_config.rank0_only
            )
        
        with FSDP.state_dict_type(
            self._fsdp_model,
                StateDictType.FULL_STATE_DICT,
                full_state_dict_config
        ):
            return self._fsdp_model.state_dict()
    
    def get_sharded_state_dict(self) -> Dict[str, Any]:
        """
        获取分片状态字典
        
        每个rank保存自己的分片。
        
        Returns:
            分片状态字典
        """
        if not FSDP_AVAILABLE or self._fsdp_model is None:
            return self._model.state_dict() if self._model else {}
        
        with self._profiler.profile_region("get_sharded_state_dict"):
            with FSDP.state_dict_type(
                self._fsdp_model,
                StateDictType.SHARDED_STATE_DICT
            ):
                return self._fsdp_model.state_dict()
    
    def get_local_state_dict(self) -> Dict[str, Any]:
        """
        获取本地状态字典
        
        返回本地（未展平）的状态字典。
        
        Returns:
            本地状态字典
        """
        if not FSDP_AVAILABLE or self._fsdp_model is None:
            return self._model.state_dict() if self._model else {}
        
        with FSDP.state_dict_type(
            self._fsdp_model,
            StateDictType.LOCAL_STATE_DICT
        ):
            return self._fsdp_model.state_dict()
    
    def get_optimizer_state_dict(
        self,
        optimizer: torch.optim.Optimizer,
        full: bool = True
    ) -> Dict[str, Any]:
        """
        获取优化器状态字典
        
        Args:
            optimizer: 优化器
            full: 是否获取完整状态
            
        Returns:
            优化器状态字典
        """
        if not FSDP_AVAILABLE or self._fsdp_model is None:
            return optimizer.state_dict()
        
        with self._profiler.profile_region("get_optimizer_state_dict"):
            if full:
                optim_state_config = FullOptimStateDictConfig(
                    offload_to_cpu=True,
                    rank0_only=self._checkpoint_config.rank0_only
                )
                
                with FSDP.state_dict_type(
                    self._fsdp_model,
                    StateDictType.FULL_STATE_DICT,
                    optim_state_dict_config=optim_state_config
                ):
                    return FSDP.optim_state_dict(self._fsdp_model, optimizer)
            else:
                return FSDP.optim_state_dict(self._fsdp_model, optimizer)
    
    def save_checkpoint(
        self, 
        path: str, 
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        full_state_dict: bool = True,
        **kwargs
    ) -> None:
        """
        保存检查点
        
        Args:
            path: 保存路径
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            step: 当前step
            full_state_dict: 是否保存完整状态字典
            **kwargs: 额外数据
        """
        with self._profiler.profile_region("save_checkpoint"):
            if full_state_dict:
                self._save_full_checkpoint(path, optimizer, scheduler, epoch, step, **kwargs)
            else:
                self._save_sharded_checkpoint(path, optimizer, scheduler, epoch, step, **kwargs)
    
    def _save_full_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        epoch: Optional[int],
        step: Optional[int],
        **kwargs
    ) -> None:
        """保存完整检查点"""
        # 只在rank 0保存
        if self.config.rank != 0 and self._checkpoint_config.rank0_only:
            # 其他rank需要参与state_dict收集
            _ = self.get_full_state_dict()
            if optimizer:
                _ = self.get_optimizer_state_dict(optimizer, full=True)
            return
        
        checkpoint = {
            'model_state_dict': self.get_full_state_dict(),
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
            'step_count': self._step_count,
            **kwargs
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = self.get_optimizer_state_dict(optimizer, full=True)
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if step is not None:
            checkpoint['step'] = step
        
        # 创建目录
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # 保存
        torch.save(checkpoint, path)
        logger.info(f"Full checkpoint saved: {path}")
    
    def _save_sharded_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        epoch: Optional[int],
        step: Optional[int],
        **kwargs
    ) -> None:
        """保存分片检查点"""
        checkpoint = {
            'model_state_dict': self.get_sharded_state_dict(),
                'rank': self.config.rank,
                'world_size': self.config.world_size,
            'step_count': self._step_count,
                **kwargs
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        if step is not None:
            checkpoint['step'] = step
        
        # 每个rank保存自己的分片
        rank_path = f"{path}.rank{self.config.rank}"
        os.makedirs(os.path.dirname(rank_path) if os.path.dirname(rank_path) else '.', exist_ok=True)
        
        torch.save(checkpoint, rank_path)
        logger.info(f"Sharded checkpoint saved: {rank_path}")
    
    def load_checkpoint(
        self, 
        path: str, 
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        full_state_dict: bool = True,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            path: 检查点路径
            optimizer: 优化器
            scheduler: 学习率调度器
            full_state_dict: 是否是完整状态字典
            strict: 是否严格匹配
            
        Returns:
            检查点数据
        """
        with self._profiler.profile_region("load_checkpoint"):
            if full_state_dict:
                return self._load_full_checkpoint(path, optimizer, scheduler, strict)
            else:
                return self._load_sharded_checkpoint(path, optimizer, scheduler, strict)
    
    def _load_full_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        strict: bool
    ) -> Dict[str, Any]:
        """加载完整检查点"""
        if not FSDP_AVAILABLE or self._fsdp_model is None:
            checkpoint = torch.load(path, map_location='cpu')
            if self._model:
                self._model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            return checkpoint
        
            checkpoint = torch.load(path, map_location='cpu')
        
        # 加载模型状态
        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True,
            rank0_only=False  # 加载时所有rank都需要
        )
        
        with FSDP.state_dict_type(
            self._fsdp_model,
            StateDictType.FULL_STATE_DICT,
            full_state_dict_config
        ):
            self._fsdp_model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # 加载优化器状态
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optim_state = FSDP.optim_state_dict_to_load(
                checkpoint['optimizer_state_dict'],
                self._fsdp_model,
                optimizer
            )
            optimizer.load_state_dict(optim_state)
        
        # 加载调度器状态
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复内部状态
        if 'step_count' in checkpoint:
            self._step_count = checkpoint['step_count']
        
        # 同步
        if dist.is_initialized():
            dist.barrier()
        
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint
    
    def _load_sharded_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        strict: bool
    ) -> Dict[str, Any]:
        """加载分片检查点"""
        rank_path = f"{path}.rank{self.config.rank}"
        checkpoint = torch.load(rank_path, map_location='cpu')
        
        # 验证
        if checkpoint.get('world_size') != self.config.world_size:
            logger.warning(f"World size mismatch: checkpoint={checkpoint.get('world_size')}, "
                         f"current={self.config.world_size}")
        
        if not FSDP_AVAILABLE or self._fsdp_model is None:
            if self._model:
                self._model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            return checkpoint
        
        with FSDP.state_dict_type(
            self._fsdp_model,
            StateDictType.SHARDED_STATE_DICT
        ):
            self._fsdp_model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'step_count' in checkpoint:
            self._step_count = checkpoint['step_count']
        
        if dist.is_initialized():
            dist.barrier()
        
        logger.info(f"Sharded checkpoint loaded: {rank_path}")
        return checkpoint
    
    # ==================== 参数访问 ====================
    
    def summon_full_params(
        self,
        writeback: bool = True,
        rank0_only: bool = True,
        offload_to_cpu: bool = False
    ):
        """
        召唤完整参数上下文
        
        临时收集所有分片参数到一个或所有rank。
        
        Args:
            writeback: 退出时是否写回修改
            rank0_only: 是否只在rank0收集
            offload_to_cpu: 是否卸载到CPU
        """
        if not FSDP_AVAILABLE or self._fsdp_model is None:
            from contextlib import nullcontext
            return nullcontext()
        
        return FSDP.summon_full_params(
            self._fsdp_model,
            writeback=writeback,
            rank0_only=rank0_only,
            offload_to_cpu=offload_to_cpu
        )
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        获取参数计数
        
        Returns:
            参数统计
        """
        model = self._fsdp_model or self._model
        if model is None:
            return {'total': 0, 'trainable': 0, 'frozen': 0}
        
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': total - trainable,
        }
    
    # ==================== 内存管理 ====================
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计"""
        if self._memory_monitor:
            return self._memory_monitor.get_memory_stats()
        return {}
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存摘要"""
        if self._memory_monitor:
            return self._memory_monitor.get_summary()
        return {}
    
    def get_memory_suggestions(self) -> List[str]:
        """获取内存优化建议"""
        if self._memory_monitor:
            return self._memory_monitor.suggest_optimizations()
        return []
    
    def clear_memory_cache(self) -> None:
        """清理内存缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        if self._memory_monitor:
            self._memory_monitor.record("cache_cleared")
    
    @contextmanager
    def track_memory(self, tag: str):
        """内存追踪上下文"""
        if self._memory_monitor:
            with self._memory_monitor.track_memory(tag):
                yield
        else:
            yield
    
    # ==================== 性能分析 ====================
    
    def enable_profiling(self) -> None:
        """启用性能分析"""
        self._profiler.enable()
    
    def disable_profiling(self) -> None:
        """禁用性能分析"""
        self._profiler.disable()
    
    def get_profiling_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self._profiler.get_all_stats()
    
    def print_profiling_summary(self) -> None:
        """打印性能摘要"""
        self._profiler.print_summary()
    
    def reset_profiling(self) -> None:
        """重置性能统计"""
        self._profiler.reset()
    
    # ==================== 诊断和调试 ====================
    
    def diagnose(self) -> Dict[str, Any]:
        """
        运行诊断
        
        Returns:
            诊断结果
        """
        diagnosis = {
            'fsdp_available': FSDP_AVAILABLE,
            'is_wrapped': self._is_wrapped,
            'is_distributed': dist.is_initialized(),
            'config': {
                'world_size': self.config.world_size,
                'rank': self.config.rank,
                'sharding_strategy': str(self.config.sharding_strategy),
                'mixed_precision': self.config.mixed_precision,
                'cpu_offload': self.config.cpu_offload,
            },
            'parameter_count': self.get_parameter_count(),
            'memory': self.get_memory_stats(),
            'step_count': self._step_count,
        }
        
        # 检查潜在问题
        issues = []
        
        if not FSDP_AVAILABLE:
            issues.append("FSDP not available - check PyTorch version")
        
        if not self._is_wrapped:
            issues.append("Model not wrapped - call wrap() first")
        
        if self._memory_monitor:
            stats = self._memory_monitor.get_memory_stats()
            if stats.get('utilization', 0) > 0.9:
                issues.append("High memory utilization (>90%)")
        
        diagnosis['issues'] = issues
        diagnosis['suggestions'] = self.get_memory_suggestions()
        
        return diagnosis
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diagnosis = self.diagnose()
        
        print("\n=== FSDP Wrapper Diagnosis ===")
        print(f"FSDP Available: {diagnosis['fsdp_available']}")
        print(f"Model Wrapped: {diagnosis['is_wrapped']}")
        print(f"Distributed: {diagnosis['is_distributed']}")
        
        print("\nConfiguration:")
        for key, value in diagnosis['config'].items():
            print(f"  {key}: {value}")
        
        print("\nParameters:")
        for key, value in diagnosis['parameter_count'].items():
            print(f"  {key}: {value:,}")
        
        print("\nMemory:")
        for key, value in diagnosis['memory'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        if diagnosis['issues']:
            print("\nIssues:")
            for issue in diagnosis['issues']:
                print(f"  ⚠ {issue}")
        
        if diagnosis['suggestions']:
            print("\nSuggestions:")
            for suggestion in diagnosis['suggestions']:
                print(f"  → {suggestion}")
    
    # ==================== 清理 ====================
    
    def cleanup(self) -> None:
        """清理资源"""
        if self._memory_monitor:
            self._memory_monitor.stop_monitoring()
        
        if self._context is not None:
            self._context.cleanup()
        
        self.clear_memory_cache()
        
        logger.info("FSDP wrapper cleaned up")


# ==================== 激活检查点 ====================

class FSDPActivationCheckpointing:
    """
    FSDP激活检查点管理器
    
    减少激活内存使用。
    """
    
    @staticmethod
    def apply(
        model: nn.Module,
        check_fn: Optional[Callable[[nn.Module], bool]] = None,
        checkpoint_impl: str = "reentrant"
    ) -> None:
        """
        应用激活检查点
        
        Args:
            model: FSDP模型
            check_fn: 检查函数，决定哪些层应用检查点
            checkpoint_impl: 检查点实现方式
        """
        if not FSDP_AVAILABLE:
            logger.warning("FSDP not available, skipping activation checkpointing")
            return
        
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
                CheckpointImpl,
                apply_activation_checkpointing
            )
            
            impl_map = {
                'reentrant': CheckpointImpl.REENTRANT,
                'no_reentrant': CheckpointImpl.NO_REENTRANT,
            }
            
            impl = impl_map.get(checkpoint_impl, CheckpointImpl.REENTRANT)
            
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=impl),
                check_fn=check_fn
            )
            
            logger.info(f"Activation checkpointing applied with {checkpoint_impl} implementation")
            
        except ImportError as e:
            logger.warning(f"Could not apply activation checkpointing: {e}")
    
    @staticmethod
    def create_transformer_check_fn(
        layer_cls: Type[nn.Module]
    ) -> Callable[[nn.Module], bool]:
        """
        创建Transformer层检查函数
        
        Args:
            layer_cls: Transformer层类
            
        Returns:
            检查函数
        """
        def check_fn(module: nn.Module) -> bool:
            return isinstance(module, layer_cls)
        return check_fn
    
    @staticmethod
    def create_interval_check_fn(
        layer_cls: Type[nn.Module],
        interval: int = 2
    ) -> Callable[[nn.Module], bool]:
        """
        创建间隔检查函数
        
        每隔interval层应用检查点。
        
        Args:
            layer_cls: 层类
            interval: 间隔
            
        Returns:
            检查函数
        """
        counter = [0]
        
        def check_fn(module: nn.Module) -> bool:
            if isinstance(module, layer_cls):
                counter[0] += 1
                return counter[0] % interval == 0
            return False
        
        return check_fn


# ==================== 便捷函数 ====================

def create_fsdp_model(
    model: nn.Module,
    config=None,
    transformer_layer_cls: Optional[List[Type[nn.Module]]] = None,
    apply_activation_checkpointing: bool = False,
    **kwargs
) -> Tuple[nn.Module, FSDPWrapper]:
    """
    创建FSDP模型
    
    Args:
        model: 原始模型
        config: FSDP配置
        transformer_layer_cls: Transformer层类列表
        apply_activation_checkpointing: 是否应用激活检查点
        **kwargs: 额外参数
        
    Returns:
        (FSDP包装的模型, FSDPWrapper实例)
    """
    from .parallel_modes import FSDPConfig
    config = config or FSDPConfig()
    
    # 合并额外参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    wrapper = FSDPWrapper(config)
    fsdp_model = wrapper.wrap(model, transformer_layer_cls=transformer_layer_cls)
    
    # 应用激活检查点
    if apply_activation_checkpointing and transformer_layer_cls:
        check_fn = FSDPActivationCheckpointing.create_transformer_check_fn(transformer_layer_cls[0])
        FSDPActivationCheckpointing.apply(fsdp_model, check_fn)
    
    return fsdp_model, wrapper


@contextmanager
def fsdp_context(config=None):
    """
    FSDP上下文管理器
    
    Args:
        config: FSDP配置
        
    Yields:
        FSDPContext实例
    """
    from .parallel_modes import FSDPConfig
    ctx = FSDPContext(config or FSDPConfig())
    ctx.initialize()
    try:
        yield ctx
    finally:
        ctx.cleanup()


def apply_fsdp_activation_checkpointing(
    model: nn.Module,
    check_fn: Optional[Callable] = None,
    checkpoint_impl: str = "reentrant"
) -> None:
    """
    应用FSDP激活检查点
    
    Args:
        model: FSDP模型
        check_fn: 检查函数
        checkpoint_impl: 检查点实现方式
    """
    FSDPActivationCheckpointing.apply(model, check_fn, checkpoint_impl)


def get_fsdp_memory_stats(device_id: int = 0) -> Dict[str, float]:
    """
    获取FSDP内存统计
    
    Args:
        device_id: GPU设备ID
        
    Returns:
        内存统计
    """
    monitor = FSDPMemoryMonitor(device_id)
    return monitor.get_memory_stats()


def estimate_fsdp_memory(
    model: nn.Module,
    world_size: int,
    sharding_strategy: str = "full_shard",
    dtype_bytes: int = 2
) -> Dict[str, float]:
    """
    估算FSDP内存使用
    
    Args:
        model: 模型
        world_size: 进程数
        sharding_strategy: 分片策略
        dtype_bytes: 数据类型字节数
        
    Returns:
        内存估算（GB）
    """
    num_params = sum(p.numel() for p in model.parameters())
    
    # 参数内存
    param_memory = num_params * dtype_bytes
    
    # 梯度内存
    grad_memory = num_params * dtype_bytes
    
    # 优化器状态（Adam: fp32, 2倍参数）
    optimizer_memory = num_params * 4 * 2
    
    # 根据策略计算分片
    if sharding_strategy == "full_shard":
        param_per_gpu = param_memory / world_size
        grad_per_gpu = grad_memory / world_size
        optimizer_per_gpu = optimizer_memory / world_size
    elif sharding_strategy == "shard_grad_op":
        param_per_gpu = param_memory
        grad_per_gpu = grad_memory / world_size
        optimizer_per_gpu = optimizer_memory / world_size
    else:  # no_shard
        param_per_gpu = param_memory
        grad_per_gpu = grad_memory
        optimizer_per_gpu = optimizer_memory
    
    total_per_gpu = param_per_gpu + grad_per_gpu + optimizer_per_gpu
    
    return {
        'params_gb': param_per_gpu / (1024**3),
        'grads_gb': grad_per_gpu / (1024**3),
        'optimizer_gb': optimizer_per_gpu / (1024**3),
        'total_per_gpu_gb': total_per_gpu / (1024**3),
        'total_model_gb': (param_memory + grad_memory + optimizer_memory) / (1024**3),
    }


def auto_configure_fsdp(
    model: nn.Module,
    num_gpus: int,
    gpu_memory_gb: float = 80.0,
    batch_size: int = 1,
    sequence_length: int = 2048
) -> Any:
    """
    自动配置FSDP
    
    根据模型和硬件自动选择最佳配置。
    
    Args:
        model: 模型
        num_gpus: GPU数量
        gpu_memory_gb: 单卡内存
        batch_size: 批次大小
        sequence_length: 序列长度
        
    Returns:
        推荐的FSDP配置 (FSDPConfig)
    """
    from .parallel_modes import FSDPConfig
    
    # 分析模型
    num_params = sum(p.numel() for p in model.parameters())
    model_memory_gb = num_params * 18 / (1024**3)  # 粗略估算
    
    # 估算激活内存
    activation_memory_gb = batch_size * sequence_length * 4096 * 4 / (1024**3)  # 粗略
    
    memory_per_gpu = (model_memory_gb + activation_memory_gb) / num_gpus
    
    # 选择分片策略
    strategy = FSDPShardingStrategy.from_memory_constraint(gpu_memory_gb, memory_per_gpu)
    
    config = FSDPConfig(
        world_size=num_gpus,
        sharding_strategy=strategy.value,
    )
    
    # 如果内存仍然紧张，启用CPU卸载
    if memory_per_gpu > gpu_memory_gb * 0.8:
        config.cpu_offload = True
        config.offload_params = memory_per_gpu > gpu_memory_gb
    
    # 大模型启用激活检查点
    if num_params > 1_000_000_000:  # >1B参数
        config.activation_checkpointing = True
    
    # 选择混合精度
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        config.param_dtype = 'bf16'
        config.reduce_dtype = 'bf16'
    else:
        config.param_dtype = 'fp16'
        config.reduce_dtype = 'fp32'
    
    logger.info(f"Auto-configured FSDP: strategy={strategy.value}, "
               f"cpu_offload={config.cpu_offload}, param_dtype={config.param_dtype}")
    
    return config

# -*- coding: utf-8 -*-
"""
DDP (Data Distributed Parallel) 包装器

提供PyTorch DistributedDataParallel的封装和管理，支持生产级功能，包括：
- 分布式环境初始化和管理
- 模型包装和同步
- 梯度同步控制
- 内存监控
- 性能分析
- 检查点管理
- 自动配置和诊断
"""

import os
import gc
import time
import json
import logging
import threading
from typing import Optional, List, Any, Dict, Union, Tuple, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .parallel_modes import DDPConfig, CommunicationBackend

logger = logging.getLogger(__name__)


# ==================== 枚举和配置类 ====================

class DDPSyncMode(Enum):
    """DDP同步模式"""
    SYNC = "sync"           # 每个step同步
    ASYNC = "async"         # 异步（梯度累积时）
    NO_SYNC = "no_sync"     # 禁用同步


class DDPReduceOp(Enum):
    """DDP归约操作"""
    SUM = "sum"
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    
    def to_dist_op(self) -> dist.ReduceOp:
        """转换为PyTorch分布式操作"""
        op_map = {
            self.SUM: dist.ReduceOp.SUM,
            self.AVG: dist.ReduceOp.AVG if hasattr(dist.ReduceOp, 'AVG') else dist.ReduceOp.SUM,
            self.MAX: dist.ReduceOp.MAX,
            self.MIN: dist.ReduceOp.MIN,
        }
        return op_map.get(self, dist.ReduceOp.SUM)


@dataclass
class DDPState:
    """
    DDP状态
    
    跟踪DDP的运行状态和统计信息。
    """
    # 初始化状态
    initialized: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"

    # 训练状态
    global_step: int = 0
    epoch: int = 0
    samples_seen: int = 0
    
    # 同步状态
    sync_enabled: bool = True
    accumulated_steps: int = 0
    
    # 性能统计
    forward_count: int = 0
    backward_count: int = 0
    sync_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'initialized': self.initialized,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'backend': self.backend,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'samples_seen': self.samples_seen,
            'sync_enabled': self.sync_enabled,
            'accumulated_steps': self.accumulated_steps,
            'forward_count': self.forward_count,
            'backward_count': self.backward_count,
            'sync_count': self.sync_count,
        }
    
    def reset_counters(self) -> None:
        """重置计数器"""
        self.forward_count = 0
        self.backward_count = 0
        self.sync_count = 0


# ==================== 内存监控 ====================

class DDPMemoryMonitor:
    """
    DDP内存监控器
    
    监控分布式训练过程中的内存使用。
    """
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._history: List[Dict[str, Any]] = []
        self._peak_memory: float = 0.0
        self._enabled = True
    
    def enable(self) -> None:
        """启用监控"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用监控"""
        self._enabled = False
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取当前内存统计"""
        if not torch.cuda.is_available():
            return {'allocated_gb': 0, 'reserved_gb': 0, 'free_gb': 0, 'total_gb': 0}
        
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
        """获取峰值内存（GB）"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device_id) / (1024**3)
        return 0.0
    
    def reset_peak_memory(self) -> None:
        """重置峰值内存统计"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
    
    def record(self, tag: str = "") -> Dict[str, Any]:
        """记录内存状态"""
        if not self._enabled:
            return {}
        
        stats = self.get_memory_stats()
        stats['tag'] = tag
        stats['timestamp'] = time.time()
        self._history.append(stats)
        
        peak = self.get_peak_memory()
        if peak > self._peak_memory:
            self._peak_memory = peak
        
        return stats
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取历史记录"""
        return self._history.copy()
    
    def clear_history(self) -> None:
        """清除历史"""
        self._history.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        if not self._history:
            return {'message': 'No records'}
        
        allocated_values = [h['allocated_gb'] for h in self._history]
        
        return {
            'peak_memory_gb': self._peak_memory,
            'avg_allocated_gb': sum(allocated_values) / len(allocated_values),
            'max_allocated_gb': max(allocated_values),
            'min_allocated_gb': min(allocated_values),
            'num_records': len(self._history),
        }
    
    @contextmanager
    def track(self, tag: str):
        """内存追踪上下文"""
        self.record(f"{tag}_start")
        try:
            yield
        finally:
            self.record(f"{tag}_end")


# ==================== 性能分析器 ====================

class DDPProfiler:
    """
    DDP性能分析器
    
    分析分布式训练的性能瓶颈。
    """
    
    def __init__(self):
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._enabled = False
        self._step_count = 0
        self._step_start_time: Optional[float] = None
        self._step_times: List[float] = []
    
    def enable(self) -> None:
        """启用分析"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用分析"""
        self._enabled = False
    
    def start_step(self) -> None:
        """开始一个训练步骤"""
        if self._enabled:
            self._step_count += 1
            self._step_start_time = time.perf_counter()
    
    def end_step(self) -> None:
        """结束一个训练步骤"""
        if self._enabled and self._step_start_time is not None:
            elapsed = time.perf_counter() - self._step_start_time
            self._step_times.append(elapsed)
            self._step_start_time = None
    
    @contextmanager
    def profile_region(self, name: str):
        """分析代码区域"""
        if not self._enabled:
            yield
            return
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            self._timings[name].append(elapsed)
    
    def record_timing(self, name: str, duration: float) -> None:
        """手动记录时间"""
        if self._enabled:
            self._timings[name].append(duration)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """获取特定区域的统计"""
        timings = self._timings.get(name, [])
        if not timings:
            return {'count': 0, 'total_ms': 0, 'avg_ms': 0, 'min_ms': 0, 'max_ms': 0}
        
        return {
            'count': len(timings),
            'total_ms': sum(timings) * 1000,
            'avg_ms': sum(timings) / len(timings) * 1000,
            'min_ms': min(timings) * 1000,
            'max_ms': max(timings) * 1000,
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有统计"""
        return {name: self.get_stats(name) for name in self._timings}
    
    def get_step_stats(self) -> Dict[str, float]:
        """获取步骤统计"""
        if not self._step_times:
            return {'count': 0, 'avg_ms': 0, 'throughput': 0}
        
        avg_time = sum(self._step_times) / len(self._step_times)
        
        return {
            'count': len(self._step_times),
            'avg_ms': avg_time * 1000,
            'min_ms': min(self._step_times) * 1000,
            'max_ms': max(self._step_times) * 1000,
            'throughput': 1.0 / avg_time if avg_time > 0 else 0,
        }
    
    def reset(self) -> None:
        """重置统计"""
        self._timings.clear()
        self._step_count = 0
        self._step_times.clear()
        self._step_start_time = None
    
    def print_summary(self) -> None:
        """打印摘要"""
        print(f"\n=== DDP Profiler Summary ({self._step_count} steps) ===")
        
        step_stats = self.get_step_stats()
        if step_stats['count'] > 0:
            print(f"\nStep Statistics:")
            print(f"  Average: {step_stats['avg_ms']:.2f}ms")
            print(f"  Min: {step_stats['min_ms']:.2f}ms")
            print(f"  Max: {step_stats['max_ms']:.2f}ms")
            print(f"  Throughput: {step_stats['throughput']:.2f} steps/sec")
        
        stats = self.get_all_stats()
        if not stats:
            print("No region profiling data")
            return
        
        total_time = sum(s.get('total_ms', 0) for s in stats.values())
        
        print(f"\nRegion Breakdown:")
        print(f"{'Region':<25} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'%':>8}")
        print("-" * 65)
        
        sorted_stats = sorted(stats.items(), key=lambda x: x[1].get('total_ms', 0), reverse=True)
        
        for name, stat in sorted_stats:
            pct = stat['total_ms'] / total_time * 100 if total_time > 0 else 0
            print(f"{name:<25} {stat['count']:>8} {stat['total_ms']:>12.2f} {stat['avg_ms']:>10.2f} {pct:>7.1f}%")


# ==================== 通信分析器 ====================

class DDPCommunicationAnalyzer:
    """
    DDP通信分析器
    
    分析分布式通信开销。
    """
    
    def __init__(self):
        self._all_reduce_times: List[float] = []
        self._broadcast_times: List[float] = []
        self._barrier_times: List[float] = []
        self._data_transferred: int = 0  # bytes
        self._enabled = False
    
    def enable(self) -> None:
        """启用分析"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用分析"""
        self._enabled = False
    
    def record_all_reduce(self, duration: float, data_bytes: int = 0) -> None:
        """记录all_reduce操作"""
        if self._enabled:
            self._all_reduce_times.append(duration)
            self._data_transferred += data_bytes
    
    def record_broadcast(self, duration: float, data_bytes: int = 0) -> None:
        """记录broadcast操作"""
        if self._enabled:
            self._broadcast_times.append(duration)
            self._data_transferred += data_bytes
    
    def record_barrier(self, duration: float) -> None:
        """记录barrier操作"""
        if self._enabled:
            self._barrier_times.append(duration)
    
    @contextmanager
    def timed_all_reduce(self, tensor: Optional[torch.Tensor] = None):
        """计时all_reduce上下文"""
        if not self._enabled:
            yield
            return
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        data_bytes = tensor.numel() * tensor.element_size() if tensor is not None else 0
        
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            duration = time.perf_counter() - start
            self.record_all_reduce(duration, data_bytes)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取通信摘要"""
        def calc_stats(times: List[float]) -> Dict[str, float]:
            if not times:
                return {'count': 0, 'total_ms': 0, 'avg_ms': 0}
            return {
                'count': len(times),
                'total_ms': sum(times) * 1000,
                'avg_ms': sum(times) / len(times) * 1000,
            }
        
        return {
            'all_reduce': calc_stats(self._all_reduce_times),
            'broadcast': calc_stats(self._broadcast_times),
            'barrier': calc_stats(self._barrier_times),
            'total_data_transferred_mb': self._data_transferred / (1024**2),
        }
    
    def reset(self) -> None:
        """重置统计"""
        self._all_reduce_times.clear()
        self._broadcast_times.clear()
        self._barrier_times.clear()
        self._data_transferred = 0
    
    def print_summary(self) -> None:
        """打印通信摘要"""
        summary = self.get_summary()
        
        print("\n=== DDP Communication Summary ===")
        print(f"Data Transferred: {summary['total_data_transferred_mb']:.2f} MB")
        
        for op_name, stats in [('All-Reduce', summary['all_reduce']),
                               ('Broadcast', summary['broadcast']),
                               ('Barrier', summary['barrier'])]:
            if stats['count'] > 0:
                print(f"\n{op_name}:")
                print(f"  Count: {stats['count']}")
                print(f"  Total: {stats['total_ms']:.2f}ms")
                print(f"  Average: {stats['avg_ms']:.2f}ms")


# ==================== DDP上下文 ====================

class DDPContext:
    """
    DDP上下文管理器
    
    管理DDP的初始化、运行和清理。
    """
    
    def __init__(self, config: DDPConfig):
        self.config = config
        self._initialized = False
        self._original_model = None
        self._ddp_model = None
        self._state = DDPState()
    
    def __enter__(self):
        """进入上下文"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        self.cleanup()
        return False
    
    def initialize(self) -> None:
        """初始化DDP环境"""
        if self._initialized:
            return
        
        if not dist.is_initialized():
            # 设置环境变量
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port
            
            # 获取backend
            backend = self.config.backend.value if isinstance(self.config.backend, CommunicationBackend) else self.config.backend
            
            # 初始化进程组
            dist.init_process_group(
                backend=backend,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
        
        # 设置设备
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
        
        # 更新状态
        self._state.initialized = True
        self._state.world_size = self.config.world_size
        self._state.rank = self.config.rank
        self._state.local_rank = self.config.local_rank
        self._state.backend = self.config.backend.value if isinstance(self.config.backend, CommunicationBackend) else self.config.backend
        
        self._initialized = True
        logger.info(f"DDP initialized: rank={self.config.rank}/{self.config.world_size}, backend={self._state.backend}")
    
    def cleanup(self) -> None:
        """清理DDP资源"""
        if dist.is_initialized():
            dist.destroy_process_group()
        
        self._state.initialized = False
        self._initialized = False
        logger.info("DDP cleaned up")
    
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
        """获取当前设备"""
        if torch.cuda.is_available():
            return torch.device(f'cuda:{self.config.local_rank}')
        return torch.device('cpu')

    @property
    def state(self) -> DDPState:
        """获取状态"""
        return self._state
    
    def barrier(self) -> None:
        """同步屏障"""
        if dist.is_initialized():
            dist.barrier()
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """广播张量"""
        if dist.is_initialized():
            dist.broadcast(tensor, src=src)
        return tensor
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: DDPReduceOp = DDPReduceOp.SUM
    ) -> torch.Tensor:
        """全归约操作"""
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op.to_dist_op())
        return tensor


# ==================== DDP包装器 ====================

class DDPWrapper:
    """
    DDP模型包装器
    
    封装PyTorch DDP的创建和管理，提供生产级功能。
    """
    
    def __init__(self, config: Optional[DDPConfig] = None):
        self.config = config or DDPConfig()
        
        # 模型
        self._model: Optional[nn.Module] = None
        self._ddp_model: Optional[DistributedDataParallel] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._lr_scheduler: Optional[Any] = None
        
        # 上下文
        self._context: Optional[DDPContext] = None
    
        # 组件
        self._memory_monitor = DDPMemoryMonitor(self.config.local_rank)
        self._profiler = DDPProfiler()
        self._comm_analyzer = DDPCommunicationAnalyzer()
        self._state = DDPState()
        
        # 同步控制
        self._sync_enabled = True
        self._gradient_accumulation_steps = 1
        self._accumulated_steps = 0
    
    # ==================== 包装和初始化 ====================
    
    def wrap(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None
    ) -> DistributedDataParallel:
        """
        将模型包装为DDP模型
        
        Args:
            model: 原始模型
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            
        Returns:
            DDP包装后的模型
        """
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        
        # 记录包装前内存
        self._memory_monitor.record("pre_wrap")
        
        # 确保环境已初始化
        if not dist.is_initialized():
            self._context = DDPContext(self.config)
            self._context.initialize()
        
        # 移动模型到正确的设备
        device = self._get_device()
        model = model.to(device)
        
        # 分析模型
        model_info = self._analyze_model(model)
        logger.info(f"Wrapping model with DDP: {model_info['num_params']:,} params")
        
        # 创建DDP模型
        self._ddp_model = DistributedDataParallel(
            model,
            device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
            output_device=self.config.local_rank if torch.cuda.is_available() else None,
            broadcast_buffers=self.config.broadcast_buffers,
            find_unused_parameters=self.config.find_unused_parameters,
            bucket_cap_mb=self.config.bucket_cap_mb,
            gradient_as_bucket_view=self.config.gradient_as_bucket_view,
            static_graph=self.config.static_graph
        )
        
        # 更新状态
        self._state.initialized = True
        self._state.world_size = self.config.world_size
        self._state.rank = self.config.rank
        self._state.local_rank = self.config.local_rank
        
        # 记录包装后内存
        self._memory_monitor.record("post_wrap")
        
        logger.info(f"Model wrapped with DDP: device={device}, rank={self.config.rank}/{self.config.world_size}")
        
        return self._ddp_model
    
    def _get_device(self) -> torch.device:
        """获取当前设备"""
        if torch.cuda.is_available():
            return torch.device(f'cuda:{self.config.local_rank}')
        return torch.device('cpu')
    
    def _analyze_model(self, model: nn.Module) -> Dict[str, Any]:
        """分析模型"""
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        num_buffers = sum(b.numel() for b in model.buffers())
        
        return {
            'num_params': num_params,
            'num_trainable': num_trainable,
            'num_frozen': num_params - num_trainable,
            'num_buffers': num_buffers,
            'param_memory_mb': num_params * 4 / (1024**2),  # 假设fp32
        }
    
    def unwrap(self) -> nn.Module:
        """
        解包DDP模型
        
        Returns:
            原始模型
        """
        if self._ddp_model is not None:
            return self._ddp_model.module
        return self._model
    
    @property
    def module(self) -> nn.Module:
        """获取内部模型"""
        return self.unwrap()
    
    @property
    def ddp_model(self) -> Optional[DistributedDataParallel]:
        """获取DDP模型"""
        return self._ddp_model
    
    @property
    def optimizer(self) -> Optional[torch.optim.Optimizer]:
        """获取优化器"""
        return self._optimizer
    
    @property
    def is_wrapped(self) -> bool:
        """是否已包装"""
        return self._ddp_model is not None
    
    @property
    def is_main_process(self) -> bool:
        """是否是主进程"""
        return self.config.rank == 0
    
    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """设置优化器"""
        self._optimizer = optimizer
    
    def set_lr_scheduler(self, scheduler: Any) -> None:
        """设置学习率调度器"""
        self._lr_scheduler = scheduler
    
    # ==================== 训练步骤 ====================
    
    def forward(self, *args, **kwargs) -> Any:
        """前向传播"""
        with self._profiler.profile_region("forward"):
            if self._ddp_model is not None:
                result = self._ddp_model(*args, **kwargs)
            else:
                result = self._model(*args, **kwargs)
        
        self._state.forward_count += 1
        return result
    
    def backward(self, loss: torch.Tensor) -> None:
        """反向传播"""
        with self._profiler.profile_region("backward"):
            loss.backward()
        
        self._state.backward_count += 1
    
    def step(self) -> None:
        """优化器步进"""
        with self._profiler.profile_region("optimizer_step"):
            if self._optimizer is not None:
                self._optimizer.step()
        
        self._state.global_step += 1
        self._accumulated_steps = 0
    
    def zero_grad(self, set_to_none: bool = True) -> None:
        """清零梯度"""
        if self._optimizer is not None:
            self._optimizer.zero_grad(set_to_none=set_to_none)
        elif self._ddp_model is not None:
            self._ddp_model.zero_grad(set_to_none=set_to_none)
        elif self._model is not None:
            self._model.zero_grad(set_to_none=set_to_none)
    
    def train_step(
        self,
        batch: Any,
        loss_fn: Callable,
        accumulate: bool = False
    ) -> torch.Tensor:
        """
        完整的训练步骤
        
        Args:
            batch: 输入批次
            loss_fn: 损失函数
            accumulate: 是否进行梯度累积
            
        Returns:
            损失值
        """
        self._profiler.start_step()
        
        # 前向传播
        output = self.forward(batch)
        
        # 计算损失
        loss = loss_fn(output)
        
        # 缩放损失（梯度累积）
        if self._gradient_accumulation_steps > 1:
            loss = loss / self._gradient_accumulation_steps
        
        # 反向传播（可能禁用同步）
        if accumulate and self._accumulated_steps < self._gradient_accumulation_steps - 1:
            with self.no_sync():
                self.backward(loss)
        else:
            self.backward(loss)
        
        self._accumulated_steps += 1
        
        # 优化器步进
        if not accumulate or self._accumulated_steps >= self._gradient_accumulation_steps:
            # 梯度裁剪
            grad_norm = self.clip_grad_norm()
            
            self.step()
            self.zero_grad()
        
        self._profiler.end_step()
        
        return loss
    
    # ==================== 梯度同步控制 ====================
    
    def sync_gradients(self) -> None:
        """同步梯度"""
        if self._ddp_model is not None:
            # DDP自动同步，但可以手动触发
            self._state.sync_count += 1
    
    @contextmanager
    def no_sync(self):
        """
        禁用梯度同步上下文
        
        用于梯度累积，避免每个微批次都同步梯度。
        """
        if self._ddp_model is not None:
            with self._ddp_model.no_sync():
                yield
        else:
            yield
    
    def enable_sync(self) -> None:
        """启用梯度同步"""
        self._sync_enabled = True
        self._state.sync_enabled = True
    
    def disable_sync(self) -> None:
        """禁用梯度同步"""
        self._sync_enabled = False
        self._state.sync_enabled = False
    
    def set_gradient_accumulation_steps(self, steps: int) -> None:
        """设置梯度累积步数"""
        self._gradient_accumulation_steps = max(1, steps)
    
    # ==================== 梯度控制 ====================
    
    def clip_grad_norm(self, max_norm: float = 1.0) -> float:
        """
        梯度裁剪
        
        Args:
            max_norm: 最大梯度范数
            
        Returns:
            裁剪前的梯度范数
        """
        model = self.unwrap()
        if model is not None:
            return torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm
            ).item()
        return 0.0
    
    def clip_grad_value(self, clip_value: float = 1.0) -> None:
        """
        梯度值裁剪
        
        Args:
            clip_value: 裁剪值
        """
        model = self.unwrap()
        if model is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
    
    def get_grad_norm(self) -> float:
        """获取当前梯度范数"""
        model = self.unwrap()
        if model is None:
            return 0.0
        
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        
        return total_norm ** 0.5
    
    def scale_gradients(self, scale: float) -> None:
        """缩放梯度"""
        model = self.unwrap()
        if model is not None:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(scale)
    
    # ==================== 学习率管理 ====================
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        if self._optimizer is not None:
            for param_group in self._optimizer.param_groups:
                return param_group['lr']
        return 0.0
    
    def set_lr(self, lr: float) -> None:
        """设置学习率"""
        if self._optimizer is not None:
            for param_group in self._optimizer.param_groups:
                param_group['lr'] = lr
    
    def step_scheduler(self, metrics: Optional[float] = None) -> None:
        """步进学习率调度器"""
        if self._lr_scheduler is not None:
            if metrics is not None:
                try:
                    self._lr_scheduler.step(metrics)
                except TypeError:
                    self._lr_scheduler.step()
            else:
                self._lr_scheduler.step()
    
    # ==================== 检查点管理 ====================
    
    def save_checkpoint(
        self,
        path: str,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        only_main_process: bool = True,
        **kwargs
    ) -> Optional[str]:
        """
        保存检查点
        
        Args:
            path: 保存路径
            save_optimizer: 是否保存优化器
            save_scheduler: 是否保存调度器
            only_main_process: 是否只在主进程保存
            **kwargs: 额外数据
            
        Returns:
            保存路径（主进程），或None（非主进程）
        """
        if only_main_process and not self.is_main_process:
            # 等待主进程保存完成
            if dist.is_initialized():
                dist.barrier()
            return None
        
        # 准备检查点数据
        checkpoint = {
            'model_state_dict': self.unwrap().state_dict(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
            'state': self._state.to_dict(),
                **kwargs
        }
        
        if save_optimizer and self._optimizer is not None:
            checkpoint['optimizer_state_dict'] = self._optimizer.state_dict()
        
        if save_scheduler and self._lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self._lr_scheduler.state_dict()
        
        # 保存
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
        # 同步
        if dist.is_initialized():
            dist.barrier()
        
        return path
    
    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            path: 检查点路径
            load_optimizer: 是否加载优化器
            load_scheduler: 是否加载调度器
            strict: 是否严格匹配
            
        Returns:
            检查点数据
        """
        # 映射到正确的设备
        map_location = {'cuda:0': f'cuda:{self.config.local_rank}'} if torch.cuda.is_available() else 'cpu'
        # weights_only=False 用于支持包含非张量数据的检查点
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        
        # 加载模型
        self.unwrap().load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # 加载优化器
        if load_optimizer and self._optimizer is not None and 'optimizer_state_dict' in checkpoint:
            self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器
        if load_scheduler and self._lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self._lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复状态
        if 'state' in checkpoint:
            state_dict = checkpoint['state']
            self._state.global_step = state_dict.get('global_step', 0)
            self._state.epoch = state_dict.get('epoch', 0)
            self._state.samples_seen = state_dict.get('samples_seen', 0)
        
        # 同步
        if dist.is_initialized():
            dist.barrier()
        
        logger.info(f"Checkpoint loaded: {path}")
        return checkpoint
    
    # ==================== 通信操作 ====================
    
    def barrier(self) -> None:
        """同步屏障"""
        if dist.is_initialized():
            start = time.perf_counter()
            dist.barrier()
            duration = time.perf_counter() - start
            self._comm_analyzer.record_barrier(duration)
    
    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0
    ) -> torch.Tensor:
        """
        广播张量
        
        Args:
            tensor: 要广播的张量
            src: 源rank
            
        Returns:
            广播后的张量
        """
        if dist.is_initialized():
            start = time.perf_counter()
            dist.broadcast(tensor, src=src)
            duration = time.perf_counter() - start
            self._comm_analyzer.record_broadcast(duration, tensor.numel() * tensor.element_size())
        return tensor
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: DDPReduceOp = DDPReduceOp.SUM,
        normalize: bool = False
    ) -> torch.Tensor:
        """
        全归约操作
        
        Args:
            tensor: 输入张量
            op: 归约操作
            normalize: 是否归一化
            
        Returns:
            归约后的张量
        """
        if dist.is_initialized():
            with self._comm_analyzer.timed_all_reduce(tensor):
                dist.all_reduce(tensor, op=op.to_dist_op())
                if normalize and op == DDPReduceOp.SUM:
                    tensor.div_(self.config.world_size)
        return tensor
    
    def all_gather(
        self,
        tensor: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        全收集操作
        
        Args:
            tensor: 输入张量
            
        Returns:
            所有rank的张量列表
        """
        if not dist.is_initialized():
            return [tensor]
        
        tensor_list = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
        dist.all_gather(tensor_list, tensor)
        return tensor_list
    
    def reduce_scatter(
        self,
        input_list: List[torch.Tensor],
        output: torch.Tensor,
        op: DDPReduceOp = DDPReduceOp.SUM
    ) -> torch.Tensor:
        """
        归约分散操作
        
        Args:
            input_list: 输入张量列表
            output: 输出张量
            op: 归约操作
            
        Returns:
            归约分散后的张量
        """
        if dist.is_initialized():
            dist.reduce_scatter(output, input_list, op=op.to_dist_op())
        return output
    
    # ==================== 内存管理 ====================
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计"""
        return self._memory_monitor.get_memory_stats()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存摘要"""
        return self._memory_monitor.get_summary()
    
    def clear_memory_cache(self) -> None:
        """清理内存缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self._memory_monitor.record("cache_cleared")
    
    @contextmanager
    def track_memory(self, tag: str):
        """内存追踪上下文"""
        with self._memory_monitor.track(tag):
            yield
    
    # ==================== 性能分析 ====================
    
    def enable_profiling(self) -> None:
        """启用性能分析"""
        self._profiler.enable()
        self._comm_analyzer.enable()
    
    def disable_profiling(self) -> None:
        """禁用性能分析"""
        self._profiler.disable()
        self._comm_analyzer.disable()
    
    def get_profiling_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'timings': self._profiler.get_all_stats(),
            'step_stats': self._profiler.get_step_stats(),
            'communication': self._comm_analyzer.get_summary(),
        }
    
    def print_profiling_summary(self) -> None:
        """打印性能摘要"""
        self._profiler.print_summary()
        self._comm_analyzer.print_summary()
    
    def reset_profiling(self) -> None:
        """重置性能统计"""
        self._profiler.reset()
        self._comm_analyzer.reset()
    
    # ==================== 状态管理 ====================
    
    def get_state(self) -> Dict[str, Any]:
        """获取状态"""
        return self._state.to_dict()
    
    def set_epoch(self, epoch: int) -> None:
        """设置当前epoch"""
        self._state.epoch = epoch
    
    def increment_samples(self, count: int) -> None:
        """增加处理的样本数"""
        self._state.samples_seen += count
    
    def get_throughput(self, batch_size: int) -> float:
        """获取吞吐量"""
        step_stats = self._profiler.get_step_stats()
        if step_stats['count'] == 0:
            return 0.0
        
        # 每秒处理的样本数
        samples_per_step = batch_size * self.config.world_size
        return samples_per_step * step_stats['throughput']
    
    # ==================== 诊断 ====================
    
    def diagnose(self) -> Dict[str, Any]:
        """
        运行诊断
        
        Returns:
            诊断结果
        """
        diagnosis = {
            'distributed_initialized': dist.is_initialized(),
            'is_wrapped': self.is_wrapped,
            'is_main_process': self.is_main_process,
            'config': {
                'world_size': self.config.world_size,
                'rank': self.config.rank,
                'local_rank': self.config.local_rank,
                'backend': self.config.backend.value if isinstance(self.config.backend, CommunicationBackend) else self.config.backend,
                'find_unused_parameters': self.config.find_unused_parameters,
                'broadcast_buffers': self.config.broadcast_buffers,
                'bucket_cap_mb': self.config.bucket_cap_mb,
                'static_graph': self.config.static_graph,
            },
            'state': self.get_state(),
            'memory': self.get_memory_stats(),
        }
        
        # 模型信息
        if self._model is not None:
            diagnosis['model_info'] = self._analyze_model(self._model)
        
        # 检查问题
        issues = []
        
        if not dist.is_initialized():
            issues.append("Distributed not initialized")
        
        if not self.is_wrapped:
            issues.append("Model not wrapped - call wrap() first")
        
        memory_stats = self.get_memory_stats()
        if memory_stats.get('utilization', 0) > 0.9:
            issues.append("High memory utilization (>90%)")
        
        if self.config.find_unused_parameters:
            issues.append("find_unused_parameters=True may slow down training")
        
        diagnosis['issues'] = issues
        
        # 建议
        suggestions = []
        
        if not self.config.gradient_as_bucket_view:
            suggestions.append("Consider enabling gradient_as_bucket_view for better memory efficiency")
        
        if not self.config.static_graph and self._state.forward_count > 100:
            suggestions.append("Consider enabling static_graph if model structure is fixed")
        
        if memory_stats.get('utilization', 0) > 0.7:
            suggestions.append("Consider reducing batch size or using gradient checkpointing")
        
        diagnosis['suggestions'] = suggestions
        
        return diagnosis
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diagnosis = self.diagnose()
        
        print("\n=== DDP Wrapper Diagnosis ===")
        print(f"Distributed Initialized: {diagnosis['distributed_initialized']}")
        print(f"Model Wrapped: {diagnosis['is_wrapped']}")
        print(f"Main Process: {diagnosis['is_main_process']}")
        
        print("\nConfiguration:")
        for key, value in diagnosis['config'].items():
            print(f"  {key}: {value}")
        
        print("\nState:")
        for key, value in diagnosis['state'].items():
            print(f"  {key}: {value}")
        
        print("\nMemory:")
        for key, value in diagnosis['memory'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        if 'model_info' in diagnosis:
            print("\nModel Info:")
            for key, value in diagnosis['model_info'].items():
                if isinstance(value, int) and value > 1000:
                    print(f"  {key}: {value:,}")
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
        if self._context is not None:
            self._context.cleanup()
        
        self._ddp_model = None
        self._model = None
        self._optimizer = None
        self._lr_scheduler = None
        
        self.clear_memory_cache()
        logger.info("DDP wrapper cleaned up")


# ==================== 便捷函数 ====================

def create_ddp_model(
    model: nn.Module,
    config: Optional[DDPConfig] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    **kwargs
) -> Tuple[DDPWrapper, DistributedDataParallel]:
    """
    创建DDP模型
    
    Args:
        model: 原始模型
        config: DDP配置
        optimizer: 优化器
        **kwargs: 额外的DDP参数
        
    Returns:
        (DDPWrapper实例, DDP模型)
    """
    config = config or DDPConfig()
    
    # 合并额外参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    wrapper = DDPWrapper(config)
    ddp_model = wrapper.wrap(model, optimizer=optimizer)
    
    return wrapper, ddp_model


def ddp_reduce_gradients(
    model: nn.Module,
    world_size: int,
    normalize: bool = True
) -> None:
    """
    手动同步梯度（通常DDP自动完成）
    
    Args:
        model: 模型
        world_size: 进程总数
        normalize: 是否归一化
    """
    if not dist.is_initialized():
        return
    
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            if normalize:
                param.grad.div_(world_size)


@contextmanager
def ddp_context(config: Optional[DDPConfig] = None):
    """
    DDP上下文管理器
    
    用法:
    ```python
    with ddp_context(config) as ctx:
        model = ctx.wrap(model)
        # 训练...
    ```
    """
    ctx = DDPContext(config or DDPConfig())
    ctx.initialize()
    try:
        yield ctx
    finally:
        ctx.cleanup()


def auto_configure_ddp(
    model: nn.Module,
    num_gpus: int = 1,
    find_unused: Optional[bool] = None
) -> DDPConfig:
    """
    自动配置DDP
    
    Args:
        model: 模型
        num_gpus: GPU数量
        find_unused: 是否查找未使用参数
        
    Returns:
        推荐的DDP配置
    """
    config = DDPConfig(
        world_size=num_gpus,
        rank=0,
        local_rank=0,
    )
    
    # 分析模型
    num_params = sum(p.numel() for p in model.parameters())
    
    # 自动设置bucket大小
    if num_params > 100_000_000:  # > 100M params
        config.bucket_cap_mb = 50
    else:
        config.bucket_cap_mb = 25
    
    # 自动设置find_unused_parameters
    if find_unused is None:
        # 检查是否有未使用的参数风险
        has_conditional_forward = False
        for name, module in model.named_modules():
            if hasattr(module, 'forward') and 'if' in str(type(module)):
                has_conditional_forward = True
                break
        config.find_unused_parameters = has_conditional_forward
    else:
        config.find_unused_parameters = find_unused
    
    # 启用优化
    config.gradient_as_bucket_view = True
    
    return config


def estimate_ddp_memory(
    model: nn.Module,
    batch_size: int,
    dtype_bytes: int = 4
) -> Dict[str, float]:
    """
    估算DDP内存使用
    
    Args:
        model: 模型
        batch_size: 批量大小
        dtype_bytes: 数据类型字节数
        
    Returns:
        内存估算（GB）
    """
    num_params = sum(p.numel() for p in model.parameters())
    num_buffers = sum(b.numel() for b in model.buffers())
    
    # 参数内存
    param_memory = num_params * dtype_bytes
    
    # 梯度内存
    grad_memory = num_params * dtype_bytes
    
    # 缓冲区内存
    buffer_memory = num_buffers * dtype_bytes
    
    # DDP桶内存（通常是梯度的副本）
    bucket_memory = param_memory
    
    total_memory = param_memory + grad_memory + buffer_memory + bucket_memory
    
    return {
        'params_gb': param_memory / (1024**3),
        'grads_gb': grad_memory / (1024**3),
        'buffers_gb': buffer_memory / (1024**3),
        'buckets_gb': bucket_memory / (1024**3),
        'total_model_gb': total_memory / (1024**3),
        'num_params': num_params,
    }


def print_ddp_info() -> None:
    """打印DDP环境信息"""
    print("\n=== DDP Environment Info ===")
    
    if dist.is_initialized():
        print(f"Initialized: True")
        print(f"Backend: {dist.get_backend()}")
        print(f"World Size: {dist.get_world_size()}")
        print(f"Rank: {dist.get_rank()}")
    else:
        print(f"Initialized: False")
    
    if torch.cuda.is_available():
        print(f"\nCUDA Available: True")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name()}")
    else:
        print(f"\nCUDA Available: False")


def compare_ddp_configs(
    configs: List[DDPConfig],
    names: Optional[List[str]] = None
) -> None:
    """
    比较DDP配置
    
    Args:
        configs: 配置列表
        names: 配置名称
    """
    if names is None:
        names = [f"Config_{i}" for i in range(len(configs))]
    
    print("\n=== DDP Configuration Comparison ===")
    
    fields = [
        'world_size', 'bucket_cap_mb', 'find_unused_parameters',
        'broadcast_buffers', 'gradient_as_bucket_view', 'static_graph'
    ]
    
    # 打印表头
    header = f"{'Field':<30}"
    for name in names:
        header += f" {name:>15}"
    print(header)
    print("-" * (30 + 16 * len(names)))
    
    # 打印各字段
    for field_name in fields:
        row = f"{field_name:<30}"
        for config in configs:
            value = getattr(config, field_name, 'N/A')
            row += f" {str(value):>15}"
        print(row)


# ==================== 工具函数 ====================

def get_ddp_rank() -> int:
    """获取当前进程rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_ddp_world_size() -> int:
    """获取总进程数"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_ddp_local_rank() -> int:
    """获取本地rank"""
    return int(os.environ.get('LOCAL_RANK', 0))


def is_ddp_main_process() -> bool:
    """是否是主进程"""
    return get_ddp_rank() == 0


def ddp_barrier() -> None:
    """同步屏障"""
    if dist.is_initialized():
        dist.barrier()


def ddp_all_reduce(
    tensor: torch.Tensor,
    op: str = 'sum',
    normalize: bool = False
) -> torch.Tensor:
    """
    全归约操作
    
    Args:
        tensor: 输入张量
        op: 操作类型 ('sum', 'avg', 'max', 'min')
        normalize: 是否归一化
        
    Returns:
        归约后的张量
    """
    if not dist.is_initialized():
        return tensor
    
    op_map = {
        'sum': dist.ReduceOp.SUM,
        'avg': dist.ReduceOp.AVG if hasattr(dist.ReduceOp, 'AVG') else dist.ReduceOp.SUM,
        'max': dist.ReduceOp.MAX,
        'min': dist.ReduceOp.MIN,
    }
    
    dist_op = op_map.get(op, dist.ReduceOp.SUM)
    dist.all_reduce(tensor, op=dist_op)
    
    if normalize and op == 'sum':
        tensor.div_(get_ddp_world_size())
    
    return tensor


def ddp_broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    广播张量
    
    Args:
        tensor: 要广播的张量
        src: 源rank
        
    Returns:
        广播后的张量
    """
    if dist.is_initialized():
        dist.broadcast(tensor, src=src)
    return tensor


def ddp_all_gather_object(obj: Any) -> List[Any]:
    """
    收集所有进程的对象
    
    Args:
        obj: 要收集的对象
        
    Returns:
        所有进程的对象列表
    """
    if not dist.is_initialized():
        return [obj]
    
    output = [None for _ in range(get_ddp_world_size())]
    dist.all_gather_object(output, obj)
    return output


def average_gradients(model: nn.Module) -> None:
    """平均所有进程的梯度"""
    world_size = get_ddp_world_size()
    if world_size == 1:
        return
    
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size

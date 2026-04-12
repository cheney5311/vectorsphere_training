# -*- coding: utf-8 -*-
"""
Pipeline Parallel 包装器

提供流水线并行的封装和管理，支持GPipe、1F1B和Interleaved调度，包括：
- 智能模型分割和负载均衡
- 多种调度策略（GPipe、1F1B、Interleaved）
- 激活通信管理（send/recv）
- 内存优化和激活检查点
- 性能分析和气泡优化
- 检查点管理
"""

import os
import gc
import time
import logging
import threading
import queue
from typing import Optional, List, Any, Dict, Union, Type, Tuple, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)


# ==================== 枚举和配置类 ====================

class PipelineSchedule(Enum):
    """
    流水线调度策略
    
    不同策略在内存使用和气泡时间上有不同权衡。
    """
    GPIPE = "gpipe"              # GPipe调度：简单，气泡较大
    ONE_F_ONE_B = "1f1b"         # 1F1B调度：减少内存
    INTERLEAVED = "interleaved"  # 交错调度：减少气泡
    VIRTUAL_PIPELINE = "virtual" # 虚拟流水线
    ZERO_BUBBLE = "zero_bubble"  # 零气泡调度（理论上）

    @property
    def memory_efficiency(self) -> str:
        """内存效率等级"""
        efficiency_map = {
            self.GPIPE: "low",           # 需要存储所有微批次激活
            self.ONE_F_ONE_B: "high",    # 只存储少量激活
            self.INTERLEAVED: "medium",  # 中等
            self.VIRTUAL_PIPELINE: "medium",
            self.ZERO_BUBBLE: "medium",
        }
        return efficiency_map.get(self, "unknown")
    
    @property
    def bubble_overhead(self) -> str:
        """气泡开销等级"""
        overhead_map = {
            self.GPIPE: "high",
            self.ONE_F_ONE_B: "medium",
            self.INTERLEAVED: "low",
            self.VIRTUAL_PIPELINE: "low",
            self.ZERO_BUBBLE: "minimal",
        }
        return overhead_map.get(self, "unknown")
    
    @classmethod
    def recommend(cls, num_stages: int, num_micro_batches: int, memory_constrained: bool = False) -> 'PipelineSchedule':
        """
        根据配置推荐调度策略
        
        Args:
            num_stages: 阶段数
            num_micro_batches: 微批次数
            memory_constrained: 是否内存受限
            
        Returns:
            推荐的调度策略
        """
        if memory_constrained:
            return cls.ONE_F_ONE_B
        
        # 如果微批次数足够多，交错调度效率更高
        if num_micro_batches >= num_stages * 2:
            return cls.INTERLEAVED
        
        # 否则使用1F1B平衡内存和效率
        return cls.ONE_F_ONE_B


@dataclass
class PipelineStageConfig:
    """
    流水线阶段配置
    
    描述单个流水线阶段的配置信息。
    """
    stage_id: int
    start_layer: int
    end_layer: int
    device: torch.device
    
    # 通信配置
    prev_rank: Optional[int] = None  # 前一阶段的rank
    next_rank: Optional[int] = None  # 后一阶段的rank

    # 阶段信息
    num_layers: int = 0
    num_parameters: int = 0
    estimated_flops: float = 0.0
    estimated_memory_mb: float = 0.0
    
    # 通信缓冲区形状
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    
    def __post_init__(self):
        """计算派生字段"""
        if self.num_layers == 0:
            self.num_layers = self.end_layer - self.start_layer
    
    @property
    def is_first_stage(self) -> bool:
        """是否是第一个阶段"""
        return self.prev_rank is None
    
    @property
    def is_last_stage(self) -> bool:
        """是否是最后一个阶段"""
        return self.next_rank is None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'stage_id': self.stage_id,
            'start_layer': self.start_layer,
            'end_layer': self.end_layer,
            'device': str(self.device),
            'prev_rank': self.prev_rank,
            'next_rank': self.next_rank,
            'num_layers': self.num_layers,
            'num_parameters': self.num_parameters,
            'estimated_flops': self.estimated_flops,
            'estimated_memory_mb': self.estimated_memory_mb,
        }


@dataclass
class MicroBatchState:
    """
    微批次状态
    
    跟踪单个微批次在流水线中的状态。
    """
    micro_batch_id: int
    
    # 状态
    forward_done: bool = False
    backward_done: bool = False
    
    # 数据
    input_tensor: Optional[torch.Tensor] = None
    output_tensor: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    
    # 时间统计
    forward_start_time: Optional[float] = None
    forward_end_time: Optional[float] = None
    backward_start_time: Optional[float] = None
    backward_end_time: Optional[float] = None
    
    @property
    def forward_time(self) -> float:
        """前向传播时间"""
        if self.forward_start_time and self.forward_end_time:
            return self.forward_end_time - self.forward_start_time
        return 0.0
    
    @property
    def backward_time(self) -> float:
        """反向传播时间"""
        if self.backward_start_time and self.backward_end_time:
            return self.backward_end_time - self.backward_start_time
        return 0.0


# ==================== 调度器基类 ====================

class BaseScheduler(ABC):
    """
    调度器基类
    
    定义流水线调度的通用接口。
    """
    
    def __init__(
        self,
        num_stages: int,
        num_micro_batches: int,
        stage_id: int
    ):
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
        self.stage_id = stage_id
    
        # 统计
        self._forward_count = 0
        self._backward_count = 0
    
    @abstractmethod
    def forward_schedule(self) -> List[int]:
        """前向调度"""
        pass
    
    @abstractmethod
    def backward_schedule(self) -> List[int]:
        """反向调度"""
        pass
    
    @abstractmethod
    def get_bubble_fraction(self) -> float:
        """计算气泡比例"""
        pass
    
    def get_efficiency(self) -> float:
        """获取流水线效率 (0-1)"""
        return 1.0 - self.get_bubble_fraction()
    
    def get_total_steps(self) -> int:
        """获取总步数"""
        return len(self.forward_schedule()) + len(self.backward_schedule())
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """获取调度信息"""
        return {
            'num_stages': self.num_stages,
            'num_micro_batches': self.num_micro_batches,
            'stage_id': self.stage_id,
            'bubble_fraction': self.get_bubble_fraction(),
            'efficiency': self.get_efficiency(),
            'total_steps': self.get_total_steps(),
        }
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._forward_count = 0
        self._backward_count = 0


class GPipeSchedule(BaseScheduler):
    """
    GPipe调度器
    
    标准的GPipe调度：先完成所有前向，再完成所有反向。
    特点：实现简单，但内存使用高，气泡时间长。
    """
    
    def forward_schedule(self) -> List[int]:
        """
        前向调度
        
        GPipe: 按顺序执行所有微批次的前向
        """
        return list(range(self.num_micro_batches))
    
    def backward_schedule(self) -> List[int]:
        """
        反向调度
        
        GPipe: 逆序执行所有微批次的反向
        """
        return list(reversed(range(self.num_micro_batches)))
    
    def get_bubble_fraction(self) -> float:
        """
        计算气泡比例
        
        GPipe气泡比例 = (p-1) / (p-1+m)
        其中 p 是阶段数，m 是微批次数。
        """
        p = self.num_stages
        m = self.num_micro_batches
        if p + m <= 1:
            return 0.0
        return (p - 1) / (p - 1 + m)

    def get_warmup_steps(self) -> int:
        """获取预热步数"""
        return self.num_stages - 1
    
    def get_steady_steps(self) -> int:
        """获取稳定阶段步数"""
        return self.num_micro_batches - self.num_stages + 1
    
    def visualize_schedule(self) -> str:
        """
        可视化调度
        
        Returns:
            调度的ASCII可视化
        """
        lines = []
        lines.append(f"GPipe Schedule (stages={self.num_stages}, micro_batches={self.num_micro_batches})")
        lines.append("=" * 60)
        
        # 简化的可视化
        total_time = self.num_micro_batches + self.num_stages - 1
        
        for stage in range(self.num_stages):
            line = f"Stage {stage}: "
            # 前向
            for t in range(total_time):
                mb = t - stage
                if 0 <= mb < self.num_micro_batches:
                    line += f"F{mb} "
                else:
                    line += "-- "
            lines.append(line)
        
        lines.append("-" * 60)
        
        for stage in range(self.num_stages - 1, -1, -1):
            line = f"Stage {stage}: "
            # 反向
            offset = self.num_micro_batches
            for t in range(total_time):
                mb = self.num_micro_batches - 1 - (t - (self.num_stages - 1 - stage))
                if 0 <= mb < self.num_micro_batches:
                    line += f"B{mb} "
                else:
                    line += "-- "
            lines.append(line)
        
        return '\n'.join(lines)


class OneFOneBSchedule(BaseScheduler):
    """
    1F1B调度器
    
    1F1B (One Forward One Backward) 调度：
    - 预热阶段：只做前向
    - 稳定阶段：交替进行1次前向和1次反向
    - 冷却阶段：只做反向
    
    特点：内存使用较低，气泡时间中等。
    """
    
    def __init__(
        self,
        num_stages: int,
        num_micro_batches: int,
        stage_id: int
    ):
        super().__init__(num_stages, num_micro_batches, stage_id)
        
        # 计算各阶段步数
        self._warmup_steps = min(self.stage_id + 1, self.num_micro_batches)
        self._steady_steps = max(0, self.num_micro_batches - self.stage_id - 1)
        self._cooldown_steps = min(self.num_stages - self.stage_id - 1, self.num_micro_batches)
    
    def forward_schedule(self) -> List[int]:
        """前向调度"""
        schedule = []
        
        # 预热阶段
        for i in range(self._warmup_steps):
            schedule.append(i)
        
        # 稳定阶段的前向
        for i in range(self._steady_steps):
            schedule.append(self._warmup_steps + i)
        
        return schedule
    
    def backward_schedule(self) -> List[int]:
        """反向调度"""
        schedule = []
        
        # 稳定阶段的反向（交错）
        for i in range(self._steady_steps):
            schedule.append(i)
        
        # 冷却阶段
        for i in range(self._warmup_steps):
            schedule.append(self._steady_steps + i)
        
        return schedule
    
    def get_interleaved_schedule(self) -> List[Tuple[str, int]]:
        """
        获取交错调度
        
        Returns:
            (操作类型, 微批次ID) 列表
        """
        schedule = []
        
        # 预热阶段：只有前向
        for i in range(self._warmup_steps):
            schedule.append(('forward', i))
        
        # 稳定阶段：1F1B
        for i in range(self._steady_steps):
            schedule.append(('forward', self._warmup_steps + i))
            schedule.append(('backward', i))
        
        # 冷却阶段：只有反向
        for i in range(self._warmup_steps):
            schedule.append(('backward', self._steady_steps + i))
        
        return schedule
    
    def get_bubble_fraction(self) -> float:
        """
        计算气泡比例
        
        1F1B的气泡比例约为 (p-1) / m
        """
        p = self.num_stages
        m = self.num_micro_batches
        if m == 0:
            return 0.0
        return (p - 1) / m
    
    def get_peak_memory_factor(self) -> float:
        """
        获取峰值内存因子
        
        相对于GPipe的内存使用比例。
        """
        # 1F1B只需要存储约 num_stages 个微批次的激活
        return min(self.num_stages, self.num_micro_batches) / self.num_micro_batches


class InterleavedSchedule(BaseScheduler):
    """
    交错调度器
    
    1F1B调度的改进版，每个worker处理多个虚拟阶段。
    特点：进一步减少气泡时间。
    """
    
    def __init__(
        self,
        num_stages: int,
        num_micro_batches: int,
        stage_id: int,
        num_model_chunks: int = 2
    ):
        super().__init__(num_stages, num_micro_batches, stage_id)
        self.num_model_chunks = num_model_chunks
        
        # 虚拟阶段ID列表
        self.virtual_stages = [
            stage_id + i * num_stages
            for i in range(num_model_chunks)
        ]
    
    def forward_schedule(self) -> List[int]:
        """前向调度"""
        schedule = []
        for mb in range(self.num_micro_batches):
            schedule.append(mb)
        return schedule
    
    def backward_schedule(self) -> List[int]:
        """反向调度"""
        return list(reversed(range(self.num_micro_batches)))
    
    def get_full_schedule(self) -> List[Tuple[str, int, int]]:
        """
        获取完整的交错调度
        
        Returns:
            (操作类型, 微批次ID, 虚拟阶段ID) 列表
        """
        schedule = []
        
        # 预热阶段：填充流水线
        warmup_steps = self.num_stages - 1
        for i in range(warmup_steps):
            for virtual_stage in self.virtual_stages:
                schedule.append(('forward', i % self.num_micro_batches, virtual_stage))
        
        # 稳定阶段：1F1B
        for i in range(warmup_steps, self.num_micro_batches):
            for virtual_stage in self.virtual_stages:
                schedule.append(('forward', i, virtual_stage))
            for virtual_stage in reversed(self.virtual_stages):
                backward_idx = i - warmup_steps
                schedule.append(('backward', backward_idx, virtual_stage))
        
        # 冷却阶段：清空流水线
        for i in range(warmup_steps):
            for virtual_stage in reversed(self.virtual_stages):
                backward_idx = self.num_micro_batches - warmup_steps + i
                schedule.append(('backward', backward_idx, virtual_stage))
        
        return schedule
    
    def get_bubble_fraction(self) -> float:
        """
        计算气泡比例
        
        交错调度的气泡比例更低: (p-1) / (m*v)
        """
        p = self.num_stages
        m = self.num_micro_batches
        v = self.num_model_chunks
        if m * v == 0:
            return 0.0
        return (p - 1) / (m * v)
    
    def get_communication_volume_factor(self) -> float:
        """
        获取通信量因子
        
        交错调度需要更多的通信。
        """
        return self.num_model_chunks


# ==================== 通信管理器 ====================

class PipelineCommunicator:
    """
    流水线通信管理器
    
    管理流水线阶段之间的激活传输。
    """
    
    def __init__(
        self,
        stage_config: PipelineStageConfig,
        dtype: torch.dtype = torch.float16
    ):
        self.stage_config = stage_config
        self.dtype = dtype
        
        # 通信缓冲区
        self._send_buffer: Optional[torch.Tensor] = None
        self._recv_buffer: Optional[torch.Tensor] = None
        
        # 异步通信句柄
        self._pending_sends: List[Any] = []
        self._pending_recvs: List[Any] = []
        
        # 统计
        self._send_count = 0
        self._recv_count = 0
        self._total_bytes_sent = 0
        self._total_bytes_recv = 0
    
    def _get_tensor_size_bytes(self, tensor: torch.Tensor) -> int:
        """获取张量大小（字节）"""
        return tensor.numel() * tensor.element_size()
    
    def send_forward(
        self,
        tensor: torch.Tensor,
        async_op: bool = True
    ) -> Optional[Any]:
        """
        发送前向激活到下一阶段
        
        Args:
            tensor: 要发送的张量
            async_op: 是否异步操作
            
        Returns:
            异步操作句柄（如果async_op=True）
        """
        if self.stage_config.is_last_stage:
            return None
        
        if not dist.is_initialized():
            logger.warning("Distributed not initialized, skipping send")
            return None
        
        # 确保张量是连续的
        tensor = tensor.contiguous()
        
        # 发送
        handle = dist.isend(tensor, dst=self.stage_config.next_rank)
        
        self._send_count += 1
        self._total_bytes_sent += self._get_tensor_size_bytes(tensor)
        
        if async_op:
            self._pending_sends.append(handle)
            return handle
        else:
            handle.wait()
            return None
    
    def recv_forward(
        self,
        shape: Tuple[int, ...],
        async_op: bool = True
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        从前一阶段接收前向激活
        
        Args:
            shape: 期望的张量形状
            async_op: 是否异步操作
            
        Returns:
            (接收的张量, 异步操作句柄)
        """
        if self.stage_config.is_first_stage:
            return None, None
        
        if not dist.is_initialized():
            logger.warning("Distributed not initialized, skipping recv")
            return torch.zeros(shape, dtype=self.dtype, device=self.stage_config.device), None
        
        # 创建接收缓冲区
        recv_tensor = torch.empty(shape, dtype=self.dtype, device=self.stage_config.device)
        
        # 接收
        handle = dist.irecv(recv_tensor, src=self.stage_config.prev_rank)
        
        self._recv_count += 1
        self._total_bytes_recv += self._get_tensor_size_bytes(recv_tensor)
        
        if async_op:
            self._pending_recvs.append(handle)
            return recv_tensor, handle
        else:
            handle.wait()
            return recv_tensor, None
    
    def send_backward(
        self,
        tensor: torch.Tensor,
        async_op: bool = True
    ) -> Optional[Any]:
        """
        发送反向梯度到前一阶段
        
        Args:
            tensor: 要发送的梯度张量
            async_op: 是否异步操作
            
        Returns:
            异步操作句柄
        """
        if self.stage_config.is_first_stage:
            return None
        
        if not dist.is_initialized():
            return None
        
        tensor = tensor.contiguous()
        handle = dist.isend(tensor, dst=self.stage_config.prev_rank)
        
        self._send_count += 1
        self._total_bytes_sent += self._get_tensor_size_bytes(tensor)
        
        if async_op:
            self._pending_sends.append(handle)
            return handle
        else:
            handle.wait()
            return None
    
    def recv_backward(
        self,
        shape: Tuple[int, ...],
        async_op: bool = True
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """
        从后一阶段接收反向梯度
        
        Args:
            shape: 期望的张量形状
            async_op: 是否异步操作
            
        Returns:
            (接收的张量, 异步操作句柄)
        """
        if self.stage_config.is_last_stage:
            return None, None
        
        if not dist.is_initialized():
            return torch.zeros(shape, dtype=self.dtype, device=self.stage_config.device), None
        
        recv_tensor = torch.empty(shape, dtype=self.dtype, device=self.stage_config.device)
        handle = dist.irecv(recv_tensor, src=self.stage_config.next_rank)
        
        self._recv_count += 1
        self._total_bytes_recv += self._get_tensor_size_bytes(recv_tensor)
        
        if async_op:
            self._pending_recvs.append(handle)
            return recv_tensor, handle
        else:
            handle.wait()
            return recv_tensor, None
    
    def wait_all_sends(self) -> None:
        """等待所有发送完成"""
        for handle in self._pending_sends:
            handle.wait()
        self._pending_sends.clear()
    
    def wait_all_recvs(self) -> None:
        """等待所有接收完成"""
        for handle in self._pending_recvs:
            handle.wait()
        self._pending_recvs.clear()
    
    def synchronize(self) -> None:
        """同步所有通信"""
        self.wait_all_sends()
        self.wait_all_recvs()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取通信统计"""
        return {
            'send_count': self._send_count,
            'recv_count': self._recv_count,
            'total_bytes_sent': self._total_bytes_sent,
            'total_bytes_recv': self._total_bytes_recv,
            'total_mb_sent': self._total_bytes_sent / (1024**2),
            'total_mb_recv': self._total_bytes_recv / (1024**2),
        }
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._send_count = 0
        self._recv_count = 0
        self._total_bytes_sent = 0
        self._total_bytes_recv = 0


# ==================== 内存管理器 ====================

class PipelineMemoryManager:
    """
    流水线内存管理器
    
    管理激活内存，支持激活检查点。
    """
    
    def __init__(
        self,
        num_micro_batches: int,
        device: torch.device,
        checkpoint_activations: bool = False
    ):
        self.num_micro_batches = num_micro_batches
        self.device = device
        self.checkpoint_activations = checkpoint_activations
        
        # 激活存储
        self._activations: Dict[int, torch.Tensor] = {}  # micro_batch_id -> activation
        self._activation_shapes: Dict[int, Tuple[int, ...]] = {}
        
        # 内存统计
        self._peak_memory = 0.0
        self._current_memory = 0.0
    
    def store_activation(
        self,
        micro_batch_id: int,
        activation: torch.Tensor,
        detach: bool = True
    ) -> None:
        """
        存储激活
        
        Args:
            micro_batch_id: 微批次ID
            activation: 激活张量
            detach: 是否分离计算图
        """
        if detach:
            activation = activation.detach()
        
        if self.checkpoint_activations:
            # 检查点模式：不存储，需要重计算
            self._activation_shapes[micro_batch_id] = activation.shape
        else:
            self._activations[micro_batch_id] = activation
            self._activation_shapes[micro_batch_id] = activation.shape
        
        # 更新内存统计
        self._update_memory_stats()
    
    def get_activation(self, micro_batch_id: int) -> Optional[torch.Tensor]:
        """
        获取激活
        
        Args:
            micro_batch_id: 微批次ID
            
        Returns:
            激活张量
        """
        return self._activations.get(micro_batch_id)
    
    def get_activation_shape(self, micro_batch_id: int) -> Optional[Tuple[int, ...]]:
        """获取激活形状"""
        return self._activation_shapes.get(micro_batch_id)
    
    def clear_activation(self, micro_batch_id: int) -> None:
        """清除激活"""
        if micro_batch_id in self._activations:
            del self._activations[micro_batch_id]
        if micro_batch_id in self._activation_shapes:
            del self._activation_shapes[micro_batch_id]
    
    def clear_all(self) -> None:
        """清除所有激活"""
        self._activations.clear()
        self._activation_shapes.clear()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def _update_memory_stats(self) -> None:
        """更新内存统计"""
        if torch.cuda.is_available():
            current = torch.cuda.memory_allocated(self.device) / (1024**3)
            self._current_memory = current
            if current > self._peak_memory:
                self._peak_memory = current
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计"""
        num_stored = len(self._activations)
        total_size = sum(
            act.numel() * act.element_size()
            for act in self._activations.values()
        ) / (1024**2)  # MB
        
        return {
            'num_stored_activations': num_stored,
            'total_activation_size_mb': total_size,
            'peak_memory_gb': self._peak_memory,
            'current_memory_gb': self._current_memory,
            'checkpoint_mode': self.checkpoint_activations,
        }
    
    def estimate_memory_usage(self, activation_size_bytes: int) -> float:
        """
        估算内存使用
        
        Args:
            activation_size_bytes: 单个激活的大小
            
        Returns:
            估算的内存使用（MB）
        """
        if self.checkpoint_activations:
            # 检查点模式：只需要2个激活
            return 2 * activation_size_bytes / (1024**2)
        else:
            # 正常模式：需要存储所有激活
            return self.num_micro_batches * activation_size_bytes / (1024**2)


# ==================== 性能分析器 ====================

class PipelineProfiler:
    """
    流水线性能分析器
    
    分析流水线并行的性能瓶颈。
    """
    
    def __init__(self):
        self._stage_times: Dict[int, List[float]] = defaultdict(list)  # stage_id -> times
        self._forward_times: List[float] = []
        self._backward_times: List[float] = []
        self._comm_times: List[float] = []
        self._bubble_times: List[float] = []
        
        self._enabled = False
        self._start_time: Optional[float] = None
    
    def enable(self) -> None:
        """启用分析"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用分析"""
        self._enabled = False
    
    def start_iteration(self) -> None:
        """开始一次迭代"""
        if self._enabled:
            self._start_time = time.perf_counter()
    
    def end_iteration(self) -> None:
        """结束一次迭代"""
        if self._enabled and self._start_time:
            elapsed = time.perf_counter() - self._start_time
            self._start_time = None
    
    def record_forward(self, stage_id: int, duration: float) -> None:
        """记录前向时间"""
        if self._enabled:
            self._forward_times.append(duration)
            self._stage_times[stage_id].append(duration)
    
    def record_backward(self, stage_id: int, duration: float) -> None:
        """记录反向时间"""
        if self._enabled:
            self._backward_times.append(duration)
            self._stage_times[stage_id].append(duration)
    
    def record_communication(self, duration: float) -> None:
        """记录通信时间"""
        if self._enabled:
            self._comm_times.append(duration)
    
    def record_bubble(self, duration: float) -> None:
        """记录气泡时间"""
        if self._enabled:
            self._bubble_times.append(duration)
    
    @contextmanager
    def profile_forward(self, stage_id: int):
        """前向分析上下文"""
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
            duration = time.perf_counter() - start
            self.record_forward(stage_id, duration)
    
    @contextmanager
    def profile_backward(self, stage_id: int):
        """反向分析上下文"""
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
            duration = time.perf_counter() - start
            self.record_backward(stage_id, duration)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        def safe_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0
        
        def safe_sum(lst):
            return sum(lst)
        
        total_forward = safe_sum(self._forward_times)
        total_backward = safe_sum(self._backward_times)
        total_comm = safe_sum(self._comm_times)
        total_bubble = safe_sum(self._bubble_times)
        total_time = total_forward + total_backward + total_comm + total_bubble
        
        return {
            'forward': {
                'count': len(self._forward_times),
                'total_ms': total_forward * 1000,
                'avg_ms': safe_mean(self._forward_times) * 1000,
            },
            'backward': {
                'count': len(self._backward_times),
                'total_ms': total_backward * 1000,
                'avg_ms': safe_mean(self._backward_times) * 1000,
            },
            'communication': {
                'count': len(self._comm_times),
                'total_ms': total_comm * 1000,
                'avg_ms': safe_mean(self._comm_times) * 1000,
            },
            'bubble': {
                'count': len(self._bubble_times),
                'total_ms': total_bubble * 1000,
            },
            'total_time_ms': total_time * 1000,
            'compute_ratio': (total_forward + total_backward) / total_time if total_time > 0 else 0,
        }
    
    def get_stage_balance(self) -> Dict[int, float]:
        """
        获取阶段负载均衡情况
        
        Returns:
            各阶段的平均执行时间
        """
        return {
            stage_id: sum(times) / len(times) if times else 0.0
            for stage_id, times in self._stage_times.items()
        }
    
    def get_imbalance_factor(self) -> float:
        """
        获取负载不均衡因子
        
        Returns:
            最大/最小阶段时间比值
        """
        balance = self.get_stage_balance()
        if not balance:
            return 1.0
        
        times = list(balance.values())
        if min(times) == 0:
            return float('inf')
        
        return max(times) / min(times)
    
    def reset(self) -> None:
        """重置统计"""
        self._stage_times.clear()
        self._forward_times.clear()
        self._backward_times.clear()
        self._comm_times.clear()
        self._bubble_times.clear()
    
    def print_summary(self) -> None:
        """打印性能摘要"""
        stats = self.get_stats()
        
        print("\n=== Pipeline Performance Summary ===")
        print(f"Forward:  {stats['forward']['count']:4d} ops, "
              f"total={stats['forward']['total_ms']:.2f}ms, "
              f"avg={stats['forward']['avg_ms']:.2f}ms")
        print(f"Backward: {stats['backward']['count']:4d} ops, "
              f"total={stats['backward']['total_ms']:.2f}ms, "
              f"avg={stats['backward']['avg_ms']:.2f}ms")
        print(f"Comm:     {stats['communication']['count']:4d} ops, "
              f"total={stats['communication']['total_ms']:.2f}ms")
        print(f"Bubble:   {stats['bubble']['total_ms']:.2f}ms")
        print(f"Total:    {stats['total_time_ms']:.2f}ms")
        print(f"Compute Ratio: {stats['compute_ratio']:.1%}")
        print(f"Imbalance Factor: {self.get_imbalance_factor():.2f}x")


# ==================== 流水线包装器 ====================

class PipelineWrapper:
    """
    流水线并行包装器
    
    封装流水线并行的模型分割、调度和执行。
    """
    
    def __init__(self, config=None):
        from .parallel_modes import PipelineConfig
        self.config: PipelineConfig = config or PipelineConfig()
        
        # 模型和阶段
        self._model: Optional[nn.Module] = None
        self._stages: List[nn.Module] = []
        self._stage_configs: List[PipelineStageConfig] = []
        
        # 调度器
        self._scheduler: Optional[BaseScheduler] = None
        
        # 组件
        self._communicator: Optional[PipelineCommunicator] = None
        self._memory_manager: Optional[PipelineMemoryManager] = None
        self._profiler: PipelineProfiler = PipelineProfiler()
        
        # 微批次状态
        self._micro_batch_states: Dict[int, MicroBatchState] = {}
        
        # 状态
        self._is_split = False
        self._step_count = 0
    
    # ==================== 模型分割 ====================
    
    def split_model(
        self,
        model: nn.Module,
        split_points: Optional[List[str]] = None,
        balance: Optional[List[int]] = None
    ) -> List[nn.Module]:
        """
        分割模型为多个阶段
        
        Args:
            model: 原始模型
            split_points: 分割点（层名称列表）
            balance: 各阶段的层数平衡
            
        Returns:
            模型阶段列表
        """
        self._model = model
        
        # 分析模型
        model_info = self._analyze_model(model)
        logger.info(f"Splitting model: {model_info['num_params']:,} params, "
                   f"{model_info['num_layers']} layers")
        
        # 获取所有层
        layers = list(model.children())
        num_layers = len(layers)
        
        if split_points:
            self._stages = self._split_by_points(model, split_points)
        elif balance:
            self._stages = self._split_by_balance(layers, balance)
        else:
            # 使用配置中的balance或均匀分割
            if self.config.balance:
                self._stages = self._split_by_balance(layers, self.config.balance)
            else:
                balance = self._compute_optimal_balance(layers, self.config.num_stages)
                self._stages = self._split_by_balance(layers, balance)
        
        # 配置阶段
        self._configure_stages()
        
        # 初始化组件
        self._init_components()
        
        self._is_split = True
        logger.info(f"Model split into {len(self._stages)} stages")
        
        return self._stages
    
    def _analyze_model(self, model: nn.Module) -> Dict[str, Any]:
        """分析模型结构"""
        num_params = sum(p.numel() for p in model.parameters())
        num_layers = len(list(model.children()))
        
        layer_params = []
        for name, child in model.named_children():
            params = sum(p.numel() for p in child.parameters())
            layer_params.append({
                'name': name,
                'params': params,
                'type': type(child).__name__
            })
        
        return {
            'num_params': num_params,
            'num_layers': num_layers,
            'layer_params': layer_params,
        }
    
    def _compute_optimal_balance(
        self,
        layers: List[nn.Module],
        num_stages: int
    ) -> List[int]:
        """
        计算最优负载均衡
        
        尝试平衡各阶段的参数量。
        
        Args:
            layers: 层列表
            num_stages: 阶段数
            
        Returns:
            各阶段的层数列表
        """
        if len(layers) <= num_stages:
            # 层数不足，每个阶段一层
            return [1] * len(layers) + [0] * (num_stages - len(layers))
        
        # 计算每层的参数量
        layer_params = [sum(p.numel() for p in layer.parameters()) for layer in layers]
        total_params = sum(layer_params)
        target_per_stage = total_params / num_stages
        
        balance = []
        current_params = 0
        current_layers = 0
        
        for params in layer_params:
            current_params += params
            current_layers += 1
            
            # 如果当前阶段参数量接近目标，开始新阶段
            if current_params >= target_per_stage and len(balance) < num_stages - 1:
                balance.append(current_layers)
                current_params = 0
                current_layers = 0
        
        # 剩余层分配给最后一个阶段
        if current_layers > 0:
            balance.append(current_layers)
        
        # 确保阶段数正确
        while len(balance) < num_stages:
            balance.append(0)
        
        return balance
    
    def _split_by_points(
        self,
        model: nn.Module,
        split_points: List[str]
    ) -> List[nn.Module]:
        """按分割点分割"""
        stages = []
        current_layers = []
        
        for name, module in model.named_children():
            current_layers.append(module)
            
            if name in split_points:
                stage = nn.Sequential(*current_layers)
                stages.append(stage)
                current_layers = []
        
        if current_layers:
            stage = nn.Sequential(*current_layers)
            stages.append(stage)
        
        return stages
    
    def _split_by_balance(
        self,
        layers: List[nn.Module],
        balance: List[int]
    ) -> List[nn.Module]:
        """按balance分割"""
        stages = []
        start = 0
        
        for num_layers in balance:
            if num_layers > 0:
                end = min(start + num_layers, len(layers))
                stage = nn.Sequential(*layers[start:end])
                stages.append(stage)
                start = end
            else:
                # 空阶段
                stages.append(nn.Sequential())
        
        return stages
    
    def _configure_stages(self) -> None:
        """配置各阶段"""
        num_stages = len(self._stages)
        self._stage_configs.clear()
        
        layer_offset = 0
        for i, stage in enumerate(self._stages):
            # 确定设备
            if torch.cuda.is_available():
                device_id = self.config.local_rank if self.config.local_rank >= 0 else i % torch.cuda.device_count()
                device = torch.device(f'cuda:{device_id}')
            else:
                device = torch.device('cpu')
            
            # 移动到设备
            stage.to(device)
            
            # 计算阶段信息
            num_layers = len(list(stage.children()))
            num_params = sum(p.numel() for p in stage.parameters())
            
            # 创建配置
            config = PipelineStageConfig(
                stage_id=i,
                start_layer=layer_offset,
                end_layer=layer_offset + num_layers,
                device=device,
                prev_rank=i - 1 if i > 0 else None,
                next_rank=i + 1 if i < num_stages - 1 else None,
                num_layers=num_layers,
                num_parameters=num_params,
            )
            self._stage_configs.append(config)
            
            layer_offset += num_layers
    
    def _init_components(self) -> None:
        """初始化组件"""
        # 获取当前阶段配置
        current_stage_config = self.get_current_stage_config()
        
        if current_stage_config:
            # 检查配置是否有mixed_precision属性
            use_fp16 = getattr(self.config, 'mixed_precision', False)
            checkpoint_acts = getattr(self.config, 'checkpoint_activations', False)
            
            # 初始化通信器
            self._communicator = PipelineCommunicator(
                current_stage_config,
                dtype=torch.float16 if use_fp16 else torch.float32
            )
            
            # 初始化内存管理器
            self._memory_manager = PipelineMemoryManager(
                num_micro_batches=self.config.num_micro_batches,
                device=current_stage_config.device,
                checkpoint_activations=checkpoint_acts
            )
    
    # ==================== 调度器管理 ====================
    
    def create_scheduler(
        self,
        schedule_type: Optional[str] = None
    ) -> BaseScheduler:
        """
        创建调度器
        
        Args:
            schedule_type: 调度类型
            
        Returns:
            调度器实例
        """
        schedule = schedule_type or self.config.schedule
        
        if schedule == "gpipe":
            self._scheduler = GPipeSchedule(
                num_stages=self.config.num_stages,
                num_micro_batches=self.config.num_micro_batches,
                stage_id=self.config.rank
            )
        elif schedule == "1f1b":
            self._scheduler = OneFOneBSchedule(
                num_stages=self.config.num_stages,
                num_micro_batches=self.config.num_micro_batches,
                stage_id=self.config.rank
            )
        elif schedule == "interleaved":
            self._scheduler = InterleavedSchedule(
                num_stages=self.config.num_stages,
                num_micro_batches=self.config.num_micro_batches,
                stage_id=self.config.rank
            )
        else:
            # 默认1F1B
            self._scheduler = OneFOneBSchedule(
                num_stages=self.config.num_stages,
                num_micro_batches=self.config.num_micro_batches,
                stage_id=self.config.rank
            )
        
        logger.info(f"Created {schedule} scheduler with efficiency {self._scheduler.get_efficiency():.1%}")
        return self._scheduler
    
    def get_scheduler(self) -> Optional[BaseScheduler]:
        """获取调度器"""
        return self._scheduler
    
    def get_schedule_info(self) -> Dict[str, Any]:
        """获取调度信息"""
        if self._scheduler:
            return self._scheduler.get_schedule_info()
        return {}
    
    # ==================== 前向/反向传播 ====================
    
    def forward(
        self,
        inputs: torch.Tensor,
        stage_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        前向传播（单阶段）
        
        Args:
            inputs: 输入张量
            stage_id: 阶段ID
            
        Returns:
            输出张量
        """
        stage_id = stage_id if stage_id is not None else self.config.rank
        
        if stage_id >= len(self._stages):
            return inputs
        
        stage = self._stages[stage_id]
        
        with self._profiler.profile_forward(stage_id):
            output = stage(inputs)
        
        return output
    
    def forward_step(
        self,
        micro_batch_id: int,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        单个微批次的前向步骤
        
        Args:
            micro_batch_id: 微批次ID
            input_tensor: 输入张量
            
        Returns:
            输出张量
        """
        # 初始化微批次状态
        if micro_batch_id not in self._micro_batch_states:
            self._micro_batch_states[micro_batch_id] = MicroBatchState(micro_batch_id)
        
        state = self._micro_batch_states[micro_batch_id]
        state.forward_start_time = time.perf_counter()
        state.input_tensor = input_tensor
        
        # 执行前向
        output = self.forward(input_tensor)
        
        state.output_tensor = output
        state.forward_end_time = time.perf_counter()
        state.forward_done = True
        
        # 存储激活（用于反向传播）
        if self._memory_manager:
            self._memory_manager.store_activation(micro_batch_id, output)
        
        return output
    
    def backward_step(
        self,
        micro_batch_id: int,
        output_grad: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable] = None
    ) -> Optional[torch.Tensor]:
        """
        单个微批次的反向步骤
        
        Args:
            micro_batch_id: 微批次ID
            output_grad: 输出梯度（从下一阶段接收）
            loss_fn: 损失函数（最后一个阶段使用）
            
        Returns:
            输入梯度（发送给前一阶段）
        """
        state = self._micro_batch_states.get(micro_batch_id)
        if state is None:
            raise RuntimeError(f"Micro batch {micro_batch_id} not found")
        
        state.backward_start_time = time.perf_counter()
        
        # 获取存储的激活
        output = state.output_tensor
        if output is None and self._memory_manager:
            output = self._memory_manager.get_activation(micro_batch_id)
        
        if output is None:
            raise RuntimeError(f"No activation found for micro batch {micro_batch_id}")
        
        # 计算梯度
        if output_grad is not None:
            # 使用接收的梯度
            output.backward(output_grad)
        elif loss_fn is not None:
            # 最后一个阶段，计算损失
            loss = loss_fn(output)
            state.loss = loss
            loss.backward()
        else:
            raise RuntimeError("Either output_grad or loss_fn must be provided")
        
        # 获取输入梯度（发送给前一阶段）
        input_grad = None
        if state.input_tensor is not None and state.input_tensor.grad is not None:
            input_grad = state.input_tensor.grad.clone()
        
        state.backward_end_time = time.perf_counter()
        state.backward_done = True
        
        # 清理激活
        if self._memory_manager:
            self._memory_manager.clear_activation(micro_batch_id)
        
        return input_grad
    
    def run_pipeline(
        self,
        batches: List[torch.Tensor],
        loss_fn: Callable,
        labels: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        运行完整的流水线
        
        Args:
            batches: 微批次输入列表
            loss_fn: 损失函数
            labels: 标签列表（可选）
            
        Returns:
            (总损失, 输出列表)
        """
        if self._scheduler is None:
            self.create_scheduler()
        
        self._profiler.start_iteration()
        
        outputs = []
        losses = []
        
        # 清理状态
        self._micro_batch_states.clear()
        if self._memory_manager:
            self._memory_manager.clear_all()
        
        # 前向传播
        for micro_batch_id in self._scheduler.forward_schedule():
            batch = batches[micro_batch_id]
            output = self.forward_step(micro_batch_id, batch)
            outputs.append(output)
        
        # 反向传播
        for micro_batch_id in self._scheduler.backward_schedule():
            output = outputs[micro_batch_id]
            
            # 创建损失函数包装器
            if labels is not None:
                label = labels[micro_batch_id]
                def loss_wrapper(out):
                    return loss_fn(out, label)
            else:
                loss_wrapper = loss_fn
            
            self.backward_step(micro_batch_id, loss_fn=loss_wrapper)
            
            state = self._micro_batch_states[micro_batch_id]
            if state.loss is not None:
                losses.append(state.loss.detach())
        
        # 聚合损失
        if losses:
            total_loss = torch.stack(losses).mean()
        else:
            total_loss = torch.tensor(0.0)
        
        self._profiler.end_iteration()
        self._step_count += 1
        
        return total_loss, outputs
    
    def run_1f1b_pipeline(
        self,
        batches: List[torch.Tensor],
        loss_fn: Callable,
        labels: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        运行1F1B流水线
        
        更优化的1F1B执行，交错前向和反向。
        
        Args:
            batches: 微批次输入列表
            loss_fn: 损失函数
            labels: 标签列表
            
        Returns:
            (总损失, 输出列表)
        """
        if not isinstance(self._scheduler, OneFOneBSchedule):
            self.create_scheduler("1f1b")
        
        self._profiler.start_iteration()
        
        outputs = [None] * len(batches)
        losses = []
        
        # 清理状态
        self._micro_batch_states.clear()
        if self._memory_manager:
            self._memory_manager.clear_all()
        
        # 获取交错调度
        schedule = self._scheduler.get_interleaved_schedule()
        
        for op_type, micro_batch_id in schedule:
            if op_type == 'forward':
                batch = batches[micro_batch_id]
                output = self.forward_step(micro_batch_id, batch)
                outputs[micro_batch_id] = output
            else:  # backward
                if labels is not None:
                    label = labels[micro_batch_id]
                    def loss_wrapper(out):
                        return loss_fn(out, label)
                else:
                    loss_wrapper = loss_fn
                
                self.backward_step(micro_batch_id, loss_fn=loss_wrapper)
                
                state = self._micro_batch_states[micro_batch_id]
                if state.loss is not None:
                    losses.append(state.loss.detach())
        
        # 聚合损失
        if losses:
            total_loss = torch.stack(losses).mean()
        else:
            total_loss = torch.tensor(0.0)
        
        self._profiler.end_iteration()
        self._step_count += 1
        
        return total_loss, outputs
    
    # ==================== 阶段访问 ====================
    
    def get_stage(self, stage_id: int) -> nn.Module:
        """获取指定阶段"""
        if stage_id >= len(self._stages):
            raise IndexError(f"Stage {stage_id} not found")
        return self._stages[stage_id]
    
    def get_current_stage(self) -> nn.Module:
        """获取当前rank对应的阶段"""
        idx = self.config.rank % len(self._stages) if self._stages else 0
        return self._stages[idx] if self._stages else None
    
    def get_current_stage_config(self) -> Optional[PipelineStageConfig]:
        """获取当前阶段配置"""
        if not self._stage_configs:
            return None
        idx = self.config.rank % len(self._stage_configs)
        return self._stage_configs[idx]
    
    @property
    def num_stages(self) -> int:
        """阶段数"""
        return len(self._stages)
    
    @property
    def is_split(self) -> bool:
        """模型是否已分割"""
        return self._is_split
    
    # ==================== 效率分析 ====================
    
    def get_bubble_ratio(self) -> float:
        """获取气泡比例"""
        if self._scheduler:
            return self._scheduler.get_bubble_fraction()
        
        # 默认使用GPipe公式估算
        p = self.config.num_stages
        m = self.config.num_micro_batches
        if p + m <= 1:
            return 0.0
        return (p - 1) / (p - 1 + m)
    
    def get_efficiency(self) -> float:
        """获取流水线效率"""
        return 1.0 - self.get_bubble_ratio()
    
    def get_optimal_micro_batches(self, target_efficiency: float = 0.9) -> int:
        """
        计算达到目标效率所需的微批次数
        
        Args:
            target_efficiency: 目标效率
            
        Returns:
            推荐的微批次数
        """
        p = self.config.num_stages
        # (p-1) / (p-1+m) = 1 - target_efficiency
        # m = (p-1) / (1 - target_efficiency) - (p-1)
        # m = (p-1) * target_efficiency / (1 - target_efficiency)
        
        if target_efficiency >= 1.0:
            return p * 10  # 非常大的值
        
        bubble = 1.0 - target_efficiency
        m = int((p - 1) * target_efficiency / bubble)
        
        return max(m, p)  # 至少等于阶段数
    
    def get_load_balance(self) -> Dict[str, Any]:
        """
        获取负载均衡信息
        
        Returns:
            负载均衡统计
        """
        if not self._stage_configs:
            return {}
        
        params = [cfg.num_parameters for cfg in self._stage_configs]
        total_params = sum(params)
        
        if total_params == 0:
            return {'balanced': True, 'imbalance_ratio': 1.0}
        
        avg_params = total_params / len(params)
        max_params = max(params)
        min_params = min(params) if min(params) > 0 else 1
        
        return {
            'stage_params': params,
            'total_params': total_params,
            'avg_params': avg_params,
            'max_params': max_params,
            'min_params': min_params,
            'imbalance_ratio': max_params / min_params if min_params > 0 else float('inf'),
            'balanced': max_params / avg_params < 1.2 if avg_params > 0 else True,
        }
    
    def suggest_rebalance(self) -> Optional[List[int]]:
        """
        建议重新平衡
        
        Returns:
            建议的新balance，如果当前已平衡返回None
        """
        balance_info = self.get_load_balance()
        
        if balance_info.get('balanced', True):
            return None
        
        # 使用参数量重新计算balance
        if self._model:
            layers = list(self._model.children())
            return self._compute_optimal_balance(layers, self.config.num_stages)
        
        return None
    
    # ==================== 检查点管理 ====================
    
    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        保存检查点
        
        Args:
            path: 保存路径
            optimizer: 优化器
            epoch: 当前epoch
            **kwargs: 额外数据
        """
        # 收集所有阶段的状态
        stage_state_dicts = [stage.state_dict() for stage in self._stages]
        
        checkpoint = {
            'stage_state_dicts': stage_state_dicts,
                'num_stages': len(self._stages),
            'stage_configs': [cfg.to_dict() for cfg in self._stage_configs],
            'step_count': self._step_count,
            'config': {
                'num_stages': self.config.num_stages,
                'num_micro_batches': self.config.num_micro_batches,
                'schedule': self.config.schedule,
            },
                **kwargs
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
        
        # 只在rank 0保存
        if self.config.rank == 0:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            torch.save(checkpoint, path)
            logger.info(f"Pipeline checkpoint saved: {path}")
    
        # 同步
        if dist.is_initialized():
            dist.barrier()
    
    def save_stage_checkpoint(
        self,
        path: str,
        stage_id: Optional[int] = None
    ) -> None:
        """
        保存单个阶段的检查点
        
        Args:
            path: 保存路径
            stage_id: 阶段ID（默认当前阶段）
        """
        stage_id = stage_id if stage_id is not None else self.config.rank
        
        if stage_id >= len(self._stages):
            return
        
        stage = self._stages[stage_id]
        stage_path = f"{path}.stage{stage_id}"
        
        checkpoint = {
            'stage_state_dict': stage.state_dict(),
            'stage_id': stage_id,
            'stage_config': self._stage_configs[stage_id].to_dict() if stage_id < len(self._stage_configs) else {},
        }
        
        os.makedirs(os.path.dirname(stage_path) if os.path.dirname(stage_path) else '.', exist_ok=True)
        torch.save(checkpoint, stage_path)
        logger.info(f"Stage {stage_id} checkpoint saved: {stage_path}")
    
    def load_checkpoint(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            path: 检查点路径
            optimizer: 优化器
            strict: 是否严格匹配
            
        Returns:
            检查点数据
        """
        checkpoint = torch.load(path, map_location='cpu')
        
        state_dicts = checkpoint.get('stage_state_dicts', [])
        for i, stage in enumerate(self._stages):
            if i < len(state_dicts):
                stage.load_state_dict(state_dicts[i], strict=strict)
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'step_count' in checkpoint:
            self._step_count = checkpoint['step_count']
        
        # 同步
        if dist.is_initialized():
            dist.barrier()
        
        logger.info(f"Pipeline checkpoint loaded: {path}")
        return checkpoint
    
    def load_stage_checkpoint(
        self,
        path: str,
        stage_id: Optional[int] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        加载单个阶段的检查点
        
        Args:
            path: 检查点路径
            stage_id: 阶段ID
            strict: 是否严格匹配
            
        Returns:
            检查点数据
        """
        stage_id = stage_id if stage_id is not None else self.config.rank
        stage_path = f"{path}.stage{stage_id}"
        
        checkpoint = torch.load(stage_path, map_location='cpu')
        
        if stage_id < len(self._stages):
            self._stages[stage_id].load_state_dict(
                checkpoint['stage_state_dict'],
                strict=strict
            )
        
        logger.info(f"Stage {stage_id} checkpoint loaded: {stage_path}")
        return checkpoint
    
    # ==================== 性能分析 ====================
    
    def enable_profiling(self) -> None:
        """启用性能分析"""
        self._profiler.enable()
    
    def disable_profiling(self) -> None:
        """禁用性能分析"""
        self._profiler.disable()
    
    def get_profiling_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self._profiler.get_stats()
    
    def print_profiling_summary(self) -> None:
        """打印性能摘要"""
        self._profiler.print_summary()
    
    def reset_profiling(self) -> None:
        """重置性能统计"""
        self._profiler.reset()
    
    # ==================== 内存管理 ====================
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计"""
        if self._memory_manager:
            return self._memory_manager.get_memory_stats()
        return {}
    
    def clear_memory(self) -> None:
        """清理内存"""
        if self._memory_manager:
            self._memory_manager.clear_all()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def cleanup(self) -> None:
        """清理Pipeline资源"""
        # 清理内存
        self.clear_memory()
        
        # 清理微批次状态
        self._micro_batch_states.clear()
        
        # 重置状态
        self._is_split = False
        self._stages = []
        self._stage_configs = []
        self._scheduler = None
        self._communicator = None
        self._model = None
        
        logger.info("Pipeline cleaned up")
    
    # ==================== 通信统计 ====================
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """获取通信统计"""
        if self._communicator:
            return self._communicator.get_stats()
        return {}
    
    def reset_communication_stats(self) -> None:
        """重置通信统计"""
        if self._communicator:
            self._communicator.reset_stats()
    
    # ==================== 诊断 ====================
    
    def diagnose(self) -> Dict[str, Any]:
        """
        运行诊断
        
        Returns:
            诊断结果
        """
        diagnosis = {
            'is_split': self._is_split,
            'num_stages': self.num_stages,
            'scheduler': type(self._scheduler).__name__ if self._scheduler else None,
            'efficiency': self.get_efficiency(),
            'bubble_ratio': self.get_bubble_ratio(),
            'load_balance': self.get_load_balance(),
            'memory': self.get_memory_stats(),
            'communication': self.get_communication_stats(),
            'step_count': self._step_count,
        }
        
        # 检查问题
        issues = []
        
        if not self._is_split:
            issues.append("Model not split - call split_model() first")
        
        if self._scheduler is None:
            issues.append("Scheduler not created - call create_scheduler() first")
        
        balance_info = self.get_load_balance()
        if not balance_info.get('balanced', True):
            issues.append(f"Load imbalance detected: ratio={balance_info.get('imbalance_ratio', 0):.2f}")
        
        if self.get_efficiency() < 0.8:
            issues.append(f"Low pipeline efficiency: {self.get_efficiency():.1%}")
        
        diagnosis['issues'] = issues
        
        # 建议
        suggestions = []
        
        if self.get_efficiency() < 0.9:
            optimal_mb = self.get_optimal_micro_batches(0.9)
            suggestions.append(f"Consider using {optimal_mb} micro-batches for 90% efficiency")
        
        if balance_info.get('imbalance_ratio', 1.0) > 1.5:
            suggestions.append("Consider rebalancing the pipeline stages")
        
        diagnosis['suggestions'] = suggestions
        
        return diagnosis
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diagnosis = self.diagnose()
        
        print("\n=== Pipeline Wrapper Diagnosis ===")
        print(f"Model Split: {diagnosis['is_split']}")
        print(f"Num Stages: {diagnosis['num_stages']}")
        print(f"Scheduler: {diagnosis['scheduler']}")
        print(f"Efficiency: {diagnosis['efficiency']:.1%}")
        print(f"Bubble Ratio: {diagnosis['bubble_ratio']:.1%}")
        
        print("\nLoad Balance:")
        for key, value in diagnosis['load_balance'].items():
            print(f"  {key}: {value}")
        
        if diagnosis['issues']:
            print("\nIssues:")
            for issue in diagnosis['issues']:
                print(f"  ⚠ {issue}")
        
        if diagnosis['suggestions']:
            print("\nSuggestions:")
            for suggestion in diagnosis['suggestions']:
                print(f"  → {suggestion}")


# ==================== 便捷函数 ====================

def create_pipeline_model(
    model: nn.Module,
    num_stages: int = 4,
    num_micro_batches: int = 8,
    split_points: Optional[List[str]] = None,
    balance: Optional[List[int]] = None,
    schedule: str = "1f1b",
    **kwargs
) -> Tuple[PipelineWrapper, List[nn.Module]]:
    """
    创建流水线模型
    
    Args:
        model: 原始模型
        num_stages: 阶段数
        num_micro_batches: 微批次数
        split_points: 分割点
        balance: 层数平衡
        schedule: 调度策略
        **kwargs: 其他配置
        
    Returns:
        (PipelineWrapper实例, 阶段列表)
    """
    from .parallel_modes import PipelineConfig
    
    config = PipelineConfig(
        num_stages=num_stages,
        num_micro_batches=num_micro_batches,
        split_points=split_points or [],
        balance=balance or [],
        schedule=schedule,
        **kwargs
    )
    
    wrapper = PipelineWrapper(config)
    stages = wrapper.split_model(model, split_points, balance)
    wrapper.create_scheduler(schedule)
    
    return wrapper, stages


@contextmanager
def pipeline_context(config=None):
    """
    流水线上下文管理器
    
    Args:
        config: 流水线配置
        
    Yields:
        PipelineWrapper实例
    """
    from .parallel_modes import PipelineConfig
    wrapper = PipelineWrapper(config or PipelineConfig())
    try:
        yield wrapper
    finally:
        wrapper.clear_memory()


def estimate_pipeline_efficiency(
    num_stages: int,
    num_micro_batches: int,
    schedule: str = "1f1b"
) -> Dict[str, float]:
    """
    估算流水线效率
    
    Args:
        num_stages: 阶段数
        num_micro_batches: 微批次数
        schedule: 调度策略
        
    Returns:
        效率估算
    """
    p = num_stages
    m = num_micro_batches
    
    if schedule == "gpipe":
        bubble = (p - 1) / (p - 1 + m) if p + m > 1 else 0
    elif schedule == "1f1b":
        bubble = (p - 1) / m if m > 0 else 0
    elif schedule == "interleaved":
        v = 2  # 默认2个模型块
        bubble = (p - 1) / (m * v) if m * v > 0 else 0
    else:
        bubble = (p - 1) / (p - 1 + m) if p + m > 1 else 0
    
    return {
        'bubble_ratio': bubble,
        'efficiency': 1.0 - bubble,
        'schedule': schedule,
        'num_stages': p,
        'num_micro_batches': m,
    }


def recommend_pipeline_config(
    model: nn.Module,
    num_gpus: int,
    target_efficiency: float = 0.9,
    memory_constrained: bool = False
) -> Dict[str, Any]:
    """
    推荐流水线配置
    
    Args:
        model: 模型
        num_gpus: GPU数量
        target_efficiency: 目标效率
        memory_constrained: 是否内存受限
        
    Returns:
        推荐配置
    """
    # 分析模型
    num_layers = len(list(model.children()))
    num_params = sum(p.numel() for p in model.parameters())
    
    # 推荐阶段数
    num_stages = min(num_gpus, num_layers)
    
    # 推荐调度策略
    schedule = PipelineSchedule.recommend(num_stages, num_stages * 2, memory_constrained)
    
    # 计算达到目标效率的微批次数
    # 对于1F1B: bubble = (p-1)/m, 所以 m = (p-1)/(1-efficiency)
    if target_efficiency >= 1.0:
        num_micro_batches = num_stages * 10
    else:
        bubble = 1.0 - target_efficiency
        num_micro_batches = max(int((num_stages - 1) / bubble), num_stages)
    
    # 计算balance
    layers = list(model.children())
    layer_params = [sum(p.numel() for p in layer.parameters()) for layer in layers]
    
    total_params = sum(layer_params)
    target_per_stage = total_params / num_stages
    
    balance = []
    current = 0
    count = 0
    for params in layer_params:
        current += params
        count += 1
        if current >= target_per_stage and len(balance) < num_stages - 1:
            balance.append(count)
            current = 0
            count = 0
    if count > 0:
        balance.append(count)
    
    # 确保balance长度正确
    while len(balance) < num_stages:
        balance.append(0)
    
    return {
        'num_stages': num_stages,
        'num_micro_batches': num_micro_batches,
        'schedule': schedule.value,
        'balance': balance,
        'estimated_efficiency': estimate_pipeline_efficiency(num_stages, num_micro_batches, schedule.value)['efficiency'],
        'model_info': {
            'num_layers': num_layers,
            'num_params': num_params,
        }
    }


def visualize_pipeline_schedule(
    num_stages: int,
    num_micro_batches: int,
    schedule: str = "gpipe"
) -> str:
    """
    可视化流水线调度
    
    Args:
        num_stages: 阶段数
        num_micro_batches: 微批次数
        schedule: 调度策略
        
    Returns:
        ASCII可视化字符串
    """
    if schedule == "gpipe":
        scheduler = GPipeSchedule(num_stages, num_micro_batches, 0)
        return scheduler.visualize_schedule()
    
    # 简化的可视化
    lines = []
    lines.append(f"{schedule.upper()} Schedule (stages={num_stages}, micro_batches={num_micro_batches})")
    lines.append("=" * 60)
    
    efficiency = estimate_pipeline_efficiency(num_stages, num_micro_batches, schedule)
    lines.append(f"Efficiency: {efficiency['efficiency']:.1%}")
    lines.append(f"Bubble Ratio: {efficiency['bubble_ratio']:.1%}")
    
    return '\n'.join(lines)

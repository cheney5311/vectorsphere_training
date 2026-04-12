# -*- coding: utf-8 -*-
"""
ZeRO (Zero Redundancy Optimizer) 包装器

提供DeepSpeed ZeRO优化的封装和管理，支持生产级功能，包括：
- ZeRO Stage 1/2/3优化
- CPU/NVMe卸载
- 内存监控和优化
- 梯度累积和同步控制
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
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist

logger = logging.getLogger(__name__)

# 尝试导入DeepSpeed
try:
    import deepspeed
    from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
    from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    logger.warning("DeepSpeed not available, ZeRO optimizations disabled")


# ==================== 枚举和配置类 ====================

class ZeROStage(Enum):
    """
    ZeRO阶段
    
    不同阶段提供不同的内存优化级别。
    """
    STAGE_0 = 0  # 禁用ZeRO
    STAGE_1 = 1  # 优化器状态分片
    STAGE_2 = 2  # 优化器状态 + 梯度分片
    STAGE_3 = 3  # 优化器状态 + 梯度 + 参数分片

    @property
    def memory_efficiency(self) -> float:
        """
        内存效率系数
        
        相对于Stage 0的内存节省比例（近似值）。
        """
        efficiency_map = {
            self.STAGE_0: 1.0,
            self.STAGE_1: 0.75,  # 优化器状态约占4倍参数
            self.STAGE_2: 0.5,   # + 梯度
            self.STAGE_3: 0.25,  # + 参数
        }
        return efficiency_map.get(self, 1.0)
    
    @property
    def communication_overhead(self) -> str:
        """通信开销等级"""
        overhead_map = {
            self.STAGE_0: "none",
            self.STAGE_1: "low",
            self.STAGE_2: "medium",
            self.STAGE_3: "high",
        }
        return overhead_map.get(self, "unknown")
    
    @property
    def description(self) -> str:
        """阶段描述"""
        desc_map = {
            self.STAGE_0: "Disabled - No ZeRO optimization",
            self.STAGE_1: "Optimizer State Partitioning",
            self.STAGE_2: "Optimizer + Gradient Partitioning",
            self.STAGE_3: "Full Parameter Partitioning",
        }
        return desc_map.get(self, "Unknown")
    
    @classmethod
    def from_int(cls, value: int) -> 'ZeROStage':
        """从整数创建"""
        for stage in cls:
            if stage.value == value:
                return stage
        raise ValueError(f"Invalid ZeRO stage: {value}")
    
    @classmethod
    def recommend(
        cls,
        model_params: int,
        num_gpus: int,
        gpu_memory_gb: float = 80.0
    ) -> 'ZeROStage':
        """
        根据模型大小和资源推荐ZeRO阶段
        
        Args:
            model_params: 模型参数量
            num_gpus: GPU数量
            gpu_memory_gb: 单卡内存
            
        Returns:
            推荐的ZeRO阶段
        """
        # 估算内存需求 (bytes)
        # 参数 + 梯度 + 优化器状态(Adam: 2倍参数) ≈ 16-20 bytes/param (fp16)
        memory_per_param = 18
        total_memory_gb = model_params * memory_per_param / (1024**3)
        memory_per_gpu = total_memory_gb / num_gpus
        
        if memory_per_gpu <= gpu_memory_gb * 0.4:
            return cls.STAGE_0  # 内存充足，不需要ZeRO
        elif memory_per_gpu <= gpu_memory_gb * 0.6:
            return cls.STAGE_1
        elif memory_per_gpu <= gpu_memory_gb * 0.8:
            return cls.STAGE_2
        else:
            return cls.STAGE_3


class OffloadDevice(Enum):
    """卸载设备"""
    NONE = "none"
    CPU = "cpu"
    NVME = "nvme"

    @property
    def requires_pin_memory(self) -> bool:
        """是否需要固定内存"""
        return self in (self.CPU, self.NVME)


@dataclass
class ZeROOffloadConfig:
    """
    ZeRO卸载配置
    
    控制优化器状态和参数的CPU/NVMe卸载。
    """
    # 优化器卸载
    offload_optimizer: bool = False
    optimizer_device: OffloadDevice = OffloadDevice.CPU
    optimizer_pin_memory: bool = True
    
    # 参数卸载（仅Stage 3）
    offload_param: bool = False
    param_device: OffloadDevice = OffloadDevice.CPU
    param_pin_memory: bool = True
    
    # NVMe配置
    nvme_path: str = "/local_nvme"
    nvme_buffer_size: int = 100_000_000  # 100MB
    nvme_buffer_count: int = 4
    
    def validate(self) -> List[str]:
        """验证配置"""
        warnings = []
        
        if self.offload_param and not self.offload_optimizer:
            warnings.append("offload_param without offload_optimizer may not be optimal")
        
        if self.optimizer_device == OffloadDevice.NVME:
            if not os.path.exists(self.nvme_path):
                warnings.append(f"NVMe path does not exist: {self.nvme_path}")
        
        return warnings
    
    def to_deepspeed_dict(self, stage: int) -> Dict[str, Any]:
        """转换为DeepSpeed配置"""
        config = {}
        
        if self.offload_optimizer:
            config["offload_optimizer"] = {
                "device": self.optimizer_device.value if self.optimizer_device != OffloadDevice.NONE else "cpu",
                "pin_memory": self.optimizer_pin_memory,
            }
            if self.optimizer_device == OffloadDevice.NVME:
                config["offload_optimizer"]["nvme_path"] = self.nvme_path
                config["offload_optimizer"]["buffer_size"] = self.nvme_buffer_size
                config["offload_optimizer"]["buffer_count"] = self.nvme_buffer_count
        
        if self.offload_param and stage == 3:
            config["offload_param"] = {
                "device": self.param_device.value if self.param_device != OffloadDevice.NONE else "cpu",
                "pin_memory": self.param_pin_memory,
            }
            if self.param_device == OffloadDevice.NVME:
                config["offload_param"]["nvme_path"] = self.nvme_path
        
        return config


@dataclass
class ZeROCommunicationConfig:
    """
    ZeRO通信配置
    
    控制ZeRO的通信优化。
    """
    # 通信重叠
    overlap_comm: bool = True
    
    # 连续梯度
    contiguous_gradients: bool = True
    
    # 桶大小
    reduce_bucket_size: int = 500_000_000  # 500MB
    allgather_bucket_size: int = 500_000_000
    
    # Stage 3特定
    stage3_max_live_parameters: int = 1_000_000_000  # 1B
    stage3_max_reuse_distance: int = 1_000_000_000
    stage3_prefetch_bucket_size: int = 500_000_000
    stage3_param_persistence_threshold: int = 100_000
    
    # 通信优化
    reduce_scatter: bool = True
    allgather_partitions: bool = True
    
    def get_optimal_bucket_size(self, model_params: int) -> int:
        """
        根据模型大小计算最优桶大小
        
        Args:
            model_params: 模型参数量
            
        Returns:
            推荐的桶大小
        """
        # 经验公式：桶大小约为模型参数的1-5%
        suggested = model_params * 4 // 100  # 假设fp32
        
        # 限制在合理范围内
        min_size = 100_000_000   # 100MB
        max_size = 2_000_000_000  # 2GB
        
        return max(min_size, min(suggested, max_size))
    
    def to_deepspeed_dict(self, stage: int) -> Dict[str, Any]:
        """转换为DeepSpeed配置"""
        config = {
            "overlap_comm": self.overlap_comm,
            "contiguous_gradients": self.contiguous_gradients,
            "reduce_bucket_size": self.reduce_bucket_size,
            "allgather_bucket_size": self.allgather_bucket_size,
            "reduce_scatter": self.reduce_scatter,
            "allgather_partitions": self.allgather_partitions,
        }
        
        if stage == 3:
            config.update({
                "stage3_max_live_parameters": self.stage3_max_live_parameters,
                "stage3_max_reuse_distance": self.stage3_max_reuse_distance,
                "stage3_prefetch_bucket_size": self.stage3_prefetch_bucket_size,
                "stage3_param_persistence_threshold": self.stage3_param_persistence_threshold,
            })
        
        return config


@dataclass
class ZeROMixedPrecisionConfig:
    """
    ZeRO混合精度配置
    
    控制FP16/BF16训练。
    """
    # 精度选择
    enabled: bool = True
    fp16: bool = True
    bf16: bool = False
    
    # FP16配置
    loss_scale: float = 0  # 0表示动态loss scale
    initial_scale_power: int = 16
    loss_scale_window: int = 1000
    hysteresis: int = 2
    min_loss_scale: float = 1.0
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if self.fp16 and self.bf16:
            errors.append("Cannot enable both FP16 and BF16")
        
        if self.bf16:
            if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
                errors.append("BF16 not supported on this GPU")
        
        return errors
    
    def to_deepspeed_dict(self) -> Dict[str, Any]:
        """转换为DeepSpeed配置"""
        if not self.enabled:
            return {}
        
        if self.bf16:
            return {"bf16": {"enabled": True}}
        
        if self.fp16:
            return {
                "fp16": {
                    "enabled": True,
                    "loss_scale": self.loss_scale,
                    "initial_scale_power": self.initial_scale_power,
                    "loss_scale_window": self.loss_scale_window,
                    "hysteresis": self.hysteresis,
                    "min_loss_scale": self.min_loss_scale,
                }
            }
        
        return {}


@dataclass
class ZeROConfig:
    """
    ZeRO完整配置
    
    DeepSpeed ZeRO优化的完整配置。
    """
    # ZeRO阶段
    stage: ZeROStage = ZeROStage.STAGE_2
    
    # 优化器配置
    optimizer_type: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    
    # 子配置
    offload: ZeROOffloadConfig = field(default_factory=ZeROOffloadConfig)
    communication: ZeROCommunicationConfig = field(default_factory=ZeROCommunicationConfig)
    mixed_precision: ZeROMixedPrecisionConfig = field(default_factory=ZeROMixedPrecisionConfig)
    
    # 梯度相关
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    
    # 激活检查点
    activation_checkpointing: bool = False
    partition_activations: bool = False
    cpu_checkpointing: bool = False
    contiguous_checkpointing: bool = True
    
    # 批量大小（auto表示自动计算）
    train_batch_size: Union[int, str] = "auto"
    train_micro_batch_size_per_gpu: Union[int, str] = "auto"
    
    # 兼容性字段
    offload_optimizer: bool = False
    offload_param: bool = False
    offload_device: str = "cpu"
    nvme_path: str = "/local_nvme"
    fp16_enabled: bool = True
    bf16_enabled: bool = False
    fp16_opt_level: str = "O2"
    loss_scale: float = 0
    loss_scale_window: int = 1000
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    reduce_bucket_size: int = 500_000_000
    allgather_bucket_size: int = 500_000_000
    stage3_max_live_parameters: int = 1_000_000_000
    stage3_max_reuse_distance: int = 1_000_000_000
    stage3_prefetch_bucket_size: int = 500_000_000
    stage3_param_persistence_threshold: int = 100_000
    
    def __post_init__(self):
        """初始化后处理"""
        # 类型转换
        if isinstance(self.stage, int):
            self.stage = ZeROStage.from_int(self.stage)
        
        # 同步兼容性字段到子配置
        self._sync_compat_fields()
    
    def _sync_compat_fields(self):
        """同步兼容性字段"""
        # Offload
        self.offload.offload_optimizer = self.offload_optimizer
        self.offload.offload_param = self.offload_param
        if self.offload_device == "nvme":
            self.offload.optimizer_device = OffloadDevice.NVME
            self.offload.param_device = OffloadDevice.NVME
        self.offload.nvme_path = self.nvme_path
        
        # Communication
        self.communication.overlap_comm = self.overlap_comm
        self.communication.contiguous_gradients = self.contiguous_gradients
        self.communication.reduce_bucket_size = self.reduce_bucket_size
        self.communication.allgather_bucket_size = self.allgather_bucket_size
        self.communication.stage3_max_live_parameters = self.stage3_max_live_parameters
        self.communication.stage3_max_reuse_distance = self.stage3_max_reuse_distance
        self.communication.stage3_prefetch_bucket_size = self.stage3_prefetch_bucket_size
        self.communication.stage3_param_persistence_threshold = self.stage3_param_persistence_threshold
        
        # Mixed Precision
        self.mixed_precision.fp16 = self.fp16_enabled
        self.mixed_precision.bf16 = self.bf16_enabled
        self.mixed_precision.loss_scale = self.loss_scale
        self.mixed_precision.loss_scale_window = self.loss_scale_window
    
    def validate(self) -> Tuple[List[str], List[str]]:
        """
        验证配置
        
        Returns:
            (errors, warnings)
        """
        errors = []
        warnings = []
        
        # 验证stage
        if not isinstance(self.stage, ZeROStage):
            errors.append(f"Invalid stage type: {type(self.stage)}")
        
        # 验证子配置
        warnings.extend(self.offload.validate())
        mp_errors = self.mixed_precision.validate()
        errors.extend(mp_errors)
        
        # 验证Stage 3特定配置
        if self.stage == ZeROStage.STAGE_3:
            if self.offload_param and not self.offload_optimizer:
                warnings.append("offload_param without offload_optimizer may cause issues in Stage 3")
        
        # 验证梯度累积
        if self.gradient_accumulation_steps < 1:
            errors.append(f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}")
        
        return errors, warnings
    
    def to_deepspeed_config(self) -> Dict[str, Any]:
        """转换为DeepSpeed配置字典"""
        stage_value = self.stage.value if isinstance(self.stage, ZeROStage) else self.stage
        
        config = {
            "train_batch_size": self.train_batch_size,
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_clipping": self.gradient_clipping,
            
            "optimizer": {
                "type": self.optimizer_type,
                "params": {
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "betas": list(self.betas),
                    "eps": self.eps
                }
            },
            
            "zero_optimization": {
                "stage": stage_value,
            }
        }
        
        # 合并通信配置
        config["zero_optimization"].update(
            self.communication.to_deepspeed_dict(stage_value)
        )
        
        # 合并卸载配置
        config["zero_optimization"].update(
            self.offload.to_deepspeed_dict(stage_value)
        )
        
        # 合并混合精度配置
        config.update(self.mixed_precision.to_deepspeed_dict())
        
        # 激活检查点
        if self.activation_checkpointing:
            config["activation_checkpointing"] = {
                "partition_activations": self.partition_activations,
                "cpu_checkpointing": self.cpu_checkpointing,
                "contiguous_memory_optimization": self.contiguous_checkpointing,
                "synchronize_checkpoint_boundary": False
            }
        
        return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为普通字典"""
        return {
            'stage': self.stage.value if isinstance(self.stage, ZeROStage) else self.stage,
            'optimizer_type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'betas': list(self.betas),
            'eps': self.eps,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'gradient_clipping': self.gradient_clipping,
            'offload_optimizer': self.offload_optimizer,
            'offload_param': self.offload_param,
            'fp16_enabled': self.fp16_enabled,
            'bf16_enabled': self.bf16_enabled,
            'activation_checkpointing': self.activation_checkpointing,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ZeROConfig':
        """从字典创建配置"""
        if 'stage' in data and isinstance(data['stage'], int):
            data['stage'] = ZeROStage.from_int(data['stage'])
        return cls(**data)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ZeROConfig':
        """从JSON字符串创建"""
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: str) -> None:
        """保存配置到文件"""
        with open(path, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, path: str) -> 'ZeROConfig':
        """从文件加载配置"""
        with open(path, 'r') as f:
            return cls.from_json(f.read())
    
    def summary(self) -> str:
        """生成配置摘要"""
        lines = [
            f"=== ZeRO Configuration ===",
            f"Stage: {self.stage.value} ({self.stage.description})",
            f"Optimizer: {self.optimizer_type} (lr={self.learning_rate})",
            f"Gradient Accumulation: {self.gradient_accumulation_steps}",
            f"Gradient Clipping: {self.gradient_clipping}",
            f"Offload Optimizer: {self.offload_optimizer}",
            f"Offload Param: {self.offload_param}",
            f"Mixed Precision: FP16={self.fp16_enabled}, BF16={self.bf16_enabled}",
            f"Activation Checkpointing: {self.activation_checkpointing}",
        ]
        return '\n'.join(lines)


# ==================== 内存监控 ====================

class ZeROMemoryMonitor:
    """
    ZeRO内存监控器
    
    监控ZeRO训练过程中的内存使用。
    """
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._history: List[Dict[str, float]] = []
        self._peak_memory: float = 0.0
    
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
        """获取峰值内存"""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(self.device_id) / (1024**3)
        return 0.0
    
    def reset_peak_memory(self) -> None:
        """重置峰值内存统计"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
    
    def record(self, tag: str = "") -> Dict[str, float]:
        """记录内存状态"""
        stats = self.get_memory_stats()
        stats['tag'] = tag
        stats['timestamp'] = time.time()
        self._history.append(stats)
        
        peak = self.get_peak_memory()
        if peak > self._peak_memory:
            self._peak_memory = peak
        
        return stats
    
    def get_history(self) -> List[Dict[str, float]]:
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

class ZeROProfiler:
    """
    ZeRO性能分析器
    
    分析ZeRO训练的性能瓶颈。
    """
    
    def __init__(self):
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._enabled = False
        self._step_count = 0
    
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
        """记录时间"""
        if self._enabled:
            self._timings[name].append(duration)
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """获取特定区域的统计"""
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
        return {name: self.get_stats(name) for name in self._timings}
    
    def reset(self) -> None:
        """重置统计"""
        self._timings.clear()
        self._step_count = 0
    
    def print_summary(self) -> None:
        """打印摘要"""
        print(f"\n=== ZeRO Profiler Summary ({self._step_count} steps) ===")
        
        stats = self.get_all_stats()
        if not stats:
            print("No profiling data")
            return
        
        total_time = sum(s.get('total_ms', 0) for s in stats.values())
        
        print(f"{'Region':<25} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'%':>8}")
        print("-" * 65)
        
        sorted_stats = sorted(stats.items(), key=lambda x: x[1].get('total_ms', 0), reverse=True)
        
        for name, stat in sorted_stats:
            pct = stat['total_ms'] / total_time * 100 if total_time > 0 else 0
            print(f"{name:<25} {stat['count']:>8} {stat['total_ms']:>12.2f} {stat['avg_ms']:>10.2f} {pct:>7.1f}%")


# ==================== 训练状态 ====================

@dataclass
class ZeROTrainingState:
    """
    ZeRO训练状态
    
    跟踪训练进度和统计。
    """
    # 进度
    global_step: int = 0
    epoch: int = 0
    samples_seen: int = 0
    
    # 损失
    loss_history: List[float] = field(default_factory=list)
    avg_loss: float = 0.0
    
    # 学习率
    current_lr: float = 0.0
    
    # 梯度
    grad_norm_history: List[float] = field(default_factory=list)
    avg_grad_norm: float = 0.0
    
    # 时间
    total_training_time: float = 0.0
    step_times: List[float] = field(default_factory=list)
    
    def update_loss(self, loss: float) -> None:
        """更新损失"""
        self.loss_history.append(loss)
        # 滑动平均
        window = min(100, len(self.loss_history))
        self.avg_loss = sum(self.loss_history[-window:]) / window
    
    def update_grad_norm(self, grad_norm: float) -> None:
        """更新梯度范数"""
        self.grad_norm_history.append(grad_norm)
        window = min(100, len(self.grad_norm_history))
        self.avg_grad_norm = sum(self.grad_norm_history[-window:]) / window
    
    def update_step_time(self, duration: float) -> None:
        """更新步骤时间"""
        self.step_times.append(duration)
        self.total_training_time += duration
    
    def get_throughput(self, batch_size: int) -> float:
        """获取吞吐量（samples/sec）"""
        if not self.step_times:
            return 0.0
        avg_step_time = sum(self.step_times) / len(self.step_times)
        if avg_step_time == 0:
            return 0.0
        return batch_size / avg_step_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'samples_seen': self.samples_seen,
            'avg_loss': self.avg_loss,
            'current_lr': self.current_lr,
            'avg_grad_norm': self.avg_grad_norm,
            'total_training_time': self.total_training_time,
        }


# ==================== ZeRO包装器 ====================

class ZeROWrapper:
    """
    ZeRO模型包装器
    
    封装DeepSpeed ZeRO的初始化和管理，提供生产级功能。
    """
    
    def __init__(self, config: Optional[ZeROConfig] = None):
        self.config = config or ZeROConfig()
        
        # 模型和优化器
        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._lr_scheduler: Optional[Any] = None
        self._engine: Optional[Any] = None  # DeepSpeed engine
        
        # 组件
        self._memory_monitor: ZeROMemoryMonitor = ZeROMemoryMonitor()
        self._profiler: ZeROProfiler = ZeROProfiler()
        self._training_state: ZeROTrainingState = ZeROTrainingState()
        
        # 状态
        self._is_wrapped: bool = False
        self._gradient_accumulation_count: int = 0
    
    # ==================== 包装和初始化 ====================
    
    def wrap(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
        model_parameters: Optional[Iterator] = None
    ) -> Tuple[Any, Any, Any, Any]:
        """
        使用DeepSpeed包装模型
        
        Args:
            model: PyTorch模型
            optimizer: 优化器（可选）
            lr_scheduler: 学习率调度器
            model_parameters: 模型参数迭代器
            
        Returns:
            (engine, optimizer, model, lr_scheduler)
        """
        if not DEEPSPEED_AVAILABLE:
            logger.warning("DeepSpeed not available, returning original model")
            self._model = model
            self._optimizer = optimizer
            self._lr_scheduler = lr_scheduler
            return model, optimizer, model, lr_scheduler
        
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        
        # 验证配置
        errors, warnings = self.config.validate()
        if errors:
            raise ValueError(f"Invalid ZeRO config: {errors}")
        for warning in warnings:
            logger.warning(f"ZeRO config warning: {warning}")
        
        # 记录包装前内存
        self._memory_monitor.record("pre_wrap")
        
        # 分析模型
        model_info = self._analyze_model(model)
        logger.info(f"Wrapping model with ZeRO: {model_info['num_params']:,} params, "
                   f"stage={self.config.stage.value}")
        
        # 转换配置
        ds_config = self.config.to_deepspeed_config()
        
        # 初始化DeepSpeed
        self._engine, self._optimizer, _, self._lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=ds_config,
            model_parameters=model_parameters
        )
        
        self._is_wrapped = True
        
        # 记录包装后内存
        self._memory_monitor.record("post_wrap")
        
        logger.info(f"Model wrapped with DeepSpeed ZeRO Stage {self.config.stage.value}")
        
        return self._engine, self._optimizer, self._engine, self._lr_scheduler
    
    def _analyze_model(self, model: nn.Module) -> Dict[str, Any]:
        """分析模型"""
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'num_params': num_params,
            'num_trainable': num_trainable,
            'num_frozen': num_params - num_trainable,
            'param_memory_mb': num_params * 4 / (1024**2),  # fp32
        }
    
    # ==================== 模型访问 ====================
    
    @property
    def module(self) -> nn.Module:
        """获取内部模型"""
        if self._engine is not None:
            return self._engine.module
        return self._model
    
    @property
    def engine(self) -> Optional[Any]:
        """获取DeepSpeed engine"""
        return self._engine
    
    @property
    def optimizer(self) -> Optional[torch.optim.Optimizer]:
        """获取优化器"""
        return self._optimizer
    
    @property
    def lr_scheduler(self) -> Optional[Any]:
        """获取学习率调度器"""
        return self._lr_scheduler
    
    @property
    def is_wrapped(self) -> bool:
        """是否已包装"""
        return self._is_wrapped
    
    # ==================== 训练步骤 ====================
    
    def forward(self, *args, **kwargs) -> Any:
        """前向传播"""
        with self._profiler.profile_region("forward"):
            if self._engine is not None:
                return self._engine(*args, **kwargs)
        return self._model(*args, **kwargs)
    
    def backward(self, loss: torch.Tensor) -> None:
        """反向传播"""
        with self._profiler.profile_region("backward"):
            if self._engine is not None:
                self._engine.backward(loss)
            else:
                loss.backward()
        
        # 更新训练状态
        self._training_state.update_loss(loss.item())
    
    def step(self) -> None:
        """优化器步进"""
        with self._profiler.profile_region("optimizer_step"):
            if self._engine is not None:
                self._engine.step()
            elif self._optimizer is not None:
                self._optimizer.step()
        
        self._training_state.global_step += 1
        self._gradient_accumulation_count = 0
    
    def zero_grad(self) -> None:
        """清零梯度"""
        if self._engine is not None:
            # DeepSpeed自动处理
            pass
        elif self._optimizer is not None:
            self._optimizer.zero_grad()
    
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
            loss_fn: 损失函数，接受模型输出返回损失
            accumulate: 是否进行梯度累积
            
        Returns:
            损失值
        """
        self._profiler.start_step()
        step_start = time.perf_counter()
        
        # 前向传播
        output = self.forward(batch)
        
        # 计算损失
        loss = loss_fn(output)
        
        # 缩放损失（梯度累积）
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        
        # 反向传播
        self.backward(loss)
        
        self._gradient_accumulation_count += 1
        
        # 优化器步进
        if not accumulate or self._gradient_accumulation_count >= self.config.gradient_accumulation_steps:
            # 梯度裁剪
            grad_norm = self.clip_grad_norm()
            self._training_state.update_grad_norm(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
            
            self.step()
            self.zero_grad()
        
        # 更新时间统计
        step_time = time.perf_counter() - step_start
        self._training_state.update_step_time(step_time)
        
        return loss
    
    # ==================== 梯度控制 ====================
    
    def clip_grad_norm(self, max_norm: Optional[float] = None) -> float:
        """
        梯度裁剪
        
        Args:
            max_norm: 最大梯度范数，None使用配置值
            
        Returns:
            裁剪前的梯度范数
        """
        max_norm = max_norm or self.config.gradient_clipping
        
        if self._engine is not None:
            # DeepSpeed自动处理梯度裁剪
            return 0.0
        
        if self._model is not None:
            return torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                max_norm
            ).item()
        
        return 0.0
    
    def get_grad_norm(self) -> float:
        """获取当前梯度范数"""
        if self._model is None:
            return 0.0
        
        total_norm = 0.0
        for p in self._model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        
        return total_norm ** 0.5
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        缩放损失（用于混合精度）
        
        Args:
            loss: 原始损失
            
        Returns:
            缩放后的损失
        """
        if self._engine is not None and hasattr(self._engine, 'scale_loss'):
            return self._engine.scale_loss(loss)
        return loss
    
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
        self._training_state.current_lr = lr
    
    def step_scheduler(self, metrics: Optional[float] = None) -> None:
        """步进学习率调度器"""
        if self._lr_scheduler is not None:
            if metrics is not None and hasattr(self._lr_scheduler, 'step'):
                # ReduceLROnPlateau类型
                try:
                    self._lr_scheduler.step(metrics)
                except TypeError:
                    self._lr_scheduler.step()
            else:
                self._lr_scheduler.step()
        
        self._training_state.current_lr = self.get_lr()
    
    # ==================== 检查点管理 ====================
    
    def save_checkpoint(
        self,
        save_dir: str,
        tag: Optional[str] = None,
        client_state: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """
        保存检查点
        
        Args:
            save_dir: 保存目录
            tag: 检查点标签
            client_state: 客户端状态
            **kwargs: 额外数据
            
        Returns:
            保存路径
        """
        # 准备客户端状态
        state = client_state or {}
        state.update({
            'training_state': self._training_state.to_dict(),
            'config': self.config.to_dict(),
            **kwargs
        })
        
        if self._engine is not None:
            self._engine.save_checkpoint(save_dir, tag=tag, client_state=state)
            logger.info(f"ZeRO checkpoint saved: {save_dir}/{tag or 'latest'}")
            return os.path.join(save_dir, tag or 'latest')
        else:
            # 普通保存
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{tag or 'checkpoint'}.pt")
            
            checkpoint = {
                'model_state_dict': self._model.state_dict() if self._model else {},
                'optimizer_state_dict': self._optimizer.state_dict() if self._optimizer else {},
                **state
            }
            
            if self._lr_scheduler is not None:
                checkpoint['scheduler_state_dict'] = self._lr_scheduler.state_dict()
            
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved: {path}")
            return path
    
    def load_checkpoint(
        self,
        load_dir: str,
        tag: Optional[str] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_module_only: bool = False
    ) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            load_dir: 检查点目录
            tag: 检查点标签
            load_optimizer: 是否加载优化器状态
            load_scheduler: 是否加载调度器状态
            load_module_only: 是否只加载模块权重
            
        Returns:
            客户端状态
        """
        if self._engine is not None:
            _, client_state = self._engine.load_checkpoint(
                load_dir,
                tag=tag,
                load_optimizer_states=load_optimizer and not load_module_only,
                load_lr_scheduler_states=load_scheduler and not load_module_only,
                load_module_only=load_module_only
            )
            
            # 恢复训练状态
            if client_state and 'training_state' in client_state:
                ts = client_state['training_state']
                self._training_state.global_step = ts.get('global_step', 0)
                self._training_state.epoch = ts.get('epoch', 0)
                self._training_state.samples_seen = ts.get('samples_seen', 0)
            
            logger.info(f"ZeRO checkpoint loaded: {load_dir}/{tag or 'latest'}")
            return client_state or {}
        else:
            path = os.path.join(load_dir, f"{tag or 'checkpoint'}.pt")
            checkpoint = torch.load(path, map_location='cpu')
            
            if self._model and 'model_state_dict' in checkpoint:
                self._model.load_state_dict(checkpoint['model_state_dict'])
            
            if load_optimizer and self._optimizer and 'optimizer_state_dict' in checkpoint:
                self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if load_scheduler and self._lr_scheduler and 'scheduler_state_dict' in checkpoint:
                self._lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # 恢复训练状态
            if 'training_state' in checkpoint:
                ts = checkpoint['training_state']
                self._training_state.global_step = ts.get('global_step', 0)
                self._training_state.epoch = ts.get('epoch', 0)
            
            logger.info(f"Checkpoint loaded: {path}")
            return checkpoint
    
    def get_checkpoint_path(self, save_dir: str, tag: Optional[str] = None) -> str:
        """获取检查点路径"""
        if self._engine is not None:
            return os.path.join(save_dir, tag or 'latest')
        return os.path.join(save_dir, f"{tag or 'checkpoint'}.pt")
    
    # ==================== 内存管理 ====================
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计"""
        return self._memory_monitor.get_memory_stats()
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """获取内存摘要"""
        return self._memory_monitor.get_summary()
    
    def estimate_memory(self) -> Dict[str, float]:
        """
        估算内存需求
        
        Returns:
            各类内存需求估算
        """
        if self._model is None:
            return {}
        
        num_params = sum(p.numel() for p in self._model.parameters())
        stage_value = self.config.stage.value
        
        estimates = {
            'num_params': num_params,
            'num_params_billions': num_params / 1e9,
        }
        
        # 基础内存估算
        param_memory = num_params * 2 / (1024**3)  # fp16参数
        grad_memory = num_params * 2 / (1024**3)   # fp16梯度
        optimizer_memory = num_params * 8 / (1024**3)  # Adam状态（2个fp32）
        
        # 根据Stage计算
        if stage_value == 0:
            estimates['param_memory_gb'] = param_memory
            estimates['grad_memory_gb'] = grad_memory
            estimates['optimizer_memory_gb'] = optimizer_memory
        elif stage_value == 1:
            # Stage 1: 分片优化器状态
            estimates['param_memory_gb'] = param_memory
            estimates['grad_memory_gb'] = grad_memory
            estimates['optimizer_memory_gb'] = optimizer_memory  # 假设单GPU估算
        elif stage_value == 2:
            # Stage 2: 分片优化器 + 梯度
            estimates['param_memory_gb'] = param_memory
            estimates['grad_memory_gb'] = grad_memory
            estimates['optimizer_memory_gb'] = optimizer_memory
        else:
            # Stage 3: 全部分片
            estimates['param_memory_gb'] = param_memory
            estimates['grad_memory_gb'] = grad_memory
            estimates['optimizer_memory_gb'] = optimizer_memory
        
        estimates['total_memory_gb'] = (
            estimates.get('param_memory_gb', 0) +
            estimates.get('grad_memory_gb', 0) +
            estimates.get('optimizer_memory_gb', 0)
        )
        
        # DeepSpeed估算（如果可用）
        # 注意：这些函数只打印估算信息，不返回值
        if DEEPSPEED_AVAILABLE and self._model is not None:
            try:
                if stage_value == 2:
                    estimate_zero2_model_states_mem_needs_all_live(
                        self._model,
                        num_gpus_per_node=1,
                        num_nodes=1
                    )
                    estimates['deepspeed_estimated'] = True
                if stage_value == 3:
                    estimate_zero3_model_states_mem_needs_all_live(
                        self._model,
                        num_gpus_per_node=1,
                        num_nodes=1
                    )
                    estimates['deepspeed_estimated'] = True
            except Exception as e:
                logger.debug(f"DeepSpeed memory estimation failed: {e}")
        
        return estimates
    
    def clear_memory_cache(self) -> None:
        """清理内存缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self._memory_monitor.record("cache_cleared")
    
    def cleanup(self) -> None:
        """清理ZeRO资源"""
        # 清理内存缓存
        self.clear_memory_cache()
        
        # 重置引擎和状态
        self._engine = None
        self._optimizer = None
        self._scheduler = None
        self._model = None
        
        logger.info("ZeRO cleaned up")
    
    @contextmanager
    def track_memory(self, tag: str):
        """内存追踪上下文"""
        with self._memory_monitor.track(tag):
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
    
    # ==================== 训练状态 ====================
    
    def get_training_state(self) -> Dict[str, Any]:
        """获取训练状态"""
        return self._training_state.to_dict()
    
    def get_throughput(self, batch_size: int) -> float:
        """获取吞吐量"""
        return self._training_state.get_throughput(batch_size)
    
    def set_epoch(self, epoch: int) -> None:
        """设置当前epoch"""
        self._training_state.epoch = epoch
    
    def increment_samples(self, count: int) -> None:
        """增加处理的样本数"""
        self._training_state.samples_seen += count
    
    # ==================== 诊断 ====================
    
    def diagnose(self) -> Dict[str, Any]:
        """
        运行诊断
        
        Returns:
            诊断结果
        """
        diagnosis = {
            'deepspeed_available': DEEPSPEED_AVAILABLE,
            'is_wrapped': self._is_wrapped,
            'config': {
                'stage': self.config.stage.value,
                'offload_optimizer': self.config.offload_optimizer,
                'offload_param': self.config.offload_param,
                'fp16': self.config.fp16_enabled,
                'bf16': self.config.bf16_enabled,
                'gradient_accumulation': self.config.gradient_accumulation_steps,
            },
            'memory': self.get_memory_stats(),
            'memory_estimate': self.estimate_memory(),
            'training_state': self.get_training_state(),
        }
        
        # 检查问题
        issues = []
        
        if not DEEPSPEED_AVAILABLE:
            issues.append("DeepSpeed not available - install with: pip install deepspeed")
        
        if not self._is_wrapped:
            issues.append("Model not wrapped - call wrap() first")
        
        memory_stats = self.get_memory_stats()
        if memory_stats.get('utilization', 0) > 0.9:
            issues.append("High memory utilization (>90%)")
        
        if self.config.stage == ZeROStage.STAGE_3 and not self.config.offload_optimizer:
            if memory_stats.get('utilization', 0) > 0.7:
                issues.append("Consider enabling offload_optimizer for Stage 3")
        
        diagnosis['issues'] = issues
        
        # 建议
        suggestions = []
        
        if self.config.stage.value < 3 and memory_stats.get('utilization', 0) > 0.8:
            suggestions.append(f"Consider upgrading to ZeRO Stage {self.config.stage.value + 1}")
        
        if not self.config.activation_checkpointing and memory_stats.get('utilization', 0) > 0.7:
            suggestions.append("Consider enabling activation_checkpointing")
        
        diagnosis['suggestions'] = suggestions
        
        return diagnosis
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diagnosis = self.diagnose()
        
        print("\n=== ZeRO Wrapper Diagnosis ===")
        print(f"DeepSpeed Available: {diagnosis['deepspeed_available']}")
        print(f"Model Wrapped: {diagnosis['is_wrapped']}")
        
        print("\nConfiguration:")
        for key, value in diagnosis['config'].items():
            print(f"  {key}: {value}")
        
        print("\nMemory:")
        for key, value in diagnosis['memory'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        print("\nTraining State:")
        for key, value in diagnosis['training_state'].items():
            print(f"  {key}: {value}")
        
        if diagnosis['issues']:
            print("\nIssues:")
            for issue in diagnosis['issues']:
                print(f"  ⚠ {issue}")
        
        if diagnosis['suggestions']:
            print("\nSuggestions:")
            for suggestion in diagnosis['suggestions']:
                print(f"  → {suggestion}")
    
    # ==================== 静态方法 ====================
    
    @staticmethod
    def get_recommended_config(
        model_size_billions: float,
        num_gpus: int,
        gpu_memory_gb: float = 80
    ) -> ZeROConfig:
        """
        获取推荐配置
        
        Args:
            model_size_billions: 模型参数量（B）
            num_gpus: GPU数量
            gpu_memory_gb: 单卡内存（GB）
            
        Returns:
            推荐的ZeRO配置
        """
        model_params = int(model_size_billions * 1e9)
        stage = ZeROStage.recommend(model_params, num_gpus, gpu_memory_gb)
        
        # 估算内存
        memory_per_param = 18
        total_memory_gb = model_size_billions * 1e9 * memory_per_param / (1024**3)
        memory_per_gpu = total_memory_gb / num_gpus
        
        config = ZeROConfig(stage=stage)
            
        # 配置卸载
        if memory_per_gpu > gpu_memory_gb:
            config.offload_optimizer = True
            if memory_per_gpu > gpu_memory_gb * 1.5:
                config.offload_param = True
        
        # 大模型启用激活检查点
        if model_size_billions > 1.0:
            config.activation_checkpointing = True
        
        # 选择混合精度
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            config.fp16_enabled = False
            config.bf16_enabled = True
        
        logger.info(f"Recommended ZeRO config: stage={config.stage.value}, "
                   f"offload_optimizer={config.offload_optimizer}, "
                   f"offload_param={config.offload_param}")
        
        return config
    
    @staticmethod
    def is_available() -> bool:
        """检查DeepSpeed是否可用"""
        return DEEPSPEED_AVAILABLE


# ==================== 便捷函数 ====================

def create_zero_optimizer(
    model: nn.Module,
    stage: int = 2,
    learning_rate: float = 1e-4,
    **kwargs
) -> Tuple[ZeROWrapper, Any]:
    """
    创建ZeRO优化器
    
    Args:
        model: 模型
        stage: ZeRO阶段
        learning_rate: 学习率
        **kwargs: 其他配置
        
    Returns:
        (ZeROWrapper实例, DeepSpeed engine)
    """
    config = ZeROConfig(
        stage=ZeROStage.from_int(stage),
        learning_rate=learning_rate,
        **kwargs
    )
    
    wrapper = ZeROWrapper(config)
    engine, optimizer, _, scheduler = wrapper.wrap(model)
    
    return wrapper, engine


@contextmanager
def zero_context(config: Optional[ZeROConfig] = None):
    """
    ZeRO上下文管理器
    
    Args:
        config: ZeRO配置
        
    Yields:
        ZeROWrapper实例
    """
    wrapper = ZeROWrapper(config or ZeROConfig())
    try:
        yield wrapper
    finally:
        wrapper.clear_memory_cache()


def auto_configure_zero(
    model: nn.Module,
    num_gpus: int = 1,
    gpu_memory_gb: float = 80
) -> ZeROConfig:
    """
    自动配置ZeRO
    
    根据模型大小和可用资源自动选择最佳配置。
    
    Args:
        model: 模型
        num_gpus: GPU数量
        gpu_memory_gb: 单卡内存
        
    Returns:
        推荐的ZeRO配置
    """
    num_params = sum(p.numel() for p in model.parameters())
    model_size_billions = num_params / 1e9
    
    return ZeROWrapper.get_recommended_config(
        model_size_billions,
        num_gpus,
        gpu_memory_gb
    )


def estimate_zero_memory(
    model: nn.Module,
    stage: int,
    num_gpus: int = 1,
    dtype_bytes: int = 2
) -> Dict[str, float]:
    """
    估算ZeRO内存使用
    
    Args:
        model: 模型
        stage: ZeRO阶段
        num_gpus: GPU数量
        dtype_bytes: 数据类型字节数
        
    Returns:
        内存估算（GB）
    """
    num_params = sum(p.numel() for p in model.parameters())
    
    # 参数内存
    param_memory = num_params * dtype_bytes
    
    # 梯度内存
    grad_memory = num_params * dtype_bytes
    
    # 优化器状态（Adam: 2倍fp32）
    optimizer_memory = num_params * 4 * 2
    
    # 根据Stage和GPU数计算
    if stage == 0:
        total_per_gpu = param_memory + grad_memory + optimizer_memory
    elif stage == 1:
        total_per_gpu = param_memory + grad_memory + optimizer_memory / num_gpus
    elif stage == 2:
        total_per_gpu = param_memory + (grad_memory + optimizer_memory) / num_gpus
    else:  # stage 3
        total_per_gpu = (param_memory + grad_memory + optimizer_memory) / num_gpus
    
    return {
        'params_gb': param_memory / (1024**3),
        'grads_gb': grad_memory / (1024**3),
        'optimizer_gb': optimizer_memory / (1024**3),
        'total_per_gpu_gb': total_per_gpu / (1024**3),
        'total_model_gb': (param_memory + grad_memory + optimizer_memory) / (1024**3),
        'stage': stage,
        'num_gpus': num_gpus,
    }


def get_zero_stage_description(stage: int) -> str:
    """
    获取ZeRO阶段描述
    
    Args:
        stage: ZeRO阶段
        
    Returns:
        阶段描述
    """
    try:
        return ZeROStage.from_int(stage).description
    except ValueError:
        return "Unknown stage"


def compare_zero_stages(
    model: nn.Module,
    num_gpus: int = 1
) -> Dict[str, Dict[str, float]]:
    """
    比较不同ZeRO阶段的内存使用
    
    Args:
        model: 模型
        num_gpus: GPU数量
        
    Returns:
        各阶段的内存估算
    """
    results = {}
    
    for stage in range(4):
        estimates = estimate_zero_memory(model, stage, num_gpus)
        results[f'stage_{stage}'] = estimates
    
    return results


def print_zero_comparison(model: nn.Module, num_gpus: int = 1) -> None:
    """
    打印ZeRO阶段比较
    
    Args:
        model: 模型
        num_gpus: GPU数量
    """
    comparison = compare_zero_stages(model, num_gpus)
    
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"\n=== ZeRO Stage Comparison ({num_params:,} params, {num_gpus} GPUs) ===")
    print(f"{'Stage':<10} {'Description':<35} {'Memory/GPU (GB)':>15}")
    print("-" * 62)
    
    for stage in range(4):
        key = f'stage_{stage}'
        memory = comparison[key]['total_per_gpu_gb']
        desc = get_zero_stage_description(stage)
        print(f"Stage {stage:<4} {desc:<35} {memory:>15.2f}")

# -*- coding: utf-8 -*-
"""
分布式并行模式定义

定义各种分布式并行策略的配置和枚举类型，提供生产级的配置管理、验证和自动优化。
"""

import os
import logging
import json
import math
from typing import Dict, Any, Optional, List, Tuple, Union, Type
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from abc import ABC, abstractmethod
from copy import deepcopy

logger = logging.getLogger(__name__)


class ParallelMode(Enum):
    """分布式并行模式"""
    # 单机模式
    NONE = "none"
    
    # 数据并行
    DDP = "ddp"                    # Data Distributed Parallel
    FSDP = "fsdp"                  # Fully Sharded Data Parallel
    
    # 模型并行
    TENSOR = "tensor"              # Tensor Parallel
    PIPELINE = "pipeline"          # Pipeline Parallel
    
    # DeepSpeed优化
    ZERO_1 = "zero_1"              # ZeRO Stage 1
    ZERO_2 = "zero_2"              # ZeRO Stage 2
    ZERO_3 = "zero_3"              # ZeRO Stage 3
    
    # 混合并行
    HYBRID = "hybrid"              # 混合并行（DDP + Pipeline/Tensor）
    MEGATRON = "megatron"          # Megatron-LM风格

    @classmethod
    def from_string(cls, value: str) -> 'ParallelMode':
        """从字符串安全转换"""
        try:
            return cls(value.lower())
        except ValueError:
            logger.warning(f"Unknown parallel mode '{value}', falling back to NONE")
            return cls.NONE
    
    @property
    def is_data_parallel(self) -> bool:
        """是否是数据并行模式"""
        return self in (self.DDP, self.FSDP)
    
    @property
    def is_model_parallel(self) -> bool:
        """是否是模型并行模式"""
        return self in (self.TENSOR, self.PIPELINE)
    
    @property
    def is_zero(self) -> bool:
        """是否是ZeRO模式"""
        return self in (self.ZERO_1, self.ZERO_2, self.ZERO_3)
    
    @property
    def requires_deepspeed(self) -> bool:
        """是否需要DeepSpeed"""
        return self.is_zero


class CommunicationBackend(Enum):
    """通信后端"""
    NCCL = "nccl"                  # NVIDIA GPU
    GLOO = "gloo"                  # CPU或跨平台
    MPI = "mpi"                    # MPI

    @classmethod
    def auto_select(cls, use_cuda: bool = True) -> 'CommunicationBackend':
        """
        自动选择通信后端
        
        Args:
            use_cuda: 是否使用CUDA
            
        Returns:
            推荐的通信后端
        """
        if use_cuda:
            return cls.NCCL
        return cls.GLOO
    
    def is_available(self) -> bool:
        """检查后端是否可用"""
        try:
            import torch.distributed as dist
            return dist.is_backend_available(self.value)
        except Exception:
            return False


class ShardingStrategy(Enum):
    """FSDP分片策略"""
    FULL_SHARD = "full_shard"      # 完全分片
    SHARD_GRAD_OP = "shard_grad_op"  # 只分片梯度和优化器状态
    NO_SHARD = "no_shard"          # 不分片（类似DDP）
    HYBRID_SHARD = "hybrid_shard"  # 混合分片

    @property
    def memory_efficiency(self) -> float:
        """
        内存效率系数（相对于DDP）
        
        FULL_SHARD最省内存，NO_SHARD等同于DDP
        """
        efficiency_map = {
            self.FULL_SHARD: 1.0,      # 最高效
            self.SHARD_GRAD_OP: 0.6,   # 中等
            self.HYBRID_SHARD: 0.8,    # 较高
            self.NO_SHARD: 0.0         # 无优化
        }
        return efficiency_map.get(self, 0.0)


# ==================== 配置验证异常 ====================

class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


# ==================== 基础配置类 ====================

@dataclass
class DistributedConfig:
    """
    分布式基础配置
    
    所有分布式配置的基类，提供通用的验证、序列化和环境设置功能。
    """
    # 基本设置
    mode: ParallelMode = ParallelMode.DDP
    backend: CommunicationBackend = CommunicationBackend.NCCL
    
    # 进程组配置
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # 网络配置
    master_addr: str = "localhost"
    master_port: str = "12355"
    
    # 超时配置
    timeout_minutes: int = 30
    
    # 调试
    find_unused_parameters: bool = False
    
    def __post_init__(self):
        """初始化后处理：类型转换和验证"""
        self._convert_types()
        self._post_init_hook()
        # 延迟验证，在子类完成初始化后执行
    
    def _convert_types(self):
        """类型转换"""
        if isinstance(self.mode, str):
            self.mode = ParallelMode(self.mode)
        if isinstance(self.backend, str):
            self.backend = CommunicationBackend(self.backend)

    def _post_init_hook(self):
        """子类可覆盖的初始化钩子"""
        pass

    # ==================== 验证方法 ====================
    
    def validate(self) -> List[str]:
        """
        验证配置有效性
        
        Returns:
            验证警告列表（空列表表示验证通过）
            
        Raises:
            ConfigValidationError: 配置无效时抛出
        """
        errors = []
        warnings = []
        
        # 验证world_size
        if self.world_size < 1:
            errors.append(f"world_size must be >= 1, got {self.world_size}")
        
        # 验证rank
        if self.rank < 0 or self.rank >= self.world_size:
            errors.append(f"rank must be in [0, world_size), got rank={self.rank}, world_size={self.world_size}")
        
        # 验证local_rank
        if self.local_rank < 0:
            errors.append(f"local_rank must be >= 0, got {self.local_rank}")
        
        # 验证端口
        try:
            port = int(self.master_port)
            if port < 1024 or port > 65535:
                warnings.append(f"master_port {port} may require elevated privileges or be invalid")
        except ValueError:
            errors.append(f"master_port must be a valid integer, got '{self.master_port}'")
        
        # 验证超时
        if self.timeout_minutes < 1:
            warnings.append(f"timeout_minutes={self.timeout_minutes} is very short")
        
        # 执行子类验证
        sub_errors, sub_warnings = self._validate_specific()
        errors.extend(sub_errors)
        warnings.extend(sub_warnings)
        
        if errors:
            raise ConfigValidationError(f"Configuration validation failed: {'; '.join(errors)}")
        
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
        
        return warnings
    
    def _validate_specific(self) -> Tuple[List[str], List[str]]:
        """子类特定验证，返回 (errors, warnings)"""
        return [], []

    # ==================== 序列化方法 ====================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典（可JSON序列化）
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Enum):
                result[f.name] = value.value
            elif hasattr(value, 'to_dict'):
                result[f.name] = value.to_dict()
            else:
                result[f.name] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedConfig':
        """
        从字典创建配置
        
        Args:
            data: 配置字典
            
        Returns:
            配置实例
        """
        # 过滤掉不存在的字段
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'DistributedConfig':
        """从JSON字符串创建配置"""
        return cls.from_dict(json.loads(json_str))
    
    def save(self, path: str) -> None:
        """保存配置到文件"""
        with open(path, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'DistributedConfig':
        """从文件加载配置"""
        with open(path, 'r') as f:
            return cls.from_json(f.read())

    # ==================== 环境变量方法 ====================
    
    def setup_env(self) -> Dict[str, str]:
        """
        设置分布式训练环境变量
        
        Returns:
            设置的环境变量字典
        """
        env_vars = {
            'MASTER_ADDR': self.master_addr,
            'MASTER_PORT': str(self.master_port),
            'WORLD_SIZE': str(self.world_size),
            'RANK': str(self.rank),
            'LOCAL_RANK': str(self.local_rank),
        }
        
        # 设置环境变量
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # 添加额外的优化环境变量
        extra_env = self._get_extra_env_vars()
        for key, value in extra_env.items():
            os.environ[key] = value
        env_vars.update(extra_env)
        
        logger.debug(f"Environment variables set: {list(env_vars.keys())}")
        return env_vars
    
    def _get_extra_env_vars(self) -> Dict[str, str]:
        """获取额外的环境变量（子类可覆盖）"""
        return {}
    
    @classmethod
    def from_env(cls) -> 'DistributedConfig':
        """
        从环境变量创建配置
        
        Returns:
            配置实例
        """
        return cls(
            world_size=int(os.environ.get('WORLD_SIZE', 1)),
            rank=int(os.environ.get('RANK', 0)),
            local_rank=int(os.environ.get('LOCAL_RANK', 0)),
            master_addr=os.environ.get('MASTER_ADDR', 'localhost'),
            master_port=os.environ.get('MASTER_PORT', '12355'),
        )

    # ==================== 设备和进程管理 ====================
    
    def get_device(self) -> Any:
        """
        获取当前进程对应的设备
        
        Returns:
            torch设备对象 (torch.device)
        """
        import torch
        if torch.cuda.is_available():
            return torch.device(f'cuda:{self.local_rank}')
        return torch.device('cpu')
    
    @property
    def is_main_process(self) -> bool:
        """是否是主进程"""
        return self.rank == 0
    
    @property
    def is_distributed(self) -> bool:
        """是否是分布式模式"""
        return self.world_size > 1

    # ==================== 内存估算 ====================
    
    def estimate_memory_requirements(
        self,
        model_params: int,
        batch_size: int,
        sequence_length: int = 512,
        dtype_bytes: int = 2  # fp16
    ) -> Dict[str, float]:
        """
        估算内存需求（GB）
        
        Args:
            model_params: 模型参数量
            batch_size: 批次大小
            sequence_length: 序列长度
            dtype_bytes: 数据类型字节数
            
        Returns:
            各类内存需求估算
        """
        # 基础内存估算（参数 + 梯度 + 优化器状态）
        param_memory = model_params * dtype_bytes / (1024**3)
        gradient_memory = param_memory
        
        # Adam优化器状态: 2倍参数（momentum + variance）
        optimizer_memory = 2 * model_params * 4 / (1024**3)  # fp32
        
        # 激活内存估算（粗略）
        # 假设每个token需要约1KB激活内存
        activation_memory = batch_size * sequence_length * 1024 / (1024**3)
        
        # 应用分布式策略的内存优化
        memory_factor = self._get_memory_factor()
        
        return {
            'params_gb': param_memory,
            'gradients_gb': gradient_memory * memory_factor,
            'optimizer_gb': optimizer_memory * memory_factor,
            'activation_gb': activation_memory,
            'total_per_gpu_gb': (param_memory + gradient_memory * memory_factor + 
                                  optimizer_memory * memory_factor + activation_memory),
            'memory_factor': memory_factor,
            'model_params': model_params,
        }
    
    def _get_memory_factor(self) -> float:
        """获取内存优化系数（子类可覆盖）"""
        if self.mode == ParallelMode.DDP:
            return 1.0  # DDP不优化内存
        return 1.0

    # ==================== 配置合并与复制 ====================
    
    def merge(self, other: 'DistributedConfig') -> 'DistributedConfig':
        """
        合并另一个配置（other覆盖self中的非默认值）
        
        Args:
            other: 要合并的配置
            
        Returns:
            合并后的新配置
        """
        merged_data = self.to_dict()
        other_data = other.to_dict()
        
        for key, value in other_data.items():
            if value is not None:
                merged_data[key] = value
        
        return self.__class__.from_dict(merged_data)
    
    def copy(self) -> 'DistributedConfig':
        """深拷贝配置"""
        return deepcopy(self)

    # ==================== 字符串表示 ====================
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mode={self.mode.value}, world_size={self.world_size}, rank={self.rank})"
    
    def summary(self) -> str:
        """生成配置摘要"""
        lines = [
            f"=== {self.__class__.__name__} ===",
            f"Mode: {self.mode.value}",
            f"Backend: {self.backend.value}",
            f"World Size: {self.world_size}",
            f"Rank: {self.rank} (Local: {self.local_rank})",
            f"Master: {self.master_addr}:{self.master_port}",
        ]
        return '\n'.join(lines)


# ==================== DDP配置 ====================

@dataclass
class DDPConfig(DistributedConfig):
    """
    DDP配置
    
    PyTorch DistributedDataParallel配置，提供DDP特定的参数管理和优化。
    """
    # DDP特定配置
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    
    # 梯度相关
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    
    # 混合精度
    mixed_precision: bool = True
    
    def _post_init_hook(self):
        """设置模式为DDP"""
        self.mode = ParallelMode.DDP

    def _validate_specific(self) -> Tuple[List[str], List[str]]:
        """DDP特定验证"""
        errors = []
        warnings = []
        
        if self.bucket_cap_mb < 1:
            errors.append(f"bucket_cap_mb must be >= 1, got {self.bucket_cap_mb}")
        elif self.bucket_cap_mb > 100:
            warnings.append(f"bucket_cap_mb={self.bucket_cap_mb} is large, may impact performance")
        
        if self.static_graph and self.find_unused_parameters:
            warnings.append("static_graph=True with find_unused_parameters=True may cause issues")
        
        return errors, warnings
    
    def to_ddp_kwargs(self) -> Dict[str, Any]:
        """
        转换为PyTorch DDP构造参数
        
        Returns:
            可直接传递给DistributedDataParallel的kwargs
        """
        import torch
        
        kwargs = {
            'broadcast_buffers': self.broadcast_buffers,
            'bucket_cap_mb': self.bucket_cap_mb,
            'find_unused_parameters': self.find_unused_parameters,
            'gradient_as_bucket_view': self.gradient_as_bucket_view,
            'static_graph': self.static_graph,
        }
        
        if torch.cuda.is_available():
            kwargs['device_ids'] = [self.local_rank]
            kwargs['output_device'] = self.local_rank
        
        return kwargs
    
    def get_optimal_bucket_size(self, model_params: int) -> int:
        """
        根据模型大小计算最优bucket大小
        
        Args:
            model_params: 模型参数量
            
        Returns:
            推荐的bucket大小（MB）
        """
        # 经验公式：较大模型使用较大bucket
        # 参考PyTorch官方推荐
        if model_params < 10_000_000:  # <10M
            return 10
        elif model_params < 100_000_000:  # <100M
            return 25
        elif model_params < 1_000_000_000:  # <1B
            return 50
        else:
            return 100
    
    def optimize_for_model(self, model_params: int) -> 'DDPConfig':
        """
        根据模型大小优化配置
        
        Args:
            model_params: 模型参数量
            
        Returns:
            优化后的配置
        """
        config = self.copy()
        config.bucket_cap_mb = self.get_optimal_bucket_size(model_params)
        
        # 大模型启用静态图优化
        if model_params > 500_000_000:
            config.static_graph = True
            config.find_unused_parameters = False
        
        return config
    
    @classmethod
    def get_recommended_config(
        cls,
        num_gpus: int,
        model_params: int,
        use_mixed_precision: bool = True
    ) -> 'DDPConfig':
        """
        获取推荐配置
        
        Args:
            num_gpus: GPU数量
            model_params: 模型参数量
            use_mixed_precision: 是否使用混合精度
            
        Returns:
            推荐的DDP配置
        """
        config = cls(
            world_size=num_gpus,
            mixed_precision=use_mixed_precision,
        )
        return config.optimize_for_model(model_params)


# ==================== FSDP配置 ====================

@dataclass
class FSDPConfig(DistributedConfig):
    """
    FSDP配置
    
    PyTorch FullyShardedDataParallel配置，支持大模型训练的内存优化。
    """
    # 分片策略
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    
    # CPU Offload
    cpu_offload: bool = False
    offload_params: bool = False
    
    # 预取
    backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST
    forward_prefetch: bool = False
    
    # 包装策略
    auto_wrap_policy: str = "transformer_auto_wrap"
    min_num_params: int = 100_000_000  # 100M参数
    
    # 混合精度
    mixed_precision: bool = True
    param_dtype: str = "fp32"  # fp32, fp16, bf16
    reduce_dtype: str = "fp32"
    buffer_dtype: str = "fp32"
    
    # 状态字典
    state_dict_type: str = "FULL_STATE_DICT"  # FULL_STATE_DICT, LOCAL_STATE_DICT, SHARDED_STATE_DICT
    
    # 激活检查点
    activation_checkpointing: bool = False
    
    def _post_init_hook(self):
        """设置模式和类型转换"""
        self.mode = ParallelMode.FSDP
        if isinstance(self.sharding_strategy, str):
            self.sharding_strategy = ShardingStrategy(self.sharding_strategy)

    def _validate_specific(self) -> Tuple[List[str], List[str]]:
        """FSDP特定验证"""
        errors = []
        warnings = []
        
        valid_dtypes = {'fp32', 'fp16', 'bf16'}
        for dtype_name in ['param_dtype', 'reduce_dtype', 'buffer_dtype']:
            dtype = getattr(self, dtype_name)
            if dtype not in valid_dtypes:
                errors.append(f"{dtype_name} must be one of {valid_dtypes}, got '{dtype}'")
        
        if self.cpu_offload and not self.offload_params:
            warnings.append("cpu_offload=True but offload_params=False, only optimizer states will be offloaded")
        
        if self.min_num_params < 1_000_000:
            warnings.append(f"min_num_params={self.min_num_params} is small, may create many FSDP units")
        
        valid_prefetch = {'BACKWARD_PRE', 'BACKWARD_POST'}
        if self.backward_prefetch not in valid_prefetch:
            errors.append(f"backward_prefetch must be one of {valid_prefetch}, got '{self.backward_prefetch}'")
        
        return errors, warnings
    
    def _get_memory_factor(self) -> float:
        """FSDP内存优化系数"""
        base_factor = 1.0 / self.world_size  # 分片优化
        
        # 根据分片策略调整
        strategy_factor = {
            ShardingStrategy.FULL_SHARD: 1.0,
            ShardingStrategy.SHARD_GRAD_OP: 1.5,
            ShardingStrategy.HYBRID_SHARD: 1.2,
            ShardingStrategy.NO_SHARD: self.world_size,
        }
        
        return base_factor * strategy_factor.get(self.sharding_strategy, 1.0)
    
    def to_fsdp_kwargs(self) -> Dict[str, Any]:
        """
        转换为PyTorch FSDP构造参数
        
        Returns:
            FSDP构造参数字典
        """
        # 延迟导入避免循环依赖
        try:
            from torch.distributed.fsdp import (
                ShardingStrategy as FSDPShardingStrategy,
                MixedPrecision,
                BackwardPrefetch,
                CPUOffload,
            )
            import torch
        except ImportError:
            logger.warning("FSDP not available")
            return {}
        
        # 映射分片策略
        strategy_map = {
            ShardingStrategy.FULL_SHARD: FSDPShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP: FSDPShardingStrategy.SHARD_GRAD_OP,
            ShardingStrategy.NO_SHARD: FSDPShardingStrategy.NO_SHARD,
            ShardingStrategy.HYBRID_SHARD: FSDPShardingStrategy.HYBRID_SHARD,
        }
        
        # 映射数据类型
        dtype_map = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16,
        }
        
        # 映射预取策略
        prefetch_map = {
            'BACKWARD_PRE': BackwardPrefetch.BACKWARD_PRE,
            'BACKWARD_POST': BackwardPrefetch.BACKWARD_POST,
        }
        
        kwargs = {
            'sharding_strategy': strategy_map.get(self.sharding_strategy, FSDPShardingStrategy.FULL_SHARD),
            'backward_prefetch': prefetch_map.get(self.backward_prefetch, BackwardPrefetch.BACKWARD_PRE),
            'forward_prefetch': self.forward_prefetch,
            'use_orig_params': True,
            'limit_all_gathers': True,
        }
        
        # 混合精度配置
        if self.mixed_precision:
            kwargs['mixed_precision'] = MixedPrecision(
                param_dtype=dtype_map.get(self.param_dtype, torch.float32),
                reduce_dtype=dtype_map.get(self.reduce_dtype, torch.float32),
                buffer_dtype=dtype_map.get(self.buffer_dtype, torch.float32),
            )
        
        # CPU Offload配置
        if self.cpu_offload:
            kwargs['cpu_offload'] = CPUOffload(offload_params=self.offload_params)
        
        # 设备ID
        if torch.cuda.is_available():
            kwargs['device_id'] = self.local_rank
        
        return kwargs
    
    def get_mixed_precision_policy(self) -> Optional[Any]:
        """
        获取混合精度策略对象
        
        Returns:
            PyTorch MixedPrecision对象或None
        """
        if not self.mixed_precision:
            return None
        
        try:
            from torch.distributed.fsdp import MixedPrecision
            import torch
            
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
        except ImportError:
            return None
    
    def estimate_shard_size(self, model_params: int) -> Dict[str, float]:
        """
        估算分片大小
        
        Args:
            model_params: 模型总参数量
            
        Returns:
            分片信息
        """
        params_per_shard = model_params / self.world_size
        dtype_bytes = {'fp32': 4, 'fp16': 2, 'bf16': 2}.get(self.param_dtype, 4)
        
        shard_size_gb = params_per_shard * dtype_bytes / (1024**3)
        
        return {
            'total_params': model_params,
            'params_per_shard': params_per_shard,
            'shard_size_gb': shard_size_gb,
            'world_size': self.world_size,
            'sharding_strategy': self.sharding_strategy.value,
        }
    
    @classmethod
    def get_recommended_config(
        cls,
        num_gpus: int,
        model_params: int,
        gpu_memory_gb: float = 80.0,
        use_cpu_offload: bool = False
    ) -> 'FSDPConfig':
        """
        获取推荐FSDP配置
        
        Args:
            num_gpus: GPU数量
            model_params: 模型参数量
            gpu_memory_gb: 单卡GPU内存（GB）
            use_cpu_offload: 是否使用CPU卸载
            
        Returns:
            推荐的FSDP配置
        """
        # 估算每GPU内存需求
        # 假设fp16训练，参数+梯度+优化器状态约18bytes/param
        memory_per_param = 18
        total_memory_gb = model_params * memory_per_param / (1024**3)
        memory_per_gpu = total_memory_gb / num_gpus
        
        config = cls(world_size=num_gpus)
        
        # 根据内存情况选择策略
        if memory_per_gpu <= gpu_memory_gb * 0.5:
            config.sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        elif memory_per_gpu <= gpu_memory_gb * 0.8:
            config.sharding_strategy = ShardingStrategy.FULL_SHARD
        else:
            config.sharding_strategy = ShardingStrategy.FULL_SHARD
            config.activation_checkpointing = True
        
        # CPU卸载
        if use_cpu_offload or memory_per_gpu > gpu_memory_gb:
            config.cpu_offload = True
            config.offload_params = memory_per_gpu > gpu_memory_gb * 1.5
        
        # 大模型使用bf16
        if model_params > 1_000_000_000:
            config.param_dtype = 'bf16'
            config.reduce_dtype = 'fp32'
        
        logger.info(f"Recommended FSDP config: strategy={config.sharding_strategy.value}, "
                   f"cpu_offload={config.cpu_offload}, activation_checkpointing={config.activation_checkpointing}")
        
        return config


# ==================== Pipeline配置 ====================

@dataclass
class PipelineConfig(DistributedConfig):
    """
    流水线并行配置
    
    支持GPipe和Interleaved调度，提供气泡分析和优化。
    """
    # 流水线配置
    num_stages: int = 4
    num_micro_batches: int = 8
    
    # 调度策略
    schedule: str = "gpipe"  # gpipe, 1f1b, interleaved
    
    # 模型分割
    split_points: List[str] = field(default_factory=list)  # 模型分割点
    balance: List[int] = field(default_factory=list)  # 各阶段层数平衡
    
    # 通信
    async_comm: bool = True
    
    # 内存优化
    checkpoint_activations: bool = True
    
    def _post_init_hook(self):
        """设置模式"""
        self.mode = ParallelMode.PIPELINE

    def _validate_specific(self) -> Tuple[List[str], List[str]]:
        """Pipeline特定验证"""
        errors = []
        warnings = []
        
        if self.num_stages < 2:
            errors.append(f"num_stages must be >= 2 for pipeline parallel, got {self.num_stages}")
        
        if self.num_micro_batches < self.num_stages:
            warnings.append(f"num_micro_batches ({self.num_micro_batches}) < num_stages ({self.num_stages}), "
                          "pipeline efficiency will be low")
        
        valid_schedules = {'gpipe', '1f1b', 'interleaved'}
        if self.schedule not in valid_schedules:
            errors.append(f"schedule must be one of {valid_schedules}, got '{self.schedule}'")
        
        if self.balance and len(self.balance) != self.num_stages:
            errors.append(f"balance length ({len(self.balance)}) must match num_stages ({self.num_stages})")
        
        return errors, warnings
    
    def calculate_bubble_ratio(self) -> float:
        """
        计算流水线气泡比例
        
        气泡是流水线中的空闲时间，降低训练效率。
        
        Returns:
            气泡比例 (0-1)
        """
        p = self.num_stages
        m = self.num_micro_batches
        
        if self.schedule == 'gpipe':
            # GPipe气泡: (p-1) / (p-1+m)
            return (p - 1) / (p - 1 + m)
        elif self.schedule == '1f1b':
            # 1F1B气泡略低于GPipe
            return (p - 1) / (p - 1 + m) * 0.9
        elif self.schedule == 'interleaved':
            # 交错调度气泡更低（假设2个模型块）
            v = 2  # 虚拟阶段数
            return (p - 1) / (p - 1 + m * v)
        
        return (p - 1) / (p - 1 + m)
    
    def get_pipeline_efficiency(self) -> float:
        """
        获取流水线效率
        
        Returns:
            效率百分比 (0-100)
        """
        return (1 - self.calculate_bubble_ratio()) * 100
    
    def get_optimal_micro_batches(self, target_efficiency: float = 0.9) -> int:
        """
        计算达到目标效率所需的最小微批次数
        
        Args:
            target_efficiency: 目标效率 (0-1)
            
        Returns:
            推荐的微批次数
        """
        p = self.num_stages
        
        if self.schedule == 'interleaved':
            # (p-1) / (p-1+m*v) <= 1-target
            # m >= (p-1) / ((1-target) * v) - (p-1)/v
            v = 2
            target_bubble = 1 - target_efficiency
            m = math.ceil((p - 1) / (target_bubble * v))
        else:
            # (p-1) / (p-1+m) <= 1-target
            # m >= (p-1) / (1-target) - (p-1)
            target_bubble = 1 - target_efficiency
            m = math.ceil((p - 1) * target_efficiency / target_bubble)
        
        return max(m, self.num_stages)  # 至少等于阶段数
    
    def validate_balance(self, layer_counts: List[int]) -> bool:
        """
        验证分割平衡是否合理
        
        Args:
            layer_counts: 各阶段的计算量（层数或FLOPs）
            
        Returns:
            是否平衡
        """
        if not layer_counts:
            return True
        
        avg = sum(layer_counts) / len(layer_counts)
        max_deviation = max(abs(c - avg) for c in layer_counts) / avg
        
        # 允许20%的偏差
        return max_deviation <= 0.2
    
    def get_recommended_balance(self, total_layers: int) -> List[int]:
        """
        获取推荐的层数分配
        
        Args:
            total_layers: 总层数
            
        Returns:
            各阶段的层数列表
        """
        base = total_layers // self.num_stages
        remainder = total_layers % self.num_stages
        
        # 将余数分配给中间阶段（减少通信开销）
        balance = [base] * self.num_stages
        start = (self.num_stages - remainder) // 2
        for i in range(remainder):
            balance[start + i] += 1
        
        return balance
    
    def optimize_for_efficiency(self, target_efficiency: float = 0.9) -> 'PipelineConfig':
        """
        优化配置以达到目标效率
        
        Args:
            target_efficiency: 目标效率
            
        Returns:
            优化后的配置
        """
        config = self.copy()
        config.num_micro_batches = self.get_optimal_micro_batches(target_efficiency)
        
        # 如果目标效率很高，考虑使用交错调度
        if target_efficiency > 0.95 and self.schedule != 'interleaved':
            config.schedule = 'interleaved'
            config.num_micro_batches = self.get_optimal_micro_batches(target_efficiency)
        
        return config
    
    @classmethod
    def get_recommended_config(
        cls,
        num_gpus: int,
        total_layers: int,
        target_efficiency: float = 0.9
    ) -> 'PipelineConfig':
        """
        获取推荐的流水线配置
        
        Args:
            num_gpus: GPU数量
            total_layers: 总层数
            target_efficiency: 目标效率
            
        Returns:
            推荐配置
        """
        config = cls(
            world_size=num_gpus,
            num_stages=num_gpus,
        )
        
        # 计算推荐的微批次数
        config.num_micro_batches = config.get_optimal_micro_batches(target_efficiency)
        
        # 推荐的层分配
        config.balance = config.get_recommended_balance(total_layers)
        
        # 大规模流水线使用交错调度
        if num_gpus >= 8:
            config.schedule = 'interleaved'
        
        logger.info(f"Recommended Pipeline config: stages={config.num_stages}, "
                   f"micro_batches={config.num_micro_batches}, schedule={config.schedule}, "
                   f"efficiency={config.get_pipeline_efficiency():.1f}%")
        
        return config


# ==================== ZeRO配置 ====================

@dataclass
class ZeROConfig(DistributedConfig):
    """
    ZeRO优化配置
    
    DeepSpeed ZeRO优化配置，支持Stage 1/2/3的内存优化。
    """
    # ZeRO阶段
    stage: int = 2  # 1, 2, 3
    
    # 优化器分片
    optimizer_partition: bool = True
    
    # 梯度分片
    gradient_partition: bool = True
    reduce_scatter: bool = True
    
    # 参数分片（Stage 3）
    param_partition: bool = False
    
    # CPU Offload
    offload_optimizer: bool = False
    offload_param: bool = False
    offload_optimizer_device: str = "cpu"  # cpu, nvme
    offload_param_device: str = "cpu"
    
    # NVMe Offload配置
    nvme_path: str = "/local_nvme"
    
    # 通信优化
    overlap_comm: bool = True
    contiguous_gradients: bool = True
    
    # 内存优化
    allgather_bucket_size: int = 500_000_000  # 500MB
    reduce_bucket_size: int = 500_000_000
    
    # Infinity内存
    zero_infinity: bool = False
    
    def _post_init_hook(self):
        """设置模式"""
        if self.stage == 1:
            self.mode = ParallelMode.ZERO_1
        elif self.stage == 2:
            self.mode = ParallelMode.ZERO_2
        else:
            self.mode = ParallelMode.ZERO_3

    def _validate_specific(self) -> Tuple[List[str], List[str]]:
        """ZeRO特定验证"""
        errors = []
        warnings = []
        
        if self.stage not in (1, 2, 3):
            errors.append(f"stage must be 1, 2, or 3, got {self.stage}")
        
        if self.offload_param and self.stage < 3:
            errors.append("offload_param requires stage=3")
        
        valid_devices = {'cpu', 'nvme'}
        if self.offload_optimizer_device not in valid_devices:
            errors.append(f"offload_optimizer_device must be one of {valid_devices}")
        if self.offload_param_device not in valid_devices:
            errors.append(f"offload_param_device must be one of {valid_devices}")
        
        if self.zero_infinity and self.stage < 3:
            warnings.append("zero_infinity is most effective with stage=3")
        
        return errors, warnings
    
    def _get_memory_factor(self) -> float:
        """ZeRO内存优化系数"""
        # Stage 1: 优化器状态分片
        # Stage 2: + 梯度分片
        # Stage 3: + 参数分片
        base_factor = {
            1: 1.0 / self.world_size * 4,  # 优化器状态约占4倍参数
            2: 1.0 / self.world_size * 2,  # + 梯度分片
            3: 1.0 / self.world_size,      # + 参数分片
        }.get(self.stage, 1.0)
        
        # CPU卸载进一步减少GPU内存
        if self.offload_optimizer:
            base_factor *= 0.5
        if self.offload_param:
            base_factor *= 0.5
        
        return base_factor
    
    def to_deepspeed_config(self) -> Dict[str, Any]:
        """
        转换为DeepSpeed配置字典
        
        Returns:
            DeepSpeed配置
        """
        config = {
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            
            "zero_optimization": {
                "stage": self.stage,
                "overlap_comm": self.overlap_comm,
                "contiguous_gradients": self.contiguous_gradients,
                "reduce_bucket_size": self.reduce_bucket_size,
                "allgather_bucket_size": self.allgather_bucket_size,
            }
        }
        
        # Stage 3特定配置
        if self.stage == 3:
            config["zero_optimization"].update({
                "stage3_max_live_parameters": 1_000_000_000,
                "stage3_max_reuse_distance": 1_000_000_000,
                "stage3_prefetch_bucket_size": 500_000_000,
                "stage3_param_persistence_threshold": 100_000,
            })
        
        # CPU Offload
        if self.offload_optimizer:
            config["zero_optimization"]["offload_optimizer"] = {
                "device": self.offload_optimizer_device,
                "pin_memory": True,
            }
            if self.offload_optimizer_device == "nvme":
                config["zero_optimization"]["offload_optimizer"]["nvme_path"] = self.nvme_path
        
        if self.offload_param:
            config["zero_optimization"]["offload_param"] = {
                "device": self.offload_param_device,
                "pin_memory": True,
            }
            if self.offload_param_device == "nvme":
                config["zero_optimization"]["offload_param"]["nvme_path"] = self.nvme_path
        
        # ZeRO Infinity
        if self.zero_infinity:
            config["zero_optimization"]["zero_infinity"] = {
                "enabled": True,
            }
        
        return config
    
    def estimate_memory_per_gpu(
        self,
        model_params: int,
        dtype_bytes: int = 2
    ) -> Dict[str, float]:
        """
        估算每GPU内存需求
        
        Args:
            model_params: 模型参数量
            dtype_bytes: 数据类型字节数
            
        Returns:
            内存估算
        """
        # 参数内存
        param_memory = model_params * dtype_bytes
        
        # 梯度内存
        grad_memory = model_params * dtype_bytes
        
        # 优化器状态（Adam: 2倍参数，fp32）
        optimizer_memory = model_params * 4 * 2
        
        # 根据Stage计算分片
        if self.stage == 1:
            # 只分片优化器状态
            param_per_gpu = param_memory
            grad_per_gpu = grad_memory
            optimizer_per_gpu = optimizer_memory / self.world_size
        elif self.stage == 2:
            # 分片优化器和梯度
            param_per_gpu = param_memory
            grad_per_gpu = grad_memory / self.world_size
            optimizer_per_gpu = optimizer_memory / self.world_size
        else:  # stage == 3
            # 全部分片
            param_per_gpu = param_memory / self.world_size
            grad_per_gpu = grad_memory / self.world_size
            optimizer_per_gpu = optimizer_memory / self.world_size
        
        # CPU卸载
        if self.offload_optimizer:
            optimizer_per_gpu = 0
        if self.offload_param and self.stage == 3:
            param_per_gpu = 0
        
        total_per_gpu = param_per_gpu + grad_per_gpu + optimizer_per_gpu
        
        return {
            'param_memory_gb': param_per_gpu / (1024**3),
            'grad_memory_gb': grad_per_gpu / (1024**3),
            'optimizer_memory_gb': optimizer_per_gpu / (1024**3),
            'total_per_gpu_gb': total_per_gpu / (1024**3),
            'stage': self.stage,
            'world_size': self.world_size,
        }
    
    @classmethod
    def get_recommended_stage(
        cls,
        model_params: int,
        num_gpus: int,
        gpu_memory_gb: float = 80.0
    ) -> int:
        """
        推荐ZeRO Stage
        
        Args:
            model_params: 模型参数量
            num_gpus: GPU数量
            gpu_memory_gb: 单卡内存
            
        Returns:
            推荐的Stage
        """
        # 估算总内存需求
        memory_per_param = 18  # bytes (fp16 + optimizer)
        total_memory_gb = model_params * memory_per_param / (1024**3)
        
        # Stage 1: 优化器状态分片
        stage1_memory = total_memory_gb - (model_params * 8 / (1024**3)) * (1 - 1/num_gpus)
        
        # Stage 2: + 梯度分片
        stage2_memory = total_memory_gb / 2 + (model_params * 2 / (1024**3))
        
        # Stage 3: 全部分片
        stage3_memory = total_memory_gb / num_gpus
        
        if stage1_memory / num_gpus <= gpu_memory_gb * 0.7:
            return 1
        elif stage2_memory / num_gpus <= gpu_memory_gb * 0.7:
            return 2
        else:
            return 3
    
    @classmethod
    def get_recommended_config(
        cls,
        model_params: int,
        num_gpus: int,
        gpu_memory_gb: float = 80.0
    ) -> 'ZeROConfig':
        """
        获取推荐ZeRO配置
        
        Args:
            model_params: 模型参数量
            num_gpus: GPU数量
            gpu_memory_gb: 单卡内存
            
        Returns:
            推荐配置
        """
        stage = cls.get_recommended_stage(model_params, num_gpus, gpu_memory_gb)
        
        config = cls(
            world_size=num_gpus,
            stage=stage,
        )
        
        # 估算内存
        memory_info = config.estimate_memory_per_gpu(model_params)
        
        # 如果还是不够，启用CPU卸载
        if memory_info['total_per_gpu_gb'] > gpu_memory_gb * 0.8:
            config.offload_optimizer = True
            
            # 重新估算
            memory_info = config.estimate_memory_per_gpu(model_params)
            
            if memory_info['total_per_gpu_gb'] > gpu_memory_gb * 0.8 and stage == 3:
                config.offload_param = True
        
        logger.info(f"Recommended ZeRO config: stage={config.stage}, "
                   f"offload_optimizer={config.offload_optimizer}, "
                   f"offload_param={config.offload_param}, "
                   f"estimated_memory={memory_info['total_per_gpu_gb']:.2f}GB/GPU")
        
        return config


# ==================== Tensor Parallel配置 ====================

@dataclass
class TensorParallelConfig(DistributedConfig):
    """
    张量并行配置
    
    Megatron风格的张量并行，支持注意力头和MLP分割。
    """
    # 张量并行度
    tensor_parallel_size: int = 2
    
    # 分割维度
    split_dim: str = "column"  # column, row
    
    # 注意力分割
    attention_heads_per_partition: Optional[int] = None
    
    # 序列并行
    sequence_parallel: bool = False
    
    # 通信组
    tensor_model_parallel_group: Optional[Any] = None
    
    def _post_init_hook(self):
        """设置模式"""
        self.mode = ParallelMode.TENSOR

    def _validate_specific(self) -> Tuple[List[str], List[str]]:
        """Tensor Parallel特定验证"""
        errors = []
        warnings = []
        
        if self.tensor_parallel_size < 1:
            errors.append(f"tensor_parallel_size must be >= 1, got {self.tensor_parallel_size}")
        
        if self.tensor_parallel_size > 8:
            warnings.append(f"tensor_parallel_size={self.tensor_parallel_size} is large, "
                          "communication overhead may be high")
        
        valid_dims = {'column', 'row'}
        if self.split_dim not in valid_dims:
            errors.append(f"split_dim must be one of {valid_dims}, got '{self.split_dim}'")
        
        return errors, warnings
    
    def validate_attention_heads(self, num_heads: int) -> bool:
        """
        验证注意力头是否可被张量并行度整除
        
        Args:
            num_heads: 注意力头数量
            
        Returns:
            是否有效
        """
        if num_heads % self.tensor_parallel_size != 0:
            logger.error(f"num_heads ({num_heads}) must be divisible by "
                        f"tensor_parallel_size ({self.tensor_parallel_size})")
            return False
        return True
    
    def get_attention_head_partition(self, num_heads: int) -> int:
        """
        计算每个分区的注意力头数
        
        Args:
            num_heads: 总注意力头数
            
        Returns:
            每分区注意力头数
        """
        if not self.validate_attention_heads(num_heads):
            raise ConfigValidationError(
                f"num_heads ({num_heads}) not divisible by tensor_parallel_size ({self.tensor_parallel_size})"
            )
        return num_heads // self.tensor_parallel_size
    
    def get_hidden_dim_partition(self, hidden_dim: int) -> int:
        """
        计算每个分区的隐藏维度
        
        Args:
            hidden_dim: 总隐藏维度
            
        Returns:
            每分区隐藏维度
        """
        if hidden_dim % self.tensor_parallel_size != 0:
            raise ConfigValidationError(
                f"hidden_dim ({hidden_dim}) not divisible by tensor_parallel_size ({self.tensor_parallel_size})"
            )
        return hidden_dim // self.tensor_parallel_size
    
    def estimate_communication_volume(
        self,
        batch_size: int,
        sequence_length: int,
        hidden_dim: int
    ) -> Dict[str, float]:
        """
        估算通信量
        
        Args:
            batch_size: 批次大小
            sequence_length: 序列长度
            hidden_dim: 隐藏维度
            
        Returns:
            通信量估算
        """
        # 张量并行每层需要2次all-reduce（前向+反向各1次）
        # 通信量 = batch_size * seq_len * hidden_dim * 2 bytes (fp16)
        comm_per_layer = batch_size * sequence_length * hidden_dim * 2
        
        # 序列并行减少通信
        if self.sequence_parallel:
            comm_per_layer *= 0.5
        
        return {
            'comm_per_layer_mb': comm_per_layer / (1024**2),
            'tensor_parallel_size': self.tensor_parallel_size,
            'sequence_parallel': self.sequence_parallel,
        }
    
    @classmethod
    def get_recommended_config(
        cls,
        num_gpus: int,
        num_attention_heads: int,
        hidden_dim: int
    ) -> 'TensorParallelConfig':
        """
        获取推荐配置
        
        Args:
            num_gpus: GPU数量
            num_attention_heads: 注意力头数
            hidden_dim: 隐藏维度
            
        Returns:
            推荐配置
        """
        # 找到能整除attention_heads的最大tensor_parallel_size
        tp_size = 1
        for candidate in [8, 4, 2]:
            if candidate <= num_gpus and num_attention_heads % candidate == 0 and hidden_dim % candidate == 0:
                tp_size = candidate
                break
        
        config = cls(
            world_size=num_gpus,
            tensor_parallel_size=tp_size,
        )
        
        # 大规模模型启用序列并行
        if hidden_dim >= 4096:
            config.sequence_parallel = True
        
        logger.info(f"Recommended Tensor Parallel config: tp_size={tp_size}, "
                   f"sequence_parallel={config.sequence_parallel}")
        
        return config


# ==================== 混合并行配置 ====================

@dataclass
class HybridParallelConfig(DistributedConfig):
    """
    混合并行配置
    
    组合多种并行策略，支持3D并行（DP + TP + PP）。
    """
    # 数据并行
    data_parallel_size: int = 1
    data_parallel_mode: ParallelMode = ParallelMode.DDP
    
    # 张量并行
    tensor_parallel_size: int = 1
    tensor_parallel_enabled: bool = False
    
    # 流水线并行
    pipeline_parallel_size: int = 1
    pipeline_parallel_enabled: bool = False
    num_micro_batches: int = 1
    
    # ZeRO
    zero_stage: int = 0
    zero_enabled: bool = False
    
    # 序列并行
    sequence_parallel_enabled: bool = False
    
    # 专家并行（MoE）
    expert_parallel_size: int = 1
    expert_parallel_enabled: bool = False
    
    def _post_init_hook(self):
        """设置模式和计算world_size"""
        self.mode = ParallelMode.HYBRID
        
        # 计算总world_size
        total = self.data_parallel_size * self.tensor_parallel_size * self.pipeline_parallel_size
        if total > 1 and self.world_size == 1:
            self.world_size = total
    
    def _validate_specific(self) -> Tuple[List[str], List[str]]:
        """混合并行特定验证"""
        errors = []
        warnings = []
        
        # 验证并行度乘积
        total = self.data_parallel_size * self.tensor_parallel_size * self.pipeline_parallel_size
        if total != self.world_size and self.world_size > 1:
            errors.append(f"dp_size * tp_size * pp_size ({total}) != world_size ({self.world_size})")
        
        # 验证各并行度
        for name, size in [
            ('data_parallel_size', self.data_parallel_size),
            ('tensor_parallel_size', self.tensor_parallel_size),
            ('pipeline_parallel_size', self.pipeline_parallel_size),
        ]:
            if size < 1:
                errors.append(f"{name} must be >= 1, got {size}")
        
        # ZeRO与FSDP不兼容
        if self.zero_enabled and self.data_parallel_mode == ParallelMode.FSDP:
            errors.append("ZeRO and FSDP cannot be used together")
        
        # 专家并行检查
        if self.expert_parallel_enabled and self.expert_parallel_size > self.data_parallel_size:
            warnings.append("expert_parallel_size > data_parallel_size may cause issues")
        
        return errors, warnings
    
    @property
    def total_gpus(self) -> int:
        """计算总GPU数"""
        return self.data_parallel_size * self.tensor_parallel_size * self.pipeline_parallel_size
    
    def validate_topology(self) -> bool:
        """
        验证并行拓扑是否有效
        
        Returns:
            是否有效
        """
        try:
            self.validate()
            return True
        except ConfigValidationError:
            return False
    
    def get_process_groups_config(self) -> Dict[str, Any]:
        """
        获取进程组配置
        
        Returns:
            进程组配置信息
        """
        dp = self.data_parallel_size
        tp = self.tensor_parallel_size
        pp = self.pipeline_parallel_size
        
        return {
            'data_parallel': {
                'size': dp,
                'ranks_per_group': list(range(0, self.world_size, tp * pp)),
            },
            'tensor_parallel': {
                'size': tp,
                'ranks_per_group': [list(range(i, i + tp)) for i in range(0, self.world_size, tp)],
            },
            'pipeline_parallel': {
                'size': pp,
                'ranks_per_group': [list(range(i, self.world_size, dp * tp)) for i in range(dp * tp)],
            },
        }
    
    def calculate_communication_overhead(self) -> Dict[str, str]:
        """
        计算各类通信开销
        
        Returns:
            通信开销描述
        """
        overhead = {}
        
        # 数据并行通信
        if self.data_parallel_size > 1:
            if self.zero_enabled:
                overhead['data_parallel'] = f"ZeRO Stage {self.zero_stage} gradient sync"
            else:
                overhead['data_parallel'] = "AllReduce gradients"
        
        # 张量并行通信
        if self.tensor_parallel_enabled:
            overhead['tensor_parallel'] = "AllReduce per layer (high frequency)"
        
        # 流水线并行通信
        if self.pipeline_parallel_enabled:
            overhead['pipeline_parallel'] = "Point-to-point activation transfer"
        
        return overhead
    
    def get_memory_efficiency_score(self) -> float:
        """
        计算内存效率得分
        
        Returns:
            效率得分 (0-1)
        """
        score = 1.0
        
        # FSDP/ZeRO提升内存效率
        if self.data_parallel_mode == ParallelMode.FSDP:
            score *= 1.0 / self.data_parallel_size
        elif self.zero_enabled:
            zero_factor = {1: 0.75, 2: 0.5, 3: 0.25}.get(self.zero_stage, 1.0)
            score *= zero_factor
        
        # 张量并行分割参数
        if self.tensor_parallel_enabled:
            score *= 1.0 / self.tensor_parallel_size
        
        # 流水线并行分割模型
        if self.pipeline_parallel_enabled:
            score *= 1.0 / self.pipeline_parallel_size
        
        return score
    
    @classmethod
    def get_recommended_config(
        cls,
        num_gpus: int,
        model_params: int,
        num_attention_heads: int = 32,
        num_layers: int = 32,
        gpu_memory_gb: float = 80.0
    ) -> 'HybridParallelConfig':
        """
        获取推荐的3D并行配置
        
        Args:
            num_gpus: 总GPU数量
            model_params: 模型参数量
            num_attention_heads: 注意力头数
            num_layers: 层数
            gpu_memory_gb: 单卡内存
            
        Returns:
            推荐配置
        """
        # 估算内存需求
        memory_per_param = 18  # bytes
        total_memory_gb = model_params * memory_per_param / (1024**3)
        
        config = cls(world_size=num_gpus)
        
        # 决定张量并行度（优先使用，减少内存）
        tp_candidates = [8, 4, 2, 1]
        for tp in tp_candidates:
            if tp <= num_gpus and num_attention_heads % tp == 0:
                memory_after_tp = total_memory_gb / tp
                if memory_after_tp / (num_gpus / tp) <= gpu_memory_gb * 0.7:
                    config.tensor_parallel_size = tp
                    config.tensor_parallel_enabled = tp > 1
                    break
        
        remaining_gpus = num_gpus // config.tensor_parallel_size
        
        # 决定流水线并行度
        if remaining_gpus > 1 and num_layers >= 8:
            # 每个流水线阶段至少4层
            max_pp = min(remaining_gpus, num_layers // 4)
            for pp in [max_pp, max_pp // 2, 2, 1]:
                if pp <= remaining_gpus and remaining_gpus % pp == 0:
                    config.pipeline_parallel_size = pp
                    config.pipeline_parallel_enabled = pp > 1
                    break
        
        # 剩余GPU用于数据并行
        config.data_parallel_size = num_gpus // (config.tensor_parallel_size * config.pipeline_parallel_size)
        
        # 决定是否使用ZeRO
        if config.data_parallel_size > 1:
            memory_per_dp_gpu = total_memory_gb / (config.tensor_parallel_size * config.pipeline_parallel_size)
            if memory_per_dp_gpu > gpu_memory_gb * 0.5:
                config.zero_enabled = True
                config.zero_stage = ZeROConfig.get_recommended_stage(
                    model_params // config.tensor_parallel_size // config.pipeline_parallel_size,
                    config.data_parallel_size,
                    gpu_memory_gb
                )
        
        # 设置微批次数
        if config.pipeline_parallel_enabled:
            config.num_micro_batches = max(4, config.pipeline_parallel_size * 2)
        
        logger.info(f"Recommended Hybrid config: DP={config.data_parallel_size}, "
                   f"TP={config.tensor_parallel_size}, PP={config.pipeline_parallel_size}, "
                   f"ZeRO={config.zero_stage if config.zero_enabled else 'disabled'}")
        
        return config


# ==================== 工厂函数 ====================

def create_distributed_config(
    mode: str,
    world_size: int = 1,
    validate: bool = True,
    **kwargs
) -> DistributedConfig:
    """
    创建分布式配置
    
    Args:
        mode: 并行模式
        world_size: 进程总数
        validate: 是否验证配置
        **kwargs: 其他配置参数
        
    Returns:
        配置实例
    """
    mode_enum = ParallelMode(mode) if isinstance(mode, str) else mode
    
    config_map = {
        ParallelMode.NONE: DistributedConfig,
        ParallelMode.DDP: DDPConfig,
        ParallelMode.FSDP: FSDPConfig,
        ParallelMode.PIPELINE: PipelineConfig,
        ParallelMode.ZERO_1: ZeROConfig,
        ParallelMode.ZERO_2: ZeROConfig,
        ParallelMode.ZERO_3: ZeROConfig,
        ParallelMode.TENSOR: TensorParallelConfig,
        ParallelMode.HYBRID: HybridParallelConfig,
        ParallelMode.MEGATRON: HybridParallelConfig
    }
    
    config_class = config_map.get(mode_enum, DistributedConfig)
    
    # 特殊处理ZeRO stage
    if mode_enum in (ParallelMode.ZERO_1, ParallelMode.ZERO_2, ParallelMode.ZERO_3):
        stage = int(mode_enum.value.split('_')[1])
        kwargs['stage'] = stage
    
    config = config_class(world_size=world_size, **kwargs)

    if validate:
        config.validate()
    
    return config


def auto_select_parallel_mode(
    model_params: int,
    num_gpus: int,
    gpu_memory_gb: float = 80.0,
    num_attention_heads: int = 32,
    num_layers: int = 32
) -> ParallelMode:
    """
    自动选择最佳并行模式
    
    根据模型大小和硬件资源自动选择。
    
    Args:
        model_params: 模型参数量
        num_gpus: GPU数量
        gpu_memory_gb: 单卡内存
        num_attention_heads: 注意力头数
        num_layers: 层数
        
    Returns:
        推荐的并行模式
    """
    # 估算内存需求
    memory_per_param = 18  # bytes (fp16 + optimizer)
    total_memory_gb = model_params * memory_per_param / (1024**3)
    memory_per_gpu = total_memory_gb / num_gpus
    
    # 小模型或单GPU
    if num_gpus == 1:
        if memory_per_gpu <= gpu_memory_gb * 0.7:
            return ParallelMode.NONE
        else:
            return ParallelMode.NONE  # 需要梯度检查点或其他优化
    
    # 多GPU情况
    if memory_per_gpu <= gpu_memory_gb * 0.5:
        # 内存充足，使用DDP
        return ParallelMode.DDP
    elif memory_per_gpu <= gpu_memory_gb * 0.8:
        # 内存较紧，使用FSDP或ZeRO
        return ParallelMode.FSDP
    elif memory_per_gpu <= gpu_memory_gb * 1.5:
        # 需要更激进的内存优化
        return ParallelMode.ZERO_3
    else:
        # 超大模型，需要混合并行
        return ParallelMode.HYBRID


def get_optimal_config(
    model_params: int,
    num_gpus: int,
    gpu_memory_gb: float = 80.0,
    num_attention_heads: int = 32,
    num_layers: int = 32,
    target_efficiency: float = 0.9
) -> DistributedConfig:
    """
    获取最优分布式配置
    
    综合考虑内存、通信和效率，返回最佳配置。
    
    Args:
        model_params: 模型参数量
        num_gpus: GPU数量
        gpu_memory_gb: 单卡内存
        num_attention_heads: 注意力头数
        num_layers: 层数
        target_efficiency: 目标效率
        
    Returns:
        最优配置
    """
    mode = auto_select_parallel_mode(
        model_params, num_gpus, gpu_memory_gb, num_attention_heads, num_layers
    )
    
    if mode == ParallelMode.NONE:
        return DistributedConfig(mode=mode)
    elif mode == ParallelMode.DDP:
        return DDPConfig.get_recommended_config(num_gpus, model_params)
    elif mode == ParallelMode.FSDP:
        return FSDPConfig.get_recommended_config(num_gpus, model_params, gpu_memory_gb)
    elif mode == ParallelMode.ZERO_3:
        return ZeROConfig.get_recommended_config(model_params, num_gpus, gpu_memory_gb)
    elif mode == ParallelMode.HYBRID:
        return HybridParallelConfig.get_recommended_config(
            num_gpus, model_params, num_attention_heads, num_layers, gpu_memory_gb
        )
    else:
        return create_distributed_config(mode.value, world_size=num_gpus)


# ==================== 便捷函数 ====================

def print_config_summary(config: DistributedConfig) -> None:
    """打印配置摘要"""
    print(config.summary())


def validate_all_configs(configs: List[DistributedConfig]) -> bool:
    """
    批量验证配置
    
    Args:
        configs: 配置列表
        
    Returns:
        是否全部有效
    """
    all_valid = True
    for i, config in enumerate(configs):
        try:
            config.validate()
            logger.info(f"Config {i}: Valid")
        except ConfigValidationError as e:
            logger.error(f"Config {i}: Invalid - {e}")
            all_valid = False
    return all_valid

# -*- coding: utf-8 -*-
"""
分布式管理器

统一管理分布式训练的初始化、通信和资源清理。
"""

import os
import time
import logging
from typing import Optional, List, Any, Dict, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from collections import defaultdict
import threading
import json

import torch
import torch.nn as nn
import torch.distributed as dist

from .parallel_modes import (
    ParallelMode, DistributedConfig, DDPConfig, FSDPConfig, 
    PipelineConfig, ZeROConfig, HybridParallelConfig,
    CommunicationBackend
)

logger = logging.getLogger(__name__)


class AllReduceOp(Enum):
    """AllReduce操作类型"""
    SUM = "sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    BAND = "band"
    BOR = "bor"
    BXOR = "bxor"
    
    def to_torch_op(self) -> 'dist.ReduceOp':
        """转换为PyTorch ReduceOp"""
        op_map = {
            AllReduceOp.SUM: dist.ReduceOp.SUM,
            AllReduceOp.PRODUCT: dist.ReduceOp.PRODUCT,
            AllReduceOp.MIN: dist.ReduceOp.MIN,
            AllReduceOp.MAX: dist.ReduceOp.MAX,
        }
        return op_map.get(self, dist.ReduceOp.SUM)


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class DistributedState:
    """分布式状态"""
    initialized: bool = False
    backend: str = "nccl"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    local_world_size: int = 1
    node_rank: int = 0
    num_nodes: int = 1
    
    # 并行组
    data_parallel_group: Optional[Any] = None
    model_parallel_group: Optional[Any] = None
    pipeline_parallel_group: Optional[Any] = None
    tensor_parallel_group: Optional[Any] = None
    
    # 运行时状态
    parallel_mode: ParallelMode = ParallelMode.NONE
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_health_check: float = 0.0
    
    # 统计信息
    total_communication_ops: int = 0
    failed_communication_ops: int = 0
    total_sync_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'initialized': self.initialized,
            'backend': self.backend,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'local_world_size': self.local_world_size,
            'node_rank': self.node_rank,
            'num_nodes': self.num_nodes,
            'parallel_mode': self.parallel_mode.value if isinstance(self.parallel_mode, ParallelMode) else str(self.parallel_mode),
            'health_status': self.health_status.value,
            'total_communication_ops': self.total_communication_ops,
            'failed_communication_ops': self.failed_communication_ops,
            'total_sync_time': self.total_sync_time,
        }


@dataclass
class ProcessGroupConfig:
    """进程组配置"""
    name: str
    ranks: List[int]
    backend: Optional[str] = None
    timeout: int = 1800  # 30分钟
    
    def validate(self) -> None:
        """验证配置"""
        if not self.ranks:
            raise ValueError("Process group must have at least one rank")
        if len(self.ranks) != len(set(self.ranks)):
            raise ValueError("Duplicate ranks in process group")


class DistributedHealthMonitor:
    """分布式健康监控器"""
    
    def __init__(self, check_interval: float = 60.0):
        """
        初始化健康监控器
        
        Args:
            check_interval: 健康检查间隔（秒）
        """
        self.check_interval = check_interval
        self._last_check_time = 0.0
        self._health_history: List[Tuple[float, HealthStatus]] = []
        self._max_history = 100
    
    def check_health(self, state: DistributedState) -> HealthStatus:
        """
        检查分布式环境健康状态
        
        Args:
            state: 分布式状态
            
        Returns:
            健康状态
        """
        current_time = time.time()
        
        # 如果距离上次检查时间太短，返回缓存结果
        if current_time - self._last_check_time < self.check_interval:
            return state.health_status
        
        self._last_check_time = current_time
        
        if not state.initialized:
            status = HealthStatus.UNKNOWN
        elif not dist.is_initialized():
            status = HealthStatus.UNHEALTHY
        else:
            # 检查通信是否正常
            try:
                if state.world_size > 1:
                    # 尝试一个简单的barrier操作
                    test_tensor = torch.zeros(1)
                    if torch.cuda.is_available():
                        test_tensor = test_tensor.cuda()
                    
                    # 超时检查
                    start = time.time()
                    if dist.is_initialized():
                        dist.barrier()
                    duration = time.time() - start
                    
                    # 如果barrier耗时过长，认为性能下降
                    if duration > 5.0:
                        status = HealthStatus.DEGRADED
                    else:
                        status = HealthStatus.HEALTHY
                else:
                    status = HealthStatus.HEALTHY
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                status = HealthStatus.UNHEALTHY
        
        # 记录历史
        self._health_history.append((current_time, status))
        if len(self._health_history) > self._max_history:
            self._health_history.pop(0)
        
        return status
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要"""
        if not self._health_history:
            return {'status': 'no_data', 'checks': 0}
        
        status_counts = defaultdict(int)
        for _, status in self._health_history:
            status_counts[status.value] += 1
        
        return {
            'current_status': self._health_history[-1][1].value,
            'total_checks': len(self._health_history),
            'status_distribution': dict(status_counts),
            'last_check_time': self._health_history[-1][0],
        }


class CommunicationProfiler:
    """通信性能分析器"""
    
    def __init__(self):
        self._enabled = False
        self._stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'total_bytes': 0,
            'errors': 0
        })
        self._current_op: Optional[Tuple[str, float]] = None
    
    def enable(self) -> None:
        """启用性能分析"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用性能分析"""
        self._enabled = False
    
    def start_op(self, op_name: str) -> None:
        """开始记录操作"""
        if self._enabled:
            self._current_op = (op_name, time.time())
    
    def end_op(self, op_name: str, num_bytes: int = 0, success: bool = True) -> None:
        """结束记录操作"""
        if not self._enabled or not self._current_op:
            return
        
        recorded_op, start_time = self._current_op
        if recorded_op != op_name:
            logger.warning(f"Operation mismatch: expected {recorded_op}, got {op_name}")
            return
        
        duration = time.time() - start_time
        stats = self._stats[op_name]
        stats['count'] += 1
        stats['total_time'] += duration
        stats['total_bytes'] += num_bytes
        if not success:
            stats['errors'] += 1
        
        self._current_op = None
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取统计信息"""
        result = {}
        for op_name, stats in self._stats.items():
            result[op_name] = {
                'count': stats['count'],
                'total_time_ms': stats['total_time'] * 1000,
                'avg_time_ms': (stats['total_time'] / stats['count'] * 1000) if stats['count'] > 0 else 0,
                'total_bytes': stats['total_bytes'],
                'errors': stats['errors'],
                'success_rate': (stats['count'] - stats['errors']) / stats['count'] if stats['count'] > 0 else 0,
            }
        return result
    
    def reset(self) -> None:
        """重置统计"""
        self._stats.clear()
        self._current_op = None
    
    def print_summary(self) -> None:
        """打印统计摘要"""
        stats = self.get_stats()
        if not stats:
            print("No communication operations recorded")
            return
        
        print("\n" + "="*80)
        print("Communication Profiling Summary")
        print("="*80)
        
        for op_name, op_stats in sorted(stats.items()):
            print(f"\nOperation: {op_name}")
            print(f"  Count: {op_stats['count']}")
            print(f"  Total Time: {op_stats['total_time_ms']:.2f} ms")
            print(f"  Avg Time: {op_stats['avg_time_ms']:.2f} ms")
            print(f"  Total Bytes: {op_stats['total_bytes']:,}")
            print(f"  Errors: {op_stats['errors']}")
            print(f"  Success Rate: {op_stats['success_rate']:.2%}")
        
        print("="*80)


# 全局分布式管理器实例
_distributed_manager: Optional['DistributedManager'] = None
_manager_lock = threading.Lock()


class DistributedManager:
    """
    分布式管理器
    
    统一管理分布式训练的所有方面。
    """
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        self.state = DistributedState()
        
        # 包装器
        self._ddp_wrapper = None
        self._fsdp_wrapper = None
        self._pipeline_wrapper = None
        self._zero_wrapper = None
        
        # 监控和分析组件
        self._health_monitor = DistributedHealthMonitor()
        self._comm_profiler = CommunicationProfiler()
        
        # 进程组管理
        self._process_groups: Dict[str, Any] = {}
        self._process_group_configs: Dict[str, ProcessGroupConfig] = {}
        
        # 统计信息
        self._init_time: Optional[float] = None
        self._cleanup_time: Optional[float] = None
    
    def initialize(
        self,
        backend: Optional[str] = None,
        init_method: Optional[str] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        timeout_minutes: int = 30
    ) -> None:
        """
        初始化分布式环境
        
        Args:
            backend: 通信后端 (nccl, gloo, mpi)
            init_method: 初始化方法 (env://, tcp://, file://)
            world_size: 进程总数
            rank: 当前进程rank
            timeout_minutes: 超时时间（分钟）
        """
        if self.state.initialized:
            logger.warning("Distributed already initialized")
            return
        
        self._init_time = time.time()
        
        # 从环境变量或参数获取配置
        backend = backend or self.config.backend
        if isinstance(backend, CommunicationBackend):
            backend = backend.value
        
        world_size = world_size or int(os.environ.get('WORLD_SIZE', self.config.world_size))
        rank = rank or int(os.environ.get('RANK', self.config.rank))
        local_rank = int(os.environ.get('LOCAL_RANK', self.config.local_rank))
        
        # 设置环境变量
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', self.config.master_addr)
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', self.config.master_port)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        
        # 初始化进程组
        if not dist.is_initialized():
            timeout = torch.distributed.timedelta(minutes=timeout_minutes)
            if init_method:
                dist.init_process_group(
                    backend=backend,
                    init_method=init_method,
                    world_size=world_size,
                    rank=rank,
                    timeout=timeout
                )
            else:
                dist.init_process_group(
                    backend=backend,
                    world_size=world_size,
                    rank=rank,
                    timeout=timeout
                )
        
        # 设置设备
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        # 更新状态
        self.state.initialized = True
        self.state.backend = backend
        self.state.world_size = world_size
        self.state.rank = rank
        self.state.local_rank = local_rank
        self.state.parallel_mode = self.config.mode
        
        # 计算节点信息
        self.state.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        if self.state.local_world_size > 0:
            self.state.num_nodes = world_size // self.state.local_world_size
            self.state.node_rank = rank // self.state.local_world_size
        
        # 初始化健康状态
        self.state.health_status = self._health_monitor.check_health(self.state)
        
        logger.info(f"Distributed initialized: backend={backend}, "
                   f"world_size={world_size}, rank={rank}, local_rank={local_rank}, "
                   f"init_time={time.time() - self._init_time:.3f}s")
    
    def cleanup(self) -> None:
        """清理分布式资源"""
        self._cleanup_time = time.time()
        
        # 清理进程组
        for group_name in list(self._process_groups.keys()):
            try:
                self.destroy_process_group(group_name)
            except Exception as e:
                logger.error(f"Failed to destroy process group {group_name}: {e}")
        
        # 清理包装器
        if self._ddp_wrapper:
            try:
                self._ddp_wrapper.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup DDP wrapper: {e}")
        
        if self._fsdp_wrapper:
            try:
                self._fsdp_wrapper.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup FSDP wrapper: {e}")
        
        if self._pipeline_wrapper:
            try:
                self._pipeline_wrapper.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup Pipeline wrapper: {e}")
        
        if self._zero_wrapper:
            try:
                self._zero_wrapper.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup ZeRO wrapper: {e}")
        
        # 销毁进程组
        if dist.is_initialized():
            try:
                dist.barrier()  # 确保所有进程同步
                dist.destroy_process_group()
            except Exception as e:
                logger.error(f"Failed to destroy process group: {e}")
        
        # 重置状态
        self.state = DistributedState()
        self._ddp_wrapper = None
        self._fsdp_wrapper = None
        self._pipeline_wrapper = None
        self._zero_wrapper = None
        
        logger.info(f"Distributed cleaned up, cleanup_time={time.time() - self._cleanup_time:.3f}s")
    
    # ==================== 进程组管理 ====================
    
    def create_process_group(
        self,
        config: ProcessGroupConfig
    ) -> Any:
        """
        创建自定义进程组
        
        Args:
            config: 进程组配置
            
        Returns:
            进程组对象
        """
        if not self.state.initialized:
            raise RuntimeError("Distributed not initialized")
        
        config.validate()
        
        if config.name in self._process_groups:
            logger.warning(f"Process group {config.name} already exists")
            return self._process_groups[config.name]
        
        backend = config.backend or self.state.backend
        timeout = torch.distributed.timedelta(seconds=config.timeout)
        
        group = dist.new_group(
            ranks=config.ranks,
            backend=backend,
            timeout=timeout
        )
        
        self._process_groups[config.name] = group
        self._process_group_configs[config.name] = config
        
        logger.info(f"Created process group: {config.name}, ranks={config.ranks}")
        return group
    
    def get_process_group(self, name: str) -> Optional[Any]:
        """获取进程组"""
        return self._process_groups.get(name)
    
    def destroy_process_group(self, name: str) -> None:
        """销毁进程组"""
        if name in self._process_groups:
            group = self._process_groups.pop(name)
            self._process_group_configs.pop(name, None)
            try:
                dist.destroy_process_group(group)
                logger.info(f"Destroyed process group: {name}")
            except Exception as e:
                logger.error(f"Failed to destroy process group {name}: {e}")
    
    def list_process_groups(self) -> List[str]:
        """列出所有进程组"""
        return list(self._process_groups.keys())
    
    # ==================== 健康检查 ====================
    
    def check_health(self) -> HealthStatus:
        """
        检查分布式环境健康状态
        
        Returns:
            健康状态
        """
        status = self._health_monitor.check_health(self.state)
        self.state.health_status = status
        self.state.last_health_check = time.time()
        return status
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要"""
        return self._health_monitor.get_health_summary()
    
    def is_healthy(self) -> bool:
        """检查是否健康"""
        return self.check_health() == HealthStatus.HEALTHY
    
    # ==================== 性能分析 ====================
    
    def enable_profiling(self) -> None:
        """启用通信性能分析"""
        self._comm_profiler.enable()
        logger.info("Communication profiling enabled")
    
    def disable_profiling(self) -> None:
        """禁用通信性能分析"""
        self._comm_profiler.disable()
        logger.info("Communication profiling disabled")
    
    def get_profiling_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取性能分析统计"""
        return self._comm_profiler.get_stats()
    
    def print_profiling_summary(self) -> None:
        """打印性能分析摘要"""
        self._comm_profiler.print_summary()
    
    def reset_profiling_stats(self) -> None:
        """重置性能分析统计"""
        self._comm_profiler.reset()
    
    def get_wrapper(self, mode: Optional[ParallelMode] = None) -> Optional[Any]:
        """
        获取对应的包装器
        
        Args:
            mode: 并行模式，如果为None则使用当前配置的模式
            
        Returns:
            对应的包装器实例
        """
        mode = mode or self.config.mode
        
        if mode == ParallelMode.DDP:
            return self._ddp_wrapper
        elif mode == ParallelMode.FSDP:
            return self._fsdp_wrapper
        elif mode == ParallelMode.PIPELINE:
            return self._pipeline_wrapper
        elif mode in (ParallelMode.ZERO_1, ParallelMode.ZERO_2, ParallelMode.ZERO_3):
            return self._zero_wrapper
        else:
            return None
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计信息
        
        Returns:
            内存统计字典
        """
        stats = {}
        
        if torch.cuda.is_available():
            stats['cuda_allocated_gb'] = torch.cuda.memory_allocated() / 1024**3
            stats['cuda_reserved_gb'] = torch.cuda.memory_reserved() / 1024**3
            stats['cuda_max_allocated_gb'] = torch.cuda.max_memory_allocated() / 1024**3
        
        # 获取包装器的内存统计
        wrapper = self.get_wrapper()
        if wrapper and hasattr(wrapper, 'get_memory_summary'):
            stats['wrapper_memory'] = wrapper.get_memory_summary()
        
        return stats
    
    def synchronize(self) -> None:
        """
        同步所有进程和设备
        """
        if self.state.world_size > 1 and dist.is_initialized():
            dist.barrier()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def gather_metrics(
        self,
        metrics: Dict[str, float],
        dst: int = 0
    ) -> Optional[List[Dict[str, float]]]:
        """
        收集所有进程的指标到指定进程
        
        Args:
            metrics: 本进程的指标字典
            dst: 目标进程rank
            
        Returns:
            如果是目标进程，返回所有进程的指标列表；否则返回None
        """
        if not self.state.initialized or self.state.world_size <= 1:
            return [metrics]
        
        # 序列化指标
        metrics_str = json.dumps(metrics)
        
        # 收集所有进程的指标
        gathered = [None] * self.state.world_size
        dist.all_gather_object(gathered, metrics_str)
        
        # 反序列化
        if self.state.rank == dst:
            return [json.loads(m) for m in gathered]
        return None
    
    def average_metrics(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        对所有进程的指标求平均
        
        Args:
            metrics: 本进程的指标字典
            
        Returns:
            平均后的指标字典
        """
        if not self.state.initialized or self.state.world_size <= 1:
            return metrics
        
        # 转换为tensor
        keys = sorted(metrics.keys())
        values = torch.tensor([metrics[k] for k in keys], dtype=torch.float32)
        
        if torch.cuda.is_available():
            values = values.cuda()
        
        # AllReduce求和
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        
        # 除以world_size
        values /= self.state.world_size
        
        # 转回字典
        averaged = {k: v.item() for k, v in zip(keys, values)}
        return averaged
    
    def wrap_model(
        self,
        model: nn.Module,
        mode: Optional[ParallelMode] = None,
        **kwargs
    ) -> nn.Module:
        """
        包装模型以支持分布式训练
        
        Args:
            model: 原始模型
            mode: 并行模式
            **kwargs: 额外配置
            
        Returns:
            包装后的模型
        """
        mode = mode or self.config.mode
        
        if mode == ParallelMode.NONE or self.state.world_size <= 1:
            return model
        
        if mode == ParallelMode.DDP:
            return self._wrap_ddp(model, **kwargs)
        elif mode == ParallelMode.FSDP:
            return self._wrap_fsdp(model, **kwargs)
        elif mode == ParallelMode.PIPELINE:
            return self._wrap_pipeline(model, **kwargs)
        elif mode in (ParallelMode.ZERO_1, ParallelMode.ZERO_2, ParallelMode.ZERO_3):
            return self._wrap_zero(model, **kwargs)
        elif mode == ParallelMode.HYBRID:
            return self._wrap_hybrid(model, **kwargs)
        else:
            logger.warning(f"Unknown parallel mode: {mode}, returning original model")
            return model
    
    def _wrap_ddp(self, model: nn.Module, **kwargs) -> nn.Module:
        """DDP包装"""
        from .ddp_wrapper import DDPWrapper
        
        config = DDPConfig(
            world_size=self.state.world_size,
            rank=self.state.rank,
            local_rank=self.state.local_rank,
            **kwargs
        )
        
        self._ddp_wrapper = DDPWrapper(config)
        return self._ddp_wrapper.wrap(model)
    
    def _wrap_fsdp(self, model: nn.Module, **kwargs) -> nn.Module:
        """FSDP包装"""
        from .fsdp_wrapper import FSDPWrapper
        
        config = FSDPConfig(
            world_size=self.state.world_size,
            rank=self.state.rank,
            local_rank=self.state.local_rank,
            **kwargs
        )
        
        self._fsdp_wrapper = FSDPWrapper(config)
        return self._fsdp_wrapper.wrap(model)
    
    def _wrap_pipeline(self, model: nn.Module, **kwargs) -> nn.Module:
        """Pipeline包装"""
        from .pipeline_wrapper import PipelineWrapper
        
        config = PipelineConfig(
            world_size=self.state.world_size,
            rank=self.state.rank,
            local_rank=self.state.local_rank,
            **kwargs
        )
        
        self._pipeline_wrapper = PipelineWrapper(config)
        stages = self._pipeline_wrapper.split_model(model)
        
        # 返回当前rank对应的阶段
        return self._pipeline_wrapper.get_current_stage()
    
    def _wrap_zero(self, model: nn.Module, **kwargs) -> nn.Module:
        """ZeRO包装"""
        from .zero_wrapper import ZeROWrapper, ZeROStage
        
        # 根据模式确定stage
        mode = self.config.mode
        if mode == ParallelMode.ZERO_1:
            stage = ZeROStage.STAGE_1
        elif mode == ParallelMode.ZERO_2:
            stage = ZeROStage.STAGE_2
        else:
            stage = ZeROStage.STAGE_3
        
        from .zero_wrapper import ZeROConfig as ZeroOptConfig
        config = ZeroOptConfig(stage=stage, **kwargs)
        
        self._zero_wrapper = ZeROWrapper(config)
        engine, _, _, _ = self._zero_wrapper.wrap(model)
        
        return engine
    
    def _wrap_hybrid(self, model: nn.Module, **kwargs) -> nn.Module:
        """混合并行包装"""
        # 混合并行通常需要更复杂的配置
        # 这里提供一个简化实现
        hybrid_config = HybridParallelConfig(**kwargs)
        
        if hybrid_config.data_parallel_size > 1:
            if hybrid_config.data_parallel_mode == ParallelMode.FSDP:
                model = self._wrap_fsdp(model)
            else:
                model = self._wrap_ddp(model)
        
        return model
    
    # ==================== 通信操作 ====================
    
    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        group: Optional[Any] = None,
        async_op: bool = False
    ) -> Union[torch.Tensor, Any]:
        """
        广播张量
        
        Args:
            tensor: 要广播的张量
            src: 源进程rank
            group: 进程组
            async_op: 是否异步操作
            
        Returns:
            广播后的张量或异步工作句柄
        """
        if not self.state.initialized or self.state.world_size <= 1:
            return tensor
        
        self._comm_profiler.start_op('broadcast')
        success = True
        
        try:
            work = dist.broadcast(tensor, src=src, group=group, async_op=async_op)
            self.state.total_communication_ops += 1
            num_bytes = tensor.numel() * tensor.element_size()
            self._comm_profiler.end_op('broadcast', num_bytes=num_bytes, success=True)
            return work if async_op else tensor
        except Exception as e:
            self.state.failed_communication_ops += 1
            self._comm_profiler.end_op('broadcast', success=False)
            logger.error(f"Broadcast failed: {e}")
            raise
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: AllReduceOp = AllReduceOp.SUM,
        group: Optional[Any] = None,
        async_op: bool = False
    ) -> Union[torch.Tensor, Any]:
        """
        AllReduce操作
        
        Args:
            tensor: 输入张量
            op: Reduce操作类型
            group: 进程组
            async_op: 是否异步操作
            
        Returns:
            Reduce后的张量或异步工作句柄
        """
        if not self.state.initialized or self.state.world_size <= 1:
            return tensor
        
        self._comm_profiler.start_op('all_reduce')
        
        try:
            torch_op = op.to_torch_op()
            work = dist.all_reduce(tensor, op=torch_op, group=group, async_op=async_op)
            self.state.total_communication_ops += 1
            num_bytes = tensor.numel() * tensor.element_size()
            self._comm_profiler.end_op('all_reduce', num_bytes=num_bytes, success=True)
            return work if async_op else tensor
        except Exception as e:
            self.state.failed_communication_ops += 1
            self._comm_profiler.end_op('all_reduce', success=False)
            logger.error(f"AllReduce failed: {e}")
            raise
    
    def all_gather(
        self,
        tensor: torch.Tensor,
        group: Optional[Any] = None,
        async_op: bool = False
    ) -> Union[List[torch.Tensor], Any]:
        """
        AllGather操作
        
        Args:
            tensor: 输入张量
            group: 进程组
            async_op: 是否异步操作
            
        Returns:
            收集的张量列表或异步工作句柄
        """
        if not self.state.initialized or self.state.world_size <= 1:
            return [tensor]
        
        self._comm_profiler.start_op('all_gather')
        
        try:
            world_size = group.size() if group else self.state.world_size
            tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            work = dist.all_gather(tensor_list, tensor, group=group, async_op=async_op)
            self.state.total_communication_ops += 1
            num_bytes = tensor.numel() * tensor.element_size() * world_size
            self._comm_profiler.end_op('all_gather', num_bytes=num_bytes, success=True)
            return work if async_op else tensor_list
        except Exception as e:
            self.state.failed_communication_ops += 1
            self._comm_profiler.end_op('all_gather', success=False)
            logger.error(f"AllGather failed: {e}")
            raise
    
    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op: AllReduceOp = AllReduceOp.SUM,
        group: Optional[Any] = None,
        async_op: bool = False
    ) -> Union[torch.Tensor, Any]:
        """
        ReduceScatter操作
        
        Args:
            output: 输出张量
            input_list: 输入张量列表
            op: Reduce操作类型
            group: 进程组
            async_op: 是否异步操作
            
        Returns:
            Reduce后的张量或异步工作句柄
        """
        if not self.state.initialized or self.state.world_size <= 1:
            return input_list[0] if input_list else output
        
        self._comm_profiler.start_op('reduce_scatter')
        
        try:
            torch_op = op.to_torch_op()
            work = dist.reduce_scatter(output, input_list, op=torch_op, group=group, async_op=async_op)
            self.state.total_communication_ops += 1
            num_bytes = sum(t.numel() * t.element_size() for t in input_list)
            self._comm_profiler.end_op('reduce_scatter', num_bytes=num_bytes, success=True)
            return work if async_op else output
        except Exception as e:
            self.state.failed_communication_ops += 1
            self._comm_profiler.end_op('reduce_scatter', success=False)
            logger.error(f"ReduceScatter failed: {e}")
            raise
    
    def barrier(self, group: Optional[Any] = None, async_op: bool = False) -> Optional[Any]:
        """
        同步屏障
        
        Args:
            group: 进程组
            async_op: 是否异步操作
            
        Returns:
            如果async_op=True，返回异步工作句柄；否则返回None
        """
        if self.state.initialized and self.state.world_size > 1:
            self._comm_profiler.start_op('barrier')
            start_time = time.time()
            
            try:
                work = dist.barrier(group=group, async_op=async_op)
                sync_time = time.time() - start_time
                self.state.total_sync_time += sync_time
                self.state.total_communication_ops += 1
                self._comm_profiler.end_op('barrier', success=True)
                return work if async_op else None
            except Exception as e:
                self.state.failed_communication_ops += 1
                self._comm_profiler.end_op('barrier', success=False)
                logger.error(f"Barrier failed: {e}")
                raise
        return None
    
    def all_gather_object(
        self,
        obj: Any,
        group: Optional[Any] = None
    ) -> List[Any]:
        """
        收集所有进程的Python对象
        
        Args:
            obj: 要收集的对象
            group: 进程组
            
        Returns:
            所有进程的对象列表
        """
        if not self.state.initialized or self.state.world_size <= 1:
            return [obj]
        
        self._comm_profiler.start_op('all_gather_object')
        
        try:
            world_size = group.size() if group else self.state.world_size
            object_list = [None] * world_size
            dist.all_gather_object(object_list, obj, group=group)
            self.state.total_communication_ops += 1
            self._comm_profiler.end_op('all_gather_object', success=True)
            return object_list
        except Exception as e:
            self.state.failed_communication_ops += 1
            self._comm_profiler.end_op('all_gather_object', success=False)
            logger.error(f"AllGatherObject failed: {e}")
            raise
    
    def scatter_object_list(
        self,
        output_list: List[Any],
        input_list: Optional[List[Any]] = None,
        src: int = 0,
        group: Optional[Any] = None
    ) -> None:
        """
        从源进程分发对象列表到所有进程
        
        Args:
            output_list: 输出对象列表
            input_list: 输入对象列表（仅源进程需要）
            src: 源进程rank
            group: 进程组
        """
        if not self.state.initialized or self.state.world_size <= 1:
            if input_list:
                output_list[:] = input_list
            return
        
        self._comm_profiler.start_op('scatter_object_list')
        
        try:
            dist.scatter_object_list(output_list, input_list, src=src, group=group)
            self.state.total_communication_ops += 1
            self._comm_profiler.end_op('scatter_object_list', success=True)
        except Exception as e:
            self.state.failed_communication_ops += 1
            self._comm_profiler.end_op('scatter_object_list', success=False)
            logger.error(f"ScatterObjectList failed: {e}")
            raise
    
    # ==================== 属性 ====================
    
    @property
    def is_initialized(self) -> bool:
        """是否已初始化"""
        return self.state.initialized
    
    @property
    def is_main_process(self) -> bool:
        """是否是主进程"""
        return self.state.rank == 0
    
    @property
    def world_size(self) -> int:
        """进程总数"""
        return self.state.world_size
    
    @property
    def rank(self) -> int:
        """当前进程rank"""
        return self.state.rank
    
    @property
    def local_rank(self) -> int:
        """本地rank"""
        return self.state.local_rank
    
    @property
    def device(self) -> torch.device:
        """当前设备"""
        if torch.cuda.is_available():
            return torch.device(f'cuda:{self.state.local_rank}')
        return torch.device('cpu')
    
    def get_info(self) -> Dict[str, Any]:
        """获取分布式信息"""
        info = {
            'initialized': self.state.initialized,
            'backend': self.state.backend,
            'world_size': self.state.world_size,
            'rank': self.state.rank,
            'local_rank': self.state.local_rank,
            'local_world_size': self.state.local_world_size,
            'num_nodes': self.state.num_nodes,
            'node_rank': self.state.node_rank,
            'device': str(self.device),
            'parallel_mode': self.state.parallel_mode.value if isinstance(self.state.parallel_mode, ParallelMode) else str(self.state.parallel_mode),
            'health_status': self.state.health_status.value,
            'total_communication_ops': self.state.total_communication_ops,
            'failed_communication_ops': self.state.failed_communication_ops,
            'total_sync_time': self.state.total_sync_time,
            'process_groups': list(self._process_groups.keys()),
        }
        
        # 添加包装器信息
        if self._ddp_wrapper:
            info['ddp_wrapper'] = 'initialized'
        if self._fsdp_wrapper:
            info['fsdp_wrapper'] = 'initialized'
        if self._pipeline_wrapper:
            info['pipeline_wrapper'] = 'initialized'
        if self._zero_wrapper:
            info['zero_wrapper'] = 'initialized'
        
        return info
    
    def diagnose(self) -> Dict[str, Any]:
        """
        诊断分布式环境
        
        Returns:
            诊断信息字典
        """
        diagnosis = {
            'status': 'unknown',
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # 检查初始化状态
        if not self.state.initialized:
            diagnosis['status'] = 'not_initialized'
            diagnosis['issues'].append("Distributed environment not initialized")
            return diagnosis
        
        # 检查进程组状态
        if not dist.is_initialized():
            diagnosis['status'] = 'unhealthy'
            diagnosis['issues'].append("PyTorch distributed process group not initialized")
        
        # 检查健康状态
        health = self.check_health()
        if health == HealthStatus.UNHEALTHY:
            diagnosis['status'] = 'unhealthy'
            diagnosis['issues'].append("Health check failed - communication issues detected")
        elif health == HealthStatus.DEGRADED:
            diagnosis['status'] = 'degraded'
            diagnosis['warnings'].append("Performance degradation detected - slow communication")
        else:
            diagnosis['status'] = 'healthy'
        
        # 检查失败的通信操作
        if self.state.failed_communication_ops > 0:
            failure_rate = self.state.failed_communication_ops / max(self.state.total_communication_ops, 1)
            if failure_rate > 0.01:  # 1%以上失败率
                diagnosis['warnings'].append(
                    f"High communication failure rate: {failure_rate:.2%} "
                    f"({self.state.failed_communication_ops}/{self.state.total_communication_ops})"
                )
        
        # 检查同步时间
        if self.state.total_communication_ops > 0:
            avg_sync_time = self.state.total_sync_time / self.state.total_communication_ops
            if avg_sync_time > 0.1:  # 平均同步时间超过100ms
                diagnosis['warnings'].append(
                    f"High average sync time: {avg_sync_time*1000:.1f}ms"
                )
                diagnosis['recommendations'].append(
                    "Consider reducing synchronization frequency or optimizing network"
                )
        
        # 检查CUDA可用性
        if not torch.cuda.is_available() and self.state.backend == 'nccl':
            diagnosis['warnings'].append("NCCL backend requires CUDA but CUDA is not available")
            diagnosis['recommendations'].append("Use 'gloo' backend for CPU-only training")
        
        # 检查world_size
        if self.state.world_size == 1:
            diagnosis['warnings'].append("Running with world_size=1, no actual parallelism")
            diagnosis['recommendations'].append("Increase world_size for distributed training benefits")
        
        return diagnosis
    
    def print_info(self) -> None:
        """打印分布式信息"""
        info = self.get_info()
        
        print("\n" + "="*80)
        print("Distributed Manager Information")
        print("="*80)
        
        print(f"\nStatus:")
        print(f"  Initialized: {info['initialized']}")
        print(f"  Backend: {info['backend']}")
        print(f"  Parallel Mode: {info['parallel_mode']}")
        print(f"  Health Status: {info['health_status']}")
        
        print(f"\nTopology:")
        print(f"  World Size: {info['world_size']}")
        print(f"  Rank: {info['rank']}")
        print(f"  Local Rank: {info['local_rank']}")
        print(f"  Local World Size: {info['local_world_size']}")
        print(f"  Num Nodes: {info['num_nodes']}")
        print(f"  Node Rank: {info['node_rank']}")
        print(f"  Device: {info['device']}")
        
        print(f"\nStatistics:")
        print(f"  Total Communication Ops: {info['total_communication_ops']}")
        print(f"  Failed Communication Ops: {info['failed_communication_ops']}")
        print(f"  Total Sync Time: {info['total_sync_time']:.3f}s")
        
        if info.get('process_groups'):
            print(f"\nProcess Groups: {', '.join(info['process_groups'])}")
        
        wrappers = []
        if 'ddp_wrapper' in info:
            wrappers.append('DDP')
        if 'fsdp_wrapper' in info:
            wrappers.append('FSDP')
        if 'pipeline_wrapper' in info:
            wrappers.append('Pipeline')
        if 'zero_wrapper' in info:
            wrappers.append('ZeRO')
        
        if wrappers:
            print(f"\nActive Wrappers: {', '.join(wrappers)}")
        
        print("="*80)
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diagnosis = self.diagnose()
        
        print("\n" + "="*80)
        print("Distributed Environment Diagnosis")
        print("="*80)
        
        print(f"\nOverall Status: {diagnosis['status'].upper()}")
        
        if diagnosis['issues']:
            print("\nIssues:")
            for issue in diagnosis['issues']:
                print(f"  - {issue}")
        
        if diagnosis['warnings']:
            print("\nWarnings:")
            for warning in diagnosis['warnings']:
                print(f"  - {warning}")
        
        if diagnosis['recommendations']:
            print("\nRecommendations:")
            for rec in diagnosis['recommendations']:
                print(f"  - {rec}")
        
        if not diagnosis['issues'] and not diagnosis['warnings']:
            print("\nNo issues detected")
        
        print("="*80)


# ==================== 全局函数 ====================

def get_distributed_manager() -> DistributedManager:
    """获取全局分布式管理器实例"""
    global _distributed_manager
    
    with _manager_lock:
        if _distributed_manager is None:
            _distributed_manager = DistributedManager()
        return _distributed_manager


def reset_distributed_manager() -> None:
    """重置全局分布式管理器"""
    global _distributed_manager
    
    with _manager_lock:
        if _distributed_manager is not None:
            _distributed_manager.cleanup()
        _distributed_manager = None


def init_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    config: Optional[DistributedConfig] = None
) -> DistributedManager:
    """
    初始化分布式环境
    
    便捷函数，初始化全局分布式管理器。
    
    Args:
        backend: 通信后端
        init_method: 初始化方法
        world_size: 进程总数
        rank: 当前进程rank
        config: 分布式配置（可选）
        
    Returns:
        分布式管理器实例
    """
    global _distributed_manager
    
    with _manager_lock:
        if _distributed_manager is None:
            _distributed_manager = DistributedManager(config)
    
    manager = _distributed_manager
    manager.initialize(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    return manager


def cleanup_distributed() -> None:
    """清理分布式资源"""
    global _distributed_manager
    
    with _manager_lock:
        if _distributed_manager is not None:
            _distributed_manager.cleanup()
            _distributed_manager = None


def is_distributed_initialized() -> bool:
    """检查分布式是否已初始化"""
    manager = get_distributed_manager()
    return manager.is_initialized


def get_rank() -> int:
    """获取当前进程rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """获取进程总数"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """获取本地rank"""
    return int(os.environ.get('LOCAL_RANK', 0))


def is_main_process() -> bool:
    """检查是否是主进程"""
    return get_rank() == 0


# ==================== 通信便捷函数 ====================

def broadcast(
    tensor: torch.Tensor,
    src: int = 0,
    group: Optional[Any] = None,
    async_op: bool = False
) -> Union[torch.Tensor, Any]:
    """广播张量"""
    manager = get_distributed_manager()
    return manager.broadcast(tensor, src, group, async_op)


def all_reduce(
    tensor: torch.Tensor,
    op: AllReduceOp = AllReduceOp.SUM,
    group: Optional[Any] = None,
    async_op: bool = False
) -> Union[torch.Tensor, Any]:
    """AllReduce操作"""
    manager = get_distributed_manager()
    return manager.all_reduce(tensor, op, group, async_op)


def all_gather(
    tensor: torch.Tensor,
    group: Optional[Any] = None,
    async_op: bool = False
) -> Union[List[torch.Tensor], Any]:
    """AllGather操作"""
    manager = get_distributed_manager()
    return manager.all_gather(tensor, group, async_op)


def reduce_scatter(
    output: torch.Tensor,
    input_list: List[torch.Tensor],
    op: AllReduceOp = AllReduceOp.SUM,
    group: Optional[Any] = None,
    async_op: bool = False
) -> Union[torch.Tensor, Any]:
    """ReduceScatter操作"""
    manager = get_distributed_manager()
    return manager.reduce_scatter(output, input_list, op, group, async_op)


def barrier(group: Optional[Any] = None) -> None:
    """同步屏障"""
    manager = get_distributed_manager()
    manager.barrier(group)


def synchronize() -> None:
    """同步所有进程和设备"""
    manager = get_distributed_manager()
    manager.synchronize()


def gather_metrics(
    metrics: Dict[str, float],
    dst: int = 0
) -> Optional[List[Dict[str, float]]]:
    """收集所有进程的指标"""
    manager = get_distributed_manager()
    return manager.gather_metrics(metrics, dst)


def average_metrics(
    metrics: Dict[str, float]
) -> Dict[str, float]:
    """对所有进程的指标求平均"""
    manager = get_distributed_manager()
    return manager.average_metrics(metrics)


def all_gather_object(
    obj: Any,
    group: Optional[Any] = None
) -> List[Any]:
    """收集所有进程的Python对象"""
    manager = get_distributed_manager()
    return manager.all_gather_object(obj, group)


# ==================== 装饰器 ====================

def main_process_only(func):
    """装饰器：仅在主进程执行"""
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
        return None
    return wrapper


def synchronized(func):
    """装饰器：在函数前后添加barrier同步"""
    def wrapper(*args, **kwargs):
        if is_distributed_initialized():
            barrier()
        result = func(*args, **kwargs)
        if is_distributed_initialized():
            barrier()
        return result
    return wrapper


@contextmanager
def distributed_context(
    backend: str = "nccl",
    config: Optional[DistributedConfig] = None,
    **kwargs
):
    """
    分布式上下文管理器
    
    用法:
    ```python
    with distributed_context() as manager:
        model = manager.wrap_model(model)
        # 训练...
    ```
    
    Args:
        backend: 通信后端
        config: 分布式配置
        **kwargs: 其他初始化参数
    """
    manager = init_distributed(backend=backend, config=config, **kwargs)
    try:
        yield manager
    finally:
        cleanup_distributed()


@contextmanager
def managed_distributed(
    backend: str = "nccl",
    enable_profiling: bool = False,
    check_health: bool = True
):
    """
    托管的分布式上下文管理器
    
    提供自动的健康检查和性能分析
    
    Args:
        backend: 通信后端
        enable_profiling: 是否启用性能分析
        check_health: 是否检查健康状态
    """
    manager = init_distributed(backend=backend)
    
    if enable_profiling:
        manager.enable_profiling()
    
    if check_health:
        status = manager.check_health()
        if status != HealthStatus.HEALTHY:
            logger.warning(f"Distributed environment health status: {status.value}")
    
    try:
        yield manager
    finally:
        if enable_profiling:
            manager.print_profiling_summary()
            manager.disable_profiling()
        
        cleanup_distributed()


@contextmanager
def only_on_main_process():
    """
    上下文管理器：仅在主进程执行代码块
    
    用法:
    ```python
    with only_on_main_process():
        print("Only main process prints this")
        save_model(model)
    ```
    """
    should_execute = is_main_process()
    if should_execute:
        yield
    else:
        # 非主进程等待
        if is_distributed_initialized():
            barrier()


# ==================== 辅助工具函数 ====================

def print_on_main(message: str, *args, **kwargs) -> None:
    """仅在主进程打印消息"""
    if is_main_process():
        print(message, *args, **kwargs)


def save_on_main(save_func, *args, **kwargs) -> Any:
    """
    仅在主进程执行保存操作
    
    Args:
        save_func: 保存函数
        *args, **kwargs: 传递给保存函数的参数
        
    Returns:
        保存函数的返回值（仅主进程），其他进程返回None
    """
    result = None
    if is_main_process():
        result = save_func(*args, **kwargs)
    
    # 确保所有进程同步
    if is_distributed_initialized():
        barrier()
    
    return result


def auto_select_backend() -> str:
    """
    自动选择合适的通信后端
    
    Returns:
        推荐的后端名称
    """
    if torch.cuda.is_available():
        # CUDA可用，使用NCCL
        return "nccl"
    else:
        # CPU训练，使用GLOO
        return "gloo"


def estimate_communication_cost(
    tensor_size: int,
    world_size: int,
    operation: str = "all_reduce"
) -> Dict[str, float]:
    """
    估算通信开销
    
    Args:
        tensor_size: 张量大小（字节）
        world_size: 进程总数
        operation: 通信操作类型
        
    Returns:
        估算的开销字典
    """
    # 简化的通信开销模型
    # 实际开销取决于网络带宽、延迟等因素
    
    alpha = 10e-6  # 延迟（秒）
    beta = 1e-9    # 每字节传输时间（秒/字节）
    
    if operation == "broadcast":
        # 广播：log(P) * (alpha + beta * M)
        import math
        cost = math.log2(world_size) * (alpha + beta * tensor_size)
    elif operation == "all_reduce":
        # AllReduce (Ring算法): 2 * (P-1)/P * (alpha + beta * M)
        cost = 2 * (world_size - 1) / world_size * (alpha + beta * tensor_size)
    elif operation == "all_gather":
        # AllGather: (P-1) * (alpha + beta * M)
        cost = (world_size - 1) * (alpha + beta * tensor_size)
    elif operation == "reduce_scatter":
        # ReduceScatter: (P-1) * (alpha + beta * M/P)
        cost = (world_size - 1) * (alpha + beta * tensor_size / world_size)
    else:
        cost = 0.0
    
    return {
        'estimated_time_ms': cost * 1000,
        'tensor_size_mb': tensor_size / (1024**2),
        'world_size': world_size,
        'operation': operation
    }


def get_optimal_bucket_size(
    model_size_mb: float,
    world_size: int,
    bandwidth_gbps: float = 10.0
) -> int:
    """
    获取最优的DDP bucket大小
    
    Args:
        model_size_mb: 模型大小（MB）
        world_size: 进程总数
        bandwidth_gbps: 网络带宽（Gbps）
        
    Returns:
        推荐的bucket大小（MB）
    """
    # 根据模型大小和网络带宽推荐bucket大小
    # 一般原则：bucket越大，通信效率越高，但内存占用也越大
    
    if model_size_mb < 100:
        return 10  # 小模型用小bucket
    elif model_size_mb < 1000:
        return 25  # 中等模型
    else:
        return 50  # 大模型用大bucket


def create_custom_process_group(
    ranks: List[int],
    name: str,
    backend: Optional[str] = None
) -> Any:
    """
    创建自定义进程组
    
    Args:
        ranks: 进程rank列表
        name: 进程组名称
        backend: 通信后端
        
    Returns:
        进程组对象
    """
    manager = get_distributed_manager()
    config = ProcessGroupConfig(
        name=name,
        ranks=ranks,
        backend=backend
    )
    return manager.create_process_group(config)


def log_distributed_metrics(
    metrics: Dict[str, float],
    step: int,
    logger_func: Optional[callable] = None
) -> None:
    """
    记录分布式训练指标
    
    Args:
        metrics: 指标字典
        step: 训练步数
        logger_func: 日志记录函数（可选）
    """
    if not is_main_process():
        return
    
    # 平均所有进程的指标
    averaged = average_metrics(metrics)
    
    # 记录日志
    if logger_func:
        for key, value in averaged.items():
            logger_func(key, value, step)
    else:
        logger.info(f"Step {step}: {averaged}")


def check_distributed_consistency(
    tensor: torch.Tensor,
    tolerance: float = 1e-5
) -> bool:
    """
    检查所有进程的张量是否一致
    
    Args:
        tensor: 要检查的张量
        tolerance: 容差
        
    Returns:
        是否一致
    """
    if not is_distributed_initialized() or get_world_size() == 1:
        return True
    
    # 收集所有进程的张量
    gathered = all_gather(tensor.clone())
    
    # 检查是否一致
    reference = gathered[0]
    for t in gathered[1:]:
        if not torch.allclose(t, reference, atol=tolerance):
            return False
    
    return True


def synchronize_random_seed(seed: Optional[int] = None) -> int:
    """
    同步所有进程的随机种子
    
    Args:
        seed: 随机种子（主进程提供）
        
    Returns:
        同步后的种子
    """
    if not is_distributed_initialized():
        return seed if seed is not None else 42
    
    if seed is None and is_main_process():
        import random
        seed = random.randint(0, 2**32 - 1)
    
    # 广播种子
    seed_tensor = torch.tensor([seed if is_main_process() else 0], dtype=torch.long)
    if torch.cuda.is_available():
        seed_tensor = seed_tensor.cuda()
    
    broadcast(seed_tensor, src=0)
    
    return int(seed_tensor.item())


def get_distributed_sampler(
    dataset,
    shuffle: bool = True,
    seed: int = 0
):
    """
    获取分布式数据采样器
    
    Args:
        dataset: 数据集
        shuffle: 是否打乱
        seed: 随机种子
        
    Returns:
        分布式采样器
    """
    from torch.utils.data import DistributedSampler
    
    if not is_distributed_initialized() or get_world_size() == 1:
        return None
    
    return DistributedSampler(
        dataset,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        seed=seed
    )

# -*- coding: utf-8 -*-
"""
生产级训练策略基类

整合六层架构的底层能力，提供生产级的训练策略基础：
- backend/lib/hardware: 硬件抽象层
  - DeviceManager, MixedPrecisionManager, MemoryManager, GradientCheckpointing
- backend/lib/distributed: 分布式训练内核层
  - DistributedManager, DDPWrapper, FSDPWrapper, ZeROWrapper, PipelineWrapper
- backend/lib/adapters: 模型/模态适配器层
  - EncoderFactory, FusionFactory, AlignmentFactory, AdapterFactory
- backend/lib/losses: 目标函数层
  - LossFactory, CompositeLoss, MultiTaskLoss, ContrastiveLoss
- base_strategy.py: 策略基础组件
  - StrategyMonitor, StrategyProfiler, StrategyValidator, StrategyMetrics

架构图：
┌──────────────────────────────────────┐
│          Training Orchestrator       │
├──────────────────────────────────────┤
│ >>> Training Strategy (当前层) <<<   │
│    (调用下层提供生产级能力)           │
├──────────────────────────────────────┤
│      Loss & Objective (lib/losses)   │
├──────────────────────────────────────┤
│   Model & Modality (lib/adapters)    │
├──────────────────────────────────────┤
│   Distributed Core (lib/distributed) │
├──────────────────────────────────────┤
│   Hardware (lib/hardware)            │
└──────────────────────────────────────┘

生产级特性：
- 完整的监控和诊断能力
- 自动设备选择和内存优化
- 分布式训练封装器管理
- 灵活的损失函数组合
- 健康检查和故障恢复
"""

import logging
import time
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
# 导入 nullcontext 并重命名为 _nullcontext 避免重定义警告
from contextlib import nullcontext as _nullcontext
from enum import Enum

import torch
import torch.nn as nn

from .base_strategy import (
    TrainingStrategy, StrategyContext, StrategyResult, TrainingPhase,
    StrategyType, StrategyMonitor, StrategyProfiler, StrategyValidator, StrategyMetrics,
)

logger = logging.getLogger(__name__)


# ==================== 底层模块导入 ====================

# 硬件抽象层
from backend.lib.hardware import (
    DeviceManager, get_device_manager,
    MixedPrecisionManager, AmpContext, PrecisionMode,
    MemoryManager, GradientCheckpointing, clear_memory,
    get_available_memory, MemoryStats,
)

# 分布式训练内核层
from backend.lib.distributed import (
    DistributedManager, get_distributed_manager, ParallelMode,
    DDPWrapper, FSDPWrapper, ZeROWrapper, PipelineWrapper,
    DDPConfig, FSDPConfig, ZeROConfig, PipelineConfig,
    barrier, all_reduce, AllReduceOp,
    is_main_process, get_rank, get_world_size,
)

# 模型/模态适配器层
from backend.lib.adapters import (
    AdapterManager, get_adapter_manager,
    ModalityEncoder, EncoderFactory, create_encoder,
    FusionModule, FusionFactory, create_fusion,
    AlignmentModule, AlignmentFactory, create_alignment,
    ModelAdapter, AdapterFactory, create_adapter,
    ModalityType, FusionMethod, AlignmentMethod, AdapterType
)

# 目标函数层
from backend.lib.losses import (
    LossFactory, create_loss, create_composite_loss, create_distillation_loss,
    BaseLoss, CompositeLoss, MultiTaskLoss,
    CrossEntropyLoss, FocalLoss, MSELoss,
    SoftLabelLoss, FeatureDistillationLoss, AttentionDistillationLoss,
    InfoNCELoss, CLIPLoss, ContrastiveLoss,
    LossMonitor, LossStats,
)


# ==================== 新增数据类 ====================

@dataclass
class ProductionHealthStatus:
    """生产级健康状态"""
    is_healthy: bool = True
    hardware_ok: bool = True
    distributed_ok: bool = True
    adapters_ok: bool = True
    losses_ok: bool = True
    memory_pressure: str = "low"
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
            'hardware_ok': self.hardware_ok,
            'distributed_ok': self.distributed_ok,
            'adapters_ok': self.adapters_ok,
            'losses_ok': self.losses_ok,
            'memory_pressure': self.memory_pressure,
            'last_check_time': self.last_check_time,
            'issues': self.issues.copy(),
        }


@dataclass
class WrapperStats:
    """分布式包装器统计"""
    wrapper_type: str = "none"
    is_active: bool = False
    total_syncs: int = 0
    total_all_reduces: int = 0
    sync_time_ms: float = 0.0
    
    def record_sync(self, time_ms: float) -> None:
        """记录同步"""
        self.total_syncs += 1
        self.sync_time_ms += time_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'wrapper_type': self.wrapper_type,
            'is_active': self.is_active,
            'total_syncs': self.total_syncs,
            'total_all_reduces': self.total_all_reduces,
            'sync_time_ms': self.sync_time_ms,
            'avg_sync_time_ms': self.sync_time_ms / max(self.total_syncs, 1),
        }


@dataclass
class ProductionStrategyConfig:
    """生产级策略配置"""
    # 硬件配置
    device: str = "auto"  # auto, cuda, cpu, mps
    precision: str = "fp16"  # fp32, fp16, bf16
    enable_amp: bool = True
    enable_gradient_checkpointing: bool = False
    memory_optimization: bool = True
    
    # 分布式配置
    distributed_mode: str = "none"  # none, ddp, fsdp, zero, zero1, zero2, zero3, pipeline
    world_size: int = 1
    gradient_accumulation_steps: int = 1
    sync_bn: bool = True
    find_unused_parameters: bool = False
    static_graph: bool = False
    
    # 模态配置
    modalities: List[str] = field(default_factory=lambda: ["text"])
    hidden_size: int = 768
    fusion_method: str = "cross_attention"
    alignment_method: str = "contrastive"
    
    # 适配器配置
    adapter_type: Optional[str] = None  # lora, prefix, prompt
    lora_rank: int = 8
    adapter_alpha: float = 16.0
    
    # 损失配置
    task_loss_type: str = "cross_entropy"
    auxiliary_losses: List[Dict[str, Any]] = field(default_factory=list)
    contrastive_temperature: float = 0.07
    distillation_temperature: float = 4.0
    
    # 优化配置
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 监控配置
    enable_monitoring: bool = True
    enable_profiling: bool = False
    health_check_interval: int = 100
    log_interval: int = 10
    
    def validate(self) -> None:
        """验证配置"""
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be >= 0")
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'device': self.device,
            'precision': self.precision,
            'enable_amp': self.enable_amp,
            'enable_gradient_checkpointing': self.enable_gradient_checkpointing,
            'memory_optimization': self.memory_optimization,
            'distributed_mode': self.distributed_mode,
            'world_size': self.world_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'modalities': self.modalities.copy(),
            'hidden_size': self.hidden_size,
            'fusion_method': self.fusion_method,
            'alignment_method': self.alignment_method,
            'adapter_type': self.adapter_type,
            'task_loss_type': self.task_loss_type,
            'learning_rate': self.learning_rate,
            'max_grad_norm': self.max_grad_norm,
            'enable_monitoring': self.enable_monitoring,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductionStrategyConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def get_ddp_config(self) -> Optional[Dict[str, Any]]:
        """获取 DDP 配置（使用 DDPConfig）"""
        if DDPConfig is None:
            return None
        return {
            'find_unused_parameters': self.find_unused_parameters,
            'static_graph': self.static_graph,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
        }
    
    def get_fsdp_config(self) -> Optional[Dict[str, Any]]:
        """获取 FSDP 配置（使用 FSDPConfig）"""
        if FSDPConfig is None:
            return None
        return {
            'sharding_strategy': 'FULL_SHARD',
            'mixed_precision': self.precision != 'fp32',
        }
    
    def get_zero_config(self) -> Optional[Dict[str, Any]]:
        """获取 ZeRO 配置（使用 ZeROConfig）"""
        if ZeROConfig is None:
            return None
        stage = 2
        if self.distributed_mode == 'zero1':
            stage = 1
        elif self.distributed_mode == 'zero3':
            stage = 3
        return {
            'stage': stage,
            'offload_optimizer': stage >= 2,
            'offload_param': stage >= 3,
        }
    
    def get_pipeline_config(self) -> Optional[Dict[str, Any]]:
        """获取 Pipeline 配置（使用 PipelineConfig）"""
        if PipelineConfig is None:
            return None
        return {
            'num_stages': self.world_size,
            'micro_batch_size': 1,
        }
    
    def summary(self) -> str:
        """获取配置摘要"""
        return (
            f"ProductionConfig(device={self.device}, precision={self.precision}, "
            f"distributed={self.distributed_mode}, modalities={self.modalities})"
        )


class ProductionTrainingStrategy(TrainingStrategy):
    """
    生产级训练策略
    
    整合六层架构的所有底层能力，提供生产级训练支持：
    - 自动设备检测和管理 (DeviceManager, MemoryManager, GradientCheckpointing)
    - 混合精度训练 (MixedPrecisionManager, AmpContext, PrecisionMode)
    - 分布式训练支持 (DDPWrapper, FSDPWrapper, ZeROWrapper, PipelineWrapper)
    - 多模态编码/融合/对齐 (EncoderFactory, FusionFactory, AlignmentFactory)
    - 灵活的损失函数组合 (LossFactory, CompositeLoss, MultiTaskLoss)
    - 参数高效微调 (AdapterFactory, AdapterType)
    - 策略监控和诊断 (StrategyMonitor, StrategyProfiler, StrategyValidator)
    
    使用的底层模块：
    - backend/lib/hardware: DeviceManager, MixedPrecisionManager, MemoryManager, 
                           GradientCheckpointing, AmpContext, PrecisionMode, clear_memory
    - backend/lib/distributed: DistributedManager, DDPWrapper, FSDPWrapper, ZeROWrapper, 
                              PipelineWrapper, DDPConfig, FSDPConfig, ZeROConfig, PipelineConfig,
                              barrier, all_reduce, AllReduceOp, ParallelMode
    - backend/lib/adapters: EncoderFactory, FusionFactory, AlignmentFactory, AdapterFactory,
                           ModalityType, FusionMethod, AlignmentMethod, AdapterType
    - backend/lib/losses: LossFactory, CompositeLoss, MultiTaskLoss, CrossEntropyLoss, FocalLoss,
                         MSELoss, SoftLabelLoss, FeatureDistillationLoss, InfoNCELoss, CLIPLoss
    - base_strategy.py: StrategyMonitor, StrategyProfiler, StrategyValidator, StrategyMetrics
    """
    
    # 策略类型
    STRATEGY_TYPE = StrategyType.PRODUCTION
    
    def __init__(
        self,
        config: Optional[ProductionStrategyConfig] = None,
        name: str = "production",
        priority: int = 50
    ):
        super().__init__(name=name, priority=priority)
        self.config = config or ProductionStrategyConfig()
        
        # 验证配置
        try:
            self.config.validate()
        except ValueError as e:
            logger.warning(f"Config validation warning: {e}")
        
        # 底层管理器 (backend/lib/hardware)
        self._device_manager: Optional['DeviceManager'] = None
        self._memory_manager: Optional['MemoryManager'] = None
        self._amp_manager: Optional['MixedPrecisionManager'] = None
        self._gradient_checkpointing: Optional['GradientCheckpointing'] = None
        
        # 底层管理器 (backend/lib/distributed)
        self._distributed_manager: Optional['DistributedManager'] = None
        self._distributed_wrapper: Optional[nn.Module] = None  # DDPWrapper/FSDPWrapper/ZeROWrapper/PipelineWrapper
        self._wrapper_stats = WrapperStats()
        
        # 底层管理器 (backend/lib/adapters)
        self._adapter_manager: Optional['AdapterManager'] = None
        
        # 编码器组件 (使用 EncoderFactory, ModalityType)
        self._encoders: Dict[str, 'ModalityEncoder'] = {}
        
        # 融合组件 (使用 FusionFactory, FusionMethod)
        self._fusion: Optional['FusionModule'] = None
        
        # 对齐组件 (使用 AlignmentFactory, AlignmentMethod)
        self._alignment: Optional['AlignmentModule'] = None
        
        # 损失组件 (使用 LossFactory, CompositeLoss, MultiTaskLoss)
        self._loss_fn: Optional[nn.Module] = None
        self._contrastive_loss: Optional[nn.Module] = None
        self._distillation_loss: Optional[nn.Module] = None
        self._loss_monitor: Optional['LossMonitor'] = None
        
        # 适配器组件 (使用 AdapterFactory, AdapterType)
        self._adapter: Optional['ModelAdapter'] = None
        
        # 基础策略组件 (base_strategy.py)
        self._strategy_monitor: Optional[StrategyMonitor] = None
        self._strategy_profiler: Optional[StrategyProfiler] = None
        self._strategy_validator: Optional[StrategyValidator] = None
        self._strategy_metrics: Optional[StrategyMetrics] = None
        
        # 状态
        self._device: torch.device = torch.device('cpu')
        self._is_distributed: bool = False
        self._is_setup: bool = False
        self._health_status = ProductionHealthStatus()
        self._current_phase: TrainingPhase = TrainingPhase.WARMUP
    
    def setup(self, context: StrategyContext) -> None:
        """初始化生产级策略"""
        super().setup(context)
        
        if self._is_setup:
            return
        
        logger.info(f"Setting up ProductionTrainingStrategy: {self.name}")
        
        # 0. 初始化基础策略组件 (base_strategy.py)
        self._init_base_strategy_components()
        
        # 1. 初始化硬件层 (backend/lib/hardware)
        self._setup_hardware()
        
        # 2. 初始化分布式层 (backend/lib/distributed)
        self._setup_distributed()
        
        # 3. 初始化适配器层 (backend/lib/adapters)
        self._setup_adapters()
        
        # 4. 初始化损失层 (backend/lib/losses)
        self._setup_losses()
        
        # 5. 应用适配器到模型
        if context.model is not None:
            self._apply_adapter(context.model)
            context.model = context.model.to(self._device)
            
            # 分布式包装 (使用 DDPWrapper/FSDPWrapper/ZeROWrapper/PipelineWrapper)
            if self._is_distributed and self._distributed_manager:
                context.model = self._wrap_model_distributed(context.model)
        
        # 6. 初始健康检查
        self._check_health()
        
        self._is_setup = True
        logger.info(f"ProductionTrainingStrategy setup completed: device={self._device}, "
                   f"distributed={self._is_distributed}")
    
    def _init_base_strategy_components(self) -> None:
        """
        初始化基础策略组件
        
        使用 base_strategy.py: StrategyMonitor, StrategyProfiler, StrategyValidator, StrategyMetrics
        """
        # 初始化策略监控器
        if self.config.enable_monitoring:
            try:
                self._strategy_monitor = StrategyMonitor(history_size=10000)
            except Exception as e:
                logger.warning(f"Failed to init StrategyMonitor: {e}")
        
        # 初始化性能分析器
        if self.config.enable_profiling:
            try:
                self._strategy_profiler = StrategyProfiler()
            except Exception as e:
                logger.warning(f"Failed to init StrategyProfiler: {e}")
        
        # 初始化验证器
        try:
            self._strategy_validator = StrategyValidator()
            self._add_production_validation_rules()
        except Exception as e:
            logger.warning(f"Failed to init StrategyValidator: {e}")
        
        # 初始化指标跟踪
        try:
            self._strategy_metrics = StrategyMetrics()
        except Exception as e:
            logger.warning(f"Failed to init StrategyMetrics: {e}")
        
        logger.debug("Base strategy components initialized")
    
    def _add_production_validation_rules(self) -> None:
        """添加生产级验证规则"""
        if self._strategy_validator is None:
            return
        
        if hasattr(self._strategy_validator, 'add_check'):
            # 检查健康状态
            def check_health(result: StrategyResult) -> Tuple[bool, str]:
                if not self._health_status.is_healthy:
                    return False, f"Unhealthy: {self._health_status.issues}"
                return True, ""
            
            self._strategy_validator.add_check(check_health)
    
    def _setup_hardware(self) -> None:
        """
        设置硬件层
        
        使用: DeviceManager, get_device_manager, MixedPrecisionManager, AmpContext, 
              PrecisionMode, MemoryManager, GradientCheckpointing, clear_memory, get_available_memory
        """
        # 获取设备管理器 (DeviceManager)
        if get_device_manager is not None:
            try:
                self._device_manager = get_device_manager()
            except Exception as e:
                logger.warning(f"Failed to get device manager: {e}")
        
        # 自动选择设备
        if self.config.device == "auto":
            if self._device_manager is not None:
                self._device = self._device_manager.get_device()
            else:
                self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(self.config.device)
        
        # 设置混合精度 (MixedPrecisionManager, AmpContext, PrecisionMode)
        if self.config.enable_amp and MixedPrecisionManager is not None and PrecisionMode is not None:
            try:
                from backend.lib.hardware.mixed_precision import AmpConfig
                precision_map = {
                'fp32': PrecisionMode.FP32,
                'fp16': PrecisionMode.MIXED_FP16,
                'bf16': PrecisionMode.MIXED_BF16
                }
                amp_config = AmpConfig(
                    enabled=True,
                    precision=precision_map.get(self.config.precision, PrecisionMode.MIXED_FP16)
                )
                self._amp_manager = MixedPrecisionManager(amp_config, self._device)
            except Exception as e:
                logger.warning(f"Failed to init MixedPrecisionManager: {e}")
                return
        
        # 设置内存管理器 (MemoryManager)
        if self.config.memory_optimization and MemoryManager is not None:
            try:
                self._memory_manager = MemoryManager(device=self._device)
            except Exception as e:
                logger.warning(f"Failed to init MemoryManager: {e}")
        
        # 设置梯度检查点 (GradientCheckpointing)
        # 注意：GradientCheckpointing 需要 model 参数，但此时 model 可能还未创建
        # 所以这里先不初始化，等到 setup 时再初始化
        self._gradient_checkpointing = None
        
        # 检查可用内存 (get_available_memory)
        if get_available_memory is not None:
            try:
                available = get_available_memory()
                logger.debug(f"Available memory: {available}")
            except Exception:
                pass
        
        logger.info(f"Hardware layer initialized: device={self._device}, precision={self.config.precision}, "
                   f"amp={self._amp_manager is not None}, memory_mgr={self._memory_manager is not None}, "
                   f"grad_ckpt={self._gradient_checkpointing is not None}")
    
    def _setup_distributed(self) -> None:
        """
        设置分布式层
        
        使用: DistributedManager, get_distributed_manager, ParallelMode,
              DDPWrapper, FSDPWrapper, ZeROWrapper, PipelineWrapper,
              DDPConfig, FSDPConfig, ZeROConfig, PipelineConfig,
              barrier, all_reduce, AllReduceOp, is_main_process, get_rank, get_world_size
        """
        if self.config.distributed_mode == "none":
            return
        
        if self.config.world_size <= 1:
            return
        
        # 获取分布式管理器
        if get_distributed_manager is not None:
            try:
                self._distributed_manager = get_distributed_manager()
                self._distributed_manager.initialize(
                    backend='nccl' if torch.cuda.is_available() else 'gloo',
                    world_size=self.config.world_size
                )
                self._is_distributed = True
            except Exception as e:
                logger.warning(f"Failed to init DistributedManager: {e}")
                return
        
        # 记录包装器类型
        self._wrapper_stats.wrapper_type = self.config.distributed_mode
        self._wrapper_stats.is_active = True
        
        # 获取分布式信息 (使用 is_main_process, get_rank, get_world_size)
        dist_info = self._get_distributed_info()
        
        logger.info(f"Distributed layer initialized: mode={self.config.distributed_mode}, "
                   f"rank={dist_info.get('rank', 0)}/{dist_info.get('world_size', 1)}, "
                   f"is_main={dist_info.get('is_main', True)}")
    
    def _wrap_model_distributed(self, model: nn.Module) -> nn.Module:
        """
        包装模型以支持分布式训练
        
        使用: DDPWrapper, FSDPWrapper, ZeROWrapper, PipelineWrapper,
              DDPConfig, FSDPConfig, ZeROConfig, PipelineConfig, ParallelMode
        """
        mode = self.config.distributed_mode
        
        # 1. 优先使用 DistributedManager 的 wrap_model
        if self._distributed_manager is not None:
            try:
                parallel_mode = self._get_parallel_mode()
                wrapped = self._distributed_manager.wrap_model(model, mode=parallel_mode)
                self._distributed_wrapper = wrapped
                return wrapped
            except Exception as e:
                logger.warning(f"DistributedManager wrap failed: {e}")
        
        # 2. 使用特定的 Wrapper 类
        if mode == 'ddp' and DDPWrapper is not None:
            try:
                config = self._get_ddp_config()
                if config is not None:
                    self._distributed_wrapper = DDPWrapper(config)
                    wrapped_model = self._distributed_wrapper.wrap(model)
                    return wrapped_model
            except Exception as e:
                logger.warning("DDPWrapper failed: %s", e)
        
        elif mode == 'fsdp' and FSDPWrapper is not None:
            try:
                config = self._get_fsdp_config()
                if config is not None:
                    self._distributed_wrapper = FSDPWrapper(config)
                    wrapped_model = self._distributed_wrapper.wrap(model)
                    return wrapped_model
            except Exception as e:
                logger.warning("FSDPWrapper failed: %s", e)
        
        elif mode.startswith('zero') and ZeROWrapper is not None:
            try:
                config = self._get_zero_config()
                if config is not None:
                    self._distributed_wrapper = ZeROWrapper(config)
                    engine, _, _, _ = self._distributed_wrapper.wrap(model)
                    return engine
            except Exception as e:
                logger.warning("ZeROWrapper failed: %s", e)
        
        elif mode == 'pipeline' and PipelineWrapper is not None:
            try:
                config = self._get_pipeline_config()
                if config is not None:
                    self._distributed_wrapper = PipelineWrapper(config)
                    self._distributed_wrapper.split_model(model)
                    return self._distributed_wrapper.get_current_stage()
            except Exception as e:
                logger.warning("PipelineWrapper failed: %s", e)
        
        # 3. 回退到原生 DDP
        if torch.distributed.is_initialized():
            return nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self._device.index] if self._device.type == 'cuda' else None,
                find_unused_parameters=self.config.find_unused_parameters
            )
        
        return model
    
    def _get_ddp_config(self) -> Optional['DDPConfig']:
        """获取 DDP 配置"""
        if DDPConfig is None:
            return None
        try:
            return DDPConfig(
                find_unused_parameters=self.config.find_unused_parameters,
                static_graph=self.config.static_graph,
            )
        except Exception:
            return None
    
    def _get_fsdp_config(self) -> Optional['FSDPConfig']:
        """获取 FSDP 配置"""
        if FSDPConfig is None:
            return None
        try:
            return FSDPConfig(
                sharding_strategy='FULL_SHARD',
                mixed_precision=self.config.precision != 'fp32',
            )
        except Exception:
            return None
    
    def _get_zero_config(self) -> Optional['ZeROConfig']:
        """获取 ZeRO 配置"""
        if ZeROConfig is None:
            return None
        try:
            stage = 2
            if self.config.distributed_mode == 'zero1':
                stage = 1
            elif self.config.distributed_mode == 'zero3':
                stage = 3
            return ZeROConfig(
                stage=stage,
                offload_optimizer=stage >= 2,
                offload_param=stage >= 3,
            )
        except Exception:
            return None
    
    def _get_pipeline_config(self) -> Optional['PipelineConfig']:
        """获取 Pipeline 配置"""
        if PipelineConfig is None:
            return None
        try:
            return PipelineConfig(
                num_stages=self.config.world_size,
                num_micro_batches=8,
            )
        except Exception:
            return None
    
    def _setup_adapters(self) -> None:
        """
        设置适配器层
        
        使用: AdapterManager, get_adapter_manager, EncoderFactory, create_encoder,
              FusionFactory, create_fusion, AlignmentFactory, create_alignment,
              AdapterFactory, create_adapter, ModalityType, FusionMethod, AlignmentMethod, AdapterType
        """
        # 获取适配器管理器 (AdapterManager)
        if get_adapter_manager is not None:
            try:
                self._adapter_manager = get_adapter_manager()
            except Exception as e:
                logger.warning(f"Failed to get adapter manager: {e}")
        
        # 创建模态编码器 (使用 EncoderFactory, create_encoder, ModalityType)
        for modality in self.config.modalities:
            encoder = self._create_encoder(modality)
            if encoder is not None:
                self._encoders[modality] = encoder.to(self._device)
        
        # 创建融合模块 (使用 FusionFactory, create_fusion, FusionMethod)
        if len(self.config.modalities) > 1:
            self._fusion = self._create_fusion()
            if self._fusion is not None:
                self._fusion = self._fusion.to(self._device)
            
            # 创建对齐模块 (使用 AlignmentFactory, create_alignment, AlignmentMethod)
            self._alignment = self._create_alignment()
            if self._alignment is not None:
                self._alignment = self._alignment.to(self._device)
        
        # 创建模型适配器 (使用 AdapterFactory, create_adapter, AdapterType)
        if self.config.adapter_type:
            self._adapter = self._create_adapter()
        
        logger.info(f"Adapters layer initialized: encoders={list(self._encoders.keys())}, "
                   f"fusion={self._fusion is not None}, alignment={self._alignment is not None}, "
                   f"adapter={self._adapter is not None}")
    
    def _create_encoder(self, modality: str) -> Optional[nn.Module]:
        """
        创建模态编码器
        
        使用: EncoderFactory, create_encoder, ModalityType
        """
        # 1. 优先使用 EncoderFactory
        if EncoderFactory is not None:
            try:
                return EncoderFactory.create(modality, hidden_size=self.config.hidden_size)
            except Exception as e:
                logger.debug(f"EncoderFactory failed: {e}")
        
        # 2. 使用 create_encoder
        if create_encoder is not None:
            try:
                return create_encoder(modality, hidden_size=self.config.hidden_size)
            except Exception as e:
                logger.debug(f"create_encoder failed: {e}")
        
        # 3. 使用 ModalityType 验证
        if ModalityType is not None:
            try:
                modality_type = ModalityType(modality)
                logger.debug(f"ModalityType resolved: {modality_type}")
            except Exception:
                pass
        
        return None
    
    def _create_fusion(self) -> Optional[nn.Module]:
        """
        创建融合模块
        
        使用: FusionFactory, create_fusion, FusionMethod
        """
        fusion_method = self.config.fusion_method
        
        # 1. 优先使用 FusionFactory
        if FusionFactory is not None:
            try:
                return FusionFactory.create(
                    fusion_method,
                    hidden_size=self.config.hidden_size,
                    num_modalities=len(self.config.modalities)
                )
            except Exception as e:
                logger.debug(f"FusionFactory failed: {e}")
        
        # 2. 使用 create_fusion
        if create_fusion is not None:
            try:
                return create_fusion(fusion_method, hidden_size=self.config.hidden_size)
            except Exception as e:
                logger.debug(f"create_fusion failed: {e}")
        
        # 3. 使用 FusionMethod 验证
        if FusionMethod is not None:
            try:
                fusion_type = FusionMethod(fusion_method)
                logger.debug(f"FusionMethod resolved: {fusion_type}")
            except Exception:
                pass
        
        return None
    
    def _create_alignment(self) -> Optional[nn.Module]:
        """
        创建对齐模块
        
        使用: AlignmentFactory, create_alignment, AlignmentMethod
        """
        alignment_method = self.config.alignment_method
        
        # 1. 优先使用 AlignmentFactory
        if AlignmentFactory is not None:
            try:
                return AlignmentFactory.create(
                    alignment_method,
                    hidden_size=self.config.hidden_size,
                    temperature=self.config.contrastive_temperature
                )
            except Exception as e:
                logger.debug(f"AlignmentFactory failed: {e}")
        
        # 2. 使用 create_alignment
        if create_alignment is not None:
            try:
                return create_alignment(alignment_method, hidden_size=self.config.hidden_size)
            except Exception as e:
                logger.debug(f"create_alignment failed: {e}")
        
        # 3. 使用 AlignmentMethod 验证
        if AlignmentMethod is not None:
            try:
                align_type = AlignmentMethod(alignment_method)
                logger.debug(f"AlignmentMethod resolved: {align_type}")
            except Exception:
                pass
        
        return None
    
    def _create_adapter(self) -> Optional['ModelAdapter']:
        """
        创建模型适配器
        
        使用: AdapterFactory, create_adapter, AdapterType
        """
        adapter_type = self.config.adapter_type
        
        # 1. 优先使用 AdapterFactory
        if AdapterFactory is not None:
            try:
                return AdapterFactory.create(
                    adapter_type,
                    hidden_size=self.config.hidden_size,
                    lora_rank=self.config.lora_rank,
                    alpha=self.config.adapter_alpha
                )
            except Exception as e:
                logger.debug(f"AdapterFactory failed: {e}")
        
        # 2. 使用 create_adapter
        if create_adapter is not None:
            try:
                return create_adapter(
                    adapter_type,
                hidden_size=self.config.hidden_size,
                lora_rank=self.config.lora_rank
            )
            except Exception as e:
                logger.debug(f"create_adapter failed: {e}")
        
        # 3. 使用 AdapterType 验证
        if AdapterType is not None:
            try:
                adapt_type = AdapterType(adapter_type)
                logger.debug(f"AdapterType resolved: {adapt_type}")
            except Exception:
                pass
        
        return None
    
    def _setup_losses(self) -> None:
        """
        设置损失层
        
        使用: LossFactory, create_loss, create_composite_loss, create_distillation_loss,
              BaseLoss, CompositeLoss, MultiTaskLoss, CrossEntropyLoss, FocalLoss, MSELoss,
              SoftLabelLoss, FeatureDistillationLoss, AttentionDistillationLoss,
              InfoNCELoss, CLIPLoss, ContrastiveLoss, LossMonitor
        """
        # 创建主损失 (使用 LossFactory, create_loss 或特定损失类)
        self._loss_fn = self._create_task_loss()
        
        # 创建对比损失 (使用 InfoNCELoss, CLIPLoss, ContrastiveLoss)
        self._contrastive_loss = self._create_contrastive_loss()
        
        # 创建蒸馏损失 (使用 create_distillation_loss, SoftLabelLoss, FeatureDistillationLoss)
        self._distillation_loss = self._create_distillation_loss()
        
        # 如果有辅助损失，创建复合损失 (使用 CompositeLoss, MultiTaskLoss)
        if self.config.auxiliary_losses:
            loss_components = [('task', self._loss_fn, 1.0)]
            for aux in self.config.auxiliary_losses:
                aux_loss = self._create_auxiliary_loss(aux)
                if aux_loss is not None:
                    loss_components.append((aux['name'], aux_loss, aux.get('weight', 0.1)))
            
            self._loss_fn = self._create_composite_loss(loss_components)
        
        # 创建损失监控器 (LossMonitor)
        if LossMonitor is not None:
            try:
                self._loss_monitor = LossMonitor(max_history=10000)
            except Exception:
                pass
        
        logger.info(f"Losses layer initialized: task={self.config.task_loss_type}, "
                   f"contrastive={self._contrastive_loss is not None}, "
                   f"distillation={self._distillation_loss is not None}, "
                   f"monitor={self._loss_monitor is not None}")
    
    def _create_task_loss(self) -> nn.Module:
        """
        创建任务损失
        
        使用: LossFactory, create_loss, CrossEntropyLoss, FocalLoss, MSELoss
        """
        loss_type = self.config.task_loss_type
        
        # 1. 优先使用 LossFactory
        try:
            return LossFactory.create(loss_type)
        except Exception as e:
            logger.debug(f"LossFactory failed: {e}")
        
        # 2. 使用 create_loss
        try:
            return create_loss(loss_type)
        except Exception as e:
            logger.debug(f"create_loss failed: {e}")
        
        # 3. 使用特定损失类
        loss_map = {
            'cross_entropy': CrossEntropyLoss,
            'focal': FocalLoss,
            'mse': MSELoss,
        }
        loss_cls = loss_map.get(loss_type)
        if loss_cls is not None:
            try:
                return loss_cls()
            except Exception as e:
                logger.debug(f"Specific loss {loss_cls} failed: {e}")
        
        # 4. 回退
        return nn.CrossEntropyLoss()
    
    def _create_contrastive_loss(self) -> Optional[nn.Module]:
        """
        创建对比损失
        
        使用: InfoNCELoss, CLIPLoss, ContrastiveLoss
        """
        temperature = self.config.contrastive_temperature
        
        # 尝试 CLIPLoss
        if CLIPLoss is not None:
            try:
                return CLIPLoss(temperature=temperature)
            except Exception:
                pass
        
        # 尝试 InfoNCELoss
        if InfoNCELoss is not None:
            try:
                return InfoNCELoss(temperature=temperature)
            except Exception:
                pass
        
        # 尝试 ContrastiveLoss (使用 InfoNCELoss 作为具体实现)
        if InfoNCELoss is not None:
            try:
                return InfoNCELoss(temperature=temperature)
            except Exception:
                pass
        
        return None
    
    def _create_distillation_loss(self) -> Optional[nn.Module]:
        """
        创建蒸馏损失
        
        使用: create_distillation_loss, SoftLabelLoss, FeatureDistillationLoss, AttentionDistillationLoss
        """
        temperature = self.config.distillation_temperature
        
        # 使用 create_distillation_loss
        if create_distillation_loss is not None:
            try:
                # create_distillation_loss 需要 loss_type 参数
                return create_distillation_loss(loss_type='combined', temperature=temperature)  
            except Exception:
                pass
        
        # 尝试 SoftLabelLoss
        if SoftLabelLoss is not None:
            try:
                return SoftLabelLoss(temperature=temperature)
            except Exception:
                pass
        
        # 尝试 FeatureDistillationLoss
        if FeatureDistillationLoss is not None:
            try:
                return FeatureDistillationLoss()
            except Exception:
                pass
        
        return None
    
    def _create_auxiliary_loss(self, aux_config: Dict[str, Any]) -> Optional[nn.Module]:
        """创建辅助损失"""
        loss_type = aux_config.get('type', 'mse')
        params = aux_config.get('params', {})
        
        # 使用 create_loss
        if create_loss is not None:
            try:
                return create_loss(loss_type, **params)
            except Exception:
                pass
        
        # 特定损失类
        loss_map = {
            'mse': MSELoss,
            'contrastive': ContrastiveLoss,
            'focal': FocalLoss,
        }
        loss_cls = loss_map.get(loss_type)
        if loss_cls is not None:
            try:
                return loss_cls(**params)
            except Exception:
                try:
                    return loss_cls()
                except Exception:
                    pass
        
        return None
    
    def _create_composite_loss(self, loss_components: List) -> nn.Module:
        """
        创建复合损失
        
        使用: create_composite_loss, CompositeLoss, MultiTaskLoss
        """
        # 使用 create_composite_loss
        if create_composite_loss is not None:
            try:
                return create_composite_loss(loss_components)
            except Exception as e:
                logger.debug(f"create_composite_loss failed: {e}")
        
        # 使用 CompositeLoss
        if CompositeLoss is not None:
            try:
                losses = {name: loss for name, loss, _ in loss_components}
                weights = {name: weight for name, _, weight in loss_components}
                return CompositeLoss(losses, weights)
            except Exception as e:
                logger.debug(f"CompositeLoss failed: {e}")
        
        # 使用 MultiTaskLoss
        if MultiTaskLoss is not None:
            try:
                losses = {name: loss for name, loss, _ in loss_components}
                weights = {name: weight for name, _, weight in loss_components}
                return MultiTaskLoss(losses, weights)
            except Exception as e:
                logger.debug(f"MultiTaskLoss failed: {e}")
        
        # 回退到第一个损失
        return loss_components[0][1] if loss_components else nn.CrossEntropyLoss()
    
    def _apply_adapter(self, model: nn.Module) -> nn.Module:
        """应用适配器到模型"""
        if self._adapter is not None:
            return self._adapter.adapt(model)
        return model
    
    def _get_parallel_mode(self) -> 'ParallelMode':
        """获取并行模式"""
        mode_map = {
            'ddp': ParallelMode.DDP,
            'fsdp': ParallelMode.FSDP,
            'zero': ParallelMode.ZERO_2,
            'zero1': ParallelMode.ZERO_1,
            'zero2': ParallelMode.ZERO_2,
            'zero3': ParallelMode.ZERO_3,
            'pipeline': ParallelMode.PIPELINE
        }
        return mode_map.get(self.config.distributed_mode, ParallelMode.DDP)
    
    @property
    def device(self) -> torch.device:
        """获取设备"""
        return self._device
    
    @property
    @contextmanager
    def amp_context(self):
        """获取AMP上下文"""
        if self._amp_manager:
            with self._amp_manager.autocast_context():
                yield
        else:
            yield
    
    def get_amp_context(self):
        """获取AMP上下文（用于with语句）"""
        if self._amp_manager:
            return self._amp_manager.autocast_context()
        return _nullcontext()
    
    def encode_modality(
        self, 
        modality: str, 
        inputs: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """编码指定模态"""
        if modality not in self._encoders:
            raise ValueError(f"Unknown modality: {modality}")
        
        encoder = self._encoders[modality]
        inputs = inputs.to(self._device)
        return encoder(inputs, **kwargs)
    
    def encode_multimodal(
        self, 
        modality_inputs: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """编码多个模态"""
        features = {}
        for modality, inputs in modality_inputs.items():
            features[modality] = self.encode_modality(modality, inputs, **kwargs)
        return features
    
    def align_modalities(
        self, 
        features_a: torch.Tensor, 
        features_b: torch.Tensor,
        **kwargs
    ) -> tuple:
        """对齐两个模态"""
        if self._alignment is None:
            return features_a, features_b
        return self._alignment.align(features_a, features_b, **kwargs)
    
    def fuse_modalities(
        self, 
        features: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """融合多模态特征"""
        if self._fusion is None:
            return torch.cat(features, dim=-1)
        return self._fusion.fuse(features, **kwargs)
    
    def compute_task_loss(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """计算任务损失"""
        if self._loss_fn is None:
            return nn.functional.cross_entropy(outputs, targets)
        return self._loss_fn(outputs, targets, **kwargs)
    
    def prepare_batch(self, batch: Dict[str, Any], context: StrategyContext) -> Dict[str, Any]:
        """准备批次数据"""
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(self._device)
            else:
                prepared[key] = value
        return prepared
    
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """计算损失"""
        with self.get_amp_context():
            if 'loss' in outputs:
                loss = outputs['loss']
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                logits = outputs.get('logits', outputs.get('output'))
                labels = batch.get('labels', batch.get('targets'))
                
                if logits is not None and labels is not None:
                    loss = self.compute_task_loss(logits, labels)
                else:
                    raise ValueError("Cannot compute loss: no loss in outputs and no logits/labels")
        
        metrics = {
            'loss': loss.item() if isinstance(loss, torch.Tensor) else loss
        }
        
        return StrategyResult(loss=loss, metrics=metrics)
    
    def backward(self, loss: torch.Tensor) -> None:
        """反向传播"""
        if self._amp_manager:
            self._amp_manager.backward(loss)
        else:
            loss.backward()
    
    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """优化器步进"""
        if self._amp_manager:
            self._amp_manager.step(optimizer)
        else:
            optimizer.step()
    
    def clip_gradients(self, model: nn.Module) -> None:
        """梯度裁剪"""
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.config.max_grad_norm
            )
    
    def sync_gradients(self) -> None:
        """
        同步梯度（分布式）
        
        使用: barrier (backend/lib/distributed)
        """
        if self._is_distributed:
            try:
                start = time.time()
                barrier()
                elapsed = (time.time() - start) * 1000
                self._wrapper_stats.record_sync(elapsed)
            except Exception as e:
                logger.warning(f"Barrier failed: {e}")
    
    def all_reduce_tensor(
        self, 
        tensor: torch.Tensor, 
        op: str = "mean"
    ) -> torch.Tensor:
        """
        全局归约张量
        
        使用: all_reduce, AllReduceOp (backend/lib/distributed)
        """
        if not self._is_distributed:
            return tensor
        
        try:
            # 使用 AllReduceOp
            if AllReduceOp is not None:
                op_map = {
                    'sum': AllReduceOp.SUM,
                    'mean': AllReduceOp.SUM,  # 手动除以 world_size
                    'max': AllReduceOp.MAX,
                    'min': AllReduceOp.MIN,
                }
                reduce_op = op_map.get(op, AllReduceOp.SUM)
                all_reduce(tensor, op=reduce_op)
                
                if op == 'mean' and get_world_size is not None:
                    world_size = get_world_size()
                    tensor = tensor / world_size
            else:
                all_reduce(tensor)
            
            self._wrapper_stats.total_all_reduces += 1
            return tensor
        except Exception as e:
            logger.warning(f"all_reduce failed: {e}")
            return tensor
    
    def _get_distributed_info(self) -> Dict[str, Any]:
        """
        获取分布式信息
        
        使用: is_main_process, get_rank, get_world_size (backend/lib/distributed)
        """
        info = {
            'is_distributed': self._is_distributed,
            'rank': 0,
            'world_size': 1,
            'is_main': True,
        }
        
        # 使用 get_rank
        if get_rank is not None:
            try:
                info['rank'] = get_rank()
            except Exception:
                pass
        
        # 使用 get_world_size
        if get_world_size is not None:
            try:
                info['world_size'] = get_world_size()
            except Exception:
                pass
        
        # 使用 is_main_process
        if is_main_process is not None:
            try:
                info['is_main'] = is_main_process()
            except Exception:
                info['is_main'] = info['rank'] == 0
        else:
            info['is_main'] = info['rank'] == 0
        
        return info
    
    def should_log(self) -> bool:
        """是否应该记录日志（仅主进程）"""
        dist_info = self._get_distributed_info()
        return dist_info.get('is_main', True)
    
    def _check_health(self) -> ProductionHealthStatus:
        """
        检查健康状态
        
        使用: get_available_memory (backend/lib/hardware)
        """
        self._health_status = ProductionHealthStatus()
        self._health_status.last_check_time = time.time()
        
        # 检查硬件层
        self._health_status.hardware_ok = True
        # 检查内存压力 (使用 get_available_memory)
    
        try:
            available = get_available_memory()
            if isinstance(available, (int, float)):
                if available < 1e9:  # < 1GB
                    self._health_status.memory_pressure = "critical"
                    self._health_status.add_issue("Memory critically low")
                elif available < 4e9:  # < 4GB
                    self._health_status.memory_pressure = "high"
        except Exception:
            pass
        
        # 检查分布式层
        self._health_status.distributed_ok = not self._is_distributed
        if self._is_distributed:
            self._health_status.add_issue("Distributed layer not available")
        
        # 检查适配器层
        self._health_status.adapters_ok = len(self.config.modalities) <= 1
        if len(self.config.modalities) > 1:
            self._health_status.add_issue("Adapters layer not available for multimodal")
        
        # 检查损失层
        self._health_status.losses_ok = self._loss_fn is not None
        if self._loss_fn is None:
            self._health_status.add_issue("Losses layer not available")
        
        return self._health_status
    
    def on_step_end(self, context: StrategyContext, result: StrategyResult) -> None:
        """步骤结束回调"""
        super().on_step_end(context, result)
        
        # 记录到策略监控器
        if self._strategy_monitor is not None:
            try:
                self._strategy_monitor.record_step(
                    context.global_step,
                    result.loss.item() if hasattr(result.loss, 'item') else result.loss,
                    result.metrics
                )
            except Exception:
                pass
        
        # 记录到损失监控器
        if self._loss_monitor is not None:
            try:
                self._loss_monitor.record(
                    result.loss.item() if hasattr(result.loss, 'item') else result.loss
                )
            except Exception:
                pass
        
        # 定期健康检查
        if (self.config.health_check_interval > 0 and 
            context.global_step % self.config.health_check_interval == 0):
            self._check_health()
    
    def on_epoch_end(self, context: StrategyContext) -> None:
        """
        Epoch结束
        
        使用: clear_memory (backend/lib/hardware)
        """
        super().on_epoch_end(context)
        
        try:
            clear_memory()
        except Exception:
            pass
    
    def cleanup(self) -> None:
        """
        清理资源
        
        使用: clear_memory (backend/lib/hardware)
        """
        super().cleanup()
        
        if self._is_distributed and self._distributed_manager:
            try:
                self._distributed_manager.cleanup()
            except Exception:
                pass
        
        
        try:
            clear_memory()
        except Exception:
            pass
        
        self._is_setup = False
        logger.info(f"ProductionTrainingStrategy '{self.name}' cleaned up")
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        包含所有底层模块的使用状态
        """
        dist_info = self._get_distributed_info()
        
        return {
            'name': self.name,
            'strategy_type': str(self.STRATEGY_TYPE),
            'device': str(self._device),
            'precision': self.config.precision,
            'distributed_mode': self.config.distributed_mode,
            'modalities': self.config.modalities,
            'adapter_type': self.config.adapter_type,
            'loss_type': self.config.task_loss_type,
            
            # 组件状态
            'components_initialized': {
                # 硬件组件
                'device_manager': self._device_manager is not None,
                'memory_manager': self._memory_manager is not None,
                'amp_manager': self._amp_manager is not None,
                'gradient_checkpointing': self._gradient_checkpointing is not None,
                # 分布式组件
                'distributed_manager': self._distributed_manager is not None,
                'distributed_wrapper': self._distributed_wrapper is not None,
                # 适配器组件
                'adapter_manager': self._adapter_manager is not None,
                'encoders': list(self._encoders.keys()),
                'fusion': self._fusion is not None,
                'alignment': self._alignment is not None,
                'adapter': self._adapter is not None,
                # 损失组件
                'loss_fn': self._loss_fn is not None,
                'contrastive_loss': self._contrastive_loss is not None,
                'distillation_loss': self._distillation_loss is not None,
                'loss_monitor': self._loss_monitor is not None,
                # 基础策略组件
                'strategy_monitor': self._strategy_monitor is not None,
                'strategy_profiler': self._strategy_profiler is not None,
                'strategy_validator': self._strategy_validator is not None,
                'strategy_metrics': self._strategy_metrics is not None,
            },
            
            # 分布式信息
            'distributed_info': dist_info,
            
            # 包装器统计
            'wrapper_stats': self._wrapper_stats.to_dict(),
            
            # 健康状态
            'health_status': self._health_status.to_dict(),
        }
    
    def get_strategy_monitor(self) -> Optional[StrategyMonitor]:
        """获取策略监控器"""
        return self._strategy_monitor
    
    def get_strategy_profiler(self) -> Optional[StrategyProfiler]:
        """获取策略分析器"""
        return self._strategy_profiler
    
    def get_strategy_validator(self) -> Optional[StrategyValidator]:
        """获取策略验证器"""
        return self._strategy_validator
    
    def get_strategy_metrics(self) -> Optional[StrategyMetrics]:
        """获取策略指标"""
        return self._strategy_metrics
    
    def get_health_status(self) -> ProductionHealthStatus:
        """获取健康状态"""
        return self._health_status
    
    def get_wrapper_stats(self) -> WrapperStats:
        """获取包装器统计"""
        return self._wrapper_stats
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计
        
        使用: get_available_memory, MemoryStats (backend/lib/hardware)
        """
        stats = {}
        
        # 使用 get_available_memory
       
        try:
            available = get_available_memory()
            if isinstance(available, (int, float)):
                stats['available_gb'] = available / (1024**3) if available > 1000 else available
            elif isinstance(available, dict):
                stats['available'] = available
        except Exception:
            pass
        
        # 使用 MemoryManager
        if self._memory_manager is not None:
            try:
                mgr_stats = self._memory_manager.get_stats()
                if hasattr(mgr_stats, '__dict__'):
                    stats['manager'] = mgr_stats.__dict__
            except Exception:
                pass
        
        # PyTorch 原生
        if torch.cuda.is_available():
            stats.update({
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
            })
        
        return stats
    
    def diagnose(self) -> Dict[str, Any]:
        """
        诊断策略
        
        使用所有可用的监控和分析组件
        """
        diagnosis = {
            'health': self._check_health().to_dict(),
            'info': self.get_info(),
            'memory': self.get_memory_stats(),
            'issues': [],
            'recommendations': [],
        }
        
        # 检查策略监控器
        if self._strategy_monitor is not None:
            try:
                if hasattr(self._strategy_monitor, 'get_summary'):
                    diagnosis['monitor_summary'] = self._strategy_monitor.get_summary()
            except Exception:
                pass
        
        # 检查分析器
        if self._strategy_profiler is not None:
            try:
                if hasattr(self._strategy_profiler, 'get_stats'):
                    diagnosis['profiler_stats'] = self._strategy_profiler.get_stats()
            except Exception:
                pass
        
        return diagnosis
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diagnosis = self.diagnose()
        
        print("\n" + "=" * 60)
        print("Production Strategy Diagnosis")
        print("=" * 60)
        
        print(f"\nHealth: {'OK' if diagnosis['health']['is_healthy'] else 'ISSUES FOUND'}")
        if diagnosis['health']['issues']:
            print("  Issues:")
            for issue in diagnosis['health']['issues']:
                print(f"    - {issue}")
        
        print(f"\nLayers:")
        for layer, available in diagnosis['info']['layers_available'].items():
            print(f"  {layer}: {'✓' if available else '✗'}")
        
        if diagnosis['recommendations']:
            print("\nRecommendations:")
            for rec in diagnosis['recommendations']:
                print(f"  - {rec}")
        
        print("=" * 60)


# ==================== 生产级训练上下文 ====================

class ProductionTrainingContext:
    """
    生产级训练上下文
    
    整合六层架构的统一训练上下文，提供：
    - 设备管理和混合精度 (DeviceManager, MixedPrecisionManager, MemoryManager, GradientCheckpointing)
    - 分布式训练支持 (DistributedManager, DDPWrapper, FSDPWrapper, ZeROWrapper, PipelineWrapper)
    - 模态编码/融合/对齐 (EncoderFactory, FusionFactory, AlignmentFactory)
    - 损失函数组合 (LossFactory, CompositeLoss, MultiTaskLoss)
    
    使用的底层模块同 ProductionTrainingStrategy
    """
    
    def __init__(
        self,
        config: ProductionStrategyConfig,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.model = model
        
        # 设备设置 (使用 get_device_manager)
        if device:
            self._device = device
        elif get_device_manager is not None:
            try:
                dm = get_device_manager()
                self._device = dm.get_device()
            except Exception:
                self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 底层管理器 (backend/lib/hardware)
        self._device_manager: Optional['DeviceManager'] = None
        self._memory_manager: Optional['MemoryManager'] = None
        self._amp_manager: Optional['MixedPrecisionManager'] = None
        self._gradient_checkpointing: Optional['GradientCheckpointing'] = None
        
        # 底层管理器 (backend/lib/distributed)
        self._distributed_manager: Optional['DistributedManager'] = None
        self._distributed_wrapper: Optional[nn.Module] = None
        
        # 底层管理器 (backend/lib/adapters)
        self._adapter_manager: Optional['AdapterManager'] = None
        
        # 组件
        self._encoders: Dict[str, 'ModalityEncoder'] = {}
        self._fusion: Optional['FusionModule'] = None
        self._alignment: Optional['AlignmentModule'] = None
        self._loss_fn: Optional[nn.Module] = None
        self._contrastive_loss: Optional[nn.Module] = None
        self._loss_monitor: Optional['LossMonitor'] = None
        
        self._initialized = False
    
    def initialize(self) -> None:
        """
        初始化所有层
        
        使用: DeviceManager, MixedPrecisionManager, MemoryManager, GradientCheckpointing,
              DistributedManager, EncoderFactory, FusionFactory, AlignmentFactory,
              LossFactory, CompositeLoss, MultiTaskLoss
        """
        if self._initialized:
            return
        
        logger.info("Initializing ProductionTrainingContext...")
        
        # 1. 硬件层 (使用 DeviceManager, MixedPrecisionManager, MemoryManager, GradientCheckpointing)
        if get_device_manager is not None:
            try:
                self._device_manager = get_device_manager()
            except Exception:
                pass
            
        if self.config.enable_amp and MixedPrecisionManager is not None and PrecisionMode is not None:
            try:
                from backend.lib.hardware.mixed_precision import AmpConfig
                precision_map = {
                    'fp32': PrecisionMode.FP32,
                    'fp16': PrecisionMode.MIXED_FP16,
                    'bf16': PrecisionMode.MIXED_BF16
                }
                amp_config = AmpConfig(
                    enabled=True,
                    precision=precision_map.get(self.config.precision, PrecisionMode.MIXED_FP16)
                )
                self._amp_manager = MixedPrecisionManager(amp_config, self._device)
            except Exception as e:
                logger.warning("Failed to init MixedPrecisionManager: %s", e)
            
        # 内存管理器 (MemoryManager)
        if self.config.memory_optimization and MemoryManager is not None:
            try:
                self._memory_manager = MemoryManager(device=self._device)
            except Exception:
                pass
            
        # 梯度检查点 (GradientCheckpointing)
        # 注意：GradientCheckpointing 需要 model 参数，但此时 model 可能还未创建
        # 所以这里先不初始化，等到 setup 时再初始化
        if self.config.enable_gradient_checkpointing and GradientCheckpointing is not None and self.model is not None:
            try:
                self._gradient_checkpointing = GradientCheckpointing(model=self.model)
            except Exception:
                pass
        else:
            self._gradient_checkpointing = None
        
        # 2. 分布式层 (使用 DistributedManager, DDPConfig, FSDPConfig, ZeROConfig, PipelineConfig)
        if self.config.distributed_mode != "none":
            if self.config.world_size > 1 and get_distributed_manager is not None:
                try:
                    self._distributed_manager = get_distributed_manager()
                    self._distributed_manager.initialize(
                        backend='nccl' if torch.cuda.is_available() else 'gloo',
                        world_size=self.config.world_size
                    )
                except Exception as e:
                    logger.warning(f"Failed to init DistributedManager: {e}")
        
            # 3. 适配器层 (使用 EncoderFactory, FusionFactory, AlignmentFactory, ModalityType, FusionMethod, AlignmentMethod)
            try:
                self._adapter_manager = get_adapter_manager()
            except Exception:
                pass
            
            # 创建模态编码器 (使用 EncoderFactory, create_encoder, ModalityType)
            for modality in self.config.modalities:
                encoder = self._create_encoder(modality)
                if encoder is not None:
                    self._encoders[modality] = encoder.to(self._device)
            
            # 创建融合和对齐模块 (使用 FusionFactory, AlignmentFactory)
            if len(self.config.modalities) > 1:
                self._fusion = self._create_fusion()
                if self._fusion is not None:
                    self._fusion = self._fusion.to(self._device)
                
                self._alignment = self._create_alignment()
                if self._alignment is not None:
                    self._alignment = self._alignment.to(self._device)
        
        # 4. 损失层 (使用 LossFactory, create_loss, CompositeLoss, MultiTaskLoss, InfoNCELoss, CLIPLoss)
        # 主损失 (使用 LossFactory, create_loss)
        self._loss_fn = self._create_task_loss()
            
        # 对比损失 (使用 InfoNCELoss, CLIPLoss, ContrastiveLoss)
        self._contrastive_loss = self._create_contrastive_loss()
            
        # 复合损失 (使用 CompositeLoss, MultiTaskLoss)
        if self.config.auxiliary_losses:
            loss_components = [('task', self._loss_fn, 1.0)]
            for aux in self.config.auxiliary_losses:
                if create_loss is not None:
                    try:
                        aux_loss = create_loss(aux['type'], **aux.get('params', {}))
                        loss_components.append((aux['name'], aux_loss, aux.get('weight', 0.1)))
                    except Exception as e:
                        logger.warning(f"Failed to create auxiliary loss: {e}")
                        pass
                
            if create_composite_loss is not None:
                try:
                    self._loss_fn = create_composite_loss(loss_components)
                except Exception:
                    pass
            elif CompositeLoss is not None:
                try:
                    losses = {name: loss for name, loss, _ in loss_components}
                    weights = {name: weight for name, _, weight in loss_components}
                    self._loss_fn = CompositeLoss(losses, weights)
                except Exception:
                    pass
            
        # 损失监控 (使用 LossMonitor)
        if LossMonitor is not None:
            try:
                self._loss_monitor = LossMonitor(max_history=10000)
            except Exception:
                pass
        
        self._initialized = True
        logger.info(f"ProductionTrainingContext initialized: device={self._device}, "
                   f"encoders={list(self._encoders.keys())}, "
                   f"amp={self._amp_manager is not None}")
    
    def _create_encoder(self, modality: str) -> Optional[nn.Module]:
        """创建模态编码器 (使用 EncoderFactory, create_encoder, ModalityType)"""
        try:
            return EncoderFactory.create(modality, hidden_size=self.config.hidden_size)
        except Exception:
            pass
        try:
            return create_encoder(modality, hidden_size=self.config.hidden_size)
        except Exception:
            pass

        if ModalityType is not None:
            try:
                _ = ModalityType(modality)
            except Exception:
                pass
        return None
    
    def _create_fusion(self) -> Optional[nn.Module]:
        """创建融合模块 (使用 FusionFactory, create_fusion, FusionMethod)"""
        try:
            return FusionFactory.create(self.config.fusion_method, hidden_size=self.config.hidden_size)
        except Exception:
            pass
        try:
            return create_fusion(self.config.fusion_method, hidden_size=self.config.hidden_size)
        except Exception:
            pass
        try:
            _ = FusionMethod(self.config.fusion_method)
        except Exception:
            pass
        
        return None
    
    def _create_alignment(self) -> Optional[nn.Module]:
        """创建对齐模块 (使用 AlignmentFactory, create_alignment, AlignmentMethod)"""
        try:
            return AlignmentFactory.create(self.config.alignment_method, hidden_size=self.config.hidden_size)
        except Exception:
            pass
        try:
            return create_alignment(self.config.alignment_method, hidden_size=self.config.hidden_size)
        except Exception:
            pass
        try:
            _ = AlignmentMethod(self.config.alignment_method)
        except Exception:
            pass
        
        return None
    
    def _create_task_loss(self) -> nn.Module:
        """创建任务损失 (使用 LossFactory, create_loss, CrossEntropyLoss, FocalLoss, MSELoss)"""
        try:
            return LossFactory.create(self.config.task_loss_type)
        except Exception:
            pass
        
        try:
            return create_loss(self.config.task_loss_type)
        except Exception:
            pass

        loss_map = {'cross_entropy': CrossEntropyLoss, 'focal': FocalLoss, 'mse': MSELoss}
        loss_cls = loss_map.get(self.config.task_loss_type)
        try:
            return loss_cls()
        except Exception:
            pass
        return nn.CrossEntropyLoss()
    
    def _create_contrastive_loss(self) -> Optional[nn.Module]:
        """创建对比损失 (使用 CLIPLoss, InfoNCELoss)"""
        temperature = self.config.contrastive_temperature
        # 使用 InfoNCELoss 替代抽象类 ContrastiveLoss
        for loss_cls in [CLIPLoss, InfoNCELoss]:
            if loss_cls is not None:
                try:
                    return loss_cls(temperature=temperature)
                except Exception:
                    pass
        return None
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def is_distributed(self) -> bool:
        return self._distributed_manager is not None
    
    def get_amp_context(self):
        """获取AMP上下文"""
        if self._amp_manager:
            return self._amp_manager.autocast_context()
        # 使用 _nullcontext 避免重定义警告
        return _nullcontext()
    
    def to_device(self, data):
        """将数据移到设备"""
        if isinstance(data, torch.Tensor):
            return data.to(self._device)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(item) for item in data)
        return data
    
    def encode_modality(self, modality: str, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """编码指定模态"""
        if modality in self._encoders:
            return self._encoders[modality](inputs.to(self._device), **kwargs)
        return inputs
    
    def fuse_modalities(self, features: List[torch.Tensor], **kwargs) -> torch.Tensor:
        """融合多模态特征"""
        if self._fusion is not None:
            return self._fusion.fuse(features, **kwargs)
        return torch.cat(features, dim=-1)
    
    def align_modalities(self, feat_a: torch.Tensor, feat_b: torch.Tensor, **kwargs) -> tuple:
        """对齐两个模态"""
        if self._alignment is not None:
            return self._alignment.align(feat_a, feat_b, **kwargs)
        return feat_a, feat_b
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """计算损失"""
        return self._loss_fn(outputs, targets, **kwargs)
    
    def backward(self, loss: torch.Tensor) -> None:
        """反向传播"""
        if self._amp_manager:
            self._amp_manager.backward(loss)
        else:
            loss.backward()
    
    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        """优化器步进"""
        if self._amp_manager:
            self._amp_manager.step(optimizer)
        else:
            optimizer.step()
    
    def sync_gradients(self) -> None:
        """
        同步梯度（分布式）
        
        使用: barrier (backend/lib/distributed)
        """
        if self.is_distributed:
            try:
                barrier()
            except Exception as e:
                logger.warning(f"Barrier failed: {e}")
    
    def all_reduce_tensor(self, tensor: torch.Tensor, op: str = "mean") -> torch.Tensor:
        """
        全局归约张量
        
        使用: all_reduce, AllReduceOp (backend/lib/distributed)
        """
        if not self.is_distributed:
            return tensor
        
        try:
            if AllReduceOp is not None:
                op_map = {
                    'sum': AllReduceOp.SUM,
                    'mean': AllReduceOp.SUM,
                    'max': AllReduceOp.MAX,
                    'min': AllReduceOp.MIN,
                }
                reduce_op = op_map.get(op, AllReduceOp.SUM)
                all_reduce(tensor, op=reduce_op)
                
                if op == 'mean' and get_world_size is not None:
                    tensor = tensor / get_world_size()
            else:
                all_reduce(tensor)
            return tensor
        except Exception:
            return tensor
    
    def get_distributed_info(self) -> Dict[str, Any]:
        """
        获取分布式信息
        
        使用: is_main_process, get_rank, get_world_size (backend/lib/distributed)
        """
        info = {'rank': 0, 'world_size': 1, 'is_main': True}
        if get_rank is not None:
            try:
                info['rank'] = get_rank()
            except Exception:
                pass
        if get_world_size is not None:
            try:
                info['world_size'] = get_world_size()
            except Exception:
                pass
        if is_main_process is not None:
            try:
                info['is_main'] = is_main_process()
            except Exception:
                info['is_main'] = info['rank'] == 0
        return info
    
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """
        包装模型以支持分布式
        
        使用: DistributedManager, DDPWrapper, FSDPWrapper, ZeROWrapper, PipelineWrapper, ParallelMode
        """
        model = model.to(self._device)
        
        if self.is_distributed and self._distributed_manager and ParallelMode is not None:
            mode_map = {
                'ddp': ParallelMode.DDP,
                'fsdp': ParallelMode.FSDP,
                'zero': ParallelMode.ZERO_2,
                'zero1': ParallelMode.ZERO_1,
                'zero2': ParallelMode.ZERO_2,
                'zero3': ParallelMode.ZERO_3,
                'pipeline': ParallelMode.PIPELINE
            }
            parallel_mode = mode_map.get(self.config.distributed_mode, ParallelMode.DDP)
            
            try:
                model = self._distributed_manager.wrap_model(model, mode=parallel_mode)
                self._distributed_wrapper = model
            except Exception as e:
                logger.warning(f"wrap_model failed: {e}")
        
        return model
    
    def cleanup(self) -> None:
        """
        清理资源
        
        使用: clear_memory (backend/lib/hardware)
        """
        if self._distributed_manager:
            try:
                self._distributed_manager.cleanup()
            except Exception:
                pass
        
    
        try:
            clear_memory()
        except Exception:
            pass
        
        self._initialized = False
    
    def get_info(self) -> Dict[str, Any]:
        """获取上下文信息"""
        return {
            'device': str(self._device),
            'is_distributed': self.is_distributed,
            'distributed_info': self.get_distributed_info(),
            'encoders': list(self._encoders.keys()),
            'fusion': self._fusion is not None,
            'alignment': self._alignment is not None,
            'amp': self._amp_manager is not None,
            'memory_manager': self._memory_manager is not None,
            'gradient_checkpointing': self._gradient_checkpointing is not None,
            'contrastive_loss': self._contrastive_loss is not None,
            'loss_monitor': self._loss_monitor is not None,
            'layers_available': get_available_layers(),
        }


# ==================== 便捷函数 ====================

def create_production_strategy(
    strategy_type: str = "standard",
    **kwargs
) -> ProductionTrainingStrategy:
    """创建生产级策略"""
    config = ProductionStrategyConfig(**kwargs)
    return ProductionTrainingStrategy(config=config, name=strategy_type)


def create_production_context(
    config: Optional[ProductionStrategyConfig] = None,
    model: Optional[nn.Module] = None,
    **kwargs
) -> ProductionTrainingContext:
    """
    创建生产级训练上下文
    
    Args:
        config: 策略配置
        model: 模型
        **kwargs: 配置覆盖参数
    
    Returns:
        ProductionTrainingContext实例
    """
    if config is None:
        config = ProductionStrategyConfig(**kwargs)
    ctx = ProductionTrainingContext(config=config, model=model)
    ctx.initialize()
    return ctx


def get_available_layers() -> Dict[str, bool]:
    """获取可用的底层模块"""
    return {
        'hardware': True,
        'distributed': True,
        'adapters': True,
        'losses': True
    }


def get_layer_details() -> Dict[str, Dict[str, Any]]:
    """
    获取底层模块详细信息
    
    列出每层的可用组件
    """
    return {
        'hardware': {
            'available': True,
            'components': {
                'DeviceManager': DeviceManager is not None,
                'MixedPrecisionManager': MixedPrecisionManager is not None,
                'MemoryManager': MemoryManager is not None,
                'GradientCheckpointing': GradientCheckpointing is not None,
                'AmpContext': AmpContext is not None,
                'PrecisionMode': PrecisionMode is not None,
                'clear_memory': clear_memory is not None,
                'get_available_memory': get_available_memory is not None,
            }
        },
        'distributed': {
            'available': True,
            'components': {
                'DistributedManager': DistributedManager is not None,
                'DDPWrapper': DDPWrapper is not None,
                'FSDPWrapper': FSDPWrapper is not None,
                'ZeROWrapper': ZeROWrapper is not None,
                'PipelineWrapper': PipelineWrapper is not None,
                'DDPConfig': DDPConfig is not None,
                'FSDPConfig': FSDPConfig is not None,
                'ZeROConfig': ZeROConfig is not None,
                'PipelineConfig': PipelineConfig is not None,
                'barrier': barrier is not None,
                'all_reduce': all_reduce is not None,
                'AllReduceOp': AllReduceOp is not None,
                'is_main_process': is_main_process is not None,
                'get_rank': get_rank is not None,
                'get_world_size': get_world_size is not None,
            }
        },
        'adapters': {
            'available': True,
            'components': {
                'EncoderFactory': EncoderFactory is not None,
                'FusionFactory': FusionFactory is not None,
                'AlignmentFactory': AlignmentFactory is not None,
                'AdapterFactory': AdapterFactory is not None,
                'ModalityType': ModalityType is not None,
                'FusionMethod': FusionMethod is not None,
                'AlignmentMethod': AlignmentMethod is not None,
                'AdapterType': AdapterType is not None,
            }
        },
        'losses': {
            'available': True,
            'components': {
                'LossFactory': LossFactory is not None,
                'CompositeLoss': CompositeLoss is not None,
                'MultiTaskLoss': MultiTaskLoss is not None,
                'CrossEntropyLoss': CrossEntropyLoss is not None,
                'FocalLoss': FocalLoss is not None,
                'MSELoss': MSELoss is not None,
                'SoftLabelLoss': SoftLabelLoss is not None,
                'FeatureDistillationLoss': FeatureDistillationLoss is not None,
                'AttentionDistillationLoss': AttentionDistillationLoss is not None,
                'InfoNCELoss': InfoNCELoss is not None,
                'CLIPLoss': CLIPLoss is not None,
                'ContrastiveLoss': ContrastiveLoss is not None,
                'LossMonitor': LossMonitor is not None,
            }
        },
    }


def print_layer_info() -> None:
    """打印底层模块信息"""
    details = get_layer_details()
    
    print("\n" + "=" * 60)
    print("Production Base - Layer Information")
    print("=" * 60)
    
    for layer_name, layer_info in details.items():
        status = "✓" if layer_info['available'] else "✗"
        print(f"\n[{layer_name.upper()}] {status}")
        
        for comp_name, comp_available in layer_info['components'].items():
            status = "✓" if comp_available else "○"
            print(f"  {comp_name}: {status}")
    
    print("\n" + "=" * 60)


def create_composite_production_strategy(
    strategies: List[str],
    weights: Optional[List[float]] = None,
    **config_kwargs
):
    """
    创建组合生产级策略
    
    Args:
        strategies: 策略类型列表 ['multimodal', 'distillation', 'distributed']
        weights: 各策略权重
        **config_kwargs: 配置参数
    
    Returns:
        组合策略实例
    """
    from .base_strategy import CompositeStrategy
    
    strategy_instances = []
    for st in strategies:
        if st == 'production':
            strategy_instances.append(create_production_strategy(**config_kwargs))
        elif st == 'multimodal':
            from .multimodal_strategy import create_multimodal_strategy
            strategy_instances.append(create_multimodal_strategy(
                use_production_mode=True,
                **config_kwargs
            ))
        elif st == 'distillation':
            from .distillation_strategy import create_distillation_strategy
            strategy_instances.append(create_distillation_strategy('standard'))
        elif st == 'distributed':
            from .distributed_strategy import DistributedStrategy
            strategy_instances.append(DistributedStrategy())
        elif st == 'scenario':
            from .scenario_strategy import ScenarioStrategy
            strategy_instances.append(ScenarioStrategy())
        else:
            strategy_instances.append(create_production_strategy(st, **config_kwargs))
    
    return CompositeStrategy(strategy_instances, weights)


def diagnose_production_base() -> Dict[str, Any]:
    """诊断生产级基础模块"""
    diagnosis = {
        'layers': get_available_layers(),
        'layer_details': get_layer_details(),
        'issues': [],
        'recommendations': [],
    }
    
    return diagnosis


def print_production_base_diagnosis() -> None:
    """打印生产级基础模块诊断"""
    diagnosis = diagnose_production_base()
    
    print("\n" + "=" * 60)
    print("Production Base Diagnosis")
    print("=" * 60)
    
    print("\nLayers:")
    for layer, available in diagnosis['layers'].items():
        status = "✓" if available else "✗"
        print(f"  {layer}: {status}")
    
    if diagnosis['issues']:
        print("\nIssues:")
        for issue in diagnosis['issues']:
            print(f"  - {issue}")
    
    if diagnosis['recommendations']:
        print("\nRecommendations:")
        for rec in diagnosis['recommendations']:
            print(f"  - {rec}")
    
    print("\n" + "=" * 60)


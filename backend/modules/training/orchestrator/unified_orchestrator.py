# -*- coding: utf-8 -*-
"""
统一训练编排器

整合六层架构的统一训练编排器：
1. Training Orchestrator（训练编排层）
2. Training Strategy Abstraction（策略层）
3. Loss & Objective Composition（目标函数层）
4. Model & Modality Adapter Layer（模型/模态层）
5. Distributed Training Core（训练内核）
6. Hardware Abstraction（硬件抽象层）

架构图：
┌──────────────────────────────────────┐
│          Training Orchestrator       │ ← 当前模块
│     (Stage / Scenario / Strategy)    │
├──────────────────────────────────────┤
│       Training Strategy Abstraction  │ ← strategies/
│    (Standard / Distill / MultiTask)  │
├──────────────────────────────────────┤
│      Loss & Objective Composition    │ ← losses/
│   (Supervised / KD / Contrastive)    │
├──────────────────────────────────────┤
│      Model & Modality Adapter Layer  │ ← lib/multimodal/
│     (Text / Image / Audio / Fusion)  │
├──────────────────────────────────────┤
│        Distributed Training Core     │ ← strategies/distributed_strategy.py
│       (DDP / FSDP / Pipeline / ZeRO) │
├──────────────────────────────────────┤
│          Hardware Abstraction        │ ← hardware/
│        (GPU / NPU / TPU / CPU)       │
└──────────────────────────────────────┘
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Union, Type
from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ==================== 类型定义 ====================

class OrchestratorPhase(Enum):
    """编排器训练阶段"""
    # 标准阶段
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    PREFERENCE = "preference"
    
    # 行业阶段
    INDUSTRY_PRETRAIN = "industry_pretrain"
    INDUSTRY_ALIGN = "industry_align"
    SCENE_FINETUNE = "scene_finetune"
    
    # 多模态阶段
    MODALITY_PRETRAIN = "modality_pretrain"
    CROSS_MODAL_ALIGN = "cross_modal_align"
    INSTRUCTION_TUNING = "instruction_tuning"
    ALIGNMENT_SAFETY = "alignment_safety"
    
    # 蒸馏阶段
    TEACHER_PREPARE = "teacher_prepare"
    DISTILLATION = "distillation"
    STUDENT_FINETUNE = "student_finetune"


class OrchestratorStatus(Enum):
    """编排器状态"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
@dataclass
class LayerConfig:
    """层配置"""
    # 硬件层配置
    device_type: str = "auto"  # auto, cuda, cpu, mps
    device_ids: List[int] = field(default_factory=lambda: [0])
    precision: str = "fp16"  # fp32, fp16, bf16
    enable_amp: bool = True
    
    # 分布式层配置
    distributed_mode: str = "none"  # none, ddp, fsdp, zero
    world_size: int = 1
    gradient_accumulation_steps: int = 1
    
    # 模态层配置
    modalities: List[str] = field(default_factory=lambda: ["text"])
    fusion_method: str = "concat"  # concat, attention, cross_attention
    
    # 损失层配置
    task_loss_type: str = "cross_entropy"
    auxiliary_losses: List[Dict[str, Any]] = field(default_factory=list)
    
    # 策略层配置
    strategy_type: str = "standard"  # standard, distillation, multimodal, scenario
    strategy_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseConfig:
    """阶段配置"""
    phase: OrchestratorPhase
    epochs: int = 1
    learning_rate: float = 1e-4
    batch_size: int = 32
    warmup_ratio: float = 0.1
    
    # 层覆盖配置
    layer_overrides: Optional[LayerConfig] = None
    
    # 阶段特定配置
    freeze_layers: List[str] = field(default_factory=list)
    trainable_layers: List[str] = field(default_factory=list)
    
    # 回调
    on_epoch_start: Optional[Callable] = None
    on_epoch_end: Optional[Callable] = None


@dataclass
class OrchestratorPlan:
    """编排计划"""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "training_plan"
    description: str = ""
    
    # 阶段
    phases: List[PhaseConfig] = field(default_factory=list)
    
    # 全局配置
    global_config: LayerConfig = field(default_factory=LayerConfig)
    output_dir: str = "./outputs"
    
    # 检查点
    save_checkpoints: bool = True
    checkpoint_interval: int = 1  # epochs
    keep_last_n: int = 3
    
    # 回调
    on_plan_start: Optional[Callable] = None
    on_plan_end: Optional[Callable] = None
    on_phase_start: Optional[Callable] = None
    on_phase_end: Optional[Callable] = None


@dataclass
class PhaseResult:
    """阶段结果"""
    phase: OrchestratorPhase
    status: OrchestratorStatus
    metrics: Dict[str, float] = field(default_factory=dict)
    model_path: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


# ==================== 六层架构整合器 ====================

class LayerManager:
    """
    层管理器
    
    管理六层架构的初始化和协调。
    """
    
    def __init__(self, config: LayerConfig):
        self.config = config
        
        # 各层实例
        self._hardware = None
        self._distributed = None
        self._modality = None
        self._loss = None
        self._strategy = None
        
        self._initialized = False
    
    def initialize(self):
        """初始化所有层"""
        if self._initialized:
            return
        
        logger.info("Initializing layer manager...")
        
        # 1. 硬件层
        self._init_hardware_layer()
        
        # 2. 分布式层
        self._init_distributed_layer()
        
        # 3. 模态层
        self._init_modality_layer()
        
        # 4. 损失层
        self._init_loss_layer()
        
        # 5. 策略层
        self._init_strategy_layer()
        
        self._initialized = True
        logger.info("Layer manager initialized successfully")
    
    def check_hardware_health(self) -> Dict[str, Any]:
        """
        检查硬件健康状态
        
        调用 backend/lib/hardware 模块功能
        """
        status = {}
        if self._hardware:
            try:
                # 获取设备信息
                device = self._hardware.device
                status['device'] = str(device)
                
                # 检查内存
                if hasattr(self, '_memory_manager') and self._memory_manager:
                    try:
                        status['memory_usage'] = self._memory_manager.get_stats()
                    except Exception:
                        pass
                    
                # 检查混合精度状态
                if self._amp_manager:
                    status['amp_enabled'] = self._amp_manager.config.enabled
                    status['amp_precision'] = str(self._amp_manager.config.precision)
                    
            except Exception as e:
                logger.warning(f"Hardware health check failed: {e}")
                status['error'] = str(e)
        return status

    def optimize_memory(self):
        """
        优化内存使用
        
        调用 backend/lib/hardware 模块功能
        """
        if self._hardware:
            try:
                from backend.lib.hardware import clear_memory
                clear_memory()
                logger.info("Memory optimized via hardware layer")
            except Exception as e:
                logger.warning(f"Memory optimization failed: {e}")

    def _init_hardware_layer(self):
        """
        初始化硬件层
        
        使用 backend/lib/hardware 提供的能力：
        - DeviceManager: 设备检测和管理
        - MixedPrecisionManager: 混合精度训练
        - MemoryManager: 内存优化
        """
        try:
            from backend.lib.hardware import (
                DeviceManager, get_device_manager,
                MixedPrecisionManager, PrecisionMode,
                MemoryManager, clear_memory
            )
            from backend.lib.hardware.mixed_precision import AmpConfig
            
            # 设备管理器
            self._hardware = get_device_manager()
            # 创建 MemoryManager 实例
            device = self._hardware.device if self._hardware else None
            self._memory_manager = MemoryManager(device=device) if device else None
            self._clear_memory = clear_memory
            
            # 混合精度
            precision_map = {
                'fp32': PrecisionMode.FP32,
                'fp16': PrecisionMode.MIXED_FP16,
                'bf16': PrecisionMode.MIXED_BF16
            }
            amp_config = AmpConfig(
                enabled=self.config.enable_amp,
                precision=precision_map.get(self.config.precision, PrecisionMode.MIXED_FP16)
            )
            self._amp_manager = MixedPrecisionManager(amp_config, self._hardware.device)
            
            logger.info(f"Hardware layer initialized: device={self._hardware.device}, "
                       f"precision={self.config.precision}, amp={self.config.enable_amp}")
            
        except ImportError as e:
            logger.warning(f"Hardware layer import failed: {e}, using defaults")
            self._hardware = None
            self._memory_manager = None
            self._amp_manager = None
    
    def configure_distributed_environment(self, config: Dict[str, Any]) -> bool:
        """
        配置分布式环境
        
        调用 backend/lib/distributed 模块功能
        """
        if not self._distributed:
            return False
            
        try:
            from backend.lib.distributed import (
                DDPConfig, FSDPConfig, ZeROConfig, ParallelMode
            )
            
            mode = config.get('mode', 'ddp')
            
            if mode == 'ddp' and DDPConfig:
                ddp_config = DDPConfig(
                    find_unused_parameters=config.get('find_unused_parameters', False),
                    gradient_as_bucket_view=config.get('gradient_as_bucket_view', False)
                )
                # 假设 DistributedManager 有 configure 方法
                if hasattr(self._distributed, 'configure'):
                    self._distributed.configure(ParallelMode.DDP, ddp_config)
                    
            elif mode == 'fsdp' and FSDPConfig:
                fsdp_config = FSDPConfig(
                    sharding_strategy=config.get('sharding_strategy', 'FULL_SHARD'),
                    mixed_precision=config.get('mixed_precision', True)
                )
                if hasattr(self._distributed, 'configure'):
                    self._distributed.configure(ParallelMode.FSDP, fsdp_config)
                    
            elif mode == 'zero' and ZeROConfig:
                zero_config = ZeROConfig(
                    stage=config.get('stage', 2),
                    offload_optimizer=config.get('offload_optimizer', False)
                )
                if hasattr(self._distributed, 'configure'):
                    self._distributed.configure(ParallelMode.ZERO_2, zero_config)
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to configure distributed environment: {e}")
            return False

    def get_distributed_status(self) -> Dict[str, Any]:
        """获取分布式状态"""
        if not self._distributed:
            return {'enabled': False}
            
        try:
            return {
                'enabled': True,
                'world_size': self._distributed.world_size if hasattr(self._distributed, 'world_size') else 1,
                'rank': self._distributed.rank if hasattr(self._distributed, 'rank') else 0,
                'backend': self._distributed.backend if hasattr(self._distributed, 'backend') else 'unknown'
            }
        except Exception:
            return {'enabled': True, 'status': 'unknown'}

    def _init_distributed_layer(self):
        """
        初始化分布式层
        
        使用 backend/lib/distributed 提供的能力：
        - DDPWrapper: 数据并行
        - FSDPWrapper: 全分片数据并行
        - ZeROWrapper: DeepSpeed ZeRO优化
        - PipelineWrapper: 流水线并行
        """
        if self.config.distributed_mode == "none":
            self._distributed = None
            return
        
        try:
            # 优先使用底层库
            from backend.lib.distributed import (
                DistributedManager, get_distributed_manager, ParallelMode,
                DDPConfig, FSDPConfig, ZeROConfig
            )
            
            mode_map = {
                'ddp': ParallelMode.DDP,
                'fsdp': ParallelMode.FSDP,
                'zero': ParallelMode.ZERO_2,
                'zero1': ParallelMode.ZERO_1,
                'zero2': ParallelMode.ZERO_2,
                'zero3': ParallelMode.ZERO_3,
                'pipeline': ParallelMode.PIPELINE
            }
            
            self._distributed = get_distributed_manager()
            self._distributed_mode = mode_map.get(self.config.distributed_mode, ParallelMode.DDP)
            
            # 保存配置类引用供后续使用
            self._distributed_configs = {
                'DDP': DDPConfig,
                'FSDP': FSDPConfig,
                'ZeRO': ZeROConfig
            }
            
            if self.config.world_size > 1:
                self._distributed.initialize(
                    backend='nccl' if torch.cuda.is_available() else 'gloo',
                    world_size=self.config.world_size
                )
            
            logger.info(f"Distributed layer initialized: mode={self.config.distributed_mode}, "
                       f"world_size={self.config.world_size}")
            
        except ImportError:
            # 回退到策略层
            try:
                from backend.modules.training.strategies import (
                    DistributedStrategy, DistributedStrategyConfig, DistributedMode
                )
                
                mode_map = {
                    'ddp': DistributedMode.DDP,
                    'fsdp': DistributedMode.FSDP,
                    'zero': DistributedMode.ZERO,
                    'pipeline': DistributedMode.PIPELINE
                }
                
                dist_config = DistributedStrategyConfig(
                    mode=mode_map.get(self.config.distributed_mode, DistributedMode.DDP),
                    world_size=self.config.world_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps
                )
                
                self._distributed = DistributedStrategy(dist_config)
                logger.info(f"Distributed layer (fallback): mode={self.config.distributed_mode}")
                
            except ImportError as e:
                logger.warning(f"Distributed layer import failed: {e}")
                self._distributed = None
    
    def setup_modality_adapters(self, model: nn.Module, modalities: List[str]) -> nn.Module:
        """
        设置模态适配器
        
        调用 backend/lib/adapters 模块功能
        """
        if not self._modality:
            return model
            
        try:
            # 优先使用 AdapterManager
            if 'adapter_manager' in self._modality and self._modality['adapter_manager']:
                manager = self._modality['adapter_manager']
                if hasattr(manager, 'setup_adapters'):
                    return manager.setup_adapters(model, modalities)
            
            # 手动创建适配器
            if 'create_encoder' in self._modality:
                encoders = {}
                for modality in modalities:
                    if modality != 'text': # 假设text是基础模态
                        encoders[modality] = self._modality['create_encoder'](modality)
                
                # 如果模型支持注册编码器
                if hasattr(model, 'register_encoders'):
                    model.register_encoders(encoders)
                    
            # 设置融合模块
            if 'create_fusion' in self._modality and len(modalities) > 1:
                fusion = self._modality['create_fusion'](
                    method=self.config.fusion_method,
                    modalities=modalities
                )
                if hasattr(model, 'set_fusion_module'):
                    model.set_fusion_module(fusion)
                    
            # 设置对齐模块
            if 'create_alignment' in self._modality and len(modalities) > 1:
                alignment = self._modality['create_alignment'](modalities)
                if hasattr(model, 'set_alignment_module'):
                    model.set_alignment_module(alignment)
                    
        except Exception as e:
            logger.warning(f"Failed to setup modality adapters: {e}")
            
        return model

    def _init_modality_layer(self):
        """
        初始化模态层
        
        使用 backend/lib/adapters 提供的能力：
        - ModalityEncoder: 模态编码器（Text/Image/Audio/Video/TimeSeries）
        - FusionModule: 融合模块（Early/Middle/Late/CrossAttention）
        - AlignmentModule: 对齐模块（Contrastive/Explicit/OT）
        """
        if len(self.config.modalities) <= 1 and 'text' in self.config.modalities:
            self._modality = None
            return
        
        try:
            # 优先使用 adapters 层
            from backend.lib.adapters import (
                AdapterManager, get_adapter_manager,
                EncoderFactory, create_encoder,
                FusionFactory, create_fusion,
                AlignmentFactory, create_alignment
            )
            
            self._adapter_manager = get_adapter_manager()
            self._modality = {
                'adapter_manager': self._adapter_manager,
                'create_encoder': create_encoder,
                'create_fusion': create_fusion,
                'create_alignment': create_alignment,
                'factories': {
                    'encoder': EncoderFactory,
                    'fusion': FusionFactory,
                    'alignment': AlignmentFactory
                }
            }
            logger.info(f"Modality layer initialized (adapters): modalities={self.config.modalities}")
            
        except ImportError:
            # 回退到 multimodal 层
            try:
                from backend.lib.multimodal import (
                    ModalityEncoderFactory,
                    MultiModalFuser,
                    CrossModalAligner
                )
                
                self._modality = {
                    'encoder_factory': ModalityEncoderFactory,
                    'fuser_class': MultiModalFuser,
                    'aligner_class': CrossModalAligner
                }
                logger.info(f"Modality layer (fallback): modalities={self.config.modalities}")
                
            except ImportError as e:
                logger.warning(f"Modality layer import failed: {e}")
                self._modality = None
    
    def create_advanced_loss(self, loss_type: str, **kwargs) -> Optional[nn.Module]:
        """
        创建高级损失函数
        
        调用 backend/lib/losses 模块功能 (CompositeLoss, MultiTaskLoss)
        """
        if not self._loss:
            return None
            
        try:
            from backend.lib.losses import CompositeLoss, MultiTaskLoss
            
            if loss_type == 'composite' and CompositeLoss:
                # 假设 kwargs 中包含 'losses' 列表
                sub_losses = kwargs.get('losses', [])
                weights = kwargs.get('weights', None)
                return CompositeLoss(sub_losses, weights)
                
            elif loss_type == 'multitask' and MultiTaskLoss:
                # 假设 kwargs 中包含 'task_losses' 字典
                task_losses = kwargs.get('task_losses', {})
                return MultiTaskLoss(task_losses)
                
            # 回退到标准创建
            return self._loss['create_loss'](loss_type, **kwargs)
            
        except Exception as e:
            logger.warning(f"Failed to create advanced loss: {e}")
            return None

    def _init_loss_layer(self):
        """
        初始化损失层
        
        使用 backend/lib/losses 提供的能力：
        - SupervisedLoss: 分类/回归/分割损失
        - DistillationLoss: 知识蒸馏损失
        - ContrastiveLoss: 对比学习损失
        - CompositeLoss: 复合损失
        """
        try:
            from backend.lib.losses import (
                LossFactory, create_loss, create_composite_loss, create_distillation_loss,
                CrossEntropyLoss, FocalLoss, MSELoss,
                InfoNCELoss, CLIPLoss,
                CompositeLoss, MultiTaskLoss
            )
            
            self._loss = {
                'factory': LossFactory,
                'create_loss': create_loss,
                'create_composite': create_composite_loss,
                'create_distillation': create_distillation_loss,
                'classes': {
                    'CompositeLoss': CompositeLoss,
                    'MultiTaskLoss': MultiTaskLoss
                },
                # 常用损失
                'cross_entropy': CrossEntropyLoss,
                'focal': FocalLoss,
                'mse': MSELoss,
                'infonce': InfoNCELoss,
                'clip': CLIPLoss,
            }
            logger.info(f"Loss layer initialized: task_loss={self.config.task_loss_type}")
            
        except ImportError as e:
            logger.warning(f"Loss layer import failed: {e}, using PyTorch defaults")
            self._loss = None
    
    def create_custom_strategy(self, strategy_type: str, model: nn.Module = None, **kwargs) -> Optional[Any]:
        """
        创建自定义策略
        
        调用 backend/modules/training/strategies 模块功能
        """
        if not self._strategy:
            return None
            
        try:
            from backend.modules.training.strategies import TrainingStrategy
            
            # 创建上下文
            context = None
            if 'create_production_context' in self._strategy:
                # 使用 LayerConfig 中的信息
                prod_config = self._strategy['production_config'](
                    device=self.config.device_type,
                    precision=self.config.precision
                )
                context = self._strategy['create_production_context'](prod_config, model)
            
            # 创建策略
            strategy = self._strategy['create'](strategy_type, **kwargs)
            
            # 初始化策略
            if strategy and isinstance(strategy, TrainingStrategy) and context:
                strategy.setup(context)
                
            return strategy
            
        except Exception as e:
            logger.warning(f"Failed to create custom strategy: {e}")
            return None

    def _init_strategy_layer(self):
        """
        初始化策略层
        
        使用 backend/modules/training/strategies 提供的能力：
        - ProductionTrainingStrategy: 生产级训练策略
        - ProductionTrainingContext: 生产级训练上下文
        - MultiModalStrategy: 多模态训练策略
        - DistillationStrategy: 知识蒸馏策略
        - IndustryScenarioStrategy: 行业场景策略
        - DistributedStrategy: 分布式训练策略
        """
        try:
            from backend.modules.training.strategies import (
                create_strategy, TrainingStrategy,
                StandardTrainingStrategy, CompositeStrategy,
                ProductionTrainingStrategy, ProductionStrategyConfig,
                ProductionTrainingContext, create_production_context,
                create_composite_production_strategy,
                get_available_layers
            )
            
            self._strategy = {
                'create': create_strategy,
                'standard': StandardTrainingStrategy,
                'composite': CompositeStrategy,
                'production': ProductionTrainingStrategy,
                'production_config': ProductionStrategyConfig,
                'create_production_context': create_production_context,
                'create_composite_production': create_composite_production_strategy,
                'get_available_layers': get_available_layers,
            }
            
            # 记录可用的底层模块
            available = get_available_layers()
            logger.info(f"Strategy layer initialized: type={self.config.strategy_type}")
            logger.info(f"  Available layers: {available}")
            
        except ImportError as e:
            logger.warning(f"Strategy layer import failed: {e}")
            self._strategy = None
    
    @property
    def device(self) -> torch.device:
        """获取设备"""
        if self._hardware:
            return self._hardware.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @property
    def amp_context(self):
        """获取AMP上下文"""
        if self._amp_manager:
            return self._amp_manager.autocast_context()
        from contextlib import nullcontext
        return nullcontext()
    
    def create_loss(self, loss_type: str, **kwargs):
        """创建损失函数"""
        if self._loss:
            return self._loss['create_loss'](loss_type, **kwargs)
        
        # 默认实现
        import torch.nn as nn
        loss_map = {
            'cross_entropy': nn.CrossEntropyLoss,
            'mse': nn.MSELoss,
            'bce': nn.BCEWithLogitsLoss
        }
        return loss_map.get(loss_type, nn.CrossEntropyLoss)()
    
    def create_strategy(self, strategy_type: str, **kwargs):
        """创建训练策略"""
        if self._strategy:
            return self._strategy['create'](strategy_type, **kwargs)
        return None
    
    def wrap_model_distributed(self, model: nn.Module) -> nn.Module:
        """
        包装分布式模型
        
        使用分布式层能力包装模型以支持：
        - DDP: 数据并行
        - FSDP: 全分片数据并行
        - ZeRO: DeepSpeed优化
        - Pipeline: 流水线并行
        """
        if self._distributed is None:
            return model
        
        try:
            # 如果是 DistributedManager
            if hasattr(self._distributed, 'wrap_model'):
                return self._distributed.wrap_model(
                    model, 
                    mode=self._distributed_mode if hasattr(self, '_distributed_mode') else None
                )
            # 如果是 DistributedStrategy
            elif hasattr(self._distributed, 'setup'):
                from backend.modules.training.strategies.base_strategy import StrategyContext
                ctx = StrategyContext(model=model, device=self.device)
                self._distributed.setup(ctx)
                return ctx.model
        except Exception as e:
            logger.warning(f"Failed to wrap model for distributed: {e}")
        
        return model
    
    def create_production_context(self, model: nn.Module = None) -> Optional[Any]:
        """
        创建生产级训练上下文
        
        整合六层架构的统一上下文。
        """
        if self._strategy and 'create_production_context' in self._strategy:
            config = self._strategy['production_config'](
                device=self.config.device_type,
                precision=self.config.precision,
                enable_amp=self.config.enable_amp,
                distributed_mode=self.config.distributed_mode,
                world_size=self.config.world_size,
                modalities=self.config.modalities,
                task_loss_type=self.config.task_loss_type,
            )
            return self._strategy['create_production_context'](config, model)
        return None
    
    def register_modality_component(self, factory_type: str, name: str, component_cls: Any) -> bool:
        """注册模态组件"""
        if not self._modality or 'factories' not in self._modality:
            return False
        
        factory = self._modality['factories'].get(factory_type)
        if factory and hasattr(factory, 'register'):
            try:
                factory.register(name, component_cls)
                logger.info(f"Registered {factory_type} component: {name}")
                return True
            except Exception as e:
                logger.warning(f"Failed to register component: {e}")
        return False

    def create_distributed_config_object(self, config_type: str, **kwargs) -> Optional[Any]:
        """创建分布式配置对象"""
        if hasattr(self, '_distributed_configs'):
            config_cls = self._distributed_configs.get(config_type)
            if config_cls:
                try:
                    return config_cls(**kwargs)
                except Exception as e:
                    logger.warning(f"Failed to create config {config_type}: {e}")
        return None

    def manage_memory(self, action: str = 'stats') -> Any:
        """管理内存"""
        if action == 'clear' and hasattr(self, '_clear_memory'):
            try:
                self._clear_memory()
                return True
            except Exception:
                return False
        elif action == 'stats' and hasattr(self, '_memory_manager'):
            try:
                if self._memory_manager:
                    return self._memory_manager.get_stats()
                return {}
            except Exception:
                return {}
        return None

    def get_available_layers(self) -> Dict[str, bool]:
        """获取可用的底层模块"""
        if self._strategy and 'get_available_layers' in self._strategy:
            return self._strategy['get_available_layers']()
        return {
            'hardware': self._hardware is not None,
            'distributed': self._distributed is not None,
            'adapters': self._modality is not None,
            'losses': self._loss is not None,
        }
    
    def cleanup(self):
        """清理资源"""
        if self._distributed:
            if hasattr(self._distributed, 'cleanup'):
                self._distributed.cleanup()
        
        if hasattr(self, '_amp_manager') and self._amp_manager:
            pass  # AMP manager通常不需要清理
        
        if self._hardware:
            try:
                from backend.lib.hardware import clear_memory
                clear_memory()
            except ImportError:
                pass
        
        logger.info("Layer manager cleaned up")


# ==================== 统一训练编排器 ====================

class UnifiedTrainingOrchestrator:
    """
    统一训练编排器
    
    整合六层架构，提供统一的训练编排接口。
    """
    
    def __init__(
        self,
        output_dir: str = "./training_outputs",
        default_config: Optional[LayerConfig] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_config = default_config or LayerConfig()
        
        # 层管理器
        self._layer_manager: Optional[LayerManager] = None
        
        # 当前状态
        self._status = OrchestratorStatus.IDLE
        self._current_plan: Optional[OrchestratorPlan] = None
        self._current_phase_idx: int = 0
        self._phase_results: List[PhaseResult] = []
        
        # 模型和数据
        self._model: Optional[nn.Module] = None
        self._train_loader: Optional[DataLoader] = None
        self._val_loader: Optional[DataLoader] = None
        
        # 控制标志
        self._should_stop = False
        self._should_pause = False
        
        # 回调
        self.callbacks: Dict[str, List[Callable]] = {
            'on_train_start': [],
            'on_train_end': [],
            'on_phase_start': [],
            'on_phase_end': [],
            'on_epoch_start': [],
            'on_epoch_end': [],
            'on_step_start': [],
            'on_step_end': []
        }
        
        logger.info(f"UnifiedTrainingOrchestrator initialized: output_dir={output_dir}")
    
    # ==================== 计划创建 ====================
    
    def create_plan(
        self,
        name: str,
        phases: List[Union[OrchestratorPhase, str, Dict[str, Any]]],
        global_config: Optional[LayerConfig] = None,
        **kwargs
    ) -> OrchestratorPlan:
        """
        创建训练计划
        
        Args:
            name: 计划名称
            phases: 阶段列表
            global_config: 全局配置
            **kwargs: 其他配置
        """
        plan = OrchestratorPlan(
            name=name,
            global_config=global_config or self.default_config,
            output_dir=str(self.output_dir / name)
        )
        
        for phase_def in phases:
            if isinstance(phase_def, OrchestratorPhase):
                plan.phases.append(PhaseConfig(phase=phase_def))
            elif isinstance(phase_def, str):
                plan.phases.append(PhaseConfig(phase=OrchestratorPhase(phase_def)))
            elif isinstance(phase_def, dict):
                phase_config = PhaseConfig(
                    phase=OrchestratorPhase(phase_def['phase']),
                    **{k: v for k, v in phase_def.items() if k != 'phase'}
                )
                plan.phases.append(phase_config)
        
        for key, value in kwargs.items():
            if hasattr(plan, key):
                setattr(plan, key, value)
        
        logger.info(f"Plan created: {plan.name}, phases={[p.phase.value for p in plan.phases]}")
        return plan
    
    def create_standard_plan(
        self,
        name: str = "standard_training",
        epochs: int = 10,
        learning_rate: float = 1e-4
    ) -> OrchestratorPlan:
        """创建标准训练计划"""
        return self.create_plan(
            name=name,
            phases=[{
                'phase': 'finetune',
                'epochs': epochs,
                'learning_rate': learning_rate
            }]
        )
    
    def create_three_stage_plan(
        self,
        name: str = "three_stage_training",
        pretrain_epochs: int = 3,
        finetune_epochs: int = 5,
        preference_epochs: int = 2
    ) -> OrchestratorPlan:
        """创建三阶段训练计划"""
        return self.create_plan(
            name=name,
            phases=[
                {'phase': 'pretrain', 'epochs': pretrain_epochs},
                {'phase': 'finetune', 'epochs': finetune_epochs},
                {'phase': 'preference', 'epochs': preference_epochs}
            ]
        )
    
    def create_industry_plan(
        self,
        name: str = "industry_training",
        include_pretrain: bool = True,
        include_align: bool = True,
        include_finetune: bool = True
    ) -> OrchestratorPlan:
        """创建行业模型训练计划"""
        phases = []
        
        if include_pretrain:
            phases.append({
                'phase': 'industry_pretrain',
                'epochs': 3,
                'learning_rate': 1e-4
            })
        
        if include_align:
            phases.append({
                'phase': 'industry_align',
                'epochs': 5,
                'learning_rate': 2e-5
            })
        
        if include_finetune:
            phases.append({
                'phase': 'scene_finetune',
                'epochs': 10,
                'learning_rate': 1e-5,
                'freeze_layers': ['backbone']
            })
        
        return self.create_plan(name=name, phases=phases)
    
    def create_multimodal_plan(
        self,
        name: str = "multimodal_training",
        modalities: List[str] = None
    ) -> OrchestratorPlan:
        """创建多模态训练计划"""
        global_config = LayerConfig(
            modalities=modalities or ['text', 'image'],
            strategy_type='multimodal'
        )
        
        return self.create_plan(
            name=name,
            phases=[
                {'phase': 'modality_pretrain', 'epochs': 3},
                {'phase': 'cross_modal_align', 'epochs': 5},
                {'phase': 'instruction_tuning', 'epochs': 3},
                {'phase': 'alignment_safety', 'epochs': 2}
            ],
            global_config=global_config
        )
    
    def create_distillation_plan(
        self,
        name: str = "distillation_training",
        distillation_epochs: int = 10
    ) -> OrchestratorPlan:
        """创建知识蒸馏训练计划"""
        global_config = LayerConfig(
            strategy_type='distillation',
            strategy_config={
                'temperature': 4.0,
                'soft_loss_weight': 1.0,
                'hard_loss_weight': 0.5
            }
        )
        
        return self.create_plan(
            name=name,
            phases=[
                {'phase': 'teacher_prepare', 'epochs': 0},
                {'phase': 'distillation', 'epochs': distillation_epochs},
                {'phase': 'student_finetune', 'epochs': 2}
            ],
            global_config=global_config
        )
    
    # ==================== 训练执行 ====================
    
    def execute(
        self,
        plan: OrchestratorPlan,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        resume_from: Optional[str] = None
    ) -> List[PhaseResult]:
        """
        执行训练计划
        
        Args:
            plan: 训练计划
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            resume_from: 恢复检查点路径
        """
        self._current_plan = plan
        self._model = model
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._phase_results = []
        self._should_stop = False
        self._should_pause = False
        
        # 初始化层管理器
        self._layer_manager = LayerManager(plan.global_config)
        self._layer_manager.initialize()
        
        # 移动模型到设备
        self._model = self._model.to(self._layer_manager.device)
        
        # 可能的分布式包装
        self._model = self._layer_manager.wrap_model_distributed(self._model)
        
        # 触发回调
        self._trigger_callback('on_train_start', plan=plan)
        if plan.on_plan_start:
            plan.on_plan_start(plan)
        
        self._status = OrchestratorStatus.RUNNING
        
        try:
            # 恢复检查点
            start_phase_idx = 0
            if resume_from:
                start_phase_idx = self._load_checkpoint(resume_from)
            
            # 执行各阶段
            for idx in range(start_phase_idx, len(plan.phases)):
                if self._should_stop:
                    self._status = OrchestratorStatus.CANCELLED
                    break
                
                while self._should_pause:
                    time.sleep(0.5)
                
                self._current_phase_idx = idx
                phase_config = plan.phases[idx]
                
                result = self._execute_phase(phase_config)
                self._phase_results.append(result)
                
                if result.status == OrchestratorStatus.FAILED:
                    self._status = OrchestratorStatus.FAILED
                    break
            
            if self._status == OrchestratorStatus.RUNNING:
                self._status = OrchestratorStatus.COMPLETED
        
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._status = OrchestratorStatus.FAILED
            raise
        
        finally:
            # 触发回调
            self._trigger_callback('on_train_end', results=self._phase_results)
            if plan.on_plan_end:
                plan.on_plan_end(plan, self._phase_results)
            
            # 清理
            self._layer_manager.cleanup()
        
        return self._phase_results
    
    def _execute_phase(self, phase_config: PhaseConfig) -> PhaseResult:
        """执行单个阶段"""
        phase = phase_config.phase
        logger.info(f"Starting phase: {phase.value}")
        
        result = PhaseResult(
            phase=phase,
            status=OrchestratorStatus.RUNNING,
            start_time=datetime.now()
        )
        
        # 触发回调
        self._trigger_callback('on_phase_start', phase=phase)
        if self._current_plan.on_phase_start:
            self._current_plan.on_phase_start(phase)
        
        try:
            # 应用阶段配置
            effective_config = self._merge_config(
                self._current_plan.global_config,
                phase_config.layer_overrides
            )
            
            # 冻结/解冻层
            self._apply_layer_freezing(phase_config)
            
            # 创建优化器
            optimizer = self._create_optimizer(phase_config)
            scheduler = self._create_scheduler(optimizer, phase_config)
            
            # 创建损失函数
            loss_fn = self._layer_manager.create_loss(effective_config.task_loss_type)
            
            # 训练循环
            all_metrics = []
            for epoch in range(phase_config.epochs):
                if self._should_stop:
                    break
                
                epoch_metrics = self._train_epoch(
                    epoch, phase_config, optimizer, scheduler, loss_fn
                )
                all_metrics.append(epoch_metrics)
                
                # 保存检查点
                if self._current_plan.save_checkpoints:
                    if (epoch + 1) % self._current_plan.checkpoint_interval == 0:
                        self._save_checkpoint(phase, epoch)
            
            # 聚合指标
            result.metrics = self._aggregate_metrics(all_metrics)
            result.status = OrchestratorStatus.COMPLETED
            
            # 保存模型
            model_path = self._save_phase_model(phase)
            result.model_path = model_path
        
        except Exception as e:
            logger.error(f"Phase {phase.value} failed: {e}")
            result.status = OrchestratorStatus.FAILED
            result.error = str(e)
        
        finally:
            result.end_time = datetime.now()
            
            # 触发回调
            self._trigger_callback('on_phase_end', phase=phase, result=result)
            if self._current_plan.on_phase_end:
                self._current_plan.on_phase_end(phase, result)
        
        logger.info(f"Phase {phase.value} completed: {result.status.value}, "
                   f"duration={result.duration_seconds:.1f}s")
        
        return result
    
    def _train_epoch(
        self,
        epoch: int,
        phase_config: PhaseConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """训练单个epoch"""
        self._trigger_callback('on_epoch_start', epoch=epoch)
        if phase_config.on_epoch_start:
            phase_config.on_epoch_start(epoch)
        
        self._model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self._train_loader):
            if self._should_stop:
                break
            
            self._trigger_callback('on_step_start', step=batch_idx)
            
            # 移动到设备
            batch = self._layer_manager._hardware.to_device(batch) if self._layer_manager._hardware else batch
            
            # 前向传播（带混合精度）
            with self._layer_manager.amp_context:
                outputs = self._model(batch['input_ids'] if isinstance(batch, dict) else batch)
                labels = batch.get('labels', batch.get('targets')) if isinstance(batch, dict) else None
                
                if labels is not None:
                    loss = loss_fn(outputs, labels)
                else:
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs
            
            # 反向传播
            optimizer.zero_grad()
            
            if self._layer_manager._amp_manager:
                self._layer_manager._amp_manager.backward(loss)
                self._layer_manager._amp_manager.step(optimizer)
            else:
                loss.backward()
                optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            self._trigger_callback('on_step_end', step=batch_idx, loss=loss.item())
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # 验证
        val_metrics = {}
        if self._val_loader:
            val_metrics = self._validate()
        
        epoch_metrics = {
            'train_loss': avg_loss,
            **val_metrics
        }
        
        self._trigger_callback('on_epoch_end', epoch=epoch, metrics=epoch_metrics)
        if phase_config.on_epoch_end:
            phase_config.on_epoch_end(epoch, epoch_metrics)
        
        logger.info(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")
        
        return epoch_metrics
    
    def _validate(self) -> Dict[str, float]:
        """验证"""
        self._model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self._val_loader:
                batch = self._layer_manager._hardware.to_device(batch) if self._layer_manager._hardware else batch
                
                outputs = self._model(batch['input_ids'] if isinstance(batch, dict) else batch)
                labels = batch.get('labels', batch.get('targets')) if isinstance(batch, dict) else None
                
                if labels is not None and hasattr(outputs, 'loss'):
                    total_loss += outputs.loss.item()
                    num_batches += 1
        
        return {'val_loss': total_loss / max(num_batches, 1)}
    
    # ==================== 辅助方法 ====================
    
    def _merge_config(
        self, 
        global_config: LayerConfig, 
        override: Optional[LayerConfig]
    ) -> LayerConfig:
        """合并配置"""
        if override is None:
            return global_config
        
        # 简单合并，优先使用override中的非默认值
        merged = LayerConfig()
        for field_info in fields(LayerConfig):
            field_name = field_info.name
            global_val = getattr(global_config, field_name)
            override_val = getattr(override, field_name, None)
            
            # 使用override值如果不是默认值
            default_val = getattr(LayerConfig(), field_name)
            if override_val != default_val:
                setattr(merged, field_name, override_val)
            else:
                setattr(merged, field_name, global_val)
        
        return merged
    
    def _apply_layer_freezing(self, phase_config: PhaseConfig):
        """应用层冻结"""
        # 先解冻所有
        for param in self._model.parameters():
            param.requires_grad = True
        
        # 冻结指定层
        for layer_name in phase_config.freeze_layers:
            for name, param in self._model.named_parameters():
                if layer_name in name:
                    param.requires_grad = False
        
        # 如果指定了trainable_layers，冻结其他所有
        if phase_config.trainable_layers:
            for name, param in self._model.named_parameters():
                trainable = any(tl in name for tl in phase_config.trainable_layers)
                param.requires_grad = trainable
    
    def _create_optimizer(
        self, 
        phase_config: PhaseConfig
    ) -> torch.optim.Optimizer:
        """创建优化器"""
        trainable_params = [p for p in self._model.parameters() if p.requires_grad]
        
        return torch.optim.AdamW(
            trainable_params,
            lr=phase_config.learning_rate,
            weight_decay=0.01
        )
    
    def _create_scheduler(
        self, 
        optimizer: torch.optim.Optimizer,
        phase_config: PhaseConfig
    ) -> Optional[Any]:
        """创建学习率调度器"""
        if phase_config.warmup_ratio <= 0:
            return None
        
        total_steps = len(self._train_loader) * phase_config.epochs
        warmup_steps = int(total_steps * phase_config.warmup_ratio)
        
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps
        )
    
    def _aggregate_metrics(
        self, 
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """聚合指标"""
        if not metrics_list:
            return {}
        
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            aggregated[f"avg_{key}"] = sum(values) / len(values)
            aggregated[f"final_{key}"] = values[-1]
        
        return aggregated
    
    def _save_checkpoint(self, phase: OrchestratorPhase, epoch: int):
        """保存检查点"""
        checkpoint_dir = Path(self._current_plan.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{phase.value}_epoch_{epoch}.pt"
        
        torch.save({
            'phase': phase.value,
            'epoch': epoch,
            'model_state_dict': self._model.state_dict(),
            'phase_results': [r.__dict__ for r in self._phase_results]
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _load_checkpoint(self, path: str) -> int:
        """加载检查点"""
        checkpoint = torch.load(path)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Checkpoint loaded: {path}")
        
        # 返回下一个阶段索引
        return checkpoint.get('phase_idx', 0) + 1
    
    def _save_phase_model(self, phase: OrchestratorPhase) -> str:
        """保存阶段模型"""
        model_dir = Path(self._current_plan.output_dir) / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f"{phase.value}_model.pt"
        torch.save(self._model.state_dict(), model_path)
        
        logger.info(f"Phase model saved: {model_path}")
        return str(model_path)
    
    def _trigger_callback(self, event: str, **kwargs):
        """触发回调"""
        for callback in self.callbacks.get(event, []):
            try:
                callback(**kwargs)
            except Exception as e:
                logger.warning(f"Callback {event} failed: {e}")
    
    # ==================== 控制方法 ====================
    
    def pause(self):
        """暂停训练"""
        self._should_pause = True
        self._status = OrchestratorStatus.PAUSED
        logger.info("Training paused")
    
    def resume(self):
        """恢复训练"""
        self._should_pause = False
        self._status = OrchestratorStatus.RUNNING
        logger.info("Training resumed")
    
    def stop(self):
        """停止训练"""
        self._should_stop = True
        logger.info("Training stopping...")
    
    def add_callback(self, event: str, callback: Callable):
        """添加回调"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    @property
    def status(self) -> OrchestratorStatus:
        """获取状态"""
        return self._status
    
    @property
    def progress(self) -> Dict[str, Any]:
        """获取进度"""
        if not self._current_plan:
            return {'status': self._status.value}
        
        total_phases = len(self._current_plan.phases)
        completed_phases = len(self._phase_results)
        
        return {
            'status': self._status.value,
            'plan_name': self._current_plan.name,
            'total_phases': total_phases,
            'completed_phases': completed_phases,
            'current_phase': self._current_plan.phases[self._current_phase_idx].phase.value if self._current_phase_idx < total_phases else None,
            'progress_percent': (completed_phases / total_phases) * 100 if total_phases > 0 else 0
        }


    def diagnose(self) -> Dict[str, Any]:
        """诊断编排器状态"""
        diagnosis = {
            'orchestrator_status': self._status.value,
            'output_dir': str(self.output_dir),
            'layer_manager_initialized': self._layer_manager is not None,
            'model_loaded': self._model is not None,
            'train_loader': self._train_loader is not None,
            'val_loader': self._val_loader is not None,
            'callbacks': {k: len(v) for k, v in self.callbacks.items()},
        }
        
        if self._current_plan:
            diagnosis['current_plan'] = {
                'plan_id': self._current_plan.plan_id,
                'name': self._current_plan.name,
                'total_phases': len(self._current_plan.phases),
                'current_phase_idx': self._current_phase_idx,
                'phases': [p.phase.value for p in self._current_plan.phases],
            }
        
        if self._layer_manager:
            diagnosis['layer_availability'] = self._layer_manager.get_available_layers()
        
        if self._phase_results:
            diagnosis['phase_results'] = [
                {
                    'phase': r.phase.value,
                    'status': r.status.value,
                    'duration_seconds': r.duration_seconds,
                    'metrics': r.metrics,
                }
                for r in self._phase_results
            ]
        
        return diagnosis
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        return {
            'status': self._status.value,
            'plan_name': self._current_plan.name if self._current_plan else None,
            'phases_completed': len(self._phase_results),
            'total_phases': len(self._current_plan.phases) if self._current_plan else 0,
            'total_duration': sum(r.duration_seconds for r in self._phase_results),
            'final_metrics': self._phase_results[-1].metrics if self._phase_results else {},
        }


# ==================== 工厂函数 ====================

def create_orchestrator(
    output_dir: str = "./training_outputs",
    **config_kwargs
) -> UnifiedTrainingOrchestrator:
    """
    创建统一训练编排器
    
    Args:
        output_dir: 输出目录
        **config_kwargs: LayerConfig的参数
    """
    config = LayerConfig(**config_kwargs)
    return UnifiedTrainingOrchestrator(output_dir, config)


def create_quick_plan(
    plan_type: str,
    name: Optional[str] = None,
    **kwargs
) -> OrchestratorPlan:
    """
    快速创建训练计划
    
    Args:
        plan_type: standard, three_stage, industry, multimodal, distillation
        name: 计划名称
        **kwargs: 其他参数
    """
    orchestrator = UnifiedTrainingOrchestrator()
    
    creators = {
        'standard': orchestrator.create_standard_plan,
        'three_stage': orchestrator.create_three_stage_plan,
        'industry': orchestrator.create_industry_plan,
        'multimodal': orchestrator.create_multimodal_plan,
        'distillation': orchestrator.create_distillation_plan
    }
    
    creator = creators.get(plan_type, orchestrator.create_standard_plan)
    
    if name:
        kwargs['name'] = name
    
    return creator(**kwargs)


def diagnose_orchestrator_module() -> Dict[str, Any]:
    """诊断编排器模块"""
    # 检查各层可用性
    layer_availability = {}
    
    # 硬件层
    try:
        from backend.lib.hardware import get_device_manager
        layer_availability['hardware'] = True
    except ImportError:
        layer_availability['hardware'] = False
    
    # 分布式层
    try:
        from backend.lib.distributed import get_distributed_manager
        layer_availability['distributed'] = True
    except ImportError:
        layer_availability['distributed'] = False
    
    # 策略层
    try:
        from backend.modules.training.strategies import get_available_layers
        layer_availability['strategy'] = True
        layer_availability['strategy_layers'] = get_available_layers()
    except ImportError:
        layer_availability['strategy'] = False
    
    # 损失层
    try:
        from backend.lib.losses import LossFactory
        layer_availability['losses'] = True
    except ImportError:
        layer_availability['losses'] = False
    
    # 模态层
    try:
        from backend.lib.multimodal import ModalityEncoderFactory
        layer_availability['multimodal'] = True
    except ImportError:
        layer_availability['multimodal'] = False
    
    return {
        'module': 'unified_orchestrator',
        'layer_availability': layer_availability,
        'classes': {
            'UnifiedTrainingOrchestrator': UnifiedTrainingOrchestrator is not None,
            'LayerManager': LayerManager is not None,
            'OrchestratorPlan': OrchestratorPlan is not None,
            'PhaseConfig': PhaseConfig is not None,
            'LayerConfig': LayerConfig is not None,
        },
        'enums': {
            'OrchestratorPhase': list(OrchestratorPhase.__members__.keys()),
            'OrchestratorStatus': list(OrchestratorStatus.__members__.keys()),
        },
    }


# ==================== 导出 ====================

__all__ = [
    # 主类
    'UnifiedTrainingOrchestrator',
    'LayerManager',
    
    # 配置类
    'LayerConfig',
    'PhaseConfig',
    'OrchestratorPlan',
    'PhaseResult',
    
    # 枚举
    'OrchestratorPhase',
    'OrchestratorStatus',
    
    # 工厂函数
    'create_orchestrator',
    'create_quick_plan',
    'diagnose_orchestrator_module',
]


# -*- coding: utf-8 -*-
"""
优化器工具模块

提供训练优化相关的工具函数：
- 优化器创建和配置
- 学习率调度器
- 梯度处理（裁剪、累积）
- 权重初始化策略
- 收敛检测
- 混合精度训练支持（集成 backend/lib/hardware）

生产级功能：
- 硬件层集成（MixedPrecisionManager, AmpConfig）
- 内存监控和优化
- 策略层集成（StrategyMetrics）

架构调用层次：
├── optimizer_utils.py (本模块)
│   ├── 调用 backend/lib/hardware (硬件层)
│   └── 被 training_loop.py, three_stage_trainer.py 调用
└── 提供优化器、调度器、梯度处理等工具
"""

import math
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import contextlib

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


# ==================== 硬件层导入 ====================

from backend.lib.hardware import (
    MixedPrecisionManager as HardwareMixedPrecisionManager,
    AmpConfig,
    PrecisionMode,
    get_available_memory,
    clear_memory,
    recommend_precision,
)


class OptimizerType(Enum):
    """优化器类型枚举"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    RMSPROP = "rmsprop"
    LAMB = "lamb"  # Layer-wise Adaptive Moments


class SchedulerType(Enum):
    """学习率调度器类型枚举"""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    INVERSE_SQRT = "inverse_sqrt"
    ONE_CYCLE = "one_cycle"


class InitializationType(Enum):
    """权重初始化类型枚举"""
    XAVIER_UNIFORM = "xavier_uniform"
    XAVIER_NORMAL = "xavier_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    KAIMING_NORMAL = "kaiming_normal"
    ORTHOGONAL = "orthogonal"
    NORMAL = "normal"
    UNIFORM = "uniform"
    DEFAULT = "default"


@dataclass
class OptimizerConfig:
    """优化器配置"""
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    momentum: float = 0.9  # 用于SGD
    
    # 学习率调度
    scheduler_type: SchedulerType = SchedulerType.COSINE
    warmup_steps: int = 500
    warmup_ratio: float = 0.0  # 如果设置，会覆盖warmup_steps
    num_training_steps: int = 10000
    min_lr_ratio: float = 0.1  # 最低学习率为初始学习率的比例
    num_cycles: int = 1  # 用于cosine_with_restarts
    
    # 梯度处理
    gradient_clipping: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # 收敛检测
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    convergence_window: int = 100  # 用于检测损失是否收敛的滑动窗口


@dataclass
class TrainingState:
    """训练状态跟踪"""
    global_step: int = 0
    epoch: int = 0
    best_loss: float = float('inf')
    best_metrics: Dict[str, float] = field(default_factory=dict)
    patience_counter: int = 0
    loss_history: List[float] = field(default_factory=list)
    lr_history: List[float] = field(default_factory=list)
    
    # 收敛状态
    is_converged: bool = False
    convergence_step: Optional[int] = None
    
    # 累计统计
    total_loss: float = 0.0
    total_samples: int = 0


# =============================================================================
# 优化器创建
# =============================================================================

def create_optimizer(
    model: nn.Module,
    config: OptimizerConfig,
    no_decay_params: Optional[List[str]] = None
) -> Optimizer:
    """
    创建优化器
    
    Args:
        model: PyTorch模型
        config: 优化器配置
        no_decay_params: 不应用权重衰减的参数名模式列表
    
    Returns:
        配置好的优化器
    """
    if no_decay_params is None:
        no_decay_params = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
    
    # 分组参数：有权重衰减的和没有权重衰减的
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay_params) and p.requires_grad],
            'weight_decay': config.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay_params) and p.requires_grad],
            'weight_decay': 0.0,
        },
    ]
    
    # 根据类型创建优化器
    if config.optimizer_type == OptimizerType.ADAMW:
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon
        )
    elif config.optimizer_type == OptimizerType.ADAM:
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon
        )
    elif config.optimizer_type == OptimizerType.SGD:
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            momentum=config.momentum
        )
    elif config.optimizer_type == OptimizerType.ADAGRAD:
        optimizer = torch.optim.Adagrad(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            eps=config.epsilon
        )
    elif config.optimizer_type == OptimizerType.RMSPROP:
        optimizer = torch.optim.RMSprop(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            alpha=config.beta2,
            eps=config.epsilon,
            momentum=config.momentum
        )
    else:
        # 默认使用AdamW
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon
        )
    
    logger.info(f"Created optimizer: {config.optimizer_type.value}, lr={config.learning_rate}")
    return optimizer


# =============================================================================
# 学习率调度器
# =============================================================================

class WarmupScheduler(_LRScheduler):
    """带预热的学习率调度器基类"""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_steps:
            # 预热阶段：线性增长
            warmup_factor = self.last_epoch / max(1, self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # 预热后的调度（子类实现）
            return self._get_lr_after_warmup()
    
    def _get_lr_after_warmup(self) -> List[float]:
        """预热后的学习率计算（子类实现）"""
        return self.base_lrs


class CosineAnnealingWarmup(WarmupScheduler):
    """带预热的余弦退火调度器"""
    
    def _get_lr_after_warmup(self) -> List[float]:
        progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        
        # 余弦退火
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        decay = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay
        
        return [base_lr * decay for base_lr in self.base_lrs]


class LinearWarmup(WarmupScheduler):
    """带预热的线性衰减调度器"""
    
    def _get_lr_after_warmup(self) -> List[float]:
        progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        
        # 线性衰减
        decay = 1.0 - progress * (1.0 - self.min_lr_ratio)
        
        return [base_lr * decay for base_lr in self.base_lrs]


class PolynomialWarmup(WarmupScheduler):
    """带预热的多项式衰减调度器"""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.0, power=2.0, last_epoch=-1):
        self.power = power
        super().__init__(optimizer, warmup_steps, total_steps, min_lr_ratio, last_epoch)
    
    def _get_lr_after_warmup(self) -> List[float]:
        progress = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        
        # 多项式衰减
        decay = (1 - progress) ** self.power
        decay = self.min_lr_ratio + (1 - self.min_lr_ratio) * decay
        
        return [base_lr * decay for base_lr in self.base_lrs]


class InverseSqrtWarmup(WarmupScheduler):
    """带预热的逆平方根衰减调度器（类似Transformer论文）"""
    
    def _get_lr_after_warmup(self) -> List[float]:
        step = max(1, self.last_epoch - self.warmup_steps + 1)
        # 逆平方根衰减
        decay = (self.warmup_steps ** 0.5) / (step ** 0.5)
        decay = max(self.min_lr_ratio, decay)
        
        return [base_lr * decay for base_lr in self.base_lrs]


def create_scheduler(
    optimizer: Optimizer,
    config: OptimizerConfig,
    num_training_steps: Optional[int] = None
) -> Optional[_LRScheduler]:
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 配置
        num_training_steps: 总训练步数（覆盖配置）
    
    Returns:
        学习率调度器
    """
    total_steps = num_training_steps or config.num_training_steps
    
    # 计算warmup步数
    if config.warmup_ratio > 0:
        warmup_steps = int(total_steps * config.warmup_ratio)
    else:
        warmup_steps = config.warmup_steps
    
    if config.scheduler_type == SchedulerType.CONSTANT:
        return None  # 不使用调度器
    
    elif config.scheduler_type == SchedulerType.LINEAR:
        scheduler = LinearWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=config.min_lr_ratio
        )
    
    elif config.scheduler_type == SchedulerType.COSINE:
        scheduler = CosineAnnealingWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=config.min_lr_ratio
        )
    
    elif config.scheduler_type == SchedulerType.POLYNOMIAL:
        scheduler = PolynomialWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=config.min_lr_ratio
        )
    
    elif config.scheduler_type == SchedulerType.INVERSE_SQRT:
        scheduler = InverseSqrtWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=config.min_lr_ratio
        )
    
    elif config.scheduler_type == SchedulerType.ONE_CYCLE:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos',
            final_div_factor=1.0 / config.min_lr_ratio if config.min_lr_ratio > 0 else 1000
        )
    
    else:
        scheduler = CosineAnnealingWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=config.min_lr_ratio
        )
    
    logger.info(f"Created scheduler: {config.scheduler_type.value}, warmup={warmup_steps}, total={total_steps}")
    return scheduler


# =============================================================================
# 梯度处理
# =============================================================================

def clip_gradients(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    梯度裁剪
    
    Args:
        model: 模型
        max_norm: 最大梯度范数
        norm_type: 范数类型（2.0表示L2范数）
    
    Returns:
        裁剪前的梯度范数
    """
    if max_norm <= 0:
        return 0.0
    
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    
    total_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)
    return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm


def compute_gradient_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    计算梯度范数（用于监控）
    
    Args:
        model: 模型
        norm_type: 范数类型
    
    Returns:
        梯度范数
    """
    parameters = [p for p in model.parameters() if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    
    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    
    return total_norm ** (1.0 / norm_type)


class GradientAccumulator:
    """梯度累积器"""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = max(1, accumulation_steps)
        self.current_step = 0
    
    def should_step(self) -> bool:
        """检查是否应该执行优化器步骤"""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失以适应梯度累积"""
        if self.accumulation_steps > 1:
            return loss / self.accumulation_steps
        return loss
    
    def reset(self):
        """重置累积计数"""
        self.current_step = 0


# =============================================================================
# 权重初始化
# =============================================================================

def initialize_weights(
    model: nn.Module,
    init_type: InitializationType = InitializationType.DEFAULT,
    init_std: float = 0.02
) -> None:
    """
    初始化模型权重
    
    Args:
        model: PyTorch模型
        init_type: 初始化类型
        init_std: 标准差（用于normal初始化）
    """
    if init_type == InitializationType.DEFAULT:
        return  # 使用模型默认初始化
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'weight' in name:
            if len(param.shape) >= 2:
                # 多维权重（如Linear, Conv）
                if init_type == InitializationType.XAVIER_UNIFORM:
                    nn.init.xavier_uniform_(param)
                elif init_type == InitializationType.XAVIER_NORMAL:
                    nn.init.xavier_normal_(param)
                elif init_type == InitializationType.KAIMING_UNIFORM:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif init_type == InitializationType.KAIMING_NORMAL:
                    nn.init.kaiming_normal_(param, nonlinearity='relu')
                elif init_type == InitializationType.ORTHOGONAL:
                    nn.init.orthogonal_(param)
                elif init_type == InitializationType.NORMAL:
                    nn.init.normal_(param, mean=0.0, std=init_std)
                elif init_type == InitializationType.UNIFORM:
                    bound = init_std * math.sqrt(3)
                    nn.init.uniform_(param, -bound, bound)
            else:
                # 一维权重（如LayerNorm）
                nn.init.ones_(param)
        
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    logger.info(f"Initialized weights using {init_type.value}")


# =============================================================================
# 收敛检测
# =============================================================================

class ConvergenceDetector:
    """收敛检测器"""
    
    def __init__(
        self,
        patience: int = 5,
        threshold: float = 1e-4,
        window_size: int = 100,
        mode: str = 'min'  # 'min' 表示损失越小越好
    ):
        self.patience = patience
        self.threshold = threshold
        self.window_size = window_size
        self.mode = mode
        
        self.loss_history: List[float] = []
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.is_converged = False
    
    def update(self, loss: float) -> bool:
        """
        更新检测器
        
        Args:
            loss: 当前损失
        
        Returns:
            是否应该早停
        """
        self.loss_history.append(loss)
        
        # 检查是否改善
        is_improved = False
        if self.mode == 'min':
            if loss < self.best_value - self.threshold:
                is_improved = True
                self.best_value = loss
        else:
            if loss > self.best_value + self.threshold:
                is_improved = True
                self.best_value = loss
        
        if is_improved:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # 检查是否应该早停
        if self.patience_counter >= self.patience:
            self.is_converged = True
            return True
        
        # 检查损失是否收敛（变化很小）
        if len(self.loss_history) >= self.window_size:
            recent_losses = self.loss_history[-self.window_size:]
            loss_std = torch.tensor(recent_losses).std().item()
            if loss_std < self.threshold:
                self.is_converged = True
                logger.info(f"Loss converged: std={loss_std:.6f} < threshold={self.threshold}")
                return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取收敛统计"""
        if len(self.loss_history) == 0:
            return {}
        
        recent = self.loss_history[-min(self.window_size, len(self.loss_history)):]
        
        return {
            'current_loss': self.loss_history[-1],
            'best_loss': self.best_value,
            'patience_counter': self.patience_counter,
            'is_converged': self.is_converged,
            'recent_mean': sum(recent) / len(recent),
            'recent_std': torch.tensor(recent).std().item() if len(recent) > 1 else 0.0,
            'total_steps': len(self.loss_history)
        }


# =============================================================================
# 混合精度训练支持
# =============================================================================

class MixedPrecisionManager:
    """混合精度训练管理器
    
    集成 backend/lib/hardware 提供的混合精度功能，
    支持 fp16、bf16 和动态精度选择。
    """
    
    def __init__(
        self,
        enabled: bool = True,
        device: Optional[torch.device] = None,
        init_scale: float = 65536.0,
        precision_mode: str = 'fp16',
        use_hardware_layer: bool = True,
    ):
        """初始化混合精度管理器
        
        Args:
            enabled: 是否启用混合精度
            device: 设备
            init_scale: 初始缩放因子
            precision_mode: 精度模式 ('fp16', 'bf16', 'fp32')
            use_hardware_layer: 是否使用硬件层
        """
        self.device = device
        self.precision_mode = precision_mode
        self.init_scale = init_scale
        
        # 检测是否可以启用
        self.enabled = enabled and torch.cuda.is_available()
        
        # 尝试使用硬件层
        self._hardware_manager = None
        if self.enabled:
            self._init_hardware_manager()
        
        # 标准 GradScaler
        if self.enabled and self._hardware_manager is None:
            self.scaler = torch.cuda.amp.GradScaler(init_scale=init_scale)
        else:
            self.scaler = None
        
        # 统计信息
        self._scale_history: List[float] = []
        self._overflow_count: int = 0
    
    def _init_hardware_manager(self) -> None:
        """初始化硬件层混合精度管理器"""
        try:
            # 创建 AmpConfig（如果可用）
            amp_config = None
            if AmpConfig is not None:
                amp_config = AmpConfig(
                    init_scale=self.init_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=2000,
                )
            
            # 创建硬件层管理器
            if amp_config:
                self._hardware_manager = HardwareMixedPrecisionManager(
                    enabled=True,
                    amp_config=amp_config,
                )
            else:
                self._hardware_manager = HardwareMixedPrecisionManager(enabled=True)
            
            logger.info("Using hardware layer MixedPrecisionManager")
            
        except Exception as e:
            logger.warning(f"Failed to initialize hardware mixed precision manager: {e}")
            self._hardware_manager = None
    
    def autocast_context(self):
        """获取autocast上下文"""
        if self._hardware_manager is not None:
            try:
                return self._hardware_manager.autocast_context()
            except Exception:
                pass
        
        if self.enabled:
            # 根据精度模式选择 dtype
            if self.precision_mode == 'bf16' and torch.cuda.is_bf16_supported():
                return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        else:
                return torch.cuda.amp.autocast()
        
        return contextlib.nullcontext()
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失"""
        if self._hardware_manager is not None:
            try:
                return self._hardware_manager.scale_loss(loss)
            except Exception:
                pass
        
        if self.enabled and self.scaler:
            return self.scaler.scale(loss)
        return loss
    
    def unscale_and_step(self, optimizer: Optimizer) -> None:
        """反缩放梯度并执行优化步骤"""
        if self._hardware_manager is not None:
            try:
                self._hardware_manager.unscale_and_step(optimizer)
                return
            except Exception:
                pass
        
        if self.enabled and self.scaler:
            self.scaler.unscale_(optimizer)
    
    def step(self, optimizer: Optimizer) -> None:
        """执行优化步骤"""
        if self._hardware_manager is not None:
            try:
                self._hardware_manager.step(optimizer)
                return
            except Exception:
                pass
        
        if self.enabled and self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # 记录缩放因子历史
            current_scale = self.scaler.get_scale()
            self._scale_history.append(current_scale)
            
            # 检测溢出
            if len(self._scale_history) > 1:
                if current_scale < self._scale_history[-2]:
                    self._overflow_count += 1
        else:
            optimizer.step()
    
    def get_scale(self) -> float:
        """获取当前缩放因子"""
        if self._hardware_manager is not None:
            try:
                return self._hardware_manager.get_scale()
            except Exception:
                pass
        
        if self.enabled and self.scaler:
            return self.scaler.get_scale()
        return 1.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取混合精度训练统计信息"""
        stats = {
            'enabled': self.enabled,
            'precision_mode': self.precision_mode,
            'using_hardware_layer': self._hardware_manager is not None,
            'current_scale': self.get_scale(),
            'overflow_count': self._overflow_count,
        }
        
        if self._scale_history:
            stats['min_scale'] = min(self._scale_history)
            stats['max_scale'] = max(self._scale_history)
            stats['avg_scale'] = sum(self._scale_history) / len(self._scale_history)
        
        return stats
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MixedPrecisionManager':
        """从配置创建混合精度管理器
        
        Args:
            config: 配置字典
        
        Returns:
            MixedPrecisionManager 实例
        """
        return cls(
            enabled=config.get('use_fp16', True),
            precision_mode=config.get('precision_mode', 'fp16'),
            init_scale=config.get('init_scale', 65536.0),
            use_hardware_layer=config.get('use_hardware_layer', True),
        )
    
    @classmethod
    def get_recommended_precision(cls, device: str = 'cuda') -> str:
        """获取推荐的精度模式
        
        Args:
            device: 设备类型
        
        Returns:
            推荐的精度模式
        """
        try:
            return recommend_precision(device)
        except Exception:
            pass
        
        # 默认推荐
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return 'bf16'
            return 'fp16'
        return 'fp32'


# =============================================================================
# 工具函数
# =============================================================================

def get_optimizer_params(model: nn.Module) -> List[Dict]:
    """
    获取模型参数统计
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params,
        'trainable_ratio': trainable_params / max(1, total_params)
    }


def log_training_info(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    mixed_precision: Optional[MixedPrecisionManager] = None,
) -> None:
    """记录训练信息"""
    params_info = get_optimizer_params(model)
    
    logger.info("=" * 50)
    logger.info("Training Configuration:")
    logger.info(f"  Total parameters: {params_info['total_params']:,}")
    logger.info(f"  Trainable parameters: {params_info['trainable_params']:,}")
    logger.info(f"  Frozen parameters: {params_info['frozen_params']:,}")
    logger.info(f"  Trainable ratio: {params_info['trainable_ratio']:.2%}")
    logger.info(f"  Optimizer: {type(optimizer).__name__}")
    logger.info(f"  Learning rate: {optimizer.param_groups[0]['lr']}")
    if scheduler:
        logger.info(f"  Scheduler: {type(scheduler).__name__}")
    if mixed_precision:
        stats = mixed_precision.get_statistics()
        logger.info(f"  Mixed Precision: {stats.get('precision_mode', 'unknown')}")
        logger.info(f"  Using Hardware Layer: {stats.get('using_hardware_layer', False)}")
    logger.info("=" * 50)


def get_memory_info() -> Dict[str, Any]:
    """获取内存信息
    
    Returns:
        内存信息字典
    """
    memory_info = {
    }
    
    try:
        memory_info['available_memory_mb'] = get_available_memory()
    except Exception as e:
        logger.warning(f"Failed to get available memory: {e}")
    
    if torch.cuda.is_available():
        try:
            memory_info['cuda_allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_info['cuda_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
            memory_info['cuda_max_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024 ** 2)
        except Exception:
            pass
    
    return memory_info


def clear_training_memory() -> bool:
    """清理训练内存
    
    Returns:
        是否成功
    """
    try:
        # 使用硬件层清理
        clear_memory()
        logger.debug("Memory cleared via hardware layer")
        
        # PyTorch CUDA 清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            logger.debug("CUDA memory cache cleared")
        
        return True
    except Exception as e:
        logger.warning(f"Failed to clear memory: {e}")
        return False


def estimate_training_memory(
    model: nn.Module,
    batch_size: int = 8,
    seq_length: int = 512,
    gradient_accumulation_steps: int = 1,
    precision: str = 'fp16',
) -> Dict[str, float]:
    """估算训练内存需求
    
    Args:
        model: 模型
        batch_size: 批次大小
        seq_length: 序列长度
        gradient_accumulation_steps: 梯度累积步数
        precision: 精度模式
    
    Returns:
        内存估算字典（单位：MB）
    """
    # 获取模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算模型内存
    bytes_per_param = 4 if precision == 'fp32' else 2
    model_memory = total_params * bytes_per_param / (1024 ** 2)
    
    # 估算梯度内存
    gradient_memory = trainable_params * bytes_per_param / (1024 ** 2)
    
    # 估算优化器状态内存（AdamW 需要 2 个动量状态）
    optimizer_memory = trainable_params * 4 * 2 / (1024 ** 2)  # 优化器状态通常是 fp32
    
    # 估算激活内存（近似估算）
    effective_batch_size = batch_size / gradient_accumulation_steps
    activation_memory = effective_batch_size * seq_length * 1024 * bytes_per_param / (1024 ** 2)
    
    total_memory = model_memory + gradient_memory + optimizer_memory + activation_memory
    
    return {
        'model_memory_mb': model_memory,
        'gradient_memory_mb': gradient_memory,
        'optimizer_memory_mb': optimizer_memory,
        'activation_memory_mb': activation_memory,
        'total_estimated_mb': total_memory,
        'total_params': total_params,
        'trainable_params': trainable_params,
    }


def get_optimizer_utils_info() -> Dict[str, Any]:
    """获取优化器工具信息
    
    Returns:
        信息字典
    """
    return {
        'optimizer_types': [o.value for o in OptimizerType],
        'scheduler_types': [s.value for s in SchedulerType],
        'initialization_types': [i.value for i in InitializationType],
        'mixed_precision_support': torch.cuda.is_available(),
        'bf16_support': torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        'recommended_precision': MixedPrecisionManager.get_recommended_precision() if torch.cuda.is_available() else 'fp32',
    }


# ==================== 导出 ====================

__all__ = [
    # 配置类
    'OptimizerConfig',
    'TrainingState',
    
    # 枚举
    'OptimizerType',
    'SchedulerType',
    'InitializationType',
    
    # 创建函数
    'create_optimizer',
    'create_scheduler',
    
    # 调度器类
    'WarmupScheduler',
    'CosineAnnealingWarmup',
    'LinearWarmup',
    'PolynomialWarmup',
    'InverseSqrtWarmup',
    
    # 梯度工具
    'clip_gradients',
    'compute_gradient_norm',
    'initialize_weights',
    'GradientAccumulator',
    
    # 收敛检测
    'ConvergenceDetector',
    
    # 混合精度
    'MixedPrecisionManager',
    
    # 工具函数
    'get_optimizer_params',
    'log_training_info',
    'get_memory_info',
    'clear_training_memory',
    'estimate_training_memory',
    'get_optimizer_utils_info',
    
    # 层可用性标志
]


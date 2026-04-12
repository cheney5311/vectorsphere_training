# -*- coding: utf-8 -*-
"""
正则化损失函数

包含各种正则化相关的损失函数。
"""

import logging
import time
import math
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_loss import BaseLoss, LossConfig, LossResult, LossMonitor, register_loss

logger = logging.getLogger(__name__)


# ==================== 监控和统计组件 ====================

@dataclass
class RegularizationStats:
    """正则化统计"""
    total_steps: int = 0
    avg_reg_loss: float = 0.0
    avg_l1_norm: float = 0.0
    avg_l2_norm: float = 0.0
    avg_sparsity: float = 0.0
    avg_entropy: float = 0.0
    total_params: int = 0
    zero_params: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'avg_reg_loss': self.avg_reg_loss,
            'avg_l1_norm': self.avg_l1_norm,
            'avg_l2_norm': self.avg_l2_norm,
            'avg_sparsity': self.avg_sparsity,
            'avg_entropy': self.avg_entropy,
            'total_params': self.total_params,
            'zero_params': self.zero_params,
        }


class RegularizationMonitor:
    """正则化监控器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._history: List[Dict[str, float]] = []
        self._stats = RegularizationStats()
        
        # 累计统计
        self._totals: Dict[str, float] = defaultdict(float)
    
    def record(
        self,
        reg_loss: float = 0.0,
        l1_norm: float = 0.0,
        l2_norm: float = 0.0,
        sparsity: float = 0.0,
        entropy: float = 0.0,
        total_params: int = 0,
        zero_params: int = 0,
        **kwargs
    ) -> None:
        """记录统计"""
        record = {
            'reg_loss': reg_loss,
            'l1_norm': l1_norm,
            'l2_norm': l2_norm,
            'sparsity': sparsity,
            'entropy': entropy,
            'total_params': total_params,
            'zero_params': zero_params,
            'timestamp': time.time(),
            **kwargs
        }
        
        self._history.append(record)
        if len(self._history) > self.max_history:
            self._history.pop(0)
        
        # 更新累计统计
        self._stats.total_steps += 1
        n = self._stats.total_steps
        
        self._totals['reg_loss'] += reg_loss
        self._totals['l1_norm'] += l1_norm
        self._totals['l2_norm'] += l2_norm
        self._totals['sparsity'] += sparsity
        self._totals['entropy'] += entropy
        
        # 更新平均
        self._stats.avg_reg_loss = self._totals['reg_loss'] / n
        self._stats.avg_l1_norm = self._totals['l1_norm'] / n
        self._stats.avg_l2_norm = self._totals['l2_norm'] / n
        self._stats.avg_sparsity = self._totals['sparsity'] / n
        self._stats.avg_entropy = self._totals['entropy'] / n
        self._stats.total_params = total_params
        self._stats.zero_params = zero_params
    
    def get_stats(self) -> RegularizationStats:
        """获取统计"""
        return self._stats
    
    def get_recent(self, n: int = 10) -> List[Dict[str, float]]:
        """获取最近的记录"""
        return self._history[-n:]
    
    def get_sparsity_trend(self) -> float:
        """获取稀疏度趋势"""
        if len(self._history) < 20:
            return 0.0
        
        recent = [r['sparsity'] for r in self._history[-10:]]
        earlier = [r['sparsity'] for r in self._history[-20:-10]]
        
        return (sum(recent) / 10) - (sum(earlier) / 10)
    
    def is_regularization_effective(self) -> bool:
        """检查正则化是否有效"""
        if len(self._history) < 10:
            return True
        
        # 检查稀疏度是否在增加（对于L1）或范数是否在减小（对于L2）
        return self.get_sparsity_trend() >= 0 or self._stats.avg_l2_norm < 1.0
    
    def reset(self) -> None:
        """重置"""
        self._history.clear()
        self._stats = RegularizationStats()
        self._totals.clear()


class LambdaScheduler:
    """正则化强度调度器"""
    
    def __init__(
        self,
        initial_lambda: float = 1e-4,
        min_lambda: float = 1e-6,
        max_lambda: float = 1e-2,
        schedule: str = 'constant',  # constant, linear, cosine, warmup, adaptive
        warmup_steps: int = 0,
        total_steps: int = 10000
    ):
        self.initial_lambda = initial_lambda
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        self._step = 0
        self._current_lambda = initial_lambda
        
        # 自适应调度的损失历史
        self._loss_history: List[float] = []
    
    def step(self, loss: Optional[float] = None) -> float:
        """获取当前lambda并更新步数"""
        self._step += 1
        
        if loss is not None:
            self._loss_history.append(loss)
            if len(self._loss_history) > 100:
                self._loss_history.pop(0)
        
        self._current_lambda = self.get_lambda()
        return self._current_lambda
    
    def get_lambda(self) -> float:
        """获取当前lambda"""
        # 预热阶段
        if self._step < self.warmup_steps:
            progress = self._step / max(self.warmup_steps, 1)
            return self.min_lambda + (self.initial_lambda - self.min_lambda) * progress
        
        effective_step = self._step - self.warmup_steps
        effective_total = self.total_steps - self.warmup_steps
        
        if self.schedule == 'constant':
            return self.initial_lambda
        
        elif self.schedule == 'linear':
            # 线性增长lambda（随训练进行增加正则化）
            progress = min(effective_step / effective_total, 1.0)
            return self.initial_lambda + (self.max_lambda - self.initial_lambda) * progress
        
        elif self.schedule == 'cosine':
            # 余弦调度
            progress = min(effective_step / effective_total, 1.0)
            return self.min_lambda + (self.initial_lambda - self.min_lambda) * (1 + math.cos(progress * math.pi)) / 2
        
        elif self.schedule == 'warmup':
            # 预热后保持恒定
            return self.initial_lambda
        
        elif self.schedule == 'adaptive':
            # 根据训练损失自适应调整
            if len(self._loss_history) < 10:
                return self.initial_lambda
            
            recent_loss = sum(self._loss_history[-10:]) / 10
            
            # 如果损失在下降，可以增加正则化
            if len(self._loss_history) >= 20:
                earlier_loss = sum(self._loss_history[-20:-10]) / 10
                if recent_loss < earlier_loss * 0.95:
                    # 损失在下降，增加正则化
                    target = min(self._current_lambda * 1.1, self.max_lambda)
                elif recent_loss > earlier_loss * 1.05:
                    # 损失在上升，减少正则化
                    target = max(self._current_lambda * 0.9, self.min_lambda)
                else:
                    target = self._current_lambda
            else:
                target = self.initial_lambda
            
            # 平滑过渡
            return 0.9 * self._current_lambda + 0.1 * target
        
        return self.initial_lambda
    
    def set_total_steps(self, total_steps: int) -> None:
        """设置总步数"""
        self.total_steps = total_steps
    
    def reset(self) -> None:
        """重置"""
        self._step = 0
        self._current_lambda = self.initial_lambda
        self._loss_history.clear()


class SparsityAnalyzer:
    """稀疏度分析器"""
    
    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold
    
    def analyze_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        分析模型稀疏度
        
        Args:
            model: 要分析的模型
            
        Returns:
            稀疏度分析结果
        """
        total_params = 0
        zero_params = 0
        near_zero_params = 0
        
        layer_sparsity = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                numel = param.numel()
                zeros = (param.abs() < self.threshold).sum().item()
                near_zeros = (param.abs() < self.threshold * 10).sum().item()
                
                total_params += numel
                zero_params += zeros
                near_zero_params += near_zeros
                
                layer_sparsity[name] = {
                    'total': numel,
                    'zeros': zeros,
                    'sparsity': zeros / numel if numel > 0 else 0.0,
                    'near_zero_ratio': near_zeros / numel if numel > 0 else 0.0
                }
        
        return {
            'total_params': total_params,
            'zero_params': zero_params,
            'near_zero_params': near_zero_params,
            'overall_sparsity': zero_params / total_params if total_params > 0 else 0.0,
            'near_zero_ratio': near_zero_params / total_params if total_params > 0 else 0.0,
            'layer_sparsity': layer_sparsity
        }
    
    def analyze_tensor(self, tensor: Tensor) -> Dict[str, float]:
        """
        分析张量稀疏度
        
        Args:
            tensor: 要分析的张量
            
        Returns:
            稀疏度分析结果
        """
        numel = tensor.numel()
        zeros = (tensor.abs() < self.threshold).sum().item()
        near_zeros = (tensor.abs() < self.threshold * 10).sum().item()
        
        return {
            'total': numel,
            'zeros': zeros,
            'sparsity': zeros / numel if numel > 0 else 0.0,
            'near_zero_ratio': near_zeros / numel if numel > 0 else 0.0,
            'l1_norm': tensor.abs().sum().item(),
            'l2_norm': tensor.pow(2).sum().sqrt().item(),
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'max': tensor.abs().max().item(),
            'min': tensor.abs().min().item()
        }
    
    def get_pruning_mask(self, tensor: Tensor, target_sparsity: float) -> Tensor:
        """
        获取剪枝掩码
        
        Args:
            tensor: 要剪枝的张量
            target_sparsity: 目标稀疏度
            
        Returns:
            剪枝掩码
        """
        k = int(tensor.numel() * target_sparsity)
        if k == 0:
            return torch.ones_like(tensor, dtype=torch.bool)
        
        # 找到最小的k个元素
        threshold = tensor.abs().flatten().kthvalue(k).values
        mask = tensor.abs() > threshold
        
        return mask




class RegularizationLoss(BaseLoss):
    """正则化损失基类"""
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        lambda_schedule: str = 'constant',
        warmup_steps: int = 0,
        **kwargs
    ):
        super().__init__(config)
        
        # 新增：监控器
        self._reg_monitor = RegularizationMonitor()
        self._rl_step = 0
        
        # 新增：Lambda调度器（由子类初始化）
        self._lambda_scheduler: Optional[LambdaScheduler] = None
        self._lambda_schedule = lambda_schedule
        self._warmup_steps = warmup_steps
        
        # 新增：稀疏度分析器
        self._sparsity_analyzer = SparsityAnalyzer()
    
    def _init_lambda_scheduler(self, initial_lambda: float) -> None:
        """初始化Lambda调度器"""
        if self._lambda_scheduler is None:
            self._lambda_scheduler = LambdaScheduler(
                initial_lambda=initial_lambda,
                schedule=self._lambda_schedule,
                warmup_steps=self._warmup_steps
            )
    
    def get_effective_lambda(self) -> float:
        """获取有效的正则化强度"""
        if self._lambda_scheduler is not None:
            return self._lambda_scheduler.get_lambda()
        return 0.0
    
    def set_lambda_schedule(
        self,
        schedule: str,
        total_steps: int = 10000,
        warmup_steps: int = 0
    ) -> None:
        """设置Lambda调度"""
        self._lambda_schedule = schedule
        self._warmup_steps = warmup_steps
        if self._lambda_scheduler is not None:
            self._lambda_scheduler.schedule = schedule
            self._lambda_scheduler.set_total_steps(total_steps)
            self._lambda_scheduler.warmup_steps = warmup_steps
    
    def _compute_model_norms(self, model: nn.Module) -> Tuple[float, float, int, int]:
        """
        计算模型范数
        
        Returns:
            (l1_norm, l2_norm, total_params, zero_params)
        """
        l1_norm = 0.0
        l2_norm = 0.0
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            if param.requires_grad:
                l1_norm += param.abs().sum().item()
                l2_norm += param.pow(2).sum().item()
                total_params += param.numel()
                zero_params += (param.abs() < 1e-6).sum().item()
        
        l2_norm = math.sqrt(l2_norm)
        
        return l1_norm, l2_norm, total_params, zero_params
    
    def _compute_sparsity(self, model: nn.Module) -> float:
        """计算稀疏度"""
        analysis = self._sparsity_analyzer.analyze_model(model)
        return analysis['overall_sparsity']
    
    def _record_reg_stats(
        self,
        reg_loss: float,
        l1_norm: float = 0.0,
        l2_norm: float = 0.0,
        sparsity: float = 0.0,
        entropy: float = 0.0,
        total_params: int = 0,
        zero_params: int = 0,
        **kwargs
    ) -> None:
        """记录正则化统计"""
        self._reg_monitor.record(
            reg_loss=reg_loss,
            l1_norm=l1_norm,
            l2_norm=l2_norm,
            sparsity=sparsity,
            entropy=entropy,
            total_params=total_params,
            zero_params=zero_params,
            **kwargs
        )
        self._rl_step += 1
    
    def get_reg_stats(self) -> RegularizationStats:
        """获取正则化统计"""
        return self._reg_monitor.get_stats()
    
    def get_sparsity_trend(self) -> float:
        """获取稀疏度趋势"""
        return self._reg_monitor.get_sparsity_trend()
    
    def is_regularization_effective(self) -> bool:
        """检查正则化是否有效"""
        return self._reg_monitor.is_regularization_effective()
    
    def analyze_model_sparsity(self, model: nn.Module) -> Dict[str, Any]:
        """分析模型稀疏度"""
        return self._sparsity_analyzer.analyze_model(model)
    
    def print_summary(self) -> None:
        """打印摘要"""
        stats = self.get_reg_stats()
        
        print("\n" + "="*80)
        print(f"Regularization Loss Summary: {self.__class__.__name__}")
        print("="*80)
        
        if self._lambda_scheduler is not None:
            print(f"\nLambda: {self.get_effective_lambda():.6f}")
            print(f"Lambda schedule: {self._lambda_schedule}")
        
        print(f"\nStatistics (over {stats.total_steps} steps):")
        print(f"  Avg reg loss: {stats.avg_reg_loss:.6f}")
        if stats.avg_l1_norm > 0:
            print(f"  Avg L1 norm: {stats.avg_l1_norm:.4f}")
        if stats.avg_l2_norm > 0:
            print(f"  Avg L2 norm: {stats.avg_l2_norm:.4f}")
        if stats.avg_sparsity > 0:
            print(f"  Avg sparsity: {stats.avg_sparsity:.4f}")
        if stats.avg_entropy > 0:
            print(f"  Avg entropy: {stats.avg_entropy:.4f}")
        
        if stats.total_params > 0:
            print(f"\nModel parameters:")
            print(f"  Total: {stats.total_params:,}")
            print(f"  Zero: {stats.zero_params:,}")
            print(f"  Sparsity: {stats.zero_params / stats.total_params:.4f}")
        
        print(f"\nRegularization effective: {self.is_regularization_effective()}")
        print(f"Sparsity trend: {self.get_sparsity_trend():+.6f}")
        
        print("="*80)
    
    def reset_reg_stats(self) -> None:
        """重置统计"""
        self._reg_monitor.reset()
        if self._lambda_scheduler is not None:
            self._lambda_scheduler.reset()
        self._rl_step = 0


# ==================== L1正则化 ====================

@register_loss("l1_reg")
class L1Regularization(RegularizationLoss):
    """
    L1正则化
    
    促进稀疏性。
    R = λ * Σ|w|
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        lambda_: float = 1e-4,
        lambda_schedule: str = 'constant',
        **kwargs
    ):
        super().__init__(config, lambda_schedule=lambda_schedule)
        self.lambda_ = lambda_
        self._init_lambda_scheduler(lambda_)
    
    def forward(
        self, 
        predictions: nn.Module,  # model
        targets: Any = None,     # unused
        **kwargs
    ) -> Tensor:
        """
        计算L1正则化
        
        Args:
            predictions: 模型或参数列表
        """
        l1_norm = torch.tensor(0.0)
        
        if isinstance(predictions, nn.Module):
            params = predictions.parameters()
        elif isinstance(predictions, (list, tuple)):
            params = predictions
        else:
            return l1_norm
        
        for param in params:
            if param.requires_grad:
                l1_norm = l1_norm + param.abs().sum()
        
        effective_lambda = self.get_effective_lambda()
        return effective_lambda * l1_norm
    
    def compute(
        self, 
        predictions: nn.Module,
        targets: Any = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        l1_norm = torch.tensor(0.0)
        total_params = 0
        zero_params = 0
        
        if isinstance(predictions, nn.Module):
            params = list(predictions.parameters())
            
            for param in params:
                if param.requires_grad:
                    l1_norm = l1_norm + param.abs().sum()
                    total_params += param.numel()
                    zero_params += (param.abs() < 1e-6).sum().item()
        elif isinstance(predictions, (list, tuple)):
            for param in predictions:
                if param.requires_grad:
                    l1_norm = l1_norm + param.abs().sum()
                    total_params += param.numel()
                    zero_params += (param.abs() < 1e-6).sum().item()
        
        effective_lambda = self.get_effective_lambda()
        loss = effective_lambda * l1_norm
        
        # 计算稀疏度
        sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        # 记录统计
        self._record_reg_stats(
            reg_loss=loss.item(),
            l1_norm=l1_norm.item(),
            sparsity=sparsity,
            total_params=total_params,
            zero_params=zero_params
        )
        
        # 更新Lambda调度
        if self._lambda_scheduler is not None:
            self._lambda_scheduler.step(loss.item())
        
        return LossResult(
            loss=loss * self.config.weight,
            components={'l1_norm': l1_norm},
            metrics={
                'l1_loss': loss.item(),
                'l1_norm': l1_norm.item(),
                'lambda': effective_lambda,
                'sparsity': sparsity,
                'total_params': total_params,
                'zero_params': zero_params
            },
            step=self._rl_step
        )
    
    def get_sparsity_analysis(self, model: nn.Module) -> Dict[str, Any]:
        """获取模型稀疏度分析"""
        return self._sparsity_analyzer.analyze_model(model)


# ==================== L2正则化 ====================

@register_loss("l2_reg")
class L2Regularization(RegularizationLoss):
    """
    L2正则化（权重衰减）
    
    防止过拟合。
    R = λ * Σw²
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        lambda_: float = 1e-4,
        lambda_schedule: str = 'constant',
        **kwargs
    ):
        super().__init__(config, lambda_schedule=lambda_schedule)
        self.lambda_ = lambda_
        self._init_lambda_scheduler(lambda_)
    
    def forward(
        self, 
        predictions: nn.Module,
        targets: Any = None,
        **kwargs
    ) -> Tensor:
        """计算L2正则化"""
        l2_norm = torch.tensor(0.0)
        
        if isinstance(predictions, nn.Module):
            params = predictions.parameters()
        elif isinstance(predictions, (list, tuple)):
            params = predictions
        else:
            return l2_norm
        
        for param in params:
            if param.requires_grad:
                l2_norm = l2_norm + param.pow(2).sum()
        
        effective_lambda = self.get_effective_lambda()
        return effective_lambda * l2_norm
    
    def compute(
        self, 
        predictions: nn.Module,
        targets: Any = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        l2_norm_sq = torch.tensor(0.0)
        total_params = 0
        zero_params = 0
        max_weight = 0.0
        mean_weight = 0.0
        
        if isinstance(predictions, nn.Module):
            params = list(predictions.parameters())
            
            all_weights = []
            for param in params:
                if param.requires_grad:
                    l2_norm_sq = l2_norm_sq + param.pow(2).sum()
                    total_params += param.numel()
                    zero_params += (param.abs() < 1e-6).sum().item()
                    all_weights.append(param.abs().max().item())
            
            if all_weights:
                max_weight = max(all_weights)
        elif isinstance(predictions, (list, tuple)):
            for param in predictions:
                if param.requires_grad:
                    l2_norm_sq = l2_norm_sq + param.pow(2).sum()
                    total_params += param.numel()
                    zero_params += (param.abs() < 1e-6).sum().item()
        
        l2_norm = l2_norm_sq.sqrt()
        effective_lambda = self.get_effective_lambda()
        loss = effective_lambda * l2_norm_sq
        
        # 记录统计
        self._record_reg_stats(
            reg_loss=loss.item(),
            l2_norm=l2_norm.item(),
            total_params=total_params,
            zero_params=zero_params
        )
        
        # 更新Lambda调度
        if self._lambda_scheduler is not None:
            self._lambda_scheduler.step(loss.item())
        
        return LossResult(
            loss=loss * self.config.weight,
            components={'l2_norm_sq': l2_norm_sq, 'l2_norm': l2_norm},
            metrics={
                'l2_loss': loss.item(),
                'l2_norm': l2_norm.item(),
                'l2_norm_sq': l2_norm_sq.item(),
                'lambda': effective_lambda,
                'max_weight': max_weight,
                'total_params': total_params,
                'zero_params': zero_params
            },
            step=self._rl_step
        )
    
    def get_weight_statistics(self, model: nn.Module) -> Dict[str, float]:
        """获取权重统计"""
        all_weights = []
        
        for param in model.parameters():
            if param.requires_grad:
                all_weights.append(param.data.flatten())
        
        if not all_weights:
            return {}
        
        weights = torch.cat(all_weights)
        
        return {
            'mean': weights.mean().item(),
            'std': weights.std().item(),
            'min': weights.min().item(),
            'max': weights.max().item(),
            'l2_norm': weights.norm().item(),
            'sparsity': (weights.abs() < 1e-6).float().mean().item()
        }


# ==================== Elastic Net正则化 ====================

@register_loss("elastic_net")
class ElasticNetRegularization(RegularizationLoss):
    """
    Elastic Net正则化
    
    结合L1和L2正则化。
    R = λ * (α * Σ|w| + (1-α) * Σw²)
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        lambda_: float = 1e-4,
        alpha: float = 0.5,  # L1比例
        lambda_schedule: str = 'constant',
        **kwargs
    ):
        super().__init__(config, lambda_schedule=lambda_schedule)
        self.lambda_ = lambda_
        self.alpha = alpha
        self._init_lambda_scheduler(lambda_)
    
    def forward(
        self, 
        predictions: nn.Module,
        targets: Any = None,
        **kwargs
    ) -> Tensor:
        """计算Elastic Net正则化"""
        l1_norm = torch.tensor(0.0)
        l2_norm = torch.tensor(0.0)
        
        if isinstance(predictions, nn.Module):
            params = predictions.parameters()
        elif isinstance(predictions, (list, tuple)):
            params = predictions
        else:
            return l1_norm
        
        for param in params:
            if param.requires_grad:
                l1_norm = l1_norm + param.abs().sum()
                l2_norm = l2_norm + param.pow(2).sum()
        
        effective_lambda = self.get_effective_lambda()
        return effective_lambda * (self.alpha * l1_norm + (1 - self.alpha) * l2_norm)
    
    def compute(
        self, 
        predictions: nn.Module,
        targets: Any = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        l1_norm = torch.tensor(0.0)
        l2_norm_sq = torch.tensor(0.0)
        total_params = 0
        zero_params = 0
        
        if isinstance(predictions, nn.Module):
            params = list(predictions.parameters())
        elif isinstance(predictions, (list, tuple)):
            params = predictions
        else:
            params = []
        
        for param in params:
            if hasattr(param, 'requires_grad') and param.requires_grad:
                l1_norm = l1_norm + param.abs().sum()
                l2_norm_sq = l2_norm_sq + param.pow(2).sum()
                total_params += param.numel()
                zero_params += (param.abs() < 1e-6).sum().item()
        
        l2_norm = l2_norm_sq.sqrt()
        effective_lambda = self.get_effective_lambda()
        
        l1_loss = self.alpha * l1_norm
        l2_loss = (1 - self.alpha) * l2_norm_sq
        loss = effective_lambda * (l1_loss + l2_loss)
        
        # 计算稀疏度
        sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        # 记录统计
        self._record_reg_stats(
            reg_loss=loss.item(),
            l1_norm=l1_norm.item(),
            l2_norm=l2_norm.item(),
            sparsity=sparsity,
            total_params=total_params,
            zero_params=zero_params
        )
        
        # 更新Lambda调度
        if self._lambda_scheduler is not None:
            self._lambda_scheduler.step(loss.item())
        
        return LossResult(
            loss=loss * self.config.weight,
            components={
                'l1_norm': l1_norm,
                'l2_norm': l2_norm,
                'l1_loss': l1_loss,
                'l2_loss': l2_loss
            },
            metrics={
                'elastic_net_loss': loss.item(),
                'l1_norm': l1_norm.item(),
                'l2_norm': l2_norm.item(),
                'l1_loss': (effective_lambda * l1_loss).item(),
                'l2_loss': (effective_lambda * l2_loss).item(),
                'lambda': effective_lambda,
                'alpha': self.alpha,
                'sparsity': sparsity,
                'total_params': total_params,
                'zero_params': zero_params
            },
            step=self._rl_step
        )
    
    def set_alpha(self, alpha: float) -> None:
        """设置L1比例"""
        self.alpha = max(0.0, min(1.0, alpha))


# ==================== 一致性正则化 ====================

@register_loss("consistency_reg")
class ConsistencyRegularization(RegularizationLoss):
    """
    一致性正则化
    
    用于半监督学习，确保对同一输入的不同增强产生一致的输出。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        loss_type: str = "mse",  # mse, kl, js
        temperature: float = 1.0,
        **kwargs
    ):
        super().__init__(config)
        self.loss_type = loss_type
        self.temperature = temperature
        
        # 新增：一致性历史
        self._consistency_history: List[float] = []
        self._agreement_history: List[float] = []
    
    def forward(
        self, 
        predictions: Tensor,  # 原始预测 [N, C]
        targets: Tensor,      # 增强后预测 [N, C]
        **kwargs
    ) -> Tensor:
        """
        计算一致性正则化
        
        Args:
            predictions: 原始输入的预测
            targets: 增强输入的预测
        """
        if self.loss_type == "mse":
            return F.mse_loss(predictions, targets.detach())
        
        elif self.loss_type == "kl":
            p_log = F.log_softmax(predictions / self.temperature, dim=-1)
            q_soft = F.softmax(targets.detach() / self.temperature, dim=-1)
            return F.kl_div(p_log, q_soft, reduction='batchmean')
        
        elif self.loss_type == "js":
            # Jensen-Shannon散度
            p = F.softmax(predictions / self.temperature, dim=-1)
            q = F.softmax(targets.detach() / self.temperature, dim=-1)
            m = (p + q) / 2
            
            kl_pm = F.kl_div(torch.log(p + 1e-8), m, reduction='batchmean')
            kl_qm = F.kl_div(torch.log(q + 1e-8), m, reduction='batchmean')
            
            return (kl_pm + kl_qm) / 2
        
        else:
            return F.mse_loss(predictions, targets.detach())
    
    def compute(
        self, 
        predictions: Tensor,
        targets: Tensor,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        # 计算损失
        loss = self.forward(predictions, targets)
        
        # 计算一致性指标
        p_probs = F.softmax(predictions, dim=-1)
        q_probs = F.softmax(targets, dim=-1)
        
        # 计算预测一致率
        p_preds = predictions.argmax(dim=-1)
        q_preds = targets.argmax(dim=-1)
        agreement = (p_preds == q_preds).float().mean().item()
        
        # 计算概率分布相似度（余弦相似度）
        cosine_sim = torch.cosine_similarity(p_probs, q_probs, dim=-1).mean().item()
        
        # 计算KL散度
        kl_div = F.kl_div(
            F.log_softmax(predictions, dim=-1),
            F.softmax(targets, dim=-1),
            reduction='batchmean'
        ).item()
        
        # 记录历史
        self._consistency_history.append(loss.item())
        self._agreement_history.append(agreement)
        
        if len(self._consistency_history) > 100:
            self._consistency_history.pop(0)
            self._agreement_history.pop(0)
        
        # 记录统计
        self._record_reg_stats(reg_loss=loss.item())
        
        return LossResult(
            loss=loss * self.config.weight,
            components={'consistency_loss': loss},
            metrics={
                'consistency_loss': loss.item(),
                'loss_type': self.loss_type,
                'agreement': agreement,
                'cosine_similarity': cosine_sim,
                'kl_divergence': kl_div,
                'temperature': self.temperature
            },
            step=self._rl_step
        )
    
    def get_consistency_stats(self) -> Dict[str, float]:
        """获取一致性统计"""
        if not self._consistency_history:
            return {}
        
        return {
            'avg_loss': sum(self._consistency_history) / len(self._consistency_history),
            'avg_agreement': sum(self._agreement_history) / len(self._agreement_history),
            'recent_loss': self._consistency_history[-1] if self._consistency_history else 0.0,
            'recent_agreement': self._agreement_history[-1] if self._agreement_history else 0.0
        }
    
    def reset_consistency_stats(self) -> None:
        """重置一致性统计"""
        self._consistency_history.clear()
        self._agreement_history.clear()


# ==================== 熵正则化 ====================

@register_loss("entropy_reg")
class EntropyRegularization(RegularizationLoss):
    """
    熵正则化
    
    用于鼓励或惩罚预测的不确定性。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        minimize: bool = True,  # True: 最小化熵（更确定），False: 最大化熵（更不确定）
        target_entropy: Optional[float] = None,  # 目标熵值（可选）
        **kwargs
    ):
        super().__init__(config)
        self.minimize = minimize
        self.target_entropy = target_entropy
        
        # 新增：熵历史
        self._entropy_history: List[float] = []
        self._confidence_history: List[float] = []
    
    def forward(
        self, 
        predictions: Tensor,  # logits [N, C]
        targets: Any = None,  # unused
        **kwargs
    ) -> Tensor:
        """
        计算熵正则化
        
        Args:
            predictions: logits
        """
        probs = F.softmax(predictions, dim=-1)
        log_probs = F.log_softmax(predictions, dim=-1)
        
        # 熵: H(p) = -Σ p * log(p)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        if self.target_entropy is not None:
            # 目标熵正则化
            return (entropy - self.target_entropy).abs()
        elif self.minimize:
            return entropy
        else:
            return -entropy
    
    def compute(
        self, 
        predictions: Tensor,
        targets: Any = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        probs = F.softmax(predictions, dim=-1)
        log_probs = F.log_softmax(predictions, dim=-1)
        
        # 计算每个样本的熵
        sample_entropy = -(probs * log_probs).sum(dim=-1)  # [N]
        mean_entropy = sample_entropy.mean()
        
        # 计算损失
        if self.target_entropy is not None:
            loss = (mean_entropy - self.target_entropy).abs()
        elif self.minimize:
            loss = mean_entropy
        else:
            loss = -mean_entropy
        
        # 计算置信度（最大概率）
        max_probs = probs.max(dim=-1).values
        mean_confidence = max_probs.mean().item()
        
        # 计算最大熵（均匀分布）
        num_classes = predictions.size(-1)
        max_entropy = math.log(num_classes)
        normalized_entropy = mean_entropy.item() / max_entropy
        
        # 记录历史
        self._entropy_history.append(mean_entropy.item())
        self._confidence_history.append(mean_confidence)
        
        if len(self._entropy_history) > 100:
            self._entropy_history.pop(0)
            self._confidence_history.pop(0)
        
        # 记录统计
        self._record_reg_stats(
            reg_loss=loss.item(),
            entropy=mean_entropy.item()
        )
        
        return LossResult(
            loss=loss * self.config.weight,
            components={'entropy': mean_entropy},
            metrics={
                'entropy_loss': loss.item(),
                'mean_entropy': mean_entropy.item(),
                'std_entropy': sample_entropy.std().item(),
                'min_entropy': sample_entropy.min().item(),
                'max_entropy': sample_entropy.max().item(),
                'normalized_entropy': normalized_entropy,
                'mean_confidence': mean_confidence,
                'minimize': self.minimize,
                'target_entropy': self.target_entropy
            },
            step=self._rl_step
        )
    
    def get_entropy_stats(self) -> Dict[str, float]:
        """获取熵统计"""
        if not self._entropy_history:
            return {}
        
        return {
            'avg_entropy': sum(self._entropy_history) / len(self._entropy_history),
            'avg_confidence': sum(self._confidence_history) / len(self._confidence_history),
            'recent_entropy': self._entropy_history[-1],
            'recent_confidence': self._confidence_history[-1]
        }
    
    def reset_entropy_stats(self) -> None:
        """重置熵统计"""
        self._entropy_history.clear()
        self._confidence_history.clear()


# ==================== 特征正则化 ====================

class FeatureRegularization(RegularizationLoss):
    """
    特征正则化
    
    正则化中间特征表示。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        target_norm: float = 1.0,
        norm_type: str = 'l2',  # l2, l1, max
        **kwargs
    ):
        super().__init__(config)
        self.target_norm = target_norm
        self.norm_type = norm_type
        
        # 新增：特征统计历史
        self._norm_history: List[float] = []
        self._std_history: List[float] = []
    
    def forward(
        self, 
        predictions: Tensor,  # features [N, D]
        targets: Any = None,
        **kwargs
    ) -> Tensor:
        """正则化特征范数"""
        # 计算特征范数
        if self.norm_type == 'l2':
            norms = predictions.norm(dim=-1)  # [N]
        elif self.norm_type == 'l1':
            norms = predictions.abs().sum(dim=-1)
        elif self.norm_type == 'max':
            norms = predictions.abs().max(dim=-1).values
        else:
            norms = predictions.norm(dim=-1)
        
        # 鼓励特征范数接近目标值
        return F.mse_loss(norms, torch.full_like(norms, self.target_norm))
    
    def compute(
        self, 
        predictions: Tensor,
        targets: Any = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        # 计算特征范数
        if self.norm_type == 'l2':
            norms = predictions.norm(dim=-1)
        elif self.norm_type == 'l1':
            norms = predictions.abs().sum(dim=-1)
        elif self.norm_type == 'max':
            norms = predictions.abs().max(dim=-1).values
        else:
            norms = predictions.norm(dim=-1)
        
        loss = F.mse_loss(norms, torch.full_like(norms, self.target_norm))
        
        # 计算特征统计
        mean_norm = norms.mean().item()
        std_norm = norms.std().item()
        
        # 计算特征维度统计
        mean_feature = predictions.mean().item()
        std_feature = predictions.std().item()
        
        # 计算特征相关性
        if predictions.size(0) > 1:
            # 样本间相关性
            normed_features = F.normalize(predictions, dim=-1)
            similarity = (normed_features @ normed_features.t()).mean().item()
        else:
            similarity = 1.0
        
        # 记录历史
        self._norm_history.append(mean_norm)
        self._std_history.append(std_norm)
        
        if len(self._norm_history) > 100:
            self._norm_history.pop(0)
            self._std_history.pop(0)
        
        # 记录统计
        self._record_reg_stats(reg_loss=loss.item())
        
        return LossResult(
            loss=loss * self.config.weight,
            components={'feature_loss': loss, 'norms': norms},
            metrics={
                'feature_loss': loss.item(),
                'mean_norm': mean_norm,
                'std_norm': std_norm,
                'min_norm': norms.min().item(),
                'max_norm': norms.max().item(),
                'target_norm': self.target_norm,
                'norm_deviation': abs(mean_norm - self.target_norm),
                'mean_feature': mean_feature,
                'std_feature': std_feature,
                'sample_similarity': similarity,
                'norm_type': self.norm_type
            },
            step=self._rl_step
        )
    
    def get_feature_stats(self) -> Dict[str, float]:
        """获取特征统计"""
        if not self._norm_history:
            return {}
        
        return {
            'avg_norm': sum(self._norm_history) / len(self._norm_history),
            'avg_std': sum(self._std_history) / len(self._std_history),
            'recent_norm': self._norm_history[-1],
            'target_norm': self.target_norm
        }
    
    def reset_feature_stats(self) -> None:
        """重置特征统计"""
        self._norm_history.clear()
        self._std_history.clear()


# ==================== 谱归一化正则化 ====================

class SpectralRegularization(RegularizationLoss):
    """
    谱归一化正则化
    
    限制权重矩阵的谱范数以稳定训练。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        lambda_: float = 1e-4,
        power_iterations: int = 1,
        lambda_schedule: str = 'constant',
        **kwargs
    ):
        super().__init__(config, lambda_schedule=lambda_schedule)
        self.lambda_ = lambda_
        self.power_iterations = power_iterations
        self._init_lambda_scheduler(lambda_)
        
        # 新增：谱范数历史
        self._spectral_norms: Dict[str, List[float]] = defaultdict(list)
    
    def _compute_spectral_norm(self, weight: Tensor) -> Tensor:
        """计算谱范数"""
        # 处理Conv2d权重
        if weight.dim() > 2:
            weight = weight.flatten(1)
        
        # 幂迭代法近似谱范数
        u = weight.new_empty(weight.size(0)).normal_()
        u = F.normalize(u, dim=0)
        
        with torch.no_grad():
            for _ in range(self.power_iterations):
                v = F.normalize(weight.t() @ u, dim=0)
                u = F.normalize(weight @ v, dim=0)
        
        sigma = (u @ weight @ v).abs()
        return sigma
    
    def forward(
        self, 
        predictions: nn.Module,
        targets: Any = None,
        **kwargs
    ) -> Tensor:
        """计算谱正则化"""
        spectral_norm = torch.tensor(0.0)
        
        if isinstance(predictions, nn.Module):
            for name, module in predictions.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    weight = module.weight
                    sigma = self._compute_spectral_norm(weight)
                    spectral_norm = spectral_norm + sigma
        
        effective_lambda = self.get_effective_lambda()
        return effective_lambda * spectral_norm
    
    def compute(
        self, 
        predictions: nn.Module,
        targets: Any = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        spectral_norm = torch.tensor(0.0)
        layer_norms = {}
        layer_count = 0
        
        if isinstance(predictions, nn.Module):
            for name, module in predictions.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    weight = module.weight
                    sigma = self._compute_spectral_norm(weight)
                    spectral_norm = spectral_norm + sigma
                    
                    layer_norms[name] = sigma.item()
                    
                    # 记录历史
                    self._spectral_norms[name].append(sigma.item())
                    if len(self._spectral_norms[name]) > 100:
                        self._spectral_norms[name].pop(0)
                    
                    layer_count += 1
        
        effective_lambda = self.get_effective_lambda()
        loss = effective_lambda * spectral_norm
        
        # 计算统计
        if layer_norms:
            max_spectral = max(layer_norms.values())
            mean_spectral = sum(layer_norms.values()) / len(layer_norms)
        else:
            max_spectral = 0.0
            mean_spectral = 0.0
        
        # 记录统计
        self._record_reg_stats(reg_loss=loss.item())
        
        # 更新Lambda调度
        if self._lambda_scheduler is not None:
            self._lambda_scheduler.step(loss.item())
        
        return LossResult(
            loss=loss * self.config.weight,
            components={'spectral_norm': spectral_norm},
            metrics={
                'spectral_loss': loss.item(),
                'total_spectral_norm': spectral_norm.item(),
                'max_spectral_norm': max_spectral,
                'mean_spectral_norm': mean_spectral,
                'num_layers': layer_count,
                'lambda': effective_lambda,
                'layer_norms': layer_norms
            },
            step=self._rl_step
        )
    
    def get_spectral_stats(self) -> Dict[str, Dict[str, float]]:
        """获取谱范数统计"""
        stats = {}
        
        for name, norms in self._spectral_norms.items():
            if norms:
                stats[name] = {
                    'avg': sum(norms) / len(norms),
                    'current': norms[-1],
                    'max': max(norms),
                    'min': min(norms)
                }
        
        return stats
    
    def reset_spectral_stats(self) -> None:
        """重置谱范数统计"""
        self._spectral_norms.clear()


# ==================== 混合正则化 ====================

class MixedRegularization(RegularizationLoss):
    """
    混合正则化
    
    组合多种正则化方法。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        l1_weight: float = 0.0,
        l2_weight: float = 1e-4,
        entropy_weight: float = 0.0,
        spectral_weight: float = 0.0,
        feature_weight: float = 0.0,
        auto_balance: bool = False,
        **kwargs
    ):
        super().__init__(config)
        
        self.l1_reg = L1Regularization(lambda_=l1_weight) if l1_weight > 0 else None
        self.l2_reg = L2Regularization(lambda_=l2_weight) if l2_weight > 0 else None
        self.entropy_reg = EntropyRegularization() if entropy_weight > 0 else None
        self.spectral_reg = SpectralRegularization(lambda_=spectral_weight) if spectral_weight > 0 else None
        self.feature_reg = FeatureRegularization() if feature_weight > 0 else None
        
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.entropy_weight = entropy_weight
        self.spectral_weight = spectral_weight
        self.feature_weight = feature_weight
        
        self._auto_balance = auto_balance
        self._initial_weights = {
            'l1': l1_weight,
            'l2': l2_weight,
            'entropy': entropy_weight,
            'spectral': spectral_weight,
            'feature': feature_weight
        }
        
        # 新增：组件损失历史
        self._component_losses: Dict[str, List[float]] = defaultdict(list)
    
    def forward(
        self, 
        predictions: nn.Module,
        targets: Any = None,
        logits: Optional[Tensor] = None,
        features: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """计算混合正则化"""
        total_reg = torch.tensor(0.0)
        
        if self.l1_reg:
            total_reg = total_reg + self.l1_reg(predictions)
        
        if self.l2_reg:
            total_reg = total_reg + self.l2_reg(predictions)
        
        if self.entropy_reg and logits is not None:
            total_reg = total_reg + self.entropy_weight * self.entropy_reg(logits)
        
        if self.spectral_reg:
            total_reg = total_reg + self.spectral_reg(predictions)
        
        if self.feature_reg and features is not None:
            total_reg = total_reg + self.feature_weight * self.feature_reg(features)
        
        return total_reg
    
    def compute(
        self, 
        predictions: nn.Module,
        targets: Any = None,
        logits: Optional[Tensor] = None,
        features: Optional[Tensor] = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        components = {}
        metrics = {}
        total_reg = torch.tensor(0.0)
        
        # L1正则化
        if self.l1_reg:
            l1_result = self.l1_reg.compute(predictions)
            l1_loss = l1_result.loss / self.l1_reg.config.weight
            components['l1_loss'] = l1_loss
            metrics['l1_loss'] = l1_loss.item()
            metrics['l1_sparsity'] = l1_result.metrics.get('sparsity', 0.0)
            
            self._component_losses['l1'].append(l1_loss.item())
            total_reg = total_reg + l1_loss
        
        # L2正则化
        if self.l2_reg:
            l2_result = self.l2_reg.compute(predictions)
            l2_loss = l2_result.loss / self.l2_reg.config.weight
            components['l2_loss'] = l2_loss
            metrics['l2_loss'] = l2_loss.item()
            metrics['l2_norm'] = l2_result.metrics.get('l2_norm', 0.0)
            
            self._component_losses['l2'].append(l2_loss.item())
            total_reg = total_reg + l2_loss
        
        # 熵正则化
        if self.entropy_reg and logits is not None:
            entropy_result = self.entropy_reg.compute(logits)
            entropy_loss = entropy_result.loss / self.entropy_reg.config.weight
            weighted_entropy = self.entropy_weight * entropy_loss
            components['entropy_loss'] = weighted_entropy
            metrics['entropy_loss'] = weighted_entropy.item()
            metrics['entropy'] = entropy_result.metrics.get('mean_entropy', 0.0)
            
            self._component_losses['entropy'].append(weighted_entropy.item())
            total_reg = total_reg + weighted_entropy
        
        # 谱正则化
        if self.spectral_reg:
            spectral_result = self.spectral_reg.compute(predictions)
            spectral_loss = spectral_result.loss / self.spectral_reg.config.weight
            components['spectral_loss'] = spectral_loss
            metrics['spectral_loss'] = spectral_loss.item()
            metrics['spectral_norm'] = spectral_result.metrics.get('total_spectral_norm', 0.0)
            
            self._component_losses['spectral'].append(spectral_loss.item())
            total_reg = total_reg + spectral_loss
        
        # 特征正则化
        if self.feature_reg and features is not None:
            feature_result = self.feature_reg.compute(features)
            feature_loss = feature_result.loss / self.feature_reg.config.weight
            weighted_feature = self.feature_weight * feature_loss
            components['feature_loss'] = weighted_feature
            metrics['feature_loss'] = weighted_feature.item()
            metrics['feature_norm'] = feature_result.metrics.get('mean_norm', 0.0)
            
            self._component_losses['feature'].append(weighted_feature.item())
            total_reg = total_reg + weighted_feature
        
        # 限制历史长度
        for key in self._component_losses:
            if len(self._component_losses[key]) > 100:
                self._component_losses[key].pop(0)
        
        # 自动平衡权重
        if self._auto_balance and self._rl_step > 0 and self._rl_step % 100 == 0:
            self._update_weights()
        
        # 记录统计
        self._record_reg_stats(reg_loss=total_reg.item())
        
        metrics['total_loss'] = total_reg.item()
        metrics['weights'] = self.get_all_weights()
        
        return LossResult(
            loss=total_reg * self.config.weight,
            components=components,
            metrics=metrics,
            step=self._rl_step
        )
    
    def _update_weights(self) -> None:
        """自动更新组件权重"""
        if not self._component_losses:
            return
        
        # 计算各组件平均损失
        avg_losses = {}
        for name, losses in self._component_losses.items():
            if losses:
                avg_losses[name] = sum(losses[-50:]) / len(losses[-50:])
        
        if not avg_losses:
            return
        
        # 归一化权重（损失大的权重减小以平衡）
        total = sum(avg_losses.values()) + 1e-8
        
        for name, avg_loss in avg_losses.items():
            new_weight = 1.0 / (avg_loss / total + 1e-8)
            initial_weight = self._initial_weights.get(name, 0.0)
            
            # 混合初始权重和自适应权重
            mixed_weight = 0.7 * initial_weight + 0.3 * new_weight
            
            if name == 'l1' and self.l1_reg:
                self.l1_weight = mixed_weight
            elif name == 'l2' and self.l2_reg:
                self.l2_weight = mixed_weight
            elif name == 'entropy':
                self.entropy_weight = mixed_weight
            elif name == 'spectral':
                self.spectral_weight = mixed_weight
            elif name == 'feature':
                self.feature_weight = mixed_weight
    
    def set_auto_balance(self, enabled: bool) -> None:
        """设置自动平衡"""
        self._auto_balance = enabled
    
    def get_all_weights(self) -> Dict[str, float]:
        """获取所有权重"""
        return {
            'l1': self.l1_weight,
            'l2': self.l2_weight,
            'entropy': self.entropy_weight,
            'spectral': self.spectral_weight,
            'feature': self.feature_weight
        }
    
    def set_weights(
        self,
        l1: Optional[float] = None,
        l2: Optional[float] = None,
        entropy: Optional[float] = None,
        spectral: Optional[float] = None,
        feature: Optional[float] = None
    ) -> None:
        """设置权重"""
        if l1 is not None:
            self.l1_weight = l1
        if l2 is not None:
            self.l2_weight = l2
        if entropy is not None:
            self.entropy_weight = entropy
        if spectral is not None:
            self.spectral_weight = spectral
        if feature is not None:
            self.feature_weight = feature
    
    def get_component_stats(self) -> Dict[str, Dict[str, float]]:
        """获取组件统计"""
        stats = {}
        
        for name, losses in self._component_losses.items():
            if losses:
                stats[name] = {
                    'avg_loss': sum(losses) / len(losses),
                    'recent_loss': losses[-1],
                    'min_loss': min(losses),
                    'max_loss': max(losses)
                }
        
        return stats
    
    def print_mixed_summary(self) -> None:
        """打印混合正则化摘要"""
        stats = self.get_reg_stats()
        component_stats = self.get_component_stats()
        
        print("\n" + "="*80)
        print(f"Mixed Regularization Summary")
        print("="*80)
        
        print(f"\nCurrent Weights:")
        weights = self.get_all_weights()
        for name, weight in weights.items():
            if weight > 0:
                print(f"  {name}: {weight:.6f}")
        
        print(f"\nAuto balance: {self._auto_balance}")
        
        print(f"\nComponent Statistics:")
        for name, stat in component_stats.items():
            print(f"  {name}:")
            print(f"    Avg loss: {stat['avg_loss']:.6f}")
            print(f"    Recent loss: {stat['recent_loss']:.6f}")
        
        print(f"\nOverall Statistics (over {stats.total_steps} steps):")
        print(f"  Avg total loss: {stats.avg_reg_loss:.6f}")
        if stats.avg_sparsity > 0:
            print(f"  Avg sparsity: {stats.avg_sparsity:.4f}")
        
        print("="*80)
    
    def reset_mixed_stats(self) -> None:
        """重置混合统计"""
        self._component_losses.clear()
        if self.l1_reg:
            self.l1_reg.reset_reg_stats()
        if self.l2_reg:
            self.l2_reg.reset_reg_stats()
        if self.entropy_reg:
            self.entropy_reg.reset_reg_stats()
        if self.spectral_reg:
            self.spectral_reg.reset_reg_stats()
        if self.feature_reg:
            self.feature_reg.reset_reg_stats()


# ==================== 工具函数 ====================

def create_regularization_loss(
    reg_type: str,
    lambda_: float = 1e-4,
    lambda_schedule: str = 'constant',
    **kwargs
) -> RegularizationLoss:
    """
    创建正则化损失
    
    Args:
        reg_type: 正则化类型 (l1, l2, elastic_net, consistency, entropy, feature, spectral, mixed)
        lambda_: 正则化强度
        lambda_schedule: Lambda调度策略
        **kwargs: 额外参数
        
    Returns:
        正则化损失实例
    """
    reg_classes = {
        'l1': L1Regularization,
        'l2': L2Regularization,
        'elastic_net': ElasticNetRegularization,
        'consistency': ConsistencyRegularization,
        'entropy': EntropyRegularization,
        'feature': FeatureRegularization,
        'spectral': SpectralRegularization,
        'mixed': MixedRegularization,
    }
    
    if reg_type not in reg_classes:
        raise ValueError(f"Unknown regularization type: {reg_type}. Available: {list(reg_classes.keys())}")
    
    reg_class = reg_classes[reg_type]
    
    if reg_type in ('l1', 'l2', 'elastic_net', 'spectral'):
        return reg_class(lambda_=lambda_, lambda_schedule=lambda_schedule, **kwargs)
    else:
        return reg_class(**kwargs)


def compare_regularization_losses(losses: Dict[str, RegularizationLoss]) -> None:
    """
    对比多个正则化损失
    
    Args:
        losses: 损失字典
    """
    print("\n" + "="*100)
    print("Regularization Loss Comparison")
    print("="*100)
    
    print(f"\n{'Name':<20} {'Type':<25} {'Avg Loss':<12} {'Sparsity':<12} {'Effective':<10}")
    print("-"*100)
    
    for name, loss_fn in losses.items():
        loss_type = loss_fn.__class__.__name__
        stats = loss_fn.get_reg_stats()
        effective = loss_fn.is_regularization_effective()
        
        print(f"{name:<20} {loss_type:<25} {stats.avg_reg_loss:<12.6f} "
              f"{stats.avg_sparsity:<12.4f} {str(effective):<10}")
    
    print("="*100)


def analyze_model_weights(model: nn.Module, threshold: float = 1e-6) -> Dict[str, Any]:
    """
    分析模型权重
    
    Args:
        model: 模型
        threshold: 零值阈值
        
    Returns:
        分析结果
    """
    analyzer = SparsityAnalyzer(threshold=threshold)
    return analyzer.analyze_model(model)


def print_weight_analysis(model: nn.Module, threshold: float = 1e-6) -> None:
    """
    打印权重分析
    
    Args:
        model: 模型
        threshold: 零值阈值
    """
    analysis = analyze_model_weights(model, threshold)
    
    print("\n" + "="*80)
    print("Model Weight Analysis")
    print("="*80)
    
    print(f"\nOverall Statistics:")
    print(f"  Total parameters: {analysis['total_params']:,}")
    print(f"  Zero parameters: {analysis['zero_params']:,}")
    print(f"  Near-zero parameters: {analysis['near_zero_params']:,}")
    print(f"  Overall sparsity: {analysis['overall_sparsity']:.4f}")
    print(f"  Near-zero ratio: {analysis['near_zero_ratio']:.4f}")
    
    print(f"\nLayer-wise Sparsity (top 10 by sparsity):")
    layer_sparsity = analysis['layer_sparsity']
    sorted_layers = sorted(layer_sparsity.items(), key=lambda x: x[1]['sparsity'], reverse=True)[:10]
    
    for name, stats in sorted_layers:
        print(f"  {name[:40]:<40} sparsity: {stats['sparsity']:.4f} ({stats['zeros']}/{stats['total']})")
    
    print("="*80)


def recommend_regularization(
    model: nn.Module,
    task: str = 'classification',
    dataset_size: int = 10000,
    model_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    推荐正则化配置
    
    Args:
        model: 模型
        task: 任务类型
        dataset_size: 数据集大小
        model_size: 模型大小（参数数量，可选）
        
    Returns:
        推荐配置
    """
    if model_size is None:
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算模型/数据比
    ratio = model_size / dataset_size
    
    recommendations = {
        'l1_weight': 0.0,
        'l2_weight': 1e-4,
        'entropy_weight': 0.0,
        'spectral_weight': 0.0,
        'explanation': []
    }
    
    # 根据模型/数据比调整
    if ratio > 100:
        # 模型远大于数据，需要强正则化
        recommendations['l2_weight'] = 1e-3
        recommendations['l1_weight'] = 1e-4
        recommendations['explanation'].append("Strong regularization recommended due to high model/data ratio")
    elif ratio > 10:
        # 适中比例
        recommendations['l2_weight'] = 1e-4
        recommendations['explanation'].append("Moderate L2 regularization recommended")
    else:
        # 数据充足
        recommendations['l2_weight'] = 1e-5
        recommendations['explanation'].append("Light regularization sufficient with abundant data")
    
    # 根据任务调整
    if task == 'classification':
        if ratio > 50:
            recommendations['entropy_weight'] = 0.1
            recommendations['explanation'].append("Entropy regularization for confident predictions")
    elif task == 'generation':
        recommendations['spectral_weight'] = 1e-4
        recommendations['explanation'].append("Spectral regularization for training stability")
    
    return recommendations


def print_regularization_recommendation(
    model: nn.Module,
    task: str = 'classification',
    dataset_size: int = 10000
) -> None:
    """
    打印正则化推荐
    
    Args:
        model: 模型
        task: 任务类型
        dataset_size: 数据集大小
    """
    rec = recommend_regularization(model, task, dataset_size)
    
    print("\n" + "="*60)
    print("Regularization Recommendation")
    print("="*60)
    
    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel size: {model_size:,} parameters")
    print(f"Dataset size: {dataset_size:,}")
    print(f"Task: {task}")
    
    print(f"\nRecommended Configuration:")
    print(f"  L1 weight: {rec['l1_weight']:.6f}")
    print(f"  L2 weight: {rec['l2_weight']:.6f}")
    print(f"  Entropy weight: {rec['entropy_weight']:.4f}")
    print(f"  Spectral weight: {rec['spectral_weight']:.6f}")
    
    print(f"\nExplanation:")
    for exp in rec['explanation']:
        print(f"  - {exp}")
    
    print("="*60)



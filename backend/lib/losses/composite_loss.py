# -*- coding: utf-8 -*-
"""
复合损失函数

用于组合多个损失函数的模块。
"""

import logging
import time
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor

from .base_loss import BaseLoss, LossConfig, LossResult, LossMonitor, LossStats

logger = logging.getLogger(__name__)


# ==================== 监控和平衡组件 ====================

@dataclass
class WeightStats:
    """权重统计"""
    weights: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    step: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'weights': self.weights,
            'timestamp': self.timestamp,
            'step': self.step
        }


class WeightMonitor:
    """权重监控器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._history: List[WeightStats] = []
        
    def record(self, weights: Dict[str, float], step: int = 0) -> None:
        """记录权重"""
        stats = WeightStats(weights=weights.copy(), step=step)
        self._history.append(stats)
        
        if len(self._history) > self.max_history:
            self._history.pop(0)
    
    def get_recent_weights(self, n: int = 10) -> List[Dict[str, float]]:
        """获取最近的权重"""
        return [h.weights for h in self._history[-n:]]
    
    def get_weight_trend(self, name: str, window: int = 100) -> float:
        """获取权重趋势"""
        if len(self._history) < 2:
            return 0.0
        
        recent = self._history[-window:]
        weights = [h.weights.get(name, 0.0) for h in recent]
        
        if len(weights) < 2:
            return 0.0
        
        return weights[-1] - weights[0]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._history:
            return {}
        
        latest = self._history[-1]
        
        stats = {
            'current_weights': latest.weights,
            'total_records': len(self._history),
        }
        
        # 计算平均权重
        if self._history:
            avg_weights = defaultdict(float)
            for h in self._history:
                for name, weight in h.weights.items():
                    avg_weights[name] += weight
            
            for name in avg_weights:
                avg_weights[name] /= len(self._history)
            
            stats['avg_weights'] = dict(avg_weights)
        
        return stats
    
    def reset(self) -> None:
        """重置监控器"""
        self._history.clear()


class TaskBalancer:
    """任务平衡器"""
    
    def __init__(self, tasks: List[str]):
        self.tasks = tasks
        self._task_losses: Dict[str, List[float]] = {task: [] for task in tasks}
        self._task_weights: Dict[str, float] = {task: 1.0 for task in tasks}
        
    def record_losses(self, losses: Dict[str, float]) -> None:
        """记录任务损失"""
        for task, loss in losses.items():
            if task in self._task_losses:
                self._task_losses[task].append(loss)
                
                # 限制历史长度
                if len(self._task_losses[task]) > 100:
                    self._task_losses[task].pop(0)
    
    def compute_balanced_weights(self, method: str = 'inverse') -> Dict[str, float]:
        """
        计算平衡的权重
        
        Args:
            method: 平衡方法 ('inverse', 'softmax', 'uniform')
            
        Returns:
            平衡后的权重
        """
        if method == 'uniform':
            return {task: 1.0 for task in self.tasks}
        
        # 计算平均损失
        avg_losses = {}
        for task in self.tasks:
            if self._task_losses[task]:
                avg_losses[task] = sum(self._task_losses[task]) / len(self._task_losses[task])
            else:
                avg_losses[task] = 1.0
        
        if method == 'inverse':
            # 反比例：损失越大，权重越小
            total_inv = sum(1.0 / (loss + 1e-8) for loss in avg_losses.values())
            weights = {
                task: (1.0 / (avg_losses[task] + 1e-8)) / total_inv
                for task in self.tasks
            }
        
        elif method == 'softmax':
            # Softmax归一化
            import math
            exp_losses = {task: math.exp(-avg_losses[task]) for task in self.tasks}
            total_exp = sum(exp_losses.values())
            weights = {task: exp_losses[task] / total_exp for task in self.tasks}
        
        else:
            weights = {task: 1.0 for task in self.tasks}
        
        self._task_weights = weights
        return weights
    
    def get_current_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        return self._task_weights.copy()
    
    def get_task_stats(self) -> Dict[str, Dict[str, float]]:
        """获取任务统计"""
        stats = {}
        
        for task in self.tasks:
            if not self._task_losses[task]:
                continue
            
            losses = self._task_losses[task]
            stats[task] = {
                'avg_loss': sum(losses) / len(losses),
                'min_loss': min(losses),
                'max_loss': max(losses),
                'current_weight': self._task_weights.get(task, 1.0),
            }
        
        return stats




class CompositeLoss(BaseLoss):
    """
    复合损失
    
    将多个损失函数按权重组合。
    L = Σ w_i * L_i
    """
    
    def __init__(
        self, 
        losses: List[Tuple[str, BaseLoss, float]],
        config: Optional[LossConfig] = None,
        auto_balance: bool = False,
        balance_method: str = 'inverse',
        **kwargs
    ):
        """
        初始化复合损失
        
        Args:
            losses: 损失列表 [(name, loss_fn, weight), ...]
            config: 配置
            auto_balance: 是否自动平衡权重
            balance_method: 平衡方法
        """
        super().__init__(config)
        
        self.loss_names = []
        self.loss_fns = nn.ModuleList()
        self.loss_weights = []
        
        for name, loss_fn, weight in losses:
            self.loss_names.append(name)
            self.loss_fns.append(loss_fn)
            self.loss_weights.append(weight)
        
        # 新增：监控和平衡
        self._weight_monitor = WeightMonitor()
        self._balancer = TaskBalancer(self.loss_names)
        self._auto_balance = auto_balance
        self._balance_method = balance_method
        self._component_monitors: Dict[str, LossMonitor] = {
            name: LossMonitor() for name in self.loss_names
        }
        self._comp_step = 0
    
    def add_loss(self, name: str, loss_fn: BaseLoss, weight: float = 1.0):
        """添加损失函数"""
        self.loss_names.append(name)
        self.loss_fns.append(loss_fn)
        self.loss_weights.append(weight)
        self._component_monitors[name] = LossMonitor()
        self._balancer = TaskBalancer(self.loss_names)
    
    def remove_loss(self, name: str) -> bool:
        """移除损失函数"""
        if name not in self.loss_names:
            return False
        
        idx = self.loss_names.index(name)
        self.loss_names.pop(idx)
        del self.loss_fns[idx]
        self.loss_weights.pop(idx)
        self._component_monitors.pop(name, None)
        self._balancer = TaskBalancer(self.loss_names)
        
        return True
    
    def set_weight(self, name: str, weight: float):
        """设置损失权重"""
        if name in self.loss_names:
            idx = self.loss_names.index(name)
            self.loss_weights[idx] = weight
    
    def get_weight(self, name: str) -> Optional[float]:
        """获取损失权重"""
        if name in self.loss_names:
            idx = self.loss_names.index(name)
            return self.loss_weights[idx]
        return None
    
    def set_auto_balance(self, enabled: bool, method: str = 'inverse') -> None:
        """设置自动平衡"""
        self._auto_balance = enabled
        self._balance_method = method
    
    def forward(
        self, 
        predictions: Any, 
        targets: Any,
        **kwargs
    ) -> Tensor:
        """
        计算复合损失
        
        kwargs 可以包含各个子损失需要的额外参数。
        """
        total_loss = torch.tensor(0.0, device=self._get_device(predictions))
        
        for name, loss_fn, weight in zip(self.loss_names, self.loss_fns, self.loss_weights):
            if weight <= 0:
                continue
            
            try:
                # 尝试获取该损失的专用参数
                loss_kwargs = kwargs.get(name, {})
                if not isinstance(loss_kwargs, dict):
                    loss_kwargs = {}
                
                loss = loss_fn(predictions, targets, **{**kwargs, **loss_kwargs})
                total_loss = total_loss + weight * loss
            except Exception as e:
                logger.warning(f"Loss {name} computation failed: {e}")
        
        return total_loss
    
    def compute(
        self, 
        predictions: Any, 
        targets: Any,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        device = self._get_device(predictions)
        total_loss = torch.tensor(0.0, device=device)
        components = {}
        metrics = {}
        component_losses = {}
        
        for name, loss_fn, weight in zip(self.loss_names, self.loss_fns, self.loss_weights):
            if weight <= 0:
                continue
            
            try:
                loss_kwargs = kwargs.get(name, {})
                if not isinstance(loss_kwargs, dict):
                    loss_kwargs = {}
                
                if hasattr(loss_fn, 'compute'):
                    result = loss_fn.compute(predictions, targets, **{**kwargs, **loss_kwargs})
                    loss = result.loss
                    components[name] = loss
                    metrics.update({f"{name}_{k}": v for k, v in result.metrics.items()})
                else:
                    loss = loss_fn(predictions, targets, **{**kwargs, **loss_kwargs})
                    components[name] = loss
                    metrics[name] = loss.item()
                
                # 记录组件损失
                component_losses[name] = loss.item()
                
                # 记录到组件监控器
                comp_result = LossResult(loss=loss, step=self._comp_step)
                self._component_monitors[name].record(comp_result)
                
                total_loss = total_loss + weight * loss
            except Exception as e:
                logger.warning(f"Loss {name} computation failed: {e}")
        
        # 自动平衡
        if self._auto_balance and component_losses:
            self._balancer.record_losses(component_losses)
            if self._comp_step % 100 == 0:  # 每100步更新一次权重
                balanced_weights = self._balancer.compute_balanced_weights(self._balance_method)
                for i, name in enumerate(self.loss_names):
                    self.loss_weights[i] = balanced_weights.get(name, 1.0)
        
        # 记录权重
        current_weights = {name: self.loss_weights[i] for i, name in enumerate(self.loss_names)}
        self._weight_monitor.record(current_weights, self._comp_step)
        
        metrics['total_loss'] = total_loss.item()
        metrics['weights'] = current_weights
        
        self._comp_step += 1
        
        return LossResult(
            loss=total_loss * self.config.weight,
            components=components,
            metrics=metrics,
            step=self._comp_step
        )
    
    def _get_device(self, x: Any) -> torch.device:
        """获取设备"""
        if isinstance(x, Tensor):
            return x.device
        elif isinstance(x, dict):
            for v in x.values():
                if isinstance(v, Tensor):
                    return v.device
        return torch.device('cpu')
    
    # ==================== 新增方法 ====================
    
    def get_component_stats(self, name: str) -> Optional[LossStats]:
        """获取组件统计"""
        if name in self._component_monitors:
            return self._component_monitors[name].get_stats()
        return None
    
    def get_all_component_stats(self) -> Dict[str, LossStats]:
        """获取所有组件统计"""
        return {
            name: monitor.get_stats()
            for name, monitor in self._component_monitors.items()
        }
    
    def get_weight_stats(self) -> Dict[str, Any]:
        """获取权重统计"""
        return self._weight_monitor.get_stats()
    
    def get_weight_history(self, n: int = 10) -> List[Dict[str, float]]:
        """获取权重历史"""
        return self._weight_monitor.get_recent_weights(n)
    
    def get_balance_stats(self) -> Dict[str, Dict[str, float]]:
        """获取平衡统计"""
        return self._balancer.get_task_stats()
    
    def print_summary(self) -> None:
        """打印摘要"""
        print("\n" + "="*80)
        print("Composite Loss Summary")
        print("="*80)
        
        print(f"\nTotal components: {len(self.loss_names)}")
        print(f"Auto balance: {self._auto_balance}")
        if self._auto_balance:
            print(f"Balance method: {self._balance_method}")
        
        print(f"\nComponent Details:")
        print(f"{'Name':<20} {'Weight':<10} {'Avg Loss':<12} {'Steps':<8}")
        print("-"*80)
        
        for i, name in enumerate(self.loss_names):
            weight = self.loss_weights[i]
            stats = self.get_component_stats(name)
            if stats:
                print(f"{name:<20} {weight:<10.4f} {stats.avg_loss:<12.6f} {stats.total_steps:<8}")
            else:
                print(f"{name:<20} {weight:<10.4f} {'N/A':<12} {0:<8}")
        
        print("="*80)
    
    def reset_comp_stats(self) -> None:
        """重置所有统计"""
        self._weight_monitor.reset()
        for monitor in self._component_monitors.values():
            monitor.reset()
        self._comp_step = 0


# ==================== 多任务损失 ====================

class MultiTaskLoss(BaseLoss):
    """
    多任务损失
    
    用于多任务学习，支持任务级别的权重管理。
    """
    
    def __init__(
        self, 
        tasks: Dict[str, BaseLoss],
        task_weights: Optional[Dict[str, float]] = None,
        config: Optional[LossConfig] = None,
        auto_balance: bool = False,
        balance_method: str = 'inverse',
        **kwargs
    ):
        """
        初始化多任务损失
        
        Args:
            tasks: 任务字典 {task_name: loss_fn}
            task_weights: 任务权重 {task_name: weight}
            auto_balance: 是否自动平衡
            balance_method: 平衡方法
        """
        super().__init__(config)
        
        self.tasks = nn.ModuleDict(tasks)
        self.task_weights = task_weights or {k: 1.0 for k in tasks}
        
        # 新增：监控组件
        self._task_monitors: Dict[str, LossMonitor] = {
            name: LossMonitor() for name in tasks
        }
        self._weight_monitor = WeightMonitor()
        self._balancer = TaskBalancer(list(tasks.keys()))
        self._auto_balance = auto_balance
        self._balance_method = balance_method
        self._mt_step = 0
    
    def forward(
        self, 
        predictions: Dict[str, Tensor],  # {task: predictions}
        targets: Dict[str, Tensor],      # {task: targets}
        **kwargs
    ) -> Tensor:
        """计算多任务损失"""
        total_loss = torch.tensor(0.0)
        
        for task_name, loss_fn in self.tasks.items():
            if task_name not in predictions or task_name not in targets:
                continue
            
            weight = self.task_weights.get(task_name, 1.0)
            if weight <= 0:
                continue
            
            task_pred = predictions[task_name]
            task_target = targets[task_name]
            
            try:
                loss = loss_fn(task_pred, task_target)
                if total_loss.device != loss.device:
                    total_loss = total_loss.to(loss.device)
                total_loss = total_loss + weight * loss
            except Exception as e:
                logger.warning(f"Task {task_name} loss failed: {e}")
        
        return total_loss
    
    def compute(
        self, 
        predictions: Dict[str, Tensor], 
        targets: Dict[str, Tensor],
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        total_loss = torch.tensor(0.0)
        components = {}
        metrics = {}
        task_losses = {}
        
        for task_name, loss_fn in self.tasks.items():
            if task_name not in predictions or task_name not in targets:
                continue
            
            weight = self.task_weights.get(task_name, 1.0)
            if weight <= 0:
                continue
            
            task_pred = predictions[task_name]
            task_target = targets[task_name]
            
            try:
                if hasattr(loss_fn, 'compute'):
                    result = loss_fn.compute(task_pred, task_target)
                    loss = result.loss
                    metrics.update({f"{task_name}_{k}": v for k, v in result.metrics.items()})
                else:
                    loss = loss_fn(task_pred, task_target)
                
                components[task_name] = loss
                metrics[f"{task_name}_loss"] = loss.item()
                
                # 记录任务损失
                task_losses[task_name] = loss.item()
                
                # 记录到任务监控器
                task_result = LossResult(loss=loss, step=self._mt_step)
                self._task_monitors[task_name].record(task_result)
                
                if total_loss.device != loss.device:
                    total_loss = total_loss.to(loss.device)
                total_loss = total_loss + weight * loss
            except Exception as e:
                logger.warning(f"Task {task_name} loss failed: {e}")
        
        # 自动平衡
        if self._auto_balance and task_losses:
            self._balancer.record_losses(task_losses)
            if self._mt_step % 100 == 0:
                balanced_weights = self._balancer.compute_balanced_weights(self._balance_method)
                self.task_weights.update(balanced_weights)
        
        # 记录权重
        self._weight_monitor.record(self.task_weights.copy(), self._mt_step)
        
        metrics['task_weights'] = self.task_weights.copy()
        
        self._mt_step += 1
        
        return LossResult(
            loss=total_loss * self.config.weight,
            components=components,
            metrics=metrics,
            step=self._mt_step
        )
    
    # ==================== 新增方法 ====================
    
    def add_task(self, name: str, loss_fn: BaseLoss, weight: float = 1.0) -> None:
        """添加任务"""
        self.tasks[name] = loss_fn
        self.task_weights[name] = weight
        self._task_monitors[name] = LossMonitor()
        self._balancer = TaskBalancer(list(self.tasks.keys()))
    
    def remove_task(self, name: str) -> bool:
        """移除任务"""
        if name not in self.tasks:
            return False
        
        del self.tasks[name]
        self.task_weights.pop(name, None)
        self._task_monitors.pop(name, None)
        self._balancer = TaskBalancer(list(self.tasks.keys()))
        
        return True
    
    def set_task_weight(self, name: str, weight: float) -> None:
        """设置任务权重"""
        if name in self.tasks:
            self.task_weights[name] = weight
    
    def get_task_weight(self, name: str) -> Optional[float]:
        """获取任务权重"""
        return self.task_weights.get(name)
    
    def get_task_stats(self, name: str) -> Optional[LossStats]:
        """获取任务统计"""
        if name in self._task_monitors:
            return self._task_monitors[name].get_stats()
        return None
    
    def get_all_task_stats(self) -> Dict[str, LossStats]:
        """获取所有任务统计"""
        return {
            name: monitor.get_stats()
            for name, monitor in self._task_monitors.items()
        }
    
    def get_weight_history(self, n: int = 10) -> List[Dict[str, float]]:
        """获取权重历史"""
        return self._weight_monitor.get_recent_weights(n)
    
    def get_balance_stats(self) -> Dict[str, Dict[str, float]]:
        """获取平衡统计"""
        return self._balancer.get_task_stats()
    
    def set_auto_balance(self, enabled: bool, method: str = 'inverse') -> None:
        """设置自动平衡"""
        self._auto_balance = enabled
        self._balance_method = method
    
    def print_summary(self) -> None:
        """打印摘要"""
        print("\n" + "="*80)
        print("Multi-Task Loss Summary")
        print("="*80)
        
        print(f"\nTotal tasks: {len(self.tasks)}")
        print(f"Auto balance: {self._auto_balance}")
        if self._auto_balance:
            print(f"Balance method: {self._balance_method}")
        
        print(f"\nTask Details:")
        print(f"{'Task':<20} {'Weight':<10} {'Avg Loss':<12} {'Steps':<8}")
        print("-"*80)
        
        for name in self.tasks:
            weight = self.task_weights.get(name, 1.0)
            stats = self.get_task_stats(name)
            if stats:
                print(f"{name:<20} {weight:<10.4f} {stats.avg_loss:<12.6f} {stats.total_steps:<8}")
            else:
                print(f"{name:<20} {weight:<10.4f} {'N/A':<12} {0:<8}")
        
        print("="*80)
    
    def reset_mt_stats(self) -> None:
        """重置所有统计"""
        self._weight_monitor.reset()
        for monitor in self._task_monitors.values():
            monitor.reset()
        self._mt_step = 0


# ==================== 动态加权损失 ====================

class DynamicWeightedLoss(BaseLoss):
    """
    动态加权损失
    
    根据训练进度或损失值动态调整权重。
    """
    
    def __init__(
        self, 
        losses: Dict[str, BaseLoss],
        config: Optional[LossConfig] = None,
        warmup_steps: int = 1000,
        **kwargs
    ):
        super().__init__(config)
        
        self.losses = nn.ModuleDict(losses)
        self.warmup_steps = warmup_steps
        
        # 可学习的权重
        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in losses
        })
        
        self._dw_step = 0
        
        # 新增：监控
        self._loss_monitors: Dict[str, LossMonitor] = {
            name: LossMonitor() for name in losses
        }
        self._weight_monitor = WeightMonitor()
    
    def step(self):
        """更新步数"""
        self._dw_step += 1
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        weights = {}
        for name in self.losses:
            w = torch.exp(-self.log_weights[name]).item()
            weights[name] = w
        return weights
    
    def forward(
        self, 
        predictions: Dict[str, Tensor], 
        targets: Dict[str, Tensor],
        **kwargs
    ) -> Tensor:
        """计算动态加权损失"""
        total_loss = torch.tensor(0.0)
        
        for name, loss_fn in self.losses.items():
            if name not in predictions or name not in targets:
                continue
            
            try:
                loss = loss_fn(predictions[name], targets[name])
                
                # 动态权重
                precision = torch.exp(-self.log_weights[name])
                weighted_loss = precision * loss + self.log_weights[name]
                
                if total_loss.device != weighted_loss.device:
                    total_loss = total_loss.to(weighted_loss.device)
                total_loss = total_loss + weighted_loss
            except Exception as e:
                logger.warning(f"Loss {name} failed: {e}")
        
        return total_loss
    
    def compute(
        self, 
        predictions: Dict[str, Tensor], 
        targets: Dict[str, Tensor],
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        total_loss = torch.tensor(0.0)
        components = {}
        metrics = {}
        
        for name, loss_fn in self.losses.items():
            if name not in predictions or name not in targets:
                continue
            
            try:
                if hasattr(loss_fn, 'compute'):
                    result = loss_fn.compute(predictions[name], targets[name])
                    loss = result.loss
                else:
                    loss = loss_fn(predictions[name], targets[name])
                
                # 记录到监控器
                loss_result = LossResult(loss=loss, step=self._dw_step)
                self._loss_monitors[name].record(loss_result)
                
                # 动态权重
                precision = torch.exp(-self.log_weights[name])
                weighted_loss = precision * loss + self.log_weights[name]
                
                components[name] = loss
                metrics[f"{name}_loss"] = loss.item()
                metrics[f"{name}_weight"] = precision.item()
                
                if total_loss.device != weighted_loss.device:
                    total_loss = total_loss.to(weighted_loss.device)
                total_loss = total_loss + weighted_loss
            except Exception as e:
                logger.warning(f"Loss {name} failed: {e}")
        
        # 记录权重
        current_weights = self.get_weights()
        self._weight_monitor.record(current_weights, self._dw_step)
        metrics['dynamic_weights'] = current_weights
        
        self._dw_step += 1
        
        return LossResult(
            loss=total_loss * self.config.weight,
            components=components,
            metrics=metrics,
            step=self._dw_step
        )
    
    # ==================== 新增方法 ====================
    
    def get_loss_stats(self, name: str) -> Optional[LossStats]:
        """获取损失统计"""
        if name in self._loss_monitors:
            return self._loss_monitors[name].get_stats()
        return None
    
    def get_all_loss_stats(self) -> Dict[str, LossStats]:
        """获取所有损失统计"""
        return {
            name: monitor.get_stats()
            for name, monitor in self._loss_monitors.items()
        }
    
    def get_weight_history(self, n: int = 10) -> List[Dict[str, float]]:
        """获取权重历史"""
        return self._weight_monitor.get_recent_weights(n)
    
    def print_summary(self) -> None:
        """打印摘要"""
        print("\n" + "="*80)
        print("Dynamic Weighted Loss Summary")
        print("="*80)
        
        print(f"\nWarmup steps: {self.warmup_steps}")
        print(f"Current step: {self._dw_step}")
        
        weights = self.get_weights()
        
        print(f"\nLoss Details:")
        print(f"{'Name':<20} {'Weight':<12} {'Avg Loss':<12} {'Steps':<8}")
        print("-"*80)
        
        for name in self.losses:
            weight = weights.get(name, 0.0)
            stats = self.get_loss_stats(name)
            if stats:
                print(f"{name:<20} {weight:<12.6f} {stats.avg_loss:<12.6f} {stats.total_steps:<8}")
            else:
                print(f"{name:<20} {weight:<12.6f} {'N/A':<12} {0:<8}")
        
        print("="*80)
    
    def reset_dw_stats(self) -> None:
        """重置统计"""
        self._weight_monitor.reset()
        for monitor in self._loss_monitors.values():
            monitor.reset()


# ==================== 不确定性加权损失 ====================

class UncertaintyWeightedLoss(BaseLoss):
    """
    不确定性加权损失
    
    基于同方差不确定性自动学习任务权重。
    
    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    
    def __init__(
        self, 
        tasks: Dict[str, BaseLoss],
        config: Optional[LossConfig] = None,
        **kwargs
    ):
        super().__init__(config)
        
        self.tasks = nn.ModuleDict(tasks)
        
        # 可学习的log(σ^2)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1))
            for name in tasks
        })
        
        # 新增：监控
        self._task_monitors: Dict[str, LossMonitor] = {
            name: LossMonitor() for name in tasks
        }
        self._uncertainty_monitor = WeightMonitor()
        self._uw_step = 0
    
    def forward(
        self, 
        predictions: Dict[str, Tensor], 
        targets: Dict[str, Tensor],
        **kwargs
    ) -> Tensor:
        """
        计算不确定性加权损失
        
        L = Σ (1 / 2σ_i^2) * L_i + log(σ_i)
        """
        total_loss = torch.tensor(0.0)
        
        for name, loss_fn in self.tasks.items():
            if name not in predictions or name not in targets:
                continue
            
            try:
                loss = loss_fn(predictions[name], targets[name])
                
                # 不确定性加权
                precision = torch.exp(-self.log_vars[name])
                weighted_loss = precision * loss + self.log_vars[name]
                
                if total_loss.device != weighted_loss.device:
                    total_loss = total_loss.to(weighted_loss.device)
                total_loss = total_loss + weighted_loss
            except Exception as e:
                logger.warning(f"Task {name} loss failed: {e}")
        
        return total_loss
    
    def get_uncertainties(self) -> Dict[str, float]:
        """获取各任务的不确定性"""
        return {
            name: torch.exp(self.log_vars[name]).item()
            for name in self.tasks
        }
    
    def compute(
        self, 
        predictions: Dict[str, Tensor], 
        targets: Dict[str, Tensor],
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        total_loss = torch.tensor(0.0)
        components = {}
        metrics = {}
        
        for name, loss_fn in self.tasks.items():
            if name not in predictions or name not in targets:
                continue
            
            try:
                if hasattr(loss_fn, 'compute'):
                    result = loss_fn.compute(predictions[name], targets[name])
                    loss = result.loss
                else:
                    loss = loss_fn(predictions[name], targets[name])
                
                # 记录到监控器
                task_result = LossResult(loss=loss, step=self._uw_step)
                self._task_monitors[name].record(task_result)
                
                # 不确定性加权
                precision = torch.exp(-self.log_vars[name])
                weighted_loss = precision * loss + self.log_vars[name]
                
                components[name] = loss
                metrics[f"{name}_loss"] = loss.item()
                metrics[f"{name}_uncertainty"] = torch.exp(self.log_vars[name]).item()
                
                if total_loss.device != weighted_loss.device:
                    total_loss = total_loss.to(weighted_loss.device)
                total_loss = total_loss + weighted_loss
            except Exception as e:
                logger.warning(f"Task {name} loss failed: {e}")
        
        # 记录不确定性
        uncertainties = self.get_uncertainties()
        self._uncertainty_monitor.record(uncertainties, self._uw_step)
        metrics['uncertainties'] = uncertainties
        
        self._uw_step += 1
        
        return LossResult(
            loss=total_loss * self.config.weight,
            components=components,
            metrics=metrics,
            step=self._uw_step
        )
    
    # ==================== 新增方法 ====================
    
    def get_task_stats(self, name: str) -> Optional[LossStats]:
        """获取任务统计"""
        if name in self._task_monitors:
            return self._task_monitors[name].get_stats()
        return None
    
    def get_all_task_stats(self) -> Dict[str, LossStats]:
        """获取所有任务统计"""
        return {
            name: monitor.get_stats()
            for name, monitor in self._task_monitors.items()
        }
    
    def get_uncertainty_history(self, n: int = 10) -> List[Dict[str, float]]:
        """获取不确定性历史"""
        return self._uncertainty_monitor.get_recent_weights(n)
    
    def print_summary(self) -> None:
        """打印摘要"""
        print("\n" + "="*80)
        print("Uncertainty Weighted Loss Summary")
        print("="*80)
        
        uncertainties = self.get_uncertainties()
        
        print(f"\nTask Details:")
        print(f"{'Task':<20} {'Uncertainty':<15} {'Avg Loss':<12} {'Steps':<8}")
        print("-"*80)
        
        for name in self.tasks:
            unc = uncertainties.get(name, 0.0)
            stats = self.get_task_stats(name)
            if stats:
                print(f"{name:<20} {unc:<15.6f} {stats.avg_loss:<12.6f} {stats.total_steps:<8}")
            else:
                print(f"{name:<20} {unc:<15.6f} {'N/A':<12} {0:<8}")
        
        print("="*80)
    
    def reset_uw_stats(self) -> None:
        """重置统计"""
        self._uncertainty_monitor.reset()
        for monitor in self._task_monitors.values():
            monitor.reset()
        self._uw_step = 0


# ==================== GradNorm损失 ====================

class GradNormLoss(BaseLoss):
    """
    GradNorm损失
    
    通过梯度归一化平衡多任务学习。
    
    Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing"
    """
    
    def __init__(
        self, 
        tasks: Dict[str, BaseLoss],
        alpha: float = 1.5,
        config: Optional[LossConfig] = None,
        **kwargs
    ):
        super().__init__(config)
        
        self.tasks = nn.ModuleDict(tasks)
        self.alpha = alpha
        
        # 可学习的任务权重
        self.task_weights = nn.ParameterDict({
            name: nn.Parameter(torch.ones(1))
            for name in tasks
        })
        
        # 初始损失（用于计算相对训练速度）
        self._initial_losses: Dict[str, float] = {}
        self._current_losses: Dict[str, float] = {}
        
        # 新增：监控
        self._task_monitors: Dict[str, LossMonitor] = {
            name: LossMonitor() for name in tasks
        }
        self._weight_monitor = WeightMonitor()
        self._gn_step = 0
    
    def forward(
        self, 
        predictions: Dict[str, Tensor], 
        targets: Dict[str, Tensor],
        **kwargs
    ) -> Tensor:
        """计算GradNorm加权损失"""
        total_loss = torch.tensor(0.0)
        
        for name, loss_fn in self.tasks.items():
            if name not in predictions or name not in targets:
                continue
            
            try:
                loss = loss_fn(predictions[name], targets[name])
                
                # 记录损失
                self._current_losses[name] = loss.item()
                if name not in self._initial_losses:
                    self._initial_losses[name] = loss.item()
                
                # 应用权重
                weighted_loss = self.task_weights[name] * loss
                
                if total_loss.device != weighted_loss.device:
                    total_loss = total_loss.to(weighted_loss.device)
                total_loss = total_loss + weighted_loss
            except Exception as e:
                logger.warning(f"Task {name} loss failed: {e}")
        
        return total_loss
    
    def compute_gradnorm_loss(
        self, 
        shared_params: List[nn.Parameter]
    ) -> Tensor:
        """
        计算GradNorm损失（用于更新任务权重）
        
        应在backward前调用。
        """
        # 计算相对逆训练速度
        r_i = {}
        total_r = 0.0
        
        for name in self.tasks:
            if name in self._initial_losses and name in self._current_losses:
                r = self._current_losses[name] / (self._initial_losses[name] + 1e-8)
                r_i[name] = r
                total_r += r
        
        if not r_i:
            return torch.tensor(0.0)
        
        avg_r = total_r / len(r_i)
        
        # 计算目标梯度范数
        gradnorm_loss = torch.tensor(0.0)
        
        for name, r in r_i.items():
            target_grad_norm = avg_r * (r / avg_r) ** self.alpha
            
            # 实际梯度范数需要在具体实现中计算
            # 这里只是框架
            w = self.task_weights[name]
            gradnorm_loss = gradnorm_loss + torch.abs(w - target_grad_norm)
        
        return gradnorm_loss
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前任务权重"""
        return {
            name: self.task_weights[name].item()
            for name in self.tasks
        }
    
    def compute(
        self, 
        predictions: Dict[str, Tensor], 
        targets: Dict[str, Tensor],
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        total_loss = torch.tensor(0.0)
        components = {}
        metrics = {}
        
        for name, loss_fn in self.tasks.items():
            if name not in predictions or name not in targets:
                continue
            
            try:
                if hasattr(loss_fn, 'compute'):
                    result = loss_fn.compute(predictions[name], targets[name])
                    loss = result.loss
                else:
                    loss = loss_fn(predictions[name], targets[name])
                
                # 记录损失
                self._current_losses[name] = loss.item()
                if name not in self._initial_losses:
                    self._initial_losses[name] = loss.item()
                
                # 记录到监控器
                task_result = LossResult(loss=loss, step=self._gn_step)
                self._task_monitors[name].record(task_result)
                
                # 应用权重
                weighted_loss = self.task_weights[name] * loss
                
                components[name] = loss
                metrics[f"{name}_loss"] = loss.item()
                metrics[f"{name}_weight"] = self.task_weights[name].item()
                
                if total_loss.device != weighted_loss.device:
                    total_loss = total_loss.to(weighted_loss.device)
                total_loss = total_loss + weighted_loss
            except Exception as e:
                logger.warning(f"Task {name} loss failed: {e}")
        
        # 记录权重
        current_weights = self.get_weights()
        self._weight_monitor.record(current_weights, self._gn_step)
        metrics['gradnorm_weights'] = current_weights
        
        self._gn_step += 1
        
        return LossResult(
            loss=total_loss * self.config.weight,
            components=components,
            metrics=metrics,
            step=self._gn_step
        )
    
    # ==================== 新增方法 ====================
    
    def get_task_stats(self, name: str) -> Optional[LossStats]:
        """获取任务统计"""
        if name in self._task_monitors:
            return self._task_monitors[name].get_stats()
        return None
    
    def get_all_task_stats(self) -> Dict[str, LossStats]:
        """获取所有任务统计"""
        return {
            name: monitor.get_stats()
            for name, monitor in self._task_monitors.items()
        }
    
    def get_weight_history(self, n: int = 10) -> List[Dict[str, float]]:
        """获取权重历史"""
        return self._weight_monitor.get_recent_weights(n)
    
    def get_training_ratios(self) -> Dict[str, float]:
        """获取训练速度比率"""
        ratios = {}
        for name in self.tasks:
            if name in self._initial_losses and name in self._current_losses:
                ratios[name] = self._current_losses[name] / (self._initial_losses[name] + 1e-8)
        return ratios
    
    def print_summary(self) -> None:
        """打印摘要"""
        print("\n" + "="*80)
        print("GradNorm Loss Summary")
        print("="*80)
        
        print(f"\nAlpha: {self.alpha}")
        
        weights = self.get_weights()
        ratios = self.get_training_ratios()
        
        print(f"\nTask Details:")
        print(f"{'Task':<20} {'Weight':<12} {'Train Ratio':<12} {'Avg Loss':<12}")
        print("-"*80)
        
        for name in self.tasks:
            weight = weights.get(name, 1.0)
            ratio = ratios.get(name, 1.0)
            stats = self.get_task_stats(name)
            avg_loss = stats.avg_loss if stats else 0.0
            print(f"{name:<20} {weight:<12.6f} {ratio:<12.6f} {avg_loss:<12.6f}")
        
        print("="*80)
    
    def reset_gn_stats(self) -> None:
        """重置统计"""
        self._weight_monitor.reset()
        for monitor in self._task_monitors.values():
            monitor.reset()
        self._initial_losses.clear()
        self._current_losses.clear()
        self._gn_step = 0


# ==================== 工具函数 ====================

def create_composite_loss(
    losses: List[Tuple[str, BaseLoss, float]],
    auto_balance: bool = False,
    balance_method: str = 'inverse'
) -> CompositeLoss:
    """
    创建复合损失
    
    Args:
        losses: 损失列表 [(name, loss_fn, weight), ...]
        auto_balance: 是否自动平衡
        balance_method: 平衡方法
        
    Returns:
        CompositeLoss实例
    """
    return CompositeLoss(
        losses=losses,
        auto_balance=auto_balance,
        balance_method=balance_method
    )


def create_multitask_loss(
    tasks: Dict[str, BaseLoss],
    task_weights: Optional[Dict[str, float]] = None,
    auto_balance: bool = False
) -> MultiTaskLoss:
    """
    创建多任务损失
    
    Args:
        tasks: 任务字典
        task_weights: 任务权重
        auto_balance: 是否自动平衡
        
    Returns:
        MultiTaskLoss实例
    """
    return MultiTaskLoss(
        tasks=tasks,
        task_weights=task_weights,
        auto_balance=auto_balance
    )


def compare_composite_losses(losses: Dict[str, Union[CompositeLoss, MultiTaskLoss]]) -> None:
    """
    对比复合损失
    
    Args:
        losses: 损失字典
    """
    print("\n" + "="*100)
    print("Composite Loss Comparison")
    print("="*100)
    
    print(f"\n{'Name':<20} {'Type':<25} {'Components':<15} {'Auto Balance':<15} {'Steps':<8}")
    print("-"*100)
    
    for name, loss_fn in losses.items():
        loss_type = loss_fn.__class__.__name__
        
        if isinstance(loss_fn, CompositeLoss):
            num_components = len(loss_fn.loss_names)
            auto_balance = loss_fn._auto_balance
            steps = loss_fn._comp_step
        elif isinstance(loss_fn, MultiTaskLoss):
            num_components = len(loss_fn.tasks)
            auto_balance = loss_fn._auto_balance
            steps = loss_fn._mt_step
        else:
            num_components = 0
            auto_balance = False
            steps = 0
        
        print(f"{name:<20} {loss_type:<25} {num_components:<15} {str(auto_balance):<15} {steps:<8}")
    
    print("="*100)


def analyze_weight_evolution(
    loss_fn: Union[CompositeLoss, MultiTaskLoss, DynamicWeightedLoss],
    window: int = 100
) -> Dict[str, Any]:
    """
    分析权重演化
    
    Args:
        loss_fn: 损失函数
        window: 窗口大小
        
    Returns:
        分析结果
    """
    if not hasattr(loss_fn, 'get_weight_history'):
        return {}
    
    history = loss_fn.get_weight_history(window)
    
    if not history:
        return {}
    
    # 计算权重统计
    weight_stats = defaultdict(lambda: {'values': [], 'mean': 0.0, 'std': 0.0, 'trend': 0.0})
    
    for weights in history:
        for name, weight in weights.items():
            weight_stats[name]['values'].append(weight)
    
    import math
    for name, stats in weight_stats.items():
        values = stats['values']
        if values:
            stats['mean'] = sum(values) / len(values)
            
            # 计算标准差
            if len(values) > 1:
                variance = sum((v - stats['mean']) ** 2 for v in values) / len(values)
                stats['std'] = math.sqrt(variance)
            
            # 计算趋势（第一半vs第二半）
            if len(values) >= 4:
                mid = len(values) // 2
                first_half = sum(values[:mid]) / mid
                second_half = sum(values[mid:]) / (len(values) - mid)
                stats['trend'] = second_half - first_half
            
            # 移除原始值（太大）
            del stats['values']
    
    return dict(weight_stats)


def print_weight_evolution(
    loss_fn: Union[CompositeLoss, MultiTaskLoss, DynamicWeightedLoss],
    window: int = 100
) -> None:
    """
    打印权重演化
    
    Args:
        loss_fn: 损失函数
        window: 窗口大小
    """
    stats = analyze_weight_evolution(loss_fn, window)
    
    if not stats:
        print("No weight history available")
        return
    
    print("\n" + "="*80)
    print("Weight Evolution Analysis")
    print("="*80)
    
    print(f"\n{'Component':<20} {'Mean':<12} {'Std':<12} {'Trend':<12}")
    print("-"*80)
    
    for name, stat in stats.items():
        print(f"{name:<20} {stat['mean']:<12.6f} {stat['std']:<12.6f} {stat['trend']:+<12.6f}")
    
    print("="*80)


def balance_weights_manually(
    loss_fn: Union[CompositeLoss, MultiTaskLoss],
    method: str = 'inverse'
) -> Dict[str, float]:
    """
    手动平衡权重
    
    Args:
        loss_fn: 损失函数
        method: 平衡方法
        
    Returns:
        平衡后的权重
    """
    if isinstance(loss_fn, CompositeLoss):
        balancer = loss_fn._balancer
    elif isinstance(loss_fn, MultiTaskLoss):
        balancer = loss_fn._balancer
    else:
        return {}
    
    return balancer.compute_balanced_weights(method)


def export_composite_config(loss_fn: CompositeLoss) -> Dict[str, Any]:
    """
    导出复合损失配置
    
    Args:
        loss_fn: 复合损失函数
        
    Returns:
        配置字典
    """
    return {
        'type': 'composite',
        'components': [
            {
                'name': name,
                'weight': weight,
                'type': loss_fn.loss_fns[i].__class__.__name__
            }
            for i, (name, weight) in enumerate(zip(loss_fn.loss_names, loss_fn.loss_weights))
        ],
        'auto_balance': loss_fn._auto_balance,
        'balance_method': loss_fn._balance_method,
        'step': loss_fn._comp_step
    }


def export_multitask_config(loss_fn: MultiTaskLoss) -> Dict[str, Any]:
    """
    导出多任务损失配置
    
    Args:
        loss_fn: 多任务损失函数
        
    Returns:
        配置字典
    """
    return {
        'type': 'multitask',
        'tasks': {
            name: {
                'weight': loss_fn.task_weights.get(name, 1.0),
                'type': loss_fn.tasks[name].__class__.__name__
            }
            for name in loss_fn.tasks
        },
        'auto_balance': loss_fn._auto_balance,
        'balance_method': loss_fn._balance_method,
        'step': loss_fn._mt_step
    }




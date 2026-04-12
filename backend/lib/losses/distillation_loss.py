# -*- coding: utf-8 -*-
"""
知识蒸馏损失函数

包含各种知识蒸馏相关的损失函数。
"""

import logging
import time
import math
from typing import Optional, Dict, Any, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_loss import (
    BaseLoss, LossConfig, LossResult, LossType, LossMonitor, LossStats,
    register_loss, reduce_loss, weighted_loss
)

logger = logging.getLogger(__name__)


# ==================== 监控和统计组件 ====================

@dataclass
class DistillationStats:
    """蒸馏统计"""
    total_steps: int = 0
    avg_kd_loss: float = 0.0
    avg_ce_loss: float = 0.0
    avg_feature_loss: float = 0.0
    avg_attention_loss: float = 0.0
    avg_student_accuracy: float = 0.0
    avg_teacher_accuracy: float = 0.0
    avg_temperature: float = 4.0
    avg_kl_divergence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'avg_kd_loss': self.avg_kd_loss,
            'avg_ce_loss': self.avg_ce_loss,
            'avg_feature_loss': self.avg_feature_loss,
            'avg_attention_loss': self.avg_attention_loss,
            'avg_student_accuracy': self.avg_student_accuracy,
            'avg_teacher_accuracy': self.avg_teacher_accuracy,
            'avg_temperature': self.avg_temperature,
            'avg_kl_divergence': self.avg_kl_divergence,
        }


class DistillationMonitor:
    """蒸馏监控器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._history: List[Dict[str, float]] = []
        self._stats = DistillationStats()
        
        # 累计统计
        self._totals: Dict[str, float] = defaultdict(float)
    
    def record(
        self,
        kd_loss: float = 0.0,
        ce_loss: float = 0.0,
        feature_loss: float = 0.0,
        attention_loss: float = 0.0,
        student_accuracy: float = 0.0,
        teacher_accuracy: float = 0.0,
        temperature: float = 4.0,
        kl_divergence: float = 0.0,
        **kwargs
    ) -> None:
        """记录统计"""
        record = {
            'kd_loss': kd_loss,
            'ce_loss': ce_loss,
            'feature_loss': feature_loss,
            'attention_loss': attention_loss,
            'student_accuracy': student_accuracy,
            'teacher_accuracy': teacher_accuracy,
            'temperature': temperature,
            'kl_divergence': kl_divergence,
            'timestamp': time.time(),
            **kwargs
        }
        
        self._history.append(record)
        if len(self._history) > self.max_history:
            self._history.pop(0)
        
        # 更新累计统计
        self._stats.total_steps += 1
        n = self._stats.total_steps
        
        self._totals['kd_loss'] += kd_loss
        self._totals['ce_loss'] += ce_loss
        self._totals['feature_loss'] += feature_loss
        self._totals['attention_loss'] += attention_loss
        self._totals['student_accuracy'] += student_accuracy
        self._totals['teacher_accuracy'] += teacher_accuracy
        self._totals['temperature'] += temperature
        self._totals['kl_divergence'] += kl_divergence
        
        # 更新平均
        self._stats.avg_kd_loss = self._totals['kd_loss'] / n
        self._stats.avg_ce_loss = self._totals['ce_loss'] / n
        self._stats.avg_feature_loss = self._totals['feature_loss'] / n
        self._stats.avg_attention_loss = self._totals['attention_loss'] / n
        self._stats.avg_student_accuracy = self._totals['student_accuracy'] / n
        self._stats.avg_teacher_accuracy = self._totals['teacher_accuracy'] / n
        self._stats.avg_temperature = self._totals['temperature'] / n
        self._stats.avg_kl_divergence = self._totals['kl_divergence'] / n
    
    def get_stats(self) -> DistillationStats:
        """获取统计"""
        return self._stats
    
    def get_recent(self, n: int = 10) -> List[Dict[str, float]]:
        """获取最近的记录"""
        return self._history[-n:]
    
    def get_accuracy_gap(self) -> float:
        """获取学生和教师准确率差距"""
        return self._stats.avg_teacher_accuracy - self._stats.avg_student_accuracy
    
    def is_distillation_effective(self, min_gap_reduction: float = 0.1) -> bool:
        """
        检查蒸馏是否有效
        
        通过比较最近准确率差距与历史差距来判断
        """
        if len(self._history) < 20:
            return True
        
        # 比较最近10步和之前10步
        recent = self._history[-10:]
        earlier = self._history[-20:-10]
        
        recent_gap = sum(r.get('teacher_accuracy', 0) - r.get('student_accuracy', 0) for r in recent) / 10
        earlier_gap = sum(r.get('teacher_accuracy', 0) - r.get('student_accuracy', 0) for r in earlier) / 10
        
        # 如果差距在缩小，蒸馏是有效的
        return recent_gap < earlier_gap - min_gap_reduction or recent_gap < 0.05
    
    def reset(self) -> None:
        """重置"""
        self._history.clear()
        self._stats = DistillationStats()
        self._totals.clear()


class TemperatureScheduler:
    """温度调度器"""
    
    def __init__(
        self,
        initial_temp: float = 4.0,
        min_temp: float = 1.0,
        max_temp: float = 20.0,
        schedule: str = 'constant',  # constant, linear, cosine, adaptive
        warmup_steps: int = 0
    ):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        
        self._step = 0
        self._total_steps = 10000
        
        # 自适应调度的历史
        self._kl_history: List[float] = []
    
    def step(self, kl_divergence: Optional[float] = None) -> float:
        """获取当前温度并更新步数"""
        self._step += 1
        
        if kl_divergence is not None:
            self._kl_history.append(kl_divergence)
            if len(self._kl_history) > 100:
                self._kl_history.pop(0)
        
        return self.get_temperature()
    
    def get_temperature(self) -> float:
        """获取当前温度"""
        # 预热阶段
        if self._step < self.warmup_steps:
            progress = self._step / max(self.warmup_steps, 1)
            return self.min_temp + (self.initial_temp - self.min_temp) * progress
        
        effective_step = self._step - self.warmup_steps
        effective_total = self._total_steps - self.warmup_steps
        
        if self.schedule == 'constant':
            return self.initial_temp
        
        elif self.schedule == 'linear':
            # 线性衰减温度
            progress = min(effective_step / effective_total, 1.0)
            return self.initial_temp - (self.initial_temp - self.min_temp) * progress
        
        elif self.schedule == 'cosine':
            # 余弦退火
            progress = min(effective_step / effective_total, 1.0)
            return self.min_temp + (self.initial_temp - self.min_temp) * (1 + math.cos(progress * math.pi)) / 2
        
        elif self.schedule == 'adaptive':
            # 自适应：根据KL散度调整
            if len(self._kl_history) < 10:
                return self.initial_temp
            
            recent_kl = sum(self._kl_history[-10:]) / 10
            
            # KL散度高时提高温度（软化分布）
            # KL散度低时降低温度（锐化分布）
            if recent_kl > 1.0:
                target_temp = min(self.initial_temp * 1.5, self.max_temp)
            elif recent_kl < 0.1:
                target_temp = max(self.initial_temp * 0.5, self.min_temp)
            else:
                target_temp = self.initial_temp
            
            # 平滑过渡
            current = getattr(self, '_current_temp', self.initial_temp)
            self._current_temp = 0.9 * current + 0.1 * target_temp
            return self._current_temp
        
        return self.initial_temp
    
    def set_total_steps(self, total_steps: int) -> None:
        """设置总步数"""
        self._total_steps = total_steps
    
    def reset(self) -> None:
        """重置"""
        self._step = 0
        self._kl_history.clear()
        if hasattr(self, '_current_temp'):
            delattr(self, '_current_temp')


class LayerWeightManager:
    """层级权重管理器"""
    
    def __init__(
        self,
        num_layers: int,
        strategy: str = 'uniform',  # uniform, linear, exponential, adaptive
        importance_weights: Optional[List[float]] = None
    ):
        self.num_layers = num_layers
        self.strategy = strategy
        
        if importance_weights is not None:
            self._weights = importance_weights
        else:
            self._weights = self._compute_initial_weights()
        
        # 自适应权重的损失历史
        self._layer_losses: Dict[int, List[float]] = defaultdict(list)
    
    def _compute_initial_weights(self) -> List[float]:
        """计算初始权重"""
        if self.strategy == 'uniform':
            return [1.0 / self.num_layers] * self.num_layers
        
        elif self.strategy == 'linear':
            # 后面的层权重更大
            weights = [i + 1 for i in range(self.num_layers)]
            total = sum(weights)
            return [w / total for w in weights]
        
        elif self.strategy == 'exponential':
            # 指数增长
            weights = [2 ** i for i in range(self.num_layers)]
            total = sum(weights)
            return [w / total for w in weights]
        
        else:
            return [1.0 / self.num_layers] * self.num_layers
    
    def record_layer_loss(self, layer_idx: int, loss: float) -> None:
        """记录层损失"""
        self._layer_losses[layer_idx].append(loss)
        
        # 限制历史长度
        if len(self._layer_losses[layer_idx]) > 100:
            self._layer_losses[layer_idx].pop(0)
    
    def update_weights(self) -> List[float]:
        """更新自适应权重"""
        if self.strategy != 'adaptive':
            return self._weights
        
        # 根据层损失调整权重（损失大的层权重增加）
        if not self._layer_losses:
            return self._weights
        
        avg_losses = []
        for i in range(self.num_layers):
            if self._layer_losses[i]:
                avg_losses.append(sum(self._layer_losses[i]) / len(self._layer_losses[i]))
            else:
                avg_losses.append(1.0)
        
        # 归一化
        total = sum(avg_losses) + 1e-8
        self._weights = [loss / total for loss in avg_losses]
        
        return self._weights
    
    def get_weight(self, layer_idx: int) -> float:
        """获取层权重"""
        if 0 <= layer_idx < len(self._weights):
            return self._weights[layer_idx]
        return 1.0 / self.num_layers
    
    def get_all_weights(self) -> List[float]:
        """获取所有权重"""
        return self._weights.copy()
    
    def reset(self) -> None:
        """重置"""
        self._weights = self._compute_initial_weights()
        self._layer_losses.clear()




class DistillationLossModule(BaseLoss):
    """
    蒸馏损失基类
    
    提供蒸馏损失的通用功能。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        temperature: float = 4.0,
        temperature_schedule: str = 'constant',
        warmup_steps: int = 0,
        **kwargs
    ):
        super().__init__(config)
        self.temperature = config.temperature if config else temperature
        
        # 新增：温度调度器
        self._temp_scheduler = TemperatureScheduler(
            initial_temp=self.temperature,
            schedule=temperature_schedule,
            warmup_steps=warmup_steps
        )
        
        # 新增：监控器
        self._distillation_monitor = DistillationMonitor()
        self._dl_step = 0
    
    def get_effective_temperature(self) -> float:
        """获取有效温度"""
        return self._temp_scheduler.get_temperature()
    
    def set_temperature_schedule(
        self,
        schedule: str,
        total_steps: int = 10000,
        warmup_steps: int = 0
    ) -> None:
        """设置温度调度"""
        self._temp_scheduler.schedule = schedule
        self._temp_scheduler.set_total_steps(total_steps)
        self._temp_scheduler.warmup_steps = warmup_steps
    
    def _compute_kl_divergence(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        temperature: float
    ) -> float:
        """计算KL散度"""
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        
        kl = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        return kl.item()
    
    def _compute_accuracy(self, logits: Tensor, targets: Tensor) -> float:
        """计算准确率"""
        preds = logits.argmax(dim=-1)
        return (preds == targets).float().mean().item()
    
    def _compute_agreement(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor
    ) -> float:
        """计算学生和教师预测一致率"""
        student_preds = student_logits.argmax(dim=-1)
        teacher_preds = teacher_logits.argmax(dim=-1)
        return (student_preds == teacher_preds).float().mean().item()
    
    def _record_distillation_stats(
        self,
        kd_loss: float = 0.0,
        ce_loss: float = 0.0,
        feature_loss: float = 0.0,
        attention_loss: float = 0.0,
        student_accuracy: float = 0.0,
        teacher_accuracy: float = 0.0,
        kl_divergence: float = 0.0,
        **kwargs
    ) -> None:
        """记录蒸馏统计"""
        temp = self.get_effective_temperature()
        self._distillation_monitor.record(
            kd_loss=kd_loss,
            ce_loss=ce_loss,
            feature_loss=feature_loss,
            attention_loss=attention_loss,
            student_accuracy=student_accuracy,
            teacher_accuracy=teacher_accuracy,
            temperature=temp,
            kl_divergence=kl_divergence,
            **kwargs
        )
        self._dl_step += 1
    
    def get_distillation_stats(self) -> DistillationStats:
        """获取蒸馏统计"""
        return self._distillation_monitor.get_stats()
    
    def get_accuracy_gap(self) -> float:
        """获取准确率差距"""
        return self._distillation_monitor.get_accuracy_gap()
    
    def is_distillation_effective(self) -> bool:
        """检查蒸馏是否有效"""
        return self._distillation_monitor.is_distillation_effective()
    
    def print_summary(self) -> None:
        """打印摘要"""
        stats = self.get_distillation_stats()
        
        print("\n" + "="*80)
        print(f"Distillation Loss Summary: {self.__class__.__name__}")
        print("="*80)
        
        print(f"\nTemperature: {self.get_effective_temperature():.2f}")
        print(f"Temperature schedule: {self._temp_scheduler.schedule}")
        
        print(f"\nStatistics (over {stats.total_steps} steps):")
        print(f"  Avg KD loss: {stats.avg_kd_loss:.6f}")
        if stats.avg_ce_loss > 0:
            print(f"  Avg CE loss: {stats.avg_ce_loss:.6f}")
        if stats.avg_feature_loss > 0:
            print(f"  Avg Feature loss: {stats.avg_feature_loss:.6f}")
        if stats.avg_attention_loss > 0:
            print(f"  Avg Attention loss: {stats.avg_attention_loss:.6f}")
        
        print(f"\nAccuracy:")
        print(f"  Student: {stats.avg_student_accuracy:.4f}")
        print(f"  Teacher: {stats.avg_teacher_accuracy:.4f}")
        print(f"  Gap: {self.get_accuracy_gap():.4f}")
        
        print(f"\nKL Divergence: {stats.avg_kl_divergence:.6f}")
        print(f"Distillation effective: {self.is_distillation_effective()}")
        
        print("="*80)
    
    def reset_distillation_stats(self) -> None:
        """重置统计"""
        self._distillation_monitor.reset()
        self._temp_scheduler.reset()
        self._dl_step = 0


# ==================== 软标签蒸馏 ====================

@register_loss("soft_label")
class SoftLabelLoss(DistillationLossModule):
    """
    软标签蒸馏损失
    
    使用教师模型的软化输出作为目标。
    L = T^2 * KL(softmax(s/T) || softmax(t/T))
    
    Reference: Hinton et al., "Distilling the Knowledge in a Neural Network"
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        temperature: float = 4.0,
        **kwargs
    ):
        super().__init__(config, temperature)
    
    def forward(
        self, 
        predictions: Tensor,  # student logits
        targets: Tensor,      # teacher logits
        hard_targets: Optional[Tensor] = None,
        hard_loss_weight: float = 0.0,
        **kwargs
    ) -> Tensor:
        """
        计算软标签蒸馏损失
        
        Args:
            predictions: [N, C] 学生logits
            targets: [N, C] 教师logits
            hard_targets: [N] 硬标签（可选）
            hard_loss_weight: 硬标签损失权重
        """
        T = self.temperature
        
        # 软化学生输出
        soft_student = F.log_softmax(predictions / T, dim=-1)
        # 软化教师输出
        soft_teacher = F.softmax(targets / T, dim=-1)
        
        # KL散度损失
        kd_loss = F.kl_div(
            soft_student, 
            soft_teacher, 
            reduction='batchmean'
        ) * (T ** 2)
        
        # 如果有硬标签，添加交叉熵损失
        if hard_targets is not None and hard_loss_weight > 0:
            ce_loss = F.cross_entropy(predictions, hard_targets)
            total_loss = (1 - hard_loss_weight) * kd_loss + hard_loss_weight * ce_loss
        else:
            total_loss = kd_loss
        
        return total_loss
    
    def compute(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        hard_targets: Optional[Tensor] = None,
        hard_loss_weight: float = 0.0,
        **kwargs
    ) -> LossResult:
        """计算并返回结构化结果"""
        T = self.get_effective_temperature()
        
        soft_student = F.log_softmax(predictions / T, dim=-1)
        soft_teacher = F.softmax(targets / T, dim=-1)
        kd_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
        
        components = {'kd_loss': kd_loss}
        metrics = {'kd_loss': kd_loss.item(), 'temperature': T}
        
        total_loss = kd_loss
        ce_loss_val = 0.0
        
        if hard_targets is not None and hard_loss_weight > 0:
            ce_loss = F.cross_entropy(predictions, hard_targets)
            components['ce_loss'] = ce_loss
            metrics['ce_loss'] = ce_loss.item()
            ce_loss_val = ce_loss.item()
            total_loss = (1 - hard_loss_weight) * kd_loss + hard_loss_weight * ce_loss
            
            # 计算准确率
            student_acc = self._compute_accuracy(predictions, hard_targets)
            teacher_acc = self._compute_accuracy(targets, hard_targets)
            metrics['student_accuracy'] = student_acc
            metrics['teacher_accuracy'] = teacher_acc
            metrics['accuracy_gap'] = teacher_acc - student_acc
        else:
            student_acc = 0.0
            teacher_acc = 0.0
        
        # 计算学生-教师一致率
        agreement = self._compute_agreement(predictions, targets)
        metrics['agreement'] = agreement
        
        # 计算KL散度（用于监控）
        kl_div = self._compute_kl_divergence(predictions, targets, T)
        metrics['kl_divergence'] = kl_div
        
        # 记录统计
        self._record_distillation_stats(
            kd_loss=kd_loss.item(),
            ce_loss=ce_loss_val,
            student_accuracy=student_acc,
            teacher_accuracy=teacher_acc,
            kl_divergence=kl_div
        )
        
        # 更新温度调度
        self._temp_scheduler.step(kl_div)
        
        return LossResult(
            loss=total_loss * self.config.weight,
            components=components,
            metrics=metrics,
            step=self._dl_step
        )


# ==================== 特征蒸馏 ====================

@register_loss("feature_kd")
class FeatureDistillationLoss(DistillationLossModule):
    """
    特征蒸馏损失
    
    匹配学生和教师的中间层特征。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        loss_type: str = "mse",  # mse, cosine, l1, huber, kl
        projector_hidden_dim: int = 256,
        layer_weight_strategy: str = 'uniform',  # uniform, linear, exponential, adaptive
        **kwargs
    ):
        super().__init__(config)
        self.loss_type = loss_type
        self.projector_hidden_dim = projector_hidden_dim
        self.layer_weight_strategy = layer_weight_strategy
        
        # 特征投影器（用于维度不匹配时）
        self.projectors: nn.ModuleDict = nn.ModuleDict()
        
        # 新增：层级权重管理器（延迟初始化）
        self._layer_weight_manager: Optional[LayerWeightManager] = None
        
        # 新增：层级损失历史
        self._layer_losses: Dict[int, List[float]] = defaultdict(list)
        self._layer_similarities: Dict[int, List[float]] = defaultdict(list)
    
    def _init_layer_weights(self, num_layers: int) -> None:
        """初始化层权重"""
        if self._layer_weight_manager is None or self._layer_weight_manager.num_layers != num_layers:
            self._layer_weight_manager = LayerWeightManager(
                num_layers=num_layers,
                strategy=self.layer_weight_strategy
            )
    
    def _compute_feature_similarity(
        self, 
        student_feat: Tensor, 
        teacher_feat: Tensor
    ) -> float:
        """计算特征相似度"""
        s_flat = F.normalize(student_feat.flatten(1), dim=-1)
        t_flat = F.normalize(teacher_feat.flatten(1), dim=-1)
        return torch.cosine_similarity(s_flat, t_flat, dim=-1).mean().item()
    
    def _get_projector(
        self, 
        student_dim: int, 
        teacher_dim: int, 
        layer_id: str
    ) -> nn.Module:
        """获取或创建特征投影器"""
        key = f"{layer_id}_{student_dim}_{teacher_dim}"
        
        if key not in self.projectors:
            if student_dim != teacher_dim:
                self.projectors[key] = nn.Sequential(
                    nn.Linear(student_dim, self.projector_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.projector_hidden_dim, teacher_dim)
                )
            else:
                self.projectors[key] = nn.Identity()
        
        return self.projectors[key]
    
    def _compute_layer_loss(
        self, 
        student_feat: Tensor, 
        teacher_feat: Tensor
    ) -> Tensor:
        """计算单层特征损失"""
        if self.loss_type == "mse":
            return F.mse_loss(student_feat, teacher_feat)
        elif self.loss_type == "cosine":
            s_flat = student_feat.flatten(1)
            t_flat = teacher_feat.flatten(1)
            return 1 - torch.cosine_similarity(s_flat, t_flat, dim=-1).mean()
        elif self.loss_type == "l1":
            return F.l1_loss(student_feat, teacher_feat)
        elif self.loss_type == "huber":
            return F.smooth_l1_loss(student_feat, teacher_feat)
        elif self.loss_type == "kl":
            s_log = F.log_softmax(student_feat.flatten(1), dim=-1)
            t_soft = F.softmax(teacher_feat.flatten(1), dim=-1)
            return F.kl_div(s_log, t_soft, reduction='batchmean')
        else:
            return F.mse_loss(student_feat, teacher_feat)
    
    def forward(
        self, 
        predictions: Tuple[Tensor, ...],  # student features
        targets: Tuple[Tensor, ...],      # teacher features
        layer_indices: Optional[List[int]] = None,
        **kwargs
    ) -> Tensor:
        """
        计算特征蒸馏损失
        
        Args:
            predictions: 学生特征元组
            targets: 教师特征元组
            layer_indices: 要蒸馏的层索引
        """
        if not predictions or not targets:
            return torch.tensor(0.0)
        
        # 确定要蒸馏的层
        if layer_indices is None:
            # 默认蒸馏所有层
            layer_indices = list(range(min(len(predictions), len(targets))))
        
        total_loss = 0.0
        count = 0
        
        for idx in layer_indices:
            if idx >= len(predictions) or idx >= len(targets):
                continue
            
            s_feat = predictions[idx]
            t_feat = targets[idx]
            
            # 维度对齐
            if s_feat.shape[-1] != t_feat.shape[-1]:
                projector = self._get_projector(
                    s_feat.shape[-1], 
                    t_feat.shape[-1],
                    f"layer_{idx}"
                )
                s_feat = projector(s_feat)
            
            # 计算损失
            try:
                layer_loss = self._compute_layer_loss(s_feat, t_feat.detach())
                total_loss = total_loss + layer_loss
                count += 1
            except Exception as e:
                logger.warning(f"Layer {idx} feature loss failed: {e}")
        
        return total_loss / max(count, 1)
    
    def compute(
        self, 
        predictions: Tuple[Tensor, ...],
        targets: Tuple[Tensor, ...],
        layer_indices: Optional[List[int]] = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        if not predictions or not targets:
            return LossResult(
                loss=torch.tensor(0.0),
                components={},
                metrics={'feature_loss': 0.0},
                step=self._dl_step
            )
        
        # 确定要蒸馏的层
        if layer_indices is None:
            layer_indices = list(range(min(len(predictions), len(targets))))
        
        # 初始化层权重
        self._init_layer_weights(len(layer_indices))
        
        components = {}
        layer_losses_dict = {}
        layer_similarities_dict = {}
        
        total_loss = torch.tensor(0.0, device=predictions[0].device if predictions else 'cpu')
        count = 0
        
        for i, idx in enumerate(layer_indices):
            if idx >= len(predictions) or idx >= len(targets):
                continue
            
            s_feat = predictions[idx]
            t_feat = targets[idx]
            
            # 维度对齐
            if s_feat.shape[-1] != t_feat.shape[-1]:
                projector = self._get_projector(
                    s_feat.shape[-1],
                    t_feat.shape[-1],
                    f"layer_{idx}"
                )
                s_feat = projector(s_feat)
            
            try:
                # 计算损失
                layer_loss = self._compute_layer_loss(s_feat, t_feat.detach())
                
                # 获取层权重
                weight = self._layer_weight_manager.get_weight(i) if self._layer_weight_manager else 1.0
                
                weighted_loss = layer_loss * weight
                total_loss = total_loss + weighted_loss
                count += 1
                
                # 记录层损失
                components[f'layer_{idx}_loss'] = layer_loss
                layer_losses_dict[idx] = layer_loss.item()
                
                # 记录层级损失用于自适应权重
                if self._layer_weight_manager:
                    self._layer_weight_manager.record_layer_loss(i, layer_loss.item())
                
                # 计算特征相似度
                similarity = self._compute_feature_similarity(s_feat, t_feat)
                layer_similarities_dict[idx] = similarity
                
                # 记录历史
                self._layer_losses[idx].append(layer_loss.item())
                self._layer_similarities[idx].append(similarity)
                
                # 限制历史长度
                if len(self._layer_losses[idx]) > 100:
                    self._layer_losses[idx].pop(0)
                    self._layer_similarities[idx].pop(0)
                    
            except Exception as e:
                logger.warning(f"Layer {idx} feature loss compute failed: {e}")
        
        # 更新自适应权重
        if self._layer_weight_manager and self.layer_weight_strategy == 'adaptive':
            self._layer_weight_manager.update_weights()
        
        final_loss = total_loss / max(count, 1)
        
        # 计算平均相似度
        avg_similarity = sum(layer_similarities_dict.values()) / max(len(layer_similarities_dict), 1)
        
        # 记录统计
        self._record_distillation_stats(
            feature_loss=final_loss.item()
        )
        
        metrics = {
            'feature_loss': final_loss.item(),
            'num_layers': count,
            'avg_similarity': avg_similarity,
            'layer_losses': layer_losses_dict,
            'layer_similarities': layer_similarities_dict,
        }
        
        if self._layer_weight_manager:
            metrics['layer_weights'] = self._layer_weight_manager.get_all_weights()
        
        return LossResult(
            loss=final_loss * self.config.weight,
            components=components,
            metrics=metrics,
            step=self._dl_step
        )
    
    def get_layer_stats(self) -> Dict[int, Dict[str, float]]:
        """获取层级统计"""
        stats = {}
        for idx in self._layer_losses:
            losses = self._layer_losses[idx]
            similarities = self._layer_similarities[idx]
            if losses:
                stats[idx] = {
                    'avg_loss': sum(losses) / len(losses),
                    'min_loss': min(losses),
                    'max_loss': max(losses),
                    'avg_similarity': sum(similarities) / len(similarities) if similarities else 0.0
                }
        return stats
    
    def reset_layer_stats(self) -> None:
        """重置层级统计"""
        self._layer_losses.clear()
        self._layer_similarities.clear()
        if self._layer_weight_manager:
            self._layer_weight_manager.reset()


# ==================== 注意力蒸馏 ====================

@register_loss("attention_kd")
class AttentionDistillationLoss(DistillationLossModule):
    """
    注意力蒸馏损失
    
    匹配学生和教师的注意力图。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        loss_type: str = "mse",  # mse, kl
        layer_weight_strategy: str = 'uniform',
        **kwargs
    ):
        super().__init__(config)
        self.loss_type = loss_type
        self.layer_weight_strategy = layer_weight_strategy
        
        # 新增：层级权重管理器
        self._layer_weight_manager: Optional[LayerWeightManager] = None
        
        # 新增：层级统计
        self._layer_losses: Dict[int, List[float]] = defaultdict(list)
        self._attention_correlations: Dict[int, List[float]] = defaultdict(list)
    
    def _init_layer_weights(self, num_layers: int) -> None:
        """初始化层权重"""
        if self._layer_weight_manager is None or self._layer_weight_manager.num_layers != num_layers:
            self._layer_weight_manager = LayerWeightManager(
                num_layers=num_layers,
                strategy=self.layer_weight_strategy
            )
    
    def _compute_attention_correlation(
        self,
        student_attn: Tensor,
        teacher_attn: Tensor
    ) -> float:
        """计算注意力相关性"""
        s_flat = student_attn.flatten()
        t_flat = teacher_attn.flatten()
        
        # 计算皮尔逊相关系数
        s_mean = s_flat.mean()
        t_mean = t_flat.mean()
        
        s_centered = s_flat - s_mean
        t_centered = t_flat - t_mean
        
        correlation = (s_centered * t_centered).sum() / (
            s_centered.norm() * t_centered.norm() + 1e-8
        )
        
        return correlation.item()
    
    def forward(
        self, 
        predictions: Tuple[Tensor, ...],  # student attentions
        targets: Tuple[Tensor, ...],      # teacher attentions
        layer_indices: Optional[List[int]] = None,
        **kwargs
    ) -> Tensor:
        """
        计算注意力蒸馏损失
        
        Args:
            predictions: 学生注意力元组 [N, num_heads, seq_len, seq_len]
            targets: 教师注意力元组
            layer_indices: 要蒸馏的层索引
        """
        if not predictions or not targets:
            return torch.tensor(0.0)
        
        if layer_indices is None:
            layer_indices = list(range(min(len(predictions), len(targets))))
        
        total_loss = 0.0
        count = 0
        
        for idx in layer_indices:
            if idx >= len(predictions) or idx >= len(targets):
                continue
            
            s_attn = predictions[idx]  # [N, H, L, L]
            t_attn = targets[idx]
            
            # 处理头数不匹配
            if s_attn.size(1) != t_attn.size(1):
                # 对头维度平均
                s_attn = s_attn.mean(dim=1, keepdim=True)
                t_attn = t_attn.mean(dim=1, keepdim=True)
            
            # 处理序列长度不匹配
            if s_attn.shape[-1] != t_attn.shape[-1]:
                min_len = min(s_attn.shape[-1], t_attn.shape[-1])
                s_attn = s_attn[..., :min_len, :min_len]
                t_attn = t_attn[..., :min_len, :min_len]
            
            try:
                if self.loss_type == "mse":
                    layer_loss = F.mse_loss(s_attn, t_attn.detach())
                elif self.loss_type == "kl":
                    s_log = F.log_softmax(s_attn, dim=-1)
                    t_soft = F.softmax(t_attn.detach(), dim=-1)
                    layer_loss = F.kl_div(s_log, t_soft, reduction='batchmean')
                else:
                    layer_loss = F.mse_loss(s_attn, t_attn.detach())
                
                total_loss = total_loss + layer_loss
                count += 1
            except Exception as e:
                logger.warning(f"Layer {idx} attention loss failed: {e}")
        
        return total_loss / max(count, 1)
    
    def compute(
        self, 
        predictions: Tuple[Tensor, ...],
        targets: Tuple[Tensor, ...],
        layer_indices: Optional[List[int]] = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        if not predictions or not targets:
            return LossResult(
                loss=torch.tensor(0.0),
                components={},
                metrics={'attention_loss': 0.0},
                step=self._dl_step
            )
        
        if layer_indices is None:
            layer_indices = list(range(min(len(predictions), len(targets))))
        
        # 初始化层权重
        self._init_layer_weights(len(layer_indices))
        
        components = {}
        layer_losses_dict = {}
        layer_correlations_dict = {}
        
        total_loss = torch.tensor(0.0, device=predictions[0].device if predictions else 'cpu')
        count = 0
        
        for i, idx in enumerate(layer_indices):
            if idx >= len(predictions) or idx >= len(targets):
                continue
            
            s_attn = predictions[idx]
            t_attn = targets[idx]
            
            # 处理头数不匹配
            if s_attn.size(1) != t_attn.size(1):
                s_attn = s_attn.mean(dim=1, keepdim=True)
                t_attn = t_attn.mean(dim=1, keepdim=True)
            
            # 处理序列长度不匹配
            if s_attn.shape[-1] != t_attn.shape[-1]:
                min_len = min(s_attn.shape[-1], t_attn.shape[-1])
                s_attn = s_attn[..., :min_len, :min_len]
                t_attn = t_attn[..., :min_len, :min_len]
            
            try:
                if self.loss_type == "mse":
                    layer_loss = F.mse_loss(s_attn, t_attn.detach())
                elif self.loss_type == "kl":
                    s_log = F.log_softmax(s_attn, dim=-1)
                    t_soft = F.softmax(t_attn.detach(), dim=-1)
                    layer_loss = F.kl_div(s_log, t_soft, reduction='batchmean')
                else:
                    layer_loss = F.mse_loss(s_attn, t_attn.detach())
                
                # 获取层权重
                weight = self._layer_weight_manager.get_weight(i) if self._layer_weight_manager else 1.0
                
                weighted_loss = layer_loss * weight
                total_loss = total_loss + weighted_loss
                count += 1
                
                # 记录层损失
                components[f'layer_{idx}_loss'] = layer_loss
                layer_losses_dict[idx] = layer_loss.item()
                
                # 记录层级损失用于自适应权重
                if self._layer_weight_manager:
                    self._layer_weight_manager.record_layer_loss(i, layer_loss.item())
                
                # 计算注意力相关性
                correlation = self._compute_attention_correlation(s_attn, t_attn)
                layer_correlations_dict[idx] = correlation
                
                # 记录历史
                self._layer_losses[idx].append(layer_loss.item())
                self._attention_correlations[idx].append(correlation)
                
                # 限制历史长度
                if len(self._layer_losses[idx]) > 100:
                    self._layer_losses[idx].pop(0)
                    self._attention_correlations[idx].pop(0)
                    
            except Exception as e:
                logger.warning(f"Layer {idx} attention loss compute failed: {e}")
        
        # 更新自适应权重
        if self._layer_weight_manager and self.layer_weight_strategy == 'adaptive':
            self._layer_weight_manager.update_weights()
        
        final_loss = total_loss / max(count, 1)
        
        # 计算平均相关性
        avg_correlation = sum(layer_correlations_dict.values()) / max(len(layer_correlations_dict), 1)
        
        # 记录统计
        self._record_distillation_stats(
            attention_loss=final_loss.item()
        )
        
        metrics = {
            'attention_loss': final_loss.item(),
            'num_layers': count,
            'avg_correlation': avg_correlation,
            'layer_losses': layer_losses_dict,
            'layer_correlations': layer_correlations_dict,
        }
        
        if self._layer_weight_manager:
            metrics['layer_weights'] = self._layer_weight_manager.get_all_weights()
        
        return LossResult(
            loss=final_loss * self.config.weight,
            components=components,
            metrics=metrics,
            step=self._dl_step
        )
    
    def get_layer_stats(self) -> Dict[int, Dict[str, float]]:
        """获取层级统计"""
        stats = {}
        for idx in self._layer_losses:
            losses = self._layer_losses[idx]
            correlations = self._attention_correlations[idx]
            if losses:
                stats[idx] = {
                    'avg_loss': sum(losses) / len(losses),
                    'min_loss': min(losses),
                    'max_loss': max(losses),
                    'avg_correlation': sum(correlations) / len(correlations) if correlations else 0.0
                }
        return stats
    
    def reset_layer_stats(self) -> None:
        """重置层级统计"""
        self._layer_losses.clear()
        self._attention_correlations.clear()
        if self._layer_weight_manager:
            self._layer_weight_manager.reset()


# ==================== 关系蒸馏 ====================

@register_loss("relational_kd")
class RelationalDistillationLoss(DistillationLossModule):
    """
    关系知识蒸馏损失
    
    蒸馏样本间的关系，而非单个样本的表示。
    
    Reference: Park et al., "Relational Knowledge Distillation"
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        distance_wise: bool = True,
        angle_wise: bool = True,
        **kwargs
    ):
        super().__init__(config)
        self.distance_wise = distance_wise
        self.angle_wise = angle_wise
        
        # 新增：关系统计历史
        self._distance_losses: List[float] = []
        self._angle_losses: List[float] = []
        self._distance_correlations: List[float] = []
        self._angle_correlations: List[float] = []
    
    def _compute_distance_correlation(
        self,
        student_dist: Tensor,
        teacher_dist: Tensor
    ) -> float:
        """计算距离矩阵相关性"""
        # 取上三角（排除对角线）
        n = student_dist.size(0)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=student_dist.device), diagonal=1)
        
        s_vals = student_dist[mask]
        t_vals = teacher_dist[mask]
        
        # 皮尔逊相关系数
        s_mean = s_vals.mean()
        t_mean = t_vals.mean()
        
        s_centered = s_vals - s_mean
        t_centered = t_vals - t_mean
        
        correlation = (s_centered * t_centered).sum() / (
            s_centered.norm() * t_centered.norm() + 1e-8
        )
        
        return correlation.item()
    
    def _distance_wise_loss(
        self, 
        student_feat: Tensor, 
        teacher_feat: Tensor
    ) -> Tensor:
        """距离关系蒸馏"""
        # 计算成对距离
        s_flat = student_feat.flatten(1)  # [N, D]
        t_flat = teacher_feat.flatten(1)
        
        # 欧氏距离
        s_dist = torch.cdist(s_flat, s_flat)  # [N, N]
        t_dist = torch.cdist(t_flat, t_flat)
        
        # 归一化
        s_dist = s_dist / (s_dist.mean() + 1e-8)
        t_dist = t_dist / (t_dist.mean() + 1e-8)
        
        return F.smooth_l1_loss(s_dist, t_dist.detach())
    
    def _angle_wise_loss(
        self, 
        student_feat: Tensor, 
        teacher_feat: Tensor
    ) -> Tensor:
        """角度关系蒸馏"""
        s_flat = F.normalize(student_feat.flatten(1), dim=-1)
        t_flat = F.normalize(teacher_feat.flatten(1), dim=-1)
        
        # 计算角度（cosine similarity）
        s_angle = torch.mm(s_flat, s_flat.t())  # [N, N]
        t_angle = torch.mm(t_flat, t_flat.t())
        
        return F.smooth_l1_loss(s_angle, t_angle.detach())
    
    def forward(
        self, 
        predictions: Tensor,  # student features
        targets: Tensor,      # teacher features
        **kwargs
    ) -> Tensor:
        """计算关系蒸馏损失"""
        total_loss = 0.0
        
        if self.distance_wise:
            total_loss = total_loss + self._distance_wise_loss(predictions, targets)
        
        if self.angle_wise:
            total_loss = total_loss + self._angle_wise_loss(predictions, targets)
        
        return total_loss
    
    def compute(
        self, 
        predictions: Tensor,
        targets: Tensor,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        components = {}
        metrics = {}
        
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        # 计算距离关系蒸馏
        if self.distance_wise:
            # 计算距离矩阵
            s_flat = predictions.flatten(1)
            t_flat = targets.flatten(1)
            
            s_dist = torch.cdist(s_flat, s_flat)
            t_dist = torch.cdist(t_flat, t_flat)
            
            # 归一化
            s_dist_norm = s_dist / (s_dist.mean() + 1e-8)
            t_dist_norm = t_dist / (t_dist.mean() + 1e-8)
            
            dist_loss = F.smooth_l1_loss(s_dist_norm, t_dist_norm.detach())
            total_loss = total_loss + dist_loss
            
            components['distance_loss'] = dist_loss
            metrics['distance_loss'] = dist_loss.item()
            
            # 计算距离相关性
            dist_corr = self._compute_distance_correlation(s_dist, t_dist)
            metrics['distance_correlation'] = dist_corr
            
            # 记录历史
            self._distance_losses.append(dist_loss.item())
            self._distance_correlations.append(dist_corr)
            
            if len(self._distance_losses) > 100:
                self._distance_losses.pop(0)
                self._distance_correlations.pop(0)
        
        # 计算角度关系蒸馏
        if self.angle_wise:
            s_norm = F.normalize(predictions.flatten(1), dim=-1)
            t_norm = F.normalize(targets.flatten(1), dim=-1)
            
            s_angle = torch.mm(s_norm, s_norm.t())
            t_angle = torch.mm(t_norm, t_norm.t())
            
            angle_loss = F.smooth_l1_loss(s_angle, t_angle.detach())
            total_loss = total_loss + angle_loss
            
            components['angle_loss'] = angle_loss
            metrics['angle_loss'] = angle_loss.item()
            
            # 计算角度相关性
            angle_corr = self._compute_distance_correlation(s_angle, t_angle)
            metrics['angle_correlation'] = angle_corr
            
            # 记录历史
            self._angle_losses.append(angle_loss.item())
            self._angle_correlations.append(angle_corr)
            
            if len(self._angle_losses) > 100:
                self._angle_losses.pop(0)
                self._angle_correlations.pop(0)
        
        metrics['total_loss'] = total_loss.item()
        
        return LossResult(
            loss=total_loss * self.config.weight,
            components=components,
            metrics=metrics,
            step=self._dl_step
        )
    
    def get_relation_stats(self) -> Dict[str, Dict[str, float]]:
        """获取关系统计"""
        stats = {}
        
        if self._distance_losses:
            stats['distance'] = {
                'avg_loss': sum(self._distance_losses) / len(self._distance_losses),
                'avg_correlation': sum(self._distance_correlations) / len(self._distance_correlations),
            }
        
        if self._angle_losses:
            stats['angle'] = {
                'avg_loss': sum(self._angle_losses) / len(self._angle_losses),
                'avg_correlation': sum(self._angle_correlations) / len(self._angle_correlations),
            }
        
        return stats
    
    def reset_relation_stats(self) -> None:
        """重置关系统计"""
        self._distance_losses.clear()
        self._angle_losses.clear()
        self._distance_correlations.clear()
        self._angle_correlations.clear()


# ==================== 组合蒸馏损失 ====================

class CombinedDistillationLoss(DistillationLossModule):
    """
    组合蒸馏损失
    
    将多种蒸馏损失组合在一起。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        soft_loss_weight: float = 1.0,
        feature_loss_weight: float = 0.5,
        attention_loss_weight: float = 0.5,
        relational_loss_weight: float = 0.0,
        temperature: float = 4.0,
        temperature_schedule: str = 'constant',
        auto_weight: bool = False,
        **kwargs
    ):
        super().__init__(config, temperature, temperature_schedule)
        
        self.soft_loss_weight = soft_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.attention_loss_weight = attention_loss_weight
        self.relational_loss_weight = relational_loss_weight
        
        self._auto_weight = auto_weight
        self._initial_weights = {
            'soft': soft_loss_weight,
            'feature': feature_loss_weight,
            'attention': attention_loss_weight,
            'relational': relational_loss_weight
        }
        
        # 子损失函数
        self.soft_loss = SoftLabelLoss(config, temperature=temperature)
        self.feature_loss = FeatureDistillationLoss(config)
        self.attention_loss = AttentionDistillationLoss(config)
        self.relational_loss = RelationalDistillationLoss(config)
        
        # 新增：组件损失历史（用于自动权重调整）
        self._component_losses: Dict[str, List[float]] = defaultdict(list)
    
    def forward(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        student_features: Optional[Tuple[Tensor, ...]] = None,
        teacher_features: Optional[Tuple[Tensor, ...]] = None,
        student_attentions: Optional[Tuple[Tensor, ...]] = None,
        teacher_attentions: Optional[Tuple[Tensor, ...]] = None,
        hard_targets: Optional[Tensor] = None,
        hard_loss_weight: float = 0.5,
        **kwargs
    ) -> Tensor:
        """计算组合蒸馏损失"""
        total_loss = 0.0
        
        # 软标签损失
        if self.soft_loss_weight > 0:
            soft = self.soft_loss(
                student_logits, teacher_logits, hard_targets, hard_loss_weight
            )
            total_loss = total_loss + self.soft_loss_weight * soft
        
        # 特征蒸馏损失
        if self.feature_loss_weight > 0 and student_features and teacher_features:
            feat = self.feature_loss(student_features, teacher_features)
            total_loss = total_loss + self.feature_loss_weight * feat
        
        # 注意力蒸馏损失
        if self.attention_loss_weight > 0 and student_attentions and teacher_attentions:
            attn = self.attention_loss(student_attentions, teacher_attentions)
            total_loss = total_loss + self.attention_loss_weight * attn
        
        # 关系蒸馏损失
        if self.relational_loss_weight > 0 and student_features and teacher_features:
            # 使用最后一层特征
            rel = self.relational_loss(student_features[-1], teacher_features[-1])
            total_loss = total_loss + self.relational_loss_weight * rel
        
        return total_loss
    
    def compute(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        student_features: Optional[Tuple[Tensor, ...]] = None,
        teacher_features: Optional[Tuple[Tensor, ...]] = None,
        student_attentions: Optional[Tuple[Tensor, ...]] = None,
        teacher_attentions: Optional[Tuple[Tensor, ...]] = None,
        hard_targets: Optional[Tensor] = None,
        hard_loss_weight: float = 0.5,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        components = {}
        metrics = {}
        total_loss = torch.tensor(0.0, device=student_logits.device)
        
        # 获取当前温度
        temp = self.get_effective_temperature()
        
        kd_loss_val = 0.0
        ce_loss_val = 0.0
        feature_loss_val = 0.0
        attention_loss_val = 0.0
        student_acc = 0.0
        teacher_acc = 0.0
        
        # 软标签损失
        if self.soft_loss_weight > 0:
            soft_result = self.soft_loss.compute(
                student_logits, teacher_logits, hard_targets, hard_loss_weight
            )
            soft = soft_result.loss / self.soft_loss.config.weight  # 去除子损失的权重
            components['soft_loss'] = soft
            metrics['soft_loss'] = soft.item()
            kd_loss_val = soft_result.metrics.get('kd_loss', 0.0)
            ce_loss_val = soft_result.metrics.get('ce_loss', 0.0)
            student_acc = soft_result.metrics.get('student_accuracy', 0.0)
            teacher_acc = soft_result.metrics.get('teacher_accuracy', 0.0)
            
            self._component_losses['soft'].append(soft.item())
            total_loss = total_loss + self.soft_loss_weight * soft
        
        # 特征蒸馏损失
        if self.feature_loss_weight > 0 and student_features and teacher_features:
            feat_result = self.feature_loss.compute(student_features, teacher_features)
            feat = feat_result.loss / self.feature_loss.config.weight
            components['feature_loss'] = feat
            metrics['feature_loss'] = feat.item()
            feature_loss_val = feat.item()
            
            if 'avg_similarity' in feat_result.metrics:
                metrics['feature_similarity'] = feat_result.metrics['avg_similarity']
            
            self._component_losses['feature'].append(feat.item())
            total_loss = total_loss + self.feature_loss_weight * feat
        
        # 注意力蒸馏损失
        if self.attention_loss_weight > 0 and student_attentions and teacher_attentions:
            attn_result = self.attention_loss.compute(student_attentions, teacher_attentions)
            attn = attn_result.loss / self.attention_loss.config.weight
            components['attention_loss'] = attn
            metrics['attention_loss'] = attn.item()
            attention_loss_val = attn.item()
            
            if 'avg_correlation' in attn_result.metrics:
                metrics['attention_correlation'] = attn_result.metrics['avg_correlation']
            
            self._component_losses['attention'].append(attn.item())
            total_loss = total_loss + self.attention_loss_weight * attn
        
        # 关系蒸馏损失
        if self.relational_loss_weight > 0 and student_features and teacher_features:
            rel_result = self.relational_loss.compute(student_features[-1], teacher_features[-1])
            rel = rel_result.loss / self.relational_loss.config.weight
            components['relational_loss'] = rel
            metrics['relational_loss'] = rel.item()
            
            self._component_losses['relational'].append(rel.item())
            total_loss = total_loss + self.relational_loss_weight * rel
        
        # 限制历史长度
        for key in self._component_losses:
            if len(self._component_losses[key]) > 100:
                self._component_losses[key].pop(0)
        
        # 自动调整权重
        if self._auto_weight and self._dl_step > 0 and self._dl_step % 100 == 0:
            self._update_weights()
        
        # 计算KL散度
        kl_div = self._compute_kl_divergence(student_logits, teacher_logits, temp)
        
        # 计算学生-教师一致率
        agreement = self._compute_agreement(student_logits, teacher_logits)
        metrics['agreement'] = agreement
        
        # 记录统计
        self._record_distillation_stats(
            kd_loss=kd_loss_val,
            ce_loss=ce_loss_val,
            feature_loss=feature_loss_val,
            attention_loss=attention_loss_val,
            student_accuracy=student_acc,
            teacher_accuracy=teacher_acc,
            kl_divergence=kl_div
        )
        
        # 更新温度调度
        self._temp_scheduler.step(kl_div)
        
        metrics['total_loss'] = total_loss.item()
        metrics['temperature'] = temp
        metrics['kl_divergence'] = kl_div
        
        # 记录当前权重
        metrics['weights'] = {
            'soft': self.soft_loss_weight,
            'feature': self.feature_loss_weight,
            'attention': self.attention_loss_weight,
            'relational': self.relational_loss_weight
        }
        
        return LossResult(
            loss=total_loss * self.config.weight,
            components=components,
            metrics=metrics,
            step=self._dl_step
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
        
        # 归一化权重（损失大的权重增加）
        total = sum(avg_losses.values()) + 1e-8
        
        for name, avg_loss in avg_losses.items():
            new_weight = avg_loss / total
            initial_weight = self._initial_weights.get(name, 0.0)
            
            # 混合初始权重和自适应权重
            if name == 'soft':
                self.soft_loss_weight = 0.7 * initial_weight + 0.3 * new_weight
            elif name == 'feature':
                self.feature_loss_weight = 0.7 * initial_weight + 0.3 * new_weight
            elif name == 'attention':
                self.attention_loss_weight = 0.7 * initial_weight + 0.3 * new_weight
            elif name == 'relational':
                self.relational_loss_weight = 0.7 * initial_weight + 0.3 * new_weight
    
    def set_auto_weight(self, enabled: bool) -> None:
        """设置自动权重调整"""
        self._auto_weight = enabled
    
    def get_component_stats(self) -> Dict[str, Dict[str, float]]:
        """获取组件统计"""
        stats = {}
        
        for name, losses in self._component_losses.items():
            if losses:
                stats[name] = {
                    'avg_loss': sum(losses) / len(losses),
                    'recent_loss': sum(losses[-10:]) / max(len(losses[-10:]), 1),
                    'min_loss': min(losses),
                    'max_loss': max(losses)
                }
        
        return stats
    
    def get_all_weights(self) -> Dict[str, float]:
        """获取所有权重"""
        return {
            'soft': self.soft_loss_weight,
            'feature': self.feature_loss_weight,
            'attention': self.attention_loss_weight,
            'relational': self.relational_loss_weight
        }
    
    def set_weights(
        self,
        soft: Optional[float] = None,
        feature: Optional[float] = None,
        attention: Optional[float] = None,
        relational: Optional[float] = None
    ) -> None:
        """设置权重"""
        if soft is not None:
            self.soft_loss_weight = soft
        if feature is not None:
            self.feature_loss_weight = feature
        if attention is not None:
            self.attention_loss_weight = attention
        if relational is not None:
            self.relational_loss_weight = relational
    
    def reset_combined_stats(self) -> None:
        """重置组合统计"""
        self._component_losses.clear()
        self.soft_loss.reset_distillation_stats()
        self.feature_loss.reset_distillation_stats()
        self.attention_loss.reset_distillation_stats()
        self.relational_loss.reset_distillation_stats()
    
    def print_combined_summary(self) -> None:
        """打印组合损失摘要"""
        stats = self.get_distillation_stats()
        component_stats = self.get_component_stats()
        
        print("\n" + "="*80)
        print(f"Combined Distillation Loss Summary")
        print("="*80)
        
        print(f"\nTemperature: {self.get_effective_temperature():.2f}")
        print(f"Auto weight: {self._auto_weight}")
        
        print(f"\nCurrent Weights:")
        weights = self.get_all_weights()
        for name, weight in weights.items():
            print(f"  {name}: {weight:.4f}")
        
        print(f"\nComponent Statistics:")
        for name, stat in component_stats.items():
            print(f"  {name}:")
            print(f"    Avg loss: {stat['avg_loss']:.6f}")
            print(f"    Recent loss: {stat['recent_loss']:.6f}")
        
        print(f"\nOverall Statistics (over {stats.total_steps} steps):")
        print(f"  Student accuracy: {stats.avg_student_accuracy:.4f}")
        print(f"  Teacher accuracy: {stats.avg_teacher_accuracy:.4f}")
        print(f"  Accuracy gap: {self.get_accuracy_gap():.4f}")
        print(f"  KL Divergence: {stats.avg_kl_divergence:.6f}")
        
        print(f"\nDistillation effective: {self.is_distillation_effective()}")
        
        print("="*80)


# ==================== 工具函数 ====================

def create_distillation_loss(
    loss_type: str,
    temperature: float = 4.0,
    temperature_schedule: str = 'constant',
    **kwargs
) -> DistillationLossModule:
    """
    创建蒸馏损失
    
    Args:
        loss_type: 损失类型 (soft_label, feature_kd, attention_kd, relational_kd, combined)
        temperature: 温度
        temperature_schedule: 温度调度
        **kwargs: 额外参数
        
    Returns:
        蒸馏损失实例
    """
    loss_classes = {
        'soft_label': SoftLabelLoss,
        'feature_kd': FeatureDistillationLoss,
        'attention_kd': AttentionDistillationLoss,
        'relational_kd': RelationalDistillationLoss,
        'combined': CombinedDistillationLoss,
    }
    
    if loss_type not in loss_classes:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_classes.keys())}")
    
    loss_class = loss_classes[loss_type]
    
    if loss_type in ('soft_label', 'combined'):
        return loss_class(
            temperature=temperature,
            temperature_schedule=temperature_schedule,
            **kwargs
        )
    else:
        return loss_class(**kwargs)


def compare_distillation_losses(losses: Dict[str, DistillationLossModule]) -> None:
    """
    对比多个蒸馏损失
    
    Args:
        losses: 损失字典
    """
    print("\n" + "="*100)
    print("Distillation Loss Comparison")
    print("="*100)
    
    print(f"\n{'Name':<20} {'Type':<25} {'KD Loss':<12} {'Student Acc':<12} {'Teacher Acc':<12} {'Gap':<10}")
    print("-"*100)
    
    for name, loss_fn in losses.items():
        loss_type = loss_fn.__class__.__name__
        stats = loss_fn.get_distillation_stats()
        gap = loss_fn.get_accuracy_gap()
        
        print(f"{name:<20} {loss_type:<25} {stats.avg_kd_loss:<12.6f} "
              f"{stats.avg_student_accuracy:<12.4f} {stats.avg_teacher_accuracy:<12.4f} {gap:<10.4f}")
    
    print("="*100)


def analyze_distillation_progress(
    monitor: DistillationMonitor,
    window: int = 50
) -> Dict[str, Any]:
    """
    分析蒸馏进度
    
    Args:
        monitor: 蒸馏监控器
        window: 分析窗口大小
        
    Returns:
        分析结果
    """
    history = monitor.get_recent(window)
    
    if len(history) < 2:
        return {'status': 'insufficient_data'}
    
    # 计算趋势
    kd_losses = [r['kd_loss'] for r in history]
    student_accs = [r['student_accuracy'] for r in history]
    
    # 计算斜率（简单线性趋势）
    n = len(kd_losses)
    x_mean = n / 2
    
    kd_trend = sum((i - x_mean) * (kd_losses[i] - sum(kd_losses) / n) for i in range(n)) / (n * max(1, sum((i - x_mean)**2 for i in range(n)) / n))
    acc_trend = sum((i - x_mean) * (student_accs[i] - sum(student_accs) / n) for i in range(n)) / (n * max(1, sum((i - x_mean)**2 for i in range(n)) / n))
    
    # 判断状态
    if acc_trend > 0.001:
        status = 'improving'
    elif acc_trend < -0.001:
        status = 'degrading'
    else:
        status = 'stable'
    
    return {
        'status': status,
        'kd_loss_trend': kd_trend,
        'accuracy_trend': acc_trend,
        'current_kd_loss': kd_losses[-1] if kd_losses else 0.0,
        'current_accuracy': student_accs[-1] if student_accs else 0.0,
        'avg_kd_loss': sum(kd_losses) / n,
        'avg_accuracy': sum(student_accs) / n,
    }


def print_distillation_progress(monitor: DistillationMonitor, window: int = 50) -> None:
    """
    打印蒸馏进度
    
    Args:
        monitor: 蒸馏监控器
        window: 分析窗口大小
    """
    analysis = analyze_distillation_progress(monitor, window)
    
    print("\n" + "="*60)
    print("Distillation Progress Analysis")
    print("="*60)
    
    if analysis.get('status') == 'insufficient_data':
        print("Insufficient data for analysis")
        return
    
    print(f"\nStatus: {analysis['status'].upper()}")
    print(f"\nTrends (over {window} steps):")
    print(f"  KD Loss trend: {analysis['kd_loss_trend']:+.6f}")
    print(f"  Accuracy trend: {analysis['accuracy_trend']:+.6f}")
    
    print(f"\nCurrent Values:")
    print(f"  KD Loss: {analysis['current_kd_loss']:.6f}")
    print(f"  Accuracy: {analysis['current_accuracy']:.4f}")
    
    print(f"\nAverages:")
    print(f"  Avg KD Loss: {analysis['avg_kd_loss']:.6f}")
    print(f"  Avg Accuracy: {analysis['avg_accuracy']:.4f}")
    
    print("="*60)


def recommend_temperature(
    kl_divergence: float,
    current_temp: float,
    target_kl: float = 0.5
) -> float:
    """
    推荐温度
    
    Args:
        kl_divergence: 当前KL散度
        current_temp: 当前温度
        target_kl: 目标KL散度
        
    Returns:
        推荐温度
    """
    if kl_divergence > target_kl * 2:
        # KL散度太高，提高温度软化分布
        return min(current_temp * 1.2, 20.0)
    elif kl_divergence < target_kl * 0.5:
        # KL散度太低，降低温度锐化分布
        return max(current_temp * 0.8, 1.0)
    else:
        return current_temp


def estimate_distillation_quality(
    student_accuracy: float,
    teacher_accuracy: float,
    agreement: float
) -> Dict[str, Any]:
    """
    估计蒸馏质量
    
    Args:
        student_accuracy: 学生准确率
        teacher_accuracy: 教师准确率
        agreement: 学生-教师一致率
        
    Returns:
        质量评估
    """
    gap = teacher_accuracy - student_accuracy
    
    # 计算质量分数
    if gap < 0.01:
        quality = 'excellent'
        score = 1.0
    elif gap < 0.05:
        quality = 'good'
        score = 0.8
    elif gap < 0.1:
        quality = 'fair'
        score = 0.6
    elif gap < 0.2:
        quality = 'poor'
        score = 0.4
    else:
        quality = 'very_poor'
        score = 0.2
    
    # 调整分数基于一致率
    if agreement > 0.9:
        score = min(score * 1.1, 1.0)
    elif agreement < 0.7:
        score = score * 0.9
    
    return {
        'quality': quality,
        'score': score,
        'accuracy_gap': gap,
        'agreement': agreement,
        'recommendations': _get_distillation_recommendations(gap, agreement)
    }


def _get_distillation_recommendations(gap: float, agreement: float) -> List[str]:
    """获取蒸馏建议"""
    recommendations = []
    
    if gap > 0.1:
        recommendations.append("Consider using a larger student model")
        recommendations.append("Try increasing the temperature for softer targets")
    
    if gap > 0.05:
        recommendations.append("Add feature distillation for better knowledge transfer")
        recommendations.append("Increase training epochs")
    
    if agreement < 0.8:
        recommendations.append("The student is not well aligned with teacher")
        recommendations.append("Consider adding attention distillation")
    
    if not recommendations:
        recommendations.append("Distillation is performing well")
    
    return recommendations


def print_distillation_quality(
    student_accuracy: float,
    teacher_accuracy: float,
    agreement: float
) -> None:
    """
    打印蒸馏质量
    
    Args:
        student_accuracy: 学生准确率
        teacher_accuracy: 教师准确率
        agreement: 学生-教师一致率
    """
    quality = estimate_distillation_quality(student_accuracy, teacher_accuracy, agreement)
    
    print("\n" + "="*60)
    print("Distillation Quality Assessment")
    print("="*60)
    
    print(f"\nQuality: {quality['quality'].upper()}")
    print(f"Score: {quality['score']:.2f}")
    
    print(f"\nMetrics:")
    print(f"  Student accuracy: {student_accuracy:.4f}")
    print(f"  Teacher accuracy: {teacher_accuracy:.4f}")
    print(f"  Accuracy gap: {quality['accuracy_gap']:.4f}")
    print(f"  Agreement: {agreement:.4f}")
    
    print(f"\nRecommendations:")
    for rec in quality['recommendations']:
        print(f"  - {rec}")
    
    print("="*60)



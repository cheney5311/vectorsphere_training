# -*- coding: utf-8 -*-
"""
监督学习损失函数

包含分类、回归、分割等任务的损失函数。
"""

import logging
import time
import math
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_loss import (
    BaseLoss, LossConfig, LossResult,
    register_loss, weighted_loss
)

logger = logging.getLogger(__name__)


# ==================== 监控和统计组件 ====================

@dataclass
class ClassificationStats:
    """分类统计"""
    total_steps: int = 0
    total_samples: int = 0
    correct_samples: int = 0
    avg_loss: float = 0.0
    accuracy: float = 0.0
    top5_accuracy: float = 0.0
    
    # 类别统计
    class_correct: Dict[int, int] = field(default_factory=dict)
    class_total: Dict[int, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'total_samples': self.total_samples,
            'correct_samples': self.correct_samples,
            'avg_loss': self.avg_loss,
            'accuracy': self.accuracy,
            'top5_accuracy': self.top5_accuracy,
        }


@dataclass
class RegressionStats:
    """回归统计"""
    total_steps: int = 0
    total_samples: int = 0
    avg_loss: float = 0.0
    avg_mse: float = 0.0
    avg_mae: float = 0.0
    avg_rmse: float = 0.0
    r_squared: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'total_samples': self.total_samples,
            'avg_loss': self.avg_loss,
            'avg_mse': self.avg_mse,
            'avg_mae': self.avg_mae,
            'avg_rmse': self.avg_rmse,
            'r_squared': self.r_squared,
        }


@dataclass
class SegmentationStats:
    """分割统计"""
    total_steps: int = 0
    avg_loss: float = 0.0
    avg_dice: float = 0.0
    avg_iou: float = 0.0
    class_dice: Dict[int, float] = field(default_factory=dict)
    class_iou: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'avg_loss': self.avg_loss,
            'avg_dice': self.avg_dice,
            'avg_iou': self.avg_iou,
        }


class SupervisedMonitor:
    """监督学习监控器"""
    
    def __init__(self, task_type: str = 'classification', max_history: int = 1000):
        self.task_type = task_type
        self.max_history = max_history
        
        self._history: List[Dict[str, float]] = []
        
        # 分类统计
        self._cls_stats = ClassificationStats()
        self._confusion_matrix: Optional[Tensor] = None
        
        # 回归统计
        self._reg_stats = RegressionStats()
        self._predictions_sum = 0.0
        self._targets_sum = 0.0
        self._ss_tot = 0.0
        self._ss_res = 0.0
        
        # 分割统计
        self._seg_stats = SegmentationStats()
        
        # 累计统计
        self._total_loss = 0.0
    
    def record_classification(
        self,
        loss: float,
        predictions: Tensor,
        targets: Tensor,
        num_classes: Optional[int] = None
    ) -> Dict[str, float]:
        """记录分类统计"""
        batch_size = targets.size(0)
        
        # 计算准确率
        _, predicted = predictions.max(dim=-1)
        correct = (predicted == targets).sum().item()
        
        # Top-5准确率
        if predictions.size(-1) >= 5:
            _, top5_pred = predictions.topk(5, dim=-1)
            top5_correct = top5_pred.eq(targets.unsqueeze(-1)).any(dim=-1).sum().item()
        else:
            top5_correct = correct
        
        # 更新统计
        self._cls_stats.total_steps += 1
        self._cls_stats.total_samples += batch_size
        self._cls_stats.correct_samples += correct
        
        self._total_loss += loss * batch_size
        self._cls_stats.avg_loss = self._total_loss / self._cls_stats.total_samples
        self._cls_stats.accuracy = self._cls_stats.correct_samples / self._cls_stats.total_samples
        
        # 类别统计
        for i in range(batch_size):
            label = targets[i].item()
            pred = predicted[i].item()
            
            if label not in self._cls_stats.class_total:
                self._cls_stats.class_total[label] = 0
                self._cls_stats.class_correct[label] = 0
            
            self._cls_stats.class_total[label] += 1
            if pred == label:
                self._cls_stats.class_correct[label] += 1
        
        # 更新混淆矩阵
        if num_classes is not None:
            if self._confusion_matrix is None:
                self._confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
            
            for i in range(batch_size):
                self._confusion_matrix[targets[i], predicted[i]] += 1
        
        metrics = {
            'loss': loss,
            'accuracy': correct / batch_size,
            'top5_accuracy': top5_correct / batch_size,
            'timestamp': time.time()
        }
        
        self._history.append(metrics)
        if len(self._history) > self.max_history:
            self._history.pop(0)
        
        return metrics
    
    def record_regression(
        self,
        loss: float,
        predictions: Tensor,
        targets: Tensor
    ) -> Dict[str, float]:
        """记录回归统计"""
        batch_size = targets.numel()
        
        # 计算MSE和MAE
        diff = predictions.flatten() - targets.flatten()
        mse = (diff ** 2).mean().item()
        mae = diff.abs().mean().item()
        rmse = math.sqrt(mse)
        
        # 更新统计
        self._reg_stats.total_steps += 1
        self._reg_stats.total_samples += batch_size
        
        self._total_loss += loss * batch_size
        self._reg_stats.avg_loss = self._total_loss / self._reg_stats.total_samples
        
        # 更新R²计算所需的累计值
        self._targets_sum += targets.sum().item()
        mean_target = self._targets_sum / self._reg_stats.total_samples
        
        self._ss_res += ((predictions - targets) ** 2).sum().item()
        self._ss_tot += ((targets - mean_target) ** 2).sum().item()
        
        if self._ss_tot > 0:
            self._reg_stats.r_squared = 1 - (self._ss_res / self._ss_tot)
        
        # 更新移动平均
        alpha = 0.1
        self._reg_stats.avg_mse = (1 - alpha) * self._reg_stats.avg_mse + alpha * mse
        self._reg_stats.avg_mae = (1 - alpha) * self._reg_stats.avg_mae + alpha * mae
        self._reg_stats.avg_rmse = (1 - alpha) * self._reg_stats.avg_rmse + alpha * rmse
        
        metrics = {
            'loss': loss,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'timestamp': time.time()
        }
        
        self._history.append(metrics)
        if len(self._history) > self.max_history:
            self._history.pop(0)
        
        return metrics
    
    def record_segmentation(
        self,
        loss: float,
        dice: float,
        iou: float,
        class_dice: Optional[Dict[int, float]] = None,
        class_iou: Optional[Dict[int, float]] = None
    ) -> Dict[str, float]:
        """记录分割统计"""
        self._seg_stats.total_steps += 1
        
        self._total_loss += loss
        self._seg_stats.avg_loss = self._total_loss / self._seg_stats.total_steps
        
        # 更新移动平均
        alpha = 0.1
        self._seg_stats.avg_dice = (1 - alpha) * self._seg_stats.avg_dice + alpha * dice
        self._seg_stats.avg_iou = (1 - alpha) * self._seg_stats.avg_iou + alpha * iou
        
        # 更新类别统计
        if class_dice:
            for cls, val in class_dice.items():
                if cls not in self._seg_stats.class_dice:
                    self._seg_stats.class_dice[cls] = val
                else:
                    self._seg_stats.class_dice[cls] = (1 - alpha) * self._seg_stats.class_dice[cls] + alpha * val
        
        if class_iou:
            for cls, val in class_iou.items():
                if cls not in self._seg_stats.class_iou:
                    self._seg_stats.class_iou[cls] = val
                else:
                    self._seg_stats.class_iou[cls] = (1 - alpha) * self._seg_stats.class_iou[cls] + alpha * val
        
        metrics = {
            'loss': loss,
            'dice': dice,
            'iou': iou,
            'timestamp': time.time()
        }
        
        self._history.append(metrics)
        if len(self._history) > self.max_history:
            self._history.pop(0)
        
        return metrics
    
    def get_classification_stats(self) -> ClassificationStats:
        """获取分类统计"""
        return self._cls_stats
    
    def get_regression_stats(self) -> RegressionStats:
        """获取回归统计"""
        return self._reg_stats
    
    def get_segmentation_stats(self) -> SegmentationStats:
        """获取分割统计"""
        return self._seg_stats
    
    def get_confusion_matrix(self) -> Optional[Tensor]:
        """获取混淆矩阵"""
        return self._confusion_matrix
    
    def get_class_accuracy(self) -> Dict[int, float]:
        """获取类别准确率"""
        class_acc = {}
        for cls in self._cls_stats.class_total:
            total = self._cls_stats.class_total[cls]
            correct = self._cls_stats.class_correct.get(cls, 0)
            class_acc[cls] = correct / total if total > 0 else 0.0
        return class_acc
    
    def get_recent(self, n: int = 10) -> List[Dict[str, float]]:
        """获取最近的记录"""
        return self._history[-n:]
    
    def reset(self) -> None:
        """重置"""
        self._history.clear()
        self._cls_stats = ClassificationStats()
        self._reg_stats = RegressionStats()
        self._seg_stats = SegmentationStats()
        self._confusion_matrix = None
        self._total_loss = 0.0
        self._predictions_sum = 0.0
        self._targets_sum = 0.0
        self._ss_tot = 0.0
        self._ss_res = 0.0


class ClassWeightCalculator:
    """类别权重计算器"""
    
    @staticmethod
    def compute_class_weights(
        labels: Tensor,
        num_classes: int,
        method: str = 'inverse_freq'
    ) -> Tensor:
        """
        计算类别权重
        
        Args:
            labels: 标签张量
            num_classes: 类别数
            method: 计算方法 (inverse_freq, effective_samples, sqrt_inverse)
            
        Returns:
            类别权重
        """
        # 统计每个类别的数量
        counts = torch.bincount(labels.flatten(), minlength=num_classes).float()
        
        if method == 'inverse_freq':
            # 逆频率
            weights = 1.0 / (counts + 1e-6)
            weights = weights / weights.sum() * num_classes
            
        elif method == 'effective_samples':
            # 有效样本数方法
            beta = 0.9999
            effective_num = 1.0 - torch.pow(beta, counts)
            weights = (1.0 - beta) / (effective_num + 1e-6)
            weights = weights / weights.sum() * num_classes
            
        elif method == 'sqrt_inverse':
            # 平方根逆频率
            weights = 1.0 / (torch.sqrt(counts) + 1e-6)
            weights = weights / weights.sum() * num_classes
            
        else:
            weights = torch.ones(num_classes)
        
        return weights
    
    @staticmethod
    def compute_sample_weights(
        labels: Tensor,
        class_weights: Tensor
    ) -> Tensor:
        """
        计算样本权重
        
        Args:
            labels: 标签张量
            class_weights: 类别权重
            
        Returns:
            样本权重
        """
        return class_weights[labels]




# ==================== 基础监督损失 ====================

class SupervisedLoss(BaseLoss):
    """监督学习损失基类"""
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        task_type: str = 'classification',
        **kwargs
    ):
        super().__init__(config)
        self.num_classes = kwargs.get('num_classes', None)
        self.task_type = task_type
        
        # 新增：监控器
        self._supervised_monitor = SupervisedMonitor(task_type=task_type)
        self._sl_step = 0
    
    def get_monitor(self) -> SupervisedMonitor:
        """获取监控器"""
        return self._supervised_monitor
    
    def get_classification_stats(self) -> ClassificationStats:
        """获取分类统计"""
        return self._supervised_monitor.get_classification_stats()
    
    def get_regression_stats(self) -> RegressionStats:
        """获取回归统计"""
        return self._supervised_monitor.get_regression_stats()
    
    def get_segmentation_stats(self) -> SegmentationStats:
        """获取分割统计"""
        return self._supervised_monitor.get_segmentation_stats()
    
    def print_summary(self) -> None:
        """打印摘要"""
        print("\n" + "="*80)
        print(f"Supervised Loss Summary: {self.__class__.__name__}")
        print("="*80)
        
        if self.task_type == 'classification':
            stats = self.get_classification_stats()
            print("\nTask: Classification")
            print(f"\nStatistics (over {stats.total_steps} steps, {stats.total_samples} samples):")
            print(f"  Avg loss: {stats.avg_loss:.6f}")
            print(f"  Accuracy: {stats.accuracy:.4f}")
            print(f"  Top-5 accuracy: {stats.top5_accuracy:.4f}")
            
            class_acc = self._supervised_monitor.get_class_accuracy()
            if class_acc:
                print("\nClass-wise accuracy:")
                for cls, acc in sorted(class_acc.items())[:10]:
                    print(f"  Class {cls}: {acc:.4f}")
        
        elif self.task_type == 'regression':
            stats = self.get_regression_stats()
            print("\nTask: Regression")
            print(f"\nStatistics (over {stats.total_steps} steps, {stats.total_samples} samples):")
            print(f"  Avg loss: {stats.avg_loss:.6f}")
            print(f"  MSE: {stats.avg_mse:.6f}")
            print(f"  MAE: {stats.avg_mae:.6f}")
            print(f"  RMSE: {stats.avg_rmse:.6f}")
            print(f"  R²: {stats.r_squared:.4f}")
        
        elif self.task_type == 'segmentation':
            stats = self.get_segmentation_stats()
            print("\nTask: Segmentation")
            print(f"\nStatistics (over {stats.total_steps} steps):")
            print(f"  Avg loss: {stats.avg_loss:.6f}")
            print(f"  Avg Dice: {stats.avg_dice:.4f}")
            print(f"  Avg IoU: {stats.avg_iou:.4f}")
        
        print("="*80)
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._supervised_monitor.reset()
        self._sl_step = 0


# ==================== 分类损失 ====================

class ClassificationLoss(SupervisedLoss):
    """分类任务损失基类"""
    
    def __init__(self, config: Optional[LossConfig] = None, **kwargs):
        super().__init__(config, task_type='classification', **kwargs)
        
        # 类别权重计算器
        self._weight_calculator = ClassWeightCalculator()
    
    def _compute_accuracy(self, predictions: Tensor, targets: Tensor) -> Dict[str, float]:
        """计算准确率"""
        _, predicted = predictions.max(dim=-1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        
        accuracy = correct / total
        
        # Top-5准确率
        if predictions.size(-1) >= 5:
            _, top5_pred = predictions.topk(5, dim=-1)
            top5_correct = top5_pred.eq(targets.unsqueeze(-1)).any(dim=-1).sum().item()
            top5_accuracy = top5_correct / total
        else:
            top5_accuracy = accuracy
        
        return {
            'accuracy': accuracy,
            'top5_accuracy': top5_accuracy,
            'correct': correct,
            'total': total
        }
    
    def _compute_per_class_accuracy(self, predictions: Tensor, targets: Tensor) -> Dict[int, float]:
        """计算每类准确率"""
        _, predicted = predictions.max(dim=-1)
        
        class_acc = {}
        for cls in targets.unique().tolist():
            mask = targets == cls
            if mask.sum() > 0:
                class_correct = (predicted[mask] == targets[mask]).sum().item()
                class_total = mask.sum().item()
                class_acc[cls] = class_correct / class_total
        
        return class_acc
    
    def get_confusion_matrix(self) -> Optional[Tensor]:
        """获取混淆矩阵"""
        return self._supervised_monitor.get_confusion_matrix()
    
    def get_class_accuracy(self) -> Dict[int, float]:
        """获取类别准确率"""
        return self._supervised_monitor.get_class_accuracy()


@register_loss("cross_entropy")
class CrossEntropyLoss(ClassificationLoss):
    """
    交叉熵损失
    
    标准的多分类损失函数。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        num_classes: Optional[int] = None,
        auto_weight: bool = False,
        weight_method: str = 'inverse_freq',
        **kwargs
    ):
        super().__init__(config, num_classes=num_classes, **kwargs)
        
        self.auto_weight = auto_weight
        self.weight_method = weight_method
        
        # 类别权重
        weight = None
        if config and config.class_weights:
            weight = torch.tensor(config.class_weights)
        
        self._class_weight = weight
        self._label_smoothing = config.label_smoothing if config else 0.0
        
        self.loss_fn = nn.CrossEntropyLoss(
            weight=weight,
            reduction='none',
            label_smoothing=self._label_smoothing
        )
    
    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        计算交叉熵损失
        
        Args:
            predictions: [N, C] 或 [N, C, H, W] logits
            targets: [N] 或 [N, H, W] 标签
            sample_weights: [N] 样本权重
        """
        loss = self.loss_fn(predictions, targets)
        return weighted_loss(loss, sample_weights, self.config.reduction)
    
    def compute(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        # 自动计算类别权重
        if self.auto_weight and self._class_weight is None:
            num_classes = predictions.size(-1)
            self._class_weight = self._weight_calculator.compute_class_weights(
                targets, num_classes, self.weight_method
            ).to(predictions.device)
            self.loss_fn = nn.CrossEntropyLoss(
                weight=self._class_weight,
                reduction='none',
                label_smoothing=self._label_smoothing
            )
        
        # 计算损失
        loss = self.loss_fn(predictions, targets)
        final_loss = weighted_loss(loss, sample_weights, self.config.reduction)
        
        # 计算准确率
        acc_metrics = self._compute_accuracy(predictions, targets)
        
        # 计算每类准确率
        class_acc = self._compute_per_class_accuracy(predictions, targets)
        
        # 计算置信度统计
        probs = F.softmax(predictions, dim=-1)
        max_probs = probs.max(dim=-1).values
        mean_confidence = max_probs.mean().item()
        
        # 记录到监控器
        self._supervised_monitor.record_classification(
            loss=final_loss.item(),
            predictions=predictions,
            targets=targets,
            num_classes=self.num_classes
        )
        self._sl_step += 1
        
        return LossResult(
            loss=final_loss * self.config.weight,
            components={'ce_loss': final_loss},
            metrics={
                'ce_loss': final_loss.item(),
                'accuracy': acc_metrics['accuracy'],
                'top5_accuracy': acc_metrics['top5_accuracy'],
                'mean_confidence': mean_confidence,
                'label_smoothing': self._label_smoothing,
                'class_accuracy': class_acc,
            },
            step=self._sl_step
        )
    
    def update_class_weights(self, labels: Tensor) -> None:
        """更新类别权重"""
        num_classes = self.num_classes or labels.max().item() + 1
        self._class_weight = self._weight_calculator.compute_class_weights(
            labels, num_classes, self.weight_method
        )
        self.loss_fn = nn.CrossEntropyLoss(
            weight=self._class_weight,
            reduction='none',
            label_smoothing=self._label_smoothing
        )


@register_loss("focal")
class FocalLoss(ClassificationLoss):
    """
    Focal Loss
    
    解决类别不平衡问题，专注于难分类样本。
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        num_classes: Optional[int] = None,
        **kwargs
    ):
        super().__init__(config, num_classes=num_classes, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
        
        # 类别权重
        self.class_weights = None
        if config and config.class_weights:
            self.register_buffer(
                'class_weights', 
                torch.tensor(config.class_weights)
            )
        
        # 新增：难样本分析历史
        self._hard_sample_history: List[float] = []
    
    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        计算Focal Loss
        
        Args:
            predictions: [N, C] logits
            targets: [N] 标签
        """
        # 计算softmax概率
        probs = F.softmax(predictions, dim=-1)
        
        # 获取正确类别的概率
        num_classes = predictions.size(-1)
        
        # 创建one-hot标签
        targets_one_hot = torch.one_hot(targets, num_classes).float()
        
        # 计算p_t
        p_t = (probs * targets_one_hot).sum(dim=-1)
        
        # 计算focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # 应用focal weight和alpha
        loss = self.alpha * focal_weight * ce_loss
        
        # 应用类别权重
        if self.class_weights is not None:
            class_weight = self.class_weights[targets]
            loss = loss * class_weight
        
        return weighted_loss(loss, sample_weights, self.config.reduction)
    
    def compute(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        # 计算softmax概率
        probs = F.softmax(predictions, dim=-1)
        num_classes = predictions.size(-1)
        
        # 创建one-hot标签
        targets_one_hot = torch.one_hot(targets, num_classes).float()
        
        # 计算p_t
        p_t = (probs * targets_one_hot).sum(dim=-1)
        
        # 计算focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # 计算交叉熵
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        
        # 应用focal weight和alpha
        loss = self.alpha * focal_weight * ce_loss
        
        # 应用类别权重
        if self.class_weights is not None:
            class_weight = self.class_weights[targets]
            loss = loss * class_weight
        
        final_loss = weighted_loss(loss, sample_weights, self.config.reduction)
        
        # 计算准确率
        acc_metrics = self._compute_accuracy(predictions, targets)
        
        # 难样本分析
        hard_sample_ratio = (p_t < 0.5).float().mean().item()
        self._hard_sample_history.append(hard_sample_ratio)
        if len(self._hard_sample_history) > 100:
            self._hard_sample_history.pop(0)
        
        # 记录到监控器
        self._supervised_monitor.record_classification(
            loss=final_loss.item(),
            predictions=predictions,
            targets=targets,
            num_classes=self.num_classes
        )
        self._sl_step += 1
        
        return LossResult(
            loss=final_loss * self.config.weight,
            components={
                'focal_loss': final_loss,
                'ce_loss': ce_loss.mean(),
                'focal_weight': focal_weight.mean()
            },
            metrics={
                'focal_loss': final_loss.item(),
                'ce_loss': ce_loss.mean().item(),
                'accuracy': acc_metrics['accuracy'],
                'top5_accuracy': acc_metrics['top5_accuracy'],
                'mean_p_t': p_t.mean().item(),
                'mean_focal_weight': focal_weight.mean().item(),
                'hard_sample_ratio': hard_sample_ratio,
                'alpha': self.alpha,
                'gamma': self.gamma
            },
            step=self._sl_step
        )
    
    def get_hard_sample_trend(self) -> float:
        """获取难样本比例趋势"""
        if len(self._hard_sample_history) < 10:
            return 0.0
        
        recent = sum(self._hard_sample_history[-5:]) / 5
        earlier = sum(self._hard_sample_history[-10:-5]) / 5
        return recent - earlier
    
    def set_gamma(self, gamma: float) -> None:
        """设置gamma"""
        self.gamma = gamma
    
    def set_alpha(self, alpha: float) -> None:
        """设置alpha"""
        self.alpha = alpha


@register_loss("label_smoothing")
class LabelSmoothingLoss(ClassificationLoss):
    """
    标签平滑损失
    
    通过软化标签来提高泛化能力。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        smoothing: float = 0.1,
        num_classes: Optional[int] = None,
        **kwargs
    ):
        super().__init__(config, num_classes=num_classes, **kwargs)
        self.smoothing = config.label_smoothing if config else smoothing
        self.num_classes = num_classes
    
    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """计算标签平滑损失"""
        num_classes = predictions.size(-1)
        
        # 创建软标签
        soft_targets = torch.zeros_like(predictions)
        soft_targets.fill_(self.smoothing / (num_classes - 1))
        soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # 计算KL散度
        log_probs = F.log_softmax(predictions, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1)
        
        return weighted_loss(loss, sample_weights, self.config.reduction)
    
    def compute(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        num_classes = predictions.size(-1)
        
        # 创建软标签
        soft_targets = torch.zeros_like(predictions)
        soft_targets.fill_(self.smoothing / (num_classes - 1))
        soft_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # 计算KL散度
        log_probs = F.log_softmax(predictions, dim=-1)
        probs = F.softmax(predictions, dim=-1)
        loss = -(soft_targets * log_probs).sum(dim=-1)
        
        final_loss = weighted_loss(loss, sample_weights, self.config.reduction)
        
        # 计算准确率
        acc_metrics = self._compute_accuracy(predictions, targets)
        
        # 计算预测熵
        entropy = -(probs * log_probs).sum(dim=-1).mean().item()
        
        # 计算置信度
        max_probs = probs.max(dim=-1).values
        mean_confidence = max_probs.mean().item()
        
        # 记录到监控器
        self._supervised_monitor.record_classification(
            loss=final_loss.item(),
            predictions=predictions,
            targets=targets,
            num_classes=self.num_classes
        )
        self._sl_step += 1
        
        return LossResult(
            loss=final_loss * self.config.weight,
            components={'ls_loss': final_loss},
            metrics={
                'ls_loss': final_loss.item(),
                'accuracy': acc_metrics['accuracy'],
                'top5_accuracy': acc_metrics['top5_accuracy'],
                'smoothing': self.smoothing,
                'entropy': entropy,
                'mean_confidence': mean_confidence
            },
            step=self._sl_step
        )
    
    def set_smoothing(self, smoothing: float) -> None:
        """设置平滑系数"""
        self.smoothing = max(0.0, min(1.0, smoothing))


# ==================== 回归损失 ====================

class RegressionLoss(SupervisedLoss):
    """回归任务损失基类"""
    
    def __init__(self, config: Optional[LossConfig] = None, **kwargs):
        super().__init__(config, task_type='regression', **kwargs)
    
    def _compute_regression_metrics(self, predictions: Tensor, targets: Tensor) -> Dict[str, float]:
        """计算回归指标"""
        diff = predictions.flatten() - targets.flatten()
        
        mse = (diff ** 2).mean().item()
        mae = diff.abs().mean().item()
        rmse = math.sqrt(mse)
        
        # R²
        ss_res = (diff ** 2).sum().item()
        ss_tot = ((targets.flatten() - targets.mean()) ** 2).sum().item()
        r_squared = 1 - (ss_res / (ss_tot + 1e-8))
        
        # MAPE
        mape = (diff.abs() / (targets.abs() + 1e-8)).mean().item() * 100
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r_squared': r_squared,
            'mape': mape
        }


@register_loss("mse")
class MSELoss(RegressionLoss):
    """均方误差损失"""
    
    def __init__(self, config: Optional[LossConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.loss_fn = nn.MSELoss(reduction='none')
    
    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        loss = self.loss_fn(predictions, targets)
        if loss.dim() > 1:
            loss = loss.mean(dim=tuple(range(1, loss.dim())))
        return weighted_loss(loss, sample_weights, self.config.reduction)
    
    def compute(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        loss = self.loss_fn(predictions, targets)
        if loss.dim() > 1:
            loss = loss.mean(dim=tuple(range(1, loss.dim())))
        
        final_loss = weighted_loss(loss, sample_weights, self.config.reduction)
        
        # 计算回归指标
        reg_metrics = self._compute_regression_metrics(predictions, targets)
        
        # 记录到监控器
        self._supervised_monitor.record_regression(
            loss=final_loss.item(),
            predictions=predictions,
            targets=targets
        )
        self._sl_step += 1
        
        return LossResult(
            loss=final_loss * self.config.weight,
            components={'mse_loss': final_loss},
            metrics={
                'mse_loss': final_loss.item(),
                **reg_metrics
            },
            step=self._sl_step
        )


@register_loss("mae")
class MAELoss(RegressionLoss):
    """平均绝对误差损失"""
    
    def __init__(self, config: Optional[LossConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.loss_fn = nn.L1Loss(reduction='none')
    
    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        loss = self.loss_fn(predictions, targets)
        if loss.dim() > 1:
            loss = loss.mean(dim=tuple(range(1, loss.dim())))
        return weighted_loss(loss, sample_weights, self.config.reduction)
    
    def compute(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        loss = self.loss_fn(predictions, targets)
        if loss.dim() > 1:
            loss = loss.mean(dim=tuple(range(1, loss.dim())))
        
        final_loss = weighted_loss(loss, sample_weights, self.config.reduction)
        
        # 计算回归指标
        reg_metrics = self._compute_regression_metrics(predictions, targets)
        
        # 记录到监控器
        self._supervised_monitor.record_regression(
            loss=final_loss.item(),
            predictions=predictions,
            targets=targets
        )
        self._sl_step += 1
        
        return LossResult(
            loss=final_loss * self.config.weight,
            components={'mae_loss': final_loss},
            metrics={
                'mae_loss': final_loss.item(),
                **reg_metrics
            },
            step=self._sl_step
        )


@register_loss("huber")
class HuberLoss(RegressionLoss):
    """
    Huber损失
    
    结合MSE和MAE的优点，对异常值鲁棒。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        delta: float = 1.0,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.delta = delta
        self.loss_fn = nn.HuberLoss(reduction='none', delta=delta)
    
    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        loss = self.loss_fn(predictions, targets)
        if loss.dim() > 1:
            loss = loss.mean(dim=tuple(range(1, loss.dim())))
        return weighted_loss(loss, sample_weights, self.config.reduction)
    
    def compute(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        loss = self.loss_fn(predictions, targets)
        if loss.dim() > 1:
            loss = loss.mean(dim=tuple(range(1, loss.dim())))
        
        final_loss = weighted_loss(loss, sample_weights, self.config.reduction)
        
        # 计算回归指标
        reg_metrics = self._compute_regression_metrics(predictions, targets)
        
        # 计算在delta边界内外的比例
        diff = (predictions - targets).abs()
        within_delta = (diff <= self.delta).float().mean().item()
        
        # 记录到监控器
        self._supervised_monitor.record_regression(
            loss=final_loss.item(),
            predictions=predictions,
            targets=targets
        )
        self._sl_step += 1
        
        return LossResult(
            loss=final_loss * self.config.weight,
            components={'huber_loss': final_loss},
            metrics={
                'huber_loss': final_loss.item(),
                'delta': self.delta,
                'within_delta_ratio': within_delta,
                **reg_metrics
            },
            step=self._sl_step
        )
    
    def set_delta(self, delta: float) -> None:
        """设置delta"""
        self.delta = delta
        self.loss_fn = nn.HuberLoss(reduction='none', delta=delta)


# ==================== 分割损失 ====================

class SegmentationLoss(SupervisedLoss):
    """分割任务损失基类"""
    
    def __init__(self, config: Optional[LossConfig] = None, **kwargs):
        super().__init__(config, task_type='segmentation', **kwargs)
    
    def _compute_dice_per_class(
        self, 
        probs: Tensor, 
        targets_one_hot: Tensor,
        smooth: float = 1e-6
    ) -> Dict[int, float]:
        """计算每类Dice系数"""
        num_classes = probs.size(1)
        class_dice = {}
        
        for c in range(num_classes):
            # probs[:, c]和targets_one_hot[:, c]维度是[N, H, W]或[N]
            p = probs[:, c]
            t = targets_one_hot[:, c]
            
            # 对所有空间维度求和（除了batch维度）
            if p.dim() > 1:
                dims = tuple(range(1, p.dim()))
                intersection = (p * t).sum(dim=dims)
                union = p.sum(dim=dims) + t.sum(dim=dims)
            else:
                intersection = (p * t).sum()
                union = p.sum() + t.sum()
            
            dice = (2.0 * intersection + smooth) / (union + smooth)
            class_dice[c] = dice.mean().item()
        
        return class_dice
    
    def _compute_iou_per_class(
        self, 
        probs: Tensor, 
        targets_one_hot: Tensor,
        smooth: float = 1e-6
    ) -> Dict[int, float]:
        """计算每类IoU"""
        num_classes = probs.size(1)
        class_iou = {}
        
        for c in range(num_classes):
            p = probs[:, c]
            t = targets_one_hot[:, c]
            
            # 对所有空间维度求和（除了batch维度）
            if p.dim() > 1:
                dims = tuple(range(1, p.dim()))
                intersection = (p * t).sum(dim=dims)
                union_sum = p.sum(dim=dims) + t.sum(dim=dims)
            else:
                intersection = (p * t).sum()
                union_sum = p.sum() + t.sum()
            
            union = union_sum - intersection
            
            iou = (intersection + smooth) / (union + smooth)
            class_iou[c] = iou.mean().item()
        
        return class_iou


@register_loss("dice")
class DiceLoss(SegmentationLoss):
    """
    Dice损失
    
    用于分割任务，处理类别不平衡。
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        smooth: float = 1e-6,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.smooth = smooth
    
    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        计算Dice损失
        
        Args:
            predictions: [N, C, H, W] 或 [N, C] softmax后的预测
            targets: [N, H, W] 或 [N] 标签，或 [N, C, H, W] one-hot
        """
        # 应用softmax如果是logits
        if predictions.dim() > 2:
            probs = F.softmax(predictions, dim=1)
        else:
            probs = F.softmax(predictions, dim=-1)
        
        # 转换targets为one-hot
        if targets.dim() < predictions.dim():
            num_classes = predictions.size(1)
            if predictions.dim() == 4:  # [N, C, H, W]
                targets_one_hot = torch.one_hot(targets, num_classes)
                targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
            else:
                targets_one_hot = torch.one_hot(targets, num_classes).float()
        else:
            targets_one_hot = targets.float()
        
        # 计算Dice系数
        dims = tuple(range(2, predictions.dim()))  # 空间维度
        intersection = (probs * targets_one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # 对类别维度平均
        dice = dice.mean(dim=-1)
        
        # 1 - Dice 作为损失
        loss = 1.0 - dice
        
        return weighted_loss(loss, sample_weights, self.config.reduction)
    
    def compute(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        # 应用softmax
        if predictions.dim() > 2:
            probs = F.softmax(predictions, dim=1)
        else:
            probs = F.softmax(predictions, dim=-1)
        
        # 转换targets为one-hot
        if targets.dim() < predictions.dim():
            num_classes = predictions.size(1)
            if predictions.dim() == 4:
                targets_one_hot = torch.one_hot(targets, num_classes)
                targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
            else:
                targets_one_hot = torch.one_hot(targets, num_classes).float()
        else:
            targets_one_hot = targets.float()
            num_classes = targets_one_hot.size(1)
        
        # 计算Dice系数
        dims = tuple(range(2, predictions.dim()))
        intersection = (probs * targets_one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        mean_dice = dice.mean(dim=-1)
        
        loss = 1.0 - mean_dice
        final_loss = weighted_loss(loss, sample_weights, self.config.reduction)
        
        # 计算每类Dice
        class_dice = self._compute_dice_per_class(probs, targets_one_hot, self.smooth)
        
        # 计算IoU
        class_iou = self._compute_iou_per_class(probs, targets_one_hot, self.smooth)
        mean_iou = sum(class_iou.values()) / len(class_iou)
        
        # 记录到监控器
        self._supervised_monitor.record_segmentation(
            loss=final_loss.item(),
            dice=mean_dice.mean().item(),
            iou=mean_iou,
            class_dice=class_dice,
            class_iou=class_iou
        )
        self._sl_step += 1
        
        return LossResult(
            loss=final_loss * self.config.weight,
            components={'dice_loss': final_loss, 'dice_coeff': mean_dice.mean()},
            metrics={
                'dice_loss': final_loss.item(),
                'dice_coeff': mean_dice.mean().item(),
                'mean_iou': mean_iou,
                'class_dice': class_dice,
                'class_iou': class_iou,
                'num_classes': num_classes
            },
            step=self._sl_step
        )


@register_loss("iou")
class IoULoss(SegmentationLoss):
    """
    IoU损失
    
    基于交并比的分割损失。
    IoU = |A ∩ B| / |A ∪ B|
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        smooth: float = 1e-6,
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.smooth = smooth
    
    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """计算IoU损失"""
        probs = F.softmax(predictions, dim=1)
        
        # 转换targets为one-hot
        if targets.dim() < predictions.dim():
            num_classes = predictions.size(1)
            if predictions.dim() == 4:
                targets_one_hot = torch.one_hot(targets, num_classes)
                targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
            else:
                targets_one_hot = torch.one_hot(targets, num_classes).float()
        else:
            targets_one_hot = targets.float()
        
        # 计算IoU
        dims = tuple(range(2, predictions.dim()))
        intersection = (probs * targets_one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        iou = iou.mean(dim=-1)
        
        loss = 1.0 - iou
        
        return weighted_loss(loss, sample_weights, self.config.reduction)
    
    def compute(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        sample_weights: Optional[Tensor] = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        probs = F.softmax(predictions, dim=1)
        
        # 转换targets为one-hot
        if targets.dim() < predictions.dim():
            num_classes = predictions.size(1)
            if predictions.dim() == 4:
                targets_one_hot = torch.one_hot(targets, num_classes)
                targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
            else:
                targets_one_hot = torch.one_hot(targets, num_classes).float()
        else:
            targets_one_hot = targets.float()
            num_classes = targets_one_hot.size(1)
        
        # 计算IoU
        dims = tuple(range(2, predictions.dim()))
        intersection = (probs * targets_one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_one_hot.sum(dim=dims) - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        mean_iou = iou.mean(dim=-1)
        
        loss = 1.0 - mean_iou
        final_loss = weighted_loss(loss, sample_weights, self.config.reduction)
        
        # 计算每类IoU
        class_iou = self._compute_iou_per_class(probs, targets_one_hot, self.smooth)
        
        # 计算每类Dice
        class_dice = self._compute_dice_per_class(probs, targets_one_hot, self.smooth)
        mean_dice = sum(class_dice.values()) / len(class_dice)
        
        # 记录到监控器
        self._supervised_monitor.record_segmentation(
            loss=final_loss.item(),
            dice=mean_dice,
            iou=mean_iou.mean().item(),
            class_dice=class_dice,
            class_iou=class_iou
        )
        self._sl_step += 1
        
        return LossResult(
            loss=final_loss * self.config.weight,
            components={'iou_loss': final_loss, 'mean_iou': mean_iou.mean()},
            metrics={
                'iou_loss': final_loss.item(),
                'mean_iou': mean_iou.mean().item(),
                'mean_dice': mean_dice,
                'class_iou': class_iou,
                'class_dice': class_dice,
                'num_classes': num_classes
            },
            step=self._sl_step
        )


# ==================== 工具函数 ====================

def create_supervised_loss(
    loss_type: str,
    task_type: str = 'classification',
    num_classes: Optional[int] = None,
    **kwargs
) -> SupervisedLoss:
    """
    创建监督学习损失
    
    Args:
        loss_type: 损失类型 (cross_entropy, focal, label_smoothing, mse, mae, huber, dice, iou)
        task_type: 任务类型 (classification, regression, segmentation)
        num_classes: 类别数
        **kwargs: 额外参数
        
    Returns:
        监督学习损失实例
    """
    loss_classes = {
        'cross_entropy': CrossEntropyLoss,
        'focal': FocalLoss,
        'label_smoothing': LabelSmoothingLoss,
        'mse': MSELoss,
        'mae': MAELoss,
        'huber': HuberLoss,
        'dice': DiceLoss,
        'iou': IoULoss,
    }
    
    if loss_type not in loss_classes:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_classes.keys())}")
    
    loss_class = loss_classes[loss_type]
    
    if loss_type in ('cross_entropy', 'focal', 'label_smoothing'):
        return loss_class(num_classes=num_classes, **kwargs)
    else:
        return loss_class(**kwargs)


def compare_supervised_losses(losses: Dict[str, SupervisedLoss], task_type: str = 'classification') -> None:
    """
    对比多个监督学习损失
    
    Args:
        losses: 损失字典
        task_type: 任务类型
    """
    print("\n" + "="*100)
    print("Supervised Loss Comparison")
    print("="*100)
    
    if task_type == 'classification':
        print(f"\n{'Name':<20} {'Type':<25} {'Avg Loss':<12} {'Accuracy':<12} {'Top-5 Acc':<12}")
        print("-"*100)
        
        for name, loss_fn in losses.items():
            loss_type = loss_fn.__class__.__name__
            stats = loss_fn.get_classification_stats()
            
            print(f"{name:<20} {loss_type:<25} {stats.avg_loss:<12.6f} "
                  f"{stats.accuracy:<12.4f} {stats.top5_accuracy:<12.4f}")
    
    elif task_type == 'regression':
        print(f"\n{'Name':<20} {'Type':<25} {'Avg Loss':<12} {'MSE':<12} {'MAE':<12} {'R²':<12}")
        print("-"*100)
        
        for name, loss_fn in losses.items():
            loss_type = loss_fn.__class__.__name__
            stats = loss_fn.get_regression_stats()
            
            print(f"{name:<20} {loss_type:<25} {stats.avg_loss:<12.6f} "
                  f"{stats.avg_mse:<12.6f} {stats.avg_mae:<12.6f} {stats.r_squared:<12.4f}")
    
    elif task_type == 'segmentation':
        print(f"\n{'Name':<20} {'Type':<25} {'Avg Loss':<12} {'Dice':<12} {'IoU':<12}")
        print("-"*100)
        
        for name, loss_fn in losses.items():
            loss_type = loss_fn.__class__.__name__
            stats = loss_fn.get_segmentation_stats()
            
            print(f"{name:<20} {loss_type:<25} {stats.avg_loss:<12.6f} "
                  f"{stats.avg_dice:<12.4f} {stats.avg_iou:<12.4f}")
    
    print("="*100)


def compute_class_weights(
    labels: Tensor,
    num_classes: int,
    method: str = 'inverse_freq'
) -> Tensor:
    """
    计算类别权重
    
    Args:
        labels: 标签张量
        num_classes: 类别数
        method: 计算方法
        
    Returns:
        类别权重
    """
    return ClassWeightCalculator.compute_class_weights(labels, num_classes, method)


def print_confusion_matrix(
    confusion_matrix: Tensor,
    class_names: Optional[List[str]] = None
) -> None:
    """
    打印混淆矩阵
    
    Args:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称
    """
    num_classes = confusion_matrix.size(0)
    
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    
    print("\n" + "="*60)
    print("Confusion Matrix")
    print("="*60)
    
    # 打印表头
    header = "Pred →".ljust(10)
    for name in class_names[:10]:  # 最多显示10个类别
        header += f"{name[:8]:<10}"
    print(header)
    print("-"*60)
    
    # 打印每一行
    for i in range(min(num_classes, 10)):
        row = f"{class_names[i][:8]:<10}"
        for j in range(min(num_classes, 10)):
            row += f"{confusion_matrix[i, j].item():<10}"
        print(row)
    
    if num_classes > 10:
        print(f"... ({num_classes - 10} more classes)")
    
    print("="*60)


def analyze_classification_results(
    monitor: SupervisedMonitor,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    分析分类结果
    
    Args:
        monitor: 监控器
        class_names: 类别名称
        
    Returns:
        分析结果
    """
    stats = monitor.get_classification_stats()
    class_acc = monitor.get_class_accuracy()
    confusion_matrix = monitor.get_confusion_matrix()
    
    # 找出表现最差的类别
    worst_classes = sorted(class_acc.items(), key=lambda x: x[1])[:5]
    
    # 找出表现最好的类别
    best_classes = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)[:5]
    
    analysis = {
        'overall_accuracy': stats.accuracy,
        'top5_accuracy': stats.top5_accuracy,
        'avg_loss': stats.avg_loss,
        'total_samples': stats.total_samples,
        'worst_classes': worst_classes,
        'best_classes': best_classes,
        'class_accuracy': class_acc
    }
    
    if confusion_matrix is not None:
        # 计算精确率和召回率
        precision = {}
        recall = {}
        
        for i in range(confusion_matrix.size(0)):
            tp = confusion_matrix[i, i].item()
            fp = confusion_matrix[:, i].sum().item() - tp
            fn = confusion_matrix[i, :].sum().item() - tp
            
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        analysis['precision'] = precision
        analysis['recall'] = recall
    
    return analysis


def print_classification_analysis(
    monitor: SupervisedMonitor,
    class_names: Optional[List[str]] = None
) -> None:
    """
    打印分类分析
    
    Args:
        monitor: 监控器
        class_names: 类别名称
    """
    analysis = analyze_classification_results(monitor, class_names)
    
    print("\n" + "="*80)
    print("Classification Analysis")
    print("="*80)
    
    print(f"\nOverall Performance:")
    print(f"  Accuracy: {analysis['overall_accuracy']:.4f}")
    print(f"  Top-5 Accuracy: {analysis['top5_accuracy']:.4f}")
    print(f"  Avg Loss: {analysis['avg_loss']:.6f}")
    print(f"  Total Samples: {analysis['total_samples']:,}")
    
    print(f"\nWorst Performing Classes:")
    for cls, acc in analysis['worst_classes']:
        name = class_names[cls] if class_names and cls < len(class_names) else f"Class {cls}"
        print(f"  {name}: {acc:.4f}")
    
    print(f"\nBest Performing Classes:")
    for cls, acc in analysis['best_classes']:
        name = class_names[cls] if class_names and cls < len(class_names) else f"Class {cls}"
        print(f"  {name}: {acc:.4f}")
    
    if 'precision' in analysis:
        avg_precision = sum(analysis['precision'].values()) / len(analysis['precision'])
        avg_recall = sum(analysis['recall'].values()) / len(analysis['recall'])
        print(f"\nAvg Precision: {avg_precision:.4f}")
        print(f"Avg Recall: {avg_recall:.4f}")
    
    print("="*80)


def recommend_loss_function(
    task: str,
    class_imbalance: bool = False,
    noise_level: float = 0.0,
    num_classes: Optional[int] = None
) -> Dict[str, Any]:
    """
    推荐损失函数
    
    Args:
        task: 任务类型 (classification, regression, segmentation)
        class_imbalance: 是否有类别不平衡
        noise_level: 噪声水平 (0-1)
        num_classes: 类别数
        
    Returns:
        推荐配置
    """
    recommendations = {
        'loss_type': None,
        'params': {},
        'explanation': []
    }
    
    if task == 'classification':
        if class_imbalance:
            recommendations['loss_type'] = 'focal'
            recommendations['params'] = {'alpha': 0.25, 'gamma': 2.0}
            recommendations['explanation'].append("Focal loss recommended for class imbalance")
        elif noise_level > 0.1:
            recommendations['loss_type'] = 'label_smoothing'
            recommendations['params'] = {'smoothing': min(0.3, noise_level * 2)}
            recommendations['explanation'].append("Label smoothing recommended for noisy labels")
        else:
            recommendations['loss_type'] = 'cross_entropy'
            recommendations['params'] = {}
            recommendations['explanation'].append("Standard cross entropy for clean, balanced data")
    
    elif task == 'regression':
        if noise_level > 0.2:
            recommendations['loss_type'] = 'huber'
            recommendations['params'] = {'delta': 1.0}
            recommendations['explanation'].append("Huber loss recommended for outlier robustness")
        else:
            recommendations['loss_type'] = 'mse'
            recommendations['params'] = {}
            recommendations['explanation'].append("MSE for clean regression data")
    
    elif task == 'segmentation':
        if class_imbalance:
            recommendations['loss_type'] = 'dice'
            recommendations['params'] = {}
            recommendations['explanation'].append("Dice loss handles class imbalance in segmentation")
        else:
            recommendations['loss_type'] = 'iou'
            recommendations['params'] = {}
            recommendations['explanation'].append("IoU loss for balanced segmentation")
    
    return recommendations


def print_loss_recommendation(
    task: str,
    class_imbalance: bool = False,
    noise_level: float = 0.0,
    num_classes: Optional[int] = None
) -> None:
    """
    打印损失函数推荐
    
    Args:
        task: 任务类型
        class_imbalance: 是否有类别不平衡
        noise_level: 噪声水平
        num_classes: 类别数
    """
    rec = recommend_loss_function(task, class_imbalance, noise_level, num_classes)
    
    print("\n" + "="*60)
    print("Loss Function Recommendation")
    print("="*60)
    
    print(f"\nTask: {task}")
    print(f"Class imbalance: {class_imbalance}")
    print(f"Noise level: {noise_level:.2f}")
    if num_classes:
        print(f"Number of classes: {num_classes}")
    
    print(f"\nRecommended Loss: {rec['loss_type']}")
    if rec['params']:
        print(f"Parameters: {rec['params']}")
    
    print(f"\nExplanation:")
    for exp in rec['explanation']:
        print(f"  - {exp}")
    
    print("="*60)



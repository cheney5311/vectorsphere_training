# -*- coding: utf-8 -*-
"""
对比学习损失函数

包含各种对比学习相关的损失函数。
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
    BaseLoss, LossConfig, LossResult, LossMonitor, LossStats,
    register_loss, reduce_loss, weighted_loss
)

logger = logging.getLogger(__name__)


# ==================== 监控和统计组件 ====================

@dataclass
class ContrastiveStats:
    """对比学习统计"""
    total_steps: int = 0
    avg_pos_sim: float = 0.0
    avg_neg_sim: float = 0.0
    avg_accuracy: float = 0.0
    avg_temperature: float = 0.07
    hard_negatives_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_steps': self.total_steps,
            'avg_pos_sim': self.avg_pos_sim,
            'avg_neg_sim': self.avg_neg_sim,
            'avg_accuracy': self.avg_accuracy,
            'avg_temperature': self.avg_temperature,
            'hard_negatives_ratio': self.hard_negatives_ratio,
        }


class ContrastiveMonitor:
    """对比学习监控器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._history: List[Dict[str, float]] = []
        self._stats = ContrastiveStats()
        
        # 累计统计
        self._total_pos_sim = 0.0
        self._total_neg_sim = 0.0
        self._total_accuracy = 0.0
        self._total_temperature = 0.0
        self._total_hard_neg_ratio = 0.0
    
    def record(
        self, 
        pos_sim: float, 
        neg_sim: float, 
        accuracy: float = 0.0,
        temperature: float = 0.07,
        hard_neg_ratio: float = 0.0
    ) -> None:
        """记录统计"""
        record = {
            'pos_sim': pos_sim,
            'neg_sim': neg_sim,
            'accuracy': accuracy,
            'temperature': temperature,
            'hard_neg_ratio': hard_neg_ratio,
            'timestamp': time.time()
        }
        
        self._history.append(record)
        if len(self._history) > self.max_history:
            self._history.pop(0)
        
        # 更新累计统计
        self._stats.total_steps += 1
        self._total_pos_sim += pos_sim
        self._total_neg_sim += neg_sim
        self._total_accuracy += accuracy
        self._total_temperature += temperature
        self._total_hard_neg_ratio += hard_neg_ratio
        
        # 更新平均
        n = self._stats.total_steps
        self._stats.avg_pos_sim = self._total_pos_sim / n
        self._stats.avg_neg_sim = self._total_neg_sim / n
        self._stats.avg_accuracy = self._total_accuracy / n
        self._stats.avg_temperature = self._total_temperature / n
        self._stats.hard_negatives_ratio = self._total_hard_neg_ratio / n
    
    def get_stats(self) -> ContrastiveStats:
        """获取统计"""
        return self._stats
    
    def get_recent(self, n: int = 10) -> List[Dict[str, float]]:
        """获取最近的记录"""
        return self._history[-n:]
    
    def get_similarity_gap(self) -> float:
        """获取正负样本相似度差距"""
        return self._stats.avg_pos_sim - self._stats.avg_neg_sim
    
    def is_learning_effective(self, min_gap: float = 0.1) -> bool:
        """检查学习是否有效"""
        return self.get_similarity_gap() > min_gap
    
    def reset(self) -> None:
        """重置"""
        self._history.clear()
        self._stats = ContrastiveStats()
        self._total_pos_sim = 0.0
        self._total_neg_sim = 0.0
        self._total_accuracy = 0.0
        self._total_temperature = 0.0
        self._total_hard_neg_ratio = 0.0


class HardNegativeMiner:
    """硬负样本挖掘器"""
    
    def __init__(
        self, 
        strategy: str = 'semi_hard',  # hard, semi_hard, all
        margin: float = 0.2
    ):
        self.strategy = strategy
        self.margin = margin
    
    def mine(
        self, 
        anchor: Tensor, 
        positive: Tensor, 
        negatives: Tensor,
        pos_dist: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        挖掘硬负样本
        
        Args:
            anchor: anchor特征 [N, D]
            positive: 正样本特征 [N, D]
            negatives: 负样本特征 [K, D]
            pos_dist: 正样本距离 [N]（可选）
            
        Returns:
            (hard_negatives, indices): 硬负样本和索引
        """
        # 计算所有负样本距离
        neg_dist = torch.cdist(anchor, negatives)  # [N, K]
        
        if pos_dist is None:
            pos_dist = torch.sqrt(((anchor - positive) ** 2).sum(dim=-1))  # [N]
        
        if self.strategy == 'hard':
            # 选择最近的负样本
            hard_indices = neg_dist.argmin(dim=1)  # [N]
            hard_negatives = negatives[hard_indices]
        
        elif self.strategy == 'semi_hard':
            # 选择比正样本远但在margin内的负样本
            pos_dist_expanded = pos_dist.unsqueeze(1)  # [N, 1]
            
            # 找到semi-hard范围内的负样本
            mask = (neg_dist > pos_dist_expanded) & (neg_dist < pos_dist_expanded + self.margin)
            
            # 如果没有semi-hard负样本，退回到hard
            valid_counts = mask.sum(dim=1)
            
            hard_indices = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
            
            for i in range(anchor.size(0)):
                if valid_counts[i] > 0:
                    valid_idx = mask[i].nonzero(as_tuple=True)[0]
                    hard_indices[i] = valid_idx[torch.randint(len(valid_idx), (1,))]
                else:
                    hard_indices[i] = neg_dist[i].argmin()
            
            hard_negatives = negatives[hard_indices]
        
        else:  # all
            hard_indices = torch.arange(negatives.size(0), device=negatives.device)
            hard_negatives = negatives
        
        return hard_negatives, hard_indices
    
    def compute_hard_ratio(self, neg_dist: Tensor, pos_dist: Tensor) -> float:
        """计算硬负样本比例"""
        hard_mask = neg_dist < pos_dist.unsqueeze(1)
        return hard_mask.float().mean().item()


class TemperatureScheduler:
    """温度调度器"""
    
    def __init__(
        self, 
        initial_temp: float = 0.07,
        min_temp: float = 0.01,
        max_temp: float = 1.0,
        schedule: str = 'constant'  # constant, linear, cosine, adaptive
    ):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.schedule = schedule
        self._step = 0
        self._warmup_steps = 1000
        self._total_steps = 10000
        
        # 自适应调度的历史
        self._accuracy_history: List[float] = []
    
    def step(self, accuracy: Optional[float] = None) -> float:
        """
        获取当前温度并更新步数
        
        Args:
            accuracy: 当前准确率（用于自适应调度）
            
        Returns:
            当前温度
        """
        self._step += 1
        
        if accuracy is not None:
            self._accuracy_history.append(accuracy)
        
        return self.get_temperature()
    
    def get_temperature(self) -> float:
        """获取当前温度"""
        if self.schedule == 'constant':
            return self.initial_temp
        
        elif self.schedule == 'linear':
            # 线性衰减
            progress = min(self._step / self._total_steps, 1.0)
            return self.initial_temp - (self.initial_temp - self.min_temp) * progress
        
        elif self.schedule == 'cosine':
            # 余弦退火
            progress = min(self._step / self._total_steps, 1.0)
            return self.min_temp + (self.initial_temp - self.min_temp) * (1 + math.cos(progress * math.pi)) / 2
        
        elif self.schedule == 'adaptive':
            # 自适应：根据准确率调整
            if len(self._accuracy_history) < 10:
                return self.initial_temp
            
            recent_acc = sum(self._accuracy_history[-10:]) / 10
            
            # 准确率高时降低温度（更锐利的分布）
            if recent_acc > 0.8:
                target_temp = self.min_temp
            elif recent_acc < 0.3:
                target_temp = self.max_temp
            else:
                # 线性插值
                target_temp = self.max_temp - (recent_acc - 0.3) / 0.5 * (self.max_temp - self.min_temp)
            
            # 平滑过渡
            current = self.initial_temp if self._step == 1 else self._accuracy_history[-1] if self._accuracy_history else self.initial_temp
            return 0.9 * current + 0.1 * target_temp
        
        return self.initial_temp
    
    def set_total_steps(self, total_steps: int) -> None:
        """设置总步数"""
        self._total_steps = total_steps
    
    def reset(self) -> None:
        """重置"""
        self._step = 0
        self._accuracy_history.clear()




class ContrastiveLoss(BaseLoss):
    """
    对比学习损失基类
    
    提供对比学习损失的通用功能。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        temperature: float = 0.07,
        temperature_schedule: str = 'constant',
        hard_negative_mining: bool = False,
        mining_strategy: str = 'semi_hard',
        **kwargs
    ):
        super().__init__(config)
        self.temperature = config.temperature if config else temperature
        
        # 新增：温度调度器
        self._temp_scheduler = TemperatureScheduler(
            initial_temp=self.temperature,
            schedule=temperature_schedule
        )
        
        # 新增：硬负样本挖掘
        self._hard_mining = hard_negative_mining
        self._miner = HardNegativeMiner(strategy=mining_strategy) if hard_negative_mining else None
        
        # 新增：监控器
        self._contrastive_monitor = ContrastiveMonitor()
        self._ct_step = 0
    
    def get_effective_temperature(self) -> float:
        """获取有效温度"""
        return self._temp_scheduler.get_temperature()
    
    def set_temperature_schedule(
        self, 
        schedule: str, 
        total_steps: int = 10000
    ) -> None:
        """设置温度调度"""
        self._temp_scheduler.schedule = schedule
        self._temp_scheduler.set_total_steps(total_steps)
    
    def enable_hard_mining(self, strategy: str = 'semi_hard', margin: float = 0.2) -> None:
        """启用硬负样本挖掘"""
        self._hard_mining = True
        self._miner = HardNegativeMiner(strategy=strategy, margin=margin)
    
    def disable_hard_mining(self) -> None:
        """禁用硬负样本挖掘"""
        self._hard_mining = False
        self._miner = None
    
    def _compute_similarity(
        self, 
        x: Tensor, 
        y: Tensor, 
        normalize: bool = True
    ) -> Tensor:
        """
        计算相似度
        
        Args:
            x: 特征 [N, D] 或 [N, D]
            y: 特征 [M, D] 或 [N, D]
            normalize: 是否归一化
            
        Returns:
            相似度矩阵 [N, M] 或向量 [N]
        """
        if normalize:
            x = F.normalize(x, dim=-1)
            y = F.normalize(y, dim=-1)
        
        if x.dim() == 2 and y.dim() == 2:
            if x.size(0) == y.size(0) and x.size() == y.size():
                # 逐元素相似度
                return (x * y).sum(dim=-1)
            else:
                # 矩阵相似度
                return torch.mm(x, y.t())
        else:
            return (x * y).sum(dim=-1)
    
    def _compute_contrastive_accuracy(
        self, 
        logits: Tensor, 
        labels: Tensor
    ) -> float:
        """计算对比准确率"""
        preds = logits.argmax(dim=-1)
        return (preds == labels).float().mean().item()
    
    def _record_stats(
        self, 
        pos_sim: float, 
        neg_sim: float, 
        accuracy: float = 0.0,
        hard_neg_ratio: float = 0.0
    ) -> None:
        """记录统计"""
        temp = self.get_effective_temperature()
        self._contrastive_monitor.record(
            pos_sim=pos_sim,
            neg_sim=neg_sim,
            accuracy=accuracy,
            temperature=temp,
            hard_neg_ratio=hard_neg_ratio
        )
        self._ct_step += 1
    
    def get_contrastive_stats(self) -> ContrastiveStats:
        """获取对比学习统计"""
        return self._contrastive_monitor.get_stats()
    
    def get_similarity_gap(self) -> float:
        """获取正负样本相似度差距"""
        return self._contrastive_monitor.get_similarity_gap()
    
    def is_learning_effective(self, min_gap: float = 0.1) -> bool:
        """检查学习是否有效"""
        return self._contrastive_monitor.is_learning_effective(min_gap)
    
    def print_summary(self) -> None:
        """打印摘要"""
        stats = self.get_contrastive_stats()
        
        print("\n" + "="*80)
        print(f"Contrastive Loss Summary: {self.__class__.__name__}")
        print("="*80)
        
        print(f"\nTemperature: {self.get_effective_temperature():.4f}")
        print(f"Temperature schedule: {self._temp_scheduler.schedule}")
        print(f"Hard negative mining: {self._hard_mining}")
        
        print(f"\nStatistics (over {stats.total_steps} steps):")
        print(f"  Avg positive similarity: {stats.avg_pos_sim:.4f}")
        print(f"  Avg negative similarity: {stats.avg_neg_sim:.4f}")
        print(f"  Similarity gap: {self.get_similarity_gap():.4f}")
        print(f"  Avg accuracy: {stats.avg_accuracy:.4f}")
        
        if self._hard_mining:
            print(f"  Hard negatives ratio: {stats.hard_negatives_ratio:.4f}")
        
        print(f"\nLearning effective: {self.is_learning_effective()}")
        
        print("="*80)
    
    def reset_contrastive_stats(self) -> None:
        """重置统计"""
        self._contrastive_monitor.reset()
        self._temp_scheduler.reset()
        self._ct_step = 0


# ==================== InfoNCE损失 ====================

@register_loss("infonce")
class InfoNCELoss(ContrastiveLoss):
    """
    InfoNCE损失
    
    经典的对比学习损失，用于自监督和跨模态学习。
    L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
    
    Reference: Oord et al., "Representation Learning with Contrastive Predictive Coding"
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        temperature: float = 0.07,
        **kwargs
    ):
        super().__init__(config, temperature)
    
    def forward(
        self, 
        predictions: Tensor,  # anchor features [N, D]
        targets: Tensor,      # positive features [N, D]
        negatives: Optional[Tensor] = None,  # negative features [K, D]
        **kwargs
    ) -> Tensor:
        """
        计算InfoNCE损失
        
        Args:
            predictions: anchor特征 [N, D]
            targets: 正样本特征 [N, D]
            negatives: 负样本特征 [K, D]（可选，默认使用batch内其他样本）
        """
        # 归一化
        anchor = F.normalize(predictions, dim=-1)
        positive = F.normalize(targets, dim=-1)
        
        batch_size = anchor.size(0)
        
        # 计算正样本相似度
        pos_sim = (anchor * positive).sum(dim=-1) / self.temperature  # [N]
        
        # 计算负样本相似度
        if negatives is not None:
            # 使用提供的负样本
            negatives = F.normalize(negatives, dim=-1)
            neg_sim = torch.mm(anchor, negatives.t()) / self.temperature  # [N, K]
        else:
            # 使用batch内其他样本作为负样本
            neg_sim = torch.mm(anchor, positive.t()) / self.temperature  # [N, N]
            # 移除对角线（正样本）
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=neg_sim.device)
            neg_sim = neg_sim[mask].view(batch_size, -1)  # [N, N-1]
        
        # 拼接正负样本相似度
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [N, 1+K]
        
        # 标签：正样本在第0位
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def compute(
        self, 
        predictions: Tensor,
        targets: Tensor,
        negatives: Optional[Tensor] = None,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        # 归一化
        anchor = F.normalize(predictions, dim=-1)
        positive = F.normalize(targets, dim=-1)
        
        batch_size = anchor.size(0)
        temp = self.get_effective_temperature()
        
        # 计算正样本相似度（未缩放）
        pos_sim_raw = (anchor * positive).sum(dim=-1)  # [N]
        pos_sim = pos_sim_raw / temp  # [N]
        
        # 计算负样本相似度
        if negatives is not None:
            negatives = F.normalize(negatives, dim=-1)
            neg_sim_raw = torch.mm(anchor, negatives.t())  # [N, K]
            neg_sim = neg_sim_raw / temp
        else:
            neg_sim_raw = torch.mm(anchor, positive.t())  # [N, N]
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=neg_sim_raw.device)
            neg_sim_raw = neg_sim_raw[mask].view(batch_size, -1)
            neg_sim = neg_sim_raw / temp
        
        # 硬负样本挖掘
        hard_neg_ratio = 0.0
        if self._hard_mining and self._miner is not None and negatives is not None:
            pos_dist = 1 - pos_sim_raw
            neg_dist = 1 - neg_sim_raw
            hard_neg_ratio = self._miner.compute_hard_ratio(neg_dist, pos_dist)
        
        # 拼接正负样本相似度
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        # 计算准确率
        accuracy = self._compute_contrastive_accuracy(logits, labels)
        
        # 记录统计
        self._record_stats(
            pos_sim=pos_sim_raw.mean().item(),
            neg_sim=neg_sim_raw.mean().item(),
            accuracy=accuracy,
            hard_neg_ratio=hard_neg_ratio
        )
        
        # 更新温度调度
        self._temp_scheduler.step(accuracy)
        
        return LossResult(
            loss=loss * self.config.weight,
            components={
                'infonce_loss': loss,
                'pos_sim': pos_sim_raw.mean(),
                'neg_sim': neg_sim_raw.mean()
            },
            metrics={
                'loss': loss.item(),
                'pos_similarity': pos_sim_raw.mean().item(),
                'neg_similarity': neg_sim_raw.mean().item(),
                'similarity_gap': (pos_sim_raw.mean() - neg_sim_raw.mean()).item(),
                'accuracy': accuracy,
                'temperature': temp,
                'hard_neg_ratio': hard_neg_ratio
            },
            step=self._ct_step
        )


# ==================== NT-Xent损失 ====================

@register_loss("nt_xent")
class NTXentLoss(ContrastiveLoss):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) 损失
    
    SimCLR使用的对比学习损失。
    
    Reference: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations"
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        temperature: float = 0.5,
        **kwargs
    ):
        super().__init__(config, temperature)
    
    def forward(
        self, 
        predictions: Tensor,  # view 1 features [N, D]
        targets: Tensor,      # view 2 features [N, D]
        **kwargs
    ) -> Tensor:
        """
        计算NT-Xent损失
        
        对于每个样本，它的两个增强视图互为正样本，batch内其他样本为负样本。
        
        Args:
            predictions: 视图1特征 [N, D]
            targets: 视图2特征 [N, D]
        """
        batch_size = predictions.size(0)
        
        # 归一化
        z_i = F.normalize(predictions, dim=-1)
        z_j = F.normalize(targets, dim=-1)
        
        # 拼接两个视图
        representations = torch.cat([z_i, z_j], dim=0)  # [2N, D]
        
        # 计算相似度矩阵
        similarity = torch.mm(representations, representations.t()) / self.temperature  # [2N, 2N]
        
        # 创建标签：对角线偏移N
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(similarity.device)
        
        # 移除自身相似度
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=similarity.device)
        similarity = similarity[mask].view(2 * batch_size, -1)  # [2N, 2N-1]
        
        # 调整标签
        labels = labels - (labels > torch.arange(2 * batch_size, device=labels.device)).long()
        
        loss = F.cross_entropy(similarity, labels)
        
        return loss
    
    def compute(
        self, 
        predictions: Tensor,
        targets: Tensor,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        batch_size = predictions.size(0)
        temp = self.get_effective_temperature()
        
        # 归一化
        z_i = F.normalize(predictions, dim=-1)
        z_j = F.normalize(targets, dim=-1)
        
        # 计算正样本相似度（对角线偏移N的位置）
        pos_sim_raw = (z_i * z_j).sum(dim=-1)  # [N]
        
        # 拼接两个视图
        representations = torch.cat([z_i, z_j], dim=0)  # [2N, D]
        
        # 计算相似度矩阵（未缩放）
        similarity_raw = torch.mm(representations, representations.t())  # [2N, 2N]
        similarity = similarity_raw / temp
        
        # 计算负样本平均相似度（排除正样本和自身）
        neg_mask = torch.ones(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=similarity.device)
        neg_mask.fill_diagonal_(False)
        for i in range(batch_size):
            neg_mask[i, i + batch_size] = False
            neg_mask[i + batch_size, i] = False
        neg_sim_raw = similarity_raw[neg_mask].mean()
        
        # 创建标签
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ]).to(similarity.device)
        
        # 移除自身相似度
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=similarity.device)
        similarity_masked = similarity[mask].view(2 * batch_size, -1)
        
        # 调整标签
        labels = labels - (labels > torch.arange(2 * batch_size, device=labels.device)).long()
        
        loss = F.cross_entropy(similarity_masked, labels)
        
        # 计算准确率
        accuracy = self._compute_contrastive_accuracy(similarity_masked, labels)
        
        # 记录统计
        self._record_stats(
            pos_sim=pos_sim_raw.mean().item(),
            neg_sim=neg_sim_raw.item(),
            accuracy=accuracy
        )
        
        # 更新温度调度
        self._temp_scheduler.step(accuracy)
        
        return LossResult(
            loss=loss * self.config.weight,
            components={
                'ntxent_loss': loss,
                'pos_sim': pos_sim_raw.mean(),
                'neg_sim': neg_sim_raw
            },
            metrics={
                'loss': loss.item(),
                'pos_similarity': pos_sim_raw.mean().item(),
                'neg_similarity': neg_sim_raw.item(),
                'similarity_gap': (pos_sim_raw.mean() - neg_sim_raw).item(),
                'accuracy': accuracy,
                'temperature': temp
            },
            step=self._ct_step
        )


# ==================== Triplet损失 ====================

@register_loss("triplet")
class TripletLoss(ContrastiveLoss):
    """
    Triplet损失
    
    确保anchor与positive的距离小于与negative的距离。
    L = max(d(a, p) - d(a, n) + margin, 0)
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        margin: float = 1.0,
        distance: str = "euclidean",  # euclidean, cosine
        **kwargs
    ):
        super().__init__(config)
        self.margin = margin
        self.distance = distance
    
    def _compute_distance(self, x: Tensor, y: Tensor) -> Tensor:
        """计算距离"""
        if self.distance == "euclidean":
            return torch.sqrt(((x - y) ** 2).sum(dim=-1) + 1e-8)
        elif self.distance == "cosine":
            return 1 - torch.cosine_similarity(x, y, dim=-1)
        else:
            return torch.sqrt(((x - y) ** 2).sum(dim=-1) + 1e-8)
    
    def forward(
        self, 
        predictions: Tensor,  # anchor [N, D]
        targets: Tensor,      # positive [N, D]
        negatives: Tensor,    # negative [N, D]
        **kwargs
    ) -> Tensor:
        """
        计算Triplet损失
        
        Args:
            predictions: anchor特征 [N, D]
            targets: 正样本特征 [N, D]
            negatives: 负样本特征 [N, D]
        """
        d_pos = self._compute_distance(predictions, targets)
        d_neg = self._compute_distance(predictions, negatives)
        
        loss = F.relu(d_pos - d_neg + self.margin)
        
        return reduce_loss(loss, self.config.reduction)
    
    def compute(
        self, 
        predictions: Tensor,
        targets: Tensor,
        negatives: Tensor,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        d_pos = self._compute_distance(predictions, targets)
        d_neg = self._compute_distance(predictions, negatives)
        
        # 计算三元组损失
        triplet_loss = F.relu(d_pos - d_neg + self.margin)
        loss = reduce_loss(triplet_loss, self.config.reduction)
        
        # 硬负样本挖掘
        hard_neg_ratio = 0.0
        if self._hard_mining and self._miner is not None:
            hard_neg_ratio = self._miner.compute_hard_ratio(
                d_neg.unsqueeze(1), d_pos
            )
        
        # 计算有效三元组比例（损失大于0的）
        valid_triplets = (triplet_loss > 0).float().mean().item()
        
        # 计算相似度（转换为相似度而非距离）
        if self.distance == 'cosine':
            pos_sim = 1 - d_pos
            neg_sim = 1 - d_neg
        else:
            # 对于欧氏距离，使用负距离作为相似度代理
            pos_sim = -d_pos
            neg_sim = -d_neg
        
        # 记录统计
        self._record_stats(
            pos_sim=pos_sim.mean().item(),
            neg_sim=neg_sim.mean().item(),
            accuracy=valid_triplets,  # 使用有效三元组比例作为"准确率"
            hard_neg_ratio=hard_neg_ratio
        )
        
        return LossResult(
            loss=loss * self.config.weight,
            components={
                'triplet_loss': loss,
                'd_pos': d_pos.mean(),
                'd_neg': d_neg.mean()
            },
            metrics={
                'loss': loss.item(),
                'd_pos': d_pos.mean().item(),
                'd_neg': d_neg.mean().item(),
                'margin': self.margin,
                'valid_triplets_ratio': valid_triplets,
                'distance_gap': (d_neg - d_pos).mean().item(),
                'hard_neg_ratio': hard_neg_ratio
            },
            step=self._ct_step
        )


# ==================== Center损失 ====================

@register_loss("center")
class CenterLoss(ContrastiveLoss):
    """
    Center损失
    
    学习每个类别的中心，使同类样本更紧凑。
    
    Reference: Wen et al., "A Discriminative Feature Learning Approach for Deep Face Recognition"
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        num_classes: int = 10,
        feature_dim: int = 512,
        alpha: float = 0.5,  # 中心更新率
        **kwargs
    ):
        super().__init__(config)
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha
        
        # 类别中心
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))
        
        # 新增：类别统计
        self._class_counts: Dict[int, int] = defaultdict(int)
        self._intra_class_distances: Dict[int, List[float]] = defaultdict(list)
    
    def forward(
        self, 
        predictions: Tensor,  # features [N, D]
        targets: Tensor,      # labels [N]
        **kwargs
    ) -> Tensor:
        """
        计算Center损失
        
        Args:
            predictions: 特征 [N, D]
            targets: 类别标签 [N]
        """
        batch_size = predictions.size(0)
        
        # 获取每个样本对应的类别中心
        centers_batch = self.centers[targets]  # [N, D]
        
        # 计算到中心的距离
        distances = (predictions - centers_batch).pow(2).sum(dim=-1)  # [N]
        
        return reduce_loss(distances, self.config.reduction) / 2.0
    
    def compute(
        self, 
        predictions: Tensor,
        targets: Tensor,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        batch_size = predictions.size(0)
        
        # 获取每个样本对应的类别中心
        centers_batch = self.centers[targets]  # [N, D]
        
        # 计算到中心的距离
        distances = (predictions - centers_batch).pow(2).sum(dim=-1)  # [N]
        loss = reduce_loss(distances, self.config.reduction) / 2.0
        
        # 更新类别统计
        for i, label in enumerate(targets.tolist()):
            self._class_counts[label] += 1
            self._intra_class_distances[label].append(distances[i].item())
            
            # 限制历史长度
            if len(self._intra_class_distances[label]) > 100:
                self._intra_class_distances[label].pop(0)
        
        # 计算类间距离（中心之间的距离）
        center_distances = torch.cdist(self.centers, self.centers)
        inter_class_dist = center_distances[~torch.eye(self.num_classes, dtype=torch.bool)].mean()
        
        # 计算类内紧凑度（平均到中心距离）
        intra_class_dist = distances.mean()
        
        # 记录统计（使用距离的负值作为相似度）
        self._record_stats(
            pos_sim=-intra_class_dist.item(),  # 类内距离越小越好
            neg_sim=-inter_class_dist.item(),  # 类间距离越大越好
            accuracy=0.0  # Center loss不直接计算准确率
        )
        
        return LossResult(
            loss=loss * self.config.weight,
            components={
                'center_loss': loss,
                'intra_dist': intra_class_dist,
                'inter_dist': inter_class_dist
            },
            metrics={
                'loss': loss.item(),
                'intra_class_distance': intra_class_dist.item(),
                'inter_class_distance': inter_class_dist.item(),
                'compactness_ratio': (inter_class_dist / (intra_class_dist + 1e-8)).item(),
                'num_classes_seen': len(self._class_counts)
            },
            step=self._ct_step
        )
    
    def get_class_stats(self) -> Dict[int, Dict[str, float]]:
        """获取类别统计"""
        stats = {}
        for label in self._class_counts:
            distances = self._intra_class_distances.get(label, [])
            if distances:
                stats[label] = {
                    'count': self._class_counts[label],
                    'avg_distance': sum(distances) / len(distances),
                    'min_distance': min(distances),
                    'max_distance': max(distances)
                }
        return stats
    
    def get_center(self, class_id: int) -> Optional[Tensor]:
        """获取类别中心"""
        if 0 <= class_id < self.num_classes:
            return self.centers[class_id].detach()
        return None
    
    def reset_center_stats(self) -> None:
        """重置中心统计"""
        self._class_counts.clear()
        self._intra_class_distances.clear()


# ==================== 跨模态对比损失 ====================

@register_loss("cross_modal_contrastive")
class CrossModalContrastiveLoss(ContrastiveLoss):
    """
    跨模态对比损失
    
    用于多模态学习，对齐不同模态的表示。
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        temperature: float = 0.07,
        bidirectional: bool = True,
        **kwargs
    ):
        super().__init__(config, temperature)
        self.bidirectional = bidirectional
    
    def forward(
        self, 
        predictions: Tensor,  # 模态A特征 [N, D]
        targets: Tensor,      # 模态B特征 [N, D]
        **kwargs
    ) -> Tensor:
        """
        计算跨模态对比损失
        
        Args:
            predictions: 模态A特征 [N, D]
            targets: 模态B特征 [N, D]
        """
        # 归一化
        feat_a = F.normalize(predictions, dim=-1)
        feat_b = F.normalize(targets, dim=-1)
        
        batch_size = feat_a.size(0)
        
        # 计算相似度矩阵
        similarity = torch.mm(feat_a, feat_b.t()) / self.temperature  # [N, N]
        
        # 标签：对角线为正样本
        labels = torch.arange(batch_size, device=similarity.device)
        
        # A -> B
        loss_a2b = F.cross_entropy(similarity, labels)
        
        if self.bidirectional:
            # B -> A
            loss_b2a = F.cross_entropy(similarity.t(), labels)
            loss = (loss_a2b + loss_b2a) / 2
        else:
            loss = loss_a2b
        
        return loss
    
    def compute(
        self, 
        predictions: Tensor,
        targets: Tensor,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        temp = self.get_effective_temperature()
        
        # 归一化
        feat_a = F.normalize(predictions, dim=-1)
        feat_b = F.normalize(targets, dim=-1)
        
        batch_size = feat_a.size(0)
        
        # 计算相似度矩阵（未缩放）
        similarity_raw = torch.mm(feat_a, feat_b.t())
        similarity = similarity_raw / temp
        
        # 标签
        labels = torch.arange(batch_size, device=similarity.device)
        
        # 计算正负样本相似度
        pos_sim = similarity_raw.diag()  # 对角线是正样本
        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity_raw.device)
        neg_sim = similarity_raw[neg_mask].view(batch_size, -1)
        
        # A -> B
        loss_a2b = F.cross_entropy(similarity, labels)
        acc_a2b = self._compute_contrastive_accuracy(similarity, labels)
        
        if self.bidirectional:
            # B -> A
            loss_b2a = F.cross_entropy(similarity.t(), labels)
            acc_b2a = self._compute_contrastive_accuracy(similarity.t(), labels)
            loss = (loss_a2b + loss_b2a) / 2
            accuracy = (acc_a2b + acc_b2a) / 2
        else:
            loss = loss_a2b
            accuracy = acc_a2b
            loss_b2a = torch.tensor(0.0)
            acc_b2a = 0.0
        
        # 记录统计
        self._record_stats(
            pos_sim=pos_sim.mean().item(),
            neg_sim=neg_sim.mean().item(),
            accuracy=accuracy
        )
        
        # 更新温度调度
        self._temp_scheduler.step(accuracy)
        
        return LossResult(
            loss=loss * self.config.weight,
            components={
                'loss_a2b': loss_a2b,
                'loss_b2a': loss_b2a if self.bidirectional else torch.tensor(0.0),
                'pos_sim': pos_sim.mean(),
                'neg_sim': neg_sim.mean()
            },
            metrics={
                'loss': loss.item(),
                'loss_a2b': loss_a2b.item(),
                'loss_b2a': loss_b2a.item() if self.bidirectional else 0.0,
                'acc_a2b': acc_a2b,
                'acc_b2a': acc_b2a if self.bidirectional else 0.0,
                'accuracy': accuracy,
                'pos_similarity': pos_sim.mean().item(),
                'neg_similarity': neg_sim.mean().item(),
                'similarity_gap': (pos_sim.mean() - neg_sim.mean()).item(),
                'temperature': temp
            },
            step=self._ct_step
        )


# ==================== CLIP损失 ====================

@register_loss("clip")
class CLIPLoss(ContrastiveLoss):
    """
    CLIP损失
    
    图文对比学习损失。
    
    Reference: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision"
    """
    
    def __init__(
        self, 
        config: Optional[LossConfig] = None,
        temperature: float = 0.07,
        learnable_temperature: bool = False,
        **kwargs
    ):
        super().__init__(config, temperature)
        
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(temperature)))
        else:
            self.register_buffer('log_temperature', torch.log(torch.tensor(temperature)))
    
    @property
    def effective_temperature(self) -> Tensor:
        """获取有效温度"""
        return torch.exp(self.log_temperature)
    
    def forward(
        self, 
        predictions: Tensor,  # image features [N, D]
        targets: Tensor,      # text features [N, D]
        **kwargs
    ) -> Tensor:
        """
        计算CLIP损失
        
        Args:
            predictions: 图像特征 [N, D]
            targets: 文本特征 [N, D]
        """
        # 归一化
        image_features = F.normalize(predictions, dim=-1)
        text_features = F.normalize(targets, dim=-1)
        
        batch_size = image_features.size(0)
        
        # 计算相似度
        T = self.effective_temperature
        logits_per_image = torch.mm(image_features, text_features.t()) / T  # [N, N]
        logits_per_text = logits_per_image.t()
        
        # 标签
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        # 双向损失
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def compute(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        **kwargs
    ) -> LossResult:
        """计算并返回详细结果"""
        image_features = F.normalize(predictions, dim=-1)
        text_features = F.normalize(targets, dim=-1)
        
        batch_size = image_features.size(0)
        
        T = self.effective_temperature
        
        # 计算相似度（未缩放）
        similarity_raw = torch.mm(image_features, text_features.t())
        
        logits_per_image = similarity_raw / T
        logits_per_text = logits_per_image.t()
        
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        total_loss = (loss_i2t + loss_t2i) / 2
        
        # 计算准确率
        i2t_acc = (logits_per_image.argmax(dim=-1) == labels).float().mean().item()
        t2i_acc = (logits_per_text.argmax(dim=-1) == labels).float().mean().item()
        avg_acc = (i2t_acc + t2i_acc) / 2
        
        # 计算正负样本相似度
        pos_sim = similarity_raw.diag()
        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity_raw.device)
        neg_sim = similarity_raw[neg_mask].view(batch_size, -1)
        
        # 记录统计
        self._record_stats(
            pos_sim=pos_sim.mean().item(),
            neg_sim=neg_sim.mean().item(),
            accuracy=avg_acc
        )
        
        # 更新温度调度
        self._temp_scheduler.step(avg_acc)
        
        return LossResult(
            loss=total_loss * self.config.weight,
            components={
                'loss_i2t': loss_i2t,
                'loss_t2i': loss_t2i,
                'pos_sim': pos_sim.mean(),
                'neg_sim': neg_sim.mean()
            },
            metrics={
                'loss': total_loss.item(),
                'loss_i2t': loss_i2t.item(),
                'loss_t2i': loss_t2i.item(),
                'temperature': T.item(),
                'i2t_acc': i2t_acc,
                't2i_acc': t2i_acc,
                'accuracy': avg_acc,
                'pos_similarity': pos_sim.mean().item(),
                'neg_similarity': neg_sim.mean().item(),
                'similarity_gap': (pos_sim.mean() - neg_sim.mean()).item()
            },
            step=self._ct_step
        )
    
    def get_temperature_value(self) -> float:
        """获取当前温度值"""
        return self.effective_temperature.item()
    
    def set_temperature(self, temperature: float) -> None:
        """设置温度"""
        with torch.no_grad():
            self.log_temperature.fill_(math.log(temperature))


# ==================== 工具函数 ====================

def create_contrastive_loss(
    loss_type: str,
    temperature: float = 0.07,
    temperature_schedule: str = 'constant',
    hard_negative_mining: bool = False,
    **kwargs
) -> ContrastiveLoss:
    """
    创建对比学习损失
    
    Args:
        loss_type: 损失类型 (infonce, nt_xent, triplet, center, cross_modal, clip)
        temperature: 温度
        temperature_schedule: 温度调度
        hard_negative_mining: 是否启用硬负样本挖掘
        **kwargs: 额外参数
        
    Returns:
        对比学习损失实例
    """
    loss_classes = {
        'infonce': InfoNCELoss,
        'nt_xent': NTXentLoss,
        'triplet': TripletLoss,
        'center': CenterLoss,
        'cross_modal': CrossModalContrastiveLoss,
        'clip': CLIPLoss,
    }
    
    if loss_type not in loss_classes:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_classes.keys())}")
    
    loss_class = loss_classes[loss_type]
    
    # 特殊处理温度调度
    if loss_type in ('infonce', 'nt_xent', 'cross_modal', 'clip'):
        return loss_class(
            temperature=temperature,
            temperature_schedule=temperature_schedule,
            hard_negative_mining=hard_negative_mining,
            **kwargs
        )
    else:
        return loss_class(**kwargs)


def compare_contrastive_losses(losses: Dict[str, ContrastiveLoss]) -> None:
    """
    对比多个对比学习损失
    
    Args:
        losses: 损失字典
    """
    print("\n" + "="*100)
    print("Contrastive Loss Comparison")
    print("="*100)
    
    print(f"\n{'Name':<20} {'Type':<25} {'Pos Sim':<12} {'Neg Sim':<12} {'Gap':<10} {'Accuracy':<10}")
    print("-"*100)
    
    for name, loss_fn in losses.items():
        loss_type = loss_fn.__class__.__name__
        stats = loss_fn.get_contrastive_stats()
        gap = loss_fn.get_similarity_gap()
        
        print(f"{name:<20} {loss_type:<25} {stats.avg_pos_sim:<12.4f} "
              f"{stats.avg_neg_sim:<12.4f} {gap:<10.4f} {stats.avg_accuracy:<10.4f}")
    
    print("="*100)


def analyze_feature_distribution(
    features: Tensor,
    labels: Optional[Tensor] = None
) -> Dict[str, Any]:
    """
    分析特征分布
    
    Args:
        features: 特征 [N, D]
        labels: 标签 [N]（可选）
        
    Returns:
        分析结果
    """
    features = F.normalize(features, dim=-1)
    
    # 计算特征统计
    mean = features.mean(dim=0)
    std = features.std(dim=0)
    
    # 计算相似度矩阵
    similarity = torch.mm(features, features.t())
    
    # 排除对角线
    n = features.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=similarity.device)
    off_diag_sim = similarity[mask]
    
    result = {
        'feature_dim': features.size(1),
        'num_samples': features.size(0),
        'feature_mean_norm': mean.norm().item(),
        'feature_std_mean': std.mean().item(),
        'avg_pairwise_similarity': off_diag_sim.mean().item(),
        'min_pairwise_similarity': off_diag_sim.min().item(),
        'max_pairwise_similarity': off_diag_sim.max().item(),
    }
    
    # 如果有标签，计算类内/类间相似度
    if labels is not None:
        intra_sim = []
        inter_sim = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    intra_sim.append(similarity[i, j].item())
                else:
                    inter_sim.append(similarity[i, j].item())
        
        if intra_sim:
            result['avg_intra_class_similarity'] = sum(intra_sim) / len(intra_sim)
        if inter_sim:
            result['avg_inter_class_similarity'] = sum(inter_sim) / len(inter_sim)
        
        if intra_sim and inter_sim:
            result['class_separation_ratio'] = (
                result['avg_intra_class_similarity'] / 
                (result['avg_inter_class_similarity'] + 1e-8)
            )
    
    return result


def print_feature_analysis(features: Tensor, labels: Optional[Tensor] = None) -> None:
    """
    打印特征分析
    
    Args:
        features: 特征
        labels: 标签
    """
    analysis = analyze_feature_distribution(features, labels)
    
    print("\n" + "="*60)
    print("Feature Distribution Analysis")
    print("="*60)
    
    print(f"\nBasic Statistics:")
    print(f"  Feature dimension: {analysis['feature_dim']}")
    print(f"  Number of samples: {analysis['num_samples']}")
    print(f"  Feature mean norm: {analysis['feature_mean_norm']:.4f}")
    print(f"  Feature std mean: {analysis['feature_std_mean']:.4f}")
    
    print(f"\nPairwise Similarity:")
    print(f"  Average: {analysis['avg_pairwise_similarity']:.4f}")
    print(f"  Min: {analysis['min_pairwise_similarity']:.4f}")
    print(f"  Max: {analysis['max_pairwise_similarity']:.4f}")
    
    if 'avg_intra_class_similarity' in analysis:
        print(f"\nClass-wise Similarity:")
        print(f"  Intra-class avg: {analysis['avg_intra_class_similarity']:.4f}")
        print(f"  Inter-class avg: {analysis['avg_inter_class_similarity']:.4f}")
        print(f"  Separation ratio: {analysis['class_separation_ratio']:.4f}")
    
    print("="*60)


def compute_retrieval_metrics(
    query_features: Tensor,
    key_features: Tensor,
    query_labels: Optional[Tensor] = None,
    key_labels: Optional[Tensor] = None,
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    计算检索指标
    
    Args:
        query_features: 查询特征 [N, D]
        key_features: 键特征 [M, D]
        query_labels: 查询标签 [N]
        key_labels: 键标签 [M]
        k_values: top-k值列表
        
    Returns:
        检索指标字典
    """
    # 归一化
    query_features = F.normalize(query_features, dim=-1)
    key_features = F.normalize(key_features, dim=-1)
    
    # 计算相似度
    similarity = torch.mm(query_features, key_features.t())  # [N, M]
    
    metrics = {}
    
    # 如果没有标签，假设query和key一一对应
    if query_labels is None or key_labels is None:
        labels = torch.arange(query_features.size(0), device=similarity.device)
    else:
        labels = query_labels
    
    for k in k_values:
        # Top-k准确率
        _, top_k_indices = similarity.topk(k, dim=1)
        
        if query_labels is None or key_labels is None:
            # 检查是否在top-k中命中对应位置
            correct = (top_k_indices == labels.unsqueeze(1)).any(dim=1)
        else:
            # 检查是否在top-k中命中相同标签
            correct = torch.zeros(query_features.size(0), dtype=torch.bool, device=similarity.device)
            for i in range(query_features.size(0)):
                correct[i] = (key_labels[top_k_indices[i]] == query_labels[i]).any()
        
        metrics[f'R@{k}'] = correct.float().mean().item()
    
    # 计算MRR (Mean Reciprocal Rank)
    _, sorted_indices = similarity.sort(dim=1, descending=True)
    
    mrr = 0.0
    for i in range(query_features.size(0)):
        if query_labels is None or key_labels is None:
            rank = (sorted_indices[i] == i).nonzero(as_tuple=True)[0]
        else:
            matches = key_labels[sorted_indices[i]] == query_labels[i]
            rank = matches.nonzero(as_tuple=True)[0]
        
        if len(rank) > 0:
            mrr += 1.0 / (rank[0].item() + 1)
    
    metrics['MRR'] = mrr / query_features.size(0)
    
    return metrics


def print_retrieval_metrics(
    query_features: Tensor,
    key_features: Tensor,
    query_labels: Optional[Tensor] = None,
    key_labels: Optional[Tensor] = None,
    k_values: List[int] = [1, 5, 10]
) -> None:
    """
    打印检索指标
    
    Args:
        query_features: 查询特征
        key_features: 键特征
        query_labels: 查询标签
        key_labels: 键标签
        k_values: top-k值列表
    """
    metrics = compute_retrieval_metrics(
        query_features, key_features,
        query_labels, key_labels,
        k_values
    )
    
    print("\n" + "="*60)
    print("Retrieval Metrics")
    print("="*60)
    
    for k in k_values:
        print(f"  R@{k}: {metrics[f'R@{k}']:.4f}")
    
    print(f"  MRR: {metrics['MRR']:.4f}")
    
    print("="*60)



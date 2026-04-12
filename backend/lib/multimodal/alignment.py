# -*- coding: utf-8 -*-
"""
跨模态对齐模块

实现多种跨模态对齐算法：
- 对比学习（CLIP风格）
- 显式对齐（Attention/MLP）
- 交叉注意力对齐
- 最优传输对齐
- 知识蒸馏对齐
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .multimodal_config import (
    AlignmentMethod,
    CrossModalAlignmentConfig,
    ContrastiveLearningConfig,
    ExplicitAlignConfig
)

logger = logging.getLogger(__name__)


# ==================== 数据类和枚举 ====================

class NegativeMiningStrategy(Enum):
    """负样本挖掘策略"""
    RANDOM = "random"           # 随机负样本
    HARD = "hard"               # 硬负样本
    SEMI_HARD = "semi_hard"     # 半硬负样本
    MIXED = "mixed"             # 混合策略


@dataclass
class AlignmentStats:
    """对齐统计信息"""
    total_steps: int = 0
    total_loss: float = 0.0
    avg_loss: float = 0.0
    min_loss: float = float('inf')
    max_loss: float = 0.0
    
    # 对比学习指标
    avg_accuracy_a: float = 0.0
    avg_accuracy_b: float = 0.0
    avg_temperature: float = 0.07
    
    # 对齐质量指标
    avg_similarity: float = 0.0
    avg_alignment_score: float = 0.0
    
    def update(self, loss: float, metrics: Dict[str, float]) -> None:
        """更新统计信息"""
        self.total_steps += 1
        self.total_loss += loss
        self.avg_loss = self.total_loss / self.total_steps
        self.min_loss = min(self.min_loss, loss)
        self.max_loss = max(self.max_loss, loss)
        
        if 'contrastive_acc_a' in metrics:
            n = self.total_steps
            self.avg_accuracy_a = (self.avg_accuracy_a * (n-1) + metrics['contrastive_acc_a']) / n
        if 'contrastive_acc_b' in metrics:
            n = self.total_steps
            self.avg_accuracy_b = (self.avg_accuracy_b * (n-1) + metrics['contrastive_acc_b']) / n
        if 'temperature' in metrics:
            self.avg_temperature = metrics['temperature']
        if 'similarity' in metrics:
            n = self.total_steps
            self.avg_similarity = (self.avg_similarity * (n-1) + metrics['similarity']) / n


# ==================== 监控组件 ====================

class AlignmentMonitor:
    """对齐监控器"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._loss_history: deque = deque(maxlen=history_size)
        self._accuracy_history: deque = deque(maxlen=history_size)
        self._similarity_history: deque = deque(maxlen=history_size)
        self._temperature_history: deque = deque(maxlen=history_size)
        self._stats = AlignmentStats()
        self._step = 0
    
    def record(self, loss: float, metrics: Dict[str, float]) -> None:
        """记录对齐指标"""
        self._step += 1
        self._loss_history.append(loss)
        self._stats.update(loss, metrics)
        
        if 'contrastive_acc_a' in metrics and 'contrastive_acc_b' in metrics:
            avg_acc = (metrics['contrastive_acc_a'] + metrics['contrastive_acc_b']) / 2
            self._accuracy_history.append(avg_acc)
        
        if 'similarity' in metrics:
            self._similarity_history.append(metrics['similarity'])
        
        if 'temperature' in metrics:
            self._temperature_history.append(metrics['temperature'])
    
    def get_stats(self) -> AlignmentStats:
        """获取统计信息"""
        return self._stats
    
    def get_recent_loss(self, n: int = 100) -> List[float]:
        """获取最近的损失值"""
        return list(self._loss_history)[-n:]
    
    def get_loss_trend(self) -> str:
        """获取损失趋势"""
        if len(self._loss_history) < 10:
            return "insufficient_data"
        
        recent = list(self._loss_history)[-10:]
        older = list(self._loss_history)[-20:-10] if len(self._loss_history) >= 20 else recent
        
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        if recent_avg < older_avg * 0.95:
            return "decreasing"
        elif recent_avg > older_avg * 1.05:
            return "increasing"
        return "stable"
    
    def is_converging(self, threshold: float = 0.01) -> bool:
        """检查是否收敛"""
        if len(self._loss_history) < 100:
            return False
        
        recent = list(self._loss_history)[-50:]
        variance = sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent)
        
        return variance < threshold
    
    def get_accuracy_trend(self) -> float:
        """获取准确率趋势"""
        if len(self._accuracy_history) < 10:
            return 0.0
        
        recent = list(self._accuracy_history)[-10:]
        older = list(self._accuracy_history)[-20:-10] if len(self._accuracy_history) >= 20 else recent
        
        return sum(recent) / len(recent) - sum(older) / len(older)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        return {
            'total_steps': self._stats.total_steps,
            'avg_loss': self._stats.avg_loss,
            'min_loss': self._stats.min_loss,
            'max_loss': self._stats.max_loss,
            'avg_accuracy_a': self._stats.avg_accuracy_a,
            'avg_accuracy_b': self._stats.avg_accuracy_b,
            'avg_temperature': self._stats.avg_temperature,
            'loss_trend': self.get_loss_trend(),
            'is_converging': self.is_converging(),
        }
    
    def reset(self) -> None:
        """重置监控器"""
        self._loss_history.clear()
        self._accuracy_history.clear()
        self._similarity_history.clear()
        self._temperature_history.clear()
        self._stats = AlignmentStats()
        self._step = 0


class TemperatureScheduler:
    """温度调度器"""
    
    def __init__(
        self,
        initial_temp: float = 0.07,
        min_temp: float = 0.01,
        max_temp: float = 1.0,
        schedule: str = "constant"  # constant, linear, cosine, adaptive
    ):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.schedule = schedule
        self._step = 0
        self._total_steps = 10000
        self._current_temp = initial_temp
        
        # 自适应调度的历史
        self._loss_history: deque = deque(maxlen=100)
    
    def step(self, loss: Optional[float] = None) -> float:
        """更新温度"""
        self._step += 1
        
        if self.schedule == "constant":
            self._current_temp = self.initial_temp
        
        elif self.schedule == "linear":
            progress = min(1.0, self._step / self._total_steps)
            self._current_temp = self.initial_temp + (self.min_temp - self.initial_temp) * progress
        
        elif self.schedule == "cosine":
            progress = min(1.0, self._step / self._total_steps)
            self._current_temp = self.min_temp + (self.initial_temp - self.min_temp) * \
                                 (1 + math.cos(math.pi * progress)) / 2
        
        elif self.schedule == "adaptive" and loss is not None:
            self._loss_history.append(loss)
            
            if len(self._loss_history) >= 10:
                recent = list(self._loss_history)[-10:]
                older = list(self._loss_history)[-20:-10] if len(self._loss_history) >= 20 else recent
                
                recent_avg = sum(recent) / len(recent)
                older_avg = sum(older) / len(older)
                
                # 损失增加时降低温度
                if recent_avg > older_avg * 1.1:
                    self._current_temp = max(self.min_temp, self._current_temp * 0.95)
                # 损失稳定时略微提高温度
                elif abs(recent_avg - older_avg) / older_avg < 0.01:
                    self._current_temp = min(self.max_temp, self._current_temp * 1.02)
        
        return self._current_temp
    
    def get_temperature(self) -> float:
        """获取当前温度"""
        return self._current_temp
    
    def set_total_steps(self, total_steps: int) -> None:
        """设置总步数"""
        self._total_steps = total_steps
    
    def reset(self) -> None:
        """重置调度器"""
        self._step = 0
        self._current_temp = self.initial_temp
        self._loss_history.clear()


# ==================== 对比学习对齐 ====================

class ContrastiveLearningAlignment(nn.Module):
    """对比学习对齐模块
    
    实现CLIP风格的对比学习，支持：
    - InfoNCE损失
    - 硬负样本挖掘
    - 批内负样本
    - 温度调度
    - 多种损失类型
    """
    
    def __init__(
        self, 
        config: ContrastiveLearningConfig, 
        embed_dim: int,
        temperature_schedule: str = "constant"
    ):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        
        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.ones([]) * math.log(1 / config.temperature))
        
        # 温度调度器
        self._temp_scheduler = TemperatureScheduler(
            initial_temp=config.temperature,
            schedule=temperature_schedule
        )
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 监控器
        self._monitor = AlignmentMonitor()
        self._step = 0
        
        # 硬负样本挖掘
        self._hard_mining_enabled = config.hard_negative_mining
        self._hard_neg_ratio = config.hard_negative_ratio
        
        # 历史记录
        self._pos_sim_history: deque = deque(maxlen=1000)
        self._neg_sim_history: deque = deque(maxlen=1000)
    
    def forward(self, 
                features_a: Tensor, 
                features_b: Tensor,
                labels: Optional[Tensor] = None,
                return_features: bool = False) -> Tuple[Tensor, Dict[str, float]]:
        """计算对比学习损失
        
        Args:
            features_a: 模态A特征 [batch, dim]
            features_b: 模态B特征 [batch, dim]
            labels: 可选的匹配标签
            return_features: 是否返回归一化后的特征
            
        Returns:
            损失值和指标字典
        """
        self._step += 1
        
        # L2归一化
        proj_a = F.normalize(self.projection(features_a), dim=-1)
        proj_b = F.normalize(self.projection(features_b), dim=-1)
        
        # 计算温度
        temperature = self.temperature.exp().clamp(min=0.01, max=100)
        
        # 计算相似度矩阵
        logits = torch.matmul(proj_a, proj_b.T) / temperature
        
        # 标签（默认对角线为正样本）
        batch_size = features_a.shape[0]
        if labels is None:
            labels = torch.arange(batch_size, device=features_a.device)
        
        # 计算损失
        if self.config.loss_type == "info_nce":
            loss, metrics = self._info_nce_loss(logits, labels, temperature)
        elif self.config.loss_type == "clip":
            loss, metrics = self._clip_loss(logits, labels, temperature)
        elif self.config.loss_type == "simclr":
            loss, metrics = self._simclr_loss(proj_a, proj_b, temperature)
        else:
            loss, metrics = self._info_nce_loss(logits, labels, temperature)
        
        # 硬负样本挖掘增强
        if self._hard_mining_enabled and self.training:
            hard_loss = self._hard_negative_loss(logits, labels)
            loss = loss + 0.1 * hard_loss
            metrics['hard_negative_loss'] = hard_loss.item()
        
        # 记录正负样本相似度
        with torch.no_grad():
            pos_sim = torch.diag(logits * temperature).mean()
            neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=logits.device)
            neg_sim = (logits * temperature)[neg_mask].mean()
            
            self._pos_sim_history.append(pos_sim.item())
            self._neg_sim_history.append(neg_sim.item())
            
            metrics['pos_similarity'] = pos_sim.item()
            metrics['neg_similarity'] = neg_sim.item()
            metrics['similarity_gap'] = pos_sim.item() - neg_sim.item()
        
        # 更新监控
        self._monitor.record(loss.item(), metrics)
        
        # 更新温度调度
        self._temp_scheduler.step(loss.item())
        
        if return_features:
            return loss, metrics, proj_a, proj_b
        
        return loss, metrics
    
    def _info_nce_loss(
        self, 
        logits: Tensor, 
        labels: Tensor,
        temperature: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """InfoNCE损失"""
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        loss = (loss_a + loss_b) / 2
        
        with torch.no_grad():
            pred_a = logits.argmax(dim=1)
            pred_b = logits.T.argmax(dim=1)
            acc_a = (pred_a == labels).float().mean()
            acc_b = (pred_b == labels).float().mean()
            
            metrics = {
                'contrastive_loss': loss.item(),
                'contrastive_acc_a': acc_a.item(),
                'contrastive_acc_b': acc_b.item(),
                'temperature': temperature.item()
            }
        
        return loss, metrics
    
    def _clip_loss(
        self, 
        logits: Tensor, 
        labels: Tensor,
        temperature: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """CLIP风格损失"""
        # 对称交叉熵
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        with torch.no_grad():
            # Top-k准确率
            pred_i2t = logits.argmax(dim=1)
            pred_t2i = logits.T.argmax(dim=1)
            acc_i2t = (pred_i2t == labels).float().mean()
            acc_t2i = (pred_t2i == labels).float().mean()
            
            # Top-5准确率
            _, top5_i2t = logits.topk(5, dim=1)
            _, top5_t2i = logits.T.topk(5, dim=1)
            top5_acc_i2t = (top5_i2t == labels.unsqueeze(1)).any(dim=1).float().mean()
            top5_acc_t2i = (top5_t2i == labels.unsqueeze(1)).any(dim=1).float().mean()
            
            metrics = {
                'contrastive_loss': loss.item(),
                'contrastive_acc_a': acc_i2t.item(),
                'contrastive_acc_b': acc_t2i.item(),
                'top5_acc_i2t': top5_acc_i2t.item(),
                'top5_acc_t2i': top5_acc_t2i.item(),
                'temperature': temperature.item()
            }
        
        return loss, metrics
    
    def _simclr_loss(
        self, 
        proj_a: Tensor, 
        proj_b: Tensor,
        temperature: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """SimCLR风格损失"""
        batch_size = proj_a.shape[0]
        
        # 拼接特征
        features = torch.cat([proj_a, proj_b], dim=0)
        
        # 计算相似度
        similarity = torch.matmul(features, features.T) / temperature
        
        # 创建标签
        labels = torch.cat([
            torch.arange(batch_size, 2*batch_size),
            torch.arange(0, batch_size)
        ], dim=0).to(similarity.device)
        
        # 移除自身相似度
        mask = torch.eye(2*batch_size, dtype=torch.bool, device=similarity.device)
        similarity = similarity.masked_fill(mask, float('-inf'))
        
        loss = F.cross_entropy(similarity, labels)
        
        with torch.no_grad():
            pred = similarity.argmax(dim=1)
            acc = (pred == labels).float().mean()
            
            metrics = {
                'contrastive_loss': loss.item(),
                'contrastive_acc_a': acc.item(),
                'contrastive_acc_b': acc.item(),
                'temperature': temperature.item()
            }
        
        return loss, metrics
    
    def _hard_negative_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """硬负样本损失"""
        batch_size = logits.shape[0]
        
        # 创建负样本mask
        neg_mask = ~torch.eye(batch_size, dtype=torch.bool, device=logits.device)
        
        # 获取最难的负样本
        k = max(1, int(batch_size * self._hard_neg_ratio))
        neg_logits = logits.masked_fill(~neg_mask, float('-inf'))
        hard_neg_values, _ = neg_logits.topk(k, dim=1)
        
        # 正样本logits
        pos_logits = torch.diag(logits).unsqueeze(1)
        
        # 对比损失
        combined = torch.cat([pos_logits, hard_neg_values], dim=1)
        hard_labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        return F.cross_entropy(combined, hard_labels)
    
    def compute_similarity(self, features_a: Tensor, features_b: Tensor) -> Tensor:
        """计算相似度分数"""
        features_a = F.normalize(self.projection(features_a), dim=-1)
        features_b = F.normalize(self.projection(features_b), dim=-1)
        return torch.matmul(features_a, features_b.T)
    
    def get_monitor(self) -> AlignmentMonitor:
        """获取监控器"""
        return self._monitor
    
    def get_stats(self) -> AlignmentStats:
        """获取统计信息"""
        return self._monitor.get_stats()
    
    def get_similarity_gap(self) -> float:
        """获取正负样本相似度差距"""
        if not self._pos_sim_history or not self._neg_sim_history:
            return 0.0
        
        pos_avg = sum(self._pos_sim_history) / len(self._pos_sim_history)
        neg_avg = sum(self._neg_sim_history) / len(self._neg_sim_history)
        
        return pos_avg - neg_avg
    
    def is_learning_effective(self) -> bool:
        """判断学习是否有效"""
        gap = self.get_similarity_gap()
        stats = self.get_stats()
        
        # 相似度差距大于0.3且准确率大于50%
        return gap > 0.3 and (stats.avg_accuracy_a + stats.avg_accuracy_b) / 2 > 0.5
    
    def set_temperature_schedule(self, schedule: str, total_steps: int = 10000) -> None:
        """设置温度调度"""
        self._temp_scheduler = TemperatureScheduler(
            initial_temp=self.config.temperature,
            schedule=schedule
        )
        self._temp_scheduler.set_total_steps(total_steps)
    
    def enable_hard_mining(self, ratio: float = 0.2) -> None:
        """启用硬负样本挖掘"""
        self._hard_mining_enabled = True
        self._hard_neg_ratio = ratio
    
    def disable_hard_mining(self) -> None:
        """禁用硬负样本挖掘"""
        self._hard_mining_enabled = False
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._monitor.reset()
        self._pos_sim_history.clear()
        self._neg_sim_history.clear()
        self._step = 0
    
    def print_summary(self) -> None:
        """打印摘要"""
        summary = self._monitor.get_summary()
        print("\n" + "="*50)
        print("ContrastiveLearningAlignment Summary")
        print("="*50)
        print(f"Total steps: {summary['total_steps']}")
        print(f"Avg loss: {summary['avg_loss']:.4f}")
        print(f"Avg accuracy A: {summary['avg_accuracy_a']:.4f}")
        print(f"Avg accuracy B: {summary['avg_accuracy_b']:.4f}")
        print(f"Temperature: {summary['avg_temperature']:.4f}")
        print(f"Loss trend: {summary['loss_trend']}")
        print(f"Is converging: {summary['is_converging']}")
        print(f"Similarity gap: {self.get_similarity_gap():.4f}")
        print(f"Learning effective: {self.is_learning_effective()}")
        print("="*50)


class HardNegativeMining(nn.Module):
    """硬负样本挖掘
    
    支持多种挖掘策略：
    - 硬负样本（最难）
    - 半硬负样本
    - 混合策略
    """
    
    def __init__(
        self, 
        config: ContrastiveLearningConfig,
        strategy: NegativeMiningStrategy = NegativeMiningStrategy.HARD
    ):
        super().__init__()
        self.config = config
        self.hard_negative_ratio = config.hard_negative_ratio
        self.strategy = strategy
        
        # 统计信息
        self._hard_neg_count = 0
        self._semi_hard_neg_count = 0
        self._total_neg_count = 0
    
    def forward(self, 
                similarity_matrix: Tensor,
                labels: Tensor,
                margin: float = 0.2) -> Tuple[Tensor, Tensor]:
        """挖掘硬负样本
        
        Args:
            similarity_matrix: 相似度矩阵 [batch, batch]
            labels: 匹配标签
            margin: 半硬负样本的margin
            
        Returns:
            硬负样本索引和权重
        """
        batch_size = similarity_matrix.shape[0]
        
        # 创建mask排除正样本
        mask = ~(labels.unsqueeze(1) == labels.unsqueeze(0))
        
        # 获取负样本相似度
        neg_similarity = similarity_matrix.masked_fill(~mask, float('-inf'))
        
        # 选择样本数量
        k = max(1, int(batch_size * self.hard_negative_ratio))
        
        if self.strategy == NegativeMiningStrategy.HARD:
            # 最难的负样本
            hard_neg_indices = neg_similarity.topk(k, dim=1).indices
            self._hard_neg_count += k * batch_size
            
        elif self.strategy == NegativeMiningStrategy.SEMI_HARD:
            # 半硬负样本：比正样本难但不是最难
            pos_similarity = torch.diag(similarity_matrix).unsqueeze(1)
            
            # 在[pos - margin, pos]范围内的负样本
            semi_hard_mask = (neg_similarity < pos_similarity) & \
                            (neg_similarity > pos_similarity - margin)
            semi_hard_sim = similarity_matrix.masked_fill(~semi_hard_mask, float('-inf'))
            
            hard_neg_indices = semi_hard_sim.topk(k, dim=1).indices
            self._semi_hard_neg_count += k * batch_size
            
        elif self.strategy == NegativeMiningStrategy.MIXED:
            # 混合策略：一半硬负样本，一半半硬负样本
            k_hard = k // 2
            k_semi = k - k_hard
            
            # 硬负样本
            hard_indices = neg_similarity.topk(k_hard, dim=1).indices
            
            # 半硬负样本
            pos_similarity = torch.diag(similarity_matrix).unsqueeze(1)
            semi_hard_mask = (neg_similarity < pos_similarity) & \
                            (neg_similarity > pos_similarity - margin)
            semi_hard_sim = similarity_matrix.masked_fill(~semi_hard_mask, float('-inf'))
            semi_indices = semi_hard_sim.topk(k_semi, dim=1).indices
            
            hard_neg_indices = torch.cat([hard_indices, semi_indices], dim=1)
            self._hard_neg_count += k_hard * batch_size
            self._semi_hard_neg_count += k_semi * batch_size
            
        else:  # RANDOM
            # 随机负样本
            neg_indices = torch.where(mask)
            perm = torch.randperm(len(neg_indices[0]))[:k * batch_size]
            hard_neg_indices = neg_indices[1][perm].view(batch_size, k)
        
        self._total_neg_count += k * batch_size
        
        # 计算权重（相似度越高权重越大）
        hard_neg_similarity = torch.gather(similarity_matrix, 1, hard_neg_indices)
        weights = F.softmax(hard_neg_similarity, dim=1)
        
        return hard_neg_indices, weights
    
    def get_mining_stats(self) -> Dict[str, Any]:
        """获取挖掘统计信息"""
        return {
            'hard_neg_count': self._hard_neg_count,
            'semi_hard_neg_count': self._semi_hard_neg_count,
            'total_neg_count': self._total_neg_count,
            'hard_ratio': self._hard_neg_count / max(1, self._total_neg_count),
            'semi_hard_ratio': self._semi_hard_neg_count / max(1, self._total_neg_count),
        }
    
    def set_strategy(self, strategy: NegativeMiningStrategy) -> None:
        """设置挖掘策略"""
        self.strategy = strategy
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._hard_neg_count = 0
        self._semi_hard_neg_count = 0
        self._total_neg_count = 0


# ==================== 显式对齐 ====================

class ExplicitAlignment(nn.Module):
    """显式对齐模块
    
    使用MLP或Attention直接对齐不同模态的表示，支持：
    - MLP对齐
    - 注意力对齐
    - 线性对齐
    - 多种损失函数
    """
    
    def __init__(self, config: ExplicitAlignConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        if config.method == "mlp":
            self.alignment_net = self._build_mlp(input_dim, config.hidden_size, config.num_layers)
        elif config.method == "attention":
            self.alignment_net = self._build_attention(input_dim, config.hidden_size)
        else:
            self.alignment_net = nn.Linear(input_dim, config.hidden_size)
        
        # 监控器
        self._monitor = AlignmentMonitor()
        self._step = 0
        
        # 损失类型配置
        self._loss_type = "mse"  # mse, cosine, l1, combined
        
        # 历史记录
        self._alignment_scores: deque = deque(maxlen=1000)
    
    def _build_mlp(self, input_dim: int, hidden_size: int, num_layers: int) -> nn.Module:
        """构建MLP对齐网络"""
        layers = []
        current_dim = input_dim * 2  # 拼接两个模态
        
        for i in range(num_layers):
            out_dim = hidden_size if i < num_layers - 1 else input_dim
            layers.extend([
                nn.Linear(current_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            current_dim = out_dim
        
        return nn.Sequential(*layers)
    
    def _build_attention(self, input_dim: int, hidden_size: int) -> nn.Module:
        """构建Attention对齐网络"""
        return CrossModalAttention(input_dim, hidden_size, num_heads=8)
    
    def forward(self, 
                features_a: Tensor, 
                features_b: Tensor,
                return_aligned: bool = False) -> Tuple[Tensor, Dict[str, float]]:
        """计算对齐损失
        
        Args:
            features_a: 模态A特征 [batch, dim]
            features_b: 模态B特征 [batch, dim]
            return_aligned: 是否返回对齐后的特征
            
        Returns:
            损失值和指标
        """
        self._step += 1
        aligned_a, aligned_b = None, None
        
        if self.config.method == "mlp":
            # 拼接后预测对齐特征
            concat = torch.cat([features_a, features_b], dim=-1)
            aligned = self.alignment_net(concat)
            
            # 计算损失
            loss = self._compute_loss(aligned, features_a, features_b)
            aligned_a = aligned_b = aligned
        
        elif self.config.method == "attention":
            # 注意力对齐
            aligned_a, aligned_b, attn_weights = self.alignment_net(features_a, features_b)
            
            # 计算损失
            loss = self._compute_loss(aligned_a, aligned_b)
        
        else:
            # 简单线性对齐
            proj_a = self.alignment_net(features_a)
            proj_b = self.alignment_net(features_b)
            loss = self._compute_loss(proj_a, proj_b)
            aligned_a, aligned_b = proj_a, proj_b
        
        # 计算对齐分数
        with torch.no_grad():
            alignment_score = torch.cosine_similarity(
                aligned_a if aligned_a is not None else features_a,
                aligned_b if aligned_b is not None else features_b,
                dim=-1
            ).mean()
            self._alignment_scores.append(alignment_score.item())
        
        metrics = {
            'explicit_align_loss': loss.item(),
            'alignment_score': alignment_score.item(),
        }
        
        # 更新监控
        self._monitor.record(loss.item(), metrics)
        
        if return_aligned and aligned_a is not None:
            return loss, metrics, aligned_a, aligned_b
        
        return loss, metrics
    
    def _compute_loss(self, *args) -> Tensor:
        """计算对齐损失"""
        if len(args) == 3:
            # MLP情况：aligned, features_a, features_b
            aligned, features_a, features_b = args
            
            if self._loss_type == "mse":
                loss = (F.mse_loss(aligned, features_a) + F.mse_loss(aligned, features_b)) / 2
            elif self._loss_type == "cosine":
                loss = 2 - torch.cosine_similarity(aligned, features_a, dim=-1).mean() - \
                       torch.cosine_similarity(aligned, features_b, dim=-1).mean()
            elif self._loss_type == "l1":
                loss = (F.l1_loss(aligned, features_a) + F.l1_loss(aligned, features_b)) / 2
            else:  # combined
                mse = (F.mse_loss(aligned, features_a) + F.mse_loss(aligned, features_b)) / 2
                cos = 2 - torch.cosine_similarity(aligned, features_a, dim=-1).mean() - \
                      torch.cosine_similarity(aligned, features_b, dim=-1).mean()
                loss = mse + 0.1 * cos
        else:
            # 其他情况：features_a, features_b
            features_a, features_b = args
            
            if self._loss_type == "mse":
                loss = F.mse_loss(features_a, features_b)
            elif self._loss_type == "cosine":
                loss = 1 - torch.cosine_similarity(features_a, features_b, dim=-1).mean()
            elif self._loss_type == "l1":
                loss = F.l1_loss(features_a, features_b)
            else:  # combined
                loss = F.mse_loss(features_a, features_b) + \
                       0.1 * (1 - torch.cosine_similarity(features_a, features_b, dim=-1).mean())
        
        return loss
    
    def get_monitor(self) -> AlignmentMonitor:
        """获取监控器"""
        return self._monitor
    
    def get_stats(self) -> AlignmentStats:
        """获取统计信息"""
        return self._monitor.get_stats()
    
    def get_avg_alignment_score(self) -> float:
        """获取平均对齐分数"""
        if not self._alignment_scores:
            return 0.0
        return sum(self._alignment_scores) / len(self._alignment_scores)
    
    def set_loss_type(self, loss_type: str) -> None:
        """设置损失类型"""
        if loss_type not in ["mse", "cosine", "l1", "combined"]:
            raise ValueError(f"Invalid loss type: {loss_type}")
        self._loss_type = loss_type
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._monitor.reset()
        self._alignment_scores.clear()
        self._step = 0
    
    def print_summary(self) -> None:
        """打印摘要"""
        summary = self._monitor.get_summary()
        print("\n" + "="*50)
        print("ExplicitAlignment Summary")
        print("="*50)
        print(f"Method: {self.config.method}")
        print(f"Loss type: {self._loss_type}")
        print(f"Total steps: {summary['total_steps']}")
        print(f"Avg loss: {summary['avg_loss']:.4f}")
        print(f"Avg alignment score: {self.get_avg_alignment_score():.4f}")
        print(f"Loss trend: {summary['loss_trend']}")
        print("="*50)


class CrossModalAttention(nn.Module):
    """交叉模态注意力
    
    支持双向交叉注意力，可配置dropout和残差连接
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        hidden_dim: int, 
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_residual = use_residual
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, hidden_dim)
        self.k_proj = nn.Linear(embed_dim, hidden_dim)
        self.v_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # 统计信息
        self._attn_entropy_history: deque = deque(maxlen=1000)
    
    def forward(
        self, 
        features_a: Tensor, 
        features_b: Tensor,
        return_attention_stats: bool = False
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Dict[str, float]]]:
        """交叉注意力
        
        Args:
            features_a: [batch, dim] 或 [batch, seq_len, dim]
            features_b: [batch, dim] 或 [batch, seq_len, dim]
            return_attention_stats: 是否返回注意力统计
            
        Returns:
            对齐后的特征和注意力权重
        """
        # 确保有序列维度
        squeeze_output = features_a.dim() == 2
        if features_a.dim() == 2:
            features_a = features_a.unsqueeze(1)
        if features_b.dim() == 2:
            features_b = features_b.unsqueeze(1)
        
        batch_size = features_a.shape[0]
        
        # A attends to B
        aligned_a, attn_weights_a = self._cross_attend(features_a, features_b)
        
        # B attends to A (symmetric)
        aligned_b, attn_weights_b = self._cross_attend(features_b, features_a)
        
        # 计算注意力熵
        with torch.no_grad():
            entropy_a = self._compute_attention_entropy(attn_weights_a)
            entropy_b = self._compute_attention_entropy(attn_weights_b)
            self._attn_entropy_history.append((entropy_a + entropy_b) / 2)
        
        # Squeeze if no sequence dimension
        if squeeze_output:
            aligned_a = aligned_a.squeeze(1)
            aligned_b = aligned_b.squeeze(1)
        
        if return_attention_stats:
            stats = {
                'attn_entropy_a': entropy_a,
                'attn_entropy_b': entropy_b,
                'avg_attn_entropy': (entropy_a + entropy_b) / 2,
            }
            return aligned_a, aligned_b, attn_weights_a, stats
        
        return aligned_a, aligned_b, attn_weights_a
    
    def _cross_attend(self, query_features: Tensor, key_value_features: Tensor) -> Tuple[Tensor, Tensor]:
        """单向交叉注意力"""
        batch_size = query_features.shape[0]
        
        # 投影
        q = self.q_proj(self.norm1(query_features))
        k = self.k_proj(self.norm1(key_value_features))
        v = self.v_proj(self.norm1(key_value_features))
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # 输出投影和残差
        output = self.dropout(self.out_proj(attn_output))
        if self.use_residual:
            output = output + query_features
        output = self.norm2(output)
        
        return output, attn_weights
    
    def _compute_attention_entropy(self, attn_weights: Tensor) -> float:
        """计算注意力熵"""
        # attn_weights: [batch, heads, seq_q, seq_k]
        # 使用小值防止log(0)
        eps = 1e-10
        entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)
        return entropy.mean().item()
    
    def get_avg_attention_entropy(self) -> float:
        """获取平均注意力熵"""
        if not self._attn_entropy_history:
            return 0.0
        return sum(self._attn_entropy_history) / len(self._attn_entropy_history)
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._attn_entropy_history.clear()


# ==================== 最优传输对齐 ====================

class OptimalTransportAlignment(nn.Module):
    """最优传输对齐
    
    使用Sinkhorn算法实现最优传输对齐，支持：
    - 可配置的Sinkhorn迭代次数
    - 自适应epsilon
    - 传输计划可视化
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        sinkhorn_iters: int = 10, 
        epsilon: float = 0.1,
        adaptive_epsilon: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = epsilon
        self.adaptive_epsilon = adaptive_epsilon
        
        # 投影层
        self.projection = nn.Linear(embed_dim, embed_dim)
        
        # 监控器
        self._monitor = AlignmentMonitor()
        self._step = 0
        
        # 传输计划历史
        self._transport_cost_history: deque = deque(maxlen=1000)
        self._sparsity_history: deque = deque(maxlen=1000)
    
    def forward(self, 
                features_a: Tensor, 
                features_b: Tensor,
                return_plan: bool = False) -> Union[Tuple[Tensor, Dict[str, float]], Tuple[Tensor, Dict[str, float], Tensor]]:
        """计算最优传输损失
        
        Args:
            features_a: [batch, dim] 或 [batch, seq_a, dim]
            features_b: [batch, dim] 或 [batch, seq_b, dim]
            return_plan: 是否返回传输计划
            
        Returns:
            损失值和指标
        """
        self._step += 1
        
        # 投影
        proj_a = self.projection(features_a)
        proj_b = self.projection(features_b)
        
        # 计算成本矩阵
        if proj_a.dim() == 2:
            # 单向量情况 - 保持batch维度
            cost = torch.cdist(proj_a.unsqueeze(1), proj_b.unsqueeze(1))
            # cost shape: [batch, 1, 1]
        else:
            # 序列情况
            cost = torch.cdist(proj_a, proj_b)
            # cost shape: [batch, seq_a, seq_b]
        
        # 自适应epsilon
        epsilon = self._get_epsilon(cost)
        
        # Sinkhorn算法
        transport_plan = self._sinkhorn(cost, epsilon)
        
        # 最优传输损失
        loss = (transport_plan * cost).sum(dim=(-2, -1)).mean()
        
        # 计算指标
        with torch.no_grad():
            avg_cost = cost.mean().item()
            sparsity = (transport_plan < 0.01).float().mean().item()
            
            self._transport_cost_history.append(avg_cost)
            self._sparsity_history.append(sparsity)
        
        metrics = {
            'ot_loss': loss.item(),
            'avg_transport_cost': avg_cost,
            'plan_sparsity': sparsity,
            'epsilon': epsilon,
        }
        
        # 更新监控
        self._monitor.record(loss.item(), metrics)
        
        if return_plan:
            return loss, metrics, transport_plan
        
        return loss, metrics
    
    def _get_epsilon(self, cost: Tensor) -> float:
        """获取epsilon值"""
        if self.adaptive_epsilon:
            # 基于成本矩阵的标准差自适应调整
            std = cost.std().item()
            return max(0.01, min(1.0, std * 0.1))
        return self.epsilon
    
    def _sinkhorn(self, cost: Tensor, epsilon: float) -> Tensor:
        """Sinkhorn算法求解最优传输"""
        # 初始化
        log_p = -cost / epsilon
        
        # 检查维度 - 如果是[batch, 1, 1]则跳过迭代
        if log_p.dim() == 3 and log_p.shape[-1] == 1 and log_p.shape[-2] == 1:
            # 单点对单点，直接返回均匀分布
            return torch.ones_like(log_p)
        
        for _ in range(self.sinkhorn_iters):
            # Row normalization
            log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
            # Column normalization
            log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
        
        return log_p.exp()
    
    def get_monitor(self) -> AlignmentMonitor:
        """获取监控器"""
        return self._monitor
    
    def get_stats(self) -> AlignmentStats:
        """获取统计信息"""
        return self._monitor.get_stats()
    
    def get_avg_transport_cost(self) -> float:
        """获取平均传输成本"""
        if not self._transport_cost_history:
            return 0.0
        return sum(self._transport_cost_history) / len(self._transport_cost_history)
    
    def get_avg_sparsity(self) -> float:
        """获取平均稀疏度"""
        if not self._sparsity_history:
            return 0.0
        return sum(self._sparsity_history) / len(self._sparsity_history)
    
    def set_epsilon(self, epsilon: float) -> None:
        """设置epsilon"""
        self.epsilon = epsilon
    
    def set_sinkhorn_iters(self, iters: int) -> None:
        """设置Sinkhorn迭代次数"""
        self.sinkhorn_iters = iters
    
    def enable_adaptive_epsilon(self) -> None:
        """启用自适应epsilon"""
        self.adaptive_epsilon = True
    
    def disable_adaptive_epsilon(self) -> None:
        """禁用自适应epsilon"""
        self.adaptive_epsilon = False
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._monitor.reset()
        self._transport_cost_history.clear()
        self._sparsity_history.clear()
        self._step = 0
    
    def print_summary(self) -> None:
        """打印摘要"""
        summary = self._monitor.get_summary()
        print("\n" + "="*50)
        print("OptimalTransportAlignment Summary")
        print("="*50)
        print(f"Epsilon: {self.epsilon}")
        print(f"Sinkhorn iterations: {self.sinkhorn_iters}")
        print(f"Adaptive epsilon: {self.adaptive_epsilon}")
        print(f"Total steps: {summary['total_steps']}")
        print(f"Avg loss: {summary['avg_loss']:.4f}")
        print(f"Avg transport cost: {self.get_avg_transport_cost():.4f}")
        print(f"Avg sparsity: {self.get_avg_sparsity():.4f}")
        print(f"Loss trend: {summary['loss_trend']}")
        print("="*50)


# ==================== 统一对齐模块 ====================

class CrossModalAligner(nn.Module):
    """跨模态对齐器
    
    整合多种对齐方法，支持灵活配置：
    - 对比学习
    - 显式对齐
    - 最优传输
    - 交叉注意力
    - 多模态支持
    """
    
    def __init__(
        self, 
        config: CrossModalAlignmentConfig, 
        embed_dim: int,
        enable_cross_attention: bool = True
    ):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.cross_attention_enabled = enable_cross_attention
        
        # 投影层
        self.projection_dim = config.projection_dim
        self.projector_a = self._build_projector(embed_dim, config.projection_dim, config.projection_layers)
        self.projector_b = self._build_projector(embed_dim, config.projection_dim, config.projection_layers)
        
        # 对齐模块
        self.aligners = nn.ModuleDict()
        
        if config.method == AlignmentMethod.CONTRASTIVE or config.method == AlignmentMethod.CROSS_ATTENTION:
            self.aligners['contrastive'] = ContrastiveLearningAlignment(
                config.contrastive, config.projection_dim
            )
        
        if config.method == AlignmentMethod.EXPLICIT_ALIGN:
            self.aligners['explicit'] = ExplicitAlignment(
                config.explicit, config.projection_dim
            )
        
        if config.method == AlignmentMethod.OPTIMAL_TRANSPORT:
            self.aligners['ot'] = OptimalTransportAlignment(config.projection_dim)
        
        # 默认添加交叉注意力
        if enable_cross_attention:
            self.cross_attention = CrossModalAttention(
                config.projection_dim, 
                config.projection_dim, 
                num_heads=8
            )
        else:
            self.cross_attention = None
        
        # 监控器
        self._monitor = AlignmentMonitor()
        self._step = 0
        
        # 对齐历史
        self._alignment_scores: deque = deque(maxlen=1000)
        self._modality_pairs: List[Tuple[str, str]] = []
    
    def _build_projector(self, input_dim: int, output_dim: int, num_layers: int) -> nn.Module:
        """构建投影层"""
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            out_dim = output_dim
            layers.extend([
                nn.Linear(current_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(self.config.projection_dropout)
            ])
            current_dim = out_dim
        
        return nn.Sequential(*layers)
    
    def forward(
        self,
        features: Dict[str, Tensor],
        compute_loss: bool = True,
        apply_cross_attention: bool = True
    ) -> Tuple[Dict[str, Tensor], Optional[Tensor], Dict[str, float]]:
        """前向传播
        
        Args:
            features: 模态名称到特征的映射
            compute_loss: 是否计算对齐损失
            apply_cross_attention: 是否应用交叉注意力
            
        Returns:
            对齐后的特征、损失值、指标
        """
        self._step += 1
        modalities = list(features.keys())
        
        if len(modalities) < 2:
            return features, None, {}
        
        # 记录模态对
        if (modalities[0], modalities[1]) not in self._modality_pairs:
            self._modality_pairs.append((modalities[0], modalities[1]))
        
        # 取前两个模态进行对齐
        feat_a = features[modalities[0]]
        feat_b = features[modalities[1]]
        
        # 投影
        proj_a = self.projector_a(feat_a)
        proj_b = self.projector_b(feat_b)
        
        aligned_features = {
            modalities[0]: proj_a,
            modalities[1]: proj_b
        }
        
        # 保留其他模态
        for mod in modalities[2:]:
            aligned_features[mod] = features[mod]
        
        # 计算损失
        total_loss = None
        all_metrics = {}
        
        if compute_loss:
            for name, aligner in self.aligners.items():
                loss, metrics = aligner(proj_a, proj_b)
                
                if total_loss is None:
                    total_loss = loss * self.config.align_loss_weight
                else:
                    total_loss = total_loss + loss * self.config.align_loss_weight
                
                all_metrics.update({f"{name}_{k}": v for k, v in metrics.items()})
        
        # 交叉注意力增强
        if apply_cross_attention and self.cross_attention is not None:
            aligned_a, aligned_b, attn_weights = self.cross_attention(proj_a, proj_b)
            aligned_features[modalities[0]] = aligned_a
            aligned_features[modalities[1]] = aligned_b
            
            # 计算注意力熵
            all_metrics['cross_attn_entropy'] = self.cross_attention.get_avg_attention_entropy()
        
        # 计算对齐分数
        with torch.no_grad():
            alignment_score = self._compute_alignment_score(
                aligned_features[modalities[0]], 
                aligned_features[modalities[1]]
            )
            self._alignment_scores.append(alignment_score)
            all_metrics['alignment_score'] = alignment_score
        
        # 更新监控
        if total_loss is not None:
            self._monitor.record(total_loss.item(), all_metrics)
        
        return aligned_features, total_loss, all_metrics
    
    def _compute_alignment_score(self, features_a: Tensor, features_b: Tensor) -> float:
        """计算对齐分数"""
        # 处理不同维度
        if features_a.dim() == 3:
            features_a = features_a.mean(dim=1)
        if features_b.dim() == 3:
            features_b = features_b.mean(dim=1)
        
        similarity = torch.cosine_similarity(features_a, features_b, dim=-1).mean()
        return similarity.item()
    
    def get_alignment_score(self, features: Dict[str, Tensor]) -> float:
        """获取对齐分数"""
        modalities = list(features.keys())
        if len(modalities) < 2:
            return 1.0
        
        feat_a = features[modalities[0]]
        feat_b = features[modalities[1]]
        
        # 投影
        proj_a = self.projector_a(feat_a)
        proj_b = self.projector_b(feat_b)
        
        return self._compute_alignment_score(proj_a, proj_b)
    
    def get_monitor(self) -> AlignmentMonitor:
        """获取监控器"""
        return self._monitor
    
    def get_stats(self) -> AlignmentStats:
        """获取统计信息"""
        return self._monitor.get_stats()
    
    def get_avg_alignment_score(self) -> float:
        """获取平均对齐分数"""
        if not self._alignment_scores:
            return 0.0
        return sum(self._alignment_scores) / len(self._alignment_scores)
    
    def get_modality_pairs(self) -> List[Tuple[str, str]]:
        """获取处理过的模态对"""
        return self._modality_pairs
    
    def get_aligner(self, name: str) -> Optional[nn.Module]:
        """获取指定的对齐器"""
        return self.aligners.get(name)
    
    def add_aligner(self, name: str, aligner: nn.Module) -> None:
        """添加对齐器"""
        self.aligners[name] = aligner
    
    def remove_aligner(self, name: str) -> None:
        """移除对齐器"""
        if name in self.aligners:
            del self.aligners[name]
    
    def enable_cross_attention(self) -> None:
        """启用交叉注意力"""
        if self.cross_attention is None:
            self.cross_attention = CrossModalAttention(
                self.projection_dim,
                self.projection_dim,
                num_heads=8
            )
        self.cross_attention_enabled = True
    
    def disable_cross_attention(self) -> None:
        """禁用交叉注意力"""
        self.cross_attention_enabled = False
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._monitor.reset()
        self._alignment_scores.clear()
        self._step = 0
        
        # 重置子对齐器
        for aligner in self.aligners.values():
            if hasattr(aligner, 'reset_stats'):
                aligner.reset_stats()
        
        if self.cross_attention is not None:
            self.cross_attention.reset_stats()
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断对齐状态"""
        diagnosis = {
            'method': self.config.method.value,
            'projection_dim': self.projection_dim,
            'num_aligners': len(self.aligners),
            'aligners': list(self.aligners.keys()),
            'cross_attention_enabled': self.cross_attention is not None,
            'total_steps': self._step,
            'avg_alignment_score': self.get_avg_alignment_score(),
            'modality_pairs': self._modality_pairs,
        }
        
        # 添加各对齐器的状态
        for name, aligner in self.aligners.items():
            if hasattr(aligner, 'get_stats'):
                stats = aligner.get_stats()
                diagnosis[f'{name}_avg_loss'] = stats.avg_loss
        
        # 检查问题
        issues = []
        if self.get_avg_alignment_score() < 0.3:
            issues.append("Low alignment score - consider adjusting learning rate or loss weights")
        
        stats = self._monitor.get_stats()
        if stats.avg_loss > 5.0:
            issues.append("High alignment loss - check data quality or model architecture")
        
        diagnosis['issues'] = issues
        diagnosis['is_healthy'] = len(issues) == 0
        
        return diagnosis
    
    def print_summary(self) -> None:
        """打印摘要"""
        summary = self._monitor.get_summary()
        diagnosis = self.diagnose()
        
        print("\n" + "="*60)
        print("CrossModalAligner Summary")
        print("="*60)
        print(f"Method: {self.config.method.value}")
        print(f"Projection dim: {self.projection_dim}")
        print(f"Aligners: {list(self.aligners.keys())}")
        print(f"Cross attention: {'Enabled' if self.cross_attention else 'Disabled'}")
        print(f"\nTotal steps: {summary['total_steps']}")
        print(f"Avg loss: {summary['avg_loss']:.4f}")
        print(f"Avg alignment score: {self.get_avg_alignment_score():.4f}")
        print(f"Loss trend: {summary['loss_trend']}")
        print(f"Is converging: {summary['is_converging']}")
        print(f"\nModality pairs: {self._modality_pairs}")
        print(f"Is healthy: {diagnosis['is_healthy']}")
        
        if diagnosis['issues']:
            print("\nIssues:")
            for issue in diagnosis['issues']:
                print(f"  - {issue}")
        
        print("="*60)


# ==================== 损失函数 ====================

class AlignmentLoss(nn.Module):
    """对齐损失函数
    
    支持多种损失组合：
    - InfoNCE
    - KL散度
    - MSE
    - 余弦相似度
    """
    
    def __init__(
        self, 
        config: CrossModalAlignmentConfig,
        learnable_temperature: bool = True
    ):
        super().__init__()
        self.config = config
        
        # 温度参数
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        else:
            self.register_buffer('temperature', torch.tensor(0.07))
        
        # 监控器
        self._monitor = AlignmentMonitor()
        self._step = 0
        
        # 损失权重
        self._loss_weights = {
            'contrastive': config.align_loss_weight,
            'kl': config.kl_loss_weight,
            'mse': 0.0,
            'cosine': 0.0,
        }
    
    def info_nce_loss(
        self, 
        features_a: Tensor, 
        features_b: Tensor,
        labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, float]]:
        """InfoNCE损失"""
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)
        
        logits = torch.matmul(features_a, features_b.T) / self.temperature
        
        batch_size = features_a.shape[0]
        if labels is None:
            labels = torch.arange(batch_size, device=features_a.device)
        
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        loss = (loss_a + loss_b) / 2
        
        # 计算准确率
        with torch.no_grad():
            acc_a = (logits.argmax(dim=1) == labels).float().mean()
            acc_b = (logits.T.argmax(dim=1) == labels).float().mean()
        
        metrics = {
            'info_nce_loss': loss.item(),
            'info_nce_acc_a': acc_a.item(),
            'info_nce_acc_b': acc_b.item(),
        }
        
        return loss, metrics
    
    def kl_divergence_loss(self, 
                          features_a: Tensor, 
                          features_b: Tensor,
                          temperature: float = 1.0) -> Tuple[Tensor, Dict[str, float]]:
        """KL散度损失"""
        p_a = F.softmax(features_a / temperature, dim=-1)
        p_b = F.softmax(features_b / temperature, dim=-1)
        
        kl_ab = F.kl_div(p_a.log(), p_b, reduction='batchmean')
        kl_ba = F.kl_div(p_b.log(), p_a, reduction='batchmean')
        loss = (kl_ab + kl_ba) / 2
        
        metrics = {
            'kl_loss': loss.item(),
            'kl_ab': kl_ab.item(),
            'kl_ba': kl_ba.item(),
        }
        
        return loss, metrics
    
    def mse_loss(self, features_a: Tensor, features_b: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """MSE损失"""
        loss = F.mse_loss(features_a, features_b)
        
        metrics = {'mse_loss': loss.item()}
        return loss, metrics
    
    def cosine_loss(self, features_a: Tensor, features_b: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """余弦相似度损失"""
        similarity = torch.cosine_similarity(features_a, features_b, dim=-1)
        loss = 1 - similarity.mean()
        
        metrics = {
            'cosine_loss': loss.item(),
            'cosine_similarity': similarity.mean().item(),
        }
        return loss, metrics
    
    def forward(
        self,
        features_a: Tensor,
        features_b: Tensor,
        labels: Optional[Tensor] = None
    ) -> Tuple[Tensor, Dict[str, float]]:
        """计算对齐损失
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
            labels: 可选的匹配标签
            
        Returns:
            总损失和指标
        """
        self._step += 1
        total_loss = torch.tensor(0.0, device=features_a.device)
        all_metrics = {}
        
        # InfoNCE损失
        if self._loss_weights['contrastive'] > 0:
            contrastive_loss, metrics = self.info_nce_loss(features_a, features_b, labels)
            total_loss = total_loss + self._loss_weights['contrastive'] * contrastive_loss
            all_metrics.update(metrics)
        
        # KL散度损失
        if self._loss_weights['kl'] > 0:
            kl_loss, metrics = self.kl_divergence_loss(features_a, features_b)
            total_loss = total_loss + self._loss_weights['kl'] * kl_loss
            all_metrics.update(metrics)
        
        # MSE损失
        if self._loss_weights['mse'] > 0:
            mse_loss, metrics = self.mse_loss(features_a, features_b)
            total_loss = total_loss + self._loss_weights['mse'] * mse_loss
            all_metrics.update(metrics)
        
        # 余弦损失
        if self._loss_weights['cosine'] > 0:
            cosine_loss, metrics = self.cosine_loss(features_a, features_b)
            total_loss = total_loss + self._loss_weights['cosine'] * cosine_loss
            all_metrics.update(metrics)
        
        all_metrics['total_align_loss'] = total_loss.item()
        all_metrics['temperature'] = self.temperature.item() if isinstance(self.temperature, nn.Parameter) else self.temperature
        
        # 更新监控
        self._monitor.record(total_loss.item(), all_metrics)
        
        return total_loss, all_metrics
    
    def set_loss_weights(self, **weights) -> None:
        """设置损失权重"""
        for key, value in weights.items():
            if key in self._loss_weights:
                self._loss_weights[key] = value
    
    def get_loss_weights(self) -> Dict[str, float]:
        """获取损失权重"""
        return self._loss_weights.copy()
    
    def get_monitor(self) -> AlignmentMonitor:
        """获取监控器"""
        return self._monitor
    
    def get_stats(self) -> AlignmentStats:
        """获取统计信息"""
        return self._monitor.get_stats()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._monitor.reset()
        self._step = 0
    
    def print_summary(self) -> None:
        """打印摘要"""
        summary = self._monitor.get_summary()
        print("\n" + "="*50)
        print("AlignmentLoss Summary")
        print("="*50)
        print(f"Loss weights: {self._loss_weights}")
        print(f"Temperature: {self.temperature.item() if isinstance(self.temperature, nn.Parameter) else self.temperature:.4f}")
        print(f"Total steps: {summary['total_steps']}")
        print(f"Avg loss: {summary['avg_loss']:.4f}")
        print(f"Avg accuracy A: {summary['avg_accuracy_a']:.4f}")
        print(f"Avg accuracy B: {summary['avg_accuracy_b']:.4f}")
        print(f"Loss trend: {summary['loss_trend']}")
        print("="*50)


# ==================== 工具函数 ====================

def create_aligner(
    config: CrossModalAlignmentConfig,
    embed_dim: int,
    enable_cross_attention: bool = True
) -> CrossModalAligner:
    """
    创建跨模态对齐器
    
    Args:
        config: 对齐配置
        embed_dim: 嵌入维度
        enable_cross_attention: 是否启用交叉注意力
        
    Returns:
        对齐器实例
    """
    return CrossModalAligner(config, embed_dim, enable_cross_attention)


def create_contrastive_loss(
    embed_dim: int,
    temperature: float = 0.07,
    loss_type: str = "info_nce",
    hard_negative_mining: bool = True
) -> ContrastiveLearningAlignment:
    """
    创建对比学习对齐模块
    
    Args:
        embed_dim: 嵌入维度
        temperature: 温度参数
        loss_type: 损失类型
        hard_negative_mining: 是否启用硬负样本挖掘
        
    Returns:
        对比学习模块
    """
    config = ContrastiveLearningConfig(
        temperature=temperature,
        loss_type=loss_type,
        hard_negative_mining=hard_negative_mining
    )
    return ContrastiveLearningAlignment(config, embed_dim)


def compute_alignment_metrics(
    features_a: Tensor,
    features_b: Tensor
) -> Dict[str, float]:
    """
    计算对齐指标
    
    Args:
        features_a: 模态A特征
        features_b: 模态B特征
        
    Returns:
        指标字典
    """
    with torch.no_grad():
        # 归一化
        norm_a = F.normalize(features_a, dim=-1)
        norm_b = F.normalize(features_b, dim=-1)
        
        # 余弦相似度
        cosine_sim = torch.cosine_similarity(norm_a, norm_b, dim=-1).mean()
        
        # 欧氏距离
        euclidean_dist = torch.cdist(features_a, features_b).mean()
        
        # 相关性
        mean_a = features_a.mean(dim=0)
        mean_b = features_b.mean(dim=0)
        cov = ((features_a - mean_a) * (features_b - mean_b)).mean()
        
        return {
            'cosine_similarity': cosine_sim.item(),
            'euclidean_distance': euclidean_dist.item(),
            'covariance': cov.item(),
        }


def analyze_alignment_quality(
    aligner: CrossModalAligner,
    features: Dict[str, Tensor]
) -> Dict[str, Any]:
    """
    分析对齐质量
    
    Args:
        aligner: 对齐器
        features: 模态特征
        
    Returns:
        分析结果
    """
    # 获取对齐后的特征
    with torch.no_grad():
        aligned_features, _, metrics = aligner(features, compute_loss=True)
    
    modalities = list(features.keys())
    if len(modalities) < 2:
        return {'error': 'Need at least 2 modalities'}
    
    # 计算对齐前后的指标
    before_metrics = compute_alignment_metrics(
        features[modalities[0]], 
        features[modalities[1]]
    )
    
    after_metrics = compute_alignment_metrics(
        aligned_features[modalities[0]],
        aligned_features[modalities[1]]
    )
    
    # 计算改善
    improvement = {
        'cosine_similarity_improvement': after_metrics['cosine_similarity'] - before_metrics['cosine_similarity'],
        'euclidean_distance_reduction': before_metrics['euclidean_distance'] - after_metrics['euclidean_distance'],
    }
    
    return {
        'before_alignment': before_metrics,
        'after_alignment': after_metrics,
        'improvement': improvement,
        'loss_metrics': metrics,
        'alignment_effective': improvement['cosine_similarity_improvement'] > 0,
    }


def print_alignment_analysis(
    aligner: CrossModalAligner,
    features: Dict[str, Tensor]
) -> None:
    """打印对齐分析"""
    analysis = analyze_alignment_quality(aligner, features)
    
    print("\n" + "="*60)
    print("Alignment Quality Analysis")
    print("="*60)
    
    print("\nBefore alignment:")
    for k, v in analysis['before_alignment'].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nAfter alignment:")
    for k, v in analysis['after_alignment'].items():
        print(f"  {k}: {v:.4f}")
    
    print("\nImprovement:")
    for k, v in analysis['improvement'].items():
        print(f"  {k}: {v:+.4f}")
    
    print(f"\nAlignment effective: {analysis['alignment_effective']}")
    print("="*60)


def recommend_alignment_method(
    num_modalities: int,
    data_paired: bool = True,
    memory_constrained: bool = False
) -> AlignmentMethod:
    """
    推荐对齐方法
    
    Args:
        num_modalities: 模态数量
        data_paired: 数据是否配对
        memory_constrained: 是否内存受限
        
    Returns:
        推荐的对齐方法
    """
    if not data_paired:
        return AlignmentMethod.KNOWLEDGE_DISTILL
    
    if memory_constrained:
        return AlignmentMethod.CONTRASTIVE
    
    if num_modalities == 2:
        return AlignmentMethod.CONTRASTIVE
    
    return AlignmentMethod.CROSS_ATTENTION


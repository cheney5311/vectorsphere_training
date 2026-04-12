# -*- coding: utf-8 -*-
"""
对齐模块 - 生产级实现

提供跨模态对齐的各种方法。

生产级特性：
- 多种对齐方法（对比学习、显式对齐、最优传输、跨模态注意力、CCA、混合对齐）
- 指标收集和监控
- 梯度检查和数值稳定性
- 批量处理和缓存
- 对齐质量评估
- 数据增强支持
- 组合对齐策略
"""

import logging
import time
import math
import threading
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class AlignmentStatus(Enum):
    """对齐状态"""
    READY = "ready"
    TRAINING = "training"
    EVALUATING = "evaluating"
    ERROR = "error"


class PoolingMethod(Enum):
    """池化方法"""
    MEAN = "mean"
    MAX = "max"
    CLS = "cls"  # 使用第一个token
    ATTENTION = "attention"  # 注意力加权


class LossType(Enum):
    """损失类型"""
    INFONCE = "infonce"
    TRIPLET = "triplet"
    MSE = "mse"
    COSINE = "cosine"
    COMBINED = "combined"


class AlignmentMethod(Enum):
    """对齐方法"""
    CONTRASTIVE = "contrastive"        # 对比学习
    EXPLICIT = "explicit"              # 显式对齐（MLP映射）
    OPTIMAL_TRANSPORT = "optimal_transport"  # 最优传输
    CROSS_MODAL_ATTENTION = "cross_modal_attention"  # 跨模态注意力
    CCA = "cca"                        # 典型相关分析
    HYBRID = "hybrid"                  # 混合对齐


class AugmentationType(Enum):
    """数据增强类型"""
    NONE = "none"
    DROPOUT = "dropout"
    NOISE = "noise"
    MIXUP = "mixup"
    CUTOUT = "cutout"


class QualityMetric(Enum):
    """质量指标类型"""
    COSINE_SIMILARITY = "cosine_similarity"
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    CORRELATION = "correlation"
    MUTUAL_INFORMATION = "mutual_information"


# ==================== 数据类 ====================

@dataclass
class AlignmentMetrics:
    """对齐指标"""
    total_alignments: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    avg_similarity: float = 0.0
    avg_loss: float = 0.0
    error_count: int = 0
    last_alignment_time: Optional[datetime] = None
    quality_scores: List[float] = field(default_factory=list)
    
    def record_alignment(self, time_taken: float, similarity: float = 0.0, 
                        loss: float = 0.0) -> None:
        """记录对齐"""
        self.total_alignments += 1
        self.total_time += time_taken
        self.avg_time = self.total_time / self.total_alignments
        self.last_alignment_time = datetime.now()
        
        # 更新平均相似度
        if similarity > 0:
            self.avg_similarity = (
                (self.avg_similarity * (self.total_alignments - 1) + similarity) 
                / self.total_alignments
            )
        
        # 更新平均损失
        if loss > 0:
            self.avg_loss = (
                (self.avg_loss * (self.total_alignments - 1) + loss) 
                / self.total_alignments
            )
    
    def record_error(self) -> None:
        """记录错误"""
        self.error_count += 1
    
    def add_quality_score(self, score: float, max_history: int = 100) -> None:
        """添加质量分数"""
        self.quality_scores.append(score)
        if len(self.quality_scores) > max_history:
            self.quality_scores = self.quality_scores[-max_history:]
    
    def get_recent_quality(self, n: int = 10) -> float:
        """获取最近的平均质量"""
        if not self.quality_scores:
            return 0.0
        recent = self.quality_scores[-n:]
        return sum(recent) / len(recent)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_alignments": self.total_alignments,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "avg_similarity": self.avg_similarity,
            "avg_loss": self.avg_loss,
            "error_count": self.error_count,
            "last_alignment_time": self.last_alignment_time.isoformat() if self.last_alignment_time else None,
            "recent_quality": self.get_recent_quality()
        }


@dataclass
class AlignmentConfig:
    """对齐配置"""
    method: AlignmentMethod = AlignmentMethod.CONTRASTIVE
    hidden_size: int = 768
    projection_dim: int = 256
    temperature: float = 0.07
    num_heads: int = 8
    dropout: float = 0.1
    
    # 对比学习配置
    use_hard_negatives: bool = False
    hard_negative_weight: float = 0.5
    
    # 最优传输配置
    sinkhorn_iterations: int = 3
    sinkhorn_epsilon: float = 0.1
    
    # 池化配置
    pooling_method: PoolingMethod = PoolingMethod.MEAN
    
    # 指标配置
    enable_metrics: bool = True
    metrics_history_size: int = 100
    
    # 数据增强配置
    augmentation_type: AugmentationType = AugmentationType.NONE
    augmentation_prob: float = 0.1
    noise_std: float = 0.01
    mixup_alpha: float = 0.2
    
    # 数值稳定性
    eps: float = 1e-8
    gradient_clip: float = 1.0
    
    # CCA 配置
    cca_reg: float = 1e-4
    cca_components: int = 64
    
    # 混合对齐配置
    hybrid_methods: List[str] = field(default_factory=lambda: ["contrastive", "explicit"])
    hybrid_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlignmentModule(nn.Module, ABC):
    """
    对齐模块基类 - 生产级实现
    
    所有对齐模块的抽象基类，提供：
    - 指标收集和监控
    - 多种池化方法
    - 数据增强支持
    - 对齐质量评估
    - 数值稳定性保障
    """
    
    def __init__(self, config: AlignmentConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # 状态
        self._status = AlignmentStatus.READY
        
        # 指标
        self._metrics = AlignmentMetrics() if config.enable_metrics else None
        self._metrics_lock = threading.Lock()
        
        # 注意力池化（如果需要）
        if config.pooling_method == PoolingMethod.ATTENTION:
            self._attention_pooling = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.Tanh(),
                nn.Linear(config.hidden_size // 4, 1)
            )
        else:
            self._attention_pooling = None
        
        # 数据增强
        self._augmentation_dropout = nn.Dropout(config.augmentation_prob) if config.augmentation_type == AugmentationType.DROPOUT else None
    
    @property
    def status(self) -> AlignmentStatus:
        """获取状态"""
        return self._status
    
    @property
    def metrics(self) -> Optional[AlignmentMetrics]:
        """获取指标"""
        return self._metrics
    
    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        池化特征
        
        Args:
            features: 输入特征 [B, seq_len, hidden]
            
        Returns:
            池化后的特征 [B, hidden]
        """
        method = self.config.pooling_method
        
        if method == PoolingMethod.MEAN:
            return features.mean(dim=1)
        elif method == PoolingMethod.MAX:
            return features.max(dim=1)[0]
        elif method == PoolingMethod.CLS:
            return features[:, 0, :]
        elif method == PoolingMethod.ATTENTION:
            # 注意力加权池化
            weights = self._attention_pooling(features)  # [B, seq_len, 1]
            weights = F.softmax(weights, dim=1)
            return (features * weights).sum(dim=1)
        else:
            return features.mean(dim=1)
    
    def _apply_augmentation(self, features: torch.Tensor) -> torch.Tensor:
        """
        应用数据增强
        
        Args:
            features: 输入特征
            
        Returns:
            增强后的特征
        """
        if not self.training:
            return features
        
        aug_type = self.config.augmentation_type
        
        if aug_type == AugmentationType.NONE:
            return features
        elif aug_type == AugmentationType.DROPOUT:
            return self._augmentation_dropout(features)
        elif aug_type == AugmentationType.NOISE:
            noise = torch.randn_like(features) * self.config.noise_std
            return features + noise
        elif aug_type == AugmentationType.MIXUP:
            # 批内混合
            batch_size = features.shape[0]
            if batch_size > 1:
                indices = torch.randperm(batch_size, device=features.device)
                lam = torch.distributions.Beta(
                    self.config.mixup_alpha, self.config.mixup_alpha
                ).sample().item()
                return lam * features + (1 - lam) * features[indices]
            return features
        elif aug_type == AugmentationType.CUTOUT:
            # 随机置零
            mask = torch.rand_like(features) > self.config.augmentation_prob
            return features * mask.float()
        else:
            return features
    
    def _compute_similarity(self, features_a: torch.Tensor, 
                           features_b: torch.Tensor) -> torch.Tensor:
        """
        计算相似度
        
        Args:
            features_a: 特征A [B, hidden]
            features_b: 特征B [B, hidden]
            
        Returns:
            相似度 [B]
        """
        return torch.cosine_similarity(features_a, features_b, dim=-1)
    
    def _check_numerical_stability(self, tensor: torch.Tensor, 
                                   name: str = "tensor") -> torch.Tensor:
        """
        检查数值稳定性
        
        Args:
            tensor: 输入张量
            name: 张量名称（用于日志）
            
        Returns:
            稳定化后的张量
        """
        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {name}, replacing with zeros")
            tensor = torch.nan_to_num(tensor, nan=0.0)
        
        if torch.isinf(tensor).any():
            logger.warning(f"Inf detected in {name}, clamping values")
            tensor = torch.clamp(tensor, min=-1e10, max=1e10)
        
        return tensor
    
    @contextmanager
    def _timed_alignment(self):
        """计时对齐上下文管理器"""
        start_time = time.time()
        error_occurred = False
        try:
            yield
        except Exception as e:
            error_occurred = True
            if self._metrics:
                with self._metrics_lock:
                    self._metrics.record_error()
            raise
        finally:
            if not error_occurred and self._metrics:
                elapsed = time.time() - start_time
                with self._metrics_lock:
                    self._metrics.record_alignment(elapsed)
    
    @abstractmethod
    def align(
        self, 
        features_a: torch.Tensor, 
        features_b: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对齐两个模态的特征
        
        Args:
            features_a: 模态A的特征 [B, seq_a, hidden]
            features_b: 模态B的特征 [B, seq_b, hidden]
            
        Returns:
            对齐后的特征 (aligned_a, aligned_b)
        """
        pass
    
    def compute_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        计算对齐损失
        
        默认返回0，子类应根据需要重写。
        """
        return torch.tensor(0.0, device=features_a.device)
    
    def evaluate_quality(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        metric: QualityMetric = QualityMetric.COSINE_SIMILARITY
    ) -> Dict[str, float]:
        """
        评估对齐质量
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
            metric: 质量指标类型
            
        Returns:
            质量指标字典
        """
        with torch.no_grad():
            aligned_a, aligned_b = self.align(features_a, features_b)
            
            # 池化
            rep_a = self._pool_features(aligned_a) if aligned_a.dim() == 3 else aligned_a
            rep_b = self._pool_features(aligned_b) if aligned_b.dim() == 3 else aligned_b
            
            results = {}
            
            if metric == QualityMetric.COSINE_SIMILARITY:
                sim = self._compute_similarity(rep_a, rep_b)
                results["cosine_similarity"] = sim.mean().item()
            elif metric == QualityMetric.EUCLIDEAN_DISTANCE:
                dist = torch.norm(rep_a - rep_b, dim=-1)
                results["euclidean_distance"] = dist.mean().item()
            elif metric == QualityMetric.CORRELATION:
                # 计算相关系数
                rep_a_centered = rep_a - rep_a.mean(dim=1, keepdim=True)
                rep_b_centered = rep_b - rep_b.mean(dim=1, keepdim=True)
                numerator = (rep_a_centered * rep_b_centered).sum(dim=1)
                denominator = (rep_a_centered.norm(dim=1) * rep_b_centered.norm(dim=1) + self.config.eps)
                corr = numerator / denominator
                results["correlation"] = corr.mean().item()
            
            # 记录质量分数
            if self._metrics:
                main_score = list(results.values())[0] if results else 0.0
                with self._metrics_lock:
                    self._metrics.add_quality_score(main_score)
            
            return results
    
    def forward(
        self, 
        features_a: torch.Tensor, 
        features_b: torch.Tensor,
        apply_augmentation: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 应用数据增强
        if apply_augmentation:
            features_a = self._apply_augmentation(features_a)
            features_b = self._apply_augmentation(features_b)
        
        # 执行对齐
        with self._timed_alignment():
            aligned_a, aligned_b = self.align(features_a, features_b, **kwargs)
        
        # 检查数值稳定性
        aligned_a = self._check_numerical_stability(aligned_a, "aligned_a")
        aligned_b = self._check_numerical_stability(aligned_b, "aligned_b")
        
        return aligned_a, aligned_b
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        if self._metrics:
            with self._metrics_lock:
                return self._metrics.to_dict()
        return {}
    
    def reset_metrics(self) -> None:
        """重置指标"""
        if self._metrics:
            with self._metrics_lock:
                self._metrics = AlignmentMetrics()


class ContrastiveAlignment(AlignmentModule):
    """
    对比学习对齐 - 生产级实现
    
    使用InfoNCE损失进行对比学习，支持：
    - 困难负样本挖掘
    - 三元组损失
    - 标签平滑
    - 温度自适应
    """
    
    def __init__(self, config: AlignmentConfig):
        super().__init__(config)
        
        # 投影头
        self.projector_a = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.projection_dim)
        )
        
        self.projector_b = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.projection_dim)
        )
        
        self.temperature = config.temperature
        self.use_hard_negatives = config.use_hard_negatives
        
        # 可学习的温度参数
        self.learnable_temperature = nn.Parameter(torch.tensor(config.temperature))
        
        # 相似度历史（用于自适应温度）
        self._sim_history = deque(maxlen=100)
    
    def _get_effective_temperature(self) -> torch.Tensor:
        """获取有效温度"""
        # 使用可学习温度，但限制范围
        return torch.clamp(self.learnable_temperature, min=0.01, max=1.0)
    
    def _compute_similarity_matrix(self, proj_a: torch.Tensor, 
                                   proj_b: torch.Tensor) -> torch.Tensor:
        """计算相似度矩阵"""
        temperature = self._get_effective_temperature()
        sim_matrix = torch.matmul(proj_a, proj_b.t()) / temperature
        return self._check_numerical_stability(sim_matrix, "sim_matrix")
    
    def align(
        self, 
        features_a: torch.Tensor, 
        features_b: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对比学习对齐"""
        # 使用配置的池化方法
        rep_a = self._pool_features(features_a)  # [B, hidden]
        rep_b = self._pool_features(features_b)  # [B, hidden]
        
        # 投影并归一化
        proj_a = F.normalize(self.projector_a(rep_a), dim=-1, eps=self.config.eps)
        proj_b = F.normalize(self.projector_b(rep_b), dim=-1, eps=self.config.eps)
        
        return proj_a, proj_b
    
    def compute_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        loss_type: LossType = LossType.INFONCE,
        label_smoothing: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
            loss_type: 损失类型
            label_smoothing: 标签平滑系数
        """
        proj_a, proj_b = self.align(features_a, features_b)
        
        batch_size = proj_a.shape[0]
        device = proj_a.device
        
        # 计算相似度矩阵
        sim_matrix = self._compute_similarity_matrix(proj_a, proj_b)
        
        # 记录相似度统计
        if self._metrics:
            diag_sim = sim_matrix.diag().mean().item()
            self._sim_history.append(diag_sim)
        
        if loss_type == LossType.INFONCE:
            loss = self._compute_infonce_loss(sim_matrix, batch_size, device, label_smoothing)
        elif loss_type == LossType.TRIPLET:
            loss = self._compute_triplet_loss(sim_matrix)
        elif loss_type == LossType.COMBINED:
            loss_infonce = self._compute_infonce_loss(sim_matrix, batch_size, device, label_smoothing)
            loss_triplet = self._compute_triplet_loss(sim_matrix)
            loss = 0.7 * loss_infonce + 0.3 * loss_triplet
        else:
            loss = self._compute_infonce_loss(sim_matrix, batch_size, device, label_smoothing)
        
        # 添加困难负样本损失
        if self.use_hard_negatives:
            labels = torch.arange(batch_size, device=device)
            hard_neg_loss = self._compute_hard_negative_loss(sim_matrix, labels)
            loss = loss + self.config.hard_negative_weight * hard_neg_loss
        
        # 更新指标
        if self._metrics:
            with self._metrics_lock:
                self._metrics.avg_loss = (
                    (self._metrics.avg_loss * self._metrics.total_alignments + loss.item())
                    / (self._metrics.total_alignments + 1) if self._metrics.total_alignments > 0 else loss.item()
                )
        
        return loss
    
    def _compute_infonce_loss(self, sim_matrix: torch.Tensor, batch_size: int,
                              device: torch.device, label_smoothing: float = 0.0) -> torch.Tensor:
        """计算InfoNCE损失"""
        labels = torch.arange(batch_size, device=device)
        
        # 双向InfoNCE（带标签平滑）
        loss_a2b = F.cross_entropy(sim_matrix, labels, label_smoothing=label_smoothing)
        loss_b2a = F.cross_entropy(sim_matrix.t(), labels, label_smoothing=label_smoothing)
        
        return (loss_a2b + loss_b2a) / 2
    
    def _compute_triplet_loss(self, sim_matrix: torch.Tensor, 
                              margin: float = 0.2) -> torch.Tensor:
        """计算三元组损失"""
        batch_size = sim_matrix.shape[0]
        
        # 正样本相似度（对角线）
        pos_sim = sim_matrix.diag()
        
        # 困难负样本
        mask = torch.eye(batch_size, device=sim_matrix.device).bool()
        neg_sim_a = sim_matrix.masked_fill(mask, float('-inf')).max(dim=1)[0]
        neg_sim_b = sim_matrix.t().masked_fill(mask, float('-inf')).max(dim=1)[0]
        
        # 三元组损失
        loss_a = F.relu(neg_sim_a - pos_sim + margin)
        loss_b = F.relu(neg_sim_b - pos_sim + margin)
        
        return (loss_a.mean() + loss_b.mean()) / 2
    
    def _compute_hard_negative_loss(
        self, 
        sim_matrix: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """计算困难负样本损失"""
        batch_size = sim_matrix.shape[0]
        
        # 获取困难负样本
        mask = torch.eye(batch_size, device=sim_matrix.device).bool()
        neg_sim = sim_matrix.masked_fill(mask, float('-inf'))
        hard_negatives = neg_sim.max(dim=1)[0]
        
        # 正样本相似度
        pos_sim = sim_matrix.diag()
        
        # 对比损失
        loss = F.relu(hard_negatives - pos_sim + 0.2).mean()
        
        return loss
    
    def get_similarity_stats(self) -> Dict[str, float]:
        """获取相似度统计"""
        if not self._sim_history:
            return {"mean": 0.0, "std": 0.0}
        
        history = list(self._sim_history)
        return {
            "mean": sum(history) / len(history),
            "std": (sum((x - sum(history)/len(history))**2 for x in history) / len(history)) ** 0.5,
            "min": min(history),
            "max": max(history)
        }


class ExplicitAlignment(AlignmentModule):
    """
    显式对齐 - 生产级实现
    
    通过MLP直接映射到共享空间，支持：
    - 多层映射
    - 残差连接
    - 正交约束
    - 多种损失函数
    """
    
    def __init__(self, config: AlignmentConfig):
        super().__init__(config)
        
        # 模态到共享空间的映射
        self.mapper_a = self._build_mapper(config)
        self.mapper_b = self._build_mapper(config)
        
        # 残差投影（用于维度不匹配时）
        if config.hidden_size != config.projection_dim:
            self.residual_proj_a = nn.Linear(config.hidden_size, config.projection_dim)
            self.residual_proj_b = nn.Linear(config.hidden_size, config.projection_dim)
        else:
            self.residual_proj_a = None
            self.residual_proj_b = None
        
        # 正交约束权重
        self._orthogonal_weight = 0.01
    
    def _build_mapper(self, config: AlignmentConfig) -> nn.Module:
        """构建映射网络"""
        return nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.LayerNorm(config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.projection_dim),
            nn.LayerNorm(config.projection_dim)
        )
    
    def _compute_orthogonal_loss(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """计算正交约束损失"""
        # W^T W 应该接近单位矩阵
        wtw = torch.matmul(weight_matrix.t(), weight_matrix)
        identity = torch.eye(wtw.shape[0], device=wtw.device)
        return F.mse_loss(wtw, identity)
    
    def align(
        self, 
        features_a: torch.Tensor, 
        features_b: torch.Tensor,
        use_residual: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """显式对齐：MLP映射"""
        mapped_a = self.mapper_a(features_a)
        mapped_b = self.mapper_b(features_b)
        
        # 可选残差连接
        if use_residual:
            if self.residual_proj_a is not None:
                residual_a = self.residual_proj_a(features_a)
                residual_b = self.residual_proj_b(features_b)
            else:
                residual_a = features_a
                residual_b = features_b
            
            aligned_a = mapped_a + 0.1 * residual_a
            aligned_b = mapped_b + 0.1 * residual_b
        else:
            aligned_a = mapped_a
            aligned_b = mapped_b
        
        return aligned_a, aligned_b
    
    def compute_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        loss_type: LossType = LossType.MSE,
        use_orthogonal_constraint: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        计算对齐损失
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
            loss_type: 损失类型（MSE/COSINE/COMBINED）
            use_orthogonal_constraint: 是否使用正交约束
        """
        aligned_a, aligned_b = self.align(features_a, features_b)
        
        # 池化
        rep_a = self._pool_features(aligned_a) if aligned_a.dim() == 3 else aligned_a
        rep_b = self._pool_features(aligned_b) if aligned_b.dim() == 3 else aligned_b
        
        # 计算主损失
        if loss_type == LossType.MSE:
            loss = F.mse_loss(rep_a, rep_b)
        elif loss_type == LossType.COSINE:
            similarity = self._compute_similarity(rep_a, rep_b)
            loss = 1 - similarity.mean()
        elif loss_type == LossType.COMBINED:
            mse_loss = F.mse_loss(rep_a, rep_b)
            cosine_loss = 1 - self._compute_similarity(rep_a, rep_b).mean()
            loss = 0.5 * mse_loss + 0.5 * cosine_loss
        else:
            loss = F.mse_loss(rep_a, rep_b)
        
        # 添加正交约束
        if use_orthogonal_constraint:
            # 获取第一个线性层的权重
            first_layer_a = list(self.mapper_a.children())[0]
            first_layer_b = list(self.mapper_b.children())[0]
            if isinstance(first_layer_a, nn.Linear) and isinstance(first_layer_b, nn.Linear):
                ortho_loss_a = self._compute_orthogonal_loss(first_layer_a.weight)
                ortho_loss_b = self._compute_orthogonal_loss(first_layer_b.weight)
                loss = loss + self._orthogonal_weight * (ortho_loss_a + ortho_loss_b)
        
        return loss


class OptimalTransportAlignment(AlignmentModule):
    """
    最优传输对齐 - 生产级实现
    
    使用Sinkhorn算法进行最优传输对齐，支持：
    - 自适应迭代次数
    - 多种代价函数
    - 批量并行处理
    - 传输计划可视化
    """
    
    def __init__(self, config: AlignmentConfig):
        super().__init__(config)
        
        self.sinkhorn_iterations = config.sinkhorn_iterations
        self.epsilon = config.sinkhorn_epsilon
        
        # 投影层
        self.projector = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.projection_dim)
        )
        
        # 存储最近的传输计划（用于调试和可视化）
        self._last_transport_plans: List[torch.Tensor] = []
    
    def _compute_cost_matrix(self, a: torch.Tensor, b: torch.Tensor,
                            cost_type: str = "cosine") -> torch.Tensor:
        """
        计算代价矩阵
        
        Args:
            a: 特征A [seq_a, dim]
            b: 特征B [seq_b, dim]
            cost_type: 代价类型（cosine/euclidean/dot）
        """
        if cost_type == "cosine":
            a_norm = F.normalize(a, dim=-1, eps=self.config.eps)
            b_norm = F.normalize(b, dim=-1, eps=self.config.eps)
            cost = 1 - torch.matmul(a_norm, b_norm.t())
        elif cost_type == "euclidean":
            # [seq_a, 1, dim] - [1, seq_b, dim]
            cost = torch.cdist(a.unsqueeze(0), b.unsqueeze(0)).squeeze(0)
        else:  # dot product (negative similarity as cost)
            cost = -torch.matmul(a, b.t())
        
        return self._check_numerical_stability(cost, "cost_matrix")
    
    def _sinkhorn(
        self, 
        cost_matrix: torch.Tensor, 
        mu: torch.Tensor, 
        nu: torch.Tensor,
        adaptive: bool = True
    ) -> torch.Tensor:
        """
        Sinkhorn算法（带自适应收敛）
        
        Args:
            cost_matrix: 代价矩阵 [m, n]
            mu: 源分布 [m]
            nu: 目标分布 [n]
            adaptive: 是否使用自适应迭代
            
        Returns:
            传输计划 [m, n]
        """
        K = torch.exp(-cost_matrix / (self.epsilon + self.config.eps))
        K = self._check_numerical_stability(K, "K_matrix")
        
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)
        
        max_iters = self.sinkhorn_iterations * 3 if adaptive else self.sinkhorn_iterations
        tolerance = 1e-6
        
        for i in range(max_iters):
            u_prev = u.clone()
            
            u = mu / (K @ v + self.config.eps)
            v = nu / (K.t() @ u + self.config.eps)
            
            # 自适应收敛检查
            if adaptive and i >= self.sinkhorn_iterations:
                if torch.norm(u - u_prev) < tolerance:
                    break
        
        transport_plan = u.unsqueeze(-1) * K * v.unsqueeze(0)
        return self._check_numerical_stability(transport_plan, "transport_plan")
    
    def _batch_sinkhorn(self, cost_matrices: torch.Tensor,
                       mu: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
        """批量Sinkhorn（并行处理）"""
        batch_size = cost_matrices.shape[0]
        K = torch.exp(-cost_matrices / (self.epsilon + self.config.eps))
        
        u = torch.ones(batch_size, mu.shape[0], device=cost_matrices.device)
        v = torch.ones(batch_size, nu.shape[0], device=cost_matrices.device)
        
        for _ in range(self.sinkhorn_iterations):
            u = mu / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + self.config.eps)
            v = nu / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + self.config.eps)
        
        # [B, m, 1] * [B, m, n] * [B, 1, n]
        transport_plans = u.unsqueeze(-1) * K * v.unsqueeze(1)
        return transport_plans
    
    def align(
        self, 
        features_a: torch.Tensor, 
        features_b: torch.Tensor,
        cost_type: str = "cosine",
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """最优传输对齐"""
        batch_size = features_a.shape[0]
        
        # 投影
        proj_a = self.projector(features_a)  # [B, seq_a, proj_dim]
        proj_b = self.projector(features_b)  # [B, seq_b, proj_dim]
        
        aligned_a_list = []
        aligned_b_list = []
        self._last_transport_plans = []
        
        for i in range(batch_size):
            a = proj_a[i]  # [seq_a, proj_dim]
            b = proj_b[i]  # [seq_b, proj_dim]
            
            # 计算代价矩阵
            cost_matrix = self._compute_cost_matrix(a, b, cost_type)
            
            # 均匀分布
            mu = torch.ones(a.shape[0], device=a.device) / a.shape[0]
            nu = torch.ones(b.shape[0], device=b.device) / b.shape[0]
            
            # Sinkhorn
            transport_plan = self._sinkhorn(cost_matrix, mu, nu)
            self._last_transport_plans.append(transport_plan.detach())
            
            # 传输特征
            aligned_a = transport_plan @ b
            aligned_b = transport_plan.t() @ a
            
            aligned_a_list.append(aligned_a)
            aligned_b_list.append(aligned_b)
        
        aligned_a = torch.stack(aligned_a_list, dim=0)
        aligned_b = torch.stack(aligned_b_list, dim=0)
        
        return aligned_a, aligned_b
    
    def compute_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        regularization: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """
        计算Wasserstein距离
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
            regularization: 熵正则化权重
        """
        aligned_a, aligned_b = self.align(features_a, features_b)
        
        # 池化
        rep_a = self._pool_features(aligned_a)
        rep_b = self._pool_features(aligned_b) if aligned_b.dim() == 3 else aligned_b
        
        # MSE 损失
        loss = F.mse_loss(rep_a, rep_b)
        
        # 可选：添加传输计划的熵正则化
        if regularization > 0 and self._last_transport_plans:
            entropy_loss = sum(
                -(plan * torch.log(plan + self.config.eps)).sum()
                for plan in self._last_transport_plans
            ) / len(self._last_transport_plans)
            loss = loss - regularization * entropy_loss
        
        return loss
    
    def get_transport_plans(self) -> List[torch.Tensor]:
        """获取最近的传输计划（用于可视化）"""
        return self._last_transport_plans


class CrossModalAttentionAlignment(AlignmentModule):
    """
    跨模态注意力对齐 - 生产级实现
    
    使用交叉注意力进行对齐，支持：
    - 多层交叉注意力
    - 注意力权重可视化
    - 门控机制
    - 位置编码
    """
    
    def __init__(self, config: AlignmentConfig):
        super().__init__(config)
        
        # 确保num_heads能整除hidden_size
        self.num_heads = config.num_heads
        while config.hidden_size % self.num_heads != 0:
            self.num_heads -= 1
        
        # A -> B 的交叉注意力
        self.cross_attn_a2b = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=self.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # B -> A 的交叉注意力
        self.cross_attn_b2a = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=self.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.layer_norm_a = nn.LayerNorm(config.hidden_size)
        self.layer_norm_b = nn.LayerNorm(config.hidden_size)
        
        # 门控机制
        self.gate_a = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
        self.gate_b = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.Sigmoid()
        )
        
        # FFN（前馈网络）
        self.ffn_a = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        self.ffn_norm_a = nn.LayerNorm(config.hidden_size)
        self.ffn_norm_b = nn.LayerNorm(config.hidden_size)
        
        # 存储注意力权重
        self._last_attention_weights: Dict[str, torch.Tensor] = {}
    
    def _apply_gated_attention(self, original: torch.Tensor, attended: torch.Tensor,
                               gate: nn.Module) -> torch.Tensor:
        """应用门控注意力"""
        combined = torch.cat([original, attended], dim=-1)
        gate_values = gate(combined)
        return original * (1 - gate_values) + attended * gate_values
    
    def align(
        self, 
        features_a: torch.Tensor, 
        features_b: torch.Tensor,
        use_gating: bool = True,
        return_attention: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        跨模态注意力对齐
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
            use_gating: 是否使用门控
            return_attention: 是否存储注意力权重
        """
        # A关注B
        attended_a, attn_weights_a2b = self.cross_attn_a2b(
            features_a, features_b, features_b,
            need_weights=return_attention
        )
        
        # B关注A
        attended_b, attn_weights_b2a = self.cross_attn_b2a(
            features_b, features_a, features_a,
            need_weights=return_attention
        )
        
        # 存储注意力权重
        if return_attention:
            self._last_attention_weights = {
                "a2b": attn_weights_a2b.detach() if attn_weights_a2b is not None else None,
                "b2a": attn_weights_b2a.detach() if attn_weights_b2a is not None else None
            }
        
        # 应用门控或简单残差
        if use_gating:
            aligned_a = self._apply_gated_attention(features_a, attended_a, self.gate_a)
            aligned_b = self._apply_gated_attention(features_b, attended_b, self.gate_b)
        else:
            aligned_a = features_a + attended_a
            aligned_b = features_b + attended_b
        
        # Layer Norm
        aligned_a = self.layer_norm_a(aligned_a)
        aligned_b = self.layer_norm_b(aligned_b)
        
        # FFN
        aligned_a = self.ffn_norm_a(aligned_a + self.ffn_a(aligned_a))
        aligned_b = self.ffn_norm_b(aligned_b + self.ffn_b(aligned_b))
        
        return aligned_a, aligned_b
    
    def compute_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        loss_weights: Dict[str, float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        计算对齐损失
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
            loss_weights: 损失权重字典
        """
        loss_weights = loss_weights or {"cosine": 0.5, "mse": 0.3, "diversity": 0.2}
        
        aligned_a, aligned_b = self.align(features_a, features_b, return_attention=True)
        
        # 池化
        rep_a = self._pool_features(aligned_a)
        rep_b = self._pool_features(aligned_b)
        
        total_loss = torch.tensor(0.0, device=features_a.device)
        
        # 余弦相似度损失
        if "cosine" in loss_weights:
            similarity = self._compute_similarity(rep_a, rep_b)
            cosine_loss = 1 - similarity.mean()
            total_loss = total_loss + loss_weights["cosine"] * cosine_loss
        
        # MSE损失
        if "mse" in loss_weights:
            mse_loss = F.mse_loss(rep_a, rep_b)
            total_loss = total_loss + loss_weights["mse"] * mse_loss
        
        # 多样性损失（鼓励注意力分散）
        if "diversity" in loss_weights and self._last_attention_weights.get("a2b") is not None:
            attn = self._last_attention_weights["a2b"]
            # 计算注意力熵
            entropy = -(attn * torch.log(attn + self.config.eps)).sum(dim=-1).mean()
            diversity_loss = -entropy  # 最大化熵
            total_loss = total_loss + loss_weights["diversity"] * diversity_loss
        
        return total_loss
    
    def get_attention_weights(self) -> Dict[str, Optional[torch.Tensor]]:
        """获取最近的注意力权重"""
        return self._last_attention_weights


# ==================== CCA 对齐 ====================

class CCAAlignment(AlignmentModule):
    """
    典型相关分析对齐 - 生产级实现
    
    使用深度CCA进行跨模态对齐，支持：
    - 可学习的CCA映射
    - 正则化
    - 批量处理
    """
    
    def __init__(self, config: AlignmentConfig):
        super().__init__(config)
        
        cca_dim = config.cca_components
        
        # 模态编码器
        self.encoder_a = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, cca_dim)
        )
        
        self.encoder_b = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, cca_dim)
        )
        
        self.reg = config.cca_reg
        self._correlation_history = deque(maxlen=100)
    
    def _compute_cca_loss(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """
        计算CCA损失
        
        Args:
            h1: 模态1的隐藏表示 [B, D]
            h2: 模态2的隐藏表示 [B, D]
        """
        batch_size = h1.shape[0]
        dim = h1.shape[1]
        
        # 中心化
        h1_centered = h1 - h1.mean(dim=0, keepdim=True)
        h2_centered = h2 - h2.mean(dim=0, keepdim=True)
        
        # 协方差矩阵
        sigma11 = (h1_centered.t() @ h1_centered) / (batch_size - 1) + self.reg * torch.eye(dim, device=h1.device)
        sigma22 = (h2_centered.t() @ h2_centered) / (batch_size - 1) + self.reg * torch.eye(dim, device=h1.device)
        sigma12 = (h1_centered.t() @ h2_centered) / (batch_size - 1)
        
        # 计算相关系数
        try:
            sigma11_inv_sqrt = torch.inverse(torch.cholesky(sigma11)).t()
            sigma22_inv_sqrt = torch.inverse(torch.cholesky(sigma22)).t()
            
            T = sigma11_inv_sqrt @ sigma12 @ sigma22_inv_sqrt
            
            # 使用奇异值分解
            U, S, V = torch.svd(T)
            
            # 相关系数之和（越大越好）
            corr = S.sum()
            
            # 记录相关系数
            self._correlation_history.append(corr.item())
            
            # 返回负相关作为损失
            return -corr
        except Exception as e:
            logger.warning(f"CCA computation failed: {e}, using MSE fallback")
            return F.mse_loss(h1, h2)
    
    def align(
        self, 
        features_a: torch.Tensor, 
        features_b: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """CCA对齐"""
        # 池化
        rep_a = self._pool_features(features_a)
        rep_b = self._pool_features(features_b)
        
        # 编码
        proj_a = self.encoder_a(rep_a)
        proj_b = self.encoder_b(rep_b)
        
        return proj_a, proj_b
    
    def compute_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """计算CCA损失"""
        proj_a, proj_b = self.align(features_a, features_b)
        return self._compute_cca_loss(proj_a, proj_b)
    
    def get_correlation_stats(self) -> Dict[str, float]:
        """获取相关系数统计"""
        if not self._correlation_history:
            return {"mean": 0.0, "recent": 0.0}
        
        history = list(self._correlation_history)
        return {
            "mean": sum(history) / len(history),
            "recent": history[-1] if history else 0.0,
            "max": max(history),
            "min": min(history)
        }


# ==================== 混合对齐 ====================

class HybridAlignment(AlignmentModule):
    """
    混合对齐 - 生产级实现
    
    组合多种对齐方法，支持：
    - 动态权重
    - 级联对齐
    - 自适应选择
    """
    
    def __init__(self, config: AlignmentConfig):
        super().__init__(config)
        
        # 创建子对齐模块
        self._sub_alignments: nn.ModuleDict = nn.ModuleDict()
        self._weights: Dict[str, float] = {}
        
        for method_name, weight in zip(config.hybrid_methods, config.hybrid_weights):
            method = AlignmentMethod(method_name)
            sub_config = AlignmentConfig(
                method=method,
                hidden_size=config.hidden_size,
                projection_dim=config.projection_dim,
                temperature=config.temperature,
                num_heads=config.num_heads,
                dropout=config.dropout,
                enable_metrics=False  # 子模块不单独收集指标
            )
            
            # 创建子模块
            if method == AlignmentMethod.CONTRASTIVE:
                self._sub_alignments[method_name] = ContrastiveAlignment(sub_config)
            elif method == AlignmentMethod.EXPLICIT:
                self._sub_alignments[method_name] = ExplicitAlignment(sub_config)
            elif method == AlignmentMethod.CROSS_MODAL_ATTENTION:
                self._sub_alignments[method_name] = CrossModalAttentionAlignment(sub_config)
            elif method == AlignmentMethod.OPTIMAL_TRANSPORT:
                self._sub_alignments[method_name] = OptimalTransportAlignment(sub_config)
            elif method == AlignmentMethod.CCA:
                self._sub_alignments[method_name] = CCAAlignment(sub_config)
            
            self._weights[method_name] = weight
        
        # 融合层
        total_dim = config.projection_dim * len(self._sub_alignments)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.projection_dim)
        )
        
        # 可学习的权重
        self.learnable_weights = nn.Parameter(
            torch.tensor(list(self._weights.values()))
        )
    
    def _get_normalized_weights(self) -> Dict[str, float]:
        """获取归一化的权重"""
        weights = F.softmax(self.learnable_weights, dim=0)
        return {name: w.item() for name, w in zip(self._weights.keys(), weights)}
    
    def align(
        self, 
        features_a: torch.Tensor, 
        features_b: torch.Tensor,
        use_learned_weights: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        混合对齐
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
            use_learned_weights: 是否使用可学习权重
        """
        aligned_a_list = []
        aligned_b_list = []
        
        weights = self._get_normalized_weights() if use_learned_weights else self._weights
        
        for name, sub_alignment in self._sub_alignments.items():
            sub_a, sub_b = sub_alignment.align(features_a, features_b)
            
            # 确保维度匹配（如果需要池化）
            if sub_a.dim() == 3:
                sub_a = self._pool_features(sub_a)
            if sub_b.dim() == 3:
                sub_b = self._pool_features(sub_b)
            
            weight = weights[name]
            aligned_a_list.append(sub_a * weight)
            aligned_b_list.append(sub_b * weight)
        
        # 拼接并融合
        concat_a = torch.cat(aligned_a_list, dim=-1)
        concat_b = torch.cat(aligned_b_list, dim=-1)
        
        fused_a = self.fusion(concat_a)
        fused_b = self.fusion(concat_b)
        
        return fused_a, fused_b
    
    def compute_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """计算混合损失"""
        total_loss = torch.tensor(0.0, device=features_a.device)
        weights = self._get_normalized_weights()
        
        for name, sub_alignment in self._sub_alignments.items():
            sub_loss = sub_alignment.compute_loss(features_a, features_b)
            total_loss = total_loss + weights[name] * sub_loss
        
        # 添加融合后的一致性损失
        aligned_a, aligned_b = self.align(features_a, features_b)
        consistency_loss = 1 - self._compute_similarity(aligned_a, aligned_b).mean()
        
        return total_loss + 0.1 * consistency_loss
    
    def get_sub_alignment_stats(self) -> Dict[str, Any]:
        """获取子对齐模块统计"""
        stats = {
            "weights": self._get_normalized_weights(),
            "sub_modules": list(self._sub_alignments.keys())
        }
        return stats


# ==================== 对齐工厂 ====================

class AlignmentFactory:
    """对齐工厂 - 生产级实现"""
    
    _registry: Dict[AlignmentMethod, type] = {
        AlignmentMethod.CONTRASTIVE: ContrastiveAlignment,
        AlignmentMethod.EXPLICIT: ExplicitAlignment,
        AlignmentMethod.OPTIMAL_TRANSPORT: OptimalTransportAlignment,
        AlignmentMethod.CROSS_MODAL_ATTENTION: CrossModalAttentionAlignment,
        AlignmentMethod.CCA: CCAAlignment,
        AlignmentMethod.HYBRID: HybridAlignment
    }
    
    _metrics: Dict[str, int] = {
        "total_created": 0,
        "by_method": {}
    }
    
    @classmethod
    def register(cls, method: AlignmentMethod, alignment_cls: type) -> None:
        """注册对齐模块"""
        cls._registry[method] = alignment_cls
        logger.info(f"Registered alignment method: {method.value}")
    
    @classmethod
    def create(cls, method: Union[AlignmentMethod, str], 
              config: Optional[AlignmentConfig] = None, **kwargs) -> AlignmentModule:
        """
        创建对齐模块
        
        Args:
            method: 对齐方法
            config: 对齐配置
            **kwargs: 额外参数
        """
        if isinstance(method, str):
            method = AlignmentMethod(method)
        
        alignment_cls = cls._registry.get(method)
        if alignment_cls is None:
            raise ValueError(f"Unknown alignment method: {method}")
        
        if config is None:
            config = AlignmentConfig(method=method, **kwargs)
        
        # 更新指标
        cls._metrics["total_created"] += 1
        method_name = method.value
        cls._metrics["by_method"][method_name] = cls._metrics["by_method"].get(method_name, 0) + 1
        
        logger.debug(f"Creating alignment module: {method.value}")
        return alignment_cls(config)
    
    @classmethod
    def get_available_methods(cls) -> List[str]:
        """获取可用的对齐方法"""
        return [m.value for m in cls._registry.keys()]
    
    @classmethod
    def get_factory_metrics(cls) -> Dict[str, Any]:
        """获取工厂指标"""
        return cls._metrics.copy()


def create_alignment(
    method: Union[AlignmentMethod, str],
    hidden_size: int = 768,
    **kwargs
) -> AlignmentModule:
    """
    便捷函数：创建对齐模块
    
    Args:
        method: 对齐方法
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
    """
    config = AlignmentConfig(
        method=AlignmentMethod(method) if isinstance(method, str) else method,
        hidden_size=hidden_size,
        **kwargs
    )
    return AlignmentFactory.create(method, config)


# ==================== 配置构建器 ====================

class AlignmentConfigBuilder:
    """对齐配置构建器"""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
    
    def with_method(self, method: Union[AlignmentMethod, str]) -> 'AlignmentConfigBuilder':
        """设置对齐方法"""
        if isinstance(method, str):
            method = AlignmentMethod(method)
        self._config['method'] = method
        return self
    
    def with_hidden_size(self, size: int) -> 'AlignmentConfigBuilder':
        """设置隐藏层大小"""
        self._config['hidden_size'] = size
        return self
    
    def with_projection_dim(self, dim: int) -> 'AlignmentConfigBuilder':
        """设置投影维度"""
        self._config['projection_dim'] = dim
        return self
    
    def with_temperature(self, temp: float) -> 'AlignmentConfigBuilder':
        """设置温度"""
        self._config['temperature'] = temp
        return self
    
    def with_pooling(self, method: Union[PoolingMethod, str]) -> 'AlignmentConfigBuilder':
        """设置池化方法"""
        if isinstance(method, str):
            method = PoolingMethod(method)
        self._config['pooling_method'] = method
        return self
    
    def with_hard_negatives(self, enabled: bool = True, 
                           weight: float = 0.5) -> 'AlignmentConfigBuilder':
        """设置困难负样本"""
        self._config['use_hard_negatives'] = enabled
        self._config['hard_negative_weight'] = weight
        return self
    
    def with_augmentation(self, aug_type: Union[AugmentationType, str],
                         prob: float = 0.1) -> 'AlignmentConfigBuilder':
        """设置数据增强"""
        if isinstance(aug_type, str):
            aug_type = AugmentationType(aug_type)
        self._config['augmentation_type'] = aug_type
        self._config['augmentation_prob'] = prob
        return self
    
    def with_metrics(self, enabled: bool = True,
                    history_size: int = 100) -> 'AlignmentConfigBuilder':
        """设置指标收集"""
        self._config['enable_metrics'] = enabled
        self._config['metrics_history_size'] = history_size
        return self
    
    def with_hybrid(self, methods: List[str],
                   weights: List[float] = None) -> 'AlignmentConfigBuilder':
        """设置混合对齐"""
        self._config['method'] = AlignmentMethod.HYBRID
        self._config['hybrid_methods'] = methods
        if weights:
            self._config['hybrid_weights'] = weights
        return self
    
    def build(self) -> AlignmentConfig:
        """构建配置"""
        return AlignmentConfig(**self._config)


def build_alignment_config() -> AlignmentConfigBuilder:
    """
    便捷函数：获取配置构建器
    
    使用示例:
        config = (build_alignment_config()
            .with_method("contrastive")
            .with_hidden_size(768)
            .with_temperature(0.07)
            .with_hard_negatives(True, 0.5)
            .with_augmentation("dropout", 0.1)
            .build())
    """
    return AlignmentConfigBuilder()


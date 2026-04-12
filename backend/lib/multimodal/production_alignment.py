# -*- coding: utf-8 -*-
"""
生产级跨模态对齐模块

实现多种跨模态对齐算法：
- 对比学习（CLIP风格）
- 显式对齐（Attention/MLP）
- 交叉注意力对齐
- 最优传输对齐
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .production_multimodal_config import (
    AlignmentMethod,
    CrossModalAlignmentConfig,
    ContrastiveLearningConfig,
    ExplicitAlignConfig
)

logger = logging.getLogger(__name__)


# ==================== 对比学习对齐 ====================

class ContrastiveLearningAlignment(nn.Module):
    """对比学习对齐模块
    
    实现CLIP风格的对比学习，支持：
    - InfoNCE损失
    - 硬负样本挖掘
    - 批内负样本
    """
    
    def __init__(self, config: ContrastiveLearningConfig, embed_dim: int):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        
        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.ones([]) * math.log(1 / config.temperature))
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, 
                features_a: Tensor, 
                features_b: Tensor,
                labels: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, float]]:
        """计算对比学习损失
        
        Args:
            features_a: 模态A特征 [batch, dim]
            features_b: 模态B特征 [batch, dim]
            labels: 可选的匹配标签
            
        Returns:
            损失值和指标字典
        """
        # L2归一化
        features_a = F.normalize(self.projection(features_a), dim=-1)
        features_b = F.normalize(self.projection(features_b), dim=-1)
        
        # 计算温度
        temperature = self.temperature.exp().clamp(min=0.01, max=100)
        
        # 计算相似度矩阵
        logits = torch.matmul(features_a, features_b.T) / temperature
        
        # 标签（默认对角线为正样本）
        batch_size = features_a.shape[0]
        if labels is None:
            labels = torch.arange(batch_size, device=features_a.device)
        
        # InfoNCE损失
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        loss = (loss_a + loss_b) / 2
        
        # 计算指标
        with torch.no_grad():
            # 准确率
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
    
    def compute_similarity(self, features_a: Tensor, features_b: Tensor) -> Tensor:
        """计算相似度分数"""
        features_a = F.normalize(self.projection(features_a), dim=-1)
        features_b = F.normalize(self.projection(features_b), dim=-1)
        return torch.matmul(features_a, features_b.T)


class HardNegativeMining(nn.Module):
    """硬负样本挖掘"""
    
    def __init__(self, config: ContrastiveLearningConfig):
        super().__init__()
        self.config = config
        self.hard_negative_ratio = config.hard_negative_ratio
    
    def forward(self, 
                similarity_matrix: Tensor,
                labels: Tensor) -> Tuple[Tensor, Tensor]:
        """挖掘硬负样本
        
        Args:
            similarity_matrix: 相似度矩阵 [batch, batch]
            labels: 匹配标签
            
        Returns:
            硬负样本索引和权重
        """
        batch_size = similarity_matrix.shape[0]
        
        # 创建mask排除正样本
        mask = ~(labels.unsqueeze(1) == labels.unsqueeze(0))
        
        # 获取负样本相似度
        neg_similarity = similarity_matrix.masked_fill(~mask, float('-inf'))
        
        # 选择最难的负样本
        k = int(batch_size * self.hard_negative_ratio)
        hard_neg_indices = neg_similarity.topk(k, dim=1).indices
        
        # 计算权重（相似度越高权重越大）
        hard_neg_similarity = torch.gather(similarity_matrix, 1, hard_neg_indices)
        weights = F.softmax(hard_neg_similarity, dim=1)
        
        return hard_neg_indices, weights


# ==================== 显式对齐 ====================

class ExplicitAlignment(nn.Module):
    """显式对齐模块
    
    使用MLP或Attention直接对齐不同模态的表示
    """
    
    def __init__(self, config: ExplicitAlignConfig, input_dim: int):
        super().__init__()
        self.config = config
        
        if config.method == "mlp":
            self.alignment_net = self._build_mlp(input_dim, config.hidden_size, config.num_layers)
        elif config.method == "attention":
            self.alignment_net = self._build_attention(input_dim, config.hidden_size)
        else:
            self.alignment_net = nn.Linear(input_dim, config.hidden_size)
    
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
                features_b: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """计算对齐损失
        
        Args:
            features_a: 模态A特征 [batch, dim]
            features_b: 模态B特征 [batch, dim]
            
        Returns:
            损失值和指标
        """
        if self.config.method == "mlp":
            # 拼接后预测对齐特征
            concat = torch.cat([features_a, features_b], dim=-1)
            aligned = self.alignment_net(concat)
            
            # MSE损失
            loss = F.mse_loss(aligned, features_a) + F.mse_loss(aligned, features_b)
            loss = loss / 2
        
        elif self.config.method == "attention":
            # 注意力对齐
            aligned_a, aligned_b, attn_weights = self.alignment_net(features_a, features_b)
            
            # 对齐损失
            loss = F.mse_loss(aligned_a, aligned_b)
        
        else:
            # 简单线性对齐
            proj_a = self.alignment_net(features_a)
            proj_b = self.alignment_net(features_b)
            loss = F.mse_loss(proj_a, proj_b)
        
        metrics = {
            'explicit_align_loss': loss.item()
        }
        
        return loss, metrics


class CrossModalAttention(nn.Module):
    """交叉模态注意力"""
    
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, hidden_dim)
        self.k_proj = nn.Linear(embed_dim, hidden_dim)
        self.v_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        
        # Layer norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, features_a: Tensor, features_b: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """交叉注意力
        
        Args:
            features_a: [batch, dim] 或 [batch, seq_len, dim]
            features_b: [batch, dim] 或 [batch, seq_len, dim]
            
        Returns:
            对齐后的特征和注意力权重
        """
        # 确保有序列维度
        if features_a.dim() == 2:
            features_a = features_a.unsqueeze(1)
        if features_b.dim() == 2:
            features_b = features_b.unsqueeze(1)
        
        batch_size = features_a.shape[0]
        
        # A attends to B
        q_a = self.q_proj(self.norm1(features_a))
        k_b = self.k_proj(self.norm1(features_b))
        v_b = self.v_proj(self.norm1(features_b))
        
        # Reshape for multi-head attention
        q_a = q_a.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_b = k_b.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_b = v_b.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_weights = torch.matmul(q_a, k_b.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v_b)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        aligned_a = self.out_proj(attn_output) + features_a
        aligned_a = self.norm2(aligned_a)
        
        # B attends to A (symmetric)
        q_b = self.q_proj(self.norm1(features_b))
        k_a = self.k_proj(self.norm1(features_a))
        v_a = self.v_proj(self.norm1(features_a))
        
        q_b = q_b.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k_a = k_a.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v_a = v_a.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights_b = torch.matmul(q_b, k_a.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights_b = F.softmax(attn_weights_b, dim=-1)
        
        attn_output_b = torch.matmul(attn_weights_b, v_a)
        attn_output_b = attn_output_b.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        aligned_b = self.out_proj(attn_output_b) + features_b
        aligned_b = self.norm2(aligned_b)
        
        # Squeeze if no sequence dimension
        aligned_a = aligned_a.squeeze(1) if aligned_a.shape[1] == 1 else aligned_a
        aligned_b = aligned_b.squeeze(1) if aligned_b.shape[1] == 1 else aligned_b
        
        return aligned_a, aligned_b, attn_weights


# ==================== 最优传输对齐 ====================

class OptimalTransportAlignment(nn.Module):
    """最优传输对齐
    
    使用Sinkhorn算法实现最优传输对齐
    """
    
    def __init__(self, embed_dim: int, sinkhorn_iters: int = 10, epsilon: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = epsilon
        
        # 投影层
        self.projection = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, 
                features_a: Tensor, 
                features_b: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """计算最优传输损失
        
        Args:
            features_a: [batch, dim] 或 [batch, seq_a, dim]
            features_b: [batch, dim] 或 [batch, seq_b, dim]
            
        Returns:
            损失值和指标
        """
        # 投影
        features_a = self.projection(features_a)
        features_b = self.projection(features_b)
        
        # 计算成本矩阵
        if features_a.dim() == 2:
            # 单向量情况
            cost = torch.cdist(features_a.unsqueeze(1), features_b.unsqueeze(1))
            cost = cost.squeeze()
        else:
            # 序列情况
            cost = torch.cdist(features_a, features_b)
        
        # Sinkhorn算法
        transport_plan = self._sinkhorn(cost)
        
        # 最优传输损失
        loss = (transport_plan * cost).sum(dim=(-2, -1)).mean()
        
        metrics = {
            'ot_loss': loss.item()
        }
        
        return loss, metrics
    
    def _sinkhorn(self, cost: Tensor) -> Tensor:
        """Sinkhorn算法求解最优传输"""
        # 初始化
        log_p = -cost / self.epsilon
        
        for _ in range(self.sinkhorn_iters):
            # Row normalization
            log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)
            # Column normalization
            log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)
        
        return log_p.exp()


# ==================== 统一对齐模块 ====================

class CrossModalAligner(nn.Module):
    """跨模态对齐器
    
    整合多种对齐方法，支持灵活配置
    """
    
    def __init__(self, config: CrossModalAlignmentConfig, embed_dim: int):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        
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
        self.cross_attention = CrossModalAttention(
            config.projection_dim, 
            config.projection_dim, 
            num_heads=8
        )
    
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
                nn.Dropout(0.1)
            ])
            current_dim = out_dim
        
        return nn.Sequential(*layers)
    
    def forward(self,
                features: Dict[str, Tensor],
                compute_loss: bool = True) -> Tuple[Dict[str, Tensor], Optional[Tensor], Dict[str, float]]:
        """前向传播
        
        Args:
            features: 模态名称到特征的映射
            compute_loss: 是否计算对齐损失
            
        Returns:
            对齐后的特征、损失值、指标
        """
        modalities = list(features.keys())
        if len(modalities) < 2:
            return features, None, {}
        
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
        
        # 计算损失
        total_loss = None
        all_metrics = {}
        
        if compute_loss:
            for name, aligner in self.aligners.items():
                loss, metrics = aligner(proj_a, proj_b)
                
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss = total_loss + loss
                
                all_metrics.update(metrics)
        
        # 交叉注意力增强
        aligned_a, aligned_b, _ = self.cross_attention(proj_a, proj_b)
        aligned_features[modalities[0]] = aligned_a
        aligned_features[modalities[1]] = aligned_b
        
        return aligned_features, total_loss, all_metrics
    
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
        
        # 余弦相似度
        similarity = torch.cosine_similarity(proj_a, proj_b, dim=-1).mean()
        
        return similarity.item()


# ==================== 损失函数 ====================

class AlignmentLoss(nn.Module):
    """对齐损失函数"""
    
    def __init__(self, config: CrossModalAlignmentConfig):
        super().__init__()
        self.config = config
        
        # 温度参数
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
    
    def info_nce_loss(self, features_a: Tensor, features_b: Tensor) -> Tensor:
        """InfoNCE损失"""
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)
        
        logits = torch.matmul(features_a, features_b.T) / self.temperature
        
        batch_size = features_a.shape[0]
        labels = torch.arange(batch_size, device=features_a.device)
        
        loss_a = F.cross_entropy(logits, labels)
        loss_b = F.cross_entropy(logits.T, labels)
        
        return (loss_a + loss_b) / 2
    
    def kl_divergence_loss(self, 
                          features_a: Tensor, 
                          features_b: Tensor,
                          temperature: float = 1.0) -> Tensor:
        """KL散度损失"""
        p_a = F.softmax(features_a / temperature, dim=-1)
        p_b = F.softmax(features_b / temperature, dim=-1)
        
        kl_ab = F.kl_div(p_a.log(), p_b, reduction='batchmean')
        kl_ba = F.kl_div(p_b.log(), p_a, reduction='batchmean')
        
        return (kl_ab + kl_ba) / 2
    
    def forward(self,
                features_a: Tensor,
                features_b: Tensor,
                labels: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, float]]:
        """计算对齐损失
        
        Args:
            features_a: 模态A特征
            features_b: 模态B特征
            labels: 可选的匹配标签
            
        Returns:
            总损失和指标
        """
        # InfoNCE损失
        contrastive_loss = self.info_nce_loss(features_a, features_b)
        
        # KL散度损失
        kl_loss = self.kl_divergence_loss(features_a, features_b)
        
        # 总损失
        total_loss = self.config.align_loss_weight * contrastive_loss + \
                    self.config.kl_loss_weight * kl_loss
        
        metrics = {
            'contrastive_loss': contrastive_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_align_loss': total_loss.item()
        }
        
        return total_loss, metrics


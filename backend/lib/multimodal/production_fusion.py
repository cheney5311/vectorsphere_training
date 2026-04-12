# -*- coding: utf-8 -*-
"""
生产级多模态融合模块

实现多种融合策略：
- 早期融合（Early Fusion）: 特征层面拼接
- 中期融合（Middle Fusion）: 语义层面交互
- 后期融合（Late Fusion）: 决策层面聚合
- 高级融合架构: Q-Former, Perceiver, Flamingo
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .production_multimodal_config import (
    FusionStage, FusionMethod,
    MultiModalFusionConfig,
    QFormerConfig, PerceiverConfig
)

logger = logging.getLogger(__name__)


# ==================== 早期融合 ====================

class EarlyFusion(nn.Module):
    """早期融合
    
    在特征层面进行融合，通常使用拼接或加权求和
    """
    
    def __init__(self, modality_dims: Dict[str, int], output_dim: int, method: str = "concat"):
        super().__init__()
        self.method = method
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        
        if method == "concat":
            total_dim = sum(modality_dims.values())
            self.projection = nn.Sequential(
                nn.Linear(total_dim, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim)
            )
        elif method == "weighted_sum":
            # 可学习的权重
            self.weights = nn.ParameterDict({
                name: nn.Parameter(torch.ones(1))
                for name in modality_dims.keys()
            })
            # 投影到统一维度
            self.projections = nn.ModuleDict({
                name: nn.Linear(dim, output_dim)
                for name, dim in modality_dims.items()
            })
        else:
            raise ValueError(f"Unknown early fusion method: {method}")
    
    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        """早期融合
        
        Args:
            features: 各模态特征 {name: [batch, dim]}
            
        Returns:
            融合后的特征 [batch, output_dim]
        """
        if self.method == "concat":
            # 拼接
            concat_features = torch.cat(list(features.values()), dim=-1)
            return self.projection(concat_features)
        
        elif self.method == "weighted_sum":
            # 加权求和
            fused = None
            weight_sum = 0
            
            for name, feat in features.items():
                if name in self.projections:
                    proj_feat = self.projections[name](feat)
                    weight = F.softmax(self.weights[name], dim=0)
                    
                    if fused is None:
                        fused = weight * proj_feat
                    else:
                        fused = fused + weight * proj_feat
                    
                    weight_sum = weight_sum + weight
            
            return fused / weight_sum.clamp(min=1e-6)


# ==================== 中期融合 ====================

class MiddleFusion(nn.Module):
    """中期融合
    
    在语义层面进行融合，使用交叉注意力或Transformer
    """
    
    def __init__(self, embed_dim: int, num_layers: int = 6, num_heads: int = None,
                 method: str = "cross_attention"):
        super().__init__()
        self.embed_dim = embed_dim
        self.method = method
        
        # 自动计算 num_heads，确保 embed_dim 能被整除
        if num_heads is None:
            # 尝试常见的 head 数量
            for n in [12, 8, 4, 2, 1]:
                if embed_dim % n == 0:
                    num_heads = n
                    break
            else:
                num_heads = 1
        
        if method == "cross_attention":
            self.layers = nn.ModuleList([
                CrossAttentionFusionLayer(embed_dim, num_heads)
                for _ in range(num_layers)
            ])
        elif method == "transformer":
            # 确保 num_heads 能整除 embed_dim
            transformer_heads = num_heads
            if embed_dim % transformer_heads != 0:
                for n in [8, 4, 2, 1]:
                    if embed_dim % n == 0:
                        transformer_heads = n
                        break
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=transformer_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.modality_embedding = nn.Embedding(10, embed_dim)  # 最多10个模态
        else:
            raise ValueError(f"Unknown middle fusion method: {method}")
        
        self.output_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        """中期融合
        
        Args:
            features: 各模态特征 {name: [batch, dim] 或 [batch, seq, dim]}
            
        Returns:
            融合后的特征
        """
        modalities = list(features.keys())
        
        if self.method == "cross_attention":
            # 确保有序列维度
            processed = {}
            for name, feat in features.items():
                if feat.dim() == 2:
                    feat = feat.unsqueeze(1)
                processed[name] = feat
            
            # 初始化融合特征为第一个模态
            fused = processed[modalities[0]]
            
            # 逐层融合
            for layer in self.layers:
                for i in range(1, len(modalities)):
                    fused = layer(fused, processed[modalities[i]])
            
            # 池化
            fused = fused.mean(dim=1)
            
        elif self.method == "transformer":
            # 添加模态embedding并拼接
            all_features = []
            for i, (name, feat) in enumerate(features.items()):
                if feat.dim() == 2:
                    feat = feat.unsqueeze(1)
                
                # 添加模态embedding
                batch_size = feat.shape[0]
                modality_ids = torch.full((batch_size, feat.shape[1]), i, 
                                         device=feat.device, dtype=torch.long)
                modality_emb = self.modality_embedding(modality_ids)
                feat = feat + modality_emb
                
                all_features.append(feat)
            
            # 拼接所有模态
            concat_features = torch.cat(all_features, dim=1)
            
            # Transformer融合
            fused = self.transformer(concat_features)
            
            # 池化
            fused = fused.mean(dim=1)
        
        return self.output_norm(fused)


class CrossAttentionFusionLayer(nn.Module):
    """交叉注意力融合层"""
    
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        # 确保 embed_dim 能被 num_heads 整除
        if embed_dim % num_heads != 0:
            for n in [8, 4, 2, 1]:
                if embed_dim % n == 0:
                    num_heads = n
                    break
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        """交叉注意力
        
        Args:
            query: 查询特征 [batch, seq_q, dim]
            key_value: 键值特征 [batch, seq_kv, dim]
            
        Returns:
            融合后的特征 [batch, seq_q, dim]
        """
        # 交叉注意力
        attn_output, _ = self.cross_attn(
            self.norm1(query),
            self.norm1(key_value),
            self.norm1(key_value)
        )
        query = query + self.dropout(attn_output)
        
        # FFN
        ffn_output = self.ffn(self.norm2(query))
        query = query + self.dropout(ffn_output)
        
        return query


# ==================== 后期融合 ====================

class LateFusion(nn.Module):
    """后期融合
    
    在决策层面进行融合，聚合各模态的预测结果
    """
    
    def __init__(self, modality_dims: Dict[str, int], output_dim: int, 
                 method: str = "attention"):
        super().__init__()
        self.method = method
        
        if method == "attention":
            # 注意力融合
            self.attention = ModalityAttentionFusion(modality_dims, output_dim)
        elif method == "gated":
            # 门控融合
            self.gated = GatedFusion(modality_dims, output_dim)
        elif method == "average":
            # 简单平均
            self.projections = nn.ModuleDict({
                name: nn.Linear(dim, output_dim)
                for name, dim in modality_dims.items()
            })
        else:
            raise ValueError(f"Unknown late fusion method: {method}")
    
    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        """后期融合
        
        Args:
            features: 各模态特征 {name: [batch, dim]}
            
        Returns:
            融合后的特征 [batch, output_dim]
        """
        if self.method == "attention":
            return self.attention(features)
        elif self.method == "gated":
            return self.gated(features)
        elif self.method == "average":
            projected = [self.projections[name](feat) for name, feat in features.items() 
                        if name in self.projections]
            return torch.stack(projected, dim=0).mean(dim=0)


class ModalityAttentionFusion(nn.Module):
    """模态注意力融合"""
    
    def __init__(self, modality_dims: Dict[str, int], output_dim: int):
        super().__init__()
        
        # 模态投影
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })
        
        # 注意力得分网络
        self.attention_net = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.Tanh(),
            nn.Linear(output_dim // 2, 1)
        )
        
        self.output_dim = output_dim
    
    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        """注意力融合
        
        Args:
            features: 各模态特征
            
        Returns:
            融合特征
        """
        projected = []
        for name, feat in features.items():
            if name in self.projections:
                projected.append(self.projections[name](feat))
        
        # Stack: [batch, num_modalities, dim]
        stacked = torch.stack(projected, dim=1)
        
        # 计算注意力权重
        attn_scores = self.attention_net(stacked).squeeze(-1)  # [batch, num_modalities]
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 加权求和
        fused = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        return fused


class GatedFusion(nn.Module):
    """门控融合"""
    
    def __init__(self, modality_dims: Dict[str, int], output_dim: int):
        super().__init__()
        
        # 模态投影
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, output_dim)
            for name, dim in modality_dims.items()
        })
        
        # 门控网络
        total_dim = output_dim * len(modality_dims)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.output_proj = nn.Linear(total_dim, output_dim)
    
    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        """门控融合
        
        Args:
            features: 各模态特征
            
        Returns:
            融合特征
        """
        projected = []
        for name, feat in features.items():
            if name in self.projections:
                projected.append(self.projections[name](feat))
        
        # 拼接
        concat = torch.cat(projected, dim=-1)
        
        # 门控
        gate = self.gate(concat)
        
        # 输出
        output = self.output_proj(concat)
        
        return gate * output


# ==================== Q-Former ====================

class QFormer(nn.Module):
    """Q-Former (Query-Former)
    
    BLIP-2风格的查询变换器，用于高效的多模态融合
    """
    
    def __init__(self, config: QFormerConfig):
        super().__init__()
        self.config = config
        
        # 可学习的查询tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_query_tokens, config.hidden_size)
        )
        
        # Q-Former layers
        self.layers = nn.ModuleList([
            QFormerLayer(config) for _ in range(config.num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, 
                encoder_hidden_states: Tensor,
                encoder_attention_mask: Optional[Tensor] = None) -> Tensor:
        """Q-Former前向传播
        
        Args:
            encoder_hidden_states: 编码器输出 [batch, seq, hidden]
            encoder_attention_mask: 注意力掩码
            
        Returns:
            查询输出 [batch, num_queries, hidden]
        """
        batch_size = encoder_hidden_states.shape[0]
        
        # 扩展查询tokens
        query_embeds = self.query_tokens.expand(batch_size, -1, -1)
        
        # 通过Q-Former layers
        for i, layer in enumerate(self.layers):
            # 根据交叉注意力频率决定是否使用encoder输出
            use_cross_attention = (i % self.config.cross_attention_freq == 0)
            
            if use_cross_attention:
                query_embeds = layer(
                    query_embeds,
                    encoder_hidden_states,
                    encoder_attention_mask
                )
            else:
                query_embeds = layer(query_embeds)
        
        return self.output_norm(query_embeds)


class QFormerLayer(nn.Module):
    """Q-Former层"""
    
    def __init__(self, config: QFormerConfig):
        super().__init__()
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            config.hidden_size, config.num_heads,
            dropout=0.1, batch_first=True
        )
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            config.hidden_size, config.num_heads,
            dropout=0.1, batch_first=True
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.norm3 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self,
                hidden_states: Tensor,
                encoder_hidden_states: Optional[Tensor] = None,
                encoder_attention_mask: Optional[Tensor] = None) -> Tensor:
        """Q-Former层前向传播"""
        # 自注意力
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, hidden_states, hidden_states
        )
        hidden_states = residual + self.dropout(hidden_states)
        
        # 交叉注意力（如果有encoder输出）
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.norm2(hidden_states)
            hidden_states, _ = self.cross_attn(
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states,
                key_padding_mask=encoder_attention_mask
            )
            hidden_states = residual + self.dropout(hidden_states)
        
        # FFN
        residual = hidden_states
        hidden_states = self.ffn(self.norm3(hidden_states))
        hidden_states = residual + self.dropout(hidden_states)
        
        return hidden_states


# ==================== Perceiver ====================

class PerceiverFusion(nn.Module):
    """Perceiver融合
    
    使用潜在空间进行高效的多模态融合
    """
    
    def __init__(self, config: PerceiverConfig, input_dim: int):
        super().__init__()
        self.config = config
        
        # 潜在表示
        self.latents = nn.Parameter(
            torch.randn(1, config.num_latents, config.latent_dim)
        )
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, config.latent_dim)
        
        # 交叉注意力层（输入到潜在）
        self.cross_attention_layers = nn.ModuleList([
            PerceiverCrossAttention(config.latent_dim, config.num_heads)
            for _ in range(config.num_cross_attention_layers)
        ])
        
        # 自注意力层（潜在空间内）
        self.self_attention_layers = nn.ModuleList([
            PerceiverSelfAttention(config.latent_dim, config.num_heads)
            for _ in range(config.num_self_attention_layers)
        ])
        
        self.output_norm = nn.LayerNorm(config.latent_dim)
    
    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        """Perceiver融合
        
        Args:
            features: 各模态特征
            
        Returns:
            融合后的潜在表示
        """
        # 拼接所有模态特征
        all_features = []
        for name, feat in features.items():
            if feat.dim() == 2:
                feat = feat.unsqueeze(1)
            all_features.append(feat)
        
        inputs = torch.cat(all_features, dim=1)  # [batch, total_seq, dim]
        inputs = self.input_projection(inputs)
        
        batch_size = inputs.shape[0]
        latents = self.latents.expand(batch_size, -1, -1)
        
        # 交叉注意力
        for cross_attn in self.cross_attention_layers:
            latents = cross_attn(latents, inputs)
        
        # 自注意力
        for self_attn in self.self_attention_layers:
            latents = self_attn(latents)
        
        # 池化
        output = latents.mean(dim=1)
        
        return self.output_norm(output)


class PerceiverCrossAttention(nn.Module):
    """Perceiver交叉注意力"""
    
    def __init__(self, latent_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 4, latent_dim)
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, latents: Tensor, inputs: Tensor) -> Tensor:
        """交叉注意力"""
        residual = latents
        latents = self.norm1(latents)
        attn_output, _ = self.attn(latents, inputs, inputs)
        latents = residual + self.dropout(attn_output)
        
        residual = latents
        latents = self.ffn(self.norm2(latents))
        latents = residual + self.dropout(latents)
        
        return latents


class PerceiverSelfAttention(nn.Module):
    """Perceiver自注意力"""
    
    def __init__(self, latent_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim * 4, latent_dim)
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, latents: Tensor) -> Tensor:
        """自注意力"""
        residual = latents
        latents = self.norm1(latents)
        attn_output, _ = self.attn(latents, latents, latents)
        latents = residual + self.dropout(attn_output)
        
        residual = latents
        latents = self.ffn(self.norm2(latents))
        latents = residual + self.dropout(latents)
        
        return latents


# ==================== 统一融合模块 ====================

class MultiModalFuser(nn.Module):
    """多模态融合器
    
    整合各种融合策略的统一接口
    """
    
    def __init__(self, config: MultiModalFusionConfig, modality_dims: Dict[str, int]):
        super().__init__()
        self.config = config
        self.modality_dims = modality_dims
        
        # 根据配置选择融合策略
        if config.stage == FusionStage.EARLY:
            self.fuser = EarlyFusion(
                modality_dims, config.output_dim, 
                method=config.early.method.value if hasattr(config.early.method, 'value') else config.early.method
            )
        
        elif config.stage == FusionStage.MIDDLE:
            self.fuser = MiddleFusion(
                config.output_dim, 
                config.middle.num_layers, 
                config.middle.num_heads,
                method=config.middle.method.value if hasattr(config.middle.method, 'value') else config.middle.method
            )
            # 需要先投影到统一维度
            self.pre_projection = nn.ModuleDict({
                name: nn.Linear(dim, config.output_dim)
                for name, dim in modality_dims.items()
            })
        
        elif config.stage == FusionStage.LATE:
            self.fuser = LateFusion(
                modality_dims, config.output_dim,
                method=config.late.method.value if hasattr(config.late.method, 'value') else config.late.method
            )
        
        elif config.stage == FusionStage.ADAPTIVE:
            # 自适应融合：根据输入动态选择策略
            self.early_fuser = EarlyFusion(modality_dims, config.output_dim)
            self.late_fuser = LateFusion(modality_dims, config.output_dim)
            
            # 策略选择网络
            total_dim = sum(modality_dims.values())
            self.strategy_selector = nn.Sequential(
                nn.Linear(total_dim, config.output_dim),
                nn.ReLU(),
                nn.Linear(config.output_dim, 2),
                nn.Softmax(dim=-1)
            )
            self.fuser = None
        
        # Q-Former（高级融合）
        if config.method == FusionMethod.QFORMER:
            self.qformer = QFormer(config.qformer)
        else:
            self.qformer = None
        
        # Perceiver（高级融合）
        if config.method == FusionMethod.PERCEIVER:
            input_dim = list(modality_dims.values())[0] if modality_dims else config.output_dim
            self.perceiver = PerceiverFusion(config.perceiver, input_dim)
        else:
            self.perceiver = None
        
        # 模态embedding（可选）
        if config.use_modality_embedding:
            self.modality_embedding = nn.Embedding(len(modality_dims), config.output_dim)
        else:
            self.modality_embedding = None
        
        self.output_dim = config.output_dim
    
    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        """多模态融合
        
        Args:
            features: 各模态特征 {name: tensor}
            
        Returns:
            融合后的特征
        """
        # 添加模态embedding
        if self.modality_embedding is not None:
            for i, (name, feat) in enumerate(features.items()):
                modality_idx = torch.tensor([i], device=feat.device)
                modality_emb = self.modality_embedding(modality_idx)
                features[name] = feat + modality_emb
        
        # 使用高级融合架构
        if self.qformer is not None:
            # 需要先拼接特征
            all_features = []
            for feat in features.values():
                if feat.dim() == 2:
                    feat = feat.unsqueeze(1)
                all_features.append(feat)
            concat = torch.cat(all_features, dim=1)
            output = self.qformer(concat)
            return output.mean(dim=1)
        
        if self.perceiver is not None:
            return self.perceiver(features)
        
        # 标准融合
        if self.config.stage == FusionStage.ADAPTIVE:
            return self._adaptive_fusion(features)
        
        # 中期融合需要先投影
        if self.config.stage == FusionStage.MIDDLE:
            features = {
                name: self.pre_projection[name](feat)
                for name, feat in features.items()
                if name in self.pre_projection
            }
        
        return self.fuser(features)
    
    def _adaptive_fusion(self, features: Dict[str, Tensor]) -> Tensor:
        """自适应融合"""
        # 拼接特征用于策略选择
        concat = torch.cat(list(features.values()), dim=-1)
        weights = self.strategy_selector(concat)  # [batch, 2]
        
        # 早期和后期融合
        early_output = self.early_fuser(features)
        late_output = self.late_fuser(features)
        
        # 加权组合
        output = weights[:, 0:1] * early_output + weights[:, 1:2] * late_output
        
        return output


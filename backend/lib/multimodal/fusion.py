# -*- coding: utf-8 -*-
"""
生产级多模态融合模块

实现多种融合策略：
- 早期融合（Early Fusion）: 特征层面拼接
- 中期融合（Middle Fusion）: 语义层面交互
- 后期融合（Late Fusion）: 决策层面聚合
- 高级融合架构: Q-Former, Perceiver, Flamingo
- 融合监控和性能分析
- 融合质量评估和诊断
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .multimodal_config import (
    FusionStage, FusionMethod,
    MultiModalFusionConfig,
    QFormerConfig, PerceiverConfig
)

logger = logging.getLogger(__name__)


# ==================== 枚举和数据类 ====================

class FusionStatus(Enum):
    """融合状态"""
    INITIALIZED = "initialized"
    READY = "ready"
    FUSING = "fusing"
    ERROR = "error"


@dataclass
class FusionMetrics:
    """融合性能指标"""
    total_fusion_calls: int = 0
    total_samples_processed: int = 0
    total_fusion_time: float = 0.0
    
    avg_fusion_time: float = 0.0
    avg_throughput: float = 0.0
    
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    
    # 融合质量指标
    avg_modality_contribution: Dict[str, float] = field(default_factory=dict)
    modality_usage_count: Dict[str, int] = field(default_factory=dict)
    
    error_count: int = 0
    last_error: Optional[str] = None
    
    def update_fusion_time(self, time_ms: float, batch_size: int) -> None:
        """更新融合时间"""
        self.total_fusion_calls += 1
        self.total_samples_processed += batch_size
        self.total_fusion_time += time_ms / 1000.0
        
        self.avg_fusion_time = (self.total_fusion_time / self.total_fusion_calls) * 1000
        if self.total_fusion_time > 0:
            self.avg_throughput = self.total_samples_processed / self.total_fusion_time
    
    def update_memory(self, memory_mb: float) -> None:
        """更新内存使用"""
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        n = self.total_fusion_calls
        self.avg_memory_mb = (self.avg_memory_mb * (n - 1) + memory_mb) / n if n > 0 else memory_mb
    
    def update_modality_usage(self, modality_names: List[str]) -> None:
        """更新模态使用统计"""
        for name in modality_names:
            self.modality_usage_count[name] = self.modality_usage_count.get(name, 0) + 1
    
    def record_error(self, error: str) -> None:
        """记录错误"""
        self.error_count += 1
        self.last_error = error


class FusionMonitor:
    """融合监控器"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics = FusionMetrics()
        
        # 历史记录
        self._fusion_time_history: deque = deque(maxlen=history_size)
        self._memory_history: deque = deque(maxlen=history_size)
        self._throughput_history: deque = deque(maxlen=history_size)
        
        # 状态
        self.status = FusionStatus.INITIALIZED
    
    def record_fusion(self, time_ms: float, batch_size: int, memory_mb: float, 
                     modality_names: List[str]) -> None:
        """记录融合操作"""
        self.metrics.update_fusion_time(time_ms, batch_size)
        self.metrics.update_memory(memory_mb)
        self.metrics.update_modality_usage(modality_names)
        
        self._fusion_time_history.append(time_ms)
        self._memory_history.append(memory_mb)
        
        throughput = batch_size / (time_ms / 1000.0) if time_ms > 0 else 0
        self._throughput_history.append(throughput)
    
    def record_error(self, error: str) -> None:
        """记录错误"""
        self.metrics.record_error(error)
        self.status = FusionStatus.ERROR
    
    def get_recent_stats(self, n: int = 100) -> Dict[str, float]:
        """获取最近n次的统计"""
        recent_times = list(self._fusion_time_history)[-n:]
        recent_memory = list(self._memory_history)[-n:]
        recent_throughput = list(self._throughput_history)[-n:]
        
        return {
            'avg_fusion_time': sum(recent_times) / len(recent_times) if recent_times else 0,
            'avg_memory': sum(recent_memory) / len(recent_memory) if recent_memory else 0,
            'avg_throughput': sum(recent_throughput) / len(recent_throughput) if recent_throughput else 0,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        return {
            'status': self.status.value,
            'total_calls': self.metrics.total_fusion_calls,
            'total_samples': self.metrics.total_samples_processed,
            'avg_fusion_time_ms': self.metrics.avg_fusion_time,
            'avg_throughput': self.metrics.avg_throughput,
            'peak_memory_mb': self.metrics.peak_memory_mb,
            'avg_memory_mb': self.metrics.avg_memory_mb,
            'modality_usage': dict(self.metrics.modality_usage_count),
            'error_count': self.metrics.error_count,
        }
    
    def reset(self) -> None:
        """重置监控"""
        self.metrics = FusionMetrics()
        self._fusion_time_history.clear()
        self._memory_history.clear()
        self._throughput_history.clear()
        self.status = FusionStatus.INITIALIZED


class FusionProfiler:
    """融合性能分析器"""
    
    def __init__(self):
        self._enabled = False
        self._profiles: Dict[str, List[float]] = defaultdict(list)
    
    def enable(self) -> None:
        """启用分析"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用分析"""
        self._enabled = False
    
    def record(self, operation: str, time_ms: float) -> None:
        """记录操作时间"""
        if self._enabled:
            self._profiles[operation].append(time_ms)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """获取统计信息"""
        stats = {}
        for op, times in self._profiles.items():
            if times:
                stats[op] = {
                    'count': len(times),
                    'total_ms': sum(times),
                    'avg_ms': sum(times) / len(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                }
        return stats
    
    def reset(self) -> None:
        """重置分析数据"""
        self._profiles.clear()


# ==================== 早期融合 ====================

class EarlyFusion(nn.Module):
    """早期融合
    
    在特征层面进行融合，通常使用拼接或加权求和
    支持监控和性能分析
    """
    
    def __init__(self, modality_dims: Dict[str, int], output_dim: int, method: str = "concat"):
        super().__init__()
        self.method = method
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        
        # 监控和分析
        self._monitor = FusionMonitor()
        self._profiler = FusionProfiler()
        
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
    
    def forward_with_monitoring(self, features: Dict[str, Tensor]) -> Tensor:
        """带监控的融合"""
        start_time = time.time()
        batch_size = list(features.values())[0].size(0) if features else 0
        
        try:
            self._monitor.status = FusionStatus.FUSING
            
            # 记录初始内存
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                start_memory = 0
            
            # 融合
            output = self.forward(features)
            
            # 记录结束内存
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_used = end_memory - start_memory
            else:
                memory_used = 0
            
            # 记录时间
            elapsed_ms = (time.time() - start_time) * 1000
            
            # 更新监控
            self._monitor.record_fusion(elapsed_ms, batch_size, memory_used, list(features.keys()))
            self._monitor.status = FusionStatus.READY
            
            return output
            
        except Exception as e:
            self._monitor.record_error(str(e))
            raise
    
    def get_weights(self) -> Optional[Dict[str, float]]:
        """获取融合权重（仅weighted_sum方法）"""
        if self.method == "weighted_sum":
            weights = {}
            for name, weight in self.weights.items():
                weights[name] = F.softmax(weight, dim=0).item()
            return weights
        return None
    
    def get_monitor(self) -> FusionMonitor:
        """获取监控器"""
        return self._monitor
    
    def get_profiler(self) -> FusionProfiler:
        """获取分析器"""
        return self._profiler
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._monitor.reset()
        self._profiler.reset()


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
    
    def get_monitor(self) -> FusionMonitor:
        """获取监控器"""
        if hasattr(self.fuser, 'get_monitor'):
            return self.fuser.get_monitor()
        return FusionMonitor()  # 返回空监控器
    
    def get_profiler(self) -> FusionProfiler:
        """获取分析器"""
        if hasattr(self.fuser, 'get_profiler'):
            return self.fuser.get_profiler()
        return FusionProfiler()
    
    def count_parameters(self) -> Tuple[int, int]:
        """统计参数数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """获取参数信息"""
        total, trainable = self.count_parameters()
        return {
            'total_parameters': total,
            'trainable_parameters': trainable,
            'memory_mb': total * 4 / (1024 * 1024),
        }
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断融合器状态"""
        param_info = self.get_parameter_info()
        monitor = self.get_monitor()
        summary = monitor.get_summary()
        
        diagnosis = {
            'fusion_stage': self.config.stage.value,
            'fusion_method': self.config.method.value,
            'status': summary['status'],
            'parameters': param_info,
            'performance': summary,
        }
        
        # 性能建议
        recommendations = []
        if summary['avg_fusion_time_ms'] > 50:
            recommendations.append("High fusion time, consider simpler fusion method")
        if summary['peak_memory_mb'] > 500:
            recommendations.append("High memory usage, consider early fusion")
        if summary['error_count'] > 0:
            recommendations.append(f"Errors detected: {summary['error_count']}")
        
        diagnosis['recommendations'] = recommendations
        return diagnosis
    
    def print_summary(self) -> None:
        """打印摘要"""
        diagnosis = self.diagnose()
        
        print(f"\n{'='*60}")
        print(f"Fusion Module Summary")
        print(f"{'='*60}")
        print(f"Stage: {diagnosis['fusion_stage']}")
        print(f"Method: {diagnosis['fusion_method']}")
        print(f"Status: {diagnosis['status']}")
        print(f"\nParameters:")
        print(f"  Total: {diagnosis['parameters']['total_parameters']:,}")
        print(f"  Trainable: {diagnosis['parameters']['trainable_parameters']:,}")
        print(f"  Memory: {diagnosis['parameters']['memory_mb']:.2f} MB")
        print(f"\nPerformance:")
        print(f"  Total calls: {diagnosis['performance']['total_calls']}")
        print(f"  Avg time: {diagnosis['performance']['avg_fusion_time_ms']:.2f} ms")
        print(f"  Throughput: {diagnosis['performance']['avg_throughput']:.1f} samples/s")
        print(f"  Peak memory: {diagnosis['performance']['peak_memory_mb']:.2f} MB")
        print(f"  Modality usage: {diagnosis['performance']['modality_usage']}")
        
        if diagnosis['recommendations']:
            print(f"\nRecommendations:")
            for rec in diagnosis['recommendations']:
                print(f"  - {rec}")
        
        print(f"{'='*60}")


# ==================== 工具函数 ====================

def create_fusion_module(
    config: MultiModalFusionConfig,
    modality_dims: Dict[str, int]
) -> MultiModalFuser:
    """
    创建融合模块
    
    Args:
        config: 融合配置
        modality_dims: 模态维度
        
    Returns:
        融合模块实例
    """
    return MultiModalFuser(config, modality_dims)


def estimate_fusion_memory(
    config: MultiModalFusionConfig,
    modality_dims: Dict[str, int],
    batch_size: int = 1
) -> Dict[str, float]:
    """
    估算融合内存占用
    
    Args:
        config: 融合配置
        modality_dims: 模态维度
        batch_size: 批次大小
        
    Returns:
        内存估算（MB）
    """
    # 输入特征内存
    input_memory = sum(dim * batch_size * 4 / (1024 * 1024) for dim in modality_dims.values())
    
    # 输出特征内存
    output_memory = config.output_dim * batch_size * 4 / (1024 * 1024)
    
    # 参数内存（粗略估计）
    if config.stage == FusionStage.EARLY:
        param_memory = sum(modality_dims.values()) * config.output_dim * 4 / (1024 * 1024)
    elif config.stage == FusionStage.MIDDLE:
        param_memory = config.middle.estimate_param_count() * 4 / (1024 * 1024)
    elif config.stage == FusionStage.LATE:
        param_memory = sum(modality_dims.values()) * config.output_dim * 4 / (1024 * 1024)
    else:
        param_memory = 0
    
    total_memory = input_memory + output_memory + param_memory
    
    return {
        'input_mb': input_memory,
        'output_mb': output_memory,
        'parameters_mb': param_memory,
        'total_mb': total_memory,
    }


def recommend_fusion_strategy(
    num_modalities: int,
    modality_dims: Dict[str, int],
    quality_priority: bool = True,
    memory_constraint_mb: Optional[float] = None
) -> Tuple[FusionStage, FusionMethod]:
    """
    推荐融合策略
    
    Args:
        num_modalities: 模态数量
        modality_dims: 模态维度
        quality_priority: 是否优先质量
        memory_constraint_mb: 内存限制（MB）
        
    Returns:
        (推荐的融合阶段, 推荐的融合方法)
    """
    if memory_constraint_mb is not None and memory_constraint_mb < 1000:
        # 内存受限，使用早期融合
        return FusionStage.EARLY, FusionMethod.CONCAT
    
    if quality_priority:
        if num_modalities <= 2:
            return FusionStage.MIDDLE, FusionMethod.CROSS_ATTENTION
        else:
            return FusionStage.MIDDLE, FusionMethod.QFORMER
    else:
        if num_modalities <= 2:
            return FusionStage.EARLY, FusionMethod.CONCAT
        else:
            return FusionStage.LATE, FusionMethod.ATTENTION


def compare_fusion_strategies(
    modality_dims: Dict[str, int],
    batch_size: int = 1
) -> Dict[str, Dict[str, Any]]:
    """
    比较不同融合策略
    
    Args:
        modality_dims: 模态维度
        batch_size: 批次大小
        
    Returns:
        各策略的比较信息
    """
    from .multimodal_config import (
        EarlyFusionConfig, MiddleFusionConfig, LateFusionConfig
    )
    
    strategies = {}
    
    # 早期融合
    early_config = MultiModalFusionConfig(
        stage=FusionStage.EARLY,
        method=FusionMethod.CONCAT,
        output_dim=768
    )
    strategies['early'] = {
        'memory': estimate_fusion_memory(early_config, modality_dims, batch_size),
        'complexity': 'low',
        'quality': 'medium',
    }
    
    # 中期融合
    middle_config = MultiModalFusionConfig(
        stage=FusionStage.MIDDLE,
        method=FusionMethod.CROSS_ATTENTION,
        output_dim=768
    )
    strategies['middle'] = {
        'memory': estimate_fusion_memory(middle_config, modality_dims, batch_size),
        'complexity': 'high',
        'quality': 'high',
    }
    
    # 后期融合
    late_config = MultiModalFusionConfig(
        stage=FusionStage.LATE,
        method=FusionMethod.ATTENTION,
        output_dim=768
    )
    strategies['late'] = {
        'memory': estimate_fusion_memory(late_config, modality_dims, batch_size),
        'complexity': 'medium',
        'quality': 'medium-high',
    }
    
    return strategies


def print_fusion_comparison(
    modality_dims: Dict[str, int],
    batch_size: int = 1
) -> None:
    """
    打印融合策略比较
    
    Args:
        modality_dims: 模态维度
        batch_size: 批次大小
    """
    strategies = compare_fusion_strategies(modality_dims, batch_size)
    
    print(f"\n{'='*70}")
    print("Fusion Strategy Comparison")
    print(f"{'='*70}")
    print(f"\n{'Strategy':<15} {'Memory (MB)':<15} {'Complexity':<15} {'Quality':<15}")
    print("-" * 70)
    
    for name, info in strategies.items():
        memory = info['memory']['total_mb']
        complexity = info['complexity']
        quality = info['quality']
        print(f"{name:<15} {memory:<15.2f} {complexity:<15} {quality:<15}")
    
    print(f"{'='*70}")


def analyze_modality_contribution(
    fuser: MultiModalFuser,
    features: Dict[str, Tensor]
) -> Dict[str, float]:
    """
    分析各模态的贡献度
    
    Args:
        fuser: 融合器
        features: 模态特征
        
    Returns:
        各模态的贡献度
    """
    contributions = {}
    
    # 如果是weighted_sum方法，直接返回权重
    if hasattr(fuser.fuser, 'get_weights'):
        weights = fuser.fuser.get_weights()
        if weights:
            return weights
    
    # 否则，通过消融实验估计贡献度
    with torch.no_grad():
        full_output = fuser(features)
        
        for modality in features.keys():
            # 移除该模态
            ablated_features = {k: v for k, v in features.items() if k != modality}
            if ablated_features:
                ablated_output = fuser(ablated_features)
                # 计算差异
                diff = torch.norm(full_output - ablated_output, dim=-1).mean().item()
                contributions[modality] = diff
            else:
                contributions[modality] = 1.0
    
    # 归一化
    total = sum(contributions.values())
    if total > 0:
        contributions = {k: v / total for k, v in contributions.items()}
    
    return contributions


def print_modality_contribution(
    fuser: MultiModalFuser,
    features: Dict[str, Tensor]
) -> None:
    """
    打印模态贡献度
    
    Args:
        fuser: 融合器
        features: 模态特征
    """
    contributions = analyze_modality_contribution(fuser, features)
    
    print(f"\n{'='*60}")
    print("Modality Contribution Analysis")
    print(f"{'='*60}")
    
    sorted_modalities = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    
    for modality, contribution in sorted_modalities:
        bar_length = int(contribution * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"{modality:<15} {bar} {contribution:.2%}")
    
    print(f"{'='*60}")


def diagnose_fusion_module(fuser: MultiModalFuser) -> Dict[str, Any]:
    """
    诊断融合模块
    
    Args:
        fuser: 融合器
        
    Returns:
        诊断结果
    """
    return fuser.diagnose()


def print_fusion_diagnosis(fuser: MultiModalFuser) -> None:
    """
    打印融合模块诊断
    
    Args:
        fuser: 融合器
    """
    fuser.print_summary()


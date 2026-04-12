# -*- coding: utf-8 -*-
"""
融合模块 - 生产级实现

提供多模态特征融合的各种方法。

生产级特性：
- 多种融合方法（早期、中期、后期、交叉注意力、门控、Q-Former、Perceiver、Tensor融合）
- 指标收集和监控
- 数值稳定性保障
- 融合质量评估
- 动态模态数量支持
- 注意力权重可视化
- 残差连接和层归一化
"""

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class FusionMethod(Enum):
    """融合方法"""
    EARLY = "early"                      # 早期融合
    MIDDLE = "middle"                    # 中期融合
    LATE = "late"                        # 后期融合
    CROSS_ATTENTION = "cross_attention"  # 交叉注意力
    GATED = "gated"                      # 门控融合
    QFORMER = "qformer"                  # Q-Former
    PERCEIVER = "perceiver"              # Perceiver
    TENSOR = "tensor"                    # 张量融合
    BILINEAR = "bilinear"                # 双线性融合


class FusionStatus(Enum):
    """融合状态"""
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"


class PoolingType(Enum):
    """池化类型"""
    MEAN = "mean"
    MAX = "max"
    ATTENTION = "attention"
    CLS = "cls"


class NormType(Enum):
    """归一化类型"""
    LAYER_NORM = "layer_norm"
    BATCH_NORM = "batch_norm"
    RMS_NORM = "rms_norm"
    NONE = "none"


# ==================== 数据类 ====================

@dataclass
class FusionMetrics:
    """融合指标"""
    total_fusions: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    error_count: int = 0
    last_fusion_time: Optional[datetime] = None
    modality_counts: Dict[int, int] = field(default_factory=dict)
    
    def record_fusion(self, time_taken: float, num_modalities: int = 2) -> None:
        """记录融合"""
        self.total_fusions += 1
        self.total_time += time_taken
        self.avg_time = self.total_time / self.total_fusions
        self.last_fusion_time = datetime.now()
        self.modality_counts[num_modalities] = self.modality_counts.get(num_modalities, 0) + 1
    
    def record_error(self) -> None:
        """记录错误"""
        self.error_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_fusions": self.total_fusions,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "error_count": self.error_count,
            "last_fusion_time": self.last_fusion_time.isoformat() if self.last_fusion_time else None,
            "modality_counts": self.modality_counts
        }


@dataclass
class FusionConfig:
    """融合配置"""
    method: FusionMethod = FusionMethod.CROSS_ATTENTION
    hidden_size: int = 768
    num_heads: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    
    # Q-Former配置
    num_queries: int = 32
    
    # Perceiver配置
    num_latents: int = 64
    latent_dim: int = 512
    
    # 池化配置
    pooling_type: PoolingType = PoolingType.MEAN
    
    # 归一化配置
    norm_type: NormType = NormType.LAYER_NORM
    
    # 指标配置
    enable_metrics: bool = True
    
    # 数值稳定性
    eps: float = 1e-8
    
    # 残差连接
    use_residual: bool = True
    residual_scale: float = 0.1
    
    # 门控配置
    use_gating: bool = False
    
    # 张量融合配置
    tensor_rank: int = 16
    
    # 双线性融合配置
    bilinear_output_dim: int = 256
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FusionModule(nn.Module, ABC):
    """
    融合模块基类 - 生产级实现
    
    所有融合模块的抽象基类，提供：
    - 指标收集和监控
    - 多种池化方法
    - 数值稳定性保障
    - 融合质量评估
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # 状态
        self._status = FusionStatus.READY
        
        # 指标
        self._metrics = FusionMetrics() if config.enable_metrics else None
        self._metrics_lock = threading.Lock()
        
        # 注意力池化（如果需要）
        if config.pooling_type == PoolingType.ATTENTION:
            self._attention_pooling = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.Tanh(),
                nn.Linear(config.hidden_size // 4, 1)
            )
        else:
            self._attention_pooling = None
        
        # 归一化层
        self._norm = self._create_norm_layer(config.norm_type, config.hidden_size)
    
    def _create_norm_layer(self, norm_type: NormType, dim: int) -> Optional[nn.Module]:
        """创建归一化层"""
        if norm_type == NormType.LAYER_NORM:
            return nn.LayerNorm(dim)
        elif norm_type == NormType.BATCH_NORM:
            return nn.BatchNorm1d(dim)
        elif norm_type == NormType.RMS_NORM:
            return RMSNorm(dim)
        else:
            return None
    
    @property
    def status(self) -> FusionStatus:
        """获取状态"""
        return self._status
    
    @property
    def metrics(self) -> Optional[FusionMetrics]:
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
        pooling_type = self.config.pooling_type
        
        if pooling_type == PoolingType.MEAN:
            return features.mean(dim=1)
        elif pooling_type == PoolingType.MAX:
            return features.max(dim=1)[0]
        elif pooling_type == PoolingType.CLS:
            return features[:, 0, :]
        elif pooling_type == PoolingType.ATTENTION:
            weights = self._attention_pooling(features)  # [B, seq_len, 1]
            weights = F.softmax(weights, dim=1)
            return (features * weights).sum(dim=1)
        else:
            return features.mean(dim=1)
    
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
    
    def _apply_norm(self, tensor: torch.Tensor) -> torch.Tensor:
        """应用归一化"""
        if self._norm is None:
            return tensor
        
        if isinstance(self._norm, nn.BatchNorm1d):
            # BatchNorm需要 [B, C, L] 格式
            if tensor.dim() == 3:
                return self._norm(tensor.transpose(1, 2)).transpose(1, 2)
            else:
                return self._norm(tensor)
        else:
            return self._norm(tensor)
    
    @contextmanager
    def _timed_fusion(self, num_modalities: int = 2):
        """计时融合上下文管理器"""
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
                    self._metrics.record_fusion(elapsed, num_modalities)
    
    def _validate_inputs(self, features: List[torch.Tensor]) -> None:
        """验证输入"""
        if not features:
            raise ValueError("Features list cannot be empty")
        
        batch_size = features[0].shape[0]
        for i, feat in enumerate(features):
            if feat.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch at modality {i}: expected {batch_size}, got {feat.shape[0]}")
    
    @abstractmethod
    def fuse(self, features: List[torch.Tensor], **kwargs) -> torch.Tensor:
        """
        融合多模态特征
        
        Args:
            features: 各模态特征列表 [B, seq_len, hidden]
            **kwargs: 额外参数
            
        Returns:
            融合后的特征
        """
        pass
    
    def evaluate_fusion_quality(self, features: List[torch.Tensor],
                               fused: torch.Tensor) -> Dict[str, float]:
        """
        评估融合质量
        
        Args:
            features: 原始特征列表
            fused: 融合后的特征
            
        Returns:
            质量指标字典
        """
        with torch.no_grad():
            results = {}
            
            # 融合后特征与各模态的相似度
            fused_pooled = self._pool_features(fused) if fused.dim() == 3 else fused
            
            similarities = []
            for i, feat in enumerate(features):
                feat_pooled = self._pool_features(feat) if feat.dim() == 3 else feat
                sim = torch.cosine_similarity(fused_pooled, feat_pooled, dim=-1).mean()
                similarities.append(sim.item())
                results[f"similarity_modality_{i}"] = sim.item()
            
            results["avg_similarity"] = sum(similarities) / len(similarities)
            
            # 融合特征的信息熵（归一化后）
            fused_norm = F.softmax(fused_pooled, dim=-1)
            entropy = -(fused_norm * torch.log(fused_norm + self.config.eps)).sum(dim=-1).mean()
            results["entropy"] = entropy.item()
            
            return results
    
    def forward(self, features: List[torch.Tensor], **kwargs) -> torch.Tensor:
        """前向传播"""
        # 验证输入
        self._validate_inputs(features)
        
        # 执行融合
        with self._timed_fusion(len(features)):
            fused = self.fuse(features, **kwargs)
        
        # 检查数值稳定性
        fused = self._check_numerical_stability(fused, "fused_output")
        
        return fused
    
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
                self._metrics = FusionMetrics()
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        获取参数统计
        
        Returns:
            参数统计字典
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取内存使用（MB）
        
        Returns:
            内存使用字典
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        
        return {
            "parameters_mb": param_size / (1024 * 1024),
            "buffers_mb": buffer_size / (1024 * 1024),
            "total_mb": (param_size + buffer_size) / (1024 * 1024)
        }
    
    def fuse_with_quality_check(self, features: List[torch.Tensor],
                               quality_threshold: float = 0.5,
                               **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        融合并检查质量
        
        Args:
            features: 输入特征列表
            quality_threshold: 质量阈值
            **kwargs: 额外参数
            
        Returns:
            融合特征和质量报告的元组
        """
        # 执行融合
        fused = self.forward(features, **kwargs)
        
        # 评估质量
        quality = self.evaluate_fusion_quality(features, fused)
        quality["passed_threshold"] = quality.get("avg_similarity", 0) >= quality_threshold
        
        return fused, quality
    
    def batch_fuse(self, feature_batches: List[List[torch.Tensor]],
                  **kwargs) -> List[torch.Tensor]:
        """
        批量融合
        
        Args:
            feature_batches: 特征批次列表，每个元素是一个模态特征列表
            **kwargs: 额外参数
            
        Returns:
            融合结果列表
        """
        results = []
        for features in feature_batches:
            fused = self.forward(features, **kwargs)
            results.append(fused)
        return results
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "method": self.config.method.value if hasattr(self.config, 'method') else "unknown",
            "hidden_size": self.config.hidden_size,
            "num_heads": self.config.num_heads,
            "num_layers": self.config.num_layers,
            "dropout": self.config.dropout,
            "pooling_type": self.config.pooling_type.value,
            "norm_type": self.config.norm_type.value,
            "use_residual": self.config.use_residual,
            "use_gating": self.config.use_gating
        }


class RMSNorm(nn.Module):
    """RMS归一化"""
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.scale * x / rms


class EarlyFusion(FusionModule):
    """
    早期融合 - 生产级实现
    
    在底层直接拼接各模态特征，支持：
    - 动态模态数量
    - 残差连接
    - 多层投影
    - 模态贡献分析
    """
    
    def __init__(self, config: FusionConfig, num_modalities: int = 2, **kwargs):
        super().__init__(config)
        
        self.num_modalities = num_modalities
        
        # 多层投影
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size * num_modalities, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
        
        # 残差投影（用于跳跃连接）
        if config.use_residual:
            self.residual_proj = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.residual_proj = None
        
        # 门控（可选）
        if config.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(config.hidden_size * num_modalities, config.hidden_size),
                nn.Sigmoid()
            )
        else:
            self.gate = None
        
        # 存储模态贡献（用于分析）
        self._modality_contributions: Optional[torch.Tensor] = None
    
    def _compute_modality_contributions(self, features: List[torch.Tensor],
                                       fused: torch.Tensor) -> torch.Tensor:
        """计算各模态对融合结果的贡献"""
        with torch.no_grad():
            contributions = []
            fused_pooled = self._pool_features(fused) if fused.dim() == 3 else fused
            
            for feat in features:
                feat_pooled = self._pool_features(feat) if feat.dim() == 3 else feat
                # 使用余弦相似度作为贡献度指标
                sim = torch.cosine_similarity(fused_pooled, feat_pooled, dim=-1)
                contributions.append(sim.mean())
            
            return torch.stack(contributions)
    
    def fuse(self, features: List[torch.Tensor], 
             compute_contributions: bool = False, **kwargs) -> torch.Tensor:
        """早期融合：拼接所有模态"""
        # 验证模态数量
        if len(features) != self.num_modalities:
            logger.warning(f"Expected {self.num_modalities} modalities, got {len(features)}")
        
        # 拼接
        concat = torch.cat(features, dim=-1)  # [B, seq, hidden*num_modalities]
        
        # 投影
        fused = self.projection(concat)
        
        # 门控（可选）
        if self.gate is not None:
            gate_values = self.gate(concat)
            fused = fused * gate_values
        
        # 残差连接（使用第一个模态）
        if self.residual_proj is not None and self.config.use_residual:
            residual = self.residual_proj(features[0])
            fused = fused + self.config.residual_scale * residual
        
        # 应用归一化
        fused = self._apply_norm(fused)
        
        # 计算模态贡献
        if compute_contributions:
            self._modality_contributions = self._compute_modality_contributions(features, fused)
        
        return fused
    
    def get_modality_contributions(self) -> Optional[Dict[int, float]]:
        """获取各模态贡献度"""
        if self._modality_contributions is not None:
            return {i: c.item() for i, c in enumerate(self._modality_contributions)}
        return None


class MiddleFusion(FusionModule):
    """
    中期融合 - 生产级实现
    
    在中间层通过Transformer交互，支持：
    - 位置编码
    - 模态类型嵌入
    - 多层交互
    """
    
    def __init__(self, config: FusionConfig, **kwargs):
        super().__init__(config)
        
        # 模态类型嵌入
        self.modality_embed = nn.Embedding(10, config.hidden_size)
        
        # 位置编码
        self.pos_embed = nn.Embedding(1024, config.hidden_size)
        
        # 确保num_heads能整除hidden_size
        num_heads = config.num_heads
        while config.hidden_size % num_heads != 0:
            num_heads -= 1
        
        # Transformer融合层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # 输出投影
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 门控融合（用于最终输出）
        if config.use_gating:
            self.output_gate = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Sigmoid()
            )
        else:
            self.output_gate = None
        
        # 存储注意力权重
        self._attention_weights: Optional[torch.Tensor] = None
    
    def _add_position_encoding(self, features: torch.Tensor) -> torch.Tensor:
        """添加位置编码"""
        batch_size, seq_len, _ = features.shape
        positions = torch.arange(seq_len, device=features.device).unsqueeze(0).expand(batch_size, -1)
        return features + self.pos_embed(positions)
    
    def _compute_modality_mixing(self, features: List[torch.Tensor],
                                fused: torch.Tensor) -> Dict[str, float]:
        """计算模态混合程度"""
        with torch.no_grad():
            # 计算融合后各位置与原始模态的相似度
            mixing_scores = {}
            
            start_idx = 0
            for i, feat in enumerate(features):
                seq_len = feat.shape[1]
                end_idx = start_idx + seq_len
                
                # 获取对应位置的融合特征
                fused_segment = fused[:, start_idx:end_idx, :]
                
                # 计算与原始特征的相似度
                sim = torch.cosine_similarity(
                    fused_segment.mean(dim=1), 
                    feat.mean(dim=1), 
                    dim=-1
                ).mean()
                
                mixing_scores[f"modality_{i}_retention"] = sim.item()
                start_idx = end_idx
            
            return mixing_scores
    
    def fuse(self, features: List[torch.Tensor], return_attention: bool = False, 
             analyze_mixing: bool = False, **kwargs) -> torch.Tensor:
        """中期融合：Transformer交互"""
        batch_size = features[0].shape[0]
        device = features[0].device
        
        # 添加模态类型嵌入和位置编码
        enhanced_features = []
        for i, feat in enumerate(features):
            # 模态嵌入
            modality_emb = self.modality_embed(
                torch.full((batch_size, feat.shape[1]), i, dtype=torch.long, device=device)
            )
            enhanced = feat + modality_emb
            # 位置编码
            enhanced = self._add_position_encoding(enhanced)
            enhanced_features.append(enhanced)
        
        # 拼接所有模态
        concat = torch.cat(enhanced_features, dim=1)  # [B, total_seq, hidden]
        
        # 记录输入用于残差
        residual_input = concat
        
        # Transformer融合
        fused = self.transformer(concat)
        
        # 输出投影
        fused = self.output_proj(fused)
        
        # 门控融合（可选）
        if self.output_gate is not None:
            gate_input = torch.cat([fused, residual_input], dim=-1)
            gate_values = self.output_gate(gate_input)
            fused = fused * gate_values + residual_input * (1 - gate_values)
        elif self.config.use_residual:
            fused = fused + self.config.residual_scale * residual_input
        
        # 应用归一化
        fused = self._apply_norm(fused)
        
        # 分析模态混合
        if analyze_mixing:
            self._mixing_analysis = self._compute_modality_mixing(features, fused)
        
        return fused
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取注意力权重"""
        return self._attention_weights
    
    def get_mixing_analysis(self) -> Optional[Dict[str, float]]:
        """获取模态混合分析"""
        return getattr(self, '_mixing_analysis', None)


class LateFusion(FusionModule):
    """
    后期融合 - 生产级实现
    
    在高层融合各模态的表示，支持：
    - 多种池化策略
    - 注意力权重可视化
    - 多头注意力
    """
    
    def __init__(self, config: FusionConfig, **kwargs):
        super().__init__(config)
        
        # 确保num_heads能整除hidden_size
        num_heads = config.num_heads
        while config.hidden_size % num_heads != 0:
            num_heads -= 1
        self.num_heads = num_heads
        
        # 多头注意力融合
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # 查询生成
        self.query_gen = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # 输出投影
        self.output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
        
        # 门控
        if config.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Sigmoid()
            )
        else:
            self.gate = None
        
        # 存储注意力权重
        self._attention_weights: Optional[torch.Tensor] = None
    
    def fuse(self, features: List[torch.Tensor], return_attention: bool = False, **kwargs) -> torch.Tensor:
        """后期融合：注意力加权"""
        # 使用配置的池化方法获取每个模态的表示
        representations = []
        for feat in features:
            rep = self._pool_features(feat).unsqueeze(1)  # [B, 1, hidden]
            representations.append(rep)
        
        # 堆叠
        stacked = torch.cat(representations, dim=1)  # [B, num_modalities, hidden]
        
        # 生成查询（使用加权平均）
        query = self.query_gen(stacked.mean(dim=1, keepdim=True))  # [B, 1, hidden]
        
        # 多头注意力
        attended, attn_weights = self.multihead_attn(
            query, stacked, stacked,
            need_weights=return_attention
        )
        
        if return_attention:
            self._attention_weights = attn_weights.detach()
        
        # 门控（可选）
        if self.gate is not None:
            gate_input = torch.cat([attended, query], dim=-1)
            gate_values = self.gate(gate_input)
            attended = attended * gate_values + query * (1 - gate_values)
        
        # 输出投影
        fused = self.output(attended)
        
        # 应用归一化
        return self._apply_norm(fused)
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取注意力权重"""
        return self._attention_weights
    
    def get_modality_importance(self, features: List[torch.Tensor]) -> Dict[int, float]:
        """
        获取各模态的重要性
        
        Args:
            features: 输入特征列表
            
        Returns:
            各模态重要性字典
        """
        with torch.no_grad():
            _ = self.fuse(features, return_attention=True)
            if self._attention_weights is not None:
                weights = self._attention_weights.mean(dim=0).squeeze()  # [num_modalities]
                return {i: w.item() for i, w in enumerate(weights)}
        return {}


class CrossAttentionFusion(FusionModule):
    """
    交叉注意力融合 - 生产级实现
    
    各模态通过交叉注意力相互交互，支持：
    - 双向交叉注意力
    - 多层交互
    - 注意力权重可视化
    """
    
    def __init__(self, config: FusionConfig, **kwargs):
        super().__init__(config)
        
        # 确保num_heads能整除hidden_size
        self.num_heads = config.num_heads
        while config.hidden_size % self.num_heads != 0:
            self.num_heads -= 1
        
        # 多层交叉注意力
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=self.num_heads,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        
        # FFN层
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size * 4, config.hidden_size),
                nn.Dropout(config.dropout)
            )
            for _ in range(config.num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_size)
            for _ in range(config.num_layers * 2)
        ])
        
        # 门控
        if config.use_gating:
            self.layer_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(config.hidden_size * 2, config.hidden_size),
                    nn.Sigmoid()
                )
                for _ in range(config.num_layers)
            ])
        else:
            self.layer_gates = None
        
        # 存储注意力权重
        self._attention_weights: List[torch.Tensor] = []
    
    def fuse(self, features: List[torch.Tensor], return_attention: bool = False, **kwargs) -> torch.Tensor:
        """交叉注意力融合"""
        if len(features) < 2:
            return self._apply_norm(features[0])
        
        # 以第一个模态为query，其他为key/value
        query = features[0]
        kv = torch.cat(features[1:], dim=1)
        
        self._attention_weights = []
        
        # 多层交叉注意力
        for i, (cross_attn, ffn) in enumerate(zip(self.cross_attn_layers, self.ffn_layers)):
            # 交叉注意力
            attended, attn_weights = cross_attn(
                query, kv, kv,
                need_weights=return_attention
            )
            
            if return_attention:
                self._attention_weights.append(attn_weights.detach())
            
            # 门控残差（可选）
            if self.layer_gates is not None:
                gate_input = torch.cat([attended, query], dim=-1)
                gate_values = self.layer_gates[i](gate_input)
                query = self.layer_norms[i * 2](gate_values * attended + (1 - gate_values) * query)
            else:
                query = self.layer_norms[i * 2](query + attended)
            
            # FFN
            ffn_out = ffn(query)
            query = self.layer_norms[i * 2 + 1](query + ffn_out)
        
        return query
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """获取各层注意力权重"""
        return self._attention_weights


class GatedFusion(FusionModule):
    """
    门控融合 - 生产级实现
    
    使用门控机制选择性融合模态，支持：
    - 动态模态数量
    - 多层门控
    - 门控权重可视化
    """
    
    def __init__(self, config: FusionConfig, num_modalities: int = 2, **kwargs):
        super().__init__(config)
        
        self.num_modalities = num_modalities
        
        # 模态投影
        self.modality_proj = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size)
            for _ in range(num_modalities)
        ])
        
        # 多层门控网络
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_size * num_modalities, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, num_modalities)
        )
        
        # 温度参数（用于软化门控）
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 输出投影
        self.output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
        
        # 存储门控权重
        self._gate_weights: Optional[torch.Tensor] = None
    
    def _compute_gate_weights(self, representations: List[torch.Tensor]) -> torch.Tensor:
        """计算门控权重"""
        concat = torch.cat(representations, dim=-1)
        logits = self.gate(concat) / (self.temperature + self.config.eps)
        return F.softmax(logits, dim=-1)
    
    def fuse(self, features: List[torch.Tensor], return_gates: bool = False, **kwargs) -> torch.Tensor:
        """门控融合"""
        # 使用配置的池化方法获取表示并投影
        representations = []
        for i, feat in enumerate(features):
            rep = self._pool_features(feat)
            if i < len(self.modality_proj):
                rep = self.modality_proj[i](rep)
            representations.append(rep)
        
        # 计算门控权重
        gates = self._compute_gate_weights(representations)  # [B, num_modalities]
        
        if return_gates:
            self._gate_weights = gates.detach()
        
        # 加权融合
        stacked = torch.stack(representations, dim=1)  # [B, num_modalities, hidden]
        gates_expanded = gates.unsqueeze(-1)  # [B, num_modalities, 1]
        fused = (stacked * gates_expanded).sum(dim=1)  # [B, hidden]
        
        # 输出投影
        output = self.output(fused)
        
        # 应用归一化
        output = self._apply_norm(output)
        
        return output.unsqueeze(1)  # [B, 1, hidden]
    
    def get_gate_weights(self) -> Optional[torch.Tensor]:
        """获取门控权重"""
        return self._gate_weights
    
    def get_modality_importance(self, features: List[torch.Tensor]) -> Dict[int, float]:
        """
        获取各模态的重要性
        
        Args:
            features: 输入特征列表
            
        Returns:
            各模态重要性字典
        """
        with torch.no_grad():
            _ = self.fuse(features, return_gates=True)
            if self._gate_weights is not None:
                weights = self._gate_weights.mean(dim=0)  # [num_modalities]
                return {i: w.item() for i, w in enumerate(weights)}
        return {}


class QFormerFusion(FusionModule):
    """
    Q-Former融合 - 生产级实现
    
    BLIP-2风格的Q-Former融合，支持：
    - 可学习查询初始化策略
    - Pre-LN架构
    - 注意力权重可视化
    - 查询选择性输出
    """
    
    def __init__(self, config: FusionConfig, **kwargs):
        super().__init__(config)
        
        # 可学习的查询向量
        self.queries = nn.Parameter(torch.zeros(1, config.num_queries, config.hidden_size))
        nn.init.trunc_normal_(self.queries, std=0.02)
        
        # 确保num_heads能整除hidden_size
        self.num_heads = config.num_heads
        while config.hidden_size % self.num_heads != 0:
            self.num_heads -= 1
        
        # 交叉注意力层
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=self.num_heads,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        
        # 自注意力层
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=self.num_heads,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        
        # FFN
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size * 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size * 4, config.hidden_size),
                nn.Dropout(config.dropout)
            )
            for _ in range(config.num_layers)
        ])
        
        # Pre-LN层归一化
        self.pre_cross_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_size)
            for _ in range(config.num_layers)
        ])
        self.pre_self_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_size)
            for _ in range(config.num_layers)
        ])
        self.pre_ffn_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_size)
            for _ in range(config.num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 存储注意力权重
        self._cross_attention_weights: List[torch.Tensor] = []
        self._self_attention_weights: List[torch.Tensor] = []
    
    def fuse(self, features: List[torch.Tensor], return_attention: bool = False,
            num_output_queries: Optional[int] = None, **kwargs) -> torch.Tensor:
        """
        Q-Former融合
        
        Args:
            features: 输入特征列表
            return_attention: 是否返回注意力权重
            num_output_queries: 输出查询数量（默认全部）
        """
        batch_size = features[0].shape[0]
        
        # 扩展查询
        queries = self.queries.expand(batch_size, -1, -1)  # [B, num_queries, hidden]
        
        # 拼接所有模态特征
        kv = torch.cat(features, dim=1)  # [B, total_seq, hidden]
        
        self._cross_attention_weights = []
        self._self_attention_weights = []
        
        # 多层交互（Pre-LN架构）
        for i in range(len(self.cross_attn_layers)):
            # 交叉注意力
            normed_queries = self.pre_cross_norms[i](queries)
            attended, cross_attn_weights = self.cross_attn_layers[i](
                normed_queries, kv, kv,
                need_weights=return_attention
            )
            queries = queries + attended
            
            if return_attention:
                self._cross_attention_weights.append(cross_attn_weights.detach())
            
            # 自注意力
            normed_queries = self.pre_self_norms[i](queries)
            self_attended, self_attn_weights = self.self_attn_layers[i](
                normed_queries, normed_queries, normed_queries,
                need_weights=return_attention
            )
            queries = queries + self_attended
            
            if return_attention:
                self._self_attention_weights.append(self_attn_weights.detach())
            
            # FFN
            normed_queries = self.pre_ffn_norms[i](queries)
            ffn_out = self.ffn_layers[i](normed_queries)
            queries = queries + ffn_out
        
        # 输出投影
        queries = self.output_proj(queries)
        
        # 选择性输出
        if num_output_queries is not None and num_output_queries < queries.shape[1]:
            queries = queries[:, :num_output_queries, :]
        
        return queries
    
    def get_cross_attention_weights(self) -> List[torch.Tensor]:
        """获取交叉注意力权重"""
        return self._cross_attention_weights
    
    def get_self_attention_weights(self) -> List[torch.Tensor]:
        """获取自注意力权重"""
        return self._self_attention_weights


class PerceiverFusion(FusionModule):
    """
    Perceiver融合 - 生产级实现
    
    使用潜在空间进行高效融合，支持：
    - 迭代细化
    - 位置编码
    - 注意力权重可视化
    """
    
    def __init__(self, config: FusionConfig, **kwargs):
        super().__init__(config)
        
        # 潜在向量
        self.latents = nn.Parameter(torch.zeros(1, config.num_latents, config.latent_dim))
        nn.init.trunc_normal_(self.latents, std=0.02)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_latents, config.latent_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.latent_dim),
            nn.LayerNorm(config.latent_dim)
        )
        
        # 确保num_heads能整除latent_dim
        self.num_heads = config.num_heads
        while config.latent_dim % self.num_heads != 0:
            self.num_heads -= 1
        
        # 迭代交叉注意力（多轮细化）
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.latent_dim,
                num_heads=self.num_heads,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        
        # 自注意力层
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.latent_dim,
                num_heads=self.num_heads,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        
        # FFN层
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.latent_dim, config.latent_dim * 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.latent_dim * 4, config.latent_dim),
                nn.Dropout(config.dropout)
            )
            for _ in range(config.num_layers)
        ])
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.latent_dim)
            for _ in range(config.num_layers * 3)
        ])
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
        # 存储注意力权重
        self._cross_attention_weights: List[torch.Tensor] = []
    
    def fuse(self, features: List[torch.Tensor], return_attention: bool = False, **kwargs) -> torch.Tensor:
        """Perceiver融合"""
        batch_size = features[0].shape[0]
        
        # 扩展潜在向量并添加位置编码
        latents = self.latents.expand(batch_size, -1, -1) + self.pos_embed
        
        # 拼接并投影输入
        kv = torch.cat(features, dim=1)
        kv = self.input_projection(kv)
        
        self._cross_attention_weights = []
        
        # 迭代细化
        for i in range(len(self.cross_attn_layers)):
            # 交叉注意力
            attended, cross_attn_weights = self.cross_attn_layers[i](
                latents, kv, kv,
                need_weights=return_attention
            )
            latents = self.layer_norms[i * 3](latents + attended)
            
            if return_attention:
                self._cross_attention_weights.append(cross_attn_weights.detach())
            
            # 自注意力
            self_attended, _ = self.self_attn_layers[i](latents, latents, latents)
            latents = self.layer_norms[i * 3 + 1](latents + self_attended)
            
            # FFN
            ffn_out = self.ffn_layers[i](latents)
            latents = self.layer_norms[i * 3 + 2](latents + ffn_out)
        
        # 输出投影
        output = self.output_projection(latents)
        
        return output
    
    def get_cross_attention_weights(self) -> List[torch.Tensor]:
        """获取交叉注意力权重"""
        return self._cross_attention_weights


# ==================== 新增融合模块 ====================

class TensorFusion(FusionModule):
    """
    张量融合 - 生产级实现
    
    使用张量分解进行多模态融合，支持：
    - 低秩分解
    - 动态秩选择
    - 因子分析
    - 秩自适应
    """
    
    def __init__(self, config: FusionConfig, num_modalities: int = 2, **kwargs):
        super().__init__(config)
        
        self.num_modalities = num_modalities
        self.rank = config.tensor_rank
        
        # 模态投影
        self.modality_factors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, self.rank)
            )
            for _ in range(num_modalities)
        ])
        
        # 融合因子
        self.fusion_weights = nn.Parameter(torch.zeros(self.rank, config.hidden_size))
        nn.init.xavier_uniform_(self.fusion_weights)
        
        # 秩重要性权重（可学习）
        self.rank_importance = nn.Parameter(torch.ones(self.rank))
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
        
        # 存储因子分析
        self._factor_analysis: Optional[Dict[str, Any]] = None
    
    def _compute_factor_analysis(self, factors: List[torch.Tensor],
                                fused_factor: torch.Tensor) -> Dict[str, Any]:
        """计算因子分析"""
        with torch.no_grad():
            analysis = {
                "num_modalities": len(factors),
                "rank": self.rank,
                "modality_factor_norms": [],
                "fused_factor_norm": fused_factor.norm().item(),
                "rank_utilization": []
            }
            
            for i, factor in enumerate(factors):
                analysis["modality_factor_norms"].append(factor.norm().item())
            
            # 秩利用率（各秩分量的贡献）
            rank_contributions = fused_factor.abs().mean(dim=0)
            total_contribution = rank_contributions.sum()
            for r in range(self.rank):
                utilization = (rank_contributions[r] / (total_contribution + 1e-8)).item()
                analysis["rank_utilization"].append(utilization)
            
            # 有效秩（贡献超过阈值的秩数量）
            threshold = 0.01
            analysis["effective_rank"] = sum(1 for u in analysis["rank_utilization"] if u > threshold)
            
            return analysis
    
    def fuse(self, features: List[torch.Tensor], 
             analyze_factors: bool = False, **kwargs) -> torch.Tensor:
        """张量融合"""
        # 池化并投影各模态
        factors = []
        for i, feat in enumerate(features):
            pooled = self._pool_features(feat)
            if i < len(self.modality_factors):
                factor = self.modality_factors[i](pooled)  # [B, rank]
            else:
                factor = self.modality_factors[-1](pooled)
            factors.append(factor)
        
        # 张量积（元素乘积）并应用秩重要性
        fused_factor = factors[0]
        for factor in factors[1:]:
            fused_factor = fused_factor * factor  # [B, rank]
        
        # 应用秩重要性权重
        rank_weights = F.softmax(self.rank_importance, dim=0)
        fused_factor = fused_factor * rank_weights
        
        # 因子分析
        if analyze_factors:
            self._factor_analysis = self._compute_factor_analysis(factors, fused_factor)
        
        # 投影到输出空间
        fused = torch.matmul(fused_factor, self.fusion_weights)  # [B, hidden]
        
        # 输出投影
        output = self.output_proj(fused)
        
        # 应用归一化
        output = self._apply_norm(output)
        
        return output.unsqueeze(1)  # [B, 1, hidden]
    
    def get_factor_analysis(self) -> Optional[Dict[str, Any]]:
        """获取因子分析结果"""
        return self._factor_analysis
    
    def get_rank_importance(self) -> Dict[int, float]:
        """获取各秩的重要性"""
        weights = F.softmax(self.rank_importance, dim=0)
        return {i: w.item() for i, w in enumerate(weights)}


class BilinearFusion(FusionModule):
    """
    双线性融合 - 生产级实现
    
    使用双线性池化进行两模态融合，支持：
    - 低秩双线性（可选）
    - 交互分析
    - 递归多模态融合
    """
    
    def __init__(self, config: FusionConfig, use_low_rank: bool = False, 
                 low_rank_dim: int = 64, **kwargs):
        super().__init__(config)
        
        self.use_low_rank = use_low_rank
        
        if use_low_rank:
            # 低秩双线性分解
            self.factor1 = nn.Linear(config.hidden_size, low_rank_dim)
            self.factor2 = nn.Linear(config.hidden_size, low_rank_dim)
            self.combine = nn.Linear(low_rank_dim, config.bilinear_output_dim)
        else:
            # 标准双线性层
            self.bilinear = nn.Bilinear(
                config.hidden_size, 
                config.hidden_size, 
                config.bilinear_output_dim
            )
        
        # 投影回原始维度
        self.output_proj = nn.Sequential(
            nn.Linear(config.bilinear_output_dim, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        # 存储交互分析
        self._interaction_analysis: Optional[Dict[str, Any]] = None
    
    def _compute_bilinear_output(self, feat1: torch.Tensor, 
                                feat2: torch.Tensor) -> torch.Tensor:
        """计算双线性输出"""
        if self.use_low_rank:
            # 低秩分解：(W1 * x1) ⊙ (W2 * x2) 然后投影
            f1 = self.factor1(feat1)  # [B, low_rank_dim]
            f2 = self.factor2(feat2)  # [B, low_rank_dim]
            interaction = f1 * f2  # 元素乘积
            return self.combine(interaction)
        else:
            return self.bilinear(feat1, feat2)
    
    def _compute_interaction_analysis(self, feat1: torch.Tensor, feat2: torch.Tensor,
                                     bilinear_out: torch.Tensor) -> Dict[str, Any]:
        """计算交互分析"""
        with torch.no_grad():
            analysis = {
                "input1_norm": feat1.norm().item(),
                "input2_norm": feat2.norm().item(),
                "output_norm": bilinear_out.norm().item(),
                "input_correlation": torch.cosine_similarity(
                    feat1.mean(dim=0, keepdim=True), 
                    feat2.mean(dim=0, keepdim=True),
                    dim=-1
                ).item(),
                "interaction_strength": (bilinear_out.norm() / 
                    (feat1.norm() * feat2.norm() + 1e-8)).item()
            }
            
            # 交互方差
            analysis["output_variance"] = bilinear_out.var().item()
            
            return analysis
    
    def fuse(self, features: List[torch.Tensor], 
             analyze_interaction: bool = False, **kwargs) -> torch.Tensor:
        """双线性融合"""
        if len(features) < 2:
            return self._apply_norm(features[0])
        
        # 池化前两个模态
        feat1 = self._pool_features(features[0])  # [B, hidden]
        feat2 = self._pool_features(features[1])  # [B, hidden]
        
        # 双线性融合
        bilinear_out = self._compute_bilinear_output(feat1, feat2)
        
        # 交互分析
        if analyze_interaction:
            self._interaction_analysis = self._compute_interaction_analysis(
                feat1, feat2, bilinear_out
            )
        
        # 输出投影
        output = self.output_proj(bilinear_out)
        
        # 如果有更多模态，递归融合
        if len(features) > 2:
            remaining = [output.unsqueeze(1)] + features[2:]
            return self.fuse(remaining, analyze_interaction=False)
        
        # 应用归一化
        output = self._apply_norm(output)
        
        return output.unsqueeze(1)  # [B, 1, hidden]
    
    def get_interaction_analysis(self) -> Optional[Dict[str, Any]]:
        """获取交互分析结果"""
        return self._interaction_analysis


# ==================== 混合融合 ====================

class HybridFusion(FusionModule):
    """
    混合融合 - 生产级实现
    
    组合多种融合方法，支持：
    - 动态权重学习
    - 级联融合
    - 自适应方法选择
    """
    
    def __init__(self, config: FusionConfig, methods: List[str] = None, 
                 weights: List[float] = None, **kwargs):
        super().__init__(config)
        
        methods = methods or ["early", "late"]
        weights = weights or [1.0 / len(methods)] * len(methods)
        
        self._method_names = methods
        self._sub_fusions: nn.ModuleDict = nn.ModuleDict()
        self._initial_weights = weights
        
        # 创建子融合模块
        for method_name in methods:
            method = FusionMethod(method_name)
            sub_config = FusionConfig(
                method=method,
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                num_layers=max(1, config.num_layers // 2),
                dropout=config.dropout,
                enable_metrics=False  # 子模块不单独收集指标
            )
            
            # 根据方法创建子模块
            if method == FusionMethod.EARLY:
                self._sub_fusions[method_name] = EarlyFusion(sub_config, **kwargs)
            elif method == FusionMethod.LATE:
                self._sub_fusions[method_name] = LateFusion(sub_config, **kwargs)
            elif method == FusionMethod.CROSS_ATTENTION:
                self._sub_fusions[method_name] = CrossAttentionFusion(sub_config, **kwargs)
            elif method == FusionMethod.GATED:
                self._sub_fusions[method_name] = GatedFusion(sub_config, **kwargs)
        
        # 可学习权重
        self.learnable_weights = nn.Parameter(torch.tensor(weights))
        
        # 融合层
        total_dim = config.hidden_size * len(self._sub_fusions)
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_dim, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size)
        )
    
    def _get_normalized_weights(self) -> torch.Tensor:
        """获取归一化权重"""
        return F.softmax(self.learnable_weights, dim=0)
    
    def fuse(self, features: List[torch.Tensor], 
             use_learned_weights: bool = True, **kwargs) -> torch.Tensor:
        """混合融合"""
        weights = self._get_normalized_weights() if use_learned_weights else torch.tensor(self._initial_weights)
        
        fused_outputs = []
        for i, (method_name, sub_fusion) in enumerate(self._sub_fusions.items()):
            sub_output = sub_fusion.fuse(features, **kwargs)
            
            # 池化到统一维度
            if sub_output.dim() == 3:
                sub_output = self._pool_features(sub_output)
            
            fused_outputs.append(sub_output * weights[i])
        
        # 拼接并融合
        concat = torch.cat(fused_outputs, dim=-1)
        output = self.fusion_layer(concat)
        
        # 应用归一化
        output = self._apply_norm(output)
        
        return output.unsqueeze(1)  # [B, 1, hidden]
    
    def get_method_weights(self) -> Dict[str, float]:
        """获取各方法权重"""
        weights = self._get_normalized_weights()
        return {name: w.item() for name, w in zip(self._method_names, weights)}


# ==================== 融合质量分析器 ====================

class FusionQualityAnalyzer:
    """
    融合质量分析器
    
    提供详细的融合质量分析功能
    """
    
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self._history: deque = deque(maxlen=100)
    
    def analyze(self, original_features: List[torch.Tensor],
               fused_features: torch.Tensor) -> Dict[str, Any]:
        """
        分析融合质量
        
        Args:
            original_features: 原始特征列表
            fused_features: 融合后的特征
            
        Returns:
            质量分析报告
        """
        with torch.no_grad():
            report = {
                "timestamp": datetime.now().isoformat(),
                "num_modalities": len(original_features),
                "metrics": {}
            }
            
            # 池化融合特征
            if fused_features.dim() == 3:
                fused_pooled = fused_features.mean(dim=1)
            else:
                fused_pooled = fused_features
            
            # 1. 模态相似度分析
            similarities = []
            for i, feat in enumerate(original_features):
                if feat.dim() == 3:
                    feat_pooled = feat.mean(dim=1)
                else:
                    feat_pooled = feat
                
                sim = torch.cosine_similarity(fused_pooled, feat_pooled, dim=-1).mean()
                similarities.append(sim.item())
                report["metrics"][f"similarity_modality_{i}"] = sim.item()
            
            report["metrics"]["avg_similarity"] = sum(similarities) / len(similarities)
            report["metrics"]["similarity_variance"] = sum(
                (s - report["metrics"]["avg_similarity"]) ** 2 for s in similarities
            ) / len(similarities)
            
            # 2. 信息保留度
            total_original_norm = sum(f.norm().item() for f in original_features)
            fused_norm = fused_features.norm().item()
            report["metrics"]["information_retention"] = fused_norm / (total_original_norm + self.eps)
            
            # 3. 特征多样性（标准差）
            report["metrics"]["feature_diversity"] = fused_pooled.std(dim=-1).mean().item()
            
            # 4. 特征熵
            fused_softmax = F.softmax(fused_pooled, dim=-1)
            entropy = -(fused_softmax * torch.log(fused_softmax + self.eps)).sum(dim=-1).mean()
            report["metrics"]["feature_entropy"] = entropy.item()
            
            # 5. 模态贡献均衡度（基于相似度的标准差）
            if len(similarities) > 1:
                balance = 1 - (report["metrics"]["similarity_variance"] ** 0.5)
                report["metrics"]["modality_balance"] = max(0, balance)
            else:
                report["metrics"]["modality_balance"] = 1.0
            
            # 6. 综合质量分数
            quality_score = (
                0.3 * report["metrics"]["avg_similarity"] +
                0.2 * min(1.0, report["metrics"]["information_retention"]) +
                0.2 * min(1.0, report["metrics"]["feature_diversity"]) +
                0.15 * min(1.0, report["metrics"]["feature_entropy"] / 5.0) +
                0.15 * report["metrics"]["modality_balance"]
            )
            report["metrics"]["quality_score"] = quality_score
            
            # 记录历史
            self._history.append(quality_score)
            
            return report
    
    def get_history_stats(self) -> Dict[str, float]:
        """获取历史统计"""
        if not self._history:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        history = list(self._history)
        mean_val = sum(history) / len(history)
        variance = sum((x - mean_val) ** 2 for x in history) / len(history)
        
        return {
            "mean": mean_val,
            "std": variance ** 0.5,
            "min": min(history),
            "max": max(history),
            "count": len(history)
        }
    
    def reset(self) -> None:
        """重置历史"""
        self._history.clear()


# ==================== 融合工厂 ====================

class FusionFactory:
    """融合工厂 - 生产级实现"""
    
    _registry: Dict[FusionMethod, type] = {
        FusionMethod.EARLY: EarlyFusion,
        FusionMethod.MIDDLE: MiddleFusion,
        FusionMethod.LATE: LateFusion,
        FusionMethod.CROSS_ATTENTION: CrossAttentionFusion,
        FusionMethod.GATED: GatedFusion,
        FusionMethod.QFORMER: QFormerFusion,
        FusionMethod.PERCEIVER: PerceiverFusion,
        FusionMethod.TENSOR: TensorFusion,
        FusionMethod.BILINEAR: BilinearFusion
    }
    
    _metrics: Dict[str, Any] = {
        "total_created": 0,
        "by_method": {},
        "creation_times": []
    }
    
    @classmethod
    def register(cls, method: FusionMethod, fusion_cls: type) -> None:
        """注册融合模块"""
        cls._registry[method] = fusion_cls
        logger.info(f"Registered fusion method: {method.value}")
    
    @classmethod
    def create(cls, method: Union[FusionMethod, str], 
              config: Optional[FusionConfig] = None, **kwargs) -> FusionModule:
        """
        创建融合模块
        
        Args:
            method: 融合方法
            config: 融合配置
            **kwargs: 额外参数
        """
        start_time = time.time()
        
        if isinstance(method, str):
            method = FusionMethod(method)
        
        fusion_cls = cls._registry.get(method)
        if fusion_cls is None:
            raise ValueError(f"Unknown fusion method: {method}")
        
        if config is None:
            config = FusionConfig(method=method, **kwargs)
        
        fusion = fusion_cls(config, **kwargs)
        
        # 更新指标
        elapsed = time.time() - start_time
        cls._metrics["total_created"] += 1
        method_name = method.value
        cls._metrics["by_method"][method_name] = cls._metrics["by_method"].get(method_name, 0) + 1
        cls._metrics["creation_times"].append(elapsed)
        if len(cls._metrics["creation_times"]) > 100:
            cls._metrics["creation_times"] = cls._metrics["creation_times"][-100:]
        
        logger.debug(f"Created fusion module: {method.value} in {elapsed:.4f}s")
        
        return fusion
    
    @classmethod
    def get_available_methods(cls) -> List[str]:
        """获取可用的融合方法"""
        return [m.value for m in cls._registry.keys()]
    
    @classmethod
    def get_factory_metrics(cls) -> Dict[str, Any]:
        """获取工厂指标"""
        metrics = cls._metrics.copy()
        if metrics["creation_times"]:
            metrics["avg_creation_time"] = sum(metrics["creation_times"]) / len(metrics["creation_times"])
        else:
            metrics["avg_creation_time"] = 0.0
        return metrics
    
    @classmethod
    def reset_metrics(cls) -> None:
        """重置工厂指标"""
        cls._metrics = {
            "total_created": 0,
            "by_method": {},
            "creation_times": []
        }
    
    @classmethod
    def create_hybrid(cls, methods: List[str], weights: List[float] = None,
                     config: Optional[FusionConfig] = None, **kwargs) -> HybridFusion:
        """
        创建混合融合模块
        
        Args:
            methods: 融合方法列表
            weights: 权重列表
            config: 配置
            **kwargs: 额外参数
        """
        if config is None:
            config = FusionConfig(**kwargs)
        
        return HybridFusion(config, methods=methods, weights=weights, **kwargs)


def create_fusion(
    method: Union[FusionMethod, str],
    hidden_size: int = 768,
    **kwargs
) -> FusionModule:
    """
    便捷函数：创建融合模块
    
    Args:
        method: 融合方法
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
    """
    config = FusionConfig(
        method=FusionMethod(method) if isinstance(method, str) else method,
        hidden_size=hidden_size,
        **kwargs
    )
    return FusionFactory.create(method, config)


def create_hybrid_fusion(
    methods: List[str],
    weights: List[float] = None,
    hidden_size: int = 768,
    **kwargs
) -> HybridFusion:
    """
    便捷函数：创建混合融合模块
    
    Args:
        methods: 融合方法列表
        weights: 权重列表
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
    """
    config = FusionConfig(hidden_size=hidden_size, **kwargs)
    return FusionFactory.create_hybrid(methods, weights, config, **kwargs)


# ==================== 配置构建器 ====================

class FusionConfigBuilder:
    """融合配置构建器"""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
    
    def with_method(self, method: Union[FusionMethod, str]) -> 'FusionConfigBuilder':
        """设置融合方法"""
        if isinstance(method, str):
            method = FusionMethod(method)
        self._config['method'] = method
        return self
    
    def with_hidden_size(self, size: int) -> 'FusionConfigBuilder':
        """设置隐藏层大小"""
        self._config['hidden_size'] = size
        return self
    
    def with_num_heads(self, heads: int) -> 'FusionConfigBuilder':
        """设置注意力头数"""
        self._config['num_heads'] = heads
        return self
    
    def with_num_layers(self, layers: int) -> 'FusionConfigBuilder':
        """设置层数"""
        self._config['num_layers'] = layers
        return self
    
    def with_dropout(self, dropout: float) -> 'FusionConfigBuilder':
        """设置dropout"""
        self._config['dropout'] = dropout
        return self
    
    def with_qformer(self, num_queries: int = 32) -> 'FusionConfigBuilder':
        """设置Q-Former配置"""
        self._config['num_queries'] = num_queries
        return self
    
    def with_perceiver(self, num_latents: int = 64, 
                      latent_dim: int = 512) -> 'FusionConfigBuilder':
        """设置Perceiver配置"""
        self._config['num_latents'] = num_latents
        self._config['latent_dim'] = latent_dim
        return self
    
    def with_pooling(self, pooling_type: Union[PoolingType, str]) -> 'FusionConfigBuilder':
        """设置池化类型"""
        if isinstance(pooling_type, str):
            pooling_type = PoolingType(pooling_type)
        self._config['pooling_type'] = pooling_type
        return self
    
    def with_norm(self, norm_type: Union[NormType, str]) -> 'FusionConfigBuilder':
        """设置归一化类型"""
        if isinstance(norm_type, str):
            norm_type = NormType(norm_type)
        self._config['norm_type'] = norm_type
        return self
    
    def with_residual(self, enabled: bool = True, 
                     scale: float = 0.1) -> 'FusionConfigBuilder':
        """设置残差连接"""
        self._config['use_residual'] = enabled
        self._config['residual_scale'] = scale
        return self
    
    def with_gating(self, enabled: bool = True) -> 'FusionConfigBuilder':
        """设置门控"""
        self._config['use_gating'] = enabled
        return self
    
    def with_metrics(self, enabled: bool = True) -> 'FusionConfigBuilder':
        """设置指标收集"""
        self._config['enable_metrics'] = enabled
        return self
    
    def with_tensor_rank(self, rank: int) -> 'FusionConfigBuilder':
        """设置张量融合秩"""
        self._config['tensor_rank'] = rank
        return self
    
    def with_bilinear_output(self, dim: int) -> 'FusionConfigBuilder':
        """设置双线性输出维度"""
        self._config['bilinear_output_dim'] = dim
        return self
    
    def build(self) -> FusionConfig:
        """构建配置"""
        return FusionConfig(**self._config)


def build_fusion_config() -> FusionConfigBuilder:
    """
    便捷函数：获取配置构建器
    
    使用示例:
        config = (build_fusion_config()
            .with_method("cross_attention")
            .with_hidden_size(768)
            .with_num_heads(8)
            .with_num_layers(2)
            .with_residual(True, 0.1)
            .with_gating(True)
            .build())
    """
    return FusionConfigBuilder()


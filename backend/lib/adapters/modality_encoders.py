# -*- coding: utf-8 -*-
"""
模态编码器 - 生产级实现

为各种模态数据提供统一的编码接口。

生产级特性：
- 多种模态编码器（文本、图像、音频、视频、时序、表格、图、点云）
- 指标收集和监控
- 数据增强支持
- 数值稳定性保障
- 编码质量评估
- 特征池化策略
- 批量编码支持
- 多模态编码器组合
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

class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TIME_SERIES = "time_series"
    TABULAR = "tabular"
    GRAPH = "graph"
    POINT_CLOUD = "point_cloud"


class EncoderStatus(Enum):
    """编码器状态"""
    READY = "ready"
    ENCODING = "encoding"
    ERROR = "error"
    WARMING_UP = "warming_up"


class PoolingMethod(Enum):
    """池化方法"""
    MEAN = "mean"
    MAX = "max"
    ATTENTION = "attention"
    CLS = "cls"
    FIRST_LAST_AVG = "first_last_avg"


class AugmentationType(Enum):
    """数据增强类型"""
    NONE = "none"
    NOISE = "noise"
    DROPOUT = "dropout"
    MIXUP = "mixup"
    CUTOUT = "cutout"
    TIME_WARP = "time_warp"
    SPEC_AUGMENT = "spec_augment"


class NormalizationType(Enum):
    """归一化类型"""
    LAYER_NORM = "layer_norm"
    BATCH_NORM = "batch_norm"
    RMS_NORM = "rms_norm"
    NONE = "none"


# ==================== 数据类 ====================

@dataclass
class EncoderMetrics:
    """编码器指标"""
    total_encodings: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    error_count: int = 0
    last_encoding_time: Optional[datetime] = None
    batch_sizes: Dict[int, int] = field(default_factory=dict)
    feature_stats: Dict[str, float] = field(default_factory=dict)
    quality_scores: List[float] = field(default_factory=list)
    
    def record_encoding(self, time_taken: float, batch_size: int = 1) -> None:
        """记录编码"""
        self.total_encodings += 1
        self.total_time += time_taken
        self.avg_time = self.total_time / self.total_encodings
        self.last_encoding_time = datetime.now()
        self.batch_sizes[batch_size] = self.batch_sizes.get(batch_size, 0) + 1
    
    def record_error(self) -> None:
        """记录错误"""
        self.error_count += 1
    
    def update_feature_stats(self, features: torch.Tensor) -> None:
        """更新特征统计"""
        with torch.no_grad():
            self.feature_stats["mean"] = float(features.mean())
            self.feature_stats["std"] = float(features.std())
            self.feature_stats["min"] = float(features.min())
            self.feature_stats["max"] = float(features.max())
            self.feature_stats["norm"] = float(features.norm(p=2))
    
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
            "total_encodings": self.total_encodings,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "error_count": self.error_count,
            "last_encoding_time": self.last_encoding_time.isoformat() if self.last_encoding_time else None,
            "batch_sizes": self.batch_sizes,
            "feature_stats": self.feature_stats,
            "recent_quality": self.get_recent_quality()
        }


@dataclass
class EncoderConfig:
    """编码器配置 - 生产级"""
    modality: ModalityType = ModalityType.TEXT
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # 模态特定配置
    vocab_size: int = 30522  # Text
    max_seq_length: int = 512  # Text
    image_size: int = 224  # Image
    patch_size: int = 16  # Image
    num_channels: int = 3  # Image
    sample_rate: int = 16000  # Audio
    num_features: int = 64  # Time-series/Tabular
    
    # 生产级配置 - 池化
    pooling_method: PoolingMethod = PoolingMethod.MEAN
    
    # 生产级配置 - 归一化
    norm_type: NormalizationType = NormalizationType.LAYER_NORM
    
    # 生产级配置 - 数据增强
    augmentation_type: AugmentationType = AugmentationType.NONE
    augmentation_prob: float = 0.5
    noise_std: float = 0.1
    dropout_rate: float = 0.1
    mixup_alpha: float = 0.2
    
    # 生产级配置 - 指标
    enable_metrics: bool = True
    metrics_history_size: int = 1000
    
    # 生产级配置 - 数值稳定性
    eps: float = 1e-8
    gradient_clip: float = 1.0
    
    # 生产级配置 - 投影
    output_projection: bool = False
    projection_dim: int = 256
    
    # 生产级配置 - 视频
    num_frames: int = 8
    
    # 生产级配置 - 图
    num_nodes: int = 100
    edge_dim: int = 32
    
    # 生产级配置 - 点云
    num_points: int = 1024
    point_dim: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # 转换枚举为字符串
        result["modality"] = self.modality.value
        result["pooling_method"] = self.pooling_method.value
        result["norm_type"] = self.norm_type.value
        result["augmentation_type"] = self.augmentation_type.value
        return result


class ModalityEncoder(nn.Module, ABC):
    """
    模态编码器基类 - 生产级实现
    
    所有模态编码器的抽象基类，提供：
    - 指标收集和监控
    - 多种池化方法
    - 数据增强支持
    - 数值稳定性保障
    - 编码质量评估
    """
    
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.modality = config.modality
        self.hidden_size = config.hidden_size
        
        # 投影层（将编码器输出投影到统一维度）
        self.projector: Optional[nn.Module] = None
        if config.output_projection:
            self.projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.projection_dim),
                nn.LayerNorm(config.projection_dim)
            )
        
        # 状态
        self._status = EncoderStatus.READY
        
        # 指标
        self._metrics = EncoderMetrics() if config.enable_metrics else None
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
        self._augmentation_dropout = nn.Dropout(config.dropout_rate) if config.augmentation_type == AugmentationType.DROPOUT else None
    
    @abstractmethod
    def encode(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        编码输入
        
        Args:
            inputs: 模态特定的输入
            **kwargs: 额外参数
            
        Returns:
            编码后的特征 [batch_size, seq_len, hidden_size]
        """
        pass
    
    def forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """前向传播"""
        with self._timed_encoding():
            # 应用数据增强
            if self.training:
                inputs = self._apply_augmentation(inputs)
            
            # 编码
            features = self.encode(inputs, **kwargs)
            
            # 数值稳定性检查
            features = self._check_numerical_stability(features)
            
            # 投影
            if self.projector is not None:
                features = self.projector(features)
            
            # 更新特征统计
            if self._metrics is not None:
                self._metrics.update_feature_stats(features)
            
            return features
    
    @contextmanager
    def _timed_encoding(self):
        """计时上下文管理器"""
        self._status = EncoderStatus.ENCODING
        start_time = time.time()
        try:
            yield
        except Exception as e:
            self._status = EncoderStatus.ERROR
            if self._metrics is not None:
                with self._metrics_lock:
                    self._metrics.record_error()
            raise
        finally:
            elapsed = time.time() - start_time
            if self._metrics is not None:
                with self._metrics_lock:
                    self._metrics.record_encoding(elapsed)
            self._status = EncoderStatus.READY
    
    def _pool_features(self, features: torch.Tensor, 
                       attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        池化特征
        
        Args:
            features: 特征张量 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            池化后的特征 [batch_size, hidden_size]
        """
        pooling = self.config.pooling_method
        
        if pooling == PoolingMethod.CLS:
            # 使用第一个token
            return features[:, 0, :]
        
        elif pooling == PoolingMethod.MEAN:
            # 平均池化
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (features * mask).sum(dim=1) / (mask.sum(dim=1) + self.config.eps)
            return features.mean(dim=1)
        
        elif pooling == PoolingMethod.MAX:
            # 最大池化
            if attention_mask is not None:
                features = features.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, 
                    float('-inf')
                )
            return features.max(dim=1)[0]
        
        elif pooling == PoolingMethod.ATTENTION:
            # 注意力加权池化
            if self._attention_pooling is None:
                return features.mean(dim=1)
            
            scores = self._attention_pooling(features).squeeze(-1)  # [B, L]
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
            return (features * weights).sum(dim=1)
        
        elif pooling == PoolingMethod.FIRST_LAST_AVG:
            # 首尾平均
            first = features[:, 0, :]
            last = features[:, -1, :]
            return (first + last) / 2
        
        else:
            return features.mean(dim=1)
    
    def _apply_augmentation(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        应用数据增强
        
        Args:
            inputs: 输入张量
            
        Returns:
            增强后的输入
        """
        aug_type = self.config.augmentation_type
        aug_prob = self.config.augmentation_prob
        
        if aug_type == AugmentationType.NONE:
            return inputs
        
        if torch.rand(1).item() > aug_prob:
            return inputs
        
        if aug_type == AugmentationType.NOISE:
            noise = torch.randn_like(inputs.float()) * self.config.noise_std
            return inputs + noise
        
        elif aug_type == AugmentationType.DROPOUT:
            if self._augmentation_dropout is not None:
                return self._augmentation_dropout(inputs.float())
            return inputs
        
        elif aug_type == AugmentationType.MIXUP:
            # 简化的mixup - 随机混合
            batch_size = inputs.shape[0]
            if batch_size > 1:
                perm = torch.randperm(batch_size, device=inputs.device)
                lam = torch.distributions.Beta(
                    self.config.mixup_alpha, 
                    self.config.mixup_alpha
                ).sample().item()
                return lam * inputs + (1 - lam) * inputs[perm]
            return inputs
        
        elif aug_type == AugmentationType.CUTOUT:
            # 随机遮挡
            if inputs.dim() >= 3:
                batch_size, seq_len = inputs.shape[:2]
                mask = torch.ones_like(inputs)
                cutout_len = int(seq_len * 0.1)
                for i in range(batch_size):
                    start = torch.randint(0, seq_len - cutout_len + 1, (1,)).item()
                    mask[i, start:start + cutout_len] = 0
                return inputs * mask
            return inputs
        
        return inputs
    
    def _check_numerical_stability(self, features: torch.Tensor) -> torch.Tensor:
        """
        检查数值稳定性
        
        Args:
            features: 特征张量
            
        Returns:
            处理后的特征
        """
        eps = self.config.eps
        
        # 检查NaN
        if torch.isnan(features).any():
            logger.warning(f"{self.modality.value} encoder produced NaN values, replacing with zeros")
            features = torch.nan_to_num(features, nan=0.0)
        
        # 检查Inf
        if torch.isinf(features).any():
            logger.warning(f"{self.modality.value} encoder produced Inf values, clipping")
            features = torch.clamp(features, min=-1e6, max=1e6)
        
        return features
    
    def freeze(self) -> None:
        """冻结编码器参数"""
        for param in self.parameters():
            param.requires_grad = False
        logger.info(f"{self.modality.value} encoder frozen")
    
    def unfreeze(self) -> None:
        """解冻编码器参数"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info(f"{self.modality.value} encoder unfrozen")
    
    def get_output_dim(self) -> int:
        """获取输出维度"""
        if self.config.output_projection:
            return self.config.projection_dim
        return self.hidden_size
    
    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用量（MB）"""
        param_memory = sum(
            p.numel() * p.element_size() for p in self.parameters()
        ) / (1024 * 1024)
        
        buffer_memory = sum(
            b.numel() * b.element_size() for b in self.buffers()
        ) / (1024 * 1024)
        
        return {
            "parameters_mb": param_memory,
            "buffers_mb": buffer_memory,
            "total_mb": param_memory + buffer_memory
        }
    
    def encode_batch(self, inputs_list: List[torch.Tensor], 
                     **kwargs) -> List[torch.Tensor]:
        """
        批量编码
        
        Args:
            inputs_list: 输入张量列表
            **kwargs: 额外参数
            
        Returns:
            编码结果列表
        """
        results = []
        for inputs in inputs_list:
            features = self.forward(inputs, **kwargs)
            results.append(features)
        return results
    
    def encode_with_pooling(self, inputs: torch.Tensor, 
                            attention_mask: Optional[torch.Tensor] = None,
                            **kwargs) -> torch.Tensor:
        """
        编码并池化
        
        Args:
            inputs: 输入张量
            attention_mask: 注意力掩码
            **kwargs: 额外参数
            
        Returns:
            池化后的特征 [batch_size, hidden_size]
        """
        features = self.forward(inputs, **kwargs)
        return self._pool_features(features, attention_mask)
    
    def evaluate_quality(self, features: torch.Tensor) -> Dict[str, float]:
        """
        评估编码质量
        
        Args:
            features: 编码后的特征
            
        Returns:
            质量指标字典
        """
        with torch.no_grad():
            quality = {}
            
            # 特征范数
            quality["feature_norm"] = float(features.norm(p=2, dim=-1).mean())
            
            # 特征方差
            quality["feature_variance"] = float(features.var(dim=-1).mean())
            
            # 特征稀疏度（接近0的比例）
            sparse_ratio = (features.abs() < 0.01).float().mean()
            quality["sparsity"] = float(sparse_ratio)
            
            # 特征利用率（非零维度比例）
            non_zero_ratio = (features.abs() > self.config.eps).float().mean()
            quality["utilization"] = float(non_zero_ratio)
            
            # 综合质量分数
            quality["overall_score"] = (
                quality["feature_variance"] * 0.3 +
                quality["utilization"] * 0.4 +
                (1.0 - quality["sparsity"]) * 0.3
            )
            
            # 记录质量分数
            if self._metrics is not None:
                with self._metrics_lock:
                    self._metrics.add_quality_score(quality["overall_score"])
            
            return quality
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        if self._metrics is None:
            return {"metrics_disabled": True}
        
        with self._metrics_lock:
            return self._metrics.to_dict()
    
    def reset_metrics(self) -> None:
        """重置指标"""
        if self._metrics is not None:
            with self._metrics_lock:
                self._metrics = EncoderMetrics()
    
    def get_status(self) -> EncoderStatus:
        """获取状态"""
        return self._status
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "modality": self.modality.value,
            "hidden_size": self.hidden_size,
            "output_dim": self.get_output_dim(),
            "pooling_method": self.config.pooling_method.value,
            "augmentation_type": self.config.augmentation_type.value,
            "parameter_count": self.get_parameter_count(),
            "memory_usage": self.get_memory_usage()
        }


class TextEncoder(ModalityEncoder):
    """
    文本编码器 - 生产级实现
    
    支持BERT、RoBERTa等预训练模型风格的编码。
    
    生产级特性：
    - 注意力可视化
    - 特殊token处理
    - 子词嵌入支持
    - 序列截断和填充
    """
    
    def __init__(self, config: EncoderConfig):
        config.modality = ModalityType.TEXT
        super().__init__(config)
        
        # 简化实现：使用Transformer
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        # Token类型嵌入（用于句子对任务）
        self.token_type_embedding = nn.Embedding(2, config.hidden_size)
        
        # 嵌入层归一化和dropout
        self.embed_norm = nn.LayerNorm(config.hidden_size)
        self.embed_dropout = nn.Dropout(config.dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # 注意力权重存储（用于可视化）
        self._attention_weights: Optional[torch.Tensor] = None
    
    def encode(self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
               token_type_ids: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        编码文本
        
        Args:
            inputs: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            token_type_ids: Token类型IDs [batch_size, seq_len]
        """
        batch_size, seq_len = inputs.shape
        
        # 限制序列长度
        max_len = self.config.max_seq_length
        if seq_len > max_len:
            inputs = inputs[:, :max_len]
            seq_len = max_len
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_len]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, :max_len]
        
        # Embeddings
        positions = torch.arange(seq_len, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
        embeddings = self.embedding(inputs) + self.pos_embedding(positions)
        
        # 添加token类型嵌入
        if token_type_ids is not None:
            embeddings = embeddings + self.token_type_embedding(token_type_ids)
        
        # 嵌入层归一化和dropout
        embeddings = self.embed_norm(embeddings)
        embeddings = self.embed_dropout(embeddings)
        
        # Transformer编码
        if attention_mask is not None:
            # 转换为key_padding_mask (True表示忽略)
            key_padding_mask = attention_mask == 0
        else:
            key_padding_mask = None
        
        encoded = self.encoder(embeddings, src_key_padding_mask=key_padding_mask)
        return self.layer_norm(encoded)
    
    def _compute_attention_analysis(self, features: torch.Tensor, 
                                    attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        计算注意力分析
        
        Args:
            features: 编码后的特征
            attention_mask: 注意力掩码
            
        Returns:
            分析结果
        """
        analysis = {}
        
        # 计算token重要性（基于特征范数）
        token_importance = features.norm(dim=-1)  # [B, L]
        if attention_mask is not None:
            token_importance = token_importance * attention_mask.float()
        
        analysis["token_importance_mean"] = float(token_importance.mean())
        analysis["token_importance_std"] = float(token_importance.std())
        analysis["token_importance_max_idx"] = int(token_importance.mean(dim=0).argmax())
        
        return analysis
    
    def get_attention_analysis(self, inputs: torch.Tensor, 
                               attention_mask: Optional[torch.Tensor] = None,
                               **kwargs) -> Dict[str, Any]:
        """
        获取注意力分析
        
        Args:
            inputs: 输入token IDs
            attention_mask: 注意力掩码
            
        Returns:
            注意力分析结果
        """
        features = self.encode(inputs, attention_mask=attention_mask, **kwargs)
        return self._compute_attention_analysis(features, attention_mask)
    
    def get_token_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        获取token嵌入
        
        Args:
            token_ids: Token IDs
            
        Returns:
            Token嵌入
        """
        return self.embedding(token_ids)


class ImageEncoder(ModalityEncoder):
    """
    图像编码器 - 生产级实现
    
    支持ViT风格的编码。
    
    生产级特性：
    - Patch分析和可视化
    - 多尺度特征
    - 位置编码插值
    - 特征图可视化
    """
    
    def __init__(self, config: EncoderConfig):
        config.modality = ModalityType.IMAGE
        super().__init__(config)
        
        # ViT风格的patch embedding
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # CLS token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.hidden_size))
        
        # Patch dropout（用于正则化）
        self.patch_dropout = nn.Dropout(config.dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Patch重要性存储
        self._patch_importance: Optional[torch.Tensor] = None
    
    def encode(self, inputs: torch.Tensor, return_patch_features: bool = False, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        编码图像
        
        Args:
            inputs: 图像张量 [batch_size, channels, height, width]
            return_patch_features: 是否返回patch特征
            
        Returns:
            编码后的特征，可选返回patch特征
        """
        batch_size = inputs.shape[0]
        
        # Patch embedding
        patches = self.patch_embed(inputs)  # [B, hidden, H/P, W/P]
        H_patches, W_patches = patches.shape[2], patches.shape[3]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, hidden]
        
        # 保存patch特征
        patch_features = patches.clone()
        
        # Patch dropout
        patches = self.patch_dropout(patches)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        
        # 位置编码插值（如果图像大小不同）
        pos_embed = self._interpolate_pos_embed(x.shape[1])
        x = x + pos_embed
        
        # Transformer编码
        encoded = self.encoder(x)
        encoded = self.layer_norm(encoded)
        
        # 计算patch重要性
        self._compute_patch_importance(encoded[:, 1:, :])
        
        if return_patch_features:
            return encoded, patch_features
        return encoded
    
    def _interpolate_pos_embed(self, num_tokens: int) -> torch.Tensor:
        """
        插值位置编码
        
        Args:
            num_tokens: 目标token数量
            
        Returns:
            插值后的位置编码
        """
        if num_tokens == self.pos_embed.shape[1]:
            return self.pos_embed
        
        # 分离CLS token的位置编码
        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]
        
        # 计算原始和目标的patch数量
        orig_patches = patch_pos.shape[1]
        target_patches = num_tokens - 1
        
        if orig_patches == target_patches:
            return self.pos_embed
        
        # 插值patch位置编码
        orig_size = int(math.sqrt(orig_patches))
        target_size = int(math.sqrt(target_patches))
        
        patch_pos = patch_pos.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, 
            size=(target_size, target_size), 
            mode='bicubic', 
            align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_patches, -1)
        
        return torch.cat([cls_pos, patch_pos], dim=1)
    
    def _compute_patch_importance(self, patch_features: torch.Tensor) -> None:
        """
        计算patch重要性
        
        Args:
            patch_features: Patch特征 [B, num_patches, hidden]
        """
        with torch.no_grad():
            # 基于特征范数计算重要性
            importance = patch_features.norm(dim=-1)  # [B, num_patches]
            self._patch_importance = importance
    
    def get_patch_importance(self) -> Optional[torch.Tensor]:
        """获取patch重要性"""
        return self._patch_importance
    
    def get_patch_analysis(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        获取patch分析
        
        Args:
            inputs: 输入图像
            
        Returns:
            分析结果
        """
        encoded, patch_features = self.encode(inputs, return_patch_features=True)
        
        analysis = {}
        
        # Patch统计
        analysis["num_patches"] = patch_features.shape[1]
        analysis["patch_dim"] = patch_features.shape[2]
        
        # 重要性分析
        if self._patch_importance is not None:
            importance = self._patch_importance
            analysis["importance_mean"] = float(importance.mean())
            analysis["importance_std"] = float(importance.std())
            analysis["top_patches"] = importance.mean(dim=0).topk(5).indices.tolist()
        
        # 特征多样性
        with torch.no_grad():
            diversity = torch.cosine_similarity(
                patch_features.unsqueeze(2), 
                patch_features.unsqueeze(1), 
                dim=-1
            ).mean()
            analysis["feature_diversity"] = float(1.0 - diversity)
        
        return analysis
    
    def visualize_attention(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        获取注意力可视化数据
        
        Args:
            inputs: 输入图像
            
        Returns:
            注意力图
        """
        self.encode(inputs)
        
        if self._patch_importance is None:
            return torch.zeros(inputs.shape[0], self.image_size, self.image_size)
        
        # 重塑为2D
        batch_size = self._patch_importance.shape[0]
        side = int(math.sqrt(self._patch_importance.shape[1]))
        attention_map = self._patch_importance.view(batch_size, side, side)
        
        # 上采样到原始图像大小
        attention_map = F.interpolate(
            attention_map.unsqueeze(1), 
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        return attention_map


class AudioEncoder(ModalityEncoder):
    """
    音频编码器 - 生产级实现
    
    支持Wav2Vec、Whisper风格的编码。
    
    生产级特性：
    - 频谱分析
    - 时间标注支持
    - 多尺度特征
    - 语音活动检测
    """
    
    def __init__(self, config: EncoderConfig):
        config.modality = ModalityType.AUDIO
        super().__init__(config)
        
        self.sample_rate = config.sample_rate
        
        # 1D卷积特征提取（多尺度）
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=0),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.Conv1d(512, config.hidden_size, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, config.hidden_size),
            nn.GELU()
        )
        
        # 特征投影
        self.feature_projection = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.zeros(1, 5000, config.hidden_size))
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # 语音活动检测头
        self.vad_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 特征统计存储
        self._frame_energy: Optional[torch.Tensor] = None
    
    def encode(self, inputs: torch.Tensor, return_frame_features: bool = False, 
               **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        编码音频
        
        Args:
            inputs: 音频波形 [batch_size, time] 或 [batch_size, 1, time]
            return_frame_features: 是否返回帧级特征
            
        Returns:
            编码后的特征
        """
        # 确保输入形状
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
        
        # 计算帧能量
        self._compute_frame_energy(inputs)
        
        # 卷积特征提取
        features = self.conv_layers(inputs)  # [B, hidden, T']
        features = features.transpose(1, 2)  # [B, T', hidden]
        
        # 保存帧级特征
        frame_features = features.clone()
        
        # 特征投影
        features = self.feature_projection(features)
        
        # 添加位置编码
        seq_len = features.shape[1]
        if seq_len > self.pos_encoding.shape[1]:
            # 扩展位置编码
            pos_encoding = self._extend_pos_encoding(seq_len)
        else:
            pos_encoding = self.pos_encoding[:, :seq_len, :]
        features = features + pos_encoding
        
        # Transformer编码
        encoded = self.encoder(features)
        encoded = self.layer_norm(encoded)
        
        if return_frame_features:
            return encoded, frame_features
        return encoded
    
    def _compute_frame_energy(self, waveform: torch.Tensor) -> None:
        """
        计算帧能量
        
        Args:
            waveform: 音频波形
        """
        with torch.no_grad():
            # 简化的能量计算
            frame_size = 400  # 25ms at 16kHz
            hop_size = 160    # 10ms at 16kHz
            
            if waveform.shape[-1] < frame_size:
                self._frame_energy = waveform.pow(2).mean(dim=-1, keepdim=True)
                return
            
            # 使用unfold计算帧能量
            waveform_2d = waveform.squeeze(1)
            num_frames = (waveform_2d.shape[-1] - frame_size) // hop_size + 1
            
            frames = waveform_2d.unfold(-1, frame_size, hop_size)
            self._frame_energy = frames.pow(2).mean(dim=-1)
    
    def _extend_pos_encoding(self, target_len: int) -> torch.Tensor:
        """
        扩展位置编码
        
        Args:
            target_len: 目标长度
            
        Returns:
            扩展后的位置编码
        """
        if target_len <= self.pos_encoding.shape[1]:
            return self.pos_encoding[:, :target_len, :]
        
        # 使用插值扩展
        pos = self.pos_encoding.transpose(1, 2)  # [1, hidden, L]
        pos = F.interpolate(pos, size=target_len, mode='linear', align_corners=False)
        return pos.transpose(1, 2)
    
    def detect_voice_activity(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        语音活动检测
        
        Args:
            inputs: 音频波形
            
        Returns:
            VAD概率 [batch_size, num_frames]
        """
        encoded = self.encode(inputs)
        vad_probs = self.vad_head(encoded).squeeze(-1)
        return vad_probs
    
    def get_frame_energy(self) -> Optional[torch.Tensor]:
        """获取帧能量"""
        return self._frame_energy
    
    def get_audio_analysis(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        获取音频分析
        
        Args:
            inputs: 输入音频
            
        Returns:
            分析结果
        """
        encoded, frame_features = self.encode(inputs, return_frame_features=True)
        
        analysis = {}
        
        # 基本信息
        analysis["num_frames"] = encoded.shape[1]
        analysis["frame_dim"] = encoded.shape[2]
        
        # 能量分析
        if self._frame_energy is not None:
            energy = self._frame_energy
            analysis["energy_mean"] = float(energy.mean())
            analysis["energy_std"] = float(energy.std())
            analysis["energy_max"] = float(energy.max())
        
        # VAD分析
        vad_probs = self.detect_voice_activity(inputs)
        analysis["vad_mean"] = float(vad_probs.mean())
        analysis["speech_ratio"] = float((vad_probs > 0.5).float().mean())
        
        return analysis
    
    def compute_duration(self, inputs: torch.Tensor) -> float:
        """
        计算音频时长（秒）
        
        Args:
            inputs: 音频波形
            
        Returns:
            时长（秒）
        """
        num_samples = inputs.shape[-1]
        return num_samples / self.sample_rate


class VideoEncoder(ModalityEncoder):
    """
    视频编码器 - 生产级实现
    
    支持时空Transformer编码。
    
    生产级特性：
    - 时序分析
    - 帧重要性评估
    - 时空注意力
    - 运动特征提取
    """
    
    def __init__(self, config: EncoderConfig):
        config.modality = ModalityType.VIDEO
        super().__init__(config)
        
        # 时间采样
        self.num_frames = config.num_frames
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        
        # 3D Patch embedding
        self.patch_embed = nn.Conv3d(
            config.num_channels,
            config.hidden_size,
            kernel_size=(2, config.patch_size, config.patch_size),
            stride=(2, config.patch_size, config.patch_size)
        )
        
        # 计算patches数量
        self.num_patches_per_frame = (config.image_size // config.patch_size) ** 2
        self.num_temporal_patches = self.num_frames // 2
        self.total_patches = self.num_patches_per_frame * self.num_temporal_patches
        
        # CLS token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.total_patches + 1, config.hidden_size))
        
        # 时间位置编码
        self.temporal_embed = nn.Parameter(torch.zeros(1, self.num_temporal_patches, config.hidden_size))
        
        # Patch dropout
        self.patch_dropout = nn.Dropout(config.dropout)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # 帧重要性预测头
        self.frame_importance_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        
        # 存储
        self._frame_importance: Optional[torch.Tensor] = None
        self._temporal_features: Optional[torch.Tensor] = None
    
    def encode(self, inputs: torch.Tensor, return_temporal_features: bool = False,
               **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        编码视频
        
        Args:
            inputs: 视频张量 [batch_size, channels, frames, height, width]
            return_temporal_features: 是否返回时序特征
            
        Returns:
            编码后的特征
        """
        batch_size = inputs.shape[0]
        num_frames = inputs.shape[2]
        
        # 3D Patch embedding
        patches = self.patch_embed(inputs)  # [B, hidden, T', H', W']
        T_out, H_out, W_out = patches.shape[2], patches.shape[3], patches.shape[4]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, hidden]
        
        # Patch dropout
        patches = self.patch_dropout(patches)
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)
        
        # 添加位置编码
        pos_embed = self._interpolate_pos_embed(x.shape[1])
        x = x + pos_embed
        
        # 添加时间位置编码到patch tokens
        if T_out > 0:
            # 重塑为时序形式并添加时间位置编码
            patch_tokens = x[:, 1:, :]  # [B, T*H*W, hidden]
            num_spatial = H_out * W_out
            patch_tokens = patch_tokens.view(batch_size, T_out, num_spatial, -1)
            
            temporal_embed = self._interpolate_temporal_embed(T_out)
            patch_tokens = patch_tokens + temporal_embed.unsqueeze(2)
            
            patch_tokens = patch_tokens.view(batch_size, -1, self.config.hidden_size)
            x = torch.cat([x[:, :1, :], patch_tokens], dim=1)
        
        # Transformer编码
        encoded = self.encoder(x)
        encoded = self.layer_norm(encoded)
        
        # 提取时序特征并计算帧重要性
        self._compute_temporal_analysis(encoded, T_out, H_out * W_out)
        
        if return_temporal_features:
            return encoded, self._temporal_features
        return encoded
    
    def _interpolate_pos_embed(self, num_tokens: int) -> torch.Tensor:
        """插值位置编码"""
        if num_tokens == self.pos_embed.shape[1]:
            return self.pos_embed
        
        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]
        
        if num_tokens - 1 == patch_pos.shape[1]:
            return self.pos_embed
        
        # 插值
        patch_pos = patch_pos.transpose(1, 2)
        patch_pos = F.interpolate(
            patch_pos, 
            size=num_tokens - 1, 
            mode='linear', 
            align_corners=False
        )
        patch_pos = patch_pos.transpose(1, 2)
        
        return torch.cat([cls_pos, patch_pos], dim=1)
    
    def _interpolate_temporal_embed(self, num_temporal: int) -> torch.Tensor:
        """插值时间位置编码"""
        if num_temporal == self.temporal_embed.shape[1]:
            return self.temporal_embed
        
        temp_embed = self.temporal_embed.transpose(1, 2)
        temp_embed = F.interpolate(
            temp_embed, 
            size=num_temporal, 
            mode='linear', 
            align_corners=False
        )
        return temp_embed.transpose(1, 2)
    
    def _compute_temporal_analysis(self, encoded: torch.Tensor, 
                                   num_temporal: int, num_spatial: int) -> None:
        """
        计算时序分析
        
        Args:
            encoded: 编码后的特征
            num_temporal: 时间步数
            num_spatial: 每帧的空间位置数
        """
        with torch.no_grad():
            # 提取时序特征（排除CLS token）
            patch_features = encoded[:, 1:, :]
            batch_size = patch_features.shape[0]
            
            if num_temporal > 0 and num_spatial > 0:
                # 重塑并聚合每帧的特征
                temporal_features = patch_features.view(
                    batch_size, num_temporal, num_spatial, -1
                ).mean(dim=2)  # [B, T, hidden]
                
                self._temporal_features = temporal_features
                
                # 计算帧重要性
                self._frame_importance = self.frame_importance_head(
                    temporal_features
                ).squeeze(-1)  # [B, T]
    
    def get_frame_importance(self) -> Optional[torch.Tensor]:
        """获取帧重要性"""
        return self._frame_importance
    
    def get_temporal_features(self) -> Optional[torch.Tensor]:
        """获取时序特征"""
        return self._temporal_features
    
    def get_video_analysis(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        获取视频分析
        
        Args:
            inputs: 输入视频
            
        Returns:
            分析结果
        """
        encoded, temporal_features = self.encode(inputs, return_temporal_features=True)
        
        analysis = {}
        
        # 基本信息
        analysis["num_frames_input"] = inputs.shape[2]
        analysis["num_temporal_patches"] = temporal_features.shape[1] if temporal_features is not None else 0
        analysis["hidden_dim"] = encoded.shape[2]
        
        # 帧重要性分析
        if self._frame_importance is not None:
            importance = self._frame_importance
            analysis["frame_importance_mean"] = float(importance.mean())
            analysis["frame_importance_std"] = float(importance.std())
            analysis["most_important_frame"] = int(importance.mean(dim=0).argmax())
        
        # 时序变化分析
        if temporal_features is not None and temporal_features.shape[1] > 1:
            # 计算帧间差异
            frame_diffs = (temporal_features[:, 1:, :] - temporal_features[:, :-1, :]).norm(dim=-1)
            analysis["temporal_change_mean"] = float(frame_diffs.mean())
            analysis["temporal_change_max"] = float(frame_diffs.max())
            analysis["high_motion_frames"] = (frame_diffs > frame_diffs.mean()).sum(dim=-1).float().mean().item()
        
        return analysis
    
    def extract_keyframes(self, inputs: torch.Tensor, num_keyframes: int = 3) -> torch.Tensor:
        """
        提取关键帧索引
        
        Args:
            inputs: 输入视频
            num_keyframes: 关键帧数量
            
        Returns:
            关键帧索引
        """
        self.encode(inputs)
        
        if self._frame_importance is None:
            # 均匀采样
            total_frames = inputs.shape[2]
            indices = torch.linspace(0, total_frames - 1, num_keyframes).long()
            return indices
        
        # 基于重要性选择
        importance = self._frame_importance.mean(dim=0)
        _, indices = importance.topk(min(num_keyframes, len(importance)))
        return indices.sort()[0]


class TimeSeriesEncoder(ModalityEncoder):
    """
    时序编码器 - 生产级实现
    
    支持Transformer和多尺度特征。
    
    生产级特性：
    - 趋势分析
    - 异常检测
    - 多尺度时间特征
    - 季节性分解
    """
    
    def __init__(self, config: EncoderConfig):
        config.modality = ModalityType.TIME_SERIES
        super().__init__(config)
        
        self.num_features = config.num_features
        
        # 输入归一化
        self.input_norm = nn.LayerNorm(config.num_features)
        
        # 输入投影
        self.input_projection = nn.Linear(config.num_features, config.hidden_size)
        
        # 多尺度卷积（捕获不同时间尺度的模式）
        self.multiscale_convs = nn.ModuleList([
            nn.Conv1d(config.hidden_size, config.hidden_size // 4, kernel_size=k, padding=k//2)
            for k in [3, 5, 7, 9]
        ])
        self.scale_fusion = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 使用Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.zeros(1, config.max_seq_length, config.hidden_size))
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # 异常检测头
        self.anomaly_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 趋势预测头
        self.trend_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, config.num_features)
        )
        
        # 存储
        self._trend_features: Optional[torch.Tensor] = None
        self._anomaly_scores: Optional[torch.Tensor] = None
    
    def encode(self, inputs: torch.Tensor, return_multiscale: bool = False,
               **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        编码时序数据
        
        Args:
            inputs: 时序数据 [batch_size, seq_len, num_features]
            return_multiscale: 是否返回多尺度特征
            
        Returns:
            编码后的特征
        """
        # 输入归一化
        x = self.input_norm(inputs)
        
        # 投影到隐藏维度
        x = self.input_projection(x)
        
        # 多尺度特征提取
        x_conv = x.transpose(1, 2)  # [B, hidden, L]
        multiscale_features = []
        for conv in self.multiscale_convs:
            feat = conv(x_conv)
            multiscale_features.append(feat)
        
        # 融合多尺度特征
        multiscale_cat = torch.cat(multiscale_features, dim=1)  # [B, hidden, L]
        multiscale_cat = multiscale_cat.transpose(1, 2)  # [B, L, hidden]
        x = x + self.scale_fusion(multiscale_cat)
        
        # 添加位置编码
        seq_len = x.shape[1]
        if seq_len > self.pos_encoding.shape[1]:
            pos_encoding = self._extend_pos_encoding(seq_len)
        else:
            pos_encoding = self.pos_encoding[:, :seq_len, :]
        x = x + pos_encoding
        
        # Transformer编码
        encoded = self.encoder(x)
        encoded = self.layer_norm(encoded)
        
        # 计算趋势和异常
        self._compute_analysis(encoded)
        
        if return_multiscale:
            return encoded, multiscale_cat
        return encoded
    
    def _extend_pos_encoding(self, target_len: int) -> torch.Tensor:
        """扩展位置编码"""
        pos = self.pos_encoding.transpose(1, 2)
        pos = F.interpolate(pos, size=target_len, mode='linear', align_corners=False)
        return pos.transpose(1, 2)
    
    def _compute_analysis(self, encoded: torch.Tensor) -> None:
        """计算分析"""
        with torch.no_grad():
            # 异常分数
            self._anomaly_scores = self.anomaly_head(encoded).squeeze(-1)
            
            # 趋势特征
            self._trend_features = self.trend_head(encoded)
    
    def detect_anomalies(self, inputs: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检测异常
        
        Args:
            inputs: 时序数据
            threshold: 异常阈值
            
        Returns:
            异常分数和异常标志
        """
        self.encode(inputs)
        
        if self._anomaly_scores is None:
            return torch.zeros(inputs.shape[:2]), torch.zeros(inputs.shape[:2], dtype=torch.bool)
        
        scores = self._anomaly_scores
        anomalies = scores > threshold
        return scores, anomalies
    
    def predict_trend(self, inputs: torch.Tensor, horizon: int = 1) -> torch.Tensor:
        """
        预测趋势
        
        Args:
            inputs: 时序数据
            horizon: 预测步数
            
        Returns:
            预测值
        """
        encoded = self.encode(inputs)
        
        # 使用最后一个时间步预测
        last_features = encoded[:, -1:, :]
        
        predictions = []
        for _ in range(horizon):
            pred = self.trend_head(last_features)
            predictions.append(pred)
            # 简化：使用预测作为下一步输入
            last_features = self.input_projection(pred)
        
        return torch.cat(predictions, dim=1)
    
    def get_time_series_analysis(self, inputs: torch.Tensor) -> Dict[str, Any]:
        """
        获取时序分析
        
        Args:
            inputs: 输入时序
            
        Returns:
            分析结果
        """
        encoded, multiscale = self.encode(inputs, return_multiscale=True)
        
        analysis = {}
        
        # 基本信息
        analysis["seq_length"] = inputs.shape[1]
        analysis["num_features"] = inputs.shape[2]
        
        # 统计分析
        with torch.no_grad():
            analysis["mean"] = float(inputs.mean())
            analysis["std"] = float(inputs.std())
            analysis["min"] = float(inputs.min())
            analysis["max"] = float(inputs.max())
        
        # 异常分析
        if self._anomaly_scores is not None:
            scores = self._anomaly_scores
            analysis["anomaly_mean"] = float(scores.mean())
            analysis["anomaly_max"] = float(scores.max())
            analysis["anomaly_ratio"] = float((scores > 0.5).float().mean())
        
        # 趋势分析
        if self._trend_features is not None:
            trend = self._trend_features
            # 计算趋势方向
            if trend.shape[1] > 1:
                trend_diff = trend[:, 1:, :] - trend[:, :-1, :]
                analysis["trend_direction"] = "up" if trend_diff.mean() > 0 else "down"
                analysis["trend_strength"] = float(trend_diff.abs().mean())
        
        return analysis


class TabularEncoder(ModalityEncoder):
    """
    表格数据编码器 - 生产级实现
    
    支持数值和类别特征。
    
    生产级特性：
    - 特征重要性评估
    - 特征交互建模
    - 缺失值处理
    - 特征归一化
    """
    
    def __init__(self, config: EncoderConfig):
        config.modality = ModalityType.TABULAR
        super().__init__(config)
        
        self.num_features = config.num_features
        
        # 特征级别归一化
        self.feature_norm = nn.LayerNorm(config.num_features)
        
        # 特征嵌入（每个特征单独嵌入）
        self.feature_embeddings = nn.Linear(config.num_features, config.hidden_size * config.num_features)
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * config.num_features, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
        # 特征交互层（自注意力）
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size)
        )
        
        # 特征重要性预测
        self.importance_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_features),
            nn.Softmax(dim=-1)
        )
        
        # 存储
        self._feature_importance: Optional[torch.Tensor] = None
        self._feature_interactions: Optional[torch.Tensor] = None
    
    def encode(self, inputs: torch.Tensor, feature_mask: Optional[torch.Tensor] = None,
               **kwargs) -> torch.Tensor:
        """
        编码表格数据
        
        Args:
            inputs: 表格数据 [batch_size, num_features]
            feature_mask: 特征掩码（用于缺失值）[batch_size, num_features]
            
        Returns:
            编码后的特征
        """
        batch_size = inputs.shape[0]
        
        # 处理缺失值
        if feature_mask is not None:
            # 将缺失值替换为0
            inputs = inputs * feature_mask
        
        # 特征归一化
        x = self.feature_norm(inputs)
        
        # 特征嵌入
        embedded = self.feature_embeddings(x)  # [B, hidden * num_features]
        embedded = embedded.view(batch_size, self.num_features, self.config.hidden_size)  # [B, F, hidden]
        
        # 特征交互（自注意力）
        if feature_mask is not None:
            key_padding_mask = feature_mask == 0
        else:
            key_padding_mask = None
        
        interacted, attention_weights = self.feature_attention(
            embedded, embedded, embedded,
            key_padding_mask=key_padding_mask
        )
        self._feature_interactions = attention_weights
        
        # 融合特征
        fused = embedded + interacted
        fused = fused.view(batch_size, -1)  # [B, hidden * num_features]
        fused = self.feature_fusion(fused)  # [B, hidden]
        
        # MLP
        x = self.mlp(fused)
        
        # 计算特征重要性
        self._compute_feature_importance(x)
        
        # 添加序列维度
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        return x
    
    def _compute_feature_importance(self, features: torch.Tensor) -> None:
        """计算特征重要性"""
        with torch.no_grad():
            self._feature_importance = self.importance_head(features)
    
    def get_feature_importance(self) -> Optional[torch.Tensor]:
        """获取特征重要性"""
        return self._feature_importance
    
    def get_feature_interactions(self) -> Optional[torch.Tensor]:
        """获取特征交互矩阵"""
        return self._feature_interactions
    
    def get_tabular_analysis(self, inputs: torch.Tensor, 
                              feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        获取表格分析
        
        Args:
            inputs: 输入数据
            feature_names: 特征名称列表
            
        Returns:
            分析结果
        """
        self.encode(inputs)
        
        analysis = {}
        
        # 基本信息
        analysis["num_samples"] = inputs.shape[0]
        analysis["num_features"] = inputs.shape[1]
        
        # 特征统计
        with torch.no_grad():
            analysis["feature_means"] = inputs.mean(dim=0).tolist()
            analysis["feature_stds"] = inputs.std(dim=0).tolist()
        
        # 特征重要性
        if self._feature_importance is not None:
            importance = self._feature_importance.mean(dim=0)
            analysis["feature_importance"] = importance.tolist()
            
            # 最重要的特征
            top_indices = importance.topk(min(5, len(importance))).indices.tolist()
            if feature_names is not None:
                analysis["top_features"] = [feature_names[i] for i in top_indices]
            else:
                analysis["top_feature_indices"] = top_indices
        
        # 特征交互
        if self._feature_interactions is not None:
            interactions = self._feature_interactions.mean(dim=0)
            analysis["avg_interaction_strength"] = float(interactions.mean())
            
            # 找出最强的特征对交互
            if interactions.shape[0] > 1:
                # 排除对角线
                mask = 1 - torch.eye(interactions.shape[0], device=interactions.device)
                masked = interactions * mask
                max_idx = masked.argmax()
                i, j = max_idx // interactions.shape[1], max_idx % interactions.shape[1]
                analysis["strongest_interaction"] = (int(i), int(j), float(masked[i, j]))
        
        return analysis
    
    def handle_missing_values(self, inputs: torch.Tensor, 
                               strategy: str = "zero") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理缺失值
        
        Args:
            inputs: 输入数据（NaN表示缺失）
            strategy: 处理策略 ("zero", "mean", "median")
            
        Returns:
            处理后的数据和缺失掩码
        """
        # 创建缺失掩码
        mask = ~torch.isnan(inputs)
        
        # 替换缺失值
        filled = inputs.clone()
        
        if strategy == "zero":
            filled[~mask] = 0.0
        elif strategy == "mean":
            means = torch.nanmean(inputs, dim=0)
            for i in range(inputs.shape[1]):
                filled[~mask[:, i], i] = means[i]
        elif strategy == "median":
            # 简化：使用均值代替中位数
            means = torch.nanmean(inputs, dim=0)
            for i in range(inputs.shape[1]):
                filled[~mask[:, i], i] = means[i]
        
        return filled, mask.float()


class GraphEncoder(ModalityEncoder):
    """
    图编码器 - 生产级实现
    
    支持图神经网络风格的编码。
    
    生产级特性：
    - 多层图注意力
    - 节点重要性评估
    - 图级别表示
    - 边特征处理
    """
    
    def __init__(self, config: EncoderConfig):
        config.modality = ModalityType.GRAPH
        super().__init__(config)
        
        self.num_nodes = config.num_nodes
        self.edge_dim = config.edge_dim
        
        # 节点特征投影
        self.node_projection = nn.Linear(config.num_features, config.hidden_size)
        
        # 边特征投影（如果有边特征）
        self.edge_projection = nn.Linear(config.edge_dim, config.hidden_size) if config.edge_dim > 0 else None
        
        # 图注意力层
        self.graph_attention_layers = nn.ModuleList([
            GraphAttentionLayer(config.hidden_size, config.num_heads, config.dropout)
            for _ in range(config.num_layers)
        ])
        
        # 图级别池化
        self.graph_pooling = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1)
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # 节点重要性头
        self.node_importance_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 存储
        self._node_importance: Optional[torch.Tensor] = None
        self._attention_weights: Optional[List[torch.Tensor]] = None
    
    def encode(self, node_features: torch.Tensor, 
               adjacency: torch.Tensor,
               edge_features: Optional[torch.Tensor] = None,
               **kwargs) -> torch.Tensor:
        """
        编码图数据
        
        Args:
            node_features: 节点特征 [batch_size, num_nodes, num_features]
            adjacency: 邻接矩阵 [batch_size, num_nodes, num_nodes]
            edge_features: 边特征 [batch_size, num_nodes, num_nodes, edge_dim]
            
        Returns:
            编码后的特征
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # 节点特征投影
        x = self.node_projection(node_features)
        
        # 边特征投影
        edge_emb = None
        if edge_features is not None and self.edge_projection is not None:
            edge_emb = self.edge_projection(edge_features)
        
        # 图注意力层
        attention_weights = []
        for layer in self.graph_attention_layers:
            x, attn = layer(x, adjacency, edge_emb)
            attention_weights.append(attn)
        
        self._attention_weights = attention_weights
        
        # 层归一化
        x = self.layer_norm(x)
        
        # 计算节点重要性
        self._compute_node_importance(x)
        
        return x
    
    def _compute_node_importance(self, node_features: torch.Tensor) -> None:
        """计算节点重要性"""
        with torch.no_grad():
            self._node_importance = self.node_importance_head(node_features).squeeze(-1)
    
    def get_graph_representation(self, node_features: torch.Tensor,
                                 adjacency: torch.Tensor,
                                 edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取图级别表示
        
        Args:
            node_features: 节点特征
            adjacency: 邻接矩阵
            edge_features: 边特征
            
        Returns:
            图级别表示 [batch_size, hidden_size]
        """
        node_encoded = self.encode(node_features, adjacency, edge_features)
        
        # 注意力池化
        scores = self.graph_pooling(node_encoded).squeeze(-1)  # [B, N]
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, N, 1]
        graph_repr = (node_encoded * weights).sum(dim=1)  # [B, hidden]
        
        return graph_repr
    
    def get_node_importance(self) -> Optional[torch.Tensor]:
        """获取节点重要性"""
        return self._node_importance
    
    def get_attention_weights(self) -> Optional[List[torch.Tensor]]:
        """获取注意力权重"""
        return self._attention_weights
    
    def get_graph_analysis(self, node_features: torch.Tensor,
                          adjacency: torch.Tensor) -> Dict[str, Any]:
        """
        获取图分析
        
        Args:
            node_features: 节点特征
            adjacency: 邻接矩阵
            
        Returns:
            分析结果
        """
        self.encode(node_features, adjacency)
        
        analysis = {}
        
        # 基本信息
        analysis["num_nodes"] = node_features.shape[1]
        analysis["num_edges"] = int((adjacency > 0).sum() / adjacency.shape[0])
        
        # 节点重要性分析
        if self._node_importance is not None:
            importance = self._node_importance
            analysis["node_importance_mean"] = float(importance.mean())
            analysis["node_importance_std"] = float(importance.std())
            analysis["top_nodes"] = importance.mean(dim=0).topk(min(5, importance.shape[1])).indices.tolist()
        
        # 注意力分析
        if self._attention_weights is not None and len(self._attention_weights) > 0:
            last_attn = self._attention_weights[-1]
            analysis["attention_entropy"] = float(
                -(last_attn * (last_attn + 1e-10).log()).sum(dim=-1).mean()
            )
        
        return analysis


class GraphAttentionLayer(nn.Module):
    """图注意力层"""
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 节点特征 [B, N, hidden]
            adjacency: 邻接矩阵 [B, N, N]
            edge_features: 边特征 [B, N, N, hidden]
            
        Returns:
            更新后的节点特征和注意力权重
        """
        batch_size, num_nodes, _ = x.shape
        
        # QKV投影
        q = self.query(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用邻接掩码
        mask = adjacency.unsqueeze(1)  # [B, 1, N, N]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1)
        out = self.output(out)
        
        # 残差连接和层归一化
        out = self.layer_norm(x + out)
        
        return out, attn.mean(dim=1)  # 返回平均注意力


class PointCloudEncoder(ModalityEncoder):
    """
    点云编码器 - 生产级实现
    
    支持PointNet风格的编码。
    
    生产级特性：
    - 点级别特征
    - 全局特征聚合
    - 法向量处理
    - 点重要性评估
    """
    
    def __init__(self, config: EncoderConfig):
        config.modality = ModalityType.POINT_CLOUD
        super().__init__(config)
        
        self.num_points = config.num_points
        self.point_dim = config.point_dim
        
        # 点级别MLP
        self.point_mlp = nn.Sequential(
            nn.Linear(config.point_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, config.hidden_size),
            nn.BatchNorm1d(config.hidden_size),
            nn.ReLU()
        )
        
        # Transformer编码器（用于点间交互）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers // 2)
        
        # 全局特征提取
        self.global_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # 点重要性头
        self.point_importance_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # 存储
        self._point_importance: Optional[torch.Tensor] = None
        self._point_features: Optional[torch.Tensor] = None
    
    def encode(self, points: torch.Tensor, 
               normals: Optional[torch.Tensor] = None,
               return_point_features: bool = False,
               **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        编码点云数据
        
        Args:
            points: 点坐标 [batch_size, num_points, point_dim]
            normals: 法向量 [batch_size, num_points, 3]
            return_point_features: 是否返回点级别特征
            
        Returns:
            编码后的特征
        """
        batch_size, num_points, point_dim = points.shape
        
        # 合并法向量（如果有）
        if normals is not None:
            points = torch.cat([points, normals], dim=-1)
        
        # 点级别MLP（需要重塑以适应BatchNorm）
        points_flat = points.view(batch_size * num_points, -1)
        
        # 如果输入维度不匹配，需要投影
        if points_flat.shape[-1] != self.point_dim:
            point_projection = nn.Linear(
                points_flat.shape[-1], self.point_dim, device=points.device
            )
            points_flat = point_projection(points_flat)
        
        point_features = self._apply_point_mlp(points_flat, batch_size, num_points)
        
        # Transformer编码
        point_features = self.transformer(point_features)
        
        # 层归一化
        point_features = self.layer_norm(point_features)
        
        # 保存点级别特征
        self._point_features = point_features
        
        # 计算点重要性
        self._compute_point_importance(point_features)
        
        if return_point_features:
            return point_features, self._point_features
        return point_features
    
    def _apply_point_mlp(self, points_flat: torch.Tensor, 
                         batch_size: int, num_points: int) -> torch.Tensor:
        """应用点级别MLP"""
        # 逐层应用以处理BatchNorm
        x = points_flat
        for i, layer in enumerate(self.point_mlp):
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            else:
                x = layer(x)
        
        return x.view(batch_size, num_points, -1)
    
    def _compute_point_importance(self, point_features: torch.Tensor) -> None:
        """计算点重要性"""
        with torch.no_grad():
            self._point_importance = self.point_importance_head(point_features).squeeze(-1)
    
    def get_global_feature(self, points: torch.Tensor,
                          normals: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取全局特征
        
        Args:
            points: 点坐标
            normals: 法向量
            
        Returns:
            全局特征 [batch_size, hidden_size]
        """
        point_features = self.encode(points, normals)
        
        # 最大池化获取全局特征
        global_feature = point_features.max(dim=1)[0]
        global_feature = self.global_mlp(global_feature)
        
        return global_feature
    
    def get_point_importance(self) -> Optional[torch.Tensor]:
        """获取点重要性"""
        return self._point_importance
    
    def get_point_cloud_analysis(self, points: torch.Tensor,
                                 normals: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        获取点云分析
        
        Args:
            points: 点坐标
            normals: 法向量
            
        Returns:
            分析结果
        """
        self.encode(points, normals)
        
        analysis = {}
        
        # 基本信息
        analysis["num_points"] = points.shape[1]
        analysis["point_dim"] = points.shape[2]
        analysis["has_normals"] = normals is not None
        
        # 几何统计
        with torch.no_grad():
            analysis["centroid"] = points.mean(dim=1).mean(dim=0).tolist()
            analysis["bounding_box_min"] = points.min(dim=1)[0].mean(dim=0).tolist()
            analysis["bounding_box_max"] = points.max(dim=1)[0].mean(dim=0).tolist()
            
            # 点分布
            distances = torch.cdist(points, points)
            # 排除对角线（自身距离）
            mask = 1 - torch.eye(points.shape[1], device=points.device)
            masked_distances = distances * mask.unsqueeze(0) + (1 - mask.unsqueeze(0)) * 1e10
            analysis["avg_nearest_neighbor_dist"] = float(masked_distances.min(dim=-1)[0].mean())
        
        # 点重要性分析
        if self._point_importance is not None:
            importance = self._point_importance
            analysis["point_importance_mean"] = float(importance.mean())
            analysis["point_importance_std"] = float(importance.std())
            analysis["top_points"] = importance.mean(dim=0).topk(min(10, importance.shape[1])).indices.tolist()
        
        return analysis
    
    def sample_points(self, points: torch.Tensor, 
                      num_samples: int,
                      strategy: str = "fps") -> torch.Tensor:
        """
        采样点
        
        Args:
            points: 点坐标 [batch_size, num_points, point_dim]
            num_samples: 采样数量
            strategy: 采样策略 ("fps", "random", "importance")
            
        Returns:
            采样后的点 [batch_size, num_samples, point_dim]
        """
        batch_size, num_points, point_dim = points.shape
        
        if num_samples >= num_points:
            return points
        
        if strategy == "random":
            # 随机采样
            indices = torch.randperm(num_points)[:num_samples]
            return points[:, indices, :]
        
        elif strategy == "importance":
            # 基于重要性采样
            self.encode(points)
            if self._point_importance is not None:
                _, indices = self._point_importance.mean(dim=0).topk(num_samples)
                return points[:, indices, :]
            return points[:, :num_samples, :]
        
        else:  # fps - Farthest Point Sampling
            # 简化的FPS实现
            sampled_indices = [0]
            distances = torch.full((batch_size, num_points), float('inf'), device=points.device)
            
            for _ in range(num_samples - 1):
                last_idx = sampled_indices[-1]
                last_point = points[:, last_idx:last_idx+1, :]
                dist = torch.cdist(last_point, points).squeeze(1)
                distances = torch.min(distances, dist)
                next_idx = distances.mean(dim=0).argmax().item()
                sampled_indices.append(next_idx)
            
            indices = torch.tensor(sampled_indices, device=points.device)
            return points[:, indices, :]


# ==================== 编码器工厂 ====================

class EncoderFactory:
    """
    编码器工厂 - 生产级实现
    
    提供：
    - 编码器注册和创建
    - 工厂级别指标
    - 可用模态查询
    """
    
    _registry: Dict[ModalityType, type] = {
        ModalityType.TEXT: TextEncoder,
        ModalityType.IMAGE: ImageEncoder,
        ModalityType.AUDIO: AudioEncoder,
        ModalityType.VIDEO: VideoEncoder,
        ModalityType.TIME_SERIES: TimeSeriesEncoder,
        ModalityType.TABULAR: TabularEncoder,
        ModalityType.GRAPH: GraphEncoder,
        ModalityType.POINT_CLOUD: PointCloudEncoder
    }
    
    _metrics: Dict[str, Any] = {
        "total_created": 0,
        "created_by_modality": {},
        "errors": 0
    }
    _metrics_lock = threading.Lock()
    
    @classmethod
    def register(cls, modality: ModalityType, encoder_cls: type) -> None:
        """注册编码器"""
        cls._registry[modality] = encoder_cls
        logger.info(f"Registered encoder for modality: {modality.value}")
    
    @classmethod
    def unregister(cls, modality: ModalityType) -> bool:
        """注销编码器"""
        if modality in cls._registry:
            del cls._registry[modality]
            logger.info(f"Unregistered encoder for modality: {modality.value}")
            return True
        return False
    
    @classmethod
    def create(cls, modality: Union[ModalityType, str], 
               config: Optional[EncoderConfig] = None, **kwargs) -> ModalityEncoder:
        """
        创建编码器
        
        Args:
            modality: 模态类型
            config: 编码器配置
            **kwargs: 额外参数
            
        Returns:
            编码器实例
        """
        try:
            if isinstance(modality, str):
                modality = ModalityType(modality)
            
            encoder_cls = cls._registry.get(modality)
            if encoder_cls is None:
                raise ValueError(f"Unknown modality: {modality}")
            
            if config is None:
                config = EncoderConfig(modality=modality, **kwargs)
            
            encoder = encoder_cls(config)
            
            # 更新指标
            with cls._metrics_lock:
                cls._metrics["total_created"] += 1
                modality_key = modality.value
                cls._metrics["created_by_modality"][modality_key] = \
                    cls._metrics["created_by_modality"].get(modality_key, 0) + 1
            
            logger.debug(f"Created {modality.value} encoder")
            return encoder
            
        except Exception as e:
            with cls._metrics_lock:
                cls._metrics["errors"] += 1
            logger.error(f"Error creating encoder: {e}")
            raise
    
    @classmethod
    def get_available_modalities(cls) -> List[str]:
        """获取可用模态列表"""
        return [m.value for m in cls._registry.keys()]
    
    @classmethod
    def is_modality_supported(cls, modality: Union[ModalityType, str]) -> bool:
        """检查模态是否支持"""
        if isinstance(modality, str):
            try:
                modality = ModalityType(modality)
            except ValueError:
                return False
        return modality in cls._registry
    
    @classmethod
    def get_factory_metrics(cls) -> Dict[str, Any]:
        """获取工厂指标"""
        with cls._metrics_lock:
            return cls._metrics.copy()
    
    @classmethod
    def reset_metrics(cls) -> None:
        """重置工厂指标"""
        with cls._metrics_lock:
            cls._metrics = {
                "total_created": 0,
                "created_by_modality": {},
                "errors": 0
            }
    
    @classmethod
    def get_encoder_info(cls, modality: Union[ModalityType, str]) -> Dict[str, Any]:
        """
        获取编码器信息
        
        Args:
            modality: 模态类型
            
        Returns:
            编码器信息
        """
        if isinstance(modality, str):
            modality = ModalityType(modality)
        
        encoder_cls = cls._registry.get(modality)
        if encoder_cls is None:
            return {"error": f"Unknown modality: {modality}"}
        
        return {
            "modality": modality.value,
            "class_name": encoder_cls.__name__,
            "docstring": encoder_cls.__doc__,
            "module": encoder_cls.__module__
        }


class MultiModalEncoder(nn.Module):
    """
    多模态组合编码器 - 生产级实现
    
    组合多个模态编码器进行联合编码。
    
    生产级特性：
    - 动态模态组合
    - 模态级别特征融合
    - 缺失模态处理
    - 模态权重学习
    """
    
    def __init__(self, modality_configs: Dict[ModalityType, EncoderConfig],
                 fusion_hidden_size: int = 768,
                 fusion_method: str = "attention"):
        """
        初始化多模态编码器
        
        Args:
            modality_configs: 各模态配置
            fusion_hidden_size: 融合隐藏层大小
            fusion_method: 融合方法 ("concat", "attention", "gated")
        """
        super().__init__()
        
        self.modality_configs = modality_configs
        self.fusion_hidden_size = fusion_hidden_size
        self.fusion_method = fusion_method
        
        # 创建各模态编码器
        self.encoders = nn.ModuleDict()
        self.projectors = nn.ModuleDict()
        
        for modality, config in modality_configs.items():
            encoder = EncoderFactory.create(modality, config)
            self.encoders[modality.value] = encoder
            
            # 投影到统一维度
            if encoder.get_output_dim() != fusion_hidden_size:
                self.projectors[modality.value] = nn.Linear(
                    encoder.get_output_dim(), fusion_hidden_size
                )
        
        # 融合层
        num_modalities = len(modality_configs)
        if fusion_method == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(fusion_hidden_size * num_modalities, fusion_hidden_size * 2),
                nn.GELU(),
                nn.Linear(fusion_hidden_size * 2, fusion_hidden_size),
                nn.LayerNorm(fusion_hidden_size)
            )
        elif fusion_method == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=fusion_hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(fusion_hidden_size)
        elif fusion_method == "gated":
            self.gates = nn.ModuleDict({
                m.value: nn.Sequential(
                    nn.Linear(fusion_hidden_size, fusion_hidden_size),
                    nn.Sigmoid()
                )
                for m in modality_configs.keys()
            })
            self.fusion_norm = nn.LayerNorm(fusion_hidden_size)
        
        # 模态权重（可学习）
        self.modality_weights = nn.Parameter(
            torch.ones(num_modalities) / num_modalities
        )
        
        # 缺失模态的默认特征
        self.default_features = nn.ParameterDict({
            m.value: nn.Parameter(torch.zeros(1, 1, fusion_hidden_size))
            for m in modality_configs.keys()
        })
        
        # 指标
        self._metrics = EncoderMetrics()
        self._metrics_lock = threading.Lock()
    
    def forward(self, inputs: Dict[ModalityType, torch.Tensor],
                masks: Optional[Dict[ModalityType, torch.Tensor]] = None,
                **kwargs) -> torch.Tensor:
        """
        前向传播
        
        Args:
            inputs: 各模态输入 {modality: tensor}
            masks: 各模态掩码
            **kwargs: 额外参数
            
        Returns:
            融合后的特征
        """
        start_time = time.time()
        
        # 编码各模态
        modality_features = {}
        for modality in self.modality_configs.keys():
            if modality in inputs:
                features = self.encoders[modality.value](inputs[modality], **kwargs.get(modality.value, {}))
                
                # 投影到统一维度
                if modality.value in self.projectors:
                    features = self.projectors[modality.value](features)
                
                # 池化到单个向量
                features = features.mean(dim=1, keepdim=True)
                modality_features[modality] = features
            else:
                # 使用默认特征
                batch_size = list(inputs.values())[0].shape[0]
                modality_features[modality] = self.default_features[modality.value].expand(
                    batch_size, -1, -1
                )
        
        # 融合
        if self.fusion_method == "concat":
            fused = self._concat_fusion(modality_features)
        elif self.fusion_method == "attention":
            fused = self._attention_fusion(modality_features)
        else:
            fused = self._gated_fusion(modality_features)
        
        # 记录指标
        elapsed = time.time() - start_time
        with self._metrics_lock:
            self._metrics.record_encoding(elapsed, len(inputs))
        
        return fused
    
    def _concat_fusion(self, modality_features: Dict[ModalityType, torch.Tensor]) -> torch.Tensor:
        """拼接融合"""
        features_list = [modality_features[m] for m in self.modality_configs.keys()]
        concatenated = torch.cat(features_list, dim=-1)
        return self.fusion(concatenated)
    
    def _attention_fusion(self, modality_features: Dict[ModalityType, torch.Tensor]) -> torch.Tensor:
        """注意力融合"""
        # 堆叠模态特征
        features_list = [modality_features[m] for m in self.modality_configs.keys()]
        stacked = torch.cat(features_list, dim=1)  # [B, num_modalities, hidden]
        
        # 自注意力
        attended, _ = self.fusion_attention(stacked, stacked, stacked)
        
        # 加权平均
        weights = F.softmax(self.modality_weights, dim=0)
        fused = (attended * weights.view(1, -1, 1)).sum(dim=1, keepdim=True)
        
        return self.fusion_norm(fused)
    
    def _gated_fusion(self, modality_features: Dict[ModalityType, torch.Tensor]) -> torch.Tensor:
        """门控融合"""
        gated_features = []
        for modality in self.modality_configs.keys():
            features = modality_features[modality]
            gate = self.gates[modality.value](features)
            gated_features.append(features * gate)
        
        # 加权求和
        weights = F.softmax(self.modality_weights, dim=0)
        fused = sum(w * f for w, f in zip(weights, gated_features))
        
        return self.fusion_norm(fused)
    
    def get_modality_weights(self) -> Dict[str, float]:
        """获取模态权重"""
        weights = F.softmax(self.modality_weights, dim=0)
        return {
            m.value: float(w)
            for m, w in zip(self.modality_configs.keys(), weights)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        with self._metrics_lock:
            return self._metrics.to_dict()
    
    def get_encoder(self, modality: Union[ModalityType, str]) -> Optional[ModalityEncoder]:
        """获取单个模态的编码器"""
        if isinstance(modality, str):
            modality = ModalityType(modality)
        
        if modality.value in self.encoders:
            return self.encoders[modality.value]
        return None


# ==================== 质量分析器 ====================

class EncoderQualityAnalyzer:
    """
    编码器质量分析器
    
    提供编码质量的综合评估。
    """
    
    def __init__(self):
        self._analysis_history: List[Dict[str, Any]] = []
        self._max_history = 100
    
    def analyze(self, encoder: ModalityEncoder, 
                inputs: torch.Tensor,
                **kwargs) -> Dict[str, Any]:
        """
        分析编码质量
        
        Args:
            encoder: 编码器
            inputs: 输入数据
            **kwargs: 额外参数
            
        Returns:
            分析结果
        """
        analysis = {}
        
        with torch.no_grad():
            # 编码
            features = encoder(inputs, **kwargs)
            
            # 基本质量评估
            quality = encoder.evaluate_quality(features)
            analysis.update(quality)
            
            # 特征统计
            analysis["feature_mean"] = float(features.mean())
            analysis["feature_std"] = float(features.std())
            analysis["feature_min"] = float(features.min())
            analysis["feature_max"] = float(features.max())
            
            # 特征分布分析
            analysis["kurtosis"] = self._compute_kurtosis(features)
            analysis["skewness"] = self._compute_skewness(features)
            
            # 维度利用分析
            analysis["effective_rank"] = self._compute_effective_rank(features)
            
            # 时间戳
            analysis["timestamp"] = datetime.now().isoformat()
            analysis["modality"] = encoder.modality.value
        
        # 保存历史
        self._analysis_history.append(analysis)
        if len(self._analysis_history) > self._max_history:
            self._analysis_history = self._analysis_history[-self._max_history:]
        
        return analysis
    
    def _compute_kurtosis(self, features: torch.Tensor) -> float:
        """计算峰度"""
        mean = features.mean()
        std = features.std()
        if std < 1e-8:
            return 0.0
        normalized = (features - mean) / std
        return float(normalized.pow(4).mean() - 3)
    
    def _compute_skewness(self, features: torch.Tensor) -> float:
        """计算偏度"""
        mean = features.mean()
        std = features.std()
        if std < 1e-8:
            return 0.0
        normalized = (features - mean) / std
        return float(normalized.pow(3).mean())
    
    def _compute_effective_rank(self, features: torch.Tensor) -> float:
        """计算有效秩"""
        # 简化的有效秩计算
        features_2d = features.view(-1, features.shape[-1])
        
        # SVD
        try:
            _, s, _ = torch.svd(features_2d)
            s = s / s.sum()
            # Shannon熵
            entropy = -(s * (s + 1e-10).log()).sum()
            return float(entropy.exp())
        except:
            return float(features.shape[-1])
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取分析历史"""
        return self._analysis_history.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        if not self._analysis_history:
            return {"error": "No analysis history"}
        
        summary = {
            "total_analyses": len(self._analysis_history),
            "modalities_analyzed": list(set(a.get("modality", "unknown") for a in self._analysis_history))
        }
        
        # 计算平均质量指标
        quality_keys = ["overall_score", "feature_variance", "utilization", "sparsity"]
        for key in quality_keys:
            values = [a.get(key, 0) for a in self._analysis_history if key in a]
            if values:
                summary[f"avg_{key}"] = sum(values) / len(values)
        
        return summary


# ==================== 配置构建器 ====================

class EncoderConfigBuilder:
    """
    编码器配置构建器
    
    支持链式配置构建。
    """
    
    def __init__(self, modality: Union[ModalityType, str] = ModalityType.TEXT):
        if isinstance(modality, str):
            modality = ModalityType(modality)
        self._config = {
            "modality": modality
        }
    
    def hidden_size(self, size: int) -> "EncoderConfigBuilder":
        """设置隐藏层大小"""
        self._config["hidden_size"] = size
        return self
    
    def num_layers(self, layers: int) -> "EncoderConfigBuilder":
        """设置层数"""
        self._config["num_layers"] = layers
        return self
    
    def num_heads(self, heads: int) -> "EncoderConfigBuilder":
        """设置注意力头数"""
        self._config["num_heads"] = heads
        return self
    
    def dropout(self, rate: float) -> "EncoderConfigBuilder":
        """设置dropout率"""
        self._config["dropout"] = rate
        return self
    
    def pooling(self, method: Union[PoolingMethod, str]) -> "EncoderConfigBuilder":
        """设置池化方法"""
        if isinstance(method, str):
            method = PoolingMethod(method)
        self._config["pooling_method"] = method
        return self
    
    def augmentation(self, aug_type: Union[AugmentationType, str], 
                     prob: float = 0.5) -> "EncoderConfigBuilder":
        """设置数据增强"""
        if isinstance(aug_type, str):
            aug_type = AugmentationType(aug_type)
        self._config["augmentation_type"] = aug_type
        self._config["augmentation_prob"] = prob
        return self
    
    def with_projection(self, dim: int) -> "EncoderConfigBuilder":
        """添加输出投影"""
        self._config["output_projection"] = True
        self._config["projection_dim"] = dim
        return self
    
    def enable_metrics(self, enabled: bool = True) -> "EncoderConfigBuilder":
        """启用指标收集"""
        self._config["enable_metrics"] = enabled
        return self
    
    def vocab_size(self, size: int) -> "EncoderConfigBuilder":
        """设置词表大小（文本模态）"""
        self._config["vocab_size"] = size
        return self
    
    def max_seq_length(self, length: int) -> "EncoderConfigBuilder":
        """设置最大序列长度"""
        self._config["max_seq_length"] = length
        return self
    
    def image_size(self, size: int) -> "EncoderConfigBuilder":
        """设置图像大小"""
        self._config["image_size"] = size
        return self
    
    def patch_size(self, size: int) -> "EncoderConfigBuilder":
        """设置patch大小"""
        self._config["patch_size"] = size
        return self
    
    def num_features(self, features: int) -> "EncoderConfigBuilder":
        """设置特征数量"""
        self._config["num_features"] = features
        return self
    
    def num_frames(self, frames: int) -> "EncoderConfigBuilder":
        """设置帧数（视频模态）"""
        self._config["num_frames"] = frames
        return self
    
    def num_points(self, points: int) -> "EncoderConfigBuilder":
        """设置点数（点云模态）"""
        self._config["num_points"] = points
        return self
    
    def build(self) -> EncoderConfig:
        """构建配置"""
        return EncoderConfig(**self._config)
    
    def create_encoder(self) -> ModalityEncoder:
        """创建编码器"""
        config = self.build()
        return EncoderFactory.create(config.modality, config)


# ==================== 便捷函数 ====================

def create_encoder(
    modality: Union[ModalityType, str],
    hidden_size: int = 768,
    **kwargs
) -> ModalityEncoder:
    """
    便捷函数：创建编码器
    
    Args:
        modality: 模态类型
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        编码器实例
    """
    config = EncoderConfig(
        modality=ModalityType(modality) if isinstance(modality, str) else modality,
        hidden_size=hidden_size,
        **kwargs
    )
    return EncoderFactory.create(modality, config)


def build_encoder_config(modality: Union[ModalityType, str] = "text") -> EncoderConfigBuilder:
    """
    便捷函数：创建配置构建器
    
    Args:
        modality: 模态类型
        
    Returns:
        配置构建器
    """
    return EncoderConfigBuilder(modality)


def create_multimodal_encoder(
    modalities: List[Union[ModalityType, str]],
    hidden_size: int = 768,
    fusion_method: str = "attention",
    **kwargs
) -> MultiModalEncoder:
    """
    便捷函数：创建多模态编码器
    
    Args:
        modalities: 模态列表
        hidden_size: 隐藏层大小
        fusion_method: 融合方法
        **kwargs: 各模态额外配置
        
    Returns:
        多模态编码器
    """
    modality_configs = {}
    for modality in modalities:
        if isinstance(modality, str):
            modality = ModalityType(modality)
        
        config_kwargs = kwargs.get(modality.value, {})
        config = EncoderConfig(
            modality=modality,
            hidden_size=hidden_size,
            **config_kwargs
        )
        modality_configs[modality] = config
    
    return MultiModalEncoder(
        modality_configs=modality_configs,
        fusion_hidden_size=hidden_size,
        fusion_method=fusion_method
    )


def encoder_factory_health_check() -> Dict[str, Any]:
    """
    工厂健康检查
    
    Returns:
        健康状态
    """
    health = {
        "status": "healthy",
        "available_modalities": EncoderFactory.get_available_modalities(),
        "metrics": EncoderFactory.get_factory_metrics()
    }
    
    # 检查各模态是否可以创建
    errors = []
    for modality in ModalityType:
        if EncoderFactory.is_modality_supported(modality):
            try:
                # 尝试创建编码器
                config = EncoderConfig(modality=modality)
                encoder = EncoderFactory.create(modality, config)
                del encoder
            except Exception as e:
                errors.append(f"{modality.value}: {str(e)}")
    
    if errors:
        health["status"] = "degraded"
        health["errors"] = errors
    
    return health


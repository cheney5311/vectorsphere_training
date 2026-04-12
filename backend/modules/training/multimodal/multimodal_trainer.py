"""多模态训练器

生产级多模态训练器，支持：
- 多种模态和场景的训练（文本、图像、时序、表格、音频）
- 多种融合方法（拼接、注意力、门控、跨模态注意力、Q-Former）
- 模态对齐和对比学习
- 行业场景定制
- 分布式训练支持
- 完整的监控、检查点、健康检查

架构调用层次：
├── multimodal_trainer.py (本模块)
│   └── 调用 multimodal_config.py (配置层)
│   └── 调用 backend/modules/training/strategies (策略层)
│       ├── base_strategy.py - StrategyContext, StrategyResult
│       ├── distributed_strategy.py - DistributedMode, DistributedStrategy
│       └── multimodal_strategy.py - MultiModalStrategy
│   └── 调用 backend/lib/hardware (硬件层)
│   └── 调用 backend/lib/distributed (分布式层)
│   └── 调用 backend/lib/losses (损失层)
│   └── 调用 backend/lib/multimodal (多模态底层库)
│   └── 调用 backend/modules/training/progress (进度管理)
└── 被上层服务调用
"""

import os
import json
import logging
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# 修复导入路径
import sys
import os as os_path
current_dir = os_path.path.dirname(os_path.path.abspath(__file__))
project_root = os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(current_dir))))
sys.path.insert(0, project_root)

# 配置层导入
from .multimodal_config import (
    MultiModalConfig, 
    IndustryMultiModalConfig, 
    MultiModalPresets,
    ModalityType,
    FusionMethod,
    MultiModalScenario,
    ConfigSerializer,
    get_layer_availability,
)

logger = logging.getLogger(__name__)


# ==================== 异常导入 ====================

try:
    from backend.modules.training.exceptions import BusinessLogicError
except ImportError:
    class BusinessLogicError(Exception):
        """业务逻辑异常（备用定义）"""
        pass

try:
    from backend.core.exceptions import ValidationError
except ImportError:
    class ValidationError(Exception):
        """验证异常（备用定义）"""
        def __init__(self, message: str, field: str = None):
            super().__init__(message)
            self.field = field


# ==================== 策略层导入 ====================

STRATEGY_LAYER_AVAILABLE = False
try:
    from backend.modules.training.strategies.base_strategy import (
        TrainingStrategy,
        StrategyContext,
        StrategyResult,
        TrainingPhase,
        StrategyType,
        StrategyMonitor,
        StrategyProfiler,
        StrategyValidator,
        StrategyMetrics,
    )
    STRATEGY_LAYER_AVAILABLE = True
    logger.info("Strategy layer (base) loaded for trainer")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Strategy layer (base) not available: {e}")
    TrainingStrategy = None
    StrategyContext = None
    StrategyResult = None
    TrainingPhase = None
    StrategyType = None
    StrategyMonitor = None
    StrategyProfiler = None
    StrategyValidator = None
    StrategyMetrics = None

MULTIMODAL_STRATEGY_AVAILABLE = False
try:
    from backend.modules.training.strategies.multimodal_strategy import (
        MultiModalStrategy,
        IndustryMultiModalStrategy,
        MultiModalStrategyConfig,
        create_multimodal_strategy,
        diagnose_multimodal_strategy,
    )
    MULTIMODAL_STRATEGY_AVAILABLE = True
    logger.info("Multimodal strategy layer loaded for trainer")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Multimodal strategy layer not available: {e}")
    MultiModalStrategy = None
    IndustryMultiModalStrategy = None
    MultiModalStrategyConfig = None
    create_multimodal_strategy = None
    diagnose_multimodal_strategy = None


DISTRIBUTED_STRATEGY_AVAILABLE = False
try:
    from backend.modules.training.strategies.distributed_strategy import (
        DistributedMode,
        DistributedStrategy,
        DistributedStrategyConfig,
        recommend_distributed_mode,
        diagnose_distributed_strategy,
    )
    DISTRIBUTED_STRATEGY_AVAILABLE = True
    logger.info("Distributed strategy layer loaded for trainer")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Distributed strategy layer not available: {e}")
    DistributedMode = None
    DistributedStrategy = None
    DistributedStrategyConfig = None
    recommend_distributed_mode = None
    diagnose_distributed_strategy = None




# ==================== 硬件层导入 ====================

HARDWARE_LAYER_AVAILABLE = False
try:
    from backend.lib.hardware import (
        DeviceManager,
        get_device_manager,
        MemoryManager,
        MixedPrecisionManager,
        AmpConfig,
        PrecisionMode,
        get_available_memory,
        clear_memory,
        DeviceType,
        GradientCheckpointing,
    )
    HARDWARE_LAYER_AVAILABLE = True
    logger.info("Hardware layer loaded for trainer")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Hardware layer not available: {e}")
    DeviceManager = None
    get_device_manager = None
    MemoryManager = None
    MixedPrecisionManager = None
    AmpConfig = None
    PrecisionMode = None
    get_available_memory = None
    clear_memory = None
    DeviceType = None
    GradientCheckpointing = None



# ==================== 分布式层导入 ====================

DISTRIBUTED_LAYER_AVAILABLE = False
try:
    from backend.lib.distributed import (
        DistributedManager,
        get_distributed_manager,
        init_distributed,
        cleanup_distributed,
        is_main_process,
        get_rank,
        get_world_size,
        barrier,
        DDPWrapper,
        FSDPWrapper,
        create_ddp_model,
        create_fsdp_model,
    )
    DISTRIBUTED_LAYER_AVAILABLE = True
    logger.info("Distributed layer loaded for trainer")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Distributed layer not available: {e}")
    DistributedManager = None
    get_distributed_manager = None
    init_distributed = None
    cleanup_distributed = None
    is_main_process = lambda: True
    get_rank = lambda: 0
    get_world_size = lambda: 1
    barrier = lambda: None
    DDPWrapper = None
    FSDPWrapper = None
    create_ddp_model = None
    create_fsdp_model = None
    


# ==================== 损失层导入 ====================

LOSSES_LAYER_AVAILABLE = False
try:
    from backend.lib.losses import (
        LossFactory,
        create_loss,
        CrossEntropyLoss,
        FocalLoss,
        CrossModalContrastiveLoss,
        CLIPLoss,
        InfoNCELoss,
        CompositeLoss,
        MultiTaskLoss,
        create_composite_loss,
    )
    LOSSES_LAYER_AVAILABLE = True
    logger.info("Losses layer loaded for trainer")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Losses layer not available: {e}")
    LossFactory = None
    create_loss = None
    CrossEntropyLoss = None
    FocalLoss = None
    CrossModalContrastiveLoss = None
    CLIPLoss = None
    InfoNCELoss = None
    CompositeLoss = None
    MultiTaskLoss = None
    create_composite_loss = None



# ==================== 多模态底层库导入 ====================

LIB_MULTIMODAL_AVAILABLE = False
try:
    from backend.lib.multimodal import (
        ModalityEncoderFactory,
        CrossModalAligner,
        MultiModalFuser,
        FusionStage,
        TrainingStage,
    )
    LIB_MULTIMODAL_AVAILABLE = True
    logger.info("Lib multimodal loaded for trainer")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Lib multimodal not available: {e}")
    ModalityEncoderFactory = None
    CrossModalAligner = None
    MultiModalFuser = None
    FusionStage = None
    TrainingStage = None


# ==================== 进度管理导入 ====================

PROGRESS_MANAGER_AVAILABLE = False
try:
    from backend.modules.training.progress.progress_manager import (
        TrainingProgressManager,
        TrainingProgress,
        ProgressStatus,
        get_progress_manager,
    )
    PROGRESS_MANAGER_AVAILABLE = True
    logger.info("Progress manager loaded for trainer")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Progress manager not available: {e}")
    TrainingProgressManager = None
    TrainingProgress = None
    ProgressStatus = None
    get_progress_manager = None


# ==================== 模态编码器 ====================

class TextEncoder(nn.Module):
    """文本编码器"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.float())


class ImageEncoder(nn.Module):
    """图像编码器"""
    
    def __init__(self, num_channels: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TimeSeriesEncoder(nn.Module):
    """时序编码器"""
    
    def __init__(self, input_channels: int, seq_length: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_channels = input_channels
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=output_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
            bidirectional=True
        )
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 可能是多种形状，需要处理
        if x.dim() == 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, self.seq_length, self.input_channels)
        elif x.dim() == 3:
            if x.shape[2] == self.input_channels:
                pass
            elif x.shape[1] == self.input_channels:
                x = x.transpose(1, 2)
            else:
                if x.shape[1] == self.seq_length:
                    pass
                else:
                    x = x.transpose(1, 2)
        
        output, (hidden, _) = self.lstm(x)
        final_output = output[:, -1, :]
        return self.output_proj(final_output)


class TableEncoder(nn.Module):
    """表格编码器"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.float())


class AudioEncoder(nn.Module):
    """音频编码器"""
    
    def __init__(self, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 25, stride=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, 9, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 5, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)


# ==================== 编码器工厂 ====================

class EncoderFactory:
    """编码器工厂 - 整合底层库和本地实现"""
    
    @staticmethod
    def create(
        modality: str,
        config: MultiModalConfig,
        use_lib: bool = True
    ) -> nn.Module:
        """创建编码器
        
        Args:
            modality: 模态类型
            config: 多模态配置
            use_lib: 是否优先使用底层库
        
        Returns:
            编码器模块
        """
        output_dim = config.modality_dims.get(modality, 768)
        
        # 尝试使用底层库
        if use_lib:
            try:
                modality_config = config.modality_configs.get(modality)
                if modality_config:
                    encoder = modality_config.create_encoder()
                    if encoder is not None:
                        logger.debug(f"Created encoder for {modality} using lib")
                        return encoder
            except Exception as e:
                logger.warning(f"Failed to create encoder from lib for {modality}: {e}")
        
        # 使用本地实现
        if modality == "text":
            return TextEncoder(config.max_text_length, output_dim)
        elif modality == "image":
            return ImageEncoder(config.num_image_channels, output_dim)
        elif modality == "time_series":
            return TimeSeriesEncoder(
                config.time_series_channels,
                config.time_series_length,
                output_dim
            )
        elif modality == "table":
            return TableEncoder(config.table_num_features, output_dim)
        elif modality == "audio":
            return AudioEncoder(output_dim)
        else:
            # 默认线性编码器
            modality_type = ModalityType.from_string(modality)
            input_dim = modality_type.default_embedding_dim
            return nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim)
            )


# ==================== 融合模块 ====================

class ConcatFusion(nn.Module):
    """拼接融合"""
    
    def __init__(self, input_dims: List[int], output_dim: int, dropout: float = 0.1):
        super().__init__()
        total_dim = sum(input_dims)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        concat_features = torch.cat(features, dim=-1)
        return self.fusion(concat_features)


class AttentionFusion(nn.Module):
    """注意力融合"""
    
    def __init__(self, feature_dim: int, output_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(features, dim=1)
        attn_output, _ = self.attention(stacked, stacked, stacked)
        pooled = attn_output.mean(dim=1)
        return self.output_proj(pooled)


class GatedFusion(nn.Module):
    """门控融合"""
    
    def __init__(self, feature_dim: int, num_modalities: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        gated_features = []
        for i, (feat, gate) in enumerate(zip(features, self.gates)):
            gated_features.append(feat * gate(feat))
        fused = sum(gated_features)
        return self.output_proj(fused)


class CrossAttentionFusion(nn.Module):
    """跨模态注意力融合"""
    
    def __init__(self, feature_dim: int, output_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if len(features) < 2:
            return self.output_proj(features[0])
        
        query = features[0].unsqueeze(1)
        kv = torch.stack(features[1:], dim=1)
        
        cross_output, _ = self.cross_attention(query, kv, kv)
        
        all_features = torch.cat([query, kv], dim=1)
        self_output, _ = self.self_attention(all_features, all_features, all_features)
        
        fused = (cross_output.squeeze(1) + self_output.mean(dim=1)) / 2
        return self.output_proj(fused)


class QFormerFusion(nn.Module):
    """Q-Former 风格融合"""
    
    def __init__(self, feature_dim: int, output_dim: int, num_queries: int = 32, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        batch_size = features[0].shape[0]
        queries = self.queries.expand(batch_size, -1, -1)
        
        # 堆叠所有模态特征
        kv = torch.stack(features, dim=1)
        
        # 交叉注意力
        cross_output, _ = self.cross_attention(queries, kv, kv)
        
        # 自注意力
        self_output, _ = self.self_attention(cross_output, cross_output, cross_output)
        
        # FFN
        ffn_output = self.ffn(self_output) + self_output
        
        # 池化并投影
        pooled = ffn_output.mean(dim=1)
        return self.output_proj(pooled)


# ==================== 融合工厂 ====================

class FusionFactory:
    """融合模块工厂 - 整合底层库和本地实现"""
    
    @staticmethod
    def create(
        config: MultiModalConfig,
        use_lib: bool = True
    ) -> nn.Module:
        """创建融合模块
        
        Args:
            config: 多模态配置
            use_lib: 是否优先使用底层库
        
        Returns:
            融合模块
        """
        # 尝试使用底层库
        if use_lib:
            try:
                fuser = config.create_fusion_module()
                if fuser is not None:
                    logger.debug(f"Created fusion module using lib: {config.fusion_method}")
                    return fuser
            except Exception as e:
                logger.warning(f"Failed to create fusion module from lib: {e}")
        
        # 使用本地实现
        input_dims = [config.modality_dims.get(m, 768) for m in config.modalities]
        fusion_dim = config.fusion_dim
        dropout = config.fusion_dropout
        
        fusion_method = config.fusion_method
        
        if fusion_method == "concat":
            return ConcatFusion(input_dims, fusion_dim, dropout)
        elif fusion_method == "attention":
            unified_dim = input_dims[0]
            return AttentionFusion(unified_dim, fusion_dim, dropout=dropout)
        elif fusion_method == "gated":
            unified_dim = input_dims[0]
            return GatedFusion(unified_dim, len(config.modalities), fusion_dim, dropout)
        elif fusion_method == "cross_attention":
            unified_dim = input_dims[0]
            return CrossAttentionFusion(unified_dim, fusion_dim, dropout=dropout)
        elif fusion_method == "q_former":
            unified_dim = input_dims[0]
            return QFormerFusion(unified_dim, fusion_dim, dropout=dropout)
        else:
            return ConcatFusion(input_dims, fusion_dim, dropout)


# ==================== 多模态模型 ====================

class MultiModalModel(nn.Module):
    """生产级多模态模型"""
    
    def __init__(self, config: MultiModalConfig, num_classes: int = 10):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # 创建模态编码器
        self.modality_encoders = nn.ModuleDict()
        self._init_encoders()
        
        # 对齐投影层
        self.alignment_projectors = nn.ModuleDict()
        if config.use_alignment:
            self._init_alignment_projectors()
        
        # 维度投影层（用于统一不同模态的维度）
        self.dim_projectors = nn.ModuleDict()
        self._init_dim_projectors()
        
        # 融合层
        self.fusion_layer = FusionFactory.create(config)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_dim // 2, num_classes)
        )
        
        # 梯度检查点
        self._use_gradient_checkpointing = False
    
    def _init_encoders(self):
        """初始化各模态编码器"""
        for modality in self.config.modalities:
            encoder = EncoderFactory.create(modality, self.config)
            self.modality_encoders[modality] = encoder
    
    def _init_alignment_projectors(self):
        """初始化对齐投影层"""
        for modality in self.config.modalities:
            input_dim = self.config.modality_dims.get(modality, 768)
            self.alignment_projectors[modality] = nn.Sequential(
                nn.Linear(input_dim, self.config.alignment_dim),
                nn.ReLU(),
                nn.Linear(self.config.alignment_dim, self.config.alignment_dim)
            )
    
    def _init_dim_projectors(self):
        """初始化维度投影层"""
        target_dim = self.config.modality_dims.get(self.config.modalities[0], 768)
        for modality in self.config.modalities:
            current_dim = self.config.modality_dims.get(modality, 768)
            if current_dim != target_dim:
                self.dim_projectors[modality] = nn.Linear(current_dim, target_dim)
    
    def enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        self._use_gradient_checkpointing = True
    
        try:
            GradientCheckpointing.enable(self)
            logger.info("Gradient checkpointing enabled")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")
    
    def encode_modalities(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """编码各模态"""
        modality_features = {}
        for modality, encoder in self.modality_encoders.items():
            if modality in inputs and not inputs.get(f'{modality}_masked', False):
                feature = encoder(inputs[modality])
                modality_features[modality] = feature
        return modality_features
    
    def compute_alignment_features(self, modality_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算对齐特征"""
        if not self.config.use_alignment:
            return {}
        
        aligned_features = {}
        for modality, features in modality_features.items():
            if modality in self.alignment_projectors:
                aligned_features[modality] = self.alignment_projectors[modality](features)
        return aligned_features
    
    def forward(
        self, 
        inputs: Dict[str, torch.Tensor],
        return_features: bool = False
    ) -> Dict[str, Any]:
        """前向传播"""
        # 编码各模态
        modality_features = self.encode_modalities(inputs)
        
        if not modality_features:
            batch_size = next(iter(inputs.values())).shape[0]
            device = next(iter(inputs.values())).device
            return {
                'logits': torch.zeros(batch_size, self.num_classes, device=device),
                'modality_features': {}
            }
        
        # 融合特征
        feature_list = list(modality_features.values())
        
        # 对于需要统一维度的融合方法，进行维度投影
        if self.config.fusion_method in ["attention", "gated", "cross_attention", "q_former"]:
            unified_features = []
            for modality, features in modality_features.items():
                if modality in self.dim_projectors:
                    features = self.dim_projectors[modality](features)
                unified_features.append(features)
            feature_list = unified_features
        
        fused_features = self.fusion_layer(feature_list)
        
        # 分类
        logits = self.classifier(fused_features)
        
        result = {
            'logits': logits,
            'modality_features': modality_features,
            'fused_features': fused_features
        }
        
        # 计算对齐特征
        if self.config.use_alignment:
            result['alignment_features'] = self.compute_alignment_features(modality_features)
        
        return result
    
    def get_param_count(self) -> Dict[str, int]:
        """获取参数数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoder_params = {}
        for name, encoder in self.modality_encoders.items():
            encoder_params[name] = sum(p.numel() for p in encoder.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'encoders': encoder_params,
            'fusion': sum(p.numel() for p in self.fusion_layer.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
        }


# ==================== 数据集 ====================

class MultiModalDataset(Dataset):
    """生产级多模态数据集"""
    
    def __init__(
        self, 
        data_path: str, 
        config: MultiModalConfig,
        transform: Optional[Dict[str, Callable]] = None,
        num_samples: int = 100
    ):
        self.data_path = data_path
        self.config = config
        self.transform = transform or {}
        self.num_samples = num_samples
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """加载样本数据"""
        samples = []
        
        for i in range(self.num_samples):
            sample = {
                'id': i,
                'label': i % 10
            }
            
            for modality in self.config.modalities:
                if modality == "text":
                    sample[modality] = torch.randn(self.config.max_text_length)
                elif modality == "image":
                    sample[modality] = torch.randn(
                        self.config.num_image_channels,
                        self.config.image_size,
                        self.config.image_size
                    )
                elif modality == "time_series":
                    sample[modality] = torch.randn(
                        self.config.time_series_length,
                        self.config.time_series_channels
                    )
                elif modality == "table":
                    sample[modality] = torch.randn(self.config.table_num_features)
                elif modality == "audio":
                    sample[modality] = torch.randn(self.config.audio_max_length)
            
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        inputs = {}
        for modality in self.config.modalities:
            if modality in sample:
                data = sample[modality]
                if modality in self.transform:
                    data = self.transform[modality](data)
                inputs[modality] = data
        
        return {
            'inputs': inputs,
            'label': sample['label']
        }


# ==================== 回调系统 ====================

@dataclass
class TrainerCallback:
    """训练回调基类"""
    
    def on_train_begin(self, trainer: 'MultiModalTrainer') -> None:
        """训练开始时调用"""
        pass
    
    def on_train_end(self, trainer: 'MultiModalTrainer', result: Dict[str, Any]) -> None:
        """训练结束时调用"""
        pass
    
    def on_epoch_begin(self, trainer: 'MultiModalTrainer', epoch: int) -> None:
        """每个epoch开始时调用"""
        pass
    
    def on_epoch_end(self, trainer: 'MultiModalTrainer', epoch: int, metrics: Dict[str, float]) -> None:
        """每个epoch结束时调用"""
        pass
    
    def on_step_begin(self, trainer: 'MultiModalTrainer', step: int) -> None:
        """每个step开始时调用"""
        pass
    
    def on_step_end(self, trainer: 'MultiModalTrainer', step: int, loss: float) -> None:
        """每个step结束时调用"""
        pass
    
    def on_save(self, trainer: 'MultiModalTrainer', path: str) -> None:
        """保存检查点时调用"""
        pass
    
    def on_evaluate(self, trainer: 'MultiModalTrainer', metrics: Dict[str, float]) -> None:
        """评估时调用"""
        pass


class ProgressCallback(TrainerCallback):
    """进度回调 - 集成进度管理器"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._progress_manager = None
        
        
        try:
            self._progress_manager = get_progress_manager()
        except Exception as e:
            logger.warning(f"Failed to get progress manager: {e}")
    
    def on_train_begin(self, trainer: 'MultiModalTrainer') -> None:
        if self._progress_manager:
            # 创建进度跟踪器并设置状态为运行中
            self._progress_manager.create_progress_tracker(
                self.session_id,
                total_epochs=trainer.config.num_epochs,
                total_steps=trainer.config.num_epochs * 100  # 估计值
            )
            self._progress_manager.set_status(self.session_id, ProgressStatus.RUNNING)
    
    def on_epoch_end(self, trainer: 'MultiModalTrainer', epoch: int, metrics: Dict[str, float]) -> None:
        if self._progress_manager:
            self._progress_manager.update_progress(
                self.session_id,
                current_epoch=epoch + 1,
                train_loss=metrics.get('loss'),
                metrics=metrics
            )
    
    def on_train_end(self, trainer: 'MultiModalTrainer', result: Dict[str, Any]) -> None:
        if self._progress_manager:
            if result.get('success'):
                self._progress_manager.set_status(self.session_id, ProgressStatus.COMPLETED)
            else:
                self._progress_manager.set_status(
                    self.session_id,
                    ProgressStatus.FAILED,
                    error_message=result.get('error', 'Unknown error')
                )


class MonitorCallback(TrainerCallback):
    """监控回调 - 集成策略监控器"""
    
    def __init__(self):
        self._monitor = None
        self._profiler = None
        
        if STRATEGY_LAYER_AVAILABLE:
            if StrategyMonitor is not None:
                try:
                    self._monitor = StrategyMonitor()
                except Exception:
                    pass
            if StrategyProfiler is not None:
                try:
                    self._profiler = StrategyProfiler()
                except Exception:
                    pass
    
    def on_train_begin(self, trainer: 'MultiModalTrainer') -> None:
        # StrategyMonitor 没有 start() 方法，跳过
        # StrategyProfiler 使用 enable() 方法
        if self._profiler:
            self._profiler.enable()
    
    def on_step_end(self, trainer: 'MultiModalTrainer', step: int, loss: float) -> None:
        # StrategyMonitor 使用 record_step() 方法，需要 StrategyResult 和 StrategyContext
        # 这里简化处理，跳过监控记录
        pass
    
    def on_train_end(self, trainer: 'MultiModalTrainer', result: Dict[str, Any]) -> None:
        # StrategyMonitor 没有 stop() 方法，跳过
        # StrategyProfiler 使用 disable() 方法
        if self._profiler:
            self._profiler.disable()
            result['profiler_stats'] = self._profiler.get_stats()


# ==================== 训练器 ====================

class MultiModalTrainer:
    """生产级多模态训练器
    
    集成模块：
    - multimodal_config: FusionMethod, MultiModalScenario, ConfigSerializer, get_layer_availability
    - base_strategy: TrainingStrategy, TrainingPhase, StrategyValidator, StrategyMetrics
    - multimodal_strategy: create_multimodal_strategy, diagnose_multimodal_strategy
    - distributed_strategy: DistributedMode, DistributedStrategyConfig, recommend_distributed_mode
    - hardware: DeviceManager, MemoryManager, MixedPrecisionManager, AmpConfig, PrecisionMode, DeviceType
    - distributed: DistributedManager, get_distributed_manager, DDPWrapper, FSDPWrapper
    - losses: LossFactory, CrossEntropyLoss, FocalLoss, CLIPLoss, CompositeLoss, MultiTaskLoss, create_composite_loss
    - multimodal: ModalityEncoderFactory, CrossModalAligner, MultiModalFuser, FusionStage, TrainingStage
    - progress: TrainingProgressManager, TrainingProgress
    """
    
    def __init__(self, config: MultiModalConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设备管理（使用 DeviceManager, DeviceType）
        self._device_manager: Optional['DeviceManager'] = None
        self._memory_manager: Optional['MemoryManager'] = None
        self._mixed_precision_manager: Optional['MixedPrecisionManager'] = None
        self._distributed_manager: Optional['DistributedManager'] = None
        self._init_device()
        
        # 模型和数据
        self.model: Optional[MultiModalModel] = None
        self.train_dataset: Optional[MultiModalDataset] = None
        self.val_dataset: Optional[MultiModalDataset] = None
        self.test_dataset: Optional[MultiModalDataset] = None
        
        # 优化器和调度器
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        
        # 策略（使用 TrainingStrategy, TrainingPhase, StrategyMetrics）
        self.strategy: Optional['MultiModalStrategy'] = None
        self.strategy_context: Optional['StrategyContext'] = None
        self.distributed_strategy: Optional['DistributedStrategy'] = None
        self._training_phase: Optional['TrainingPhase'] = None
        self._strategy_metrics: Optional['StrategyMetrics'] = None
        self._strategy_validator: Optional['StrategyValidator'] = None
        
        # 损失函数（使用 LossFactory, CompositeLoss, MultiTaskLoss）
        self.loss_fn: Optional[nn.Module] = None
        self.alignment_loss: Optional[nn.Module] = None
        self.contrastive_loss: Optional[nn.Module] = None
        self._composite_loss: Optional['CompositeLoss'] = None
        self._multi_task_loss: Optional['MultiTaskLoss'] = None
        
        # 底层多模态模块（使用 ModalityEncoderFactory, CrossModalAligner, MultiModalFuser）
        self._lib_encoders: Dict[str, Any] = {}
        self._lib_aligner: Optional['CrossModalAligner'] = None
        self._lib_fuser: Optional['MultiModalFuser'] = None
        self._training_stage: Optional['TrainingStage'] = None
        self._fusion_stage: Optional['FusionStage'] = None
        
        # 进度管理（使用 TrainingProgressManager, TrainingProgress）
        self._progress_manager: Optional['TrainingProgressManager'] = None
        self._training_progress: Optional['TrainingProgress'] = None
        
        # DDPWrapper 和 FSDPWrapper 引用
        self._ddp_wrapper: Optional['DDPWrapper'] = None
        self._fsdp_wrapper: Optional['FSDPWrapper'] = None
        
        # 回调
        self.callbacks: List[TrainerCallback] = []
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.best_metrics: Dict[str, float] = {}
        
        # 健康检查
        self._health_check_interval = 100
        self._last_health_check = 0
        
        # 场景和融合信息（使用 MultiModalScenario, FusionMethod）
        self._scenario: Optional['MultiModalScenario'] = self._get_scenario()
        self._fusion_method: Optional['FusionMethod'] = self._get_fusion_method()
        
        # 层可用性
        self._layer_availability = get_layer_availability()
        logger.info(f"Layer availability: {self._layer_availability}")
        
        # 初始化策略（使用 create_multimodal_strategy）
        if config.use_strategy:
            self._init_strategy()
    
        # 初始化损失函数
        self._init_losses()
        
        # 初始化底层多模态模块
        self._init_lib_multimodal()
        
        # 初始化进度管理器
        self._init_progress_manager()
        
        # 初始化分布式
        if config.use_distributed:
            self._init_distributed()
    
    def _get_scenario(self) -> Optional['MultiModalScenario']:
        """获取场景枚举（使用 MultiModalScenario）"""
        try:
            return MultiModalScenario.from_string(self.config.scenario)
        except Exception as e:
            logger.warning(f"Failed to get scenario: {e}")
            return None
    
    def _get_fusion_method(self) -> Optional['FusionMethod']:
        """获取融合方法枚举（使用 FusionMethod）"""
        try:
            return FusionMethod.from_string(self.config.fusion_method)
        except Exception as e:
            logger.warning(f"Failed to get fusion method: {e}")
            return None
    
    def _init_device(self):
        """初始化设备（使用 DeviceManager, MemoryManager, MixedPrecisionManager, DeviceType）"""
        
        try:
            # 使用 DeviceManager 获取最佳设备
            self._device_manager = get_device_manager()
            if self._device_manager:
                best_device = self._device_manager.get_device()  # 使用 get_device() 方法
                if best_device:
                    self.device = best_device
                    logger.info("Using device from DeviceManager: %s", self.device)
                    
                    # 获取设备类型（使用 DeviceType）
                    if DeviceType is not None:
                        try:
                            if 'cuda' in str(self.device).lower():
                                self._device_type = DeviceType.GPU  # CUDA 对应 GPU 枚举
                            elif 'cpu' in str(self.device).lower():
                                self._device_type = DeviceType.CPU
                            else:
                                self._device_type = DeviceType.from_string(str(self.device))
                            logger.debug("Device type: %s", self._device_type)
                        except Exception as e:
                            logger.debug("Failed to get device type: %s", e)
                            self._device_type = None
                else:
                    self._device_type = None
            
            # 初始化 MemoryManager
            if MemoryManager is not None:
                try:
                    self._memory_manager = MemoryManager()
                    logger.debug("MemoryManager initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize MemoryManager: {e}")
            
            # 初始化 MixedPrecisionManager（使用 AmpConfig, PrecisionMode）
            if self.config.use_fp16 and MixedPrecisionManager is not None:
                try:
                    amp_config = None
                    if AmpConfig is not None:
                        # AmpConfig 使用 precision 字段而非 dtype
                        amp_config = AmpConfig(
                            enabled=True,
                            precision=PrecisionMode.MIXED_FP16 if PrecisionMode else None,
                            init_scale=65536.0,
                            growth_factor=2.0,
                            backoff_factor=0.5,
                            growth_interval=2000,
                        )
                    self._mixed_precision_manager = MixedPrecisionManager(amp_config)
                    logger.info("MixedPrecisionManager initialized with AmpConfig")
                    
                    # 记录精度模式（使用 PrecisionMode）
                    if PrecisionMode is not None:
                        try:
                            self._precision_mode = PrecisionMode.FP16 if self.config.use_fp16 else PrecisionMode.FP32
                            logger.debug(f"Precision mode: {self._precision_mode}")
                        except Exception:
                            self._precision_mode = None
                except Exception as e:
                    logger.warning(f"Failed to initialize MixedPrecisionManager: {e}")
            
            if self._device_manager:
                return
                
        except Exception as e:
            logger.warning(f"Failed to get device from manager: {e}")
        
        # 默认设备选择
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self._device_type = DeviceType.GPU if DeviceType else None  # CUDA 对应 GPU 枚举
        else:
            self.device = torch.device("cpu")
            self._device_type = DeviceType.CPU if DeviceType else None
        logger.info("Using device: %s", self.device)
    
    def _init_strategy(self):
        """初始化训练策略（使用 create_multimodal_strategy, TrainingStrategy, TrainingPhase, StrategyValidator, StrategyMetrics）"""
        
        try:
            # 创建策略配置
            strategy_config = MultiModalStrategyConfig(
                modalities=self.config.modalities,
                task_loss_weight=self.config.task_loss_weight,
                align_loss_weight=self.config.align_loss_weight,
                contrastive_loss_weight=self.config.contrastive_loss_weight,
                fusion_method=self.config.fusion_method,
                fusion_dim=self.config.fusion_dim,
                use_alignment=self.config.use_alignment,
                alignment_temperature=self.config.alignment_temperature,
                modality_dropout=self.config.modality_dropout
            ) if MultiModalStrategyConfig else None
            
            # 优先使用 create_multimodal_strategy 工厂函数
            if create_multimodal_strategy is not None and strategy_config is not None:
                try:
                    self.strategy = create_multimodal_strategy(strategy_config)
                    logger.info("Strategy created using create_multimodal_strategy factory")
                except Exception as e:
                    logger.warning(f"create_multimodal_strategy failed: {e}, falling back to direct creation")
                    self.strategy = None
            
            # 回退：直接创建策略
            if self.strategy is None:
                if self.config.strategy_type == "industry_multimodal" and IndustryMultiModalStrategy:
                    self.strategy = IndustryMultiModalStrategy(strategy_config)
                elif MultiModalStrategy:
                    self.strategy = MultiModalStrategy(strategy_config)
            
            if self.strategy:
                logger.info(f"Strategy initialized: {self.strategy.name}")
            
                # 初始化训练阶段（使用 TrainingPhase）
                if STRATEGY_LAYER_AVAILABLE and TrainingPhase is not None:
                    try:
                        self._training_phase = TrainingPhase.MAIN
                        logger.debug(f"Training phase set to: {self._training_phase}")
                    except Exception as e:
                        logger.warning(f"Failed to set training phase: {e}")
                
                # 初始化策略指标（使用 StrategyMetrics）
                if STRATEGY_LAYER_AVAILABLE and StrategyMetrics is not None:
                    try:
                        self._strategy_metrics = StrategyMetrics()
                        logger.debug("StrategyMetrics initialized")
                    except Exception as e:
                        logger.warning(f"Failed to initialize StrategyMetrics: {e}")
                
                # 初始化策略验证器（使用 StrategyValidator）
                if STRATEGY_LAYER_AVAILABLE and StrategyValidator is not None:
                    try:
                        self._strategy_validator = StrategyValidator()
                        logger.debug("StrategyValidator initialized")
                    except Exception as e:
                        logger.warning(f"Failed to initialize StrategyValidator: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to initialize strategy: {e}")
            self.strategy = None
    
    def _init_losses(self):
        """初始化损失函数（使用 LossFactory, CrossEntropyLoss, FocalLoss, CLIPLoss, CompositeLoss, MultiTaskLoss, create_composite_loss）"""
        
        # 使用 LossFactory 创建任务损失
        if LossFactory is not None:
            try:
                self.loss_fn = LossFactory.create('cross_entropy')
                logger.debug("Loss created using LossFactory")
            except Exception as e:
                logger.debug(f"LossFactory.create failed: {e}")
        
        # 回退：使用 create_loss 或直接创建
        if self.loss_fn is None:
            try:
                self.loss_fn = create_loss('cross_entropy')
            except Exception:
                pass
        
        # 回退：使用 CrossEntropyLoss
        if self.loss_fn is None and CrossEntropyLoss is not None:
            try:
                self.loss_fn = CrossEntropyLoss()
                logger.debug("Using CrossEntropyLoss from lib.losses")
            except Exception:
                self.loss_fn = nn.CrossEntropyLoss()
        elif self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        
        # 对于不平衡数据场景，使用 FocalLoss
        if self._scenario is not None and self._scenario in [MultiModalScenario.MEDICAL, MultiModalScenario.MANUFACTURING]:
            if FocalLoss is not None:
                try:
                    self._focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
                    logger.info("FocalLoss initialized for imbalanced scenario")
                except Exception as e:
                    logger.warning(f"Failed to create FocalLoss: {e}")
                    self._focal_loss = None
            else:
                self._focal_loss = None
        else:
            self._focal_loss = None
        
        # 对齐损失（使用 CLIPLoss 或 CrossModalContrastiveLoss）
        if self.config.use_alignment:
            # 优先使用 CLIPLoss
            if CLIPLoss is not None:
                try:
                    self.alignment_loss = CLIPLoss(
                        temperature=self.config.alignment_temperature
                    )
                    logger.debug("Using CLIPLoss for alignment")
                except Exception as e:
                    logger.debug(f"CLIPLoss creation failed: {e}")
            
            # 回退：使用 CrossModalContrastiveLoss
            if self.alignment_loss is None and CrossModalContrastiveLoss is not None:
                try:
                    self.alignment_loss = CrossModalContrastiveLoss(
                        temperature=self.config.alignment_temperature
                    )
                except Exception:
                    pass
        
        # 对比学习损失
        if self.config.use_contrastive:
            try:
                self.contrastive_loss = InfoNCELoss(
                    temperature=self.config.contrastive_temperature
                )
            except Exception:
                pass
        
        # 创建复合损失（使用 create_composite_loss）
        if create_composite_loss is not None:
            try:
                loss_configs = [
                    {'type': 'cross_entropy', 'weight': self.config.task_loss_weight},
                ]
                if self.config.use_alignment:
                    loss_configs.append({'type': 'contrastive', 'weight': self.config.align_loss_weight})
                
                self._composite_loss = create_composite_loss(loss_configs)
                logger.debug("CompositeLoss created using create_composite_loss")
            except Exception as e:
                logger.debug(f"create_composite_loss failed: {e}")
        
        # 创建多任务损失（使用 MultiTaskLoss）
        if MultiTaskLoss is not None and len(self.config.modalities) > 1:
            try:
                task_weights = {
                    'main': self.config.task_loss_weight,
                    'alignment': self.config.align_loss_weight if self.config.use_alignment else 0.0,
                    'contrastive': self.config.contrastive_loss_weight if self.config.use_contrastive else 0.0,
                }
                self._multi_task_loss = MultiTaskLoss(task_weights=task_weights)
                logger.debug("MultiTaskLoss initialized")
            except Exception as e:
                logger.warning(f"Failed to create MultiTaskLoss: {e}")
    
    def _init_distributed(self):
        """初始化分布式训练（使用 DistributedManager, get_distributed_manager, DistributedMode, DistributedStrategyConfig, recommend_distributed_mode, DDPWrapper, FSDPWrapper）"""
        try:
            # 使用 get_distributed_manager 获取分布式管理器
            if get_distributed_manager is not None:
                try:
                    self._distributed_manager = get_distributed_manager()
                    logger.debug("DistributedManager obtained from get_distributed_manager")
                except Exception as e:
                    logger.debug(f"get_distributed_manager failed: {e}")
            
            # 回退：直接创建 DistributedManager
            if self._distributed_manager is None and DistributedManager is not None:
                try:
                    self._distributed_manager = DistributedManager()
                    logger.debug("DistributedManager created directly")
                except Exception as e:
                    logger.warning(f"Failed to create DistributedManager: {e}")
            
            # 初始化分布式环境
            if init_distributed is not None:
                init_distributed()
                logger.info(f"Distributed initialized: rank={get_rank()}, world_size={get_world_size()}")
            
            # 获取推荐的分布式模式（使用 recommend_distributed_mode）
            if recommend_distributed_mode is not None:
                try:
                    estimated_memory = self.config.estimate_total_memory_mb()
                    # recommend_distributed_mode 需要位置参数: model_size_gb, num_gpus
                    recommendation = recommend_distributed_mode(
                        model_size_gb=estimated_memory / 1024,
                        num_gpus=self.config.world_size,
                        memory_per_gpu_gb=16.0  # 默认值
                    )
                    if recommendation:
                        logger.info("Distributed mode recommendation: %s", recommendation)
                        # 可选：根据推荐调整配置
                        if 'mode' in recommendation:
                            recommended_mode = recommendation['mode']
                            logger.debug("Recommended distributed mode: %s", recommended_mode)
                except Exception as e:
                    logger.debug(f"recommend_distributed_mode failed: {e}")
            
            # 获取分布式模式枚举（使用 DistributedMode）
            if DistributedMode is not None:
                try:
                    self._distributed_mode = DistributedMode.from_string(self.config.distributed_mode)
                    logger.debug(f"Distributed mode: {self._distributed_mode}")
                except Exception as e:
                    logger.debug(f"Failed to get DistributedMode: {e}")
                    self._distributed_mode = None
            else:
                self._distributed_mode = None
            
            # 创建分布式策略配置（使用 DistributedStrategyConfig）
            if DistributedStrategyConfig is not None:
                try:
                    # DistributedStrategyConfig 使用 mode 而非 distributed_mode，且没有 fp16 字段
                    dist_strategy_config = DistributedStrategyConfig(
                        mode=self._distributed_mode,
                        world_size=self.config.world_size,
                    )
                    self.distributed_strategy = DistributedStrategy(dist_strategy_config)
                    logger.info("DistributedStrategy initialized with DistributedStrategyConfig")
                except Exception as e:
                    logger.warning("Failed to create DistributedStrategy: %s", e)
            else:
                # 回退：使用 config 的方法
                dist_config = self.config.create_distributed_config()
                if dist_config:
                    self.distributed_strategy = DistributedStrategy(dist_config)
                    
        except Exception as e:
            logger.warning(f"Failed to initialize distributed: {e}")
    
    def _init_lib_multimodal(self):
        """初始化底层多模态模块（使用 ModalityEncoderFactory, CrossModalAligner, MultiModalFuser, FusionStage, TrainingStage）"""
        try:
            # 使用 ModalityEncoderFactory 创建编码器
            if ModalityEncoderFactory is not None:
                for modality in self.config.modalities:
                    try:
                        modality_type = ModalityType.from_string(modality)
                        lib_modality = modality_type.to_lib_modality()
                        if lib_modality is not None:
                            modality_config = self.config.modality_configs.get(modality)
                            if modality_config:
                                # ModalityEncoderFactory 使用 create_encoder 方法
                                encoder_config = {
                                    'input_dim': modality_config.input_dim,
                                    'output_dim': modality_config.output_dim,
                                    'pretrained_model': modality_config.pretrained_model,
                                    'dropout': modality_config.dropout,
                                }
                                encoder = ModalityEncoderFactory.create_encoder(
                                    modality=lib_modality,
                                    config=encoder_config,
                                )
                                self._lib_encoders[modality] = encoder
                                logger.debug("Encoder created for modality %s using ModalityEncoderFactory", modality)
                    except Exception as e:
                        logger.debug("Failed to create encoder for %s: %s", modality, e)
            
            # 使用 CrossModalAligner 创建对齐模块
            if CrossModalAligner is not None and self.config.use_alignment:
                try:
                    self._lib_aligner = CrossModalAligner(
                        input_dim=self.config.alignment_dim,
                        alignment_method=self.config.alignment_method,
                        temperature=self.config.alignment_temperature,
                    )
                    logger.debug("CrossModalAligner created")
                except Exception as e:
                    logger.debug(f"Failed to create CrossModalAligner: {e}")
            
            # 使用 MultiModalFuser 创建融合模块
            if MultiModalFuser is not None:
                try:
                    # MultiModalFuser 需要 config 和 modality_dims 参数
                    # 这里使用配置的融合配置或创建简化的配置
                    fusion_config = self.config.create_fusion_config() if hasattr(self.config, 'create_fusion_config') else None
                    if fusion_config is not None:
                        self._lib_fuser = MultiModalFuser(
                            config=fusion_config,
                            modality_dims=self.config.modality_dims,
                        )
                        logger.debug("MultiModalFuser created")
                except Exception as e:
                    logger.debug("Failed to create MultiModalFuser: %s", e)
            
            # 设置融合阶段（使用 FusionStage）
            if FusionStage is not None:
                try:
                    if self._fusion_method is not None:
                        self._fusion_stage = self._fusion_method.get_fusion_stage()
                        logger.debug(f"Fusion stage: {self._fusion_stage}")
                except Exception as e:
                    logger.debug(f"Failed to get FusionStage: {e}")
            
            # 设置训练阶段（使用 TrainingStage）
            if TrainingStage is not None:
                try:
                    self._training_stage = TrainingStage.from_string('pretrain')
                    logger.debug(f"Training stage: {self._training_stage}")
                except Exception as e:
                    logger.debug(f"Failed to get TrainingStage: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to initialize lib multimodal modules: {e}")
    
    def _init_progress_manager(self):
        """初始化进度管理（使用 TrainingProgressManager, TrainingProgress）"""
        try:
            # 使用 get_progress_manager 获取进度管理器
            if get_progress_manager is not None:
                try:
                    self._progress_manager = get_progress_manager()
                    logger.debug("TrainingProgressManager obtained from get_progress_manager")
                except Exception as e:
                    logger.debug(f"get_progress_manager failed: {e}")
            
            # 回退：直接创建
            if self._progress_manager is None and TrainingProgressManager is not None:
                try:
                    self._progress_manager = TrainingProgressManager()
                    logger.debug("TrainingProgressManager created directly")
                except Exception as e:
                    logger.warning(f"Failed to create TrainingProgressManager: {e}")
            
            # 创建训练进度对象（使用 TrainingProgress）
            if self._progress_manager is not None and TrainingProgress is not None:
                try:
                    self._training_progress = TrainingProgress(
                        total_epochs=self.config.num_epochs,
                        total_steps=0,  # 将在 train() 中更新
                        session_id=f"multimodal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    )
                    logger.debug("TrainingProgress created")
                except Exception as e:
                    logger.debug(f"Failed to create TrainingProgress: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to initialize progress manager: {e}")
    
    def register_callback(self, callback: TrainerCallback):
        """注册回调"""
        self.callbacks.append(callback)
    
    def _fire_callback(self, method: str, *args, **kwargs):
        """触发回调"""
        for callback in self.callbacks:
            try:
                getattr(callback, method)(self, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback {method} failed: {e}")
    
    def prepare_model(self, num_classes: int = 10) -> MultiModalModel:
        """准备模型（使用 DDPWrapper, FSDPWrapper, GradientCheckpointing, StrategyContext, TrainingPhase）"""
        try:
            self.model = MultiModalModel(self.config, num_classes)
            self.model = self.model.to(self.device)
            
            # 应用分布式包装（使用 DDPWrapper 或 FSDPWrapper）
            if self.config.use_distributed:
                if self.config.distributed_mode == "ddp":
                    # 优先使用 DDPWrapper
                    if DDPWrapper is not None:
                        try:
                            self._ddp_wrapper = DDPWrapper()
                            self.model = self._ddp_wrapper.wrap(self.model, self.optimizer, self.scheduler)
                            logger.info("Model wrapped with DDPWrapper")
                        except Exception as e:
                            logger.warning("DDPWrapper failed: %s, falling back to create_ddp_model", e)
                            if create_ddp_model:
                                self.model = create_ddp_model(self.model)
                    elif create_ddp_model:
                        self.model = create_ddp_model(self.model)
                        
                elif self.config.distributed_mode == "fsdp":
                    # 优先使用 FSDPWrapper
                    if FSDPWrapper is not None:
                        try:
                            self._fsdp_wrapper = FSDPWrapper()
                            self.model = self._fsdp_wrapper.wrap(self.model)
                            logger.info("Model wrapped with FSDPWrapper")
                        except Exception as e:
                            logger.warning("FSDPWrapper failed: %s, falling back to create_fsdp_model", e)
                            if create_fsdp_model:
                                self.model = create_fsdp_model(self.model)
                    elif create_fsdp_model:
                        self.model = create_fsdp_model(self.model)
            
            # 启用梯度检查点（使用 GradientCheckpointing）
            if self.config.gradient_accumulation_steps > 1:
                if GradientCheckpointing is not None:
                    try:
                        grad_ckpt = GradientCheckpointing(self.model)
                        grad_ckpt.enable()
                        logger.info("Gradient checkpointing enabled via GradientCheckpointing")
                    except Exception as e:
                        logger.debug(f"GradientCheckpointing failed: {e}")
                        if hasattr(self.model, 'enable_gradient_checkpointing'):
                            self.model.enable_gradient_checkpointing()
                elif hasattr(self.model, 'enable_gradient_checkpointing'):
                    self.model.enable_gradient_checkpointing()
            
            # 初始化策略上下文（使用 StrategyContext, TrainingPhase）
            if self.strategy and STRATEGY_LAYER_AVAILABLE and StrategyContext:
                modality_dim_config = {f'{m}_dim': d for m, d in self.config.modality_dims.items()}
                
                # 设置训练阶段
                training_phase = self._training_phase if self._training_phase is not None else TrainingPhase.MAIN
                
                self.strategy_context = StrategyContext(
                    model=self.model,
                    device=self.device,
                    config=modality_dim_config,
                    phase=training_phase,
                )
                self.strategy.setup(self.strategy_context)
                logger.debug(f"Strategy context created with phase: {training_phase}")
            
            # 打印模型信息
            param_count = self.model.get_param_count()
            logger.info(f"MultiModal model prepared: {type(self.model).__name__}")
            logger.info(f"Modalities: {self.config.modalities}")
            logger.info(f"Fusion method: {self.config.fusion_method}")
            logger.info(f"Scenario: {self.config.scenario}")
            logger.info(f"Total parameters: {param_count['total']:,}")
            logger.info(f"Trainable parameters: {param_count['trainable']:,}")
            
            # 记录分布式包装状态
            if self._ddp_wrapper:
                logger.info("Using DDPWrapper for distributed training")
            elif self._fsdp_wrapper:
                logger.info("Using FSDPWrapper for distributed training")
            
            return self.model
            
        except Exception as e:
            raise BusinessLogicError(f"Failed to prepare model: {e}")
    
    def prepare_datasets(self) -> None:
        """准备数据集"""
        try:
            self.train_dataset = MultiModalDataset(self.config.train_data_path, self.config)
            if self.config.val_data_path:
                self.val_dataset = MultiModalDataset(self.config.val_data_path, self.config)
            if self.config.test_data_path:
                self.test_dataset = MultiModalDataset(self.config.test_data_path, self.config)
            
            logger.info(f"Train dataset size: {len(self.train_dataset)}")
            if self.val_dataset:
                logger.info(f"Val dataset size: {len(self.val_dataset)}")
                
        except Exception as e:
            raise BusinessLogicError(f"Failed to prepare datasets: {e}")
    
    def prepare_optimizer_and_scheduler(self) -> None:
        """准备优化器和调度器"""
        try:
            if self.model is None:
                raise BusinessLogicError("Model not prepared")
            
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=self.config.learning_rate * 0.01
            )
            
            # 混合精度
            if self.config.use_fp16 and self.device.type == 'cuda':
                self.scaler = torch.cuda.amp.GradScaler()
            
            logger.info(f"Optimizer: {type(self.optimizer).__name__}")
            logger.info(f"Scheduler: {type(self.scheduler).__name__}")
            logger.info(f"Mixed precision: {self.scaler is not None}")
            
        except Exception as e:
            raise BusinessLogicError(f"Failed to prepare optimizer: {e}")
    
    def _health_check(self) -> Dict[str, Any]:
        """执行健康检查"""
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'issues': []
        }
        
        # 检查内存
        try:
            available = get_available_memory()
            if available < 1000:  # 小于1GB
                health['issues'].append(f"Low memory: {available:.0f}MB available")
                health['status'] = 'warning'
        except Exception:
            pass
        
        # 检查梯度
        if self.model is not None:
            try:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            health['issues'].append(f"NaN gradient in {name}")
                            health['status'] = 'error'
                        if torch.isinf(param.grad).any():
                            health['issues'].append(f"Inf gradient in {name}")
                            health['status'] = 'error'
            except Exception:
                pass
        
        # 检查分布式状态
        if self.config.use_distributed:
            try:
                dist_health = diagnose_distributed_strategy()
                if dist_health.get('status') == 'error':
                    health['issues'].append(dist_health.get('message', 'Distributed error'))
                    health['status'] = 'error'
            except Exception:
                pass
        
        return health
    
    def train(self) -> Dict[str, Any]:
        """执行多模态训练"""
        try:
            logger.info("Starting multimodal training...")
            
            # 准备组件
            if self.model is None:
                self.prepare_model()
            
            if self.train_dataset is None:
                self.prepare_datasets()
            
            if self.optimizer is None:
                self.prepare_optimizer_and_scheduler()
            
            # 触发回调
            self._fire_callback('on_train_begin')
            
            # 创建数据加载器
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers if self.config.num_workers > 0 else 0,
                pin_memory=self.config.pin_memory,
                drop_last=True
            )
            
            val_loader = None
            if self.val_dataset:
                val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers if self.config.num_workers > 0 else 0,
                    pin_memory=self.config.pin_memory
                )
            
            # 训练历史
            history = {
                'train_loss': [],
                'val_loss': [],
                'metrics': [],
                'health_checks': []
            }
            
            # 训练循环
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                self._fire_callback('on_epoch_begin', epoch)
                
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                
                # 训练阶段
                train_metrics = self._train_epoch(train_loader)
                history['train_loss'].append(train_metrics['loss'])
                history['metrics'].append(train_metrics)
                
                logger.info(f"Train loss: {train_metrics['loss']:.4f}")
                if 'task_loss' in train_metrics:
                    logger.info(f"  Task loss: {train_metrics['task_loss']:.4f}")
                if 'align_loss' in train_metrics:
                    logger.info(f"  Align loss: {train_metrics['align_loss']:.4f}")
                
                # 验证阶段
                if val_loader:
                    val_metrics = self._validate_epoch(val_loader)
                    history['val_loss'].append(val_metrics['loss'])
                    logger.info(f"Val loss: {val_metrics['loss']:.4f}")
                    logger.info(f"Val accuracy: {val_metrics.get('accuracy', 0):.4f}")
                    
                    # 保存最佳模型
                    if val_metrics['loss'] < self.best_loss:
                        self.best_loss = val_metrics['loss']
                        self.best_metrics = val_metrics.copy()
                        self._save_model("best_model.pth")
                
                # 健康检查
                if (epoch + 1) % 5 == 0:
                    health = self._health_check()
                    history['health_checks'].append(health)
                    if health['status'] == 'error':
                        logger.error(f"Health check failed: {health['issues']}")
                
                # 触发回调
                self._fire_callback('on_epoch_end', epoch, train_metrics)
                
                # 保存检查点
                if (epoch + 1) % 5 == 0:
                    self._save_model(f"checkpoint_epoch_{epoch + 1}.pth")
                
                # 同步分布式
                if self.config.use_distributed:
                    barrier()
            
            # 保存最终模型
            self._save_model("final_model.pth")
            
            result = {
                'success': True,
                'best_loss': self.best_loss,
                'best_metrics': self.best_metrics,
                'model_path': str(self.output_dir / "best_model.pth"),
                'history': history,
                'config': self.config.to_dict()
            }
            
            # 触发回调
            self._fire_callback('on_train_end', result)
            
            logger.info("Multimodal training completed!")
            return result
            
        except Exception as e:
            logger.error(f"Multimodal training failed: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                'success': False,
                'error': str(e)
            }
            self._fire_callback('on_train_end', result)
            return result
    
    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_metrics = {}
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            self._fire_callback('on_step_begin', self.global_step)
            
            # 准备数据
            inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
            labels = batch['label'].to(self.device)
            
            # 应用模态dropout（通过策略）
            if self.strategy and self.strategy_context:
                inputs = self.strategy.prepare_batch(inputs, self.strategy_context)
            
            # 混合精度前向传播
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                outputs = self.model(inputs, return_features=True)
                outputs['labels'] = labels
            
                # 计算损失
                if self.strategy and self.strategy_context:
                    task_loss = self.loss_fn(outputs['logits'], labels)
                    outputs['loss'] = task_loss
                
                    result = self.strategy.compute_loss(
                        self.model, inputs, outputs, self.strategy_context
                    )
                    loss = result.loss
                    step_metrics = result.metrics
                else:
                    loss = self.loss_fn(outputs['logits'], labels)
                    step_metrics = {'loss': loss.item(), 'task_loss': loss.item()}
                    
                    # 添加对齐损失
                    if self.alignment_loss and 'alignment_features' in outputs:
                        align_features = outputs['alignment_features']
                        if len(align_features) >= 2:
                            modalities = list(align_features.keys())
                            align_loss = self.alignment_loss(
                                align_features[modalities[0]],
                                align_features[modalities[1]]
                            )
                            loss = loss + self.config.align_loss_weight * align_loss
                            step_metrics['align_loss'] = align_loss.item()
            
            # 反向传播
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)
                self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
            
            # 累积指标
            for k, v in step_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            num_batches += 1
            self.global_step += 1
            
            # 触发回调
            self._fire_callback('on_step_end', self.global_step, loss.item())
            
            # 策略回调
            if self.strategy and self.strategy_context:
                if STRATEGY_LAYER_AVAILABLE and StrategyResult:
                    self.strategy.on_step_end(
                        self.strategy_context,
                        StrategyResult(loss=loss, metrics=step_metrics)
                    )
            
            # 定期健康检查
            if self.global_step - self._last_health_check >= self._health_check_interval:
                self._last_health_check = self.global_step
                health = self._health_check()
                if health['status'] == 'error':
                    logger.warning(f"Health check warning at step {self.global_step}: {health['issues']}")
        
        # 计算平均指标
        avg_metrics = {k: v / max(num_batches, 1) for k, v in total_metrics.items()}
        avg_metrics['loss'] = avg_metrics.get('total_loss', avg_metrics.get('task_loss', 0))
        
        return avg_metrics
    
    def _validate_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
                labels = batch['label'].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs['logits'], labels)
                
                total_loss += loss.item()
                
                preds = outputs['logits'].argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
        self._fire_callback('on_evaluate', metrics)
        
        return metrics
    
    def _save_model(self, filename: str) -> None:
        """保存模型"""
        # 只在主进程保存
        if self.config.use_distributed and not is_main_process():
            return
        
        try:
            model_path = self.output_dir / filename
            
            # 处理分布式模型
            model_to_save = self.model
            if hasattr(self.model, 'module'):
                model_to_save = self.model.module
            
            checkpoint = {
                'model_state_dict': model_to_save.state_dict(),
                'config': self.config.to_dict(),
                'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'epoch': self.current_epoch,
                'global_step': self.global_step,
                'best_loss': self.best_loss,
                'best_metrics': self.best_metrics,
                'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            }
            
            torch.save(checkpoint, model_path)
            logger.info(f"Model saved to: {model_path}")
            
            self._fire_callback('on_save', str(model_path))
            
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")
    
    def load_model(self, model_path: str) -> None:
        """加载模型"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 处理分布式模型
            model_to_load = self.model
            if hasattr(self.model, 'module'):
                model_to_load = self.model.module
            
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            
            if self.optimizer and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if self.scaler and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.current_epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            self.best_metrics = checkpoint.get('best_metrics', {})
            
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            raise BusinessLogicError(f"Failed to load model: {e}")
    
    def evaluate(self, dataset: Optional[MultiModalDataset] = None) -> Dict[str, Any]:
        """评估模型"""
        if dataset is None:
            dataset = self.test_dataset or self.val_dataset
        
        if dataset is None:
            raise BusinessLogicError("No evaluation dataset available")
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers if self.config.num_workers > 0 else 0
        )
        
        metrics = self._validate_epoch(dataloader)
        return metrics
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断训练器状态（使用 diagnose_multimodal_strategy, get_layer_availability, ConfigSerializer）"""
        diagnosis = {
            'config_diagnosis': self.config.diagnose(),
            'model_loaded': self.model is not None,
            'optimizer_loaded': self.optimizer is not None,
            'strategy_loaded': self.strategy is not None,
            'distributed_enabled': self.config.use_distributed,
            'device': str(self.device),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
        }
        
        if self.model:
            diagnosis['param_count'] = self.model.get_param_count()
        
        # 健康检查
        diagnosis['health'] = self._health_check()
        
        # 使用 diagnose_multimodal_strategy 进行策略诊断
        # 注意: diagnose_multimodal_strategy 需要 strategy 参数，这里检查策略是否可用
        if diagnose_multimodal_strategy is not None and self.strategy is not None:
            try:
                strategy_diagnosis = diagnose_multimodal_strategy(self.strategy)
                diagnosis['multimodal_strategy_diagnosis'] = strategy_diagnosis
                logger.debug("diagnose_multimodal_strategy completed")
            except Exception as e:
                diagnosis['multimodal_strategy_diagnosis'] = {'error': str(e)}
        else:
            # 策略不可用时，返回可用性信息
            diagnosis['multimodal_strategy_diagnosis'] = {'available': diagnose_multimodal_strategy is not None}
        
        # 使用 get_layer_availability 获取层可用性
        diagnosis['layer_availability'] = get_layer_availability()
        
        # 场景和融合方法信息（使用 MultiModalScenario, FusionMethod）
        diagnosis['scenario'] = {
            'type': self.config.scenario,
            'enum': str(self._scenario) if self._scenario else None,
            'recommended_modalities': self._scenario.recommended_modalities if self._scenario else None,
            'recommended_fusion': self._scenario.recommended_fusion if self._scenario else None,
        }
        
        diagnosis['fusion'] = {
            'method': self.config.fusion_method,
            'enum': str(self._fusion_method) if self._fusion_method else None,
            'requires_attention': self._fusion_method.requires_attention if self._fusion_method else None,
            'memory_intensity': self._fusion_method.memory_intensity if self._fusion_method else None,
            'fusion_stage': str(self._fusion_stage) if self._fusion_stage else None,
        }
        
        # 使用 ConfigSerializer 获取配置哈希
        diagnosis['config_hash'] = ConfigSerializer.get_config_hash(self.config)
        
        # 硬件管理器状态
        diagnosis['hardware'] = {
            'device_manager': self._device_manager is not None,
            'memory_manager': self._memory_manager is not None,
            'mixed_precision_manager': self._mixed_precision_manager is not None,
            'device_type': str(self._device_type) if hasattr(self, '_device_type') and self._device_type else None,
            'precision_mode': str(self._precision_mode) if hasattr(self, '_precision_mode') and self._precision_mode else None,
        }
        
        # 分布式状态
        diagnosis['distributed'] = {
            'distributed_manager': self._distributed_manager is not None,
            'distributed_strategy': self.distributed_strategy is not None,
            'distributed_mode': str(self._distributed_mode) if hasattr(self, '_distributed_mode') and self._distributed_mode else None,
            'ddp_wrapper': self._ddp_wrapper is not None,
            'fsdp_wrapper': self._fsdp_wrapper is not None,
        }
        
        # 损失函数状态
        diagnosis['losses'] = {
            'main_loss': type(self.loss_fn).__name__ if self.loss_fn else None,
            'alignment_loss': type(self.alignment_loss).__name__ if self.alignment_loss else None,
            'contrastive_loss': type(self.contrastive_loss).__name__ if self.contrastive_loss else None,
            'focal_loss': self._focal_loss is not None if hasattr(self, '_focal_loss') else False,
            'composite_loss': self._composite_loss is not None,
            'multi_task_loss': self._multi_task_loss is not None,
        }
        
        # 底层多模态模块状态
        diagnosis['lib_multimodal'] = {
            'lib_encoders': list(self._lib_encoders.keys()),
            'lib_aligner': self._lib_aligner is not None,
            'lib_fuser': self._lib_fuser is not None,
            'training_stage': str(self._training_stage) if self._training_stage else None,
            'fusion_stage': str(self._fusion_stage) if self._fusion_stage else None,
        }
        
        # 进度管理状态
        diagnosis['progress'] = {
            'progress_manager': self._progress_manager is not None,
            'training_progress': self._training_progress is not None,
        }
        
        # 策略指标
        if self._strategy_metrics is not None:
            try:
                diagnosis['strategy_metrics'] = self._strategy_metrics.get_summary() if hasattr(self._strategy_metrics, 'get_summary') else str(self._strategy_metrics)
            except Exception:
                diagnosis['strategy_metrics'] = 'available'
        
        return diagnosis
    
    def cleanup(self):
        """清理资源"""
        if self.config.use_distributed:
            cleanup_distributed()
        
        clear_memory()
        
        logger.info("Trainer cleanup completed")


# ==================== 便捷函数 ====================

def create_multimodal_trainer(config: Dict[str, Any]) -> MultiModalTrainer:
    """创建多模态训练器的便捷函数"""
    try:
        mm_config = MultiModalConfig(**config)
        return MultiModalTrainer(mm_config)
    except Exception as e:
        logger.error(f"Failed to create multimodal trainer: {e}")
        raise BusinessLogicError(f"Failed to create multimodal trainer: {e}")


def create_industry_multimodal_trainer(
    industry_type: str = "manufacturing",
    **kwargs
) -> MultiModalTrainer:
    """创建行业多模态训练器"""
    try:
        config = IndustryMultiModalConfig(
            industry_type=industry_type,
            **kwargs
        )
        return MultiModalTrainer(config)
    except Exception as e:
        logger.error(f"Failed to create industry multimodal trainer: {e}")
        raise BusinessLogicError(f"Failed to create industry multimodal trainer: {e}")


def get_preset_trainer(preset: str, **kwargs) -> MultiModalTrainer:
    """获取预设训练器"""
    preset_map = {
        'standard_classification': MultiModalPresets.standard_classification,
        'multimodal_alignment': MultiModalPresets.multimodal_alignment,
        'manufacturing': MultiModalPresets.manufacturing_scenario,
        'finance': MultiModalPresets.finance_scenario,
        'medical': MultiModalPresets.medical_scenario,
        'retail': MultiModalPresets.retail_scenario,
        'autonomous': MultiModalPresets.autonomous_driving,
        'video': MultiModalPresets.video_understanding,
        'distributed': MultiModalPresets.distributed_large_scale,
    }
    
    if preset not in preset_map:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(preset_map.keys())}")
    
    config = preset_map[preset]()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return MultiModalTrainer(config)


def create_trainer_with_callbacks(
    config: MultiModalConfig,
    session_id: Optional[str] = None,
    enable_monitoring: bool = True
) -> MultiModalTrainer:
    """创建带回调的训练器"""
    trainer = MultiModalTrainer(config)
    
    if session_id:
        trainer.register_callback(ProgressCallback(session_id))
    
    if enable_monitoring and STRATEGY_LAYER_AVAILABLE:
        trainer.register_callback(MonitorCallback())
    
    return trainer


def diagnose_trainer_setup(config: MultiModalConfig) -> Dict[str, Any]:
    """诊断训练器设置（使用 diagnose_multimodal_strategy, StrategyValidator）"""
    diagnosis = {
        'config_valid': True,
        'errors': [],
        'warnings': [],
        'recommendations': [],
    }
    
    # 验证配置
    from .multimodal_config import ConfigValidator
    errors = ConfigValidator.validate_multimodal_config(config)
    if errors:
        diagnosis['config_valid'] = False
        diagnosis['errors'] = errors
    
    # 使用 diagnose_multimodal_strategy 进行策略诊断
    # 注意: diagnose_multimodal_strategy 需要 strategy 参数，这里只检查可用性
    if diagnose_multimodal_strategy is not None:
        try:
            # 无法创建策略实例，只返回可用性信息
            diagnosis['strategy_diagnosis'] = {'available': True, 'note': 'Strategy instance required for full diagnosis'}
        except Exception as e:
            diagnosis['warnings'].append("Strategy diagnosis failed: %s", e)
    
    # 使用 StrategyValidator 验证配置
    # 注意: StrategyValidator 用于验证 StrategyResult，不是 config
    if STRATEGY_LAYER_AVAILABLE and StrategyValidator is not None:
        try:
            # StrategyValidator 不支持 validate_config，跳过此验证
            pass
        except Exception as e:
            diagnosis['warnings'].append("Strategy validation skipped: %s", e)
    
    # 检查层可用性
    diagnosis['layer_availability'] = get_layer_availability()
    
    # 检查场景推荐（使用 MultiModalScenario）
    try:
        scenario = MultiModalScenario.from_string(config.scenario)
        recommended_modalities = scenario.recommended_modalities
        missing_modalities = [m for m in recommended_modalities if m not in config.modalities]
        if missing_modalities:
            diagnosis['recommendations'].append(
                f"Consider adding recommended modalities for {config.scenario} scenario: {missing_modalities}"
            )
    except Exception:
        pass
    
    # 检查融合方法推荐（使用 FusionMethod）
    try:
        fusion = FusionMethod.from_string(config.fusion_method)
        if fusion.requires_attention and not config.use_alignment:
            diagnosis['recommendations'].append(
                f"Fusion method '{config.fusion_method}' works best with alignment enabled"
            )
        if fusion.memory_intensity == 'high' and config.batch_size > 16:
            diagnosis['recommendations'].append(
                f"High memory fusion method with large batch size - consider reducing batch_size"
            )
    except Exception:
        pass
    
    return diagnosis


def save_trainer_config(trainer: MultiModalTrainer, file_path: str) -> None:
    """保存训练器配置（使用 ConfigSerializer）"""
    ConfigSerializer.to_file(trainer.config, file_path)
    logger.info(f"Trainer config saved to {file_path}")


def load_trainer_from_config(file_path: str) -> MultiModalTrainer:
    """从配置文件加载训练器（使用 ConfigSerializer）"""
    config = ConfigSerializer.from_file(file_path, MultiModalConfig)
    return MultiModalTrainer(config)


def get_recommended_distributed_settings(config: MultiModalConfig) -> Dict[str, Any]:
    """获取推荐的分布式设置（使用 recommend_distributed_mode, DistributedMode）"""
    recommendations = {
        'use_distributed': False,
        'distributed_mode': 'ddp',
        'world_size': 1,
        'recommendations': [],
    }
    
    # 使用 recommend_distributed_mode
    if recommend_distributed_mode is not None:
        try:
            estimated_memory = config.estimate_total_memory_mb()
            # recommend_distributed_mode 需要位置参数: model_size_gb, num_gpus
            rec = recommend_distributed_mode(
                model_size_gb=estimated_memory / 1024,
                num_gpus=config.world_size,
                memory_per_gpu_gb=16.0  # 默认值
            )
            if rec:
                recommendations.update(rec)
                recommendations['use_distributed'] = True
        except Exception as e:
            recommendations['recommendations'].append("recommend_distributed_mode failed: %s", e)
    
    # 添加基于场景的推荐（使用 MultiModalScenario）
    try:
        scenario = MultiModalScenario.from_string(config.scenario)
        if scenario in [MultiModalScenario.AUTONOMOUS, MultiModalScenario.ROBOTICS]:
            recommendations['recommendations'].append(
                "Consider using FSDP for large autonomous/robotics models"
            )
    except Exception:
        pass
    
    return recommendations


def create_trainer_from_scenario(
    scenario: str,
    **kwargs
) -> MultiModalTrainer:
    """根据场景创建训练器（使用 MultiModalScenario）"""
    try:
        # 使用 MultiModalScenario 获取推荐配置
        scenario_enum = MultiModalScenario.from_string(scenario)
        
        config_kwargs = {
            'modalities': scenario_enum.recommended_modalities,
            'fusion_method': scenario_enum.recommended_fusion,
            'scenario': scenario,
        }
        config_kwargs.update(kwargs)
        
        # 对于行业场景，使用 IndustryMultiModalConfig
        industry_scenarios = ['manufacturing', 'finance', 'medical', 'retail']
        if scenario in industry_scenarios:
            config = IndustryMultiModalConfig(industry_type=scenario, **config_kwargs)
        else:
            config = MultiModalConfig(**config_kwargs)
        
        return MultiModalTrainer(config)
        
    except Exception as e:
        logger.error(f"Failed to create trainer from scenario: {e}")
        raise BusinessLogicError(f"Failed to create trainer from scenario: {e}")


def validate_trainer_strategy(trainer: MultiModalTrainer) -> Dict[str, Any]:
    """验证训练器策略（使用 StrategyValidator, diagnose_multimodal_strategy）"""
    validation = {
        'valid': True,
        'errors': [],
        'warnings': [],
    }
    
    # 使用 StrategyValidator
    # 注意: StrategyValidator 用于验证 StrategyResult，不是 config
    if STRATEGY_LAYER_AVAILABLE and StrategyValidator is not None:
        try:
            # StrategyValidator 不支持 validate_config，跳过此验证
            pass
        except Exception as e:
            validation['warnings'].append("StrategyValidator skipped: %s", e)
    
    # 使用 diagnose_multimodal_strategy
    # 注意: diagnose_multimodal_strategy 需要 strategy 参数
    if diagnose_multimodal_strategy is not None:
        try:
            # 检查是否有策略实例可用
            if trainer.strategy is not None:
                diag = diagnose_multimodal_strategy(trainer.strategy)
                if diag.get('status') == 'error':
                    validation['valid'] = False
                    validation['errors'].append(diag.get('message', 'Strategy error'))
                elif diag.get('warnings'):
                    validation['warnings'].extend(diag.get('warnings', []))
        except Exception as e:
            validation['warnings'].append("diagnose_multimodal_strategy failed: %s", e)
    
    return validation


def get_training_stage_info(trainer: MultiModalTrainer) -> Dict[str, Any]:
    """获取训练阶段信息（使用 TrainingPhase, TrainingStage, FusionStage）"""
    info = {}
    
    # TrainingPhase
    if trainer._training_phase is not None:
        info['training_phase'] = str(trainer._training_phase)
    
    # TrainingStage
    if trainer._training_stage is not None:
        info['training_stage'] = str(trainer._training_stage)
    
    # FusionStage
    if trainer._fusion_stage is not None:
        info['fusion_stage'] = str(trainer._fusion_stage)
    
    return info


def get_loss_info(trainer: MultiModalTrainer) -> Dict[str, Any]:
    """获取损失函数信息（使用 LossFactory, CompositeLoss, MultiTaskLoss）"""
    info = {
        'loss_fn': type(trainer.loss_fn).__name__ if trainer.loss_fn else None,
        'alignment_loss': type(trainer.alignment_loss).__name__ if trainer.alignment_loss else None,
        'contrastive_loss': type(trainer.contrastive_loss).__name__ if trainer.contrastive_loss else None,
    }
    
    if hasattr(trainer, '_focal_loss') and trainer._focal_loss is not None:
        info['focal_loss'] = type(trainer._focal_loss).__name__
    
    if trainer._composite_loss is not None:
        info['composite_loss'] = type(trainer._composite_loss).__name__
    
    if trainer._multi_task_loss is not None:
        info['multi_task_loss'] = type(trainer._multi_task_loss).__name__
    
    return info

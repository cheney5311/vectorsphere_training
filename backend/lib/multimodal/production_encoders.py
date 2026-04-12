# -*- coding: utf-8 -*-
"""
生产级模态编码器

提供各模态专属编码器的实现，包括：
- 文本编码器（BERT/RoBERTa/LLaMA）
- 图像编码器（ViT/ResNet/CLIP）
- 音频编码器（Whisper/Wav2Vec）
- 时序编码器（LSTM/Transformer）
- 视频编码器（VideoSwin/TimeSformer）
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .production_multimodal_config import (
    ModalityType, EncoderType,
    TextEncoderConfig, ImageEncoderConfig, AudioEncoderConfig,
    TimeSeriesEncoderConfig, VideoEncoderConfig, ModalEncodersConfig
)

logger = logging.getLogger(__name__)


# ==================== 基础编码器 ====================

class BaseModalityEncoder(nn.Module, ABC):
    """模态编码器基类"""
    
    def __init__(self, modality: ModalityType, hidden_size: int):
        super().__init__()
        self.modality = modality
        self.hidden_size = hidden_size
        self._is_frozen = False
    
    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播，返回编码后的特征"""
        raise NotImplementedError
    
    def freeze(self) -> None:
        """冻结编码器参数"""
        for param in self.parameters():
            param.requires_grad = False
        self._is_frozen = True
        logger.info(f"{self.modality.value} encoder frozen")
    
    def unfreeze(self) -> None:
        """解冻编码器参数"""
        for param in self.parameters():
            param.requires_grad = True
        self._is_frozen = False
        logger.info(f"{self.modality.value} encoder unfrozen")
    
    @property
    def is_frozen(self) -> bool:
        return self._is_frozen
    
    def get_output_dim(self) -> int:
        return self.hidden_size


# ==================== 文本编码器 ====================

class ProductionTextEncoder(BaseModalityEncoder):
    """生产级文本编码器
    
    支持多种预训练模型：BERT、RoBERTa、LLaMA、Qwen等
    """
    
    def __init__(self, config: TextEncoderConfig):
        super().__init__(ModalityType.TEXT, config.hidden_size)
        self.config = config
        
        # 构建编码器
        self.encoder = self._build_encoder()
        self.pooler = self._build_pooler()
        
        # 可选的Adapter层
        if config.use_adapter:
            self.adapter = AdapterModule(
                config.hidden_size, 
                config.adapter_size
            )
        else:
            self.adapter = None
    
    def _build_encoder(self) -> nn.Module:
        """构建文本编码器"""
        try:
            from transformers import AutoModel, AutoConfig
            
            model_config = AutoConfig.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            encoder = AutoModel.from_pretrained(
                self.config.model_name,
                config=model_config,
                trust_remote_code=True
            )
            return encoder
        except Exception as e:
            logger.warning(f"Failed to load pretrained model: {e}, using fallback")
            return self._build_fallback_encoder()
    
    def _build_fallback_encoder(self) -> nn.Module:
        """构建备用编码器"""
        return nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.config.hidden_size,
                nhead=12,
                dim_feedforward=3072,
                dropout=0.1
            ),
            num_layers=12
        )
    
    def _build_pooler(self) -> nn.Module:
        """构建池化层"""
        if self.config.pooling == "cls":
            return nn.Identity()
        elif self.config.pooling == "mean":
            return MeanPooler()
        elif self.config.pooling == "max":
            return MaxPooler()
        else:
            return MeanPooler()
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            inputs: 包含 input_ids, attention_mask 的字典
            
        Returns:
            编码后的文本特征 [batch, hidden_size]
        """
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        
        # 编码
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 获取隐藏状态
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]
        
        # 池化
        if self.config.pooling == "cls":
            pooled = hidden_states[:, 0, :]
        else:
            pooled = self.pooler(hidden_states, attention_mask)
        
        # Adapter
        if self.adapter is not None:
            pooled = self.adapter(pooled)
        
        return pooled


# ==================== 图像编码器 ====================

class ProductionImageEncoder(BaseModalityEncoder):
    """生产级图像编码器
    
    支持ViT、ResNet、CLIP Vision等架构
    """
    
    def __init__(self, config: ImageEncoderConfig):
        super().__init__(ModalityType.IMAGE, config.hidden_size)
        self.config = config
        
        # 构建编码器
        self.encoder = self._build_encoder()
        
        # 可选的Adapter层
        if config.use_adapter:
            self.adapter = AdapterModule(
                config.hidden_size,
                config.adapter_size
            )
        else:
            self.adapter = None
        
        # 特征投影
        self.projection = nn.Linear(
            self._get_encoder_output_dim(),
            config.hidden_size
        )
    
    def _build_encoder(self) -> nn.Module:
        """构建图像编码器"""
        if self.config.encoder_type == EncoderType.VIT:
            return self._build_vit()
        elif self.config.encoder_type == EncoderType.RESNET:
            return self._build_resnet()
        elif self.config.encoder_type == EncoderType.CLIP_VISION:
            return self._build_clip_vision()
        else:
            return self._build_vit()
    
    def _build_vit(self) -> nn.Module:
        """构建ViT编码器"""
        try:
            from transformers import ViTModel, ViTConfig
            
            config = ViTConfig(
                hidden_size=self.config.hidden_size,
                image_size=self.config.image_size,
                patch_size=self.config.patch_size
            )
            return ViTModel(config)
        except Exception as e:
            logger.warning(f"Failed to load ViT: {e}, using fallback")
            return self._build_fallback_encoder()
    
    def _build_resnet(self) -> nn.Module:
        """构建ResNet编码器"""
        try:
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            # 移除最后的全连接层
            modules = list(resnet.children())[:-1]
            return nn.Sequential(*modules)
        except Exception as e:
            logger.warning(f"Failed to load ResNet: {e}")
            return self._build_fallback_encoder()
    
    def _build_clip_vision(self) -> nn.Module:
        """构建CLIP Vision编码器"""
        try:
            from transformers import CLIPVisionModel
            return CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        except Exception as e:
            logger.warning(f"Failed to load CLIP Vision: {e}")
            return self._build_vit()
    
    def _build_fallback_encoder(self) -> nn.Module:
        """构建备用图像编码器"""
        return SimpleViT(
            image_size=self.config.image_size,
            patch_size=self.config.patch_size,
            hidden_size=self.config.hidden_size,
            num_layers=12,
            num_heads=12
        )
    
    def _get_encoder_output_dim(self) -> int:
        """获取编码器输出维度"""
        if self.config.encoder_type == EncoderType.RESNET:
            return 2048
        return self.config.hidden_size
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            inputs: 包含 pixel_values 的字典 [batch, 3, H, W]
            
        Returns:
            编码后的图像特征 [batch, hidden_size]
        """
        pixel_values = inputs.get('pixel_values')
        
        # 编码
        if hasattr(self.encoder, 'forward'):
            outputs = self.encoder(pixel_values)
        else:
            outputs = self.encoder(pixel_values)
        
        # 处理不同输出格式
        if hasattr(outputs, 'last_hidden_state'):
            # ViT/CLIP: 取CLS token
            features = outputs.last_hidden_state[:, 0, :]
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        elif isinstance(outputs, torch.Tensor):
            # ResNet: 需要flatten
            features = outputs.view(outputs.size(0), -1)
        else:
            features = outputs[0][:, 0, :]
        
        # 投影
        features = self.projection(features)
        
        # Adapter
        if self.adapter is not None:
            features = self.adapter(features)
        
        return features


# ==================== 音频编码器 ====================

class ProductionAudioEncoder(BaseModalityEncoder):
    """生产级音频编码器
    
    支持Whisper、Wav2Vec2、HuBERT等
    """
    
    def __init__(self, config: AudioEncoderConfig):
        super().__init__(ModalityType.AUDIO, config.hidden_size)
        self.config = config
        
        # 构建编码器
        self.encoder = self._build_encoder()
        self.pooler = MeanPooler()
        
        # 特征投影
        encoder_dim = self._get_encoder_output_dim()
        if encoder_dim != config.hidden_size:
            self.projection = nn.Linear(encoder_dim, config.hidden_size)
        else:
            self.projection = nn.Identity()
    
    def _build_encoder(self) -> nn.Module:
        """构建音频编码器"""
        if self.config.encoder_type == EncoderType.WHISPER:
            return self._build_whisper()
        elif self.config.encoder_type == EncoderType.WAV2VEC:
            return self._build_wav2vec()
        else:
            return self._build_fallback_encoder()
    
    def _build_whisper(self) -> nn.Module:
        """构建Whisper编码器"""
        try:
            from transformers import WhisperModel
            model = WhisperModel.from_pretrained(self.config.model_name)
            return model.encoder
        except Exception as e:
            logger.warning(f"Failed to load Whisper: {e}")
            return self._build_fallback_encoder()
    
    def _build_wav2vec(self) -> nn.Module:
        """构建Wav2Vec2编码器"""
        try:
            from transformers import Wav2Vec2Model
            return Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        except Exception as e:
            logger.warning(f"Failed to load Wav2Vec2: {e}")
            return self._build_fallback_encoder()
    
    def _build_fallback_encoder(self) -> nn.Module:
        """构建备用音频编码器"""
        return nn.Sequential(
            # 1D卷积特征提取
            nn.Conv1d(1, 64, kernel_size=10, stride=5),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(128),
            # Transformer
            Permute(0, 2, 1),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=256,
                    nhead=8,
                    dim_feedforward=1024
                ),
                num_layers=4
            )
        )
    
    def _get_encoder_output_dim(self) -> int:
        """获取编码器输出维度"""
        if self.config.encoder_type == EncoderType.WHISPER:
            return 512
        elif self.config.encoder_type == EncoderType.WAV2VEC:
            return 768
        return 256
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            inputs: 包含 input_values 的字典 [batch, seq_len]
            
        Returns:
            编码后的音频特征 [batch, hidden_size]
        """
        input_values = inputs.get('input_values')
        attention_mask = inputs.get('attention_mask')
        
        # 编码
        outputs = self.encoder(input_values, attention_mask=attention_mask)
        
        # 获取隐藏状态
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs
        
        # 池化
        features = self.pooler(hidden_states, attention_mask)
        
        # 投影
        features = self.projection(features)
        
        return features


# ==================== 时序编码器 ====================

class ProductionTimeSeriesEncoder(BaseModalityEncoder):
    """生产级时序编码器
    
    支持传感器、PLC、SCADA等时序数据
    """
    
    def __init__(self, config: TimeSeriesEncoderConfig):
        super().__init__(ModalityType.TIME_SERIES, config.hidden_size)
        self.config = config
        
        # 输入嵌入
        self.input_embedding = nn.Linear(config.input_channels, config.hidden_size)
        
        # 位置编码
        self.position_encoding = PositionalEncoding(
            config.hidden_size,
            max_len=config.seq_length
        )
        
        # 构建编码器
        self.encoder = self._build_encoder()
        
        # 池化
        self.pooler = MeanPooler()
    
    def _build_encoder(self) -> nn.Module:
        """构建时序编码器"""
        if self.config.encoder_type == EncoderType.TRANSFORMER_TS:
            return nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.config.hidden_size,
                    nhead=self.config.num_heads,
                    dim_feedforward=self.config.hidden_size * 4,
                    dropout=0.1
                ),
                num_layers=self.config.num_layers
            )
        elif self.config.encoder_type == EncoderType.LSTM:
            return nn.LSTM(
                input_size=self.config.hidden_size,
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                batch_first=True,
                bidirectional=True
            )
        else:
            return self._build_tcnn()
    
    def _build_tcnn(self) -> nn.Module:
        """构建时序卷积网络"""
        return nn.Sequential(
            nn.Conv1d(self.config.hidden_size, self.config.hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.config.hidden_size, self.config.hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.config.hidden_size, self.config.hidden_size, 3, padding=1)
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            inputs: 包含 time_series 的字典 [batch, seq_len, channels]
            
        Returns:
            编码后的时序特征 [batch, hidden_size]
        """
        time_series = inputs.get('time_series')
        attention_mask = inputs.get('attention_mask')
        
        # 输入嵌入
        x = self.input_embedding(time_series)  # [batch, seq_len, hidden_size]
        
        # 位置编码
        x = self.position_encoding(x)
        
        # 编码
        if isinstance(self.encoder, nn.TransformerEncoder):
            # Transformer: [seq_len, batch, hidden_size]
            x = x.transpose(0, 1)
            x = self.encoder(x)
            x = x.transpose(0, 1)
        elif isinstance(self.encoder, nn.LSTM):
            x, _ = self.encoder(x)
            # 双向LSTM输出合并
            x = x[:, :, :self.config.hidden_size] + x[:, :, self.config.hidden_size:]
        else:
            # CNN: [batch, hidden_size, seq_len]
            x = x.transpose(1, 2)
            x = self.encoder(x)
            x = x.transpose(1, 2)
        
        # 池化
        features = self.pooler(x, attention_mask)
        
        return features


# ==================== 视频编码器 ====================

class ProductionVideoEncoder(BaseModalityEncoder):
    """生产级视频编码器
    
    支持VideoSwin、TimeSformer等
    """
    
    def __init__(self, config: VideoEncoderConfig):
        super().__init__(ModalityType.VIDEO, config.hidden_size)
        self.config = config
        
        # 构建编码器
        self.encoder = self._build_encoder()
        
        # 特征投影
        self.projection = nn.Linear(
            self._get_encoder_output_dim(),
            config.hidden_size
        )
    
    def _build_encoder(self) -> nn.Module:
        """构建视频编码器"""
        try:
            from transformers import VideoMAEModel
            return VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        except Exception as e:
            logger.warning(f"Failed to load VideoMAE: {e}")
            return self._build_fallback_encoder()
    
    def _build_fallback_encoder(self) -> nn.Module:
        """构建备用视频编码器"""
        return Simple3DViT(
            num_frames=self.config.num_frames,
            frame_size=self.config.frame_size,
            hidden_size=self.config.hidden_size,
            num_layers=6,
            num_heads=8
        )
    
    def _get_encoder_output_dim(self) -> int:
        return self.config.hidden_size
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播
        
        Args:
            inputs: 包含 pixel_values 的字典 [batch, num_frames, 3, H, W]
            
        Returns:
            编码后的视频特征 [batch, hidden_size]
        """
        pixel_values = inputs.get('pixel_values')
        
        # 编码
        outputs = self.encoder(pixel_values)
        
        # 获取特征
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state[:, 0, :]  # CLS token
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output
        else:
            features = outputs.mean(dim=1)
        
        # 投影
        features = self.projection(features)
        
        return features


# ==================== 辅助模块 ====================

class AdapterModule(nn.Module):
    """Adapter模块，用于参数高效微调"""
    
    def __init__(self, hidden_size: int, adapter_size: int):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size)
        self.up_proj = nn.Linear(adapter_size, hidden_size)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return x + residual


class MeanPooler(nn.Module):
    """均值池化"""
    
    def forward(self, hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        
        # 扩展mask维度
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = (hidden_states * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / sum_mask


class MaxPooler(nn.Module):
    """最大池化"""
    
    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states.max(dim=1)[0]
        
        # 将padding位置设为很小的值
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        hidden_states = hidden_states.masked_fill(~mask.bool(), float('-inf'))
        return hidden_states.max(dim=1)[0]


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, hidden_size: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_len, 1, hidden_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.transpose(0, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, seq_len, hidden_size]"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class Permute(nn.Module):
    """维度置换"""
    
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class SimpleViT(nn.Module):
    """简化版ViT，作为备用"""
    
    def __init__(self, image_size: int, patch_size: int, hidden_size: int,
                 num_layers: int, num_heads: int):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_dim, hidden_size)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # 位置编码
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, hidden_size)
        )
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4
            ),
            num_layers=num_layers
        )
        
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> Any:
        """x: [batch, 3, H, W]"""
        B = x.shape[0]
        
        # Patchify
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, 3, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, 3 * self.patch_size * self.patch_size)
        
        # Embedding
        x = self.patch_embedding(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.position_embedding
        
        # Transformer
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        x = self.norm(x)
        
        # 返回类似HuggingFace的输出
        return type('Output', (), {'last_hidden_state': x})()


class Simple3DViT(nn.Module):
    """简化版3D ViT，用于视频"""
    
    def __init__(self, num_frames: int, frame_size: int, hidden_size: int,
                 num_layers: int, num_heads: int):
        super().__init__()
        self.num_frames = num_frames
        patch_size = 16
        num_patches_per_frame = (frame_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_dim, hidden_size)
        
        # Temporal embedding
        self.temporal_embedding = nn.Parameter(
            torch.randn(1, num_frames, 1, hidden_size)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4
            ),
            num_layers=num_layers
        )
        
        self.norm = nn.LayerNorm(hidden_size)
        self.patch_size = patch_size
    
    def forward(self, x: torch.Tensor) -> Any:
        """x: [batch, num_frames, 3, H, W]"""
        B, T, C, H, W = x.shape
        
        # 逐帧处理
        frame_features = []
        for t in range(T):
            frame = x[:, t]  # [B, 3, H, W]
            # Patchify
            frame = frame.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            frame = frame.contiguous().view(B, 3, -1, self.patch_size, self.patch_size)
            frame = frame.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, 3 * self.patch_size * self.patch_size)
            frame = self.patch_embedding(frame)
            frame_features.append(frame)
        
        # Stack frames: [B, T, num_patches, hidden_size]
        x = torch.stack(frame_features, dim=1)
        
        # Add temporal embedding
        x = x + self.temporal_embedding[:, :T, :, :]
        
        # Flatten: [B, T * num_patches, hidden_size]
        x = x.view(B, -1, x.size(-1))
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)
        
        x = self.norm(x)
        
        return type('Output', (), {'last_hidden_state': x})()


# ==================== 编码器工厂 ====================

class ModalityEncoderFactory:
    """模态编码器工厂"""
    
    @staticmethod
    def create_encoder(modality: ModalityType, config: Any) -> BaseModalityEncoder:
        """创建模态编码器
        
        Args:
            modality: 模态类型
            config: 编码器配置
            
        Returns:
            模态编码器实例
        """
        if modality == ModalityType.TEXT:
            return ProductionTextEncoder(config)
        elif modality == ModalityType.IMAGE:
            return ProductionImageEncoder(config)
        elif modality == ModalityType.AUDIO:
            return ProductionAudioEncoder(config)
        elif modality == ModalityType.TIME_SERIES:
            return ProductionTimeSeriesEncoder(config)
        elif modality == ModalityType.VIDEO:
            return ProductionVideoEncoder(config)
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    @staticmethod
    def create_encoders(config: ModalEncodersConfig, 
                       modalities: List[ModalityType]) -> Dict[str, BaseModalityEncoder]:
        """创建多个模态编码器
        
        Args:
            config: 编码器总配置
            modalities: 需要的模态列表
            
        Returns:
            模态名称到编码器的映射
        """
        encoders = {}
        
        for modality in modalities:
            modality_config = getattr(config, modality.value, None)
            if modality_config is not None:
                encoders[modality.value] = ModalityEncoderFactory.create_encoder(
                    modality, modality_config
                )
        
        return encoders


# ==================== 投影层 ====================

class UnifiedProjection(nn.Module):
    """统一投影层，将各模态特征投影到统一维度"""
    
    def __init__(self, input_dims: Dict[str, int], unified_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, unified_dim),
                nn.LayerNorm(unified_dim),
                nn.Dropout(dropout),
                nn.GELU(),
                nn.Linear(unified_dim, unified_dim)
            )
            for name, dim in input_dims.items()
        })
        
        self.unified_dim = unified_dim
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """投影各模态特征
        
        Args:
            features: 模态名称到特征的映射
            
        Returns:
            投影后的特征
        """
        projected = {}
        for name, feat in features.items():
            if name in self.projections:
                projected[name] = self.projections[name](feat)
        return projected


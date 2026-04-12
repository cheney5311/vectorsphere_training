# -*- coding: utf-8 -*-
"""
生产级模态编码器

提供各模态专属编码器的实现，包括：
- 文本编码器（BERT/RoBERTa/LLaMA）
- 图像编码器（ViT/ResNet/CLIP）
- 音频编码器（Whisper/Wav2Vec）
- 时序编码器（LSTM/Transformer）
- 视频编码器（VideoSwin/TimeSformer）
- 编码器监控和性能分析
- 参数分析和诊断工具
 """

import logging
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_config import (
    ModalityType, EncoderType,
    TextEncoderConfig, ImageEncoderConfig, AudioEncoderConfig,
    TimeSeriesEncoderConfig, VideoEncoderConfig, ModalEncodersConfig
)

logger = logging.getLogger(__name__)


# ==================== 枚举和数据类 ====================

class EncoderStatus(Enum):
    """编码器状态"""
    INITIALIZED = "initialized"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    FROZEN = "frozen"


@dataclass
class EncoderMetrics:
    """编码器性能指标"""
    total_forward_calls: int = 0
    total_samples_processed: int = 0
    total_processing_time: float = 0.0
    
    avg_forward_time: float = 0.0
    avg_throughput: float = 0.0  # samples/second
    
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    
    error_count: int = 0
    last_error: Optional[str] = None
    
    def update_forward_time(self, time_ms: float, batch_size: int) -> None:
        """更新前向传播时间"""
        self.total_forward_calls += 1
        self.total_samples_processed += batch_size
        self.total_processing_time += time_ms / 1000.0
        
        self.avg_forward_time = (self.total_processing_time / self.total_forward_calls) * 1000
        if self.total_processing_time > 0:
            self.avg_throughput = self.total_samples_processed / self.total_processing_time
    
    def update_memory(self, memory_mb: float) -> None:
        """更新内存使用"""
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        n = self.total_forward_calls
        self.avg_memory_mb = (self.avg_memory_mb * (n - 1) + memory_mb) / n if n > 0 else memory_mb
    
    def record_error(self, error: str) -> None:
        """记录错误"""
        self.error_count += 1
        self.last_error = error


class EncoderMonitor:
    """编码器监控器"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics = EncoderMetrics()
        
        # 历史记录
        self._forward_time_history: deque = deque(maxlen=history_size)
        self._memory_history: deque = deque(maxlen=history_size)
        self._throughput_history: deque = deque(maxlen=history_size)
        
        # 状态
        self.status = EncoderStatus.INITIALIZED
    
    def record_forward(self, time_ms: float, batch_size: int, memory_mb: float) -> None:
        """记录前向传播"""
        self.metrics.update_forward_time(time_ms, batch_size)
        self.metrics.update_memory(memory_mb)
        
        self._forward_time_history.append(time_ms)
        self._memory_history.append(memory_mb)
        
        throughput = batch_size / (time_ms / 1000.0) if time_ms > 0 else 0
        self._throughput_history.append(throughput)
    
    def record_error(self, error: str) -> None:
        """记录错误"""
        self.metrics.record_error(error)
        self.status = EncoderStatus.ERROR
    
    def get_recent_stats(self, n: int = 100) -> Dict[str, float]:
        """获取最近n次的统计"""
        recent_times = list(self._forward_time_history)[-n:]
        recent_memory = list(self._memory_history)[-n:]
        recent_throughput = list(self._throughput_history)[-n:]
        
        return {
            'avg_forward_time': sum(recent_times) / len(recent_times) if recent_times else 0,
            'avg_memory': sum(recent_memory) / len(recent_memory) if recent_memory else 0,
            'avg_throughput': sum(recent_throughput) / len(recent_throughput) if recent_throughput else 0,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        return {
            'status': self.status.value,
            'total_calls': self.metrics.total_forward_calls,
            'total_samples': self.metrics.total_samples_processed,
            'avg_forward_time_ms': self.metrics.avg_forward_time,
            'avg_throughput': self.metrics.avg_throughput,
            'peak_memory_mb': self.metrics.peak_memory_mb,
            'avg_memory_mb': self.metrics.avg_memory_mb,
            'error_count': self.metrics.error_count,
        }
    
    def reset(self) -> None:
        """重置监控"""
        self.metrics = EncoderMetrics()
        self._forward_time_history.clear()
        self._memory_history.clear()
        self._throughput_history.clear()
        self.status = EncoderStatus.INITIALIZED


class EncoderProfiler:
    """编码器性能分析器"""
    
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


# ==================== 基础编码器 ====================

class BaseModalityEncoder(nn.Module, ABC):
    """模态编码器基类
    
    提供所有模态编码器的通用功能：
    - 参数冻结/解冻
    - 性能监控
    - 参数分析
    - 诊断工具
    """
    
    def __init__(self, modality: ModalityType, hidden_size: int):
        super().__init__()
        self.modality = modality
        self.hidden_size = hidden_size
        self._is_frozen = False
        
        # 监控和分析
        self._monitor = EncoderMonitor()
        self._profiler = EncoderProfiler()
        
        # 统计信息
        self._param_count: Optional[int] = None
        self._trainable_param_count: Optional[int] = None
    
    @abstractmethod
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """前向传播，返回编码后的特征"""
        raise NotImplementedError
    
    def forward_with_monitoring(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """带监控的前向传播"""
        start_time = time.time()
        batch_size = self._get_batch_size(inputs)
        
        try:
            self._monitor.status = EncoderStatus.PROCESSING
            
            # 记录初始内存
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                start_memory = torch.cuda.memory_allocated() / (1024 * 1024)
            else:
                start_memory = 0
            
            # 前向传播
            output = self.forward(inputs)
            
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
            self._monitor.record_forward(elapsed_ms, batch_size, memory_used)
            self._monitor.status = EncoderStatus.READY
            
            return output
            
        except Exception as e:
            self._monitor.record_error(str(e))
            raise
    
    def _get_batch_size(self, inputs: Dict[str, torch.Tensor]) -> int:
        """从输入中获取batch size"""
        for value in inputs.values():
            if isinstance(value, torch.Tensor):
                return value.size(0)
        return 0
    
    def freeze(self) -> None:
        """冻结编码器参数"""
        for param in self.parameters():
            param.requires_grad = False
        self._is_frozen = True
        self._monitor.status = EncoderStatus.FROZEN
        logger.info(f"{self.modality.value} encoder frozen")
    
    def unfreeze(self) -> None:
        """解冻编码器参数"""
        for param in self.parameters():
            param.requires_grad = True
        self._is_frozen = False
        self._monitor.status = EncoderStatus.READY
        logger.info(f"{self.modality.value} encoder unfrozen")
    
    def freeze_layers(self, layer_indices: List[int]) -> None:
        """冻结指定层"""
        # 需要子类实现
        logger.warning(f"freeze_layers not implemented for {self.__class__.__name__}")
    
    @property
    def is_frozen(self) -> bool:
        return self._is_frozen
    
    def get_output_dim(self) -> int:
        return self.hidden_size
    
    def count_parameters(self) -> Tuple[int, int]:
        """统计参数数量
        
        Returns:
            (总参数数, 可训练参数数)
        """
        if self._param_count is None or self._trainable_param_count is None:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            self._param_count = total
            self._trainable_param_count = trainable
        
        return self._param_count, self._trainable_param_count
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """获取参数信息"""
        total, trainable = self.count_parameters()
        
        return {
            'total_parameters': total,
            'trainable_parameters': trainable,
            'frozen_parameters': total - trainable,
            'frozen_ratio': (total - trainable) / total if total > 0 else 0,
            'memory_mb': total * 4 / (1024 * 1024),  # FP32
        }
    
    def get_monitor(self) -> EncoderMonitor:
        """获取监控器"""
        return self._monitor
    
    def get_profiler(self) -> EncoderProfiler:
        """获取分析器"""
        return self._profiler
    
    def enable_profiling(self) -> None:
        """启用性能分析"""
        self._profiler.enable()
    
    def disable_profiling(self) -> None:
        """禁用性能分析"""
        self._profiler.disable()
    
    def get_monitor_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        return self._monitor.get_summary()
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断编码器状态"""
        param_info = self.get_parameter_info()
        monitor_summary = self.get_monitor_summary()
        
        diagnosis = {
            'modality': self.modality.value,
            'status': self._monitor.status.value,
            'is_frozen': self._is_frozen,
            'parameters': param_info,
            'performance': monitor_summary,
        }
        
        # 性能建议
        recommendations = []
        if monitor_summary['avg_forward_time_ms'] > 100:
            recommendations.append("High forward time detected, consider optimization")
        if monitor_summary['peak_memory_mb'] > 1000:
            recommendations.append("High memory usage, consider gradient checkpointing")
        if monitor_summary['error_count'] > 0:
            recommendations.append(f"Errors detected: {monitor_summary['error_count']}")
        
        diagnosis['recommendations'] = recommendations
        
        return diagnosis
    
    def print_summary(self) -> None:
        """打印摘要"""
        diagnosis = self.diagnose()
        
        print(f"\n{'='*60}")
        print(f"{self.modality.value.upper()} Encoder Summary")
        print(f"{'='*60}")
        print(f"Status: {diagnosis['status']}")
        print(f"Frozen: {diagnosis['is_frozen']}")
        print(f"\nParameters:")
        print(f"  Total: {diagnosis['parameters']['total_parameters']:,}")
        print(f"  Trainable: {diagnosis['parameters']['trainable_parameters']:,}")
        print(f"  Memory: {diagnosis['parameters']['memory_mb']:.2f} MB")
        print(f"\nPerformance:")
        print(f"  Total calls: {diagnosis['performance']['total_calls']}")
        print(f"  Avg time: {diagnosis['performance']['avg_forward_time_ms']:.2f} ms")
        print(f"  Throughput: {diagnosis['performance']['avg_throughput']:.1f} samples/s")
        print(f"  Peak memory: {diagnosis['performance']['peak_memory_mb']:.2f} MB")
        
        if diagnosis['recommendations']:
            print(f"\nRecommendations:")
            for rec in diagnosis['recommendations']:
                print(f"  - {rec}")
        
        print(f"{'='*60}")
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._monitor.reset()
        self._profiler.reset()
        logger.info(f"{self.modality.value} encoder stats reset")


# ==================== 文本编码器 ====================

class TextEncoder(BaseModalityEncoder):
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
    
    def freeze_layers(self, layer_indices: List[int]) -> None:
        """冻结指定的Transformer层"""
        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
            layers = self.encoder.encoder.layer
            for idx in layer_indices:
                if 0 <= idx < len(layers):
                    for param in layers[idx].parameters():
                        param.requires_grad = False
            logger.info(f"Frozen layers: {layer_indices}")
        else:
            super().freeze_layers(layer_indices)
    
    def get_embeddings(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取词嵌入（不经过编码器）"""
        input_ids = inputs.get('input_ids')
        if hasattr(self.encoder, 'embeddings'):
            return self.encoder.embeddings(input_ids)
        elif hasattr(self.encoder, 'embed_tokens'):
            return self.encoder.embed_tokens(input_ids)
        else:
            raise NotImplementedError("Embeddings not accessible")
    
    def get_attention_weights(self, inputs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """获取注意力权重"""
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )
        
        if hasattr(outputs, 'attentions'):
            return outputs.attentions
        return []


# ==================== 图像编码器 ====================

class ImageEncoder(BaseModalityEncoder):
    """图像编码器
    
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

class AudioEncoder(BaseModalityEncoder):
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

class TimeSeriesEncoder(BaseModalityEncoder):
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

class VideoEncoder(BaseModalityEncoder):
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
            return TextEncoder(config)
        elif modality == ModalityType.IMAGE:
            return ImageEncoder(config)
        elif modality == ModalityType.AUDIO:
            return AudioEncoder(config)
        elif modality == ModalityType.TIME_SERIES:
            return TimeSeriesEncoder(config)
        elif modality == ModalityType.VIDEO:
            return VideoEncoder(config)
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


# ==================== 工具函数 ====================

def create_encoder(
    modality: ModalityType,
    config: Union[TextEncoderConfig, ImageEncoderConfig, AudioEncoderConfig,
                  TimeSeriesEncoderConfig, VideoEncoderConfig]
) -> BaseModalityEncoder:
    """
    创建模态编码器
    
    Args:
        modality: 模态类型
        config: 编码器配置
        
    Returns:
        编码器实例
    """
    return ModalityEncoderFactory.create_encoder(modality, config)


def create_encoders_from_config(
    config: ModalEncodersConfig,
    modalities: List[ModalityType]
) -> Dict[str, BaseModalityEncoder]:
    """
    从配置创建多个编码器
    
    Args:
        config: 编码器总配置
        modalities: 需要的模态列表
        
    Returns:
        模态名称到编码器的映射
    """
    return ModalityEncoderFactory.create_encoders(config, modalities)


def freeze_encoders(
    encoders: Dict[str, BaseModalityEncoder],
    modality_names: List[str]
) -> None:
    """
    冻结指定的编码器
    
    Args:
        encoders: 编码器字典
        modality_names: 要冻结的模态名称列表
    """
    for name in modality_names:
        if name in encoders:
            encoders[name].freeze()
            logger.info(f"Frozen encoder: {name}")


def unfreeze_encoders(
    encoders: Dict[str, BaseModalityEncoder],
    modality_names: List[str]
) -> None:
    """
    解冻指定的编码器
    
    Args:
        encoders: 编码器字典
        modality_names: 要解冻的模态名称列表
    """
    for name in modality_names:
        if name in encoders:
            encoders[name].unfreeze()
            logger.info(f"Unfrozen encoder: {name}")


def get_encoders_summary(
    encoders: Dict[str, BaseModalityEncoder]
) -> Dict[str, Dict[str, Any]]:
    """
    获取所有编码器的摘要
    
    Args:
        encoders: 编码器字典
        
    Returns:
        每个编码器的诊断信息
    """
    summaries = {}
    for name, encoder in encoders.items():
        summaries[name] = encoder.diagnose()
    return summaries


def print_encoders_summary(
    encoders: Dict[str, BaseModalityEncoder]
) -> None:
    """
    打印所有编码器的摘要
    
    Args:
        encoders: 编码器字典
    """
    print(f"\n{'='*70}")
    print("ENCODERS SUMMARY")
    print(f"{'='*70}")
    
    for name, encoder in encoders.items():
        param_info = encoder.get_parameter_info()
        monitor = encoder.get_monitor_summary()
        
        print(f"\n{name.upper()}:")
        print(f"  Status: {encoder._monitor.status.value}")
        print(f"  Parameters: {param_info['total_parameters']:,} "
              f"({param_info['trainable_parameters']:,} trainable)")
        print(f"  Memory: {param_info['memory_mb']:.2f} MB")
        print(f"  Calls: {monitor['total_calls']}, "
              f"Avg time: {monitor['avg_forward_time_ms']:.2f} ms")
    
    print(f"{'='*70}")


def count_total_parameters(
    encoders: Dict[str, BaseModalityEncoder]
) -> Tuple[int, int]:
    """
    统计所有编码器的参数总数
    
    Args:
        encoders: 编码器字典
        
    Returns:
        (总参数数, 可训练参数数)
    """
    total = 0
    trainable = 0
    
    for encoder in encoders.values():
        t, tr = encoder.count_parameters()
        total += t
        trainable += tr
    
    return total, trainable


def estimate_encoders_memory(
    encoders: Dict[str, BaseModalityEncoder],
    precision: str = "fp32"
) -> Dict[str, float]:
    """
    估算编码器内存占用
    
    Args:
        encoders: 编码器字典
        precision: 精度类型 (fp32, fp16, bf16)
        
    Returns:
        内存估算（MB）
    """
    multiplier = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
    }.get(precision.lower(), 4)
    
    memory_mb = {}
    total_mb = 0
    
    for name, encoder in encoders.items():
        total_params, _ = encoder.count_parameters()
        mb = total_params * multiplier / (1024 * 1024)
        memory_mb[name] = mb
        total_mb += mb
    
    memory_mb['total'] = total_mb
    return memory_mb


def reset_all_encoders_stats(
    encoders: Dict[str, BaseModalityEncoder]
) -> None:
    """
    重置所有编码器的统计信息
    
    Args:
        encoders: 编码器字典
    """
    for encoder in encoders.values():
        encoder.reset_stats()
    logger.info("Reset all encoder statistics")


def diagnose_encoders(
    encoders: Dict[str, BaseModalityEncoder]
) -> Dict[str, Any]:
    """
    诊断所有编码器
    
    Args:
        encoders: 编码器字典
        
    Returns:
        诊断结果
    """
    diagnosis = {
        'encoders': {},
        'total_parameters': 0,
        'total_trainable': 0,
        'total_memory_mb': 0,
        'issues': [],
    }
    
    for name, encoder in encoders.items():
        enc_diag = encoder.diagnose()
        diagnosis['encoders'][name] = enc_diag
        
        # 累计统计
        diagnosis['total_parameters'] += enc_diag['parameters']['total_parameters']
        diagnosis['total_trainable'] += enc_diag['parameters']['trainable_parameters']
        diagnosis['total_memory_mb'] += enc_diag['parameters']['memory_mb']
        
        # 收集问题
        if enc_diag['recommendations']:
            diagnosis['issues'].append({
                'encoder': name,
                'recommendations': enc_diag['recommendations']
            })
    
    return diagnosis


def print_diagnosis(
    encoders: Dict[str, BaseModalityEncoder]
) -> None:
    """
    打印诊断结果
    
    Args:
        encoders: 编码器字典
    """
    diagnosis = diagnose_encoders(encoders)
    
    print(f"\n{'='*70}")
    print("ENCODERS DIAGNOSIS")
    print(f"{'='*70}")
    print(f"\nOverall Statistics:")
    print(f"  Total encoders: {len(encoders)}")
    print(f"  Total parameters: {diagnosis['total_parameters']:,}")
    print(f"  Trainable parameters: {diagnosis['total_trainable']:,}")
    print(f"  Total memory: {diagnosis['total_memory_mb']:.2f} MB")
    
    if diagnosis['issues']:
        print(f"\nIssues Found:")
        for issue in diagnosis['issues']:
            print(f"\n  {issue['encoder'].upper()}:")
            for rec in issue['recommendations']:
                print(f"    - {rec}")
    else:
        print(f"\nNo issues found.")
    
    print(f"{'='*70}")


def compare_encoders_performance(
    encoders: Dict[str, BaseModalityEncoder]
) -> Dict[str, Any]:
    """
    比较编码器性能
    
    Args:
        encoders: 编码器字典
        
    Returns:
        性能比较结果
    """
    comparison = {}
    
    for name, encoder in encoders.items():
        monitor = encoder.get_monitor_summary()
        comparison[name] = {
            'avg_time_ms': monitor['avg_forward_time_ms'],
            'throughput': monitor['avg_throughput'],
            'memory_mb': monitor['peak_memory_mb'],
            'calls': monitor['total_calls'],
        }
    
    return comparison


def print_performance_comparison(
    encoders: Dict[str, BaseModalityEncoder]
) -> None:
    """
    打印性能比较
    
    Args:
        encoders: 编码器字典
    """
    comparison = compare_encoders_performance(encoders)
    
    print(f"\n{'='*70}")
    print("ENCODERS PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Encoder':<15} {'Avg Time (ms)':<15} {'Throughput':<15} {'Memory (MB)':<15}")
    print("-" * 70)
    
    for name, perf in comparison.items():
        print(f"{name:<15} {perf['avg_time_ms']:<15.2f} "
              f"{perf['throughput']:<15.1f} {perf['memory_mb']:<15.2f}")
    
    print(f"{'='*70}")


def create_unified_projection_from_encoders(
    encoders: Dict[str, BaseModalityEncoder],
    unified_dim: int,
    dropout: float = 0.1
) -> UnifiedProjection:
    """
    从编码器创建统一投影层
    
    Args:
        encoders: 编码器字典
        unified_dim: 统一维度
        dropout: Dropout率
        
    Returns:
        统一投影层
    """
    input_dims = {
        name: encoder.get_output_dim()
        for name, encoder in encoders.items()
    }
    return UnifiedProjection(input_dims, unified_dim, dropout)


def encode_multimodal_inputs(
    encoders: Dict[str, BaseModalityEncoder],
    inputs: Dict[str, Dict[str, torch.Tensor]],
    use_monitoring: bool = True
) -> Dict[str, torch.Tensor]:
    """
    编码多模态输入
    
    Args:
        encoders: 编码器字典
        inputs: 模态名称到输入数据的映射
        use_monitoring: 是否使用监控
        
    Returns:
        模态名称到特征的映射
    """
    features = {}
    
    for modality, input_data in inputs.items():
        if modality in encoders:
            encoder = encoders[modality]
            if use_monitoring:
                features[modality] = encoder.forward_with_monitoring(input_data)
            else:
                features[modality] = encoder(input_data)
    
    return features



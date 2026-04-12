# -*- coding: utf-8 -*-
"""
多模态训练配置

定义生产级多模态训练的完整配置体系，包括：
- 数据工程配置
- 模态编码器配置
- 跨模态对齐配置
- 多模态融合配置
- 四阶段训练配置
- 分布式训练配置
- 推理部署配置
"""

import logging
import json
import yaml
import copy
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"                    # 文本（工艺文档、维修记录）
    IMAGE = "image"                  # 图像（缺陷检测、设备视觉）
    VIDEO = "video"                  # 视频
    AUDIO = "audio"                  # 音频（ASR）
    TIME_SERIES = "time_series"      # 时序信号（传感器、PLC、SCADA）
    TABLE = "table"                  # 表格（BOM、工艺参数）
    DOCUMENT = "document"            # OCR文档
    USER_LOG = "user_log"            # 用户日志
    
    @classmethod
    def from_string(cls, s: str) -> 'ModalityType':
        """从字符串创建"""
        s = s.lower().replace('-', '_')
        for item in cls:
            if item.value == s:
                return item
        raise ValueError(f"Unknown modality type: {s}")
    
    @property
    def is_visual(self) -> bool:
        """是否是视觉模态"""
        return self in (ModalityType.IMAGE, ModalityType.VIDEO)
    
    @property
    def is_sequential(self) -> bool:
        """是否是序列模态"""
        return self in (ModalityType.TEXT, ModalityType.AUDIO, ModalityType.TIME_SERIES)
    
    @property
    def is_structured(self) -> bool:
        """是否是结构化模态"""
        return self in (ModalityType.TABLE, ModalityType.DOCUMENT)
    
    @property
    def default_encoder(self) -> str:
        """获取默认编码器"""
        mapping = {
            ModalityType.TEXT: "bert",
            ModalityType.IMAGE: "vit",
            ModalityType.VIDEO: "video_swin",
            ModalityType.AUDIO: "whisper",
            ModalityType.TIME_SERIES: "transformer_ts",
            ModalityType.TABLE: "bert",
            ModalityType.DOCUMENT: "bert",
            ModalityType.USER_LOG: "bert",
        }
        return mapping.get(self, "bert")
    
    @classmethod
    def list_all(cls) -> List[str]:
        """列出所有模态类型"""
        return [item.value for item in cls]


class EncoderType(Enum):
    """编码器类型"""
    # 文本编码器
    BERT = "bert"
    ROBERTA = "roberta"
    LLAMA = "llama"
    QWEN = "qwen"
    CUSTOM_TEXT = "custom_text"
    
    # 图像编码器
    VIT = "vit"
    RESNET = "resnet"
    SWIN = "swin"
    CLIP_VISION = "clip_vision"
    CUSTOM_IMAGE = "custom_image"
    
    # 音频编码器
    WHISPER = "whisper"
    WAV2VEC = "wav2vec"
    HUBERT = "hubert"
    CUSTOM_AUDIO = "custom_audio"
    
    # 时序编码器
    LSTM = "lstm"
    TRANSFORMER_TS = "transformer_ts"
    TCNN = "tcnn"
    CUSTOM_TS = "custom_ts"
    
    # 视频编码器
    VIDEO_SWIN = "video_swin"
    TIMESFORMER = "timesformer"
    CUSTOM_VIDEO = "custom_video"
    
    @classmethod
    def from_string(cls, s: str) -> 'EncoderType':
        """从字符串创建"""
        s = s.lower()
        for item in cls:
            if item.value == s:
                return item
        raise ValueError(f"Unknown encoder type: {s}")
    
    @property
    def modality(self) -> str:
        """获取对应的模态"""
        text_encoders = {EncoderType.BERT, EncoderType.ROBERTA, EncoderType.LLAMA, EncoderType.QWEN, EncoderType.CUSTOM_TEXT}
        image_encoders = {EncoderType.VIT, EncoderType.RESNET, EncoderType.SWIN, EncoderType.CLIP_VISION, EncoderType.CUSTOM_IMAGE}
        audio_encoders = {EncoderType.WHISPER, EncoderType.WAV2VEC, EncoderType.HUBERT, EncoderType.CUSTOM_AUDIO}
        ts_encoders = {EncoderType.LSTM, EncoderType.TRANSFORMER_TS, EncoderType.TCNN, EncoderType.CUSTOM_TS}
        video_encoders = {EncoderType.VIDEO_SWIN, EncoderType.TIMESFORMER, EncoderType.CUSTOM_VIDEO}
        
        if self in text_encoders:
            return "text"
        elif self in image_encoders:
            return "image"
        elif self in audio_encoders:
            return "audio"
        elif self in ts_encoders:
            return "time_series"
        elif self in video_encoders:
            return "video"
        return "unknown"
    
    @property
    def is_pretrained(self) -> bool:
        """是否是预训练模型"""
        custom_types = {
            EncoderType.CUSTOM_TEXT, EncoderType.CUSTOM_IMAGE,
            EncoderType.CUSTOM_AUDIO, EncoderType.CUSTOM_TS, EncoderType.CUSTOM_VIDEO
        }
        return self not in custom_types
    
    @property
    def default_hidden_size(self) -> int:
        """默认隐藏层大小"""
        size_mapping = {
            EncoderType.BERT: 768,
            EncoderType.ROBERTA: 768,
            EncoderType.LLAMA: 4096,
            EncoderType.QWEN: 4096,
            EncoderType.VIT: 768,
            EncoderType.RESNET: 2048,
            EncoderType.SWIN: 768,
            EncoderType.CLIP_VISION: 768,
            EncoderType.WHISPER: 512,
            EncoderType.WAV2VEC: 768,
            EncoderType.HUBERT: 768,
            EncoderType.LSTM: 256,
            EncoderType.TRANSFORMER_TS: 256,
            EncoderType.TCNN: 256,
            EncoderType.VIDEO_SWIN: 768,
            EncoderType.TIMESFORMER: 768,
        }
        return size_mapping.get(self, 768)
    
    @classmethod
    def list_by_modality(cls, modality: str) -> List['EncoderType']:
        """按模态列出编码器"""
        return [e for e in cls if e.modality == modality]


class AlignmentMethod(Enum):
    """对齐方法"""
    CONTRASTIVE = "contrastive"      # 对比学习（CLIP风格）
    EXPLICIT_ALIGN = "explicit"       # 显式对齐
    CROSS_ATTENTION = "cross_attention"  # 交叉注意力
    OPTIMAL_TRANSPORT = "optimal_transport"  # 最优传输
    KNOWLEDGE_DISTILL = "knowledge_distill"  # 知识蒸馏对齐
    
    @classmethod
    def from_string(cls, s: str) -> 'AlignmentMethod':
        """从字符串创建"""
        s = s.lower()
        for item in cls:
            if item.value == s:
                return item
        raise ValueError(f"Unknown alignment method: {s}")
    
    @property
    def requires_paired_data(self) -> bool:
        """是否需要配对数据"""
        return self in (AlignmentMethod.CONTRASTIVE, AlignmentMethod.EXPLICIT_ALIGN)
    
    @property
    def memory_efficiency(self) -> float:
        """内存效率（0-1，越高越节省内存）"""
        efficiency = {
            AlignmentMethod.CONTRASTIVE: 0.8,
            AlignmentMethod.EXPLICIT_ALIGN: 0.7,
            AlignmentMethod.CROSS_ATTENTION: 0.5,
            AlignmentMethod.OPTIMAL_TRANSPORT: 0.4,
            AlignmentMethod.KNOWLEDGE_DISTILL: 0.6,
        }
        return efficiency.get(self, 0.5)
    
    @classmethod
    def recommend(cls, num_modalities: int, paired_data: bool = True) -> 'AlignmentMethod':
        """推荐对齐方法"""
        if not paired_data:
            return cls.KNOWLEDGE_DISTILL
        if num_modalities == 2:
            return cls.CONTRASTIVE
        return cls.CROSS_ATTENTION


class FusionStage(Enum):
    """融合阶段"""
    EARLY = "early"          # 早期融合（特征层面）
    MIDDLE = "middle"        # 中期融合（语义层面）
    LATE = "late"            # 后期融合（决策层面）
    ADAPTIVE = "adaptive"    # 自适应融合
    
    @classmethod
    def from_string(cls, s: str) -> 'FusionStage':
        """从字符串创建"""
        s = s.lower()
        for item in cls:
            if item.value == s:
                return item
        raise ValueError(f"Unknown fusion stage: {s}")
    
    @property
    def compute_cost(self) -> float:
        """计算成本（相对值）"""
        costs = {
            FusionStage.EARLY: 0.3,
            FusionStage.MIDDLE: 0.6,
            FusionStage.LATE: 0.8,
            FusionStage.ADAPTIVE: 1.0,
        }
        return costs.get(self, 0.5)
    
    @property
    def representation_quality(self) -> float:
        """表示质量（相对值）"""
        quality = {
            FusionStage.EARLY: 0.6,
            FusionStage.MIDDLE: 0.9,
            FusionStage.LATE: 0.7,
            FusionStage.ADAPTIVE: 0.95,
        }
        return quality.get(self, 0.7)


class FusionMethod(Enum):
    """融合方法"""
    CONCAT = "concat"                # 拼接
    ATTENTION = "attention"          # 注意力融合
    GATED = "gated"                  # 门控融合
    CROSS_ATTENTION = "cross_attention"  # 交叉注意力
    TRANSFORMER = "transformer"      # Transformer融合
    PERCEIVER = "perceiver"          # Perceiver架构
    FLAMINGO = "flamingo"            # Flamingo风格
    QFORMER = "qformer"              # Q-Former（BLIP-2）
    
    @classmethod
    def from_string(cls, s: str) -> 'FusionMethod':
        """从字符串创建"""
        s = s.lower()
        for item in cls:
            if item.value == s:
                return item
        raise ValueError(f"Unknown fusion method: {s}")
    
    @property
    def complexity(self) -> str:
        """复杂度"""
        high_complexity = {FusionMethod.PERCEIVER, FusionMethod.FLAMINGO, FusionMethod.QFORMER, FusionMethod.TRANSFORMER}
        medium_complexity = {FusionMethod.CROSS_ATTENTION, FusionMethod.GATED, FusionMethod.ATTENTION}
        if self in high_complexity:
            return "high"
        elif self in medium_complexity:
            return "medium"
        return "low"
    
    @property
    def param_count_multiplier(self) -> float:
        """参数量倍数（相对于基础模型）"""
        multipliers = {
            FusionMethod.CONCAT: 1.0,
            FusionMethod.ATTENTION: 1.2,
            FusionMethod.GATED: 1.3,
            FusionMethod.CROSS_ATTENTION: 1.5,
            FusionMethod.TRANSFORMER: 2.0,
            FusionMethod.PERCEIVER: 1.8,
            FusionMethod.FLAMINGO: 2.5,
            FusionMethod.QFORMER: 2.0,
        }
        return multipliers.get(self, 1.0)
    
    @classmethod
    def recommend(cls, num_modalities: int, quality_priority: bool = True) -> 'FusionMethod':
        """推荐融合方法"""
        if quality_priority:
            if num_modalities <= 2:
                return cls.CROSS_ATTENTION
            return cls.QFORMER
        else:
            if num_modalities <= 2:
                return cls.ATTENTION
            return cls.CONCAT


class TrainingStage(Enum):
    """训练阶段"""
    MODALITY_PRETRAIN = "modality_pretrain"      # 阶段一：模态预训练
    CROSS_MODAL_ALIGN = "cross_modal_align"      # 阶段二：跨模态对齐
    INSTRUCTION_TUNING = "instruction_tuning"    # 阶段三：指令微调
    ALIGNMENT_SAFETY = "alignment_safety"        # 阶段四：对齐与安全
    
    @classmethod
    def from_string(cls, s: str) -> 'TrainingStage':
        """从字符串创建"""
        s = s.lower()
        for item in cls:
            if item.value == s:
                return item
        raise ValueError(f"Unknown training stage: {s}")
    
    @property
    def stage_number(self) -> int:
        """阶段编号"""
        return list(TrainingStage).index(self) + 1
    
    @property
    def typical_epochs(self) -> int:
        """典型训练轮数"""
        epochs = {
            TrainingStage.MODALITY_PRETRAIN: 10,
            TrainingStage.CROSS_MODAL_ALIGN: 5,
            TrainingStage.INSTRUCTION_TUNING: 3,
            TrainingStage.ALIGNMENT_SAFETY: 1,
        }
        return epochs.get(self, 5)
    
    @property
    def typical_lr(self) -> float:
        """典型学习率"""
        lrs = {
            TrainingStage.MODALITY_PRETRAIN: 1e-4,
            TrainingStage.CROSS_MODAL_ALIGN: 1e-5,
            TrainingStage.INSTRUCTION_TUNING: 2e-5,
            TrainingStage.ALIGNMENT_SAFETY: 1e-6,
        }
        return lrs.get(self, 1e-5)
    
    def next_stage(self) -> Optional['TrainingStage']:
        """获取下一阶段"""
        stages = list(TrainingStage)
        idx = stages.index(self)
        if idx < len(stages) - 1:
            return stages[idx + 1]
        return None


class DataSourceType(Enum):
    """数据来源类型"""
    WEB_IMAGE_TEXT = "web_image_text"    # 图文网页
    VIDEO_ASR = "video_asr"              # 视频+ASR
    OCR_DOCUMENT = "ocr_document"        # OCR文档
    USER_LOG = "user_log"                # 用户日志
    SYNTHETIC = "synthetic"              # 合成数据
    
    @classmethod
    def from_string(cls, s: str) -> 'DataSourceType':
        """从字符串创建"""
        s = s.lower()
        for item in cls:
            if item.value == s:
                return item
        raise ValueError(f"Unknown data source type: {s}")
    
    @property
    def modalities(self) -> List[str]:
        """包含的模态"""
        modality_mapping = {
            DataSourceType.WEB_IMAGE_TEXT: ["text", "image"],
            DataSourceType.VIDEO_ASR: ["video", "audio", "text"],
            DataSourceType.OCR_DOCUMENT: ["document", "text"],
            DataSourceType.USER_LOG: ["text", "user_log"],
            DataSourceType.SYNTHETIC: ["text", "image"],
        }
        return modality_mapping.get(self, [])
    
    @property
    def data_quality(self) -> float:
        """数据质量估计（0-1）"""
        quality = {
            DataSourceType.WEB_IMAGE_TEXT: 0.6,
            DataSourceType.VIDEO_ASR: 0.7,
            DataSourceType.OCR_DOCUMENT: 0.8,
            DataSourceType.USER_LOG: 0.5,
            DataSourceType.SYNTHETIC: 0.9,
        }
        return quality.get(self, 0.5)


# ==================== 数据工程配置 ====================

@dataclass
class DataDeduplicationConfig:
    """数据去重配置"""
    enabled: bool = True
    method: str = "perceptual_hash"  # perceptual_hash, minhash, simhash
    
    # 感知哈希配置
    hash_size: int = 16
    highfreq_factor: int = 4
    
    # MinHash配置
    num_perm: int = 128
    threshold: float = 0.8
    
    # 并行处理
    num_workers: int = 4
    batch_size: int = 1000
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.method not in ["perceptual_hash", "minhash", "simhash"]:
            errors.append(f"Invalid deduplication method: {self.method}")
        if self.hash_size < 4 or self.hash_size > 64:
            errors.append(f"hash_size should be between 4 and 64, got {self.hash_size}")
        if self.threshold < 0 or self.threshold > 1:
            errors.append(f"threshold should be between 0 and 1, got {self.threshold}")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class DataFilterConfig:
    """数据过滤配置"""
    enabled: bool = True
    
    # NSFW过滤
    nsfw_filter: bool = True
    nsfw_threshold: float = 0.5
    
    # 质量过滤
    min_text_length: int = 10
    max_text_length: int = 100000
    min_image_size: int = 64
    max_image_size: int = 4096
    
    # 模态一致性
    consistency_check: bool = True
    consistency_threshold: float = 0.3
    
    # 语言过滤
    language_filter: Optional[List[str]] = None  # ['zh', 'en']
    
    # 版权合规
    copyright_scan: bool = True
    compliance_scan: bool = True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.min_text_length < 0:
            errors.append("min_text_length cannot be negative")
        if self.max_text_length < self.min_text_length:
            errors.append("max_text_length must be >= min_text_length")
        if self.min_image_size < 1:
            errors.append("min_image_size must be at least 1")
        if self.max_image_size < self.min_image_size:
            errors.append("max_image_size must be >= min_image_size")
        if self.nsfw_threshold < 0 or self.nsfw_threshold > 1:
            errors.append("nsfw_threshold must be between 0 and 1")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class DataAugmentationConfig:
    """数据增强配置"""
    enabled: bool = True
    
    # 图像增强
    image_flip: bool = True
    image_rotate: bool = True
    image_crop: bool = True
    image_color_jitter: bool = True
    mixup_alpha: float = 0.0
    cutmix_alpha: float = 0.0
    
    # 文本增强
    text_synonym_replace: bool = False
    text_back_translation: bool = False
    
    # 合成数据
    synthetic_ratio: float = 0.1  # 合成数据比例
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.mixup_alpha < 0:
            errors.append("mixup_alpha cannot be negative")
        if self.cutmix_alpha < 0:
            errors.append("cutmix_alpha cannot be negative")
        if self.synthetic_ratio < 0 or self.synthetic_ratio > 1:
            errors.append("synthetic_ratio must be between 0 and 1")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def get_augmentation_count(self) -> int:
        """获取启用的增强数量"""
        count = 0
        if self.image_flip:
            count += 1
        if self.image_rotate:
            count += 1
        if self.image_crop:
            count += 1
        if self.image_color_jitter:
            count += 1
        if self.mixup_alpha > 0:
            count += 1
        if self.cutmix_alpha > 0:
            count += 1
        if self.text_synonym_replace:
            count += 1
        if self.text_back_translation:
            count += 1
        return count


@dataclass
class DataEngineeringConfig:
    """数据工程配置"""
    # 数据来源
    data_sources: List[DataSourceType] = field(default_factory=lambda: [
        DataSourceType.WEB_IMAGE_TEXT,
        DataSourceType.VIDEO_ASR,
        DataSourceType.OCR_DOCUMENT
    ])
    
    # 去重配置
    deduplication: DataDeduplicationConfig = field(default_factory=DataDeduplicationConfig)
    
    # 过滤配置
    filtering: DataFilterConfig = field(default_factory=DataFilterConfig)
    
    # 增强配置
    augmentation: DataAugmentationConfig = field(default_factory=DataAugmentationConfig)
    
    # 数据缓存
    cache_dir: str = "./data_cache"
    use_memory_map: bool = True
    
    # 采样策略
    sampling_strategy: str = "balanced"  # balanced, weighted, uniform
    sample_weights: Optional[Dict[str, float]] = None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        
        # 验证子配置
        valid, sub_errors = self.deduplication.validate()
        if not valid:
            errors.extend([f"deduplication: {e}" for e in sub_errors])
        
        valid, sub_errors = self.filtering.validate()
        if not valid:
            errors.extend([f"filtering: {e}" for e in sub_errors])
        
        valid, sub_errors = self.augmentation.validate()
        if not valid:
            errors.extend([f"augmentation: {e}" for e in sub_errors])
        
        if self.sampling_strategy not in ["balanced", "weighted", "uniform"]:
            errors.append(f"Invalid sampling_strategy: {self.sampling_strategy}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_sources': [ds.value for ds in self.data_sources],
            'deduplication': self.deduplication.to_dict(),
            'filtering': self.filtering.to_dict(),
            'augmentation': self.augmentation.to_dict(),
            'cache_dir': self.cache_dir,
            'use_memory_map': self.use_memory_map,
            'sampling_strategy': self.sampling_strategy,
            'sample_weights': self.sample_weights,
        }
    
    def get_all_modalities(self) -> List[str]:
        """获取所有涉及的模态"""
        modalities = set()
        for source in self.data_sources:
            modalities.update(source.modalities)
        return list(modalities)
    
    def estimate_preprocessing_time(self, num_samples: int) -> float:
        """估算预处理时间（秒）"""
        base_time = num_samples * 0.01  # 基础时间：每样本10ms
        
        if self.deduplication.enabled:
            base_time *= 1.5
        if self.filtering.enabled:
            base_time *= 1.3
        if self.augmentation.enabled:
            base_time *= (1 + 0.1 * self.augmentation.get_augmentation_count())
        
        return base_time


# ==================== 编码器配置 ====================

@dataclass
class TextEncoderConfig:
    """文本编码器配置"""
    encoder_type: EncoderType = EncoderType.BERT
    model_name: str = "bert-base-chinese"
    hidden_size: int = 768
    max_length: int = 512
    freeze: bool = False
    pooling: str = "mean"  # cls, mean, max
    use_adapter: bool = False
    adapter_size: int = 64
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.encoder_type.modality != "text":
            errors.append(f"Encoder type {self.encoder_type} is not a text encoder")
        if self.hidden_size < 64:
            errors.append("hidden_size should be at least 64")
        if self.max_length < 1:
            errors.append("max_length must be positive")
        if self.pooling not in ["cls", "mean", "max"]:
            errors.append(f"Invalid pooling: {self.pooling}")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'encoder_type': self.encoder_type.value,
            'model_name': self.model_name,
            'hidden_size': self.hidden_size,
            'max_length': self.max_length,
            'freeze': self.freeze,
            'pooling': self.pooling,
            'use_adapter': self.use_adapter,
            'adapter_size': self.adapter_size,
        }
    
    def estimate_memory_mb(self, batch_size: int = 1) -> float:
        """估算内存占用（MB）"""
        # 基础模型内存
        param_count = {
            EncoderType.BERT: 110_000_000,
            EncoderType.ROBERTA: 125_000_000,
            EncoderType.LLAMA: 7_000_000_000,
            EncoderType.QWEN: 7_000_000_000,
        }
        params = param_count.get(self.encoder_type, 100_000_000)
        
        # 参数内存 + 激活内存
        param_memory = params * 4 / (1024 * 1024)  # FP32
        activation_memory = batch_size * self.max_length * self.hidden_size * 4 / (1024 * 1024)
        
        return param_memory + activation_memory


@dataclass
class ImageEncoderConfig:
    """图像编码器配置"""
    encoder_type: EncoderType = EncoderType.VIT
    model_name: str = "vit-base-patch16-224"
    hidden_size: int = 768
    image_size: int = 224
    patch_size: int = 16
    freeze: bool = False
    use_adapter: bool = False
    adapter_size: int = 64
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.encoder_type.modality != "image":
            errors.append(f"Encoder type {self.encoder_type} is not an image encoder")
        if self.image_size < 32:
            errors.append("image_size should be at least 32")
        if self.patch_size < 1:
            errors.append("patch_size must be positive")
        if self.image_size % self.patch_size != 0:
            errors.append("image_size must be divisible by patch_size")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'encoder_type': self.encoder_type.value,
            'model_name': self.model_name,
            'hidden_size': self.hidden_size,
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'freeze': self.freeze,
            'use_adapter': self.use_adapter,
            'adapter_size': self.adapter_size,
        }
    
    @property
    def num_patches(self) -> int:
        """计算patch数量"""
        return (self.image_size // self.patch_size) ** 2
    
    def estimate_memory_mb(self, batch_size: int = 1) -> float:
        """估算内存占用（MB）"""
        param_count = {
            EncoderType.VIT: 86_000_000,
            EncoderType.RESNET: 25_000_000,
            EncoderType.SWIN: 88_000_000,
            EncoderType.CLIP_VISION: 86_000_000,
        }
        params = param_count.get(self.encoder_type, 80_000_000)
        
        param_memory = params * 4 / (1024 * 1024)
        activation_memory = batch_size * self.num_patches * self.hidden_size * 4 / (1024 * 1024)
        
        return param_memory + activation_memory


@dataclass
class AudioEncoderConfig:
    """音频编码器配置"""
    encoder_type: EncoderType = EncoderType.WHISPER
    model_name: str = "whisper-base"
    hidden_size: int = 512
    sample_rate: int = 16000
    max_duration: float = 30.0
    freeze: bool = True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.encoder_type.modality != "audio":
            errors.append(f"Encoder type {self.encoder_type} is not an audio encoder")
        if self.sample_rate < 8000:
            errors.append("sample_rate should be at least 8000")
        if self.max_duration <= 0:
            errors.append("max_duration must be positive")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'encoder_type': self.encoder_type.value,
            'model_name': self.model_name,
            'hidden_size': self.hidden_size,
            'sample_rate': self.sample_rate,
            'max_duration': self.max_duration,
            'freeze': self.freeze,
        }


@dataclass
class TimeSeriesEncoderConfig:
    """时序编码器配置"""
    encoder_type: EncoderType = EncoderType.TRANSFORMER_TS
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    seq_length: int = 512
    input_channels: int = 1
    freeze: bool = False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.encoder_type.modality != "time_series":
            errors.append(f"Encoder type {self.encoder_type} is not a time series encoder")
        if self.hidden_size % self.num_heads != 0:
            errors.append("hidden_size must be divisible by num_heads")
        if self.seq_length < 1:
            errors.append("seq_length must be positive")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'encoder_type': self.encoder_type.value,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'seq_length': self.seq_length,
            'input_channels': self.input_channels,
            'freeze': self.freeze,
        }


@dataclass
class VideoEncoderConfig:
    """视频编码器配置"""
    encoder_type: EncoderType = EncoderType.VIDEO_SWIN
    model_name: str = "video-swin-base"
    hidden_size: int = 768
    num_frames: int = 8
    frame_size: int = 224
    freeze: bool = True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.encoder_type.modality != "video":
            errors.append(f"Encoder type {self.encoder_type} is not a video encoder")
        if self.num_frames < 1:
            errors.append("num_frames must be positive")
        if self.frame_size < 32:
            errors.append("frame_size should be at least 32")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'encoder_type': self.encoder_type.value,
            'model_name': self.model_name,
            'hidden_size': self.hidden_size,
            'num_frames': self.num_frames,
            'frame_size': self.frame_size,
            'freeze': self.freeze,
        }


@dataclass
class ModalEncodersConfig:
    """模态编码器总配置"""
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    image: ImageEncoderConfig = field(default_factory=ImageEncoderConfig)
    audio: AudioEncoderConfig = field(default_factory=AudioEncoderConfig)
    time_series: TimeSeriesEncoderConfig = field(default_factory=TimeSeriesEncoderConfig)
    video: VideoEncoderConfig = field(default_factory=VideoEncoderConfig)
    
    # 统一表示维度
    unified_dim: int = 768
    
    # 投影层
    use_projection: bool = True
    projection_dropout: float = 0.1
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        
        valid, sub_errors = self.text.validate()
        if not valid:
            errors.extend([f"text: {e}" for e in sub_errors])
        
        valid, sub_errors = self.image.validate()
        if not valid:
            errors.extend([f"image: {e}" for e in sub_errors])
        
        valid, sub_errors = self.audio.validate()
        if not valid:
            errors.extend([f"audio: {e}" for e in sub_errors])
        
        valid, sub_errors = self.time_series.validate()
        if not valid:
            errors.extend([f"time_series: {e}" for e in sub_errors])
        
        valid, sub_errors = self.video.validate()
        if not valid:
            errors.extend([f"video: {e}" for e in sub_errors])
        
        if self.unified_dim < 64:
            errors.append("unified_dim should be at least 64")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'text': self.text.to_dict(),
            'image': self.image.to_dict(),
            'audio': self.audio.to_dict(),
            'time_series': self.time_series.to_dict(),
            'video': self.video.to_dict(),
            'unified_dim': self.unified_dim,
            'use_projection': self.use_projection,
            'projection_dropout': self.projection_dropout,
        }
    
    def get_encoder_config(self, modality: str):
        """获取指定模态的编码器配置"""
        modality_map = {
            'text': self.text,
            'image': self.image,
            'audio': self.audio,
            'time_series': self.time_series,
            'video': self.video,
        }
        return modality_map.get(modality)
    
    def get_hidden_sizes(self) -> Dict[str, int]:
        """获取所有编码器的隐藏层大小"""
        return {
            'text': self.text.hidden_size,
            'image': self.image.hidden_size,
            'audio': self.audio.hidden_size,
            'time_series': self.time_series.hidden_size,
            'video': self.video.hidden_size,
        }
    
    def estimate_total_memory_mb(self, batch_size: int = 1) -> float:
        """估算所有编码器的总内存占用"""
        total = 0
        total += self.text.estimate_memory_mb(batch_size)
        total += self.image.estimate_memory_mb(batch_size)
        return total
    
    def get_frozen_encoders(self) -> List[str]:
        """获取冻结的编码器列表"""
        frozen = []
        if self.text.freeze:
            frozen.append('text')
        if self.image.freeze:
            frozen.append('image')
        if self.audio.freeze:
            frozen.append('audio')
        if self.time_series.freeze:
            frozen.append('time_series')
        if self.video.freeze:
            frozen.append('video')
        return frozen


# ==================== 对齐配置 ====================

@dataclass
class ContrastiveLearningConfig:
    """对比学习配置"""
    temperature: float = 0.07
    loss_type: str = "info_nce"  # info_nce, clip, simclr
    hard_negative_mining: bool = True
    hard_negative_ratio: float = 0.2
    in_batch_negatives: bool = True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.temperature <= 0:
            errors.append("temperature must be positive")
        if self.loss_type not in ["info_nce", "clip", "simclr"]:
            errors.append(f"Invalid loss_type: {self.loss_type}")
        if self.hard_negative_ratio < 0 or self.hard_negative_ratio > 1:
            errors.append("hard_negative_ratio must be between 0 and 1")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ExplicitAlignConfig:
    """显式对齐配置"""
    method: str = "attention"  # attention, mlp, linear
    hidden_size: int = 768
    num_layers: int = 2
    align_loss_weight: float = 1.0
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.method not in ["attention", "mlp", "linear"]:
            errors.append(f"Invalid method: {self.method}")
        if self.hidden_size < 64:
            errors.append("hidden_size should be at least 64")
        if self.num_layers < 1:
            errors.append("num_layers must be positive")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class CrossModalAlignmentConfig:
    """跨模态对齐配置"""
    method: AlignmentMethod = AlignmentMethod.CONTRASTIVE
    
    # 对比学习配置
    contrastive: ContrastiveLearningConfig = field(default_factory=ContrastiveLearningConfig)
    
    # 显式对齐配置
    explicit: ExplicitAlignConfig = field(default_factory=ExplicitAlignConfig)
    
    # 冻结策略
    freeze_encoders: bool = True
    freeze_text_encoder: bool = True
    freeze_vision_encoder: bool = False
    
    # 投影配置
    projection_dim: int = 512
    projection_layers: int = 2
    projection_dropout: float = 0.1
    
    # 损失权重
    align_loss_weight: float = 1.0
    kl_loss_weight: float = 0.1
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        
        valid, sub_errors = self.contrastive.validate()
        if not valid:
            errors.extend([f"contrastive: {e}" for e in sub_errors])
        
        valid, sub_errors = self.explicit.validate()
        if not valid:
            errors.extend([f"explicit: {e}" for e in sub_errors])
        
        if self.projection_dim < 64:
            errors.append("projection_dim should be at least 64")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'method': self.method.value,
            'contrastive': self.contrastive.to_dict(),
            'explicit': self.explicit.to_dict(),
            'freeze_encoders': self.freeze_encoders,
            'freeze_text_encoder': self.freeze_text_encoder,
            'freeze_vision_encoder': self.freeze_vision_encoder,
            'projection_dim': self.projection_dim,
            'projection_layers': self.projection_layers,
            'projection_dropout': self.projection_dropout,
            'align_loss_weight': self.align_loss_weight,
            'kl_loss_weight': self.kl_loss_weight,
        }
    
    def get_loss_config(self) -> Dict[str, Any]:
        """获取损失函数配置"""
        if self.method == AlignmentMethod.CONTRASTIVE:
            return {
                'type': self.contrastive.loss_type,
                'temperature': self.contrastive.temperature,
                'weight': self.align_loss_weight,
            }
        return {
            'type': 'mse',
            'weight': self.align_loss_weight,
        }


# ==================== 融合配置 ====================

@dataclass
class EarlyFusionConfig:
    """早期融合配置"""
    method: FusionMethod = FusionMethod.CONCAT
    concat_dim: int = 768
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'method': self.method.value,
            'concat_dim': self.concat_dim,
        }


@dataclass
class MiddleFusionConfig:
    """中期融合配置"""
    method: FusionMethod = FusionMethod.CROSS_ATTENTION
    num_layers: int = 6
    num_heads: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    dropout: float = 0.1
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.hidden_size % self.num_heads != 0:
            errors.append("hidden_size must be divisible by num_heads")
        if self.num_layers < 1:
            errors.append("num_layers must be positive")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'method': self.method.value,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'hidden_size': self.hidden_size,
            'intermediate_size': self.intermediate_size,
            'dropout': self.dropout,
        }
    
    def estimate_param_count(self) -> int:
        """估算参数数量"""
        # 每层大约有 4 * hidden_size^2 + hidden_size * intermediate_size * 2 参数
        per_layer = 4 * self.hidden_size ** 2 + self.hidden_size * self.intermediate_size * 2
        return self.num_layers * per_layer


@dataclass
class LateFusionConfig:
    """后期融合配置"""
    method: FusionMethod = FusionMethod.ATTENTION
    num_heads: int = 8
    fusion_dim: int = 768
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'method': self.method.value,
            'num_heads': self.num_heads,
            'fusion_dim': self.fusion_dim,
        }


@dataclass
class QFormerConfig:
    """Q-Former配置（BLIP-2风格）"""
    num_query_tokens: int = 32
    num_layers: int = 6
    num_heads: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    cross_attention_freq: int = 2
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.hidden_size % self.num_heads != 0:
            errors.append("hidden_size must be divisible by num_heads")
        if self.num_query_tokens < 1:
            errors.append("num_query_tokens must be positive")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def estimate_param_count(self) -> int:
        """估算参数数量"""
        per_layer = 4 * self.hidden_size ** 2 + self.hidden_size * self.intermediate_size * 2
        query_params = self.num_query_tokens * self.hidden_size
        return self.num_layers * per_layer + query_params


@dataclass
class PerceiverConfig:
    """Perceiver配置"""
    num_latents: int = 256
    latent_dim: int = 512
    num_self_attention_layers: int = 6
    num_cross_attention_layers: int = 2
    num_heads: int = 8
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.latent_dim % self.num_heads != 0:
            errors.append("latent_dim must be divisible by num_heads")
        if self.num_latents < 1:
            errors.append("num_latents must be positive")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class MultiModalFusionConfig:
    """多模态融合配置"""
    stage: FusionStage = FusionStage.MIDDLE
    method: FusionMethod = FusionMethod.CROSS_ATTENTION
    
    # 各阶段配置
    early: EarlyFusionConfig = field(default_factory=EarlyFusionConfig)
    middle: MiddleFusionConfig = field(default_factory=MiddleFusionConfig)
    late: LateFusionConfig = field(default_factory=LateFusionConfig)
    
    # 高级融合架构
    qformer: QFormerConfig = field(default_factory=QFormerConfig)
    perceiver: PerceiverConfig = field(default_factory=PerceiverConfig)
    
    # 输出配置
    output_dim: int = 768
    use_modality_embedding: bool = True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        
        valid, sub_errors = self.middle.validate()
        if not valid:
            errors.extend([f"middle: {e}" for e in sub_errors])
        
        valid, sub_errors = self.qformer.validate()
        if not valid:
            errors.extend([f"qformer: {e}" for e in sub_errors])
        
        valid, sub_errors = self.perceiver.validate()
        if not valid:
            errors.extend([f"perceiver: {e}" for e in sub_errors])
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'stage': self.stage.value,
            'method': self.method.value,
            'early': self.early.to_dict(),
            'middle': self.middle.to_dict(),
            'late': self.late.to_dict(),
            'qformer': self.qformer.to_dict(),
            'perceiver': self.perceiver.to_dict(),
            'output_dim': self.output_dim,
            'use_modality_embedding': self.use_modality_embedding,
        }
    
    def get_active_config(self):
        """获取当前激活的融合配置"""
        if self.stage == FusionStage.EARLY:
            return self.early
        elif self.stage == FusionStage.MIDDLE:
            return self.middle
        elif self.stage == FusionStage.LATE:
            return self.late
        return self.middle
    
    def estimate_param_count(self) -> int:
        """估算融合模块参数数量"""
        if self.method == FusionMethod.QFORMER:
            return self.qformer.estimate_param_count()
        elif self.stage == FusionStage.MIDDLE:
            return self.middle.estimate_param_count()
        return 0


# ==================== 训练阶段配置 ====================

@dataclass
class ModalityPretrainConfig:
    """阶段一：模态预训练配置"""
    enabled: bool = True
    
    # 文本预训练
    text_pretrain: bool = True
    text_task: str = "mlm"  # mlm, clm, span
    text_mask_ratio: float = 0.15
    
    # 图像预训练
    image_pretrain: bool = True
    image_task: str = "mae"  # mae, clip, dino
    image_mask_ratio: float = 0.75
    
    # 训练配置
    epochs: int = 10
    learning_rate: float = 1e-4
    batch_size: int = 256
    warmup_ratio: float = 0.1
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.text_task not in ["mlm", "clm", "span"]:
            errors.append(f"Invalid text_task: {self.text_task}")
        if self.image_task not in ["mae", "clip", "dino"]:
            errors.append(f"Invalid image_task: {self.image_task}")
        if self.text_mask_ratio < 0 or self.text_mask_ratio > 1:
            errors.append("text_mask_ratio must be between 0 and 1")
        if self.image_mask_ratio < 0 or self.image_mask_ratio > 1:
            errors.append("image_mask_ratio must be between 0 and 1")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def estimate_training_time_hours(self, num_samples: int, gpu_count: int = 1) -> float:
        """估算训练时间（小时）"""
        samples_per_hour = 50000 * gpu_count  # 估算每小时处理样本数
        total_samples = num_samples * self.epochs
        return total_samples / samples_per_hour


@dataclass
class CrossModalAlignTrainConfig:
    """阶段二：跨模态对齐训练配置"""
    enabled: bool = True
    
    # 对齐任务
    contrastive_loss: bool = True
    itm_loss: bool = True  # Image-Text Matching
    itc_loss: bool = True  # Image-Text Contrastive
    
    # 冻结策略
    freeze_text_encoder: bool = True
    freeze_image_encoder: bool = False
    
    # 训练配置
    epochs: int = 5
    learning_rate: float = 1e-5
    batch_size: int = 128
    warmup_ratio: float = 0.05
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if not any([self.contrastive_loss, self.itm_loss, self.itc_loss]):
            errors.append("At least one loss type must be enabled")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def get_enabled_losses(self) -> List[str]:
        """获取启用的损失函数"""
        losses = []
        if self.contrastive_loss:
            losses.append('contrastive')
        if self.itm_loss:
            losses.append('itm')
        if self.itc_loss:
            losses.append('itc')
        return losses


@dataclass
class InstructionTuningConfig:
    """阶段三：指令微调配置"""
    enabled: bool = True
    
    # 指令类型
    instruction_types: List[str] = field(default_factory=lambda: [
        "image_qa",        # 看图回答
        "video_summary",   # 视频总结
        "ocr_extract",     # OCR抽取
        "multimodal_chat"  # 多模态对话
    ])
    
    # 对话格式
    conversation_format: str = "chatml"  # chatml, llama, alpaca
    max_turns: int = 5
    
    # LoRA配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # 训练配置
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 32
    warmup_ratio: float = 0.03
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.conversation_format not in ["chatml", "llama", "alpaca"]:
            errors.append(f"Invalid conversation_format: {self.conversation_format}")
        if self.lora_r < 1:
            errors.append("lora_r must be positive")
        if self.lora_alpha < 1:
            errors.append("lora_alpha must be positive")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def get_lora_config(self) -> Dict[str, Any]:
        """获取LoRA配置"""
        return {
            'r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
        }


@dataclass
class AlignmentSafetyConfig:
    """阶段四：对齐与安全配置"""
    enabled: bool = True
    
    # RLHF配置
    use_rlhf: bool = True
    reward_model_path: Optional[str] = None
    ppo_epochs: int = 4
    kl_coef: float = 0.1
    
    # DPO配置
    use_dpo: bool = False
    dpo_beta: float = 0.1
    
    # 安全配置
    safety_filter: bool = True
    hallucination_detection: bool = True
    grounding_enforcement: bool = True
    
    # 训练配置
    epochs: int = 1
    learning_rate: float = 1e-6
    batch_size: int = 8
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.use_rlhf and self.use_dpo:
            errors.append("Cannot use both RLHF and DPO")
        if self.kl_coef < 0:
            errors.append("kl_coef cannot be negative")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def get_alignment_method(self) -> str:
        """获取对齐方法"""
        if self.use_rlhf:
            return 'rlhf'
        elif self.use_dpo:
            return 'dpo'
        return 'none'


@dataclass
class FourStageTrainingConfig:
    """四阶段训练配置"""
    modality_pretrain: ModalityPretrainConfig = field(default_factory=ModalityPretrainConfig)
    cross_modal_align: CrossModalAlignTrainConfig = field(default_factory=CrossModalAlignTrainConfig)
    instruction_tuning: InstructionTuningConfig = field(default_factory=InstructionTuningConfig)
    alignment_safety: AlignmentSafetyConfig = field(default_factory=AlignmentSafetyConfig)
    
    # 阶段切换
    auto_stage_switch: bool = True
    stage_save_checkpoint: bool = True
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        
        if self.modality_pretrain.enabled:
            valid, sub_errors = self.modality_pretrain.validate()
            if not valid:
                errors.extend([f"modality_pretrain: {e}" for e in sub_errors])
        
        if self.cross_modal_align.enabled:
            valid, sub_errors = self.cross_modal_align.validate()
            if not valid:
                errors.extend([f"cross_modal_align: {e}" for e in sub_errors])
        
        if self.instruction_tuning.enabled:
            valid, sub_errors = self.instruction_tuning.validate()
            if not valid:
                errors.extend([f"instruction_tuning: {e}" for e in sub_errors])
        
        if self.alignment_safety.enabled:
            valid, sub_errors = self.alignment_safety.validate()
            if not valid:
                errors.extend([f"alignment_safety: {e}" for e in sub_errors])
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'modality_pretrain': self.modality_pretrain.to_dict(),
            'cross_modal_align': self.cross_modal_align.to_dict(),
            'instruction_tuning': self.instruction_tuning.to_dict(),
            'alignment_safety': self.alignment_safety.to_dict(),
            'auto_stage_switch': self.auto_stage_switch,
            'stage_save_checkpoint': self.stage_save_checkpoint,
        }
    
    def get_enabled_stages(self) -> List[TrainingStage]:
        """获取启用的训练阶段"""
        stages = []
        if self.modality_pretrain.enabled:
            stages.append(TrainingStage.MODALITY_PRETRAIN)
        if self.cross_modal_align.enabled:
            stages.append(TrainingStage.CROSS_MODAL_ALIGN)
        if self.instruction_tuning.enabled:
            stages.append(TrainingStage.INSTRUCTION_TUNING)
        if self.alignment_safety.enabled:
            stages.append(TrainingStage.ALIGNMENT_SAFETY)
        return stages
    
    def get_stage_config(self, stage: TrainingStage):
        """获取指定阶段的配置"""
        stage_map = {
            TrainingStage.MODALITY_PRETRAIN: self.modality_pretrain,
            TrainingStage.CROSS_MODAL_ALIGN: self.cross_modal_align,
            TrainingStage.INSTRUCTION_TUNING: self.instruction_tuning,
            TrainingStage.ALIGNMENT_SAFETY: self.alignment_safety,
        }
        return stage_map.get(stage)
    
    def get_total_epochs(self) -> int:
        """获取总训练轮数"""
        total = 0
        if self.modality_pretrain.enabled:
            total += self.modality_pretrain.epochs
        if self.cross_modal_align.enabled:
            total += self.cross_modal_align.epochs
        if self.instruction_tuning.enabled:
            total += self.instruction_tuning.epochs
        if self.alignment_safety.enabled:
            total += self.alignment_safety.epochs
        return total


# ==================== 分布式训练配置 ====================

@dataclass
class DistributedConfig:
    """分布式训练配置"""
    enabled: bool = True
    
    # 并行策略
    data_parallel: bool = True
    tensor_parallel: bool = False
    tensor_parallel_size: int = 1
    pipeline_parallel: bool = False
    pipeline_parallel_size: int = 1
    
    # DeepSpeed配置
    use_deepspeed: bool = True
    deepspeed_stage: int = 2  # ZeRO stage
    offload_optimizer: bool = False
    offload_param: bool = False
    
    # 混合精度
    fp16: bool = True
    bf16: bool = False
    
    # 优化
    flash_attention: bool = True
    gradient_checkpointing: bool = True
    activation_checkpointing: bool = True
    
    # 通信优化
    overlap_comm: bool = True
    reduce_bucket_size: int = 500_000_000
    
    # 动态padding
    dynamic_padding: bool = True
    pad_to_multiple_of: int = 8
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.fp16 and self.bf16:
            errors.append("Cannot use both fp16 and bf16")
        if self.deepspeed_stage not in [0, 1, 2, 3]:
            errors.append(f"Invalid deepspeed_stage: {self.deepspeed_stage}")
        if self.tensor_parallel_size < 1:
            errors.append("tensor_parallel_size must be positive")
        if self.pipeline_parallel_size < 1:
            errors.append("pipeline_parallel_size must be positive")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def get_world_size(self, num_gpus: int) -> int:
        """计算世界大小"""
        world_size = num_gpus
        if self.tensor_parallel:
            world_size = num_gpus // self.tensor_parallel_size
        if self.pipeline_parallel:
            world_size = world_size // self.pipeline_parallel_size
        return max(1, world_size)
    
    def get_precision(self) -> str:
        """获取精度类型"""
        if self.bf16:
            return 'bf16'
        elif self.fp16:
            return 'fp16'
        return 'fp32'
    
    def to_deepspeed_config(self) -> Dict[str, Any]:
        """转换为DeepSpeed配置"""
        config = {
            'train_batch_size': 'auto',
            'gradient_accumulation_steps': 'auto',
            'zero_optimization': {
                'stage': self.deepspeed_stage,
                'offload_optimizer': {
                    'device': 'cpu' if self.offload_optimizer else 'none',
                },
                'offload_param': {
                    'device': 'cpu' if self.offload_param else 'none',
                },
                'overlap_comm': self.overlap_comm,
                'reduce_bucket_size': self.reduce_bucket_size,
            },
        }
        
        if self.fp16:
            config['fp16'] = {'enabled': True}
        elif self.bf16:
            config['bf16'] = {'enabled': True}
        
        return config


# ==================== 推理部署配置 ====================

@dataclass
class InferenceConfig:
    """推理配置"""
    # 服务架构
    encoder_service_separate: bool = True  # 编码器独立部署
    use_embedding_cache: bool = True
    cache_size_mb: int = 4096
    
    # 性能优化
    use_kv_cache: bool = True
    multimodal_kv_cache: bool = True
    token_compression: bool = False
    compression_ratio: float = 0.5
    
    # 图像特征复用
    image_feature_reuse: bool = True
    
    # 批处理
    max_batch_size: int = 32
    dynamic_batching: bool = True
    
    # 量化
    quantization: str = "none"  # none, int8, int4
    
    # 超时配置
    encoder_timeout_ms: int = 5000
    llm_timeout_ms: int = 30000
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.quantization not in ["none", "int8", "int4"]:
            errors.append(f"Invalid quantization: {self.quantization}")
        if self.compression_ratio < 0 or self.compression_ratio > 1:
            errors.append("compression_ratio must be between 0 and 1")
        if self.max_batch_size < 1:
            errors.append("max_batch_size must be positive")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def estimate_memory_mb(self, model_size_gb: float) -> float:
        """估算推理内存需求（MB）"""
        base_memory = model_size_gb * 1024  # GB to MB
        
        if self.quantization == "int8":
            base_memory *= 0.5
        elif self.quantization == "int4":
            base_memory *= 0.25
        
        if self.use_kv_cache:
            base_memory *= 1.2
        
        if self.use_embedding_cache:
            base_memory += self.cache_size_mb
        
        return base_memory


# ==================== 风险对策配置 ====================

@dataclass
class RiskMitigationConfig:
    """风险对策配置"""
    # 模态幻觉对策
    force_grounding: bool = True
    grounding_threshold: float = 0.5
    
    # 延迟优化
    encoder_decoupling: bool = True
    async_encoding: bool = True
    
    # 成本控制
    freeze_large_model: bool = True
    use_efficient_tuning: bool = True  # LoRA, Adapter
    
    # 数据偏置
    data_rebalancing: bool = True
    synthetic_augmentation: bool = True
    rebalance_factor: float = 0.3
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        if self.grounding_threshold < 0 or self.grounding_threshold > 1:
            errors.append("grounding_threshold must be between 0 and 1")
        if self.rebalance_factor < 0 or self.rebalance_factor > 1:
            errors.append("rebalance_factor must be between 0 and 1")
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def get_enabled_mitigations(self) -> List[str]:
        """获取启用的风险缓解措施"""
        mitigations = []
        if self.force_grounding:
            mitigations.append('grounding')
        if self.encoder_decoupling:
            mitigations.append('encoder_decoupling')
        if self.async_encoding:
            mitigations.append('async_encoding')
        if self.freeze_large_model:
            mitigations.append('freeze_large_model')
        if self.use_efficient_tuning:
            mitigations.append('efficient_tuning')
        if self.data_rebalancing:
            mitigations.append('data_rebalancing')
        return mitigations


# ==================== 总配置 ====================

@dataclass
class MultiModalConfig:
    """生产级多模态训练总配置"""
    # 项目配置
    project_name: str = "multimodal_training"
    output_dir: str = "./outputs"
    seed: int = 42
    
    # 模态配置
    modalities: List[ModalityType] = field(default_factory=lambda: [
        ModalityType.TEXT,
        ModalityType.IMAGE
    ])
    
    # 各模块配置
    data_engineering: DataEngineeringConfig = field(default_factory=DataEngineeringConfig)
    encoders: ModalEncodersConfig = field(default_factory=ModalEncodersConfig)
    alignment: CrossModalAlignmentConfig = field(default_factory=CrossModalAlignmentConfig)
    fusion: MultiModalFusionConfig = field(default_factory=MultiModalFusionConfig)
    training: FourStageTrainingConfig = field(default_factory=FourStageTrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    risk_mitigation: RiskMitigationConfig = field(default_factory=RiskMitigationConfig)
    
    # 日志和监控
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    use_wandb: bool = False
    use_tensorboard: bool = True
    
    # 元数据
    _created_at: Optional[str] = field(default=None, repr=False)
    _version: str = field(default="1.0.0", repr=False)
    
    def __post_init__(self):
        """初始化后处理"""
        if self._created_at is None:
            self._created_at = datetime.now().isoformat()
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        
        # 验证子配置
        valid, sub_errors = self.data_engineering.validate()
        if not valid:
            errors.extend([f"data_engineering: {e}" for e in sub_errors])
        
        valid, sub_errors = self.encoders.validate()
        if not valid:
            errors.extend([f"encoders: {e}" for e in sub_errors])
        
        valid, sub_errors = self.alignment.validate()
        if not valid:
            errors.extend([f"alignment: {e}" for e in sub_errors])
        
        valid, sub_errors = self.fusion.validate()
        if not valid:
            errors.extend([f"fusion: {e}" for e in sub_errors])
        
        valid, sub_errors = self.training.validate()
        if not valid:
            errors.extend([f"training: {e}" for e in sub_errors])
        
        valid, sub_errors = self.distributed.validate()
        if not valid:
            errors.extend([f"distributed: {e}" for e in sub_errors])
        
        valid, sub_errors = self.inference.validate()
        if not valid:
            errors.extend([f"inference: {e}" for e in sub_errors])
        
        valid, sub_errors = self.risk_mitigation.validate()
        if not valid:
            errors.extend([f"risk_mitigation: {e}" for e in sub_errors])
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'project_name': self.project_name,
            'output_dir': self.output_dir,
            'seed': self.seed,
            'modalities': [m.value for m in self.modalities],
            'data_engineering': self.data_engineering.to_dict(),
            'encoders': self.encoders.to_dict(),
            'alignment': self.alignment.to_dict(),
            'fusion': self.fusion.to_dict(),
            'training': self.training.to_dict(),
            'distributed': self.distributed.to_dict(),
            'inference': self.inference.to_dict(),
            'risk_mitigation': self.risk_mitigation.to_dict(),
            'logging_steps': self.logging_steps,
            'eval_steps': self.eval_steps,
            'save_steps': self.save_steps,
            'use_wandb': self.use_wandb,
            'use_tensorboard': self.use_tensorboard,
            '_created_at': self._created_at,
            '_version': self._version,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MultiModalConfig':
        """从字典创建"""
        # 处理模态类型
        if 'modalities' in config_dict:
            config_dict['modalities'] = [
                ModalityType.from_string(m) if isinstance(m, str) else m
                for m in config_dict['modalities']
            ]
        return cls(**config_dict)
    
    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MultiModalConfig':
        """从JSON字符串创建"""
        return cls.from_dict(json.loads(json_str))
    
    def to_yaml(self) -> str:
        """转换为YAML字符串"""
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'MultiModalConfig':
        """从YAML字符串创建"""
        return cls.from_dict(yaml.safe_load(yaml_str))
    
    def save(self, path: str) -> None:
        """保存配置到文件"""
        path_obj = Path(path)
        
        if path.endswith('.json'):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
        elif path.endswith(('.yaml', '.yml')):
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.to_yaml())
        else:
            # 默认保存为JSON
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.to_json())
        
        logger.info(f"Config saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MultiModalConfig':
        """从文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if path.endswith('.json'):
            return cls.from_json(content)
        elif path.endswith(('.yaml', '.yml')):
            return cls.from_yaml(content)
        else:
            # 尝试JSON
            try:
                return cls.from_json(content)
            except:
                return cls.from_yaml(content)
    
    def copy(self) -> 'MultiModalConfig':
        """深拷贝配置"""
        return copy.deepcopy(self)
    
    def merge(self, other: 'MultiModalConfig') -> 'MultiModalConfig':
        """合并另一个配置"""
        new_config = self.copy()
        # 简单合并：用other的非默认值覆盖
        other_dict = other.to_dict()
        for key, value in other_dict.items():
            if value is not None and not key.startswith('_'):
                setattr(new_config, key, value)
        return new_config
    
    def get_config_hash(self) -> str:
        """获取配置哈希（用于缓存）"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def summary(self) -> str:
        """生成配置摘要"""
        lines = [
            f"Project: {self.project_name}",
            f"Modalities: {[m.value for m in self.modalities]}",
            f"Training stages: {[s.value for s in self.training.get_enabled_stages()]}",
            f"Total epochs: {self.training.get_total_epochs()}",
            f"Distributed: {self.distributed.enabled}, DeepSpeed stage: {self.distributed.deepspeed_stage}",
            f"Fusion: {self.fusion.stage.value} / {self.fusion.method.value}",
            f"Alignment: {self.alignment.method.value}",
        ]
        return "\n".join(lines)
    
    def print_summary(self) -> None:
        """打印配置摘要"""
        print("\n" + "="*60)
        print("MultiModal Training Configuration")
        print("="*60)
        print(self.summary())
        print("="*60)
    
    def estimate_memory_requirements(self, batch_size: int = 1) -> Dict[str, float]:
        """估算内存需求（MB）"""
        memory = {}
        
        # 编码器内存
        memory['text_encoder'] = self.encoders.text.estimate_memory_mb(batch_size)
        memory['image_encoder'] = self.encoders.image.estimate_memory_mb(batch_size)
        
        # 融合模块内存
        fusion_params = self.fusion.estimate_param_count()
        memory['fusion_module'] = fusion_params * 4 / (1024 * 1024)  # FP32
        
        # 总内存
        memory['total'] = sum(memory.values())
        
        # 考虑混合精度
        if self.distributed.fp16 or self.distributed.bf16:
            memory['total'] *= 0.6  # 大约节省40%
        
        return memory
    
    def recommend_batch_size(self, gpu_memory_gb: float) -> int:
        """推荐batch size"""
        # 估算单样本内存（MB）
        sample_memory = self.estimate_memory_requirements(1)['total']
        
        # 可用内存（留20%余量）
        available_mb = gpu_memory_gb * 1024 * 0.8
        
        # 推荐batch size
        batch_size = int(available_mb / sample_memory)
        
        # 向下取到2的幂次
        power = 0
        while 2 ** (power + 1) <= batch_size:
            power += 1
        
        return max(1, 2 ** power)


# ==================== 预设配置 ====================

class MultiModalPresets:
    """生产级预设配置"""
    
    # 预设描述
    _descriptions = {
        'image_text_base': 'Basic image-text multimodal configuration',
        'video_understanding': 'Video understanding with audio support',
        'industrial_multimodal': 'Industrial multimodal with time series',
        'large_scale_training': 'Large scale distributed training',
        'efficient_training': 'Memory-efficient training with LoRA',
        'inference_optimized': 'Optimized for inference speed',
    }
    
    @staticmethod
    def image_text_base() -> MultiModalConfig:
        """图文基础配置"""
        return MultiModalConfig(
            project_name="image_text_base",
            modalities=[ModalityType.TEXT, ModalityType.IMAGE]
        )
    
    @staticmethod
    def video_understanding() -> MultiModalConfig:
        """视频理解配置"""
        config = MultiModalConfig(
            project_name="video_understanding",
            modalities=[ModalityType.TEXT, ModalityType.VIDEO, ModalityType.AUDIO]
        )
        config.encoders.video.num_frames = 16
        config.encoders.audio.encoder_type = EncoderType.WHISPER
        return config
    
    @staticmethod
    def industrial_multimodal() -> MultiModalConfig:
        """工业多模态配置"""
        config = MultiModalConfig(
            project_name="industrial_multimodal",
            modalities=[
                ModalityType.TEXT,
                ModalityType.IMAGE,
                ModalityType.TIME_SERIES,
                ModalityType.TABLE
            ]
        )
        config.fusion.stage = FusionStage.MIDDLE
        config.fusion.method = FusionMethod.CROSS_ATTENTION
        config.training.instruction_tuning.instruction_types = [
            "defect_detection",
            "fault_prediction",
            "process_optimization"
        ]
        return config
    
    @staticmethod
    def large_scale_training() -> MultiModalConfig:
        """大规模训练配置"""
        config = MultiModalConfig(
            project_name="large_scale_training"
        )
        config.distributed.use_deepspeed = True
        config.distributed.deepspeed_stage = 3
        config.distributed.tensor_parallel = True
        config.distributed.tensor_parallel_size = 4
        config.distributed.flash_attention = True
        return config
    
    @staticmethod
    def efficient_training() -> MultiModalConfig:
        """高效训练配置（省内存）"""
        config = MultiModalConfig(
            project_name="efficient_training"
        )
        config.training.instruction_tuning.use_lora = True
        config.training.instruction_tuning.lora_r = 8
        config.distributed.gradient_checkpointing = True
        config.distributed.activation_checkpointing = True
        config.alignment.freeze_text_encoder = True
        config.alignment.freeze_vision_encoder = True
        return config
    
    @staticmethod
    def inference_optimized() -> MultiModalConfig:
        """推理优化配置"""
        config = MultiModalConfig(
            project_name="inference_optimized"
        )
        config.inference.quantization = "int8"
        config.inference.use_kv_cache = True
        config.inference.dynamic_batching = True
        config.inference.encoder_service_separate = True
        config.inference.image_feature_reuse = True
        return config
    
    @staticmethod
    def blip2_style() -> MultiModalConfig:
        """BLIP-2风格配置"""
        config = MultiModalConfig(
            project_name="blip2_style"
        )
        config.fusion.method = FusionMethod.QFORMER
        config.fusion.qformer.num_query_tokens = 32
        config.fusion.qformer.num_layers = 6
        config.alignment.method = AlignmentMethod.CONTRASTIVE
        return config
    
    @staticmethod
    def flamingo_style() -> MultiModalConfig:
        """Flamingo风格配置"""
        config = MultiModalConfig(
            project_name="flamingo_style"
        )
        config.fusion.method = FusionMethod.FLAMINGO
        config.fusion.stage = FusionStage.MIDDLE
        config.alignment.freeze_text_encoder = False
        return config
    
    @classmethod
    def get(cls, preset_name: str) -> Optional[MultiModalConfig]:
        """根据名称获取预设"""
        preset_map = {
            'image_text_base': cls.image_text_base,
            'video_understanding': cls.video_understanding,
            'industrial_multimodal': cls.industrial_multimodal,
            'large_scale_training': cls.large_scale_training,
            'efficient_training': cls.efficient_training,
            'inference_optimized': cls.inference_optimized,
            'blip2_style': cls.blip2_style,
            'flamingo_style': cls.flamingo_style,
        }
        
        if preset_name in preset_map:
            return preset_map[preset_name]()
        return None
    
    @classmethod
    def list_presets(cls) -> List[str]:
        """列出所有预设"""
        return [
            'image_text_base', 'video_understanding', 'industrial_multimodal',
            'large_scale_training', 'efficient_training', 'inference_optimized',
            'blip2_style', 'flamingo_style'
        ]
    
    @classmethod
    def print_presets(cls) -> None:
        """打印所有预设"""
        print("\n" + "="*60)
        print("MultiModal Training Presets")
        print("="*60)
        
        for preset in cls.list_presets():
            desc = cls._descriptions.get(preset, "")
            print(f"  {preset:<25} {desc}")
        
        print("="*60)


# ==================== 工具函数 ====================

def create_config(
    modalities: List[str],
    **kwargs
) -> MultiModalConfig:
    """
    创建配置
    
    Args:
        modalities: 模态列表
        **kwargs: 其他配置参数
        
    Returns:
        配置实例
    """
    modal_types = [ModalityType.from_string(m) for m in modalities]
    return MultiModalConfig(modalities=modal_types, **kwargs)


def validate_config(config: MultiModalConfig) -> Tuple[bool, List[str]]:
    """
    验证配置
    
    Args:
        config: 配置实例
        
    Returns:
        (是否有效, 错误列表)
    """
    return config.validate()


def print_config_summary(config: MultiModalConfig) -> None:
    """打印配置摘要"""
    config.print_summary()


def compare_configs(
    config1: MultiModalConfig,
    config2: MultiModalConfig
) -> Dict[str, Any]:
    """
    比较两个配置
    
    Args:
        config1: 配置1
        config2: 配置2
        
    Returns:
        差异字典
    """
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()
    
    differences = {}
    
    def compare_dicts(d1, d2, prefix=""):
        for key in set(d1.keys()) | set(d2.keys()):
            full_key = f"{prefix}.{key}" if prefix else key
            
            if key not in d1:
                differences[full_key] = {'config1': None, 'config2': d2[key]}
            elif key not in d2:
                differences[full_key] = {'config1': d1[key], 'config2': None}
            elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                compare_dicts(d1[key], d2[key], full_key)
            elif d1[key] != d2[key]:
                differences[full_key] = {'config1': d1[key], 'config2': d2[key]}
    
    compare_dicts(dict1, dict2)
    return differences


def print_config_comparison(
    config1: MultiModalConfig,
    config2: MultiModalConfig,
    name1: str = "Config1",
    name2: str = "Config2"
) -> None:
    """打印配置比较"""
    differences = compare_configs(config1, config2)
    
    print("\n" + "="*80)
    print(f"Configuration Comparison: {name1} vs {name2}")
    print("="*80)
    
    if not differences:
        print("\nConfigurations are identical.")
    else:
        print(f"\n{'Key':<40} {name1:<18} {name2:<18}")
        print("-"*80)
        
        for key, values in sorted(differences.items()):
            v1 = str(values['config1'])[:15]
            v2 = str(values['config2'])[:15]
            print(f"{key:<40} {v1:<18} {v2:<18}")
    
    print("="*80)


def recommend_config(
    task: str,
    num_modalities: int = 2,
    gpu_memory_gb: float = 24.0,
    quality_priority: bool = True
) -> MultiModalConfig:
    """
    推荐配置
    
    Args:
        task: 任务类型 (image_text, video, industrial)
        num_modalities: 模态数量
        gpu_memory_gb: GPU内存(GB)
        quality_priority: 是否优先质量
        
    Returns:
        推荐配置
    """
    # 选择基础预设
    if task == 'video':
        config = MultiModalPresets.video_understanding()
    elif task == 'industrial':
        config = MultiModalPresets.industrial_multimodal()
    else:
        config = MultiModalPresets.image_text_base()
    
    # 根据GPU内存调整
    if gpu_memory_gb < 16:
        config.training.instruction_tuning.use_lora = True
        config.training.instruction_tuning.lora_r = 8
        config.distributed.gradient_checkpointing = True
        config.training.instruction_tuning.batch_size = 8
    elif gpu_memory_gb < 24:
        config.training.instruction_tuning.batch_size = 16
    else:
        config.training.instruction_tuning.batch_size = 32
    
    # 推荐融合方法
    config.fusion.method = FusionMethod.recommend(num_modalities, quality_priority)
    
    # 推荐对齐方法
    config.alignment.method = AlignmentMethod.recommend(num_modalities)
    
    return config


def estimate_training_resources(
    config: MultiModalConfig,
    num_samples: int,
    num_gpus: int = 1
) -> Dict[str, Any]:
    """
    估算训练资源需求
    
    Args:
        config: 配置
        num_samples: 样本数
        num_gpus: GPU数量
        
    Returns:
        资源估算
    """
    batch_size = config.training.instruction_tuning.batch_size
    total_epochs = config.training.get_total_epochs()
    
    # 估算训练时间
    samples_per_second = 10 * num_gpus  # 估算
    total_samples = num_samples * total_epochs
    training_hours = total_samples / samples_per_second / 3600
    
    # 估算内存
    memory = config.estimate_memory_requirements(batch_size)
    
    return {
        'total_epochs': total_epochs,
        'total_samples': total_samples,
        'estimated_hours': training_hours,
        'memory_per_gpu_mb': memory['total'],
        'recommended_gpus': max(1, int(memory['total'] / 20000)),  # 假设20GB可用
        'batch_size': batch_size,
    }


def print_resource_estimate(
    config: MultiModalConfig,
    num_samples: int,
    num_gpus: int = 1
) -> None:
    """打印资源估算"""
    resources = estimate_training_resources(config, num_samples, num_gpus)
    
    print("\n" + "="*60)
    print("Training Resource Estimate")
    print("="*60)
    
    print(f"\nDataset size: {num_samples:,} samples")
    print(f"GPUs: {num_gpus}")
    print(f"Batch size: {resources['batch_size']}")
    print(f"Total epochs: {resources['total_epochs']}")
    print(f"Total samples: {resources['total_samples']:,}")
    print(f"Estimated time: {resources['estimated_hours']:.1f} hours")
    print(f"Memory per GPU: {resources['memory_per_gpu_mb']:.0f} MB")
    print(f"Recommended GPUs: {resources['recommended_gpus']}")
    
    print("="*60)



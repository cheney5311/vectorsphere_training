# -*- coding: utf-8 -*-
"""
生产级多模态训练配置

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
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

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


class AlignmentMethod(Enum):
    """对齐方法"""
    CONTRASTIVE = "contrastive"      # 对比学习（CLIP风格）
    EXPLICIT_ALIGN = "explicit"       # 显式对齐
    CROSS_ATTENTION = "cross_attention"  # 交叉注意力
    OPTIMAL_TRANSPORT = "optimal_transport"  # 最优传输
    KNOWLEDGE_DISTILL = "knowledge_distill"  # 知识蒸馏对齐


class FusionStage(Enum):
    """融合阶段"""
    EARLY = "early"          # 早期融合（特征层面）
    MIDDLE = "middle"        # 中期融合（语义层面）
    LATE = "late"            # 后期融合（决策层面）
    ADAPTIVE = "adaptive"    # 自适应融合


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


class TrainingStage(Enum):
    """训练阶段"""
    MODALITY_PRETRAIN = "modality_pretrain"      # 阶段一：模态预训练
    CROSS_MODAL_ALIGN = "cross_modal_align"      # 阶段二：跨模态对齐
    INSTRUCTION_TUNING = "instruction_tuning"    # 阶段三：指令微调
    ALIGNMENT_SAFETY = "alignment_safety"        # 阶段四：对齐与安全


class DataSourceType(Enum):
    """数据来源类型"""
    WEB_IMAGE_TEXT = "web_image_text"    # 图文网页
    VIDEO_ASR = "video_asr"              # 视频+ASR
    OCR_DOCUMENT = "ocr_document"        # OCR文档
    USER_LOG = "user_log"                # 用户日志
    SYNTHETIC = "synthetic"              # 合成数据


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


@dataclass
class AudioEncoderConfig:
    """音频编码器配置"""
    encoder_type: EncoderType = EncoderType.WHISPER
    model_name: str = "whisper-base"
    hidden_size: int = 512
    sample_rate: int = 16000
    max_duration: float = 30.0
    freeze: bool = True


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


@dataclass
class VideoEncoderConfig:
    """视频编码器配置"""
    encoder_type: EncoderType = EncoderType.VIDEO_SWIN
    model_name: str = "video-swin-base"
    hidden_size: int = 768
    num_frames: int = 8
    frame_size: int = 224
    freeze: bool = True


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


# ==================== 对齐配置 ====================

@dataclass
class ContrastiveLearningConfig:
    """对比学习配置"""
    temperature: float = 0.07
    loss_type: str = "info_nce"  # info_nce, clip, simclr
    hard_negative_mining: bool = True
    hard_negative_ratio: float = 0.2
    in_batch_negatives: bool = True


@dataclass
class ExplicitAlignConfig:
    """显式对齐配置"""
    method: str = "attention"  # attention, mlp, linear
    hidden_size: int = 768
    num_layers: int = 2
    align_loss_weight: float = 1.0


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


# ==================== 融合配置 ====================

@dataclass
class EarlyFusionConfig:
    """早期融合配置"""
    method: FusionMethod = FusionMethod.CONCAT
    concat_dim: int = 768


@dataclass
class MiddleFusionConfig:
    """中期融合配置"""
    method: FusionMethod = FusionMethod.CROSS_ATTENTION
    num_layers: int = 6
    num_heads: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    dropout: float = 0.1


@dataclass
class LateFusionConfig:
    """后期融合配置"""
    method: FusionMethod = FusionMethod.ATTENTION
    num_heads: int = 8
    fusion_dim: int = 768


@dataclass
class QFormerConfig:
    """Q-Former配置（BLIP-2风格）"""
    num_query_tokens: int = 32
    num_layers: int = 6
    num_heads: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    cross_attention_freq: int = 2


@dataclass
class PerceiverConfig:
    """Perceiver配置"""
    num_latents: int = 256
    latent_dim: int = 512
    num_self_attention_layers: int = 6
    num_cross_attention_layers: int = 2
    num_heads: int = 8


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


# ==================== 总配置 ====================

@dataclass
class ProductionMultiModalConfig:
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
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        import dataclasses
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ProductionMultiModalConfig':
        """从字典创建"""
        return cls(**config_dict)


# ==================== 预设配置 ====================

class ProductionMultiModalPresets:
    """生产级预设配置"""
    
    @staticmethod
    def image_text_base() -> ProductionMultiModalConfig:
        """图文基础配置"""
        return ProductionMultiModalConfig(
            project_name="image_text_base",
            modalities=[ModalityType.TEXT, ModalityType.IMAGE]
        )
    
    @staticmethod
    def video_understanding() -> ProductionMultiModalConfig:
        """视频理解配置"""
        config = ProductionMultiModalConfig(
            project_name="video_understanding",
            modalities=[ModalityType.TEXT, ModalityType.VIDEO, ModalityType.AUDIO]
        )
        config.encoders.video.num_frames = 16
        config.encoders.audio.encoder_type = EncoderType.WHISPER
        return config
    
    @staticmethod
    def industrial_multimodal() -> ProductionMultiModalConfig:
        """工业多模态配置"""
        config = ProductionMultiModalConfig(
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
    def large_scale_training() -> ProductionMultiModalConfig:
        """大规模训练配置"""
        config = ProductionMultiModalConfig(
            project_name="large_scale_training"
        )
        config.distributed.use_deepspeed = True
        config.distributed.deepspeed_stage = 3
        config.distributed.tensor_parallel = True
        config.distributed.tensor_parallel_size = 4
        config.distributed.flash_attention = True
        return config


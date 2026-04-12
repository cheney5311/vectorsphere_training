# 生产级多模态训练模块

## 概述

本模块提供完整的生产级多模态训练能力，支持从数据工程到模型部署的全流程。

## 架构设计

```
┌──────────────────────────────────────────────────────────────────┐
│                       生产级多模态训练架构                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    1. 数据工程层                             │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │ │
│  │  │ 数据去重 │  │ 噪声过滤 │  │ 一致性  │  │ 版权合规扫描   │  │ │
│  │  │ pHash   │  │ NSFW    │  │ 校验    │  │                 │  │ │
│  │  │ MinHash │  │ 质量    │  │         │  │                 │  │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                   2. 模态编码器层                            │ │
│  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐         │ │
│  │  │ 文本  │ │ 图像  │ │ 音频  │ │ 时序  │ │ 视频  │          │ │
│  │  │ BERT  │ │ ViT   │ │Whisper│ │Trans- │ │Video- │          │ │
│  │  │ LLaMA │ │ CLIP  │ │Wav2Vec│ │former │ │Swin   │          │ │
│  │  └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘         │ │
│  │      └─────────┴─────────┴─────────┴─────────┘              │ │
│  │                          │                                   │ │
│  │              ┌───────────▼───────────┐                      │ │
│  │              │     统一投影层         │                      │ │
│  │              └───────────────────────┘                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 3. 跨模态对齐层                              │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────────┐│ │
│  │  │ 对比学习    │ │ 显式对齐    │ │ 交叉注意力 / 最优传输   ││ │
│  │  │ InfoNCE    │ │ MLP/Attn   │ │ Cross-Attention/OT     ││ │
│  │  └─────────────┘ └─────────────┘ └─────────────────────────┘│ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  4. 多模态融合层                             │ │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐               │ │
│  │  │ 早期融合   │  │ 中期融合   │  │ 后期融合   │               │ │
│  │  │ Concat    │  │ Cross-Attn│  │ Attention │               │ │
│  │  └───────────┘  └───────────┘  └───────────┘               │ │
│  │        ┌──────────────┬──────────────┐                     │ │
│  │        │              │              │                     │ │
│  │  ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐               │ │
│  │  │ Q-Former  │  │ Perceiver │  │ Flamingo  │               │ │
│  │  └───────────┘  └───────────┘  └───────────┘               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 5. 四阶段训练流程                            │ │
│  │  ┌───────────────────────────────────────────────────────┐  │ │
│  │  │ 阶段一：模态预训练 → 阶段二：跨模态对齐 →              │  │ │
│  │  │ 阶段三：指令微调 → 阶段四：对齐与安全                  │  │ │
│  │  └───────────────────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  6. 推理与部署层                             │ │
│  │  ┌───────────┐  ┌───────────┐  ┌─────────────────────────┐  │ │
│  │  │ Encoder   │  │ LLM推理   │  │ 性能优化                 │  │ │
│  │  │ 独立部署   │  │ 服务     │  │ KV Cache/Token压缩      │  │ │
│  │  └───────────┘  └───────────┘  └─────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 数据工程 (data_engineering.py)

```python
from backend.modules.training.multimodal import (
    MultiModalDataPipeline,
    DataEngineeringConfig,
    MultiModalSample
)

# 创建数据管道
config = DataEngineeringConfig(
    deduplication=DataDeduplicationConfig(
        enabled=True,
        method="perceptual_hash"
    ),
    filtering=DataFilterConfig(
        enabled=True,
        nsfw_filter=True,
        consistency_check=True
    )
)

pipeline = MultiModalDataPipeline(config)

# 处理数据
samples = [
    MultiModalSample(
        sample_id="001",
        modalities={
            'text': "这是一张猫的图片",
            'image': image_tensor
        }
    ),
    # ... 更多样本
]

processed_samples = pipeline.process(samples)
print(pipeline.get_statistics())
```

### 2. 模态编码器 (encoders.py)

```python
from backend.modules.training.multimodal import (
    ModalityEncoderFactory,
    ModalityType,
    TextEncoderConfig,
    ImageEncoderConfig
)

# 创建文本编码器
text_config = TextEncoderConfig(
    encoder_type=EncoderType.BERT,
    model_name="bert-base-chinese",
    hidden_size=768
)
text_encoder = ModalityEncoderFactory.create_encoder(ModalityType.TEXT, text_config)

# 创建图像编码器
image_config = ImageEncoderConfig(
    encoder_type=EncoderType.VIT,
    model_name="vit-base-patch16-224",
    hidden_size=768
)
image_encoder = ModalityEncoderFactory.create_encoder(ModalityType.IMAGE, image_config)

# 编码
text_features = text_encoder({'input_ids': input_ids, 'attention_mask': mask})
image_features = image_encoder({'pixel_values': images})
```

### 3. 跨模态对齐 (alignment.py)

```python
from backend.modules.training.multimodal import (
    CrossModalAligner,
    CrossModalAlignmentConfig,
    AlignmentMethod
)

# 创建对齐器
align_config = CrossModalAlignmentConfig(
    method=AlignmentMethod.CONTRASTIVE,
    projection_dim=512
)
aligner = CrossModalAligner(align_config, embed_dim=768)

# 对齐
features = {
    'text': text_features,
    'image': image_features
}
aligned_features, align_loss, metrics = aligner(features, compute_loss=True)

print(f"Alignment loss: {align_loss.item()}")
print(f"Contrastive accuracy: {metrics['contrastive_acc_a']:.2%}")
```

### 4. 多模态融合 (fusion.py)

```python
from backend.modules.training.multimodal import (
    MultiModalFuser,
    MultiModalFusionConfig,
    FusionStage,
    FusionMethod
)

# 创建融合器
fusion_config = MultiModalFusionConfig(
    stage=FusionStage.MIDDLE,
    method=FusionMethod.CROSS_ATTENTION,
    output_dim=768
)
fuser = MultiModalFuser(fusion_config, modality_dims={'text': 768, 'image': 768})

# 融合
fused_features = fuser(aligned_features)  # [batch, 768]
```

### 5. 四阶段训练 (trainer.py)

```python
from backend.modules.training.multimodal import (
    MultiModalConfig,
    MultiModalTrainer,
    create_multimodal_trainer,
    run_four_stage_training
)

# 创建配置
config = MultiModalConfig(
    project_name="multimodal_v1",
    modalities=[ModalityType.TEXT, ModalityType.IMAGE],
    training=FourStageTrainingConfig(
        modality_pretrain=ModalityPretrainConfig(
            enabled=True,
            epochs=10,
            learning_rate=1e-4
        ),
        cross_modal_align=CrossModalAlignTrainConfig(
            enabled=True,
            epochs=5,
            freeze_text_encoder=True
        ),
        instruction_tuning=InstructionTuningConfig(
            enabled=True,
            use_lora=True,
            epochs=3
        ),
        alignment_safety=AlignmentSafetyConfig(
            enabled=True,
            use_rlhf=True
        )
    )
)

# 创建训练器并运行
trainer = create_multimodal_trainer(config, train_dataloader, eval_dataloader)
state = trainer.train()

print(f"Training completed. Final stage: {state.stage}")
```

## 四阶段训练流程

### 阶段一：模态预训练 (Modality Pretraining)

**目标**：让模型学会"看"和"听"

```python
# 配置
pretrain_config = ModalityPretrainConfig(
    text_pretrain=True,
    text_task="mlm",        # Masked Language Modeling
    text_mask_ratio=0.15,
    
    image_pretrain=True,
    image_task="mae",       # Masked Autoencoder
    image_mask_ratio=0.75,
    
    epochs=10,
    learning_rate=1e-4
)
```

**支持的任务**：
- 文本：MLM、CLM、Span Prediction
- 图像：MAE、CLIP、DINO

### 阶段二：跨模态对齐 (Cross-Modal Alignment)

**目标**：对齐不同模态的语义空间

```python
# 配置
align_config = CrossModalAlignTrainConfig(
    contrastive_loss=True,   # 对比学习损失
    itm_loss=True,           # Image-Text Matching
    itc_loss=True,           # Image-Text Contrastive
    
    freeze_text_encoder=True,
    freeze_image_encoder=False,
    
    epochs=5,
    learning_rate=1e-5
)
```

**对齐方法**：
- 对比学习（CLIP风格）
- 显式对齐
- 交叉注意力
- 最优传输

### 阶段三：指令微调 (Instruction Tuning)

**目标**：让模型理解多模态指令

```python
# 配置
instruct_config = InstructionTuningConfig(
    instruction_types=[
        "image_qa",         # 看图回答
        "video_summary",    # 视频总结
        "ocr_extract",      # OCR抽取
        "multimodal_chat"   # 多模态对话
    ],
    
    conversation_format="chatml",
    use_lora=True,
    lora_r=16,
    
    epochs=3,
    learning_rate=2e-5
)
```

### 阶段四：对齐与安全 (Alignment & Safety)

**目标**：确保输出安全可靠

```python
# 配置
safety_config = AlignmentSafetyConfig(
    use_rlhf=True,
    ppo_epochs=4,
    kl_coef=0.1,
    
    safety_filter=True,
    hallucination_detection=True,
    grounding_enforcement=True,  # 强制Grounding
    
    epochs=1,
    learning_rate=1e-6
)
```

## 融合策略详解

### 早期融合 (Early Fusion)

在特征层面直接融合，适合模态间强相关的场景。

```python
# 特征拼接
fused = torch.cat([text_feat, image_feat], dim=-1)
fused = projection(fused)
```

### 中期融合 (Middle Fusion)

在语义层面进行交互，是最常用的方式。

```python
# 交叉注意力融合
for layer in cross_attention_layers:
    text_feat = layer(text_feat, image_feat)
    image_feat = layer(image_feat, text_feat)
```

### 后期融合 (Late Fusion)

在决策层面聚合，适合多任务场景。

```python
# 注意力融合
stacked = torch.stack([text_pred, image_pred], dim=1)
attn_weights = attention_net(stacked)
fused = (stacked * attn_weights).sum(dim=1)
```

### Q-Former (BLIP-2风格)

使用可学习的查询tokens高效提取信息。

```python
qformer = QFormer(QFormerConfig(
    num_query_tokens=32,
    num_layers=6,
    cross_attention_freq=2
))
query_output = qformer(image_features)
```

### Perceiver

使用潜在空间处理任意模态组合。

```python
perceiver = PerceiverFusion(PerceiverConfig(
    num_latents=256,
    latent_dim=512,
    num_self_attention_layers=6
))
fused = perceiver(multimodal_features)
```

## 分布式训练

```python
# 配置
distributed_config = DistributedConfig(
    enabled=True,
    
    # 并行策略
    data_parallel=True,
    tensor_parallel=True,
    tensor_parallel_size=4,
    
    # DeepSpeed
    use_deepspeed=True,
    deepspeed_stage=3,      # ZeRO Stage 3
    offload_optimizer=True,
    
    # 优化
    fp16=True,
    flash_attention=True,
    gradient_checkpointing=True
)
```

## 推理优化

```python
# 配置
inference_config = InferenceConfig(
    encoder_service_separate=True,  # 编码器独立部署
    use_embedding_cache=True,       # Embedding缓存
    cache_size_mb=4096,
    
    use_kv_cache=True,
    multimodal_kv_cache=True,
    token_compression=True,
    compression_ratio=0.5,
    
    quantization="int8"
)
```

## 风险对策

| 风险 | 对策 | 实现 |
|------|------|------|
| 模态幻觉 | 强制Grounding | `grounding_enforcement=True` |
| 延迟过高 | 编码器解耦 | `encoder_service_separate=True` |
| 成本不可控 | 冻结大模型 | `freeze_large_model=True` |
| 数据偏置 | 合成+再平衡 | `data_rebalancing=True` |

## 预设配置

```python
from backend.modules.training.multimodal import MultiModalPresets

# 图文基础
config = MultiModalPresets.image_text_base()

# 视频理解
config = MultiModalPresets.video_understanding()

# 工业多模态
config = MultiModalPresets.industrial_multimodal()

# 大规模训练
config = MultiModalPresets.large_scale_training()
```

## 最佳实践

1. **数据质量优先**：数据质量的收益远大于模型结构优化
2. **分阶段训练**：按顺序执行四阶段，每阶段保存检查点
3. **冻结策略**：对齐阶段冻结文本编码器，控制计算成本
4. **混合精度**：使用FP16/BF16减少显存占用
5. **梯度检查点**：大模型必须开启gradient_checkpointing
6. **监控指标**：关注对比学习准确率、模态一致性分数

## 完整示例

```python
from backend.modules.training.multimodal import (
    MultiModalConfig,
    ModalityType,
    run_four_stage_training,
    MultiModalDataPipeline,
    DataEngineeringConfig
)
from torch.utils.data import DataLoader

# 1. 数据预处理
data_config = DataEngineeringConfig()
pipeline = MultiModalDataPipeline(data_config)
processed_data = pipeline.process(raw_samples)

# 2. 创建DataLoader
train_loader = DataLoader(processed_data, batch_size=32, shuffle=True)
eval_loader = DataLoader(eval_data, batch_size=32)

# 3. 配置训练
config = MultiModalConfig(
    project_name="my_multimodal_model",
    modalities=[ModalityType.TEXT, ModalityType.IMAGE],
    output_dir="./outputs"
)

# 4. 运行四阶段训练
state = run_four_stage_training(
    config=config,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    callbacks=[
        lambda stage, epoch, loss: print(f"{stage}: Epoch {epoch}, Loss: {loss:.4f}")
    ]
)

print(f"Training completed!")
print(f"Stage metrics: {state.stage_metrics}")
```

## 文件结构

```
backend/modules/training/multimodal/
├── __init__.py                       # 模块导出
├── multimodal_config.py              # 基础配置
├── multimodal_trainer.py             # 基础训练器
├── multimodal_config.py   # 生产级配置
├── encoders.py            # 模态编码器
├── alignment.py           # 跨模态对齐
├── fusion.py              # 多模态融合
├── trainer.py             # 四阶段训练器
├── data_engineering.py    # 数据工程
├── MULTIMODAL_README.md              # 基础文档
└── PRODUCTION_MULTIMODAL_README.md   # 本文档
```

## 更新日志

- 2024-01-08: 初始版本，支持四阶段训练流程
- 支持模态：文本、图像、音频、时序、视频
- 支持融合：早期/中期/后期融合、Q-Former、Perceiver
- 支持对齐：对比学习、显式对齐、交叉注意力、最优传输


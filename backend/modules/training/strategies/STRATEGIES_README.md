# 训练策略层架构文档

## 概述

本文档详细说明 `backend/modules/training/strategies` 策略层与 `backend/lib` 六层架构底层模块的调用关系，以及各训练模式的配置和使用方法。

## 架构层次

```
┌─────────────────────────────────────────────────────────────────┐
│ launcher/training_launcher.py (统一入口)                         │
│   └── 调用 orchestrator, strategies 等模块                       │
├─────────────────────────────────────────────────────────────────┤
│ orchestrator/unified_orchestrator.py (编排层)                    │
│   └── 三阶段训练流程控制                                         │
├─────────────────────────────────────────────────────────────────┤
│ strategies/* (策略层) ←── 本模块                                 │
│   ├── IndustryScenarioStrategy → 全部六层                        │
│   ├── ThreeStageStrategy → 编排层 + 策略层 + 损失层 + 硬件层     │
│   ├── ScenarioStrategy → 策略层 + 损失层                         │
│   ├── DistributedStrategy → 分布式层 + 硬件层                    │
│   ├── DistillationStrategy → 策略层 + 损失层                     │
│   ├── MultiModalStrategy → 适配器层 + 损失层                     │
│   └── StandardTrainingStrategy → 策略层                          │
├─────────────────────────────────────────────────────────────────┤
│ backend/lib/losses (损失层)                                      │
│   └── CrossEntropyLoss, FocalLoss, SoftLabelLoss, InfoNCELoss...│
├─────────────────────────────────────────────────────────────────┤
│ backend/lib/adapters (适配器层)                                  │
│   └── 模态编码器、融合模块、对齐模块、LoRA适配器                  │
├─────────────────────────────────────────────────────────────────┤
│ backend/lib/distributed (分布式层)                               │
│   └── DistributedManager, DDPWrapper, FSDPWrapper, ZeROWrapper  │
├─────────────────────────────────────────────────────────────────┤
│ backend/lib/hardware (硬件层)                                    │
│   └── DeviceManager, MixedPrecisionManager, MemoryManager       │
└─────────────────────────────────────────────────────────────────┘
```

## 训练模式与策略调用关系

| 训练模式 | 策略类 | 调用层 | 说明 |
|----------|--------|--------|------|
| **行业模型训练** | `IndustryScenarioStrategy` | 全部六层 | 制造业、金融等行业场景 |
| **场景化训练** | `ScenarioStrategy` | 策略层 + 损失层 | 设备故障预测、质量检测等 |
| **分布式训练** | `DistributedStrategy` | 分布式层 + 硬件层 | DDP/FSDP/ZeRO/Pipeline |
| **知识蒸馏训练** | `DistillationStrategy` | 策略层 + 损失层 | Logits/特征/注意力蒸馏 |
| **多模态训练** | `MultiModalStrategy` | 适配器层 + 损失层 | 文本+图像+音频融合 |
| **三阶段训练** | `ThreeStageStrategy` | 编排层 + 策略层 + 损失层 + 硬件层 | PT→SFT→DPO |
| **标准训练** | `StandardTrainingStrategy` | 策略层 | 基础监督学习 |

---

## 各策略详细说明

### 1. StandardTrainingStrategy（标准训练）

**文件**: `base_strategy.py`

**调用层**: 策略层

**说明**: 实现基础的监督学习训练，不依赖底层 lib 模块，保持简单。

**配置示例**:
```python
from backend.modules.training.strategies import StandardTrainingStrategy

strategy = StandardTrainingStrategy()
strategy.setup(context)
result = strategy.compute_loss(model, batch, outputs, context)
```

**层调用信息**:
```python
strategy.get_layer_info()
# 返回:
# {
#     'strategy_layer': True,
#     'losses_layer': False,
#     'adapters_layer': False,
#     'distributed_layer': False,
#     'hardware_layer': False
# }
```

---

### 2. DistributedStrategy（分布式训练）

**文件**: `distributed_strategy.py`

**调用层**: 分布式层 + 硬件层

**底层模块调用**:
- `backend/lib/distributed`: DistributedManager, DDPWrapper, FSDPWrapper, ZeROWrapper
- `backend/lib/hardware`: DeviceManager, MixedPrecisionManager, MemoryManager

**支持的分布式模式**:
| 模式 | 枚举值 | 适用场景 | 内存效率 |
|------|--------|---------|---------|
| DDP | `DistributedMode.DDP` | 标准数据并行 | 低 |
| FSDP | `DistributedMode.FSDP` | 大模型训练 | 高 |
| ZeRO | `DistributedMode.ZERO` | DeepSpeed优化 | 最高 |
| Pipeline | `DistributedMode.PIPELINE` | 超长序列 | 高 |
| Tensor | `DistributedMode.TENSOR` | 张量并行 | 高 |
| Hybrid | `DistributedMode.HYBRID` | 混合并行 | 最高 |

**配置示例**:
```python
from backend.modules.training.strategies import (
    DistributedStrategy, 
    DistributedStrategyConfig,
    DistributedMode,
    ZeROStage
)

config = DistributedStrategyConfig(
    mode=DistributedMode.FSDP,
    world_size=8,
    rank=0,
    local_rank=0,
    backend='nccl',
    sync_bn=True,
    gradient_accumulation_steps=4,
    # FSDP 配置
    sharding_strategy='FULL_SHARD',
    cpu_offload=False,
    # ZeRO 配置
    zero_stage=ZeROStage.STAGE_2,
    zero_offload=False
)

strategy = DistributedStrategy(config)
strategy.setup(context)
```

**层调用信息**:
```python
strategy.get_layer_info()
# 返回:
# {
#     'distributed_layer': True,
#     'hardware_layer': True,
#     'distributed_manager': True,
#     'device_manager': True,
#     'amp_manager': True,
#     'device': 'cuda:0',
#     'mode': 'fsdp'
# }
```

---

### 3. DistillationStrategy（知识蒸馏训练）

**文件**: `distillation_strategy.py`

**调用层**: 策略层 + 损失层

**底层模块调用**:
- `backend/lib/losses`: SoftLabelLoss, FeatureDistillationLoss, AttentionDistillationLoss, InfoNCELoss

**支持的蒸馏类型**:
| 类型 | 枚举值 | 说明 |
|------|--------|------|
| Logits蒸馏 | `logits` | 软标签KL散度 |
| 特征蒸馏 | `feature` | 中间层特征匹配 |
| 注意力蒸馏 | `attention` | 注意力权重匹配 |
| 组合蒸馏 | `combined` | 多种蒸馏组合 |
| 渐进式蒸馏 | `progressive` | 逐层递进蒸馏 |
| 自蒸馏 | `self` | 无教师模型 |
| 对比蒸馏 | `contrastive` | 对比学习方式 |

**配置示例**:
```python
from backend.modules.training.strategies import (
    DistillationStrategy,
    DistillationStrategyConfig,
    create_distillation_strategy
)

config = DistillationStrategyConfig(
    temperature=4.0,
    hard_loss_weight=1.0,
    soft_loss_weight=0.5,
    feature_loss_weight=0.2,
    attention_loss_weight=0.1,
    distillation_type='combined',
    feature_layers=[-1, -2, -3],
    feature_loss_type='cosine',
    online_distillation=False
)

strategy = DistillationStrategy(config, teacher_model=teacher)
strategy.setup(context)

# 或使用便捷函数
strategy = create_distillation_strategy(
    strategy_type='industry',  # standard, self, progressive, industry, contrastive
    config={'temperature': 4.0},
    teacher_model=teacher
)
```

**层调用信息**:
```python
strategy.get_layer_info()
# 返回:
# {
#     'losses_layer': True,
#     'soft_label_loss': True,
#     'feature_loss': True,
#     'attention_loss': True,
#     'contrastive_loss': True,
#     'distillation_type': 'combined'
# }
```

---

### 4. MultiModalStrategy（多模态训练）

**文件**: `multimodal_strategy.py`

**调用层**: 适配器层 + 损失层

**底层模块调用**:
- `backend/lib/adapters`: 模态编码器、融合模块、对齐模块
- `backend/lib/losses`: CrossModalContrastiveLoss, CLIPLoss, InfoNCELoss

**支持的模态**:
- `text`: 文本
- `image`: 图像
- `audio`: 音频
- `video`: 视频
- `time_series`: 时序
- `table`: 表格

**融合方法**:
| 方法 | 说明 |
|------|------|
| `early` | 早期融合（特征拼接） |
| `middle` | 中期融合（Transformer交互） |
| `late` | 后期融合（注意力加权） |
| `cross_attention` | 交叉注意力融合 |
| `qformer` | BLIP-2风格Q-Former |
| `perceiver` | Perceiver风格 |

**对齐方法**:
| 方法 | 说明 |
|------|------|
| `contrastive` | 对比学习对齐（InfoNCE） |
| `explicit` | 显式MLP映射对齐 |
| `optimal_transport` | 最优传输对齐 |
| `cross_attention` | 交叉注意力对齐 |

**配置示例**:
```python
from backend.modules.training.strategies import (
    MultiModalStrategy,
    MultiModalStrategyConfig,
    ProductionMultiModalStrategy,
    create_multimodal_strategy
)

config = MultiModalStrategyConfig(
    modalities=['text', 'image', 'audio'],
    task_loss_weight=1.0,
    align_loss_weight=0.5,
    contrastive_loss_weight=0.2,
    fusion_method='cross_attention',
    fusion_stage='middle',
    fusion_dim=768,
    use_alignment=True,
    alignment_method='contrastive',
    alignment_temperature=0.07,
    modality_dropout=0.1,
    use_production_mode=True
)

strategy = ProductionMultiModalStrategy(config)
strategy.setup(context)

# 四阶段训练
strategy.set_training_stage('modality_pretrain')    # 阶段一
strategy.set_training_stage('cross_modal_align')    # 阶段二
strategy.set_training_stage('instruction_tuning')   # 阶段三
strategy.set_training_stage('alignment_safety')     # 阶段四
```

**层调用信息**:
```python
strategy.get_layer_info()
# 返回:
# {
#     'adapters_layer': True,
#     'losses_layer': True,
#     'lib_encoders': ['text', 'image', 'audio'],
#     'lib_fusion': True,
#     'lib_alignment': True,
#     'lib_contrastive_loss': True,
#     'lib_task_loss': True,
#     'modalities': ['text', 'image', 'audio'],
#     'fusion_method': 'cross_attention',
#     'alignment_method': 'contrastive'
# }
```

---

### 5. ScenarioStrategy（场景化训练）

**文件**: `scenario_strategy.py`

**调用层**: 策略层 + 损失层

**底层模块调用**:
- `backend/lib/losses`: CrossEntropyLoss, FocalLoss, HuberLoss, ConsistencyRegularization

**支持的场景类型**:

| 场景 | 枚举值 | 行业 |
|------|--------|------|
| 设备故障预测 | `EQUIPMENT_FAULT_PREDICTION` | 制造业 |
| 工艺参数优化 | `PROCESS_OPTIMIZATION` | 制造业 |
| 质量缺陷识别 | `QUALITY_DEFECT_DETECTION` | 制造业 |
| 能耗预测 | `ENERGY_PREDICTION` | 制造业 |
| 异常检测 | `ANOMALY_DETECTION` | 通用 |
| 风险评估 | `RISK_ASSESSMENT` | 金融 |
| 欺诈检测 | `FRAUD_DETECTION` | 金融 |
| 疾病诊断 | `DISEASE_DIAGNOSIS` | 医疗 |
| 医学影像分析 | `MEDICAL_IMAGE_ANALYSIS` | 医疗 |

**配置示例**:
```python
from backend.modules.training.strategies import (
    ScenarioStrategy,
    ScenarioStrategyConfig,
    ScenarioType
)

config = ScenarioStrategyConfig(
    scenario_type=ScenarioType.EQUIPMENT_FAULT_PREDICTION,
    scene_weight=1.0,
    task_loss_weight=1.0,
    scene_specific_loss_weight=0.2,
    freeze_backbone=True,
    use_scene_adapter=True,
    adapter_dim=64,
    augmentation_enabled=True,
    augmentation_strength=0.5,
    few_shot_enabled=False
)

strategy = ScenarioStrategy(config)
strategy.setup(context)
```

**层调用信息**:
```python
strategy.get_layer_info()
# 返回:
# {
#     'losses_layer': True,
#     'lib_task_loss': True,
#     'lib_focal_loss': True,
#     'lib_consistency_loss': True,
#     'scenario_type': 'equipment_fault_prediction',
#     'scene_weight': 1.0
# }
```

---

### 6. IndustryScenarioStrategy（行业模型训练）

**文件**: `scenario_strategy.py`

**调用层**: 全部六层

**底层模块调用**:
1. **硬件层** (`backend/lib/hardware`):
   - DeviceManager: 设备检测和选择
   - MixedPrecisionManager: 混合精度训练
   - MemoryManager: 内存管理

2. **分布式层** (`backend/lib/distributed`):
   - DistributedManager: 分布式环境管理
   - DDPWrapper/FSDPWrapper: 数据/模型并行

3. **适配器层** (`backend/lib/adapters`):
   - 模态编码器: 时序、图像、文本等
   - 融合模块: 多模态融合
   - 对齐模块: 跨模态对齐
   - LoRA适配器: 参数高效微调

4. **损失层** (`backend/lib/losses`):
   - FocalLoss: 分类场景
   - ConsistencyRegularization: 时序场景
   - MSELoss/HuberLoss: 回归场景

5. **策略层** (本模块):
   - 场景路由
   - 场景头（fault_prediction, process_optimization, quality_defect, anomaly_detection）

6. **编排层** (orchestrator):
   - ProductionTrainingContext
   - 三阶段训练流程

**配置示例**:
```python
from backend.modules.training.strategies import (
    IndustryScenarioStrategy,
    ScenarioStrategyConfig,
    ScenarioType
)
from backend.modules.training.strategies.production_base import ProductionStrategyConfig

# 场景配置
scenario_config = ScenarioStrategyConfig(
    scenario_type=ScenarioType.EQUIPMENT_FAULT_PREDICTION,
    freeze_backbone=True,
    use_scene_adapter=True,
    adapter_dim=64,
    scene_weight=1.0
)

# 生产级配置（六层架构）
production_config = ProductionStrategyConfig(
    device='auto',
    precision='fp16',
    enable_amp=True,
    enable_gradient_checkpointing=True,
    modalities=['time_series', 'text'],
    hidden_size=256,
    task_loss_type='cross_entropy',
    distributed_mode='ddp',
    world_size=4
)

strategy = IndustryScenarioStrategy(
    config=scenario_config,
    production_config=production_config
)
strategy.setup(context)

# 获取全部六层信息
layers_info = strategy.get_all_layers_info()
```

**层调用信息**:
```python
strategy.get_all_layers_info()
# 返回:
# {
#     'hardware_layer': {
#         'available': True,
#         'device_manager': True,
#         'amp_manager': True
#     },
#     'distributed_layer': {
#         'available': True,
#         'distributed_manager': True,
#         'is_distributed': False
#     },
#     'adapters_layer': {
#         'available': True,
#         'encoders': ['time_series'],
#         'fusion': False,
#         'alignment': False,
#         'scene_adapter': True
#     },
#     'losses_layer': {
#         'available': True,
#         'focal_loss': True,
#         'consistency_loss': True,
#         'mse_loss': True,
#         'task_loss': True
#     },
#     'strategy_layer': {
#         'scenario_type': 'equipment_fault_prediction',
#         'scenario_heads': ['fault_prediction', 'process_optimization', 'quality_defect', 'anomaly_detection'],
#         'scene_weight': 1.0,
#         'router': True
#     },
#     'orchestrator_layer': {
#         'production_context': True
#     }
# }
```

---

### 7. ThreeStageStrategy（三阶段训练）

**文件**: `three_stage_strategy.py`

**调用层**: 编排层 + 策略层 + 损失层 + 硬件层

**架构图**:
```
┌──────────────────────────────────────┐
│   three_stage/ (业务层)              │
│   (数据加载、配置管理、进度回调)      │
├──────────────────────────────────────┤
│   ThreeStageStrategy (策略层)        │
│   (三阶段训练策略核心逻辑)            │
├──────────────────────────────────────┤
│   StandardStrategy + Orchestrator    │
│   (标准训练流程 + 编排能力)           │
├──────────────────────────────────────┤
│   backend/lib/* (底层能力)           │
│   (hardware/distributed/losses/...)  │
└──────────────────────────────────────┘
```

**底层模块调用**:
1. **硬件层** (`backend/lib/hardware`):
   - DeviceManager: 设备检测和选择
   - MixedPrecisionManager: 混合精度训练（AMP）
   
2. **分布式层** (`backend/lib/distributed`):
   - DistributedManager: 分布式环境管理（可选）
   
3. **损失层** (`backend/lib/losses`):
   - CrossEntropyLoss: 预训练和微调阶段
   - DPO损失计算: 偏好优化阶段

4. **策略层** (本模块):
   - ProductionTrainingStrategy: 生产级基础能力
   - StandardTrainingStrategy: 标准训练流程

**三阶段训练流程**:

| 阶段 | 枚举值 | 学习目标 | 损失函数 |
|------|--------|----------|----------|
| 预训练 | `ThreeStagePhase.PRETRAIN` | 语言建模 | 交叉熵 |
| 监督微调 | `ThreeStagePhase.FINETUNE` | 指令跟随 | 交叉熵 |
| 偏好优化 | `ThreeStagePhase.PREFERENCE` | 人类对齐 | DPO损失 |

**配置示例**:
```python
from backend.modules.training.strategies import (
    ThreeStageStrategy,
    ThreeStageStrategyConfig,
    ThreeStagePhase,
    create_three_stage_strategy
)

# 配置
config = ThreeStageStrategyConfig(
    # 设备和精度
    device='cuda',
    precision='fp16',
    enable_amp=True,
    
    # 预训练配置
    pretrain_learning_rate=1e-4,
    pretrain_epochs=1,
    pretrain_warmup_steps=500,
    
    # 微调配置
    finetune_learning_rate=2e-5,
    finetune_epochs=3,
    finetune_warmup_steps=100,
    
    # 偏好优化配置
    preference_learning_rate=1e-5,
    preference_epochs=2,
    preference_warmup_steps=50,
    dpo_beta=0.1,  # DPO温度参数
    
    # 优化配置
    gradient_accumulation_steps=4,
    gradient_clipping=1.0,
    weight_decay=0.01,
    
    # 阶段控制
    enabled_stages=['pretrain', 'finetune', 'preference'],
    pass_model_between_stages=True
)

# 创建策略
strategy = ThreeStageStrategy(config=config)

# 设置策略
strategy.setup(context)

# 设置当前阶段
strategy.set_phase(ThreeStagePhase.PRETRAIN)

# 计算损失
result = strategy.compute_loss(model, batch, outputs, context)

# 反向传播
strategy.backward(result.loss)

# 优化器步骤
strategy.optimizer_step(optimizer, model)
```

**DPO偏好优化**:
```python
# 在偏好优化阶段，需要设置参考模型
strategy.set_phase(ThreeStagePhase.PREFERENCE)
strategy.setup_reference_model(model)  # 冻结的参考模型

# DPO数据格式
batch = {
    'chosen_input_ids': chosen_ids,
    'chosen_attention_mask': chosen_mask,
    'rejected_input_ids': rejected_ids,
    'rejected_attention_mask': rejected_mask
}

# 计算DPO损失
result = strategy.compute_loss(model, batch, outputs, context)
# result.metrics 包含: loss, chosen_reward, rejected_reward, reward_margin

# 训练结束后清理
strategy.cleanup_reference_model()
```

**层调用信息**:
```python
strategy.get_layer_info()
# 返回:
# {
#     'strategy_layer': True,
#     'losses_layer': True,  # 使用 backend/lib/losses
#     'adapters_layer': False,
#     'distributed_layer': True,  # 如果配置了分布式
#     'hardware_layer': True,  # 使用 backend/lib/hardware
#     'current_phase': 'pretrain',  # 当前阶段
#     'total_steps': 1000,
#     'phase_steps': 500
# }
```

**与业务层集成**:
```python
# 在 three_stage_trainer.py 中使用策略
from backend.modules.training.three_stage import (
    ThreeStageTrainer,
    ThreeStageConfig
)

config = ThreeStageConfig(
    base_model_path='gpt2',
    output_dir='./outputs',
    use_fp16=True
)

trainer = ThreeStageTrainer(config)

# 策略层信息
layer_info = trainer.get_strategy_layer_info()
print(layer_info)
# {
#     'strategy_available': True,
#     'strategy_initialized': True,
#     'losses_layer': True,
#     'hardware_layer': True,
#     ...
# }

# 执行训练
result = trainer.train()
```

---

## 组合策略

使用 `CompositeStrategy` 可以组合多个策略：

```python
from backend.modules.training.strategies import (
    CompositeStrategy,
    StandardTrainingStrategy,
    DistillationStrategy,
    MultiModalStrategy,
    create_composite_production_strategy
)

# 方式一：手动组合
strategies = [
    StandardTrainingStrategy(),
    DistillationStrategy(config),
    MultiModalStrategy(mm_config)
]
weights = [1.0, 0.5, 0.3]

composite = CompositeStrategy(strategies, weights)
composite.setup(context)

# 方式二：使用便捷函数
composite = create_composite_production_strategy(
    strategies=['production', 'distillation', 'multimodal'],
    weights=[1.0, 0.5, 0.3],
    modalities=['text', 'image']
)
```

---

## 生产级训练上下文

`ProductionTrainingContext` 整合六层架构，提供统一的训练上下文：

```python
from backend.modules.training.strategies.production_base import (
    ProductionTrainingContext,
    ProductionStrategyConfig,
    create_production_context
)

config = ProductionStrategyConfig(
    device='auto',
    precision='fp16',
    enable_amp=True,
    distributed_mode='ddp',
    world_size=4,
    modalities=['text', 'image'],
    hidden_size=768,
    fusion_method='cross_attention',
    alignment_method='contrastive',
    task_loss_type='cross_entropy'
)

# 创建上下文
ctx = create_production_context(config=config, model=model)

# 使用上下文
batch = ctx.to_device(batch)
with ctx.get_amp_context():
    outputs = model(batch)
    loss = ctx.compute_loss(outputs, targets)

ctx.backward(loss)
ctx.optimizer_step(optimizer)
ctx.sync_gradients()

# 清理
ctx.cleanup()
```

---

## 训练启动器配置

在 `launcher/training_launcher.py` 中配置训练模式：

```yaml
# config.yaml
training:
  mode: "industry"  # standard, distributed, distillation, multimodal, industry, scenario, three_stage
  
  # 行业模型配置
  industry:
    type: "manufacturing"
    scenario: "equipment_fault_prediction"
  
  # 分布式配置
  distributed:
    mode: "fsdp"
    world_size: 8
    
  # 知识蒸馏配置
  distillation:
    temperature: 4.0
    type: "combined"
    
  # 多模态配置
  multimodal:
    modalities: ["text", "image", "audio"]
    fusion_method: "cross_attention"
    
  # 通用配置
  precision: "fp16"
  enable_amp: true
  output_dir: "./outputs"
```

```python
from backend.modules.training.launcher import TrainingLauncher

launcher = TrainingLauncher.from_config("config.yaml")
launcher.run()
```

---

## 目录结构

```
backend/modules/training/strategies/
├── __init__.py                 # 模块导出
├── STRATEGIES_README.md        # 本文档
├── base_strategy.py            # 基类定义
│   ├── TrainingStrategy        # 策略基类
│   ├── StandardTrainingStrategy# 标准训练策略
│   └── CompositeStrategy       # 组合策略
├── production_base.py          # 生产级基类
│   ├── ProductionTrainingStrategy
│   └── ProductionTrainingContext
├── distributed_strategy.py     # 分布式策略
│   ├── DistributedStrategy
│   └── IndustryDistributedStrategy
├── distillation_strategy.py    # 蒸馏策略
│   ├── DistillationStrategy
│   ├── SelfDistillationStrategy
│   ├── ProgressiveDistillationStrategy
│   ├── IndustryDistillationStrategy
│   └── ContrastiveDistillationStrategy
├── multimodal_strategy.py      # 多模态策略
│   ├── MultiModalStrategy
│   ├── ProductionMultiModalStrategy
│   ├── IndustryMultiModalStrategy
│   └── MultiModalTrainingPipeline
└── scenario_strategy.py        # 场景化策略
    ├── ScenarioStrategy
    └── IndustryScenarioStrategy
```

---

## 底层模块依赖

```
backend/lib/
├── hardware/                   # 硬件抽象层
│   ├── DeviceManager          # 设备管理
│   ├── MixedPrecisionManager  # 混合精度
│   └── MemoryManager          # 内存管理
│
├── distributed/               # 分布式训练层
│   ├── DistributedManager     # 分布式管理
│   ├── DDPWrapper             # DDP封装
│   ├── FSDPWrapper            # FSDP封装
│   └── ZeROWrapper            # ZeRO封装
│
├── adapters/                  # 适配器层
│   ├── 模态编码器             # Text/Image/Audio/Video/TimeSeries
│   ├── 融合模块               # Early/Middle/Late/CrossAttention
│   ├── 对齐模块               # Contrastive/Explicit/OptimalTransport
│   └── 模型适配器             # LoRA/Prefix/Prompt
│
└── losses/                    # 损失函数层
    ├── 监督损失               # CrossEntropy/Focal/MSE
    ├── 蒸馏损失               # SoftLabel/Feature/Attention
    ├── 对比损失               # InfoNCE/CLIP/NTXent
    └── 复合损失               # Composite/MultiTask
```

---

## 版本历史

- **v1.1.0** (2026-01-09): 策略层架构整合
  - 完善各策略对 backend/lib 的调用
  - 添加 get_layer_info() 方法
  - 支持七种训练模式
  - 文档化架构调用关系


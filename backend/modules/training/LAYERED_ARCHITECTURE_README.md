# 六层训练架构说明

## 架构概览

本项目采用六层分层架构设计，实现了高度模块化、可扩展的训练系统：

```
┌──────────────────────────────────────┐
│          Training Orchestrator       │ ← 训练编排层
│     (Stage / Scenario / Strategy)    │
├──────────────────────────────────────┤
│       Training Strategy Abstraction  │ ← 策略层
│    (Standard / Distill / MultiTask)  │
├──────────────────────────────────────┤
│      Loss & Objective Composition    │ ← 目标函数层
│   (Supervised / KD / Contrastive)    │
├──────────────────────────────────────┤
│      Model & Modality Adapter Layer  │ ← 模型/模态层
│     (Text / Image / Audio / Fusion)  │
├──────────────────────────────────────┤
│        Distributed Training Core     │ ← 训练内核层
│       (DDP / FSDP / Pipeline / ZeRO) │
├──────────────────────────────────────┤
│          Hardware Abstraction        │ ← 硬件抽象层
│        (GPU / NPU / TPU / CPU)       │
└──────────────────────────────────────┘
```

## 各层详细说明

### 1. Hardware Abstraction Layer (硬件抽象层)

**位置**: `backend/modules/training/hardware/`

**功能**: 统一管理训练所需的硬件资源

**核心组件**:
- `DeviceManager`: 设备检测和管理
- `MemoryManager`: 内存优化和管理
- `MixedPrecisionManager`: 混合精度训练支持
- `DeviceScheduler`: 设备调度和分配

**使用示例**:
```python
from backend.modules.training.hardware import (
    DeviceManager, get_device_manager,
    MixedPrecisionManager, AmpContext
)

# 获取最佳设备
device_manager = get_device_manager()
device = device_manager.get_best_device()

# 混合精度训练
with AmpContext() as amp:
    output = model(input)
    loss = criterion(output, target)
    amp.backward(loss)
```

---

### 2. Distributed Training Core (分布式训练内核层)

**位置**: `backend/modules/training/distributed/`

**功能**: 提供分布式训练的核心功能

**核心组件**:
- `DistributedManager`: 统一的分布式管理器
- `DDPWrapper`: 数据并行封装
- `FSDPWrapper`: 全分片数据并行封装
- `PipelineWrapper`: 流水线并行封装
- `ZeROWrapper`: DeepSpeed ZeRO优化封装

**支持的并行模式**:
| 模式 | 适用场景 | 内存效率 |
|------|---------|---------|
| DDP | 标准数据并行 | 低 |
| FSDP | 大模型训练 | 高 |
| ZeRO Stage 1 | 优化器状态分片 | 中 |
| ZeRO Stage 2 | 梯度+优化器分片 | 高 |
| ZeRO Stage 3 | 参数+梯度+优化器分片 | 最高 |
| Pipeline | 超长序列/超大模型 | 高 |

**使用示例**:
```python
from backend.modules.training.distributed import (
    DistributedManager, get_distributed_manager,
    ParallelMode, DDPConfig
)

# 初始化分布式
manager = get_distributed_manager()
manager.initialize(backend='nccl', world_size=8)

# 包装模型
model = manager.wrap_model(model, mode=ParallelMode.FSDP)

# 同步梯度
manager.barrier()
```

---

### 3. Model & Modality Adapter Layer (模型/模态适配器层)

**位置**: `backend/modules/training/adapters/`

**功能**: 统一管理各种模型架构和多模态适配

**核心组件**:

#### 模态编码器
- `TextEncoder`: 文本编码 (BERT/RoBERTa风格)
- `ImageEncoder`: 图像编码 (ViT风格)
- `AudioEncoder`: 音频编码 (Wav2Vec风格)
- `VideoEncoder`: 视频编码 (时空Transformer)
- `TimeSeriesEncoder`: 时序编码
- `TabularEncoder`: 表格数据编码

#### 融合模块
- `EarlyFusion`: 早期融合 (特征拼接)
- `MiddleFusion`: 中期融合 (Transformer交互)
- `LateFusion`: 后期融合 (注意力加权)
- `CrossAttentionFusion`: 交叉注意力融合
- `GatedFusion`: 门控融合
- `QFormerFusion`: BLIP-2风格Q-Former
- `PerceiverFusion`: Perceiver风格

#### 对齐模块
- `ContrastiveAlignment`: 对比学习对齐 (InfoNCE)
- `ExplicitAlignment`: 显式MLP映射对齐
- `OptimalTransportAlignment`: 最优传输对齐 (Sinkhorn)
- `CrossModalAttentionAlignment`: 交叉注意力对齐

#### 模型适配器
- `LoRAAdapter`: Low-Rank Adaptation
- `PrefixAdapter`: Prefix Tuning
- `PromptAdapter`: Prompt Tuning
- `BackboneAdapter`: 骨干网络冻结/解冻
- `TaskHeadAdapter`: 任务头添加

**使用示例**:
```python
from backend.modules.training.adapters import (
    AdapterManager, get_adapter_manager,
    create_encoder, create_fusion, create_alignment
)

# 获取适配器管理器
manager = get_adapter_manager()

# 创建多模态管道
pipeline = manager.create_multimodal_pipeline(
    modalities=['text', 'image'],
    fusion_method='cross_attention',
    alignment_method='contrastive',
    hidden_size=768
)

# 使用编码器
text_features = pipeline['encoders']['text'](text_input)
image_features = pipeline['encoders']['image'](image_input)

# 对齐
aligned_text, aligned_image = pipeline['alignment'].align(text_features, image_features)

# 融合
fused = pipeline['fusion']([aligned_text, aligned_image])
```

---

### 4. Loss & Objective Composition (目标函数层)

**位置**: `backend/modules/training/losses/`

**功能**: 统一管理训练中的各种损失函数和目标函数组合

**核心组件**:

#### 监督学习损失
- `CrossEntropyLoss`: 交叉熵损失
- `FocalLoss`: 焦点损失 (类别不平衡)
- `LabelSmoothingLoss`: 标签平滑
- `MSELoss`, `MAELoss`, `HuberLoss`: 回归损失
- `DiceLoss`, `IoULoss`: 分割损失

#### 蒸馏损失
- `SoftLabelLoss`: 软标签蒸馏
- `FeatureDistillationLoss`: 特征蒸馏
- `AttentionDistillationLoss`: 注意力蒸馏

#### 对比学习损失
- `InfoNCELoss`: InfoNCE对比损失
- `NTXentLoss`: NT-Xent损失
- `CLIPLoss`: CLIP风格损失

#### 复合损失
- `CompositeLoss`: 多损失组合
- `MultiTaskLoss`: 多任务损失

**使用示例**:
```python
from backend.modules.training.losses import (
    LossFactory, create_loss, CompositeLoss,
    CrossEntropyLoss, SoftLabelLoss, InfoNCELoss
)

# 创建单一损失
loss_fn = create_loss('cross_entropy', num_classes=10)

# 创建复合损失
composite = CompositeLoss([
    ('task', CrossEntropyLoss(), 1.0),
    ('kd', SoftLabelLoss(temperature=4.0), 0.5),
    ('contrastive', InfoNCELoss(), 0.1)
])

# 计算损失
loss = composite(outputs, targets, teacher_logits=teacher_out)
```

---

### 5. Training Strategy Abstraction (策略层)

**位置**: `backend/modules/training/strategies/`

**功能**: 提供统一的训练策略接口

**核心策略**:
- `StandardTrainingStrategy`: 标准监督学习
- `DistillationStrategy`: 知识蒸馏 (Logits/Feature/Attention)
- `MultiModalStrategy`: 多模态联合训练
- `ScenarioStrategy`: 场景化训练 (行业场景)
- `DistributedStrategy`: 分布式训练策略

**使用示例**:
```python
from backend.modules.training.strategies import (
    create_strategy, DistillationStrategy, MultiModalStrategy
)

# 创建蒸馏策略
strategy = create_strategy(
    'distillation',
    temperature=4.0,
    soft_loss_weight=1.0,
    hard_loss_weight=0.5
)

# 设置策略
strategy.setup(context)

# 执行训练步骤
result = strategy.train_step(batch)
```

---

### 6. Training Orchestrator (训练编排层)

**位置**: `backend/modules/training/orchestrator/`

**功能**: 训练流程的最高层抽象，负责协调所有层

**核心组件**:
- `UnifiedTrainingOrchestrator`: 统一训练编排器
- `LayerManager`: 六层架构管理器
- `OrchestratorPlan`: 训练计划定义
- `PhaseConfig`: 阶段配置

**支持的训练阶段**:
| 阶段类型 | 描述 |
|---------|------|
| `PRETRAIN` | 预训练阶段 |
| `FINETUNE` | 微调阶段 |
| `PREFERENCE` | 偏好对齐阶段 |
| `INDUSTRY_PRETRAIN` | 行业预训练 |
| `INDUSTRY_ALIGN` | 行业对齐 |
| `SCENE_FINETUNE` | 场景微调 |
| `MODALITY_PRETRAIN` | 模态预训练 |
| `CROSS_MODAL_ALIGN` | 跨模态对齐 |
| `INSTRUCTION_TUNING` | 指令微调 |
| `ALIGNMENT_SAFETY` | 对齐与安全 |

**使用示例**:
```python
from backend.modules.training.orchestrator import (
    UnifiedTrainingOrchestrator,
    create_three_stage_plan,
    create_multimodal_plan
)

# 创建编排器
orchestrator = UnifiedTrainingOrchestrator(output_dir="./outputs")

# 创建三阶段训练计划
plan = orchestrator.create_three_stage_plan(
    name="my_training",
    pretrain_epochs=3,
    finetune_epochs=5,
    preference_epochs=2
)

# 或创建多模态训练计划
plan = orchestrator.create_multimodal_plan(
    name="multimodal_training",
    modalities=['text', 'image', 'audio']
)

# 执行训练
results = orchestrator.execute(plan, model, train_loader, val_loader)

# 查看结果
for result in results:
    print(f"Phase: {result.phase}, Status: {result.status}, Metrics: {result.metrics}")
```

---

## 完整使用示例

### 示例1: 标准训练

```python
from backend.modules.training.orchestrator import UnifiedTrainingOrchestrator
from backend.modules.training.strategies import create_strategy

# 创建编排器
orchestrator = UnifiedTrainingOrchestrator(output_dir="./outputs")

# 创建标准训练计划
plan = orchestrator.create_standard_plan(
    name="standard_training",
    epochs=10,
    learning_rate=1e-4
)

# 执行训练
results = orchestrator.execute(plan, model, train_loader)
```

### 示例2: 知识蒸馏训练

```python
from backend.modules.training.orchestrator import UnifiedTrainingOrchestrator, LayerConfig

# 配置蒸馏
config = LayerConfig(
    strategy_type='distillation',
    strategy_config={
        'temperature': 4.0,
        'soft_loss_weight': 1.0,
        'hard_loss_weight': 0.5
    }
)

# 创建编排器
orchestrator = UnifiedTrainingOrchestrator(
    output_dir="./distillation_outputs",
    default_config=config
)

# 创建蒸馏计划
plan = orchestrator.create_distillation_plan(
    name="distillation_training",
    distillation_epochs=10
)

# 执行训练
results = orchestrator.execute(plan, student_model, train_loader)
```

### 示例3: 多模态四阶段训练

```python
from backend.modules.training.orchestrator import UnifiedTrainingOrchestrator, LayerConfig

# 配置多模态
config = LayerConfig(
    modalities=['text', 'image', 'audio'],
    fusion_method='cross_attention',
    strategy_type='multimodal'
)

# 创建编排器
orchestrator = UnifiedTrainingOrchestrator(
    output_dir="./multimodal_outputs",
    default_config=config
)

# 创建多模态训练计划
plan = orchestrator.create_multimodal_plan(
    name="multimodal_training",
    modalities=['text', 'image', 'audio']
)

# 执行训练 (四阶段: 模态预训练 -> 跨模态对齐 -> 指令微调 -> 对齐与安全)
results = orchestrator.execute(plan, model, train_loader)
```

### 示例4: 行业模型训练

```python
from backend.modules.training.orchestrator import UnifiedTrainingOrchestrator, LayerConfig

# 配置行业模型
config = LayerConfig(
    strategy_type='scenario',
    strategy_config={
        'industry': 'manufacturing',
        'scenario': 'fault_prediction'
    }
)

# 创建编排器
orchestrator = UnifiedTrainingOrchestrator(
    output_dir="./industry_outputs",
    default_config=config
)

# 创建行业训练计划
plan = orchestrator.create_industry_plan(
    name="industry_training",
    include_pretrain=True,
    include_align=True,
    include_finetune=True
)

# 执行训练
results = orchestrator.execute(plan, model, train_loader)
```

---

## 扩展指南

### 添加新的模态编码器

```python
from backend.modules.training.adapters.modality_encoders import (
    ModalityEncoder, EncoderConfig, EncoderFactory, ModalityType
)

class MyCustomEncoder(ModalityEncoder):
    def __init__(self, config: EncoderConfig):
        super().__init__(config)
        # 自定义初始化
    
    def encode(self, inputs, **kwargs):
        # 自定义编码逻辑
        return features

# 注册编码器
EncoderFactory.register(ModalityType.CUSTOM, MyCustomEncoder)
```

### 添加新的融合方法

```python
from backend.modules.training.adapters.fusion_modules import (
    FusionModule, FusionConfig, FusionFactory, FusionMethod
)

class MyCustomFusion(FusionModule):
    def __init__(self, config: FusionConfig):
        super().__init__(config)
        # 自定义初始化
    
    def fuse(self, features, **kwargs):
        # 自定义融合逻辑
        return fused_features

# 注册融合模块
FusionFactory.register(FusionMethod.CUSTOM, MyCustomFusion)
```

### 添加新的训练策略

```python
from backend.modules.training.strategies.base_strategy import (
    TrainingStrategy, StrategyContext, StrategyResult
)

class MyCustomStrategy(TrainingStrategy):
    def __init__(self, config):
        super().__init__(name="custom", priority=5)
        self.config = config
    
    def setup(self, context: StrategyContext):
        super().setup(context)
        # 自定义设置
    
    def train_step(self, batch, context: StrategyContext) -> StrategyResult:
        # 自定义训练步骤
        return StrategyResult(loss=loss, metrics=metrics)

# 使用自定义策略
strategy = MyCustomStrategy(config)
```

---

## 目录结构

```
backend/modules/training/
├── __init__.py
├── LAYERED_ARCHITECTURE_README.md  # 本文档
│
├── orchestrator/                   # 训练编排层
│   ├── __init__.py
│   ├── unified_orchestrator.py     # 统一编排器
│   └── training_orchestrator.py    # 传统编排器
│
├── strategies/                     # 策略层
│   ├── __init__.py
│   ├── base_strategy.py
│   ├── standard_strategy.py
│   ├── distillation_strategy.py
│   ├── multimodal_strategy.py
│   ├── scenario_strategy.py
│   └── distributed_strategy.py
│
├── losses/                         # 目标函数层
│   ├── __init__.py
│   ├── base_loss.py
│   ├── supervised_loss.py
│   ├── distillation_loss.py
│   ├── contrastive_loss.py
│   ├── composite_loss.py
│   └── loss_factory.py
│
├── adapters/                       # 模型/模态适配器层
│   ├── __init__.py
│   ├── modality_encoders.py
│   ├── fusion_modules.py
│   ├── alignment_modules.py
│   ├── model_adapters.py
│   └── adapter_manager.py
│
├── distributed/                    # 分布式训练内核层
│   ├── __init__.py
│   ├── parallel_modes.py
│   ├── ddp_wrapper.py
│   ├── fsdp_wrapper.py
│   ├── pipeline_wrapper.py
│   ├── zero_wrapper.py
│   └── distributed_manager.py
│
└── hardware/                       # 硬件抽象层
    ├── __init__.py
    ├── device_types.py
    ├── device_manager.py
    ├── memory_manager.py
    ├── mixed_precision.py
    └── device_scheduler.py
```

---

## 策略层与底层模块调用关系

策略层 (`strategies/`) 对底层 `backend/lib/*` 模块的调用关系如下：

### 调用关系总览

| 训练模式 | 策略类 | 调用层 |
|----------|--------|--------|
| **行业模型训练** | `IndustryScenarioStrategy` | 全部六层 |
| **场景化训练** | `ScenarioStrategy` | 策略层 + 损失层 |
| **分布式训练** | `DistributedStrategy` | 分布式层 + 硬件层 |
| **知识蒸馏训练** | `DistillationStrategy` | 策略层 + 损失层 |
| **多模态训练** | `MultiModalStrategy` | 适配器层 + 损失层 |
| **三阶段训练** | `ThreeStageStrategy` | 编排层 + 策略层 + 损失层 + 硬件层 |
| **标准训练** | `StandardTrainingStrategy` | 策略层 |

### 详细调用关系

```
strategies/
├── StandardTrainingStrategy
│   └── 策略层（无底层lib依赖）
│
├── DistributedStrategy
│   ├── backend/lib/distributed
│   │   ├── DistributedManager
│   │   ├── DDPWrapper / FSDPWrapper / ZeROWrapper
│   │   └── barrier, all_reduce, all_gather
│   └── backend/lib/hardware
│       ├── DeviceManager
│       ├── MixedPrecisionManager
│       └── MemoryManager
│
├── DistillationStrategy
│   └── backend/lib/losses
│       ├── SoftLabelLoss
│       ├── FeatureDistillationLoss
│       ├── AttentionDistillationLoss
│       └── InfoNCELoss
│
├── MultiModalStrategy
│   ├── backend/lib/adapters
│   │   ├── create_encoder (Text/Image/Audio/Video/TimeSeries)
│   │   ├── create_fusion (Early/Middle/Late/CrossAttention)
│   │   └── create_alignment (Contrastive/Explicit/OptimalTransport)
│   └── backend/lib/losses
│       ├── CrossModalContrastiveLoss
│       ├── CLIPLoss
│       └── InfoNCELoss
│
├── ScenarioStrategy
│   └── backend/lib/losses
│       ├── CrossEntropyLoss / FocalLoss
│       ├── HuberLoss / MSELoss
│       └── ConsistencyRegularization
│
└── IndustryScenarioStrategy (全部六层)
    ├── backend/lib/hardware
    │   ├── DeviceManager
    │   ├── MixedPrecisionManager
    │   └── MemoryManager
    ├── backend/lib/distributed
    │   ├── DistributedManager
    │   └── DDPWrapper / FSDPWrapper
    ├── backend/lib/adapters
    │   ├── 模态编码器
    │   ├── LoRA适配器
    │   └── 融合/对齐模块
    ├── backend/lib/losses
    │   ├── FocalLoss
    │   ├── ConsistencyRegularization
    │   └── MSELoss
    ├── 策略层 (场景路由/场景头)
    └── 编排层 (ProductionTrainingContext)
```

### 获取层调用信息

每个策略类都提供 `get_layer_info()` 方法，返回底层模块的调用状态：

```python
# 示例：DistributedStrategy
strategy = DistributedStrategy(config)
strategy.setup(context)
print(strategy.get_layer_info())
# {
#     'distributed_layer': True,
#     'hardware_layer': True,
#     'distributed_manager': True,
#     'device_manager': True,
#     'amp_manager': True,
#     'device': 'cuda:0',
#     'mode': 'fsdp'
# }

# 示例：IndustryScenarioStrategy
strategy = IndustryScenarioStrategy(config)
strategy.setup(context)
print(strategy.get_all_layers_info())
# {
#     'hardware_layer': {'available': True, 'device_manager': True, ...},
#     'distributed_layer': {'available': True, ...},
#     'adapters_layer': {'available': True, 'encoders': [...], ...},
#     'losses_layer': {'available': True, 'focal_loss': True, ...},
#     'strategy_layer': {'scenario_type': '...', ...},
#     'orchestrator_layer': {'production_context': True}
# }
```

> 详细配置和使用方法请参考 `strategies/STRATEGIES_README.md`

---

## 版本历史

- **v1.1.0** (2026-01-09): 策略层架构整合
  - 完善各策略对 backend/lib 的调用
  - 添加 get_layer_info() 方法
  - 支持七种训练模式
  - 文档化架构调用关系

- **v1.0.0** (2026-01-08): 初始六层架构实现
  - 实现完整的六层分层架构
  - 支持多种分布式并行策略
  - 支持多模态融合和对齐
  - 支持多种训练策略
  - 统一的训练编排接口

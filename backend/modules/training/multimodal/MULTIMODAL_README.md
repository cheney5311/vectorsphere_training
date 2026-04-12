# 多模态训练模块 (MultiModal Training)

多模态训练模块提供支持多种模态和场景的统一训练框架。

## 功能概述

### 支持的模态类型

| 模态 | 说明 | 编码器 |
|------|------|--------|
| `text` | 文本数据 | Transformer/MLP |
| `image` | 图像数据 | CNN |
| `time_series` | 时序信号 | LSTM |
| `table` | 表格数据 | MLP |
| `audio` | 音频数据 | 1D CNN |

### 融合方法

| 方法 | 说明 |
|------|------|
| `concat` | 拼接融合 - 简单高效 |
| `attention` | 注意力融合 - 自适应权重 |
| `gated` | 门控融合 - 可学习门控 |
| `cross_attention` | 跨模态注意力 - 深度交互 |

### 行业场景

| 场景 | 模态组合 | 说明 |
|------|----------|------|
| `manufacturing` | 文本+表格+时序+图像 | 制造业场景 |
| `finance` | 文本+表格+时序 | 金融场景 |
| `medical` | 文本+图像+表格 | 医疗场景 |
| `retail` | 文本+图像+表格 | 零售场景 |

## 快速开始

### 基本用法

```python
from backend.modules.training.multimodal import (
    MultiModalConfig,
    MultiModalTrainer,
    get_preset_trainer
)

# 创建配置
config = MultiModalConfig(
    modalities=['text', 'image'],
    fusion_method='attention',
    num_epochs=10,
    batch_size=16,
    output_dir='./outputs/multimodal'
)

# 创建训练器并训练
trainer = MultiModalTrainer(config)
result = trainer.train()
print(f"训练结果: {result}")
```

### 使用预设配置

```python
# 获取制造业预设训练器
trainer = get_preset_trainer('manufacturing')
result = trainer.train()

# 或金融场景
trainer = get_preset_trainer('finance')
```

### 使用策略层

```python
config = MultiModalConfig(
    modalities=['text', 'image'],
    use_strategy=True,
    strategy_type='multimodal',
    use_alignment=True,
    use_contrastive=True
)

trainer = MultiModalTrainer(config)
result = trainer.train()
```

## 配置详解

### MultiModalConfig

```python
@dataclass
class MultiModalConfig:
    # 模态配置
    modalities: List[str] = ['text', 'image']
    modality_dims: Dict[str, int] = {'text': 768, 'image': 2048}
    
    # 融合配置
    fusion_method: str = 'concat'  # concat, attention, gated, cross_attention
    fusion_dim: int = 1024
    
    # 对齐配置
    use_alignment: bool = True
    alignment_temperature: float = 0.07
    
    # 对比学习配置
    use_contrastive: bool = False
    contrastive_weight: float = 0.1
    
    # 损失权重
    task_loss_weight: float = 1.0
    align_loss_weight: float = 0.5
    contrastive_loss_weight: float = 0.1
    
    # 训练配置
    learning_rate: float = 5e-5
    batch_size: int = 16
    num_epochs: int = 10
    
    # 策略配置
    use_strategy: bool = True
    strategy_type: str = 'multimodal'  # multimodal, industry_multimodal
```

### IndustryMultiModalConfig

```python
@dataclass
class IndustryMultiModalConfig(MultiModalConfig):
    # 行业特定模态
    modalities: List[str] = ['text', 'table', 'time_series', 'image']
    
    # 行业配置
    industry_type: str = 'manufacturing'
    
    # 时序信号配置
    sensor_channels: int = 32
    plc_signals: int = 16
    
    # 工艺文档配置
    document_max_length: int = 1024
    
    # 行业特定融合
    fusion_method: str = 'attention'
```

## 模型架构

### MultiModalModel

```
┌─────────────────────────────────────────────────────────────┐
│                    MultiModalModel                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐              │
│   │   Text    │  │   Image   │  │TimeSeries │  ...         │
│   │  Encoder  │  │  Encoder  │  │  Encoder  │              │
│   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘              │
│         │              │              │                      │
│         └──────────────┼──────────────┘                      │
│                        │                                     │
│                        ▼                                     │
│              ┌─────────────────┐                            │
│              │  Fusion Layer   │                            │
│              │  (concat/attn)  │                            │
│              └────────┬────────┘                            │
│                       │                                      │
│                       ▼                                      │
│              ┌─────────────────┐                            │
│              │   Classifier    │                            │
│              └────────┬────────┘                            │
│                       │                                      │
│                       ▼                                      │
│                   Logits                                     │
└─────────────────────────────────────────────────────────────┘
```

### 编码器

#### TextEncoder
```python
TextEncoder(
    input_dim=512,      # max_text_length
    output_dim=768,     # embedding dim
    dropout=0.1
)
```

#### ImageEncoder
```python
ImageEncoder(
    num_channels=3,     # RGB
    output_dim=2048,    # embedding dim
    dropout=0.1
)
```

#### TimeSeriesEncoder
```python
TimeSeriesEncoder(
    input_channels=16,  # 传感器通道数
    seq_length=128,     # 序列长度
    output_dim=256,     # embedding dim
    dropout=0.1
)
```

#### TableEncoder
```python
TableEncoder(
    input_dim=64,       # 特征数
    output_dim=256,     # embedding dim
    dropout=0.1
)
```

## 策略集成

### MultiModalStrategy

多模态训练策略提供：
- 模态对齐损失
- 对比学习损失
- 模态dropout增强
- 自适应损失权重

```python
from backend.modules.training.strategies.multimodal_strategy import (
    MultiModalStrategy,
    MultiModalStrategyConfig
)

# 配置策略
strategy_config = MultiModalStrategyConfig(
    modalities=['text', 'image'],
    task_loss_weight=1.0,
    align_loss_weight=0.5,
    contrastive_loss_weight=0.1,
    modality_dropout=0.1
)

strategy = MultiModalStrategy(strategy_config)
```

### IndustryMultiModalStrategy

行业多模态策略针对行业场景优化：
- 文本（工艺文档、维修记录）
- 表格（BOM、工艺参数）
- 时序（传感器、PLC、SCADA）
- 图像（缺陷检测、设备视觉）

```python
from backend.modules.training.strategies.multimodal_strategy import (
    IndustryMultiModalStrategy
)

strategy = IndustryMultiModalStrategy()
# 自动配置行业优化参数
```

## 使用示例

### 制造业场景

```python
from backend.modules.training.multimodal import (
    IndustryMultiModalConfig,
    MultiModalTrainer
)

# 制造业配置
config = IndustryMultiModalConfig(
    industry_type='manufacturing',
    modalities=['text', 'table', 'time_series', 'image'],
    sensor_channels=32,
    plc_signals=16,
    document_max_length=1024,
    fusion_method='attention',
    use_strategy=True,
    strategy_type='industry_multimodal'
)

trainer = MultiModalTrainer(config)
result = trainer.train()
```

### 金融场景

```python
from backend.modules.training.multimodal import get_preset_trainer

trainer = get_preset_trainer('finance',
    num_epochs=20,
    batch_size=32,
    output_dir='./outputs/finance'
)
result = trainer.train()
```

### 自定义模态组合

```python
from backend.modules.training.multimodal import (
    MultiModalConfig,
    ModalityConfig,
    MultiModalTrainer
)

# 自定义模态配置
config = MultiModalConfig(
    modalities=['text', 'time_series', 'audio'],
    modality_dims={
        'text': 512,
        'time_series': 256,
        'audio': 512
    },
    fusion_method='gated',
    fusion_dim=512
)

trainer = MultiModalTrainer(config)
```

## 训练回调

```python
def my_callback(epoch, metrics):
    print(f"Epoch {epoch}: loss={metrics['loss']:.4f}")

trainer = MultiModalTrainer(config)
trainer.register_callback(my_callback)
trainer.train()
```

## 模型保存和加载

```python
# 训练并自动保存
trainer = MultiModalTrainer(config)
result = trainer.train()
# 自动保存: best_model.pth, final_model.pth

# 加载模型
trainer.load_model('./outputs/best_model.pth')

# 评估
metrics = trainer.evaluate()
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

## 便捷函数

### create_multimodal_trainer

```python
from backend.modules.training.multimodal import create_multimodal_trainer

trainer = create_multimodal_trainer({
    'modalities': ['text', 'image'],
    'fusion_method': 'attention',
    'output_dir': './outputs'
})
```

### create_industry_multimodal_trainer

```python
from backend.modules.training.multimodal import create_industry_multimodal_trainer

trainer = create_industry_multimodal_trainer(
    industry_type='manufacturing',
    output_dir='./outputs'
)
```

### get_preset_trainer

```python
from backend.modules.training.multimodal import get_preset_trainer

# 可用预设
presets = [
    'standard_classification',
    'multimodal_alignment',
    'manufacturing',
    'finance',
    'medical',
    'retail'
]

trainer = get_preset_trainer('manufacturing')
```

## 性能优化

### 混合精度训练

```python
config = MultiModalConfig(
    use_fp16=True,
    gradient_accumulation_steps=4
)
```

### 梯度裁剪

```python
config = MultiModalConfig(
    gradient_clipping=1.0
)
```

### 模态Dropout

```python
config = MultiModalConfig(
    modality_dropout=0.1  # 随机丢弃10%模态
)
```

## 最佳实践

1. **模态选择**
   - 根据数据可用性选择模态
   - 不必使用所有模态

2. **融合方法**
   - 简单任务用 `concat`
   - 复杂任务用 `attention` 或 `cross_attention`

3. **损失权重**
   - 任务损失为主
   - 对齐和对比损失为辅

4. **策略使用**
   - 行业场景建议使用 `industry_multimodal` 策略
   - 通用场景使用 `multimodal` 策略

5. **数据准备**
   - 确保各模态数据对齐
   - 注意数据归一化


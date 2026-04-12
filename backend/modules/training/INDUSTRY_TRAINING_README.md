# 公司级行业模型训练平台

## 概述

本模块实现了公司级行业模型训练平台，支持多种训练范式和行业场景，基于统一的训练策略层和编排器设计。

## 架构设计

```
┌──────────────────────────────────────────────────────────┐
│                  Training Platform                        │
├──────────────────────────────────────────────────────────┤
│    Training Orchestrator (Job / Stage / Scene)           │
├──────────────────────────────────────────────────────────┤
│        Strategy Layer (可组合训练策略)                    │
│   Multimodal | Distillation | Scenario | Distributed     │
├──────────────────────────────────────────────────────────┤
│         Loss & Objective Composition                      │
├──────────────────────────────────────────────────────────┤
│       Industry Model Abstraction Layer                    │
│      (Text / Table / Image / TimeSeries)                 │
├──────────────────────────────────────────────────────────┤
│         Distributed Training Core                         │
│        (DDP / FSDP / ZeRO / Pipeline)                    │
├──────────────────────────────────────────────────────────┤
│         Runtime & Resource Management                     │
└──────────────────────────────────────────────────────────┘
```

## 核心模块

### 1. 训练策略层 (`strategies/`)

提供可组合的训练策略：

- **StandardTrainingStrategy**: 标准训练策略
- **MultiModalStrategy**: 多模态训练策略
- **DistillationStrategy**: 知识蒸馏策略
- **ScenarioStrategy**: 场景化训练策略
- **DistributedStrategy**: 分布式训练策略

```python
from backend.modules.training.strategies import (
    create_strategy, create_composite_strategy
)

# 创建单个策略
multimodal_strategy = create_strategy('multimodal')

# 创建组合策略
composite = create_composite_strategy(
    ['standard', 'multimodal', 'distillation'],
    weights=[1.0, 0.5, 0.3]
)
```

### 2. 行业模型抽象层 (`industry/`)

定义行业模型的统一抽象：

```
IndustryModel = Backbone + ModalityAdapters + ScenarioHeads
```

支持的行业：
- **制造业 (ManufacturingModel)**: 设备故障预测、工艺优化、质量检测、能耗预测
- **金融 (FinanceModel)**: 风险评估、欺诈检测、信用评分
- **医疗 (HealthcareModel)**: 疾病诊断、医学影像分析、药物相互作用

```python
from backend.modules.training.industry import (
    create_industry_model, ManufacturingModel
)

# 创建制造业模型
model = create_industry_model('manufacturing')

# 或直接使用
model = ManufacturingModel()
```

### 3. 训练编排器 (`orchestrator/`)

支持多阶段训练流程编排：

```python
from backend.modules.training.orchestrator import (
    TrainingOrchestrator, TrainingPlan
)

# 创建编排器
orchestrator = TrainingOrchestrator(output_dir='./outputs')

# 创建行业三阶段训练计划
plan = orchestrator.create_industry_plan(
    model_name='manufacturing_model',
    include_pretrain=True,   # Stage 1: 行业表征预训练
    include_align=True,      # Stage 2: 行业能力对齐
    include_finetune=True    # Stage 3: 场景精调
)

# 提交任务
job = orchestrator.submit_job(plan)
```

### 4. 训练启动器 (`launcher/`)

统一的训练入口，支持多种训练模式：

```python
from backend.modules.training.launcher.training_launcher import (
    launch_training_system, create_industry_training_config
)

# 创建行业训练配置
config = create_industry_training_config(
    industry_type='manufacturing',
    model_name='my_model',
    output_dir='./outputs'
)

# 启动训练
result = launch_training_system(config)
```

## 行业三阶段训练

### Stage 1: 行业表征预训练

- **数据**: 行业无标注/弱标注数据
- **目标**: 行业语义与知识建模
- **策略**: 自监督/多模态对齐，大batch/大LR

### Stage 2: 行业能力对齐

- **数据**: 行业标注数据
- **目标**: 任务能力学习
- **策略**: 多任务训练，知识蒸馏

### Stage 3: 场景精调

- **数据**: 场景小样本
- **目标**: 业务指标最优
- **策略**: 冻结backbone，小LR

## 训练优先级

训练模式选择优先级（从高到低）：

1. **行业模型训练** - 完整的三阶段行业训练
2. **场景化训练** - 特定业务场景训练
3. **分布式训练** - 多机多卡训练
4. **知识蒸馏** - 模型压缩与知识迁移
5. **多模态训练** - 多模态数据联合训练
6. **三阶段训练** - 标准三阶段流程
7. **标准训练** - 基础训练

## 配置示例

### 制造业模型训练配置

```python
config = {
    'model': {
        'name': 'manufacturing_model',
        'type': 'industry'
    },
    'industry': {
        'enabled': True,
        'type': 'manufacturing',
        'include_pretrain': True,
        'include_align': True,
        'include_finetune': True,
        'use_multimodal': True,
        'stage_configs': {
            'pretrain_industry': {
                'epochs': 3,
                'learning_rate': 1e-4,
                'batch_size': 32
            },
            'align_industry': {
                'epochs': 5,
                'learning_rate': 2e-5,
                'batch_size': 16
            },
            'finetune_scene': {
                'epochs': 10,
                'learning_rate': 1e-5,
                'batch_size': 8,
                'freeze_backbone': True
            }
        }
    },
    'output_dir': './outputs/manufacturing',
    'training': {
        'num_epochs': 10,
        'batch_size': 16,
        'learning_rate': 2e-5
    },
    'data': {
        'train_path': './data/train',
        'val_path': './data/val'
    }
}
```

## 策略组合示例

```python
from backend.modules.training.strategies import (
    CompositeStrategy,
    MultiModalStrategy, MultiModalStrategyConfig,
    DistillationStrategy, DistillationStrategyConfig,
    ScenarioStrategy, ScenarioStrategyConfig
)

# 创建多模态策略
mm_config = MultiModalStrategyConfig(
    modalities=['text', 'image', 'time_series'],
    task_loss_weight=1.0,
    align_loss_weight=0.5
)
mm_strategy = MultiModalStrategy(mm_config)

# 创建蒸馏策略
dist_config = DistillationStrategyConfig(
    temperature=4.0,
    hard_loss_weight=1.0,
    soft_loss_weight=0.3
)
dist_strategy = DistillationStrategy(dist_config)

# 组合策略
composite = CompositeStrategy(
    strategies=[mm_strategy, dist_strategy],
    weights=[1.0, 0.5]
)
```

## 模块依赖

```
backend/modules/training/
├── strategies/              # 训练策略层
│   ├── base_strategy.py
│   ├── multimodal_strategy.py
│   ├── distillation_strategy.py
│   ├── scenario_strategy.py
│   └── distributed_strategy.py
├── industry/                # 行业模型抽象
│   └── industry_model.py
├── orchestrator/            # 训练编排
│   └── training_orchestrator.py
├── launcher/                # 训练启动器
│   └── training_launcher.py
├── three_stage/             # 三阶段训练
├── multimodal/              # 多模态训练
├── distillation/            # 知识蒸馏
├── scenarios/               # 场景管理
└── core/                    # 核心组件
```

## API 参考

### TrainingStrategy 接口

```python
class TrainingStrategy(ABC):
    def setup(self, context: StrategyContext) -> None:
        """策略初始化"""
        pass
    
    def prepare_batch(self, batch, context) -> Dict:
        """准备批次数据"""
        return batch
    
    @abstractmethod
    def compute_loss(self, model, batch, outputs, context) -> StrategyResult:
        """计算损失"""
        raise NotImplementedError
    
    def on_step_end(self, context, result) -> None:
        """步骤结束回调"""
        pass
```

### IndustryModel 接口

```python
class IndustryModel(nn.Module):
    def forward(self, inputs: Dict, scenario: str = None) -> Dict:
        """
        前向传播
        
        Args:
            inputs: 多模态输入字典
            scenario: 指定的场景
        
        Returns:
            输出字典，包含各场景头的预测
        """
        pass
    
    def add_scenario_head(self, name: str, num_classes: int) -> None:
        """添加场景头"""
        pass
```

### TrainingOrchestrator 接口

```python
class TrainingOrchestrator:
    def create_plan(self, model_name, stages, **kwargs) -> TrainingPlan:
        """创建训练计划"""
        pass
    
    def create_industry_plan(self, **kwargs) -> TrainingPlan:
        """创建行业训练计划"""
        pass
    
    def submit_job(self, plan: TrainingPlan) -> TrainingJob:
        """提交训练任务"""
        pass
    
    def execute_job(self, job, model, dataloaders, ...) -> Dict:
        """执行训练任务"""
        pass
```

## 注意事项

1. **数据准备**: 确保各阶段的数据加载器正确配置
2. **资源管理**: 大模型训练注意GPU内存管理
3. **检查点保存**: 启用中间模型保存以支持断点续训
4. **日志监控**: 配置适当的日志级别便于调试

## 扩展指南

### 添加新的训练策略

1. 继承 `TrainingStrategy` 基类
2. 实现 `compute_loss` 方法
3. 在 `strategies/__init__.py` 中注册

### 添加新的行业模型

1. 继承 `IndustryModel` 基类
2. 配置行业特定的模态和场景头
3. 在 `industry/__init__.py` 中注册


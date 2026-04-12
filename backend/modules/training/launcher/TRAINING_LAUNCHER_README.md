# 训练启动器 (Training Launcher)

训练启动器提供统一的训练入口，支持多种训练模式的自动选择和配置。

## 功能概述

### 支持的训练模式

训练启动器支持以下训练模式，按优先级从高到低排列：

1. **行业模型训练** - 针对特定行业的模型训练
2. **场景化训练** - 基于场景的训练管理
3. **分布式训练** - 多机多卡训练
4. **知识蒸馏训练** - 多场景知识蒸馏
5. **多模态训练** - 多模态模型训练
6. **三阶段训练** - 预训练、微调、偏好优化
7. **标准训练** - 通用训练流程

### 知识蒸馏场景

训练启动器支持多种知识蒸馏场景：

| 场景 | 说明 | 特点 |
|------|------|------|
| `standard` | 标准蒸馏 | 平衡精度和压缩率 |
| `edge_deploy` | 边缘部署蒸馏 | 高压缩率，低延迟 |
| `high_accuracy` | 高精度蒸馏 | 保持最高精度 |
| `industry` | 行业蒸馏 | 针对特定行业优化 |
| `multimodal` | 多模态蒸馏 | 支持多模态特征 |
| `progressive` | 渐进式蒸馏 | 分阶段蒸馏 |
| `self` | 自蒸馏 | 无需教师模型 |

## 快速开始

### 基本用法

```python
from backend.modules.training.launcher.training_launcher import (
    TrainingSystemLauncher,
    launch_training_system,
    create_distillation_training_config
)

# 创建蒸馏配置
config = create_distillation_training_config(
    scenario='edge_deploy',
    teacher_model_path='path/to/teacher',
    student_model_path='path/to/student',
    output_dir='./outputs'
)

# 启动训练
result = launch_training_system(config)
print(f"训练结果: {result}")
```

### 使用蒸馏服务

```python
# 启用蒸馏服务（支持租户管理和监控）
config = create_distillation_training_config(
    scenario='industry',
    use_service=True,
    tenant_id='tenant_001',
    user_id='user_001',
    task_name='my_distillation_task'
)

result = launch_training_system(config)
print(f"任务ID: {result.get('task_id')}")
```

## 配置示例

### 标准蒸馏配置

```python
config = create_distillation_training_config(
    scenario='standard',
    teacher_model_path='bert-base-uncased',
    student_model_path='bert-tiny',
    output_dir='./outputs/standard'
)
```

### 边缘部署蒸馏配置

```python
config = create_distillation_training_config(
    scenario='edge_deploy',
    teacher_model_path='bert-base-uncased',
    student_model_path='bert-tiny',
    output_dir='./outputs/edge',
    # 边缘部署场景默认设置
    # temperature=6.0, alpha=0.9, beta=0.1
)
```

### 行业蒸馏配置

```python
config = create_distillation_training_config(
    scenario='industry',
    teacher_model_path='industry-bert',
    student_model_path='industry-bert-tiny',
    industry_type='finance',  # manufacturing, healthcare, retail
    use_service=True,
    tenant_id='finance_tenant'
)
```

### 高精度蒸馏配置

```python
config = create_distillation_training_config(
    scenario='high_accuracy',
    teacher_model_path='bert-large',
    student_model_path='bert-base',
    output_dir='./outputs/high_acc',
    # 高精度场景默认设置
    # temperature=2.0, alpha=0.5, beta=0.5
)
```

## 高级用法

### 使用 TrainingSystemLauncher

```python
from backend.modules.training.launcher.training_launcher import TrainingSystemLauncher

# 自定义配置
config = {
    'distillation': {
        'enabled': True,
        'scenario': 'progressive',
        'use_service': False,
        'teacher_model_path': 'large-model',
        'student_model_path': 'small-model',
        'temperature': 4.0,
        'alpha': 0.7,
        'beta': 0.3,
        'use_feature_distillation': True,
        'use_attention_distillation': True,
        'progressive_stages': 4,
    },
    'output_dir': './outputs/progressive',
    'training': {
        'num_epochs': 20,
        'batch_size': 32,
        'learning_rate': 1e-4
    }
}

# 创建启动器
launcher = TrainingSystemLauncher(config)

# 分析配置
analysis = launcher.analyze_config()
print(f"配置分析: {analysis}")

# 选择训练器
trainer = launcher.select_trainer(analysis)
print(f"训练器类型: {type(trainer).__name__}")

# 启动训练
result = launcher.launch_training()
```

### 策略组合

```python
# 启动器自动根据场景选择策略
config = create_distillation_training_config(scenario='industry')
launcher = TrainingSystemLauncher(config)

analysis = launcher.analyze_config()
strategies = launcher._setup_strategies(analysis)

# 输出: ['industry_distillation', 'standard']
print([s.name for s in strategies])
```

## 场景默认配置

### standard（标准蒸馏）
```python
{
    'temperature': 4.0,
    'alpha': 0.7,
    'beta': 0.3,
}
```

### edge_deploy（边缘部署）
```python
{
    'temperature': 6.0,
    'alpha': 0.9,
    'beta': 0.1,
    'use_feature_distillation': True,
    'use_attention_distillation': False,
}
```

### high_accuracy（高精度）
```python
{
    'temperature': 2.0,
    'alpha': 0.5,
    'beta': 0.5,
    'use_feature_distillation': True,
    'use_attention_distillation': True,
}
```

### industry（行业蒸馏）
```python
{
    'temperature': 4.0,
    'alpha': 0.7,
    'beta': 0.3,
    'use_feature_distillation': True,
    'industry_type': 'manufacturing',  # 可配置
}
```

### multimodal（多模态）
```python
{
    'temperature': 4.0,
    'alpha': 0.6,
    'beta': 0.4,
    'use_contrastive_distillation': True,
}
```

### progressive（渐进式）
```python
{
    'temperature': 4.0,
    'alpha': 0.7,
    'beta': 0.3,
    'progressive_stages': 4,
}
```

## 服务模式特性

当 `use_service=True` 时，启用以下功能：

1. **租户隔离** - 多租户数据隔离
2. **任务管理** - 创建、启动、停止、监控任务
3. **进度跟踪** - 实时训练进度
4. **报告生成** - 训练报告和统计

### 服务模式示例

```python
config = create_distillation_training_config(
    scenario='industry',
    use_service=True,
    tenant_id='my_tenant',
    user_id='my_user',
    task_name='finance_model_distillation'
)

launcher = TrainingSystemLauncher(config)
trainer = launcher.select_trainer(launcher.analyze_config())

# 启动训练
result = trainer.train()
print(f"任务ID: {result['task_id']}")

# 获取状态
status = trainer.get_status()
print(f"进度: {status['progress']}%")

# 生成报告
report = trainer.generate_report()
print(f"最终损失: {report['final_loss']}")
```

## 与其他模块集成

### 与蒸馏服务集成

```python
from backend.modules.training.distillation import get_distillation_service

# 获取蒸馏服务
service = get_distillation_service()

# 推荐场景
recommendation = service.recommend_scenario({
    'target_device': 'mobile',
    'target_latency_ms': 50,
    'industry': 'finance'
})

# 使用推荐的场景创建配置
config = create_distillation_training_config(
    scenario=recommendation['recommended_scenario']
)
```

### 与场景管理器集成

```python
from backend.modules.training.distillation import get_scenario_manager

manager = get_scenario_manager()

# 获取可用场景
scenarios = manager.get_available_scenarios()

# 获取场景策略
strategy = manager.get_strategy_for_scenario(task_config)
```

## 训练结果

### 场景模式结果

```python
{
    'success': True,
    'scenario': 'edge_deploy',
    'strategy': 'distillation',
    'final_loss': 0.4267,
    'epochs_trained': 10,
    'best_loss': 0.4123
}
```

### 服务模式结果

```python
{
    'success': True,
    'task_id': 'a25246d1-7cc2-4207-be31-4594b80138de',
    'scenario': 'industry',
    'status': 'completed',
    'message': '蒸馏任务已启动: industry'
}
```

## 错误处理

训练启动器提供多级回退机制：

1. **蒸馏服务** → 场景训练器 → 基础训练器 → 统一训练器
2. 自动检测模块可用性
3. 详细的日志记录

```python
try:
    result = launch_training_system(config)
except Exception as e:
    logger.error(f"训练失败: {e}")
    # 自动回退到基础训练器
```

## 最佳实践

1. **场景选择**
   - 边缘设备部署：使用 `edge_deploy`
   - 追求高精度：使用 `high_accuracy`
   - 特定行业：使用 `industry` + `industry_type`

2. **服务模式**
   - 生产环境建议启用 `use_service=True`
   - 便于监控和管理

3. **配置优化**
   - 根据实际需求调整 `temperature`, `alpha`, `beta`
   - 大模型建议启用特征蒸馏和注意力蒸馏

4. **资源管理**
   - 设置合适的 `batch_size`
   - 监控显存使用


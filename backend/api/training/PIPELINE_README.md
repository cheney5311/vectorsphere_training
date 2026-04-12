# 训练流水线服务使用示例

本文档提供 `PipelineService` 的使用示例，包括流水线创建、执行、管理和模板功能。

## 目录

1. [服务初始化](#1-服务初始化)
2. [流水线创建与管理](#2-流水线创建与管理)
3. [流水线执行](#3-流水线执行)
4. [执行控制（暂停/恢复/回滚）](#4-执行控制)
5. [模板管理](#5-模板管理)
6. [API端点参考](#6-api端点参考)

---

## 1. 服务初始化

### Python 代码示例

```python
import uuid
from backend.services.pipeline_service import get_pipeline_service

# 创建服务实例（使用内存存储进行测试）
service = get_pipeline_service(use_memory_storage=True)
print('✅ PipelineService created')

# 生成测试用的租户ID和用户ID
tenant_id = str(uuid.uuid4())
user_id = str(uuid.uuid4())

print(f"Using tenant_id: {tenant_id[:8]}...")
print(f"Using user_id: {user_id[:8]}...")
```

---

## 2. 流水线创建与管理

### 创建流水线

```python
print('\n=== Testing create_pipeline() ===')

# 定义流水线步骤
steps_config = [
    {
        'name': 'data_preprocessing',
        'type': 'data_processing',
        'params': {'batch_size': 32, 'max_length': 512},
        'on_fail': 'stop'
    },
    {
        'name': 'model_pretrain',
        'type': 'pretrain',
        'params': {'epochs': 10, 'learning_rate': 0.001},
        'on_fail': 'rollback'
    },
    {
        'name': 'supervised_finetune',
        'type': 'sft',
        'params': {'epochs': 5, 'learning_rate': 0.0001},
        'on_fail': 'rollback'
    },
    {
        'name': 'preference_optimization',
        'type': 'preference_optim',
        'params': {'method': 'dpo', 'beta': 0.1},
        'on_fail': 'continue'
    },
    {
        'name': 'final_evaluation',
        'type': 'evaluation',
        'params': {'metrics': ['accuracy', 'f1', 'perplexity']},
        'on_fail': 'continue'
    }
]

result = service.create_pipeline(
    name='LLM_Training_Pipeline',
    steps_config=steps_config,
    tenant_id=tenant_id,
    user_id=user_id,
    description='Complete LLM training pipeline with SFT and DPO',
    model_name='llama-7b',
    enable_rollback=True,
    tags=['llm', 'training', 'production']
)

pipeline_id = result.get('pipeline_id')
print(f'Pipeline created: {pipeline_id}')
print(f'Status: {result.get("status")}')
print(f'Steps: {len(result.get("steps_config", []))}')
```

### 获取流水线详情

```python
print('\n=== Testing get_pipeline() ===')

pipeline = service.get_pipeline(pipeline_id, tenant_id)
print(f'Pipeline Name: {pipeline.get("name")}')
print(f'Status: {pipeline.get("status")}')
print(f'Enable Rollback: {pipeline.get("enable_rollback")}')
print(f'Steps:')
for idx, step in enumerate(pipeline.get('steps_config', [])):
    print(f'  {idx+1}. {step.get("name")} ({step.get("type")}) - on_fail: {step.get("on_fail")}')
```

### 获取流水线列表

```python
print('\n=== Testing list_pipelines() ===')

result = service.list_pipelines(
    tenant_id=tenant_id,
    status=None,
    limit=10
)

print(f'Found {result["total"]} pipelines')
for p in result['pipelines']:
    print(f'  - {p.get("name")}: {p.get("status")}')
```

### 更新流水线

```python
print('\n=== Testing update_pipeline() ===')

updated = service.update_pipeline(
    pipeline_id=pipeline_id,
    tenant_id=tenant_id,
    updates={
        'description': 'Updated: Production LLM training pipeline',
        'tags': ['llm', 'training', 'production', 'v2']
    }
)

if updated:
    print(f'Pipeline updated: {updated.get("description")}')
    print(f'Tags: {updated.get("tags")}')
```

### 获取统计信息

```python
print('\n=== Testing get_pipeline_statistics() ===')

stats = service.get_pipeline_statistics(tenant_id, user_id)
print(f'Total Pipelines: {stats.get("total")}')
print(f'By Status: {stats.get("by_status")}')
```

---

## 3. 流水线执行

### 启动流水线

```python
print('\n=== Testing start_pipeline() ===')

# 启动流水线执行
result = service.start_pipeline(
    pipeline_id=pipeline_id,
    tenant_id=tenant_id,
    user_id=user_id,
    session_id=f'session_{uuid.uuid4().hex[:8]}',
    runtime_config={
        'gpu_count': 4,
        'mixed_precision': True
    }
)

if result.get('success'):
    execution_id = result.get('execution_id')
    session_id = result.get('session_id')
    print(f'Pipeline started!')
    print(f'  Execution ID: {execution_id}')
    print(f'  Session ID: {session_id}')
    print(f'  Status: {result.get("status")}')
    print(f'  Total Steps: {result.get("total_steps")}')
else:
    print(f'Failed to start: {result.get("error")}')
```

### 获取执行状态

```python
print('\n=== Testing get_execution_status() ===')

import time
time.sleep(1)  # 等待执行开始

execution = service.get_execution_status(execution_id, tenant_id)
if execution:
    print(f'Execution Status: {execution.get("status")}')
    print(f'Progress: {execution.get("progress")}%')
    print(f'Current Step: {execution.get("current_step")}/{execution.get("total_steps")}')
    
    # 显示步骤详情
    if 'steps' in execution:
        print(f'Steps:')
        for step in execution['steps']:
            print(f'  - {step.get("step_name")}: {step.get("status")}')
```

### 获取执行记录列表

```python
print('\n=== Testing list_executions() ===')

result = service.list_executions(
    tenant_id=tenant_id,
    pipeline_id=pipeline_id,
    limit=10
)

print(f'Found {result["total"]} executions')
for e in result['executions']:
    print(f'  - {e.get("execution_id")}: {e.get("status")} ({e.get("progress")}%)')
```

---

## 4. 执行控制

### 暂停执行

```python
print('\n=== Testing pause_execution() ===')

result = service.pause_execution(session_id, tenant_id)
if result.get('success'):
    print(f'Execution paused: {session_id}')
else:
    print(f'Pause failed: {result.get("error")}')
```

### 恢复执行

```python
print('\n=== Testing resume_execution() ===')

result = service.resume_execution(session_id, tenant_id)
if result.get('success'):
    print(f'Execution resumed: {session_id}')
else:
    print(f'Resume failed: {result.get("error")}')
```

### 回滚执行

```python
print('\n=== Testing rollback_execution() ===')

result = service.rollback_execution(
    pipeline_id=pipeline_id,
    tenant_id=tenant_id,
    session_id=session_id
)

if result.get('success'):
    print(f'Pipeline rolled back!')
    print(f'Event: {result.get("event")}')
else:
    print(f'Rollback failed: {result.get("error")}')
```

---

## 5. 模板管理

### 创建模板

```python
print('\n=== Testing create_template() ===')

template_steps = [
    {
        'name': 'data_prep',
        'type': 'data_processing',
        'params': {'normalize': True}
    },
    {
        'name': 'train',
        'type': 'finetune',
        'params': {'epochs': '{{epochs}}', 'lr': '{{learning_rate}}'}
    },
    {
        'name': 'evaluate',
        'type': 'evaluation',
        'params': {'metrics': ['accuracy']}
    }
]

template = service.create_template(
    name='Basic Fine-tuning Template',
    steps_template=template_steps,
    tenant_id=tenant_id,
    user_id=user_id,
    description='Basic template for model fine-tuning',
    category='nlp',
    default_config={
        'epochs': 5,
        'learning_rate': 0.0001
    },
    required_params=['model_name', 'dataset_id'],
    tags=['finetune', 'basic']
)

template_id = template.get('template_id')
print(f'Template created: {template_id}')
print(f'Name: {template.get("name")}')
print(f'Category: {template.get("category")}')
```

### 获取模板列表

```python
print('\n=== Testing list_templates() ===')

result = service.list_templates(
    tenant_id=tenant_id,
    category='nlp',
    include_system=True,
    limit=10
)

print(f'Found {result["total"]} templates')
for t in result['templates']:
    print(f'  - {t.get("name")} ({t.get("template_type")}): {t.get("category")}')
```

### 从模板创建流水线

```python
print('\n=== Testing create_pipeline_from_template() ===')

result = service.create_pipeline_from_template(
    template_id=template_id,
    name='Production_Finetune_Pipeline',
    tenant_id=tenant_id,
    user_id=user_id,
    params={
        'epochs': 10,
        'learning_rate': 0.00005
    },
    description='Production fine-tuning pipeline created from template',
    model_name='bert-base',
    tags=['production', 'bert']
)

if result.get('success') != False:
    print(f'Pipeline created from template!')
    print(f'Pipeline ID: {result.get("pipeline_id")}')
    print(f'Steps: {len(result.get("steps_config", []))}')
else:
    print(f'Failed: {result.get("error")}')
```

---

## 6. API端点参考

### 流水线管理

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/training/pipeline/create` | 创建流水线 |
| GET | `/api/v1/training/pipeline/pipelines` | 获取流水线列表 |
| GET | `/api/v1/training/pipeline/pipelines/<id>` | 获取流水线详情 |
| PUT | `/api/v1/training/pipeline/pipelines/<id>` | 更新流水线 |
| DELETE | `/api/v1/training/pipeline/pipelines/<id>` | 删除流水线 |
| GET | `/api/v1/training/pipeline/status/<name>` | 获取流水线状态（兼容旧API） |
| GET | `/api/v1/training/pipeline/statistics` | 获取统计信息 |

### 流水线执行

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/training/pipeline/start` | 启动流水线 |
| POST | `/api/v1/training/pipeline/pause` | 暂停执行 |
| POST | `/api/v1/training/pipeline/resume` | 恢复执行 |
| POST | `/api/v1/training/pipeline/rollback` | 回滚流水线 |
| GET | `/api/v1/training/pipeline/executions` | 获取执行记录列表 |
| GET | `/api/v1/training/pipeline/executions/<id>` | 获取执行状态详情 |

### 模板管理

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/api/v1/training/pipeline/templates` | 获取模板列表 |
| POST | `/api/v1/training/pipeline/templates` | 创建模板 |
| GET | `/api/v1/training/pipeline/templates/<id>` | 获取模板详情 |
| POST | `/api/v1/training/pipeline/templates/<id>/create-pipeline` | 从模板创建流水线 |

### 其他

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/api/v1/training/pipeline/step-types` | 获取支持的步骤类型 |

---

## 步骤类型说明

支持的步骤类型：

| 类型 | 名称 | 描述 |
|------|------|------|
| `pretrain` | Pre-training | 模型预训练阶段 |
| `finetune` | Fine-tuning | 模型微调阶段 |
| `sft` | Supervised Fine-Tuning | 监督微调阶段 |
| `preference_optim` | Preference Optimization | DPO/RLHF偏好优化阶段 |
| `evaluation` | Evaluation | 模型评估阶段 |
| `validation` | Validation | 模型验证阶段 |
| `data_processing` | Data Processing | 数据预处理阶段 |
| `model_export` | Model Export | 模型导出/转换阶段 |
| `deployment` | Deployment | 模型部署阶段 |
| `checkpoint` | Checkpoint | 检查点保存阶段 |
| `custom` | Custom | 自定义步骤类型 |

---

## 失败策略说明

| 策略 | 描述 |
|------|------|
| `continue` | 失败后继续执行后续步骤 |
| `stop` | 失败后停止流水线执行 |
| `rollback` | 失败后回滚到上一步骤 |
| `retry` | 失败后重试当前步骤 |

---

## 流水线状态说明

| 状态 | 描述 |
|------|------|
| `draft` | 草稿状态 |
| `created` | 已创建，未执行 |
| `pending` | 等待执行 |
| `running` | 运行中 |
| `paused` | 已暂停 |
| `completed` | 已完成 |
| `failed` | 执行失败 |
| `cancelled` | 已取消 |
| `rolled_back` | 已回滚 |

---

## 执行状态说明

| 状态 | 描述 |
|------|------|
| `queued` | 排队中 |
| `initializing` | 初始化中 |
| `running` | 运行中 |
| `step_completed` | 步骤完成 |
| `paused` | 已暂停 |
| `resuming` | 恢复中 |
| `completing` | 完成中 |
| `completed` | 已完成 |
| `failed` | 失败 |
| `cancelled` | 已取消 |

---

## 完整示例：创建并执行LLM训练流水线

```python
import uuid
from backend.services.pipeline_service import get_pipeline_service

# 初始化
service = get_pipeline_service(use_memory_storage=True)
tenant_id = str(uuid.uuid4())
user_id = str(uuid.uuid4())

# 1. 创建流水线
pipeline = service.create_pipeline(
    name='GPT_Training_Pipeline',
    steps_config=[
        {'name': 'tokenize', 'type': 'data_processing', 'on_fail': 'stop'},
        {'name': 'pretrain', 'type': 'pretrain', 'on_fail': 'rollback',
         'params': {'epochs': 100, 'batch_size': 32}},
        {'name': 'sft', 'type': 'sft', 'on_fail': 'rollback',
         'params': {'epochs': 10, 'learning_rate': 1e-5}},
        {'name': 'dpo', 'type': 'preference_optim', 'on_fail': 'continue',
         'params': {'method': 'dpo', 'beta': 0.1}},
        {'name': 'eval', 'type': 'evaluation', 'on_fail': 'continue'}
    ],
    tenant_id=tenant_id,
    user_id=user_id,
    model_name='gpt-2',
    enable_rollback=True
)

pipeline_id = pipeline['pipeline_id']
print(f'Created pipeline: {pipeline_id}')

# 2. 启动执行
result = service.start_pipeline(
    pipeline_id=pipeline_id,
    tenant_id=tenant_id,
    user_id=user_id
)

if result['success']:
    print(f'Started execution: {result["execution_id"]}')
    print(f'Session: {result["session_id"]}')
else:
    print(f'Failed: {result["error"]}')
```

---

## 结论

本示例展示了 `PipelineService` 的核心功能。该服务提供了完整的训练流水线管理能力，包括：

- **流水线管理**：创建、更新、删除、查询流水线
- **执行控制**：启动、暂停、恢复、回滚流水线执行
- **步骤管理**：支持多种步骤类型和失败策略
- **模板系统**：创建和复用流水线模板
- **统计分析**：流水线和执行统计信息

所有操作都支持租户级别的数据隔离，确保多租户环境下的数据安全。



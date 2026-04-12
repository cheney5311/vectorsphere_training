# 三阶段训练服务使用示例

本文档提供 `ThreeStageTrainingService` 的使用示例，包括训练会话管理、训练控制、进度跟踪和报告功能。

## 目录

1. [服务初始化](#1-服务初始化)
2. [会话管理](#2-会话管理)
3. [训练控制](#3-训练控制)
4. [进度和报告](#4-进度和报告)
5. [统计信息](#5-统计信息)
6. [API端点参考](#6-api端点参考)

---

## 1. 服务初始化

### Python 代码示例

```python
import uuid
from backend.services.three_stage_training_service import get_three_stage_training_service

# 创建服务实例（使用内存存储进行测试）
service = get_three_stage_training_service(use_memory_storage=True)
print('✅ ThreeStageTrainingService created')

# 生成测试用的租户ID和用户ID
tenant_id = str(uuid.uuid4())
user_id = str(uuid.uuid4())

print(f"Using tenant_id: {tenant_id[:8]}...")
print(f"Using user_id: {user_id[:8]}...")
```

---

## 2. 会话管理

### 创建训练会话

```python
print('\n=== Testing create_session() ===')

# 定义训练配置
config = {
    'model_name': 'llama-7b',
    'base_model_path': 'meta-llama/Llama-2-7b-hf',
    'output_dir': './output/llama-7b-trained',
    'pass_model_between_stages': True,
    'stages': {
        'pt': {
            'enabled': True,
            'epochs': 10,
            'batch_size': 8,
            'learning_rate': 2e-5,
            'data_path': './data/pretrain'
        },
        'sft': {
            'enabled': True,
            'epochs': 5,
            'batch_size': 4,
            'learning_rate': 1e-5,
            'data_path': './data/sft'
        },
        'dpo': {
            'enabled': True,
            'epochs': 3,
            'batch_size': 2,
            'learning_rate': 5e-6,
            'beta': 0.1,
            'data_path': './data/dpo'
        }
    }
}

session = service.create_session(
    name='LLaMA-7B Full Training',
    model_name='llama-7b',
    config=config,
    tenant_id=tenant_id,
    user_id=user_id,
    description='Complete three-stage training for LLaMA-7B'
)

session_id = session.get('session_id')
print(f'Session created: {session_id}')
print(f'Status: {session.get("status")}')
print(f'Model: {session.get("model_name")}')
```

### 获取会话详情

```python
print('\n=== Testing get_session() ===')

session = service.get_session(session_id, tenant_id)
print(f'Session Name: {session.get("name")}')
print(f'Status: {session.get("status")}')
print(f'Progress: {session.get("progress")}%')
print(f'Config stages:')
for stage, cfg in session.get('config', {}).get('stages', {}).items():
    if cfg.get('enabled'):
        print(f'  - {stage}: {cfg.get("epochs")} epochs')
```

### 获取会话列表

```python
print('\n=== Testing list_sessions() ===')

result = service.list_sessions(
    tenant_id=tenant_id,
    user_id=user_id,
    limit=10
)

print(f'Found {result["total"]} sessions')
for s in result['sessions']:
    print(f'  - {s.get("name")}: {s.get("status")} ({s.get("progress")}%)')
```

### 更新会话

```python
print('\n=== Testing update_session() ===')

updated = service.update_session(
    session_id=session_id,
    tenant_id=tenant_id,
    updates={
        'description': 'Updated: Production LLaMA training pipeline'
    }
)

if updated:
    print(f'Session updated: {updated.get("description")}')
```

---

## 3. 训练控制

### 启动训练

```python
print('\n=== Testing start_training() ===')

result = service.start_training(session_id, tenant_id, user_id)

if result.get('success'):
    print(f'Training started!')
    print(f'  Session ID: {result.get("session_id")}')
    print(f'  Status: {result.get("status")}')
    print(f'  Started at: {result.get("started_at")}')
else:
    print(f'Failed to start: {result.get("error")}')
```

### 快捷方法：创建并启动

```python
print('\n=== Testing create_and_start() ===')

# 一步完成创建和启动
result = service.create_and_start(
    name='Quick Training Session',
    model_name='gpt2',
    config={
        'stages': {
            'sft': {'enabled': True, 'epochs': 3},
            'dpo': {'enabled': True, 'epochs': 2}
        }
    },
    tenant_id=tenant_id,
    user_id=user_id,
    description='Quick test training'
)

print(f'Session created and started: {result.get("session_id")}')
print(f'Start result: {result.get("start_result", {}).get("success")}')
```

### 暂停训练

```python
print('\n=== Testing pause_training() ===')

result = service.pause_training(session_id, tenant_id)
if result.get('success'):
    print(f'Training paused: {session_id}')
else:
    print(f'Pause failed: {result.get("error")}')
```

### 恢复训练

```python
print('\n=== Testing resume_training() ===')

result = service.resume_training(session_id, tenant_id)
if result.get('success'):
    print(f'Training resumed: {session_id}')
else:
    print(f'Resume failed: {result.get("error")}')
```

### 停止训练

```python
print('\n=== Testing stop_training() ===')

result = service.stop_training(session_id, tenant_id)
if result.get('success'):
    print(f'Training stopped: {session_id}')
    print(f'Completed at: {result.get("completed_at")}')
else:
    print(f'Stop failed: {result.get("error")}')
```

---

## 4. 进度和报告

### 获取训练进度

```python
print('\n=== Testing get_progress() ===')

progress = service.get_progress(session_id, tenant_id)
print(f'Session: {progress.get("session_id")}')
print(f'Status: {progress.get("status")}')
print(f'Overall Progress: {progress.get("progress")}%')
print(f'Current Stage: {progress.get("current_stage")}')
print(f'Stage Progress:')
for stage, info in progress.get('stages', {}).items():
    print(f'  - {stage}: {info.get("progress")}%')
```

### 获取进度历史

```python
print('\n=== Testing get_progress_history() ===')

history = service.get_progress_history(
    session_id=session_id,
    tenant_id=tenant_id,
    stage='sft',  # 可选：按阶段过滤
    limit=50
)

print(f'Found {history.get("total")} progress records')
for record in history.get('records', [])[:5]:
    print(f'  - {record.get("stage")} epoch {record.get("epoch")}: '
          f'loss={record.get("loss")}, acc={record.get("accuracy")}')
```

### 获取训练报告

```python
print('\n=== Testing get_report() ===')

report = service.get_report(session_id, tenant_id)
print(f'Session: {report.get("session_id")}')
print(f'Model: {report.get("model_name")}')
print(f'Status: {report.get("status")}')
print(f'Progress:')
print(f'  - Overall: {report.get("progress", {}).get("overall")}%')
print(f'  - Pretrain: {report.get("progress", {}).get("pretrain")}%')
print(f'  - Finetune: {report.get("progress", {}).get("finetune")}%')
print(f'  - Preference: {report.get("progress", {}).get("preference")}%')
```

---

## 5. 统计信息

### 获取统计信息

```python
print('\n=== Testing get_statistics() ===')

stats = service.get_statistics(tenant_id, user_id)
print(f'Total Sessions: {stats.get("total")}')
print(f'Running: {stats.get("running_count")}')
print(f'Completed: {stats.get("completed_count")}')
print(f'Failed: {stats.get("failed_count")}')
print(f'By Status: {stats.get("by_status")}')
print(f'By Model: {stats.get("by_model")}')
```

---

## 6. API端点参考

### 会话管理

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/training/three-stage/sessions` | 创建训练会话 |
| POST | `/api/v1/training/three-stage/start` | 创建并启动训练 |
| GET | `/api/v1/training/three-stage/sessions` | 获取会话列表 |
| GET | `/api/v1/training/three-stage/status` | 获取所有训练状态 |
| GET | `/api/v1/training/three-stage/<session_id>` | 获取会话详情 |
| PUT | `/api/v1/training/three-stage/<session_id>` | 更新会话 |
| DELETE | `/api/v1/training/three-stage/<session_id>` | 删除会话 |

### 训练控制

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/training/three-stage/<session_id>/start` | 启动训练 |
| POST | `/api/v1/training/three-stage/<session_id>/stop` | 停止训练 |
| POST | `/api/v1/training/three-stage/<session_id>/pause` | 暂停训练 |
| POST | `/api/v1/training/three-stage/<session_id>/resume` | 恢复训练 |

### 进度和报告

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/api/v1/training/three-stage/<session_id>/progress` | 获取训练进度 |
| GET | `/api/v1/training/three-stage/<session_id>/progress/history` | 获取进度历史 |
| GET | `/api/v1/training/three-stage/history` | 获取训练历史 |
| GET | `/api/v1/training/three-stage/report` | 获取报告列表 |
| GET | `/api/v1/training/three-stage/<session_id>/report` | 获取特定会话报告 |

### 统计

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/api/v1/training/three-stage/statistics` | 获取统计信息 |

---

## 训练阶段说明

三阶段训练包括以下阶段：

| 阶段 | 代码 | 描述 |
|------|------|------|
| 预训练 | `pt` / `pretrain` | 模型预训练，使用大规模无标注数据 |
| 监督微调 | `sft` / `finetune` | 有监督微调，使用标注数据 |
| 偏好优化 | `dpo` / `preference` | DPO/RLHF偏好学习，使用偏好数据 |

---

## 训练状态说明

| 状态 | 描述 |
|------|------|
| `pending` | 等待开始 |
| `running` | 运行中 |
| `paused` | 已暂停 |
| `stopped` | 已停止 |
| `completed` | 已完成 |
| `failed` | 失败 |
| `error` | 错误 |

---

## 配置格式示例

### 新格式（推荐）

```json
{
  "model_name": "llama-7b",
  "base_model": "meta-llama/Llama-2-7b-hf",
  "output_dir": "./output/llama-7b",
  "pass_model_between_stages": true,
  "stages": {
    "pretrain": {
      "enabled": true,
      "epochs": 10,
      "batch_size": 8,
      "learning_rate": 2e-5,
      "data_path": "./data/pretrain"
    },
    "finetune": {
      "enabled": true,
      "epochs": 5,
      "batch_size": 4,
      "learning_rate": 1e-5,
      "data_path": "./data/sft"
    },
    "preference": {
      "enabled": true,
      "epochs": 3,
      "batch_size": 2,
      "learning_rate": 5e-6,
      "beta": 0.1,
      "data_path": "./data/dpo"
    }
  }
}
```

### 旧格式（兼容）

```json
{
  "model_name": "llama-7b",
  "base_model": "meta-llama/Llama-2-7b-hf",
  "pt_enabled": true,
  "pt_epochs": 10,
  "pt_batch_size": 8,
  "pt_learning_rate": 2e-5,
  "pt_data_path": "./data/pretrain",
  "sft_enabled": true,
  "sft_epochs": 5,
  "sft_batch_size": 4,
  "sft_learning_rate": 1e-5,
  "sft_data_path": "./data/sft",
  "dpo_enabled": true,
  "dpo_epochs": 3,
  "dpo_batch_size": 2,
  "dpo_learning_rate": 5e-6,
  "dpo_beta": 0.1,
  "dpo_data_path": "./data/dpo"
}
```

---

## 完整示例：创建并执行三阶段训练

```python
import uuid
from backend.services.three_stage_training_service import get_three_stage_training_service

# 初始化
service = get_three_stage_training_service(use_memory_storage=True)
tenant_id = str(uuid.uuid4())
user_id = str(uuid.uuid4())

# 1. 创建并启动训练
result = service.create_and_start(
    name='GPT-2 Three Stage Training',
    model_name='gpt2',
    config={
        'base_model_path': 'gpt2',
        'output_dir': './output/gpt2-trained',
        'stages': {
            'sft': {'enabled': True, 'epochs': 3, 'batch_size': 8},
            'dpo': {'enabled': True, 'epochs': 2, 'batch_size': 4, 'beta': 0.1}
        }
    },
    tenant_id=tenant_id,
    user_id=user_id
)

session_id = result['session_id']
print(f'Created and started session: {session_id}')

# 2. 监控进度
import time
for _ in range(5):
    time.sleep(1)
    progress = service.get_progress(session_id, tenant_id)
    print(f'Progress: {progress.get("progress")}% - Stage: {progress.get("current_stage")}')
    if progress.get('status') == 'completed':
        break

# 3. 获取报告
report = service.get_report(session_id, tenant_id)
print(f'Final Status: {report.get("status")}')
print(f'Result: {report.get("result")}')
```

---

## 结论

本示例展示了 `ThreeStageTrainingService` 的核心功能。该服务提供了完整的三阶段训练管理能力，包括：

- **会话管理**：创建、更新、删除、查询训练会话
- **训练控制**：启动、暂停、恢复、停止训练
- **进度跟踪**：实时进度监控和历史记录
- **报告生成**：训练报告和统计分析
- **配置灵活**：支持新旧两种配置格式

所有操作都支持租户级别的数据隔离，确保多租户环境下的数据安全。


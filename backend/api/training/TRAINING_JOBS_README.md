# 训练任务 API 文档

## 概述

训练任务 API 提供完整的训练任务生命周期管理功能，支持多种训练场景（标准训练、分布式训练、多模态训练、知识蒸馏、三阶段训练等）。

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│  API Layer (training_jobs_api.py)                           │
│    └── RESTful API 端点                                      │
├─────────────────────────────────────────────────────────────┤
│  Service Layer (training_jobs_service.py)                   │
│    └── 业务逻辑处理                                          │
│        ├── 任务生命周期管理                                  │
│        ├── 训练执行控制                                      │
│        └── 统计和日志                                        │
├─────────────────────────────────────────────────────────────┤
│  Repository Layer (training_job_repository.py)              │
│    └── 数据持久化                                            │
│        ├── TrainingJob 模型                                  │
│        └── TrainingJobLog 模型                               │
├─────────────────────────────────────────────────────────────┤
│  Training Module (backend/modules/training)                 │
│    └── 实际训练执行                                          │
│        ├── TrainingLauncher                                  │
│        ├── ScenarioManager                                   │
│        └── 各种训练策略                                      │
└─────────────────────────────────────────────────────────────┘
```

## API 端点

### 任务 CRUD

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/training/jobs` | 创建训练任务 |
| GET | `/api/v1/training/jobs` | 获取任务列表 |
| GET | `/api/v1/training/jobs/<id>` | 获取任务详情 |
| DELETE | `/api/v1/training/jobs/<id>` | 删除任务 |

### 任务控制

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/training/jobs/<id>/start` | 开始任务 |
| POST | `/api/v1/training/jobs/<id>/pause` | 暂停任务 |
| POST | `/api/v1/training/jobs/<id>/resume` | 恢复任务 |
| POST | `/api/v1/training/jobs/<id>/cancel` | 取消任务 |
| POST | `/api/v1/training/jobs/<id>/restart` | 重启任务 |

### 日志和指标

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/api/v1/training/jobs/<id>/logs` | 获取任务日志 |
| GET | `/api/v1/training/jobs/<id>/metrics` | 获取任务指标 |
| GET | `/api/v1/training/jobs/<id>/checkpoints` | 获取检查点 |

### 统计信息

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/api/v1/training/statistics` | 获取训练统计 |
| GET | `/api/v1/training/statistics/by-scenario` | 按场景统计 |

### 批量操作

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/training/jobs/batch/cancel` | 批量取消任务 |

## 使用示例

### 1. 创建训练任务

```bash
curl -X POST http://localhost:5000/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "name": "LLaMA微调任务",
    "model_name": "llama-7b",
    "scenario_type": "three_stage",
    "config": {
      "epochs": 10,
      "batch_size": 32,
      "learning_rate": 1e-4,
      "pretrain": {
        "enabled": true,
        "data_path": "/data/pretrain",
        "num_epochs": 3
      },
      "finetune": {
        "enabled": true,
        "data_path": "/data/finetune",
        "num_epochs": 5
      },
      "preference": {
        "enabled": true,
        "data_path": "/data/dpo",
        "num_epochs": 2
      }
    },
    "schedule": {
      "type": "immediate",
      "priority": "high"
    }
  }'
```

响应：
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "LLaMA微调任务",
  "status": "pending",
  "scenario_type": "three_stage",
  "created_at": "2025-01-09T10:00:00Z"
}
```

### 2. 获取任务列表

```bash
curl -X GET "http://localhost:5000/api/v1/training/jobs?status=running&page=1&per_page=10" \
  -H "Authorization: Bearer <token>"
```

响应：
```json
{
  "success": true,
  "data": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "LLaMA微调任务",
      "status": "running",
      "scenario_type": "three_stage",
      "progress": 45.5,
      "created_at": "2025-01-09T10:00:00Z"
    }
  ],
  "page": 1,
  "limit": 10,
  "total": 1,
  "message": "获取训练任务列表成功"
}
```

### 3. 获取任务详情

```bash
curl -X GET http://localhost:5000/api/v1/training/jobs/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer <token>"
```

响应：
```json
{
  "success": true,
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "LLaMA微调任务",
    "status": "running",
    "scenario_type": "three_stage",
    "training_mode": "standard",
    "model_name": "llama-7b",
    "progress": 45.5,
    "current_epoch": 5,
    "total_epochs": 10,
    "metrics": {
      "loss": 0.15,
      "accuracy": 0.92
    },
    "config": {...},
    "created_at": "2025-01-09T10:00:00Z",
    "started_at": "2025-01-09T10:01:00Z"
  },
  "message": "获取训练任务详情成功"
}
```

### 4. 暂停任务

```bash
curl -X POST http://localhost:5000/api/v1/training/jobs/550e8400-e29b-41d4-a716-446655440000/pause \
  -H "Authorization: Bearer <token>"
```

响应：
```json
{
  "success": true,
  "data": {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "paused",
    "checkpoint_path": "/checkpoints/job_550e8400/epoch_5"
  },
  "message": "训练任务已暂停"
}
```

### 5. 恢复任务（可选指定检查点）

```bash
curl -X POST http://localhost:5000/api/v1/training/jobs/550e8400-e29b-41d4-a716-446655440000/resume \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "checkpoint_path": "/checkpoints/job_550e8400/epoch_3"
  }'
```

### 6. 获取任务指标

```bash
curl -X GET "http://localhost:5000/api/v1/training/jobs/550e8400-e29b-41d4-a716-446655440000/metrics?start_epoch=1&end_epoch=5" \
  -H "Authorization: Bearer <token>"
```

响应：
```json
{
  "success": true,
  "data": {
    "current_metrics": {
      "loss": 0.15,
      "accuracy": 0.92
    },
    "history": [
      {"epoch": 1, "loss": 0.5, "accuracy": 0.7},
      {"epoch": 2, "loss": 0.35, "accuracy": 0.8},
      {"epoch": 3, "loss": 0.25, "accuracy": 0.85},
      {"epoch": 4, "loss": 0.2, "accuracy": 0.88},
      {"epoch": 5, "loss": 0.15, "accuracy": 0.92}
    ]
  },
  "message": "获取指标成功"
}
```

### 7. 获取训练统计

```bash
curl -X GET http://localhost:5000/api/v1/training/statistics \
  -H "Authorization: Bearer <token>"
```

响应：
```json
{
  "success": true,
  "data": {
    "total_jobs": 150,
    "running_jobs": 5,
    "pending_jobs": 10,
    "completed_jobs": 120,
    "failed_jobs": 10,
    "cancelled_jobs": 5,
    "success_rate": 0.92,
    "average_duration": 3600
  },
  "message": "获取训练统计信息成功"
}
```

### 8. 批量取消任务

```bash
curl -X POST http://localhost:5000/api/v1/training/jobs/batch/cancel \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "job_ids": ["job-1", "job-2", "job-3"]
  }'
```

响应：
```json
{
  "success": true,
  "data": {
    "cancelled": ["job-1", "job-2"],
    "failed": [{"id": "job-3", "reason": "任务不存在"}]
  },
  "message": "批量取消完成"
}
```

## 任务状态流转

```
                    ┌─────────┐
                    │ pending │
                    └────┬────┘
                         │ start
                         ▼
            ┌────────────────────────┐
            │                        │
  pause     │       running          │ complete
    ┌───────┤                        ├───────┐
    │       └────────────────────────┘       │
    │                  │                     │
    ▼                  │ cancel              ▼
┌────────┐             │              ┌───────────┐
│ paused │             │              │ completed │
└───┬────┘             │              └───────────┘
    │ resume           │
    │                  │
    └──────────────────┼──────────────────────┐
                       │                      │
                       ▼                      ▼
                ┌───────────┐          ┌──────────┐
                │ cancelled │          │  failed  │
                └───────────┘          └──────────┘
```

## 支持的训练场景

| 场景类型 | 描述 |
|---------|------|
| `standard` | 标准单机训练 |
| `distributed` | 分布式训练（DDP/FSDP/DeepSpeed） |
| `multimodal` | 多模态训练（文本+图像+音频等） |
| `distillation` | 知识蒸馏训练 |
| `three_stage` | 三阶段训练（预训练+微调+偏好优化） |
| `industry` | 行业模型训练 |
| `scenario` | 场景化训练 |

## 数据模型

### TrainingJob

| 字段 | 类型 | 描述 |
|------|------|------|
| job_id | String | 任务唯一标识 |
| user_id | String | 用户ID |
| tenant_id | String | 租户ID |
| name | String | 任务名称 |
| description | Text | 任务描述 |
| scenario_type | String | 训练场景类型 |
| training_mode | String | 训练模式 |
| config | JSON | 完整训练配置 |
| status | String | 任务状态 |
| progress | Float | 进度(0-100) |
| current_epoch | Integer | 当前轮次 |
| total_epochs | Integer | 总轮次 |
| metrics | JSON | 当前指标 |
| result | JSON | 训练结果 |
| error_message | Text | 错误信息 |
| created_at | DateTime | 创建时间 |
| started_at | DateTime | 开始时间 |
| completed_at | DateTime | 完成时间 |

### TrainingJobLog

| 字段 | 类型 | 描述 |
|------|------|------|
| job_id | String | 任务ID |
| log_type | String | 日志类型 |
| log_level | String | 日志级别 |
| message | Text | 日志消息 |
| details | JSON | 详细信息 |
| epoch | Integer | 轮次 |
| step | Integer | 步骤 |
| metrics | JSON | 指标快照 |
| created_at | DateTime | 创建时间 |

## 错误码

| HTTP 状态码 | 错误类型 | 描述 |
|------------|---------|------|
| 400 | ValidationError | 请求参数验证失败 |
| 400 | BusinessLogicError | 业务逻辑错误（如状态不允许的操作） |
| 404 | NotFound | 任务不存在 |
| 500 | InternalError | 服务器内部错误 |

## 与训练模块集成

Service 层通过以下方式与 `backend/modules/training` 集成：

1. **TrainingLauncher**: 选择合适的训练器
2. **ScenarioManager**: 管理训练场景
3. **各种训练策略**: 
   - StandardStrategy
   - DistributedStrategy
   - MultiModalStrategy
   - DistillationStrategy
   - ThreeStageStrategy

```python
# 示例：Service 层如何调用训练模块
from backend.modules.training.launcher import TrainingSystemLauncher

launcher = TrainingSystemLauncher(config)
analysis = launcher.analyze_config()
trainer = launcher.select_trainer(analysis)
result = trainer.train()
```


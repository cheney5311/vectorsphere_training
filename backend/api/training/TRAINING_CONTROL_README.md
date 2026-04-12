# 训练任务控制 API 文档

## 概述

训练任务控制模块提供完整的训练任务生命周期管理，包括创建、启动、暂停、恢复、取消等操作。该模块采用三层架构设计，确保业务逻辑与数据访问分离。

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                         API 层                                   │
│              training_control_api.py                             │
│    提供 RESTful API 接口，处理 HTTP 请求/响应                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                       Service 层                                 │
│             training_control_service.py                          │
│    业务逻辑处理、任务调度、进度监控、日志记录                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     Repository 层                                │
│             training_job_repository.py                           │
│    数据持久化、CRUD 操作、统计查询                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                      Schema 层                                   │
│         training_models.py (TrainingJob/TrainingJobLog)          │
│    数据模型定义、字段约束                                        │
└─────────────────────────────────────────────────────────────────┘
```

## 数据模型

### TrainingJob（训练任务）

| 字段 | 类型 | 说明 |
|------|------|------|
| job_id | String(64) | 任务唯一标识 |
| tenant_id | UUID | 租户 ID |
| user_id | String(36) | 用户 ID |
| name | String(200) | 任务名称 |
| description | Text | 任务描述 |
| scenario_type | String(50) | 训练场景类型 |
| training_mode | String(50) | 训练模式 |
| status | String(20) | 任务状态 |
| progress | Float | 进度 (0-100) |
| current_epoch | Integer | 当前轮次 |
| total_epochs | Integer | 总轮次 |
| current_step | Integer | 当前步骤 |
| total_steps | Integer | 总步骤数 |
| metrics | JSON | 当前指标 |
| best_metrics | JSON | 最佳指标 |
| config | JSON | 训练配置 |
| resource_config | JSON | 资源配置 |
| checkpoint_path | String(500) | 检查点路径 |
| started_at | DateTime | 开始时间 |
| completed_at | DateTime | 完成时间 |
| error_message | Text | 错误信息 |

### TrainingJobLog（训练任务日志）

| 字段 | 类型 | 说明 |
|------|------|------|
| job_id | String(64) | 任务 ID |
| log_type | String(20) | 日志类型 |
| log_level | String(10) | 日志级别 |
| message | Text | 日志消息 |
| from_status | String(20) | 变更前状态 |
| to_status | String(20) | 变更后状态 |
| epoch | Integer | 轮次 |
| step | Integer | 步骤 |
| metrics | JSON | 指标快照 |

## API 接口

### 1. 创建训练任务

```http
POST /api/v1/training/control/jobs
```

**请求体：**
```json
{
    "name": "模型训练任务",
    "description": "训练描述",
    "scenario_type": "supervised",
    "training_mode": "standard",
    "model_id": "model_001",
    "model_name": "bert-base",
    "dataset_id": "dataset_001",
    "priority": 5,
    "config": {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001
    },
    "resource_config": {
        "gpu_count": 1,
        "memory_gb": 16
    },
    "tags": ["nlp", "classification"]
}
```

**响应：**
```json
{
    "success": true,
    "data": {
        "job_id": "job_abc123",
        "status": "pending",
        "message": "训练任务创建成功",
        "job": {...}
    },
    "message": "训练任务创建成功"
}
```

### 2. 启动训练任务

```http
POST /api/v1/training/control/jobs/{job_id}/start
```

**响应：**
```json
{
    "success": true,
    "data": {
        "job_id": "job_abc123",
        "status": "running",
        "message": "训练任务已开始"
    },
    "message": "训练任务已开始"
}
```

### 3. 暂停训练任务

```http
POST /api/v1/training/control/jobs/{job_id}/pause
```

### 4. 恢复训练任务

```http
POST /api/v1/training/control/jobs/{job_id}/resume
```

### 5. 取消训练任务

```http
POST /api/v1/training/control/jobs/{job_id}/cancel
```

**请求体（可选）：**
```json
{
    "reason": "取消原因"
}
```

### 6. 重新开始训练任务

```http
POST /api/v1/training/control/jobs/{job_id}/restart
```

**请求体（可选）：**
```json
{
    "from_checkpoint": true
}
```

### 7. 获取任务状态

```http
GET /api/v1/training/control/jobs/{job_id}/status
```

**响应：**
```json
{
    "success": true,
    "data": {
        "job_id": "job_abc123",
        "name": "模型训练任务",
        "status": "running",
        "progress": 45.5,
        "current_epoch": 5,
        "total_epochs": 10,
        "current_step": 1500,
        "total_steps": 3000,
        "metrics": {
            "loss": 0.234,
            "accuracy": 0.89
        },
        "started_at": "2024-01-15T10:30:00Z",
        "duration_seconds": 3600
    }
}
```

### 8. 获取任务进度

```http
GET /api/v1/training/control/jobs/{job_id}/progress
```

**响应：**
```json
{
    "success": true,
    "data": {
        "job_id": "job_abc123",
        "progress": 45.5,
        "current_epoch": 5,
        "total_epochs": 10,
        "current_step": 1500,
        "total_steps": 3000,
        "eta": 3600,
        "metrics": {
            "loss": 0.234,
            "accuracy": 0.89,
            "learning_rate": 0.0001
        },
        "best_metrics": {
            "loss": 0.189,
            "accuracy": 0.92
        },
        "checkpoint_path": "/checkpoints/job_abc123/epoch_5",
        "checkpoint_epoch": 5
    }
}
```

### 9. 获取任务列表

```http
GET /api/v1/training/control/jobs
```

**查询参数：**
- `status`: 状态过滤
- `scenario_type`: 场景类型过滤
- `user_id`: 用户 ID 过滤
- `limit`: 限制数量（默认 100）
- `offset`: 偏移量（默认 0）

**响应：**
```json
{
    "success": true,
    "data": {
        "jobs": [...],
        "total": 50,
        "limit": 100,
        "offset": 0
    }
}
```

### 10. 获取统计信息

```http
GET /api/v1/training/control/statistics
```

**响应：**
```json
{
    "success": true,
    "data": {
        "total_jobs": 100,
        "pending_jobs": 5,
        "running_jobs": 3,
        "completed_jobs": 80,
        "failed_jobs": 10,
        "paused_jobs": 2,
        "cancelled_jobs": 0,
        "scheduler_running": true,
        "max_concurrent_jobs": 3,
        "queue_size": 5
    }
}
```

### 11. 获取任务日志

```http
GET /api/v1/training/control/jobs/{job_id}/logs
```

**查询参数：**
- `log_type`: 日志类型过滤（status_change/progress/error/checkpoint/metric）
- `limit`: 限制数量（默认 100）

### 12. 批量操作

```http
POST /api/v1/training/control/jobs/batch/start
POST /api/v1/training/control/jobs/batch/cancel
```

**请求体：**
```json
{
    "job_ids": ["job_001", "job_002", "job_003"],
    "reason": "批量操作原因"
}
```

## 使用示例

### Python 客户端示例

```python
import requests

BASE_URL = "http://localhost:5000/api/v1/training/control"
HEADERS = {
    "Authorization": "Bearer <token>",
    "X-Tenant-ID": "tenant_001",
    "Content-Type": "application/json"
}

# 1. 创建训练任务
def create_job():
    response = requests.post(
        f"{BASE_URL}/jobs",
        headers=HEADERS,
        json={
            "name": "BERT 分类模型训练",
            "scenario_type": "classification",
            "training_mode": "standard",
            "config": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 2e-5
            }
        }
    )
    return response.json()

# 2. 启动训练
def start_job(job_id):
    response = requests.post(
        f"{BASE_URL}/jobs/{job_id}/start",
        headers=HEADERS
    )
    return response.json()

# 3. 监控进度
def get_progress(job_id):
    response = requests.get(
        f"{BASE_URL}/jobs/{job_id}/progress",
        headers=HEADERS
    )
    return response.json()

# 4. 暂停训练
def pause_job(job_id):
    response = requests.post(
        f"{BASE_URL}/jobs/{job_id}/pause",
        headers=HEADERS
    )
    return response.json()

# 使用示例
result = create_job()
job_id = result["data"]["job_id"]
print(f"Created job: {job_id}")

start_result = start_job(job_id)
print(f"Started: {start_result}")

import time
while True:
    progress = get_progress(job_id)
    print(f"Progress: {progress['data']['progress']}%")
    if progress["data"]["progress"] >= 100:
        break
    time.sleep(5)
```

### 与训练模块集成

```python
from backend.services.training_control_service import (
    get_training_control_service,
    TrainingJobConfig
)

# 获取服务实例
service = get_training_control_service()

# 创建任务配置
config = TrainingJobConfig(
    name="三阶段训练任务",
    scenario_type="three_stage",
    training_mode="three_stage",
    config={
        "pretrain": {"enabled": True, "epochs": 5},
        "finetune": {"enabled": True, "epochs": 10},
        "preference": {"enabled": True, "epochs": 3}
    }
)

# 创建任务
result = service.create_job(
    tenant_id="tenant_001",
    user_id="user_001",
    config=config
)

# 启动训练
job_id = result["job_id"]
service.start_job(job_id, "tenant_001", "user_001")

# 注册进度回调
def on_progress(progress, metrics):
    print(f"Progress: {progress}%, Loss: {metrics.get('loss')}")

service._progress_callbacks[job_id] = on_progress
```

## 任务状态流转

```
                    ┌─────────────┐
                    │   pending   │
                    └──────┬──────┘
                           │ start
                    ┌──────▼──────┐
           ┌───────>│   running   │<───────┐
           │        └──────┬──────┘        │
           │               │               │
      resume│          pause│          restart
           │               │               │
           │        ┌──────▼──────┐        │
           └────────│   paused    │────────┘
                    └─────────────┘
                           │
                           │ cancel
                    ┌──────▼──────┐
         ┌─────────>│  cancelled  │
         │          └─────────────┘
         │
    cancel│
         │          ┌─────────────┐
         ├─────────>│   failed    │
         │          └─────────────┘
         │
         │          ┌─────────────┐
         └─────────>│  completed  │
                    └─────────────┘
```

## 错误处理

所有 API 接口在出错时返回统一的错误格式：

```json
{
    "success": false,
    "error": "错误描述",
    "code": 400
}
```

常见错误码：
- `400`: 请求参数错误
- `404`: 资源不存在
- `500`: 服务器内部错误

## 配置说明

### 训练场景类型

| 场景类型 | 说明 |
|---------|------|
| supervised | 监督学习 |
| unsupervised | 无监督学习 |
| semi_supervised | 半监督学习 |
| reinforcement | 强化学习 |
| transfer | 迁移学习 |
| fine_tuning | 微调 |
| classification | 分类任务 |
| regression | 回归任务 |

### 训练模式

| 模式 | 说明 |
|------|------|
| standard | 标准训练 |
| distributed | 分布式训练 |
| multimodal | 多模态训练 |
| distillation | 知识蒸馏 |
| three_stage | 三阶段训练 |
| scenario | 场景化训练 |
| industry | 行业模型训练 |


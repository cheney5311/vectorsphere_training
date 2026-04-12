# Scheduler API 文档

## 概述

Scheduler API 提供生产级的任务调度管理功能，包括任务调度、周期性任务、模板管理、执行监控等。

## 架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Scheduler API Layer                                  │
│                      (scheduler_api.py)                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                          Service Layer                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    SchedulerService                                  │    │
│  │                 (scheduler_service.py)                               │    │
│  │  ─────────────────────────────────────────────────────────────────  │    │
│  │  - 任务调度与执行         - 模板管理                                 │    │
│  │  - 周期性任务             - 依赖管理                                 │    │
│  │  - 重试与超时控制         - 统计与指标                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Repository Layer                                     │
│                    (scheduler_repository.py)                                 │
│  ┌────────────────┬────────────────┬────────────────┬────────────────┐      │
│  │  ScheduledTask │  ExecutionLog  │   Template     │   Metrics      │      │
│  │     Repo       │     Repo       │     Repo       │     Repo       │      │
│  └────────────────┴────────────────┴────────────────┴────────────────┘      │
├─────────────────────────────────────────────────────────────────────────────┤
│                          Schema Layer                                        │
│                    (scheduler_models.py)                                     │
│            (SQLAlchemy ORM Models & Dataclasses & Enums)                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 功能模块

### 1. 调度器控制

#### 1.1 获取调度器状态
```http
GET /api/v1/scheduler/status
Authorization: Bearer <token>
```

**响应：**
```json
{
  "success": true,
  "data": {
    "running": true,
    "executing_count": 2,
    "max_concurrent": 10,
    "statistics": {
      "total": 50,
      "by_status": {
        "scheduled": 20,
        "completed": 25,
        "failed": 5
      }
    }
  }
}
```

#### 1.2 启动调度器
```http
POST /api/v1/scheduler/start
Authorization: Bearer <token>
```

#### 1.3 停止调度器
```http
POST /api/v1/scheduler/stop
Authorization: Bearer <token>
```

### 2. 任务调度

#### 2.1 调度单次任务
```http
POST /api/v1/scheduler/tasks
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体：**
```json
{
  "task_config": {
    "model": "gpt2",
    "training": {
      "epochs": 3,
      "batch_size": 16,
      "learning_rate": 5e-5
    }
  },
  "schedule_time": "2026-01-12T10:00:00Z",
  "name": "GPT2 Training Job",
  "priority": "high",
  "task_type": "training",
  "template_id": "builtin_basic_text_generation",
  "max_retries": 3,
  "timeout_seconds": 3600,
  "depends_on": ["task_abc123"],
  "tags": ["production", "gpt2"],
  "metadata": {
    "project": "nlp-pipeline"
  }
}
```

**参数说明：**
| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| task_config | object | 是 | 任务配置 |
| schedule_time | string | 是 | 调度时间 (ISO 8601) |
| name | string | 否 | 任务名称 |
| priority | string | 否 | 优先级: low, normal, high, urgent, critical |
| task_type | string | 否 | 任务类型: training, evaluation, inference, data_processing, model_export, cleanup, backup, custom |
| task_id | string | 否 | 自定义任务ID |
| template_id | string | 否 | 使用的模板ID |
| max_retries | integer | 否 | 最大重试次数 (默认: 3) |
| timeout_seconds | integer | 否 | 超时秒数 |
| depends_on | array | 否 | 依赖的任务ID列表 |
| tags | array | 否 | 标签 |
| metadata | object | 否 | 自定义元数据 |

**响应：**
```json
{
  "success": true,
  "data": {
    "id": "task_abc123def456",
    "name": "GPT2 Training Job",
    "task_type": "training",
    "schedule_time": "2026-01-12T10:00:00Z",
    "status": "scheduled",
    "priority": "high",
    "created_at": "2026-01-11T12:00:00Z"
  },
  "message": "Task scheduled successfully"
}
```

#### 2.2 调度周期性任务
```http
POST /api/v1/scheduler/tasks/recurring
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体（Cron 方式）：**
```json
{
  "task_config": {
    "type": "backup",
    "target": "/models"
  },
  "cron_expression": "0 2 * * *",
  "name": "Daily Model Backup",
  "priority": "normal",
  "task_type": "backup"
}
```

**请求体（间隔方式）：**
```json
{
  "task_config": {
    "type": "cleanup",
    "max_age_days": 30
  },
  "interval_seconds": 3600,
  "name": "Hourly Cleanup",
  "task_type": "cleanup"
}
```

### 3. 任务管理

#### 3.1 列出任务
```http
GET /api/v1/scheduler/tasks?status=scheduled&priority=high&limit=50&offset=0
Authorization: Bearer <token>
```

**查询参数：**
| 参数 | 类型 | 描述 |
|------|------|------|
| status | string | 状态筛选 |
| priority | string | 优先级筛选 |
| task_type | string | 类型筛选 |
| limit | integer | 限制数量 (默认: 100) |
| offset | integer | 偏移量 (默认: 0) |

#### 3.2 获取任务详情
```http
GET /api/v1/scheduler/tasks/<task_id>
Authorization: Bearer <token>
```

#### 3.3 更新任务
```http
PUT /api/v1/scheduler/tasks/<task_id>
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体：**
```json
{
  "name": "Updated Task Name",
  "priority": "urgent",
  "config": {
    "epochs": 5
  },
  "schedule_time": "2026-01-13T10:00:00Z"
}
```

#### 3.4 删除任务
```http
DELETE /api/v1/scheduler/tasks/<task_id>
Authorization: Bearer <token>
```

#### 3.5 取消任务
```http
POST /api/v1/scheduler/tasks/<task_id>/cancel
Authorization: Bearer <token>
```

#### 3.6 暂停任务
```http
POST /api/v1/scheduler/tasks/<task_id>/pause
Authorization: Bearer <token>
```

#### 3.7 恢复任务
```http
POST /api/v1/scheduler/tasks/<task_id>/resume
Authorization: Bearer <token>
```

#### 3.8 重试任务
```http
POST /api/v1/scheduler/tasks/<task_id>/retry
Authorization: Bearer <token>
```

#### 3.9 获取任务日志
```http
GET /api/v1/scheduler/tasks/<task_id>/logs?limit=100
Authorization: Bearer <token>
```

### 4. 模板管理

#### 4.1 列出模板
```http
GET /api/v1/scheduler/templates?category=training&include_system=true
Authorization: Bearer <token>
```

**响应：**
```json
{
  "success": true,
  "data": [
    {
      "id": "builtin_basic_text_generation",
      "name": "basic_text_generation",
      "description": "基础文本生成模型训练",
      "category": "builtin",
      "task_type": "training",
      "is_system": true,
      "usage_count": 150
    }
  ],
  "count": 10
}
```

#### 4.2 创建模板
```http
POST /api/v1/scheduler/templates
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体：**
```json
{
  "name": "Custom BERT Training",
  "description": "BERT 模型微调模板",
  "category": "nlp",
  "task_type": "training",
  "config_template": {
    "model": {
      "type": "bert",
      "name": "bert-base-uncased"
    },
    "training": {
      "epochs": 3,
      "batch_size": 16,
      "learning_rate": 2e-5
    }
  },
  "default_priority": "normal",
  "default_timeout_seconds": 7200,
  "default_max_retries": 3,
  "parameters": {
    "epochs": {
      "type": "integer",
      "min": 1,
      "max": 100,
      "default": 3
    }
  },
  "tags": ["bert", "nlp", "fine-tuning"]
}
```

#### 4.3 获取模板
```http
GET /api/v1/scheduler/templates/<template_id>
Authorization: Bearer <token>
```

#### 4.4 更新模板
```http
PUT /api/v1/scheduler/templates/<template_id>
Authorization: Bearer <token>
Content-Type: application/json
```

#### 4.5 删除模板
```http
DELETE /api/v1/scheduler/templates/<template_id>
Authorization: Bearer <token>
```

### 5. 统计与指标

#### 5.1 获取统计信息
```http
GET /api/v1/scheduler/statistics
Authorization: Bearer <token>
```

**响应：**
```json
{
  "success": true,
  "data": {
    "total": 100,
    "by_status": {
      "pending": 5,
      "scheduled": 20,
      "executing": 3,
      "completed": 60,
      "failed": 10,
      "cancelled": 2
    },
    "by_priority": {
      "low": 10,
      "normal": 50,
      "high": 30,
      "urgent": 10
    },
    "by_type": {
      "training": 70,
      "evaluation": 20,
      "inference": 10
    },
    "service_counters": {
      "tasks_scheduled": 100,
      "tasks_executed": 95,
      "tasks_completed": 60,
      "tasks_failed": 10
    },
    "executing_count": 3
  }
}
```

#### 5.2 获取指标
```http
GET /api/v1/scheduler/metrics?period_type=hour&limit=24
Authorization: Bearer <token>
```

### 6. 健康检查

```http
GET /api/v1/scheduler/health
```

**响应：**
```json
{
  "success": true,
  "status": "healthy",
  "scheduler_running": true,
  "timestamp": "2026-01-11T12:00:00Z"
}
```

## 任务状态流转

```
                    ┌─────────┐
                    │ pending │
                    └────┬────┘
                         │ schedule
                         ▼
                    ┌─────────┐
         ┌─────────│scheduled│─────────┐
         │         └────┬────┘         │
         │ pause        │ execute      │ cancel
         ▼              ▼              ▼
    ┌────────┐    ┌──────────┐    ┌─────────┐
    │ paused │    │executing │    │cancelled│
    └───┬────┘    └────┬─────┘    └─────────┘
        │ resume       │
        └──────────────┤
                       │
           ┌───────────┼───────────┐
           │           │           │
           ▼           ▼           ▼
      ┌────────┐  ┌────────┐  ┌────────┐
      │completed│  │ failed │  │timeout │
      └────────┘  └───┬────┘  └───┬────┘
                      │           │
                      │ retry     │ retry
                      ▼           ▼
                 ┌─────────┐
                 │scheduled│
                 └─────────┘
```

## 内置模板

| 模板名称 | 描述 |
|---------|------|
| basic_text_generation | 基础文本生成模型训练 |
| moe_training | MoE (Mixture of Experts) 大模型训练 |
| multimodal_training | 多模态训练（文本+图像）|
| distributed_training | 分布式训练配置 |
| knowledge_distillation | 知识蒸馏训练 |
| model_compression | 模型压缩（量化+剪枝）|
| hyperparameter_search | 超参数搜索优化 |
| lr_finder | 学习率查找器 |
| database_training | 数据库驱动的训练 |
| production_config | 生产环境配置 |

## 错误处理

### HTTP 状态码
| 状态码 | 描述 |
|--------|------|
| 200 | 成功 |
| 201 | 创建成功 |
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 403 | 权限不足 |
| 404 | 资源未找到 |
| 409 | 资源冲突 |
| 500 | 服务器内部错误 |

### 错误响应格式
```json
{
  "success": false,
  "error": "错误描述",
  "details": []
}
```

## 使用示例

### Python 示例

```python
import requests
from datetime import datetime, timedelta

BASE_URL = "http://localhost:5000/api/v1/scheduler"
headers = {"Authorization": "Bearer your-token"}

# 调度训练任务
schedule_time = (datetime.now() + timedelta(hours=1)).isoformat()
response = requests.post(
    f"{BASE_URL}/tasks",
    headers=headers,
    json={
        "task_config": {
            "model": "gpt2",
            "epochs": 3
        },
        "schedule_time": schedule_time,
        "name": "GPT2 Training",
        "priority": "high",
        "task_type": "training",
        "template_id": "builtin_basic_text_generation"
    }
)
task = response.json()["data"]

# 查看任务状态
response = requests.get(
    f"{BASE_URL}/tasks/{task['id']}",
    headers=headers
)

# 取消任务
response = requests.post(
    f"{BASE_URL}/tasks/{task['id']}/cancel",
    headers=headers
)
```

### cURL 示例

```bash
# 调度任务
curl -X POST http://localhost:5000/api/v1/scheduler/tasks \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "task_config": {"model": "gpt2"},
    "schedule_time": "2026-01-12T10:00:00Z",
    "name": "Test Task",
    "priority": "normal"
  }'

# 列出任务
curl -X GET "http://localhost:5000/api/v1/scheduler/tasks?status=scheduled" \
  -H "Authorization: Bearer your-token"

# 获取统计
curl -X GET http://localhost:5000/api/v1/scheduler/statistics \
  -H "Authorization: Bearer your-token"
```

# 训练执行 API 文档

## 概述

训练执行 API 提供完整的训练任务执行生命周期管理，支持多种训练场景的执行控制，包括启动、暂停、恢复、停止训练，以及获取训练状态和进度。

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        API 层                                    │
│              training_execution_api.py                           │
│   (接收 HTTP 请求, 参数验证, 调用 Service 层)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Service 层                                  │
│            training_execution_service.py                         │
│   (业务逻辑: 训练控制, 资源管理, 指标监控, 持久化调用)           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Repository 层                                 │
│          training_execution_repository.py                        │
│   (数据访问: CRUD 操作, 查询统计, 日志记录)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      数据模型层                                  │
│             training_models.py                                   │
│   (TrainingExecution, TrainingExecutionLog)                      │
└─────────────────────────────────────────────────────────────────┘
```

## 数据模型

### TrainingExecution（训练执行记录）

| 字段 | 类型 | 说明 |
|------|------|------|
| execution_id | String(64) | 执行唯一标识 |
| session_id | String(36) | 关联会话ID |
| tenant_id | UUID | 租户ID |
| user_id | String(36) | 用户ID |
| name | String(200) | 执行名称 |
| scenario_type | String(50) | 训练场景类型 |
| training_mode | String(50) | 训练模式 |
| trainer_type | String(100) | 训练器类型 |
| training_config | JSON | 训练配置 |
| resource_config | JSON | 资源配置 |
| status | String(20) | 执行状态 |
| progress | Float | 执行进度(0-100) |
| current_epoch | Integer | 当前轮次 |
| total_epochs | Integer | 总轮次 |
| current_step | Integer | 当前步骤 |
| total_steps | Integer | 总步骤数 |
| metrics | JSON | 当前指标 |
| best_metrics | JSON | 最佳指标 |
| checkpoint_path | String(500) | 最新检查点路径 |
| started_at | DateTime | 开始时间 |
| completed_at | DateTime | 完成时间 |
| error_message | Text | 错误信息 |

### TrainingExecutionLog（执行日志）

| 字段 | 类型 | 说明 |
|------|------|------|
| execution_id | String(64) | 执行ID |
| log_type | String(30) | 日志类型 |
| log_level | String(10) | 日志级别 |
| message | Text | 日志消息 |
| from_status | String(20) | 变更前状态 |
| to_status | String(20) | 变更后状态 |
| epoch | Integer | 轮次 |
| step | Integer | 步骤 |
| progress | Float | 进度 |
| metrics | JSON | 指标快照 |

## API 端点

### 1. 场景化训练

#### 启动场景化训练
```http
POST /api/v1/training/execution/scenario/start
```

请求体:
```json
{
    "session_id": "sess_123",
    "scenario_type": "standard",
    "config": {
        "model_name": "bert-base",
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001
    },
    "resource_config": {
        "gpus": 1,
        "cpu_cores": 4,
        "memory_mb": 8192
    }
}
```

支持的场景类型:
- `standard` - 标准训练
- `distributed` - 分布式训练
- `multimodal` - 多模态训练
- `distillation` - 知识蒸馏
- `three_stage` - 三阶段训练
- `industry` - 行业场景训练
- `scenario` - 自定义场景训练

响应:
```json
{
    "success": true,
    "data": {
        "execution_id": "exec_abc123",
        "session_id": "sess_123",
        "scenario_type": "standard",
        "trainer_type": "StandardTrainer",
        "status": "running",
        "started_at": "2025-01-07T10:00:00Z"
    },
    "message": "standard 场景训练启动成功"
}
```

### 2. 训练会话控制

#### 启动训练
```http
POST /api/v1/training/execution/sessions/<session_id>/start
```

#### 暂停训练
```http
POST /api/v1/training/execution/sessions/<session_id>/pause
```

#### 恢复训练
```http
POST /api/v1/training/execution/sessions/<session_id>/resume
```

#### 停止训练
```http
POST /api/v1/training/execution/sessions/<session_id>/stop
```

请求体:
```json
{
    "save_checkpoint": true,
    "reason": "用户手动停止"
}
```

### 3. 状态和进度查询

#### 获取训练状态
```http
GET /api/v1/training/execution/sessions/<session_id>/status
```

响应:
```json
{
    "success": true,
    "data": {
        "session_id": "sess_123",
        "status": "running",
        "progress": 45.5,
        "current_epoch": 5,
        "total_epochs": 10,
        "current_step": 1500,
        "total_steps": 3000,
        "metrics": {
            "loss": 0.234,
            "accuracy": 0.876
        },
        "started_at": "2025-01-07T10:00:00Z",
        "elapsed_time": 3600
    }
}
```

#### 获取训练进度
```http
GET /api/v1/training/execution/sessions/<session_id>/progress
```

### 4. 执行记录管理

#### 列出执行记录
```http
GET /api/v1/training/execution/executions
```

查询参数:
- `status` - 状态过滤
- `scenario_type` - 场景类型过滤
- `user_id` - 用户ID过滤
- `limit` - 限制数量（默认100）
- `offset` - 偏移量（默认0）

#### 获取执行详情
```http
GET /api/v1/training/execution/executions/<execution_id>
```

#### 更新执行进度
```http
PUT /api/v1/training/execution/executions/<execution_id>/progress
```

请求体:
```json
{
    "progress": 50.0,
    "current_step": 500,
    "current_epoch": 3,
    "metrics": {
        "loss": 0.345,
        "accuracy": 0.812
    }
}
```

#### 更新执行状态
```http
PUT /api/v1/training/execution/executions/<execution_id>/status
```

请求体:
```json
{
    "status": "completed",
    "result": {
        "final_accuracy": 0.92,
        "final_loss": 0.15
    }
}
```

#### 获取执行日志
```http
GET /api/v1/training/execution/executions/<execution_id>/logs
```

查询参数:
- `log_type` - 日志类型过滤
- `limit` - 限制数量
- `offset` - 偏移量

#### 删除执行记录
```http
DELETE /api/v1/training/execution/executions/<execution_id>
```

#### 列出运行中的执行
```http
GET /api/v1/training/execution/executions/running
```

#### 获取执行统计
```http
GET /api/v1/training/execution/executions/statistics
```

响应:
```json
{
    "success": true,
    "data": {
        "total_executions": 100,
        "pending_executions": 5,
        "running_executions": 3,
        "completed_executions": 80,
        "failed_executions": 10,
        "paused_executions": 2,
        "cancelled_executions": 0
    }
}
```

### 5. 指标更新

#### 更新训练指标
```http
POST /api/v1/training/execution/sessions/<session_id>/metrics
```

请求体:
```json
{
    "epoch": 5,
    "step": 1500,
    "loss": 0.234,
    "accuracy": 0.876,
    "learning_rate": 0.0001,
    "throughput": 128.5,
    "memory_usage": 4.2,
    "gpu_utilization": 85.0
}
```

### 6. 历史和统计

#### 获取训练历史
```http
GET /api/v1/training/execution/sessions/history
```

查询参数:
- `limit` - 限制数量（默认50）
- `status` - 状态过滤
- `scenario_type` - 场景类型过滤

#### 获取统计信息
```http
GET /api/v1/training/execution/statistics
```

### 7. 批量操作

#### 批量停止训练
```http
POST /api/v1/training/execution/batch/stop
```

请求体:
```json
{
    "session_ids": ["sess_1", "sess_2", "sess_3"],
    "reason": "批量停止"
}
```

### 8. 健康检查

```http
GET /api/v1/training/execution/health
```

## 使用示例

### Python 客户端示例

```python
import requests

BASE_URL = "http://localhost:5000/api/v1/training/execution"
HEADERS = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json"
}

# 1. 启动场景化训练
response = requests.post(
    f"{BASE_URL}/scenario/start",
    headers=HEADERS,
    json={
        "session_id": "sess_001",
        "scenario_type": "distributed",
        "config": {
            "model_name": "llama-7b",
            "epochs": 5,
            "batch_size": 8,
            "learning_rate": 1e-5
        },
        "resource_config": {
            "gpus": 4,
            "cpu_cores": 16,
            "memory_mb": 65536
        }
    }
)
result = response.json()
execution_id = result['data']['execution_id']

# 2. 查询执行状态
response = requests.get(
    f"{BASE_URL}/executions/{execution_id}",
    headers=HEADERS
)
execution = response.json()['data']
print(f"状态: {execution['status']}, 进度: {execution['progress']}%")

# 3. 更新执行进度
requests.put(
    f"{BASE_URL}/executions/{execution_id}/progress",
    headers=HEADERS,
    json={
        "progress": 30.0,
        "current_epoch": 2,
        "current_step": 1000,
        "metrics": {"loss": 0.5, "accuracy": 0.75}
    }
)

# 4. 获取执行日志
response = requests.get(
    f"{BASE_URL}/executions/{execution_id}/logs",
    headers=HEADERS,
    params={"limit": 50}
)
logs = response.json()['data']['logs']

# 5. 停止训练
requests.post(
    f"{BASE_URL}/sessions/sess_001/stop",
    headers=HEADERS,
    json={"save_checkpoint": True, "reason": "完成测试"}
)

# 6. 获取统计信息
response = requests.get(
    f"{BASE_URL}/executions/statistics",
    headers=HEADERS
)
stats = response.json()['data']
print(f"总执行数: {stats['total_executions']}")
print(f"运行中: {stats['running_executions']}")
```

## 状态流转

```
                    ┌─────────┐
                    │ PENDING │
                    └────┬────┘
                         │ start
                         ▼
        pause      ┌─────────┐      resume
     ┌─────────────│ RUNNING │──────────────┐
     │             └────┬────┘              │
     │                  │                   │
     ▼                  │                   ▼
┌────────┐         stop │ fail        ┌────────┐
│ PAUSED │◄────────────┘└─────────────│ PAUSED │
└────┬───┘                            └────────┘
     │ cancel
     ▼
┌───────────┐       ┌─────────┐      ┌──────────┐
│ CANCELLED │       │ FAILED  │      │COMPLETED │
└───────────┘       └─────────┘      └──────────┘
```

## 注意事项

1. **租户隔离**: 所有操作都基于租户隔离，确保数据安全
2. **幂等性**: 启动训练是幂等的，重复请求会返回已有的训练状态
3. **资源管理**: 系统会自动管理 GPU、CPU、内存等资源的分配和释放
4. **检查点**: 暂停和停止时可选择保存检查点，用于后续恢复
5. **监控**: 系统自动监控资源使用情况，异常时会发出告警

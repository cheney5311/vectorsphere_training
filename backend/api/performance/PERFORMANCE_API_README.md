# Performance API 性能模块接口文档

## 概述

Performance API 提供完整的异步任务处理、性能监控、告警管理等 REST API 接口，支持：

- **异步任务处理**：基于优先级队列的任务调度和执行
- **性能指标收集**：系统资源监控（CPU、内存、磁盘、GPU）
- **告警管理**：告警规则配置、告警触发和处理
- **数据库连接池**：连接池状态监控和优化
- **健康检查**：全面的系统健康状态检查

## 架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                      Performance API Layer                       │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐        │
│  │  Async Task   │  │   Monitoring  │  │    Alert      │        │
│  │   Endpoints   │  │   Endpoints   │  │   Endpoints   │        │
│  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘        │
└──────────┼──────────────────┼──────────────────┼────────────────┘
           │                  │                  │
┌──────────┼──────────────────┼──────────────────┼────────────────┐
│          ▼                  ▼                  ▼                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 PerformanceService                       │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │    │
│  │  │ Task Mgmt    │  │ Metric Coll  │  │ Alert Mgmt   │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                         Service Layer                            │
└─────────────────────────────────────────────────────────────────┘
           │                  │                  │
┌──────────┼──────────────────┼──────────────────┼────────────────┐
│          ▼                  ▼                  ▼                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │AsyncTaskRepo │  │ MetricRepo   │  │  AlertRepo   │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│  ┌──────────────┐  ┌──────────────┐                              │
│  │SnapshotRepo  │  │AlertRuleRepo │                              │
│  └──────────────┘  └──────────────┘                              │
│                      Repository Layer                            │
└─────────────────────────────────────────────────────────────────┘
           │                  │                  │
┌──────────┼──────────────────┼──────────────────┼────────────────┐
│          ▼                  ▼                  ▼                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │             performance_models.py (SQLAlchemy)           │    │
│  │  AsyncTaskModel │ MetricModel │ AlertModel │ RuleModel   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                        Schema Layer                              │
└─────────────────────────────────────────────────────────────────┘
```

## API 端点

### 1. 异步任务管理

#### 1.1 提交异步任务
```http
POST /api/performance/async/tasks
Content-Type: application/json

{
    "task_name": "data_preprocessing",
    "priority": "NORMAL",
    "params": {
        "dataset_id": "ds_001"
    },
    "timeout": 300
}
```

**支持的任务类型**（通过装饰器注册）：

| 任务名称 | 分类 | 描述 |
|---------|------|------|
| `data_preprocessing` | data | 数据预处理 |
| `data_quality_assessment` | data | 数据质量评估 |
| `model_evaluation` | model | 模型评估 |
| `model_compression` | model | 模型压缩 |
| `model_inference_optimization` | model | 推理优化 |
| `training_start` | training | 启动训练 |
| `resource_optimization` | system | 资源优化 |
| `performance_analysis` | system | 性能分析 |
| `database_cleanup` | system | 数据库清理 |
| `cache_warmup` | system | 缓存预热 |

#### 1.2 获取任务状态
```http
GET /api/performance/async/tasks/{task_id}
```

**响应示例**：
```json
{
    "success": true,
    "data": {
        "id": "task-uuid",
        "status": "completed",
        "priority": "NORMAL",
        "result": {...},
        "execution_time": 2.35,
        "created_at_iso": "2025-01-12T10:00:00Z",
        "completed_at_iso": "2025-01-12T10:00:02Z"
    }
}
```

#### 1.3 获取任务类型列表
```http
GET /api/performance/async/tasks/types?category=data
```

#### 1.4 获取处理器统计
```http
GET /api/performance/async/stats?include_health=true&include_workers=true
```

### 2. 持久化任务管理

#### 2.1 创建任务记录
```http
POST /api/performance/tasks
Content-Type: application/json

{
    "name": "my_task",
    "category": "processing",
    "description": "Task description",
    "priority": "high",
    "params": {"key": "value"},
    "timeout": 600
}
```

#### 2.2 获取任务列表
```http
GET /api/performance/tasks?status=pending&category=processing&limit=50
```

#### 2.3 取消任务
```http
POST /api/performance/tasks/{task_id}/cancel
```

#### 2.4 任务统计
```http
GET /api/performance/tasks/statistics
```

#### 2.5 清理旧任务
```http
POST /api/performance/tasks/cleanup
Content-Type: application/json

{
    "max_age_days": 7
}
```

### 3. 系统监控

#### 3.1 获取当前系统快照
```http
GET /api/performance/snapshots/current
```

**响应示例**：
```json
{
    "success": true,
    "data": {
        "timestamp": "2025-01-12T10:00:00Z",
        "cpu": {
            "percent": 45.2,
            "count": 8,
            "load_average": [1.2, 1.5, 1.8]
        },
        "memory": {
            "percent": 62.5,
            "total_gb": 32.0,
            "used_gb": 20.0,
            "available_gb": 12.0
        },
        "disk": {
            "percent": 55.0,
            "total_gb": 500.0,
            "used_gb": 275.0,
            "free_gb": 225.0
        },
        "network": {
            "bytes_sent": 1234567890,
            "bytes_recv": 9876543210
        },
        "process_count": 150
    }
}
```

#### 3.2 获取快照历史
```http
GET /api/performance/snapshots/history?limit=100&start_time=2025-01-12T00:00:00Z
```

#### 3.3 获取当前指标
```http
GET /api/performance/monitoring/metrics?include_gpu=true&include_training=true
```

#### 3.4 获取性能摘要
```http
GET /api/performance/monitoring/summary?time_range=1h&include_recommendations=true
```

### 4. 指标收集

#### 4.1 启动指标收集
```http
POST /api/performance/collection/start
Content-Type: application/json

{
    "interval": 10
}
```

#### 4.2 停止指标收集
```http
POST /api/performance/collection/stop
```

#### 4.3 记录自定义指标
```http
POST /api/performance/metrics/record
Content-Type: application/json

{
    "metric_type": "custom",
    "metric_name": "api_latency",
    "metric_value": 125.5,
    "metric_unit": "ms",
    "resource_id": "api_server_01",
    "tags": {"endpoint": "/api/users"}
}
```

#### 4.4 获取指标历史
```http
GET /api/performance/metrics/history?metric_type=system&metric_name=cpu_percent&limit=100
```

### 5. 告警管理

#### 5.1 获取活跃告警
```http
GET /api/performance/alerts/active?level=critical
```

#### 5.2 确认告警
```http
POST /api/performance/alerts/{alert_id}/acknowledge
Content-Type: application/json

{
    "user_id": "admin_001"
}
```

#### 5.3 解决告警
```http
POST /api/performance/alerts/{alert_id}/resolve
Content-Type: application/json

{
    "user_id": "admin_001",
    "notes": "Issue has been resolved by scaling up resources"
}
```

#### 5.4 告警统计
```http
GET /api/performance/alerts/statistics
```

### 6. 告警规则管理

#### 6.1 创建告警规则
```http
POST /api/performance/rules
Content-Type: application/json

{
    "name": "High CPU Usage",
    "metric_type": "system",
    "metric_name": "cpu_percent",
    "operator": ">",
    "threshold": 85,
    "severity": "high",
    "description": "CPU usage exceeds 85%",
    "duration": 60,
    "notification_channels": ["email", "slack"]
}
```

#### 6.2 获取规则列表
```http
GET /api/performance/rules?enabled=true&metric_type=system
```

#### 6.3 更新规则
```http
PUT /api/performance/rules/{rule_id}
Content-Type: application/json

{
    "threshold": 90,
    "severity": "critical"
}
```

#### 6.4 切换规则状态
```http
POST /api/performance/rules/{rule_id}/toggle
Content-Type: application/json

{
    "enabled": false
}
```

#### 6.5 删除规则
```http
DELETE /api/performance/rules/{rule_id}
```

### 7. 数据库连接池

#### 7.1 获取连接池状态
```http
GET /api/performance/db/status?include_health=true&include_optimization=true
```

#### 7.2 数据库健康检查
```http
GET /api/performance/db/health?detailed=true&include_latency=true
```

### 8. 健康检查

#### 8.1 综合健康检查
```http
GET /api/performance/monitoring/health?include_components=true&include_metrics=true
```

#### 8.2 服务健康检查
```http
GET /api/performance/service/health
```

### 9. 统计信息

#### 9.1 综合统计
```http
GET /api/performance/statistics
```

**响应示例**：
```json
{
    "success": true,
    "data": {
        "tasks": {
            "total": 1500,
            "by_status": {
                "completed": 1200,
                "failed": 50,
                "pending": 100,
                "running": 50
            },
            "avg_execution_time": 2.5
        },
        "alerts": {
            "total": 25,
            "active": 3,
            "by_level": {
                "critical": 1,
                "high": 2,
                "medium": 10,
                "low": 12
            }
        },
        "collecting": true,
        "collection_interval": 10,
        "timestamp": "2025-01-12T10:00:00Z"
    }
}
```

## 数据模型

### 任务状态枚举
```python
class TaskStatusEnum(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
```

### 告警级别枚举
```python
class AlertLevelEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

### 指标类型枚举
```python
class MetricTypeEnum(str, Enum):
    SYSTEM = "system"
    GPU = "gpu"
    TRAINING = "training"
    DATABASE = "database"
    ASYNC_PROCESSOR = "async_processor"
    CUSTOM = "custom"
```

## 错误码

| 错误码 | 描述 |
|-------|------|
| `MISSING_TASK_NAME` | 缺少任务名称 |
| `UNKNOWN_TASK` | 未知任务类型 |
| `INVALID_PARAMS` | 无效的任务参数 |
| `TASK_NOT_FOUND` | 任务未找到 |
| `PROCESSOR_NOT_RUNNING` | 异步处理器未运行 |
| `POOL_NOT_INITIALIZED` | 数据库连接池未初始化 |
| `SERVICE_UNAVAILABLE` | 服务不可用 |
| `INTERNAL_ERROR` | 内部错误 |

## 使用示例

### Python 客户端示例

```python
import requests

BASE_URL = "http://localhost:5000/api/performance"

# 1. 提交异步任务
response = requests.post(f"{BASE_URL}/async/tasks", json={
    "task_name": "data_preprocessing",
    "priority": "HIGH",
    "params": {"dataset_id": "ds_001"}
})
task_id = response.json()['task_id']

# 2. 查询任务状态
response = requests.get(f"{BASE_URL}/async/tasks/{task_id}")
status = response.json()['data']['status']

# 3. 获取系统快照
response = requests.get(f"{BASE_URL}/snapshots/current")
snapshot = response.json()['data']

# 4. 创建告警规则
response = requests.post(f"{BASE_URL}/rules", json={
    "name": "High Memory",
    "metric_type": "system",
    "metric_name": "memory_percent",
    "operator": ">",
    "threshold": 80,
    "severity": "high"
})
```

## 配置

配置文件位置：`config/performance_api.yaml`

```yaml
resource_thresholds:
  cpu:
    warning: 80
    critical: 90
  memory:
    warning: 75
    critical: 85

async_processor:
  queue:
    max_size: 1000
  workers:
    max_count: 10

alerts:
  default_duration_minutes: 60

initialization:
  auto_start_monitoring: true
  setup_default_alerts: true
```

## 注意事项

1. **任务持久化**：持久化任务使用仓库层存储，支持内存和数据库两种模式
2. **异步任务**：异步任务使用 `AsyncProcessor` 执行，结果存储在内存中
3. **指标收集**：需要显式调用 `start_collection` 启动收集
4. **告警规则**：规则创建后需要启用才会生效
5. **多租户**：所有端点支持 `tenant_id` 参数进行数据隔离

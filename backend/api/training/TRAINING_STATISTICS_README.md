# 训练统计API文档

## 概述

训练统计API提供全面的训练任务统计分析功能，支持多维度的统计查询、趋势分析、资源监控等。

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                   Training Statistics API                    │
│            /api/v1/training/statistics/*                     │
├─────────────────────────────────────────────────────────────┤
│                TrainingStatisticsService                     │
│        业务逻辑层：统计聚合、计算、分析                         │
├─────────────────────────────────────────────────────────────┤
│              TrainingStatisticsRepository                    │
│        数据访问层：数据库查询、聚合、缓存                        │
├─────────────────────────────────────────────────────────────┤
│                     Database Layer                           │
│   TrainingJob | TrainingSession | TrainingProgress           │
└─────────────────────────────────────────────────────────────┘
```

## API 端点

### 1. 基础统计

#### GET /api/v1/training/statistics/basic

获取基础训练统计信息。

**请求参数：**
| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| user_id | string | 否 | 过滤特定用户的统计 |

**响应示例：**
```json
{
    "success": true,
    "data": {
        "total_jobs": 100,
        "completed_jobs": 75,
        "failed_jobs": 10,
        "running_jobs": 8,
        "pending_jobs": 5,
        "paused_jobs": 2,
        "cancelled_jobs": 0,
        "success_rate": 0.75,
        "average_duration": 3600,
        "total_training_hours": 250.5,
        "timestamp": "2024-01-15T10:30:00Z"
    },
    "message": "获取基础统计信息成功"
}
```

### 2. 详细统计

#### GET /api/v1/training/statistics/detailed

获取详细训练统计信息，包括资源使用、模型分布等。

**请求参数：**
| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| days | integer | 否 | 统计天数，默认30天 |
| user_id | string | 否 | 过滤特定用户 |

**响应示例：**
```json
{
    "success": true,
    "data": {
        "total_jobs": 100,
        "running_jobs": 8,
        "completed_jobs": 75,
        "failed_jobs": 10,
        "cancelled_jobs": 5,
        "paused_jobs": 2,
        "pending_jobs": 0,
        "success_rate": 0.75,
        "average_training_time": 3600.5,
        "total_training_hours": 250.5,
        "most_used_model": "gpt2-medium",
        "resource_usage": {
            "cpu_avg": 65.5,
            "memory_avg": 32.8,
            "gpu_avg": 85.2,
            "gpu_memory_avg": 12.5
        },
        "daily_stats": [
            {"date": "2024-01-14", "jobs_count": 12, "success_count": 10},
            {"date": "2024-01-15", "jobs_count": 15, "success_count": 13}
        ],
        "top_models": [...],
        "scenario_breakdown": [...],
        "performance_metrics": {...}
    },
    "message": "获取详细统计信息成功"
}
```

### 3. 统计概览

#### GET /api/v1/training/statistics/overview

获取统计信息综合概览。

**响应示例：**
```json
{
    "success": true,
    "data": {
        "overall_stats": {...},
        "recent_trends": [...],
        "performance_metrics": {...},
        "top_models": [...],
        "scenario_breakdown": [...]
    },
    "message": "获取统计概览成功"
}
```

### 4. 趋势统计

#### GET /api/v1/training/statistics/trends

获取指定时间范围内的统计趋势数据。

**请求参数：**
| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| days | integer | 否 | 统计天数，默认7天 |
| group_by | string | 否 | 分组方式(hour/day/week/month)，默认day |

**响应示例：**
```json
{
    "success": true,
    "data": {
        "period": "Last 7 days",
        "group_by": "day",
        "total_jobs": 85,
        "completed_jobs": 70,
        "failed_jobs": 8,
        "success_rate": 82.35,
        "trend_data": [
            {"date": "2024-01-09", "jobs": 10, "completed": 8, "failed": 1, "success_rate": 80.0},
            {"date": "2024-01-10", "jobs": 12, "completed": 11, "failed": 0, "success_rate": 91.67}
        ]
    },
    "message": "获取趋势统计成功"
}
```

### 5. 每日统计

#### GET /api/v1/training/statistics/daily

获取每日统计数据列表。

**请求参数：**
| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| days | integer | 否 | 统计天数，默认30天 |

### 6. 资源统计

#### GET /api/v1/training/statistics/resources

获取GPU、CPU等资源使用统计。

**请求参数：**
| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| days | integer | 否 | 统计天数，默认7天 |

**响应示例：**
```json
{
    "success": true,
    "data": {
        "period": "Last 7 days",
        "gpu": {
            "average_utilization": 78.5,
            "max_utilization": 98.2,
            "average_memory_used_gb": 14.3
        },
        "cpu": {
            "average_utilization": 45.2,
            "max_utilization": 85.6,
            "average_memory_used_gb": 28.5
        }
    },
    "message": "获取资源统计成功"
}
```

### 7. 模型统计

#### GET /api/v1/training/statistics/models

获取各模型的使用频率和成功率统计。

**请求参数：**
| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| limit | integer | 否 | 返回数量限制，默认10 |

**响应示例：**
```json
{
    "success": true,
    "data": {
        "top_models": [
            {"model_name": "gpt2-medium", "usage_count": 45, "success_count": 40, "success_rate": 88.89},
            {"model_name": "bert-base", "usage_count": 30, "success_count": 28, "success_rate": 93.33}
        ],
        "total_models": 8
    },
    "message": "获取模型统计成功"
}
```

### 8. 场景统计

#### GET /api/v1/training/statistics/scenarios

获取各训练场景的使用情况统计。

**响应示例：**
```json
{
    "success": true,
    "data": {
        "scenarios": [
            {"scenario_type": "standard", "total_count": 50, "completed_count": 45, "failed_count": 3, "success_rate": 90.0},
            {"scenario_type": "distributed", "total_count": 30, "completed_count": 25, "failed_count": 3, "success_rate": 83.33},
            {"scenario_type": "multimodal", "total_count": 20, "completed_count": 18, "failed_count": 1, "success_rate": 90.0}
        ],
        "total_scenarios": 6
    },
    "message": "获取场景统计成功"
}
```

### 9. 性能指标统计

#### GET /api/v1/training/statistics/performance

获取训练性能相关指标的统计信息。

**响应示例：**
```json
{
    "success": true,
    "data": {
        "loss": {
            "average": 0.245,
            "minimum": 0.012
        },
        "accuracy": {
            "average": 0.876,
            "maximum": 0.965
        },
        "throughput": {
            "average_samples_per_second": 256.5,
            "max_samples_per_second": 512.8
        }
    },
    "message": "获取性能指标统计成功"
}
```

### 10. 任务统计

#### GET /api/v1/training/statistics/jobs/{job_id}

获取特定任务的统计信息。

**路径参数：**
| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| job_id | string | 是 | 任务ID |

**响应示例：**
```json
{
    "success": true,
    "data": {
        "job_id": "job_123456",
        "status": "completed",
        "created_at": "2024-01-15T08:00:00Z",
        "result": {...},
        "error": null,
        "duration": "2 hours 15 minutes"
    },
    "message": "获取任务统计成功"
}
```

### 11. 整体统计

#### GET /api/v1/training/statistics/overall

获取全面的训练系统统计概览。

**响应示例：**
```json
{
    "success": true,
    "data": {
        "period": {...},
        "summary": {...},
        "time_statistics": {...},
        "resource_usage": {...},
        "top_models": [...],
        "scenario_breakdown": [...],
        "performance_metrics": {...},
        "uptime": "5 days, 12 hours"
    },
    "message": "获取整体统计成功"
}
```

### 12. 实时统计

#### GET /api/v1/training/statistics/realtime

获取当日实时统计数据。

**响应示例：**
```json
{
    "success": true,
    "data": {
        "today": {
            "total_jobs": 15,
            "running_jobs": 3,
            "completed_jobs": 10,
            "failed_jobs": 2
        },
        "current_time": "2024-01-15T10:30:00Z",
        "uptime": "5 days, 12 hours"
    },
    "message": "获取实时统计成功"
}
```

### 13. 导出统计

#### GET /api/v1/training/statistics/export

导出统计数据为JSON或CSV格式。

**请求参数：**
| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| format | string | 否 | 导出格式(json/csv)，默认json |
| days | integer | 否 | 统计天数，默认30天 |

**响应：**
- JSON格式：返回完整统计数据的JSON响应
- CSV格式：触发文件下载

### 14. 时间段比较

#### POST /api/v1/training/statistics/compare

比较两个时间段的统计数据。

**请求体：**
```json
{
    "period1": {
        "start_date": "2024-01-01",
        "end_date": "2024-01-07"
    },
    "period2": {
        "start_date": "2024-01-08",
        "end_date": "2024-01-14"
    }
}
```

**响应示例：**
```json
{
    "success": true,
    "data": {
        "period1": {
            "date_range": "2024-01-01 - 2024-01-07",
            "statistics": {...}
        },
        "period2": {
            "date_range": "2024-01-08 - 2024-01-14",
            "statistics": {...}
        },
        "changes": {
            "total_jobs_change": 15,
            "completed_jobs_change": 12,
            "failed_jobs_change": -2
        }
    },
    "message": "统计对比成功"
}
```

### 15. 聚合查询

#### POST /api/v1/training/statistics/aggregate

支持自定义维度和指标的聚合查询。

**请求体：**
```json
{
    "dimensions": ["scenario_type", "model_name"],
    "metrics": ["count", "success_rate", "avg_duration"],
    "filters": {
        "status": ["completed", "failed"],
        "start_date": "2024-01-01",
        "end_date": "2024-01-15"
    },
    "limit": 100
}
```

**支持的维度：**
- `scenario_type` - 训练场景类型
- `model_name` - 模型名称
- `status` - 任务状态
- `training_mode` - 训练模式

**支持的指标：**
- `count` - 数量统计
- `success_rate` - 成功率
- `avg_duration` - 平均时长
- `total_duration` - 总时长

### 16. 健康检查

#### GET /api/v1/training/statistics/health

API健康检查端点，无需认证。

**响应示例：**
```json
{
    "status": "healthy",
    "service": "training_statistics",
    "timestamp": "2024-01-15T10:30:00Z",
    "repository_available": true,
    "scenario_manager_available": true
}
```

## 使用示例

### Python 客户端示例

```python
import requests

# 基础配置
BASE_URL = "http://localhost:5000/api/v1/training/statistics"
headers = {
    "Authorization": "Bearer <your_jwt_token>",
    "X-Tenant-ID": "tenant_001"
}

# 1. 获取基础统计
response = requests.get(f"{BASE_URL}/basic", headers=headers)
basic_stats = response.json()
print(f"总任务数: {basic_stats['data']['total_jobs']}")
print(f"成功率: {basic_stats['data']['success_rate'] * 100}%")

# 2. 获取趋势数据
response = requests.get(
    f"{BASE_URL}/trends",
    headers=headers,
    params={"days": 7, "group_by": "day"}
)
trend_data = response.json()
for item in trend_data['data']['trend_data']:
    print(f"{item['date']}: {item['jobs']} 任务, {item['success_rate']}% 成功率")

# 3. 获取资源使用统计
response = requests.get(
    f"{BASE_URL}/resources",
    headers=headers,
    params={"days": 7}
)
resource_stats = response.json()
print(f"GPU平均使用率: {resource_stats['data']['gpu']['average_utilization']}%")

# 4. 导出CSV报告
response = requests.get(
    f"{BASE_URL}/export",
    headers=headers,
    params={"format": "csv", "days": 30}
)
with open("training_stats.csv", "w") as f:
    f.write(response.text)

# 5. 比较两个时间段
comparison_data = {
    "period1": {"start_date": "2024-01-01", "end_date": "2024-01-07"},
    "period2": {"start_date": "2024-01-08", "end_date": "2024-01-14"}
}
response = requests.post(
    f"{BASE_URL}/compare",
    headers=headers,
    json=comparison_data
)
comparison = response.json()
print(f"任务数变化: {comparison['data']['changes']['total_jobs_change']}")
```

### JavaScript/前端示例

```javascript
// 使用 fetch API

const API_BASE = '/api/v1/training/statistics';

// 获取认证头
const getHeaders = () => ({
    'Authorization': `Bearer ${localStorage.getItem('jwt_token')}`,
    'X-Tenant-ID': 'tenant_001',
    'Content-Type': 'application/json'
});

// 1. 获取基础统计
async function getBasicStats() {
    const response = await fetch(`${API_BASE}/basic`, {
        headers: getHeaders()
    });
    const data = await response.json();
    
    if (data.success) {
        console.log('总任务数:', data.data.total_jobs);
        console.log('成功率:', (data.data.success_rate * 100).toFixed(2) + '%');
    }
    return data;
}

// 2. 获取趋势图表数据
async function getTrendData(days = 7) {
    const response = await fetch(`${API_BASE}/trends?days=${days}`, {
        headers: getHeaders()
    });
    const data = await response.json();
    
    if (data.success) {
        // 转换为图表数据格式
        const chartData = data.data.trend_data.map(item => ({
            x: item.date,
            jobs: item.jobs,
            completed: item.completed,
            failed: item.failed
        }));
        return chartData;
    }
    return [];
}

// 3. 实时更新统计
async function startRealtimeUpdates(callback, interval = 30000) {
    const update = async () => {
        const response = await fetch(`${API_BASE}/realtime`, {
            headers: getHeaders()
        });
        const data = await response.json();
        
        if (data.success) {
            callback(data.data);
        }
    };
    
    // 立即执行一次
    await update();
    
    // 定时更新
    return setInterval(update, interval);
}

// 使用示例
startRealtimeUpdates((stats) => {
    document.getElementById('running-count').textContent = stats.today.running_jobs;
    document.getElementById('completed-count').textContent = stats.today.completed_jobs;
});
```

## 数据模型

### TrainingJob 训练任务

| 字段 | 类型 | 描述 |
|------|------|------|
| job_id | string | 任务唯一标识 |
| tenant_id | string | 租户ID |
| user_id | string | 用户ID |
| name | string | 任务名称 |
| scenario_type | string | 训练场景类型 |
| model_name | string | 模型名称 |
| status | string | 任务状态 |
| progress | float | 进度(0-100) |
| started_at | datetime | 开始时间 |
| completed_at | datetime | 完成时间 |
| metrics | JSON | 训练指标 |

### TrainingProgress 训练进度

| 字段 | 类型 | 描述 |
|------|------|------|
| session_id | string | 会话ID |
| epoch | integer | 当前轮次 |
| step | integer | 当前步骤 |
| loss | float | 损失值 |
| accuracy | float | 准确率 |
| gpu_utilization | float | GPU使用率 |
| cpu_utilization | float | CPU使用率 |

## 错误处理

### 错误响应格式

```json
{
    "success": false,
    "error": "错误描述信息",
    "code": 400
}
```

### 常见错误码

| 错误码 | 描述 |
|--------|------|
| 400 | 请求参数错误 |
| 401 | 未认证 |
| 403 | 无权限 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |
| 503 | 服务暂不可用 |

## 性能优化建议

1. **使用合适的时间范围**：避免查询过长时间范围的数据，建议不超过90天
2. **缓存策略**：对于变化不频繁的统计数据，可在客户端进行缓存
3. **分页查询**：使用 `limit` 参数限制返回数量
4. **按需请求**：根据实际需要选择合适的API端点，避免请求过多数据

## 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0.0 | 2024-01-15 | 初始版本，支持基础统计、详细统计、趋势分析 |
| 1.1.0 | 2024-01-20 | 新增资源统计、模型统计、场景统计 |
| 1.2.0 | 2024-01-25 | 新增导出功能、时间段比较、聚合查询 |


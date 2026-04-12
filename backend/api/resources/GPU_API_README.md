# GPU 资源管理 API

## 概述

GPU 资源管理 API 提供生产级的 GPU 资源监控、分配、释放和节点管理功能。支持多租户隔离、灵活的分配策略和完整的使用历史追踪。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│                   gpu_api.py (Flask Blueprint)               │
├─────────────────────────────────────────────────────────────┤
│                      Service Layer                           │
│                 gpu_resource_service.py                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   资源分配   │  │   节点管理   │  │   监控采集   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│                    Repository Layer                          │
│               gpu_resource_repository.py                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ NodeRepo │ │DeviceRepo│ │AllocRepo │ │UsageRepo │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
├─────────────────────────────────────────────────────────────┤
│                     Schema Layer                             │
│               gpu_resource_models.py                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │GPUNode   │ │GPUDevice │ │Allocation│ │UsageHist │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## API 端点

### 指标 API

#### 获取 GPU 指标
```http
GET /api/v1/gpus/metrics
```

**响应示例：**
```json
{
  "success": true,
  "data": {
    "last": {
      "gpu_count": 4,
      "gpus": [
        {
          "index": 0,
          "utilization_percent": 45,
          "memory_used_mb": 8192,
          "memory_total_mb": 16384
        }
      ]
    },
    "updated_at": 1704931200.0
  }
}
```

#### 列出 GPU 设备
```http
GET /api/v1/gpus/list
```

#### 获取 GPU 详细信息
```http
GET /api/v1/gpus/details
```

#### 获取 GPU 汇总
```http
GET /api/v1/gpus/summary
```

### 资源分配 API

#### 分配资源
```http
POST /api/v1/gpus/allocate
Content-Type: application/json

{
  "gpu_count": 2,
  "gpu_memory_mb": 8192,
  "cpu_cores": 4,
  "memory_mb": 16384,
  "priority": 5,
  "labels_affinity": {
    "zone": "gpu",
    "type": "A100"
  },
  "prefer_same_node": true,
  "task_id": "train_xxx",
  "lease_duration_seconds": 3600,
  "strategy": "best_fit"
}
```

**响应示例：**
```json
{
  "success": true,
  "data": {
    "allocated": true,
    "allocation_id": "alloc_abc123",
    "node_id": "node_xyz789",
    "gpu_indices": [0, 1]
  }
}
```

#### 释放分配
```http
POST /api/v1/gpus/release/{allocation_id}
```
或
```http
DELETE /api/v1/gpus/release/{allocation_id}
```

#### 列出分配记录
```http
GET /api/v1/gpus/allocations?status=active&node_id=xxx&limit=50
```

#### 获取分配详情
```http
GET /api/v1/gpus/allocations/{allocation_id}
```

### 节点管理 API

#### 列出节点
```http
GET /api/v1/gpus/nodes?status=online&is_healthy=true
```

#### 注册节点
```http
POST /api/v1/gpus/nodes
Content-Type: application/json

{
  "hostname": "gpu-node-01",
  "ip_address": "192.168.1.100",
  "port": 8080,
  "labels": {
    "zone": "gpu",
    "type": "A100"
  },
  "capabilities": ["cuda", "tensorrt"]
}
```

#### 获取节点详情
```http
GET /api/v1/gpus/nodes/{node_id}
```

#### 注销节点
```http
DELETE /api/v1/gpus/nodes/{node_id}
```

#### 节点心跳
```http
POST /api/v1/gpus/nodes/{node_id}/heartbeat
Content-Type: application/json

{
  "cpu_used": 50.0,
  "memory_used_mb": 8192,
  "used_gpu_memory_mb": 16384
}
```

### 统计 API

#### 获取统计信息
```http
GET /api/v1/gpus/statistics
```

**响应示例：**
```json
{
  "success": true,
  "data": {
    "nodes": {
      "total_nodes": 4,
      "healthy_nodes": 4,
      "total_gpus": 16,
      "total_gpu_memory_mb": 262144,
      "used_gpu_memory_mb": 65536
    },
    "allocations": {
      "total_allocations": 25,
      "active_allocations": 8,
      "total_gpus_allocated": 12
    },
    "gpus": {
      "total": 16,
      "available": 4,
      "allocated": 12,
      "total_memory_mb": 262144,
      "used_memory_mb": 196608
    },
    "timestamp": "2026-01-11T06:30:00Z"
  }
}
```

#### 获取使用历史
```http
GET /api/v1/gpus/usage/history?node_id=xxx&period_type=hour&limit=24
```

### 健康检查

```http
GET /api/v1/gpus/health
```

**响应示例：**
```json
{
  "success": true,
  "status": "healthy",
  "healthy": true,
  "monitoring": true,
  "gpu_available": true,
  "gpu_count": 4,
  "timestamp": "2026-01-11T06:30:00Z"
}
```

### 监控控制

#### 启动监控
```http
POST /api/v1/gpus/monitoring/start
Content-Type: application/json

{
  "interval": 10
}
```

#### 停止监控
```http
POST /api/v1/gpus/monitoring/stop
```

## 数据模型

### 节点状态 (NodeStatusEnum)
| 状态 | 说明 |
|------|------|
| online | 在线 |
| offline | 离线 |
| maintenance | 维护中 |
| unhealthy | 不健康 |
| draining | 排空中 |

### GPU 状态 (GPUStatusEnum)
| 状态 | 说明 |
|------|------|
| available | 可用 |
| allocated | 已分配 |
| reserved | 已预留 |
| faulty | 故障 |
| maintenance | 维护中 |

### 分配状态 (AllocationStatusEnum)
| 状态 | 说明 |
|------|------|
| pending | 等待中 |
| active | 活跃 |
| releasing | 释放中 |
| released | 已释放 |
| expired | 已过期 |
| failed | 失败 |

### 分配策略 (AllocationStrategyEnum)
| 策略 | 说明 |
|------|------|
| best_fit | 最佳适应 - 选择刚好满足需求的资源 |
| worst_fit | 最差适应 - 选择剩余资源最多的节点 |
| first_fit | 首次适应 - 选择第一个满足条件的节点 |
| round_robin | 轮询 - 均匀分配到各节点 |
| priority_based | 优先级 - 基于任务优先级分配 |
| affinity_based | 亲和性 - 基于标签亲和性分配 |

## 配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| GPU_MONITOR_INTERVAL_SECONDS | 10 | 监控采集间隔（秒） |
| GPU_ALLOC_WAIT_SECONDS | 5 | 分配等待超时（秒） |
| GPU_ALLOC_RETRY_INTERVAL | 0.5 | 分配重试间隔（秒） |

## 认证

所有 API（除健康检查外）需要 JWT 认证：

```http
Authorization: Bearer <token>
```

## 多租户支持

通过请求头传递租户 ID：

```http
X-Tenant-ID: tenant_xxx
```

## 错误响应

```json
{
  "success": false,
  "error": "错误描述"
}
```

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 201 | 创建成功 |
| 400 | 请求参数错误 |
| 404 | 资源不存在 |
| 409 | 资源冲突（如无可用资源） |
| 500 | 服务器内部错误 |

## 使用示例

### Python 客户端

```python
import requests

BASE_URL = "http://localhost:5000/api/v1/gpus"
HEADERS = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json",
    "X-Tenant-ID": "tenant_001"
}

# 获取 GPU 指标
response = requests.get(f"{BASE_URL}/metrics", headers=HEADERS)
print(response.json())

# 分配资源
allocation_request = {
    "gpu_count": 2,
    "gpu_memory_mb": 8192,
    "priority": 5,
    "task_id": "train_001"
}
response = requests.post(
    f"{BASE_URL}/allocate", 
    json=allocation_request, 
    headers=HEADERS
)
result = response.json()

if result["success"]:
    allocation_id = result["data"]["allocation_id"]
    print(f"分配成功: {allocation_id}")
    
    # 使用完成后释放
    response = requests.post(
        f"{BASE_URL}/release/{allocation_id}",
        headers=HEADERS
    )
    print(response.json())
```

### cURL 示例

```bash
# 获取指标
curl -H "Authorization: Bearer <token>" \
     http://localhost:5000/api/v1/gpus/metrics

# 分配资源
curl -X POST \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{"gpu_count": 1, "gpu_memory_mb": 4096}' \
     http://localhost:5000/api/v1/gpus/allocate

# 释放资源
curl -X POST \
     -H "Authorization: Bearer <token>" \
     http://localhost:5000/api/v1/gpus/release/alloc_xxx
```

## 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0.0 | 2026-01-11 | 初始版本，完整的 GPU 资源管理功能 |

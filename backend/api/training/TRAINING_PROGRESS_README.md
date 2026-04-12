# 训练进度管理 API 文档

## 概述

训练进度管理模块提供完整的训练进度监控、日志、指标、事件管理以及**实时推送**能力。

### 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        客户端                                    │
│  (Web/Mobile/CLI)                                               │
└─────────────────────────────────────────────────────────────────┘
                │               │               │
                │ REST API      │ WebSocket     │ SSE
                ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API 层                                      │
│  ┌──────────────────┐  ┌─────────────────────────────────────┐ │
│  │training_progress_│  │training_progress_websocket_api.py   │ │
│  │api.py (RESTful)  │  │(实时推送: WebSocket/SSE/长轮询)     │ │
│  └────────┬─────────┘  └──────────────────┬──────────────────┘ │
└───────────┼───────────────────────────────┼────────────────────┘
            │                               │
            ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Service 层                                  │
│           training_progress_service.py                           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ - 进度查询/更新          - 订阅管理                         ││
│  │ - 日志/指标/事件         - 实时推送                         ││
│  │ - 检查点/资源监控        - 进度事件广播                     ││
│  └─────────────────────────────────────────────────────────────┘│
└────────────────────────────────┬────────────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│   Repository 层   │ │  Progress Manager │ │  Push Manager     │
│training_progress_ │ │  (实时内存状态)   │ │  (订阅/广播)      │
│repository.py      │ │                   │ │                   │
└─────────┬─────────┘ └───────────────────┘ └───────────────────┘
          │
          ▼
┌───────────────────┐
│     Database      │
│  TrainingSession  │
│  TrainingProgress │
└───────────────────┘
```

## RESTful API 端点

### 基础路径
`/api/v1/training/progress`

### 端点列表

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/<session_id>` | 获取训练进度 |
| POST | `/<session_id>` | 更新训练进度 |
| GET | `/<session_id>/history` | 获取进度历史 |
| GET | `/<session_id>/logs` | 获取训练日志 |
| GET | `/<session_id>/metrics` | 获取训练指标 |
| GET | `/<session_id>/metrics/summary` | 获取指标摘要 |
| GET | `/<session_id>/events` | 获取训练事件 |
| GET | `/<session_id>/checkpoints` | 获取检查点 |
| GET | `/<session_id>/resources` | 获取资源使用 |
| GET | `/<session_id>/summary` | 获取进度摘要 |
| GET | `/<session_id>/realtime` | 获取实时进度 |
| GET | `/health` | 健康检查 |

### 示例

#### 获取训练进度

```bash
curl -X GET "http://localhost:5000/api/v1/training/progress/{session_id}" \
  -H "Authorization: Bearer {token}" \
  -H "X-Tenant-ID: tenant_001"
```

响应:
```json
{
  "success": true,
  "data": {
    "session_id": "uuid",
    "current_epoch": 5,
    "total_epochs": 10,
    "current_step": 500,
    "total_steps": 1000,
    "current_stage": "finetune",
    "loss": 0.15,
    "accuracy": 0.92,
    "learning_rate": 1e-4,
    "status": "running",
    "progress_percentage": 50.0,
    "updated_at": "2025-01-09T10:00:00Z"
  },
  "message": "获取训练进度成功"
}
```

#### 更新训练进度

```bash
curl -X POST "http://localhost:5000/api/v1/training/progress/{session_id}" \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "stage": "finetune",
    "epoch": 5,
    "step": 500,
    "total_steps": 1000,
    "loss": 0.15,
    "accuracy": 0.92,
    "learning_rate": 1e-4,
    "gpu_utilization": 0.85,
    "gpu_memory_used": 12.5,
    "gpu_memory_total": 16.0
  }'
```

## 实时推送 API

### 基础路径
`/api/v1/training/progress/realtime`

### 端点列表

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/subscribe/<session_id>` | 订阅进度更新 |
| POST | `/unsubscribe/<session_id>` | 取消订阅 |
| GET | `/subscriptions` | 获取订阅列表 |
| POST | `/push/<session_id>` | 推送进度更新 |
| POST | `/push/batch` | 批量推送 |
| GET | `/stream/<session_id>` | SSE 流式推送 |
| GET | `/poll/<session_id>` | 长轮询 |
| GET | `/stats` | 推送统计 |
| GET | `/subscribers/<session_id>` | 获取订阅者 |
| GET | `/health` | 健康检查 |

### 三种实时推送方式

#### 1. WebSocket（推荐）

最佳实时体验，全双工通信。

**连接方式:**
```javascript
// 使用 Socket.IO 客户端
const socket = io('http://localhost:5000');

// 认证
socket.emit('authenticate', { token: 'your-jwt-token' });

// 订阅进度
socket.emit('subscribe_progress', { 
    session_id: 'training-session-uuid',
    user_id: 'your-user-id'
});

// 监听进度更新
socket.on('training_progress', (data) => {
    console.log('Progress update:', data);
    // {
    //     type: 'progress_update',
    //     session_id: 'uuid',
    //     timestamp: '2025-01-09T10:00:00Z',
    //     data: {
    //         epoch: 5,
    //         step: 500,
    //         loss: 0.15,
    //         accuracy: 0.92
    //     }
    // }
});

// 取消订阅
socket.emit('unsubscribe_progress', { session_id: 'training-session-uuid' });
```

#### 2. SSE (Server-Sent Events)

浏览器兼容性好，单向推送。

**连接方式:**
```javascript
// 首先订阅
await fetch('/api/v1/training/progress/realtime/subscribe/session-uuid', {
    method: 'POST',
    headers: { 'Authorization': 'Bearer token' }
});

// 创建 EventSource
const eventSource = new EventSource(
    '/api/v1/training/progress/realtime/stream/session-uuid',
    { withCredentials: true }
);

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Progress update:', data);
};

eventSource.onerror = function(error) {
    console.error('SSE error:', error);
    eventSource.close();
};

// 关闭连接
eventSource.close();
```

#### 3. 长轮询

无需额外依赖，兼容性最好。

**使用方式:**
```javascript
async function pollProgress(sessionId, lastTimestamp = null) {
    const params = new URLSearchParams({
        timeout: '30',
        limit: '50'
    });
    if (lastTimestamp) {
        params.append('since', lastTimestamp);
    }
    
    const response = await fetch(
        `/api/v1/training/progress/realtime/poll/${sessionId}?${params}`,
        { headers: { 'Authorization': 'Bearer token' } }
    );
    
    const result = await response.json();
    
    if (result.success && result.data.updates.length > 0) {
        // 处理更新
        result.data.updates.forEach(update => {
            console.log('Progress update:', update);
        });
        
        // 记录最后时间戳用于下次轮询
        const lastUpdate = result.data.updates[result.data.updates.length - 1];
        lastTimestamp = new Date(lastUpdate.timestamp).getTime();
    }
    
    // 继续轮询
    setTimeout(() => pollProgress(sessionId, lastTimestamp), 1000);
}

// 开始轮询
pollProgress('session-uuid');
```

### 订阅管理

#### 订阅进度

```bash
curl -X POST "http://localhost:5000/api/v1/training/progress/realtime/subscribe/{session_id}" \
  -H "Authorization: Bearer {token}"
```

响应:
```json
{
  "success": true,
  "data": {
    "session_id": "uuid",
    "subscribed": true,
    "channels": {
      "websocket": "training_progress_uuid",
      "sse": "/api/v1/training/progress/realtime/stream/uuid",
      "poll": "/api/v1/training/progress/realtime/poll/uuid"
    }
  },
  "message": "订阅成功"
}
```

#### 取消订阅

```bash
curl -X POST "http://localhost:5000/api/v1/training/progress/realtime/unsubscribe/{session_id}" \
  -H "Authorization: Bearer {token}"
```

### 进度推送（内部使用）

训练模块在训练过程中调用此接口推送进度:

```bash
curl -X POST "http://localhost:5000/api/v1/training/progress/realtime/push/{session_id}" \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "epoch": 5,
    "step": 500,
    "total_steps": 1000,
    "loss": 0.15,
    "accuracy": 0.92,
    "learning_rate": 1e-4
  }'
```

## 服务层集成

### 在训练代码中推送进度

```python
from backend.services.training_progress_service import get_training_progress_service

service = get_training_progress_service()

# 方式1：更新进度（同时持久化和推送）
result = service.update_progress(
    session_id='training-session-uuid',
    user_id='user-001',
    tenant_id='tenant-001',
    progress_data={
        'epoch': 5,
        'step': 500,
        'loss': 0.15,
        'accuracy': 0.92
    },
    push_realtime=True  # 启用实时推送
)

# 方式2：仅推送事件（不持久化）
pushed_count = service.push_progress_event(
    session_id='training-session-uuid',
    event_type='checkpoint_saved',
    event_data={
        'epoch': 5,
        'path': '/checkpoints/epoch_5.pt'
    }
)

# 订阅/取消订阅
service.subscribe_progress('session-uuid', 'user-001')
service.unsubscribe_progress('session-uuid', 'user-001')
```

### 在训练回调中使用

```python
from backend.api.training.training_progress_websocket_api import (
    push_training_progress,
    get_push_manager
)

class TrainingCallback:
    def __init__(self, session_id: str):
        self.session_id = session_id
    
    def on_epoch_end(self, epoch: int, logs: dict):
        # 推送进度
        push_training_progress(self.session_id, {
            'type': 'epoch_end',
            'epoch': epoch,
            'loss': logs.get('loss'),
            'accuracy': logs.get('accuracy')
        })
    
    def on_batch_end(self, batch: int, logs: dict):
        # 高频更新可以降低推送频率
        if batch % 100 == 0:
            push_training_progress(self.session_id, {
                'type': 'batch_progress',
                'batch': batch,
                'loss': logs.get('loss')
            })
```

## 数据模型

### TrainingProgress 表结构

| 字段 | 类型 | 描述 |
|------|------|------|
| id | String(36) | 进度记录ID |
| session_id | String(36) | 训练会话ID |
| stage | String(50) | 训练阶段 |
| epoch | Integer | 当前轮次 |
| step | Integer | 当前步数 |
| total_steps | Integer | 总步数 |
| loss | Float | 损失值 |
| accuracy | Float | 准确率 |
| learning_rate | Float | 学习率 |
| metrics | JSON | 额外指标 |
| gpu_utilization | Float | GPU利用率 |
| gpu_memory_used | Float | GPU已用内存 |
| gpu_memory_total | Float | GPU总内存 |
| gpu_temperature | Float | GPU温度 |
| gpu_power_draw | Float | GPU功耗 |
| cpu_utilization | Float | CPU利用率 |
| cpu_memory_used | Float | CPU已用内存 |
| cpu_memory_total | Float | CPU总内存 |
| cpu_temperature | Float | CPU温度 |
| samples_per_second | Float | 每秒样本数 |
| tokens_per_second | Float | 每秒token数 |
| batch_size | Integer | 批次大小 |
| gradient_norm | Float | 梯度范数 |
| disk_read_speed | Float | 磁盘读取速度 |
| disk_write_speed | Float | 磁盘写入速度 |
| disk_utilization | Float | 磁盘利用率 |
| network_download_speed | Float | 网络下载速度 |
| network_upload_speed | Float | 网络上传速度 |
| network_latency | Float | 网络延迟 |
| created_at | DateTime | 创建时间 |

## 前端集成示例

### React Hook

```typescript
import { useState, useEffect, useCallback } from 'react';

interface ProgressData {
  epoch: number;
  step: number;
  loss: number;
  accuracy: number;
}

export function useTrainingProgress(sessionId: string) {
  const [progress, setProgress] = useState<ProgressData | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // 订阅
    fetch(`/api/v1/training/progress/realtime/subscribe/${sessionId}`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${getToken()}` }
    });

    // SSE 连接
    const eventSource = new EventSource(
      `/api/v1/training/progress/realtime/stream/${sessionId}`
    );

    eventSource.onopen = () => setConnected(true);
    
    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'progress_update') {
        setProgress(data.data);
      }
    };

    eventSource.onerror = (e) => {
      setError('Connection error');
      setConnected(false);
    };

    return () => {
      eventSource.close();
      fetch(`/api/v1/training/progress/realtime/unsubscribe/${sessionId}`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${getToken()}` }
      });
    };
  }, [sessionId]);

  return { progress, connected, error };
}

// 使用
function TrainingMonitor({ sessionId }) {
  const { progress, connected, error } = useTrainingProgress(sessionId);

  if (error) return <div>Error: {error}</div>;
  if (!connected) return <div>Connecting...</div>;
  if (!progress) return <div>Waiting for data...</div>;

  return (
    <div>
      <p>Epoch: {progress.epoch}</p>
      <p>Loss: {progress.loss.toFixed(4)}</p>
      <p>Accuracy: {(progress.accuracy * 100).toFixed(2)}%</p>
    </div>
  );
}
```

### Vue Composition API

```typescript
import { ref, onMounted, onUnmounted } from 'vue';

export function useTrainingProgress(sessionId: string) {
  const progress = ref(null);
  const connected = ref(false);
  let eventSource: EventSource | null = null;

  onMounted(async () => {
    // 订阅
    await fetch(`/api/v1/training/progress/realtime/subscribe/${sessionId}`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${getToken()}` }
    });

    // SSE
    eventSource = new EventSource(
      `/api/v1/training/progress/realtime/stream/${sessionId}`
    );

    eventSource.onopen = () => connected.value = true;
    eventSource.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === 'progress_update') {
        progress.value = data.data;
      }
    };
  });

  onUnmounted(() => {
    eventSource?.close();
    fetch(`/api/v1/training/progress/realtime/unsubscribe/${sessionId}`, {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${getToken()}` }
    });
  });

  return { progress, connected };
}
```

## 性能考虑

1. **高频更新优化**: 对于 batch 级别的更新，建议降低推送频率（如每 100 batch 推送一次）
2. **队列限制**: SSE 队列最大保留 100 条消息，超出会自动清理旧消息
3. **心跳机制**: SSE 每 15 秒发送心跳，WebSocket 默认 25 秒 ping 间隔
4. **断线重连**: 客户端应实现断线重连逻辑

## 错误码

| 错误码 | 描述 |
|--------|------|
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 404 | 会话不存在 |
| 500 | 服务器内部错误 |

## 版本历史

- **v1.0.0**: 基础进度查询和更新
- **v1.1.0**: 添加实时推送支持（WebSocket/SSE/长轮询）
- **v1.2.0**: 添加订阅管理和批量推送

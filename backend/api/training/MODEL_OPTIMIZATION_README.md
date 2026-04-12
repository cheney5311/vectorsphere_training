# 模型优化 API 文档

## 概述

模型优化 API 提供完整的模型压缩和推理优化功能，支持租户隔离和持久化存储。该模块采用分层架构设计，包括 API 层、服务层、仓库层和数据模型层。

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│              model_optimization_api.py                      │
│   (REST接口、参数验证、响应格式化、租户隔离)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Service Layer                          │
│            model_optimization_service.py                    │
│   (业务逻辑、优化执行、持久化调用)                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Repository Layer                         │
│          model_optimization_repository.py                   │
│   (数据访问、CRUD操作、查询过滤、统计计算)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Schema Layer                            │
│              training_models.py                             │
│   (数据模型定义：ModelOptimization)                         │
└─────────────────────────────────────────────────────────────┘
```

## 数据模型

### ModelOptimization (模型优化记录)

```python
class ModelOptimization(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型优化记录模型"""
    __tablename__ = 'model_optimizations'
    
    # 关联信息
    optimization_id = Column(String(100), unique=True, nullable=False)   # 优化ID
    original_model_id = Column(String(100), nullable=False)              # 原始模型ID
    optimized_model_id = Column(String(100))                             # 优化后模型ID
    user_id = Column(String(36), nullable=False)                         # 用户ID
    
    # 优化类型
    optimization_type = Column(String(50), nullable=False)               # compression/inference/auto
    technique = Column(String(50))                                       # 优化技术
    strategy = Column(String(50))                                        # 压缩策略
    status = Column(String(20), default='pending')                       # 状态
    
    # 压缩配置
    compression_ratio = Column(Float)                                    # 压缩率
    quantization_bits = Column(Integer)                                  # 量化位数
    preserve_accuracy = Column(Boolean, default=True)                    # 是否保持精度
    
    # 推理优化配置
    hardware_target = Column(String(50))                                 # 目标硬件
    graph_optimization = Column(Boolean, default=False)                  # 图优化
    operator_fusion = Column(Boolean, default=False)                     # 算子融合
    constant_folding = Column(Boolean, default=False)                    # 常量折叠
    dead_code_elimination = Column(Boolean, default=False)               # 死代码消除
    memory_optimization = Column(Boolean, default=False)                 # 内存优化
    
    # 结果指标
    accuracy_preserved = Column(Float)                                   # 精度保持率
    model_size_reduction = Column(Float)                                 # 模型大小减少率
    inference_speedup = Column(Float)                                    # 推理加速比
    latency_reduction = Column(Float)                                    # 延迟降低率
    memory_usage_reduction = Column(Float)                               # 内存使用降低率
    throughput_improvement = Column(Float)                               # 吞吐量提升倍数
    
    # 时间信息
    started_at = Column(DateTime)                                        # 优化开始时间
    completed_at = Column(DateTime)                                      # 优化完成时间
    optimization_time_seconds = Column(Float)                            # 优化耗时(秒)
```

## API 端点

### 1. 模型压缩

#### POST `/api/training/optimization/models/<model_id>/compress`

对模型进行压缩优化。

**请求头:**
```
Authorization: Bearer <token>
X-Tenant-ID: <tenant_id>
```

**请求体:**
```json
{
    "technique": "quantization",
    "compression_ratio": 0.5,
    "strategy": "structured",
    "quantization_bits": 8,
    "preserve_accuracy": true
}
```

**参数说明:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| technique | string | 否 | 压缩技术，默认 "pruning" |
| compression_ratio | float | 否 | 压缩率 (0-1)，默认 0.5 |
| strategy | string | 否 | 压缩策略 |
| quantization_bits | int | 否 | 量化位数，默认 8 |
| preserve_accuracy | bool | 否 | 是否保持精度，默认 true |

**支持的压缩技术:**
- `quantization`: 量化压缩，将模型权重从高精度转换为低精度表示
- `pruning`: 剪枝压缩，移除模型中不重要的权重或神经元
- `knowledge_distillation`: 知识蒸馏，使用小模型学习大模型的知识
- `low_rank_decomposition`: 低秩分解，将权重矩阵分解为低秩矩阵

**支持的压缩策略:**
- `structured`: 结构化剪枝，移除整个神经元或卷积核
- `unstructured`: 非结构化剪枝，移除单个权重
- `mixed`: 混合策略，结合结构化和非结构化方法

**响应:**
```json
{
    "success": true,
    "data": {
        "original_model_id": "uuid-xxx",
        "optimized_model_id": "optimized_quantization_uuid-xxx",
        "technique": "quantization",
        "compression_ratio": 0.5,
        "accuracy_preserved": 0.98,
        "model_size_reduction": 0.5,
        "inference_speedup": 1.3,
        "optimization_time": 5.23,
        "metrics": {
            "original_size_mb": 100.0,
            "compressed_size_mb": 50.0,
            "accuracy_before": 0.92,
            "accuracy_after": 0.90
        }
    }
}
```

### 2. 推理优化

#### POST `/api/training/optimization/models/<model_id>/optimize-inference`

对模型进行推理优化。

**请求体:**
```json
{
    "graph_optimization": true,
    "operator_fusion": true,
    "constant_folding": true,
    "dead_code_elimination": true,
    "memory_optimization": true,
    "hardware_target": "gpu"
}
```

**参数说明:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| graph_optimization | bool | 否 | 图优化，默认 true |
| operator_fusion | bool | 否 | 算子融合，默认 true |
| constant_folding | bool | 否 | 常量折叠，默认 true |
| dead_code_elimination | bool | 否 | 死代码消除，默认 true |
| memory_optimization | bool | 否 | 内存优化，默认 true |
| hardware_target | string | 否 | 目标硬件，默认 "cpu" |

**支持的硬件目标:**
- `cpu`: 通用 CPU 优化
- `gpu`: NVIDIA/AMD GPU 优化
- `tpu`: Google TPU 优化
- `edge`: 边缘设备优化

**响应:**
```json
{
    "success": true,
    "data": {
        "original_model_id": "uuid-xxx",
        "optimized_model_id": "inference_optimized_uuid-xxx",
        "optimization_config": {
            "graph_optimization": true,
            "operator_fusion": true,
            "constant_folding": true,
            "dead_code_elimination": true,
            "memory_optimization": true,
            "hardware_target": "gpu"
        },
        "latency_reduction": 0.35,
        "memory_usage_reduction": 0.25,
        "throughput_improvement": 1.5,
        "optimization_time": 3.15,
        "metrics": {
            "original_latency_ms": 120.0,
            "optimized_latency_ms": 78.0,
            "original_memory_mb": 200.0,
            "optimized_memory_mb": 150.0
        }
    }
}
```

### 3. 自动优化

#### POST `/api/training/optimization/models/<model_id>/auto-optimize`

根据目标约束自动选择最优的优化策略。

**请求体:**
```json
{
    "target_constraints": {
        "size_reduction": 0.6,
        "accuracy_preservation": 0.95,
        "speed_improvement": 1.5
    }
}
```

**参数说明:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| target_constraints.size_reduction | float | 否 | 目标大小减少率 (0-1) |
| target_constraints.accuracy_preservation | float | 否 | 精度保持要求 (0-1) |
| target_constraints.speed_improvement | float | 否 | 速度提升目标 |

**响应:**
```json
{
    "success": true,
    "data": {
        "original_model_id": "uuid-xxx",
        "optimized_model_id": "optimized_pruning_uuid-xxx",
        "technique": "pruning",
        "compression_ratio": 0.6,
        "accuracy_preserved": 0.95,
        "model_size_reduction": 0.6,
        "inference_speedup": 1.2,
        "optimization_time": 8.5,
        "auto_selected": true
    }
}
```

### 4. 获取优化技术

#### GET `/api/training/optimization/techniques`

获取支持的优化技术和策略列表。

**响应:**
```json
{
    "success": true,
    "data": {
        "techniques": [
            {
                "value": "quantization",
                "name": "QUANTIZATION",
                "description": "量化压缩，将模型权重从高精度转换为低精度表示"
            },
            {
                "value": "pruning",
                "name": "PRUNING",
                "description": "剪枝压缩，移除模型中不重要的权重或神经元"
            }
        ],
        "strategies": [
            {
                "value": "structured",
                "name": "STRUCTURED",
                "description": "结构化剪枝，移除整个神经元或卷积核"
            }
        ],
        "hardware_targets": [
            {"value": "cpu", "name": "CPU", "description": "通用CPU优化"},
            {"value": "gpu", "name": "GPU", "description": "NVIDIA/AMD GPU优化"}
        ]
    }
}
```

### 5. 获取优化历史

#### GET `/api/training/optimization/models/<model_id>/optimization-history`

获取模型的优化历史记录。

**查询参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| optimization_type | string | 否 | 优化类型过滤 |
| status | string | 否 | 状态过滤 |
| limit | int | 否 | 返回数量限制，默认 10 |

**响应:**
```json
{
    "success": true,
    "data": {
        "history": [
            {
                "optimization_id": "opt_abc123",
                "original_model_id": "uuid-xxx",
                "optimized_model_id": "optimized_quantization_uuid-xxx",
                "optimization_type": "compression",
                "technique": "quantization",
                "status": "completed",
                "compression_ratio": 0.5,
                "accuracy_preserved": 0.98,
                "model_size_reduction": 0.5,
                "inference_speedup": 1.3,
                "optimization_time_seconds": 5.23,
                "created_at": "2024-01-15T10:30:00Z"
            }
        ],
        "total": 5,
        "model_id": "uuid-xxx"
    }
}
```

### 6. 获取优化详情

#### GET `/api/training/optimization/optimization/<optimization_id>`

获取单个优化记录的详细信息。

**响应:**
```json
{
    "success": true,
    "data": {
        "optimization_id": "opt_abc123",
        "original_model_id": "uuid-xxx",
        "optimized_model_id": "optimized_quantization_uuid-xxx",
        "tenant_id": "tenant_001",
        "user_id": "user_001",
        "optimization_type": "compression",
        "technique": "quantization",
        "strategy": "structured",
        "status": "completed",
        "compression_ratio": 0.5,
        "quantization_bits": 8,
        "preserve_accuracy": true,
        "accuracy_preserved": 0.98,
        "model_size_reduction": 0.5,
        "inference_speedup": 1.3,
        "original_size_mb": 100.0,
        "optimized_size_mb": 50.0,
        "optimization_config": {...},
        "metrics": {...},
        "started_at": "2024-01-15T10:30:00Z",
        "completed_at": "2024-01-15T10:30:05Z",
        "optimization_time_seconds": 5.23,
        "created_at": "2024-01-15T10:30:00Z"
    }
}
```

### 7. 删除优化记录

#### DELETE `/api/training/optimization/optimization/<optimization_id>`

删除指定的优化记录。

**响应:**
```json
{
    "success": true,
    "message": "优化记录已删除: opt_abc123"
}
```

### 8. 获取用户优化列表

#### GET `/api/training/optimization/optimizations`

获取当前用户的所有优化记录。

**查询参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| optimization_type | string | 否 | 优化类型过滤 |
| technique | string | 否 | 技术过滤 |
| status | string | 否 | 状态过滤 |
| limit | int | 否 | 返回数量限制，默认 100 |
| offset | int | 否 | 偏移量，默认 0 |

**响应:**
```json
{
    "success": true,
    "data": {
        "optimizations": [...],
        "total": 25,
        "has_more": true
    }
}
```

### 9. 获取优化统计

#### GET `/api/training/optimization/statistics`

获取优化统计信息。

**查询参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model_id | string | 否 | 模型ID过滤 |

**响应:**
```json
{
    "success": true,
    "data": {
        "total_optimizations": 100,
        "by_status": {
            "completed": 85,
            "failed": 10,
            "pending": 5
        },
        "by_type": {
            "compression": 60,
            "inference": 30,
            "auto": 10
        },
        "completed_count": 85,
        "failed_count": 10,
        "avg_size_reduction": 0.48,
        "avg_speedup": 1.35
    }
}
```

### 10. 获取最佳优化

#### GET `/api/training/optimization/models/<model_id>/best`

获取模型的最佳优化记录。

**查询参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| optimization_type | string | 否 | 优化类型过滤 |
| metric | string | 否 | 评估指标，默认 "inference_speedup" |

**支持的指标:**
- `inference_speedup`: 推理加速比
- `model_size_reduction`: 模型大小减少率
- `accuracy_preserved`: 精度保持率

**响应:**
```json
{
    "success": true,
    "data": {
        "optimization_id": "opt_abc123",
        "technique": "quantization",
        "inference_speedup": 1.5,
        "model_size_reduction": 0.5,
        "accuracy_preserved": 0.98
    }
}
```

### 11. 比较优化记录

#### POST `/api/training/optimization/compare`

比较多个优化记录。

**请求体:**
```json
{
    "optimization_ids": ["opt_001", "opt_002", "opt_003"]
}
```

**响应:**
```json
{
    "success": true,
    "data": {
        "optimizations": [
            {
                "optimization_id": "opt_001",
                "technique": "quantization",
                "model_size_reduction": 0.5,
                "inference_speedup": 1.3,
                "accuracy_preserved": 0.98,
                "composite_score": 0.75
            }
        ],
        "rankings": {
            "by_size_reduction": ["opt_002", "opt_001", "opt_003"],
            "by_speedup": ["opt_001", "opt_002", "opt_003"],
            "by_accuracy": ["opt_001", "opt_003", "opt_002"]
        },
        "best_optimization_id": "opt_001",
        "recommendation": "推荐使用 opt_001，技术: quantization"
    }
}
```

## 使用示例

### Python 客户端示例

```python
import requests

BASE_URL = "http://localhost:5000/api/training/optimization"
TOKEN = "your-jwt-token"
TENANT_ID = "your-tenant-id"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "X-Tenant-ID": TENANT_ID,
    "Content-Type": "application/json"
}

# 1. 模型压缩
def compress_model(model_id):
    response = requests.post(
        f"{BASE_URL}/models/{model_id}/compress",
        headers=headers,
        json={
            "technique": "quantization",
            "compression_ratio": 0.5,
            "quantization_bits": 8
        }
    )
    return response.json()

# 2. 推理优化
def optimize_inference(model_id):
    response = requests.post(
        f"{BASE_URL}/models/{model_id}/optimize-inference",
        headers=headers,
        json={
            "graph_optimization": True,
            "operator_fusion": True,
            "hardware_target": "gpu"
        }
    )
    return response.json()

# 3. 自动优化
def auto_optimize(model_id):
    response = requests.post(
        f"{BASE_URL}/models/{model_id}/auto-optimize",
        headers=headers,
        json={
            "target_constraints": {
                "size_reduction": 0.6,
                "accuracy_preservation": 0.95
            }
        }
    )
    return response.json()

# 4. 获取优化历史
def get_optimization_history(model_id, limit=10):
    response = requests.get(
        f"{BASE_URL}/models/{model_id}/optimization-history",
        headers=headers,
        params={"limit": limit}
    )
    return response.json()

# 5. 获取统计信息
def get_statistics():
    response = requests.get(
        f"{BASE_URL}/statistics",
        headers=headers
    )
    return response.json()

# 使用示例
if __name__ == "__main__":
    MODEL_ID = "your-model-id"
    
    # 执行压缩
    result = compress_model(MODEL_ID)
    print(f"Compression: speedup={result['data']['inference_speedup']}x")
    
    # 获取历史
    history = get_optimization_history(MODEL_ID)
    print(f"History: {history['data']['total']} records")
```

### cURL 示例

```bash
# 模型压缩
curl -X POST http://localhost:5000/api/training/optimization/models/uuid-xxx/compress \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "technique": "quantization",
    "compression_ratio": 0.5
  }'

# 推理优化
curl -X POST http://localhost:5000/api/training/optimization/models/uuid-xxx/optimize-inference \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "graph_optimization": true,
    "hardware_target": "gpu"
  }'

# 获取优化历史
curl -X GET "http://localhost:5000/api/training/optimization/models/uuid-xxx/optimization-history?limit=10" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID"

# 获取统计信息
curl -X GET "http://localhost:5000/api/training/optimization/statistics" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID"
```

## 优化流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     优化请求                                     │
│       (compress / optimize-inference / auto-optimize)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               1. 创建优化记录 (status=running)                   │
│                    生成 optimization_id                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               2. 获取模型信息                                    │
│                    验证模型存在                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               3. 执行优化                                        │
│                    - 压缩: 量化/剪枝/蒸馏/分解                   │
│                    - 推理: 图优化/算子融合/内存优化              │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   成功: 更新记录         │     │   失败: 更新记录         │
│   status=completed       │     │   status=failed          │
│   保存指标值             │     │   保存错误信息           │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               4. 返回优化结果                                    │
│                    包含 optimization_id 和各项指标               │
└─────────────────────────────────────────────────────────────────┘
```

## 错误处理

### 错误码

| 错误码 | 说明 |
|--------|------|
| 400 | 请求参数错误（不支持的技术/策略等） |
| 401 | 未授权（缺少或无效的 JWT Token） |
| 403 | 禁止访问（租户权限不足） |
| 404 | 资源不存在（优化记录/模型不存在） |
| 500 | 服务器内部错误（优化执行失败等） |

### 错误响应格式

```json
{
    "success": false,
    "error": "错误描述信息"
}
```

## 配置说明

### 压缩配置 (OptimizationConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| technique | enum | pruning | 压缩技术 |
| compression_ratio | float | 0.5 | 压缩率 |
| strategy | enum | None | 压缩策略 |
| quantization_bits | int | 8 | 量化位数 |
| preserve_accuracy | bool | true | 是否保持精度 |

### 推理配置 (InferenceOptimizationConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| graph_optimization | bool | true | 图优化 |
| operator_fusion | bool | true | 算子融合 |
| constant_folding | bool | true | 常量折叠 |
| dead_code_elimination | bool | true | 死代码消除 |
| memory_optimization | bool | true | 内存优化 |
| hardware_target | string | "cpu" | 目标硬件 |

## 租户隔离

所有 API 支持通过 `X-Tenant-ID` 请求头实现租户隔离：

```
X-Tenant-ID: tenant_001
```

租户隔离确保：
- 每个租户只能访问自己的优化记录
- 统计数据仅包含当前租户的数据
- 删除操作需要验证租户权限

## 文件结构

```
backend/
├── api/training/
│   ├── model_optimization_api.py      # API 层实现
│   └── MODEL_OPTIMIZATION_README.md   # 本文档
├── services/
│   └── model_optimization_service.py  # 服务层实现
├── repositories/
│   └── model_optimization_repository.py  # 仓库层实现
└── schemas/
    └── training_models.py             # 数据模型定义
```

## 版本历史

### v1.1.0 (当前版本)
- 新增租户隔离支持
- 新增优化记录持久化
- 新增优化历史查询
- 新增优化统计功能
- 新增最佳优化查询
- 新增优化记录比较

### v1.0.0
- 基础模型压缩功能
- 推理优化功能
- 自动优化功能
- 获取支持的技术列表


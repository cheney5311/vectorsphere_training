# 模型评估 API 文档

## 概述

模型评估 API 提供完整的模型评估和对比功能，支持租户隔离和持久化存储。该模块采用分层架构设计，包括 API 层、服务层、仓库层和数据模型层。

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│              model_evaluation_api.py                        │
│   (REST接口、参数验证、响应格式化、租户隔离)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Service Layer                          │
│            model_evaluation_service.py                      │
│   (业务逻辑、评估执行、对比分析、持久化调用)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Repository Layer                         │
│          model_evaluation_repository.py                     │
│   (数据访问、CRUD操作、查询过滤、统计计算)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Schema Layer                            │
│              training_models.py                             │
│   (数据模型定义、字段约束、关系映射、序列化)                  │
└─────────────────────────────────────────────────────────────┘
```

## 数据模型

### ModelEvaluation (模型评估记录)

```python
class ModelEvaluation(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型评估记录模型"""
    __tablename__ = 'model_evaluations'
    
    # 关联信息
    evaluation_id = Column(String(100), unique=True, nullable=False)  # 评估ID
    model_id = Column(String(100), nullable=False)                     # 模型ID
    dataset_id = Column(String(100), nullable=False)                   # 数据集ID
    user_id = Column(String(36), nullable=False)                       # 用户ID
    
    # 评估配置
    evaluation_type = Column(String(50), default='automated')          # 评估类型
    validation_strategy = Column(String(50), default='holdout')        # 验证策略
    cross_validation_folds = Column(Integer, default=5)                # 交叉验证折数
    test_size = Column(Float, default=0.2)                             # 测试集比例
    
    # 状态
    status = Column(String(20), default='pending')                     # 状态
    
    # 汇总指标
    accuracy = Column(Float)                                           # 准确率
    precision = Column(Float)                                          # 精确率
    recall = Column(Float)                                             # 召回率
    f1_score = Column(Float)                                           # F1分数
    auc = Column(Float)                                                # AUC值
    loss = Column(Float)                                               # 损失值
    
    # 时间信息
    started_at = Column(DateTime)                                      # 评估开始时间
    completed_at = Column(DateTime)                                    # 评估完成时间
    duration_seconds = Column(Float)                                   # 评估耗时
```

### ModelEvaluationMetric (评估指标详情)

```python
class ModelEvaluationMetric(Base, UUIDMixin, TimestampMixin):
    """模型评估指标模型"""
    __tablename__ = 'model_evaluation_metrics'
    
    evaluation_id = Column(UUID, ForeignKey('model_evaluations.id'))   # 评估记录ID
    metric_name = Column(String(100), nullable=False)                  # 指标名称
    metric_type = Column(String(50), nullable=False)                   # 指标类型
    metric_value = Column(Float, nullable=False)                       # 指标值
    confidence_lower = Column(Float)                                   # 置信区间下限
    confidence_upper = Column(Float)                                   # 置信区间上限
    confidence_level = Column(Float, default=0.95)                     # 置信水平
    per_class_values = Column(JSON)                                    # 每个类别的指标值
```

### ModelComparisonRecord (模型对比记录)

```python
class ModelComparisonRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型对比记录模型"""
    __tablename__ = 'model_comparison_records'
    
    comparison_id = Column(String(100), unique=True, nullable=False)   # 对比ID
    user_id = Column(String(36), nullable=False)                       # 用户ID
    dataset_id = Column(String(100), nullable=False)                   # 数据集ID
    model_ids = Column(JSON, nullable=False)                           # 参与对比的模型ID列表
    winner_model_id = Column(String(100))                              # 获胜模型ID
    status = Column(String(20), default='pending')                     # 状态
    ranking = Column(JSON)                                             # 模型排名
    recommendations = Column(JSON)                                     # 推荐建议列表
    risk_assessment = Column(JSON)                                     # 风险评估
    detailed_results = Column(JSON)                                    # 详细对比结果
```

## API 端点

### 1. 模型评估

#### POST `/api/v1/training/model/evaluate`

评估模型性能，支持多种验证策略和评估指标。

**请求头:**
```
Authorization: Bearer <token>
X-Tenant-ID: <tenant_id>  # 可选，用于租户隔离
```

**请求体:**
```json
{
    "model_id": "uuid-xxx",
    "dataset_id": "uuid-xxx",
    "evaluation_config": {
        "validation_strategy": "holdout",
        "metrics": ["accuracy", "precision", "recall", "f1_score"],
        "cross_validation_folds": 5,
        "test_size": 0.2
    }
}
```

**响应:**
```json
{
    "success": true,
    "data": {
        "evaluation_id": "eval_abc123",
        "model_id": "uuid-xxx",
        "dataset_id": "uuid-xxx",
        "metrics": [
            {
                "name": "accuracy",
                "value": 0.92,
                "type": "accuracy",
                "description": "Baseline accuracy using RandomForest"
            },
            {
                "name": "precision",
                "value": 0.89,
                "type": "precision",
                "description": "Baseline precision using RandomForest"
            }
        ],
        "evaluation_config": {...},
        "timestamp": "2024-01-15T10:30:00Z",
        "duration_seconds": 5.23
    }
}
```

#### GET `/api/v1/training/model/evaluation/results`

获取评估结果列表。

**查询参数:**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model_id | string | 否 | 模型ID过滤 |
| evaluation_id | string | 否 | 评估ID过滤 |
| status | string | 否 | 状态过滤 (pending, running, completed, failed) |
| limit | int | 否 | 返回数量限制，默认100 |
| offset | int | 否 | 偏移量，默认0 |

**响应:**
```json
{
    "success": true,
    "data": {
        "evaluations": [...],
        "total": 25,
        "has_more": true
    }
}
```

#### GET `/api/v1/training/model/evaluation/<evaluation_id>`

获取单个评估结果详情。

**查询参数:**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| include_metrics | bool | 否 | 是否包含详细指标，默认true |

#### DELETE `/api/v1/training/model/evaluation/<evaluation_id>`

删除评估记录。

### 2. 模型对比

#### POST `/api/v1/training/model/models/compare`

对比多个模型的性能。

**请求体:**
```json
{
    "model_ids": ["uuid-1", "uuid-2", "uuid-3"],
    "dataset_id": "uuid-xxx",
    "comparison_config": {
        "comparison_metrics": ["accuracy", "f1_score", "inference_speed"],
        "decision_criteria": "multi_objective",
        "business_constraints": {}
    }
}
```

**响应:**
```json
{
    "success": true,
    "data": {
        "winner_model_id": "uuid-1",
        "recommendations": ["推荐使用模型 uuid-1"],
        "risk_assessment": {
            "overall_risk": "low",
            "performance_variance": 0.05,
            "data_drift_risk": "low"
        },
        "comparison_metrics": [...]
    }
}
```

#### GET `/api/v1/training/model/comparison/<comparison_id>`

获取对比记录详情。

#### GET `/api/v1/training/model/comparison/history`

获取对比历史。

**查询参数:**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| status | string | 否 | 状态过滤 |
| limit | int | 否 | 返回数量限制，默认20 |

#### DELETE `/api/v1/training/model/comparison/<comparison_id>`

删除对比记录。

### 3. 评估历史与统计

#### GET `/api/v1/training/model/models/<model_id>/evaluation-history`

获取模型的评估历史。

**查询参数:**
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| limit | int | 否 | 返回数量限制，默认10 |
| dataset_id | string | 否 | 数据集ID过滤 |

**响应:**
```json
{
    "success": true,
    "data": {
        "history": [
            {
                "evaluation_id": "eval_abc123",
                "model_id": "uuid-xxx",
                "dataset_id": "uuid-xxx",
                "metrics": {
                    "accuracy": 0.92,
                    "precision": 0.89,
                    "recall": 0.91,
                    "f1_score": 0.90
                },
                "status": "completed",
                "duration_seconds": 5.23,
                "created_at": "2024-01-15T10:30:00Z"
            }
        ],
        "total": 5,
        "model_id": "uuid-xxx"
    }
}
```

#### GET `/api/v1/training/model/evaluation/statistics`

获取评估统计信息。

**响应:**
```json
{
    "success": true,
    "data": {
        "total_evaluations": 100,
        "by_status": {
            "completed": 85,
            "failed": 10,
            "pending": 5
        },
        "avg_accuracy": 0.87,
        "completed_count": 85,
        "failed_count": 10
    }
}
```

### 4. 批量操作

#### POST `/api/v1/training/model/evaluate/batch`

批量评估多个模型。

**请求体:**
```json
{
    "model_ids": ["uuid-1", "uuid-2", "uuid-3"],
    "dataset_id": "uuid-xxx",
    "evaluation_config": {
        "metrics": ["accuracy", "f1_score"]
    }
}
```

**响应:**
```json
{
    "success": true,
    "data": {
        "results": [
            {
                "model_id": "uuid-1",
                "status": "completed",
                "metrics": {"accuracy": 0.92, "f1_score": 0.90},
                "evaluation_id": "eval_abc123"
            },
            {
                "model_id": "uuid-2",
                "status": "completed",
                "metrics": {"accuracy": 0.88, "f1_score": 0.86},
                "evaluation_id": "eval_def456"
            }
        ],
        "summary": {
            "total": 3,
            "completed": 3,
            "failed": 0
        }
    }
}
```

#### POST `/api/v1/training/model/models/best`

获取最佳模型。

**请求体:**
```json
{
    "model_ids": ["uuid-1", "uuid-2", "uuid-3"],
    "dataset_id": "uuid-xxx",
    "metric_name": "accuracy"
}
```

**响应:**
```json
{
    "success": true,
    "data": {
        "model_id": "uuid-1",
        "score": 0.92,
        "metric_name": "accuracy",
        "evaluation_id": "eval_abc123"
    }
}
```

### 5. 指标类型

#### GET `/api/v1/training/model/metrics/types`

获取支持的评估指标类型。

**响应:**
```json
{
    "success": true,
    "data": {
        "metric_types": [
            {
                "type": "accuracy",
                "name": "ACCURACY",
                "description": "准确率，正确预测的样本占总样本的比例"
            },
            {
                "type": "precision",
                "name": "PRECISION",
                "description": "精确率，预测为正的样本中实际为正的比例"
            },
            {
                "type": "recall",
                "name": "RECALL",
                "description": "召回率，实际为正的样本中被正确预测为正的比例"
            },
            {
                "type": "f1_score",
                "name": "F1_SCORE",
                "description": "F1分数，精确率和召回率的调和平均数"
            },
            {
                "type": "auc",
                "name": "AUC",
                "description": "AUC值，ROC曲线下的面积"
            }
        ]
    }
}
```

## 使用示例

### Python 客户端示例

```python
import requests

BASE_URL = "http://localhost:5000/api/v1/training/model"
TOKEN = "your-jwt-token"
TENANT_ID = "your-tenant-id"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "X-Tenant-ID": TENANT_ID,
    "Content-Type": "application/json"
}

# 1. 评估单个模型
def evaluate_model(model_id, dataset_id):
    response = requests.post(
        f"{BASE_URL}/evaluate",
        headers=headers,
        json={
            "model_id": model_id,
            "dataset_id": dataset_id,
            "evaluation_config": {
                "validation_strategy": "holdout",
                "metrics": ["accuracy", "precision", "recall", "f1_score"],
                "test_size": 0.2
            }
        }
    )
    return response.json()

# 2. 对比多个模型
def compare_models(model_ids, dataset_id):
    response = requests.post(
        f"{BASE_URL}/models/compare",
        headers=headers,
        json={
            "model_ids": model_ids,
            "dataset_id": dataset_id,
            "comparison_config": {
                "comparison_metrics": ["accuracy", "f1_score"],
                "decision_criteria": "multi_objective"
            }
        }
    )
    return response.json()

# 3. 获取评估历史
def get_evaluation_history(model_id, limit=10):
    response = requests.get(
        f"{BASE_URL}/models/{model_id}/evaluation-history",
        headers=headers,
        params={"limit": limit}
    )
    return response.json()

# 4. 批量评估
def batch_evaluate(model_ids, dataset_id):
    response = requests.post(
        f"{BASE_URL}/evaluate/batch",
        headers=headers,
        json={
            "model_ids": model_ids,
            "dataset_id": dataset_id
        }
    )
    return response.json()

# 5. 获取最佳模型
def get_best_model(model_ids, dataset_id, metric="accuracy"):
    response = requests.post(
        f"{BASE_URL}/models/best",
        headers=headers,
        json={
            "model_ids": model_ids,
            "dataset_id": dataset_id,
            "metric_name": metric
        }
    )
    return response.json()

# 使用示例
if __name__ == "__main__":
    MODEL_ID = "your-model-id"
    DATASET_ID = "your-dataset-id"
    
    # 评估模型
    result = evaluate_model(MODEL_ID, DATASET_ID)
    print(f"Evaluation ID: {result['data']['evaluation_id']}")
    print(f"Accuracy: {result['data']['metrics'][0]['value']}")
    
    # 对比模型
    model_ids = ["model-1", "model-2", "model-3"]
    comparison = compare_models(model_ids, DATASET_ID)
    print(f"Winner: {comparison['data']['winner_model_id']}")
    
    # 获取最佳模型
    best = get_best_model(model_ids, DATASET_ID, "accuracy")
    print(f"Best Model: {best['data']['model_id']}, Score: {best['data']['score']}")
```

### cURL 示例

```bash
# 评估模型
curl -X POST http://localhost:5000/api/v1/training/model/evaluate \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "uuid-xxx",
    "dataset_id": "uuid-xxx",
    "evaluation_config": {
      "metrics": ["accuracy", "precision", "recall", "f1_score"]
    }
  }'

# 获取评估历史
curl -X GET "http://localhost:5000/api/v1/training/model/models/uuid-xxx/evaluation-history?limit=10" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID"

# 对比模型
curl -X POST http://localhost:5000/api/v1/training/model/models/compare \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "model_ids": ["uuid-1", "uuid-2", "uuid-3"],
    "dataset_id": "uuid-xxx"
  }'

# 获取评估统计
curl -X GET "http://localhost:5000/api/v1/training/model/evaluation/statistics?model_id=uuid-xxx" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID"
```

## 评估流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     评估请求 (POST /evaluate)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               1. 创建评估记录 (status=running)                   │
│                    生成 evaluation_id                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               2. 获取模型和数据集信息                            │
│                    验证模型和数据集存在                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               3. 执行评估                                        │
│                    - 加载数据集                                  │
│                    - 根据验证策略分割数据                        │
│                    - 计算评估指标                                │
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
│               4. 保存详细指标到 metrics 表                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               5. 返回评估结果                                    │
│                    包含 evaluation_id 和各项指标                 │
└─────────────────────────────────────────────────────────────────┘
```

## 错误处理

### 错误码

| 错误码 | 说明 |
|--------|------|
| 400 | 请求参数错误（缺少必需参数、参数格式错误等） |
| 401 | 未授权（缺少或无效的 JWT Token） |
| 403 | 禁止访问（租户权限不足） |
| 404 | 资源不存在（评估记录、模型、数据集不存在） |
| 500 | 服务器内部错误（评估执行失败等） |

### 错误响应格式

```json
{
    "success": false,
    "error": "错误描述信息"
}
```

## 配置说明

### 评估配置 (evaluation_config)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| validation_strategy | string | "holdout" | 验证策略: holdout, cross_validation, bootstrap |
| metrics | array | ["accuracy", "precision", "recall", "f1_score"] | 评估指标列表 |
| cross_validation_folds | int | 5 | 交叉验证折数 |
| test_size | float | 0.2 | 测试集比例 |

### 对比配置 (comparison_config)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| comparison_metrics | array | ["accuracy", "f1_score"] | 对比指标列表 |
| decision_criteria | string | "multi_objective" | 决策标准: multi_objective, single_metric, weighted |
| business_constraints | object | {} | 业务约束 |

## 租户隔离

所有 API 支持通过 `X-Tenant-ID` 请求头实现租户隔离：

```
X-Tenant-ID: tenant_001
```

租户隔离确保：
- 每个租户只能访问自己的评估记录
- 统计数据仅包含当前租户的数据
- 删除操作需要验证租户权限

## 性能优化建议

1. **批量评估**: 使用 `/evaluate/batch` 端点一次评估多个模型，减少请求次数
2. **分页查询**: 使用 `limit` 和 `offset` 参数分页获取大量数据
3. **指标缓存**: 评估结果会持久化存储，重复查询可直接从数据库获取
4. **异步处理**: 对于耗时较长的评估任务，可考虑使用消息队列异步处理

## 文件结构

```
backend/
├── api/training/
│   ├── model_evaluation_api.py      # API 层实现
│   └── MODEL_EVALUATION_README.md   # 本文档
├── services/
│   └── model_evaluation_service.py  # 服务层实现
├── repositories/
│   └── model_evaluation_repository.py  # 仓库层实现
└── schemas/
    └── training_models.py           # 数据模型定义
```

## 版本历史

### v1.1.0 (当前版本)
- 新增租户隔离支持
- 新增评估记录持久化
- 新增批量评估功能
- 新增最佳模型查询
- 新增评估统计功能
- 新增对比历史记录

### v1.0.0
- 基础模型评估功能
- 模型对比功能
- 评估指标类型查询


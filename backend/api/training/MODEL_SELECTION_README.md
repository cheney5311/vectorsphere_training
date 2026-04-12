# 模型选择 API 文档

## 概述

模型选择 API 提供完整的模型推荐、配置和管理功能，支持租户隔离和持久化存储。该模块采用分层架构设计，包括 API 层、服务层、仓库层和数据模型层。

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│              model_selection_api.py                         │
│   (REST接口、参数验证、响应格式化、租户隔离)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Service Layer                          │
│            model_selection_service.py                       │
│   (业务逻辑、推荐算法、配置生成、持久化调用)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Repository Layer                         │
│          model_selection_repository.py                      │
│   (数据访问、CRUD操作、查询过滤、统计计算)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Schema Layer                            │
│              training_models.py                             │
│   (数据模型定义：推荐记录、配置记录、目录条目)               │
└─────────────────────────────────────────────────────────────┘
```

## 数据模型

### ModelRecommendationRecord (模型推荐记录)

```python
class ModelRecommendationRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型推荐记录模型"""
    __tablename__ = 'model_recommendation_records'
    
    recommendation_id = Column(String(100), unique=True, nullable=False)  # 推荐ID
    user_id = Column(String(36), nullable=False)                          # 用户ID
    task_type = Column(String(100), nullable=False)                       # 任务类型
    requirements = Column(JSON)                                           # 硬件和环境要求
    performance_requirements = Column(JSON)                               # 性能要求
    recommended_models = Column(JSON, nullable=False)                     # 推荐的模型列表
    top_recommendation = Column(String(200))                              # 首推模型名称
    top_confidence = Column(Float)                                        # 首推置信度
    num_recommendations = Column(Integer, default=0)                      # 推荐数量
    selected_model = Column(String(200))                                  # 用户选择的模型
    feedback_score = Column(Float)                                        # 用户反馈评分 (1-5)
    feedback_comment = Column(Text)                                       # 用户反馈评论
    is_helpful = Column(Boolean)                                          # 推荐是否有帮助
    status = Column(String(20), default='completed')                      # 状态
    response_time_ms = Column(Float)                                      # 响应时间(毫秒)
```

### ModelConfigurationRecord (模型配置记录)

```python
class ModelConfigurationRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型配置记录模型"""
    __tablename__ = 'model_configuration_records'
    
    configuration_id = Column(String(100), unique=True, nullable=False)   # 配置ID
    user_id = Column(String(36), nullable=False)                          # 用户ID
    model_name = Column(String(200), nullable=False)                      # 模型名称
    task_type = Column(String(100), nullable=False)                       # 任务类型
    framework = Column(String(50))                                        # 框架
    model_type = Column(String(50))                                       # 模型类型
    dataset_info = Column(JSON)                                           # 数据集信息
    hyperparameters = Column(JSON)                                        # 超参数配置
    training_config = Column(JSON)                                        # 训练配置
    hardware_config = Column(JSON)                                        # 硬件配置
    config_source = Column(String(50), default='auto')                    # 配置来源
    is_default = Column(Boolean, default=False)                           # 是否为默认配置
    usage_count = Column(Integer, default=0)                              # 使用次数
```

### ModelCatalogEntry (模型目录条目)

```python
class ModelCatalogEntry(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型目录条目模型"""
    __tablename__ = 'model_catalog_entries'
    
    model_name = Column(String(200), nullable=False)                      # 模型名称
    task_type = Column(String(100), nullable=False)                       # 适用任务类型
    framework = Column(String(50), nullable=False)                        # 框架
    model_type = Column(String(50), nullable=False)                       # 模型类型
    description = Column(Text)                                            # 模型描述
    performance_metrics = Column(JSON)                                    # 性能指标
    hardware_requirements = Column(JSON)                                  # 硬件要求
    default_hyperparameters = Column(JSON)                                # 默认超参数
    is_enabled = Column(Boolean, default=True)                            # 是否启用
    is_public = Column(Boolean, default=True)                             # 是否公开
    usage_count = Column(Integer, default=0)                              # 使用次数
    recommendation_count = Column(Integer, default=0)                     # 推荐次数
    selection_count = Column(Integer, default=0)                          # 被选择次数
```

## API 端点

### 1. 模型推荐

#### POST `/api/v1/training/models/recommend`

推荐适合任务的模型。

**请求头:**
```
Authorization: Bearer <token>
X-Tenant-ID: <tenant_id>
```

**请求体:**
```json
{
    "task_type": "text_classification",
    "requirements": {
        "gpu_memory": "8GB",
        "cpu_cores": 4
    },
    "performance_requirements": {
        "accuracy": 0.85,
        "speed": 0.7
    }
}
```

**参数说明:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| task_type | string | 是 | 任务类型 |
| requirements | object | 否 | 硬件和环境要求 |
| performance_requirements | object | 否 | 性能要求 |

**支持的任务类型:**
- `text_classification`: 文本分类
- `text_generation`: 文本生成
- `image_classification`: 图像分类
- `object_detection`: 目标检测
- `regression`: 回归
- `clustering`: 聚类

**响应:**
```json
{
    "success": true,
    "data": {
        "recommendations": [
            {
                "model_name": "bert-base-uncased",
                "framework": "huggingface",
                "model_type": "classification",
                "description": "BERT基础模型，适用于文本分类任务",
                "confidence": 0.87,
                "recommended_for": ["text_classification"],
                "performance_metrics": {
                    "accuracy": 0.87,
                    "speed": 0.75
                }
            }
        ],
        "total": 3,
        "task_type": "text_classification"
    },
    "message": "模型推荐完成"
}
```

### 2. 获取模型配置

#### POST `/api/v1/training/models/<model_name>/config`

获取模型的推荐配置。

**请求体:**
```json
{
    "task_type": "text_classification",
    "dataset_info": {
        "num_samples": 5000,
        "num_classes": 5
    },
    "save_configuration": true
}
```

**响应:**
```json
{
    "success": true,
    "data": {
        "model_name": "bert-base-uncased",
        "framework": "huggingface",
        "model_type": "classification",
        "hyperparameters": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 3,
            "warmup_steps": 500,
            "weight_decay": 0.01
        },
        "training_config": {
            "output_dir": "./outputs/bert-base-uncased_20240115_103000",
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "logging_steps": 100
        },
        "hardware_config": {
            "gpu_memory": "8GB",
            "cpu_cores": 4,
            "distributed_training": false,
            "mixed_precision": true
        }
    },
    "message": "模型配置获取成功"
}
```

### 3. 搜索模型

#### GET `/api/v1/training/models/search`

搜索模型目录。

**查询参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| q | string | 是 | 搜索关键词 |
| limit | int | 否 | 返回数量限制，默认10 |

**响应:**
```json
{
    "success": true,
    "data": {
        "results": [
            {
                "model_name": "bert-base-uncased",
                "task_type": "text_classification",
                "framework": "huggingface",
                "model_type": "classification",
                "description": "BERT基础模型，适用于文本分类任务",
                "performance": {"accuracy": 0.87, "speed": 0.75}
            }
        ],
        "total": 3,
        "query": "bert"
    },
    "message": "模型搜索完成"
}
```

### 4. 获取任务类型

#### GET `/api/v1/training/models/task-types`

获取支持的任务类型列表。

**响应:**
```json
{
    "success": true,
    "data": {
        "task_types": [
            "text_classification",
            "text_generation",
            "image_classification"
        ],
        "total": 3
    },
    "message": "获取任务类型成功"
}
```

### 5. 获取任务的模型列表

#### GET `/api/v1/training/models/by-task/<task_type>`

获取特定任务类型的可用模型列表。

**查询参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| framework | string | 否 | 框架过滤 |
| limit | int | 否 | 返回数量限制，默认50 |

**响应:**
```json
{
    "success": true,
    "data": {
        "models": [
            {
                "model_name": "bert-base-uncased",
                "task_type": "text_classification",
                "framework": "huggingface",
                "model_type": "classification",
                "description": "BERT基础模型",
                "performance": {"accuracy": 0.87},
                "requirements": {"gpu_memory": "8GB"},
                "source": "catalog"
            }
        ],
        "total": 3,
        "task_type": "text_classification"
    },
    "message": "获取模型列表成功"
}
```

### 6. 获取推荐历史

#### GET `/api/v1/training/recommendations/history`

获取用户的推荐历史记录。

**查询参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| task_type | string | 否 | 任务类型过滤 |
| limit | int | 否 | 返回数量限制，默认20 |

**响应:**
```json
{
    "success": true,
    "data": {
        "history": [
            {
                "recommendation_id": "rec_abc123",
                "task_type": "text_classification",
                "top_recommendation": "bert-base-uncased",
                "top_confidence": 0.87,
                "num_recommendations": 3,
                "selected_model": "bert-base-uncased",
                "feedback_score": 4.5,
                "created_at": "2024-01-15T10:30:00Z"
            }
        ],
        "total": 10
    },
    "message": "获取推荐历史成功"
}
```

### 7. 提交推荐反馈

#### POST `/api/v1/training/recommendations/<recommendation_id>/feedback`

提交对推荐结果的反馈。

**请求体:**
```json
{
    "selected_model": "bert-base-uncased",
    "feedback_score": 4.5,
    "feedback_comment": "推荐的模型很适合我的任务",
    "is_helpful": true
}
```

**参数说明:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| selected_model | string | 否 | 用户选择的模型 |
| feedback_score | float | 否 | 反馈评分 (1-5) |
| feedback_comment | string | 否 | 反馈评论 |
| is_helpful | bool | 否 | 推荐是否有帮助 |

**响应:**
```json
{
    "success": true,
    "data": {
        "recommendation_id": "rec_abc123"
    },
    "message": "反馈提交成功"
}
```

### 8. 获取推荐统计

#### GET `/api/v1/training/recommendations/statistics`

获取推荐统计信息。

**查询参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| task_type | string | 否 | 任务类型过滤 |

**响应:**
```json
{
    "success": true,
    "data": {
        "total_recommendations": 100,
        "by_task_type": {
            "text_classification": 45,
            "text_generation": 30,
            "image_classification": 25
        },
        "avg_confidence": 0.85,
        "helpful_rate": 0.92,
        "feedback_count": 75
    },
    "message": "获取统计信息成功"
}
```

### 9. 获取配置历史

#### GET `/api/v1/training/configurations/history`

获取用户的配置历史记录。

**查询参数:**

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model_name | string | 否 | 模型名称过滤 |
| task_type | string | 否 | 任务类型过滤 |
| limit | int | 否 | 返回数量限制，默认20 |

### 10. 保存自定义配置

#### POST `/api/v1/training/configurations`

保存自定义模型配置。

**请求体:**
```json
{
    "model_name": "bert-base-uncased",
    "task_type": "text_classification",
    "hyperparameters": {
        "learning_rate": 3e-5,
        "batch_size": 32,
        "num_epochs": 5
    },
    "training_config": {
        "warmup_ratio": 0.1,
        "save_strategy": "epoch"
    },
    "hardware_config": {
        "gpu_memory": "16GB",
        "cpu_cores": 8
    },
    "is_default": true,
    "tags": ["production", "optimized"]
}
```

### 11. 删除配置记录

#### DELETE `/api/v1/training/configurations/<configuration_id>`

删除指定的配置记录。

### 12. 添加模型到目录

#### POST `/api/v1/training/catalog/models`

添加自定义模型到目录。

**请求体:**
```json
{
    "model_name": "custom-bert-classifier",
    "task_type": "text_classification",
    "framework": "huggingface",
    "model_type": "classification",
    "description": "定制化的BERT分类模型",
    "performance_metrics": {
        "accuracy": 0.92,
        "speed": 0.8
    },
    "hardware_requirements": {
        "gpu_memory": "8GB",
        "cpu_cores": 4
    },
    "default_hyperparameters": {
        "learning_rate": 2e-5,
        "batch_size": 16
    },
    "tags": ["custom", "production"]
}
```

## 使用示例

### Python 客户端示例

```python
import requests

BASE_URL = "http://localhost:5000/api/v1/training"
TOKEN = "your-jwt-token"
TENANT_ID = "your-tenant-id"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "X-Tenant-ID": TENANT_ID,
    "Content-Type": "application/json"
}

# 1. 推荐模型
def recommend_models(task_type, requirements=None):
    response = requests.post(
        f"{BASE_URL}/models/recommend",
        headers=headers,
        json={
            "task_type": task_type,
            "requirements": requirements,
            "performance_requirements": {"accuracy": 0.8}
        }
    )
    return response.json()

# 2. 获取模型配置
def get_model_config(model_name, task_type):
    response = requests.post(
        f"{BASE_URL}/models/{model_name}/config",
        headers=headers,
        json={
            "task_type": task_type,
            "dataset_info": {"num_samples": 5000}
        }
    )
    return response.json()

# 3. 搜索模型
def search_models(query, limit=10):
    response = requests.get(
        f"{BASE_URL}/models/search",
        headers=headers,
        params={"q": query, "limit": limit}
    )
    return response.json()

# 4. 提交反馈
def submit_feedback(recommendation_id, selected_model, score):
    response = requests.post(
        f"{BASE_URL}/recommendations/{recommendation_id}/feedback",
        headers=headers,
        json={
            "selected_model": selected_model,
            "feedback_score": score,
            "is_helpful": True
        }
    )
    return response.json()

# 使用示例
if __name__ == "__main__":
    # 推荐模型
    result = recommend_models("text_classification", {"gpu_memory": "8GB"})
    print(f"推荐模型数: {result['data']['total']}")
    
    top_model = result['data']['recommendations'][0]
    print(f"首推模型: {top_model['model_name']}, 置信度: {top_model['confidence']}")
    
    # 获取配置
    config = get_model_config(top_model['model_name'], "text_classification")
    print(f"学习率: {config['data']['hyperparameters']['learning_rate']}")
```

### cURL 示例

```bash
# 推荐模型
curl -X POST http://localhost:5000/api/v1/training/models/recommend \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "text_classification",
    "requirements": {"gpu_memory": "8GB"}
  }'

# 获取模型配置
curl -X POST http://localhost:5000/api/v1/training/models/bert-base-uncased/config \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "text_classification"
  }'

# 搜索模型
curl -X GET "http://localhost:5000/api/v1/training/models/search?q=bert&limit=10" \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID"

# 获取任务类型
curl -X GET http://localhost:5000/api/v1/training/models/task-types \
  -H "Authorization: Bearer $TOKEN" \
  -H "X-Tenant-ID: $TENANT_ID"
```

## 推荐算法流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     推荐请求                                     │
│            (task_type + requirements)                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               1. 智能推荐系统                                    │
│                    (如果可用)                                    │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │ 失败                           │ 成功
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   2. 基于目录的推荐      │     │   返回推荐结果           │
│   - 过滤硬件要求        │     │                         │
│   - 过滤性能要求        │     │                         │
│   - 计算置信度          │     │                         │
└─────────────────────────┘     └─────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│               3. 保存推荐记录                                    │
│                    (recommendation_id)                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               4. 更新模型推荐计数                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               5. 返回推荐结果                                    │
│                    (按置信度排序)                                │
└─────────────────────────────────────────────────────────────────┘
```

## 置信度计算

置信度由以下因素计算:

```python
confidence = 0.5  # 基础置信度

# 根据性能指标调整 (最多 +0.5)
if accuracy:
    confidence += accuracy * 0.3
if speed:
    confidence += speed * 0.2

# 根据硬件匹配度调整 (最多 +0.3)
hardware_match_rate = matches / total_requirements
confidence += hardware_match_rate * 0.3

# 限制在 [0, 1] 范围内
confidence = max(0.0, min(1.0, confidence))
```

## 错误处理

### 错误码

| 错误码 | 说明 |
|--------|------|
| 400 | 请求参数错误（缺少必要参数、参数格式错误等） |
| 401 | 未授权（缺少或无效的 JWT Token） |
| 403 | 禁止访问（租户权限不足） |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

### 错误响应格式

```json
{
    "success": false,
    "error": "错误描述信息"
}
```

## 文件结构

```
backend/
├── api/training/
│   ├── model_selection_api.py       # API 层实现
│   └── MODEL_SELECTION_README.md    # 本文档
├── services/
│   └── model_selection_service.py   # 服务层实现
├── repositories/
│   └── model_selection_repository.py  # 仓库层实现
└── schemas/
    └── training_models.py           # 数据模型定义
```

## 版本历史

### v1.1.0 (当前版本)
- 新增租户隔离支持
- 新增推荐记录持久化
- 新增配置记录持久化
- 新增模型目录管理
- 新增推荐历史查询
- 新增用户反馈功能
- 新增统计分析功能

### v1.0.0
- 基础模型推荐功能
- 模型配置生成
- 模型搜索功能


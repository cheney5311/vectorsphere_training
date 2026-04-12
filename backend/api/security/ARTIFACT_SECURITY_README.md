# 工件安全策略 API 文档

## 概述

工件安全策略 API 提供工件（Artifact）的安全管理功能，包括安全策略配置、工件管理、版本控制、文件上传下载和访问控制等。

## 架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Artifact Security API Layer                           │
│                    (artifact_security_api.py)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Service Layer                                      │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │   ArtifactSecurityService   │  │    ArtifactManagementService        │   │
│  │   (artifact_security.py)    │  │    (artifact_management.py)         │   │
│  │  ──────────────────────────││  │  ─────────────────────────────────  │   │
│  │  - 文件验证和存储           │  │  - 工件创建、更新、删除              │   │
│  │  - 安全策略管理             │  │  - 版本管理和比较                    │   │
│  │  - 加密和访问控制           │  │  - 依赖关系管理和解析                │   │
│  │  - 审计日志                 │  │  - 生命周期管理                      │   │
│  │  - 恶意软件扫描             │  │  - 旧版本清理                        │   │
│  └─────────────────────────────┘  └─────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Repository Layer                                   │
│                       (artifact_repository.py)                               │
│  ┌────────────┬────────────┬────────────┬────────────┬────────────────┐     │
│  │   Policy   │  Artifact  │  Version   │ Dependency │  AccessLog     │     │
│  │    Repo    │    Repo    │    Repo    │    Repo    │    Repo        │     │
│  └────────────┴────────────┴────────────┴────────────┴────────────────┘     │
│  ┌────────────┬────────────┐                                                 │
│  │    File    │  Metadata  │                                                 │
│  │    Repo    │    Repo    │                                                 │
│  └────────────┴────────────┘                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                            Schema Layer                                      │
│                        (artifact_models.py)                                  │
│              (SQLAlchemy ORM Models & Enums & Dataclasses)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 服务职责划分

| 服务 | 职责 |
|------|------|
| **ArtifactSecurityService** | 负责安全相关的核心功能，包括文件验证、策略管理、加密存储、访问控制和审计 |
| **ArtifactManagementService** | 负责工件的业务管理，包括版本控制、依赖解析、生命周期管理和清理策略 |

## 功能模块

### 1. 安全策略管理

#### 1.1 获取安全策略列表
```http
GET /api/security/artifacts/policies
Authorization: Bearer <token>
```

**查询参数：**
| 参数 | 类型 | 描述 |
|-----|------|------|
| security_level | string | 安全级别筛选 |

**响应：**
```json
{
  "success": true,
  "data": [
    {
      "id": "public_default",
      "name": "公开文件默认策略",
      "description": "适用于公开文件的默认安全策略",
      "security_level": "public",
      "allowed_file_types": [".txt", ".pdf", ".jpg"],
      "max_file_size": 10485760,
      "encryption_required": false,
      "virus_scan_required": true,
      "access_control_enabled": false,
      "audit_enabled": true,
      "retention_days": 365,
      "is_default": true,
      "is_active": true,
      "created_at": "2026-01-10T12:00:00Z",
      "updated_at": "2026-01-10T12:00:00Z"
    }
  ]
}
```

#### 1.2 获取单个安全策略
```http
GET /api/security/artifacts/policies/<policy_id>
Authorization: Bearer <token>
```

#### 1.3 创建安全策略
```http
POST /api/security/artifacts/policies
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体：**
```json
{
  "name": "自定义策略",
  "description": "自定义安全策略描述",
  "security_level": "internal",
  "allowed_file_types": [".txt", ".pdf", ".csv"],
  "max_file_size": 52428800,
  "encryption_required": true,
  "virus_scan_required": true,
  "access_control_enabled": true,
  "audit_enabled": true,
  "retention_days": 730
}
```

#### 1.4 更新安全策略
```http
PUT /api/security/artifacts/policies/<policy_id>
Authorization: Bearer <token>
```

#### 1.5 删除安全策略
```http
DELETE /api/security/artifacts/policies/<policy_id>
Authorization: Bearer <token>
```

### 2. 工件管理

#### 2.1 列出工件
```http
GET /api/security/artifacts/artifacts
Authorization: Bearer <token>
```

**查询参数：**
| 参数 | 类型 | 描述 |
|-----|------|------|
| artifact_type | string | 工件类型 |
| security_level | string | 安全级别 |
| status | string | 状态 |
| owner_id | string | 所有者ID |
| limit | integer | 限制数量（默认100）|
| offset | integer | 偏移量（默认0）|

**工件类型：**
- `training_data` - 训练数据
- `model_file` - 模型文件
- `config_file` - 配置文件
- `log_file` - 日志文件
- `report` - 报告
- `backup` - 备份
- `temp` - 临时文件

**安全级别：**
- `public` - 公开
- `internal` - 内部
- `confidential` - 机密
- `restricted` - 限制

**工件状态：**
- `active` - 活跃
- `archived` - 已归档
- `deleted` - 已删除
- `pending` - 待处理
- `locked` - 已锁定

#### 2.2 创建工件
```http
POST /api/security/artifacts/artifacts
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体：**
```json
{
  "name": "训练数据集",
  "description": "用于模型训练的数据集",
  "artifact_type": "training_data",
  "security_level": "internal",
  "tags": ["training", "dataset"],
  "metadata": {
    "format": "csv",
    "records": 10000
  },
  "policy_id": "internal_default"
}
```

#### 2.3 获取工件详情
```http
GET /api/security/artifacts/artifacts/<artifact_id>
Authorization: Bearer <token>
```

**响应：**
```json
{
  "success": true,
  "data": {
    "id": "artifact_abc123",
    "name": "训练数据集",
    "description": "用于模型训练的数据集",
    "artifact_type": "training_data",
    "security_level": "internal",
    "status": "active",
    "owner_id": "user_001",
    "current_version": "1.0.0",
    "version_count": 3,
    "total_size": 1048576,
    "tags": ["training", "dataset"],
    "metadata": {},
    "created_at": "2026-01-10T12:00:00Z",
    "updated_at": "2026-01-10T12:00:00Z",
    "versions": [
      {
        "id": "ver_001",
        "version": "1.0.0",
        "status": "active",
        "file_size": 524288,
        "changelog": "Initial version",
        "created_by": "user_001",
        "created_at": "2026-01-10T12:00:00Z",
        "tags": []
      }
    ],
    "dependencies": [
      {
        "target_artifact_id": "artifact_xyz789",
        "dependency_type": "required",
        "version_constraint": ">=1.0.0"
      }
    ],
    "dependents": []
  }
}
```

#### 2.4 更新工件状态
```http
PUT /api/security/artifacts/artifacts/<artifact_id>/status
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体：**
```json
{
  "status": "archived"
}
```

### 3. 版本管理

#### 3.1 上传工件版本
```http
POST /api/security/artifacts/artifacts/<artifact_id>/upload
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

**表单字段：**
| 字段 | 类型 | 必填 | 描述 |
|-----|------|-----|------|
| file | file | 是 | 上传的文件 |
| version | string | 是 | 版本号 |
| changelog | string | 否 | 变更日志 |
| tags | array | 否 | 标签 |

**响应：**
```json
{
  "success": true,
  "data": {
    "id": "ver_002",
    "version": "1.1.0",
    "artifact_id": "artifact_abc123",
    "file_size": 524288
  },
  "message": "版本上传成功"
}
```

#### 3.2 下载工件版本
```http
GET /api/security/artifacts/artifacts/<artifact_id>/versions/<version>/download
Authorization: Bearer <token>
```

#### 3.3 删除工件版本
```http
DELETE /api/security/artifacts/artifacts/<artifact_id>/versions/<version>
Authorization: Bearer <token>
```

### 4. 工件操作

#### 4.1 更新工件
```http
PUT /api/security/artifacts/artifacts/<artifact_id>
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体：**
```json
{
  "name": "Updated Artifact Name",
  "description": "Updated description",
  "tags": ["updated", "tag"],
  "metadata": {"key": "value"}
}
```

#### 4.2 删除工件
```http
DELETE /api/security/artifacts/artifacts/<artifact_id>
Authorization: Bearer <token>
```

**注意：** 如果有其他工件依赖此工件，删除将失败。

#### 4.3 获取版本列表
```http
GET /api/security/artifacts/artifacts/<artifact_id>/versions?include_deleted=false
Authorization: Bearer <token>
```

**响应：**
```json
{
  "success": true,
  "data": [
    {
      "id": "ver_001",
      "version": "1.0.0",
      "status": "active",
      "file_size": 524288,
      "file_hash": "abc123...",
      "mime_type": "application/octet-stream",
      "changelog": "Initial version",
      "created_by": "user_001",
      "created_at": "2026-01-10T12:00:00Z",
      "tags": []
    }
  ],
  "count": 1
}
```

#### 4.4 比较版本
```http
POST /api/security/artifacts/artifacts/<artifact_id>/versions/compare
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体：**
```json
{
  "version1": "1.0.0",
  "version2": "2.0.0"
}
```

**响应：**
```json
{
  "success": true,
  "data": {
    "version1": {
      "version": "1.0.0",
      "file_size": 524288,
      "file_hash": "abc123...",
      "created_at": "2026-01-10T12:00:00Z",
      "changelog": "Initial version"
    },
    "version2": {
      "version": "2.0.0",
      "file_size": 1048576,
      "file_hash": "def456...",
      "created_at": "2026-01-11T12:00:00Z",
      "changelog": "Major update"
    },
    "size_diff": 524288,
    "hash_changed": true,
    "is_newer": true
  }
}
```

### 5. 依赖管理

#### 5.1 获取依赖树
```http
GET /api/security/artifacts/artifacts/<artifact_id>/dependencies/tree
Authorization: Bearer <token>
```

**响应：**
```json
{
  "success": true,
  "data": {
    "artifact_id": "art_001",
    "name": "Main Artifact",
    "current_version": "1.0.0",
    "dependencies": {
      "art_002": {
        "dependency_type": "required",
        "version_constraint": ">=1.0.0",
        "dependencies": {}
      }
    }
  }
}
```

#### 5.2 获取依赖此工件的列表
```http
GET /api/security/artifacts/artifacts/<artifact_id>/dependents
Authorization: Bearer <token>
```

**响应：**
```json
{
  "success": true,
  "data": [
    {
      "source_artifact_id": "art_003",
      "dependency_type": "required",
      "version_constraint": ">=1.0.0"
    }
  ],
  "count": 1
}
```

#### 5.3 添加依赖
```http
POST /api/security/artifacts/artifacts/<artifact_id>/dependencies
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体：**
```json
{
  "target_artifact_id": "artifact_xyz789",
  "dependency_type": "required",
  "version_constraint": ">=1.0.0"
}
```

**依赖类型：**
- `required` - 必需依赖
- `optional` - 可选依赖
- `dev` - 开发依赖

#### 5.4 移除依赖
```http
DELETE /api/security/artifacts/artifacts/<artifact_id>/dependencies/<target_artifact_id>
Authorization: Bearer <token>
```

### 6. 文件管理

#### 6.1 列出文件
```http
GET /api/security/artifacts/files
Authorization: Bearer <token>
```

**查询参数：**
| 参数 | 类型 | 描述 |
|-----|------|------|
| artifact_type | string | 工件类型筛选 |
| limit | integer | 限制数量 |
| offset | integer | 偏移量 |

#### 6.2 获取文件元数据
```http
GET /api/security/artifacts/files/<file_id>
Authorization: Bearer <token>
```

#### 6.3 下载文件
```http
GET /api/security/artifacts/files/<file_id>/download
Authorization: Bearer <token>
```

#### 6.4 删除文件
```http
DELETE /api/security/artifacts/files/<file_id>
Authorization: Bearer <token>
```

### 7. 清理功能

#### 7.1 清理工件
```http
POST /api/security/artifacts/cleanup
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体：**
```json
{
  "retention_days": 90
}
```

**响应：**
```json
{
  "success": true,
  "data": {
    "cleaned_files": 15,
    "cleaned_versions": 8
  },
  "message": "清理完成"
}
```

### 8. 健康检查

```http
GET /api/security/artifacts/health
```

**响应：**
```json
{
  "success": true,
  "status": "healthy",
  "timestamp": "2026-01-10T12:00:00Z",
  "service": "artifact_security"
}
```

## 数据模型

### SecurityPolicy（安全策略）
| 字段 | 类型 | 描述 |
|-----|------|------|
| id | String | 策略ID |
| name | String | 策略名称 |
| description | String | 描述 |
| security_level | Enum | 安全级别 |
| allowed_file_types | Array | 允许的文件类型 |
| max_file_size | Integer | 最大文件大小（字节）|
| encryption_required | Boolean | 是否需要加密 |
| virus_scan_required | Boolean | 是否需要病毒扫描 |
| access_control_enabled | Boolean | 是否启用访问控制 |
| audit_enabled | Boolean | 是否启用审计 |
| retention_days | Integer | 保留天数 |
| is_default | Boolean | 是否为默认策略 |
| is_active | Boolean | 是否激活 |

### Artifact（工件）
| 字段 | 类型 | 描述 |
|-----|------|------|
| id | String | 工件ID |
| name | String | 工件名称 |
| description | String | 描述 |
| artifact_type | Enum | 工件类型 |
| security_level | Enum | 安全级别 |
| status | Enum | 状态 |
| owner_id | String | 所有者ID |
| current_version | String | 当前版本 |
| version_count | Integer | 版本数量 |
| total_size | Integer | 总大小 |
| tags | Array | 标签 |
| metadata | Object | 元数据 |

### ArtifactVersion（工件版本）
| 字段 | 类型 | 描述 |
|-----|------|------|
| id | String | 版本ID |
| artifact_id | String | 工件ID |
| version | String | 版本号 |
| status | String | 状态 |
| file_path | String | 文件路径 |
| file_size | Integer | 文件大小 |
| file_hash | String | 文件哈希 |
| mime_type | String | MIME类型 |
| changelog | String | 变更日志 |
| created_by | String | 创建者 |

### FileMetadata（文件元数据）
| 字段 | 类型 | 描述 |
|-----|------|------|
| id | String | 文件ID |
| original_name | String | 原始文件名 |
| stored_name | String | 存储文件名 |
| file_path | String | 文件路径 |
| file_type | Enum | 文件类型 |
| mime_type | String | MIME类型 |
| size | Integer | 文件大小 |
| hash_sha256 | String | SHA256哈希 |
| security_level | Enum | 安全级别 |
| is_encrypted | Boolean | 是否加密 |

## 安全特性

### 1. 文件验证
- **类型验证**：根据安全策略验证文件扩展名
- **大小验证**：根据安全策略限制文件大小
- **病毒扫描**：上传时执行安全扫描

### 2. 加密存储
- **按需加密**：机密和限制级别文件自动加密
- **安全密钥**：使用配置的加密密钥

### 3. 访问控制
- **权限检查**：基于角色的访问控制
- **操作审计**：所有操作记录日志

### 4. 文件权限
| 安全级别 | Linux权限 | 描述 |
|---------|----------|------|
| public | 644 | 所有人可读 |
| internal | 640 | 组内可读 |
| confidential | 600 | 仅所有者可读写 |
| restricted | 600 | 仅所有者可读写 |

## 错误处理

### HTTP状态码
| 状态码 | 描述 |
|-------|------|
| 200 | 成功 |
| 201 | 创建成功 |
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 403 | 权限不足 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

### 响应格式
**成功：**
```json
{
  "success": true,
  "data": {...},
  "message": "操作成功"
}
```

**失败：**
```json
{
  "success": false,
  "error": "错误描述"
}
```

## 使用示例

### Python 客户端示例
```python
import requests

BASE_URL = "http://localhost:5000/api/security/artifacts"
HEADERS = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json"
}

# 创建工件
response = requests.post(
    f"{BASE_URL}/artifacts",
    headers=HEADERS,
    json={
        "name": "My Model",
        "artifact_type": "model_file",
        "security_level": "internal"
    }
)
artifact = response.json()["data"]

# 上传版本
with open("model.pkl", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/artifacts/{artifact['id']}/upload",
        headers={"Authorization": "Bearer <token>"},
        files={"file": f},
        data={"version": "1.0.0", "changelog": "Initial release"}
    )

# 下载版本
response = requests.get(
    f"{BASE_URL}/artifacts/{artifact['id']}/versions/1.0.0/download",
    headers={"Authorization": "Bearer <token>"}
)
with open("downloaded_model.pkl", "wb") as f:
    f.write(response.content)
```

### cURL 示例
```bash
# 获取安全策略列表
curl -X GET "http://localhost:5000/api/security/artifacts/policies" \
  -H "Authorization: Bearer <token>"

# 创建工件
curl -X POST "http://localhost:5000/api/security/artifacts/artifacts" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Artifact", "artifact_type": "training_data", "security_level": "internal"}'

# 上传版本
curl -X POST "http://localhost:5000/api/security/artifacts/artifacts/<artifact_id>/upload" \
  -H "Authorization: Bearer <token>" \
  -F "file=@data.csv" \
  -F "version=1.0.0" \
  -F "changelog=Initial version"
```

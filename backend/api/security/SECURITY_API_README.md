# Security API 文档

## 概述

Security API 提供统一的安全服务接口，包括用户认证、访问控制、审计日志、加密服务和合规检查功能。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Security API Layer                      │
│                   (security_api.py)                          │
├─────────────────────────────────────────────────────────────┤
│                     Service Layer                            │
│                 (security_service.py)                        │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │   Auth   │  Access  │  Audit   │Encryption│Compliance│   │
│  │ Service  │ Service  │ Service  │ Service  │ Service  │   │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    Repository Layer                          │
│                (security_repository.py)                      │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐   │
│  │ Session  │  Role    │  Policy  │   Key    │Processing│   │
│  │  Repo    │  Repo    │   Repo   │   Repo   │   Repo   │   │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘   │
├─────────────────────────────────────────────────────────────┤
│                     Schema Layer                             │
│                 (security_models.py)                         │
│         (SQLAlchemy ORM Models & Pydantic Schemas)          │
└─────────────────────────────────────────────────────────────┘
```

## 功能模块

### 1. 用户认证 (Authentication)

#### 1.1 用户登录
```http
POST /api/v1/security/auth/login
```

**请求体：**
```json
{
  "username": "user@example.com",
  "password": "password123"
}
```

**响应（成功）：**
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expires_at": "2026-01-11T10:00:00Z",
    "user_info": {
      "id": "user-001",
      "username": "user@example.com",
      "roles": ["user"]
    }
  }
}
```

**响应（需要MFA）：**
```json
{
  "success": false,
  "requires_mfa": true,
  "temp_token": "temp_token_xxx",
  "mfa_methods": ["totp"]
}
```

#### 1.2 用户登出
```http
POST /api/v1/security/auth/logout
Authorization: Bearer <token>
```

#### 1.3 验证令牌
```http
POST /api/v1/security/auth/token/validate
```

**请求体：**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

#### 1.4 刷新令牌
```http
POST /api/v1/security/auth/token/refresh
Authorization: Bearer <token>
```

#### 1.5 MFA验证
```http
POST /api/v1/security/auth/mfa/verify
```

**请求体：**
```json
{
  "temp_token": "temp_token_xxx",
  "mfa_code": "123456",
  "mfa_method": "totp"
}
```

#### 1.6 设置MFA
```http
POST /api/v1/security/auth/mfa/setup
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "user_id": "user-001",
  "mfa_method": "totp"
}
```

**响应：**
```json
{
  "success": true,
  "data": {
    "secret": "JBSWY3DPEHPK3PXP",
    "provisioning_uri": "otpauth://totp/VectorSphere:user@example.com?...",
    "qr_code": "data:image/png;base64,..."
  }
}
```

#### 1.7 确认MFA设置
```http
POST /api/v1/security/auth/mfa/confirm
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "user_id": "user-001",
  "verification_code": "123456"
}
```

#### 1.8 获取MFA状态
```http
GET /api/v1/security/auth/mfa/status/<user_id>
Authorization: Bearer <token>
```

#### 1.9 禁用MFA
```http
POST /api/v1/security/auth/mfa/disable
Authorization: Bearer <token>
```

#### 1.10 获取用户会话
```http
GET /api/v1/security/auth/sessions?user_id=<user_id>
Authorization: Bearer <token>
```

#### 1.11 撤销会话
```http
DELETE /api/v1/security/auth/sessions/<session_id>
Authorization: Bearer <token>
```

### 2. 访问控制 (Access Control)

#### 2.1 检查权限
```http
POST /api/v1/security/access/check
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "user_id": "user-001",
  "resource": "training:job:123",
  "action": "read",
  "context": {
    "department": "ml"
  }
}
```

**响应：**
```json
{
  "success": true,
  "data": {
    "allowed": true,
    "reason": "Matched policy: allow-ml-read",
    "matched_policies": ["allow-ml-read"],
    "risk_score": 0.1
  }
}
```

#### 2.2 分配角色
```http
POST /api/v1/security/access/roles
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "user_id": "user-001",
  "role": "admin",
  "expires_at": "2026-12-31T23:59:59Z"
}
```

#### 2.3 撤销角色
```http
POST /api/v1/security/access/roles/revoke
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "user_id": "user-001",
  "role": "admin"
}
```

#### 2.4 获取用户角色
```http
GET /api/v1/security/access/roles/<user_id>
Authorization: Bearer <token>
```

#### 2.5 创建策略
```http
POST /api/v1/security/access/policies
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "name": "allow-ml-read",
  "description": "Allow ML team to read training resources",
  "effect": "allow",
  "principals": ["role:ml-engineer"],
  "resources": ["training:*"],
  "actions": ["read", "list"],
  "conditions": {
    "department": {"equals": "ml"}
  },
  "priority": 50
}
```

#### 2.6 列出策略
```http
GET /api/v1/security/access/policies
Authorization: Bearer <token>
```

#### 2.7 获取策略详情
```http
GET /api/v1/security/access/policies/<policy_id>
Authorization: Bearer <token>
```

#### 2.8 更新策略
```http
PUT /api/v1/security/access/policies/<policy_id>
Authorization: Bearer <token>
```

#### 2.9 删除策略
```http
DELETE /api/v1/security/access/policies/<policy_id>
Authorization: Bearer <token>
```

### 3. 审计日志 (Audit Logging)

#### 3.1 记录审计事件
```http
POST /api/v1/security/audit/log
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "event_type": "user_login",
  "message": "User logged in successfully",
  "user_id": "user-001",
  "resource": "auth:session",
  "action": "create",
  "result": "success",
  "level": "info",
  "details": {
    "ip": "192.168.1.1"
  },
  "tags": ["auth", "login"]
}
```

#### 3.2 查询审计事件
```http
POST /api/v1/security/audit/query
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "start_time": "2026-01-01T00:00:00Z",
  "end_time": "2026-01-10T23:59:59Z",
  "event_types": ["user_login", "user_logout"],
  "levels": ["info", "warning"],
  "user_ids": ["user-001"],
  "limit": 100,
  "offset": 0
}
```

#### 3.3 获取审计统计
```http
GET /api/v1/security/audit/statistics?start_time=2026-01-01T00:00:00Z&end_time=2026-01-10T23:59:59Z
Authorization: Bearer <token>
```

#### 3.4 清理旧日志
```http
POST /api/v1/security/audit/cleanup
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "retention_days": 90
}
```

### 4. 加密服务 (Encryption)

#### 4.1 加密数据
```http
POST /api/v1/security/encryption/encrypt
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "data": "sensitive data to encrypt",
  "algorithm": "aes-256-gcm",
  "key_id": "key-001"
}
```

**响应：**
```json
{
  "success": true,
  "data": {
    "encrypted_data": "a1b2c3d4...",
    "key_id": "key-001",
    "algorithm": "aes-256-gcm",
    "iv": "e5f6g7h8...",
    "tag": "i9j0k1l2..."
  }
}
```

#### 4.2 解密数据
```http
POST /api/v1/security/encryption/decrypt
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "encrypted_data": "a1b2c3d4...",
  "key_id": "key-001",
  "iv": "e5f6g7h8...",
  "tag": "i9j0k1l2..."
}
```

#### 4.3 生成密钥
```http
POST /api/v1/security/encryption/keys
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "name": "data-encryption-key",
  "algorithm": "aes-256-gcm",
  "description": "Key for encrypting user data",
  "expires_days": 365
}
```

#### 4.4 列出密钥
```http
GET /api/v1/security/encryption/keys?status=active
Authorization: Bearer <token>
```

#### 4.5 获取密钥信息
```http
GET /api/v1/security/encryption/keys/<key_id>
Authorization: Bearer <token>
```

#### 4.6 轮换密钥
```http
POST /api/v1/security/encryption/keys/<key_id>/rotate
Authorization: Bearer <token>
```

#### 4.7 删除密钥
```http
DELETE /api/v1/security/encryption/keys/<key_id>
Authorization: Bearer <token>
```

### 5. 合规检查 (Compliance)

#### 5.1 检查合规性
```http
POST /api/v1/security/compliance/check
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "standard": "gdpr",
  "context": {
    "data_categories": ["personal", "sensitive"],
    "processing_location": "EU"
  }
}
```

**响应：**
```json
{
  "success": true,
  "data": {
    "report_id": "report-001",
    "standard": "gdpr",
    "overall_level": "compliant",
    "score": 85,
    "total_rules": 20,
    "compliant_rules": 17,
    "non_compliant_rules": 3,
    "violations": [
      {
        "rule_id": "gdpr-005",
        "description": "Data retention period exceeds limit",
        "severity": "medium"
      }
    ],
    "recommendations": [
      "Reduce data retention period to 24 months",
      "Add explicit consent for marketing purposes"
    ]
  }
}
```

#### 5.2 记录数据处理活动
```http
POST /api/v1/security/compliance/records
Authorization: Bearer <token>
```

**请求体：**
```json
{
  "data_subject_id": "user-001",
  "data_categories": ["personal", "contact"],
  "processing_purposes": ["service_delivery", "marketing"],
  "legal_basis": "consent",
  "consent_given": true,
  "retention_period_days": 730,
  "processing_location": "EU",
  "third_party_sharing": false,
  "third_parties": []
}
```

#### 5.3 列出数据处理记录
```http
GET /api/v1/security/compliance/records?data_subject_id=user-001&limit=50
Authorization: Bearer <token>
```

#### 5.4 列出合规报告
```http
GET /api/v1/security/compliance/reports?standard=gdpr
Authorization: Bearer <token>
```

#### 5.5 获取合规报告详情
```http
GET /api/v1/security/compliance/reports/<report_id>
Authorization: Bearer <token>
```

#### 5.6 列出支持的合规标准
```http
GET /api/v1/security/compliance/standards
```

**响应：**
```json
{
  "success": true,
  "data": [
    {"id": "gdpr", "name": "GDPR", "description": "EU General Data Protection Regulation"},
    {"id": "ccpa", "name": "CCPA", "description": "California Consumer Privacy Act"},
    {"id": "soc2", "name": "SOC 2", "description": "Service Organization Control 2"},
    {"id": "iso27001", "name": "ISO 27001", "description": "Information Security Management"},
    {"id": "hipaa", "name": "HIPAA", "description": "Health Insurance Portability and Accountability Act"},
    {"id": "pci_dss", "name": "PCI DSS", "description": "Payment Card Industry Data Security Standard"}
  ]
}
```

### 6. 健康检查

```http
GET /api/v1/security/health
```

**响应：**
```json
{
  "success": true,
  "status": "healthy",
  "timestamp": "2026-01-10T12:00:00Z",
  "components": {
    "auth": "ok",
    "access": "ok",
    "audit": "ok",
    "encryption": "ok",
    "compliance": "ok"
  }
}
```

## 数据模型

### UserSession (用户会话)
| 字段 | 类型 | 描述 |
|-----|------|------|
| id | String | 会话ID |
| session_token | String | JWT令牌 |
| user_id | String | 用户ID |
| tenant_id | String | 租户ID |
| auth_method | String | 认证方法 |
| mfa_verified | Boolean | MFA是否验证 |
| status | String | 状态 (active/expired/revoked) |
| ip_address | String | IP地址 |
| expires_at | DateTime | 过期时间 |

### SecurityAuditLog (审计日志)
| 字段 | 类型 | 描述 |
|-----|------|------|
| id | String | 日志ID |
| event_type | String | 事件类型 |
| event_level | String | 事件级别 |
| user_id | String | 用户ID |
| resource | String | 资源 |
| action | String | 操作 |
| result | String | 结果 |
| risk_score | Float | 风险分数 |
| created_at | DateTime | 创建时间 |

### AccessPolicy (访问策略)
| 字段 | 类型 | 描述 |
|-----|------|------|
| policy_id | String | 策略ID |
| name | String | 策略名称 |
| effect | String | 效果 (allow/deny) |
| principals | List[String] | 主体列表 |
| resources | List[String] | 资源列表 |
| actions | List[String] | 操作列表 |
| conditions | JSON | 条件 |
| priority | Integer | 优先级 |

### EncryptionKey (加密密钥)
| 字段 | 类型 | 描述 |
|-----|------|------|
| key_id | String | 密钥ID |
| name | String | 密钥名称 |
| algorithm | String | 算法 |
| key_size | Integer | 密钥大小 |
| status | String | 状态 |
| expires_at | DateTime | 过期时间 |

### ComplianceReport (合规报告)
| 字段 | 类型 | 描述 |
|-----|------|------|
| report_id | String | 报告ID |
| standard | String | 合规标准 |
| overall_level | String | 整体级别 |
| score | Float | 分数 |
| violations | List[JSON] | 违规列表 |
| recommendations | List[String] | 建议列表 |

## 错误处理

所有API响应格式：

**成功：**
```json
{
  "success": true,
  "data": {...}
}
```

**失败：**
```json
{
  "success": false,
  "error": "Error message"
}
```

## HTTP状态码

| 状态码 | 描述 |
|-------|------|
| 200 | 成功 |
| 201 | 创建成功 |
| 400 | 请求参数错误 |
| 401 | 未认证 |
| 403 | 无权限 |
| 404 | 资源不存在 |
| 500 | 服务器内部错误 |

## 安全注意事项

1. **令牌管理**：JWT令牌默认1小时过期，可通过刷新接口延长
2. **MFA**：支持TOTP方式的双因素认证
3. **会话管理**：支持多设备登录和单点撤销
4. **权限检查**：基于RBAC和ABAC混合模式
5. **审计日志**：所有敏感操作自动记录
6. **加密**：支持AES-256-GCM对称加密
7. **合规性**：支持GDPR、CCPA等主流标准检查


# 数据库模块

## 概述

数据库模块是VectorSphere智能平台的核心数据访问层，提供统一的数据库连接管理、ORM模型定义、数据操作服务等功能。该模块基于SQLAlchemy实现，支持PostgreSQL数据库，提供了完整的数据持久化解决方案。

## 功能特性

### 1. 数据库连接管理
- 统一的数据库连接池管理
- 支持连接配置和环境变量配置
- 会话管理和事务处理
- 健康检查和连接监控

### 2. ORM模型定义
- 基础模型类和混入类
- 认证相关模型（用户、会话、API密钥等）
- 训练相关模型（训练会话、进度跟踪等）
- 监控相关模型（系统指标、告警等）
- 项目相关模型（项目、数据集、模型等）

### 3. 数据操作服务
- CRUD操作封装
- 查询和过滤功能
- 批量操作支持
- 事务管理

### 4. 异常处理
- 完整的数据库异常体系
- 连接异常、查询异常、事务异常等

## 目录结构

```
database/
├── __init__.py              # 模块初始化文件
├── models.py                # 基础模型类
├── enums.py                 # 统一枚举定义
├── config.py                # 数据库配置管理
├── manager.py               # 数据库管理器
├── service.py               # 数据库操作服务
├── auth_models.py           # 认证相关模型
├── training_models.py       # 训练相关模型
├── monitoring_models.py     # 监控相关模型
├── project_models.py        # 项目相关模型
├── exceptions.py            # 数据库异常类
├── api.py                   # 数据库API接口
└── tests/                   # 测试文件
    └── test_database_service.py
```

## 核心组件

### 基础模型类
- `Base`: SQLAlchemy基础类
- `TimestampMixin`: 时间戳混入类
- `UUIDMixin`: UUID主键混入类
- `TenantMixin`: 租户混入类

### 枚举定义
- 训练相关枚举（状态、类型、场景等）
- 监控相关枚举（指标类型、告警级别等）
- 用户相关枚举（状态、角色等）
- 项目相关枚举（状态、类型等）

### 配置管理
- `DatabaseConfig`: 数据库配置类
- 环境变量配置支持
- 连接池配置

### 数据库管理器
- `DatabaseManager`: 数据库管理器类
- 连接池管理
- 会话管理
- 表结构管理

### 数据库服务
- `DatabaseService`: 数据库操作服务类
- CRUD操作封装
- 查询和过滤功能

## 使用示例

### 数据库配置
```python
from backend_new.modules.database.config import DatabaseConfig

# 从环境变量创建配置
config = DatabaseConfig.from_env()

# 手动创建配置
config = DatabaseConfig(
    host="localhost",
    port=5432,
    database="vectorsphere",
    username="vectorsphere",
    password="vectorsphere"
)
```

### 数据库管理器使用
```python
from backend_new.modules.database.manager import get_database_manager

# 获取数据库管理器实例
db_manager = get_database_manager()

# 获取会话
with db_manager.get_db_session() as session:
    # 执行数据库操作
    result = session.execute("SELECT * FROM users")
    
# 健康检查
is_healthy = db_manager.health_check()
```

### 数据库服务使用
```python
from backend_new.modules.database.service import get_database_service
from backend_new.modules.database.auth_models import User

# 获取数据库服务实例
db_service = get_database_service()

# 创建用户
user = User(
    username="testuser",
    email="test@example.com",
    password_hash="hashed_password"
)
created_user = db_service.create(user)

# 查询用户
user = db_service.get_by_id(User, "user-id")

# 更新用户
updated_user = db_service.update(user, {"username": "new_username"})

# 删除用户
db_service.delete(user)

# 过滤用户
users = db_service.filter_by(User, username="testuser")
```

### 模型使用
```python
from backend_new.modules.database.auth_models import User
from backend_new.modules.database.training_models import TrainingSession

# 创建用户实例
user = User(
    username="testuser",
    email="test@example.com",
    password_hash="hashed_password"
)

# 创建训练会话实例
training_session = TrainingSession(
    user_id="user-id",
    name="My Training Session",
    scenario="basic_model",
    method="standard",
    config={"learning_rate": 0.001}
)
```

## API接口

### 健康检查
```
GET /api/v1/database/health
```

### 获取表信息
```
GET /api/v1/database/tables
```

### 获取数据库统计信息
```
GET /api/v1/database/stats
```

## 异常处理

数据库模块定义了多种异常类型，用于处理数据库操作过程中可能出现的错误：

- `DatabaseException`: 基础数据库异常
- `DatabaseConnectionException`: 数据库连接异常
- `DatabaseQueryException`: 数据库查询异常
- `DatabaseTransactionException`: 数据库事务异常
- `DatabaseModelNotFoundException`: 数据库模型未找到异常
- `DatabaseConstraintException`: 数据库约束异常
- `DatabaseConfigurationException`: 数据库配置异常

## 测试

数据库模块包含完整的单元测试，可以通过以下命令运行：

```bash
python -m pytest backend_new/modules/database/tests/
```

## 配置说明

数据库模块的配置可以通过环境变量进行设置：

- `DB_HOST`: 数据库主机地址
- `DB_PORT`: 数据库端口
- `DB_DATABASE`: 数据库名称
- `DB_USERNAME`: 数据库用户名
- `DB_PASSWORD`: 数据库密码
- `DB_POOL_SIZE`: 连接池大小
- `DB_MAX_OVERFLOW`: 最大溢出连接数
- `DB_POOL_TIMEOUT`: 连接池超时时间
- `DB_POOL_RECYCLE`: 连接回收时间
- `DB_ECHO`: 是否输出SQL语句

## 最佳实践

1. **使用上下文管理器**: 在执行数据库操作时使用`get_db_session()`上下文管理器确保会话正确关闭
2. **异常处理**: 妥善处理数据库异常，避免程序崩溃
3. **连接池配置**: 根据应用负载合理配置连接池参数
4. **索引优化**: 为常用查询字段添加数据库索引
5. **事务管理**: 合理使用事务确保数据一致性
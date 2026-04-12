# VectorSphere Intelligent Platform

🚀 **现代化智能训练平台** - 基于多租户架构的分布式机器学习训练平台

## 📋 项目概述

VectorSphere Intelligent Platform 是一个企业级的智能训练平台，专为多租户环境下的大规模机器学习训练而设计。平台采用现代化的微服务架构，支持分布式训练、资源管理、模型版本控制等核心功能。

### 🎯 核心特性

- **🏢 多租户架构**: 完整的租户隔离和资源管理
- **🔄 分布式训练**: 支持大规模分布式机器学习训练
- **🤖 多提供者 LLM**: 集成本地模型、ChatGPT、DeepSeek 等多种语言模型
- **💬 智能对话**: 基于 LangChain 的智能训练助手和对话系统
- **📊 实时监控**: 全面的系统监控和性能指标
- **🔐 安全认证**: JWT认证和基于角色的访问控制
- **📁 文件管理**: 完整的文件上传、下载和管理功能
- **🎛️ 资源调度**: 智能的资源分配和任务调度
- **📈 可视化界面**: 直观的训练过程监控和管理界面

## 🏗️ 技术架构

### 架构层次

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                           │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────────────┐ │
│  │  Auth   │Training │ Models  │ Files   │    Admin        │ │
│  └─────────┴─────────┴─────────┴─────────┴─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                      Service Layer                         │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────────────┐ │
│  │Distributed│Queue  │Monitoring│Cluster │    Storage      │ │
│  └─────────┴─────────┴─────────┴─────────┴─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                       Model Layer                          │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────────────┐ │
│  │  User   │Training │ Models  │ Files   │    Tenants      │ │
│  └─────────┴─────────┴─────────┴─────────┴─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Database Layer                          │
│  ┌─────────────────┬─────────────────┬─────────────────────┐ │
│  │   PostgreSQL    │      Redis      │    File Storage     │ │
│  └─────────────────┴─────────────────┴─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

- **后端框架**: Flask + SQLAlchemy
- **数据库**: PostgreSQL + Redis
- **LLM 集成**: LangChain + OpenAI + DeepSeek + Ollama
- **任务队列**: Celery + RabbitMQ
- **监控**: Prometheus + Grafana
- **容器化**: Docker + Docker Compose
- **认证**: JWT + Flask-JWT-Extended
- **API文档**: Flask-RESTX (Swagger)

## 🚀 快速开始

### 环境要求

- Python 3.9+
- Docker & Docker Compose
- PostgreSQL 14+
- Redis 7+

### 安装步骤

#### 1. 克隆项目

```bash
git clone <repository-url>
cd VectorSphere-intelligent-platform
```

#### 2. 环境配置

```bash
# 复制环境变量配置文件
cp .env.example .env

# 编辑配置文件
vim .env
```

#### 3. 使用Docker Compose启动（推荐）

```bash
# 启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f app
```

#### 4. 本地开发环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 初始化数据库
flask db upgrade

# 启动开发服务器
flask run --host=0.0.0.0 --port=8080
```

### 🔧 配置说明

主要配置项说明（详见 `.env.example`）：

```bash
# 基础配置
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=your-secret-key

# 数据库配置
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vectorsphere
DB_USER=postgres
DB_PASSWORD=password

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# JWT配置
JWT_SECRET_KEY=your-jwt-secret
JWT_ACCESS_TOKEN_EXPIRES=3600

# LLM 配置
OPENAI_API_KEY=your-openai-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
OLLAMA_HOST=localhost
OLLAMA_PORT=11434
```

### 🤖 多提供者 LLM 配置

平台支持多种 LLM 提供者，详细配置请参考 [多提供者 LLM 集成指南](docs/multi_provider_llm_guide.md)：

- **本地模型**: 使用 Ollama 部署的本地模型
- **ChatGPT**: OpenAI 的 GPT 系列模型
- **DeepSeek**: DeepSeek 的专业代码模型

```bash
# 启动 Ollama 服务（本地模型）
ollama serve
ollama pull llama2

# 配置 API 密钥
export OPENAI_API_KEY="your_openai_api_key"
export DEEPSEEK_API_KEY="your_deepseek_api_key"
```

## 📚 API文档

启动服务后，访问以下地址查看API文档：

- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI JSON**: http://localhost:8080/swagger.json

### 主要API端点

| 模块 | 端点 | 描述 |
|------|------|------|
| 认证 | `/api/auth/*` | 用户认证、注册、登录 |
| 训练 | `/api/training/*` | 训练任务管理 |
| 模型 | `/api/models/*` | 模型版本管理 |
| LLM推理 | `/api/inference/*` | 多提供者LLM推理服务 |
| 智能助手 | `/api/assistant/*` | 智能训练助手对话 |
| 文件 | `/api/files/*` | 文件上传下载 |
| 数据集 | `/api/datasets/*` | 数据集管理 |
| 租户 | `/api/tenants/*` | 多租户管理 |
| 监控 | `/api/monitoring/*` | 系统监控 |
| 管理 | `/api/admin/*` | 系统管理 |

## 🔍 监控和日志

### 监控面板

- **Grafana**: http://localhost:3000 (admin/admin)
  - 启用预置：`GRAFANA_ENABLE_PROVISIONING=true`，`GRAFANA_PROVISION_DIR=/etc/grafana/provisioning`
  - 强制重写（触发重载）：`GRAFANA_PROVISION_FORCE=true`
  - 数据源环境变量：
    - Prometheus：`GRAFANA_DS_PROMETHEUS_URL=http://prometheus:9090`，可选 UID：`GRAFANA_DS_PROM_UID`（默认 `prometheus`）
    - InfluxDB：`GRAFANA_DS_INFLUX_URL`、`GRAFANA_DS_INFLUX_ORG`、`GRAFANA_DS_INFLUX_TOKEN`、`GRAFANA_DS_INFLUX_BUCKET`，可选 UID：`GRAFANA_DS_INFLUX_UID`（默认 `influxdb`）
    - TimescaleDB：`GRAFANA_DS_PG_HOST`、`GRAFANA_DS_PG_DB`、`GRAFANA_DS_PG_USER`、`GRAFANA_DS_PG_PASSWORD`、`GRAFANA_DS_PG_PORT`，可选 UID：`GRAFANA_DS_PG_UID`（默认 `timescaledb`）
  - 仪表盘 UID：`GRAFANA_DASHBOARD_UID_TRAINING`（默认 `vectorsphere-training-overview`）
  - 预置结果：生成 `datasources.yaml` 与 `training_overview.json`、`dashboards.yaml`
  - 有状态更新策略：默认“内容未变化不重写”，避免重复导入；设置 `GRAFANA_PROVISION_FORCE=true` 时强制覆写上述文件


分布式训练与容错
- 启动参数：
  - `DIST_RETRY_MAX`（默认 3）
  - `DIST_RETRY_INITIAL_DELAY`（默认 1.0s）
  - `DIST_RETRY_BACKOFF_BASE`（默认 2.0）
  - `DIST_RETRY_JITTER`（默认 0.5）
- 调度器参数：
  - `SCHEDULER_INTERVAL`（默认 5s）
  - `SCHEDULER_MONITOR_INTERVAL`（默认 10s）
  - `TASK_MAX_RETRIES`（默认 3）
  - `TASK_PROGRESS_STALE_SECONDS`（默认 120s）
  - `TASK_MAX_HEALTH_FAILS`（默认 3）
- 行为说明：
  - 启动训练时会为任务尝试创建 `lease`（租约）并在训练循环中发送心跳。Lease 到期会被 `LeaseManager` 检测并触发 `FaultToleranceManager` 回调。
  - API：
    - 训练心跳：POST `/api/v1/training/execution/sessions/<session_id>/heartbeat`  body: `{ "lease_id": "..." }`（优先调用训练 service 中的 handler，否则直接调用 LeaseManager）
    - 节点心跳：POST `/api/v1/nodes/heartbeat`  body: `{ "node_id": "...", "hostname": "...", "gpus": [...], "memory_total": ..., ... }`
  - `TaskScheduler` 启动时会同时运行一个运行态任务监控循环。监控到租约到期或长时间无进度的任务会被标记为 `FAILED` 并上报容错管理器；根据 `TASK_MAX_RETRIES` 会尝试重新入队。
  - `CheckpointManager` 提供检查点创建/恢复/验证能力，容错管理器在故障时会尝试从最新检查点恢复任务。

- **Prometheus**: http://localhost:9090
- **应用健康检查**: http://localhost:8080/health

### 日志管理

```bash
# 查看应用日志
docker-compose logs -f app

# 查看数据库日志
docker-compose logs -f postgres

# 查看Redis日志
docker-compose logs -f redis
```

## 🧪 测试

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_auth.py

# 生成覆盖率报告
pytest --cov=. --cov-report=html
```

### 测试环境

```bash
# 启动测试环境
docker-compose -f docker-compose.test.yml up -d

# 运行集成测试
pytest tests/integration/
```

## 🚀 部署

### 生产环境部署

```bash
# 构建生产镜像
docker build -t vectorsphere-platform:latest .

# 使用生产配置启动
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes部署

```bash
# 应用Kubernetes配置
kubectl apply -f k8s/

# 查看部署状态
kubectl get pods -n vectorsphere
```

## 🛠️ 开发指南

### 项目结构

```
VectorSphere-intelligent-platform/
├── api/                    # API层
│   ├── auth.py            # 认证API
│   ├── training.py        # 训练API
│   ├── models.py          # 模型API
│   └── ...
├── services/              # 服务层
│   ├── distributed.py     # 分布式服务
│   ├── monitoring.py      # 监控服务
│   └── ...
├── models/                # 数据模型
│   ├── user.py           # 用户模型
│   ├── training.py       # 训练模型
│   └── ...
├── middleware/            # 中间件
│   ├── auth.py           # 认证中间件
│   ├── tenant.py         # 租户中间件
│   └── ...
├── utils/                 # 工具函数
├── config.py             # 配置文件
├── app.py                # 应用入口
└── wsgi.py               # WSGI入口
```

### 代码规范

- 遵循 PEP 8 代码风格
- 使用类型注解
- 编写单元测试
- 添加适当的文档字符串

### 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 支持

如果您遇到问题或有建议，请：

1. 查看 [FAQ](docs/FAQ.md)
2. 搜索 [Issues](../../issues)
3. 创建新的 [Issue](../../issues/new)
4. 联系开发团队

## 🔗 相关链接

- [技术架构文档](docs/architecture.md)
- [API参考文档](docs/api.md)
- [部署指南](docs/deployment.md)
- [开发指南](docs/development.md)

## 监控与可观测性使用指南（Prometheus/PushGateway/InfluxDB/TimescaleDB/Alertmanager）

- Prometheus 抓取：访问 `GET /metrics` 暴露指标
- PushGateway（可选）：
  - 环境变量：`ENABLE_PUSHGATEWAY=true`，`PUSHGATEWAY_URL=http://pushgateway:9091`
  - 训练指标记录时自动推送默认注册表
- InfluxDB v2（可选）：
  - 环境变量：`ENABLE_INFLUX=true`，`INFLUX_URL`、`INFLUX_ORG`、`INFLUX_BUCKET`、`INFLUX_TOKEN`
  - 使用 HTTP line protocol 写入 `training_metrics`
- TimescaleDB（可选持久化与生命周期）：
  - 环境变量：
    - `ENABLE_TIMESCALE=true`
    - 连接：`TS_DB_DSN` 或 `TS_DB_HOST`、`TS_DB_USER`、`TS_DB_PASSWORD`、`TS_DB_NAME`、`TS_DB_PORT`
    - 表名：`TS_TABLE`（默认 `training_metrics`）
    - 保留策略：`TS_RETENTION_DAYS`（>0 启用，如 30）
    - 压缩：`TS_ENABLE_COMPRESSION=true`，`TS_COMPRESS_INTERVAL='7 days'`
  - 代码中会幂等创建 hypertable，并按以上配置添加 `add_retention_policy` 与 `add_compression_policy`
- Alertmanager 路由与通知：
  - 环境变量：`ALERTMANAGER_URL=http://alertmanager:9093`
  - 额外路由标签（辅助与路由匹配）：`ALERT_LABEL_TEAM`、`ALERT_LABEL_SERVICE`、`ALERT_LABEL_ENV`
  - 告警触发时将携带上述标签，便于在 Alertmanager `route`/`receivers` 中进行匹配

### Alertmanager 路由示例（alertmanager.yml）

```yaml
route:
  receiver: default
  routes:
    - matchers:
      - severity="critical"
      - team="platform"
      receiver: ops-slack
    - matchers:
      - service="training"
      - environment="prod"
      receiver: training-email

receivers:
  - name: default
    email_configs:
      - to: default@company.com
  - name: ops-slack
    webhook_configs:
      - url: https://hooks.slack.com/services/xxx/yyy/zzz
  - name: training-email
    email_configs:
      - to: training@company.com
```

### 验证步骤
- 启动服务后访问 `/metrics` 与 `/health`
- 设置 `ENABLE_TIMESCALE=true` 并配置数据库，确认写入成功（表 `training_metrics`）
- 配置 `TS_RETENTION_DAYS` 与 `TS_ENABLE_COMPRESSION`，检查 TimescaleDB `add_retention_policy`/`add_compression_policy` 生效
- 设置 `ALERTMANAGER_URL` 和标签变量，触发告警后在 Alertmanager UI 检查路由与通知

---

**VectorSphere Intelligent Platform** - 让机器学习训练更简单、更高效！

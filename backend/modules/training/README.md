# 训练模块

训练模块提供完整的模型训练流水线功能，包括数据准备、模型训练、模型评估、模型优化和模型部署等阶段。

## 功能特性

### 1. 数据准备阶段
- 数据发现与接入
- 数据质量管理
- 数据预处理

### 2. 模型训练阶段
- 模型选择与配置
- 超参数优化
- 训练执行管理

### 3. 模型评估阶段
- 自动化评估
- 模型对比与选择

### 4. 模型优化阶段
- 模型压缩
- 推理优化

### 5. 模型部署阶段
- 部署策略
- 服务化封装

### 6. 监控运维阶段
- 性能监控
- 运维自动化

### 7. 智能化特性
- AI驱动的自动化
- 知识图谱驱动

## 模块结构

```
training/
├── api/                    # API接口
├── config/                 # 配置文件
├── core/                   # 核心组件
├── database/               # 数据库相关
├── distributed/            # 分布式训练
├── exceptions/             # 异常定义
├── launcher/               # 训练启动器
├── monitoring/             # 监控组件
├── multimodal/             # 多模态支持
├── optimization/           # 优化组件
├── progress/               # 进度管理
├── pt_sft_dpo/             # PT/SFT/DPO训练
├── repositories/           # 数据访问层
├── scenarios/              # 训练场景
├── scheduler/              # 调度器
├── schemas/                # 数据模式
├── services/               # 业务服务
├── tests/                  # 测试文件
├── three_stage/            # 三阶段训练
└── utils/                  # 工具函数
```

## API接口

### 训练管理API
- `POST /api/training/jobs` - 创建训练任务
- `GET /api/training/jobs` - 获取训练任务列表
- `GET /api/training/jobs/<job_id>` - 获取训练任务详情
- `PUT /api/training/jobs/<job_id>` - 更新训练任务
- `DELETE /api/training/jobs/<job_id>` - 删除训练任务

### 训练进度API
- `GET /api/training/progress/<session_id>` - 获取训练进度
- `POST /api/training/progress/<session_id>/pause` - 暂停训练
- `POST /api/training/progress/<session_id>/resume` - 恢复训练

### 三阶段训练API
- `POST /api/training/three-stage/init` - 初始化三阶段训练
- `POST /api/training/three-stage/pt` - 预训练阶段
- `POST /api/training/three-stage/sft` - 有监督微调阶段
- `POST /api/training/three-stage/dpo` - 偏好优化阶段

### 超参数优化API
- `POST /api/training/hyperparameter/optimize` - 执行超参数优化
- `GET /api/training/hyperparameter/studies` - 获取优化研究列表
- `GET /api/training/hyperparameter/studies/<study_id>` - 获取优化研究详情

### 模型选择API
- `POST /api/training/models/recommend` - 推荐模型
- `GET /api/training/models/configurations` - 获取模型配置

### 训练执行API
- `POST /api/training/sessions/<session_id>/start` - 启动训练
- `POST /api/training/sessions/<session_id>/pause` - 暂停训练
- `POST /api/training/sessions/<session_id>/resume` - 恢复训练
- `POST /api/training/sessions/<session_id>/stop` - 停止训练

### 模型评估API
- `POST /api/training/evaluation/models/<model_id>/evaluate` - 评估模型
- `POST /api/training/evaluation/models/compare` - 对比模型
- `GET /api/training/evaluation/metrics/types` - 获取评估指标类型
- `GET /api/training/evaluation/models/<model_id>/history` - 获取评估历史

### 模型优化API
- `POST /api/training/optimization/models/<model_id>/compress` - 模型压缩
- `POST /api/training/optimization/models/<model_id>/optimize-inference` - 推理优化
- `POST /api/training/optimization/models/<model_id>/auto-optimize` - 自动优化
- `GET /api/training/optimization/techniques` - 获取优化技术类型
- `GET /api/training/optimization/models/<model_id>/history` - 获取优化历史

### 模型部署API
- `POST /api/training/deployment/models/<model_id>/deploy` - 部署模型
- `POST /api/training/deployment/models/<model_id>/service` - 服务化封装模型
- `POST /api/training/deployment/deployments/<deployment_id>/undeploy` - 取消部署模型
- `GET /api/training/deployment/deployments/<deployment_id>/status` - 获取部署状态
- `POST /api/training/deployment/deployments/<deployment_id>/scale` - 扩缩容部署
- `GET /api/training/deployment/modes` - 获取部署模式和发布策略
- `GET /api/training/deployment/models/<model_id>/history` - 获取部署历史

### 监控运维API
- `GET /api/training/monitoring/deployments/<deployment_id>/metrics` - 获取性能指标
- `POST /api/training/monitoring/alerts/rules` - 创建告警规则
- `GET /api/training/monitoring/deployments/<deployment_id>/alerts` - 检查告警
- `POST /api/training/monitoring/deployments/<deployment_id>/automation` - 执行自动化任务
- `GET /api/training/monitoring/deployments/<deployment_id>/analytics` - 获取部署分析数据
- `GET /api/training/monitoring/metrics/types` - 获取监控指标类型
- `GET /api/training/monitoring/deployments/<deployment_id>/alerts/history` - 获取告警历史
- `GET /api/training/monitoring/automation/tasks/<task_id>` - 获取自动化任务状态

### 智能化决策API
- `POST /api/training/intelligent/decisions` - 智能决策
- `POST /api/training/intelligent/optimization/adaptive` - 自适应优化
- `POST /api/training/intelligent/knowledge/graph` - 获取知识图谱
- `POST /api/training/intelligent/knowledge/base` - 更新知识库
- `POST /api/training/intelligent/experience/accumulate` - 积累经验
- `GET /api/training/intelligent/scenarios` - 获取决策场景和算法
- `GET /api/training/intelligent/decisions/history` - 获取决策历史
- `GET /api/training/intelligent/optimization/history` - 获取优化历史

## 使用示例

### 创建训练任务
```bash
curl -X POST http://localhost:8080/api/training/jobs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "model_id": "model_123",
    "dataset_id": "dataset_456",
    "config": {
      "epochs": 10,
      "batch_size": 32,
      "learning_rate": 0.001
    }
  }'
```

### 启动训练
```bash
curl -X POST http://localhost:8080/api/training/sessions/session_789/start \
  -H "Authorization: Bearer <token>"
```

### 评估模型
```bash
curl -X POST http://localhost:8080/api/training/evaluation/models/model_123/evaluate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "dataset_id": "dataset_456",
    "evaluation_config": {
      "metrics": ["accuracy", "precision", "recall"]
    }
  }'
```

### 对比模型
```bash
curl -X POST http://localhost:8080/api/training/evaluation/models/compare \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "model_ids": ["model_123", "model_456"],
    "dataset_id": "dataset_789",
    "comparison_config": {
      "comparison_metrics": ["accuracy", "f1_score"]
    }
  }'
```

### 模型压缩
```bash
curl -X POST http://localhost:8080/api/training/optimization/models/model_123/compress \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "technique": "pruning",
    "compression_ratio": 0.6,
    "strategy": "unstructured"
  }'
```

### 推理优化
```bash
curl -X POST http://localhost:8080/api/training/optimization/models/model_123/optimize-inference \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "graph_optimization": true,
    "operator_fusion": true,
    "constant_folding": true,
    "memory_optimization": true,
    "hardware_target": "cpu"
  }'
```

### 部署模型
```bash
curl -X POST http://localhost:8080/api/training/deployment/models/model_123/deploy \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "mode": "online",
    "release_strategy": "rolling",
    "replicas": 3,
    "resources": {
      "cpu": "2",
      "memory": "4Gi"
    }
  }'
```

### 服务化封装
```bash
curl -X POST http://localhost:8080/api/training/deployment/models/model_123/service \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "api_type": "rest",
    "load_balancing": true,
    "circuit_breaker": true,
    "rate_limiting": true,
    "timeout": 30
  }'
```

### 监控性能指标
```bash
curl -X GET "http://localhost:8080/api/training/monitoring/deployments/deploy_123/metrics?metric_types=qps,response_time" \
  -H "Authorization: Bearer <token>"
```

### 创建告警规则
```bash
curl -X POST http://localhost:8080/api/training/monitoring/alerts/rules \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "name": "high_error_rate",
    "metric_type": "error_rate",
    "threshold": 0.05,
    "operator": ">",
    "severity": "warning",
    "duration": 300
  }'
```

### 智能决策
```bash
curl -X POST http://localhost:8080/api/training/intelligent/decisions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "scenario": "model_architecture",
    "inputs": {
      "task_type": "classification",
      "data_characteristics": {
        "size": 10000,
        "features": 100
      }
    }
  }'
```

### 自适应优化
```bash
curl -X POST http://localhost:8080/api/training/intelligent/optimization/adaptive \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "parameter_name": "learning_rate",
    "current_value": 0.001,
    "adjustment_strategy": "gradient_based",
    "monitoring_metrics": ["loss", "accuracy"]
  }'
```

## 服务类

### ModelEvaluationService
提供模型评估和对比功能。

主要方法：
- `automated_evaluation()` - 自动化模型评估
- `model_comparison()` - 模型对比与选择

### ModelOptimizationService
提供模型压缩和推理优化功能。

主要方法：
- `model_compression()` - 模型压缩
- `inference_optimization()` - 推理优化
- `auto_optimization()` - 自动优化

### ModelDeploymentService
提供模型部署策略和服务化封装功能。

主要方法：
- `deploy_model()` - 部署模型
- `service_model()` - 服务化封装
- `undeploy_model()` - 取消部署模型
- `get_deployment_status()` - 获取部署状态
- `scale_deployment()` - 扩缩容部署

### MonitoringOperationsService
提供性能监控和运维自动化功能。

主要方法：
- `collect_performance_metrics()` - 收集性能指标
- `create_alert_rule()` - 创建告警规则
- `check_alerts()` - 检查告警
- `execute_automation_task()` - 执行自动化任务
- `get_deployment_analytics()` - 获取部署分析数据

### IntelligentDecisionService
提供AI驱动的自动化和知识图谱驱动功能。

主要方法：
- `make_intelligent_decision()` - 智能决策
- `adaptive_optimization()` - 自适应优化
- `update_knowledge_base()` - 更新知识库
- `get_knowledge_graph()` - 获取知识图谱
- `experience_accumulation()` - 经验积累

## 配置说明

训练模块的配置位于 `config/` 目录下，包括：
- 训练超参数配置
- 分布式训练配置
- 资源调度配置
- 监控告警配置
- 智能化决策配置

## 测试

运行训练模块的测试：
```bash
python -m pytest backend/modules/training/tests/
```
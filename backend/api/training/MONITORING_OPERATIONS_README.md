# 监控运维服务使用示例

本文档提供 `MonitoringOperationsService` 的使用示例，包括性能指标收集、告警规则管理、自动化任务执行和监控报告生成。

## 目录

1. [服务初始化](#1-服务初始化)
2. [性能指标收集](#2-性能指标收集)
3. [告警规则管理](#3-告警规则管理)
4. [告警检查与处理](#4-告警检查与处理)
5. [自动化任务执行](#5-自动化任务执行)
6. [监控报告与分析](#6-监控报告与分析)
7. [API端点参考](#7-api端点参考)

---

## 1. 服务初始化

### Python 代码示例

```python
import uuid
from backend.services.monitoring_operations_service import (
    get_monitoring_operations_service,
    MonitoringMetricType,
    AlertSeverity,
    AlertRule
)

# 创建服务实例（使用内存存储进行测试）
service = get_monitoring_operations_service(use_memory_storage=True)
print('✅ MonitoringOperationsService created')

# 生成测试用的租户ID和用户ID
tenant_id = str(uuid.uuid4())
user_id = str(uuid.uuid4())
deployment_id = f"deploy_{uuid.uuid4().hex[:8]}"

print(f"Using tenant_id: {tenant_id[:8]}...")
print(f"Using deployment_id: {deployment_id}")
```

---

## 2. 性能指标收集

### 收集部署的性能指标

```python
print('\n=== Testing collect_performance_metrics() ===')

# 收集所有类型的性能指标
report = service.collect_performance_metrics(
    deployment_id=deployment_id,
    metric_types=None,  # None表示收集所有指标
    tenant_id=tenant_id,
    save_to_db=True
)

print(f'Deployment: {report.deployment_id}')
print(f'Metrics collected: {len(report.metrics)}')
for metric in report.metrics[:5]:
    print(f'  - {metric.metric_type.value}: {metric.value:.2f}')
print(f'Recommendations: {report.recommendations[:2]}')
```

### 获取历史指标数据

```python
print('\n=== Testing get_metrics_history() ===')

from datetime import datetime, timedelta

result = service.get_metrics_history(
    tenant_id=tenant_id,
    deployment_id=deployment_id,
    metric_types=['cpu_usage', 'memory_usage'],
    start_time=datetime.utcnow() - timedelta(hours=1),
    end_time=datetime.utcnow(),
    limit=10
)

print(f'Found {result["total"]} metric records')
for m in result['metrics'][:3]:
    print(f'  - {m.get("metric_type")}: {m.get("value"):.2f}')
```

### 获取最新指标值

```python
print('\n=== Testing get_latest_metrics() ===')

latest = service.get_latest_metrics(
    tenant_id=tenant_id,
    deployment_id=deployment_id,
    metric_types=['cpu_usage', 'memory_usage', 'response_time']
)

print(f'Latest metrics:')
for metric_type, data in latest.items():
    print(f'  - {metric_type}: {data.get("value", 0):.2f}')
```

---

## 3. 告警规则管理

### 创建告警规则

```python
print('\n=== Testing create_alert_rule() ===')

# 创建CPU使用率告警规则
cpu_rule = AlertRule(
    name='High CPU Usage Alert',
    metric_type=MonitoringMetricType.CPU_USAGE,
    threshold=80.0,
    operator='>',
    severity=AlertSeverity.WARNING,
    duration=60,
    enabled=True,
    description='Alert when CPU usage exceeds 80%'
)

result = service.create_alert_rule(
    rule=cpu_rule,
    tenant_id=tenant_id,
    user_id=user_id,
    deployment_id=deployment_id
)

print(f'Rule created: {result.get("name")}')
print(f'Rule ID: {result.get("rule_id")}')
rule_id = result.get('rule_id')

# 创建内存使用率告警规则
memory_rule = AlertRule(
    name='High Memory Usage Alert',
    metric_type=MonitoringMetricType.MEMORY_USAGE,
    threshold=85.0,
    operator='>=',
    severity=AlertSeverity.ERROR,
    duration=120,
    enabled=True,
    description='Alert when memory usage exceeds 85%'
)

service.create_alert_rule(
    rule=memory_rule,
    tenant_id=tenant_id,
    user_id=user_id,
    deployment_id=deployment_id
)
```

### 获取告警规则列表

```python
print('\n=== Testing get_alert_rules_list() ===')

rules_result = service.get_alert_rules_list(
    tenant_id=tenant_id,
    enabled_only=True,
    deployment_id=deployment_id,
    limit=10
)

print(f'Found {rules_result["total"]} rules')
for rule in rules_result['rules']:
    print(f'  - {rule.get("name")}: {rule.get("metric_type")} {rule.get("operator")} {rule.get("threshold")}')
```

### 更新告警规则

```python
print('\n=== Testing update_alert_rule() ===')

if rule_id:
    updated_rule = service.update_alert_rule(
        rule_id=rule_id,
        tenant_id=tenant_id,
        updates={
            'threshold': 75.0,
            'description': 'Updated: Alert when CPU usage exceeds 75%'
        }
    )
    
    if updated_rule:
        print(f'Rule updated: threshold={updated_rule.get("threshold")}')
```

### 删除告警规则

```python
print('\n=== Testing delete_alert_rule() ===')

if rule_id:
    success = service.delete_alert_rule(rule_id, tenant_id)
    print(f'Rule deleted: {success}')
```

---

## 4. 告警检查与处理

### 检查告警

```python
print('\n=== Testing check_alerts() ===')

# 先创建一个低阈值的规则以确保触发
test_rule = AlertRule(
    name='Test Alert Rule',
    metric_type=MonitoringMetricType.CPU_USAGE,
    threshold=1.0,  # 极低阈值，几乎肯定会触发
    operator='>',
    severity=AlertSeverity.INFO,
    duration=0,
    enabled=True
)

service.create_alert_rule(
    rule=test_rule,
    tenant_id=tenant_id,
    user_id=user_id,
    deployment_id=deployment_id
)

# 执行告警检查
alerts = service.check_alerts(
    deployment_id=deployment_id,
    tenant_id=tenant_id
)

print(f'Found {len(alerts)} alerts')
for alert in alerts:
    print(f'  - [{alert.severity.value}] {alert.message}')
```

### 获取告警历史

```python
print('\n=== Testing get_alert_history_list() ===')

history_result = service.get_alert_history_list(
    tenant_id=tenant_id,
    deployment_id=deployment_id,
    severity=None,
    resolved=None,
    limit=10
)

print(f'Found {history_result["total"]} alert history records')
for h in history_result['alerts'][:3]:
    print(f'  - [{h.get("severity")}] {h.get("message")[:50]}...')
```

### 确认和解决告警

```python
print('\n=== Testing acknowledge_alert() and resolve_alert() ===')

# 获取一个未解决的告警
if history_result['alerts']:
    alert_id = history_result['alerts'][0].get('alert_id')
    
    # 确认告警
    ack_result = service.acknowledge_alert(
        alert_id=alert_id,
        tenant_id=tenant_id,
        user_id=user_id
    )
    if ack_result:
        print(f'Alert acknowledged: {alert_id}')
    
    # 解决告警
    resolve_result = service.resolve_alert_by_id(
        alert_id=alert_id,
        tenant_id=tenant_id,
        resolution_notes='Issue resolved by scaling up resources'
    )
    if resolve_result:
        print(f'Alert resolved: {alert_id}')
```

### 获取告警统计

```python
print('\n=== Testing get_alert_statistics() ===')

stats = service.get_alert_statistics(
    tenant_id=tenant_id,
    deployment_id=deployment_id
)

print(f'Alert Statistics:')
print(f'  Total: {stats.get("total", 0)}')
print(f'  Resolved: {stats.get("resolved", 0)}')
print(f'  Unresolved: {stats.get("unresolved", 0)}')
print(f'  By Severity: {stats.get("by_severity", {})}')
```

---

## 5. 自动化任务执行

### 执行自动扩缩容任务

```python
print('\n=== Testing execute_automation_task() - Auto Scaling ===')

task = service.execute_automation_task(
    deployment_id=deployment_id,
    task_type='auto_scaling',
    parameters={
        'current_replicas': 2,
        'target_replicas': 4,
        'reason': 'High CPU usage detected'
    },
    tenant_id=tenant_id,
    user_id=user_id
)

print(f'Task ID: {task.task_id}')
print(f'Status: {task.status}')
print(f'Result: {task.result}')
```

### 执行故障恢复任务

```python
print('\n=== Testing execute_automation_task() - Fault Recovery ===')

task = service.execute_automation_task(
    deployment_id=deployment_id,
    task_type='fault_recovery',
    parameters={
        'component': 'inference_service',
        'action': 'restart'
    },
    tenant_id=tenant_id,
    user_id=user_id
)

print(f'Task ID: {task.task_id}')
print(f'Status: {task.status}')
```

### 获取任务列表

```python
print('\n=== Testing get_task_list() ===')

tasks_result = service.get_task_list(
    tenant_id=tenant_id,
    status=None,  # 所有状态
    task_type=None,  # 所有类型
    deployment_id=deployment_id,
    limit=10
)

print(f'Found {tasks_result["total"]} tasks')
for t in tasks_result['tasks']:
    print(f'  - [{t.get("status")}] {t.get("task_type")}: {t.get("name")}')
```

### 获取任务统计

```python
print('\n=== Testing get_task_statistics() ===')

task_stats = service.get_task_statistics(tenant_id, deployment_id)

print(f'Task Statistics:')
print(f'  Total: {task_stats.get("total", 0)}')
print(f'  By Status: {task_stats.get("by_status", {})}')
print(f'  By Type: {task_stats.get("by_type", {})}')
print(f'  Success Rate: {task_stats.get("success_rate", 0):.2%}')
```

---

## 6. 监控报告与分析

### 获取部署分析数据

```python
print('\n=== Testing get_deployment_analytics() ===')

from datetime import datetime, timedelta

time_range = {
    'start': datetime.utcnow() - timedelta(days=7),
    'end': datetime.utcnow()
}

analytics = service.get_deployment_analytics(
    deployment_id=deployment_id,
    time_range=time_range,
    tenant_id=tenant_id,
    save_report=True
)

print(f'Analytics for deployment: {analytics.get("deployment_id")}')
print(f'Time Range: {analytics.get("time_range")}')
print(f'Capacity Analysis: {analytics.get("capacity_analysis")}')
print(f'Cost Analysis: {analytics.get("cost_analysis")}')
print(f'Recommendations: {analytics.get("recommendations")[:2]}')
print(f'Report ID: {analytics.get("report_id")}')
```

### 获取监控报告列表

```python
print('\n=== Testing get_report_list() ===')

reports_result = service.get_report_list(
    tenant_id=tenant_id,
    report_type=None,
    deployment_id=deployment_id,
    limit=10
)

print(f'Found {reports_result["total"]} reports')
for r in reports_result['reports']:
    print(f'  - [{r.get("report_type")}] {r.get("name")} ({r.get("status")})')
```

### 获取指标统计

```python
print('\n=== Testing get_metrics_statistics() ===')

metric_stats = service.get_metrics_statistics(
    tenant_id=tenant_id,
    deployment_id=deployment_id
)

print(f'Metrics Statistics:')
print(f'  Total Records: {metric_stats.get("total_records", 0)}')
print(f'  Deployments: {metric_stats.get("deployments", [])}')
print(f'  Metric Types:')
for m_type, stats in metric_stats.get('metric_types', {}).items():
    print(f'    - {m_type}: avg={stats.get("avg", 0):.2f}, min={stats.get("min", 0):.2f}, max={stats.get("max", 0):.2f}')
```

---

## 7. API端点参考

### 性能指标

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/api/v1/training/monitoring/deployments/<id>/metrics` | 收集性能指标 |
| GET | `/api/v1/training/monitoring/deployments/<id>/metrics/history` | 获取历史指标 |
| GET | `/api/v1/training/monitoring/deployments/<id>/metrics/latest` | 获取最新指标 |
| GET | `/api/v1/training/monitoring/metrics/statistics` | 获取指标统计 |

### 告警规则

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/training/monitoring/alerts/rules` | 创建告警规则 |
| GET | `/api/v1/training/monitoring/alerts/rules` | 获取规则列表 |
| GET | `/api/v1/training/monitoring/alerts/rules/<id>` | 获取指定规则 |
| PUT | `/api/v1/training/monitoring/alerts/rules/<id>` | 更新告警规则 |
| DELETE | `/api/v1/training/monitoring/alerts/rules/<id>` | 删除告警规则 |

### 告警管理

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/api/v1/training/monitoring/deployments/<id>/alerts` | 检查/获取告警 |
| POST | `/api/v1/training/monitoring/deployments/<id>/alerts/<alert_id>/resolve` | 解决告警 |
| POST | `/api/v1/training/monitoring/deployments/<id>/alerts/<alert_id>/acknowledge` | 确认告警 |
| GET | `/api/v1/training/monitoring/alerts/statistics` | 获取告警统计 |

### 自动化任务

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/training/monitoring/deployments/<id>/automation` | 执行自动化任务 |
| GET | `/api/v1/training/monitoring/automation/tasks` | 获取任务列表 |
| GET | `/api/v1/training/monitoring/automation/tasks/<id>` | 获取任务状态 |
| GET | `/api/v1/training/monitoring/automation/tasks/statistics` | 获取任务统计 |

### 监控报告

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/api/v1/training/monitoring/deployments/<id>/analytics` | 获取分析数据 |
| GET | `/api/v1/training/monitoring/reports` | 获取报告列表 |
| GET | `/api/v1/training/monitoring/reports/<id>` | 获取指定报告 |

### 数据清理

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/v1/training/monitoring/cleanup` | 清理过期数据 |

---

## 任务类型说明

支持的自动化任务类型：

| 任务类型 | 描述 | 参数示例 |
|----------|------|----------|
| `auto_scaling` | 自动扩缩容 | `{"current_replicas": 2, "target_replicas": 4}` |
| `fault_recovery` | 故障恢复 | `{"component": "service_name", "action": "restart"}` |
| `capacity_planning` | 容量规划 | `{"deployment_id": "..."}` |
| `resource_optimization` | 资源优化 | `{"optimization_type": "cpu"}` |
| `alert_management` | 告警管理 | `{"action": "check", "deployment_id": "..."}` |

---

## 告警严重程度

| 级别 | 描述 |
|------|------|
| `info` | 信息性告警 |
| `warning` | 警告，需要关注 |
| `error` | 错误，需要处理 |
| `critical` | 严重，需要立即处理 |

---

## 结论

本示例展示了 `MonitoringOperationsService` 的核心功能。该服务提供了完整的监控运维能力，包括：

- **性能指标收集**：支持多种指标类型，自动持久化存储
- **告警规则管理**：灵活的规则配置，支持租户级别隔离
- **告警检查与处理**：自动检测异常，支持确认和解决流程
- **自动化任务**：多种任务类型，支持自动触发和手动执行
- **监控报告**：综合分析，包括趋势、容量和成本分析

所有操作都支持租户级别的数据隔离，确保多租户环境下的数据安全。


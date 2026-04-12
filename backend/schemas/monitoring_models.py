"""监控相关数据模型

定义系统指标、告警等监控相关数据模型。
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from backend.schemas.base_models import Base, UUIDMixin, TimestampMixin, TenantMixin
from .enums import MetricType, AlertLevel, AlertStatus, ResourceType, MonitoringTarget


class SystemMetric(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """系统指标模型"""
    __tablename__ = 'system_metrics'
    
    name = Column(String(100), nullable=False, index=True, comment="指标名称")
    type = Column(String(20), nullable=False, index=True, comment="指标类型")
    value = Column(Float, nullable=False, comment="指标值")
    unit = Column(String(20), comment="单位")
    source = Column(String(100), index=True, comment="数据源")
    target = Column(String(50), index=True, comment="监控目标")
    tags = Column(JSON, comment="标签")
    description = Column(Text, comment="描述")
    
    def __repr__(self):
        return f"<SystemMetric(id='{self.id}', name='{self.name}', value={self.value})>"


class Alert(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """告警模型"""
    __tablename__ = 'alerts'
    
    name = Column(String(200), nullable=False, comment="告警名称")
    level = Column(String(20), nullable=False, index=True, comment="告警级别")
    status = Column(String(20), default=AlertStatus.ACTIVE.value, index=True, comment="告警状态")
    source = Column(String(100), index=True, comment="告警源")
    target = Column(String(50), index=True, comment="告警目标")
    metric_name = Column(String(100), index=True, comment="指标名称")
    threshold = Column(Float, comment="阈值")
    current_value = Column(Float, comment="当前值")
    description = Column(Text, comment="告警描述")
    triggered_at = Column(DateTime, nullable=False, comment="触发时间")
    resolved_at = Column(DateTime, comment="解决时间")
    acknowledged_at = Column(DateTime, comment="确认时间")
    acknowledged_by = Column(String(36), comment="确认人ID")
    resolution_notes = Column(Text, comment="解决说明")
    
    def __repr__(self):
        return f"<Alert(id='{self.id}', name='{self.name}', level='{self.level}', status='{self.status}')>"


class SystemLog(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """系统日志模型"""
    __tablename__ = 'system_logs'
    
    level = Column(String(20), nullable=False, index=True, comment="日志级别")
    source = Column(String(100), index=True, comment="日志源")
    message = Column(Text, nullable=False, comment="日志消息")
    module = Column(String(100), index=True, comment="模块")
    function = Column(String(100), comment="函数")
    line_number = Column(Integer, comment="行号")
    context = Column(JSON, comment="上下文信息")
    trace_id = Column(String(36), index=True, comment="追踪ID")
    
    def __repr__(self):
        return f"<SystemLog(id='{self.id}', level='{self.level}', source='{self.source}')>"


# ==============================================================================
# 性能指标记录模型
# ==============================================================================

class PerformanceMetricRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """性能指标记录模型
    
    用于持久化存储部署的性能指标数据
    """
    __tablename__ = 'performance_metric_records'
    
    metric_id = Column(String(100), unique=True, nullable=False, index=True, comment="指标ID")
    deployment_id = Column(String(100), nullable=False, index=True, comment="部署ID")
    metric_type = Column(String(50), nullable=False, index=True, comment="指标类型 (qps, response_time, cpu_usage等)")
    value = Column(Float, nullable=False, comment="指标值")
    unit = Column(String(20), comment="单位")
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow, comment="采集时间")
    
    # 元数据
    source = Column(String(100), comment="数据来源 (prometheus, system, simulated)")
    tags = Column(JSON, comment="标签")
    metadata_ = Column(JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<PerformanceMetricRecord(id='{self.id}', deployment_id='{self.deployment_id}', type='{self.metric_type}', value={self.value})>"
    
    def to_dict(self):
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'metric_id': self.metric_id,
            'deployment_id': self.deployment_id,
            'metric_type': self.metric_type,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'source': self.source,
            'tags': self.tags,
            'metadata': self.metadata_,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# ==============================================================================
# 告警规则模型
# ==============================================================================

class AlertRuleRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """告警规则记录模型
    
    用于持久化存储告警规则配置
    """
    __tablename__ = 'alert_rule_records'
    
    rule_id = Column(String(100), unique=True, nullable=False, index=True, comment="规则ID")
    name = Column(String(200), nullable=False, comment="规则名称")
    description = Column(Text, comment="规则描述")
    
    # 规则配置
    metric_type = Column(String(50), nullable=False, index=True, comment="监控指标类型")
    threshold = Column(Float, nullable=False, comment="阈值")
    operator = Column(String(10), nullable=False, comment="操作符 (>, <, >=, <=, ==, !=)")
    severity = Column(String(20), nullable=False, index=True, comment="严重程度 (info, warning, error, critical)")
    duration = Column(Integer, default=0, comment="持续时间(秒)，超过此时间才触发")
    
    # 适用范围
    deployment_id = Column(String(100), index=True, comment="绑定的部署ID，为空表示全局规则")
    scope = Column(String(50), default='deployment', comment="作用范围 (deployment, tenant, global)")
    
    # 状态
    enabled = Column(Boolean, default=True, index=True, comment="是否启用")
    
    # 通知配置
    notification_channels = Column(JSON, comment="通知渠道配置 (email, webhook, slack等)")
    cooldown_seconds = Column(Integer, default=300, comment="告警冷却时间(秒)")
    
    # 统计
    trigger_count = Column(Integer, default=0, comment="触发次数")
    last_triggered_at = Column(DateTime, comment="上次触发时间")
    
    # 创建者
    created_by = Column(String(36), index=True, comment="创建者用户ID")
    
    def __repr__(self):
        return f"<AlertRuleRecord(id='{self.id}', name='{self.name}', metric_type='{self.metric_type}')>"
    
    def to_dict(self):
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'rule_id': self.rule_id,
            'name': self.name,
            'description': self.description,
            'metric_type': self.metric_type,
            'threshold': self.threshold,
            'operator': self.operator,
            'severity': self.severity,
            'duration': self.duration,
            'deployment_id': self.deployment_id,
            'scope': self.scope,
            'enabled': self.enabled,
            'notification_channels': self.notification_channels,
            'cooldown_seconds': self.cooldown_seconds,
            'trigger_count': self.trigger_count,
            'last_triggered_at': self.last_triggered_at.isoformat() if self.last_triggered_at else None,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# ==============================================================================
# 告警记录模型
# ==============================================================================

class AlertHistoryRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """告警历史记录模型
    
    用于持久化存储已触发的告警历史
    """
    __tablename__ = 'alert_history_records'
    
    alert_id = Column(String(100), unique=True, nullable=False, index=True, comment="告警ID")
    rule_id = Column(String(100), index=True, comment="触发的规则ID")
    rule_name = Column(String(200), comment="规则名称")
    deployment_id = Column(String(100), nullable=False, index=True, comment="部署ID")
    
    # 告警信息
    severity = Column(String(20), nullable=False, index=True, comment="严重程度")
    message = Column(Text, nullable=False, comment="告警消息")
    metric_type = Column(String(50), index=True, comment="指标类型")
    metric_value = Column(Float, comment="触发时的指标值")
    threshold = Column(Float, comment="阈值")
    
    # 时间信息
    triggered_at = Column(DateTime, nullable=False, index=True, comment="触发时间")
    resolved_at = Column(DateTime, comment="解决时间")
    acknowledged_at = Column(DateTime, comment="确认时间")
    
    # 处理信息
    resolved = Column(Boolean, default=False, index=True, comment="是否已解决")
    acknowledged = Column(Boolean, default=False, comment="是否已确认")
    acknowledged_by = Column(String(36), comment="确认人ID")
    resolution_notes = Column(Text, comment="解决说明")
    
    # 通知状态
    notification_sent = Column(Boolean, default=False, comment="是否已发送通知")
    notification_channels_used = Column(JSON, comment="已使用的通知渠道")
    
    # 元数据
    context = Column(JSON, comment="上下文信息")
    metadata_ = Column(JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<AlertHistoryRecord(id='{self.id}', alert_id='{self.alert_id}', severity='{self.severity}', resolved={self.resolved})>"
    
    def to_dict(self):
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'alert_id': self.alert_id,
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'deployment_id': self.deployment_id,
            'severity': self.severity,
            'message': self.message,
            'metric_type': self.metric_type,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved': self.resolved,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'resolution_notes': self.resolution_notes,
            'notification_sent': self.notification_sent,
            'notification_channels_used': self.notification_channels_used,
            'context': self.context,
            'metadata': self.metadata_,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# ==============================================================================
# 自动化任务记录模型
# ==============================================================================

class AutomationTaskRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """自动化任务记录模型
    
    用于持久化存储自动化运维任务
    """
    __tablename__ = 'automation_task_records'
    
    task_id = Column(String(100), unique=True, nullable=False, index=True, comment="任务ID")
    name = Column(String(200), nullable=False, comment="任务名称")
    description = Column(Text, comment="任务描述")
    
    # 任务类型和状态
    task_type = Column(String(50), nullable=False, index=True, 
                      comment="任务类型 (auto_scaling, fault_recovery, capacity_planning, resource_optimization, alert_management)")
    status = Column(String(20), default='pending', index=True, 
                   comment="任务状态 (pending, running, completed, failed, cancelled)")
    
    # 关联信息
    deployment_id = Column(String(100), index=True, comment="关联的部署ID")
    alert_id = Column(String(100), index=True, comment="触发告警ID（如果由告警触发）")
    
    # 任务参数和结果
    parameters = Column(JSON, comment="任务参数")
    result = Column(JSON, comment="执行结果")
    error_message = Column(Text, comment="错误消息")
    
    # 时间信息
    scheduled_at = Column(DateTime, comment="计划执行时间")
    started_at = Column(DateTime, comment="开始执行时间")
    completed_at = Column(DateTime, comment="完成时间")
    
    # 执行信息
    executed_by = Column(String(50), default='system', comment="执行者 (system, user, scheduled)")
    user_id = Column(String(36), index=True, comment="触发用户ID（如果是手动触发）")
    retry_count = Column(Integer, default=0, comment="重试次数")
    max_retries = Column(Integer, default=3, comment="最大重试次数")
    
    # 优先级
    priority = Column(String(20), default='normal', comment="优先级 (low, normal, high, critical)")
    
    # 元数据
    metadata_ = Column(JSON, comment="额外元数据")
    
    def __repr__(self):
        return f"<AutomationTaskRecord(id='{self.id}', task_id='{self.task_id}', type='{self.task_type}', status='{self.status}')>"
    
    def to_dict(self):
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'task_id': self.task_id,
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type,
            'status': self.status,
            'deployment_id': self.deployment_id,
            'alert_id': self.alert_id,
            'parameters': self.parameters,
            'result': self.result,
            'error_message': self.error_message,
            'scheduled_at': self.scheduled_at.isoformat() if self.scheduled_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'executed_by': self.executed_by,
            'user_id': self.user_id,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'priority': self.priority,
            'metadata': self.metadata_,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# ==============================================================================
# 监控报告记录模型
# ==============================================================================

class MonitoringReportRecord(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """监控报告记录模型
    
    用于持久化存储监控分析报告
    """
    __tablename__ = 'monitoring_report_records'
    
    report_id = Column(String(100), unique=True, nullable=False, index=True, comment="报告ID")
    name = Column(String(200), comment="报告名称")
    report_type = Column(String(50), nullable=False, index=True, 
                        comment="报告类型 (performance, trend, capacity, cost, comprehensive)")
    
    # 报告范围
    deployment_id = Column(String(100), index=True, comment="部署ID")
    scope = Column(String(50), default='deployment', comment="报告范围 (deployment, tenant, global)")
    
    # 时间范围
    time_range_start = Column(DateTime, nullable=False, comment="统计开始时间")
    time_range_end = Column(DateTime, nullable=False, comment="统计结束时间")
    
    # 报告内容
    summary = Column(JSON, comment="摘要信息")
    metrics_data = Column(JSON, comment="指标数据")
    trends = Column(JSON, comment="趋势分析")
    capacity_analysis = Column(JSON, comment="容量分析")
    cost_analysis = Column(JSON, comment="成本分析")
    recommendations = Column(JSON, comment="建议列表")
    
    # 生成信息
    generated_by = Column(String(50), default='system', comment="生成方式 (system, user, scheduled)")
    user_id = Column(String(36), comment="请求用户ID")
    
    # 状态
    status = Column(String(20), default='completed', comment="报告状态 (generating, completed, failed)")
    
    def __repr__(self):
        return f"<MonitoringReportRecord(id='{self.id}', report_id='{self.report_id}', type='{self.report_type}')>"
    
    def to_dict(self):
        return {
            'id': str(self.id) if self.id else None,
            'tenant_id': str(self.tenant_id) if self.tenant_id else None,
            'report_id': self.report_id,
            'name': self.name,
            'report_type': self.report_type,
            'deployment_id': self.deployment_id,
            'scope': self.scope,
            'time_range_start': self.time_range_start.isoformat() if self.time_range_start else None,
            'time_range_end': self.time_range_end.isoformat() if self.time_range_end else None,
            'summary': self.summary,
            'metrics_data': self.metrics_data,
            'trends': self.trends,
            'capacity_analysis': self.capacity_analysis,
            'cost_analysis': self.cost_analysis,
            'recommendations': self.recommendations,
            'generated_by': self.generated_by,
            'user_id': self.user_id,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


# ==============================================================================
# 创建索引
# ==============================================================================

Index('idx_system_metrics_name', SystemMetric.name)
Index('idx_system_metrics_type', SystemMetric.type)
Index('idx_system_metrics_source', SystemMetric.source)
Index('idx_alerts_level', Alert.level)
Index('idx_alerts_status', Alert.status)
Index('idx_alerts_source', Alert.source)
Index('idx_system_logs_level', SystemLog.level)
Index('idx_system_logs_source', SystemLog.source)
Index('idx_system_logs_trace_id', SystemLog.trace_id)

# 性能指标索引
Index('idx_perf_metric_deployment_time', PerformanceMetricRecord.deployment_id, PerformanceMetricRecord.timestamp)
Index('idx_perf_metric_type_time', PerformanceMetricRecord.metric_type, PerformanceMetricRecord.timestamp)
Index('idx_perf_metric_tenant_deployment', PerformanceMetricRecord.tenant_id, PerformanceMetricRecord.deployment_id)

# 告警规则索引
Index('idx_alert_rule_tenant_enabled', AlertRuleRecord.tenant_id, AlertRuleRecord.enabled)
Index('idx_alert_rule_deployment', AlertRuleRecord.deployment_id)

# 告警历史索引
Index('idx_alert_history_deployment_time', AlertHistoryRecord.deployment_id, AlertHistoryRecord.triggered_at)
Index('idx_alert_history_tenant_severity', AlertHistoryRecord.tenant_id, AlertHistoryRecord.severity)
Index('idx_alert_history_resolved', AlertHistoryRecord.resolved, AlertHistoryRecord.triggered_at)

# 自动化任务索引
Index('idx_automation_task_deployment_status', AutomationTaskRecord.deployment_id, AutomationTaskRecord.status)
Index('idx_automation_task_tenant_type', AutomationTaskRecord.tenant_id, AutomationTaskRecord.task_type)
Index('idx_automation_task_status_created', AutomationTaskRecord.status, AutomationTaskRecord.created_at)

# 监控报告索引
Index('idx_monitoring_report_tenant_type', MonitoringReportRecord.tenant_id, MonitoringReportRecord.report_type)
Index('idx_monitoring_report_deployment', MonitoringReportRecord.deployment_id)
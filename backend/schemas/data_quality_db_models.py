"""数据质量数据库模型定义

定义数据质量管理相关的SQLAlchemy ORM模型。
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, Index, JSON
from datetime import datetime
import uuid

from .base_models import Base, UUIDMixin, TimestampMixin, GUID


class QualityAssessment(Base, UUIDMixin):
    """质量评估记录模型
    
    存储数据质量评估的结果。
    """
    __tablename__ = 'quality_assessments'
    
    dataset_id = Column(String(36), nullable=False, index=True, comment="数据集ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    
    # 评估结果
    overall_score = Column(Float, default=0.0, comment="总体质量评分")
    dimension_scores = Column(JSON, comment="各维度评分")
    column_metrics = Column(JSON, comment="列级指标")
    
    # 统计信息
    total_records = Column(Integer, default=0, comment="总记录数")
    total_columns = Column(Integer, default=0, comment="总列数")
    missing_values_count = Column(Integer, default=0, comment="缺失值数量")
    missing_values_rate = Column(Float, default=0.0, comment="缺失值比率")
    duplicate_records_count = Column(Integer, default=0, comment="重复记录数量")
    duplicate_records_rate = Column(Float, default=0.0, comment="重复记录比率")
    outliers_count = Column(Integer, default=0, comment="异常值数量")
    outliers_rate = Column(Float, default=0.0, comment="异常值比率")
    
    # 状态
    status = Column(String(20), default="completed", index=True, comment="状态: pending/in_progress/completed/failed")
    error_message = Column(Text, comment="错误信息")
    
    # 时间信息
    assessed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True, comment="评估时间")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, comment="创建时间")
    
    # 元数据
    metadata_ = Column('metadata', JSON, comment="元数据")
    
    __table_args__ = (
        Index('ix_quality_assessments_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_quality_assessments_dataset', 'dataset_id', 'assessed_at'),
    )
    
    def __repr__(self):
        return f"<QualityAssessment(id='{self.id}', dataset_id='{self.dataset_id}', score={self.overall_score})>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'assessment_id': str(self.id),
            'dataset_id': self.dataset_id,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'overall_score': self.overall_score,
            'dimension_scores': self.dimension_scores or {},
            'column_metrics': self.column_metrics or [],
            'total_records': self.total_records,
            'total_columns': self.total_columns,
            'missing_values_count': self.missing_values_count,
            'missing_values_rate': self.missing_values_rate,
            'duplicate_records_count': self.duplicate_records_count,
            'duplicate_records_rate': self.duplicate_records_rate,
            'outliers_count': self.outliers_count,
            'outliers_rate': self.outliers_rate,
            'status': self.status,
            'error_message': self.error_message,
            'assessed_at': self.assessed_at.isoformat() if self.assessed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metadata': self.metadata_,
        }


class QualityIssue(Base, UUIDMixin):
    """质量问题记录模型
    
    存储检测到的数据质量问题。
    """
    __tablename__ = 'quality_issues'
    
    dataset_id = Column(String(36), nullable=False, index=True, comment="数据集ID")
    assessment_id = Column(GUID(), index=True, comment="评估记录ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    
    # 问题信息
    issue_type = Column(String(50), nullable=False, index=True, comment="问题类型")
    severity = Column(String(20), default="medium", index=True, comment="严重程度: critical/high/medium/low/info")
    column_name = Column(String(200), index=True, comment="相关列名")
    description = Column(Text, nullable=False, comment="问题描述")
    
    # 影响范围
    affected_count = Column(Integer, default=0, comment="影响记录数")
    affected_rate = Column(Float, default=0.0, comment="影响比率")
    sample_values = Column(JSON, comment="示例值")
    
    # 建议
    recommendation = Column(Text, comment="修复建议")
    auto_fixable = Column(Boolean, default=False, index=True, comment="是否可自动修复")
    
    # 状态
    status = Column(String(20), default="open", index=True, comment="状态: open/resolved/ignored")
    resolved_at = Column(DateTime, comment="解决时间")
    resolved_by = Column(String(36), comment="解决者ID")
    
    # 时间信息
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True, comment="检测时间")
    
    __table_args__ = (
        Index('ix_quality_issues_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_quality_issues_dataset_status', 'dataset_id', 'status'),
        Index('ix_quality_issues_severity', 'severity', 'status'),
    )
    
    def __repr__(self):
        return f"<QualityIssue(id='{self.id}', type='{self.issue_type}', severity='{self.severity}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'issue_id': str(self.id),
            'dataset_id': self.dataset_id,
            'assessment_id': str(self.assessment_id) if self.assessment_id else None,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'issue_type': self.issue_type,
            'severity': self.severity,
            'column_name': self.column_name,
            'description': self.description,
            'affected_count': self.affected_count,
            'affected_rate': self.affected_rate,
            'sample_values': self.sample_values or [],
            'recommendation': self.recommendation,
            'auto_fixable': self.auto_fixable,
            'status': self.status,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved_by': self.resolved_by,
            'detected_at': self.detected_at.isoformat() if self.detected_at else None,
        }


class CleaningRecord(Base, UUIDMixin, TimestampMixin):
    """清理记录模型
    
    存储数据清理操作的记录。
    """
    __tablename__ = 'cleaning_records'
    
    dataset_id = Column(String(36), nullable=False, index=True, comment="数据集ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    
    # 清理配置
    config = Column(JSON, comment="清理配置")
    
    # 清理结果
    status = Column(String(20), default="pending", index=True, comment="状态: pending/in_progress/completed/failed/rolled_back")
    original_dataset_id = Column(String(36), index=True, comment="原始数据集ID")
    cleaned_dataset_id = Column(String(36), index=True, comment="清理后数据集ID")
    
    # 统计信息
    original_record_count = Column(Integer, default=0, comment="原始记录数")
    cleaned_record_count = Column(Integer, default=0, comment="清理后记录数")
    total_records_affected = Column(Integer, default=0, comment="受影响记录数")
    
    # 操作结果
    operation_results = Column(JSON, comment="操作结果列表")
    
    # 质量评分
    original_quality_score = Column(Float, default=0.0, comment="原始质量评分")
    cleaned_quality_score = Column(Float, default=0.0, comment="清理后质量评分")
    improvement = Column(Float, default=0.0, comment="质量提升")
    
    # 备份信息
    backup_path = Column(Text, comment="备份路径")
    
    # 错误信息
    error_message = Column(Text, comment="错误信息")
    
    # 时间信息
    started_at = Column(DateTime, comment="开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    execution_time_ms = Column(Float, default=0.0, comment="执行时长（毫秒）")
    
    __table_args__ = (
        Index('ix_cleaning_records_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_cleaning_records_dataset', 'dataset_id', 'status'),
    )
    
    def __repr__(self):
        return f"<CleaningRecord(id='{self.id}', dataset_id='{self.dataset_id}', status='{self.status}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'cleaning_id': str(self.id),
            'dataset_id': self.dataset_id,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'config': self.config or {},
            'status': self.status,
            'original_dataset_id': self.original_dataset_id,
            'cleaned_dataset_id': self.cleaned_dataset_id,
            'original_record_count': self.original_record_count,
            'cleaned_record_count': self.cleaned_record_count,
            'total_records_affected': self.total_records_affected,
            'operation_results': self.operation_results or [],
            'original_quality_score': self.original_quality_score,
            'cleaned_quality_score': self.cleaned_quality_score,
            'improvement': self.improvement,
            'backup_path': self.backup_path,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'execution_time_ms': self.execution_time_ms,
        }


class QualityRule(Base, UUIDMixin, TimestampMixin):
    """质量规则模型
    
    存储用户定义的数据质量规则。
    """
    __tablename__ = 'quality_rules'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    
    # 规则信息
    name = Column(String(200), nullable=False, index=True, comment="规则名称")
    description = Column(Text, comment="规则描述")
    rule_type = Column(String(50), nullable=False, index=True, comment="规则类型")
    
    # 规则配置
    target_column = Column(String(200), comment="目标列")
    condition = Column(Text, nullable=False, comment="规则条件")
    parameters = Column(JSON, comment="规则参数")
    
    # 违反规则的严重程度
    severity = Column(String(20), default="medium", comment="严重程度")
    
    # 状态
    enabled = Column(Boolean, default=True, index=True, comment="是否启用")
    
    # 使用信息
    dataset_ids = Column(JSON, comment="应用此规则的数据集ID列表")
    
    __table_args__ = (
        Index('ix_quality_rules_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_quality_rules_enabled', 'enabled'),
    )
    
    def __repr__(self):
        return f"<QualityRule(id='{self.id}', name='{self.name}', type='{self.rule_type}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'rule_id': str(self.id),
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'name': self.name,
            'description': self.description,
            'rule_type': self.rule_type,
            'target_column': self.target_column,
            'condition': self.condition,
            'parameters': self.parameters or {},
            'severity': self.severity,
            'enabled': self.enabled,
            'dataset_ids': self.dataset_ids or [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class RuleValidationRecord(Base, UUIDMixin):
    """规则验证记录模型
    
    存储规则验证的执行结果。
    """
    __tablename__ = 'rule_validation_records'
    
    dataset_id = Column(String(36), nullable=False, index=True, comment="数据集ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    
    # 验证结果
    total_rules = Column(Integer, default=0, comment="总规则数")
    passed_rules = Column(Integer, default=0, comment="通过规则数")
    failed_rules = Column(Integer, default=0, comment="失败规则数")
    pass_rate = Column(Float, default=0.0, comment="通过率")
    
    # 详细结果
    results = Column(JSON, comment="验证结果详情")
    
    # 时间信息
    validated_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True, comment="验证时间")
    
    __table_args__ = (
        Index('ix_rule_validation_records_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_rule_validation_records_dataset', 'dataset_id', 'validated_at'),
    )
    
    def __repr__(self):
        return f"<RuleValidationRecord(id='{self.id}', pass_rate={self.pass_rate})>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'validation_id': str(self.id),
            'dataset_id': self.dataset_id,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'total_rules': self.total_rules,
            'passed_rules': self.passed_rules,
            'failed_rules': self.failed_rules,
            'pass_rate': self.pass_rate,
            'results': self.results or [],
            'validated_at': self.validated_at.isoformat() if self.validated_at else None,
        }


class QualityReport(Base, UUIDMixin):
    """质量报告模型
    
    存储生成的数据质量报告。
    """
    __tablename__ = 'quality_reports'
    
    dataset_id = Column(String(36), nullable=False, index=True, comment="数据集ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    
    # 报告内容
    dataset_name = Column(String(200), comment="数据集名称")
    report_content = Column(JSON, nullable=False, comment="报告内容")
    
    # 摘要
    summary = Column(Text, comment="报告摘要")
    recommendations = Column(JSON, comment="改进建议列表")
    
    # 评分
    overall_score = Column(Float, default=0.0, comment="总体评分")
    
    # 时间信息
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True, comment="生成时间")
    
    __table_args__ = (
        Index('ix_quality_reports_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_quality_reports_dataset', 'dataset_id', 'generated_at'),
    )
    
    def __repr__(self):
        return f"<QualityReport(id='{self.id}', dataset_id='{self.dataset_id}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'report_id': str(self.id),
            'dataset_id': self.dataset_id,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'dataset_name': self.dataset_name,
            'report_content': self.report_content or {},
            'summary': self.summary,
            'recommendations': self.recommendations or [],
            'overall_score': self.overall_score,
            'generated_at': self.generated_at.isoformat() if self.generated_at else None,
        }


class QualityMonitoringConfig(Base, UUIDMixin, TimestampMixin):
    """质量监控配置模型
    
    存储数据集的质量监控配置。
    """
    __tablename__ = 'quality_monitoring_configs'
    
    dataset_id = Column(String(36), nullable=False, unique=True, index=True, comment="数据集ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    
    # 监控配置
    enabled = Column(Boolean, default=True, index=True, comment="是否启用监控")
    thresholds = Column(JSON, comment="质量阈值配置")
    check_interval_minutes = Column(Integer, default=60, comment="检查间隔（分钟）")
    alert_channels = Column(JSON, comment="告警渠道配置")
    
    # 状态
    last_check_at = Column(DateTime, comment="最后检查时间")
    next_check_at = Column(DateTime, index=True, comment="下次检查时间")
    
    __table_args__ = (
        Index('ix_quality_monitoring_configs_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_quality_monitoring_configs_enabled', 'enabled', 'next_check_at'),
    )
    
    def __repr__(self):
        return f"<QualityMonitoringConfig(id='{self.id}', dataset_id='{self.dataset_id}', enabled={self.enabled})>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'config_id': str(self.id),
            'dataset_id': self.dataset_id,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'enabled': self.enabled,
            'thresholds': self.thresholds or [],
            'check_interval_minutes': self.check_interval_minutes,
            'alert_channels': self.alert_channels or [],
            'last_check_at': self.last_check_at.isoformat() if self.last_check_at else None,
            'next_check_at': self.next_check_at.isoformat() if self.next_check_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class QualityAlert(Base, UUIDMixin):
    """质量告警模型
    
    存储触发的质量告警。
    """
    __tablename__ = 'quality_alerts'
    
    dataset_id = Column(String(36), nullable=False, index=True, comment="数据集ID")
    config_id = Column(GUID(), index=True, comment="监控配置ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    
    # 告警信息
    dimension = Column(String(50), nullable=False, comment="质量维度")
    current_score = Column(Float, default=0.0, comment="当前评分")
    threshold_score = Column(Float, default=0.0, comment="阈值评分")
    severity = Column(String(20), default="medium", index=True, comment="严重程度")
    message = Column(Text, nullable=False, comment="告警消息")
    
    # 状态
    acknowledged = Column(Boolean, default=False, index=True, comment="是否已确认")
    acknowledged_by = Column(String(36), comment="确认者ID")
    acknowledged_at = Column(DateTime, comment="确认时间")
    
    # 时间信息
    triggered_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True, comment="触发时间")
    
    __table_args__ = (
        Index('ix_quality_alerts_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_quality_alerts_dataset', 'dataset_id', 'triggered_at'),
        Index('ix_quality_alerts_acknowledged', 'acknowledged', 'triggered_at'),
    )
    
    def __repr__(self):
        return f"<QualityAlert(id='{self.id}', dimension='{self.dimension}', severity='{self.severity}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'alert_id': str(self.id),
            'dataset_id': self.dataset_id,
            'config_id': str(self.config_id) if self.config_id else None,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'dimension': self.dimension,
            'current_score': self.current_score,
            'threshold_score': self.threshold_score,
            'severity': self.severity,
            'message': self.message,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
        }

"""数据预处理数据库模型定义

定义数据预处理相关的SQLAlchemy ORM模型。
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, Index, JSON
from datetime import datetime
import uuid

from .base_models import Base, UUIDMixin, TimestampMixin, GUID


class PreprocessingTask(Base, UUIDMixin, TimestampMixin):
    """预处理任务模型
    
    记录数据预处理任务的执行信息。
    """
    __tablename__ = 'preprocessing_tasks'
    
    dataset_id = Column(String(36), nullable=False, index=True, comment="数据集ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    
    task_type = Column(String(50), default="preprocessing", index=True, comment="任务类型: preprocessing/feature_engineering/augmentation/split")
    task_name = Column(String(200), comment="任务名称")
    description = Column(Text, comment="任务描述")
    
    status = Column(String(20), default="pending", index=True, comment="状态: pending/processing/completed/failed/cancelled")
    priority = Column(Integer, default=0, comment="任务优先级")
    
    config = Column(JSON, comment="任务配置")
    result = Column(JSON, comment="任务结果")
    error_message = Column(Text, comment="错误信息")
    
    # 数据快照路径（用于回滚）
    snapshot_path = Column(Text, comment="快照路径")
    
    # 统计信息
    original_rows = Column(Integer, default=0, comment="原始行数")
    final_rows = Column(Integer, default=0, comment="最终行数")
    original_columns = Column(Integer, default=0, comment="原始列数")
    final_columns = Column(Integer, default=0, comment="最终列数")
    
    # 时间信息
    started_at = Column(DateTime, comment="开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    
    # 元数据
    metadata_ = Column('metadata', JSON, comment="元数据")
    
    __table_args__ = (
        Index('ix_preprocessing_tasks_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_preprocessing_tasks_dataset_status', 'dataset_id', 'status'),
    )
    
    def __repr__(self):
        return f"<PreprocessingTask(id='{self.id}', type='{self.task_type}', status='{self.status}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'task_id': str(self.id),
            'dataset_id': self.dataset_id,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'task_type': self.task_type,
            'task_name': self.task_name,
            'description': self.description,
            'status': self.status,
            'priority': self.priority,
            'config': self.config or {},
            'result': self.result,
            'error_message': self.error_message,
            'snapshot_path': self.snapshot_path,
            'original_rows': self.original_rows,
            'final_rows': self.final_rows,
            'original_columns': self.original_columns,
            'final_columns': self.final_columns,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'metadata': self.metadata_,
        }


class PreprocessingHistory(Base, UUIDMixin):
    """预处理历史记录模型
    
    记录每次预处理操作的详细信息，支持回滚。
    """
    __tablename__ = 'preprocessing_histories'
    
    dataset_id = Column(String(36), nullable=False, index=True, comment="数据集ID")
    task_id = Column(GUID(), index=True, comment="任务ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    
    operation_type = Column(String(50), nullable=False, index=True, comment="操作类型")
    operation_name = Column(String(200), comment="操作名称")
    operation_config = Column(JSON, comment="操作配置")
    operation_result = Column(JSON, comment="操作结果")
    
    # 数据变更记录
    rows_before = Column(Integer, default=0, comment="操作前行数")
    rows_after = Column(Integer, default=0, comment="操作后行数")
    columns_before = Column(Integer, default=0, comment="操作前列数")
    columns_after = Column(Integer, default=0, comment="操作后列数")
    columns_added = Column(JSON, comment="新增的列")
    columns_removed = Column(JSON, comment="删除的列")
    columns_modified = Column(JSON, comment="修改的列")
    
    # 快照信息（用于回滚）
    snapshot_path = Column(Text, comment="快照路径")
    can_rollback = Column(Boolean, default=True, index=True, comment="是否可回滚")
    
    executed_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True, comment="执行时间")
    duration_ms = Column(Integer, default=0, comment="执行时长（毫秒）")
    
    __table_args__ = (
        Index('ix_preprocessing_histories_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_preprocessing_histories_dataset', 'dataset_id', 'executed_at'),
    )
    
    def __repr__(self):
        return f"<PreprocessingHistory(id='{self.id}', type='{self.operation_type}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'history_id': str(self.id),
            'dataset_id': self.dataset_id,
            'task_id': str(self.task_id) if self.task_id else None,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'operation_type': self.operation_type,
            'operation_name': self.operation_name,
            'operation_config': self.operation_config or {},
            'operation_result': self.operation_result or {},
            'rows_before': self.rows_before,
            'rows_after': self.rows_after,
            'columns_before': self.columns_before,
            'columns_after': self.columns_after,
            'columns_added': self.columns_added or [],
            'columns_removed': self.columns_removed or [],
            'columns_modified': self.columns_modified or [],
            'snapshot_path': self.snapshot_path,
            'can_rollback': self.can_rollback,
            'executed_at': self.executed_at.isoformat() if self.executed_at else None,
            'duration_ms': self.duration_ms,
        }


class PreprocessingPipeline(Base, UUIDMixin, TimestampMixin):
    """预处理流水线模型
    
    存储可复用的预处理流水线模板。
    """
    __tablename__ = 'preprocessing_pipelines'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    
    name = Column(String(200), nullable=False, index=True, comment="流水线名称")
    description = Column(Text, comment="流水线描述")
    
    operations = Column(JSON, nullable=False, comment="操作列表")
    
    is_template = Column(Boolean, default=False, index=True, comment="是否为模板")
    is_public = Column(Boolean, default=False, index=True, comment="是否公开")
    
    usage_count = Column(Integer, default=0, comment="使用次数")
    
    __table_args__ = (
        Index('ix_preprocessing_pipelines_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_preprocessing_pipelines_template', 'is_template', 'is_public'),
    )
    
    def __repr__(self):
        return f"<PreprocessingPipeline(id='{self.id}', name='{self.name}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'pipeline_id': str(self.id),
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'name': self.name,
            'description': self.description,
            'operations': self.operations or [],
            'is_template': self.is_template,
            'is_public': self.is_public,
            'usage_count': self.usage_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class FeatureStore(Base, UUIDMixin, TimestampMixin):
    """特征存储模型
    
    存储特征工程产生的特征定义。
    """
    __tablename__ = 'feature_stores'
    
    dataset_id = Column(String(36), nullable=False, index=True, comment="数据集ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    
    feature_name = Column(String(200), nullable=False, index=True, comment="特征名称")
    feature_type = Column(String(50), nullable=False, index=True, comment="特征类型: numeric/categorical/text/embedding")
    description = Column(Text, comment="特征描述")
    
    # 特征定义
    expression = Column(Text, comment="特征表达式")
    source_columns = Column(JSON, comment="源列")
    transform_config = Column(JSON, comment="转换配置")
    
    # 特征统计
    statistics = Column(JSON, comment="特征统计信息")
    importance_score = Column(Float, comment="特征重要性评分")
    
    # 版本信息
    version = Column(Integer, default=1, comment="版本号")
    
    __table_args__ = (
        Index('ix_feature_stores_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_feature_stores_dataset', 'dataset_id'),
        Index('ix_feature_stores_name', 'dataset_id', 'feature_name'),
    )
    
    def __repr__(self):
        return f"<FeatureStore(id='{self.id}', name='{self.feature_name}', type='{self.feature_type}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'feature_id': str(self.id),
            'dataset_id': self.dataset_id,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'feature_name': self.feature_name,
            'feature_type': self.feature_type,
            'description': self.description,
            'expression': self.expression,
            'source_columns': self.source_columns or [],
            'transform_config': self.transform_config or {},
            'statistics': self.statistics or {},
            'importance_score': self.importance_score,
            'version': self.version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

"""数据发现数据库模型定义

定义数据发现相关的SQLAlchemy ORM模型。
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, Index, JSON
from datetime import datetime
import uuid

from .base_models import Base, UUIDMixin, TimestampMixin, GUID


class DataSource(Base, UUIDMixin, TimestampMixin):
    """数据源模型
    
    存储数据源连接配置和状态信息。
    """
    __tablename__ = 'data_sources'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    source_type = Column(String(50), nullable=False, default="file_system", index=True, comment="数据源类型")
    location = Column(Text, nullable=False, comment="数据源位置")
    name = Column(String(200), nullable=False, index=True, comment="数据源名称")
    description = Column(Text, comment="数据源描述")
    credentials = Column(JSON, comment="连接凭证（加密存储）")
    config = Column(JSON, comment="数据源配置")
    status = Column(String(20), default="active", index=True, comment="状态: active/inactive/error")
    last_scan_at = Column(DateTime, comment="最后扫描时间")
    last_scan_result = Column(JSON, comment="最后扫描结果")
    
    __table_args__ = (
        Index('ix_data_sources_user_tenant', 'user_id', 'tenant_id'),
    )
    
    def __repr__(self):
        return f"<DataSource(id='{self.id}', name='{self.name}', type='{self.source_type}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'source_id': str(self.id),
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'source_type': self.source_type,
            'location': self.location,
            'name': self.name,
            'description': self.description,
            'credentials': self.credentials,
            'config': self.config or {},
            'status': self.status,
            'last_scan_at': self.last_scan_at.isoformat() if self.last_scan_at else None,
            'last_scan_result': self.last_scan_result,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class DiscoveryRecord(Base, UUIDMixin, TimestampMixin):
    """数据发现记录模型
    
    记录每次数据发现任务的执行信息。
    """
    __tablename__ = 'discovery_records'
    
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    source_id = Column(GUID(), index=True, comment="数据源ID")
    source_type = Column(String(50), nullable=False, default="file_system", comment="数据源类型")
    source_location = Column(Text, comment="数据源位置")
    status = Column(String(20), default="pending", index=True, comment="状态: pending/scanning/discovered/failed/completed")
    datasets_discovered = Column(Integer, default=0, comment="发现的数据集数量")
    datasets_ingested = Column(Integer, default=0, comment="已接入的数据集数量")
    discovered_items = Column(JSON, comment="发现的数据项列表")
    scan_config = Column(JSON, comment="扫描配置")
    error_message = Column(Text, comment="错误信息")
    completed_at = Column(DateTime, comment="完成时间")
    metadata_ = Column('metadata', JSON, comment="元数据")
    
    __table_args__ = (
        Index('ix_discovery_records_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_discovery_records_source', 'source_id'),
    )
    
    def __repr__(self):
        return f"<DiscoveryRecord(id='{self.id}', status='{self.status}', discovered={self.datasets_discovered})>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'record_id': str(self.id),
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'source_id': str(self.source_id) if self.source_id else None,
            'source_type': self.source_type,
            'source_location': self.source_location,
            'status': self.status,
            'datasets_discovered': self.datasets_discovered,
            'datasets_ingested': self.datasets_ingested,
            'discovered_items': self.discovered_items or [],
            'scan_config': self.scan_config or {},
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'metadata': self.metadata_,
        }


class DiscoveredDataset(Base, UUIDMixin):
    """发现的数据集模型
    
    存储通过数据发现找到的数据集信息。
    """
    __tablename__ = 'discovered_datasets'
    
    record_id = Column(GUID(), nullable=False, index=True, comment="发现记录ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    dataset_name = Column(String(200), nullable=False, comment="数据集名称")
    source_id = Column(String(36), nullable=False, index=True, comment="数据源ID")
    source_type = Column(String(50), nullable=False, default="file_system", comment="数据源类型")
    source_path = Column(Text, nullable=False, comment="数据源路径")
    data_format = Column(String(50), default="unknown", index=True, comment="数据格式")
    size_bytes = Column(Integer, default=0, comment="数据大小（字节）")
    row_count = Column(Integer, comment="行数")
    column_count = Column(Integer, comment="列数")
    schema_info = Column(JSON, comment="模式信息")
    preview_data = Column(JSON, comment="预览数据")
    quality_score = Column(Float, comment="质量评分")
    completeness = Column(Float, comment="完整性")
    status = Column(String(20), default="discovered", index=True, comment="状态: discovered/ingesting/ingested/failed/ignored")
    ingested_dataset_id = Column(String(36), index=True, comment="接入后的数据集ID")
    discovered_at = Column(DateTime, default=datetime.utcnow, nullable=False, comment="发现时间")
    metadata_ = Column('metadata', JSON, comment="元数据")
    
    __table_args__ = (
        Index('ix_discovered_datasets_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_discovered_datasets_record', 'record_id'),
    )
    
    def __repr__(self):
        return f"<DiscoveredDataset(id='{self.id}', name='{self.dataset_name}', status='{self.status}')>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'discovery_id': str(self.id),
            'record_id': str(self.record_id) if self.record_id else None,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'dataset_name': self.dataset_name,
            'source_id': self.source_id,
            'source_type': self.source_type,
            'source_path': self.source_path,
            'data_format': self.data_format,
            'size_bytes': self.size_bytes,
            'row_count': self.row_count,
            'column_count': self.column_count,
            'schema_info': self.schema_info,
            'preview_data': self.preview_data,
            'quality_score': self.quality_score,
            'completeness': self.completeness,
            'status': self.status,
            'ingested_dataset_id': self.ingested_dataset_id,
            'discovered_at': self.discovered_at.isoformat() if self.discovered_at else None,
            'metadata': self.metadata_,
        }


class SyncConfig(Base, UUIDMixin, TimestampMixin):
    """同步配置模型
    
    存储数据集的增量同步配置。
    """
    __tablename__ = 'sync_configs'
    
    dataset_id = Column(String(36), nullable=False, unique=True, index=True, comment="数据集ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    tenant_id = Column(String(36), index=True, comment="租户ID")
    sync_enabled = Column(Boolean, default=True, index=True, comment="是否启用同步")
    frequency = Column(String(20), default="daily", comment="同步频率: hourly/daily/weekly/monthly")
    incremental_column = Column(String(100), comment="增量字段")
    incremental_method = Column(String(20), default="timestamp", comment="增量方法: timestamp/id/hash")
    cron_expression = Column(String(100), comment="Cron表达式")
    timezone = Column(String(50), default="UTC", comment="时区")
    conflict_resolution = Column(String(20), default="update", comment="冲突解决策略: update/skip/error")
    last_sync_at = Column(DateTime, comment="最后同步时间")
    last_sync_status = Column(String(20), comment="最后同步状态")
    last_sync_rows = Column(Integer, comment="最后同步行数")
    next_sync_at = Column(DateTime, index=True, comment="下次同步时间")
    last_error = Column(Text, comment="最后错误信息")
    config = Column(JSON, comment="额外配置")
    
    __table_args__ = (
        Index('ix_sync_configs_user_tenant', 'user_id', 'tenant_id'),
        Index('ix_sync_configs_next_sync', 'sync_enabled', 'next_sync_at'),
    )
    
    def __repr__(self):
        return f"<SyncConfig(id='{self.id}', dataset_id='{self.dataset_id}', enabled={self.sync_enabled})>"
    
    def to_dict(self):
        """转换为字典"""
        return {
            'sync_id': str(self.id),
            'dataset_id': self.dataset_id,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'sync_enabled': self.sync_enabled,
            'frequency': self.frequency,
            'incremental_column': self.incremental_column,
            'incremental_method': self.incremental_method,
            'cron_expression': self.cron_expression,
            'timezone': self.timezone,
            'conflict_resolution': self.conflict_resolution,
            'last_sync_at': self.last_sync_at.isoformat() if self.last_sync_at else None,
            'last_sync_status': self.last_sync_status,
            'last_sync_rows': self.last_sync_rows,
            'next_sync_at': self.next_sync_at.isoformat() if self.next_sync_at else None,
            'last_error': self.last_error,
            'config': self.config or {},
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

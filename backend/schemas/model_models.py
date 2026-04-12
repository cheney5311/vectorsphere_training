#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型相关的数据库模型

定义模型相关的 SQLAlchemy 数据库模型和数据类。
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from sqlalchemy import (
    Column, String, Text, JSON, Boolean, Integer, Float, DateTime,
    Enum as SQLEnum, ForeignKey, Index
)

from backend.schemas.base_models import Base, UUIDMixin, TimestampMixin, TenantMixin, GUID
from backend.schemas.enums import ModelType, ModelFramework, ModelStatus


# ==================== 枚举类型 ====================

class ModelArchitectureEnum(str, Enum):
    """模型架构枚举"""
    TRANSFORMER = "transformer"
    CNN = "cnn"
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"
    RESNET = "resnet"
    BERT = "bert"
    GPT = "gpt"
    LLAMA = "llama"
    CUSTOM = "custom"


class ModelDeploymentStatusEnum(str, Enum):
    """模型部署状态"""
    NOT_DEPLOYED = "not_deployed"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    STOPPED = "stopped"


class ModelExportFormatEnum(str, Enum):
    """模型导出格式"""
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TENSORFLOW_SAVED_MODEL = "tensorflow_saved_model"
    TENSORRT = "tensorrt"
    COREML = "coreml"
    SAFETENSORS = "safetensors"


class ModelEventTypeEnum(str, Enum):
    """模型事件类型"""
    CREATED = "created"
    UPDATED = "updated"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    EXPORTED = "exported"
    ARCHIVED = "archived"
    DELETED = "deleted"


# ==================== SQLAlchemy 模型 ====================

class ModelDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型数据库模型"""
    __tablename__ = 'models'
    
    # 基本信息
    name = Column(String(200), nullable=False, index=True, comment="模型名称")
    description = Column(Text, comment="模型描述")
    version = Column(String(50), default="1.0.0", comment="模型版本")
    
    # 模型类型和框架
    model_type = Column(SQLEnum(ModelType), nullable=False, index=True, comment="模型类型")
    framework = Column(SQLEnum(ModelFramework), nullable=False, comment="模型框架")
    architecture = Column(String(100), comment="模型架构")
    status = Column(SQLEnum(ModelStatus), default=ModelStatus.DRAFT, index=True, comment="模型状态")
    
    # 存储和配置
    storage_path = Column(String(500), comment="模型存储路径")
    config = Column(JSON, comment="模型配置信息")
    
    # 模型性能指标
    accuracy = Column(Float, comment="准确率")
    loss = Column(Float, comment="损失值")
    f1_score = Column(Float, comment="F1分数")
    
    # 模型大小信息
    size_mb = Column(Float, comment="模型大小(MB)")
    parameters_count = Column(Integer, comment="参数数量")
    
    # 部署信息
    deployment_status = Column(String(50), default='not_deployed', comment="部署状态")
    deployment_endpoint = Column(String(500), comment="部署端点")
    deployment_config = Column(JSON, comment="部署配置")
    
    # 标签和分类
    tags = Column(JSON, default=list, comment="标签列表")
    category = Column(String(100), comment="分类")
    
    # 可见性
    is_public = Column(Boolean, default=False, comment="是否公开")
    is_archived = Column(Boolean, default=False, comment="是否归档")
    
    # 关联信息
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    training_session_id = Column(String(36), index=True, comment="训练会话ID")
    dataset_id = Column(String(36), index=True, comment="数据集ID")
    parent_model_id = Column(String(36), index=True, comment="父模型ID")
    
    # 索引
    __table_args__ = (
        Index('ix_models_user_status', 'user_id', 'status'),
        Index('ix_models_tenant_type', 'tenant_id', 'model_type'),
    )
    
    def __repr__(self):
        return f"<ModelDB(id='{self.id}', name='{self.name}', type='{self.model_type}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'model_type': self.model_type.value if self.model_type else None,
            'framework': self.framework.value if self.framework else None,
            'architecture': self.architecture,
            'status': self.status.value if self.status else None,
            'storage_path': self.storage_path,
            'config': self.config,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'f1_score': self.f1_score,
            'size_mb': self.size_mb,
            'parameters_count': self.parameters_count,
            'deployment_status': self.deployment_status,
            'deployment_endpoint': self.deployment_endpoint,
            'tags': self.tags,
            'category': self.category,
            'is_public': self.is_public,
            'is_archived': self.is_archived,
            'user_id': self.user_id,
            'training_session_id': self.training_session_id,
            'dataset_id': self.dataset_id,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class ModelMetadataDB(Base, UUIDMixin, TimestampMixin):
    """模型元数据数据库模型"""
    __tablename__ = 'model_metadata'
    
    model_id = Column(GUID(), nullable=False, index=True, comment="模型ID")
    
    # 模型大小和复杂度
    size_mb = Column(Float, comment="模型大小(MB)")
    parameters_count = Column(Integer, comment="参数数量")
    layers_count = Column(Integer, comment="层数")
    trainable_params = Column(Integer, comment="可训练参数数量")
    
    # 性能指标
    accuracy = Column(Float, comment="准确率")
    precision = Column(Float, comment="精确率")
    recall = Column(Float, comment="召回率")
    f1_score = Column(Float, comment="F1分数")
    loss = Column(Float, comment="损失值")
    auc = Column(Float, comment="AUC值")
    
    # 运行时性能
    training_time_seconds = Column(Float, comment="训练时间(秒)")
    inference_time_ms = Column(Float, comment="推理时间(毫秒)")
    memory_usage_mb = Column(Float, comment="内存使用量(MB)")
    gpu_memory_mb = Column(Float, comment="GPU内存使用量(MB)")
    
    # 训练信息
    epochs_trained = Column(Integer, comment="训练轮数")
    batch_size = Column(Integer, comment="批次大小")
    learning_rate = Column(Float, comment="学习率")
    optimizer = Column(String(50), comment="优化器")
    
    # 数据集信息
    training_samples = Column(Integer, comment="训练样本数")
    validation_samples = Column(Integer, comment="验证样本数")
    test_samples = Column(Integer, comment="测试样本数")
    
    # 额外指标
    custom_metrics = Column(JSON, default=dict, comment="自定义指标")
    
    def __repr__(self):
        return f"<ModelMetadataDB(model_id='{self.model_id}', size='{self.size_mb}MB')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'model_id': str(self.model_id) if self.model_id else None,
            'size_mb': self.size_mb,
            'parameters_count': self.parameters_count,
            'layers_count': self.layers_count,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'loss': self.loss,
            'training_time_seconds': self.training_time_seconds,
            'inference_time_ms': self.inference_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'epochs_trained': self.epochs_trained,
            'custom_metrics': self.custom_metrics,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class ModelVersionDB(Base, UUIDMixin, TimestampMixin):
    """模型版本数据库模型"""
    __tablename__ = 'model_versions'
    
    model_id = Column(GUID(), nullable=False, index=True, comment="模型ID")
    version = Column(String(50), nullable=False, comment="版本号")
    description = Column(Text, comment="版本描述")
    is_active = Column(Boolean, default=False, comment="是否激活")
    
    # 版本存储
    storage_path = Column(String(500), comment="版本存储路径")
    checksum = Column(String(128), comment="校验和")
    
    # 版本指标
    accuracy = Column(Float, comment="准确率")
    loss = Column(Float, comment="损失值")
    
    # 版本元数据
    changelog = Column(Text, comment="变更日志")
    tags = Column(JSON, default=list, comment="标签")
    
    # 关联
    created_by = Column(String(36), comment="创建者ID")
    parent_version_id = Column(String(36), comment="父版本ID")
    
    __table_args__ = (
        Index('ix_model_versions_model_version', 'model_id', 'version'),
    )
    
    def __repr__(self):
        return f"<ModelVersionDB(model_id='{self.model_id}', version='{self.version}', active='{self.is_active}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'model_id': str(self.model_id) if self.model_id else None,
            'version': self.version,
            'description': self.description,
            'is_active': self.is_active,
            'storage_path': self.storage_path,
            'checksum': self.checksum,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'changelog': self.changelog,
            'tags': self.tags,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class ModelArtifactDB(Base, UUIDMixin, TimestampMixin):
    """模型工件数据库模型"""
    __tablename__ = 'model_artifacts'
    
    model_id = Column(GUID(), nullable=False, index=True, comment="模型ID")
    version_id = Column(GUID(), index=True, comment="版本ID")
    
    # 工件信息
    artifact_type = Column(String(50), nullable=False, comment="工件类型")
    artifact_name = Column(String(200), nullable=False, comment="工件名称")
    artifact_path = Column(String(500), nullable=False, comment="工件路径")
    
    # 文件信息
    file_size_bytes = Column(Integer, comment="文件大小(字节)")
    mime_type = Column(String(100), comment="MIME类型")
    checksum = Column(String(128), comment="校验和")
    
    # 元数据
    metadata_ = Column('metadata', JSON, default=dict, comment="元数据")
    
    def __repr__(self):
        return f"<ModelArtifactDB(model_id='{self.model_id}', type='{self.artifact_type}', name='{self.artifact_name}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'model_id': str(self.model_id) if self.model_id else None,
            'version_id': str(self.version_id) if self.version_id else None,
            'artifact_type': self.artifact_type,
            'artifact_name': self.artifact_name,
            'artifact_path': self.artifact_path,
            'file_size_bytes': self.file_size_bytes,
            'mime_type': self.mime_type,
            'checksum': self.checksum,
            'metadata': self.metadata_,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class ModelEventDB(Base, UUIDMixin, TimestampMixin):
    """模型事件数据库模型"""
    __tablename__ = 'model_events'
    
    model_id = Column(GUID(), nullable=False, index=True, comment="模型ID")
    
    # 事件信息
    event_type = Column(String(50), nullable=False, index=True, comment="事件类型")
    event_message = Column(Text, comment="事件消息")
    severity = Column(String(20), default='info', comment="严重级别")
    
    # 事件来源
    source = Column(String(100), comment="事件来源")
    user_id = Column(String(36), comment="操作用户ID")
    
    # 事件数据
    event_data = Column(JSON, default=dict, comment="事件数据")
    
    # 时间
    event_time = Column(DateTime, default=datetime.utcnow, index=True, comment="事件时间")
    
    __table_args__ = (
        Index('ix_model_events_model_time', 'model_id', 'event_time'),
    )
    
    def __repr__(self):
        return f"<ModelEventDB(model_id='{self.model_id}', type='{self.event_type}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'model_id': str(self.model_id) if self.model_id else None,
            'event_type': self.event_type,
            'event_message': self.event_message,
            'severity': self.severity,
            'source': self.source,
            'user_id': self.user_id,
            'event_data': self.event_data,
            'event_time': self.event_time.isoformat() if self.event_time else None,
        }


class ModelExportDB(Base, UUIDMixin, TimestampMixin):
    """模型导出记录数据库模型"""
    __tablename__ = 'model_exports'
    
    model_id = Column(GUID(), nullable=False, index=True, comment="模型ID")
    version_id = Column(GUID(), index=True, comment="版本ID")
    
    # 导出信息
    export_format = Column(String(50), nullable=False, comment="导出格式")
    export_path = Column(String(500), comment="导出路径")
    export_status = Column(String(50), default='pending', comment="导出状态")
    
    # 导出配置
    export_config = Column(JSON, default=dict, comment="导出配置")
    
    # 文件信息
    file_size_bytes = Column(Integer, comment="文件大小(字节)")
    checksum = Column(String(128), comment="校验和")
    
    # 导出结果
    error_message = Column(Text, comment="错误信息")
    
    # 关联
    user_id = Column(String(36), comment="导出用户ID")
    
    def __repr__(self):
        return f"<ModelExportDB(model_id='{self.model_id}', format='{self.export_format}', status='{self.export_status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'model_id': str(self.model_id) if self.model_id else None,
            'version_id': str(self.version_id) if self.version_id else None,
            'export_format': self.export_format,
            'export_path': self.export_path,
            'export_status': self.export_status,
            'export_config': self.export_config,
            'file_size_bytes': self.file_size_bytes,
            'error_message': self.error_message,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


# ==================== 数据类 ====================

@dataclass
class ModelInfo:
    """模型信息数据类"""
    id: str = ''
    name: str = ''
    description: Optional[str] = None
    version: str = '1.0.0'
    model_type: str = 'classification'
    framework: str = 'pytorch'
    architecture: Optional[str] = None
    status: str = 'draft'
    
    storage_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    f1_score: Optional[float] = None
    size_mb: Optional[float] = None
    parameters_count: Optional[int] = None
    
    deployment_status: str = 'not_deployed'
    deployment_endpoint: Optional[str] = None
    
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    is_public: bool = False
    is_archived: bool = False
    
    user_id: str = ''
    tenant_id: Optional[str] = None
    training_session_id: Optional[str] = None
    dataset_id: Optional[str] = None
    
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'model_type': self.model_type,
            'framework': self.framework,
            'architecture': self.architecture,
            'status': self.status,
            'storage_path': self.storage_path,
            'config': self.config,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'f1_score': self.f1_score,
            'size_mb': self.size_mb,
            'parameters_count': self.parameters_count,
            'deployment_status': self.deployment_status,
            'deployment_endpoint': self.deployment_endpoint,
            'tags': self.tags,
            'category': self.category,
            'is_public': self.is_public,
            'is_archived': self.is_archived,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class ModelMetrics:
    """模型指标数据类"""
    model_id: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    loss: Optional[float] = None
    auc: Optional[float] = None
    
    training_time_seconds: Optional[float] = None
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'loss': self.loss,
            'auc': self.auc,
            'training_time_seconds': self.training_time_seconds,
            'inference_time_ms': self.inference_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'custom_metrics': self.custom_metrics,
        }


@dataclass
class ModelVersion:
    """模型版本数据类"""
    id: str = ''
    model_id: str = ''
    version: str = '1.0.0'
    description: Optional[str] = None
    is_active: bool = False
    storage_path: Optional[str] = None
    checksum: Optional[str] = None
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    changelog: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'model_id': self.model_id,
            'version': self.version,
            'description': self.description,
            'is_active': self.is_active,
            'storage_path': self.storage_path,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'changelog': self.changelog,
            'tags': self.tags,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class ModelSummary:
    """模型摘要数据类"""
    total_models: int = 0
    draft_models: int = 0
    trained_models: int = 0
    deployed_models: int = 0
    archived_models: int = 0
    
    by_type: Dict[str, int] = field(default_factory=dict)
    by_framework: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)
    
    avg_accuracy: Optional[float] = None
    total_size_mb: float = 0.0
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_models': self.total_models,
            'draft_models': self.draft_models,
            'trained_models': self.trained_models,
            'deployed_models': self.deployed_models,
            'archived_models': self.archived_models,
            'by_type': self.by_type,
            'by_framework': self.by_framework,
            'by_status': self.by_status,
            'avg_accuracy': self.avg_accuracy,
            'total_size_mb': self.total_size_mb,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


# ==================== 导出 ====================

__all__ = [
    # 枚举
    'ModelArchitectureEnum',
    'ModelDeploymentStatusEnum',
    'ModelExportFormatEnum',
    'ModelEventTypeEnum',
    
    # SQLAlchemy 模型
    'ModelDB',
    'ModelMetadataDB',
    'ModelVersionDB',
    'ModelArtifactDB',
    'ModelEventDB',
    'ModelExportDB',
    
    # 数据类
    'ModelInfo',
    'ModelMetrics',
    'ModelVersion',
    'ModelSummary',
]
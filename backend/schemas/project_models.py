"""项目和数据集相关数据模型

定义项目、数据集等业务相关数据模型。
"""

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, ForeignKey, Index, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from .base_models import Base, UUIDMixin, TimestampMixin, TenantMixin, GUID
from .enums import DatasetStatus, ModelStatus, DeploymentStatus, ProjectStatus


class Project(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """项目模型"""
    __tablename__ = 'projects'
    
    name = Column(String(200), nullable=False, index=True, comment="项目名称")
    description = Column(Text, comment="项目描述")
    status = Column(String(20), default=ProjectStatus.ACTIVE.value, index=True, comment="项目状态")
    owner_id = Column(String(36), nullable=False, index=True, comment="所有者ID")
    collaborators = Column(JSON, comment="协作者列表")
    tags = Column(JSON, comment="标签")
    metadata_ = Column('metadata', JSON, comment="元数据")
    
    def __repr__(self):
        return f"<Project(id='{self.id}', name='{self.name}', status='{self.status}')>"


class Dataset(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """数据集模型"""
    __tablename__ = 'datasets'
    
    name = Column(String(200), nullable=False, index=True, comment="数据集名称")
    description = Column(Text, comment="数据集描述")
    status = Column(String(20), default=DatasetStatus.PENDING.value, index=True, comment="数据集状态")
    type = Column(String(50), nullable=False, comment="数据集类型")
    format = Column(String(50), comment="数据格式")
    size = Column(Integer, comment="数据集大小(字节)")
    record_count = Column(Integer, comment="记录数")
    features = Column(JSON, comment="特征信息")
    labels = Column(JSON, comment="标签信息")
    source = Column(String(200), comment="数据源")
    version = Column(String(20), comment="版本")
    path = Column(Text, comment="存储路径")
    checksum = Column(String(64), comment="校验和")
    
    def __repr__(self):
        return f"<Dataset(id='{self.id}', name='{self.name}', status='{self.status}')>"


class Model(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型模型"""
    __tablename__ = 'project_models'
    
    name = Column(String(200), nullable=False, comment="模型名称")
    description = Column(Text, comment="模型描述")
    type = Column(String(50), nullable=False, comment="模型类型")
    framework = Column(String(50), nullable=False, comment="模型框架")
    status = Column(String(20), default=ModelStatus.DRAFT.value, comment="模型状态")
    owner_id = Column(String(36), nullable=False, comment="所有者ID")
    version = Column(String(20), comment="版本")
    path = Column(Text, comment="存储路径")
    checksum = Column(String(64), comment="校验和")
    size = Column(Integer, comment="模型大小(字节)")
    metrics = Column(JSON, comment="性能指标")
    hyperparameters = Column(JSON, comment="超参数")
    tags = Column(JSON, comment="标签")
    metadata_ = Column('metadata', JSON, comment="元数据")
    
    # 显式定义索引以避免命名冲突
    __table_args__ = (
        Index('ix_project_models_name', 'name'),
        Index('ix_project_models_status', 'status'),
        Index('ix_project_models_owner_id', 'owner_id'),
    )
    
    def __repr__(self):
        return f"<Model(id='{self.id}', name='{self.name}', status='{self.status}')>"


class ModelVersion(Base, UUIDMixin, TimestampMixin):
    """模型版本"""
    __tablename__ = 'project_model_versions'
    
    model_id = Column(GUID(), ForeignKey('project_models.id'), nullable=False, comment="模型ID")
    version = Column(String(20), nullable=False, comment="版本号")
    description = Column(Text, comment="版本描述")
    path = Column(Text, comment="存储路径")
    checksum = Column(String(64), comment="校验和")
    size = Column(Integer, comment="模型大小(字节)")
    metrics = Column(JSON, comment="性能指标")
    hyperparameters = Column(JSON, comment="超参数")
    created_by = Column(String(36), nullable=False, comment="创建者ID")
    
    # 显式定义索引以避免命名冲突
    __table_args__ = (
        Index('ix_project_model_versions_model_id', 'model_id'),
    )
    
    def __repr__(self):
        return f"<ModelVersion(id='{self.id}', model_id='{self.model_id}', version='{self.version}')>"


class ModelDeployment(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型部署模型"""
    __tablename__ = 'project_model_deployments'
    
    model_id = Column(GUID(), ForeignKey('project_models.id'), nullable=False, comment="模型ID")
    name = Column(String(200), nullable=False, comment="部署名称")
    description = Column(Text, comment="部署描述")
    status = Column(String(20), default=DeploymentStatus.PENDING.value, comment="部署状态")
    endpoint = Column(String(200), comment="访问端点")
    version = Column(String(20), comment="部署版本")
    config = Column(JSON, comment="部署配置")
    resources = Column(JSON, comment="资源配置")
    metrics = Column(JSON, comment="部署指标")
    error_message = Column(Text, comment="错误信息")
    deployed_at = Column(DateTime, comment="部署时间")
    deployed_by = Column(String(36), comment="部署者ID")
    
    # 显式定义索引以避免命名冲突
    __table_args__ = (
        Index('ix_project_model_deployments_model_id', 'model_id'),
        Index('ix_project_model_deployments_name', 'name'),
        Index('ix_project_model_deployments_status', 'status'),
    )
    
    def __repr__(self):
        return f"<ModelDeployment(id='{self.id}', model_id='{self.model_id}', status='{self.status}')>"
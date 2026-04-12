# -*- coding: utf-8 -*-
"""工件安全数据模型

定义工件、安全策略、文件元数据等相关的数据库模型。
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from sqlalchemy import (
    Column, String, Integer, BigInteger, Float, Boolean, Text, JSON,
    DateTime, ForeignKey, Index, Enum as SQLEnum
)
from sqlalchemy.orm import relationship

try:
    from backend.schemas.base_models import BaseModel
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base
    BaseModel = declarative_base()


# ==================== 枚举类型 ====================

class SecurityLevelEnum(str, Enum):
    """安全级别"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ArtifactTypeEnum(str, Enum):
    """工件类型"""
    TRAINING_DATA = "training_data"
    MODEL_FILE = "model_file"
    CONFIG_FILE = "config_file"
    LOG_FILE = "log_file"
    REPORT = "report"
    BACKUP = "backup"
    TEMP = "temp"


class FileTypeEnum(str, Enum):
    """文件类型"""
    DOCUMENT = "document"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    ARCHIVE = "archive"
    CODE = "code"
    DATA = "data"
    MODEL = "model"
    UNKNOWN = "unknown"


class ArtifactStatusEnum(str, Enum):
    """工件状态"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    PENDING = "pending"
    LOCKED = "locked"


class VersionStatusEnum(str, Enum):
    """版本状态"""
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    DELETED = "deleted"


class ScanStatusEnum(str, Enum):
    """扫描状态"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"
    SKIPPED = "skipped"


# ==================== 数据库模型 ====================

class SecurityPolicyModel(BaseModel):
    """安全策略模型"""
    __tablename__ = 'security_policies'
    
    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), index=True, nullable=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    security_level = Column(String(32), nullable=False)
    allowed_file_types = Column(JSON, nullable=False, default=list)
    max_file_size = Column(BigInteger, nullable=False, default=10485760)  # 10MB
    encryption_required = Column(Boolean, default=False)
    virus_scan_required = Column(Boolean, default=True)
    access_control_enabled = Column(Boolean, default=True)
    audit_enabled = Column(Boolean, default=True)
    retention_days = Column(Integer, default=365)
    is_default = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_by = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_policy_tenant', 'tenant_id'),
        Index('idx_policy_level', 'security_level'),
    )


class ArtifactModel(BaseModel):
    """工件模型"""
    __tablename__ = 'artifacts'
    
    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), index=True, nullable=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    artifact_type = Column(String(32), nullable=False)
    security_level = Column(String(32), nullable=False)
    status = Column(String(32), default='active')
    owner_id = Column(String(64), nullable=False, index=True)
    current_version = Column(String(32), nullable=True)
    version_count = Column(Integer, default=0)
    total_size = Column(BigInteger, default=0)
    tags = Column(JSON, default=list)
    extra_metadata = Column('metadata', JSON, default=dict)
    policy_id = Column(String(64), ForeignKey('security_policies.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 关联
    versions = relationship("ArtifactVersionModel", back_populates="artifact", cascade="all, delete-orphan")
    policy = relationship("SecurityPolicyModel")
    
    __table_args__ = (
        Index('idx_artifact_tenant', 'tenant_id'),
        Index('idx_artifact_owner', 'owner_id'),
        Index('idx_artifact_type', 'artifact_type'),
        Index('idx_artifact_status', 'status'),
    )


class ArtifactVersionModel(BaseModel):
    """工件版本模型"""
    __tablename__ = 'artifact_versions'
    
    id = Column(String(64), primary_key=True)
    artifact_id = Column(String(64), ForeignKey('artifacts.id'), nullable=False, index=True)
    version = Column(String(32), nullable=False)
    status = Column(String(32), default='active')
    file_path = Column(String(512), nullable=True)
    file_size = Column(BigInteger, default=0)
    file_hash = Column(String(128), nullable=True)
    mime_type = Column(String(128), nullable=True)
    changelog = Column(Text, nullable=True)
    tags = Column(JSON, default=list)
    extra_metadata = Column('metadata', JSON, default=dict)
    created_by = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关联
    artifact = relationship("ArtifactModel", back_populates="versions")
    
    __table_args__ = (
        Index('idx_version_artifact', 'artifact_id'),
        Index('idx_version_status', 'status'),
    )


class FileMetadataModel(BaseModel):
    """文件元数据模型"""
    __tablename__ = 'file_metadata'
    
    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), index=True, nullable=True)
    original_name = Column(String(255), nullable=False)
    stored_name = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(32), nullable=False)
    mime_type = Column(String(128), nullable=True)
    size = Column(BigInteger, nullable=False)
    hash_sha256 = Column(String(64), nullable=True, index=True)
    hash_md5 = Column(String(32), nullable=True)
    security_level = Column(String(32), nullable=False)
    artifact_type = Column(String(32), nullable=True)
    artifact_id = Column(String(64), ForeignKey('artifacts.id'), nullable=True)
    version_id = Column(String(64), ForeignKey('artifact_versions.id'), nullable=True)
    owner_id = Column(String(64), nullable=False, index=True)
    is_encrypted = Column(Boolean, default=False)
    encryption_key_id = Column(String(64), nullable=True)
    tags = Column(JSON, default=list)
    extra_metadata = Column('metadata', JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    accessed_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        Index('idx_file_tenant', 'tenant_id'),
        Index('idx_file_owner', 'owner_id'),
        Index('idx_file_artifact', 'artifact_id'),
    )


class SecurityScanResultModel(BaseModel):
    """安全扫描结果模型"""
    __tablename__ = 'security_scan_results'
    
    id = Column(String(64), primary_key=True)
    file_id = Column(String(64), ForeignKey('file_metadata.id'), nullable=False, index=True)
    scan_type = Column(String(32), nullable=False)  # virus_scan, malware_scan, integrity_check
    status = Column(String(32), nullable=False)  # passed, failed, warning
    threats_found = Column(JSON, default=list)
    scanner_name = Column(String(64), nullable=True)
    scanner_version = Column(String(32), nullable=True)
    scan_duration_ms = Column(Integer, nullable=True)
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_scan_file', 'file_id'),
        Index('idx_scan_status', 'status'),
    )


class ArtifactDependencyModel(BaseModel):
    """工件依赖关系模型"""
    __tablename__ = 'artifact_dependencies'
    
    id = Column(String(64), primary_key=True)
    source_artifact_id = Column(String(64), ForeignKey('artifacts.id'), nullable=False, index=True)
    target_artifact_id = Column(String(64), ForeignKey('artifacts.id'), nullable=False, index=True)
    dependency_type = Column(String(32), default='required')  # required, optional, dev
    version_constraint = Column(String(64), nullable=True)  # >=1.0.0, ^1.0.0, *
    created_by = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_dep_source', 'source_artifact_id'),
        Index('idx_dep_target', 'target_artifact_id'),
    )


class ArtifactAccessLogModel(BaseModel):
    """工件访问日志模型"""
    __tablename__ = 'artifact_access_logs'
    
    id = Column(String(64), primary_key=True)
    tenant_id = Column(String(64), index=True, nullable=True)
    artifact_id = Column(String(64), ForeignKey('artifacts.id'), nullable=True)
    file_id = Column(String(64), ForeignKey('file_metadata.id'), nullable=True)
    user_id = Column(String(64), nullable=False, index=True)
    operation = Column(String(32), nullable=False)  # read, write, delete, download, upload
    result = Column(String(32), nullable=False)  # success, denied, failed
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(String(256), nullable=True)
    details = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_access_tenant', 'tenant_id'),
        Index('idx_access_artifact', 'artifact_id'),
        Index('idx_access_user', 'user_id'),
        Index('idx_access_time', 'created_at'),
    )


class ArtifactPermissionModel(BaseModel):
    """工件权限模型"""
    __tablename__ = 'artifact_permissions'
    
    id = Column(String(64), primary_key=True)
    artifact_id = Column(String(64), ForeignKey('artifacts.id'), nullable=False, index=True)
    principal_type = Column(String(32), nullable=False)  # user, role, group
    principal_id = Column(String(64), nullable=False, index=True)
    permissions = Column(JSON, default=list)  # ['read', 'write', 'delete', 'share']
    granted_by = Column(String(64), nullable=True)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_perm_artifact', 'artifact_id'),
        Index('idx_perm_principal', 'principal_type', 'principal_id'),
    )


# ==================== Pydantic 模型（用于API请求/响应）====================

from pydantic import BaseModel as PydanticBaseModel, Field


class SecurityPolicyCreate(PydanticBaseModel):
    """创建安全策略请求"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    security_level: str = Field(..., description="安全级别")
    allowed_file_types: List[str] = Field(..., min_items=1)
    max_file_size: int = Field(..., gt=0)
    encryption_required: bool = Field(default=False)
    virus_scan_required: bool = Field(default=True)
    access_control_enabled: bool = Field(default=True)
    audit_enabled: bool = Field(default=True)
    retention_days: int = Field(default=365, gt=0)


class SecurityPolicyUpdate(PydanticBaseModel):
    """更新安全策略请求"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    allowed_file_types: Optional[List[str]] = None
    max_file_size: Optional[int] = Field(None, gt=0)
    encryption_required: Optional[bool] = None
    virus_scan_required: Optional[bool] = None
    access_control_enabled: Optional[bool] = None
    audit_enabled: Optional[bool] = None
    retention_days: Optional[int] = Field(None, gt=0)


class SecurityPolicyResponse(PydanticBaseModel):
    """安全策略响应"""
    id: str
    name: str
    description: Optional[str]
    security_level: str
    allowed_file_types: List[str]
    max_file_size: int
    encryption_required: bool
    virus_scan_required: bool
    access_control_enabled: bool
    audit_enabled: bool
    retention_days: int
    is_default: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime


class ArtifactCreate(PydanticBaseModel):
    """创建工件请求"""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    artifact_type: str = Field(...)
    security_level: str = Field(...)
    tags: List[str] = Field(default=[])
    metadata: Dict[str, Any] = Field(default={})
    policy_id: Optional[str] = None


class ArtifactUpdate(PydanticBaseModel):
    """更新工件请求"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ArtifactResponse(PydanticBaseModel):
    """工件响应"""
    id: str
    name: str
    description: Optional[str]
    artifact_type: str
    security_level: str
    status: str
    owner_id: str
    current_version: Optional[str]
    version_count: int
    total_size: int
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ArtifactVersionCreate(PydanticBaseModel):
    """创建工件版本请求"""
    version: str = Field(..., min_length=1, max_length=32)
    changelog: Optional[str] = Field(None, max_length=1000)
    tags: List[str] = Field(default=[])
    metadata: Dict[str, Any] = Field(default={})


class ArtifactVersionResponse(PydanticBaseModel):
    """工件版本响应"""
    id: str
    artifact_id: str
    version: str
    status: str
    file_size: int
    file_hash: Optional[str]
    mime_type: Optional[str]
    changelog: Optional[str]
    tags: List[str]
    created_by: str
    created_at: datetime


class DependencyCreate(PydanticBaseModel):
    """创建依赖请求"""
    target_artifact_id: str = Field(...)
    dependency_type: str = Field(default="required")
    version_constraint: Optional[str] = Field(default="*")


class PermissionGrant(PydanticBaseModel):
    """授权请求"""
    principal_type: str = Field(...)  # user, role, group
    principal_id: str = Field(...)
    permissions: List[str] = Field(...)  # read, write, delete, share
    expires_at: Optional[datetime] = None

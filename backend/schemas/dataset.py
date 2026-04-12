"""数据集模型定义

包含数据集相关的SQLAlchemy ORM模型和数据传输对象(DTO)。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid
import json

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, JSON, Index, func

from .base_models import Base, UUIDMixin, TimestampMixin, TenantMixin, GUID
from .enums import DatasetStatus


# ============================================================================
# SQLAlchemy ORM 模型定义
# ============================================================================

class DatasetEntity(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """数据集实体模型 - SQLAlchemy ORM
    
    用于持久化数据集信息到数据库。
    
    Attributes:
        id: 数据集唯一标识符 (UUID)
        user_id: 所属用户ID
        name: 数据集名称
        description: 数据集描述
        dataset_type: 数据集类型 (text/image/audio/video/tabular/mixed)
        format: 数据格式 (json/csv/parquet/tfrecord/arrow/custom)
        status: 数据集状态 (pending/uploading/processing/ready/error/archived)
        storage_path: 存储路径
        size: 数据集大小(字节)
        record_count: 记录数
        features: 特征信息 (JSON)
        labels: 标签信息 (JSON)
        config: 配置信息 (JSON)
        metadata_: 元数据 (JSON)
        source: 数据源
        version: 版本号
        checksum: 校验和
        ready: 是否就绪
        validated: 是否已验证
        validation_result: 验证结果 (JSON)
        processed_at: 处理时间
        created_at: 创建时间
        updated_at: 更新时间
    """
    __tablename__ = 'dataset_entities'
    
    # 基本信息
    user_id = Column(String(36), nullable=False, index=True, comment="所属用户ID")
    name = Column(String(200), nullable=False, index=True, comment="数据集名称")
    description = Column(Text, comment="数据集描述")
    
    # 类型和格式
    dataset_type = Column(String(50), nullable=False, default='text', comment="数据集类型")
    format = Column(String(50), default='json', comment="数据格式")
    
    # 状态
    status = Column(String(20), default=DatasetStatus.PENDING.value, index=True, comment="数据集状态")
    ready = Column(Boolean, default=False, comment="是否就绪")
    validated = Column(Boolean, default=False, comment="是否已验证")
    
    # 存储信息
    storage_path = Column(Text, comment="存储路径")
    size = Column(Integer, comment="数据集大小(字节)")
    record_count = Column(Integer, comment="记录数")
    checksum = Column(String(64), comment="校验和")
    
    # 数据结构信息
    features = Column(JSON, comment="特征信息")
    labels = Column(JSON, comment="标签信息")
    schema_info = Column(JSON, comment="数据模式信息")
    
    # 配置和元数据
    config = Column(JSON, comment="配置信息")
    metadata_ = Column('metadata', JSON, comment="元数据")
    validation_result = Column(JSON, comment="验证结果")
    
    # 版本控制
    source = Column(String(200), comment="数据源")
    version = Column(String(20), default='1.0', comment="版本号")
    
    # 处理时间
    processed_at = Column(DateTime, comment="处理时间")
    
    # 索引定义
    __table_args__ = (
        Index('ix_dataset_entities_user_status', 'user_id', 'status'),
        Index('ix_dataset_entities_type_format', 'dataset_type', 'format'),
        Index('ix_dataset_entities_name_user', 'name', 'user_id'),
    )
    
    def __repr__(self):
        return f"<DatasetEntity(id='{self.id}', name='{self.name}', status='{self.status}')>"
    
    def to_dto(self) -> 'Dataset':
        """转换为数据传输对象
        
        Returns:
            Dataset: 数据集DTO对象
        """
        return Dataset(
            dataset_id=str(self.id) if self.id else '',
            user_id=self.user_id or '',
            name=self.name or '',
            description=self.description,
            dataset_type=self.dataset_type or 'text',
            format=self.format or 'json',
            storage_path=self.storage_path or '',
            config=self.config or {},
            created_at=self.created_at or datetime.utcnow(),
            updated_at=self.updated_at,
            status=self.status or 'pending',
            ready=self.ready or False,
            size=self.size,
            record_count=self.record_count,
            features=self.features,
            labels=self.labels,
            version=self.version,
            checksum=self.checksum,
            validated=self.validated or False,
            validation_result=self.validation_result
        )
    
    @classmethod
    def from_dto(cls, dto: 'Dataset', tenant_id: str = None) -> 'DatasetEntity':
        """从数据传输对象创建实体
        
        Args:
            dto: 数据集DTO对象
            tenant_id: 租户ID
            
        Returns:
            DatasetEntity: 数据集实体对象
        """
        entity = cls(
            user_id=dto.user_id,
            name=dto.name,
            description=dto.description,
            dataset_type=dto.dataset_type,
            format=dto.format,
            storage_path=dto.storage_path,
            config=dto.config,
            status=dto.status,
            ready=dto.ready,
            size=dto.size,
            record_count=dto.record_count,
            features=dto.features,
            labels=dto.labels,
            version=dto.version,
            checksum=dto.checksum,
            validated=dto.validated,
            validation_result=dto.validation_result,
            tenant_id=tenant_id or dto.user_id
        )
        # 如果有ID，则设置
        if dto.dataset_id:
            try:
                entity.id = uuid.UUID(dto.dataset_id)
            except (ValueError, TypeError):
                pass
        return entity


class DatasetVersionEntity(Base, UUIDMixin, TimestampMixin):
    """数据集版本实体模型 - SQLAlchemy ORM
    
    用于跟踪数据集的版本历史。
    
    Attributes:
        id: 版本唯一标识符 (UUID)
        dataset_id: 所属数据集ID
        version: 版本号
        description: 版本描述
        storage_path: 存储路径
        size: 数据大小(字节)
        record_count: 记录数
        checksum: 校验和
        created_by: 创建者ID
        changes: 变更内容 (JSON)
        parent_version_id: 父版本ID
    """
    __tablename__ = 'dataset_versions'
    
    dataset_id = Column(GUID(), nullable=False, index=True, comment="数据集ID")
    version = Column(String(20), nullable=False, comment="版本号")
    description = Column(Text, comment="版本描述")
    storage_path = Column(Text, comment="存储路径")
    size = Column(Integer, comment="数据大小(字节)")
    record_count = Column(Integer, comment="记录数")
    checksum = Column(String(64), comment="校验和")
    created_by = Column(String(36), nullable=False, comment="创建者ID")
    changes = Column(JSON, comment="变更内容")
    parent_version_id = Column(GUID(), comment="父版本ID")
    
    __table_args__ = (
        Index('ix_dataset_versions_dataset_version', 'dataset_id', 'version'),
    )
    
    def __repr__(self):
        return f"<DatasetVersionEntity(dataset_id='{self.dataset_id}', version='{self.version}')>"


class DatasetStatisticsEntity(Base, UUIDMixin, TimestampMixin):
    """数据集统计信息实体模型 - SQLAlchemy ORM
    
    存储数据集的统计分析结果。
    
    Attributes:
        id: 统计记录唯一标识符 (UUID)
        dataset_id: 数据集ID
        total_records: 总记录数
        total_size: 总大小(字节)
        column_count: 列数
        numeric_columns: 数值列数
        categorical_columns: 分类列数
        missing_values: 缺失值统计 (JSON)
        data_types: 数据类型统计 (JSON)
        value_distribution: 值分布 (JSON)
        correlation_matrix: 相关性矩阵 (JSON)
        outlier_summary: 异常值摘要 (JSON)
        quality_score: 数据质量分数
        computed_at: 计算时间
    """
    __tablename__ = 'dataset_statistics'
    
    dataset_id = Column(GUID(), nullable=False, index=True, comment="数据集ID")
    total_records = Column(Integer, comment="总记录数")
    total_size = Column(Integer, comment="总大小(字节)")
    column_count = Column(Integer, comment="列数")
    numeric_columns = Column(Integer, comment="数值列数")
    categorical_columns = Column(Integer, comment="分类列数")
    missing_values = Column(JSON, comment="缺失值统计")
    data_types = Column(JSON, comment="数据类型统计")
    value_distribution = Column(JSON, comment="值分布")
    correlation_matrix = Column(JSON, comment="相关性矩阵")
    outlier_summary = Column(JSON, comment="异常值摘要")
    quality_score = Column(Float, comment="数据质量分数")
    computed_at = Column(DateTime, default=func.now(), comment="计算时间")
    
    def __repr__(self):
        return f"<DatasetStatisticsEntity(dataset_id='{self.dataset_id}', quality_score={self.quality_score})>"


class DatasetTagEntity(Base, UUIDMixin, TimestampMixin):
    """数据集标签实体模型 - SQLAlchemy ORM
    
    用于数据集的标签管理。
    
    Attributes:
        id: 标签唯一标识符 (UUID)
        dataset_id: 数据集ID
        tag_name: 标签名称
        tag_value: 标签值
        created_by: 创建者ID
    """
    __tablename__ = 'dataset_tags'
    
    dataset_id = Column(GUID(), nullable=False, index=True, comment="数据集ID")
    tag_name = Column(String(100), nullable=False, index=True, comment="标签名称")
    tag_value = Column(String(255), comment="标签值")
    created_by = Column(String(36), comment="创建者ID")
    
    __table_args__ = (
        Index('ix_dataset_tags_dataset_tag', 'dataset_id', 'tag_name'),
    )
    
    def __repr__(self):
        return f"<DatasetTagEntity(dataset_id='{self.dataset_id}', tag_name='{self.tag_name}')>"


class DatasetAccessLogEntity(Base, UUIDMixin):
    """数据集访问日志实体模型 - SQLAlchemy ORM
    
    记录数据集的访问历史。
    
    Attributes:
        id: 日志唯一标识符 (UUID)
        dataset_id: 数据集ID
        user_id: 访问用户ID
        action: 操作类型 (read/write/delete/download/process)
        details: 操作详情 (JSON)
        ip_address: IP地址
        user_agent: 用户代理
        timestamp: 访问时间
    """
    __tablename__ = 'dataset_access_logs'
    
    dataset_id = Column(GUID(), nullable=False, index=True, comment="数据集ID")
    user_id = Column(String(36), nullable=False, index=True, comment="访问用户ID")
    action = Column(String(50), nullable=False, index=True, comment="操作类型")
    details = Column(JSON, comment="操作详情")
    ip_address = Column(String(45), comment="IP地址")
    user_agent = Column(Text, comment="用户代理")
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True, comment="访问时间")
    
    __table_args__ = (
        Index('ix_dataset_access_logs_dataset_time', 'dataset_id', 'timestamp'),
        Index('ix_dataset_access_logs_user_time', 'user_id', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<DatasetAccessLogEntity(dataset_id='{self.dataset_id}', action='{self.action}')>"


# ============================================================================
# 数据传输对象 (DTO) 定义
# ============================================================================

@dataclass
class Dataset:
    """数据集数据传输对象 (DTO)
    
    用于在服务层之间传递数据集信息。
    
    Attributes:
        dataset_id: 数据集唯一标识符
        user_id: 所属用户ID
        name: 数据集名称
        description: 数据集描述
        dataset_type: 数据集类型 (text/image/audio/video/tabular/mixed)
        format: 数据格式 (json/csv/parquet/tfrecord/arrow/custom)
        storage_path: 存储路径
        config: 配置信息
        created_at: 创建时间
        updated_at: 更新时间
        status: 数据集状态
        ready: 是否就绪
        size: 数据集大小(字节)
        record_count: 记录数
        features: 特征信息
        labels: 标签信息
        version: 版本号
        checksum: 校验和
        validated: 是否已验证
        validation_result: 验证结果
    """
    dataset_id: str = field(default_factory=lambda: "")
    user_id: str = ""
    name: str = ""
    description: Optional[str] = None
    dataset_type: str = "text"
    format: str = "json"
    storage_path: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    status: str = "pending"
    ready: bool = False
    size: Optional[int] = None
    record_count: Optional[int] = None
    features: Optional[Dict[str, Any]] = None
    labels: Optional[Dict[str, Any]] = None
    version: Optional[str] = "1.0"
    checksum: Optional[str] = None
    validated: bool = False
    validation_result: Optional[Dict[str, Any]] = None

    @property
    def id(self) -> str:
        """兼容仓库可能使用的 id 属性
        
        Returns:
            str: 数据集ID
        """
        return self.dataset_id

    def process(self) -> None:
        """将数据集状态设置为处理中"""
        self.status = "processing"
        self.updated_at = datetime.utcnow()

    def mark_ready(self) -> None:
        """将数据集标记为就绪状态"""
        self.ready = True
        self.status = "ready"
        self.updated_at = datetime.utcnow()

    def mark_error(self, error_message: str = None) -> None:
        """将数据集标记为错误状态
        
        Args:
            error_message: 错误信息
        """
        self.status = "error"
        self.ready = False
        self.updated_at = datetime.utcnow()
        if error_message and self.config is not None:
            self.config['error_message'] = error_message

    def validate_dataset(self, validation_result: Dict[str, Any]) -> None:
        """记录验证结果
        
        Args:
            validation_result: 验证结果字典
        """
        self.validated = True
        self.validation_result = validation_result
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 数据集信息字典
        """
        return {
            "dataset_id": self.dataset_id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "dataset_type": self.dataset_type,
            "format": self.format,
            "storage_path": self.storage_path,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status,
            "ready": self.ready,
            "size": self.size,
            "record_count": self.record_count,
            "features": self.features,
            "labels": self.labels,
            "version": self.version,
            "checksum": self.checksum,
            "validated": self.validated,
            "validation_result": self.validation_result
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dataset':
        """从字典创建数据集对象
        
        Args:
            data: 数据集信息字典
            
        Returns:
            Dataset: 数据集对象
        """
        # 处理时间字段
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        elif created_at is None:
            created_at = datetime.utcnow()
            
        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            
        return cls(
            dataset_id=data.get('dataset_id', ''),
            user_id=data.get('user_id', ''),
            name=data.get('name', ''),
            description=data.get('description'),
            dataset_type=data.get('dataset_type', 'text'),
            format=data.get('format', 'json'),
            storage_path=data.get('storage_path', ''),
            config=data.get('config', {}),
            created_at=created_at,
            updated_at=updated_at,
            status=data.get('status', 'pending'),
            ready=data.get('ready', False),
            size=data.get('size'),
            record_count=data.get('record_count'),
            features=data.get('features'),
            labels=data.get('labels'),
            version=data.get('version', '1.0'),
            checksum=data.get('checksum'),
            validated=data.get('validated', False),
            validation_result=data.get('validation_result')
        )


@dataclass
class DatasetVersion:
    """数据集版本数据传输对象 (DTO)
    
    Attributes:
        version_id: 版本唯一标识符
        dataset_id: 所属数据集ID
        version: 版本号
        description: 版本描述
        storage_path: 存储路径
        size: 数据大小(字节)
        record_count: 记录数
        checksum: 校验和
        created_by: 创建者ID
        created_at: 创建时间
        changes: 变更内容
        parent_version_id: 父版本ID
    """
    version_id: str = field(default_factory=lambda: "")
    dataset_id: str = ""
    version: str = "1.0"
    description: Optional[str] = None
    storage_path: Optional[str] = None
    size: Optional[int] = None
    record_count: Optional[int] = None
    checksum: Optional[str] = None
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    changes: Optional[Dict[str, Any]] = None
    parent_version_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "version_id": self.version_id,
            "dataset_id": self.dataset_id,
            "version": self.version,
            "description": self.description,
            "storage_path": self.storage_path,
            "size": self.size,
            "record_count": self.record_count,
            "checksum": self.checksum,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "changes": self.changes,
            "parent_version_id": self.parent_version_id
        }


@dataclass
class DatasetStatistics:
    """数据集统计信息数据传输对象 (DTO)
    
    Attributes:
        statistics_id: 统计记录唯一标识符
        dataset_id: 数据集ID
        total_records: 总记录数
        total_size: 总大小(字节)
        column_count: 列数
        numeric_columns: 数值列数
        categorical_columns: 分类列数
        missing_values: 缺失值统计
        data_types: 数据类型统计
        value_distribution: 值分布
        correlation_matrix: 相关性矩阵
        outlier_summary: 异常值摘要
        quality_score: 数据质量分数
        computed_at: 计算时间
    """
    statistics_id: str = field(default_factory=lambda: "")
    dataset_id: str = ""
    total_records: int = 0
    total_size: int = 0
    column_count: int = 0
    numeric_columns: int = 0
    categorical_columns: int = 0
    missing_values: Optional[Dict[str, Any]] = None
    data_types: Optional[Dict[str, Any]] = None
    value_distribution: Optional[Dict[str, Any]] = None
    correlation_matrix: Optional[Dict[str, Any]] = None
    outlier_summary: Optional[Dict[str, Any]] = None
    quality_score: float = 0.0
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "statistics_id": self.statistics_id,
            "dataset_id": self.dataset_id,
            "total_records": self.total_records,
            "total_size": self.total_size,
            "column_count": self.column_count,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "missing_values": self.missing_values,
            "data_types": self.data_types,
            "value_distribution": self.value_distribution,
            "correlation_matrix": self.correlation_matrix,
            "outlier_summary": self.outlier_summary,
            "quality_score": self.quality_score,
            "computed_at": self.computed_at.isoformat() if self.computed_at else None
        }


# ============================================================================
# 请求和响应模型定义
# ============================================================================

@dataclass
class CreateDatasetRequest:
    """创建数据集请求模型
    
    Attributes:
        name: 数据集名称 (必填)
        description: 数据集描述 (可选)
        dataset_type: 数据集类型 (默认: text)
        format: 数据格式 (默认: json)
        storage_path: 存储路径 (可选)
        config: 配置信息 (可选)
    """
    name: str
    description: Optional[str] = None
    dataset_type: str = "text"
    format: str = "json"
    storage_path: str = ""
    config: Optional[Dict[str, Any]] = None
    
    def validate(self) -> List[str]:
        """验证请求参数
        
        Returns:
            List[str]: 错误信息列表，为空表示验证通过
        """
        errors = []
        if not self.name or len(self.name.strip()) == 0:
            errors.append("数据集名称不能为空")
        if len(self.name) > 200:
            errors.append("数据集名称长度不能超过200个字符")
        
        valid_types = ['text', 'image', 'audio', 'video', 'tabular', 'mixed']
        if self.dataset_type not in valid_types:
            errors.append(f"无效的数据集类型，支持: {', '.join(valid_types)}")
            
        valid_formats = ['json', 'csv', 'parquet', 'tfrecord', 'arrow', 'custom']
        if self.format not in valid_formats:
            errors.append(f"无效的数据格式，支持: {', '.join(valid_formats)}")
            
        return errors
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CreateDatasetRequest':
        """从字典创建请求对象"""
        return cls(
            name=data.get('name', ''),
            description=data.get('description'),
            dataset_type=data.get('type', data.get('dataset_type', 'text')),
            format=data.get('format', 'json'),
            storage_path=data.get('storage_path', ''),
            config=data.get('config')
        )


@dataclass
class UpdateDatasetRequest:
    """更新数据集请求模型
    
    Attributes:
        name: 数据集名称 (可选)
        description: 数据集描述 (可选)
        config: 配置信息 (可选)
        status: 数据集状态 (可选)
    """
    name: Optional[str] = None
    description: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    
    def validate(self) -> List[str]:
        """验证请求参数
        
        Returns:
            List[str]: 错误信息列表，为空表示验证通过
        """
        errors = []
        if self.name is not None:
            if len(self.name.strip()) == 0:
                errors.append("数据集名称不能为空")
            if len(self.name) > 200:
                errors.append("数据集名称长度不能超过200个字符")
                
        if self.status is not None:
            valid_statuses = ['pending', 'uploading', 'processing', 'ready', 'error', 'archived']
            if self.status not in valid_statuses:
                errors.append(f"无效的状态，支持: {', '.join(valid_statuses)}")
                
        return errors
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UpdateDatasetRequest':
        """从字典创建请求对象"""
        return cls(
            name=data.get('name'),
            description=data.get('description'),
            config=data.get('config'),
            status=data.get('status')
        )


@dataclass
class DatasetListResponse:
    """数据集列表响应模型
    
    Attributes:
        datasets: 数据集列表
        total: 总数
        limit: 限制数量
        offset: 偏移量
        has_more: 是否还有更多数据
    """
    datasets: List[Dataset]
    total: int = 0
    limit: int = 50
    offset: int = 0
    has_more: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "datasets": [d.to_dict() for d in self.datasets],
            "total": self.total,
            "limit": self.limit,
            "offset": self.offset,
            "has_more": self.has_more
        }


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # ORM模型
    'DatasetEntity',
    'DatasetVersionEntity',
    'DatasetStatisticsEntity',
    'DatasetTagEntity',
    'DatasetAccessLogEntity',
    # DTO模型
    'Dataset',
    'DatasetVersion',
    'DatasetStatistics',
    # 请求响应模型
    'CreateDatasetRequest',
    'UpdateDatasetRequest',
    'DatasetListResponse'
]

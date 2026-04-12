"""数据发现与接入模块数据模型

定义数据发现与接入相关的请求和响应模型。
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum


# ============================================================================
# 数据源相关枚举
# ============================================================================

class DataSourceType(str, Enum):
    """数据源类型枚举"""
    FILE_SYSTEM = "file_system"          # 文件系统
    DATABASE = "database"                 # 数据库
    API = "api"                           # API接口
    S3 = "s3"                             # AWS S3
    GCS = "gcs"                           # Google Cloud Storage
    AZURE_BLOB = "azure_blob"             # Azure Blob Storage
    HDFS = "hdfs"                         # Hadoop分布式文件系统
    KAFKA = "kafka"                       # Kafka消息队列
    FTP = "ftp"                           # FTP服务器
    SFTP = "sftp"                         # SFTP服务器
    HTTP = "http"                         # HTTP端点
    WEBSOCKET = "websocket"               # WebSocket数据流


class DataFormat(str, Enum):
    """数据格式枚举"""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    AVRO = "avro"
    XML = "xml"
    EXCEL = "excel"
    SQL = "sql"
    TEXT = "text"
    BINARY = "binary"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class DiscoveryStatus(str, Enum):
    """发现状态枚举"""
    PENDING = "pending"                   # 等待发现
    SCANNING = "scanning"                 # 扫描中
    DISCOVERED = "discovered"             # 已发现
    INGESTING = "ingesting"               # 接入中
    INGESTED = "ingested"                 # 已接入
    FAILED = "failed"                     # 失败
    SKIPPED = "skipped"                   # 已跳过


class SyncStatus(str, Enum):
    """同步状态枚举"""
    DISABLED = "disabled"                 # 禁用
    ENABLED = "enabled"                   # 启用
    SYNCING = "syncing"                   # 同步中
    PAUSED = "paused"                     # 暂停
    ERROR = "error"                       # 错误


class SyncFrequency(str, Enum):
    """同步频率枚举"""
    REALTIME = "realtime"                 # 实时
    HOURLY = "hourly"                     # 每小时
    DAILY = "daily"                       # 每天
    WEEKLY = "weekly"                     # 每周
    MONTHLY = "monthly"                   # 每月
    CUSTOM = "custom"                     # 自定义


# ============================================================================
# 数据源扫描相关模型
# ============================================================================

class DataSourceCredentials(BaseModel):
    """数据源凭证模型"""
    username: Optional[str] = Field(None, description="用户名")
    password: Optional[str] = Field(None, description="密码")
    access_key: Optional[str] = Field(None, description="访问密钥")
    secret_key: Optional[str] = Field(None, description="密钥")
    token: Optional[str] = Field(None, description="令牌")
    certificate_path: Optional[str] = Field(None, description="证书路径")
    
    class Config:
        # 不在序列化时包含密码等敏感信息
        extra = "allow"


class DataSourceConfig(BaseModel):
    """数据源配置模型"""
    source_type: DataSourceType = Field(..., description="数据源类型")
    location: str = Field(..., description="数据源位置（路径/URL/连接字符串）")
    credentials: Optional[DataSourceCredentials] = Field(None, description="数据源凭证")
    
    # 文件系统相关配置
    include_patterns: Optional[List[str]] = Field(None, description="包含的文件模式")
    exclude_patterns: Optional[List[str]] = Field(None, description="排除的文件模式")
    recursive: bool = Field(True, description="是否递归扫描")
    
    # 数据库相关配置
    database_name: Optional[str] = Field(None, description="数据库名称")
    schema_name: Optional[str] = Field(None, description="模式名称")
    table_patterns: Optional[List[str]] = Field(None, description="表名模式")
    
    # 通用配置
    max_depth: int = Field(10, description="最大扫描深度")
    max_files: int = Field(1000, description="最大扫描文件数")
    timeout_seconds: int = Field(300, description="扫描超时时间（秒）")
    
    class Config:
        extra = "allow"


class ScanDataSourcesRequest(BaseModel):
    """扫描数据源请求模型"""
    sources: List[DataSourceConfig] = Field(..., description="要扫描的数据源列表")
    parallel_scan: bool = Field(True, description="是否并行扫描")
    include_preview: bool = Field(True, description="是否包含数据预览")
    preview_rows: int = Field(5, ge=1, le=100, description="预览行数")
    
    @validator('sources')
    def validate_sources(cls, v):
        if not v:
            raise ValueError("至少需要一个数据源配置")
        return v


class DataSourceInfo(BaseModel):
    """数据源信息模型"""
    source_id: str = Field(..., description="数据源ID")
    source_type: DataSourceType = Field(..., description="数据源类型")
    location: str = Field(..., description="数据源位置")
    status: DiscoveryStatus = Field(..., description="发现状态")
    
    # 发现结果
    files_found: Optional[int] = Field(None, description="发现的文件数")
    tables_found: Optional[int] = Field(None, description="发现的表数")
    total_size_bytes: Optional[int] = Field(None, description="总大小（字节）")
    
    # 时间信息
    scanned_at: Optional[datetime] = Field(None, description="扫描时间")
    scan_duration_ms: Optional[int] = Field(None, description="扫描耗时（毫秒）")
    
    # 错误信息
    error_message: Optional[str] = Field(None, description="错误信息")
    
    class Config:
        extra = "allow"


class ScanDataSourcesResponse(BaseModel):
    """扫描数据源响应模型"""
    scan_id: str = Field(..., description="扫描任务ID")
    sources: List[DataSourceInfo] = Field(..., description="数据源信息列表")
    total_scanned: int = Field(..., description="总扫描数")
    success_count: int = Field(..., description="成功数")
    failed_count: int = Field(..., description="失败数")
    scan_started_at: datetime = Field(..., description="扫描开始时间")
    scan_completed_at: Optional[datetime] = Field(None, description="扫描完成时间")


# ============================================================================
# 数据集发现相关模型
# ============================================================================

class DiscoveryConfig(BaseModel):
    """数据集发现配置模型"""
    source_ids: Optional[List[str]] = Field(None, description="指定数据源ID列表")
    auto_detect_format: bool = Field(True, description="自动检测数据格式")
    auto_detect_schema: bool = Field(True, description="自动检测数据模式")
    sample_size: int = Field(1000, ge=10, le=100000, description="采样大小")
    include_statistics: bool = Field(True, description="是否包含统计信息")
    detect_duplicates: bool = Field(True, description="是否检测重复数据")
    detect_anomalies: bool = Field(True, description="是否检测异常数据")


class ColumnInfo(BaseModel):
    """列信息模型"""
    name: str = Field(..., description="列名")
    data_type: str = Field(..., description="数据类型")
    nullable: bool = Field(True, description="是否可为空")
    unique: bool = Field(False, description="是否唯一")
    primary_key: bool = Field(False, description="是否主键")
    
    # 统计信息
    null_count: Optional[int] = Field(None, description="空值数量")
    null_percentage: Optional[float] = Field(None, description="空值百分比")
    unique_count: Optional[int] = Field(None, description="唯一值数量")
    
    # 数值类型统计
    min_value: Optional[Any] = Field(None, description="最小值")
    max_value: Optional[Any] = Field(None, description="最大值")
    mean_value: Optional[float] = Field(None, description="平均值")
    std_value: Optional[float] = Field(None, description="标准差")
    
    # 字符串类型统计
    min_length: Optional[int] = Field(None, description="最小长度")
    max_length: Optional[int] = Field(None, description="最大长度")
    avg_length: Optional[float] = Field(None, description="平均长度")
    
    # 样本值
    sample_values: Optional[List[Any]] = Field(None, description="样本值")


class DatasetPreview(BaseModel):
    """数据集预览模型"""
    columns: List[str] = Field(..., description="列名列表")
    sample_data: List[Dict[str, Any]] = Field(..., description="样本数据")
    total_rows: int = Field(..., description="总行数")
    preview_rows: int = Field(..., description="预览行数")


class DiscoveredDataset(BaseModel):
    """发现的数据集模型"""
    discovery_id: str = Field(..., description="发现ID")
    dataset_name: str = Field(..., description="数据集名称")
    source_id: str = Field(..., description="数据源ID")
    source_type: DataSourceType = Field(..., description="数据源类型")
    source_path: str = Field(..., description="数据源路径")
    
    # 数据信息
    format: DataFormat = Field(..., description="数据格式")
    size_bytes: int = Field(..., description="大小（字节）")
    row_count: Optional[int] = Field(None, description="行数")
    column_count: Optional[int] = Field(None, description="列数")
    
    # 模式信息
    schema_info: Optional[List[ColumnInfo]] = Field(None, description="模式信息")
    
    # 预览数据
    preview: Optional[DatasetPreview] = Field(None, description="预览数据")
    
    # 质量信息
    quality_score: Optional[float] = Field(None, ge=0, le=100, description="数据质量评分")
    completeness: Optional[float] = Field(None, ge=0, le=100, description="数据完整性")
    anomaly_count: Optional[int] = Field(None, description="异常数量")
    duplicate_count: Optional[int] = Field(None, description="重复数量")
    
    # 状态信息
    status: DiscoveryStatus = Field(..., description="发现状态")
    discovered_at: datetime = Field(..., description="发现时间")
    
    class Config:
        extra = "allow"


class DiscoverDatasetsRequest(BaseModel):
    """发现数据集请求模型"""
    config: DiscoveryConfig = Field(default_factory=DiscoveryConfig, description="发现配置")


class DiscoverDatasetsResponse(BaseModel):
    """发现数据集响应模型"""
    discovery_task_id: str = Field(..., description="发现任务ID")
    datasets: List[DiscoveredDataset] = Field(..., description="发现的数据集列表")
    total_discovered: int = Field(..., description="总发现数")
    discovery_started_at: datetime = Field(..., description="发现开始时间")
    discovery_completed_at: Optional[datetime] = Field(None, description="发现完成时间")


# ============================================================================
# 数据集接入相关模型
# ============================================================================

class IngestConfig(BaseModel):
    """数据集接入配置模型"""
    dataset_name: Optional[str] = Field(None, description="数据集名称（不填则自动生成）")
    description: Optional[str] = Field(None, description="数据集描述")
    dataset_type: str = Field("generic", description="数据集类型")
    
    # 存储配置
    storage_path: Optional[str] = Field(None, description="存储路径")
    storage_format: Optional[DataFormat] = Field(None, description="存储格式")
    compress: bool = Field(False, description="是否压缩")
    partition_by: Optional[List[str]] = Field(None, description="分区字段")
    
    # 转换配置
    apply_transformations: bool = Field(True, description="是否应用自动转换")
    normalize_columns: bool = Field(True, description="是否规范化列名")
    remove_duplicates: bool = Field(False, description="是否移除重复数据")
    handle_missing: str = Field("keep", description="缺失值处理方式：keep/drop/fill")
    
    # 同步配置
    enable_sync: bool = Field(False, description="是否启用同步")
    sync_frequency: SyncFrequency = Field(SyncFrequency.DAILY, description="同步频率")
    sync_config: Optional[Dict[str, Any]] = Field(None, description="同步配置详情")


class IngestDatasetRequest(BaseModel):
    """接入数据集请求模型"""
    discovery_id: Optional[str] = Field(None, description="发现ID（从发现结果接入）")
    source_config: Optional[DataSourceConfig] = Field(None, description="数据源配置（直接接入）")
    ingest_config: IngestConfig = Field(default_factory=IngestConfig, description="接入配置")
    
    @validator('source_config', always=True)
    def validate_source_or_discovery(cls, v, values):
        if not v and not values.get('discovery_id'):
            raise ValueError("必须提供 discovery_id 或 source_config")
        return v


class IngestDatasetResponse(BaseModel):
    """接入数据集响应模型"""
    dataset_id: str = Field(..., description="数据集ID")
    dataset_name: str = Field(..., description="数据集名称")
    status: str = Field(..., description="接入状态")
    storage_path: str = Field(..., description="存储路径")
    size_bytes: int = Field(..., description="大小（字节）")
    row_count: Optional[int] = Field(None, description="行数")
    ingested_at: datetime = Field(..., description="接入时间")
    sync_enabled: bool = Field(False, description="是否启用同步")


# ============================================================================
# 数据模式推断相关模型
# ============================================================================

class InferSchemaRequest(BaseModel):
    """推断模式请求模型"""
    sample_size: int = Field(1000, ge=10, le=100000, description="采样大小")
    detect_nested: bool = Field(True, description="是否检测嵌套结构")
    infer_constraints: bool = Field(True, description="是否推断约束")


class InferredSchema(BaseModel):
    """推断的模式模型"""
    dataset_id: str = Field(..., description="数据集ID")
    columns: List[ColumnInfo] = Field(..., description="列信息列表")
    row_count: int = Field(..., description="行数")
    
    # 推断元数据
    inferred_at: datetime = Field(..., description="推断时间")
    sample_size_used: int = Field(..., description="使用的采样大小")
    confidence_score: float = Field(..., ge=0, le=100, description="推断置信度")
    
    # 关系和约束
    primary_key_candidates: Optional[List[str]] = Field(None, description="主键候选列")
    foreign_key_hints: Optional[List[Dict[str, str]]] = Field(None, description="外键提示")
    index_recommendations: Optional[List[str]] = Field(None, description="索引建议")


# ============================================================================
# 数据转换相关模型
# ============================================================================

class TransformOperation(BaseModel):
    """转换操作模型"""
    operation_type: str = Field(..., description="操作类型")
    column_name: Optional[str] = Field(None, description="目标列名")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="操作参数")


class TransformConfig(BaseModel):
    """转换配置模型"""
    operations: List[TransformOperation] = Field(default_factory=list, description="转换操作列表")
    
    # 自动转换配置
    auto_normalize: bool = Field(True, description="自动规范化")
    auto_encode_categories: bool = Field(False, description="自动编码类别")
    auto_handle_missing: bool = Field(True, description="自动处理缺失值")
    auto_detect_datetime: bool = Field(True, description="自动检测日期时间")
    
    # 输出配置
    output_format: Optional[DataFormat] = Field(None, description="输出格式")
    create_new_dataset: bool = Field(False, description="是否创建新数据集")


class TransformDatasetRequest(BaseModel):
    """转换数据集请求模型"""
    config: TransformConfig = Field(default_factory=TransformConfig, description="转换配置")


class TransformResult(BaseModel):
    """转换结果模型"""
    dataset_id: str = Field(..., description="数据集ID")
    operations_applied: List[str] = Field(..., description="应用的操作列表")
    rows_affected: int = Field(..., description="影响的行数")
    columns_modified: List[str] = Field(..., description="修改的列列表")
    columns_added: List[str] = Field(default_factory=list, description="添加的列列表")
    columns_removed: List[str] = Field(default_factory=list, description="移除的列列表")
    transformed_at: datetime = Field(..., description="转换时间")
    
    # 新数据集（如果创建了）
    new_dataset_id: Optional[str] = Field(None, description="新数据集ID")


# ============================================================================
# 增量同步相关模型
# ============================================================================

class SyncConfig(BaseModel):
    """同步配置模型"""
    sync_enabled: bool = Field(True, description="是否启用同步")
    frequency: SyncFrequency = Field(SyncFrequency.DAILY, description="同步频率")
    
    # 增量策略
    incremental_column: Optional[str] = Field(None, description="增量列（用于检测变化）")
    incremental_method: str = Field("timestamp", description="增量方法：timestamp/id/hash")
    
    # 调度配置
    cron_expression: Optional[str] = Field(None, description="Cron表达式（自定义频率时使用）")
    timezone: str = Field("UTC", description="时区")
    
    # 冲突处理
    conflict_resolution: str = Field("update", description="冲突处理：update/skip/error")
    
    # 通知配置
    notify_on_success: bool = Field(False, description="成功时通知")
    notify_on_failure: bool = Field(True, description="失败时通知")
    notification_channels: List[str] = Field(default_factory=list, description="通知渠道")


class SetupSyncRequest(BaseModel):
    """设置同步请求模型"""
    config: SyncConfig = Field(..., description="同步配置")


class SyncStatusResponse(BaseModel):
    """同步状态响应模型"""
    dataset_id: str = Field(..., description="数据集ID")
    sync_status: SyncStatus = Field(..., description="同步状态")
    sync_config: SyncConfig = Field(..., description="同步配置")
    
    # 同步历史
    last_sync_at: Optional[datetime] = Field(None, description="最后同步时间")
    last_sync_status: Optional[str] = Field(None, description="最后同步状态")
    rows_synced: Optional[int] = Field(None, description="已同步行数")
    
    # 下次同步
    next_sync_at: Optional[datetime] = Field(None, description="下次同步时间")
    
    # 错误信息
    last_error: Optional[str] = Field(None, description="最后错误信息")


# ============================================================================
# 发现记录相关模型
# ============================================================================

class DiscoveryRecord(BaseModel):
    """发现记录模型"""
    record_id: str = Field(..., description="记录ID")
    user_id: str = Field(..., description="用户ID")
    tenant_id: Optional[str] = Field(None, description="租户ID")
    
    # 发现信息
    source_type: DataSourceType = Field(..., description="数据源类型")
    source_location: str = Field(..., description="数据源位置")
    discovery_status: DiscoveryStatus = Field(..., description="发现状态")
    
    # 结果信息
    datasets_discovered: int = Field(0, description="发现的数据集数")
    datasets_ingested: int = Field(0, description="已接入的数据集数")
    
    # 时间信息
    created_at: datetime = Field(..., description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    
    # 元数据
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class ListDiscoveriesRequest(BaseModel):
    """列出发现记录请求模型"""
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(20, ge=1, le=100, description="每页数量")
    status_filter: Optional[List[DiscoveryStatus]] = Field(None, description="状态过滤")
    source_type_filter: Optional[List[DataSourceType]] = Field(None, description="数据源类型过滤")
    date_from: Optional[datetime] = Field(None, description="开始日期")
    date_to: Optional[datetime] = Field(None, description="结束日期")


class ListDiscoveriesResponse(BaseModel):
    """列出发现记录响应模型"""
    records: List[DiscoveryRecord] = Field(..., description="发现记录列表")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页数量")
    total_pages: int = Field(..., description="总页数")

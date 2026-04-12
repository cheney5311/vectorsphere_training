#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型下载相关数据模型

定义模型下载相关的数据库模型和数据类：
- 下载记录
- 下载任务
- 临时下载链接
- 下载统计
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import uuid
import hashlib
import hmac
import base64

from sqlalchemy import (
    Column, String, Text, JSON, Boolean, Integer, Float, DateTime,
    Enum as SQLEnum, Index
)

from backend.schemas.base_models import Base, UUIDMixin, TimestampMixin, TenantMixin, GUID


# ==================== 枚举类型 ====================

class DownloadStatusEnum(str, Enum):
    """下载状态枚举"""
    PENDING = "pending"          # 等待下载
    PREPARING = "preparing"      # 准备中（转换格式等）
    READY = "ready"              # 就绪可下载
    DOWNLOADING = "downloading"  # 下载中
    COMPLETED = "completed"      # 下载完成
    FAILED = "failed"            # 下载失败
    EXPIRED = "expired"          # 链接已过期
    CANCELLED = "cancelled"      # 已取消


class DownloadFormatEnum(str, Enum):
    """下载格式枚举"""
    PYTORCH = "pytorch"           # .pt/.pth 格式
    ONNX = "onnx"                  # ONNX 格式
    TENSORFLOW = "tensorflow"      # TensorFlow SavedModel
    TORCHSCRIPT = "torchscript"   # TorchScript 格式
    SAFETENSORS = "safetensors"   # SafeTensors 格式
    TENSORRT = "tensorrt"         # TensorRT 格式
    COREML = "coreml"             # CoreML 格式
    CHECKPOINT = "checkpoint"     # 检查点格式
    WEIGHTS_ONLY = "weights_only" # 仅权重


class DownloadSourceEnum(str, Enum):
    """下载来源枚举"""
    MODEL = "model"               # 从模型直接下载
    TRAINING = "training"         # 从训练结果下载
    EXPORT = "export"             # 从导出结果下载
    VERSION = "version"           # 从特定版本下载


# ==================== SQLAlchemy 模型 ====================

class ModelDownloadRecordDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型下载记录数据库模型"""
    __tablename__ = 'model_download_records'
    
    # 关联信息
    model_id = Column(GUID(), nullable=False, index=True, comment="模型ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    version_id = Column(GUID(), index=True, comment="版本ID")
    training_session_id = Column(String(36), index=True, comment="训练会话ID")
    
    # 下载信息
    download_format = Column(String(50), default='pytorch', comment="下载格式")
    download_source = Column(String(50), default='model', comment="下载来源")
    status = Column(String(50), default='pending', index=True, comment="下载状态")
    
    # 文件信息
    file_path = Column(String(500), comment="文件路径")
    file_size = Column(Integer, comment="文件大小(字节)")
    file_name = Column(String(200), comment="文件名")
    checksum = Column(String(128), comment="文件校验和")
    mime_type = Column(String(100), comment="MIME类型")
    
    # 下载链接
    download_url = Column(String(1000), comment="下载URL")
    download_token = Column(String(256), index=True, comment="下载令牌")
    expire_at = Column(DateTime, index=True, comment="过期时间")
    
    # 下载统计
    download_count = Column(Integer, default=0, comment="下载次数")
    last_download_at = Column(DateTime, comment="最后下载时间")
    download_ip = Column(String(50), comment="下载IP")
    user_agent = Column(String(500), comment="用户代理")
    
    # 转换相关
    conversion_started_at = Column(DateTime, comment="转换开始时间")
    conversion_completed_at = Column(DateTime, comment="转换完成时间")
    conversion_error = Column(Text, comment="转换错误信息")
    
    # 元数据
    metadata_ = Column('metadata', JSON, default=dict, comment="元数据")
    
    # 索引
    __table_args__ = (
        Index('ix_download_records_model_user', 'model_id', 'user_id'),
        Index('ix_download_records_token_expire', 'download_token', 'expire_at'),
    )
    
    def __repr__(self):
        return f"<ModelDownloadRecordDB(model_id='{self.model_id}', format='{self.download_format}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'model_id': str(self.model_id) if self.model_id else None,
            'user_id': self.user_id,
            'version_id': str(self.version_id) if self.version_id else None,
            'training_session_id': self.training_session_id,
            'download_format': self.download_format,
            'download_source': self.download_source,
            'status': self.status,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'file_name': self.file_name,
            'checksum': self.checksum,
            'download_url': self.download_url,
            'expire_at': self.expire_at.isoformat() if self.expire_at else None,
            'download_count': self.download_count,
            'last_download_at': self.last_download_at.isoformat() if self.last_download_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class ModelDownloadStatisticsDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型下载统计数据库模型"""
    __tablename__ = 'model_download_statistics'
    
    # 关联信息
    model_id = Column(GUID(), nullable=False, index=True, comment="模型ID")
    
    # 统计周期
    stat_date = Column(DateTime, nullable=False, index=True, comment="统计日期")
    stat_type = Column(String(20), default='daily', comment="统计类型(daily/weekly/monthly)")
    
    # 下载统计
    total_downloads = Column(Integer, default=0, comment="总下载次数")
    unique_users = Column(Integer, default=0, comment="独立用户数")
    total_size_bytes = Column(Integer, default=0, comment="总下载大小(字节)")
    
    # 格式统计
    format_breakdown = Column(JSON, default=dict, comment="格式分布")
    
    # 来源统计
    source_breakdown = Column(JSON, default=dict, comment="来源分布")
    
    # 版本统计
    version_breakdown = Column(JSON, default=dict, comment="版本分布")
    
    # 时间分布
    hourly_distribution = Column(JSON, default=dict, comment="小时分布")
    
    __table_args__ = (
        Index('ix_download_stats_model_date', 'model_id', 'stat_date'),
    )
    
    def __repr__(self):
        return f"<ModelDownloadStatisticsDB(model_id='{self.model_id}', date='{self.stat_date}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'model_id': str(self.model_id) if self.model_id else None,
            'stat_date': self.stat_date.isoformat() if self.stat_date else None,
            'stat_type': self.stat_type,
            'total_downloads': self.total_downloads,
            'unique_users': self.unique_users,
            'total_size_bytes': self.total_size_bytes,
            'format_breakdown': self.format_breakdown,
            'source_breakdown': self.source_breakdown,
        }


# ==================== 数据类 ====================

@dataclass
class DownloadRequest:
    """下载请求数据类"""
    model_id: str
    user_id: str
    download_format: str = 'pytorch'
    version: Optional[str] = None
    training_session_id: Optional[str] = None
    expire_hours: int = 24
    tenant_id: Optional[str] = None
    
    def validate(self) -> List[str]:
        """验证请求参数"""
        errors = []
        if not self.model_id:
            errors.append("model_id is required")
        if not self.user_id:
            errors.append("user_id is required")
        
        valid_formats = ['pytorch', 'onnx', 'tensorflow', 'torchscript', 'safetensors', 'checkpoint']
        if self.download_format not in valid_formats:
            errors.append(f"Invalid format. Must be one of: {valid_formats}")
        
        if self.expire_hours < 1 or self.expire_hours > 168:  # 最长7天
            errors.append("expire_hours must be between 1 and 168")
        
        return errors


@dataclass
class DownloadInfo:
    """下载信息数据类"""
    model_id: str
    model_name: str
    version: str
    status: str
    available_formats: List[str] = field(default_factory=list)
    file_size: int = 0
    checksum: Optional[str] = None
    created_at: Optional[datetime] = None
    last_downloaded: Optional[datetime] = None
    download_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'version': self.version,
            'status': self.status,
            'available_formats': self.available_formats,
            'file_size': self.file_size,
            'checksum': self.checksum,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_downloaded': self.last_downloaded.isoformat() if self.last_downloaded else None,
            'download_count': self.download_count,
        }


@dataclass
class DownloadLink:
    """下载链接数据类"""
    download_id: str
    download_url: str
    download_token: str
    expire_at: datetime
    format: str
    file_name: str
    file_size: int = 0
    
    def is_expired(self) -> bool:
        """检查链接是否过期"""
        return datetime.utcnow() > self.expire_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'download_id': self.download_id,
            'download_url': self.download_url,
            'expire_at': self.expire_at.isoformat(),
            'format': self.format,
            'file_name': self.file_name,
            'file_size': self.file_size,
            'is_expired': self.is_expired(),
        }


@dataclass
class DownloadStatistics:
    """下载统计数据类"""
    model_id: str
    total_downloads: int = 0
    unique_users: int = 0
    total_size_bytes: int = 0
    
    format_breakdown: Dict[str, int] = field(default_factory=dict)
    source_breakdown: Dict[str, int] = field(default_factory=dict)
    daily_downloads: List[Dict[str, Any]] = field(default_factory=list)
    
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_id': self.model_id,
            'total_downloads': self.total_downloads,
            'unique_users': self.unique_users,
            'total_size_bytes': self.total_size_bytes,
            'total_size_mb': round(self.total_size_bytes / (1024 * 1024), 2) if self.total_size_bytes else 0,
            'format_breakdown': self.format_breakdown,
            'source_breakdown': self.source_breakdown,
            'daily_downloads': self.daily_downloads,
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None,
        }


# ==================== 工具函数 ====================

def generate_download_token(model_id: str, user_id: str, expire_at: datetime, secret_key: str = None) -> str:
    """生成下载令牌
    
    Args:
        model_id: 模型ID
        user_id: 用户ID
        expire_at: 过期时间
        secret_key: 密钥
        
    Returns:
        下载令牌
    """
    if secret_key is None:
        secret_key = "vectorsphere_download_secret_key"
    
    # 构建待签名数据
    data = f"{model_id}:{user_id}:{expire_at.timestamp()}"
    
    # 生成签名
    signature = hmac.new(
        secret_key.encode('utf-8'),
        data.encode('utf-8'),
        hashlib.sha256
    ).digest()
    
    # 编码为 base64
    token = base64.urlsafe_b64encode(signature).decode('utf-8').rstrip('=')
    
    return token


def verify_download_token(
    token: str,
    model_id: str,
    user_id: str,
    expire_at: datetime,
    secret_key: str = None
) -> bool:
    """验证下载令牌
    
    Args:
        token: 待验证令牌
        model_id: 模型ID
        user_id: 用户ID
        expire_at: 过期时间
        secret_key: 密钥
        
    Returns:
        是否有效
    """
    # 检查是否过期
    if datetime.utcnow() > expire_at:
        return False
    
    # 重新生成令牌并比较
    expected_token = generate_download_token(model_id, user_id, expire_at, secret_key)
    
    return hmac.compare_digest(token, expected_token)


def get_file_extension(format_type: str) -> str:
    """获取格式对应的文件扩展名
    
    Args:
        format_type: 格式类型
        
    Returns:
        文件扩展名
    """
    extension_map = {
        'pytorch': '.pt',
        'onnx': '.onnx',
        'tensorflow': '.pb',
        'torchscript': '.pt',
        'safetensors': '.safetensors',
        'tensorrt': '.trt',
        'coreml': '.mlmodel',
        'checkpoint': '.ckpt',
        'weights_only': '.bin',
    }
    return extension_map.get(format_type, '.bin')


def get_mime_type(format_type: str) -> str:
    """获取格式对应的MIME类型
    
    Args:
        format_type: 格式类型
        
    Returns:
        MIME类型
    """
    mime_map = {
        'pytorch': 'application/octet-stream',
        'onnx': 'application/octet-stream',
        'tensorflow': 'application/x-protobuf',
        'torchscript': 'application/octet-stream',
        'safetensors': 'application/octet-stream',
        'tensorrt': 'application/octet-stream',
        'coreml': 'application/octet-stream',
        'checkpoint': 'application/octet-stream',
        'weights_only': 'application/octet-stream',
    }
    return mime_map.get(format_type, 'application/octet-stream')


def calculate_file_checksum(file_path: str, algorithm: str = 'sha256') -> str:
    """计算文件校验和
    
    Args:
        file_path: 文件路径
        algorithm: 算法 (md5/sha256)
        
    Returns:
        校验和字符串
    """
    import os
    
    if not os.path.exists(file_path):
        return ""
    
    if algorithm == 'md5':
        hash_func = hashlib.md5()
    else:
        hash_func = hashlib.sha256()
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


# ==================== 导出 ====================

__all__ = [
    # 枚举
    'DownloadStatusEnum',
    'DownloadFormatEnum',
    'DownloadSourceEnum',
    
    # SQLAlchemy 模型
    'ModelDownloadRecordDB',
    'ModelDownloadStatisticsDB',
    
    # 数据类
    'DownloadRequest',
    'DownloadInfo',
    'DownloadLink',
    'DownloadStatistics',
    
    # 工具函数
    'generate_download_token',
    'verify_download_token',
    'get_file_extension',
    'get_mime_type',
    'calculate_file_checksum',
]

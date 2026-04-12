#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""嵌入向量数据模型

定义嵌入向量模块使用的数据类和数据库模型：
- 嵌入记录模型
- 嵌入模型配置
- 嵌入任务记录
- 相似度搜索记录

架构调用关系：
Repository层 (embedding_repository.py)
    -> 使用本模块定义的数据模型
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import uuid

from sqlalchemy import (
    Column, String, Text, JSON, Boolean, Integer, Float, DateTime,
    Enum as SQLEnum, ForeignKey, Index, LargeBinary
)
from sqlalchemy.dialects.postgresql import ARRAY

from backend.schemas.base_models import Base, UUIDMixin, TimestampMixin, TenantMixin, GUID


# ==================== 枚举类型 ====================

class EmbeddingModelTypeEnum(str, Enum):
    """嵌入模型类型枚举"""
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    TFIDF = "tfidf"
    WORD2VEC = "word2vec"
    FASTTEXT = "fasttext"
    BGE = "bge"
    M3E = "m3e"
    CUSTOM = "custom"


class EmbeddingTaskStatusEnum(str, Enum):
    """嵌入任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EmbeddingStorageTypeEnum(str, Enum):
    """嵌入存储类型枚举"""
    DATABASE = "database"
    VECTOR_DB = "vector_db"
    FILE = "file"
    MEMORY = "memory"


class SimilarityMetricEnum(str, Enum):
    """相似度计算方法枚举"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"


# ==================== SQLAlchemy 数据库模型 ====================

class EmbeddingRecordDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """嵌入记录数据库模型
    
    存储生成的嵌入向量及其元数据
    """
    __tablename__ = 'embedding_records'
    
    # 关联信息
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 原始内容
    content_hash = Column(String(64), nullable=False, index=True, comment="内容哈希值")
    content_type = Column(String(50), default='text', comment="内容类型: text, image, audio")
    content_preview = Column(Text, comment="内容预览(截断)")
    
    # 嵌入向量信息
    model_type = Column(String(50), nullable=False, index=True, comment="模型类型")
    model_name = Column(String(200), comment="模型名称")
    dimension = Column(Integer, nullable=False, comment="向量维度")
    
    # 嵌入向量存储（二进制存储以节省空间）
    embedding_blob = Column(LargeBinary, comment="嵌入向量(二进制)")
    
    # 元数据
    metadata_json = Column(JSON, default=dict, comment="元数据")
    
    # 统计信息
    processing_time_ms = Column(Float, comment="处理时间(毫秒)")
    token_count = Column(Integer, comment="令牌数量")
    
    # 来源信息
    source_type = Column(String(50), comment="来源类型")
    source_id = Column(String(36), comment="来源ID")
    
    __table_args__ = (
        Index('ix_emb_rec_user_model', 'user_id', 'model_type'),
        Index('ix_emb_rec_hash_model', 'content_hash', 'model_type'),
        {'extend_existing': True},
    )
    
    def __repr__(self):
        return f"<EmbeddingRecordDB(id='{self.id}', dim={self.dimension})>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'content_hash': self.content_hash,
            'content_type': self.content_type,
            'content_preview': self.content_preview,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'dimension': self.dimension,
            'metadata': self.metadata_json,
            'processing_time_ms': self.processing_time_ms,
            'token_count': self.token_count,
            'source_type': self.source_type,
            'source_id': self.source_id,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class EmbeddingModelConfigDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """嵌入模型配置数据库模型
    
    存储不同嵌入模型的配置信息
    """
    __tablename__ = 'embedding_model_configs'
    
    # 模型信息
    model_type = Column(String(50), nullable=False, index=True, comment="模型类型")
    model_name = Column(String(200), nullable=False, comment="模型名称")
    model_version = Column(String(50), default='1.0.0', comment="模型版本")
    
    # 配置信息
    dimension = Column(Integer, nullable=False, comment="输出维度")
    max_tokens = Column(Integer, default=512, comment="最大令牌数")
    normalize_output = Column(Boolean, default=True, comment="是否归一化输出")
    
    # API配置（用于远程模型）
    api_endpoint = Column(String(500), comment="API端点")
    api_key_encrypted = Column(String(500), comment="加密的API密钥")
    
    # 模型路径（用于本地模型）
    model_path = Column(String(500), comment="模型路径")
    
    # 性能配置
    batch_size = Column(Integer, default=32, comment="批处理大小")
    timeout_seconds = Column(Integer, default=30, comment="超时秒数")
    max_retries = Column(Integer, default=3, comment="最大重试次数")
    
    # 状态
    is_active = Column(Boolean, default=True, comment="是否激活")
    is_default = Column(Boolean, default=False, comment="是否默认模型")
    
    # 统计信息
    total_requests = Column(Integer, default=0, comment="总请求数")
    total_tokens = Column(Integer, default=0, comment="总令牌数")
    avg_latency_ms = Column(Float, default=0.0, comment="平均延迟(毫秒)")
    
    # 配置详情
    config_json = Column(JSON, default=dict, comment="详细配置")
    
    __table_args__ = (
        Index('ix_emb_model_type_name', 'model_type', 'model_name'),
        {'extend_existing': True},
    )
    
    def __repr__(self):
        return f"<EmbeddingModelConfigDB(model_type='{self.model_type}', name='{self.model_name}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'dimension': self.dimension,
            'max_tokens': self.max_tokens,
            'normalize_output': self.normalize_output,
            'batch_size': self.batch_size,
            'timeout_seconds': self.timeout_seconds,
            'is_active': self.is_active,
            'is_default': self.is_default,
            'total_requests': self.total_requests,
            'total_tokens': self.total_tokens,
            'avg_latency_ms': self.avg_latency_ms,
            'config': self.config_json,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class EmbeddingTaskDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """嵌入任务数据库模型
    
    记录批量嵌入生成任务
    """
    __tablename__ = 'embedding_tasks'
    
    # 关联信息
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 任务信息
    task_type = Column(String(50), default='batch_generate', comment="任务类型")
    status = Column(String(50), default='pending', index=True, comment="任务状态")
    priority = Column(Integer, default=5, comment="优先级(1-10)")
    
    # 模型配置
    model_type = Column(String(50), nullable=False, comment="模型类型")
    model_name = Column(String(200), comment="模型名称")
    
    # 输入信息
    total_items = Column(Integer, default=0, comment="总项数")
    processed_items = Column(Integer, default=0, comment="已处理项数")
    failed_items = Column(Integer, default=0, comment="失败项数")
    
    # 进度信息
    progress = Column(Float, default=0.0, comment="进度(0-100)")
    
    # 时间信息
    started_at = Column(DateTime, comment="开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    estimated_completion = Column(DateTime, comment="预计完成时间")
    
    # 资源使用
    total_tokens = Column(Integer, default=0, comment="总令牌数")
    processing_time_ms = Column(Float, default=0.0, comment="处理时间(毫秒)")
    
    # 错误信息
    error_message = Column(Text, comment="错误信息")
    error_details = Column(JSON, comment="错误详情")
    
    # 结果信息
    result_summary = Column(JSON, default=dict, comment="结果摘要")
    
    __table_args__ = (
        Index('ix_emb_task_user_status', 'user_id', 'status'),
        {'extend_existing': True},
    )
    
    def __repr__(self):
        return f"<EmbeddingTaskDB(id='{self.id}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'task_type': self.task_type,
            'status': self.status,
            'priority': self.priority,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'total_items': self.total_items,
            'processed_items': self.processed_items,
            'failed_items': self.failed_items,
            'progress': self.progress,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'total_tokens': self.total_tokens,
            'processing_time_ms': self.processing_time_ms,
            'error_message': self.error_message,
            'result_summary': self.result_summary,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class SimilaritySearchRecordDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """相似度搜索记录数据库模型
    
    记录相似度搜索历史
    """
    __tablename__ = 'similarity_search_records'
    
    # 关联信息
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 查询信息
    query_hash = Column(String(64), comment="查询哈希值")
    query_preview = Column(Text, comment="查询预览")
    
    # 搜索配置
    model_type = Column(String(50), comment="模型类型")
    similarity_metric = Column(String(50), default='cosine', comment="相似度计算方法")
    top_k = Column(Integer, default=10, comment="返回数量")
    threshold = Column(Float, comment="相似度阈值")
    
    # 结果信息
    result_count = Column(Integer, default=0, comment="结果数量")
    results_json = Column(JSON, default=list, comment="结果列表")
    
    # 性能信息
    search_time_ms = Column(Float, comment="搜索时间(毫秒)")
    
    __table_args__ = (
        Index('ix_sim_search_user_time', 'user_id', 'created_at'),
        {'extend_existing': True},
    )
    
    def __repr__(self):
        return f"<SimilaritySearchRecordDB(id='{self.id}', results={self.result_count})>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'query_hash': self.query_hash,
            'query_preview': self.query_preview,
            'model_type': self.model_type,
            'similarity_metric': self.similarity_metric,
            'top_k': self.top_k,
            'threshold': self.threshold,
            'result_count': self.result_count,
            'results': self.results_json,
            'search_time_ms': self.search_time_ms,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


# ==================== 数据类 ====================

@dataclass
class EmbeddingRequest:
    """嵌入请求数据类"""
    text: str
    model_type: str = "sentence-transformers"
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """验证请求参数"""
        errors = []
        if not self.text:
            errors.append("文本内容不能为空")
        if not self.model_type:
            errors.append("模型类型不能为空")
        return errors


@dataclass
class BatchEmbeddingRequest:
    """批量嵌入请求数据类"""
    texts: List[str]
    model_type: str = "sentence-transformers"
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """验证请求参数"""
        errors = []
        if not self.texts:
            errors.append("文本列表不能为空")
        if len(self.texts) > 1000:
            errors.append("单次批量请求最多支持1000条文本")
        if not self.model_type:
            errors.append("模型类型不能为空")
        return errors


@dataclass
class EmbeddingResult:
    """嵌入结果数据类"""
    embedding: List[float]
    dimension: int
    model_type: str
    processing_time_ms: float = 0.0
    token_count: int = 0
    cached: bool = False
    record_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'embedding': self.embedding,
            'dimension': self.dimension,
            'modelType': self.model_type,
            'processingTimeMs': self.processing_time_ms,
            'tokenCount': self.token_count,
            'cached': self.cached,
            'recordId': self.record_id,
        }


@dataclass
class BatchEmbeddingResult:
    """批量嵌入结果数据类"""
    embeddings: List[List[float]]
    dimension: int
    model_type: str
    total_count: int
    success_count: int
    failed_count: int
    processing_time_ms: float = 0.0
    total_tokens: int = 0
    task_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'embeddings': self.embeddings,
            'dimension': self.dimension,
            'modelType': self.model_type,
            'totalCount': self.total_count,
            'successCount': self.success_count,
            'failedCount': self.failed_count,
            'processingTimeMs': self.processing_time_ms,
            'totalTokens': self.total_tokens,
            'taskId': self.task_id,
        }


@dataclass
class SimilarityResult:
    """相似度结果数据类"""
    similarity: float
    metric: str = "cosine"
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'similarity': self.similarity,
            'metric': self.metric,
            'processingTimeMs': self.processing_time_ms,
        }


@dataclass
class SimilaritySearchResult:
    """相似度搜索结果数据类"""
    query: str
    results: List[Dict[str, Any]]
    total_count: int
    search_time_ms: float = 0.0
    model_type: str = "sentence-transformers"
    metric: str = "cosine"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'results': self.results,
            'totalCount': self.total_count,
            'searchTimeMs': self.search_time_ms,
            'modelType': self.model_type,
            'metric': self.metric,
        }


@dataclass
class EmbeddingModelInfo:
    """嵌入模型信息数据类"""
    model_type: str
    model_name: str
    dimension: int
    max_tokens: int
    is_active: bool = True
    is_default: bool = False
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'modelType': self.model_type,
            'modelName': self.model_name,
            'dimension': self.dimension,
            'maxTokens': self.max_tokens,
            'isActive': self.is_active,
            'isDefault': self.is_default,
            'description': self.description,
        }


@dataclass
class EmbeddingStats:
    """嵌入统计数据类"""
    total_embeddings: int = 0
    total_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    avg_processing_time_ms: float = 0.0
    total_processing_time_ms: float = 0.0
    cache_size: int = 0
    models_used: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'totalEmbeddings': self.total_embeddings,
            'totalTokens': self.total_tokens,
            'cacheHits': self.cache_hits,
            'cacheMisses': self.cache_misses,
            'cacheHitRate': self.cache_hit_rate,
            'avgProcessingTimeMs': self.avg_processing_time_ms,
            'totalProcessingTimeMs': self.total_processing_time_ms,
            'cacheSize': self.cache_size,
            'modelsUsed': self.models_used,
        }


# ==================== 工具函数 ====================

def compute_content_hash(content: str, model_type: str = "default") -> str:
    """计算内容哈希值
    
    Args:
        content: 文本内容
        model_type: 模型类型
        
    Returns:
        SHA256哈希值
    """
    import hashlib
    hash_input = f"{model_type}:{content}"
    return hashlib.sha256(hash_input.encode('utf-8')).hexdigest()


def estimate_token_count(text: str, model_type: str = "default") -> int:
    """估算令牌数量
    
    Args:
        text: 文本内容
        model_type: 模型类型
        
    Returns:
        估算的令牌数量
    """
    # 简化的令牌估算：中文按字符计算，英文按空格分词
    chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
    english_words = len([word for word in text.split() if word.isascii()])
    
    # 中文约1字符=1令牌，英文约1词=1.3令牌
    estimated_tokens = chinese_chars + int(english_words * 1.3)
    return max(estimated_tokens, 1)


def truncate_text(text: str, max_length: int = 100) -> str:
    """截断文本用于预览
    
    Args:
        text: 原始文本
        max_length: 最大长度
        
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


# ==================== 导出 ====================

__all__ = [
    # 枚举
    'EmbeddingModelTypeEnum',
    'EmbeddingTaskStatusEnum',
    'EmbeddingStorageTypeEnum',
    'SimilarityMetricEnum',
    
    # SQLAlchemy 模型
    'EmbeddingRecordDB',
    'EmbeddingModelConfigDB',
    'EmbeddingTaskDB',
    'SimilaritySearchRecordDB',
    
    # 数据类
    'EmbeddingRequest',
    'BatchEmbeddingRequest',
    'EmbeddingResult',
    'BatchEmbeddingResult',
    'SimilarityResult',
    'SimilaritySearchResult',
    'EmbeddingModelInfo',
    'EmbeddingStats',
    
    # 工具函数
    'compute_content_hash',
    'estimate_token_count',
    'truncate_text',
]

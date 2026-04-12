#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型管理相关数据模型

定义模型管理模块使用的数据类和数据库模型：
- 模型性能记录
- 模型验证结果
- 模型导入记录
- 模型比较结果
- 模型训练历史
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from sqlalchemy import (
    Column, String, Text, JSON, Boolean, Integer, Float, DateTime,
    Enum as SQLEnum, ForeignKey, Index
)
from sqlalchemy.dialects.postgresql import UUID

from backend.schemas.base_models import Base, UUIDMixin, TimestampMixin, TenantMixin


# ==================== 枚举类型 ====================

class ValidationStatusEnum(str, Enum):
    """验证状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ImportStatusEnum(str, Enum):
    """导入状态枚举"""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class ImportSourceEnum(str, Enum):
    """导入来源枚举"""
    LOCAL = "local"
    URL = "url"
    CLOUD = "cloud"
    HUGGINGFACE = "huggingface"
    S3 = "s3"
    GCS = "gcs"


class PerformanceMetricTypeEnum(str, Enum):
    """性能指标类型枚举"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    LOSS = "loss"
    AUC = "auc"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    GPU_UTILIZATION = "gpu_utilization"


# ==================== SQLAlchemy 数据库模型 ====================

class ModelPerformanceRecordDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型性能记录数据库模型"""
    __tablename__ = 'model_performance_records'
    
    # 关联信息
    model_id = Column(String(36), nullable=False, index=True, comment="模型ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 核心性能指标
    accuracy = Column(Float, comment="准确率")
    precision = Column(Float, comment="精确率")
    recall = Column(Float, comment="召回率")
    f1_score = Column(Float, comment="F1分数")
    loss = Column(Float, comment="损失值")
    auc = Column(Float, comment="AUC值")
    
    # 回归指标
    mae = Column(Float, comment="平均绝对误差")
    mse = Column(Float, comment="均方误差")
    rmse = Column(Float, comment="均方根误差")
    
    # 运行时性能
    training_time_seconds = Column(Float, comment="训练时间(秒)")
    inference_time_ms = Column(Float, comment="推理时间(毫秒)")
    throughput_per_second = Column(Float, comment="每秒吞吐量")
    
    # 资源使用
    memory_usage_mb = Column(Float, comment="内存使用量(MB)")
    gpu_memory_mb = Column(Float, comment="GPU内存使用量(MB)")
    cpu_utilization = Column(Float, comment="CPU利用率(%)")
    gpu_utilization = Column(Float, comment="GPU利用率(%)")
    
    # 评估数据信息
    test_data_size = Column(Integer, comment="测试数据大小")
    evaluation_time = Column(DateTime, comment="评估时间")
    
    # 自定义指标
    custom_metrics = Column(JSON, default=dict, comment="自定义指标")
    
    # 元数据
    description = Column(Text, comment="描述")
    evaluation_config = Column(JSON, default=dict, comment="评估配置")
    
    __table_args__ = (
        Index('ix_model_perf_model_time', 'model_id', 'evaluation_time'),
    )
    
    def __repr__(self):
        return f"<ModelPerformanceRecordDB(model_id='{self.model_id}', accuracy={self.accuracy})>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'model_id': self.model_id,
            'user_id': self.user_id,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'loss': self.loss,
            'auc': self.auc,
            'mae': self.mae,
            'mse': self.mse,
            'rmse': self.rmse,
            'training_time_seconds': self.training_time_seconds,
            'inference_time_ms': self.inference_time_ms,
            'throughput_per_second': self.throughput_per_second,
            'memory_usage_mb': self.memory_usage_mb,
            'gpu_memory_mb': self.gpu_memory_mb,
            'cpu_utilization': self.cpu_utilization,
            'gpu_utilization': self.gpu_utilization,
            'test_data_size': self.test_data_size,
            'evaluation_time': self.evaluation_time.isoformat() if self.evaluation_time else None,
            'custom_metrics': self.custom_metrics,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class ModelValidationResultDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型验证结果数据库模型"""
    __tablename__ = 'model_validation_results'
    
    # 关联信息
    model_id = Column(String(36), nullable=False, index=True, comment="模型ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 验证状态
    status = Column(String(50), default='pending', index=True, comment="验证状态")
    
    # 验证指标
    accuracy = Column(Float, comment="准确率")
    precision = Column(Float, comment="精确率")
    recall = Column(Float, comment="召回率")
    f1_score = Column(Float, comment="F1分数")
    loss = Column(Float, comment="损失值")
    
    # 验证详情
    validation_time = Column(DateTime, comment="验证时间")
    test_data_size = Column(Integer, comment="测试数据大小")
    passed_tests = Column(Integer, default=0, comment="通过的测试数")
    failed_tests = Column(Integer, default=0, comment="失败的测试数")
    
    # 验证配置
    validation_config = Column(JSON, default=dict, comment="验证配置")
    
    # 详细结果
    test_results = Column(JSON, default=list, comment="测试结果详情")
    confusion_matrix = Column(JSON, comment="混淆矩阵")
    classification_report = Column(JSON, comment="分类报告")
    
    # 错误信息
    error_message = Column(Text, comment="错误信息")
    
    __table_args__ = (
        Index('ix_model_val_model_status', 'model_id', 'status'),
    )
    
    def __repr__(self):
        return f"<ModelValidationResultDB(model_id='{self.model_id}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'model_id': self.model_id,
            'user_id': self.user_id,
            'status': self.status,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'loss': self.loss,
            'validation_time': self.validation_time.isoformat() if self.validation_time else None,
            'test_data_size': self.test_data_size,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'validation_config': self.validation_config,
            'test_results': self.test_results,
            'confusion_matrix': self.confusion_matrix,
            'classification_report': self.classification_report,
            'error_message': self.error_message,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class ModelImportRecordDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型导入记录数据库模型"""
    __tablename__ = 'model_import_records'
    
    # 关联信息
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    target_model_id = Column(String(36), index=True, comment="导入后的模型ID")
    
    # 导入状态
    status = Column(String(50), default='pending', index=True, comment="导入状态")
    
    # 导入来源信息
    import_source = Column(String(50), default='local', comment="导入来源")
    source_path = Column(String(1000), comment="来源路径")
    source_url = Column(String(1000), comment="来源URL")
    
    # 模型信息
    model_name = Column(String(200), comment="模型名称")
    model_type = Column(String(50), comment="模型类型")
    model_framework = Column(String(50), comment="模型框架")
    model_format = Column(String(50), comment="模型格式")
    
    # 文件信息
    file_size_bytes = Column(Integer, comment="文件大小(字节)")
    checksum = Column(String(128), comment="校验和")
    
    # 导入配置
    import_config = Column(JSON, default=dict, comment="导入配置")
    
    # 处理进度
    progress = Column(Float, default=0.0, comment="处理进度(0-100)")
    
    # 时间信息
    started_at = Column(DateTime, comment="开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    
    # 错误信息
    error_message = Column(Text, comment="错误信息")
    
    # 导入结果
    import_result = Column(JSON, default=dict, comment="导入结果")
    
    def __repr__(self):
        return f"<ModelImportRecordDB(model_name='{self.model_name}', status='{self.status}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'target_model_id': self.target_model_id,
            'status': self.status,
            'import_source': self.import_source,
            'source_path': self.source_path,
            'source_url': self.source_url,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'model_framework': self.model_framework,
            'model_format': self.model_format,
            'file_size_bytes': self.file_size_bytes,
            'checksum': self.checksum,
            'import_config': self.import_config,
            'progress': self.progress,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'import_result': self.import_result,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class ModelComparisonRecordDB(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """模型比较记录数据库模型"""
    __tablename__ = 'model_comparison_records'
    
    # 关联信息
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    
    # 比较的模型
    model_ids = Column(JSON, nullable=False, comment="比较的模型ID列表")
    
    # 比较配置
    comparison_config = Column(JSON, default=dict, comment="比较配置")
    metrics_to_compare = Column(JSON, default=list, comment="比较的指标列表")
    
    # 比较结果
    comparison_result = Column(JSON, default=dict, comment="比较结果")
    winner_model_id = Column(String(36), comment="胜出模型ID")
    
    # 比较详情
    comparison_time = Column(DateTime, default=datetime.utcnow, comment="比较时间")
    description = Column(Text, comment="比较描述")
    
    def __repr__(self):
        return f"<ModelComparisonRecordDB(user_id='{self.user_id}', models={self.model_ids})>"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'model_ids': self.model_ids,
            'comparison_config': self.comparison_config,
            'metrics_to_compare': self.metrics_to_compare,
            'comparison_result': self.comparison_result,
            'winner_model_id': self.winner_model_id,
            'comparison_time': self.comparison_time.isoformat() if self.comparison_time else None,
            'description': self.description,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


# ==================== 数据类 ====================

@dataclass
class ModelPerformanceMetrics:
    """模型性能指标数据类"""
    model_id: str
    model_name: str
    
    # 核心指标
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    loss: Optional[float] = None
    auc: Optional[float] = None
    
    # 回归指标
    mae: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    
    # 运行时性能
    training_time_seconds: Optional[float] = None
    inference_time_ms: Optional[float] = None
    throughput_per_second: Optional[float] = None
    
    # 资源使用
    memory_usage_mb: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    
    # 元数据
    evaluation_time: Optional[datetime] = None
    test_data_size: Optional[int] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'modelId': self.model_id,
            'modelName': self.model_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1Score': self.f1_score,
            'loss': self.loss,
            'auc': self.auc,
            'mae': self.mae,
            'mse': self.mse,
            'rmse': self.rmse,
            'trainingTime': self.training_time_seconds,
            'inferenceTimeMs': self.inference_time_ms,
            'throughputPerSecond': self.throughput_per_second,
            'memoryUsageMb': self.memory_usage_mb,
            'gpuMemoryMb': self.gpu_memory_mb,
            'evaluationTime': self.evaluation_time.isoformat() if self.evaluation_time else None,
            'testDataSize': self.test_data_size,
            'customMetrics': self.custom_metrics,
        }
    
    def validate(self) -> List[str]:
        """验证数据"""
        errors = []
        if not self.model_id:
            errors.append("模型ID不能为空")
        if not self.model_name:
            errors.append("模型名称不能为空")
        return errors


@dataclass
class ModelValidationRequest:
    """模型验证请求数据类"""
    model_id: str
    user_id: str
    test_data: Any = None
    test_data_path: Optional[str] = None
    validation_config: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """验证请求参数"""
        errors = []
        if not self.model_id:
            errors.append("模型ID不能为空")
        if not self.user_id:
            errors.append("用户ID不能为空")
        if self.test_data is None and not self.test_data_path:
            errors.append("测试数据或测试数据路径不能同时为空")
        return errors


@dataclass
class ModelValidationResult:
    """模型验证结果数据类"""
    model_id: str
    model_name: str
    status: str = "pending"
    
    # 核心指标
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    loss: Optional[float] = None
    
    # 验证详情
    validation_time: Optional[datetime] = None
    test_data_size: Optional[int] = None
    passed_tests: int = 0
    failed_tests: int = 0
    
    # 详细结果
    test_results: List[Dict[str, Any]] = field(default_factory=list)
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None
    
    # 错误信息
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'modelId': self.model_id,
            'modelName': self.model_name,
            'status': self.status,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1Score': self.f1_score,
            'loss': self.loss,
            'validationTime': self.validation_time.isoformat() if self.validation_time else None,
            'testDataSize': self.test_data_size,
            'passedTests': self.passed_tests,
            'failedTests': self.failed_tests,
            'testResults': self.test_results,
            'confusionMatrix': self.confusion_matrix,
            'classificationReport': self.classification_report,
            'errorMessage': self.error_message,
        }


@dataclass
class ModelImportRequest:
    """模型导入请求数据类"""
    user_id: str
    model_name: str
    import_source: str = "local"
    source_path: Optional[str] = None
    source_url: Optional[str] = None
    model_type: Optional[str] = None
    model_framework: Optional[str] = None
    import_config: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """验证请求参数"""
        errors = []
        if not self.user_id:
            errors.append("用户ID不能为空")
        if not self.model_name:
            errors.append("模型名称不能为空")
        if self.import_source in ['local', 's3', 'gcs'] and not self.source_path:
            errors.append("本地/云存储导入需要提供来源路径")
        if self.import_source in ['url', 'huggingface'] and not self.source_url:
            errors.append("URL/HuggingFace导入需要提供来源URL")
        return errors


@dataclass
class ModelImportResult:
    """模型导入结果数据类"""
    import_id: str
    status: str = "pending"
    model_name: Optional[str] = None
    target_model_id: Optional[str] = None
    
    # 模型信息
    model_type: Optional[str] = None
    model_framework: Optional[str] = None
    model_format: Optional[str] = None
    file_size_bytes: Optional[int] = None
    parameters_count: Optional[int] = None
    
    # 时间信息
    import_time: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 进度
    progress: float = 0.0
    
    # 错误信息
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'importId': self.import_id,
            'status': self.status,
            'modelName': self.model_name,
            'targetModelId': self.target_model_id,
            'modelType': self.model_type,
            'modelFramework': self.model_framework,
            'modelFormat': self.model_format,
            'fileSizeBytes': self.file_size_bytes,
            'parametersCount': self.parameters_count,
            'importTime': self.import_time.isoformat() if self.import_time else None,
            'startedAt': self.started_at.isoformat() if self.started_at else None,
            'completedAt': self.completed_at.isoformat() if self.completed_at else None,
            'progress': self.progress,
            'errorMessage': self.error_message,
        }


@dataclass
class ModelComparisonRequest:
    """模型比较请求数据类"""
    user_id: str
    model_ids: List[str]
    metrics_to_compare: List[str] = field(default_factory=lambda: ['accuracy', 'precision', 'recall', 'f1_score', 'loss'])
    comparison_config: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """验证请求参数"""
        errors = []
        if not self.user_id:
            errors.append("用户ID不能为空")
        if not self.model_ids or len(self.model_ids) < 2:
            errors.append("至少需要2个模型进行比较")
        if len(self.model_ids) > 10:
            errors.append("最多比较10个模型")
        return errors


@dataclass
class ModelComparisonResult:
    """模型比较结果数据类"""
    models: List[Dict[str, Any]] = field(default_factory=list)
    comparison_metrics: List[Dict[str, Any]] = field(default_factory=list)
    winner_model_id: Optional[str] = None
    comparison_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'models': self.models,
            'comparisonMetrics': self.comparison_metrics,
            'winnerModelId': self.winner_model_id,
            'comparisonTime': self.comparison_time.isoformat() if self.comparison_time else None,
        }


@dataclass
class TrainingHistoryItem:
    """训练历史记录项数据类"""
    session_id: str
    model_name: str
    status: str
    
    # 时间信息
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    
    # 最终指标
    final_loss: Optional[float] = None
    final_accuracy: Optional[float] = None
    
    # 训练配置
    config: Dict[str, Any] = field(default_factory=dict)
    
    # 错误信息
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sessionId': self.session_id,
            'modelName': self.model_name,
            'status': self.status,
            'startTime': self.start_time.isoformat() if self.start_time else None,
            'endTime': self.end_time.isoformat() if self.end_time else None,
            'duration': self.duration_seconds,
            'finalLoss': self.final_loss,
            'finalAccuracy': self.final_accuracy,
            'config': self.config,
            'errorMessage': self.error_message,
        }


# ==================== 工具函数 ====================

def calculate_comparison_score(metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """计算模型综合得分
    
    Args:
        metrics: 指标字典
        weights: 权重字典（可选）
        
    Returns:
        综合得分
    """
    if not weights:
        weights = {
            'accuracy': 0.3,
            'precision': 0.2,
            'recall': 0.2,
            'f1_score': 0.2,
            'loss': -0.1,  # 负权重，loss越低越好
        }
    
    score = 0.0
    for metric, weight in weights.items():
        value = metrics.get(metric)
        if value is not None:
            score += value * weight
    
    return score


def determine_winner(models_metrics: List[Dict[str, Any]], metric: str = 'accuracy') -> Optional[str]:
    """确定指标最优的模型
    
    Args:
        models_metrics: 模型指标列表
        metric: 用于比较的指标名称
        
    Returns:
        胜出模型的ID
    """
    if not models_metrics:
        return None
    
    # 对于loss，值越小越好
    reverse = metric != 'loss'
    
    sorted_models = sorted(
        models_metrics,
        key=lambda x: x.get('metrics', {}).get(metric) or 0,
        reverse=reverse
    )
    
    if sorted_models:
        return sorted_models[0].get('id')
    
    return None


# ==================== 导出 ====================

__all__ = [
    # 枚举
    'ValidationStatusEnum',
    'ImportStatusEnum',
    'ImportSourceEnum',
    'PerformanceMetricTypeEnum',
    
    # SQLAlchemy 模型
    'ModelPerformanceRecordDB',
    'ModelValidationResultDB',
    'ModelImportRecordDB',
    'ModelComparisonRecordDB',
    
    # 数据类
    'ModelPerformanceMetrics',
    'ModelValidationRequest',
    'ModelValidationResult',
    'ModelImportRequest',
    'ModelImportResult',
    'ModelComparisonRequest',
    'ModelComparisonResult',
    'TrainingHistoryItem',
    
    # 工具函数
    'calculate_comparison_score',
    'determine_winner',
]

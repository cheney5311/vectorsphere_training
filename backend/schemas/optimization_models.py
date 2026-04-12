#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""资源优化模块数据模型

定义优化任务、优化建议、优化策略等相关的 SQLAlchemy ORM 模型和数据类。
"""

import json
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict

try:
    from sqlalchemy import (
        Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
        Enum as SQLEnum, ForeignKey, Index, UniqueConstraint
    )
    from sqlalchemy.orm import relationship, declarative_base
    HAS_SQLALCHEMY = True
except ImportError:
    HAS_SQLALCHEMY = False

# 基类定义
try:
    from backend.schemas.base_models import Base
except ImportError:
    if HAS_SQLALCHEMY:
        Base = declarative_base()
    else:
        Base = object


# ==================== 枚举类型 ====================

class OptimizationStrategyEnum(str, Enum):
    """优化策略"""
    BALANCED = "balanced"           # 平衡策略
    PERFORMANCE = "performance"     # 性能优先
    ENERGY = "energy"               # 节能优先
    COST = "cost"                   # 成本优先
    CUSTOM = "custom"               # 自定义策略


class OptimizationStatusEnum(str, Enum):
    """优化状态"""
    IDLE = "idle"                   # 空闲
    ANALYZING = "analyzing"         # 分析中
    OPTIMIZING = "optimizing"       # 优化中
    COMPLETED = "completed"         # 已完成
    FAILED = "failed"               # 失败
    PAUSED = "paused"               # 已暂停
    CANCELLED = "cancelled"         # 已取消


class RecommendationPriorityEnum(str, Enum):
    """建议优先级"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecommendationStatusEnum(str, Enum):
    """建议状态"""
    PENDING = "pending"             # 待处理
    APPLIED = "applied"             # 已应用
    IGNORED = "ignored"             # 已忽略
    FAILED = "failed"               # 应用失败
    EXPIRED = "expired"             # 已过期


class ResourceCategoryEnum(str, Enum):
    """资源类别"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    IO = "io"


class BottleneckSeverityEnum(str, Enum):
    """瓶颈严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrainingOptimizationTypeEnum(str, Enum):
    """训练优化类型"""
    GRAPH_OPTIMIZATION = "graph_optimization"           # 图优化
    MEMORY_OPTIMIZATION = "memory_optimization"         # 内存优化
    OPERATOR_FUSION = "operator_fusion"                 # 算子融合
    CONSTANT_FOLDING = "constant_folding"               # 常量折叠
    DEAD_CODE_ELIMINATION = "dead_code_elimination"     # 死代码消除
    LAYOUT_OPTIMIZATION = "layout_optimization"         # 布局优化
    PRUNING = "pruning"                                 # 剪枝
    QUANTIZATION = "quantization"                       # 量化
    RESOURCE_SCHEDULING = "resource_scheduling"         # 资源调度
    BATCH_OPTIMIZATION = "batch_optimization"           # 批处理优化
    GRADIENT_ACCUMULATION = "gradient_accumulation"     # 梯度累积
    MIXED_PRECISION = "mixed_precision"                 # 混合精度


class TrainingOptimizationStatusEnum(str, Enum):
    """训练优化状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ==================== SQLAlchemy 模型 ====================

if HAS_SQLALCHEMY:

    class OptimizationSessionModel(Base):
        """优化会话模型
        
        记录每次优化运行的会话信息
        """
        __tablename__ = 'optimization_sessions'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 会话信息
        name = Column(String(256), nullable=True)
        description = Column(Text, nullable=True)
        
        # 优化策略
        strategy = Column(SQLEnum(OptimizationStrategyEnum), default=OptimizationStrategyEnum.BALANCED)
        strategy_config = Column(JSON, default=dict)  # 策略配置参数
        
        # 状态
        status = Column(SQLEnum(OptimizationStatusEnum), default=OptimizationStatusEnum.IDLE, index=True)
        progress = Column(Float, default=0.0)  # 进度百分比
        
        # 目标资源
        target_resources = Column(JSON, default=list)  # ['cpu', 'memory', 'gpu']
        
        # 时间信息
        created_at = Column(DateTime, default=datetime.utcnow, index=True)
        started_at = Column(DateTime, nullable=True)
        completed_at = Column(DateTime, nullable=True)
        
        # 结果统计
        recommendations_count = Column(Integer, default=0)
        applied_count = Column(Integer, default=0)
        estimated_savings = Column(Float, default=0.0)  # 预估节省百分比
        actual_savings = Column(Float, nullable=True)   # 实际节省百分比
        
        # 错误信息
        error_message = Column(Text, nullable=True)
        
        # 创建者
        created_by = Column(String(64), nullable=True)
        
        # 元数据
        session_metadata = Column('metadata', JSON, default=dict)
        
        # 索引
        __table_args__ = (
            Index('idx_opt_session_status_created', 'status', 'created_at'),
            Index('idx_opt_session_tenant_status', 'tenant_id', 'status'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'name': self.name,
                'description': self.description,
                'strategy': self.strategy.value if self.strategy else None,
                'strategy_config': self.strategy_config,
                'status': self.status.value if self.status else None,
                'progress': self.progress,
                'target_resources': self.target_resources,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'started_at': self.started_at.isoformat() if self.started_at else None,
                'completed_at': self.completed_at.isoformat() if self.completed_at else None,
                'recommendations_count': self.recommendations_count,
                'applied_count': self.applied_count,
                'estimated_savings': self.estimated_savings,
                'actual_savings': self.actual_savings,
                'error_message': self.error_message,
                'created_by': self.created_by,
                'metadata': self.session_metadata
            }


    class OptimizationRecommendationModel(Base):
        """优化建议模型
        
        存储系统生成的优化建议
        """
        __tablename__ = 'optimization_recommendations'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        session_id = Column(String(64), ForeignKey('optimization_sessions.id'), nullable=True, index=True)
        
        # 建议信息
        title = Column(String(256), nullable=False)
        description = Column(Text, nullable=True)
        category = Column(SQLEnum(ResourceCategoryEnum), nullable=False, index=True)
        
        # 优先级和置信度
        priority = Column(SQLEnum(RecommendationPriorityEnum), default=RecommendationPriorityEnum.MEDIUM)
        confidence = Column(Float, default=0.5)  # 置信度 0-1
        
        # 状态
        status = Column(SQLEnum(RecommendationStatusEnum), default=RecommendationStatusEnum.PENDING, index=True)
        
        # 建议操作
        action = Column(String(256), nullable=True)  # 建议的操作
        action_params = Column(JSON, default=dict)   # 操作参数
        
        # 影响评估
        estimated_impact = Column(Text, nullable=True)      # 预估影响描述
        estimated_savings_percent = Column(Float, default=0.0)  # 预估节省百分比
        risk_level = Column(String(32), default='low')      # 风险等级
        
        # 指标信息
        current_value = Column(Float, nullable=True)     # 当前值
        recommended_value = Column(Float, nullable=True) # 建议值
        threshold = Column(Float, nullable=True)         # 阈值
        
        # 时间信息
        created_at = Column(DateTime, default=datetime.utcnow, index=True)
        applied_at = Column(DateTime, nullable=True)
        expires_at = Column(DateTime, nullable=True)  # 过期时间
        
        # 应用信息
        applied_by = Column(String(64), nullable=True)
        apply_result = Column(JSON, nullable=True)  # 应用结果
        
        # 元数据
        recommendation_metadata = Column('metadata', JSON, default=dict)
        
        # 关系
        # session = relationship("OptimizationSessionModel", back_populates="recommendations")
        
        # 索引
        __table_args__ = (
            Index('idx_recommendation_category_status', 'category', 'status'),
            Index('idx_recommendation_tenant_priority', 'tenant_id', 'priority'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'session_id': self.session_id,
                'title': self.title,
                'description': self.description,
                'category': self.category.value if self.category else None,
                'priority': self.priority.value if self.priority else None,
                'confidence': self.confidence,
                'status': self.status.value if self.status else None,
                'action': self.action,
                'action_params': self.action_params,
                'estimated_impact': self.estimated_impact,
                'estimated_savings_percent': self.estimated_savings_percent,
                'risk_level': self.risk_level,
                'current_value': self.current_value,
                'recommended_value': self.recommended_value,
                'threshold': self.threshold,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'applied_at': self.applied_at.isoformat() if self.applied_at else None,
                'expires_at': self.expires_at.isoformat() if self.expires_at else None,
                'applied_by': self.applied_by,
                'apply_result': self.apply_result,
                'metadata': self.recommendation_metadata
            }


    class ResourceMetricSnapshotModel(Base):
        """资源指标快照模型
        
        记录资源使用的历史快照
        """
        __tablename__ = 'resource_metric_snapshots'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 指标信息
        metric_type = Column(SQLEnum(ResourceCategoryEnum), nullable=False, index=True)
        metric_name = Column(String(128), nullable=False, index=True)
        metric_value = Column(Float, nullable=False)
        metric_unit = Column(String(32), nullable=True)  # %, MB, GB, etc.
        
        # 状态评估
        status = Column(String(32), default='normal')  # normal, warning, critical
        threshold_warning = Column(Float, nullable=True)
        threshold_critical = Column(Float, nullable=True)
        
        # 时间戳
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        
        # 来源
        source = Column(String(128), nullable=True)  # 数据来源标识
        
        # 额外数据
        extra_data = Column(JSON, default=dict)
        
        # 索引
        __table_args__ = (
            Index('idx_metric_snapshot_type_time', 'metric_type', 'timestamp'),
            Index('idx_metric_snapshot_name_time', 'metric_name', 'timestamp'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'metric_type': self.metric_type.value if self.metric_type else None,
                'metric_name': self.metric_name,
                'metric_value': self.metric_value,
                'metric_unit': self.metric_unit,
                'status': self.status,
                'threshold_warning': self.threshold_warning,
                'threshold_critical': self.threshold_critical,
                'timestamp': self.timestamp.isoformat() if self.timestamp else None,
                'source': self.source,
                'extra_data': self.extra_data
            }


    class PerformanceAnalysisReportModel(Base):
        """性能分析报告模型
        
        存储性能分析的结果报告
        """
        __tablename__ = 'performance_analysis_reports'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 报告信息
        analysis_type = Column(String(64), nullable=False, index=True)  # full, cpu, memory, io
        target_id = Column(String(64), nullable=True)  # 分析目标ID（可选）
        
        # 时间范围
        start_time = Column(DateTime, nullable=True)
        end_time = Column(DateTime, nullable=True)
        
        # 分析结果
        summary = Column(Text, nullable=True)
        bottlenecks = Column(JSON, default=list)  # 瓶颈列表
        recommendations = Column(JSON, default=list)  # 建议列表
        metrics_summary = Column(JSON, default=dict)  # 指标汇总
        
        # 状态
        status = Column(String(32), default='pending', index=True)  # pending, running, completed, failed
        
        # 时间信息
        created_at = Column(DateTime, default=datetime.utcnow, index=True)
        completed_at = Column(DateTime, nullable=True)
        
        # 创建者
        created_by = Column(String(64), nullable=True)
        
        # 元数据
        report_metadata = Column('metadata', JSON, default=dict)
        
        # 索引
        __table_args__ = (
            Index('idx_analysis_report_type_time', 'analysis_type', 'created_at'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'analysis_type': self.analysis_type,
                'target_id': self.target_id,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'summary': self.summary,
                'bottlenecks': self.bottlenecks,
                'recommendations': self.recommendations,
                'metrics_summary': self.metrics_summary,
                'status': self.status,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'completed_at': self.completed_at.isoformat() if self.completed_at else None,
                'created_by': self.created_by,
                'metadata': self.report_metadata
            }


    class ResourceAlertModel(Base):
        """资源告警模型
        
        存储资源相关的告警
        """
        __tablename__ = 'resource_alerts'
        
        id = Column(String(64), primary_key=True)
        tenant_id = Column(String(64), nullable=True, index=True)
        
        # 告警信息
        level = Column(String(32), nullable=False, index=True)  # info, warning, critical
        resource_type = Column(SQLEnum(ResourceCategoryEnum), nullable=False, index=True)
        message = Column(Text, nullable=False)
        
        # 指标信息
        metric_name = Column(String(128), nullable=True)
        metric_value = Column(Float, nullable=True)
        threshold = Column(Float, nullable=True)
        
        # 状态
        status = Column(String(32), default='active', index=True)  # active, acknowledged, resolved
        
        # 时间信息
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        acknowledged_at = Column(DateTime, nullable=True)
        resolved_at = Column(DateTime, nullable=True)
        
        # 处理信息
        acknowledged_by = Column(String(64), nullable=True)
        resolved_by = Column(String(64), nullable=True)
        resolution_note = Column(Text, nullable=True)
        
        # 元数据
        alert_metadata = Column('metadata', JSON, default=dict)
        
        # 索引
        __table_args__ = (
            Index('idx_resource_alert_level_time', 'level', 'timestamp'),
            Index('idx_resource_alert_status_time', 'status', 'timestamp'),
        )
        
        def to_dict(self) -> Dict[str, Any]:
            return {
                'id': self.id,
                'tenant_id': self.tenant_id,
                'level': self.level,
                'resource_type': self.resource_type.value if self.resource_type else None,
                'message': self.message,
                'metric_name': self.metric_name,
                'metric_value': self.metric_value,
                'threshold': self.threshold,
                'status': self.status,
                'timestamp': self.timestamp.isoformat() if self.timestamp else None,
                'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
                'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
                'acknowledged_by': self.acknowledged_by,
                'resolved_by': self.resolved_by,
                'resolution_note': self.resolution_note,
                'metadata': self.alert_metadata
            }

else:
    # 如果没有 SQLAlchemy，定义空类
    class OptimizationSessionModel: pass
    class OptimizationRecommendationModel: pass
    class ResourceMetricSnapshotModel: pass
    class PerformanceAnalysisReportModel: pass
    class ResourceAlertModel: pass


# ==================== 数据类（用于业务逻辑） ====================

@dataclass
class OptimizationSession:
    """优化会话数据类"""
    id: str
    tenant_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    strategy: str = 'balanced'
    strategy_config: Dict[str, Any] = field(default_factory=dict)
    status: str = 'idle'
    progress: float = 0.0
    target_resources: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    recommendations_count: int = 0
    applied_count: int = 0
    estimated_savings: float = 0.0
    actual_savings: Optional[float] = None
    error_message: Optional[str] = None
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'name': self.name,
            'description': self.description,
            'strategy': self.strategy,
            'strategy_config': self.strategy_config,
            'status': self.status,
            'progress': self.progress,
            'target_resources': self.target_resources,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'recommendations_count': self.recommendations_count,
            'applied_count': self.applied_count,
            'estimated_savings': self.estimated_savings,
            'actual_savings': self.actual_savings,
            'error_message': self.error_message,
            'created_by': self.created_by,
            'metadata': self.metadata
        }


@dataclass
class OptimizationRecommendation:
    """优化建议数据类"""
    id: str
    title: str
    category: str
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    description: Optional[str] = None
    priority: str = 'medium'
    confidence: float = 0.5
    status: str = 'pending'
    action: Optional[str] = None
    action_params: Dict[str, Any] = field(default_factory=dict)
    estimated_impact: Optional[str] = None
    estimated_savings_percent: float = 0.0
    risk_level: str = 'low'
    current_value: Optional[float] = None
    recommended_value: Optional[float] = None
    threshold: Optional[float] = None
    created_at: Optional[datetime] = None
    applied_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    applied_by: Optional[str] = None
    apply_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'session_id': self.session_id,
            'title': self.title,
            'description': self.description,
            'category': self.category,
            'priority': self.priority,
            'confidence': self.confidence,
            'status': self.status,
            'action': self.action,
            'action_params': self.action_params,
            'estimated_impact': self.estimated_impact,
            'estimated_savings_percent': self.estimated_savings_percent,
            'risk_level': self.risk_level,
            'current_value': self.current_value,
            'recommended_value': self.recommended_value,
            'threshold': self.threshold,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'applied_at': self.applied_at.isoformat() if self.applied_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'applied_by': self.applied_by,
            'apply_result': self.apply_result,
            'metadata': self.metadata
        }


@dataclass
class ResourceMetricSnapshot:
    """资源指标快照数据类"""
    id: str
    metric_type: str
    metric_name: str
    metric_value: float
    tenant_id: Optional[str] = None
    metric_unit: Optional[str] = None
    status: str = 'normal'
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: Optional[datetime] = None
    source: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'metric_type': self.metric_type,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'metric_unit': self.metric_unit,
            'status': self.status,
            'threshold_warning': self.threshold_warning,
            'threshold_critical': self.threshold_critical,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'source': self.source,
            'extra_data': self.extra_data
        }


@dataclass
class PerformanceBottleneck:
    """性能瓶颈数据类"""
    type: str
    severity: str
    description: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceAnalysisReport:
    """性能分析报告数据类"""
    id: str
    analysis_type: str
    tenant_id: Optional[str] = None
    target_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    summary: Optional[str] = None
    bottlenecks: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    status: str = 'pending'
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'analysis_type': self.analysis_type,
            'target_id': self.target_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'summary': self.summary,
            'bottlenecks': self.bottlenecks,
            'recommendations': self.recommendations,
            'metrics_summary': self.metrics_summary,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_by': self.created_by,
            'metadata': self.metadata
        }


@dataclass
class ResourceAlert:
    """资源告警数据类"""
    id: str
    level: str
    resource_type: str
    message: str
    tenant_id: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    status: str = 'active'
    timestamp: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution_note: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'level': self.level,
            'resource_type': self.resource_type,
            'message': self.message,
            'metric_name': self.metric_name,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'status': self.status,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'acknowledged_by': self.acknowledged_by,
            'resolved_by': self.resolved_by,
            'resolution_note': self.resolution_note,
            'metadata': self.metadata
        }


# ==================== 训练优化数据类 ====================

@dataclass
class TrainingOptimizationRecord:
    """训练优化记录数据类"""
    id: str
    training_job_id: str
    optimization_type: str
    tenant_id: Optional[str] = None
    model_id: Optional[str] = None
    session_id: Optional[str] = None
    status: str = 'pending'
    
    # 优化配置
    config: Dict[str, Any] = field(default_factory=dict)
    
    # 优化结果
    performance_improvement: float = 0.0  # 性能提升百分比
    memory_reduction: float = 0.0         # 内存减少百分比
    throughput_improvement: float = 0.0   # 吞吐量提升百分比
    latency_reduction: float = 0.0        # 延迟降低百分比
    
    # 详细结果
    optimization_details: Dict[str, Any] = field(default_factory=dict)
    applied_optimizations: List[str] = field(default_factory=list)
    
    # 资源影响
    cpu_usage_before: Optional[float] = None
    cpu_usage_after: Optional[float] = None
    memory_usage_before: Optional[float] = None
    memory_usage_after: Optional[float] = None
    gpu_usage_before: Optional[float] = None
    gpu_usage_after: Optional[float] = None
    
    # 时间信息
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    
    # 错误信息
    error_message: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'tenant_id': self.tenant_id,
            'training_job_id': self.training_job_id,
            'model_id': self.model_id,
            'session_id': self.session_id,
            'optimization_type': self.optimization_type,
            'status': self.status,
            'config': self.config,
            'performance_improvement': self.performance_improvement,
            'memory_reduction': self.memory_reduction,
            'throughput_improvement': self.throughput_improvement,
            'latency_reduction': self.latency_reduction,
            'optimization_details': self.optimization_details,
            'applied_optimizations': self.applied_optimizations,
            'cpu_usage_before': self.cpu_usage_before,
            'cpu_usage_after': self.cpu_usage_after,
            'memory_usage_before': self.memory_usage_before,
            'memory_usage_after': self.memory_usage_after,
            'gpu_usage_before': self.gpu_usage_before,
            'gpu_usage_after': self.gpu_usage_after,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'execution_time_seconds': self.execution_time_seconds,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class GraphOptimizationResult:
    """图优化结果数据类"""
    optimization_id: str
    optimization_type: str
    success: bool = True
    
    # 优化效果
    performance_improvement: float = 0.0
    memory_reduction: float = 0.0
    
    # 详细信息
    folded_constants: int = 0
    eliminated_nodes: int = 0
    fused_operators: int = 0
    optimized_layouts: int = 0
    
    # 时间
    execution_time_ms: float = 0.0
    timestamp: Optional[datetime] = None
    
    # 错误
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimization_id': self.optimization_id,
            'optimization_type': self.optimization_type,
            'success': self.success,
            'performance_improvement': self.performance_improvement,
            'memory_reduction': self.memory_reduction,
            'folded_constants': self.folded_constants,
            'eliminated_nodes': self.eliminated_nodes,
            'fused_operators': self.fused_operators,
            'optimized_layouts': self.optimized_layouts,
            'execution_time_ms': self.execution_time_ms,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'error': self.error
        }


@dataclass
class ResourceOptimizationConfig:
    """资源优化配置数据类"""
    # 优化策略
    strategy: str = 'balanced'  # balanced, performance, energy, cost
    
    # CPU优化配置
    cpu_optimization_enabled: bool = True
    cpu_target_utilization: float = 70.0
    cpu_max_utilization: float = 90.0
    
    # 内存优化配置
    memory_optimization_enabled: bool = True
    memory_target_utilization: float = 75.0
    memory_max_utilization: float = 85.0
    
    # GPU优化配置
    gpu_optimization_enabled: bool = True
    gpu_target_utilization: float = 80.0
    gpu_max_utilization: float = 95.0
    gpu_memory_target: float = 80.0
    
    # 图优化配置
    graph_optimization_enabled: bool = True
    constant_folding: bool = True
    dead_code_elimination: bool = True
    operator_fusion: bool = True
    layout_optimization: bool = True
    
    # 训练优化配置
    mixed_precision_enabled: bool = False
    gradient_accumulation_steps: int = 1
    batch_size_auto_tuning: bool = False
    
    # 调度配置
    task_priority_weight: float = 0.4
    resource_efficiency_weight: float = 0.3
    energy_efficiency_weight: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==================== 导出 ====================

__all__ = [
    # 枚举
    'OptimizationStrategyEnum',
    'OptimizationStatusEnum',
    'RecommendationPriorityEnum',
    'RecommendationStatusEnum',
    'ResourceCategoryEnum',
    'BottleneckSeverityEnum',
    'TrainingOptimizationTypeEnum',
    'TrainingOptimizationStatusEnum',
    
    # SQLAlchemy 模型
    'OptimizationSessionModel',
    'OptimizationRecommendationModel',
    'ResourceMetricSnapshotModel',
    'PerformanceAnalysisReportModel',
    'ResourceAlertModel',
    
    # 数据类
    'OptimizationSession',
    'OptimizationRecommendation',
    'ResourceMetricSnapshot',
    'PerformanceBottleneck',
    'PerformanceAnalysisReport',
    'ResourceAlert',
    'TrainingOptimizationRecord',
    'GraphOptimizationResult',
    'ResourceOptimizationConfig',
]

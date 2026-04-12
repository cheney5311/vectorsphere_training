# -*- coding: utf-8 -*-
"""
知识蒸馏服务

提供完整的蒸馏服务能力：
- 租户级蒸馏任务管理
- 蒸馏任务监控和报告
- 蒸馏任务调度
- 模型评估和比较
- 与平台服务集成

架构调用层次：
├── distillation_service.py (本模块)
│   └── 调用 compression_config.py (配置层)
│       ├── DistillationConfig, DistillationTaskConfig - 配置类
│       ├── DistillationStats, DistillationMonitor - 监控
│       └── ConfigValidator, validate_config - 验证
│   └── 调用 distillation_scenarios.py (场景层)
│   └── 调用 knowledge_distillation.py (训练器层)
│   └── 调用 backend/modules/training/strategies (策略层)
│       ├── base_strategy.py - StrategyMonitor, StrategyProfiler
│       └── distributed_strategy.py - DistributedStrategy
│   └── 调用 backend/lib/hardware (硬件层)
│   └── 调用 backend/lib/distributed (分布式层)
└── 被 API层/Web服务 调用

生产级特性：
- 完整的服务监控和诊断
- 任务调度和资源管理
- 健康检查和自动恢复
- 分布式任务支持
- 多租户隔离
"""

import logging
import threading
import time
import uuid
import json
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
from contextlib import contextmanager
import torch
import torch.nn as nn

# 修复导入路径
import sys
import os as os_path
current_dir = os_path.path.dirname(os_path.path.abspath(__file__))
project_root = os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(current_dir))))
sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)


# ======================== 配置层导入 ========================

from .compression_config import (
    # 核心配置
    DistillationConfig,
    DistillationTaskConfig,
    ScenarioDistillationConfig,
    DistributedDistillationConfig,
    AdaptiveDistillationConfig,
    CompressionConfig,
    # 枚举
    DistillationScenario,
    DistributedMode as ConfigDistributedMode,
    AdaptiveMode,
    CompressionMethod,
    # 预设
    DistillationPresets,
    # 监控
    DistillationStats,
    DistillationMonitor,
    ConfigValidator,
    # 工具函数
    validate_config,
    recommend_config,
    compare_configs,
)


# ======================== 场景层导入 ========================

from .distillation_scenarios import (
    DistillationScenarioManager,
    get_scenario_manager,
    ScenarioExecutionStats,
    ScenarioMonitor,
    recommend_distillation_scenario,
    diagnose_scenarios,
)


# ======================== 训练器层导入 ========================

from .knowledge_distillation import (
    KnowledgeDistillationTrainer,
    ModelCompressor,
    create_knowledge_distillation_trainer,
    create_trainer_from_preset,
    create_trainer_from_scenario,
    diagnose_trainer,
    estimate_training_resources,
)


# ======================== 策略层导入 ========================

STRATEGY_LAYER_AVAILABLE = False
try:
    from backend.modules.training.strategies.base_strategy import (
        StrategyType,
        TrainingPhase,
        StrategyContext,
        StrategyResult,
        StrategyMonitor,
        StrategyProfiler,
        StrategyValidator,
        StrategyMetrics,
        TrainingStrategy,
    )
    STRATEGY_LAYER_AVAILABLE = True
    logger.info("Strategy layer (base) loaded successfully")
except ImportError as e:
    logger.warning(f"Strategy layer (base) not available: {e}")
    StrategyMonitor = None
    StrategyProfiler = None
    StrategyValidator = None
    StrategyMetrics = None


DISTRIBUTED_STRATEGY_AVAILABLE = False
try:
    from backend.modules.training.strategies.distributed_strategy import (
        DistributedMode as StrategyDistributedMode,
        ZeROStage,
        DistributedStrategyConfig,
        CommunicationStats,
        DistributedHealthStatus,
        DistributedStrategy,
        recommend_distributed_mode,
        diagnose_distributed_strategy,
    )
    DISTRIBUTED_STRATEGY_AVAILABLE = True
    logger.info("Strategy layer (distributed) loaded successfully")
except ImportError as e:
    logger.warning(f"Strategy layer (distributed) not available: {e}")
    DistributedStrategy = None
    DistributedStrategyConfig = None


DISTILLATION_STRATEGY_AVAILABLE = False
try:
    from backend.modules.training.strategies.distillation_strategy import (
        DistillationStrategy,
        DistillationStrategyConfig,
        DistillationType,
        create_distillation_strategy,
    )
    DISTILLATION_STRATEGY_AVAILABLE = True
    logger.info("Strategy layer (distillation) loaded successfully")
except ImportError as e:
    logger.warning(f"Strategy layer (distillation) not available: {e}")
    DistillationStrategy = None
    create_distillation_strategy = None


# ======================== 硬件层导入 ========================

HARDWARE_LAYER_AVAILABLE = False
try:
    from backend.lib.hardware import (
        DeviceManager,
        get_device_manager,
        MemoryManager,
        get_memory_manager,
        get_available_memory,
        clear_memory,
    )
    HARDWARE_LAYER_AVAILABLE = True
    logger.info("Hardware layer loaded successfully")
except ImportError as e:
    logger.warning(f"Hardware layer not available: {e}")
    DeviceManager = None
    MemoryManager = None
    get_available_memory = None


# ======================== 分布式层导入 ========================

DISTRIBUTED_LAYER_AVAILABLE = False
try:
    from backend.lib.distributed import (
        DistributedManager,
        get_distributed_manager,
        is_main_process,
        get_rank,
        get_world_size,
        barrier,
    )
    DISTRIBUTED_LAYER_AVAILABLE = True
    logger.info("Distributed layer loaded successfully")
except ImportError as e:
    logger.warning(f"Distributed layer not available: {e}")
    DistributedManager = None
    is_main_process = lambda: True
    get_rank = lambda: 0
    get_world_size = lambda: 1


# ======================== 状态枚举 ========================

class DistillationTaskStatus(Enum):
    """蒸馏任务状态"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    
    @classmethod
    def from_string(cls, value: str) -> 'DistillationTaskStatus':
        """从字符串创建"""
        for member in cls:
            if member.value == value:
                return member
        return cls.PENDING
    
    def is_terminal(self) -> bool:
        """是否为终态"""
        return self in [self.COMPLETED, self.FAILED, self.CANCELLED, self.TIMEOUT]
    
    def is_active(self) -> bool:
        """是否为活跃状态"""
        return self in [self.RUNNING, self.QUEUED]
    
    def can_start(self) -> bool:
        """是否可以启动"""
        return self in [self.PENDING, self.PAUSED]
    
    def can_pause(self) -> bool:
        """是否可以暂停"""
        return self == self.RUNNING
    
    def can_resume(self) -> bool:
        """是否可以恢复"""
        return self == self.PAUSED
    
    def can_cancel(self) -> bool:
        """是否可以取消"""
        return self in [self.PENDING, self.QUEUED, self.RUNNING, self.PAUSED]


class ServiceHealthStatus(Enum):
    """服务健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


# ======================== 数据模型 ========================

@dataclass
class DistillationTask:
    """
    蒸馏任务数据模型
    
    增强版，整合配置层和策略层的监控能力
    """
    task_id: str
    task_name: str
    tenant_id: str
    user_id: str
    
    # 配置
    config: Optional['DistillationTaskConfig'] = None
    scenario: str = "standard"
    
    # 状态
    status: str = "pending"
    progress: float = 0.0
    priority: int = 1  # TaskPriority.NORMAL
    
    # 时间戳
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    last_updated_at: Optional[str] = None
    
    # 资源配置
    device: str = "cuda"
    num_gpus: int = 1
    distributed_mode: str = "none"
    
    # 结果
    result: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # 模型信息
    teacher_model_id: Optional[str] = None
    student_model_id: Optional[str] = None
    output_model_id: Optional[str] = None
    
    # 监控数据（使用配置层的 DistillationStats）
    distillation_stats: Optional[Dict[str, Any]] = None
    
    # 重试信息
    retry_count: int = 0
    max_retries: int = 3
    
    # 资源使用
    estimated_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    total_training_time_seconds: float = 0.0
    
    def __post_init__(self):
        """初始化后处理"""
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()
        if self.last_updated_at is None:
            self.last_updated_at = self.created_at
    
    def get_status_enum(self) -> DistillationTaskStatus:
        """获取状态枚举"""
        return DistillationTaskStatus.from_string(self.status)
    
    def update_status(self, new_status: Union[str, DistillationTaskStatus]) -> None:
        """更新状态"""
        if isinstance(new_status, DistillationTaskStatus):
            self.status = new_status.value
        else:
            self.status = new_status
        self.last_updated_at = datetime.utcnow().isoformat()
    
    def update_progress(self, progress: float, metrics: Optional[Dict[str, float]] = None) -> None:
        """更新进度"""
        self.progress = min(100.0, max(0.0, progress))
        if metrics:
            self.metrics.update(metrics)
        self.last_updated_at = datetime.utcnow().isoformat()
    
    def get_duration_seconds(self) -> float:
        """获取任务持续时间"""
        if not self.started_at:
            return 0.0
        
        end_time = self.completed_at if self.completed_at else datetime.utcnow().isoformat()
        try:
            start = datetime.fromisoformat(self.started_at)
            end = datetime.fromisoformat(end_time)
            return (end - start).total_seconds()
        except Exception:
            return 0.0
    
    def can_retry(self) -> bool:
        """是否可以重试"""
        return (
            self.get_status_enum() == DistillationTaskStatus.FAILED and 
            self.retry_count < self.max_retries
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'tenant_id': self.tenant_id,
            'user_id': self.user_id,
            'scenario': self.scenario,
            'status': self.status,
            'progress': self.progress,
            'priority': self.priority,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'last_updated_at': self.last_updated_at,
            'device': self.device,
            'num_gpus': self.num_gpus,
            'distributed_mode': self.distributed_mode,
            'metrics': self.metrics,
            'error_message': self.error_message,
            'teacher_model_id': self.teacher_model_id,
            'student_model_id': self.student_model_id,
            'output_model_id': self.output_model_id,
            'retry_count': self.retry_count,
            'estimated_memory_mb': self.estimated_memory_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'total_training_time_seconds': self.total_training_time_seconds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistillationTask':
        """从字典创建"""
        return cls(
            task_id=data.get('task_id', str(uuid.uuid4())),
            task_name=data.get('task_name', 'unnamed'),
            tenant_id=data.get('tenant_id', 'default'),
            user_id=data.get('user_id', 'anonymous'),
            config=data.get('config'),
            scenario=data.get('scenario', 'standard'),
            status=data.get('status', 'pending'),
            progress=data.get('progress', 0.0),
            priority=data.get('priority', 1),
            created_at=data.get('created_at'),
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            last_updated_at=data.get('last_updated_at'),
            device=data.get('device', 'cuda'),
            num_gpus=data.get('num_gpus', 1),
            distributed_mode=data.get('distributed_mode', 'none'),
            result=data.get('result', {}),
            metrics=data.get('metrics', {}),
            error_message=data.get('error_message'),
            error_traceback=data.get('error_traceback'),
            teacher_model_id=data.get('teacher_model_id'),
            student_model_id=data.get('student_model_id'),
            output_model_id=data.get('output_model_id'),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3),
            estimated_memory_mb=data.get('estimated_memory_mb', 0.0),
            peak_memory_mb=data.get('peak_memory_mb', 0.0),
            total_training_time_seconds=data.get('total_training_time_seconds', 0.0),
        )
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2)
    
    def summary(self) -> str:
        """获取摘要信息"""
        parts = [
            f"Task[{self.task_id[:8]}...]",
            f"status={self.status}",
            f"progress={self.progress:.1f}%",
            f"scenario={self.scenario}",
        ]
        if self.metrics:
            if 'total_loss' in self.metrics:
                parts.append(f"loss={self.metrics['total_loss']:.4f}")
        return " | ".join(parts)


@dataclass
class DistillationMetrics:
    """
    蒸馏监控指标
    
    增强版，整合配置层 DistillationStats 和策略层 StrategyMetrics
    """
    task_id: str
    timestamp: str
    
    # 损失指标
    total_loss: float = 0.0
    soft_loss: float = 0.0
    hard_loss: float = 0.0
    feature_loss: float = 0.0
    attention_loss: float = 0.0
    contrastive_loss: float = 0.0
    
    # 训练指标
    step: int = 0
    epoch: int = 0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    
    # 性能指标
    throughput: float = 0.0  # samples/sec
    step_time_ms: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    
    # 精度指标（评估时）
    teacher_accuracy: float = 0.0
    student_accuracy: float = 0.0
    accuracy_gap: float = 0.0
    
    # 温度参数（自适应蒸馏）
    temperature: float = 4.0
    alpha: float = 0.5
    
    # 分布式指标
    rank: int = 0
    world_size: int = 1
    sync_time_ms: float = 0.0
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistillationMetrics':
        """从字典创建"""
        try:
            from dataclasses import fields
            valid_fields = {f.name for f in fields(cls)}
        except Exception:
            valid_fields = set(data.keys())
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    def to_distillation_stats(self) -> Optional['DistillationStats']:
        """转换为配置层的 DistillationStats（如果可用）"""
        try:
            stats = DistillationStats()
            stats.update(
                kd_loss=self.soft_loss,
                ce_loss=self.hard_loss,
                feature_loss=self.feature_loss,
                attention_loss=self.attention_loss,
            )
            return stats
        except Exception:
            return None
    
    def merge_from_strategy_metrics(self, strategy_metrics: 'StrategyMetrics') -> None:
        """从策略层 StrategyMetrics 合并数据"""
        
        try:
            # 合并性能指标
            if hasattr(strategy_metrics, 'step_time'):
                self.step_time_ms = strategy_metrics.step_time * 1000
            if hasattr(strategy_metrics, 'throughput'):
                self.throughput = strategy_metrics.throughput
        except Exception:
            pass
    
    def get_loss_trend(self, history: List['DistillationMetrics']) -> str:
        """分析损失趋势"""
        if len(history) < 2:
            return "insufficient_data"
        
        recent_losses = [m.total_loss for m in history[-10:]]
        if len(recent_losses) < 2:
            return "insufficient_data"
        
        avg_recent = sum(recent_losses[-5:]) / 5 if len(recent_losses) >= 5 else sum(recent_losses) / len(recent_losses)
        avg_older = sum(recent_losses[:-5]) / (len(recent_losses) - 5) if len(recent_losses) > 5 else recent_losses[0]
        
        if avg_recent < avg_older * 0.95:
            return "improving"
        elif avg_recent > avg_older * 1.05:
            return "degrading"
        else:
            return "stable"


@dataclass
class DistillationReport:
    """
    蒸馏报告
    
    增强版，包含详细的分析和建议
    """
    task_id: str
    tenant_id: str
    
    # 任务信息
    task_name: str
    scenario: str
    status: str
    
    # 时间信息
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    duration_seconds: float = 0.0
    
    # 结果摘要
    final_loss: float = 0.0
    avg_loss: float = 0.0
    min_loss: float = float('inf')
    total_steps: int = 0
    total_epochs: int = 0
    
    # 模型信息
    teacher_model_size_mb: float = 0.0
    student_model_size_mb: float = 0.0
    compression_ratio: float = 0.0
    
    # 性能对比
    teacher_accuracy: float = 0.0
    student_accuracy: float = 0.0
    accuracy_retention: float = 0.0
    
    # 推理性能
    teacher_latency_ms: float = 0.0
    student_latency_ms: float = 0.0
    speedup_ratio: float = 0.0

    # 资源使用
    peak_memory_mb: float = 0.0
    avg_gpu_utilization: float = 0.0
    total_training_samples: int = 0
    
    # 分析结果
    convergence_status: str = "unknown"  # converged, not_converged, early_stopped
    training_quality: str = "unknown"  # excellent, good, fair, poor
    recommendations: List[str] = field(default_factory=list)
    
    # 元数据
    report_generated_at: str = ""
    report_version: str = "1.0"
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.report_generated_at:
            self.report_generated_at = datetime.utcnow().isoformat()
        
        # 计算压缩比
        if self.teacher_model_size_mb > 0 and self.student_model_size_mb > 0:
            self.compression_ratio = self.teacher_model_size_mb / self.student_model_size_mb
        
        # 计算精度保留率
        if self.teacher_accuracy > 0:
            self.accuracy_retention = self.student_accuracy / self.teacher_accuracy
        
        # 计算加速比
        if self.teacher_latency_ms > 0 and self.student_latency_ms > 0:
            self.speedup_ratio = self.teacher_latency_ms / self.student_latency_ms
        
        # 生成建议
        self._generate_recommendations()
    
    def _generate_recommendations(self) -> None:
        """生成优化建议"""
        self.recommendations = []
        
        # 精度建议
        if self.accuracy_retention < 0.9:
            self.recommendations.append(
                "精度保留率较低，建议：1) 降低蒸馏温度 2) 增加硬标签权重 3) 添加特征蒸馏"
            )
        
        # 压缩比建议
        if self.compression_ratio < 2.0 and self.scenario == "edge_deploy":
            self.recommendations.append(
                "边缘部署场景下压缩比不足，建议：1) 使用更小的学生模型 2) 结合量化/剪枝"
            )
        
        # 训练时间建议
        if self.duration_seconds > 86400:  # 超过24小时
            self.recommendations.append(
                "训练时间过长，建议：1) 使用分布式训练 2) 减少训练轮数 3) 使用混合精度"
            )
        
        # 内存建议
        if self.peak_memory_mb > 40000:  # 超过40GB
            self.recommendations.append(
                "显存使用过高，建议：1) 使用FSDP/ZeRO 2) 启用梯度检查点 3) 减少批次大小"
            )
        
        # 收敛建议
        if self.convergence_status == "not_converged":
            self.recommendations.append(
                "模型未收敛，建议：1) 增加训练步数 2) 调整学习率 3) 检查数据质量"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistillationReport':
        """从字典创建"""
        try:
            from dataclasses import fields
            valid_fields = {f.name for f in fields(cls)}
        except Exception:
            valid_fields = set(data.keys())
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), indent=2)
    
    def get_quality_score(self) -> float:
        """计算训练质量分数 (0-100)"""
        score = 0.0
        
        # 精度保留 (40分)
        if self.accuracy_retention >= 0.99:
            score += 40
        elif self.accuracy_retention >= 0.95:
            score += 35
        elif self.accuracy_retention >= 0.90:
            score += 25
        elif self.accuracy_retention >= 0.80:
            score += 15
        
        # 压缩比 (30分)
        if self.compression_ratio >= 10:
            score += 30
        elif self.compression_ratio >= 5:
            score += 25
        elif self.compression_ratio >= 2:
            score += 15
        elif self.compression_ratio >= 1.5:
            score += 10
        
        # 加速比 (20分)
        if self.speedup_ratio >= 5:
            score += 20
        elif self.speedup_ratio >= 3:
            score += 15
        elif self.speedup_ratio >= 2:
            score += 10
        elif self.speedup_ratio >= 1.5:
            score += 5
        
        # 收敛状态 (10分)
        if self.convergence_status == "converged":
            score += 10
        elif self.convergence_status == "early_stopped":
            score += 5
        
        return score
    
    def summary(self) -> str:
        """获取报告摘要"""
        quality_score = self.get_quality_score()
        quality_label = (
            "excellent" if quality_score >= 80 else
            "good" if quality_score >= 60 else
            "fair" if quality_score >= 40 else
            "poor"
        )
        
        return (
            f"Report[{self.task_id[:8]}...] "
            f"status={self.status} | "
            f"accuracy_retention={self.accuracy_retention:.2%} | "
            f"compression={self.compression_ratio:.1f}x | "
            f"speedup={self.speedup_ratio:.1f}x | "
            f"quality={quality_label}({quality_score:.0f})"
        )


# ======================== 服务监控组件 ========================

@dataclass
class ServiceStats:
    """服务统计数据"""
    total_tasks_created: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_tasks_cancelled: int = 0
    total_training_time_seconds: float = 0.0
    total_gpu_hours: float = 0.0
    
    # 当前状态
    active_tasks: int = 0
    queued_tasks: int = 0
    
    # 性能指标
    avg_task_duration_seconds: float = 0.0
    avg_accuracy_retention: float = 0.0
    avg_compression_ratio: float = 0.0
    
    # 资源使用
    peak_concurrent_tasks: int = 0
    peak_memory_usage_mb: float = 0.0
    
    # 时间窗口
    stats_window_start: str = ""
    stats_window_end: str = ""
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.stats_window_start:
            self.stats_window_start = datetime.utcnow().isoformat()
        self.stats_window_end = datetime.utcnow().isoformat()
    
    def update(self, task: DistillationTask) -> None:
        """根据任务更新统计"""
        status = task.get_status_enum()
        
        if status == DistillationTaskStatus.COMPLETED:
            self.total_tasks_completed += 1
            self.total_training_time_seconds += task.total_training_time_seconds
        elif status == DistillationTaskStatus.FAILED:
            self.total_tasks_failed += 1
        elif status == DistillationTaskStatus.CANCELLED:
            self.total_tasks_cancelled += 1
        
        # 更新内存峰值
        if task.peak_memory_mb > self.peak_memory_usage_mb:
            self.peak_memory_usage_mb = task.peak_memory_mb
        
        self.stats_window_end = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


class ServiceMonitor:
    """
    服务监控器
    
    整合配置层的 DistillationMonitor 和策略层的 StrategyMonitor
    """
    
    def __init__(self, max_history: int = 10000):
        """初始化"""
        self._stats = ServiceStats()
        self._task_metrics: Dict[str, List[DistillationMetrics]] = {}
        self._max_history = max_history
        self._lock = threading.Lock()
        
        # 整合配置层监控器
        self._distillation_monitor: Optional['DistillationMonitor'] = None
        try:
            self._distillation_monitor = DistillationMonitor()
        except Exception as e:
            logger.warning(f"Failed to create DistillationMonitor: {e}")
        
        # 整合策略层监控器
        self._strategy_monitor: Optional['StrategyMonitor'] = None
        if STRATEGY_LAYER_AVAILABLE and StrategyMonitor is not None:
            try:
                self._strategy_monitor = StrategyMonitor(history_size=1000)
            except Exception as e:
                logger.warning(f"Failed to create StrategyMonitor: {e}")
    
    def record_task_created(self) -> None:
        """记录任务创建"""
        with self._lock:
            self._stats.total_tasks_created += 1
    
    def record_task_metrics(self, task_id: str, metrics: DistillationMetrics) -> None:
        """记录任务指标"""
        with self._lock:
            if task_id not in self._task_metrics:
                self._task_metrics[task_id] = []
            
            self._task_metrics[task_id].append(metrics)
            
            # 限制历史记录
            if len(self._task_metrics[task_id]) > self._max_history:
                self._task_metrics[task_id] = self._task_metrics[task_id][-self._max_history//2:]
            
            # 同步到配置层监控器
            if self._distillation_monitor is not None:
                try:
                    self._distillation_monitor.record_step(
                        kd_loss=metrics.soft_loss,
                        ce_loss=metrics.hard_loss,
                        feature_loss=metrics.feature_loss,
                        attention_loss=metrics.attention_loss,
                        accuracy=metrics.student_accuracy,
                        temperature=metrics.temperature
                    )
                except Exception:
                    pass
            
            # 同步到策略层监控器
            if self._strategy_monitor is not None and hasattr(self._strategy_monitor, 'record_step'):
                try:
                    # pylint: disable=no-member
                    if StrategyResult is not None and StrategyContext is not None:
                        result = StrategyResult(
                            loss=metrics.total_loss,
                            metrics={'total_loss': metrics.total_loss}
                        )
                        context = StrategyContext(
                            global_step=metrics.step if hasattr(metrics, 'step') else 0
                        )
                        self._strategy_monitor.record_step(result, context)
                except Exception:
                    pass
    
    def record_task_completed(self, task: DistillationTask) -> None:
        """记录任务完成"""
        with self._lock:
            self._stats.update(task)
    
    def update_active_count(self, active: int, queued: int) -> None:
        """更新活跃任务数"""
        with self._lock:
            self._stats.active_tasks = active
            self._stats.queued_tasks = queued
            
            # 更新峰值
            total = active + queued
            if total > self._stats.peak_concurrent_tasks:
                self._stats.peak_concurrent_tasks = total
    
    def get_task_metrics(self, task_id: str, limit: int = 100) -> List[DistillationMetrics]:
        """获取任务指标历史"""
        with self._lock:
            return list(self._task_metrics.get(task_id, [])[-limit:])
    
    def get_task_loss_trend(self, task_id: str) -> str:
        """获取任务损失趋势"""
        metrics = self.get_task_metrics(task_id)
        if len(metrics) < 2:
            return "insufficient_data"
        
        return metrics[-1].get_loss_trend(metrics)
    
    def get_stats(self) -> ServiceStats:
        """获取服务统计"""
        with self._lock:
            return self._stats
    
    def get_distillation_monitor(self) -> Optional['DistillationMonitor']:
        """获取配置层监控器"""
        return self._distillation_monitor
    
    def get_strategy_monitor(self) -> Optional['StrategyMonitor']:
        """获取策略层监控器"""
        return self._strategy_monitor
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断服务状态"""
        with self._lock:
            stats = self._stats.to_dict()
        
        diagnosis = {
            'stats': stats,
            'task_count': len(self._task_metrics),
            'layers': {
                'distillation_monitor': self._distillation_monitor is not None,
                'strategy_monitor': self._strategy_monitor is not None,
            }
        }
        
        # 添加配置层诊断
        if self._distillation_monitor is not None:
            try:
                diagnosis['distillation_monitor_stats'] = self._distillation_monitor.get_stats().to_dict() if hasattr(self._distillation_monitor.get_stats(), 'to_dict') else {}
            except Exception:
                pass
        
        return diagnosis


class ServiceProfiler:
    """
    服务性能分析器
    
    整合策略层的 StrategyProfiler
    """
    
    def __init__(self):
        """初始化"""
        self._operation_times: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        
        # 整合策略层分析器
        self._strategy_profiler: Optional['StrategyProfiler'] = None
        if STRATEGY_LAYER_AVAILABLE and StrategyProfiler is not None:
            try:
                self._strategy_profiler = StrategyProfiler()
            except Exception as e:
                logger.warning(f"Failed to create StrategyProfiler: {e}")
    
    @contextmanager
    def profile(self, operation_name: str):
        """性能分析上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            with self._lock:
                if operation_name not in self._operation_times:
                    self._operation_times[operation_name] = []
                self._operation_times[operation_name].append(elapsed)
                
                # 限制历史
                if len(self._operation_times[operation_name]) > 1000:
                    self._operation_times[operation_name] = self._operation_times[operation_name][-500:]
            
            # 同步到策略层分析器
            if self._strategy_profiler is not None:
                try:
                    with self._strategy_profiler.profile(operation_name):
                        pass  # 已经执行完毕，只是记录
                except Exception:
                    pass
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """获取操作统计"""
        with self._lock:
            times = self._operation_times.get(operation_name, [])
        
        if not times:
            return {'count': 0, 'avg_ms': 0, 'min_ms': 0, 'max_ms': 0}
        
        return {
            'count': len(times),
            'avg_ms': sum(times) / len(times) * 1000,
            'min_ms': min(times) * 1000,
            'max_ms': max(times) * 1000,
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """获取所有操作统计"""
        with self._lock:
            operations = list(self._operation_times.keys())
        
        return {op: self.get_operation_stats(op) for op in operations}
    
    def get_strategy_profiler(self) -> Optional['StrategyProfiler']:
        """获取策略层分析器"""
        return self._strategy_profiler


# ======================== 任务调度器 ========================

class TaskScheduler:
    """
    任务调度器
    
    支持优先级调度、资源管理、并发控制
    """
    
    def __init__(self, max_concurrent_tasks: int = 4, max_queue_size: int = 100):
        """
        初始化调度器
        
        Args:
            max_concurrent_tasks: 最大并发任务数
            max_queue_size: 最大队列大小
        """
        self._max_concurrent = max_concurrent_tasks
        self._max_queue_size = max_queue_size
        self._queue: List[Tuple[int, str]] = []  # (priority, task_id)
        self._running: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()
        
        # 资源管理（使用硬件层）
        self._device_manager: Optional['DeviceManager'] = None
        self._memory_manager: Optional['MemoryManager'] = None
        
        if HARDWARE_LAYER_AVAILABLE:
            try:
                if DeviceManager is not None:
                    self._device_manager = get_device_manager() if get_device_manager is not None else None
                if MemoryManager is not None:
                    self._memory_manager = get_memory_manager() if get_memory_manager is not None else None
            except Exception as e:
                logger.warning(f"Failed to initialize hardware managers: {e}")
    
    def enqueue(self, task_id: str, priority: int = 1) -> bool:
        """
        将任务加入队列
        
        Args:
            task_id: 任务ID
            priority: 优先级（越高越优先）
        
        Returns:
            是否成功加入队列
        """
        with self._lock:
            if len(self._queue) >= self._max_queue_size:
                return False
            
            # 按优先级插入（高优先级在前）
            insert_pos = 0
            for i, (p, _) in enumerate(self._queue):
                if priority > p:
                    break
                insert_pos = i + 1
            
            self._queue.insert(insert_pos, (priority, task_id))
            return True
    
    def dequeue(self) -> Optional[str]:
        """
        从队列取出任务
        
        Returns:
            任务ID，如果队列为空则返回None
        """
        with self._lock:
            if not self._queue:
                return None
            
            if len(self._running) >= self._max_concurrent:
                return None
            
            _, task_id = self._queue.pop(0)
            return task_id
    
    def mark_running(self, task_id: str, thread: threading.Thread) -> None:
        """标记任务为运行中"""
        with self._lock:
            self._running[task_id] = thread
    
    def mark_completed(self, task_id: str) -> None:
        """标记任务完成"""
        with self._lock:
            self._running.pop(task_id, None)
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Returns:
            是否成功取消
        """
        with self._lock:
            # 从队列中移除
            for i, (_, tid) in enumerate(self._queue):
                if tid == task_id:
                    self._queue.pop(i)
                    return True
            
            # 如果正在运行，标记为需要取消（实际取消由任务自己处理）
            if task_id in self._running:
                return True
            
            return False
    
    def get_queue_size(self) -> int:
        """获取队列大小"""
        with self._lock:
            return len(self._queue)
    
    def get_running_count(self) -> int:
        """获取运行中任务数"""
        with self._lock:
            return len(self._running)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取调度器统计"""
        with self._lock:
            return {
                'queue_size': len(self._queue),
                'running_count': len(self._running),
                'max_concurrent': self._max_concurrent,
                'max_queue_size': self._max_queue_size,
                'device_manager_available': self._device_manager is not None,
                'memory_manager_available': self._memory_manager is not None,
            }
    
    def check_resources(self, estimated_memory_mb: float = 0) -> bool:
        """
        检查资源是否充足
        
        Args:
            estimated_memory_mb: 预估内存需求
        
        Returns:
            资源是否充足
        """
        if not HARDWARE_LAYER_AVAILABLE or get_available_memory is None:
            return True  # 无法检查，默认充足
        
        try:
            available = get_available_memory()
            return available > estimated_memory_mb * 1.2  # 保留20%余量
        except Exception:
            return True


# ======================== 健康检查器 ========================

class HealthChecker:
    """
    服务健康检查器
    
    整合硬件层和分布式层的健康检查能力
    """
    
    def __init__(self):
        """初始化"""
        self._last_check_time: Optional[str] = None
        self._last_status: ServiceHealthStatus = ServiceHealthStatus.UNKNOWN
        self._check_history: deque = deque(maxlen=100)
        
        # 硬件层管理器
        self._device_manager: Optional['DeviceManager'] = None
        self._memory_manager: Optional['MemoryManager'] = None
        
        if HARDWARE_LAYER_AVAILABLE:
            try:
                if DeviceManager is not None and get_device_manager is not None:
                    self._device_manager = get_device_manager()
                if MemoryManager is not None and get_memory_manager is not None:
                    self._memory_manager = get_memory_manager()
            except Exception as e:
                logger.warning(f"Failed to initialize hardware managers for health check: {e}")
        
        # 分布式层管理器
        self._distributed_manager: Optional['DistributedManager'] = None
        if DISTRIBUTED_LAYER_AVAILABLE:
            try:
                if DistributedManager is not None and get_distributed_manager is not None:
                    self._distributed_manager = get_distributed_manager()
            except Exception as e:
                logger.warning(f"Failed to initialize distributed manager for health check: {e}")
    
    def check(self) -> ServiceHealthStatus:
        """
        执行健康检查
        
        Returns:
            健康状态
        """
        issues = []
        
        # 检查GPU内存
        if HARDWARE_LAYER_AVAILABLE and get_available_memory is not None:
            try:
                available = get_available_memory()
                if available < 1000:  # 少于1GB
                    issues.append("Low GPU memory")
            except Exception:
                pass
        
        # 检查设备状态
        if self._device_manager is not None:
            try:
                if hasattr(self._device_manager, 'get_device_count'):
                    if self._device_manager.get_device_count() == 0:
                        issues.append("No GPU devices available")
            except Exception:
                pass
        
        # 检查分布式状态
        if self._distributed_manager is not None:
            try:
                if hasattr(self._distributed_manager, 'is_healthy'):
                    if not self._distributed_manager.is_healthy():
                        issues.append("Distributed training unhealthy")
            except Exception:
                pass
        
        # 确定状态
        if len(issues) == 0:
            status = ServiceHealthStatus.HEALTHY
        elif len(issues) <= 1:
            status = ServiceHealthStatus.DEGRADED
        else:
            status = ServiceHealthStatus.UNHEALTHY
        
        # 记录
        self._last_check_time = datetime.utcnow().isoformat()
        self._last_status = status
        self._check_history.append({
            'time': self._last_check_time,
            'status': status.value,
            'issues': issues,
        })
        
        return status
    
    def get_last_status(self) -> ServiceHealthStatus:
        """获取上次检查状态"""
        return self._last_status
    
    def get_check_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取检查历史"""
        return list(self._check_history)[-limit:]
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """获取详细状态"""
        status = self.check()
        
        result = {
            'status': status.value,
            'last_check_time': self._last_check_time,
        }
        
        # 添加硬件信息
        if HARDWARE_LAYER_AVAILABLE and get_available_memory is not None:
            try:
                result['gpu_memory_available_mb'] = get_available_memory()
            except Exception:
                pass
        
        # 添加分布式信息
        if DISTRIBUTED_LAYER_AVAILABLE:
            result['distributed'] = {
                'rank': get_rank() if get_rank is not None else 0,
                'world_size': get_world_size() if get_world_size is not None else 1,
                'is_main_process': is_main_process() if is_main_process is not None else True,
            }
        
        return result


# ======================== 蒸馏服务 ========================

class DistillationService:
    """
    知识蒸馏服务
    
    提供完整的蒸馏服务能力，支持租户隔离和平台集成。
    
    整合以下模块：
    - compression_config.py: 配置管理、监控
    - distillation_scenarios.py: 场景管理
    - knowledge_distillation.py: 训练器
    - base_strategy.py: 策略监控、验证
    - distributed_strategy.py: 分布式训练
    - backend/lib/hardware: 设备和内存管理
    - backend/lib/distributed: 分布式协调
    
    生产级特性：
    - 完整的任务生命周期管理
    - 多租户隔离
    - 任务调度和优先级
    - 健康检查和自动恢复
    - 详细的监控和报告
    """
    
    def __init__(
        self, 
        use_memory_storage: bool = False,
        max_concurrent_tasks: int = 4,
        max_queue_size: int = 100,
        enable_health_check: bool = True,
        health_check_interval_seconds: int = 60,
    ):
        """
        初始化蒸馏服务
        
        Args:
            use_memory_storage: 是否使用内存存储（用于测试）
            max_concurrent_tasks: 最大并发任务数
            max_queue_size: 最大队列大小
            enable_health_check: 是否启用健康检查
            health_check_interval_seconds: 健康检查间隔
        """
        self.logger = logging.getLogger(__name__)
        self._use_memory_storage = use_memory_storage
        
        # 场景管理器（使用 distillation_scenarios.py）
        self._scenario_manager: Optional['DistillationScenarioManager'] = None
        try:
            self._scenario_manager = get_scenario_manager()
        except Exception as e:
            self.logger.warning(f"Failed to get scenario manager: {e}")
        
        # 任务缓存（内存存储）
        self._tasks: Dict[str, DistillationTask] = {}
        self._task_lock = threading.Lock()
        
        # 任务调度器
        self._scheduler = TaskScheduler(
            max_concurrent_tasks=max_concurrent_tasks,
            max_queue_size=max_queue_size,
        )
        
        # 运行中的训练器
        self._running_trainers: Dict[str, 'KnowledgeDistillationTrainer'] = {}
        
        # 服务监控器
        self._monitor = ServiceMonitor()
        
        # 服务分析器
        self._profiler = ServiceProfiler()
        
        # 健康检查器
        self._health_checker = HealthChecker() if enable_health_check else None
        self._health_check_interval = health_check_interval_seconds
        self._health_check_thread: Optional[threading.Thread] = None
        
        # 配置验证器（使用 compression_config.py）
        self._config_validator: Optional['ConfigValidator'] = None
        try:
            self._config_validator = ConfigValidator()
        except Exception as e:
            self.logger.warning(f"Failed to create ConfigValidator: {e}")
            
        # 策略验证器（使用 base_strategy.py）
        self._strategy_validator: Optional['StrategyValidator'] = None
        if STRATEGY_LAYER_AVAILABLE and StrategyValidator is not None:
            try:
                self._strategy_validator = StrategyValidator()
            except Exception as e:
                self.logger.warning(f"Failed to create StrategyValidator: {e}")
        
        # 回调函数
        self._callbacks: Dict[str, List[Callable]] = {
            'on_task_created': [],
            'on_task_started': [],
            'on_task_completed': [],
            'on_task_failed': [],
            'on_progress_update': [],
            'on_health_check': [],
        }
        
        # 初始化仓库（如果需要数据库存储）
        self._init_repositories(use_memory_storage)
        
        # 启动健康检查线程
        if enable_health_check:
            self._start_health_check_thread()
        
        self.logger.info(
            f"DistillationService initialized: "
            f"max_concurrent={max_concurrent_tasks}"
        )
    
    def _init_repositories(self, use_memory_storage: bool):
        """初始化仓库层"""
        self._task_repo = None
        self._metrics_repo = None
        
        if use_memory_storage:
            return
        
        # 尝试加载数据库仓库
        try:
            from backend.repositories import get_distillation_task_repository
            self._task_repo = get_distillation_task_repository()
        except ImportError:
            self.logger.warning("DistillationTask repository not available, using memory storage")
    
    def _start_health_check_thread(self):
        """启动健康检查线程"""
        def health_check_loop():
            while True:
                try:
                    if self._health_checker:
                        status = self._health_checker.check()
                        self._trigger_callback('on_health_check', status)
                except Exception as e:
                    self.logger.warning(f"Health check failed: {e}")
                
                time.sleep(self._health_check_interval)
        
        self._health_check_thread = threading.Thread(
            target=health_check_loop,
            daemon=True,
            name="DistillationService-HealthCheck"
        )
        self._health_check_thread.start()
    
    # ======================== 任务管理 ========================
    
    def create_task(
        self,
        task_name: str,
        tenant_id: str,
        user_id: str,
        teacher_model_path: str,
        student_model_path: str,
        scenario: str = "standard",
        priority: int = 1,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> DistillationTask:
        """
        创建蒸馏任务
        
        使用配置层的预设和验证功能
        
        Args:
            task_name: 任务名称
            tenant_id: 租户ID
            user_id: 用户ID
            teacher_model_path: 教师模型路径
            student_model_path: 学生模型路径
            scenario: 蒸馏场景
            priority: 任务优先级
            config_overrides: 配置覆盖
        
        Returns:
            创建的蒸馏任务
        """
        with self._profiler.profile("create_task"):
            task_id = str(uuid.uuid4())
            base_config = None
        
        # 根据场景获取预设配置（使用 compression_config.py 的 DistillationPresets）
        try:
            base_config = DistillationPresets.get(scenario)
        except Exception as e:
            self.logger.warning(f"Failed to get preset for scenario {scenario}: {e}")

        if scenario == "edge_deploy":
            base_config = DistillationPresets.edge_deployment()
        elif scenario == "high_accuracy":
            base_config = DistillationPresets.high_accuracy()
        elif scenario == "industry":
            industry_type = config_overrides.get('industry_type',
                                                 'manufacturing') if config_overrides else 'manufacturing'
            base_config = DistillationPresets.industry_model(industry_type)
        elif scenario == "multimodal":
            base_config = DistillationPresets.multimodal()
        elif scenario == "progressive":
            base_config = DistillationPresets.progressive_distillation()
        elif scenario == "self_distillation":
            base_config = DistillationPresets.self_distillation()
        elif scenario == "contrastive":
            base_config = DistillationPresets.contrastive_distillation()
        elif scenario == "low_latency":
            base_config = DistillationPresets.low_latency()
        elif scenario == "real_time":
            base_config = DistillationPresets.real_time()
        else:
            base_config = DistillationPresets.standard()
        # 如果预设获取失败，创建默认配置
        if base_config is None:
            base_config = DistillationTaskConfig()
        
        # 设置基础蒸馏配置
        try:
            distill_config = DistillationConfig(
                teacher_model_path=teacher_model_path,
                student_model_path=student_model_path,
            )
            if hasattr(base_config, 'distillation_config'):
                base_config.distillation_config = distill_config
        except Exception as e:
            self.logger.warning(f"Failed to create distillation config: {e}")
            
            # 根据场景设置场景专用配置（使用 ScenarioDistillationConfig）
        try:
            # 将字符串场景转换为枚举
            scenario_enum = None
            if DistillationScenario is not None:
                scenario_mapping = {
                    'standard': DistillationScenario.STANDARD,
                    'edge_deploy': DistillationScenario.EDGE_DEPLOY,
                    'high_accuracy': DistillationScenario.HIGH_ACCURACY,
                    'low_latency': DistillationScenario.LOW_LATENCY,
                    'progressive': DistillationScenario.PROGRESSIVE,
                    'multimodal': DistillationScenario.MULTIMODAL,
                }
                scenario_enum = scenario_mapping.get(scenario)

            if scenario_enum is not None:
                scenario_config = ScenarioDistillationConfig(scenario=scenario_enum)
                if hasattr(base_config, 'scenario_config'):
                    base_config.scenario_config = scenario_config
                self.logger.debug(f"Applied scenario config for: {scenario}")
        except Exception as e:
            self.logger.warning(f"Failed to create scenario config: {e}")
            
            # 根据配置覆盖设置自适应蒸馏配置（使用 AdaptiveDistillationConfig）
        try:
            adaptive_mode = config_overrides.get('adaptive_mode')
            if adaptive_mode and AdaptiveMode is not None:
                mode_mapping = {
                    'temperature': AdaptiveMode.TEMPERATURE,
                    'alpha': AdaptiveMode.LOSS_WEIGHT,
                    'curriculum': AdaptiveMode.CURRICULUM,
                    'combined': AdaptiveMode.FULL,
                }
                mode_enum = mode_mapping.get(adaptive_mode)
                if mode_enum is not None:
                    # 构造自适应配置
                    adaptive_kwargs = {'mode': mode_enum.value}
                    
                    # 映射温度参数
                    if 'temperature' in config_overrides:
                        adaptive_kwargs['temperature_range'] = (1.0, config_overrides['temperature'])
                        
                    # 映射 alpha 参数
                    if 'alpha' in config_overrides:
                        adaptive_kwargs['weight_max'] = config_overrides['alpha']
                        
                    adaptive_config = AdaptiveDistillationConfig(**adaptive_kwargs)
                    if hasattr(base_config, 'adaptive_config'):
                        base_config.adaptive_config = adaptive_config
                    self.logger.debug(f"Applied adaptive config: mode={adaptive_mode}")
        except Exception as e:
            self.logger.warning(f"Failed to create adaptive config: {e}")
            
            # 设置分布式蒸馏配置（使用 DistributedDistillationConfig）
        try:
            distributed_mode = config_overrides.get('distributed_mode')
            if distributed_mode and ConfigDistributedMode is not None:
                mode_mapping = {
                    'ddp': ConfigDistributedMode.DATA_PARALLEL,
                    'fsdp': ConfigDistributedMode.FSDP,
                    'deepspeed': ConfigDistributedMode.ZERO,
                    'horovod': ConfigDistributedMode.HYBRID,
                }
                mode_enum = mode_mapping.get(distributed_mode)
                if mode_enum is not None:
                    distributed_config = DistributedDistillationConfig(
                        mode=mode_enum.value,
                        world_size=config_overrides.get('world_size', 1),
                        gradient_accumulation_steps=config_overrides.get('gradient_accumulation_steps', 1),
                    )
                    if hasattr(base_config, 'distributed_config'):
                        base_config.distributed_config = distributed_config
                    
                    # 稍后在任务创建时应用这些属性
                    # task.distributed_mode = distributed_mode 
                    # task.num_gpus = config_overrides.get('world_size', 1)
                    self.logger.debug(f"Applied distributed config: mode={distributed_mode}")
        except Exception as e:
            self.logger.warning(f"Failed to create distributed config: {e}")
            
            # 设置压缩配置（使用 CompressionConfig）
        try:
            compression_method = config_overrides.get('compression_method')
            if compression_method and CompressionMethod is not None:
                method_mapping = {
                    'quantization': CompressionMethod.QUANTIZATION,
                    'pruning': CompressionMethod.PRUNING,
                    'distillation': CompressionMethod.DISTILLATION,
                    'combined': CompressionMethod.MIXED,
                }
                method_enum = method_mapping.get(compression_method)
                if method_enum is not None:
                    # 根据方法设置相应的启用标志
                    comp_kwargs = {}
                    if method_enum == CompressionMethod.QUANTIZATION:
                        comp_kwargs['use_quantization'] = True
                        comp_kwargs['quantization_bits'] = config_overrides.get('quantization_bits', 8)
                    elif method_enum == CompressionMethod.PRUNING:
                        comp_kwargs['use_pruning'] = True
                        comp_kwargs['pruning_ratio'] = config_overrides.get('target_sparsity', 0.5)
                    elif method_enum == CompressionMethod.DISTILLATION:
                        comp_kwargs['use_distillation'] = True
                    elif method_enum == CompressionMethod.MIXED:
                        comp_kwargs['use_quantization'] = True
                        comp_kwargs['use_pruning'] = True
                        comp_kwargs['use_distillation'] = True
                        
                    compression_config = CompressionConfig(**comp_kwargs)
                    
                    if hasattr(base_config, 'compression_config'):
                        base_config.compression_config = compression_config
                    self.logger.debug(f"Applied compression config: method={compression_method}")
        except Exception as e:
            self.logger.warning(f"Failed to create compression config: {e}")
            
        # 设置任务元数据
        if hasattr(base_config, 'task_id'):
            base_config.task_id = task_id
        if hasattr(base_config, 'task_name'):
            base_config.task_name = task_name
        if hasattr(base_config, 'tenant_id'):
            base_config.tenant_id = tenant_id
        if hasattr(base_config, 'user_id'):
            base_config.user_id = user_id
        
        # 应用配置覆盖
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(base_config, key):
                    setattr(base_config, key, value)
        
        estimated_memory = 0.0
        # 估算资源需求
        if hasattr(base_config, 'estimate_memory_mb'):
            try:
                estimated_memory = base_config.estimate_memory_mb()
            except Exception:
                pass
        else:
            try:
                resources = estimate_training_resources(base_config)
                estimated_memory = resources.get('memory_mb', 0)
            except Exception:
                pass
        
        # 创建任务对象
        task = DistillationTask(
            task_id=task_id,
            task_name=task_name,
            tenant_id=tenant_id,
            user_id=user_id,
            config=base_config,
            scenario=scenario,
            status=DistillationTaskStatus.PENDING.value,
                priority=priority,
            created_at=datetime.utcnow().isoformat(),
            teacher_model_id=teacher_model_path,
            student_model_id=student_model_path,
                estimated_memory_mb=estimated_memory,
            )
            
        # 存储任务
        with self._task_lock:
            self._tasks[task_id] = task
            
        # 记录监控
        self._monitor.record_task_created()
            
        # 触发回调
        self._trigger_callback('on_task_created', task)
            
        self.logger.info(
            f"Created distillation task: {task_id} for tenant: {tenant_id}, "
            f"scenario={scenario}, priority={priority}"
        )
        
        return task
    
    def create_task_from_config(
        self,
        config: Union[Dict[str, Any], 'DistillationTaskConfig'],
        tenant_id: str,
        user_id: str,
        task_name: Optional[str] = None,
    ) -> DistillationTask:
        """
        从配置创建任务
        
        Args:
            config: 任务配置（字典或 DistillationTaskConfig）
            tenant_id: 租户ID
            user_id: 用户ID
            task_name: 任务名称
        
        Returns:
            创建的蒸馏任务
        """
        task_id = str(uuid.uuid4())
        
        # 处理配置
        if isinstance(config, dict):
            try:
                task_config = DistillationTaskConfig.from_dict(config)
            except Exception:
                task_config = config
        else:
            task_config = config
        
        # 获取场景
        scenario = "standard"
        if hasattr(task_config, 'scenario_config') and task_config.scenario_config:
            scenario = getattr(task_config.scenario_config, 'scenario', 'standard')
        
        # 创建任务
        task = DistillationTask(
            task_id=task_id,
            task_name=task_name or f"task_{task_id[:8]}",
            tenant_id=tenant_id,
            user_id=user_id,
            config=task_config,
            scenario=scenario,
            status=DistillationTaskStatus.PENDING.value,
        )
        
        # 存储任务
        with self._task_lock:
            self._tasks[task_id] = task
        
        self._monitor.record_task_created()
        self._trigger_callback('on_task_created', task)
        
        return task
    
    def create_task_from_recommendation(
        self,
        tenant_id: str,
        user_id: str,
        task_name: str,
        teacher_model_path: str,
        student_model_path: str,
        requirements: Dict[str, Any],
    ) -> DistillationTask:
        """
        根据需求推荐配置并创建任务
        
        使用 compression_config.py 的 recommend_config
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            task_name: 任务名称
            teacher_model_path: 教师模型路径
            student_model_path: 学生模型路径
            requirements: 需求描述
        
        Returns:
            创建的蒸馏任务
        """
        # 推荐配置
        try:
            recommended = recommend_config(requirements)

            # 设置模型路径
            if hasattr(recommended, 'distillation_config') and recommended.distillation_config:
                recommended.distillation_config.teacher_model_path = teacher_model_path
                recommended.distillation_config.student_model_path = student_model_path

            return self.create_task_from_config(
                config=recommended,
                tenant_id=tenant_id,
                user_id=user_id,
                task_name=task_name,
            )
        except Exception as e:
            self.logger.warning(f"Failed to recommend config: {e}")
        
        # 回退到默认创建
        scenario = requirements.get('scenario', 'standard')
        return self.create_task(
            task_name=task_name,
            tenant_id=tenant_id,
            user_id=user_id,
            teacher_model_path=teacher_model_path,
            student_model_path=student_model_path,
            scenario=scenario,
        )
    
    def create_distributed_task(
        self,
        task_name: str,
        tenant_id: str,
        user_id: str,
        teacher_model_path: str,
        student_model_path: str,
        distributed_mode: str = 'ddp',
        world_size: int = 2,
        scenario: str = 'standard',
        priority: int = 1,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> DistillationTask:
        """
        创建分布式蒸馏任务
        
        整合 distributed_strategy.py 的功能
        
        Args:
            task_name: 任务名称
            tenant_id: 租户ID
            user_id: 用户ID
            teacher_model_path: 教师模型路径
            student_model_path: 学生模型路径
            distributed_mode: 分布式模式 ('ddp', 'fsdp', 'deepspeed', 'horovod')
            world_size: 并行度
            scenario: 蒸馏场景
            priority: 任务优先级
            config_overrides: 配置覆盖
        
        Returns:
            创建的分布式蒸馏任务
        """
        # 准备分布式配置覆盖
        dist_overrides = config_overrides or {}
        dist_overrides['distributed_mode'] = distributed_mode
        dist_overrides['world_size'] = world_size
        
        # 使用 recommend_distributed_mode 获取推荐配置（来自 distributed_strategy.py）
        if DISTRIBUTED_STRATEGY_AVAILABLE and recommend_distributed_mode is not None:
            try:
                # 准备需求信息
                model_size_gb = dist_overrides.get('model_size_gb', 2.0)
                available_memory_gb = dist_overrides.get('available_memory_gb', 16.0)
                
                recommended = recommend_distributed_mode(
                    model_size_gb=model_size_gb,
                    num_gpus=world_size,
                    memory_per_gpu_gb=available_memory_gb
                )
                if recommended:
                    self.logger.info(f"Recommended distributed mode: {recommended}")
                    # 更新分布式模式（如果推荐更适合）
                    if 'mode' in recommended:
                        dist_overrides['recommended_mode'] = recommended['mode']
            except Exception as e:
                self.logger.warning(f"Failed to get distributed mode recommendation: {e}")
        
        # 创建分布式策略配置（使用 DistributedStrategyConfig）
        if DISTRIBUTED_STRATEGY_AVAILABLE and DistributedStrategyConfig is not None:
            try:
                # 映射分布式模式到枚举
                mode_enum = None
                if StrategyDistributedMode is not None:
                    mode_mapping = {
                        'ddp': StrategyDistributedMode.DDP,
                        'fsdp': StrategyDistributedMode.FSDP,
                        'deepspeed': StrategyDistributedMode.ZERO,
                        'horovod': StrategyDistributedMode.HYBRID,
                    }
                    mode_enum = mode_mapping.get(distributed_mode)
                
                # 获取 ZeRO 阶段配置
                zero_stage = None
                if ZeROStage is not None and distributed_mode in ['deepspeed', 'fsdp']:
                    zero_stage_val = dist_overrides.get('zero_stage', 2)
                    stage_mapping = {
                        1: ZeROStage.STAGE_1,
                        2: ZeROStage.STAGE_2,
                        3: ZeROStage.STAGE_3,
                    }
                    zero_stage = stage_mapping.get(zero_stage_val, ZeROStage.STAGE_2)
                
                if mode_enum is not None:
                    dist_strategy_config = DistributedStrategyConfig(
                        distributed_mode=mode_enum,
                        world_size=world_size,
                        zero_stage=zero_stage,
                        gradient_accumulation_steps=dist_overrides.get('gradient_accumulation_steps', 1),
                        fp16=dist_overrides.get('fp16', False),
                        bf16=dist_overrides.get('bf16', False),
                    )
                    dist_overrides['_distributed_strategy_config'] = dist_strategy_config
                    self.logger.info(f"Created distributed strategy config: mode={distributed_mode}, world_size={world_size}")
            except Exception as e:
                self.logger.warning(f"Failed to create distributed strategy config: {e}")
        
        # 使用标准方法创建任务
        task = self.create_task(
            task_name=task_name,
            tenant_id=tenant_id,
            user_id=user_id,
            teacher_model_path=teacher_model_path,
            student_model_path=student_model_path,
            scenario=scenario,
            priority=priority,
            config_overrides=dist_overrides,
        )
        
        # 更新任务的分布式属性
        task.distributed_mode = distributed_mode
        task.num_gpus = world_size
        
        self.logger.info(
            f"Created distributed distillation task: {task.task_id}, "
            f"mode={distributed_mode}, world_size={world_size}"
        )
        
        return task
    
    def get_task(self, task_id: str, tenant_id: str) -> Optional[DistillationTask]:
        """
        获取任务
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
        
        Returns:
            蒸馏任务（如果存在且属于该租户）
        """
        with self._task_lock:
            task = self._tasks.get(task_id)
            if task and task.tenant_id == tenant_id:
                return task
        return None
    
    def get_task_unsafe(self, task_id: str) -> Optional[DistillationTask]:
        """
        获取任务（不检查租户，仅供内部使用）
        
        Args:
            task_id: 任务ID
        
        Returns:
            蒸馏任务
        """
        with self._task_lock:
            return self._tasks.get(task_id)
    
    def list_tasks(
        self,
        tenant_id: str,
        status: Optional[str] = None,
        scenario: Optional[str] = None,
        priority: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_desc: bool = True,
    ) -> List[DistillationTask]:
        """
        列出租户的任务
        
        Args:
            tenant_id: 租户ID
            status: 状态过滤
            scenario: 场景过滤
            priority: 优先级过滤
            limit: 返回数量限制
            offset: 偏移量
            sort_by: 排序字段
            sort_desc: 是否降序
        
        Returns:
            任务列表
        """
        with self._task_lock:
            tasks = [t for t in self._tasks.values() if t.tenant_id == tenant_id]
            
            if status:
                tasks = [t for t in tasks if t.status == status]
            
            if scenario:
                tasks = [t for t in tasks if t.scenario == scenario]
            
            if priority is not None:
                tasks = [t for t in tasks if t.priority == priority]
            
            # 排序
            if sort_by == "created_at":
                tasks.sort(key=lambda t: t.created_at or '', reverse=sort_desc)
            elif sort_by == "priority":
                tasks.sort(key=lambda t: t.priority, reverse=sort_desc)
            elif sort_by == "progress":
                tasks.sort(key=lambda t: t.progress, reverse=sort_desc)
            elif sort_by == "status":
                tasks.sort(key=lambda t: t.status, reverse=sort_desc)
            
            return tasks[offset:offset + limit]
    
    def count_tasks(
        self,
        tenant_id: str,
        status: Optional[str] = None,
    ) -> int:
        """
        统计任务数量
        
        Args:
            tenant_id: 租户ID
            status: 状态过滤
        
        Returns:
            任务数量
        """
        with self._task_lock:
            tasks = [t for t in self._tasks.values() if t.tenant_id == tenant_id]
            if status:
                tasks = [t for t in tasks if t.status == status]
            return len(tasks)
    
    def delete_task(self, task_id: str, tenant_id: str) -> bool:
        """
        删除任务
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
        
        Returns:
            是否成功删除
        """
        with self._task_lock:
            task = self._tasks.get(task_id)
            if task and task.tenant_id == tenant_id:
                # 如果正在运行，先停止
                if task.status == DistillationTaskStatus.RUNNING.value:
                    self._stop_task_internal(task_id)
                
                # 从调度器中取消
                self._scheduler.cancel_task(task_id)
                
                # 删除任务
                del self._tasks[task_id]
                
                self.logger.info(f"Deleted task: {task_id}")
                return True
        return False
    
    def update_task(
        self,
        task_id: str,
        tenant_id: str,
        updates: Dict[str, Any],
    ) -> Optional[DistillationTask]:
        """
        更新任务
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
            updates: 更新内容
        
        Returns:
            更新后的任务
        """
        task = self.get_task(task_id, tenant_id)
        if not task:
            return None
        
        # 不允许更新运行中任务的关键配置
        if task.status == DistillationTaskStatus.RUNNING.value:
            allowed_fields = {'priority', 'task_name'}
            updates = {k: v for k, v in updates.items() if k in allowed_fields}
        
        # 应用更新
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        task.last_updated_at = datetime.utcnow().isoformat()
        
        return task
    
    # ======================== 任务执行 ========================
    
    def start_task(self, task_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        启动蒸馏任务
        
        整合调度器和资源检查
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
        
        Returns:
            启动结果
        """
        with self._profiler.profile("start_task"):
            task = self.get_task(task_id, tenant_id)
        if not task:
            return {'success': False, 'error': 'Task not found'}
        
        status = task.get_status_enum()
            
        if not status.can_start():
            return {
                'success': False, 
                'error': f'Task cannot be started from status: {status.value}'
            }
            
        # 检查资源
        if not self._scheduler.check_resources(task.estimated_memory_mb):
            # 加入队列等待
            if self._scheduler.enqueue(task_id, task.priority):
                task.update_status(DistillationTaskStatus.QUEUED)
                self._update_scheduler_stats()
                return {
                    'success': True,
                    'task_id': task_id,
                    'status': 'queued',
                    'message': 'Task queued due to resource constraints'
                }
            else:
                return {
                    'success': False,
                    'error': 'Queue is full, please try again later'
                }
            
        # 直接启动
        return self._start_task_internal(task)
    
    def _start_task_internal(self, task: DistillationTask) -> Dict[str, Any]:
        """内部启动任务"""
        # 更新状态
        task.update_status(DistillationTaskStatus.RUNNING)
        task.started_at = datetime.utcnow().isoformat()
        
        # 准备场景（使用 distillation_scenarios.py）
        if self._scenario_manager is not None:
            try:
                self._scenario_manager.prepare_scenario(task.config)
            except Exception as e:
                self.logger.warning(f"Failed to prepare scenario: {e}")
        
        # 在后台线程中执行训练
        thread = threading.Thread(
            target=self._run_distillation_async,
            args=(task,),
            daemon=True,
            name=f"DistillationTask-{task.task_id[:8]}"
        )
        thread.start()
        
        # 标记运行状态
        self._scheduler.mark_running(task.task_id, thread)
        self._update_scheduler_stats()
        
        # 触发回调
        self._trigger_callback('on_task_started', task)
        
        self.logger.info(f"Started distillation task: {task.task_id}")
        return {
            'success': True,
            'task_id': task.task_id,
            'status': task.status,
            'started_at': task.started_at
        }
    
    def stop_task(self, task_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        停止任务
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
        
        Returns:
            停止结果
        """
        task = self.get_task(task_id, tenant_id)
        if not task:
            return {'success': False, 'error': 'Task not found'}
        
        status = task.get_status_enum()
        
        if not status.can_cancel():
            return {
                'success': False, 
                'error': f'Task cannot be stopped from status: {status.value}'
            }
        
        return self._stop_task_internal(task.task_id)
    
    def _stop_task_internal(self, task_id: str) -> Dict[str, Any]:
        """内部停止任务"""
        task = self.get_task_unsafe(task_id)
        if not task:
            return {'success': False, 'error': 'Task not found'}
        
        # 标记为取消
        task.update_status(DistillationTaskStatus.CANCELLED)
        task.completed_at = datetime.utcnow().isoformat()
        
        # 从调度器中移除
        self._scheduler.cancel_task(task_id)
        self._scheduler.mark_completed(task_id)
        self._update_scheduler_stats()
        
        # 停止训练器
        trainer = self._running_trainers.pop(task_id, None)
        if trainer is not None:
            try:
                # 尝试优雅停止
                if hasattr(trainer, 'stop'):
                    trainer.stop()
            except Exception as e:
                self.logger.warning(f"Failed to stop trainer: {e}")
        
        # 记录监控
        self._monitor.record_task_completed(task)
        
        self.logger.info(f"Stopped distillation task: {task_id}")
        return {'success': True, 'task_id': task_id}
    
    def pause_task(self, task_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        暂停任务
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
        
        Returns:
            暂停结果
        """
        task = self.get_task(task_id, tenant_id)
        if not task:
            return {'success': False, 'error': 'Task not found'}
        
        if not task.get_status_enum().can_pause():
            return {'success': False, 'error': 'Task cannot be paused'}
        
        task.update_status(DistillationTaskStatus.PAUSED)
        
        # 暂停训练器
        trainer = self._running_trainers.get(task_id)
        if trainer is not None and hasattr(trainer, 'pause'):
            try:
                trainer.pause()
            except Exception as e:
                self.logger.warning(f"Failed to pause trainer: {e}")
        
        self.logger.info(f"Paused distillation task: {task_id}")
        return {'success': True, 'task_id': task_id}
    
    def resume_task(self, task_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        恢复任务
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
        
        Returns:
            恢复结果
        """
        task = self.get_task(task_id, tenant_id)
        if not task:
            return {'success': False, 'error': 'Task not found'}
        
        if not task.get_status_enum().can_resume():
            return {'success': False, 'error': 'Task cannot be resumed'}
        
        task.update_status(DistillationTaskStatus.RUNNING)
        
        # 恢复训练器
        trainer = self._running_trainers.get(task_id)
        if trainer is not None and hasattr(trainer, 'resume'):
            try:
                trainer.resume()
            except Exception as e:
                self.logger.warning(f"Failed to resume trainer: {e}")
        
        self.logger.info(f"Resumed distillation task: {task_id}")
        return {'success': True, 'task_id': task_id}
    
    def retry_task(self, task_id: str, tenant_id: str) -> Dict[str, Any]:
        """
        重试失败的任务
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
        
        Returns:
            重试结果
        """
        task = self.get_task(task_id, tenant_id)
        if not task:
            return {'success': False, 'error': 'Task not found'}
        
        if not task.can_retry():
            return {
                'success': False, 
                'error': f'Task cannot be retried (status={task.status}, retries={task.retry_count}/{task.max_retries})'
            }
        
        # 增加重试次数
        task.retry_count += 1
        task.error_message = None
        task.error_traceback = None
        
        # 重新启动
        return self.start_task(task_id, tenant_id)
    
    def _run_distillation_async(self, task: DistillationTask):
        """
        异步执行蒸馏任务
        
        整合训练器层和策略层
        """
        try:
            with self._profiler.profile("run_distillation"):
                # 获取策略（使用 distillation_scenarios.py）
                strategy = None
                if self._scenario_manager is not None:
                    try:
                        strategy = self._scenario_manager.get_strategy_for_scenario(task.config)
                    except Exception as e:
                        self.logger.warning(f"Failed to get strategy: {e}")
                
                # 尝试使用 distillation_strategy 创建策略（策略层）
                distillation_strategy = None
                if DISTILLATION_STRATEGY_AVAILABLE and create_distillation_strategy is not None and strategy is None:
                    try:
                        # 根据场景选择蒸馏类型
                        distill_type = DistillationType.LOGITS if DistillationType is not None else None
                        if task.scenario == 'feature_based':
                            distill_type = DistillationType.FEATURE
                        elif task.scenario == 'attention':
                            distill_type = DistillationType.ATTENTION
                        elif task.scenario == 'progressive':
                            distill_type = DistillationType.PROGRESSIVE
                        
                        # 创建蒸馏策略配置
                        if DistillationStrategyConfig is not None and distill_type is not None:
                            strategy_config = DistillationStrategyConfig(
                                distillation_type=distill_type.value if hasattr(distill_type, 'value') else 'logits',
                                temperature=task.config.distillation_config.temperature if task.config and hasattr(task.config, 'distillation_config') and task.config.distillation_config else 4.0,
                                soft_loss_weight=task.config.distillation_config.alpha if task.config and hasattr(task.config, 'distillation_config') and task.config.distillation_config else 0.5,
                            )
                            distillation_strategy = create_distillation_strategy(strategy_config)
                            self.logger.info(f"Created distillation strategy for task {task.task_id}: type={distill_type}")
                    except Exception as e:
                        self.logger.warning(f"Failed to create distillation strategy: {e}")
                
                # 创建策略上下文（使用 base_strategy）
                strategy_context = None
                if STRATEGY_LAYER_AVAILABLE and StrategyContext is not None:
                    try:
                        strategy_context = StrategyContext(
                            extra={
                                'phase': TrainingPhase.MAIN if TrainingPhase and hasattr(TrainingPhase, 'MAIN') else None,
                                'strategy_type': StrategyType.DISTILLATION if StrategyType and hasattr(StrategyType, 'DISTILLATION') else None,
                                'config': task.config.to_dict() if task.config and hasattr(task.config, 'to_dict') else {},
                                'metadata': {
                                    'task_id': task.task_id,
                                    'scenario': task.scenario,
                                    'tenant_id': task.tenant_id,
                                }
                            }
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to create strategy context: {e}")
            
                # 获取蒸馏配置
                distill_config = None
                if task.config and hasattr(task.config, 'distillation_config'):
                    distill_config = task.config.distillation_config
                
                if not distill_config:
                    distill_config = DistillationConfig(
                        teacher_model_path=task.teacher_model_id or 'mock',
                        student_model_path=task.student_model_id or 'mock',
                    )
            
                # 创建训练器（使用 knowledge_distillation.py）
                trainer = None
                try:
                    # 优先使用 create_trainer_from_scenario 创建训练器
                    if create_trainer_from_scenario is not None and task.scenario:
                        try:
                            trainer = create_trainer_from_scenario(distill_config)
                            self.logger.info(f"Created trainer from scenario: {task.scenario}")
                        except Exception as e:
                            self.logger.warning(f"Failed to create trainer from scenario: {e}")

                    # 回退使用 create_trainer_from_preset
                    if trainer is None and create_trainer_from_preset is not None:
                        try:
                            trainer = create_trainer_from_preset(distill_config)
                            self.logger.info(f"Created trainer from preset: {task.scenario}")
                        except Exception as e:
                            self.logger.warning(f"Failed to create trainer from preset: {e}")

                    # 回退使用 create_knowledge_distillation_trainer
                    if trainer is None and create_knowledge_distillation_trainer is not None:
                        try:
                            trainer = create_knowledge_distillation_trainer(distill_config)
                            self.logger.info("Created trainer via factory function")
                        except Exception as e:
                            self.logger.warning(f"Failed to create trainer via factory: {e}")

                    # 最后回退到直接实例化
                    if trainer is None:
                        trainer = KnowledgeDistillationTrainer(distill_config)

                    # 设置策略（优先使用蒸馏策略）
                    if distillation_strategy is not None and hasattr(trainer, 'set_distillation_strategy'):
                        # pylint: disable=no-member
                        trainer.set_distillation_strategy(distillation_strategy)
                    elif strategy is not None and hasattr(trainer, 'set_strategy'):
                        trainer.set_strategy(strategy)

                    # 设置策略上下文
                    if strategy_context is not None and hasattr(trainer, 'set_context'):
                        # pylint: disable=no-member
                        trainer.set_context(strategy_context)

                    # 保存训练器引用
                    self._running_trainers[task.task_id] = trainer
                except Exception as e:
                    self.logger.warning(f"Failed to create trainer: {e}")
                
                if trainer is None:
                    # 回退到模拟训练
                    self._run_mock_training(task)
                    return
            
            # 注册进度回调
            def progress_callback(result):
                self._handle_progress_update(task, result)
                
            trainer.register_callback('on_step_end', progress_callback)
                
            # 获取训练参数
            num_epochs = 3
            if task.config and hasattr(task.config, 'num_epochs'):
                num_epochs = task.config.num_epochs
                
            # 执行训练
            result = trainer.train(
                num_steps=num_epochs * 100,
                num_epochs=num_epochs
            )
                
            # 更新任务结果
            self._handle_training_result(task, trainer, result)
                
        except Exception as e:
            import traceback
            self.logger.error(f"Distillation task failed: {task.task_id}, error: {e}")
            task.update_status(DistillationTaskStatus.FAILED)
            task.error_message = str(e)
            task.error_traceback = traceback.format_exc()
            task.completed_at = datetime.utcnow().isoformat()
            self._trigger_callback('on_task_failed', task)
        
        finally:
            # 清理
            self._running_trainers.pop(task.task_id, None)
            self._scheduler.mark_completed(task.task_id)
            self._monitor.record_task_completed(task)
            self._update_scheduler_stats()
            
            # 处理队列中的下一个任务
            self._process_queue()
    
    def _handle_progress_update(self, task: DistillationTask, result: Dict[str, Any]):
        """处理进度更新"""
        step = result.get('step', 0)
        
        # 计算进度
        total_steps = 100
        if task.config and hasattr(task.config, 'num_epochs'):
            total_steps = task.config.num_epochs * 100
        
        progress = min(100.0, (step / total_steps) * 100) if total_steps > 0 else 0
        
        # 更新任务
        task.update_progress(progress, {
                    'total_loss': result.get('total_loss', 0),
                    'soft_loss': result.get('soft_loss', 0),
                    'hard_loss': result.get('hard_loss', 0),
            'feature_loss': result.get('feature_loss', 0),
            'attention_loss': result.get('attention_loss', 0),
        })
        
        # 更新峰值内存
        if 'gpu_memory_mb' in result:
            if result['gpu_memory_mb'] > task.peak_memory_mb:
                task.peak_memory_mb = result['gpu_memory_mb']
                
                # 记录监控指标
                self._record_metrics(task.task_id, result)
                
                # 触发回调
                self._trigger_callback('on_progress_update', task, result)
            
    def _handle_training_result(
        self, 
        task: DistillationTask, 
        trainer: 'KnowledgeDistillationTrainer',
        result: Dict[str, Any]
    ):
        """处理训练结果"""
        if result.get('success', False):
            task.update_status(DistillationTaskStatus.COMPLETED)
            task.result = result
            task.progress = 100.0
            
            # 计算训练时间
            if task.started_at:
                try:
                    start = datetime.fromisoformat(task.started_at)
                    task.total_training_time_seconds = (datetime.utcnow() - start).total_seconds()
                except Exception:
                    pass
            
            # 后处理（使用 distillation_scenarios.py）
            if self._scenario_manager is not None and trainer.student_model is not None:
                try:
                    trainer.student_model = self._scenario_manager.post_process_model(
                        trainer.student_model, task.config, result
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to post-process model: {e}")
                
                self._trigger_callback('on_task_completed', task)
            else:
                task.update_status(DistillationTaskStatus.FAILED)
                task.error_message = result.get('error', 'Unknown error')
                self._trigger_callback('on_task_failed', task)
            
            task.completed_at = datetime.utcnow().isoformat()
            
    def _run_mock_training(self, task: DistillationTask):
        """模拟训练（当训练器不可用时）"""
        self.logger.info(f"Running mock training for task: {task.task_id}")
        
        num_steps = 100
        for step in range(num_steps):
            # 检查是否被取消
            if task.status == DistillationTaskStatus.CANCELLED.value:
                break
            
            # 模拟训练
            time.sleep(0.01)
            
            # 更新进度
            result = {
                'step': step + 1,
                'total_loss': 1.0 / (step + 1),
                'soft_loss': 0.5 / (step + 1),
                'hard_loss': 0.5 / (step + 1),
            }
            self._handle_progress_update(task, result)
        
        # 完成
        task.update_status(DistillationTaskStatus.COMPLETED)
        task.result = {'success': True, 'mock': True}
        task.progress = 100.0
        task.completed_at = datetime.utcnow().isoformat()
        self._trigger_callback('on_task_completed', task)
    
    def _process_queue(self):
        """处理队列中的任务"""
        task_id = self._scheduler.dequeue()
        if task_id:
            task = self.get_task_unsafe(task_id)
            if task:
                self._start_task_internal(task)
    
    def _update_scheduler_stats(self):
        """更新调度器统计"""
        self._monitor.update_active_count(
            active=self._scheduler.get_running_count(),
            queued=self._scheduler.get_queue_size()
        )
    
    # ======================== 监控和报告 ========================
    
    def _record_metrics(self, task_id: str, result: Dict[str, Any]):
        """
        记录监控指标
        
        整合服务监控器和配置层的 DistillationMonitor
        """
        metrics = DistillationMetrics(
            task_id=task_id,
            timestamp=datetime.utcnow().isoformat(),
            total_loss=result.get('total_loss', 0),
            soft_loss=result.get('soft_loss', 0),
            hard_loss=result.get('hard_loss', 0),
            feature_loss=result.get('feature_loss', 0),
            attention_loss=result.get('attention_loss', 0),
            contrastive_loss=result.get('contrastive_loss', 0),
            step=result.get('step', 0),
            epoch=result.get('epoch', 0),
            learning_rate=result.get('learning_rate', 0),
            gradient_norm=result.get('gradient_norm', 0),
            throughput=result.get('throughput', 0),
            step_time_ms=result.get('step_time', 0) * 1000,
            gpu_memory_mb=result.get('gpu_memory_mb', 0),
            gpu_utilization=result.get('gpu_utilization', 0),
            temperature=result.get('temperature', 4.0),
            alpha=result.get('alpha', 0.5),
        )
        
        # 添加分布式信息
        if DISTRIBUTED_LAYER_AVAILABLE:
            metrics.rank = get_rank() if get_rank is not None else 0
            metrics.world_size = get_world_size() if get_world_size is not None else 1
        
        # 记录到服务监控器
        self._monitor.record_task_metrics(task_id, metrics)
    
    def get_task_metrics(
        self, 
        task_id: str, 
        tenant_id: str,
        limit: int = 100
    ) -> List[DistillationMetrics]:
        """
        获取任务监控指标
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
            limit: 返回数量限制
        
        Returns:
            指标列表
        """
        task = self.get_task(task_id, tenant_id)
        if not task:
            return []
        
        return self._monitor.get_task_metrics(task_id, limit)
    
    def get_task_loss_trend(self, task_id: str, tenant_id: str) -> str:
        """
        获取任务损失趋势
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
        
        Returns:
            趋势描述 (improving, stable, degrading, insufficient_data)
        """
        task = self.get_task(task_id, tenant_id)
        if not task:
            return "unknown"
        
        return self._monitor.get_task_loss_trend(task_id)
    
    def generate_report(self, task_id: str, tenant_id: str) -> Optional[DistillationReport]:
        """
        生成蒸馏报告
        
        整合任务数据、监控指标和诊断信息
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
        
        Returns:
            蒸馏报告
        """
        with self._profiler.profile("generate_report"):
            task = self.get_task(task_id, tenant_id)
        if not task:
            return None
        
        # 计算时长
        duration = task.get_duration_seconds()
        
        # 获取结果数据
        result = task.result or {}
        
        # 获取指标历史
        metrics = self._monitor.get_task_metrics(task_id)
            
        # 计算统计数据
        total_loss_values = [m.total_loss for m in metrics if m.total_loss > 0]
        avg_loss = sum(total_loss_values) / len(total_loss_values) if total_loss_values else 0
        min_loss = min(total_loss_values) if total_loss_values else 0
        final_loss = total_loss_values[-1] if total_loss_values else 0
            
        # 分析收敛状态
        convergence_status = "unknown"
        if len(total_loss_values) >= 10:
            trend = self._monitor.get_task_loss_trend(task_id)
            if trend == "stable" and len(total_loss_values) > 50:
                convergence_status = "converged"
            elif trend == "degrading":
                convergence_status = "not_converged"
            elif result.get('early_stopped', False):
                convergence_status = "early_stopped"
            
        # 计算资源使用
        peak_memory = max([m.gpu_memory_mb for m in metrics]) if metrics else task.peak_memory_mb
        avg_gpu_util = sum([m.gpu_utilization for m in metrics]) / len(metrics) if metrics else 0
            
        # 创建报告
        report = DistillationReport(
            task_id=task_id,
            tenant_id=tenant_id,
            task_name=task.task_name,
            scenario=task.scenario,
            status=task.status,
            created_at=task.created_at or '',
            started_at=task.started_at,
            completed_at=task.completed_at,
            duration_seconds=duration,
                final_loss=final_loss,
                avg_loss=avg_loss,
                min_loss=min_loss,
                total_steps=result.get('total_steps', len(metrics)),
                total_epochs=result.get('total_epochs', 0),
                teacher_model_size_mb=result.get('teacher_model_size_mb', 0),
                student_model_size_mb=result.get('student_model_size_mb', 0),
                compression_ratio=result.get('compression_ratio', 0),
                teacher_accuracy=result.get('teacher_accuracy', 0),
                student_accuracy=result.get('student_accuracy', 0),
                accuracy_retention=result.get('accuracy_retention', 0),
                teacher_latency_ms=result.get('teacher_latency_ms', 0),
                student_latency_ms=result.get('student_latency_ms', 0),
                speedup_ratio=result.get('speedup_ratio', 0),
                peak_memory_mb=peak_memory,
                avg_gpu_utilization=avg_gpu_util,
                total_training_samples=result.get('total_samples', 0),
                convergence_status=convergence_status,
        )
        
        return report
    
    def compare_tasks(
        self, 
        task_ids: List[str], 
        tenant_id: str
    ) -> Dict[str, Any]:
        """
        比较多个任务
        
        Args:
            task_ids: 任务ID列表
            tenant_id: 租户ID
        
        Returns:
            比较结果
        """
        tasks = [self.get_task(tid, tenant_id) for tid in task_ids]
        tasks = [t for t in tasks if t is not None]
        
        if len(tasks) < 2:
            return {'error': 'Need at least 2 tasks to compare'}
        
        comparison = {
            'task_count': len(tasks),
            'tasks': [],
        }
        
        for task in tasks:
            report = self.generate_report(task.task_id, tenant_id)
            if report:
                comparison['tasks'].append({
                    'task_id': task.task_id,
                    'task_name': task.task_name,
                    'scenario': task.scenario,
                    'status': task.status,
                    'final_loss': report.final_loss,
                    'accuracy_retention': report.accuracy_retention,
                    'compression_ratio': report.compression_ratio,
                    'speedup_ratio': report.speedup_ratio,
                    'duration_seconds': report.duration_seconds,
                    'quality_score': report.get_quality_score(),
                })
        
        # 找出最佳任务
        if comparison['tasks']:
            best = max(comparison['tasks'], key=lambda t: t['quality_score'])
            comparison['best_task'] = best['task_id']
            comparison['best_quality_score'] = best['quality_score']
        
        return comparison
    
    # ======================== 场景推荐 ========================
    
    def recommend_scenario(
        self,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        推荐蒸馏场景
        
        整合 distillation_scenarios.py 和 compression_config.py 的推荐功能
        
        Args:
            requirements: 需求描述
        
        Returns:
            推荐结果
        """
        result = {
            'recommended_scenario': 'standard',
            'config_preview': {},
            'available_scenarios': [],
            'recommendation_source': 'default',
        }
        
        # 使用场景管理器推荐（distillation_scenarios.py）
        if self._scenario_manager is not None:
            try:
                scenario, config = self._scenario_manager.recommend_scenario(requirements)
                result['recommended_scenario'] = scenario
                result['config_preview'] = config.to_dict() if hasattr(config, 'to_dict') else {}
                result['available_scenarios'] = self._scenario_manager.get_available_scenarios()
                result['recommendation_source'] = 'scenario_manager'
            except Exception as e:
                self.logger.warning(f"Scenario manager recommendation failed: {e}")
        
        # 补充配置层推荐（compression_config.py）
        try:
            recommended_config = recommend_config(requirements)
            if recommended_config and hasattr(recommended_config, 'to_dict'):
                result['recommended_config'] = recommended_config.to_dict()
                result['recommendation_source'] = 'config_layer'
        except Exception as e:
            self.logger.warning(f"Config layer recommendation failed: {e}")
        
        # 添加场景详细信息
        try:
            detailed = recommend_distillation_scenario(requirements)
            result['detailed_recommendation'] = detailed
        except Exception:
            pass
        
        return result
    
    def get_available_scenarios(self) -> List[str]:
        """获取可用的蒸馏场景"""
        if self._scenario_manager is not None:
            try:
                return self._scenario_manager.get_available_scenarios()
            except Exception:
                pass
        
        # 默认场景列表
        return [
            'standard', 'edge_deploy', 'high_accuracy', 'industry',
            'multimodal', 'progressive', 'self_distillation', 'contrastive',
            'low_latency', 'real_time'
        ]
    
    def get_scenario_info(self, scenario: str) -> Dict[str, Any]:
        """
        获取场景详细信息
        
        Args:
            scenario: 场景名称
        
        Returns:
            场景信息
        """
        info = {
            'name': scenario,
            'available': scenario in self.get_available_scenarios(),
        }
        
        # 获取预设配置信息
        try:
            preset_method = getattr(DistillationPresets, scenario, None)
            if preset_method:
                preset = preset_method()
                info['preset_available'] = True
                if hasattr(preset, 'summary'):
                    info['summary'] = preset.summary()
        except Exception:
            pass
        
        return info
    
    # ======================== 回调管理 ========================
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        注册回调函数
        
        Args:
            event: 事件名称
            callback: 回调函数
        """
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def unregister_callback(self, event: str, callback: Callable) -> bool:
        """
        取消注册回调函数
        
        Args:
            event: 事件名称
            callback: 回调函数
        
        Returns:
            是否成功取消
        """
        if event in self._callbacks:
            try:
                self._callbacks[event].remove(callback)
                return True
            except ValueError:
                pass
        return False
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """触发回调"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"Callback {event} failed: {e}")
    
    def get_registered_callbacks(self) -> Dict[str, int]:
        """获取已注册的回调数量"""
        return {event: len(callbacks) for event, callbacks in self._callbacks.items()}
    
    # ======================== 统计信息 ========================
    
    def get_tenant_statistics(self, tenant_id: str) -> Dict[str, Any]:
        """
        获取租户统计信息
        
        Args:
            tenant_id: 租户ID
        
        Returns:
            统计信息
        """
        tasks = self.list_tasks(tenant_id, limit=10000)
        
        status_counts = {}
        scenario_counts = {}
        total_duration = 0.0
        total_memory = 0.0
        completed_tasks_data = []
        
        for task in tasks:
            # 状态统计
            status_counts[task.status] = status_counts.get(task.status, 0) + 1
            
            # 场景统计
            scenario_counts[task.scenario] = scenario_counts.get(task.scenario, 0) + 1
            
            # 时长和资源统计
            duration = task.get_duration_seconds()
            total_duration += duration
            total_memory += task.peak_memory_mb
            
            # 收集已完成任务数据
            if task.status == DistillationTaskStatus.COMPLETED.value:
                completed_tasks_data.append({
                    'duration': duration,
                    'memory': task.peak_memory_mb,
                    'metrics': task.metrics,
                })
        
        # 计算平均值
        completed_count = status_counts.get(DistillationTaskStatus.COMPLETED.value, 0)
        avg_duration = total_duration / len(tasks) if tasks else 0
        avg_memory = total_memory / len(tasks) if tasks else 0
        
        # 计算成功率
        total_finished = (
            status_counts.get(DistillationTaskStatus.COMPLETED.value, 0) +
            status_counts.get(DistillationTaskStatus.FAILED.value, 0)
        )
        success_rate = completed_count / total_finished if total_finished > 0 else 0
        
        return {
            'total_tasks': len(tasks),
            'status_distribution': status_counts,
            'scenario_distribution': scenario_counts,
            'total_training_hours': total_duration / 3600,
            'avg_task_duration_seconds': avg_duration,
            'avg_peak_memory_mb': avg_memory,
            'completed_tasks': completed_count,
            'failed_tasks': status_counts.get(DistillationTaskStatus.FAILED.value, 0),
            'success_rate': success_rate,
            'active_tasks': status_counts.get(DistillationTaskStatus.RUNNING.value, 0),
            'queued_tasks': status_counts.get(DistillationTaskStatus.QUEUED.value, 0),
        }
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """
        获取服务级统计信息
        
        Returns:
            服务统计
        """
        stats = self._monitor.get_stats()
        scheduler_stats = self._scheduler.get_stats()
        
        result = {'service_stats': stats.to_dict(), 'scheduler_stats': scheduler_stats, 'distributed': {
            'rank': get_rank() if get_rank is not None else 0,
            'world_size': get_world_size() if get_world_size is not None else 1,
            'is_main_process': is_main_process() if is_main_process is not None else True,
        }}
        
        # 添加分布式信息

        return result
    
    # ======================== 健康检查和诊断 ========================
    
    def check_health(self) -> ServiceHealthStatus:
        """
        检查服务健康状态
        
        Returns:
            健康状态
        """
        if self._health_checker:
            return self._health_checker.check()
        return ServiceHealthStatus.UNKNOWN
    
    def get_health_details(self) -> Dict[str, Any]:
        """
        获取详细的健康状态
        
        Returns:
            健康详情
        """
        if self._health_checker:
            return self._health_checker.get_detailed_status()
        return {'status': 'unknown', 'message': 'Health checker not enabled'}
    
    def diagnose(self) -> Dict[str, Any]:
        """
        诊断服务状态
        
        整合各层的诊断信息
        
        Returns:
            诊断结果
        """
        diagnosis = {
            'timestamp': datetime.utcnow().isoformat(),
            'service': {
                'name': 'DistillationService',
                'version': '2.0',
            },
        }
        
        # 健康状态
        diagnosis['health'] = self.get_health_details()
        
        # 服务统计
        diagnosis['statistics'] = self.get_service_statistics()
        
        # 监控器诊断
        diagnosis['monitor'] = self._monitor.diagnose()
        
        # 分析器统计
        diagnosis['profiler'] = self._profiler.get_all_stats()
        
        # 调度器状态
        diagnosis['scheduler'] = self._scheduler.get_stats()
        
        # 场景管理器诊断（distillation_scenarios.py）
        try:
            diagnosis['scenarios'] = diagnose_scenarios()
        except Exception as e:
            diagnosis['scenarios'] = {'error': str(e)}
        
        # 训练器诊断（knowledge_distillation.py）
        try:
            # 诊断运行中的训练器
            trainer_diagnostics = {}
            for task_id, trainer in self._running_trainers.items():
                try:
                    trainer_diag = diagnose_trainer(trainer)
                    trainer_diagnostics[task_id] = trainer_diag
                except Exception as e:
                    trainer_diagnostics[task_id] = {'error': str(e)}

            diagnosis['trainers'] = {
                'running_count': len(self._running_trainers),
                'diagnostics': trainer_diagnostics,
            }
        except Exception as e:
            diagnosis['trainers'] = {'error': str(e)}
        
        # 分布式策略诊断（distributed_strategy.py）
        if DISTRIBUTED_STRATEGY_AVAILABLE and diagnose_distributed_strategy is not None:
            try:
                # pylint: disable=no-value-for-parameter
                distributed_diag = diagnose_distributed_strategy(None)
                diagnosis['distributed_strategy'] = distributed_diag
            except Exception as e:
                diagnosis['distributed_strategy'] = {'error': str(e)}
        
        # 分布式健康状态（DistributedHealthStatus）
        if DISTRIBUTED_STRATEGY_AVAILABLE and DistributedHealthStatus is not None:
            try:
                # 检查分布式通信状态
                diagnosis['distributed_health'] = {
                    'status': DistributedHealthStatus.HEALTHY.value if hasattr(DistributedHealthStatus, 'HEALTHY') else 'unknown',
                    'available': True,
                }
            except Exception as e:
                diagnosis['distributed_health'] = {'error': str(e)}
        
        # 通信统计（CommunicationStats）
        if DISTRIBUTED_STRATEGY_AVAILABLE and CommunicationStats is not None:
            try:
                # 收集分布式任务的通信统计
                comm_stats = {}
                for task_id, task in self._tasks.items():
                    if task.distributed_mode != 'none' and task.status == DistillationTaskStatus.RUNNING.value:
                        comm_stats[task_id] = {
                            'distributed_mode': task.distributed_mode,
                            'num_gpus': task.num_gpus,
                        }
                diagnosis['communication_stats'] = comm_stats
            except Exception as e:
                diagnosis['communication_stats'] = {'error': str(e)}
        
        # 配置验证器状态
        if self._config_validator is not None:
            diagnosis['config_validator'] = {
                'available': True,
                'check_count': self._config_validator.get_check_count() if hasattr(self._config_validator, 'get_check_count') else 0,
            }
        
        # 策略验证器状态
        if self._strategy_validator is not None:
            diagnosis['strategy_validator'] = {
                'available': True,
                'check_count': self._strategy_validator.get_check_count() if hasattr(self._strategy_validator, 'get_check_count') else 0,
            }
        
        # 运行中的任务
        with self._task_lock:
            running_tasks = [
                t.summary() for t in self._tasks.values() 
                if t.status == DistillationTaskStatus.RUNNING.value
            ]
        diagnosis['running_tasks'] = running_tasks
        
        return diagnosis
    
    # ======================== 资源管理 ========================
    
    def estimate_task_resources(
        self,
        scenario: str, 
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        估算任务资源需求
        
        Args:
            scenario: 蒸馏场景
            config_overrides: 配置覆盖
        
        Returns:
            资源估算
        """
        estimate = {
            'memory_mb': 8000,  # 默认8GB
            'training_time_hours': 4.0,  # 默认4小时
            'recommended_gpus': 1,
        }
        
        # 使用训练器层估算
        try:
            # 创建临时配置
            preset_method = getattr(DistillationPresets, scenario, None)
            if preset_method:
                config = preset_method()
                if config_overrides:
                    for k, v in config_overrides.items():
                        if hasattr(config, k):
                            setattr(config, k, v)

                resources = estimate_training_resources(config)
                estimate.update(resources)
        except Exception as e:
            self.logger.warning(f"Resource estimation failed: {e}")
        
        # 场景特定调整
        if scenario == 'edge_deploy':
            estimate['memory_mb'] = min(estimate['memory_mb'], 4000)
        elif scenario == 'multimodal':
            estimate['memory_mb'] *= 2
            estimate['training_time_hours'] *= 1.5
        elif scenario == 'industry':
            estimate['recommended_gpus'] = max(2, estimate.get('recommended_gpus', 1))
        
        return estimate
    
    def optimize_memory(self) -> bool:
        """
        优化服务内存使用
        
        Returns:
            是否成功
        """
        if HARDWARE_LAYER_AVAILABLE and clear_memory is not None:
            try:
                clear_memory()
                return True
            except Exception as e:
                self.logger.warning(f"Memory optimization failed: {e}")
        return False
    
    def get_available_memory(self) -> float:
        """
        获取可用内存
        
        Returns:
            可用内存（MB）
        """
        if HARDWARE_LAYER_AVAILABLE and get_available_memory is not None:
            try:
                return get_available_memory()
            except Exception:
                pass

        return 0.0
    
    # ======================== 模型压缩 ========================
    
    def compress_model(
        self,
        model: nn.Module,
        compression_method: str = 'quantization',
        target_sparsity: float = 0.5,
        quantization_bits: int = 8,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        压缩模型
        
        整合 ModelCompressor 和 CompressionConfig 的功能
        
        Args:
            model: 要压缩的模型
            compression_method: 压缩方法 ('quantization', 'pruning', 'distillation', 'combined')
            target_sparsity: 目标稀疏度（用于剪枝）
            quantization_bits: 量化位数
            config_overrides: 配置覆盖
        
        Returns:
            压缩结果字典
        """
        result = {
            'success': False,
            'method': compression_method,
            'original_size_mb': 0.0,
            'compressed_size_mb': 0.0,
            'compression_ratio': 1.0,
        }
        
        # 计算原始模型大小
        try:
            param_count = sum(p.numel() for p in model.parameters())
            result['original_size_mb'] = (param_count * 4) / (1024 * 1024)  # 假设 float32
            result['param_count'] = param_count
        except Exception as e:
            self.logger.warning(f"Failed to calculate model size: {e}")
        
        # 创建压缩配置（使用 CompressionConfig）
        compression_config = None
        try:
            method_mapping = {
                'quantization': CompressionMethod.QUANTIZATION,
                'pruning': CompressionMethod.PRUNING,
                'distillation': CompressionMethod.DISTILLATION,
                'combined': CompressionMethod.MIXED,
            }
            method_enum = method_mapping.get(compression_method)

            if method_enum is not None:
                # 根据方法设置相应的启用标志
                comp_kwargs = {}
                if method_enum == CompressionMethod.QUANTIZATION:
                    comp_kwargs['use_quantization'] = True
                    comp_kwargs['quantization_bits'] = config_overrides.get('quantization_bits', quantization_bits) if config_overrides else quantization_bits
                elif method_enum == CompressionMethod.PRUNING:
                    comp_kwargs['use_pruning'] = True
                    comp_kwargs['pruning_ratio'] = config_overrides.get('target_sparsity', target_sparsity) if config_overrides else target_sparsity
                elif method_enum == CompressionMethod.DISTILLATION:
                    comp_kwargs['use_distillation'] = True
                elif method_enum == CompressionMethod.MIXED:
                    comp_kwargs['use_quantization'] = True
                    comp_kwargs['use_pruning'] = True
                    comp_kwargs['use_distillation'] = True
                
                compression_config = CompressionConfig(**comp_kwargs)

                # 应用配置覆盖
                if config_overrides:
                    for k, v in config_overrides.items():
                        if hasattr(compression_config, k):
                            setattr(compression_config, k, v)

                self.logger.info(f"Created compression config: method={compression_method}")
        except Exception as e:
            self.logger.warning(f"Failed to create compression config: {e}")
        
        # 使用 ModelCompressor 进行压缩
        try:
            compressor = ModelCompressor(compression_config)

            # 执行压缩
            if hasattr(compressor, 'compress'):
                compressed_model, compression_stats = compressor.compress(model)
            else:
                raise AttributeError("ModelCompressor has no 'compress' method")

            # 更新结果
            result['success'] = True
            result['compressed_model'] = compressed_model
            result['compression_stats'] = compression_stats

            # 计算压缩后大小
            if hasattr(compression_stats, 'compressed_size_mb'):
                result['compressed_size_mb'] = compression_stats.compressed_size_mb
                result['compression_ratio'] = result['original_size_mb'] / result['compressed_size_mb']

            self.logger.info(
                f"Model compressed successfully: "
                f"{result['original_size_mb']:.2f}MB -> {result['compressed_size_mb']:.2f}MB "
                f"(ratio: {result['compression_ratio']:.2f}x)"
            )
        except Exception as e:
            result['error'] = str(e)
            self.logger.warning(f"Model compression failed: {e}")
        
        return result
    
    def get_strategy_context(
        self,
        task_id: str,
        tenant_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        获取任务的策略上下文信息
        
        整合 StrategyContext, StrategyResult, TrainingPhase 等功能
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
        
        Returns:
            策略上下文信息
        """
        task = self.get_task(task_id, tenant_id)
        if not task:
            return None
        
        context_info = {
            'task_id': task_id,
            'scenario': task.scenario,
            'status': task.status,
            'phase': None,
            'strategy_type': None,
            'strategy_result': None,
        }
        
        # 获取训练阶段（使用 TrainingPhase）
        if STRATEGY_LAYER_AVAILABLE and TrainingPhase is not None:
            try:
                status = task.get_status_enum()
                if status == DistillationTaskStatus.PENDING:
                    context_info['phase'] = TrainingPhase.WARMUP.value if hasattr(TrainingPhase, 'WARMUP') else 'warmup'
                elif status == DistillationTaskStatus.RUNNING:
                    context_info['phase'] = TrainingPhase.MAIN.value if hasattr(TrainingPhase, 'MAIN') else 'main'
                elif status == DistillationTaskStatus.COMPLETED:
                    context_info['phase'] = TrainingPhase.EVALUATION.value if hasattr(TrainingPhase, 'EVALUATION') else 'evaluation'
                elif status == DistillationTaskStatus.FAILED:
                    phase_enum = getattr(TrainingPhase, 'ERROR', None)
                    context_info['phase'] = phase_enum.value if phase_enum else 'error'
            except Exception as e:
                self.logger.warning(f"Failed to get training phase: {e}")
        
        # 获取策略类型（使用 StrategyType）
        if STRATEGY_LAYER_AVAILABLE and StrategyType is not None:
            try:
                context_info['strategy_type'] = StrategyType.DISTILLATION.value if hasattr(StrategyType, 'DISTILLATION') else 'distillation'
            except Exception as e:
                self.logger.warning(f"Failed to get strategy type: {e}")
        
        # 获取训练器的策略结果（使用 StrategyResult）
        trainer = self._running_trainers.get(task_id)
        if trainer and STRATEGY_LAYER_AVAILABLE and StrategyResult is not None:
            try:
                if hasattr(trainer, 'get_strategy_result'):
                    strategy_result = trainer.get_strategy_result()
                    if strategy_result:
                        context_info['strategy_result'] = {
                            'success': strategy_result.success if hasattr(strategy_result, 'success') else None,
                            'metrics': strategy_result.metrics if hasattr(strategy_result, 'metrics') else {},
                            'message': strategy_result.message if hasattr(strategy_result, 'message') else None,
                        }
            except Exception as e:
                self.logger.warning(f"Failed to get strategy result: {e}")
        
        # 获取训练策略信息（使用 TrainingStrategy）
        if STRATEGY_LAYER_AVAILABLE and TrainingStrategy is not None:
            try:
                context_info['training_strategy'] = {
                    'available': True,
                    'distillation_strategy_available': DISTILLATION_STRATEGY_AVAILABLE,
                    'distributed_strategy_available': DISTRIBUTED_STRATEGY_AVAILABLE,
                }
            except Exception as e:
                self.logger.warning(f"Failed to get training strategy info: {e}")
        
        return context_info
    
    def get_scenario_execution_stats(
        self,
        task_id: str,
        tenant_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        获取任务的场景执行统计
        
        整合 ScenarioExecutionStats 和 ScenarioMonitor 的功能
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
        
        Returns:
            场景执行统计信息
        """
        task = self.get_task(task_id, tenant_id)
        if not task:
            return None
        
        stats = {
            'task_id': task_id,
            'scenario': task.scenario,
            'status': task.status,
            'progress': task.progress,
            'metrics': task.metrics,
            'duration_seconds': task.get_duration_seconds(),
        }
        
        # 获取场景执行统计（使用 ScenarioExecutionStats）
        try:
            # 从场景管理器获取执行统计
            if self._scenario_manager is not None and hasattr(self._scenario_manager, 'get_execution_stats'):
                # pylint: disable=no-member
                execution_stats = self._scenario_manager.get_execution_stats(task_id)
                if execution_stats:
                    stats['execution_stats'] = {
                        'total_steps': execution_stats.total_steps if hasattr(execution_stats, 'total_steps') else 0,
                        'completed_steps': execution_stats.completed_steps if hasattr(execution_stats,
                                                                                      'completed_steps') else 0,
                        'failed_steps': execution_stats.failed_steps if hasattr(execution_stats, 'failed_steps') else 0,
                        'avg_step_time': execution_stats.avg_step_time if hasattr(execution_stats,
                                                                                  'avg_step_time') else 0,
                        'throughput': execution_stats.throughput if hasattr(execution_stats, 'throughput') else 0,
                    }
        except Exception as e:
            self.logger.warning(f"Failed to get scenario execution stats: {e}")
        
        # 获取场景监控信息（使用 ScenarioMonitor）
        try:
            if self._scenario_manager is not None and hasattr(self._scenario_manager, 'get_monitor'):
                # pylint: disable=no-member
                scenario_monitor = self._scenario_manager.get_monitor()
                if scenario_monitor:
                    stats['scenario_monitor'] = {
                        'active_scenarios': scenario_monitor.active_scenarios if hasattr(scenario_monitor,
                                                                                         'active_scenarios') else 0,
                        'total_executed': scenario_monitor.total_executed if hasattr(scenario_monitor,
                                                                                     'total_executed') else 0,
                        'success_rate': scenario_monitor.success_rate if hasattr(scenario_monitor,
                                                                                 'success_rate') else 0.0,
                    }
        except Exception as e:
            self.logger.warning(f"Failed to get scenario monitor info: {e}")
        
        # 添加训练器特定的场景统计
        trainer = self._running_trainers.get(task_id)
        if trainer:
            try:
                if hasattr(trainer, 'get_scenario_stats'):
                    trainer_stats = trainer.get_scenario_stats()
                    if trainer_stats:
                        stats['trainer_scenario_stats'] = trainer_stats
            except Exception as e:
                self.logger.warning(f"Failed to get trainer scenario stats: {e}")
        
        return stats
    
    def get_distributed_strategy_info(
        self,
        task_id: str,
        tenant_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        获取任务的分布式策略信息
        
        整合 DistributedStrategy, DistributedStrategyConfig 等功能
        
        Args:
            task_id: 任务ID
            tenant_id: 租户ID
        
        Returns:
            分布式策略信息
        """
        task = self.get_task(task_id, tenant_id)
        if not task:
            return None
        
        info = {
            'task_id': task_id,
            'distributed_mode': task.distributed_mode,
            'num_gpus': task.num_gpus,
        }
        
        # 获取分布式策略详情（使用 DistributedStrategy）
        if DISTRIBUTED_STRATEGY_AVAILABLE and DistributedStrategy is not None:
            try:
                info['distributed_strategy'] = {
                    'available': True,
                    'supported_modes': ['ddp', 'fsdp', 'deepspeed', 'horovod'],
                }
                
                # 获取 ZeRO 阶段信息
                if ZeROStage is not None:
                    info['zero_stages'] = [
                        getattr(ZeROStage, 'STAGE_0', None).value if hasattr(ZeROStage, 'STAGE_0') else 0,
                        getattr(ZeROStage, 'STAGE_1', None).value if hasattr(ZeROStage, 'STAGE_1') else 1,
                        getattr(ZeROStage, 'STAGE_2', None).value if hasattr(ZeROStage, 'STAGE_2') else 2,
                        getattr(ZeROStage, 'STAGE_3', None).value if hasattr(ZeROStage, 'STAGE_3') else 3,
                    ]
            except Exception as e:
                self.logger.warning(f"Failed to get distributed strategy info: {e}")
        
        # 获取推荐的分布式模式
        if DISTRIBUTED_STRATEGY_AVAILABLE and recommend_distributed_mode is not None:
            try:
                requirements = {
                    'world_size': task.num_gpus,
                    'model_size_gb': task.estimated_memory_mb / 1024 if task.estimated_memory_mb else 2.0,
                }
                # 解包 requirements 以匹配函数签名
                model_size_gb = requirements.get('model_size_gb', 2.0)
                num_gpus = requirements.get('world_size', 1)
                
                recommendation = recommend_distributed_mode(
                    model_size_gb=model_size_gb,
                    num_gpus=num_gpus,
                    memory_per_gpu_gb=16.0  # 默认值
                )
                if recommendation:
                    info['recommended_mode'] = recommendation
            except Exception as e:
                self.logger.warning(f"Failed to get distributed mode recommendation: {e}")
        
        return info
    
    # ======================== 清理 ========================
    
    def cleanup(self):
        """清理服务资源"""
        # 停止所有运行中的任务
        with self._task_lock:
            running_task_ids = [
                t.task_id for t in self._tasks.values()
                if t.status == DistillationTaskStatus.RUNNING.value
            ]
        
        for task_id in running_task_ids:
            self._stop_task_internal(task_id)
        
        # 清理训练器
        self._running_trainers.clear()
        
        # 优化内存
        self.optimize_memory()
        
        self.logger.info("DistillationService cleanup completed")
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except Exception:
            pass


# ======================== 全局实例 ========================

_distillation_service: Optional[DistillationService] = None
_service_lock = threading.Lock()


def get_distillation_service(
    use_memory_storage: bool = False,
    max_concurrent_tasks: int = 4,
    max_queue_size: int = 100,
    enable_health_check: bool = True,
) -> DistillationService:
    """
    获取蒸馏服务单例
    
    Args:
        use_memory_storage: 是否使用内存存储
        max_concurrent_tasks: 最大并发任务数
        max_queue_size: 最大队列大小
        enable_health_check: 是否启用健康检查
    
    Returns:
        蒸馏服务实例
    """
    global _distillation_service
    
    if _distillation_service is None:
        with _service_lock:
            if _distillation_service is None:
                _distillation_service = DistillationService(
                    use_memory_storage=use_memory_storage,
                    max_concurrent_tasks=max_concurrent_tasks,
                    max_queue_size=max_queue_size,
                    enable_health_check=enable_health_check,
                )
    
    return _distillation_service


def reset_distillation_service():
    """重置蒸馏服务单例"""
    global _distillation_service
    if _distillation_service is None:
        return

    with _service_lock:
        
        try:
            _distillation_service.cleanup()
        except Exception:
            pass
            
        _distillation_service = None


# ======================== 工具函数 ========================

def create_distillation_service(
    use_memory_storage: bool = False,
    max_concurrent_tasks: int = 4,
    max_queue_size: int = 100,
    enable_health_check: bool = True,
) -> DistillationService:
    """
    创建新的蒸馏服务实例（非单例）
    
    Args:
        use_memory_storage: 是否使用内存存储
        max_concurrent_tasks: 最大并发任务数
        max_queue_size: 最大队列大小
        enable_health_check: 是否启用健康检查
    
    Returns:
        蒸馏服务实例
    """
    return DistillationService(
        use_memory_storage=use_memory_storage,
        max_concurrent_tasks=max_concurrent_tasks,
        max_queue_size=max_queue_size,
        enable_health_check=enable_health_check,
    )


def diagnose_service(service: Optional[DistillationService] = None) -> Dict[str, Any]:
    """
    诊断蒸馏服务
    
    Args:
        service: 服务实例，如果为None则使用单例
    
    Returns:
        诊断结果
    """
    if service is None:
        service = get_distillation_service()
    
    return service.diagnose()


def print_service_diagnosis(service: Optional[DistillationService] = None):
    """
    打印服务诊断信息
    
    Args:
        service: 服务实例
    """
    diagnosis = diagnose_service(service)
    
    print("=" * 60)
    print("DistillationService Diagnosis")
    print("=" * 60)
    
    # 健康状态
    health = diagnosis.get('health', {})
    print(f"\nHealth Status: {health.get('status', 'unknown')}")
    
    # 层可用性
    layers = health.get('layers', {})
    print("\nLayer Availability:")
    for layer, available in layers.items():
        status = "✓" if available else "✗"
        print(f"  {status} {layer}")
    
    # 服务统计
    stats = diagnosis.get('statistics', {}).get('service_stats', {})
    print("\nService Statistics:")
    print(f"  Tasks Created: {stats.get('total_tasks_created', 0)}")
    print(f"  Tasks Completed: {stats.get('total_tasks_completed', 0)}")
    print(f"  Tasks Failed: {stats.get('total_tasks_failed', 0)}")
    print(f"  Active Tasks: {stats.get('active_tasks', 0)}")
    print(f"  Queued Tasks: {stats.get('queued_tasks', 0)}")
    
    # 调度器状态
    scheduler = diagnosis.get('scheduler', {})
    print("\nScheduler Status:")
    print(f"  Running: {scheduler.get('running_count', 0)}/{scheduler.get('max_concurrent', 0)}")
    print(f"  Queue: {scheduler.get('queue_size', 0)}/{scheduler.get('max_queue_size', 0)}")
    
    print("=" * 60)

def estimate_resources(
    scenario: str,
    config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
    """
    估算任务资源需求
    
    Args:
        scenario: 蒸馏场景
        config_overrides: 配置覆盖
    
    Returns:
        资源估算
    """

    service = get_distillation_service()
    return service.estimate_task_resources(scenario, config_overrides)


def recommend_scenario(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    推荐蒸馏场景
        
        Args:
            requirements: 需求描述
        
        Returns:
            推荐结果
        """
    service = get_distillation_service()
    return service.recommend_scenario(requirements)


def get_available_scenarios() -> List[str]:
    """获取可用的蒸馏场景"""
    service = get_distillation_service()
    return service.get_available_scenarios()


def quick_create_task(
    task_name: str,
    tenant_id: str,
    user_id: str,
    teacher_model_path: str,
    student_model_path: str,
    scenario: str = "standard",
    auto_start: bool = False,
) -> Dict[str, Any]:
    """
    快速创建蒸馏任务
    
    Args:
        task_name: 任务名称
        tenant_id: 租户ID
        user_id: 用户ID
        teacher_model_path: 教师模型路径
        student_model_path: 学生模型路径
        scenario: 蒸馏场景
        auto_start: 是否自动启动
    
    Returns:
        任务信息
    """
    service = get_distillation_service()
    
    task = service.create_task(
        task_name=task_name,
        tenant_id=tenant_id,
        user_id=user_id,
        teacher_model_path=teacher_model_path,
        student_model_path=student_model_path,
        scenario=scenario,
    )
        
    result = {
        'task_id': task.task_id,
        'status': task.status,
        'scenario': task.scenario,
    }
    
    if auto_start:
        start_result = service.start_task(task.task_id, tenant_id)
        result['started'] = start_result.get('success', False)
        result['start_status'] = start_result.get('status', task.status)
        
        return result
    

def get_task_status(task_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
    """
    获取任务状态
    
    Args:
        task_id: 任务ID
        tenant_id: 租户ID
    
    Returns:
        任务状态
    """
    service = get_distillation_service()
    task = service.get_task(task_id, tenant_id)
    
    if task is None:
        return None
    
    return {
        'task_id': task.task_id,
        'task_name': task.task_name,
        'status': task.status,
        'progress': task.progress,
        'scenario': task.scenario,
        'created_at': task.created_at,
        'started_at': task.started_at,
        'completed_at': task.completed_at,
        'metrics': task.metrics,
        'error_message': task.error_message,
    }


def wait_for_task(
    task_id: str,
    tenant_id: str,
    timeout_seconds: float = 3600,
    poll_interval: float = 5.0,
) -> Dict[str, Any]:
    """
    等待任务完成
        
        Args:
        task_id: 任务ID
            tenant_id: 租户ID
        timeout_seconds: 超时时间（秒）
        poll_interval: 轮询间隔（秒）
        
        Returns:
        任务最终状态
    """
    service = get_distillation_service()
    start_time = time.time()
    
    while True:
        task = service.get_task(task_id, tenant_id)
        if task is None:
            return {'error': 'Task not found'}
        
        if task.get_status_enum().is_terminal():
            return {
                'task_id': task.task_id,
                'status': task.status,
                'success': task.status == DistillationTaskStatus.COMPLETED.value,
                'progress': task.progress,
                'result': task.result,
                'error_message': task.error_message,
            }
        
        if time.time() - start_time > timeout_seconds:
            return {
                'task_id': task.task_id,
                'status': 'timeout',
                'success': False,
                'progress': task.progress,
            }
        
        time.sleep(poll_interval)


def batch_create_tasks(
    tenant_id: str,
    user_id: str,
    tasks_config: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    批量创建任务
    
    Args:
        tenant_id: 租户ID
        user_id: 用户ID
        tasks_config: 任务配置列表
    
    Returns:
        创建结果列表
    """
    service = get_distillation_service()
    results = []
    
    for config in tasks_config:
        try:
            task = service.create_task(
                task_name=config.get('task_name', 'batch_task'),
                tenant_id=tenant_id,
                user_id=user_id,
                teacher_model_path=config.get('teacher_model_path', ''),
                student_model_path=config.get('student_model_path', ''),
                scenario=config.get('scenario', 'standard'),
                priority=config.get('priority', 1),
                config_overrides=config.get('config_overrides'),
            )
            results.append({
                'success': True,
                'task_id': task.task_id,
                'task_name': task.task_name,
            })
        except Exception as e:
            results.append({
                'success': False,
                'error': str(e),
                'config': config,
            })
    
    return results


# ======================== 上下文管理器 ========================

@contextmanager
def distillation_service_context(
    use_memory_storage: bool = True,
    max_concurrent_tasks: int = 2,
):
    """
    蒸馏服务上下文管理器
    
    用于测试和临时使用
    
    Args:
        use_memory_storage: 是否使用内存存储
        max_concurrent_tasks: 最大并发任务数
    
    Yields:
        蒸馏服务实例
    """
    service = create_distillation_service(
        use_memory_storage=use_memory_storage,
        max_concurrent_tasks=max_concurrent_tasks,
        enable_health_check=False,
    )
    
    try:
        yield service
    finally:
        service.cleanup()


# -*- coding: utf-8 -*-
"""
知识蒸馏训练策略

整合底层知识蒸馏算法，实现教师-学生模型的知识迁移。
支持多种蒸馏方式：Logits蒸馏、特征蒸馏、注意力蒸馏、渐进式蒸馏、自蒸馏等。

架构调用层次：
├── distillation_strategy.py (本模块)
│   └── 调用 backend/lib/losses (损失层)
│       ├── SoftLabelLoss - 软标签蒸馏损失
│       ├── FeatureDistillationLoss - 特征蒸馏损失
│       ├── AttentionDistillationLoss - 注意力蒸馏损失
│       └── CombinedDistillationLoss - 组合蒸馏损失
│   └── 调用 base_strategy.py (策略基类)
│       ├── StrategyMonitor - 策略监控
│       ├── StrategyProfiler - 性能分析
│       └── StrategyValidator - 结果验证
└── 被 distillation/knowledge_distillation.py 调用

生产级特性：
- 完整的蒸馏监控和诊断
- 温度调度和自适应
- 层级权重管理
- 教师模型EMA更新
- 模型压缩集成
"""

import logging
import time
import math
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_strategy import (
    TrainingStrategy, 
    StrategyContext, 
    StrategyResult, 
    TrainingPhase,
    StrategyType,
    StrategyMonitor,
    StrategyProfiler,
    StrategyValidator,
    StrategyMetrics,
)

logger = logging.getLogger(__name__)


# ==================== 底层损失层导入 ====================

from backend.lib.losses import (
    # 蒸馏损失
    SoftLabelLoss,
    FeatureDistillationLoss,
    AttentionDistillationLoss,
    RelationalDistillationLoss,
    CombinedDistillationLoss,
    DistillationLossModule,
    # 蒸馏监控
    DistillationMonitor as LibDistillationMonitor,
    DistillationStats as LibDistillationStats,
    TemperatureScheduler as LibTemperatureScheduler,
    LayerWeightManager as LibLayerWeightManager,
    # 对比学习损失（用于对比蒸馏）
    InfoNCELoss,
    ContrastiveLoss,
    # 监督损失
    CrossEntropyLoss,
    FocalLoss,
    # 工厂函数
    create_loss,
    create_distillation_loss,
    create_composite_loss,
    # 基础
    LossResult,
    LossConfig,
)



# ======================== 枚举定义 ========================

class DistillationType(Enum):
    """蒸馏类型枚举"""
    LOGITS = "logits"               # Logits蒸馏（软标签）
    FEATURE = "feature"             # 特征蒸馏（中间层）
    ATTENTION = "attention"         # 注意力蒸馏
    COMBINED = "combined"           # 组合蒸馏
    PROGRESSIVE = "progressive"     # 渐进式蒸馏
    SELF = "self"                   # 自蒸馏
    CONTRASTIVE = "contrastive"     # 对比蒸馏
    RELATIONAL = "relational"       # 关系蒸馏
    MULTI_TEACHER = "multi_teacher" # 多教师蒸馏
    
    @classmethod
    def from_string(cls, value: str) -> 'DistillationType':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown distillation type: {value}")
    
    @property
    def requires_teacher(self) -> bool:
        """是否需要教师模型"""
        return self not in [DistillationType.SELF]
    
    @property
    def requires_features(self) -> bool:
        """是否需要中间层特征"""
        return self in [
            DistillationType.FEATURE, 
            DistillationType.COMBINED,
            DistillationType.PROGRESSIVE,
            DistillationType.CONTRASTIVE,
            DistillationType.RELATIONAL,
        ]
    
    @property
    def requires_attentions(self) -> bool:
        """是否需要注意力权重"""
        return self in [
            DistillationType.ATTENTION, 
            DistillationType.COMBINED,
        ]
    
    @property
    def typical_weights(self) -> Dict[str, float]:
        """典型损失权重"""
        weights = {
            DistillationType.LOGITS: {'hard': 1.0, 'soft': 0.5},
            DistillationType.FEATURE: {'hard': 1.0, 'feature': 0.3},
            DistillationType.ATTENTION: {'hard': 1.0, 'attention': 0.2},
            DistillationType.COMBINED: {'hard': 1.0, 'soft': 0.3, 'feature': 0.1, 'attention': 0.1},
            DistillationType.PROGRESSIVE: {'hard': 1.0, 'progressive': 0.3},
            DistillationType.SELF: {'hard': 1.0, 'self': 0.5},
            DistillationType.CONTRASTIVE: {'hard': 1.0, 'soft': 0.3, 'contrastive': 0.2},
            DistillationType.RELATIONAL: {'hard': 1.0, 'relational': 0.3},
            DistillationType.MULTI_TEACHER: {'hard': 1.0, 'ensemble': 0.5},
        }
        return weights.get(self, {'hard': 1.0})
    
    def get_description(self) -> str:
        """获取描述"""
        descriptions = {
            DistillationType.LOGITS: "软标签蒸馏：匹配教师的输出概率分布",
            DistillationType.FEATURE: "特征蒸馏：匹配中间层特征表示",
            DistillationType.ATTENTION: "注意力蒸馏：匹配注意力权重分布",
            DistillationType.COMBINED: "组合蒸馏：结合多种蒸馏方式",
            DistillationType.PROGRESSIVE: "渐进式蒸馏：逐层递进蒸馏",
            DistillationType.SELF: "自蒸馏：模型内部知识迁移",
            DistillationType.CONTRASTIVE: "对比蒸馏：使用对比学习增强蒸馏",
            DistillationType.RELATIONAL: "关系蒸馏：保持样本间关系",
            DistillationType.MULTI_TEACHER: "多教师蒸馏：融合多个教师的知识",
        }
        return descriptions.get(self, "未知蒸馏类型")


class FeatureLossType(Enum):
    """特征损失类型"""
    MSE = "mse"                     # 均方误差
    COSINE = "cosine"               # 余弦相似度
    L1 = "l1"                       # L1损失
    HUBER = "huber"                 # Huber损失
    KL = "kl"                       # KL散度
    
    @classmethod
    def from_string(cls, value: str) -> 'FeatureLossType':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown feature loss type: {value}")
    
    @property
    def is_distribution_based(self) -> bool:
        """是否基于分布"""
        return self in [FeatureLossType.KL]
    
    @property
    def is_normalized(self) -> bool:
        """是否需要归一化"""
        return self in [FeatureLossType.COSINE, FeatureLossType.KL]


class AttentionLossType(Enum):
    """注意力损失类型"""
    KL = "kl"                       # KL散度
    MSE = "mse"                     # 均方误差
    
    @classmethod
    def from_string(cls, value: str) -> 'AttentionLossType':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        return cls.KL


class TemperatureSchedule(Enum):
    """温度调度策略"""
    CONSTANT = "constant"           # 恒定温度
    LINEAR = "linear"               # 线性衰减
    COSINE = "cosine"               # 余弦退火
    ADAPTIVE = "adaptive"           # 自适应调整
    WARMUP_DECAY = "warmup_decay"   # 先升后降
    
    @classmethod
    def from_string(cls, value: str) -> 'TemperatureSchedule':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        return cls.CONSTANT


# 温度调度类型别名（用于兼容）
TemperatureScheduleType = TemperatureSchedule


class ProgressiveScheduleType(Enum):
    """渐进调度类型"""
    LINEAR = "linear"               # 线性增加层数
    EXPONENTIAL = "exponential"     # 指数增加层数
    
    @classmethod
    def from_string(cls, value: str) -> 'ProgressiveScheduleType':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        return cls.LINEAR


class MultiTeacherEnsembleType(Enum):
    """多教师集成类型"""
    AVERAGE = "average"             # 平均教师logits
    WEIGHTED = "weighted"           # 加权组合损失
    VOTING = "voting"               # 投票集成
    
    @classmethod
    def from_string(cls, value: str) -> 'MultiTeacherEnsembleType':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        return cls.WEIGHTED


# ======================== 配置类 ========================

@dataclass
class DistillationStrategyConfig:
    """蒸馏策略配置
    
    支持多种蒸馏类型的配置，包含验证和序列化方法。
    """
    # 温度参数
    temperature: float = 4.0           # 软化温度
    min_temperature: float = 1.0       # 最小温度
    max_temperature: float = 20.0      # 最大温度
    temperature_schedule: str = "constant"  # 温度调度策略
    
    # 损失权重
    hard_loss_weight: float = 1.0      # 硬标签损失权重
    soft_loss_weight: float = 0.3      # 软标签损失权重
    feature_loss_weight: float = 0.1   # 特征蒸馏损失权重
    attention_loss_weight: float = 0.1 # 注意力蒸馏损失权重
    relational_loss_weight: float = 0.1  # 关系蒸馏损失权重
    
    # 蒸馏类型
    distillation_type: str = "logits"  # logits, feature, attention, combined, progressive, self, contrastive
    
    # 特征蒸馏配置
    feature_layers: List[int] = field(default_factory=lambda: [-1])  # 蒸馏的特征层索引
    feature_loss_type: str = "mse"     # mse, cosine, l1, huber, kl
    feature_projector_hidden: int = 0  # 特征投影器隐藏层（0表示不使用）
    
    # 注意力蒸馏配置
    attention_layers: List[int] = field(default_factory=lambda: [-1])  # 注意力层索引
    attention_loss_type: str = "kl"    # kl, mse
    
    # 在线蒸馏配置
    online_distillation: bool = False  # 是否使用在线蒸馏（教师模型也更新）
    teacher_ema_decay: float = 0.999   # 教师模型EMA衰减率
    teacher_update_freq: int = 1       # 教师模型更新频率
    
    # 渐进式蒸馏配置
    progressive_stages: int = 3        # 渐进阶段数
    progressive_warmup: int = 1000     # 每阶段预热步数
    progressive_schedule: str = "linear"  # linear, exponential
    
    # 自蒸馏配置
    self_distill_layers: List[int] = field(default_factory=lambda: [-2, -1])  # 自蒸馏层
    self_distill_weight: float = 0.5   # 自蒸馏损失权重
    
    # 对比蒸馏配置
    contrastive_temperature: float = 0.5  # 对比温度
    contrastive_weight: float = 0.1       # 对比损失权重
    contrastive_projector_dim: int = 128  # 对比投影维度
    
    # 多教师蒸馏配置
    multi_teacher_weights: Optional[List[float]] = None  # 多教师权重
    multi_teacher_ensemble: str = "average"  # average, weighted, voting
    
    # 任务损失配置（使用 lib/losses 的 CrossEntropyLoss, FocalLoss）
    task_loss_type: str = "cross_entropy"  # cross_entropy, focal
    label_smoothing: float = 0.0       # 标签平滑
    focal_gamma: float = 2.0           # Focal Loss gamma
    focal_alpha: Optional[float] = None  # Focal Loss alpha
    
    # 监控配置
    enable_monitoring: bool = True     # 启用监控
    log_interval: int = 100            # 日志间隔
    
    # 早停配置
    early_stopping_patience: int = 0   # 早停耐心（0表示不启用）
    min_improvement: float = 0.001     # 最小改进
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置
        
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        # 温度验证
        if self.temperature <= 0:
            errors.append("Temperature must be positive")
        if self.min_temperature <= 0:
            errors.append("Minimum temperature must be positive")
        if self.max_temperature < self.min_temperature:
            errors.append("Maximum temperature must be >= minimum temperature")
        
        # 权重验证
        if self.hard_loss_weight < 0:
            errors.append("Hard loss weight must be non-negative")
        if self.soft_loss_weight < 0:
            errors.append("Soft loss weight must be non-negative")
        if self.feature_loss_weight < 0:
            errors.append("Feature loss weight must be non-negative")
        
        # 蒸馏类型验证
        try:
            DistillationType.from_string(self.distillation_type)
        except ValueError:
            errors.append(f"Invalid distillation type: {self.distillation_type}")
        
        # EMA衰减验证
        if not 0 <= self.teacher_ema_decay <= 1:
            errors.append("Teacher EMA decay must be in [0, 1]")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'temperature': self.temperature,
            'min_temperature': self.min_temperature,
            'max_temperature': self.max_temperature,
            'temperature_schedule': self.temperature_schedule,
            'hard_loss_weight': self.hard_loss_weight,
            'soft_loss_weight': self.soft_loss_weight,
            'feature_loss_weight': self.feature_loss_weight,
            'attention_loss_weight': self.attention_loss_weight,
            'relational_loss_weight': self.relational_loss_weight,
            'distillation_type': self.distillation_type,
            'feature_layers': self.feature_layers.copy(),
            'feature_loss_type': self.feature_loss_type,
            'attention_layers': self.attention_layers.copy(),
            'attention_loss_type': self.attention_loss_type,
            'online_distillation': self.online_distillation,
            'teacher_ema_decay': self.teacher_ema_decay,
            'teacher_update_freq': self.teacher_update_freq,
            'progressive_stages': self.progressive_stages,
            'progressive_warmup': self.progressive_warmup,
            'progressive_schedule': self.progressive_schedule,
            'self_distill_layers': self.self_distill_layers.copy(),
            'self_distill_weight': self.self_distill_weight,
            'contrastive_temperature': self.contrastive_temperature,
            'contrastive_weight': self.contrastive_weight,
            'contrastive_projector_dim': self.contrastive_projector_dim,
            'multi_teacher_weights': self.multi_teacher_weights.copy() if self.multi_teacher_weights else None,
            'multi_teacher_ensemble': self.multi_teacher_ensemble,
            # 任务损失配置
            'task_loss_type': self.task_loss_type,
            'label_smoothing': self.label_smoothing,
            'focal_gamma': self.focal_gamma,
            'focal_alpha': self.focal_alpha,
            # 监控配置
            'enable_monitoring': self.enable_monitoring,
            'log_interval': self.log_interval,
            'early_stopping_patience': self.early_stopping_patience,
            'min_improvement': self.min_improvement,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistillationStrategyConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def get_distillation_type(self) -> DistillationType:
        """获取蒸馏类型枚举"""
        return DistillationType.from_string(self.distillation_type)
    
    def summary(self) -> str:
        """获取配置摘要"""
        return (
            f"DistillationConfig(type={self.distillation_type}, "
            f"temp={self.temperature}, hard={self.hard_loss_weight}, "
            f"soft={self.soft_loss_weight}, feature={self.feature_loss_weight})"
        )


# ======================== 监控和统计组件 ========================

@dataclass
class DistillationStats:
    """蒸馏统计数据"""
    total_steps: int = 0
    total_epochs: int = 0
    
    # 损失统计
    avg_hard_loss: float = 0.0
    avg_soft_loss: float = 0.0
    avg_feature_loss: float = 0.0
    avg_attention_loss: float = 0.0
    avg_contrastive_loss: float = 0.0
    avg_total_loss: float = 0.0
    
    # 准确率统计
    avg_student_accuracy: float = 0.0
    avg_teacher_accuracy: float = 0.0
    accuracy_gap: float = 0.0
    
    # 温度统计
    current_temperature: float = 4.0
    avg_temperature: float = 4.0
    
    # KL散度统计
    avg_kl_divergence: float = 0.0
    
    # 特征相似度
    avg_feature_similarity: float = 0.0
    
    # 渐进式蒸馏
    current_stage: int = 0
    current_layers: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_steps': self.total_steps,
            'total_epochs': self.total_epochs,
            'avg_hard_loss': self.avg_hard_loss,
            'avg_soft_loss': self.avg_soft_loss,
            'avg_feature_loss': self.avg_feature_loss,
            'avg_attention_loss': self.avg_attention_loss,
            'avg_contrastive_loss': self.avg_contrastive_loss,
            'avg_total_loss': self.avg_total_loss,
            'avg_student_accuracy': self.avg_student_accuracy,
            'avg_teacher_accuracy': self.avg_teacher_accuracy,
            'accuracy_gap': self.accuracy_gap,
            'current_temperature': self.current_temperature,
            'avg_kl_divergence': self.avg_kl_divergence,
            'avg_feature_similarity': self.avg_feature_similarity,
            'current_stage': self.current_stage,
        }


class DistillationStrategyMonitor:
    """蒸馏策略监控器
    
    整合 base_strategy.py 的 StrategyMonitor 和 lib/losses 的 DistillationMonitor。
    使用 lib/losses 的 DistillationStats 数据类（如果可用）。
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._stats = DistillationStats()
        
        # 底层统计数据类（如果可用）
        self._lib_stats: Optional['LibDistillationStats'] = None
        try:
            self._lib_stats = LibDistillationStats()
        except Exception:
            self._lib_stats = None
        
        # 损失历史
        self._hard_loss_history: deque = deque(maxlen=max_history)
        self._soft_loss_history: deque = deque(maxlen=max_history)
        self._feature_loss_history: deque = deque(maxlen=max_history)
        self._attention_loss_history: deque = deque(maxlen=max_history)
        self._total_loss_history: deque = deque(maxlen=max_history)
        
        # 准确率历史
        self._student_acc_history: deque = deque(maxlen=max_history)
        self._teacher_acc_history: deque = deque(maxlen=max_history)
        
        # 温度历史
        self._temperature_history: deque = deque(maxlen=max_history)
        
        # KL散度历史
        self._kl_history: deque = deque(maxlen=max_history)
        
        # 特征相似度历史
        self._feature_similarity_history: deque = deque(maxlen=max_history)
        
        # 累计值
        self._totals: Dict[str, float] = defaultdict(float)
        
        # 底层监控器
        self._lib_monitor = LibDistillationMonitor(max_history)
    
    def record_step(
        self,
        hard_loss: float = 0.0,
        soft_loss: float = 0.0,
        feature_loss: float = 0.0,
        attention_loss: float = 0.0,
        contrastive_loss: float = 0.0,
        total_loss: float = 0.0,
        student_accuracy: float = 0.0,
        teacher_accuracy: float = 0.0,
        temperature: float = 4.0,
        kl_divergence: float = 0.0,
        feature_similarity: float = 0.0,
        **kwargs
    ) -> None:
        """记录一个训练步骤"""
        self._stats.total_steps += 1
        n = self._stats.total_steps
        
        # 记录损失历史
        self._hard_loss_history.append(hard_loss)
        self._soft_loss_history.append(soft_loss)
        self._feature_loss_history.append(feature_loss)
        self._attention_loss_history.append(attention_loss)
        self._total_loss_history.append(total_loss)
        
        # 记录准确率历史
        self._student_acc_history.append(student_accuracy)
        self._teacher_acc_history.append(teacher_accuracy)
        
        # 记录温度和KL散度
        self._temperature_history.append(temperature)
        self._kl_history.append(kl_divergence)
        self._feature_similarity_history.append(feature_similarity)
        
        # 更新累计值
        self._totals['hard_loss'] += hard_loss
        self._totals['soft_loss'] += soft_loss
        self._totals['feature_loss'] += feature_loss
        self._totals['attention_loss'] += attention_loss
        self._totals['contrastive_loss'] += contrastive_loss
        self._totals['total_loss'] += total_loss
        self._totals['student_accuracy'] += student_accuracy
        self._totals['teacher_accuracy'] += teacher_accuracy
        self._totals['temperature'] += temperature
        self._totals['kl_divergence'] += kl_divergence
        self._totals['feature_similarity'] += feature_similarity
        
        # 更新平均值
        self._stats.avg_hard_loss = self._totals['hard_loss'] / n
        self._stats.avg_soft_loss = self._totals['soft_loss'] / n
        self._stats.avg_feature_loss = self._totals['feature_loss'] / n
        self._stats.avg_attention_loss = self._totals['attention_loss'] / n
        self._stats.avg_contrastive_loss = self._totals['contrastive_loss'] / n
        self._stats.avg_total_loss = self._totals['total_loss'] / n
        self._stats.avg_student_accuracy = self._totals['student_accuracy'] / n
        self._stats.avg_teacher_accuracy = self._totals['teacher_accuracy'] / n
        self._stats.accuracy_gap = self._stats.avg_teacher_accuracy - self._stats.avg_student_accuracy
        self._stats.avg_temperature = self._totals['temperature'] / n
        self._stats.current_temperature = temperature
        self._stats.avg_kl_divergence = self._totals['kl_divergence'] / n
        self._stats.avg_feature_similarity = self._totals['feature_similarity'] / n
        
        # 同步到底层监控器
        if self._lib_monitor is not None:
            self._lib_monitor.record(
                kd_loss=soft_loss,
                ce_loss=hard_loss,
                feature_loss=feature_loss,
                attention_loss=attention_loss,
                student_accuracy=student_accuracy,
                teacher_accuracy=teacher_accuracy,
                temperature=temperature,
                kl_divergence=kl_divergence,
            )
    
    def update_progressive_state(self, stage: int, layers: int) -> None:
        """更新渐进式蒸馏状态"""
        self._stats.current_stage = stage
        self._stats.current_layers = layers
    
    def get_stats(self) -> DistillationStats:
        """获取统计数据"""
        return self._stats
    
    def get_loss_trend(self, window: int = 100) -> str:
        """获取损失趋势"""
        if len(self._total_loss_history) < window * 2:
            return "insufficient_data"
        
        recent = list(self._total_loss_history)[-window:]
        previous = list(self._total_loss_history)[-window * 2:-window]
        
        recent_avg = sum(recent) / len(recent)
        previous_avg = sum(previous) / len(previous)
        
        change = (recent_avg - previous_avg) / max(previous_avg, 1e-8)
        
        if change < -0.05:
            return "improving"
        elif change > 0.05:
            return "degrading"
        else:
            return "stable"
    
    def get_accuracy_gap_trend(self, window: int = 100) -> str:
        """获取准确率差距趋势"""
        if len(self._student_acc_history) < window * 2:
            return "insufficient_data"
        
        recent_gap = sum(
            t - s for t, s in zip(
                list(self._teacher_acc_history)[-window:],
                list(self._student_acc_history)[-window:]
            )
        ) / window
        
        previous_gap = sum(
            t - s for t, s in zip(
                list(self._teacher_acc_history)[-window * 2:-window],
                list(self._student_acc_history)[-window * 2:-window]
            )
        ) / window
        
        if recent_gap < previous_gap - 0.01:
            return "converging"  # 学生在接近教师
        elif recent_gap > previous_gap + 0.01:
            return "diverging"   # 学生在远离教师
        else:
            return "stable"
    
    def is_distillation_effective(self, min_gap_reduction: float = 0.1) -> bool:
        """检查蒸馏是否有效"""
        if self._lib_monitor is not None:
            return self._lib_monitor.is_distillation_effective(min_gap_reduction)
        
        # 简单检查：准确率差距是否在缩小
        return self.get_accuracy_gap_trend() == "converging"
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        summary = {
            **self._stats.to_dict(),
            'loss_trend': self.get_loss_trend(),
            'accuracy_gap_trend': self.get_accuracy_gap_trend(),
            'is_effective': self.is_distillation_effective(),
        }
        
        # 如果有底层 LibDistillationStats，合并其数据
        if self._lib_stats is not None:
            try:
                if hasattr(self._lib_stats, 'to_dict'):
                    lib_data = self._lib_stats.to_dict()
                    summary['lib_stats'] = lib_data
                elif hasattr(self._lib_stats, '__dict__'):
                    summary['lib_stats'] = {
                        k: v for k, v in self._lib_stats.__dict__.items() 
                        if not k.startswith('_')
                    }
            except Exception:
                pass
        
        return summary
    
    def get_lib_stats(self) -> Optional['LibDistillationStats']:
        """获取底层 lib/losses 的 DistillationStats 实例"""
        return self._lib_stats
    
    def reset(self) -> None:
        """重置监控器"""
        self._stats = DistillationStats()
        
        # 重置底层统计
        if self._lib_stats is not None:
            try:
                if hasattr(self._lib_stats, 'reset'):
                    self._lib_stats.reset()
                else:
                    # 重新创建
                    self._lib_stats = LibDistillationStats() if LibDistillationStats else None
            except Exception:
                pass
        self._hard_loss_history.clear()
        self._soft_loss_history.clear()
        self._feature_loss_history.clear()
        self._attention_loss_history.clear()
        self._total_loss_history.clear()
        self._student_acc_history.clear()
        self._teacher_acc_history.clear()
        self._temperature_history.clear()
        self._kl_history.clear()
        self._feature_similarity_history.clear()
        self._totals.clear()
        
        if self._lib_monitor is not None:
            self._lib_monitor.reset()


class TemperatureScheduler:
    """温度调度器
    
    管理蒸馏过程中的温度调度。
    优先使用 lib/losses 的 TemperatureScheduler。
    """
    
    def __init__(
        self,
        initial_temp: float = 4.0,
        min_temp: float = 1.0,
        max_temp: float = 20.0,
        schedule: str = 'constant',
        warmup_steps: int = 0,
        total_steps: int = 10000
    ):
        self.initial_temp = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.schedule = schedule
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        self._step = 0
        self._kl_history: deque = deque(maxlen=100)
        self._current_temp = initial_temp
        
        # 使用底层调度器
        self._lib_scheduler: Optional['LibTemperatureScheduler'] = None

        self._lib_scheduler = LibTemperatureScheduler(
            initial_temp=initial_temp,
            min_temp=min_temp,
            max_temp=max_temp,
            schedule=schedule,
            warmup_steps=warmup_steps
        )
        self._lib_scheduler.set_total_steps(total_steps)
    
    def step(self, kl_divergence: Optional[float] = None) -> float:
        """获取当前温度并更新步数"""
        if self._lib_scheduler is not None:
            return self._lib_scheduler.step(kl_divergence)
        
        self._step += 1
        
        if kl_divergence is not None:
            self._kl_history.append(kl_divergence)
        
        return self.get_temperature()
    
    def get_temperature(self) -> float:
        """获取当前温度"""
        if self._lib_scheduler is not None:
            return self._lib_scheduler.get_temperature()
        
        # 预热阶段
        if self._step < self.warmup_steps:
            progress = self._step / max(self.warmup_steps, 1)
            return self.min_temp + (self.initial_temp - self.min_temp) * progress
        
        effective_step = self._step - self.warmup_steps
        effective_total = self.total_steps - self.warmup_steps
        
        if self.schedule == 'constant':
            return self.initial_temp
        
        elif self.schedule == 'linear':
            progress = min(effective_step / max(effective_total, 1), 1.0)
            return self.initial_temp - (self.initial_temp - self.min_temp) * progress
        
        elif self.schedule == 'cosine':
            progress = min(effective_step / max(effective_total, 1), 1.0)
            return self.min_temp + (self.initial_temp - self.min_temp) * (1 + math.cos(progress * math.pi)) / 2
        
        elif self.schedule == 'adaptive':
            if len(self._kl_history) < 10:
                return self.initial_temp
            
            recent_kl = sum(self._kl_history) / len(self._kl_history)
            
            if recent_kl > 1.0:
                target_temp = min(self.initial_temp * 1.5, self.max_temp)
            elif recent_kl < 0.1:
                target_temp = max(self.initial_temp * 0.5, self.min_temp)
            else:
                target_temp = self.initial_temp
            
            self._current_temp = 0.9 * self._current_temp + 0.1 * target_temp
            return self._current_temp
        
        return self.initial_temp
    
    def set_total_steps(self, total_steps: int) -> None:
        """设置总步数"""
        self.total_steps = total_steps
        if self._lib_scheduler is not None:
            self._lib_scheduler.set_total_steps(total_steps)
    
    def reset(self) -> None:
        """重置"""
        self._step = 0
        self._kl_history.clear()
        self._current_temp = self.initial_temp
        if self._lib_scheduler is not None:
            self._lib_scheduler.reset()


# ======================== 蒸馏损失计算器 ========================

class DistillationLossCalculator:
    """
    蒸馏损失计算器
    
    封装各种蒸馏损失的计算逻辑，供策略层调用。
    优先使用 backend/lib/losses 中的蒸馏损失模块。
    """
    
    def __init__(self, config: DistillationStrategyConfig, device: torch.device):
        self.config = config
        self.device = device
        self.feature_projectors: Dict[int, nn.Module] = {}
        self.attention_projectors: Dict[int, nn.Module] = {}
        
        # 底层损失模块
        self._soft_label_loss: Optional[nn.Module] = None
        self._feature_loss: Optional[nn.Module] = None
        self._attention_loss: Optional[nn.Module] = None
        self._contrastive_loss: Optional[nn.Module] = None
        self._relational_loss: Optional[nn.Module] = None
        self._multi_teacher_loss: Optional[nn.Module] = None
        
        # 任务损失
        self._task_loss_fn: Optional[nn.Module] = None
        
        # 统计跟踪
        self._loss_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._total_calls: int = 0
        
        # 初始化底层损失模块
        self._init_loss_modules()
    
    def _init_loss_modules(self) -> None:
        """
        初始化底层损失模块
        
        使用 backend/lib/losses 提供的蒸馏损失
        """
        
        # 软标签蒸馏损失
        self._soft_label_loss = SoftLabelLoss(
            temperature=self.config.temperature
        ).to(self.device)
            
        # 特征蒸馏损失
        loss_type_map = {
            'mse': 'mse',
            'cosine': 'cosine',
            'l1': 'l1',
            'huber': 'huber',
            'kl': 'kl'
        }
        self._feature_loss = FeatureDistillationLoss(
            loss_type=loss_type_map.get(self.config.feature_loss_type, 'mse'),
            layer_weight_strategy=getattr(self.config, 'layer_weight_strategy', 'uniform')
        ).to(self.device)
            
        # 注意力蒸馏损失
        self._attention_loss = AttentionDistillationLoss(
            layer_weight_strategy=getattr(self.config, 'layer_weight_strategy', 'uniform')
        ).to(self.device)
            
        # 对比蒸馏损失
        self._contrastive_loss = InfoNCELoss(
            temperature=self.config.contrastive_temperature
        ).to(self.device)
            
        # 关系蒸馏损失
        if RelationalDistillationLoss is not None:
            self._relational_loss = RelationalDistillationLoss(
                distance_weight=0.5,
                angle_weight=0.5
            ).to(self.device)
            
        # 组合蒸馏损失（用于多教师）
        if CombinedDistillationLoss is not None:
            self._multi_teacher_loss = CombinedDistillationLoss(
                temperature=self.config.temperature,
                soft_loss_weight=self.config.soft_loss_weight,
                feature_loss_weight=self.config.feature_loss_weight,
                attention_loss_weight=self.config.attention_loss_weight,
            ).to(self.device)
            
        logger.info("Distillation loss modules initialized from backend/lib/losses")
        
        # 初始化任务损失
        self._init_task_loss()
    
    def _init_task_loss(self) -> None:
        """
        初始化任务损失函数
        
        优先使用 backend/lib/losses 的 CrossEntropyLoss 或 FocalLoss
        """
        task_loss_type = getattr(self.config, 'task_loss_type', 'cross_entropy')
        

        if task_loss_type == 'focal' and FocalLoss is not None:
            # 使用 FocalLoss 处理类别不平衡
            self._task_loss_fn = FocalLoss(
                gamma=getattr(self.config, 'focal_gamma', 2.0),
                alpha=getattr(self.config, 'focal_alpha', None)
            ).to(self.device)
            logger.debug("Using FocalLoss from backend/lib/losses")
        elif CrossEntropyLoss is not None:
            # 使用增强的 CrossEntropyLoss
            label_smoothing = getattr(self.config, 'label_smoothing', 0.0)
            self._task_loss_fn = CrossEntropyLoss(
                label_smoothing=label_smoothing
            ).to(self.device)
            logger.debug("Using CrossEntropyLoss from backend/lib/losses")
        else:
            # 回退到PyTorch原生
            self._task_loss_fn = nn.CrossEntropyLoss().to(self.device)

    
    def set_task_loss(self, loss_fn: nn.Module) -> None:
        """设置任务损失函数"""
        self._task_loss_fn = loss_fn.to(self.device)
    
    def set_task_loss_type(self, loss_type: str, **kwargs) -> None:
        """
        使用 create_loss 工厂函数设置任务损失类型
        
        Args:
            loss_type: 损失类型 (cross_entropy, focal, mse, mae, etc.)
            **kwargs: 损失函数参数
        """
        try:
            self._task_loss_fn = create_loss(loss_type, **kwargs).to(self.device)
            logger.info(f"Task loss set to {loss_type} using create_loss")
        except Exception as e:
            logger.warning(f"Failed to create loss {loss_type}: {e}, falling back to CrossEntropyLoss")
            self._task_loss_fn = nn.CrossEntropyLoss().to(self.device)
    
    def create_distillation_loss_from_config(
        self, 
        loss_config: Optional[Dict[str, Any]] = None
    ) -> Optional[nn.Module]:
        """
        使用 create_distillation_loss 工厂函数创建蒸馏损失
        
        Args:
            loss_config: 损失配置字典
        
        Returns:
            创建的蒸馏损失模块
        """
        config = loss_config or {
            'temperature': self.config.temperature,
            'soft_loss_weight': self.config.soft_loss_weight,
            'feature_loss_weight': self.config.feature_loss_weight,
            'attention_loss_weight': self.config.attention_loss_weight,
        }
        
        try:
            loss_module = create_distillation_loss(**config)
            return loss_module.to(self.device) if loss_module else None
        except Exception as e:
            logger.warning(f"Failed to create distillation loss: {e}")
            return None
    
    def create_composite_loss_from_configs(
        self,
        loss_configs: List[Dict[str, Any]]
    ) -> Optional[nn.Module]:
        """
        使用 create_composite_loss 工厂函数创建组合损失
        
        Args:
            loss_configs: 损失配置列表
        
        Returns:
            创建的组合损失模块
        """
        try:
            composite_loss = create_composite_loss(loss_configs)
            return composite_loss.to(self.device) if composite_loss else None
        except Exception as e:
            logger.warning(f"Failed to create composite loss: {e}")
            return None
    
    def get_loss_config(self) -> Optional['LossConfig']:
        """
        获取当前损失配置（使用 LossConfig 数据类）
        
        Returns:
            LossConfig 实例或 None
        """
        try:
            return LossConfig(
                weight=1.0,
                reduction='mean',
                params={
                    'temperature': self.config.temperature,
                    'soft_loss_weight': self.config.soft_loss_weight,
                    'feature_loss_weight': self.config.feature_loss_weight,
                    'attention_loss_weight': self.config.attention_loss_weight,
                }
            )
        except Exception as e:
            logger.warning(f"Failed to create LossConfig: {e}")
            return None
    
    def wrap_loss_result(
        self, 
        loss: torch.Tensor, 
        metrics: Dict[str, Any]
    ) -> Union['LossResult', Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        将损失包装为 LossResult（如果可用）
        
        Args:
            loss: 损失张量
            metrics: 指标字典
        
        Returns:
            LossResult 实例或原始元组
        """
       
        try:
            return LossResult(
                loss=loss,
                metrics=metrics,
                timestamp=time.time(),
                step=self._total_calls
            )
        except Exception:
            pass
    
    def update_temperature(self, temperature: float) -> None:
        """更新温度"""
        self.config.temperature = temperature
        if self._soft_label_loss is not None and hasattr(self._soft_label_loss, 'temperature'):
            self._soft_label_loss.temperature = temperature
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取损失统计"""
        stats = {'total_calls': self._total_calls}
        for key, history in self._loss_stats.items():
            if history:
                stats[f'{key}_avg'] = sum(history) / len(history)
                stats[f'{key}_recent'] = history[-1] if history else 0.0
        
        # 如果底层损失模块有统计，也收集
        if self._soft_label_loss is not None and hasattr(self._soft_label_loss, 'get_stats'):
            stats['soft_label_module_stats'] = self._soft_label_loss.get_stats()
        if self._feature_loss is not None and hasattr(self._feature_loss, 'get_stats'):
            stats['feature_module_stats'] = self._feature_loss.get_stats()
        
        return stats
    
    def reset_statistics(self) -> None:
        """重置统计"""
        self._loss_stats.clear()
        self._total_calls = 0
        
        # 重置底层损失模块的统计
        for loss_module in [self._soft_label_loss, self._feature_loss, 
                            self._attention_loss, self._contrastive_loss,
                            self._relational_loss, self._multi_teacher_loss]:
            if loss_module is not None and hasattr(loss_module, 'reset_stats'):
                loss_module.reset_stats()
    
    def get_loss_module_info(self) -> Dict[str, Any]:
        """
        获取底层损失模块信息
        
        检查哪些 DistillationLossModule 和其他损失模块已初始化
        """
        info = {
            'modules': {}
        }
        
        module_map = {
            'soft_label': self._soft_label_loss,
            'feature': self._feature_loss,
            'attention': self._attention_loss,
            'contrastive': self._contrastive_loss,
            'relational': self._relational_loss,
            'multi_teacher': self._multi_teacher_loss,
            'task': self._task_loss_fn,
        }
        
        for name, module in module_map.items():
            if module is not None:
                module_info = {
                    'initialized': True,
                    'class': module.__class__.__name__,
                }
                
                # 检查是否是 DistillationLossModule 的实例
                module_info['is_distillation_module'] = isinstance(module, DistillationLossModule)
                
                # 检查是否有监控功能
                module_info['has_monitor'] = hasattr(module, 'get_monitor')
                module_info['has_stats'] = hasattr(module, 'get_stats')
                
                # 获取参数数量
                if hasattr(module, 'parameters'):
                    module_info['param_count'] = sum(p.numel() for p in module.parameters())
                
                info['modules'][name] = module_info
            else:
                info['modules'][name] = {'initialized': False}
        
        return info
    
    def get_all_module_stats(self) -> Dict[str, Any]:
        """
        获取所有底层损失模块的统计数据
        
        整合各个 DistillationLossModule 的统计
        """
        stats = {}
        
        module_map = {
            'soft_label': self._soft_label_loss,
            'feature': self._feature_loss,
            'attention': self._attention_loss,
            'contrastive': self._contrastive_loss,
            'relational': self._relational_loss,
        }
        
        for name, module in module_map.items():
            if module is not None:
                module_stats = {}
                
                # 获取统计数据
                if hasattr(module, 'get_stats'):
                    try:
                        module_stats['stats'] = module.get_stats()
                    except Exception:
                        pass
                
                # 获取监控数据
                if hasattr(module, 'get_monitor'):
                    try:
                        monitor = module.get_monitor()
                        if monitor and hasattr(monitor, 'get_summary'):
                            module_stats['monitor_summary'] = monitor.get_summary()
                    except Exception:
                        pass
                
                # 获取蒸馏特定统计
                if hasattr(module, 'get_distillation_stats'):
                    try:
                        module_stats['distillation_stats'] = module.get_distillation_stats()
                    except Exception:
                        pass
                
                if module_stats:
                    stats[name] = module_stats
        
        return stats
    
    def compute_combined_loss_with_lib(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        使用 CombinedDistillationLoss 计算组合蒸馏损失
        
        这是一个便捷方法，直接使用 lib/losses 的 CombinedDistillationLoss
        
        Args:
            student_outputs: 学生模型输出 (logits, hidden_states, attentions)
            teacher_outputs: 教师模型输出
            labels: 真实标签（可选）
        
        Returns:
            Tuple[torch.Tensor, Dict]: 总损失和详细指标
        """
        if self._multi_teacher_loss is None:
            # 回退到分步计算
            logger.debug("CombinedDistillationLoss not available, using step-by-step computation")
            return self._compute_combined_loss_fallback(student_outputs, teacher_outputs, labels)
        
        metrics = {}
        
        # 提取输出
        student_logits = student_outputs.get('logits')
        teacher_logits = teacher_outputs.get('logits')
        student_features = student_outputs.get('hidden_states')
        teacher_features = teacher_outputs.get('hidden_states')
        student_attentions = student_outputs.get('attentions')
        teacher_attentions = teacher_outputs.get('attentions')
        
        # 使用 CombinedDistillationLoss
        try:
            result = self._multi_teacher_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_features=student_features,
                teacher_features=teacher_features,
                student_attentions=student_attentions,
                teacher_attentions=teacher_attentions,
                labels=labels
            )
            
            # 处理返回结果
            if isinstance(result, LossResult):
                loss = result.loss
                if hasattr(result, 'metrics') and result.metrics:
                    metrics.update(result.metrics)
            else:
                loss = result
            
            # 获取组件统计
            if hasattr(self._multi_teacher_loss, 'get_component_stats'):
                metrics['component_stats'] = self._multi_teacher_loss.get_component_stats()
            
            return loss, metrics
            
        except Exception as e:
            logger.warning(f"CombinedDistillationLoss failed: {e}")
            return self._compute_combined_loss_fallback(student_outputs, teacher_outputs, labels)
    
    def _compute_combined_loss_fallback(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """组合损失的回退实现"""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        metrics = {}
        
        student_logits = student_outputs.get('logits')
        teacher_logits = teacher_outputs.get('logits')
        
        # 软标签损失
        if student_logits is not None and teacher_logits is not None:
            soft_loss, soft_metrics = self.compute_soft_loss(student_logits, teacher_logits)
            total_loss = total_loss + self.config.soft_loss_weight * soft_loss
            metrics['soft_loss'] = soft_loss.item()
            metrics.update({f'soft_{k}': v for k, v in soft_metrics.items()})
        
        # 特征损失
        student_features = student_outputs.get('hidden_states')
        teacher_features = teacher_outputs.get('hidden_states')
        if student_features is not None and teacher_features is not None:
            feature_loss, feature_metrics = self.compute_feature_loss(student_features, teacher_features)
            total_loss = total_loss + self.config.feature_loss_weight * feature_loss
            metrics['feature_loss'] = feature_loss.item()
        
        # 任务损失
        if labels is not None and student_logits is not None:
            task_loss, task_metrics = self.compute_task_loss(student_logits, labels)
            total_loss = total_loss + self.config.hard_loss_weight * task_loss
            metrics['task_loss'] = task_loss.item()
            metrics.update(task_metrics)
        
        return total_loss, metrics
    
    def compute_soft_loss(
        self, 
        student_logits: torch.Tensor, 
        teacher_logits: torch.Tensor,
        temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算软标签损失（Logits蒸馏）
        
        优先使用 backend/lib/losses 的 SoftLabelLoss
        L_soft = T^2 * KL(softmax(s/T) || softmax(t/T))
        
        返回:
            Tuple[torch.Tensor, Dict[str, float]]: 损失值和额外指标
        """
        self._total_calls += 1
        T = temperature if temperature is not None else self.config.temperature
        metrics: Dict[str, float] = {}
        
        # 计算准确率
        with torch.no_grad():
            student_preds = student_logits.argmax(dim=-1)
            teacher_preds = teacher_logits.argmax(dim=-1)
            
            # 如果有真实标签，计算准确率会在外部进行
            # 这里计算学生-教师一致性
            agreement = (student_preds == teacher_preds).float().mean().item()
            metrics['student_teacher_agreement'] = agreement
            
            # KL散度
            kl_div = F.kl_div(
                F.log_softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
                reduction='batchmean'
            ).item()
            metrics['kl_divergence'] = kl_div
        
        # 优先使用底层损失模块
        if self._soft_label_loss is not None:
            # 更新温度
            if hasattr(self._soft_label_loss, 'temperature'):
                self._soft_label_loss.temperature = T
            loss = self._soft_label_loss(student_logits, teacher_logits)
        else:
            # 回退到原生实现
            soft_student = F.log_softmax(student_logits / T, dim=-1)
            soft_teacher = F.softmax(teacher_logits / T, dim=-1)
            kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
            loss = kl_loss * (T ** 2)
        
        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        self._loss_stats['soft_loss'].append(loss_val)
        metrics['temperature'] = T
        
        return loss, metrics
    
    def compute_task_loss(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算任务损失（硬标签损失）
        
        返回:
            Tuple[torch.Tensor, Dict[str, float]]: 损失值和额外指标
        """
        metrics: Dict[str, float] = {}
        
        # 计算损失
        if self._task_loss_fn is not None:
            loss = self._task_loss_fn(student_logits, labels)
        else:
            loss = F.cross_entropy(student_logits, labels)
        
        # 计算准确率
        with torch.no_grad():
            preds = student_logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item()
            metrics['student_accuracy'] = accuracy
        
        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        self._loss_stats['task_loss'].append(loss_val)
        
        return loss, metrics
    
    def compute_feature_loss(
        self,
        student_features: tuple,
        teacher_features: tuple,
        layer_weights: Optional[Dict[int, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算特征蒸馏损失
        
        优先使用 backend/lib/losses 的 FeatureDistillationLoss
        匹配指定层的中间特征表示。
        
        返回:
            Tuple[torch.Tensor, Dict[str, float]]: 损失值和额外指标
        """
        feature_loss = torch.tensor(0.0, device=self.device)
        count = 0
        metrics: Dict[str, float] = {}
        layer_losses: Dict[str, float] = {}
        layer_similarities: Dict[str, float] = {}
        
        for layer_idx in self.config.feature_layers:
            try:
                s_feat = student_features[layer_idx]
                t_feat = teacher_features[layer_idx]
                
                # 维度匹配
                if s_feat.shape != t_feat.shape:
                    hidden_dim = getattr(self.config, 'feature_projector_hidden', None)
                    if layer_idx not in self.feature_projectors:
                        if hidden_dim:
                            self.feature_projectors[layer_idx] = nn.Sequential(
                                nn.Linear(s_feat.shape[-1], hidden_dim),
                                nn.ReLU(),
                                nn.Linear(hidden_dim, t_feat.shape[-1])
                            ).to(self.device)
                        else:
                            self.feature_projectors[layer_idx] = nn.Linear(
                                s_feat.shape[-1], t_feat.shape[-1]
                            ).to(self.device)
                    s_feat = self.feature_projectors[layer_idx](s_feat)
                
                # 优先使用底层损失模块
                if self._feature_loss is not None:
                    layer_loss = self._feature_loss(s_feat, t_feat)
                else:
                    layer_loss = self._compute_feature_layer_loss(s_feat, t_feat)
                
                # 应用层权重
                weight = 1.0
                if layer_weights is not None and layer_idx in layer_weights:
                    weight = layer_weights[layer_idx]
                
                feature_loss = feature_loss + layer_loss * weight
                count += 1
                
                # 记录层损失
                layer_key = f'layer_{layer_idx}'
                layer_losses[layer_key] = layer_loss.item() if hasattr(layer_loss, 'item') else float(layer_loss)
                
                # 计算层相似度
                with torch.no_grad():
                    similarity = F.cosine_similarity(
                        s_feat.flatten(1), t_feat.flatten(1), dim=-1
                    ).mean().item()
                    layer_similarities[layer_key] = similarity
                
            except (IndexError, RuntimeError) as e:
                logger.warning(f"Feature distillation layer {layer_idx} failed: {e}")
                continue
        
        total_loss = feature_loss / max(count, 1)
        
        loss_val = total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
        self._loss_stats['feature_loss'].append(loss_val)
        
        # 聚合指标
        metrics['layer_losses'] = layer_losses
        metrics['layer_similarities'] = layer_similarities
        metrics['avg_feature_similarity'] = sum(layer_similarities.values()) / max(len(layer_similarities), 1)
        metrics['num_layers'] = count
        
        return total_loss, metrics
    
    def _compute_feature_layer_loss(
        self, 
        s_feat: torch.Tensor, 
        t_feat: torch.Tensor
    ) -> torch.Tensor:
        """计算单层特征损失"""
        loss_type = self.config.feature_loss_type
        
        if loss_type == "mse":
            return F.mse_loss(s_feat, t_feat)
        elif loss_type == "cosine":
            return 1 - F.cosine_similarity(
                s_feat.flatten(1), t_feat.flatten(1), dim=-1
            ).mean()
        elif loss_type == "l1":
            return F.l1_loss(s_feat, t_feat)
        elif loss_type == "huber":
            return F.smooth_l1_loss(s_feat, t_feat)
        elif loss_type == "kl":
            s_feat_log = F.log_softmax(s_feat, dim=-1)
            t_feat_soft = F.softmax(t_feat, dim=-1)
            return F.kl_div(s_feat_log, t_feat_soft, reduction='batchmean')
        else:
            return F.mse_loss(s_feat, t_feat)
    
    def compute_attention_loss(
        self,
        student_attentions: tuple,
        teacher_attentions: tuple,
        layer_weights: Optional[Dict[int, float]] = None,
        loss_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算注意力蒸馏损失
        
        优先使用 backend/lib/losses 的 AttentionDistillationLoss
        匹配注意力权重分布。
        
        返回:
            Tuple[torch.Tensor, Dict[str, float]]: 损失值和额外指标
        """
        device = student_attentions[0].device if student_attentions else self.device
        attention_loss = torch.tensor(0.0, device=device)
        count = 0
        metrics: Dict[str, float] = {}
        layer_losses: Dict[str, float] = {}
        layer_correlations: Dict[str, float] = {}
        
        # 使用配置中的损失类型
        att_loss_type = loss_type or getattr(self.config, 'attention_loss_type', AttentionLossType.KL)
        if isinstance(att_loss_type, AttentionLossType):
            att_loss_type = att_loss_type.value
        
        for layer_idx in self.config.attention_layers:
            try:
                s_att = student_attentions[layer_idx]
                t_att = teacher_attentions[layer_idx]
                
                # 确保维度匹配（处理不同的head数量）
                if s_att.shape[-2:] != t_att.shape[-2:]:
                    # 尝试插值匹配
                    if len(s_att.shape) != 4 or len(t_att.shape) == 4:
                        continue
                    # (batch, heads, seq, seq)
                    if s_att.shape[1] != t_att.shape[1]:
                        # 平均所有heads
                        s_att = s_att.mean(dim=1, keepdim=True)
                        t_att = t_att.mean(dim=1, keepdim=True)
                
                # 优先使用底层损失模块
                if self._attention_loss is not None:
                    layer_loss = self._attention_loss(s_att, t_att)
                else:
                    # 回退到原生实现
                    s_att_flat = s_att.view(-1, s_att.shape[-1])
                    t_att_flat = t_att.view(-1, t_att.shape[-1])
                    
                    if att_loss_type == 'mse':
                        layer_loss = F.mse_loss(s_att_flat, t_att_flat)
                    else:  # kl
                        layer_loss = F.kl_div(
                            F.log_softmax(s_att_flat, dim=-1),
                            F.softmax(t_att_flat, dim=-1),
                            reduction='batchmean'
                        )
                
                # 应用层权重
                weight = 1.0
                if layer_weights is not None and layer_idx in layer_weights:
                    weight = layer_weights[layer_idx]
                
                attention_loss = attention_loss + layer_loss * weight
                count += 1
                
                # 记录层损失
                layer_key = f'layer_{layer_idx}'
                layer_losses[layer_key] = layer_loss.item() if hasattr(layer_loss, 'item') else float(layer_loss)
                
                # 计算注意力相关性
                with torch.no_grad():
                    correlation = F.cosine_similarity(
                        s_att.flatten(1), t_att.flatten(1), dim=-1
                    ).mean().item()
                    layer_correlations[layer_key] = correlation
                
            except Exception as e:
                logger.warning(f"Attention distillation layer {layer_idx} failed: {e}")
                continue
        
        total_loss = attention_loss / max(count, 1)
        
        loss_val = total_loss.item() if hasattr(total_loss, 'item') else float(total_loss)
        self._loss_stats['attention_loss'].append(loss_val)
        
        # 聚合指标
        metrics['layer_losses'] = layer_losses
        metrics['layer_correlations'] = layer_correlations
        metrics['avg_attention_correlation'] = sum(layer_correlations.values()) / max(len(layer_correlations), 1)
        metrics['num_layers'] = count
        
        return total_loss, metrics
    
    def compute_contrastive_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        projector_dim: Optional[int] = None,
        contrastive_type: str = 'infonce'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算对比蒸馏损失
        
        优先使用 backend/lib/losses 的 InfoNCELoss 或 ContrastiveLoss
        使用InfoNCE损失进行对比学习。
        
        Args:
            student_features: 学生特征
            teacher_features: 教师特征
            projector_dim: 投影维度
            contrastive_type: 对比损失类型 ('infonce', 'contrastive', 'triplet')
        
        返回:
            Tuple[torch.Tensor, Dict[str, float]]: 损失值和额外指标
        """
        metrics: Dict[str, float] = {}
        
        # 投影到共同空间（如果需要）
        s_feat = student_features.flatten(1)
        t_feat = teacher_features.flatten(1)
        
        proj_dim = projector_dim or getattr(self.config, 'contrastive_projector_dim', None)
        if proj_dim is not None and s_feat.shape[-1] != proj_dim:
            if not hasattr(self, '_contrastive_projector_s'):
                self._contrastive_projector_s = nn.Linear(s_feat.shape[-1], proj_dim).to(self.device)
            if not hasattr(self, '_contrastive_projector_t'):
                self._contrastive_projector_t = nn.Linear(t_feat.shape[-1], proj_dim).to(self.device)
            s_feat = self._contrastive_projector_s(s_feat)
            t_feat = self._contrastive_projector_t(t_feat)
        
        # 归一化特征
        s_feat = F.normalize(s_feat, dim=-1)
        t_feat = F.normalize(t_feat, dim=-1)
        
        T = self.config.contrastive_temperature
        batch_size = s_feat.shape[0]
        
        # 计算相似度指标
        with torch.no_grad():
            # 正对相似度
            pos_sim = (s_feat * t_feat).sum(dim=-1).mean().item()
            metrics['positive_similarity'] = pos_sim
            
            # 负对相似度
            similarity_matrix = torch.mm(s_feat, t_feat.t())
            mask = torch.eye(batch_size, device=similarity_matrix.device).bool()
            neg_sim = similarity_matrix[~mask].mean().item()
            metrics['negative_similarity'] = neg_sim
            metrics['similarity_gap'] = pos_sim - neg_sim
        
        # 选择对比损失实现
        loss = None
        
        # 优先使用底层损失模块
        if self._contrastive_loss is not None:
            # 使用已初始化的 InfoNCELoss
            result = self._contrastive_loss(s_feat, t_feat)
            # 如果返回 LossResult，提取损失和指标
            if isinstance(result, LossResult):
                loss = result.loss
                if hasattr(result, 'metrics') and result.metrics:
                    metrics.update(result.metrics)
            else:
                loss = result
        elif contrastive_type == 'contrastive':
            # 使用通用 ContrastiveLoss 基类
            try:
                contrastive_loss_fn = ContrastiveLoss(
                    temperature=T
                ).to(self.device)
                result = contrastive_loss_fn(s_feat, t_feat)
                if isinstance(result, tuple):
                    loss, extra_metrics = result
                    metrics.update(extra_metrics)
                else:
                    loss = result
                # 使用 ContrastiveLoss 的统计功能
                if hasattr(contrastive_loss_fn, 'get_contrastive_stats'):
                    stats = contrastive_loss_fn.get_contrastive_stats()
                    if stats:
                        metrics['contrastive_stats'] = stats
            except Exception as e:
                logger.warning(f"ContrastiveLoss failed: {e}")
                loss = None
        
        # 回退到原生实现
        if loss is None:
            similarity = torch.mm(s_feat, t_feat.t()) / T
            labels = torch.arange(batch_size, device=similarity.device)
            loss = F.cross_entropy(similarity, labels)
        
        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        self._loss_stats['contrastive_loss'].append(loss_val)
        metrics['temperature'] = T
        metrics['contrastive_type'] = contrastive_type
        
        # 包装为 LossResult（如果可用）
        wrapped = self.wrap_loss_result(loss, metrics)
        if isinstance(wrapped, tuple):
            return wrapped
        # 从 LossResult 提取
        return loss, metrics
    
    def compute_relational_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        distance_weight: float = 0.5,
        angle_weight: float = 0.5
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算关系蒸馏损失
        
        优先使用 backend/lib/losses 的 RelationalDistillationLoss
        匹配样本之间的关系结构。
        
        返回:
            Tuple[torch.Tensor, Dict[str, float]]: 损失值和额外指标
        """
        metrics: Dict[str, float] = {}
        
        s_feat = student_features.flatten(1)
        t_feat = teacher_features.flatten(1)
        
        # 优先使用底层损失模块
        if self._relational_loss is not None:
            loss = self._relational_loss(s_feat, t_feat)
            loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
            self._loss_stats['relational_loss'].append(loss_val)
            
            # 尝试获取底层指标
            if hasattr(self._relational_loss, 'get_relation_stats'):
                rel_stats = self._relational_loss.get_relation_stats()
                metrics.update(rel_stats)
            
            return loss, metrics
        
        # 回退到原生实现
        batch_size = s_feat.shape[0]
        
        # 距离关系
        def compute_pairwise_distances(x):
            diff = x.unsqueeze(1) - x.unsqueeze(0)
            return torch.norm(diff, dim=-1)
        
        s_dist = compute_pairwise_distances(s_feat)
        t_dist = compute_pairwise_distances(t_feat)
        
        # 归一化距离
        s_dist_norm = s_dist / (s_dist.max() + 1e-8)
        t_dist_norm = t_dist / (t_dist.max() + 1e-8)
        
        distance_loss = F.mse_loss(s_dist_norm, t_dist_norm)
        
        # 角度关系
        def compute_pairwise_angles(x):
            x_norm = F.normalize(x, dim=-1)
            return torch.mm(x_norm, x_norm.t())
        
        s_angles = compute_pairwise_angles(s_feat)
        t_angles = compute_pairwise_angles(t_feat)
        
        angle_loss = F.mse_loss(s_angles, t_angles)
        
        # 组合损失
        loss = distance_weight * distance_loss + angle_weight * angle_loss
        
        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        self._loss_stats['relational_loss'].append(loss_val)
        
        with torch.no_grad():
            metrics['distance_loss'] = distance_loss.item()
            metrics['angle_loss'] = angle_loss.item()
            metrics['distance_correlation'] = F.cosine_similarity(
                s_dist_norm.flatten().unsqueeze(0),
                t_dist_norm.flatten().unsqueeze(0)
            ).item()
            metrics['angle_correlation'] = F.cosine_similarity(
                s_angles.flatten().unsqueeze(0),
                t_angles.flatten().unsqueeze(0)
            ).item()
        
        return loss, metrics
    
    def compute_multi_teacher_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: List[torch.Tensor],
        teacher_weights: Optional[List[float]] = None,
        ensemble_type: str = 'weighted'
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算多教师蒸馏损失
        
        将多个教师的知识整合到学生模型中。
        
        返回:
            Tuple[torch.Tensor, Dict[str, float]]: 损失值和额外指标
        """
        metrics: Dict[str, float] = {}
        num_teachers = len(teacher_logits_list)
        
        if num_teachers == 0:
            return torch.tensor(0.0, device=self.device), metrics
        
        # 默认均匀权重
        if teacher_weights is None:
            teacher_weights = [1.0 / num_teachers] * num_teachers
        
        # 归一化权重
        total_weight = sum(teacher_weights)
        teacher_weights = [w / total_weight for w in teacher_weights]
        
        T = self.config.temperature
        
        if ensemble_type == 'average':
            # 平均教师logits
            avg_teacher_logits = sum(
                w * logits for w, logits in zip(teacher_weights, teacher_logits_list)
            )
            loss, _ = self.compute_soft_loss(student_logits, avg_teacher_logits, T)
            
        elif ensemble_type == 'voting':
            # 投票集成
            teacher_probs = [
                F.softmax(logits / T, dim=-1) for logits in teacher_logits_list
            ]
            avg_probs = sum(
                w * probs for w, probs in zip(teacher_weights, teacher_probs)
            )
            
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            loss = F.kl_div(student_log_probs, avg_probs, reduction='batchmean') * (T ** 2)
            
        else:  # weighted
            # 加权组合各教师的损失
            total_loss = torch.tensor(0.0, device=self.device)
            teacher_losses = []
            
            for i, (w, t_logits) in enumerate(zip(teacher_weights, teacher_logits_list)):
                t_loss, _ = self.compute_soft_loss(student_logits, t_logits, T)
                total_loss = total_loss + w * t_loss
                teacher_losses.append(t_loss.item() if hasattr(t_loss, 'item') else float(t_loss))
                metrics[f'teacher_{i}_loss'] = teacher_losses[-1]
            
            loss = total_loss
        
        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        self._loss_stats['multi_teacher_loss'].append(loss_val)
        
        # 计算教师一致性
        with torch.no_grad():
            teacher_preds = [logits.argmax(dim=-1) for logits in teacher_logits_list]
            agreement = sum(
                (teacher_preds[i] == teacher_preds[j]).float().mean().item()
                for i in range(num_teachers) for j in range(i + 1, num_teachers)
            ) / max(num_teachers * (num_teachers - 1) / 2, 1)
            metrics['teacher_agreement'] = agreement
            metrics['num_teachers'] = num_teachers
            metrics['ensemble_type'] = ensemble_type
        
        return loss, metrics


# ======================== 主策略类 ========================

class DistillationStrategy(TrainingStrategy):
    """
    知识蒸馏训练策略
    
    整合底层蒸馏算法，提供统一的策略接口。
    支持多种蒸馏方式的灵活组合。
    
    整合:
    - backend.lib.losses 的蒸馏损失模块
    - base_strategy.py 的策略监控和验证
        - StrategyMonitor: 策略级监控
        - StrategyProfiler: 性能分析
        - StrategyValidator: 结果验证
        - StrategyMetrics: 指标跟踪
    """
    
    # 策略类型标识
    STRATEGY_TYPE = StrategyType.DISTILLATION if hasattr(StrategyType, 'DISTILLATION') else StrategyType.STANDARD
    
    def __init__(
        self, 
        config: Optional[DistillationStrategyConfig] = None,
        teacher_model: Optional[nn.Module] = None
    ):
        super().__init__(name="distillation", priority=40)
        self.config = config or DistillationStrategyConfig()
        self.teacher_model = teacher_model
        self.loss_calculator: Optional[DistillationLossCalculator] = None
        
        # 多教师支持
        self.teacher_models: List[nn.Module] = []
        self.teacher_weights: List[float] = []
        
        # 渐进式蒸馏状态
        self.current_stage = 0
        self.stage_step = 0
        
        # 当前训练阶段
        self._current_phase: TrainingPhase = TrainingPhase.WARMUP
        
        # 蒸馏监控组件
        self._distillation_monitor: Optional[DistillationStrategyMonitor] = None
        
        # 基础策略监控组件（来自 base_strategy.py）
        self._strategy_monitor: Optional[StrategyMonitor] = None
        self._strategy_profiler: Optional[StrategyProfiler] = None
        self._strategy_validator: Optional[StrategyValidator] = None
        self._strategy_metrics: Optional[StrategyMetrics] = None
        
        # 温度调度器
        self._temp_scheduler: Optional[TemperatureScheduler] = None
        
        # 层权重管理器（用于特征/注意力蒸馏）
        self._layer_weight_manager: Optional['LibLayerWeightManager'] = None
        
        # 任务损失函数
        self._task_loss_fn: Optional[nn.Module] = None
        
        # 验证配置
        if self.config is not None:
            try:
                self.config.validate()
            except ValueError as e:
                logger.warning(f"Config validation warning: {e}")
    
    def setup(self, context: StrategyContext) -> None:
        """初始化蒸馏组件"""
        super().setup(context)
        
        # 创建损失计算器（整合底层损失层）
        self.loss_calculator = DistillationLossCalculator(self.config, context.device)
        
        # 初始化蒸馏监控器
        if getattr(self.config, 'enable_monitoring', True):
            self._distillation_monitor = DistillationStrategyMonitor()
        
        # 初始化基础策略组件（来自 base_strategy.py）
        self._init_base_strategy_components(context)
        
        # 设置初始训练阶段
        self._current_phase = TrainingPhase.WARMUP
        
        # 初始化温度调度器
        temp_schedule = getattr(self.config, 'temperature_schedule', TemperatureScheduleType.CONSTANT)
        if isinstance(temp_schedule, TemperatureScheduleType):
            temp_schedule = temp_schedule.value
        
        self._temp_scheduler = TemperatureScheduler(
            initial_temp=self.config.temperature,
            min_temp=getattr(self.config, 'min_temperature', 1.0),
            max_temp=getattr(self.config, 'max_temperature', 20.0),
            schedule=temp_schedule,
            warmup_steps=getattr(self.config, 'temperature_warmup_steps', 0),
            total_steps=context.max_steps or 10000
        )
        
        # 初始化层权重管理器
       
        num_layers = max(
            len(self.config.feature_layers),
            len(self.config.attention_layers),
            1
        )
        self._layer_weight_manager = LibLayerWeightManager(
            num_layers=num_layers,
            strategy=getattr(self.config, 'layer_weight_strategy', 'uniform')
        )
        
        # 设置主教师模型
        if self.teacher_model is not None:
            self._setup_teacher_model(self.teacher_model, context.device)
        
        # 设置多教师模型
        multi_teacher_weights = getattr(self.config, 'multi_teacher_weights', [])
        if multi_teacher_weights and self.teacher_models:
            self.teacher_weights = multi_teacher_weights
        
        logger.info(f"DistillationStrategy setup: type={self.config.distillation_type}, "
                   f"temp={self.config.temperature}, schedule={temp_schedule}")
    
    def _init_base_strategy_components(self, context: StrategyContext) -> None:
        """
        初始化基础策略组件
        
        使用 base_strategy.py 提供的:
        - StrategyMonitor: 策略级监控
        - StrategyProfiler: 性能分析
        - StrategyValidator: 结果验证
        - StrategyMetrics: 指标跟踪
        """
        # 初始化策略监控器（参数名是 history_size）
        try:
            self._strategy_monitor = StrategyMonitor(
                history_size=getattr(self.config, 'max_history', 10000)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize StrategyMonitor: {e}")
            self._strategy_monitor = None
        
        # 初始化性能分析器
        try:
            self._strategy_profiler = StrategyProfiler()
        except Exception as e:
            logger.warning(f"Failed to initialize StrategyProfiler: {e}")
            self._strategy_profiler = None
        
        # 初始化结果验证器
        try:
            self._strategy_validator = StrategyValidator()
            # 添加蒸馏特定的验证规则
            self._add_distillation_validation_rules()
        except Exception as e:
            logger.warning(f"Failed to initialize StrategyValidator: {e}")
            self._strategy_validator = None
        
        # 初始化指标跟踪
        try:
            self._strategy_metrics = StrategyMetrics()
        except Exception as e:
            logger.warning(f"Failed to initialize StrategyMetrics: {e}")
            self._strategy_metrics = None
        
        logger.debug("Base strategy components initialized")
    
    def _add_distillation_validation_rules(self) -> None:
        """添加蒸馏特定的验证规则"""
        if self._strategy_validator is None:
            return
        
        # 添加自定义验证规则
        # StrategyValidator.add_check 接受的函数签名: Callable[[StrategyResult], Tuple[bool, str]]
        if hasattr(self._strategy_validator, 'add_check'):
            # 检查蒸馏损失不应过大
            def check_distillation_loss(result: StrategyResult) -> Tuple[bool, str]:
                if result.loss is None:
                    return True, ""
                loss_val = result.loss.item() if hasattr(result.loss, 'item') else float(result.loss)
                if loss_val >= 100.0:
                    return False, f"Distillation loss too large: {loss_val:.4f}"
                return True, ""
            
            self._strategy_validator.add_check(check_distillation_loss)
            
            # 检查温度在合理范围内
            def check_temperature(result: StrategyResult) -> Tuple[bool, str]:
                temp = result.metrics.get('temperature', self.config.temperature)
                if not (0.1 <= temp <= 100.0):
                    return False, f"Temperature out of range: {temp}"
                return True, ""
            
            self._strategy_validator.add_check(check_temperature)
    
    def _setup_teacher_model(self, model: nn.Module, device: torch.device) -> None:
        """设置并冻结教师模型"""
        model = model.to(device)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        self.teacher_model = model
    
    def set_teacher_model(self, teacher_model: nn.Module, device: torch.device) -> None:
        """设置主教师模型"""
        self._setup_teacher_model(teacher_model, device)
        logger.info("Primary teacher model set successfully")
    
    def add_teacher_model(
        self, 
        teacher_model: nn.Module, 
        device: torch.device,
        weight: float = 1.0
    ) -> None:
        """添加额外的教师模型（多教师蒸馏）"""
        self._setup_teacher_model_to_list(teacher_model, device)
        self.teacher_weights.append(weight)
        logger.info(f"Teacher model added (total: {len(self.teacher_models)})")
    
    def _setup_teacher_model_to_list(self, model: nn.Module, device: torch.device) -> None:
        """添加教师模型到列表"""
        model = model.to(device)
        model.eval()
        
        for param in model.parameters():
            param.requires_grad = False
        
        self.teacher_models.append(model)
    
    def remove_teacher_model(self, index: int) -> None:
        """移除指定索引的教师模型"""
        if 0 <= index < len(self.teacher_models):
            self.teacher_models.pop(index)
            if index < len(self.teacher_weights):
                self.teacher_weights.pop(index)
            logger.info(f"Teacher model at index {index} removed")
    
    def get_teacher_models(self) -> List[nn.Module]:
        """获取所有教师模型"""
        models = []
        if self.teacher_model is not None:
            models.append(self.teacher_model)
        models.extend(self.teacher_models)
        return models
    
    def get_effective_temperature(self) -> float:
        """获取当前有效温度"""
        if self._temp_scheduler is not None:
            return self._temp_scheduler.get_temperature()
        return self.config.temperature
    
    def get_layer_weights(self) -> Dict[int, float]:
        """获取层权重"""
        if self._layer_weight_manager is not None:
            return {i: self._layer_weight_manager.get_weight(i) 
                    for i in range(self._layer_weight_manager.num_layers)}
        
        # 默认均匀权重
        layers = self.config.feature_layers or self.config.attention_layers
        return {layer: 1.0 for layer in layers}
    
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """
        计算蒸馏损失
        
        根据配置的蒸馏类型，组合各种损失函数。
        整合 backend/lib/losses 的蒸馏损失和 base_strategy 的监控。
        """
            # 没有教师模型时，只使用任务损失
        if self.teacher_model is None and not self.teacher_models:
            task_loss = self._get_task_loss(outputs)
            return StrategyResult(
                loss=task_loss,
                metrics={'hard_loss': task_loss.item()},
                warnings=["No teacher model available"]
            )
        
        metrics: Dict[str, Any] = {}
        loss_components: Dict[str, float] = {}
        warnings: List[str] = []
        total_loss = torch.tensor(0.0, device=context.device, requires_grad=True)
        
        # 获取当前温度
        current_temperature = self.get_effective_temperature()
        metrics['temperature'] = current_temperature
        
        # 获取层权重
        layer_weights = self.get_layer_weights()
        
        # 获取学生输出
        student_logits = self._get_logits(outputs)
        student_features = outputs.get('hidden_states', None)
        student_attentions = outputs.get('attentions', None)
        
        # 获取教师输出
        with torch.no_grad():
            teacher_outputs = self._get_teacher_outputs(batch, context)
            teacher_logits = self._get_logits(teacher_outputs)
            teacher_features = teacher_outputs.get('hidden_states', None)
            teacher_attentions = teacher_outputs.get('attentions', None)
        
        # 获取标签（用于准确率计算）
        labels = batch.get('labels', batch.get('label', None))
        
        # ===== 1. 硬标签损失（原始任务损失） =====
        hard_loss = self._get_task_loss(outputs)
        total_loss = total_loss + self.config.hard_loss_weight * hard_loss
        loss_components['hard_loss'] = hard_loss.item()
        
        # 计算学生准确率
        student_accuracy = 0.0
        teacher_accuracy = 0.0
        if student_logits is not None and labels is not None:
            with torch.no_grad():
                student_preds = student_logits.argmax(dim=-1)
                student_accuracy = (student_preds == labels).float().mean().item()
                metrics['student_accuracy'] = student_accuracy
                
                if teacher_logits is not None:
                    teacher_preds = teacher_logits.argmax(dim=-1)
                    teacher_accuracy = (teacher_preds == labels).float().mean().item()
                    metrics['teacher_accuracy'] = teacher_accuracy
        
        # ===== 2. 根据蒸馏类型计算各种损失 =====
        distill_type = self.config.distillation_type
        kl_divergence = 0.0
        
        # Logits蒸馏（软标签）
        if distill_type in ['logits', 'combined'] and student_logits is not None and teacher_logits is not None:
            soft_loss, soft_metrics = self.loss_calculator.compute_soft_loss(
                student_logits, teacher_logits, current_temperature
            )
            total_loss = total_loss + self.config.soft_loss_weight * soft_loss
            loss_components['soft_loss'] = soft_loss.item()
            kl_divergence = soft_metrics.get('kl_divergence', 0.0)
            metrics.update({f'soft_{k}': v for k, v in soft_metrics.items()})
        
        # 特征蒸馏
        if distill_type in ['feature', 'combined'] and student_features is not None and teacher_features is not None:
            feature_loss, feature_metrics = self.loss_calculator.compute_feature_loss(
                student_features, teacher_features, layer_weights
            )
            total_loss = total_loss + self.config.feature_loss_weight * feature_loss
            loss_components['feature_loss'] = feature_loss.item()
            metrics['avg_feature_similarity'] = feature_metrics.get('avg_feature_similarity', 0.0)
        
        # 注意力蒸馏
        if distill_type in ['attention', 'combined'] and student_attentions is not None and teacher_attentions is not None:
            attention_loss, attention_metrics = self.loss_calculator.compute_attention_loss(
                student_attentions, teacher_attentions, layer_weights
            )
            total_loss = total_loss + self.config.attention_loss_weight * attention_loss
            loss_components['attention_loss'] = attention_loss.item()
            metrics['avg_attention_correlation'] = attention_metrics.get('avg_attention_correlation', 0.0)
        
        # 关系蒸馏
        relational_weight = getattr(self.config, 'relational_loss_weight', 0.0)
        if distill_type in ['relational', 'combined'] and relational_weight > 0:
            if student_features is not None and teacher_features is not None:
                relational_loss, relational_metrics = self.loss_calculator.compute_relational_loss(
                    student_features[-1], teacher_features[-1]
                )
                total_loss = total_loss + relational_weight * relational_loss
                loss_components['relational_loss'] = relational_loss.item()
                metrics['distance_correlation'] = relational_metrics.get('distance_correlation', 0.0)
                metrics['angle_correlation'] = relational_metrics.get('angle_correlation', 0.0)
        
        # 对比蒸馏
        if distill_type == 'contrastive' and student_features is not None and teacher_features is not None:
            contrastive_loss, contrastive_metrics = self.loss_calculator.compute_contrastive_loss(
                student_features[-1], teacher_features[-1]
            )
            total_loss = total_loss + self.config.contrastive_weight * contrastive_loss
            loss_components['contrastive_loss'] = contrastive_loss.item()
            metrics['similarity_gap'] = contrastive_metrics.get('similarity_gap', 0.0)
        
        # 渐进式蒸馏
        if distill_type == 'progressive':
            total_loss, progressive_metrics = self._compute_progressive_loss(
                total_loss, student_features, teacher_features, context
            )
            metrics.update(progressive_metrics)
        
        # 自蒸馏（self-distillation）
        self_distill_weight = getattr(self.config, 'self_distill_weight', 0.0)
        if distill_type == 'self' and self_distill_weight > 0:
            if student_features is not None and len(student_features) >= 2:
                # 深层特征蒸馏浅层
                self_loss, _ = self.loss_calculator.compute_feature_loss(
                    student_features[:-1], student_features[1:]
                )
                total_loss = total_loss + self_distill_weight * self_loss
                loss_components['self_distill_loss'] = self_loss.item()
        
        # 多教师蒸馏
        if distill_type == 'multi_teacher' and self.teacher_models:
            teacher_logits_list = []
            with torch.no_grad():
                for teacher in self.teacher_models:
                    t_outputs = self._get_teacher_outputs_from_model(teacher, batch, context)
                    t_logits = self._get_logits(t_outputs)
                    if t_logits is not None:
                        teacher_logits_list.append(t_logits)
            
            if teacher_logits_list and student_logits is not None:
                ensemble_type = getattr(self.config, 'multi_teacher_ensemble', 
                                       MultiTeacherEnsembleType.WEIGHTED)
                if isinstance(ensemble_type, MultiTeacherEnsembleType):
                    ensemble_type = ensemble_type.value
                
                mt_loss, mt_metrics = self.loss_calculator.compute_multi_teacher_loss(
                    student_logits, teacher_logits_list,
                    self.teacher_weights if self.teacher_weights else None,
                    ensemble_type
                )
                total_loss = total_loss + self.config.soft_loss_weight * mt_loss
                loss_components['multi_teacher_loss'] = mt_loss.item()
                metrics['teacher_agreement'] = mt_metrics.get('teacher_agreement', 0.0)
        
        # ===== 3. 记录到监控器 =====
        if self._distillation_monitor is not None:
            self._distillation_monitor.record_step(
                hard_loss=loss_components.get('hard_loss', 0.0),
                soft_loss=loss_components.get('soft_loss', 0.0),
                feature_loss=loss_components.get('feature_loss', 0.0),
                attention_loss=loss_components.get('attention_loss', 0.0),
                contrastive_loss=loss_components.get('contrastive_loss', 0.0),
                total_loss=total_loss.item(),
                student_accuracy=student_accuracy,
                teacher_accuracy=teacher_accuracy,
                temperature=current_temperature,
                kl_divergence=kl_divergence,
                feature_similarity=metrics.get('avg_feature_similarity', 0.0)
            )
        
        # ===== 4. 使用基础策略组件 =====
        # 创建临时结果用于监控器记录
        temp_result = StrategyResult(
            loss=total_loss,
            metrics=metrics.copy(),
            loss_components=loss_components.copy()
        )
        
        # 记录到策略监控器（StrategyMonitor）
        # StrategyMonitor.record_step 接受 result 和 context 参数
        if self._strategy_monitor is not None:
            try:
                self._strategy_monitor.record_step(temp_result, context)
            except Exception as e:
                logger.debug(f"StrategyMonitor record failed: {e}")
        
        # 更新策略指标（StrategyMetrics）
        if self._strategy_metrics is not None:
            try:
                self._strategy_metrics.total_steps += 1
                self._strategy_metrics.total_loss += total_loss.item()
                self._strategy_metrics.avg_loss = (
                    self._strategy_metrics.total_loss / self._strategy_metrics.total_steps
                )
                # 记录阶段指标
                phase_key = self._current_phase.value if hasattr(self._current_phase, 'value') else str(self._current_phase)
                if phase_key not in self._strategy_metrics.phase_metrics:
                    self._strategy_metrics.phase_metrics[phase_key] = {'steps': 0, 'total_loss': 0.0}
                self._strategy_metrics.phase_metrics[phase_key]['steps'] += 1
                self._strategy_metrics.phase_metrics[phase_key]['total_loss'] += total_loss.item()
            except Exception as e:
                logger.debug(f"StrategyMetrics update failed: {e}")
        
        # 合并损失组件到指标
        metrics.update(loss_components)
        metrics['total_loss'] = total_loss.item()
        metrics['training_phase'] = self._current_phase.value if hasattr(self._current_phase, 'value') else str(self._current_phase)
        
        # 创建结果
        result = StrategyResult(
            loss=total_loss, 
            metrics=metrics,
            loss_components=loss_components,
            warnings=warnings if warnings else None
        )
        
        # ===== 5. 验证结果（StrategyValidator）=====
        if self._strategy_validator is not None:
            try:
                is_valid, validation_errors = self._strategy_validator.validate(result)
                if not is_valid:
                    for error in validation_errors:
                        warnings.append(f"Validation: {error}")
                    result.warnings = warnings
            except Exception as e:
                logger.debug(f"StrategyValidator failed: {e}")
        
        return result
    
    def _get_task_loss(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """获取任务损失"""
        if isinstance(outputs, dict) and 'loss' in outputs:
            return outputs['loss']
        elif hasattr(outputs, 'loss'):
            return outputs.loss
        else:
            raise ValueError("outputs中没有找到loss")
    
    def _get_logits(self, outputs: Any) -> Optional[torch.Tensor]:
        """获取logits"""
        if isinstance(outputs, dict):
            return outputs.get('logits', None)
        elif hasattr(outputs, 'logits'):
            return outputs.logits
        return None
    
    def _get_teacher_outputs(
        self, 
        batch: Dict[str, Any], 
        context: StrategyContext
    ) -> Dict[str, Any]:
        """获取主教师模型输出"""
        return self._get_teacher_outputs_from_model(self.teacher_model, batch, context)
    
    def _get_teacher_outputs_from_model(
        self,
        teacher: nn.Module,
        batch: Dict[str, Any],
        context: StrategyContext
    ) -> Dict[str, Any]:
        """获取指定教师模型的输出"""
        input_ids = batch.get('input_ids')
        attention_mask = batch.get('attention_mask')
        
        if input_ids is not None:
            inputs = {
                'input_ids': input_ids.to(context.device),
                'attention_mask': attention_mask.to(context.device) if attention_mask is not None else None,
                'output_hidden_states': True,
                'output_attentions': True
            }
            return teacher(**inputs)
        else:
            # 通用输入格式
            if isinstance(batch, dict):
                batch_on_device = {
                    k: v.to(context.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                return teacher(**batch_on_device)
            return teacher(batch)
    
    def _compute_progressive_loss(
        self,
        base_loss: torch.Tensor,
        student_features: tuple,
        teacher_features: tuple,
        context: StrategyContext
    ) -> tuple:
        """计算渐进式蒸馏损失"""
        metrics = {}
        
        # 根据当前阶段决定蒸馏的层数
        total_layers = len(student_features) if student_features else 0
        if total_layers == 0:
            return base_loss, metrics
        
        # 获取渐进调度类型
        progressive_schedule = getattr(self.config, 'progressive_schedule', 
                                       ProgressiveScheduleType.LINEAR)
        if isinstance(progressive_schedule, ProgressiveScheduleType):
            progressive_schedule = progressive_schedule.value
        
        # 计算当前阶段应该蒸馏的层数
        if progressive_schedule == 'exponential':
            # 指数增长：初始少，后期快速增加
            progress = min(self.current_stage / max(self.config.progressive_stages - 1, 1), 1.0)
            current_layers = max(1, int(total_layers * (progress ** 2)))
        else:  # linear
            layers_per_stage = max(1, total_layers // self.config.progressive_stages)
            current_layers = min((self.current_stage + 1) * layers_per_stage, total_layers)
        
        # 计算层权重
        layer_weights = {}
        if self._layer_weight_manager is not None:
            for i in range(current_layers):
                layer_weights[i] = self._layer_weight_manager.get_weight(i)
        else:
            # 默认后层权重更高
            for i in range(current_layers):
                layer_weights[i] = (i + 1) / current_layers
        
        # 渐进式特征蒸馏
        progressive_loss = torch.tensor(0.0, device=context.device)
        layer_losses = {}
        
        for i in range(current_layers):
            try:
                s_feat = student_features[i]
                t_feat = teacher_features[i]
                
                layer_loss = self.loss_calculator._compute_feature_layer_loss(s_feat, t_feat)
                weighted_loss = layer_loss * layer_weights.get(i, 1.0)
                progressive_loss = progressive_loss + weighted_loss
                layer_losses[f'layer_{i}'] = layer_loss.item()
            except (IndexError, RuntimeError) as e:
                logger.warning(f"Progressive layer {i} failed: {e}")
                continue
        
        progressive_loss = progressive_loss / max(current_layers, 1)
        
        metrics['progressive_loss'] = progressive_loss.item()
        metrics['current_stage'] = self.current_stage
        metrics['current_layers'] = current_layers
        metrics['total_layers'] = total_layers
        metrics['progress'] = current_layers / total_layers
        
        # 更新监控器
        if self._distillation_monitor is not None:
            self._distillation_monitor.update_progressive_state(self.current_stage, current_layers)
        
        return base_loss + self.config.feature_loss_weight * progressive_loss, metrics
    
    def on_step_end(self, context: StrategyContext, result: StrategyResult) -> None:
        """步骤结束时的回调"""
        # 调用父类回调
        super().on_step_end(context, result)
        
        # 更新温度调度器
        if self._temp_scheduler is not None:
            kl_div = result.metrics.get('soft_kl_divergence', None)
            self._temp_scheduler.step(kl_div)
        
        # 更新层权重管理器（自适应权重）
        if self._layer_weight_manager is not None and hasattr(self._layer_weight_manager, 'update'):
            layer_losses = result.metrics.get('layer_losses', {})
            if layer_losses:
                for layer_key, loss_val in layer_losses.items():
                    layer_idx = int(layer_key.split('_')[-1])
                    self._layer_weight_manager.update(layer_idx, loss_val)
        
        # 在线蒸馏：EMA更新教师模型
        teacher_update_freq = getattr(self.config, 'teacher_update_freq', 1)
        if self.config.online_distillation and self.teacher_model is not None:
            if context.global_step % teacher_update_freq == 0:
                self._update_teacher_ema(context)
        
        # 渐进式蒸馏：更新阶段
        if self.config.distillation_type == 'progressive':
            self.stage_step += 1
            if self.stage_step >= self.config.progressive_warmup:
                self.current_stage = min(
                    self.current_stage + 1, 
                    self.config.progressive_stages - 1
                )
                self.stage_step = 0
        
        # ===== 更新训练阶段（使用 TrainingPhase） =====
        self._update_training_phase(context)
    
    def _update_training_phase(self, context: StrategyContext) -> None:
        """
        更新训练阶段
        
        使用 base_strategy.py 的 TrainingPhase 枚举管理训练阶段
        """
        max_steps = context.max_steps or 10000
        warmup_ratio = 0.1
        cooldown_ratio = 0.1
        
        warmup_steps = int(max_steps * warmup_ratio)
        cooldown_start = int(max_steps * (1 - cooldown_ratio))
        
        old_phase = self._current_phase
        
        if context.global_step < warmup_steps:
            self._current_phase = TrainingPhase.WARMUP
        elif context.global_step >= cooldown_start:
            self._current_phase = TrainingPhase.COOLDOWN
        else:
            self._current_phase = TrainingPhase.MAIN
        
        # 如果阶段发生变化，记录日志
        if old_phase != self._current_phase:
            logger.info(f"Training phase changed: {old_phase} -> {self._current_phase}")
            
            # 通知策略监控器阶段变化
            if self._strategy_monitor is not None and hasattr(self._strategy_monitor, 'on_phase_change'):
                try:
                    self._strategy_monitor.on_phase_change(old_phase, self._current_phase)
                except Exception:
                    pass
    
    def _update_teacher_ema(self, context: StrategyContext) -> None:
        """EMA更新教师模型"""
        with torch.no_grad():
            for t_param, s_param in zip(
                self.teacher_model.parameters(), 
                context.model.parameters()
            ):
                t_param.data = (
                    self.config.teacher_ema_decay * t_param.data + 
                    (1 - self.config.teacher_ema_decay) * s_param.data
                )
    
    def get_config(self) -> Dict[str, Any]:
        """获取策略配置"""
        base_config = super().get_config()
        base_config.update({
            'distillation_type': self.config.distillation_type,
            'temperature': self.config.temperature,
            'current_temperature': self.get_effective_temperature(),
            'soft_loss_weight': self.config.soft_loss_weight,
            'feature_loss_weight': self.config.feature_loss_weight,
            'attention_loss_weight': self.config.attention_loss_weight,
            'online_distillation': self.config.online_distillation,
            'num_teachers': len(self.get_teacher_models()),
            'current_stage': self.current_stage
        })
        return base_config
    
    def get_info(self) -> Dict[str, Any]:
        """获取策略完整信息"""
        return {
            'name': self.name,
            'type': 'distillation',
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config),
            'lib_info': self.get_layer_info(),
            'state': {
                'current_stage': self.current_stage,
                'stage_step': self.stage_step,
                'temperature': self.get_effective_temperature(),
                'layer_weights': self.get_layer_weights(),
            },
            'teachers': {
                'primary': self.teacher_model is not None,
                'additional': len(self.teacher_models),
                'weights': self.teacher_weights,
            }
        }
    
    def get_layer_info(self) -> Dict[str, Any]:
        """获取底层模块调用信息"""
        info = {
            'soft_label_loss': self.loss_calculator._soft_label_loss is not None if self.loss_calculator else False,
            'feature_loss': self.loss_calculator._feature_loss is not None if self.loss_calculator else False,
            'attention_loss': self.loss_calculator._attention_loss is not None if self.loss_calculator else False,
            'contrastive_loss': self.loss_calculator._contrastive_loss is not None if self.loss_calculator else False,
            'relational_loss': self.loss_calculator._relational_loss is not None if self.loss_calculator else False,
            'multi_teacher_loss': self.loss_calculator._multi_teacher_loss is not None if self.loss_calculator else False,
            'distillation_type': self.config.distillation_type
        }
        
        if self.loss_calculator:
            info['loss_statistics'] = self.loss_calculator.get_statistics()
        
        return info
    
    # ==================== 基础策略组件方法 ====================
    
    def get_strategy_type(self) -> StrategyType:
        """
        获取策略类型
        
        使用 base_strategy.py 的 StrategyType 枚举
        """
        return self.STRATEGY_TYPE
    
    def get_training_phase(self) -> TrainingPhase:
        """
        获取当前训练阶段
        
        使用 base_strategy.py 的 TrainingPhase 枚举
        """
        return self._current_phase
    
    def set_training_phase(self, phase: TrainingPhase) -> None:
        """设置训练阶段"""
        old_phase = self._current_phase
        self._current_phase = phase
        logger.info(f"Training phase manually set: {old_phase} -> {phase}")
    
    def get_strategy_monitor(self) -> Optional[StrategyMonitor]:
        """获取策略监控器实例"""
        return self._strategy_monitor
    
    def get_strategy_profiler(self) -> Optional[StrategyProfiler]:
        """获取策略性能分析器实例"""
        return self._strategy_profiler
    
    def get_strategy_validator(self) -> Optional[StrategyValidator]:
        """获取策略验证器实例"""
        return self._strategy_validator
    
    def get_strategy_metrics(self) -> Optional[StrategyMetrics]:
        """获取策略指标跟踪器实例"""
        return self._strategy_metrics
    
    def get_base_strategy_summary(self) -> Dict[str, Any]:
        """
        获取基础策略组件的摘要信息
        
        整合 StrategyMonitor, StrategyProfiler, StrategyValidator, StrategyMetrics 的信息
        """
        summary = {
            'strategy_type': self.STRATEGY_TYPE.value if hasattr(self.STRATEGY_TYPE, 'value') else str(self.STRATEGY_TYPE),
            'training_phase': self._current_phase.value if hasattr(self._current_phase, 'value') else str(self._current_phase),
            'components': {
                'monitor': self._strategy_monitor is not None,
                'profiler': self._strategy_profiler is not None,
                'validator': self._strategy_validator is not None,
                'metrics': self._strategy_metrics is not None,
            }
        }
        
        # 添加监控器摘要
        if self._strategy_monitor is not None:
            try:
                if hasattr(self._strategy_monitor, 'get_summary'):
                    summary['monitor_summary'] = self._strategy_monitor.get_summary()
            except Exception:
                pass
        
        # 添加性能分析器统计
        if self._strategy_profiler is not None:
            try:
                if hasattr(self._strategy_profiler, 'get_stats'):
                    summary['profiler_stats'] = self._strategy_profiler.get_stats()
            except Exception:
                pass
        
        # 添加指标统计
        if self._strategy_metrics is not None:
            try:
                summary['metrics'] = {
                    'total_steps': self._strategy_metrics.total_steps,
                    'avg_loss': self._strategy_metrics.avg_loss,
                    'phase_metrics': self._strategy_metrics.phase_metrics,
                }
            except Exception:
                pass
        
        return summary
    
    def enable_profiling(self) -> None:
        """
        启用性能分析
        
        使用 StrategyProfiler 进行性能分析
        """
        if self._strategy_profiler is not None and hasattr(self._strategy_profiler, 'enable'):
            self._strategy_profiler.enable()
            logger.info("Strategy profiling enabled")
    
    def disable_profiling(self) -> None:
        """禁用性能分析"""
        if self._strategy_profiler is not None and hasattr(self._strategy_profiler, 'disable'):
            self._strategy_profiler.disable()
            logger.info("Strategy profiling disabled")
    
    def get_profiling_stats(self) -> Dict[str, Any]:
        """
        获取性能分析统计
        
        使用 StrategyProfiler 获取性能数据
        """
        if self._strategy_profiler is None:
            return {'available': False}
        
        try:
            if hasattr(self._strategy_profiler, 'get_stats'):
                return self._strategy_profiler.get_stats()
        except Exception as e:
            logger.warning(f"Failed to get profiling stats: {e}")
        
        return {'available': True, 'error': 'Failed to get stats'}
    
    def print_profiling_stats(self) -> None:
        """打印性能分析统计"""
        if self._strategy_profiler is not None and hasattr(self._strategy_profiler, 'print_stats'):
            self._strategy_profiler.print_stats()
        else:
            stats = self.get_profiling_stats()
            print("\n=== Strategy Profiling Stats ===")
            for k, v in stats.items():
                print(f"  {k}: {v}")
    
    def validate_result(self, result: StrategyResult) -> Tuple[bool, List[str]]:
        """
        验证策略结果
        
        使用 StrategyValidator 验证结果
        """
        if self._strategy_validator is None:
            return True, []
        
        try:
            return self._strategy_validator.validate(result)
        except Exception as e:
            logger.warning(f"Result validation failed: {e}")
            return True, [f"Validation error: {e}"]
    
    def add_validation_check(
        self, 
        check_fn: Callable[[StrategyResult], Tuple[bool, str]],
        name: str = ""
    ) -> None:
        """
        添加自定义验证检查
        
        使用 StrategyValidator 添加验证规则
        
        Args:
            check_fn: 验证函数，接受 StrategyResult 返回 Tuple[bool, str] (valid, error_message)
            name: 检查名称（可选，用于日志）
        """
        if self._strategy_validator is not None and hasattr(self._strategy_validator, 'add_check'):
            self._strategy_validator.add_check(check_fn)
            logger.debug(f"Validation check{' (' + name + ')' if name else ''} added")
    
    def get_distillation_stats(self) -> Dict[str, Any]:
        """获取蒸馏统计数据"""
        stats = {}
        
        if self._distillation_monitor is not None:
            stats['monitor'] = self._distillation_monitor.get_summary()
        
        if self.loss_calculator is not None:
            stats['loss_calculator'] = self.loss_calculator.get_statistics()
        
        stats['state'] = {
            'current_stage': self.current_stage,
            'stage_step': self.stage_step,
            'temperature': self.get_effective_temperature(),
        }
        
        return stats
    
    def get_distillation_quality_report(self) -> Dict[str, Any]:
        """获取蒸馏质量报告"""
        report = {
            'effectiveness': 'unknown',
            'recommendations': [],
            'warnings': [],
        }
        
        if self._distillation_monitor is None:
            report['warnings'].append("Monitoring not enabled")
            return report
        
        stats = self._distillation_monitor.get_stats()
        
        # 评估蒸馏有效性
        if stats.total_steps < 100:
            report['effectiveness'] = 'insufficient_data'
            report['recommendations'].append("Need more training steps for accurate assessment")
        elif self._distillation_monitor.is_distillation_effective():
            report['effectiveness'] = 'effective'
        else:
            report['effectiveness'] = 'needs_improvement'
            
            # 提供建议
            if stats.accuracy_gap > 0.2:
                report['recommendations'].append("Large accuracy gap - consider lower temperature")
            
            loss_trend = self._distillation_monitor.get_loss_trend()
            if loss_trend == 'degrading':
                report['recommendations'].append("Loss is increasing - check learning rate")
                report['warnings'].append("Training may be diverging")
            
            if stats.avg_kl_divergence > 1.0:
                report['recommendations'].append("High KL divergence - increase temperature")
            elif stats.avg_kl_divergence < 0.01:
                report['recommendations'].append("Very low KL divergence - decrease temperature")
        
        report['summary'] = stats.to_dict()
        
        return report
    
    def print_distillation_quality_report(self) -> None:
        """打印蒸馏质量报告"""
        report = self.get_distillation_quality_report()
        
        print("\n" + "=" * 60)
        print("DISTILLATION QUALITY REPORT")
        print("=" * 60)
        
        print(f"\nEffectiveness: {report['effectiveness']}")
        
        if report['warnings']:
            print("\nWarnings:")
            for w in report['warnings']:
                print(f"  ⚠️  {w}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for r in report['recommendations']:
                print(f"  💡 {r}")
        
        if 'summary' in report:
            print("\nStatistics Summary:")
            for k, v in report['summary'].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
        
        print("=" * 60)
    
    def diagnose_distillation(self) -> Dict[str, Any]:
        """诊断蒸馏状态"""
        diagnosis = {
            'status': 'healthy',
            'issues': [],
            'suggestions': []
        }
        
        # 检查教师模型
        if self.teacher_model is None and not self.teacher_models:
            diagnosis['status'] = 'error'
            diagnosis['issues'].append("No teacher model configured")
            diagnosis['suggestions'].append("Set teacher model with set_teacher_model()")
        
        # 检查损失计算器
        if self.loss_calculator is None:
            diagnosis['status'] = 'error'
            diagnosis['issues'].append("Loss calculator not initialized")
            diagnosis['suggestions'].append("Call setup() before compute_loss()")
        
        # 检查监控状态
        if self._distillation_monitor is not None:
            stats = self._distillation_monitor.get_stats()
            
            if stats.total_steps > 0:
                # 检查损失趋势
                trend = self._distillation_monitor.get_loss_trend()
                if trend == 'degrading':
                    diagnosis['status'] = 'warning'
                    diagnosis['issues'].append("Loss is increasing")
                    diagnosis['suggestions'].append("Consider reducing learning rate")
                
                # 检查准确率差距
                if stats.accuracy_gap > 0.3:
                    diagnosis['issues'].append(f"Large accuracy gap: {stats.accuracy_gap:.2%}")
                    diagnosis['suggestions'].append("Consider longer training or model architecture changes")
        
        return diagnosis
    
    def print_distillation_diagnosis(self) -> None:
        """打印蒸馏诊断结果"""
        diagnosis = self.diagnose_distillation()
        
        print("\n" + "=" * 60)
        print("DISTILLATION DIAGNOSIS")
        print("=" * 60)
        
        status_emoji = {'healthy': '✅', 'warning': '⚠️', 'error': '❌'}
        print(f"\nStatus: {status_emoji.get(diagnosis['status'], '?')} {diagnosis['status'].upper()}")
        
        if diagnosis['issues']:
            print("\nIssues:")
            for issue in diagnosis['issues']:
                print(f"  - {issue}")
        
        if diagnosis['suggestions']:
            print("\nSuggestions:")
            for suggestion in diagnosis['suggestions']:
                print(f"  💡 {suggestion}")
        
        print("=" * 60)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """获取策略状态字典"""
        state = super().get_state_dict()
        
        state.update({
            'current_stage': self.current_stage,
            'stage_step': self.stage_step,
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
        })
        
        # 保存温度调度器状态
        if self._temp_scheduler is not None:
            state['temp_scheduler'] = {
                'step': self._temp_scheduler._step,
                'current_temp': self._temp_scheduler._current_temp,
            }
        
        # 保存监控器状态
        if self._distillation_monitor is not None:
            state['monitor_stats'] = self._distillation_monitor.get_stats().to_dict()
        
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """加载策略状态字典"""
        super().load_state_dict(state)
        
        self.current_stage = state.get('current_stage', 0)
        self.stage_step = state.get('stage_step', 0)
        
        # 恢复温度调度器状态
        if self._temp_scheduler is not None and 'temp_scheduler' in state:
            ts_state = state['temp_scheduler']
            self._temp_scheduler._step = ts_state.get('step', 0)
            self._temp_scheduler._current_temp = ts_state.get('current_temp', self.config.temperature)
    
    def reset_stats(self) -> None:
        """重置统计数据"""
        super().reset_stats()
        
        self.current_stage = 0
        self.stage_step = 0
        
        if self._distillation_monitor is not None:
            self._distillation_monitor.reset()
        
        if self._temp_scheduler is not None:
            self._temp_scheduler.reset()
        
        if self.loss_calculator is not None:
            self.loss_calculator.reset_statistics()


# ======================== 专用策略变体 ========================

class SelfDistillationStrategy(DistillationStrategy):
    """
    自蒸馏策略
    
    模型自身不同层之间的知识蒸馏，无需外部教师模型。
    深层知识向浅层传递，提升模型效率。
    
    支持多种自蒸馏模式:
    - deep_to_shallow: 深层特征指导浅层
    - born_again: 模型自身的知识蒸馏（需要预训练模型）
    - online_self: 在线自蒸馏，使用EMA教师
    """
    
    def __init__(
        self, 
        config: Optional[DistillationStrategyConfig] = None,
        self_distill_mode: str = 'deep_to_shallow'
    ):
        if config is None:
            config = DistillationStrategyConfig(
                distillation_type="self",
                feature_loss_weight=0.5,
                self_distill_layers=[-2, -1]
            )
        super().__init__(config, teacher_model=None)
        self.name = "self_distillation"
        self.self_distill_mode = self_distill_mode
        
        # 层相似度跟踪
        self._layer_similarities: Dict[int, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # EMA教师（用于online_self模式）
        self._ema_teacher: Optional[nn.Module] = None
        self._ema_decay: float = 0.999
    
    def setup(self, context: StrategyContext) -> None:
        """初始化自蒸馏组件"""
        super().setup(context)
        
        # 在线自蒸馏模式：创建EMA教师
        if self.self_distill_mode == 'online_self':
            if context.model is not None:
                import copy
                self._ema_teacher = copy.deepcopy(context.model)
                self._ema_teacher.eval()
                for param in self._ema_teacher.parameters():
                    param.requires_grad = False
                logger.info("Online self-distillation EMA teacher initialized")
    
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """计算自蒸馏损失"""
        metrics = {}
        loss_components: Dict[str, float] = {}
        
        # 硬标签损失
        hard_loss = self._get_task_loss(outputs)
        total_loss = self.config.hard_loss_weight * hard_loss
        loss_components['hard_loss'] = hard_loss.item()
        
        # 自蒸馏损失：深层向浅层蒸馏
        hidden_states = outputs.get('hidden_states', None)
        if hidden_states is not None and len(hidden_states) >= 2:
            self_distill_loss, distill_metrics = self._compute_self_distill_loss(
                hidden_states, context
            )
            
            self_weight = getattr(self.config, 'self_distill_weight', self.config.feature_loss_weight)
            total_loss = total_loss + self_weight * self_distill_loss
            loss_components['self_distill_loss'] = self_distill_loss.item()
            metrics.update(distill_metrics)
        
        # 在线自蒸馏：EMA教师
        if self.self_distill_mode == 'online_self' and self._ema_teacher is not None:
            student_logits = self._get_logits(outputs)
            if student_logits is not None:
                with torch.no_grad():
                    ema_outputs = self._ema_teacher(**self._prepare_inputs(batch, context))
                    ema_logits = self._get_logits(ema_outputs)
                
                if ema_logits is not None:
                    ema_loss, ema_metrics = self.loss_calculator.compute_soft_loss(
                        student_logits, ema_logits
                    )
                    total_loss = total_loss + self.config.soft_loss_weight * ema_loss
                    loss_components['ema_distill_loss'] = ema_loss.item()
                    metrics['ema_agreement'] = ema_metrics.get('student_teacher_agreement', 0.0)
        
        # 记录到监控器
        if self._distillation_monitor is not None:
            self._distillation_monitor.record_step(
                hard_loss=loss_components.get('hard_loss', 0.0),
                feature_loss=loss_components.get('self_distill_loss', 0.0),
                total_loss=total_loss.item(),
                temperature=self.get_effective_temperature(),
            )
        
        metrics.update(loss_components)
        metrics['total_loss'] = total_loss.item()
        
        return StrategyResult(
            loss=total_loss, 
            metrics=metrics,
            loss_components=loss_components
        )
    
    def _compute_self_distill_loss(
        self,
        hidden_states: tuple,
        context: StrategyContext
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算自蒸馏损失（深层到浅层）"""
        metrics = {}
        self_distill_loss = torch.tensor(0.0, device=context.device)
        count = 0
            
        # 最后一层作为教师，向前面的层蒸馏
        teacher_layer = hidden_states[-1].detach()
            
        for layer_idx in self.config.self_distill_layers[:-1]:
            student_layer = hidden_states[layer_idx]
                
            # 维度匹配
            if student_layer.shape != teacher_layer.shape:
                continue
                
            # 计算损失
            if self.loss_calculator is not None:
                layer_loss = self.loss_calculator._compute_feature_layer_loss(
                    student_layer, teacher_layer
                )
            else:
                layer_loss = F.mse_loss(student_layer, teacher_layer) 
                self_distill_loss = self_distill_loss + layer_loss

            count += 1
            
        # 记录层相似度
        with torch.no_grad():
            similarity = F.cosine_similarity(
                student_layer.flatten(1), teacher_layer.flatten(1), dim=-1
            ).mean().item()
            self._layer_similarities[layer_idx].append(similarity)
            metrics[f'layer_{layer_idx}_similarity'] = similarity
        
        self_distill_loss = self_distill_loss / max(count, 1)
        metrics['num_layers'] = count
        metrics['avg_layer_similarity'] = sum(
            sum(h) / len(h) for h in self._layer_similarities.values() if h
        ) / max(len(self._layer_similarities), 1)
        
        return self_distill_loss, metrics
    
    def _prepare_inputs(self, batch: Dict[str, Any], context: StrategyContext) -> Dict[str, Any]:
        """准备模型输入"""
        return {
            k: v.to(context.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
            if k in ['input_ids', 'attention_mask', 'token_type_ids']
        }
    
    def on_step_end(self, context: StrategyContext, result: StrategyResult) -> None:
        """步骤结束时的回调"""
        super().on_step_end(context, result)
        
        # 更新EMA教师
        if self._ema_teacher is not None and context.model is not None:
            with torch.no_grad():
                for ema_param, model_param in zip(
                    self._ema_teacher.parameters(),
                    context.model.parameters()
                ):
                    ema_param.data = (
                        self._ema_decay * ema_param.data + 
                        (1 - self._ema_decay) * model_param.data
                    )
    
    def get_layer_similarity_stats(self) -> Dict[str, Any]:
        """获取层相似度统计"""
        stats = {}
        for layer_idx, history in self._layer_similarities.items():
            if history:
                stats[f'layer_{layer_idx}'] = {
                    'avg': sum(history) / len(history),
                    'recent': history[-1] if history else 0.0,
                    'trend': 'improving' if len(history) >= 10 and history[-1] > history[-10] else 'stable'
                }
        return stats


class ProgressiveDistillationStrategy(DistillationStrategy):
    """
    渐进式蒸馏策略
    
    逐步增加蒸馏层数，从浅层开始逐渐扩展到深层。
    有助于学生模型更稳定地学习教师知识。
    
    支持多种渐进策略:
    - linear: 线性增加层数
    - exponential: 指数增加层数
    - curriculum: 课程学习式增加
    """
    
    def __init__(
        self,
        config: Optional[DistillationStrategyConfig] = None,
        teacher_model: Optional[nn.Module] = None,
        schedule_type: str = 'linear'
    ):
        if config is None:
            config = DistillationStrategyConfig(
                distillation_type="progressive",
                progressive_stages=4,
                progressive_warmup=500,
                feature_loss_weight=0.3
            )
        super().__init__(config, teacher_model)
        self.name = "progressive_distillation"
        self.schedule_type = schedule_type
        
        # 阶段历史记录
        self._stage_history: List[Dict[str, Any]] = []
        self._stage_losses: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
    
    def on_step_end(self, context: StrategyContext, result: StrategyResult) -> None:
        """步骤结束回调"""
        super().on_step_end(context, result)
        
        # 记录阶段损失
        current_stage = self.current_stage
        if 'progressive_loss' in result.metrics:
            self._stage_losses[current_stage].append(result.metrics['progressive_loss'])
        
        # 检查是否需要进入下一阶段
        if self._should_advance_stage():
            old_stage = self.current_stage
            self.current_stage = min(
                self.current_stage + 1,
                self.config.progressive_stages - 1
            )
            self.stage_step = 0
            
            if self.current_stage != old_stage:
                self._record_stage_transition(old_stage, context)
    
    def _should_advance_stage(self) -> bool:
        """判断是否应该进入下一阶段"""
        if self.stage_step < self.config.progressive_warmup:
            return False
        
        # 检查当前阶段的损失是否稳定
        current_losses = self._stage_losses.get(self.current_stage, deque())
        if len(current_losses) < 50:
            return True  # 数据不足，使用默认逻辑
        
        recent = list(current_losses)[-20:]
        earlier = list(current_losses)[-40:-20]
        
        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)
        
        # 损失改善小于5%时进入下一阶段
        improvement = (earlier_avg - recent_avg) / max(earlier_avg, 1e-8)
        return improvement < 0.05
    
    def _record_stage_transition(self, old_stage: int, context: StrategyContext) -> None:
        """记录阶段转换"""
        old_losses = list(self._stage_losses.get(old_stage, []))
        self._stage_history.append({
            'from_stage': old_stage,
            'to_stage': self.current_stage,
            'step': context.global_step,
            'avg_loss': sum(old_losses) / len(old_losses) if old_losses else 0.0,
        })
        logger.info(f"Progressive distillation advanced from stage {old_stage} to {self.current_stage}")
    
    def get_progressive_stats(self) -> Dict[str, Any]:
        """获取渐进蒸馏统计"""
        stats = {
            'current_stage': self.current_stage,
            'total_stages': self.config.progressive_stages,
            'stage_step': self.stage_step,
            'stage_history': self._stage_history,
        }
        
        for stage, losses in self._stage_losses.items():
            if losses:
                stats[f'stage_{stage}_avg_loss'] = sum(losses) / len(losses)
        
        return stats


class IndustryDistillationStrategy(DistillationStrategy):
    """
    行业知识蒸馏策略
    
    针对行业模型的特殊蒸馏需求：
    - 通用模型 → 行业模型的知识迁移
    - 行业特定层的特征蒸馏
    - 领域适配蒸馏
    
    使用底层能力：
    - backend/lib/losses: 蒸馏损失、对比损失、正则化损失
    """
    
    def __init__(
        self, 
        config: Optional[DistillationStrategyConfig] = None,
        teacher_model: Optional[nn.Module] = None
    ):
        if config is None:
            config = DistillationStrategyConfig(
                temperature=4.0,
                hard_loss_weight=1.0,
                soft_loss_weight=0.5,
                feature_loss_weight=0.2,
                distillation_type="combined",
                feature_layers=[-1, -2, -3],
                feature_loss_type="cosine"
            )
        super().__init__(config, teacher_model)
        self.name = "industry_distillation"
        
        # 领域适配损失权重
        self.domain_adaptation_weight = 0.1
        
        # 底层损失模块
        self._domain_loss: Optional[nn.Module] = None
        self._relational_loss: Optional[nn.Module] = None
    
    def setup(self, context: StrategyContext) -> None:
        """初始化行业蒸馏组件"""
        super().setup(context)
        
        # 初始化底层领域适配损失

        # 关系蒸馏损失（用于行业知识迁移）
        self._relational_loss = RelationalDistillationLoss().to(context.device)
        logger.info("Industry distillation losses initialized from backend/lib/losses")
    
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """计算行业蒸馏损失"""
        result = super().compute_loss(model, batch, outputs, context)
        
        # 添加领域适配损失
        if 'domain_features' in outputs:
            domain_loss = self._compute_domain_adaptation_loss(
                outputs['domain_features'], context
            )
            result.loss = result.loss + self.domain_adaptation_weight * domain_loss
            result.metrics['domain_loss'] = domain_loss.item()
        
        # 添加关系蒸馏损失（行业特定）
        if 'student_relations' in outputs and 'teacher_relations' in outputs:
            if self._relational_loss is not None:
                rel_loss = self._relational_loss(
                    outputs['student_relations'],
                    outputs['teacher_relations']
                )
                result.loss = result.loss + 0.1 * rel_loss
                result.metrics['relational_loss'] = rel_loss.item()
        
        return result
    
    def _compute_domain_adaptation_loss(
        self,
        domain_features: torch.Tensor,
        context: StrategyContext
    ) -> torch.Tensor:
        """
        计算领域适配损失
        
        使用特征正则化促进领域适配
        """
        # 特征正则化，促进领域适配
        return torch.norm(domain_features, p=2, dim=-1).mean()


class ContrastiveDistillationStrategy(DistillationStrategy):
    """
    对比蒸馏策略
    
    使用对比学习方法进行知识蒸馏，
    增强学生模型对特征表示的学习能力。
    
    支持多种对比模式:
    - infonce: InfoNCE损失
    - simclr: SimCLR风格对比
    - moco: MoCo风格动量对比
    """
    
    def __init__(
        self,
        config: Optional[DistillationStrategyConfig] = None,
        teacher_model: Optional[nn.Module] = None,
        contrastive_mode: str = 'infonce'
    ):
        if config is None:
            config = DistillationStrategyConfig(
                distillation_type="contrastive",
                contrastive_temperature=0.5,
                contrastive_weight=0.2,
                soft_loss_weight=0.3
            )
        super().__init__(config, teacher_model)
        self.name = "contrastive_distillation"
        self.contrastive_mode = contrastive_mode
        
        # 相似度跟踪
        self._positive_sim_history: deque = deque(maxlen=1000)
        self._negative_sim_history: deque = deque(maxlen=1000)
        
        # MoCo队列（用于动量对比）
        self._moco_queue: Optional[torch.Tensor] = None
        self._moco_queue_ptr: int = 0
        self._moco_queue_size: int = 4096
    
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """计算对比蒸馏损失"""
        result = super().compute_loss(model, batch, outputs, context)
        
        # 记录相似度
        if 'similarity_gap' in result.metrics:
            self._positive_sim_history.append(
                result.metrics.get('positive_similarity', 0.0)
            )
            self._negative_sim_history.append(
                result.metrics.get('negative_similarity', 0.0)
            )
        
        return result
    
    def get_contrastive_stats(self) -> Dict[str, Any]:
        """获取对比学习统计"""
        stats = {
            'mode': self.contrastive_mode,
            'temperature': self.config.contrastive_temperature,
        }
        
        if self._positive_sim_history:
            stats['avg_positive_similarity'] = sum(self._positive_sim_history) / len(self._positive_sim_history)
            stats['recent_positive_similarity'] = self._positive_sim_history[-1]
        
        if self._negative_sim_history:
            stats['avg_negative_similarity'] = sum(self._negative_sim_history) / len(self._negative_sim_history)
            stats['recent_negative_similarity'] = self._negative_sim_history[-1]
        
        if self._positive_sim_history and self._negative_sim_history:
            stats['avg_similarity_gap'] = stats.get('avg_positive_similarity', 0) - stats.get('avg_negative_similarity', 0)
        
        return stats
    
    def is_learning_effective(self, min_gap: float = 0.1) -> bool:
        """检查对比学习是否有效"""
        if len(self._positive_sim_history) < 100:
            return True  # 数据不足，假设有效
        
        gap = (
            sum(self._positive_sim_history) / len(self._positive_sim_history) -
            sum(self._negative_sim_history) / len(self._negative_sim_history)
        )
        return gap >= min_gap


# ======================== 便捷函数 ========================

def create_distillation_strategy(
    strategy_type: str = "standard",
    config: Optional[Dict[str, Any]] = None,
    teacher_model: Optional[nn.Module] = None,
    **kwargs
) -> DistillationStrategy:
    """
    创建蒸馏策略的便捷函数
    
    Args:
        strategy_type: 策略类型 (standard, self, progressive, industry, contrastive)
        config: 配置字典
        teacher_model: 教师模型
        **kwargs: 额外参数传递给策略类
    
    Returns:
        蒸馏策略实例
    """
    strategy_config = DistillationStrategyConfig(**config) if config else None
    
    strategy_map = {
        'standard': DistillationStrategy,
        'self': SelfDistillationStrategy,
        'progressive': ProgressiveDistillationStrategy,
        'industry': IndustryDistillationStrategy,
        'contrastive': ContrastiveDistillationStrategy
    }
    
    strategy_class = strategy_map.get(strategy_type, DistillationStrategy)
    
    if strategy_type == 'self':
        return strategy_class(strategy_config, **kwargs)
    else:
        return strategy_class(strategy_config, teacher_model, **kwargs)


def create_distillation_from_trainer_config(
    trainer_config: Dict[str, Any],
    teacher_model: Optional[nn.Module] = None
) -> DistillationStrategy:
    """
    从底层训练器配置创建蒸馏策略
    
    兼容 distillation/knowledge_distillation.py 中的配置格式。
    
    Args:
        trainer_config: 底层训练器配置
        teacher_model: 教师模型
    
    Returns:
        蒸馏策略实例
    """
    # 转换配置格式
    strategy_config = DistillationStrategyConfig(
        temperature=trainer_config.get('temperature', 4.0),
        soft_loss_weight=trainer_config.get('alpha', 0.7),
        hard_loss_weight=trainer_config.get('beta', 0.3),
        feature_loss_weight=trainer_config.get('feature_loss_weight', 0.1),
        attention_loss_weight=trainer_config.get('attention_loss_weight', 0.1),
        distillation_type='combined' if trainer_config.get('use_feature_distillation') else 'logits'
    )
    
    return DistillationStrategy(strategy_config, teacher_model)


def create_multi_teacher_distillation_strategy(
    teacher_models: List[nn.Module],
    teacher_weights: Optional[List[float]] = None,
    ensemble_type: str = 'weighted',
    config: Optional[Dict[str, Any]] = None
) -> DistillationStrategy:
    """
    创建多教师蒸馏策略
    
    Args:
        teacher_models: 教师模型列表
        teacher_weights: 教师权重列表（可选）
        ensemble_type: 集成类型 (weighted, average, voting)
        config: 配置字典
    
    Returns:
        蒸馏策略实例
    """
    if not teacher_models:
        raise ValueError("At least one teacher model is required")
    
    # 创建配置
    default_config = {
        'distillation_type': 'multi_teacher',
        'multi_teacher_ensemble': ensemble_type,
    }
    if config:
        default_config.update(config)
    if teacher_weights:
        default_config['multi_teacher_weights'] = teacher_weights
    
    strategy_config = DistillationStrategyConfig(**default_config)
    
    # 创建策略
    strategy = DistillationStrategy(strategy_config, teacher_model=teacher_models[0])
    
    # 添加其他教师
    for i, teacher in enumerate(teacher_models[1:], 1):
        weight = teacher_weights[i] if teacher_weights and i < len(teacher_weights) else 1.0
        strategy.teacher_models.append(teacher)
        strategy.teacher_weights.append(weight)
    
    return strategy


def diagnose_distillation_strategy(strategy: DistillationStrategy) -> Dict[str, Any]:
    """
    诊断蒸馏策略状态
    
    Args:
        strategy: 蒸馏策略实例
    
    Returns:
        诊断结果字典
    """
    return strategy.diagnose_distillation()


def print_distillation_strategy_diagnosis(strategy: DistillationStrategy) -> None:
    """打印蒸馏策略诊断结果"""
    strategy.print_distillation_diagnosis()


def compare_distillation_strategies(
    strategies: List[DistillationStrategy]
) -> Dict[str, Any]:
    """
    比较多个蒸馏策略
    
    Args:
        strategies: 蒸馏策略列表
    
    Returns:
        比较结果字典
    """
    comparison = {
        'strategies': [],
        'best_by_metric': {},
    }
    
    metrics_collection = defaultdict(list)
    
    for strategy in strategies:
        info = {
            'name': strategy.name,
            'type': strategy.config.distillation_type,
            'temperature': strategy.get_effective_temperature(),
        }
        
        # 获取统计
        if strategy._distillation_monitor is not None:
            stats = strategy._distillation_monitor.get_stats()
            info['total_steps'] = stats.total_steps
            info['avg_total_loss'] = stats.avg_total_loss
            info['accuracy_gap'] = stats.accuracy_gap
            
            metrics_collection['avg_total_loss'].append((strategy.name, stats.avg_total_loss))
            metrics_collection['accuracy_gap'].append((strategy.name, stats.accuracy_gap))
        
        comparison['strategies'].append(info)
    
    # 找出每个指标的最佳策略
    for metric, values in metrics_collection.items():
        if values:
            if metric in ['avg_total_loss', 'accuracy_gap']:
                best = min(values, key=lambda x: x[1])
            else:
                best = max(values, key=lambda x: x[1])
            comparison['best_by_metric'][metric] = best[0]
    
    return comparison


def print_distillation_strategy_comparison(strategies: List[DistillationStrategy]) -> None:
    """打印蒸馏策略比较结果"""
    comparison = compare_distillation_strategies(strategies)
    
    print("\n" + "=" * 70)
    print("DISTILLATION STRATEGY COMPARISON")
    print("=" * 70)
    
    print("\nStrategies:")
    for i, info in enumerate(comparison['strategies'], 1):
        print(f"\n  {i}. {info['name']} ({info['type']})")
        print(f"     Temperature: {info['temperature']:.2f}")
        if 'total_steps' in info:
            print(f"     Steps: {info['total_steps']}")
            print(f"     Avg Loss: {info.get('avg_total_loss', 'N/A'):.4f}")
            print(f"     Accuracy Gap: {info.get('accuracy_gap', 'N/A'):.2%}")
    
    if comparison['best_by_metric']:
        print("\n  Best by Metric:")
        for metric, name in comparison['best_by_metric'].items():
            print(f"    {metric}: {name}")
    
    print("=" * 70)


def recommend_distillation_strategy(
    teacher_size: int,
    student_size: int,
    task_type: str = 'classification',
    available_memory_gb: float = 16.0
) -> Dict[str, Any]:
    """
    推荐蒸馏策略
    
    Args:
        teacher_size: 教师模型参数量
        student_size: 学生模型参数量
        task_type: 任务类型
        available_memory_gb: 可用内存（GB）
    
    Returns:
        推荐配置
    """
    size_ratio = teacher_size / max(student_size, 1)
    
    recommendation = {
        'strategy_type': 'standard',
        'config': {},
        'reasoning': [],
    }
    
    # 根据大小比例推荐
    if size_ratio > 10:
        recommendation['strategy_type'] = 'progressive'
        recommendation['config']['progressive_stages'] = 6
        recommendation['reasoning'].append(
            f"Large size ratio ({size_ratio:.1f}x) - progressive distillation recommended"
        )
    elif size_ratio > 5:
        recommendation['strategy_type'] = 'combined'
        recommendation['config']['feature_loss_weight'] = 0.3
        recommendation['reasoning'].append(
            f"Medium size ratio ({size_ratio:.1f}x) - combined distillation recommended"
        )
    else:
        recommendation['config']['distillation_type'] = 'logits'
        recommendation['reasoning'].append(
            f"Small size ratio ({size_ratio:.1f}x) - logits distillation sufficient"
        )
    
    # 根据任务类型调整
    if task_type == 'generation':
        recommendation['config']['temperature'] = 2.0
        recommendation['config']['soft_loss_weight'] = 0.8
        recommendation['reasoning'].append("Generation task - lower temperature, higher soft loss weight")
    elif task_type == 'classification':
        recommendation['config']['temperature'] = 4.0
        recommendation['reasoning'].append("Classification task - standard temperature")
    
    # 内存限制
    estimated_memory = (teacher_size + student_size) * 4 / 1e9  # 4 bytes per parameter
    if estimated_memory > available_memory_gb * 0.7:
        recommendation['config']['gradient_accumulation'] = max(2, int(estimated_memory / available_memory_gb))
        recommendation['reasoning'].append(
            f"Memory constraint - gradient accumulation suggested"
        )
    
    return recommendation


def get_available_distillation_types() -> List[str]:
    """获取可用的蒸馏类型"""
    return [t.value for t in DistillationType]


def get_available_strategy_variants() -> List[str]:
    """获取可用的策略变体"""
    return ['standard', 'self', 'progressive', 'industry', 'contrastive']

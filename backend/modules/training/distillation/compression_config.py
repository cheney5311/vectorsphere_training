# -*- coding: utf-8 -*-
"""
模型压缩和知识蒸馏配置模型

提供完整的配置支持，包括：
- 基础蒸馏配置
- 场景化蒸馏配置（行业、多模态、边缘部署等）
- 分布式蒸馏配置
- 自适应蒸馏配置
- 模型压缩配置

架构调用层次：
├── compression_config.py (本模块)
│   └── 调用 backend/modules/training/strategies (策略层)
│       ├── base_strategy.py - StrategyType, TrainingPhase, StrategyContext
│       ├── distributed_strategy.py - DistributedMode, ZeROStage
│       └── distillation_strategy.py - DistillationType
│   └── 调用 backend/lib/losses (损失函数层)
│   └── 调用 backend/lib/distributed (分布式层)
│   └── 调用 backend/lib/hardware (硬件层)
└── 被 knowledge_distillation.py, distillation_service.py 调用
"""

import sys
import os as os_path
import json
import time
import logging
from pathlib import Path

current_dir = os_path.path.dirname(os_path.path.abspath(__file__))
project_root = os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(current_dir))))
sys.path.insert(0, project_root)

from dataclasses import dataclass, field, fields
from typing import Optional, List, Dict, Any, Tuple, Union, Callable
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from backend.core.exceptions import ValidationError
except ImportError:
    class ValidationError(Exception):
        """验证异常"""

# ======================== 策略层导入 ========================

from backend.modules.training.strategies.base_strategy import (
    StrategyType,
    TrainingPhase,
    StrategyContext,
    StrategyResult,
    StrategyMonitor,
    StrategyProfiler,
    StrategyValidator,
    StrategyMetrics,
)

from backend.modules.training.strategies.distributed_strategy import (
    DistributedMode as StrategyDistributedMode,
    ZeROStage,
    DistributedStrategyConfig,
    DistributedStrategy,
    DistributedHealthStatus,
    CommunicationStats,
)

# ======================== 底层 lib 模块导入 ========================

from backend.lib.losses import (
    LossFactory,
    create_loss,
    BaseLoss,
    LossMonitor,
    LossStats,
    LossResult,
)

from backend.lib.hardware import (
    DeviceManager,
    get_device_manager,
    MemoryManager,
    get_available_memory,
    MemoryStats,
)

from backend.lib.distributed import (
    DistributedManager,
    get_distributed_manager,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
)


# ======================== 枚举定义 ========================

class DistillationScenario(Enum):
    """蒸馏场景枚举"""
    STANDARD = "standard"  # 标准蒸馏
    INDUSTRY = "industry"  # 行业蒸馏
    MULTIMODAL = "multimodal"  # 多模态蒸馏
    EDGE_DEPLOY = "edge_deploy"  # 边缘部署蒸馏
    REAL_TIME = "real_time"  # 实时推理蒸馏
    LOW_LATENCY = "low_latency"  # 低延迟蒸馏
    HIGH_ACCURACY = "high_accuracy"  # 高精度蒸馏
    PROGRESSIVE = "progressive"  # 渐进式蒸馏
    SELF_DISTILLATION = "self_distillation"  # 自蒸馏
    CONTRASTIVE = "contrastive"  # 对比蒸馏

    @classmethod
    def from_string(cls, value: str) -> 'DistillationScenario':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown distillation scenario: {value}")

    @property
    def requires_teacher(self) -> bool:
        """是否需要教师模型"""
        return self not in [DistillationScenario.SELF_DISTILLATION]

    @property
    def supports_feature_distillation(self) -> bool:
        """是否支持特征蒸馏"""
        return self in [
            DistillationScenario.STANDARD,
            DistillationScenario.INDUSTRY,
            DistillationScenario.MULTIMODAL,
            DistillationScenario.HIGH_ACCURACY,
            DistillationScenario.PROGRESSIVE,
        ]

    @property
    def memory_efficiency(self) -> float:
        """内存效率评分 (0-1, 越高越好)"""
        scores = {
            DistillationScenario.STANDARD: 0.5,
            DistillationScenario.INDUSTRY: 0.6,
            DistillationScenario.MULTIMODAL: 0.4,
            DistillationScenario.EDGE_DEPLOY: 0.9,
            DistillationScenario.REAL_TIME: 0.8,
            DistillationScenario.LOW_LATENCY: 0.85,
            DistillationScenario.HIGH_ACCURACY: 0.3,
            DistillationScenario.PROGRESSIVE: 0.6,
            DistillationScenario.SELF_DISTILLATION: 0.7,
            DistillationScenario.CONTRASTIVE: 0.5,
        }
        return scores.get(self, 0.5)

    @property
    def typical_compression_ratio(self) -> float:
        """典型压缩比"""
        ratios = {
            DistillationScenario.STANDARD: 0.5,
            DistillationScenario.INDUSTRY: 0.6,
            DistillationScenario.MULTIMODAL: 0.7,
            DistillationScenario.EDGE_DEPLOY: 0.2,
            DistillationScenario.REAL_TIME: 0.3,
            DistillationScenario.LOW_LATENCY: 0.25,
            DistillationScenario.HIGH_ACCURACY: 0.8,
            DistillationScenario.PROGRESSIVE: 0.5,
            DistillationScenario.SELF_DISTILLATION: 1.0,
            DistillationScenario.CONTRASTIVE: 0.5,
        }
        return ratios.get(self, 0.5)

    def to_strategy_type(self) -> Optional['StrategyType']:
        """转换为策略类型"""
        return StrategyType.DISTILLATION

    def get_description(self) -> str:
        """获取场景描述"""
        descriptions = {
            DistillationScenario.STANDARD: "标准知识蒸馏，适用于通用场景",
            DistillationScenario.INDUSTRY: "行业模型蒸馏，针对垂直领域优化",
            DistillationScenario.MULTIMODAL: "多模态蒸馏，支持跨模态知识迁移",
            DistillationScenario.EDGE_DEPLOY: "边缘部署蒸馏，极致压缩优化",
            DistillationScenario.REAL_TIME: "实时推理蒸馏，优化推理延迟",
            DistillationScenario.LOW_LATENCY: "低延迟蒸馏，追求极低响应时间",
            DistillationScenario.HIGH_ACCURACY: "高精度蒸馏，最大化保留精度",
            DistillationScenario.PROGRESSIVE: "渐进式蒸馏，分阶段知识迁移",
            DistillationScenario.SELF_DISTILLATION: "自蒸馏，无需教师模型",
            DistillationScenario.CONTRASTIVE: "对比蒸馏，基于对比学习",
        }
        return descriptions.get(self, "未知蒸馏场景")

    @classmethod
    def recommend(cls, target_device: str, target_latency_ms: float,
                  accuracy_priority: float = 0.5) -> 'DistillationScenario':
        """根据目标需求推荐场景"""
        if target_device in ['edge', 'mobile', 'embedded']:
            return cls.EDGE_DEPLOY
        if target_latency_ms < 10:
            return cls.LOW_LATENCY
        if target_latency_ms < 50:
            return cls.REAL_TIME
        if accuracy_priority > 0.8:
            return cls.HIGH_ACCURACY
        return cls.STANDARD


class DistributedMode(Enum):
    """分布式模式枚举"""
    SINGLE = "single"  # 单机单卡
    DATA_PARALLEL = "data_parallel"  # 数据并行
    MODEL_PARALLEL = "model_parallel"  # 模型并行
    PIPELINE = "pipeline"  # 流水线并行
    HYBRID = "hybrid"  # 混合并行
    FSDP = "fsdp"  # 全分片数据并行
    ZERO = "zero"  # DeepSpeed ZeRO

    @classmethod
    def from_string(cls, value: str) -> 'DistributedMode':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown distributed mode: {value}")

    @property
    def requires_multiple_gpus(self) -> bool:
        """是否需要多GPU"""
        return self != DistributedMode.SINGLE

    @property
    def supports_cpu_offload(self) -> bool:
        """是否支持CPU卸载"""
        return self in [DistributedMode.FSDP, DistributedMode.ZERO]

    @property
    def memory_efficiency(self) -> float:
        """内存效率评分"""
        scores = {
            DistributedMode.SINGLE: 0.3,
            DistributedMode.DATA_PARALLEL: 0.4,
            DistributedMode.MODEL_PARALLEL: 0.7,
            DistributedMode.PIPELINE: 0.75,
            DistributedMode.HYBRID: 0.85,
            DistributedMode.FSDP: 0.8,
            DistributedMode.ZERO: 0.9,
        }
        return scores.get(self, 0.5)

    def to_strategy_distributed_mode(self) -> Optional['StrategyDistributedMode']:
        """转换为策略层分布式模式"""
        mode_map = {
            DistributedMode.SINGLE: None,
            DistributedMode.DATA_PARALLEL: StrategyDistributedMode.DDP,
            DistributedMode.MODEL_PARALLEL: StrategyDistributedMode.TENSOR,
            DistributedMode.PIPELINE: StrategyDistributedMode.PIPELINE,
            DistributedMode.HYBRID: StrategyDistributedMode.HYBRID,
            DistributedMode.FSDP: StrategyDistributedMode.FSDP,
            DistributedMode.ZERO: StrategyDistributedMode.ZERO,
        }
        return mode_map.get(self)

    def get_description(self) -> str:
        """获取描述"""
        descriptions = {
            DistributedMode.SINGLE: "单机单卡训练",
            DistributedMode.DATA_PARALLEL: "数据并行，复制模型到多卡",
            DistributedMode.MODEL_PARALLEL: "模型并行，分割模型到多卡",
            DistributedMode.PIPELINE: "流水线并行，模型按层分割",
            DistributedMode.HYBRID: "混合并行，组合多种策略",
            DistributedMode.FSDP: "全分片数据并行，支持大模型",
            DistributedMode.ZERO: "DeepSpeed ZeRO优化",
        }
        return descriptions.get(self, "未知分布式模式")


class AdaptiveMode(Enum):
    """自适应模式枚举"""
    NONE = "none"  # 不启用自适应
    TEMPERATURE = "temperature"  # 自适应温度
    LAYER = "layer"  # 自适应层选择
    LOSS_WEIGHT = "loss_weight"  # 自适应损失权重
    FULL = "full"  # 全自适应
    CURRICULUM = "curriculum"  # 课程学习
    DYNAMIC_KD = "dynamic_kd"  # 动态知识蒸馏

    @classmethod
    def from_string(cls, value: str) -> 'AdaptiveMode':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown adaptive mode: {value}")

    @property
    def complexity(self) -> float:
        """复杂度评分 (0-1)"""
        scores = {
            AdaptiveMode.NONE: 0.0,
            AdaptiveMode.TEMPERATURE: 0.2,
            AdaptiveMode.LAYER: 0.4,
            AdaptiveMode.LOSS_WEIGHT: 0.3,
            AdaptiveMode.FULL: 1.0,
            AdaptiveMode.CURRICULUM: 0.5,
            AdaptiveMode.DYNAMIC_KD: 0.6,
        }
        return scores.get(self, 0.5)

    @property
    def requires_validation_set(self) -> bool:
        """是否需要验证集"""
        return self in [AdaptiveMode.FULL, AdaptiveMode.CURRICULUM, AdaptiveMode.DYNAMIC_KD]

    def get_description(self) -> str:
        """获取描述"""
        descriptions = {
            AdaptiveMode.NONE: "不启用自适应，使用固定参数",
            AdaptiveMode.TEMPERATURE: "自适应调整蒸馏温度",
            AdaptiveMode.LAYER: "自适应选择蒸馏层",
            AdaptiveMode.LOSS_WEIGHT: "自适应调整损失权重",
            AdaptiveMode.FULL: "全自适应，综合调整所有参数",
            AdaptiveMode.CURRICULUM: "课程学习，由易到难训练",
            AdaptiveMode.DYNAMIC_KD: "动态知识蒸馏，根据学生状态调整",
        }
        return descriptions.get(self, "未知自适应模式")


class CompressionMethod(Enum):
    """压缩方法枚举"""
    QUANTIZATION = "quantization"  # 量化
    PRUNING = "pruning"  # 剪枝
    DISTILLATION = "distillation"  # 蒸馏
    LOW_RANK = "low_rank"  # 低秩分解
    WEIGHT_SHARING = "weight_sharing"  # 权重共享
    MIXED = "mixed"  # 混合压缩

    @classmethod
    def from_string(cls, value: str) -> 'CompressionMethod':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown compression method: {value}")

    @property
    def typical_speedup(self) -> float:
        """典型加速比"""
        speedups = {
            CompressionMethod.QUANTIZATION: 2.0,
            CompressionMethod.PRUNING: 1.5,
            CompressionMethod.DISTILLATION: 3.0,
            CompressionMethod.LOW_RANK: 1.3,
            CompressionMethod.WEIGHT_SHARING: 1.2,
            CompressionMethod.MIXED: 4.0,
        }
        return speedups.get(self, 1.0)

    @property
    def typical_size_reduction(self) -> float:
        """典型模型大小缩减比例"""
        reductions = {
            CompressionMethod.QUANTIZATION: 0.25,  # 4x smaller
            CompressionMethod.PRUNING: 0.5,
            CompressionMethod.DISTILLATION: 0.3,
            CompressionMethod.LOW_RANK: 0.6,
            CompressionMethod.WEIGHT_SHARING: 0.7,
            CompressionMethod.MIXED: 0.1,
        }
        return reductions.get(self, 0.5)


# ======================== 监控和统计组件 ========================

@dataclass
class DistillationStats:
    """蒸馏统计数据"""
    total_steps: int = 0
    total_kd_loss: float = 0.0
    total_ce_loss: float = 0.0
    total_feature_loss: float = 0.0
    total_attention_loss: float = 0.0
    avg_kd_loss: float = 0.0
    avg_ce_loss: float = 0.0
    avg_feature_loss: float = 0.0
    avg_attention_loss: float = 0.0
    best_accuracy: float = 0.0
    teacher_accuracy: float = 0.0
    student_accuracy: float = 0.0
    compression_ratio: float = 1.0
    speedup_ratio: float = 1.0

    def update(self, kd_loss: float, ce_loss: float = 0.0,
               feature_loss: float = 0.0, attention_loss: float = 0.0) -> None:
        """更新统计"""
        self.total_steps += 1
        self.total_kd_loss += kd_loss
        self.total_ce_loss += ce_loss
        self.total_feature_loss += feature_loss
        self.total_attention_loss += attention_loss

        self.avg_kd_loss = self.total_kd_loss / self.total_steps
        self.avg_ce_loss = self.total_ce_loss / self.total_steps if ce_loss > 0 else 0
        self.avg_feature_loss = self.total_feature_loss / self.total_steps if feature_loss > 0 else 0
        self.avg_attention_loss = self.total_attention_loss / self.total_steps if attention_loss > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_steps': self.total_steps,
            'avg_kd_loss': self.avg_kd_loss,
            'avg_ce_loss': self.avg_ce_loss,
            'avg_feature_loss': self.avg_feature_loss,
            'avg_attention_loss': self.avg_attention_loss,
            'best_accuracy': self.best_accuracy,
            'teacher_accuracy': self.teacher_accuracy,
            'student_accuracy': self.student_accuracy,
            'compression_ratio': self.compression_ratio,
            'speedup_ratio': self.speedup_ratio,
        }


class DistillationMonitor:
    """蒸馏过程监控器"""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._stats = DistillationStats()
        self._loss_history: List[float] = []
        self._accuracy_history: List[float] = []
        self._temperature_history: List[float] = []

        # 使用策略层监控器（如果可用）
        self._strategy_monitor: Optional['StrategyMonitor'] = None
        try:
            self._strategy_monitor = StrategyMonitor(history_size=history_size)
        except Exception as e:
            logger.warning(f"Failed to init StrategyMonitor: {e}")

        # 使用损失层监控器（如果可用）
        self._loss_monitor: Optional['LossMonitor'] = None
        try:
            self._loss_monitor = LossMonitor()
        except Exception as e:
            logger.warning(f"Failed to init LossMonitor: {e}")

    def record_step(self, kd_loss: float, ce_loss: float = 0.0,
                    feature_loss: float = 0.0, attention_loss: float = 0.0,
                    accuracy: float = 0.0, temperature: float = 0.0) -> None:
        """记录一步"""
        self._stats.update(kd_loss, ce_loss, feature_loss, attention_loss)

        total_loss = kd_loss + ce_loss + feature_loss + attention_loss
        self._loss_history.append(total_loss)
        if len(self._loss_history) > self.history_size:
            self._loss_history.pop(0)

        if accuracy > 0:
            self._accuracy_history.append(accuracy)
            if len(self._accuracy_history) > self.history_size:
                self._accuracy_history.pop(0)
            if accuracy > self._stats.best_accuracy:
                self._stats.best_accuracy = accuracy

        if temperature > 0:
            self._temperature_history.append(temperature)
            if len(self._temperature_history) > self.history_size:
                self._temperature_history.pop(0)

        # 同步到策略层监控器
        if self._strategy_monitor is not None:
            try:
                # 创建一个简单的 StrategyResult 
                if StrategyResult is not None:
                    import torch
                    result = StrategyResult(
                        loss=torch.tensor(total_loss),
                        metrics={'kd_loss': kd_loss, 'accuracy': accuracy},
                        # 添加缺失的参数
                        step_time=0.0,
                    )
                    if StrategyContext is not None:
                        context = StrategyContext(global_step=self._stats.total_steps)
                        self._strategy_monitor.record_step(result, context)
            except Exception:
                pass

        # 同步到损失层监控器
        if self._loss_monitor is not None:
            try:
                if LossResult is not None:
                    import torch
                    loss_result = LossResult(
                        loss=torch.tensor(total_loss),
                        components={
                            'kd': torch.tensor(kd_loss),
                            'ce': torch.tensor(ce_loss)
                        },
                        metrics={
                            'feature_loss': feature_loss,
                            'attention_loss': attention_loss,
                            'total_loss': total_loss
                        }
                    )
                    self._loss_monitor.record(loss_result)
            except Exception:
                pass

    def get_stats(self) -> DistillationStats:
        """获取统计数据"""
        return self._stats

    def get_loss_trend(self, window: int = 100) -> str:
        """获取损失趋势"""
        if len(self._loss_history) < window:
            return "insufficient_data"

        recent = self._loss_history[-window:]
        first_half = sum(recent[:window // 2]) / (window // 2)
        second_half = sum(recent[window // 2:]) / (window // 2)

        if second_half < first_half * 0.95:
            return "improving"
        elif second_half > first_half * 1.05:
            return "degrading"
        else:
            return "stable"

    def is_converged(self, patience: int = 10, threshold: float = 1e-4) -> bool:
        """检查是否收敛"""
        if len(self._loss_history) < patience:
            return False

        recent = self._loss_history[-patience:]
        min_loss = min(recent)
        max_loss = max(recent)

        return (max_loss - min_loss) < threshold

    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        summary = {
            'stats': self._stats.to_dict(),
            'loss_trend': self.get_loss_trend(),
            'is_converged': self.is_converged(),
            'loss_history_size': len(self._loss_history),
            'strategy_monitor_available': self._strategy_monitor is not None,
            'loss_monitor_available': self._loss_monitor is not None,
        }

        # 添加策略监控器摘要
        if self._strategy_monitor is not None and hasattr(self._strategy_monitor, 'get_summary'):
            try:
                summary['strategy_summary'] = self._strategy_monitor.get_summary()
            except Exception:
                pass

        return summary


class ConfigValidator:
    """配置验证器"""

    def __init__(self):
        self._checks: List[Callable[[Any], Tuple[bool, str]]] = []

        # 使用策略层验证器
        self._strategy_validator: Optional['StrategyValidator'] = None
        try:
            self._strategy_validator = StrategyValidator()
        except Exception:
            pass

    def add_check(self, check: Callable[[Any], Tuple[bool, str]]) -> None:
        """添加验证规则"""
        self._checks.append(check)

    def validate(self, config: Any) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []

        for check in self._checks:
            try:
                is_valid, message = check(config)
                if not is_valid:
                    errors.append(message)
            except Exception as e:
                errors.append(f"Validation check failed: {e}")

        return len(errors) == 0, errors

    def get_strategy_validator(self) -> Optional['StrategyValidator']:
        """获取策略验证器"""
        return self._strategy_validator


# ======================== 配置类 ========================

@dataclass
class DistillationConfig:
    """
    知识蒸馏基础配置
    
    生产级配置类，支持：
    - 完整的参数验证
    - 序列化和反序列化
    - 与策略层集成
    - 自动优化推荐
    """
    teacher_model_path: str = ""
    student_model_path: str = ""
    temperature: float = 4.0
    alpha: float = 0.7  # 蒸馏损失权重（软标签）
    beta: float = 0.3  # 学生损失权重（硬标签）

    # 特征蒸馏
    use_feature_distillation: bool = True
    feature_loss_weight: float = 0.1
    feature_layers: List[int] = field(default_factory=lambda: [-1, -2, -3])
    feature_loss_type: str = "mse"  # mse, cosine, l1

    # 注意力蒸馏
    use_attention_distillation: bool = True
    attention_loss_weight: float = 0.1
    attention_layers: List[int] = field(default_factory=lambda: [-1])
    attention_loss_type: str = "kl"  # kl, mse

    # 对比蒸馏
    use_contrastive_distillation: bool = False
    contrastive_loss_weight: float = 0.1
    contrastive_temperature: float = 0.5
    contrastive_projector_dim: int = 256

    # 关系蒸馏
    use_relational_distillation: bool = False
    relational_loss_weight: float = 0.1

    # 温度调度
    temperature_schedule: str = "constant"  # constant, linear, cosine, adaptive
    min_temperature: float = 1.0
    max_temperature: float = 10.0

    # 层权重策略
    layer_weight_strategy: str = "uniform"  # uniform, linear, exponential, adaptive

    # 创建时间戳
    _created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """初始化后处理"""
        # 验证参数约束
        if self.alpha + self.beta > 1.0:
            logger.warning(f"alpha ({self.alpha}) + beta ({self.beta}) > 1.0, normalizing...")
            total = self.alpha + self.beta
            self.alpha = self.alpha / total
            self.beta = self.beta / total

    def validate(self) -> None:
        """验证配置参数"""
        errors = []

        if not self.teacher_model_path:
            errors.append("教师模型路径不能为空")

        if not self.student_model_path:
            errors.append("学生模型路径不能为空")

        if self.temperature <= 0:
            errors.append("温度参数必须大于0")

        if self.alpha < 0 or self.alpha > 1:
            errors.append("alpha参数必须在0-1之间")

        if self.beta < 0 or self.beta > 1:
            errors.append("beta参数必须在0-1之间")

        if self.feature_loss_weight < 0:
            errors.append("feature_loss_weight必须>=0")

        if self.attention_loss_weight < 0:
            errors.append("attention_loss_weight必须>=0")

        if self.contrastive_loss_weight < 0:
            errors.append("contrastive_loss_weight必须>=0")

        if self.min_temperature > self.max_temperature:
            errors.append("min_temperature不能大于max_temperature")

        if errors:
            raise ValidationError("; ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'teacher_model_path': self.teacher_model_path,
            'student_model_path': self.student_model_path,
            'temperature': self.temperature,
            'alpha': self.alpha,
            'beta': self.beta,
            'use_feature_distillation': self.use_feature_distillation,
            'feature_loss_weight': self.feature_loss_weight,
            'feature_layers': self.feature_layers,
            'feature_loss_type': self.feature_loss_type,
            'use_attention_distillation': self.use_attention_distillation,
            'attention_loss_weight': self.attention_loss_weight,
            'attention_layers': self.attention_layers,
            'attention_loss_type': self.attention_loss_type,
            'use_contrastive_distillation': self.use_contrastive_distillation,
            'contrastive_loss_weight': self.contrastive_loss_weight,
            'contrastive_temperature': self.contrastive_temperature,
            'contrastive_projector_dim': self.contrastive_projector_dim,
            'use_relational_distillation': self.use_relational_distillation,
            'relational_loss_weight': self.relational_loss_weight,
            'temperature_schedule': self.temperature_schedule,
            'min_temperature': self.min_temperature,
            'max_temperature': self.max_temperature,
            'layer_weight_strategy': self.layer_weight_strategy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistillationConfig':
        """从字典创建"""
        try:
            valid_fields = {f.name for f in fields(cls) if not f.name.startswith('_')}
        except (TypeError, ValueError):
            # Fallback if not a dataclass or other issue
            valid_fields = set(data.keys())
            
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'DistillationConfig':
        """从JSON创建"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save(self, path: str) -> None:
        """保存到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> 'DistillationConfig':
        """从文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())

    def copy(self) -> 'DistillationConfig':
        """创建副本"""
        return DistillationConfig.from_dict(self.to_dict())

    def get_total_loss_weight(self) -> float:
        """获取总损失权重"""
        total = self.alpha + self.beta
        if self.use_feature_distillation:
            total += self.feature_loss_weight
        if self.use_attention_distillation:
            total += self.attention_loss_weight
        if self.use_contrastive_distillation:
            total += self.contrastive_loss_weight
        if self.use_relational_distillation:
            total += self.relational_loss_weight
        return total

    def get_enabled_methods(self) -> List[str]:
        """获取启用的蒸馏方法"""
        methods = ['soft_label']  # 软标签始终启用
        if self.use_feature_distillation:
            methods.append('feature')
        if self.use_attention_distillation:
            methods.append('attention')
        if self.use_contrastive_distillation:
            methods.append('contrastive')
        if self.use_relational_distillation:
            methods.append('relational')
        return methods

    def create_loss_functions(self) -> Dict[str, Any]:
        """
        创建损失函数
        
        使用 backend/lib/losses 创建损失函数（如果可用）
        """
        losses = {}

        try:
            # 软标签损失（KL散度）
            losses['kl_loss'] = create_loss('kl_div')

            # 特征损失
            if self.use_feature_distillation:
                if self.feature_loss_type == 'mse':
                    losses['feature_loss'] = create_loss('mse')
                elif self.feature_loss_type == 'cosine':
                    losses['feature_loss'] = create_loss('cosine')
                else:
                    losses['feature_loss'] = create_loss('l1')

            # 对比损失
            if self.use_contrastive_distillation:
                losses['contrastive_loss'] = create_loss('infonce')

            logger.info(f"Created {len(losses)} loss functions via lib.losses")
        except Exception as e:
            logger.warning(f"Failed to create losses via lib.losses: {e}")

        return losses

    def estimate_memory_mb(self, model_size_gb: float) -> float:
        """估算蒸馏所需内存(MB)"""
        # 基础：教师模型 + 学生模型
        base_mb = model_size_gb * 1024 * 2  # 2x for both models

        # 特征蒸馏需要额外激活内存
        if self.use_feature_distillation:
            base_mb *= 1.2

        # 注意力蒸馏
        if self.use_attention_distillation:
            base_mb *= 1.1

        # 对比蒸馏需要额外投影器
        if self.use_contrastive_distillation:
            base_mb += 50  # projector

        return base_mb

    def summary(self) -> str:
        """获取配置摘要"""
        methods = self.get_enabled_methods()
        return (
            f"DistillationConfig(T={self.temperature}, α={self.alpha}, β={self.beta}, "
            f"methods={methods}, schedule={self.temperature_schedule})"
        )


@dataclass
class ScenarioDistillationConfig:
    """
    场景化蒸馏配置
    
    针对不同业务场景的蒸馏配置优化，支持：
    - 行业场景（制造、金融、医疗等）
    - 多模态场景
    - 边缘部署场景
    - 实时推理场景
    - 高精度场景
    
    与策略层集成，支持自动配置优化
    """
    scenario: str = "standard"  # 蒸馏场景

    # 行业场景配置
    industry_type: Optional[str] = None  # manufacturing, finance, healthcare, etc.
    domain_adaptation: bool = False
    domain_loss_weight: float = 0.1
    domain_specific_layers: List[str] = field(default_factory=list)

    # 多模态场景配置
    modalities: List[str] = field(default_factory=lambda: ["text"])
    modality_weights: Dict[str, float] = field(default_factory=dict)
    cross_modal_distillation: bool = False
    modality_alignment: bool = False

    # 边缘部署场景配置
    target_device: str = "cpu"  # cpu, gpu, npu, edge, mobile
    target_latency_ms: float = 100.0
    target_memory_mb: float = 512.0
    enable_quantization: bool = True
    quantization_bits: int = 8
    enable_pruning: bool = False
    pruning_ratio: float = 0.0

    # 实时推理场景配置
    max_batch_size: int = 1
    streaming_mode: bool = False
    prefetch_enabled: bool = False

    # 精度优化场景配置
    target_accuracy: float = 0.95  # 目标精度保持率
    accuracy_threshold: float = 0.02  # 允许的精度下降阈值
    accuracy_monitor_interval: int = 100

    # 渐进式蒸馏配置
    progressive_stages: int = 1
    progressive_schedule: str = "linear"  # linear, exponential, step

    # 监控配置
    enable_monitoring: bool = True
    monitoring_interval: int = 50

    def __post_init__(self):
        """初始化后处理"""
        # 规范化场景名
        self.scenario = self.scenario.lower().strip()

        # 设置默认模态权重
        if not self.modality_weights and self.modalities:
            weight = 1.0 / len(self.modalities)
            self.modality_weights = {m: weight for m in self.modalities}

    def validate(self) -> None:
        """验证配置"""
        errors = []

        # 验证场景
        try:
            DistillationScenario.from_string(self.scenario)
        except ValueError:
            errors.append(f"Unknown scenario: {self.scenario}")

        # 验证目标设备
        if self.target_device not in ['cpu', 'gpu', 'npu', 'edge', 'mobile', 'embedded']:
            errors.append(f"Unknown target device: {self.target_device}")

        # 验证量化位数
        if self.quantization_bits not in [4, 8, 16, 32]:
            errors.append(f"Invalid quantization bits: {self.quantization_bits}")

        # 验证延迟目标
        if self.target_latency_ms <= 0:
            errors.append("target_latency_ms must be positive")

        # 验证精度目标
        if self.target_accuracy < 0 or self.target_accuracy > 1:
            errors.append("target_accuracy must be in [0, 1]")

        # 验证剪枝比例
        if self.pruning_ratio < 0 or self.pruning_ratio > 0.9:
            errors.append("pruning_ratio must be in [0, 0.9]")

        if errors:
            raise ValidationError("; ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'scenario': self.scenario,
            'industry_type': self.industry_type,
            'domain_adaptation': self.domain_adaptation,
            'domain_loss_weight': self.domain_loss_weight,
            'domain_specific_layers': self.domain_specific_layers,
            'modalities': self.modalities,
            'modality_weights': self.modality_weights,
            'cross_modal_distillation': self.cross_modal_distillation,
            'modality_alignment': self.modality_alignment,
            'target_device': self.target_device,
            'target_latency_ms': self.target_latency_ms,
            'target_memory_mb': self.target_memory_mb,
            'enable_quantization': self.enable_quantization,
            'quantization_bits': self.quantization_bits,
            'enable_pruning': self.enable_pruning,
            'pruning_ratio': self.pruning_ratio,
            'max_batch_size': self.max_batch_size,
            'streaming_mode': self.streaming_mode,
            'prefetch_enabled': self.prefetch_enabled,
            'target_accuracy': self.target_accuracy,
            'accuracy_threshold': self.accuracy_threshold,
            'accuracy_monitor_interval': self.accuracy_monitor_interval,
            'progressive_stages': self.progressive_stages,
            'progressive_schedule': self.progressive_schedule,
            'enable_monitoring': self.enable_monitoring,
            'monitoring_interval': self.monitoring_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioDistillationConfig':
        """从字典创建"""
        try:
            valid_fields = {f.name for f in fields(cls)}
        except (TypeError, ValueError):
            valid_fields = set(data.keys())
            
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def get_scenario_enum(self) -> DistillationScenario:
        """获取场景枚举"""
        return DistillationScenario.from_string(self.scenario)

    def get_strategy_type(self) -> Optional['StrategyType']:
        """获取策略类型"""
        scenario_enum = self.get_scenario_enum()
        return scenario_enum.to_strategy_type()

    def get_training_phase(self) -> Optional['TrainingPhase']:
        """获取建议的训练阶段"""
        if self.scenario == "industry":
            return TrainingPhase.FINETUNE_SCENE
        elif self.scenario in ["edge_deploy", "low_latency", "real_time"]:
            return TrainingPhase.FINETUNE
        else:
            return TrainingPhase.MAIN

    def get_optimized_config(self) -> Dict[str, Any]:
        """根据场景获取优化后的配置"""
        config = {}

        if self.scenario == "edge_deploy":
            config.update({
                'temperature': 6.0,  # 更高温度，更软的分布
                'alpha': 0.9,  # 更重视软标签
                'use_feature_distillation': True,
                'feature_layers': [-1],  # 只蒸馏最后一层
                'enable_quantization': True,
                'quantization_bits': self.quantization_bits,
            })
        elif self.scenario == "low_latency":
            config.update({
                'temperature': 4.0,
                'alpha': 0.7,
                'use_attention_distillation': False,  # 跳过注意力蒸馏
                'max_batch_size': 1,
            })
        elif self.scenario == "high_accuracy":
            config.update({
                'temperature': 2.0,  # 较低温度保持精度
                'alpha': 0.5,
                'beta': 0.5,
                'use_feature_distillation': True,
                'use_attention_distillation': True,
                'feature_layers': [-1, -2, -3, -4],  # 更多层
            })
        elif self.scenario == "industry":
            config.update({
                'temperature': 4.0,
                'domain_adaptation': True,
                'use_feature_distillation': True,
                'domain_loss_weight': self.domain_loss_weight,
            })
        elif self.scenario == "multimodal":
            config.update({
                'temperature': 4.0,
                'cross_modal_distillation': self.cross_modal_distillation,
                'modality_alignment': self.modality_alignment,
                'modality_weights': self.modality_weights,
            })
        elif self.scenario == "progressive":
            config.update({
                'temperature': 4.0,
                'progressive_stages': self.progressive_stages,
                'progressive_schedule': self.progressive_schedule,
            })
        elif self.scenario == "real_time":
            config.update({
                'temperature': 4.0,
                'alpha': 0.8,
                'streaming_mode': self.streaming_mode,
                'max_batch_size': self.max_batch_size,
            })
        elif self.scenario == "self_distillation":
            config.update({
                'temperature': 3.0,
                'alpha': 0.5,
                'beta': 0.5,
                'use_feature_distillation': True,
            })
        elif self.scenario == "contrastive":
            config.update({
                'temperature': 0.5,
                'use_contrastive_distillation': True,
                'contrastive_loss_weight': 0.3,
            })

        return config

    def estimate_compression_ratio(self) -> float:
        """估算压缩比"""
        base_ratio = 1.0

        # 量化影响
        if self.enable_quantization:
            base_ratio *= (self.quantization_bits / 32.0)

        # 剪枝影响
        if self.enable_pruning:
            base_ratio *= (1.0 - self.pruning_ratio)

        # 场景特定因子
        scenario_enum = self.get_scenario_enum()
        base_ratio *= scenario_enum.typical_compression_ratio

        return base_ratio

    def estimate_latency_ms(self, base_latency_ms: float) -> float:
        """估算推理延迟"""
        estimated = base_latency_ms

        # 量化加速
        if self.enable_quantization:
            estimated *= (self.quantization_bits / 32.0) ** 0.5

        # 剪枝加速
        if self.enable_pruning:
            estimated *= (1.0 - self.pruning_ratio * 0.5)

        return estimated

    def meets_requirements(self, actual_latency_ms: float, actual_memory_mb: float,
                           actual_accuracy: float) -> Dict[str, bool]:
        """检查是否满足要求"""
        return {
            'latency_ok': actual_latency_ms <= self.target_latency_ms,
            'memory_ok': actual_memory_mb <= self.target_memory_mb,
            'accuracy_ok': actual_accuracy >= self.target_accuracy,
        }

    def summary(self) -> str:
        """获取配置摘要"""
        return (
            f"ScenarioConfig(scenario={self.scenario}, device={self.target_device}, "
            f"latency≤{self.target_latency_ms}ms, memory≤{self.target_memory_mb}MB, "
            f"accuracy≥{self.target_accuracy:.0%})"
        )


@dataclass
class DistributedDistillationConfig:
    """
    分布式蒸馏配置
    
    支持多GPU/多节点蒸馏训练，与策略层分布式策略集成：
    - 调用 backend/modules/training/strategies/distributed_strategy.py
    - 调用 backend/lib/distributed 分布式层
    - 调用 backend/lib/hardware 硬件层
    """
    mode: str = "single"  # 分布式模式

    # 数据并行配置
    world_size: int = 1  # 总进程数
    rank: int = 0  # 当前进程rank
    local_rank: int = 0  # 本地GPU rank

    # 通信配置
    backend: str = "nccl"  # nccl, gloo, mpi
    master_addr: str = "localhost"
    master_port: int = 29500
    timeout_minutes: int = 30

    # 同步配置
    sync_bn: bool = True  # 同步BatchNorm
    gradient_accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    find_unused_parameters: bool = False

    # 混合精度配置
    use_amp: bool = True  # 自动混合精度
    amp_dtype: str = "float16"  # float16, bfloat16

    # ZeRO优化配置
    use_zero: bool = False
    zero_stage: int = 2  # 0, 1, 2, 3
    zero_offload: bool = False
    zero_offload_optimizer: bool = False

    # FSDP配置
    use_fsdp: bool = False
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False

    # 检查点配置
    activation_checkpointing: bool = False
    checkpoint_interval: int = 1000
    checkpoint_path: str = "./checkpoints"

    # 健康检查配置
    health_check_interval: int = 100
    auto_recovery: bool = True
    max_recovery_attempts: int = 3

    # 监控配置
    enable_profiling: bool = False
    enable_communication_stats: bool = True

    def __post_init__(self):
        """初始化后处理"""
        self.mode = self.mode.lower().strip()

    def validate(self) -> None:
        """验证配置"""
        errors = []

        # 验证分布式模式
        try:
            DistributedMode.from_string(self.mode)
        except ValueError:
            errors.append(f"Unknown distributed mode: {self.mode}")

        # 验证 world_size 和 rank
        if self.world_size < 1:
            errors.append("world_size must be >= 1")
        if self.rank < 0 or self.rank >= self.world_size:
            errors.append(f"rank must be in [0, {self.world_size})")
        if self.local_rank < 0:
            errors.append("local_rank must be >= 0")

        # 验证通信后端
        if self.backend not in ['nccl', 'gloo', 'mpi']:
            errors.append(f"Unknown backend: {self.backend}")

        # 验证 ZeRO 阶段
        if self.use_zero and self.zero_stage not in [0, 1, 2, 3]:
            errors.append(f"Invalid ZeRO stage: {self.zero_stage}")

        # 验证梯度累积步数
        if self.gradient_accumulation_steps < 1:
            errors.append("gradient_accumulation_steps must be >= 1")

        if errors:
            raise ValidationError("; ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'mode': self.mode,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'backend': self.backend,
            'master_addr': self.master_addr,
            'master_port': self.master_port,
            'timeout_minutes': self.timeout_minutes,
            'sync_bn': self.sync_bn,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'gradient_clip_norm': self.gradient_clip_norm,
            'find_unused_parameters': self.find_unused_parameters,
            'use_amp': self.use_amp,
            'amp_dtype': self.amp_dtype,
            'use_zero': self.use_zero,
            'zero_stage': self.zero_stage,
            'zero_offload': self.zero_offload,
            'zero_offload_optimizer': self.zero_offload_optimizer,
            'use_fsdp': self.use_fsdp,
            'fsdp_sharding_strategy': self.fsdp_sharding_strategy,
            'fsdp_cpu_offload': self.fsdp_cpu_offload,
            'activation_checkpointing': self.activation_checkpointing,
            'checkpoint_interval': self.checkpoint_interval,
            'checkpoint_path': self.checkpoint_path,
            'health_check_interval': self.health_check_interval,
            'auto_recovery': self.auto_recovery,
            'max_recovery_attempts': self.max_recovery_attempts,
            'enable_profiling': self.enable_profiling,
            'enable_communication_stats': self.enable_communication_stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedDistillationConfig':
        """从字典创建"""
        try:
            valid_fields = {f.name for f in fields(cls)}
        except (TypeError, ValueError):
            valid_fields = set(data.keys())
            
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_env(cls) -> 'DistributedDistillationConfig':
        """从环境变量创建"""
        import os
        return cls(
            world_size=int(os_path.environ.get('WORLD_SIZE', '1')),
            rank=int(os_path.environ.get('RANK', '0')),
            local_rank=int(os_path.environ.get('LOCAL_RANK', '0')),
            master_addr=os_path.environ.get('MASTER_ADDR', 'localhost'),
            master_port=int(os_path.environ.get('MASTER_PORT', '29500')),
        )

    def is_distributed(self) -> bool:
        """是否为分布式模式"""
        return self.world_size > 1

    def is_main_process(self) -> bool:
        """是否为主进程"""
        # 优先使用策略层的 is_main_process
        try:
            return is_main_process()
        except Exception:
            pass
        return self.rank == 0

    def get_mode_enum(self) -> DistributedMode:
        """获取分布式模式枚举"""
        return DistributedMode.from_string(self.mode)

    def to_strategy_config(self) -> Optional['DistributedStrategyConfig']:
        """
        转换为策略层分布式配置
        
        使用 backend/modules/training/strategies/distributed_strategy.py
        """
        try:
            # 映射分布式模式
            mode_enum = self.get_mode_enum()
            strategy_mode = mode_enum.to_strategy_distributed_mode()

            if strategy_mode is None:
                strategy_mode = StrategyDistributedMode.DDP

            # 映射 ZeRO 阶段
            zero_stage_enum = None
            if self.use_zero and ZeROStage is not None:
                try:
                    zero_stage_enum = ZeROStage.from_int(self.zero_stage)
                except ValueError:
                    zero_stage_enum = ZeROStage.STAGE_2

            return DistributedStrategyConfig(
                mode=strategy_mode,
                world_size=self.world_size,
                rank=self.rank,
                local_rank=self.local_rank,
                master_addr=self.master_addr,
                master_port=str(self.master_port),
                backend=self.backend,
                timeout_minutes=self.timeout_minutes,
                find_unused_parameters=self.find_unused_parameters,
                broadcast_buffers=True,
                sync_bn=self.sync_bn,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                gradient_clip_norm=self.gradient_clip_norm,
                zero_stage=zero_stage_enum if zero_stage_enum else ZeROStage.STAGE_2,
                zero_offload=self.zero_offload,
                zero_offload_optimizer=self.zero_offload_optimizer,
                sharding_strategy=self.fsdp_sharding_strategy,
                cpu_offload=self.fsdp_cpu_offload,
                enable_monitoring=True,
                enable_profiling=self.enable_profiling,
                health_check_interval=self.health_check_interval,
                auto_recovery=self.auto_recovery,
                max_recovery_attempts=self.max_recovery_attempts,
            )
        except Exception as e:
            logger.warning(f"Failed to create DistributedStrategyConfig: {e}")
            return None

    def create_distributed_strategy(self) -> Optional['DistributedStrategy']:
        """
        创建分布式策略
        
        使用 backend/modules/training/strategies/distributed_strategy.py
        """
        try:
            config = self.to_strategy_config()
            if config is not None:
                strategy = DistributedStrategy(config)
                logger.info(f"Created DistributedStrategy: {strategy.config.summary()}")
                return strategy
        except Exception as e:
            logger.warning(f"Failed to create DistributedStrategy: {e}")

        return None

    def get_communication_stats(self) -> Optional['CommunicationStats']:
        """获取通信统计"""
        return CommunicationStats()

    def get_health_status(self) -> Optional['DistributedHealthStatus']:
        """获取健康状态"""
        return DistributedHealthStatus()

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计
        
        使用 backend/lib/hardware
        """
        stats = {}

        try:
            if get_available_memory is not None:
                stats['available_memory_gb'] = get_available_memory()
            if get_device_manager is not None:
                dm = get_device_manager()
                if hasattr(dm, 'get_device_info'):
                    stats['device_info'] = str(dm.get_device_info())
        except Exception as e:
            logger.warning(f"Failed to get memory stats: {e}")

        return stats

    def sync_all_processes(self) -> None:
        """
        同步所有进程
        
        使用 backend/lib/distributed
        """
        try:
            barrier()
        except Exception as e:
            logger.warning(f"Failed to sync processes: {e}")

    def get_effective_batch_size(self, per_device_batch_size: int) -> int:
        """获取有效批次大小"""
        return per_device_batch_size * self.world_size * self.gradient_accumulation_steps

    def summary(self) -> str:
        """获取配置摘要"""
        mode_str = f"mode={self.mode}"
        if self.use_zero:
            mode_str += f", ZeRO-{self.zero_stage}"
        if self.use_fsdp:
            mode_str += f", FSDP"

        return (
            f"DistributedConfig({mode_str}, "
            f"world_size={self.world_size}, rank={self.rank}, "
            f"backend={self.backend}, amp={self.use_amp})"
        )


@dataclass
class AdaptiveDistillationConfig:
    """
    自适应蒸馏配置
    
    根据训练过程动态调整蒸馏参数，支持：
    - 自适应温度调度
    - 自适应层选择
    - 自适应损失权重
    - 课程学习
    - 早停机制
    
    与策略层集成，使用 StrategyMonitor 进行监控
    """
    mode: str = "none"  # 自适应模式

    # 自适应温度配置
    temperature_range: Tuple[float, float] = (1.0, 10.0)
    temperature_decay: float = 0.99
    temperature_schedule: str = "constant"  # constant, linear, cosine, adaptive
    temperature_warmup_steps: int = 0

    # 自适应层配置
    layer_selection_strategy: str = "loss_based"  # loss_based, gradient_based, attention_based, random
    min_layers: int = 1
    max_layers: int = 6
    layer_update_interval: int = 500
    layer_importance_threshold: float = 0.1

    # 自适应损失权重配置
    weight_adjustment_interval: int = 100
    weight_smoothing: float = 0.9
    weight_min: float = 0.01
    weight_max: float = 1.0
    weight_balance_method: str = "inverse"  # inverse, softmax, uniform

    # 课程学习配置
    curriculum_enabled: bool = False
    curriculum_schedule: str = "linear"  # linear, exponential, step
    curriculum_start_ratio: float = 0.5
    curriculum_end_ratio: float = 1.0
    curriculum_warmup_epochs: int = 2

    # 早停配置
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    early_stopping_metric: str = "loss"  # loss, accuracy

    # 学习率自适应
    lr_adaptive: bool = True
    lr_warmup_steps: int = 1000
    lr_decay_factor: float = 0.95
    lr_min: float = 1e-7
    lr_schedule: str = "cosine"  # constant, linear, cosine, step

    # 监控配置
    monitor_interval: int = 50

    def __post_init__(self):
        """初始化后处理"""
        self.mode = self.mode.lower().strip()

        # 确保 temperature_range 是元组
        if isinstance(self.temperature_range, list):
            self.temperature_range = tuple(self.temperature_range)

    def validate(self) -> None:
        """验证配置"""
        errors = []

        # 验证自适应模式
        try:
            AdaptiveMode.from_string(self.mode)
        except ValueError:
            errors.append(f"Unknown adaptive mode: {self.mode}")

        # 验证温度范围
        if self.temperature_range[0] >= self.temperature_range[1]:
            errors.append("temperature_range[0] must be < temperature_range[1]")
        if self.temperature_range[0] <= 0:
            errors.append("temperature_range[0] must be positive")

        # 验证层范围
        if self.min_layers < 1:
            errors.append("min_layers must be >= 1")
        if self.max_layers < self.min_layers:
            errors.append("max_layers must be >= min_layers")

        # 验证权重范围
        if self.weight_min < 0 or self.weight_max > 1:
            errors.append("weight range must be in [0, 1]")
        if self.weight_min >= self.weight_max:
            errors.append("weight_min must be < weight_max")

        # 验证早停
        if self.early_stopping and self.patience < 1:
            errors.append("patience must be >= 1 when early_stopping is enabled")

        if errors:
            raise ValidationError("; ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'mode': self.mode,
            'temperature_range': list(self.temperature_range),
            'temperature_decay': self.temperature_decay,
            'temperature_schedule': self.temperature_schedule,
            'temperature_warmup_steps': self.temperature_warmup_steps,
            'layer_selection_strategy': self.layer_selection_strategy,
            'min_layers': self.min_layers,
            'max_layers': self.max_layers,
            'layer_update_interval': self.layer_update_interval,
            'layer_importance_threshold': self.layer_importance_threshold,
            'weight_adjustment_interval': self.weight_adjustment_interval,
            'weight_smoothing': self.weight_smoothing,
            'weight_min': self.weight_min,
            'weight_max': self.weight_max,
            'weight_balance_method': self.weight_balance_method,
            'curriculum_enabled': self.curriculum_enabled,
            'curriculum_schedule': self.curriculum_schedule,
            'curriculum_start_ratio': self.curriculum_start_ratio,
            'curriculum_end_ratio': self.curriculum_end_ratio,
            'curriculum_warmup_epochs': self.curriculum_warmup_epochs,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'early_stopping_metric': self.early_stopping_metric,
            'lr_adaptive': self.lr_adaptive,
            'lr_warmup_steps': self.lr_warmup_steps,
            'lr_decay_factor': self.lr_decay_factor,
            'lr_min': self.lr_min,
            'lr_schedule': self.lr_schedule,
            'monitor_interval': self.monitor_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdaptiveDistillationConfig':
        """从字典创建"""
        try:
            valid_fields = {f.name for f in fields(cls)}
        except (TypeError, ValueError):
            valid_fields = set(data.keys())
            
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        # 处理 temperature_range
        if 'temperature_range' in filtered and isinstance(filtered['temperature_range'], list):
            filtered['temperature_range'] = tuple(filtered['temperature_range'])
        return cls(**filtered)

    def get_mode_enum(self) -> AdaptiveMode:
        """获取自适应模式枚举"""
        return AdaptiveMode.from_string(self.mode)

    def get_temperature(self, step: int, total_steps: int) -> float:
        """
        获取当前步骤的温度
        
        根据调度策略计算温度
        """
        min_temp, max_temp = self.temperature_range

        if self.temperature_schedule == "constant":
            return (min_temp + max_temp) / 2

        # 处理预热
        if step < self.temperature_warmup_steps:
            progress = step / max(self.temperature_warmup_steps, 1)
            return min_temp + (max_temp - min_temp) * progress

        # 计算主调度阶段的进度
        adjusted_step = step - self.temperature_warmup_steps
        adjusted_total = max(total_steps - self.temperature_warmup_steps, 1)
        progress = min(adjusted_step / adjusted_total, 1.0)

        if self.temperature_schedule == "linear":
            return max_temp - (max_temp - min_temp) * progress
        elif self.temperature_schedule == "cosine":
            import math
            return min_temp + (max_temp - min_temp) * 0.5 * (1 + math.cos(math.pi * progress))
        elif self.temperature_schedule == "adaptive":
            # 自适应需要外部反馈，这里返回衰减值
            decay_factor = self.temperature_decay ** step
            return max(min_temp, max_temp * decay_factor)
        else:
            return (min_temp + max_temp) / 2

    def get_curriculum_ratio(self, epoch: int, total_epochs: int) -> float:
        """
        获取课程学习比例
        
        决定当前训练使用多少比例的数据难度
        """
        if not self.curriculum_enabled:
            return 1.0

        if epoch < self.curriculum_warmup_epochs:
            return self.curriculum_start_ratio

        adjusted_epoch = epoch - self.curriculum_warmup_epochs
        adjusted_total = max(total_epochs - self.curriculum_warmup_epochs, 1)
        progress = min(adjusted_epoch / adjusted_total, 1.0)

        if self.curriculum_schedule == "linear":
            return self.curriculum_start_ratio + (self.curriculum_end_ratio - self.curriculum_start_ratio) * progress
        elif self.curriculum_schedule == "exponential":
            import math
            return self.curriculum_start_ratio + (self.curriculum_end_ratio - self.curriculum_start_ratio) * (
                        1 - math.exp(-3 * progress))
        elif self.curriculum_schedule == "step":
            steps = 5
            step_progress = int(progress * steps) / steps
            return self.curriculum_start_ratio + (
                        self.curriculum_end_ratio - self.curriculum_start_ratio) * step_progress
        else:
            return 1.0

    def should_update_layers(self, step: int) -> bool:
        """判断是否应该更新层选择"""
        mode = self.get_mode_enum()
        if mode not in [AdaptiveMode.LAYER, AdaptiveMode.FULL]:
            return False
        return step > 0 and step % self.layer_update_interval == 0

    def should_update_weights(self, step: int) -> bool:
        """判断是否应该更新权重"""
        mode = self.get_mode_enum()
        if mode not in [AdaptiveMode.LOSS_WEIGHT, AdaptiveMode.FULL]:
            return False
        return step > 0 and step % self.weight_adjustment_interval == 0

    def get_learning_rate(self, step: int, base_lr: float, total_steps: int) -> float:
        """获取当前学习率"""
        if not self.lr_adaptive:
            return base_lr

        # 预热
        if step < self.lr_warmup_steps:
            return base_lr * (step + 1) / self.lr_warmup_steps

        # 主调度
        adjusted_step = step - self.lr_warmup_steps
        adjusted_total = max(total_steps - self.lr_warmup_steps, 1)
        progress = min(adjusted_step / adjusted_total, 1.0)

        if self.lr_schedule == "constant":
            return base_lr
        elif self.lr_schedule == "linear":
            return max(self.lr_min, base_lr * (1 - progress))
        elif self.lr_schedule == "cosine":
            import math
            return max(self.lr_min, self.lr_min + (base_lr - self.lr_min) * 0.5 * (1 + math.cos(math.pi * progress)))
        elif self.lr_schedule == "step":
            decay_steps = adjusted_step // 1000
            return max(self.lr_min, base_lr * (self.lr_decay_factor ** decay_steps))
        else:
            return base_lr

    def create_monitor(self) -> Optional[DistillationMonitor]:
        """创建监控器"""
        return DistillationMonitor(history_size=10000)

    def summary(self) -> str:
        """获取配置摘要"""
        features = []
        if self.mode != "none":
            features.append(f"mode={self.mode}")
        if self.early_stopping:
            features.append(f"early_stop(patience={self.patience})")
        if self.curriculum_enabled:
            features.append("curriculum")
        if self.lr_adaptive:
            features.append(f"lr_sched={self.lr_schedule}")

        return f"AdaptiveConfig({', '.join(features) if features else 'none'})"


@dataclass
class DistillationTaskConfig:
    """
    蒸馏任务配置
    
    完整的蒸馏任务配置，包含所有子配置，支持：
    - 任务管理（ID、租户、用户）
    - 完整的蒸馏配置
    - 分布式训练配置
    - 自适应配置
    - 数据和输出管理
    - 与策略层的完整集成
    """
    # 基础配置
    task_id: Optional[str] = None
    task_name: str = "distillation_task"
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # 蒸馏配置
    distillation_config: Optional[DistillationConfig] = None

    # 场景配置
    scenario_config: Optional[ScenarioDistillationConfig] = None

    # 分布式配置
    distributed_config: Optional[DistributedDistillationConfig] = None

    # 自适应配置
    adaptive_config: Optional[AdaptiveDistillationConfig] = None

    # 训练配置
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    seed: int = 42

    # 数据配置
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    data_format: str = "json"  # json, csv, parquet
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    preprocessing_num_workers: int = 4

    # 输出配置
    output_dir: str = "./output"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    save_best_only: bool = True

    # 监控配置
    enable_monitoring: bool = True
    monitoring_interval: int = 10
    enable_profiling: bool = False
    enable_tensorboard: bool = False
    enable_wandb: bool = False

    # 时间戳
    _created_at: float = field(default_factory=time.time)
    _updated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """初始化后处理"""
        import uuid
        if self.task_id is None:
            self.task_id = str(uuid.uuid4())[:8]

    def validate(self) -> None:
        """验证所有配置"""
        errors = []

        # 验证基础参数
        if self.num_epochs < 1:
            errors.append("num_epochs must be >= 1")
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            errors.append("warmup_ratio must be in [0, 1]")

        # 验证子配置
        try:
            if self.distillation_config:
                self.distillation_config.validate()
        except ValidationError as e:
            errors.append(f"distillation_config: {e}")

        try:
            if self.scenario_config:
                self.scenario_config.validate()
        except ValidationError as e:
            errors.append(f"scenario_config: {e}")

        try:
            if self.distributed_config:
                self.distributed_config.validate()
        except ValidationError as e:
            errors.append(f"distributed_config: {e}")

        try:
            if self.adaptive_config:
                self.adaptive_config.validate()
        except ValidationError as e:
            errors.append(f"adaptive_config: {e}")

        if errors:
            raise ValidationError("; ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'tenant_id': self.tenant_id,
            'user_id': self.user_id,
            'description': self.description,
            'tags': self.tags,
            'distillation_config': self.distillation_config.to_dict() if self.distillation_config else None,
            'scenario_config': self.scenario_config.to_dict() if self.scenario_config else None,
            'distributed_config': self.distributed_config.to_dict() if self.distributed_config else None,
            'adaptive_config': self.adaptive_config.to_dict() if self.adaptive_config else None,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm,
            'warmup_ratio': self.warmup_ratio,
            'seed': self.seed,
            'train_data_path': self.train_data_path,
            'eval_data_path': self.eval_data_path,
            'data_format': self.data_format,
            'max_train_samples': self.max_train_samples,
            'max_eval_samples': self.max_eval_samples,
            'preprocessing_num_workers': self.preprocessing_num_workers,
            'output_dir': self.output_dir,
            'save_steps': self.save_steps,
            'eval_steps': self.eval_steps,
            'logging_steps': self.logging_steps,
            'save_total_limit': self.save_total_limit,
            'save_best_only': self.save_best_only,
            'enable_monitoring': self.enable_monitoring,
            'monitoring_interval': self.monitoring_interval,
            'enable_profiling': self.enable_profiling,
            'enable_tensorboard': self.enable_tensorboard,
            'enable_wandb': self.enable_wandb,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistillationTaskConfig':
        """从字典创建"""
        # 处理子配置
        if 'distillation_config' in data and data['distillation_config']:
            if isinstance(data['distillation_config'], dict):
                data['distillation_config'] = DistillationConfig.from_dict(data['distillation_config'])

        if 'scenario_config' in data and data['scenario_config']:
            if isinstance(data['scenario_config'], dict):
                data['scenario_config'] = ScenarioDistillationConfig.from_dict(data['scenario_config'])

        if 'distributed_config' in data and data['distributed_config']:
            if isinstance(data['distributed_config'], dict):
                data['distributed_config'] = DistributedDistillationConfig.from_dict(data['distributed_config'])

        if 'adaptive_config' in data and data['adaptive_config']:
            if isinstance(data['adaptive_config'], dict):
                data['adaptive_config'] = AdaptiveDistillationConfig.from_dict(data['adaptive_config'])

        try:
            valid_fields = {f.name for f in fields(cls) if not f.name.startswith('_')}
        except (TypeError, ValueError):
            valid_fields = set(data.keys())
            
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> 'DistillationTaskConfig':
        """从JSON创建"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save(self, path: str) -> None:
        """保存到文件"""
        # 确保只在主进程保存
        if self.distributed_config and not self.distributed_config.is_main_process():
            return

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        logger.info(f"Task config saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'DistillationTaskConfig':
        """从文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())

    def copy(self) -> 'DistillationTaskConfig':
        """创建副本"""
        return DistillationTaskConfig.from_dict(self.to_dict())

    def get_total_steps(self, dataset_size: int) -> int:
        """计算总训练步数"""
        steps_per_epoch = dataset_size // self.batch_size

        if self.distributed_config and self.distributed_config.is_distributed():
            steps_per_epoch = steps_per_epoch // self.distributed_config.world_size
            if self.distributed_config.gradient_accumulation_steps > 1:
                steps_per_epoch = steps_per_epoch // self.distributed_config.gradient_accumulation_steps

        return steps_per_epoch * self.num_epochs

    def get_warmup_steps(self, total_steps: int) -> int:
        """计算预热步数"""
        return int(total_steps * self.warmup_ratio)

    def get_effective_batch_size(self) -> int:
        """获取有效批次大小"""
        effective = self.batch_size
        if self.distributed_config:
            effective = self.distributed_config.get_effective_batch_size(self.batch_size)
        return effective

    def get_strategy_context(self) -> Optional['StrategyContext']:
        """
        获取策略上下文
        
        使用 backend/modules/training/strategies/base_strategy.py
        """
        try:
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            return StrategyContext(
                device=device,
                config={
                    'task_id': self.task_id,
                    'task_name': self.task_name,
                    'learning_rate': self.learning_rate,
                    'num_epochs': self.num_epochs,
                    'batch_size': self.batch_size,
                },
                max_epochs=self.num_epochs,
            )
        except Exception as e:
            logger.warning(f"Failed to create StrategyContext: {e}")
            return None

    def create_distributed_strategy(self) -> Optional['DistributedStrategy']:
        """创建分布式策略"""
        if self.distributed_config:
            return self.distributed_config.create_distributed_strategy()
        return None

    def create_monitor(self) -> DistillationMonitor:
        """创建监控器"""
        if self.adaptive_config:
            monitor = self.adaptive_config.create_monitor()
            if monitor:
                return monitor
        return DistillationMonitor()

    def estimate_training_time_hours(self, samples_per_second: float = 10.0) -> float:
        """估算训练时间（小时）"""
        if self.train_data_path is None:
            return 0.0

        # 估算数据集大小
        estimated_samples = self.max_train_samples or 100000
        total_steps = self.get_total_steps(estimated_samples)

        # 计算时间
        total_seconds = total_steps / samples_per_second
        return total_seconds / 3600

    def summary(self) -> str:
        """获取配置摘要"""
        parts = [f"Task({self.task_name})"]

        if self.distillation_config:
            parts.append(self.distillation_config.summary())
        if self.scenario_config:
            parts.append(self.scenario_config.summary())
        if self.distributed_config and self.distributed_config.is_distributed():
            parts.append(self.distributed_config.summary())
        if self.adaptive_config and self.adaptive_config.mode != "none":
            parts.append(self.adaptive_config.summary())

        return " | ".join(parts)


@dataclass
class CompressionConfig:
    """
    模型压缩配置
    
    支持多种压缩方法的组合：
    - 量化（动态/静态/QAT）
    - 剪枝（幅度/结构化/非结构化）
    - 知识蒸馏
    - 低秩分解
    - 权重共享
    
    与策略层集成，支持自动优化
    """
    # 量化配置
    use_quantization: bool = True
    quantization_type: str = "dynamic"  # dynamic, static, qat
    quantization_bits: int = 8
    quantization_backend: str = "fbgemm"  # fbgemm, qnnpack
    quantization_calibration_samples: int = 100
    quantization_per_channel: bool = True

    # 剪枝配置
    use_pruning: bool = True
    pruning_method: str = "magnitude"  # magnitude, structured, unstructured, movement
    pruning_ratio: float = 0.1
    pruning_schedule: str = "one_shot"  # one_shot, iterative, gradual
    pruning_start_epoch: int = 0
    pruning_end_epoch: int = -1  # -1 means last epoch
    pruning_frequency: int = 1

    # 知识蒸馏
    use_distillation: bool = True
    distillation_config: Optional[DistillationConfig] = None

    # 低秩分解
    use_low_rank: bool = False
    low_rank_ratio: float = 0.5
    low_rank_target_layers: List[str] = field(default_factory=list)

    # 权重共享
    use_weight_sharing: bool = False
    sharing_groups: List[str] = field(default_factory=list)

    # 组合策略
    compression_order: List[str] = field(default_factory=lambda: ["pruning", "quantization", "distillation"])

    # 目标约束
    target_size_mb: Optional[float] = None
    target_latency_ms: Optional[float] = None
    target_accuracy_retention: float = 0.95

    # 监控配置
    enable_monitoring: bool = True
    profile_compression: bool = False

    def __post_init__(self):
        """初始化后处理"""
        # 如果没有指定蒸馏配置但启用了蒸馏，创建默认配置
        if self.use_distillation and self.distillation_config is None:
            self.distillation_config = DistillationConfig()

    def validate(self) -> None:
        """验证配置参数"""
        errors = []

        # 验证量化配置
        if self.use_quantization:
            if self.quantization_bits not in [4, 8, 16, 32]:
                errors.append("量化位数必须是4、8、16或32")
            if self.quantization_type not in ['dynamic', 'static', 'qat']:
                errors.append(f"Unknown quantization type: {self.quantization_type}")
            if self.quantization_backend not in ['fbgemm', 'qnnpack']:
                errors.append(f"Unknown quantization backend: {self.quantization_backend}")

        # 验证剪枝配置
        if self.use_pruning:
            if self.pruning_ratio < 0 or self.pruning_ratio > 0.9:
                errors.append("剪枝比例必须在0-0.9之间")
            if self.pruning_method not in ['magnitude', 'structured', 'unstructured', 'movement']:
                errors.append(f"Unknown pruning method: {self.pruning_method}")
            if self.pruning_schedule not in ['one_shot', 'iterative', 'gradual']:
                errors.append(f"Unknown pruning schedule: {self.pruning_schedule}")

        # 验证低秩分解
        if self.use_low_rank:
            if self.low_rank_ratio <= 0 or self.low_rank_ratio > 1:
                errors.append("low_rank_ratio must be in (0, 1]")

        # 验证蒸馏配置
        if self.use_distillation and self.distillation_config:
            try:
                self.distillation_config.validate()
            except ValidationError as e:
                errors.append(f"distillation_config: {e}")

        # 验证目标约束
        if self.target_accuracy_retention < 0 or self.target_accuracy_retention > 1:
            errors.append("target_accuracy_retention must be in [0, 1]")

        if errors:
            raise ValidationError("; ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'use_quantization': self.use_quantization,
            'quantization_type': self.quantization_type,
            'quantization_bits': self.quantization_bits,
            'quantization_backend': self.quantization_backend,
            'quantization_calibration_samples': self.quantization_calibration_samples,
            'quantization_per_channel': self.quantization_per_channel,
            'use_pruning': self.use_pruning,
            'pruning_method': self.pruning_method,
            'pruning_ratio': self.pruning_ratio,
            'pruning_schedule': self.pruning_schedule,
            'pruning_start_epoch': self.pruning_start_epoch,
            'pruning_end_epoch': self.pruning_end_epoch,
            'pruning_frequency': self.pruning_frequency,
            'use_distillation': self.use_distillation,
            'distillation_config': self.distillation_config.to_dict() if self.distillation_config else None,
            'use_low_rank': self.use_low_rank,
            'low_rank_ratio': self.low_rank_ratio,
            'low_rank_target_layers': self.low_rank_target_layers,
            'use_weight_sharing': self.use_weight_sharing,
            'sharing_groups': self.sharing_groups,
            'compression_order': self.compression_order,
            'target_size_mb': self.target_size_mb,
            'target_latency_ms': self.target_latency_ms,
            'target_accuracy_retention': self.target_accuracy_retention,
            'enable_monitoring': self.enable_monitoring,
            'profile_compression': self.profile_compression,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressionConfig':
        """从字典创建"""
        if 'distillation_config' in data and data['distillation_config']:
            if isinstance(data['distillation_config'], dict):
                data['distillation_config'] = DistillationConfig.from_dict(data['distillation_config'])

        try:
            valid_fields = {f.name for f in fields(cls)}
        except (TypeError, ValueError):
            valid_fields = set(data.keys())
            
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def get_enabled_methods(self) -> List[str]:
        """获取启用的压缩方法"""
        methods = []
        if self.use_quantization:
            methods.append('quantization')
        if self.use_pruning:
            methods.append('pruning')
        if self.use_distillation:
            methods.append('distillation')
        if self.use_low_rank:
            methods.append('low_rank')
        if self.use_weight_sharing:
            methods.append('weight_sharing')
        return methods

    def estimate_compression_ratio(self) -> float:
        """估算总压缩比"""
        ratio = 1.0

        if self.use_quantization:
            ratio *= (self.quantization_bits / 32.0)

        if self.use_pruning:
            ratio *= (1.0 - self.pruning_ratio)

        if self.use_low_rank:
            ratio *= self.low_rank_ratio

        return ratio

    def estimate_speedup(self) -> float:
        """估算加速比"""
        speedup = 1.0

        if self.use_quantization:
            # 量化通常带来2-4x加速
            speedup *= (32.0 / self.quantization_bits) ** 0.5

        if self.use_pruning:
            # 剪枝加速取决于稀疏性
            speedup *= 1.0 / (1.0 - self.pruning_ratio * 0.5)

        return speedup

    def estimate_final_size_mb(self, original_size_mb: float) -> float:
        """估算压缩后大小"""
        return original_size_mb * self.estimate_compression_ratio()

    def meets_constraints(self, actual_size_mb: float, actual_latency_ms: float,
                          actual_accuracy: float, original_accuracy: float) -> Dict[str, bool]:
        """检查是否满足约束"""
        results = {}

        if self.target_size_mb is not None:
            results['size_ok'] = actual_size_mb <= self.target_size_mb

        if self.target_latency_ms is not None:
            results['latency_ok'] = actual_latency_ms <= self.target_latency_ms

        accuracy_retention = actual_accuracy / original_accuracy if original_accuracy > 0 else 0
        results['accuracy_ok'] = accuracy_retention >= self.target_accuracy_retention

        return results

    def create_strategy_metrics(self) -> Optional['StrategyMetrics']:
        """创建策略指标"""
        return StrategyMetrics()

    def create_profiler(self) -> Optional['StrategyProfiler']:
        """创建性能分析器"""
        if self.profile_compression:
            return StrategyProfiler()
        return None

    def summary(self) -> str:
        """获取配置摘要"""
        methods = self.get_enabled_methods()
        ratio = self.estimate_compression_ratio()
        speedup = self.estimate_speedup()

        return (
            f"CompressionConfig(methods={methods}, "
            f"ratio≈{ratio:.2f}, speedup≈{speedup:.1f}x)"
        )


# ======================== 预设配置模板 ========================

class DistillationPresets:
    """
    蒸馏配置预设模板
    
    提供常用场景的预配置，支持：
    - 边缘部署
    - 高精度
    - 行业模型
    - 多模态
    - 大规模分布式
    - 渐进式蒸馏
    - 自蒸馏
    - 对比蒸馏
    """

    _presets: Dict[str, Callable[[], DistillationTaskConfig]] = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        """注册预设的装饰器"""

        def decorator(func: Callable[[], DistillationTaskConfig]) -> Callable:
            cls._presets[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str, **kwargs) -> DistillationTaskConfig:
        """获取预设配置"""
        name = name.lower().strip()

        # 检查注册的预设
        if name in cls._presets:
            return cls._presets[name](**kwargs)

        # 检查静态方法
        method_map = {
            'edge_deployment': cls.edge_deployment,
            'edge': cls.edge_deployment,
            'high_accuracy': cls.high_accuracy,
            'accuracy': cls.high_accuracy,
            'industry': cls.industry_model,
            'multimodal': cls.multimodal,
            'distributed': cls.distributed_large_scale,
            'progressive': cls.progressive_distillation,
            'self_distillation': cls.self_distillation,
            'contrastive': cls.contrastive_distillation,
            'standard': cls.standard,
            'low_latency': cls.low_latency,
            'real_time': cls.real_time,
        }

        if name in method_map:
            return method_map[name](**kwargs)

        raise ValueError(f"Unknown preset: {name}. Available: {list(method_map.keys())}")

    @classmethod
    def list_presets(cls) -> List[str]:
        """列出所有预设"""
        return [
            'edge_deployment', 'high_accuracy', 'industry', 'multimodal',
            'distributed', 'progressive', 'self_distillation', 'contrastive',
            'standard', 'low_latency', 'real_time'
        ] + list(cls._presets.keys())

    @staticmethod
    def standard() -> DistillationTaskConfig:
        """标准蒸馏场景预设"""
        return DistillationTaskConfig(
            task_name="standard_distillation",
            distillation_config=DistillationConfig(
                temperature=4.0,
                alpha=0.7,
                beta=0.3,
                use_feature_distillation=True,
                feature_layers=[-1, -2, -3],
                use_attention_distillation=True,
            ),
            scenario_config=ScenarioDistillationConfig(
                scenario="standard",
            ),
            num_epochs=10,
            batch_size=32,
        )

    @staticmethod
    def edge_deployment() -> DistillationTaskConfig:
        """边缘部署场景预设"""
        return DistillationTaskConfig(
            task_name="edge_deployment_distillation",
            description="针对边缘设备优化的蒸馏配置",
            distillation_config=DistillationConfig(
                temperature=6.0,
                alpha=0.9,
                beta=0.1,
                use_feature_distillation=True,
                feature_layers=[-1],
                use_attention_distillation=False,
            ),
            scenario_config=ScenarioDistillationConfig(
                scenario="edge_deploy",
                target_device="edge",
                target_latency_ms=50.0,
                target_memory_mb=256.0,
                enable_quantization=True,
                quantization_bits=8,
                enable_pruning=True,
                pruning_ratio=0.3,
            ),
            num_epochs=5,
            batch_size=16,
        )

    @staticmethod
    def high_accuracy() -> DistillationTaskConfig:
        """高精度场景预设"""
        return DistillationTaskConfig(
            task_name="high_accuracy_distillation",
            description="最大化保留精度的蒸馏配置",
            distillation_config=DistillationConfig(
                temperature=2.0,
                alpha=0.5,
                beta=0.5,
                use_feature_distillation=True,
                feature_layers=[-1, -2, -3, -4],
                use_attention_distillation=True,
                attention_layers=[-1, -2],
                use_relational_distillation=True,
            ),
            scenario_config=ScenarioDistillationConfig(
                scenario="high_accuracy",
                target_accuracy=0.98,
                accuracy_threshold=0.01,
            ),
            adaptive_config=AdaptiveDistillationConfig(
                mode="full",
                early_stopping=True,
                patience=15,
                min_delta=0.0001,
            ),
            num_epochs=20,
            batch_size=32,
        )

    @staticmethod
    def industry_model(industry_type: str = "manufacturing") -> DistillationTaskConfig:
        """行业模型场景预设"""
        return DistillationTaskConfig(
            task_name=f"{industry_type}_industry_distillation",
            description=f"针对{industry_type}行业的蒸馏配置",
            distillation_config=DistillationConfig(
                temperature=4.0,
                alpha=0.7,
                beta=0.3,
                use_feature_distillation=True,
                feature_layers=[-1, -2, -3],
            ),
            scenario_config=ScenarioDistillationConfig(
                scenario="industry",
                industry_type=industry_type,
                domain_adaptation=True,
                domain_loss_weight=0.15,
            ),
            num_epochs=15,
            batch_size=32,
        )

    @staticmethod
    def multimodal() -> DistillationTaskConfig:
        """多模态场景预设"""
        return DistillationTaskConfig(
            task_name="multimodal_distillation",
            description="多模态知识蒸馏配置",
            distillation_config=DistillationConfig(
                temperature=4.0,
                alpha=0.6,
                beta=0.4,
                use_feature_distillation=True,
                use_contrastive_distillation=True,
                contrastive_loss_weight=0.2,
            ),
            scenario_config=ScenarioDistillationConfig(
                scenario="multimodal",
                modalities=["text", "image"],
                cross_modal_distillation=True,
                modality_alignment=True,
                modality_weights={"text": 0.6, "image": 0.4},
            ),
            num_epochs=15,
            batch_size=24,
        )

    @staticmethod
    def distributed_large_scale(world_size: int = 4) -> DistillationTaskConfig:
        """大规模分布式场景预设"""
        return DistillationTaskConfig(
            task_name="distributed_distillation",
            description="大规模分布式蒸馏配置",
            distillation_config=DistillationConfig(
                temperature=4.0,
                alpha=0.7,
                beta=0.3,
            ),
            distributed_config=DistributedDistillationConfig(
                mode="data_parallel",
                world_size=world_size,
                use_amp=True,
                use_zero=True,
                zero_stage=2,
                gradient_accumulation_steps=4,
                sync_bn=True,
            ),
            num_epochs=10,
            batch_size=128,  # 分布式更大批次
        )

    @staticmethod
    def progressive_distillation(stages: int = 3) -> DistillationTaskConfig:
        """渐进式蒸馏场景预设"""
        return DistillationTaskConfig(
            task_name="progressive_distillation",
            description="渐进式多阶段蒸馏配置",
            distillation_config=DistillationConfig(
                temperature=4.0,
                alpha=0.7,
                beta=0.3,
                use_feature_distillation=True,
                layer_weight_strategy="linear",
            ),
            scenario_config=ScenarioDistillationConfig(
                scenario="progressive",
                progressive_stages=stages,
                progressive_schedule="exponential",
            ),
            adaptive_config=AdaptiveDistillationConfig(
                mode="layer",
                layer_selection_strategy="loss_based",
                curriculum_enabled=True,
            ),
            num_epochs=20,
            batch_size=32,
        )

    @staticmethod
    def self_distillation() -> DistillationTaskConfig:
        """自蒸馏场景预设"""
        return DistillationTaskConfig(
            task_name="self_distillation",
            description="无需教师模型的自蒸馏配置",
            distillation_config=DistillationConfig(
                temperature=3.0,
                alpha=0.5,
                beta=0.5,
                use_feature_distillation=True,
                feature_layers=[-1, -2],
            ),
            scenario_config=ScenarioDistillationConfig(
                scenario="self_distillation",
            ),
            num_epochs=10,
            batch_size=32,
        )

    @staticmethod
    def contrastive_distillation() -> DistillationTaskConfig:
        """对比蒸馏场景预设"""
        return DistillationTaskConfig(
            task_name="contrastive_distillation",
            description="基于对比学习的蒸馏配置",
            distillation_config=DistillationConfig(
                temperature=0.5,
                alpha=0.5,
                beta=0.3,
                use_contrastive_distillation=True,
                contrastive_loss_weight=0.3,
                contrastive_temperature=0.07,
                contrastive_projector_dim=256,
            ),
            scenario_config=ScenarioDistillationConfig(
                scenario="contrastive",
            ),
            num_epochs=15,
            batch_size=64,
        )

    @staticmethod
    def low_latency() -> DistillationTaskConfig:
        """低延迟场景预设"""
        return DistillationTaskConfig(
            task_name="low_latency_distillation",
            description="极低延迟优化的蒸馏配置",
            distillation_config=DistillationConfig(
                temperature=4.0,
                alpha=0.8,
                beta=0.2,
                use_feature_distillation=True,
                feature_layers=[-1],
                use_attention_distillation=False,
            ),
            scenario_config=ScenarioDistillationConfig(
                scenario="low_latency",
                target_latency_ms=10.0,
                target_device="gpu",
                enable_quantization=True,
                quantization_bits=8,
            ),
            num_epochs=8,
            batch_size=32,
        )

    @staticmethod
    def real_time() -> DistillationTaskConfig:
        """实时推理场景预设"""
        return DistillationTaskConfig(
            task_name="real_time_distillation",
            description="实时推理优化的蒸馏配置",
            distillation_config=DistillationConfig(
                temperature=4.0,
                alpha=0.75,
                beta=0.25,
                use_feature_distillation=True,
                feature_layers=[-1],
            ),
            scenario_config=ScenarioDistillationConfig(
                scenario="real_time",
                target_latency_ms=30.0,
                max_batch_size=1,
                streaming_mode=True,
            ),
            num_epochs=10,
            batch_size=16,
        )

    @classmethod
    def print_presets(cls) -> None:
        """打印所有预设"""
        print("\n" + "=" * 60)
        print("Available Distillation Presets")
        print("=" * 60)
        for name in cls.list_presets():
            try:
                config = cls.get(name)
                print(f"\n{name}:")
                print(f"  {config.description or 'No description'}")
                if config.scenario_config:
                    print(f"  Scenario: {config.scenario_config.scenario}")
                if config.distributed_config and config.distributed_config.is_distributed():
                    print(f"  Distributed: world_size={config.distributed_config.world_size}")
            except Exception:
                print(f"\n{name}: (error loading)")
        print("=" * 60)


# ======================== 工具函数 ========================

def create_distillation_config(
        teacher_path: str,
        student_path: str,
        scenario: str = "standard",
        **kwargs
) -> DistillationTaskConfig:
    """
    创建蒸馏配置的便捷函数
    
    Args:
        teacher_path: 教师模型路径
        student_path: 学生模型路径
        scenario: 蒸馏场景
        **kwargs: 其他配置参数
    
    Returns:
        DistillationTaskConfig 实例
    """
    # 从预设开始
    try:
        config = DistillationPresets.get(scenario)
    except ValueError:
        config = DistillationPresets.standard()

    # 设置模型路径
    if config.distillation_config:
        config.distillation_config.teacher_model_path = teacher_path
        config.distillation_config.student_model_path = student_path

    # 应用其他参数
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif config.distillation_config and hasattr(config.distillation_config, key):
            setattr(config.distillation_config, key, value)
        elif config.scenario_config and hasattr(config.scenario_config, key):
            setattr(config.scenario_config, key, value)

    return config


def validate_config(config: Union[DistillationConfig, DistillationTaskConfig, CompressionConfig]) -> Tuple[
    bool, List[str]]:
    """
    验证配置
    
    Returns:
        (is_valid, error_messages)
    """
    try:
        config.validate()
        return True, []
    except ValidationError as e:
        return False, [str(e)]
    except Exception as e:
        return False, [f"Validation error: {e}"]


def compare_configs(config1: DistillationTaskConfig, config2: DistillationTaskConfig) -> Dict[str, Any]:
    """比较两个配置"""
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()

    differences = {}
    all_keys = set(dict1.keys()) | set(dict2.keys())

    for key in all_keys:
        val1 = dict1.get(key)
        val2 = dict2.get(key)
        if val1 != val2:
            differences[key] = {'config1': val1, 'config2': val2}

    return {
        'are_equal': len(differences) == 0,
        'differences': differences,
        'total_differences': len(differences),
    }


def recommend_config(
        model_size_gb: float,
        target_device: str = "gpu",
        target_latency_ms: Optional[float] = None,
        target_accuracy: float = 0.95,
        num_gpus: int = 1,
) -> DistillationTaskConfig:
    """
    根据需求推荐配置
    
    Args:
        model_size_gb: 模型大小（GB）
        target_device: 目标设备
        target_latency_ms: 目标延迟（毫秒）
        target_accuracy: 目标精度保持率
        num_gpus: GPU数量
    
    Returns:
        推荐的配置
    """
    # 推荐场景
    scenario = DistillationScenario.recommend(
        target_device=target_device,
        target_latency_ms=target_latency_ms or 100.0,
        accuracy_priority=target_accuracy
    )

    # 获取基础配置
    config = DistillationPresets.get(scenario.value)

    # 调整分布式配置
    if num_gpus > 1:
        config.distributed_config = DistributedDistillationConfig(
            mode="data_parallel",
            world_size=num_gpus,
            use_amp=True,
        )

        # 大模型使用 FSDP 或 ZeRO
        if model_size_gb > 10:
            config.distributed_config.use_zero = True
            config.distributed_config.zero_stage = 2 if model_size_gb < 30 else 3

    # 调整场景配置
    if config.scenario_config:
        config.scenario_config.target_device = target_device
        if target_latency_ms:
            config.scenario_config.target_latency_ms = target_latency_ms
        config.scenario_config.target_accuracy = target_accuracy

    return config


def print_config_summary(config: DistillationTaskConfig) -> None:
    """打印配置摘要"""
    print("\n" + "=" * 60)
    print(f"Distillation Task: {config.task_name}")
    print("=" * 60)

    print(f"\nTask ID: {config.task_id}")
    print(f"Description: {config.description or 'N/A'}")

    if config.distillation_config:
        print(f"\n{config.distillation_config.summary()}")
        print(f"  Enabled methods: {config.distillation_config.get_enabled_methods()}")

    if config.scenario_config:
        print(f"\n{config.scenario_config.summary()}")

    if config.distributed_config and config.distributed_config.is_distributed():
        print(f"\n{config.distributed_config.summary()}")

    if config.adaptive_config and config.adaptive_config.mode != "none":
        print(f"\n{config.adaptive_config.summary()}")

    print(f"\nTraining:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Effective batch size: {config.get_effective_batch_size()}")
    print(f"  Learning rate: {config.learning_rate}")

    print(f"\nOutput: {config.output_dir}")
    print("=" * 60)

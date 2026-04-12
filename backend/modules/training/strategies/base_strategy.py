# -*- coding: utf-8 -*-
"""
训练策略基类

定义策略接口规范，支持可组合的训练策略。
基于技术方案中的Strategy Layer设计。

策略体系：
- Multimodal: 多模态训练策略
- Distillation: 知识蒸馏策略
- Scenario: 场景化训练策略
- Distributed: 分布式训练策略

生产级特性：
- 完整的监控和诊断能力
- 策略状态管理和持久化
- 性能分析和优化建议
- 与底层lib模块的集成
"""

import logging
import time
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from collections import deque
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ==================== 底层模块条件导入 ====================

# 损失层
from backend.lib.losses import (
    LossFactory,
    BaseLoss,
    LossResult as LibLossResult,
    LossMonitor as LibLossMonitor,
    create_loss_from_config,
)

# 硬件层
from backend.lib.hardware import (
    DeviceManager,
    get_device_manager,
    MemoryManager,
)


# 分布式层
from backend.lib.distributed import (
    DistributedManager,
    get_distributed_manager,
    is_main_process,
    get_rank,
    get_world_size,
)


class StrategyType(Enum):
    """策略类型枚举"""
    STANDARD = "standard"           # 标准训练
    MULTIMODAL = "multimodal"       # 多模态训练
    DISTILLATION = "distillation"   # 知识蒸馏
    SCENARIO = "scenario"           # 场景化训练
    DISTRIBUTED = "distributed"     # 分布式训练
    THREE_STAGE = "three_stage"     # 三阶段训练
    INDUSTRY = "industry"           # 行业模型训练
    PRODUCTION = "production"       # 生产级训练
    COMPOSITE = "composite"         # 组合策略
    
    @classmethod
    def from_string(cls, value: str) -> 'StrategyType':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown strategy type: {value}")
    
    @property
    def requires_lib_losses(self) -> bool:
        """是否需要 lib/losses 模块"""
        return self in [
            StrategyType.MULTIMODAL,
            StrategyType.DISTILLATION,
            StrategyType.SCENARIO,
            StrategyType.PRODUCTION,
        ]
    
    @property
    def requires_lib_distributed(self) -> bool:
        """是否需要 lib/distributed 模块"""
        return self in [
            StrategyType.DISTRIBUTED,
            StrategyType.PRODUCTION,
        ]
    
    @property
    def requires_lib_hardware(self) -> bool:
        """是否需要 lib/hardware 模块"""
        return self in [
            StrategyType.DISTRIBUTED,
            StrategyType.PRODUCTION,
        ]
    
    @property
    def is_composable(self) -> bool:
        """是否可以与其他策略组合"""
        return self not in [StrategyType.COMPOSITE]
    
    @property
    def default_priority(self) -> int:
        """获取默认优先级（数值越小优先级越高）"""
        priorities = {
            StrategyType.DISTRIBUTED: 10,
            StrategyType.PRODUCTION: 20,
            StrategyType.MULTIMODAL: 30,
            StrategyType.DISTILLATION: 40,
            StrategyType.SCENARIO: 50,
            StrategyType.THREE_STAGE: 60,
            StrategyType.INDUSTRY: 70,
            StrategyType.STANDARD: 100,
            StrategyType.COMPOSITE: 0,
        }
        return priorities.get(self, 100)
    
    def get_description(self) -> str:
        """获取策略描述"""
        descriptions = {
            StrategyType.STANDARD: "标准监督学习训练策略",
            StrategyType.MULTIMODAL: "多模态训练策略，支持文本、图像、音频等",
            StrategyType.DISTILLATION: "知识蒸馏策略，支持教师-学生模型",
            StrategyType.SCENARIO: "场景化训练策略，针对特定业务场景优化",
            StrategyType.DISTRIBUTED: "分布式训练策略，支持多GPU/多节点",
            StrategyType.THREE_STAGE: "三阶段训练策略：预训练-微调-偏好对齐",
            StrategyType.INDUSTRY: "行业模型训练策略，面向垂直领域",
            StrategyType.PRODUCTION: "生产级训练策略，整合六层架构",
            StrategyType.COMPOSITE: "组合策略，支持多策略协同",
        }
        return descriptions.get(self, "未知策略")


class TrainingPhase(Enum):
    """训练阶段枚举"""
    # 行业模型三阶段
    PRETRAIN_INDUSTRY = "pretrain_industry"     # 行业表征预训练
    ALIGN_INDUSTRY = "align_industry"           # 行业能力对齐
    FINETUNE_SCENE = "finetune_scene"           # 场景精调
    
    # 通用阶段
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    PREFERENCE = "preference"
    
    # 扩展阶段
    WARMUP = "warmup"               # 预热阶段
    MAIN = "main"                   # 主训练阶段
    COOLDOWN = "cooldown"           # 冷却阶段
    EVALUATION = "evaluation"       # 评估阶段
    
    @classmethod
    def from_string(cls, value: str) -> 'TrainingPhase':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown training phase: {value}")
    
    @property
    def is_training(self) -> bool:
        """是否为训练阶段"""
        return self not in [TrainingPhase.EVALUATION]
    
    @property
    def typical_epochs(self) -> int:
        """典型训练轮数"""
        epochs = {
            TrainingPhase.PRETRAIN_INDUSTRY: 10,
            TrainingPhase.ALIGN_INDUSTRY: 5,
            TrainingPhase.FINETUNE_SCENE: 3,
            TrainingPhase.PRETRAIN: 10,
            TrainingPhase.FINETUNE: 3,
            TrainingPhase.PREFERENCE: 1,
            TrainingPhase.WARMUP: 1,
            TrainingPhase.MAIN: 5,
            TrainingPhase.COOLDOWN: 1,
            TrainingPhase.EVALUATION: 1,
        }
        return epochs.get(self, 3)
    
    @property
    def typical_lr_scale(self) -> float:
        """典型学习率倍数"""
        scales = {
            TrainingPhase.PRETRAIN_INDUSTRY: 1.0,
            TrainingPhase.ALIGN_INDUSTRY: 0.5,
            TrainingPhase.FINETUNE_SCENE: 0.1,
            TrainingPhase.PRETRAIN: 1.0,
            TrainingPhase.FINETUNE: 0.1,
            TrainingPhase.PREFERENCE: 0.01,
            TrainingPhase.WARMUP: 0.1,
            TrainingPhase.MAIN: 1.0,
            TrainingPhase.COOLDOWN: 0.1,
            TrainingPhase.EVALUATION: 0.0,
        }
        return scales.get(self, 1.0)
    
    def next_phase(self) -> Optional['TrainingPhase']:
        """获取下一个阶段"""
        phase_order = [
            TrainingPhase.WARMUP,
            TrainingPhase.PRETRAIN,
            TrainingPhase.FINETUNE,
            TrainingPhase.PREFERENCE,
            TrainingPhase.COOLDOWN,
            TrainingPhase.EVALUATION,
        ]
        try:
            idx = phase_order.index(self)
            if idx < len(phase_order) - 1:
                return phase_order[idx + 1]
        except ValueError:
            pass
        return None
    
    def get_description(self) -> str:
        """获取阶段描述"""
        descriptions = {
            TrainingPhase.PRETRAIN_INDUSTRY: "行业表征预训练：学习领域知识",
            TrainingPhase.ALIGN_INDUSTRY: "行业能力对齐：跨模态/任务对齐",
            TrainingPhase.FINETUNE_SCENE: "场景精调：针对具体场景优化",
            TrainingPhase.PRETRAIN: "预训练：学习通用表征",
            TrainingPhase.FINETUNE: "微调：针对下游任务优化",
            TrainingPhase.PREFERENCE: "偏好对齐：RLHF/DPO等",
            TrainingPhase.WARMUP: "预热阶段：逐步增加学习率",
            TrainingPhase.MAIN: "主训练阶段：核心训练过程",
            TrainingPhase.COOLDOWN: "冷却阶段：逐步降低学习率",
            TrainingPhase.EVALUATION: "评估阶段：验证模型性能",
        }
        return descriptions.get(self, "未知阶段")


@dataclass
class StrategyContext:
    """策略上下文
    
    在训练过程中传递的上下文信息，包含模型、优化器、数据等。
    支持状态持久化和恢复。
    """
    model: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[Any] = None
    dataloader: Optional[Any] = None
    device: torch.device = field(default_factory=lambda: torch.device('cpu'))
    
    # 训练状态
    epoch: int = 0
    global_step: int = 0
    local_step: int = 0  # 当前epoch内的步数
    phase: Optional[TrainingPhase] = None
    
    # 配置
    config: Dict[str, Any] = field(default_factory=dict)
    
    # 中间结果
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # 额外数据
    extra: Dict[str, Any] = field(default_factory=dict)
    
    # 扩展字段
    batch_size: int = 0
    total_samples: int = 0
    samples_seen: int = 0
    max_epochs: int = 0
    max_steps: int = 0
    learning_rate: float = 0.0
    gradient_accumulation_steps: int = 1
    
    # 分布式信息
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    is_main_process: bool = True
    
    # 时间戳
    start_time: float = 0.0
    epoch_start_time: float = 0.0
    step_start_time: float = 0.0
    
    def update_step(self) -> None:
        """更新步数"""
        self.global_step += 1
        self.local_step += 1
        self.step_start_time = time.time()
    
    def update_epoch(self) -> None:
        """更新轮数"""
        self.epoch += 1
        self.local_step = 0
        self.epoch_start_time = time.time()
    
    def update_samples(self, batch_size: int) -> None:
        """更新样本计数"""
        self.samples_seen += batch_size
        self.batch_size = batch_size
    
    def get_progress(self) -> float:
        """获取训练进度 (0.0 - 1.0)"""
        if self.max_steps > 0:
            return min(self.global_step / self.max_steps, 1.0)
        elif self.max_epochs > 0 and self.total_samples > 0:
            epoch_progress = self.epoch / self.max_epochs
            step_progress = self.samples_seen / (self.total_samples * self.max_epochs)
            return min(step_progress, 1.0)
        return 0.0
    
    def get_elapsed_time(self) -> float:
        """获取已用时间（秒）"""
        if self.start_time > 0:
            return time.time() - self.start_time
        return 0.0
    
    def get_eta(self) -> float:
        """估计剩余时间（秒）"""
        elapsed = self.get_elapsed_time()
        progress = self.get_progress()
        if progress > 0 and elapsed > 0:
            return elapsed * (1.0 - progress) / progress
        return 0.0
    
    def is_distributed(self) -> bool:
        """是否为分布式训练"""
        return self.world_size > 1
    
    def should_log(self, log_interval: int = 100) -> bool:
        """是否应该记录日志"""
        return self.is_main_process and self.global_step % log_interval == 0
    
    def should_save(self, save_interval: int = 1000) -> bool:
        """是否应该保存检查点"""
        return self.is_main_process and self.global_step % save_interval == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'local_step': self.local_step,
            'phase': self.phase.value if self.phase else None,
            'batch_size': self.batch_size,
            'samples_seen': self.samples_seen,
            'learning_rate': self.learning_rate,
            'world_size': self.world_size,
            'rank': self.rank,
            'metrics': self.metrics.copy(),
            'config': self.config.copy(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyContext':
        """从字典创建"""
        ctx = cls()
        ctx.epoch = data.get('epoch', 0)
        ctx.global_step = data.get('global_step', 0)
        ctx.local_step = data.get('local_step', 0)
        phase_value = data.get('phase')
        ctx.phase = TrainingPhase.from_string(phase_value) if phase_value else None
        ctx.batch_size = data.get('batch_size', 0)
        ctx.samples_seen = data.get('samples_seen', 0)
        ctx.learning_rate = data.get('learning_rate', 0.0)
        ctx.world_size = data.get('world_size', 1)
        ctx.rank = data.get('rank', 0)
        ctx.metrics = data.get('metrics', {})
        ctx.config = data.get('config', {})
        return ctx
    
    def clone(self) -> 'StrategyContext':
        """克隆上下文"""
        return StrategyContext.from_dict(self.to_dict())
    
    def summary(self) -> str:
        """获取摘要"""
        return (
            f"StrategyContext(epoch={self.epoch}, step={self.global_step}, "
            f"phase={self.phase.value if self.phase else 'None'}, "
            f"progress={self.get_progress():.2%}, "
            f"samples_seen={self.samples_seen})"
        )
    
    def __repr__(self) -> str:
        return self.summary()


@dataclass
class StrategyResult:
    """策略执行结果
    
    包含损失、指标、输出和控制信号。
    """
    loss: torch.Tensor = None
    metrics: Dict[str, float] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    should_stop: bool = False
    message: str = ""

    # 扩展字段
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    step_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    # 损失组件
    loss_components: Dict[str, float] = field(default_factory=dict)
    
    # 警告和错误
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def get_loss_value(self) -> float:
        """获取损失值"""
        if self.loss is None:
            return 0.0
        if isinstance(self.loss, torch.Tensor):
            return self.loss.item()
        return float(self.loss)
    
    def add_metric(self, name: str, value: float) -> None:
        """添加指标"""
        self.metrics[name] = value
    
    def add_loss_component(self, name: str, value: float) -> None:
        """添加损失组件"""
        self.loss_components[name] = value
    
    def add_warning(self, message: str) -> None:
        """添加警告"""
        self.warnings.append(message)
    
    def add_error(self, message: str) -> None:
        """添加错误"""
        self.errors.append(message)
    
    def has_warnings(self) -> bool:
        """是否有警告"""
        return len(self.warnings) > 0
    
    def has_errors(self) -> bool:
        """是否有错误"""
        return len(self.errors) > 0
    
    def is_valid(self) -> bool:
        """结果是否有效"""
        if self.loss is None:
            return False
        if isinstance(self.loss, torch.Tensor):
            if torch.isnan(self.loss) or torch.isinf(self.loss):
                return False
        return not self.has_errors()
    
    def merge(self, other: 'StrategyResult', weight: float = 1.0) -> 'StrategyResult':
        """合并两个结果"""
        merged = StrategyResult()
        
        # 合并损失
        if self.loss is not None and other.loss is not None:
            merged.loss = self.loss + other.loss * weight
        elif self.loss is not None:
            merged.loss = self.loss
        elif other.loss is not None:
            merged.loss = other.loss * weight
        
        # 合并指标
        merged.metrics = {**self.metrics}
        for key, value in other.metrics.items():
            merged.metrics[f"other_{key}"] = value
        
        # 合并损失组件
        merged.loss_components = {**self.loss_components, **other.loss_components}
        
        # 合并警告和错误
        merged.warnings = self.warnings + other.warnings
        merged.errors = self.errors + other.errors
        
        # 合并输出
        merged.outputs = {**self.outputs, **other.outputs}
        
        # 控制信号
        merged.should_stop = self.should_stop or other.should_stop
        
        return merged
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'loss': self.get_loss_value(),
            'metrics': self.metrics.copy(),
            'loss_components': self.loss_components.copy(),
            'grad_norm': self.grad_norm,
            'learning_rate': self.learning_rate,
            'step_time': self.step_time,
            'should_stop': self.should_stop,
            'message': self.message,
            'warnings': self.warnings.copy(),
            'errors': self.errors.copy(),
        }
    
    def summary(self) -> str:
        """获取摘要"""
        loss_str = f"{self.get_loss_value():.4f}" if self.loss is not None else "None"
        return (
            f"StrategyResult(loss={loss_str}, "
            f"metrics={len(self.metrics)}, "
            f"valid={self.is_valid()}, "
            f"stop={self.should_stop})"
        )
    
    def __repr__(self) -> str:
        return self.summary()


# ==================== 监控和诊断组件 ====================

@dataclass
class StrategyMetrics:
    """策略指标数据类"""
    total_steps: int = 0
    total_epochs: int = 0
    total_samples: int = 0
    total_loss: float = 0.0
    avg_loss: float = 0.0
    min_loss: float = float('inf')
    max_loss: float = float('-inf')
    avg_step_time: float = 0.0
    total_time: float = 0.0
    
    # 各阶段指标
    phase_steps: Dict[str, int] = field(default_factory=dict)
    phase_losses: Dict[str, float] = field(default_factory=dict)
    
    # 梯度信息
    avg_grad_norm: float = 0.0
    grad_norm_history: List[float] = field(default_factory=list)
    
    def update(self, result: StrategyResult, step_time: float = 0.0) -> None:
        """更新指标"""
        self.total_steps += 1
        loss_value = result.get_loss_value()
        
        self.total_loss += loss_value
        self.avg_loss = self.total_loss / self.total_steps
        self.min_loss = min(self.min_loss, loss_value)
        self.max_loss = max(self.max_loss, loss_value)
        
        if step_time > 0:
            self.total_time += step_time
            self.avg_step_time = self.total_time / self.total_steps
        
        if result.grad_norm > 0:
            self.grad_norm_history.append(result.grad_norm)
            self.avg_grad_norm = sum(self.grad_norm_history) / len(self.grad_norm_history)
    
    def update_phase(self, phase: TrainingPhase, loss: float) -> None:
        """更新阶段指标"""
        phase_name = phase.value
        self.phase_steps[phase_name] = self.phase_steps.get(phase_name, 0) + 1
        
        # 计算阶段平均损失
        old_avg = self.phase_losses.get(phase_name, 0.0)
        old_steps = self.phase_steps[phase_name] - 1
        if old_steps > 0:
            self.phase_losses[phase_name] = (old_avg * old_steps + loss) / self.phase_steps[phase_name]
        else:
            self.phase_losses[phase_name] = loss
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_steps': self.total_steps,
            'total_epochs': self.total_epochs,
            'total_samples': self.total_samples,
            'avg_loss': self.avg_loss,
            'min_loss': self.min_loss if self.min_loss != float('inf') else None,
            'max_loss': self.max_loss if self.max_loss != float('-inf') else None,
            'avg_step_time': self.avg_step_time,
            'total_time': self.total_time,
            'avg_grad_norm': self.avg_grad_norm,
            'phase_steps': self.phase_steps.copy(),
            'phase_losses': self.phase_losses.copy(),
        }


class StrategyMonitor:
    """策略监控器
    
    记录和分析策略执行的各项指标。
    """
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        self.metrics = StrategyMetrics()
        
        # 历史记录
        self._loss_history: deque = deque(maxlen=history_size)
        self._step_time_history: deque = deque(maxlen=history_size)
        self._result_history: deque = deque(maxlen=1000)
        
        # 阶段记录
        self._phase_history: List[Tuple[TrainingPhase, int, float]] = []
        
        # 早停检测
        self._best_loss = float('inf')
        self._steps_without_improvement = 0
        
        # 集成 LibLossMonitor
        self._lib_monitor = None
        try:
            if 'LibLossMonitor' in globals() and LibLossMonitor:
                self._lib_monitor = LibLossMonitor(max_history=100)
        except Exception:
            pass
    
    def record_step(
        self, 
        result: StrategyResult, 
        context: StrategyContext,
        step_time: float = 0.0
    ) -> None:
        """记录一个训练步骤"""
        loss_value = result.get_loss_value()
        
        # 更新历史
        self._loss_history.append(loss_value)
        if step_time > 0:
            self._step_time_history.append(step_time)
        
        # 更新指标
        self.metrics.update(result, step_time)
        
        # 更新 LibLossMonitor
        if self._lib_monitor:
            # LibLossMonitor (LossMonitor) 使用 record 方法，接受 LossResult 对象
            # 这里的 loss_value 是 float，需要包装或适配
            try:
                if hasattr(self._lib_monitor, 'update'):
                    self._lib_monitor.update(loss_value)
                elif hasattr(self._lib_monitor, 'record'):
                    # 尝试构造简单的 LossResult 或类似结构，如果 LibLossMonitor 是 LossMonitor
                    pass 
            except Exception:
                pass
        
        # 更新阶段指标
        if context.phase:
            self.metrics.update_phase(context.phase, loss_value)
        
        # 早停检测
        if loss_value < self._best_loss:
            self._best_loss = loss_value
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1
        
        # 记录结果
        self._result_history.append({
            'step': context.global_step,
            'loss': loss_value,
            'metrics': result.metrics.copy(),
            'timestamp': result.timestamp,
        })
    
    def record_epoch(self, epoch: int, phase: TrainingPhase, metrics: Dict[str, float]) -> None:
        """记录一个epoch"""
        self.metrics.total_epochs += 1
        loss = metrics.get('loss', 0.0)
        self._phase_history.append((phase, epoch, loss))
    
    def get_loss_trend(self, window: int = 100) -> str:
        """获取损失趋势"""
        if len(self._loss_history) < window * 2:
            return "insufficient_data"
        
        recent = list(self._loss_history)[-window:]
        previous = list(self._loss_history)[-window * 2:-window]
        
        recent_avg = sum(recent) / len(recent)
        previous_avg = sum(previous) / len(previous)
        
        change = (recent_avg - previous_avg) / max(previous_avg, 1e-8)
        
        if change < -0.05:
            return "improving"
        elif change > 0.05:
            return "degrading"
        else:
            return "stable"
    
    def check_early_stopping(self, patience: int = 10) -> bool:
        """检查是否应该早停"""
        return self._steps_without_improvement >= patience
    
    def get_recent_losses(self, n: int = 100) -> List[float]:
        """获取最近的损失值"""
        return list(self._loss_history)[-n:]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        return {
            **self.metrics.to_dict(),
            'loss_trend': self.get_loss_trend(),
            'best_loss': self._best_loss,
            'steps_without_improvement': self._steps_without_improvement,
        }
    
    def reset(self) -> None:
        """重置监控器"""
        self.metrics = StrategyMetrics()
        self._loss_history.clear()
        self._step_time_history.clear()
        self._result_history.clear()
        self._phase_history.clear()
        self._best_loss = float('inf')
        self._steps_without_improvement = 0


class StrategyProfiler:
    """策略性能分析器
    
    分析策略执行的性能特征。
    """
    
    def __init__(self):
        self._enabled = False
        self._profiles: Dict[str, List[float]] = {}
        self._current_profile: Optional[str] = None
        self._profile_start: float = 0.0
    
    def enable(self) -> None:
        """启用分析"""
        self._enabled = True
    
    def disable(self) -> None:
        """禁用分析"""
        self._enabled = False
    
    @contextmanager
    def profile(self, name: str):
        """性能分析上下文"""
        if not self._enabled:
            yield
            return
        
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            if name not in self._profiles:
                self._profiles[name] = []
            self._profiles[name].append(elapsed)
    
    def record(self, name: str, elapsed: float) -> None:
        """记录一次执行时间"""
        if not self._enabled:
            return
        if name not in self._profiles:
            self._profiles[name] = []
        self._profiles[name].append(elapsed)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """获取统计信息"""
        stats = {}
        for name, times in self._profiles.items():
            if times:
                stats[name] = {
                    'count': len(times),
                    'total': sum(times),
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                }
        return stats
    
    def print_stats(self) -> None:
        """打印统计信息"""
        stats = self.get_stats()
        if not stats:
            print("No profiling data")
            return
        
        print("\n" + "="*60)
        print("Strategy Profiler Statistics")
        print("="*60)
        for name, data in sorted(stats.items(), key=lambda x: -x[1]['total']):
            print(f"\n{name}:")
            print(f"  Count: {data['count']}")
            print(f"  Total: {data['total']:.4f}s")
            print(f"  Avg: {data['avg']*1000:.2f}ms")
            print(f"  Min: {data['min']*1000:.2f}ms")
            print(f"  Max: {data['max']*1000:.2f}ms")
    
    def reset(self) -> None:
        """重置分析器"""
        self._profiles.clear()


class StrategyValidator:
    """策略验证器
    
    验证策略执行结果的有效性。
    """
    
    def __init__(self):
        self._checks: List[Callable[[StrategyResult], Tuple[bool, str]]] = []
        self._add_default_checks()
    
    def _add_default_checks(self) -> None:
        """添加默认检查"""
        # NaN检查
        self._checks.append(self._check_nan)
        # Inf检查
        self._checks.append(self._check_inf)
        # 负损失检查
        self._checks.append(self._check_negative)
        # 梯度爆炸检查
        self._checks.append(self._check_gradient_explosion)
    
    def _check_nan(self, result: StrategyResult) -> Tuple[bool, str]:
        """检查NaN"""
        if result.loss is not None and isinstance(result.loss, torch.Tensor):
            if torch.isnan(result.loss):
                return False, "Loss is NaN"
        return True, ""
    
    def _check_inf(self, result: StrategyResult) -> Tuple[bool, str]:
        """检查Inf"""
        if result.loss is not None and isinstance(result.loss, torch.Tensor):
            if torch.isinf(result.loss):
                return False, "Loss is Inf"
        return True, ""
    
    def _check_negative(self, result: StrategyResult) -> Tuple[bool, str]:
        """检查负损失"""
        loss_value = result.get_loss_value()
        if loss_value < 0:
            return False, f"Negative loss: {loss_value}"
        return True, ""
    
    def _check_gradient_explosion(self, result: StrategyResult) -> Tuple[bool, str]:
        """检查梯度爆炸"""
        if result.grad_norm > 1000:
            return False, f"Gradient explosion: norm={result.grad_norm}"
        return True, ""
    
    def add_check(self, check: Callable[[StrategyResult], Tuple[bool, str]]) -> None:
        """添加自定义检查"""
        self._checks.append(check)
    
    def validate(self, result: StrategyResult) -> Tuple[bool, List[str]]:
        """验证结果"""
        errors = []
        for check in self._checks:
            valid, message = check(result)
            if not valid:
                errors.append(message)
                result.add_error(message)
        return len(errors) == 0, errors
    
    def get_check_count(self) -> int:
        """获取检查数量"""
        return len(self._checks)


# ==================== 策略基类 ====================

class TrainingStrategy(ABC):
    """
    训练策略基类
    
    定义策略接口规范，所有具体策略需实现这些方法。
    策略可以组合使用，通过Trainer统一调用。
    
    生产级特性：
    - 完整的监控和诊断能力
    - 策略状态管理和持久化
    - 性能分析和优化建议
    - 与底层lib模块的集成
    
    使用示例:
    ```python
    class MyStrategy(TrainingStrategy):
        def compute_loss(self, model, batch, outputs, context):
            return outputs['loss']
    
    trainer = Trainer(model, [strategy1, strategy2], optimizer)
    trainer.train_step(batch)
    ```
    """
    
    def __init__(
        self, 
        name: str = "base", 
        priority: int = 100,
        strategy_type: Optional[StrategyType] = None,
        enable_monitoring: bool = True,
        enable_profiling: bool = False,
        enable_validation: bool = True
    ):
        """
        初始化策略
        
        Args:
            name: 策略名称
            priority: 优先级（数值越小优先级越高）
            strategy_type: 策略类型
            enable_monitoring: 是否启用监控
            enable_profiling: 是否启用性能分析
            enable_validation: 是否启用结果验证
        """
        self.name = name
        self.priority = priority
        self.strategy_type = strategy_type or StrategyType.STANDARD
        self.is_enabled = True
        self._initialized = False
        
        # 监控和诊断组件
        self._monitor: Optional[StrategyMonitor] = None
        self._profiler: Optional[StrategyProfiler] = None
        self._validator: Optional[StrategyValidator] = None
        
        self._enable_monitoring = enable_monitoring
        self._enable_profiling = enable_profiling
        self._enable_validation = enable_validation
        
        # 状态
        self._setup_time: float = 0.0
        self._total_compute_time: float = 0.0
        self._compute_count: int = 0
        
        # 资源管理器
        self._memory_manager: Optional[MemoryManager] = None
        self._distributed_manager: Optional[DistributedManager] = None
        
        # 回调
        self._callbacks: List[Callable] = []
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self) -> None:
        """初始化监控和诊断组件"""
        if self._enable_monitoring:
            self._monitor = StrategyMonitor()
        
        if self._enable_profiling:
            self._profiler = StrategyProfiler()
            self._profiler.enable()
        
        if self._enable_validation:
            self._validator = StrategyValidator()
            
        # 初始化内存管理器
        try:
            self._memory_manager = MemoryManager()
        except Exception:
            pass
    
    def setup(self, context: StrategyContext) -> None:
        """
        策略初始化
        
        在训练开始前调用，用于初始化策略所需的资源。
        
        Args:
            context: 策略上下文
        """
        setup_start = time.time()
        
        # 初始化分布式信息
        try:
            self._distributed_manager = get_distributed_manager()
            context.world_size = get_world_size()
            context.rank = get_rank()
            context.is_main_process = is_main_process()
        except Exception:
            pass
        
        # 初始化时间戳
        context.start_time = time.time()
        
        self._initialized = True
        self._setup_time = time.time() - setup_start
        
        logger.info(f"Strategy '{self.name}' setup completed in {self._setup_time:.4f}s")
    
    def prepare_batch(self, batch: Dict[str, Any], context: StrategyContext) -> Dict[str, Any]:
        """
        准备批次数据
        
        在前向传播前对批次数据进行预处理。
        
        Args:
            batch: 输入批次数据
            context: 策略上下文
        
        Returns:
            处理后的批次数据
        """
        # 性能分析
        if self._profiler:
            with self._profiler.profile('prepare_batch'):
                return self._prepare_batch_impl(batch, context)
        return self._prepare_batch_impl(batch, context)
    
    def _prepare_batch_impl(self, batch: Dict[str, Any], context: StrategyContext) -> Dict[str, Any]:
        """批次准备的实际实现"""
        # 移动到设备
        device = context.device
        prepared = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(device)
            elif isinstance(value, dict):
                prepared[key] = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                prepared[key] = value
        return prepared
    
    def check_memory(self) -> Dict[str, float]:
        """
        检查内存使用情况
        
        Returns:
            内存统计信息
        """
        if self._memory_manager:
            stats = self._memory_manager.get_stats()
            if hasattr(stats, 'to_dict'):
                return stats.to_dict()
            return {
                'allocated': stats.allocated_gb if hasattr(stats, 'allocated_gb') else 0.0,
                'reserved': stats.reserved_gb if hasattr(stats, 'reserved_gb') else 0.0
            }
        return {}
    
    def clear_memory(self) -> None:
        """清理内存"""
        if self._memory_manager:
            if hasattr(self._memory_manager, 'clear_cache'):
                self._memory_manager.clear_cache()
            elif hasattr(self._memory_manager, 'clear_memory'):
                self._memory_manager.clear_memory()
            
    def sync_processes(self) -> None:
        """同步所有进程（分布式训练）"""
        if self._distributed_manager:
            self._distributed_manager.barrier()

    def check_device_health(self) -> Dict[str, Any]:
        """
        检查设备健康状态
        
        使用 DeviceManager 获取设备状态信息。
        
        Returns:
            设备健康状态字典
        """
        status = {'status': 'unknown'}
        try:
            manager = get_device_manager()
            if isinstance(manager, DeviceManager):
                # 尝试获取设备信息
                device = manager.get_device()
                status['device'] = str(device)
                status['status'] = 'healthy'
                
                # 如果有更多健康检查方法可以在此调用
                if hasattr(manager, 'check_health'):
                    health = manager.check_health()
                    status.update(health)
        except Exception as e:
            status['status'] = 'error'
            status['error'] = str(e)
            
        return status
    
    @abstractmethod
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """
        计算损失
        
        核心方法，计算训练损失。每个具体策略必须实现此方法。
        
        Args:
            model: 模型
            batch: 批次数据
            outputs: 模型输出
            context: 策略上下文
        
        Returns:
            策略结果，包含损失和指标
        """
        raise NotImplementedError
    
    def compute_loss_with_monitoring(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """
        带监控的损失计算
        
        包装compute_loss方法，添加监控、验证和性能分析。
        """
        step_start = time.time()
        
        # 性能分析
        if self._profiler:
            with self._profiler.profile('compute_loss'):
                result = self.compute_loss(model, batch, outputs, context)
        else:
            result = self.compute_loss(model, batch, outputs, context)
        
        step_time = time.time() - step_start
        result.step_time = step_time
        self._total_compute_time += step_time
        self._compute_count += 1
        
        # 验证结果
        if self._validator:
            valid, errors = self._validator.validate(result)
            if not valid:
                logger.warning(f"Strategy '{self.name}' result validation failed: {errors}")
        
        # 记录监控
        if self._monitor:
            self._monitor.record_step(result, context, step_time)
        
        return result
    
    def on_step_start(self, context: StrategyContext) -> None:
        """训练步骤开始时的回调"""
        context.step_start_time = time.time()
        
        # 调用注册的回调
        for callback in self._callbacks:
            if hasattr(callback, 'on_step_start'):
                callback.on_step_start(context)
    
    def on_step_end(self, context: StrategyContext, result: StrategyResult) -> None:
        """训练步骤结束时的回调"""
        # 调用注册的回调
        for callback in self._callbacks:
            if hasattr(callback, 'on_step_end'):
                callback.on_step_end(context, result)
    
    def on_epoch_start(self, context: StrategyContext) -> None:
        """Epoch开始时的回调"""
        context.epoch_start_time = time.time()
        context.local_step = 0
        
        for callback in self._callbacks:
            if hasattr(callback, 'on_epoch_start'):
                callback.on_epoch_start(context)
    
    def on_epoch_end(self, context: StrategyContext) -> None:
        """Epoch结束时的回调"""
        if self._monitor:
            self._monitor.metrics.total_epochs += 1
        
        for callback in self._callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(context)
    
    def on_phase_start(self, phase: TrainingPhase, context: StrategyContext) -> None:
        """训练阶段开始时的回调"""
        context.phase = phase
        logger.info(f"Strategy '{self.name}' starting phase: {phase.value}")
        
        for callback in self._callbacks:
            if hasattr(callback, 'on_phase_start'):
                callback.on_phase_start(phase, context)
    
    def on_phase_end(self, phase: TrainingPhase, context: StrategyContext) -> None:
        """训练阶段结束时的回调"""
        logger.info(f"Strategy '{self.name}' completed phase: {phase.value}")
        
        for callback in self._callbacks:
            if hasattr(callback, 'on_phase_end'):
                callback.on_phase_end(phase, context)
    
    def on_training_start(self, context: StrategyContext) -> None:
        """训练开始时的回调"""
        context.start_time = time.time()
        logger.info(f"Strategy '{self.name}' training started")
    
    def on_training_end(self, context: StrategyContext) -> None:
        """训练结束时的回调"""
        elapsed = context.get_elapsed_time()
        logger.info(f"Strategy '{self.name}' training completed in {elapsed:.2f}s")
    
    def add_callback(self, callback: Callable) -> None:
        """添加回调"""
        self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> bool:
        """移除回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
            return True
        return False
    
    def cleanup(self) -> None:
        """清理资源"""
        self._initialized = False
        
        # 重置监控
        if self._monitor:
            self._monitor.reset()
        
        # 重置分析器
        if self._profiler:
            self._profiler.reset()
        
        logger.debug(f"Strategy '{self.name}' cleaned up")
    
    # ==================== 监控和诊断方法 ====================
    
    def get_monitor(self) -> Optional[StrategyMonitor]:
        """获取监控器"""
        return self._monitor
    
    def get_profiler(self) -> Optional[StrategyProfiler]:
        """获取性能分析器"""
        return self._profiler
    
    def get_validator(self) -> Optional[StrategyValidator]:
        """获取验证器"""
        return self._validator
    
    def enable_monitoring(self) -> None:
        """启用监控"""
        if self._monitor is None:
            self._monitor = StrategyMonitor()
        self._enable_monitoring = True
    
    def disable_monitoring(self) -> None:
        """禁用监控"""
        self._enable_monitoring = False
    
    def enable_profiling(self) -> None:
        """启用性能分析"""
        if self._profiler is None:
            self._profiler = StrategyProfiler()
        self._profiler.enable()
        self._enable_profiling = True
    
    def disable_profiling(self) -> None:
        """禁用性能分析"""
        if self._profiler:
            self._profiler.disable()
        self._enable_profiling = False
    
    def get_monitor_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        if self._monitor:
            return self._monitor.get_summary()
        return {}
    
    def get_profiling_stats(self) -> Dict[str, Dict[str, float]]:
        """获取性能分析统计"""
        if self._profiler:
            return self._profiler.get_stats()
        return {}
    
    def print_profiling_stats(self) -> None:
        """打印性能分析统计"""
        if self._profiler:
            self._profiler.print_stats()
    
    def check_early_stopping(self, patience: int = 10) -> bool:
        """检查是否应该早停"""
        if self._monitor:
            return self._monitor.check_early_stopping(patience)
        return False
    
    def get_loss_trend(self) -> str:
        """获取损失趋势"""
        if self._monitor:
            return self._monitor.get_loss_trend()
        return "unknown"
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断策略状态"""
        diagnosis = {
            'status': 'healthy',
            'issues': [],
            'recommendations': [],
            'info': self.get_info(),
        }
        
        # 检查初始化状态
        if not self._initialized:
            diagnosis['issues'].append("Strategy not initialized")
            diagnosis['recommendations'].append("Call setup() before training")
        
        # 检查监控
        if self._monitor:
            summary = self._monitor.get_summary()
            
            # 检查损失趋势
            trend = summary.get('loss_trend', 'unknown')
            if trend == 'degrading':
                diagnosis['issues'].append("Loss is degrading")
                diagnosis['recommendations'].append("Consider reducing learning rate")
            
            # 检查早停
            if summary.get('steps_without_improvement', 0) > 1000:
                diagnosis['issues'].append("No improvement for many steps")
                diagnosis['recommendations'].append("Consider early stopping")
        
        # 检查性能
        if self._compute_count > 0:
            avg_time = self._total_compute_time / self._compute_count
            if avg_time > 1.0:
                diagnosis['issues'].append(f"Slow compute time: {avg_time:.2f}s/step")
                diagnosis['recommendations'].append("Consider optimizing model or batch size")
        
        # 设置状态
        if len(diagnosis['issues']) > 3:
            diagnosis['status'] = 'error'
        elif len(diagnosis['issues']) > 0:
            diagnosis['status'] = 'warning'
        
        return diagnosis
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diagnosis = self.diagnose()
        
        print("\n" + "="*60)
        print(f"Strategy Diagnosis: {self.name}")
        print("="*60)
        print(f"Status: {diagnosis['status'].upper()}")
        
        print("\n--- Info ---")
        for key, value in diagnosis['info'].items():
            print(f"  {key}: {value}")
        
        if diagnosis['issues']:
            print("\n--- Issues ---")
            for issue in diagnosis['issues']:
                print(f"  ⚠️ {issue}")
        
        if diagnosis['recommendations']:
            print("\n--- Recommendations ---")
            for rec in diagnosis['recommendations']:
                print(f"  💡 {rec}")
        
        print("="*60)
    
    # ==================== 状态管理方法 ====================
    
    def get_config(self) -> Dict[str, Any]:
        """获取策略配置"""
        return {
            'name': self.name,
            'priority': self.priority,
            'strategy_type': self.strategy_type.value,
            'is_enabled': self.is_enabled,
            'enable_monitoring': self._enable_monitoring,
            'enable_profiling': self._enable_profiling,
            'enable_validation': self._enable_validation,
        }
    
    def get_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        info = {
            'name': self.name,
            'type': self.strategy_type.value,
            'priority': self.priority,
            'initialized': self._initialized,
            'enabled': self.is_enabled,
            'setup_time': self._setup_time,
            'compute_count': self._compute_count,
            'avg_compute_time': (
                self._total_compute_time / self._compute_count 
                if self._compute_count > 0 else 0.0
            )
        }
        return info
    
    def get_state_dict(self) -> Dict[str, Any]:
        """获取策略状态字典（用于保存）"""
        state = {
            'name': self.name,
            'config': self.get_config(),
            'compute_count': self._compute_count,
            'total_compute_time': self._total_compute_time,
        }
        
        if self._monitor:
            state['monitor'] = self._monitor.get_summary()
        
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """加载策略状态字典"""
        self._compute_count = state.get('compute_count', 0)
        self._total_compute_time = state.get('total_compute_time', 0.0)
    
    def reset_stats(self) -> None:
        """重置统计数据"""
        self._compute_count = 0
        self._total_compute_time = 0.0
        
        if self._monitor:
            self._monitor.reset()
        
        if self._profiler:
            self._profiler.reset()
    
    def summary(self) -> str:
        """获取策略摘要"""
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"type={self.strategy_type.value}, "
            f"priority={self.priority}, "
            f"initialized={self._initialized})"
        )
    
    def print_summary(self) -> None:
        """打印策略摘要"""
        info = self.get_info()
        
        print("\n" + "="*60)
        print(f"Strategy Summary: {self.name}")
        print("="*60)
        
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        
        if self._monitor:
            print("\n--- Monitor Summary ---")
            summary = self._monitor.get_summary()
            for key, value in summary.items():
                if not isinstance(value, dict):
                    print(f"  {key}: {value}")
        
        print("="*60)
    
    def __repr__(self) -> str:
        return self.summary()


class StandardTrainingStrategy(TrainingStrategy):
    """
    标准训练策略
    
    实现基础的训练损失计算，作为默认策略使用。
    
    调用层次：
    - 策略层：基础训练流程控制
    - 可选使用底层 lib/losses 模块
    
    适用场景：
    - 简单的监督学习训练
    - 作为其他策略的基础
    
    生产级特性：
    - 支持多种损失函数
    - 梯度裁剪和累积
    - 学习率热身
    - 标签平滑
    """
    
    def __init__(
        self, 
        loss_type: str = "cross_entropy",
        use_lib_losses: bool = False,
        label_smoothing: float = 0.0,
        gradient_clip_norm: float = 0.0,
        **kwargs
    ):
        """
        初始化标准策略
        
        Args:
            loss_type: 损失类型 (cross_entropy, mse, mae, focal, etc.)
            use_lib_losses: 是否使用 lib/losses 模块
            label_smoothing: 标签平滑系数
            gradient_clip_norm: 梯度裁剪范数（0表示不裁剪）
        """
        super().__init__(
            name="standard", 
            priority=100,
            strategy_type=StrategyType.STANDARD,
            **kwargs
        )
        
        self.loss_type = loss_type
        self.use_lib_losses = use_lib_losses
        self.label_smoothing = label_smoothing
        self.gradient_clip_norm = gradient_clip_norm
        
        self._step_count = 0
        self._loss_fn: Optional[nn.Module] = None
        
        # 损失历史（用于自适应调整）
        self._recent_losses: deque = deque(maxlen=100)
        
        # 梯度统计
        self._grad_norms: deque = deque(maxlen=100)
    
    def setup(self, context: StrategyContext) -> None:
        """初始化标准策略"""
        super().setup(context)
        self._step_count = 0
        
        # 初始化损失函数
        self._init_loss_fn()
        
        logger.info(f"StandardTrainingStrategy setup: loss_type={self.loss_type}, "
                   f"use_lib_losses={self.use_lib_losses}")
    
    def _init_loss_fn(self) -> None:
        """初始化损失函数"""
        if self.use_lib_losses:
            try:
                from backend.lib.losses import LossFactory
                factory = LossFactory()
                self._loss_fn = factory.create(self.loss_type)
                logger.info(f"Using lib/losses: {self.loss_type}")
                return
            except Exception as e:
                logger.warning(f"Failed to create loss from lib/losses: {e}")
        
        # 回退到PyTorch内置损失
        loss_map = {
            'cross_entropy': nn.CrossEntropyLoss(label_smoothing=self.label_smoothing),
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'bce': nn.BCEWithLogitsLoss(),
            'nll': nn.NLLLoss(),
            'kl_div': nn.KLDivLoss(reduction='batchmean'),
        }
        self._loss_fn = loss_map.get(self.loss_type, nn.CrossEntropyLoss())
    
    def reconfigure_loss(self, config: Dict[str, Any]) -> bool:
        """
        重新配置损失函数
        
        使用 create_loss_from_config 从配置字典创建新的损失函数。
        
        Args:
            config: 损失函数配置
            
        Returns:
            是否成功重新配置
        """
        try:
            if create_loss_from_config:
                new_loss = create_loss_from_config(config)
                if new_loss:
                    self._loss_fn = new_loss
                    self.loss_type = config.get('type', self.loss_type)
                    self.use_lib_losses = True
                    logger.info(f"Reconfigured loss from config: {self.loss_type}")
                    return True
        except Exception as e:
            logger.error(f"Failed to reconfigure loss: {e}")
        return False
        
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """
        计算标准训练损失
        """
        result = StrategyResult()
        
        # 获取或计算损失
        if 'loss' in outputs:
            loss = outputs['loss']
        elif hasattr(outputs, 'loss') and outputs.loss is not None:
            loss = outputs.loss
        else:
            # 使用损失函数计算
            logits = outputs.get('logits', outputs.get('output'))
            labels = batch.get('labels', batch.get('targets'))
            
            if logits is None or labels is None:
                result.add_error("Cannot compute loss: missing logits or labels")
                return result
            
            if self._loss_fn is not None:
                loss = self._loss_fn(logits, labels)
            else:
                loss = nn.functional.cross_entropy(logits, labels)
        
        # 处理 LibLossResult
        loss_metrics = {}
        # 检查是否为 BaseLoss 或其结果
        is_base_loss = isinstance(self._loss_fn, BaseLoss) if self._loss_fn else False
        
        if 'LibLossResult' in globals() and isinstance(loss, LibLossResult):
            loss_metrics = loss.metrics
            loss = loss.loss
        elif is_base_loss and hasattr(loss, 'metrics'):
             # 处理可能的自定义 BaseLoss 返回结构
             loss_metrics = getattr(loss, 'metrics', {})
             if hasattr(loss, 'loss'):
                 loss = loss.loss
            
        self._step_count += 1
        loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
        self._recent_losses.append(loss_value)
        
        # 构建结果
        result.loss = loss
        result.metrics = {
            'loss': loss_value,
            'step': self._step_count,
            **loss_metrics  # 添加额外指标
        }
        
        # 添加额外指标
        if len(self._recent_losses) >= 10:
            result.metrics['avg_loss_10'] = sum(list(self._recent_losses)[-10:]) / 10
        
        # 计算准确率（如果是分类任务）
        if 'logits' in outputs and 'labels' in batch:
            logits = outputs['logits']
            labels = batch['labels']
            if logits.dim() > 1 and labels.dim() == 1:
                preds = logits.argmax(dim=-1)
                accuracy = (preds == labels).float().mean().item()
                result.metrics['accuracy'] = accuracy
        
        # 损失组件
        result.add_loss_component('task_loss', loss_value)
        
        return result
    
    def clip_gradients(self, model: nn.Module) -> float:
        """裁剪梯度并返回梯度范数"""
        if self.gradient_clip_norm <= 0:
            # 只计算范数，不裁剪
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5
        
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.gradient_clip_norm
        )
        grad_norm_value = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        self._grad_norms.append(grad_norm_value)
        return grad_norm_value
    
    def get_loss_statistics(self) -> Dict[str, float]:
        """获取损失统计"""
        if not self._recent_losses:
            return {}
        
        losses = list(self._recent_losses)
        return {
            'count': len(losses),
            'mean': sum(losses) / len(losses),
            'min': min(losses),
            'max': max(losses),
            'std': (sum((x - sum(losses)/len(losses))**2 for x in losses) / len(losses)) ** 0.5,
        }
    
    def get_gradient_statistics(self) -> Dict[str, float]:
        """获取梯度统计"""
        if not self._grad_norms:
            return {}
        
        norms = list(self._grad_norms)
        return {
            'count': len(norms),
            'mean': sum(norms) / len(norms),
            'min': min(norms),
            'max': max(norms),
        }
    
    def get_layer_info(self) -> Dict[str, Any]:
        """获取底层模块调用信息"""
        return {
            'strategy_layer': True,
            'losses_layer': self.use_lib_losses,
            'adapters_layer': False,
            'distributed_layer': False,
            'hardware_layer': False,
            'step_count': self._step_count,
            'loss_type': self.loss_type,
            'label_smoothing': self.label_smoothing,
            'gradient_clip_norm': self.gradient_clip_norm,
        }
    
    def get_config(self) -> Dict[str, Any]:
        """获取策略配置"""
        config = super().get_config()
        config.update({
            'loss_type': self.loss_type,
            'use_lib_losses': self.use_lib_losses,
            'label_smoothing': self.label_smoothing,
            'gradient_clip_norm': self.gradient_clip_norm,
        })
        return config
    
    def reset_stats(self) -> None:
        """重置统计数据"""
        super().reset_stats()
        self._step_count = 0
        self._recent_losses.clear()
        self._grad_norms.clear()


class CompositeStrategy(TrainingStrategy):
    """
    组合策略
    
    将多个策略组合在一起，按优先级顺序执行。
    损失函数为各策略损失的加权和。
    
    生产级特性：
    - 动态权重调整
    - 策略启用/禁用控制
    - 子策略监控聚合
    - 策略依赖管理
    """
    
    def __init__(
        self, 
        strategies: List[TrainingStrategy], 
        weights: Optional[List[float]] = None,
        auto_balance: bool = False,
        balance_method: str = "inverse",
        **kwargs
    ):
        """
        初始化组合策略
        
        Args:
            strategies: 策略列表
            weights: 各策略的权重（默认均等）
            auto_balance: 是否自动平衡权重
            balance_method: 平衡方法 (inverse, softmax, uniform)
        """
        super().__init__(
            name="composite", 
            priority=0,
            strategy_type=StrategyType.COMPOSITE,
            **kwargs
        )
        
        # 按优先级排序
        self.strategies = sorted(strategies, key=lambda s: s.priority)
        
        # 设置权重
        if weights is None:
            self.weights = [1.0] * len(strategies)
        else:
            if len(weights) != len(strategies):
                raise ValueError("权重数量必须与策略数量相同")
            self.weights = list(weights)
        
        self.auto_balance = auto_balance
        self.balance_method = balance_method
        
        # 各策略的损失历史（用于自动平衡）
        self._strategy_losses: Dict[str, deque] = {
            s.name: deque(maxlen=100) for s in strategies
        }
        
        # 权重历史
        self._weight_history: List[Dict[str, float]] = []
        
        # 步数计数
        self._composite_step = 0
        
        logger.info(f"CompositeStrategy created with {len(strategies)} strategies: "
                   f"{[s.name for s in strategies]}")
    
    def setup(self, context: StrategyContext) -> None:
        """初始化所有策略"""
        super().setup(context)
        for strategy in self.strategies:
            strategy.setup(context)
        
        logger.info(f"CompositeStrategy setup: {len(self.strategies)} strategies initialized")
    
    def prepare_batch(self, batch: Dict[str, Any], context: StrategyContext) -> Dict[str, Any]:
        """依次调用各策略的prepare_batch"""
        for strategy in self.strategies:
            if strategy.is_enabled:
                batch = strategy.prepare_batch(batch, context)
        return batch
    
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """计算组合损失"""
        self._composite_step += 1
        
        total_loss = None
        all_metrics = {}
        all_loss_components = {}
        all_warnings = []
        all_errors = []
        
        for strategy, weight in zip(self.strategies, self.weights):
            if not strategy.is_enabled:
                continue
            
            # 使用带监控的计算
            if hasattr(strategy, 'compute_loss_with_monitoring'):
                result = strategy.compute_loss_with_monitoring(model, batch, outputs, context)
            else:
                result = strategy.compute_loss(model, batch, outputs, context)
            
            if result.loss is not None:
                loss_value = result.get_loss_value()
                
                # 记录策略损失
                self._strategy_losses[strategy.name].append(loss_value)
                
                # 加权损失
                weighted_loss = result.loss * weight
                if total_loss is None:
                    total_loss = weighted_loss
                else:
                    total_loss = total_loss + weighted_loss
                
                # 记录损失组件
                all_loss_components[f"{strategy.name}_loss"] = loss_value
                all_loss_components[f"{strategy.name}_weighted"] = loss_value * weight
            
            # 收集指标
            for key, value in result.metrics.items():
                all_metrics[f"{strategy.name}_{key}"] = value
        
            # 收集警告和错误
            all_warnings.extend([f"[{strategy.name}] {w}" for w in result.warnings])
            all_errors.extend([f"[{strategy.name}] {e}" for e in result.errors])
        
        # 自动平衡权重
        if self.auto_balance and self._composite_step % 100 == 0:
            self._update_weights()
        
        # 构建结果
        result = StrategyResult(
            loss=total_loss,
            metrics=all_metrics,
            loss_components=all_loss_components,
        )
        result.warnings = all_warnings
        result.errors = all_errors
        
        if total_loss is not None:
            result.metrics['total_loss'] = result.get_loss_value()
        
        # 记录权重
        weight_dict = {s.name: w for s, w in zip(self.strategies, self.weights)}
        result.metrics['weights'] = weight_dict
        
        return result
    
    def _update_weights(self) -> None:
        """更新权重（自动平衡）"""
        if self.balance_method == "inverse":
            # 反比例：损失越大，权重越小
            new_weights = []
            for strategy in self.strategies:
                losses = list(self._strategy_losses[strategy.name])
                if losses:
                    avg_loss = sum(losses) / len(losses)
                    new_weights.append(1.0 / (avg_loss + 1e-8))
                else:
                    new_weights.append(1.0)
            
            # 归一化
            total = sum(new_weights)
            new_weights = [w / total * len(new_weights) for w in new_weights]
            
        elif self.balance_method == "softmax":
            # Softmax：基于损失的软权重
            import math
            new_weights = []
            for strategy in self.strategies:
                losses = list(self._strategy_losses[strategy.name])
                if losses:
                    avg_loss = sum(losses) / len(losses)
                    new_weights.append(-avg_loss)  # 负损失，损失越小分数越高
                else:
                    new_weights.append(0.0)
            
            # Softmax
            max_w = max(new_weights)
            exp_weights = [math.exp(w - max_w) for w in new_weights]
            total = sum(exp_weights)
            new_weights = [w / total * len(new_weights) for w in exp_weights]
            
        else:  # uniform
            new_weights = [1.0] * len(self.strategies)
        
        # 平滑更新
        alpha = 0.1
        self.weights = [
            alpha * new + (1 - alpha) * old 
            for new, old in zip(new_weights, self.weights)
        ]
        
        # 记录权重历史
        self._weight_history.append({
            s.name: w for s, w in zip(self.strategies, self.weights)
        })
    
    def on_step_start(self, context: StrategyContext) -> None:
        """调用所有策略的step_start回调"""
        super().on_step_start(context)
        for strategy in self.strategies:
            if strategy.is_enabled:
                strategy.on_step_start(context)
    
    def on_step_end(self, context: StrategyContext, result: StrategyResult) -> None:
        """调用所有策略的step_end回调"""
        super().on_step_end(context, result)
        for strategy in self.strategies:
            if strategy.is_enabled:
                strategy.on_step_end(context, result)
    
    def on_epoch_start(self, context: StrategyContext) -> None:
        """调用所有策略的epoch_start回调"""
        super().on_epoch_start(context)
        for strategy in self.strategies:
            if strategy.is_enabled:
                strategy.on_epoch_start(context)
    
    def on_epoch_end(self, context: StrategyContext) -> None:
        """调用所有策略的epoch_end回调"""
        super().on_epoch_end(context)
        for strategy in self.strategies:
            if strategy.is_enabled:
                strategy.on_epoch_end(context)
    
    def on_phase_start(self, phase: TrainingPhase, context: StrategyContext) -> None:
        """调用所有策略的phase_start回调"""
        super().on_phase_start(phase, context)
        for strategy in self.strategies:
            if strategy.is_enabled:
                strategy.on_phase_start(phase, context)
    
    def on_phase_end(self, phase: TrainingPhase, context: StrategyContext) -> None:
        """调用所有策略的phase_end回调"""
        super().on_phase_end(phase, context)
        for strategy in self.strategies:
            if strategy.is_enabled:
                strategy.on_phase_end(phase, context)
    
    def cleanup(self) -> None:
        """清理所有策略"""
        super().cleanup()
        for strategy in self.strategies:
            strategy.cleanup()
    
    def add_strategy(self, strategy: TrainingStrategy, weight: float = 1.0) -> None:
        """添加策略"""
        self.strategies.append(strategy)
        self.weights.append(weight)
        self._strategy_losses[strategy.name] = deque(maxlen=100)
        
        # 重新排序
        combined = list(zip(self.strategies, self.weights))
        combined.sort(key=lambda x: x[0].priority)
        self.strategies, self.weights = map(list, zip(*combined))
        
        logger.info(f"Added strategy: {strategy.name} with weight {weight}")
    
    def remove_strategy(self, name: str) -> bool:
        """移除策略"""
        for i, strategy in enumerate(self.strategies):
            if strategy.name == name:
                self.strategies.pop(i)
                self.weights.pop(i)
                del self._strategy_losses[name]
                logger.info(f"Removed strategy: {name}")
                return True
        return False
    
    def get_strategy(self, name: str) -> Optional[TrainingStrategy]:
        """获取策略"""
        for strategy in self.strategies:
            if strategy.name == name:
                return strategy
        return None
    
    def set_weight(self, name: str, weight: float) -> bool:
        """设置策略权重"""
        for i, strategy in enumerate(self.strategies):
            if strategy.name == name:
                self.weights[i] = weight
                return True
        return False
    
    def get_weight(self, name: str) -> Optional[float]:
        """获取策略权重"""
        for i, strategy in enumerate(self.strategies):
            if strategy.name == name:
                return self.weights[i]
        return None
    
    def enable_strategy(self, name: str) -> bool:
        """启用策略"""
        strategy = self.get_strategy(name)
        if strategy:
            strategy.is_enabled = True
            return True
        return False
    
    def disable_strategy(self, name: str) -> bool:
        """禁用策略"""
        strategy = self.get_strategy(name)
        if strategy:
            strategy.is_enabled = False
            return True
        return False
    
    def set_auto_balance(self, enabled: bool, method: str = "inverse") -> None:
        """设置自动平衡"""
        self.auto_balance = enabled
        self.balance_method = method
    
    def get_weight_history(self) -> List[Dict[str, float]]:
        """获取权重历史"""
        return self._weight_history.copy()
    
    def get_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取各策略统计"""
        stats = {}
        for strategy in self.strategies:
            losses = list(self._strategy_losses[strategy.name])
            strategy_stats = {
                'enabled': strategy.is_enabled,
                'weight': self.get_weight(strategy.name),
                'loss_count': len(losses),
            }
            
            if losses:
                strategy_stats['avg_loss'] = sum(losses) / len(losses)
                strategy_stats['min_loss'] = min(losses)
                strategy_stats['max_loss'] = max(losses)
            
            # 获取子策略监控
            if strategy._monitor:
                strategy_stats['monitor'] = strategy._monitor.get_summary()
            
            stats[strategy.name] = strategy_stats
        
        return stats
    
    def get_config(self) -> Dict[str, Any]:
        """获取策略配置"""
        config = super().get_config()
        config.update({
            'strategies': [s.name for s in self.strategies],
            'weights': self.weights.copy(),
            'auto_balance': self.auto_balance,
            'balance_method': self.balance_method,
        })
        return config
    
    def get_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        info = super().get_info()
        info.update({
            'num_strategies': len(self.strategies),
            'enabled_strategies': sum(1 for s in self.strategies if s.is_enabled),
            'strategy_names': [s.name for s in self.strategies],
            'current_weights': {s.name: w for s, w in zip(self.strategies, self.weights)},
        })
        return info
    
    def print_summary(self) -> None:
        """打印组合策略摘要"""
        print("\n" + "="*60)
        print(f"CompositeStrategy Summary: {self.name}")
        print("="*60)
        
        print(f"\nStrategies ({len(self.strategies)}):")
        for strategy, weight in zip(self.strategies, self.weights):
            status = "✓" if strategy.is_enabled else "✗"
            losses = list(self._strategy_losses[strategy.name])
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            print(f"  {status} {strategy.name}: weight={weight:.4f}, avg_loss={avg_loss:.4f}")
        
        print(f"\nAuto Balance: {self.auto_balance} ({self.balance_method})")
        print(f"Total Steps: {self._composite_step}")
        
        if self._monitor:
            print("\n--- Composite Monitor ---")
            summary = self._monitor.get_summary()
            print(f"  Avg Loss: {summary.get('avg_loss', 0.0):.4f}")
            print(f"  Loss Trend: {summary.get('loss_trend', 'unknown')}")
        
        print("="*60)


# ==================== 工具函数 ====================

def create_standard_strategy(
    loss_type: str = "cross_entropy",
    use_lib_losses: bool = False,
    label_smoothing: float = 0.0,
    gradient_clip_norm: float = 1.0,
    **kwargs
) -> StandardTrainingStrategy:
    """创建标准训练策略
    
    Args:
        loss_type: 损失类型
        use_lib_losses: 是否使用 lib/losses 模块
        label_smoothing: 标签平滑系数
        gradient_clip_norm: 梯度裁剪范数
        **kwargs: 其他参数
        
    Returns:
        StandardTrainingStrategy 实例
    """
    return StandardTrainingStrategy(
        loss_type=loss_type,
        use_lib_losses=use_lib_losses,
        label_smoothing=label_smoothing,
        gradient_clip_norm=gradient_clip_norm,
        **kwargs
    )


def create_composite_strategy(
    strategies: List[TrainingStrategy],
    weights: Optional[List[float]] = None,
    auto_balance: bool = False,
    **kwargs
) -> CompositeStrategy:
    """创建组合策略
    
    Args:
        strategies: 策略列表
        weights: 权重列表
        auto_balance: 是否自动平衡
        **kwargs: 其他参数
        
    Returns:
        CompositeStrategy 实例
    """
    return CompositeStrategy(
        strategies=strategies,
        weights=weights,
        auto_balance=auto_balance,
        **kwargs
    )


def create_context(
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    max_epochs: int = 10,
    max_steps: int = 0,
    **kwargs
) -> StrategyContext:
    """创建策略上下文
    
    Args:
        model: 模型
        optimizer: 优化器
        device: 设备
        max_epochs: 最大训练轮数
        max_steps: 最大训练步数
        **kwargs: 其他配置
        
    Returns:
        StrategyContext 实例
    """
    if device is None:
        try:
            device_manager = get_device_manager()
            device = device_manager.get_device()
        except Exception:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    context = StrategyContext(
        model=model,
        optimizer=optimizer,
        device=device,
        max_epochs=max_epochs,
        max_steps=max_steps,
        config=kwargs,
    )
    
    # 设置分布式信息

    try:
        context.world_size = get_world_size()
        context.rank = get_rank()
        context.is_main_process = is_main_process()
    except Exception:
        pass
    
    return context


def diagnose_strategy(strategy: TrainingStrategy) -> Dict[str, Any]:
    """诊断策略状态
    
    Args:
        strategy: 策略实例
        
    Returns:
        诊断信息
    """
    return strategy.diagnose()


def print_strategy_diagnosis(strategy: TrainingStrategy) -> None:
    """打印策略诊断信息"""
    strategy.print_diagnosis()


def compare_strategies(
    strategies: List[TrainingStrategy],
    context: Optional[StrategyContext] = None
) -> Dict[str, Any]:
    """比较多个策略
    
    Args:
        strategies: 策略列表
        context: 上下文（用于setup）
        
    Returns:
        比较结果
    """
    comparison = {
        'strategies': {},
        'summary': {},
    }
    
    for strategy in strategies:
        info = strategy.get_info()
        comparison['strategies'][strategy.name] = info
        
        # 监控摘要
        if strategy._monitor:
            comparison['strategies'][strategy.name]['monitor'] = strategy._monitor.get_summary()
    
    # 汇总
    comparison['summary'] = {
        'count': len(strategies),
        'types': list(set(s.strategy_type.value for s in strategies)),
        'enabled': sum(1 for s in strategies if s.is_enabled),
    }
    
    return comparison


def print_strategy_comparison(strategies: List[TrainingStrategy]) -> None:
    """打印策略比较"""
    comparison = compare_strategies(strategies)
    
    print("\n" + "="*60)
    print("Strategy Comparison")
    print("="*60)
    
    for name, info in comparison['strategies'].items():
        print(f"\n{name}:")
        print(f"  Type: {info.get('type', 'unknown')}")
        print(f"  Priority: {info.get('priority', 0)}")
        print(f"  Enabled: {info.get('enabled', False)}")
        print(f"  Compute Count: {info.get('compute_count', 0)}")
        
        if 'monitor' in info:
            monitor = info['monitor']
            print(f"  Avg Loss: {monitor.get('avg_loss', 0.0):.4f}")
            print(f"  Loss Trend: {monitor.get('loss_trend', 'unknown')}")
    
    print("\n--- Summary ---")
    print(f"  Total Strategies: {comparison['summary']['count']}")
    print(f"  Types: {comparison['summary']['types']}")
    print(f"  Enabled: {comparison['summary']['enabled']}")
    print("="*60)


def get_available_strategy_types() -> List[str]:
    """获取可用的策略类型"""
    return [st.value for st in StrategyType]


def get_available_training_phases() -> List[str]:
    """获取可用的训练阶段"""
    return [tp.value for tp in TrainingPhase]


def validate_strategy(strategy: TrainingStrategy) -> Tuple[bool, List[str]]:
    """验证策略配置
    
    Args:
        strategy: 策略实例
        
    Returns:
        (是否有效, 错误列表)
    """
    errors = []
    
    # 检查基本属性
    if not strategy.name:
        errors.append("Strategy name is empty")
    
    if strategy.priority < 0:
        errors.append("Priority should be non-negative")

    return len(errors) == 0, errors


@contextmanager
def strategy_context(strategy: TrainingStrategy, context: StrategyContext):
    """策略执行上下文管理器
    
    自动处理setup和cleanup。
    
    Usage:
        with strategy_context(strategy, ctx) as s:
            result = s.compute_loss(model, batch, outputs, ctx)
    """
    try:
        strategy.setup(context)
        strategy.on_training_start(context)
        yield strategy
    finally:
        strategy.on_training_end(context)
        strategy.cleanup()


def save_strategy_state(strategy: TrainingStrategy, path: str) -> None:
    """保存策略状态
    
    Args:
        strategy: 策略实例
        path: 保存路径
    """
    state = strategy.get_state_dict()
    
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    logger.info(f"Saved strategy state to {path}")


def load_strategy_state(strategy: TrainingStrategy, path: str) -> None:
    """加载策略状态
    
    Args:
        strategy: 策略实例
        path: 加载路径
    """
    with open(path, 'r') as f:
        state = json.load(f)
    
    strategy.load_state_dict(state)
    logger.info(f"Loaded strategy state from {path}")


def print_strategy_info(strategy: TrainingStrategy) -> None:
    """打印策略信息"""
    strategy.print_summary()


def print_all_strategy_types() -> None:
    """打印所有策略类型信息"""
    print("\n" + "="*60)
    print("Available Strategy Types")
    print("="*60)
    
    for st in StrategyType:
        print(f"\n{st.value}:")
        print(f"  Description: {st.get_description()}")
        print(f"  Default Priority: {st.default_priority}")
        print(f"  Requires losses: {st.requires_lib_losses}")
        print(f"  Requires distributed: {st.requires_lib_distributed}")
        print(f"  Requires hardware: {st.requires_lib_hardware}")
        print(f"  Composable: {st.is_composable}")
    
    print("="*60)


def print_all_training_phases() -> None:
    """打印所有训练阶段信息"""
    print("\n" + "="*60)
    print("Available Training Phases")
    print("="*60)
    
    for tp in TrainingPhase:
        print(f"\n{tp.value}:")
        print(f"  Description: {tp.get_description()}")
        print(f"  Typical Epochs: {tp.typical_epochs}")
        print(f"  LR Scale: {tp.typical_lr_scale}")
        print(f"  Is Training: {tp.is_training}")
        next_phase = tp.next_phase()
        if next_phase:
            print(f"  Next Phase: {next_phase.value}")
    
    print("="*60)


# -*- coding: utf-8 -*-
"""
三阶段训练策略

将三阶段训练（预训练、微调、偏好优化）的策略逻辑从业务模块中分离，
提供生产级的训练策略能力。

调用层次：
- StandardStrategy + Orchestrator
- backend/lib/hardware: 设备管理、混合精度
- backend/lib/distributed: 分布式训练
- backend/lib/losses: 损失函数组合
- backend/lib/adapters: 模型适配器

架构图：
┌──────────────────────────────────────┐
│          three_stage/ (业务层)       │
│     (数据加载、配置、进度回调)        │
├──────────────────────────────────────┤
│ >>> ThreeStageStrategy (当前层) <<<  │
│    (三阶段训练策略核心逻辑)           │
├──────────────────────────────────────┤
│      StandardStrategy + Orchestrator │
│    (标准训练流程 + 编排能力)          │
├──────────────────────────────────────┤
│      backend/lib/* (底层能力)        │
│   (hardware/distributed/losses/...)  │
└──────────────────────────────────────┘

生产级特性：
- 完整的监控和诊断能力
- 阶段状态跟踪和持久化
- 健康检查和自动恢复
- 与底层lib模块的深度集成
"""

import logging
import copy
import time
import json
import os
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import nullcontext, contextmanager
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_strategy import (
    TrainingStrategy, StrategyContext, StrategyResult, TrainingPhase,
    StrategyType, StrategyMonitor, StrategyProfiler, StrategyValidator, StrategyMetrics
)
from .production_base import (
    ProductionTrainingStrategy, ProductionStrategyConfig,
    ProductionTrainingContext, create_production_context,
    ProductionHealthStatus, WrapperStats
)

logger = logging.getLogger(__name__)


# ==================== 额外的底层模块导入 ====================

# 硬件层详细导入
from backend.lib.hardware import (
        DeviceManager, get_device_manager,
        MixedPrecisionManager, PrecisionMode,
        MemoryManager, GradientCheckpointing, clear_memory,
        get_available_memory, MemoryStats,
)

# 分布式层详细导入
from backend.lib.distributed import (
    DistributedManager, get_distributed_manager,
    is_main_process, get_rank, get_world_size,
    barrier, all_reduce, AllReduceOp,
)

# 损失层详细导入
from backend.lib.losses import (
    LossFactory, create_loss, create_composite_loss,
    BaseLoss, CompositeLoss, MultiTaskLoss,
    CrossEntropyLoss, MSELoss, LossMonitor, LossStats, LossResult,
)


# ==================== 三阶段枚举 ====================

class ThreeStagePhase(Enum):
    """三阶段训练阶段枚举"""
    PRETRAIN = "pretrain"      # 预训练：语言建模
    FINETUNE = "finetune"      # 监督微调：指令跟随
    PREFERENCE = "preference"  # 偏好优化：DPO/RLHF
    
    @classmethod
    def from_string(cls, value: str) -> 'ThreeStagePhase':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown three-stage phase: {value}")
    
    @property
    def stage_number(self) -> int:
        """获取阶段序号"""
        return {
            ThreeStagePhase.PRETRAIN: 1,
            ThreeStagePhase.FINETUNE: 2,
            ThreeStagePhase.PREFERENCE: 3,
        }[self]
    
    @property
    def typical_epochs(self) -> int:
        """获取典型训练轮数"""
        return {
            ThreeStagePhase.PRETRAIN: 1,
            ThreeStagePhase.FINETUNE: 3,
            ThreeStagePhase.PREFERENCE: 1,
        }[self]
    
    @property
    def typical_lr(self) -> float:
        """获取典型学习率"""
        return {
            ThreeStagePhase.PRETRAIN: 1e-4,
            ThreeStagePhase.FINETUNE: 2e-5,
            ThreeStagePhase.PREFERENCE: 1e-5,
        }[self]
    
    @property
    def loss_type(self) -> str:
        """获取损失函数类型"""
        return {
            ThreeStagePhase.PRETRAIN: "cross_entropy",
            ThreeStagePhase.FINETUNE: "cross_entropy",
            ThreeStagePhase.PREFERENCE: "dpo",
        }[self]
    
    @property
    def requires_reference_model(self) -> bool:
        """是否需要参考模型"""
        return self == ThreeStagePhase.PREFERENCE
    
    def next_phase(self) -> Optional['ThreeStagePhase']:
        """获取下一阶段"""
        if self == ThreeStagePhase.PRETRAIN:
            return ThreeStagePhase.FINETUNE
        elif self == ThreeStagePhase.FINETUNE:
            return ThreeStagePhase.PREFERENCE
        return None
    
    def previous_phase(self) -> Optional['ThreeStagePhase']:
        """获取上一阶段"""
        if self == ThreeStagePhase.FINETUNE:
            return ThreeStagePhase.PRETRAIN
        elif self == ThreeStagePhase.PREFERENCE:
            return ThreeStagePhase.FINETUNE
        return None
    
    def get_description(self) -> str:
        """获取阶段描述"""
        return {
            ThreeStagePhase.PRETRAIN: "预训练阶段：学习通用语言表示",
            ThreeStagePhase.FINETUNE: "监督微调阶段：学习指令跟随能力",
            ThreeStagePhase.PREFERENCE: "偏好优化阶段：对齐人类偏好",
        }[self]
    
    def to_training_phase(self) -> TrainingPhase:
        """转换为 TrainingPhase"""
        return {
            ThreeStagePhase.PRETRAIN: TrainingPhase.PRETRAIN,
            ThreeStagePhase.FINETUNE: TrainingPhase.FINETUNE,
            ThreeStagePhase.PREFERENCE: TrainingPhase.PREFERENCE,
        }[self]


# ==================== 数据类 ====================

@dataclass
class PhaseStats:
    """阶段统计数据"""
    phase: str = ""
    total_steps: int = 0
    total_epochs: int = 0
    total_loss: float = 0.0
    avg_loss: float = 0.0
    min_loss: float = float('inf')
    max_loss: float = float('-inf')
    best_loss: float = float('inf')
    perplexity: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    total_time: float = 0.0
    
    # DPO 特定
    avg_chosen_reward: float = 0.0
    avg_rejected_reward: float = 0.0
    avg_reward_margin: float = 0.0
    
    def update(self, loss: float, **kwargs) -> None:
        """更新统计"""
        self.total_steps += 1
        self.total_loss += loss
        self.avg_loss = self.total_loss / self.total_steps
        self.min_loss = min(self.min_loss, loss)
        self.max_loss = max(self.max_loss, loss)
        self.best_loss = min(self.best_loss, loss)
        
        # 计算困惑度
        if loss < 20:
            self.perplexity = torch.exp(torch.tensor(loss)).item()
        
        # 更新 DPO 指标
        if 'chosen_reward' in kwargs:
            n = self.total_steps
            self.avg_chosen_reward = ((n - 1) * self.avg_chosen_reward + kwargs['chosen_reward']) / n
        if 'rejected_reward' in kwargs:
            n = self.total_steps
            self.avg_rejected_reward = ((n - 1) * self.avg_rejected_reward + kwargs['rejected_reward']) / n
        if 'reward_margin' in kwargs:
            n = self.total_steps
            self.avg_reward_margin = ((n - 1) * self.avg_reward_margin + kwargs['reward_margin']) / n
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'phase': self.phase,
            'total_steps': self.total_steps,
            'total_epochs': self.total_epochs,
            'avg_loss': self.avg_loss,
            'min_loss': self.min_loss if self.min_loss != float('inf') else None,
            'max_loss': self.max_loss if self.max_loss != float('-inf') else None,
            'best_loss': self.best_loss if self.best_loss != float('inf') else None,
            'perplexity': self.perplexity,
            'total_time': self.total_time,
            'avg_chosen_reward': self.avg_chosen_reward,
            'avg_rejected_reward': self.avg_rejected_reward,
            'avg_reward_margin': self.avg_reward_margin,
        }


@dataclass
class ThreeStageStats:
    """三阶段统计数据"""
    pretrain: PhaseStats = field(default_factory=lambda: PhaseStats(phase='pretrain'))
    finetune: PhaseStats = field(default_factory=lambda: PhaseStats(phase='finetune'))
    preference: PhaseStats = field(default_factory=lambda: PhaseStats(phase='preference'))
    
    current_phase: Optional[str] = None
    completed_phases: List[str] = field(default_factory=list)
    total_training_time: float = 0.0
    
    def get_phase_stats(self, phase: Union[ThreeStagePhase, str]) -> PhaseStats:
        """获取指定阶段的统计"""
        if isinstance(phase, ThreeStagePhase):
            phase = phase.value
        return getattr(self, phase, PhaseStats())
    
    def update_phase(self, phase: Union[ThreeStagePhase, str], loss: float, **kwargs) -> None:
        """更新阶段统计"""
        if isinstance(phase, ThreeStagePhase):
            phase = phase.value
        stats = getattr(self, phase)
        stats.update(loss, **kwargs)
    
    def mark_phase_completed(self, phase: Union[ThreeStagePhase, str]) -> None:
        """标记阶段完成"""
        if isinstance(phase, ThreeStagePhase):
            phase = phase.value
        if phase not in self.completed_phases:
            self.completed_phases.append(phase)
        
        # 更新阶段结束时间
        stats = getattr(self, phase)
        stats.end_time = time.time()
        stats.total_time = stats.end_time - stats.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'pretrain': self.pretrain.to_dict(),
            'finetune': self.finetune.to_dict(),
            'preference': self.preference.to_dict(),
            'current_phase': self.current_phase,
            'completed_phases': self.completed_phases.copy(),
            'total_training_time': self.total_training_time,
        }


@dataclass
class ThreeStageHealthStatus:
    """三阶段健康状态"""
    is_healthy: bool = True
    current_phase_ok: bool = True
    ref_model_ok: bool = True
    memory_ok: bool = True
    loss_ok: bool = True
    convergence_ok: bool = True
    last_check_time: float = 0.0
    issues: List[str] = field(default_factory=list)
    
    def add_issue(self, issue: str) -> None:
        """添加问题"""
        self.issues.append(issue)
        self.is_healthy = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'is_healthy': self.is_healthy,
            'current_phase_ok': self.current_phase_ok,
            'ref_model_ok': self.ref_model_ok,
            'memory_ok': self.memory_ok,
            'loss_ok': self.loss_ok,
            'convergence_ok': self.convergence_ok,
            'last_check_time': self.last_check_time,
            'issues': self.issues.copy(),
        }


# ==================== 配置类 ====================

@dataclass
class ThreeStageStrategyConfig(ProductionStrategyConfig):
    """三阶段训练策略配置"""
    # 阶段控制
    enabled_stages: List[str] = field(default_factory=lambda: ["pretrain", "finetune", "preference"])
    pass_model_between_stages: bool = True
    save_intermediate_models: bool = True
    
    # 预训练配置
    pretrain_learning_rate: float = 1e-4
    pretrain_epochs: int = 1
    pretrain_warmup_steps: int = 500
    pretrain_warmup_ratio: float = 0.0
    
    # 微调配置
    finetune_learning_rate: float = 2e-5
    finetune_epochs: int = 3
    finetune_warmup_steps: int = 100
    finetune_warmup_ratio: float = 0.0
    
    # 偏好优化配置
    preference_learning_rate: float = 1e-5
    preference_epochs: int = 2
    preference_warmup_steps: int = 50
    preference_warmup_ratio: float = 0.0
    dpo_beta: float = 0.1  # DPO温度参数
    dpo_loss_type: str = "sigmoid"  # sigmoid, hinge, ipo
    dpo_label_smoothing: float = 0.0
    dpo_reference_free: bool = False
    
    # 优化器配置（继承自ProductionStrategyConfig）
    gradient_clipping: float = 1.0
    
    # 早停配置
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    early_stopping_metric: str = "loss"  # loss, perplexity, reward_margin
    
    # 检查点配置
    checkpoint_dir: str = "./checkpoints/three_stage"
    save_every_n_steps: int = 1000
    save_best_only: bool = False
    
    # 监控配置
    log_interval: int = 10
    eval_interval: int = 500
    
    def __post_init__(self):
        """初始化后验证"""
        self.validate()
    
    def validate(self) -> None:
        """验证配置"""
        super().validate() if hasattr(super(), 'validate') else None
        
        # 验证阶段
        valid_stages = {'pretrain', 'finetune', 'preference'}
        for stage in self.enabled_stages:
            if stage not in valid_stages:
                raise ValueError(f"Invalid stage: {stage}")
        
        # 验证学习率
        if self.pretrain_learning_rate <= 0:
            raise ValueError("pretrain_learning_rate must be > 0")
        if self.finetune_learning_rate <= 0:
            raise ValueError("finetune_learning_rate must be > 0")
        if self.preference_learning_rate <= 0:
            raise ValueError("preference_learning_rate must be > 0")
        
        # 验证 DPO 参数
        if self.dpo_beta <= 0:
            raise ValueError("dpo_beta must be > 0")
        if self.dpo_loss_type not in ['sigmoid', 'hinge', 'ipo']:
            raise ValueError(f"Invalid dpo_loss_type: {self.dpo_loss_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        base_dict = super().to_dict() if hasattr(super(), 'to_dict') else {}
        base_dict.update({
            'enabled_stages': self.enabled_stages.copy(),
            'pass_model_between_stages': self.pass_model_between_stages,
            'pretrain_learning_rate': self.pretrain_learning_rate,
            'pretrain_epochs': self.pretrain_epochs,
            'finetune_learning_rate': self.finetune_learning_rate,
            'finetune_epochs': self.finetune_epochs,
            'preference_learning_rate': self.preference_learning_rate,
            'preference_epochs': self.preference_epochs,
            'dpo_beta': self.dpo_beta,
            'dpo_loss_type': self.dpo_loss_type,
            'early_stopping_patience': self.early_stopping_patience,
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThreeStageStrategyConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def get_phase_config(self, phase: Union[ThreeStagePhase, str]) -> Dict[str, Any]:
        """获取阶段配置"""
        if isinstance(phase, str):
            phase = ThreeStagePhase.from_string(phase)
        
        if phase == ThreeStagePhase.PRETRAIN:
            return {
                'learning_rate': self.pretrain_learning_rate,
                'epochs': self.pretrain_epochs,
                'warmup_steps': self.pretrain_warmup_steps,
                'warmup_ratio': self.pretrain_warmup_ratio,
            }
        elif phase == ThreeStagePhase.FINETUNE:
            return {
                'learning_rate': self.finetune_learning_rate,
                'epochs': self.finetune_epochs,
                'warmup_steps': self.finetune_warmup_steps,
                'warmup_ratio': self.finetune_warmup_ratio,
            }
        elif phase == ThreeStagePhase.PREFERENCE:
            return {
                'learning_rate': self.preference_learning_rate,
                'epochs': self.preference_epochs,
                'warmup_steps': self.preference_warmup_steps,
                'warmup_ratio': self.preference_warmup_ratio,
                'dpo_beta': self.dpo_beta,
                'dpo_loss_type': self.dpo_loss_type,
            }
        return {}
    
    def summary(self) -> str:
        """获取配置摘要"""
        lines = [
            "ThreeStageStrategyConfig Summary:",
            f"  Enabled stages: {self.enabled_stages}",
            f"  Pretrain: lr={self.pretrain_learning_rate}, epochs={self.pretrain_epochs}",
            f"  Finetune: lr={self.finetune_learning_rate}, epochs={self.finetune_epochs}",
            f"  Preference: lr={self.preference_learning_rate}, epochs={self.preference_epochs}, beta={self.dpo_beta}",
            f"  Early stopping: patience={self.early_stopping_patience}",
        ]
        return '\n'.join(lines)


# ==================== 监控组件 ====================

class ThreeStageMonitor:
    """三阶段训练监控器
    
    整合 base_strategy.py 的 StrategyMonitor 并添加三阶段特定功能
    """
    
    def __init__(self, history_size: int = 10000):
        self.history_size = history_size
        
        # 使用基础监控器
        self._base_monitor = StrategyMonitor(history_size)
        
        # 三阶段统计
        self._stats = ThreeStageStats()
        
        # 阶段历史
        self._phase_loss_history: Dict[str, deque] = {
            'pretrain': deque(maxlen=history_size),
            'finetune': deque(maxlen=history_size),
            'preference': deque(maxlen=history_size),
        }
        
        # DPO 特定历史
        self._reward_margin_history: deque = deque(maxlen=history_size)
        
        # 健康状态
        self._health_status = ThreeStageHealthStatus()
    
    def record_step(
        self,
        result: StrategyResult,
        context: StrategyContext,
        phase: ThreeStagePhase,
        step_time: float = 0.0
    ) -> None:
        """记录训练步骤"""
        # 使用基础监控器
        self._base_monitor.record_step(result, context, step_time)
        
        # 更新三阶段统计
        loss_value = result.get_loss_value()
        phase_name = phase.value
        
        self._stats.current_phase = phase_name
        self._phase_loss_history[phase_name].append(loss_value)
        
        # 提取 DPO 指标
        dpo_kwargs = {}
        if 'chosen_reward' in result.metrics:
            dpo_kwargs['chosen_reward'] = result.metrics['chosen_reward']
            dpo_kwargs['rejected_reward'] = result.metrics.get('rejected_reward', 0.0)
            dpo_kwargs['reward_margin'] = result.metrics.get('reward_margin', 0.0)
            self._reward_margin_history.append(dpo_kwargs['reward_margin'])
        
        self._stats.update_phase(phase, loss_value, **dpo_kwargs)
    
    def record_epoch(self, phase: ThreeStagePhase, epoch: int, metrics: Dict[str, float]) -> None:
        """记录epoch"""
        phase_stats = self._stats.get_phase_stats(phase)
        phase_stats.total_epochs += 1
        
        # 调用基础监控器
        self._base_monitor.record_epoch(epoch, phase.to_training_phase(), metrics)
    
    def start_phase(self, phase: ThreeStagePhase) -> None:
        """开始阶段"""
        phase_stats = self._stats.get_phase_stats(phase)
        phase_stats.start_time = time.time()
        self._stats.current_phase = phase.value
    
    def end_phase(self, phase: ThreeStagePhase) -> None:
        """结束阶段"""
        self._stats.mark_phase_completed(phase)
    
    def get_phase_loss_trend(self, phase: ThreeStagePhase, window: int = 100) -> str:
        """获取阶段损失趋势"""
        history = self._phase_loss_history[phase.value]
        if len(history) < window * 2:
            return "insufficient_data"
        
        recent = list(history)[-window:]
        previous = list(history)[-window * 2:-window]
        
        recent_avg = sum(recent) / len(recent)
        previous_avg = sum(previous) / len(previous)
        
        change = (recent_avg - previous_avg) / max(previous_avg, 1e-8)
        
        if change < -0.05:
            return "improving"
        elif change > 0.05:
            return "degrading"
        else:
            return "stable"
    
    def get_reward_margin_trend(self, window: int = 100) -> str:
        """获取奖励差距趋势（DPO）"""
        if len(self._reward_margin_history) < window * 2:
            return "insufficient_data"
        
        recent = list(self._reward_margin_history)[-window:]
        previous = list(self._reward_margin_history)[-window * 2:-window]
        
        recent_avg = sum(recent) / len(recent)
        previous_avg = sum(previous) / len(previous)
        
        change = (recent_avg - previous_avg) / max(abs(previous_avg), 1e-8)
        
        if change > 0.05:
            return "improving"
        elif change < -0.05:
            return "degrading"
        else:
            return "stable"
    
    def check_health(self) -> ThreeStageHealthStatus:
        """检查健康状态"""
        self._health_status = ThreeStageHealthStatus()
        self._health_status.last_check_time = time.time()
        
        # 检查收敛
        loss_trend = self._base_monitor.get_loss_trend()
        if loss_trend == "degrading":
            self._health_status.convergence_ok = False
            self._health_status.add_issue("Loss is degrading")
        
        # 检查早停
        if self._base_monitor.check_early_stopping(patience=10):
            self._health_status.convergence_ok = False
            self._health_status.add_issue("Training may have converged (early stopping)")
        
        return self._health_status
    
    def get_stats(self) -> ThreeStageStats:
        """获取统计数据"""
        return self._stats
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        return {
            'base_monitor': self._base_monitor.get_summary(),
            'three_stage_stats': self._stats.to_dict(),
            'health_status': self._health_status.to_dict(),
            'phase_trends': {
                phase: self.get_phase_loss_trend(ThreeStagePhase.from_string(phase))
                for phase in ['pretrain', 'finetune', 'preference']
            },
            'reward_margin_trend': self.get_reward_margin_trend(),
        }
    
    def reset(self) -> None:
        """重置监控器"""
        self._base_monitor.reset()
        self._stats = ThreeStageStats()
        for history in self._phase_loss_history.values():
            history.clear()
        self._reward_margin_history.clear()
        self._health_status = ThreeStageHealthStatus()


class PhaseTracker:
    """阶段跟踪器
    
    跟踪三阶段训练的进度和状态
    """
    
    def __init__(self, config: ThreeStageStrategyConfig):
        self.config = config
        self._completed_phases: List[ThreeStagePhase] = []
        self._current_phase: Optional[ThreeStagePhase] = None
        self._phase_checkpoints: Dict[str, str] = {}
        self._phase_start_times: Dict[str, float] = {}
        self._phase_end_times: Dict[str, float] = {}
    
    def start_phase(self, phase: ThreeStagePhase) -> bool:
        """开始阶段"""
        # 检查阶段是否启用
        if phase.value not in self.config.enabled_stages:
            logger.info(f"Phase {phase.value} is not enabled, skipping")
            return False
        
        # 检查依赖
        if phase.previous_phase() and phase.previous_phase() not in self._completed_phases:
            if self.config.pass_model_between_stages:
                logger.warning(f"Previous phase {phase.previous_phase().value} not completed")
        
        self._current_phase = phase
        self._phase_start_times[phase.value] = time.time()
        logger.info(f"Starting phase: {phase.value} - {phase.get_description()}")
        return True
    
    def end_phase(self, phase: ThreeStagePhase, checkpoint_path: Optional[str] = None) -> None:
        """结束阶段"""
        self._completed_phases.append(phase)
        self._phase_end_times[phase.value] = time.time()
        
        if checkpoint_path:
            self._phase_checkpoints[phase.value] = checkpoint_path
        
        self._current_phase = None
        
        duration = self._phase_end_times[phase.value] - self._phase_start_times.get(phase.value, 0)
        logger.info(f"Completed phase: {phase.value} in {duration:.2f}s")
    
    def get_current_phase(self) -> Optional[ThreeStagePhase]:
        """获取当前阶段"""
        return self._current_phase
    
    def get_next_phase(self) -> Optional[ThreeStagePhase]:
        """获取下一个待执行阶段"""
        for phase in [ThreeStagePhase.PRETRAIN, ThreeStagePhase.FINETUNE, ThreeStagePhase.PREFERENCE]:
            if phase.value in self.config.enabled_stages and phase not in self._completed_phases:
                return phase
        return None
    
    def is_phase_completed(self, phase: ThreeStagePhase) -> bool:
        """检查阶段是否完成"""
        return phase in self._completed_phases
    
    def get_phase_checkpoint(self, phase: ThreeStagePhase) -> Optional[str]:
        """获取阶段检查点路径"""
        return self._phase_checkpoints.get(phase.value)
    
    def get_progress(self) -> Dict[str, Any]:
        """获取进度"""
        total_phases = len(self.config.enabled_stages)
        completed = len(self._completed_phases)
        
        return {
            'total_phases': total_phases,
            'completed_phases': completed,
            'progress': completed / max(total_phases, 1),
            'current_phase': self._current_phase.value if self._current_phase else None,
            'completed': [p.value for p in self._completed_phases],
            'remaining': [p for p in self.config.enabled_stages if p not in [c.value for c in self._completed_phases]],
        }
    
    def save_state(self, path: str) -> None:
        """保存状态"""
        state = {
            'completed_phases': [p.value for p in self._completed_phases],
            'current_phase': self._current_phase.value if self._current_phase else None,
            'phase_checkpoints': self._phase_checkpoints,
            'phase_start_times': self._phase_start_times,
            'phase_end_times': self._phase_end_times,
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: str) -> None:
        """加载状态"""
        if not os.path.exists(path):
            return
        
        with open(path, 'r') as f:
            state = json.load(f)
        
        self._completed_phases = [ThreeStagePhase.from_string(p) for p in state.get('completed_phases', [])]
        current = state.get('current_phase')
        self._current_phase = ThreeStagePhase.from_string(current) if current else None
        self._phase_checkpoints = state.get('phase_checkpoints', {})
        self._phase_start_times = state.get('phase_start_times', {})
        self._phase_end_times = state.get('phase_end_times', {})


# ==================== DPO损失计算器 ====================

class DPOLossCalculator:
    """
    DPO损失计算器
    
    实现Direct Preference Optimization损失计算，支持多种变体：
    - sigmoid: 标准 DPO 损失
    - hinge: Hinge loss 变体
    - ipo: IPO (Identity Preference Optimization) 变体
    
    整合 backend/lib/losses 如果可用
    """
    
    def __init__(
        self,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        label_smoothing: float = 0.0,
        reference_free: bool = False
    ):
        """
        初始化DPO损失计算器
        
        Args:
            beta: DPO温度参数，控制KL散度的惩罚强度
            loss_type: 损失类型 (sigmoid, hinge, ipo)
            label_smoothing: 标签平滑
            reference_free: 是否使用无参考模型变体
        """
        self.beta = beta
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.reference_free = reference_free
        
        # 统计
        self._total_calls = 0
        self._total_loss = 0.0
        self._loss_history: deque = deque(maxlen=1000)
        self._reward_margin_history: deque = deque(maxlen=1000)
        
        # 使用 lib/losses 如果可用
        self._lib_loss_monitor: Optional[Any] = None
        self._lib_loss_monitor = LossMonitor()
        
        logger.info(f"DPOLossCalculator initialized: beta={beta}, type={loss_type}")
    
    def compute_loss(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module],
        batch: Dict[str, Any],
        device: torch.device
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算DPO损失
        
        DPO损失公式：
        L = -log(sigmoid(beta * (r_chosen - r_rejected)))
        其中 r = log(pi(y|x)) - log(ref(y|x))
        
        Args:
            model: 策略模型
            ref_model: 参考模型（冻结），如果 reference_free 为 True 则可以为 None
            batch: 包含 chosen 和 rejected 数据的批次
            device: 设备
        
        Returns:
            (loss, metrics_dict)
        """
        self._total_calls += 1
        
        # 获取数据
        chosen_input_ids = batch['chosen_input_ids'].to(device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(device)
        rejected_input_ids = batch['rejected_input_ids'].to(device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(device)
        
        # 策略模型前向传播
        chosen_outputs = model(
            input_ids=chosen_input_ids,
            attention_mask=chosen_attention_mask
        )
        rejected_outputs = model(
            input_ids=rejected_input_ids,
            attention_mask=rejected_attention_mask
        )
        
        # 计算策略模型对数概率
        chosen_logps = self._get_batch_logps(
            chosen_outputs.logits, chosen_input_ids
        )
        rejected_logps = self._get_batch_logps(
            rejected_outputs.logits, rejected_input_ids
        )
        
        if self.reference_free:
            # 无参考模型变体
            logits = chosen_logps - rejected_logps
        else:
            # 需要参考模型
            if ref_model is None:
                raise ValueError("Reference model required for non-reference-free DPO")
            
            # 参考模型前向传播（不计算梯度）
            with torch.no_grad():
                ref_chosen_outputs = ref_model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask
                )
                ref_rejected_outputs = ref_model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask
                )
                
                ref_chosen_logps = self._get_batch_logps(
                    ref_chosen_outputs.logits, chosen_input_ids
                )
                ref_rejected_logps = self._get_batch_logps(
                    ref_rejected_outputs.logits, rejected_input_ids
                )
            
            # 计算对数比率
            pi_logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = pi_logratios - ref_logratios
        
        # 根据损失类型计算损失
        if self.loss_type == "sigmoid":
            loss = self._compute_sigmoid_loss(logits)
        elif self.loss_type == "hinge":
            loss = self._compute_hinge_loss(logits)
        elif self.loss_type == "ipo":
            loss = self._compute_ipo_loss(logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 应用标签平滑
        if self.label_smoothing > 0:
            loss = loss * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # 计算指标
        reward_margin = (chosen_logps - rejected_logps).mean().item()
        metrics = {
            'chosen_reward': chosen_logps.mean().item(),
            'rejected_reward': rejected_logps.mean().item(),
            'reward_margin': reward_margin,
            'accuracy': (logits > 0).float().mean().item(),
        }
        
        # 更新统计
        loss_value = loss.item()
        self._total_loss += loss_value
        self._loss_history.append(loss_value)
        self._reward_margin_history.append(reward_margin)
        
        # 记录到 lib loss monitor
        if self._lib_loss_monitor is not None:
            # LossMonitor.record 接受 LossResult 对象
            loss_result = LossResult(
                loss=loss,
                metrics=metrics,
                components={'dpo_loss': loss}
            )
            self._lib_loss_monitor.record(loss_result)
        
        return loss, metrics
    
    def _compute_sigmoid_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """计算 Sigmoid DPO 损失"""
        return -torch.logsigmoid(self.beta * logits).mean()
    
    def _compute_hinge_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """计算 Hinge DPO 损失"""
        return torch.relu(1 - self.beta * logits).mean()
    
    def _compute_ipo_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """计算 IPO 损失"""
        return (logits - 1 / (2 * self.beta)) ** 2
    
    def _get_batch_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        average: bool = False
    ) -> torch.Tensor:
        """
        计算批次对数概率
        
        Args:
            logits: 模型输出logits [batch, seq, vocab]
            labels: 目标标签 [batch, seq]
            average: 是否平均
        
        Returns:
            对数概率
        """
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # 创建mask（忽略padding，假设padding token id = -100）
        loss_mask = (shift_labels != -100)
        
        # 替换-100为0以避免索引错误
        shift_labels_safe = shift_labels.clone()
        shift_labels_safe[shift_labels == -100] = 0
        
        # 计算log softmax
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # 获取目标token的概率
        per_token_logps = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_labels_safe.unsqueeze(-1)
        ).squeeze(-1)
        
        # 应用mask
        per_token_logps = per_token_logps * loss_mask.float()
        
        if average:
            return per_token_logps.sum(-1) / loss_mask.sum(-1).clamp(min=1)
        else:
            return per_token_logps.sum(-1)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_calls': self._total_calls,
            'avg_loss': self._total_loss / max(self._total_calls, 1),
            'recent_loss': sum(self._loss_history) / max(len(self._loss_history), 1) if self._loss_history else 0.0,
            'recent_reward_margin': sum(self._reward_margin_history) / max(len(self._reward_margin_history), 1) if self._reward_margin_history else 0.0,
            'beta': self.beta,
            'loss_type': self.loss_type,
        }
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._total_calls = 0
        self._total_loss = 0.0
        self._loss_history.clear()
        self._reward_margin_history.clear()


# ==================== 三阶段训练策略 ====================

class ThreeStageStrategy(ProductionTrainingStrategy):
    """
    三阶段训练策略
    
    实现生产级的三阶段训练策略：
    1. 预训练（Pretrain）：语言建模，学习通用语言表示
    2. 监督微调（SFT）：指令跟随，学习任务特定能力
    3. 偏好优化（DPO）：对齐人类偏好，优化模型输出
    
    调用层次：
    - 继承 ProductionTrainingStrategy 获得六层架构能力
    - 调用 backend/lib/hardware 进行设备管理和混合精度
    - 调用 backend/lib/distributed 进行分布式训练
    - 调用 backend/lib/losses 进行损失计算
    - 调用 backend/lib/adapters 进行模型适配
    - 使用 base_strategy.py 的监控和验证组件
    """
    
    def __init__(
        self,
        config: Optional[ThreeStageStrategyConfig] = None,
        name: str = "three_stage",
        priority: int = 40
    ):
        # 创建基础配置
        if config is None:
            config = ThreeStageStrategyConfig()
        
        # 使用父类的 ProductionStrategyConfig 部分初始化
        base_config = ProductionStrategyConfig(
            device=config.device,
            precision=config.precision,
            enable_amp=config.enable_amp,
            distributed_mode=config.distributed_mode,
            world_size=config.world_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            modalities=config.modalities,
            hidden_size=config.hidden_size,
            task_loss_type=config.task_loss_type,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_grad_norm=config.gradient_clipping,
            enable_monitoring=config.enable_monitoring,
            enable_profiling=config.enable_profiling,
            log_interval=config.log_interval,
        )
        
        super().__init__(config=base_config, name=name, priority=priority)
        
        self.three_stage_config = config
        
        # 策略类型标识
        self._strategy_type = StrategyType.THREE_STAGE
        
        # 阶段状态
        self._current_phase: Optional[ThreeStagePhase] = None
        self._phase_results: Dict[str, Dict[str, Any]] = {}
        
        # 三阶段监控器
        self._three_stage_monitor = ThreeStageMonitor()
        
        # 阶段跟踪器
        self._phase_tracker = PhaseTracker(config)
        
        # 参考模型（用于DPO）
        self._ref_model: Optional[nn.Module] = None
        
        # 损失计算器
        self._pretrain_loss_fn: Optional[nn.Module] = None
        self._finetune_loss_fn: Optional[nn.Module] = None
        self._dpo_loss_calculator: Optional[DPOLossCalculator] = None
        
        # 使用 lib/losses 的损失函数和组件
        self._lib_cross_entropy: Optional[Any] = None
        self._lib_mse_loss: Optional[Any] = None
        self._lib_loss_monitor: Optional[Any] = None
        self._lib_loss_stats: Optional[Any] = None
        self._lib_loss_factory: Optional[Any] = None
        self._lib_composite_loss: Optional[Any] = None
        self._lib_multitask_loss: Optional[Any] = None
        
        # 训练组件
        self._gradient_accumulator: Optional['GradientAccumulator'] = None
        self._convergence_detector: Optional['ConvergenceDetector'] = None
        self._mixed_precision_manager: Optional['MixedPrecisionManager'] = None
        
        # 使用 lib/hardware 的组件
        self._device_manager: Optional[Any] = None
        self._memory_manager: Optional[Any] = None
        self._gradient_checkpointing: Optional[Any] = None
        self._memory_stats: Optional[Any] = None
        
        # 使用 lib/distributed 的组件
        self._distributed_manager: Optional[Any] = None
        self._wrapper_stats = WrapperStats()
        
        # 生产级健康状态
        self._production_health = ProductionHealthStatus()
        
        # 策略指标（使用 base_strategy.py 的 StrategyMetrics）
        self._strategy_metrics = StrategyMetrics()
        
        # 健康状态
        self._health_status = ThreeStageHealthStatus()
        
        # 统计信息
        self._total_steps = 0
        self._phase_steps = 0
        
        logger.info(f"ThreeStageStrategy initialized with config: {config.summary()}")
    
    def setup(self, context: StrategyContext) -> None:
        """初始化三阶段策略"""
        super().setup(context)
        
        logger.info(f"Setting up ThreeStageStrategy with config: {self.three_stage_config.summary()}")
        
        # 初始化硬件层组件
        self._setup_hardware_components()
        
        # 初始化分布式层组件
        self._setup_distributed_components()
        
        # 初始化损失函数
        self._setup_loss_functions()
        
        # 初始化训练组件
        self._setup_training_components()
        
        # 初始化健康检查
        self._health_status = ThreeStageHealthStatus()
        self._production_health = ProductionHealthStatus()
        
        # 注册自定义验证器
        self._register_custom_validators()
        
        logger.info("ThreeStageStrategy setup completed")
    
    def _register_custom_validators(self) -> None:
        """
        注册自定义验证器
        
        使用 StrategyValidator 提供的 add_check 接口添加三阶段特定的检查逻辑
        """
        if self._validator is None:
            return

        def check_dpo_metrics(result: StrategyResult) -> Tuple[bool, str]:
            if self._current_phase == ThreeStagePhase.PREFERENCE:
                # 检查 DPO 关键指标是否存在
                if 'reward_margin' not in result.metrics:
                    return False, "Missing reward_margin in DPO phase"
            return True, ""
            
        def check_loss_validity(result: StrategyResult) -> Tuple[bool, str]:
            # 检查损失值是否有效
            if result.loss is not None and isinstance(result.loss, torch.Tensor):
                if torch.isnan(result.loss).any():
                    return False, "Loss contains NaN"
                if torch.isinf(result.loss).any():
                    return False, "Loss contains Inf"
            return True, ""

        if hasattr(self._validator, 'add_check'):
            self._validator.add_check(check_dpo_metrics)
            self._validator.add_check(check_loss_validity)
            logger.info("Registered custom validators for ThreeStageStrategy")

    def enhance_loss_with_composite(self, main_loss: nn.Module, aux_losses: List[nn.Module], weights: List[float]) -> nn.Module:
        """
        使用 backend.lib.losses.create_composite_loss 创建组合损失
        
        Args:
            main_loss: 主损失函数
            aux_losses: 辅助损失函数列表
            weights: 权重列表
        """
        try:
            # create_composite_loss 接受损失列表，权重需要包含在元组中
            # 格式: [(name, loss_fn, weight), ...]
            loss_tuples = []
            # 主损失
            if isinstance(main_loss, BaseLoss):
                loss_tuples.append(('main', main_loss, 1.0))
            # 辅助损失
            for i, (aux_loss, weight) in enumerate(zip(aux_losses, weights)):
                if isinstance(aux_loss, BaseLoss):
                    loss_tuples.append((f'aux_{i}', aux_loss, weight))
            
            if loss_tuples:
                logger.info("Creating composite loss with %d components", len(loss_tuples))
                return create_composite_loss(loss_tuples)
        except Exception as e:
            logger.warning("Failed to create composite loss: %s", e)
        return main_loss

    def check_managers_health(self) -> Dict[str, Any]:
        """
        检查底层管理器健康状态
        
        利用 DeviceManager 和 DistributedManager 的能力
        """
        health_info = {'hardware': 'unknown', 'distributed': 'unknown'}
        
        # 检查硬件管理器
        if self._device_manager is not None and DeviceManager is not None and isinstance(self._device_manager, DeviceManager):
            try:
                # 假设 DeviceManager 有 check_health 方法，或者我们通过获取设备来检查
                if hasattr(self._device_manager, 'check_health'):
                    health_info['hardware'] = self._device_manager.check_health()
                else:
                    self._device_manager.get_device() # 尝试获取设备作为检查
                    health_info['hardware'] = 'healthy'
            except Exception as e:
                health_info['hardware'] = f"error: {e}"
                
        # 检查分布式管理器
        if self._distributed_manager is not None and DistributedManager is not None and isinstance(self._distributed_manager, DistributedManager):
            try:
                if hasattr(self._distributed_manager, 'check_health'):
                    health_info['distributed'] = self._distributed_manager.check_health()
                elif hasattr(self._distributed_manager, 'is_initialized'):
                    # is_initialized 可能是属性或方法
                    is_init_attr = getattr(self._distributed_manager, 'is_initialized')
                    if callable(is_init_attr):
                        is_init = is_init_attr()
                    else:
                        is_init = is_init_attr
                    health_info['distributed'] = 'initialized' if is_init else 'uninitialized'
            except Exception as e:
                health_info['distributed'] = f"error: {e}"
                
        return health_info
        
    def verify_loss_instance(self, loss_fn: Any) -> bool:
        """验证损失函数是否为 BaseLoss 实例"""
        if BaseLoss is not None and isinstance(loss_fn, BaseLoss):
            return True
        return False

    def _setup_hardware_components(self) -> None:
        """设置硬件层组件 - 使用 backend/lib/hardware"""
        try:
            # 使用 DeviceManager 获取设备管理器
            if get_device_manager is not None:
                self._device_manager = get_device_manager()
                logger.info(f"DeviceManager initialized: {self._device_manager}")
            
            # 初始化 MemoryManager
            if MemoryManager is not None:
                self._memory_manager = MemoryManager()
                logger.info("MemoryManager initialized")
            
            # 初始化 GradientCheckpointing（如果启用）
            # 注意：GradientCheckpointing 需要 model 参数，但此时 model 可能还未创建
            # 所以这里先不初始化，等到 setup 时再初始化
            self._gradient_checkpointing = None
            
            # 获取初始内存统计
            if MemoryStats is not None and self._memory_manager is not None:
                self._memory_stats = self._memory_manager.get_stats() if hasattr(self._memory_manager, 'get_stats') else None
            
            self._production_health.hardware_ok = True
            
        except Exception as e:
            logger.warning(f"Failed to setup hardware components: {e}")
            self._production_health.hardware_ok = False
            self._production_health.add_issue(f"Hardware setup failed: {e}")
    
    def _setup_distributed_components(self) -> None:
        """设置分布式层组件 - 使用 backend/lib/distributed"""
        
        try:
            # 使用 get_distributed_manager 获取分布式管理器
            if get_distributed_manager is not None:
                self._distributed_manager = get_distributed_manager()
                logger.info(f"DistributedManager initialized: {self._distributed_manager}")
            
            # 获取分布式信息
            rank = get_rank() if callable(get_rank) else 0
            world_size = get_world_size() if callable(get_world_size) else 1
            main_process = is_main_process() if callable(is_main_process) else True
            
            logger.info(f"Distributed info: rank={rank}, world_size={world_size}, is_main={main_process}")
            
            # 初始化 wrapper 统计
            self._wrapper_stats = WrapperStats(
                wrapper_type=self.three_stage_config.distributed_mode,
                is_active=world_size > 1
            )
            
            self._production_health.distributed_ok = True
            
        except Exception as e:
            logger.warning(f"Failed to setup distributed components: {e}")
            self._production_health.distributed_ok = False
            self._production_health.add_issue(f"Distributed setup failed: {e}")
    
    def _setup_loss_functions(self) -> None:
        """设置各阶段的损失函数 - 使用 backend/lib/losses"""
        # 使用 backend/lib/losses 如果可用
        try:
            # 获取 LossFactory 实例
            if LossFactory is not None:
                self._lib_loss_factory = LossFactory()
                logger.info("LossFactory initialized")
                
            # 预训练使用交叉熵损失 - 优先使用 CrossEntropyLoss 类
            if CrossEntropyLoss is not None:
                self._pretrain_loss_fn = CrossEntropyLoss()
                self._lib_cross_entropy = self._pretrain_loss_fn
            elif create_loss is not None:
                self._pretrain_loss_fn = create_loss('cross_entropy')
                self._lib_cross_entropy = self._pretrain_loss_fn
            else:
                self._pretrain_loss_fn = nn.CrossEntropyLoss()
            
            # 微调使用交叉熵损失
            if CrossEntropyLoss is not None:
                self._finetune_loss_fn = CrossEntropyLoss()
            elif create_loss is not None:
                self._finetune_loss_fn = create_loss('cross_entropy')
            else:
                self._finetune_loss_fn = nn.CrossEntropyLoss()
                
            # 创建 MSE 损失（用于特征对齐等）
            if MSELoss is not None:
                self._lib_mse_loss = MSELoss()
            elif create_loss is not None:
                self._lib_mse_loss = create_loss('mse')
                
            # 创建复合损失（用于多任务学习场景）
            if CompositeLoss is not None:
                self._lib_composite_loss = CompositeLoss([])
                logger.info("CompositeLoss initialized")
                
            # 创建多任务损失
            if MultiTaskLoss is not None:
                self._lib_multitask_loss = MultiTaskLoss([])
                logger.info("MultiTaskLoss initialized")
                
            # 创建损失监控器
            if LossMonitor is not None:
                self._lib_loss_monitor = LossMonitor()
                logger.info("LossMonitor initialized")
                
            # 创建损失统计
            if LossStats is not None:
                self._lib_loss_stats = LossStats()
                logger.info("LossStats initialized")
                
            self._production_health.losses_ok = True
            logger.info("Using backend.lib.losses for loss computation")
                
        except Exception as e:
            logger.warning(f"Failed to use lib.losses, falling back to PyTorch: {e}")
            self._pretrain_loss_fn = nn.CrossEntropyLoss()
            self._finetune_loss_fn = nn.CrossEntropyLoss()
            self._production_health.losses_ok = False
            self._production_health.add_issue(f"Losses setup failed: {e}")

        # DPO损失计算器
        self._dpo_loss_calculator = DPOLossCalculator(
            beta=self.three_stage_config.dpo_beta,
            loss_type=self.three_stage_config.dpo_loss_type,
            label_smoothing=self.three_stage_config.dpo_label_smoothing,
            reference_free=self.three_stage_config.dpo_reference_free
        )
    
    def _setup_training_components(self) -> None:
        """设置训练组件"""
        # 梯度累积器
        self._gradient_accumulator = GradientAccumulator(
            self.three_stage_config.gradient_accumulation_steps
        )
        
        # 收敛检测器
        self._convergence_detector = ConvergenceDetector(
            patience=self.three_stage_config.early_stopping_patience,
            threshold=self.three_stage_config.early_stopping_threshold
        )
        
        # 混合精度管理器 - 使用 backend/lib/hardware 如果可用
        if self.three_stage_config.enable_amp:
            try:
                precision_map = {
                    'fp32': PrecisionMode.FP32,
                    'fp16': PrecisionMode.MIXED_FP16,
                    'bf16': PrecisionMode.MIXED_BF16
                }
                from backend.lib.hardware.mixed_precision import AmpConfig
                amp_config = AmpConfig(
                    enabled=True,
                    precision=precision_map.get(self.three_stage_config.precision, PrecisionMode.MIXED_FP16)
                )
                self._mixed_precision_manager = MixedPrecisionManager(amp_config, self._device)
                logger.info("Using backend.lib.hardware for mixed precision")
            except Exception as e:
                logger.warning(f"Failed to use lib.hardware for AMP, falling back: {e}")
                self._mixed_precision_manager = SimpleMixedPrecisionManager(
                    enabled=torch.cuda.is_available()
                )
        else:
            # 使用简单的混合精度管理器
            self._mixed_precision_manager = SimpleMixedPrecisionManager(
                enabled=self.three_stage_config.enable_amp and torch.cuda.is_available()
            )
    
    def set_phase(self, phase: Union[ThreeStagePhase, str]) -> bool:
        """设置当前训练阶段"""
        if isinstance(phase, str):
            phase = ThreeStagePhase.from_string(phase)
        
        # 使用阶段跟踪器
        if not self._phase_tracker.start_phase(phase):
            return False
        
        self._current_phase = phase
        self._phase_steps = 0
        
        # 通知监控器
        self._three_stage_monitor.start_phase(phase)
        
        logger.info(f"ThreeStageStrategy phase set to: {phase.value}")
        return True
    
    def complete_phase(self, checkpoint_path: Optional[str] = None) -> None:
        """完成当前阶段"""
        if self._current_phase is None:
            return
        
        # 通知监控器
        self._three_stage_monitor.end_phase(self._current_phase)
        
        # 通知跟踪器
        self._phase_tracker.end_phase(self._current_phase, checkpoint_path)
        
        # 保存阶段结果
        phase_stats = self._three_stage_monitor.get_stats().get_phase_stats(self._current_phase)
        self._phase_results[self._current_phase.value] = phase_stats.to_dict()
        
        self._current_phase = None
    
    def setup_reference_model(self, model: nn.Module) -> None:
        """
        设置参考模型（用于DPO）
        
        在偏好优化阶段需要一个冻结的参考模型来计算KL散度
        """
        logger.info("Setting up reference model for DPO...")
        
        # 深拷贝模型
        self._ref_model = copy.deepcopy(model)
        self._ref_model.eval()
        
        # 冻结所有参数
        for param in self._ref_model.parameters():
            param.requires_grad = False
        
        # 移动到正确的设备
        self._ref_model = self._ref_model.to(self._device)
        
        # 更新健康状态
        self._health_status.ref_model_ok = True
        
        # 使用 lib/hardware 优化内存如果可用
        try:
            # 尝试清理缓存
            clear_memory()
            logger.info("Cleared memory cache after reference model setup")
        except Exception as e:
            logger.warning(f"Failed to clear memory: {e}")
        
        logger.info("Reference model setup completed")
    
    def cleanup_reference_model(self) -> None:
        """清理参考模型"""
        if self._ref_model is not None:
            del self._ref_model
            self._ref_model = None
            
            # 清理GPU内存
            try:
                clear_memory()
            except Exception:
                pass
            
            self._health_status.ref_model_ok = False
            logger.info("Reference model cleaned up")
    
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """
        计算损失
        
        根据当前阶段选择不同的损失计算方式
        """
        # 使用基础策略的分析器进行性能分析
        with self._profiler.profile("compute_loss"):
            if self._current_phase is None:
                # 默认使用微调损失
                result = self._compute_finetune_loss(model, batch, outputs, context)
        
        if self._current_phase == ThreeStagePhase.PRETRAIN:
                result = self._compute_pretrain_loss(model, batch, outputs, context)
        elif self._current_phase == ThreeStagePhase.FINETUNE:
                result = self._compute_finetune_loss(model, batch, outputs, context)
        elif self._current_phase == ThreeStagePhase.PREFERENCE:
                result = self._compute_preference_loss(model, batch, outputs, context)
        else:
                result = self._compute_finetune_loss(model, batch, outputs, context)
        
        # 使用基础验证器验证结果
        is_valid, message = self._validator.validate(result)
        if not is_valid:
            logger.warning(f"Loss validation failed: {message}")
            result.add_warning(message)
            self._health_status.loss_ok = False
        else:
            self._health_status.loss_ok = True
        
        # 记录到三阶段监控器
        if self._current_phase:
            self._three_stage_monitor.record_step(
                result, context, self._current_phase
            )
        
        # 记录到基础监控器
        self._monitor.record_step(result, context)
        
        return result
    
    def _compute_pretrain_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """
        计算预训练损失
        
        预训练阶段使用语言建模损失（交叉熵）
        """
        with self._get_amp_context():
            # 从outputs获取损失
            if 'loss' in outputs:
                loss = outputs['loss']
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                # 手动计算语言建模损失
                logits = outputs.get('logits', getattr(outputs, 'logits', None))
                labels = batch.get('labels', batch.get('input_ids'))
                
                if logits is not None and labels is not None:
                    # Shift for next token prediction
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    loss = self._pretrain_loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1)
                    )
                else:
                    raise ValueError("Cannot compute pretrain loss: missing logits or labels")
        
        # 计算困惑度
        perplexity = self._compute_perplexity(loss.item())
        
        self._total_steps += 1
        self._phase_steps += 1
        
        # 记录到 lib loss monitor
        if self._lib_loss_monitor is not None:
            # LossMonitor.record 接受 LossResult 对象
            loss_result = LossResult(
                loss=loss,
                metrics={'perplexity': perplexity},
                components={'pretrain_loss': loss}
            )
            self._lib_loss_monitor.record(loss_result)
        
        return StrategyResult(
            loss=loss,
            metrics={
                'loss': loss.item(),
                'perplexity': perplexity,
                'phase': 'pretrain',
                'phase_steps': self._phase_steps,
                'total_steps': self._total_steps,
            }
        )
    
    def _compute_finetune_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """
        计算监督微调损失
        
        微调阶段使用交叉熵损失进行指令跟随学习
        """
        with self._get_amp_context():
            if 'loss' in outputs:
                loss = outputs['loss']
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
            else:
                logits = outputs.get('logits', getattr(outputs, 'logits', None))
                labels = batch.get('labels')
                
                if logits is not None and labels is not None:
                    loss = self._finetune_loss_fn(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )
                else:
                    raise ValueError("Cannot compute finetune loss")
        
        self._total_steps += 1
        self._phase_steps += 1
        
        # 记录到 lib loss monitor
        if self._lib_loss_monitor is not None:
            # LossMonitor.record 接受 LossResult 对象
            loss_result = LossResult(
                loss=loss,
                metrics={},
                components={'finetune_loss': loss}
            )
            self._lib_loss_monitor.record(loss_result)
        
        return StrategyResult(
            loss=loss,
            metrics={
                'loss': loss.item(),
                'phase': 'finetune',
                'phase_steps': self._phase_steps,
                'total_steps': self._total_steps,
            }
        )
    
    def _compute_preference_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """
        计算偏好优化损失（DPO）
        
        使用Direct Preference Optimization计算损失
        """
        if self._ref_model is None and not self.three_stage_config.dpo_reference_free:
            raise ValueError("Reference model not set for DPO. Call setup_reference_model first.")
        
        with self._get_amp_context():
            loss, dpo_metrics = self._dpo_loss_calculator.compute_loss(
                model=model,
                ref_model=self._ref_model,
                batch=batch,
                device=self._device
            )
        
        self._total_steps += 1
        self._phase_steps += 1
        
        return StrategyResult(
            loss=loss,
            metrics={
                'loss': loss.item(),
                'chosen_reward': dpo_metrics.get('chosen_reward', 0.0),
                'rejected_reward': dpo_metrics.get('rejected_reward', 0.0),
                'reward_margin': dpo_metrics.get('reward_margin', 0.0),
                'accuracy': dpo_metrics.get('accuracy', 0.0),
                'phase': 'preference',
                'phase_steps': self._phase_steps,
                'total_steps': self._total_steps,
            }
        )
    
    def _get_amp_context(self):
        """获取混合精度上下文"""
        if self._mixed_precision_manager is not None:
            return self._mixed_precision_manager.autocast_context()
        return nullcontext()
    
    def _compute_perplexity(self, loss: float) -> float:
        """计算困惑度"""
        if loss < 20:  # 防止数值溢出
            return torch.exp(torch.tensor(loss)).item()
        return float('inf')
    
    def backward(self, loss: torch.Tensor) -> None:
        """反向传播"""
        with self._profiler.profile("backward"):
            # 梯度累积缩放
            scaled_loss = self._gradient_accumulator.scale_loss(loss)
        
        if self._mixed_precision_manager is not None:
            self._mixed_precision_manager.backward(scaled_loss)
        else:
            scaled_loss.backward()
    
    def optimizer_step(
        self, 
        optimizer: torch.optim.Optimizer,
        model: nn.Module
    ) -> bool:
        """
        执行优化器步骤
        
        Returns:
            是否执行了优化器步骤
        """
        if not self._gradient_accumulator.should_step():
            return False
        
        with self._profiler.profile("optimizer_step"):
            # 梯度裁剪
            if self.three_stage_config.gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.three_stage_config.gradient_clipping
                )
        
        # 优化器步骤
        if self._mixed_precision_manager is not None:
            self._mixed_precision_manager.step(optimizer)
        else:
            optimizer.step()
        
        optimizer.zero_grad()
        
        return True
    
    def check_convergence(self, loss: float) -> bool:
        """检查是否收敛"""
        converged = self._convergence_detector.update(loss)
        self._health_status.convergence_ok = not converged
        return converged
    
    def check_health(self) -> ThreeStageHealthStatus:
        """检查健康状态"""
        self._health_status.last_check_time = time.time()
        
        # 检查内存
        try:
            available = get_available_memory()
            if available < 1.0:  # 小于1GB
                self._health_status.memory_ok = False
                self._health_status.add_issue("Low available memory")
            else:
                self._health_status.memory_ok = True
        except Exception:
                pass
        
        # 检查损失趋势
        loss_trend = self._three_stage_monitor.get_phase_loss_trend(
            self._current_phase or ThreeStagePhase.FINETUNE
        )
        if loss_trend == "degrading":
            self._health_status.convergence_ok = False
            self._health_status.add_issue("Loss is degrading")
        
        # 检查参考模型（如果是DPO阶段）
        if self._current_phase == ThreeStagePhase.PREFERENCE:
            if self._ref_model is None and not self.three_stage_config.dpo_reference_free:
                self._health_status.ref_model_ok = False
                self._health_status.add_issue("Reference model not set for DPO")
        
        return self._health_status
    
    def get_phase_config(self, phase: ThreeStagePhase) -> Dict[str, Any]:
        """获取阶段配置"""
        return self.three_stage_config.get_phase_config(phase)
    
    def save_phase_result(self, phase: ThreeStagePhase, result: Dict[str, Any]) -> None:
        """保存阶段结果"""
        self._phase_results[phase.value] = result
    
    def get_phase_results(self) -> Dict[str, Dict[str, Any]]:
        """获取所有阶段结果"""
        return self._phase_results
    
    def get_progress(self) -> Dict[str, Any]:
        """获取训练进度"""
        return self._phase_tracker.get_progress()
    
    def on_phase_start(self, phase: TrainingPhase, context: StrategyContext) -> None:
        """阶段开始回调"""
        super().on_phase_start(phase, context)
        self._phase_steps = 0
        self._convergence_detector.reset()
    
    def on_phase_end(self, phase: TrainingPhase, context: StrategyContext) -> None:
        """阶段结束回调"""
        super().on_phase_end(phase, context)
        
        # 如果是DPO阶段结束，清理参考模型
        if self._current_phase == ThreeStagePhase.PREFERENCE:
            self.cleanup_reference_model()
    
    def on_step_end(self, context: StrategyContext, result: StrategyResult) -> None:
        """步骤结束回调"""
        super().on_step_end(context, result)
        
        # 更新策略指标
        step_time = time.time() - context.step_start_time if context.step_start_time > 0 else 0.0
        self.update_strategy_metrics(result, step_time)
        
        # 定期健康检查
        if self._total_steps % self.three_stage_config.health_check_interval == 0:
            self.check_health()
            self._check_production_health()
        
        # 定期内存优化
        if self._total_steps % 500 == 0:
            self.optimize_memory()
        
        # 同步分布式 - 使用 barrier 和记录统计
        if self.three_stage_config.world_size > 1:
            try:
                if self._total_steps % 100 == 0:
                    barrier()
                    self._wrapper_stats.record_sync(0.0)
            except Exception:
                pass
        
        # 日志记录（仅主进程）
        if self.should_log() and self._total_steps % self.three_stage_config.log_interval == 0:
            self._log_step_info(context, result)
    
    def _check_production_health(self) -> None:
        """检查生产级健康状态"""
        self._production_health.last_check_time = time.time()
        
        # 检查硬件层
        try:
            available = get_available_memory()
            if available < 1.0:
                self._production_health.memory_pressure = "high"
            elif available < 2.0:
                self._production_health.memory_pressure = "medium"
            else:
                self._production_health.memory_pressure = "low"
        except Exception:
                pass
        
        # 检查分布式层
        if self.three_stage_config.world_size > 1:
            try:
                # 简单健康检查
                rank = get_rank() if callable(get_rank) else 0
                world_size = get_world_size() if callable(get_world_size) else 1
                self._production_health.distributed_ok = (world_size > 0)
            except Exception:
                self._production_health.distributed_ok = False
    
    def _log_step_info(self, context: StrategyContext, result: StrategyResult) -> None:
        """记录步骤信息"""
        loss_value = result.get_loss_value()
        phase_name = self._current_phase.value if self._current_phase else "unknown"
        
        # 获取分布式信息
        dist_info = self.get_distributed_info()
        
        logger.info(
            f"Step {self._total_steps} | Phase: {phase_name} | "
            f"Loss: {loss_value:.4f} | "
            f"Rank: {dist_info['rank']}/{dist_info['world_size']}"
        )
    
    def cleanup(self) -> None:
        """清理资源"""
        super().cleanup()
        
        # 清理参考模型
        self.cleanup_reference_model()
        
        # 重置状态
        self._current_phase = None
        self._total_steps = 0
        self._phase_steps = 0
        
        logger.info("ThreeStageStrategy cleaned up")
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断策略状态"""
        # 处理收敛检测器可能为空的情况
        convergence_info = None
        if self._convergence_detector is not None:
            convergence_info = {
                'is_converged': self._convergence_detector.is_converged,
                'best_value': self._convergence_detector.best_value,
                'patience_counter': self._convergence_detector.patience_counter,
            }
        
        return {
            'name': self.name,
            'strategy_type': self._strategy_type.value,
            'current_phase': self._current_phase.value if self._current_phase else None,
            'health_status': self._health_status.to_dict(),
            'production_health': self._production_health.to_dict(),
            'progress': self._phase_tracker.get_progress(),
            'monitor_summary': self._three_stage_monitor.get_summary(),
            'strategy_metrics': self._strategy_metrics.to_dict(),
            'dpo_calculator_stats': self._dpo_loss_calculator.get_stats() if self._dpo_loss_calculator else None,
            'convergence': convergence_info,
            'layer_info': self.get_layer_info(),
            'distributed_info': self.get_distributed_info(),
            'memory_stats': self.get_memory_stats(),
            'loss_stats': self.get_loss_stats(),
            'wrapper_stats': self._wrapper_stats.to_dict() if self._wrapper_stats else None,
            'profiler_stats': self._profiler.get_stats() if hasattr(self, '_profiler') and self._profiler else None,
            'managers_health': self.check_managers_health(),
        }
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diag = self.diagnose()
        print("\n" + "=" * 60)
        print("ThreeStageStrategy Diagnosis")
        print("=" * 60)
        print(f"Name: {diag['name']}")
        print(f"Current Phase: {diag['current_phase']}")
        print(f"\nHealth Status:")
        for key, value in diag['health_status'].items():
            print(f"  {key}: {value}")
        print(f"\nProgress:")
        for key, value in diag['progress'].items():
            print(f"  {key}: {value}")
        print(f"\nLayer Info:")
        for key, value in diag['layer_info'].items():
            print(f"  {key}: {value}")
        print("=" * 60)
    
    def get_layer_info(self) -> Dict[str, Any]:
        """获取底层模块调用信息"""
        return {
            'strategy_layer': True,
            'strategy_type': self._strategy_type.value,
            'current_phase': self._current_phase.value if self._current_phase else None,
            'total_steps': self._total_steps,
            'phase_steps': self._phase_steps,
            # lib/losses 组件使用情况
            'lib_cross_entropy_used': self._lib_cross_entropy is not None,
            'lib_mse_loss_used': self._lib_mse_loss is not None,
            'lib_loss_monitor_used': self._lib_loss_monitor is not None,
            'lib_loss_stats_used': self._lib_loss_stats is not None,
            'lib_loss_factory_used': self._lib_loss_factory is not None,
            'lib_composite_loss_used': self._lib_composite_loss is not None,
            'lib_multitask_loss_used': self._lib_multitask_loss is not None,
            # lib/hardware 组件使用情况
            'device_manager_used': self._device_manager is not None,
            'memory_manager_used': self._memory_manager is not None,
            'gradient_checkpointing_used': self._gradient_checkpointing is not None,
            # lib/distributed 组件使用情况
            'distributed_manager_used': self._distributed_manager is not None,
        }
    
    def get_strategy_type(self) -> StrategyType:
        """获取策略类型"""
        return self._strategy_type
    
    def get_strategy_metrics(self) -> StrategyMetrics:
        """获取策略指标"""
        return self._strategy_metrics
    
    def get_production_health(self) -> ProductionHealthStatus:
        """获取生产级健康状态"""
        return self._production_health
    
    def get_wrapper_stats(self) -> WrapperStats:
        """获取分布式包装器统计"""
        return self._wrapper_stats
    
    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """获取内存统计 - 使用 lib/hardware"""
        try:
            stats = {}
            
            # 使用 MemoryManager 获取统计
            if self._memory_manager is not None and hasattr(self._memory_manager, 'get_stats'):
                mem_stats = self._memory_manager.get_stats()
                if isinstance(mem_stats, dict):
                    stats.update(mem_stats)
                elif hasattr(mem_stats, 'to_dict'):
                    stats.update(mem_stats.to_dict())
            
            # 使用 get_available_memory
            if get_available_memory is not None:
                stats['available_memory_gb'] = get_available_memory()
            
            # 使用 MemoryStats 如果有
            if self._memory_stats is not None:
                if hasattr(self._memory_stats, 'to_dict'):
                    stats['memory_stats'] = self._memory_stats.to_dict()
            
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to get memory stats: {e}")
            return None
    
    def get_distributed_info(self) -> Dict[str, Any]:
        """获取分布式信息 - 使用 lib/distributed"""
        info = {
            'is_distributed':  self.three_stage_config.world_size > 1,
            'world_size': 1,
            'rank': 0,
            'is_main_process': True,
        }
        
        try:
            info['world_size'] = get_world_size() if callable(get_world_size) else 1
            info['rank'] = get_rank() if callable(get_rank) else 0
            info['is_main_process'] = is_main_process() if callable(is_main_process) else True
        except Exception:
            pass
        
        # 添加 wrapper 统计
        if self._wrapper_stats is not None:
            info['wrapper_stats'] = self._wrapper_stats.to_dict()
        
        return info

    def get_loss_stats(self) -> Optional[Dict[str, Any]]:
        """获取损失统计 - 使用 lib/losses"""
        stats = {}
        
        # 使用 LossMonitor
        if self._lib_loss_monitor is not None:
            if hasattr(self._lib_loss_monitor, 'get_stats'):
                stats['monitor'] = self._lib_loss_monitor.get_stats()
            elif hasattr(self._lib_loss_monitor, 'get_summary'):
                stats['monitor'] = self._lib_loss_monitor.get_summary()
        
        # 使用 LossStats
        if self._lib_loss_stats is not None:
            if hasattr(self._lib_loss_stats, 'to_dict'):
                stats['loss_stats'] = self._lib_loss_stats.to_dict()
        
        return stats
    
    def sync_gradients(self, model: nn.Module) -> None:
        """同步梯度 - 使用 lib/distributed 的 all_reduce"""
        if self.three_stage_config.world_size <= 1:
            return
        
        if all_reduce is None or AllReduceOp is None:
            return
        
        try:
            with self._profiler.profile("sync_gradients"):
                for param in model.parameters():
                    if param.grad is not None:
                        # 使用 all_reduce 同步梯度
                        all_reduce(param.grad.data, op=AllReduceOp.SUM)
                        param.grad.data /= self.three_stage_config.world_size
                
                # 记录同步统计
                self._wrapper_stats.record_sync(0.0)  # TODO: 添加实际时间
                
        except Exception as e:
            logger.warning(f"Failed to sync gradients: {e}")
    
    def all_reduce_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """All-reduce 损失 - 使用 lib/distributed"""
        if self.three_stage_config.world_size <= 1:
            return loss
        
        if all_reduce is None or AllReduceOp is None:
            return loss
        
        try:
            # 克隆损失以避免修改原始张量
            reduced_loss = loss.clone()
            all_reduce(reduced_loss, op=AllReduceOp.SUM)
            reduced_loss = reduced_loss / self.three_stage_config.world_size
            
            # 记录统计
            self._wrapper_stats.total_all_reduces += 1
            
            return reduced_loss
            
        except Exception as e:
            logger.warning(f"Failed to all_reduce loss: {e}")
            return loss
    
    def should_log(self) -> bool:
        """检查是否应该记录日志 - 使用 lib/distributed"""
        try:
            return is_main_process() if callable(is_main_process) else True
        except Exception:
            return True
    
    def apply_gradient_checkpointing(self, model: nn.Module) -> None:
        """应用梯度检查点 - 使用 lib/hardware"""
        if self._gradient_checkpointing is None:
            logger.info("Gradient checkpointing not available, skipping")
            return
        
        try:
            if hasattr(self._gradient_checkpointing, 'apply'):
                self._gradient_checkpointing.apply(model)
                logger.info("Gradient checkpointing applied to model")
            elif hasattr(self._gradient_checkpointing, 'enable'):
                self._gradient_checkpointing.enable(model)
                logger.info("Gradient checkpointing enabled for model")
        except Exception as e:
            logger.warning(f"Failed to apply gradient checkpointing: {e}")
    
    def optimize_memory(self) -> None:
        """优化内存 - 使用 lib/hardware"""
        try:
            # 使用 MemoryManager 优化
            if self._memory_manager is not None and hasattr(self._memory_manager, 'optimize'):
                self._memory_manager.clear_cache()
            
            # 使用 clear_memory
            if clear_memory is not None:
                clear_memory()
            
            logger.debug("Memory optimized")
            
        except Exception as e:
            logger.warning(f"Failed to optimize memory: {e}")
    
    def create_training_context(self) -> Optional[ProductionTrainingContext]:
        """创建生产级训练上下文 - 使用 production_base"""
        try:
            if create_production_context is not None:
                context = create_production_context(
                    device=self.three_stage_config.device,
                    precision=self.three_stage_config.precision,
                    distributed_mode=self.three_stage_config.distributed_mode,
                )
                logger.info("ProductionTrainingContext created")
                return context
        except Exception as e:
            logger.warning(f"Failed to create production context: {e}")
        
        return None
    
    def update_strategy_metrics(self, result: StrategyResult, step_time: float = 0.0) -> None:
        """更新策略指标 - 使用 base_strategy 的 StrategyMetrics"""
        # 更新基础指标
        self._strategy_metrics.update(result, step_time)
        
        # 更新阶段指标
        if self._current_phase:
            phase = self._current_phase.to_training_phase()
            loss_value = result.get_loss_value()
            self._strategy_metrics.update_phase(phase, loss_value)
    
    def validate_with_profiling(self, result: StrategyResult) -> Tuple[bool, str]:
        """使用分析器验证结果 - 使用 StrategyProfiler 和 StrategyValidator"""
        with self._profiler.profile("validation"):
            is_valid, message = self._validator.validate(result)
        return is_valid, message
    
    def get_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        info = super().get_info()
        info.update({
            'enabled_stages': self.three_stage_config.enabled_stages,
            'current_phase': self._current_phase.value if self._current_phase else None,
            'phase_results': list(self._phase_results.keys()),
            'dpo_beta': self.three_stage_config.dpo_beta,
            'dpo_loss_type': self.three_stage_config.dpo_loss_type,
        })
        return info
    
    def get_state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        state = super().get_state_dict() if hasattr(super(), 'get_state_dict') else {}
        state.update({
            'current_phase': self._current_phase.value if self._current_phase else None,
            'phase_results': self._phase_results,
            'total_steps': self._total_steps,
            'phase_steps': self._phase_steps,
            'three_stage_stats': self._three_stage_monitor.get_stats().to_dict(),
            'convergence_detector': {
                'best_value': self._convergence_detector.best_value,
                'patience_counter': self._convergence_detector.patience_counter,
                'is_converged': self._convergence_detector.is_converged,
            },
        })
        return state
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """加载状态字典"""
        if hasattr(super(), 'load_state_dict'):
            super().load_state_dict(state)
        
        phase_value = state.get('current_phase')
        if phase_value:
            self._current_phase = ThreeStagePhase.from_string(phase_value)
        
        self._phase_results = state.get('phase_results', {})
        self._total_steps = state.get('total_steps', 0)
        self._phase_steps = state.get('phase_steps', 0)
        
        # 恢复收敛检测器
        conv_state = state.get('convergence_detector', {})
        self._convergence_detector.best_value = conv_state.get('best_value', float('inf'))
        self._convergence_detector.patience_counter = conv_state.get('patience_counter', 0)
        self._convergence_detector.is_converged = conv_state.get('is_converged', False)


# ==================== 辅助类 ====================

class GradientAccumulator:
    """梯度累积器"""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = max(1, accumulation_steps)
        self.current_step = 0
    
    def should_step(self) -> bool:
        """检查是否应该执行优化器步骤"""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失以适应梯度累积"""
        if self.accumulation_steps > 1:
            return loss / self.accumulation_steps
        return loss
    
    def reset(self):
        """重置累积计数"""
        self.current_step = 0
    
    def get_effective_batch_size(self, batch_size: int) -> int:
        """获取有效批次大小"""
        return batch_size * self.accumulation_steps


class ConvergenceDetector:
    """收敛检测器"""
    
    def __init__(
        self,
        patience: int = 5,
        threshold: float = 1e-4,
        mode: str = 'min'
    ):
        self.patience = patience
        self.threshold = threshold
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.is_converged = False
    
    def update(self, value: float) -> bool:
        """
        更新检测器
        
        Args:
            value: 当前损失/指标值
        
        Returns:
            是否应该早停
        """
        is_improved = False
        if self.mode == 'min':
            if value < self.best_value - self.threshold:
                is_improved = True
                self.best_value = value
        else:
            if value > self.best_value + self.threshold:
                is_improved = True
                self.best_value = value
        
        if is_improved:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.is_converged = True
            return True
        
        return False
    
    def reset(self):
        """重置检测器"""
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.is_converged = False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'best_value': self.best_value,
            'patience_counter': self.patience_counter,
            'is_converged': self.is_converged,
            'patience': self.patience,
            'threshold': self.threshold,
        }


class SimpleMixedPrecisionManager:
    """简单的混合精度管理器"""
    
    def __init__(self, enabled: bool = True, dtype: torch.dtype = torch.float16):
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        
        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # 统计
        self._total_steps = 0
        self._overflow_count = 0
    
    def autocast_context(self):
        """获取autocast上下文"""
        if self.enabled:
            return torch.cuda.amp.autocast(dtype=self.dtype)
        return nullcontext()
    
    def backward(self, loss: torch.Tensor) -> None:
        """反向传播"""
        if self.enabled and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
    
    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        执行优化步骤
        
        Returns:
            是否成功执行（无溢出）
        """
        self._total_steps += 1
        
        if self.enabled and self.scaler:
            # 检查溢出
            old_scale = self.scaler.get_scale()
            self.scaler.step(optimizer)
            self.scaler.update()
            new_scale = self.scaler.get_scale()
            
            if new_scale < old_scale:
                self._overflow_count += 1
                return False
            return True
        else:
            optimizer.step()
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'enabled': self.enabled,
            'total_steps': self._total_steps,
            'overflow_count': self._overflow_count,
            'overflow_rate': self._overflow_count / max(self._total_steps, 1),
            'current_scale': self.scaler.get_scale() if self.scaler else 1.0,
        }


# ==================== 便捷函数 ====================

def create_three_stage_strategy(
    config: Optional[Union[Dict[str, Any], ThreeStageStrategyConfig]] = None,
    **kwargs
) -> ThreeStageStrategy:
    """
    创建三阶段训练策略
    
    Args:
        config: 配置字典或配置对象
        **kwargs: 额外配置参数
    
    Returns:
        ThreeStageStrategy实例
    """
    if config is None:
        strategy_config = ThreeStageStrategyConfig(**kwargs)
    elif isinstance(config, dict):
        merged = {**config, **kwargs}
        strategy_config = ThreeStageStrategyConfig(**merged)
    else:
        strategy_config = config
    
    return ThreeStageStrategy(config=strategy_config)


def get_three_stage_phases() -> List[str]:
    """获取所有三阶段阶段名称"""
    return [phase.value for phase in ThreeStagePhase]


def get_phase_info(phase: Union[ThreeStagePhase, str]) -> Dict[str, Any]:
    """
    获取阶段信息
    
    Args:
        phase: 阶段名称或枚举
    
    Returns:
        阶段信息字典
    """
    if isinstance(phase, str):
        phase = ThreeStagePhase.from_string(phase)
    
    return {
        'value': phase.value,
        'stage_number': phase.stage_number,
        'typical_epochs': phase.typical_epochs,
        'typical_lr': phase.typical_lr,
        'loss_type': phase.loss_type,
        'requires_reference_model': phase.requires_reference_model,
        'description': phase.get_description(),
    }


def print_phase_info(phase: Optional[Union[ThreeStagePhase, str]] = None) -> None:
    """打印阶段信息"""
    print("\n" + "=" * 60)
    print("Three-Stage Training Phases")
    print("=" * 60)
    
    phases = [phase] if phase else list(ThreeStagePhase)
    
    for p in phases:
        if isinstance(p, str):
            p = ThreeStagePhase.from_string(p)
        
        info = get_phase_info(p)
        print(f"\n{p.value.upper()} (Stage {info['stage_number']}):")
        print(f"  Description: {info['description']}")
        print(f"  Typical LR: {info['typical_lr']}")
        print(f"  Typical Epochs: {info['typical_epochs']}")
        print(f"  Loss Type: {info['loss_type']}")
        print(f"  Requires Ref Model: {info['requires_reference_model']}")
    
    print("\n" + "=" * 60)


def diagnose_three_stage_strategy(strategy: ThreeStageStrategy) -> Dict[str, Any]:
    """诊断三阶段策略"""
    return strategy.diagnose()


def print_three_stage_diagnosis(strategy: ThreeStageStrategy) -> None:
    """打印三阶段策略诊断"""
    strategy.print_diagnosis()


def compare_dpo_loss_types() -> Dict[str, str]:
    """比较 DPO 损失类型"""
    return {
        'sigmoid': "标准 DPO 损失，使用 log-sigmoid 函数",
        'hinge': "Hinge 损失变体，更鲁棒但梯度可能消失",
        'ipo': "IPO 损失，直接优化偏好而不是 KL 散度",
    }


def print_dpo_loss_comparison() -> None:
    """打印 DPO 损失类型比较"""
    print("\n" + "=" * 60)
    print("DPO Loss Types Comparison")
    print("=" * 60)
    
    for loss_type, description in compare_dpo_loss_types().items():
        print(f"\n{loss_type}:")
        print(f"  {description}")
    
    print("\n" + "=" * 60)


def estimate_training_time(
    config: ThreeStageStrategyConfig,
    samples_per_second: float = 10.0
) -> Dict[str, float]:
    """
    估计训练时间
    
    Args:
        config: 策略配置
        samples_per_second: 每秒处理的样本数
    
    Returns:
        各阶段估计时间（小时）
    """
    estimates = {}
    
    # 假设每个 epoch 10000 步
    steps_per_epoch = 10000
    
    for phase in ThreeStagePhase:
        if phase.value not in config.enabled_stages:
            continue
        
        phase_config = config.get_phase_config(phase)
        epochs = phase_config.get('epochs', 1)
        total_steps = epochs * steps_per_epoch
        time_seconds = total_steps / samples_per_second
        estimates[phase.value] = time_seconds / 3600  # 转换为小时
    
    estimates['total'] = sum(estimates.values())
    
    return estimates


def print_training_time_estimate(
    config: ThreeStageStrategyConfig,
    samples_per_second: float = 10.0
) -> None:
    """打印训练时间估计"""
    estimates = estimate_training_time(config, samples_per_second)
    
    print("\n" + "=" * 60)
    print("Training Time Estimate")
    print(f"(Assuming {samples_per_second} samples/second)")
    print("=" * 60)
    
    for phase, hours in estimates.items():
        if phase == 'total':
            continue
        print(f"  {phase}: {hours:.2f} hours")
    
    print(f"\n  TOTAL: {estimates['total']:.2f} hours")
    print("=" * 60)


def get_layer_availability() -> Dict[str, bool]:
    """获取底层能力可用性"""
    return {
    }


def print_layer_availability() -> None:
    """打印底层能力可用性"""
    print("\n" + "=" * 60)
    print("Backend Layer Availability")
    print("=" * 60)
    
    for layer, available in get_layer_availability().items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {layer}: {status}")
    
    print("=" * 60)

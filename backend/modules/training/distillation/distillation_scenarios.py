# -*- coding: utf-8 -*-
"""
蒸馏场景管理器

提供多种业务场景的蒸馏能力：
- 标准蒸馏：基础的教师-学生蒸馏
- 行业蒸馏：针对特定行业的领域适配蒸馏
- 多模态蒸馏：跨模态知识迁移
- 边缘部署蒸馏：针对边缘设备优化的蒸馏
- 实时推理蒸馏：低延迟场景优化
- 渐进式蒸馏：逐步增加蒸馏复杂度
- 自蒸馏：模型自身层间蒸馏

架构调用层次：
├── distillation_scenarios.py (本模块)
│   └── 调用 compression_config.py (配置层)
│       ├── DistillationConfig - 蒸馏配置
│       ├── ScenarioDistillationConfig - 场景配置
│       ├── DistillationPresets - 预设模板
│       └── DistillationMonitor - 蒸馏监控
│   └── 调用 backend/modules/training/strategies (策略层)
│       ├── base_strategy.py - StrategyContext, StrategyResult, StrategyMonitor
│       ├── distributed_strategy.py - DistributedStrategy, DistributedMode
│       └── distillation_strategy.py - DistillationStrategy
│   └── 调用 backend/lib (底层)
│       ├── losses - 损失函数
│       ├── hardware - 硬件管理
│       └── distributed - 分布式管理
└── 被 distillation_service.py, knowledge_distillation.py 调用

生产级特性：
- 完整的场景监控和诊断
- 与策略层的深度集成
- 自动配置优化
- 分布式场景支持
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
import time
from datetime import datetime

# 修复导入路径
import sys
import os as os_path
current_dir = os_path.path.dirname(os_path.path.abspath(__file__))
project_root = os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(current_dir))))
sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)

# ======================== 配置层导入 ========================

from .compression_config import (
    DistillationConfig, 
    ScenarioDistillationConfig,
    DistributedDistillationConfig,
    AdaptiveDistillationConfig,
    DistillationTaskConfig,
    DistillationScenario,
    DistributedMode,
    AdaptiveMode,
    CompressionMethod,
    DistillationPresets,
    DistillationStats,
    DistillationMonitor,
    ConfigValidator,
    create_distillation_config,
    validate_config,
    recommend_config,
)

# ======================== 策略层导入 ========================

from backend.modules.training.strategies.base_strategy import (
    TrainingStrategy,
    StrategyContext,
    StrategyResult,
    TrainingPhase,
    StrategyType,
    StrategyMonitor as BaseStrategyMonitor,
    StrategyProfiler,
    StrategyValidator,
    StrategyMetrics,
)

from backend.modules.training.strategies.distributed_strategy import (
    DistributedStrategy,
    DistributedStrategyConfig,
    DistributedMode as StrategyDistributedMode,
    ZeROStage,
    DistributedHealthStatus,
    CommunicationStats,
    create_distributed_strategy,
    recommend_distributed_mode,
)

from backend.modules.training.strategies.distillation_strategy import (
    DistillationStrategy,
    DistillationStrategyConfig,
    DistillationLossCalculator,
    DistillationType,
    create_distillation_strategy,
)

# ======================== 底层 lib 模块导入 ========================

from backend.lib.losses import (
    LossFactory,
    create_loss,
    BaseLoss,
    LossMonitor as LibLossMonitor,
    LossStats as LibLossStats,
    LossResult,
)

from backend.lib.hardware import (
    DeviceManager,
    get_device_manager,
    MemoryManager,
    get_available_memory,
    clear_memory,
)

from backend.lib.distributed import (
    DistributedManager,
    get_distributed_manager,
    is_main_process,
    get_rank,
    get_world_size,
    barrier,
    all_reduce,
)


# ======================== 场景统计和监控 ========================

@dataclass
class ScenarioExecutionStats:
    """
    场景执行统计
    
    整合 base_strategy.py 的 StrategyMetrics 进行更全面的指标跟踪
    """
    scenario_name: str = ""
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    total_time_seconds: float = 0.0
    avg_time_seconds: float = 0.0
    best_accuracy: float = 0.0
    best_compression_ratio: float = 0.0
    last_run_time: Optional[str] = None
    
    # 扩展统计字段
    min_accuracy: float = float('inf')
    total_loss: float = 0.0
    min_loss: float = float('inf')
    best_loss: float = float('inf')
    total_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    total_samples: int = 0
    
    # 用于趋势分析
    _accuracy_history: List[float] = field(default_factory=list)
    _loss_history: List[float] = field(default_factory=list)
    _duration_history: List[float] = field(default_factory=list)
    
    def record_run(self, success: bool, duration: float, 
                   accuracy: float = 0.0, compression_ratio: float = 0.0,
                   loss: float = 0.0, memory_mb: float = 0.0,
                   samples: int = 0) -> None:
        """记录一次运行"""
        self.total_runs += 1
        if success:
            self.successful_runs += 1
        else:
            self.failed_runs += 1
        
        self.total_time_seconds += duration
        self.avg_time_seconds = self.total_time_seconds / self.total_runs
        
        # 更新精度统计
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        if accuracy < self.min_accuracy and accuracy > 0:
            self.min_accuracy = accuracy
        
        # 更新压缩比统计
        if compression_ratio > self.best_compression_ratio:
            self.best_compression_ratio = compression_ratio
        
        # 更新损失统计
        if loss > 0:
            self.total_loss += loss
            if loss < self.best_loss:
                self.best_loss = loss
            if loss < self.min_loss:
                self.min_loss = loss
        
        # 更新内存统计
        if memory_mb > 0:
            self.total_memory_mb += memory_mb
            if memory_mb > self.peak_memory_mb:
                self.peak_memory_mb = memory_mb
        
        # 更新样本统计
        self.total_samples += samples
        
        # 记录历史用于趋势分析
        if accuracy > 0:
            self._accuracy_history.append(accuracy)
        if loss > 0:
            self._loss_history.append(loss)
        self._duration_history.append(duration)
        
        # 限制历史大小
        max_history = 1000
        if len(self._accuracy_history) > max_history:
            self._accuracy_history = self._accuracy_history[-max_history:]
        if len(self._loss_history) > max_history:
            self._loss_history = self._loss_history[-max_history:]
        if len(self._duration_history) > max_history:
            self._duration_history = self._duration_history[-max_history:]
        
        self.last_run_time = datetime.utcnow().isoformat()
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_runs == 0:
            return 0.0
        return self.successful_runs / self.total_runs
    
    @property
    def avg_accuracy(self) -> float:
        """平均精度"""
        if not self._accuracy_history:
            return 0.0
        return sum(self._accuracy_history) / len(self._accuracy_history)
    
    @property
    def avg_loss(self) -> float:
        """平均损失"""
        if not self._loss_history:
            return 0.0
        return sum(self._loss_history) / len(self._loss_history)
    
    @property
    def avg_memory_mb(self) -> float:
        """平均内存使用"""
        if self.total_runs == 0:
            return 0.0
        return self.total_memory_mb / self.total_runs
    
    def get_accuracy_trend(self, window: int = 10) -> str:
        """获取精度趋势"""
        if len(self._accuracy_history) < window:
            return "insufficient_data"
        
        recent = self._accuracy_history[-window:]
        older = self._accuracy_history[-window*2:-window] if len(self._accuracy_history) >= window*2 else self._accuracy_history[:window]
        
        recent_avg = sum(recent) / len(recent) if recent else 0
        older_avg = sum(older) / len(older) if older else 0
        
        diff = recent_avg - older_avg
        if diff > 0.01:
            return "improving"
        elif diff < -0.01:
            return "declining"
        return "stable"
    
    def get_loss_trend(self, window: int = 10) -> str:
        """获取损失趋势"""
        if len(self._loss_history) < window:
            return "insufficient_data"
        
        recent = self._loss_history[-window:]
        older = self._loss_history[-window*2:-window] if len(self._loss_history) >= window*2 else self._loss_history[:window]
        
        recent_avg = sum(recent) / len(recent) if recent else 0
        older_avg = sum(older) / len(older) if older else 0
        
        diff = recent_avg - older_avg
        if diff < -0.01:
            return "improving"  # 损失下降是好的
        elif diff > 0.01:
            return "declining"
        return "stable"
    
    def is_converged(self, threshold: float = 0.001, window: int = 10) -> bool:
        """检查是否收敛"""
        if len(self._loss_history) < window:
            return False
        
        recent = self._loss_history[-window:]
        variance = sum((x - sum(recent)/len(recent))**2 for x in recent) / len(recent)
        return variance < threshold
    
    def to_strategy_metrics(self) -> Optional['StrategyMetrics']:
        """
        转换为 base_strategy.py 的 StrategyMetrics
        
        调用 base_strategy.py 的 StrategyMetrics
        """
        
        try:
            metrics = StrategyMetrics()
            metrics.update({
                'total_runs': self.total_runs,
                'success_rate': self.success_rate,
                'avg_accuracy': self.avg_accuracy,
                'best_accuracy': self.best_accuracy,
                'avg_loss': self.avg_loss,
                'best_loss': self.best_loss,
                'avg_time': self.avg_time_seconds,
                'total_time': self.total_time_seconds,
            })
            return metrics
        except Exception:
            return None
    
    def merge(self, other: 'ScenarioExecutionStats') -> 'ScenarioExecutionStats':
        """合并另一个统计对象"""
        return ScenarioExecutionStats(
            scenario_name=self.scenario_name,
            total_runs=self.total_runs + other.total_runs,
            successful_runs=self.successful_runs + other.successful_runs,
            failed_runs=self.failed_runs + other.failed_runs,
            total_time_seconds=self.total_time_seconds + other.total_time_seconds,
            avg_time_seconds=(self.total_time_seconds + other.total_time_seconds) / 
                            (self.total_runs + other.total_runs) if (self.total_runs + other.total_runs) > 0 else 0,
            best_accuracy=max(self.best_accuracy, other.best_accuracy),
            best_compression_ratio=max(self.best_compression_ratio, other.best_compression_ratio),
            best_loss=min(self.best_loss, other.best_loss) if self.best_loss != float('inf') and other.best_loss != float('inf') else 
                     self.best_loss if self.best_loss != float('inf') else other.best_loss,
            total_samples=self.total_samples + other.total_samples,
            peak_memory_mb=max(self.peak_memory_mb, other.peak_memory_mb),
        )
    
    def reset(self) -> None:
        """重置统计"""
        self.total_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        self.total_time_seconds = 0.0
        self.avg_time_seconds = 0.0
        self.best_accuracy = 0.0
        self.best_compression_ratio = 0.0
        self.last_run_time = None
        self.min_accuracy = float('inf')
        self.total_loss = 0.0
        self.min_loss = float('inf')
        self.best_loss = float('inf')
        self.total_memory_mb = 0.0
        self.peak_memory_mb = 0.0
        self.total_samples = 0
        self._accuracy_history.clear()
        self._loss_history.clear()
        self._duration_history.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'scenario_name': self.scenario_name,
            'total_runs': self.total_runs,
            'successful_runs': self.successful_runs,
            'failed_runs': self.failed_runs,
            'success_rate': self.success_rate,
            'total_time_seconds': self.total_time_seconds,
            'avg_time_seconds': self.avg_time_seconds,
            'best_accuracy': self.best_accuracy,
            'min_accuracy': self.min_accuracy if self.min_accuracy != float('inf') else 0.0,
            'avg_accuracy': self.avg_accuracy,
            'best_compression_ratio': self.best_compression_ratio,
            'best_loss': self.best_loss if self.best_loss != float('inf') else 0.0,
            'avg_loss': self.avg_loss,
            'peak_memory_mb': self.peak_memory_mb,
            'avg_memory_mb': self.avg_memory_mb,
            'total_samples': self.total_samples,
            'accuracy_trend': self.get_accuracy_trend(),
            'loss_trend': self.get_loss_trend(),
            'is_converged': self.is_converged(),
            'last_run_time': self.last_run_time,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioExecutionStats':
        """从字典创建"""
        stats = cls(
            scenario_name=data.get('scenario_name', ''),
            total_runs=data.get('total_runs', 0),
            successful_runs=data.get('successful_runs', 0),
            failed_runs=data.get('failed_runs', 0),
            total_time_seconds=data.get('total_time_seconds', 0.0),
            avg_time_seconds=data.get('avg_time_seconds', 0.0),
            best_accuracy=data.get('best_accuracy', 0.0),
            best_compression_ratio=data.get('best_compression_ratio', 0.0),
            last_run_time=data.get('last_run_time'),
        )
        return stats


class ScenarioMonitor:
    """
    场景监控器
    
    整合 compression_config.py 的 DistillationMonitor、
    base_strategy.py 的 StrategyMonitor 和
    distributed_strategy.py 的 DistributedHealthStatus
    """
    
    def __init__(self, scenario_name: str, history_size: int = 1000):
        self.scenario_name = scenario_name
        self.history_size = history_size
        self._stats = ScenarioExecutionStats(scenario_name=scenario_name)
        
        # 使用 compression_config.py 的监控器
        self._distillation_monitor = DistillationMonitor(history_size=history_size)
        
        # 使用 base_strategy.py 的监控器
        self._strategy_monitor: Optional['BaseStrategyMonitor'] = None
        try:
            self._strategy_monitor = BaseStrategyMonitor(history_size=history_size)
        except Exception as e:
            logger.warning(f"Failed to init BaseStrategyMonitor: {e}")
        
        # 使用 lib/losses 的监控器
        self._loss_monitor: Optional['LibLossMonitor'] = None
        try:
            self._loss_monitor = LibLossMonitor()
        except Exception:
            pass
        
        # 使用 base_strategy.py 的指标跟踪器
        self._strategy_metrics: Optional['StrategyMetrics'] = None
        try:
            self._strategy_metrics = StrategyMetrics()
        except Exception:
            pass
        
        # 使用 distributed_strategy.py 的健康状态（如果可用）
        self._distributed_health: Optional['DistributedHealthStatus'] = None
        
        # 使用 lib/hardware 的设备管理器
        self._device_manager: Optional['DeviceManager'] = None
        try:
            self._device_manager = get_device_manager()
        except Exception:
            pass
        
        # 运行历史
        self._run_history: List[Dict[str, Any]] = []
        self._current_run_start: float = 0.0
        self._current_run_memory_start: float = 0.0
        
        # 步骤计数
        self._step_count: int = 0
        self._epoch_count: int = 0
    
    def start_run(self) -> None:
        """开始一次运行"""
        self._current_run_start = time.time()
        
        # 记录运行前内存使用
        try:
            self._current_run_memory_start = get_available_memory()
        except Exception:
            self._current_run_memory_start = 0.0
        
        self._step_count = 0
    
    def start_epoch(self, epoch: int) -> None:
        """开始一个 epoch"""
        self._epoch_count = epoch
    
    def record_step(self, kd_loss: float, ce_loss: float = 0.0,
                   accuracy: float = 0.0, feature_loss: float = 0.0,
                   contrastive_loss: float = 0.0, **metrics) -> None:
        """记录一步"""
        self._step_count += 1
        total_loss = kd_loss + ce_loss + feature_loss + contrastive_loss
        
        # 使用 DistillationMonitor (compression_config.py)
        self._distillation_monitor.record_step(
            kd_loss=kd_loss,
            ce_loss=ce_loss,
            accuracy=accuracy,
        )
        
        # 同步到策略监控器 (base_strategy.py)
        if self._strategy_monitor is not None and StrategyResult is not None:
            try:
                result = StrategyResult(
                    loss=torch.tensor(total_loss),
                    metrics={
                        'kd_loss': kd_loss, 
                        'ce_loss': ce_loss, 
                        'feature_loss': feature_loss,
                        'contrastive_loss': contrastive_loss,
                        'accuracy': accuracy, 
                        **metrics
                    }
                )
                if StrategyContext is not None:
                    context = StrategyContext(
                        global_step=self._step_count,
                        epoch=self._epoch_count,
                    )
                    self._strategy_monitor.record_step(result, context)
            except Exception:
                pass
        
        # 同步到损失监控器 (lib/losses)
        if self._loss_monitor is not None:
            try:
                if LossResult is not None:
                    loss_result = LossResult(
                        loss=torch.tensor(total_loss),
                        components={
                            'kd': torch.tensor(kd_loss),
                            'ce': torch.tensor(ce_loss)
                        },
                        metrics={
                            'feature_loss': feature_loss,
                            'contrastive': contrastive_loss
                        }
                    )
                    self._loss_monitor.record(loss_result)
            except Exception:
                pass
        
        # 更新策略指标 (base_strategy.py)
        if self._strategy_metrics is not None:
            try:
                self._strategy_metrics.update({
                    'step': self._step_count,
                    'epoch': self._epoch_count,
                    'loss': total_loss,
                    'accuracy': accuracy,
                })
            except Exception:
                pass
    
    def end_run(self, success: bool, accuracy: float = 0.0, 
                compression_ratio: float = 0.0, loss: float = 0.0,
                samples: int = 0) -> None:
        """结束一次运行"""
        duration = time.time() - self._current_run_start
        
        # 计算内存使用
        memory_mb = 0.0
        try:
            memory_end = get_available_memory()
            memory_mb = self._current_run_memory_start - memory_end  # 已使用的内存
        except Exception:
            pass
        
        # 使用增强的 record_run
        self._stats.record_run(
            success=success, 
            duration=duration, 
            accuracy=accuracy, 
            compression_ratio=compression_ratio,
            loss=loss,
            memory_mb=memory_mb,
            samples=samples,
        )
        
        # 获取蒸馏统计
        distillation_stats = self._distillation_monitor.get_stats()
        
        # 记录到历史
        run_record = {
            'success': success,
            'duration': duration,
            'accuracy': accuracy,
            'compression_ratio': compression_ratio,
            'loss': loss,
            'memory_mb': memory_mb,
            'samples': samples,
            'steps': self._step_count,
            'timestamp': datetime.utcnow().isoformat(),
            'distillation_stats': distillation_stats.to_dict(),
        }
        
        # 添加策略监控摘要
        if self._strategy_monitor is not None and hasattr(self._strategy_monitor, 'get_summary'):
            try:
                run_record['strategy_summary'] = self._strategy_monitor.get_summary()
            except Exception:
                pass
        
        # 添加损失监控器统计
        if self._loss_monitor is not None and hasattr(self._loss_monitor, 'get_stats'):
            try:
                run_record['loss_stats'] = self._loss_monitor.get_stats()
            except Exception:
                pass
        
        self._run_history.append(run_record)
        
        if len(self._run_history) > self.history_size:
            self._run_history.pop(0)
    
    def get_stats(self) -> ScenarioExecutionStats:
        """获取统计数据"""
        return self._stats
    
    def get_distillation_stats(self) -> DistillationStats:
        """获取蒸馏统计 (compression_config.py)"""
        return self._distillation_monitor.get_stats()
    
    def get_strategy_metrics(self) -> Optional['StrategyMetrics']:
        """获取策略指标 (base_strategy.py)"""
        return self._strategy_metrics
    
    def get_loss_trend(self, window: int = 100) -> str:
        """获取损失趋势 (compression_config.py)"""
        return self._distillation_monitor.get_loss_trend(window=window)
    
    def is_converged(self, threshold: float = 1e-4, patience: int = 10) -> bool:
        """检查是否收敛 (compression_config.py)"""
        return self._distillation_monitor.is_converged(threshold=threshold, patience=patience)
    
    def get_run_history(self) -> List[Dict[str, Any]]:
        """获取运行历史"""
        return self._run_history.copy()
    
    def get_recent_runs(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取最近的运行记录"""
        return self._run_history[-count:]
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取内存使用情况
        
        调用 lib/hardware 的 get_available_memory
        """
        result = {
            'peak_memory_mb': self._stats.peak_memory_mb,
            'avg_memory_mb': self._stats.avg_memory_mb,
            'current_available_mb': 0.0,
        }

        try:
            result['current_available_mb'] = get_available_memory()
        except Exception:
            pass
        
        return result
    
    def sync_with_distributed(self) -> None:
        """
        与分布式进程同步
        
        调用 lib/distributed 的 barrier
        """
        try:
            barrier()
        except Exception:
            pass
    
    def should_log(self) -> bool:
        """
        检查是否应该记录日志
        
        调用 lib/distributed 的 is_main_process
        """
        return is_main_process()
    
    def get_distributed_info(self) -> Dict[str, Any]:
        """
        获取分布式信息
        
        调用 lib/distributed 的 get_rank, get_world_size
        """
        return {
            'rank': get_rank(),
            'world_size': get_world_size(),
            'is_main_process': is_main_process(),
        }
    
    def all_reduce_stats(self, stats: Dict[str, float]) -> Dict[str, float]:
        """
        跨进程聚合统计
        
        调用 lib/distributed 的 all_reduce
        """
        try:
            result = {}
            for key, value in stats.items():
                tensor = torch.tensor([value])
                all_reduce(tensor)
                result[key] = tensor.item() / get_world_size()
            return result
        except Exception:
            return stats
    
    def update_distributed_health(self, health_status: Optional['DistributedHealthStatus']) -> None:
        """
        更新分布式健康状态
        
        接收 distributed_strategy.py 的 DistributedHealthStatus
        """
        self._distributed_health = health_status
    
    def get_distributed_health(self) -> Optional[Dict[str, Any]]:
        """获取分布式健康状态"""
        if self._distributed_health is not None and hasattr(self._distributed_health, 'to_dict'):
            return self._distributed_health.to_dict()
        return None
    
    def reset(self) -> None:
        """重置监控器"""
        self._stats.reset()
        self._distillation_monitor = DistillationMonitor(history_size=self.history_size)
        self._run_history.clear()
        self._step_count = 0
        self._epoch_count = 0
        
        # 重新初始化策略监控器
        try:
            self._strategy_monitor = BaseStrategyMonitor(history_size=self.history_size)
        except Exception:
            pass
    
    def get_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        summary = {
            'scenario_name': self.scenario_name,
            'execution_stats': self._stats.to_dict(),
            'distillation_stats': self._distillation_monitor.get_stats().to_dict(),
            'loss_trend': self.get_loss_trend(),
            'is_converged': self.is_converged(),
            'step_count': self._step_count,
            'epoch_count': self._epoch_count,
            'memory_usage': self.get_memory_usage(),
            'distributed_info': self.get_distributed_info(),
            'monitors_available': {
                'distillation_monitor': True,
                'strategy_monitor': self._strategy_monitor is not None,
                'loss_monitor': self._loss_monitor is not None,
                'strategy_metrics': self._strategy_metrics is not None,
                'device_manager': self._device_manager is not None,
            },
        }
        
        # 添加策略监控摘要 (base_strategy.py)
        if self._strategy_monitor is not None and hasattr(self._strategy_monitor, 'get_summary'):
            try:
                summary['strategy_summary'] = self._strategy_monitor.get_summary()
            except Exception:
                pass
        
        # 添加分布式健康状态 (distributed_strategy.py)
        if self._distributed_health is not None:
            summary['distributed_health'] = self.get_distributed_health()
        
        # 添加策略指标 (base_strategy.py)
        if self._strategy_metrics is not None and hasattr(self._strategy_metrics, 'to_dict'):
            try:
                summary['strategy_metrics'] = self._strategy_metrics.to_dict()
            except Exception:
                pass
        
        return summary


# ======================== 场景处理器基类 ========================

class DistillationScenarioHandler(ABC):
    """
    蒸馏场景处理器基类
    
    生产级特性：
    - 与策略层集成 (base_strategy.py, distributed_strategy.py, distillation_strategy.py)
    - 与配置层集成 (compression_config.py)
    - 与 lib 层集成 (losses, hardware, distributed)
    - 完整的监控和诊断
    - 配置验证
    - 分布式支持
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # 监控器 (compression_config.py 的 DistillationMonitor + base_strategy.py 的 StrategyMonitor)
        self._monitor = ScenarioMonitor(name)
        
        # 配置验证器 (compression_config.py)
        self._validator = ConfigValidator()
        self._setup_validation_rules()
        
        # 性能分析器 (base_strategy.py)
        self._profiler: Optional['StrategyProfiler'] = None
        try:
            self._profiler = StrategyProfiler()
        except Exception:
            pass
        
        # 指标跟踪 (base_strategy.py)
        self._metrics: Optional['StrategyMetrics'] = None
        try:
            self._metrics = StrategyMetrics()
        except Exception:
            pass
        
        # 策略验证器 (base_strategy.py)
        self._strategy_validator: Optional['StrategyValidator'] = None
        try:
            self._strategy_validator = StrategyValidator()
        except Exception:
            pass
        
        # 设备管理器 (lib/hardware)
        self._device_manager: Optional['DeviceManager'] = None
        try:
            self._device_manager = get_device_manager()
        except Exception:
            pass
        
        # 内存管理器 (lib/hardware)
        self._memory_manager: Optional['MemoryManager'] = None
        try:
            self._memory_manager = MemoryManager()
        except Exception:
            pass
        
        # 分布式管理器 (lib/distributed)
        self._distributed_manager: Optional['DistributedManager'] = None

        try:
            self._distributed_manager = get_distributed_manager()
        except Exception:
            pass
        
        # 损失工厂 (lib/losses)
        self._loss_factory: Optional['LossFactory'] = None
        try:
            self._loss_factory = LossFactory()
        except Exception:
            pass
        
        # 当前策略和配置
        self._current_strategy: Optional['DistillationStrategy'] = None
        self._current_config: Optional[DistillationTaskConfig] = None
        self._distributed_strategy: Optional['DistributedStrategy'] = None
        
        # 通信统计 (distributed_strategy.py)
        self._communication_stats: Optional['CommunicationStats'] = None
        
        # 状态
        self._is_initialized: bool = False
        self._is_running: bool = False
    
    def _setup_validation_rules(self) -> None:
        """
        设置验证规则
        
        子类可以覆盖添加特定规则
        """
        # 添加基本验证规则到 ConfigValidator (使用 add_check 方法)
        def check_teacher_model_path(cfg) -> Tuple[bool, str]:
            if cfg.distillation_config and cfg.distillation_config.teacher_model_path == '':
                return False, "Teacher model path is required"
            return True, ""
        
        def check_student_model_path(cfg) -> Tuple[bool, str]:
            if cfg.distillation_config and cfg.distillation_config.student_model_path == '':
                return False, "Student model path is required"
            return True, ""
        
        self._validator.add_check(check_teacher_model_path)
        self._validator.add_check(check_student_model_path)
    
    @abstractmethod
    def prepare(self, config: DistillationTaskConfig) -> Dict[str, Any]:
        """准备场景所需的资源和配置"""
        pass
    
    @abstractmethod
    def get_strategy(self, config: DistillationTaskConfig) -> Optional['DistillationStrategy']:
        """获取该场景对应的蒸馏策略"""
        pass
    
    @abstractmethod
    def post_process(self, model: nn.Module, result: Dict[str, Any]) -> nn.Module:
        """后处理：对蒸馏后的模型进行场景特定处理"""
        pass
    
    def initialize(self, config: DistillationTaskConfig) -> bool:
        """
        初始化场景处理器
        
        调用 compression_config.py 的验证和策略层的设置
        """
        if self._is_initialized:
            return True
        
        # 验证配置
        is_valid, errors = self.validate_config(config)
        if not is_valid:
            self.logger.error(f"Configuration validation failed: {errors}")
            return False
        
        # 保存当前配置
        self._current_config = config
        
        # 使用性能分析器记录初始化过程
        if self._profiler is not None:
            with self._profiler.profile("initialize"):
                self._do_initialize(config)
        else:
            self._do_initialize(config)
        
        self._is_initialized = True
        return True
    
    def _do_initialize(self, config: DistillationTaskConfig) -> None:
        """执行初始化逻辑"""
        # 创建分布式策略（如果需要）
        if config.distributed_config and config.distributed_config.is_distributed():
            self._distributed_strategy = self.create_distributed_strategy(config)
            
            # 同步进程
            self.sync_processes()
        
        # 创建蒸馏策略
        self._current_strategy = self.get_strategy(config)
        
        # 优化内存
        self.optimize_memory()
    
    def validate_config(self, config: DistillationTaskConfig) -> Tuple[bool, List[str]]:
        """
        验证配置是否适用于该场景
        
        使用 compression_config.py 的 validate_config 和 ConfigValidator
        """
        errors = []
        
        # 使用 compression_config.py 的全局验证
        is_valid, config_errors = validate_config(config)
        errors.extend(config_errors)
        
        # 使用本地验证器
        local_valid, local_errors = self._validator.validate(config)
        if not local_valid:
            errors.extend(local_errors)
        
        # 使用策略层验证器 (base_strategy.py)
        if self._strategy_validator is not None:
            try:
                is_valid, strategy_errors = self._strategy_validator.validate(config)
                if strategy_errors:
                    errors.extend(strategy_errors)
            except Exception:
                pass
        
        return len(errors) == 0, errors
    
    def get_monitor(self) -> ScenarioMonitor:
        """获取监控器"""
        return self._monitor
    
    def get_profiler(self) -> Optional['StrategyProfiler']:
        """获取性能分析器 (base_strategy.py)"""
        return self._profiler
    
    def get_metrics(self) -> Optional['StrategyMetrics']:
        """获取指标跟踪器 (base_strategy.py)"""
        return self._metrics
    
    def get_validator(self) -> ConfigValidator:
        """获取配置验证器 (compression_config.py)"""
        return self._validator
    
    def get_device_manager(self) -> Optional['DeviceManager']:
        """获取设备管理器 (lib/hardware)"""
        return self._device_manager
    
    def get_memory_manager(self) -> Optional['MemoryManager']:
        """获取内存管理器 (lib/hardware)"""
        return self._memory_manager
    
    def get_distributed_manager(self) -> Optional['DistributedManager']:
        """获取分布式管理器 (lib/distributed)"""
        return self._distributed_manager
    
    def get_loss_factory(self) -> Optional['LossFactory']:
        """获取损失工厂 (lib/losses)"""
        return self._loss_factory
    
    def create_strategy_context(self, config: DistillationTaskConfig) -> Optional['StrategyContext']:
        """
        创建策略上下文
        
        使用 base_strategy.py 的 StrategyContext
        """
        try:
            # 确定设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self._device_manager is not None and hasattr(self._device_manager, 'get_device'):
                try:
                    device = self._device_manager.get_device()
                except Exception:
                    pass
            
            # 创建上下文
            context = StrategyContext(
                device=device,
                config={
                    'scenario': self.name,
                    'task_name': config.task_name,
                    'task_id': config.task_id,
                    'distillation_type': config.distillation_config.distillation_type if config.distillation_config else 'logits',
                },
                max_epochs=config.num_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
            )
            
            return context
        except Exception as e:
            self.logger.warning(f"Failed to create StrategyContext: {e}")
            return None
    
    def create_distributed_strategy(self, config: DistillationTaskConfig) -> Optional['DistributedStrategy']:
        """
        创建分布式策略
        
        使用 distributed_strategy.py 的 create_distributed_strategy
        """
        if config.distributed_config and config.distributed_config.is_distributed():
            try:
                # 使用性能分析器
                if self._profiler is not None:
                    with self._profiler.profile("create_distributed_strategy"):
                        strategy = create_distributed_strategy(
                            mode=config.distributed_config.mode,
                            world_size=config.distributed_config.world_size,
                            rank=config.distributed_config.rank,
                        )
                else:
                    strategy = create_distributed_strategy(
                        mode=config.distributed_config.mode,
                        world_size=config.distributed_config.world_size,
                        rank=config.distributed_config.rank,
                    )
                
                return strategy
            except Exception as e:
                self.logger.warning(f"Failed to create DistributedStrategy: {e}")
        
        return None
    
    def create_distillation_strategy(self, strategy_type: str = 'standard', 
                                     **kwargs) -> Optional['DistillationStrategy']:
        """
        创建蒸馏策略
        
        使用 distillation_strategy.py 的 create_distillation_strategy
        """
        try:
            if self._profiler is not None:
                with self._profiler.profile(f"create_distillation_strategy_{strategy_type}"):
                    return create_distillation_strategy(strategy_type, **kwargs)
            return create_distillation_strategy(strategy_type, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create distillation strategy: {e}")
            return None
    
    def create_loss_function(self, loss_type: str = 'kl_div', **kwargs) -> Optional['BaseLoss']:
        """
        创建损失函数
        
        使用 lib/losses 的 LossFactory 和 create_loss
        """
        try:
            if self._loss_factory is not None and hasattr(self._loss_factory, 'create'):
                return self._loss_factory.create(loss_type, **kwargs)
            elif create_loss is not None:
                return create_loss(loss_type, **kwargs)
        except Exception as e:
            self.logger.warning(f"Failed to create loss function: {e}")
        
        return None
    
    def optimize_memory(self) -> None:
        """
        优化内存
        
        使用 backend/lib/hardware
        """
        try:
            clear_memory()
            self.logger.debug("Memory cleared")
        except Exception:
            pass
        
        # 使用内存管理器进行更深度的优化
        if self._memory_manager is not None:
            try:
                if hasattr(self._memory_manager, 'clear_memory'):
                    self._memory_manager.clear_memory()
            except Exception:
                pass
    
    def get_available_memory(self) -> float:
        """
        获取可用内存
        
        使用 lib/hardware 的 get_available_memory
        """
        try:
            return get_available_memory()
        except Exception:
            pass

        return 0.0
    
    def sync_processes(self) -> None:
        """
        同步进程
        
        使用 backend/lib/distributed
        """

        try:
            barrier()
        except Exception:
            pass
    
    def all_reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        跨进程聚合张量
        
        使用 lib/distributed 的 all_reduce
        """
        try:
            all_reduce(tensor)
            return tensor / get_world_size()
        except Exception:
            return tensor
    
    def should_log(self) -> bool:
        """
        检查是否应该记录日志
        
        使用 lib/distributed 的 is_main_process
        """
        return is_main_process()
    
    def get_rank(self) -> int:
        """
        获取当前进程的 rank
        
        使用 lib/distributed 的 get_rank
        """
        return get_rank()
    
    def get_world_size(self) -> int:
        """
        获取总进程数
        
        使用 lib/distributed 的 get_world_size
        """
        get_world_size()
    
    def recommend_mode(self, model_size_gb: float, num_gpus: int) -> str:
        """
        推荐分布式模式
        
        使用 distributed_strategy.py 的 recommend_distributed_mode
        """
        try:
            return recommend_distributed_mode(model_size_gb, num_gpus)
        except Exception:
            return 'data_parallel'
    
    def get_communication_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取通信统计
        
        使用 distributed_strategy.py 的 CommunicationStats
        """
        try:
            stats = self._distributed_strategy.get_communication_stats()
            if isinstance(stats, dict):
                return stats
            
            # 使用 getattr 避免 Pylint 错误
            to_dict_method = getattr(stats, 'to_dict', None)
            if callable(to_dict_method):
                return to_dict_method()  # pylint: disable=not-callable
        except Exception:
            pass
        return None
    
    def update_metrics(self, **kwargs) -> None:
        """
        更新指标
        
        使用 base_strategy.py 的 StrategyMetrics
        """
        if self._metrics is not None:
            try:
                self._metrics.update(kwargs)
            except Exception:
                pass
    
    def profile_operation(self, operation_name: str):
        """
        性能分析上下文管理器
        
        使用 base_strategy.py 的 StrategyProfiler
        """
        if self._profiler is not None:
            return self._profiler.profile(operation_name)
        
        # 返回空上下文管理器
        from contextlib import nullcontext
        return nullcontext()
    
    def record_step(self, kd_loss: float, ce_loss: float = 0.0, 
                   accuracy: float = 0.0, **metrics) -> None:
        """
        记录一个训练步骤
        
        同时更新监控器和策略指标
        """
        # 更新监控器
        self._monitor.record_step(kd_loss, ce_loss, accuracy, **metrics)
        
        # 更新策略指标
        self.update_metrics(
            kd_loss=kd_loss,
            ce_loss=ce_loss,
            accuracy=accuracy,
            **metrics,
        )
    
    def get_training_phase(self) -> Optional['TrainingPhase']:
        """
        获取当前训练阶段
        
        使用 base_strategy.py 的 TrainingPhase
        """
        # 根据当前状态返回相应的训练阶段
        if not self._is_initialized:
            return TrainingPhase.WARMUP
        elif self._is_running:
            return TrainingPhase.MAIN
        else:
            return TrainingPhase.EVALUATION
    
    def get_strategy_type(self) -> Optional['StrategyType']:
        """
        获取策略类型
        
        使用 base_strategy.py 的 StrategyType
        """
        try:
            return StrategyType.DISTILLATION
        except Exception:
            pass
    
    def cleanup(self) -> None:
        """清理资源"""
        self._is_running = False
        self._is_initialized = False
        self._current_strategy = None
        self._distributed_strategy = None
        
        # 优化内存
        self.optimize_memory()
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断场景状态"""
        diagnosis = {
            'scenario_name': self.name,
            'is_initialized': self._is_initialized,
            'is_running': self._is_running,
            'monitor_summary': self._monitor.get_summary(),
            'profiler_available': self._profiler is not None,
            'metrics_available': self._metrics is not None,
            'strategy_validator_available': self._strategy_validator is not None,
            'device_manager_available': self._device_manager is not None,
            'memory_manager_available': self._memory_manager is not None,
            'distributed_manager_available': self._distributed_manager is not None,
            'loss_factory_available': self._loss_factory is not None,
            'current_strategy': self._current_strategy is not None,
            'distributed_strategy': self._distributed_strategy is not None,
        }
        
        # 添加性能分析数据
        if self._profiler is not None:
            try:
                diagnosis['profiler_summary'] = self._profiler.get_stats()
            except Exception:
                pass
        
        # 添加指标数据
        if self._metrics is not None and hasattr(self._metrics, 'to_dict'):
            try:
                diagnosis['metrics'] = self._metrics.to_dict()
            except Exception:
                pass
        
        # 添加内存信息
        diagnosis['available_memory_mb'] = self.get_available_memory()
        
        # 添加分布式信息
        diagnosis['distributed_info'] = {
            'rank': self.get_rank(),
            'world_size': self.get_world_size(),
            'is_main_process': self.should_log(),
        }
        
        # 添加通信统计
        comm_stats = self.get_communication_stats()
        if comm_stats:
            diagnosis['communication_stats'] = comm_stats
        
        return diagnosis


# ======================== 具体场景处理器 ========================

class StandardScenarioHandler(DistillationScenarioHandler):
    """
    标准蒸馏场景处理器
    
    基础的教师-学生蒸馏，适用于大多数场景
    
    特性：
    - 支持 logits、feature、attention 蒸馏
    - 集成策略层 (base_strategy.py, distributed_strategy.py)
    - 集成配置层 (compression_config.py)
    - 完整的监控和诊断
    """
    
    def __init__(self):
        super().__init__("standard")
        
        # 标准场景特定配置
        self._default_temperature = 4.0
        self._default_soft_loss_weight = 0.7
        self._default_hard_loss_weight = 0.3
    
    def _setup_validation_rules(self) -> None:
        """设置标准场景的验证规则"""
        super()._setup_validation_rules()
        
        # 添加标准场景特定的验证规则
        def check_temperature_range(cfg) -> Tuple[bool, str]:
            if cfg.distillation_config:
                temp = cfg.distillation_config.temperature
                if not (1.0 <= temp <= 20.0):
                    return False, "Temperature should be between 1.0 and 20.0"
            return True, ""
        
        self._validator.add_check(check_temperature_range)
    
    def prepare(self, config: DistillationTaskConfig) -> Dict[str, Any]:
        """
        准备标准蒸馏场景
        
        调用 compression_config.py 和策略层
        """
        self._monitor.start_run()
        self._is_running = True
        
        if self.should_log():
            self.logger.info("Preparing standard distillation scenario")
        
        # 使用性能分析器 (base_strategy.py)
        with self.profile_operation("prepare_standard"):
            # 验证并初始化
            if not self._is_initialized:
                self.initialize(config)
            
            # 获取优化后的配置 (compression_config.py)
            optimized = {}
            if config.scenario_config:
                optimized = config.scenario_config.get_optimized_config()
            
            # 创建策略上下文 (base_strategy.py)
            strategy_context = self.create_strategy_context(config)
            
            # 获取推荐配置 (compression_config.py)
            recommended = recommend_config(
                model_size_gb=config.distillation_config.get('model_size_gb', 1.0) if config.distillation_config else 1.0,
                target_accuracy=0.95,
            ) if recommend_config else None
            
            # 创建损失函数 (lib/losses)
            loss_fn = self.create_loss_function('kl_div', temperature=self._default_temperature)
            
            # 创建分布式策略（如果需要）(distributed_strategy.py)
            distributed_strategy = None
            if config.distributed_config and config.distributed_config.is_distributed():
                distributed_strategy = self.create_distributed_strategy(config)
            
            # 同步进程 (lib/distributed)
            self.sync_processes()
            
            # 更新指标 (base_strategy.py)
            self.update_metrics(
                scenario='standard',
                temperature=self._default_temperature,
                soft_weight=self._default_soft_loss_weight,
                hard_weight=self._default_hard_loss_weight,
            )
        
        return {
            'prepared': True,
            'scenario': 'standard',
            'optimized_config': optimized,
            'strategy_context': strategy_context,
            'loss_function': loss_fn,
            'distributed_strategy': distributed_strategy,
            'recommended_config': recommended.task_name if recommended else None,
            'temperature': self._default_temperature,
            'soft_loss_weight': self._default_soft_loss_weight,
            'hard_loss_weight': self._default_hard_loss_weight,
            'available_memory_mb': self.get_available_memory(),
        }
    
    def get_strategy(self, config: DistillationTaskConfig) -> Optional['DistillationStrategy']:
        """
        获取标准蒸馏策略
        
        使用 distillation_strategy.py 的 create_distillation_strategy
        """
        try:
            # 获取配置参数
            temperature = self._default_temperature
            if config.distillation_config:
                temperature = config.distillation_config.temperature
            
            # 构建策略配置
            strategy_config = {
                'temperature': temperature,
                'soft_loss_weight': self._default_soft_loss_weight,
                'hard_loss_weight': self._default_hard_loss_weight,
                'distillation_type': 'logits',
            }
            
            # 使用 profiler 进行性能分析 (base_strategy.py)
            with self.profile_operation("create_strategy"):
                strategy = create_distillation_strategy('standard', config=strategy_config)
            
            self._current_strategy = strategy
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to create strategy: {e}")
            return None
    
    def post_process(self, model: nn.Module, result: Dict[str, Any]) -> nn.Module:
        """后处理标准蒸馏模型"""
        self._is_running = False
        
        # 记录运行结果
        self._monitor.end_run(
            success=result.get('success', True),
            accuracy=result.get('accuracy', 0.0),
            compression_ratio=result.get('compression_ratio', 0.0),
            loss=result.get('final_loss', 0.0),
            samples=result.get('total_samples', 0),
        )
        
        # 优化内存 (lib/hardware)
        self.optimize_memory()
        
        # 同步进程 (lib/distributed)
        self.sync_processes()
        
        return model
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'temperature': self._default_temperature,
            'soft_loss_weight': self._default_soft_loss_weight,
            'hard_loss_weight': self._default_hard_loss_weight,
            'distillation_type': 'logits',
        }
    
    def estimate_memory_requirement(self, config: DistillationTaskConfig) -> float:
        """
        估算内存需求
        
        使用 compression_config.py 的配置和 lib/hardware
        """
        if config.distillation_config:
            return config.distillation_config.estimate_memory_mb()
        return 1000.0  # 默认 1GB


class IndustryScenarioHandler(DistillationScenarioHandler):
    """
    行业蒸馏场景处理器
    
    针对特定行业（制造、金融、医疗等）的领域适配蒸馏
    
    特性：
    - 行业特定的蒸馏配置
    - 特征层蒸馏
    - 领域适配
    - 集成策略层和配置层
    """
    
    def __init__(self):
        # 先初始化 industry_configs，再调用父类构造函数
        self.industry_configs = {
            'manufacturing': {
                'temperature': 4.0,
                'domain_loss_weight': 0.15,
                'feature_layers': [-1, -2, -3],
                'soft_loss_weight': 0.6,
                'hard_loss_weight': 0.25,
                'feature_loss_weight': 0.15,
                'distillation_type': 'combined',
                'recommended_compression': 2.0,
                'description': '制造业场景，重视特征蒸馏',
            },
            'finance': {
                'temperature': 3.0,
                'domain_loss_weight': 0.2,
                'feature_layers': [-1, -2],
                'soft_loss_weight': 0.5,
                'hard_loss_weight': 0.3,
                'feature_loss_weight': 0.2,
                'distillation_type': 'combined',
                'recommended_compression': 1.5,
                'description': '金融场景，重视精度保持',
            },
            'healthcare': {
                'temperature': 2.5,
                'domain_loss_weight': 0.25,
                'feature_layers': [-1, -2, -3, -4],
                'soft_loss_weight': 0.4,
                'hard_loss_weight': 0.35,
                'feature_loss_weight': 0.25,
                'distillation_type': 'combined',
                'recommended_compression': 1.3,
                'description': '医疗场景，全层蒸馏，精度优先',
            },
            'retail': {
                'temperature': 4.5,
                'domain_loss_weight': 0.1,
                'feature_layers': [-1, -2],
                'soft_loss_weight': 0.7,
                'hard_loss_weight': 0.2,
                'feature_loss_weight': 0.1,
                'distillation_type': 'logits',
                'recommended_compression': 3.0,
                'description': '零售场景，轻量化优先',
            },
            'energy': {
                'temperature': 3.5,
                'domain_loss_weight': 0.18,
                'feature_layers': [-1, -2, -3],
                'soft_loss_weight': 0.55,
                'hard_loss_weight': 0.27,
                'feature_loss_weight': 0.18,
                'distillation_type': 'combined',
                'recommended_compression': 2.0,
                'description': '能源场景，平衡蒸馏',
            },
            'automotive': {
                'temperature': 3.0,
                'domain_loss_weight': 0.2,
                'feature_layers': [-1, -2, -3],
                'soft_loss_weight': 0.5,
                'hard_loss_weight': 0.3,
                'feature_loss_weight': 0.2,
                'distillation_type': 'combined',
                'recommended_compression': 2.5,
                'description': '汽车场景，安全和实时性优先',
            },
        }
        
        # 当前选择的行业
        self._current_industry: str = 'manufacturing'
        
        # 调用父类构造函数
        super().__init__("industry")
    
    def _setup_validation_rules(self) -> None:
        """设置行业场景的验证规则"""
        super()._setup_validation_rules()
        
        # 添加行业特定的验证规则
        valid_industries = list(self.industry_configs.keys())
        def check_industry_type(cfg) -> Tuple[bool, str]:
            if cfg.scenario_config and cfg.scenario_config.industry_type:
                if cfg.scenario_config.industry_type not in valid_industries:
                    return False, f"Industry type should be one of {valid_industries}"
            return True, ""
        
        self._validator.add_check(check_industry_type)
    
    def prepare(self, config: DistillationTaskConfig) -> Dict[str, Any]:
        """
        准备行业蒸馏场景
        
        调用 compression_config.py、策略层和 lib 层
        """
        self._monitor.start_run()
        self._is_running = True
        
        # 确定行业类型
        industry_type = 'manufacturing'
        if config.scenario_config and config.scenario_config.industry_type:
            industry_type = config.scenario_config.industry_type
        self._current_industry = industry_type
        
        if self.should_log():
            self.logger.info(f"Preparing industry distillation: {industry_type}")
        
        with self.profile_operation("prepare_industry"):
            # 初始化
            if not self._is_initialized:
                self.initialize(config)
            
            # 获取行业特定配置
            industry_cfg = self.industry_configs.get(industry_type, self.industry_configs['manufacturing'])
            
            # 获取场景优化配置 (compression_config.py)
            optimized = {}
            if config.scenario_config:
                optimized = config.scenario_config.get_optimized_config()
            
            # 创建策略上下文 (base_strategy.py)
            strategy_context = self.create_strategy_context(config)
            
            # 创建多个损失函数 (lib/losses)
            loss_functions = {}
            loss_functions['kl_div'] = self.create_loss_function('kl_div', temperature=industry_cfg['temperature'])
            loss_functions['feature'] = self.create_loss_function('mse')
            
            # 创建分布式策略 (distributed_strategy.py)
            distributed_strategy = None
            if config.distributed_config and config.distributed_config.is_distributed():
                distributed_strategy = self.create_distributed_strategy(config)
            
            # 同步进程 (lib/distributed)
            self.sync_processes()
            
            # 更新指标 (base_strategy.py)
            self.update_metrics(
                scenario='industry',
                industry_type=industry_type,
                temperature=industry_cfg['temperature'],
                domain_loss_weight=industry_cfg['domain_loss_weight'],
            )
        
        return {
            'prepared': True,
            'scenario': 'industry',
            'industry_type': industry_type,
            'industry_config': industry_cfg,
            'optimized_config': optimized,
            'strategy_context': strategy_context,
            'loss_functions': loss_functions,
            'distributed_strategy': distributed_strategy,
            'feature_layers': industry_cfg['feature_layers'],
            'available_memory_mb': self.get_available_memory(),
        }
    
    def get_strategy(self, config: DistillationTaskConfig) -> Optional['DistillationStrategy']:
        """
        获取行业蒸馏策略
        
        使用 distillation_strategy.py 的 create_distillation_strategy
        """
        industry_type = self._current_industry
        if config.scenario_config and config.scenario_config.industry_type:
            industry_type = config.scenario_config.industry_type
        
        industry_cfg = self.industry_configs.get(industry_type, self.industry_configs['manufacturing'])
        
        try:
            strategy_config = {
                'temperature': industry_cfg.get('temperature', 4.0),
                'distillation_type': industry_cfg.get('distillation_type', 'combined'),
                'feature_layers': industry_cfg.get('feature_layers', [-1, -2, -3]),
                'feature_loss_type': 'cosine',
                'soft_loss_weight': industry_cfg.get('soft_loss_weight', 0.6),
                'hard_loss_weight': industry_cfg.get('hard_loss_weight', 0.25),
                'feature_loss_weight': industry_cfg.get('feature_loss_weight', 0.15),
            }
            
            with self.profile_operation("create_industry_strategy"):
                strategy = create_distillation_strategy('industry', config=strategy_config)
            
            self._current_strategy = strategy
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to create industry strategy: {e}")
            return None
    
    def post_process(self, model: nn.Module, result: Dict[str, Any]) -> nn.Module:
        """后处理行业蒸馏模型"""
        self._is_running = False
        
        # 添加行业标签
        industry_type = result.get('industry_type', self._current_industry)
        if hasattr(model, 'config'):
            model.config.domain = industry_type
            model.config.industry_optimized = True
        
        # 记录运行结果
        self._monitor.end_run(
            success=result.get('success', True),
            accuracy=result.get('accuracy', 0.0),
            compression_ratio=result.get('compression_ratio', 0.0),
            loss=result.get('final_loss', 0.0),
            samples=result.get('total_samples', 0),
        )
        
        # 优化内存 (lib/hardware)
        self.optimize_memory()
        
        # 同步进程 (lib/distributed)
        self.sync_processes()
        
        return model
    
    def get_available_industries(self) -> List[str]:
        """获取支持的行业列表"""
        return list(self.industry_configs.keys())
    
    def get_industry_description(self, industry_type: str) -> str:
        """获取行业描述"""
        cfg = self.industry_configs.get(industry_type, {})
        return cfg.get('description', '未知行业')
    
    def get_industry_config(self, industry_type: str) -> Dict[str, Any]:
        """获取行业完整配置"""
        return self.industry_configs.get(industry_type, self.industry_configs['manufacturing']).copy()
    
    def register_industry(self, industry_type: str, config: Dict[str, Any]) -> None:
        """注册新的行业配置"""
        required_keys = ['temperature', 'domain_loss_weight', 'feature_layers', 'description']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")
        
        self.industry_configs[industry_type] = config
        self.logger.info(f"Registered new industry: {industry_type}")
    
    def recommend_industry_config(self, model_size_gb: float, accuracy_requirement: float) -> str:
        """
        根据需求推荐行业配置
        
        使用 compression_config.py 的推荐逻辑
        """
        if accuracy_requirement > 0.98:
            return 'healthcare'  # 精度要求最高
        elif accuracy_requirement > 0.95:
            return 'finance'  # 高精度
        elif model_size_gb > 10:
            return 'retail'  # 需要更多压缩
        else:
            return 'manufacturing'  # 默认平衡配置
    
    def estimate_compression_ratio(self, industry_type: str) -> float:
        """估算行业配置的压缩比"""
        cfg = self.industry_configs.get(industry_type, {})
        return cfg.get('recommended_compression', 2.0)


class EdgeDeployScenarioHandler(DistillationScenarioHandler):
    """
    边缘部署蒸馏场景处理器
    
    针对边缘设备（手机、IoT设备等）优化的蒸馏
    
    特性：
    - 动态量化支持
    - 延迟优化
    - 内存优化
    - 模型压缩集成
    - 支持多种边缘设备目标
    """
    
    def __init__(self):
        # 先初始化属性，再调用父类构造函数
        # 边缘设备配置
        self._device_configs = {
            'cpu': {
                'temperature': 6.0,
                'soft_loss_weight': 0.9,
                'hard_loss_weight': 0.1,
                'target_compression': 4.0,
                'quantization_bits': 8,
            },
            'mobile': {
                'temperature': 7.0,
                'soft_loss_weight': 0.85,
                'hard_loss_weight': 0.15,
                'target_compression': 6.0,
                'quantization_bits': 8,
            },
            'edge': {
                'temperature': 8.0,
                'soft_loss_weight': 0.8,
                'hard_loss_weight': 0.2,
                'target_compression': 8.0,
                'quantization_bits': 4,
            },
            'iot': {
                'temperature': 10.0,
                'soft_loss_weight': 0.75,
                'hard_loss_weight': 0.25,
                'target_compression': 10.0,
                'quantization_bits': 4,
            },
        }
        
        # 当前目标设备
        self._target_device = 'cpu'
        self._target_latency_ms = 100.0
        self._enable_quantization = True
        
        # 调用父类构造函数
        super().__init__("edge_deploy")
    
    def _setup_validation_rules(self) -> None:
        """设置边缘部署场景的验证规则"""
        super()._setup_validation_rules()
        
        # 添加边缘部署特定的验证规则
        def check_target_latency(cfg) -> Tuple[bool, str]:
            if cfg.scenario_config:
                if cfg.scenario_config.target_latency_ms <= 0:
                    return False, "Target latency must be positive"
            return True, ""
        
        self._validator.add_check(check_target_latency)
    
    def prepare(self, config: DistillationTaskConfig) -> Dict[str, Any]:
        """
        准备边缘部署蒸馏场景
        
        调用 compression_config.py、策略层和 lib 层
        """
        self._monitor.start_run()
        self._is_running = True
        
        # 获取配置参数
        target_device = 'cpu'
        target_latency = 100.0
        enable_quantization = True
        
        if config.scenario_config:
            target_device = config.scenario_config.target_device
            target_latency = config.scenario_config.target_latency_ms
            enable_quantization = config.scenario_config.enable_quantization
        
        self._target_device = target_device
        self._target_latency_ms = target_latency
        self._enable_quantization = enable_quantization
        
        if self.should_log():
            self.logger.info(f"Preparing edge deployment: device={target_device}, latency={target_latency}ms")
        
        with self.profile_operation("prepare_edge_deploy"):
            # 初始化
            if not self._is_initialized:
                self.initialize(config)
            
            # 获取优化配置 (compression_config.py)
            optimized = {}
            if config.scenario_config:
                optimized = config.scenario_config.get_optimized_config()
            
            # 估算压缩比 (compression_config.py)
            compression_ratio = 1.0
            if config.scenario_config:
                compression_ratio = config.scenario_config.estimate_compression_ratio()
            
            # 获取设备特定配置
            device_cfg = self._device_configs.get(target_device, self._device_configs['cpu'])
            
            # 使用预设获取边缘部署配置 (compression_config.py)
            edge_preset = DistillationPresets.get('edge_deployment')
            
            # 创建策略上下文 (base_strategy.py)
            strategy_context = self.create_strategy_context(config)
            
            # 创建损失函数 (lib/losses)
            loss_fn = self.create_loss_function('kl_div', temperature=device_cfg['temperature'])
            
            # 创建分布式策略 (distributed_strategy.py)
            distributed_strategy = None
            if config.distributed_config and config.distributed_config.is_distributed():
                distributed_strategy = self.create_distributed_strategy(config)
            
            # 检查可用内存 (lib/hardware)
            available_memory = self.get_available_memory()
            
            # 同步进程 (lib/distributed)
            self.sync_processes()
            
            # 更新指标 (base_strategy.py)
            self.update_metrics(
                scenario='edge_deploy',
                target_device=target_device,
                target_latency_ms=target_latency,
                enable_quantization=enable_quantization,
                temperature=device_cfg['temperature'],
            )
        
        return {
            'prepared': True,
            'scenario': 'edge_deploy',
            'target_device': target_device,
            'target_latency_ms': target_latency,
            'enable_quantization': enable_quantization,
            'optimized_config': optimized,
            'estimated_compression_ratio': compression_ratio,
            'device_config': device_cfg,
            'edge_preset': edge_preset.task_name if edge_preset else None,
            'strategy_context': strategy_context,
            'loss_function': loss_fn,
            'distributed_strategy': distributed_strategy,
            'available_memory_mb': available_memory,
        }
    
    def get_strategy(self, config: DistillationTaskConfig) -> Optional['DistillationStrategy']:
        """
        获取边缘部署蒸馏策略
        
        使用 distillation_strategy.py 的 create_distillation_strategy
        """
        try:
            # 获取设备特定配置
            device_cfg = self._device_configs.get(self._target_device, self._device_configs['cpu'])
            
            # 边缘部署使用更激进的蒸馏配置
            strategy_config = {
                'temperature': device_cfg['temperature'],
                'soft_loss_weight': device_cfg['soft_loss_weight'],
                'hard_loss_weight': device_cfg['hard_loss_weight'],
                'distillation_type': 'logits',
                'feature_loss_weight': 0.05,
            }
            
            with self.profile_operation("create_edge_deploy_strategy"):
                strategy = create_distillation_strategy('standard', config=strategy_config)
            
            self._current_strategy = strategy
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to create edge deploy strategy: {e}")
            return None
    
    def post_process(self, model: nn.Module, result: Dict[str, Any]) -> nn.Module:
        """后处理边缘部署模型"""
        self._is_running = False
        
        with self.profile_operation("post_process_edge"):
            # 应用量化
            if result.get('enable_quantization', self._enable_quantization):
                model = self._apply_quantization(model)
            
            # 应用剪枝（如果配置了）
            if result.get('enable_pruning', False):
                model = self._apply_pruning(model, result.get('pruning_ratio', 0.3))
            
            # 添加边缘部署元数据
            if hasattr(model, 'config'):
                model.config.edge_optimized = True
                model.config.target_device = self._target_device
                model.config.target_latency_ms = self._target_latency_ms
        
        # 优化内存 (lib/hardware)
        self.optimize_memory()
        
        # 记录运行结果
        self._monitor.end_run(
            success=result.get('success', True),
            accuracy=result.get('accuracy', 0.0),
            compression_ratio=result.get('compression_ratio', 0.0),
            loss=result.get('final_loss', 0.0),
            samples=result.get('total_samples', 0),
        )
        
        # 同步进程 (lib/distributed)
        self.sync_processes()
        
        return model
    
    def _apply_quantization(self, model: nn.Module) -> nn.Module:
        """应用动态量化"""
        try:
            device_cfg = self._device_configs.get(self._target_device, {})
            quantization_bits = device_cfg.get('quantization_bits', 8)
            
            # 使用 PyTorch 动态量化
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},
                dtype=torch.qint8
            )
            
            if self.should_log():
                self.logger.info(f"Applied {quantization_bits}-bit dynamic quantization for edge deployment")
            
            return quantized_model
        except Exception as e:
            self.logger.warning(f"Quantization failed: {e}")
            return model
    
    def _apply_pruning(self, model: nn.Module, pruning_ratio: float) -> nn.Module:
        """应用结构化剪枝"""
        try:
            import torch.nn.utils.prune as prune
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                    prune.remove(module, 'weight')
            
            if self.should_log():
                self.logger.info(f"Applied {pruning_ratio:.1%} pruning for edge deployment")
            
            return model
        except Exception as e:
            self.logger.warning(f"Pruning failed: {e}")
            return model
    
    def get_device_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取所有设备配置"""
        return self._device_configs.copy()
    
    def register_device(self, device_name: str, config: Dict[str, Any]) -> None:
        """注册新的边缘设备配置"""
        required_keys = ['temperature', 'soft_loss_weight', 'hard_loss_weight', 'target_compression']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")
        
        self._device_configs[device_name] = config
        self.logger.info(f"Registered new edge device: {device_name}")
    
    def estimate_latency(self, model: nn.Module, input_shape: Tuple[int, ...]) -> float:
        """估算模型在目标设备上的延迟"""
        try:
            # 简单的延迟估算
            param_count = sum(p.numel() for p in model.parameters())
            device_cfg = self._device_configs.get(self._target_device, {})
            compression = device_cfg.get('target_compression', 4.0)
            
            # 粗略估算：每百万参数约 1ms（经过压缩后）
            estimated_latency = (param_count / 1e6) / compression
            return estimated_latency
        except Exception:
            return self._target_latency_ms
    
    def meets_latency_requirement(self, model: nn.Module, input_shape: Tuple[int, ...]) -> bool:
        """检查是否满足延迟要求"""
        estimated = self.estimate_latency(model, input_shape)
        return estimated <= self._target_latency_ms


class MultimodalScenarioHandler(DistillationScenarioHandler):
    """
    多模态蒸馏场景处理器
    
    支持跨模态知识迁移
    
    特性：
    - 多模态蒸馏（文本、图像、音频、视频）
    - 跨模态对比学习
    - 模态特定的蒸馏配置
    - 模态权重平衡
    """
    
    def __init__(self):
        # 先初始化属性，再调用父类构造函数
        # 模态配置
        self._modality_configs = {
            'text': {
                'temperature': 4.0,
                'soft_loss_weight': 0.6,
                'feature_loss_weight': 0.2,
                'contrastive_weight': 0.2,
                'encoder_type': 'transformer',
            },
            'image': {
                'temperature': 5.0,
                'soft_loss_weight': 0.5,
                'feature_loss_weight': 0.3,
                'contrastive_weight': 0.2,
                'encoder_type': 'vit',
            },
            'audio': {
                'temperature': 4.5,
                'soft_loss_weight': 0.55,
                'feature_loss_weight': 0.25,
                'contrastive_weight': 0.2,
                'encoder_type': 'wav2vec',
            },
            'video': {
                'temperature': 5.5,
                'soft_loss_weight': 0.5,
                'feature_loss_weight': 0.3,
                'contrastive_weight': 0.2,
                'encoder_type': 'timesformer',
            },
        }
        
        # 跨模态配置
        self._cross_modal_configs = {
            ('text', 'image'): {
                'alignment_method': 'contrastive',
                'alignment_weight': 0.3,
                'temperature': 0.07,
            },
            ('text', 'audio'): {
                'alignment_method': 'contrastive',
                'alignment_weight': 0.25,
                'temperature': 0.1,
            },
            ('image', 'audio'): {
                'alignment_method': 'optimal_transport',
                'alignment_weight': 0.2,
                'temperature': 0.1,
            },
        }
        
        # 当前模态配置
        self._current_modalities: List[str] = ['text']
        self._cross_modal_enabled: bool = False
        
        # 调用父类构造函数
        super().__init__("multimodal")
    
    def _setup_validation_rules(self) -> None:
        """设置多模态场景的验证规则"""
        super()._setup_validation_rules()
        
        # 添加多模态特定的验证规则
        supported_modalities = list(self._modality_configs.keys())
        def check_modalities(cfg) -> Tuple[bool, str]:
            if cfg.scenario_config and cfg.scenario_config.modalities:
                for m in cfg.scenario_config.modalities:
                    if m not in supported_modalities:
                        return False, f"Modalities must be from {supported_modalities}"
            return True, ""
        
        self._validator.add_check(check_modalities)
    
    def prepare(self, config: DistillationTaskConfig) -> Dict[str, Any]:
        """
        准备多模态蒸馏场景
        
        调用 compression_config.py、策略层和 lib 层
        """
        self._monitor.start_run()
        self._is_running = True
        
        # 获取配置参数
        modalities = ['text']
        cross_modal = False
        modality_weights = {}
        
        if config.scenario_config:
            modalities = config.scenario_config.modalities
            cross_modal = config.scenario_config.cross_modal_distillation
            modality_weights = config.scenario_config.modality_weights
        
        self._current_modalities = modalities
        self._cross_modal_enabled = cross_modal
        
        if self.should_log():
            self.logger.info(f"Preparing multimodal distillation: {modalities}")
        
        with self.profile_operation("prepare_multimodal"):
            # 初始化
            if not self._is_initialized:
                self.initialize(config)
            
            # 获取优化配置 (compression_config.py)
            optimized = {}
            if config.scenario_config:
                optimized = config.scenario_config.get_optimized_config()
            
            # 获取多模态预设 (compression_config.py)
            multimodal_preset = DistillationPresets.get('multimodal')
            
            # 构建模态特定配置
            modal_configs = {}
            for modality in modalities:
                modal_configs[modality] = self._modality_configs.get(
                    modality, self._modality_configs['text']
                )
            
            # 应用模态权重
            if modality_weights:
                for modality, weight in modality_weights.items():
                    if modality in modal_configs:
                        modal_configs[modality]['weight'] = weight
            
            # 获取跨模态配置
            cross_modal_config = {}
            if cross_modal and len(modalities) >= 2:
                for i, m1 in enumerate(modalities):
                    for m2 in modalities[i+1:]:
                        key = (m1, m2) if (m1, m2) in self._cross_modal_configs else (m2, m1)
                        if key in self._cross_modal_configs:
                            cross_modal_config[f"{m1}_{m2}"] = self._cross_modal_configs[key]
            
            # 创建策略上下文 (base_strategy.py)
            strategy_context = self.create_strategy_context(config)
            
            # 创建损失函数 (lib/losses)
            loss_functions = {}
            for modality in modalities:
                cfg = modal_configs[modality]
                loss_functions[f'{modality}_kl'] = self.create_loss_function(
                    'kl_div', temperature=cfg['temperature']
                )
            
            # 如果启用跨模态，创建对比损失
            if cross_modal:
                loss_functions['contrastive'] = self.create_loss_function('contrastive')
            
            # 创建分布式策略 (distributed_strategy.py)
            distributed_strategy = None
            if config.distributed_config and config.distributed_config.is_distributed():
                distributed_strategy = self.create_distributed_strategy(config)
            
            # 检查可用内存 (lib/hardware)
            available_memory = self.get_available_memory()
            
            # 同步进程 (lib/distributed)
            self.sync_processes()
            
            # 更新指标 (base_strategy.py)
            self.update_metrics(
                scenario='multimodal',
                modalities=modalities,
                cross_modal=cross_modal,
                num_modalities=len(modalities),
            )
        
        return {
            'prepared': True,
            'scenario': 'multimodal',
            'modalities': modalities,
            'cross_modal': cross_modal,
            'modality_weights': modality_weights,
            'modal_configs': modal_configs,
            'cross_modal_config': cross_modal_config,
            'optimized_config': optimized,
            'multimodal_preset': multimodal_preset.task_name if multimodal_preset else None,
            'strategy_context': strategy_context,
            'loss_functions': loss_functions,
            'distributed_strategy': distributed_strategy,
            'available_memory_mb': available_memory,
        }
    
    def get_strategy(self, config: DistillationTaskConfig) -> Optional['DistillationStrategy']:
        """
        获取多模态蒸馏策略
        
        使用 distillation_strategy.py 的 create_distillation_strategy
        """
        try:
            # 计算混合配置
            avg_temperature = sum(
                self._modality_configs[m]['temperature'] 
                for m in self._current_modalities
            ) / len(self._current_modalities)
            
            avg_contrastive = sum(
                self._modality_configs[m]['contrastive_weight'] 
                for m in self._current_modalities
            ) / len(self._current_modalities)
            
            strategy_config = {
                'temperature': avg_temperature,
                'distillation_type': 'combined',
                'feature_loss_type': 'cosine',
                'contrastive_weight': avg_contrastive if self._cross_modal_enabled else 0.0,
                'soft_loss_weight': 0.5,
                'feature_loss_weight': 0.3,
                'hard_loss_weight': 0.2,
            }
            
            with self.profile_operation("create_multimodal_strategy"):
                strategy = create_distillation_strategy('contrastive', config=strategy_config)
            
            self._current_strategy = strategy
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to create multimodal strategy: {e}")
            return None
    
    def post_process(self, model: nn.Module, result: Dict[str, Any]) -> nn.Module:
        """后处理多模态蒸馏模型"""
        self._is_running = False
        
        # 添加多模态元数据
        if hasattr(model, 'config'):
            model.config.modalities = self._current_modalities
            model.config.cross_modal_enabled = self._cross_modal_enabled
            model.config.multimodal_optimized = True
        
        # 记录运行结果
        self._monitor.end_run(
            success=result.get('success', True),
            accuracy=result.get('accuracy', 0.0),
            compression_ratio=result.get('compression_ratio', 0.0),
            loss=result.get('final_loss', 0.0),
            samples=result.get('total_samples', 0),
        )
        
        # 优化内存 (lib/hardware)
        self.optimize_memory()
        
        # 同步进程 (lib/distributed)
        self.sync_processes()
        
        return model
    
    def get_supported_modalities(self) -> List[str]:
        """获取支持的模态列表"""
        return list(self._modality_configs.keys())
    
    def get_modality_config(self, modality: str) -> Dict[str, Any]:
        """获取模态配置"""
        return self._modality_configs.get(modality, {}).copy()
    
    def register_modality(self, modality: str, config: Dict[str, Any]) -> None:
        """注册新的模态配置"""
        required_keys = ['temperature', 'soft_loss_weight', 'feature_loss_weight', 'contrastive_weight']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")
        
        self._modality_configs[modality] = config
        self.logger.info(f"Registered new modality: {modality}")
    
    def get_cross_modal_config(self, modality1: str, modality2: str) -> Optional[Dict[str, Any]]:
        """获取跨模态配置"""
        key1 = (modality1, modality2)
        key2 = (modality2, modality1)
        
        if key1 in self._cross_modal_configs:
            return self._cross_modal_configs[key1].copy()
        elif key2 in self._cross_modal_configs:
            return self._cross_modal_configs[key2].copy()
        return None
    
    def estimate_memory_requirement(self, modalities: List[str]) -> float:
        """估算多模态蒸馏的内存需求"""
        # 基础内存
        base_memory = 2000.0  # 2GB 基础
        
        # 每个模态额外内存
        modality_memory = {
            'text': 500.0,
            'image': 1500.0,
            'audio': 1000.0,
            'video': 3000.0,
        }
        
        total = base_memory
        for modality in modalities:
            total += modality_memory.get(modality, 500.0)
        
        # 跨模态额外开销
        if len(modalities) >= 2:
            total *= 1.2
        
        return total


class RealTimeScenarioHandler(DistillationScenarioHandler):
    """
    实时推理蒸馏场景处理器
    
    优化低延迟场景
    
    特性：
    - 延迟优化
    - 流式推理支持
    - 批处理大小优化
    - 模型编译优化
    """
    
    def __init__(self):
        # 先初始化属性，再调用父类构造函数
        # 实时场景配置
        self._latency_configs = {
            'ultra_low': {  # <10ms
                'temperature': 6.0,
                'soft_loss_weight': 0.85,
                'hard_loss_weight': 0.15,
                'max_batch_size': 1,
                'target_latency_ms': 10,
            },
            'low': {  # 10-50ms
                'temperature': 5.0,
                'soft_loss_weight': 0.8,
                'hard_loss_weight': 0.2,
                'max_batch_size': 4,
                'target_latency_ms': 50,
            },
            'medium': {  # 50-100ms
                'temperature': 4.0,
                'soft_loss_weight': 0.75,
                'hard_loss_weight': 0.25,
                'max_batch_size': 8,
                'target_latency_ms': 100,
            },
        }
        
        self._current_latency_tier = 'low'
        self._streaming_mode = False
        
        # 调用父类构造函数
        super().__init__("real_time")
    
    def _setup_validation_rules(self) -> None:
        """设置实时场景的验证规则"""
        super()._setup_validation_rules()
        
        def check_max_batch_size(cfg) -> Tuple[bool, str]:
            if cfg.scenario_config:
                if cfg.scenario_config.max_batch_size < 1:
                    return False, "Max batch size must be at least 1"
            return True, ""
        
        self._validator.add_check(check_max_batch_size)
    
    def prepare(self, config: DistillationTaskConfig) -> Dict[str, Any]:
        """
        准备实时推理蒸馏场景
        
        调用 compression_config.py、策略层和 lib 层
        """
        self._monitor.start_run()
        self._is_running = True
        
        max_batch_size = 1
        streaming_mode = False
        target_latency = 50.0
        
        if config.scenario_config:
            max_batch_size = config.scenario_config.max_batch_size
            streaming_mode = config.scenario_config.streaming_mode
            target_latency = config.scenario_config.target_latency_ms
        
        self._streaming_mode = streaming_mode
        
        # 确定延迟级别
        if target_latency < 10:
            self._current_latency_tier = 'ultra_low'
        elif target_latency < 50:
            self._current_latency_tier = 'low'
        else:
            self._current_latency_tier = 'medium'
        
        if self.should_log():
            self.logger.info(f"Preparing real-time inference distillation: tier={self._current_latency_tier}")
        
        with self.profile_operation("prepare_real_time"):
            # 初始化
            if not self._is_initialized:
                self.initialize(config)
            
            # 获取延迟级别配置
            latency_cfg = self._latency_configs.get(self._current_latency_tier, self._latency_configs['low'])
            
            # 获取低延迟预设 (compression_config.py)
            low_latency_preset = DistillationPresets.get('low_latency')
            real_time_preset = DistillationPresets.get('real_time')
            
            # 创建策略上下文 (base_strategy.py)
            strategy_context = self.create_strategy_context(config)
            
            # 创建损失函数 (lib/losses)
            loss_fn = self.create_loss_function('kl_div', temperature=latency_cfg['temperature'])
            
            # 创建分布式策略 (distributed_strategy.py)
            distributed_strategy = None
            if config.distributed_config and config.distributed_config.is_distributed():
                distributed_strategy = self.create_distributed_strategy(config)
            
            # 同步进程 (lib/distributed)
            self.sync_processes()
            
            # 更新指标 (base_strategy.py)
            self.update_metrics(
                scenario='real_time',
                latency_tier=self._current_latency_tier,
                streaming_mode=streaming_mode,
                max_batch_size=max_batch_size,
            )
        
        return {
            'prepared': True,
            'scenario': 'real_time',
            'max_batch_size': max_batch_size,
            'streaming_mode': streaming_mode,
            'latency_tier': self._current_latency_tier,
            'latency_config': latency_cfg,
            'strategy_context': strategy_context,
            'loss_function': loss_fn,
            'distributed_strategy': distributed_strategy,
            'presets': {
                'low_latency': low_latency_preset.task_name if low_latency_preset else None,
                'real_time': real_time_preset.task_name if real_time_preset else None,
            },
            'available_memory_mb': self.get_available_memory(),
        }
    
    def get_strategy(self, config: DistillationTaskConfig) -> Optional['DistillationStrategy']:
        """获取实时推理蒸馏策略"""
        try:
            latency_cfg = self._latency_configs.get(self._current_latency_tier, self._latency_configs['low'])
            
            strategy_config = {
                'temperature': latency_cfg['temperature'],
                'distillation_type': 'logits',
                'soft_loss_weight': latency_cfg['soft_loss_weight'],
                'hard_loss_weight': latency_cfg['hard_loss_weight'],
            }
            
            with self.profile_operation("create_real_time_strategy"):
                strategy = create_distillation_strategy('standard', config=strategy_config)
            
            self._current_strategy = strategy
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to create real-time strategy: {e}")
            return None
    
    def post_process(self, model: nn.Module, result: Dict[str, Any]) -> nn.Module:
        """后处理实时推理模型"""
        self._is_running = False
        
        with self.profile_operation("post_process_real_time"):
            # 优化模型用于推理
            model.eval()
            
            # 尝试编译模型（如果支持）
            if hasattr(torch, 'compile') and result.get('enable_compile', True):
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    if self.should_log():
                        self.logger.info("Applied torch.compile for inference optimization")
                except Exception:
                    pass
            
            # 添加实时元数据
            if hasattr(model, 'config'):
                model.config.real_time_optimized = True
                model.config.latency_tier = self._current_latency_tier
                model.config.streaming_mode = self._streaming_mode
        
        # 记录运行结果
        self._monitor.end_run(
            success=result.get('success', True),
            accuracy=result.get('accuracy', 0.0),
            compression_ratio=result.get('compression_ratio', 0.0),
            loss=result.get('final_loss', 0.0),
        )
        
        # 优化内存 (lib/hardware)
        self.optimize_memory()
        
        return model
    
    def get_latency_tiers(self) -> List[str]:
        """获取延迟级别列表"""
        return list(self._latency_configs.keys())
    
    def get_latency_config(self, tier: str) -> Dict[str, Any]:
        """获取延迟级别配置"""
        return self._latency_configs.get(tier, {}).copy()


class ProgressiveScenarioHandler(DistillationScenarioHandler):
    """
    渐进式蒸馏场景处理器
    
    逐步增加蒸馏的层数和复杂度
    
    特性：
    - 分阶段蒸馏
    - 渐进式层选择
    - 温度调度
    - 损失权重调度
    """
    
    def __init__(self):
        # 先初始化属性，再调用父类构造函数
        self._current_stage = 0
        self._total_stages = 4
        
        # 阶段配置
        self._stage_configs = {
            0: {
                'feature_layers': [-1],
                'temperature': 6.0,
                'soft_loss_weight': 0.9,
                'hard_loss_weight': 0.1,
                'feature_loss_weight': 0.0,
                'description': '第一阶段：仅最后一层',
            },
            1: {
                'feature_layers': [-1, -2],
                'temperature': 5.0,
                'soft_loss_weight': 0.8,
                'hard_loss_weight': 0.15,
                'feature_loss_weight': 0.05,
                'description': '第二阶段：最后两层',
            },
            2: {
                'feature_layers': [-1, -2, -3],
                'temperature': 4.0,
                'soft_loss_weight': 0.7,
                'hard_loss_weight': 0.2,
                'feature_loss_weight': 0.1,
                'description': '第三阶段：最后三层',
            },
            3: {
                'feature_layers': [-1, -2, -3, -4],
                'temperature': 3.0,
                'soft_loss_weight': 0.6,
                'hard_loss_weight': 0.25,
                'feature_loss_weight': 0.15,
                'description': '第四阶段：最后四层',
            },
            4: {
                'feature_layers': [-1, -2, -3, -4, -5],
                'temperature': 2.5,
                'soft_loss_weight': 0.5,
                'hard_loss_weight': 0.3,
                'feature_loss_weight': 0.2,
                'description': '第五阶段：全层蒸馏',
            },
        }
        
        # 阶段进度
        self._stage_progress: Dict[int, float] = {}
        self._warmup_steps_per_stage = 500
        
        # 调用父类构造函数
        super().__init__("progressive")
    
    def _setup_validation_rules(self) -> None:
        """设置渐进式场景的验证规则"""
        super()._setup_validation_rules()
        
        def check_progressive_stages(cfg) -> Tuple[bool, str]:
            if cfg.scenario_config:
                stages = cfg.scenario_config.progressive_stages
                if not (1 <= stages <= 5):
                    return False, "Progressive stages must be between 1 and 5"
            return True, ""
        
        self._validator.add_check(check_progressive_stages)
    
    def prepare(self, config: DistillationTaskConfig) -> Dict[str, Any]:
        """
        准备渐进式蒸馏场景
        
        调用 compression_config.py、策略层和 lib 层
        """
        self._monitor.start_run()
        self._is_running = True
        
        stages = 3
        if config.scenario_config:
            stages = config.scenario_config.progressive_stages
        
        self._total_stages = stages
        self._current_stage = 0
        self._stage_progress = {i: 0.0 for i in range(stages)}
        
        if self.should_log():
            self.logger.info(f"Preparing progressive distillation with {stages} stages")
        
        with self.profile_operation("prepare_progressive"):
            # 初始化
            if not self._is_initialized:
                self.initialize(config)
            
            # 获取渐进式预设 (compression_config.py)
            progressive_preset = DistillationPresets.get('progressive')
            
            # 获取当前阶段配置
            stage_cfg = self._stage_configs.get(0, self._stage_configs[0])
            
            # 创建策略上下文 (base_strategy.py)
            strategy_context = self.create_strategy_context(config)
            
            # 为每个阶段创建损失函数 (lib/losses)
            stage_loss_functions = {}
            for stage in range(stages):
                stage_config = self._stage_configs.get(stage, self._stage_configs[0])
                stage_loss_functions[stage] = self.create_loss_function(
                    'kl_div', temperature=stage_config['temperature']
                )
            
            # 创建分布式策略 (distributed_strategy.py)
            distributed_strategy = None
            if config.distributed_config and config.distributed_config.is_distributed():
                distributed_strategy = self.create_distributed_strategy(config)
            
            # 同步进程 (lib/distributed)
            self.sync_processes()
            
            # 更新指标 (base_strategy.py)
            self.update_metrics(
                scenario='progressive',
                total_stages=stages,
                current_stage=0,
            )
        
        return {
            'prepared': True,
            'scenario': 'progressive',
            'stages': stages,
            'warmup_per_stage': self._warmup_steps_per_stage,
            'current_stage': self._current_stage,
            'stage_config': stage_cfg,
            'all_stage_configs': {i: self._stage_configs.get(i, {}) for i in range(stages)},
            'strategy_context': strategy_context,
            'stage_loss_functions': stage_loss_functions,
            'distributed_strategy': distributed_strategy,
            'progressive_preset': progressive_preset.task_name if progressive_preset else None,
            'available_memory_mb': self.get_available_memory(),
        }
    
    def get_strategy(self, config: DistillationTaskConfig) -> Optional['DistillationStrategy']:
        """获取渐进式蒸馏策略"""
        try:
            stage_cfg = self._stage_configs.get(self._current_stage, self._stage_configs[0])
            
            strategy_config = {
                'temperature': stage_cfg['temperature'],
                'distillation_type': 'combined',
                'feature_layers': stage_cfg['feature_layers'],
                'soft_loss_weight': stage_cfg['soft_loss_weight'],
                'hard_loss_weight': stage_cfg['hard_loss_weight'],
                'feature_loss_weight': stage_cfg['feature_loss_weight'],
            }
            
            with self.profile_operation("create_progressive_strategy"):
                strategy = create_distillation_strategy('progressive', config=strategy_config)
            
            self._current_strategy = strategy
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to create progressive strategy: {e}")
            return None
    
    def post_process(self, model: nn.Module, result: Dict[str, Any]) -> nn.Module:
        """后处理渐进式蒸馏模型"""
        self._is_running = False
        
        # 添加渐进式元数据
        if hasattr(model, 'config'):
            model.config.progressive_optimized = True
            model.config.total_stages = self._total_stages
            model.config.completed_stages = self._current_stage + 1
        
        # 记录运行结果
        self._monitor.end_run(
            success=result.get('success', True),
            accuracy=result.get('accuracy', 0.0),
            compression_ratio=result.get('compression_ratio', 0.0),
            loss=result.get('final_loss', 0.0),
        )
        
        # 优化内存 (lib/hardware)
        self.optimize_memory()
        
        return model
    
    def advance_stage(self) -> int:
        """前进到下一阶段"""
        if self._current_stage < self._total_stages - 1:
            self._current_stage += 1
            
            # 更新指标 (base_strategy.py)
            self.update_metrics(current_stage=self._current_stage)
            
            # 更新策略
            self._current_strategy = None  # 强制重新创建策略
            
            if self.should_log():
                self.logger.info(f"Advanced to stage {self._current_stage}")
        
        return self._current_stage
    
    def get_current_stage(self) -> int:
        """获取当前阶段"""
        return self._current_stage
    
    def get_total_stages(self) -> int:
        """获取总阶段数"""
        return self._total_stages
    
    def get_stage_config(self, stage: int) -> Dict[str, Any]:
        """获取阶段配置"""
        return self._stage_configs.get(stage, self._stage_configs[0]).copy()
    
    def update_stage_progress(self, progress: float) -> None:
        """更新当前阶段进度"""
        self._stage_progress[self._current_stage] = progress
    
    def get_overall_progress(self) -> float:
        """获取整体进度"""
        if self._total_stages == 0:
            return 0.0
        
        completed = sum(1.0 for p in self._stage_progress.values() if p >= 1.0)
        current = self._stage_progress.get(self._current_stage, 0.0)
        
        return (completed + current) / self._total_stages
    
    def should_advance_stage(self, loss: float, accuracy: float, step: int) -> bool:
        """判断是否应该前进到下一阶段"""
        if self._current_stage >= self._total_stages - 1:
            return False
        
        # 检查是否完成了预热步骤
        if step < self._warmup_steps_per_stage * (self._current_stage + 1):
            return False
        
        # 检查损失是否收敛
        if self._monitor.is_converged():
            return True
        
        return False
    
    def get_stage_description(self, stage: int) -> str:
        """获取阶段描述"""
        cfg = self._stage_configs.get(stage, {})
        return cfg.get('description', f'Stage {stage}')


class SelfDistillationScenarioHandler(DistillationScenarioHandler):
    """
    自蒸馏场景处理器
    
    模型自身不同层之间的知识蒸馏，无需教师模型
    
    特性：
    - 层间知识蒸馏
    - 无需单独的教师模型
    - 支持多种蒸馏方式（深层到浅层、分类器辅助）
    - 特征匹配
    """
    
    def __init__(self):
        # 先初始化属性，再调用父类构造函数
        # 自蒸馏模式配置
        self._mode_configs = {
            'deep_to_shallow': {
                'source_layers': [-1],
                'target_layers': [-2, -3, -4],
                'temperature': 4.0,
                'loss_weight': 0.5,
                'description': '深层到浅层蒸馏',
            },
            'classifier_assisted': {
                'source_layers': [-1],
                'target_layers': [-2],
                'temperature': 3.0,
                'loss_weight': 0.3,
                'use_auxiliary_classifiers': True,
                'description': '分类器辅助蒸馏',
            },
            'multi_exit': {
                'source_layers': [-1],
                'target_layers': [-2, -4, -6],
                'temperature': 4.5,
                'loss_weight': 0.4,
                'description': '多出口蒸馏',
            },
        }
        
        self._current_mode = 'deep_to_shallow'
        
        # 调用父类构造函数
        super().__init__("self")
    
    def _setup_validation_rules(self) -> None:
        """设置自蒸馏场景的验证规则"""
        super()._setup_validation_rules()
        
        # 自蒸馏不需要教师模型（空验证规则）
        def check_self_distillation(cfg) -> Tuple[bool, str]:
            # 自蒸馏场景不需要教师模型路径
            return True, ""
        
        self._validator.add_check(check_self_distillation)
    
    def prepare(self, config: DistillationTaskConfig) -> Dict[str, Any]:
        """
        准备自蒸馏场景
        
        调用 compression_config.py、策略层和 lib 层
        """
        self._monitor.start_run()
        self._is_running = True
        
        if self.should_log():
            self.logger.info("Preparing self-distillation")
        
        with self.profile_operation("prepare_self_distillation"):
            # 初始化
            if not self._is_initialized:
                self.initialize(config)
            
            # 获取自蒸馏预设 (compression_config.py)
            self_preset = DistillationPresets.get('self_distillation')
            
            # 获取当前模式配置
            mode_cfg = self._mode_configs.get(self._current_mode, self._mode_configs['deep_to_shallow'])
            
            # 创建策略上下文 (base_strategy.py)
            strategy_context = self.create_strategy_context(config)
            
            # 创建损失函数 (lib/losses)
            loss_fn = self.create_loss_function('mse')
            kl_loss = self.create_loss_function('kl_div', temperature=mode_cfg['temperature'])
            
            # 创建分布式策略 (distributed_strategy.py)
            distributed_strategy = None
            if config.distributed_config and config.distributed_config.is_distributed():
                distributed_strategy = self.create_distributed_strategy(config)
            
            # 同步进程 (lib/distributed)
            self.sync_processes()
            
            # 更新指标 (base_strategy.py)
            self.update_metrics(
                scenario='self',
                mode=self._current_mode,
                source_layers=mode_cfg['source_layers'],
                target_layers=mode_cfg['target_layers'],
            )
        
        return {
            'prepared': True,
            'scenario': 'self',
            'mode': self._current_mode,
            'mode_config': mode_cfg,
            'distill_layers': mode_cfg['target_layers'],
            'strategy_context': strategy_context,
            'loss_functions': {'mse': loss_fn, 'kl_div': kl_loss},
            'distributed_strategy': distributed_strategy,
            'self_preset': self_preset.task_name if self_preset else None,
            'available_memory_mb': self.get_available_memory(),
        }
    
    def get_strategy(self, config: DistillationTaskConfig) -> Optional['DistillationStrategy']:
        """获取自蒸馏策略"""
        try:
            mode_cfg = self._mode_configs.get(self._current_mode, self._mode_configs['deep_to_shallow'])
            
            strategy_config = {
                'temperature': mode_cfg['temperature'],
                'distillation_type': 'self',
                'source_layers': mode_cfg['source_layers'],
                'target_layers': mode_cfg['target_layers'],
                'self_loss_weight': mode_cfg['loss_weight'],
            }
            
            with self.profile_operation("create_self_strategy"):
                strategy = create_distillation_strategy('self', config=strategy_config)
            
            self._current_strategy = strategy
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to create self-distillation strategy: {e}")
            return None
    
    def post_process(self, model: nn.Module, result: Dict[str, Any]) -> nn.Module:
        """后处理自蒸馏模型"""
        self._is_running = False
        
        # 添加自蒸馏元数据
        if hasattr(model, 'config'):
            model.config.self_distilled = True
            model.config.distillation_mode = self._current_mode
        
        # 记录运行结果
        self._monitor.end_run(
            success=result.get('success', True),
            accuracy=result.get('accuracy', 0.0),
            compression_ratio=result.get('compression_ratio', 0.0),
            loss=result.get('final_loss', 0.0),
        )
        
        # 优化内存 (lib/hardware)
        self.optimize_memory()
        
        return model
    
    def get_available_modes(self) -> List[str]:
        """获取可用的自蒸馏模式"""
        return list(self._mode_configs.keys())
    
    def set_mode(self, mode: str) -> None:
        """设置自蒸馏模式"""
        if mode in self._mode_configs:
            self._current_mode = mode
        else:
            raise ValueError(f"Unknown mode: {mode}. Available: {self.get_available_modes()}")
    
    def get_mode_config(self, mode: str) -> Dict[str, Any]:
        """获取模式配置"""
        return self._mode_configs.get(mode, {}).copy()


class ContrastiveScenarioHandler(DistillationScenarioHandler):
    """
    对比蒸馏场景处理器
    
    基于对比学习的知识蒸馏
    
    特性：
    - 对比学习损失
    - 投影层设计
    - 负样本采样
    - 表征学习优化
    """
    
    def __init__(self):
        # 先初始化属性，再调用父类构造函数
        # 对比蒸馏配置
        self._contrastive_configs = {
            'simple': {
                'contrastive_temperature': 0.07,
                'projector_dim': 128,
                'negative_samples': 256,
                'use_queue': False,
                'soft_loss_weight': 0.5,
                'contrastive_weight': 0.5,
                'description': '简单对比蒸馏',
            },
            'moco_style': {
                'contrastive_temperature': 0.07,
                'projector_dim': 256,
                'negative_samples': 1024,
                'use_queue': True,
                'queue_size': 65536,
                'soft_loss_weight': 0.4,
                'contrastive_weight': 0.6,
                'description': 'MoCo风格对比蒸馏',
            },
            'simclr_style': {
                'contrastive_temperature': 0.5,
                'projector_dim': 512,
                'negative_samples': 512,
                'use_queue': False,
                'soft_loss_weight': 0.3,
                'contrastive_weight': 0.7,
                'description': 'SimCLR风格对比蒸馏',
            },
            'clip_style': {
                'contrastive_temperature': 0.07,
                'projector_dim': 512,
                'negative_samples': 1024,
                'use_queue': False,
                'soft_loss_weight': 0.2,
                'contrastive_weight': 0.8,
                'use_cross_modal': True,
                'description': 'CLIP风格对比蒸馏',
            },
        }
        
        self._current_style = 'simple'
        
        # 调用父类构造函数
        super().__init__("contrastive")
    
    def _setup_validation_rules(self) -> None:
        """设置对比蒸馏场景的验证规则"""
        super()._setup_validation_rules()
        
        # 对比蒸馏温度由配置自动设置
        def check_contrastive_temperature(cfg) -> Tuple[bool, str]:
            # 温度由配置自动设置
            return True, ""
        
        self._validator.add_check(check_contrastive_temperature)
    
    def prepare(self, config: DistillationTaskConfig) -> Dict[str, Any]:
        """
        准备对比蒸馏场景
        
        调用 compression_config.py、策略层和 lib 层
        """
        self._monitor.start_run()
        self._is_running = True
        
        if self.should_log():
            self.logger.info(f"Preparing contrastive distillation: style={self._current_style}")
        
        with self.profile_operation("prepare_contrastive"):
            # 初始化
            if not self._is_initialized:
                self.initialize(config)
            
            # 获取对比蒸馏预设 (compression_config.py)
            contrastive_preset = DistillationPresets.get('contrastive')
            
            # 获取当前风格配置
            style_cfg = self._contrastive_configs.get(self._current_style, self._contrastive_configs['simple'])
            
            # 创建策略上下文 (base_strategy.py)
            strategy_context = self.create_strategy_context(config)
            
            # 创建损失函数 (lib/losses)
            loss_functions = {
                'contrastive': self.create_loss_function('contrastive', 
                    temperature=style_cfg['contrastive_temperature']),
                'kl_div': self.create_loss_function('kl_div', temperature=4.0),
            }
            
            # 创建分布式策略 (distributed_strategy.py)
            distributed_strategy = None
            if config.distributed_config and config.distributed_config.is_distributed():
                distributed_strategy = self.create_distributed_strategy(config)
                
                # 对比学习在分布式环境下需要同步负样本
                if self.should_log():
                    self.logger.info(f"Distributed contrastive distillation: world_size={self.get_world_size()}")
            
            # 同步进程 (lib/distributed)
            self.sync_processes()
            
            # 更新指标 (base_strategy.py)
            self.update_metrics(
                scenario='contrastive',
                style=self._current_style,
                contrastive_temperature=style_cfg['contrastive_temperature'],
                projector_dim=style_cfg['projector_dim'],
            )
        
        return {
            'prepared': True,
            'scenario': 'contrastive',
            'style': self._current_style,
            'style_config': style_cfg,
            'contrastive_temperature': style_cfg['contrastive_temperature'],
            'projector_dim': style_cfg['projector_dim'],
            'strategy_context': strategy_context,
            'loss_functions': loss_functions,
            'distributed_strategy': distributed_strategy,
            'contrastive_preset': contrastive_preset.task_name if contrastive_preset else None,
            'available_memory_mb': self.get_available_memory(),
            'world_size': self.get_world_size(),
        }
    
    def get_strategy(self, config: DistillationTaskConfig) -> Optional['DistillationStrategy']:
        """获取对比蒸馏策略"""
        try:
            style_cfg = self._contrastive_configs.get(self._current_style, self._contrastive_configs['simple'])
            
            strategy_config = {
                'temperature': 4.0,  # KL散度温度
                'contrastive_temperature': style_cfg['contrastive_temperature'],
                'contrastive_weight': style_cfg['contrastive_weight'],
                'soft_loss_weight': style_cfg['soft_loss_weight'],
                'projector_dim': style_cfg['projector_dim'],
                'negative_samples': style_cfg['negative_samples'],
            }
            
            with self.profile_operation("create_contrastive_strategy"):
                strategy = create_distillation_strategy('contrastive', config=strategy_config)
            
            self._current_strategy = strategy
            return strategy
        except Exception as e:
            self.logger.error(f"Failed to create contrastive strategy: {e}")
            return None
    
    def post_process(self, model: nn.Module, result: Dict[str, Any]) -> nn.Module:
        """后处理对比蒸馏模型"""
        self._is_running = False
        
        # 添加对比蒸馏元数据
        if hasattr(model, 'config'):
            model.config.contrastive_distilled = True
            model.config.contrastive_style = self._current_style
        
        # 记录运行结果
        self._monitor.end_run(
            success=result.get('success', True),
            accuracy=result.get('accuracy', 0.0),
            compression_ratio=result.get('compression_ratio', 0.0),
            loss=result.get('final_loss', 0.0),
        )
        
        # 优化内存 (lib/hardware)
        self.optimize_memory()
        
        # 同步进程 (lib/distributed)
        self.sync_processes()
        
        return model
    
    def get_available_styles(self) -> List[str]:
        """获取可用的对比蒸馏风格"""
        return list(self._contrastive_configs.keys())
    
    def set_style(self, style: str) -> None:
        """设置对比蒸馏风格"""
        if style in self._contrastive_configs:
            self._current_style = style
        else:
            raise ValueError(f"Unknown style: {style}. Available: {self.get_available_styles()}")
    
    def get_style_config(self, style: str) -> Dict[str, Any]:
        """获取风格配置"""
        return self._contrastive_configs.get(style, {}).copy()
    
    def estimate_memory_for_contrastive(self, batch_size: int, projector_dim: int) -> float:
        """估算对比学习的内存需求"""
        style_cfg = self._contrastive_configs.get(self._current_style, {})
        negative_samples = style_cfg.get('negative_samples', 256)
        
        # 基础内存 + 投影层 + 负样本
        base_memory = 500.0
        projector_memory = projector_dim * batch_size * 4 / (1024 * 1024)  # float32
        negative_memory = projector_dim * negative_samples * 4 / (1024 * 1024)
        
        # 如果使用队列，额外内存
        if style_cfg.get('use_queue', False):
            queue_size = style_cfg.get('queue_size', 65536)
            negative_memory += projector_dim * queue_size * 4 / (1024 * 1024)
        
        return base_memory + projector_memory + negative_memory


# ======================== 场景管理器 ========================

class DistillationScenarioManager:
    """
    蒸馏场景管理器
    
    生产级特性：
    - 统一管理各种蒸馏场景
    - 场景选择和配置优化
    - 与策略层深度集成 (base_strategy.py, distributed_strategy.py, distillation_strategy.py)
    - 与配置层集成 (compression_config.py)
    - 与 lib 层集成 (losses, hardware, distributed)
    - 分布式支持
    - 完整的监控和诊断
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 注册场景处理器
        self._handlers: Dict[str, DistillationScenarioHandler] = {
            'standard': StandardScenarioHandler(),
            'industry': IndustryScenarioHandler(),
            'edge_deploy': EdgeDeployScenarioHandler(),
            'multimodal': MultimodalScenarioHandler(),
            'real_time': RealTimeScenarioHandler(),
            'progressive': ProgressiveScenarioHandler(),
            'self': SelfDistillationScenarioHandler(),
            'contrastive': ContrastiveScenarioHandler(),
        }
        
        # 任务缓存
        self._active_tasks: Dict[str, Dict[str, Any]] = {}
        self._completed_tasks: Dict[str, Dict[str, Any]] = {}
        self._task_lock = threading.Lock()
        
        # 全局监控
        self._global_stats = ScenarioExecutionStats(scenario_name="global")
        self._global_monitor = ScenarioMonitor("global")
        
        # 使用策略层的组件 (base_strategy.py)
        self._strategy_validator: Optional['StrategyValidator'] = None
        try:
            self._strategy_validator = StrategyValidator()
        except Exception:
            pass
        
        self._strategy_profiler: Optional['StrategyProfiler'] = None
        try:
            self._strategy_profiler = StrategyProfiler()
        except Exception:
            pass
        
        self._strategy_metrics: Optional['StrategyMetrics'] = None
        try:
            self._strategy_metrics = StrategyMetrics()
        except Exception:
            pass
        
        # 使用 lib/hardware
        self._device_manager: Optional['DeviceManager'] = None
        try:
            self._device_manager = get_device_manager()
        except Exception:
            pass
        
        self._memory_manager: Optional['MemoryManager'] = None
        try:
            self._memory_manager = MemoryManager()
        except Exception:
            pass
        
        # 使用 lib/distributed
        self._distributed_manager: Optional['DistributedManager'] = None
        try:
            self._distributed_manager = get_distributed_manager()
        except Exception:
            pass
        
        # 使用 lib/losses
        self._loss_factory: Optional['LossFactory'] = None
        try:
            self._loss_factory = LossFactory()
        except Exception:
            pass
        
        # 分布式配置
        self._is_distributed = False
        self._world_size = 1
        self._rank = 0
    
    def get_available_scenarios(self) -> List[str]:
        """获取可用的场景列表"""
        return list(self._handlers.keys())
    
    def get_handler(self, scenario: str) -> Optional[DistillationScenarioHandler]:
        """获取场景处理器"""
        return self._handlers.get(scenario)
    
    def register_handler(self, scenario: str, handler: DistillationScenarioHandler) -> None:
        """注册自定义场景处理器"""
        self._handlers[scenario] = handler
        self.logger.info(f"Registered custom scenario handler: {scenario}")
    
    def unregister_handler(self, scenario: str) -> bool:
        """取消注册场景处理器"""
        if scenario in self._handlers:
            del self._handlers[scenario]
            self.logger.info(f"Unregistered scenario handler: {scenario}")
            return True
        return False
    
    def initialize_distributed(self) -> None:
        """
        初始化分布式环境
        
        使用 lib/distributed
        """
        self._is_distributed = get_world_size() > 1
        self._world_size = get_world_size()
        self._rank = get_rank()

        if self.should_log():
            self.logger.info(f"Distributed initialized: world_size={self._world_size}, rank={self._rank}")
    
    def should_log(self) -> bool:
        """检查是否应该记录日志"""
        return is_main_process()
    
    def sync_processes(self) -> None:
        """
        同步所有进程
        
        使用 lib/distributed
        """
        try:
            barrier()
        except Exception:
            pass
    
    def optimize_memory(self) -> None:
        """
        优化内存
        
        使用 lib/hardware
        """
        try:
            clear_memory()
        except Exception:
            pass
        
        if self._memory_manager is not None:
            try:
                if hasattr(self._memory_manager, 'clear_memory'):
                    self._memory_manager.clear_memory()
            except Exception:
                pass
    
    def get_available_memory(self) -> float:
        """
        获取可用内存
        
        使用 lib/hardware
        """
        try:
            return get_available_memory()
        except Exception:
            pass
    
    def prepare_scenario(self, config: DistillationTaskConfig) -> Dict[str, Any]:
        """
        准备场景
        
        使用 compression_config.py 的配置类和策略层
        """
        scenario = 'standard'
        if config.scenario_config:
            scenario = config.scenario_config.scenario
        
        handler = self._handlers.get(scenario)
        if not handler:
            self.logger.warning(f"Unknown scenario: {scenario}, falling back to standard")
            handler = self._handlers['standard']
        
        # 验证配置 (compression_config.py)
        is_valid, errors = handler.validate_config(config)
        if not is_valid:
            self.logger.warning(f"Config validation warnings: {errors}")
        
        # 使用 profiler 分析准备过程 (base_strategy.py)
        if self._strategy_profiler is not None:
            with self._strategy_profiler.profile(f"prepare_{scenario}"):
                result = handler.prepare(config)
        else:
            result = handler.prepare(config)
        
        # 更新指标 (base_strategy.py)
        if self._strategy_metrics is not None:
            try:
                self._strategy_metrics.update({
                    'scenario': scenario,
                    'prepared': True,
                })
            except Exception:
                pass
        
        return result
    
    def get_strategy_for_scenario(self, config: DistillationTaskConfig) -> Optional['DistillationStrategy']:
        """
        获取场景对应的蒸馏策略
        
        使用 distillation_strategy.py 的策略类
        """
        scenario = 'standard'
        if config.scenario_config:
            scenario = config.scenario_config.scenario
        
        handler = self._handlers.get(scenario, self._handlers['standard'])
        
        # 使用 profiler 分析策略创建 (base_strategy.py)
        if self._strategy_profiler is not None:
            with self._strategy_profiler.profile(f"get_strategy_{scenario}"):
                return handler.get_strategy(config)
        
        return handler.get_strategy(config)
    
    def get_distributed_strategy(self, config: DistillationTaskConfig) -> Optional['DistributedStrategy']:
        """
        获取分布式策略
        
        使用 distributed_strategy.py
        """
        scenario = 'standard'
        if config.scenario_config:
            scenario = config.scenario_config.scenario
        
        handler = self._handlers.get(scenario, self._handlers['standard'])
        return handler.create_distributed_strategy(config)
    
    def create_strategy_context(self, config: DistillationTaskConfig) -> Optional['StrategyContext']:
        """
        创建策略上下文
        
        使用 base_strategy.py 的 StrategyContext
        """
        scenario = 'standard'
        if config.scenario_config:
            scenario = config.scenario_config.scenario
        
        handler = self._handlers.get(scenario, self._handlers['standard'])
        return handler.create_strategy_context(config)
    
    def post_process_model(
        self, 
        model: nn.Module, 
        config: DistillationTaskConfig,
        result: Dict[str, Any]
    ) -> nn.Module:
        """后处理模型"""
        scenario = 'standard'
        if config.scenario_config:
            scenario = config.scenario_config.scenario
        
        handler = self._handlers.get(scenario, self._handlers['standard'])
        
        # 使用 profiler (base_strategy.py)
        if self._strategy_profiler is not None:
            with self._strategy_profiler.profile(f"post_process_{scenario}"):
                return handler.post_process(model, result)
        
        return handler.post_process(model, result)
    
    def recommend_scenario(
        self, 
        requirements: Dict[str, Any]
    ) -> Tuple[str, DistillationTaskConfig]:
        """
        根据需求推荐场景和配置
        
        使用 compression_config.py 的 recommend_config 和 DistillationPresets
        """
        target_device = requirements.get('target_device', 'gpu')
        target_latency = requirements.get('target_latency_ms', 1000)
        target_accuracy = requirements.get('target_accuracy', 0.95)
        industry = requirements.get('industry')
        modalities = requirements.get('modalities', ['text'])
        model_size_gb = requirements.get('model_size_gb', 1.0)
        num_gpus = requirements.get('num_gpus', 1)
        use_progressive = requirements.get('use_progressive', False)
        use_contrastive = requirements.get('use_contrastive', False)
        use_self_distillation = requirements.get('use_self_distillation', False)
        
        # 使用 profiler 分析推荐过程 (base_strategy.py)
        if self._strategy_profiler is not None:
            with self._strategy_profiler.profile("recommend_scenario"):
                scenario, config = self._do_recommend_scenario(
                    target_device, target_latency, target_accuracy,
                    industry, modalities, model_size_gb, num_gpus,
                    use_progressive, use_contrastive, use_self_distillation
                )
        else:
            scenario, config = self._do_recommend_scenario(
                target_device, target_latency, target_accuracy,
                industry, modalities, model_size_gb, num_gpus,
                use_progressive, use_contrastive, use_self_distillation
            )
        
        if self.should_log():
            self.logger.info(f"Recommended scenario: {scenario}")
        
        return scenario, config
    
    def _do_recommend_scenario(
        self,
        target_device: str,
        target_latency: float,
        target_accuracy: float,
        industry: Optional[str],
        modalities: List[str],
        model_size_gb: float,
        num_gpus: int,
        use_progressive: bool,
        use_contrastive: bool,
        use_self_distillation: bool,
    ) -> Tuple[str, DistillationTaskConfig]:
        """执行场景推荐逻辑"""
        # 优先使用 compression_config.py 的推荐函数和预设
        if use_self_distillation:
            scenario = 'self'
            config = DistillationPresets.get('self_distillation')
        elif use_contrastive:
            scenario = 'contrastive'
            config = DistillationPresets.get('contrastive')
        elif use_progressive:
            scenario = 'progressive'
            config = DistillationPresets.get('progressive')
        elif target_device in ['edge', 'mobile', 'iot']:
            scenario = 'edge_deploy'
            config = DistillationPresets.get('edge_deployment')
        elif target_latency < 50:
            scenario = 'real_time'
            config = DistillationPresets.get('real_time')
        elif target_accuracy > 0.97:
            scenario = 'standard'
            config = DistillationPresets.get('high_accuracy')
        elif industry:
            scenario = 'industry'
            config = DistillationPresets.get('industry', industry_type=industry)
        elif len(modalities) > 1:
            scenario = 'multimodal'
            config = DistillationPresets.get('multimodal')
        else:
            scenario = 'standard'
            # 使用 recommend_config 获取推荐配置 (compression_config.py)
            config = recommend_config(
                model_size_gb=model_size_gb,
                target_device=target_device,
                target_latency_ms=target_latency,
                target_accuracy=target_accuracy,
                num_gpus=num_gpus,
            )
        
        return scenario, config
    
    def create_task(self, config: DistillationTaskConfig) -> str:
        """创建蒸馏任务"""
        import uuid
        task_id = config.task_id or str(uuid.uuid4())[:8]
        
        with self._task_lock:
            self._active_tasks[task_id] = {
                'task_id': task_id,
                'config': config,
                'status': 'created',
                'created_at': datetime.utcnow().isoformat(),
                'progress': 0.0,
                'scenario': config.scenario_config.scenario if config.scenario_config else 'standard',
                'rank': self._rank,
                'world_size': self._world_size,
            }
        
        if self.should_log():
            self.logger.info(f"Created distillation task: {task_id}")
        
        return task_id
    
    def start_task(self, task_id: str) -> bool:
        """开始任务"""
        with self._task_lock:
            if task_id in self._active_tasks:
                self._active_tasks[task_id]['status'] = 'running'
                self._active_tasks[task_id]['started_at'] = datetime.utcnow().isoformat()
                
                # 开始全局监控
                self._global_monitor.start_run()
                
                return True
        return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        with self._task_lock:
            if task_id in self._active_tasks:
                return self._active_tasks[task_id].copy()
            if task_id in self._completed_tasks:
                return self._completed_tasks[task_id].copy()
        return None
    
    def update_task_progress(
        self, 
        task_id: str, 
        progress: float, 
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """更新任务进度"""
        with self._task_lock:
            if task_id in self._active_tasks:
                self._active_tasks[task_id]['progress'] = progress
                self._active_tasks[task_id]['status'] = 'running'
                self._active_tasks[task_id]['last_update'] = datetime.utcnow().isoformat()
                if metrics:
                    self._active_tasks[task_id]['metrics'] = metrics
                    
                    # 更新全局监控
                    self._global_monitor.record_step(
                        kd_loss=metrics.get('kd_loss', 0.0),
                        ce_loss=metrics.get('ce_loss', 0.0),
                        accuracy=metrics.get('accuracy', 0.0),
                    )
    
    def complete_task(self, task_id: str, result: Dict[str, Any]) -> None:
        """完成任务"""
        with self._task_lock:
            if task_id in self._active_tasks:
                task = self._active_tasks.pop(task_id)
                task['status'] = 'completed'
                task['progress'] = 100.0
                task['result'] = result
                task['completed_at'] = datetime.utcnow().isoformat()
                
                # 移动到已完成任务
                self._completed_tasks[task_id] = task
                
                # 限制已完成任务数量
                if len(self._completed_tasks) > 1000:
                    oldest = min(self._completed_tasks.keys())
                    del self._completed_tasks[oldest]
                
                # 更新全局统计
                self._global_stats.record_run(
                    success=result.get('success', True),
                    duration=result.get('duration', 0.0),
                    accuracy=result.get('accuracy', 0.0),
                    compression_ratio=result.get('compression_ratio', 0.0),
                    loss=result.get('final_loss', 0.0),
                )
                
                # 结束全局监控
                self._global_monitor.end_run(
                    success=result.get('success', True),
                    accuracy=result.get('accuracy', 0.0),
                    compression_ratio=result.get('compression_ratio', 0.0),
                    loss=result.get('final_loss', 0.0),
                )
    
    def fail_task(self, task_id: str, error: str) -> None:
        """标记任务失败"""
        with self._task_lock:
            if task_id in self._active_tasks:
                task = self._active_tasks.pop(task_id)
                task['status'] = 'failed'
                task['error'] = error
                task['failed_at'] = datetime.utcnow().isoformat()
                
                # 移动到已完成任务
                self._completed_tasks[task_id] = task
                
                # 更新全局统计
                self._global_stats.record_run(success=False, duration=0.0)
                
                # 结束全局监控
                self._global_monitor.end_run(success=False)
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self._task_lock:
            if task_id in self._active_tasks:
                task = self._active_tasks.pop(task_id)
                task['status'] = 'cancelled'
                task['cancelled_at'] = datetime.utcnow().isoformat()
                
                self._completed_tasks[task_id] = task
                return True
        return False
    
    def get_active_tasks(self) -> Dict[str, Dict[str, Any]]:
        """获取所有活动任务"""
        with self._task_lock:
            return self._active_tasks.copy()
    
    def get_completed_tasks(self, limit: int = 100) -> Dict[str, Dict[str, Any]]:
        """获取已完成任务"""
        with self._task_lock:
            items = list(self._completed_tasks.items())[-limit:]
            return dict(items)
    
    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """获取所有任务"""
        with self._task_lock:
            return {**self._active_tasks, **self._completed_tasks}
    
    def get_global_stats(self) -> ScenarioExecutionStats:
        """获取全局统计"""
        return self._global_stats
    
    def get_global_monitor(self) -> ScenarioMonitor:
        """获取全局监控器"""
        return self._global_monitor
    
    def get_scenario_stats(self, scenario: str) -> Optional[ScenarioExecutionStats]:
        """获取场景统计"""
        handler = self._handlers.get(scenario)
        if handler:
            return handler.get_monitor().get_stats()
        return None
    
    def get_all_scenario_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有场景的统计"""
        stats = {}
        for name, handler in self._handlers.items():
            stats[name] = handler.get_monitor().get_stats().to_dict()
        return stats
    
    def get_profiler_summary(self) -> Optional[Dict[str, Any]]:
        """
        获取性能分析摘要
        
        使用 base_strategy.py 的 StrategyProfiler
        """
        if self._strategy_profiler is not None:
            try:
                return self._strategy_profiler.get_stats()
            except Exception:
                pass
        return None
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        获取策略指标
        
        使用 base_strategy.py 的 StrategyMetrics
        """
        if self._strategy_metrics is not None and hasattr(self._strategy_metrics, 'to_dict'):
            try:
                return self._strategy_metrics.to_dict()
            except Exception:
                pass
        return None
    
    def get_distributed_info(self) -> Dict[str, Any]:
        """
        获取分布式信息
        
        使用 lib/distributed
        """
        return {
            'is_distributed': self._is_distributed,
            'world_size': self._world_size,
            'rank': self._rank,
            'is_main_process': self.should_log(),
        }
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """
        获取硬件信息
        
        使用 lib/hardware
        """
        info = {
            'available_memory_mb': self.get_available_memory(),
            'device_manager_available': self._device_manager is not None,
            'memory_manager_available': self._memory_manager is not None,
        }
        
        if self._device_manager is not None and hasattr(self._device_manager, 'get_device_info'):
            try:
                info['device_info'] = self._device_manager.get_device_info()
            except Exception:
                pass
        
        return info
    
    def cleanup(self) -> None:
        """清理资源"""
        # 清理所有场景处理器
        for handler in self._handlers.values():
            try:
                handler.cleanup()
            except Exception:
                pass
        
        # 优化内存
        self.optimize_memory()
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断场景管理器状态"""
        diagnosis = {
            'available_scenarios': self.get_available_scenarios(),
            'global_stats': self._global_stats.to_dict(),
            'global_monitor_summary': self._global_monitor.get_summary(),
            'active_tasks_count': len(self._active_tasks),
            'completed_tasks_count': len(self._completed_tasks),
            'strategy_validator_available': self._strategy_validator is not None,
            'strategy_profiler_available': self._strategy_profiler is not None,
            'strategy_metrics_available': self._strategy_metrics is not None,
            'device_manager_available': self._device_manager is not None,
            'memory_manager_available': self._memory_manager is not None,
            'distributed_manager_available': self._distributed_manager is not None,
            'loss_factory_available': self._loss_factory is not None,
            'distributed_info': self.get_distributed_info(),
            'hardware_info': self.get_hardware_info(),
            'scenario_diagnostics': {},
        }
        
        # 添加每个场景的诊断
        for name, handler in self._handlers.items():
            diagnosis['scenario_diagnostics'][name] = handler.diagnose()
        
        # 添加性能分析摘要
        profiler_summary = self.get_profiler_summary()
        if profiler_summary:
            diagnosis['profiler_summary'] = profiler_summary
        
        # 添加策略指标
        metrics = self.get_metrics()
        if metrics:
            diagnosis['strategy_metrics'] = metrics
        
        return diagnosis
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        if not self.should_log():
            return
        
        diagnosis = self.diagnose()
        print("\n" + "=" * 60)
        print("Distillation Scenario Manager Diagnosis")
        print("=" * 60)
        print(f"\nAvailable Scenarios: {diagnosis['available_scenarios']}")
        print(f"Active Tasks: {diagnosis['active_tasks_count']}")
        print(f"Completed Tasks: {diagnosis['completed_tasks_count']}")
        
        print(f"\nGlobal Stats:")
        for key, value in diagnosis['global_stats'].items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
        
        print(f"\nDistributed Info:")
        for key, value in diagnosis['distributed_info'].items():
            print(f"  {key}: {value}")
        
        print(f"\nHardware Info:")
        for key, value in diagnosis['hardware_info'].items():
            print(f"  {key}: {value}")
        
        print(f"\nLayer Availability:")
        for layer, available in diagnosis['layers_available'].items():
            status = "✓" if available else "✗"
            print(f"  {status} {layer}")
        
        print(f"\nComponent Availability:")
        components = [
            ('Strategy Validator', diagnosis['strategy_validator_available']),
            ('Strategy Profiler', diagnosis['strategy_profiler_available']),
            ('Strategy Metrics', diagnosis['strategy_metrics_available']),
            ('Device Manager', diagnosis['device_manager_available']),
            ('Memory Manager', diagnosis['memory_manager_available']),
            ('Distributed Manager', diagnosis['distributed_manager_available']),
            ('Loss Factory', diagnosis['loss_factory_available']),
        ]
        for name, available in components:
            status = "✓" if available else "✗"
            print(f"  {status} {name}")
        
        print("=" * 60)


# ======================== 全局实例 ========================

_scenario_manager: Optional[DistillationScenarioManager] = None
_manager_lock = threading.Lock()


def get_scenario_manager() -> DistillationScenarioManager:
    """获取场景管理器单例"""
    global _scenario_manager
    
    if _scenario_manager is None:
        with _manager_lock:
            if _scenario_manager is None:
                _scenario_manager = DistillationScenarioManager()
    
    return _scenario_manager


def reset_scenario_manager() -> DistillationScenarioManager:
    """重置场景管理器"""
    global _scenario_manager
    
    with _manager_lock:
        if _scenario_manager is not None:
            _scenario_manager.cleanup()
        _scenario_manager = DistillationScenarioManager()
    
    return _scenario_manager


def create_scenario_handler(scenario: str) -> Optional[DistillationScenarioHandler]:
    """创建场景处理器"""
    manager = get_scenario_manager()
    return manager.get_handler(scenario)


def list_available_scenarios() -> List[str]:
    """列出可用场景"""
    manager = get_scenario_manager()
    return manager.get_available_scenarios()


def get_scenario_description(scenario: str) -> str:
    """
    获取场景描述
    
    返回场景的详细描述
    """
    descriptions = {
        'standard': '标准蒸馏：基础的教师-学生蒸馏，适用于大多数场景',
        'industry': '行业蒸馏：针对特定行业（制造、金融、医疗等）的领域适配蒸馏',
        'edge_deploy': '边缘部署蒸馏：针对边缘设备（手机、IoT设备等）优化的蒸馏',
        'multimodal': '多模态蒸馏：支持跨模态知识迁移',
        'real_time': '实时推理蒸馏：优化低延迟场景',
        'progressive': '渐进式蒸馏：逐步增加蒸馏的层数和复杂度',
        'self': '自蒸馏：模型自身不同层之间的知识蒸馏，无需教师模型',
        'contrastive': '对比蒸馏：基于对比学习的知识蒸馏',
    }
    return descriptions.get(scenario, f'未知场景: {scenario}')


def recommend_distillation_scenario(
    target_device: str = "gpu",
    target_latency_ms: float = 100.0,
    target_accuracy: float = 0.95,
    industry: Optional[str] = None,
    modalities: Optional[List[str]] = None,
    model_size_gb: float = 1.0,
    num_gpus: int = 1,
    use_progressive: bool = False,
    use_contrastive: bool = False,
    use_self_distillation: bool = False,
) -> Tuple[str, DistillationTaskConfig]:
    """
    推荐蒸馏场景
    
    便捷函数，使用场景管理器推荐
    
    Args:
        target_device: 目标设备类型 ('gpu', 'cpu', 'edge', 'mobile', 'iot')
        target_latency_ms: 目标延迟（毫秒）
        target_accuracy: 目标精度
        industry: 行业类型（可选）
        modalities: 模态列表（可选）
        model_size_gb: 模型大小（GB）
        num_gpus: GPU 数量
        use_progressive: 是否使用渐进式蒸馏
        use_contrastive: 是否使用对比蒸馏
        use_self_distillation: 是否使用自蒸馏
    
    Returns:
        Tuple[str, DistillationTaskConfig]: 推荐的场景名称和配置
    """
    manager = get_scenario_manager()
    return manager.recommend_scenario({
        'target_device': target_device,
        'target_latency_ms': target_latency_ms,
        'target_accuracy': target_accuracy,
        'industry': industry,
        'modalities': modalities or ['text'],
        'model_size_gb': model_size_gb,
        'num_gpus': num_gpus,
        'use_progressive': use_progressive,
        'use_contrastive': use_contrastive,
        'use_self_distillation': use_self_distillation,
    })


def prepare_scenario(config: DistillationTaskConfig) -> Dict[str, Any]:
    """
    准备蒸馏场景
    
    便捷函数，调用场景管理器准备场景
    """
    manager = get_scenario_manager()
    return manager.prepare_scenario(config)


def get_strategy_for_scenario(config: DistillationTaskConfig) -> Optional['DistillationStrategy']:
    """
    获取场景对应的蒸馏策略
    
    便捷函数，调用场景管理器获取策略
    """
    manager = get_scenario_manager()
    return manager.get_strategy_for_scenario(config)


def post_process_model(
    model: nn.Module,
    config: DistillationTaskConfig,
    result: Dict[str, Any]
) -> nn.Module:
    """
    后处理蒸馏模型
    
    便捷函数，调用场景管理器后处理模型
    """
    manager = get_scenario_manager()
    return manager.post_process_model(model, config, result)


def create_distillation_task(config: DistillationTaskConfig) -> str:
    """
    创建蒸馏任务
    
    便捷函数，调用场景管理器创建任务
    """
    manager = get_scenario_manager()
    return manager.create_task(config)


def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    获取任务状态
    
    便捷函数，调用场景管理器获取任务状态
    """
    manager = get_scenario_manager()
    return manager.get_task_status(task_id)


def update_task_progress(task_id: str, progress: float, metrics: Optional[Dict[str, float]] = None) -> None:
    """
    更新任务进度
    
    便捷函数，调用场景管理器更新任务进度
    """
    manager = get_scenario_manager()
    manager.update_task_progress(task_id, progress, metrics)


def complete_task(task_id: str, result: Dict[str, Any]) -> None:
    """
    完成任务
    
    便捷函数，调用场景管理器完成任务
    """
    manager = get_scenario_manager()
    manager.complete_task(task_id, result)


def fail_task(task_id: str, error: str) -> None:
    """
    标记任务失败
    
    便捷函数，调用场景管理器标记任务失败
    """
    manager = get_scenario_manager()
    manager.fail_task(task_id, error)


def get_global_stats() -> ScenarioExecutionStats:
    """
    获取全局统计
    
    便捷函数，调用场景管理器获取全局统计
    """
    manager = get_scenario_manager()
    return manager.get_global_stats()


def get_scenario_stats(scenario: str) -> Optional[ScenarioExecutionStats]:
    """
    获取场景统计
    
    便捷函数，调用场景管理器获取场景统计
    """
    manager = get_scenario_manager()
    return manager.get_scenario_stats(scenario)


def get_all_scenario_stats() -> Dict[str, Dict[str, Any]]:
    """
    获取所有场景的统计
    
    便捷函数，调用场景管理器获取所有场景统计
    """
    manager = get_scenario_manager()
    return manager.get_all_scenario_stats()


def diagnose_scenarios() -> Dict[str, Any]:
    """
    诊断场景管理器状态
    
    便捷函数，调用场景管理器诊断
    """
    manager = get_scenario_manager()
    return manager.diagnose()


def print_scenario_summary() -> None:
    """打印场景摘要"""
    manager = get_scenario_manager()
    
    if not manager.should_log():
        return
    
    print("\n" + "=" * 60)
    print("Available Distillation Scenarios")
    print("=" * 60)
    
    for scenario in manager.get_available_scenarios():
        handler = manager.get_handler(scenario)
        if handler:
            stats = handler.get_monitor().get_stats()
            description = get_scenario_description(scenario)
            print(f"\n{scenario}:")
            print(f"  Description: {description}")
            print(f"  Total runs: {stats.total_runs}")
            print(f"  Success rate: {stats.success_rate:.1%}")
            print(f"  Best accuracy: {stats.best_accuracy:.2%}")
            if stats.avg_loss > 0:
                print(f"  Avg loss: {stats.avg_loss:.4f}")
    
    print("\n" + "=" * 60)


def print_diagnosis() -> None:
    """
    打印场景管理器诊断信息
    
    便捷函数，调用场景管理器打印诊断
    """
    manager = get_scenario_manager()
    manager.print_diagnosis()


def compare_scenarios(
    scenario1: str,
    scenario2: str,
) -> Dict[str, Any]:
    """
    比较两个场景
    
    返回两个场景的统计对比
    """
    manager = get_scenario_manager()
    
    handler1 = manager.get_handler(scenario1)
    handler2 = manager.get_handler(scenario2)
    
    if not handler1 or not handler2:
        return {'error': 'One or both scenarios not found'}
    
    stats1 = handler1.get_monitor().get_stats()
    stats2 = handler2.get_monitor().get_stats()
    
    return {
        'scenario1': {
            'name': scenario1,
            'stats': stats1.to_dict(),
        },
        'scenario2': {
            'name': scenario2,
            'stats': stats2.to_dict(),
        },
        'comparison': {
            'success_rate_diff': stats1.success_rate - stats2.success_rate,
            'best_accuracy_diff': stats1.best_accuracy - stats2.best_accuracy,
            'avg_time_diff': stats1.avg_time_seconds - stats2.avg_time_seconds,
        },
    }


def estimate_scenario_requirements(
    scenario: str,
    config: DistillationTaskConfig,
) -> Dict[str, Any]:
    """
    估算场景需求
    
    返回内存、时间等估算
    """
    manager = get_scenario_manager()
    handler = manager.get_handler(scenario)
    
    if not handler:
        return {'error': f'Scenario not found: {scenario}'}
    
    requirements = {
        'scenario': scenario,
        'available_memory_mb': handler.get_available_memory(),
    }
    
    # 尝试获取场景特定的估算
    if hasattr(handler, 'estimate_memory_requirement'):
        try:
            requirements['estimated_memory_mb'] = handler.estimate_memory_requirement(config)
        except Exception:
            pass
    
    if config.distillation_config:
        requirements['distillation_memory_mb'] = config.distillation_config.estimate_memory_mb()
    
    return requirements

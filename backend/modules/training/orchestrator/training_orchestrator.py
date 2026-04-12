# -*- coding: utf-8 -*-
"""训练编排器

生产级统一训练流程编排，支持：
- 多阶段训练（行业预训练 → 行业对齐 → 场景精调）
- 策略层集成
- 硬件层集成
- 分布式训练
- 进度追踪
- 任务调度
- 状态管理

架构调用层次：
├── training_orchestrator.py (本模块)
│   ├── 调用 backend/modules/training/strategies/base_strategy (策略层)
│   ├── 调用 backend/modules/training/strategies/distributed_strategy (分布式策略)
│   ├── 调用 backend/lib/hardware (硬件层)
│   ├── 调用 backend/lib/distributed (分布式层)
│   ├── 调用 backend/lib/losses (损失层)
│   └── 调用 backend/modules/training/progress (进度管理)
└── 被服务层和场景层调用
"""

import logging
import os
import sys
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

logger = logging.getLogger(__name__)


# ==================== 策略层导入 ====================

STRATEGY_LAYER_AVAILABLE = False
StrategyContext = None
StrategyResult = None
StrategyMetrics = None
StrategyMonitor = None
TrainingStrategy = None

try:
    from backend.modules.training.strategies.base_strategy import (
        StrategyContext, StrategyResult, StrategyMetrics,
        StrategyMonitor, TrainingStrategy,
    )
    STRATEGY_LAYER_AVAILABLE = True
    logger.info("Strategy layer loaded for training_orchestrator")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Strategy layer not available for training_orchestrator: {e}")


# ==================== 分布式策略层导入 ====================

DISTRIBUTED_STRATEGY_AVAILABLE = False
DistributedMode = None
DistributedStrategyConfig = None
DistributedStrategy = None
recommend_distributed_mode = None

try:
    from backend.modules.training.strategies.distributed_strategy import (
        DistributedMode, DistributedStrategyConfig, DistributedStrategy,
        recommend_distributed_mode,
    )
    DISTRIBUTED_STRATEGY_AVAILABLE = True
    logger.info("Distributed strategy layer loaded for training_orchestrator")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Distributed strategy not available for training_orchestrator: {e}")


# ==================== 硬件层导入 ====================

HARDWARE_LAYER_AVAILABLE = False
DeviceManager = None
get_device_manager = None
get_available_memory = None
clear_memory = None
recommend_precision = None

try:
    from backend.lib.hardware import (
        DeviceManager, get_device_manager,
        get_available_memory, clear_memory,
        recommend_precision,
    )
    HARDWARE_LAYER_AVAILABLE = True
    logger.info("Hardware layer loaded for training_orchestrator")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Hardware layer not available for training_orchestrator: {e}")


# ==================== 分布式层导入 ====================

DISTRIBUTED_LAYER_AVAILABLE = False
DistributedManager = None
get_distributed_manager = None

try:
    from backend.lib.distributed import (
        DistributedManager, get_distributed_manager,
    )
    DISTRIBUTED_LAYER_AVAILABLE = True
    logger.info("Distributed layer loaded for training_orchestrator")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Distributed layer not available for training_orchestrator: {e}")


# ==================== 损失层导入 ====================

LOSSES_LAYER_AVAILABLE = False
LossFactory = None
create_loss = None

try:
    from backend.lib.losses import (
        LossFactory, create_loss,
    )
    LOSSES_LAYER_AVAILABLE = True
    logger.info("Losses layer loaded for training_orchestrator")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Losses layer not available for training_orchestrator: {e}")


# ==================== 进度管理导入 ====================

PROGRESS_MANAGER_AVAILABLE = False
TrainingProgressManager = None
TrainingProgress = None
get_progress_manager = None

try:
    from backend.modules.training.progress import (
        TrainingProgressManager, TrainingProgress,
        get_progress_manager,
    )
    PROGRESS_MANAGER_AVAILABLE = True
    logger.info("Progress manager loaded for training_orchestrator")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Progress manager not available for training_orchestrator: {e}")


# ==================== 枚举定义 ====================

class TrainingPhase(Enum):
    """训练阶段"""
    PRETRAIN_INDUSTRY = "pretrain_industry"     # 行业表征预训练
    ALIGN_INDUSTRY = "align_industry"           # 行业能力对齐
    FINETUNE_SCENE = "finetune_scene"           # 场景精调
    
    # 通用阶段
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    PREFERENCE = "preference"
    EVALUATION = "evaluation"
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        names = {
            TrainingPhase.PRETRAIN_INDUSTRY: "行业表征预训练",
            TrainingPhase.ALIGN_INDUSTRY: "行业能力对齐",
            TrainingPhase.FINETUNE_SCENE: "场景精调",
            TrainingPhase.PRETRAIN: "预训练",
            TrainingPhase.FINETUNE: "微调",
            TrainingPhase.PREFERENCE: "偏好优化",
            TrainingPhase.EVALUATION: "评估",
        }
        return names.get(self, self.value)


class TrainingStatus(Enum):
    """训练状态"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ==================== 数据类定义 ====================

@dataclass
class TrainingPlan:
    """训练计划
    
    定义完整的训练流程，包括多个阶段。
    """
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = "industry_model"
    stages: List[TrainingPhase] = field(default_factory=list)
    
    # 阶段配置
    stage_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 全局配置
    output_dir: str = "./outputs"
    pass_model_between_stages: bool = True
    save_intermediate_models: bool = True
    
    # 分布式配置
    use_distributed: bool = False
    distributed_mode: str = "ddp"
    world_size: int = 1
    
    # 回调
    on_stage_start: Optional[Callable] = None
    on_stage_end: Optional[Callable] = None
    on_checkpoint: Optional[Callable] = None
    
    def add_stage(
        self, 
        phase: TrainingPhase, 
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """添加训练阶段"""
        self.stages.append(phase)
        if config:
            self.stage_configs[phase.value] = config
    
    def get_stage_config(self, phase: TrainingPhase) -> Dict[str, Any]:
        """获取阶段配置"""
        return self.stage_configs.get(phase.value, {})
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'plan_id': self.plan_id,
            'model_name': self.model_name,
            'stages': [s.value for s in self.stages],
            'stage_configs': self.stage_configs,
            'output_dir': self.output_dir,
            'pass_model_between_stages': self.pass_model_between_stages,
            'save_intermediate_models': self.save_intermediate_models,
            'use_distributed': self.use_distributed,
            'distributed_mode': self.distributed_mode,
            'world_size': self.world_size,
        }


@dataclass
class StageResult:
    """阶段执行结果"""
    phase: TrainingPhase
    status: TrainingStatus
    model_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    
    # 策略层扩展
    strategy_result: Optional['StrategyResult'] = None
    
    @property
    def duration(self) -> float:
        """执行时长（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'phase': self.phase.value,
            'status': self.status.value,
            'model_path': self.model_path,
            'metrics': self.metrics,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error': self.error,
            'duration': self.duration,
        }


@dataclass
class TrainingJob:
    """训练任务"""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    plan: Optional[TrainingPlan] = None
    status: TrainingStatus = TrainingStatus.PENDING
    current_phase: Optional[TrainingPhase] = None
    phase_results: List[StageResult] = field(default_factory=list)
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 控制
    should_stop: bool = False
    should_pause: bool = False

    # 策略层扩展
    strategy_context: Optional['StrategyContext'] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'job_id': self.job_id,
            'plan': self.plan.to_dict() if self.plan else None,
            'status': self.status.value,
            'current_phase': self.current_phase.value if self.current_phase else None,
            'phase_results': [r.to_dict() for r in self.phase_results],
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
        }


# ==================== 策略化训练器 ====================

class StrategyTrainer:
    """策略化训练器
    
    实现技术方案中的Trainer内核，支持多策略组合。
    集成策略层、硬件层、损失层能力。
    """
    
    def __init__(
        self,
        model: nn.Module,
        strategies: List,  # List[TrainingStrategy]
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: torch.device = None
    ):
        self.model = model
        self.strategies = strategies
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # 使用硬件层获取设备
        if device is None:
            if HARDWARE_LAYER_AVAILABLE and get_device_manager is not None:
                try:
                    device_manager = get_device_manager()
                    if device_manager is not None and hasattr(device_manager, 'device'):
                        device = device_manager.device
                except Exception:
                    pass
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        
        # 策略层组件
        self._strategy_context: Optional['StrategyContext'] = None
        self._strategy_metrics: Optional['StrategyMetrics'] = None
        
        # 按优先级排序策略
        if self.strategies:
            self.strategies = sorted(
                self.strategies, 
                key=lambda s: getattr(s, 'priority', 0)
            )
        
        # 初始化策略上下文
        self._init_strategy_context()
    
    def _init_strategy_context(self) -> None:
        """初始化策略上下文"""
        if not STRATEGY_LAYER_AVAILABLE:
            return
        
        try:
            if StrategyContext is not None:
                self._strategy_context = StrategyContext(
                    model=self.model,
                    device=self.device,
                )
            
            if StrategyMetrics is not None:
                self._strategy_metrics = StrategyMetrics()
        except Exception as e:
            logger.warning(f"Failed to init strategy context: {e}")
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个训练步骤
        
        流程：
        1. 各策略prepare_batch
        2. 模型前向传播
        3. 各策略compute_loss
        4. 反向传播和优化
        """
        # 1. 准备批次
        for strategy in self.strategies:
            if hasattr(strategy, 'is_enabled') and strategy.is_enabled:
                if hasattr(strategy, 'prepare_batch'):
                    batch = strategy.prepare_batch(batch, self._strategy_context)
        
        # 2. 前向传播
        self.model.train()
        outputs = self.model(batch)
        
        # 3. 计算损失
        total_loss = torch.tensor(0.0, device=self.device)
        all_metrics = {}
        
        for strategy in self.strategies:
            if not (hasattr(strategy, 'is_enabled') and strategy.is_enabled):
                continue
            
            if hasattr(strategy, 'compute_loss'):
                result = strategy.compute_loss(
                    self.model, batch, outputs, self._strategy_context
                )
                
                if result.loss is not None:
                    total_loss = total_loss + result.loss
                
                for key, value in result.metrics.items():
                    strategy_name = getattr(strategy, 'name', 'unknown')
                    all_metrics[f"{strategy_name}_{key}"] = value
        
        # 4. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        # 5. 策略回调
        for strategy in self.strategies:
            if hasattr(strategy, 'is_enabled') and strategy.is_enabled:
                if hasattr(strategy, 'on_step_end'):
                    strategy.on_step_end(self._strategy_context, result)
        
        # 6. 更新策略指标
        if self._strategy_metrics is not None:
            try:
                self._strategy_metrics.update(all_metrics)
            except Exception:
                pass
        
        return {
            'loss': total_loss.item(),
            'metrics': all_metrics
        }

    def get_strategy_metrics(self) -> Dict[str, Any]:
        """获取策略指标"""
        if self._strategy_metrics is not None:
            try:
                return self._strategy_metrics.to_dict() if hasattr(
                    self._strategy_metrics, 'to_dict'
                ) else {}
            except Exception:
                return {}
        return {}


# ==================== 训练编排器 ====================

class TrainingOrchestrator:
    """训练编排器
    
    生产级训练编排器，负责：
    - 训练计划管理
    - 多阶段训练执行
    - 策略组合调度
    - 状态跟踪和恢复
    - 分布式训练协调
    """
    
    def __init__(
        self,
        output_dir: str = "./training_outputs",
        max_concurrent_jobs: int = 1
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent_jobs = max_concurrent_jobs
        
        # 任务管理
        self.jobs: Dict[str, TrainingJob] = {}
        self.active_job: Optional[TrainingJob] = None
        
        # 回调
        self.on_job_start: Optional[Callable] = None
        self.on_job_end: Optional[Callable] = None
        self.on_phase_start: Optional[Callable] = None
        self.on_phase_end: Optional[Callable] = None
        
        # 策略层组件
        self._strategy_monitor: Optional['StrategyMonitor'] = None
        self._distributed_strategy: Optional['DistributedStrategy'] = None
        
        # 硬件层组件
        self._device_manager: Optional['DeviceManager'] = None
        self._device = None
        
        # 分布式层组件
        self._distributed_manager: Optional['DistributedManager'] = None
        
        # 进度管理器
        self._progress_manager: Optional['TrainingProgressManager'] = None
        
        # 初始化各层组件
        self._init_components()
        
        logger.info(f"TrainingOrchestrator initialized: output_dir={output_dir}")
        logger.info(f"  Strategy layer: {STRATEGY_LAYER_AVAILABLE}")
        logger.info(f"  Distributed strategy: {DISTRIBUTED_STRATEGY_AVAILABLE}")
        logger.info(f"  Hardware layer: {HARDWARE_LAYER_AVAILABLE}")
        logger.info(f"  Distributed layer: {DISTRIBUTED_LAYER_AVAILABLE}")
        logger.info(f"  Losses layer: {LOSSES_LAYER_AVAILABLE}")
        logger.info(f"  Progress manager: {PROGRESS_MANAGER_AVAILABLE}")
    
    def _init_components(self) -> None:
        """初始化各层组件"""
        self._init_strategy_components()
        self._init_hardware_components()
        self._init_distributed_components()
        self._init_progress_manager()
    
    def _init_strategy_components(self) -> None:
        """初始化策略层组件"""
        if not STRATEGY_LAYER_AVAILABLE:
            return
        
        try:
            if StrategyMonitor is not None:
                self._strategy_monitor = StrategyMonitor()
        except Exception as e:
            logger.warning(f"Failed to init strategy components: {e}")
    
    def _init_hardware_components(self) -> None:
        """初始化硬件层组件"""
        if not HARDWARE_LAYER_AVAILABLE:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            return
        
        try:
            if get_device_manager is not None:
                self._device_manager = get_device_manager()
                if self._device_manager is not None and hasattr(self._device_manager, 'device'):
                    self._device = self._device_manager.device
        except Exception as e:
            logger.warning(f"Failed to init hardware components: {e}")
        
        if self._device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def _init_distributed_components(self) -> None:
        """初始化分布式层组件"""
        if DISTRIBUTED_LAYER_AVAILABLE and get_distributed_manager is not None:
            try:
                self._distributed_manager = get_distributed_manager()
            except Exception as e:
                logger.warning(f"Failed to init distributed components: {e}")
    
    def _init_progress_manager(self) -> None:
        """初始化进度管理器"""
        if PROGRESS_MANAGER_AVAILABLE and get_progress_manager is not None:
            try:
                self._progress_manager = get_progress_manager()
            except Exception as e:
                logger.warning(f"Failed to init progress manager: {e}")
    
    def create_plan(
        self,
        model_name: str,
        stages: List[Union[str, TrainingPhase]],
        **kwargs
    ) -> TrainingPlan:
        """创建训练计划
        
        Args:
            model_name: 模型名称
            stages: 训练阶段列表
            **kwargs: 其他配置
        
        Returns:
            训练计划
        """
        plan = TrainingPlan(
            model_name=model_name,
            output_dir=str(self.output_dir / model_name)
        )
        
        for stage in stages:
            if isinstance(stage, str):
                stage = TrainingPhase(stage)
            plan.add_stage(stage)
        
        # 应用额外配置
        for key, value in kwargs.items():
            if hasattr(plan, key):
                setattr(plan, key, value)
        
        logger.info(f"Training plan created: {plan.plan_id}, stages={[s.value for s in plan.stages]}")
        return plan
    
    def create_industry_plan(
        self,
        model_name: str = "industry_model",
        include_pretrain: bool = True,
        include_align: bool = True,
        include_finetune: bool = True,
        stage_configs: Optional[Dict[str, Dict]] = None,
        use_distributed: bool = False,
        distributed_mode: str = "ddp",
    ) -> TrainingPlan:
        """创建行业模型三阶段训练计划
        
        Args:
            model_name: 模型名称
            include_pretrain: 是否包含预训练阶段
            include_align: 是否包含对齐阶段
            include_finetune: 是否包含精调阶段
            stage_configs: 阶段配置
            use_distributed: 是否使用分布式
            distributed_mode: 分布式模式
        
        Returns:
            行业训练计划
        """
        stages = []
        
        if include_pretrain:
            stages.append(TrainingPhase.PRETRAIN_INDUSTRY)
        if include_align:
            stages.append(TrainingPhase.ALIGN_INDUSTRY)
        if include_finetune:
            stages.append(TrainingPhase.FINETUNE_SCENE)
        
        plan = self.create_plan(model_name, stages)
        plan.use_distributed = use_distributed
        plan.distributed_mode = distributed_mode
        
        # 使用分布式策略层推荐模式
        if use_distributed and DISTRIBUTED_STRATEGY_AVAILABLE and recommend_distributed_mode is not None:
            try:
                recommended = recommend_distributed_mode()
                if recommended:
                    logger.info(f"Recommended distributed mode: {recommended}")
            except Exception:
                pass
        
        # 应用阶段配置
        if stage_configs:
            plan.stage_configs = stage_configs
        else:
            # 默认行业训练配置
            plan.stage_configs = {
                TrainingPhase.PRETRAIN_INDUSTRY.value: {
                    'epochs': 3,
                    'learning_rate': 1e-4,
                    'batch_size': 32,
                    'warmup_ratio': 0.1,
                    'description': '行业表征预训练'
                },
                TrainingPhase.ALIGN_INDUSTRY.value: {
                    'epochs': 5,
                    'learning_rate': 2e-5,
                    'batch_size': 16,
                    'warmup_ratio': 0.1,
                    'description': '行业能力对齐'
                },
                TrainingPhase.FINETUNE_SCENE.value: {
                    'epochs': 10,
                    'learning_rate': 1e-5,
                    'batch_size': 8,
                    'freeze_backbone': True,
                    'description': '场景精调'
                }
            }
        
        return plan
    
    def submit_job(self, plan: TrainingPlan) -> TrainingJob:
        """提交训练任务
        
        Args:
            plan: 训练计划
        
        Returns:
            训练任务
        """
        job = TrainingJob(plan=plan)
        self.jobs[job.job_id] = job
        
        # 创建进度跟踪器
        if self._progress_manager is not None:
            try:
                self._progress_manager.create_progress_tracker(
                    session_id=job.job_id,
                    total_epochs=sum(
                        plan.stage_configs.get(s.value, {}).get('epochs', 0)
                        for s in plan.stages
                    )
                )
            except Exception:
                pass
        
        logger.info(f"Training job submitted: {job.job_id}")
        return job
    
    def execute_job(
        self,
        job: TrainingJob,
        model: nn.Module,
        train_dataloaders: Dict[str, DataLoader],
        strategies: Optional[List] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """执行训练任务
        
        Args:
            job: 训练任务
            model: 模型
            train_dataloaders: 各阶段的数据加载器
            strategies: 训练策略列表
            progress_callback: 进度回调
        
        Returns:
            执行结果
        """
        if job.plan is None:
            raise ValueError("Job has no training plan")
        
        self.active_job = job
        job.status = TrainingStatus.RUNNING
        job.started_at = datetime.now()
        
        # 更新进度管理器
        if self._progress_manager is not None:
            try:
                self._progress_manager.set_status(job.job_id, 'running')
            except Exception:
                pass
        
        # 清理内存
        self._clear_memory()
        
        # 创建策略上下文
        if STRATEGY_LAYER_AVAILABLE and StrategyContext is not None:
            try:
                job.strategy_context = StrategyContext(
                    model=model,
                    device=self._device,
                    config=job.plan.to_dict(),
                )
            except Exception:
                pass
        
        # 设置分布式训练
        if job.plan.use_distributed:
            model = self._setup_distributed(model, job.plan)
        
        # 回调
        on_job_start_callback = self.on_job_start
        if on_job_start_callback is not None and callable(on_job_start_callback):
            on_job_start_callback(job)  # pylint: disable=not-callable
        
        try:
            # 创建输出目录
            job_output_dir = self.output_dir / job.job_id
            job_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 依次执行各阶段
            current_model_path = None
            
            for phase in job.plan.stages:
                # 检查是否应该停止
                if job.should_stop:
                    job.status = TrainingStatus.CANCELLED
                    break
                
                # 等待暂停恢复
                while job.should_pause:
                    job.status = TrainingStatus.PAUSED
                    time.sleep(1)
                
                job.status = TrainingStatus.RUNNING
                job.current_phase = phase
                
                # 更新进度管理器
                if self._progress_manager is not None:
                    try:
                        self._progress_manager.update_stage_progress(
                            job.job_id, phase.value, 0
                        )
                    except Exception:
                        pass
                
                # 执行阶段
                result = self._execute_phase(
                    phase=phase,
                    job=job,
                    model=model,
                    train_loader=train_dataloaders.get(phase.value),
                    strategies=strategies,
                    output_dir=job_output_dir,
                    prev_model_path=current_model_path,
                    progress_callback=progress_callback
                )
                
                job.phase_results.append(result)
                
                if result.status == TrainingStatus.FAILED:
                    job.status = TrainingStatus.FAILED
                    break
                
                # 更新模型路径（用于阶段间传递）
                if result.model_path and job.plan.pass_model_between_stages:
                    current_model_path = result.model_path
                
                # 更新进度管理器
                if self._progress_manager is not None:
                    try:
                        self._progress_manager.update_stage_progress(
                            job.job_id, phase.value, 100
                        )
                    except Exception:
                        pass
            
            # 完成
            if job.status == TrainingStatus.RUNNING:
                job.status = TrainingStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Job execution failed: {e}")
            job.status = TrainingStatus.FAILED
            
            # 更新进度管理器
            if self._progress_manager is not None:
                try:
                    self._progress_manager.set_status(job.job_id, 'failed', str(e))
                except Exception:
                    pass
            
            raise
        
        finally:
            job.completed_at = datetime.now()
            self.active_job = None
            
            # 更新进度管理器
            if self._progress_manager is not None:
                try:
                    self._progress_manager.set_status(job.job_id, job.status.value)
                except Exception:
                    pass
            
            # 回调
            on_job_end_callback = self.on_job_end
            if on_job_end_callback is not None and callable(on_job_end_callback):
                on_job_end_callback(job)  # pylint: disable=not-callable
            
            # 清理内存
            self._clear_memory()
        
        return self._build_job_result(job)
    
    def _setup_distributed(self, model: nn.Module, plan: TrainingPlan) -> nn.Module:
        """设置分布式训练"""
        # 使用分布式策略层
        if DISTRIBUTED_STRATEGY_AVAILABLE and DistributedStrategy is not None:
            try:
                if DistributedMode is not None:
                    mode_map = {
                        'ddp': DistributedMode.DDP,
                        'fsdp': DistributedMode.FSDP,
                        'zero': DistributedMode.ZERO,
                    }
                    mode = mode_map.get(plan.distributed_mode)
                    
                    if mode and DistributedStrategyConfig is not None:
                        config = DistributedStrategyConfig(
                            mode=mode,
                            world_size=plan.world_size,
                        )
                        self._distributed_strategy = DistributedStrategy(config)
                        
                        if STRATEGY_LAYER_AVAILABLE and StrategyContext is not None:
                            ctx = StrategyContext(model=model, device=self._device)
                            self._distributed_strategy.setup(ctx)
                            return ctx.model
            except Exception as e:
                logger.warning(f"Failed to setup distributed via strategy: {e}")
        
        # 使用分布式层
        if DISTRIBUTED_LAYER_AVAILABLE and self._distributed_manager is not None:
            try:
                if hasattr(self._distributed_manager, 'wrap_model'):
                    return self._distributed_manager.wrap_model(model)
            except Exception as e:
                logger.warning(f"Failed to setup distributed via layer: {e}")
        
        # 回退到 PyTorch DDP
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            try:
                model = nn.DataParallel(model)
                logger.info("Using PyTorch DataParallel")
            except Exception:
                pass
        
        return model
    
    def _execute_phase(
        self,
        phase: TrainingPhase,
        job: TrainingJob,
        model: nn.Module,
        train_loader: Optional[DataLoader],
        strategies: Optional[List],
        output_dir: Path,
        prev_model_path: Optional[str],
        progress_callback: Optional[Callable]
    ) -> StageResult:
        """执行单个训练阶段"""
        result = StageResult(
            phase=phase,
            status=TrainingStatus.RUNNING,
            start_time=datetime.now()
        )
        
        # 回调
        on_phase_start_callback = self.on_phase_start
        if on_phase_start_callback is not None and callable(on_phase_start_callback):
            on_phase_start_callback(job, phase)  # pylint: disable=not-callable
        if job.plan.on_stage_start:
            job.plan.on_stage_start(phase)
        
        try:
            # 获取阶段配置
            stage_config = job.plan.get_stage_config(phase)
            
            # 加载上一阶段模型（如果需要）
            if prev_model_path and job.plan.pass_model_between_stages:
                self._load_model(model, prev_model_path)
            
            # 执行训练
            if train_loader:
                phase_metrics = self._train_phase(
                    model=model,
                    train_loader=train_loader,
                    strategies=strategies,
                    config=stage_config,
                    phase=phase,
                    job=job,
                    progress_callback=progress_callback
                )
                result.metrics = phase_metrics
            else:
                logger.warning(f"No data loader for phase {phase.value}, skipping")
            
            # 保存模型
            if job.plan.save_intermediate_models:
                model_path = output_dir / f"{phase.value}_model.pt"
                self._save_model(model, model_path)
                result.model_path = str(model_path)
            
            result.status = TrainingStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"Phase {phase.value} failed: {e}")
            result.status = TrainingStatus.FAILED
            result.error = str(e)
        
        finally:
            result.end_time = datetime.now()
            
            # 回调
            on_phase_end_callback = self.on_phase_end
            if on_phase_end_callback is not None and callable(on_phase_end_callback):
                on_phase_end_callback(job, phase, result)  # pylint: disable=not-callable
            if job.plan.on_stage_end:
                job.plan.on_stage_end(phase, result)
        
        return result
    
    def _train_phase(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        strategies: Optional[List],
        config: Dict[str, Any],
        phase: TrainingPhase,
        job: TrainingJob,
        progress_callback: Optional[Callable]
    ) -> Dict[str, float]:
        """执行阶段训练"""
        model = model.to(self._device)
        
        # 配置
        epochs = config.get('epochs', 3)
        lr = config.get('learning_rate', 1e-4)
        freeze_backbone = config.get('freeze_backbone', False)
        
        # 冻结backbone
        if freeze_backbone:
            self._freeze_backbone(model)
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr
        )
        
        # 创建损失函数（使用损失层）
        criterion = None
        if LOSSES_LAYER_AVAILABLE and create_loss is not None:
            try:
                loss_type = config.get('loss_type', 'cross_entropy')
                criterion = create_loss(loss_type)
            except Exception:
                pass
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        # 训练循环
        total_loss = 0.0
        total_steps = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # 移动数据到设备
                batch = self._move_batch_to_device(batch, self._device)
                
                # 前向传播
                model.train()
                outputs = model(batch)
                
                # 计算损失
                if strategies:
                    loss = self._compute_strategies_loss(
                        model, batch, outputs, strategies, job
                    )
                else:
                    loss = self._get_default_loss(outputs, criterion)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_steps += 1
                total_steps += 1
            
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            total_loss += epoch_loss
            
            # 进度回调
            if progress_callback:
                progress_callback(phase.value, epoch + 1, {
                    'loss': avg_epoch_loss,
                    'epoch': epoch + 1,
                    'steps': total_steps
                })
            
            # 更新进度管理器
            if self._progress_manager is not None:
                try:
                    progress_pct = ((epoch + 1) / epochs) * 100
                    self._progress_manager.update_stage_progress(
                        job.job_id, phase.value, progress_pct,
                        train_loss=avg_epoch_loss,
                        current_epoch=epoch + 1,
                    )
                except Exception:
                    pass
            
            logger.info(f"[{phase.value}] Epoch {epoch+1}/{epochs}: loss={avg_epoch_loss:.4f}")
        
        return {
            'final_loss': total_loss / max(total_steps, 1),
            'total_steps': total_steps,
            'epochs': epochs
        }
    
    def _compute_strategies_loss(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        outputs: Any,
        strategies: List,
        job: TrainingJob
    ) -> torch.Tensor:
        """计算策略组合损失"""
        total_loss = torch.tensor(0.0, device=self._device)
        
        for strategy in strategies:
            if hasattr(strategy, 'is_enabled') and strategy.is_enabled:
                if hasattr(strategy, 'compute_loss'):
                    result = strategy.compute_loss(
                        model, batch, outputs, job.strategy_context
                    )
                    if result.loss is not None:
                        total_loss = total_loss + result.loss
        
        return total_loss
    
    def _get_default_loss(self, outputs: Any, criterion: nn.Module) -> torch.Tensor:
        """获取默认损失"""
        if isinstance(outputs, dict):
            if 'loss' in outputs:
                return outputs['loss']
            # 尝试计算分类损失
            if 'logits' in outputs and 'labels' in outputs:
                return criterion(outputs['logits'], outputs['labels'])
            # 计算所有场景头的损失
            losses = []
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    losses.append(value.mean())
            if losses:
                return sum(losses) / len(losses)
        elif hasattr(outputs, 'loss'):
            return outputs.loss
        
        raise ValueError("Cannot determine loss from outputs")
    
    def _move_batch_to_device(
        self, 
        batch: Dict[str, Any], 
        device: torch.device
    ) -> Dict[str, Any]:
        """移动批次数据到设备"""
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device)
            elif isinstance(value, dict):
                result[key] = self._move_batch_to_device(value, device)
            else:
                result[key] = value
        return result
    
    def _freeze_backbone(self, model: nn.Module) -> None:
        """冻结backbone"""
        for name, param in model.named_parameters():
            if 'backbone' in name.lower():
                param.requires_grad = False
    
    def _save_model(self, model: nn.Module, path: Path) -> None:
        """保存模型"""
        # 处理 DataParallel 包装
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state_dict, path)
        logger.info(f"Model saved: {path}")
    
    def _load_model(self, model: nn.Module, path: str) -> None:
        """加载模型"""
        state_dict = torch.load(path, map_location='cpu')
        # 处理 DataParallel 包装
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Model loaded: {path}")
    
    def _build_job_result(self, job: TrainingJob) -> Dict[str, Any]:
        """构建任务结果"""
        return {
            'job_id': job.job_id,
            'status': job.status.value,
            'started_at': job.started_at.isoformat() if job.started_at else None,
            'completed_at': job.completed_at.isoformat() if job.completed_at else None,
            'duration': (job.completed_at - job.started_at).total_seconds() if job.started_at and job.completed_at else 0,
            'phases': [r.to_dict() for r in job.phase_results],
            'layers_available': {
                'strategy': STRATEGY_LAYER_AVAILABLE,
                'distributed_strategy': DISTRIBUTED_STRATEGY_AVAILABLE,
                'hardware': HARDWARE_LAYER_AVAILABLE,
                'distributed': DISTRIBUTED_LAYER_AVAILABLE,
                'losses': LOSSES_LAYER_AVAILABLE,
                'progress': PROGRESS_MANAGER_AVAILABLE,
            }
        }
    
    def _clear_memory(self) -> None:
        """清理内存"""
        if HARDWARE_LAYER_AVAILABLE and clear_memory is not None:
            try:
                clear_memory()
            except Exception:
                pass
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
    def stop_job(self, job_id: str) -> bool:
        """停止任务"""
        if job_id in self.jobs:
            self.jobs[job_id].should_stop = True
            return True
        return False
    
    def pause_job(self, job_id: str) -> bool:
        """暂停任务"""
        if job_id in self.jobs:
            self.jobs[job_id].should_pause = True
            return True
        return False
    
    def resume_job(self, job_id: str) -> bool:
        """恢复任务"""
        if job_id in self.jobs:
            self.jobs[job_id].should_pause = False
            return True
        return False
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            return {
                'job_id': job.job_id,
                'status': job.status.value,
                'current_phase': job.current_phase.value if job.current_phase else None,
                'completed_phases': len(job.phase_results),
                'total_phases': len(job.plan.stages) if job.plan else 0
            }
        return None

    def diagnose(self) -> Dict[str, Any]:
        """诊断编排器状态"""
        return {
            'active_job': self.active_job.job_id if self.active_job else None,
            'total_jobs': len(self.jobs),
            'output_dir': str(self.output_dir),
            'device': str(self._device),
            'layers_available': {
                'strategy': STRATEGY_LAYER_AVAILABLE,
                'distributed_strategy': DISTRIBUTED_STRATEGY_AVAILABLE,
                'hardware': HARDWARE_LAYER_AVAILABLE,
                'distributed': DISTRIBUTED_LAYER_AVAILABLE,
                'losses': LOSSES_LAYER_AVAILABLE,
                'progress': PROGRESS_MANAGER_AVAILABLE,
            },
            'components': {
                'strategy_monitor': self._strategy_monitor is not None,
                'distributed_strategy': self._distributed_strategy is not None,
                'device_manager': self._device_manager is not None,
                'distributed_manager': self._distributed_manager is not None,
                'progress_manager': self._progress_manager is not None,
            }
        }


# ==================== 便捷函数 ====================

def create_orchestrator(**kwargs) -> TrainingOrchestrator:
    """创建训练编排器的工厂函数"""
    return TrainingOrchestrator(**kwargs)


def get_layer_availability() -> Dict[str, bool]:
    """获取各层可用性"""
    return {
        'strategy': STRATEGY_LAYER_AVAILABLE,
        'distributed_strategy': DISTRIBUTED_STRATEGY_AVAILABLE,
        'hardware': HARDWARE_LAYER_AVAILABLE,
        'distributed': DISTRIBUTED_LAYER_AVAILABLE,
        'losses': LOSSES_LAYER_AVAILABLE,
        'progress': PROGRESS_MANAGER_AVAILABLE,
    }


# ==================== 导出 ====================

__all__ = [
    # 主要类
    'TrainingOrchestrator',
    'StrategyTrainer',
    
    # 数据类
    'TrainingPlan',
    'TrainingJob',
    'StageResult',
    
    # 枚举
    'TrainingPhase',
    'TrainingStatus',
    
    # 便捷函数
    'create_orchestrator',
    'get_layer_availability',
    
    # 层可用性标志
    'STRATEGY_LAYER_AVAILABLE',
    'DISTRIBUTED_STRATEGY_AVAILABLE',
    'HARDWARE_LAYER_AVAILABLE',
    'DISTRIBUTED_LAYER_AVAILABLE',
    'LOSSES_LAYER_AVAILABLE',
    'PROGRESS_MANAGER_AVAILABLE',
]

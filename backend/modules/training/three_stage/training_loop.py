# -*- coding: utf-8 -*-
"""
训练循环核心模块

实现标准化的训练循环，包括：
- 前向传播计算预测值
- 计算损失函数评估误差
- 反向传播计算梯度
- 使用优化器更新参数
- 循环迭代直到模型收敛

生产级功能：
- 策略层集成（StrategyContext, StrategyMetrics）
- 硬件层集成（DeviceManager, MemoryManager）
- 损失层集成（LossFactory）
- 进度管理器集成（TrainingProgressManager）

架构调用层次：
├── training_loop.py (本模块)
│   ├── 调用 optimizer_utils.py (优化器工具)
│   ├── 调用 backend/modules/training/strategies (策略层)
│   ├── 调用 backend/lib/hardware (硬件层)
│   └── 调用 backend/modules/training/progress (进度管理)
└── 被 three_stage_trainer.py 调用
"""

import time
import logging
import sys
import os
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# 处理导入路径
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from .optimizer_utils import (
    OptimizerConfig, TrainingState,
    create_optimizer, create_scheduler,
    clip_gradients, compute_gradient_norm,
    GradientAccumulator, ConvergenceDetector,
    MixedPrecisionManager, log_training_info
)

logger = logging.getLogger(__name__)


# ==================== 策略层导入 ====================

from backend.modules.training.strategies.base_strategy import (
    StrategyContext, StrategyMetrics,
)


# ==================== 硬件层导入 ====================

from backend.lib.hardware import (
    get_available_memory, clear_memory,
)


# ==================== 损失层导入 ====================

LOSSES_LAYER_AVAILABLE = False
LossFactory = None
create_loss = None

try:
    from backend.lib.losses import (
        LossFactory, create_loss,
    )
    LOSSES_LAYER_AVAILABLE = True
    logger.info("Losses layer loaded for training_loop")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Losses layer not available for training_loop: {e}")


# ==================== 进度管理导入 ====================

PROGRESS_MANAGER_AVAILABLE = False
TrainingProgressManager = None

try:
    from backend.modules.training.progress.progress_manager import (
        TrainingProgressManager,
    )
    PROGRESS_MANAGER_AVAILABLE = True
    logger.info("Progress manager loaded for training_loop")
except (ImportError, SyntaxError, IndentationError) as e:
    logger.warning(f"Progress manager not available for training_loop: {e}")


class TrainingStage(Enum):
    """训练阶段"""
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"  
    PREFERENCE = "preference"


@dataclass
class TrainingMetrics:
    """训练指标"""
    loss: float = 0.0
    accuracy: float = 0.0
    perplexity: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    throughput: float = 0.0  # samples/sec
    step_time: float = 0.0
    
    # DPO 特定指标
    chosen_reward: float = 0.0
    rejected_reward: float = 0.0
    reward_margin: float = 0.0
    
    # 附加指标
    extra: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, float]:
        result = {
            'loss': self.loss,
            'accuracy': self.accuracy,
            'perplexity': self.perplexity,
            'learning_rate': self.learning_rate,
            'gradient_norm': self.gradient_norm,
            'throughput': self.throughput,
            'step_time': self.step_time,
        }
        if self.chosen_reward != 0:
            result['chosen_reward'] = self.chosen_reward
            result['rejected_reward'] = self.rejected_reward
            result['reward_margin'] = self.reward_margin
        result.update(self.extra)
        return result


@dataclass 
class TrainingLoopConfig:
    """训练循环配置"""
    # 基础配置
    epochs: int = 3
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    
    # 优化配置
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    # 混合精度
    use_fp16: bool = True
    fp16_opt_level: str = "O1"
    
    # 收敛和早停
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    
    # 回调配置
    max_steps: int = -1  # -1表示不限制


class TrainingLoop:
    """
    标准训练循环实现
    
    实现完整的训练流程：
    1. 初始化参数（权重和偏置）
    2. 前向传播计算预测值
    3. 计算损失函数评估误差
    4. 反向传播计算梯度
    5. 优化器更新参数
    6. 重复迭代直到收敛或达到最大轮数
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingLoopConfig,
        device: torch.device,
        progress_callback: Optional[Callable] = None,
        status_checker: Optional[Callable] = None,
        control_session_id: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.device = device
        self.progress_callback = progress_callback
        self.status_checker = status_checker
        self.control_session_id = control_session_id
        
        # 训练组件
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.gradient_accumulator: Optional[GradientAccumulator] = None
        self.convergence_detector: Optional[ConvergenceDetector] = None
        self.mixed_precision: Optional[MixedPrecisionManager] = None
        
        # 状态跟踪
        self.state = TrainingState()
        self.metrics_history: List[TrainingMetrics] = []
        
        # 参考模型（用于DPO）
        self.ref_model: Optional[nn.Module] = None
        
        # 扩展组件初始化
        self.progress_manager: Optional[TrainingProgressManager] = None
        self.loss_factory: Optional[LossFactory] = None
        self._init_extended_components()
    
    def _init_extended_components(self) -> None:
        """初始化扩展组件"""
        # 初始化进度管理器
        try:
            self.progress_manager = TrainingProgressManager()
        except Exception as e:
            logger.warning(f"Failed to init progress manager: {e}")
        
        # 初始化损失工厂
        try:
            self.loss_factory = LossFactory()
        except Exception as e:
            logger.warning(f"Failed to init loss factory: {e}")

    def create_custom_loss(self, loss_name: str, **kwargs) -> Optional[nn.Module]:
        """使用 LossFactory 创建自定义损失"""
        try:
            return create_loss(loss_name, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to create loss {loss_name}: {e}")
        
        return None

    def update_task_progress(self, stage: str, current: int, total: int, metrics: Dict[str, float]) -> None:
        """更新任务进度"""
        if self.progress_manager and self.control_session_id:
            try:
                percent = min(100.0, (current / max(1, total)) * 100)
                self.progress_manager.update_progress(
                    self.control_session_id,
                    current_stage=stage,
                    progress=percent,
                    metrics=metrics,
                    status="running"
                )
            except Exception:
                pass

    def setup(
        self,
        train_loader: DataLoader,
        # eval_loader: Optional[DataLoader] = None, # Unused
        ref_model: Optional[nn.Module] = None
    ) -> None:
        """
        设置训练环境
        
        Args:
            train_loader: 训练数据加载器
            # eval_loader: 评估数据加载器
            ref_model: 参考模型（用于DPO）
        """
        # 计算总步数
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * self.config.epochs
        
        if self.config.max_steps > 0:
            total_steps = min(total_steps, self.config.max_steps)
        
        # 更新配置
        self.config.optimizer_config.num_training_steps = total_steps
        
        # 创建优化器
        self.optimizer = create_optimizer(self.model, self.config.optimizer_config)
        
        # 创建调度器
        self.scheduler = create_scheduler(
            self.optimizer, 
            self.config.optimizer_config,
            total_steps
        )
        
        # 创建梯度累积器
        self.gradient_accumulator = GradientAccumulator(
            self.config.optimizer_config.gradient_accumulation_steps
        )
        
        # 创建收敛检测器
        if self.config.early_stopping:
            self.convergence_detector = ConvergenceDetector(
                patience=self.config.early_stopping_patience,
                threshold=self.config.early_stopping_threshold
            )
        
        # 创建混合精度管理器
        self.mixed_precision = MixedPrecisionManager(
            enabled=self.config.use_fp16,
            device=self.device
        )
        
        # 设置参考模型
        if ref_model is not None:
            self.ref_model = ref_model
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        
        # 记录训练信息
        log_training_info(self.model, self.optimizer, self.scheduler)
        
        logger.info(f"Training setup complete: {total_steps} total steps, {steps_per_epoch} steps/epoch")
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        stage: TrainingStage = TrainingStage.FINETUNE
    ) -> TrainingMetrics:
        """
        训练一个epoch
        
        完整的训练循环：
        1. 遍历数据批次
        2. 前向传播
        3. 计算损失
        4. 反向传播
        5. 更新参数
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            stage: 训练阶段
        
        Returns:
            该epoch的平均指标
        """
        self.model.train()
        
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_samples = 0
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # 检查是否需要停止
            if self._should_stop():
                logger.info("Training stopped by status checker")
                break
            
            # 检查最大步数限制
            if self.config.max_steps > 0 and self.state.global_step >= self.config.max_steps:
                logger.info(f"Reached max steps: {self.config.max_steps}")
                break
            
            step_start_time = time.time()
            
            # 执行训练步骤
            if stage == TrainingStage.PREFERENCE:
                metrics = self._train_step_dpo(batch)
            else:
                metrics = self._train_step(batch)
            
            step_time = time.time() - step_start_time
            metrics.step_time = step_time
            
            # 更新状态
            epoch_loss += metrics.loss
            epoch_steps += 1
            epoch_samples += self._get_batch_size(batch)
            self.state.global_step += 1
            self.state.total_loss += metrics.loss
            self.state.total_samples += self._get_batch_size(batch)
            
            # 日志记录
            if self.state.global_step % self.config.logging_steps == 0:
                self._log_step(epoch, batch_idx, len(train_loader), metrics)
                # 更新细粒度进度
                current_epoch_progress = (batch_idx + 1) / len(train_loader)
                global_progress = (epoch + current_epoch_progress)
                self.update_task_progress(stage.value, global_progress, self.config.epochs, metrics.to_dict())
            
            # 收敛检测
            if self.convergence_detector and self.convergence_detector.update(metrics.loss):
                logger.info(f"Early stopping triggered at step {self.state.global_step}")
                self.state.is_converged = True
                self.state.convergence_step = self.state.global_step
                break
        
        # 计算epoch指标
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / max(1, epoch_steps)
        throughput = epoch_samples / max(0.001, epoch_time)
        
        epoch_metrics = TrainingMetrics(
            loss=avg_loss,
            perplexity=self._compute_perplexity(avg_loss),
            learning_rate=self._get_current_lr(),
            throughput=throughput,
            step_time=epoch_time / max(1, epoch_steps)
        )
        
        self.state.epoch = epoch + 1
        self.metrics_history.append(epoch_metrics)
        
        # 更新最佳损失
        if avg_loss < self.state.best_loss:
            self.state.best_loss = avg_loss
            self.state.patience_counter = 0
        else:
            self.state.patience_counter += 1
        
        logger.info(f"Epoch {epoch + 1} completed: loss={avg_loss:.4f}, "
                   f"perplexity={epoch_metrics.perplexity:.2f}, "
                   f"throughput={throughput:.1f} samples/sec")
        
        # 进度回调
        if self.progress_callback:
            self.progress_callback(stage.value, epoch + 1, epoch_metrics.to_dict())
            
        # 更新任务进度 (ProgressManager)
        self.update_task_progress(stage.value, epoch + 1, self.config.epochs, epoch_metrics.to_dict())
        
        return epoch_metrics
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """
        执行单个训练步骤（标准语言模型训练）
        
        流程：
        1. 数据移动到设备
        2. 前向传播
        3. 计算损失
        4. 反向传播
        5. 梯度裁剪
        6. 优化器步骤
        7. 学习率调度
        
        Args:
            batch: 输入批次
        
        Returns:
            训练指标
        """
        # 1. 数据准备
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        # 2. 前向传播（计算预测值）
        with self.mixed_precision.autocast_context():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # 3. 计算损失（评估误差）
            loss = outputs.loss
            
            # 梯度累积缩放
            loss = self.gradient_accumulator.scale_loss(loss)
        
        # 4. 反向传播（计算梯度）
        if self.mixed_precision.enabled:
            self.mixed_precision.scale_loss(loss).backward()
        else:
            loss.backward()
        
        # 5. 优化器步骤（更新参数）
        if self.gradient_accumulator.should_step():
            # 反缩放梯度（如果使用混合精度）
            self.mixed_precision.unscale_and_step(self.optimizer)
            
            # 梯度裁剪
            grad_norm = clip_gradients(
                self.model,
                self.config.optimizer_config.gradient_clipping
            )
            
            # 优化器步骤
            self.mixed_precision.step(self.optimizer)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
        else:
            grad_norm = compute_gradient_norm(self.model)
        
        # 计算指标
        loss_value = loss.item() * self.config.optimizer_config.gradient_accumulation_steps
        
        return TrainingMetrics(
            loss=loss_value,
            perplexity=self._compute_perplexity(loss_value),
            learning_rate=self._get_current_lr(),
            gradient_norm=grad_norm
        )
    
    def _train_step_dpo(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """
        执行单个DPO训练步骤
        
        DPO损失公式：
        L = -log(sigmoid(beta * (r_chosen - r_rejected)))
        其中 r = log(pi(y|x)) - log(ref(y|x))
        
        Args:
            batch: DPO批次数据
        
        Returns:
            训练指标
        """
        # 获取数据
        chosen_input_ids = batch['chosen_input_ids'].to(self.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
        rejected_input_ids = batch['rejected_input_ids'].to(self.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
        
        beta = 0.1  # DPO温度参数
        
        with self.mixed_precision.autocast_context():
            # 策略模型前向传播
            chosen_outputs = self.model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask
            )
            rejected_outputs = self.model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask
            )
            
            # 计算对数概率
            chosen_logps = self._compute_logps(chosen_outputs.logits, chosen_input_ids)
            rejected_logps = self._compute_logps(rejected_outputs.logits, rejected_input_ids)
            
            # 参考模型前向传播
            with torch.no_grad():
                ref_chosen_outputs = self.ref_model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask
                )
                ref_rejected_outputs = self.ref_model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask
                )
                
                ref_chosen_logps = self._compute_logps(ref_chosen_outputs.logits, chosen_input_ids)
                ref_rejected_logps = self._compute_logps(ref_rejected_outputs.logits, rejected_input_ids)
            
            # 计算奖励差
            pi_logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = pi_logratios - ref_logratios
            
            # DPO损失
            # pylint: disable=not-callable
            loss = -F.logsigmoid(beta * logits).mean()
            loss = self.gradient_accumulator.scale_loss(loss)
        
        # 反向传播
        if self.mixed_precision.enabled:
            self.mixed_precision.scale_loss(loss).backward()
        else:
            loss.backward()
        
        # 优化器步骤
        grad_norm = 0.0
        if self.gradient_accumulator.should_step():
            self.mixed_precision.unscale_and_step(self.optimizer)
            grad_norm = clip_gradients(
                self.model,
                self.config.optimizer_config.gradient_clipping
            )
            self.mixed_precision.step(self.optimizer)
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()
        
        # 计算指标
        loss_value = loss.item() * self.config.optimizer_config.gradient_accumulation_steps
        
        return TrainingMetrics(
            loss=loss_value,
            learning_rate=self._get_current_lr(),
            gradient_norm=grad_norm,
            chosen_reward=chosen_logps.mean().item(),
            rejected_reward=rejected_logps.mean().item(),
            reward_margin=(chosen_logps - rejected_logps).mean().item()
        )
    
    def _compute_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        average: bool = True
    ) -> torch.Tensor:
        """
        计算序列的对数概率
        
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
        
        # 计算log softmax
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        
        # 获取目标token的概率
        per_token_logps = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # 创建mask（忽略padding）
        mask = (shift_labels != -100).float()
        per_token_logps = per_token_logps * mask
        
        if average:
            return per_token_logps.sum(-1) / mask.sum(-1).clamp(min=1)
        else:
            return per_token_logps.sum(-1)
    
    def evaluate(
        self,
        eval_loader: DataLoader,
        stage: TrainingStage = TrainingStage.FINETUNE
    ) -> TrainingMetrics:
        """
        评估模型
        
        Args:
            eval_loader: 评估数据加载器
            stage: 训练阶段
        
        Returns:
            评估指标
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                if stage == TrainingStage.PREFERENCE:
                    metrics = self._eval_step_dpo(batch)
                else:
                    metrics = self._eval_step(batch)
                
                batch_size = self._get_batch_size(batch)
                total_loss += metrics.loss * batch_size
                total_samples += batch_size
        
        self.model.train()
        
        avg_loss = total_loss / max(1, total_samples)
        
        return TrainingMetrics(
            loss=avg_loss,
            perplexity=self._compute_perplexity(avg_loss)
        )
    
    def _eval_step(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """评估步骤（标准）"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        with self.mixed_precision.autocast_context():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
        
        return TrainingMetrics(loss=loss.item())
    
    def _eval_step_dpo(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """评估步骤（DPO）"""
        chosen_input_ids = batch['chosen_input_ids'].to(self.device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(self.device)
        rejected_input_ids = batch['rejected_input_ids'].to(self.device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(self.device)
        
        beta = 0.1
        
        with self.mixed_precision.autocast_context():
            chosen_outputs = self.model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask
            )
            rejected_outputs = self.model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask
            )
            
            chosen_logps = self._compute_logps(chosen_outputs.logits, chosen_input_ids)
            rejected_logps = self._compute_logps(rejected_outputs.logits, rejected_input_ids)
            
            ref_chosen_outputs = self.ref_model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask
            )
            ref_rejected_outputs = self.ref_model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask
            )
            
            ref_chosen_logps = self._compute_logps(ref_chosen_outputs.logits, chosen_input_ids)
            ref_rejected_logps = self._compute_logps(ref_rejected_outputs.logits, rejected_input_ids)
            
            pi_logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logits = pi_logratios - ref_logratios
            
            # pylint: disable=not-callable
            loss = -F.logsigmoid(beta * logits).mean()
        
        return TrainingMetrics(
            loss=loss.item(),
            chosen_reward=chosen_logps.mean().item(),
            rejected_reward=rejected_logps.mean().item(),
            reward_margin=(chosen_logps - rejected_logps).mean().item()
        )
    
    def run(
        self,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        stage: TrainingStage = TrainingStage.FINETUNE
    ) -> Dict[str, Any]:
        """
        运行完整的训练循环
        
        Args:
            train_loader: 训练数据加载器
            eval_loader: 评估数据加载器
            stage: 训练阶段
        
        Returns:
            训练结果
        """
        logger.info(f"Starting training for {self.config.epochs} epochs, stage={stage.value}")
        
        start_time = time.time()
        best_eval_loss = float('inf')
        
        for epoch in range(self.config.epochs):
            logger.info("=" * 50)
            logger.info(f"Epoch {epoch + 1}/{self.config.epochs}")
            logger.info("=" * 50)
            
            # 训练一个epoch
            self.train_epoch(train_loader, epoch, stage)
            
            # 评估
            if eval_loader and (epoch + 1) % 1 == 0:  # 每个epoch评估
                eval_metrics = self.evaluate(eval_loader, stage)
                logger.info(f"Evaluation: loss={eval_metrics.loss:.4f}, "
                           f"perplexity={eval_metrics.perplexity:.2f}")
                
                if eval_metrics.loss < best_eval_loss:
                    best_eval_loss = eval_metrics.loss
                    self.state.best_metrics = eval_metrics.to_dict()
            
            # 检查是否收敛
            if self.state.is_converged:
                logger.info(f"Training converged at epoch {epoch + 1}")
                break
            
            # 检查是否应该停止
            if self._should_stop():
                logger.info("Training stopped by external request")
                break
        
        total_time = time.time() - start_time
        
        # 返回结果
        return {
            'epochs_completed': self.state.epoch,
            'steps_completed': self.state.global_step,
            'final_loss': self.state.total_loss / max(1, self.state.global_step),
            'best_loss': self.state.best_loss,
            'best_metrics': self.state.best_metrics,
            'is_converged': self.state.is_converged,
            'convergence_step': self.state.convergence_step,
            'total_time': total_time,
            'metrics_history': [m.to_dict() for m in self.metrics_history]
        }
    
    # =========================================================================
    # 辅助方法
    # =========================================================================
    
    def _should_stop(self) -> bool:
        """检查是否应该停止训练"""
        if self.status_checker and self.control_session_id:
            status = (self.status_checker(self.control_session_id) or "").lower()
            return status in ("cancelled", "failed", "stopped")
        return False
    
    def _get_current_lr(self) -> float:
        """获取当前学习率"""
        if self.optimizer:
            return self.optimizer.param_groups[0]['lr']
        return 0.0
    
    def _get_batch_size(self, batch: Dict[str, torch.Tensor]) -> int:
        """获取批次大小"""
        if 'input_ids' in batch:
            return batch['input_ids'].size(0)
        elif 'chosen_input_ids' in batch:
            return batch['chosen_input_ids'].size(0)
        return 1
    
    def _compute_perplexity(self, loss: float) -> float:
        """计算困惑度"""
        if loss < 20:  # 防止数值溢出
            return torch.exp(torch.tensor(loss)).item()
        return float('inf')
    
    def _log_step(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        metrics: TrainingMetrics
    ) -> None:
        """记录训练步骤"""
        progress = (batch_idx + 1) / total_batches * 100
        
        log_msg = (
            f"Epoch {epoch + 1} [{batch_idx + 1}/{total_batches}] ({progress:.1f}%) - "
            f"loss: {metrics.loss:.4f}, lr: {metrics.learning_rate:.2e}"
        )
        
        if metrics.gradient_norm > 0:
            log_msg += f", grad_norm: {metrics.gradient_norm:.4f}"
        
        logger.info(log_msg)
    
    def save_checkpoint(
        self,
        path: str,
        include_optimizer: bool = True
    ) -> None:
        """
        保存检查点
        
        Args:
            path: 保存路径
            include_optimizer: 是否包含优化器状态
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'state': {
                'global_step': self.state.global_step,
                'epoch': self.state.epoch,
                'best_loss': self.state.best_loss,
                'is_converged': self.state.is_converged
            },
            'config': {
                'epochs': self.config.epochs,
                'learning_rate': self.config.optimizer_config.learning_rate
            }
        }
        
        if include_optimizer and self.optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> None:
        """
        加载检查点
        
        Args:
            path: 检查点路径
            load_optimizer: 是否加载优化器状态
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'state' in checkpoint:
            state = checkpoint['state']
            self.state.global_step = state.get('global_step', 0)
            self.state.epoch = state.get('epoch', 0)
            self.state.best_loss = state.get('best_loss', float('inf'))
            self.state.is_converged = state.get('is_converged', False)
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint and self.optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")


# =============================================================================
# 工厂函数
# =============================================================================

def create_training_loop(
    model: nn.Module,
    stage_config,  # StageConfig from three_stage_config
    device: torch.device,
    use_fp16: bool = True,
    progress_callback: Optional[Callable] = None,
    status_checker: Optional[Callable] = None,
    control_session_id: Optional[str] = None
) -> TrainingLoop:
    """
    创建训练循环
    
    Args:
        model: 模型
        stage_config: 阶段配置
        device: 设备
        use_fp16: 是否使用混合精度
        progress_callback: 进度回调
        status_checker: 状态检查器
        control_session_id: 控制会话ID
    
    Returns:
        配置好的训练循环
    """
    # 创建优化器配置
    optimizer_config = OptimizerConfig(
        learning_rate=stage_config.learning_rate,
        weight_decay=stage_config.weight_decay,
        warmup_steps=stage_config.warmup_steps,
        gradient_clipping=stage_config.gradient_clipping,
        gradient_accumulation_steps=stage_config.gradient_accumulation_steps
    )
    
    # 创建训练循环配置
    loop_config = TrainingLoopConfig(
        epochs=stage_config.epochs,
        logging_steps=stage_config.logging_steps,
        eval_steps=stage_config.eval_steps,
        save_steps=stage_config.save_steps,
        optimizer_config=optimizer_config,
        use_fp16=use_fp16
    )
    
    # 创建训练循环
    return TrainingLoop(
        model=model,
        config=loop_config,
        device=device,
        progress_callback=progress_callback,
        status_checker=status_checker,
        control_session_id=control_session_id
    )


def get_training_loop_info() -> Dict[str, Any]:
    """获取训练循环信息
    
    Returns:
        训练循环信息字典
    """
    return {
        'supported_stages': [
            TrainingStage.PRETRAIN.value,
            TrainingStage.FINETUNE.value,
            TrainingStage.PREFERENCE.value,
        ],
        'features': {
            'mixed_precision': True,
            'gradient_accumulation': True,
            'early_stopping': True,
            'checkpoint_saving': True,
            'dpo_training': True,
        }
    }


def create_strategy_context_for_loop(
    model: nn.Module,
    device: torch.device,
    stage: TrainingStage,
    config: Dict[str, Any] = None,
) -> Optional['StrategyContext']:
    """为训练循环创建策略上下文
    
    Args:
        model: 模型
        device: 设备
        stage: 训练阶段
        config: 配置字典
    
    Returns:
        策略上下文或 None
    """
    try:
        return StrategyContext(
            model=model,
            device=device,
            config=config or {},
            extra={
                'stage': stage.value,
                'source': 'training_loop',
            }
        )
    except Exception as e:
        logger.warning(f"Failed to create strategy context: {e}")
        return None


def create_strategy_metrics_for_loop() -> Optional['StrategyMetrics']:
    """为训练循环创建策略指标跟踪器
    
    Returns:
        策略指标跟踪器或 None
    """
    try:
        return StrategyMetrics()
    except Exception as e:
        logger.warning(f"Failed to create strategy metrics: {e}")
        return None


def get_memory_info() -> Dict[str, Any]:
    """获取内存信息
    
    Returns:
        内存信息字典
    """
    memory_info = {
    }
    
    try:
        memory_info['available_memory_mb'] = get_available_memory()
    except Exception as e:
        logger.warning(f"Failed to get available memory: {e}")
    
    if torch.cuda.is_available():
        try:
            memory_info['cuda_allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
            memory_info['cuda_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
        except Exception:
            pass
    
    return memory_info


def clear_training_memory() -> bool:
    """清理训练内存
    
    Returns:
        是否成功
    """
    try:
        # 使用硬件层清理
        clear_memory()
        
        # PyTorch CUDA 清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        logger.warning(f"Failed to clear memory: {e}")
        return False


# ==================== 导出 ====================

__all__ = [
    # 主要类
    'TrainingLoop',
    'TrainingLoopConfig',
    'TrainingMetrics',
    'TrainingStage',
    
    # 工厂函数
    'create_training_loop',
    
    # 便捷函数
    'get_training_loop_info',
    'create_strategy_context_for_loop',
    'create_strategy_metrics_for_loop',
    'get_memory_info',
    'clear_training_memory',
]


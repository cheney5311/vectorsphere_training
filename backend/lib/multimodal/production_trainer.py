# -*- coding: utf-8 -*-
"""
生产级多模态训练器

实现四阶段训练流程：
1. 模态预训练（Modality Pretraining）
2. 跨模态对齐（Cross-Modal Alignment）
3. 指令微调（Instruction Tuning）
4. 对齐与安全（Alignment & Safety）
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .production_multimodal_config import (
    ProductionMultiModalConfig,
    TrainingStage,
    ModalityType,
    FourStageTrainingConfig
)
from .production_encoders import (
    BaseModalityEncoder,
    ModalityEncoderFactory,
    UnifiedProjection
)
from .production_alignment import CrossModalAligner, AlignmentLoss
from .production_fusion import MultiModalFuser

logger = logging.getLogger(__name__)


# ==================== 训练状态 ====================

@dataclass
class TrainingState:
    """训练状态"""
    stage: TrainingStage = TrainingStage.MODALITY_PRETRAIN
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    
    # 各阶段完成状态
    stage_completed: Dict[str, bool] = field(default_factory=dict)
    stage_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # 检查点
    checkpoint_path: Optional[str] = None
    
    def is_stage_completed(self, stage: TrainingStage) -> bool:
        return self.stage_completed.get(stage.value, False)
    
    def mark_stage_completed(self, stage: TrainingStage, metrics: Dict[str, float]):
        self.stage_completed[stage.value] = True
        self.stage_metrics[stage.value] = metrics


# ==================== 多模态模型 ====================

class ProductionMultiModalModel(nn.Module):
    """生产级多模态模型
    
    整合编码器、对齐器、融合器的完整模型
    """
    
    def __init__(self, config: ProductionMultiModalConfig):
        super().__init__()
        self.config = config
        
        # 模态编码器
        self.encoders = nn.ModuleDict()
        self._init_encoders()
        
        # 统一投影
        modality_dims = {
            m.value: getattr(config.encoders, m.value).hidden_size
            for m in config.modalities
            if hasattr(config.encoders, m.value)
        }
        self.projection = UnifiedProjection(
            modality_dims,
            config.encoders.unified_dim,
            config.encoders.projection_dropout
        )
        
        # 跨模态对齐器
        self.aligner = CrossModalAligner(
            config.alignment,
            config.encoders.unified_dim
        )
        
        # 多模态融合器
        unified_dims = {m: config.encoders.unified_dim for m in modality_dims}
        self.fuser = MultiModalFuser(config.fusion, unified_dims)
        
        # 输出头（根据任务配置）
        self.lm_head = None  # 语言模型头
        self.cls_head = None  # 分类头
        
        # 当前训练阶段
        self.current_stage = TrainingStage.MODALITY_PRETRAIN
    
    def _init_encoders(self):
        """初始化各模态编码器"""
        for modality in self.config.modalities:
            encoder_config = getattr(self.config.encoders, modality.value, None)
            if encoder_config is not None:
                encoder = ModalityEncoderFactory.create_encoder(modality, encoder_config)
                self.encoders[modality.value] = encoder
                logger.info(f"Initialized {modality.value} encoder")
    
    def set_training_stage(self, stage: TrainingStage):
        """设置当前训练阶段，调整冻结策略"""
        self.current_stage = stage
        
        if stage == TrainingStage.MODALITY_PRETRAIN:
            # 阶段一：所有编码器可训练
            for encoder in self.encoders.values():
                encoder.unfreeze()
            self.aligner.eval()
            self.fuser.eval()
            
        elif stage == TrainingStage.CROSS_MODAL_ALIGN:
            # 阶段二：根据配置冻结部分编码器
            align_config = self.config.training.cross_modal_align
            for name, encoder in self.encoders.items():
                if name == "text" and align_config.freeze_text_encoder:
                    encoder.freeze()
                elif name == "image" and align_config.freeze_image_encoder:
                    encoder.freeze()
                else:
                    encoder.unfreeze()
            self.aligner.train()
            self.fuser.eval()
            
        elif stage == TrainingStage.INSTRUCTION_TUNING:
            # 阶段三：冻结大部分，只训练适配层
            for encoder in self.encoders.values():
                encoder.freeze()
            self.aligner.eval()
            self.fuser.train()
            
        elif stage == TrainingStage.ALIGNMENT_SAFETY:
            # 阶段四：冻结大部分，微调
            for encoder in self.encoders.values():
                encoder.freeze()
            self.aligner.eval()
            self.fuser.eval()
    
    def forward(self,
                inputs: Dict[str, Dict[str, Tensor]],
                labels: Optional[Tensor] = None,
                return_dict: bool = True) -> Dict[str, Any]:
        """前向传播
        
        Args:
            inputs: 各模态输入 {modality: {input_key: tensor}}
            labels: 标签（可选）
            return_dict: 是否返回字典
            
        Returns:
            输出字典，包含loss、features等
        """
        # 1. 编码各模态
        modality_features = {}
        for modality, encoder in self.encoders.items():
            if modality in inputs:
                features = encoder(inputs[modality])
                modality_features[modality] = features
        
        # 2. 统一投影
        projected_features = self.projection(modality_features)
        
        # 3. 跨模态对齐
        aligned_features, align_loss, align_metrics = self.aligner(
            projected_features,
            compute_loss=(self.current_stage == TrainingStage.CROSS_MODAL_ALIGN)
        )
        
        # 4. 多模态融合
        fused_features = self.fuser(aligned_features)
        
        # 5. 计算损失（根据阶段）
        outputs = {
            'modality_features': modality_features,
            'projected_features': projected_features,
            'aligned_features': aligned_features,
            'fused_features': fused_features,
            'align_loss': align_loss,
            'align_metrics': align_metrics
        }
        
        if labels is not None:
            if self.cls_head is not None:
                logits = self.cls_head(fused_features)
                loss = F.cross_entropy(logits, labels)
                outputs['logits'] = logits
                outputs['loss'] = loss
            elif self.lm_head is not None:
                logits = self.lm_head(fused_features)
                outputs['logits'] = logits
        
        return outputs


# ==================== 四阶段训练器 ====================

class ProductionMultiModalTrainer:
    """生产级多模态训练器
    
    实现四阶段训练流程
    """
    
    def __init__(self,
                 model: ProductionMultiModalModel,
                 config: ProductionMultiModalConfig,
                 train_dataloader: Optional[DataLoader] = None,
                 eval_dataloader: Optional[DataLoader] = None,
                 callbacks: Optional[List[Callable]] = None):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.callbacks = callbacks or []
        
        # 训练状态
        self.state = TrainingState()
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 优化器（各阶段可能不同）
        self.optimizer = None
        self.scheduler = None
        
        # 混合精度
        self.scaler = None
        if config.distributed.fp16 or config.distributed.bf16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # 梯度累积
        self.gradient_accumulation_steps = 1
        
        # 日志
        self.logger = logging.getLogger(__name__)
    
    def train(self):
        """执行完整的四阶段训练"""
        stages = [
            TrainingStage.MODALITY_PRETRAIN,
            TrainingStage.CROSS_MODAL_ALIGN,
            TrainingStage.INSTRUCTION_TUNING,
            TrainingStage.ALIGNMENT_SAFETY
        ]
        
        for stage in stages:
            stage_config = self._get_stage_config(stage)
            
            if not stage_config.enabled:
                self.logger.info(f"Skipping stage: {stage.value}")
                continue
            
            if self.state.is_stage_completed(stage):
                self.logger.info(f"Stage already completed: {stage.value}")
                continue
            
            self.logger.info(f"Starting stage: {stage.value}")
            
            # 设置训练阶段
            self.state.stage = stage
            self.model.set_training_stage(stage)
            
            # 配置优化器
            self._setup_optimizer(stage_config)
            
            # 训练该阶段
            metrics = self._train_stage(stage, stage_config)
            
            # 标记阶段完成
            self.state.mark_stage_completed(stage, metrics)
            
            # 保存检查点
            if self.config.training.stage_save_checkpoint:
                self._save_checkpoint(f"checkpoint_{stage.value}")
            
            self.logger.info(f"Completed stage: {stage.value}, metrics: {metrics}")
        
        return self.state
    
    def _get_stage_config(self, stage: TrainingStage) -> Any:
        """获取阶段配置"""
        stage_configs = {
            TrainingStage.MODALITY_PRETRAIN: self.config.training.modality_pretrain,
            TrainingStage.CROSS_MODAL_ALIGN: self.config.training.cross_modal_align,
            TrainingStage.INSTRUCTION_TUNING: self.config.training.instruction_tuning,
            TrainingStage.ALIGNMENT_SAFETY: self.config.training.alignment_safety
        }
        return stage_configs.get(stage)
    
    def _setup_optimizer(self, stage_config: Any):
        """配置优化器"""
        # 获取可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # AdamW优化器
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=stage_config.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        total_steps = len(self.train_dataloader) * stage_config.epochs if self.train_dataloader else 1000
        warmup_steps = int(total_steps * stage_config.warmup_ratio)
        
        self.scheduler = self._get_scheduler(total_steps, warmup_steps)
    
    def _get_scheduler(self, total_steps: int, warmup_steps: int):
        """获取学习率调度器"""
        from torch.optim.lr_scheduler import LambdaLR
        
        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def _train_stage(self, stage: TrainingStage, stage_config: Any) -> Dict[str, float]:
        """训练单个阶段"""
        self.model.train()
        
        total_loss = 0.0
        num_steps = 0
        
        for epoch in range(stage_config.epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            if self.train_dataloader is None:
                # 模拟训练
                self.logger.warning("No train dataloader, using mock training")
                for step in range(100):
                    loss = self._mock_train_step()
                    epoch_loss += loss
                    epoch_steps += 1
            else:
                for step, batch in enumerate(self.train_dataloader):
                    loss = self._train_step(batch, stage)
                    epoch_loss += loss
                    epoch_steps += 1
                    
                    if step % self.config.logging_steps == 0:
                        self.logger.info(
                            f"Stage {stage.value}, Epoch {epoch}, Step {step}, "
                            f"Loss: {loss:.4f}"
                        )
            
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            total_loss += avg_epoch_loss
            num_steps += 1
            
            # 评估
            if self.eval_dataloader is not None:
                eval_metrics = self._evaluate()
                self.logger.info(f"Epoch {epoch} eval metrics: {eval_metrics}")
            
            # 回调
            for callback in self.callbacks:
                callback(stage, epoch, avg_epoch_loss)
        
        return {
            'loss': total_loss / max(num_steps, 1),
            'epochs': stage_config.epochs
        }
    
    def _train_step(self, batch: Dict[str, Any], stage: TrainingStage) -> float:
        """单步训练"""
        # 移动到设备
        inputs = self._prepare_batch(batch)
        labels = batch.get('labels')
        if labels is not None:
            labels = labels.to(self.device)
        
        # 混合精度训练
        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs, labels=labels)
                loss = self._compute_loss(outputs, stage)
            
            self.scaler.scale(loss).backward()
            
            if self.state.global_step % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            outputs = self.model(inputs, labels=labels)
            loss = self._compute_loss(outputs, stage)
            loss.backward()
            
            if self.state.global_step % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        
        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()
        
        self.state.global_step += 1
        
        return loss.item()
    
    def _mock_train_step(self) -> float:
        """模拟训练步骤"""
        # 创建模拟输入
        batch_size = 4
        seq_len = 32
        image_size = 224
        
        inputs = {}
        
        if 'text' in self.model.encoders:
            inputs['text'] = {
                'input_ids': torch.randint(0, 30000, (batch_size, seq_len)).to(self.device),
                'attention_mask': torch.ones(batch_size, seq_len).to(self.device)
            }
        
        if 'image' in self.model.encoders:
            inputs['image'] = {
                'pixel_values': torch.randn(batch_size, 3, image_size, image_size).to(self.device)
            }
        
        # 前向传播
        outputs = self.model(inputs)
        
        # 模拟损失
        loss = outputs.get('align_loss')
        if loss is None:
            fused = outputs['fused_features']
            loss = fused.mean()
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item() if isinstance(loss, torch.Tensor) else loss
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Dict[str, Tensor]]:
        """准备批次数据"""
        inputs = {}
        
        for modality in self.config.modalities:
            modality_name = modality.value
            modality_inputs = {}
            
            # 处理不同模态的输入
            if modality_name == 'text':
                if 'input_ids' in batch:
                    modality_inputs['input_ids'] = batch['input_ids'].to(self.device)
                    modality_inputs['attention_mask'] = batch.get('attention_mask', 
                        torch.ones_like(batch['input_ids'])).to(self.device)
            
            elif modality_name == 'image':
                if 'pixel_values' in batch:
                    modality_inputs['pixel_values'] = batch['pixel_values'].to(self.device)
            
            elif modality_name == 'audio':
                if 'input_values' in batch:
                    modality_inputs['input_values'] = batch['input_values'].to(self.device)
            
            elif modality_name == 'time_series':
                if 'time_series' in batch:
                    modality_inputs['time_series'] = batch['time_series'].to(self.device)
            
            if modality_inputs:
                inputs[modality_name] = modality_inputs
        
        return inputs
    
    def _compute_loss(self, outputs: Dict[str, Any], stage: TrainingStage) -> Tensor:
        """计算损失"""
        total_loss = torch.tensor(0.0, device=self.device)
        
        # 阶段一：模态预训练损失
        if stage == TrainingStage.MODALITY_PRETRAIN:
            if 'loss' in outputs:
                total_loss = total_loss + outputs['loss']
            else:
                # 对比学习损失
                features = outputs['projected_features']
                if len(features) >= 2:
                    modalities = list(features.keys())
                    feat_a = features[modalities[0]]
                    feat_b = features[modalities[1]]
                    
                    # InfoNCE
                    feat_a = F.normalize(feat_a, dim=-1)
                    feat_b = F.normalize(feat_b, dim=-1)
                    logits = torch.matmul(feat_a, feat_b.T) / 0.07
                    labels = torch.arange(feat_a.shape[0], device=self.device)
                    total_loss = F.cross_entropy(logits, labels)
        
        # 阶段二：对齐损失
        elif stage == TrainingStage.CROSS_MODAL_ALIGN:
            if outputs.get('align_loss') is not None:
                total_loss = outputs['align_loss']
        
        # 阶段三/四：任务损失
        elif stage in [TrainingStage.INSTRUCTION_TUNING, TrainingStage.ALIGNMENT_SAFETY]:
            if 'loss' in outputs:
                total_loss = outputs['loss']
        
        return total_loss
    
    def _evaluate(self) -> Dict[str, float]:
        """评估"""
        self.model.eval()
        total_loss = 0.0
        num_steps = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                inputs = self._prepare_batch(batch)
                labels = batch.get('labels')
                if labels is not None:
                    labels = labels.to(self.device)
                
                outputs = self.model(inputs, labels=labels)
                loss = self._compute_loss(outputs, self.state.stage)
                
                total_loss += loss.item()
                num_steps += 1
        
        self.model.train()
        
        return {
            'eval_loss': total_loss / max(num_steps, 1)
        }
    
    def _save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pt")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'training_state': self.state,
            'config': self.config
        }, checkpoint_path)
        
        self.state.checkpoint_path = checkpoint_path
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.state = checkpoint['training_state']
        
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")


# ==================== 便捷函数 ====================

def create_multimodal_trainer(
    config: ProductionMultiModalConfig,
    train_dataloader: Optional[DataLoader] = None,
    eval_dataloader: Optional[DataLoader] = None
) -> ProductionMultiModalTrainer:
    """创建多模态训练器
    
    Args:
        config: 配置
        train_dataloader: 训练数据加载器
        eval_dataloader: 评估数据加载器
        
    Returns:
        训练器实例
    """
    model = ProductionMultiModalModel(config)
    trainer = ProductionMultiModalTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )
    return trainer


def run_four_stage_training(
    config: ProductionMultiModalConfig,
    train_dataloader: Optional[DataLoader] = None,
    eval_dataloader: Optional[DataLoader] = None,
    callbacks: Optional[List[Callable]] = None
) -> TrainingState:
    """运行四阶段训练
    
    Args:
        config: 配置
        train_dataloader: 训练数据加载器
        eval_dataloader: 评估数据加载器
        callbacks: 回调函数列表
        
    Returns:
        训练状态
    """
    trainer = create_multimodal_trainer(config, train_dataloader, eval_dataloader)
    trainer.callbacks = callbacks or []
    
    return trainer.train()


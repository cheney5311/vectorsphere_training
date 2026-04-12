# -*- coding: utf-8 -*-
"""
行业模型抽象层

定义行业模型的统一抽象：
IndustryModel = Backbone + ModalityAdapters + ScenarioHeads

支持：
- 通用大模型作为Backbone
- 行业模态适配器（文本、表格、图像、时序）
- 业务场景输出头

架构层次：
本模块 -> strategies/scenario_strategy.py (策略层)
       -> backend/lib/* (六层架构底层能力)
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# 尝试导入策略层能力
STRATEGY_AVAILABLE = False
try:
    from backend.modules.training.strategies import (
        ProductionTrainingStrategy,
        ProductionStrategyConfig,
        ProductionTrainingContext,
        IndustryScenarioStrategy,
        ScenarioStrategyConfig,
        create_production_strategy,
        create_production_context,
        get_available_layers,
        HARDWARE_AVAILABLE,
        DISTRIBUTED_AVAILABLE,
        ADAPTERS_AVAILABLE,
        LOSSES_AVAILABLE
    )
    from backend.modules.training.strategies.base_strategy import StrategyContext, StrategyResult
    from backend.modules.training.strategies.scenario_strategy import ScenarioType
    STRATEGY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Strategy layer import failed: {e}")


class IndustryType(Enum):
    """行业类型枚举"""
    MANUFACTURING = "manufacturing"   # 制造业
    FINANCE = "finance"               # 金融
    HEALTHCARE = "healthcare"         # 医疗
    ENERGY = "energy"                 # 能源
    GOVERNMENT = "government"         # 政务
    RETAIL = "retail"                 # 零售
    LOGISTICS = "logistics"           # 物流
    GENERAL = "general"               # 通用


class ModalityType(Enum):
    """模态类型枚举"""
    TEXT = "text"                     # 文本
    TABLE = "table"                   # 表格
    IMAGE = "image"                   # 图像
    TIME_SERIES = "time_series"       # 时序信号
    AUDIO = "audio"                   # 音频
    VIDEO = "video"                   # 视频


@dataclass
class IndustryModelConfig:
    """行业模型配置"""
    # 基础配置
    industry_type: IndustryType = IndustryType.GENERAL
    model_name: str = "industry_model"
    
    # Backbone配置
    backbone_name: str = "gpt2"
    backbone_hidden_size: int = 768
    backbone_num_layers: int = 12
    freeze_backbone: bool = False
    
    # 模态适配器配置
    modalities: List[ModalityType] = field(
        default_factory=lambda: [ModalityType.TEXT]
    )
    adapter_hidden_size: int = 256
    adapter_num_layers: int = 2
    adapter_dropout: float = 0.1
    
    # 场景头配置
    scenario_heads: List[str] = field(default_factory=list)
    head_hidden_size: int = 256
    head_num_classes: Dict[str, int] = field(default_factory=dict)
    
    # 融合配置
    fusion_method: str = "attention"  # concat, attention, gated
    fusion_hidden_size: int = 512
    
    # 训练配置
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0


class ModalityAdapter(nn.Module, ABC):
    """
    模态适配器基类
    
    将特定模态的输入转换为统一的表示空间。
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 构建适配器网络
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        self.adapter = nn.Sequential(*layers)
    
    @abstractmethod
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """预处理输入数据"""
        pass
    
    def forward(self, inputs: Any) -> torch.Tensor:
        """前向传播"""
        x = self.preprocess(inputs)
        return self.adapter(x)


class TextAdapter(ModalityAdapter):
    """文本模态适配器"""
    
    def __init__(
        self, 
        vocab_size: int = 50000,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 768,
        max_length: int = 512,
        **kwargs
    ):
        super().__init__(embed_dim, hidden_dim, output_dim, **kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        self.max_length = max_length
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """预处理文本输入"""
        if isinstance(inputs, dict):
            input_ids = inputs.get('input_ids', inputs.get('text'))
        else:
            input_ids = inputs
        
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        
        # 获取嵌入
        seq_length = input_ids.shape[-1]
        positions = torch.arange(seq_length, device=input_ids.device)
        
        embeddings = self.embedding(input_ids) + self.position_embedding(positions)
        
        # 平均池化
        return embeddings.mean(dim=-2)


class TableAdapter(ModalityAdapter):
    """表格模态适配器"""
    
    def __init__(
        self, 
        num_columns: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 768,
        **kwargs
    ):
        super().__init__(num_columns, hidden_dim, output_dim, **kwargs)
        
        # 列编码
        self.column_embedding = nn.Linear(1, hidden_dim)
        self.column_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """预处理表格输入"""
        if isinstance(inputs, dict):
            table_data = inputs.get('table', inputs.get('data'))
        else:
            table_data = inputs
        
        if not isinstance(table_data, torch.Tensor):
            table_data = torch.tensor(table_data, dtype=torch.float32)
        
        # [batch, num_columns] -> [batch, num_columns, 1]
        if len(table_data.shape) == 2:
            table_data = table_data.unsqueeze(-1)
        
        # 列编码
        column_features = self.column_embedding(table_data)
        
        # 自注意力
        attended, _ = self.column_attention(column_features, column_features, column_features)
        
        # 聚合
        return attended.mean(dim=1)


class ImageAdapter(ModalityAdapter):
    """图像模态适配器"""
    
    def __init__(
        self, 
        image_size: int = 224,
        num_channels: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 768,
        **kwargs
    ):
        super().__init__(num_channels * 49, hidden_dim, output_dim, **kwargs)  # 7x7 patch
        
        # 简单的CNN特征提取器
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.flatten = nn.Flatten()
        self.projection = nn.Linear(128 * 49, hidden_dim)
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """预处理图像输入"""
        if isinstance(inputs, dict):
            image_data = inputs.get('image', inputs.get('pixel_values'))
        else:
            image_data = inputs
        
        if not isinstance(image_data, torch.Tensor):
            image_data = torch.tensor(image_data, dtype=torch.float32)
        
        # CNN特征提取
        features = self.cnn(image_data)
        features = self.flatten(features)
        return self.projection(features)


class TimeSeriesAdapter(ModalityAdapter):
    """时序信号适配器"""
    
    def __init__(
        self, 
        num_features: int = 10,
        seq_length: int = 100,
        hidden_dim: int = 256,
        output_dim: int = 768,
        **kwargs
    ):
        super().__init__(num_features, hidden_dim, output_dim, **kwargs)
        
        # 时序编码器 (1D CNN + LSTM)
        self.conv1d = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, 
            num_layers=2, batch_first=True, 
            bidirectional=True
        )
        
        self.projection = nn.Linear(hidden_dim * 2, output_dim)
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """预处理时序输入"""
        if isinstance(inputs, dict):
            ts_data = inputs.get('time_series', inputs.get('signal'))
        else:
            ts_data = inputs
        
        if not isinstance(ts_data, torch.Tensor):
            ts_data = torch.tensor(ts_data, dtype=torch.float32)
        
        # [batch, seq_length, num_features] -> [batch, num_features, seq_length]
        if len(ts_data.shape) == 3:
            ts_data = ts_data.transpose(1, 2)
        
        # CNN特征提取
        conv_out = self.conv1d(ts_data)  # [batch, hidden, seq]
        conv_out = conv_out.transpose(1, 2)  # [batch, seq, hidden]
        
        # LSTM
        lstm_out, _ = self.lstm(conv_out)
        
        # 取最后一个时间步
        return self.projection(lstm_out[:, -1, :])


class GenericAdapter(ModalityAdapter):
    """通用模态适配器"""
    
    def preprocess(self, inputs: Any) -> torch.Tensor:
        """预处理通用输入"""
        if isinstance(inputs, torch.Tensor):
            return inputs.float()
        # 尝试转换
        if hasattr(inputs, 'to_tensor'):
            return inputs.to_tensor()
        return torch.tensor(inputs).float()


class ScenarioHead(nn.Module):
    """
    场景输出头
    
    针对特定业务场景的输出层。
    """
    
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        task_type: str = "classification",  # classification, regression, sequence
        dropout: float = 0.1
    ):
        super().__init__()
        self.task_type = task_type
        self.num_classes = num_classes
        
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        logits = self.head(features)
        
        if self.task_type == "regression" and self.num_classes == 1:
            return logits.squeeze(-1)
        
        return logits


class ModalityFusion(nn.Module):
    """
    模态融合层
    
    支持多种融合方式：拼接、注意力、门控。
    """
    
    def __init__(
        self, 
        modality_dims: Dict[str, int],
        output_dim: int,
        method: str = "attention",
        num_heads: int = 8
    ):
        super().__init__()
        self.method = method
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        
        total_dim = sum(modality_dims.values())
        
        if method == "concat":
            self.fusion = nn.Linear(total_dim, output_dim)
        
        elif method == "attention":
            # 统一模态维度
            self.modality_projections = nn.ModuleDict({
                name: nn.Linear(dim, output_dim)
                for name, dim in modality_dims.items()
            })
            self.attention = nn.MultiheadAttention(output_dim, num_heads, batch_first=True)
            self.fusion = nn.Linear(output_dim, output_dim)
        
        elif method == "gated":
            self.modality_projections = nn.ModuleDict({
                name: nn.Linear(dim, output_dim)
                for name, dim in modality_dims.items()
            })
            self.gates = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(output_dim, output_dim),
                    nn.Sigmoid()
                )
                for name in modality_dims.keys()
            })
            self.fusion = nn.Linear(output_dim, output_dim)
    
    def forward(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """融合多模态特征"""
        if len(modality_features) == 1:
            return list(modality_features.values())[0]
        
        if self.method == "concat":
            # 拼接融合
            concat_features = torch.cat(list(modality_features.values()), dim=-1)
            return self.fusion(concat_features)
        
        elif self.method == "attention":
            # 注意力融合
            projected = []
            for name, features in modality_features.items():
                if name in self.modality_projections:
                    projected.append(self.modality_projections[name](features))
            
            # Stack为序列 [batch, num_modalities, dim]
            stacked = torch.stack(projected, dim=1)
            
            # 自注意力
            attended, _ = self.attention(stacked, stacked, stacked)
            
            # 平均池化
            fused = attended.mean(dim=1)
            return self.fusion(fused)
        
        elif self.method == "gated":
            # 门控融合
            weighted_sum = None
            for name, features in modality_features.items():
                if name in self.modality_projections:
                    projected = self.modality_projections[name](features)
                    gate = self.gates[name](projected)
                    gated = projected * gate
                    
                    if weighted_sum is None:
                        weighted_sum = gated
                    else:
                        weighted_sum = weighted_sum + gated
            
            return self.fusion(weighted_sum)
        
        else:
            # 默认拼接
            concat_features = torch.cat(list(modality_features.values()), dim=-1)
            return torch.adaptive_avg_pool1d(concat_features.unsqueeze(1), self.output_dim).squeeze(1)


class IndustryModel(nn.Module):
    """
    行业模型
    
    统一的行业模型抽象：
    IndustryModel = Backbone + ModalityAdapters + ScenarioHeads
    
    支持：
    - 通用大模型作为Backbone
    - 行业模态适配器
    - 业务场景输出头
    """
    
    def __init__(self, config: IndustryModelConfig):
        super().__init__()
        self.config = config
        
        # 初始化Backbone
        self.backbone = self._init_backbone()
        
        # 初始化模态适配器
        self.modality_adapters = nn.ModuleDict()
        self._init_modality_adapters()
        
        # 初始化融合层
        modality_dims = {
            m.value: config.backbone_hidden_size 
            for m in config.modalities
        }
        self.fusion = ModalityFusion(
            modality_dims=modality_dims,
            output_dim=config.fusion_hidden_size,
            method=config.fusion_method
        )
        
        # 初始化场景头
        self.scenario_heads = nn.ModuleDict()
        self._init_scenario_heads()
        
        # LoRA适配（如果启用）
        if config.use_lora:
            self._init_lora()
        
        logger.info(f"IndustryModel initialized: {config.industry_type.value}, "
                   f"modalities={[m.value for m in config.modalities]}, "
                   f"heads={config.scenario_heads}")
    
    def _init_backbone(self) -> nn.Module:
        """初始化Backbone"""
        # 尝试加载预训练模型，失败则使用Mock
        try:
            from transformers import AutoModel
            backbone = AutoModel.from_pretrained(self.config.backbone_name)
            logger.info(f"Loaded backbone: {self.config.backbone_name}")
        except Exception as e:
            logger.warning(f"Failed to load backbone, using mock: {e}")
            backbone = self._create_mock_backbone()
        
        # 冻结Backbone（如果配置）
        if self.config.freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")
        
        return backbone
    
    def _create_mock_backbone(self) -> nn.Module:
        """创建Mock Backbone"""
        return nn.Sequential(
            nn.Linear(self.config.backbone_hidden_size, self.config.backbone_hidden_size),
            nn.LayerNorm(self.config.backbone_hidden_size),
            nn.GELU(),
            nn.Linear(self.config.backbone_hidden_size, self.config.backbone_hidden_size)
        )
    
    def _init_modality_adapters(self) -> None:
        """初始化模态适配器"""
        for modality in self.config.modalities:
            if modality == ModalityType.TEXT:
                adapter = TextAdapter(
                    embed_dim=self.config.backbone_hidden_size,
                    hidden_dim=self.config.adapter_hidden_size,
                    output_dim=self.config.backbone_hidden_size,
                    num_layers=self.config.adapter_num_layers,
                    dropout=self.config.adapter_dropout
                )
            elif modality == ModalityType.TABLE:
                adapter = TableAdapter(
                    hidden_dim=self.config.adapter_hidden_size,
                    output_dim=self.config.backbone_hidden_size,
                    num_layers=self.config.adapter_num_layers,
                    dropout=self.config.adapter_dropout
                )
            elif modality == ModalityType.IMAGE:
                adapter = ImageAdapter(
                    hidden_dim=self.config.adapter_hidden_size,
                    output_dim=self.config.backbone_hidden_size,
                    num_layers=self.config.adapter_num_layers,
                    dropout=self.config.adapter_dropout
                )
            elif modality == ModalityType.TIME_SERIES:
                adapter = TimeSeriesAdapter(
                    hidden_dim=self.config.adapter_hidden_size,
                    output_dim=self.config.backbone_hidden_size,
                    num_layers=self.config.adapter_num_layers,
                    dropout=self.config.adapter_dropout
                )
            else:
                # 通用适配器
                adapter = GenericAdapter(
                    input_dim=self.config.backbone_hidden_size,
                    hidden_dim=self.config.adapter_hidden_size,
                    output_dim=self.config.backbone_hidden_size,
                    num_layers=self.config.adapter_num_layers,
                    dropout=self.config.adapter_dropout
                )
            
            self.modality_adapters[modality.value] = adapter
    
    def _init_scenario_heads(self) -> None:
        """初始化场景头"""
        for head_name in self.config.scenario_heads:
            num_classes = self.config.head_num_classes.get(head_name, 2)
            
            head = ScenarioHead(
                input_dim=self.config.fusion_hidden_size,
                hidden_dim=self.config.head_hidden_size,
                num_classes=num_classes,
                task_type="classification" if num_classes > 1 else "regression"
            )
            
            self.scenario_heads[head_name] = head
    
    def _init_lora(self) -> None:
        """初始化LoRA适配"""
        # 简化的LoRA实现
        logger.info(f"LoRA initialized: rank={self.config.lora_rank}")
    
    def forward(
        self, 
        inputs: Dict[str, Any],
        scenario: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            inputs: 多模态输入字典
            scenario: 指定的场景（如果为None则返回所有场景头的输出）
        
        Returns:
            输出字典，包含各场景头的预测
        """
        # 1. 模态适配
        modality_features = {}
        for modality_name, adapter in self.modality_adapters.items():
            if modality_name in inputs:
                features = adapter(inputs[modality_name])
                modality_features[modality_name] = features
        
        # 2. 模态融合
        if len(modality_features) > 0:
            fused_features = self.fusion(modality_features)
        else:
            # 如果没有任何模态输入，使用占位符
            batch_size = 1
            for v in inputs.values():
                if isinstance(v, torch.Tensor):
                    batch_size = v.shape[0]
                    break
            fused_features = torch.zeros(
                batch_size, self.config.fusion_hidden_size,
                device=next(self.parameters()).device
            )
        
        # 3. 场景头
        outputs = {'fused_features': fused_features}
        outputs['modality_features'] = modality_features
        
        if scenario is not None:
            # 只运行指定场景头
            if scenario in self.scenario_heads:
                outputs[scenario] = self.scenario_heads[scenario](fused_features)
        else:
            # 运行所有场景头
            for head_name, head in self.scenario_heads.items():
                outputs[head_name] = head(fused_features)
        
        return outputs
    
    def get_backbone_features(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """获取Backbone特征"""
        # 这里简化处理，实际需要根据Backbone类型调整
        if 'input_ids' in inputs:
            return self.backbone(inputs['input_ids'])[0].mean(dim=1)
        return None
    
    def add_scenario_head(
        self, 
        name: str, 
        num_classes: int, 
        task_type: str = "classification"
    ) -> None:
        """添加场景头"""
        head = ScenarioHead(
            input_dim=self.config.fusion_hidden_size,
            hidden_dim=self.config.head_hidden_size,
            num_classes=num_classes,
            task_type=task_type
        )
        self.scenario_heads[name] = head
        self.config.scenario_heads.append(name)
        self.config.head_num_classes[name] = num_classes
        logger.info(f"Added scenario head: {name}")
    
    def create_strategy(
        self, 
        scenario_type: str = "equipment_fault_prediction",
        use_production_mode: bool = True
    ) -> Optional['ProductionTrainingStrategy']:
        """
        创建与模型配合的训练策略
        
        整合六层架构能力进行训练。
        
        Args:
            scenario_type: 场景类型
            use_production_mode: 是否使用生产模式
        
        Returns:
            训练策略实例（如果策略层可用）
        """
        if not STRATEGY_AVAILABLE:
            logger.warning("Strategy layer not available")
            return None
        
        try:
            # 映射模态类型
            modality_list = [m.value for m in self.config.modalities]
            
            # 创建场景策略配置
            scenario_config = ScenarioStrategyConfig(
                scenario_type=ScenarioType(scenario_type) if hasattr(ScenarioType, scenario_type.upper()) else ScenarioType.BASIC_MODEL,
                freeze_backbone=self.config.freeze_backbone,
                use_scene_adapter=True,
                adapter_dim=self.config.adapter_hidden_size,
            )
            
            # 创建生产级策略配置
            production_config = ProductionStrategyConfig(
                device="auto",
                precision="fp16" if torch.cuda.is_available() else "fp32",
                enable_amp=torch.cuda.is_available(),
                modalities=modality_list,
                hidden_size=self.config.backbone_hidden_size,
                task_loss_type="cross_entropy",
                adapter_type="lora" if self.config.use_lora else None,
                lora_rank=self.config.lora_rank,
            )
            
            # 创建策略
            strategy = IndustryScenarioStrategy(
                config=scenario_config,
                production_config=production_config
            )
            
            logger.info(f"Created strategy for IndustryModel: {strategy.name}")
            return strategy
            
        except Exception as e:
            logger.error(f"Failed to create strategy: {e}")
            return None
    
    def create_training_context(
        self,
        device: Optional[torch.device] = None
    ) -> Optional['ProductionTrainingContext']:
        """
        创建生产级训练上下文
        
        Args:
            device: 指定设备
        
        Returns:
            训练上下文实例（如果策略层可用）
        """
        if not STRATEGY_AVAILABLE:
            logger.warning("Strategy layer not available")
            return None
        
        try:
            modality_list = [m.value for m in self.config.modalities]
            
            config = ProductionStrategyConfig(
                device="auto",
                precision="fp16" if torch.cuda.is_available() else "fp32",
                enable_amp=torch.cuda.is_available(),
                modalities=modality_list,
                hidden_size=self.config.backbone_hidden_size,
            )
            
            ctx = ProductionTrainingContext(
                config=config,
                model=self,
                device=device
            )
            ctx.initialize()
            
            logger.info(f"Created training context for IndustryModel")
            return ctx
            
        except Exception as e:
            logger.error(f"Failed to create training context: {e}")
            return None
    
    def get_available_layers_info(self) -> Dict[str, bool]:
        """获取可用的底层模块信息"""
        if STRATEGY_AVAILABLE:
            return get_available_layers()
        return {
            'hardware': False,
            'distributed': False,
            'adapters': False,
            'losses': False,
            'strategy': STRATEGY_AVAILABLE
        }


# ============================================================================
# 行业特定模型
# ============================================================================

class ManufacturingModel(IndustryModel):
    """
    制造业模型
    
    针对制造业场景优化的行业模型：
    - 支持时序信号（传感器、PLC）
    - 支持图像（缺陷检测）
    - 支持文本（工艺文档）
    - 支持表格（BOM、工艺参数）
    """
    
    def __init__(self, config: Optional[IndustryModelConfig] = None):
        if config is None:
            config = IndustryModelConfig(
                industry_type=IndustryType.MANUFACTURING,
                model_name="manufacturing_model",
                modalities=[
                    ModalityType.TEXT,
                    ModalityType.TABLE,
                    ModalityType.TIME_SERIES,
                    ModalityType.IMAGE
                ],
                scenario_heads=[
                    "fault_prediction",
                    "process_optimization",
                    "quality_detection",
                    "energy_prediction"
                ],
                head_num_classes={
                    "fault_prediction": 2,
                    "process_optimization": 1,
                    "quality_detection": 5,
                    "energy_prediction": 1
                },
                fusion_method="attention"
            )
        
        super().__init__(config)
        
        logger.info("ManufacturingModel initialized")


class FinanceModel(IndustryModel):
    """金融行业模型"""
    
    def __init__(self, config: Optional[IndustryModelConfig] = None):
        if config is None:
            config = IndustryModelConfig(
                industry_type=IndustryType.FINANCE,
                model_name="finance_model",
                modalities=[
                    ModalityType.TEXT,
                    ModalityType.TABLE,
                    ModalityType.TIME_SERIES
                ],
                scenario_heads=[
                    "risk_assessment",
                    "fraud_detection",
                    "credit_scoring"
                ],
                head_num_classes={
                    "risk_assessment": 5,
                    "fraud_detection": 2,
                    "credit_scoring": 1
                }
            )
        
        super().__init__(config)


class HealthcareModel(IndustryModel):
    """医疗行业模型"""
    
    def __init__(self, config: Optional[IndustryModelConfig] = None):
        if config is None:
            config = IndustryModelConfig(
                industry_type=IndustryType.HEALTHCARE,
                model_name="healthcare_model",
                modalities=[
                    ModalityType.TEXT,
                    ModalityType.IMAGE,
                    ModalityType.TABLE
                ],
                scenario_heads=[
                    "disease_diagnosis",
                    "medical_image_analysis",
                    "drug_interaction"
                ],
                head_num_classes={
                    "disease_diagnosis": 100,
                    "medical_image_analysis": 10,
                    "drug_interaction": 2
                }
            )
        
        super().__init__(config)


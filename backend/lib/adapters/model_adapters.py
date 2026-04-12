# -*- coding: utf-8 -*-
"""
模型适配器 - 生产级实现

提供各种参数高效微调方法。

生产级特性：
- 多种适配器类型（LoRA、Prefix、Prompt、Adapter Layers、BitFit）
- 指标收集和监控
- 适配器融合和组合
- 参数统计和分析
- 梯度检查点支持
- 适配器保存和加载
- 动态秩调整
"""

import logging
import math
import time
import threading
import json
import os
from typing import Optional, Dict, Any, List, Union, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class AdapterType(Enum):
    """适配器类型"""
    BACKBONE = "backbone"      # 骨干网络适配
    TASK_HEAD = "task_head"    # 任务头适配
    LORA = "lora"              # LoRA
    PREFIX = "prefix"          # Prefix Tuning
    PROMPT = "prompt"          # Prompt Tuning
    ADAPTER = "adapter"        # Adapter Layers
    BITFIT = "bitfit"          # BitFit (bias-only)
    IA3 = "ia3"                # IA3 (Infused Adapter by Inhibiting and Amplifying)
    COMPACTER = "compacter"    # Compacter


class AdapterStatus(Enum):
    """适配器状态"""
    READY = "ready"
    ADAPTING = "adapting"
    MERGED = "merged"
    ERROR = "error"


class MergeStrategy(Enum):
    """合并策略"""
    LINEAR = "linear"          # 线性插值
    TASK_ARITHMETIC = "task_arithmetic"  # 任务算术
    TIES = "ties"              # TIES合并
    DARE = "dare"              # DARE合并


class InitStrategy(Enum):
    """初始化策略"""
    KAIMING = "kaiming"
    XAVIER = "xavier"
    NORMAL = "normal"
    ZEROS = "zeros"
    IDENTITY = "identity"


# ==================== 数据类 ====================

@dataclass
class AdapterMetrics:
    """适配器指标"""
    total_adaptations: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    error_count: int = 0
    last_adaptation_time: Optional[datetime] = None
    parameter_stats: Dict[str, Any] = field(default_factory=dict)
    gradient_stats: Dict[str, float] = field(default_factory=dict)
    
    def record_adaptation(self, time_taken: float) -> None:
        """记录适配"""
        self.total_adaptations += 1
        self.total_time += time_taken
        self.avg_time = self.total_time / self.total_adaptations
        self.last_adaptation_time = datetime.now()
    
    def record_error(self) -> None:
        """记录错误"""
        self.error_count += 1
    
    def update_parameter_stats(self, adapter: nn.Module) -> None:
        """更新参数统计"""
        total_params = 0
        trainable_params = 0
        
        for param in adapter.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        self.parameter_stats = {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
            "trainable_ratio": trainable_params / max(total_params, 1)
        }
    
    def update_gradient_stats(self, adapter: nn.Module) -> None:
        """更新梯度统计"""
        grad_norms = []
        for param in adapter.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        if grad_norms:
            self.gradient_stats = {
                "mean_grad_norm": sum(grad_norms) / len(grad_norms),
                "max_grad_norm": max(grad_norms),
                "min_grad_norm": min(grad_norms)
            }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_adaptations": self.total_adaptations,
            "total_time": self.total_time,
            "avg_time": self.avg_time,
            "error_count": self.error_count,
            "last_adaptation_time": self.last_adaptation_time.isoformat() if self.last_adaptation_time else None,
            "parameter_stats": self.parameter_stats,
            "gradient_stats": self.gradient_stats
        }


@dataclass
class AdapterConfig:
    """适配器配置"""
    adapter_type: AdapterType = AdapterType.LORA
    hidden_size: int = 768
    
    # LoRA配置
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None
    lora_use_dora: bool = False  # DoRA: Weight-Decomposed Low-Rank Adaptation
    lora_use_rslora: bool = False  # RS-LoRA: Rank-Stabilized LoRA
    
    # Prefix配置
    prefix_length: int = 20
    prefix_projection: bool = False
    prefix_hidden_size: int = 512
    
    # Prompt配置
    num_virtual_tokens: int = 20
    prompt_init_text: Optional[str] = None
    
    # Adapter Layers配置
    adapter_bottleneck: int = 64
    adapter_non_linearity: str = "relu"
    adapter_residual: bool = True
    adapter_scaling: float = 1.0
    
    # BitFit配置
    bitfit_bias_only: bool = True
    
    # IA3配置
    ia3_target_modules: List[str] = None
    
    # 生产级配置 - 初始化
    init_strategy: InitStrategy = InitStrategy.KAIMING
    init_std: float = 0.02
    
    # 生产级配置 - 指标
    enable_metrics: bool = True
    
    # 生产级配置 - 梯度检查点
    use_gradient_checkpointing: bool = False
    
    # 生产级配置 - 合并
    merge_strategy: MergeStrategy = MergeStrategy.LINEAR
    merge_weight: float = 1.0
    
    # 生产级配置 - 正则化
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    
    # 生产级配置 - 动态秩
    dynamic_rank: bool = False
    rank_pattern: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ['q_proj', 'v_proj']
        if self.ia3_target_modules is None:
            self.ia3_target_modules = ['k_proj', 'v_proj', 'down_proj']
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # 转换枚举为字符串
        result["adapter_type"] = self.adapter_type.value
        result["init_strategy"] = self.init_strategy.value
        result["merge_strategy"] = self.merge_strategy.value
        return result


class ModelAdapter(nn.Module, ABC):
    """
    模型适配器基类 - 生产级实现
    
    所有适配器的抽象基类，提供：
    - 指标收集和监控
    - 参数统计和分析
    - 适配器保存和加载
    - 合并功能
    """
    
    def __init__(self, config: AdapterConfig):
        super().__init__()
        self.config = config
        self.adapter_type = config.adapter_type
        
        # 状态
        self._status = AdapterStatus.READY
        
        # 指标
        self._metrics = AdapterMetrics() if config.enable_metrics else None
        self._metrics_lock = threading.Lock()
        
        # 已适配的模块名称
        self._adapted_modules: Set[str] = set()
        
        # 原始权重缓存（用于合并/重置）
        self._original_weights: Dict[str, torch.Tensor] = {}
    
    @abstractmethod
    def adapt(self, model: nn.Module) -> nn.Module:
        """
        适配模型
        
        Args:
            model: 原始模型
            
        Returns:
            适配后的模型
        """
        pass
    
    @contextmanager
    def _timed_adaptation(self):
        """计时上下文管理器"""
        self._status = AdapterStatus.ADAPTING
        start_time = time.time()
        try:
            yield
        except Exception as e:
            self._status = AdapterStatus.ERROR
            if self._metrics is not None:
                with self._metrics_lock:
                    self._metrics.record_error()
            raise
        finally:
            elapsed = time.time() - start_time
            if self._metrics is not None:
                with self._metrics_lock:
                    self._metrics.record_adaptation(elapsed)
            self._status = AdapterStatus.READY
    
    def get_trainable_params(self, model: nn.Module) -> List[nn.Parameter]:
        """获取可训练参数"""
        return [p for p in model.parameters() if p.requires_grad]
    
    def get_trainable_param_names(self, model: nn.Module) -> List[str]:
        """获取可训练参数名称"""
        return [n for n, p in model.named_parameters() if p.requires_grad]
    
    def freeze_base_model(self, model: nn.Module) -> None:
        """冻结基础模型参数"""
        for param in model.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self, model: nn.Module) -> None:
        """解冻基础模型参数"""
        for param in model.parameters():
            param.requires_grad = True
    
    def get_parameter_count(self) -> Dict[str, int]:
        """获取参数数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_ratio": trainable / max(total, 1)
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用量（MB）"""
        param_memory = sum(
            p.numel() * p.element_size() for p in self.parameters()
        ) / (1024 * 1024)
        
        buffer_memory = sum(
            b.numel() * b.element_size() for b in self.buffers()
        ) / (1024 * 1024)
        
        return {
            "parameters_mb": param_memory,
            "buffers_mb": buffer_memory,
            "total_mb": param_memory + buffer_memory
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标"""
        if self._metrics is None:
            return {"metrics_disabled": True}
        
        with self._metrics_lock:
            metrics = self._metrics.to_dict()
            metrics["parameter_count"] = self.get_parameter_count()
            metrics["memory_usage"] = self.get_memory_usage()
            return metrics
    
    def reset_metrics(self) -> None:
        """重置指标"""
        if self._metrics is not None:
            with self._metrics_lock:
                self._metrics = AdapterMetrics()
    
    def get_status(self) -> AdapterStatus:
        """获取状态"""
        return self._status
    
    def get_adapted_modules(self) -> Set[str]:
        """获取已适配的模块"""
        return self._adapted_modules.copy()
    
    def _cache_original_weight(self, name: str, weight: torch.Tensor) -> None:
        """缓存原始权重"""
        self._original_weights[name] = weight.clone().detach()
    
    def _get_original_weight(self, name: str) -> Optional[torch.Tensor]:
        """获取原始权重"""
        return self._original_weights.get(name)
    
    def save_adapter(self, path: str) -> None:
        """
        保存适配器
        
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        state = {
            "adapter_type": self.adapter_type.value,
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
            "adapted_modules": list(self._adapted_modules),
            "timestamp": datetime.now().isoformat()
        }
        
        torch.save(state, path)
        logger.info(f"Adapter saved to {path}")
    
    def load_adapter(self, path: str, strict: bool = True) -> None:
        """
        加载适配器
        
        Args:
            path: 加载路径
            strict: 是否严格匹配
        """
        state = torch.load(path, map_location="cpu")
        
        self.load_state_dict(state["state_dict"], strict=strict)
        self._adapted_modules = set(state.get("adapted_modules", []))
        
        logger.info(f"Adapter loaded from {path}")
    
    def merge_into_model(self, model: nn.Module) -> nn.Module:
        """
        将适配器合并到模型中（子类可重写）
        
        Args:
            model: 目标模型
            
        Returns:
            合并后的模型
        """
        logger.warning(f"{self.__class__.__name__} does not support merging")
        return model
    
    def unmerge_from_model(self, model: nn.Module) -> nn.Module:
        """
        从模型中移除合并的适配器（子类可重写）
        
        Args:
            model: 目标模型
            
        Returns:
            移除后的模型
        """
        logger.warning(f"{self.__class__.__name__} does not support unmerging")
        return model
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            "adapter_type": self.adapter_type.value,
            "status": self._status.value,
            "parameter_count": self.get_parameter_count(),
            "adapted_modules_count": len(self._adapted_modules)
        }
    
    def apply_gradient_clipping(self, max_norm: Optional[float] = None) -> float:
        """
        应用梯度裁剪
        
        Args:
            max_norm: 最大梯度范数
            
        Returns:
            实际梯度范数
        """
        max_norm = max_norm or self.config.max_grad_norm
        params = [p for p in self.parameters() if p.grad is not None]
        if params:
            return torch.nn.utils.clip_grad_norm_(params, max_norm).item()
        return 0.0
    
    def update_gradient_stats(self) -> None:
        """更新梯度统计"""
        if self._metrics is not None:
            with self._metrics_lock:
                self._metrics.update_gradient_stats(self)


class BackboneAdapter(ModelAdapter):
    """
    骨干网络适配器
    
    冻结骨干网络，只训练特定层。
    """
    
    def __init__(self, config: AdapterConfig, trainable_layers: List[str] = None):
        super().__init__(config)
        self.trainable_layers = trainable_layers or []
    
    def adapt(self, model: nn.Module) -> nn.Module:
        """适配骨干网络"""
        # 先冻结所有参数
        self.freeze_base_model(model)
        
        # 解冻指定层
        for name, param in model.named_parameters():
            for layer_name in self.trainable_layers:
                if layer_name in name:
                    param.requires_grad = True
                    break
        
        logger.info(f"BackboneAdapter: unfroze {self.trainable_layers}")
        return model


class TaskHeadAdapter(ModelAdapter):
    """
    任务头适配器
    
    添加任务特定的输出头。
    """
    
    def __init__(
        self, 
        config: AdapterConfig, 
        num_classes: int = 2,
        task_type: str = "classification"
    ):
        super().__init__(config)
        self.num_classes = num_classes
        self.task_type = task_type
        
        # 任务头
        if task_type == "classification":
            self.head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.hidden_size, num_classes)
            )
        elif task_type == "regression":
            self.head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, 1)
            )
        else:
            self.head = nn.Linear(config.hidden_size, num_classes)
    
    def adapt(self, model: nn.Module) -> nn.Module:
        """添加任务头"""
        # 冻结基础模型
        self.freeze_base_model(model)
        
        # 将任务头注册到模型
        model.task_head = self.head
        
        logger.info(f"TaskHeadAdapter: added {self.task_type} head with {self.num_classes} outputs")
        return model
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 使用[CLS] token或mean pooling
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, 0]  # [CLS] token
        return self.head(hidden_states)


class LoRALayer(nn.Module):
    """
    LoRA层 - 生产级实现
    
    Low-Rank Adaptation of Large Language Models
    
    支持:
    - 标准LoRA
    - DoRA (Weight-Decomposed Low-Rank Adaptation)
    - RS-LoRA (Rank-Stabilized LoRA)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        use_dora: bool = False,
        use_rslora: bool = False,
        init_strategy: InitStrategy = InitStrategy.KAIMING
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.use_dora = use_dora
        self.use_rslora = use_rslora
        
        # RS-LoRA使用不同的缩放
        if use_rslora:
            self.scaling = alpha / math.sqrt(rank)
        else:
            self.scaling = alpha / rank
        
        # LoRA矩阵
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # DoRA: 方向和幅度分解
        if use_dora:
            self.magnitude = nn.Parameter(torch.ones(out_features))
        else:
            self.magnitude = None
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        self._initialize_weights(init_strategy)
        
        # 是否已合并
        self._merged = False
        self._original_weight: Optional[torch.Tensor] = None
    
    def _initialize_weights(self, init_strategy: InitStrategy) -> None:
        """初始化权重"""
        if init_strategy == InitStrategy.KAIMING:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        elif init_strategy == InitStrategy.XAVIER:
            nn.init.xavier_uniform_(self.lora_A)
        elif init_strategy == InitStrategy.NORMAL:
            nn.init.normal_(self.lora_A, std=0.02)
        elif init_strategy == InitStrategy.ZEROS:
            nn.init.zeros_(self.lora_A)
        else:  # IDENTITY
            nn.init.eye_(self.lora_A[:min(self.in_features, self.rank), :])
        
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LoRA前向传播"""
        lora_output = self.dropout(x @ self.lora_A @ self.lora_B) * self.scaling
        return lora_output
    
    def forward_with_dora(self, x: torch.Tensor, base_weight: torch.Tensor) -> torch.Tensor:
        """
        DoRA前向传播
        
        Args:
            x: 输入
            base_weight: 基础权重
            
        Returns:
            DoRA输出
        """
        if not self.use_dora:
            return self.forward(x)
        
        # 计算合并后的权重
        lora_weight = (self.lora_A @ self.lora_B) * self.scaling
        merged_weight = base_weight + lora_weight
        
        # 计算方向（归一化）
        weight_norm = merged_weight.norm(dim=0, keepdim=True)
        direction = merged_weight / (weight_norm + 1e-8)
        
        # 应用幅度
        output_weight = self.magnitude.unsqueeze(0) * direction
        
        return x @ output_weight
    
    def get_delta_weight(self) -> torch.Tensor:
        """获取LoRA增量权重"""
        return (self.lora_A @ self.lora_B) * self.scaling
    
    def merge_weight(self, base_weight: torch.Tensor) -> torch.Tensor:
        """
        合并权重
        
        Args:
            base_weight: 基础权重
            
        Returns:
            合并后的权重
        """
        delta = self.get_delta_weight()
        return base_weight + delta
    
    def get_effective_rank(self) -> float:
        """计算有效秩"""
        with torch.no_grad():
            weight = self.lora_A @ self.lora_B
            _, s, _ = torch.svd(weight)
            s_normalized = s / s.sum()
            entropy = -(s_normalized * (s_normalized + 1e-10).log()).sum()
            return entropy.exp().item()
    
    def get_layer_analysis(self) -> Dict[str, Any]:
        """获取层分析"""
        analysis = {
            "rank": self.rank,
            "alpha": self.alpha,
            "scaling": self.scaling,
            "use_dora": self.use_dora,
            "use_rslora": self.use_rslora,
            "effective_rank": self.get_effective_rank()
        }
        
        with torch.no_grad():
            delta = self.get_delta_weight()
            analysis["delta_norm"] = float(delta.norm())
            analysis["delta_mean"] = float(delta.mean())
            analysis["delta_std"] = float(delta.std())
            
            # A和B矩阵分析
            analysis["lora_A_norm"] = float(self.lora_A.norm())
            analysis["lora_B_norm"] = float(self.lora_B.norm())
        
        return analysis


class LoRAAdapter(ModelAdapter):
    """
    LoRA适配器 - 生产级实现
    
    为指定模块添加LoRA层。
    
    生产级特性：
    - DoRA/RS-LoRA支持
    - 动态秩调整
    - 权重合并/拆分
    - 层级分析
    """
    
    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.lora_layers: Dict[str, LoRALayer] = {}
        self._original_forwards: Dict[str, Callable] = {}
    
    def adapt(self, model: nn.Module) -> nn.Module:
        """为模型添加LoRA"""
        with self._timed_adaptation():
            # 冻结基础模型
            self.freeze_base_model(model)
            
            # 为目标模块添加LoRA
            for name, module in model.named_modules():
                if any(target in name for target in self.config.lora_target_modules):
                    if isinstance(module, nn.Linear):
                        rank = self._get_rank_for_module(name)
                        self._add_lora_to_linear(model, name, module, rank)
            
            # 更新指标
            if self._metrics is not None:
                with self._metrics_lock:
                    self._metrics.update_parameter_stats(self)
            
            logger.info(f"LoRAAdapter: added LoRA to {len(self.lora_layers)} modules")
            return model
    
    def _get_rank_for_module(self, name: str) -> int:
        """获取模块的秩"""
        if self.config.dynamic_rank and self.config.rank_pattern:
            for pattern, rank in self.config.rank_pattern.items():
                if pattern in name:
                    return rank
        return self.config.lora_rank
    
    def _add_lora_to_linear(
        self, 
        model: nn.Module, 
        name: str, 
        module: nn.Linear,
        rank: int
    ) -> None:
        """为Linear层添加LoRA"""
        # 缓存原始权重
        self._cache_original_weight(name, module.weight)
        
        # 创建LoRA层
        lora = LoRALayer(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=rank,
            alpha=self.config.lora_alpha,
            dropout=self.config.lora_dropout,
            use_dora=self.config.lora_use_dora,
            use_rslora=self.config.lora_use_rslora,
            init_strategy=self.config.init_strategy
        )
        
        # 保存原始forward
        original_forward = module.forward
        self._original_forwards[name] = original_forward
        
        # 创建新的forward
        if self.config.lora_use_dora:
            def new_forward(x, lora=lora, module=module):
                return module.weight @ x.transpose(-1, -2) + module.bias.unsqueeze(-1) if module.bias is not None else lora.forward_with_dora(x, module.weight.T)
        else:
            def new_forward(x, lora=lora, original=original_forward):
                return original(x) + lora(x)
        
        # 替换forward
        module.forward = new_forward
        
        # 注册LoRA层
        self.lora_layers[name] = lora
        self._adapted_modules.add(name)
        
        # 将LoRA参数注册到模型
        for param_name, param in lora.named_parameters():
            model.register_parameter(f"{name}_lora_{param_name}", param)
    
    def merge_into_model(self, model: nn.Module) -> nn.Module:
        """
        将LoRA合并到模型权重中
        
        Args:
            model: 目标模型
            
        Returns:
            合并后的模型
        """
        for name, lora in self.lora_layers.items():
            # 找到对应的模块
            module = dict(model.named_modules()).get(name)
            if module is not None and isinstance(module, nn.Linear):
                with torch.no_grad():
                    delta = lora.get_delta_weight()
                    module.weight.data += delta.T
                lora._merged = True
        
        self._status = AdapterStatus.MERGED
        logger.info(f"LoRA merged into {len(self.lora_layers)} modules")
        return model
    
    def unmerge_from_model(self, model: nn.Module) -> nn.Module:
        """
        从模型中移除合并的LoRA
        
        Args:
            model: 目标模型
            
        Returns:
            移除后的模型
        """
        for name, lora in self.lora_layers.items():
            if lora._merged:
                module = dict(model.named_modules()).get(name)
                if module is not None and isinstance(module, nn.Linear):
                    with torch.no_grad():
                        delta = lora.get_delta_weight()
                        module.weight.data -= delta.T
                    lora._merged = False
        
        self._status = AdapterStatus.READY
        logger.info(f"LoRA unmerged from {len(self.lora_layers)} modules")
        return model
    
    def get_lora_analysis(self) -> Dict[str, Any]:
        """获取LoRA分析"""
        analysis = {
            "num_lora_layers": len(self.lora_layers),
            "target_modules": self.config.lora_target_modules,
            "total_rank": self.config.lora_rank,
            "alpha": self.config.lora_alpha,
            "use_dora": self.config.lora_use_dora,
            "use_rslora": self.config.lora_use_rslora,
            "layers": {}
        }
        
        for name, lora in self.lora_layers.items():
            analysis["layers"][name] = lora.get_layer_analysis()
        
        # 汇总统计
        if self.lora_layers:
            delta_norms = [l.get_layer_analysis()["delta_norm"] for l in self.lora_layers.values()]
            analysis["avg_delta_norm"] = sum(delta_norms) / len(delta_norms)
            analysis["max_delta_norm"] = max(delta_norms)
            
            effective_ranks = [l.get_effective_rank() for l in self.lora_layers.values()]
            analysis["avg_effective_rank"] = sum(effective_ranks) / len(effective_ranks)
        
        return analysis
    
    def adjust_rank(self, new_rank: int, module_name: Optional[str] = None) -> None:
        """
        动态调整秩
        
        Args:
            new_rank: 新秩
            module_name: 指定模块名称（None表示所有）
        """
        if not self.config.dynamic_rank:
            logger.warning("Dynamic rank is not enabled")
            return
        
        for name, lora in self.lora_layers.items():
            if module_name is None or module_name == name:
                if new_rank != lora.rank:
                    # 创建新的LoRA层
                    new_lora = LoRALayer(
                        in_features=lora.in_features,
                        out_features=lora.out_features,
                        rank=new_rank,
                        alpha=lora.alpha,
                        dropout=lora.dropout.p,
                        use_dora=lora.use_dora,
                        use_rslora=lora.use_rslora
                    )
                    
                    # 复制部分权重
                    min_rank = min(lora.rank, new_rank)
                    with torch.no_grad():
                        new_lora.lora_A.data[:, :min_rank] = lora.lora_A.data[:, :min_rank]
                        new_lora.lora_B.data[:min_rank, :] = lora.lora_B.data[:min_rank, :]
                    
                    self.lora_layers[name] = new_lora
                    logger.info(f"Adjusted rank for {name}: {lora.rank} -> {new_rank}")


class PrefixAdapter(ModelAdapter):
    """
    Prefix Tuning适配器
    
    在注意力机制的key和value前添加前缀。
    """
    
    def __init__(self, config: AdapterConfig, num_layers: int = 12):
        super().__init__(config)
        self.num_layers = num_layers
        
        # 前缀嵌入
        self.prefix_tokens = nn.Embedding(
            config.prefix_length, 
            config.hidden_size * 2 * num_layers  # key和value
        )
        
        # 可选的投影层
        if config.prefix_projection:
            self.prefix_projection = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, config.hidden_size * 2 * num_layers)
            )
        else:
            self.prefix_projection = None
    
    def get_prefix(self, batch_size: int) -> torch.Tensor:
        """获取前缀"""
        device = self.prefix_tokens.weight.device
        prefix_ids = torch.arange(self.config.prefix_length, device=device)
        prefix_ids = prefix_ids.unsqueeze(0).expand(batch_size, -1)
        
        prefix = self.prefix_tokens(prefix_ids)  # [B, prefix_len, hidden*2*layers]
        
        if self.prefix_projection is not None:
            prefix = self.prefix_projection(prefix[:, :, :self.config.hidden_size])
        
        # 重塑为 [layers, 2, B, prefix_len, hidden]
        prefix = prefix.view(
            batch_size,
            self.config.prefix_length,
            self.num_layers,
            2,
            self.config.hidden_size
        )
        prefix = prefix.permute(2, 3, 0, 1, 4)  # [layers, 2, B, prefix_len, hidden]
        
        return prefix
    
    def adapt(self, model: nn.Module) -> nn.Module:
        """添加Prefix Tuning"""
        self.freeze_base_model(model)
        
        # 将prefix模块添加到模型
        model.prefix_adapter = self
        
        logger.info(f"PrefixAdapter: added prefix with length {self.config.prefix_length}")
        return model


class PromptAdapter(ModelAdapter):
    """
    Prompt Tuning适配器
    
    在输入序列前添加可学习的软提示。
    """
    
    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        
        # 虚拟token嵌入
        self.prompt_embeddings = nn.Embedding(
            config.num_virtual_tokens,
            config.hidden_size
        )
        
        # 初始化
        nn.init.normal_(self.prompt_embeddings.weight, std=0.02)
    
    def get_prompt_embeddings(self, batch_size: int) -> torch.Tensor:
        """获取提示嵌入"""
        device = self.prompt_embeddings.weight.device
        prompt_ids = torch.arange(self.config.num_virtual_tokens, device=device)
        prompt_ids = prompt_ids.unsqueeze(0).expand(batch_size, -1)
        
        return self.prompt_embeddings(prompt_ids)  # [B, num_tokens, hidden]
    
    def adapt(self, model: nn.Module) -> nn.Module:
        """添加Prompt Tuning"""
        self.freeze_base_model(model)
        
        # 将prompt模块添加到模型
        model.prompt_adapter = self
        
        logger.info(f"PromptAdapter: added {self.config.num_virtual_tokens} virtual tokens")
        return model
    
    def prepend_prompt(
        self, 
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """在输入前添加提示"""
        batch_size = input_embeds.shape[0]
        
        # 获取提示嵌入
        prompt = self.get_prompt_embeddings(batch_size)
        
        # 拼接
        embeds = torch.cat([prompt, input_embeds], dim=1)
        
        # 更新attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size, self.config.num_virtual_tokens,
                device=attention_mask.device
            )
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        
        return embeds, attention_mask


class AdapterLayerModule(nn.Module):
    """
    Adapter层模块
    
    瓶颈结构的适配器层。
    """
    
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        non_linearity: str = "relu",
        residual: bool = True,
        scaling: float = 1.0
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.residual = residual
        self.scaling = scaling
        
        # 下投影
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        
        # 非线性激活
        if non_linearity == "relu":
            self.activation = nn.ReLU()
        elif non_linearity == "gelu":
            self.activation = nn.GELU()
        elif non_linearity == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # 上投影
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 初始化
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = x
        
        # 下投影 -> 激活 -> 上投影
        output = self.down_proj(x)
        output = self.activation(output)
        output = self.up_proj(output)
        
        # 缩放
        output = output * self.scaling
        
        # 残差连接
        if self.residual:
            output = output + residual
        
        return self.layer_norm(output)


class AdapterLayersAdapter(ModelAdapter):
    """
    Adapter Layers适配器 - 生产级实现
    
    在Transformer层之间插入适配器瓶颈层。
    
    生产级特性：
    - 可配置的瓶颈大小
    - 多种激活函数
    - 残差连接
    - 层级缩放
    """
    
    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.adapter_layers: Dict[str, AdapterLayerModule] = {}
    
    def adapt(self, model: nn.Module) -> nn.Module:
        """添加Adapter Layers"""
        with self._timed_adaptation():
            # 冻结基础模型
            self.freeze_base_model(model)
            
            # 查找并适配Transformer层
            for name, module in model.named_modules():
                # 在MLP/FFN输出后添加适配器
                if self._should_add_adapter(name, module):
                    self._add_adapter_layer(model, name, module)
            
            # 更新指标
            if self._metrics is not None:
                with self._metrics_lock:
                    self._metrics.update_parameter_stats(self)
            
            logger.info(f"AdapterLayersAdapter: added {len(self.adapter_layers)} adapter layers")
            return model
    
    def _should_add_adapter(self, name: str, module: nn.Module) -> bool:
        """判断是否应该添加适配器"""
        # 在MLP/FFN的最后一个Linear层后添加
        target_patterns = ['mlp', 'ffn', 'intermediate', 'output']
        return (
            isinstance(module, nn.Linear) and 
            any(p in name.lower() for p in target_patterns)
        )
    
    def _add_adapter_layer(
        self, 
        model: nn.Module, 
        name: str, 
        module: nn.Module
    ) -> None:
        """添加适配器层"""
        adapter = AdapterLayerModule(
            hidden_size=self.config.hidden_size,
            bottleneck_size=self.config.adapter_bottleneck,
            non_linearity=self.config.adapter_non_linearity,
            residual=self.config.adapter_residual,
            scaling=self.config.adapter_scaling
        )
        
        # 保存原始forward
        original_forward = module.forward
        
        # 创建新的forward
        def new_forward(x, adapter=adapter, original=original_forward):
            output = original(x)
            return adapter(output)
        
        # 替换forward
        module.forward = new_forward
        
        # 注册适配器
        self.adapter_layers[name] = adapter
        self._adapted_modules.add(name)
        
        # 注册参数
        for param_name, param in adapter.named_parameters():
            model.register_parameter(f"{name}_adapter_{param_name}", param)
    
    def get_adapter_analysis(self) -> Dict[str, Any]:
        """获取适配器分析"""
        analysis = {
            "num_adapter_layers": len(self.adapter_layers),
            "bottleneck_size": self.config.adapter_bottleneck,
            "non_linearity": self.config.adapter_non_linearity,
            "layers": {}
        }
        
        for name, adapter in self.adapter_layers.items():
            with torch.no_grad():
                analysis["layers"][name] = {
                    "down_proj_norm": float(adapter.down_proj.weight.norm()),
                    "up_proj_norm": float(adapter.up_proj.weight.norm())
                }
        
        return analysis


class BitFitAdapter(ModelAdapter):
    """
    BitFit适配器 - 生产级实现
    
    只训练偏置参数的高效微调方法。
    
    生产级特性：
    - 选择性偏置训练
    - 偏置分析
    - 偏置重置
    """
    
    def __init__(self, config: AdapterConfig, target_modules: Optional[List[str]] = None):
        super().__init__(config)
        self.target_modules = target_modules  # None表示所有偏置
        self._bias_params: Dict[str, nn.Parameter] = {}
    
    def adapt(self, model: nn.Module) -> nn.Module:
        """应用BitFit"""
        with self._timed_adaptation():
            # 先冻结所有参数
            self.freeze_base_model(model)
            
            # 解冻偏置参数
            for name, param in model.named_parameters():
                if self._should_train_bias(name):
                    param.requires_grad = True
                    self._bias_params[name] = param
                    self._adapted_modules.add(name)
            
            # 更新指标
            if self._metrics is not None:
                with self._metrics_lock:
                    self._metrics.update_parameter_stats(model)
            
            logger.info(f"BitFitAdapter: enabled training for {len(self._bias_params)} bias parameters")
            return model
    
    def _should_train_bias(self, name: str) -> bool:
        """判断是否应该训练该偏置"""
        if 'bias' not in name:
            return False
        
        if self.target_modules is None:
            return True
        
        return any(target in name for target in self.target_modules)
    
    def get_bias_analysis(self) -> Dict[str, Any]:
        """获取偏置分析"""
        analysis = {
            "num_bias_params": len(self._bias_params),
            "total_bias_elements": sum(p.numel() for p in self._bias_params.values()),
            "biases": {}
        }
        
        for name, param in self._bias_params.items():
            with torch.no_grad():
                analysis["biases"][name] = {
                    "size": param.numel(),
                    "mean": float(param.mean()),
                    "std": float(param.std()),
                    "norm": float(param.norm())
                }
        
        return analysis
    
    def reset_biases(self) -> None:
        """重置偏置为零"""
        for param in self._bias_params.values():
            with torch.no_grad():
                param.zero_()
        logger.info("BitFitAdapter: reset all biases to zero")


class IA3Layer(nn.Module):
    """
    IA3层
    
    Infused Adapter by Inhibiting and Amplifying Inner Activations
    """
    
    def __init__(self, size: int):
        super().__init__()
        # 可学习的缩放向量
        self.learned_vector = nn.Parameter(torch.ones(size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """IA3前向传播"""
        return x * self.learned_vector


class IA3Adapter(ModelAdapter):
    """
    IA3适配器 - 生产级实现
    
    通过抑制和放大内部激活进行适配。
    
    生产级特性：
    - 针对K、V和FFN的缩放
    - 缩放向量分析
    - 动态缩放调整
    """
    
    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        self.ia3_layers: Dict[str, IA3Layer] = {}
    
    def adapt(self, model: nn.Module) -> nn.Module:
        """应用IA3"""
        with self._timed_adaptation():
            # 冻结基础模型
            self.freeze_base_model(model)
            
            # 为目标模块添加IA3
            for name, module in model.named_modules():
                if any(target in name for target in self.config.ia3_target_modules):
                    if isinstance(module, nn.Linear):
                        self._add_ia3_to_linear(model, name, module)
            
            # 更新指标
            if self._metrics is not None:
                with self._metrics_lock:
                    self._metrics.update_parameter_stats(self)
            
            logger.info(f"IA3Adapter: added IA3 to {len(self.ia3_layers)} modules")
            return model
    
    def _add_ia3_to_linear(
        self, 
        model: nn.Module, 
        name: str, 
        module: nn.Linear
    ) -> None:
        """为Linear层添加IA3"""
        # 创建IA3层
        ia3 = IA3Layer(module.out_features)
        
        # 保存原始forward
        original_forward = module.forward
        
        # 创建新的forward
        def new_forward(x, ia3=ia3, original=original_forward):
            return ia3(original(x))
        
        # 替换forward
        module.forward = new_forward
        
        # 注册IA3层
        self.ia3_layers[name] = ia3
        self._adapted_modules.add(name)
        
        # 注册参数
        model.register_parameter(f"{name}_ia3_vector", ia3.learned_vector)
    
    def get_ia3_analysis(self) -> Dict[str, Any]:
        """获取IA3分析"""
        analysis = {
            "num_ia3_layers": len(self.ia3_layers),
            "target_modules": self.config.ia3_target_modules,
            "layers": {}
        }
        
        for name, ia3 in self.ia3_layers.items():
            vec = ia3.learned_vector
            with torch.no_grad():
                analysis["layers"][name] = {
                    "size": vec.numel(),
                    "mean": float(vec.mean()),
                    "std": float(vec.std()),
                    "min": float(vec.min()),
                    "max": float(vec.max()),
                    "near_zero_ratio": float((vec.abs() < 0.1).float().mean())
                }
        
        return analysis


class CompacterLayer(nn.Module):
    """
    Compacter层
    
    使用低秩和参数共享的紧凑适配器。
    """
    
    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int = 64,
        rank: int = 4,
        num_adapters: int = 1
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.rank = rank
        
        # 共享的低秩矩阵
        self.shared_A = nn.Parameter(torch.randn(hidden_size, rank) / math.sqrt(rank))
        self.shared_B = nn.Parameter(torch.randn(rank, bottleneck_size) / math.sqrt(rank))
        
        # 特定的缩放因子
        self.scale_down = nn.Parameter(torch.ones(bottleneck_size))
        self.scale_up = nn.Parameter(torch.ones(hidden_size))
        
        # 非线性
        self.activation = nn.GELU()
        
        # 上投影的低秩矩阵
        self.up_A = nn.Parameter(torch.randn(bottleneck_size, rank) / math.sqrt(rank))
        self.up_B = nn.Parameter(torch.randn(rank, hidden_size) / math.sqrt(rank))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compacter前向传播"""
        # 下投影：使用低秩分解
        down = x @ self.shared_A @ self.shared_B
        down = down * self.scale_down
        
        # 非线性
        down = self.activation(down)
        
        # 上投影：使用低秩分解
        up = down @ self.up_A @ self.up_B
        up = up * self.scale_up
        
        return x + up


class CompacterAdapter(ModelAdapter):
    """
    Compacter适配器 - 生产级实现
    
    紧凑的参数高效适配器。
    
    生产级特性：
    - 低秩分解
    - 参数共享
    - 紧凑表示
    """
    
    def __init__(self, config: AdapterConfig, rank: int = 4):
        super().__init__(config)
        self.rank = rank
        self.compacter_layers: Dict[str, CompacterLayer] = {}
    
    def adapt(self, model: nn.Module) -> nn.Module:
        """应用Compacter"""
        with self._timed_adaptation():
            # 冻结基础模型
            self.freeze_base_model(model)
            
            # 查找并适配目标模块
            for name, module in model.named_modules():
                if self._should_add_compacter(name, module):
                    self._add_compacter_layer(model, name, module)
            
            # 更新指标
            if self._metrics is not None:
                with self._metrics_lock:
                    self._metrics.update_parameter_stats(self)
            
            logger.info(f"CompacterAdapter: added {len(self.compacter_layers)} compacter layers")
            return model
    
    def _should_add_compacter(self, name: str, module: nn.Module) -> bool:
        """判断是否应该添加Compacter"""
        target_patterns = ['mlp', 'ffn', 'attention']
        return (
            isinstance(module, nn.Linear) and 
            any(p in name.lower() for p in target_patterns)
        )
    
    def _add_compacter_layer(
        self, 
        model: nn.Module, 
        name: str, 
        module: nn.Module
    ) -> None:
        """添加Compacter层"""
        compacter = CompacterLayer(
            hidden_size=self.config.hidden_size,
            bottleneck_size=self.config.adapter_bottleneck,
            rank=self.rank
        )
        
        # 保存原始forward
        original_forward = module.forward
        
        # 创建新的forward
        def new_forward(x, compacter=compacter, original=original_forward):
            output = original(x)
            return compacter(output)
        
        # 替换forward
        module.forward = new_forward
        
        # 注册
        self.compacter_layers[name] = compacter
        self._adapted_modules.add(name)
        
        # 注册参数
        for param_name, param in compacter.named_parameters():
            model.register_parameter(f"{name}_compacter_{param_name}", param)
    
    def get_compacter_analysis(self) -> Dict[str, Any]:
        """获取Compacter分析"""
        analysis = {
            "num_compacter_layers": len(self.compacter_layers),
            "rank": self.rank,
            "bottleneck_size": self.config.adapter_bottleneck,
            "layers": {}
        }
        
        for name, compacter in self.compacter_layers.items():
            with torch.no_grad():
                analysis["layers"][name] = {
                    "shared_A_norm": float(compacter.shared_A.norm()),
                    "shared_B_norm": float(compacter.shared_B.norm()),
                    "scale_down_mean": float(compacter.scale_down.mean()),
                    "scale_up_mean": float(compacter.scale_up.mean())
                }
        
        return analysis


# ==================== 适配器融合 ====================

class AdapterFusion(nn.Module):
    """
    适配器融合模块
    
    组合多个适配器的输出。
    """
    
    def __init__(
        self, 
        adapters: List[ModelAdapter],
        fusion_method: str = "attention",
        hidden_size: int = 768
    ):
        super().__init__()
        
        self.adapters = nn.ModuleList(adapters)
        self.fusion_method = fusion_method
        self.num_adapters = len(adapters)
        
        if fusion_method == "attention":
            # 注意力融合
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.Tanh(),
                nn.Linear(hidden_size // 4, self.num_adapters)
            )
        elif fusion_method == "gated":
            # 门控融合
            self.gates = nn.Parameter(torch.ones(self.num_adapters) / self.num_adapters)
        else:
            # 简单平均
            pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """融合前向传播"""
        # 获取各适配器输出
        outputs = [adapter(x) for adapter in self.adapters]
        stacked = torch.stack(outputs, dim=-1)  # [B, L, H, num_adapters]
        
        if self.fusion_method == "attention":
            # 注意力权重
            weights = F.softmax(self.attention(x), dim=-1)  # [B, L, num_adapters]
            weights = weights.unsqueeze(2)  # [B, L, 1, num_adapters]
            fused = (stacked * weights).sum(dim=-1)
        elif self.fusion_method == "gated":
            # 门控权重
            weights = F.softmax(self.gates, dim=-1)
            fused = (stacked * weights).sum(dim=-1)
        else:
            # 简单平均
            fused = stacked.mean(dim=-1)
        
        return fused
    
    def get_fusion_weights(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取融合权重"""
        if self.fusion_method == "attention" and x is not None:
            return F.softmax(self.attention(x), dim=-1)
        elif self.fusion_method == "gated":
            return F.softmax(self.gates, dim=-1)
        else:
            return torch.ones(self.num_adapters) / self.num_adapters


class AdapterMerger:
    """
    适配器合并器
    
    支持多种合并策略。
    """
    
    @staticmethod
    def merge_linear(
        adapters: List[ModelAdapter],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        线性插值合并
        
        Args:
            adapters: 适配器列表
            weights: 权重列表
            
        Returns:
            合并后的参数
        """
        if len(adapters) != len(weights):
            raise ValueError("Number of adapters must match number of weights")
        
        # 归一化权重
        total = sum(weights)
        weights = [w / total for w in weights]
        
        merged = {}
        for i, adapter in enumerate(adapters):
            for name, param in adapter.named_parameters():
                if name not in merged:
                    merged[name] = torch.zeros_like(param)
                merged[name] += weights[i] * param
        
        return merged
    
    @staticmethod
    def merge_task_arithmetic(
        base_adapter: ModelAdapter,
        task_adapters: List[ModelAdapter],
        task_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        任务算术合并
        
        Args:
            base_adapter: 基础适配器
            task_adapters: 任务适配器列表
            task_weights: 任务权重
            
        Returns:
            合并后的参数
        """
        merged = {}
        
        # 获取基础参数
        for name, param in base_adapter.named_parameters():
            merged[name] = param.clone()
        
        # 添加任务向量
        for adapter, weight in zip(task_adapters, task_weights):
            for name, param in adapter.named_parameters():
                if name in merged:
                    # 任务向量 = 任务参数 - 基础参数
                    task_vector = param - merged[name]
                    merged[name] = merged[name] + weight * task_vector
        
        return merged
    
    @staticmethod
    def merge_ties(
        adapters: List[ModelAdapter],
        threshold: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        """
        TIES合并：修剪、选举、合并
        
        Args:
            adapters: 适配器列表
            threshold: 修剪阈值
            
        Returns:
            合并后的参数
        """
        merged = {}
        
        # 收集所有参数
        all_params = {}
        for adapter in adapters:
            for name, param in adapter.named_parameters():
                if name not in all_params:
                    all_params[name] = []
                all_params[name].append(param)
        
        # 对每个参数进行TIES合并
        for name, params in all_params.items():
            stacked = torch.stack(params, dim=0)
            
            # 1. 修剪：保留幅度最大的值
            mask = stacked.abs() > threshold * stacked.abs().max()
            trimmed = stacked * mask.float()
            
            # 2. 选举：确定符号
            sign_sum = trimmed.sign().sum(dim=0)
            elected_sign = sign_sum.sign()
            
            # 3. 合并：对符号一致的值取平均
            same_sign = (trimmed.sign() == elected_sign.unsqueeze(0)).float()
            merged[name] = (trimmed * same_sign).sum(dim=0) / (same_sign.sum(dim=0) + 1e-8)
        
        return merged


# ==================== 质量分析器 ====================

class AdapterQualityAnalyzer:
    """
    适配器质量分析器
    
    评估适配器的效果和质量。
    """
    
    def __init__(self):
        self._analysis_history: List[Dict[str, Any]] = []
        self._max_history = 100
    
    def analyze_adapter(self, adapter: ModelAdapter) -> Dict[str, Any]:
        """分析适配器质量"""
        analysis = {
            "adapter_type": adapter.adapter_type.value,
            "status": adapter.get_status().value,
            "parameter_count": adapter.get_parameter_count(),
            "memory_usage": adapter.get_memory_usage(),
            "timestamp": datetime.now().isoformat()
        }
        
        # 参数分析
        param_stats = self._analyze_parameters(adapter)
        analysis["parameter_stats"] = param_stats
        
        # 保存历史
        self._analysis_history.append(analysis)
        if len(self._analysis_history) > self._max_history:
            self._analysis_history = self._analysis_history[-self._max_history:]
        
        return analysis
    
    def _analyze_parameters(self, adapter: ModelAdapter) -> Dict[str, Any]:
        """分析参数"""
        stats = {}
        
        with torch.no_grad():
            norms = []
            means = []
            stds = []
            
            for name, param in adapter.named_parameters():
                norms.append(param.norm().item())
                means.append(param.mean().item())
                stds.append(param.std().item())
            
            if norms:
                stats["avg_norm"] = sum(norms) / len(norms)
                stats["max_norm"] = max(norms)
                stats["avg_mean"] = sum(means) / len(means)
                stats["avg_std"] = sum(stds) / len(stds)
        
        return stats
    
    def compare_adapters(self, adapters: List[ModelAdapter]) -> Dict[str, Any]:
        """比较多个适配器"""
        comparison = {
            "num_adapters": len(adapters),
            "adapters": []
        }
        
        for adapter in adapters:
            analysis = self.analyze_adapter(adapter)
            comparison["adapters"].append(analysis)
        
        return comparison
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取分析历史"""
        return self._analysis_history.copy()


# ==================== 适配器工厂 ====================

class AdapterFactory:
    """
    适配器工厂 - 生产级实现
    
    提供：
    - 适配器注册和创建
    - 工厂级别指标
    - 可用适配器查询
    """
    
    _registry: Dict[AdapterType, type] = {
        AdapterType.BACKBONE: BackboneAdapter,
        AdapterType.TASK_HEAD: TaskHeadAdapter,
        AdapterType.LORA: LoRAAdapter,
        AdapterType.PREFIX: PrefixAdapter,
        AdapterType.PROMPT: PromptAdapter,
        AdapterType.ADAPTER: AdapterLayersAdapter,
        AdapterType.BITFIT: BitFitAdapter,
        AdapterType.IA3: IA3Adapter,
        AdapterType.COMPACTER: CompacterAdapter
    }
    
    _metrics: Dict[str, Any] = {
        "total_created": 0,
        "created_by_type": {},
        "errors": 0
    }
    _metrics_lock = threading.Lock()
    
    @classmethod
    def register(cls, adapter_type: AdapterType, adapter_cls: type) -> None:
        """注册适配器"""
        cls._registry[adapter_type] = adapter_cls
        logger.info(f"Registered adapter: {adapter_type.value}")
    
    @classmethod
    def unregister(cls, adapter_type: AdapterType) -> bool:
        """注销适配器"""
        if adapter_type in cls._registry:
            del cls._registry[adapter_type]
            logger.info(f"Unregistered adapter: {adapter_type.value}")
            return True
        return False
    
    @classmethod
    def create(
        cls, 
        adapter_type: Union[AdapterType, str], 
        config: Optional[AdapterConfig] = None, 
        **kwargs
    ) -> ModelAdapter:
        """
        创建适配器
        
        Args:
            adapter_type: 适配器类型
            config: 适配器配置
            **kwargs: 额外参数
            
        Returns:
            适配器实例
        """
        try:
            if isinstance(adapter_type, str):
                adapter_type = AdapterType(adapter_type)
            
            adapter_cls = cls._registry.get(adapter_type)
            if adapter_cls is None:
                raise ValueError(f"Unknown adapter type: {adapter_type}")
            
            if config is None:
                config = AdapterConfig(
                    adapter_type=adapter_type, 
                    **{k: v for k, v in kwargs.items() if hasattr(AdapterConfig, k)}
                )
            
            # 过滤适配器特定的参数
            adapter_kwargs = {k: v for k, v in kwargs.items() if not hasattr(AdapterConfig, k)}
            
            adapter = adapter_cls(config, **adapter_kwargs)
            
            # 更新指标
            with cls._metrics_lock:
                cls._metrics["total_created"] += 1
                type_key = adapter_type.value
                cls._metrics["created_by_type"][type_key] = \
                    cls._metrics["created_by_type"].get(type_key, 0) + 1
            
            logger.debug(f"Created {adapter_type.value} adapter")
            return adapter
            
        except Exception as e:
            with cls._metrics_lock:
                cls._metrics["errors"] += 1
            logger.error(f"Error creating adapter: {e}")
            raise
    
    @classmethod
    def get_available_types(cls) -> List[str]:
        """获取可用适配器类型列表"""
        return [t.value for t in cls._registry.keys()]
    
    @classmethod
    def is_type_supported(cls, adapter_type: Union[AdapterType, str]) -> bool:
        """检查适配器类型是否支持"""
        if isinstance(adapter_type, str):
            try:
                adapter_type = AdapterType(adapter_type)
            except ValueError:
                return False
        return adapter_type in cls._registry
    
    @classmethod
    def get_factory_metrics(cls) -> Dict[str, Any]:
        """获取工厂指标"""
        with cls._metrics_lock:
            return cls._metrics.copy()
    
    @classmethod
    def reset_metrics(cls) -> None:
        """重置工厂指标"""
        with cls._metrics_lock:
            cls._metrics = {
                "total_created": 0,
                "created_by_type": {},
                "errors": 0
            }
    
    @classmethod
    def get_adapter_info(cls, adapter_type: Union[AdapterType, str]) -> Dict[str, Any]:
        """
        获取适配器信息
        
        Args:
            adapter_type: 适配器类型
            
        Returns:
            适配器信息
        """
        if isinstance(adapter_type, str):
            adapter_type = AdapterType(adapter_type)
        
        adapter_cls = cls._registry.get(adapter_type)
        if adapter_cls is None:
            return {"error": f"Unknown adapter type: {adapter_type}"}
        
        return {
            "type": adapter_type.value,
            "class_name": adapter_cls.__name__,
            "docstring": adapter_cls.__doc__,
            "module": adapter_cls.__module__
        }
    
    @classmethod
    def create_fusion(
        cls,
        adapters: List[ModelAdapter],
        fusion_method: str = "attention",
        hidden_size: int = 768
    ) -> AdapterFusion:
        """
        创建适配器融合
        
        Args:
            adapters: 适配器列表
            fusion_method: 融合方法
            hidden_size: 隐藏层大小
            
        Returns:
            融合模块
        """
        return AdapterFusion(adapters, fusion_method, hidden_size)


# ==================== 配置构建器 ====================

class AdapterConfigBuilder:
    """
    适配器配置构建器
    
    支持链式配置构建。
    """
    
    def __init__(self, adapter_type: Union[AdapterType, str] = AdapterType.LORA):
        if isinstance(adapter_type, str):
            adapter_type = AdapterType(adapter_type)
        self._config = {
            "adapter_type": adapter_type
        }
    
    def hidden_size(self, size: int) -> "AdapterConfigBuilder":
        """设置隐藏层大小"""
        self._config["hidden_size"] = size
        return self
    
    def lora_rank(self, rank: int) -> "AdapterConfigBuilder":
        """设置LoRA秩"""
        self._config["lora_rank"] = rank
        return self
    
    def lora_alpha(self, alpha: int) -> "AdapterConfigBuilder":
        """设置LoRA alpha"""
        self._config["lora_alpha"] = alpha
        return self
    
    def lora_dropout(self, dropout: float) -> "AdapterConfigBuilder":
        """设置LoRA dropout"""
        self._config["lora_dropout"] = dropout
        return self
    
    def lora_target_modules(self, modules: List[str]) -> "AdapterConfigBuilder":
        """设置LoRA目标模块"""
        self._config["lora_target_modules"] = modules
        return self
    
    def use_dora(self, enabled: bool = True) -> "AdapterConfigBuilder":
        """启用DoRA"""
        self._config["lora_use_dora"] = enabled
        return self
    
    def use_rslora(self, enabled: bool = True) -> "AdapterConfigBuilder":
        """启用RS-LoRA"""
        self._config["lora_use_rslora"] = enabled
        return self
    
    def prefix_length(self, length: int) -> "AdapterConfigBuilder":
        """设置前缀长度"""
        self._config["prefix_length"] = length
        return self
    
    def num_virtual_tokens(self, num: int) -> "AdapterConfigBuilder":
        """设置虚拟token数量"""
        self._config["num_virtual_tokens"] = num
        return self
    
    def adapter_bottleneck(self, size: int) -> "AdapterConfigBuilder":
        """设置适配器瓶颈大小"""
        self._config["adapter_bottleneck"] = size
        return self
    
    def init_strategy(self, strategy: Union[InitStrategy, str]) -> "AdapterConfigBuilder":
        """设置初始化策略"""
        if isinstance(strategy, str):
            strategy = InitStrategy(strategy)
        self._config["init_strategy"] = strategy
        return self
    
    def merge_strategy(self, strategy: Union[MergeStrategy, str]) -> "AdapterConfigBuilder":
        """设置合并策略"""
        if isinstance(strategy, str):
            strategy = MergeStrategy(strategy)
        self._config["merge_strategy"] = strategy
        return self
    
    def enable_metrics(self, enabled: bool = True) -> "AdapterConfigBuilder":
        """启用指标收集"""
        self._config["enable_metrics"] = enabled
        return self
    
    def gradient_checkpointing(self, enabled: bool = True) -> "AdapterConfigBuilder":
        """启用梯度检查点"""
        self._config["use_gradient_checkpointing"] = enabled
        return self
    
    def dynamic_rank(self, enabled: bool = True, pattern: Optional[Dict[str, int]] = None) -> "AdapterConfigBuilder":
        """启用动态秩"""
        self._config["dynamic_rank"] = enabled
        if pattern:
            self._config["rank_pattern"] = pattern
        return self
    
    def build(self) -> AdapterConfig:
        """构建配置"""
        return AdapterConfig(**self._config)
    
    def create_adapter(self, **kwargs) -> ModelAdapter:
        """创建适配器"""
        config = self.build()
        return AdapterFactory.create(config.adapter_type, config, **kwargs)


# ==================== 便捷函数 ====================

def create_adapter(
    adapter_type: Union[AdapterType, str],
    hidden_size: int = 768,
    **kwargs
) -> ModelAdapter:
    """
    便捷函数：创建适配器
    
    Args:
        adapter_type: 适配器类型
        hidden_size: 隐藏层大小
        **kwargs: 其他配置
        
    Returns:
        适配器实例
    """
    config = AdapterConfig(
        adapter_type=AdapterType(adapter_type) if isinstance(adapter_type, str) else adapter_type,
        hidden_size=hidden_size,
        **{k: v for k, v in kwargs.items() if hasattr(AdapterConfig, k)}
    )
    return AdapterFactory.create(adapter_type, config, **kwargs)


def build_adapter_config(adapter_type: Union[AdapterType, str] = "lora") -> AdapterConfigBuilder:
    """
    便捷函数：创建配置构建器
    
    Args:
        adapter_type: 适配器类型
        
    Returns:
        配置构建器
    """
    return AdapterConfigBuilder(adapter_type)


def create_lora_adapter(
    hidden_size: int = 768,
    rank: int = 8,
    alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    use_dora: bool = False,
    use_rslora: bool = False,
    **kwargs
) -> LoRAAdapter:
    """
    便捷函数：创建LoRA适配器
    
    Args:
        hidden_size: 隐藏层大小
        rank: LoRA秩
        alpha: LoRA alpha
        target_modules: 目标模块
        use_dora: 是否使用DoRA
        use_rslora: 是否使用RS-LoRA
        **kwargs: 其他配置
        
    Returns:
        LoRA适配器
    """
    config = AdapterConfig(
        adapter_type=AdapterType.LORA,
        hidden_size=hidden_size,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_target_modules=target_modules or ['q_proj', 'v_proj'],
        lora_use_dora=use_dora,
        lora_use_rslora=use_rslora,
        **kwargs
    )
    return LoRAAdapter(config)


def create_adapter_fusion(
    adapters: List[ModelAdapter],
    fusion_method: str = "attention",
    hidden_size: int = 768
) -> AdapterFusion:
    """
    便捷函数：创建适配器融合
    
    Args:
        adapters: 适配器列表
        fusion_method: 融合方法
        hidden_size: 隐藏层大小
        
    Returns:
        融合模块
    """
    return AdapterFactory.create_fusion(adapters, fusion_method, hidden_size)


def merge_adapters(
    adapters: List[ModelAdapter],
    weights: Optional[List[float]] = None,
    strategy: MergeStrategy = MergeStrategy.LINEAR
) -> Dict[str, torch.Tensor]:
    """
    便捷函数：合并适配器
    
    Args:
        adapters: 适配器列表
        weights: 权重列表
        strategy: 合并策略
        
    Returns:
        合并后的参数
    """
    if weights is None:
        weights = [1.0 / len(adapters)] * len(adapters)
    
    if strategy == MergeStrategy.LINEAR:
        return AdapterMerger.merge_linear(adapters, weights)
    elif strategy == MergeStrategy.TIES:
        return AdapterMerger.merge_ties(adapters)
    else:
        return AdapterMerger.merge_linear(adapters, weights)


def adapter_factory_health_check() -> Dict[str, Any]:
    """
    工厂健康检查
    
    Returns:
        健康状态
    """
    health = {
        "status": "healthy",
        "available_types": AdapterFactory.get_available_types(),
        "metrics": AdapterFactory.get_factory_metrics()
    }
    
    # 检查各适配器类型是否可以创建
    errors = []
    for adapter_type in AdapterType:
        if AdapterFactory.is_type_supported(adapter_type):
            try:
                config = AdapterConfig(adapter_type=adapter_type)
                adapter = AdapterFactory.create(adapter_type, config)
                del adapter
            except Exception as e:
                errors.append(f"{adapter_type.value}: {str(e)}")
    
    if errors:
        health["status"] = "degraded"
        health["errors"] = errors
    
    return health


def get_adapter_summary(adapter: ModelAdapter) -> Dict[str, Any]:
    """
    获取适配器摘要
    
    Args:
        adapter: 适配器
        
    Returns:
        摘要信息
    """
    summary = {
        "type": adapter.adapter_type.value,
        "status": adapter.get_status().value,
        "parameter_count": adapter.get_parameter_count(),
        "memory_usage": adapter.get_memory_usage(),
        "adapted_modules": list(adapter.get_adapted_modules()),
        "metrics": adapter.get_metrics()
    }
    
    # 类型特定的摘要
    if isinstance(adapter, LoRAAdapter):
        summary["lora_analysis"] = adapter.get_lora_analysis()
    elif isinstance(adapter, AdapterLayersAdapter):
        summary["adapter_analysis"] = adapter.get_adapter_analysis()
    elif isinstance(adapter, BitFitAdapter):
        summary["bias_analysis"] = adapter.get_bias_analysis()
    elif isinstance(adapter, IA3Adapter):
        summary["ia3_analysis"] = adapter.get_ia3_analysis()
    elif isinstance(adapter, CompacterAdapter):
        summary["compacter_analysis"] = adapter.get_compacter_analysis()
    
    return summary


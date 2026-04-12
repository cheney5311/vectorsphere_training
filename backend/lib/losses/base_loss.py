# -*- coding: utf-8 -*-
"""
损失函数基类

定义所有损失函数的统一接口和基础功能。
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from enum import Enum
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class LossType(Enum):
    """损失类型枚举"""
    # 监督学习
    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"
    LABEL_SMOOTHING = "label_smoothing"
    MSE = "mse"
    MAE = "mae"
    HUBER = "huber"
    DICE = "dice"
    IOU = "iou"
    
    # 蒸馏
    SOFT_LABEL = "soft_label"
    FEATURE_KD = "feature_kd"
    ATTENTION_KD = "attention_kd"
    RELATIONAL_KD = "relational_kd"
    
    # 对比学习
    INFONCE = "infonce"
    NT_XENT = "nt_xent"
    TRIPLET = "triplet"
    CENTER = "center"
    CLIP = "clip"
    
    # 复合
    COMPOSITE = "composite"
    MULTI_TASK = "multi_task"
    
    # 正则化
    L1_REG = "l1_reg"
    L2_REG = "l2_reg"
    ELASTIC_NET = "elastic_net"
    
    @classmethod
    def from_string(cls, s: str) -> 'LossType':
        """从字符串创建"""
        try:
            return cls(s.lower())
        except ValueError:
            # 尝试通过名称匹配
            for loss_type in cls:
                if loss_type.name.lower() == s.lower():
                    return loss_type
            raise ValueError(f"Unknown loss type: {s}")
    
    @property
    def is_supervised(self) -> bool:
        """是否为监督学习损失"""
        return self in (
            LossType.CROSS_ENTROPY, LossType.FOCAL, LossType.LABEL_SMOOTHING,
            LossType.MSE, LossType.MAE, LossType.HUBER, LossType.DICE, LossType.IOU
        )
    
    @property
    def is_distillation(self) -> bool:
        """是否为蒸馏损失"""
        return self in (
            LossType.SOFT_LABEL, LossType.FEATURE_KD, 
            LossType.ATTENTION_KD, LossType.RELATIONAL_KD
        )
    
    @property
    def is_contrastive(self) -> bool:
        """是否为对比学习损失"""
        return self in (
            LossType.INFONCE, LossType.NT_XENT, 
            LossType.TRIPLET, LossType.CENTER, LossType.CLIP
        )
    
    @property
    def is_regularization(self) -> bool:
        """是否为正则化损失"""
        return self in (LossType.L1_REG, LossType.L2_REG, LossType.ELASTIC_NET)


@dataclass
class LossConfig:
    """损失函数配置"""
    loss_type: LossType = LossType.CROSS_ENTROPY
    weight: float = 1.0
    reduction: str = "mean"  # mean, sum, none
    
    # 类别权重
    class_weights: Optional[List[float]] = None
    
    # 标签平滑
    label_smoothing: float = 0.0
    
    # 温度（用于蒸馏）
    temperature: float = 1.0
    
    # 其他参数
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    # 新增：调度和自适应
    dynamic_weight: bool = False
    weight_schedule: Optional[Callable[[int], float]] = None
    
    # 新增：梯度处理
    clip_grad: Optional[float] = None
    normalize_grad: bool = False
    
    # 新增：监控
    log_freq: int = 100
    track_metrics: bool = True
    
    def validate(self) -> List[str]:
        """
        验证配置
        
        Returns:
            错误列表
        """
        errors = []
        
        if self.weight < 0:
            errors.append(f"weight must be non-negative, got {self.weight}")
        
        if self.reduction not in ("mean", "sum", "none"):
            errors.append(f"reduction must be mean/sum/none, got {self.reduction}")
        
        if not 0 <= self.label_smoothing <= 1:
            errors.append(f"label_smoothing must be in [0, 1], got {self.label_smoothing}")
        
        if self.temperature <= 0:
            errors.append(f"temperature must be positive, got {self.temperature}")
        
        if self.clip_grad is not None and self.clip_grad <= 0:
            errors.append(f"clip_grad must be positive, got {self.clip_grad}")
        
        if self.log_freq <= 0:
            errors.append(f"log_freq must be positive, got {self.log_freq}")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'loss_type': self.loss_type.value,
            'weight': self.weight,
            'reduction': self.reduction,
            'class_weights': self.class_weights,
            'label_smoothing': self.label_smoothing,
            'temperature': self.temperature,
            'dynamic_weight': self.dynamic_weight,
            'clip_grad': self.clip_grad,
            'normalize_grad': self.normalize_grad,
            'log_freq': self.log_freq,
            'track_metrics': self.track_metrics,
            'extra_params': self.extra_params,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LossConfig':
        """从字典创建"""
        loss_type = data.get('loss_type')
        if isinstance(loss_type, str):
            loss_type = LossType.from_string(loss_type)
        elif not isinstance(loss_type, LossType):
            loss_type = LossType.CROSS_ENTROPY
        
        return cls(
            loss_type=loss_type,
            weight=data.get('weight', 1.0),
            reduction=data.get('reduction', 'mean'),
            class_weights=data.get('class_weights'),
            label_smoothing=data.get('label_smoothing', 0.0),
            temperature=data.get('temperature', 1.0),
            dynamic_weight=data.get('dynamic_weight', False),
            clip_grad=data.get('clip_grad'),
            normalize_grad=data.get('normalize_grad', False),
            log_freq=data.get('log_freq', 100),
            track_metrics=data.get('track_metrics', True),
            extra_params=data.get('extra_params', {}),
        )
    
    def get_effective_weight(self, step: int) -> float:
        """
        获取有效权重
        
        Args:
            step: 当前步骤
            
        Returns:
            有效权重
        """
        if not self.dynamic_weight or self.weight_schedule is None:
            return self.weight
        
        return self.weight * self.weight_schedule(step)


@dataclass
class LossResult:
    """损失计算结果"""
    loss: Tensor  # 总损失
    components: Dict[str, Tensor] = field(default_factory=dict)  # 各组件损失
    metrics: Dict[str, float] = field(default_factory=dict)  # 额外指标
    
    # 新增：元数据
    timestamp: float = field(default_factory=time.time)
    step: Optional[int] = None
    grad_norm: Optional[float] = None
    
    def backward(self, retain_graph: bool = False, create_graph: bool = False):
        """反向传播"""
        if self.loss.requires_grad:
            self.loss.backward(retain_graph=retain_graph, create_graph=create_graph)
    
    def item(self) -> float:
        """获取标量值"""
        return self.loss.item()
    
    def detach(self) -> 'LossResult':
        """分离计算图"""
        return LossResult(
            loss=self.loss.detach(),
            components={k: v.detach() if isinstance(v, Tensor) else v 
                       for k, v in self.components.items()},
            metrics=self.metrics.copy(),
            timestamp=self.timestamp,
            step=self.step,
            grad_norm=self.grad_norm
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'loss': self.loss.item(),
            'metrics': self.metrics,
            'timestamp': self.timestamp,
        }
        
        if self.step is not None:
            result['step'] = self.step
        
        if self.grad_norm is not None:
            result['grad_norm'] = self.grad_norm
        
        for name, comp in self.components.items():
            result[name] = comp.item() if isinstance(comp, Tensor) else comp
        
        return result
    
    def __add__(self, other: 'LossResult') -> 'LossResult':
        """组合多个损失结果"""
        combined_components = {**self.components}
        for name, comp in other.components.items():
            if name in combined_components:
                combined_components[f"{name}_2"] = comp
            else:
                combined_components[name] = comp
        
        combined_metrics = {**self.metrics, **other.metrics}
        
        return LossResult(
            loss=self.loss + other.loss,
            components=combined_components,
            metrics=combined_metrics,
            timestamp=self.timestamp,
            step=self.step
        )
    
    def scale(self, factor: float) -> 'LossResult':
        """缩放损失"""
        return LossResult(
            loss=self.loss * factor,
            components={k: v * factor if isinstance(v, Tensor) else v 
                       for k, v in self.components.items()},
            metrics=self.metrics.copy(),
            timestamp=self.timestamp,
            step=self.step,
            grad_norm=self.grad_norm
        )


@dataclass
class LossStats:
    """损失统计"""
    total_steps: int = 0
    total_loss: float = 0.0
    avg_loss: float = 0.0
    min_loss: float = float('inf')
    max_loss: float = float('-inf')
    
    # 组件统计
    component_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_steps': self.total_steps,
            'total_loss': self.total_loss,
            'avg_loss': self.avg_loss,
            'min_loss': self.min_loss if self.min_loss != float('inf') else None,
            'max_loss': self.max_loss if self.max_loss != float('-inf') else None,
            'component_stats': self.component_stats,
        }


class LossMonitor:
    """损失监控器"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._history: List[LossResult] = []
        self._stats = LossStats()
        
    def record(self, result: LossResult) -> None:
        """记录损失结果"""
        self._history.append(result.detach())
        if len(self._history) > self.max_history:
            self._history.pop(0)
        
        # 更新统计
        self._update_stats(result)
    
    def _update_stats(self, result: LossResult) -> None:
        """更新统计"""
        loss_val = result.item()
        
        self._stats.total_steps += 1
        self._stats.total_loss += loss_val
        self._stats.avg_loss = self._stats.total_loss / self._stats.total_steps
        self._stats.min_loss = min(self._stats.min_loss, loss_val)
        self._stats.max_loss = max(self._stats.max_loss, loss_val)
        
        # 更新组件统计
        for name, comp in result.components.items():
            if name not in self._stats.component_stats:
                self._stats.component_stats[name] = {
                    'total': 0.0,
                    'avg': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                }
            
            comp_val = comp.item() if isinstance(comp, Tensor) else comp
            stats = self._stats.component_stats[name]
            stats['total'] += comp_val
            stats['avg'] = stats['total'] / self._stats.total_steps
            stats['min'] = min(stats['min'], comp_val)
            stats['max'] = max(stats['max'], comp_val)
    
    def get_stats(self) -> LossStats:
        """获取统计信息"""
        return self._stats
    
    def get_recent_losses(self, n: int = 10) -> List[float]:
        """获取最近的损失值"""
        return [r.item() for r in self._history[-n:]]
    
    def get_trend(self, window: int = 100) -> float:
        """
        获取损失趋势
        
        Returns:
            趋势值（正数表示上升，负数表示下降）
        """
        if len(self._history) < 2:
            return 0.0
        
        recent = self._history[-window:]
        if len(recent) < 2:
            return 0.0
        
        first_avg = sum(r.item() for r in recent[:len(recent)//2]) / (len(recent)//2)
        second_avg = sum(r.item() for r in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        
        return second_avg - first_avg
    
    def is_diverging(self, threshold: float = 2.0) -> bool:
        """
        检查损失是否发散
        
        Args:
            threshold: 阈值倍数
            
        Returns:
            是否发散
        """
        if len(self._history) < 10:
            return False
        
        recent = self.get_recent_losses(10)
        recent_avg = sum(recent) / len(recent)
        
        return recent_avg > self._stats.avg_loss * threshold
    
    def reset(self) -> None:
        """重置监控器"""
        self._history.clear()
        self._stats = LossStats()


class LossValidator:
    """损失验证器"""
    
    def __init__(self):
        self._checks: List[Callable[[LossResult], Optional[str]]] = []
    
    def add_check(self, check: Callable[[LossResult], Optional[str]]) -> None:
        """
        添加验证检查
        
        Args:
            check: 检查函数，返回None表示通过，返回字符串表示错误
        """
        self._checks.append(check)
    
    def validate(self, result: LossResult) -> List[str]:
        """
        验证损失结果
        
        Args:
            result: 损失结果
            
        Returns:
            错误列表
        """
        errors = []
        
        # 内置检查
        if torch.isnan(result.loss):
            errors.append("Loss is NaN")
        
        if torch.isinf(result.loss):
            errors.append("Loss is Inf")
        
        if result.loss.item() < 0:
            errors.append(f"Loss is negative: {result.loss.item()}")
        
        # 自定义检查
        for check in self._checks:
            error = check(result)
            if error:
                errors.append(error)
        
        return errors


class BaseLoss(nn.Module, ABC):
    """
    损失函数基类
    
    所有损失函数都继承此类，提供统一的接口：
    - forward: 计算损失
    - compute: 计算并返回 LossResult
    - extra_repr: 字符串表示
    """
    
    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__()
        self.config = config or LossConfig()
        self._name = self.__class__.__name__
        
        # 新增：内部组件
        self._monitor = LossMonitor()
        self._validator = LossValidator()
        self._step = 0
        
        # 验证配置
        errors = self.config.validate()
        if errors:
            logger.warning(f"Loss config validation errors: {errors}")
        
    @property
    def name(self) -> str:
        """损失函数名称"""
        return self._name
    
    @abstractmethod
    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor, 
        **kwargs
    ) -> Tensor:
        """
        计算损失
        
        Args:
            predictions: 模型预测
            targets: 真实标签
            **kwargs: 额外参数
            
        Returns:
            损失张量
        """
        pass
    
    def compute(
        self, 
        predictions: Tensor, 
        targets: Tensor, 
        **kwargs
    ) -> LossResult:
        """
        计算损失并返回结构化结果
        
        Args:
            predictions: 模型预测
            targets: 真实标签
            **kwargs: 额外参数
            
        Returns:
            LossResult 对象
        """
        # 计算基础损失
        loss = self.forward(predictions, targets, **kwargs)
        
        # 获取有效权重
        effective_weight = self.config.get_effective_weight(self._step)
        
        # 创建结果
        result = LossResult(
            loss=loss * effective_weight,
            components={self.name: loss},
            metrics={f'{self.name}_raw': loss.item()},
            step=self._step
        )
        
        # 记录到监控器
        if self.config.track_metrics:
            self._monitor.record(result)
        
        # 验证结果
        if self.training:
            errors = self._validator.validate(result)
            if errors:
                logger.warning(f"Loss validation errors: {errors}")
        
        # 增加步数
        self._step += 1
        
        return result
    
    def extra_repr(self) -> str:
        """字符串表示"""
        return f"weight={self.config.weight}, reduction={self.config.reduction}"
    
    # ==================== 新增方法 ====================
    
    def get_monitor(self) -> LossMonitor:
        """获取监控器"""
        return self._monitor
    
    def get_validator(self) -> LossValidator:
        """获取验证器"""
        return self._validator
    
    def get_stats(self) -> LossStats:
        """获取统计信息"""
        return self._monitor.get_stats()
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._monitor.reset()
        self._step = 0
    
    def is_diverging(self) -> bool:
        """检查是否发散"""
        return self._monitor.is_diverging()
    
    def set_weight(self, weight: float) -> None:
        """设置权重"""
        self.config.weight = weight
    
    def set_weight_schedule(self, schedule: Callable[[int], float]) -> None:
        """
        设置权重调度
        
        Args:
            schedule: 调度函数，输入步数，输出权重倍数
        """
        self.config.weight_schedule = schedule
        self.config.dynamic_weight = True
    
    def enable_tracking(self) -> None:
        """启用跟踪"""
        self.config.track_metrics = True
    
    def disable_tracking(self) -> None:
        """禁用跟踪"""
        self.config.track_metrics = False
    
    def add_validation_check(self, check: Callable[[LossResult], Optional[str]]) -> None:
        """添加验证检查"""
        self._validator.add_check(check)
    
    def _apply_grad_clipping(self, loss: Tensor) -> Tensor:
        """应用梯度裁剪"""
        if self.config.clip_grad is not None and loss.requires_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.clip_grad)
        return loss
    
    def _apply_grad_normalization(self, loss: Tensor) -> Tensor:
        """应用梯度归一化"""
        if self.config.normalize_grad and loss.requires_grad:
            # 这将在backward时自动归一化梯度
            pass
        return loss
    
    def summary(self) -> Dict[str, Any]:
        """获取损失函数摘要"""
        stats = self.get_stats()
        
        return {
            'name': self.name,
            'type': self.config.loss_type.value,
            'weight': self.config.weight,
            'steps': self._step,
            'stats': stats.to_dict(),
            'config': self.config.to_dict(),
        }
    
    def print_summary(self) -> None:
        """打印摘要"""
        summary = self.summary()
        
        print("\n" + "="*80)
        print(f"Loss Function: {summary['name']}")
        print("="*80)
        
        print(f"\nType: {summary['type']}")
        print(f"Weight: {summary['weight']}")
        print(f"Steps: {summary['steps']}")
        
        stats = summary['stats']
        print(f"\nStatistics:")
        print(f"  Avg Loss: {stats['avg_loss']:.6f}")
        print(f"  Min Loss: {stats['min_loss']:.6f}")
        print(f"  Max Loss: {stats['max_loss']:.6f}")
        
        print("="*80)


class LossRegistry:
    """
    损失函数注册表
    
    用于动态注册和获取损失函数类。
    """
    
    _registry: Dict[str, type] = {}
    _configs: Dict[str, LossConfig] = {}
    _instances: Dict[str, BaseLoss] = {}  # 新增：缓存实例
    _metadata: Dict[str, Dict[str, Any]] = {}  # 新增：元数据
    
    @classmethod
    def register(
        cls, 
        name: str, 
        loss_class: type,
        default_config: Optional[LossConfig] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        注册损失函数
        
        Args:
            name: 损失函数名称
            loss_class: 损失函数类
            default_config: 默认配置
            metadata: 元数据
        """
        cls._registry[name.lower()] = loss_class
        if default_config:
            cls._configs[name.lower()] = default_config
        if metadata:
            cls._metadata[name.lower()] = metadata
        logger.debug(f"Registered loss: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """
        获取损失函数类
        
        Args:
            name: 损失函数名称
            
        Returns:
            损失函数类或 None
        """
        return cls._registry.get(name.lower())
    
    @classmethod
    def get_config(cls, name: str) -> Optional[LossConfig]:
        """获取默认配置"""
        return cls._configs.get(name.lower())
    
    @classmethod
    def get_metadata(cls, name: str) -> Optional[Dict[str, Any]]:
        """获取元数据"""
        return cls._metadata.get(name.lower())
    
    @classmethod
    def create(
        cls, 
        name: str, 
        config: Optional[LossConfig] = None,
        cache: bool = False,
        **kwargs
    ) -> Optional[BaseLoss]:
        """
        创建损失函数实例
        
        Args:
            name: 损失函数名称
            config: 配置
            cache: 是否缓存实例
            **kwargs: 额外参数
            
        Returns:
            损失函数实例或 None
        """
        # 检查缓存
        if cache and name.lower() in cls._instances:
            return cls._instances[name.lower()]
        
        loss_class = cls.get(name)
        if loss_class is None:
            logger.warning(f"Loss function not found: {name}")
            return None
        
        # 合并配置
        final_config = config or cls.get_config(name) or LossConfig()
        
        try:
            instance = loss_class(config=final_config, **kwargs)
            
            # 缓存实例
            if cache:
                cls._instances[name.lower()] = instance
            
            return instance
        except Exception as e:
            logger.error(f"Failed to create loss {name}: {e}")
            return None
    
    @classmethod
    def list_registered(cls) -> List[str]:
        """列出所有已注册的损失函数"""
        return list(cls._registry.keys())
    
    @classmethod
    def list_by_type(cls, loss_type: LossType) -> List[str]:
        """
        按类型列出损失函数
        
        Args:
            loss_type: 损失类型
            
        Returns:
            损失函数名称列表
        """
        results = []
        for name, config in cls._configs.items():
            if config.loss_type == loss_type:
                results.append(name)
        return results
    
    @classmethod
    def search(cls, keyword: str) -> List[str]:
        """
        搜索损失函数
        
        Args:
            keyword: 关键词
            
        Returns:
            匹配的损失函数名称列表
        """
        keyword = keyword.lower()
        return [name for name in cls._registry.keys() if keyword in name]
    
    @classmethod
    def get_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        获取损失函数信息
        
        Args:
            name: 损失函数名称
            
        Returns:
            信息字典
        """
        loss_class = cls.get(name)
        if loss_class is None:
            return None
        
        config = cls.get_config(name)
        metadata = cls.get_metadata(name)
        
        return {
            'name': name,
            'class': loss_class.__name__,
            'config': config.to_dict() if config else None,
            'metadata': metadata,
            'doc': loss_class.__doc__,
        }
    
    @classmethod
    def validate_name(cls, name: str) -> bool:
        """验证名称是否已注册"""
        return name.lower() in cls._registry
    
    @classmethod
    def clear(cls):
        """清空注册表"""
        cls._registry.clear()
        cls._configs.clear()
        cls._instances.clear()
        cls._metadata.clear()
    
    @classmethod
    def clear_cache(cls):
        """清空实例缓存"""
        cls._instances.clear()
    
    @classmethod
    def print_registry(cls) -> None:
        """打印注册表"""
        print("\n" + "="*80)
        print("Loss Function Registry")
        print("="*80)
        
        print(f"\nTotal registered: {len(cls._registry)}")
        
        # 按类型分组
        by_type = defaultdict(list)
        for name, config in cls._configs.items():
            by_type[config.loss_type.value].append(name)
        
        for loss_type, names in sorted(by_type.items()):
            print(f"\n{loss_type}:")
            for name in sorted(names):
                print(f"  - {name}")
        
        print("="*80)


# ==================== 辅助函数 ====================

def register_loss(name: str, default_config: Optional[LossConfig] = None, 
                 metadata: Optional[Dict[str, Any]] = None):
    """
    装饰器：注册损失函数
    
    Usage:
        @register_loss("my_loss")
        class MyLoss(BaseLoss):
            ...
    """
    def decorator(cls):
        LossRegistry.register(name, cls, default_config, metadata)
        return cls
    return decorator


def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """
    应用损失缩减
    
    Args:
        loss: 原始损失
        reduction: 缩减方式 (mean, sum, none)
        
    Returns:
        缩减后的损失
    """
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def weighted_loss(
    loss: Tensor, 
    weights: Optional[Tensor] = None,
    reduction: str = "mean"
) -> Tensor:
    """
    应用样本权重
    
    Args:
        loss: 原始损失 [N] 或 [N, ...]
        weights: 样本权重 [N]
        reduction: 缩减方式
        
    Returns:
        加权后的损失
    """
    if weights is not None:
        # 确保权重维度匹配
        while weights.dim() < loss.dim():
            weights = weights.unsqueeze(-1)
        loss = loss * weights
    
    return reduce_loss(loss, reduction)


# ==================== 新增工具函数 ====================

def combine_losses(losses: List[LossResult], weights: Optional[List[float]] = None) -> LossResult:
    """
    组合多个损失
    
    Args:
        losses: 损失结果列表
        weights: 权重列表
        
    Returns:
        组合后的损失结果
    """
    if not losses:
        raise ValueError("No losses to combine")
    
    if weights is None:
        weights = [1.0] * len(losses)
    
    if len(losses) != len(weights):
        raise ValueError("Number of losses and weights must match")
    
    # 组合损失
    total_loss = sum(loss.loss * weight for loss, weight in zip(losses, weights))
    
    # 合并组件
    combined_components = {}
    for loss in losses:
        for name, comp in loss.components.items():
            if name in combined_components:
                combined_components[f"{name}_dup"] = comp
            else:
                combined_components[name] = comp
    
    # 合并指标
    combined_metrics = {}
    for loss in losses:
        combined_metrics.update(loss.metrics)
    
    return LossResult(
        loss=total_loss,
        components=combined_components,
        metrics=combined_metrics,
        step=losses[0].step
    )


def validate_loss_tensor(loss: Tensor, name: str = "loss") -> List[str]:
    """
    验证损失张量
    
    Args:
        loss: 损失张量
        name: 损失名称
        
    Returns:
        错误列表
    """
    errors = []
    
    if torch.isnan(loss).any():
        errors.append(f"{name} contains NaN")
    
    if torch.isinf(loss).any():
        errors.append(f"{name} contains Inf")
    
    if (loss < 0).any():
        errors.append(f"{name} contains negative values")
    
    return errors


def create_loss_from_config(config_dict: Dict[str, Any]) -> Optional[BaseLoss]:
    """
    从配置字典创建损失函数
    
    Args:
        config_dict: 配置字典
        
    Returns:
        损失函数实例
    """
    name = config_dict.get('name')
    if not name:
        logger.error("Loss name not specified in config")
        return None
    
    config_data = config_dict.get('config', {})
    config = LossConfig.from_dict(config_data)
    
    return LossRegistry.create(name, config)


def linear_weight_schedule(start: float = 0.0, end: float = 1.0, 
                          steps: int = 1000) -> Callable[[int], float]:
    """
    创建线性权重调度
    
    Args:
        start: 起始权重倍数
        end: 结束权重倍数
        steps: 步数
        
    Returns:
        调度函数
    """
    def schedule(step: int) -> float:
        if step >= steps:
            return end
        progress = step / steps
        return start + (end - start) * progress
    
    return schedule


def cosine_weight_schedule(min_weight: float = 0.1, max_weight: float = 1.0,
                          period: int = 1000) -> Callable[[int], float]:
    """
    创建余弦权重调度
    
    Args:
        min_weight: 最小权重倍数
        max_weight: 最大权重倍数
        period: 周期
        
    Returns:
        调度函数
    """
    import math
    
    def schedule(step: int) -> float:
        progress = (step % period) / period
        cosine_val = (1 + math.cos(progress * math.pi)) / 2
        return min_weight + (max_weight - min_weight) * cosine_val
    
    return schedule


def exponential_weight_schedule(start: float = 1.0, decay: float = 0.95,
                               decay_steps: int = 100) -> Callable[[int], float]:
    """
    创建指数衰减权重调度
    
    Args:
        start: 起始权重倍数
        decay: 衰减因子
        decay_steps: 衰减步数
        
    Returns:
        调度函数
    """
    def schedule(step: int) -> float:
        return start * (decay ** (step / decay_steps))
    
    return schedule


@contextmanager
def loss_computation_context(loss_fn: BaseLoss, track: bool = True):
    """
    损失计算上下文
    
    Args:
        loss_fn: 损失函数
        track: 是否跟踪
        
    Yields:
        损失函数
    """
    original_track = loss_fn.config.track_metrics
    
    if not track:
        loss_fn.disable_tracking()
    
    try:
        yield loss_fn
    finally:
        if not track and original_track:
            loss_fn.enable_tracking()


@contextmanager
def temporary_weight(loss_fn: BaseLoss, weight: float):
    """
    临时权重上下文
    
    Args:
        loss_fn: 损失函数
        weight: 临时权重
        
    Yields:
        损失函数
    """
    original_weight = loss_fn.config.weight
    loss_fn.set_weight(weight)
    
    try:
        yield loss_fn
    finally:
        loss_fn.set_weight(original_weight)


def print_loss_comparison(losses: Dict[str, BaseLoss]) -> None:
    """
    打印损失函数对比
    
    Args:
        losses: 损失函数字典
    """
    print("\n" + "="*80)
    print("Loss Function Comparison")
    print("="*80)
    
    print(f"\n{'Name':<20} {'Type':<15} {'Weight':<10} {'Avg Loss':<12} {'Steps':<8}")
    print("-"*80)
    
    for name, loss_fn in losses.items():
        stats = loss_fn.get_stats()
        print(f"{name:<20} {loss_fn.config.loss_type.value:<15} "
              f"{loss_fn.config.weight:<10.2f} {stats.avg_loss:<12.6f} {stats.total_steps:<8}")
    
    print("="*80)


def aggregate_loss_stats(losses: List[BaseLoss]) -> Dict[str, Any]:
    """
    聚合损失统计
    
    Args:
        losses: 损失函数列表
        
    Returns:
        聚合统计
    """
    total_steps = sum(l.get_stats().total_steps for l in losses)
    total_avg_loss = sum(l.get_stats().avg_loss * l.get_stats().total_steps 
                        for l in losses) / max(total_steps, 1)
    
    return {
        'total_losses': len(losses),
        'total_steps': total_steps,
        'avg_loss': total_avg_loss,
        'individual_stats': [l.get_stats().to_dict() for l in losses],
    }



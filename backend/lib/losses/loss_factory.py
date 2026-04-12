# -*- coding: utf-8 -*-
"""
损失函数工厂

提供统一的损失函数创建接口。
"""

import logging
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn

from .base_loss import BaseLoss, LossConfig, LossResult, LossRegistry, LossType
from .supervised_loss import (
    CrossEntropyLoss, FocalLoss, LabelSmoothingLoss,
    MSELoss, MAELoss, HuberLoss, DiceLoss, IoULoss,
    SupervisedMonitor, ClassificationStats, RegressionStats, SegmentationStats,
    create_supervised_loss, recommend_loss_function
)
from .distillation_loss import (
    SoftLabelLoss, FeatureDistillationLoss, 
    AttentionDistillationLoss, RelationalDistillationLoss,
    CombinedDistillationLoss,
    create_distillation_loss as create_kd_loss
)
from .contrastive_loss import (
    InfoNCELoss, NTXentLoss, TripletLoss, 
    CenterLoss, CrossModalContrastiveLoss, CLIPLoss,
    create_contrastive_loss
)
from .composite_loss import (
    CompositeLoss, MultiTaskLoss, 
    DynamicWeightedLoss, UncertaintyWeightedLoss, GradNormLoss,
    create_composite_loss as create_comp_loss,
    create_multitask_loss as create_mt_loss
)
from .regularization_loss import (
    L1Regularization, L2Regularization, ElasticNetRegularization,
    ConsistencyRegularization, EntropyRegularization,
    FeatureRegularization, SpectralRegularization, MixedRegularization,
    create_regularization_loss
)

logger = logging.getLogger(__name__)


# ==================== 损失函数元数据 ====================

@dataclass
class LossMetadata:
    """损失函数元数据"""
    name: str
    category: str  # supervised, distillation, contrastive, composite, regularization
    task_type: str  # classification, regression, segmentation, etc.
    description: str = ""
    default_params: Dict[str, Any] = field(default_factory=dict)
    required_params: List[str] = field(default_factory=list)
    optional_params: List[str] = field(default_factory=list)
    supports_weighting: bool = True
    supports_reduction: bool = True
    reference: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LossMetadata':
        return cls(**data)


# 损失函数元数据注册
LOSS_METADATA: Dict[str, LossMetadata] = {
    # 监督学习 - 分类
    'cross_entropy': LossMetadata(
        name='cross_entropy',
        category='supervised',
        task_type='classification',
        description='Standard cross entropy loss for multi-class classification',
        default_params={'label_smoothing': 0.0},
        optional_params=['num_classes', 'class_weights', 'label_smoothing'],
    ),
    'focal': LossMetadata(
        name='focal',
        category='supervised',
        task_type='classification',
        description='Focal loss for handling class imbalance',
        default_params={'alpha': 0.25, 'gamma': 2.0},
        optional_params=['num_classes', 'alpha', 'gamma'],
        reference='Lin et al., Focal Loss for Dense Object Detection',
    ),
    'label_smoothing': LossMetadata(
        name='label_smoothing',
        category='supervised',
        task_type='classification',
        description='Label smoothing loss for improved generalization',
        default_params={'smoothing': 0.1},
        optional_params=['num_classes', 'smoothing'],
    ),
    
    # 监督学习 - 回归
    'mse': LossMetadata(
        name='mse',
        category='supervised',
        task_type='regression',
        description='Mean squared error loss',
        default_params={},
    ),
    'mae': LossMetadata(
        name='mae',
        category='supervised',
        task_type='regression',
        description='Mean absolute error loss',
        default_params={},
    ),
    'huber': LossMetadata(
        name='huber',
        category='supervised',
        task_type='regression',
        description='Huber loss for robust regression',
        default_params={'delta': 1.0},
        optional_params=['delta'],
    ),
    
    # 监督学习 - 分割
    'dice': LossMetadata(
        name='dice',
        category='supervised',
        task_type='segmentation',
        description='Dice loss for segmentation',
        default_params={'smooth': 1e-6},
        optional_params=['smooth'],
    ),
    'iou': LossMetadata(
        name='iou',
        category='supervised',
        task_type='segmentation',
        description='IoU loss for segmentation',
        default_params={'smooth': 1e-6},
        optional_params=['smooth'],
    ),
    
    # 蒸馏
    'soft_label': LossMetadata(
        name='soft_label',
        category='distillation',
        task_type='knowledge_distillation',
        description='Soft label distillation loss',
        default_params={'temperature': 4.0, 'alpha': 0.5},
        optional_params=['temperature', 'alpha'],
    ),
    'feature_kd': LossMetadata(
        name='feature_kd',
        category='distillation',
        task_type='knowledge_distillation',
        description='Feature distillation loss',
        default_params={},
    ),
    'attention_kd': LossMetadata(
        name='attention_kd',
        category='distillation',
        task_type='knowledge_distillation',
        description='Attention distillation loss',
        default_params={},
    ),
    
    # 对比学习
    'infonce': LossMetadata(
        name='infonce',
        category='contrastive',
        task_type='representation_learning',
        description='InfoNCE contrastive loss',
        default_params={'temperature': 0.07},
        optional_params=['temperature'],
    ),
    'nt_xent': LossMetadata(
        name='nt_xent',
        category='contrastive',
        task_type='representation_learning',
        description='NT-Xent loss (SimCLR)',
        default_params={'temperature': 0.5},
        optional_params=['temperature'],
    ),
    'triplet': LossMetadata(
        name='triplet',
        category='contrastive',
        task_type='metric_learning',
        description='Triplet margin loss',
        default_params={'margin': 1.0},
        optional_params=['margin'],
    ),
    'clip': LossMetadata(
        name='clip',
        category='contrastive',
        task_type='multimodal',
        description='CLIP-style contrastive loss',
        default_params={'temperature': 0.07, 'learnable_temperature': True},
        optional_params=['temperature', 'learnable_temperature'],
    ),
    
    # 正则化
    'l1_reg': LossMetadata(
        name='l1_reg',
        category='regularization',
        task_type='regularization',
        description='L1 regularization for sparsity',
        default_params={'lambda_': 1e-4},
        optional_params=['lambda_', 'lambda_schedule'],
    ),
    'l2_reg': LossMetadata(
        name='l2_reg',
        category='regularization',
        task_type='regularization',
        description='L2 regularization (weight decay)',
        default_params={'lambda_': 1e-4},
        optional_params=['lambda_', 'lambda_schedule'],
    ),
}


# ==================== 损失函数映射 ====================

LOSS_MAPPING = {
    # 监督学习
    'cross_entropy': CrossEntropyLoss,
    'ce': CrossEntropyLoss,
    'focal': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    'mse': MSELoss,
    'l2': MSELoss,
    'mae': MAELoss,
    'l1': MAELoss,
    'huber': HuberLoss,
    'smooth_l1': HuberLoss,
    'dice': DiceLoss,
    'iou': IoULoss,
    
    # 蒸馏
    'soft_label': SoftLabelLoss,
    'kd': SoftLabelLoss,
    'feature_kd': FeatureDistillationLoss,
    'attention_kd': AttentionDistillationLoss,
    'relational_kd': RelationalDistillationLoss,
    'combined_kd': CombinedDistillationLoss,
    
    # 对比学习
    'infonce': InfoNCELoss,
    'nt_xent': NTXentLoss,
    'simclr': NTXentLoss,
    'triplet': TripletLoss,
    'center': CenterLoss,
    'cross_modal': CrossModalContrastiveLoss,
    'clip': CLIPLoss,
    
    # 正则化
    'l1_reg': L1Regularization,
    'l2_reg': L2Regularization,
    'elastic_net': ElasticNetRegularization,
    'consistency': ConsistencyRegularization,
    'entropy': EntropyRegularization,
    'feature_reg': FeatureRegularization,
    'spectral_reg': SpectralRegularization,
    'mixed_reg': MixedRegularization,
}

# 类别到损失函数的映射
CATEGORY_MAPPING = {
    'supervised': ['cross_entropy', 'focal', 'label_smoothing', 'mse', 'mae', 'huber', 'dice', 'iou'],
    'distillation': ['soft_label', 'feature_kd', 'attention_kd', 'relational_kd', 'combined_kd'],
    'contrastive': ['infonce', 'nt_xent', 'triplet', 'center', 'cross_modal', 'clip'],
    'regularization': ['l1_reg', 'l2_reg', 'elastic_net', 'consistency', 'entropy', 'feature_reg', 'spectral_reg', 'mixed_reg'],
}

# 任务类型到损失函数的映射
TASK_MAPPING = {
    'classification': ['cross_entropy', 'focal', 'label_smoothing'],
    'regression': ['mse', 'mae', 'huber'],
    'segmentation': ['dice', 'iou', 'cross_entropy'],
    'knowledge_distillation': ['soft_label', 'feature_kd', 'attention_kd', 'combined_kd'],
    'representation_learning': ['infonce', 'nt_xent', 'triplet', 'center'],
    'multimodal': ['clip', 'cross_modal'],
    'regularization': ['l1_reg', 'l2_reg', 'elastic_net'],
}


@dataclass
class FactoryStats:
    """工厂统计"""
    total_created: int = 0
    creation_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    failed_creations: int = 0
    last_created: Optional[str] = None
    last_created_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_created': self.total_created,
            'creation_counts': dict(self.creation_counts),
            'failed_creations': self.failed_creations,
            'last_created': self.last_created,
            'last_created_time': self.last_created_time,
}


class LossFactory:
    """
    损失函数工厂
    
    统一的损失函数创建和管理接口。
    """
    
    _instance = None
    _custom_losses: Dict[str, type] = {}
    _custom_metadata: Dict[str, LossMetadata] = {}
    _stats: FactoryStats = FactoryStats()
    _cache: Dict[str, BaseLoss] = {}
    _cache_enabled: bool = False
    _validators: Dict[str, Callable] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(
        cls, 
        name: str, 
        loss_class: type,
        metadata: Optional[LossMetadata] = None
    ):
        """
        注册自定义损失函数
        
        Args:
            name: 损失函数名称
            loss_class: 损失函数类
            metadata: 元数据（可选）
        """
        cls._custom_losses[name.lower()] = loss_class
        
        if metadata:
            cls._custom_metadata[name.lower()] = metadata
        
        logger.info(f"Registered custom loss: {name}")
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        注销损失函数
        
        Args:
            name: 损失函数名称
            
        Returns:
            是否成功注销
        """
        name = name.lower()
        
        if name in cls._custom_losses:
            del cls._custom_losses[name]
            if name in cls._custom_metadata:
                del cls._custom_metadata[name]
            logger.info(f"Unregistered loss: {name}")
            return True
        
        return False
    
    @classmethod
    def get_class(cls, name: str) -> Optional[type]:
        """
        获取损失函数类
        
        Args:
            name: 损失函数名称
            
        Returns:
            损失函数类或None
        """
        name = name.lower()
        
        # 先查找自定义损失
        if name in cls._custom_losses:
            return cls._custom_losses[name]
        
        # 再查找内置损失
        if name in LOSS_MAPPING:
            return LOSS_MAPPING[name]
        
        # 最后查找注册表
        return LossRegistry.get(name)
    
    @classmethod
    def get_metadata(cls, name: str) -> Optional[LossMetadata]:
        """
        获取损失函数元数据
        
        Args:
            name: 损失函数名称
            
        Returns:
            元数据或None
        """
        name = name.lower()
        
        if name in cls._custom_metadata:
            return cls._custom_metadata[name]
        
        if name in LOSS_METADATA:
            return LOSS_METADATA[name]
        
        return None
    
    @classmethod
    def add_validator(cls, name: str, validator: Callable[[Dict[str, Any]], bool]):
        """
        添加参数验证器
        
        Args:
            name: 损失函数名称
            validator: 验证函数，接收参数字典，返回是否有效
        """
        cls._validators[name.lower()] = validator
    
    @classmethod
    def validate_params(cls, name: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证参数
        
        Args:
            name: 损失函数名称
            params: 参数字典
            
        Returns:
            (是否有效, 错误信息)
        """
        name = name.lower()
        
        # 使用自定义验证器
        if name in cls._validators:
            try:
                if not cls._validators[name](params):
                    return False, "Custom validation failed"
            except Exception as e:
                return False, f"Validation error: {e}"
        
        # 检查元数据中的必需参数
        metadata = cls.get_metadata(name)
        if metadata:
            for param in metadata.required_params:
                if param not in params:
                    return False, f"Missing required parameter: {param}"
        
        return True, ""
    
    @classmethod
    def enable_cache(cls):
        """启用缓存"""
        cls._cache_enabled = True
    
    @classmethod
    def disable_cache(cls):
        """禁用缓存"""
        cls._cache_enabled = False
    
    @classmethod
    def clear_cache(cls):
        """清空缓存"""
        cls._cache.clear()
    
    @classmethod
    def create(
        cls, 
        name: str, 
        config: Optional[LossConfig] = None,
        use_cache: bool = False,
        validate: bool = True,
        **kwargs
    ) -> Optional[BaseLoss]:
        """
        创建损失函数实例
        
        Args:
            name: 损失函数名称
            config: 配置
            use_cache: 是否使用缓存
            validate: 是否验证参数
            **kwargs: 额外参数
            
        Returns:
            损失函数实例或None
        """
        name_lower = name.lower()
        
        # 检查缓存
        cache_key = f"{name_lower}:{hash(frozenset(kwargs.items()))}"
        if (use_cache or cls._cache_enabled) and cache_key in cls._cache:
            return cls._cache[cache_key]
        
        # 验证参数
        if validate:
            is_valid, error_msg = cls.validate_params(name_lower, kwargs)
            if not is_valid:
                logger.warning(f"Invalid parameters for {name}: {error_msg}")
        
        loss_class = cls.get_class(name_lower)
        
        if loss_class is None:
            logger.warning(f"Unknown loss function: {name}")
            cls._stats.failed_creations += 1
            return None
        
        try:
            # 创建配置
            if config is None:
                config = LossConfig()
            
            # 合并默认参数
            metadata = cls.get_metadata(name_lower)
            if metadata:
                merged_kwargs = {**metadata.default_params, **kwargs}
            else:
                merged_kwargs = kwargs
            
            loss_fn = loss_class(config=config, **merged_kwargs)
            
            # 更新统计
            cls._stats.total_created += 1
            cls._stats.creation_counts[name_lower] += 1
            cls._stats.last_created = name_lower
            cls._stats.last_created_time = time.time()
            
            # 缓存
            if use_cache or cls._cache_enabled:
                cls._cache[cache_key] = loss_fn
            
            return loss_fn
            
        except Exception as e:
            logger.error(f"Failed to create loss {name}: {e}")
            cls._stats.failed_creations += 1
            return None
    
    @classmethod
    def create_batch(
        cls,
        loss_specs: List[Dict[str, Any]]
    ) -> Dict[str, Optional[BaseLoss]]:
        """
        批量创建损失函数
        
        Args:
            loss_specs: 损失函数规格列表
            
        Returns:
            名称到损失函数实例的字典
        """
        results = {}
        
        for spec in loss_specs:
            name = spec.get('name', spec.get('type', ''))
            params = spec.get('params', {})
            weight = spec.get('weight', 1.0)
            
            config = LossConfig(weight=weight)
            loss_fn = cls.create(name, config=config, **params)
            
            results[name] = loss_fn
        
        return results
    
    @classmethod
    def list_available(cls) -> List[str]:
        """列出所有可用的损失函数"""
        available = set(LOSS_MAPPING.keys())
        available.update(cls._custom_losses.keys())
        available.update(LossRegistry.list_registered())
        return sorted(list(available))
    
    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """
        按类别列出损失函数
        
        Args:
            category: 类别名称
            
        Returns:
            损失函数名称列表
        """
        if category in CATEGORY_MAPPING:
            return CATEGORY_MAPPING[category].copy()
        return []
    
    @classmethod
    def list_by_task(cls, task: str) -> List[str]:
        """
        按任务类型列出损失函数
        
        Args:
            task: 任务类型
            
        Returns:
            损失函数名称列表
        """
        if task in TASK_MAPPING:
            return TASK_MAPPING[task].copy()
        return []
    
    @classmethod
    def get_stats(cls) -> FactoryStats:
        """获取工厂统计"""
        return cls._stats
    
    @classmethod
    def reset_stats(cls):
        """重置统计"""
        cls._stats = FactoryStats()
    
    @classmethod
    def create_from_config(
        cls, 
        loss_config: Dict[str, Any]
    ) -> Optional[BaseLoss]:
        """
        从配置字典创建损失函数
        
        Args:
            loss_config: 配置字典 {
                'type': 'cross_entropy',
                'weight': 1.0,
                'params': {...}
            }
        """
        loss_type = loss_config.get('type', 'cross_entropy')
        weight = loss_config.get('weight', 1.0)
        params = loss_config.get('params', {})
        
        config = LossConfig(weight=weight)
        
        return cls.create(loss_type, config=config, **params)
    
    @classmethod
    def save_config(cls, loss_fn: BaseLoss, path: str) -> bool:
        """
        保存损失函数配置到文件
        
        Args:
            loss_fn: 损失函数实例
            path: 文件路径
            
        Returns:
            是否成功保存
        """
        try:
            config = {
                'type': loss_fn.__class__.__name__,
                'config': loss_fn.config.to_dict() if hasattr(loss_fn.config, 'to_dict') else {},
            }
            
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save loss config: {e}")
            return False
    
    @classmethod
    def load_config(cls, path: str) -> Optional[BaseLoss]:
        """
        从文件加载损失函数配置
        
        Args:
            path: 文件路径
            
        Returns:
            损失函数实例或None
        """
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            
            return cls.create_from_config(config)
        except Exception as e:
            logger.error(f"Failed to load loss config: {e}")
            return None
    
    @classmethod
    def recommend(
        cls,
        task: str,
        class_imbalance: bool = False,
        noise_level: float = 0.0,
        num_classes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        推荐损失函数
        
        Args:
            task: 任务类型
            class_imbalance: 是否有类别不平衡
            noise_level: 噪声水平
            num_classes: 类别数
            
        Returns:
            推荐配置
        """
        return recommend_loss_function(task, class_imbalance, noise_level, num_classes)
    
    @classmethod
    def print_info(cls, name: str) -> None:
        """
        打印损失函数信息
        
        Args:
            name: 损失函数名称
        """
        metadata = cls.get_metadata(name.lower())
        
        print("\n" + "="*60)
        print(f"Loss Function: {name}")
        print("="*60)
        
        if metadata:
            print(f"\nCategory: {metadata.category}")
            print(f"Task type: {metadata.task_type}")
            print(f"Description: {metadata.description}")
            
            if metadata.default_params:
                print(f"\nDefault parameters:")
                for k, v in metadata.default_params.items():
                    print(f"  {k}: {v}")
            
            if metadata.optional_params:
                print(f"\nOptional parameters: {metadata.optional_params}")
            
            if metadata.reference:
                print(f"\nReference: {metadata.reference}")
        else:
            loss_class = cls.get_class(name.lower())
            if loss_class:
                print(f"\nClass: {loss_class.__name__}")
                if loss_class.__doc__:
                    print(f"Documentation: {loss_class.__doc__[:200]}...")
            else:
                print("\nLoss function not found.")
        
        print("="*60)
    
    @classmethod
    def print_stats(cls) -> None:
        """打印工厂统计"""
        stats = cls._stats
        
        print("\n" + "="*60)
        print("Loss Factory Statistics")
        print("="*60)
        
        print(f"\nTotal created: {stats.total_created}")
        print(f"Failed creations: {stats.failed_creations}")
        
        if stats.creation_counts:
            print(f"\nCreation counts:")
            for name, count in sorted(stats.creation_counts.items(), key=lambda x: -x[1])[:10]:
                print(f"  {name}: {count}")
        
        if stats.last_created:
            print(f"\nLast created: {stats.last_created}")
        
        print(f"\nCache enabled: {cls._cache_enabled}")
        print(f"Cached items: {len(cls._cache)}")
        
        print("="*60)


# ==================== 便捷函数 ====================

def create_loss(
    name: str, 
    config: Optional[LossConfig] = None,
    **kwargs
) -> Optional[BaseLoss]:
    """
    创建损失函数
    
    Args:
        name: 损失函数名称
        config: 配置
        **kwargs: 额外参数
        
    Returns:
        损失函数实例
        
    Example:
        >>> loss = create_loss('cross_entropy', num_classes=10)
        >>> loss = create_loss('focal', alpha=0.25, gamma=2.0)
    """
    return LossFactory.create(name, config, **kwargs)


def create_composite_loss(
    losses: List[Union[str, Tuple[str, float], Tuple[str, Dict[str, Any], float]]]
) -> CompositeLoss:
    """
    创建复合损失
    
    Args:
        losses: 损失列表，支持多种格式：
            - ['cross_entropy', 'focal']  # 默认权重1.0
            - [('cross_entropy', 1.0), ('focal', 0.5)]
            - [('cross_entropy', {'num_classes': 10}, 1.0)]
            
    Returns:
        CompositeLoss实例
        
    Example:
        >>> composite = create_composite_loss([
        ...     ('cross_entropy', 1.0),
        ...     ('focal', {'alpha': 0.25}, 0.5),
        ...     'mse'
        ... ])
    """
    loss_tuples = []
    
    for item in losses:
        if isinstance(item, str):
            # 只有名称
            name = item
            weight = 1.0
            kwargs = {}
        elif isinstance(item, tuple):
            if len(item) == 2:
                if isinstance(item[1], dict):
                    # (name, kwargs)
                    name, kwargs = item
                    weight = 1.0
                else:
                    # (name, weight)
                    name, weight = item
                    kwargs = {}
            elif len(item) == 3:
                # (name, kwargs, weight)
                name, kwargs, weight = item
            else:
                continue
        else:
            continue
        
        loss_fn = create_loss(name, **kwargs)
        if loss_fn:
            loss_tuples.append((name, loss_fn, weight))
    
    return CompositeLoss(loss_tuples)


def create_distillation_loss(
    soft_weight: float = 1.0,
    feature_weight: float = 0.5,
    attention_weight: float = 0.5,
    relational_weight: float = 0.0,
    temperature: float = 4.0,
    **kwargs
) -> CombinedDistillationLoss:
    """
    创建蒸馏损失
    
    Args:
        soft_weight: 软标签损失权重
        feature_weight: 特征蒸馏权重
        attention_weight: 注意力蒸馏权重
        relational_weight: 关系蒸馏权重
        temperature: 温度参数
        
    Returns:
        CombinedDistillationLoss实例
        
    Example:
        >>> kd_loss = create_distillation_loss(
        ...     soft_weight=1.0,
        ...     feature_weight=0.5,
        ...     temperature=4.0
        ... )
    """
    return CombinedDistillationLoss(
        soft_loss_weight=soft_weight,
        feature_loss_weight=feature_weight,
        attention_loss_weight=attention_weight,
        relational_loss_weight=relational_weight,
        temperature=temperature,
        **kwargs
    )


def create_multitask_loss(
    tasks: Dict[str, Union[str, BaseLoss]],
    task_weights: Optional[Dict[str, float]] = None,
    uncertainty_weighted: bool = False
) -> Union[MultiTaskLoss, UncertaintyWeightedLoss]:
    """
    创建多任务损失
    
    Args:
        tasks: 任务字典 {task_name: loss_name或loss_fn}
        task_weights: 任务权重
        uncertainty_weighted: 是否使用不确定性加权
        
    Returns:
        多任务损失实例
        
    Example:
        >>> mtl_loss = create_multitask_loss({
        ...     'classification': 'cross_entropy',
        ...     'regression': 'mse'
        ... })
    """
    # 转换字符串为损失函数
    task_losses = {}
    for name, loss in tasks.items():
        if isinstance(loss, str):
            task_losses[name] = create_loss(loss)
        else:
            task_losses[name] = loss
    
    if uncertainty_weighted:
        return UncertaintyWeightedLoss(task_losses)
    else:
        return MultiTaskLoss(task_losses, task_weights)


def create_regularization_losses(
    l1_weight: float = 0.0,
    l2_weight: float = 1e-4,
    entropy_weight: float = 0.0,
    spectral_weight: float = 0.0
) -> Optional[BaseLoss]:
    """
    创建正则化损失
    
    Args:
        l1_weight: L1正则化权重
        l2_weight: L2正则化权重
        entropy_weight: 熵正则化权重
        spectral_weight: 谱正则化权重
        
    Returns:
        正则化损失实例
        
    Example:
        >>> reg_loss = create_regularization_losses(l1_weight=1e-4, l2_weight=1e-3)
    """
    if l1_weight > 0 or l2_weight > 0 or entropy_weight > 0 or spectral_weight > 0:
        return MixedRegularization(
            l1_weight=l1_weight,
            l2_weight=l2_weight,
            entropy_weight=entropy_weight,
            spectral_weight=spectral_weight
        )
    return None


def get_loss_registry() -> LossRegistry:
    """获取损失函数注册表"""
    return LossRegistry


def get_loss_metadata(name: str) -> Optional[LossMetadata]:
    """获取损失函数元数据"""
    return LossFactory.get_metadata(name)


def list_losses_by_category(category: str) -> List[str]:
    """按类别列出损失函数"""
    return LossFactory.list_by_category(category)


def list_losses_by_task(task: str) -> List[str]:
    """按任务列出损失函数"""
    return LossFactory.list_by_task(task)


def print_loss_info(name: str) -> None:
    """打印损失函数信息"""
    LossFactory.print_info(name)


def print_all_losses() -> None:
    """打印所有可用损失函数"""
    losses = LossFactory.list_available()
    
    print("\n" + "="*80)
    print("Available Loss Functions")
    print("="*80)
    
    # 按类别分组
    for category in ['supervised', 'distillation', 'contrastive', 'regularization']:
        category_losses = LossFactory.list_by_category(category)
        if category_losses:
            print(f"\n{category.upper()}:")
            for name in category_losses:
                metadata = LossFactory.get_metadata(name)
                desc = metadata.description[:50] + "..." if metadata and len(metadata.description) > 50 else (metadata.description if metadata else "")
                print(f"  {name:<20} {desc}")
    
    # 其他（未分类的）
    all_categorized = set()
    for cat_losses in CATEGORY_MAPPING.values():
        all_categorized.update(cat_losses)
    
    other_losses = [l for l in losses if l not in all_categorized and l not in ['ce', 'l2', 'l1', 'smooth_l1', 'kd', 'simclr']]
    if other_losses:
        print(f"\nOTHER:")
        for name in other_losses:
            print(f"  {name}")
    
    print("="*80)


def compare_losses(
    loss_names: List[str],
    test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> Dict[str, Any]:
    """
    对比多个损失函数
    
    Args:
        loss_names: 损失函数名称列表
        test_data: 测试数据（预测、目标）
        
    Returns:
        对比结果
    """
    results = {}
    
    for name in loss_names:
        loss_fn = create_loss(name)
        if loss_fn:
            metadata = LossFactory.get_metadata(name)
            
            results[name] = {
                'class': loss_fn.__class__.__name__,
                'category': metadata.category if metadata else 'unknown',
                'task_type': metadata.task_type if metadata else 'unknown',
            }
            
            # 如果提供了测试数据，计算损失
            if test_data is not None:
                try:
                    predictions, targets = test_data
                    loss_value = loss_fn(predictions, targets)
                    results[name]['loss_value'] = loss_value.item()
                except Exception as e:
                    results[name]['error'] = str(e)
    
    return results


def print_loss_comparison(loss_names: List[str]) -> None:
    """
    打印损失函数对比
    
    Args:
        loss_names: 损失函数名称列表
    """
    results = compare_losses(loss_names)
    
    print("\n" + "="*80)
    print("Loss Function Comparison")
    print("="*80)
    
    print(f"\n{'Name':<20} {'Class':<25} {'Category':<15} {'Task Type':<20}")
    print("-"*80)
    
    for name, info in results.items():
        print(f"{name:<20} {info['class']:<25} {info['category']:<15} {info['task_type']:<20}")
    
    print("="*80)


def auto_select_loss(
    task: str,
    data_characteristics: Optional[Dict[str, Any]] = None
) -> BaseLoss:
    """
    自动选择损失函数
    
    Args:
        task: 任务类型
        data_characteristics: 数据特征 {
            'class_imbalance': bool,
            'noise_level': float,
            'num_classes': int,
            ...
        }
        
    Returns:
        选择的损失函数
    """
    if data_characteristics is None:
        data_characteristics = {}
    
    class_imbalance = data_characteristics.get('class_imbalance', False)
    noise_level = data_characteristics.get('noise_level', 0.0)
    num_classes = data_characteristics.get('num_classes')
    
    recommendation = LossFactory.recommend(
        task=task,
        class_imbalance=class_imbalance,
        noise_level=noise_level,
        num_classes=num_classes
    )
    
    loss_type = recommendation.get('loss_type', 'cross_entropy')
    params = recommendation.get('params', {})
    
    if num_classes:
        params['num_classes'] = num_classes
    
    return create_loss(loss_type, **params)


# ==================== 预设配置 ====================

class LossPresets:
    """损失函数预设配置"""
    
    # 预设描述
    _descriptions = {
        'classification': 'Standard cross entropy for multi-class classification',
        'classification_imbalanced': 'Focal loss for imbalanced classification',
        'regression': 'MSE loss for regression tasks',
        'regression_robust': 'Huber loss for robust regression',
        'segmentation': 'Combined CE + Dice for segmentation',
        'knowledge_distillation': 'Combined distillation loss',
        'contrastive_learning': 'InfoNCE for self-supervised learning',
        'clip_style': 'CLIP-style contrastive loss',
        'multimodal_alignment': 'Combined CLIP + cross-modal loss',
    }
    
    @staticmethod
    def classification() -> BaseLoss:
        """分类任务预设"""
        return create_loss('cross_entropy')
    
    @staticmethod
    def classification_imbalanced(alpha: float = 0.25, gamma: float = 2.0) -> BaseLoss:
        """类别不平衡分类预设"""
        return create_loss('focal', alpha=alpha, gamma=gamma)
    
    @staticmethod
    def classification_with_smoothing(smoothing: float = 0.1) -> BaseLoss:
        """带标签平滑的分类预设"""
        return create_loss('label_smoothing', smoothing=smoothing)
    
    @staticmethod
    def regression() -> BaseLoss:
        """回归任务预设"""
        return create_loss('mse')
    
    @staticmethod
    def regression_robust(delta: float = 1.0) -> BaseLoss:
        """鲁棒回归预设"""
        return create_loss('huber', delta=delta)
    
    @staticmethod
    def regression_mae() -> BaseLoss:
        """MAE回归预设"""
        return create_loss('mae')
    
    @staticmethod
    def segmentation() -> CompositeLoss:
        """分割任务预设"""
        return create_composite_loss([
            ('cross_entropy', 1.0),
            ('dice', 1.0)
        ])
    
    @staticmethod
    def segmentation_dice_only() -> BaseLoss:
        """仅Dice的分割预设"""
        return create_loss('dice')
    
    @staticmethod
    def segmentation_iou() -> BaseLoss:
        """IoU分割预设"""
        return create_loss('iou')
    
    @staticmethod
    def knowledge_distillation(temperature: float = 4.0) -> CombinedDistillationLoss:
        """知识蒸馏预设"""
        return create_distillation_loss(
            soft_weight=1.0,
            feature_weight=0.5,
            attention_weight=0.5,
            temperature=temperature
        )
    
    @staticmethod
    def knowledge_distillation_soft_only(temperature: float = 4.0) -> BaseLoss:
        """仅软标签蒸馏预设"""
        return create_loss('soft_label', temperature=temperature)
    
    @staticmethod
    def knowledge_distillation_feature() -> BaseLoss:
        """特征蒸馏预设"""
        return create_loss('feature_kd')
    
    @staticmethod
    def contrastive_learning(temperature: float = 0.07) -> BaseLoss:
        """对比学习预设"""
        return create_loss('infonce', temperature=temperature)
    
    @staticmethod
    def contrastive_simclr(temperature: float = 0.5) -> BaseLoss:
        """SimCLR风格对比学习预设"""
        return create_loss('nt_xent', temperature=temperature)
    
    @staticmethod
    def metric_learning(margin: float = 1.0) -> BaseLoss:
        """度量学习预设"""
        return create_loss('triplet', margin=margin)
    
    @staticmethod
    def clip_style(temperature: float = 0.07) -> BaseLoss:
        """CLIP风格预设"""
        return create_loss('clip', temperature=temperature, learnable_temperature=True)
    
    @staticmethod
    def multimodal_alignment() -> CompositeLoss:
        """多模态对齐预设"""
        return create_composite_loss([
            ('clip', {'temperature': 0.07}, 1.0),
            ('cross_modal', {'bidirectional': True}, 0.5)
        ])
    
    @staticmethod
    def regularization_l1(lambda_: float = 1e-4) -> BaseLoss:
        """L1正则化预设"""
        return create_loss('l1_reg', lambda_=lambda_)
    
    @staticmethod
    def regularization_l2(lambda_: float = 1e-4) -> BaseLoss:
        """L2正则化预设"""
        return create_loss('l2_reg', lambda_=lambda_)
    
    @staticmethod
    def regularization_mixed(l1_weight: float = 1e-5, l2_weight: float = 1e-4) -> BaseLoss:
        """混合正则化预设"""
        return create_regularization_losses(l1_weight=l1_weight, l2_weight=l2_weight)
    
    @classmethod
    def get(cls, preset_name: str, **kwargs) -> Optional[BaseLoss]:
        """
        根据名称获取预设
        
        Args:
            preset_name: 预设名称
            **kwargs: 额外参数
            
        Returns:
            损失函数实例
        """
        preset_map = {
            'classification': cls.classification,
            'classification_imbalanced': cls.classification_imbalanced,
            'classification_smoothing': cls.classification_with_smoothing,
            'regression': cls.regression,
            'regression_robust': cls.regression_robust,
            'regression_mae': cls.regression_mae,
            'segmentation': cls.segmentation,
            'segmentation_dice': cls.segmentation_dice_only,
            'segmentation_iou': cls.segmentation_iou,
            'knowledge_distillation': cls.knowledge_distillation,
            'kd_soft_only': cls.knowledge_distillation_soft_only,
            'kd_feature': cls.knowledge_distillation_feature,
            'contrastive': cls.contrastive_learning,
            'simclr': cls.contrastive_simclr,
            'metric_learning': cls.metric_learning,
            'clip': cls.clip_style,
            'multimodal': cls.multimodal_alignment,
            'reg_l1': cls.regularization_l1,
            'reg_l2': cls.regularization_l2,
            'reg_mixed': cls.regularization_mixed,
        }
        
        if preset_name in preset_map:
            return preset_map[preset_name](**kwargs)
        
        return None
    
    @classmethod
    def list_presets(cls) -> List[str]:
        """列出所有预设"""
        return [
            'classification', 'classification_imbalanced', 'classification_smoothing',
            'regression', 'regression_robust', 'regression_mae',
            'segmentation', 'segmentation_dice', 'segmentation_iou',
            'knowledge_distillation', 'kd_soft_only', 'kd_feature',
            'contrastive', 'simclr', 'metric_learning',
            'clip', 'multimodal',
            'reg_l1', 'reg_l2', 'reg_mixed'
        ]
    
    @classmethod
    def print_presets(cls) -> None:
        """打印所有预设"""
        print("\n" + "="*80)
        print("Loss Function Presets")
        print("="*80)
        
        for preset in cls.list_presets():
            desc = cls._descriptions.get(preset, "")
            print(f"  {preset:<25} {desc}")
        
        print("\n" + "="*80)


# ==================== 配置构建器 ====================

class LossConfigBuilder:
    """
    损失函数配置构建器
    
    使用链式调用构建损失配置。
    """
    
    def __init__(self, loss_type: str = 'cross_entropy'):
        self._config = {
            'type': loss_type,
            'weight': 1.0,
            'params': {}
        }
    
    def weight(self, weight: float) -> 'LossConfigBuilder':
        """设置权重"""
        self._config['weight'] = weight
        return self
    
    def param(self, key: str, value: Any) -> 'LossConfigBuilder':
        """设置参数"""
        self._config['params'][key] = value
        return self
    
    def params(self, **kwargs) -> 'LossConfigBuilder':
        """批量设置参数"""
        self._config['params'].update(kwargs)
        return self
    
    def build(self) -> Optional[BaseLoss]:
        """构建损失函数"""
        return LossFactory.create_from_config(self._config)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()


def loss_builder(loss_type: str = 'cross_entropy') -> LossConfigBuilder:
    """
    创建损失配置构建器
    
    Example:
        >>> loss = loss_builder('focal').weight(1.0).param('alpha', 0.25).param('gamma', 2.0).build()
    """
    return LossConfigBuilder(loss_type)



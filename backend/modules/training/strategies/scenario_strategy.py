# -*- coding: utf-8 -*-
"""
场景化训练策略

针对特定业务场景的训练策略，支持场景权重和场景路由。
基于技术方案中的场景化训练策略设计。

架构调用层次：
├── scenario_strategy.py (本模块)
│   └── ScenarioStrategy: 调用策略层 + 损失层
│       └── 使用 base_strategy.py: StrategyMonitor, StrategyProfiler, StrategyValidator
│       └── 使用 backend/lib/losses: FocalLoss, CrossEntropyLoss, CompositeLoss, MultiTaskLoss
│   └── IndustryScenarioStrategy: 调用全部六层架构
│       └── 继承 ProductionTrainingStrategy
│       └── 调用 backend/lib/* (六层架构)
│       └── 使用 production_base.py: ProductionHealthStatus, WrapperStats, get_layer_details
└── 被 scenarios/industry_scenario.py 调用

生产级特性：
- 完整的场景监控和诊断能力
- 场景特定的损失组合
- 健康检查和自动恢复
- 多任务学习支持
"""

import logging
import time
from contextlib import nullcontext
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_strategy import (
    TrainingStrategy, StrategyContext, StrategyResult, 
    TrainingPhase, StrategyType,
    # 监控和诊断组件
    StrategyMonitor, StrategyProfiler, StrategyValidator, StrategyMetrics,
)
from .production_base import (
    ProductionTrainingStrategy,
    ProductionStrategyConfig,
    ProductionTrainingContext,
    # 新增数据类
    ProductionHealthStatus, WrapperStats,
    # 新增函数
    get_available_layers, get_layer_details,
)

logger = logging.getLogger(__name__)


# ==================== 底层损失层导入 ====================

from backend.lib.losses import (
    # 监督损失
    CrossEntropyLoss, FocalLoss, LabelSmoothingLoss,
    MSELoss, MAELoss, HuberLoss,
    # 复合损失
    CompositeLoss, MultiTaskLoss,
    # 正则化损失
    ConsistencyRegularization,
    # 工厂函数
    create_loss, create_composite_loss,
    # 监控组件
    LossMonitor, LossStats,
)

# 定义可用的损失函数类列表
LOSSES_AVAILABLE = [
    CrossEntropyLoss, FocalLoss, LabelSmoothingLoss,
    MSELoss, MAELoss, HuberLoss,
    CompositeLoss, MultiTaskLoss,
    ConsistencyRegularization,
]


# ==================== 新增数据类 ====================

@dataclass
class ScenarioHealthStatus:
    """场景健康状态"""
    is_healthy: bool = True
    scenario_type: str = ""
    loss_stable: bool = True
    metrics_improving: bool = True
    last_check_time: float = 0.0
    consecutive_failures: int = 0
    issues: List[str] = field(default_factory=list)
    
    def add_issue(self, issue: str) -> None:
        """添加问题"""
        self.issues.append(issue)
        self.is_healthy = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'is_healthy': self.is_healthy,
            'scenario_type': self.scenario_type,
            'loss_stable': self.loss_stable,
            'metrics_improving': self.metrics_improving,
            'last_check_time': self.last_check_time,
            'consecutive_failures': self.consecutive_failures,
            'issues': self.issues.copy(),
        }


@dataclass
class ScenarioStats:
    """场景统计"""
    total_samples: int = 0
    total_steps: int = 0
    avg_loss: float = 0.0
    avg_scene_loss: float = 0.0
    best_metric: float = float('inf')
    loss_history: List[float] = field(default_factory=list)
    metric_history: List[float] = field(default_factory=list)
    
    def update(self, loss: float, metric: Optional[float] = None) -> None:
        """更新统计"""
        self.total_steps += 1
        self.loss_history.append(loss)
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]
        self.avg_loss = sum(self.loss_history) / len(self.loss_history)
        
        if metric is not None:
            self.metric_history.append(metric)
            if len(self.metric_history) > 1000:
                self.metric_history = self.metric_history[-1000:]
            if metric < self.best_metric:
                self.best_metric = metric
    
    def get_loss_trend(self, window: int = 100) -> str:
        """获取损失趋势"""
        if len(self.loss_history) < window * 2:
            return "insufficient_data"
        
        recent = sum(self.loss_history[-window:]) / window
        previous = sum(self.loss_history[-window*2:-window]) / window
        
        if recent < previous * 0.95:
            return "improving"
        elif recent > previous * 1.05:
            return "degrading"
        return "stable"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_samples': self.total_samples,
            'total_steps': self.total_steps,
            'avg_loss': self.avg_loss,
            'avg_scene_loss': self.avg_scene_loss,
            'best_metric': self.best_metric,
            'loss_trend': self.get_loss_trend(),
        }


class ScenarioType(Enum):
    """业务场景类型"""
    # 通用场景
    BASIC_MODEL = "basic_model"
    SCHEDULED_TASK = "scheduled_task"
    ADVANCED_MODEL = "advanced_model"
    RESEARCH_EXPERIMENT = "research_experiment"
    PRODUCTION_FINETUNE = "production_finetune"
    
    # 行业场景
    EQUIPMENT_FAULT_PREDICTION = "equipment_fault_prediction"  # 设备故障预测
    PROCESS_OPTIMIZATION = "process_optimization"              # 工艺参数优化
    QUALITY_DEFECT_DETECTION = "quality_defect_detection"      # 质量缺陷识别
    ENERGY_PREDICTION = "energy_prediction"                    # 能耗预测
    ANOMALY_DETECTION = "anomaly_detection"                    # 异常检测
    
    # 金融场景
    RISK_ASSESSMENT = "risk_assessment"                        # 风险评估
    FRAUD_DETECTION = "fraud_detection"                        # 欺诈检测
    
    # 医疗场景
    DISEASE_DIAGNOSIS = "disease_diagnosis"                    # 疾病诊断
    MEDICAL_IMAGE_ANALYSIS = "medical_image_analysis"          # 医学影像分析
    
    @classmethod
    def from_string(cls, value: str) -> 'ScenarioType':
        """从字符串创建"""
        value = value.lower().strip()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown scenario type: {value}")
    
    @property
    def is_classification(self) -> bool:
        """是否为分类场景"""
        return self in [
            ScenarioType.EQUIPMENT_FAULT_PREDICTION,
            ScenarioType.QUALITY_DEFECT_DETECTION,
            ScenarioType.FRAUD_DETECTION,
            ScenarioType.DISEASE_DIAGNOSIS,
            ScenarioType.RISK_ASSESSMENT,
        ]
    
    @property
    def is_regression(self) -> bool:
        """是否为回归场景"""
        return self in [
            ScenarioType.PROCESS_OPTIMIZATION,
            ScenarioType.ENERGY_PREDICTION,
        ]
    
    @property
    def is_time_series(self) -> bool:
        """是否为时序场景"""
        return self in [
            ScenarioType.EQUIPMENT_FAULT_PREDICTION,
            ScenarioType.ANOMALY_DETECTION,
            ScenarioType.ENERGY_PREDICTION,
        ]
    
    @property
    def is_imbalanced(self) -> bool:
        """是否为类别不平衡场景"""
        return self in [
            ScenarioType.QUALITY_DEFECT_DETECTION,
            ScenarioType.FRAUD_DETECTION,
            ScenarioType.ANOMALY_DETECTION,
        ]
    
    @property
    def recommended_loss(self) -> str:
        """推荐的损失函数"""
        if self.is_imbalanced:
            return "focal"
        elif self.is_regression:
            return "huber"
        return "cross_entropy"
    
    def get_description(self) -> str:
        """获取场景描述"""
        descriptions = {
            ScenarioType.EQUIPMENT_FAULT_PREDICTION: "设备故障预测：基于时序数据预测设备故障",
            ScenarioType.PROCESS_OPTIMIZATION: "工艺参数优化：优化生产工艺参数",
            ScenarioType.QUALITY_DEFECT_DETECTION: "质量缺陷识别：识别产品质量缺陷",
            ScenarioType.ENERGY_PREDICTION: "能耗预测：预测设备/系统能耗",
            ScenarioType.ANOMALY_DETECTION: "异常检测：检测数据异常模式",
            ScenarioType.FRAUD_DETECTION: "欺诈检测：识别欺诈行为",
            ScenarioType.DISEASE_DIAGNOSIS: "疾病诊断：辅助疾病诊断",
        }
        return descriptions.get(self, f"场景: {self.value}")


@dataclass
class ScenarioStrategyConfig:
    """场景策略配置"""
    # 场景类型
    scenario_type: ScenarioType = ScenarioType.BASIC_MODEL
    
    # 场景权重（用于多任务学习）
    scene_weight: float = 1.0
    
    # 损失权重
    task_loss_weight: float = 1.0
    scene_specific_loss_weight: float = 0.2
    consistency_loss_weight: float = 0.1
    
    # 场景特定配置
    freeze_backbone: bool = False      # 是否冻结backbone
    use_scene_adapter: bool = True     # 是否使用场景适配器
    adapter_dim: int = 64              # 适配器维度
    
    # 损失函数配置
    loss_type: str = "auto"            # auto, cross_entropy, focal, mse, huber
    focal_gamma: float = 2.0           # FocalLoss gamma
    focal_alpha: Optional[float] = None  # FocalLoss alpha
    label_smoothing: float = 0.0       # 标签平滑
    
    # 数据增强配置
    augmentation_enabled: bool = True
    augmentation_strength: float = 0.5
    
    # 小样本学习配置
    few_shot_enabled: bool = False
    support_set_size: int = 5
    
    # 早停配置（场景级）
    early_stopping_metric: str = "loss"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    
    # 监控配置
    enable_monitoring: bool = True
    enable_profiling: bool = False
    health_check_interval: int = 100
    log_interval: int = 10
    
    # 场景特定参数
    scene_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> None:
        """验证配置"""
        if self.scene_weight <= 0:
            raise ValueError("scene_weight must be > 0")
        if self.task_loss_weight < 0:
            raise ValueError("task_loss_weight must be >= 0")
        if self.adapter_dim <= 0:
            raise ValueError("adapter_dim must be > 0")
        if self.focal_gamma < 0:
            raise ValueError("focal_gamma must be >= 0")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'scenario_type': self.scenario_type.value,
            'scene_weight': self.scene_weight,
            'task_loss_weight': self.task_loss_weight,
            'scene_specific_loss_weight': self.scene_specific_loss_weight,
            'freeze_backbone': self.freeze_backbone,
            'use_scene_adapter': self.use_scene_adapter,
            'adapter_dim': self.adapter_dim,
            'loss_type': self.loss_type,
            'augmentation_enabled': self.augmentation_enabled,
            'enable_monitoring': self.enable_monitoring,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioStrategyConfig':
        """从字典创建"""
        if 'scenario_type' in data and isinstance(data['scenario_type'], str):
            data['scenario_type'] = ScenarioType.from_string(data['scenario_type'])
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def get_loss_type(self) -> str:
        """获取损失函数类型"""
        if self.loss_type == "auto":
            return self.scenario_type.recommended_loss
        return self.loss_type
    
    def summary(self) -> str:
        """获取配置摘要"""
        return (
            f"ScenarioConfig(type={self.scenario_type.value}, "
            f"weight={self.scene_weight}, loss={self.get_loss_type()})"
        )


class ScenarioRouter:
    """
    场景路由器
    
    根据样本特征将数据路由到对应的场景处理器。
    支持规则路由、特征路由和混合路由。
    """
    
    def __init__(self, routing_rules: Optional[Dict[str, Callable]] = None):
        """
        初始化场景路由器
        
        Args:
            routing_rules: 路由规则字典 {场景名: 判断函数}
        """
        self.routing_rules = routing_rules or {}
        self.default_scenario = ScenarioType.BASIC_MODEL
        
        # 路由统计
        self._route_counts: Dict[str, int] = {}
        self._total_routes: int = 0
        self._failed_routes: int = 0
    
    def route(self, sample: Dict[str, Any]) -> ScenarioType:
        """
        路由样本到对应场景
        
        Args:
            sample: 输入样本
        
        Returns:
            场景类型
        """
        self._total_routes += 1
        
        # 如果样本中已经包含场景信息
        if 'scenario' in sample:
            try:
                scenario = ScenarioType(sample['scenario'])
                self._record_route(scenario)
                return scenario
            except ValueError:
                pass
        
        # 应用路由规则
        for scenario_name, rule_func in self.routing_rules.items():
            try:
                if rule_func(sample):
                    scenario = ScenarioType(scenario_name)
                    self._record_route(scenario)
                    return scenario
            except Exception as e:
                logger.warning(f"Routing rule failed for {scenario_name}: {e}")
                self._failed_routes += 1
        
        self._record_route(self.default_scenario)
        return self.default_scenario
    
    def _record_route(self, scenario: ScenarioType) -> None:
        """记录路由统计"""
        key = scenario.value
        self._route_counts[key] = self._route_counts.get(key, 0) + 1
    
    def add_rule(self, scenario: ScenarioType, rule_func: Callable) -> None:
        """添加路由规则"""
        self.routing_rules[scenario.value] = rule_func
    
    def remove_rule(self, scenario: ScenarioType) -> None:
        """移除路由规则"""
        self.routing_rules.pop(scenario.value, None)
    
    def set_default(self, scenario: ScenarioType) -> None:
        """设置默认场景"""
        self.default_scenario = scenario
    
    def batch_route(self, batch: Dict[str, Any]) -> Dict[ScenarioType, List[int]]:
        """
        批量路由
        
        返回每个场景对应的样本索引。
        """
        scenario_indices: Dict[ScenarioType, List[int]] = {}
        
        batch_size = self._get_batch_size(batch)
        for i in range(batch_size):
            sample = self._get_sample_from_batch(batch, i)
            scenario = self.route(sample)
            
            if scenario not in scenario_indices:
                scenario_indices[scenario] = []
            scenario_indices[scenario].append(i)
        
        return scenario_indices
    
    def _get_batch_size(self, batch: Dict[str, Any]) -> int:
        """获取批次大小"""
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                return value.shape[0]
        return 1
    
    def _get_sample_from_batch(self, batch: Dict[str, Any], idx: int) -> Dict[str, Any]:
        """从批次中获取单个样本"""
        sample = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and len(value.shape) > 0:
                sample[key] = value[idx]
            elif isinstance(value, list) and len(value) > idx:
                sample[key] = value[idx]
            else:
                sample[key] = value
        return sample
    
    def get_route_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        return {
            'total_routes': self._total_routes,
            'failed_routes': self._failed_routes,
            'route_distribution': self._route_counts.copy(),
            'num_rules': len(self.routing_rules),
            'default_scenario': self.default_scenario.value,
        }
    
    def reset_stats(self) -> None:
        """重置统计"""
        self._route_counts.clear()
        self._total_routes = 0
        self._failed_routes = 0


class ScenarioStrategy(TrainingStrategy):
    """
    场景化训练策略
    
    整合策略层 + 损失层能力：
    - 策略层：场景路由、适配器、权重调整
    - base_strategy.py: StrategyMonitor, StrategyProfiler, StrategyValidator, StrategyMetrics
    - backend/lib/losses: FocalLoss, CrossEntropyLoss, LabelSmoothingLoss, MSELoss, MAELoss, 
                         HuberLoss, CompositeLoss, MultiTaskLoss, ConsistencyRegularization,
                         create_loss, create_composite_loss, LossMonitor
    
    实现场景特定的训练逻辑：
    - 场景权重调整
    - 场景适配器
    - 场景特定损失（使用 CompositeLoss/MultiTaskLoss 组合）
    - 小样本学习支持
    - 健康检查和自动恢复
    """
    
    # 策略类型
    STRATEGY_TYPE = StrategyType.PRODUCTION
    
    def __init__(self, config: Optional[ScenarioStrategyConfig] = None):
        super().__init__(name="scenario", priority=30)
        self.config = config or ScenarioStrategyConfig()
        
        # 验证配置
        try:
            self.config.validate()
        except ValueError as e:
            logger.warning(f"Config validation warning: {e}")
        
        # 场景适配器
        self.scene_adapter: Optional[nn.Module] = None
        
        # 场景路由器
        self.router = ScenarioRouter()
        
        # 底层损失模块（使用 backend/lib/losses）
        self._lib_task_loss: Optional[nn.Module] = None
        self._lib_focal_loss: Optional[nn.Module] = None
        self._lib_label_smoothing_loss: Optional[nn.Module] = None
        self._lib_mse_loss: Optional[nn.Module] = None
        self._lib_mae_loss: Optional[nn.Module] = None
        self._lib_huber_loss: Optional[nn.Module] = None
        self._lib_consistency_loss: Optional[nn.Module] = None
        self._lib_composite_loss: Optional[nn.Module] = None
        self._lib_multitask_loss: Optional[nn.Module] = None
        self._lib_loss_monitor: Optional['LossMonitor'] = None
        
        # 基础策略组件 (base_strategy.py)
        self._strategy_monitor: Optional[StrategyMonitor] = None
        self._strategy_profiler: Optional[StrategyProfiler] = None
        self._strategy_validator: Optional[StrategyValidator] = None
        self._strategy_metrics: Optional[StrategyMetrics] = None
        
        # 场景状态
        self._health_status = ScenarioHealthStatus()
        self._scenario_stats = ScenarioStats()
        self._current_phase: TrainingPhase = TrainingPhase.WARMUP
    
    def setup(self, context: StrategyContext) -> None:
        """
        初始化场景组件
        
        使用: StrategyMonitor, StrategyProfiler, StrategyValidator, StrategyMetrics (base_strategy.py)
              LossMonitor (backend/lib/losses)
        """
        super().setup(context)
        
        # 初始化基础策略组件 (base_strategy.py)
        self._init_base_strategy_components()
        
        # 冻结backbone（如果配置）
        if self.config.freeze_backbone and context.model is not None:
            self._freeze_backbone(context.model)
        
        # 初始化场景适配器
        if self.config.use_scene_adapter:
            self._init_scene_adapter(context)
        
        # 初始化底层损失层 (backend/lib/losses)
        self._setup_losses_layer(context)
        
        # 初始健康检查
        self._check_health()
        
        logger.info(f"ScenarioStrategy setup: type={self.config.scenario_type.value}, "
                   f"weight={self.config.scene_weight},"
                   f"monitor={self._strategy_monitor is not None}")
    
    def _init_base_strategy_components(self) -> None:
        """
        初始化基础策略组件
        
        使用 base_strategy.py: StrategyMonitor, StrategyProfiler, StrategyValidator, StrategyMetrics
        """
        # 初始化策略监控器
        if self.config.enable_monitoring:
            try:
                self._strategy_monitor = StrategyMonitor(history_size=10000)
            except Exception as e:
                logger.warning(f"Failed to init StrategyMonitor: {e}")
        
        # 初始化性能分析器
        if self.config.enable_profiling:
            try:
                self._strategy_profiler = StrategyProfiler()
            except Exception as e:
                logger.warning(f"Failed to init StrategyProfiler: {e}")
        
        # 初始化验证器
        try:
            self._strategy_validator = StrategyValidator()
            self._add_scenario_validation_rules()
        except Exception as e:
            logger.warning(f"Failed to init StrategyValidator: {e}")
        
        # 初始化指标跟踪
        try:
            self._strategy_metrics = StrategyMetrics()
        except Exception as e:
            logger.warning(f"Failed to init StrategyMetrics: {e}")
        
        logger.debug("Base strategy components initialized")
    
    def _add_scenario_validation_rules(self) -> None:
        """添加场景验证规则"""
        if self._strategy_validator is None:
            return
        
        if hasattr(self._strategy_validator, 'add_check'):
            # 检查场景健康状态
            def check_scenario_health(result: StrategyResult) -> Tuple[bool, str]:
                if not self._health_status.is_healthy:
                    return False, f"Scenario unhealthy: {self._health_status.issues}"
                return True, ""
            
            self._strategy_validator.add_check(check_scenario_health)
    
    def _setup_losses_layer(self, context: StrategyContext) -> None:
        """
        初始化底层损失层
        
        使用 backend/lib/losses: CrossEntropyLoss, FocalLoss, LabelSmoothingLoss,
                                MSELoss, MAELoss, HuberLoss, CompositeLoss, MultiTaskLoss,
                                ConsistencyRegularization, create_loss, create_composite_loss, LossMonitor
        """
        try:
            loss_type = self.config.get_loss_type()
            
            # 1. 初始化各种损失函数
            # 分类损失
            if CrossEntropyLoss is not None:
                self._lib_task_loss = CrossEntropyLoss().to(context.device)
            
            if FocalLoss is not None:
                self._lib_focal_loss = FocalLoss(
                    gamma=self.config.focal_gamma,
                    alpha=self.config.focal_alpha
                ).to(context.device)
            
            if LabelSmoothingLoss is not None and self.config.label_smoothing > 0:
                self._lib_label_smoothing_loss = LabelSmoothingLoss(
                    smoothing=self.config.label_smoothing
                ).to(context.device)
            
            # 回归损失
            if MSELoss is not None:
                self._lib_mse_loss = MSELoss().to(context.device)
            
            if MAELoss is not None:
                self._lib_mae_loss = MAELoss().to(context.device)
            
            if HuberLoss is not None:
                self._lib_huber_loss = HuberLoss().to(context.device)
            
            # 2. 一致性正则化（用于时序场景）
            if ConsistencyRegularization is not None:
                if self.config.scenario_type.is_time_series:
                    self._lib_consistency_loss = ConsistencyRegularization().to(context.device)
            
            # 3. 使用 create_loss 工厂函数创建主损失
            if create_loss is not None:
                try:
                    self._lib_task_loss = create_loss(loss_type).to(context.device)
                except Exception as e:
                    logger.debug(f"create_loss failed: {e}")
            
            # 4. 创建复合损失 (CompositeLoss)
            if CompositeLoss is not None and self.config.scenario_type.is_imbalanced:
                try:
                    losses = {'task': self._lib_task_loss or nn.CrossEntropyLoss()}
                    if self._lib_focal_loss is not None:
                        losses['focal'] = self._lib_focal_loss
                    weights = {'task': 0.6, 'focal': 0.4}
                    self._lib_composite_loss = CompositeLoss(losses, weights).to(context.device)
                except Exception as e:
                    logger.debug(f"CompositeLoss failed: {e}")
            
            # 5. 创建多任务损失 (MultiTaskLoss)
            if MultiTaskLoss is not None:
                try:
                    task_losses = {'primary': self._lib_task_loss or nn.CrossEntropyLoss()}
                    if self._lib_consistency_loss is not None:
                        task_losses['consistency'] = self._lib_consistency_loss
                    task_weights = {'primary': 1.0, 'consistency': self.config.consistency_loss_weight}
                    self._lib_multitask_loss = MultiTaskLoss(task_losses, task_weights).to(context.device)
                except Exception as e:
                    logger.debug(f"MultiTaskLoss failed: {e}")
            
            # 6. 使用 create_composite_loss 工厂函数
            if create_composite_loss is not None:
                try:
                    # 创建场景特定的复合损失
                    loss_configs = [
                        ('task', self._lib_task_loss or nn.CrossEntropyLoss(), self.config.task_loss_weight)
                    ]
                    if self._lib_focal_loss is not None and self.config.scenario_type.is_imbalanced:
                        loss_configs.append(('focal', self._lib_focal_loss, 0.2))
                    if self._lib_consistency_loss is not None:
                        loss_configs.append(('consistency', self._lib_consistency_loss, self.config.consistency_loss_weight))
                    
                    self._lib_composite_loss = create_composite_loss(loss_configs)
                except Exception as e:
                    logger.debug(f"create_composite_loss failed: {e}")
            
            # 7. 初始化损失监控器 (LossMonitor)
            if LossMonitor is not None:
                try:
                    self._lib_loss_monitor = LossMonitor(max_history=10000)
                except Exception as e:
                    logger.debug("LossMonitor failed: %s", e)
            
            logger.info(f"Scenario losses initialized: type={loss_type}, "
                       f"focal={self._lib_focal_loss is not None}, "
                       f"composite={self._lib_composite_loss is not None}, "
                       f"multitask={self._lib_multitask_loss is not None}, "
                       f"monitor={self._lib_loss_monitor is not None}")
        except Exception as e:
            logger.warning(f"Failed to initialize scenario losses: {e}")
    
    def _freeze_backbone(self, model: nn.Module) -> None:
        """冻结backbone参数"""
        frozen_count = 0
        for name, param in model.named_parameters():
            # 只冻结backbone层，保留head层
            if 'head' not in name.lower() and 'classifier' not in name.lower():
                param.requires_grad = False
                frozen_count += 1
        
        logger.info(f"Frozen {frozen_count} backbone parameters")
    
    def _init_scene_adapter(self, context: StrategyContext) -> None:
        """初始化场景适配器"""
        # 获取模型维度
        hidden_dim = context.config.get('hidden_dim', 768)
        
        self.scene_adapter = nn.Sequential(
            nn.Linear(hidden_dim, self.config.adapter_dim),
            nn.GELU(),
            nn.Linear(self.config.adapter_dim, hidden_dim)
        ).to(context.device)
        
        logger.info(f"Scene adapter initialized: {hidden_dim} -> {self.config.adapter_dim} -> {hidden_dim}")
    
    def prepare_batch(self, batch: Dict[str, Any], context: StrategyContext) -> Dict[str, Any]:
        """准备场景化批次数据"""
        # 添加场景权重
        batch['scene_weight'] = self.config.scene_weight
        batch['scenario_type'] = self.config.scenario_type.value
        
        # 数据增强（如果启用）
        if self.config.augmentation_enabled and context.model.training:
            batch = self._apply_augmentation(batch, context)
        
        return batch
    
    def _apply_augmentation(
        self, 
        batch: Dict[str, Any], 
        context: StrategyContext
    ) -> Dict[str, Any]:
        """应用场景特定的数据增强"""
        # 基于场景类型应用不同的增强策略
        if self.config.scenario_type == ScenarioType.EQUIPMENT_FAULT_PREDICTION:
            # 时序数据增强：添加噪声
            if 'time_series' in batch:
                noise_level = self.config.augmentation_strength * 0.1
                batch['time_series'] = batch['time_series'] + torch.randn_like(
                    batch['time_series']
                ) * noise_level
        
        elif self.config.scenario_type == ScenarioType.QUALITY_DEFECT_DETECTION:
            # 图像数据增强（简化实现）
            pass
        
        return batch
    
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """
        计算场景化损失
        
        整合底层损失层能力：
        - 使用 backend/lib/losses: CompositeLoss, MultiTaskLoss, FocalLoss, create_loss
        - 使用 base_strategy.py: StrategyMonitor, StrategyValidator, StrategyMetrics
        
        总损失 = scene_weight * (task_loss + scene_specific_loss + consistency_loss)
        """
        metrics = {}
        scene_weight = batch.get('scene_weight', self.config.scene_weight)
        
        # 使用分析器（如果启用）
        profiler_ctx = (self._strategy_profiler.profile('compute_loss') 
                       if self._strategy_profiler is not None else nullcontext())
        
        with profiler_ctx:
            # 1. 任务损失（优先使用 CompositeLoss 或 MultiTaskLoss）
            task_loss = self._compute_task_loss_with_lib(outputs, batch, context)
            metrics['task_loss'] = task_loss.item()
            
            # 2. 场景特定损失
            scene_loss = torch.tensor(0.0, device=context.device)
            
            # 根据场景类型计算特定损失
            if self.config.scenario_type == ScenarioType.EQUIPMENT_FAULT_PREDICTION:
                scene_loss = self._compute_fault_prediction_loss(outputs, batch)
            elif self.config.scenario_type == ScenarioType.QUALITY_DEFECT_DETECTION:
                scene_loss = self._compute_defect_detection_loss(outputs, batch)
            elif self.config.scenario_type == ScenarioType.ANOMALY_DETECTION:
                scene_loss = self._compute_anomaly_detection_loss(outputs, batch)
            elif self.config.scenario_type == ScenarioType.PROCESS_OPTIMIZATION:
                scene_loss = self._compute_optimization_loss(outputs, batch, context)
            elif self.config.scenario_type == ScenarioType.ENERGY_PREDICTION:
                scene_loss = self._compute_energy_prediction_loss(outputs, batch, context)
        
        if scene_loss.item() > 0:
            metrics['scene_loss'] = scene_loss.item()
        
            # 3. 一致性损失（时序场景）
            consistency_loss = torch.tensor(0.0, device=context.device)
            if self.config.scenario_type.is_time_series and self._lib_consistency_loss is not None:
                consistency_loss = self._compute_consistency_loss(outputs, batch)
                if consistency_loss.item() > 0:
                    metrics['consistency_loss'] = consistency_loss.item()
            
            # 4. 计算总损失
        total_loss = scene_weight * (
            self.config.task_loss_weight * task_loss + 
                self.config.scene_specific_loss_weight * scene_loss +
                self.config.consistency_loss_weight * consistency_loss
        )
        
        metrics['total_loss'] = total_loss.item()
        metrics['scene_weight'] = scene_weight
        metrics['scenario_type'] = self.config.scenario_type.value
        
        # 记录到策略监控器
        if self._strategy_monitor is not None:
            try:
                self._strategy_monitor.record_step(
                    context.global_step,
                    total_loss.item(),
                    metrics
                )
            except Exception:
                pass
        
        # 记录到损失监控器
        if self._lib_loss_monitor is not None:
            try:
                self._lib_loss_monitor.record(total_loss.item())
            except Exception:
                pass
        
        # 更新场景统计
        self._scenario_stats.update(total_loss.item())
        
        # 验证结果
        result = StrategyResult(loss=total_loss, metrics=metrics)
        if self._strategy_validator is not None:
            try:
                is_valid = self._strategy_validator.validate(result)
                if not is_valid:
                    logger.warning("Result validation failed")
            except Exception:
                pass
        
        return result
    
    def _compute_task_loss_with_lib(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        context: StrategyContext
    ) -> torch.Tensor:
        """
        使用底层损失层计算任务损失
        
        优先使用 backend/lib/losses: CompositeLoss, MultiTaskLoss, FocalLoss, 
                                    LabelSmoothingLoss, MSELoss, HuberLoss, create_loss
        """
        # 如果 outputs 中已有 loss，直接使用
        if 'loss' in outputs:
            return outputs['loss']
        elif hasattr(outputs, 'loss'):
            return outputs.loss
        
        # 否则使用底层损失模块计算
        logits = outputs.get('logits')
        labels = batch.get('labels')
        
        if logits is None or labels is None:
            raise ValueError("Cannot compute task loss: no logits or labels")
        
        # 1. 优先使用复合损失 (CompositeLoss)
        if self._lib_composite_loss is not None:
            try:
                return self._lib_composite_loss(logits, labels)
            except Exception as e:
                logger.debug(f"CompositeLoss failed: {e}")
        
        # 2. 使用多任务损失 (MultiTaskLoss)
        if self._lib_multitask_loss is not None:
            try:
                return self._lib_multitask_loss(logits, labels)
            except Exception as e:
                logger.debug(f"MultiTaskLoss failed: {e}")
        
        # 3. 根据场景类型选择损失函数
        # 类别不平衡场景使用 FocalLoss
        if self.config.scenario_type.is_imbalanced and self._lib_focal_loss is not None:
            try:
                return self._lib_focal_loss(logits, labels)
            except Exception as e:
                logger.debug(f"FocalLoss failed: {e}")
        
        # 标签平滑
        if self._lib_label_smoothing_loss is not None:
            try:
                return self._lib_label_smoothing_loss(logits, labels)
            except Exception as e:
                logger.debug(f"LabelSmoothingLoss failed: {e}")
        
        # 回归场景使用 HuberLoss 或 MSELoss
        if self.config.scenario_type.is_regression:
            predictions = outputs.get('predictions', logits)
            targets = batch.get('targets', labels)
            
            if self._lib_huber_loss is not None:
                try:
                    return self._lib_huber_loss(predictions.squeeze(), targets.float())
                except Exception as e:
                    logger.debug(f"HuberLoss failed: {e}")
            
            if self._lib_mse_loss is not None:
                try:
                    return self._lib_mse_loss(predictions.squeeze(), targets.float())
                except Exception as e:
                    logger.debug(f"MSELoss failed: {e}")
            
            if self._lib_mae_loss is not None:
                try:
                    return self._lib_mae_loss(predictions.squeeze(), targets.float())
                except Exception as e:
                    logger.debug(f"MAELoss failed: {e}")
        
        # 4. 使用基础任务损失
        if self._lib_task_loss is not None:
            try:
                return self._lib_task_loss(logits, labels)
            except Exception as e:
                logger.debug(f"Task loss failed: {e}")
        
        # 5. 回退到 PyTorch 原生
        if self.config.scenario_type.is_regression:
            return F.mse_loss(logits.squeeze(), labels.float())
            return F.cross_entropy(logits, labels)
    
    def _get_task_loss(self, outputs: Dict[str, Any]) -> torch.Tensor:
        """获取任务损失"""
        if 'loss' in outputs:
            return outputs['loss']
        elif hasattr(outputs, 'loss'):
            return outputs.loss
        else:
            raise ValueError("outputs中没有找到loss")
    
    def _compute_fault_prediction_loss(
        self, 
        outputs: Dict[str, Any], 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        计算故障预测场景的额外损失
        
        使用底层损失层的一致性正则化
        """
        device = outputs.get('logits', torch.tensor(0.0)).device
        
        # 如果有时间敏感性要求，添加时序一致性损失
        if 'predictions' in outputs and 'sequence_length' in batch:
            predictions = outputs['predictions']
            # 时序一致性：相邻时间步的预测不应该剧烈变化
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # 优先使用底层一致性损失
                if self._lib_consistency_loss is not None:
                    try:
                        return self._lib_consistency_loss(
                            predictions[:, :-1], predictions[:, 1:]
                        )
                    except Exception as e:
                        logger.warning(f"Consistency loss failed: {e}")
                
                # 回退到原生实现
                consistency_loss = torch.mean(
                    torch.abs(predictions[:, 1:] - predictions[:, :-1])
                )
                return consistency_loss
        
        return torch.tensor(0.0, device=device)
    
    def _compute_defect_detection_loss(
        self, 
        outputs: Dict[str, Any], 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        计算缺陷检测场景的额外损失
        
        使用底层损失层的 FocalLoss 处理类别不平衡
        """
        device = outputs.get('logits', torch.tensor(0.0)).device
        
        # 添加类别平衡损失（处理类别不平衡问题）
        if 'logits' in outputs and 'labels' in batch:
            logits = outputs['logits']
            labels = batch['labels']
            
            # 优先使用底层 FocalLoss
            if self._lib_focal_loss is not None:
                try:
                    focal_loss = self._lib_focal_loss(logits, labels)
                    return focal_loss * 0.1  # 权重系数
                except Exception as e:
                    logger.warning(f"Focal loss failed: {e}")
            
            # 回退到原生实现
            probs = F.softmax(logits, dim=-1)
            focal_weight = (1 - probs) ** 2
            
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            focal_loss = (focal_weight * ce_loss.unsqueeze(-1)).mean()
            
            return focal_loss * 0.1  # 权重系数
        
        return torch.tensor(0.0, device=device)
    
    def _compute_anomaly_detection_loss(
        self, 
        outputs: Dict[str, Any], 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        计算异常检测场景的额外损失
        
        使用 MSELoss (backend/lib/losses)
        """
        device = outputs.get('logits', torch.tensor(0.0)).device
        
        # 添加重构损失（用于自编码器类型的异常检测）
        if 'reconstructions' in outputs and 'inputs' in batch:
            reconstructions = outputs['reconstructions']
            inputs = batch['inputs']
            
            # 优先使用底层 MSELoss
            if self._lib_mse_loss is not None:
                try:
                    return self._lib_mse_loss(reconstructions, inputs)
                except Exception as e:
                    logger.debug(f"MSELoss failed: {e}")
            
            return F.mse_loss(reconstructions, inputs)
        
        return torch.tensor(0.0, device=device)
    
    def _compute_optimization_loss(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        context: StrategyContext
    ) -> torch.Tensor:
        """
        计算工艺优化场景的额外损失
        
        使用 HuberLoss, MSELoss (backend/lib/losses)
        """
        device = context.device
        
        if 'predictions' in outputs and 'targets' in batch:
            predictions = outputs['predictions']
            targets = batch['targets']
            
            # 优先使用 HuberLoss（对异常值更鲁棒）
            if self._lib_huber_loss is not None:
                try:
                    return self._lib_huber_loss(predictions.squeeze(), targets.float())
                except Exception as e:
                    logger.debug(f"HuberLoss failed: {e}")
            
            # 使用 MSELoss
            if self._lib_mse_loss is not None:
                try:
                    return self._lib_mse_loss(predictions.squeeze(), targets.float())
                except Exception as e:
                    logger.debug(f"MSELoss failed: {e}")
            
            return F.mse_loss(predictions.squeeze(), targets.float())
        
        return torch.tensor(0.0, device=device)
    
    def _compute_energy_prediction_loss(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        context: StrategyContext
    ) -> torch.Tensor:
        """
        计算能耗预测场景的额外损失
        
        使用 MAELoss, MSELoss (backend/lib/losses)
        """
        device = context.device
        
        if 'predictions' in outputs and 'targets' in batch:
            predictions = outputs['predictions']
            targets = batch['targets']
            
            # 能耗预测通常使用 MAE（对异常值更鲁棒）
            if self._lib_mae_loss is not None:
                try:
                    return self._lib_mae_loss(predictions.squeeze(), targets.float())
                except Exception as e:
                    logger.debug(f"MAELoss failed: {e}")
            
            # 使用 MSELoss
            if self._lib_mse_loss is not None:
                try:
                    return self._lib_mse_loss(predictions.squeeze(), targets.float())
                except Exception as e:
                    logger.debug(f"MSELoss failed: {e}")
            
            return F.l1_loss(predictions.squeeze(), targets.float())
        
        return torch.tensor(0.0, device=device)
    
    def _compute_consistency_loss(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        计算一致性损失
        
        使用 ConsistencyRegularization (backend/lib/losses)
        """
        device = outputs.get('logits', torch.tensor(0.0)).device
        
        if 'predictions' in outputs:
            predictions = outputs['predictions']
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # 使用底层一致性损失
                if self._lib_consistency_loss is not None:
                    try:
                        return self._lib_consistency_loss(
                            predictions[:, :-1], predictions[:, 1:]
                        )
                    except Exception as e:
                        logger.debug(f"ConsistencyRegularization failed: {e}")
                
                # 回退到原生实现
                return torch.mean(torch.abs(predictions[:, 1:] - predictions[:, :-1]))
        
        return torch.tensor(0.0, device=device)
    
    def _check_health(self) -> ScenarioHealthStatus:
        """检查场景健康状态"""
        self._health_status = ScenarioHealthStatus()
        self._health_status.scenario_type = self.config.scenario_type.value
        self._health_status.last_check_time = time.time()
        
        # 检查损失稳定性
        trend = self._scenario_stats.get_loss_trend()
        self._health_status.loss_stable = trend != "degrading"
        if not self._health_status.loss_stable:
            self._health_status.add_issue("Loss is degrading")
        
        # 检查指标是否改善
        if self._scenario_stats.total_steps > 100:
            self._health_status.metrics_improving = trend == "improving" or trend == "stable"
            if not self._health_status.metrics_improving and trend == "degrading":
                self._health_status.add_issue("Metrics not improving")
        
        return self._health_status
    
    def on_phase_start(self, phase: TrainingPhase, context: StrategyContext) -> None:
        """训练阶段开始时的回调"""
        super().on_phase_start(phase, context)
        self._current_phase = phase
        
        # 场景精调阶段使用更小的学习率
        if phase == TrainingPhase.FINETUNE_SCENE:
            if context.optimizer is not None:
                for param_group in context.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                logger.info("Reduced learning rate for scene finetuning")
    
    def on_step_end(self, context: StrategyContext, result: StrategyResult) -> None:
        """步骤结束回调"""
        super().on_step_end(context, result)
        
        # 定期健康检查
        if (self.config.health_check_interval > 0 and 
            context.global_step % self.config.health_check_interval == 0):
            self._check_health()
    
    def get_layer_info(self) -> Dict[str, Any]:
        """
        获取底层模块调用信息
        
        包含 backend/lib/losses 和 base_strategy.py 组件状态
        """
        return {
            # 损失层组件 (backend/lib/losses)
            'lib_task_loss': self._lib_task_loss is not None,
            'lib_focal_loss': self._lib_focal_loss is not None,
            'lib_label_smoothing_loss': self._lib_label_smoothing_loss is not None,
            'lib_mse_loss': self._lib_mse_loss is not None,
            'lib_mae_loss': self._lib_mae_loss is not None,
            'lib_huber_loss': self._lib_huber_loss is not None,
            'lib_consistency_loss': self._lib_consistency_loss is not None,
            'lib_composite_loss': self._lib_composite_loss is not None,
            'lib_multitask_loss': self._lib_multitask_loss is not None,
            'lib_loss_monitor': self._lib_loss_monitor is not None,
            
            # 基础策略组件 (base_strategy.py)
            'strategy_monitor': self._strategy_monitor is not None,
            'strategy_profiler': self._strategy_profiler is not None,
            'strategy_validator': self._strategy_validator is not None,
            'strategy_metrics': self._strategy_metrics is not None,
            
            # 场景信息
            'scenario_type': self.config.scenario_type.value,
            'scene_weight': self.config.scene_weight,
            'loss_type': self.config.get_loss_type(),
            'is_classification': self.config.scenario_type.is_classification,
            'is_regression': self.config.scenario_type.is_regression,
            'is_time_series': self.config.scenario_type.is_time_series,
            'is_imbalanced': self.config.scenario_type.is_imbalanced,
        }
    
    def get_strategy_monitor(self) -> Optional[StrategyMonitor]:
        """获取策略监控器"""
        return self._strategy_monitor
    
    def get_strategy_profiler(self) -> Optional[StrategyProfiler]:
        """获取策略分析器"""
        return self._strategy_profiler
    
    def get_strategy_validator(self) -> Optional[StrategyValidator]:
        """获取策略验证器"""
        return self._strategy_validator
    
    def get_strategy_metrics(self) -> Optional[StrategyMetrics]:
        """获取策略指标"""
        return self._strategy_metrics
    
    def get_health_status(self) -> ScenarioHealthStatus:
        """获取健康状态"""
        return self._health_status
    
    def get_scenario_stats(self) -> ScenarioStats:
        """获取场景统计"""
        return self._scenario_stats
    
    def get_route_stats(self) -> Dict[str, Any]:
        """获取路由统计"""
        return self.router.get_route_stats()
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断策略"""
        diagnosis = {
            'health': self._check_health().to_dict(),
            'layer_info': self.get_layer_info(),
            'scenario_stats': self._scenario_stats.to_dict(),
            'route_stats': self.get_route_stats(),
            'config': self.config.to_dict(),
            'issues': [],
            'recommendations': [],
        }
        
        # 检查策略监控器
        if self._strategy_monitor is not None:
            try:
                if hasattr(self._strategy_monitor, 'get_summary'):
                    diagnosis['monitor_summary'] = self._strategy_monitor.get_summary()
            except Exception:
                pass
        
        if self.config.scenario_type.is_imbalanced and self._lib_focal_loss is None:
            diagnosis['recommendations'].append("Consider using FocalLoss for imbalanced scenarios")
        
        if self.config.scenario_type.is_time_series and self._lib_consistency_loss is None:
            diagnosis['recommendations'].append("Consider using ConsistencyRegularization for time series")
        
        return diagnosis
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diagnosis = self.diagnose()
        
        print("\n" + "=" * 60)
        print("Scenario Strategy Diagnosis")
        print("=" * 60)
        
        print(f"\nScenario: {self.config.scenario_type.value}")
        print(f"Health: {'OK' if diagnosis['health']['is_healthy'] else 'ISSUES FOUND'}")
        
        if diagnosis['health']['issues']:
            print("  Issues:")
            for issue in diagnosis['health']['issues']:
                print(f"    - {issue}")
        
        print(f"\nLoss type: {self.config.get_loss_type()}")
        print(f"Loss trend: {diagnosis['scenario_stats']['loss_trend']}")
        
        print(f"\nLosses layer components:")
        for key, value in diagnosis['layer_info'].items():
            if key.startswith('lib_'):
                status = "✓" if value else "✗"
                print(f"  {key}: {status}")
        
        if diagnosis['recommendations']:
            print("\nRecommendations:")
            for rec in diagnosis['recommendations']:
                print(f"  - {rec}")
        
        print("=" * 60)


class IndustryScenarioStrategy(ProductionTrainingStrategy):
    """
    行业场景训练策略
    
    针对制造业等行业的特定场景优化。
    整合全部六层架构底层能力：
    
    1. 硬件层 (backend/lib/hardware)：
       - DeviceManager: 设备检测和选择
       - MixedPrecisionManager: 混合精度训练
       - MemoryManager: 内存管理
    
    2. 分布式层 (backend/lib/distributed)：
       - DistributedManager: 分布式环境管理
       - DDPWrapper/FSDPWrapper: 数据/模型并行
    
    3. 适配器层 (backend/lib/adapters)：
       - 模态编码器: 时序、图像、文本等
       - 融合模块: 多模态融合
       - 对齐模块: 跨模态对齐
    
    4. 损失层 (backend/lib/losses)：
       - 监督损失: CrossEntropy, Focal, MSE, MAE, Huber, LabelSmoothing
       - 复合损失: CompositeLoss, MultiTaskLoss
       - 场景特定损失: 一致性、正则化
       - 监控: LossMonitor
    
    5. 策略层 (本模块 + base_strategy.py)：
       - 场景路由、适配器、权重调整
       - StrategyMonitor, StrategyProfiler, StrategyValidator, StrategyMetrics
    
    6. production_base.py:
       - ProductionHealthStatus, WrapperStats, get_layer_details
    
    7. 编排层 (orchestrator)：
       - 三阶段训练流程控制
    """
    
    def __init__(
        self, 
        config: Optional[ScenarioStrategyConfig] = None,
        production_config: Optional[ProductionStrategyConfig] = None
    ):
        # 场景配置
        if config is None:
            config = ScenarioStrategyConfig(
                scenario_type=ScenarioType.EQUIPMENT_FAULT_PREDICTION,
                freeze_backbone=True,
                use_scene_adapter=True,
                adapter_dim=64,
                scene_weight=1.0,
                enable_monitoring=True,
            )
        self.scenario_config = config
        
        # 验证配置
        try:
            config.validate()
        except ValueError as e:
            logger.warning(f"Scenario config validation warning: {e}")
        
        # 生产级配置（整合六层架构）
        if production_config is None:
            production_config = ProductionStrategyConfig(
                device="auto",
                precision="fp16",
                enable_amp=True,
                modalities=["time_series"],
                hidden_size=config.adapter_dim * 4,
                task_loss_type=config.get_loss_type(),
                # 启用所有底层能力
                enable_gradient_checkpointing=True,
                enable_monitoring=config.enable_monitoring,
                enable_profiling=config.enable_profiling,
            )
        
        super().__init__(config=production_config, name="industry_scenario", priority=30)
        
        # 行业特定的场景头
        self.scenario_heads: Dict[str, nn.Module] = {}
        
        # 场景路由器
        self.router = ScenarioRouter()
        
        # 生产级上下文 (production_base.py)
        self._production_context: Optional[ProductionTrainingContext] = None
        
        # 场景健康状态
        self._scenario_health = ScenarioHealthStatus()
        self._scenario_stats = ScenarioStats()
        
        # 底层损失模块（场景特定）(backend/lib/losses)
        self._lib_focal_loss: Optional[nn.Module] = None
        self._lib_label_smoothing_loss: Optional[nn.Module] = None
        self._lib_consistency_loss: Optional[nn.Module] = None
        self._lib_mse_loss: Optional[nn.Module] = None
        self._lib_mae_loss: Optional[nn.Module] = None
        self._lib_huber_loss: Optional[nn.Module] = None
        self._lib_composite_loss: Optional[nn.Module] = None
        self._lib_multitask_loss: Optional[nn.Module] = None
        self._lib_loss_monitor: Optional['LossMonitor'] = None
    
    def setup(self, context: StrategyContext) -> None:
        """
        初始化行业场景组件
        
        整合全部六层架构：
        1. 硬件层 - 通过父类 ProductionTrainingStrategy
        2. 分布式层 - 通过父类 ProductionTrainingStrategy
        3. 适配器层 - 通过父类 ProductionTrainingStrategy
        4. 损失层 - 本方法初始化场景特定损失
        5. 策略层 - 场景路由、适配器
        6. 编排层 - 由外部 orchestrator 调用
        """
        # 调用父类初始化（设置硬件层、分布式层、适配器层）
        super().setup(context)
        
        # 初始化生产级上下文（整合六层）
        self._production_context = ProductionTrainingContext(
            config=self.config,
            model=context.model,
            device=context.device
        )
        self._production_context.initialize()
        
        # 初始化底层损失层组件（场景特定）
        self._setup_scenario_losses(context)
        
        # 冻结backbone（如果配置）
        if self.scenario_config.freeze_backbone and context.model is not None:
            self._freeze_backbone(context.model)
        
        # 初始化场景适配器（使用适配器层）
        if self.scenario_config.use_scene_adapter:
            self._init_scene_adapter(context)
        
        # 初始化行业场景头
        hidden_dim = context.config.get('hidden_dim', 768)
        num_classes = context.config.get('num_classes', 2)
        
        # 设备故障预测头
        self.scenario_heads['fault_prediction'] = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        ).to(context.device)
        
        # 工艺优化头
        self.scenario_heads['process_optimization'] = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)  # 回归任务
        ).to(context.device)
        
        # 质量缺陷检测头
        self.scenario_heads['quality_defect'] = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        ).to(context.device)
        
        # 异常检测头（自编码器风格）
        self.scenario_heads['anomaly_detection'] = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        ).to(context.device)
        
        logger.info(f"Industry scenario initialized: type={self.scenario_config.scenario_type.value}, "
                   f"heads={list(self.scenario_heads.keys())}, "
                   f"layers={self.get_all_layers_info()}")
    
    def _setup_scenario_losses(self, context: StrategyContext) -> None:
        """
        初始化场景特定的底层损失模块
        
        使用 backend/lib/losses: FocalLoss, LabelSmoothingLoss, MSELoss, MAELoss, HuberLoss,
                                ConsistencyRegularization, CompositeLoss, MultiTaskLoss,
                                create_loss, create_composite_loss, LossMonitor
        """
        try:
            # 1. Focal Loss（分类场景）
            if FocalLoss is not None:
                self._lib_focal_loss = FocalLoss(
                    gamma=self.scenario_config.focal_gamma,
                    alpha=self.scenario_config.focal_alpha
                ).to(context.device)
            
            # 2. LabelSmoothing Loss
            if LabelSmoothingLoss is not None and self.scenario_config.label_smoothing > 0:
                self._lib_label_smoothing_loss = LabelSmoothingLoss(
                    smoothing=self.scenario_config.label_smoothing
                ).to(context.device)
            
            # 3. 回归损失
            if MSELoss is not None:
                self._lib_mse_loss = MSELoss().to(context.device)
            if MAELoss is not None:
                self._lib_mae_loss = MAELoss().to(context.device)
            if HuberLoss is not None:
                self._lib_huber_loss = HuberLoss().to(context.device)
            
            # 4. 一致性损失（时序场景）
            if ConsistencyRegularization is not None and self.scenario_config.scenario_type.is_time_series:
                self._lib_consistency_loss = ConsistencyRegularization().to(context.device)
            
            # 5. 使用 create_loss 工厂函数
            if create_loss is not None:
                try:
                    loss_type = self.scenario_config.get_loss_type()
                    self._loss_fn = create_loss(loss_type).to(context.device)
                except Exception as e:
                    logger.debug(f"create_loss failed: {e}")
            
            # 6. 创建复合损失 (CompositeLoss)
            if CompositeLoss is not None and self.scenario_config.scenario_type.is_imbalanced:
                try:
                    losses = {'task': self._loss_fn or nn.CrossEntropyLoss()}
                    if self._lib_focal_loss is not None:
                        losses['focal'] = self._lib_focal_loss
                    weights = {'task': 0.6, 'focal': 0.4}
                    self._lib_composite_loss = CompositeLoss(losses, weights).to(context.device)
                except Exception as e:
                    logger.debug(f"CompositeLoss failed: {e}")
            
            # 7. 创建多任务损失 (MultiTaskLoss)
            if MultiTaskLoss is not None:
                try:
                    task_losses = {'primary': self._loss_fn or nn.CrossEntropyLoss()}
                    if self._lib_consistency_loss is not None:
                        task_losses['consistency'] = self._lib_consistency_loss
                    task_weights = {'primary': 1.0, 'consistency': self.scenario_config.consistency_loss_weight}
                    self._lib_multitask_loss = MultiTaskLoss(task_losses, task_weights).to(context.device)
                except Exception as e:
                    logger.debug(f"MultiTaskLoss failed: {e}")
            
            # 8. 使用 create_composite_loss 工厂函数
            if create_composite_loss is not None:
                try:
                    loss_configs = [
                        ('task', self._loss_fn or nn.CrossEntropyLoss(), self.scenario_config.task_loss_weight)
                    ]
                    if self._lib_focal_loss is not None and self.scenario_config.scenario_type.is_imbalanced:
                        loss_configs.append(('focal', self._lib_focal_loss, 0.2))
                    self._lib_composite_loss = create_composite_loss(loss_configs)
                except Exception as e:
                    logger.debug(f"create_composite_loss failed: {e}")
            
            # 9. 初始化损失监控器 (LossMonitor)
            if LossMonitor is not None:
                try:
                    self._lib_loss_monitor = LossMonitor(max_history=10000)
                except Exception as e:
                    logger.debug(f"LossMonitor failed: {e}")
            
            logger.info(f"Industry scenario losses initialized: "
                       f"focal={self._lib_focal_loss is not None}, "
                       f"composite={self._lib_composite_loss is not None}, "
                       f"multitask={self._lib_multitask_loss is not None}, "
                       f"monitor={self._lib_loss_monitor is not None}")
        except Exception as e:
            logger.warning(f"Failed to initialize scenario losses: {e}")
    
    def _freeze_backbone(self, model: nn.Module) -> None:
        """冻结backbone参数"""
        frozen_count = 0
        for name, param in model.named_parameters():
            # 只冻结backbone层，保留head层
            if 'head' not in name.lower() and 'classifier' not in name.lower():
                param.requires_grad = False
                frozen_count += 1
        logger.info(f"Frozen {frozen_count} backbone parameters")
    
    def _init_scene_adapter(self, context: StrategyContext) -> None:
        """初始化场景适配器"""
        hidden_dim = context.config.get('hidden_dim', 768)
        
        # 如果有适配器层可用，使用底层适配器
        from backend.lib.adapters import create_adapter
        self._adapter = create_adapter(
            'lora',
            hidden_size=hidden_dim,
            lora_rank=self.scenario_config.adapter_dim
        )
        
        logger.info(f"Scene adapter initialized: dim={self.scenario_config.adapter_dim}")
    
    def prepare_batch(self, batch: Dict[str, Any], context: StrategyContext) -> Dict[str, Any]:
        """准备场景化批次数据"""
        # 使用生产级上下文移动数据到设备
        if self._production_context:
            batch = self._production_context.to_device(batch)
        
        # 添加场景权重
        batch['scene_weight'] = self.scenario_config.scene_weight
        batch['scenario_type'] = self.scenario_config.scenario_type.value
        
        # 数据增强（如果启用）
        if self.scenario_config.augmentation_enabled and context.model.training:
            batch = self._apply_augmentation(batch, context)
        
        return batch
    
    def _apply_augmentation(
        self, 
        batch: Dict[str, Any], 
        context: StrategyContext
    ) -> Dict[str, Any]:
        """应用场景特定的数据增强"""
        # 基于场景类型应用不同的增强策略
        if self.scenario_config.scenario_type == ScenarioType.EQUIPMENT_FAULT_PREDICTION:
            # 时序数据增强：添加噪声
            if 'time_series' in batch:
                noise_level = self.scenario_config.augmentation_strength * 0.1
                batch['time_series'] = batch['time_series'] + torch.randn_like(
                    batch['time_series']
                ) * noise_level
        
        elif self.scenario_config.scenario_type == ScenarioType.QUALITY_DEFECT_DETECTION:
            # 图像数据增强（简化实现）
            pass
        
        return batch
    
    def compute_loss(
        self, 
        model: nn.Module, 
        batch: Dict[str, Any], 
        outputs: Dict[str, Any],
        context: StrategyContext
    ) -> StrategyResult:
        """
        计算场景化损失
        
        整合六层架构能力：
        - 使用硬件层的混合精度
        - 使用损失层的场景特定损失
        - 使用策略层的损失组合
        """
        metrics = {}
        scene_weight = batch.get('scene_weight', self.scenario_config.scene_weight)
        
        # 使用生产级AMP上下文（硬件层）
        amp_ctx = self._production_context.get_amp_context() if self._production_context else None
        
        with amp_ctx if amp_ctx else torch.no_grad():
            # 1. 任务损失（优先使用损失层）
            task_loss = self._compute_task_loss_industry(outputs, batch, context)
            metrics['task_loss'] = task_loss.item()
        
        # 2. 场景特定损失（使用底层损失层）
        scene_loss = torch.tensor(0.0, device=context.device)
        
        if self.scenario_config.scenario_type == ScenarioType.EQUIPMENT_FAULT_PREDICTION:
            scene_loss = self._compute_fault_prediction_loss(outputs, batch)
        elif self.scenario_config.scenario_type == ScenarioType.QUALITY_DEFECT_DETECTION:
            scene_loss = self._compute_defect_detection_loss(outputs, batch)
        elif self.scenario_config.scenario_type == ScenarioType.ANOMALY_DETECTION:
            scene_loss = self._compute_anomaly_detection_loss(outputs, batch)
        elif self.scenario_config.scenario_type == ScenarioType.PROCESS_OPTIMIZATION:
            scene_loss = self._compute_optimization_loss(outputs, batch)
        
        if scene_loss.item() > 0:
            metrics['scene_loss'] = scene_loss.item()
        
        # 3. 计算总损失（策略层）
        total_loss = scene_weight * (
            self.scenario_config.task_loss_weight * task_loss + 
            self.scenario_config.scene_specific_loss_weight * scene_loss
        )
        
        metrics['total_loss'] = total_loss.item()
        metrics['scene_weight'] = scene_weight
        metrics['layers_used'] = {
            'hardware': amp_ctx is not None,
            'losses': [cls.__name__ for cls in LOSSES_AVAILABLE if cls is not None]
        }
        
        return StrategyResult(loss=total_loss, metrics=metrics)
    
    def _compute_task_loss_industry(
        self,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        context: StrategyContext
    ) -> torch.Tensor:
        """
        使用底层损失层计算任务损失
        
        优先使用 backend/lib/losses 的损失函数
        """
        # 如果 outputs 中已有 loss
        if 'loss' in outputs:
            return outputs['loss']
        elif hasattr(outputs, 'loss'):
            return outputs.loss
        
        # 使用生产级上下文计算
        if 'logits' in outputs and 'labels' in batch:
            logits = outputs['logits']
            labels = batch['labels']
            
            # 根据场景类型选择损失函数
            if self.scenario_config.scenario_type in [
                ScenarioType.QUALITY_DEFECT_DETECTION,
                ScenarioType.FRAUD_DETECTION
            ] and self._lib_focal_loss is not None:
                # 使用 Focal Loss 处理类别不平衡
                return self._lib_focal_loss(logits, labels)
            elif self.scenario_config.scenario_type in [
                ScenarioType.PROCESS_OPTIMIZATION,
                ScenarioType.ENERGY_PREDICTION
            ] and self._lib_mse_loss is not None:
                # 使用 MSE Loss 回归
                predictions = outputs.get('predictions', logits)
                targets = batch.get('targets', labels)
                return self._lib_mse_loss(predictions.squeeze(), targets.float())
            elif self._production_context:
                # 使用生产级上下文
                return self._production_context.compute_loss(logits, labels)
            else:
                return F.cross_entropy(logits, labels)
        
        raise ValueError("Cannot compute task loss: no logits or labels")
    
    def _compute_fault_prediction_loss(
        self, 
        outputs: Dict[str, Any], 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        计算故障预测场景的额外损失
        
        使用底层损失层的一致性正则化
        """
        device = outputs.get('logits', torch.tensor(0.0)).device
        
        if 'predictions' in outputs and 'sequence_length' in batch:
            predictions = outputs['predictions']
            # 时序一致性：相邻时间步的预测不应该剧烈变化
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # 优先使用底层一致性损失
                if self._lib_consistency_loss is not None:
                    try:
                        return self._lib_consistency_loss(
                            predictions[:, :-1], predictions[:, 1:]
                        )
                    except Exception as e:
                        logger.warning(f"Consistency loss failed: {e}")
                
                # 回退到原生实现
                consistency_loss = torch.mean(
                    torch.abs(predictions[:, 1:] - predictions[:, :-1])
                )
                return consistency_loss
        
        return torch.tensor(0.0, device=device)
    
    def _compute_defect_detection_loss(
        self, 
        outputs: Dict[str, Any], 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        计算缺陷检测场景的额外损失
        
        使用底层损失层的 Focal Loss
        """
        device = outputs.get('logits', torch.tensor(0.0)).device
        
        if 'logits' in outputs and 'labels' in batch:
            logits = outputs['logits']
            labels = batch['labels']
            
            # 优先使用底层 Focal Loss
            if self._lib_focal_loss is not None:
                try:
                    return self._lib_focal_loss(logits, labels) * 0.1
                except Exception as e:
                    logger.warning(f"Focal loss failed: {e}")
            
            # 回退到原生实现
            probs = F.softmax(logits, dim=-1)
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            pt = probs.gather(1, labels.unsqueeze(1)).squeeze()
            focal_weight = (1 - pt) ** 2
            focal_loss = (focal_weight * ce_loss).mean()
            
            return focal_loss * 0.1
        
        return torch.tensor(0.0, device=device)
    
    def _compute_anomaly_detection_loss(
        self, 
        outputs: Dict[str, Any], 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        计算异常检测场景的额外损失
        
        使用底层损失层的 MSE Loss（重构损失）
        """
        device = outputs.get('logits', torch.tensor(0.0)).device
        
        if 'features' in outputs and 'anomaly_detection' in self.scenario_heads:
            features = outputs['features']
            reconstructed = self.scenario_heads['anomaly_detection'](features)
            
            # 优先使用底层 MSE Loss
            if self._lib_mse_loss is not None:
                try:
                    return self._lib_mse_loss(reconstructed, features)
                except Exception as e:
                    logger.warning(f"MSE loss failed: {e}")
            
            # 回退到原生实现
            return F.mse_loss(reconstructed, features)
        
        return torch.tensor(0.0, device=device)
    
    def _compute_optimization_loss(
        self, 
        outputs: Dict[str, Any], 
        batch: Dict[str, Any]
    ) -> torch.Tensor:
        """
        计算工艺优化场景的额外损失
        
        使用底层损失层的 MSE/Huber Loss
        """
        device = outputs.get('logits', torch.tensor(0.0)).device
        
        if 'predictions' in outputs and 'targets' in batch:
            predictions = outputs['predictions']
            targets = batch['targets']
            
            # 优先使用底层 MSE Loss
            if self._lib_mse_loss is not None:
                try:
                    return self._lib_mse_loss(predictions.squeeze(), targets.float())
                except Exception as e:
                    logger.warning(f"MSE loss failed: {e}")
            
            # 回退到原生实现
            return F.mse_loss(predictions.squeeze(), targets.float())
        
        return torch.tensor(0.0, device=device)
    
    def get_scenario_head(self, scenario: str) -> Optional[nn.Module]:
        """获取场景特定的头部网络"""
        return self.scenario_heads.get(scenario)
    
    def on_phase_start(self, phase: TrainingPhase, context: StrategyContext) -> None:
        """训练阶段开始时的回调"""
        # 场景精调阶段使用更小的学习率
        if phase == TrainingPhase.FINETUNE_SCENE:
            if context.optimizer is not None:
                for param_group in context.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
                logger.info("Reduced learning rate for scene finetuning")
    
    def on_step_end(self, context: StrategyContext, result: StrategyResult) -> None:
        """步骤结束回调"""
        super().on_step_end(context, result)
        
        # 更新场景统计
        self._scenario_stats.update(result.loss.item() if hasattr(result.loss, 'item') else result.loss)
        
        # 记录到损失监控器
        if self._lib_loss_monitor is not None:
            try:
                self._lib_loss_monitor.record(result.loss.item() if hasattr(result.loss, 'item') else result.loss)
            except Exception:
                pass
        
        # 定期健康检查
        if (self.scenario_config.health_check_interval > 0 and 
            context.global_step % self.scenario_config.health_check_interval == 0):
            self._check_scenario_health()
    
    def _check_scenario_health(self) -> ScenarioHealthStatus:
        """检查场景健康状态"""
        self._scenario_health = ScenarioHealthStatus()
        self._scenario_health.scenario_type = self.scenario_config.scenario_type.value
        self._scenario_health.last_check_time = time.time()
        
        # 检查损失稳定性
        trend = self._scenario_stats.get_loss_trend()
        self._scenario_health.loss_stable = trend != "degrading"
        if not self._scenario_health.loss_stable:
            self._scenario_health.add_issue("Loss is degrading")
        
        # 获取父类健康状态 (ProductionHealthStatus)
        parent_health = self.get_health_status()
        if not parent_health.is_healthy:
            self._scenario_health.is_healthy = False
            self._scenario_health.issues.extend(parent_health.issues)
        
        return self._scenario_health
    
    def cleanup(self) -> None:
        """清理资源"""
        super().cleanup()
        if self._production_context:
            self._production_context.cleanup()
        logger.info("IndustryScenarioStrategy cleaned up")
    
    def get_scenario_health(self) -> ScenarioHealthStatus:
        """获取场景健康状态"""
        return self._scenario_health
    
    def get_scenario_stats(self) -> ScenarioStats:
        """获取场景统计"""
        return self._scenario_stats
    
    def get_all_layers_info(self) -> Dict[str, Any]:
        """
        获取全部六层架构的调用信息
        
        返回各层的初始化状态和可用性
        使用: get_layer_details, get_available_layers (production_base.py)
              ProductionHealthStatus, WrapperStats (production_base.py)
        """
        # 获取父类的层信息 (使用 get_info from ProductionTrainingStrategy)
        base_info = self.get_info() if hasattr(self, 'get_info') else {}
        
        # 使用 get_layer_details 获取底层详情
        layer_details = get_layer_details()
        
        # 使用 get_available_layers 获取可用性
        available_layers = get_available_layers()
        
        return {
            # 1. 硬件层 (使用 layer_details['hardware'])
            'hardware_layer': {
                'available': available_layers.get('hardware', False),
                'device_manager': self._device_manager is not None if hasattr(self, '_device_manager') else False,
                'memory_manager': self._memory_manager is not None if hasattr(self, '_memory_manager') else False,
                'amp_manager': self._amp_manager is not None if hasattr(self, '_amp_manager') else False,
                'gradient_checkpointing': self._gradient_checkpointing is not None if hasattr(self, '_gradient_checkpointing') else False,
                'components': layer_details.get('hardware', {}).get('components', {})
            },
            # 2. 分布式层 (使用 layer_details['distributed'])
            'distributed_layer': {
                'available': available_layers.get('distributed', False),
                'distributed_manager': self._distributed_manager is not None if hasattr(self, '_distributed_manager') else False,
                'distributed_wrapper': self._distributed_wrapper is not None if hasattr(self, '_distributed_wrapper') else False,
                'is_distributed': self._is_distributed if hasattr(self, '_is_distributed') else False,
                'wrapper_stats': self._wrapper_stats.to_dict() if hasattr(self, '_wrapper_stats') else {},
                'components': layer_details.get('distributed', {}).get('components', {})
            },
            # 3. 适配器层 (使用 layer_details['adapters'])
            'adapters_layer': {
                'available': available_layers.get('adapters', False),
                'adapter_manager': self._adapter_manager is not None if hasattr(self, '_adapter_manager') else False,
                'encoders': list(self._encoders.keys()) if hasattr(self, '_encoders') else [],
                'fusion': self._fusion is not None if hasattr(self, '_fusion') else False,
                'alignment': self._alignment is not None if hasattr(self, '_alignment') else False,
                'scene_adapter': self._adapter is not None if hasattr(self, '_adapter') else False,
                'components': layer_details.get('adapters', {}).get('components', {})
            },
            # 4. 损失层 (使用 layer_details['losses'])
            'losses_layer': {
                'available': available_layers.get('losses', False),
                'focal_loss': self._lib_focal_loss is not None,
                'label_smoothing_loss': self._lib_label_smoothing_loss is not None,
                'consistency_loss': self._lib_consistency_loss is not None,
                'mse_loss': self._lib_mse_loss is not None,
                'mae_loss': self._lib_mae_loss is not None,
                'huber_loss': self._lib_huber_loss is not None,
                'composite_loss': self._lib_composite_loss is not None,
                'multitask_loss': self._lib_multitask_loss is not None,
                'loss_monitor': self._lib_loss_monitor is not None,
                'task_loss': self._loss_fn is not None if hasattr(self, '_loss_fn') else False,
                'components': layer_details.get('losses', {}).get('components', {})
            },
            # 5. 策略层 (base_strategy.py)
            'strategy_layer': {
                'scenario_type': self.scenario_config.scenario_type.value,
                'scenario_heads': list(self.scenario_heads.keys()),
                'scene_weight': self.scenario_config.scene_weight,
                'router': self.router is not None,
                'router_stats': self.router.get_route_stats(),
                'strategy_monitor': self._strategy_monitor is not None if hasattr(self, '_strategy_monitor') else False,
                'strategy_profiler': self._strategy_profiler is not None if hasattr(self, '_strategy_profiler') else False,
                'strategy_validator': self._strategy_validator is not None if hasattr(self, '_strategy_validator') else False,
                'strategy_metrics': self._strategy_metrics is not None if hasattr(self, '_strategy_metrics') else False,
            },
            # 6. 生产级组件 (production_base.py)
            'production_layer': {
                'production_context': self._production_context is not None,
                'production_health': base_info.get('health_status', {}),
                'wrapper_stats': base_info.get('wrapper_stats', {}),
            },
            # 7. 场景组件
            'scenario_layer': {
                'scenario_health': self._scenario_health.to_dict(),
                'scenario_stats': self._scenario_stats.to_dict(),
            }
        }
    
    def diagnose(self) -> Dict[str, Any]:
        """
        诊断策略
        
        整合父类诊断和场景诊断
        """
        # 获取父类诊断
        parent_diagnosis = super().diagnose()
        
        # 场景诊断
        scenario_diagnosis = {
            'scenario_type': self.scenario_config.scenario_type.value,
            'scenario_health': self._check_scenario_health().to_dict(),
            'scenario_stats': self._scenario_stats.to_dict(),
            'route_stats': self.router.get_route_stats(),
            'all_layers': self.get_all_layers_info(),
        }
        
        # 合并
        diagnosis = {**parent_diagnosis, **scenario_diagnosis}
        
        # 添加场景特定建议
        if self.scenario_config.scenario_type.is_imbalanced and self._lib_focal_loss is None:
            diagnosis['recommendations'].append("Use FocalLoss for imbalanced scenario")
        
        if self.scenario_config.scenario_type.is_time_series and self._lib_consistency_loss is None:
            diagnosis['recommendations'].append("Use ConsistencyRegularization for time series scenario")
        
        return diagnosis
    
    def print_diagnosis(self) -> None:
        """打印诊断信息"""
        diagnosis = self.diagnose()
        
        print("\n" + "=" * 60)
        print("Industry Scenario Strategy Diagnosis")
        print("=" * 60)
        
        print(f"\nScenario: {diagnosis['scenario_type']}")
        print(f"Description: {self.scenario_config.scenario_type.get_description()}")
        
        print(f"\nScenario Health:")
        print(f"  is_healthy: {diagnosis['scenario_health']['is_healthy']}")
        print(f"  loss_stable: {diagnosis['scenario_health']['loss_stable']}")
        if diagnosis['scenario_health']['issues']:
            print("  Issues:")
            for issue in diagnosis['scenario_health']['issues']:
                print(f"    - {issue}")
        
        print(f"\nLoss Trend: {diagnosis['scenario_stats']['loss_trend']}")
        print(f"Total Steps: {diagnosis['scenario_stats']['total_steps']}")
        
        print(f"\nLayers Status:")
        for layer_name in ['hardware_layer', 'distributed_layer', 'adapters_layer', 'losses_layer', 'strategy_layer']:
            layer_info = diagnosis['all_layers'].get(layer_name, {})
            available = layer_info.get('available', False)
            print(f"  {layer_name}: {'✓' if available else '✗'}")
        
        if diagnosis['recommendations']:
            print("\nRecommendations:")
            for rec in diagnosis['recommendations']:
                print(f"  - {rec}")
        
        print("=" * 60)


# ==================== 便捷函数 ====================

def create_scenario_strategy(
    scenario_type: str = "basic_model",
    **kwargs
) -> ScenarioStrategy:
    """
    创建场景策略
    
    Args:
        scenario_type: 场景类型
        **kwargs: 其他配置参数
    
    Returns:
        ScenarioStrategy实例
    """
    try:
        scenario = ScenarioType.from_string(scenario_type)
    except ValueError:
        scenario = ScenarioType.BASIC_MODEL
    
    config = ScenarioStrategyConfig(
        scenario_type=scenario,
        **{k: v for k, v in kwargs.items() if hasattr(ScenarioStrategyConfig, k)}
    )
    return ScenarioStrategy(config=config)


def create_industry_scenario_strategy(
    scenario_type: str = "equipment_fault_prediction",
    **kwargs
) -> IndustryScenarioStrategy:
    """
    创建行业场景策略
    
    Args:
        scenario_type: 场景类型
        **kwargs: 其他配置参数
    
    Returns:
        IndustryScenarioStrategy实例
    """
    try:
        scenario = ScenarioType.from_string(scenario_type)
    except ValueError:
        scenario = ScenarioType.EQUIPMENT_FAULT_PREDICTION
    
    scenario_config = ScenarioStrategyConfig(
        scenario_type=scenario,
        **{k: v for k, v in kwargs.items() if hasattr(ScenarioStrategyConfig, k)}
    )
    
    production_kwargs = {k: v for k, v in kwargs.items() if hasattr(ProductionStrategyConfig, k)}
    production_config = ProductionStrategyConfig(**production_kwargs) if production_kwargs else None
    
    return IndustryScenarioStrategy(
        config=scenario_config,
        production_config=production_config
    )


def get_available_scenarios() -> List[str]:
    """获取可用的场景类型"""
    return [st.value for st in ScenarioType]


def get_scenario_info(scenario_type: str) -> Dict[str, Any]:
    """
    获取场景信息
    
    Args:
        scenario_type: 场景类型
    
    Returns:
        场景信息字典
    """
    try:
        scenario = ScenarioType.from_string(scenario_type)
    except ValueError:
        return {'error': f'Unknown scenario type: {scenario_type}'}
    
    return {
        'value': scenario.value,
        'description': scenario.get_description(),
        'is_classification': scenario.is_classification,
        'is_regression': scenario.is_regression,
        'is_time_series': scenario.is_time_series,
        'is_imbalanced': scenario.is_imbalanced,
        'recommended_loss': scenario.recommended_loss,
    }


def print_scenario_info(scenario_type: Optional[str] = None) -> None:
    """打印场景信息"""
    print("\n" + "=" * 60)
    print("Scenario Types Information")
    print("=" * 60)
    
    scenarios = [scenario_type] if scenario_type else get_available_scenarios()
    
    for st in scenarios:
        info = get_scenario_info(st)
        if 'error' in info:
            print(f"\n{st}: {info['error']}")
            continue
        
        print(f"\n{st}:")
        print(f"  Description: {info['description']}")
        print(f"  Classification: {info['is_classification']}")
        print(f"  Regression: {info['is_regression']}")
        print(f"  Time Series: {info['is_time_series']}")
        print(f"  Imbalanced: {info['is_imbalanced']}")
        print(f"  Recommended Loss: {info['recommended_loss']}")
    
    print("\n" + "=" * 60)


def diagnose_scenario_strategy(strategy: ScenarioStrategy) -> Dict[str, Any]:
    """诊断场景策略"""
    return strategy.diagnose()


def print_scenario_diagnosis(strategy: ScenarioStrategy) -> None:
    """打印场景策略诊断"""
    strategy.print_diagnosis()


def compare_scenario_strategies(
    strategies: List[ScenarioStrategy]
) -> Dict[str, Any]:
    """
    比较多个场景策略
    
    Args:
        strategies: 策略列表
    
    Returns:
        比较结果
    """
    comparison = {
        'strategies': [],
        'common_issues': [],
        'recommendations': [],
    }
    
    all_issues = []
    for strategy in strategies:
        diag = strategy.diagnose()
        comparison['strategies'].append({
            'name': strategy.name,
            'scenario_type': strategy.config.scenario_type.value,
            'health': diag.get('health', {}),
            'layer_info': diag.get('layer_info', {}),
        })
        all_issues.extend(diag.get('health', {}).get('issues', []))
    
    # 找出共同问题
    from collections import Counter
    issue_counts = Counter(all_issues)
    comparison['common_issues'] = [issue for issue, count in issue_counts.items() if count > 1]
    
    return comparison


def recommend_scenario(
    task_type: str,
    data_characteristics: Dict[str, Any]
) -> ScenarioType:
    """
    推荐场景类型
    
    Args:
        task_type: 任务类型 (classification, regression, detection)
        data_characteristics: 数据特征 (imbalanced, time_series, image, etc.)
    
    Returns:
        推荐的场景类型
    """
    is_imbalanced = data_characteristics.get('imbalanced', False)
    is_time_series = data_characteristics.get('time_series', False)
    is_image = data_characteristics.get('image', False)
    domain = data_characteristics.get('domain', 'general')
    
    # 行业场景推荐
    if domain == 'manufacturing':
        if is_time_series:
            return ScenarioType.EQUIPMENT_FAULT_PREDICTION
        if is_image:
            return ScenarioType.QUALITY_DEFECT_DETECTION
        return ScenarioType.PROCESS_OPTIMIZATION
    
    if domain == 'finance':
        if is_imbalanced:
            return ScenarioType.FRAUD_DETECTION
        return ScenarioType.RISK_ASSESSMENT
    
    if domain == 'healthcare':
        if is_image:
            return ScenarioType.MEDICAL_IMAGE_ANALYSIS
        return ScenarioType.DISEASE_DIAGNOSIS
    
    # 通用场景推荐
    if is_time_series and task_type == 'detection':
        return ScenarioType.ANOMALY_DETECTION
    
    if task_type == 'regression':
        return ScenarioType.ENERGY_PREDICTION
    
    return ScenarioType.BASIC_MODEL

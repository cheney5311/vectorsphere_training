"""训练插件接口

生产级插件系统接口定义，支持：
- 训练生命周期钩子
- 策略层集成
- 硬件层集成
- 进度回调

架构位置：
├── plugins/interface.py (本模块)
│   └── 定义插件接口和基础类
├── plugins/registry.py
│   └── 插件注册和管理
└── 被 orchestrator, pipeline 调用
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# ==================== 策略层导入 ====================

STRATEGY_LAYER_AVAILABLE = False
StrategyContext = None
StrategyResult = None

try:
    from backend.modules.training.strategies.base_strategy import (
        StrategyContext, StrategyResult,
    )
    STRATEGY_LAYER_AVAILABLE = True
except (ImportError, SyntaxError, IndentationError):
    pass


# ==================== 枚举定义 ====================

class PluginType(str, Enum):
    """插件类型"""
    CALLBACK = "callback"
    STRATEGY = "strategy"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    DATA_AUGMENTATION = "data_augmentation"
    LOGGING = "logging"
    CHECKPOINT = "checkpoint"
    MONITORING = "monitoring"
    CUSTOM = "custom"


class PluginPriority(int, Enum):
    """插件优先级"""
    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


class HookPoint(str, Enum):
    """钩子点"""
    # 训练生命周期
    ON_TRAINING_START = "on_training_start"
    ON_TRAINING_END = "on_training_end"
    
    # Epoch 生命周期
    ON_EPOCH_START = "on_epoch_start"
    ON_EPOCH_END = "on_epoch_end"
    
    # Step 生命周期
    ON_STEP_START = "on_step_start"
    ON_STEP_END = "on_step_end"
    
    # 阶段生命周期
    ON_STAGE_START = "on_stage_start"
    ON_STAGE_END = "on_stage_end"
    
    # 检查点
    ON_CHECKPOINT_SAVE = "on_checkpoint_save"
    ON_CHECKPOINT_LOAD = "on_checkpoint_load"
    
    # 评估
    ON_EVALUATION_START = "on_evaluation_start"
    ON_EVALUATION_END = "on_evaluation_end"
    
    # 异常
    ON_ERROR = "on_error"
    ON_EXCEPTION = "on_exception"


# ==================== 数据类定义 ====================

@dataclass
class PluginConfig:
    """插件配置"""
    name: str
    plugin_type: PluginType = PluginType.CALLBACK
    priority: PluginPriority = PluginPriority.NORMAL
    enabled: bool = True
    hooks: List[HookPoint] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'plugin_type': self.plugin_type.value,
            'priority': self.priority.value,
            'enabled': self.enabled,
            'hooks': [h.value for h in self.hooks],
            'params': self.params,
        }


@dataclass
class PluginContext:
    """插件执行上下文"""
    hook: HookPoint
    session_id: str = ""
    epoch: int = 0
    step: int = 0
    stage: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    model: Any = None
    optimizer: Any = None
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 策略层上下文
    strategy_context: Optional['StrategyContext'] = None
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PluginResult:
    """插件执行结果"""
    success: bool = True
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    should_stop: bool = False
    modified_data: Optional[Dict[str, Any]] = None


# ==================== 插件基类 ====================

class TrainingPlugin(ABC):
    """训练插件基类
    
    所有训练插件的抽象基类，定义插件生命周期和接口。
    """

    def __init__(self, config: Optional[PluginConfig] = None):
        self.config = config or PluginConfig(name=self.__class__.__name__)
        self._enabled = self.config.enabled
        self._initialized = False
    
    @property
    def name(self) -> str:
        """插件名称"""
        return self.config.name
    
    @property
    def plugin_type(self) -> PluginType:
        """插件类型"""
        return self.config.plugin_type
    
    @property
    def priority(self) -> int:
        """插件优先级"""
        return self.config.priority.value if isinstance(self.config.priority, PluginPriority) else self.config.priority
    
    @property
    def enabled(self) -> bool:
        """是否启用"""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
    
    def initialize(self) -> bool:
        """初始化插件
        
        Returns:
            是否初始化成功
        """
        self._initialized = True
        logger.debug(f"Plugin initialized: {self.name}")
        return True
    
    def shutdown(self) -> None:
        """关闭插件"""
        self._initialized = False
        logger.debug(f"Plugin shutdown: {self.name}")
    
    def supports_hook(self, hook: HookPoint) -> bool:
        """检查是否支持指定钩子"""
        if not self.config.hooks:
            return True  # 空列表表示支持所有钩子
        return hook in self.config.hooks
    
    @abstractmethod
    def execute(self, context: PluginContext) -> PluginResult:
        """执行插件逻辑
        
        Args:
            context: 插件执行上下文
        
        Returns:
            插件执行结果
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        return {
            'name': self.name,
            'type': self.plugin_type.value,
            'priority': self.priority,
            'enabled': self.enabled,
            'initialized': self._initialized,
            'config': self.config.to_dict(),
        }


# ==================== 回调插件基类 ====================

class CallbackPlugin(TrainingPlugin):
    """回调插件基类"""
    
    def __init__(self, config: Optional[PluginConfig] = None):
        if config is None:
            config = PluginConfig(
                name=self.__class__.__name__,
                plugin_type=PluginType.CALLBACK,
            )
        super().__init__(config)
    
    def on_training_start(self, context: PluginContext) -> PluginResult:
        """训练开始回调"""
        return PluginResult()
    
    def on_training_end(self, context: PluginContext) -> PluginResult:
        """训练结束回调"""
        return PluginResult()
    
    def on_epoch_start(self, context: PluginContext) -> PluginResult:
        """Epoch 开始回调"""
        return PluginResult()
    
    def on_epoch_end(self, context: PluginContext) -> PluginResult:
        """Epoch 结束回调"""
        return PluginResult()
    
    def on_step_start(self, context: PluginContext) -> PluginResult:
        """Step 开始回调"""
        return PluginResult()
    
    def on_step_end(self, context: PluginContext) -> PluginResult:
        """Step 结束回调"""
        return PluginResult()
    
    def execute(self, context: PluginContext) -> PluginResult:
        """执行回调"""
        hook_handlers = {
            HookPoint.ON_TRAINING_START: self.on_training_start,
            HookPoint.ON_TRAINING_END: self.on_training_end,
            HookPoint.ON_EPOCH_START: self.on_epoch_start,
            HookPoint.ON_EPOCH_END: self.on_epoch_end,
            HookPoint.ON_STEP_START: self.on_step_start,
            HookPoint.ON_STEP_END: self.on_step_end,
        }
        
        handler = hook_handlers.get(context.hook)
        if handler:
            return handler(context)
        
        return PluginResult()


# ==================== 监控插件基类 ====================

class MonitoringPlugin(TrainingPlugin):
    """监控插件基类"""
    
    def __init__(self, config: Optional[PluginConfig] = None):
        if config is None:
            config = PluginConfig(
                name=self.__class__.__name__,
                plugin_type=PluginType.MONITORING,
                hooks=[
                    HookPoint.ON_STEP_END,
                    HookPoint.ON_EPOCH_END,
                ],
            )
        super().__init__(config)
        self._metrics_history: List[Dict[str, Any]] = []
    
    def record_metrics(self, metrics: Dict[str, Any]) -> None:
        """记录指标"""
        self._metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
        })
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """获取指标历史"""
        return self._metrics_history
    
    def execute(self, context: PluginContext) -> PluginResult:
        """执行监控"""
        if context.metrics:
            self.record_metrics(context.metrics)
        return PluginResult(metrics=context.metrics)


# ==================== 检查点插件基类 ====================

class CheckpointPlugin(TrainingPlugin):
    """检查点插件基类"""
    
    def __init__(self, config: Optional[PluginConfig] = None):
        if config is None:
            config = PluginConfig(
                name=self.__class__.__name__,
                plugin_type=PluginType.CHECKPOINT,
                hooks=[
                    HookPoint.ON_CHECKPOINT_SAVE,
                    HookPoint.ON_CHECKPOINT_LOAD,
                ],
            )
        super().__init__(config)
    
    def save_checkpoint(self, context: PluginContext, path: str) -> PluginResult:
        """保存检查点"""
        return PluginResult()
    
    def load_checkpoint(self, context: PluginContext, path: str) -> PluginResult:
        """加载检查点"""
        return PluginResult()
    
    def execute(self, context: PluginContext) -> PluginResult:
        """执行检查点操作"""
        path = context.data.get('checkpoint_path', '')
        
        if context.hook == HookPoint.ON_CHECKPOINT_SAVE:
            return self.save_checkpoint(context, path)
        elif context.hook == HookPoint.ON_CHECKPOINT_LOAD:
            return self.load_checkpoint(context, path)
        
        return PluginResult()


# ==================== 便捷函数 ====================

def create_plugin_config(
    name: str,
    plugin_type: str = "callback",
    priority: int = 50,
    **kwargs
) -> PluginConfig:
    """创建插件配置"""
    try:
        ptype = PluginType(plugin_type)
    except ValueError:
        ptype = PluginType.CUSTOM
    
    try:
        ppriority = PluginPriority(priority)
    except ValueError:
        ppriority = PluginPriority.NORMAL
    
    return PluginConfig(
        name=name,
        plugin_type=ptype,
        priority=ppriority,
        **kwargs
    )


def create_plugin_context(
    hook: str,
    session_id: str = "",
    **kwargs
) -> PluginContext:
    """创建插件上下文"""
    try:
        hook_point = HookPoint(hook)
    except ValueError:
        hook_point = HookPoint.ON_STEP_END
    
    return PluginContext(
        hook=hook_point,
        session_id=session_id,
        **kwargs
    )


# ==================== 导出 ====================

__all__ = [
    # 基类
    'TrainingPlugin',
    'CallbackPlugin',
    'MonitoringPlugin',
    'CheckpointPlugin',
    
    # 数据类
    'PluginConfig',
    'PluginContext',
    'PluginResult',
    
    # 枚举
    'PluginType',
    'PluginPriority',
    'HookPoint',
    
    # 便捷函数
    'create_plugin_config',
    'create_plugin_context',
    
    # 层可用性
    'STRATEGY_LAYER_AVAILABLE',
]

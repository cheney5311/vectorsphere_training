"""基础训练场景抽象类

生产级场景化训练基础类，支持：
- 策略层集成（StrategyContext, StrategyMetrics）
- 硬件层集成（DeviceManager, MemoryManager）
- 分布式层集成（DistributedManager）
- 进度管理器集成（TrainingProgressManager）
- 配置验证和诊断

架构调用层次：
├── base_scenario.py (本模块)
│   ├── 调用 backend/modules/training/strategies (策略层)
│   ├── 调用 backend/lib/hardware (硬件层)
│   ├── 调用 backend/lib/distributed (分布式层)
│   └── 调用 backend/modules/training/progress (进度管理)
└── 被 basic_model_scenario.py, industry_scenario.py 等继承
"""

import logging
import os
import sys
import time
import json
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, project_root)

logger = logging.getLogger(__name__)


# ==================== 全局常量 ====================

STRATEGY_LAYER_AVAILABLE = True
HARDWARE_LAYER_AVAILABLE = True
DISTRIBUTED_LAYER_AVAILABLE = True
PROGRESS_MANAGER_AVAILABLE = True
DISTRIBUTED_STRATEGY_AVAILABLE = True
SCENARIO_STRATEGY_AVAILABLE = True

try:
    from backend.modules.training.strategies.base_strategy import StrategyContext
except ImportError:
    STRATEGY_LAYER_AVAILABLE = False

try:
    from backend.lib.hardware import DeviceManager
except ImportError:
    HARDWARE_LAYER_AVAILABLE = False

try:
    from backend.lib.distributed import DistributedManager
except ImportError:
    DISTRIBUTED_LAYER_AVAILABLE = False

try:
    from backend.modules.training.progress.progress_manager import TrainingProgressManager
except ImportError:
    PROGRESS_MANAGER_AVAILABLE = False

try:
    from backend.modules.training.strategies.distributed_strategy import DistributedStrategyConfig
except ImportError:
    DISTRIBUTED_STRATEGY_AVAILABLE = False

try:
    from backend.modules.training.strategies.scenario_strategy import ScenarioType
except ImportError:
    SCENARIO_STRATEGY_AVAILABLE = False


# ==================== 策略层导入 ====================
from backend.modules.training.strategies.base_strategy import (
    StrategyContext, StrategyMetrics,
    StrategyMonitor, StrategyProfiler,
)


# ==================== 分布式策略层导入 ====================

from backend.modules.training.strategies.distributed_strategy import (
    DistributedMode, DistributedStrategyConfig,
    recommend_distributed_mode,
)


# ==================== 场景策略层导入 ====================

from backend.modules.training.strategies.scenario_strategy import (
    ScenarioType,
)


# ==================== 硬件层导入 ====================

from backend.lib.hardware import (
    DeviceManager, get_device_manager,
    MemoryManager,
    get_available_memory, clear_memory,
)
# get_memory_manager 不存在，直接使用 MemoryManager


# ==================== 分布式层导入 ====================

from backend.lib.distributed import (
    DistributedManager, get_distributed_manager,
)


# ==================== 进度管理导入 ====================

from backend.modules.training.progress.progress_manager import (
    TrainingProgressManager, get_progress_manager,
)


# ==================== 编排器模块导入 ====================

from backend.modules.training.orchestrator import (
    UnifiedTrainingOrchestrator, LayerManager, LayerConfig,
    OrchestratorPlan,
    create_orchestrator,
)


# ==================== 流水线模块导入 ====================

from backend.modules.training.pipeline import (
    PipelineDefinition,
    PipelineExecutor, PipelineRunner,
    create_pipeline,
)


# ==================== 插件模块导入 ====================

from backend.modules.training.plugins import (
    TrainingPlugin,
    PluginRegistry, PluginContext, PluginResult, HookPoint,
    get_plugin_registry, execute_hook,
)


# ==================== 枚举定义 ====================

class TrainingStage(str, Enum):
    """训练阶段枚举"""
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    PREFERENCE = "preference"
    EVALUATION = "evaluation"
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        names = {
            TrainingStage.PRETRAIN: "预训练",
            TrainingStage.FINETUNE: "微调",
            TrainingStage.PREFERENCE: "偏好优化",
            TrainingStage.EVALUATION: "评估",
        }
        return names.get(self, self.value)
    
    @classmethod
    def from_string(cls, value: str) -> 'TrainingStage':
        """从字符串创建"""
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Unknown training stage: {value}")


class TrainingScenario(str, Enum):
    """训练场景枚举"""
    BASIC_MODEL = "basic_model"
    ADVANCED_MODEL = "advanced_model"
    SCHEDULED_TASK = "scheduled_task"
    RESEARCH_EXPERIMENT = "research_experiment"
    INDUSTRY_SCENARIO = "industry_scenario"
    MULTIMODAL = "multimodal"
    DISTILLATION = "distillation"
    THREE_STAGE = "three_stage"
    
    @property
    def display_name(self) -> str:
        """获取显示名称"""
        names = {
            TrainingScenario.BASIC_MODEL: "基础模型",
            TrainingScenario.ADVANCED_MODEL: "高级模型",
            TrainingScenario.SCHEDULED_TASK: "定时任务",
            TrainingScenario.RESEARCH_EXPERIMENT: "研究实验",
            TrainingScenario.INDUSTRY_SCENARIO: "行业场景",
            TrainingScenario.MULTIMODAL: "多模态",
            TrainingScenario.DISTILLATION: "知识蒸馏",
            TrainingScenario.THREE_STAGE: "三阶段训练",
        }
        return names.get(self, self.value)
    
    def to_scenario_type(self) -> Optional['ScenarioType']:
        """转换为策略层场景类型"""
        mapping = {
            TrainingScenario.BASIC_MODEL: ScenarioType.BASIC_MODEL if hasattr(ScenarioType, 'BASIC_MODEL') else None,
            TrainingScenario.ADVANCED_MODEL: ScenarioType.ADVANCED_MODEL if hasattr(ScenarioType, 'ADVANCED_MODEL') else None,
        }
        
        return mapping.get(self)


class ScenarioStatus(str, Enum):
    """场景状态枚举"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ==================== 配置类 ====================

@dataclass
class ScenarioConfigBase:
    """场景配置基类"""
    name: str = ""
    description: str = ""
    scenario: TrainingScenario = TrainingScenario.BASIC_MODEL
    output_dir: str = "./outputs/scenarios"
    
    # 策略配置
    use_strategy: bool = True
    strategy_type: str = "default"
    
    # 硬件配置
    device: str = "auto"
    use_fp16: bool = True
    
    # 分布式配置
    use_distributed: bool = False
    distributed_mode: str = "ddp"
    world_size: int = 1
    
    # 进度配置
    enable_progress_tracking: bool = True
    progress_update_interval: int = 100  # 步数
    
    # 回调配置
    enable_callbacks: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'scenario': self.scenario.value if isinstance(self.scenario, TrainingScenario) else self.scenario,
            'output_dir': self.output_dir,
            'use_strategy': self.use_strategy,
            'strategy_type': self.strategy_type,
            'device': self.device,
            'use_fp16': self.use_fp16,
            'use_distributed': self.use_distributed,
            'distributed_mode': self.distributed_mode,
            'world_size': self.world_size,
            'enable_progress_tracking': self.enable_progress_tracking,
            'progress_update_interval': self.progress_update_interval,
            'enable_callbacks': self.enable_callbacks,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioConfigBase':
        """从字典创建"""
        if 'scenario' in data and isinstance(data['scenario'], str):
            try:
                data['scenario'] = TrainingScenario(data['scenario'])
            except ValueError:
                data['scenario'] = TrainingScenario.BASIC_MODEL
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    def get_config_hash(self) -> str:
        """获取配置哈希值"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]


# ==================== 场景结果类 ====================

@dataclass
class ScenarioResult:
    """场景执行结果"""
    success: bool = False
    status: ScenarioStatus = ScenarioStatus.PENDING
    message: str = ""
    error: Optional[str] = None
    
    # 时间信息
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    # 训练结果
    model_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, Any] = field(default_factory=dict)
    
    # 场景信息
    session_id: str = ""
    scenario_type: str = ""
    stages_completed: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'success': self.success,
            'status': self.status.value if isinstance(self.status, ScenarioStatus) else self.status,
            'message': self.message,
            'error': self.error,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'model_path': self.model_path,
            'metrics': self.metrics,
            'history': self.history,
            'session_id': self.session_id,
            'scenario_type': self.scenario_type,
            'stages_completed': self.stages_completed,
        }


# ==================== 基础场景类 ====================

class BaseScenario(ABC):
    """基础训练场景抽象类
    
    生产级场景化训练基础类，集成多层架构：
    - 策略层：StrategyContext, StrategyMetrics
    - 硬件层：DeviceManager, MemoryManager
    - 分布式层：DistributedManager
    - 进度管理：TrainingProgressManager
    """
    
    def __init__(
        self,
        config: Union[ScenarioConfigBase, Dict[str, Any], Any],
        session_id: str = None
    ):
        # 处理配置
        if isinstance(config, dict):
            self.config = ScenarioConfigBase.from_dict(config)
        elif isinstance(config, ScenarioConfigBase):
            self.config = config
        else:
            # 兼容旧的配置格式
            self.config = config
        
        self.session_id = session_id or f"session_{int(time.time())}_{os.getpid()}"
        
        # 时间跟踪
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # 状态跟踪
        self.status = ScenarioStatus.PENDING
        self.current_stage: Optional[TrainingStage] = None
        
        # 训练统计
        self.training_stats: Dict[str, Any] = {
            'total_time': 0,
            'stages': {},
            'errors': [],
            'metrics': {},
            'layer_availability': get_layer_availability(),
        }
        
        # 回调
        self.callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # 策略层组件
        self._strategy_context: Optional['StrategyContext'] = None
        self._strategy_metrics: Optional['StrategyMetrics'] = None
        self._strategy_monitor: Optional['StrategyMonitor'] = None
        self._strategy_profiler: Optional['StrategyProfiler'] = None
        
        # 硬件层组件
        self._device_manager: Optional['DeviceManager'] = None
        self._memory_manager: Optional['MemoryManager'] = None
        
        # 分布式层组件
        self._distributed_manager: Optional['DistributedManager'] = None
        
        # 进度管理组件
        self._progress_manager: Optional['TrainingProgressManager'] = None
        
        # 编排器组件
        self._orchestrator: Optional['UnifiedTrainingOrchestrator'] = None
        self._layer_manager: Optional['LayerManager'] = None
        
        # 流水线组件
        self._pipeline: Optional['PipelineDefinition'] = None
        self._pipeline_executor: Optional['PipelineExecutor'] = None
        self._pipeline_runner: Optional['PipelineRunner'] = None
        
        # 插件组件
        self._plugin_registry: Optional['PluginRegistry'] = None
        self._registered_plugins: List[str] = []
        
        # 初始化各层组件
        self._init_components()
        
        # 创建输出目录
        if hasattr(self.config, 'output_dir'):
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        scenario_name = getattr(self.config, 'scenario', 'unknown')
        if hasattr(scenario_name, 'value'):
            scenario_name = scenario_name.value
        
        logger.info("Initialized training scenario: %s", scenario_name)
        logger.info("Session ID: %s", self.session_id)
    
    def _init_components(self) -> None:
        """初始化各层组件"""
        self._init_strategy_components()
        self._init_hardware_components()
        self._init_distributed_components()
        self._init_progress_manager()
        self._init_orchestrator()
        self._init_pipeline()
        self._init_plugins()
    
    def _init_strategy_components(self) -> None:
        """初始化策略层组件"""
        try:
            # 创建策略指标跟踪器
            if StrategyMetrics is not None:
                self._strategy_metrics = StrategyMetrics()
                logger.debug("Strategy metrics initialized")
            
            # 创建策略监控器
            if StrategyMonitor is not None:
                self._strategy_monitor = StrategyMonitor()
                logger.debug("Strategy monitor initialized")
            
            # 创建策略分析器
            if StrategyProfiler is not None:
                self._strategy_profiler = StrategyProfiler()
                logger.debug("Strategy profiler initialized")
                
        except Exception as e:
            logger.warning("Failed to initialize strategy components: %s", e)
    
    def _init_hardware_components(self) -> None:
        """初始化硬件层组件"""
        try:
            # 获取设备管理器
            if get_device_manager is not None:
                self._device_manager = get_device_manager()
                if self._device_manager is not None:
                    logger.debug("Device manager initialized")
            
            # 创建内存管理器实例
            # get_memory_manager 不存在，直接创建 MemoryManager 实例
            try:
                self._memory_manager = MemoryManager()
                logger.debug("Memory manager initialized")
            except Exception:
                self._memory_manager = None
            
            # 记录可用内存
            if get_available_memory is not None:
                try:
                    available_mem = get_available_memory()
                    self.training_stats['available_memory_mb'] = available_mem
                    logger.info("Available memory: %.0f MB", available_mem)
                except Exception:
                    pass
                    
        except Exception as e:
            logger.warning("Failed to initialize hardware components: %s", e)
    
    def _init_distributed_components(self) -> None:
        """初始化分布式层组件"""
        use_distributed = getattr(self.config, 'use_distributed', False)
        
        if not use_distributed:
            return
        
        try:
            self._distributed_manager = get_distributed_manager()
            if self._distributed_manager is not None:
                logger.info("Distributed manager initialized")
                    
        except Exception as e:
            logger.warning("Failed to initialize distributed components: %s", e)
    
    def _init_progress_manager(self) -> None:
        """初始化进度管理器"""
        enable_tracking = getattr(self.config, 'enable_progress_tracking', True)
        
        if not enable_tracking:
            return
        
        try:
            if get_progress_manager is not None:
                self._progress_manager = get_progress_manager()
            elif TrainingProgressManager is not None:
                self._progress_manager = TrainingProgressManager()
            
            if self._progress_manager is not None:
                logger.info("Progress manager initialized")
                
        except Exception as e:
            logger.warning("Failed to initialize progress manager: %s", e)
    
    def _init_orchestrator(self) -> None:
        """初始化编排器组件"""
        try:
            # 创建编排器
            if create_orchestrator is not None:
                output_dir = getattr(self.config, 'output_dir', './outputs')
                self._orchestrator = create_orchestrator(output_dir=output_dir)
                logger.debug("Orchestrator created")
            
            # 创建层管理器
            if LayerManager is not None and LayerConfig is not None:
                layer_config = LayerConfig(
                    device_type=getattr(self.config, 'device', 'auto'),
                    precision='fp16' if getattr(self.config, 'use_fp16', True) else 'fp32',
                    distributed_mode=getattr(self.config, 'distributed_mode', 'none'),
                    world_size=getattr(self.config, 'world_size', 1),
                )
                self._layer_manager = LayerManager(layer_config)
                logger.debug("Layer manager created")
                
        except Exception as e:
            logger.warning("Failed to initialize orchestrator: %s", e)
    
    def _init_pipeline(self) -> None:
        """初始化流水线组件"""
        try:
            # 创建流水线运行器
            if PipelineRunner is not None:
                self._pipeline_runner = PipelineRunner(session_id=self.session_id)
                logger.debug("Pipeline runner created")
                
        except Exception as e:
            logger.warning("Failed to initialize pipeline: %s", e)
    
    def _init_plugins(self) -> None:
        """初始化插件组件"""
        try:
            # 获取插件注册表
            if get_plugin_registry is not None:
                self._plugin_registry = get_plugin_registry()
                logger.debug("Plugin registry obtained")
                
        except Exception as e:
            logger.warning("Failed to initialize plugins: %s", e)
    
    def add_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """添加回调函数
        
        Args:
            callback: 回调函数，接收事件名称和数据字典
        """
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> bool:
        """移除回调函数
        
        Args:
            callback: 要移除的回调函数
        
        Returns:
            是否成功移除
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            return True
        return False
    
    def _trigger_callback(self, event: str, data: Dict[str, Any]) -> None:
        """触发回调函数
        
        Args:
            event: 事件名称
            data: 事件数据
        """
        enable_callbacks = getattr(self.config, 'enable_callbacks', True)
        if not enable_callbacks:
            return
        
        # 添加通用信息
        data['session_id'] = self.session_id
        data['timestamp'] = datetime.now().isoformat()
        data['status'] = self.status.value if isinstance(self.status, ScenarioStatus) else self.status
        
        for callback in self.callbacks:
            try:
                callback(event, data)
            except Exception as e:
                logger.error("Callback execution error: %s", e)
    
    def create_strategy_context(self, model=None, device=None) -> Optional['StrategyContext']:
        """创建策略上下文
        
        Args:
            model: 模型
            device: 设备
        
        Returns:
            策略上下文或 None
        """
        try:
            config_dict = self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
            
            # StrategyContext 是 dataclass，使用 extra 字段存储额外数据
            self._strategy_context = StrategyContext()
            self._strategy_context.model = model
            self._strategy_context.device = device
            self._strategy_context.config = config_dict
            self._strategy_context.extra = {
                'session_id': self.session_id,
                'scenario_type': getattr(self.config, 'scenario', 'unknown'),
            }
            
            return self._strategy_context
            
        except Exception as e:
            logger.warning("Failed to create strategy context: %s", e)
            return None
    
    def update_progress(
        self,
        stage: TrainingStage,
        epoch: int = 0,
        step: int = 0,
        metrics: Dict[str, Any] = None
    ) -> None:
        """更新训练进度
        
        Args:
            stage: 训练阶段
            epoch: 当前 epoch
            step: 当前步骤
            metrics: 指标字典
        """
        metrics = metrics or {}
        
        # 使用进度管理器
        if self._progress_manager is not None:
            try:
                # TrainingProgressManager 使用 update_progress 方法，而不是 update
                # TrainingProgress 是 dataclass，不接受构造函数参数
                self._progress_manager.update_progress(
                    session_id=self.session_id,
                    current_epoch=epoch,
                    current_step=step,
                    metrics=metrics,
                )
            except Exception as e:
                logger.warning("Failed to update progress via manager: %s", e)
        
        # 触发回调
        self._trigger_callback('progress', {
            'stage': stage.value,
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
        })
    
    def update_stats(self, stage: TrainingStage, metrics: Dict[str, Any]) -> None:
        """更新训练统计信息
        
        Args:
            stage: 训练阶段
            metrics: 指标字典
        """
        stage_value = stage.value if isinstance(stage, TrainingStage) else stage
        
        if stage_value not in self.training_stats['stages']:
            self.training_stats['stages'][stage_value] = []
        
        self.training_stats['stages'][stage_value].append(metrics)
        self.training_stats['metrics'].update(metrics)
        
        # 更新策略指标
        if self._strategy_metrics is not None:
            try:
                self._strategy_metrics.update(metrics)
            except Exception:
                pass
        
        logger.info("Updated training stats - stage: %s, metrics: %s", stage_value, metrics)
    
    # ==================== 编排器方法 ====================
    
    def create_orchestrator_plan(
        self,
        phases: List[str] = None,
        **kwargs
    ) -> Optional['OrchestratorPlan']:
        """创建编排器计划
        
        Args:
            phases: 阶段列表
            **kwargs: 其他配置
            
        Returns:
            编排器计划或 None
        """
        if self._orchestrator is None:
            return None
        
        try:
            name = getattr(self.config, 'name', 'scenario_plan')
            
            if phases is None:
                phases = []
                if getattr(self.config, 'enable_pretrain', False):
                    phases.append('pretrain')
                if getattr(self.config, 'enable_finetune', True):
                    phases.append('finetune')
                if getattr(self.config, 'enable_preference', False):
                    phases.append('preference')
            
            plan = self._orchestrator.create_plan(
                name=name,
                phases=phases,
                **kwargs
            )
            
            logger.info("Created orchestrator plan: %s, phases=%s", name, phases)
            return plan
            
        except Exception as e:
            logger.warning("Failed to create orchestrator plan: %s", e)
            return None
    
    def get_orchestrator(self) -> Optional['UnifiedTrainingOrchestrator']:
        """获取编排器实例"""
        return self._orchestrator
    
    def get_layer_manager(self) -> Optional['LayerManager']:
        """获取层管理器实例"""
        return self._layer_manager
    
    # ==================== 流水线方法 ====================
    
    def create_training_pipeline(
        self,
        steps: List[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional['PipelineDefinition']:
        """创建训练流水线
        
        Args:
            steps: 步骤列表
            **kwargs: 其他配置
            
        Returns:
            流水线定义或 None
        """
        try:
            name = f"{getattr(self.config, 'name', 'scenario')}_pipeline"
            
            if steps is None:
                steps = self._build_default_pipeline_steps()
            
            self._pipeline = create_pipeline(
                name=name,
                steps=steps,
                session_id=self.session_id,
                **kwargs
            )
            
            logger.info("Created pipeline: %s, steps=%d", name, len(steps))
            return self._pipeline
            
        except Exception as e:
            logger.warning("Failed to create pipeline: %s", e)
            return None
    
    def _build_default_pipeline_steps(self) -> List[Dict[str, Any]]:
        """构建默认流水线步骤"""
        steps = []
        
        if getattr(self.config, 'enable_pretrain', False):
            steps.append({
                'name': 'pretrain',
                'type': 'pretrain',
                'params': {
                    'num_epochs': getattr(self.config, 'num_epochs', 3),
                    'batch_size': getattr(self.config, 'batch_size', 16),
                },
                'on_fail': 'stop',
            })
        
        if getattr(self.config, 'enable_finetune', True):
            steps.append({
                'name': 'finetune',
                'type': 'finetune',
                'params': {
                    'num_epochs': getattr(self.config, 'num_epochs', 3),
                    'batch_size': getattr(self.config, 'batch_size', 16),
                    'learning_rate': getattr(self.config, 'learning_rate', 2e-5),
                },
                'on_fail': 'stop',
            })
        
        if getattr(self.config, 'enable_preference', False):
            steps.append({
                'name': 'preference',
                'type': 'preference',
                'params': {
                    'num_epochs': getattr(self.config, 'num_epochs', 1),
                },
                'on_fail': 'continue',
            })
        
        return steps
    
    def execute_pipeline(
        self,
        pipeline: 'PipelineDefinition' = None
    ) -> Optional[Dict[str, Any]]:
        """执行流水线
        
        Args:
            pipeline: 流水线定义，为空则使用当前流水线
            
        Returns:
            执行结果或 None
        """
        pipeline = pipeline or self._pipeline
        if pipeline is None:
            logger.warning("No pipeline to execute")
            return None
        
        try:
            if self._pipeline_runner is None and PipelineRunner is not None:
                self._pipeline_runner = PipelineRunner(session_id=self.session_id)
            
            if self._pipeline_executor is None and PipelineExecutor is not None:
                self._pipeline_executor = PipelineExecutor(
                    runner=self._pipeline_runner,
                    session_id=self.session_id,
                )
            
            if self._pipeline_executor is not None:
                result = self._pipeline_executor.execute(pipeline)
                logger.info("Pipeline executed: success=%s", result.success)
                return result.to_dict() if hasattr(result, 'to_dict') else {'success': result.success}
            
        except Exception as e:
            logger.error("Pipeline execution failed: %s", e)
            return {'success': False, 'error': str(e)}
        
        return None
    
    def get_pipeline(self) -> Optional['PipelineDefinition']:
        """获取当前流水线"""
        return self._pipeline
    
    # ==================== 插件方法 ====================
    
    def register_plugin(self, plugin: 'TrainingPlugin') -> bool:
        """注册插件
        
        Args:
            plugin: 插件实例
            
        Returns:
            是否成功注册
        """
        if self._plugin_registry is None:
            return False
        
        try:
            self._plugin_registry.register(plugin)
            self._registered_plugins.append(plugin.name)
            logger.info("Registered plugin: %s", plugin.name)
            return True
            
        except Exception as e:
            logger.warning("Failed to register plugin: %s", e)
            return False
    
    def execute_plugin_hook(
        self,
        hook: 'HookPoint',
        context_data: Dict[str, Any] = None
    ) -> List['PluginResult']:
        """执行插件钩子
        
        Args:
            hook: 钩子点
            context_data: 上下文数据
            
        Returns:
            插件执行结果列表
        """
        try:
            context_data = context_data or {}
            context_data['session_id'] = self.session_id
            context_data['scenario_type'] = getattr(self.config, 'scenario', 'unknown')
            
            if PluginContext is not None:
                context = PluginContext(
                    hook=hook,
                    session_id=self.session_id,
                    **context_data
                )
                results = execute_hook(hook, context)
                return results
            
        except Exception as e:
            logger.warning("Failed to execute plugin hook: %s", e)
        
        return []
    
    def trigger_plugin_event(
        self,
        event_name: str,
        **kwargs
    ) -> None:
        """触发插件事件
        
        Args:
            event_name: 事件名称
            **kwargs: 事件数据
        """
        # 映射事件名称到钩子点
        hook_mapping = {
            'training_start': HookPoint.ON_TRAINING_START if hasattr(HookPoint, 'ON_TRAINING_START') else None,
            'training_end': HookPoint.ON_TRAINING_END if hasattr(HookPoint, 'ON_TRAINING_END') else None,
            'epoch_start': HookPoint.ON_EPOCH_START if hasattr(HookPoint, 'ON_EPOCH_START') else None,
            'epoch_end': HookPoint.ON_EPOCH_END if hasattr(HookPoint, 'ON_EPOCH_END') else None,
            'step_start': HookPoint.ON_STEP_START if hasattr(HookPoint, 'ON_STEP_START') else None,
            'step_end': HookPoint.ON_STEP_END if hasattr(HookPoint, 'ON_STEP_END') else None,
            'stage_start': HookPoint.ON_STAGE_START if hasattr(HookPoint, 'ON_STAGE_START') else None,
            'stage_end': HookPoint.ON_STAGE_END if hasattr(HookPoint, 'ON_STAGE_END') else None,
        }
        
        hook = hook_mapping.get(event_name)
        if hook is not None:
            self.execute_plugin_hook(hook, kwargs)
    
    def get_registered_plugins(self) -> List[str]:
        """获取已注册的插件名称"""
        return self._registered_plugins.copy()
    
    @abstractmethod
    def run(self) -> Union[Dict[str, Any], ScenarioResult]:
        """运行训练场景
        
        Returns:
            训练结果
        """
        pass
    
    def pause(self) -> bool:
        """暂停训练
        
        Returns:
            是否成功暂停
        """
        if self.status != ScenarioStatus.RUNNING:
            logger.warning("Cannot pause: current status is %s", self.status)
            return False
        
        self.status = ScenarioStatus.PAUSED
        logger.info("Paused training scenario: %s", self.session_id)
        self._trigger_callback("paused", {"session_id": self.session_id})
        return True
    
    def resume(self) -> bool:
        """恢复训练
        
        Returns:
            是否成功恢复
        """
        if self.status != ScenarioStatus.PAUSED:
            logger.warning("Cannot resume: current status is %s", self.status)
            return False
        
        self.status = ScenarioStatus.RUNNING
        logger.info("Resumed training scenario: %s", self.session_id)
        self._trigger_callback("resumed", {"session_id": self.session_id})
        return True
    
    def cancel(self) -> bool:
        """取消训练
        
        Returns:
            是否成功取消
        """
        if self.status in [ScenarioStatus.COMPLETED, ScenarioStatus.FAILED, ScenarioStatus.CANCELLED]:
            logger.warning("Cannot cancel: current status is %s", self.status)
            return False
        
        self.status = ScenarioStatus.CANCELLED
        self.end_time = datetime.now()
        
        logger.info("Cancelled training scenario: %s", self.session_id)
        self._trigger_callback("cancelled", {"session_id": self.session_id})
        
        # 清理资源
        self.cleanup()
        
        return True
    
    def get_progress(self) -> Dict[str, Any]:
        """获取训练进度
        
        Returns:
            进度信息字典
        """
        elapsed_time = 0
        if self.start_time:
            if self.end_time:
                elapsed_time = (self.end_time - self.start_time).total_seconds()
            else:
                elapsed_time = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "session_id": self.session_id,
            "status": self.status.value if isinstance(self.status, ScenarioStatus) else self.status,
            "current_stage": self.current_stage.value if self.current_stage else None,
            "stats": self.training_stats,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "elapsed_seconds": elapsed_time,
            "layer_availability": get_layer_availability(),
        }
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断场景状态
        
        Returns:
            诊断结果字典
        """
        diagnosis = {
            'session_id': self.session_id,
            'status': self.status.value if isinstance(self.status, ScenarioStatus) else self.status,
            'layer_availability': get_layer_availability(),
            'errors': [],
            'warnings': [],
            'components': {
                'strategy_context': self._strategy_context is not None,
                'strategy_metrics': self._strategy_metrics is not None,
                'device_manager': self._device_manager is not None,
                'memory_manager': self._memory_manager is not None,
                'distributed_manager': self._distributed_manager is not None,
                'progress_manager': self._progress_manager is not None,
                'orchestrator': self._orchestrator is not None,
                'layer_manager': self._layer_manager is not None,
                'pipeline': self._pipeline is not None,
                'pipeline_executor': self._pipeline_executor is not None,
                'pipeline_runner': self._pipeline_runner is not None,
                'plugin_registry': self._plugin_registry is not None,
            },
            'registered_plugins': self._registered_plugins,
        }
        
        # 诊断分布式策略
        # diagnose_distributed_strategy 需要 strategy 参数
        # 这里没有策略实例，返回可用性信息
        try:
            # 检查是否有分布式管理器
            if self._distributed_manager is not None:
                diagnosis['distributed_diagnosis'] = {
                    'available': True,
                    'message': 'Distributed manager is available',
                    'manager_type': type(self._distributed_manager).__name__
                }
            else:
                diagnosis['distributed_diagnosis'] = {
                    'available': False,
                    'message': 'No distributed manager instance available'
                }
        except Exception as e:
            diagnosis['warnings'].append(f"Distributed diagnosis failed: {e}")
        
        # 诊断场景策略
        # diagnose_scenario_strategy 需要 strategy 参数
        # 这里没有策略实例，返回可用性信息
        try:
            # 检查是否有策略上下文
            if self._strategy_context is not None:
                diagnosis['scenario_diagnosis'] = {
                    'available': True,
                    'message': 'Strategy context is available',
                    'has_model': self._strategy_context.model is not None,
                    'device': str(self._strategy_context.device)
                }
            else:
                diagnosis['scenario_diagnosis'] = {
                    'available': False,
                    'message': 'No strategy context available'
                }
        except Exception as e:
            diagnosis['warnings'].append(f"Scenario diagnosis failed: {e}")
        
        # 获取内存信息
        try:
            diagnosis['available_memory_mb'] = get_available_memory()
        except Exception:
            pass
        
        return diagnosis
    
    def get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况
        
        Returns:
            内存使用信息字典
        """
        memory_info = {}
        
        try:
            memory_info['available_mb'] = get_available_memory()
        except Exception:
            pass
            
        try:
            if hasattr(self._memory_manager, 'get_usage'):
                memory_info['usage'] = self._memory_manager.get_usage()
        except Exception:
            pass
        
        # PyTorch 内存
        try:
            import torch
            if torch.cuda.is_available():
                memory_info['cuda_allocated_mb'] = torch.cuda.memory_allocated() / (1024 ** 2)
                memory_info['cuda_reserved_mb'] = torch.cuda.memory_reserved() / (1024 ** 2)
        except Exception:
            pass
        
        return memory_info
    
    def cleanup(self) -> None:
        """清理资源"""
        # 清理策略组件
        self._strategy_context = None
        self._strategy_metrics = None
        self._strategy_monitor = None
        self._strategy_profiler = None
        
        # 使用硬件层清理内存
        try:
            clear_memory()
            logger.debug("Memory cleared via hardware layer")
        except Exception as e:
            logger.warning("Failed to clear memory: %s", e)
        
        # PyTorch 内存清理
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        logger.info("Cleaned up resources for scenario: %s", self.session_id)
    
    def __enter__(self) -> 'BaseScenario':
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.cleanup()


# ==================== 便捷函数 ====================

def get_layer_availability() -> Dict[str, bool]:
    """获取各层可用性
    
    Returns:
        层可用性字典
    """
    return {
        'strategy_layer': STRATEGY_LAYER_AVAILABLE,
        'hardware_layer': HARDWARE_LAYER_AVAILABLE,
        'distributed_layer': DISTRIBUTED_LAYER_AVAILABLE,
        'progress_manager': PROGRESS_MANAGER_AVAILABLE,
        'distributed_strategy': DISTRIBUTED_STRATEGY_AVAILABLE,
        'scenario_strategy': SCENARIO_STRATEGY_AVAILABLE,
    }


def get_available_scenarios() -> List[str]:
    """获取可用的训练场景
    
    Returns:
        场景名称列表
    """
    return [s.value for s in TrainingScenario]


def get_available_stages() -> List[str]:
    """获取可用的训练阶段
    
    Returns:
        阶段名称列表
    """
    return [s.value for s in TrainingStage]


def create_scenario_result(
    success: bool,
    message: str = "",
    **kwargs
) -> ScenarioResult:
    """创建场景结果
    
    Args:
        success: 是否成功
        message: 消息
        **kwargs: 其他参数
    
    Returns:
        ScenarioResult 实例
    """
    return ScenarioResult(
        success=success,
        status=ScenarioStatus.COMPLETED if success else ScenarioStatus.FAILED,
        message=message,
        **kwargs
    )


# ==================== 导出 ====================

__all__ = [
    # 基础类
    'BaseScenario',
    
    # 枚举
    'TrainingStage',
    'TrainingScenario',
    'ScenarioStatus',
    
    # 配置和结果类
    'ScenarioConfigBase',
    'ScenarioResult',
    
    # 便捷函数
    'get_layer_availability',
    'get_available_scenarios',
    'get_available_stages',
    'create_scenario_result',
    
    # 常量
    'STRATEGY_LAYER_AVAILABLE',
    'HARDWARE_LAYER_AVAILABLE',
    'DISTRIBUTED_LAYER_AVAILABLE',
    'PROGRESS_MANAGER_AVAILABLE',
    'DISTRIBUTED_STRATEGY_AVAILABLE',
    'SCENARIO_STRATEGY_AVAILABLE',
]

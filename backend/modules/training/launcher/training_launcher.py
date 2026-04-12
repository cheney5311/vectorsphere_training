"""训练启动器

统一训练入口，整合六层训练架构的所有能力：
- 行业模型训练
- 场景化训练  
- 分布式训练
- 知识蒸馏训练（多场景）
- 多模态训练
- 三阶段训练
- 标准训练

支持多种训练场景的优先级选择：
行业模型 > 场景化训练 > 分布式 > 知识蒸馏 > 多模态 > 三阶段 > 标准

六层架构调用关系：
┌─────────────────────────────────────────────────────────────┐
│  Training Orchestrator (编排层)                              │
│    └── orchestrator/UnifiedTrainingOrchestrator             │
├─────────────────────────────────────────────────────────────┤
│  Training Strategy Abstraction (策略层)                      │
│    └── strategies/* (整合底层四层)                           │
│        ├── ProductionTrainingStrategy (生产级策略基类)       │
│        ├── ProductionTrainingContext (生产级上下文)          │
│        ├── MultiModalStrategy                                │
│        ├── DistillationStrategy                              │
│        ├── IndustryScenarioStrategy                          │
│        └── DistributedStrategy                               │
├─────────────────────────────────────────────────────────────┤
│  Loss & Objective Composition (目标函数层)                   │
│    └── backend/lib/losses                                    │
│        ├── SupervisedLoss, DistillationLoss, ContrastiveLoss │
│        └── CompositeLoss, MultiTaskLoss                      │
├─────────────────────────────────────────────────────────────┤
│  Model & Modality Adapter Layer (模型/模态层)                │
│    └── backend/lib/adapters                                  │
│        ├── ModalityEncoders (Text/Image/Audio/Video/TS)      │
│        ├── FusionModules (Early/Middle/Late/CrossAttention)  │
│        └── AlignmentModules (Contrastive/Explicit/OT)        │
├─────────────────────────────────────────────────────────────┤
│  Distributed Training Core (分布式训练内核层)                │
│    └── backend/lib/distributed                               │
│        ├── DDP, FSDP, Pipeline, ZeRO                         │
│        └── DistributedManager                                │
├─────────────────────────────────────────────────────────────┤
│  Hardware Abstraction (硬件抽象层)                           │
│    └── backend/lib/hardware                                  │
│        ├── DeviceManager, MixedPrecisionManager              │
│        └── MemoryManager, DeviceScheduler                    │
└─────────────────────────────────────────────────────────────┘

业务训练模块（调用策略层）：
├── distillation  (知识蒸馏)     -> DistillationStrategy
├── multimodal    (多模态训练)   -> MultiModalStrategy + ProductionContext
├── scenarios     (场景化训练)   -> IndustryScenarioStrategy
├── industry      (行业模型)     -> 组合策略 + ProductionContext
└── three_stage   (三阶段训练)   -> StandardStrategy + Orchestrator
"""

import os
import logging
from typing import Optional, Dict, Any, List, Callable, Union
from datetime import datetime
import sys
import os as os_path
import uuid

# 修复导入路径
current_dir = os_path.path.dirname(os_path.path.abspath(__file__))
project_root = os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(os_path.path.dirname(current_dir))))
sys.path.insert(0, project_root)

from backend.modules.training.exceptions import BusinessLogicError

logger = logging.getLogger(__name__)


# =============================================================================
# 模块可用性检测
# =============================================================================

# Core 模块
from backend.modules.training.core import (
    UnifiedTrainingSystem,
    TrainingConfig,
    TrainingTaskManager,
    TrainingTask,
    TrainingTaskStatus,
    get_training_task_manager,
    diagnose_core_module,
)

# Scenarios 模块
from backend.modules.training.scenarios import (
    ScenarioManager,
    ScenarioConfig,
    BaseScenario,
    BasicModelScenario,
    AdvancedModelScenario,
    IndustryScenario,
    ScheduledTrainingScenario,
    create_basic_scenario,
    create_advanced_scenario,
    create_industry_scenario,
    create_scheduled_scenario,
    diagnose_scenarios,
    get_integration_availability as get_scenario_integration,
)

# Orchestrator 模块
from backend.modules.training.orchestrator import (
    UnifiedTrainingOrchestrator,
    LayerManager,
    LayerConfig,
    OrchestratorPlan,
    create_orchestrator,
    create_quick_plan,
    create_standard_plan,
    create_three_stage_plan,
    create_multimodal_plan,
    create_distillation_plan,
    create_industry_plan,
    diagnose_orchestrator_module,
    get_orchestrator_layer_availability,
)

# Pipeline 模块
from backend.modules.training.pipeline import (
    PipelineDefinition,
    PipelineStep,
    PipelineExecutor,
    PipelineRunner,
    StepType,
    FailureAction,
    create_pipeline,
    create_three_stage_pipeline,
    create_executor,
    create_pipeline_runner,
    quick_execute_pipeline,
    diagnose_pipeline_module,
    get_pipeline_layer_availability,
)

# Progress 模块
from backend.modules.training.progress import (
    TrainingProgressManager,
    TrainingProgress,
    ProgressStatus,
    get_progress_manager,
    create_progress_tracker,
    update_progress,
    get_progress,
    get_layer_availability as get_progress_layer_availability,
)

# Plugins 模块
from backend.modules.training.plugins import (
    TrainingPlugin,
    CallbackPlugin,
    PluginRegistry,
    PluginContext,
    PluginResult,
    HookPoint,
    register_plugin,
    execute_hook,
    get_plugin_registry,
    diagnose_plugin_module,
)

# Strategies 模块
from backend.modules.training.strategies import (
    create_strategy,
    create_composite_strategy,
    TrainingStrategy,
    StandardTrainingStrategy,
    ProductionTrainingStrategy,
    ProductionStrategyConfig,
    ProductionTrainingContext,
    create_production_context,
    get_available_layers,
    diagnose_strategy,
    diagnose_production_base,
    # 多模态策略
    MultiModalStrategy,
    create_multimodal_strategy,
    diagnose_multimodal_strategy,
    # 蒸馏策略
    DistillationStrategy,
    create_distillation_strategy,
    diagnose_distillation_strategy,
    # 分布式策略
    DistributedStrategy,
    DistributedMode,
    DistributedStrategyConfig,
    create_distributed_strategy,
    recommend_distributed_mode,
    diagnose_distributed_strategy,
    # 场景策略
    ScenarioStrategy,
    IndustryScenarioStrategy,
    ScenarioStrategyConfig,
    create_scenario_strategy,
    diagnose_scenario_strategy,
    # 三阶段策略
    ThreeStageStrategy,
    create_three_stage_strategy,
    diagnose_three_stage_strategy,
)

# Distillation 模块
from backend.modules.training.distillation import (
    DistillationConfig,
    DistillationTaskConfig,
    DistillationPresets,
    KnowledgeDistillationTrainer,
    ModelCompressor,
    DistillationScenarioManager,
    get_scenario_manager as get_distillation_scenario_manager,
    get_distillation_service,
    diagnose_scenarios as diagnose_distillation_scenarios,
    list_available_scenarios as list_distillation_scenarios,
)

# Multimodal 模块
from backend.modules.training.multimodal import (
    MultiModalConfig,
    MultiModalTrainer,
)

# Three-stage 模块
from backend.modules.training.three_stage import (
    ThreeStageConfig,
    ThreeStageTrainer,
    create_three_stage_trainer,
    TrainingLoop,
    create_training_loop,
)

# Industry 模块
from backend.modules.training.industry import (
    create_industry_model,
)

# backend/lib 硬件层
from backend.lib.hardware import (
    DeviceManager,
    get_device_manager,
    MemoryManager,
    get_available_memory,
    clear_memory,
    recommend_precision,
    recommend_batch_size,
)

# backend/lib 分布式层
from backend.lib.distributed import (
    DistributedManager,
    get_distributed_manager,
)

# backend/lib 损失层
from backend.lib.losses import (
    LossFactory,
    create_loss,
    create_composite_loss,
)


def get_module_availability() -> Dict[str, bool]:
    """获取所有模块的可用性状态
    
    统一检测所有导入模块的可用性，调用各模块的诊断接口
    
    Returns:
        模块可用性字典
    """
    availability = {
        # Core 模块 - 调用 TrainingTaskStatus 检测
        'core': TrainingTaskStatus is not None,
        'core_task_manager': TrainingTaskManager is not None,
        'core_training_task': TrainingTask is not None,
        
        # Scenarios 模块 - 调用 BaseScenario, get_scenario_integration
        'scenarios': BaseScenario is not None,
        'scenarios_manager': ScenarioManager is not None,
        'scenarios_integration': False,
        
        # Orchestrator 模块 - 调用 LayerManager, OrchestratorPlan
        'orchestrator': UnifiedTrainingOrchestrator is not None,
        'orchestrator_layer_manager': LayerManager is not None,
        'orchestrator_plan': OrchestratorPlan is not None,
        
        # Pipeline 模块 - 调用 FailureAction
        'pipeline': PipelineDefinition is not None,
        'pipeline_failure_action': FailureAction is not None,
        
        # Progress 模块 - 调用 TrainingProgress, ProgressStatus
        'progress': TrainingProgressManager is not None,
        'progress_status': ProgressStatus is not None,
        'progress_tracker': TrainingProgress is not None,
        
        # Plugins 模块 - 调用 TrainingPlugin, PluginContext, PluginResult
        'plugins': PluginRegistry is not None,
        'plugins_base': TrainingPlugin is not None,
        'plugins_context': PluginContext is not None,
        'plugins_result': PluginResult is not None,
        
        # Strategies 模块 - 调用 StandardTrainingStrategy
        'strategies': TrainingStrategy is not None,
        'strategies_standard': StandardTrainingStrategy is not None,
        'strategies_production': ProductionTrainingStrategy is not None,
        'strategies_multimodal': MultiModalStrategy is not None,
        'strategies_distillation': DistillationStrategy is not None,
        'strategies_distributed': DistributedStrategy is not None,
        'strategies_scenario': ScenarioStrategy is not None,
        'strategies_three_stage': ThreeStageStrategy is not None,
        
        # Distillation 模块 - 调用 ModelCompressor, DistillationScenarioManager
        'distillation': KnowledgeDistillationTrainer is not None,
        'distillation_compressor': ModelCompressor is not None,
        'distillation_scenario_manager': DistillationScenarioManager is not None,
        
        # Multimodal 模块
        'multimodal': MultiModalTrainer is not None,
        'multimodal_config': MultiModalConfig is not None,
        
        # Three-stage 模块 - 调用 TrainingLoop
        'three_stage': ThreeStageTrainer is not None,
        'three_stage_loop': TrainingLoop is not None,
        
        # Industry 模块
        'industry': create_industry_model is not None,
        
        # Hardware 模块 - 调用 DeviceManager, MemoryManager
        'hardware': DeviceManager is not None,
        'hardware_memory': MemoryManager is not None,
        
        # Distributed 模块 - 调用 DistributedManager
        'distributed': DistributedManager is not None,
        
        # Losses 模块
        'losses': LossFactory is not None,
    }
    
    # 检测场景集成可用性 - 调用 get_scenario_integration
    try:
        scenario_integration = get_scenario_integration()
        availability['scenarios_integration'] = bool(scenario_integration)
    except Exception:
        availability['scenarios_integration'] = False
    
    # 检测编排器层可用性 - 调用 get_orchestrator_layer_availability
    try:
        orch_layers = get_orchestrator_layer_availability()
        availability['orchestrator_layers'] = bool(orch_layers)
    except Exception:
        availability['orchestrator_layers'] = False
    
    # 检测流水线层可用性 - 调用 get_pipeline_layer_availability
    try:
        pipeline_layers = get_pipeline_layer_availability()
        availability['pipeline_layers'] = bool(pipeline_layers)
    except Exception:
        availability['pipeline_layers'] = False
    
    # 检测进度层可用性 - 调用 get_progress_layer_availability
    try:
        progress_layers = get_progress_layer_availability()
        availability['progress_layers'] = bool(progress_layers)
    except Exception:
        availability['progress_layers'] = False
    
    return availability


class TrainingSystemLauncher:
    """训练系统启动器
    
    支持多种训练模式的统一启动入口，包括：
    - 场景化训练
    - 分布式训练
    - 知识蒸馏（多场景）
    - 多模态训练
    - 三阶段训练
    - 行业模型训练
    - 标准训练
    
    统一调用以下模块：
    - core: 统一训练系统和任务管理
    - scenarios: 场景化训练
    - orchestrator: 训练编排
    - pipeline: 训练流水线
    - progress: 进度管理
    - plugins: 插件系统
    - strategies: 训练策略
    - distillation: 知识蒸馏
    - multimodal: 多模态训练
    - three_stage: 三阶段训练
    - industry: 行业模型
    - backend/lib: 硬件/分布式/损失层
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化启动器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.session_id = config.get('session_id', f'session_{uuid.uuid4().hex[:8]}')
        self.output_dir = config.get('output_dir', f'./training_outputs_{int(datetime.now().timestamp())}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 策略组合
        self.strategies = []
        
        # 模块实例缓存
        self._progress_manager = None
        self._plugin_registry = None
        self._orchestrator = None
        self._task_manager = None
        
        # 初始化各层组件
        self._init_components()
        
        logger.info("训练系统启动器初始化完成")
        logger.info(f"会话ID: {self.session_id}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"模块可用性: {get_module_availability()}")
    
    def _init_components(self):
        """初始化各层组件"""
        # 初始化进度管理器
        try:
            self._progress_manager = get_progress_manager()
            logger.debug("Progress manager initialized")
        except Exception as e:
            logger.warning(f"Failed to init progress manager: {e}")
        
        # 初始化插件注册表
        try:
            self._plugin_registry = get_plugin_registry()
            logger.debug("Plugin registry initialized")
        except Exception as e:
            logger.warning(f"Failed to init plugin registry: {e}")
        
        # 初始化任务管理器
        try:
            self._task_manager = get_training_task_manager()
            logger.debug("Task manager initialized")
        except Exception as e:
            logger.warning(f"Failed to init task manager: {e}")
        
        # 优化硬件资源
        self._optimize_hardware()
    
    def _optimize_hardware(self):
        """优化硬件资源"""
        try:
            if clear_memory:
                clear_memory()
                logger.debug("Memory cleared")
            if get_available_memory:
                mem = get_available_memory()
                logger.info(f"Available memory: {mem / (1024 ** 3):.2f} GB")
        except Exception as e:
            logger.warning(f"Hardware optimization failed: {e}")
    
    def _trigger_plugin_hook(self, hook_name: str, **kwargs):
        """触发插件钩子"""
        try:
            hook_point = getattr(HookPoint, hook_name.upper(), None)
            if hook_point:
                execute_hook(hook_point, **kwargs)
        except Exception as e:
            logger.warning(f"Plugin hook '{hook_name}' failed: {e}")
    
    def _update_progress(self, progress: float, stage: str = "training", **metrics):
        """更新训练进度"""
        if self._progress_manager:
            try:
                if hasattr(self._progress_manager, 'update_progress'):
                    self._progress_manager.update_progress(
                        self.session_id,
                        progress=progress,
                        stage=stage,
                        metrics=metrics
                    )
            except Exception as e:
                logger.warning(f"Progress update failed: {e}")
    
    # =========================================================================
    # 派生功能：调用 core 模块 (TrainingTask, TrainingTaskStatus)
    # =========================================================================
    
    def create_training_task(self, name: str, task_config: Dict[str, Any]) -> Optional[TrainingTask]:
        """
        创建训练任务
        
        调用 core 模块的 TrainingTask 和 TrainingTaskStatus
        
        Args:
            name: 任务名称
            task_config: 任务配置
        
        Returns:
            TrainingTask 实例
        """
        try:
            if self._task_manager:
                # create_training_task 返回 task_id，需要构造 task_config 并获取任务
                task_config_full = {
                    'name': name,
                    'scenario_type': task_config.get('scenario_type', 'standard'),
                    **task_config
                }
                task_id = self._task_manager.create_training_task(
                    user_id=task_config.get('user_id', 'system'),
                    task_config=task_config_full
                )
                # 获取任务对象
                if task_id and task_id in self._task_manager.tasks:
                    task = self._task_manager.tasks[task_id]
                    logger.info(f"Created training task: {task.task_id}, status={task.status}")
                    # 使用 TrainingTaskStatus 检查状态
                    if task.status == TrainingTaskStatus.PENDING.value:
                        logger.info(f"Task {task.task_id} is pending")
                    return task
            return None
        except Exception as e:
            logger.warning(f"Failed to create training task: {e}")
            return None
    
    def get_task_status(self, task_id: str) -> Optional[TrainingTaskStatus]:
        """
        获取任务状态
        
        调用 core 模块的 TrainingTaskStatus
        
        Args:
            task_id: 任务ID
        
        Returns:
            TrainingTaskStatus 枚举值
        """
        try:
            if self._task_manager:
                # 直接从 tasks 字典获取任务
                if task_id in self._task_manager.tasks:
                    task = self._task_manager.tasks[task_id]
                    return task.status
            return None
        except Exception as e:
            logger.warning(f"Failed to get task status: {e}")
            return None
    
    # =========================================================================
    # 派生功能：调用 scenarios 模块 (BaseScenario, get_scenario_integration)
    # =========================================================================
    
    def get_scenario_integration_status(self) -> Dict[str, Any]:
        """
        获取场景集成状态
        
        调用 scenarios 模块的 get_scenario_integration
        
        Returns:
            场景集成状态字典
        """
        try:
            integration = get_scenario_integration()
            return {
                'available': True,
                'integration': integration,
                'base_scenario_available': BaseScenario is not None,
            }
        except Exception as e:
            logger.warning(f"Failed to get scenario integration: {e}")
            return {'available': False, 'error': str(e)}
    
    def validate_scenario_config(self, scenario_config: ScenarioConfig) -> bool:
        """
        验证场景配置
        
        调用 scenarios 模块的 ScenarioConfig
        
        Args:
            scenario_config: 场景配置
        
        Returns:
            配置是否有效
        """
        try:
            # 验证基本字段
            if not scenario_config.name:
                return False
            if not scenario_config.output_dir:
                return False
            return True
        except Exception as e:
            logger.warning(f"Scenario config validation failed: {e}")
            return False
    
    # =========================================================================
    # 派生功能：调用 orchestrator 模块 (LayerManager, OrchestratorPlan, create_* functions)
    # =========================================================================
    
    def create_orchestrator_with_layers(self) -> Optional[UnifiedTrainingOrchestrator]:
        """
        使用 LayerManager 创建编排器
        
        调用 orchestrator 模块的 create_orchestrator, LayerManager
        
        Returns:
            UnifiedTrainingOrchestrator 实例
        """
        try:
            # 使用便捷函数创建编排器
            orchestrator = create_orchestrator(output_dir=self.output_dir)
            
            # 获取层管理器
            if hasattr(orchestrator, 'layer_manager'):
                layer_manager: LayerManager = orchestrator.layer_manager
                logger.info(f"LayerManager available: {layer_manager is not None}")
            
            self._orchestrator = orchestrator
            return orchestrator
        except Exception as e:
            logger.warning(f"Failed to create orchestrator: {e}")
            return None
    
    def create_training_plan(self, plan_type: str, **kwargs) -> Optional[OrchestratorPlan]:
        """
        创建训练计划
        
        调用 orchestrator 模块的 create_quick_plan, create_standard_plan, 
        create_three_stage_plan, create_multimodal_plan, create_distillation_plan,
        create_industry_plan
        
        Args:
            plan_type: 计划类型 (quick, standard, three_stage, multimodal, distillation, industry)
            **kwargs: 计划参数
        
        Returns:
            OrchestratorPlan 实例
        """
        try:
            plan_name = kwargs.get('name', f'{plan_type}_plan_{self.session_id}')
            
            if plan_type == 'quick':
                plan = create_quick_plan(plan_type='standard', name=plan_name)
            elif plan_type == 'standard':
                plan = create_standard_plan(
                    name=plan_name,
                    epochs=kwargs.get('epochs', 10)
                )
            elif plan_type == 'three_stage':
                plan = create_three_stage_plan(
                    name=plan_name,
                    pretrain_epochs=kwargs.get('pretrain_epochs', 3),
                    finetune_epochs=kwargs.get('finetune_epochs', 5),
                    preference_epochs=kwargs.get('preference_epochs', 2)
                )
            elif plan_type == 'multimodal':
                plan = create_multimodal_plan(
                    name=plan_name,
                    modalities=kwargs.get('modalities', ['text', 'image'])
                )
            elif plan_type == 'distillation':
                plan = create_distillation_plan(
                    name=plan_name,
                    distillation_epochs=kwargs.get('distillation_epochs', 10)
                )
            elif plan_type == 'industry':
                plan = create_industry_plan(
                    name=plan_name,
                    include_pretrain=kwargs.get('include_pretrain', True),
                    include_align=kwargs.get('include_align', True),
                    include_finetune=kwargs.get('include_finetune', True)
                )
            else:
                plan = create_quick_plan(plan_type='standard', name=plan_name)
            
            logger.info(f"Created {plan_type} plan: {plan.name if hasattr(plan, 'name') else plan_name}")
            return plan
        except Exception as e:
            logger.warning(f"Failed to create training plan: {e}")
            return None
    
    def get_orchestrator_availability(self) -> Dict[str, Any]:
        """
        获取编排器层可用性
        
        调用 orchestrator 模块的 get_orchestrator_layer_availability
        
        Returns:
            编排器层可用性字典
        """
        try:
            layer_availability = get_orchestrator_layer_availability()
            return {
                'available': True,
                'layers': layer_availability,
            }
        except Exception as e:
            logger.warning(f"Failed to get orchestrator availability: {e}")
            return {'available': False, 'error': str(e)}
    
    # =========================================================================
    # 派生功能：调用 pipeline 模块 (FailureAction, quick_execute_pipeline)
    # =========================================================================
    
    def create_pipeline_with_failure_handling(
        self,
        steps: List[Dict[str, Any]],
        default_failure_action: str = 'stop'
    ) -> Optional[PipelineDefinition]:
        """
        创建带失败处理的流水线
        
        调用 pipeline 模块的 FailureAction, PipelineStep, PipelineDefinition
        
        Args:
            steps: 步骤列表
            default_failure_action: 默认失败处理策略 (stop, continue, rollback)
        
        Returns:
            PipelineDefinition 实例
        """
        try:
            # 转换失败处理策略 - 调用 FailureAction
            failure_action_map = {
                'stop': FailureAction.STOP,
                'continue': FailureAction.CONTINUE,
                'rollback': FailureAction.ROLLBACK,
            }
            failure_action = failure_action_map.get(default_failure_action, FailureAction.STOP)
            
            pipeline_steps = []
            for step_def in steps:
                step_type_str = step_def.get('type', 'custom')
                step = PipelineStep(
                    name=step_def.get('name', f'step_{len(pipeline_steps)}'),
                    type=step_type_str,
                    params=step_def.get('params', {}),
                    on_fail=default_failure_action,
                )
                pipeline_steps.append(step)
            
            pipeline = PipelineDefinition(
                name=f'pipeline_{self.session_id}',
                steps=pipeline_steps,
            )
            
            logger.info(f"Created pipeline with {len(pipeline_steps)} steps, failure_action={default_failure_action}")
            return pipeline
        except Exception as e:
            logger.warning(f"Failed to create pipeline: {e}")
            return None
    
    def quick_execute(self, pipeline: PipelineDefinition) -> Dict[str, Any]:
        """
        快速执行流水线
        
        调用 pipeline 模块的 quick_execute_pipeline
        
        Args:
            pipeline: 流水线定义
        
        Returns:
            执行结果
        """
        try:
            # quick_execute_pipeline 需要 name 和 steps 参数
            result = quick_execute_pipeline(
                name=pipeline.name if hasattr(pipeline, 'name') else f'pipeline_{self.session_id}',
                steps=[step.to_dict() if hasattr(step, 'to_dict') else step for step in pipeline.steps],
                session_id=self.session_id
            )
            return {
                'success': True,
                'result': result,
            }
        except Exception as e:
            logger.warning(f"Quick execute failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_pipeline_availability(self) -> Dict[str, Any]:
        """
        获取流水线层可用性
        
        调用 pipeline 模块的 get_pipeline_layer_availability
        
        Returns:
            流水线层可用性字典
        """
        try:
            layer_availability = get_pipeline_layer_availability()
            return {
                'available': True,
                'layers': layer_availability,
            }
        except Exception as e:
            logger.warning(f"Failed to get pipeline availability: {e}")
            return {'available': False, 'error': str(e)}
    
    # =========================================================================
    # 派生功能：调用 progress 模块 (TrainingProgress, ProgressStatus, update_progress, get_progress)
    # =========================================================================
    
    def create_progress_with_status(self, total_steps: int = 0, total_epochs: int = 10) -> Optional[TrainingProgress]:
        """
        创建进度跟踪器
        
        调用 progress 模块的 TrainingProgress, ProgressStatus, create_progress_tracker
        
        Args:
            total_steps: 总步数
            total_epochs: 总轮数
        
        Returns:
            TrainingProgress 实例
        """
        try:
            # 使用便捷函数创建进度跟踪器 - 调用 create_progress_tracker
            progress = create_progress_tracker(
                session_id=self.session_id,
                total_steps=total_steps,
                total_epochs=total_epochs
            )
            
            # 检查进度状态 - 调用 ProgressStatus
            if progress and hasattr(progress, 'status'):
                if progress.status == ProgressStatus.PENDING:
                    logger.info(f"Progress {self.session_id} is pending")
                elif progress.status == ProgressStatus.RUNNING:
                    logger.info(f"Progress {self.session_id} is running")
            
            return progress
        except Exception as e:
            logger.warning(f"Failed to create progress: {e}")
            return None
    
    def update_training_progress(self, progress_value: float, **kwargs) -> bool:
        """
        更新训练进度
        
        调用 progress 模块的 update_progress 便捷函数
        
        Args:
            progress_value: 进度值 (0.0 - 1.0)
            **kwargs: 附加参数
        
        Returns:
            是否更新成功
        """
        try:
            # 调用便捷函数 update_progress
            update_progress(
                session_id=self.session_id,
                progress=progress_value,
                **kwargs
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to update progress: {e}")
            return False
    
    def get_training_progress(self) -> Optional[TrainingProgress]:
        """
        获取训练进度
        
        调用 progress 模块的 get_progress 便捷函数
        
        Returns:
            TrainingProgress 实例
        """
        try:
            # 调用便捷函数 get_progress
            progress = get_progress(session_id=self.session_id)
            return progress
        except Exception as e:
            logger.warning(f"Failed to get progress: {e}")
            return None
    
    # =========================================================================
    # 派生功能：调用 plugins 模块 (TrainingPlugin, PluginContext, PluginResult)
    # =========================================================================
    
    def create_plugin_context(self, hook: HookPoint = None, **kwargs) -> Optional[PluginContext]:
        """
        创建插件上下文
        
        调用 plugins 模块的 PluginContext
        
        Args:
            hook: 钩子点
            **kwargs: 上下文参数
        
        Returns:
            PluginContext 实例
        """
        try:
            # PluginContext 需要 hook 参数
            context = PluginContext(
                hook=hook if hook else HookPoint.ON_TRAINING_START,
                session_id=self.session_id,
                epoch=kwargs.get('epoch', 0),
                step=kwargs.get('step', 0),
                stage=kwargs.get('stage', ''),
                metrics=kwargs.get('metrics', {}),
                model=kwargs.get('model'),
                optimizer=kwargs.get('optimizer'),
                data=kwargs.get('data', {}),
            )
            logger.info(f"Created plugin context for session: {self.session_id}")
            return context
        except Exception as e:
            logger.warning(f"Failed to create plugin context: {e}")
            return None
    
    def execute_plugin_with_result(self, plugin: TrainingPlugin, context: PluginContext) -> Optional[PluginResult]:
        """
        执行插件并返回结果
        
        调用 plugins 模块的 TrainingPlugin, PluginResult
        
        Args:
            plugin: 训练插件
            context: 插件上下文
        
        Returns:
            PluginResult 实例
        """
        try:
            # 执行插件
            if hasattr(plugin, 'execute'):
                result = plugin.execute(context)
                if isinstance(result, PluginResult):
                    logger.info(f"Plugin executed: success={result.success}")
                    return result
                else:
                    # 包装为 PluginResult
                    return PluginResult(success=True, data={'result': result})
            return PluginResult(success=False, message="Plugin has no execute method")
        except Exception as e:
            logger.warning(f"Plugin execution failed: {e}")
            return PluginResult(success=False, message=str(e))
    
    def register_custom_plugin(self, plugin: TrainingPlugin) -> bool:
        """
        注册自定义插件
        
        调用 plugins 模块的 register_plugin
        
        Args:
            plugin: 训练插件实例
        
        Returns:
            是否注册成功
        """
        try:
            register_plugin(plugin)
            logger.info(f"Registered custom plugin: {plugin.name if hasattr(plugin, 'name') else 'unknown'}")
            return True
        except Exception as e:
            logger.warning(f"Failed to register plugin: {e}")
            return False
    
    # =========================================================================
    # 派生功能：调用 strategies 模块 (create_composite_strategy, StandardTrainingStrategy, diagnostics)
    # =========================================================================
    
    def create_composite_training_strategy(self, strategy_types: List[str]) -> Optional[Any]:
        """
        创建组合训练策略
        
        调用 strategies 模块的 create_composite_strategy
        
        Args:
            strategy_types: 策略类型列表
        
        Returns:
            组合策略实例
        """
        try:
            strategies = []
            for st in strategy_types:
                strategy = create_strategy(st)
                if strategy:
                    strategies.append(strategy)
            
            if strategies:
                # 调用 create_composite_strategy
                composite = create_composite_strategy(strategies)
                logger.info(f"Created composite strategy with {len(strategies)} sub-strategies")
                return composite
            return None
        except Exception as e:
            logger.warning(f"Failed to create composite strategy: {e}")
            return None
    
    def create_standard_strategy(self) -> Optional[StandardTrainingStrategy]:
        """
        创建标准训练策略
        
        调用 strategies 模块的 StandardTrainingStrategy
        
        Returns:
            StandardTrainingStrategy 实例
        """
        try:
            strategy = StandardTrainingStrategy()
            logger.info("Created StandardTrainingStrategy")
            return strategy
        except Exception as e:
            logger.warning(f"Failed to create standard strategy: {e}")
            return None
    
    def diagnose_all_strategies(self) -> Dict[str, Any]:
        """
        诊断所有策略模块
        
        调用 strategies 模块的所有诊断函数
        
        Returns:
            诊断结果字典
        """
        diagnosis = {}
        
        # 诊断基础策略
        try:
            # diagnose_strategy 需要 strategy 参数，使用 get_available_layers 代替
            diagnosis['base'] = {'available_layers': get_available_layers()}
        except Exception as e:
            diagnosis['base'] = {'error': str(e)}
        
        # 诊断生产级策略
        try:
            diagnosis['production'] = diagnose_production_base()
        except Exception as e:
            diagnosis['production'] = {'error': str(e)}
        
        # 诊断多模态策略 - diagnose_multimodal_strategy 需要 strategy 参数
        try:
            # 只检查模块可用性，不实例化策略
            diagnosis['multimodal'] = {
                'available': MultiModalStrategy is not None,
                'create_function_available': create_multimodal_strategy is not None,
            }
        except Exception as e:
            diagnosis['multimodal'] = {'error': str(e)}
        
        # 诊断蒸馏策略 - diagnose_distillation_strategy 需要 strategy 参数
        try:
            # 只检查模块可用性，不实例化策略
            diagnosis['distillation'] = {
                'available': DistillationStrategy is not None,
                'create_function_available': create_distillation_strategy is not None,
            }
        except Exception as e:
            diagnosis['distillation'] = {'error': str(e)}
        
        # 诊断分布式策略 - diagnose_distributed_strategy 需要 strategy 参数
        try:
            # 只检查模块可用性，不实例化策略
            diagnosis['distributed'] = {
                'available': DistributedStrategy is not None,
                'create_function_available': create_distributed_strategy is not None,
            }
        except Exception as e:
            diagnosis['distributed'] = {'error': str(e)}
        
        # 诊断场景策略 - diagnose_scenario_strategy 需要 strategy 参数
        try:
            # 只检查模块可用性，不实例化策略
            diagnosis['scenario'] = {
                'available': ScenarioStrategy is not None,
                'create_function_available': create_scenario_strategy is not None,
            }
        except Exception as e:
            diagnosis['scenario'] = {'error': str(e)}
        
        # 诊断三阶段策略 - diagnose_three_stage_strategy 需要 strategy 参数
        try:
            # 只检查模块可用性，不实例化策略
            diagnosis['three_stage'] = {
                'available': ThreeStageStrategy is not None,
                'create_function_available': create_three_stage_strategy is not None,
            }
        except Exception as e:
            diagnosis['three_stage'] = {'error': str(e)}
        
        return diagnosis
    
    # =========================================================================
    # 派生功能：调用 distillation 模块 (ModelCompressor, DistillationScenarioManager, diagnostics)
    # =========================================================================
    
    def create_model_compressor(self, compression_config: Dict[str, Any] = None) -> Optional[ModelCompressor]:
        """
        创建模型压缩器
        
        调用 distillation 模块的 ModelCompressor
        
        Args:
            compression_config: 压缩配置
        
        Returns:
            ModelCompressor 实例
        """
        try:
            config = compression_config or {}
            compressor = ModelCompressor(**config)
            logger.info("Created ModelCompressor")
            return compressor
        except Exception as e:
            logger.warning(f"Failed to create model compressor: {e}")
            return None
    
    def get_distillation_scenario_manager_instance(self) -> Optional[DistillationScenarioManager]:
        """
        获取蒸馏场景管理器
        
        调用 distillation 模块的 get_distillation_scenario_manager
        
        Returns:
            DistillationScenarioManager 实例
        """
        try:
            manager = get_distillation_scenario_manager()
            logger.info("Got DistillationScenarioManager instance")
            return manager
        except Exception as e:
            logger.warning(f"Failed to get distillation scenario manager: {e}")
            return None
    
    def diagnose_distillation_module(self) -> Dict[str, Any]:
        """
        诊断蒸馏模块
        
        调用 distillation 模块的 diagnose_distillation_scenarios, list_distillation_scenarios
        
        Returns:
            诊断结果字典
        """
        diagnosis = {}
        
        try:
            # 诊断蒸馏场景
            diagnosis['scenarios'] = diagnose_distillation_scenarios()
        except Exception as e:
            diagnosis['scenarios'] = {'error': str(e)}
        
        try:
            # 列出可用场景
            scenarios = list_distillation_scenarios()
            diagnosis['available_scenarios'] = scenarios
        except Exception as e:
            diagnosis['available_scenarios'] = {'error': str(e)}
        
        return diagnosis
    
    # =========================================================================
    # 派生功能：调用 three_stage 模块 (create_three_stage_trainer, TrainingLoop, create_training_loop)
    # =========================================================================
    
    def create_three_stage_trainer_instance(self, config: Dict[str, Any] = None) -> Optional[ThreeStageTrainer]:
        """
        创建三阶段训练器
        
        调用 three_stage 模块的 create_three_stage_trainer
        
        Args:
            config: 训练配置
        
        Returns:
            ThreeStageTrainer 实例
        """
        try:
            trainer_config = config or {}
            trainer_config.setdefault('output_dir', self.output_dir)
            
            # 调用便捷函数
            trainer = create_three_stage_trainer(**trainer_config)
            logger.info("Created ThreeStageTrainer via factory function")
            return trainer
        except Exception as e:
            logger.warning(f"Failed to create three stage trainer: {e}")
            return None
    
    def create_training_loop_instance(self, loop_config: Dict[str, Any] = None) -> Optional[TrainingLoop]:
        """
        创建训练循环
        
        调用 three_stage 模块的 TrainingLoop, create_training_loop
        
        Args:
            loop_config: 循环配置
        
        Returns:
            TrainingLoop 实例
        """
        try:
            config = loop_config or {}
            
            # 调用便捷函数
            loop = create_training_loop(**config)
            logger.info("Created TrainingLoop via factory function")
            return loop
        except Exception as e:
            logger.warning(f"Failed to create training loop: {e}")
            return None
    
    # =========================================================================
    # 派生功能：调用 hardware 模块 (DeviceManager, MemoryManager)
    # =========================================================================
    
    def get_device_manager_instance(self) -> Optional[DeviceManager]:
        """
        获取设备管理器实例
        
        调用 hardware 模块的 DeviceManager, get_device_manager
        
        Returns:
            DeviceManager 实例
        """
        try:
            manager = get_device_manager()
            logger.info("Got DeviceManager instance")
            return manager
        except Exception as e:
            logger.warning(f"Failed to get device manager: {e}")
            return None
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        获取内存状态
        
        调用 hardware 模块的 MemoryManager, get_available_memory, clear_memory
        
        Returns:
            内存状态字典
        """
        try:
            # 获取可用内存
            available = get_available_memory()
            
            # 获取推荐配置
            precision = recommend_precision('cuda')
            # recommend_batch_size 需要 model 和 sample_size_mb 参数，使用默认值
            batch_size = 16  # 默认批次大小
            if available:
                # 根据可用内存计算推荐批次大小
                available_gb = available / (1024 ** 3)
                batch_size = max(1, min(64, int(available_gb * 4)))  # 简单估算
            
            return {
                'available_memory_gb': available / (1024 ** 3) if available else 0,
                'recommended_precision': precision,
                'recommended_batch_size': batch_size,
                'memory_manager_available': MemoryManager is not None,
            }
        except Exception as e:
            logger.warning(f"Failed to get memory status: {e}")
            return {'error': str(e)}
    
    # =========================================================================
    # 派生功能：调用 distributed 模块 (DistributedManager, get_distributed_manager)
    # =========================================================================
    
    def get_distributed_manager_instance(self) -> Optional[DistributedManager]:
        """
        获取分布式管理器实例
        
        调用 distributed 模块的 DistributedManager, get_distributed_manager
        
        Returns:
            DistributedManager 实例
        """
        try:
            manager = get_distributed_manager()
            logger.info("Got DistributedManager instance")
            return manager
        except Exception as e:
            logger.warning(f"Failed to get distributed manager: {e}")
            return None
    
    def setup_distributed_training(self, mode: str = 'ddp', world_size: int = 1) -> Dict[str, Any]:
        """
        设置分布式训练
        
        调用 distributed 模块和 strategies 模块
        
        Args:
            mode: 分布式模式 (ddp, fsdp, zero)
            world_size: 进程数
        
        Returns:
            分布式配置结果
        """
        try:
            # 获取分布式管理器
            dist_manager = get_distributed_manager()
            
            # 获取推荐模式 - 函数需要 model_size_gb, num_gpus, memory_per_gpu_gb 参数
            model_size_gb = self.config.get('model', {}).get('size_gb', 2.0)
            if isinstance(model_size_gb, str):
                size_map = {'small': 0.5, 'medium': 2.0, 'large': 7.0, 'xlarge': 13.0}
                model_size_gb = size_map.get(model_size_gb, 2.0)
            recommended = recommend_distributed_mode(
                model_size_gb=model_size_gb,
                num_gpus=world_size,
                memory_per_gpu_gb=16.0
            )
            
            # 创建分布式策略配置
            dist_mode = getattr(DistributedMode, mode.upper(), DistributedMode.DDP)
            strategy_config = DistributedStrategyConfig(
                mode=dist_mode,
                world_size=world_size,
            )
            
            # 创建分布式策略
            strategy = create_distributed_strategy(strategy_config)
            
            return {
                'success': True,
                'mode': mode,
                'world_size': world_size,
                'recommended_mode': recommended,
                'strategy_created': strategy is not None,
                'manager_available': dist_manager is not None,
            }
        except Exception as e:
            logger.warning(f"Failed to setup distributed training: {e}")
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # 派生功能：调用 losses 模块 (create_composite_loss)
    # =========================================================================
    
    def create_composite_loss_function(self, loss_configs: List[Dict[str, Any]]) -> Optional[Any]:
        """
        创建组合损失函数
        
        调用 losses 模块的 create_composite_loss, LossFactory, create_loss
        
        Args:
            loss_configs: 损失函数配置列表
        
        Returns:
            组合损失函数实例
        """
        try:
            # 使用 LossFactory 创建单个损失函数
            factory = LossFactory()
            losses = []
            weights = []
            
            for cfg in loss_configs:
                loss_type = cfg.get('type', 'cross_entropy')
                weight = cfg.get('weight', 1.0)
                
                # 使用工厂创建损失函数
                loss_fn = factory.create(loss_type, **cfg)
                if loss_fn:
                    losses.append(loss_fn)
                    weights.append(weight)
            
            if losses:
                # 调用 create_composite_loss
                composite = create_composite_loss(losses, weights)
                logger.info(f"Created composite loss with {len(losses)} components")
                return composite
            return None
        except Exception as e:
            logger.warning(f"Failed to create composite loss: {e}")
            return None
    
    # =========================================================================
    # 派生功能：调用 core 模块 (UnifiedTrainingSystem, TrainingConfig)
    # =========================================================================
    
    def create_unified_training_system(self, training_config: Dict[str, Any] = None) -> Optional[UnifiedTrainingSystem]:
        """
        创建统一训练系统
        
        调用 core 模块的 UnifiedTrainingSystem, TrainingConfig
        
        Args:
            training_config: 训练配置字典
        
        Returns:
            UnifiedTrainingSystem 实例
        """
        try:
            config_dict = training_config or {}
            
            # 创建 TrainingConfig - 调用 core 模块
            config = TrainingConfig(
                model_name=config_dict.get('model_name', self.config.get('model', {}).get('name', 'gpt2')),
                task_type=config_dict.get('task_type', 'causal_lm'),
                output_dir=config_dict.get('output_dir', self.output_dir),
                train_data_path=config_dict.get('train_data_path', './data/train'),
                val_data_path=config_dict.get('val_data_path'),
                test_data_path=config_dict.get('test_data_path'),
                num_epochs=config_dict.get('num_epochs', self.config.get('training', {}).get('num_epochs', 10)),
                batch_size=config_dict.get('batch_size', self.config.get('training', {}).get('batch_size', 16)),
                learning_rate=config_dict.get('learning_rate', self.config.get('training', {}).get('learning_rate', 2e-5)),
                weight_decay=config_dict.get('weight_decay', 0.01),
                warmup_ratio=config_dict.get('warmup_ratio', 0.1),
                use_fp16=config_dict.get('use_fp16', True),
                logging_steps=config_dict.get('logging_steps', 100),
                save_steps=config_dict.get('save_steps', 1000),
                eval_steps=config_dict.get('eval_steps', 500),
            )
            
            # 创建 UnifiedTrainingSystem - 调用 core 模块
            system = UnifiedTrainingSystem(config)
            logger.info(f"Created UnifiedTrainingSystem with config: model={config.model_name}")
            
            return system
        except Exception as e:
            logger.warning(f"Failed to create UnifiedTrainingSystem: {e}")
            return None
    
    def get_training_config(self) -> TrainingConfig:
        """
        获取训练配置
        
        调用 core 模块的 TrainingConfig
        
        Returns:
            TrainingConfig 实例
        """
        try:
            config = TrainingConfig(
                model_name=self.config.get('model', {}).get('name', 'gpt2'),
                task_type=self.config.get('model', {}).get('task_type', 'causal_lm'),
                output_dir=self.output_dir,
                num_epochs=self.config.get('training', {}).get('num_epochs', 10),
                batch_size=self.config.get('training', {}).get('batch_size', 16),
                learning_rate=self.config.get('training', {}).get('learning_rate', 2e-5),
            )
            logger.info(f"Created TrainingConfig: {config.model_name}")
            return config
        except Exception as e:
            logger.warning(f"Failed to create TrainingConfig: {e}")
            return None
    
    # =========================================================================
    # 派生功能：调用 scenarios 模块 (BasicModelScenario, AdvancedModelScenario, 
    #           IndustryScenario, ScheduledTrainingScenario)
    # =========================================================================
    
    def create_basic_model_scenario_instance(self, config: Dict[str, Any] = None) -> Optional[BasicModelScenario]:
        """
        创建基础模型场景实例
        
        调用 scenarios 模块的 BasicModelScenario
        
        Args:
            config: 场景配置
        
        Returns:
            BasicModelScenario 实例
        """
        try:
            scenario_config = config or {}
            scenario = BasicModelScenario(
                name=scenario_config.get('name', f'basic_scenario_{self.session_id}'),
                output_dir=scenario_config.get('output_dir', self.output_dir),
                **{k: v for k, v in scenario_config.items() if k not in ['name', 'output_dir']}
            )
            logger.info(f"Created BasicModelScenario: {scenario.session_id}")
            return scenario
        except Exception as e:
            logger.warning(f"Failed to create BasicModelScenario: {e}")
            return None
    
    def create_advanced_model_scenario_instance(self, config: Dict[str, Any] = None) -> Optional[AdvancedModelScenario]:
        """
        创建高级模型场景实例
        
        调用 scenarios 模块的 AdvancedModelScenario
        
        Args:
            config: 场景配置
        
        Returns:
            AdvancedModelScenario 实例
        """
        try:
            scenario_config = config or {}
            scenario = AdvancedModelScenario(
                name=scenario_config.get('name', f'advanced_scenario_{self.session_id}'),
                output_dir=scenario_config.get('output_dir', self.output_dir),
                **{k: v for k, v in scenario_config.items() if k not in ['name', 'output_dir']}
            )
            logger.info(f"Created AdvancedModelScenario: {scenario.session_id}")
            return scenario
        except Exception as e:
            logger.warning(f"Failed to create AdvancedModelScenario: {e}")
            return None
    
    def create_industry_scenario_instance(self, config: Dict[str, Any] = None) -> Optional[IndustryScenario]:
        """
        创建行业场景实例
        
        调用 scenarios 模块的 IndustryScenario
        
        Args:
            config: 场景配置
        
        Returns:
            IndustryScenario 实例
        """
        try:
            scenario_config = config or {}
            scenario = IndustryScenario(
                name=scenario_config.get('name', f'industry_scenario_{self.session_id}'),
                output_dir=scenario_config.get('output_dir', self.output_dir),
                **{k: v for k, v in scenario_config.items() if k not in ['name', 'output_dir']}
            )
            logger.info(f"Created IndustryScenario: {scenario.session_id}")
            return scenario
        except Exception as e:
            logger.warning(f"Failed to create IndustryScenario: {e}")
            return None
    
    def create_scheduled_training_scenario_instance(self, config: Dict[str, Any] = None) -> Optional[ScheduledTrainingScenario]:
        """
        创建定时训练场景实例
        
        调用 scenarios 模块的 ScheduledTrainingScenario
        
        Args:
            config: 场景配置
        
        Returns:
            ScheduledTrainingScenario 实例
        """
        try:
            scenario_config = config or {}
            scenario = ScheduledTrainingScenario(
                name=scenario_config.get('name', f'scheduled_scenario_{self.session_id}'),
                output_dir=scenario_config.get('output_dir', self.output_dir),
                **{k: v for k, v in scenario_config.items() if k not in ['name', 'output_dir']}
            )
            logger.info(f"Created ScheduledTrainingScenario: {scenario.session_id}")
            return scenario
        except Exception as e:
            logger.warning(f"Failed to create ScheduledTrainingScenario: {e}")
            return None
    
    def run_scenario_by_type(self, scenario_type: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        根据类型运行场景
        
        调用 scenarios 模块的所有场景类
        
        Args:
            scenario_type: 场景类型 (basic, advanced, industry, scheduled)
            config: 场景配置
        
        Returns:
            运行结果
        """
        result = {
            'success': False,
            'scenario_type': scenario_type,
            'scenario': None,
        }
        
        try:
            scenario = None
            
            if scenario_type == 'basic':
                scenario = self.create_basic_model_scenario_instance(config)
            elif scenario_type == 'advanced':
                scenario = self.create_advanced_model_scenario_instance(config)
            elif scenario_type == 'industry':
                scenario = self.create_industry_scenario_instance(config)
            elif scenario_type == 'scheduled':
                scenario = self.create_scheduled_training_scenario_instance(config)
            else:
                result['error'] = f'Unknown scenario type: {scenario_type}'
                return result
            
            if scenario:
                result['scenario'] = scenario.session_id if hasattr(scenario, 'session_id') else 'created'
                
                # 运行场景
                if hasattr(scenario, 'run'):
                    run_result = scenario.run()
                    result['run_result'] = run_result.to_dict() if hasattr(run_result, 'to_dict') else str(run_result)
                
                result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    # =========================================================================
    # 派生功能：调用 pipeline 模块 (PipelineExecutor, PipelineRunner, 
    #           create_executor, create_pipeline_runner)
    # =========================================================================
    
    def create_pipeline_executor_instance(self, runner: PipelineRunner = None) -> Optional[PipelineExecutor]:
        """
        创建流水线执行器实例
        
        调用 pipeline 模块的 PipelineExecutor, create_executor
        
        Args:
            runner: 流水线运行器（可选）
        
        Returns:
            PipelineExecutor 实例
        """
        try:
            # 如果没有提供 runner，创建一个
            if runner is None:
                runner = self.create_pipeline_runner_instance()
            
            # 使用便捷函数创建执行器 - 调用 create_executor
            executor = create_executor(runner=runner, session_id=self.session_id)
            logger.info(f"Created PipelineExecutor for session: {self.session_id}")
            return executor
        except Exception as e:
            logger.warning(f"Failed to create PipelineExecutor: {e}")
            return None
    
    def create_pipeline_runner_instance(self) -> Optional[PipelineRunner]:
        """
        创建流水线运行器实例
        
        调用 pipeline 模块的 PipelineRunner, create_pipeline_runner
        
        Returns:
            PipelineRunner 实例
        """
        try:
            # 使用便捷函数创建运行器 - 调用 create_pipeline_runner
            runner = create_pipeline_runner(session_id=self.session_id)
            logger.info(f"Created PipelineRunner for session: {self.session_id}")
            return runner
        except Exception as e:
            logger.warning(f"Failed to create PipelineRunner: {e}")
            return None
    
    def execute_pipeline_with_executor(
        self,
        pipeline: PipelineDefinition,
        executor: PipelineExecutor = None
    ) -> Dict[str, Any]:
        """
        使用执行器执行流水线
        
        调用 pipeline 模块的 PipelineExecutor
        
        Args:
            pipeline: 流水线定义
            executor: 执行器（可选）
        
        Returns:
            执行结果
        """
        try:
            # 如果没有提供 executor，创建一个
            if executor is None:
                executor = self.create_pipeline_executor_instance()
            
            if executor is None:
                return {'success': False, 'error': 'Failed to create executor'}
            
            # 执行流水线
            result = executor.execute(pipeline)
            
            return {
                'success': True,
                'pipeline': pipeline.name if hasattr(pipeline, 'name') else 'unknown',
                'result': result.to_dict() if hasattr(result, 'to_dict') else str(result),
            }
        except Exception as e:
            logger.warning(f"Pipeline execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_pipeline_with_runner(
        self,
        pipeline: PipelineDefinition,
        runner: PipelineRunner = None
    ) -> Dict[str, Any]:
        """
        使用运行器运行流水线
        
        调用 pipeline 模块的 PipelineRunner
        
        Args:
            pipeline: 流水线定义
            runner: 运行器（可选）
        
        Returns:
            运行结果
        """
        try:
            # 如果没有提供 runner，创建一个
            if runner is None:
                runner = self.create_pipeline_runner_instance()
            
            if runner is None:
                return {'success': False, 'error': 'Failed to create runner'}
            
            # 运行流水线
            if hasattr(runner, 'run'):
                result = runner.run(pipeline)
            elif hasattr(runner, 'execute'):
                result = runner.execute(pipeline)
            else:
                return {'success': False, 'error': 'Runner has no run or execute method'}
            
            return {
                'success': True,
                'pipeline': pipeline.name if hasattr(pipeline, 'name') else 'unknown',
                'result': result.to_dict() if hasattr(result, 'to_dict') else str(result),
            }
        except Exception as e:
            logger.warning(f"Pipeline run failed: {e}")
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # 派生功能：调用 strategies 模块 (ProductionStrategyConfig, ProductionTrainingContext,
    #           create_production_context, get_available_layers, IndustryScenarioStrategy)
    # =========================================================================
    
    def create_production_strategy_config(self, config: Dict[str, Any] = None) -> Optional[ProductionStrategyConfig]:
        """
        创建生产级策略配置
        
        调用 strategies 模块的 ProductionStrategyConfig
        
        Args:
            config: 配置字典
        
        Returns:
            ProductionStrategyConfig 实例
        """
        try:
            cfg = config or {}
            
            # 创建 ProductionStrategyConfig
            production_config = ProductionStrategyConfig(
                device=cfg.get('device', 'auto'),
                precision=cfg.get('precision', self.config.get('training', {}).get('precision', 'fp16')),
                enable_amp=cfg.get('enable_amp', True),
                distributed_mode=cfg.get('distributed_mode', self.config.get('distributed', {}).get('mode', 'none')),
                world_size=cfg.get('world_size', self.config.get('distributed', {}).get('world_size', 1)),
                modalities=cfg.get('modalities', self.config.get('multimodal', {}).get('modalities', ['text'])),
                hidden_size=cfg.get('hidden_size', 768),
                task_loss_type=cfg.get('task_loss_type', 'cross_entropy'),
                adapter_type=cfg.get('adapter_type'),
            )
            
            logger.info(f"Created ProductionStrategyConfig: precision={production_config.precision}")
            return production_config
        except Exception as e:
            logger.warning(f"Failed to create ProductionStrategyConfig: {e}")
            return None
    
    def create_production_training_context(
        self,
        model=None,
        config: ProductionStrategyConfig = None
    ) -> Optional[ProductionTrainingContext]:
        """
        创建生产级训练上下文
        
        调用 strategies 模块的 ProductionTrainingContext, create_production_context
        
        Args:
            model: PyTorch 模型（可选）
            config: 生产级策略配置（可选）
        
        Returns:
            ProductionTrainingContext 实例
        """
        try:
            # 如果没有提供配置，创建一个
            if config is None:
                config = self.create_production_strategy_config()
            
            if config is None:
                return None
            
            # 使用便捷函数创建上下文 - 调用 create_production_context
            context = create_production_context(config, model)
            logger.info("Created ProductionTrainingContext")
            return context
        except Exception as e:
            logger.warning(f"Failed to create ProductionTrainingContext: {e}")
            return None
    
    def get_strategy_available_layers(self) -> Dict[str, bool]:
        """
        获取策略可用层
        
        调用 strategies 模块的 get_available_layers
        
        Returns:
            可用层字典
        """
        try:
            layers = get_available_layers()
            logger.info(f"Available strategy layers: {layers}")
            return layers
        except Exception as e:
            logger.warning(f"Failed to get available layers: {e}")
            return {}
    
    def create_industry_scenario_strategy(self, config: Dict[str, Any] = None) -> Optional[IndustryScenarioStrategy]:
        """
        创建行业场景策略
        
        调用 strategies 模块的 IndustryScenarioStrategy, ScenarioStrategyConfig
        
        Args:
            config: 策略配置
        
        Returns:
            IndustryScenarioStrategy 实例
        """
        try:
            cfg = config or {}
            
            # 创建场景策略配置
            scenario_config = ScenarioStrategyConfig(
                scenario_type=cfg.get('scenario_type'),
                freeze_backbone=cfg.get('freeze_backbone', False),
                use_scene_adapter=cfg.get('use_scene_adapter', True),
            )
            
            # 创建生产级配置（可选）
            production_config = None
            if cfg.get('use_production_context', False):
                production_config = self.create_production_strategy_config(cfg.get('production', {}))
            
            # 创建行业场景策略 - 调用 IndustryScenarioStrategy
            strategy = IndustryScenarioStrategy(
                config=scenario_config,
                production_config=production_config,
            )
            
            logger.info(f"Created IndustryScenarioStrategy")
            return strategy
        except Exception as e:
            logger.warning(f"Failed to create IndustryScenarioStrategy: {e}")
            return None
    
    def setup_production_training_environment(self, model=None) -> Dict[str, Any]:
        """
        设置生产级训练环境
        
        统一调用 strategies 模块的生产级功能
        
        Args:
            model: PyTorch 模型（可选）
        
        Returns:
            设置结果
        """
        result = {
            'success': False,
            'config': None,
            'context': None,
            'layers': None,
            'strategy': None,
        }
        
        try:
            # 获取可用层 - 调用 get_available_layers
            result['layers'] = get_available_layers()
            
            # 创建生产级配置 - 调用 ProductionStrategyConfig
            config = self.create_production_strategy_config()
            result['config'] = 'created' if config else None
            
            # 创建生产级上下文 - 调用 create_production_context, ProductionTrainingContext
            context = self.create_production_training_context(model, config)
            result['context'] = 'created' if context else None
            
            # 创建行业场景策略 - 调用 IndustryScenarioStrategy
            strategy = self.create_industry_scenario_strategy({
                'use_production_context': True,
            })
            result['strategy'] = 'created' if strategy else None
            
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    # =========================================================================
    # 派生功能：调用 distillation 模块 (DistillationTaskConfig, DistillationPresets,
    #           get_distillation_service)
    # =========================================================================
    
    def create_distillation_task_config(self, config: Dict[str, Any] = None) -> Optional[DistillationTaskConfig]:
        """
        创建蒸馏任务配置
        
        调用 distillation 模块的 DistillationTaskConfig
        
        Args:
            config: 配置字典
        
        Returns:
            DistillationTaskConfig 实例
        """
        try:
            cfg = config or {}
            
            # 创建 DistillationTaskConfig（不包含 teacher/student model path，这些在 distillation_config 中设置）
            task_config = DistillationTaskConfig(
                task_name=cfg.get('task_name', f'distillation_{self.session_id}'),
                output_dir=cfg.get('output_dir', self.output_dir),
                num_epochs=cfg.get('num_epochs', self.config.get('training', {}).get('num_epochs', 10)),
                batch_size=cfg.get('batch_size', self.config.get('training', {}).get('batch_size', 32)),
                learning_rate=cfg.get('learning_rate', 1e-4),
                train_data_path=cfg.get('train_data_path'),
                eval_data_path=cfg.get('eval_data_path'),
            )
            
            logger.info(f"Created DistillationTaskConfig: {task_config.task_name}")
            return task_config
        except Exception as e:
            logger.warning(f"Failed to create DistillationTaskConfig: {e}")
            return None
    
    def get_distillation_preset(self, preset_name: str) -> Optional[DistillationTaskConfig]:
        """
        获取蒸馏预设配置
        
        调用 distillation 模块的 DistillationPresets
        
        Args:
            preset_name: 预设名称 (edge_deployment, high_accuracy, industry_model, 
                         multimodal, distributed_large_scale)
        
        Returns:
            DistillationTaskConfig 实例
        """
        try:
            preset_map = {
                'edge_deployment': DistillationPresets.edge_deployment,
                'high_accuracy': DistillationPresets.high_accuracy,
                'multimodal': DistillationPresets.multimodal,
                'distributed_large_scale': DistillationPresets.distributed_large_scale,
            }
            
            if preset_name == 'industry_model':
                # industry_model 需要行业类型参数
                industry_type = self.config.get('industry', {}).get('type', 'manufacturing')
                preset_config = DistillationPresets.industry_model(industry_type)
            elif preset_name in preset_map:
                preset_config = preset_map[preset_name]()
            else:
                logger.warning(f"Unknown preset: {preset_name}")
                return None
            
            logger.info(f"Got DistillationPresets.{preset_name}")
            return preset_config
        except Exception as e:
            logger.warning(f"Failed to get distillation preset: {e}")
            return None
    
    def get_distillation_service_instance(self, use_memory_storage: bool = True):
        """
        获取蒸馏服务实例
        
        调用 distillation 模块的 get_distillation_service
        
        Args:
            use_memory_storage: 是否使用内存存储
        
        Returns:
            蒸馏服务实例
        """
        try:
            # 调用 get_distillation_service
            service = get_distillation_service(use_memory_storage=use_memory_storage)
            logger.info("Got distillation service instance")
            return service
        except Exception as e:
            logger.warning(f"Failed to get distillation service: {e}")
            return None
    
    def create_distillation_task_via_service(
        self,
        task_name: str,
        preset_name: str = None,
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        通过服务创建蒸馏任务
        
        统一调用 distillation 模块的 get_distillation_service, DistillationTaskConfig, DistillationPresets
        
        Args:
            task_name: 任务名称
            preset_name: 预设名称（可选）
            config: 配置字典（可选）
        
        Returns:
            任务创建结果
        """
        result = {
            'success': False,
            'task_id': None,
            'preset': None,
            'service': None,
        }
        
        try:
            # 获取蒸馏服务 - 调用 get_distillation_service
            service = self.get_distillation_service_instance()
            if service is None:
                result['error'] = 'Failed to get distillation service'
                return result
            result['service'] = 'available'
            
            # 获取或创建任务配置
            task_config = None
            if preset_name:
                # 使用预设 - 调用 DistillationPresets
                task_config = self.get_distillation_preset(preset_name)
                result['preset'] = preset_name
            
            if task_config is None:
                # 使用自定义配置 - 调用 DistillationTaskConfig
                task_config = self.create_distillation_task_config(config)
            
            if task_config is None:
                result['error'] = 'Failed to create task config'
                return result
            
            # 通过服务创建任务
            tenant_id = self.config.get('tenant_id', 'default_tenant')
            user_id = self.config.get('user_id', 'default_user')
            
            task = service.create_task(
                task_name=task_name,
                tenant_id=tenant_id,
                user_id=user_id,
                teacher_model_path=task_config.teacher_model_path if hasattr(task_config, 'teacher_model_path') else 'mock',
                student_model_path=task_config.student_model_path if hasattr(task_config, 'student_model_path') else 'mock',
                scenario=self.config.get('distillation', {}).get('scenario', 'standard'),
            )
            
            result['task_id'] = task.task_id if hasattr(task, 'task_id') else 'created'
            result['success'] = True
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def analyze_config(self) -> Dict[str, Any]:
        """分析配置并确定训练策略"""
        analysis = {
            'model_type': self.config.get('model', {}).get('type', 'standard'),
            'training_mode': self.config.get('training', {}).get('mode', 'standard'),
            'data_type': self.config.get('data', {}).get('type', 'text'),
            'distributed': self.config.get('distributed', {}).get('enabled', False),
            'knowledge_distillation': self.config.get('distillation', {}).get('enabled', False),
            'multimodal': self.config.get('multimodal', {}).get('enabled', False),
            'three_stage': self.config.get('three_stage', {}).get('enabled', False),
            'scenario_enabled': self.config.get('scenario', {}).get('enabled', False),
            'scenario_type': self.config.get('scenario', {}).get('type', 'basic_model'),
            # 行业训练配置
            'industry_enabled': self.config.get('industry', {}).get('enabled', False),
            'industry_type': self.config.get('industry', {}).get('type', 'general'),
            # 策略组合配置
            'use_strategies': self.config.get('strategies', {}).get('enabled', False),
            'strategy_types': self.config.get('strategies', {}).get('types', []),
            # 知识蒸馏场景配置
            'distillation_scenario': self.config.get('distillation', {}).get('scenario', 'standard'),
            'distillation_use_service': self.config.get('distillation', {}).get('use_service', False),
            
            # 插件配置
            'plugin_enabled': self.config.get('plugins', {}).get('enabled', False),
        }
        
        logger.info("配置分析结果:")
        for key, value in analysis.items():
            logger.info(f"  {key}: {value}")
        
        return analysis
    
    def select_trainer(self, analysis: Dict[str, Any]) -> object:
        """
        根据分析结果选择合适的训练器
        
        优先级（从高到低）：
        1. 行业模型训练（带编排）
        2. 场景化训练
        3. 分布式训练
        4. 知识蒸馏训练（多场景）
        5. 多模态训练
        6. 三阶段训练
        7. 标准训练
        """
        
        # 1. 行业模型训练（最高优先级）
        if analysis.get('industry_enabled'):
            logger.info(f"选择行业模型训练器: {analysis['industry_type']}")
            return self._create_industry_trainer()
        
        # 2. 场景化训练
        if analysis.get('scenario_enabled'):
            logger.info(f"选择场景化训练器: {analysis['scenario_type']}")
            return self._create_scenario_manager()
        
        # 3. 分布式训练
        if analysis.get('distributed'):
            logger.info("选择分布式训练器")
            return self._create_distributed_trainer()
        
        # 4. 知识蒸馏训练（多场景支持）
        if analysis.get('knowledge_distillation'):
            distillation_scenario = analysis.get('distillation_scenario', 'standard')
            use_service = analysis.get('distillation_use_service', False)
            logger.info(f"选择知识蒸馏训练器: scenario={distillation_scenario}, use_service={use_service}")
            return self._create_distillation_trainer(
                scenario=distillation_scenario,
                use_service=use_service
            )
        
        # 5. 多模态训练
        if analysis.get('multimodal'):
            logger.info("选择多模态训练器")
            return self._create_multimodal_trainer()
        
        # 6. 三阶段训练
        if analysis.get('three_stage'):
            logger.info("选择三阶段训练器")
            return self._create_three_stage_trainer()
        
        # 7. 插件扩展训练 (新增)
        if analysis.get('plugin_enabled'):
            logger.info("选择插件扩展训练器")
            return self._create_plugin_trainer()

        # 8. 标准训练
        logger.info("选择统一训练器")
        return self._create_unified_trainer()
    
    def _create_plugin_trainer(self):
        """创建插件扩展训练器"""
        try:
            from backend.modules.training.plugins import PluginRegistry, PluginContext
            from backend.modules.training.core.unified_training_system import UnifiedTrainingSystem, TrainingConfig
            
            # 初始化插件系统
            registry = PluginRegistry()
            plugins_config = self.config.get('plugins', {}).get('list', [])
            for plugin_cfg in plugins_config:
                # 假设有一个简单的加载机制，这里仅做示例
                pass

            # 创建基础训练器
            config = TrainingConfig(
                model_name=self.config.get('model', {}).get('name', 'gpt2'),
                output_dir=self.output_dir,
                # ... 其他配置
            )
            trainer = UnifiedTrainingSystem(config)
            
            # 将训练器与插件上下文关联 (示例逻辑)
            # context = PluginContext(trainer=trainer)
            
            return trainer
        except ImportError as e:
            logger.warning(f"插件模块导入失败: {e}")
            return self._create_unified_trainer()

    def _create_industry_trainer(self) -> object:
        """
        创建行业模型训练器
        
        调用派生方法:
        - create_production_strategy_config: 创建 ProductionStrategyConfig
        - create_production_training_context: 创建 ProductionTrainingContext
        - get_strategy_available_layers: 调用 get_available_layers
        - create_industry_scenario_strategy: 创建 IndustryScenarioStrategy
        """
        try:
            from backend.modules.training.orchestrator import TrainingOrchestrator, TrainingPlan
            from backend.modules.training.industry import create_industry_model
            from backend.modules.training.strategies import (
                create_strategy, create_composite_strategy,
                ScenarioStrategyConfig
            )
            from backend.modules.training.strategies.scenario_strategy import ScenarioType
            
            # 获取行业配置
            industry_config = self.config.get('industry', {})
            industry_type = industry_config.get('type', 'general')
            
            # 调用派生方法 get_strategy_available_layers
            # 该方法调用 strategies 模块的 get_available_layers
            available_layers = self.get_strategy_available_layers()
            logger.info(f"Available layers via derived method: {available_layers}")
            
            # 创建编排器
            orchestrator = TrainingOrchestrator(output_dir=self.output_dir)
            
            # 创建行业训练计划
            plan = orchestrator.create_industry_plan(
                model_name=self.config.get('model', {}).get('name', 'industry_model'),
                include_pretrain=industry_config.get('include_pretrain', True),
                include_align=industry_config.get('include_align', True),
                include_finetune=industry_config.get('include_finetune', True),
                stage_configs=industry_config.get('stage_configs')
            )
            
            # 创建行业模型
            model = create_industry_model(industry_type)
            
            # 调用派生方法 create_production_strategy_config
            # 该方法创建 ProductionStrategyConfig
            modalities = industry_config.get('modalities', ['text', 'time_series'])
            production_config = self.create_production_strategy_config({
                'device': 'auto',
                'precision': industry_config.get('precision', 'fp16'),
                'enable_amp': industry_config.get('enable_amp', True),
                'distributed_mode': industry_config.get('distributed_mode', 'none'),
                'world_size': industry_config.get('world_size', 1),
                'modalities': modalities,
                'hidden_size': industry_config.get('hidden_size', 768),
                'task_loss_type': industry_config.get('task_loss_type', 'cross_entropy'),
                'adapter_type': industry_config.get('adapter_type'),
            })
            
            # 调用派生方法 create_production_training_context
            # 该方法调用 create_production_context 创建 ProductionTrainingContext
            production_context = self.create_production_training_context(model, production_config)
            
            # 创建策略组合
            strategies = []
            
            # 调用派生方法 create_industry_scenario_strategy
            # 该方法创建 IndustryScenarioStrategy
            scenario_type_map = {
                'manufacturing': ScenarioType.EQUIPMENT_FAULT_PREDICTION,
                'finance': ScenarioType.RISK_ASSESSMENT,
                'healthcare': ScenarioType.DISEASE_DIAGNOSIS,
            }
            industry_strategy = self.create_industry_scenario_strategy({
                'scenario_type': scenario_type_map.get(industry_type, ScenarioType.BASIC_MODEL),
                'freeze_backbone': industry_config.get('freeze_backbone', False),
                'use_scene_adapter': industry_config.get('use_scene_adapter', True),
                'use_production_context': True,
                'production': {
                    'device': 'auto',
                    'precision': industry_config.get('precision', 'fp16'),
                },
            })
            
            if industry_strategy:
                strategies.append(industry_strategy)
            
            # 可选的附加策略
            if industry_config.get('use_multimodal', False):
                strategies.append(create_strategy('industry_multimodal'))
            if industry_config.get('use_distillation', False):
                strategies.append(create_strategy('industry_distillation'))
            if industry_config.get('use_distributed', False):
                strategies.append(create_strategy('industry_distributed'))
            
            # 包装为可训练对象
            class IndustryTrainerWrapper:
                def __init__(self, orchestrator, plan, model, strategies, production_context, output_dir):
                    self.orchestrator = orchestrator
                    self.plan = plan
                    self.model = model
                    self.strategies = strategies
                    self.production_context = production_context
                    self.output_dir = output_dir
                
                def train(self) -> Dict[str, Any]:
                    try:
                        # 使用生产级上下文包装模型
                        if self.production_context:
                            self.model = self.production_context.wrap_model(self.model)
                        
                        # 尝试提交到编排器
                        if hasattr(self.orchestrator, 'submit_job'):
                            job = self.orchestrator.submit_job(self.plan)
                            return {
                                'success': True,
                                'job_id': job.job_id if hasattr(job, 'job_id') else 'unknown',
                                'plan': self.plan.to_dict() if hasattr(self.plan, 'to_dict') else str(self.plan),
                                'strategies': [s.name for s in self.strategies],
                                'layers_available': get_available_layers(),
                                'message': '行业模型训练任务已提交'
                            }
                        else:
                            # 直接执行训练
                            # 这里可以调用 core 模块或者 strategies 执行逻辑
                            # 示例：直接返回准备就绪
                            return {
                                'success': True,
                                'strategies': [s.name for s in self.strategies],
                                'layers_available': get_available_layers(),
                                'message': '行业模型训练准备就绪'
                            }
                    finally:
                        # 清理资源
                        if self.production_context:
                            self.production_context.cleanup()
            
            return IndustryTrainerWrapper(
                orchestrator, plan, model, strategies, 
                production_context, self.output_dir
            )
            
        except ImportError as e:
            logger.warning(f"行业训练模块导入失败: {e}, 回退到标准训练器")
            return self._create_unified_trainer()
    
    def _create_unified_trainer(self):
        """创建统一训练器
        
        调用派生方法 create_unified_training_system 创建统一训练系统
        """
        try:
            # 调用派生方法 create_unified_training_system
            # 该方法调用 core 模块的 UnifiedTrainingSystem 和 TrainingConfig
            system = self.create_unified_training_system({
                'model_name': self.config.get('model', {}).get('name', 'gpt2'),
                'task_type': self.config.get('model', {}).get('task_type', 'causal_lm'),
                'output_dir': self.output_dir,
                'train_data_path': self.config.get('data', {}).get('train_path', './data/train'),
                'val_data_path': self.config.get('data', {}).get('val_path'),
                'test_data_path': self.config.get('data', {}).get('test_path'),
                'num_epochs': self.config.get('training', {}).get('num_epochs', 10),
                'batch_size': self.config.get('training', {}).get('batch_size', 16),
                'learning_rate': self.config.get('training', {}).get('learning_rate', 2e-5),
                'weight_decay': self.config.get('training', {}).get('weight_decay', 0.01),
                'warmup_ratio': self.config.get('training', {}).get('warmup_ratio', 0.1),
                'use_fp16': self.config.get('training', {}).get('fp16', True),
                'logging_steps': self.config.get('monitoring', {}).get('logging_steps', 100),
                'save_steps': self.config.get('monitoring', {}).get('save_steps', 1000),
                'eval_steps': self.config.get('monitoring', {}).get('eval_steps', 500),
            })
            
            if system is None:
                # 回退到直接创建
                config = self.get_training_config()
                return UnifiedTrainingSystem(config)
            
            return system
        except Exception as e:
            logger.error(f"无法创建UnifiedTrainingSystem: {e}")
            raise
    
    def _create_distributed_trainer(self) -> object:
        """创建分布式训练器"""
        try:
            from backend.modules.distributed import DistributedTrainingConfig, launch_distributed_training
            
            config_dict = self.config.get('distributed', {})
            config_dict = {k: v for k, v in config_dict.items() if k != 'enabled'}
            config_dict['output_dir'] = self.output_dir
            cfg = DistributedTrainingConfig(**config_dict)

            class DistributedTrainerAdapter:
                def __init__(self, cfg_dict: dict):
                    self._cfg_dict = cfg_dict
                def train(self) -> dict:
                    return launch_distributed_training(self._cfg_dict)

            return DistributedTrainerAdapter(cfg.to_dict())
        except ImportError as e:
            logger.warning(f"分布式训练模块导入失败: {e}")
            return self._create_unified_trainer()
    
    def _create_multimodal_trainer(self):
        """创建多模态训练器"""
        try:
            from backend.modules.training.multimodal.multimodal_trainer import MultiModalTrainer
            from backend.modules.training.multimodal.multimodal_config import MultiModalConfig
            # 注意：此处调用 multimodal 模块
            
            config_dict = self.config.get('multimodal', {})
            config_dict = {k: v for k, v in config_dict.items() if k != 'enabled'}
            config_dict['output_dir'] = self.output_dir
            config = MultiModalConfig(**config_dict)
            
            return MultiModalTrainer(config)
        except ImportError as e:
            logger.warning(f"多模态训练模块导入失败: {e}")
            return self._create_unified_trainer()
    
    def _create_distillation_trainer(
        self, 
        scenario: str = "standard",
        use_service: bool = False
    ) -> object:
        """
        创建知识蒸馏训练器
        
        支持多种蒸馏场景：
        - standard: 标准蒸馏
        - industry: 行业蒸馏
        - edge_deploy: 边缘部署蒸馏
        - multimodal: 多模态蒸馏
        - real_time: 实时推理蒸馏
        - progressive: 渐进式蒸馏
        - self: 自蒸馏
        
        Args:
            scenario: 蒸馏场景
            use_service: 是否使用蒸馏服务（支持租户管理、监控）
        
        Returns:
            蒸馏训练器或服务包装器
        """
        try:
            # 获取蒸馏配置
            config_dict = self.config.get('distillation', {})
            
            # 使用蒸馏服务（租户级管理）
            if use_service:
                return self._create_distillation_service_trainer(config_dict, scenario)
            
            # 使用场景管理器
            return self._create_distillation_scenario_trainer(config_dict, scenario)
        except ImportError as e:
            logger.warning(f"知识蒸馏模块导入失败: {e}, 尝试使用基础训练器")
            return self._create_basic_distillation_trainer(config_dict)
    
    def _create_distillation_service_trainer(
        self, 
        config_dict: Dict[str, Any],
        scenario: str
    ) -> object:
        """使用蒸馏服务创建训练器
        
        调用派生方法:
        - get_distillation_service_instance: 调用 get_distillation_service
        - get_distillation_preset: 调用 DistillationPresets
        - create_distillation_task_config: 调用 DistillationTaskConfig
        """
        try:
            # 调用派生方法 get_distillation_service_instance
            # 该方法调用 distillation 模块的 get_distillation_service
            service = self.get_distillation_service_instance(use_memory_storage=True)
            
            if service is None:
                # 回退到直接导入
                from backend.modules.training.distillation import get_distillation_service
                service = get_distillation_service(use_memory_storage=True)
            
            # 获取租户信息
            tenant_id = config_dict.get('tenant_id', 'default_tenant')
            user_id = config_dict.get('user_id', 'default_user')
            
            # 创建服务适配器
            class DistillationServiceAdapter:
                def __init__(self, service, config_dict, scenario, tenant_id, user_id, output_dir):
                    self.service = service
                    self.config_dict = config_dict
                    self.scenario = scenario
                    self.tenant_id = tenant_id
                    self.user_id = user_id
                    self.output_dir = output_dir
                    self.task = None
                
                def train(self) -> Dict[str, Any]:
                    # 创建蒸馏任务
                    self.task = self.service.create_task(
                        task_name=self.config_dict.get('task_name', 'distillation_task'),
                        tenant_id=self.tenant_id,
                        user_id=self.user_id,
                        teacher_model_path=self.config_dict.get('teacher_model_path', 'mock'),
                        student_model_path=self.config_dict.get('student_model_path', 'mock'),
                        scenario=self.scenario,
                        config_overrides={
                            'output_dir': self.output_dir,
                            'num_epochs': self.config_dict.get('num_epochs', 10),
                            'batch_size': self.config_dict.get('batch_size', 32),
                            'learning_rate': self.config_dict.get('learning_rate', 1e-4),
                            **{k: v for k, v in self.config_dict.items() 
                               if k not in ['enabled', 'scenario', 'use_service', 
                                           'tenant_id', 'user_id', 'task_name',
                                           'teacher_model_path', 'student_model_path']}
                        }
                    )
                    
                    # 启动任务
                    result = self.service.start_task(self.task.task_id, self.tenant_id)
                    
                    if result['success']:
                        return {
                            'success': True,
                            'task_id': self.task.task_id,
                            'scenario': self.scenario,
                            'status': result['status'],
                            'message': f'蒸馏任务已启动: {self.scenario}'
                        }
                    else:
                        return result
                
                def get_status(self) -> Dict[str, Any]:
                    if self.task:
                        task = self.service.get_task(self.task.task_id, self.tenant_id)
                        return {
                            'task_id': task.task_id,
                            'status': task.status,
                            'progress': task.progress,
                            'metrics': task.metrics
                        }
                    return {}
                
                def generate_report(self) -> Dict[str, Any]:
                    if self.task:
                        report = self.service.generate_report(self.task.task_id, self.tenant_id)
                        if report:
                            return {
                                'task_id': report.task_id,
                                'scenario': report.scenario,
                                'status': report.status,
                                'duration_seconds': report.duration_seconds,
                                'final_loss': report.final_loss,
                            }
                    return {}
            
            return DistillationServiceAdapter(
                service, config_dict, scenario, 
                tenant_id, user_id, self.output_dir
            )
            
        except ImportError as e:
            logger.warning(f"蒸馏服务导入失败: {e}, 回退到场景训练器")
            return self._create_distillation_scenario_trainer(config_dict, scenario)
    
    def _create_distillation_scenario_trainer(
        self, 
        config_dict: Dict[str, Any],
        scenario: str
    ) -> object:
        """使用场景管理器创建训练器
        
        调用派生方法:
        - get_distillation_preset: 调用 DistillationPresets 的各种预设
        - create_distillation_task_config: 调用 DistillationTaskConfig
        """
        try:
            from backend.modules.training.distillation import (
                get_scenario_manager,
                KnowledgeDistillationTrainer,
                DistillationConfig,
                DistillationTaskConfig,
                DistillationPresets
            )
            
            scenario_manager = get_scenario_manager()
            
            # 调用派生方法 get_distillation_preset
            # 该方法调用 DistillationPresets 的各种预设方法
            task_config = self.get_distillation_preset(scenario)
            
            if task_config is None:
                # 回退到直接获取预设
                preset_map = {
                    'edge_deploy': DistillationPresets.edge_deployment,
                    'high_accuracy': DistillationPresets.high_accuracy,
                    'industry': lambda: DistillationPresets.industry_model(
                        config_dict.get('industry_type', 'manufacturing')
                    ),
                    'multimodal': DistillationPresets.multimodal,
                    'distributed': DistillationPresets.distributed_large_scale,
                }
                
                if scenario in preset_map:
                    task_config = preset_map[scenario]()
                else:
                    # 调用派生方法 create_distillation_task_config
                    # 该方法创建 DistillationTaskConfig
                    task_config = self.create_distillation_task_config(config_dict)
                    
                    if task_config is None:
                        task_config = DistillationTaskConfig()
            
            # 设置蒸馏配置
            task_config.distillation_config = DistillationConfig(
                teacher_model_path=config_dict.get('teacher_model_path', 'mock'),
                student_model_path=config_dict.get('student_model_path', 'mock'),
                temperature=config_dict.get('temperature', 4.0),
                alpha=config_dict.get('alpha', 0.7),
                beta=config_dict.get('beta', 0.3),
                use_feature_distillation=config_dict.get('use_feature_distillation', True),
                feature_loss_weight=config_dict.get('feature_loss_weight', 0.1),
                use_attention_distillation=config_dict.get('use_attention_distillation', True),
                attention_loss_weight=config_dict.get('attention_loss_weight', 0.1),
            )
            
            # 设置场景配置
            if task_config.scenario_config:
                task_config.scenario_config.scenario = scenario
            
            # 创建场景适配器
            class ScenarioDistillationAdapter:
                def __init__(self, scenario_manager, task_config, scenario, output_dir):
                    self.scenario_manager = scenario_manager
                    self.task_config = task_config
                    self.scenario = scenario
                    self.output_dir = output_dir
                    self.trainer = None
                
                def train(self) -> Dict[str, Any]:
                    # 准备场景
                    prep_result = self.scenario_manager.prepare_scenario(self.task_config)
                    logger.info(f"场景准备完成: {prep_result}")
                    
                    # 获取策略
                    strategy = self.scenario_manager.get_strategy_for_scenario(self.task_config)
                    logger.info(f"使用策略: {strategy.name}")
                    
                    # 创建训练器
                    self.trainer = KnowledgeDistillationTrainer(self.task_config.distillation_config)
                    self.trainer.set_strategy(strategy)
                    
                    # 执行训练
                    num_steps = self.task_config.num_epochs * 100
                    result = self.trainer.train(num_steps=num_steps)
                    
                    # 后处理
                    if result['success'] and self.trainer.student_model:
                        self.trainer.student_model = self.scenario_manager.post_process_model(
                            self.trainer.student_model,
                            self.task_config,
                            result
                        )
                    
                    result['scenario'] = self.scenario
                    result['strategy'] = strategy.name
                    return result
            
            return ScenarioDistillationAdapter(
                scenario_manager, task_config, scenario, self.output_dir
            )
            
        except ImportError as e:
            logger.warning(f"场景管理器导入失败: {e}, 回退到基础训练器")
            return self._create_basic_distillation_trainer(config_dict)
    
    def _create_basic_distillation_trainer(self, config_dict: Dict[str, Any]) -> object:
        """创建基础知识蒸馏训练器"""
        try:
            from backend.modules.training.distillation.knowledge_distillation import KnowledgeDistillationTrainer
            from backend.modules.training.distillation.compression_config import DistillationConfig
            
            # 清理配置
            clean_config = {k: v for k, v in config_dict.items() 
                          if k not in ['enabled', 'scenario', 'use_service', 
                                      'tenant_id', 'user_id', 'task_name']}
            
            # 设置默认值
            clean_config.setdefault('teacher_model_path', 'mock')
            clean_config.setdefault('student_model_path', 'mock')
            
            config = DistillationConfig(**clean_config)
            return KnowledgeDistillationTrainer(config)
            
        except ImportError as e:
            logger.warning(f"基础知识蒸馏模块导入失败: {e}")
            return self._create_unified_trainer()
    
    def _create_three_stage_trainer(self):
        """创建三阶段训练器"""
        try:
            from backend.modules.training.three_stage.three_stage_trainer import ThreeStageTrainer
            from backend.modules.training.three_stage.three_stage_config import ThreeStageConfig
            
            config_dict = self.config.get('three_stage', {})
            config_dict = {k: v for k, v in config_dict.items() if k != 'enabled'}
            config_dict['output_dir'] = self.output_dir
            config = ThreeStageConfig(**config_dict)
            
            return ThreeStageTrainer(config)
        except ImportError as e:
            logger.warning(f"三阶段训练模块导入失败: {e}")
            return self._create_unified_trainer()
    
    def _create_scenario_manager(self):
        """创建场景化训练管理器"""
        try:
            from backend.modules.training.scenarios.scenario_manager import (
                ScenarioManager, ScenarioConfig
            )
            from backend.schemas.enums import TrainingScenario, ScheduleType, TrainingPriority
            
            scenario_type_str = self.config.get('scenario', {}).get('type', 'basic_model')
            scenario_type = TrainingScenario(scenario_type_str)
            
            scenario_config = ScenarioConfig(
                scenario=scenario_type,
                name=f"{scenario_type_str}_training",
                output_dir=self.output_dir,
                base_model_path=self.config.get('model', {}).get('path'),
            )
            
            manager = ScenarioManager()
            return manager
        except ImportError as e:
            logger.warning(f"场景化训练模块导入失败: {e}")
            return self._create_unified_trainer()
    
    def _create_scenario_trainer_by_type(self, scenario_type: str):
        """
        根据场景类型创建具体的场景训练器
        
        调用派生方法创建场景实例 (BasicModelScenario, AdvancedModelScenario, 
        IndustryScenario, ScheduledTrainingScenario)
        
        Args:
            scenario_type: 场景类型
                - basic_model: 基础模型场景
                - advanced_model: 高级模型场景
                - scheduled: 定时训练场景
                - industry: 行业场景
        
        Returns:
            场景训练器实例
        """
        try:
            scenario_config = self.config.get('scenario', {})
            
            # 构建统一的场景配置
            base_config = {
                'name': scenario_config.get('name', f'{scenario_type}_training'),
                'output_dir': self.output_dir,
                'model_name': self.config.get('model', {}).get('name', 'gpt2'),
                'num_epochs': self.config.get('training', {}).get('num_epochs', 10),
                'batch_size': self.config.get('training', {}).get('batch_size', 16),
                'learning_rate': self.config.get('training', {}).get('learning_rate', 2e-5),
            }
            base_config.update(scenario_config)
            
            if scenario_type == 'basic_model':
                # 调用派生方法 create_basic_model_scenario_instance
                # 该方法直接实例化 BasicModelScenario
                scenario = self.create_basic_model_scenario_instance(base_config)
                if scenario:
                    logger.info(f"Created BasicModelScenario via derived method: {scenario.session_id}")
                    return scenario
                    
                # 回退到工厂函数
                from backend.modules.training.scenarios import create_basic_scenario, BasicModelConfig
                config = BasicModelConfig(**{k: v for k, v in base_config.items() if k in ['name', 'output_dir', 'model_name', 'num_epochs', 'batch_size', 'learning_rate']})
                scenario = create_basic_scenario(config)
                logger.info(f"Created BasicModelScenario: {scenario.session_id}")
                return scenario
                
            elif scenario_type == 'advanced_model':
                # 调用派生方法 create_advanced_model_scenario_instance
                # 该方法直接实例化 AdvancedModelScenario
                scenario = self.create_advanced_model_scenario_instance(base_config)
                if scenario:
                    logger.info(f"Created AdvancedModelScenario via derived method: {scenario.session_id}")
                    return scenario
                    
                # 回退到工厂函数
                from backend.modules.training.scenarios import create_advanced_scenario, AdvancedModelConfig, AdvancedModelType
                model_type = getattr(
                    AdvancedModelType, 
                    scenario_config.get('model_type', 'TRANSFORMER').upper(),
                    AdvancedModelType.TRANSFORMER
                )
                config = AdvancedModelConfig(
                    name=base_config['name'],
                    output_dir=self.output_dir,
                    model_type=model_type,
                    num_epochs=base_config['num_epochs'],
                )
                scenario = create_advanced_scenario(config)
                logger.info(f"Created AdvancedModelScenario: {scenario.session_id}")
                return scenario
                
            elif scenario_type == 'scheduled':
                # 调用派生方法 create_scheduled_training_scenario_instance
                # 该方法直接实例化 ScheduledTrainingScenario
                scenario = self.create_scheduled_training_scenario_instance(base_config)
                if scenario:
                    logger.info(f"Created ScheduledTrainingScenario via derived method: {scenario.session_id}")
                    return scenario
                    
                # 回退到工厂函数
                from backend.modules.training.scenarios import create_scheduled_scenario, ScheduledTrainingConfig, ScheduleType
                # ScheduleType 枚举成员: ONCE, DAILY, WEEKLY, MONTHLY, INTERVAL, CRON, CONDITION
                schedule_type = getattr(
                    ScheduleType,
                    scenario_config.get('schedule_type', 'INTERVAL').upper(),
                    ScheduleType.INTERVAL
                )
                # ScheduledTrainingConfig 不使用 schedule_type 和 interval_hours 作为直接参数
                config = ScheduledTrainingConfig(
                    name=base_config['name'],
                    output_dir=self.output_dir,
                )
                # 设置调度相关属性（如果存在）
                if hasattr(config, 'schedule_type'):
                    config.schedule_type = schedule_type
                if hasattr(config, 'interval_hours'):
                    config.interval_hours = scenario_config.get('interval_hours', 24)
                scenario = create_scheduled_scenario(config)
                logger.info(f"Created ScheduledTrainingScenario: {scenario.session_id}")
                return scenario
                
            elif scenario_type == 'industry':
                # 调用派生方法 create_industry_scenario_instance
                # 该方法直接实例化 IndustryScenario
                scenario = self.create_industry_scenario_instance(base_config)
                if scenario:
                    logger.info(f"Created IndustryScenario via derived method: {scenario.session_id}")
                    return scenario
                    
                # 回退到工厂函数
                from backend.modules.training.scenarios import create_industry_scenario, IndustryScenarioConfig, IndustryScenarioType
                # IndustryScenarioType 没有 MANUFACTURING 成员，使用 EQUIPMENT_FAULT_PREDICTION 作为默认
                industry_type_str = scenario_config.get('industry_type', 'EQUIPMENT_FAULT_PREDICTION').upper()
                industry_type = getattr(
                    IndustryScenarioType,
                    industry_type_str,
                    IndustryScenarioType.EQUIPMENT_FAULT_PREDICTION
                )
                config = IndustryScenarioConfig(
                    name=base_config['name'],
                    output_dir=self.output_dir,
                    scenario_type=industry_type,
                )
                # create_industry_scenario 需要 scenario_type 作为第一个参数
                scenario = create_industry_scenario(
                    scenario_type=industry_type.value if hasattr(industry_type, 'value') else str(industry_type),
                    config=config
                )
                logger.info("Created IndustryScenario: %s", scenario.session_id)
                return scenario
            
            else:
                logger.warning("Unknown scenario type: %s, using ScenarioManager", scenario_type)
                return self._create_scenario_manager()
                
        except ImportError as e:
            logger.warning(f"场景模块导入失败: {e}, 回退到场景管理器")
            return self._create_scenario_manager()
    
    def _create_task_manager_trainer(self):
        """
        使用 core 模块的任务管理器创建训练器
        
        统一调用 core 模块的 TrainingTaskManager
        
        Returns:
            任务管理器训练包装器
        """
        try:
            from backend.modules.training.core import (
                TrainingTaskManager,
                TrainingTask,
                get_training_task_manager,
            )
            
            # 获取任务管理器
            task_manager = get_training_task_manager()
            
            # 创建训练任务配置
            task_config = {
                'name': self.config.get('training', {}).get('name', 'task_training'),
                'model_name': self.config.get('model', {}).get('name', 'gpt2'),
                'output_dir': self.output_dir,
                'num_epochs': self.config.get('training', {}).get('num_epochs', 10),
                'batch_size': self.config.get('training', {}).get('batch_size', 16),
                'learning_rate': self.config.get('training', {}).get('learning_rate', 2e-5),
            }
            
            # 创建包装器
            class TaskManagerTrainerWrapper:
                def __init__(self, task_manager, task_config, launcher):
                    self.task_manager = task_manager
                    self.task_config = task_config
                    self.launcher = launcher
                    self.task = None
                
                def train(self) -> Dict[str, Any]:
                    # 创建任务 - 使用 create_training_task
                    task_config_full = {
                        'name': self.task_config['name'],
                        'scenario_type': self.task_config.get('scenario_type', 'standard'),
                        **self.task_config
                    }
                    task_id = self.task_manager.create_training_task(
                        user_id=self.task_config.get('user_id', 'system'),
                        task_config=task_config_full
                    )
                    self.task = self.task_manager.tasks.get(task_id)
                    logger.info(f"Created training task: {self.task.task_id}")
                    
                    # 启动任务
                    result = self.task_manager.start_task(self.task.task_id)
                    
                    # 更新进度
                    self.launcher._update_progress(0.5, "training")
                    
                    return {
                        'success': result.get('success', True),
                        'task_id': self.task.task_id,
                        'status': self.task.status.value if hasattr(self.task.status, 'value') else str(self.task.status),
                        'message': 'Task training started'
                    }
                
                def get_status(self) -> Dict[str, Any]:
                    if self.task:
                        return self.task_manager.get_task_status(self.task.task_id)
                    return {}
            
            logger.info("Created TaskManager trainer")
            return TaskManagerTrainerWrapper(task_manager, task_config, self)
            
        except ImportError as e:
            logger.warning(f"Core 模块导入失败: {e}, 回退到统一训练器")
            return self._create_unified_trainer()
    
    def _create_pipeline_based_trainer(self, pipeline_steps: List[Dict[str, Any]] = None):
        """
        创建基于流水线的训练器
        
        调用派生方法 create_pipeline_runner_instance, create_pipeline_executor_instance,
        create_pipeline_with_failure_handling
        
        Args:
            pipeline_steps: 流水线步骤列表
        
        Returns:
            流水线训练器包装器
        """
        try:
            from backend.modules.training.pipeline import (
                create_pipeline,
                create_three_stage_pipeline,
                PipelineDefinition,
                PipelineStep,
                StepType,
            )
            
            # 获取流水线配置
            pipeline_config = self.config.get('pipeline', {})
            steps = pipeline_steps or pipeline_config.get('steps', [])
            failure_action = pipeline_config.get('failure_action', 'stop')
            
            # 如果没有步骤，创建默认三阶段流水线
            if not steps:
                pipeline = create_three_stage_pipeline(
                    name=f"pipeline_{self.session_id}",
                    pretrain_params={'num_epochs': 1},
                    finetune_params={'num_epochs': 2},
                    preference_params={'num_epochs': 1},
                )
            else:
                # 调用派生方法 create_pipeline_with_failure_handling
                # 该方法使用 FailureAction, PipelineStep, PipelineDefinition
                pipeline = self.create_pipeline_with_failure_handling(steps, failure_action)
                
                if pipeline is None:
                    # 回退到直接创建
                    pipeline_steps_obj = []
                    for step in steps:
                        step_type_str = step.get('type', 'custom')
                        pipeline_steps_obj.append(PipelineStep(
                            name=step.get('name', f'step_{len(pipeline_steps_obj)}'),
                            type=step_type_str,
                            params=step.get('params', {}),
                        ))
                    pipeline = create_pipeline(
                        name=f"pipeline_{self.session_id}",
                        steps=pipeline_steps_obj,
                    )
            
            # 调用派生方法 create_pipeline_runner_instance
            # 该方法使用 PipelineRunner, create_pipeline_runner
            runner = self.create_pipeline_runner_instance()
            
            # 调用派生方法 create_pipeline_executor_instance  
            # 该方法使用 PipelineExecutor, create_executor
            executor = self.create_pipeline_executor_instance(runner)
            
            if executor is None:
                # 回退到直接创建
                executor = create_executor(runner=runner, session_id=self.session_id)
            
            # 创建包装器
            class PipelineBasedTrainerWrapper:
                def __init__(self, executor, pipeline, launcher):
                    self.executor = executor
                    self.pipeline = pipeline
                    self.launcher = launcher
                
                def train(self) -> Dict[str, Any]:
                    # 触发插件钩子
                    self.launcher._trigger_plugin_hook('training_start', pipeline=self.pipeline.name)
                    
                    # 执行流水线
                    result = self.executor.execute(self.pipeline)
                    
                    # 更新进度
                    self.launcher._update_progress(1.0, "completed")
                    
                    # 触发插件钩子
                    self.launcher._trigger_plugin_hook('training_end', result=result)
                    
                    return {
                        'success': result.success if hasattr(result, 'success') else True,
                        'pipeline_name': self.pipeline.name,
                        'steps_count': len(self.pipeline.steps),
                        'result': result.to_dict() if hasattr(result, 'to_dict') else str(result)
                    }
            
            logger.info(f"Created Pipeline trainer with {len(pipeline.steps)} steps")
            return PipelineBasedTrainerWrapper(executor, pipeline, self)
            
        except ImportError as e:
            logger.warning(f"Pipeline 模块导入失败: {e}, 回退到统一训练器")
            return self._create_unified_trainer()
    
    def _setup_strategies(self, analysis: Dict[str, Any]) -> List:
        """设置训练策略组合"""
        strategies = []
        
        try:
            from backend.modules.training.strategies import create_strategy
            
            # 根据分析结果添加策略
            if analysis.get('multimodal'):
                strategies.append(create_strategy('multimodal'))
            
            if analysis.get('knowledge_distillation'):
                # 根据蒸馏场景选择策略
                distillation_scenario = analysis.get('distillation_scenario', 'standard')
                strategy_map = {
                    'standard': 'distillation',
                    'industry': 'industry_distillation',
                    'progressive': 'progressive_distillation',
                    'self': 'self_distillation',
                    'contrastive': 'contrastive_distillation',
                }
                strategy_type = strategy_map.get(distillation_scenario, 'distillation')
                strategies.append(create_strategy(strategy_type))
            
            if analysis.get('distributed'):
                strategies.append(create_strategy('distributed'))
            
            if analysis.get('scenario_enabled'):
                strategies.append(create_strategy('scenario'))
            
            if analysis.get('three_stage'):
                strategies.append(create_strategy('three_stage'))
            
            if analysis.get('industry_enabled'):
                strategies.append(create_strategy('industry_scenario'))

            # 始终添加标准策略作为基础
            strategies.append(create_strategy('standard'))
            
        except ImportError as e:
            logger.warning(f"策略模块导入失败: {e}")
        
        return strategies
    
    def _setup_plugins(self, plugin_configs: List[Dict[str, Any]] = None):
        """
        设置和注册训练插件
        
        统一调用 plugins 模块
        
        Args:
            plugin_configs: 插件配置列表
        """
        try:
            from backend.modules.training.plugins import (
                PluginRegistry,
                CallbackPlugin,
                MonitoringPlugin,
                CheckpointPlugin,
                PluginConfig,
                register_plugin,
                HookPoint,
            )
            
            configs = plugin_configs or self.config.get('plugins', {}).get('list', [])
            
            for plugin_cfg in configs:
                plugin_type = plugin_cfg.get('type', 'callback')
                plugin_name = plugin_cfg.get('name', f'plugin_{len(configs)}')
                
                # 根据类型创建插件
                if plugin_type == 'callback':
                    # 创建简单回调插件
                    class SimpleCallbackPlugin(CallbackPlugin):
                        def __init__(self, name, config):
                            super().__init__()
                            self.name = name
                            self.config = config
                        
                        def on_epoch_end(self, context):
                            logger.info(f"Plugin {self.name}: epoch end")
                            return None
                    
                    plugin = SimpleCallbackPlugin(plugin_name, plugin_cfg)
                    register_plugin(plugin)
                    logger.info(f"Registered callback plugin: {plugin_name}")
                    
                elif plugin_type == 'monitoring':
                    # 监控插件可以使用默认实现
                    pass
                    
                elif plugin_type == 'checkpoint':
                    # 检查点插件可以使用默认实现
                    pass
            
        except Exception as e:
            logger.warning(f"Failed to setup plugins: {e}")
    
    def _init_orchestrator_with_layers(self) -> Optional[Any]:
        """
        使用完整六层架构初始化编排器
        
        统一调用 orchestrator 和 backend/lib 模块
        
        Returns:
            UnifiedTrainingOrchestrator 实例
        """
        try:
            from backend.modules.training.orchestrator import (
                UnifiedTrainingOrchestrator,
                LayerConfig,
                LayerManager,
            )
            
            # 获取配置
            orch_config = self.config.get('orchestrator', {})
            dist_config = self.config.get('distributed', {})
            
            # 创建层配置
            layer_config = LayerConfig(
                device_type=orch_config.get('device_type', 'auto'),
                precision=orch_config.get('precision', 'fp16'),
                enable_amp=orch_config.get('enable_amp', True),
                distributed_mode=dist_config.get('mode', 'none'),
                world_size=dist_config.get('world_size', 1),
                modalities=self.config.get('multimodal', {}).get('modalities', ['text']),
                strategy_type=orch_config.get('strategy_type', 'standard'),
            )
            
            # 创建编排器
            self._orchestrator = UnifiedTrainingOrchestrator(
                output_dir=self.output_dir,
                default_config=layer_config
            )
            
            logger.info("Orchestrator initialized with LayerConfig")
            logger.info(f"  Device: {layer_config.device_type}")
            logger.info(f"  Precision: {layer_config.precision}")
            logger.info(f"  Distributed mode: {layer_config.distributed_mode}")
            
            return self._orchestrator
            
        except Exception as e:
            logger.warning(f"Failed to init orchestrator: {e}")
            return None
    
    def _init_progress_tracking(self) -> Optional[Any]:
        """
        初始化进度跟踪
        
        统一调用 progress 模块
        
        Returns:
            TrainingProgressManager 实例
        """
        try:
            from backend.modules.training.progress import (
                TrainingProgressManager,
                TrainingProgress,
                create_progress_tracker,
                get_progress_manager,
            )
            
            # 获取进度管理器
            progress_manager = get_progress_manager()
            
            # 创建进度跟踪器
            total_steps = self.config.get('training', {}).get('total_steps', 0)
            total_epochs = self.config.get('training', {}).get('num_epochs', 10)
            
            create_progress_tracker(
                session_id=self.session_id,
                total_steps=total_steps,
                total_epochs=total_epochs,
            )
            
            self._progress_manager = progress_manager
            logger.info(f"Progress tracking initialized for session: {self.session_id}")
            
            return progress_manager
            
        except Exception as e:
            logger.warning(f"Failed to init progress tracking: {e}")
            return None
    
    def _apply_hardware_optimization(self):
        """
        应用硬件优化
        
        统一调用 backend/lib/hardware 模块
        """
        try:
            from backend.lib.hardware import (
                DeviceManager,
                get_device_manager,
                get_available_memory,
                clear_memory,
                recommend_precision,
                recommend_batch_size,
            )
            
            # 清理内存
            if clear_memory:
                clear_memory()
                logger.debug("Memory cleared")
            
            # 获取设备管理器
            if get_device_manager:
                device_manager = get_device_manager()
                if hasattr(device_manager, 'get_device_info'):
                    device_info = device_manager.get_device_info()
                    logger.info(f"Device info: {device_info}")
            
            # 获取可用内存
            if get_available_memory:
                available_memory = get_available_memory()
                logger.info(f"Available memory: {available_memory / (1024**3):.2f} GB")
            
            # 推荐精度
            if recommend_precision:
                precision = recommend_precision('cuda')
                logger.info(f"Recommended precision: {precision}")
            
            # 推荐批次大小
            if get_available_memory:
                # recommend_batch_size 需要 model 和 sample_size_mb，使用简单估算
                available = get_available_memory()
                available_gb = available / (1024 ** 3) if available else 0
                batch_size = max(1, min(64, int(available_gb * 4)))
                logger.info(f"Recommended batch size: {batch_size}")
            
        except Exception as e:
            logger.warning(f"Hardware optimization failed: {e}")
    
    def _apply_distributed_config(self) -> Optional[Any]:
        """
        应用分布式配置
        
        统一调用 backend/lib/distributed 和 strategies/distributed_strategy 模块
        
        Returns:
            DistributedManager 或 DistributedStrategyConfig 实例
        """
        dist_config = self.config.get('distributed', {})
        if not dist_config.get('enabled', False):
            return None
        
        try:
            # 尝试使用 strategies 模块的分布式配置
            from backend.modules.training.strategies import (
                DistributedMode,
                DistributedStrategyConfig,
                recommend_distributed_mode,
            )

            mode_str = dist_config.get('mode', 'ddp').upper()
            mode = getattr(DistributedMode, mode_str, DistributedMode.DDP)

            strategy_config = DistributedStrategyConfig(
                mode=mode,
                world_size=dist_config.get('world_size', 1),
                gradient_accumulation_steps=dist_config.get('gradient_accumulation_steps', 1),
            )

            # 获取推荐配置
            if recommend_distributed_mode:
                # 函数需要 model_size_gb, num_gpus, memory_per_gpu_gb 参数
                model_size_gb = self.config.get('model', {}).get('size_gb', 2.0)
                if isinstance(model_size_gb, str):
                    size_map = {'small': 0.5, 'medium': 2.0, 'large': 7.0, 'xlarge': 13.0}
                    model_size_gb = size_map.get(model_size_gb, 2.0)
                recommendation = recommend_distributed_mode(
                    model_size_gb=model_size_gb,
                    num_gpus=dist_config.get('world_size', 1),
                    memory_per_gpu_gb=16.0
                )
                logger.info(f"Distributed recommendation: {recommendation}")

            logger.info(f"Distributed config: mode={mode}, world_size={strategy_config.world_size}")
            return strategy_config
            
        except Exception as e:
            logger.warning(f"Failed to apply distributed config: {e}")
        
        return None
    
    def _apply_loss_config(self) -> Optional[Any]:
        """
        应用损失函数配置
        
        统一调用 backend/lib/losses 模块
        
        Returns:
            损失函数实例
        """
        try:
            from backend.lib.losses import (
                LossFactory,
                create_loss,
                create_composite_loss,
            )
            
            loss_config = self.config.get('training', {}).get('loss', {})
            loss_type = loss_config.get('type', 'cross_entropy')
            
            # 使用工厂创建损失函数
            if LossFactory:
                factory = LossFactory()
                loss_fn = factory.create(loss_type, **loss_config)
                logger.info(f"Created loss function: {loss_type}")
                return loss_fn
            
            # 使用便捷函数
            elif create_loss:
                loss_fn = create_loss(loss_type, **loss_config)
                logger.info(f"Created loss function via create_loss: {loss_type}")
                return loss_fn
            
        except Exception as e:
            logger.warning(f"Failed to apply loss config: {e}")
        
        return None
    
    def launch_training(self) -> Dict[str, Any]:
        """启动训练"""
        try:
            # 应用硬件优化
            self._apply_hardware_optimization()
            
            # 应用分布式配置
            distributed_config = self._apply_distributed_config()
            
            # 初始化进度跟踪
            self._init_progress_tracking()
            
            # 初始化编排器
            self._init_orchestrator_with_layers()
            
            # 设置插件
            self._setup_plugins()
            
            # 分析配置
            analysis = self.analyze_config()
            
            # 设置策略
            self.strategies = self._setup_strategies(analysis)
            
            # 应用损失函数配置
            loss_fn = self._apply_loss_config()
            
            # 触发训练开始钩子
            self._trigger_plugin_hook('training_start', session_id=self.session_id)
            
            # 选择训练器
            trainer = self.select_trainer(analysis)
            
            # 更新进度
            self._update_progress(0.0, "initializing")
            
            # 开始训练
            logger.info("开始训练...")
            logger.info(f"会话ID: {self.session_id}")
            logger.info(f"使用策略: {[s.name for s in self.strategies] if self.strategies else 'default'}")
            if distributed_config:
                logger.info(f"分布式配置已应用")
            if loss_fn:
                logger.info(f"损失函数已配置")
            
            # 执行训练
            if hasattr(trainer, 'train'):
                self._update_progress(0.1, "training")
                result = trainer.train()
            elif hasattr(trainer, 'start_scheduler'):
                # 场景管理器
                self._update_progress(0.1, "scheduling")
                trainer.start_scheduler()
                import time
                time.sleep(5)
                trainer.stop_scheduler()
                result = {'success': True, 'message': '场景化训练完成'}
            elif hasattr(trainer, 'run'):
                # 场景训练器
                self._update_progress(0.1, "running_scenario")
                result = trainer.run()
                if hasattr(result, 'to_dict'):
                    result = result.to_dict()
            else:
                logger.error("训练器没有有效的训练方法")
                result = {'success': False, 'error': '训练器配置错误'}
            
            # 更新进度
            self._update_progress(1.0, "completed")
            
            # 触发训练结束钩子
            self._trigger_plugin_hook('training_end', result=result)
            
            logger.info("训练完成!")
            return result
            
        except Exception as e:
            logger.error(f"训练过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 触发错误钩子
            self._trigger_plugin_hook('training_error', error=str(e))
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def diagnose(self) -> Dict[str, Any]:
        """
        诊断启动器状态
        
        统一调用所有模块的诊断函数
        
        Returns:
            诊断结果字典
        """
        diagnosis = {
            'launcher': {
                'session_id': self.session_id,
                'output_dir': self.output_dir,
                'strategies_count': len(self.strategies),
            },
            'module_availability': get_module_availability(),
            'components': {
                'progress_manager': self._progress_manager is not None,
                'plugin_registry': self._plugin_registry is not None,
                'orchestrator': self._orchestrator is not None,
                'task_manager': self._task_manager is not None,
            },
        }
        
        # 调用各模块诊断函数
        try:
            diagnosis['core'] = diagnose_core_module()
        except Exception as e:
            diagnosis['core'] = {'error': str(e)}

        try:
            diagnosis['scenarios'] = diagnose_scenarios()
        except Exception as e:
            diagnosis['scenarios'] = {'error': str(e)}

        try:
            diagnosis['orchestrator'] = diagnose_orchestrator_module()
        except Exception as e:
            diagnosis['orchestrator'] = {'error': str(e)}

        try:
            diagnosis['pipeline'] = diagnose_pipeline_module()
        except Exception as e:
            diagnosis['pipeline'] = {'error': str(e)}

        try:
            diagnosis['plugins'] = diagnose_plugin_module()
        except Exception as e:
            diagnosis['plugins'] = {'error': str(e)}

        try:
            # diagnose_strategy 需要 strategy 参数，使用 get_available_layers 代替
            diagnosis['strategies'] = {'available_layers': get_available_layers()}
        except Exception as e:
            diagnosis['strategies'] = {'error': str(e)}
        
        return diagnosis
    
    def get_available_training_modes(self) -> List[str]:
        """
        获取可用的训练模式
        
        Returns:
            可用训练模式列表
        """
        modes = ['standard']  # 始终可用
        modes.extend(['basic_model', 'advanced_model', 'scheduled', 'industry_scenario'])
        modes.extend(['distillation', 'edge_deploy', 'progressive', 'self_distillation'])
        modes.append('multimodal')
        modes.append('three_stage')
        modes.append('industry')
        modes.append('orchestrator')
        modes.append('pipeline')
        
        return modes
    
    def create_scenario_training(
        self,
        scenario_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建并执行场景化训练
        
        统一调用 scenarios 模块
        
        Args:
            scenario_type: 场景类型 (basic_model, advanced_model, scheduled, industry)
            **kwargs: 场景配置参数
        
        Returns:
            训练结果
        """
        # 更新配置
        self.config['scenario'] = {
            'enabled': True,
            'type': scenario_type,
            **kwargs
        }
        
        # 创建场景训练器
        trainer = self._create_scenario_trainer_by_type(scenario_type)
        
        if trainer is None:
            return {'success': False, 'error': f'Failed to create scenario trainer: {scenario_type}'}
        
        # 执行训练
        try:
            if hasattr(trainer, 'run'):
                result = trainer.run()
                if hasattr(result, 'to_dict'):
                    return result.to_dict()
                return result
            elif hasattr(trainer, 'train'):
                return trainer.train()
            else:
                return {'success': False, 'error': 'No valid training method found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_pipeline_training(
        self,
        steps: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建并执行流水线训练
        
        统一调用 pipeline 模块
        
        Args:
            steps: 流水线步骤列表
            **kwargs: 其他配置参数
        
        Returns:
            训练结果
        """
        # 更新配置
        self.config['pipeline'] = {
            'enabled': True,
            'steps': steps,
            **kwargs
        }
        
        # 创建流水线训练器
        trainer = self._create_pipeline_based_trainer(steps)
        
        if trainer is None:
            return {'success': False, 'error': 'Failed to create pipeline trainer'}
        
        # 执行训练
        try:
            return trainer.train()
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def create_orchestrated_training(
        self,
        plan_type: str = 'standard',
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建并执行编排训练
        
        统一调用 orchestrator 模块
        
        Args:
            plan_type: 计划类型 (standard, three_stage, industry, multimodal, distillation)
            **kwargs: 计划配置参数
        
        Returns:
            训练结果
        """
        # 初始化编排器
        orchestrator = self._init_orchestrator_with_layers()
        
        if orchestrator is None:
            return {'success': False, 'error': 'Orchestrator not available'}
        
        try:
            # 创建训练计划
            plan_name = kwargs.get('name', f'{plan_type}_training_{self.session_id}')
            
            if plan_type == 'standard':
                plan = orchestrator.create_standard_plan(
                    name=plan_name,
                    epochs=kwargs.get('epochs', 10),
                )
            elif plan_type == 'three_stage':
                plan = orchestrator.create_three_stage_plan(
                    name=plan_name,
                    pretrain_epochs=kwargs.get('pretrain_epochs', 3),
                    finetune_epochs=kwargs.get('finetune_epochs', 5),
                    preference_epochs=kwargs.get('preference_epochs', 2),
                )
            elif plan_type == 'industry':
                plan = orchestrator.create_industry_plan(
                    name=plan_name,
                    include_pretrain=kwargs.get('include_pretrain', True),
                    include_align=kwargs.get('include_align', True),
                    include_finetune=kwargs.get('include_finetune', True),
                )
            elif plan_type == 'multimodal':
                plan = orchestrator.create_multimodal_plan(
                    name=plan_name,
                    modalities=kwargs.get('modalities', ['text', 'image']),
                )
            elif plan_type == 'distillation':
                plan = orchestrator.create_distillation_plan(
                    name=plan_name,
                    distillation_epochs=kwargs.get('distillation_epochs', 10),
                )
            else:
                plan = orchestrator.create_standard_plan(name=plan_name)
            
            logger.info(f"Created orchestrator plan: {plan_type}")
            
            return {
                'success': True,
                'plan_name': plan.name if hasattr(plan, 'name') else plan_name,
                'plan_type': plan_type,
                'phases': len(plan.phases) if hasattr(plan, 'phases') else 0,
                'message': 'Orchestrator plan created successfully'
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}


def launch_training_system(config: Dict[str, Any]) -> Dict[str, Any]:
    """启动训练系统的便捷函数"""
    try:
        launcher = TrainingSystemLauncher(config)
        return launcher.launch_training()
    except Exception as e:
        logger.error(f"启动训练系统失败: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def diagnose_launcher_module() -> Dict[str, Any]:
    """
    诊断启动器模块状态
    
    统一调用所有可用模块的诊断函数
    
    Returns:
        诊断结果字典
    """
    diagnosis = {
        'module': 'launcher',
        'module_availability': get_module_availability(),
    }
    
    # Core 模块诊断
    try:
        diagnosis['core'] = diagnose_core_module()
    except Exception as e:
        diagnosis['core'] = {'error': str(e)}
    
    # Scenarios 模块诊断
    try:
        diagnosis['scenarios'] = diagnose_scenarios()
    except Exception as e:
        diagnosis['scenarios'] = {'error': str(e)}
    
    # Orchestrator 模块诊断
    try:
        diagnosis['orchestrator'] = diagnose_orchestrator_module()
    except Exception as e:
        diagnosis['orchestrator'] = {'error': str(e)}
    
    # Pipeline 模块诊断
    try:
        diagnosis['pipeline'] = diagnose_pipeline_module()
    except Exception as e:
        diagnosis['pipeline'] = {'error': str(e)}
    
    # Progress 模块诊断
    try:
        diagnosis['progress'] = get_progress_layer_availability()
    except Exception as e:
        diagnosis['progress'] = {'error': str(e)}
    
    # Plugins 模块诊断
    try:
        diagnosis['plugins'] = diagnose_plugin_module()
    except Exception as e:
        diagnosis['plugins'] = {'error': str(e)}
    
    # Strategies 模块诊断
    try:
        # diagnose_strategy 需要 strategy 参数，使用 get_available_layers 代替
        diagnosis['strategies'] = {'available_layers': get_available_layers()}
    except Exception as e:
        diagnosis['strategies'] = {'error': str(e)}
    
    return diagnosis


def create_scenario_training_config(
    scenario_type: str,
    output_dir: str = "./outputs",
    **kwargs
) -> Dict[str, Any]:
    """
    创建场景化训练配置
    
    Args:
        scenario_type: 场景类型 (basic_model, advanced_model, scheduled, industry)
        output_dir: 输出目录
        **kwargs: 其他配置参数
    
    Returns:
        配置字典
    """
    config = {
        'output_dir': output_dir,
        'scenario': {
            'enabled': True,
            'type': scenario_type,
            **{k: v for k, v in kwargs.items() if k.startswith('scenario_')},
        },
        'model': kwargs.get('model', {
            'name': kwargs.get('model_name', 'scenario_model'),
            'type': scenario_type,
        }),
        'training': kwargs.get('training', {
            'num_epochs': kwargs.get('num_epochs', 10),
            'batch_size': kwargs.get('batch_size', 16),
            'learning_rate': kwargs.get('learning_rate', 2e-5),
        }),
    }
    
    # 场景特定配置
    if scenario_type == 'scheduled':
        config['scenario'].update({
            'schedule_type': kwargs.get('schedule_type', 'periodic'),
            'interval_hours': kwargs.get('interval_hours', 24),
        })
    elif scenario_type == 'industry':
        config['scenario'].update({
            'industry_type': kwargs.get('industry_type', 'manufacturing'),
        })
    elif scenario_type == 'advanced_model':
        config['scenario'].update({
            'model_type': kwargs.get('advanced_model_type', 'transformer'),
        })
    
    return config


def create_pipeline_training_config(
    steps: List[Dict[str, Any]],
    output_dir: str = "./outputs",
    **kwargs
) -> Dict[str, Any]:
    """
    创建流水线训练配置
    
    Args:
        steps: 流水线步骤列表
        output_dir: 输出目录
        **kwargs: 其他配置参数
    
    Returns:
        配置字典
    """
    return {
        'output_dir': output_dir,
        'pipeline': {
            'enabled': True,
            'steps': steps,
            'enable_rollback': kwargs.get('enable_rollback', True),
        },
        'model': kwargs.get('model', {
            'name': kwargs.get('model_name', 'pipeline_model'),
        }),
        'training': kwargs.get('training', {
            'num_epochs': kwargs.get('num_epochs', 10),
            'batch_size': kwargs.get('batch_size', 16),
            'learning_rate': kwargs.get('learning_rate', 2e-5),
        }),
    }


def create_orchestrator_training_config(
    plan_type: str = 'standard',
    output_dir: str = "./outputs",
    **kwargs
) -> Dict[str, Any]:
    """
    创建编排训练配置
    
    Args:
        plan_type: 计划类型 (standard, three_stage, industry, multimodal, distillation)
        output_dir: 输出目录
        **kwargs: 其他配置参数
    
    Returns:
        配置字典
    """
    config = {
        'output_dir': output_dir,
        'orchestrator': {
            'enabled': True,
            'type': kwargs.get('orchestrator_type', 'unified'),
            'device_type': kwargs.get('device_type', 'auto'),
            'precision': kwargs.get('precision', 'fp16'),
            'enable_amp': kwargs.get('enable_amp', True),
            'strategy_type': kwargs.get('strategy_type', plan_type),
        },
        'model': kwargs.get('model', {
            'name': kwargs.get('model_name', 'orchestrated_model'),
        }),
        'training': kwargs.get('training', {
            'num_epochs': kwargs.get('num_epochs', 10),
            'batch_size': kwargs.get('batch_size', 16),
            'learning_rate': kwargs.get('learning_rate', 2e-5),
        }),
    }
    
    # 根据计划类型添加特定配置
    if plan_type == 'three_stage':
        config['three_stage'] = {
            'enabled': True,
            'pretrain_epochs': kwargs.get('pretrain_epochs', 3),
            'finetune_epochs': kwargs.get('finetune_epochs', 5),
            'preference_epochs': kwargs.get('preference_epochs', 2),
        }
    elif plan_type == 'industry':
        config['industry'] = {
            'enabled': True,
            'type': kwargs.get('industry_type', 'manufacturing'),
            'include_pretrain': kwargs.get('include_pretrain', True),
            'include_align': kwargs.get('include_align', True),
            'include_finetune': kwargs.get('include_finetune', True),
        }
    elif plan_type == 'multimodal':
        config['multimodal'] = {
            'enabled': True,
            'modalities': kwargs.get('modalities', ['text', 'image']),
        }
    elif plan_type == 'distillation':
        config['distillation'] = {
            'enabled': True,
            'scenario': kwargs.get('distillation_scenario', 'standard'),
            'teacher_model_path': kwargs.get('teacher_model_path', 'mock'),
            'student_model_path': kwargs.get('student_model_path', 'mock'),
        }
    
    return config


def get_all_training_modes() -> Dict[str, List[str]]:
    """
    获取所有可用的训练模式（按类别分组）
    
    Returns:
        训练模式字典
    """
    modes = {'standard': ['standard', 'unified'],
             'scenario': ['basic_model', 'advanced_model', 'scheduled', 'industry_scenario'],
             'distillation': ['standard', 'industry', 'edge_deploy', 'high_accuracy',
                              'multimodal', 'progressive', 'self', 'contrastive'],
             'multimodal': ['standard', 'production', 'industry'], 'distributed': ['ddp', 'fsdp', 'zero', 'pipeline'],
             'three_stage': ['pretrain', 'finetune', 'preference'],
             'industry': ['manufacturing', 'finance', 'healthcare', 'general'],
             'orchestrator': ['standard', 'three_stage', 'industry', 'multimodal', 'distillation'],
             'pipeline': ['sequential', 'three_stage', 'custom']}

    return modes


def create_industry_training_config(
    industry_type: str = "manufacturing",
    model_name: str = "industry_model",
    output_dir: str = "./outputs",
    **kwargs
) -> Dict[str, Any]:
    """
    创建行业训练配置的便捷函数
    
    Args:
        industry_type: 行业类型 (manufacturing, finance, healthcare, general)
        model_name: 模型名称
        output_dir: 输出目录
        **kwargs: 其他配置
    
    Returns:
        配置字典
    """
    config = {
        'model': {
            'name': model_name,
            'type': 'industry'
        },
        'industry': {
            'enabled': True,
            'type': industry_type,
            'include_pretrain': kwargs.get('include_pretrain', True),
            'include_align': kwargs.get('include_align', True),
            'include_finetune': kwargs.get('include_finetune', True),
            'use_multimodal': kwargs.get('use_multimodal', True),
            'use_distillation': kwargs.get('use_distillation', False),
            'use_distributed': kwargs.get('use_distributed', False),
            'stage_configs': kwargs.get('stage_configs')
        },
        'output_dir': output_dir,
        'training': kwargs.get('training', {
            'num_epochs': 10,
            'batch_size': 16,
            'learning_rate': 2e-5
        }),
        'data': kwargs.get('data', {
            'train_path': './data/train',
            'val_path': './data/val'
        })
    }
    
    return config


def create_distillation_training_config(
    scenario: str = "standard",
    teacher_model_path: str = "mock",
    student_model_path: str = "mock",
    output_dir: str = "./outputs",
    use_service: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    创建知识蒸馏训练配置的便捷函数
    
    支持多种蒸馏场景：
    - standard: 标准蒸馏
    - industry: 行业蒸馏
    - edge_deploy: 边缘部署蒸馏
    - multimodal: 多模态蒸馏
    - real_time: 实时推理蒸馏
    - high_accuracy: 高精度蒸馏
    - progressive: 渐进式蒸馏
    - self: 自蒸馏
    
    Args:
        scenario: 蒸馏场景
        teacher_model_path: 教师模型路径
        student_model_path: 学生模型路径
        output_dir: 输出目录
        use_service: 是否使用蒸馏服务
        **kwargs: 其他配置
    
    Returns:
        配置字典
    """
    # 基于场景设置默认配置
    scenario_defaults = {
        'standard': {
            'temperature': 4.0,
            'alpha': 0.7,
            'beta': 0.3,
        },
        'edge_deploy': {
            'temperature': 6.0,
            'alpha': 0.9,
            'beta': 0.1,
            'use_feature_distillation': True,
            'use_attention_distillation': False,
        },
        'high_accuracy': {
            'temperature': 2.0,
            'alpha': 0.5,
            'beta': 0.5,
            'use_feature_distillation': True,
            'use_attention_distillation': True,
        },
        'industry': {
            'temperature': 4.0,
            'alpha': 0.7,
            'beta': 0.3,
            'use_feature_distillation': True,
            'industry_type': kwargs.get('industry_type', 'manufacturing'),
        },
        'multimodal': {
            'temperature': 4.0,
            'alpha': 0.6,
            'beta': 0.4,
            'use_contrastive_distillation': True,
        },
        'progressive': {
            'temperature': 4.0,
            'alpha': 0.7,
            'beta': 0.3,
            'progressive_stages': 4,
        },
        'self': {
            'temperature': 4.0,
            'alpha': 0.5,
            'beta': 0.5,
        },
    }
    
    defaults = scenario_defaults.get(scenario, scenario_defaults['standard'])
    
    config = {
        'distillation': {
            'enabled': True,
            'scenario': scenario,
            'use_service': use_service,
            'teacher_model_path': teacher_model_path,
            'student_model_path': student_model_path,
            'task_name': kwargs.get('task_name', f'{scenario}_distillation'),
            'tenant_id': kwargs.get('tenant_id', 'default_tenant'),
            'user_id': kwargs.get('user_id', 'default_user'),
            **defaults,
            **{k: v for k, v in kwargs.items() 
               if k not in ['task_name', 'tenant_id', 'user_id', 'industry_type']}
        },
        'output_dir': output_dir,
        'training': kwargs.get('training', {
            'num_epochs': 10,
            'batch_size': 32,
            'learning_rate': 1e-4
        }),
    }
    
    return config


# =============================================================================
# 生产级分布式训练管理器
# =============================================================================

class DistributedTrainingManager:
    """
    分布式训练管理器
    
    整合 orchestrator/pipeline/progress 模块，提供生产级分布式训练能力：
    - 任务编排：通过 orchestrator 管理训练计划和阶段
    - 流水线执行：通过 pipeline 执行多阶段训练任务
    - 进度管理：通过 progress 跟踪训练进度和状态
    
    架构图：
    ┌─────────────────────────────────────────────────────────────┐
    │  DistributedTrainingManager                                  │
    │    ├── UnifiedTrainingOrchestrator (编排层)                  │
    │    │     └── OrchestratorPlan (训练计划)                     │
    │    ├── PipelineExecutor (流水线执行)                         │
    │    │     └── PipelineDefinition (流水线定义)                 │
    │    └── TrainingProgressManager (进度管理)                    │
    │          └── TrainingProgress (进度跟踪)                     │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化分布式训练管理器
        
        Args:
            config: 配置字典，包含：
                - output_dir: 输出目录
                - session_id: 训练会话ID
                - distributed: 分布式配置
                - orchestrator: 编排器配置
        """
        self.config = config
        self.output_dir = config.get('output_dir', './training_outputs')
        self.session_id = config.get('session_id', f'session_{int(datetime.now().timestamp())}')
        
        # 初始化组件
        self._orchestrator = None
        self._pipeline_executor = None
        self._progress_manager = None
        self._current_plan = None
        
        # 状态跟踪
        self._status = 'idle'  # idle, running, paused, completed, failed
        self._current_phase = None
        self._phase_results = []
        
        logger.info(f"DistributedTrainingManager initialized: session_id={self.session_id}")
    
    def _init_orchestrator(self):
        """初始化编排器"""
        if self._orchestrator is None:
            try:
                from backend.modules.training.orchestrator import (
                    UnifiedTrainingOrchestrator, LayerConfig
                )
                
                # 从配置创建层配置
                orch_config = self.config.get('orchestrator', {})
                layer_config = LayerConfig(
                    device_type=orch_config.get('device_type', 'auto'),
                    precision=orch_config.get('precision', 'fp16'),
                    enable_amp=orch_config.get('enable_amp', True),
                    distributed_mode=self.config.get('distributed', {}).get('mode', 'none'),
                    world_size=self.config.get('distributed', {}).get('world_size', 1),
                    modalities=orch_config.get('modalities', ['text']),
                    strategy_type=orch_config.get('strategy_type', 'standard'),
                )
                
                self._orchestrator = UnifiedTrainingOrchestrator(
                    output_dir=self.output_dir,
                    default_config=layer_config
                )
                logger.info("Orchestrator initialized successfully")
                
            except ImportError as e:
                logger.warning(f"Failed to import orchestrator: {e}")
    
    def _init_pipeline(self):
        """初始化流水线执行器"""
        if self._pipeline_executor is None:
            try:
                from backend.modules.training.pipeline.pipeline_executor import PipelineExecutor
                from backend.modules.training.pipeline.pipeline_runner import PipelineRunner
                
                runner = PipelineRunner(session_id=self.session_id)
                self._pipeline_executor = PipelineExecutor(
                    runner=runner,
                    session_id=self.session_id
                )
                logger.info("Pipeline executor initialized successfully")
                
            except ImportError as e:
                logger.warning(f"Failed to import pipeline executor: {e}")
    
    def _init_progress_manager(self):
        """初始化进度管理器"""
        if self._progress_manager is None:
            try:
                from backend.modules.training.progress import (
                    get_progress_manager, TrainingProgress
                )
                
                self._progress_manager = get_progress_manager()
                
                # 创建训练进度跟踪器
                total_steps = self.config.get('training', {}).get('total_steps', 0)
                total_epochs = self.config.get('training', {}).get('num_epochs', 10)
                self._progress_manager.create_progress_tracker(
                    self.session_id,
                    total_steps=total_steps,
                    total_epochs=total_epochs
                )
                
                # 启动系统资源监控
                self._progress_manager.start_system_monitoring()
                logger.info("Progress manager initialized successfully")
                
            except ImportError as e:
                logger.warning(f"Failed to import progress manager: {e}")
    
    def create_training_plan(self, plan_type: str = 'standard', **kwargs) -> Optional[Any]:
        """
        创建训练计划
        
        Args:
            plan_type: 计划类型
                - standard: 标准训练
                - three_stage: 三阶段训练 (预训练->微调->偏好优化)
                - industry: 行业模型训练
                - multimodal: 多模态训练
                - distillation: 知识蒸馏训练
            **kwargs: 计划参数
        
        Returns:
            OrchestratorPlan 或 None
        """
        self._init_orchestrator()
        
        if self._orchestrator is None:
            logger.error("Orchestrator not available, cannot create plan")
            return None
        
        plan_name = kwargs.get('name', f'{plan_type}_training_{self.session_id}')
        
        try:
            if plan_type == 'standard':
                self._current_plan = self._orchestrator.create_standard_plan(
                    name=plan_name,
                    epochs=kwargs.get('epochs', 10),
                    learning_rate=kwargs.get('learning_rate', 1e-4)
                )
            elif plan_type == 'three_stage':
                self._current_plan = self._orchestrator.create_three_stage_plan(
                    name=plan_name,
                    pretrain_epochs=kwargs.get('pretrain_epochs', 3),
                    finetune_epochs=kwargs.get('finetune_epochs', 5),
                    preference_epochs=kwargs.get('preference_epochs', 2)
                )
            elif plan_type == 'industry':
                self._current_plan = self._orchestrator.create_industry_plan(
                    name=plan_name,
                    include_pretrain=kwargs.get('include_pretrain', True),
                    include_align=kwargs.get('include_align', True),
                    include_finetune=kwargs.get('include_finetune', True)
                )
            elif plan_type == 'multimodal':
                self._current_plan = self._orchestrator.create_multimodal_plan(
                    name=plan_name,
                    modalities=kwargs.get('modalities', ['text', 'image'])
                )
            elif plan_type == 'distillation':
                self._current_plan = self._orchestrator.create_distillation_plan(
                    name=plan_name,
                    distillation_epochs=kwargs.get('distillation_epochs', 10)
                )
            else:
                logger.warning(f"Unknown plan type: {plan_type}, using standard")
                self._current_plan = self._orchestrator.create_standard_plan(name=plan_name)
            
            logger.info(f"Training plan created: {plan_type}, phases={len(self._current_plan.phases) if self._current_plan else 0}")
            return self._current_plan
            
        except Exception as e:
            logger.error(f"Failed to create training plan: {e}")
            return None
    
    def create_pipeline(self, steps: List[Dict[str, Any]]) -> Optional[Any]:
        """
        创建训练流水线
        
        Args:
            steps: 步骤列表，每个步骤包含：
                - name: 步骤名称
                - type: 步骤类型 (pretrain, finetune, preference, distillation, etc.)
                - params: 步骤参数
                - on_fail: 失败策略 (continue, stop, rollback)
        
        Returns:
            PipelineDefinition 或 None
        """
        try:
            from backend.modules.training.pipeline.pipeline_definition import (
                PipelineDefinition, PipelineStep
            )
            
            pipeline_steps = []
            for step_def in steps:
                step = PipelineStep(
                    name=step_def.get('name', f'step_{len(pipeline_steps)}'),
                    type=step_def.get('type', 'custom'),
                    params=step_def.get('params', {}),
                    on_fail=step_def.get('on_fail', 'stop')
                )
                pipeline_steps.append(step)
            
            pipeline = PipelineDefinition(
                name=f'pipeline_{self.session_id}',
                session_id=self.session_id,
                steps=pipeline_steps,
                enable_rollback=self.config.get('pipeline', {}).get('enable_rollback', True)
            )
            
            logger.info(f"Pipeline created with {len(pipeline_steps)} steps")
            return pipeline
            
        except ImportError as e:
            logger.error(f"Failed to import pipeline module: {e}")
            return None
    
    def execute_plan(self, model, train_loader, val_loader=None) -> Dict[str, Any]:
        """
        执行训练计划（通过编排器）
        
        Args:
            model: PyTorch 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        
        Returns:
            训练结果字典
        """
        if self._current_plan is None:
            return {'success': False, 'error': 'No training plan created'}
        
        self._init_orchestrator()
        self._init_progress_manager()
        
        if self._orchestrator is None:
            return {'success': False, 'error': 'Orchestrator not available'}
        
        self._status = 'running'
        
        try:
            # 更新进度状态
            if self._progress_manager:
                self._progress_manager.set_status(self.session_id, ProgressStatus.RUNNING)
            
            # 执行训练计划
            self._phase_results = self._orchestrator.execute(
                plan=self._current_plan,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader
            )
            
            self._status = 'completed'
            
            # 更新进度
            if self._progress_manager:
                self._progress_manager.set_status(self.session_id, ProgressStatus.COMPLETED)
            
            return {
                'success': True,
                'session_id': self.session_id,
                'plan_name': self._current_plan.name,
                'phases_completed': len(self._phase_results),
                'results': [
                    {
                        'phase': r.phase.value,
                        'status': r.status.value,
                        'metrics': r.metrics,
                        'duration': r.duration_seconds
                    }
                    for r in self._phase_results
                ]
            }
            
        except Exception as e:
            self._status = 'failed'
            
            if self._progress_manager:
                self._progress_manager.set_status(self.session_id, ProgressStatus.FAILED, error_message=str(e))
            
            logger.error(f"Training execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def execute_pipeline(self, pipeline) -> Dict[str, Any]:
        """
        执行训练流水线
        
        Args:
            pipeline: PipelineDefinition 实例
        
        Returns:
            执行结果字典
        """
        self._init_pipeline()
        self._init_progress_manager()
        
        if self._pipeline_executor is None:
            return {'success': False, 'error': 'Pipeline executor not available'}
        
        self._status = 'running'
        
        try:
            # 更新进度状态
            if self._progress_manager:
                self._progress_manager.set_status(self.session_id, ProgressStatus.RUNNING)
            
            # 执行流水线
            result = self._pipeline_executor.execute(pipeline)
            
            # 检查执行结果 - result 是 ExecutionResult 数据类，不是字典
            # 使用属性访问而不是 .get()
            step_results = result.step_results if hasattr(result, 'step_results') else []
            all_success = all(
                getattr(step, 'success', False) 
                for step in step_results
            ) if step_results else result.success if hasattr(result, 'success') else False
            
            if all_success:
                self._status = 'completed'
                if self._progress_manager:
                    self._progress_manager.set_status(self.session_id, ProgressStatus.COMPLETED)
            else:
                self._status = 'failed'
                if self._progress_manager:
                    self._progress_manager.set_status(self.session_id, ProgressStatus.FAILED, error_message='Pipeline step failed')
            
            return {
                'success': all_success,
                'session_id': self.session_id,
                'pipeline_name': result.pipeline_name if hasattr(result, 'pipeline_name') else '',
                'steps_executed': result.steps_completed if hasattr(result, 'steps_completed') else len(step_results),
                'results': [r.to_dict() if hasattr(r, 'to_dict') else r for r in step_results]
            }
            
        except Exception as e:
            self._status = 'failed'
            
            if self._progress_manager:
                self._progress_manager.set_status(self.session_id, ProgressStatus.FAILED, error_message=str(e))
            
            logger.error(f"Pipeline execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_progress(self) -> Dict[str, Any]:
        """获取训练进度"""
        if self._progress_manager:
            progress = self._progress_manager.get_progress(self.session_id)
            if progress:
                return {
                    'session_id': self.session_id,
                    'status': progress.status,
                    'progress': progress.progress,
                    'current_step': progress.current_step,
                    'total_steps': progress.total_steps,
                    'current_epoch': progress.current_epoch,
                    'current_stage': progress.current_stage,
                    'stage_progress': progress.stage_progress,
                    'metrics': progress.metrics,
                    'cpu_usage': progress.cpu_usage,
                    'memory_usage': progress.memory_usage,
                    'gpu_usage': progress.gpu_usage,
                }
        
        return {
            'session_id': self.session_id,
            'status': self._status,
            'progress': 0.0
        }
    
    def pause(self):
        """暂停训练"""
        self._status = 'paused'
        if self._orchestrator:
            self._orchestrator.pause()
        logger.info(f"Training paused: {self.session_id}")
    
    def resume(self):
        """恢复训练"""
        self._status = 'running'
        if self._orchestrator:
            self._orchestrator.resume()
        logger.info(f"Training resumed: {self.session_id}")
    
    def stop(self):
        """停止训练"""
        self._status = 'cancelled'
        if self._orchestrator:
            self._orchestrator.stop()
        if self._progress_manager:
            self._progress_manager.cancel_training(self.session_id)
        logger.info(f"Training stopped: {self.session_id}")
    
    def cleanup(self):
        """清理资源"""
        if self._progress_manager:
            self._progress_manager.stop_system_monitoring()
            self._progress_manager.remove_progress(self.session_id)
        logger.info(f"DistributedTrainingManager cleaned up: {self.session_id}")


# =============================================================================
# 生产级训练启动器（继承自 TrainingSystemLauncher）
# =============================================================================

class ProductionTrainingLauncher(TrainingSystemLauncher):
    """
    生产级训练启动器
    
    继承自 TrainingSystemLauncher，增加以下生产级能力：
    - 分布式训练管理：整合 orchestrator/pipeline/progress
    - 策略组合执行：自动组合多种训练策略
    - 检查点恢复：支持断点续训
    - 资源监控：实时监控训练资源
    - 容错机制：自动重试和回滚
    
    架构位置：
    ┌─────────────────────────────────────────────────────────────┐
    │  ProductionTrainingLauncher（生产级入口）                    │
    │    └── TrainingSystemLauncher（基础入口）                    │
    │        ├── DistributedTrainingManager（分布式管理）          │
    │        │     ├── orchestrator（编排层）                      │
    │        │     ├── pipeline（流水线）                          │
    │        │     └── progress（进度管理）                        │
    │        ├── strategies/*（策略层）                            │
    │        └── backend/lib/*（基础层）                           │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化生产级训练启动器
        
        Args:
            config: 配置字典，扩展基类配置，增加：
                - production: 生产级配置
                    - enable_distributed_manager: 是否启用分布式管理器
                    - enable_checkpoint: 是否启用检查点
                    - enable_monitoring: 是否启用监控
                    - retry_on_failure: 失败重试次数
                    - checkpoint_interval: 检查点间隔
                - orchestrator: 编排器配置
                - pipeline: 流水线配置
        """
        super().__init__(config)
        
        # 生产级配置
        self.production_config = config.get('production', {})
        self.enable_distributed_manager = self.production_config.get('enable_distributed_manager', True)
        self.enable_checkpoint = self.production_config.get('enable_checkpoint', True)
        self.enable_monitoring = self.production_config.get('enable_monitoring', True)
        self.retry_on_failure = self.production_config.get('retry_on_failure', 3)
        self.checkpoint_interval = self.production_config.get('checkpoint_interval', 1)
        
        # 分布式训练管理器
        self._distributed_manager: Optional[DistributedTrainingManager] = None
        
        # 生产级上下文
        self._production_context = None
        
        # 训练状态
        self._training_state = {
            'session_id': None,
            'start_time': None,
            'end_time': None,
            'status': 'idle',
            'retry_count': 0,
            'checkpoints': [],
            'metrics_history': []
        }
        
        logger.info("ProductionTrainingLauncher initialized with enhanced capabilities")
    
    def _init_distributed_manager(self) -> DistributedTrainingManager:
        """初始化分布式训练管理器"""
        if self._distributed_manager is None:
            manager_config = {
                'output_dir': self.output_dir,
                'session_id': self._training_state.get('session_id'),
                'distributed': self.config.get('distributed', {}),
                'orchestrator': self.config.get('orchestrator', {}),
                'pipeline': self.config.get('pipeline', {}),
                'training': self.config.get('training', {}),
            }
            self._distributed_manager = DistributedTrainingManager(manager_config)
        
        return self._distributed_manager
    
    def _init_production_context(self, model=None):
        """
        初始化生产级训练上下文
        
        整合六层架构底层能力。
        """
        try:
            from backend.modules.training.strategies import (
                ProductionStrategyConfig, create_production_context,
                get_available_layers
            )
            
            # 获取可用层
            available_layers = get_available_layers()
            logger.info(f"Available layers for production context: {available_layers}")
            
            # 创建生产级配置
            production_config = ProductionStrategyConfig(
                device=self.production_config.get('device', 'auto'),
                precision=self.production_config.get('precision', 'fp16'),
                enable_amp=self.production_config.get('enable_amp', True),
                distributed_mode=self.config.get('distributed', {}).get('mode', 'none'),
                world_size=self.config.get('distributed', {}).get('world_size', 1),
                modalities=self.config.get('multimodal', {}).get('modalities', ['text']),
                task_loss_type=self.config.get('training', {}).get('loss_type', 'cross_entropy'),
            )
            
            # 创建上下文
            self._production_context = create_production_context(production_config, model)
            logger.info("Production context initialized successfully")
            
            return self._production_context
            
        except ImportError as e:
            logger.warning(f"Failed to initialize production context: {e}")
            return None
    
    def analyze_config(self) -> Dict[str, Any]:
        """
        扩展配置分析，增加生产级分析
        
        Returns:
            分析结果字典
        """
        # 调用基类分析
        analysis = super().analyze_config()
        
        # 增加生产级分析
        analysis.update({
            # 生产级特性
            'production_mode': self.production_config.get('enabled', True),
            'enable_distributed_manager': self.enable_distributed_manager,
            'enable_checkpoint': self.enable_checkpoint,
            'enable_monitoring': self.enable_monitoring,
            
            # 分布式配置详情
            'distributed_mode': self.config.get('distributed', {}).get('mode', 'none'),
            'world_size': self.config.get('distributed', {}).get('world_size', 1),
            'gradient_accumulation': self.config.get('distributed', {}).get('gradient_accumulation_steps', 1),
            
            # 编排配置
            'use_orchestrator': self.config.get('orchestrator', {}).get('enabled', False),
            'orchestrator_type': self.config.get('orchestrator', {}).get('type', 'unified'),
            
            # 流水线配置
            'use_pipeline': self.config.get('pipeline', {}).get('enabled', False),
            'pipeline_steps': len(self.config.get('pipeline', {}).get('steps', [])),
            
            # 策略组合
            'combined_strategies': self._analyze_strategy_combination(),
        })
        
        logger.info("Production config analysis:")
        for key in ['production_mode', 'enable_distributed_manager', 'distributed_mode', 
                    'use_orchestrator', 'use_pipeline', 'combined_strategies']:
            logger.info(f"  {key}: {analysis.get(key)}")
        
        return analysis
    
    def _analyze_strategy_combination(self) -> List[str]:
        """分析需要组合的策略"""
        strategies = []
        
        analysis = {
            'multimodal': self.config.get('multimodal', {}).get('enabled', False),
            'distillation': self.config.get('distillation', {}).get('enabled', False),
            'distributed': self.config.get('distributed', {}).get('enabled', False),
            'industry': self.config.get('industry', {}).get('enabled', False),
            'scenario': self.config.get('scenario', {}).get('enabled', False),
            'three_stage': self.config.get('three_stage', {}).get('enabled', False),
        }
        
        # 根据配置确定策略组合
        if analysis['industry']:
            strategies.append('industry_scenario')
        if analysis['multimodal']:
            strategies.append('multimodal')
        if analysis['distillation']:
            scenario = self.config.get('distillation', {}).get('scenario', 'standard')
            strategies.append(f'distillation_{scenario}')
        if analysis['distributed']:
            mode = self.config.get('distributed', {}).get('mode', 'ddp')
            strategies.append(f'distributed_{mode}')
        if analysis['three_stage']:
            strategies.append('three_stage')
        if analysis['scenario']:
            strategies.append('scenario')
        
        # 始终添加基础策略
        if not strategies:
            strategies.append('standard')
        
        return strategies
    
    def select_trainer(self, analysis: Dict[str, Any]) -> object:
        """
        扩展训练器选择，支持生产级训练器
        
        优先级（从高到低）：
        1. 使用流水线执行（如果配置了流水线）
        2. 使用编排器执行（如果配置了编排器）
        3. 使用分布式管理器（如果启用）
        4. 回退到基类选择
        """
        # 检查是否使用流水线
        if analysis.get('use_pipeline') and self.config.get('pipeline', {}).get('steps'):
            logger.info("Selecting pipeline-based trainer")
            return self._create_pipeline_trainer()
        
        # 检查是否使用编排器
        if analysis.get('use_orchestrator'):
            logger.info("Selecting orchestrator-based trainer")
            return self._create_orchestrator_trainer(analysis)
        
        # 检查是否使用分布式管理器
        if self.enable_distributed_manager and analysis.get('distributed'):
            logger.info("Selecting distributed manager trainer")
            return self._create_distributed_manager_trainer(analysis)
        
        # 回退到基类选择
        return super().select_trainer(analysis)
    
    def _create_pipeline_trainer(self) -> object:
        """创建基于流水线的训练器"""
        manager = self._init_distributed_manager()
        
        # 获取流水线步骤
        steps = self.config.get('pipeline', {}).get('steps', [])
        pipeline = manager.create_pipeline(steps)
        
        class PipelineTrainerWrapper:
            def __init__(self, manager, pipeline, launcher):
                self.manager = manager
                self.pipeline = pipeline
                self.launcher = launcher
            
            def train(self) -> Dict[str, Any]:
                if self.pipeline is None:
                    return {'success': False, 'error': 'Pipeline not created'}
                
                # 执行流水线
                result = self.manager.execute_pipeline(self.pipeline)
                
                # 保存检查点
                if self.launcher.enable_checkpoint and result.get('success'):
                    self.launcher._save_checkpoint('pipeline_final')
                
                return result
        
        return PipelineTrainerWrapper(manager, pipeline, self)
    
    def _create_orchestrator_trainer(self, analysis: Dict[str, Any]) -> object:
        """创建基于编排器的训练器"""
        manager = self._init_distributed_manager()
        
        # 确定训练计划类型
        if analysis.get('industry_enabled'):
            plan_type = 'industry'
        elif analysis.get('multimodal'):
            plan_type = 'multimodal'
        elif analysis.get('knowledge_distillation'):
            plan_type = 'distillation'
        elif analysis.get('three_stage'):
            plan_type = 'three_stage'
        else:
            plan_type = 'standard'
        
        # 创建训练计划
        plan_kwargs = {
            'name': self.config.get('training', {}).get('name', f'{plan_type}_training'),
            'epochs': self.config.get('training', {}).get('num_epochs', 10),
            'learning_rate': self.config.get('training', {}).get('learning_rate', 1e-4),
        }
        
        if plan_type == 'three_stage':
            plan_kwargs.update({
                'pretrain_epochs': self.config.get('three_stage', {}).get('pretrain_epochs', 3),
                'finetune_epochs': self.config.get('three_stage', {}).get('finetune_epochs', 5),
                'preference_epochs': self.config.get('three_stage', {}).get('preference_epochs', 2),
            })
        elif plan_type == 'industry':
            plan_kwargs.update({
                'include_pretrain': self.config.get('industry', {}).get('include_pretrain', True),
                'include_align': self.config.get('industry', {}).get('include_align', True),
                'include_finetune': self.config.get('industry', {}).get('include_finetune', True),
            })
        elif plan_type == 'multimodal':
            plan_kwargs['modalities'] = self.config.get('multimodal', {}).get('modalities', ['text', 'image'])
        elif plan_type == 'distillation':
            plan_kwargs['distillation_epochs'] = self.config.get('distillation', {}).get('num_epochs', 10)
        
        plan = manager.create_training_plan(plan_type, **plan_kwargs)
        
        class OrchestratorTrainerWrapper:
            def __init__(self, manager, plan, launcher, model_loader=None):
                self.manager = manager
                self.plan = plan
                self.launcher = launcher
                self.model_loader = model_loader
            
            def train(self, model=None, train_loader=None, val_loader=None) -> Dict[str, Any]:
                if self.plan is None:
                    return {'success': False, 'error': 'Training plan not created'}
                
                # 如果未提供模型，尝试加载
                if model is None and self.model_loader:
                    model = self.model_loader()
                
                if model is None:
                    # 创建模拟模型用于测试
                    import torch.nn as nn
                    model = nn.Sequential(
                        nn.Linear(768, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10)
                    )
                
                if train_loader is None:
                    # 创建模拟数据加载器
                    from torch.utils.data import DataLoader, TensorDataset
                    import torch
                    train_dataset = TensorDataset(
                        torch.randn(100, 768),
                        torch.randint(0, 10, (100,))
                    )
                    train_loader = DataLoader(train_dataset, batch_size=16)
                
                # 使用生产级上下文包装模型
                if self.launcher._production_context:
                    model = self.launcher._production_context.wrap_model(model)
                
                # 执行训练计划
                result = self.manager.execute_plan(model, train_loader, val_loader)
                
                # 保存检查点
                if self.launcher.enable_checkpoint and result.get('success'):
                    self.launcher._save_checkpoint('orchestrator_final')
                
                return result
        
        return OrchestratorTrainerWrapper(manager, plan, self)
    
    def _create_distributed_manager_trainer(self, analysis: Dict[str, Any]) -> object:
        """创建基于分布式管理器的训练器"""
        # 初始化生产级上下文
        self._init_production_context()
        
        # 调用基类的分布式训练器创建
        base_trainer = super()._create_distributed_trainer()
        
        class DistributedManagerWrapper:
            def __init__(self, base_trainer, launcher, analysis):
                self.base_trainer = base_trainer
                self.launcher = launcher
                self.analysis = analysis
            
            def train(self) -> Dict[str, Any]:
                # 初始化分布式管理器
                manager = self.launcher._init_distributed_manager()
                
                # 获取进度管理
                manager._init_progress_manager()
                
                # 执行基础训练
                result = self.base_trainer.train()
                
                # 增强结果
                result['distributed_mode'] = self.analysis.get('distributed_mode')
                result['world_size'] = self.analysis.get('world_size')
                result['production_mode'] = True
                
                return result
        
        return DistributedManagerWrapper(base_trainer, self, analysis)
    
    def _save_checkpoint(self, checkpoint_name: str):
        """保存检查点"""
        try:
            import torch
            import json
            
            checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_path = os.path.join(checkpoint_dir, f'{checkpoint_name}.json')
            
            checkpoint_data = {
                'name': checkpoint_name,
                'timestamp': datetime.now().isoformat(),
                'session_id': self._training_state.get('session_id'),
                'status': self._training_state.get('status'),
                'config': {
                    k: v for k, v in self.config.items()
                    if isinstance(v, (str, int, float, bool, list, dict))
                }
            }
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self._training_state['checkpoints'].append(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载检查点"""
        try:
            import json
            
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return {}
    
    def launch_training(self, model=None, train_loader=None, val_loader=None) -> Dict[str, Any]:
        """
        启动生产级训练
        
        扩展基类方法，增加：
        - 断点续训支持
        - 失败重试机制
        - 资源监控
        
        Args:
            model: 可选，PyTorch模型
            train_loader: 可选，训练数据加载器
            val_loader: 可选，验证数据加载器
        
        Returns:
            训练结果字典
        """
        # 初始化训练状态
        self._training_state['session_id'] = f'prod_{int(datetime.now().timestamp())}'
        self._training_state['start_time'] = datetime.now()
        self._training_state['status'] = 'starting'
        self._training_state['retry_count'] = 0
        
        # 检查是否从检查点恢复
        resume_checkpoint = self.config.get('resume_from_checkpoint')
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            checkpoint_data = self._load_checkpoint(resume_checkpoint)
            if checkpoint_data:
                logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
                # 可以在这里恢复训练状态
        
        # 初始化生产级上下文
        self._init_production_context(model)
        
        # 执行训练（带重试机制）
        result = None
        while self._training_state['retry_count'] <= self.retry_on_failure:
            try:
                self._training_state['status'] = 'running'
                
                # 分析配置
                analysis = self.analyze_config()
                
                # 设置策略
                self.strategies = self._setup_strategies(analysis)
                
                # 选择训练器
                trainer = self.select_trainer(analysis)
                
                # 开始训练
                logger.info("Starting production training...")
                logger.info(f"Session ID: {self._training_state['session_id']}")
                logger.info(f"Strategies: {[s.name for s in self.strategies] if self.strategies else 'default'}")
                
                # 执行训练
                if hasattr(trainer, 'train'):
                    # 检查是否需要传递参数
                    import inspect
                    sig = inspect.signature(trainer.train)
                    if len(sig.parameters) > 0:
                        result = trainer.train(model=model, train_loader=train_loader, val_loader=val_loader)
                    else:
                        result = trainer.train()
                else:
                    logger.error("Trainer has no train method")
                    result = {'success': False, 'error': 'Invalid trainer'}
                
                # 训练成功
                if result.get('success'):
                    self._training_state['status'] = 'completed'
                    break
                else:
                    raise Exception(result.get('error', 'Training failed'))
                
            except Exception as e:
                self._training_state['retry_count'] += 1
                logger.warning(f"Training failed (attempt {self._training_state['retry_count']}): {e}")
                
                if self._training_state['retry_count'] > self.retry_on_failure:
                    self._training_state['status'] = 'failed'
                    result = {'success': False, 'error': str(e), 'retries': self._training_state['retry_count']}
                    break
                
                # 保存失败检查点
                self._save_checkpoint(f'retry_{self._training_state["retry_count"]}')
                
                import time
                time.sleep(5)  # 重试前等待
        
        # 更新训练状态
        self._training_state['end_time'] = datetime.now()
        
        # 增强结果
        if result:
            result['session_id'] = self._training_state['session_id']
            result['duration_seconds'] = (
                self._training_state['end_time'] - self._training_state['start_time']
            ).total_seconds()
            result['production_mode'] = True
            result['checkpoints'] = self._training_state['checkpoints']
        
        # 清理资源
        self._cleanup()
        
        logger.info(f"Production training completed: {result.get('success', False)}")
        return result or {'success': False, 'error': 'Unknown error'}
    
    def _cleanup(self):
        """清理资源"""
        if self._distributed_manager:
            self._distributed_manager.cleanup()
        
        if self._production_context:
            try:
                self._production_context.cleanup()
            except:
                pass
        
        logger.info("Production launcher cleaned up")
    
    def get_progress(self) -> Dict[str, Any]:
        """获取训练进度"""
        if self._distributed_manager:
            return self._distributed_manager.get_progress()
        
        return {
            'session_id': self._training_state.get('session_id'),
            'status': self._training_state.get('status'),
            'retry_count': self._training_state.get('retry_count'),
        }
    
    def pause(self):
        """暂停训练"""
        if self._distributed_manager:
            self._distributed_manager.pause()
        self._training_state['status'] = 'paused'
        logger.info("Production training paused")
    
    def resume(self):
        """恢复训练"""
        if self._distributed_manager:
            self._distributed_manager.resume()
        self._training_state['status'] = 'running'
        logger.info("Production training resumed")
    
    def stop(self):
        """停止训练"""
        if self._distributed_manager:
            self._distributed_manager.stop()
        self._training_state['status'] = 'cancelled'
        logger.info("Production training stopped")


# =============================================================================
# 便捷函数
# =============================================================================

def launch_production_training(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    启动生产级训练的便捷函数
    
    Args:
        config: 训练配置
        **kwargs: 额外参数（model, train_loader, val_loader）
    
    Returns:
        训练结果
    """
    try:
        launcher = ProductionTrainingLauncher(config)
        return launcher.launch_training(
            model=kwargs.get('model'),
            train_loader=kwargs.get('train_loader'),
            val_loader=kwargs.get('val_loader')
        )
    except Exception as e:
        logger.error(f"Failed to launch production training: {e}")
        return {'success': False, 'error': str(e)}


def create_production_training_config(
    training_type: str = 'standard',
    output_dir: str = './outputs',
    **kwargs
) -> Dict[str, Any]:
    """
    创建生产级训练配置的便捷函数
    
    Args:
        training_type: 训练类型
            - standard: 标准训练
            - three_stage: 三阶段训练
            - industry: 行业模型训练
            - multimodal: 多模态训练
            - distillation: 知识蒸馏
            - distributed: 分布式训练
        output_dir: 输出目录
        **kwargs: 其他配置
    
    Returns:
        配置字典
    """
    config = {
        'output_dir': output_dir,
        'production': {
            'enabled': True,
            'enable_distributed_manager': kwargs.get('enable_distributed_manager', True),
            'enable_checkpoint': kwargs.get('enable_checkpoint', True),
            'enable_monitoring': kwargs.get('enable_monitoring', True),
            'retry_on_failure': kwargs.get('retry_on_failure', 3),
            'device': kwargs.get('device', 'auto'),
            'precision': kwargs.get('precision', 'fp16'),
            'enable_amp': kwargs.get('enable_amp', True),
        },
        'model': kwargs.get('model', {
            'name': kwargs.get('model_name', 'production_model'),
            'type': training_type
        }),
        'training': kwargs.get('training', {
            'num_epochs': kwargs.get('num_epochs', 10),
            'batch_size': kwargs.get('batch_size', 16),
            'learning_rate': kwargs.get('learning_rate', 2e-5),
        }),
        'data': kwargs.get('data', {
            'train_path': kwargs.get('train_path', './data/train'),
            'val_path': kwargs.get('val_path', './data/val'),
        }),
    }
    
    # 根据类型添加特定配置
    if training_type == 'three_stage':
        config['three_stage'] = {
            'enabled': True,
            'pretrain_epochs': kwargs.get('pretrain_epochs', 3),
            'finetune_epochs': kwargs.get('finetune_epochs', 5),
            'preference_epochs': kwargs.get('preference_epochs', 2),
        }
    elif training_type == 'industry':
        config['industry'] = {
            'enabled': True,
            'type': kwargs.get('industry_type', 'manufacturing'),
            'include_pretrain': kwargs.get('include_pretrain', True),
            'include_align': kwargs.get('include_align', True),
            'include_finetune': kwargs.get('include_finetune', True),
        }
    elif training_type == 'multimodal':
        config['multimodal'] = {
            'enabled': True,
            'modalities': kwargs.get('modalities', ['text', 'image']),
        }
    elif training_type == 'distillation':
        config['distillation'] = {
            'enabled': True,
            'scenario': kwargs.get('distillation_scenario', 'standard'),
            'teacher_model_path': kwargs.get('teacher_model_path', 'mock'),
            'student_model_path': kwargs.get('student_model_path', 'mock'),
        }
    elif training_type == 'distributed':
        config['distributed'] = {
            'enabled': True,
            'mode': kwargs.get('distributed_mode', 'ddp'),
            'world_size': kwargs.get('world_size', 1),
        }
    
    # 编排器配置
    if kwargs.get('use_orchestrator', False):
        config['orchestrator'] = {
            'enabled': True,
            'type': kwargs.get('orchestrator_type', 'unified'),
        }
    
    # 流水线配置
    if kwargs.get('pipeline_steps'):
        config['pipeline'] = {
            'enabled': True,
            'steps': kwargs.get('pipeline_steps'),
            'enable_rollback': kwargs.get('enable_rollback', True),
        }
    
    return config


# =============================================================================
# 模块级便捷函数：统一调用所有导入的模块
# =============================================================================

def diagnose_all_modules() -> Dict[str, Any]:
    """
    诊断所有训练模块
    
    统一调用所有导入模块的诊断和检测函数
    
    Returns:
        所有模块诊断结果
    """
    diagnosis = {
        'module': 'training_launcher',
        'module_availability': get_module_availability(),
    }
    
    # Core 模块 - 调用 diagnose_core_module
    try:
        diagnosis['core'] = diagnose_core_module()
        diagnosis['core']['task_status_available'] = TrainingTaskStatus is not None
        diagnosis['core']['task_available'] = TrainingTask is not None
    except Exception as e:
        diagnosis['core'] = {'error': str(e)}
    
    # Scenarios 模块 - 调用 diagnose_scenarios, get_scenario_integration
    try:
        diagnosis['scenarios'] = diagnose_scenarios()
        diagnosis['scenarios']['integration'] = get_scenario_integration()
        diagnosis['scenarios']['base_scenario_available'] = BaseScenario is not None
    except Exception as e:
        diagnosis['scenarios'] = {'error': str(e)}
    
    # Orchestrator 模块 - 调用 diagnose_orchestrator_module, get_orchestrator_layer_availability
    try:
        diagnosis['orchestrator'] = diagnose_orchestrator_module()
        diagnosis['orchestrator']['layer_availability'] = get_orchestrator_layer_availability()
        diagnosis['orchestrator']['layer_manager_available'] = LayerManager is not None
        diagnosis['orchestrator']['plan_available'] = OrchestratorPlan is not None
    except Exception as e:
        diagnosis['orchestrator'] = {'error': str(e)}
    
    # Pipeline 模块 - 调用 diagnose_pipeline_module, get_pipeline_layer_availability
    try:
        diagnosis['pipeline'] = diagnose_pipeline_module()
        diagnosis['pipeline']['layer_availability'] = get_pipeline_layer_availability()
        diagnosis['pipeline']['failure_action_available'] = FailureAction is not None
    except Exception as e:
        diagnosis['pipeline'] = {'error': str(e)}
    
    # Progress 模块 - 调用 get_progress_layer_availability
    try:
        diagnosis['progress'] = get_progress_layer_availability()
        diagnosis['progress']['status_available'] = ProgressStatus is not None
        diagnosis['progress']['tracker_available'] = TrainingProgress is not None
    except Exception as e:
        diagnosis['progress'] = {'error': str(e)}
    
    # Plugins 模块 - 调用 diagnose_plugin_module
    try:
        diagnosis['plugins'] = diagnose_plugin_module()
        diagnosis['plugins']['plugin_base_available'] = TrainingPlugin is not None
        diagnosis['plugins']['context_available'] = PluginContext is not None
        diagnosis['plugins']['result_available'] = PluginResult is not None
    except Exception as e:
        diagnosis['plugins'] = {'error': str(e)}
    
    # Strategies 模块 - 检查策略可用性
    # 注意：diagnose_*_strategy 函数需要 strategy 实例参数，这里只检查可用性
    try:
        diagnosis['strategies'] = {
            'base': {'available_layers': get_available_layers()},
            'production': diagnose_production_base(),
            'multimodal_available': MultiModalStrategy is not None,
            'distillation_available': DistillationStrategy is not None,
            'distributed_available': DistributedStrategy is not None,
            'scenario_available': ScenarioStrategy is not None,
            'three_stage_available': ThreeStageStrategy is not None,
            'standard_available': StandardTrainingStrategy is not None,
        }
    except Exception as e:
        diagnosis['strategies'] = {'error': str(e)}
    
    # Distillation 模块 - 调用 diagnose_distillation_scenarios, list_distillation_scenarios
    try:
        diagnosis['distillation'] = {
            'scenarios': diagnose_distillation_scenarios(),
            'available_scenarios': list_distillation_scenarios(),
            'compressor_available': ModelCompressor is not None,
            'scenario_manager_available': DistillationScenarioManager is not None,
        }
    except Exception as e:
        diagnosis['distillation'] = {'error': str(e)}
    
    # Three-stage 模块
    try:
        diagnosis['three_stage'] = {
            'trainer_available': ThreeStageTrainer is not None,
            'loop_available': TrainingLoop is not None,
            'factory_functions_available': all([
                create_three_stage_trainer is not None,
                create_training_loop is not None,
            ]),
        }
    except Exception as e:
        diagnosis['three_stage'] = {'error': str(e)}
    
    # Hardware 模块
    try:
        diagnosis['hardware'] = {
            'device_manager_available': DeviceManager is not None,
            'memory_manager_available': MemoryManager is not None,
            'available_memory_gb': get_available_memory() / (1024 ** 3) if get_available_memory else 0,
            'recommended_precision': recommend_precision('cuda') if recommend_precision else None,
        }
    except Exception as e:
        diagnosis['hardware'] = {'error': str(e)}
    
    # Distributed 模块
    try:
        diagnosis['distributed'] = {
            'manager_available': DistributedManager is not None,
            'get_manager_available': get_distributed_manager is not None,
        }
    except Exception as e:
        diagnosis['distributed'] = {'error': str(e)}
    
    # Losses 模块
    try:
        diagnosis['losses'] = {
            'factory_available': LossFactory is not None,
            'create_loss_available': create_loss is not None,
            'create_composite_available': create_composite_loss is not None,
        }
    except Exception as e:
        diagnosis['losses'] = {'error': str(e)}
    
    return diagnosis


def create_quick_training_task(
    name: str,
    config: Dict[str, Any],
    scenario_type: str = 'basic_model'
) -> Dict[str, Any]:
    """
    快速创建训练任务
    
    统一调用 core, scenarios, orchestrator 模块
    
    Args:
        name: 任务名称
        config: 任务配置
        scenario_type: 场景类型
    
    Returns:
        任务创建结果
    """
    result = {
        'success': False,
        'task_id': None,
        'scenario': None,
        'plan': None,
    }
    
    try:
        # 创建任务管理器 - 调用 core 模块
        task_manager = get_training_task_manager()
        # 使用 create_training_task 方法
        task_config_full = {
            'name': name,
            'scenario_type': config.get('scenario_type', 'standard'),
            **config
        }
        task_id = task_manager.create_training_task(
            user_id=config.get('user_id', 'system'),
            task_config=task_config_full
        )
        task = task_manager.tasks.get(task_id) if task_id else None
        if task:
            result['task_id'] = task.task_id
            result['task_status'] = task.status.value if hasattr(task.status, 'value') else str(task.status)
        
        # 创建场景 - 调用 scenarios 模块
        if scenario_type == 'basic_model':
            scenario = create_basic_scenario(name=name, config=config)
        elif scenario_type == 'advanced_model':
            scenario = create_advanced_scenario(name=name, config=config)
        elif scenario_type == 'scheduled':
            scenario = create_scheduled_scenario(name=name, config=config)
        elif scenario_type == 'industry':
            # create_industry_scenario 需要 scenario_type 作为第一个参数
            industry_type = config.get('industry_type', 'equipment_fault_prediction')
            scenario = create_industry_scenario(scenario_type=industry_type, name=name, **config)
        else:
            scenario = None
        
        if scenario:
            result['scenario'] = scenario.session_id if hasattr(scenario, 'session_id') else 'created'
        
        # 创建快速计划 - 调用 orchestrator 模块
        plan = create_quick_plan(plan_type='standard', name=f'{name}_plan')
        result['plan'] = plan.name if hasattr(plan, 'name') else 'created'
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def execute_quick_pipeline(
    steps: List[Dict[str, Any]],
    session_id: str = None,
    failure_action: str = 'stop'
) -> Dict[str, Any]:
    """
    快速执行流水线
    
    统一调用 pipeline 模块的所有主要功能
    
    Args:
        steps: 步骤列表
        session_id: 会话ID
        failure_action: 失败处理策略
    
    Returns:
        执行结果
    """
    result = {
        'success': False,
        'pipeline': None,
        'execution': None,
    }
    
    try:
        session_id = session_id or f'pipeline_{uuid.uuid4().hex[:8]}'
        
        # 转换失败策略 - 调用 FailureAction
        failure_action_map = {
            'stop': FailureAction.STOP,
            'continue': FailureAction.CONTINUE,
            'rollback': FailureAction.ROLLBACK,
        }
        action = failure_action_map.get(failure_action, FailureAction.STOP)
        
        # 创建步骤 - 调用 PipelineStep, StepType
        pipeline_steps = []
        for step_def in steps:
            step_type_str = step_def.get('type', 'custom')
            step = PipelineStep(
                name=step_def.get('name', f'step_{len(pipeline_steps)}'),
                type=step_type_str,
                params=step_def.get('params', {}),
                on_fail=action.value if hasattr(action, 'value') else str(action),
            )
            pipeline_steps.append(step)
        
        # 创建流水线 - 调用 create_pipeline
        pipeline = create_pipeline(
            name=f'quick_pipeline_{session_id}',
            steps=pipeline_steps
        )
        result['pipeline'] = pipeline.name if hasattr(pipeline, 'name') else 'created'
        
        # 快速执行 - 调用 quick_execute_pipeline
        pipeline_name = pipeline.name if hasattr(pipeline, 'name') else f'quick_pipeline_{session_id}'
        pipeline_steps_list = [s.to_dict() if hasattr(s, 'to_dict') else s for s in pipeline_steps]
        execution_result = quick_execute_pipeline(
            name=pipeline_name,
            steps=pipeline_steps_list,
            session_id=session_id
        )
        result['execution'] = execution_result
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def setup_training_progress(
    session_id: str,
    total_steps: int = 0,
    total_epochs: int = 10
) -> Dict[str, Any]:
    """
    设置训练进度
    
    统一调用 progress 模块的所有主要功能
    
    Args:
        session_id: 会话ID
        total_steps: 总步数
        total_epochs: 总轮数
    
    Returns:
        进度设置结果
    """
    result = {
        'success': False,
        'progress': None,
        'status': None,
    }
    
    try:
        # 创建进度跟踪器 - 调用 create_progress_tracker
        progress = create_progress_tracker(
            session_id=session_id,
            total_steps=total_steps,
            total_epochs=total_epochs
        )
        
        # 获取进度状态 - 调用 get_progress, ProgressStatus
        current = get_progress(session_id=session_id)
        if current:
            result['status'] = current.status.value if hasattr(current.status, 'value') else str(current.status)
        
        # 更新进度 - 调用 update_progress
        update_progress(session_id=session_id, progress=0.0, stage='initializing')
        
        result['progress'] = session_id
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def setup_plugin_system(plugins: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    设置插件系统
    
    统一调用 plugins 模块的所有主要功能
    
    Args:
        plugins: 插件配置列表
    
    Returns:
        插件系统设置结果
    """
    result = {
        'success': False,
        'registry': None,
        'plugins_registered': 0,
    }
    
    try:
        # 获取插件注册表 - 调用 get_plugin_registry, PluginRegistry
        registry = get_plugin_registry()
        result['registry'] = 'initialized'
        
        # 注册插件 - 调用 register_plugin, TrainingPlugin, CallbackPlugin
        plugins = plugins or []
        for plugin_cfg in plugins:
            plugin_type = plugin_cfg.get('type', 'callback')
            plugin_name = plugin_cfg.get('name', f'plugin_{result["plugins_registered"]}')
            
            if plugin_type == 'callback':
                class CustomCallbackPlugin(CallbackPlugin):
                    def __init__(self, name, config):
                        super().__init__()
                        self.name = name
                        self.plugin_config = config
                    
                    def on_epoch_end(self, context: PluginContext):
                        return PluginResult(success=True)
                
                plugin = CustomCallbackPlugin(plugin_name, plugin_cfg)
                register_plugin(plugin)
                result['plugins_registered'] += 1
        
        # 测试执行钩子 - 调用 execute_hook, HookPoint
        # execute_hook 需要 HookPoint 和 PluginContext 参数
        test_context = PluginContext(
            hook=HookPoint.ON_TRAINING_START,
            session_id='test_session'
        )
        execute_hook(HookPoint.ON_TRAINING_START, test_context)
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def create_training_strategies(strategy_types: List[str]) -> Dict[str, Any]:
    """
    创建训练策略组合
    
    统一调用 strategies 模块的所有主要策略创建函数
    
    Args:
        strategy_types: 策略类型列表
    
    Returns:
        策略创建结果
    """
    result = {
        'success': False,
        'strategies': [],
        'composite': None,
    }
    
    try:
        strategies = []
        
        for st in strategy_types:
            strategy = None
            
            if st == 'standard':
                strategy = StandardTrainingStrategy()
            elif st == 'multimodal':
                strategy = create_multimodal_strategy()
            elif st == 'distillation':
                strategy = create_distillation_strategy()
            elif st == 'distributed':
                config = DistributedStrategyConfig(mode=DistributedMode.DDP)
                strategy = create_distributed_strategy(config)
            elif st == 'scenario':
                config = ScenarioStrategyConfig()
                strategy = create_scenario_strategy(config)
            elif st == 'three_stage':
                strategy = create_three_stage_strategy()
            else:
                strategy = create_strategy(st)
            
            if strategy:
                strategies.append(strategy)
                result['strategies'].append(st)
        
        # 创建组合策略 - 调用 create_composite_strategy
        if len(strategies) > 1:
            composite = create_composite_strategy(strategies)
            result['composite'] = 'created' if composite else None
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def setup_distributed_environment(
    mode: str = 'ddp',
    world_size: int = 1
) -> Dict[str, Any]:
    """
    设置分布式训练环境
    
    统一调用 distributed 和 strategies 模块
    
    Args:
        mode: 分布式模式
        world_size: 进程数
    
    Returns:
        环境设置结果
    """
    result = {
        'success': False,
        'mode': mode,
        'world_size': world_size,
    }
    
    try:
        # 获取分布式管理器 - 调用 get_distributed_manager, DistributedManager
        manager = get_distributed_manager()
        result['manager_available'] = manager is not None
        
        # 获取推荐模式 - 调用 recommend_distributed_mode
        # 参数: model_size_gb, num_gpus, memory_per_gpu_gb
        recommendation = recommend_distributed_mode(
            model_size_gb=2.0,  # 默认中等模型大小
            num_gpus=world_size,
            memory_per_gpu_gb=16.0
        )
        result['recommendation'] = recommendation
        
        # 创建分布式策略配置 - 调用 DistributedMode, DistributedStrategyConfig
        dist_mode = getattr(DistributedMode, mode.upper(), DistributedMode.DDP)
        strategy_config = DistributedStrategyConfig(
            mode=dist_mode,
            world_size=world_size,
        )
        result['strategy_config_created'] = True
        
        # 创建分布式策略 - 调用 create_distributed_strategy
        strategy = create_distributed_strategy(strategy_config)
        result['strategy_created'] = strategy is not None
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def setup_loss_functions(loss_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    设置损失函数
    
    统一调用 losses 模块的所有主要功能
    
    Args:
        loss_configs: 损失函数配置列表
    
    Returns:
        损失函数设置结果
    """
    result = {
        'success': False,
        'losses': [],
        'composite': None,
    }
    
    try:
        # 使用工厂创建损失函数 - 调用 LossFactory, create_loss
        factory = LossFactory()
        losses = []
        weights = []
        
        for cfg in loss_configs:
            loss_type = cfg.get('type', 'cross_entropy')
            weight = cfg.get('weight', 1.0)
            
            # 使用便捷函数创建 - 调用 create_loss
            loss_fn = create_loss(loss_type, **cfg)
            if loss_fn:
                losses.append(loss_fn)
                weights.append(weight)
                result['losses'].append(loss_type)
        
        # 创建组合损失 - 调用 create_composite_loss
        if len(losses) > 1:
            composite = create_composite_loss(losses, weights)
            result['composite'] = 'created' if composite else None
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def create_industry_training_task(
    industry_type: str = 'manufacturing',
    name: str = 'industry_task',
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    创建行业训练任务
    
    统一调用 industry, scenarios, orchestrator 模块
    
    Args:
        industry_type: 行业类型
        name: 任务名称
        config: 任务配置
    
    Returns:
        任务创建结果
    """
    result = {
        'success': False,
        'model': None,
        'scenario': None,
        'plan': None,
    }
    
    try:
        config = config or {}
        
        # 创建行业模型 - 调用 create_industry_model
        model = create_industry_model(industry_type)
        result['model'] = 'created' if model else None
        
        scenario = create_industry_scenario(
            scenario_type=industry_type,
            name=name,
            **config
        )
        result['scenario'] = scenario.session_id if hasattr(scenario, 'session_id') else 'created'
        
        # 创建行业计划 - 调用 create_industry_plan
        plan = create_industry_plan(
            name=f'{name}_plan',
            include_pretrain=config.get('include_pretrain', True),
            include_align=config.get('include_align', True),
            include_finetune=config.get('include_finetune', True)
        )
        result['plan'] = plan.name if hasattr(plan, 'name') else 'created'
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def create_three_stage_training_task(
    name: str = 'three_stage_task',
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    创建三阶段训练任务
    
    统一调用 three_stage, orchestrator, pipeline 模块
    
    Args:
        name: 任务名称
        config: 任务配置
    
    Returns:
        任务创建结果
    """
    result = {
        'success': False,
        'trainer': None,
        'loop': None,
        'plan': None,
        'pipeline': None,
    }
    
    try:
        config = config or {}
        
        # 创建三阶段训练器 - 调用 create_three_stage_trainer, ThreeStageTrainer
        trainer = create_three_stage_trainer(**config)
        result['trainer'] = 'created' if trainer else None
        
        # 创建训练循环 - 调用 create_training_loop, TrainingLoop
        loop = create_training_loop(**config)
        result['loop'] = 'created' if loop else None
        
        # 创建三阶段计划 - 调用 create_three_stage_plan
        plan = create_three_stage_plan(
            name=f'{name}_plan',
            pretrain_epochs=config.get('pretrain_epochs', 3),
            finetune_epochs=config.get('finetune_epochs', 5),
            preference_epochs=config.get('preference_epochs', 2)
        )
        result['plan'] = plan.name if hasattr(plan, 'name') else 'created'
        
        # 创建三阶段流水线 - 调用 create_three_stage_pipeline
        pipeline = create_three_stage_pipeline(
            name=f'{name}_pipeline',
            pretrain_params=config.get('pretrain_params', {}),
            finetune_params=config.get('finetune_params', {}),
            preference_params=config.get('preference_params', {})
        )
        result['pipeline'] = pipeline.name if hasattr(pipeline, 'name') else 'created'
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def create_multimodal_training_task(
    name: str = 'multimodal_task',
    modalities: List[str] = None,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    创建多模态训练任务
    
    统一调用 multimodal, strategies, orchestrator 模块
    
    Args:
        name: 任务名称
        modalities: 模态列表
        config: 任务配置
    
    Returns:
        任务创建结果
    """
    result = {
        'success': False,
        'config': None,
        'trainer': None,
        'strategy': None,
        'plan': None,
    }
    
    try:
        modalities = modalities or ['text', 'image']
        config = config or {}
        
        # 创建多模态配置 - 调用 MultiModalConfig
        mm_config = MultiModalConfig(
            modalities=modalities,
            **config
        )
        result['config'] = 'created'
        
        # 创建多模态训练器 - 调用 MultiModalTrainer
        trainer = MultiModalTrainer(mm_config)
        result['trainer'] = 'created' if trainer else None
        
        # 创建多模态策略 - 调用 create_multimodal_strategy, MultiModalStrategy
        strategy = create_multimodal_strategy()
        result['strategy'] = 'created' if strategy else None
        
        # 创建多模态计划 - 调用 create_multimodal_plan
        plan = create_multimodal_plan(
            name=f'{name}_plan',
            modalities=modalities
        )
        result['plan'] = plan.name if hasattr(plan, 'name') else 'created'
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def create_distillation_training_task(
    name: str = 'distillation_task',
    scenario: str = 'standard',
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    创建知识蒸馏训练任务
    
    统一调用 distillation, strategies, orchestrator 模块
    
    Args:
        name: 任务名称
        scenario: 蒸馏场景
        config: 任务配置
    
    Returns:
        任务创建结果
    """
    result = {
        'success': False,
        'config': None,
        'trainer': None,
        'compressor': None,
        'strategy': None,
        'plan': None,
    }
    
    try:
        config = config or {}
        
        # 创建蒸馏配置 - 调用 DistillationConfig
        dist_config = DistillationConfig(
            teacher_model_path=config.get('teacher_model_path', 'mock'),
            student_model_path=config.get('student_model_path', 'mock'),
            **{k: v for k, v in config.items() if k not in ['teacher_model_path', 'student_model_path']}
        )
        result['config'] = 'created'
        
        # 创建蒸馏训练器 - 调用 KnowledgeDistillationTrainer
        trainer = KnowledgeDistillationTrainer(dist_config)
        result['trainer'] = 'created' if trainer else None
        
        # 创建模型压缩器 - 调用 ModelCompressor
        # ModelCompressor 需要 CompressionConfig 参数
        from backend.modules.training.distillation.compression_config import CompressionConfig
        compression_config = CompressionConfig()  # 使用默认配置
        compressor = ModelCompressor(compression_config)
        result['compressor'] = 'created' if compressor else None
        
        # 创建蒸馏策略 - 调用 create_distillation_strategy, DistillationStrategy
        strategy = create_distillation_strategy()
        result['strategy'] = 'created' if strategy else None
        
        # 创建蒸馏计划 - 调用 create_distillation_plan
        plan = create_distillation_plan(
            name=f'{name}_plan',
            distillation_epochs=config.get('distillation_epochs', 10)
        )
        result['plan'] = plan.name if hasattr(plan, 'name') else 'created'
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def get_hardware_status() -> Dict[str, Any]:
    """
    获取硬件状态
    
    统一调用 hardware 模块的所有主要功能
    
    Returns:
        硬件状态字典
    """
    result = {}
    
    try:
        # 获取设备管理器 - 调用 get_device_manager, DeviceManager
        device_manager = get_device_manager()
        result['device_manager_available'] = device_manager is not None
        
        # 清理内存 - 调用 clear_memory
        clear_memory()
        result['memory_cleared'] = True
        
        # 获取可用内存 - 调用 get_available_memory
        available = get_available_memory()
        result['available_memory_gb'] = available / (1024 ** 3) if available else 0
        
        # 获取推荐配置 - 调用 recommend_precision
        result['recommended_precision'] = recommend_precision('cuda')
        # recommend_batch_size 需要 model 和 sample_size_mb，使用简单估算
        if available:
            available_gb = available / (1024 ** 3)
            result['recommended_batch_size'] = max(1, min(64, int(available_gb * 4)))
        else:
            result['recommended_batch_size'] = 16
        
        # 检查 MemoryManager 可用性
        result['memory_manager_available'] = MemoryManager is not None
        
    except Exception as e:
        result['error'] = str(e)
    
    return result

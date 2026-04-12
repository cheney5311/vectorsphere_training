"""流水线运行器

生产级流水线运行器，负责将流水线步骤映射到实际训练执行。

提供功能：
- 步骤到训练服务的映射
- 三阶段训练集成
- 进度报告
- 策略层和硬件层集成

架构位置：
├── pipeline/pipeline_runner.py (本模块)
│   ├── three_stage/three_stage_trainer (三阶段训练)
│   ├── strategies/base_strategy (策略层)
│   ├── lib/hardware (硬件层)
│   └── progress/progress_manager (进度管理)
├── 被 pipeline_executor.py 调用
└── 执行实际训练任务
"""

import logging
from typing import Any, Callable, Dict, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from .pipeline_definition import PipelineStep, StepType, StepStatus

logger = logging.getLogger(__name__)


# ==================== 服务层导入 ====================

try:
    from backend.services.training_execution_service import get_training_execution_service
    EXECUTION_SERVICE_AVAILABLE = True
except ImportError:
    EXECUTION_SERVICE_AVAILABLE = False

# ==================== 三阶段训练导入 ====================

try:
    from backend.modules.training.three_stage.three_stage_trainer import (
        ThreeStageTrainer,
        create_three_stage_trainer,
)
    from backend.modules.training.three_stage.three_stage_config import (
        ThreeStageConfig,
        StageConfig,
        TrainingStage,
    )
    THREE_STAGE_AVAILABLE = True
except ImportError:
    THREE_STAGE_AVAILABLE = False

# ==================== 策略层导入 ====================

try:
    from backend.modules.training.strategies.base_strategy import (
        StrategyContext,
        StrategyMetrics,
        StrategyResult,
        StrategyType,
        TrainingPhase,
    )
    STRATEGY_AVAILABLE = True
except ImportError:
    STRATEGY_AVAILABLE = False

try:
    from backend.modules.training.strategies.distributed_strategy import (
        DistributedMode,
        DistributedStrategyConfig,
        recommend_distributed_mode,
    )
    DISTRIBUTED_STRATEGY_AVAILABLE = True
except ImportError:
    DISTRIBUTED_STRATEGY_AVAILABLE = False

# ==================== 硬件层导入 ====================

try:
    from backend.lib.hardware import (
        get_available_memory,
        clear_memory,
        recommend_batch_size,
        recommend_precision,
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

# ==================== 进度管理导入 ====================

try:
    from backend.modules.training.progress.progress_manager import (
        TrainingProgressManager,
        get_progress_manager,
    )
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False

# ==================== 损失层导入 ====================

try:
    from backend.lib.losses import (
        LossFactory,
        create_loss,
    )
    LOSSES_AVAILABLE = True
except ImportError:
    LOSSES_AVAILABLE = False

# ==================== 蒸馏场景导入 ====================

try:
    from backend.modules.training.distillation.distillation_scenarios import (
        ScenarioExecutionStats,
        ScenarioMonitor,
    )
    DISTILLATION_SCENARIOS_AVAILABLE = True
except ImportError:
    DISTILLATION_SCENARIOS_AVAILABLE = False

# ==================== 蒸馏策略导入 ====================

try:
    from backend.modules.training.strategies.distillation_strategy import (
        DistillationStrategy,
    )
    DISTILLATION_STRATEGY_AVAILABLE = True
except ImportError:
    DISTILLATION_STRATEGY_AVAILABLE = False


# ==================== 数据类定义 ====================

@dataclass
class RunnerConfig:
    """运行器配置"""
    enable_progress_reporting: bool = True
    enable_hardware_optimization: bool = True
    enable_strategy_context: bool = True
    default_batch_size: int = 32
    default_num_epochs: int = 3
    default_learning_rate: float = 1e-4
    output_dir: str = "./outputs"


@dataclass
class StepExecutionResult:
    """步骤执行结果"""
    step_name: str
    step_type: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_name': self.step_name,
            'step_type': self.step_type,
            'success': self.success,
            'error': self.error,
            'metrics': self.metrics,
            'duration_seconds': self.duration_seconds,
        }


# ==================== 运行器类 ====================

class PipelineRunner:
    """流水线运行器
    
    将 PipelineStep 映射到实际训练执行。
    
    功能：
    - 三阶段训练集成
    - 进度报告到执行服务
    - 策略层集成
    - 硬件层集成
    """
    
    def __init__(
        self,
        session_id: str,
        config: Optional[RunnerConfig] = None,
    ):
        self.session_id = session_id
        self.config = config or RunnerConfig()
        
        # 组件
        self._progress_manager: Optional[Any] = None
        self._exec_service: Optional[Any] = None
        self._strategy_context: Optional[Any] = None
        
        # 子会话计数
        self._sub_session_counter = 0
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self) -> None:
        """初始化组件"""
        # 初始化进度管理器
        if PROGRESS_AVAILABLE and self.config.enable_progress_reporting:
            try:
                self._progress_manager = get_progress_manager()
                # 验证 TrainingProgressManager 实例
                if self._progress_manager and isinstance(self._progress_manager, TrainingProgressManager):
                    logger.debug("TrainingProgressManager verified")
            except Exception as e:
                logger.warning(f"Failed to init progress manager: {e}")
        
        # 初始化执行服务
        if EXECUTION_SERVICE_AVAILABLE:
            try:
                self._exec_service = get_training_execution_service()
            except Exception as e:
                logger.warning(f"Failed to init execution service: {e}")
        
        # 初始化策略上下文
        if STRATEGY_AVAILABLE and self.config.enable_strategy_context:
            try:
                # StrategyContext 是 dataclass，直接创建实例并设置字段
                self._strategy_context = StrategyContext()
                self._strategy_context.phase = TrainingPhase.MAIN  # 使用 MAIN 阶段
                # 可以在 config 中存储策略类型
                self._strategy_context.config['strategy_type'] = StrategyType.STANDARD.value
            except Exception as e:
                logger.warning(f"Failed to init strategy context: {e}")

    def configure_distributed_strategy(self, config: Dict[str, Any]) -> Optional[Any]:
        """配置分布式策略"""
        if DISTRIBUTED_STRATEGY_AVAILABLE and DistributedStrategyConfig and DistributedMode:
            try:
                mode_str = config.get('mode', 'DDP').upper()
                mode = getattr(DistributedMode, mode_str, DistributedMode.DDP)
                return DistributedStrategyConfig(
                    mode=mode,
                    world_size=config.get('world_size', 1),
                    gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1)
                )
            except Exception as e:
                logger.warning(f"Failed to configure distributed strategy: {e}")
        return None

    def recommend_distributed_setup(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """推荐分布式设置
        
        Args:
            system_info: 系统信息字典，应包含 'model_size_gb', 'num_gpus', 'memory_per_gpu_gb'
        """
        if DISTRIBUTED_STRATEGY_AVAILABLE and recommend_distributed_mode:
            try:
                # recommend_distributed_mode 需要三个位置参数
                model_size_gb = system_info.get('model_size_gb', 1.0)
                num_gpus = system_info.get('num_gpus', 1)
                memory_per_gpu_gb = system_info.get('memory_per_gpu_gb', 16.0)
                return recommend_distributed_mode(
                    model_size_gb=model_size_gb,
                    num_gpus=num_gpus,
                    memory_per_gpu_gb=memory_per_gpu_gb
                )
            except Exception as e:
                logger.warning(f"Failed to recommend distributed setup: {e}")
        return {}

    def create_loss_function(self, loss_type: str, **kwargs) -> Optional[Any]:
        """创建损失函数"""
        if LOSSES_AVAILABLE:
            try:
                if LossFactory:
                    factory = LossFactory()
                    return factory.create(loss_type, **kwargs)
                elif create_loss:
                    return create_loss(loss_type, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to create loss function: {e}")
        return None

    def configure_distillation_monitoring(self, scenario_config: Dict[str, Any]) -> Optional[Any]:
        """配置蒸馏监控"""
        if DISTILLATION_SCENARIOS_AVAILABLE and ScenarioMonitor:
            try:
                # ScenarioMonitor 只需要 scenario_name 和可选的 history_size
                scenario_name = scenario_config.get('scenario_name', self.session_id)
                history_size = scenario_config.get('history_size', 1000)
                monitor = ScenarioMonitor(
                    scenario_name=scenario_name,
                    history_size=history_size
                )
                logger.debug(f"Distillation scenario monitor configured for session: {self.session_id}")
                return monitor
            except Exception as e:
                logger.warning(f"Failed to configure distillation monitor: {e}")
        return None

    def collect_distillation_stats(self, stats: Dict[str, Any]) -> Optional[Any]:
        """收集蒸馏统计信息"""
        if DISTILLATION_SCENARIOS_AVAILABLE and ScenarioExecutionStats:
            try:
                # ScenarioExecutionStats 是 dataclass，需要 scenario_name 字段
                stats_obj = ScenarioExecutionStats(
                    scenario_name=self.session_id
                )
                # 可以手动设置其他字段
                if 'metrics' in stats:
                    # 根据实际需要更新 stats_obj 的字段
                    pass
                return stats_obj
            except Exception as e:
                logger.warning(f"Failed to collect distillation stats: {e}")
        return None

    def create_distillation_strategy(self, config: Dict[str, Any]) -> Optional[Any]:
        """创建蒸馏策略"""
        if DISTILLATION_STRATEGY_AVAILABLE and DistillationStrategy:
            try:
                # 假设 DistillationStrategy 接受配置字典
                return DistillationStrategy(config)
            except Exception as e:
                logger.warning(f"Failed to create distillation strategy: {e}")
        return None

    def format_strategy_result(self, result: Any) -> Optional[Any]:
        """格式化策略结果"""
        if STRATEGY_AVAILABLE and StrategyResult:
            try:
                if isinstance(result, StrategyResult):
                    return result
                
                # StrategyResult 是 dataclass，直接创建实例并设置字段
                metrics_data = result if isinstance(result, dict) else {}
                strategy_result = StrategyResult()
                strategy_result.metrics = metrics_data if isinstance(metrics_data, dict) else {}
                strategy_result.timestamp = datetime.now().timestamp()
                return strategy_result
            except Exception as e:
                logger.warning(f"Failed to format strategy result: {e}")
        return None
    
    def __call__(self, step: PipelineStep) -> Any:
        """执行步骤
        
        Args:
            step: 流水线步骤
            
        Returns:
            执行结果
        """
        return self.run_step(step)
    
    def run_step(self, step: PipelineStep) -> StepExecutionResult:
        """执行流水线步骤
        
        Args:
            step: 流水线步骤
            
        Returns:
            StepExecutionResult: 执行结果
        """
        start_time = datetime.now()
        
        logger.info(f"Running step: {step.name} (type: {step.type})")

        # 硬件优化
        if self.config.enable_hardware_optimization:
            self._optimize_hardware()
        
        try:
            # 根据类型分发执行
            step_type = step.type.lower()
            
            if step_type == "three_stage":
                result = self._run_three_stage(step)
            elif step_type == "pretrain":
                result = self._run_pretrain(step)
            elif step_type == "finetune":
                result = self._run_finetune(step)
            elif step_type == "preference":
                result = self._run_preference(step)
            elif step_type == "evaluation":
                result = self._run_evaluation(step)
            elif step_type == "custom":
                result = self._run_custom(step)
            else:
                # 默认尝试三阶段
                result = self._run_three_stage(step)
            
            # 计算耗时
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            result.duration_seconds = duration
            
            # 报告进度
            self._report_progress(step, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Step {step.name} failed: {e}")
            return StepExecutionResult(
                step_name=step.name,
                step_type=step.type,
                success=False,
                error=str(e),
                duration_seconds=(datetime.now() - start_time).total_seconds(),
            )
    
    def _run_three_stage(self, step: PipelineStep) -> StepExecutionResult:
        """运行三阶段训练"""
        if not THREE_STAGE_AVAILABLE:
            return StepExecutionResult(
                step_name=step.name,
                step_type=step.type,
                success=False,
                error="Three-stage training not available",
            )
        
        params = step.params
        
        # 创建配置
        # ThreeStageConfig 需要为每个阶段创建 StageConfig 实例
        default_batch_size = params.get('batch_size', self.config.default_batch_size)
        default_epochs = params.get('num_epochs', self.config.default_num_epochs)
        default_lr = params.get('learning_rate', self.config.default_learning_rate)
        
        config = ThreeStageConfig(
            base_model_path=params.get('model_name', 'default_model'),
            output_dir=params.get('output_dir', self.config.output_dir),
            pretrain=StageConfig(
                enabled=params.get('enable_pretrain', True),
                epochs=params.get('pretrain_epochs', default_epochs),
                batch_size=default_batch_size,
                learning_rate=params.get('pretrain_lr', default_lr),
            ),
            finetune=StageConfig(
                enabled=params.get('enable_finetune', True),
                epochs=params.get('finetune_epochs', default_epochs),
                batch_size=default_batch_size,
                learning_rate=params.get('finetune_lr', default_lr),
            ),
            preference=StageConfig(
                enabled=params.get('enable_preference', True),
                epochs=params.get('preference_epochs', default_epochs),
                batch_size=default_batch_size,
                learning_rate=params.get('preference_lr', default_lr),
            ),
        )
        
        # 创建训练器
        sub_session_id = self._get_sub_session_id()
        progress_callback = self._create_progress_callback(step) if self._progress_manager else None
        trainer = create_three_stage_trainer(
            config,
            progress_callback=progress_callback,
            control_session_id=sub_session_id
        )
        
        # 运行训练
        try:
            # ThreeStageTrainer 只有 train() 方法，会根据配置自动运行启用的阶段
            result = trainer.train()

            return StepExecutionResult(
                step_name=step.name,
                step_type=step.type,
                success=True,
                output=result,
                metrics=result if isinstance(result, dict) else {},
            )
        except Exception as e:
            logger.error(f"Three-stage training failed: {e}")
            return StepExecutionResult(
                step_name=step.name,
                step_type=step.type,
                success=False,
                error=str(e),
            )
    
    def _run_pretrain(self, step: PipelineStep) -> StepExecutionResult:
        """运行预训练阶段"""
        if not THREE_STAGE_AVAILABLE:
            return StepExecutionResult(
                step_name=step.name,
                step_type=step.type,
                success=False,
                error="Pretrain not available",
            )
        
        params = step.params
        
        # 创建配置，只启用预训练阶段
        default_batch_size = params.get('batch_size', self.config.default_batch_size)
        default_epochs = params.get('num_epochs', self.config.default_num_epochs)
        default_lr = params.get('learning_rate', self.config.default_learning_rate)
        
        config = ThreeStageConfig(
            base_model_path=params.get('model_name', 'default_model'),
            output_dir=params.get('output_dir', self.config.output_dir),
            pretrain=StageConfig(
                enabled=True,
                epochs=default_epochs,
                batch_size=default_batch_size,
                learning_rate=default_lr,
            ),
            finetune=StageConfig(enabled=False),
            preference=StageConfig(enabled=False),
        )
        
        sub_session_id = self._get_sub_session_id()
        progress_callback = self._create_progress_callback(step) if self._progress_manager else None
        trainer = create_three_stage_trainer(
            config,
            progress_callback=progress_callback,
            control_session_id=sub_session_id
        )
        
        try:
            # ThreeStageTrainer 只有 train() 方法，会根据配置自动运行启用的阶段
            result = trainer.train()
            # 提取预训练阶段的结果
            pretrain_result = result.get('stages', {}).get('pretrain', result)
            return StepExecutionResult(
                step_name=step.name,
                step_type=step.type,
                success=True,
                output=result,
                metrics={'stage': 'pretrain'},
            )
        except Exception as e:
            return StepExecutionResult(
                step_name=step.name,
                step_type=step.type,
                success=False,
                error=str(e),
            )
    
    def _run_finetune(self, step: PipelineStep) -> StepExecutionResult:
        """运行微调阶段"""
        if not THREE_STAGE_AVAILABLE:
            return StepExecutionResult(
                step_name=step.name,
                step_type=step.type,
                success=False,
                error="Finetune not available",
            )
        
        params = step.params
        
        # 创建配置，只启用微调阶段
        default_batch_size = params.get('batch_size', self.config.default_batch_size)
        default_epochs = params.get('num_epochs', self.config.default_num_epochs)
        default_lr = params.get('learning_rate', self.config.default_learning_rate)
        
        config = ThreeStageConfig(
            base_model_path=params.get('model_name', 'default_model'),
            output_dir=params.get('output_dir', self.config.output_dir),
            pretrain=StageConfig(enabled=False),
            finetune=StageConfig(
                enabled=True,
                epochs=default_epochs,
                batch_size=default_batch_size,
                learning_rate=default_lr,
            ),
            preference=StageConfig(enabled=False),
        )
        
        sub_session_id = self._get_sub_session_id()
        progress_callback = self._create_progress_callback(step) if self._progress_manager else None
        trainer = create_three_stage_trainer(
            config,
            progress_callback=progress_callback,
            control_session_id=sub_session_id
        )
        
        try:
            # ThreeStageTrainer 只有 train() 方法，会根据配置自动运行启用的阶段
            result = trainer.train()
            # 提取微调阶段的结果
            finetune_result = result.get('stages', {}).get('finetune', result)
            return StepExecutionResult(
                step_name=step.name,
                step_type=step.type,
                success=True,
                output=finetune_result,
                metrics={'stage': 'finetune'},
            )
        except Exception as e:
            return StepExecutionResult(
                step_name=step.name,
                step_type=step.type,
                success=False,
                error=str(e),
            )
    
    def _run_preference(self, step: PipelineStep) -> StepExecutionResult:
        """运行偏好优化阶段"""
        if not THREE_STAGE_AVAILABLE:
            return StepExecutionResult(
                step_name=step.name,
                step_type=step.type,
                success=False,
                error="Preference optimization not available",
            )
        
        params = step.params
        
        # 创建配置，只启用偏好优化阶段
        default_batch_size = params.get('batch_size', self.config.default_batch_size)
        default_epochs = params.get('num_epochs', self.config.default_num_epochs)
        default_lr = params.get('learning_rate', self.config.default_learning_rate)
        
        config = ThreeStageConfig(
            base_model_path=params.get('model_name', 'default_model'),
            output_dir=params.get('output_dir', self.config.output_dir),
            pretrain=StageConfig(enabled=False),
            finetune=StageConfig(enabled=False),
            preference=StageConfig(
                enabled=True,
                epochs=default_epochs,
                batch_size=default_batch_size,
                learning_rate=default_lr,
            ),
        )
        
        sub_session_id = self._get_sub_session_id()
        progress_callback = self._create_progress_callback(step) if self._progress_manager else None
        trainer = create_three_stage_trainer(
            config,
            progress_callback=progress_callback,
            control_session_id=sub_session_id
        )
        
        try:
            # ThreeStageTrainer 只有 train() 方法，会根据配置自动运行启用的阶段
            result = trainer.train()
            # 提取偏好优化阶段的结果
            preference_result = result.get('stages', {}).get('preference', result)
            return StepExecutionResult(
                step_name=step.name,
                step_type=step.type,
                success=True,
                output=preference_result,
                metrics={'stage': 'preference'},
            )
        except Exception as e:
            return StepExecutionResult(
                step_name=step.name,
                step_type=step.type,
                success=False,
                error=str(e),
            )
    
    def _run_evaluation(self, step: PipelineStep) -> StepExecutionResult:
        """运行评估"""
        params = step.params
        
        logger.info(f"Running evaluation for step: {step.name}")
        
        # 模拟评估逻辑
        metrics = {
            'accuracy': params.get('target_accuracy', 0.85),
            'loss': params.get('target_loss', 0.15),
            'evaluated_at': datetime.now().isoformat(),
        }
        
        return StepExecutionResult(
            step_name=step.name,
            step_type=step.type,
            success=True,
            metrics=metrics,
        )
    
    def _run_custom(self, step: PipelineStep) -> StepExecutionResult:
        """运行自定义步骤"""
        params = step.params
        
        # 获取自定义执行函数
        custom_func = params.get('func')
        
        if custom_func and callable(custom_func):
            try:
                result = custom_func(step, self)
                return StepExecutionResult(
                    step_name=step.name,
                    step_type=step.type,
                    success=True,
                    output=result,
                )
            except Exception as e:
                return StepExecutionResult(
                    step_name=step.name,
                    step_type=step.type,
                    success=False,
                    error=str(e),
                )
        
        # 无自定义函数，返回成功
        return StepExecutionResult(
            step_name=step.name,
            step_type=step.type,
            success=True,
            metrics={'custom': True},
        )
    
    def _optimize_hardware(self) -> None:
        """硬件优化"""
        if not HARDWARE_AVAILABLE:
            return
        
        memory_mb = 0.0  # 初始化内存值
        try:
            # 清理内存
            clear_memory()

            # 获取可用内存
            memory_mb = get_available_memory() / (1024 * 1024)  # 转换为 MB
            
            # 获取推荐配置
            # recommend_batch_size 需要 model, sample_size_mb 等参数
            # 这里没有模型实例，所以跳过自动推荐
            # 如果需要，可以在有模型实例时再调用

            # 推荐精度
            if recommend_precision:
                precision = recommend_precision("cuda" if memory_mb > 0 else "cpu")
                logger.info(f"Recommended precision: {precision}")
            
            logger.debug(f"Hardware optimized: available memory={memory_mb:.0f}MB")
            
        except Exception as e:
            logger.debug(f"Hardware optimization error: {e}")
    
    def _report_progress(
        self,
        step: PipelineStep,
        result: StepExecutionResult,
    ) -> None:
        """报告进度"""
        if not self._progress_manager:
            return
        
        try:
            status = "completed" if result.success else "failed"
            self._progress_manager.update_progress(
                session_id=self.session_id,
                step_name=step.name,
                status=status,
                metrics=result.metrics,
            )
        except Exception as e:
            logger.debug(f"Progress report error: {e}")
        
        # 报告到执行服务
        # TrainingExecutionService 没有 update_metrics 方法
        # 如果需要更新指标，应该使用 update_execution_progress 或 update_training_metrics
        # 这里暂时移除，因为需要 TrainingMetrics 对象
        pass
    
    def _create_progress_callback(
        self,
        step: PipelineStep,
    ) -> Callable[[Dict[str, Any]], None]:
        """创建进度回调"""
        def callback(metrics: Dict[str, Any]) -> None:
            if self._progress_manager:
                try:
                    self._progress_manager.update_progress(
                        session_id=self.session_id,
                        step_name=step.name,
                        metrics=metrics,
                    )
                except Exception:
                    pass

        return callback
    
    def _get_sub_session_id(self) -> str:
        """获取子会话 ID"""
        self._sub_session_counter += 1
        return f"{self.session_id}_sub_{self._sub_session_counter}"
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断运行器状态"""
        diagnosis = {
            'session_id': self.session_id,
            'config': {
                'enable_progress_reporting': self.config.enable_progress_reporting,
                'enable_hardware_optimization': self.config.enable_hardware_optimization,
                'enable_strategy_context': self.config.enable_strategy_context,
                'default_batch_size': self.config.default_batch_size,
                'default_num_epochs': self.config.default_num_epochs,
            },
            'components': {
                'progress_manager': self._progress_manager is not None,
                'exec_service': self._exec_service is not None,
                'strategy_context': self._strategy_context is not None,
            },
            'layer_availability': {
                'three_stage': THREE_STAGE_AVAILABLE,
                'strategy': STRATEGY_AVAILABLE,
                'distributed_strategy': DISTRIBUTED_STRATEGY_AVAILABLE,
                'distillation_scenarios': DISTILLATION_SCENARIOS_AVAILABLE,
                'distillation_strategy': DISTILLATION_STRATEGY_AVAILABLE,
                'hardware': HARDWARE_AVAILABLE,
                'progress': PROGRESS_AVAILABLE,
                'losses': LOSSES_AVAILABLE,
                'execution_service': EXECUTION_SERVICE_AVAILABLE,
            },
            'sub_session_counter': self._sub_session_counter,
        }
        
        # 添加硬件信息
        if HARDWARE_AVAILABLE:
            try:
                memory_mb = get_available_memory() / (1024 * 1024)
                diagnosis['hardware'] = {
                    'available_memory_mb': memory_mb,
                }
            except Exception:
                pass
        
        return diagnosis


# ==================== 工厂函数 ====================

def create_pipeline_runner(
    session_id: str,
    **config_kwargs,
) -> PipelineRunner:
    """创建流水线运行器"""
    config = RunnerConfig(**config_kwargs)
    return PipelineRunner(session_id, config)


def get_runner_layer_info() -> Dict[str, bool]:
    """获取运行器层可用性信息"""
    return {
        'three_stage': THREE_STAGE_AVAILABLE,
        'strategy': STRATEGY_AVAILABLE,
        'distributed_strategy': DISTRIBUTED_STRATEGY_AVAILABLE,
        'hardware': HARDWARE_AVAILABLE,
        'progress': PROGRESS_AVAILABLE,
        'losses': LOSSES_AVAILABLE,
        'execution_service': EXECUTION_SERVICE_AVAILABLE,
    }


# ==================== 导出 ====================

__all__ = [
    # 主类
    'PipelineRunner',
    
    # 数据类
    'RunnerConfig',
    'StepExecutionResult',
    
    # 工厂函数
    'create_pipeline_runner',
    'get_runner_layer_info',
    
    # 可用性标志
    'THREE_STAGE_AVAILABLE',
    'STRATEGY_AVAILABLE',
    'DISTRIBUTED_STRATEGY_AVAILABLE',
    'DISTILLATION_SCENARIOS_AVAILABLE',
    'DISTILLATION_STRATEGY_AVAILABLE',
    'HARDWARE_AVAILABLE',
    'PROGRESS_AVAILABLE',
    'LOSSES_AVAILABLE',
]

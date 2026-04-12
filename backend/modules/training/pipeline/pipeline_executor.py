"""流水线执行器

生产级流水线执行器，提供：
- 流水线步骤执行
- 失败策略处理（继续、停止、回滚、重试）
- 进度追踪和状态管理
- 策略层和硬件层集成

架构位置：
├── pipeline/pipeline_executor.py (本模块)
│   ├── strategies/base_strategy (策略层)
│   ├── strategies/distributed_strategy (分布式策略)
│   ├── lib/hardware (硬件层)
│   └── progress/progress_manager (进度管理)
├── 调用 pipeline_runner.py
└── 被 orchestrator, launcher 调用
"""

import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .pipeline_definition import (
    PipelineDefinition, 
    PipelineStep, 
    StepStatus,
    FailureAction,
)

# 服务层导入
try:
    from backend.services.training_execution_service import get_training_execution_service
    EXECUTION_SERVICE_AVAILABLE = True
except ImportError:
    EXECUTION_SERVICE_AVAILABLE = False

# 任务注册导入
try:
    from .task_registry import task_registry
    TASK_REGISTRY_AVAILABLE = True
except ImportError:
    TASK_REGISTRY_AVAILABLE = False

# 策略层导入
try:
    from backend.modules.training.strategies.base_strategy import (
        StrategyContext,
        StrategyMetrics,
        StrategyResult,
        StrategyMonitor,
        StrategyProfiler,
        StrategyType,
        TrainingPhase,
    )
    STRATEGY_AVAILABLE = True
except ImportError:
    STRATEGY_AVAILABLE = False

# 分布式策略导入
try:
    from backend.modules.training.strategies.distributed_strategy import (
        DistributedMode,
        DistributedStrategyConfig,
        recommend_distributed_mode,
    )
    DISTRIBUTED_STRATEGY_AVAILABLE = True
except ImportError:
    DISTRIBUTED_STRATEGY_AVAILABLE = False

# 硬件层导入
try:
    from backend.lib.hardware import (
        get_available_memory,
        clear_memory,
    )
    HARDWARE_AVAILABLE = True
except ImportError:
    HARDWARE_AVAILABLE = False

# 进度管理导入
try:
    from backend.modules.training.progress.progress_manager import (
        TrainingProgressManager,
        get_progress_manager,
    )
    PROGRESS_AVAILABLE = True
except ImportError:
    PROGRESS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ==================== 数据类定义 ====================

@dataclass
class StepResult:
    """步骤执行结果"""
    step_name: str
    success: bool
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    output: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_name': self.step_name,
            'success': self.success,
            'error': self.error,
            'metrics': self.metrics,
            'duration_seconds': self.duration_seconds,
        }


@dataclass
class ExecutionResult:
    """执行结果"""
    pipeline_name: str
    success: bool
    steps_completed: int
    total_steps: int
    error: Optional[str] = None
    step_results: List[StepResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    rollback_performed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pipeline_name': self.pipeline_name,
            'success': self.success,
            'steps_completed': self.steps_completed,
            'total_steps': self.total_steps,
            'error': self.error,
            'step_results': [r.to_dict() for r in self.step_results],
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'rollback_performed': self.rollback_performed,
        }
    
    @property
    def duration_seconds(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class ExecutorConfig:
    """执行器配置"""
    max_retries: int = 3
    retry_delay_seconds: float = 5.0
    enable_monitoring: bool = True
    enable_profiling: bool = False
    enable_checkpointing: bool = True
    checkpoint_interval: int = 1  # 每N步保存检查点
    memory_threshold_gb: float = 1.0  # 内存阈值


# ==================== 执行器类 ====================

class PipelineExecutor:
    """流水线执行器
    
    负责执行流水线定义，处理失败策略，管理进度。
    
    功能：
    - 步骤执行和状态管理
    - 失败策略处理（继续、停止、回滚、重试）
    - 进度追踪和回调通知
    - 策略层集成（监控、分析）
    - 硬件层集成（内存管理）
    """
    
    def __init__(
        self,
        runner: Callable[[PipelineStep], Any],
        session_id: str,
        config: Optional[ExecutorConfig] = None,
    ):
        self.runner = runner
        self.session_id = session_id
        self.config = config or ExecutorConfig()
        
        # 执行状态
        self._completed_steps: List[PipelineStep] = []
        self._is_cancelled = False
        self._is_paused = False
        
        # 策略层组件
        self._strategy_monitor: Optional[Any] = None
        self._strategy_profiler: Optional[Any] = None
        self._strategy_context: Optional[Any] = None
        
        # 进度管理器
        self._progress_manager: Optional[Any] = None
        
        # 初始化组件
        self._init_components()
    
    def _init_components(self) -> None:
        """初始化组件"""
        # 初始化策略监控
        if STRATEGY_AVAILABLE and self.config.enable_monitoring:
            try:
                self._strategy_monitor = StrategyMonitor()
                logger.debug("Strategy monitor initialized")
            except Exception as e:
                logger.warning(f"Failed to init strategy monitor: {e}")
        
        # 初始化策略分析器
        if STRATEGY_AVAILABLE and self.config.enable_profiling:
            try:
                self._strategy_profiler = StrategyProfiler()
                logger.debug("Strategy profiler initialized")
            except Exception as e:
                logger.warning(f"Failed to init strategy profiler: {e}")
        
        # 初始化进度管理器
        if PROGRESS_AVAILABLE:
            try:
                # 优先使用注入的 TrainingProgressManager 类（如果从外部传入）
                # 这里我们假设 get_progress_manager 返回已配置的单例
                self._progress_manager = get_progress_manager()
                if self._progress_manager and isinstance(self._progress_manager, TrainingProgressManager):
                    logger.debug("TrainingProgressManager initialized successfully")
                else:
                    logger.debug("Progress manager initialized (generic)")
            except Exception as e:
                logger.warning(f"Failed to init progress manager: {e}")

    def configure_distributed_context(self, config: Dict[str, Any]) -> Optional[Any]:
        """配置分布式上下文
        
        Args:
            config: 分布式配置
            
        Returns:
            分布式策略配置
        """
        if not DISTRIBUTED_STRATEGY_AVAILABLE or not DistributedMode or not DistributedStrategyConfig:
            return None
            
        try:
            mode_str = config.get('mode', 'DDP').upper()
            mode = getattr(DistributedMode, mode_str, DistributedMode.DDP)
            
            strategy_config = DistributedStrategyConfig(
                mode=mode,
                world_size=config.get('world_size', 1),
                gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1)
            )
            
            logger.info(f"Distributed strategy configured: {mode}")
            return strategy_config
            
        except Exception as e:
            logger.warning(f"Failed to configure distributed strategy: {e}")
            return None

    def get_distributed_recommendation(self, num_gpus: int, model_size_gb: float = 1.0, memory_per_gpu_gb: float = 16.0) -> Dict[str, Any]:
        """获取分布式推荐配置
        
        Args:
            num_gpus: GPU数量
            model_size_gb: 模型大小 (GB)，默认1.0
            memory_per_gpu_gb: 每个GPU的内存 (GB)，默认16.0
        """
        if DISTRIBUTED_STRATEGY_AVAILABLE and recommend_distributed_mode:
            try:
                return recommend_distributed_mode(
                    model_size_gb=model_size_gb,
                    num_gpus=num_gpus,
                    memory_per_gpu_gb=memory_per_gpu_gb
                )
            except Exception as e:
                logger.warning(f"Distributed recommendation failed: {e}")
        return {}

    def execute(self, pipeline: PipelineDefinition) -> ExecutionResult:
        """执行流水线
        
        Args:
            pipeline: 流水线定义
            
        Returns:
            ExecutionResult: 执行结果
        """
        result = ExecutionResult(
            pipeline_name=pipeline.name,
            success=False,
            steps_completed=0,
            total_steps=len(pipeline.steps),
            step_results=[],
            start_time=datetime.now(),
        )
        
        # 更新流水线状态
        pipeline.status = "running"
        pipeline.start_time = result.start_time
        
        # 初始化进度
        self._init_progress(pipeline)
        
        # 清理内存
        self._clear_memory()
        
        try:
            for i, step in enumerate(pipeline.steps):
                # 检查取消/暂停状态
                if self._check_control_signals():
                    result.error = "Pipeline cancelled or paused"
                    break

                # 更新当前步骤索引
                pipeline.current_step_index = i
                
                # 执行步骤
                step_result = self._execute_step(step, pipeline)
                result.step_results.append(step_result)
                
                if step_result.success:
                    result.steps_completed += 1
                    self._completed_steps.append(step)
                else:
                    # 处理失败
                    action = self._handle_failure(step, step_result, pipeline)
                    
                    if action == FailureAction.CONTINUE:
                        continue
                    elif action == FailureAction.ROLLBACK:
                        self._rollback(pipeline)
                        result.rollback_performed = True
                        result.error = step_result.error
                        break
                    elif action == FailureAction.RETRY:
                        # 重试已在 _execute_step 中处理
                        if step.retry_count >= step.max_retries:
                            result.error = step_result.error
                            break
                    else:  # STOP
                        result.error = step_result.error
                        break

            # 判断成功
            result.success = result.steps_completed == result.total_steps
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            result.error = str(e)
            
        finally:
            result.end_time = datetime.now()
            pipeline.end_time = result.end_time
            pipeline.status = "completed" if result.success else "failed"
            
            # 最终进度更新
            self._finalize_progress(pipeline, result)
            
            # 清理内存
            self._clear_memory()
        
        return result
    
    def _execute_step(
        self, 
        step: PipelineStep, 
        pipeline: PipelineDefinition
    ) -> StepResult:
        """执行单个步骤"""
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now()
        
        logger.info(f"Executing step: {step.name} ({step.type})")
        
        # 更新进度
        self._update_progress(pipeline, step, "running")
        
        # 开始监控（StrategyMonitor 不需要显式启动，直接使用 record_step 即可）
        # 如果需要重置监控器，可以调用 reset()
        if self._strategy_monitor:
            try:
                self._strategy_monitor.reset()  # 重置监控器以开始新的监控周期
            except Exception as e:
                logger.debug(f"Monitor reset error: {e}")
        
        try:
            # 执行步骤
            output = self.runner(step)
            
            # 成功
            step.status = StepStatus.COMPLETED
            step.end_time = datetime.now()
            
            # 收集指标
            metrics = self._collect_step_metrics(step)
            step.metrics = metrics
            
            logger.info(f"Step {step.name} completed in {step.duration_seconds:.2f}s")
            
            return StepResult(
                step_name=step.name,
                success=True,
                metrics=metrics,
                duration_seconds=step.duration_seconds,
                output=output,
            )
            
        except Exception as e:
            logger.error(f"Step {step.name} failed: {e}")
            
            step.status = StepStatus.FAILED
            step.end_time = datetime.now()
            step.error = str(e)
            
            return StepResult(
                step_name=step.name,
                success=False,
                error=str(e),
                duration_seconds=step.duration_seconds,
            )
        
        finally:
            # 停止监控（StrategyMonitor 不需要显式停止，监控数据已记录）
            # 可以在这里获取监控摘要
            if self._strategy_monitor:
                try:
                    # 获取监控摘要（可选）
                    summary = self._strategy_monitor.get_summary()
                    if summary:
                        logger.debug(f"Monitor summary: {summary}")
                except Exception as e:
                    logger.debug(f"Monitor summary error: {e}")
    
    def _handle_failure(
        self,
        step: PipelineStep,
        result: StepResult,
        pipeline: PipelineDefinition,
    ) -> FailureAction:
        """处理步骤失败"""
        action_str = step.on_fail.lower()
        
        try:
            action = FailureAction(action_str)
        except ValueError:
            action = FailureAction.STOP
        
        # 重试逻辑
        if action == FailureAction.RETRY:
            if step.retry_count < step.max_retries:
                step.retry_count += 1
                logger.info(f"Retrying step {step.name} ({step.retry_count}/{step.max_retries})")
                
                # 等待后重试
                time.sleep(self.config.retry_delay_seconds)
                
                # 重置状态重新执行
                step.status = StepStatus.PENDING
                step.error = None
                
                # 递归重试
                retry_result = self._execute_step(step, pipeline)
                if retry_result.success:
                    return FailureAction.CONTINUE
        
        return action
    
    def _rollback(self, pipeline: PipelineDefinition) -> None:
        """回滚已完成的步骤"""
        if not pipeline.enable_rollback:
            logger.info("Rollback disabled")
            return
        
        logger.info(f"Rolling back {len(self._completed_steps)} steps")
        
        # 逆序回滚
        for step in reversed(self._completed_steps):
            try:
                self._rollback_step(step)
                step.status = StepStatus.CANCELLED
                logger.info(f"Rolled back step: {step.name}")
            except Exception as e:
                logger.error(f"Rollback failed for step {step.name}: {e}")
        
        self._completed_steps.clear()
    
    def _rollback_step(self, step: PipelineStep) -> None:
        """回滚单个步骤"""
        # 清理内存
        self._clear_memory()
        
        # 可扩展：调用步骤特定的回滚逻辑
        logger.debug(f"Rolling back step: {step.name}")
    
    def _check_control_signals(self) -> bool:
        """检查控制信号（取消/暂停）"""
        # 检查任务注册表
        if TASK_REGISTRY_AVAILABLE:
            try:
                status = task_registry.get_task_status(self.session_id)
                if status == "cancelled":
                    self._is_cancelled = True
                elif status == "paused":
                    self._is_paused = True
            except Exception:
                pass
        
        # 检查执行服务
        if EXECUTION_SERVICE_AVAILABLE:
            try:
                exec_service = get_training_execution_service()
                status = exec_service.get_training_status(self.session_id)
                if status and status.get('status') == 'cancelled':
                    self._is_cancelled = True
            except Exception:
                pass
        
        return self._is_cancelled
    
    def _init_progress(self, pipeline: PipelineDefinition) -> None:
        """初始化进度追踪"""
        if not self._progress_manager:
            return
        
        try:
            self._progress_manager.create_progress_tracker(
                session_id=self.session_id,
                total_steps=len(pipeline.steps),
            )
        except Exception as e:
            logger.debug(f"Progress init error: {e}")
    
    def _update_progress(
        self,
        pipeline: PipelineDefinition,
        step: PipelineStep,
        status: str,
    ) -> None:
        """更新进度"""
        if not self._progress_manager:
            return
        
        try:
            progress = (pipeline.current_step_index / len(pipeline.steps)) * 100
            self._progress_manager.update_progress(
                session_id=self.session_id,
                progress=progress,
                current_step=pipeline.current_step_index,
                step_name=step.name,
                status=status,
            )
        except Exception as e:
            logger.debug(f"Progress update error: {e}")
    
    def _finalize_progress(
        self,
        pipeline: PipelineDefinition,
        result: ExecutionResult,
    ) -> None:
        """最终化进度"""
        if not self._progress_manager:
            return

        try:
            status = "completed" if result.success else "failed"
            self._progress_manager.set_status(
                session_id=self.session_id,
                status=status,
            )
        except Exception as e:
            logger.debug(f"Progress finalize error: {e}")
    
    def _collect_step_metrics(self, step: PipelineStep) -> Dict[str, Any]:
        """收集步骤指标"""
        metrics = {
            'duration_seconds': step.duration_seconds,
            'status': step.status.value,
        }
        
        # 从监控器收集指标
        if self._strategy_monitor:
            try:
                monitor_summary = self._strategy_monitor.get_summary()
                if monitor_summary:
                    metrics['monitor'] = monitor_summary
            except Exception:
                pass
        
        # 收集内存信息
        if HARDWARE_AVAILABLE:
            try:
                memory_mb = get_available_memory() / (1024 * 1024)
                metrics['available_memory_mb'] = memory_mb
            except Exception:
                pass
        
        return metrics
    
    def _clear_memory(self) -> None:
        """清理内存"""
        if HARDWARE_AVAILABLE:
            try:
                clear_memory()
            except Exception as e:
                logger.debug(f"Memory clear error: {e}")
    
    def cancel(self) -> None:
        """取消执行"""
        self._is_cancelled = True
        logger.info(f"Pipeline execution cancelled for session: {self.session_id}")
    
    def pause(self) -> None:
        """暂停执行"""
        self._is_paused = True
        logger.info(f"Pipeline execution paused for session: {self.session_id}")
    
    def resume(self) -> None:
        """恢复执行"""
        self._is_paused = False
        logger.info(f"Pipeline execution resumed for session: {self.session_id}")
    
    def diagnose(self) -> Dict[str, Any]:
        """诊断执行器状态"""
        return {
            'session_id': self.session_id,
            'is_cancelled': self._is_cancelled,
            'is_paused': self._is_paused,
            'completed_steps': len(self._completed_steps),
            'config': {
                'max_retries': self.config.max_retries,
                'enable_monitoring': self.config.enable_monitoring,
                'enable_profiling': self.config.enable_profiling,
            },
            'components': {
                'strategy_monitor': self._strategy_monitor is not None,
                'strategy_profiler': self._strategy_profiler is not None,
                'progress_manager': self._progress_manager is not None,
            },
            'layer_availability': {
                'strategy': STRATEGY_AVAILABLE,
                'distributed_strategy': DISTRIBUTED_STRATEGY_AVAILABLE,
                'hardware': HARDWARE_AVAILABLE,
                'progress': PROGRESS_AVAILABLE,
                'execution_service': EXECUTION_SERVICE_AVAILABLE,
                'task_registry': TASK_REGISTRY_AVAILABLE,
            },
        }


# ==================== 工厂函数 ====================

def create_executor(
    runner: Callable[[PipelineStep], Any],
    session_id: str,
    **config_kwargs,
) -> PipelineExecutor:
    """创建流水线执行器"""
    config = ExecutorConfig(**config_kwargs)
    return PipelineExecutor(runner, session_id, config)


def execute_pipeline(
    pipeline: PipelineDefinition,
    runner: Callable[[PipelineStep], Any],
    session_id: Optional[str] = None,
) -> ExecutionResult:
    """便捷函数：执行流水线"""
    session_id = session_id or pipeline.session_id or pipeline.pipeline_id
    executor = create_executor(runner, session_id)
    return executor.execute(pipeline)


# ==================== 导出 ====================

__all__ = [
    # 主类
    'PipelineExecutor',
    
    # 数据类
    'StepResult',
    'ExecutionResult',
    'ExecutorConfig',
    
    # 工厂函数
    'create_executor',
    'execute_pipeline',
    
    # 可用性标志
    'STRATEGY_AVAILABLE',
    'DISTRIBUTED_STRATEGY_AVAILABLE',
    'HARDWARE_AVAILABLE',
    'PROGRESS_AVAILABLE',
]

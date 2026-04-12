"""训练流水线模块

生产级训练流水线，提供：
- 声明式流水线定义
- 步骤执行和状态管理
- 失败策略处理
- 策略层和硬件层集成

架构位置：
├── pipeline/ (本模块)
│   ├── pipeline_definition.py - 流水线和步骤定义
│   ├── pipeline_executor.py - 流水线执行器
│   ├── pipeline_runner.py - 步骤运行器
│   ├── task_registry.py - 任务注册表
│   └── task_registry_interface.py - 注册表接口
├── 依赖 strategies/, lib/hardware, progress/
└── 被 orchestrator/, launcher 调用

使用示例：
from backend.modules.training.pipeline import (
        create_pipeline,
        create_three_stage_pipeline,
        PipelineExecutor,
        PipelineRunner,
)

    # 创建三阶段流水线
    pipeline = create_three_stage_pipeline(
        name="my_training",
        pretrain_params={'num_epochs': 1},
        finetune_params={'num_epochs': 2},
    )
    
    # 创建运行器和执行器
    runner = PipelineRunner(session_id="session_001")
    executor = PipelineExecutor(runner, session_id="session_001")

# 执行流水线
result = executor.execute(pipeline)
    print(f"Success: {result.success}")
"""

# 流水线定义
from .pipeline_definition import (
    # 数据类
    PipelineDefinition,
    PipelineStep,
    
    # 枚举
    StepType,
    FailureAction,
    StepStatus,
    
    # 工厂函数
    create_step,
    create_pipeline,
    create_three_stage_pipeline,
)

# 流水线执行器
from .pipeline_executor import (
    # 主类
    PipelineExecutor,
    
    # 数据类
    StepResult,
    ExecutionResult,
    ExecutorConfig,
    
    # 工厂函数
    create_executor,
    execute_pipeline,
    
    # 可用性标志
    STRATEGY_AVAILABLE as EXECUTOR_STRATEGY_AVAILABLE,
    HARDWARE_AVAILABLE as EXECUTOR_HARDWARE_AVAILABLE,
    PROGRESS_AVAILABLE as EXECUTOR_PROGRESS_AVAILABLE,
)

# 流水线运行器
from .pipeline_runner import (
    # 主类
    PipelineRunner,
    
    # 数据类
    RunnerConfig,
    StepExecutionResult,
    
    # 工厂函数
    create_pipeline_runner,
    get_runner_layer_info,
    
    # 可用性标志
    THREE_STAGE_AVAILABLE,
    STRATEGY_AVAILABLE as RUNNER_STRATEGY_AVAILABLE,
    HARDWARE_AVAILABLE as RUNNER_HARDWARE_AVAILABLE,
    PROGRESS_AVAILABLE as RUNNER_PROGRESS_AVAILABLE,
    LOSSES_AVAILABLE as RUNNER_LOSSES_AVAILABLE,
)

# 任务注册表
try:
    from .task_registry import task_registry
    from .task_registry_interface import get_task_registry
    TASK_REGISTRY_AVAILABLE = True
except ImportError:
    task_registry = None
    def get_task_registry():
        return None
    TASK_REGISTRY_AVAILABLE = False


# ==================== 便捷函数 ====================

def quick_execute_pipeline(
    name: str,
    steps: list,
    session_id: str = "default_session",
    **kwargs
) -> 'ExecutionResult':
    """快速执行流水线
    
    Args:
        name: 流水线名称
        steps: 步骤列表
        session_id: 会话 ID
        **kwargs: 其他配置
    
    Returns:
        ExecutionResult: 执行结果
    """
    pipeline = create_pipeline(name=name, steps=steps, **kwargs)
    runner = create_pipeline_runner(session_id=session_id)
    executor = create_executor(runner=runner, session_id=session_id)
    return executor.execute(pipeline)


def get_pipeline_layer_availability() -> dict:
    """获取流水线模块层可用性"""
    return {
        'pipeline_definition': True,
        'pipeline_executor': True,
        'pipeline_runner': True,
        'task_registry': TASK_REGISTRY_AVAILABLE,
        'executor_layers': {
            'strategy': EXECUTOR_STRATEGY_AVAILABLE,
            'hardware': EXECUTOR_HARDWARE_AVAILABLE,
            'progress': EXECUTOR_PROGRESS_AVAILABLE,
        },
        'runner_layers': {
            'three_stage': THREE_STAGE_AVAILABLE,
            'strategy': RUNNER_STRATEGY_AVAILABLE,
            'hardware': RUNNER_HARDWARE_AVAILABLE,
            'progress': RUNNER_PROGRESS_AVAILABLE,
            'losses': RUNNER_LOSSES_AVAILABLE,
        },
    }


def diagnose_pipeline_module() -> dict:
    """诊断流水线模块"""
    return {
        'module': 'pipeline',
        'layer_availability': get_pipeline_layer_availability(),
        'components': {
            'PipelineDefinition': PipelineDefinition is not None,
            'PipelineStep': PipelineStep is not None,
            'PipelineExecutor': PipelineExecutor is not None,
            'PipelineRunner': PipelineRunner is not None,
        },
        'factories': {
            'create_pipeline': create_pipeline is not None,
            'create_three_stage_pipeline': create_three_stage_pipeline is not None,
            'create_executor': create_executor is not None,
            'create_pipeline_runner': create_pipeline_runner is not None,
        },
    }


# ==================== 导出 ====================

__all__ = [
    # 流水线定义
    'PipelineDefinition',
    'PipelineStep',
    'StepType',
    'FailureAction',
    'StepStatus',
    'create_step',
    'create_pipeline',
    'create_three_stage_pipeline',
    
    # 执行器
    'PipelineExecutor',
    'StepResult',
    'ExecutionResult',
    'ExecutorConfig',
    'create_executor',
    'execute_pipeline',
    
    # 运行器
    'PipelineRunner',
    'RunnerConfig',
    'StepExecutionResult',
    'create_pipeline_runner',
    'get_runner_layer_info',
    
    # 任务注册表
    'task_registry',
    'get_task_registry',
    
    # 便捷函数
    'quick_execute_pipeline',
    'get_pipeline_layer_availability',
    'diagnose_pipeline_module',
    
    # 可用性标志
    'TASK_REGISTRY_AVAILABLE',
    'THREE_STAGE_AVAILABLE',
]

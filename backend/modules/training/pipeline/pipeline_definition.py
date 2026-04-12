"""训练流水线定义与加载

生产级流水线定义，提供：
- 声明式流水线配置
- 步骤依赖管理
- 失败策略
- 策略层集成

架构位置：
├── pipeline/pipeline_definition.py (本模块)
│   └── 流水线和步骤定义
├── pipeline/pipeline_executor.py
│   └── 流水线执行
└── 被 orchestrator, launcher 调用
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class StepType(str, Enum):
    """步骤类型"""
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    PREFERENCE = "preference"
    EVALUATION = "evaluation"
    THREE_STAGE = "three_stage"
    CUSTOM = "custom"


class FailureAction(str, Enum):
    """失败动作"""
    CONTINUE = "continue"
    STOP = "stop"
    ROLLBACK = "rollback"
    RETRY = "retry"


class StepStatus(str, Enum):
    """步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


# ==================== 数据类定义 ====================

@dataclass
class PipelineStep:
    """流水线步骤
    
    定义流水线中的单个执行步骤。
    """
    name: str
    type: str  # e.g. "pretrain", "finetune", "preference_optim", "custom"
    params: Dict[str, Any] = field(default_factory=dict)
    on_fail: str = "rollback"  # "continue" | "stop" | "rollback" | "retry"
    
    # 扩展字段
    step_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    timeout_seconds: int = 0  # 0 表示无超时
    retry_count: int = 0
    max_retries: int = 3
    depends_on: List[str] = field(default_factory=list)  # 依赖的步骤名称
    
    # 运行时状态
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'type': self.type,
            'params': self.params,
            'on_fail': self.on_fail,
            'step_id': self.step_id,
            'description': self.description,
            'timeout_seconds': self.timeout_seconds,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'depends_on': self.depends_on,
            'status': self.status.value if isinstance(self.status, StepStatus) else self.status,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'error': self.error,
            'metrics': self.metrics,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineStep':
        """从字典创建"""
        if 'status' in data and isinstance(data['status'], str):
            try:
                data['status'] = StepStatus(data['status'])
            except ValueError:
                data['status'] = StepStatus.PENDING
        
        if 'start_time' in data and isinstance(data['start_time'], str):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        
        if 'end_time' in data and isinstance(data['end_time'], str):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k) or k in ['name', 'type', 'params', 'on_fail']})
    
    @property
    def duration_seconds(self) -> float:
        """执行时长（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def reset(self) -> None:
        """重置步骤状态"""
        self.status = StepStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.error = None
        self.metrics = {}


@dataclass
class PipelineDefinition:
    """流水线定义
    
    定义完整的训练流水线。
    """
    name: str
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None
    model_name: Optional[str] = None
    steps: List[PipelineStep] = field(default_factory=list)
    enable_rollback: bool = True

    # 扩展字段
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    output_dir: str = "./outputs"
    
    # 全局配置
    global_params: Dict[str, Any] = field(default_factory=dict)
    
    # 运行时状态
    status: str = "pending"
    current_step_index: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'pipeline_id': self.pipeline_id,
            'tenant_id': self.tenant_id,
            'session_id': self.session_id,
            'model_name': self.model_name,
            'description': self.description,
            'output_dir': self.output_dir,
            'steps': [s.to_dict() for s in self.steps],
            'enable_rollback': self.enable_rollback,
            'global_params': self.global_params,
            'status': self.status,
            'current_step_index': self.current_step_index,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PipelineDefinition":
        """从字典创建"""
        steps = []
        for s in d.get("steps", []):
            if isinstance(s, PipelineStep):
                steps.append(s)
            elif isinstance(s, dict):
                steps.append(PipelineStep.from_dict(s))
        
        start_time = d.get('start_time')
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)
        
        end_time = d.get('end_time')
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)
        
        return PipelineDefinition(
            name=d.get("name", "training_pipeline"),
            pipeline_id=d.get("pipeline_id", str(uuid.uuid4())),
            tenant_id=d.get("tenant_id"),
            session_id=d.get("session_id"),
            model_name=d.get("model_name"),
            description=d.get("description", ""),
            output_dir=d.get("output_dir", "./outputs"),
            steps=steps,
            enable_rollback=bool(d.get("enable_rollback", True)),
            global_params=d.get("global_params", {}),
            status=d.get("status", "pending"),
            current_step_index=d.get("current_step_index", 0),
            start_time=start_time,
            end_time=end_time,
        )
    
    def add_step(self, step: Union[PipelineStep, Dict[str, Any]]) -> None:
        """添加步骤"""
        if isinstance(step, dict):
            step = PipelineStep.from_dict(step)
        self.steps.append(step)
    
    def get_step(self, name: str) -> Optional[PipelineStep]:
        """获取步骤"""
        for step in self.steps:
            if step.name == name:
                return step
        return None
    
    def get_current_step(self) -> Optional[PipelineStep]:
        """获取当前步骤"""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def get_completed_steps(self) -> List[PipelineStep]:
        """获取已完成的步骤"""
        return [s for s in self.steps if s.status == StepStatus.COMPLETED]
    
    def get_pending_steps(self) -> List[PipelineStep]:
        """获取待执行的步骤"""
        return [s for s in self.steps if s.status == StepStatus.PENDING]
    
    def reset(self) -> None:
        """重置流水线状态"""
        self.status = "pending"
        self.current_step_index = 0
        self.start_time = None
        self.end_time = None
        for step in self.steps:
            step.reset()
    
    @property
    def progress(self) -> float:
        """计算进度百分比"""
        if not self.steps:
            return 0.0
        completed = len(self.get_completed_steps())
        return (completed / len(self.steps)) * 100
    
    @property
    def duration_seconds(self) -> float:
        """执行时长（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


# ==================== 工厂函数 ====================

def create_step(
    name: str,
    step_type: str,
    params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PipelineStep:
    """创建流水线步骤"""
    return PipelineStep(
        name=name,
        type=step_type,
        params=params or {},
        **kwargs
    )


def create_pipeline(
    name: str,
    steps: Optional[List[Union[PipelineStep, Dict[str, Any]]]] = None,
    **kwargs
) -> PipelineDefinition:
    """创建流水线定义"""
    pipeline = PipelineDefinition(name=name, **kwargs)
    
    if steps:
        for step in steps:
            pipeline.add_step(step)
    
    return pipeline


def create_three_stage_pipeline(
    name: str = "three_stage_pipeline",
    pretrain_params: Optional[Dict[str, Any]] = None,
    finetune_params: Optional[Dict[str, Any]] = None,
    preference_params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> PipelineDefinition:
    """创建三阶段训练流水线"""
    steps = []
    
    if pretrain_params is not None:
        steps.append(PipelineStep(
            name="pretrain",
            type="pretrain",
            params=pretrain_params,
            on_fail="stop",
        ))
    
    if finetune_params is not None:
        steps.append(PipelineStep(
            name="finetune",
            type="finetune",
            params=finetune_params,
            on_fail="stop",
            depends_on=["pretrain"] if pretrain_params else [],
        ))
    
    if preference_params is not None:
        steps.append(PipelineStep(
            name="preference",
            type="preference",
            params=preference_params,
            on_fail="stop",
            depends_on=["finetune"] if finetune_params else [],
        ))
    
    return create_pipeline(name=name, steps=steps, **kwargs)


# ==================== 导出 ====================

__all__ = [
    # 数据类
    'PipelineStep',
    'PipelineDefinition',
    
    # 枚举
    'StepType',
    'FailureAction',
    'StepStatus',
    
    # 工厂函数
    'create_step',
    'create_pipeline',
    'create_three_stage_pipeline',
]

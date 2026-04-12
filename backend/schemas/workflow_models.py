"""工作流数据库模型

定义工作流相关的数据库模型，包括：
- 工作流定义 (Workflow)
- 工作流执行记录 (WorkflowExecution)
- 工作流步骤 (WorkflowStep)
- 工作流模板 (WorkflowTemplate)
- 工作流日志 (WorkflowLog)
"""

from sqlalchemy import Column, String, Text, Boolean, Integer, Float, DateTime, Numeric, func
from decimal import Decimal
from datetime import datetime
from enum import Enum

from .base_models import Base, UUIDMixin, TimestampMixin, TenantMixin


# ============================================================================
# 工作流相关枚举
# ============================================================================

class WorkflowStatus(str, Enum):
    """工作流状态枚举"""
    DRAFT = "draft"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class WorkflowType(str, Enum):
    """工作流类型枚举"""
    DATA_PREPROCESSING = "data_preprocessing"
    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    MODEL_DEPLOYMENT = "model_deployment"
    DATA_PIPELINE = "data_pipeline"
    ETL = "etl"
    INFERENCE = "inference"
    CUSTOM = "custom"


class ExecutionStatus(str, Enum):
    """执行状态枚举"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StepStatus(str, Enum):
    """步骤状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class StepType(str, Enum):
    """步骤类型枚举"""
    DATA_LOAD = "data_load"
    DATA_TRANSFORM = "data_transform"
    DATA_VALIDATE = "data_validate"
    TRAIN = "train"
    EVALUATE = "evaluate"
    DEPLOY = "deploy"
    NOTIFY = "notify"
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"
    CUSTOM = "custom"


class LogLevel(str, Enum):
    """日志级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ============================================================================
# 工作流数据库模型
# ============================================================================

class Workflow(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """工作流定义模型"""
    __tablename__ = 'workflows'
    
    name = Column(String(200), nullable=False, index=True, comment="工作流名称")
    description = Column(Text, comment="工作流描述")
    workflow_type = Column(String(50), nullable=False, index=True, comment="工作流类型")
    status = Column(String(20), default='draft', index=True, comment="工作流状态")
    version = Column(Integer, default=1, comment="版本号")
    
    # 创建者信息
    created_by = Column(String(36), nullable=False, index=True, comment="创建者用户ID")
    updated_by = Column(String(36), comment="最后更新者用户ID")
    
    # 配置信息
    config = Column(Text, comment="工作流配置(JSON)")
    steps_config = Column(Text, comment="步骤配置(JSON)")
    trigger_config = Column(Text, comment="触发器配置(JSON)")
    notification_config = Column(Text, comment="通知配置(JSON)")
    
    # 调度信息
    schedule_enabled = Column(Boolean, default=False, comment="是否启用调度")
    schedule_cron = Column(String(100), comment="调度CRON表达式")
    schedule_timezone = Column(String(50), default='UTC', comment="调度时区")
    next_run_at = Column(DateTime, comment="下次运行时间")
    last_run_at = Column(DateTime, comment="上次运行时间")
    
    # 统计信息
    execution_count = Column(Integer, default=0, comment="执行次数")
    success_count = Column(Integer, default=0, comment="成功次数")
    failure_count = Column(Integer, default=0, comment="失败次数")
    avg_duration_seconds = Column(Float, default=0, comment="平均执行时长(秒)")
    
    # 资源限制
    timeout_seconds = Column(Integer, default=3600, comment="超时时间(秒)")
    max_retries = Column(Integer, default=3, comment="最大重试次数")
    retry_delay_seconds = Column(Integer, default=60, comment="重试间隔(秒)")
    
    # 标签和分类
    tags = Column(Text, comment="标签(JSON数组)")
    category = Column(String(100), comment="分类")
    
    # 模板关联
    template_id = Column(String(36), index=True, comment="模板ID")
    is_template = Column(Boolean, default=False, index=True, comment="是否为模板")
    
    def __repr__(self):
        return f"<Workflow(id='{self.id}', name='{self.name}', type='{self.workflow_type}')>"


class WorkflowExecution(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """工作流执行记录模型"""
    __tablename__ = 'workflow_executions'
    
    workflow_id = Column(String(36), nullable=False, index=True, comment="工作流ID")
    workflow_name = Column(String(200), comment="工作流名称(冗余)")
    workflow_version = Column(Integer, comment="执行时的工作流版本")
    
    # 执行状态
    status = Column(String(20), default='pending', index=True, comment="执行状态")
    progress = Column(Float, default=0, comment="执行进度(0-100)")
    current_step = Column(String(200), comment="当前步骤")
    current_step_index = Column(Integer, default=0, comment="当前步骤索引")
    total_steps = Column(Integer, default=0, comment="总步骤数")
    
    # 执行者信息
    triggered_by = Column(String(36), index=True, comment="触发者用户ID")
    trigger_type = Column(String(50), default='manual', comment="触发类型")
    
    # 时间信息
    started_at = Column(DateTime, comment="开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    duration_seconds = Column(Float, comment="执行时长(秒)")
    
    # 输入输出
    input_data = Column(Text, comment="输入数据(JSON)")
    output_data = Column(Text, comment="输出数据(JSON)")
    context_data = Column(Text, comment="上下文数据(JSON)")
    
    # 错误信息
    error_message = Column(Text, comment="错误消息")
    error_step = Column(String(200), comment="错误步骤")
    error_details = Column(Text, comment="错误详情(JSON)")
    
    # 重试信息
    retry_count = Column(Integer, default=0, comment="重试次数")
    parent_execution_id = Column(String(36), index=True, comment="父执行ID(重试时)")
    
    # 资源使用
    cpu_seconds = Column(Float, default=0, comment="CPU使用(秒)")
    memory_mb_seconds = Column(Float, default=0, comment="内存使用(MB*秒)")
    gpu_seconds = Column(Float, default=0, comment="GPU使用(秒)")
    
    # 优先级
    priority = Column(Integer, default=5, comment="优先级(1-10)")
    
    # 取消信息
    cancelled_by = Column(String(36), comment="取消者用户ID")
    cancel_reason = Column(Text, comment="取消原因")
    
    def __repr__(self):
        return f"<WorkflowExecution(id='{self.id}', workflow_id='{self.workflow_id}', status='{self.status}')>"


class WorkflowStep(Base, UUIDMixin, TimestampMixin):
    """工作流步骤执行记录模型"""
    __tablename__ = 'workflow_steps'
    
    execution_id = Column(String(36), nullable=False, index=True, comment="执行记录ID")
    workflow_id = Column(String(36), nullable=False, index=True, comment="工作流ID")
    
    # 步骤信息
    step_name = Column(String(200), nullable=False, comment="步骤名称")
    step_type = Column(String(50), comment="步骤类型")
    step_index = Column(Integer, default=0, comment="步骤顺序")
    
    # 状态信息
    status = Column(String(20), default='pending', index=True, comment="步骤状态")
    progress = Column(Float, default=0, comment="步骤进度(0-100)")
    
    # 时间信息
    started_at = Column(DateTime, comment="开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    duration_seconds = Column(Float, comment="执行时长(秒)")
    
    # 输入输出
    input_data = Column(Text, comment="输入数据(JSON)")
    output_data = Column(Text, comment="输出数据(JSON)")
    
    # 配置
    config = Column(Text, comment="步骤配置(JSON)")
    
    # 错误信息
    error_message = Column(Text, comment="错误消息")
    error_details = Column(Text, comment="错误详情(JSON)")
    
    # 重试信息
    retry_count = Column(Integer, default=0, comment="重试次数")
    max_retries = Column(Integer, default=3, comment="最大重试次数")
    
    def __repr__(self):
        return f"<WorkflowStep(id='{self.id}', execution_id='{self.execution_id}', name='{self.step_name}')>"


class WorkflowTemplate(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """工作流模板模型"""
    __tablename__ = 'workflow_templates'
    
    name = Column(String(200), nullable=False, index=True, comment="模板名称")
    description = Column(Text, comment="模板描述")
    workflow_type = Column(String(50), nullable=False, index=True, comment="工作流类型")
    
    # 创建者信息
    created_by = Column(String(36), nullable=False, index=True, comment="创建者用户ID")
    
    # 配置信息
    config = Column(Text, comment="模板配置(JSON)")
    steps_config = Column(Text, comment="步骤配置(JSON)")
    default_params = Column(Text, comment="默认参数(JSON)")
    
    # 可见性
    is_public = Column(Boolean, default=False, index=True, comment="是否公开")
    is_system = Column(Boolean, default=False, index=True, comment="是否系统模板")
    
    # 统计信息
    use_count = Column(Integer, default=0, comment="使用次数")
    
    # 版本信息
    version = Column(String(50), default='1.0.0', comment="版本号")
    
    # 标签和分类
    tags = Column(Text, comment="标签(JSON数组)")
    category = Column(String(100), comment="分类")
    
    # 图标和缩略图
    icon = Column(String(200), comment="图标URL")
    thumbnail = Column(String(500), comment="缩略图URL")
    
    def __repr__(self):
        return f"<WorkflowTemplate(id='{self.id}', name='{self.name}', type='{self.workflow_type}')>"


class WorkflowLog(Base, UUIDMixin):
    """工作流日志模型"""
    __tablename__ = 'workflow_logs'
    
    execution_id = Column(String(36), nullable=False, index=True, comment="执行记录ID")
    workflow_id = Column(String(36), nullable=False, index=True, comment="工作流ID")
    step_id = Column(String(36), index=True, comment="步骤ID")
    
    # 日志信息
    level = Column(String(20), default='info', index=True, comment="日志级别")
    message = Column(Text, nullable=False, comment="日志消息")
    details = Column(Text, comment="详细信息(JSON)")
    
    # 时间信息
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True, comment="时间戳")
    
    # 来源信息
    source = Column(String(100), comment="日志来源")
    
    def __repr__(self):
        return f"<WorkflowLog(id='{self.id}', execution_id='{self.execution_id}', level='{self.level}')>"


class WorkflowVariable(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """工作流变量模型（全局变量/环境变量）"""
    __tablename__ = 'workflow_variables'
    
    workflow_id = Column(String(36), index=True, comment="工作流ID(空为全局)")
    
    name = Column(String(100), nullable=False, index=True, comment="变量名")
    value = Column(Text, comment="变量值")
    value_type = Column(String(50), default='string', comment="值类型")
    is_secret = Column(Boolean, default=False, comment="是否为密钥")
    description = Column(Text, comment="变量描述")
    
    # 创建者
    created_by = Column(String(36), nullable=False, comment="创建者用户ID")
    
    def __repr__(self):
        return f"<WorkflowVariable(id='{self.id}', name='{self.name}')>"


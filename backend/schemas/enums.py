"""统一枚举定义模块

整合所有模块中的枚举定义，消除重复代码
遵循DRY原则，提供统一的枚举接口
"""

from enum import Enum


# ============================================================================
# 训练相关枚举
# ============================================================================

class TrainingStatus(str, Enum):
    """训练状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StageStatus(str, Enum):
    """阶段状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TrainingType(str, Enum):
    """训练类型枚举"""
    SINGLE_NODE = "single_node"      # 单节点训练
    DISTRIBUTED = "distributed"      # 分布式训练
    FEDERATED = "federated"          # 联邦学习
    TRANSFER = "transfer"            # 迁移学习
    FINE_TUNING = "fine_tuning"      # 微调
    HYPERPARAMETER = "hyperparameter" # 超参数优化


class TrainingStage(str, Enum):
    """训练阶段枚举"""
    INITIALIZATION = "initialization"
    DATA_LOADING = "data_loading"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    VALIDATION = "validation"
    EVALUATION = "evaluation"
    SAVING = "saving"
    COMPLETED = "completed"


class TrainingScenario(str, Enum):
    """训练场景枚举"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    SEMI_SUPERVISED = "semi_supervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    FINE_TUNING = "fine_tuning"
    # 添加实际使用的场景类型
    BASIC_MODEL = "basic_model"
    SCHEDULED_TASK = "scheduled_task"
    ADVANCED_MODEL = "advanced_model"
    RESEARCH_EXPERIMENT = "research_experiment"
    PRODUCTION_FINETUNE = "production_finetune"
    # 超参数优化常用场景
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"


class TrainingMethod(str, Enum):
    """训练方法枚举"""
    STANDARD = "standard"
    DISTRIBUTED = "distributed"
    FEDERATED = "federated"
    INCREMENTAL = "incremental"
    ONLINE = "online"
    BATCH = "batch"
    MULTIMODAL = "multimodal"      # 多模态训练
    TEXT_ONLY = "text_only"        # 纯文本训练
    DATABASE = "database"          # 数据库训练
    MOE = "moe"                    # 混合专家模型训练
    KNOWLEDGE_DISTILLATION = "knowledge_distillation" # 知识蒸馏
    HYPERPARAMETER_SEARCH = "hyperparameter_search" # 超参数搜索
    ENHANCED = "enhanced"          # 增强型训练


class TrainingPriority(str, Enum):
    """训练优先级枚举"""
    LOW = "low"                    # 低优先级
    NORMAL = "normal"              # 普通优先级
    HIGH = "high"                  # 高优先级
    URGENT = "urgent"              # 紧急优先级
    CRITICAL = "critical"          # 关键优先级


class ScheduleType(str, Enum):
    """调度类型枚举"""
    IMMEDIATE = "immediate"        # 立即执行
    SCHEDULED = "scheduled"        # 定时执行
    RECURRING = "recurring"        # 循环执行
    CONDITIONAL = "conditional"    # 条件触发
    MANUAL = "manual"              # 手动触发


# ============================================================================
# 监控和告警相关枚举
# ============================================================================


class ModelStatus(str, Enum):
    """模型状态枚举"""
    DRAFT = "draft"
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class DatasetStatus(str, Enum):
    """数据集状态枚举"""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    ARCHIVED = "archived"


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# 用户和权限相关枚举
# ============================================================================

class UserStatus(str, Enum):
    """用户状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"
    DELETED = "deleted"


class UserRole(str, Enum):
    """用户角色枚举"""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    DEVELOPER = "developer"
    SUPERUSER = "superuser"


class RoleType(str, Enum):
    """用户角色类型枚举（别名）"""
    USER = "user"
    ADMIN = "admin"
    SUPERUSER = "superuser"
    VIEWER = "viewer"


# ============================================================================
# 租户相关枚举
# ============================================================================

class TenantStatus(str, Enum):
    """租户状态枚举"""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ERROR = "error"
    DELETED = "deleted"


class PermissionType(str, Enum):
    """权限类型枚举"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


class ResourceType(str, Enum):
    """资源类型枚举"""
    MODEL = "model"
    DATASET = "dataset"
    TRAINING = "training"
    PROJECT = "project"
    USER = "user"


class NotificationType(str, Enum):
    """通知类型枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class LogLevel(str, Enum):
    """日志级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """指标类型枚举"""
    ACCURACY = "accuracy"
    LOSS = "loss"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    CUSTOM = "custom"


class OptimizationStatus(str, Enum):
    """优化状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeploymentStatus(str, Enum):
    """部署状态枚举"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    STOPPED = "stopped"
    UPDATING = "updating"


class EnvironmentType(str, Enum):
    """环境类型枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class StorageType(str, Enum):
    """存储类型枚举"""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    HDFS = "hdfs"


class ComputeType(str, Enum):
    """计算类型枚举"""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    DISTRIBUTED = "distributed"


class SchedulerStatus(str, Enum):
    """调度器状态枚举"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class WorkflowStatus(str, Enum):
    """工作流状态枚举"""
    DRAFT = "draft"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class AgentStatus(str, Enum):
    """智能体状态枚举"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    TRAINING = "training"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class AgentType(str, Enum):
    """智能体类型枚举"""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    GENERATOR = "generator"
    DETECTOR = "detector"
    RECOMMENDER = "recommender"
    CUSTOM = "custom"


# ============================================================================
# 训练流水线相关枚举
# ============================================================================

class PipelineStatus(str, Enum):
    """流水线状态枚举"""
    DRAFT = "draft"                  # 草稿
    CREATED = "created"              # 已创建
    PENDING = "pending"              # 等待执行
    RUNNING = "running"              # 运行中
    PAUSED = "paused"                # 已暂停
    COMPLETED = "completed"          # 已完成
    FAILED = "failed"                # 失败
    CANCELLED = "cancelled"          # 已取消
    ROLLED_BACK = "rolled_back"      # 已回滚


class PipelineStepType(str, Enum):
    """流水线步骤类型枚举"""
    PRETRAIN = "pretrain"                    # 预训练
    FINETUNE = "finetune"                    # 微调
    SFT = "sft"                              # 监督微调
    PREFERENCE_OPTIM = "preference_optim"    # 偏好优化 (DPO/RLHF)
    EVALUATION = "evaluation"                # 评估
    VALIDATION = "validation"                # 验证
    DATA_PROCESSING = "data_processing"      # 数据处理
    MODEL_EXPORT = "model_export"            # 模型导出
    DEPLOYMENT = "deployment"                # 部署
    CHECKPOINT = "checkpoint"                # 检查点保存
    CUSTOM = "custom"                        # 自定义步骤


class PipelineFailurePolicy(str, Enum):
    """流水线失败策略枚举"""
    CONTINUE = "continue"            # 继续执行后续步骤
    STOP = "stop"                    # 停止流水线
    ROLLBACK = "rollback"            # 回滚到上一步骤
    RETRY = "retry"                  # 重试当前步骤


class PipelineExecutionStatus(str, Enum):
    """流水线执行状态枚举"""
    QUEUED = "queued"                # 排队中
    INITIALIZING = "initializing"    # 初始化中
    RUNNING = "running"              # 运行中
    STEP_COMPLETED = "step_completed"  # 步骤完成
    PAUSED = "paused"                # 已暂停
    RESUMING = "resuming"            # 恢复中
    COMPLETING = "completing"        # 完成中
    COMPLETED = "completed"          # 已完成
    FAILED = "failed"                # 失败
    CANCELLED = "cancelled"          # 已取消


class AlertLevel(str, Enum):
    """告警级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """告警状态枚举"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


class MonitoringTarget(str, Enum):
    """监控目标枚举"""
    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    STORAGE = "storage"
    TRAINING = "training"
    MODEL = "model"


class ProjectStatus(str, Enum):
    """项目状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ModelType(str, Enum):
    """模型类型枚举"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    DETECTION = "detection"
    GENERATION = "generation"
    RECOMMENDATION = "recommendation"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    CUSTOM = "custom"


class ModelFramework(str, Enum):
    """模型框架枚举"""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SCIKIT_LEARN = "scikit_learn"
    KERAS = "keras"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    ONNX = "onnx"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


# ============================================================================
# 导出所有枚举
# ============================================================================

__all__ = [
    # 训练相关
    'TrainingStatus',
    'StageStatus',
    'TrainingType',
    'TrainingStage',
    'TrainingScenario',
    'TrainingMethod',
    'TrainingPriority',
    'ScheduleType',
    
    # 监控和告警相关
    'MetricType',
    'AlertLevel',
    'AlertStatus',
    'ResourceType',
    'MonitoringTarget',
    
    # 用户和权限相关
    'UserStatus',
    'UserRole',
    'RoleType',
    
    # 项目和数据集相关
    'ProjectStatus',
    'DatasetStatus',
    
    # 模型相关
    'ModelType',
    'ModelFramework',
    'ModelStatus',
    'DeploymentStatus',
    
    # 租户相关
    'TenantStatus',
    
    # 任务相关
    'TaskStatus',
    
    # 权限相关
    'PermissionType',
    
    # 通知相关
    'NotificationType',
    
    # 日志相关
    'LogLevel',
    
    # 优化相关
    'OptimizationStatus',
    
    # 环境相关
    'EnvironmentType',
    
    # 存储相关
    'StorageType',
    
    # 计算相关
    'ComputeType',
    
    # 调度器相关
    'SchedulerStatus',
    
    # 工作流相关
    'WorkflowStatus',
    
    # 智能体相关
    'AgentStatus',
    'AgentType',
    
    # 训练流水线相关
    'PipelineStatus',
    'PipelineStepType',
    'PipelineFailurePolicy',
]
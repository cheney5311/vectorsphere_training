"""数据库模块

提供数据库连接、操作和管理功能。
"""

from backend.schemas.base_models import Base, TimestampMixin, UUIDMixin, TenantMixin
from backend.schemas.enums import (
    TrainingStatus, TrainingStage, TrainingScenario, TrainingMethod,
    MetricType, AlertLevel, AlertStatus, ResourceType, MonitoringTarget,
    UserRole, PermissionType,
    ProjectStatus, DatasetStatus,
    ModelType, ModelFramework, ModelStatus, DeploymentStatus
)
# 移除schema模型的导入以避免循环导入
# 这些模型可以在需要时直接从对应的模块导入
from .config import DatabaseConfig, get_database_config, set_database_config
from .manager import DatabaseManager, get_database_manager, close_database_manager
from .service import DatabaseService, get_database_service, close_database_service
from .trainer import (
    DatabaseTrainingConfig,
    DatabaseTrainer,
    create_database_trainer,
    launch_database_training
)
from backend.api.database.api import database_bp
from typing import Optional
import redis

# 添加获取数据库会话的便捷函数
def get_db_session():
    """获取数据库会话"""
    db_manager = get_database_manager()
    return db_manager.get_session()

# 添加获取Redis客户端的便捷函数（如果有的话）
try:
    from backend.core.redis_client import get_redis_client
except ImportError:
    def get_redis_client() -> Optional[redis.Redis]:
        """获取Redis客户端（模拟实现）"""
        return None

__all__ = [
    # 基础模型
    'Base',
    'TimestampMixin',
    'UUIDMixin',
    'TenantMixin',
    
    # Redis客户端
    'get_redis_client',
    
    # 枚举
    'TrainingStatus',
    'StageStatus',
    'TrainingType',
    'MetricType',
    'AlertLevel',
    'AlertStatus',
    'ResourceType',
    'MonitoringTarget',
    'UserStatus',
    'RoleType',
    'ProjectStatus',
    'DatasetStatus',
    'ModelType',
    'ModelFramework',
    'ModelStatus',
    'DeploymentStatus',
    'TrainingScenario',
    'TrainingStage',
    'TrainingMethod',
    'ScheduleType',
    'TrainingPriority',
    
    # 认证模型
    'User',
    'UserSession',
    'ApiKey',
    'Role',
    'UserRole',
    'Permission',
    'role_permissions',
    
    # 训练模型
    'TrainingSession',
    'TrainingProgress',
    'ThreeStageSession',
    'ThreeStageProgress',
    
    # 监控模型
    'SystemMetric',
    'Alert',
    'SystemLog',
    
    # 项目模型
    'Project',
    'Dataset',
    'Model',
    'ModelVersion',
    'ModelDeployment',
    
    # 配置
    'DatabaseConfig',
    'get_database_config',
    'set_database_config',
    
    # 管理器
    'DatabaseManager',
    'get_database_manager',
    'close_database_manager',
    
    # 服务
    'DatabaseService',
    'get_database_service',
    'close_database_service',
    
    # 训练能力
    'DatabaseTrainingConfig',
    'DatabaseTrainer',
    'create_database_trainer',
    'launch_database_training',
    
    # API
    'database_bp',
]
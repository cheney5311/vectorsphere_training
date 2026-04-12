"""配置管理模块

提供统一的配置管理功能，适配新架构需求。
"""

from .config import (
    DatabaseConfig,
    RedisConfig,
    JWTConfig,
    SecurityConfig,
    EmailConfig,
    StorageConfig,
    LoggingConfig,
    MonitoringConfig,
    DistributedConfig,
    TrainingConfig,
    APIConfig,
    TenantConfig,
    Config,
    get_config,
    reload_config,
    get_database_url,
    get_redis_url,
    is_development,
    is_production,
    is_testing
)

from .optimized_config import (
    ConfigSource,
    ConfigValidationError,
    ConfigMetadata,
    ConfigObserver,
    OptimizedConfigManager,
    DynamicTrainingConfig,
    TrainingConfigObserver,
    get_config_manager,
    load_training_config,
    update_training_config
)

__all__ = [
    'DatabaseConfig',
    'RedisConfig', 
    'JWTConfig',
    'SecurityConfig',
    'EmailConfig',
    'StorageConfig',
    'LoggingConfig',
    'MonitoringConfig',
    'DistributedConfig',
    'TrainingConfig',
    'APIConfig',
    'TenantConfig',
    'Config',
    'get_config',
    'reload_config',
    'get_database_url',
    'get_redis_url',
    'is_development',
    'is_production',
    'is_testing',
    'ConfigSource',
    'ConfigValidationError',
    'ConfigMetadata',
    'ConfigObserver',
    'OptimizedConfigManager',
    'DynamicTrainingConfig',
    'TrainingConfigObserver',
    'get_config_manager',
    'load_training_config',
    'update_training_config'
]
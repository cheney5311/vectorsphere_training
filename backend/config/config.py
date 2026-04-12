"""VectorSphere Intelligent Platform - 配置管理 (新架构)

统一管理所有系统配置参数和环境变量，适配新架构需求。
"""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# 配置日志记录器
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """数据库配置"""
    # 数据库类型
    db_type: str = field(default_factory=lambda: os.getenv('DB_TYPE', 'postgresql'))
    
    # PostgreSQL/MySQL 配置
    host: str = field(default_factory=lambda: os.getenv('DB_HOST', 'localhost'))
    port: int = field(default_factory=lambda: int(os.getenv('DB_PORT', '5432')))
    name: str = field(default_factory=lambda: os.getenv('DB_NAME', 'vectorsphere'))
    user: str = field(default_factory=lambda: os.getenv('DB_USER', 'postgres'))
    password: str = field(default_factory=lambda: os.getenv('DB_PASSWORD', 'password'))
    
    # SQLite 配置
    sqlite_path: str = field(default_factory=lambda: os.getenv('DB_SQLITE_PATH', str(PROJECT_ROOT / 'data' / 'vectorsphere.db')))
    
    # 连接池配置
    pool_size: int = field(default_factory=lambda: int(os.getenv('DB_POOL_SIZE', '20')))
    max_overflow: int = field(default_factory=lambda: int(os.getenv('DB_MAX_OVERFLOW', '30')))
    pool_timeout: int = field(default_factory=lambda: int(os.getenv('DB_POOL_TIMEOUT', '30')))
    pool_recycle: int = field(default_factory=lambda: int(os.getenv('DB_POOL_RECYCLE', '3600')))
    
    # SSL配置 (仅适用于PostgreSQL/MySQL)
    ssl_mode: str = field(default_factory=lambda: os.getenv('DB_SSL_MODE', 'prefer'))
    ssl_cert: Optional[str] = field(default_factory=lambda: os.getenv('DB_SSL_CERT'))
    ssl_key: Optional[str] = field(default_factory=lambda: os.getenv('DB_SSL_KEY'))
    ssl_ca: Optional[str] = field(default_factory=lambda: os.getenv('DB_SSL_CA'))
    
    # 其他配置
    echo: bool = field(default_factory=lambda: os.getenv('DB_ECHO', 'false').lower() == 'true')
    echo_pool: bool = field(default_factory=lambda: os.getenv('DB_ECHO_POOL', 'false').lower() == 'true')
    
    @property
    def url(self) -> str:
        """生成数据库连接URL"""
        # 优先使用DATABASE_URL环境变量（向后兼容性）
        database_url = os.getenv('DATABASE_URL')
        if database_url:
            return database_url
            
        if self.db_type == 'postgresql':
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        elif self.db_type == 'mysql':
            return f"mysql+pymysql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        elif self.db_type == 'sqlite':
            # 确保SQLite数据库目录存在
            sqlite_dir = Path(self.sqlite_path).parent
            sqlite_dir.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{self.sqlite_path}"
        else:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")
    
    def get_port_for_type(self) -> int:
        """根据数据库类型获取默认端口"""
        if self.db_type == 'postgresql':
            return int(os.getenv('DB_PORT', '5432'))
        elif self.db_type == 'mysql':
            return int(os.getenv('DB_PORT', '3306'))
        else:
            return self.port


@dataclass
class RedisConfig:
    """Redis配置"""
    host: str = os.getenv('REDIS_HOST', 'localhost')
    port: int = int(os.getenv('REDIS_PORT', '6379'))
    db: int = int(os.getenv('REDIS_DB', '0'))
    password: Optional[str] = os.getenv('REDIS_PASSWORD')
    max_connections: int = int(os.getenv('REDIS_MAX_CONNECTIONS', '100'))
    socket_timeout: int = int(os.getenv('REDIS_SOCKET_TIMEOUT', '5'))
    socket_connect_timeout: int = int(os.getenv('REDIS_SOCKET_CONNECT_TIMEOUT', '5'))
    retry_on_timeout: bool = os.getenv('REDIS_RETRY_ON_TIMEOUT', 'true').lower() == 'true'
    health_check_interval: int = int(os.getenv('REDIS_HEALTH_CHECK_INTERVAL', '30'))
    
    @property
    def url(self) -> str:
        """获取Redis连接URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


@dataclass
class JWTConfig:
    """JWT配置"""
    secret_key: str = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
    algorithm: str = os.getenv('JWT_ALGORITHM', 'HS256')
    access_token_expires: int = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', '3600'))  # 1小时
    refresh_token_expires: int = int(os.getenv('JWT_REFRESH_TOKEN_EXPIRES', '2592000'))  # 30天
    issuer: str = os.getenv('JWT_ISSUER', 'vectorsphere')
    audience: str = os.getenv('JWT_AUDIENCE', 'vectorsphere-api')


@dataclass
class SecurityConfig:
    """安全配置"""
    password_min_length: int = int(os.getenv('PASSWORD_MIN_LENGTH', '8'))
    password_require_uppercase: bool = os.getenv('PASSWORD_REQUIRE_UPPERCASE', 'true').lower() == 'true'
    password_require_lowercase: bool = os.getenv('PASSWORD_REQUIRE_LOWERCASE', 'true').lower() == 'true'
    password_require_numbers: bool = os.getenv('PASSWORD_REQUIRE_NUMBERS', 'true').lower() == 'true'
    password_require_symbols: bool = os.getenv('PASSWORD_REQUIRE_SYMBOLS', 'false').lower() == 'true'
    max_login_attempts: int = int(os.getenv('MAX_LOGIN_ATTEMPTS', '5'))
    lockout_duration: int = int(os.getenv('LOCKOUT_DURATION', '900'))  # 15分钟
    session_timeout: int = int(os.getenv('SESSION_TIMEOUT', '3600'))  # 1小时
    csrf_protection: bool = os.getenv('CSRF_PROTECTION', 'true').lower() == 'true'
    cors_origins: List[str] = field(default_factory=lambda: os.getenv('CORS_ORIGINS', '*').split(','))
    rate_limit_per_minute: int = int(os.getenv('RATE_LIMIT_PER_MINUTE', '100'))
    rate_limit_per_hour: int = int(os.getenv('RATE_LIMIT_PER_HOUR', '5000'))


@dataclass
class EmailConfig:
    """邮件配置"""
    smtp_server: str = os.getenv('SMTP_SERVER', 'localhost')
    smtp_port: int = int(os.getenv('SMTP_PORT', '587'))
    smtp_username: Optional[str] = os.getenv('SMTP_USERNAME')
    smtp_password: Optional[str] = os.getenv('SMTP_PASSWORD')
    use_tls: bool = os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'
    use_ssl: bool = os.getenv('SMTP_USE_SSL', 'false').lower() == 'true'
    from_email: str = os.getenv('FROM_EMAIL', 'noreply@vectorsphere.com')
    from_name: str = os.getenv('FROM_NAME', 'VectorSphere Platform')
    template_dir: str = os.getenv('EMAIL_TEMPLATE_DIR', str(PROJECT_ROOT / 'templates' / 'email'))


@dataclass
class StorageConfig:
    """存储配置"""
    # 本地存储
    local_storage_path: str = os.getenv('LOCAL_STORAGE_PATH', str(PROJECT_ROOT / 'storage'))
    max_file_size: int = int(os.getenv('MAX_FILE_SIZE', str(500 * 1024 * 1024)))  # 500MB
    allowed_extensions: List[str] = field(default_factory=lambda: os.getenv('ALLOWED_EXTENSIONS', '.txt,.csv,.json,.pkl,.h5,.pt,.pth,.onnx,.zip,.tar.gz,.yaml,.yml').split(','))
    
    # 云存储
    cloud_provider: str = os.getenv('CLOUD_PROVIDER', 'local')  # local, aws, azure, gcp
    
    # AWS S3
    aws_access_key_id: Optional[str] = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key: Optional[str] = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region: str = os.getenv('AWS_REGION', 'us-east-1')
    aws_bucket_name: str = os.getenv('AWS_BUCKET_NAME', 'vectorsphere-storage')
    
    # Azure Blob Storage
    azure_account_name: Optional[str] = os.getenv('AZURE_ACCOUNT_NAME')
    azure_account_key: Optional[str] = os.getenv('AZURE_ACCOUNT_KEY')
    azure_container_name: str = os.getenv('AZURE_CONTAINER_NAME', 'vectorsphere-storage')
    
    # Google Cloud Storage
    gcp_project_id: Optional[str] = os.getenv('GCP_PROJECT_ID')
    gcp_bucket_name: str = os.getenv('GCP_BUCKET_NAME', 'vectorsphere-storage')
    gcp_credentials_path: Optional[str] = os.getenv('GCP_CREDENTIALS_PATH')


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = os.getenv('LOG_LEVEL', 'INFO')
    format: str = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_path: str = os.getenv('LOG_FILE_PATH', str(PROJECT_ROOT / 'logs' / 'app.log'))
    max_file_size: int = int(os.getenv('LOG_MAX_FILE_SIZE', str(50 * 1024 * 1024)))  # 50MB
    backup_count: int = int(os.getenv('LOG_BACKUP_COUNT', '10'))
    enable_console: bool = os.getenv('LOG_ENABLE_CONSOLE', 'true').lower() == 'true'
    enable_file: bool = os.getenv('LOG_ENABLE_FILE', 'true').lower() == 'true'
    enable_json: bool = os.getenv('LOG_ENABLE_JSON', 'true').lower() == 'true'
    
    # 日志聚合
    enable_elk: bool = os.getenv('LOG_ENABLE_ELK', 'false').lower() == 'true'
    elasticsearch_host: str = os.getenv('ELASTICSEARCH_HOST', 'localhost')
    elasticsearch_port: int = int(os.getenv('ELASTICSEARCH_PORT', '9200'))
    elasticsearch_index: str = os.getenv('ELASTICSEARCH_INDEX', 'vectorsphere-logs')


@dataclass
class MonitoringConfig:
    """监控配置"""
    enable_prometheus: bool = os.getenv('ENABLE_PROMETHEUS', 'true').lower() == 'true'
    prometheus_port: int = int(os.getenv('PROMETHEUS_PORT', '9090'))
    metrics_path: str = os.getenv('METRICS_PATH', '/metrics')
    
    # 健康检查
    health_check_interval: int = int(os.getenv('HEALTH_CHECK_INTERVAL', '30'))
    health_check_timeout: int = int(os.getenv('HEALTH_CHECK_TIMEOUT', '5'))
    
    # 告警
    enable_alerts: bool = os.getenv('ENABLE_ALERTS', 'true').lower() == 'true'
    alert_webhook_url: Optional[str] = os.getenv('ALERT_WEBHOOK_URL')
    alert_email_recipients: List[str] = field(default_factory=lambda: os.getenv('ALERT_EMAIL_RECIPIENTS', '').split(',') if os.getenv('ALERT_EMAIL_RECIPIENTS') else [])
    
    # 性能监控
    enable_profiling: bool = os.getenv('ENABLE_PROFILING', 'true').lower() == 'true'
    profiling_sample_rate: float = float(os.getenv('PROFILING_SAMPLE_RATE', '0.1'))


@dataclass
class DistributedConfig:
    """分布式配置"""
    # 集群配置
    cluster_name: str = os.getenv('CLUSTER_NAME', 'vectorsphere-cluster')
    node_id: str = os.getenv('NODE_ID', 'node-1')
    node_type: str = os.getenv('NODE_TYPE', 'worker')  # master, worker, hybrid
    
    # 服务发现
    enable_service_discovery: bool = os.getenv('ENABLE_SERVICE_DISCOVERY', 'true').lower() == 'true'
    consul_host: str = os.getenv('CONSUL_HOST', 'localhost')
    consul_port: int = int(os.getenv('CONSUL_PORT', '8500'))
    
    # 负载均衡
    load_balancer_type: str = os.getenv('LOAD_BALANCER_TYPE', 'least_connections')  # round_robin, least_connections, weighted
    
    # 消息队列
    message_broker: str = os.getenv('MESSAGE_BROKER', 'rabbitmq')  # redis, rabbitmq, kafka
    
    # RabbitMQ
    rabbitmq_host: str = os.getenv('RABBITMQ_HOST', 'localhost')
    rabbitmq_port: int = int(os.getenv('RABBITMQ_PORT', '5672'))
    rabbitmq_username: str = os.getenv('RABBITMQ_USERNAME', 'guest')
    rabbitmq_password: str = os.getenv('RABBITMQ_PASSWORD', 'guest')
    rabbitmq_vhost: str = os.getenv('RABBITMQ_VHOST', '/')
    
    # Kafka
    kafka_bootstrap_servers: List[str] = field(default_factory=lambda: os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(','))
    kafka_topic_prefix: str = os.getenv('KAFKA_TOPIC_PREFIX', 'vectorsphere')
    
    # 分布式锁
    distributed_lock_backend: str = os.getenv('DISTRIBUTED_LOCK_BACKEND', 'redis')  # redis, zookeeper
    lock_timeout: int = int(os.getenv('LOCK_TIMEOUT', '30'))
    
    # 分布式缓存
    cache_backend: str = os.getenv('CACHE_BACKEND', 'redis')  # redis, memcached
    cache_ttl: int = int(os.getenv('CACHE_TTL', '3600'))
    cache_max_size: int = int(os.getenv('CACHE_MAX_SIZE', '5000'))


@dataclass
class TrainingConfig:
    """训练配置"""
    # 默认训练参数
    default_batch_size: int = int(os.getenv('DEFAULT_BATCH_SIZE', '64'))
    default_learning_rate: float = float(os.getenv('DEFAULT_LEARNING_RATE', '0.0001'))
    default_epochs: int = int(os.getenv('DEFAULT_EPOCHS', '50'))
    default_optimizer: str = os.getenv('DEFAULT_OPTIMIZER', 'adamw')
    
    # 资源限制
    max_concurrent_jobs: int = int(os.getenv('MAX_CONCURRENT_JOBS', '20'))
    max_job_duration: int = int(os.getenv('MAX_JOB_DURATION', '172800'))  # 48小时
    max_gpu_per_job: int = int(os.getenv('MAX_GPU_PER_JOB', '8'))
    max_cpu_per_job: int = int(os.getenv('MAX_CPU_PER_JOB', '32'))
    max_memory_per_job: int = int(os.getenv('MAX_MEMORY_PER_JOB', '64'))  # GB
    
    # 检查点
    checkpoint_interval: int = int(os.getenv('CHECKPOINT_INTERVAL', '500'))  # 步数
    max_checkpoints: int = int(os.getenv('MAX_CHECKPOINTS', '10'))
    
    # 早停
    early_stopping_patience: int = int(os.getenv('EARLY_STOPPING_PATIENCE', '5'))
    early_stopping_min_delta: float = float(os.getenv('EARLY_STOPPING_MIN_DELTA', '0.0001'))
    
    # 模型保存
    model_storage_path: str = os.getenv('MODEL_STORAGE_PATH', str(PROJECT_ROOT / 'models'))
    auto_save_best_model: bool = os.getenv('AUTO_SAVE_BEST_MODEL', 'true').lower() == 'true'
    
    # 分布式训练
    enable_distributed_training: bool = os.getenv('ENABLE_DISTRIBUTED_TRAINING', 'true').lower() == 'true'
    distributed_backend: str = os.getenv('DISTRIBUTED_BACKEND', 'nccl')  # nccl, gloo, mpi
    
    # 混合精度
    enable_mixed_precision: bool = os.getenv('ENABLE_MIXED_PRECISION', 'true').lower() == 'true'
    
    # 梯度累积
    gradient_accumulation_steps: int = int(os.getenv('GRADIENT_ACCUMULATION_STEPS', '1'))
    
    # 学习率调度
    lr_scheduler: str = os.getenv('LR_SCHEDULER', 'cosine_with_warmup')  # cosine, linear, exponential, cosine_with_warmup
    warmup_steps: int = int(os.getenv('WARMUP_STEPS', '1000'))


@dataclass
class APIConfig:
    """API配置"""
    host: str = os.getenv('API_HOST', '0.0.0.0')
    port: int = int(os.getenv('API_PORT', '8080'))
    debug: bool = os.getenv('API_DEBUG', 'false').lower() == 'true'
    
    # API版本
    version: str = os.getenv('API_VERSION', 'v1')
    title: str = os.getenv('API_TITLE', 'VectorSphere Intelligent Platform API')
    description: str = os.getenv('API_DESCRIPTION', 'AI训练平台API接口')
    
    # 请求限制
    max_request_size: int = int(os.getenv('MAX_REQUEST_SIZE', str(50 * 1024 * 1024)))  # 50MB
    request_timeout: int = int(os.getenv('REQUEST_TIMEOUT', '600'))  # 10分钟
    
    # 分页
    default_page_size: int = int(os.getenv('DEFAULT_PAGE_SIZE', '50'))
    max_page_size: int = int(os.getenv('MAX_PAGE_SIZE', '200'))
    
    # 文档
    enable_docs: bool = os.getenv('ENABLE_DOCS', 'true').lower() == 'true'
    docs_url: str = os.getenv('DOCS_URL', '/docs')
    redoc_url: str = os.getenv('REDOC_URL', '/redoc')
    
    # CORS
    enable_cors: bool = os.getenv('ENABLE_CORS', 'true').lower() == 'true'
    cors_allow_origins: List[str] = field(default_factory=lambda: os.getenv('CORS_ALLOW_ORIGINS', '*').split(','))
    cors_allow_methods: List[str] = field(default_factory=lambda: os.getenv('CORS_ALLOW_METHODS', 'GET,POST,PUT,DELETE,OPTIONS,PATCH').split(','))
    cors_allow_headers: List[str] = field(default_factory=lambda: os.getenv('CORS_ALLOW_HEADERS', 'Content-Type,Authorization,X-Requested-With,Accept,Origin').split(','))


@dataclass
class TenantConfig:
    """多租户配置"""
    enable_multi_tenancy: bool = os.getenv('ENABLE_MULTI_TENANCY', 'true').lower() == 'true'
    default_tenant_id: str = os.getenv('DEFAULT_TENANT_ID', 'default')
    tenant_isolation_level: str = os.getenv('TENANT_ISOLATION_LEVEL', 'schema')  # schema, database, application
    
    # 资源配额
    default_cpu_quota: int = int(os.getenv('DEFAULT_CPU_QUOTA', '8'))
    default_memory_quota: int = int(os.getenv('DEFAULT_MEMORY_QUOTA', '16'))  # GB
    default_storage_quota: int = int(os.getenv('DEFAULT_STORAGE_QUOTA', '500'))  # GB
    default_gpu_quota: int = int(os.getenv('DEFAULT_GPU_QUOTA', '2'))
    
    # 计费
    enable_billing: bool = os.getenv('ENABLE_BILLING', 'true').lower() == 'true'
    billing_currency: str = os.getenv('BILLING_CURRENCY', 'USD')
    cpu_hour_price: float = float(os.getenv('CPU_HOUR_PRICE', '0.05'))
    memory_gb_hour_price: float = float(os.getenv('MEMORY_GB_HOUR_PRICE', '0.005'))
    gpu_hour_price: float = float(os.getenv('GPU_HOUR_PRICE', '2.0'))
    storage_gb_month_price: float = float(os.getenv('STORAGE_GB_MONTH_PRICE', '0.05'))


@dataclass
class Config:
    """主配置类"""
    # 环境
    environment: str = field(default_factory=lambda: os.getenv('ENVIRONMENT', 'development'))
    debug: bool = field(default_factory=lambda: os.getenv('DEBUG', 'false').lower() == 'true')
    testing: bool = field(default_factory=lambda: os.getenv('TESTING', 'false').lower() == 'true')
    
    # 应用信息
    app_name: str = field(default_factory=lambda: os.getenv('APP_NAME', 'VectorSphere Intelligent Platform'))
    app_version: str = field(default_factory=lambda: os.getenv('APP_VERSION', '1.0.0'))
    
    # 密钥
    secret_key: str = field(default_factory=lambda: os.getenv('SECRET_KEY', 'your-secret-key-change-in-production'))
    
    # 子配置
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    jwt: JWTConfig = field(default_factory=JWTConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    tenant: TenantConfig = field(default_factory=TenantConfig)
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建必要的目录
        self._create_directories()
        
        # 验证配置
        self._validate_config()
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.storage.local_storage_path,
            self.training.model_storage_path,
            os.path.dirname(self.logging.file_path),
            self.email.template_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _validate_config(self):
        """验证配置"""
        # 验证必需的配置项
        if self.environment == 'production':
            if self.secret_key == 'your-secret-key-change-in-production':
                raise ValueError("生产环境必须设置SECRET_KEY")
            
            if self.jwt.secret_key == 'your-secret-key-change-in-production':
                raise ValueError("生产环境必须设置JWT_SECRET_KEY")
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """从YAML文件加载配置
        
        支持配置文件包含功能，通过 'includes' 字段可以包含其他配置文件
        """
        config_data = cls._load_yaml_with_includes(config_path)
        
        # 将YAML配置转换为环境变量
        def set_env_from_dict(data: Dict[str, Any], prefix: str = ''):
            for key, value in data.items():
                # 跳过非字符串键
                if not isinstance(key, str):
                    continue
                
                # 特殊处理已知的配置映射
                if prefix == '' and key == 'database':
                    # database配置映射到DB_前缀
                    if isinstance(value, dict):
                        set_env_from_dict(value, 'DB_')
                    continue
                elif prefix == '' and key in ['redis', 'jwt', 'security', 'email', 'storage', 'billing']:
                    # 其他子配置保持原有逻辑
                    if isinstance(value, dict):
                        set_env_from_dict(value, f"{key.upper()}_")
                    continue
                    
                env_key = f"{prefix}{key.upper()}" if prefix else key.upper()
                if isinstance(value, dict):
                    set_env_from_dict(value, f"{env_key}_")
                elif value is not None:  # 跳过None值
                    os.environ[env_key] = str(value)
        
        set_env_from_dict(config_data)
        return cls()
    
    @classmethod
    def _load_yaml_with_includes(cls, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件，支持includes功能"""
        config_dir = Path(config_path).parent
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
        
        # 处理includes字段
        includes = config_data.pop('includes', [])
        if includes:
            for include_file in includes:
                include_path = config_dir / include_file
                if include_path.exists():
                    try:
                        with open(include_path, 'r', encoding='utf-8') as f:
                            include_data = yaml.safe_load(f) or {}
                        # 递归合并配置，主配置文件优先级更高
                        config_data = cls._merge_configs(include_data, config_data)
                        logger.info(f"Included configuration from: {include_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load included config {include_path}: {e}")
                else:
                    logger.warning(f"Included config file not found: {include_path}")
        
        return config_data
    
    @classmethod
    def _merge_configs(cls, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置字典，override_config优先级更高"""
        result = base_config.copy()
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(item) for item in obj]
            else:
                return obj
        
        return dataclass_to_dict(self)
    
    def get_database_url(self) -> str:
        """获取数据库连接URL"""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """获取Redis连接URL"""
        return self.redis.url
    
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.environment == 'development'
    
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.environment == 'production'
    
    def is_testing(self) -> bool:
        """是否为测试环境"""
        return self.testing or self.environment == 'testing'


# 全局配置实例
_config = None


def get_config() -> Config:
    """获取配置实例
    
    配置优先级（从高到低）：
    1. 环境变量 CONFIG_FILE 指定的文件
    2. /config/config.yaml（项目根目录同级）
    3. backend/config 默认配置
    4. 环境变量
    """
    global _config
    if _config is None:
        config_loaded = False
        
        # 优先级1: 环境变量指定的配置文件
        config_file = os.getenv('CONFIG_FILE')
        if config_file and os.path.exists(config_file):
            try:
                _config = Config.from_yaml(config_file)
                config_loaded = True
                logger.info(f"Loaded configuration from environment specified file: {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config from {config_file}: {e}")
        
        # 优先级2: /config/config.yaml（项目根目录同级）
        if not config_loaded:
            project_root = Path(__file__).parent.parent.parent
            main_config_file = project_root / "config" / "config.yaml"
            if main_config_file.exists():
                try:
                    _config = Config.from_yaml(str(main_config_file))
                    config_loaded = True
                    logger.info(f"Loaded configuration from: {main_config_file}")
                except Exception as e:
                    logger.warning(f"Failed to load config from {main_config_file}: {e}")
        
        # 优先级3: 默认配置（backend/config + 环境变量）
        if not config_loaded:
            _config = Config()
            logger.info("Using default configuration with environment variables")
            
    return _config


def reload_config():
    """重新加载配置"""
    global _config
    _config = None
    return get_config()


# 便捷函数
def get_database_url() -> str:
    """获取数据库连接URL"""
    return get_config().get_database_url()


def get_redis_url() -> str:
    """获取Redis连接URL"""
    return get_config().get_redis_url()


def is_development() -> bool:
    """是否为开发环境"""
    return get_config().is_development()


def is_production() -> bool:
    """是否为生产环境"""
    return get_config().is_production()


def is_testing() -> bool:
    """是否为测试环境"""
    return get_config().is_testing()
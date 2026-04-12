"""配置验证模块

提供系统配置的验证和安全检查功能。
"""
import os
import re
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConfigLevel(Enum):
    """配置级别"""
    REQUIRED = "required"
    OPTIONAL = "optional"
    DEPRECATED = "deprecated"


class ConfigType(Enum):
    """配置类型"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    URL = "url"
    EMAIL = "email"
    PATH = "path"
    SECRET = "secret"


@dataclass
class ConfigRule:
    """配置规则"""
    name: str
    config_type: ConfigType
    level: ConfigLevel
    default: Any = None
    description: str = ""
    pattern: Optional[str] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    validator: Optional[callable] = None


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.rules = {}
        self.errors = []
        self.warnings = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """设置默认配置规则"""
        default_rules = [
            # 数据库配置
            ConfigRule(
                "DATABASE_URL",
                ConfigType.URL,
                ConfigLevel.REQUIRED,
                description="数据库连接URL"
            ),
            ConfigRule(
                "DATABASE_POOL_SIZE",
                ConfigType.INTEGER,
                ConfigLevel.OPTIONAL,
                default=10,
                min_value=1,
                max_value=100,
                description="数据库连接池大小"
            ),
            
            # Redis配置
            ConfigRule(
                "REDIS_URL",
                ConfigType.URL,
                ConfigLevel.OPTIONAL,
                description="Redis连接URL"
            ),
            
            # 安全配置
            ConfigRule(
                "SECRET_KEY",
                ConfigType.SECRET,
                ConfigLevel.REQUIRED,
                description="应用密钥"
            ),
            ConfigRule(
                "JWT_SECRET_KEY",
                ConfigType.SECRET,
                ConfigLevel.REQUIRED,
                description="JWT密钥"
            ),
            ConfigRule(
                "JWT_ACCESS_TOKEN_EXPIRES",
                ConfigType.INTEGER,
                ConfigLevel.OPTIONAL,
                default=3600,
                min_value=300,
                max_value=86400,
                description="JWT访问令牌过期时间（秒）"
            ),
            
            # 服务器配置
            ConfigRule(
                "HOST",
                ConfigType.STRING,
                ConfigLevel.OPTIONAL,
                default="0.0.0.0",
                description="服务器主机地址"
            ),
            ConfigRule(
                "PORT",
                ConfigType.INTEGER,
                ConfigLevel.OPTIONAL,
                default=5000,
                min_value=1,
                max_value=65535,
                description="服务器端口"
            ),
            ConfigRule(
                "DEBUG",
                ConfigType.BOOLEAN,
                ConfigLevel.OPTIONAL,
                default=False,
                description="调试模式"
            ),
            
            # 日志配置
            ConfigRule(
                "LOG_LEVEL",
                ConfigType.STRING,
                ConfigLevel.OPTIONAL,
                default="INFO",
                allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                description="日志级别"
            ),
            ConfigRule(
                "LOG_FILE",
                ConfigType.PATH,
                ConfigLevel.OPTIONAL,
                description="日志文件路径"
            ),
            
            # GPU配置
            ConfigRule(
                "CUDA_VISIBLE_DEVICES",
                ConfigType.STRING,
                ConfigLevel.OPTIONAL,
                description="可见的CUDA设备"
            ),
            ConfigRule(
                "GPU_MEMORY_FRACTION",
                ConfigType.FLOAT,
                ConfigLevel.OPTIONAL,
                default=0.8,
                min_value=0.1,
                max_value=1.0,
                description="GPU内存使用比例"
            ),
            
            # 训练配置
            ConfigRule(
                "MAX_TRAINING_SESSIONS",
                ConfigType.INTEGER,
                ConfigLevel.OPTIONAL,
                default=10,
                min_value=1,
                max_value=100,
                description="最大并发训练会话数"
            ),
            ConfigRule(
                "CHECKPOINT_DIR",
                ConfigType.PATH,
                ConfigLevel.OPTIONAL,
                default="/tmp/checkpoints",
                description="检查点目录"
            ),
            
            # 监控配置
            ConfigRule(
                "ENABLE_METRICS",
                ConfigType.BOOLEAN,
                ConfigLevel.OPTIONAL,
                default=True,
                description="启用指标收集"
            ),
            ConfigRule(
                "METRICS_PORT",
                ConfigType.INTEGER,
                ConfigLevel.OPTIONAL,
                default=9090,
                min_value=1,
                max_value=65535,
                description="指标服务端口"
            ),
            
            # API配置
            ConfigRule(
                "API_RATE_LIMIT",
                ConfigType.STRING,
                ConfigLevel.OPTIONAL,
                default="100/hour",
                pattern=r"^\d+/(second|minute|hour|day)$",
                description="API速率限制"
            ),
            ConfigRule(
                "MAX_REQUEST_SIZE",
                ConfigType.INTEGER,
                ConfigLevel.OPTIONAL,
                default=100 * 1024 * 1024,  # 100MB
                min_value=1024,
                description="最大请求大小（字节）"
            ),
            
            # 存储配置
            ConfigRule(
                "STORAGE_TYPE",
                ConfigType.STRING,
                ConfigLevel.OPTIONAL,
                default="local",
                allowed_values=["local", "s3", "gcs", "azure"],
                description="存储类型"
            ),
            ConfigRule(
                "STORAGE_PATH",
                ConfigType.PATH,
                ConfigLevel.OPTIONAL,
                default="/tmp/storage",
                description="本地存储路径"
            ),
        ]
        
        for rule in default_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule: ConfigRule):
        """添加配置规则"""
        self.rules[rule.name] = rule
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证配置
        
        Args:
            config: 配置字典
            
        Returns:
            验证结果
        """
        self.errors = []
        self.warnings = []
        validated_config = {}
        
        # 检查所有规则
        for name, rule in self.rules.items():
            value = config.get(name)
            
            # 检查必需配置
            if rule.level == ConfigLevel.REQUIRED and value is None:
                self.errors.append(f"Required config '{name}' is missing")
                continue
            
            # 使用默认值
            if value is None and rule.default is not None:
                value = rule.default
                self.warnings.append(f"Using default value for '{name}': {value}")
            
            # 验证配置值
            if value is not None:
                validated_value = self._validate_value(name, value, rule)
                if validated_value is not None:
                    validated_config[name] = validated_value
        
        # 检查未知配置
        for name in config:
            if name not in self.rules:
                self.warnings.append(f"Unknown config '{name}' found")
                validated_config[name] = config[name]
        
        return {
            'config': validated_config,
            'errors': self.errors,
            'warnings': self.warnings,
            'is_valid': len(self.errors) == 0
        }
    
    def _validate_value(self, name: str, value: Any, rule: ConfigRule) -> Any:
        """验证单个配置值"""
        try:
            # 类型转换和验证
            validated_value = self._convert_type(value, rule.config_type)
            
            # 模式匹配
            if rule.pattern and isinstance(validated_value, str):
                if not re.match(rule.pattern, validated_value):
                    self.errors.append(f"Config '{name}' does not match pattern: {rule.pattern}")
                    return None
            
            # 数值范围检查
            if rule.config_type in [ConfigType.INTEGER, ConfigType.FLOAT]:
                if rule.min_value is not None and validated_value < rule.min_value:
                    self.errors.append(f"Config '{name}' is below minimum value: {rule.min_value}")
                    return None
                if rule.max_value is not None and validated_value > rule.max_value:
                    self.errors.append(f"Config '{name}' is above maximum value: {rule.max_value}")
                    return None
            
            # 允许值检查
            if rule.allowed_values and validated_value not in rule.allowed_values:
                self.errors.append(f"Config '{name}' must be one of: {rule.allowed_values}")
                return None
            
            # 自定义验证器
            if rule.validator:
                if not rule.validator(validated_value):
                    self.errors.append(f"Config '{name}' failed custom validation")
                    return None
            
            # 特殊类型验证
            if rule.config_type == ConfigType.URL:
                if not self._validate_url(validated_value):
                    self.errors.append(f"Config '{name}' is not a valid URL")
                    return None
            
            elif rule.config_type == ConfigType.EMAIL:
                if not self._validate_email(validated_value):
                    self.errors.append(f"Config '{name}' is not a valid email")
                    return None
            
            elif rule.config_type == ConfigType.PATH:
                if not self._validate_path(validated_value):
                    self.warnings.append(f"Config '{name}' path may not be accessible: {validated_value}")
            
            elif rule.config_type == ConfigType.SECRET:
                if not self._validate_secret(validated_value):
                    self.errors.append(f"Config '{name}' is not a secure secret")
                    return None
            
            return validated_value
            
        except Exception as e:
            self.errors.append(f"Error validating config '{name}': {e}")
            return None
    
    def _convert_type(self, value: Any, config_type: ConfigType) -> Any:
        """类型转换"""
        if config_type == ConfigType.STRING:
            return str(value)
        
        elif config_type == ConfigType.INTEGER:
            return int(value)
        
        elif config_type == ConfigType.FLOAT:
            return float(value)
        
        elif config_type == ConfigType.BOOLEAN:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ['true', '1', 'yes', 'on']
            return bool(value)
        
        elif config_type == ConfigType.LIST:
            if isinstance(value, str):
                return [item.strip() for item in value.split(',')]
            return list(value)
        
        elif config_type == ConfigType.DICT:
            if isinstance(value, str):
                import json
                return json.loads(value)
            return dict(value)
        
        else:
            return str(value)
    
    def _validate_url(self, url: str) -> bool:
        """验证URL格式"""
        # 支持HTTP/HTTPS和数据库URL
        url_pattern = re.compile(
            r'^(?:https?|postgresql|mysql|sqlite|redis)://'  # 支持多种协议
            r'(?:[A-Z0-9._%-]+(?::[A-Z0-9._%-]*)?@)?'  # 可选的用户名:密码@
            r'(?:'
            r'(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'  # ...or ip
            r')'
            r'(?::\d+)?'  # optional port
            r'(?:/[^\s]*)?$', re.IGNORECASE)  # path
        return url_pattern.match(url) is not None
    
    def _validate_email(self, email: str) -> bool:
        """验证邮箱格式"""
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        return email_pattern.match(email) is not None
    
    def _validate_path(self, path: str) -> bool:
        """验证路径"""
        try:
            # 检查路径是否可以创建
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return True
        except Exception:
            return False
    
    def _validate_secret(self, secret: str) -> bool:
        """验证密钥强度"""
        if len(secret) < 32:
            return False
        
        # 检查是否包含多种字符类型
        has_upper = any(c.isupper() for c in secret)
        has_lower = any(c.islower() for c in secret)
        has_digit = any(c.isdigit() for c in secret)
        has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in secret)
        
        return sum([has_upper, has_lower, has_digit, has_special]) >= 3
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        summary = {
            'total_rules': len(self.rules),
            'required_rules': len([r for r in self.rules.values() if r.level == ConfigLevel.REQUIRED]),
            'optional_rules': len([r for r in self.rules.values() if r.level == ConfigLevel.OPTIONAL]),
            'deprecated_rules': len([r for r in self.rules.values() if r.level == ConfigLevel.DEPRECATED]),
            'rules_by_type': {}
        }
        
        for config_type in ConfigType:
            count = len([r for r in self.rules.values() if r.config_type == config_type])
            if count > 0:
                summary['rules_by_type'][config_type.value] = count
        
        return summary


def validate_environment_config() -> Dict[str, Any]:
    """验证环境变量配置"""
    validator = ConfigValidator()
    
    # 从环境变量获取配置
    config = {}
    for name in validator.rules.keys():
        value = os.getenv(name)
        if value is not None:
            config[name] = value
    
    return validator.validate_config(config)


def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """从文件加载配置"""
    import json
    import yaml
    
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            elif file_path.endswith(('.yml', '.yaml')):
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_path}")
    
    except Exception as e:
        logger.error(f"Failed to load config from {file_path}: {e}")
        return {}


def validate_config_file(file_path: str) -> Dict[str, Any]:
    """验证配置文件"""
    config = load_config_from_file(file_path)
    validator = ConfigValidator()
    return validator.validate_config(config)


def generate_config_template(output_path: str = "config_template.yaml"):
    """生成配置模板文件"""
    validator = ConfigValidator()
    
    template = {
        "# VectorSphere Configuration Template": None,
        "# Generated automatically - modify as needed": None,
        "": None
    }
    
    # 按类别组织配置
    categories = {
        "Database": ["DATABASE_URL", "DATABASE_POOL_SIZE"],
        "Security": ["SECRET_KEY", "JWT_SECRET_KEY", "JWT_ACCESS_TOKEN_EXPIRES"],
        "Server": ["HOST", "PORT", "DEBUG"],
        "Logging": ["LOG_LEVEL", "LOG_FILE"],
        "GPU": ["CUDA_VISIBLE_DEVICES", "GPU_MEMORY_FRACTION"],
        "Training": ["MAX_TRAINING_SESSIONS", "CHECKPOINT_DIR"],
        "Monitoring": ["ENABLE_METRICS", "METRICS_PORT"],
        "API": ["API_RATE_LIMIT", "MAX_REQUEST_SIZE"],
        "Storage": ["STORAGE_TYPE", "STORAGE_PATH"]
    }
    
    for category, config_names in categories.items():
        template[f"# {category} Configuration"] = None
        for name in config_names:
            if name in validator.rules:
                rule = validator.rules[name]
                template[name] = {
                    "value": rule.default,
                    "description": rule.description,
                    "type": rule.config_type.value,
                    "required": rule.level == ConfigLevel.REQUIRED
                }
        template[""] = None
    
    # 写入文件
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(template, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Configuration template generated: {output_path}")


if __name__ == "__main__":
    # 示例用法
    result = validate_environment_config()
    print(f"Configuration validation result: {result}")
    
    # 生成配置模板
    generate_config_template()
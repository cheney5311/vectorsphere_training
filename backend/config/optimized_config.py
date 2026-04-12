"""优化的配置管理模块 (新架构)

提供更灵活、可扩展的配置管理功能，支持动态配置更新和热重载，适配新架构需求。
"""

import os
import logging
import json
import yaml
from typing import Dict, Any, Optional, List, Union, Type, TypeVar, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import RLock
from datetime import datetime
from enum import Enum

# 配置日志记录器
logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

# 类型变量用于泛型支持
T = TypeVar('T')


class ConfigSource(Enum):
    """配置源枚举"""
    ENVIRONMENT = "environment"
    YAML_FILE = "yaml_file"
    JSON_FILE = "json_file"
    DATABASE = "database"
    DEFAULT = "default"


class ConfigValidationError(Exception):
    """配置验证错误异常"""
    pass


@dataclass
class ConfigMetadata:
    """配置元数据"""
    source: ConfigSource
    loaded_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    description: str = ""
    tags: List[str] = field(default_factory=list)


class ConfigObserver:
    """配置观察者基类"""
    
    def on_config_change(self, config_name: str, old_value: Any, new_value: Any):
        """配置变更回调"""
        pass


class OptimizedConfigManager:
    """优化的配置管理器
    
    提供统一的配置管理接口，支持多种配置源、动态更新和观察者模式。
    """
    
    def __init__(self):
        self._configs: Dict[str, Any] = {}
        self._metadata: Dict[str, ConfigMetadata] = {}
        self._observers: Dict[str, List[ConfigObserver]] = {}
        self._lock = RLock()
        self._config_sources: List[ConfigSource] = [
            ConfigSource.ENVIRONMENT,
            ConfigSource.YAML_FILE,
            ConfigSource.JSON_FILE,
            ConfigSource.DEFAULT
        ]
        
    def register_observer(self, config_name: str, observer: ConfigObserver):
        """注册配置观察者
        
        Args:
            config_name: 配置名称
            observer: 观察者实例
        """
        with self._lock:
            if config_name not in self._observers:
                self._observers[config_name] = []
            self._observers[config_name].append(observer)
            
    def unregister_observer(self, config_name: str, observer: ConfigObserver):
        """注销配置观察者
        
        Args:
            config_name: 配置名称
            observer: 观察者实例
        """
        with self._lock:
            if config_name in self._observers and observer in self._observers[config_name]:
                self._observers[config_name].remove(observer)
                
    def _notify_observers(self, config_name: str, old_value: Any, new_value: Any):
        """通知观察者配置变更"""
        if config_name in self._observers:
            for observer in self._observers[config_name]:
                try:
                    observer.on_config_change(config_name, old_value, new_value)
                except Exception as e:
                    logger.error(f"配置观察者通知失败: {e}")
                    
    def set_config(self, name: str, value: Any, source: ConfigSource = ConfigSource.DEFAULT):
        """设置配置值
        
        Args:
            name: 配置名称
            value: 配置值
            source: 配置源
        """
        with self._lock:
            old_value = self._configs.get(name)
            self._configs[name] = value
            self._metadata[name] = ConfigMetadata(source=source)
            
            # 通知观察者
            if old_value != value:
                self._notify_observers(name, old_value, value)
                
    def get_config(self, name: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            name: 配置名称
            default: 默认值
            
        Returns:
            配置值
        """
        with self._lock:
            return self._configs.get(name, default)
            
    def get_config_with_metadata(self, name: str, default: Any = None) -> tuple:
        """获取配置值和元数据
        
        Args:
            name: 配置名称
            default: 默认值
            
        Returns:
            (配置值, 元数据)
        """
        with self._lock:
            value = self._configs.get(name, default)
            metadata = self._metadata.get(name, ConfigMetadata(source=ConfigSource.DEFAULT))
            return value, metadata
            
    def load_from_environment(self, prefix: str = "") -> Dict[str, Any]:
        """从环境变量加载配置
        
        Args:
            prefix: 环境变量前缀
            
        Returns:
            加载的配置字典
        """
        loaded_configs = {}
        
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix):
                continue
                
            config_name = key[len(prefix):] if prefix else key
            # 尝试转换类型
            converted_value = self._convert_value(value)
            self.set_config(config_name, converted_value, ConfigSource.ENVIRONMENT)
            loaded_configs[config_name] = converted_value
            
        logger.info(f"从环境变量加载了 {len(loaded_configs)} 个配置项")
        return loaded_configs
        
    def load_from_yaml(self, file_path: str) -> Dict[str, Any]:
        """从YAML文件加载配置
        
        Args:
            file_path: YAML文件路径
            
        Returns:
            加载的配置字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
                
            loaded_configs = self._flatten_dict(config_data)
            
            for key, value in loaded_configs.items():
                self.set_config(key, value, ConfigSource.YAML_FILE)
                
            logger.info(f"从YAML文件 {file_path} 加载了 {len(loaded_configs)} 个配置项")
            return loaded_configs
            
        except Exception as e:
            logger.error(f"从YAML文件加载配置失败: {e}")
            raise
            
    def load_from_json(self, file_path: str) -> Dict[str, Any]:
        """从JSON文件加载配置
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            加载的配置字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            loaded_configs = self._flatten_dict(config_data)
            
            for key, value in loaded_configs.items():
                self.set_config(key, value, ConfigSource.JSON_FILE)
                
            logger.info(f"从JSON文件 {file_path} 加载了 {len(loaded_configs)} 个配置项")
            return loaded_configs
            
        except Exception as e:
            logger.error(f"从JSON文件加载配置失败: {e}")
            raise
            
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """将嵌套字典扁平化
        
        Args:
            d: 嵌套字典
            parent_key: 父键
            sep: 分隔符
            
        Returns:
            扁平化字典
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
        
    def _convert_value(self, value: str) -> Union[str, int, float, bool, list, dict]:
        """尝试转换字符串值为合适的数据类型
        
        Args:
            value: 字符串值
            
        Returns:
            转换后的值
        """
        # 布尔值
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
            
        # 整数
        try:
            if '.' not in value and 'e' not in value.lower():
                return int(value)
        except ValueError:
            pass
            
        # 浮点数
        try:
            return float(value)
        except ValueError:
            pass
            
        # JSON对象或数组
        if value.startswith(('{', '[')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
                
        # 逗号分隔的列表
        if ',' in value and not value.startswith(('{', '[')):
            return [item.strip() for item in value.split(',')]
            
        # 默认返回字符串
        return value
        
    def validate_config(self, name: str, validator: Callable[[Any], bool], error_message: str = ""):
        """验证配置值
        
        Args:
            name: 配置名称
            validator: 验证函数
            error_message: 错误消息
            
        Raises:
            ConfigValidationError: 验证失败时抛出
        """
        value = self.get_config(name)
        if not validator(value):
            raise ConfigValidationError(error_message or f"配置 {name} 验证失败")
            
    def get_all_configs(self) -> Dict[str, Any]:
        """获取所有配置
        
        Returns:
            所有配置的字典
        """
        with self._lock:
            return self._configs.copy()
            
    def get_config_names(self) -> List[str]:
        """获取所有配置名称
        
        Returns:
            配置名称列表
        """
        with self._lock:
            return list(self._configs.keys())
            
    def export_to_dict(self) -> Dict[str, Any]:
        """导出配置为字典
        
        Returns:
            配置字典
        """
        with self._lock:
            return {
                'configs': self._configs.copy(),
                'metadata': {k: asdict(v) for k, v in self._metadata.items()},
                'exported_at': datetime.now().isoformat()
            }
            
    def export_to_json(self, file_path: str):
        """导出配置为JSON文件
        
        Args:
            file_path: 导出文件路径
        """
        export_data = self.export_to_dict()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        logger.info(f"配置已导出到 {file_path}")
        
    def reload_config(self, name: str, sources: Optional[List[ConfigSource]] = None):
        """重新加载指定配置
        
        Args:
            name: 配置名称
            sources: 配置源列表，None表示使用默认源顺序
        """
        if sources is None:
            sources = self._config_sources
            
        # 按优先级顺序尝试加载
        for source in sources:
            try:
                if source == ConfigSource.ENVIRONMENT:
                    env_key = name.upper()
                    if env_key in os.environ:
                        value = self._convert_value(os.environ[env_key])
                        self.set_config(name, value, source)
                        return
                        
                # 其他源需要预先知道文件路径等信息，这里简化处理
                # 实际应用中可能需要更复杂的逻辑
                
            except Exception as e:
                logger.warning(f"从 {source.value} 重新加载配置 {name} 失败: {e}")
                continue
                
    def batch_update(self, configs: Dict[str, Any], source: ConfigSource = ConfigSource.DEFAULT):
        """批量更新配置
        
        Args:
            configs: 配置字典
            source: 配置源
        """
        with self._lock:
            for name, value in configs.items():
                self.set_config(name, value, source)
        logger.info(f"批量更新了 {len(configs)} 个配置项")


# 全局配置管理器实例
_global_config_manager = OptimizedConfigManager()


def get_config_manager() -> OptimizedConfigManager:
    """获取全局配置管理器实例
    
    Returns:
        OptimizedConfigManager: 配置管理器实例
    """
    return _global_config_manager


@dataclass
class DynamicTrainingConfig:
    """动态训练配置类
    
    支持运行时动态调整的训练配置。
    """
    
    # 基础配置
    model_name: str = "default_model"
    output_dir: str = "./outputs"
    experiment_name: str = ""
    
    # 训练参数
    batch_size: int = 64
    learning_rate: float = 0.0001
    num_epochs: int = 50
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    
    # 优化器配置
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # 调度器配置
    scheduler: str = "cosine_with_warmup"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # 检查点配置
    save_steps: int = 500
    save_total_limit: int = 10
    checkpoint_dir: str = "./checkpoints"
    
    # 早停配置
    early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 1e-4
    
    # 混合精度
    use_fp16: bool = True
    fp16_opt_level: str = "O1"
    
    # 分布式训练
    use_distributed: bool = True
    world_size: int = 1
    master_port: str = "12355"
    
    # 设备配置
    device: str = "auto"  # auto, cpu, cuda, cuda:0, cuda:1, etc.
    
    # 数据配置
    max_seq_length: int = 512
    train_data_path: str = ""
    val_data_path: str = ""
    test_data_path: str = ""
    
    # 监控配置
    logging_steps: int = 100
    eval_steps: int = 500
    log_level: str = "INFO"
    
    # 高级配置
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    enable_wandb: bool = True
    wandb_project: str = "vectorsphere-training"
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.experiment_name:
            self.experiment_name = f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
    @classmethod
    def from_config_manager(cls, config_manager: OptimizedConfigManager = None) -> 'DynamicTrainingConfig':
        """从配置管理器创建实例
        
        Args:
            config_manager: 配置管理器实例
            
        Returns:
            DynamicTrainingConfig: 训练配置实例
        """
        if config_manager is None:
            config_manager = get_config_manager()
            
        # 获取所有训练相关配置
        config_dict = {}
        prefix = "TRAINING_"
        
        for name in config_manager.get_config_names():
            if name.startswith(prefix):
                key = name[len(prefix):].lower()
                config_dict[key] = config_manager.get_config(name)
                
        return cls(**config_dict)
        
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """从字典更新配置
        
        Args:
            config_dict: 配置字典
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
                
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            配置字典
        """
        return asdict(self)
        
    def validate(self):
        """验证配置
        
        Raises:
            ConfigValidationError: 验证失败时抛出
        """
        if self.batch_size <= 0:
            raise ConfigValidationError("batch_size 必须大于0")
        if self.learning_rate <= 0:
            raise ConfigValidationError("learning_rate 必须大于0")
        if self.num_epochs <= 0:
            raise ConfigValidationError("num_epochs 必须大于0")
        if self.max_seq_length <= 0:
            raise ConfigValidationError("max_seq_length 必须大于0")
            
    def get_optimizer_config(self) -> Dict[str, Any]:
        """获取优化器配置
        
        Returns:
            优化器配置字典
        """
        return {
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_grad_norm': self.max_grad_norm
        }
        
    def get_scheduler_config(self) -> Dict[str, Any]:
        """获取调度器配置
        
        Returns:
            调度器配置字典
        """
        config = {
            'scheduler': self.scheduler,
            'warmup_steps': self.warmup_steps,
        }
        config.update(self.scheduler_params)
        return config


# 配置观察者示例
class TrainingConfigObserver(ConfigObserver):
    """训练配置观察者"""
    
    def __init__(self, callback: Callable[[str, Any, Any], None]):
        self.callback = callback
        
    def on_config_change(self, config_name: str, old_value: Any, new_value: Any):
        """配置变更回调"""
        try:
            self.callback(config_name, old_value, new_value)
        except Exception as e:
            logger.error(f"训练配置观察者回调失败: {e}")


# 便捷函数
def load_training_config(config_path: Optional[str] = None) -> DynamicTrainingConfig:
    """加载训练配置
    
    Args:
        config_path: 配置文件路径，None表示使用默认配置
        
    Returns:
        DynamicTrainingConfig: 训练配置实例
    """
    config_manager = get_config_manager()
    
    # 从环境变量加载
    config_manager.load_from_environment("TRAINING_")
    
    # 从文件加载（如果指定了路径）
    if config_path:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config_manager.load_from_yaml(config_path)
        elif config_path.endswith('.json'):
            config_manager.load_from_json(config_path)
            
    # 创建训练配置实例
    return DynamicTrainingConfig.from_config_manager(config_manager)


def update_training_config(config: DynamicTrainingConfig, updates: Dict[str, Any]):
    """更新训练配置
    
    Args:
        config: 训练配置实例
        updates: 更新字典
    """
    config_manager = get_config_manager()
    config_manager.batch_update({f"TRAINING_{k.upper()}": v for k, v in updates.items()})
    config.update_from_dict(updates)
"""优化配置管理器

管理资源优化配置的加载、更新和获取。
"""

import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from .models import OptimizationConfig

logger = logging.getLogger(__name__)

# 全局配置实例
_global_config: Optional[OptimizationConfig] = None


class ConfigManager:
    """配置管理器"""

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """初始化配置管理器

        Args:
            config: 优化配置对象
        """
        self.config = config or OptimizationConfig()

    def load_from_file(self, file_path: str) -> bool:
        """从文件加载配置

        Args:
            file_path: 配置文件路径

        Returns:
            是否加载成功
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"Config file not found: {file_path}")
                return False

            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 更新配置
            self._update_config_from_dict(self.config, data)
            logger.info(f"Configuration loaded from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            return False

    def save_to_file(self, file_path: str) -> bool:
        """保存配置到文件

        Args:
            file_path: 配置文件路径

        Returns:
            是否保存成功
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # 转换配置为字典
            config_dict = self._config_to_dict(self.config)

            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=2)

            logger.info(f"Configuration saved to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config to {file_path}: {e}")
            return False

    def get_config(self) -> OptimizationConfig:
        """获取当前配置

        Returns:
            优化配置对象
        """
        return self.config

    def update_config(self, new_config: OptimizationConfig) -> bool:
        """更新配置

        Args:
            new_config: 新的配置对象

        Returns:
            是否更新成功
        """
        try:
            self.config = new_config
            logger.info("Configuration updated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False

    def _update_config_from_dict(self, config: OptimizationConfig, data: Dict[str, Any]):
        """从字典更新配置对象

        Args:
            config: 配置对象
            data: 配置数据字典
        """
        # 这里可以实现更复杂的配置更新逻辑
        # 为简化起见，我们只更新简单的属性
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def _config_to_dict(self, config: OptimizationConfig) -> Dict[str, Any]:
        """将配置对象转换为字典

        Args:
            config: 配置对象

        Returns:
            配置字典
        """
        # 这里可以实现更复杂的配置序列化逻辑
        # 为简化起见，我们只序列化简单的属性
        result = {}
        for key, value in config.__dict__.items():
            if not key.startswith('_'):
                result[key] = value
        return result


def get_optimization_config() -> OptimizationConfig:
    """获取全局优化配置

    Returns:
        优化配置对象
    """
    global _global_config
    if _global_config is None:
        _global_config = OptimizationConfig()
    return _global_config


def set_optimization_config(config: OptimizationConfig):
    """设置全局优化配置

    Args:
        config: 优化配置对象
    """
    global _global_config
    _global_config = config


def load_config_from_file(file_path: str) -> bool:
    """从文件加载全局配置

    Args:
        file_path: 配置文件路径

    Returns:
        是否加载成功
    """
    config_manager = ConfigManager(get_optimization_config())
    return config_manager.load_from_file(file_path)


def save_config_to_file(file_path: str) -> bool:
    """保存全局配置到文件

    Args:
        file_path: 配置文件路径

    Returns:
        是否保存成功
    """
    config_manager = ConfigManager(get_optimization_config())
    return config_manager.save_to_file(file_path)
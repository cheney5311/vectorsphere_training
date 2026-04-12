"""配置管理模块

提供统一的配置管理功能，支持多种配置源。
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from enum import Enum

from .exceptions import SystemError


class ConfigSource(Enum):
    """配置源枚举"""
    ENVIRONMENT = "environment"
    YAML_FILE = "yaml_file"
    JSON_FILE = "json_file"
    DEFAULT = "default"


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._configs: Dict[str, Any] = {}
        self._sources: Dict[str, ConfigSource] = {}
        
    def load_from_environment(self, prefix: str = "") -> None:
        """从环境变量加载配置
        
        Args:
            prefix: 环境变量前缀
        """
        for key, value in os.environ.items():
            if prefix and not key.startswith(prefix):
                continue
                
            config_name = key[len(prefix):] if prefix else key
            # 尝试转换类型
            converted_value = self._convert_value(value)
            self._configs[config_name] = converted_value
            self._sources[config_name] = ConfigSource.ENVIRONMENT
            
    def load_from_yaml(self, file_path: str) -> None:
        """从YAML文件加载配置
        
        Args:
            file_path: YAML文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
                
            flattened_configs = self._flatten_dict(config_data)
            
            for key, value in flattened_configs.items():
                self._configs[key] = value
                self._sources[key] = ConfigSource.YAML_FILE
                
        except Exception as e:
            raise SystemError(f"从YAML文件加载配置失败: {e}", component="config")
            
    def load_from_json(self, file_path: str) -> None:
        """从JSON文件加载配置
        
        Args:
            file_path: JSON文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            flattened_configs = self._flatten_dict(config_data)
            
            for key, value in flattened_configs.items():
                self._configs[key] = value
                self._sources[key] = ConfigSource.JSON_FILE
                
        except Exception as e:
            raise SystemError(f"从JSON文件加载配置失败: {e}", component="config")
            
    def set_default(self, name: str, value: Any) -> None:
        """设置默认配置值
        
        Args:
            name: 配置名称
            value: 配置值
        """
        if name not in self._configs:
            self._configs[name] = value
            self._sources[name] = ConfigSource.DEFAULT
            
    def get(self, name: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            name: 配置名称
            default: 默认值
            
        Returns:
            配置值
        """
        return self._configs.get(name, default)
        
    def get_source(self, name: str) -> Optional[ConfigSource]:
        """获取配置源
        
        Args:
            name: 配置名称
            
        Returns:
            配置源
        """
        return self._sources.get(name)
        
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
        if ',' in value:
            return [item.strip() for item in value.split(',')]
            
        # 默认返回字符串
        return value


# 全局配置管理器实例
_global_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例
    
    Returns:
        ConfigManager: 配置管理器实例
    """
    return _global_config_manager


def load_config(config_paths: Optional[Dict[str, str]] = None) -> None:
    """加载配置
    
    Args:
        config_paths: 配置文件路径字典，格式为 {"yaml": "path", "json": "path"}
    """
    config_manager = get_config_manager()
    
    # 从环境变量加载
    config_manager.load_from_environment()
    
    # 从配置文件加载
    if config_paths:
        if "yaml" in config_paths:
            config_manager.load_from_yaml(config_paths["yaml"])
        if "json" in config_paths:
            config_manager.load_from_json(config_paths["json"])
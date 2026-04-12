"""配置管理器

提供配置文件的加载和管理功能。
"""

import os
import json
import yaml
from typing import Dict, Any, Optional


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config_data: Dict[str, Any] = {}
        
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> bool:
        """加载配置文件
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            加载是否成功
        """
        try:
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"配置文件不存在: {config_file}")
            
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.endswith('.json'):
                    self.config_data = json.load(f)
                elif config_file.endswith(('.yml', '.yaml')):
                    self.config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_file}")
            
            self.config_file = config_file
            return True
            
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键（支持点号分隔的嵌套键）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值
        
        Args:
            key: 配置键（支持点号分隔的嵌套键）
            value: 配置值
        """
        keys = key.split('.')
        config = self.config_data
        
        # 导航到最后一层
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
    
    def save_config(self, config_file: Optional[str] = None) -> bool:
        """保存配置到文件
        
        Args:
            config_file: 配置文件路径（可选）
            
        Returns:
            保存是否成功
        """
        try:
            file_path = config_file or self.config_file
            if not file_path:
                raise ValueError("未指定配置文件路径")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)
                elif file_path.endswith(('.yml', '.yaml')):
                    yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
                else:
                    raise ValueError(f"不支持的配置文件格式: {file_path}")
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"保存配置文件失败: {e}")
    
    def update(self, updates: Dict[str, Any]) -> None:
        """批量更新配置
        
        Args:
            updates: 更新的配置字典
        """
        for key, value in updates.items():
            self.set(key, value)


# 全局配置管理器实例
_global_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    return _global_config_manager


def load_config(config_file: str) -> ConfigManager:
    """加载配置文件并返回配置管理器实例
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        配置管理器实例
    """
    manager = ConfigManager()
    manager.load_config(config_file)
    return manager
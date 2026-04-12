"""训练配置管理

提供训练配置的加载和管理功能。
"""

import sys
import os
import json
from typing import Dict, Any, Optional
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))


class TrainingConfigManager:
    """训练配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self.config = {}
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # 如果配置文件不存在，使用默认配置
            self.config = self.get_default_config()
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {e}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "max_concurrent_jobs": 3,
            "default_output_dir": "./outputs",
            "default_device": "auto",
            "enable_wandb": False,
            "scheduler": {
                "check_interval": 1  # 秒
            },
            "training": {
                "pretrain": {
                    "enabled": True,
                    "num_epochs": 1,
                    "batch_size": 8,
                    "learning_rate": 1e-4
                },
                "finetune": {
                    "enabled": True,
                    "num_epochs": 3,
                    "batch_size": 16,
                    "learning_rate": 2e-5
                },
                "preference": {
                    "enabled": True,
                    "num_epochs": 1,
                    "batch_size": 8,
                    "learning_rate": 1e-5
                }
            }
        }
    
    def get_config(self, key: str, default=None) -> Any:
        """获取配置项"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_config(self, key: str, value: Any):
        """设置配置项"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save_config(self, config_file: Optional[str] = None):
        """保存配置到文件"""
        save_file = config_file or self.config_file
        if not save_file:
            raise ValueError("未指定配置文件路径")
        
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)


# 全局配置管理器实例
_global_config_manager = None


def get_training_config_manager() -> TrainingConfigManager:
    """获取全局训练配置管理器实例"""
    global _global_config_manager
    if _global_config_manager is None:
        # 尝试加载默认配置文件
        default_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config",
            "training_config_optimized.yaml"
        )
        _global_config_manager = TrainingConfigManager(default_config_path)
    return _global_config_manager
"""命令行基础命令类

所有CLI命令的基类，定义了命令的基本结构和接口。
"""

import argparse
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseCommand(ABC):
    """所有CLI命令的基类"""
    
    def __init__(self):
        self.name = self.get_command_name()
        self.help = self.get_command_help()
        
    @abstractmethod
    def get_command_name(self) -> str:
        """返回命令名称"""
        pass
    
    @abstractmethod
    def get_command_help(self) -> str:
        """返回命令帮助描述"""
        pass
    
    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """向解析器添加命令特定的参数"""
        pass
    
    @abstractmethod
    def execute(self, args: argparse.Namespace) -> int:
        """执行命令。返回0表示成功，非0表示失败"""
        pass
    
    def validate_args(self, args: argparse.Namespace) -> bool:
        """验证命令参数。如有需要可重写"""
        return True
    
    def setup(self, args: argparse.Namespace) -> bool:
        """执行前的设置。如有需要可重写"""
        return True
    
    def cleanup(self, args: argparse.Namespace) -> None:
        """执行后的清理。如有需要可重写"""
        pass
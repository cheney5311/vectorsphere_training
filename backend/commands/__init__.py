"""命令行接口命令模块

为VectorSphere智能平台提供命令行接口支持。
"""

from .base_command import BaseCommand
from .serve_command import ServeCommand
from .info_command import InfoCommand
from .train_command import TrainCommand
from .test_command import TestCommand
from .command_manager import CommandManager

__all__ = [
    'BaseCommand',
    'ServeCommand', 
    'InfoCommand',
    'TrainCommand',
    'TestCommand',
    'CommandManager'
]
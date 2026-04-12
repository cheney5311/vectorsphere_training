#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VectorSphere 智能训练平台测试脚本
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.commands.command_manager import CommandManager


def test_commands():
    """测试命令"""
    # 创建命令管理器
    manager = CommandManager("VectorSphere 智能训练平台")
    
    # 测试帮助命令
    print("=== 测试帮助命令 ===")
    manager.run(['--help'])
    
    # 测试测试命令
    print("\n=== 测试测试命令 ===")
    manager.run(['test', '--help'])
    
    # 测试训练命令
    print("\n=== 测试训练命令 ===")
    manager.run(['train', '--help'])


if __name__ == '__main__':
    test_commands()
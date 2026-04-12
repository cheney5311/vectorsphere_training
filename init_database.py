#!/usr/bin/env python3
"""数据库初始化脚本

用于初始化数据库表结构。
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.modules.database.manager import get_database_manager


def init_database():
    """初始化数据库"""
    try:
        # 获取数据库管理器
        db_manager = get_database_manager()
        
        # 创建表
        db_manager.create_tables()
        
        print("数据库初始化成功！")
        return True
        
    except Exception as e:
        print(f"数据库初始化失败: {e}")
        return False


if __name__ == '__main__':
    init_database()
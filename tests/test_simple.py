"""简化测试文件

用于测试backend_new的基本功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试导入功能"""
    try:
        from backend_new.core.exceptions import ValidationError, BusinessLogicError
        print("✓ core.exceptions 导入成功")
    except Exception as e:
        print(f"✗ core.exceptions 导入失败: {e}")
    
    try:
        from backend_new.utils.response import success_response, error_response
        print("✓ utils.response 导入成功")
    except Exception as e:
        print(f"✗ utils.response 导入失败: {e}")
    
    try:
        from backend_new.modules.training.scenarios import get_scenario_manager
        print("✓ modules.training.scenarios 导入成功")
    except Exception as e:
        print(f"✗ modules.training.scenarios 导入失败: {e}")

if __name__ == '__main__':
    test_imports()
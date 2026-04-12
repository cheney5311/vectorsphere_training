#!/usr/bin/env python3
"""
简单测试关闭处理器注册情况
"""

import sys
import os

# 添加项目路径
sys.path.append('/root/seetaSearch/VectorSphere-intelligent-platform')

def test_shutdown_handlers():
    """测试关闭处理器是否正确注册"""
    try:
        from backend.utils.graceful_shutdown import get_shutdown_manager
        
        shutdown_manager = get_shutdown_manager()
        if shutdown_manager:
            handlers = getattr(shutdown_manager, 'shutdown_handlers', [])
            print(f"✅ 已注册 {len(handlers)} 个关闭处理器")
            
            for i, (handler, name) in enumerate(handlers):
                handler_name = getattr(handler, '__name__', name)
                print(f"  {i+1}. {handler_name} ({name})")
            
            return len(handlers) > 0
        else:
            print("❌ 未找到优雅退出管理器")
            return False
            
    except Exception as e:
        print(f"❌ 检查关闭处理器失败: {e}")
        return False

if __name__ == '__main__':
    print("检查关闭处理器注册情况...")
    success = test_shutdown_handlers()
    
    if success:
        print("\n🎉 关闭处理器注册测试通过！")
    else:
        print("\n⚠️ 关闭处理器注册测试失败")
    
    exit(0 if success else 1)
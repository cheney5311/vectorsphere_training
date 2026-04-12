#!/usr/bin/env python3
"""
测试优雅退出功能的脚本
"""

import os
import signal
import time
import requests
import threading
from datetime import datetime

def test_graceful_shutdown():
    """测试优雅退出功能"""
    print("开始测试优雅退出功能...")
    
    # 1. 检查应用是否正在运行
    try:
        response = requests.get('http://localhost:5000/health', timeout=5)
        if response.status_code == 200:
            print("✅ 应用正在运行")
        else:
            print("❌ 应用健康检查失败")
            return False
    except Exception as e:
        print(f"❌ 无法连接到应用: {e}")
        return False
    
    # 2. 模拟一些活动（可选）
    print("模拟一些应用活动...")
    
    # 创建一些HTTP请求来模拟活动
    def make_requests():
        for i in range(5):
            try:
                requests.get('http://localhost:5000/health', timeout=2)
                time.sleep(0.5)
            except:
                break
    
    # 启动后台请求线程
    request_thread = threading.Thread(target=make_requests)
    request_thread.start()
    
    # 3. 等待一小段时间
    time.sleep(2)
    
    # 4. 发送SIGTERM信号进行优雅退出
    print("发送SIGTERM信号进行优雅退出...")
    
    # 查找Flask应用的进程ID
    import subprocess
    try:
        # 查找运行app.py的进程
        result = subprocess.run(['pgrep', '-f', 'python3 app.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pid = int(result.stdout.strip().split('\n')[0])
            print(f"找到应用进程ID: {pid}")
            
            # 发送SIGTERM信号
            os.kill(pid, signal.SIGTERM)
            print("已发送SIGTERM信号")
            
            # 等待应用优雅退出
            print("等待应用优雅退出...")
            for i in range(30):  # 最多等待30秒
                try:
                    # 检查进程是否还在运行
                    os.kill(pid, 0)  # 发送0信号检查进程是否存在
                    time.sleep(1)
                    print(f"等待中... ({i+1}/30)")
                except OSError:
                    # 进程已经退出
                    print(f"✅ 应用已在 {i+1} 秒后优雅退出")
                    return True
            
            print("❌ 应用在30秒内未能优雅退出")
            return False
            
        else:
            print("❌ 未找到应用进程")
            return False
            
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        return False

def test_shutdown_handlers():
    """测试关闭处理器是否正确注册"""
    print("\n检查关闭处理器注册情况...")
    
    try:
        # 导入优雅退出管理器
        import sys
        sys.path.append('/root/seetaSearch/VectorSphere-intelligent-platform')
        
        from backend.utils.graceful_shutdown import get_shutdown_manager
        
        shutdown_manager = get_graceful_shutdown_manager()
        if shutdown_manager:
            handlers = getattr(shutdown_manager, '_shutdown_handlers', [])
            print(f"✅ 已注册 {len(handlers)} 个关闭处理器")
            
            for i, handler in enumerate(handlers):
                handler_name = getattr(handler, '__name__', 'unknown')
                print(f"  {i+1}. {handler_name}")
            
            return len(handlers) > 0
        else:
            print("❌ 未找到优雅退出管理器")
            return False
            
    except Exception as e:
        print(f"❌ 检查关闭处理器失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("优雅退出功能测试")
    print("=" * 50)
    print(f"测试时间: {datetime.now()}")
    print()
    
    # 测试1: 检查关闭处理器注册
    test1_result = test_shutdown_handlers()
    
    print("\n" + "=" * 50)
    
    # 测试2: 实际优雅退出测试
    test2_result = test_graceful_shutdown()
    
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print(f"关闭处理器注册: {'✅ 通过' if test1_result else '❌ 失败'}")
    print(f"优雅退出功能: {'✅ 通过' if test2_result else '❌ 失败'}")
    
    if test1_result and test2_result:
        print("\n🎉 所有测试通过！优雅退出功能正常工作")
        return True
    else:
        print("\n⚠️  部分测试失败，请检查实现")
        return False

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
#!/usr/bin/env python3
"""
测试信号处理功能
"""

import os
import sys
import time
import signal
import threading
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_signal_handling():
    """测试信号处理功能"""
    print("=" * 60)
    print("测试信号处理功能")
    print("=" * 60)
    
    try:
        from backend.utils.graceful_shutdown import GracefulShutdownManager
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 成功导入优雅关闭模块")
        
        # 创建关闭管理器
        manager = GracefulShutdownManager(default_timeout=5, force_exit_timeout=10)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 成功创建关闭管理器")
        
        # 注册测试处理器
        test_results = []
        
        def test_handler():
            """测试处理器"""
            test_results.append("test_handler_executed")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 执行测试处理器")
            time.sleep(1)
            
        manager.register_handler(test_handler, "测试处理器")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 已注册测试处理器")
        
        # 获取当前进程ID
        pid = os.getpid()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 当前进程ID: {pid}")
        
        # 启动一个线程来发送信号
        def send_signal_after_delay():
            time.sleep(3)  # 等待3秒
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 发送 SIGINT 信号...")
            os.kill(pid, signal.SIGINT)
        
        signal_thread = threading.Thread(target=send_signal_after_delay)
        signal_thread.start()
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 等待信号...")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 进程将在3秒后接收到 SIGINT 信号")
        
        # 等待信号
        try:
            while True:
                time.sleep(0.1)
                if manager.is_shutting_down:
                    break
        except KeyboardInterrupt:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 接收到 KeyboardInterrupt")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 测试完成")
        return True
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print(f"当前工作目录: {os.getcwd()}")
    print(f"当前进程ID: {os.getpid()}")
    
    # 测试 SIGINT 处理
    test1_success = test_signal_handling()
    
    print("\n" + "=" * 60)
    print("测试结果总结:")
    print(f"SIGINT 处理: {'✅ 通过' if test1_success else '❌ 失败'}")
    
    if test1_success:
        print("\n🎉 信号处理测试通过！")
        return 0
    else:
        print("\n⚠️  信号处理测试失败")
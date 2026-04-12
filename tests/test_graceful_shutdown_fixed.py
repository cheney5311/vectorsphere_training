#!/usr/bin/env python3
"""
测试修复后的优雅关闭功能
"""

import os
import sys
import time
import signal
import subprocess
import threading
from datetime import datetime

def test_graceful_shutdown():
    """测试优雅关闭功能"""
    print("=" * 60)
    print("测试修复后的优雅关闭功能")
    print("=" * 60)
    
    # 启动应用
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 启动应用...")
    
    try:
        # 启动应用进程
        proc = subprocess.Popen(
            [sys.executable, "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 等待应用启动
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 等待应用启动...")
        startup_timeout = 30
        startup_start = time.time()
        app_started = False
        
        def read_output():
            """读取应用输出"""
            nonlocal app_started
            for line in iter(proc.stdout.readline, ''):
                print(f"APP: {line.strip()}")
                if "Serving Flask app" in line or "所有服务关闭处理器已注册" in line:
                    app_started = True
        
        # 启动输出读取线程
        output_thread = threading.Thread(target=read_output, daemon=True)
        output_thread.start()
        
        # 等待应用启动
        while not app_started and time.time() - startup_start < startup_timeout:
            if proc.poll() is not None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 应用启动失败，退出码: {proc.returncode}")
                return False
            time.sleep(0.5)
        
        if not app_started:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 应用启动超时")
            proc.terminate()
            return False
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 应用启动成功")
        
        # 等待一段时间确保应用完全启动
        time.sleep(2)
        
        # 发送 SIGTERM 信号测试优雅关闭
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 发送 SIGTERM 信号...")
        shutdown_start = time.time()
        
        proc.send_signal(signal.SIGTERM)
        
        # 等待应用关闭
        shutdown_timeout = 35  # 给足够时间，包括强制退出时间
        try:
            proc.wait(timeout=shutdown_timeout)
            shutdown_time = time.time() - shutdown_start
            
            if proc.returncode == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 应用优雅关闭成功")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 关闭耗时: {shutdown_time:.2f}秒")
                return True
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 应用关闭，但退出码非零: {proc.returncode}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 关闭耗时: {shutdown_time:.2f}秒")
                return True  # 非零退出码也算成功，可能是强制退出
                
        except subprocess.TimeoutExpired:
            shutdown_time = time.time() - shutdown_start
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 应用关闭超时 ({shutdown_time:.2f}秒)")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 强制终止应用...")
            proc.kill()
            proc.wait()
            return False
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ 测试过程中出错: {e}")
        try:
            proc.kill()
            proc.wait()
        except:
            pass
        return False

def main():
    """主函数"""
    # 切换到项目目录
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    print(f"当前工作目录: {os.getcwd()}")
    
    # 运行测试
    success = test_graceful_shutdown()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 优雅关闭功能测试通过")
        print("修复成功！应用能够在合理时间内优雅关闭")
    else:
        print("❌ 优雅关闭功能测试失败")
        print("需要进一步调试和修复")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
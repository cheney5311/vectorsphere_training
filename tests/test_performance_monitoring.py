#!/usr/bin/env python3
"""测试性能监控和错误处理机制"""

import sys
import os
import time
import threading
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 直接复制装饰器定义用于测试
import functools
import time
import logging

# 配置日志
logger = logging.getLogger(__name__)

# 简单的内存缓存实现
_cache = {}
_CACHE_TIMEOUT = 30  # 缓存30秒

def cached(timeout=_CACHE_TIMEOUT):
    """简单的缓存装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 清理过期缓存
            current_time = datetime.now().timestamp()
            expired_keys = []
            for key, (_, timestamp) in _cache.items():
                if current_time - timestamp >= timeout:
                    expired_keys.append(key)
            for key in expired_keys:
                del _cache[key]
            
            # 生成缓存键
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 检查缓存
            if cache_key in _cache:
                result, timestamp = _cache[cache_key]
                if current_time - timestamp < timeout:
                    logger.debug(f"缓存命中: {cache_key}")
                    return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            _cache[cache_key] = (result, current_time)
            logger.debug(f"缓存更新: {cache_key}")
            return result
        return wrapper
    return decorator

def retry(max_attempts=3, delay=1.0):
    """重试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = Exception("Unknown error")
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {str(e)}")
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))  # 指数退避
            logger.error(f"函数 {func.__name__} 在 {max_attempts} 次尝试后仍然失败")
            raise last_exception
        return wrapper
    return decorator

def test_cache_performance():
    """测试缓存性能"""
    print("开始测试缓存性能...")
    
    # 定义一个模拟的耗时函数
    @cached(timeout=5)
    def slow_function(n):
        # 模拟耗时操作
        time.sleep(0.1)
        return n * n
    
    # 第一次调用（未缓存）
    start_time = time.time()
    result1 = slow_function(5)
    first_call_time = time.time() - start_time
    
    # 第二次调用（缓存命中）
    start_time = time.time()
    result2 = slow_function(5)
    second_call_time = time.time() - start_time
    
    print(f"第一次调用耗时: {first_call_time:.4f}秒")
    print(f"第二次调用耗时: {second_call_time:.4f}秒")
    print(f"性能提升: {first_call_time/second_call_time:.2f}倍")
    print(f"结果一致性: {result1 == result2}")
    
    # 测试缓存过期
    print("\n测试缓存过期...")
    time.sleep(6)  # 等待缓存过期
    start_time = time.time()
    result3 = slow_function(5)
    third_call_time = time.time() - start_time
    print(f"缓存过期后调用耗时: {third_call_time:.4f}秒")

def test_retry_mechanism():
    """测试重试机制"""
    print("\n开始测试重试机制...")
    
    attempt_count = 0
    
    @retry(max_attempts=3, delay=1.0)
    def unreliable_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise Exception(f"模拟错误 #{attempt_count}")
        return "成功"
    
    try:
        result = unreliable_function()
        print(f"重试机制测试结果: {result}")
        print(f"总共尝试次数: {attempt_count}")
    except Exception as e:
        print(f"重试机制测试失败: {e}")

def test_cache_cleanup():
    """测试缓存清理"""
    print("\n开始测试缓存清理...")
    
    # 添加一些测试数据到缓存
    _cache.clear()  # 清空现有缓存
    
    # 添加一些过期的缓存项
    expired_time = datetime.now().timestamp() - 100  # 100秒前
    _cache['expired_key'] = ('expired_value', expired_time)
    
    # 添加一些有效的缓存项
    valid_time = datetime.now().timestamp()
    _cache['valid_key'] = ('valid_value', valid_time)
    
    print(f"清理前缓存项数: {len(_cache)}")
    
    # 定义一个带短超时的缓存函数来触发清理
    @cached(timeout=1)
    def cleanup_test():
        return "test"
    
    cleanup_test()
    
    print(f"清理后缓存项数: {len(_cache)}")
    print(f"过期项是否被清理: {'expired_key' not in _cache}")

def simulate_concurrent_access():
    """模拟并发访问"""
    print("\n开始模拟并发访问...")
    
    @cached(timeout=5)
    def concurrent_function(n):
        # 模拟耗时操作
        time.sleep(0.05)
        return sum(i * i for i in range(n))
    
    # 创建多个线程同时访问
    threads = []
    results = []
    
    def worker(n, results_list):
        start_time = time.time()
        result = concurrent_function(n)
        end_time = time.time()
        results_list.append((n, result, end_time - start_time))
    
    # 启动多个线程
    for i in range(10):
        thread = threading.Thread(target=worker, args=(1000 + i, results))
        threads.append(thread)
        thread.start()
    
    # 等待所有线程完成
    for thread in threads:
        thread.join()
    
    # 分析结果
    total_time = sum(result[2] for result in results)
    avg_time = total_time / len(results)
    print(f"并发访问完成，平均响应时间: {avg_time:.4f}秒")
    print(f"结果一致性: {len(set(result[1] for result in results)) == 1}")

if __name__ == "__main__":
    print("开始性能监控和错误处理测试...")
    
    test_cache_performance()
    test_retry_mechanism()
    test_cache_cleanup()
    simulate_concurrent_access()
    
    print("\n所有测试完成！")
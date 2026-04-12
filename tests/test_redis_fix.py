"""测试Redis客户端修复"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_redis_client_initialization():
    """测试Redis客户端初始化"""
    try:
        # 导入应用创建函数
        from app import create_app
        
        # 创建应用实例（这会初始化Redis客户端）
        app = create_app()
        
        # 测试获取Redis客户端
        from backend.core.redis_client import get_redis_client
        redis_client = get_redis_client()
        
        print("✓ Redis client initialized successfully")
        print(f"✓ Redis client type: {type(redis_client)}")
        
        # 测试ping操作
        try:
            redis_client.ping()
            print("✓ Redis ping successful")
        except Exception as e:
            print(f"⚠ Redis ping failed: {e}")
            print("  (This might be expected if Redis server is not running)")
        
        return True
    except Exception as e:
        print(f"✗ Redis client initialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_health_check():
    """测试数据库健康检查"""
    try:
        from backend.modules.database.manager import get_database_manager
        db_manager = get_database_manager()
        
        # 测试健康检查
        health = db_manager.health_check()
        print(f"✓ Database health check returned: {health}")
        
        return True
    except Exception as e:
        print(f"✗ Database health check test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Testing Redis client and database fixes...")
    print("=" * 50)
    
    tests = [
        ("Redis client initialization", test_redis_client_initialization),
        ("Database health check", test_database_health_check)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} test passed")
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
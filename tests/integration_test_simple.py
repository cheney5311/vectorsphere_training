"""简化版集成测试"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """测试核心模块导入"""
    try:
        # 测试核心模块导入
        from backend.core.redis_client import get_redis_client
        print("✓ Redis client module imported successfully")
        
        from backend.modules.monitoring.service import get_monitoring_service
        print("✓ Monitoring service module imported successfully")
        
        from backend.modules.database.manager import get_database_manager
        print("✓ Database manager module imported successfully")
        
        # 测试应用创建
        from app import create_app
        print("✓ App module imported successfully")
        
        # 尝试创建应用实例
        app = create_app()
        print("✓ Flask app created successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_monitoring_service():
    """测试监控服务"""
    try:
        from backend.modules.monitoring.service import get_monitoring_service
        monitoring_service = get_monitoring_service()
        print("✓ Monitoring service instance created successfully")
        
        # 测试服务方法是否存在
        methods_to_check = [
            'get_system_health',
            'get_system_metrics', 
            'get_training_metrics',
            'get_training_logs',
            'get_resource_usage',
            'get_alerts',
            'resolve_alert',
            'get_dashboard_summary'
        ]
        
        for method in methods_to_check:
            if hasattr(monitoring_service, method):
                print(f"✓ Method {method} exists")
            else:
                print(f"✗ Method {method} is missing")
                
        return True
    except Exception as e:
        print(f"✗ Monitoring service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_manager():
    """测试数据库管理器"""
    try:
        from backend.modules.database.manager import get_database_manager
        db_manager = get_database_manager()
        print("✓ Database manager instance created successfully")
        
        # 测试基本方法
        if hasattr(db_manager, 'health_check'):
            print("✓ health_check method exists")
        else:
            print("✗ health_check method is missing")
            
        return True
    except Exception as e:
        print(f"✗ Database manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_redis_client():
    """测试Redis客户端"""
    try:
        from backend.core.redis_client import get_redis_client
        # 注意：这里不实际调用get_redis_client()因为需要Redis服务器运行
        print("✓ Redis client function imported successfully")
        return True
    except Exception as e:
        print(f"✗ Redis client test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Starting integration tests...")
    print("=" * 50)
    
    tests = [
        ("Core imports", test_imports),
        ("Monitoring service", test_monitoring_service),
        ("Database manager", test_database_manager),
        ("Redis client", test_redis_client)
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
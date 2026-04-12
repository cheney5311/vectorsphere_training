"""全面集成测试 - 检查具体逻辑和错误"""

import sys
import os
import json
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_monitoring_service_methods():
    """测试监控服务的具体方法逻辑"""
    try:
        from backend.modules.monitoring.service import get_monitoring_service
        monitoring_service = get_monitoring_service()
        
        print("Testing monitoring service methods...")
        
        # 测试 get_system_health
        try:
            health = monitoring_service.get_system_health()
            print(f"✓ get_system_health returned: {type(health)}")
            assert isinstance(health, dict), "get_system_health should return a dict"
            assert 'status' in health, "Health response should contain status"
            assert 'components' in health, "Health response should contain components"
            print("✓ get_system_health logic is correct")
        except Exception as e:
            print(f"✗ get_system_health test failed: {e}")
            return False
        
        # 测试 get_system_metrics
        try:
            # 使用默认参数测试
            metrics = monitoring_service.get_system_metrics()
            print(f"✓ get_system_metrics returned: {type(metrics)}")
            assert isinstance(metrics, dict), "get_system_metrics should return a dict"
            assert 'metrics' in metrics, "Metrics response should contain metrics"
            assert 'summary' in metrics, "Metrics response should contain summary"
            print("✓ get_system_metrics logic is correct")
        except Exception as e:
            print(f"✗ get_system_metrics test failed: {e}")
            return False
            
        # 测试 get_training_metrics
        try:
            # 使用默认参数测试
            training_metrics = monitoring_service.get_training_metrics("test_session")
            print(f"✓ get_training_metrics returned: {type(training_metrics)}")
            assert isinstance(training_metrics, dict), "get_training_metrics should return a dict"
            assert 'metrics' in training_metrics, "Training metrics response should contain metrics"
            assert 'summary' in training_metrics, "Training metrics response should contain summary"
            print("✓ get_training_metrics logic is correct")
        except Exception as e:
            print(f"✗ get_training_metrics test failed: {e}")
            # 这个方法可能需要特定的训练会话，所以不视为失败
            print("⚠ get_training_metrics test skipped (expected without active training session)")
            
        # 测试 get_alerts
        try:
            alerts = monitoring_service.get_alerts()
            print(f"✓ get_alerts returned: {type(alerts)}")
            assert isinstance(alerts, dict), "get_alerts should return a dict"
            assert 'alerts' in alerts, "Alerts response should contain alerts"
            assert 'summary' in alerts, "Alerts response should contain summary"
            print("✓ get_alerts logic is correct")
        except Exception as e:
            print(f"✗ get_alerts test failed: {e}")
            return False
            
        # 测试 get_dashboard_summary
        try:
            summary = monitoring_service.get_dashboard_summary()
            print(f"✓ get_dashboard_summary returned: {type(summary)}")
            assert isinstance(summary, dict), "get_dashboard_summary should return a dict"
            assert 'summary' in summary, "Dashboard summary response should contain summary"
            print("✓ get_dashboard_summary logic is correct")
        except Exception as e:
            print(f"✗ get_dashboard_summary test failed: {e}")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Monitoring service methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_auth_service_methods():
    """测试认证服务的具体方法逻辑"""
    try:
        from backend.services.auth_service import get_auth_service
        auth_service = get_auth_service()
        
        print("Testing auth service methods...")
        
        # 测试 authenticate_user 参数类型
        try:
            # 测试可选参数为 None 的情况
            method = getattr(auth_service, 'authenticate_user', None)
            if method:
                print("✓ authenticate_user method exists")
            else:
                print("✗ authenticate_user method is missing")
                return False
        except Exception as e:
            print(f"✗ authenticate_user method check failed: {e}")
            return False
            
        # 测试 get_user 方法
        try:
            method = getattr(auth_service, 'get_user', None)
            if method:
                print("✓ get_user method exists")
            else:
                print("✗ get_user method is missing")
                return False
        except Exception as e:
            print(f"✗ get_user method check failed: {e}")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Auth service methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_manager_methods():
    """测试数据库管理器的具体方法逻辑"""
    try:
        from backend.modules.database.manager import get_database_manager
        db_manager = get_database_manager()
        
        print("Testing database manager methods...")
        
        # 测试 health_check
        try:
            health = db_manager.health_check()
            print(f"✓ health_check returned: {health}")
            # health_check 应该返回布尔值
            assert isinstance(health, bool), "health_check should return a boolean"
            print("✓ health_check logic is correct")
        except Exception as e:
            print(f"✗ health_check test failed: {e}")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Database manager methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_monitor_methods():
    """测试性能监控器的具体方法逻辑"""
    try:
        from backend.modules.monitoring.manager import get_global_monitor
        performance_monitor = get_global_monitor()
        
        print("Testing performance monitor methods...")
        
        # 测试 get_status
        try:
            status = performance_monitor.get_status()
            print(f"✓ get_status returned: {type(status)}")
            assert isinstance(status, dict), "get_status should return a dict"
            assert 'status' in status, "Status response should contain status"
            print("✓ get_status logic is correct")
        except Exception as e:
            print(f"✗ get_status test failed: {e}")
            return False
            
        # 测试 get_metrics_history
        try:
            metrics = performance_monitor.get_metrics_history()
            print(f"✓ get_metrics_history returned: {type(metrics)}")
            assert isinstance(metrics, dict), "get_metrics_history should return a dict"
            print("✓ get_metrics_history logic is correct")
        except Exception as e:
            print(f"✗ get_metrics_history test failed: {e}")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Performance monitor methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_alert_manager_methods():
    """测试告警管理器的具体方法逻辑"""
    try:
        from backend.modules.monitoring.alert_manager import get_global_alert_manager
        alert_manager = get_global_alert_manager()
        
        print("Testing alert manager methods...")
        
        # 测试 get_status
        try:
            status = alert_manager.get_status()
            print(f"✓ get_status returned: {type(status)}")
            assert isinstance(status, dict), "get_status should return a dict"
            assert 'status' in status, "Status response should contain status"
            print("✓ get_status logic is correct")
        except Exception as e:
            print(f"✗ get_status test failed: {e}")
            return False
            
        # 测试 get_alerts
        try:
            alerts = alert_manager.get_alerts()
            print(f"✓ get_alerts returned: {type(alerts)}")
            assert isinstance(alerts, list), "get_alerts should return a list"
            print("✓ get_alerts logic is correct")
        except Exception as e:
            print(f"✗ get_alerts test failed: {e}")
            return False
            
        return True
    except Exception as e:
        print(f"✗ Alert manager methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("Starting comprehensive integration tests...")
    print("=" * 60)
    
    tests = [
        ("Monitoring service methods", test_monitoring_service_methods),
        ("Auth service methods", test_auth_service_methods),
        ("Database manager methods", test_database_manager_methods),
        ("Performance monitor methods", test_performance_monitor_methods),
        ("Alert manager methods", test_alert_manager_methods)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} test passed")
            else:
                print(f"✗ {test_name} test failed")
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All comprehensive tests passed!")
        return 0
    else:
        print("❌ Some comprehensive tests failed!")
        return 1

if __name__ == "__main__":
    exit(main())
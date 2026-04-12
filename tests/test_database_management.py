"""数据库管理API测试

测试 database_management_service.py 和 database_api.py 的所有功能。
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDatabaseManagementService(unittest.TestCase):
    """数据库管理服务测试"""
    
    def setUp(self):
        """测试准备"""
        # 模拟数据库管理器
        self.mock_db_manager = Mock()
        self.mock_db_manager.health_check.return_value = True
        self.mock_db_manager.get_connection_stats.return_value = {
            'pool_size': 10,
            'checked_in': 8,
            'checked_out': 2,
            'overflow': 0,
            'invalid': 0
        }
        
        # 模拟引擎
        self.mock_engine = Mock()
        self.mock_engine.url = 'postgresql://localhost/test'
        self.mock_db_manager.engine = self.mock_engine
        
        # 模拟配置
        self.mock_config = Mock()
        self.mock_config.pool_size = 10
        self.mock_config.max_overflow = 20
        self.mock_config.pool_timeout = 30
        self.mock_config.pool_recycle = 3600
        
    @patch('backend.services.database_management_service.get_database_manager')
    @patch('backend.services.database_management_service.get_database_config')
    def test_check_health(self, mock_get_config, mock_get_manager):
        """测试健康检查"""
        mock_get_manager.return_value = self.mock_db_manager
        mock_get_config.return_value = self.mock_config
        
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        health = service.check_health()
        
        # 验证结果
        self.assertEqual(health['status'], 'healthy')
        self.assertTrue(health['connection_available'])
        self.assertIn('database_type', health)
        self.assertIn('timestamp', health)
        
    @patch('backend.services.database_management_service.get_database_manager')
    @patch('backend.services.database_management_service.get_database_config')
    def test_check_health_unhealthy(self, mock_get_config, mock_get_manager):
        """测试不健康状态"""
        self.mock_db_manager.health_check.return_value = False
        mock_get_manager.return_value = self.mock_db_manager
        mock_get_config.return_value = self.mock_config
        
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        health = service.check_health()
        
        self.assertEqual(health['status'], 'unhealthy')
        self.assertFalse(health['connection_available'])
        
    @patch('backend.services.database_management_service.get_database_manager')
    @patch('backend.services.database_management_service.get_database_config')
    def test_get_pool_stats(self, mock_get_config, mock_get_manager):
        """测试获取连接池统计"""
        mock_get_manager.return_value = self.mock_db_manager
        mock_get_config.return_value = self.mock_config
        
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        stats = service.get_pool_stats()
        
        # 验证结果
        self.assertEqual(stats['pool_size'], 10)
        self.assertEqual(stats['checked_in'], 8)
        self.assertEqual(stats['checked_out'], 2)
        self.assertIn('utilization_percent', stats)
        self.assertIn('config', stats)
        
    @patch('backend.services.database_management_service.get_database_manager')
    @patch('backend.services.database_management_service.get_database_config')
    @patch('sqlalchemy.inspect')
    def test_list_tables(self, mock_inspect, mock_get_config, mock_get_manager):
        """测试列出表"""
        mock_get_manager.return_value = self.mock_db_manager
        mock_get_config.return_value = self.mock_config
        
        # 模拟 inspector
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ['users', 'projects', 'models']
        mock_inspector.get_columns.return_value = [
            {'name': 'id', 'type': 'UUID', 'nullable': False},
            {'name': 'name', 'type': 'VARCHAR', 'nullable': True}
        ]
        mock_inspector.get_indexes.return_value = [
            {'name': 'ix_users_email', 'column_names': ['email'], 'unique': True}
        ]
        mock_inspector.get_pk_constraint.return_value = {'constrained_columns': ['id']}
        mock_inspect.return_value = mock_inspector
        
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        result = service.list_tables()
        
        # 验证结果
        self.assertIn('tables', result)
        self.assertIn('total_count', result)
        self.assertEqual(result['total_count'], 3)
            
    @patch('backend.services.database_management_service.get_database_manager')
    @patch('backend.services.database_management_service.get_database_config')
    @patch('sqlalchemy.inspect')
    def test_list_tables_with_search(self, mock_inspect, mock_get_config, mock_get_manager):
        """测试搜索表"""
        mock_get_manager.return_value = self.mock_db_manager
        mock_get_config.return_value = self.mock_config
        
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ['users', 'user_sessions', 'projects']
        mock_inspector.get_columns.return_value = []
        mock_inspector.get_indexes.return_value = []
        mock_inspector.get_pk_constraint.return_value = {'constrained_columns': []}
        mock_inspect.return_value = mock_inspector
        
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        result = service.list_tables(search='user')
        
        # 验证只返回包含 'user' 的表
        self.assertEqual(result['total_count'], 2)
        table_names = [t['name'] for t in result['tables']]
        self.assertIn('users', table_names)
        self.assertIn('user_sessions', table_names)
        self.assertNotIn('projects', table_names)
            
    @patch('backend.services.database_management_service.get_database_manager')
    @patch('backend.services.database_management_service.get_database_config')
    def test_get_table_count(self, mock_get_config, mock_get_manager):
        """测试获取表记录数"""
        mock_get_manager.return_value = self.mock_db_manager
        mock_get_config.return_value = self.mock_config
        
        # 模拟会话
        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1000
        mock_session.execute.return_value = mock_result
        
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_session
        mock_context_manager.__exit__.return_value = None
        self.mock_db_manager.get_db_session.return_value = mock_context_manager
        
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        result = service.get_table_count('users')
        
        # 验证结果
        self.assertEqual(result['table_name'], 'users')
        self.assertEqual(result['count'], 1000)
        self.assertIn('counted_at', result)
        
    @patch('backend.services.database_management_service.get_database_manager')
    @patch('backend.services.database_management_service.get_database_config')
    @patch('sqlalchemy.inspect')
    def test_create_backup(self, mock_inspect, mock_get_config, mock_get_manager):
        """测试创建备份"""
        mock_get_manager.return_value = self.mock_db_manager
        mock_get_config.return_value = self.mock_config
        
        # 模拟表列表
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ['users', 'projects']
        mock_inspector.get_columns.return_value = []
        mock_inspector.get_indexes.return_value = []
        mock_inspector.get_pk_constraint.return_value = {'constrained_columns': []}
        mock_inspect.return_value = mock_inspector
        
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        result = service.create_backup(backup_name='test_backup')
        
        # 验证结果
        self.assertIn('backup_id', result)
        self.assertEqual(result['backup_name'], 'test_backup')
        self.assertIn('backup_path', result)
        self.assertIn('tables_count', result)
        self.assertIn('created_at', result)
            
    @patch('backend.services.database_management_service.get_database_manager')
    @patch('backend.services.database_management_service.get_database_config')
    def test_list_backups(self, mock_get_config, mock_get_manager):
        """测试列出备份"""
        mock_get_manager.return_value = self.mock_db_manager
        mock_get_config.return_value = self.mock_config
        
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        result = service.list_backups()
        
        # 验证结果
        self.assertIn('backups', result)
        self.assertIn('total_count', result)
        self.assertIsInstance(result['backups'], list)
        
    @patch('backend.services.database_management_service.get_database_manager')
    @patch('backend.services.database_management_service.get_database_config')
    def test_format_size(self, mock_get_config, mock_get_manager):
        """测试大小格式化"""
        mock_get_manager.return_value = self.mock_db_manager
        mock_get_config.return_value = self.mock_config
        
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        
        # 测试不同大小
        self.assertEqual(service._format_size(500), '500 B')
        self.assertEqual(service._format_size(1024), '1.00 KB')
        self.assertEqual(service._format_size(1024 * 1024), '1.00 MB')
        self.assertEqual(service._format_size(1024 * 1024 * 1024), '1.00 GB')


class TestDatabaseManagementAPI(unittest.TestCase):
    """数据库管理API测试"""
    
    def test_api_blueprint_exists(self):
        """测试API蓝图存在"""
        from backend.api.database.api import database_bp
        
        self.assertIsNotNone(database_bp)
        self.assertEqual(database_bp.url_prefix, '/api/v1/database')
        self.assertEqual(database_bp.name, 'database')
        
    def test_health_endpoint_exists(self):
        """测试健康检查端点存在"""
        from backend.api.database.api import health_check
        
        self.assertIsNotNone(health_check)
        self.assertTrue(callable(health_check))


class TestDatabaseManagementIntegration(unittest.TestCase):
    """数据库管理集成测试"""
    
    @patch('backend.services.database_management_service.get_database_manager')
    @patch('backend.services.database_management_service.get_database_config')
    @patch('sqlalchemy.inspect')
    def test_detailed_health_workflow(self, mock_inspect, mock_get_config, mock_get_manager):
        """测试详细健康检查工作流"""
        # 设置模拟
        mock_db_manager = Mock()
        mock_db_manager.health_check.return_value = True
        mock_db_manager.get_connection_stats.return_value = {
            'pool_size': 10,
            'checked_in': 8,
            'checked_out': 2,
            'overflow': 0,
            'invalid': 0
        }
        mock_engine = Mock()
        mock_engine.url = 'postgresql://localhost/test'
        mock_db_manager.engine = mock_engine
        
        mock_config = Mock()
        mock_config.pool_size = 10
        mock_config.max_overflow = 20
        mock_config.pool_timeout = 30
        mock_config.pool_recycle = 3600
        
        mock_get_manager.return_value = mock_db_manager
        mock_get_config.return_value = mock_config
        
        # 模拟 inspector
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ['users', 'projects']
        mock_inspector.get_columns.return_value = []
        mock_inspector.get_indexes.return_value = []
        mock_inspector.get_pk_constraint.return_value = {'constrained_columns': []}
        mock_inspect.return_value = mock_inspector
        
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        
        # 执行详细健康检查
        detailed = service.get_detailed_health()
        
        # 验证结果
        self.assertEqual(detailed['status'], 'healthy')
        self.assertIn('connection', detailed)
        self.assertIn('pool', detailed)
        self.assertIn('tables', detailed)
        self.assertTrue(detailed['connection']['available'])
            
    @patch('backend.services.database_management_service.get_database_manager')
    @patch('backend.services.database_management_service.get_database_config')
    @patch('sqlalchemy.inspect')
    def test_database_stats_workflow(self, mock_inspect, mock_get_config, mock_get_manager):
        """测试数据库统计工作流"""
        # 设置模拟
        mock_db_manager = Mock()
        mock_db_manager.health_check.return_value = True
        mock_db_manager.get_connection_stats.return_value = {
            'pool_size': 10,
            'checked_in': 8,
            'checked_out': 2,
            'overflow': 0,
            'invalid': 0
        }
        mock_engine = Mock()
        mock_engine.url = 'postgresql://localhost/test'
        mock_db_manager.engine = mock_engine
        
        # 模拟会话
        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 100
        mock_session.execute.return_value = mock_result
        
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_session
        mock_context_manager.__exit__.return_value = None
        mock_db_manager.get_db_session.return_value = mock_context_manager
        
        mock_config = Mock()
        mock_config.pool_size = 10
        mock_config.max_overflow = 20
        mock_config.pool_timeout = 30
        mock_config.pool_recycle = 3600
        
        mock_get_manager.return_value = mock_db_manager
        mock_get_config.return_value = mock_config
        
        # 模拟 inspector
        mock_inspector = Mock()
        mock_inspector.get_table_names.return_value = ['users', 'projects']
        mock_inspector.get_columns.return_value = []
        mock_inspector.get_indexes.return_value = []
        mock_inspector.get_pk_constraint.return_value = {'constrained_columns': []}
        mock_inspect.return_value = mock_inspector
        
        from backend.services.database_management_service import DatabaseManagementService
        
        service = DatabaseManagementService()
        
        # 获取数据库统计
        stats = service.get_database_stats()
        
        # 验证结果
        self.assertIn('database_info', stats)
        self.assertIn('tables', stats)
        self.assertIn('storage', stats)
        self.assertIn('connection_pool', stats)


def run_service_tests():
    """运行服务层测试"""
    print("\n" + "=" * 60)
    print("数据库管理服务测试")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseManagementService))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseManagementAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseManagementIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


def run_functional_tests():
    """运行功能测试"""
    print("\n" + "=" * 60)
    print("功能测试")
    print("=" * 60)
    
    # 测试1: 验证服务初始化
    print("\n[1] 测试服务导入...")
    try:
        from backend.services.database_management_service import DatabaseManagementService
        print("   ✓ 服务模块导入成功")
    except Exception as e:
        print(f"   ✗ 服务导入失败: {e}")
        
    # 测试2: 验证API蓝图加载
    print("\n[2] 测试API蓝图加载...")
    try:
        from backend.api.database.api import database_bp
        assert database_bp is not None
        assert database_bp.name == 'database'
        print("   ✓ API蓝图加载成功")
    except Exception as e:
        print(f"   ✗ API蓝图加载失败: {e}")
        
    # 测试3: 验证全局服务获取函数
    print("\n[3] 测试全局服务获取...")
    try:
        from backend.services.database_management_service import get_database_management_service
        # 这个会失败因为需要实际数据库连接，但导入应该成功
        print("   ✓ 服务获取函数可用")
    except Exception as e:
        print(f"   ✗ 服务获取函数导入失败: {e}")
        
    # 测试4: 验证大小格式化
    print("\n[4] 测试大小格式化...")
    try:
        from backend.services.database_management_service import DatabaseManagementService
        from unittest.mock import Mock, patch
        
        with patch('backend.services.database_management_service.get_database_manager') as mock_mgr, \
             patch('backend.services.database_management_service.get_database_config') as mock_cfg:
            
            mock_mgr.return_value = Mock()
            mock_mgr.return_value.engine = Mock()
            mock_cfg.return_value = Mock()
            
            service = DatabaseManagementService()
            
            assert service._format_size(100) == '100 B'
            assert service._format_size(1024) == '1.00 KB'
            assert service._format_size(1024 * 1024) == '1.00 MB'
            assert service._format_size(1024 * 1024 * 1024) == '1.00 GB'
            
        print("   ✓ 大小格式化正确")
    except Exception as e:
        print(f"   ✗ 大小格式化测试失败: {e}")
        
    print("\n" + "=" * 60)
    print("功能测试完成")
    print("=" * 60)


if __name__ == '__main__':
    # 运行单元测试
    result = run_service_tests()
    
    # 运行功能测试
    run_functional_tests()
    
    # 输出结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success = total - failures - errors
    
    print(f"总测试数: {total}")
    print(f"成功: {success}")
    print(f"失败: {failures}")
    print(f"错误: {errors}")
    
    if failures == 0 and errors == 0:
        print("\n✅ 所有测试通过!")
    else:
        print("\n❌ 部分测试失败")
        
    sys.exit(0 if failures == 0 and errors == 0 else 1)

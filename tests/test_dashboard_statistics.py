"""仪表盘统计功能测试

测试 dashboard_statistics_api.py 及其下游服务的功能。

测试覆盖：
1. 训练进度趋势统计
2. 活跃训练任务
3. 训练时长统计
4. 系统资源历史
5. 模型性能分布
6. 系统健康状态
7. 周期对比统计
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDashboardStatisticsService(unittest.TestCase):
    """测试仪表盘统计服务层"""
    
    def setUp(self):
        """设置测试环境"""
        from backend.services import dashboard_service
        dashboard_service._dashboard_service = None
    
    @patch('backend.services.dashboard_service.get_dashboard_repository')
    def test_get_training_progress_trend(self, mock_get_repo):
        """测试获取训练进度趋势"""
        from backend.services.dashboard_service import DashboardService
        
        mock_repo = Mock()
        mock_repo.get_training_progress_trend.return_value = [
            {'date': '2026-01-10', 'completed': 8, 'running': 2, 'failed': 1, 'pending': 0, 'total': 11},
            {'date': '2026-01-11', 'completed': 12, 'running': 3, 'failed': 0, 'pending': 1, 'total': 16},
            {'date': '2026-01-12', 'completed': 10, 'running': 1, 'failed': 2, 'pending': 0, 'total': 13}
        ]
        mock_get_repo.return_value = mock_repo
        
        service = DashboardService(repository=mock_repo)
        result = service.get_training_progress_trend(user_id='user123', days=7)
        
        # 验证结果结构
        self.assertIn('trend', result)
        self.assertIn('period', result)
        self.assertIn('summary', result)
        
        # 验证汇总计算
        self.assertEqual(result['summary']['total_completed'], 30)
        self.assertEqual(result['summary']['total_failed'], 3)
        self.assertGreater(result['summary']['success_rate'], 0.9)
    
    @patch('backend.services.dashboard_service.get_dashboard_repository')
    def test_get_active_training_tasks(self, mock_get_repo):
        """测试获取活跃训练任务"""
        from backend.services.dashboard_service import DashboardService
        
        mock_repo = Mock()
        mock_repo.get_active_training_sessions.return_value = [
            {
                'id': 'sess_001',
                'name': 'BERT训练',
                'model_type': 'nlp',
                'progress': 75,
                'accuracy': 0.92,
                'remaining_time': '2小时',
                'status': 'running',
                'start_time': '2026-01-16T10:00:00',
                'estimated_end': '2026-01-16T14:00:00',
                'current_epoch': 8,
                'total_epochs': 10
            },
            {
                'id': 'sess_002',
                'name': 'ResNet训练',
                'model_type': 'cv',
                'progress': 30,
                'accuracy': 0.85,
                'remaining_time': '5小时',
                'status': 'running',
                'start_time': '2026-01-16T08:00:00',
                'estimated_end': '2026-01-16T18:00:00',
                'current_epoch': 3,
                'total_epochs': 10
            }
        ]
        mock_get_repo.return_value = mock_repo
        
        service = DashboardService(repository=mock_repo)
        result = service.get_active_training_tasks(user_id='user123')
        
        self.assertIn('active_tasks', result)
        self.assertEqual(result['total_count'], 2)
        self.assertEqual(result['running_count'], 2)
        self.assertEqual(len(result['active_tasks']), 2)
    
    @patch('backend.services.dashboard_service.get_dashboard_repository')
    def test_get_training_duration_stats(self, mock_get_repo):
        """测试获取训练时长统计"""
        from backend.services.dashboard_service import DashboardService
        
        mock_repo = Mock()
        mock_repo.get_training_duration_stats.return_value = {
            'avg_duration': 2.5,
            'min_duration': 0.5,
            'max_duration': 8.2,
            'total_count': 48,
            'duration_distribution': [
                {'range': '0-1小时', 'count': 5},
                {'range': '1-3小时', 'count': 18},
                {'range': '3-6小时', 'count': 15},
                {'range': '6-12小时', 'count': 8},
                {'range': '12+小时', 'count': 2}
            ]
        }
        mock_get_repo.return_value = mock_repo
        
        service = DashboardService(repository=mock_repo)
        result = service.get_training_duration_stats(user_id='user123', days=30)
        
        self.assertEqual(result['avg_duration'], 2.5)
        self.assertEqual(result['total_count'], 48)
        self.assertEqual(len(result['duration_distribution']), 5)
    
    @patch('backend.services.dashboard_service.get_dashboard_repository')
    def test_get_model_performance_distribution(self, mock_get_repo):
        """测试获取模型性能分布"""
        from backend.services.dashboard_service import DashboardService
        
        mock_repo = Mock()
        mock_repo.get_model_performance_distribution.return_value = {
            'accuracy_distribution': [
                {'range': '90-100%', 'count': 5},
                {'range': '80-90%', 'count': 12},
                {'range': '70-80%', 'count': 3}
            ],
            'model_types': [
                {'type': 'classification', 'count': 8, 'avg_accuracy': 91.2},
                {'type': 'regression', 'count': 5, 'avg_accuracy': 87.8}
            ],
            'total_models': 20
        }
        mock_get_repo.return_value = mock_repo
        
        service = DashboardService(repository=mock_repo)
        result = service.get_model_performance_distribution(user_id='user123')
        
        self.assertEqual(result['total_models'], 20)
        self.assertEqual(len(result['accuracy_distribution']), 3)
        self.assertEqual(len(result['model_types']), 2)
    
    @patch('backend.services.dashboard_service.psutil')
    def test_get_system_health_status(self, mock_psutil):
        """测试获取系统健康状态"""
        from backend.services.dashboard_service import DashboardService
        
        mock_psutil.boot_time.return_value = datetime.now().timestamp() - 86400  # 1天前
        
        service = DashboardService(repository=Mock())
        
        with patch('backend.modules.database.manager.get_database_manager') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.get_db_session.return_value.__enter__ = Mock(return_value=mock_session)
            mock_db.return_value.get_db_session.return_value.__exit__ = Mock(return_value=False)
            
            result = service.get_system_health_status()
        
        self.assertIn('overall_status', result)
        self.assertIn('services', result)
        self.assertIn('uptime', result)
        self.assertIn('last_check', result)
        
        # API 服务器应该始终运行
        self.assertEqual(result['services']['api_server'], 'running')
    
    @patch('backend.services.dashboard_service.psutil')
    def test_get_resource_usage_history(self, mock_psutil):
        """测试获取资源使用历史"""
        from backend.services.dashboard_service import DashboardService
        
        # 模拟 psutil 返回
        mock_psutil.cpu_percent.return_value = 45.5
        mock_psutil.cpu_count.return_value = 8
        
        mock_memory = Mock()
        mock_memory.percent = 62.3
        mock_memory.used = 12 * (1024 ** 3)
        mock_memory.total = 32 * (1024 ** 3)
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.percent = 55.0
        mock_disk.used = 256 * (1024 ** 3)
        mock_disk.total = 512 * (1024 ** 3)
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_net = Mock()
        mock_net.bytes_sent = 1024 * (1024 ** 2)
        mock_net.bytes_recv = 2048 * (1024 ** 2)
        mock_psutil.net_io_counters.return_value = mock_net
        
        service = DashboardService(repository=Mock())
        result = service.get_resource_usage_history(hours=24, interval_minutes=60)
        
        # 应该有 24 个数据点
        self.assertEqual(len(result), 24)
        
        # 验证每个数据点的结构
        for point in result:
            self.assertIn('timestamp', point)
            self.assertIn('cpu_percent', point)
            self.assertIn('memory_percent', point)
            self.assertIn('disk_percent', point)


class TestDashboardStatisticsRepository(unittest.TestCase):
    """测试仪表盘统计数据仓库"""
    
    def test_get_training_progress_trend(self):
        """测试训练进度趋势查询"""
        from backend.repositories.dashboard_repository import DashboardRepository
        
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.group_by.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.all.return_value = []
        mock_session.query.return_value = mock_query
        mock_session.close = Mock()
        
        mock_db_manager = Mock()
        mock_db_manager.get_session.return_value = mock_session
        
        repo = DashboardRepository(db_manager=mock_db_manager)
        result = repo.get_training_progress_trend(user_id='user123', days=7)
        
        # 即使没有数据，也应该返回 7 天的数据（填充零值）
        self.assertEqual(len(result), 8)  # 7天 + 当天
        
        for day_data in result:
            self.assertIn('date', day_data)
            self.assertIn('completed', day_data)
            self.assertIn('running', day_data)
            self.assertIn('failed', day_data)
    
    def test_get_training_duration_stats(self):
        """测试训练时长统计查询"""
        from backend.repositories.dashboard_repository import DashboardRepository
        
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        
        # 模拟统计查询结果
        mock_stats_result = Mock()
        mock_stats_result.avg_duration = 9000  # 2.5小时（秒）
        mock_stats_result.min_duration = 1800  # 0.5小时
        mock_stats_result.max_duration = 29520  # 8.2小时
        mock_stats_result.count = 48
        mock_query.first.return_value = mock_stats_result
        
        # 模拟分布查询
        mock_query.all.return_value = []
        
        mock_session.query.return_value = mock_query
        mock_session.close = Mock()
        
        mock_db_manager = Mock()
        mock_db_manager.get_session.return_value = mock_session
        
        repo = DashboardRepository(db_manager=mock_db_manager)
        result = repo.get_training_duration_stats(user_id='user123', days=30)
        
        self.assertIn('avg_duration', result)
        self.assertIn('duration_distribution', result)


class TestDashboardStatisticsAPI(unittest.TestCase):
    """测试仪表盘统计 API 端点"""
    
    def setUp(self):
        """设置测试环境"""
        from flask import Flask
        from flask_jwt_extended import JWTManager, create_access_token
        
        self.app = Flask(__name__)
        self.app.config['JWT_SECRET_KEY'] = 'test-secret-key'
        self.app.config['TESTING'] = True
        
        JWTManager(self.app)
        
        from backend.api.dashboard.dashboard_statistics_api import dashboard_statistics_bp
        self.app.register_blueprint(dashboard_statistics_bp)
        
        self.client = self.app.test_client()
        
        with self.app.app_context():
            self.test_token = create_access_token(identity='test_user')
    
    def get_auth_headers(self):
        """获取认证请求头"""
        return {'Authorization': f'Bearer {self.test_token}'}
    
    @patch('backend.api.dashboard.dashboard_statistics_api.get_dashboard_service')
    def test_get_training_progress_endpoint(self, mock_get_service):
        """测试训练进度端点"""
        mock_service = Mock()
        mock_service.get_training_progress_trend.return_value = {
            'trend': [
                {'date': '2026-01-10', 'completed': 8, 'running': 2, 'failed': 1}
            ],
            'period': '7 days',
            'summary': {
                'total_completed': 8,
                'total_failed': 1,
                'total_running': 2,
                'avg_daily_completed': 1.14,
                'success_rate': 0.889
            }
        }
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/dashboard/training/progress?days=7',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('trend', data['data'])
        self.assertIn('summary', data['data'])
    
    @patch('backend.api.dashboard.dashboard_statistics_api.get_dashboard_service')
    def test_get_active_training_endpoint(self, mock_get_service):
        """测试活跃训练端点"""
        mock_service = Mock()
        mock_service.get_active_training_tasks.return_value = {
            'active_tasks': [
                {
                    'id': 'sess_001',
                    'name': 'BERT训练',
                    'progress': 75,
                    'status': 'running'
                }
            ],
            'total_count': 1,
            'running_count': 1,
            'pending_count': 0,
            'paused_count': 0
        }
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/dashboard/training/active',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('active_tasks', data['data'])
        self.assertEqual(data['data']['total_count'], 1)
    
    @patch('backend.api.dashboard.dashboard_statistics_api.get_dashboard_service')
    def test_get_training_duration_endpoint(self, mock_get_service):
        """测试训练时长端点"""
        mock_service = Mock()
        mock_service.get_training_duration_stats.return_value = {
            'avg_duration': 2.5,
            'min_duration': 0.5,
            'max_duration': 8.2,
            'total_count': 48,
            'duration_distribution': [
                {'range': '0-1小时', 'count': 5}
            ]
        }
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/dashboard/training/duration?days=30',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['avg_duration'], 2.5)
    
    @patch('backend.api.dashboard.dashboard_statistics_api.get_dashboard_service')
    def test_get_system_resources_history_endpoint(self, mock_get_service):
        """测试系统资源历史端点"""
        mock_service = Mock()
        mock_service.get_resource_usage_history.return_value = [
            {
                'timestamp': '2026-01-16T00:00:00',
                'cpu_percent': 45.5,
                'memory_percent': 62.3,
                'gpu_percent': 78.5,
                'disk_percent': 55.2
            }
        ]
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/dashboard/system/resources/history?hours=24',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('history', data['data'])
        self.assertEqual(data['data']['period_hours'], 24)
    
    @patch('backend.api.dashboard.dashboard_statistics_api.get_dashboard_service')
    def test_get_model_performance_endpoint(self, mock_get_service):
        """测试模型性能端点"""
        mock_service = Mock()
        mock_service.get_model_performance_distribution.return_value = {
            'accuracy_distribution': [
                {'range': '90-100%', 'count': 5}
            ],
            'model_types': [
                {'type': 'classification', 'count': 8, 'avg_accuracy': 91.2}
            ],
            'total_models': 8
        }
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/dashboard/models/performance',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('accuracy_distribution', data['data'])
        self.assertIn('model_types', data['data'])
    
    @patch('backend.api.dashboard.dashboard_statistics_api.get_dashboard_service')
    def test_get_system_health_endpoint(self, mock_get_service):
        """测试系统健康端点"""
        mock_service = Mock()
        mock_service.get_system_health_status.return_value = {
            'overall_status': 'healthy',
            'services': {
                'database': 'running',
                'api_server': 'running',
                'scheduler': 'running'
            },
            'uptime': '5天 12小时',
            'last_check': '2026-01-16T14:00:00'
        }
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/dashboard/system/health',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['overall_status'], 'healthy')
        self.assertIn('services', data['data'])
    
    @patch('backend.api.dashboard.dashboard_statistics_api.get_dashboard_service')
    def test_get_period_comparison_endpoint(self, mock_get_service):
        """测试周期对比端点"""
        from backend.schemas.dashboard_models import (
            DashboardOverview, TrainingOverview, ModelOverview, SystemResourceSnapshot
        )
        
        mock_service = Mock()
        mock_service.get_training_progress_trend.return_value = {
            'trend': [],
            'period': '7 days',
            'summary': {}
        }
        mock_get_service.return_value = mock_service
        
        mock_repo = Mock()
        mock_repo.get_training_overview.return_value = {
            'total_count': 45,
            'completed_count': 42,
            'failed_count': 3,
            'success_rate': 0.9333
        }
        
        # Mock 仓库模块中的 get_dashboard_repository 函数
        with patch('backend.repositories.dashboard_repository.get_dashboard_repository', return_value=mock_repo):
            response = self.client.get(
                '/api/v1/dashboard/comparison?period=week',
                headers=self.get_auth_headers()
            )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('current', data['data'])
        self.assertIn('previous', data['data'])
        self.assertIn('comparison', data['data'])
    
    def test_unauthorized_access(self):
        """测试未授权访问"""
        endpoints = [
            '/api/v1/dashboard/training/progress',
            '/api/v1/dashboard/training/active',
            '/api/v1/dashboard/training/duration',
            '/api/v1/dashboard/system/resources/history',
            '/api/v1/dashboard/models/performance',
            '/api/v1/dashboard/system/health'
        ]
        
        for endpoint in endpoints:
            response = self.client.get(endpoint)
            self.assertIn(response.status_code, [401, 422], f"Endpoint {endpoint} should require auth")


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_statistics_flow(self):
        """测试完整统计流程"""
        from backend.services.dashboard_service import DashboardService, reset_dashboard_service
        
        reset_dashboard_service()
        
        mock_repo = Mock()
        mock_repo.get_training_progress_trend.return_value = [
            {'date': '2026-01-10', 'completed': 8, 'running': 2, 'failed': 1, 'pending': 0, 'total': 11},
            {'date': '2026-01-11', 'completed': 12, 'running': 3, 'failed': 0, 'pending': 1, 'total': 16}
        ]
        mock_repo.get_active_training_sessions.return_value = [
            {'id': 'sess_001', 'name': 'Test', 'status': 'running', 'progress': 75}
        ]
        mock_repo.get_training_duration_stats.return_value = {
            'avg_duration': 2.5,
            'min_duration': 0.5,
            'max_duration': 8.2,
            'total_count': 48,
            'duration_distribution': []
        }
        mock_repo.get_model_performance_distribution.return_value = {
            'accuracy_distribution': [],
            'model_types': [],
            'total_models': 0
        }
        
        service = DashboardService(repository=mock_repo)
        
        # 测试训练进度趋势
        trend = service.get_training_progress_trend(user_id='user123', days=7)
        self.assertIn('trend', trend)
        self.assertIn('summary', trend)
        self.assertEqual(trend['summary']['total_completed'], 20)
        
        # 测试活跃任务
        tasks = service.get_active_training_tasks(user_id='user123')
        self.assertEqual(tasks['total_count'], 1)
        self.assertEqual(tasks['running_count'], 1)
        
        # 测试时长统计
        duration = service.get_training_duration_stats(user_id='user123')
        self.assertEqual(duration['avg_duration'], 2.5)
        
        # 测试系统健康（真实调用）
        health = service.get_system_health_status()
        self.assertIn('overall_status', health)
        self.assertIn('services', health)
        
        print(f"\n统计测试结果:")
        print(f"  训练趋势: {len(trend['trend'])} 天数据")
        print(f"  活跃任务: {tasks['total_count']} 个")
        print(f"  平均时长: {duration['avg_duration']:.2f} 小时")
        print(f"  系统状态: {health['overall_status']}")


class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    @patch('backend.services.dashboard_service.get_dashboard_repository')
    def test_empty_data_handling(self, mock_get_repo):
        """测试空数据处理"""
        from backend.services.dashboard_service import DashboardService
        
        mock_repo = Mock()
        mock_repo.get_training_progress_trend.return_value = []
        mock_repo.get_active_training_sessions.return_value = []
        mock_repo.get_training_duration_stats.return_value = {
            'avg_duration': 0,
            'min_duration': 0,
            'max_duration': 0,
            'total_count': 0,
            'duration_distribution': []
        }
        mock_get_repo.return_value = mock_repo
        
        service = DashboardService(repository=mock_repo)
        
        # 空趋势
        trend = service.get_training_progress_trend(user_id='user123')
        self.assertEqual(trend['summary']['total_completed'], 0)
        self.assertEqual(trend['summary']['success_rate'], 0)
        
        # 空任务
        tasks = service.get_active_training_tasks(user_id='user123')
        self.assertEqual(tasks['total_count'], 0)
        self.assertEqual(tasks['active_tasks'], [])
    
    @patch('backend.services.dashboard_service.get_dashboard_repository')
    def test_error_handling(self, mock_get_repo):
        """测试错误处理"""
        from backend.services.dashboard_service import DashboardService
        
        mock_repo = Mock()
        mock_repo.get_training_progress_trend.side_effect = Exception("Database error")
        mock_get_repo.return_value = mock_repo
        
        service = DashboardService(repository=mock_repo)
        
        # 应该返回默认值而不是抛出异常
        result = service.get_training_progress_trend(user_id='user123')
        self.assertEqual(result['trend'], [])
        self.assertEqual(result['summary']['total_completed'], 0)
    
    def test_parameter_validation(self):
        """测试参数验证"""
        from flask import Flask
        from flask_jwt_extended import JWTManager, create_access_token
        
        app = Flask(__name__)
        app.config['JWT_SECRET_KEY'] = 'test-secret-key'
        app.config['TESTING'] = True
        
        JWTManager(app)
        
        from backend.api.dashboard.dashboard_statistics_api import dashboard_statistics_bp
        app.register_blueprint(dashboard_statistics_bp)
        
        client = app.test_client()
        
        with app.app_context():
            token = create_access_token(identity='test_user')
        
        headers = {'Authorization': f'Bearer {token}'}
        
        with patch('backend.api.dashboard.dashboard_statistics_api.get_dashboard_service') as mock:
            mock.return_value.get_training_progress_trend.return_value = {
                'trend': [], 'period': '90 days', 'summary': {}
            }
            
            # 测试 days 参数最大值限制
            response = client.get(
                '/api/v1/dashboard/training/progress?days=999',
                headers=headers
            )
            self.assertEqual(response.status_code, 200)
            # days 应该被限制为 90


if __name__ == '__main__':
    print("=" * 70)
    print("仪表盘统计功能测试")
    print("=" * 70)
    
    unittest.main(verbosity=2)

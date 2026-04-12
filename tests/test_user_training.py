"""用户训练功能测试

测试 user_training_api.py 及其下游服务的功能。

测试覆盖：
1. 用户训练概览
2. 训练会话管理
3. 训练统计
4. 训练趋势
5. 模型性能排行
6. 时长统计
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestUserTrainingService(unittest.TestCase):
    """测试用户训练服务层"""
    
    def setUp(self):
        """设置测试环境"""
        from backend.services import user_training_service
        user_training_service._user_training_service = None
    
    def test_get_user_overview(self):
        """测试获取用户训练概览"""
        from backend.services.user_training_service import UserTrainingService
        
        mock_repo = Mock()
        mock_repo.get_user_training_overview.return_value = {
            'active_count': 2,
            'completed_count': 45,
            'total_models': 12,
            'avg_accuracy': 87.5,
            'success_rate': 92.3,
            'total_training_hours': 156.5
        }
        
        service = UserTrainingService(repository=mock_repo)
        overview = service.get_user_overview(user_id='user123')
        
        # 验证结果
        self.assertEqual(overview.active_sessions, 2)
        self.assertEqual(overview.completed_sessions, 45)
        self.assertEqual(overview.total_models, 12)
        self.assertEqual(overview.avg_accuracy, 87.5)
        
        # 测试 to_dict
        result = overview.to_dict()
        self.assertEqual(result['activeSessions'], 2)
        self.assertEqual(result['completedSessions'], 45)
    
    def test_get_recent_sessions(self):
        """测试获取最近会话"""
        from backend.services.user_training_service import UserTrainingService
        
        mock_repo = Mock()
        mock_repo.get_recent_sessions.return_value = [
            {
                'session_id': 'sess_001',
                'name': 'BERT训练',
                'training_type': 'classification',
                'status': 'completed',
                'progress': 100,
                'accuracy': 0.92,
                'loss': 0.08,
                'created_at': datetime.utcnow(),
                'started_at': datetime.utcnow() - timedelta(hours=2),
                'completed_at': datetime.utcnow()
            },
            {
                'session_id': 'sess_002',
                'name': 'ResNet训练',
                'training_type': 'image',
                'status': 'running',
                'progress': 65,
                'accuracy': 0.85,
                'loss': 0.12,
                'created_at': datetime.utcnow() - timedelta(hours=3),
                'started_at': datetime.utcnow() - timedelta(hours=3),
                'completed_at': None
            }
        ]
        
        service = UserTrainingService(repository=mock_repo)
        sessions = service.get_recent_sessions(user_id='user123', limit=5)
        
        self.assertEqual(len(sessions), 2)
        self.assertEqual(sessions[0].id, 'sess_001')
        self.assertEqual(sessions[0].status, 'completed')
        self.assertEqual(sessions[1].status, 'running')
    
    def test_get_user_sessions_paginated(self):
        """测试获取会话列表（分页）"""
        from backend.services.user_training_service import UserTrainingService
        
        mock_repo = Mock()
        mock_repo.get_user_sessions.return_value = (
            [
                {
                    'session_id': 'sess_001',
                    'training_type': 'classification',
                    'status': 'completed',
                    'progress': 100,
                    'created_at': datetime.utcnow()
                }
            ],
            50  # 总数
        )
        
        service = UserTrainingService(repository=mock_repo)
        sessions, total = service.get_user_sessions(
            user_id='user123',
            page=1,
            limit=10,
            status='completed'
        )
        
        self.assertEqual(len(sessions), 1)
        self.assertEqual(total, 50)
    
    def test_get_user_statistics(self):
        """测试获取用户统计"""
        from backend.services.user_training_service import UserTrainingService
        
        mock_repo = Mock()
        mock_repo.get_user_statistics.return_value = {
            'total_count': 50,
            'completed_count': 45,
            'running_count': 2,
            'pending_count': 1,
            'failed_count': 2,
            'cancelled_count': 0,
            'success_rate': 90.0,
            'avg_training_time': 120.5,
            'total_training_time': 156.5,
            'avg_accuracy': 87.5,
            'best_accuracy': 95.2,
            'avg_loss': 0.125,
            'best_loss': 0.056
        }
        
        service = UserTrainingService(repository=mock_repo)
        stats = service.get_user_statistics(user_id='user123', days=30)
        
        self.assertEqual(stats.total_tasks, 50)
        self.assertEqual(stats.completed_tasks, 45)
        self.assertEqual(stats.success_rate, 90.0)
        
        # 测试 to_dict
        result = stats.to_dict()
        self.assertEqual(result['totalTasks'], 50)
        self.assertEqual(result['successRate'], 90.0)
    
    def test_get_training_trend(self):
        """测试获取训练趋势"""
        from backend.services.user_training_service import UserTrainingService
        
        mock_repo = Mock()
        mock_repo.get_training_trend.return_value = [
            {'date': '2026-01-10', 'completed': 5, 'running': 1, 'failed': 0, 'total': 6},
            {'date': '2026-01-11', 'completed': 8, 'running': 2, 'failed': 1, 'total': 11}
        ]
        
        service = UserTrainingService(repository=mock_repo)
        trend = service.get_training_trend(user_id='user123', days=7)
        
        self.assertEqual(len(trend), 2)
        self.assertEqual(trend[0].date, '2026-01-10')
        self.assertEqual(trend[0].completed, 5)
    
    def test_get_model_performance_ranking(self):
        """测试获取模型性能排行"""
        from backend.services.user_training_service import UserTrainingService
        
        mock_repo = Mock()
        mock_repo.get_model_performance_ranking.return_value = [
            {
                'model_id': 'model_001',
                'model_name': 'BERT-base',
                'model_type': 'classification',
                'best_accuracy': 0.95,
                'best_loss': 0.05,
                'training_count': 5,
                'last_trained': '2026-01-15T14:45:00'
            }
        ]
        
        service = UserTrainingService(repository=mock_repo)
        ranking = service.get_model_performance_ranking(user_id='user123', limit=10)
        
        self.assertEqual(len(ranking), 1)
        self.assertEqual(ranking[0].model_id, 'model_001')
        self.assertEqual(ranking[0].best_accuracy, 0.95)


class TestUserTrainingRepository(unittest.TestCase):
    """测试用户训练仓库层"""
    
    def test_repository_initialization(self):
        """测试仓库初始化"""
        from backend.repositories.user_training_repository import UserTrainingRepository
        
        mock_db_manager = Mock()
        repo = UserTrainingRepository(db_manager=mock_db_manager)
        
        self.assertIsNotNone(repo)
        self.assertEqual(repo._db_manager, mock_db_manager)
    
    def test_get_recent_sessions(self):
        """测试获取最近会话查询"""
        from backend.repositories.user_training_repository import UserTrainingRepository
        
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.order_by.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.first.return_value = None
        mock_query.all.return_value = []
        mock_session.query.return_value = mock_query
        mock_session.close = Mock()
        
        mock_db_manager = Mock()
        mock_db_manager.get_session.return_value = mock_session
        
        repo = UserTrainingRepository(db_manager=mock_db_manager)
        result = repo.get_recent_sessions(user_id='user123', limit=5)
        
        self.assertIsInstance(result, list)


class TestUserTrainingAPI(unittest.TestCase):
    """测试用户训练 API 端点"""
    
    def setUp(self):
        """设置测试环境"""
        from flask import Flask
        from flask_jwt_extended import JWTManager, create_access_token
        
        self.app = Flask(__name__)
        self.app.config['JWT_SECRET_KEY'] = 'test-secret-key'
        self.app.config['TESTING'] = True
        
        JWTManager(self.app)
        
        from backend.api.dashboard.user_training_api import user_training_bp
        self.app.register_blueprint(user_training_bp)
        
        self.client = self.app.test_client()
        
        with self.app.app_context():
            self.test_token = create_access_token(identity='test_user')
    
    def get_auth_headers(self):
        """获取认证请求头"""
        return {'Authorization': f'Bearer {self.test_token}'}
    
    @patch('backend.api.dashboard.user_training_api.get_user_training_service')
    def test_get_overview_endpoint(self, mock_get_service):
        """测试概览端点"""
        from backend.services.user_training_service import UserTrainingOverview
        
        mock_service = Mock()
        mock_service.get_user_overview.return_value = UserTrainingOverview(
            active_sessions=2,
            completed_sessions=45,
            total_models=12,
            avg_accuracy=87.5,
            success_rate=92.3,
            total_training_hours=156.5
        )
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/user/training/overview',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['activeSessions'], 2)
        self.assertEqual(data['data']['completedSessions'], 45)
    
    @patch('backend.api.dashboard.user_training_api.get_user_training_service')
    def test_get_recent_sessions_endpoint(self, mock_get_service):
        """测试最近会话端点"""
        from backend.services.user_training_service import TrainingSessionInfo
        
        mock_service = Mock()
        mock_service.get_recent_sessions.return_value = [
            TrainingSessionInfo(
                id='sess_001',
                name='BERT训练',
                model_type='classification',
                status='completed',
                progress=100,
                accuracy=0.92
            )
        ]
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/user/training/recent-sessions?limit=5',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(len(data['data']['recentSessions']), 1)
        self.assertEqual(data['data']['count'], 1)
    
    @patch('backend.api.dashboard.user_training_api.get_user_training_service')
    def test_get_sessions_paginated_endpoint(self, mock_get_service):
        """测试会话列表端点（分页）"""
        from backend.services.user_training_service import TrainingSessionInfo
        
        mock_service = Mock()
        mock_service.get_user_sessions.return_value = (
            [TrainingSessionInfo(id='sess_001', name='Test', status='completed')],
            50
        )
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/user/training/sessions?page=1&limit=10&status=completed',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['total'], 50)
        self.assertEqual(data['data']['page'], 1)
        self.assertEqual(data['data']['limit'], 10)
        self.assertEqual(data['data']['totalPages'], 5)
    
    @patch('backend.api.dashboard.user_training_api.get_user_training_service')
    def test_get_session_detail_endpoint(self, mock_get_service):
        """测试会话详情端点"""
        mock_service = Mock()
        mock_service.get_session_detail.return_value = {
            'session_id': 'sess_001',
            'user_id': 'test_user',
            'status': 'completed',
            'progress': 100,
            'config': {'epochs': 10}
        }
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/user/training/sessions/sess_001',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['session_id'], 'sess_001')
    
    @patch('backend.api.dashboard.user_training_api.get_user_training_service')
    def test_get_session_detail_not_found(self, mock_get_service):
        """测试会话详情不存在"""
        mock_service = Mock()
        mock_service.get_session_detail.return_value = None
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/user/training/sessions/nonexistent',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 404)
    
    @patch('backend.api.dashboard.user_training_api.get_user_training_service')
    def test_get_active_sessions_endpoint(self, mock_get_service):
        """测试活跃会话端点"""
        from backend.services.user_training_service import TrainingSessionInfo
        
        mock_service = Mock()
        mock_service.get_active_sessions.return_value = [
            TrainingSessionInfo(id='sess_001', name='Test', status='running', progress=65)
        ]
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/user/training/active',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['count'], 1)
        self.assertEqual(data['data']['runningCount'], 1)
    
    @patch('backend.api.dashboard.user_training_api.get_user_training_service')
    def test_get_statistics_endpoint(self, mock_get_service):
        """测试统计端点"""
        from backend.services.user_training_service import UserTrainingStatistics
        
        mock_service = Mock()
        mock_service.get_user_statistics.return_value = UserTrainingStatistics(
            total_tasks=50,
            completed_tasks=45,
            running_tasks=2,
            pending_tasks=1,
            failed_tasks=2,
            success_rate=90.0,
            avg_training_time=120.5
        )
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/user/training/statistics?days=30',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['totalTasks'], 50)
        self.assertEqual(data['data']['successRate'], 90.0)
    
    @patch('backend.api.dashboard.user_training_api.get_user_training_service')
    def test_get_trend_endpoint(self, mock_get_service):
        """测试趋势端点"""
        from backend.services.user_training_service import TrainingTrendData
        
        mock_service = Mock()
        mock_service.get_training_trend.return_value = [
            TrainingTrendData(date='2026-01-10', completed=5, running=1, failed=0, total=6),
            TrainingTrendData(date='2026-01-11', completed=8, running=2, failed=1, total=11)
        ]
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/user/training/trend?days=7',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(len(data['data']['trend']), 2)
        self.assertIn('summary', data['data'])
    
    @patch('backend.api.dashboard.user_training_api.get_user_training_service')
    def test_get_duration_stats_endpoint(self, mock_get_service):
        """测试时长统计端点"""
        mock_service = Mock()
        mock_service.get_training_duration_stats.return_value = {
            'avgDuration': 45.5,
            'minDuration': 10.2,
            'maxDuration': 180.5,
            'totalDuration': 156.5,
            'totalCount': 45,
            'distribution': [
                {'range': '0-30分钟', 'count': 10}
            ]
        }
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/user/training/duration-stats?days=30',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['avgDuration'], 45.5)
    
    @patch('backend.api.dashboard.user_training_api.get_user_training_service')
    def test_get_model_ranking_endpoint(self, mock_get_service):
        """测试模型排行端点"""
        from backend.services.user_training_service import ModelPerformance
        
        mock_service = Mock()
        mock_service.get_model_performance_ranking.return_value = [
            ModelPerformance(
                model_id='model_001',
                model_name='BERT-base',
                model_type='classification',
                best_accuracy=0.95,
                training_count=5
            )
        ]
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/user/training/model-ranking?limit=10',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(len(data['data']['ranking']), 1)
        self.assertEqual(data['data']['ranking'][0]['modelId'], 'model_001')
    
    def test_health_check_endpoint(self):
        """测试健康检查端点"""
        response = self.client.get('/api/v1/user/training/health')
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['status'], 'healthy')
    
    def test_unauthorized_access(self):
        """测试未授权访问"""
        endpoints = [
            '/api/v1/user/training/overview',
            '/api/v1/user/training/recent-sessions',
            '/api/v1/user/training/sessions',
            '/api/v1/user/training/active',
            '/api/v1/user/training/statistics',
            '/api/v1/user/training/trend',
            '/api/v1/user/training/duration-stats',
            '/api/v1/user/training/model-ranking'
        ]
        
        for endpoint in endpoints:
            response = self.client.get(endpoint)
            self.assertIn(response.status_code, [401, 422], f"Endpoint {endpoint} should require auth")


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_user_training_flow(self):
        """测试完整用户训练流程"""
        from backend.services.user_training_service import UserTrainingService, reset_user_training_service
        
        reset_user_training_service()
        
        mock_repo = Mock()
        
        # Mock 概览
        mock_repo.get_user_training_overview.return_value = {
            'active_count': 2,
            'completed_count': 45,
            'total_models': 12,
            'avg_accuracy': 87.5,
            'success_rate': 92.3,
            'total_training_hours': 156.5
        }
        
        # Mock 最近会话
        mock_repo.get_recent_sessions.return_value = [
            {
                'session_id': 'sess_001',
                'training_type': 'classification',
                'status': 'completed',
                'progress': 100,
                'created_at': datetime.utcnow()
            }
        ]
        
        # Mock 统计
        mock_repo.get_user_statistics.return_value = {
            'total_count': 50,
            'completed_count': 45,
            'running_count': 2,
            'pending_count': 1,
            'failed_count': 2,
            'cancelled_count': 0,
            'success_rate': 90.0,
            'avg_training_time': 120.5,
            'total_training_time': 156.5,
            'avg_accuracy': 87.5,
            'best_accuracy': 95.2,
            'avg_loss': 0.125,
            'best_loss': 0.056
        }
        
        # Mock 趋势
        mock_repo.get_training_trend.return_value = [
            {'date': '2026-01-10', 'completed': 5, 'running': 1, 'failed': 0, 'total': 6}
        ]
        
        service = UserTrainingService(repository=mock_repo)
        
        # 测试概览
        overview = service.get_user_overview(user_id='user123')
        self.assertEqual(overview.active_sessions, 2)
        self.assertEqual(overview.completed_sessions, 45)
        
        # 测试最近会话
        sessions = service.get_recent_sessions(user_id='user123', limit=5)
        self.assertEqual(len(sessions), 1)
        
        # 测试统计
        stats = service.get_user_statistics(user_id='user123', days=30)
        self.assertEqual(stats.total_tasks, 50)
        self.assertEqual(stats.success_rate, 90.0)
        
        # 测试趋势
        trend = service.get_training_trend(user_id='user123', days=7)
        self.assertEqual(len(trend), 1)
        
        print(f"\n用户训练测试结果:")
        print(f"  概览: 活跃 {overview.active_sessions}, 完成 {overview.completed_sessions}")
        print(f"  统计: 成功率 {stats.success_rate}%")
        print(f"  趋势: {len(trend)} 天数据")


class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def test_empty_data_handling(self):
        """测试空数据处理"""
        from backend.services.user_training_service import UserTrainingService
        
        mock_repo = Mock()
        mock_repo.get_user_training_overview.return_value = {
            'active_count': 0,
            'completed_count': 0,
            'total_models': 0,
            'avg_accuracy': 0.0,
            'success_rate': 0.0,
            'total_training_hours': 0.0
        }
        mock_repo.get_recent_sessions.return_value = []
        mock_repo.get_user_statistics.return_value = {
            'total_count': 0,
            'completed_count': 0,
            'success_rate': 0.0
        }
        
        service = UserTrainingService(repository=mock_repo)
        
        # 空概览
        overview = service.get_user_overview(user_id='user123')
        self.assertEqual(overview.active_sessions, 0)
        self.assertEqual(overview.success_rate, 0.0)
        
        # 空会话
        sessions = service.get_recent_sessions(user_id='user123')
        self.assertEqual(len(sessions), 0)
    
    def test_error_handling(self):
        """测试错误处理"""
        from backend.services.user_training_service import UserTrainingService
        
        mock_repo = Mock()
        mock_repo.get_user_training_overview.side_effect = Exception("Database error")
        
        service = UserTrainingService(repository=mock_repo)
        
        # 应该返回默认值而不是抛出异常
        overview = service.get_user_overview(user_id='user123')
        self.assertEqual(overview.active_sessions, 0)
        self.assertEqual(overview.completed_sessions, 0)
    
    def test_parameter_validation(self):
        """测试参数验证"""
        from flask import Flask
        from flask_jwt_extended import JWTManager, create_access_token
        
        app = Flask(__name__)
        app.config['JWT_SECRET_KEY'] = 'test-secret-key'
        app.config['TESTING'] = True
        
        JWTManager(app)
        
        from backend.api.dashboard.user_training_api import user_training_bp
        app.register_blueprint(user_training_bp)
        
        client = app.test_client()
        
        with app.app_context():
            token = create_access_token(identity='test_user')
        
        headers = {'Authorization': f'Bearer {token}'}
        
        with patch('backend.api.dashboard.user_training_api.get_user_training_service') as mock:
            from backend.services.user_training_service import UserTrainingStatistics
            mock.return_value.get_user_statistics.return_value = UserTrainingStatistics()
            
            # 测试 days 参数最大值限制
            response = client.get(
                '/api/v1/user/training/statistics?days=999',
                headers=headers
            )
            self.assertEqual(response.status_code, 200)
            # days 应该被限制为 365


if __name__ == '__main__':
    print("=" * 70)
    print("用户训练功能测试")
    print("=" * 70)
    
    unittest.main(verbosity=2)

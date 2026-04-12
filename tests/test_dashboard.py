"""仪表盘功能测试

测试 dashboard_service.py、dashboard_repository.py 和 dashboard_api.py 的功能。

测试覆盖：
1. DashboardService 服务层测试
2. DashboardRepository 数据仓库测试
3. Dashboard API 端点测试
4. 系统资源监控测试
5. 数据聚合测试
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from dataclasses import asdict

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDashboardModels(unittest.TestCase):
    """测试仪表盘数据模型"""
    
    def test_training_overview_model(self):
        """测试 TrainingOverview 模型"""
        from backend.schemas.dashboard_models import TrainingOverview
        
        overview = TrainingOverview(
            active_count=3,
            completed_count=45,
            failed_count=2,
            pending_count=1,
            paused_count=0,
            success_rate=0.9574,
            total_training_time_hours=156.5,
            avg_training_time_hours=3.26
        )
        
        # 验证属性
        self.assertEqual(overview.active_count, 3)
        self.assertEqual(overview.completed_count, 45)
        self.assertEqual(overview.failed_count, 2)
        self.assertAlmostEqual(overview.success_rate, 0.9574, places=4)
        
        # 验证 to_dict
        data = overview.to_dict()
        self.assertIn('active_count', data)
        self.assertIn('success_rate', data)
        self.assertEqual(data['active_count'], 3)
    
    def test_model_overview_model(self):
        """测试 ModelOverview 模型"""
        from backend.schemas.dashboard_models import ModelOverview
        
        overview = ModelOverview(
            total_count=15,
            deployed_count=5,
            draft_count=8,
            archived_count=2,
            avg_accuracy=0.8745,
            best_accuracy=0.9523
        )
        
        self.assertEqual(overview.total_count, 15)
        self.assertEqual(overview.deployed_count, 5)
        
        data = overview.to_dict()
        self.assertIn('avg_accuracy', data)
        self.assertAlmostEqual(data['avg_accuracy'], 0.8745, places=4)
    
    def test_system_resource_snapshot_model(self):
        """测试 SystemResourceSnapshot 模型"""
        from backend.schemas.dashboard_models import SystemResourceSnapshot
        
        snapshot = SystemResourceSnapshot(
            cpu_usage=45.5,
            cpu_count=8,
            memory_usage=62.3,
            memory_used_gb=12.5,
            memory_total_gb=32.0,
            disk_usage=55.2
        )
        
        self.assertAlmostEqual(snapshot.cpu_usage, 45.5, places=1)
        self.assertEqual(snapshot.cpu_count, 8)
        
        data = snapshot.to_dict()
        self.assertIn('timestamp', data)
        self.assertIn('cpu_usage', data)
    
    def test_gpu_resource_snapshot_model(self):
        """测试 GPUResourceSnapshot 模型"""
        from backend.schemas.dashboard_models import GPUResourceSnapshot
        
        gpu = GPUResourceSnapshot(
            device_id=0,
            name="NVIDIA RTX 3090",
            utilization=78.5,
            memory_used_gb=18.2,
            memory_total_gb=24.0,
            temperature=65.0
        )
        
        self.assertEqual(gpu.device_id, 0)
        self.assertEqual(gpu.name, "NVIDIA RTX 3090")
        self.assertAlmostEqual(gpu.utilization, 78.5, places=1)
        
        data = gpu.to_dict()
        self.assertIn('device_id', data)
        self.assertIn('name', data)
    
    def test_dashboard_filter_date_range(self):
        """测试 DashboardFilter 日期范围计算"""
        from backend.schemas.dashboard_models import DashboardFilter, DashboardTimeRange
        
        # 测试 LAST_24_HOURS
        filter_obj = DashboardFilter(time_range=DashboardTimeRange.LAST_24_HOURS)
        start_date, end_date = filter_obj.get_date_range()
        
        self.assertIsInstance(start_date, datetime)
        self.assertIsInstance(end_date, datetime)
        self.assertLess(start_date, end_date)
        
        # 验证时间差大约是24小时
        delta = end_date - start_date
        self.assertAlmostEqual(delta.total_seconds() / 3600, 24, delta=1)
        
        # 测试 LAST_7_DAYS
        filter_obj = DashboardFilter(time_range=DashboardTimeRange.LAST_7_DAYS)
        start_date, end_date = filter_obj.get_date_range()
        delta = end_date - start_date
        self.assertAlmostEqual(delta.total_seconds() / 86400, 7, delta=1)
    
    def test_dashboard_overview_aggregation(self):
        """测试 DashboardOverview 聚合模型"""
        from backend.schemas.dashboard_models import (
            DashboardOverview,
            TrainingOverview,
            ModelOverview,
            SystemResourceSnapshot
        )
        
        dashboard = DashboardOverview(
            training=TrainingOverview(active_count=3, completed_count=45),
            models=ModelOverview(total_count=15),
            system=SystemResourceSnapshot(cpu_usage=45.5),
            alerts_count=2
        )
        
        data = dashboard.to_dict()
        self.assertIn('training', data)
        self.assertIn('models', data)
        self.assertIn('system', data)
        self.assertEqual(data['alerts_count'], 2)


class TestDashboardService(unittest.TestCase):
    """测试仪表盘服务层"""
    
    def setUp(self):
        """设置测试环境"""
        # 重置服务单例
        from backend.services import dashboard_service
        dashboard_service._dashboard_service = None
    
    def test_get_service_singleton(self):
        """测试服务单例获取"""
        from backend.services.dashboard_service import get_dashboard_service
        
        service1 = get_dashboard_service()
        service2 = get_dashboard_service()
        
        self.assertIs(service1, service2)
    
    @patch('backend.services.dashboard_service.get_dashboard_repository')
    def test_get_dashboard_overview(self, mock_get_repo):
        """测试获取仪表盘概览"""
        from backend.services.dashboard_service import DashboardService
        from backend.schemas.dashboard_models import DashboardTimeRange
        
        # 模拟 repository
        mock_repo = Mock()
        mock_repo.get_training_overview.return_value = {
            'active_count': 3,
            'completed_count': 45,
            'failed_count': 2,
            'pending_count': 1,
            'paused_count': 0,
            'success_rate': 0.9574,
            'total_training_time_hours': 156.5,
            'avg_training_time_hours': 3.26
        }
        mock_repo.get_model_overview.return_value = {
            'total_count': 15,
            'deployed_count': 5,
            'draft_count': 8,
            'archived_count': 2,
            'avg_accuracy': 0.8745,
            'best_accuracy': 0.9523,
            'avg_f1_score': 0.8612,
            'total_size_gb': 12.5
        }
        mock_repo.get_user_activity_summary.return_value = {
            'total_training_count': 48,
            'total_models_created': 15,
            'total_datasets_used': 8,
            'last_active_at': None,
            'most_used_model_type': 'classification',
            'avg_training_time_hours': 3.26
        }
        mock_repo.get_active_alerts_count.return_value = 2
        mock_get_repo.return_value = mock_repo
        
        # 创建服务
        service = DashboardService(repository=mock_repo)
        
        # 调用方法
        overview = service.get_dashboard_overview(
            user_id='user123',
            tenant_id='tenant1',
            time_range=DashboardTimeRange.LAST_24_HOURS
        )
        
        # 验证结果
        self.assertEqual(overview.training.active_count, 3)
        self.assertEqual(overview.training.completed_count, 45)
        self.assertEqual(overview.models.total_count, 15)
        self.assertEqual(overview.alerts_count, 2)
        
        # 验证 repository 被调用
        mock_repo.get_training_overview.assert_called_once()
        mock_repo.get_model_overview.assert_called_once()
    
    @patch('backend.services.dashboard_service.psutil')
    def test_get_system_resources(self, mock_psutil):
        """测试获取系统资源"""
        from backend.services.dashboard_service import DashboardService
        
        # 模拟 psutil 返回值
        mock_psutil.cpu_percent.return_value = 45.5
        mock_psutil.cpu_count.return_value = 8
        
        mock_memory = Mock()
        mock_memory.percent = 62.3
        mock_memory.used = 12.5 * (1024 ** 3)
        mock_memory.total = 32.0 * (1024 ** 3)
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.percent = 55.2
        mock_disk.used = 256.5 * (1024 ** 3)
        mock_disk.total = 512.0 * (1024 ** 3)
        mock_psutil.disk_usage.return_value = mock_disk
        
        mock_net = Mock()
        mock_net.bytes_sent = 1024.5 * (1024 ** 2)
        mock_net.bytes_recv = 2048.3 * (1024 ** 2)
        mock_psutil.net_io_counters.return_value = mock_net
        
        # 创建服务
        service = DashboardService(repository=Mock())
        
        # 调用方法
        snapshot = service._get_current_system_resources()
        
        # 验证结果
        self.assertAlmostEqual(snapshot.cpu_usage, 45.5, places=1)
        self.assertEqual(snapshot.cpu_count, 8)
        self.assertAlmostEqual(snapshot.memory_usage, 62.3, places=1)
        self.assertAlmostEqual(snapshot.disk_usage, 55.2, places=1)
    
    @patch('backend.services.dashboard_service.get_dashboard_repository')
    def test_get_training_detailed_stats(self, mock_get_repo):
        """测试获取详细训练统计"""
        from backend.services.dashboard_service import DashboardService
        from backend.schemas.dashboard_models import DashboardTimeRange, MetricGranularity
        
        # 模拟 repository
        mock_repo = Mock()
        mock_repo.get_training_overview.return_value = {
            'active_count': 3,
            'completed_count': 45,
            'failed_count': 2,
            'pending_count': 1,
            'paused_count': 0,
            'success_rate': 0.9574,
            'total_training_time_hours': 156.5,
            'avg_training_time_hours': 3.26
        }
        mock_repo.get_training_trends.return_value = [
            {
                'timestamp': '2026-01-10T00:00:00',
                'date_label': '2026-01-10',
                'count': 8,
                'success_count': 7,
                'failed_count': 1,
                'avg_duration_hours': 2.5
            }
        ]
        mock_repo.get_training_by_type.return_value = {
            'classification': 25,
            'regression': 15
        }
        mock_repo.get_recent_training_sessions.return_value = []
        mock_get_repo.return_value = mock_repo
        
        # 创建服务
        service = DashboardService(repository=mock_repo)
        
        # 调用方法
        stats = service.get_training_detailed_stats(
            user_id='user123',
            time_range=DashboardTimeRange.LAST_7_DAYS,
            granularity=MetricGranularity.DAY
        )
        
        # 验证结果
        self.assertEqual(stats.overview.active_count, 3)
        self.assertEqual(len(stats.trends), 1)
        self.assertIn('classification', stats.by_type)
    
    @patch('backend.services.dashboard_service.get_dashboard_repository')
    def test_get_model_stats(self, mock_get_repo):
        """测试获取模型统计"""
        from backend.services.dashboard_service import DashboardService
        
        mock_repo = Mock()
        mock_repo.get_model_overview.return_value = {
            'total_count': 15,
            'deployed_count': 5,
            'draft_count': 8,
            'archived_count': 2,
            'avg_accuracy': 0.8745,
            'best_accuracy': 0.9523,
            'avg_f1_score': 0.8612,
            'total_size_gb': 12.5
        }
        mock_repo.get_model_distribution.return_value = {
            'by_type': {'classification': 8},
            'by_framework': {'pytorch': 10},
            'by_status': {'deployed': 5},
            'by_category': {'image': 6}
        }
        mock_repo.get_top_models.return_value = [
            {'id': 'model1', 'name': 'Test Model', 'accuracy': 0.95}
        ]
        mock_get_repo.return_value = mock_repo
        
        service = DashboardService(repository=mock_repo)
        stats = service.get_model_stats(user_id='user123')
        
        self.assertIn('overview', stats)
        self.assertIn('distribution', stats)
        self.assertIn('top_models', stats)
        self.assertEqual(stats['overview']['total_count'], 15)


class TestDashboardRepository(unittest.TestCase):
    """测试仪表盘数据仓库"""
    
    def test_get_training_overview_empty(self):
        """测试空数据库的训练概览"""
        from backend.repositories.dashboard_repository import DashboardRepository
        
        # 模拟数据库会话
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.group_by.return_value = mock_query
        mock_query.all.return_value = []
        mock_query.first.return_value = None
        mock_session.query.return_value = mock_query
        mock_session.close = Mock()
        
        mock_db_manager = Mock()
        mock_db_manager.get_session.return_value = mock_session
        
        repo = DashboardRepository(db_manager=mock_db_manager)
        result = repo.get_training_overview()
        
        # 验证返回默认值
        self.assertEqual(result['active_count'], 0)
        self.assertEqual(result['completed_count'], 0)
        self.assertEqual(result['success_rate'], 0.0)
    
    def test_get_model_overview_empty(self):
        """测试空数据库的模型概览"""
        from backend.repositories.dashboard_repository import DashboardRepository
        
        mock_session = Mock()
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.group_by.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.all.return_value = []
        mock_session.close = Mock()
        
        # 模拟 metrics 查询结果
        mock_metrics_result = Mock()
        mock_metrics_result.avg_accuracy = None
        mock_metrics_result.best_accuracy = None
        mock_metrics_result.avg_f1_score = None
        mock_metrics_result.total_size_mb = None
        mock_query.first.return_value = mock_metrics_result
        
        mock_session.query.return_value = mock_query
        
        mock_db_manager = Mock()
        mock_db_manager.get_session.return_value = mock_session
        
        repo = DashboardRepository(db_manager=mock_db_manager)
        result = repo.get_model_overview()
        
        self.assertEqual(result['total_count'], 0)
        self.assertEqual(result['avg_accuracy'], 0.0)


class TestDashboardAPI(unittest.TestCase):
    """测试仪表盘 API 端点"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建测试应用
        from flask import Flask
        from flask_jwt_extended import JWTManager, create_access_token
        
        self.app = Flask(__name__)
        self.app.config['JWT_SECRET_KEY'] = 'test-secret-key'
        self.app.config['TESTING'] = True
        
        JWTManager(self.app)
        
        # 注册蓝图
        from backend.api.dashboard.dashboard_api import dashboard_bp
        self.app.register_blueprint(dashboard_bp)
        
        self.client = self.app.test_client()
        
        # 创建测试 token
        with self.app.app_context():
            self.test_token = create_access_token(identity='test_user')
    
    def get_auth_headers(self):
        """获取认证请求头"""
        return {'Authorization': f'Bearer {self.test_token}'}
    
    def test_health_endpoint(self):
        """测试健康检查端点"""
        response = self.client.get('/api/v1/dashboard/health')
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['status'], 'healthy')
    
    @patch('backend.api.dashboard.dashboard_api.get_dashboard_service')
    def test_get_overview_endpoint(self, mock_get_service):
        """测试概览端点"""
        from backend.schemas.dashboard_models import (
            DashboardOverview,
            TrainingOverview,
            ModelOverview,
            SystemResourceSnapshot
        )
        
        # 模拟服务返回
        mock_service = Mock()
        mock_service.get_dashboard_overview.return_value = DashboardOverview(
            training=TrainingOverview(active_count=3, completed_count=45),
            models=ModelOverview(total_count=15),
            system=SystemResourceSnapshot(cpu_usage=45.5),
            alerts_count=2
        )
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/dashboard/overview?time_range=last_24_hours',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('overview', data['data'])
        self.assertEqual(data['data']['overview']['training']['active_count'], 3)
    
    @patch('backend.api.dashboard.dashboard_api.get_dashboard_service')
    def test_get_training_stats_endpoint(self, mock_get_service):
        """测试训练统计端点"""
        from backend.schemas.dashboard_models import (
            TrainingDetailedStats,
            TrainingOverview,
            TrainingTrend
        )
        
        mock_service = Mock()
        mock_service.get_training_detailed_stats.return_value = TrainingDetailedStats(
            overview=TrainingOverview(active_count=3),
            trends=[
                TrainingTrend(
                    timestamp=datetime.utcnow(),
                    date_label='2026-01-16',
                    count=8
                )
            ],
            by_type={'classification': 25}
        )
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/dashboard/training-stats?time_range=last_7_days&granularity=day',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('stats', data['data'])
    
    @patch('backend.api.dashboard.dashboard_api.get_dashboard_service')
    def test_get_system_metrics_endpoint(self, mock_get_service):
        """测试系统指标端点"""
        from backend.schemas.dashboard_models import SystemMetricsHistory, SystemResourceSnapshot
        
        mock_service = Mock()
        mock_service.get_system_metrics_history.return_value = SystemMetricsHistory(
            cpu_usage=[{'timestamp': '2026-01-16T00:00:00', 'value': 45.5}],
            memory_usage=[{'timestamp': '2026-01-16T00:00:00', 'value': 62.3}]
        )
        mock_service.get_current_system_snapshot.return_value = SystemResourceSnapshot(
            cpu_usage=45.5,
            memory_usage=62.3
        )
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/dashboard/system-metrics?hours=24',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('metrics', data['data'])
        self.assertIn('current', data['data'])
    
    @patch('backend.api.dashboard.dashboard_api.get_dashboard_service')
    def test_get_model_stats_endpoint(self, mock_get_service):
        """测试模型统计端点"""
        mock_service = Mock()
        mock_service.get_model_stats.return_value = {
            'overview': {'total_count': 15},
            'distribution': {'by_type': {'classification': 8}},
            'top_models': [{'id': 'model1', 'name': 'Test', 'accuracy': 0.95}]
        }
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/dashboard/model-stats',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertIn('stats', data['data'])
    
    @patch('backend.api.dashboard.dashboard_api.get_dashboard_service')
    def test_get_gpu_status_endpoint(self, mock_get_service):
        """测试 GPU 状态端点"""
        from backend.schemas.dashboard_models import GPUResourceSnapshot
        
        mock_service = Mock()
        mock_service.get_current_gpu_snapshot.return_value = [
            GPUResourceSnapshot(
                device_id=0,
                name='NVIDIA RTX 3090',
                utilization=78.5
            )
        ]
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/dashboard/gpu-status',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['data']['gpu_count'], 1)
        self.assertTrue(data['data']['gpu_available'])
    
    @patch('backend.api.dashboard.dashboard_api.get_dashboard_service')
    def test_get_gpu_status_no_gpu(self, mock_get_service):
        """测试无 GPU 的状态端点"""
        mock_service = Mock()
        mock_service.get_current_gpu_snapshot.return_value = []
        mock_get_service.return_value = mock_service
        
        response = self.client.get(
            '/api/v1/dashboard/gpu-status',
            headers=self.get_auth_headers()
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['data']['gpu_count'], 0)
        self.assertFalse(data['data']['gpu_available'])
    
    def test_unauthorized_access(self):
        """测试未授权访问"""
        response = self.client.get('/api/v1/dashboard/overview')
        
        # 应该返回 401 或 422（缺少 token）
        self.assertIn(response.status_code, [401, 422])


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_dashboard_flow(self):
        """测试完整仪表盘流程"""
        from backend.services.dashboard_service import DashboardService, reset_dashboard_service
        from backend.schemas.dashboard_models import DashboardTimeRange
        
        # 重置服务
        reset_dashboard_service()
        
        # 使用模拟的 repository
        mock_repo = Mock()
        mock_repo.get_training_overview.return_value = {
            'active_count': 3,
            'completed_count': 45,
            'failed_count': 2,
            'pending_count': 1,
            'paused_count': 0,
            'success_rate': 0.9574,
            'total_training_time_hours': 156.5,
            'avg_training_time_hours': 3.26
        }
        mock_repo.get_model_overview.return_value = {
            'total_count': 15,
            'deployed_count': 5,
            'draft_count': 8,
            'archived_count': 2,
            'avg_accuracy': 0.8745,
            'best_accuracy': 0.9523,
            'avg_f1_score': 0.8612,
            'total_size_gb': 12.5
        }
        mock_repo.get_user_activity_summary.return_value = {
            'total_training_count': 48,
            'total_models_created': 15,
            'total_datasets_used': 8,
            'last_active_at': None,
            'most_used_model_type': 'classification',
            'avg_training_time_hours': 3.26
        }
        mock_repo.get_active_alerts_count.return_value = 2
        
        # 创建服务
        service = DashboardService(repository=mock_repo)
        
        # 测试获取概览
        overview = service.get_dashboard_overview(
            user_id='user123',
            time_range=DashboardTimeRange.LAST_24_HOURS
        )
        
        # 验证概览数据
        self.assertIsNotNone(overview)
        self.assertEqual(overview.training.active_count, 3)
        self.assertEqual(overview.models.total_count, 15)
        
        # 测试系统资源（真实获取）
        system_snapshot = service.get_current_system_snapshot()
        self.assertIsNotNone(system_snapshot)
        self.assertGreaterEqual(system_snapshot.cpu_usage, 0)
        self.assertLessEqual(system_snapshot.cpu_usage, 100)
        self.assertGreater(system_snapshot.memory_total_gb, 0)
        
        print(f"\n系统资源快照:")
        print(f"  CPU 使用率: {system_snapshot.cpu_usage:.1f}%")
        print(f"  内存使用率: {system_snapshot.memory_usage:.1f}%")
        print(f"  磁盘使用率: {system_snapshot.disk_usage:.1f}%")


if __name__ == '__main__':
    print("=" * 70)
    print("仪表盘功能测试")
    print("=" * 70)
    
    # 运行测试
    unittest.main(verbosity=2)

"""测试用户训练信息API接口

验证用户训练信息API接口是否正常工作。
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class TestUserTrainingAPI(unittest.TestCase):
    """测试用户训练信息API接口"""
    
    def setUp(self):
        """测试前准备"""
        # 创建模拟的Flask应用和客户端
        from backend.app import create_app
        self.app = create_app()
        self.client = self.app.test_client()
        
        # 创建模拟的JWT令牌
        with self.app.app_context():
            from flask_jwt_extended import create_access_token
            self.test_user_id = "test_user_123"
            self.access_token = create_access_token(identity=self.test_user_id)
            self.auth_header = {'Authorization': f'Bearer {self.access_token}'}
    
    @patch('backend.modules.dashboard.api.user_training_api.get_database_manager')
    def test_get_user_training_overview(self, mock_db_manager):
        """测试获取用户训练概览接口"""
        # 模拟数据库管理器和查询结果
        mock_db_session = MagicMock()
        mock_db_manager.return_value.get_db_session.return_value.__enter__.return_value = mock_db_session
        
        # 模拟训练会话查询结果
        mock_session = MagicMock()
        mock_session.status = 'completed'
        mock_db_session.query.return_value.filter.return_value.count.return_value = 5
        
        # 发送GET请求
        response = self.client.get('/api/v1/user/training/overview', headers=self.auth_header)
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应数据
        data = response.get_json()
        self.assertIn('data', data)
        self.assertIn('activeSessions', data['data'])
        self.assertIn('completedSessions', data['data'])
        self.assertIn('totalModels', data['data'])
        self.assertIn('avgAccuracy', data['data'])
    
    @patch('backend.modules.dashboard.api.user_training_api.get_database_manager')
    def test_get_recent_training_sessions(self, mock_db_manager):
        """测试获取最近训练会话接口"""
        # 模拟数据库管理器和查询结果
        mock_db_session = MagicMock()
        mock_db_manager.return_value.get_db_session.return_value.__enter__.return_value = mock_db_session
        
        # 模拟训练会话查询结果
        mock_session = MagicMock()
        mock_session.id = "session_123"
        mock_session.status = 'completed'
        mock_session.created_at.isoformat.return_value = "2023-01-01T00:00:00"
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_session]
        
        # 发送GET请求
        response = self.client.get('/api/v1/user/training/recent-sessions', headers=self.auth_header)
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应数据
        data = response.get_json()
        self.assertIn('data', data)
        self.assertIn('recentSessions', data['data'])
    
    @patch('backend.modules.dashboard.api.user_training_api.get_database_manager')
    def test_get_user_training_sessions(self, mock_db_manager):
        """测试获取用户训练会话列表接口"""
        # 模拟数据库管理器和查询结果
        mock_db_session = MagicMock()
        mock_db_manager.return_value.get_db_session.return_value.__enter__.return_value = mock_db_session
        
        # 模拟训练会话查询结果
        mock_session = MagicMock()
        mock_session.id = "session_123"
        mock_session.status = 'completed'
        mock_session.created_at.isoformat.return_value = "2023-01-01T00:00:00"
        mock_db_session.query.return_value.filter.return_value.count.return_value = 1
        mock_db_session.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [mock_session]
        
        # 发送GET请求
        response = self.client.get('/api/v1/user/training/sessions', headers=self.auth_header)
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应数据
        data = response.get_json()
        self.assertIn('data', data)
        self.assertIn('sessions', data['data'])
        self.assertIn('total', data['data'])
        self.assertIn('page', data['data'])
        self.assertIn('limit', data['data'])
    
    @patch('backend.modules.dashboard.api.user_training_api.get_database_manager')
    def test_get_user_training_statistics(self, mock_db_manager):
        """测试获取用户训练统计信息接口"""
        # 模拟数据库管理器和查询结果
        mock_db_session = MagicMock()
        mock_db_manager.return_value.get_db_session.return_value.__enter__.return_value = mock_db_session
        
        # 模拟训练会话查询结果
        mock_db_session.query.return_value.filter.return_value.count.return_value = 5
        
        # 发送GET请求
        response = self.client.get('/api/v1/user/training/statistics', headers=self.auth_header)
        
        # 验证响应状态码
        self.assertEqual(response.status_code, 200)
        
        # 验证响应数据
        data = response.get_json()
        self.assertIn('data', data)
        self.assertIn('totalTasks', data['data'])
        self.assertIn('completedTasks', data['data'])
        self.assertIn('runningTasks', data['data'])
        self.assertIn('failedTasks', data['data'])
        self.assertIn('successRate', data['data'])
        self.assertIn('avgTrainingTime', data['data'])

if __name__ == '__main__':
    unittest.main()
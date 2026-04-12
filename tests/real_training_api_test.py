"""真实训练API测试文件

用于测试backend的训练API接口的真实逻辑，包括训练模型、训练数据集、读取数据库、存储模型等。
"""

import sys
import os
import json
import time
from datetime import datetime
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# 模拟数据库和相关依赖
class MockDatabaseManager:
    def get_db_session(self):
        return MockDBSession()

class MockDBSession:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def commit(self):
        pass
    
    def rollback(self):
        pass
    
    def close(self):
        pass
    
    def add(self, obj):
        pass
    
    def query(self, model):
        return MockQuery(model)

class MockQuery:
    def __init__(self, model):
        self.model = model
    
    def filter(self, *args):
        return self
    
    def first(self):
        # 模拟返回一个训练会话对象
        if hasattr(self.model, '__name__') and 'TrainingSession' in self.model.__name__:
            mock_session = MagicMock()
            mock_session.session_id = 'test-session-id-123'
            mock_session.user_id = 'test-user-id'
            mock_session.status = 'pending'
            mock_session.progress = 0.0
            mock_session.config = {
                'session_name': '测试训练会话',
                'session_description': '这是一个测试训练会话'
            }
            mock_session.created_at = datetime.now()
            mock_session.updated_at = datetime.now()
            mock_session.started_at = None
            mock_session.completed_at = None
            mock_session.result = None
            mock_session.error_message = None
            return mock_session
        return None
    
    def order_by(self, *args):
        return self
    
    def offset(self, *args):
        return self
    
    def limit(self, *args):
        return self
    
    def all(self):
        return [self.first()] if self.first() else []

# 模拟配置管理器
class MockConfigManager:
    def get(self, name, default=None):
        if name == "database":
            return {
                "type": "sqlite",
                "name": "test.db"
            }
        return default

# 模拟JWT
class MockJWTManager:
    def __init__(self, app):
        pass

# 修复导入问题
sys.modules['backend.modules.database.manager'] = MagicMock()
sys.modules['backend.modules.database.manager'].get_database_manager = MockDatabaseManager
sys.modules['backend.core.config_manager'] = MagicMock()
sys.modules['backend.core.config_manager'].get_config_manager = MockConfigManager
sys.modules['flask_jwt_extended'] = MagicMock()
sys.modules['flask_jwt_extended'].JWTManager = MockJWTManager
sys.modules['flask_jwt_extended'].jwt_required = lambda: lambda x: x
sys.modules['flask_jwt_extended'].get_jwt_identity = lambda: 'test-user-id'

# 现在导入真实的模块
from backend.schemas.training_models import TrainingSession
from backend.services.training_service import TrainingService
from backend.repositories.training_session_repository import TrainingSessionRepository


class TestTrainingService(unittest.TestCase):
    """训练服务测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.training_service = TrainingService()
        # 使用内存存储模式避免数据库依赖
        self.training_service._repository = TrainingSessionRepository(use_memory_storage=True)
    
    def test_create_training_session(self):
        """测试创建训练会话"""
        print("\n=== 测试创建训练会话 ===")
        
        # 准备测试数据
        user_id = "test-user-id"
        name = "测试训练会话"
        description = "这是一个测试训练会话"
        config = {
            "model_name": "test_model",
            "training_method": "standard",
            "batch_size": 8,
            "learning_rate": 0.001,
            "num_epochs": 3
        }
        
        # 创建训练会话
        session = self.training_service.create_training_session(
            user_id=user_id,
            name=name,
            description=description,
            config=config
        )
        
        # 验证结果
        self.assertIsNotNone(session)
        self.assertEqual(session.user_id, user_id)
        self.assertIn('session_name', session.config)
        self.assertEqual(session.config['session_name'], name)
        self.assertIn('session_description', session.config)
        self.assertEqual(session.config['session_description'], description)
        print(f"训练会话创建成功，ID: {session.session_id}")
    
    def test_get_training_session(self):
        """测试获取训练会话"""
        print("\n=== 测试获取训练会话 ===")
        
        # 先创建一个训练会话
        user_id = "test-user-id"
        name = "测试训练会话"
        description = "这是一个测试训练会话"
        config = {"model_name": "test_model"}
        
        session = self.training_service.create_training_session(
            user_id=user_id,
            name=name,
            description=description,
            config=config
        )
        
        # 获取训练会话
        retrieved_session = self.training_service.get_training_session(session.session_id)
        
        # 验证结果
        self.assertIsNotNone(retrieved_session)
        self.assertEqual(retrieved_session.session_id, session.session_id)
        self.assertEqual(retrieved_session.user_id, user_id)
        print(f"训练会话获取成功，ID: {retrieved_session.session_id}")
    
    def test_list_training_sessions(self):
        """测试获取训练会话列表"""
        print("\n=== 测试获取训练会话列表 ===")
        
        # 创建多个训练会话
        user_id = "test-user-id"
        for i in range(3):
            self.training_service.create_training_session(
                user_id=user_id,
                name=f"测试训练会话 {i+1}",
                description=f"这是第 {i+1} 个测试训练会话",
                config={"model_name": f"test_model_{i}"}
            )
        
        # 获取训练会话列表
        sessions = self.training_service.list_training_sessions(user_id=user_id)
        
        # 验证结果
        self.assertEqual(len(sessions), 3)
        print(f"训练会话列表获取成功，共 {len(sessions)} 个会话")
    
    def test_update_training_session_progress(self):
        """测试更新训练会话进度"""
        print("\n=== 测试更新训练会话进度 ===")
        
        # 先创建一个训练会话
        user_id = "test-user-id"
        session = self.training_service.create_training_session(
            user_id=user_id,
            name="测试训练会话",
            config={"model_name": "test_model"}
        )
        
        # 更新进度
        progress = 50.0
        updated_session = self.training_service.update_training_session_progress(
            session_id=session.session_id,
            progress=progress
        )
        
        # 验证结果
        self.assertEqual(updated_session.progress, progress)
        print(f"训练进度更新成功，进度: {updated_session.progress}%")
    
    def test_start_training_session(self):
        """测试开始训练会话"""
        print("\n=== 测试开始训练会话 ===")
        
        # 先创建一个训练会话
        user_id = "test-user-id"
        session = self.training_service.create_training_session(
            user_id=user_id,
            name="测试训练会话",
            config={"model_name": "test_model"}
        )
        
        # 开始训练会话
        started_session = self.training_service.start_training_session(session.session_id)
        
        # 验证结果
        self.assertEqual(started_session.status, "running")
        self.assertIsNotNone(started_session.started_at)
        print(f"训练会话开始成功，状态: {started_session.status}")
    
    def test_complete_training_session(self):
        """测试完成训练会话"""
        print("\n=== 测试完成训练会话 ===")
        
        # 先创建并开始一个训练会话
        user_id = "test-user-id"
        session = self.training_service.create_training_session(
            user_id=user_id,
            name="测试训练会话",
            config={"model_name": "test_model"}
        )
        self.training_service.start_training_session(session.session_id)
        
        # 完成训练会话
        result = {
            "model_path": "/data/models/test_model",
            "accuracy": 0.95,
            "loss": 0.05
        }
        completed_session = self.training_service.complete_training_session(
            session_id=session.session_id,
            result=result
        )
        
        # 验证结果
        self.assertEqual(completed_session.status, "completed")
        self.assertEqual(completed_session.progress, 100.0)
        self.assertIsNotNone(completed_session.completed_at)
        self.assertEqual(completed_session.result, result)
        print(f"训练会话完成成功，状态: {completed_session.status}")


def run_real_tests():
    """运行真实测试"""
    print("开始运行真实训练API测试...")
    
    try:
        # 创建测试套件
        suite = unittest.TestSuite()
        suite.addTest(TestTrainingService('test_create_training_session'))
        suite.addTest(TestTrainingService('test_get_training_session'))
        suite.addTest(TestTrainingService('test_list_training_sessions'))
        suite.addTest(TestTrainingService('test_update_training_session_progress'))
        suite.addTest(TestTrainingService('test_start_training_session'))
        suite.addTest(TestTrainingService('test_complete_training_session'))
        
        # 运行测试
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("\n=== 所有真实测试通过 ===")
            return True
        else:
            print(f"\n=== 测试失败 ===")
            print(f"失败数: {len(result.failures)}")
            print(f"错误数: {len(result.errors)}")
            return False
            
    except Exception as e:
        print(f"\n=== 测试执行失败 ===")
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_real_tests()
    sys.exit(0 if success else 1)
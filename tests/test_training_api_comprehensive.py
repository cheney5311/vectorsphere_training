"""全面测试训练API接口

测试backend/modules/training下的API接口，包括创建、获取、更新训练会话等功能。
"""

import sys
import os
import json
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestTrainingAPIComprehensive(unittest.TestCase):
    """全面测试训练API接口"""
    
    def setUp(self):
        """测试前准备"""
        # 创建应用
        from app import create_app
        self.app = create_app()
        self.client = self.app.test_client()
        
        # 创建测试用户和JWT令牌
        with self.app.app_context():
            from flask_jwt_extended import create_access_token
            self.test_user_id = "test_user_123"
            self.access_token = create_access_token(identity=self.test_user_id)
            self.auth_header = {'Authorization': f'Bearer {self.access_token}'}
    
    def test_create_training_session(self):
        """测试创建训练会话"""
        print("测试创建训练会话...")
        
        # 准备测试数据
        test_data = {
            "name": "测试训练会话",
            "description": "这是一个测试训练会话",
            "config": {
                "model_name": "test_model",
                "training_method": "standard",
                "epochs": 3,
                "batch_size": 16
            }
        }
        
        # 发送POST请求
        response = self.client.post(
            '/api/v1/training/sessions',
            data=json.dumps(test_data),
            content_type='application/json',
            headers=self.auth_header
        )
        
        # 检查响应
        print(f"状态码: {response.status_code}")
        response_data = response.get_json()
        print(f"响应数据: {response_data}")
        
        # 验证响应
        self.assertEqual(response.status_code, 201)
        self.assertIn('data', response_data)
        self.assertIn('id', response_data['data'])
        self.assertEqual(response_data['data']['name'], test_data['name'])
        self.assertEqual(response_data['data']['description'], test_data['description'])
        
        # 保存会话ID用于后续测试
        self.session_id = response_data['data']['id']
    
    def test_get_training_session(self):
        """测试获取训练会话"""
        print("测试获取训练会话...")
        
        # 首先创建一个训练会话
        test_data = {
            "name": "测试训练会话2",
            "description": "用于获取测试的训练会话",
            "config": {
                "model_name": "test_model_2",
                "training_method": "standard"
            }
        }
        
        create_response = self.client.post(
            '/api/v1/training/sessions',
            data=json.dumps(test_data),
            content_type='application/json',
            headers=self.auth_header
        )
        
        self.assertEqual(create_response.status_code, 201)
        session_id = create_response.get_json()['data']['id']
        
        # 获取训练会话
        response = self.client.get(
            f'/api/v1/training/sessions/{session_id}',
            headers=self.auth_header
        )
        
        # 检查响应
        print(f"状态码: {response.status_code}")
        response_data = response.get_json()
        print(f"响应数据: {response_data}")
        
        # 验证响应
        self.assertEqual(response.status_code, 200)
        self.assertIn('data', response_data)
        self.assertEqual(response_data['data']['id'], session_id)
        self.assertEqual(response_data['data']['name'], test_data['name'])
    
    def test_list_training_sessions(self):
        """测试获取训练会话列表"""
        print("测试获取训练会话列表...")
        
        # 创建多个训练会话
        for i in range(3):
            test_data = {
                "name": f"测试训练会话列表{i}",
                "description": f"用于列表测试的训练会话{i}",
                "config": {
                    "model_name": f"test_model_{i}",
                    "training_method": "standard"
                }
            }
            
            create_response = self.client.post(
                '/api/v1/training/sessions',
                data=json.dumps(test_data),
                content_type='application/json',
                headers=self.auth_header
            )
            
            self.assertEqual(create_response.status_code, 201)
        
        # 获取训练会话列表
        response = self.client.get(
            '/api/v1/training/sessions',
            headers=self.auth_header
        )
        
        # 检查响应
        print(f"状态码: {response.status_code}")
        response_data = response.get_json()
        print(f"响应数据: {response_data}")
        
        # 验证响应
        self.assertEqual(response.status_code, 200)
        self.assertIn('data', response_data)
        self.assertIsInstance(response_data['data'], list)
        # 应该至少有3个会话
        self.assertGreaterEqual(len(response_data['data']), 3)
    
    def test_start_training_session(self):
        """测试开始训练会话"""
        print("测试开始训练会话...")
        
        # 首先创建一个训练会话
        test_data = {
            "name": "测试开始训练会话",
            "description": "用于开始测试的训练会话",
            "config": {
                "model_name": "test_model_start",
                "training_method": "standard"
            }
        }
        
        create_response = self.client.post(
            '/api/v1/training/sessions',
            data=json.dumps(test_data),
            content_type='application/json',
            headers=self.auth_header
        )
        
        self.assertEqual(create_response.status_code, 201)
        session_id = create_response.get_json()['data']['id']
        
        # 开始训练会话
        response = self.client.post(
            f'/api/v1/training/sessions/{session_id}/start',
            headers=self.auth_header
        )
        
        # 检查响应
        print(f"状态码: {response.status_code}")
        response_data = response.get_json()
        print(f"响应数据: {response_data}")
        
        # 验证响应
        self.assertEqual(response.status_code, 200)
        self.assertIn('data', response_data)
        self.assertEqual(response_data['data']['id'], session_id)
        self.assertEqual(response_data['data']['status'], 'running')
    
    def test_update_training_progress(self):
        """测试更新训练进度"""
        print("测试更新训练进度...")
        
        # 首先创建并开始一个训练会话
        test_data = {
            "name": "测试进度更新训练会话",
            "description": "用于进度更新测试的训练会话",
            "config": {
                "model_name": "test_model_progress",
                "training_method": "standard"
            }
        }
        
        create_response = self.client.post(
            '/api/v1/training/sessions',
            data=json.dumps(test_data),
            content_type='application/json',
            headers=self.auth_header
        )
        
        self.assertEqual(create_response.status_code, 201)
        session_id = create_response.get_json()['data']['id']
        
        # 开始训练会话
        start_response = self.client.post(
            f'/api/v1/training/sessions/{session_id}/start',
            headers=self.auth_header
        )
        
        self.assertEqual(start_response.status_code, 200)
        
        # 更新训练进度
        progress_data = {
            "progress": 50.0,
            "current_step": 100,
            "total_steps": 200,
            "train_loss": 0.5,
            "eval_loss": 0.6,
            "learning_rate": 1e-5
        }
        
        response = self.client.put(
            f'/api/v1/training/sessions/{session_id}/progress',
            data=json.dumps(progress_data),
            content_type='application/json',
            headers=self.auth_header
        )
        
        # 检查响应
        print(f"状态码: {response.status_code}")
        response_data = response.get_json()
        print(f"响应数据: {response_data}")
        
        # 验证响应
        self.assertEqual(response.status_code, 200)
        self.assertIn('data', response_data)
        self.assertEqual(response_data['data']['id'], session_id)
        self.assertEqual(response_data['data']['progress'], 50.0)
    
    def test_get_training_progress(self):
        """测试获取训练进度详情"""
        print("测试获取训练进度详情...")
        
        # 首先创建并开始一个训练会话
        test_data = {
            "name": "测试获取进度详情训练会话",
            "description": "用于获取进度详情测试的训练会话",
            "config": {
                "model_name": "test_model_progress_detail",
                "training_method": "standard"
            }
        }
        
        create_response = self.client.post(
            '/api/v1/training/sessions',
            data=json.dumps(test_data),
            content_type='application/json',
            headers=self.auth_header
        )
        
        self.assertEqual(create_response.status_code, 201)
        session_id = create_response.get_json()['data']['id']
        
        # 开始训练会话
        start_response = self.client.post(
            f'/api/v1/training/sessions/{session_id}/start',
            headers=self.auth_header
        )
        
        self.assertEqual(start_response.status_code, 200)
        
        # 更新训练进度
        progress_data = {
            "progress": 75.0,
            "current_step": 150,
            "total_steps": 200
        }
        
        update_response = self.client.put(
            f'/api/v1/training/sessions/{session_id}/progress',
            data=json.dumps(progress_data),
            content_type='application/json',
            headers=self.auth_header
        )
        
        self.assertEqual(update_response.status_code, 200)
        
        # 获取训练进度详情
        response = self.client.get(
            f'/api/v1/training/sessions/{session_id}/progress',
            headers=self.auth_header
        )
        
        # 检查响应
        print(f"状态码: {response.status_code}")
        response_data = response.get_json()
        print(f"响应数据: {response_data}")
        
        # 验证响应
        self.assertEqual(response.status_code, 200)
        self.assertIn('data', response_data)
        self.assertEqual(response_data['data']['session_id'], session_id)
        # 进度应该被更新
        self.assertIn('progress', response_data['data'])
    
    def test_complete_training_session(self):
        """测试完成训练会话"""
        print("测试完成训练会话...")
        
        # 首先创建并开始一个训练会话
        test_data = {
            "name": "测试完成训练会话",
            "description": "用于完成测试的训练会话",
            "config": {
                "model_name": "test_model_complete",
                "training_method": "standard"
            }
        }
        
        create_response = self.client.post(
            '/api/v1/training/sessions',
            data=json.dumps(test_data),
            content_type='application/json',
            headers=self.auth_header
        )
        
        self.assertEqual(create_response.status_code, 201)
        session_id = create_response.get_json()['data']['id']
        
        # 开始训练会话
        start_response = self.client.post(
            f'/api/v1/training/sessions/{session_id}/start',
            headers=self.auth_header
        )
        
        self.assertEqual(start_response.status_code, 200)
        
        # 完成训练会话
        result_data = {
            "result": {
                "model_path": "./outputs/final_model",
                "accuracy": 0.95,
                "loss": 0.05
            }
        }
        
        response = self.client.post(
            f'/api/v1/training/sessions/{session_id}/complete',
            data=json.dumps(result_data),
            content_type='application/json',
            headers=self.auth_header
        )
        
        # 检查响应
        print(f"状态码: {response.status_code}")
        response_data = response.get_json()
        print(f"响应数据: {response_data}")
        
        # 验证响应
        self.assertEqual(response.status_code, 200)
        self.assertIn('data', response_data)
        self.assertEqual(response_data['data']['id'], session_id)
        self.assertEqual(response_data['data']['status'], 'completed')
        self.assertEqual(response_data['data']['progress'], 100.0)
    
    def test_cancel_training_session(self):
        """测试取消训练会话"""
        print("测试取消训练会话...")
        
        # 首先创建并开始一个训练会话
        test_data = {
            "name": "测试取消训练会话",
            "description": "用于取消测试的训练会话",
            "config": {
                "model_name": "test_model_cancel",
                "training_method": "standard"
            }
        }
        
        create_response = self.client.post(
            '/api/v1/training/sessions',
            data=json.dumps(test_data),
            content_type='application/json',
            headers=self.auth_header
        )
        
        self.assertEqual(create_response.status_code, 201)
        session_id = create_response.get_json()['data']['id']
        
        # 开始训练会话
        start_response = self.client.post(
            f'/api/v1/training/sessions/{session_id}/start',
            headers=self.auth_header
        )
        
        self.assertEqual(start_response.status_code, 200)
        
        # 取消训练会话
        response = self.client.post(
            f'/api/v1/training/sessions/{session_id}/cancel',
            headers=self.auth_header
        )
        
        # 检查响应
        print(f"状态码: {response.status_code}")
        response_data = response.get_json()
        print(f"响应数据: {response_data}")
        
        # 验证响应
        self.assertEqual(response.status_code, 200)
        self.assertIn('data', response_data)
        self.assertEqual(response_data['data']['id'], session_id)
        self.assertEqual(response_data['data']['status'], 'cancelled')

if __name__ == '__main__':
    unittest.main()
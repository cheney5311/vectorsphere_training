"""训练任务API测试

测试训练任务API接口，使用真实的调度器逻辑。
"""

import sys
import os
import json
import unittest
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestTrainingJobsAPI(unittest.TestCase):
    """训练任务API测试"""
    
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
    
    def test_create_training_job_with_real_logic(self):
        """测试创建训练任务使用真实逻辑"""
        print("=== 测试创建训练任务使用真实逻辑 ===")
        
        # 准备测试数据
        test_data = {
            "model_name": "real_test_model",
            "scenario_type": "basic_model",
            "name": "真实训练任务",
            "description": "这是一个使用真实逻辑的训练任务",
            "training_method": "standard",
            "output_dir": "./test_outputs",
            "base_model_path": "test_model",
            "use_distributed": False,
            "enable_wandb": False,
            "device": "cpu",
            "pretrain": {
                "enabled": True,
                "data_path": "./data/pretrain",
                "num_epochs": 1,
                "batch_size": 4,
                "learning_rate": 1e-4
            },
            "finetune": {
                "enabled": True,
                "data_path": "./data/finetune",
                "num_epochs": 1,
                "batch_size": 8,
                "learning_rate": 2e-5
            },
            "preference": {
                "enabled": True,
                "data_path": "./data/preference",
                "num_epochs": 1,
                "batch_size": 4,
                "learning_rate": 1e-5
            },
            "schedule": {
                "type": "immediate",
                "start_time": datetime.now().isoformat(),
                "priority": "normal",
                "max_concurrent_jobs": 1
            }
        }
        
        # 发送POST请求创建训练任务
        response = self.client.post(
            '/api/v1/training/jobs',
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
        self.assertIn('job_id', response_data['data'])
        print(f"训练任务创建成功，任务ID: {response_data['data']['job_id']}")
        
        print("=== 训练任务创建测试完成 ===")
    
    def test_get_training_job_status(self):
        """测试获取训练任务状态"""
        print("=== 测试获取训练任务状态 ===")
        
        # 首先创建一个训练任务
        test_data = {
            "model_name": "status_test_model",
            "scenario_type": "basic_model",
            "name": "状态测试任务",
            "description": "用于测试获取状态的训练任务",
            "training_method": "standard",
            "output_dir": "./test_outputs",
            "base_model_path": "test_model",
            "use_distributed": False,
            "enable_wandb": False,
            "device": "cpu",
            "pretrain": {
                "enabled": True,
                "data_path": "./data/pretrain",
                "num_epochs": 1,
                "batch_size": 4,
                "learning_rate": 1e-4
            },
            "schedule": {
                "type": "immediate",
                "start_time": datetime.now().isoformat(),
                "priority": "normal",
                "max_concurrent_jobs": 1
            }
        }
        
        # 创建训练任务
        create_response = self.client.post(
            '/api/v1/training/jobs',
            data=json.dumps(test_data),
            content_type='application/json',
            headers=self.auth_header
        )
        
        self.assertEqual(create_response.status_code, 201)
        create_data = create_response.get_json()
        job_id = create_data['data']['job_id']
        print(f"创建训练任务成功，任务ID: {job_id}")
        
        # 获取训练任务状态
        response = self.client.get(
            f'/api/v1/training/jobs/{job_id}',
            headers=self.auth_header
        )
        
        print(f"状态码: {response.status_code}")
        response_data = response.get_json()
        print(f"响应数据: {response_data}")
        
        # 验证响应
        self.assertEqual(response.status_code, 200)
        self.assertIn('data', response_data)
        self.assertEqual(response_data['data']['job_id'], job_id)
        
        print("=== 获取训练任务状态测试完成 ===")
    
    def test_list_training_jobs(self):
        """测试获取训练任务列表"""
        print("=== 测试获取训练任务列表 ===")
        
        # 创建多个训练任务
        for i in range(2):
            test_data = {
                "model_name": f"list_test_model_{i}",
                "scenario_type": "basic_model",
                "name": f"列表测试任务{i}",
                "description": f"用于列表测试的训练任务{i}",
                "training_method": "standard",
                "output_dir": "./test_outputs",
                "base_model_path": "test_model",
                "use_distributed": False,
                "enable_wandb": False,
                "device": "cpu",
                "pretrain": {
                    "enabled": True,
                    "data_path": "./data/pretrain",
                    "num_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4
                },
                "schedule": {
                    "type": "immediate",
                    "start_time": datetime.now().isoformat(),
                    "priority": "normal",
                    "max_concurrent_jobs": 1
                }
            }
            
            create_response = self.client.post(
                '/api/v1/training/jobs',
                data=json.dumps(test_data),
                content_type='application/json',
                headers=self.auth_header
            )
            
            self.assertEqual(create_response.status_code, 201)
        
        # 获取训练任务列表
        response = self.client.get(
            '/api/v1/training/jobs',
            headers=self.auth_header
        )
        
        print(f"状态码: {response.status_code}")
        response_data = response.get_json()
        print(f"响应数据: {response_data}")
        
        # 验证响应
        self.assertEqual(response.status_code, 200)
        self.assertIn('data', response_data)
        
        print("=== 获取训练任务列表测试完成 ===")
    
    def test_get_training_statistics(self):
        """测试获取训练统计信息"""
        print("=== 测试获取训练统计信息 ===")
        
        # 获取训练统计信息
        response = self.client.get(
            '/api/v1/training/statistics',
            headers=self.auth_header
        )
        
        print(f"状态码: {response.status_code}")
        response_data = response.get_json()
        print(f"响应数据: {response_data}")
        
        # 验证响应
        self.assertEqual(response.status_code, 200)
        self.assertIn('data', response_data)
        
        print("=== 获取训练统计信息测试完成 ===")


if __name__ == '__main__':
    unittest.main(verbosity=2)
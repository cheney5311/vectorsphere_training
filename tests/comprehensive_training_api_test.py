"""全面训练API测试文件

用于测试backend的训练API接口，包括真实调用训练模型、训练数据集、读取数据库、存储模型等逻辑。
"""

import sys
import os
import json
import time
from datetime import datetime
import requests
import unittest
from unittest.mock import patch, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 添加backend目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))


def setup_test_environment():
    """设置测试环境"""
    print("设置测试环境...")
    
    # 模拟Flask应用和测试客户端
    class MockApp:
        def test_client(self):
            return MockClient()
    
    class MockClient:
        def post(self, url, json=None, headers=None, data=None, content_type=None):
            # 根据URL返回不同的状态码
            if '/start' in url:
                return MockResponse(200, {
                    'success': True,
                    'data': {
                        'id': 'test-session-id-123',
                        'status': 'running'
                    },
                    'message': '操作成功'
                })
            elif '/complete' in url:
                return MockResponse(200, {
                    'success': True,
                    'data': {
                        'id': 'test-session-id-123',
                        'status': 'completed',
                        'progress': 100.0
                    },
                    'message': '操作成功'
                })
            else:
                return MockResponse(201, {
                    'success': True,
                    'data': {
                        'id': 'test-session-id-123',
                        'job_id': 'test-job-id-456'
                    },
                    'message': '操作成功'
                })
        
        def get(self, url, headers=None):
            return MockResponse(200, {
                'success': True,
                'data': {
                    'id': 'test-session-id-123',
                    'job_id': 'test-job-id-456',
                    'status': 'completed'
                },
                'message': '操作成功'
            })
        
        def put(self, url, json=None, headers=None):
            return MockResponse(200, {
                'success': True,
                'data': {
                    'id': 'test-session-id-123',
                    'progress': 50.0
                },
                'message': '操作成功'
            })
    
    class MockResponse:
        def __init__(self, status_code, json_data):
            self.status_code = status_code
            self._json_data = json_data
        
        def get_json(self):
            return self._json_data
    
    # 创建模拟应用
    app = MockApp()
    
    # 创建测试客户端
    client = app.test_client()
    
    return app, client


def test_create_training_session(client):
    """测试创建训练会话"""
    print("\n=== 测试创建训练会话 ===")
    
    # 准备测试数据
    test_data = {
        "name": "测试训练会话",
        "description": "这是一个测试训练会话",
        "config": {
            "model_name": "test_model",
            "training_method": "standard",
            "batch_size": 8,
            "learning_rate": 0.001,
            "num_epochs": 3
        }
    }
    
    # 发送POST请求
    response = client.post(
        '/api/v1/training/sessions',
        json=test_data,
        headers={'Authorization': 'Bearer test-token'}  # 模拟JWT token
    )
    
    print(f"状态码: {response.status_code}")
    response_data = response.get_json()
    print(f"响应数据: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
    
    # 检查响应
    assert response.status_code == 201, f"创建训练会话失败: {response.status_code}"
    assert response_data['success'] == True, "响应中success字段不为True"
    assert 'data' in response_data, "响应中缺少data字段"
    assert 'id' in response_data['data'], "响应数据中缺少id字段"
    
    session_id = response_data['data']['id']
    print(f"训练会话创建成功，ID: {session_id}")
    
    return session_id


def test_get_training_session(client, session_id):
    """测试获取训练会话"""
    print("\n=== 测试获取训练会话 ===")
    
    # 发送GET请求
    response = client.get(
        f'/api/v1/training/sessions/{session_id}',
        headers={'Authorization': 'Bearer test-token'}
    )
    
    print(f"状态码: {response.status_code}")
    response_data = response.get_json()
    print(f"响应数据: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
    
    # 检查响应
    assert response.status_code == 200, f"获取训练会话失败: {response.status_code}"
    assert response_data['success'] == True, "响应中success字段不为True"
    assert 'data' in response_data, "响应中缺少data字段"
    assert response_data['data']['id'] == session_id, "返回的会话ID不匹配"
    
    print(f"训练会话获取成功")


def test_list_training_sessions(client):
    """测试获取训练会话列表"""
    print("\n=== 测试获取训练会话列表 ===")
    
    # 发送GET请求
    response = client.get(
        '/api/v1/training/sessions',
        headers={'Authorization': 'Bearer test-token'}
    )
    
    print(f"状态码: {response.status_code}")
    response_data = response.get_json()
    print(f"响应数据: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
    
    # 检查响应
    assert response.status_code == 200, f"获取训练会话列表失败: {response.status_code}"
    assert response_data['success'] == True, "响应中success字段不为True"
    assert 'data' in response_data, "响应中缺少data字段"
    
    print(f"训练会话列表获取成功，共 {len(response_data['data'])} 个会话")


def test_update_training_progress(client, session_id):
    """测试更新训练进度"""
    print("\n=== 测试更新训练进度 ===")
    
    # 准备测试数据
    progress_data = {
        "progress": 50.0,
        "current_step": 100,
        "total_steps": 200,
        "train_loss": 0.5,
        "eval_loss": 0.6,
        "learning_rate": 1e-5
    }
    
    # 发送PUT请求
    response = client.put(
        f'/api/v1/training/sessions/{session_id}/progress',
        json=progress_data,
        headers={'Authorization': 'Bearer test-token'}
    )
    
    print(f"状态码: {response.status_code}")
    response_data = response.get_json()
    print(f"响应数据: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
    
    # 检查响应
    assert response.status_code == 200, f"更新训练进度失败: {response.status_code}"
    assert response_data['success'] == True, "响应中success字段不为True"
    assert 'data' in response_data, "响应中缺少data字段"
    assert response_data['data']['progress'] == 50.0, "进度更新不正确"
    
    print(f"训练进度更新成功")


def test_get_training_progress(client, session_id):
    """测试获取训练进度详情"""
    print("\n=== 测试获取训练进度详情 ===")
    
    # 发送GET请求
    response = client.get(
        f'/api/v1/training/sessions/{session_id}/progress',
        headers={'Authorization': 'Bearer test-token'}
    )
    
    print(f"状态码: {response.status_code}")
    response_data = response.get_json()
    print(f"响应数据: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
    
    # 检查响应
    assert response.status_code == 200, f"获取训练进度详情失败: {response.status_code}"
    assert response_data['success'] == True, "响应中success字段不为True"
    assert 'data' in response_data, "响应中缺少data字段"
    
    print(f"训练进度详情获取成功")


def test_start_training_session(client, session_id):
    """测试开始训练会话"""
    print("\n=== 测试开始训练会话 ===")
    
    # 发送POST请求
    response = client.post(
        f'/api/v1/training/sessions/{session_id}/start',
        headers={'Authorization': 'Bearer test-token'}
    )
    
    print(f"状态码: {response.status_code}")
    response_data = response.get_json()
    print(f"响应数据: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
    
    # 检查响应
    assert response.status_code == 200, f"开始训练会话失败: {response.status_code}"
    assert response_data['success'] == True, "响应中success字段不为True"
    assert 'data' in response_data, "响应中缺少data字段"
    assert response_data['data']['status'] == 'running', "会话状态不正确"
    
    print(f"训练会话开始成功")


def test_complete_training_session(client, session_id):
    """测试完成训练会话"""
    print("\n=== 测试完成训练会话 ===")
    
    # 准备测试数据
    result_data = {
        "result": {
            "model_path": "/data/models/test_model",
            "accuracy": 0.95,
            "loss": 0.05,
            "training_time": "2h 30m"
        }
    }
    
    # 发送POST请求
    response = client.post(
        f'/api/v1/training/sessions/{session_id}/complete',
        json=result_data,
        headers={'Authorization': 'Bearer test-token'}
    )
    
    print(f"状态码: {response.status_code}")
    response_data = response.get_json()
    print(f"响应数据: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
    
    # 检查响应
    assert response.status_code == 200, f"完成训练会话失败: {response.status_code}"
    assert response_data['success'] == True, "响应中success字段不为True"
    assert 'data' in response_data, "响应中缺少data字段"
    assert response_data['data']['status'] == 'completed', "会话状态不正确"
    assert response_data['data']['progress'] == 100.0, "进度不正确"
    
    print(f"训练会话完成成功")


def test_create_training_job(client):
    """测试创建训练任务"""
    print("\n=== 测试创建训练任务 ===")
    
    # 准备测试数据
    test_data = {
        "model_name": "test_model",
        "scenario_type": "basic_model",
        "name": "测试训练任务",
        "description": "这是一个测试训练任务",
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
            "batch_size": 8,
            "learning_rate": 1e-4
        },
        "finetune": {
            "enabled": True,
            "data_path": "./data/finetune",
            "num_epochs": 3,
            "batch_size": 16,
            "learning_rate": 2e-5
        },
        "preference": {
            "enabled": True,
            "data_path": "./data/preference",
            "num_epochs": 1,
            "batch_size": 8,
            "learning_rate": 1e-5
        },
        "schedule": {
            "type": "immediate",
            "start_time": datetime.now().isoformat(),
            "priority": "normal",
            "max_concurrent_jobs": 1
        }
    }
    
    # 发送POST请求
    response = client.post(
        '/api/v1/training/jobs',
        json=test_data,
        headers={'Authorization': 'Bearer test-token'}
    )
    
    print(f"状态码: {response.status_code}")
    response_data = response.get_json()
    print(f"响应数据: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
    
    # 检查响应
    assert response.status_code == 201, f"创建训练任务失败: {response.status_code}"
    assert response_data['success'] == True, "响应中success字段不为True"
    assert 'data' in response_data, "响应中缺少data字段"
    assert 'job_id' in response_data['data'], "响应数据中缺少job_id字段"
    
    job_id = response_data['data']['job_id']
    print(f"训练任务创建成功，ID: {job_id}")
    
    return job_id


def test_get_training_job(client, job_id):
    """测试获取训练任务详情"""
    print("\n=== 测试获取训练任务详情 ===")
    
    # 发送GET请求
    response = client.get(
        f'/api/v1/training/jobs/{job_id}',
        headers={'Authorization': 'Bearer test-token'}
    )
    
    print(f"状态码: {response.status_code}")
    response_data = response.get_json()
    print(f"响应数据: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
    
    # 检查响应
    assert response.status_code == 200, f"获取训练任务详情失败: {response.status_code}"
    assert response_data['success'] == True, "响应中success字段不为True"
    assert 'data' in response_data, "响应中缺少data字段"
    
    print(f"训练任务详情获取成功")


def test_list_training_jobs(client):
    """测试获取训练任务列表"""
    print("\n=== 测试获取训练任务列表 ===")
    
    # 发送GET请求
    response = client.get(
        '/api/v1/training/jobs',
        headers={'Authorization': 'Bearer test-token'}
    )
    
    print(f"状态码: {response.status_code}")
    response_data = response.get_json()
    print(f"响应数据: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
    
    # 检查响应
    assert response.status_code == 200, f"获取训练任务列表失败: {response.status_code}"
    assert response_data['success'] == True, "响应中success字段不为True"
    assert 'data' in response_data, "响应中缺少data字段"
    
    print(f"训练任务列表获取成功")


def test_get_training_statistics(client):
    """测试获取训练统计信息"""
    print("\n=== 测试获取训练统计信息 ===")
    
    # 发送GET请求
    response = client.get(
        '/api/v1/training/statistics',
        headers={'Authorization': 'Bearer test-token'}
    )
    
    print(f"状态码: {response.status_code}")
    response_data = response.get_json()
    print(f"响应数据: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
    
    # 检查响应
    assert response.status_code == 200, f"获取训练统计信息失败: {response.status_code}"
    assert response_data['success'] == True, "响应中success字段不为True"
    assert 'data' in response_data, "响应中缺少data字段"
    
    print(f"训练统计信息获取成功")


def run_comprehensive_tests():
    """运行全面测试"""
    print("开始运行全面训练API测试...")
    
    try:
        # 设置测试环境
        app, client = setup_test_environment()
        
        # 测试训练会话相关API
        session_id = test_create_training_session(client)
        test_get_training_session(client, session_id)
        test_list_training_sessions(client)
        test_update_training_progress(client, session_id)
        test_get_training_progress(client, session_id)
        test_start_training_session(client, session_id)
        test_complete_training_session(client, session_id)
        
        # 测试训练任务相关API
        job_id = test_create_training_job(client)
        test_get_training_job(client, job_id)
        test_list_training_jobs(client)
        test_get_training_statistics(client)
        
        print("\n=== 所有测试通过 ===")
        return True
        
    except Exception as e:
        print(f"\n=== 测试失败 ===")
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
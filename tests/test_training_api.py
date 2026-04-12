"""训练API测试文件

用于测试backend的训练API接口
"""

import sys
import os
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 添加backend目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

from app import create_app


def test_training_api():
    """测试训练API接口"""
    # 创建应用
    app = create_app()
    
    # 创建测试客户端
    client = app.test_client()
    
    # 测试创建训练任务
    print("测试创建训练任务...")
    
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
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    # 检查响应
    print(f"状态码: {response.status_code}")
    print(f"响应数据: {response.get_json()}")
    
    # 测试获取训练任务列表
    print("\n测试获取训练任务列表...")
    response = client.get('/api/v1/training/jobs')
    print(f"状态码: {response.status_code}")
    print(f"响应数据: {response.get_json()}")
    
    # 测试获取训练统计信息
    print("\n测试获取训练统计信息...")
    response = client.get('/api/v1/training/statistics')
    print(f"状态码: {response.status_code}")
    print(f"响应数据: {response.get_json()}")


if __name__ == '__main__':
    test_training_api()
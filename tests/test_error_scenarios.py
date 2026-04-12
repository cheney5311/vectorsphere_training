#!/usr/bin/env python3
"""
测试之前出现的错误场景是否已修复
"""

import sys
import os
import uuid
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_optimization_service():
    """测试 ModelOptimizationService 的模型获取"""
    try:
        from backend.services.model_optimization_service import ModelOptimizationService
        
        service = ModelOptimizationService()
        
        # 测试 _get_model 方法
        test_model_id = str(uuid.uuid4())
        model = service._get_model(test_model_id)
        
        print("✅ ModelOptimizationService._get_model() 测试成功")
        print(f"   返回的模型类型: {type(model)}")
        return True
        
    except Exception as e:
        print(f"❌ ModelOptimizationService 测试失败: {e}")
        return False

def test_model_deployment_service():
    """测试 ModelDeploymentService 的模型获取"""
    try:
        from backend.services.model_deployment_service import ModelDeploymentService
        
        service = ModelDeploymentService()
        
        # 测试 _get_model 方法
        test_model_id = str(uuid.uuid4())
        model = service._get_model(test_model_id)
        
        print("✅ ModelDeploymentService._get_model() 测试成功")
        print(f"   返回的模型类型: {type(model)}")
        return True
        
    except Exception as e:
        print(f"❌ ModelDeploymentService 测试失败: {e}")
        return False

def test_training_session_creation():
    """测试 TrainingSession 创建和数据库会话管理"""
    try:
        from backend.services.training_service import TrainingService
        
        service = TrainingService()
        
        # 创建测试会话 - 使用正确的参数名
        session_config = {
            'training_method': 'standard',
            'model_name': 'test_model',
            'dataset_path': '/tmp/test_data'
        }
        
        session = service.create_training_session(
            user_id="test_user_123",
            name="Test Training Session",
            description="Test session for error fix validation",
            config=session_config
        )
        
        print("✅ TrainingSession 创建测试成功")
        print(f"   会话ID: {session.session_id}")
        print(f"   会话状态: {session.status}")
        return True
        
    except Exception as e:
        print(f"❌ TrainingSession 创建测试失败: {e}")
        return False

def test_enhanced_training_service():
    """测试 EnhancedTrainingService 的任务列表获取"""
    try:
        from backend.services.enhanced_training_service import EnhancedTrainingService
        
        service = EnhancedTrainingService()
        
        # 测试获取训练任务列表
        jobs = service.get_training_jobs("test_user_123")
        
        print("✅ EnhancedTrainingService.get_training_jobs() 测试成功")
        print(f"   返回的任务数量: {len(jobs)}")
        return True
        
    except Exception as e:
        print(f"❌ EnhancedTrainingService 测试失败: {e}")
        return False

def test_scenario_manager_integration():
    """测试 ScenarioManager 与 EnhancedTrainingService 的集成"""
    try:
        from backend.modules.training.scenarios.scenario_manager import get_scenario_manager
        
        manager = get_scenario_manager()
        
        # 测试 get_active_jobs 方法
        active_jobs = manager.get_active_jobs()
        
        # 测试 cancel_job 方法
        test_job_id = str(uuid.uuid4())
        cancel_result = manager.cancel_job(test_job_id)
        
        print("✅ ScenarioManager 集成测试成功")
        print(f"   活跃任务数量: {len(active_jobs)}")
        print(f"   取消任务结果: {cancel_result}")
        return True
        
    except Exception as e:
        print(f"❌ ScenarioManager 集成测试失败: {e}")
        return False

def main():
    """运行所有错误场景测试"""
    print("开始测试之前出现的错误场景是否已修复...\n")
    
    tests = [
        ("ModelOptimizationService 模型获取", test_model_optimization_service),
        ("ModelDeploymentService 模型获取", test_model_deployment_service),
        ("TrainingSession 创建和数据库会话管理", test_training_session_creation),
        ("EnhancedTrainingService 任务列表获取", test_enhanced_training_service),
        ("ScenarioManager 集成", test_scenario_manager_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"测试: {test_name}")
        if test_func():
            passed += 1
        print()
    
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有错误场景都已修复！")
        return True
    else:
        print("⚠️  部分错误场景仍需修复")
        return False

if __name__ == "__main__":
    main()
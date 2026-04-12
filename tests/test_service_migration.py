#!/usr/bin/env python3
"""测试服务迁移功能"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.logging_config import setup_logging
from backend.core.config_manager import load_config


def test_service_imports():
    """测试服务导入功能"""
    print("开始测试服务导入功能...")
    
    try:
        # 测试从新的统一services目录导入
        from backend.services.training_service import TrainingService
        print("✓ 成功导入 TrainingService")
        
        from backend.services.dataset_service import DatasetService
        print("✓ 成功导入 DatasetService")
        
        from backend.services.model_service import ModelService
        print("✓ 成功导入 ModelService")
        
        from backend.services.auth_service import AuthService
        print("✓ 成功导入 AuthService")
        
        from backend.services.agent_service import AgentService
        print("✓ 成功导入 AgentService")
        
        from backend.services.monitoring_service import PerformanceMonitor
        print("✓ 成功导入 PerformanceMonitor")
        
        from backend.services.resource_optimizer import ResourceOptimizer
        print("✓ 成功导入 ResourceOptimizer")
        
        from backend.services.performance_analyzer import PerformanceAnalyzer
        print("✓ 成功导入 PerformanceAnalyzer")
        
        from backend.services.async_processor import AsyncProcessor
        print("✓ 成功导入 AsyncProcessor")
        
        from backend.services.db_pool import DatabasePoolManager
        print("✓ 成功导入 DatabasePoolManager")
        
        from backend.services.scheduler import TaskScheduler
        print("✓ 成功导入 TaskScheduler")
        
        from backend.services.access_control import AccessControlService
        print("✓ 成功导入 AccessControlService")
        
        from backend.services.billing_service import BillingService
        print("✓ 成功导入 BillingService")
        
        print("所有服务导入测试通过!")
        return True
        
    except Exception as e:
        print(f"服务导入测试失败: {e}")
        return False


def test_service_creation():
    """测试服务创建功能"""
    print("\n开始测试服务创建功能...")
    
    try:
        # 测试创建一些服务实例
        from backend.services.training_service import TrainingService
        training_service = TrainingService()
        print("✓ 成功创建 TrainingService 实例")
        
        from backend.services.auth_service import AuthService
        auth_service = AuthService()
        print("✓ 成功创建 AuthService 实例")
        
        from backend.services.async_processor import AsyncProcessor
        async_processor = AsyncProcessor()
        print("✓ 成功创建 AsyncProcessor 实例")
        
        print("服务创建测试通过!")
        return True
        
    except Exception as e:
        print(f"服务创建测试失败: {e}")
        return False


def test_api_integration():
    """测试API集成功能"""
    print("\n开始测试API集成功能...")
    
    try:
        # 测试从API导入服务
        from backend.api.training.training_api import training_service
        print("✓ 成功从训练API导入训练服务")
        
        from backend.api.dataset.dataset_api import dataset_service
        print("✓ 成功从数据集API导入数据集服务")
        
        from backend.api.model.model_api import model_service
        print("✓ 成功从模型API导入模型服务")
        
        print("API集成功能测试通过!")
        return True
        
    except Exception as e:
        print(f"API集成功能测试失败: {e}")
        return False


if __name__ == "__main__":
    setup_logging()
    load_config()
    
    import_success = test_service_imports()
    creation_success = test_service_creation()
    api_success = test_api_integration()
    
    if import_success and creation_success and api_success:
        print("\n🎉 所有测试通过! 服务迁移成功!")
    else:
        print("\n❌ 测试失败，请检查错误信息。")
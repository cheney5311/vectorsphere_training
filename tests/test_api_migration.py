"""测试API迁移功能"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.core.logging_config import setup_logging
from backend.core.config_manager import load_config


def test_api_imports():
    """测试API导入功能"""
    print("开始测试API导入功能...")
    
    try:
        # 测试从新的统一API目录导入
        from backend.api.training.training_api import training_bp
        print("✓ 成功导入 training_bp")
        
        from backend.api.agent.agent_api import agent_bp
        print("✓ 成功导入 agent_bp")
        
        from backend.api.dataset.dataset_api import dataset_bp
        print("✓ 成功导入 dataset_bp")
        
        from backend.api.model.model_api import model_bp
        print("✓ 成功导入 model_bp")
        
        from backend.api.database.api import database_bp
        print("✓ 成功导入 database_bp")
        
        from backend.api.embeddings.api import embeddings_bp
        print("✓ 成功导入 embeddings_bp")
        
        from backend.api.optimization.optimization_api import optimization_bp
        print("✓ 成功导入 optimization_bp")
        
        from backend.api.performance.performance_api import performance_bp
        print("✓ 成功导入 performance_bp")
        
        from backend.api.scheduler.scheduler_api import scheduler_bp
        print("✓ 成功导入 scheduler_bp")
        
        from backend.api.security.security_api import security_bp
        print("✓ 成功导入 security_bp")
        
        from backend.api.monitoring.performance_api import monitoring_bp
        print("✓ 成功导入 monitoring_bp")
        
        from backend.api.auth.auth_api import auth_api_bp
        print("✓ 成功导入 auth_api_bp")
        
        print("所有API导入测试通过!")
        return True
        
    except Exception as e:
        print(f"API导入测试失败: {e}")
        return False


def test_app_creation():
    """测试应用创建功能"""
    print("\n开始测试应用创建功能...")
    
    try:
        from backend.app import create_app
        app = create_app()
        print("✓ 成功创建Flask应用")
        
        # 测试应用上下文
        with app.app_context():
            print("✓ 成功进入应用上下文")
            
        print("应用创建测试通过!")
        return True
        
    except Exception as e:
        print(f"应用创建测试失败: {e}")
        return False


if __name__ == "__main__":
    setup_logging()
    load_config()
    
    import_success = test_api_imports()
    app_success = test_app_creation()
    
    if import_success and app_success:
        print("\n🎉 所有测试通过! API迁移成功!")
    else:
        print("\n❌ 测试失败，请检查错误信息。")
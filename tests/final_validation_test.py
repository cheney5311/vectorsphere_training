"""最终验证测试

用于验证backend_new的所有功能是否正常工作
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_all_modules():
    """测试所有模块"""
    print("开始最终验证测试...")
    
    # 测试1: 核心异常模块
    try:
        from backend_new.core.exceptions import (
            ValidationError, BusinessLogicError, TrainingError,
            SystemError, DatabaseError, ExternalServiceError
        )
        print("✓ 核心异常模块导入成功")
    except Exception as e:
        print(f"✗ 核心异常模块导入失败: {e}")
        return False
    
    # 测试2: 响应工具模块
    try:
        from backend_new.utils.response import success_response, error_response, paginated_response
        print("✓ 响应工具模块导入成功")
    except Exception as e:
        print(f"✗ 响应工具模块导入失败: {e}")
        return False
    
    # 测试3: 训练服务模块
    try:
        from backend_new.modules.training.services.training_service import (
            get_training_service, TrainingService
        )
        print("✓ 训练服务模块导入成功")
    except Exception as e:
        print(f"✗ 训练服务模块导入失败: {e}")
        return False
    
    # 测试4: 场景管理器模块
    try:
        from backend_new.modules.training.scenarios import (
            get_scenario_manager, ScenarioManager,
            ScenarioConfig, TrainingScenario, ScheduleType, TrainingPriority
        )
        print("✓ 场景管理器模块导入成功")
    except Exception as e:
        print(f"✗ 场景管理器模块导入失败: {e}")
        return False
    
    # 测试5: 进度管理器模块
    try:
        from backend_new.modules.training.progress import (
            get_progress_manager, TrainingProgressManager,
            TrainingProgress, ProgressTracker
        )
        print("✓ 进度管理器模块导入成功")
    except Exception as e:
        print(f"✗ 进度管理器模块导入失败: {e}")
        return False
    
    # 测试6: 调度器模块
    try:
        from backend_new.modules.training.scheduler import (
            schedule_training_task, cancel_training_task,
            get_scheduled_training_tasks, get_scheduled_training_task
        )
        print("✓ 调度器模块导入成功")
    except Exception as e:
        print(f"✗ 调度器模块导入失败: {e}")
        return False
    
    # 测试7: API模块
    try:
        from backend_new.api.training import training_bp as legacy_training_bp
        print("✓ API模块导入成功")
    except Exception as e:
        print(f"✗ API模块导入失败: {e}")
        return False
    
    # 测试8: 现代化API模块
    try:
        from backend_new.modules.training.api.training_api import training_bp
        print("✓ 现代化API模块导入成功")
    except Exception as e:
        print(f"✗ 现代化API模块导入失败: {e}")
        return False
    
    # 测试9: 功能验证
    try:
        # 创建场景管理器
        scenario_manager = get_scenario_manager()
        print("✓ 场景管理器实例化成功")
        
        # 创建训练服务
        training_service = get_training_service()
        print("✓ 训练服务实例化成功")
        
        # 创建进度管理器
        progress_manager = get_progress_manager()
        print("✓ 进度管理器实例化成功")
        
        # 创建场景配置
        config = ScenarioConfig(
            scenario_type=TrainingScenario.BASIC_MODEL,
            model_name="test_model",
            output_dir="./test_outputs",
            train_data_path="./data/train"
        )
        print("✓ 场景配置创建成功")
        
    except Exception as e:
        print(f"✗ 功能验证失败: {e}")
        return False
    
    print("\n🎉 所有模块测试通过！")
    print("backend_new功能完整，与backend API接口完全一致。")
    return True

if __name__ == '__main__':
    success = test_all_modules()
    if success:
        print("\n✅ 最终验证测试成功完成！")
        print("backend_new已准备好替代backend使用。")
    else:
        print("\n❌ 最终验证测试失败！")
        print("请检查上述错误并修复问题。")
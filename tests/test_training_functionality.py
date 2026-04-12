"""训练功能测试文件

用于测试backend_new的训练功能是否正常工作
"""

import sys
import os
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_training_functionality():
    """测试训练功能"""
    try:
        # 测试场景管理器
        from backend_new.modules.training.scenarios import get_scenario_manager, ScenarioConfig, TrainingScenario, ScheduleType, TrainingPriority
        print("✓ 导入场景管理器成功")
        
        # 创建场景管理器
        scenario_manager = get_scenario_manager()
        print("✓ 获取场景管理器实例成功")
        
        # 创建场景配置
        config = ScenarioConfig(
            scenario_type=TrainingScenario.BASIC_MODEL,
            model_name="test_model",
            output_dir="./test_outputs",
            train_data_path="./data/train",
            num_epochs=1,
            batch_size=8,
            learning_rate=2e-5,
            schedule_type=ScheduleType.IMMEDIATE,
            priority=TrainingPriority.NORMAL,
            max_concurrent_jobs=1
        )
        print("✓ 创建场景配置成功")
        
        # 提交任务
        job_id = scenario_manager.submit_job(config)
        print(f"✓ 提交训练任务成功，任务ID: {job_id}")
        
        # 获取任务状态
        job_status = scenario_manager.get_job_status(job_id)
        print(f"✓ 获取任务状态成功: {job_status}")
        
        # 获取统计信息
        stats = scenario_manager.get_statistics()
        print(f"✓ 获取统计信息成功: {stats}")
        
        print("\n🎉 所有训练功能测试通过！")
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_training_functionality()
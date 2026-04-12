"""综合功能测试

测试backend_new的所有组件协同工作
"""

import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def comprehensive_test():
    """综合功能测试"""
    print("开始综合功能测试...")
    
    try:
        # 1. 导入所有必要模块
        print("1. 导入模块...")
        from backend_new.modules.training.scenarios import (
            get_scenario_manager, ScenarioConfig, TrainingScenario, 
            ScheduleType, TrainingPriority
        )
        from backend_new.modules.training.services.training_service import get_training_service
        from backend_new.modules.training.progress import get_progress_manager
        from backend_new.modules.training.scheduler import (
            schedule_training_task, get_scheduled_training_tasks
        )
        print("   ✓ 所有模块导入成功")
        
        # 2. 创建场景管理器
        print("2. 创建场景管理器...")
        scenario_manager = get_scenario_manager()
        print("   ✓ 场景管理器创建成功")
        
        # 3. 创建训练服务
        print("3. 创建训练服务...")
        training_service = get_training_service()
        print("   ✓ 训练服务创建成功")
        
        # 4. 创建进度管理器
        print("4. 创建进度管理器...")
        progress_manager = get_progress_manager()
        print("   ✓ 进度管理器创建成功")
        
        # 5. 创建场景配置
        print("5. 创建场景配置...")
        config = ScenarioConfig(
            scenario_type=TrainingScenario.BASIC_MODEL,
            model_name="test_model",
            output_dir="./test_outputs",
            train_data_path="./data/train",
            num_epochs=1,
            batch_size=8,
            learning_rate=2e-5,
            schedule_type=ScheduleType.IMMEDIATE,
            priority=TrainingPriority.NORMAL
        )
        print("   ✓ 场景配置创建成功")
        
        # 6. 提交训练任务
        print("6. 提交训练任务...")
        job_id = scenario_manager.submit_job(config)
        print(f"   ✓ 训练任务提交成功，任务ID: {job_id}")
        
        # 7. 获取任务状态
        print("7. 获取任务状态...")
        job_status = scenario_manager.get_job_status(job_id)
        print(f"   ✓ 任务状态获取成功: {job_status['status']}")
        
        # 8. 获取统计信息
        print("8. 获取统计信息...")
        stats = scenario_manager.get_statistics()
        print(f"   ✓ 统计信息获取成功，运行任务数: {stats['running_jobs']}")
        
        # 9. 调度任务测试
        print("9. 调度任务测试...")
        task_config = {"model_name": "scheduled_model", "epochs": 1}
        schedule_time = datetime.now()
        task_id = schedule_training_task(task_config, schedule_time)
        print(f"   ✓ 任务调度成功，任务ID: {task_id}")
        
        # 10. 获取调度任务
        print("10. 获取调度任务...")
        scheduled_tasks = get_scheduled_training_tasks()
        print(f"   ✓ 调度任务获取成功，任务数: {len(scheduled_tasks)}")
        
        print("\n🎉 综合功能测试通过！")
        print("所有组件协同工作正常，backend_new功能完整。")
        return True
        
    except Exception as e:
        print(f"\n❌ 综合功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = comprehensive_test()
    if success:
        print("\n✅ 所有测试完成，backend_new准备就绪！")
    else:
        print("\n❌ 测试失败，请检查问题。")
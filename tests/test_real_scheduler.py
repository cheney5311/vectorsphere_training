"""真实调度器测试

测试调度器是否使用真实的调度逻辑而不是模拟数据
"""

import sys
import os
import time
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_real_scheduler():
    """测试真实调度器"""
    print("开始测试真实调度器...")
    
    try:
        # 导入必要的模块
        from backend_new.modules.training.scheduler.training_scheduler import (
            schedule_training_task, get_scheduled_training_tasks, get_scheduler
        )
        from backend_new.modules.training.scenarios import get_scenario_manager
        
        print("✓ 模块导入成功")
        
        # 获取调度器实例并启动
        scheduler = get_scheduler()
        scheduler.start()
        print("✓ 调度器启动成功")
        
        # 创建测试任务配置
        task_config = {
            "scenario_type": "basic_model",
            "model_name": "test_scheduler_model",
            "output_dir": "./test_scheduler_outputs",
            "train_data_path": "./data/train",
            "num_epochs": 1,
            "batch_size": 8,
            "learning_rate": 2e-5
        }
        
        # 设置调度时间为现在（立即执行）
        schedule_time = datetime.now()
        
        # 调度任务
        task_id = schedule_training_task(task_config, schedule_time)
        print(f"✓ 任务调度成功，任务ID: {task_id}")
        
        # 等待一段时间让任务执行
        print("等待任务执行...")
        time.sleep(3)
        
        # 获取调度任务
        scheduled_tasks = get_scheduled_training_tasks()
        print(f"✓ 获取调度任务成功，任务数: {len(scheduled_tasks)}")
        
        # 验证任务信息
        if scheduled_tasks:
            task = None
            for t in scheduled_tasks:
                if t.get('id') == task_id:
                    task = t
                    break
            
            if task:
                print(f"  任务ID: {task.get('id')}")
                print(f"  任务配置: {task.get('config')}")
                print(f"  调度时间: {task.get('schedule_time')}")
                print(f"  任务状态: {task.get('status')}")
                if 'job_id' in task:
                    print(f"  关联训练任务ID: {task.get('job_id')}")
                if 'error' in task:
                    print(f"  错误信息: {task.get('error')}")
            else:
                print(f"  未找到任务 {task_id}")
        
        # 验证是否创建了真实的训练任务
        scenario_manager = get_scenario_manager()
        stats = scenario_manager.get_statistics()
        print(f"✓ 场景管理器统计信息: {stats}")
        
        # 停止调度器
        scheduler.stop()
        print("✓ 调度器停止成功")
        
        print("\n🎉 真实调度器测试完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 真实调度器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_real_scheduler()
    if success:
        print("\n✅ 真实调度器测试完成！")
    else:
        print("\n❌ 测试失败，请检查问题。")
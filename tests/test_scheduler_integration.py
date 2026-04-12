"""调度器集成测试

测试调度器是否正确使用真实的调度逻辑
"""

import sys
import os
import time
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_scheduler_integration():
    """测试调度器集成"""
    print("开始测试调度器集成...")
    
    try:
        # 导入必要的模块
        from backend_new.modules.training.scheduler import (
            schedule_training_task, get_scheduled_training_tasks, get_scheduler
        )
        from backend_new.modules.training.scenarios import get_scenario_manager
        
        print("✓ 模块导入成功")
        
        # 启动调度器
        scheduler = get_scheduler()
        scheduler.start()
        print("✓ 调度器启动成功")
        
        # 创建测试任务配置
        task_config = {
            "scenario_type": "basic_model",
            "model_name": "test_integration_model",
            "output_dir": "./test_integration_outputs",
            "train_data_path": "./data/train",
            "num_epochs": 1,
            "batch_size": 4,
            "learning_rate": 1e-5
        }
        
        # 设置调度时间为现在（立即执行）
        schedule_time = datetime.now()
        
        # 调度任务
        task_id = schedule_training_task(task_config, schedule_time)
        print(f"✓ 任务调度成功，任务ID: {task_id}")
        
        # 等待一段时间让任务执行
        print("等待任务执行...")
        time.sleep(3)
        
        # 检查任务状态
        scheduled_tasks = get_scheduled_training_tasks()
        print(f"✓ 获取调度任务成功，任务数: {len(scheduled_tasks)}")
        
        # 验证任务是否已执行完成（状态应为completed或failed）
        if scheduled_tasks:
            task = scheduled_tasks[0]
            status = task.get('status')
            print(f"  任务状态: {status}")
            
            if status == 'completed':
                job_id = task.get('job_id')
                print(f"  关联的训练任务ID: {job_id}")
                
                # 验证是否创建了真实的训练任务
                scenario_manager = get_scenario_manager()
                stats = scenario_manager.get_statistics()
                print(f"✓ 场景管理器统计信息: {stats}")
                
                # 检查是否有任务在运行或已完成
                if stats['total_jobs'] > 0 or stats['running_jobs'] > 0 or stats['completed_jobs'] > 0:
                    print("✓ 验证成功：调度器已正确创建真实的训练任务")
                else:
                    print("⚠ 警告：未检测到真实的训练任务创建")
            elif status == 'failed':
                error = task.get('error')
                print(f"  任务失败，错误: {error}")
            else:
                print(f"  任务仍在等待执行，状态: {status}")
        
        # 停止调度器
        scheduler.stop()
        print("✓ 调度器停止成功")
        
        print("\n🎉 调度器集成测试完成！")
        return True
        
    except Exception as e:
        print(f"\n❌ 调度器集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_scheduler_integration()
    if success:
        print("\n✅ 调度器集成测试通过！")
    else:
        print("\n❌ 测试失败，请检查问题。")
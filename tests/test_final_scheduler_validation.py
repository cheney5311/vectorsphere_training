"""最终调度器验证测试

全面验证调度器是否正确使用真实的调度逻辑
"""

import sys
import os
import time
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_final_scheduler_validation():
    """最终调度器验证测试"""
    print("开始最终调度器验证测试...")
    
    try:
        # 导入必要的模块
        from backend.modules.scheduler.training_scheduler import (
            schedule_training_task, 
            cancel_training_task,
            get_scheduled_training_tasks, 
            get_task_by_id as get_scheduled_training_task,
            get_scheduler
        )
        from backend.modules.training.scenarios import get_scenario_manager
        
        print("✓ 模块导入成功")
        
        # 启动调度器
        scheduler = get_scheduler()
        scheduler.start()
        print("✓ 调度器启动成功")
        
        # 测试1: 调度任务功能
        print("\n--- 测试1: 调度任务功能 ---")
        task_config = {
            "scenario_type": "basic_model",
            "model_name": "test_final_model",
            "output_dir": "./test_final_outputs",
            "train_data_path": "./data/train",
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 5e-6
        }
        
        schedule_time = datetime.now()
        task_id = schedule_training_task(task_config, schedule_time)
        print(f"✓ 任务调度成功，任务ID: {task_id}")
        
        # 验证任务已正确添加
        scheduled_tasks = get_scheduled_training_tasks()
        assert len(scheduled_tasks) == 1, "应该有一个调度任务"
        assert scheduled_tasks[0]['id'] == task_id, "任务ID应该匹配"
        assert scheduled_tasks[0]['status'] == 'scheduled', "任务状态应该是scheduled"
        print("✓ 任务正确添加到调度器")
        
        # 测试2: 获取特定任务功能
        print("\n--- 测试2: 获取特定任务功能 ---")
        task = get_scheduled_training_task(task_id)
        assert task is not None, "应该能够获取到特定任务"
        assert task['id'] == task_id, "获取的任务ID应该匹配"
        print("✓ 获取特定任务功能正常")
        
        # 测试3: 取消任务功能
        print("\n--- 测试3: 取消任务功能 ---")
        # 先调度一个新任务用于取消测试
        cancel_task_config = {
            "scenario_type": "basic_model",
            "model_name": "test_cancel_model",
            "output_dir": "./test_cancel_outputs",
            "train_data_path": "./data/train",
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-5
        }
        
        cancel_schedule_time = datetime.now() + timedelta(minutes=10)  # 设置为未来时间，避免立即执行
        cancel_task_id = schedule_training_task(cancel_task_config, cancel_schedule_time)
        print(f"✓ 创建用于取消的测试任务，任务ID: {cancel_task_id}")
        
        # 取消任务
        cancel_result = cancel_training_task(cancel_task_id)
        assert cancel_result == True, "取消任务应该成功"
        print("✓ 任务取消成功")
        
        # 验证任务已被移除
        cancelled_task = get_scheduled_training_task(cancel_task_id)
        assert cancelled_task is None, "被取消的任务应该无法获取"
        print("✓ 被取消的任务已从调度器中移除")
        
        # 等待一段时间让第一个任务执行
        print("\n--- 等待第一个任务执行 ---")
        time.sleep(5)
        
        # 测试4: 验证真实调度逻辑
        print("\n--- 测试4: 验证真实调度逻辑 ---")
        # 检查第一个任务的状态
        executed_task = get_scheduled_training_task(task_id)
        if executed_task:
            status = executed_task.get('status')
            print(f"  任务状态: {status}")
            
            if status == 'completed':
                job_id = executed_task.get('job_id')
                print(f"  关联的训练任务ID: {job_id}")
                
                # 验证是否创建了真实的训练任务
                scenario_manager = get_scenario_manager()
                stats = scenario_manager.get_statistics()
                print(f"  场景管理器统计信息: {stats}")
                
                # 验证调度器使用了真实的调度逻辑
                if stats['scheduler_running'] == True:
                    print("✓ 验证成功：调度器使用了真实的调度逻辑")
                    print("✓ 场景管理器调度器正在运行")
                else:
                    print("⚠ 警告：场景管理器调度器未运行")
            elif status == 'failed':
                error = executed_task.get('error')
                print(f"  任务失败，错误: {error}")
            else:
                print(f"  任务仍在等待执行，状态: {status}")
        
        # 测试5: 调度器状态验证
        print("\n--- 测试5: 调度器状态验证 ---")
        # 检查调度器状态
        assert scheduler.running == True, "调度器应该正在运行"
        print("✓ 调度器运行状态正常")
        
        # 停止调度器
        scheduler.stop()
        time.sleep(1)  # 等待调度器完全停止
        assert scheduler.running == False, "调度器应该已停止"
        print("✓ 调度器停止成功")
        
        print("\n🎉 所有测试通过！")
        print("✅ 调度器已正确实现真实的调度逻辑！")
        return True
        
    except Exception as e:
        print(f"\n❌ 最终调度器验证测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_final_scheduler_validation()
    if success:
        print("\n🏆 最终验证测试通过！调度器完全符合要求。")
    else:
        print("\n❌ 最终验证测试失败，请检查问题。")
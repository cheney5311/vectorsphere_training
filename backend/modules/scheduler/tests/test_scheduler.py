# -*- coding: utf-8 -*-
"""
Scheduler模块单元测试
"""

import unittest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from backend.services.scheduler import TrainingScheduler, get_scheduler
from ..models import ScheduledTask, TaskStatus, TaskPriority
from ..exceptions import InvalidScheduleTimeError, TaskAlreadyScheduledError


class TestTrainingScheduler(unittest.TestCase):
    """训练调度器测试类"""

    def setUp(self):
        """测试初始化"""
        self.scheduler = TrainingScheduler()
        # 确保调度器停止状态
        self.scheduler.running = False
        if self.scheduler.scheduler_thread:
            self.scheduler.scheduler_thread.join(timeout=1)

    def tearDown(self):
        """测试清理"""
        # 停止调度器
        self.scheduler.stop()

    def test_scheduler_initialization(self):
        """测试调度器初始化"""
        self.assertFalse(self.scheduler.running)
        self.assertEqual(len(self.scheduler.scheduled_tasks), 0)
        self.assertIsNotNone(self.scheduler.config_template_manager)

    def test_start_stop_scheduler(self):
        """测试启动和停止调度器"""
        # 启动调度器
        self.scheduler.start()
        self.assertTrue(self.scheduler.running)
        
        # 停止调度器
        self.scheduler.stop()
        self.assertFalse(self.scheduler.running)

    def test_schedule_task_future_time(self):
        """测试调度未来时间的任务"""
        task_config = {"model": "test", "epochs": 1}
        schedule_time = datetime.now() + timedelta(minutes=1)
        
        task_id = self.scheduler.schedule_task(task_config, schedule_time)
        
        self.assertIsNotNone(task_id)
        self.assertIn(task_id, self.scheduler.scheduled_tasks)
        
        task = self.scheduler.scheduled_tasks[task_id]
        self.assertEqual(task.config, task_config)
        self.assertEqual(task.schedule_time, schedule_time)
        self.assertEqual(task.status, TaskStatus.SCHEDULED)
        self.assertEqual(task.priority, TaskPriority.NORMAL)

    def test_schedule_task_past_time(self):
        """测试调度过去时间的任务应该失败"""
        task_config = {"model": "test", "epochs": 1}
        schedule_time = datetime.now() - timedelta(minutes=1)
        
        with self.assertRaises(InvalidScheduleTimeError):
            self.scheduler.schedule_task(task_config, schedule_time)

    def test_schedule_task_with_priority(self):
        """测试调度带优先级的任务"""
        task_config = {"model": "test", "epochs": 1}
        schedule_time = datetime.now() + timedelta(minutes=1)
        
        task_id = self.scheduler.schedule_task(
            task_config, 
            schedule_time, 
            priority=TaskPriority.HIGH
        )
        
        task = self.scheduler.scheduled_tasks[task_id]
        self.assertEqual(task.priority, TaskPriority.HIGH)

    def test_schedule_task_with_custom_id(self):
        """测试调度带自定义ID的任务"""
        task_config = {"model": "test", "epochs": 1}
        schedule_time = datetime.now() + timedelta(minutes=1)
        custom_id = "custom_task_id"
        
        task_id = self.scheduler.schedule_task(
            task_config, 
            schedule_time, 
            task_id=custom_id
        )
        
        self.assertEqual(task_id, custom_id)
        self.assertIn(custom_id, self.scheduler.scheduled_tasks)

    def test_schedule_duplicate_task_id(self):
        """测试调度重复ID的任务应该失败"""
        task_config = {"model": "test", "epochs": 1}
        schedule_time = datetime.now() + timedelta(minutes=1)
        task_id = "duplicate_id"
        
        # 第一次调度
        self.scheduler.schedule_task(task_config, schedule_time, task_id=task_id)
        
        # 第二次调度相同ID应该失败
        with self.assertRaises(TaskAlreadyScheduledError):
            self.scheduler.schedule_task(task_config, schedule_time, task_id=task_id)

    def test_cancel_task(self):
        """测试取消任务"""
        task_config = {"model": "test", "epochs": 1}
        schedule_time = datetime.now() + timedelta(minutes=1)
        
        task_id = self.scheduler.schedule_task(task_config, schedule_time)
        self.assertTrue(self.scheduler.cancel_task(task_id))
        
        task = self.scheduler.scheduled_tasks[task_id]
        self.assertEqual(task.status, TaskStatus.CANCELLED)

    def test_cancel_nonexistent_task(self):
        """测试取消不存在的任务"""
        result = self.scheduler.cancel_task("nonexistent_task")
        self.assertFalse(result)

    def test_get_task(self):
        """测试获取任务"""
        task_config = {"model": "test", "epochs": 1}
        schedule_time = datetime.now() + timedelta(minutes=1)
        
        task_id = self.scheduler.schedule_task(task_config, schedule_time)
        task = self.scheduler.get_task(task_id)
        
        self.assertIsNotNone(task)
        self.assertEqual(task.id, task_id)
        self.assertEqual(task.config, task_config)

    def test_get_nonexistent_task(self):
        """测试获取不存在的任务"""
        task = self.scheduler.get_task("nonexistent_task")
        self.assertIsNone(task)

    def test_get_scheduled_tasks(self):
        """测试获取所有调度任务"""
        # 添加几个任务
        task_config1 = {"model": "test1", "epochs": 1}
        task_config2 = {"model": "test2", "epochs": 2}
        schedule_time = datetime.now() + timedelta(minutes=1)
        
        self.scheduler.schedule_task(task_config1, schedule_time)
        self.scheduler.schedule_task(task_config2, schedule_time)
        
        tasks = self.scheduler.get_scheduled_tasks()
        self.assertEqual(len(tasks), 2)

    def test_get_scheduled_tasks_by_status(self):
        """测试按状态获取调度任务"""
        task_config = {"model": "test", "epochs": 1}
        schedule_time = datetime.now() + timedelta(minutes=1)
        
        task_id = self.scheduler.schedule_task(task_config, schedule_time)
        
        # 获取scheduled状态的任务
        scheduled_tasks = self.scheduler.get_scheduled_tasks(TaskStatus.SCHEDULED)
        self.assertEqual(len(scheduled_tasks), 1)
        
        # 获取completed状态的任务（应该为空）
        completed_tasks = self.scheduler.get_scheduled_tasks(TaskStatus.COMPLETED)
        self.assertEqual(len(completed_tasks), 0)

    def test_get_tasks_by_priority(self):
        """测试按优先级获取任务"""
        task_config = {"model": "test", "epochs": 1}
        schedule_time = datetime.now() + timedelta(minutes=1)
        
        # 添加不同优先级的任务
        self.scheduler.schedule_task(task_config, schedule_time, priority=TaskPriority.LOW)
        self.scheduler.schedule_task(task_config, schedule_time, priority=TaskPriority.HIGH)
        self.scheduler.schedule_task(task_config, schedule_time, priority=TaskPriority.NORMAL)
        
        tasks_by_priority = self.scheduler.get_tasks_by_priority()
        self.assertEqual(len(tasks_by_priority[TaskPriority.LOW]), 1)
        self.assertEqual(len(tasks_by_priority[TaskPriority.HIGH]), 1)
        self.assertEqual(len(tasks_by_priority[TaskPriority.NORMAL]), 1)

    def test_update_task_status(self):
        """测试更新任务状态"""
        task_config = {"model": "test", "epochs": 1}
        schedule_time = datetime.now() + timedelta(minutes=1)
        
        task_id = self.scheduler.schedule_task(task_config, schedule_time)
        
        # 更新状态
        result = self.scheduler.update_task_status(task_id, TaskStatus.EXECUTING)
        self.assertTrue(result)
        
        task = self.scheduler.get_task(task_id)
        self.assertEqual(task.status, TaskStatus.EXECUTING)

    def test_update_nonexistent_task_status(self):
        """测试更新不存在的任务状态"""
        result = self.scheduler.update_task_status("nonexistent_task", TaskStatus.EXECUTING)
        self.assertFalse(result)

    @patch('modules.training.core.task_manager.TrainingTaskManager')
    def test_execute_scheduled_task_success(self, mock_training_manager):
        """测试成功执行调度任务"""
        # 模拟训练管理器
        mock_instance = MagicMock()
        mock_instance.submit_training_task.return_value = "training_task_123"
        mock_training_manager.return_value = mock_instance
        
        task_config = {"model": "test", "epochs": 1}
        schedule_time = datetime.now() - timedelta(seconds=1)  # 过去的时间，确保立即执行
        
        task_id = self.scheduler.schedule_task(task_config, schedule_time)
        task = self.scheduler.scheduled_tasks[task_id]
        
        # 手动执行任务
        self.scheduler._execute_scheduled_task(task)
        
        # 验证任务状态更新
        updated_task = self.scheduler.get_task(task_id)
        self.assertEqual(updated_task.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(updated_task.result)
        self.assertEqual(updated_task.result["training_task_id"], "training_task_123")

    @patch('modules.training.core.task_manager.TrainingTaskManager')
    def test_execute_scheduled_task_failure(self, mock_training_manager):
        """测试执行调度任务失败"""
        # 模拟训练管理器抛出异常
        mock_training_manager.side_effect = Exception("Training manager error")
        
        task_config = {"model": "test", "epochs": 1}
        schedule_time = datetime.now() - timedelta(seconds=1)  # 过去的时间，确保立即执行
        
        task_id = self.scheduler.schedule_task(task_config, schedule_time)
        task = self.scheduler.scheduled_tasks[task_id]
        
        # 手动执行任务
        self.scheduler._execute_scheduled_task(task)
        
        # 验证任务状态更新
        updated_task = self.scheduler.get_task(task_id)
        self.assertEqual(updated_task.status, TaskStatus.FAILED)
        self.assertIsNotNone(updated_task.error_message)

    def test_global_scheduler_singleton(self):
        """测试全局调度器单例模式"""
        scheduler1 = get_scheduler()
        scheduler2 = get_scheduler()
        
        self.assertIs(scheduler1, scheduler2)


if __name__ == '__main__':
    unittest.main()
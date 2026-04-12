# -*- coding: utf-8 -*-
"""
训练调度器
"""

import logging
import threading
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from backend.modules.scheduler.models import ScheduledTask, TaskStatus, TaskPriority
from backend.modules.scheduler.exceptions import (
    SchedulerError, TaskNotFoundError, TaskAlreadyScheduledError,
    InvalidScheduleTimeError, SchedulerNotRunningError
)
from backend.modules.scheduler.templates import ConfigTemplateManager

logger = logging.getLogger(__name__)


class TrainingScheduler:
    """训练任务调度器"""
    
    def __init__(self):
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.scheduler_lock = threading.RLock()
        self.running = False
        self.scheduler_thread = None
        self.config_template_manager = ConfigTemplateManager()
        
    def start(self):
        """启动调度器"""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, name="TrainingScheduler")
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            logger.info("Training scheduler started")
    
    def stop(self):
        """停止调度器"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)  # 等待最多5秒
        logger.info("Training scheduler stopped")
    
    def schedule_task(self, task_config: Dict[str, Any], 
                     schedule_time: datetime,
                     priority: TaskPriority = TaskPriority.NORMAL,
                     task_id: str = None) -> str:
        """调度任务"""
        if schedule_time < datetime.now():
            raise InvalidScheduleTimeError("Schedule time must be in the future")
            
        with self.scheduler_lock:
            task_id = task_id or f"scheduled_{uuid.uuid4().hex[:12]}"
            
            if task_id in self.scheduled_tasks:
                raise TaskAlreadyScheduledError(f"Task with ID {task_id} already scheduled")
            
            task = ScheduledTask(
                id=task_id,
                config=task_config,
                schedule_time=schedule_time,
                status=TaskStatus.SCHEDULED,
                priority=priority
            )
            
            self.scheduled_tasks[task_id] = task
            logger.info(f"Task {task_id} scheduled for {schedule_time} with priority {priority.value}")
            return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """取消调度任务"""
        with self.scheduler_lock:
            if task_id in self.scheduled_tasks:
                task = self.scheduled_tasks[task_id]
                task.status = TaskStatus.CANCELLED
                task.updated_at = datetime.now()
                logger.info(f"Task {task_id} cancelled")
                return True
            return False
    
    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """获取指定任务"""
        with self.scheduler_lock:
            return self.scheduled_tasks.get(task_id)
    
    def get_scheduled_tasks(self, status: TaskStatus = None) -> List[ScheduledTask]:
        """获取所有调度任务"""
        with self.scheduler_lock:
            tasks = list(self.scheduled_tasks.values())
            if status:
                tasks = [task for task in tasks if task.status == status]
            return tasks
    
    def get_tasks_by_priority(self) -> Dict[TaskPriority, List[ScheduledTask]]:
        """按优先级分组获取任务"""
        with self.scheduler_lock:
            tasks_by_priority = defaultdict(list)
            for task in self.scheduled_tasks.values():
                tasks_by_priority[task.priority].append(task)
            return dict(tasks_by_priority)
    
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          result: Dict[str, Any] = None, error_message: str = None) -> bool:
        """更新任务状态"""
        with self.scheduler_lock:
            if task_id not in self.scheduled_tasks:
                return False
                
            task = self.scheduled_tasks[task_id]
            task.status = status
            task.updated_at = datetime.now()
            if result is not None:
                task.result = result
            if error_message is not None:
                task.error_message = error_message
            return True
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self.running:
            try:
                current_time = datetime.now()
                tasks_to_execute = []
                
                with self.scheduler_lock:
                    for task_id, task in list(self.scheduled_tasks.items()):
                        # 执行已到期且状态为scheduled的任务
                        if (task.status == TaskStatus.SCHEDULED and 
                            task.schedule_time <= current_time):
                            task.status = TaskStatus.EXECUTING
                            task.updated_at = datetime.now()
                            tasks_to_execute.append(task)
                
                # 执行到期的任务
                for task in tasks_to_execute:
                    self._execute_scheduled_task(task)
                
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)
    
    def _execute_scheduled_task(self, task: ScheduledTask):
        """执行调度任务"""
        try:
            logger.info(f"Executing scheduled task {task.id}")
            
            # 这里应该调用训练管理器来执行任务
            # 由于我们是在重构，暂时模拟执行过程
            from modules.training.core.task_manager import TrainingTaskManager as TrainingManager
            training_manager = TrainingManager()
            
            # 提交训练任务
            training_task_id = training_manager.submit_training_task(task.config)
            
            # 更新任务状态
            self.update_task_status(
                task.id, 
                TaskStatus.COMPLETED, 
                result={"training_task_id": training_task_id}
            )
            
            logger.info(f"Scheduled task {task.id} executed with training task ID: {training_task_id}")
        except Exception as e:
            logger.error(f"Failed to execute scheduled task {task.id}: {e}")
            self.update_task_status(
                task.id, 
                TaskStatus.FAILED, 
                error_message=str(e)
            )

    def get_template_manager(self) -> ConfigTemplateManager:
        """获取配置模板管理器"""
        return self.config_template_manager


# 全局调度器实例
_scheduler = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> TrainingScheduler:
    """获取调度器实例"""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = TrainingScheduler()
        return _scheduler


def schedule_training_task(task_config: Dict[str, Any], 
                          schedule_time: datetime,
                          priority: TaskPriority = TaskPriority.NORMAL) -> str:
    """调度训练任务"""
    scheduler = get_scheduler()
    if not scheduler.running:
        scheduler.start()
    return scheduler.schedule_task(task_config, schedule_time, priority)
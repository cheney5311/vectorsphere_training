#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调度器服务层

提供调度任务的业务逻辑处理，包括任务调度、执行、监控等功能。
使用仓库层进行数据持久化。
"""

import logging
import threading
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ==================== 数据类 ====================

@dataclass
class TaskResult:
    """任务执行结果"""
    success: bool
    task_id: str
    status: str
    result: Dict[str, Any] = None
    error_message: str = None
    execution_time_seconds: float = None


# ==================== 服务类 ====================

class SchedulerService:
    """调度器服务
    
    提供任务调度、执行、监控等功能。
    委托仓库层进行数据持久化。
    """
    
    def __init__(self, config: Dict[str, Any] = None, use_memory_storage: bool = True):
        self.config = config or {}
        self._use_memory_storage = use_memory_storage
        
        # 调度器状态
        self._running = False
        self._scheduler_lock = threading.RLock()
        self._scheduler_thread = None
        
        # 初始化仓库
        self._init_repositories()
        
        # 执行中的任务
        self._executing_tasks: Dict[str, threading.Thread] = {}
        
        # 配置
        self._max_concurrent_tasks = self.config.get('max_concurrent_tasks', 10)
        self._check_interval = self.config.get('check_interval', 1)
        
        # 统计
        self._stats = {
            'tasks_scheduled': 0,
            'tasks_executed': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_cancelled': 0
        }
    
    def _init_repositories(self):
        """初始化仓库"""
        try:
            from backend.repositories.scheduler_repository import (
                get_scheduled_task_repository,
                get_execution_log_repository,
                get_task_template_repository,
                get_scheduler_metrics_repository
            )
            self._task_repo = get_scheduled_task_repository(use_memory=self._use_memory_storage)
            self._log_repo = get_execution_log_repository(use_memory=self._use_memory_storage)
            self._template_repo = get_task_template_repository(use_memory=self._use_memory_storage)
            self._metrics_repo = get_scheduler_metrics_repository(use_memory=self._use_memory_storage)
        except ImportError as e:
            logger.warning(f"Failed to import repositories: {e}")
            self._task_repo = None
            self._log_repo = None
            self._template_repo = None
            self._metrics_repo = None
    
    @property
    def running(self) -> bool:
        """调度器是否运行中"""
        return self._running
    
    # ==================== 调度器控制 ====================
    
    def start(self) -> bool:
        """启动调度器"""
        with self._scheduler_lock:
            if self._running:
                return True
            
            self._running = True
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="SchedulerService"
            )
            self._scheduler_thread.daemon = True
            self._scheduler_thread.start()
            logger.info("Scheduler service started")
            return True
    
    def stop(self) -> bool:
        """停止调度器"""
        with self._scheduler_lock:
            self._running = False
            if self._scheduler_thread:
                self._scheduler_thread.join(timeout=5)
            logger.info("Scheduler service stopped")
            return True
    
    def get_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        stats = self._task_repo.get_statistics() if self._task_repo else {}
        
        return {
            'running': self._running,
            'executing_count': len(self._executing_tasks),
            'max_concurrent': self._max_concurrent_tasks,
            'statistics': stats,
            'counters': self._stats.copy()
        }
    
    # ==================== 任务调度 ====================
    
    def schedule_task(
        self,
        task_config: Dict[str, Any],
        schedule_time: datetime,
        name: str = None,
        priority: str = "normal",
        task_type: str = "training",
        task_id: str = None,
        tenant_id: str = None,
        user_id: str = None,
        template_id: str = None,
        max_retries: int = 3,
        timeout_seconds: int = None,
        depends_on: List[str] = None,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """调度任务
        
        Args:
            task_config: 任务配置
            schedule_time: 调度时间
            name: 任务名称
            priority: 优先级
            task_type: 任务类型
            task_id: 任务ID
            tenant_id: 租户ID
            user_id: 用户ID
            template_id: 模板ID
            max_retries: 最大重试次数
            timeout_seconds: 超时秒数
            depends_on: 依赖的任务ID列表
            tags: 标签
            metadata: 元数据
            
        Returns:
            (是否成功, 消息, 任务数据)
        """
        try:
            if not self._task_repo:
                return False, "Repository not available", None
            
            # 验证调度时间
            if schedule_time < datetime.utcnow():
                return False, "Schedule time must be in the future", None
            
            # 如果使用模板，合并配置
            if template_id:
                template = self._template_repo.get_by_id(template_id, tenant_id) if self._template_repo else None
                if template:
                    base_config = template.get('config_template', {})
                    task_config = {**base_config, **task_config}
                    self._template_repo.increment_usage(template_id)
            
            # 检查任务ID是否已存在
            if task_id:
                existing = self._task_repo.get_by_id(task_id, tenant_id)
                if existing:
                    return False, f"Task with ID {task_id} already exists", None
            
            # 创建任务
            task_data = {
                'id': task_id,
                'tenant_id': tenant_id,
                'name': name or f"Task-{task_id or 'auto'}",
                'task_type': task_type,
                'schedule_type': 'once',
                'schedule_time': schedule_time,
                'status': 'scheduled',
                'priority': priority,
                'config': task_config,
                'template_id': template_id,
                'max_retries': max_retries,
                'timeout_seconds': timeout_seconds,
                'depends_on': depends_on or [],
                'tags': tags or [],
                'metadata': metadata or {},
                'created_by': user_id,
                'next_run_time': schedule_time
            }
            
            result = self._task_repo.create(task_data)
            if not result:
                return False, "Failed to create task", None
            
            self._stats['tasks_scheduled'] += 1
            logger.info(f"Task {result['id']} scheduled for {schedule_time}")
            
            return True, "Task scheduled successfully", result
            
        except Exception as e:
            logger.error(f"Failed to schedule task: {e}")
            return False, f"Schedule failed: {str(e)}", None
    
    def schedule_recurring_task(
        self,
        task_config: Dict[str, Any],
        cron_expression: str = None,
        interval_seconds: int = None,
        name: str = None,
        priority: str = "normal",
        task_type: str = "training",
        tenant_id: str = None,
        user_id: str = None,
        **kwargs
    ) -> Tuple[bool, str, Optional[Dict]]:
        """调度周期性任务
        
        Args:
            task_config: 任务配置
            cron_expression: Cron表达式
            interval_seconds: 间隔秒数
            其他参数同 schedule_task
            
        Returns:
            (是否成功, 消息, 任务数据)
        """
        try:
            if not self._task_repo:
                return False, "Repository not available", None
            
            if not cron_expression and not interval_seconds:
                return False, "Either cron_expression or interval_seconds is required", None
            
            # 确定调度类型
            schedule_type = 'cron' if cron_expression else 'interval'
            
            # 计算下次运行时间
            next_run_time = self._calculate_next_run_time(cron_expression, interval_seconds)
            
            task_data = {
                'tenant_id': tenant_id,
                'name': name or f"Recurring-{schedule_type}",
                'task_type': task_type,
                'schedule_type': schedule_type,
                'cron_expression': cron_expression,
                'interval_seconds': interval_seconds,
                'status': 'scheduled',
                'priority': priority,
                'config': task_config,
                'created_by': user_id,
                'next_run_time': next_run_time,
                **kwargs
            }
            
            result = self._task_repo.create(task_data)
            if not result:
                return False, "Failed to create recurring task", None
            
            self._stats['tasks_scheduled'] += 1
            logger.info(f"Recurring task {result['id']} scheduled")
            
            return True, "Recurring task scheduled successfully", result
            
        except Exception as e:
            logger.error(f"Failed to schedule recurring task: {e}")
            return False, f"Schedule failed: {str(e)}", None
    
    def _calculate_next_run_time(
        self,
        cron_expression: str = None,
        interval_seconds: int = None
    ) -> datetime:
        """计算下次运行时间"""
        if interval_seconds:
            return datetime.utcnow() + timedelta(seconds=interval_seconds)
        
        if cron_expression:
            # 简化的cron解析，实际应使用croniter库
            try:
                from croniter import croniter
                cron = croniter(cron_expression, datetime.utcnow())
                return cron.get_next(datetime)
            except ImportError:
                # 如果没有croniter库，默认1小时后
                return datetime.utcnow() + timedelta(hours=1)
        
        return datetime.utcnow() + timedelta(hours=1)
    
    # ==================== 任务管理 ====================
    
    def get_task(self, task_id: str, tenant_id: str = None) -> Optional[Dict]:
        """获取任务详情"""
        if not self._task_repo:
            return None
        return self._task_repo.get_by_id(task_id, tenant_id)
    
    def list_tasks(
        self,
        tenant_id: str = None,
        status: str = None,
        priority: str = None,
        task_type: str = None,
        created_by: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """列出任务"""
        if not self._task_repo:
            return []
        return self._task_repo.get_all(
            tenant_id=tenant_id,
            status=status,
            priority=priority,
            task_type=task_type,
            created_by=created_by,
            limit=limit,
            offset=offset
        )
    
    def cancel_task(self, task_id: str, tenant_id: str = None, user_id: str = None) -> Tuple[bool, str]:
        """取消任务"""
        try:
            if not self._task_repo:
                return False, "Repository not available"
            
            task = self._task_repo.get_by_id(task_id, tenant_id)
            if not task:
                return False, "Task not found"
            
            status = task.get('status')
            if status in ['completed', 'failed', 'cancelled']:
                return False, f"Cannot cancel task in {status} status"
            
            # 如果正在执行，尝试中断
            if task_id in self._executing_tasks:
                # 标记为取消，执行线程会检查这个状态
                pass
            
            success = self._task_repo.update_status(task_id, 'cancelled', tenant_id=tenant_id)
            if success:
                self._stats['tasks_cancelled'] += 1
                logger.info(f"Task {task_id} cancelled by {user_id}")
                return True, "Task cancelled successfully"
            
            return False, "Failed to cancel task"
            
        except Exception as e:
            logger.error(f"Failed to cancel task {task_id}: {e}")
            return False, f"Cancel failed: {str(e)}"
    
    def pause_task(self, task_id: str, tenant_id: str = None) -> Tuple[bool, str]:
        """暂停任务"""
        try:
            if not self._task_repo:
                return False, "Repository not available"
            
            task = self._task_repo.get_by_id(task_id, tenant_id)
            if not task:
                return False, "Task not found"
            
            if task.get('status') != 'scheduled':
                return False, "Only scheduled tasks can be paused"
            
            success = self._task_repo.update_status(task_id, 'paused', tenant_id=tenant_id)
            return (True, "Task paused") if success else (False, "Failed to pause")
            
        except Exception as e:
            logger.error(f"Failed to pause task {task_id}: {e}")
            return False, f"Pause failed: {str(e)}"
    
    def resume_task(self, task_id: str, tenant_id: str = None) -> Tuple[bool, str]:
        """恢复任务"""
        try:
            if not self._task_repo:
                return False, "Repository not available"
            
            task = self._task_repo.get_by_id(task_id, tenant_id)
            if not task:
                return False, "Task not found"
            
            if task.get('status') != 'paused':
                return False, "Only paused tasks can be resumed"
            
            success = self._task_repo.update_status(task_id, 'scheduled', tenant_id=tenant_id)
            return (True, "Task resumed") if success else (False, "Failed to resume")
            
        except Exception as e:
            logger.error(f"Failed to resume task {task_id}: {e}")
            return False, f"Resume failed: {str(e)}"
    
    def retry_task(self, task_id: str, tenant_id: str = None) -> Tuple[bool, str]:
        """重试任务"""
        try:
            if not self._task_repo:
                return False, "Repository not available"
            
            task = self._task_repo.get_by_id(task_id, tenant_id)
            if not task:
                return False, "Task not found"
            
            if task.get('status') not in ['failed', 'timeout']:
                return False, "Only failed or timeout tasks can be retried"
            
            # 重置状态并更新调度时间
            updates = {
                'status': 'scheduled',
                'schedule_time': datetime.utcnow(),
                'next_run_time': datetime.utcnow(),
                'retry_count': task.get('retry_count', 0) + 1,
                'error_message': None,
                'result': None
            }
            
            success = self._task_repo.update(task_id, updates, tenant_id)
            return (True, "Task scheduled for retry") if success else (False, "Failed to retry")
            
        except Exception as e:
            logger.error(f"Failed to retry task {task_id}: {e}")
            return False, f"Retry failed: {str(e)}"
    
    def update_task(
        self,
        task_id: str,
        updates: Dict[str, Any],
        tenant_id: str = None
    ) -> Tuple[bool, str]:
        """更新任务"""
        try:
            if not self._task_repo:
                return False, "Repository not available"
            
            task = self._task_repo.get_by_id(task_id, tenant_id)
            if not task:
                return False, "Task not found"
            
            # 只允许更新特定字段
            allowed_fields = {'name', 'description', 'priority', 'config', 'tags', 
                            'metadata', 'schedule_time', 'timeout_seconds', 'max_retries'}
            filtered_updates = {k: v for k, v in updates.items() if k in allowed_fields}
            
            if not filtered_updates:
                return False, "No valid fields to update"
            
            # 如果更新了调度时间，同步更新next_run_time
            if 'schedule_time' in filtered_updates:
                filtered_updates['next_run_time'] = filtered_updates['schedule_time']
            
            success = self._task_repo.update(task_id, filtered_updates, tenant_id)
            return (True, "Task updated") if success else (False, "Failed to update")
            
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {e}")
            return False, f"Update failed: {str(e)}"
    
    def delete_task(self, task_id: str, tenant_id: str = None) -> Tuple[bool, str]:
        """删除任务"""
        try:
            if not self._task_repo:
                return False, "Repository not available"
            
            task = self._task_repo.get_by_id(task_id, tenant_id)
            if not task:
                return False, "Task not found"
            
            # 不允许删除正在执行的任务
            if task.get('status') == 'executing':
                return False, "Cannot delete executing task"
            
            success = self._task_repo.delete(task_id, tenant_id)
            return (True, "Task deleted") if success else (False, "Failed to delete")
            
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            return False, f"Delete failed: {str(e)}"
    
    # ==================== 执行日志 ====================
    
    def get_task_logs(self, task_id: str, limit: int = 100) -> List[Dict]:
        """获取任务执行日志"""
        if not self._log_repo:
            return []
        return self._log_repo.get_by_task(task_id, limit)
    
    # ==================== 模板管理 ====================
    
    def create_template(
        self,
        name: str,
        config_template: Dict[str, Any],
        description: str = "",
        category: str = None,
        task_type: str = "training",
        tenant_id: str = None,
        user_id: str = None,
        **kwargs
    ) -> Tuple[bool, str, Optional[Dict]]:
        """创建模板"""
        try:
            if not self._template_repo:
                return False, "Repository not available", None
            
            # 检查名称是否已存在
            existing = self._template_repo.get_by_name(name, tenant_id)
            if existing:
                return False, "Template name already exists", None
            
            template_data = {
                'tenant_id': tenant_id,
                'name': name,
                'description': description,
                'category': category,
                'task_type': task_type,
                'config_template': config_template,
                'created_by': user_id,
                **kwargs
            }
            
            result = self._template_repo.create(template_data)
            if result:
                return True, "Template created", result
            return False, "Failed to create template", None
            
        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            return False, f"Create failed: {str(e)}", None
    
    def get_template(self, template_id: str, tenant_id: str = None) -> Optional[Dict]:
        """获取模板"""
        if not self._template_repo:
            return None
        return self._template_repo.get_by_id(template_id, tenant_id)
    
    def get_template_by_name(self, name: str, tenant_id: str = None) -> Optional[Dict]:
        """根据名称获取模板"""
        if not self._template_repo:
            return None
        return self._template_repo.get_by_name(name, tenant_id)
    
    def list_templates(
        self,
        tenant_id: str = None,
        category: str = None,
        include_system: bool = True
    ) -> List[Dict]:
        """列出模板"""
        if not self._template_repo:
            return []
        return self._template_repo.get_all(
            tenant_id=tenant_id,
            category=category,
            include_system=include_system
        )
    
    def update_template(
        self,
        template_id: str,
        updates: Dict[str, Any],
        tenant_id: str = None
    ) -> Tuple[bool, str]:
        """更新模板"""
        try:
            if not self._template_repo:
                return False, "Repository not available"
            
            success = self._template_repo.update(template_id, updates, tenant_id)
            return (True, "Template updated") if success else (False, "Failed to update or system template")
            
        except Exception as e:
            logger.error(f"Failed to update template: {e}")
            return False, f"Update failed: {str(e)}"
    
    def delete_template(self, template_id: str, tenant_id: str = None) -> Tuple[bool, str]:
        """删除模板"""
        try:
            if not self._template_repo:
                return False, "Repository not available"
            
            success = self._template_repo.delete(template_id, tenant_id)
            return (True, "Template deleted") if success else (False, "Failed to delete or system template")
            
        except Exception as e:
            logger.error(f"Failed to delete template: {e}")
            return False, f"Delete failed: {str(e)}"
    
    # ==================== 统计和指标 ====================
    
    def get_statistics(self, tenant_id: str = None) -> Dict[str, Any]:
        """获取统计信息"""
        if not self._task_repo:
            return {}
        
        stats = self._task_repo.get_statistics(tenant_id)
        stats['service_counters'] = self._stats.copy()
        stats['executing_count'] = len(self._executing_tasks)
        
        return stats
    
    def get_metrics(
        self,
        tenant_id: str = None,
        period_type: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[Dict]:
        """获取指标"""
        if not self._metrics_repo:
            return []
        return self._metrics_repo.get_metrics(
            tenant_id=tenant_id,
            period_type=period_type,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
    
    # ==================== 调度器主循环 ====================
    
    def _scheduler_loop(self):
        """调度器主循环"""
        while self._running:
            try:
                self._process_pending_tasks()
                self._check_timeout_tasks()
                self._record_metrics()
                time.sleep(self._check_interval)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)
    
    def _process_pending_tasks(self):
        """处理待执行的任务"""
        if not self._task_repo:
            return
        
        # 检查并发限制
        current_executing = len(self._executing_tasks)
        if current_executing >= self._max_concurrent_tasks:
            return
        
        # 获取待执行的任务
        pending_tasks = self._task_repo.get_pending_tasks(before_time=datetime.utcnow())
        
        for task in pending_tasks:
            if len(self._executing_tasks) >= self._max_concurrent_tasks:
                break
            
            # 检查依赖
            if not self._check_dependencies(task):
                continue
            
            # 执行任务
            self._execute_task(task)
    
    def _check_dependencies(self, task: Dict) -> bool:
        """检查任务依赖是否满足"""
        depends_on = task.get('depends_on', [])
        if not depends_on:
            return True
        
        for dep_task_id in depends_on:
            dep_task = self._task_repo.get_by_id(dep_task_id)
            if not dep_task or dep_task.get('status') != 'completed':
                return False
        
        return True
    
    def _execute_task(self, task: Dict):
        """执行任务"""
        task_id = task['id']
        
        # 更新状态为执行中
        self._task_repo.update_status(task_id, 'executing')
        
        # 创建执行日志
        log_data = {
            'task_id': task_id,
            'tenant_id': task.get('tenant_id'),
            'execution_number': task.get('retry_count', 0) + 1,
            'status': 'executing',
            'started_at': datetime.utcnow()
        }
        if self._log_repo:
            self._log_repo.create(log_data)
        
        # 在线程中执行
        thread = threading.Thread(
            target=self._run_task,
            args=(task,),
            name=f"Task-{task_id}"
        )
        thread.daemon = True
        self._executing_tasks[task_id] = thread
        thread.start()
        
        self._stats['tasks_executed'] += 1
    
    def _run_task(self, task: Dict):
        """运行任务"""
        task_id = task['id']
        tenant_id = task.get('tenant_id')
        start_time = datetime.utcnow()
        
        try:
            # 获取任务配置
            config = task.get('config', {})
            task_type = task.get('task_type', 'training')
            
            # 根据任务类型执行
            result = self._execute_task_by_type(task_type, config, task)
            
            # 计算执行时间
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # 更新状态
            self._task_repo.update_status(
                task_id,
                'completed',
                result=result,
                tenant_id=tenant_id
            )
            
            # 更新执行时间
            self._task_repo.update(task_id, {
                'execution_time_seconds': execution_time,
                'last_run_time': datetime.utcnow()
            }, tenant_id)
            
            self._stats['tasks_completed'] += 1
            logger.info(f"Task {task_id} completed in {execution_time:.2f}s")
            
            # 处理周期性任务
            self._handle_recurring_task(task)
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # 检查是否可以重试
            retry_count = task.get('retry_count', 0)
            max_retries = task.get('max_retries', 3)
            
            if retry_count < max_retries:
                # 调度重试
                self._schedule_retry(task)
            else:
                # 标记为失败
                self._task_repo.update_status(
                    task_id,
                    'failed',
                    error_message=str(e),
                    tenant_id=tenant_id
                )
                self._stats['tasks_failed'] += 1
        
        finally:
            # 从执行列表中移除
            if task_id in self._executing_tasks:
                del self._executing_tasks[task_id]
    
    def _execute_task_by_type(
        self,
        task_type: str,
        config: Dict[str, Any],
        task: Dict
    ) -> Dict[str, Any]:
        """根据类型执行任务"""
        if task_type == 'training':
            return self._execute_training_task(config, task)
        elif task_type == 'evaluation':
            return self._execute_evaluation_task(config, task)
        elif task_type == 'inference':
            return self._execute_inference_task(config, task)
        elif task_type == 'data_processing':
            return self._execute_data_processing_task(config, task)
        elif task_type == 'cleanup':
            return self._execute_cleanup_task(config, task)
        else:
            return self._execute_custom_task(config, task)
    
    def _execute_training_task(self, config: Dict, task: Dict) -> Dict:
        """执行训练任务"""
        try:
            from backend.modules.training.core.task_manager import TrainingTaskManager
            manager = TrainingTaskManager()
            training_id = manager.submit_training_task(config)
            return {'training_task_id': training_id}
        except ImportError:
            logger.warning("TrainingTaskManager not available, simulating execution")
            time.sleep(1)  # 模拟执行
            return {'status': 'simulated', 'message': 'Training task simulated'}
    
    def _execute_evaluation_task(self, config: Dict, task: Dict) -> Dict:
        """执行评估任务"""
        logger.info(f"Executing evaluation task with config: {config}")
        return {'status': 'completed', 'type': 'evaluation'}
    
    def _execute_inference_task(self, config: Dict, task: Dict) -> Dict:
        """执行推理任务"""
        logger.info(f"Executing inference task with config: {config}")
        return {'status': 'completed', 'type': 'inference'}
    
    def _execute_data_processing_task(self, config: Dict, task: Dict) -> Dict:
        """执行数据处理任务"""
        logger.info(f"Executing data processing task with config: {config}")
        return {'status': 'completed', 'type': 'data_processing'}
    
    def _execute_cleanup_task(self, config: Dict, task: Dict) -> Dict:
        """执行清理任务"""
        logger.info(f"Executing cleanup task with config: {config}")
        return {'status': 'completed', 'type': 'cleanup'}
    
    def _execute_custom_task(self, config: Dict, task: Dict) -> Dict:
        """执行自定义任务"""
        logger.info(f"Executing custom task with config: {config}")
        return {'status': 'completed', 'type': 'custom'}
    
    def _handle_recurring_task(self, task: Dict):
        """处理周期性任务"""
        schedule_type = task.get('schedule_type')
        if schedule_type not in ['cron', 'interval']:
            return
        
        # 计算下次运行时间
        next_run = self._calculate_next_run_time(
            task.get('cron_expression'),
            task.get('interval_seconds')
        )
        
        # 更新任务
        self._task_repo.update(task['id'], {
            'status': 'scheduled',
            'next_run_time': next_run,
            'schedule_time': next_run
        }, task.get('tenant_id'))
    
    def _schedule_retry(self, task: Dict):
        """调度重试"""
        retry_delay = task.get('retry_delay_seconds', 60)
        next_run = datetime.utcnow() + timedelta(seconds=retry_delay)
        
        self._task_repo.update(task['id'], {
            'status': 'scheduled',
            'retry_count': task.get('retry_count', 0) + 1,
            'schedule_time': next_run,
            'next_run_time': next_run
        }, task.get('tenant_id'))
        
        logger.info(f"Task {task['id']} scheduled for retry at {next_run}")
    
    def _check_timeout_tasks(self):
        """检查超时任务"""
        if not self._task_repo:
            return
        
        executing_tasks = self._task_repo.get_by_status('executing')
        now = datetime.utcnow()
        
        for task in executing_tasks:
            timeout = task.get('timeout_seconds')
            if not timeout:
                continue
            
            started_at = task.get('started_at')
            if not started_at:
                continue
            
            if isinstance(started_at, str):
                started_at = datetime.fromisoformat(started_at)
            
            if (now - started_at).total_seconds() > timeout:
                self._task_repo.update_status(
                    task['id'],
                    'timeout',
                    error_message='Task execution timed out',
                    tenant_id=task.get('tenant_id')
                )
                logger.warning(f"Task {task['id']} timed out")
    
    def _record_metrics(self):
        """记录指标"""
        # 每分钟记录一次
        pass


# ==================== 单例获取函数 ====================

_scheduler_service = None
_service_lock = threading.Lock()


def get_scheduler_service(config: Dict[str, Any] = None, use_memory: bool = True) -> SchedulerService:
    """获取调度器服务实例"""
    global _scheduler_service
    with _service_lock:
        if _scheduler_service is None:
            _scheduler_service = SchedulerService(config, use_memory_storage=use_memory)
        return _scheduler_service


def reset_scheduler_service():
    """重置服务实例（用于测试）"""
    global _scheduler_service
    with _service_lock:
        if _scheduler_service:
            _scheduler_service.stop()
        _scheduler_service = None

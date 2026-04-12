#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调度器仓库层

提供调度任务、执行日志、模板等数据的持久化存储。
支持内存存储和数据库存储两种模式。
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class ScheduledTaskRepository:
    """调度任务仓库"""
    
    def __init__(self, use_memory: bool = True, db_session=None):
        self._use_memory = use_memory
        self._db_session = db_session
        self._memory_store: Dict[str, Dict] = {}
        
        if not use_memory:
            self._init_db()
    
    def _init_db(self):
        """初始化数据库连接"""
        try:
            if self._db_session is None:
                from backend.services.database_service import get_database_manager
                db_manager = get_database_manager()
                if db_manager:
                    self._db_session = db_manager.get_db_session()
        except Exception as e:
            logger.warning(f"Failed to init database, falling back to memory: {e}")
            self._use_memory = True
    
    def _generate_id(self) -> str:
        """生成任务ID"""
        return f"task_{uuid.uuid4().hex[:12]}"
    
    def create(self, task_data: Dict[str, Any]) -> Optional[Dict]:
        """创建任务"""
        try:
            task_id = task_data.get('id') or self._generate_id()
            now = datetime.utcnow()
            
            task = {
                'id': task_id,
                'tenant_id': task_data.get('tenant_id'),
                'name': task_data.get('name', f'Task-{task_id}'),
                'description': task_data.get('description', ''),
                'task_type': task_data.get('task_type', 'training'),
                'schedule_type': task_data.get('schedule_type', 'once'),
                'schedule_time': task_data.get('schedule_time'),
                'cron_expression': task_data.get('cron_expression'),
                'interval_seconds': task_data.get('interval_seconds'),
                'status': task_data.get('status', 'pending'),
                'priority': task_data.get('priority', 'normal'),
                'execution_mode': task_data.get('execution_mode', 'async'),
                'config': task_data.get('config', {}),
                'template_id': task_data.get('template_id'),
                'max_retries': task_data.get('max_retries', 3),
                'retry_count': task_data.get('retry_count', 0),
                'retry_delay_seconds': task_data.get('retry_delay_seconds', 60),
                'timeout_seconds': task_data.get('timeout_seconds'),
                'depends_on': task_data.get('depends_on', []),
                'result': task_data.get('result'),
                'error_message': task_data.get('error_message'),
                'tags': task_data.get('tags', []),
                'metadata': task_data.get('metadata', {}),
                'created_by': task_data.get('created_by'),
                'updated_by': task_data.get('updated_by'),
                'created_at': now,
                'updated_at': now,
                'started_at': None,
                'completed_at': None,
                'execution_time_seconds': None,
                'next_run_time': task_data.get('schedule_time'),
                'last_run_time': None
            }
            
            if self._use_memory:
                self._memory_store[task_id] = task
                return task
            else:
                return self._create_db(task)
                
        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            return None
    
    def _create_db(self, task: Dict) -> Optional[Dict]:
        """数据库创建"""
        try:
            from backend.schemas.scheduler_models import ScheduledTaskModel, TaskStatusEnum, TaskPriorityEnum
            
            model = ScheduledTaskModel(
                id=task['id'],
                tenant_id=task['tenant_id'],
                name=task['name'],
                description=task['description'],
                task_type=task['task_type'],
                schedule_type=task['schedule_type'],
                schedule_time=task['schedule_time'],
                cron_expression=task['cron_expression'],
                interval_seconds=task['interval_seconds'],
                status=TaskStatusEnum(task['status']) if task['status'] else TaskStatusEnum.PENDING,
                priority=TaskPriorityEnum(task['priority']) if task['priority'] else TaskPriorityEnum.NORMAL,
                config=task['config'],
                template_id=task['template_id'],
                max_retries=task['max_retries'],
                retry_count=task['retry_count'],
                timeout_seconds=task['timeout_seconds'],
                depends_on=task['depends_on'],
                tags=task['tags'],
                metadata=task['metadata'],
                created_by=task['created_by'],
                created_at=task['created_at'],
                updated_at=task['updated_at'],
                next_run_time=task['next_run_time']
            )
            
            self._db_session.add(model)
            self._db_session.commit()
            return model.to_dict()
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB create failed: {e}")
            return None
    
    def get_by_id(self, task_id: str, tenant_id: str = None) -> Optional[Dict]:
        """根据ID获取任务"""
        try:
            if self._use_memory:
                task = self._memory_store.get(task_id)
                if task and (tenant_id is None or task.get('tenant_id') == tenant_id):
                    return task
                return None
            else:
                return self._get_by_id_db(task_id, tenant_id)
                
        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return None
    
    def _get_by_id_db(self, task_id: str, tenant_id: str = None) -> Optional[Dict]:
        """数据库查询"""
        try:
            from backend.schemas.scheduler_models import ScheduledTaskModel
            
            query = self._db_session.query(ScheduledTaskModel).filter_by(id=task_id)
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            
            model = query.first()
            return model.to_dict() if model else None
            
        except Exception as e:
            logger.error(f"DB get failed: {e}")
            return None
    
    def get_all(
        self,
        tenant_id: str = None,
        status: str = None,
        priority: str = None,
        task_type: str = None,
        created_by: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """获取任务列表"""
        try:
            if self._use_memory:
                tasks = list(self._memory_store.values())
                
                if tenant_id:
                    tasks = [t for t in tasks if t.get('tenant_id') == tenant_id]
                if status:
                    tasks = [t for t in tasks if t.get('status') == status]
                if priority:
                    tasks = [t for t in tasks if t.get('priority') == priority]
                if task_type:
                    tasks = [t for t in tasks if t.get('task_type') == task_type]
                if created_by:
                    tasks = [t for t in tasks if t.get('created_by') == created_by]
                
                # 按创建时间倒序排序
                tasks.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
                
                return tasks[offset:offset + limit]
            else:
                return self._get_all_db(tenant_id, status, priority, task_type, created_by, limit, offset)
                
        except Exception as e:
            logger.error(f"Failed to get tasks: {e}")
            return []
    
    def _get_all_db(
        self,
        tenant_id: str,
        status: str,
        priority: str,
        task_type: str,
        created_by: str,
        limit: int,
        offset: int
    ) -> List[Dict]:
        """数据库查询列表"""
        try:
            from backend.schemas.scheduler_models import ScheduledTaskModel
            
            query = self._db_session.query(ScheduledTaskModel)
            
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            if status:
                query = query.filter_by(status=status)
            if priority:
                query = query.filter_by(priority=priority)
            if task_type:
                query = query.filter_by(task_type=task_type)
            if created_by:
                query = query.filter_by(created_by=created_by)
            
            query = query.order_by(ScheduledTaskModel.created_at.desc())
            query = query.offset(offset).limit(limit)
            
            return [m.to_dict() for m in query.all()]
            
        except Exception as e:
            logger.error(f"DB get all failed: {e}")
            return []
    
    def update(self, task_id: str, updates: Dict[str, Any], tenant_id: str = None) -> bool:
        """更新任务"""
        try:
            updates['updated_at'] = datetime.utcnow()
            
            if self._use_memory:
                if task_id not in self._memory_store:
                    return False
                task = self._memory_store[task_id]
                if tenant_id and task.get('tenant_id') != tenant_id:
                    return False
                task.update(updates)
                return True
            else:
                return self._update_db(task_id, updates, tenant_id)
                
        except Exception as e:
            logger.error(f"Failed to update task {task_id}: {e}")
            return False
    
    def _update_db(self, task_id: str, updates: Dict, tenant_id: str = None) -> bool:
        """数据库更新"""
        try:
            from backend.schemas.scheduler_models import ScheduledTaskModel
            
            query = self._db_session.query(ScheduledTaskModel).filter_by(id=task_id)
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            
            count = query.update(updates)
            self._db_session.commit()
            return count > 0
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB update failed: {e}")
            return False
    
    def update_status(
        self,
        task_id: str,
        status: str,
        result: Dict = None,
        error_message: str = None,
        tenant_id: str = None
    ) -> bool:
        """更新任务状态"""
        updates = {
            'status': status,
            'updated_at': datetime.utcnow()
        }
        
        if result is not None:
            updates['result'] = result
        if error_message is not None:
            updates['error_message'] = error_message
        
        # 更新时间字段
        if status in ['executing']:
            updates['started_at'] = datetime.utcnow()
        elif status in ['completed', 'failed', 'cancelled', 'timeout']:
            updates['completed_at'] = datetime.utcnow()
            # 计算执行时间
            task = self.get_by_id(task_id, tenant_id)
            if task and task.get('started_at'):
                started = task['started_at']
                if isinstance(started, str):
                    started = datetime.fromisoformat(started)
                updates['execution_time_seconds'] = (datetime.utcnow() - started).total_seconds()
        
        return self.update(task_id, updates, tenant_id)
    
    def delete(self, task_id: str, tenant_id: str = None) -> bool:
        """删除任务"""
        try:
            if self._use_memory:
                if task_id in self._memory_store:
                    task = self._memory_store[task_id]
                    if tenant_id and task.get('tenant_id') != tenant_id:
                        return False
                    del self._memory_store[task_id]
                    return True
                return False
            else:
                return self._delete_db(task_id, tenant_id)
                
        except Exception as e:
            logger.error(f"Failed to delete task {task_id}: {e}")
            return False
    
    def _delete_db(self, task_id: str, tenant_id: str = None) -> bool:
        """数据库删除"""
        try:
            from backend.schemas.scheduler_models import ScheduledTaskModel
            
            query = self._db_session.query(ScheduledTaskModel).filter_by(id=task_id)
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            
            count = query.delete()
            self._db_session.commit()
            return count > 0
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB delete failed: {e}")
            return False
    
    def get_pending_tasks(self, before_time: datetime = None, tenant_id: str = None) -> List[Dict]:
        """获取待执行的任务"""
        try:
            if before_time is None:
                before_time = datetime.utcnow()
            
            if self._use_memory:
                tasks = []
                for task in self._memory_store.values():
                    if tenant_id and task.get('tenant_id') != tenant_id:
                        continue
                    if task.get('status') != 'scheduled':
                        continue
                    schedule_time = task.get('schedule_time')
                    if schedule_time:
                        if isinstance(schedule_time, str):
                            schedule_time = datetime.fromisoformat(schedule_time)
                        if schedule_time <= before_time:
                            tasks.append(task)
                
                # 按优先级和调度时间排序
                priority_order = {'critical': 0, 'urgent': 1, 'high': 2, 'normal': 3, 'low': 4}
                tasks.sort(key=lambda x: (
                    priority_order.get(x.get('priority', 'normal'), 3),
                    x.get('schedule_time', datetime.max)
                ))
                return tasks
            else:
                return self._get_pending_tasks_db(before_time, tenant_id)
                
        except Exception as e:
            logger.error(f"Failed to get pending tasks: {e}")
            return []
    
    def _get_pending_tasks_db(self, before_time: datetime, tenant_id: str = None) -> List[Dict]:
        """数据库查询待执行任务"""
        try:
            from backend.schemas.scheduler_models import ScheduledTaskModel, TaskStatusEnum
            
            query = self._db_session.query(ScheduledTaskModel).filter(
                ScheduledTaskModel.status == TaskStatusEnum.SCHEDULED,
                ScheduledTaskModel.schedule_time <= before_time
            )
            
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            
            query = query.order_by(
                ScheduledTaskModel.priority.desc(),
                ScheduledTaskModel.schedule_time.asc()
            )
            
            return [m.to_dict() for m in query.all()]
            
        except Exception as e:
            logger.error(f"DB get pending failed: {e}")
            return []
    
    def get_by_status(self, status: str, tenant_id: str = None, limit: int = 100) -> List[Dict]:
        """根据状态获取任务"""
        return self.get_all(tenant_id=tenant_id, status=status, limit=limit)
    
    def increment_retry(self, task_id: str, tenant_id: str = None) -> bool:
        """增加重试次数"""
        task = self.get_by_id(task_id, tenant_id)
        if not task:
            return False
        
        return self.update(task_id, {
            'retry_count': task.get('retry_count', 0) + 1,
            'status': 'retrying'
        }, tenant_id)
    
    def count(self, tenant_id: str = None, status: str = None) -> int:
        """统计任务数量"""
        try:
            if self._use_memory:
                tasks = list(self._memory_store.values())
                if tenant_id:
                    tasks = [t for t in tasks if t.get('tenant_id') == tenant_id]
                if status:
                    tasks = [t for t in tasks if t.get('status') == status]
                return len(tasks)
            else:
                return self._count_db(tenant_id, status)
                
        except Exception as e:
            logger.error(f"Failed to count tasks: {e}")
            return 0
    
    def _count_db(self, tenant_id: str = None, status: str = None) -> int:
        """数据库统计"""
        try:
            from backend.schemas.scheduler_models import ScheduledTaskModel
            
            query = self._db_session.query(ScheduledTaskModel)
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            if status:
                query = query.filter_by(status=status)
            
            return query.count()
            
        except Exception as e:
            logger.error(f"DB count failed: {e}")
            return 0
    
    def get_statistics(self, tenant_id: str = None) -> Dict[str, Any]:
        """获取任务统计"""
        try:
            statuses = ['pending', 'scheduled', 'queued', 'executing', 'completed', 'failed', 'cancelled']
            stats = {
                'total': 0,
                'by_status': {},
                'by_priority': {},
                'by_type': {}
            }
            
            tasks = self.get_all(tenant_id=tenant_id, limit=10000)
            stats['total'] = len(tasks)
            
            for task in tasks:
                status = task.get('status', 'unknown')
                priority = task.get('priority', 'normal')
                task_type = task.get('task_type', 'training')
                
                stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
                stats['by_priority'][priority] = stats['by_priority'].get(priority, 0) + 1
                stats['by_type'][task_type] = stats['by_type'].get(task_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'total': 0, 'by_status': {}, 'by_priority': {}, 'by_type': {}}


class TaskExecutionLogRepository:
    """任务执行日志仓库"""
    
    def __init__(self, use_memory: bool = True, db_session=None):
        self._use_memory = use_memory
        self._db_session = db_session
        self._memory_store: Dict[str, Dict] = {}
    
    def _generate_id(self) -> str:
        """生成日志ID"""
        return f"log_{uuid.uuid4().hex[:12]}"
    
    def create(self, log_data: Dict[str, Any]) -> Optional[Dict]:
        """创建执行日志"""
        try:
            log_id = log_data.get('id') or self._generate_id()
            now = datetime.utcnow()
            
            log = {
                'id': log_id,
                'task_id': log_data.get('task_id'),
                'tenant_id': log_data.get('tenant_id'),
                'execution_number': log_data.get('execution_number', 1),
                'status': log_data.get('status'),
                'started_at': log_data.get('started_at', now),
                'completed_at': log_data.get('completed_at'),
                'execution_time_seconds': log_data.get('execution_time_seconds'),
                'result': log_data.get('result'),
                'error_message': log_data.get('error_message'),
                'error_stack': log_data.get('error_stack'),
                'resource_usage': log_data.get('resource_usage'),
                'log_output': log_data.get('log_output'),
                'metadata': log_data.get('metadata', {}),
                'created_at': now
            }
            
            if self._use_memory:
                self._memory_store[log_id] = log
                return log
            else:
                return self._create_db(log)
                
        except Exception as e:
            logger.error(f"Failed to create execution log: {e}")
            return None
    
    def _create_db(self, log: Dict) -> Optional[Dict]:
        """数据库创建"""
        try:
            from backend.schemas.scheduler_models import TaskExecutionLogModel, TaskStatusEnum
            
            model = TaskExecutionLogModel(
                id=log['id'],
                task_id=log['task_id'],
                tenant_id=log['tenant_id'],
                execution_number=log['execution_number'],
                status=TaskStatusEnum(log['status']) if log['status'] else None,
                started_at=log['started_at'],
                completed_at=log['completed_at'],
                execution_time_seconds=log['execution_time_seconds'],
                result=log['result'],
                error_message=log['error_message'],
                error_stack=log['error_stack'],
                resource_usage=log['resource_usage'],
                log_output=log['log_output'],
                metadata=log['metadata'],
                created_at=log['created_at']
            )
            
            self._db_session.add(model)
            self._db_session.commit()
            return model.to_dict()
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB create log failed: {e}")
            return None
    
    def get_by_task(self, task_id: str, limit: int = 100) -> List[Dict]:
        """获取任务的执行日志"""
        try:
            if self._use_memory:
                logs = [l for l in self._memory_store.values() if l.get('task_id') == task_id]
                logs.sort(key=lambda x: x.get('created_at', datetime.min), reverse=True)
                return logs[:limit]
            else:
                return self._get_by_task_db(task_id, limit)
                
        except Exception as e:
            logger.error(f"Failed to get logs for task {task_id}: {e}")
            return []
    
    def _get_by_task_db(self, task_id: str, limit: int) -> List[Dict]:
        """数据库查询"""
        try:
            from backend.schemas.scheduler_models import TaskExecutionLogModel
            
            query = self._db_session.query(TaskExecutionLogModel).filter_by(task_id=task_id)
            query = query.order_by(TaskExecutionLogModel.created_at.desc())
            query = query.limit(limit)
            
            return [m.to_dict() for m in query.all()]
            
        except Exception as e:
            logger.error(f"DB get logs failed: {e}")
            return []
    
    def update(self, log_id: str, updates: Dict[str, Any]) -> bool:
        """更新日志"""
        try:
            if self._use_memory:
                if log_id in self._memory_store:
                    self._memory_store[log_id].update(updates)
                    return True
                return False
            else:
                from backend.schemas.scheduler_models import TaskExecutionLogModel
                count = self._db_session.query(TaskExecutionLogModel).filter_by(id=log_id).update(updates)
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to update log {log_id}: {e}")
            return False


class TaskTemplateRepository:
    """任务模板仓库"""
    
    def __init__(self, use_memory: bool = True, db_session=None):
        self._use_memory = use_memory
        self._db_session = db_session
        self._memory_store: Dict[str, Dict] = {}
        
        # 加载内置模板
        self._load_builtin_templates()
    
    def _load_builtin_templates(self):
        """加载内置模板"""
        try:
            from backend.modules.scheduler.templates import ConfigTemplateManager
            template_manager = ConfigTemplateManager()
            
            for name in template_manager.list_templates():
                template_id = f"builtin_{name}"
                config = template_manager.get_template(name)
                
                self._memory_store[template_id] = {
                    'id': template_id,
                    'name': name,
                    'description': self._get_template_description(name),
                    'category': 'builtin',
                    'task_type': 'training',
                    'config_template': config,
                    'default_priority': 'normal',
                    'is_active': True,
                    'is_system': True,
                    'usage_count': 0,
                    'tags': ['builtin'],
                    'created_at': datetime.utcnow(),
                    'updated_at': datetime.utcnow()
                }
        except Exception as e:
            logger.warning(f"Failed to load builtin templates: {e}")
    
    def _get_template_description(self, name: str) -> str:
        """获取模板描述"""
        descriptions = {
            'basic_text_generation': '基础文本生成模型训练',
            'moe_training': 'MoE (Mixture of Experts) 大模型训练',
            'multimodal_training': '多模态训练（文本+图像）',
            'distributed_training': '分布式训练配置',
            'knowledge_distillation': '知识蒸馏训练',
            'model_compression': '模型压缩（量化+剪枝）',
            'hyperparameter_search': '超参数搜索优化',
            'lr_finder': '学习率查找器',
            'database_training': '数据库驱动的训练',
            'production_config': '生产环境配置'
        }
        return descriptions.get(name, name)
    
    def _generate_id(self) -> str:
        """生成模板ID"""
        return f"template_{uuid.uuid4().hex[:12]}"
    
    def create(self, template_data: Dict[str, Any]) -> Optional[Dict]:
        """创建模板"""
        try:
            template_id = template_data.get('id') or self._generate_id()
            now = datetime.utcnow()
            
            template = {
                'id': template_id,
                'tenant_id': template_data.get('tenant_id'),
                'name': template_data.get('name'),
                'description': template_data.get('description', ''),
                'category': template_data.get('category'),
                'task_type': template_data.get('task_type', 'training'),
                'config_template': template_data.get('config_template', {}),
                'default_priority': template_data.get('default_priority', 'normal'),
                'default_timeout_seconds': template_data.get('default_timeout_seconds'),
                'default_max_retries': template_data.get('default_max_retries', 3),
                'parameters': template_data.get('parameters'),
                'is_active': template_data.get('is_active', True),
                'is_system': template_data.get('is_system', False),
                'usage_count': 0,
                'tags': template_data.get('tags', []),
                'metadata': template_data.get('metadata', {}),
                'created_by': template_data.get('created_by'),
                'created_at': now,
                'updated_at': now
            }
            
            if self._use_memory:
                self._memory_store[template_id] = template
                return template
            else:
                return self._create_db(template)
                
        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            return None
    
    def _create_db(self, template: Dict) -> Optional[Dict]:
        """数据库创建"""
        try:
            from backend.schemas.scheduler_models import TaskTemplateModel
            
            model = TaskTemplateModel(**template)
            self._db_session.add(model)
            self._db_session.commit()
            return model.to_dict()
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB create template failed: {e}")
            return None
    
    def get_by_id(self, template_id: str, tenant_id: str = None) -> Optional[Dict]:
        """根据ID获取模板"""
        try:
            if self._use_memory:
                template = self._memory_store.get(template_id)
                if template:
                    # 系统模板对所有租户可见
                    if template.get('is_system'):
                        return template
                    if tenant_id is None or template.get('tenant_id') == tenant_id:
                        return template
                return None
            else:
                return self._get_by_id_db(template_id, tenant_id)
                
        except Exception as e:
            logger.error(f"Failed to get template {template_id}: {e}")
            return None
    
    def _get_by_id_db(self, template_id: str, tenant_id: str = None) -> Optional[Dict]:
        """数据库查询"""
        try:
            from backend.schemas.scheduler_models import TaskTemplateModel
            
            model = self._db_session.query(TaskTemplateModel).filter_by(id=template_id).first()
            if model:
                if model.is_system or tenant_id is None or model.tenant_id == tenant_id:
                    return model.to_dict()
            return None
            
        except Exception as e:
            logger.error(f"DB get template failed: {e}")
            return None
    
    def get_by_name(self, name: str, tenant_id: str = None) -> Optional[Dict]:
        """根据名称获取模板"""
        try:
            if self._use_memory:
                for template in self._memory_store.values():
                    if template.get('name') == name:
                        if template.get('is_system'):
                            return template
                        if tenant_id is None or template.get('tenant_id') == tenant_id:
                            return template
                return None
            else:
                from backend.schemas.scheduler_models import TaskTemplateModel
                model = self._db_session.query(TaskTemplateModel).filter_by(name=name).first()
                if model and (model.is_system or tenant_id is None or model.tenant_id == tenant_id):
                    return model.to_dict()
                return None
                
        except Exception as e:
            logger.error(f"Failed to get template by name {name}: {e}")
            return None
    
    def get_all(
        self,
        tenant_id: str = None,
        category: str = None,
        is_active: bool = None,
        include_system: bool = True
    ) -> List[Dict]:
        """获取模板列表"""
        try:
            if self._use_memory:
                templates = []
                for template in self._memory_store.values():
                    # 过滤系统模板
                    if not include_system and template.get('is_system'):
                        continue
                    # 租户过滤
                    if not template.get('is_system') and tenant_id and template.get('tenant_id') != tenant_id:
                        continue
                    if category and template.get('category') != category:
                        continue
                    if is_active is not None and template.get('is_active') != is_active:
                        continue
                    templates.append(template)
                
                return templates
            else:
                return self._get_all_db(tenant_id, category, is_active, include_system)
                
        except Exception as e:
            logger.error(f"Failed to get templates: {e}")
            return []
    
    def _get_all_db(
        self,
        tenant_id: str,
        category: str,
        is_active: bool,
        include_system: bool
    ) -> List[Dict]:
        """数据库查询"""
        try:
            from backend.schemas.scheduler_models import TaskTemplateModel
            from sqlalchemy import or_
            
            query = self._db_session.query(TaskTemplateModel)
            
            if tenant_id and include_system:
                query = query.filter(
                    or_(
                        TaskTemplateModel.tenant_id == tenant_id,
                        TaskTemplateModel.is_system == True
                    )
                )
            elif tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            
            if category:
                query = query.filter_by(category=category)
            if is_active is not None:
                query = query.filter_by(is_active=is_active)
            
            return [m.to_dict() for m in query.all()]
            
        except Exception as e:
            logger.error(f"DB get templates failed: {e}")
            return []
    
    def update(self, template_id: str, updates: Dict[str, Any], tenant_id: str = None) -> bool:
        """更新模板"""
        try:
            updates['updated_at'] = datetime.utcnow()
            
            if self._use_memory:
                if template_id not in self._memory_store:
                    return False
                template = self._memory_store[template_id]
                # 不允许更新系统模板
                if template.get('is_system'):
                    return False
                if tenant_id and template.get('tenant_id') != tenant_id:
                    return False
                template.update(updates)
                return True
            else:
                from backend.schemas.scheduler_models import TaskTemplateModel
                query = self._db_session.query(TaskTemplateModel).filter_by(id=template_id, is_system=False)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                count = query.update(updates)
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to update template {template_id}: {e}")
            return False
    
    def delete(self, template_id: str, tenant_id: str = None) -> bool:
        """删除模板"""
        try:
            if self._use_memory:
                if template_id not in self._memory_store:
                    return False
                template = self._memory_store[template_id]
                # 不允许删除系统模板
                if template.get('is_system'):
                    return False
                if tenant_id and template.get('tenant_id') != tenant_id:
                    return False
                del self._memory_store[template_id]
                return True
            else:
                from backend.schemas.scheduler_models import TaskTemplateModel
                query = self._db_session.query(TaskTemplateModel).filter_by(id=template_id, is_system=False)
                if tenant_id:
                    query = query.filter_by(tenant_id=tenant_id)
                count = query.delete()
                self._db_session.commit()
                return count > 0
                
        except Exception as e:
            logger.error(f"Failed to delete template {template_id}: {e}")
            return False
    
    def increment_usage(self, template_id: str) -> bool:
        """增加使用次数"""
        try:
            if self._use_memory:
                if template_id in self._memory_store:
                    self._memory_store[template_id]['usage_count'] = \
                        self._memory_store[template_id].get('usage_count', 0) + 1
                    return True
                return False
            else:
                from backend.schemas.scheduler_models import TaskTemplateModel
                query = self._db_session.query(TaskTemplateModel).filter_by(id=template_id)
                query.update({'usage_count': TaskTemplateModel.usage_count + 1})
                self._db_session.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to increment usage for {template_id}: {e}")
            return False


class SchedulerMetricsRepository:
    """调度器指标仓库"""
    
    def __init__(self, use_memory: bool = True, db_session=None):
        self._use_memory = use_memory
        self._db_session = db_session
        self._memory_store: List[Dict] = []
    
    def _generate_id(self) -> str:
        """生成指标ID"""
        return f"metrics_{uuid.uuid4().hex[:12]}"
    
    def record_metrics(self, metrics_data: Dict[str, Any]) -> Optional[Dict]:
        """记录指标"""
        try:
            metrics_id = self._generate_id()
            now = datetime.utcnow()
            
            metrics = {
                'id': metrics_id,
                'tenant_id': metrics_data.get('tenant_id'),
                'period_start': metrics_data.get('period_start'),
                'period_end': metrics_data.get('period_end'),
                'period_type': metrics_data.get('period_type', 'minute'),
                'tasks_scheduled': metrics_data.get('tasks_scheduled', 0),
                'tasks_executed': metrics_data.get('tasks_executed', 0),
                'tasks_completed': metrics_data.get('tasks_completed', 0),
                'tasks_failed': metrics_data.get('tasks_failed', 0),
                'tasks_cancelled': metrics_data.get('tasks_cancelled', 0),
                'tasks_timeout': metrics_data.get('tasks_timeout', 0),
                'avg_wait_time_seconds': metrics_data.get('avg_wait_time_seconds'),
                'avg_execution_time_seconds': metrics_data.get('avg_execution_time_seconds'),
                'max_execution_time_seconds': metrics_data.get('max_execution_time_seconds'),
                'min_execution_time_seconds': metrics_data.get('min_execution_time_seconds'),
                'avg_cpu_usage': metrics_data.get('avg_cpu_usage'),
                'avg_memory_usage': metrics_data.get('avg_memory_usage'),
                'peak_concurrent_tasks': metrics_data.get('peak_concurrent_tasks'),
                'created_at': now
            }
            
            if self._use_memory:
                self._memory_store.append(metrics)
                # 保持最近1000条记录
                if len(self._memory_store) > 1000:
                    self._memory_store = self._memory_store[-1000:]
                return metrics
            else:
                return self._record_db(metrics)
                
        except Exception as e:
            logger.error(f"Failed to record metrics: {e}")
            return None
    
    def _record_db(self, metrics: Dict) -> Optional[Dict]:
        """数据库记录"""
        try:
            from backend.schemas.scheduler_models import SchedulerMetricsModel
            
            model = SchedulerMetricsModel(**metrics)
            self._db_session.add(model)
            self._db_session.commit()
            return model.to_dict()
            
        except Exception as e:
            self._db_session.rollback()
            logger.error(f"DB record metrics failed: {e}")
            return None
    
    def get_metrics(
        self,
        tenant_id: str = None,
        period_type: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> List[Dict]:
        """获取指标"""
        try:
            if self._use_memory:
                metrics = self._memory_store.copy()
                
                if tenant_id:
                    metrics = [m for m in metrics if m.get('tenant_id') == tenant_id]
                if period_type:
                    metrics = [m for m in metrics if m.get('period_type') == period_type]
                if start_time:
                    metrics = [m for m in metrics if m.get('period_start') and m['period_start'] >= start_time]
                if end_time:
                    metrics = [m for m in metrics if m.get('period_end') and m['period_end'] <= end_time]
                
                metrics.sort(key=lambda x: x.get('period_start', datetime.min), reverse=True)
                return metrics[:limit]
            else:
                return self._get_metrics_db(tenant_id, period_type, start_time, end_time, limit)
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return []
    
    def _get_metrics_db(
        self,
        tenant_id: str,
        period_type: str,
        start_time: datetime,
        end_time: datetime,
        limit: int
    ) -> List[Dict]:
        """数据库查询"""
        try:
            from backend.schemas.scheduler_models import SchedulerMetricsModel
            
            query = self._db_session.query(SchedulerMetricsModel)
            
            if tenant_id:
                query = query.filter_by(tenant_id=tenant_id)
            if period_type:
                query = query.filter_by(period_type=period_type)
            if start_time:
                query = query.filter(SchedulerMetricsModel.period_start >= start_time)
            if end_time:
                query = query.filter(SchedulerMetricsModel.period_end <= end_time)
            
            query = query.order_by(SchedulerMetricsModel.period_start.desc())
            query = query.limit(limit)
            
            return [m.to_dict() for m in query.all()]
            
        except Exception as e:
            logger.error(f"DB get metrics failed: {e}")
            return []


# ==================== 单例获取函数 ====================

_task_repo = None
_log_repo = None
_template_repo = None
_metrics_repo = None


def get_scheduled_task_repository(use_memory: bool = True) -> ScheduledTaskRepository:
    """获取任务仓库实例"""
    global _task_repo
    if _task_repo is None:
        _task_repo = ScheduledTaskRepository(use_memory=use_memory)
    return _task_repo


def get_execution_log_repository(use_memory: bool = True) -> TaskExecutionLogRepository:
    """获取执行日志仓库实例"""
    global _log_repo
    if _log_repo is None:
        _log_repo = TaskExecutionLogRepository(use_memory=use_memory)
    return _log_repo


def get_task_template_repository(use_memory: bool = True) -> TaskTemplateRepository:
    """获取模板仓库实例"""
    global _template_repo
    if _template_repo is None:
        _template_repo = TaskTemplateRepository(use_memory=use_memory)
    return _template_repo


def get_scheduler_metrics_repository(use_memory: bool = True) -> SchedulerMetricsRepository:
    """获取指标仓库实例"""
    global _metrics_repo
    if _metrics_repo is None:
        _metrics_repo = SchedulerMetricsRepository(use_memory=use_memory)
    return _metrics_repo


def reset_scheduler_repositories():
    """重置所有仓库实例（用于测试）"""
    global _task_repo, _log_repo, _template_repo, _metrics_repo
    _task_repo = None
    _log_repo = None
    _template_repo = None
    _metrics_repo = None

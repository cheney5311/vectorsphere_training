"""工作流数据访问层

提供工作流相关的数据库访问功能，包括：
- 工作流定义 (Workflow)
- 工作流执行记录 (WorkflowExecution)
- 工作流步骤 (WorkflowStep)
- 工作流模板 (WorkflowTemplate)
- 工作流日志 (WorkflowLog)
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

from backend.core.exceptions import ValidationError, DatabaseError
from backend.schemas.workflow_models import (
    Workflow, WorkflowExecution, WorkflowStep,
    WorkflowTemplate, WorkflowLog, WorkflowVariable,
    WorkflowStatus, ExecutionStatus, StepStatus
)

logger = logging.getLogger(__name__)


def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


class WorkflowRepository:
    """工作流数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化工作流仓库
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._workflows: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._workflows: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, workflow_data: Dict[str, Any]) -> Workflow:
        """创建工作流"""
        try:
            workflow_id = workflow_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                workflow_data['id'] = workflow_id
                workflow_data['created_at'] = datetime.utcnow().isoformat()
                workflow_data['updated_at'] = datetime.utcnow().isoformat()
                workflow_data.setdefault('status', 'draft')
                workflow_data.setdefault('version', 1)
                workflow_data.setdefault('execution_count', 0)
                workflow_data.setdefault('success_count', 0)
                workflow_data.setdefault('failure_count', 0)
                self._workflows[workflow_id] = workflow_data
                return workflow_data
            
            with self._db_manager.get_db_session() as db:
                # 处理JSON字段
                config = workflow_data.get('config', {})
                if isinstance(config, dict):
                    config = json.dumps(config)
                
                steps_config = workflow_data.get('steps_config', [])
                if isinstance(steps_config, (list, dict)):
                    steps_config = json.dumps(steps_config)
                
                trigger_config = workflow_data.get('trigger_config', {})
                if isinstance(trigger_config, dict):
                    trigger_config = json.dumps(trigger_config)
                
                notification_config = workflow_data.get('notification_config', {})
                if isinstance(notification_config, dict):
                    notification_config = json.dumps(notification_config)
                
                tags = workflow_data.get('tags', [])
                if isinstance(tags, list):
                    tags = json.dumps(tags)
                
                workflow = Workflow(
                    id=workflow_id,
                    tenant_id=workflow_data.get('tenant_id'),
                    name=workflow_data['name'],
                    description=workflow_data.get('description'),
                    workflow_type=workflow_data['workflow_type'],
                    status=workflow_data.get('status', 'draft'),
                    version=workflow_data.get('version', 1),
                    created_by=workflow_data['created_by'],
                    updated_by=workflow_data.get('updated_by'),
                    config=config,
                    steps_config=steps_config,
                    trigger_config=trigger_config,
                    notification_config=notification_config,
                    schedule_enabled=workflow_data.get('schedule_enabled', False),
                    schedule_cron=workflow_data.get('schedule_cron'),
                    schedule_timezone=workflow_data.get('schedule_timezone', 'UTC'),
                    timeout_seconds=workflow_data.get('timeout_seconds', 3600),
                    max_retries=workflow_data.get('max_retries', 3),
                    retry_delay_seconds=workflow_data.get('retry_delay_seconds', 60),
                    tags=tags,
                    category=workflow_data.get('category'),
                    template_id=workflow_data.get('template_id'),
                    is_template=workflow_data.get('is_template', False)
                )
                db.add(workflow)
                db.commit()
                db.refresh(workflow)
                return workflow
                
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            raise DatabaseError(f"Failed to create workflow: {e}", operation="create_workflow")
    
    def get_by_id(self, workflow_id: str) -> Optional[Workflow]:
        """根据ID获取工作流"""
        try:
            if self._use_memory_storage:
                return self._workflows.get(workflow_id)
            
            with self._db_manager.get_db_session() as db:
                return db.query(Workflow).filter(Workflow.id == workflow_id).first()
                
        except Exception as e:
            logger.error(f"Failed to get workflow: {e}")
            return None
    
    def get_by_tenant(self, tenant_id: str, workflow_type: Optional[str] = None,
                     status: Optional[str] = None, search: Optional[str] = None,
                     limit: int = 100, offset: int = 0) -> Tuple[List[Workflow], int]:
        """获取租户的工作流列表"""
        try:
            if self._use_memory_storage:
                workflows = [w for w in self._workflows.values() if w.get('tenant_id') == tenant_id]
                if workflow_type:
                    workflows = [w for w in workflows if w.get('workflow_type') == workflow_type]
                if status:
                    workflows = [w for w in workflows if w.get('status') == status]
                if search:
                    search_lower = search.lower()
                    workflows = [w for w in workflows if search_lower in w.get('name', '').lower() or search_lower in w.get('description', '').lower()]
                
                # 按创建时间排序
                workflows.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(workflows)
                return workflows[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(Workflow).filter(Workflow.tenant_id == tenant_id)
                
                if workflow_type:
                    query = query.filter(Workflow.workflow_type == workflow_type)
                if status:
                    query = query.filter(Workflow.status == status)
                if search:
                    search_pattern = f"%{search}%"
                    query = query.filter(
                        (Workflow.name.ilike(search_pattern)) |
                        (Workflow.description.ilike(search_pattern))
                    )
                
                total = query.count()
                workflows = query.order_by(Workflow.created_at.desc()).offset(offset).limit(limit).all()
                return workflows, total
                
        except Exception as e:
            logger.error(f"Failed to get workflows by tenant: {e}")
            return [], 0
    
    def get_by_user(self, user_id: str, tenant_id: Optional[str] = None,
                   limit: int = 100, offset: int = 0) -> Tuple[List[Workflow], int]:
        """获取用户创建的工作流"""
        try:
            if self._use_memory_storage:
                workflows = [w for w in self._workflows.values() if w.get('created_by') == user_id]
                if tenant_id:
                    workflows = [w for w in workflows if w.get('tenant_id') == tenant_id]
                workflows.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(workflows)
                return workflows[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(Workflow).filter(Workflow.created_by == user_id)
                if tenant_id:
                    query = query.filter(Workflow.tenant_id == tenant_id)
                
                total = query.count()
                workflows = query.order_by(Workflow.created_at.desc()).offset(offset).limit(limit).all()
                return workflows, total
                
        except Exception as e:
            logger.error(f"Failed to get workflows by user: {e}")
            return [], 0
    
    def update(self, workflow_id: str, update_data: Dict[str, Any]) -> Optional[Workflow]:
        """更新工作流"""
        try:
            if self._use_memory_storage:
                if workflow_id in self._workflows:
                    self._workflows[workflow_id].update(update_data)
                    self._workflows[workflow_id]['updated_at'] = datetime.utcnow().isoformat()
                    return self._workflows[workflow_id]
                return None
            
            with self._db_manager.get_db_session() as db:
                workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
                if not workflow:
                    return None
                
                for key, value in update_data.items():
                    if key in ('config', 'steps_config', 'trigger_config', 'notification_config') and isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    if key == 'tags' and isinstance(value, list):
                        value = json.dumps(value)
                    if hasattr(workflow, key):
                        setattr(workflow, key, value)
                
                workflow.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(workflow)
                return workflow
                
        except Exception as e:
            logger.error(f"Failed to update workflow: {e}")
            raise DatabaseError(f"Failed to update workflow: {e}", operation="update_workflow")
    
    def delete(self, workflow_id: str) -> bool:
        """删除工作流"""
        try:
            if self._use_memory_storage:
                if workflow_id in self._workflows:
                    del self._workflows[workflow_id]
                    return True
                return False
            
            with self._db_manager.get_db_session() as db:
                workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
                if not workflow:
                    return False
                
                db.delete(workflow)
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete workflow: {e}")
            return False
    
    def increment_execution_count(self, workflow_id: str, success: bool = True) -> bool:
        """增加执行计数"""
        try:
            if self._use_memory_storage:
                if workflow_id in self._workflows:
                    self._workflows[workflow_id]['execution_count'] = self._workflows[workflow_id].get('execution_count', 0) + 1
                    if success:
                        self._workflows[workflow_id]['success_count'] = self._workflows[workflow_id].get('success_count', 0) + 1
                    else:
                        self._workflows[workflow_id]['failure_count'] = self._workflows[workflow_id].get('failure_count', 0) + 1
                    return True
                return False
            
            with self._db_manager.get_db_session() as db:
                workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
                if not workflow:
                    return False
                
                workflow.execution_count = (workflow.execution_count or 0) + 1
                if success:
                    workflow.success_count = (workflow.success_count or 0) + 1
                else:
                    workflow.failure_count = (workflow.failure_count or 0) + 1
                
                workflow.last_run_at = datetime.utcnow()
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to increment execution count: {e}")
            return False
    
    def get_statistics(self, tenant_id: str) -> Dict[str, Any]:
        """获取工作流统计信息"""
        try:
            if self._use_memory_storage:
                workflows = [w for w in self._workflows.values() if w.get('tenant_id') == tenant_id]
                
                type_counts = {}
                status_counts = {}
                total_executions = 0
                total_success = 0
                total_failure = 0
                
                for w in workflows:
                    wtype = w.get('workflow_type', 'unknown')
                    type_counts[wtype] = type_counts.get(wtype, 0) + 1
                    
                    wstatus = w.get('status', 'unknown')
                    status_counts[wstatus] = status_counts.get(wstatus, 0) + 1
                    
                    total_executions += w.get('execution_count', 0)
                    total_success += w.get('success_count', 0)
                    total_failure += w.get('failure_count', 0)
                
                return {
                    'total_workflows': len(workflows),
                    'by_type': type_counts,
                    'by_status': status_counts,
                    'total_executions': total_executions,
                    'total_success': total_success,
                    'total_failure': total_failure,
                    'success_rate': round(total_success / total_executions * 100, 2) if total_executions > 0 else 0
                }
            
            with self._db_manager.get_db_session() as db:
                from sqlalchemy import func as sql_func
                
                # 总数
                total = db.query(sql_func.count(Workflow.id)).filter(Workflow.tenant_id == tenant_id).scalar() or 0
                
                # 按类型统计
                type_results = db.query(
                    Workflow.workflow_type,
                    sql_func.count(Workflow.id)
                ).filter(Workflow.tenant_id == tenant_id).group_by(Workflow.workflow_type).all()
                type_counts = {r[0]: r[1] for r in type_results}
                
                # 按状态统计
                status_results = db.query(
                    Workflow.status,
                    sql_func.count(Workflow.id)
                ).filter(Workflow.tenant_id == tenant_id).group_by(Workflow.status).all()
                status_counts = {r[0]: r[1] for r in status_results}
                
                # 执行统计
                exec_stats = db.query(
                    sql_func.sum(Workflow.execution_count),
                    sql_func.sum(Workflow.success_count),
                    sql_func.sum(Workflow.failure_count)
                ).filter(Workflow.tenant_id == tenant_id).first()
                
                total_executions = exec_stats[0] or 0
                total_success = exec_stats[1] or 0
                total_failure = exec_stats[2] or 0
                
                return {
                    'total_workflows': total,
                    'by_type': type_counts,
                    'by_status': status_counts,
                    'total_executions': total_executions,
                    'total_success': total_success,
                    'total_failure': total_failure,
                    'success_rate': round(total_success / total_executions * 100, 2) if total_executions > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get workflow statistics: {e}")
            return {}


class WorkflowExecutionRepository:
    """工作流执行记录数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化执行记录仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._executions: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._executions: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, execution_data: Dict[str, Any]) -> WorkflowExecution:
        """创建执行记录"""
        try:
            execution_id = execution_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                execution_data['id'] = execution_id
                execution_data['created_at'] = datetime.utcnow().isoformat()
                execution_data['updated_at'] = datetime.utcnow().isoformat()
                execution_data.setdefault('status', 'pending')
                execution_data.setdefault('progress', 0)
                execution_data.setdefault('retry_count', 0)
                self._executions[execution_id] = execution_data
                return execution_data
            
            with self._db_manager.get_db_session() as db:
                # 处理JSON字段
                input_data = execution_data.get('input_data', {})
                if isinstance(input_data, dict):
                    input_data = json.dumps(input_data)
                
                output_data = execution_data.get('output_data', {})
                if isinstance(output_data, dict):
                    output_data = json.dumps(output_data)
                
                context_data = execution_data.get('context_data', {})
                if isinstance(context_data, dict):
                    context_data = json.dumps(context_data)
                
                error_details = execution_data.get('error_details', {})
                if isinstance(error_details, dict):
                    error_details = json.dumps(error_details)
                
                execution = WorkflowExecution(
                    id=execution_id,
                    tenant_id=execution_data.get('tenant_id'),
                    workflow_id=execution_data['workflow_id'],
                    workflow_name=execution_data.get('workflow_name'),
                    workflow_version=execution_data.get('workflow_version'),
                    status=execution_data.get('status', 'pending'),
                    progress=execution_data.get('progress', 0),
                    current_step=execution_data.get('current_step'),
                    current_step_index=execution_data.get('current_step_index', 0),
                    total_steps=execution_data.get('total_steps', 0),
                    triggered_by=execution_data.get('triggered_by'),
                    trigger_type=execution_data.get('trigger_type', 'manual'),
                    started_at=execution_data.get('started_at'),
                    input_data=input_data,
                    output_data=output_data,
                    context_data=context_data,
                    error_details=error_details,
                    priority=execution_data.get('priority', 5)
                )
                db.add(execution)
                db.commit()
                db.refresh(execution)
                return execution
                
        except Exception as e:
            logger.error(f"Failed to create execution: {e}")
            raise DatabaseError(f"Failed to create execution: {e}", operation="create_execution")
    
    def get_by_id(self, execution_id: str) -> Optional[WorkflowExecution]:
        """根据ID获取执行记录"""
        try:
            if self._use_memory_storage:
                return self._executions.get(execution_id)
            
            with self._db_manager.get_db_session() as db:
                return db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
                
        except Exception as e:
            logger.error(f"Failed to get execution: {e}")
            return None
    
    def get_by_workflow(self, workflow_id: str, status: Optional[str] = None,
                       limit: int = 100, offset: int = 0) -> Tuple[List[WorkflowExecution], int]:
        """获取工作流的执行记录"""
        try:
            if self._use_memory_storage:
                executions = [e for e in self._executions.values() if e.get('workflow_id') == workflow_id]
                if status:
                    executions = [e for e in executions if e.get('status') == status]
                executions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(executions)
                return executions[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(WorkflowExecution).filter(WorkflowExecution.workflow_id == workflow_id)
                if status:
                    query = query.filter(WorkflowExecution.status == status)
                
                total = query.count()
                executions = query.order_by(WorkflowExecution.created_at.desc()).offset(offset).limit(limit).all()
                return executions, total
                
        except Exception as e:
            logger.error(f"Failed to get executions by workflow: {e}")
            return [], 0
    
    def get_by_tenant(self, tenant_id: str, workflow_id: Optional[str] = None,
                     status: Optional[str] = None, triggered_by: Optional[str] = None,
                     limit: int = 100, offset: int = 0) -> Tuple[List[WorkflowExecution], int]:
        """获取租户的执行记录"""
        try:
            if self._use_memory_storage:
                executions = [e for e in self._executions.values() if e.get('tenant_id') == tenant_id]
                if workflow_id:
                    executions = [e for e in executions if e.get('workflow_id') == workflow_id]
                if status:
                    executions = [e for e in executions if e.get('status') == status]
                if triggered_by:
                    executions = [e for e in executions if e.get('triggered_by') == triggered_by]
                executions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(executions)
                return executions[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(WorkflowExecution).filter(WorkflowExecution.tenant_id == tenant_id)
                if workflow_id:
                    query = query.filter(WorkflowExecution.workflow_id == workflow_id)
                if status:
                    query = query.filter(WorkflowExecution.status == status)
                if triggered_by:
                    query = query.filter(WorkflowExecution.triggered_by == triggered_by)
                
                total = query.count()
                executions = query.order_by(WorkflowExecution.created_at.desc()).offset(offset).limit(limit).all()
                return executions, total
                
        except Exception as e:
            logger.error(f"Failed to get executions by tenant: {e}")
            return [], 0
    
    def update(self, execution_id: str, update_data: Dict[str, Any]) -> Optional[WorkflowExecution]:
        """更新执行记录"""
        try:
            if self._use_memory_storage:
                if execution_id in self._executions:
                    self._executions[execution_id].update(update_data)
                    self._executions[execution_id]['updated_at'] = datetime.utcnow().isoformat()
                    return self._executions[execution_id]
                return None
            
            with self._db_manager.get_db_session() as db:
                execution = db.query(WorkflowExecution).filter(WorkflowExecution.id == execution_id).first()
                if not execution:
                    return None
                
                for key, value in update_data.items():
                    if key in ('input_data', 'output_data', 'context_data', 'error_details') and isinstance(value, dict):
                        value = json.dumps(value)
                    if hasattr(execution, key):
                        setattr(execution, key, value)
                
                execution.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(execution)
                return execution
                
        except Exception as e:
            logger.error(f"Failed to update execution: {e}")
            raise DatabaseError(f"Failed to update execution: {e}", operation="update_execution")
    
    def update_status(self, execution_id: str, status: str, 
                     error_message: Optional[str] = None,
                     output_data: Optional[Dict] = None) -> Optional[WorkflowExecution]:
        """更新执行状态"""
        update_data = {'status': status}
        
        if status in ('completed', 'failed', 'cancelled', 'timeout'):
            update_data['completed_at'] = datetime.utcnow()
            # 计算执行时长
            execution = self.get_by_id(execution_id)
            if execution:
                started_at = execution.get('started_at') if isinstance(execution, dict) else execution.started_at
                if started_at:
                    if isinstance(started_at, str):
                        started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    duration = (datetime.utcnow() - started_at).total_seconds()
                    update_data['duration_seconds'] = duration
        
        if error_message:
            update_data['error_message'] = error_message
        
        if output_data:
            update_data['output_data'] = output_data
        
        return self.update(execution_id, update_data)
    
    def update_progress(self, execution_id: str, progress: float, 
                       current_step: Optional[str] = None,
                       current_step_index: Optional[int] = None) -> Optional[WorkflowExecution]:
        """更新执行进度"""
        update_data = {'progress': progress}
        if current_step:
            update_data['current_step'] = current_step
        if current_step_index is not None:
            update_data['current_step_index'] = current_step_index
        return self.update(execution_id, update_data)
    
    def get_running_executions(self, tenant_id: Optional[str] = None) -> List[WorkflowExecution]:
        """获取正在运行的执行记录"""
        try:
            if self._use_memory_storage:
                executions = [e for e in self._executions.values() if e.get('status') in ('pending', 'running', 'queued')]
                if tenant_id:
                    executions = [e for e in executions if e.get('tenant_id') == tenant_id]
                return executions
            
            with self._db_manager.get_db_session() as db:
                query = db.query(WorkflowExecution).filter(
                    WorkflowExecution.status.in_(['pending', 'running', 'queued'])
                )
                if tenant_id:
                    query = query.filter(WorkflowExecution.tenant_id == tenant_id)
                return query.all()
                
        except Exception as e:
            logger.error(f"Failed to get running executions: {e}")
            return []
    
    def get_execution_statistics(self, tenant_id: str, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """获取执行统计"""
        try:
            if self._use_memory_storage:
                executions = [e for e in self._executions.values() if e.get('tenant_id') == tenant_id]
                if workflow_id:
                    executions = [e for e in executions if e.get('workflow_id') == workflow_id]
                
                status_counts = {}
                total_duration = 0
                completed_count = 0
                
                for e in executions:
                    status = e.get('status', 'unknown')
                    status_counts[status] = status_counts.get(status, 0) + 1
                    
                    if e.get('duration_seconds'):
                        total_duration += e['duration_seconds']
                        completed_count += 1
                
                return {
                    'total_executions': len(executions),
                    'by_status': status_counts,
                    'avg_duration_seconds': round(total_duration / completed_count, 2) if completed_count > 0 else 0,
                    'success_rate': round(status_counts.get('completed', 0) / len(executions) * 100, 2) if executions else 0
                }
            
            with self._db_manager.get_db_session() as db:
                from sqlalchemy import func as sql_func
                
                query = db.query(WorkflowExecution).filter(WorkflowExecution.tenant_id == tenant_id)
                if workflow_id:
                    query = query.filter(WorkflowExecution.workflow_id == workflow_id)
                
                total = query.count()
                
                # 按状态统计
                status_results = db.query(
                    WorkflowExecution.status,
                    sql_func.count(WorkflowExecution.id)
                ).filter(WorkflowExecution.tenant_id == tenant_id)
                
                if workflow_id:
                    status_results = status_results.filter(WorkflowExecution.workflow_id == workflow_id)
                
                status_results = status_results.group_by(WorkflowExecution.status).all()
                status_counts = {r[0]: r[1] for r in status_results}
                
                # 平均时长
                avg_duration = db.query(sql_func.avg(WorkflowExecution.duration_seconds)).filter(
                    WorkflowExecution.tenant_id == tenant_id,
                    WorkflowExecution.duration_seconds.isnot(None)
                )
                if workflow_id:
                    avg_duration = avg_duration.filter(WorkflowExecution.workflow_id == workflow_id)
                avg_duration = avg_duration.scalar() or 0
                
                return {
                    'total_executions': total,
                    'by_status': status_counts,
                    'avg_duration_seconds': round(avg_duration, 2),
                    'success_rate': round(status_counts.get('completed', 0) / total * 100, 2) if total > 0 else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get execution statistics: {e}")
            return {}


class WorkflowStepRepository:
    """工作流步骤数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化步骤仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._steps: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._steps: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, step_data: Dict[str, Any]) -> WorkflowStep:
        """创建步骤记录"""
        try:
            step_id = step_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                step_data['id'] = step_id
                step_data['created_at'] = datetime.utcnow().isoformat()
                step_data['updated_at'] = datetime.utcnow().isoformat()
                step_data.setdefault('status', 'pending')
                self._steps[step_id] = step_data
                return step_data
            
            with self._db_manager.get_db_session() as db:
                input_data = step_data.get('input_data', {})
                if isinstance(input_data, dict):
                    input_data = json.dumps(input_data)
                
                output_data = step_data.get('output_data', {})
                if isinstance(output_data, dict):
                    output_data = json.dumps(output_data)
                
                config = step_data.get('config', {})
                if isinstance(config, dict):
                    config = json.dumps(config)
                
                error_details = step_data.get('error_details', {})
                if isinstance(error_details, dict):
                    error_details = json.dumps(error_details)
                
                step = WorkflowStep(
                    id=step_id,
                    execution_id=step_data['execution_id'],
                    workflow_id=step_data['workflow_id'],
                    step_name=step_data['step_name'],
                    step_type=step_data.get('step_type'),
                    step_index=step_data.get('step_index', 0),
                    status=step_data.get('status', 'pending'),
                    input_data=input_data,
                    output_data=output_data,
                    config=config,
                    error_details=error_details,
                    max_retries=step_data.get('max_retries', 3)
                )
                db.add(step)
                db.commit()
                db.refresh(step)
                return step
                
        except Exception as e:
            logger.error(f"Failed to create step: {e}")
            raise DatabaseError(f"Failed to create step: {e}", operation="create_step")
    
    def get_by_execution(self, execution_id: str) -> List[WorkflowStep]:
        """获取执行的所有步骤"""
        try:
            if self._use_memory_storage:
                steps = [s for s in self._steps.values() if s.get('execution_id') == execution_id]
                steps.sort(key=lambda x: x.get('step_index', 0))
                return steps
            
            with self._db_manager.get_db_session() as db:
                return db.query(WorkflowStep).filter(
                    WorkflowStep.execution_id == execution_id
                ).order_by(WorkflowStep.step_index).all()
                
        except Exception as e:
            logger.error(f"Failed to get steps by execution: {e}")
            return []
    
    def update(self, step_id: str, update_data: Dict[str, Any]) -> Optional[WorkflowStep]:
        """更新步骤"""
        try:
            if self._use_memory_storage:
                if step_id in self._steps:
                    self._steps[step_id].update(update_data)
                    self._steps[step_id]['updated_at'] = datetime.utcnow().isoformat()
                    return self._steps[step_id]
                return None
            
            with self._db_manager.get_db_session() as db:
                step = db.query(WorkflowStep).filter(WorkflowStep.id == step_id).first()
                if not step:
                    return None
                
                for key, value in update_data.items():
                    if key in ('input_data', 'output_data', 'config', 'error_details') and isinstance(value, dict):
                        value = json.dumps(value)
                    if hasattr(step, key):
                        setattr(step, key, value)
                
                step.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(step)
                return step
                
        except Exception as e:
            logger.error(f"Failed to update step: {e}")
            raise DatabaseError(f"Failed to update step: {e}", operation="update_step")
    
    def update_status(self, step_id: str, status: str, 
                     error_message: Optional[str] = None,
                     output_data: Optional[Dict] = None) -> Optional[WorkflowStep]:
        """更新步骤状态"""
        update_data = {'status': status}
        
        if status == 'running':
            update_data['started_at'] = datetime.utcnow()
        elif status in ('completed', 'failed', 'skipped'):
            update_data['completed_at'] = datetime.utcnow()
            # 计算时长
            step = self._steps.get(step_id) if self._use_memory_storage else None
            if step:
                started_at = step.get('started_at')
                if started_at:
                    if isinstance(started_at, str):
                        started_at = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    update_data['duration_seconds'] = (datetime.utcnow() - started_at).total_seconds()
        
        if error_message:
            update_data['error_message'] = error_message
        
        if output_data:
            update_data['output_data'] = output_data
        
        return self.update(step_id, update_data)


class WorkflowTemplateRepository:
    """工作流模板数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化模板仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._templates: Dict[str, Dict] = {}
            self._db_manager = None
            self._init_default_templates()
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._templates: Dict[str, Dict] = {}
                self._db_manager = None
                self._init_default_templates()
    
    def _init_default_templates(self):
        """初始化默认模板"""
        default_templates = [
            {
                'id': 'tpl_data_preprocessing',
                'name': '数据预处理模板',
                'description': '标准数据预处理工作流模板，包含数据清洗、标准化、特征提取等步骤',
                'workflow_type': 'data_preprocessing',
                'is_public': True,
                'is_system': True,
                'category': '数据处理',
                'config': {
                    'timeout_seconds': 3600,
                    'max_retries': 3
                },
                'steps_config': [
                    {'name': 'load_data', 'type': 'data_load'},
                    {'name': 'clean_data', 'type': 'data_transform'},
                    {'name': 'normalize', 'type': 'data_transform'},
                    {'name': 'feature_extraction', 'type': 'data_transform'},
                    {'name': 'validate', 'type': 'data_validate'}
                ],
                'default_params': {
                    'batch_size': 100,
                    'normalize_method': 'standard'
                }
            },
            {
                'id': 'tpl_model_training',
                'name': '模型训练模板',
                'description': '标准模型训练工作流模板，包含数据准备、训练、验证等步骤',
                'workflow_type': 'model_training',
                'is_public': True,
                'is_system': True,
                'category': '模型训练',
                'config': {
                    'timeout_seconds': 7200,
                    'max_retries': 2
                },
                'steps_config': [
                    {'name': 'prepare_data', 'type': 'data_load'},
                    {'name': 'train_model', 'type': 'train'},
                    {'name': 'validate_model', 'type': 'evaluate'},
                    {'name': 'save_model', 'type': 'custom'}
                ],
                'default_params': {
                    'epochs': 10,
                    'learning_rate': 0.001,
                    'batch_size': 32
                }
            },
            {
                'id': 'tpl_model_evaluation',
                'name': '模型评估模板',
                'description': '标准模型评估工作流模板，用于评估模型性能',
                'workflow_type': 'model_evaluation',
                'is_public': True,
                'is_system': True,
                'category': '模型评估',
                'config': {
                    'timeout_seconds': 1800,
                    'max_retries': 2
                },
                'steps_config': [
                    {'name': 'load_model', 'type': 'data_load'},
                    {'name': 'load_test_data', 'type': 'data_load'},
                    {'name': 'evaluate', 'type': 'evaluate'},
                    {'name': 'generate_report', 'type': 'custom'}
                ],
                'default_params': {
                    'metrics': ['accuracy', 'precision', 'recall', 'f1']
                }
            },
            {
                'id': 'tpl_etl_pipeline',
                'name': 'ETL流水线模板',
                'description': 'ETL数据流水线模板，用于数据抽取、转换、加载',
                'workflow_type': 'etl',
                'is_public': True,
                'is_system': True,
                'category': '数据处理',
                'config': {
                    'timeout_seconds': 3600,
                    'max_retries': 3
                },
                'steps_config': [
                    {'name': 'extract', 'type': 'data_load'},
                    {'name': 'transform', 'type': 'data_transform'},
                    {'name': 'load', 'type': 'custom'}
                ],
                'default_params': {
                    'chunk_size': 1000
                }
            }
        ]
        
        for tpl in default_templates:
            tpl['created_at'] = datetime.utcnow().isoformat()
            tpl['updated_at'] = datetime.utcnow().isoformat()
            tpl['created_by'] = 'system'
            tpl['use_count'] = 0
            tpl['version'] = '1.0.0'
            self._templates[tpl['id']] = tpl
    
    def create(self, template_data: Dict[str, Any]) -> WorkflowTemplate:
        """创建模板"""
        try:
            template_id = template_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                template_data['id'] = template_id
                template_data['created_at'] = datetime.utcnow().isoformat()
                template_data['updated_at'] = datetime.utcnow().isoformat()
                template_data.setdefault('use_count', 0)
                template_data.setdefault('version', '1.0.0')
                self._templates[template_id] = template_data
                return template_data
            
            with self._db_manager.get_db_session() as db:
                config = template_data.get('config', {})
                if isinstance(config, dict):
                    config = json.dumps(config)
                
                steps_config = template_data.get('steps_config', [])
                if isinstance(steps_config, list):
                    steps_config = json.dumps(steps_config)
                
                default_params = template_data.get('default_params', {})
                if isinstance(default_params, dict):
                    default_params = json.dumps(default_params)
                
                tags = template_data.get('tags', [])
                if isinstance(tags, list):
                    tags = json.dumps(tags)
                
                template = WorkflowTemplate(
                    id=template_id,
                    tenant_id=template_data.get('tenant_id'),
                    name=template_data['name'],
                    description=template_data.get('description'),
                    workflow_type=template_data['workflow_type'],
                    created_by=template_data['created_by'],
                    config=config,
                    steps_config=steps_config,
                    default_params=default_params,
                    is_public=template_data.get('is_public', False),
                    is_system=template_data.get('is_system', False),
                    version=template_data.get('version', '1.0.0'),
                    tags=tags,
                    category=template_data.get('category'),
                    icon=template_data.get('icon'),
                    thumbnail=template_data.get('thumbnail')
                )
                db.add(template)
                db.commit()
                db.refresh(template)
                return template
                
        except Exception as e:
            logger.error(f"Failed to create template: {e}")
            raise DatabaseError(f"Failed to create template: {e}", operation="create_template")
    
    def get_by_id(self, template_id: str) -> Optional[WorkflowTemplate]:
        """根据ID获取模板"""
        try:
            if self._use_memory_storage:
                return self._templates.get(template_id)
            
            with self._db_manager.get_db_session() as db:
                return db.query(WorkflowTemplate).filter(WorkflowTemplate.id == template_id).first()
                
        except Exception as e:
            logger.error(f"Failed to get template: {e}")
            return None
    
    def get_available_templates(self, tenant_id: Optional[str] = None,
                               workflow_type: Optional[str] = None,
                               category: Optional[str] = None,
                               limit: int = 100, offset: int = 0) -> Tuple[List[WorkflowTemplate], int]:
        """获取可用模板（公开模板 + 租户模板）"""
        try:
            if self._use_memory_storage:
                templates = list(self._templates.values())
                
                # 过滤：公开模板 或 租户模板
                templates = [t for t in templates if t.get('is_public') or t.get('tenant_id') == tenant_id]
                
                if workflow_type:
                    templates = [t for t in templates if t.get('workflow_type') == workflow_type]
                if category:
                    templates = [t for t in templates if t.get('category') == category]
                
                templates.sort(key=lambda x: (not x.get('is_system', False), x.get('name', '')))
                total = len(templates)
                return templates[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                from sqlalchemy import or_
                
                query = db.query(WorkflowTemplate).filter(
                    or_(
                        WorkflowTemplate.is_public == True,
                        WorkflowTemplate.tenant_id == tenant_id
                    )
                )
                
                if workflow_type:
                    query = query.filter(WorkflowTemplate.workflow_type == workflow_type)
                if category:
                    query = query.filter(WorkflowTemplate.category == category)
                
                total = query.count()
                templates = query.order_by(
                    WorkflowTemplate.is_system.desc(),
                    WorkflowTemplate.name
                ).offset(offset).limit(limit).all()
                return templates, total
                
        except Exception as e:
            logger.error(f"Failed to get available templates: {e}")
            return [], 0
    
    def increment_use_count(self, template_id: str) -> bool:
        """增加使用次数"""
        try:
            if self._use_memory_storage:
                if template_id in self._templates:
                    self._templates[template_id]['use_count'] = self._templates[template_id].get('use_count', 0) + 1
                    return True
                return False
            
            with self._db_manager.get_db_session() as db:
                template = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == template_id).first()
                if not template:
                    return False
                
                template.use_count = (template.use_count or 0) + 1
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to increment template use count: {e}")
            return False
    
    def update(self, template_id: str, update_data: Dict[str, Any]) -> Optional[WorkflowTemplate]:
        """更新模板"""
        try:
            if self._use_memory_storage:
                if template_id in self._templates:
                    self._templates[template_id].update(update_data)
                    self._templates[template_id]['updated_at'] = datetime.utcnow().isoformat()
                    return self._templates[template_id]
                return None
            
            with self._db_manager.get_db_session() as db:
                template = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == template_id).first()
                if not template:
                    return None
                
                for key, value in update_data.items():
                    if key in ('config', 'steps_config', 'default_params') and isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    if key == 'tags' and isinstance(value, list):
                        value = json.dumps(value)
                    if hasattr(template, key):
                        setattr(template, key, value)
                
                template.updated_at = datetime.utcnow()
                db.commit()
                db.refresh(template)
                return template
                
        except Exception as e:
            logger.error(f"Failed to update template: {e}")
            raise DatabaseError(f"Failed to update template: {e}", operation="update_template")
    
    def delete(self, template_id: str) -> bool:
        """删除模板"""
        try:
            if self._use_memory_storage:
                if template_id in self._templates:
                    # 不允许删除系统模板
                    if self._templates[template_id].get('is_system'):
                        return False
                    del self._templates[template_id]
                    return True
                return False
            
            with self._db_manager.get_db_session() as db:
                template = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == template_id).first()
                if not template or template.is_system:
                    return False
                
                db.delete(template)
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete template: {e}")
            return False


class WorkflowLogRepository:
    """工作流日志数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化日志仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._logs: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database service, falling back to memory storage")
                self._use_memory_storage = True
                self._logs: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, log_data: Dict[str, Any]) -> WorkflowLog:
        """创建日志"""
        try:
            log_id = log_data.get('id') or _generate_id()
            
            if self._use_memory_storage:
                log_data['id'] = log_id
                log_data['timestamp'] = datetime.utcnow().isoformat()
                self._logs[log_id] = log_data
                return log_data
            
            with self._db_manager.get_db_session() as db:
                details = log_data.get('details', {})
                if isinstance(details, dict):
                    details = json.dumps(details)
                
                log = WorkflowLog(
                    id=log_id,
                    execution_id=log_data['execution_id'],
                    workflow_id=log_data['workflow_id'],
                    step_id=log_data.get('step_id'),
                    level=log_data.get('level', 'info'),
                    message=log_data['message'],
                    details=details,
                    source=log_data.get('source')
                )
                db.add(log)
                db.commit()
                db.refresh(log)
                return log
                
        except Exception as e:
            logger.error(f"Failed to create workflow log: {e}")
            raise DatabaseError(f"Failed to create workflow log: {e}", operation="create_workflow_log")
    
    def get_by_execution(self, execution_id: str, level: Optional[str] = None,
                        limit: int = 1000, offset: int = 0) -> Tuple[List[WorkflowLog], int]:
        """获取执行的日志"""
        try:
            if self._use_memory_storage:
                logs = [l for l in self._logs.values() if l.get('execution_id') == execution_id]
                if level:
                    logs = [l for l in logs if l.get('level') == level]
                logs.sort(key=lambda x: x.get('timestamp', ''))
                total = len(logs)
                return logs[offset:offset + limit], total
            
            with self._db_manager.get_db_session() as db:
                query = db.query(WorkflowLog).filter(WorkflowLog.execution_id == execution_id)
                if level:
                    query = query.filter(WorkflowLog.level == level)
                
                total = query.count()
                logs = query.order_by(WorkflowLog.timestamp).offset(offset).limit(limit).all()
                return logs, total
                
        except Exception as e:
            logger.error(f"Failed to get logs by execution: {e}")
            return [], 0
    
    def get_by_step(self, step_id: str) -> List[WorkflowLog]:
        """获取步骤的日志"""
        try:
            if self._use_memory_storage:
                logs = [l for l in self._logs.values() if l.get('step_id') == step_id]
                logs.sort(key=lambda x: x.get('timestamp', ''))
                return logs
            
            with self._db_manager.get_db_session() as db:
                return db.query(WorkflowLog).filter(
                    WorkflowLog.step_id == step_id
                ).order_by(WorkflowLog.timestamp).all()
                
        except Exception as e:
            logger.error(f"Failed to get logs by step: {e}")
            return []
    
    def delete_by_execution(self, execution_id: str) -> int:
        """删除执行的所有日志"""
        try:
            if self._use_memory_storage:
                to_delete = [k for k, v in self._logs.items() if v.get('execution_id') == execution_id]
                for key in to_delete:
                    del self._logs[key]
                return len(to_delete)
            
            with self._db_manager.get_db_session() as db:
                count = db.query(WorkflowLog).filter(WorkflowLog.execution_id == execution_id).delete()
                db.commit()
                return count
                
        except Exception as e:
            logger.error(f"Failed to delete logs: {e}")
            return 0


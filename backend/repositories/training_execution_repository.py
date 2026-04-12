"""训练执行仓库

提供训练执行记录的数据访问接口，支持CRUD操作和高级查询。
"""

import logging
import os
import sys
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.exceptions import ResourceNotFoundError, DatabaseError
from backend.schemas.training_models import TrainingExecution, TrainingExecutionLog
from backend.schemas.enums import TrainingStatus

logger = logging.getLogger(__name__)


class TrainingExecutionRepository:
    """训练执行仓库"""
    
    def __init__(self, db_service=None, use_memory_storage=False):
        """初始化训练执行仓库
        
        Args:
            db_service: 数据库服务实例
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._executions: Dict[str, TrainingExecution] = {}
            self._logs: Dict[str, List[TrainingExecutionLog]] = {}
            self._db_service = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_service = db_service or get_database_manager()
            except ImportError:
                logger.warning("无法导入数据库服务，回退到内存存储模式")
                self._use_memory_storage = True
                self._executions: Dict[str, TrainingExecution] = {}
                self._logs: Dict[str, List[TrainingExecutionLog]] = {}
                self._db_service = None
    
    # ==================== 基础CRUD操作 ====================
    
    def create(self, execution: Union[TrainingExecution, Dict[str, Any]]) -> TrainingExecution:
        """创建训练执行记录
        
        Args:
            execution: 执行记录对象或数据字典
            
        Returns:
            创建的执行记录
        """
        try:
            if self._use_memory_storage:
                if isinstance(execution, dict):
                    execution = TrainingExecution(**execution)
                
                execution_id = getattr(execution, 'execution_id', None)
                if not execution_id:
                    import uuid
                    execution_id = f"exec_{uuid.uuid4().hex[:16]}"
                    execution.execution_id = execution_id
                
                self._executions[execution_id] = execution
                logger.info(f"Training execution created: {execution_id}")
                return execution
            else:
                with self._db_service.get_db_session() as db_session:
                    if isinstance(execution, dict):
                        training_execution = TrainingExecution(**execution)
                    else:
                        training_execution = execution
                    
                    db_session.add(training_execution)
                    db_session.commit()
                    db_session.refresh(training_execution)
                    db_session.expunge(training_execution)
                    
                    logger.info(f"Training execution created in database: {training_execution.execution_id}")
                    return training_execution
                    
        except Exception as e:
            logger.error(f"Failed to create training execution: {e}")
            raise DatabaseError(f"创建训练执行记录失败: {str(e)}", operation="create")
    
    def get_by_execution_id(self, execution_id: str, tenant_id: str = None) -> Optional[TrainingExecution]:
        """根据执行ID获取执行记录
        
        Args:
            execution_id: 执行ID
            tenant_id: 租户ID
            
        Returns:
            执行记录
        """
        try:
            if self._use_memory_storage:
                execution = self._executions.get(execution_id)
                if execution and tenant_id and str(getattr(execution, 'tenant_id', '')) != tenant_id:
                    return None
                return execution
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingExecution).filter(
                        TrainingExecution.execution_id == execution_id
                    )
                    if tenant_id:
                        query = query.filter(TrainingExecution.tenant_id == tenant_id)
                    
                    execution = query.first()
                    if execution:
                        db_session.refresh(execution)
                        db_session.expunge(execution)
                    return execution
                    
        except Exception as e:
            logger.error(f"Failed to get training execution: {e}")
            raise DatabaseError(f"获取训练执行记录失败: {str(e)}", operation="get_by_execution_id")
    
    def get_by_session_id(self, session_id: str, tenant_id: str = None) -> Optional[TrainingExecution]:
        """根据会话ID获取最新执行记录
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            
        Returns:
            执行记录
        """
        try:
            if self._use_memory_storage:
                executions = [e for e in self._executions.values() if e.session_id == session_id]
                if tenant_id:
                    executions = [e for e in executions if str(getattr(e, 'tenant_id', '')) == tenant_id]
                if executions:
                    executions.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
                    return executions[0]
                return None
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingExecution).filter(
                        TrainingExecution.session_id == session_id
                    )
                    if tenant_id:
                        query = query.filter(TrainingExecution.tenant_id == tenant_id)
                    
                    execution = query.order_by(desc(TrainingExecution.created_at)).first()
                    if execution:
                        db_session.refresh(execution)
                        db_session.expunge(execution)
                    return execution
                    
        except Exception as e:
            logger.error(f"Failed to get training execution by session: {e}")
            raise DatabaseError(f"获取训练执行记录失败: {str(e)}", operation="get_by_session_id")
    
    def update(self, execution: TrainingExecution) -> TrainingExecution:
        """更新训练执行记录
        
        Args:
            execution: 执行记录
            
        Returns:
            更新后的执行记录
        """
        try:
            if self._use_memory_storage:
                execution_id = getattr(execution, 'execution_id', None)
                if execution_id not in self._executions:
                    raise ResourceNotFoundError(f"训练执行记录不存在: {execution_id}")
                
                self._executions[execution_id] = execution
                logger.info(f"Training execution updated: {execution_id}")
                return execution
            else:
                with self._db_service.get_db_session() as db_session:
                    existing = db_session.query(TrainingExecution).filter(
                        TrainingExecution.execution_id == execution.execution_id
                    ).first()
                    
                    if not existing:
                        raise ResourceNotFoundError(f"训练执行记录不存在: {execution.execution_id}")
                    
                    for key, value in execution.__dict__.items():
                        if not key.startswith('_') and hasattr(existing, key):
                            setattr(existing, key, value)
                    
                    existing.updated_at = datetime.utcnow()
                    db_session.commit()
                    db_session.refresh(existing)
                    db_session.expunge(existing)
                    
                    logger.info(f"Training execution updated in database: {execution.execution_id}")
                    return existing
                    
        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update training execution: {e}")
            raise DatabaseError(f"更新训练执行记录失败: {str(e)}", operation="update")
    
    def update_status(self, execution_id: str, status: TrainingStatus,
                     tenant_id: str = None, **kwargs) -> Optional[TrainingExecution]:
        """更新执行状态
        
        Args:
            execution_id: 执行ID
            status: 新状态
            tenant_id: 租户ID
            **kwargs: 其他要更新的字段
            
        Returns:
            更新后的执行记录
        """
        try:
            status_value = status.value if isinstance(status, TrainingStatus) else status
            
            if self._use_memory_storage:
                execution = self._executions.get(execution_id)
                if not execution:
                    return None
                if tenant_id and str(getattr(execution, 'tenant_id', '')) != tenant_id:
                    return None
                
                execution.status = status_value
                for key, value in kwargs.items():
                    if hasattr(execution, key):
                        setattr(execution, key, value)
                
                logger.info(f"Training execution status updated: {execution_id} -> {status_value}")
                return execution
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingExecution).filter(
                        TrainingExecution.execution_id == execution_id
                    )
                    if tenant_id:
                        query = query.filter(TrainingExecution.tenant_id == tenant_id)
                    
                    execution = query.first()
                    if not execution:
                        return None
                    
                    execution.status = status_value
                    for key, value in kwargs.items():
                        if hasattr(execution, key):
                            setattr(execution, key, value)
                    execution.updated_at = datetime.utcnow()
                    
                    db_session.commit()
                    db_session.refresh(execution)
                    db_session.expunge(execution)
                    
                    logger.info(f"Training execution status updated in database: {execution_id} -> {status_value}")
                    return execution
                    
        except Exception as e:
            logger.error(f"Failed to update training execution status: {e}")
            raise DatabaseError(f"更新训练执行状态失败: {str(e)}", operation="update_status")
    
    def update_progress(self, execution_id: str, progress: float,
                       current_step: int = None, current_epoch: int = None,
                       metrics: Dict[str, Any] = None, tenant_id: str = None) -> Optional[TrainingExecution]:
        """更新执行进度
        
        Args:
            execution_id: 执行ID
            progress: 进度值(0-100)
            current_step: 当前步骤
            current_epoch: 当前轮次
            metrics: 当前指标
            tenant_id: 租户ID
            
        Returns:
            更新后的执行记录
        """
        try:
            update_data = {'progress': min(100.0, max(0.0, progress))}
            
            if current_step is not None:
                update_data['current_step'] = current_step
            if current_epoch is not None:
                update_data['current_epoch'] = current_epoch
            if metrics is not None:
                update_data['metrics'] = metrics
            
            if self._use_memory_storage:
                execution = self._executions.get(execution_id)
                if not execution:
                    return None
                if tenant_id and str(getattr(execution, 'tenant_id', '')) != tenant_id:
                    return None
                
                for key, value in update_data.items():
                    setattr(execution, key, value)
                return execution
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingExecution).filter(
                        TrainingExecution.execution_id == execution_id
                    )
                    if tenant_id:
                        query = query.filter(TrainingExecution.tenant_id == tenant_id)
                    
                    execution = query.first()
                    if not execution:
                        return None
                    
                    for key, value in update_data.items():
                        setattr(execution, key, value)
                    execution.updated_at = datetime.utcnow()
                    
                    db_session.commit()
                    db_session.refresh(execution)
                    db_session.expunge(execution)
                    return execution
                    
        except Exception as e:
            logger.error(f"Failed to update training execution progress: {e}")
            raise DatabaseError(f"更新训练执行进度失败: {str(e)}", operation="update_progress")
    
    def delete(self, execution_id: str, tenant_id: str = None) -> bool:
        """删除训练执行记录
        
        Args:
            execution_id: 执行ID
            tenant_id: 租户ID
            
        Returns:
            是否删除成功
        """
        try:
            if self._use_memory_storage:
                execution = self._executions.get(execution_id)
                if not execution:
                    return False
                if tenant_id and str(getattr(execution, 'tenant_id', '')) != tenant_id:
                    return False
                
                del self._executions[execution_id]
                if execution_id in self._logs:
                    del self._logs[execution_id]
                
                logger.info(f"Training execution deleted: {execution_id}")
                return True
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingExecution).filter(
                        TrainingExecution.execution_id == execution_id
                    )
                    if tenant_id:
                        query = query.filter(TrainingExecution.tenant_id == tenant_id)
                    
                    execution = query.first()
                    if not execution:
                        return False
                    
                    # 删除关联日志
                    db_session.query(TrainingExecutionLog).filter(
                        TrainingExecutionLog.execution_id == execution_id
                    ).delete()
                    
                    db_session.delete(execution)
                    db_session.commit()
                    
                    logger.info(f"Training execution deleted from database: {execution_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to delete training execution: {e}")
            raise DatabaseError(f"删除训练执行记录失败: {str(e)}", operation="delete")
    
    # ==================== 列表查询 ====================
    
    def list_by_tenant(self, tenant_id: str, status: str = None,
                      scenario_type: str = None, user_id: str = None,
                      limit: int = 100, offset: int = 0) -> List[TrainingExecution]:
        """按租户列出执行记录
        
        Args:
            tenant_id: 租户ID
            status: 状态过滤
            scenario_type: 场景类型过滤
            user_id: 用户ID过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            执行记录列表
        """
        try:
            if self._use_memory_storage:
                executions = [e for e in self._executions.values()
                             if str(getattr(e, 'tenant_id', '')) == tenant_id]
                
                if status:
                    executions = [e for e in executions if e.status == status]
                if scenario_type:
                    executions = [e for e in executions if e.scenario_type == scenario_type]
                if user_id:
                    executions = [e for e in executions if e.user_id == user_id]
                
                executions.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
                return executions[offset:offset + limit]
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingExecution).filter(
                        TrainingExecution.tenant_id == tenant_id
                    )
                    
                    if status:
                        query = query.filter(TrainingExecution.status == status)
                    if scenario_type:
                        query = query.filter(TrainingExecution.scenario_type == scenario_type)
                    if user_id:
                        query = query.filter(TrainingExecution.user_id == user_id)
                    
                    executions = query.order_by(desc(TrainingExecution.created_at)).offset(offset).limit(limit).all()
                    
                    for execution in executions:
                        db_session.refresh(execution)
                        db_session.expunge(execution)
                    
                    return executions
                    
        except Exception as e:
            logger.error(f"Failed to list training executions: {e}")
            raise DatabaseError(f"列出训练执行记录失败: {str(e)}", operation="list_by_tenant")
    
    def list_by_user(self, user_id: str, tenant_id: str = None,
                    status: str = None, limit: int = 100, offset: int = 0) -> List[TrainingExecution]:
        """按用户列出执行记录
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            status: 状态过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            执行记录列表
        """
        try:
            if self._use_memory_storage:
                executions = [e for e in self._executions.values() if e.user_id == user_id]
                
                if tenant_id:
                    executions = [e for e in executions if str(getattr(e, 'tenant_id', '')) == tenant_id]
                if status:
                    executions = [e for e in executions if e.status == status]
                
                executions.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
                return executions[offset:offset + limit]
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingExecution).filter(
                        TrainingExecution.user_id == user_id
                    )
                    
                    if tenant_id:
                        query = query.filter(TrainingExecution.tenant_id == tenant_id)
                    if status:
                        query = query.filter(TrainingExecution.status == status)
                    
                    executions = query.order_by(desc(TrainingExecution.created_at)).offset(offset).limit(limit).all()
                    
                    for execution in executions:
                        db_session.refresh(execution)
                        db_session.expunge(execution)
                    
                    return executions
                    
        except Exception as e:
            logger.error(f"Failed to list training executions by user: {e}")
            raise DatabaseError(f"按用户列出执行记录失败: {str(e)}", operation="list_by_user")
    
    def list_running(self, tenant_id: str = None) -> List[TrainingExecution]:
        """列出运行中的执行记录
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            运行中的执行记录列表
        """
        return self.list_by_status(TrainingStatus.RUNNING.value, tenant_id)
    
    def list_by_status(self, status: str, tenant_id: str = None) -> List[TrainingExecution]:
        """按状态列出执行记录
        
        Args:
            status: 状态
            tenant_id: 租户ID
            
        Returns:
            执行记录列表
        """
        try:
            if self._use_memory_storage:
                executions = [e for e in self._executions.values() if e.status == status]
                if tenant_id:
                    executions = [e for e in executions if str(getattr(e, 'tenant_id', '')) == tenant_id]
                return executions
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingExecution).filter(
                        TrainingExecution.status == status
                    )
                    if tenant_id:
                        query = query.filter(TrainingExecution.tenant_id == tenant_id)
                    
                    executions = query.all()
                    for execution in executions:
                        db_session.refresh(execution)
                        db_session.expunge(execution)
                    return executions
                    
        except Exception as e:
            logger.error(f"Failed to list executions by status: {e}")
            raise DatabaseError(f"按状态列出执行记录失败: {str(e)}", operation="list_by_status")
    
    # ==================== 统计查询 ====================
    
    def get_statistics(self, tenant_id: str, user_id: str = None) -> Dict[str, Any]:
        """获取执行统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            统计信息
        """
        try:
            if self._use_memory_storage:
                executions = [e for e in self._executions.values()
                             if str(getattr(e, 'tenant_id', '')) == tenant_id]
                if user_id:
                    executions = [e for e in executions if e.user_id == user_id]
                
                return {
                    'total_executions': len(executions),
                    'pending_executions': len([e for e in executions if e.status == TrainingStatus.PENDING.value]),
                    'running_executions': len([e for e in executions if e.status == TrainingStatus.RUNNING.value]),
                    'completed_executions': len([e for e in executions if e.status == TrainingStatus.COMPLETED.value]),
                    'failed_executions': len([e for e in executions if e.status == TrainingStatus.FAILED.value]),
                    'paused_executions': len([e for e in executions if e.status == TrainingStatus.PAUSED.value]),
                    'cancelled_executions': len([e for e in executions if e.status == TrainingStatus.CANCELLED.value])
                }
            else:
                with self._db_service.get_db_session() as db_session:
                    base_query = db_session.query(TrainingExecution).filter(
                        TrainingExecution.tenant_id == tenant_id
                    )
                    if user_id:
                        base_query = base_query.filter(TrainingExecution.user_id == user_id)
                    
                    return {
                        'total_executions': base_query.count(),
                        'pending_executions': base_query.filter(TrainingExecution.status == TrainingStatus.PENDING.value).count(),
                        'running_executions': base_query.filter(TrainingExecution.status == TrainingStatus.RUNNING.value).count(),
                        'completed_executions': base_query.filter(TrainingExecution.status == TrainingStatus.COMPLETED.value).count(),
                        'failed_executions': base_query.filter(TrainingExecution.status == TrainingStatus.FAILED.value).count(),
                        'paused_executions': base_query.filter(TrainingExecution.status == TrainingStatus.PAUSED.value).count(),
                        'cancelled_executions': base_query.filter(TrainingExecution.status == TrainingStatus.CANCELLED.value).count()
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get execution statistics: {e}")
            return {}
    
    # ==================== 日志操作 ====================
    
    def create_log(self, log: Union[TrainingExecutionLog, Dict[str, Any]]) -> TrainingExecutionLog:
        """创建执行日志
        
        Args:
            log: 日志对象或数据字典
            
        Returns:
            创建的日志对象
        """
        try:
            if self._use_memory_storage:
                if isinstance(log, dict):
                    log = TrainingExecutionLog(**log)
                
                execution_id = log.execution_id
                if execution_id not in self._logs:
                    self._logs[execution_id] = []
                self._logs[execution_id].append(log)
                return log
            else:
                with self._db_service.get_db_session() as db_session:
                    if isinstance(log, dict):
                        execution_log = TrainingExecutionLog(**log)
                    else:
                        execution_log = log
                    
                    db_session.add(execution_log)
                    db_session.commit()
                    db_session.refresh(execution_log)
                    db_session.expunge(execution_log)
                    return execution_log
                    
        except Exception as e:
            logger.error(f"Failed to create execution log: {e}")
            raise DatabaseError(f"创建执行日志失败: {str(e)}", operation="create_log")
    
    def list_logs(self, execution_id: str, log_type: str = None,
                 limit: int = 100, offset: int = 0) -> List[TrainingExecutionLog]:
        """列出执行日志
        
        Args:
            execution_id: 执行ID
            log_type: 日志类型过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            日志列表
        """
        try:
            if self._use_memory_storage:
                logs = self._logs.get(execution_id, [])
                if log_type:
                    logs = [l for l in logs if l.log_type == log_type]
                logs.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
                return logs[offset:offset + limit]
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingExecutionLog).filter(
                        TrainingExecutionLog.execution_id == execution_id
                    )
                    if log_type:
                        query = query.filter(TrainingExecutionLog.log_type == log_type)
                    
                    logs = query.order_by(desc(TrainingExecutionLog.created_at)).offset(offset).limit(limit).all()
                    
                    for log in logs:
                        db_session.refresh(log)
                        db_session.expunge(log)
                    
                    return logs
                    
        except Exception as e:
            logger.error(f"Failed to list execution logs: {e}")
            raise DatabaseError(f"列出执行日志失败: {str(e)}", operation="list_logs")


# 全局实例
_training_execution_repository = None


def get_training_execution_repository(use_memory_storage: bool = False) -> TrainingExecutionRepository:
    """获取训练执行仓库实例
    
    Args:
        use_memory_storage: 是否使用内存存储
        
    Returns:
        训练执行仓库实例
    """
    global _training_execution_repository
    if _training_execution_repository is None:
        _training_execution_repository = TrainingExecutionRepository(use_memory_storage=use_memory_storage)
    return _training_execution_repository


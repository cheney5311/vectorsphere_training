"""训练任务仓库

提供训练任务数据访问接口，支持CRUD操作和高级查询。
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
from backend.schemas.training_models import TrainingJob, TrainingJobLog
from backend.schemas.enums import TrainingStatus

logger = logging.getLogger(__name__)


class TrainingJobRepository:
    """训练任务仓库"""
    
    def __init__(self, db_service=None, use_memory_storage=False):
        """初始化训练任务仓库
        
        Args:
            db_service: 数据库服务实例
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._jobs: Dict[str, TrainingJob] = {}
            self._logs: Dict[str, List[TrainingJobLog]] = {}
            self._db_service = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_service = db_service or get_database_manager()
            except ImportError:
                logger.warning("无法导入数据库服务，回退到内存存储模式")
                self._use_memory_storage = True
                self._jobs: Dict[str, TrainingJob] = {}
                self._logs: Dict[str, List[TrainingJobLog]] = {}
                self._db_service = None
    
    # ==================== 基础CRUD操作 ====================
    
    def create(self, job: Union[TrainingJob, Dict[str, Any]]) -> TrainingJob:
        """创建训练任务
        
        Args:
            job: 训练任务对象或数据字典
            
        Returns:
            创建的训练任务对象
        """
        try:
            if self._use_memory_storage:
                if isinstance(job, dict):
                    job = TrainingJob(**job)
                
                job_id = getattr(job, 'job_id', None)
                if not job_id:
                    import uuid
                    job_id = str(uuid.uuid4())
                    job.job_id = job_id
                
                self._jobs[job_id] = job
                logger.info(f"Training job created: {job_id}")
                return job
            else:
                with self._db_service.get_db_session() as db_session:
                    if isinstance(job, dict):
                        training_job = TrainingJob(**job)
                    else:
                        training_job = job
                    
                    db_session.add(training_job)
                    db_session.commit()
                    db_session.refresh(training_job)
                    db_session.expunge(training_job)
                    
                    logger.info(f"Training job created in database: {training_job.job_id}")
                    return training_job
                    
        except Exception as e:
            logger.error(f"Failed to create training job: {e}")
            raise DatabaseError(f"创建训练任务失败: {str(e)}", operation="create")
    
    def get_by_job_id(self, job_id: str, tenant_id: str = None) -> Optional[TrainingJob]:
        """根据任务ID获取训练任务
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID（可选，用于权限验证）
            
        Returns:
            训练任务对象
        """
        try:
            if self._use_memory_storage:
                job = self._jobs.get(job_id)
                if job and tenant_id and getattr(job, 'tenant_id', None) != tenant_id:
                    return None
                return job
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingJob).filter(
                        TrainingJob.job_id == job_id
                    )
                    if tenant_id:
                        query = query.filter(TrainingJob.tenant_id == tenant_id)
                    
                    job = query.first()
                    if job:
                        db_session.refresh(job)
                        db_session.expunge(job)
                    return job
                    
        except Exception as e:
            logger.error(f"Failed to get training job: {e}")
            raise DatabaseError(f"获取训练任务失败: {str(e)}", operation="get_by_job_id")
    
    def get_by_id(self, id: str, tenant_id: str = None) -> Optional[TrainingJob]:
        """根据主键ID获取训练任务
        
        Args:
            id: 主键ID
            tenant_id: 租户ID
            
        Returns:
            训练任务对象
        """
        try:
            if self._use_memory_storage:
                for job in self._jobs.values():
                    if str(getattr(job, 'id', '')) == id:
                        if tenant_id and getattr(job, 'tenant_id', None) != tenant_id:
                            return None
                        return job
                return None
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingJob).filter(TrainingJob.id == id)
                    if tenant_id:
                        query = query.filter(TrainingJob.tenant_id == tenant_id)
                    
                    job = query.first()
                    if job:
                        db_session.refresh(job)
                        db_session.expunge(job)
                    return job
                    
        except Exception as e:
            logger.error(f"Failed to get training job by id: {e}")
            raise DatabaseError(f"获取训练任务失败: {str(e)}", operation="get_by_id")
    
    def update(self, job: TrainingJob) -> TrainingJob:
        """更新训练任务
        
        Args:
            job: 训练任务对象
            
        Returns:
            更新后的训练任务对象
        """
        try:
            if self._use_memory_storage:
                job_id = getattr(job, 'job_id', None)
                if job_id not in self._jobs:
                    raise ResourceNotFoundError(f"训练任务不存在: {job_id}")
                
                self._jobs[job_id] = job
                logger.info(f"Training job updated: {job_id}")
                return job
            else:
                with self._db_service.get_db_session() as db_session:
                    existing_job = db_session.query(TrainingJob).filter(
                        TrainingJob.job_id == job.job_id
                    ).first()
                    
                    if not existing_job:
                        raise ResourceNotFoundError(f"训练任务不存在: {job.job_id}")
                    
                    for key, value in job.__dict__.items():
                        if not key.startswith('_') and hasattr(existing_job, key):
                            setattr(existing_job, key, value)
                    
                    existing_job.updated_at = datetime.utcnow()
                    db_session.commit()
                    db_session.refresh(existing_job)
                    db_session.expunge(existing_job)
                    
                    logger.info(f"Training job updated in database: {job.job_id}")
                    return existing_job
                    
        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update training job: {e}")
            raise DatabaseError(f"更新训练任务失败: {str(e)}", operation="update")
    
    def update_status(self, job_id: str, status: TrainingStatus, 
                     tenant_id: str = None, **kwargs) -> Optional[TrainingJob]:
        """更新训练任务状态
        
        Args:
            job_id: 任务ID
            status: 新状态
            tenant_id: 租户ID
            **kwargs: 其他要更新的字段
            
        Returns:
            更新后的训练任务对象
        """
        try:
            if self._use_memory_storage:
                job = self._jobs.get(job_id)
                if not job:
                    return None
                if tenant_id and getattr(job, 'tenant_id', None) != tenant_id:
                    return None
                
                job.status = status.value if isinstance(status, TrainingStatus) else status
                for key, value in kwargs.items():
                    if hasattr(job, key):
                        setattr(job, key, value)
                
                logger.info(f"Training job status updated: {job_id} -> {status}")
                return job
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingJob).filter(
                        TrainingJob.job_id == job_id
                    )
                    if tenant_id:
                        query = query.filter(TrainingJob.tenant_id == tenant_id)
                    
                    job = query.first()
                    if not job:
                        return None
                    
                    job.status = status.value if isinstance(status, TrainingStatus) else status
                    for key, value in kwargs.items():
                        if hasattr(job, key):
                            setattr(job, key, value)
                    job.updated_at = datetime.utcnow()
                    
                    db_session.commit()
                    db_session.refresh(job)
                    db_session.expunge(job)
                    
                    logger.info(f"Training job status updated in database: {job_id} -> {status}")
                    return job
                    
        except Exception as e:
            logger.error(f"Failed to update training job status: {e}")
            raise DatabaseError(f"更新训练任务状态失败: {str(e)}", operation="update_status")
    
    def update_progress(self, job_id: str, progress: float, 
                       current_step: int = None, current_epoch: int = None,
                       metrics: Dict[str, Any] = None, tenant_id: str = None) -> Optional[TrainingJob]:
        """更新训练任务进度
        
        Args:
            job_id: 任务ID
            progress: 进度值(0-100)
            current_step: 当前步骤
            current_epoch: 当前轮次
            metrics: 当前指标
            tenant_id: 租户ID
            
        Returns:
            更新后的训练任务对象
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
                job = self._jobs.get(job_id)
                if not job:
                    return None
                if tenant_id and getattr(job, 'tenant_id', None) != tenant_id:
                    return None
                
                for key, value in update_data.items():
                    setattr(job, key, value)
                return job
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingJob).filter(
                        TrainingJob.job_id == job_id
                    )
                    if tenant_id:
                        query = query.filter(TrainingJob.tenant_id == tenant_id)
                    
                    job = query.first()
                    if not job:
                        return None
                    
                    for key, value in update_data.items():
                        setattr(job, key, value)
                    job.updated_at = datetime.utcnow()
                    
                    db_session.commit()
                    db_session.refresh(job)
                    db_session.expunge(job)
                    return job
                    
        except Exception as e:
            logger.error(f"Failed to update training job progress: {e}")
            raise DatabaseError(f"更新训练任务进度失败: {str(e)}", operation="update_progress")
    
    def delete(self, job_id: str, tenant_id: str = None) -> bool:
        """删除训练任务
        
        Args:
            job_id: 任务ID
            tenant_id: 租户ID
            
        Returns:
            是否删除成功
        """
        try:
            if self._use_memory_storage:
                job = self._jobs.get(job_id)
                if not job:
                    return False
                if tenant_id and getattr(job, 'tenant_id', None) != tenant_id:
                    return False
                
                del self._jobs[job_id]
                if job_id in self._logs:
                    del self._logs[job_id]
                
                logger.info(f"Training job deleted: {job_id}")
                return True
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingJob).filter(
                        TrainingJob.job_id == job_id
                    )
                    if tenant_id:
                        query = query.filter(TrainingJob.tenant_id == tenant_id)
                    
                    job = query.first()
                    if not job:
                        return False
                    
                    # 删除关联日志
                    db_session.query(TrainingJobLog).filter(
                        TrainingJobLog.job_id == job_id
                    ).delete()
                    
                    db_session.delete(job)
                    db_session.commit()
                    
                    logger.info(f"Training job deleted from database: {job_id}")
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to delete training job: {e}")
            raise DatabaseError(f"删除训练任务失败: {str(e)}", operation="delete")
    
    # ==================== 列表查询 ====================
    
    def list_by_tenant(self, tenant_id: str, status: str = None,
                      user_id: str = None, scenario_type: str = None,
                      limit: int = 100, offset: int = 0,
                      order_by: str = 'created_at', order_desc: bool = True) -> List[TrainingJob]:
        """按租户列出训练任务
        
        Args:
            tenant_id: 租户ID
            status: 状态过滤
            user_id: 用户ID过滤
            scenario_type: 场景类型过滤
            limit: 限制数量
            offset: 偏移量
            order_by: 排序字段
            order_desc: 是否降序
            
        Returns:
            训练任务列表
        """
        try:
            if self._use_memory_storage:
                jobs = [j for j in self._jobs.values() 
                       if getattr(j, 'tenant_id', None) == tenant_id]
                
                if status:
                    jobs = [j for j in jobs if j.status == status]
                if user_id:
                    jobs = [j for j in jobs if j.user_id == user_id]
                if scenario_type:
                    jobs = [j for j in jobs if j.scenario_type == scenario_type]
                
                # 排序
                reverse = order_desc
                jobs.sort(key=lambda x: getattr(x, order_by, datetime.min) or datetime.min, 
                         reverse=reverse)
                
                return jobs[offset:offset + limit]
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingJob).filter(
                        TrainingJob.tenant_id == tenant_id
                    )
                    
                    if status:
                        query = query.filter(TrainingJob.status == status)
                    if user_id:
                        query = query.filter(TrainingJob.user_id == user_id)
                    if scenario_type:
                        query = query.filter(TrainingJob.scenario_type == scenario_type)
                    
                    # 排序
                    order_column = getattr(TrainingJob, order_by, TrainingJob.created_at)
                    if order_desc:
                        query = query.order_by(desc(order_column))
                    else:
                        query = query.order_by(asc(order_column))
                    
                    jobs = query.offset(offset).limit(limit).all()
                    
                    for job in jobs:
                        db_session.refresh(job)
                        db_session.expunge(job)
                    
                    return jobs
                    
        except Exception as e:
            logger.error(f"Failed to list training jobs: {e}")
            raise DatabaseError(f"列出训练任务失败: {str(e)}", operation="list_by_tenant")
    
    def list_by_user(self, user_id: str, tenant_id: str = None,
                    status: str = None, limit: int = 100, offset: int = 0) -> List[TrainingJob]:
        """按用户列出训练任务
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            status: 状态过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            训练任务列表
        """
        try:
            if self._use_memory_storage:
                jobs = [j for j in self._jobs.values() if j.user_id == user_id]
                
                if tenant_id:
                    jobs = [j for j in jobs if getattr(j, 'tenant_id', None) == tenant_id]
                if status:
                    jobs = [j for j in jobs if j.status == status]
                
                jobs.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
                return jobs[offset:offset + limit]
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingJob).filter(
                        TrainingJob.user_id == user_id
                    )
                    
                    if tenant_id:
                        query = query.filter(TrainingJob.tenant_id == tenant_id)
                    if status:
                        query = query.filter(TrainingJob.status == status)
                    
                    jobs = query.order_by(desc(TrainingJob.created_at)).offset(offset).limit(limit).all()
                    
                    for job in jobs:
                        db_session.refresh(job)
                        db_session.expunge(job)
                    
                    return jobs
                    
        except Exception as e:
            logger.error(f"Failed to list training jobs by user: {e}")
            raise DatabaseError(f"按用户列出训练任务失败: {str(e)}", operation="list_by_user")
    
    def list_running(self, tenant_id: str = None) -> List[TrainingJob]:
        """列出运行中的训练任务
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            运行中的训练任务列表
        """
        return self.list_by_status(TrainingStatus.RUNNING.value, tenant_id)
    
    def list_by_status(self, status: str, tenant_id: str = None) -> List[TrainingJob]:
        """按状态列出训练任务
        
        Args:
            status: 状态
            tenant_id: 租户ID
            
        Returns:
            训练任务列表
        """
        try:
            if self._use_memory_storage:
                jobs = [j for j in self._jobs.values() if j.status == status]
                if tenant_id:
                    jobs = [j for j in jobs if getattr(j, 'tenant_id', None) == tenant_id]
                return jobs
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingJob).filter(
                        TrainingJob.status == status
                    )
                    if tenant_id:
                        query = query.filter(TrainingJob.tenant_id == tenant_id)
                    
                    jobs = query.all()
                    for job in jobs:
                        db_session.refresh(job)
                        db_session.expunge(job)
                    return jobs
                    
        except Exception as e:
            logger.error(f"Failed to list training jobs by status: {e}")
            raise DatabaseError(f"按状态列出训练任务失败: {str(e)}", operation="list_by_status")
    
    # ==================== 统计查询 ====================
    
    def count_by_tenant(self, tenant_id: str, status: str = None) -> int:
        """统计租户的训练任务数量
        
        Args:
            tenant_id: 租户ID
            status: 状态过滤
            
        Returns:
            任务数量
        """
        try:
            if self._use_memory_storage:
                jobs = [j for j in self._jobs.values() 
                       if getattr(j, 'tenant_id', None) == tenant_id]
                if status:
                    jobs = [j for j in jobs if j.status == status]
                return len(jobs)
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(func.count(TrainingJob.id)).filter(
                        TrainingJob.tenant_id == tenant_id
                    )
                    if status:
                        query = query.filter(TrainingJob.status == status)
                    return query.scalar() or 0
                    
        except Exception as e:
            logger.error(f"Failed to count training jobs: {e}")
            return 0
    
    def get_statistics(self, tenant_id: str, user_id: str = None) -> Dict[str, Any]:
        """获取训练任务统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID
            
        Returns:
            统计信息
        """
        try:
            if self._use_memory_storage:
                jobs = [j for j in self._jobs.values() 
                       if getattr(j, 'tenant_id', None) == tenant_id]
                if user_id:
                    jobs = [j for j in jobs if j.user_id == user_id]
                
                stats = {
                    'total_jobs': len(jobs),
                    'pending_jobs': len([j for j in jobs if j.status == TrainingStatus.PENDING.value]),
                    'running_jobs': len([j for j in jobs if j.status == TrainingStatus.RUNNING.value]),
                    'completed_jobs': len([j for j in jobs if j.status == TrainingStatus.COMPLETED.value]),
                    'failed_jobs': len([j for j in jobs if j.status == TrainingStatus.FAILED.value]),
                    'paused_jobs': len([j for j in jobs if j.status == TrainingStatus.PAUSED.value]),
                    'cancelled_jobs': len([j for j in jobs if j.status == TrainingStatus.CANCELLED.value])
                }
                return stats
            else:
                with self._db_service.get_db_session() as db_session:
                    base_query = db_session.query(TrainingJob).filter(
                        TrainingJob.tenant_id == tenant_id
                    )
                    if user_id:
                        base_query = base_query.filter(TrainingJob.user_id == user_id)
                    
                    stats = {
                        'total_jobs': base_query.count(),
                        'pending_jobs': base_query.filter(TrainingJob.status == TrainingStatus.PENDING.value).count(),
                        'running_jobs': base_query.filter(TrainingJob.status == TrainingStatus.RUNNING.value).count(),
                        'completed_jobs': base_query.filter(TrainingJob.status == TrainingStatus.COMPLETED.value).count(),
                        'failed_jobs': base_query.filter(TrainingJob.status == TrainingStatus.FAILED.value).count(),
                        'paused_jobs': base_query.filter(TrainingJob.status == TrainingStatus.PAUSED.value).count(),
                        'cancelled_jobs': base_query.filter(TrainingJob.status == TrainingStatus.CANCELLED.value).count()
                    }
                    return stats
                    
        except Exception as e:
            logger.error(f"Failed to get training job statistics: {e}")
            return {}
    
    # ==================== 日志操作 ====================
    
    def create_log(self, log: Union[TrainingJobLog, Dict[str, Any]]) -> TrainingJobLog:
        """创建训练任务日志
        
        Args:
            log: 日志对象或数据字典
            
        Returns:
            创建的日志对象
        """
        try:
            if self._use_memory_storage:
                if isinstance(log, dict):
                    log = TrainingJobLog(**log)
                
                job_id = log.job_id
                if job_id not in self._logs:
                    self._logs[job_id] = []
                self._logs[job_id].append(log)
                return log
            else:
                with self._db_service.get_db_session() as db_session:
                    if isinstance(log, dict):
                        job_log = TrainingJobLog(**log)
                    else:
                        job_log = log
                    
                    db_session.add(job_log)
                    db_session.commit()
                    db_session.refresh(job_log)
                    db_session.expunge(job_log)
                    return job_log
                    
        except Exception as e:
            logger.error(f"Failed to create training job log: {e}")
            raise DatabaseError(f"创建训练任务日志失败: {str(e)}", operation="create_log")
    
    def list_logs(self, job_id: str, log_type: str = None,
                 limit: int = 100, offset: int = 0) -> List[TrainingJobLog]:
        """列出训练任务日志
        
        Args:
            job_id: 任务ID
            log_type: 日志类型过滤
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            日志列表
        """
        try:
            if self._use_memory_storage:
                logs = self._logs.get(job_id, [])
                if log_type:
                    logs = [l for l in logs if l.log_type == log_type]
                logs.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
                return logs[offset:offset + limit]
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingJobLog).filter(
                        TrainingJobLog.job_id == job_id
                    )
                    if log_type:
                        query = query.filter(TrainingJobLog.log_type == log_type)
                    
                    logs = query.order_by(desc(TrainingJobLog.created_at)).offset(offset).limit(limit).all()
                    
                    for log in logs:
                        db_session.refresh(log)
                        db_session.expunge(log)
                    
                    return logs
                    
        except Exception as e:
            logger.error(f"Failed to list training job logs: {e}")
            raise DatabaseError(f"列出训练任务日志失败: {str(e)}", operation="list_logs")


class TrainingJobLogRepository:
    """训练任务日志仓库（独立仓库类）"""
    
    def __init__(self, db_service=None, use_memory_storage=False):
        """初始化日志仓库"""
        self._use_memory_storage = use_memory_storage
        self._logs: Dict[str, List[TrainingJobLog]] = {}
        
        if not use_memory_storage:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_service = db_service or get_database_manager()
            except ImportError:
                logger.warning("无法导入数据库服务，回退到内存存储模式")
                self._use_memory_storage = True
                self._db_service = None
    
    def create(self, log: Union[TrainingJobLog, Dict[str, Any]]) -> TrainingJobLog:
        """创建日志"""
        try:
            if self._use_memory_storage:
                if isinstance(log, dict):
                    log = TrainingJobLog(**log)
                
                job_id = log.job_id
                if job_id not in self._logs:
                    self._logs[job_id] = []
                self._logs[job_id].append(log)
                return log
            else:
                with self._db_service.get_db_session() as db_session:
                    if isinstance(log, dict):
                        job_log = TrainingJobLog(**log)
                    else:
                        job_log = log
                    
                    db_session.add(job_log)
                    db_session.commit()
                    db_session.refresh(job_log)
                    db_session.expunge(job_log)
                    return job_log
                    
        except Exception as e:
            logger.error(f"Failed to create log: {e}")
            raise DatabaseError(f"创建日志失败: {str(e)}", operation="create")
    
    def list_by_job(self, job_id: str, log_type: str = None,
                   limit: int = 100, offset: int = 0) -> List[TrainingJobLog]:
        """按任务列出日志"""
        try:
            if self._use_memory_storage:
                logs = self._logs.get(job_id, [])
                if log_type:
                    logs = [l for l in logs if l.log_type == log_type]
                logs.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
                return logs[offset:offset + limit]
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingJobLog).filter(
                        TrainingJobLog.job_id == job_id
                    )
                    if log_type:
                        query = query.filter(TrainingJobLog.log_type == log_type)
                    
                    logs = query.order_by(desc(TrainingJobLog.created_at)).offset(offset).limit(limit).all()
                    
                    for log in logs:
                        db_session.refresh(log)
                        db_session.expunge(log)
                    
                    return logs
                    
        except Exception as e:
            logger.error(f"Failed to list logs: {e}")
            raise DatabaseError(f"列出日志失败: {str(e)}", operation="list_by_job")


# 全局实例
_training_job_repository = None
_training_job_log_repository = None


def get_training_job_repository(use_memory_storage: bool = False) -> TrainingJobRepository:
    """获取训练任务仓库实例
    
    Args:
        use_memory_storage: 是否使用内存存储
        
    Returns:
        训练任务仓库实例
    """
    global _training_job_repository
    if _training_job_repository is None:
        _training_job_repository = TrainingJobRepository(use_memory_storage=use_memory_storage)
    return _training_job_repository


def get_training_job_log_repository(use_memory_storage: bool = False) -> TrainingJobLogRepository:
    """获取训练任务日志仓库实例
    
    Args:
        use_memory_storage: 是否使用内存存储
        
    Returns:
        训练任务日志仓库实例
    """
    global _training_job_log_repository
    if _training_job_log_repository is None:
        _training_job_log_repository = TrainingJobLogRepository(use_memory_storage=use_memory_storage)
    return _training_job_log_repository


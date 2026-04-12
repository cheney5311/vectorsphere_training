"""训练流水线数据访问层

提供训练流水线相关的数据库访问功能，包括：
- 流水线定义管理
- 流水线执行记录
- 步骤执行记录
- 流水线模板管理
"""

import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


def _generate_id() -> str:
    """生成唯一ID"""
    return str(uuid.uuid4())


# ==============================================================================
# 训练流水线仓库
# ==============================================================================

class TrainingPipelineRepository:
    """训练流水线数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化流水线仓库
        
        Args:
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._pipelines: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database manager, falling back to memory storage")
                self._use_memory_storage = True
                self._pipelines: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建流水线
        
        Args:
            pipeline_data: 流水线数据
            
        Returns:
            创建的流水线记录
        """
        try:
            record_id = pipeline_data.get('id') or _generate_id()
            pipeline_id = pipeline_data.get('pipeline_id') or f"pipe_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                pipeline_data['id'] = record_id
                pipeline_data['pipeline_id'] = pipeline_id
                pipeline_data['created_at'] = datetime.utcnow().isoformat()
                pipeline_data['updated_at'] = datetime.utcnow().isoformat()
                pipeline_data.setdefault('status', 'draft')
                pipeline_data.setdefault('version', 1)
                self._pipelines[record_id] = pipeline_data
                return pipeline_data
            
            from backend.schemas.training_models import TrainingPipeline
            
            with self._db_manager.get_db_session() as db:
                pipeline = TrainingPipeline(
                    id=record_id,
                    tenant_id=pipeline_data.get('tenant_id'),
                    pipeline_id=pipeline_id,
                    name=pipeline_data.get('name'),
                    description=pipeline_data.get('description'),
                    user_id=pipeline_data.get('user_id'),
                    status=pipeline_data.get('status', 'draft'),
                    model_name=pipeline_data.get('model_name'),
                    model_id=pipeline_data.get('model_id'),
                    dataset_id=pipeline_data.get('dataset_id'),
                    steps_config=pipeline_data.get('steps_config', []),
                    global_config=pipeline_data.get('global_config'),
                    enable_rollback=pipeline_data.get('enable_rollback', True),
                    version=pipeline_data.get('version', 1),
                    parent_pipeline_id=pipeline_data.get('parent_pipeline_id'),
                    tags=pipeline_data.get('tags'),
                    metadata_=pipeline_data.get('metadata')
                )
                
                db.add(pipeline)
                db.commit()
                db.refresh(pipeline)
                
                return pipeline.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            raise
    
    def get_by_id(self, pipeline_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """通过流水线ID获取
        
        Args:
            pipeline_id: 流水线ID
            tenant_id: 租户ID
            
        Returns:
            流水线记录
        """
        try:
            if self._use_memory_storage:
                for record in self._pipelines.values():
                    if record.get('pipeline_id') == pipeline_id and record.get('tenant_id') == tenant_id:
                        return record
                return None
            
            from backend.schemas.training_models import TrainingPipeline
            
            with self._db_manager.get_db_session() as db:
                pipeline = db.query(TrainingPipeline).filter(
                    TrainingPipeline.pipeline_id == pipeline_id,
                    TrainingPipeline.tenant_id == tenant_id
                ).first()
                
                return pipeline.to_dict() if pipeline else None
                
        except Exception as e:
            logger.error(f"Failed to get pipeline by ID: {e}")
            return None
    
    def get_by_name(self, name: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """通过名称获取流水线
        
        Args:
            name: 流水线名称
            tenant_id: 租户ID
            
        Returns:
            流水线记录
        """
        try:
            if self._use_memory_storage:
                for record in self._pipelines.values():
                    if record.get('name') == name and record.get('tenant_id') == tenant_id:
                        return record
                return None
            
            from backend.schemas.training_models import TrainingPipeline
            
            with self._db_manager.get_db_session() as db:
                pipeline = db.query(TrainingPipeline).filter(
                    TrainingPipeline.name == name,
                    TrainingPipeline.tenant_id == tenant_id
                ).first()
                
                return pipeline.to_dict() if pipeline else None
                
        except Exception as e:
            logger.error(f"Failed to get pipeline by name: {e}")
            return None
    
    def update(self, pipeline_id: str, tenant_id: str, 
              updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新流水线
        
        Args:
            pipeline_id: 流水线ID
            tenant_id: 租户ID
            updates: 更新内容
            
        Returns:
            更新后的流水线记录
        """
        try:
            if self._use_memory_storage:
                for record_id, record in self._pipelines.items():
                    if record.get('pipeline_id') == pipeline_id and record.get('tenant_id') == tenant_id:
                        record.update(updates)
                        record['updated_at'] = datetime.utcnow().isoformat()
                        return record
                return None
            
            from backend.schemas.training_models import TrainingPipeline
            
            with self._db_manager.get_db_session() as db:
                pipeline = db.query(TrainingPipeline).filter(
                    TrainingPipeline.pipeline_id == pipeline_id,
                    TrainingPipeline.tenant_id == tenant_id
                ).first()
                
                if not pipeline:
                    return None
                
                for key, value in updates.items():
                    if hasattr(pipeline, key):
                        setattr(pipeline, key, value)
                
                db.commit()
                db.refresh(pipeline)
                
                return pipeline.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to update pipeline: {e}")
            return None
    
    def update_status(self, pipeline_id: str, tenant_id: str, 
                     status: str) -> Optional[Dict[str, Any]]:
        """更新流水线状态
        
        Args:
            pipeline_id: 流水线ID
            tenant_id: 租户ID
            status: 新状态
            
        Returns:
            更新后的流水线记录
        """
        return self.update(pipeline_id, tenant_id, {'status': status})
    
    def delete(self, pipeline_id: str, tenant_id: str) -> bool:
        """删除流水线
        
        Args:
            pipeline_id: 流水线ID
            tenant_id: 租户ID
            
        Returns:
            是否删除成功
        """
        try:
            if self._use_memory_storage:
                for record_id, record in list(self._pipelines.items()):
                    if record.get('pipeline_id') == pipeline_id and record.get('tenant_id') == tenant_id:
                        del self._pipelines[record_id]
                        return True
                return False
            
            from backend.schemas.training_models import TrainingPipeline
            
            with self._db_manager.get_db_session() as db:
                deleted = db.query(TrainingPipeline).filter(
                    TrainingPipeline.pipeline_id == pipeline_id,
                    TrainingPipeline.tenant_id == tenant_id
                ).delete(synchronize_session=False)
                
                db.commit()
                return deleted > 0
                
        except Exception as e:
            logger.error(f"Failed to delete pipeline: {e}")
            return False
    
    def list_by_tenant(self, tenant_id: str, status: Optional[str] = None,
                      user_id: Optional[str] = None,
                      model_id: Optional[str] = None,
                      limit: int = 50, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """获取租户的流水线列表
        
        Args:
            tenant_id: 租户ID
            status: 状态过滤
            user_id: 用户ID过滤
            model_id: 模型ID过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            (流水线列表, 总数)
        """
        try:
            if self._use_memory_storage:
                results = []
                for record in self._pipelines.values():
                    if record.get('tenant_id') != tenant_id:
                        continue
                    if status and record.get('status') != status:
                        continue
                    if user_id and record.get('user_id') != user_id:
                        continue
                    if model_id and record.get('model_id') != model_id:
                        continue
                    results.append(record)
                
                results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(results)
                return results[offset:offset + limit], total
            
            from backend.schemas.training_models import TrainingPipeline
            
            with self._db_manager.get_db_session() as db:
                query = db.query(TrainingPipeline).filter(
                    TrainingPipeline.tenant_id == tenant_id
                )
                
                if status:
                    query = query.filter(TrainingPipeline.status == status)
                if user_id:
                    query = query.filter(TrainingPipeline.user_id == user_id)
                if model_id:
                    query = query.filter(TrainingPipeline.model_id == model_id)
                
                total = query.count()
                pipelines = query.order_by(TrainingPipeline.created_at.desc()).offset(offset).limit(limit).all()
                
                return [p.to_dict() for p in pipelines], total
                
        except Exception as e:
            logger.error(f"Failed to list pipelines: {e}")
            return [], 0
    
    def get_statistics(self, tenant_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取流水线统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID（可选）
            
        Returns:
            统计信息
        """
        try:
            if self._use_memory_storage:
                stats = {
                    'total': 0,
                    'by_status': {},
                    'recent_count': 0
                }
                
                cutoff_time = (datetime.utcnow() - timedelta(days=7)).isoformat()
                
                for record in self._pipelines.values():
                    if record.get('tenant_id') != tenant_id:
                        continue
                    if user_id and record.get('user_id') != user_id:
                        continue
                    
                    stats['total'] += 1
                    
                    status = record.get('status', 'unknown')
                    stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
                    
                    if record.get('created_at', '') >= cutoff_time:
                        stats['recent_count'] += 1
                
                return stats
            
            from backend.schemas.training_models import TrainingPipeline
            from sqlalchemy import func
            
            with self._db_manager.get_db_session() as db:
                query = db.query(TrainingPipeline).filter(
                    TrainingPipeline.tenant_id == tenant_id
                )
                
                if user_id:
                    query = query.filter(TrainingPipeline.user_id == user_id)
                
                total = query.count()
                
                # 按状态统计
                status_stats = db.query(
                    TrainingPipeline.status,
                    func.count(TrainingPipeline.id)
                ).filter(TrainingPipeline.tenant_id == tenant_id)
                
                if user_id:
                    status_stats = status_stats.filter(TrainingPipeline.user_id == user_id)
                
                status_stats = status_stats.group_by(TrainingPipeline.status).all()
                
                return {
                    'total': total,
                    'by_status': {s: c for s, c in status_stats}
                }
                
        except Exception as e:
            logger.error(f"Failed to get pipeline statistics: {e}")
            return {'total': 0, 'by_status': {}}


# ==============================================================================
# 流水线执行仓库
# ==============================================================================

class PipelineExecutionRepository:
    """流水线执行记录数据访问层"""
    
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
                logger.warning("Cannot import database manager, falling back to memory storage")
                self._use_memory_storage = True
                self._executions: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建执行记录"""
        try:
            record_id = execution_data.get('id') or _generate_id()
            execution_id = execution_data.get('execution_id') or f"exec_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{_generate_id()[:4]}"
            session_id = execution_data.get('session_id') or f"sess_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                execution_data['id'] = record_id
                execution_data['execution_id'] = execution_id
                execution_data['session_id'] = session_id
                execution_data['created_at'] = datetime.utcnow().isoformat()
                execution_data['updated_at'] = datetime.utcnow().isoformat()
                execution_data['queued_at'] = datetime.utcnow().isoformat()
                execution_data.setdefault('status', 'queued')
                execution_data.setdefault('current_step', 0)
                execution_data.setdefault('progress', 0.0)
                execution_data.setdefault('retry_count', 0)
                self._executions[record_id] = execution_data
                return execution_data
            
            from backend.schemas.training_models import PipelineExecution
            
            with self._db_manager.get_db_session() as db:
                execution = PipelineExecution(
                    id=record_id,
                    tenant_id=execution_data.get('tenant_id'),
                    execution_id=execution_id,
                    pipeline_id=execution_data.get('pipeline_id'),
                    session_id=session_id,
                    user_id=execution_data.get('user_id'),
                    status=execution_data.get('status', 'queued'),
                    current_step=execution_data.get('current_step', 0),
                    total_steps=execution_data.get('total_steps', 0),
                    progress=execution_data.get('progress', 0.0),
                    pipeline_snapshot=execution_data.get('pipeline_snapshot'),
                    runtime_config=execution_data.get('runtime_config'),
                    parent_execution_id=execution_data.get('parent_execution_id'),
                    metadata_=execution_data.get('metadata')
                )
                
                db.add(execution)
                db.commit()
                db.refresh(execution)
                
                return execution.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create pipeline execution: {e}")
            raise
    
    def get_by_id(self, execution_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """通过执行ID获取"""
        try:
            if self._use_memory_storage:
                for record in self._executions.values():
                    if record.get('execution_id') == execution_id and record.get('tenant_id') == tenant_id:
                        return record
                return None
            
            from backend.schemas.training_models import PipelineExecution
            
            with self._db_manager.get_db_session() as db:
                execution = db.query(PipelineExecution).filter(
                    PipelineExecution.execution_id == execution_id,
                    PipelineExecution.tenant_id == tenant_id
                ).first()
                
                return execution.to_dict() if execution else None
                
        except Exception as e:
            logger.error(f"Failed to get execution by ID: {e}")
            return None
    
    def get_by_session_id(self, session_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """通过会话ID获取"""
        try:
            if self._use_memory_storage:
                for record in self._executions.values():
                    if record.get('session_id') == session_id and record.get('tenant_id') == tenant_id:
                        return record
                return None
            
            from backend.schemas.training_models import PipelineExecution
            
            with self._db_manager.get_db_session() as db:
                execution = db.query(PipelineExecution).filter(
                    PipelineExecution.session_id == session_id,
                    PipelineExecution.tenant_id == tenant_id
                ).first()
                
                return execution.to_dict() if execution else None
                
        except Exception as e:
            logger.error(f"Failed to get execution by session ID: {e}")
            return None
    
    def update(self, execution_id: str, tenant_id: str, 
              updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新执行记录"""
        try:
            now = datetime.utcnow()
            
            if self._use_memory_storage:
                for record in self._executions.values():
                    if record.get('execution_id') == execution_id and record.get('tenant_id') == tenant_id:
                        record.update(updates)
                        record['updated_at'] = now.isoformat()
                        return record
                return None
            
            from backend.schemas.training_models import PipelineExecution
            
            with self._db_manager.get_db_session() as db:
                execution = db.query(PipelineExecution).filter(
                    PipelineExecution.execution_id == execution_id,
                    PipelineExecution.tenant_id == tenant_id
                ).first()
                
                if not execution:
                    return None
                
                for key, value in updates.items():
                    if hasattr(execution, key):
                        setattr(execution, key, value)
                
                db.commit()
                db.refresh(execution)
                
                return execution.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to update execution: {e}")
            return None
    
    def update_status(self, execution_id: str, tenant_id: str, status: str,
                     result: Optional[Dict] = None,
                     error_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """更新执行状态"""
        updates = {'status': status}
        
        now = datetime.utcnow()
        if status == 'running':
            updates['started_at'] = now.isoformat() if self._use_memory_storage else now
        elif status in ['completed', 'failed', 'cancelled']:
            updates['completed_at'] = now.isoformat() if self._use_memory_storage else now
        elif status == 'paused':
            updates['paused_at'] = now.isoformat() if self._use_memory_storage else now
        elif status == 'resuming':
            updates['resumed_at'] = now.isoformat() if self._use_memory_storage else now
        
        if result is not None:
            updates['result'] = result
        if error_message:
            updates['error_message'] = error_message
        
        return self.update(execution_id, tenant_id, updates)
    
    def update_progress(self, execution_id: str, tenant_id: str, 
                       current_step: int, progress: float) -> Optional[Dict[str, Any]]:
        """更新执行进度"""
        return self.update(execution_id, tenant_id, {
            'current_step': current_step,
            'progress': progress
        })
    
    def list_by_pipeline(self, pipeline_id: str, tenant_id: str,
                        status: Optional[str] = None,
                        limit: int = 20, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """获取流水线的执行记录列表"""
        try:
            if self._use_memory_storage:
                results = []
                for record in self._executions.values():
                    if record.get('pipeline_id') != pipeline_id:
                        continue
                    if record.get('tenant_id') != tenant_id:
                        continue
                    if status and record.get('status') != status:
                        continue
                    results.append(record)
                
                results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(results)
                return results[offset:offset + limit], total
            
            from backend.schemas.training_models import PipelineExecution
            
            with self._db_manager.get_db_session() as db:
                query = db.query(PipelineExecution).filter(
                    PipelineExecution.pipeline_id == pipeline_id,
                    PipelineExecution.tenant_id == tenant_id
                )
                
                if status:
                    query = query.filter(PipelineExecution.status == status)
                
                total = query.count()
                executions = query.order_by(PipelineExecution.created_at.desc()).offset(offset).limit(limit).all()
                
                return [e.to_dict() for e in executions], total
                
        except Exception as e:
            logger.error(f"Failed to list executions by pipeline: {e}")
            return [], 0
    
    def list_by_tenant(self, tenant_id: str, status: Optional[str] = None,
                      user_id: Optional[str] = None,
                      limit: int = 50, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """获取租户的执行记录列表"""
        try:
            if self._use_memory_storage:
                results = []
                for record in self._executions.values():
                    if record.get('tenant_id') != tenant_id:
                        continue
                    if status and record.get('status') != status:
                        continue
                    if user_id and record.get('user_id') != user_id:
                        continue
                    results.append(record)
                
                results.sort(key=lambda x: x.get('created_at', ''), reverse=True)
                total = len(results)
                return results[offset:offset + limit], total
            
            from backend.schemas.training_models import PipelineExecution
            
            with self._db_manager.get_db_session() as db:
                query = db.query(PipelineExecution).filter(
                    PipelineExecution.tenant_id == tenant_id
                )
                
                if status:
                    query = query.filter(PipelineExecution.status == status)
                if user_id:
                    query = query.filter(PipelineExecution.user_id == user_id)
                
                total = query.count()
                executions = query.order_by(PipelineExecution.created_at.desc()).offset(offset).limit(limit).all()
                
                return [e.to_dict() for e in executions], total
                
        except Exception as e:
            logger.error(f"Failed to list executions: {e}")
            return [], 0
    
    def get_running_executions(self, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取正在运行的执行记录"""
        try:
            if self._use_memory_storage:
                results = []
                for record in self._executions.values():
                    if tenant_id and record.get('tenant_id') != tenant_id:
                        continue
                    if record.get('status') == 'running':
                        results.append(record)
                return results
            
            from backend.schemas.training_models import PipelineExecution
            
            with self._db_manager.get_db_session() as db:
                query = db.query(PipelineExecution).filter(
                    PipelineExecution.status == 'running'
                )
                
                if tenant_id:
                    query = query.filter(PipelineExecution.tenant_id == tenant_id)
                
                executions = query.all()
                return [e.to_dict() for e in executions]
                
        except Exception as e:
            logger.error(f"Failed to get running executions: {e}")
            return []


# ==============================================================================
# 步骤执行仓库
# ==============================================================================

class PipelineStepExecutionRepository:
    """流水线步骤执行记录数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化步骤执行记录仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._steps: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database manager, falling back to memory storage")
                self._use_memory_storage = True
                self._steps: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建步骤执行记录"""
        try:
            record_id = step_data.get('id') or _generate_id()
            step_execution_id = step_data.get('step_execution_id') or f"step_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                step_data['id'] = record_id
                step_data['step_execution_id'] = step_execution_id
                step_data['created_at'] = datetime.utcnow().isoformat()
                step_data['updated_at'] = datetime.utcnow().isoformat()
                step_data.setdefault('status', 'pending')
                step_data.setdefault('progress', 0.0)
                step_data.setdefault('retry_count', 0)
                self._steps[record_id] = step_data
                return step_data
            
            from backend.schemas.training_models import PipelineStepExecution
            
            with self._db_manager.get_db_session() as db:
                step = PipelineStepExecution(
                    id=record_id,
                    tenant_id=step_data.get('tenant_id'),
                    step_execution_id=step_execution_id,
                    execution_id=step_data.get('execution_id'),
                    step_index=step_data.get('step_index', 0),
                    step_name=step_data.get('step_name'),
                    step_type=step_data.get('step_type'),
                    status=step_data.get('status', 'pending'),
                    progress=step_data.get('progress', 0.0),
                    step_config=step_data.get('step_config'),
                    input_data=step_data.get('input_data'),
                    failure_policy=step_data.get('failure_policy', 'rollback'),
                    metadata_=step_data.get('metadata')
                )
                
                db.add(step)
                db.commit()
                db.refresh(step)
                
                return step.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create step execution: {e}")
            raise
    
    def batch_create(self, steps_data: List[Dict[str, Any]], tenant_id: str) -> List[Dict[str, Any]]:
        """批量创建步骤执行记录"""
        results = []
        for step_data in steps_data:
            step_data['tenant_id'] = tenant_id
            try:
                result = self.create(step_data)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to create step: {e}")
                continue
        return results
    
    def get_by_id(self, step_execution_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """通过步骤执行ID获取"""
        try:
            if self._use_memory_storage:
                for record in self._steps.values():
                    if record.get('step_execution_id') == step_execution_id and record.get('tenant_id') == tenant_id:
                        return record
                return None
            
            from backend.schemas.training_models import PipelineStepExecution
            
            with self._db_manager.get_db_session() as db:
                step = db.query(PipelineStepExecution).filter(
                    PipelineStepExecution.step_execution_id == step_execution_id,
                    PipelineStepExecution.tenant_id == tenant_id
                ).first()
                
                return step.to_dict() if step else None
                
        except Exception as e:
            logger.error(f"Failed to get step execution: {e}")
            return None
    
    def update(self, step_execution_id: str, tenant_id: str, 
              updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新步骤执行记录"""
        try:
            now = datetime.utcnow()
            
            if self._use_memory_storage:
                for record in self._steps.values():
                    if record.get('step_execution_id') == step_execution_id and record.get('tenant_id') == tenant_id:
                        record.update(updates)
                        record['updated_at'] = now.isoformat()
                        return record
                return None
            
            from backend.schemas.training_models import PipelineStepExecution
            
            with self._db_manager.get_db_session() as db:
                step = db.query(PipelineStepExecution).filter(
                    PipelineStepExecution.step_execution_id == step_execution_id,
                    PipelineStepExecution.tenant_id == tenant_id
                ).first()
                
                if not step:
                    return None
                
                for key, value in updates.items():
                    if hasattr(step, key):
                        setattr(step, key, value)
                
                db.commit()
                db.refresh(step)
                
                return step.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to update step execution: {e}")
            return None
    
    def update_status(self, step_execution_id: str, tenant_id: str, status: str,
                     output_data: Optional[Dict] = None,
                     metrics: Optional[Dict] = None,
                     error_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """更新步骤执行状态"""
        updates = {'status': status}
        
        now = datetime.utcnow()
        if status == 'running':
            updates['started_at'] = now.isoformat() if self._use_memory_storage else now
        elif status in ['completed', 'failed', 'skipped']:
            updates['completed_at'] = now.isoformat() if self._use_memory_storage else now
            # 计算执行时长
            # 需要先获取 started_at 来计算
        
        if output_data is not None:
            updates['output_data'] = output_data
        if metrics is not None:
            updates['metrics'] = metrics
        if error_message:
            updates['error_message'] = error_message
        
        return self.update(step_execution_id, tenant_id, updates)
    
    def list_by_execution(self, execution_id: str, tenant_id: str) -> List[Dict[str, Any]]:
        """获取执行记录的所有步骤"""
        try:
            if self._use_memory_storage:
                results = []
                for record in self._steps.values():
                    if record.get('execution_id') == execution_id and record.get('tenant_id') == tenant_id:
                        results.append(record)
                
                results.sort(key=lambda x: x.get('step_index', 0))
                return results
            
            from backend.schemas.training_models import PipelineStepExecution
            
            with self._db_manager.get_db_session() as db:
                steps = db.query(PipelineStepExecution).filter(
                    PipelineStepExecution.execution_id == execution_id,
                    PipelineStepExecution.tenant_id == tenant_id
                ).order_by(PipelineStepExecution.step_index).all()
                
                return [s.to_dict() for s in steps]
                
        except Exception as e:
            logger.error(f"Failed to list steps by execution: {e}")
            return []


# ==============================================================================
# 流水线模板仓库
# ==============================================================================

class PipelineTemplateRepository:
    """流水线模板数据访问层"""
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化模板仓库"""
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            self._templates: Dict[str, Dict] = {}
            self._db_manager = None
        else:
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_manager = get_database_manager()
            except ImportError:
                logger.warning("Cannot import database manager, falling back to memory storage")
                self._use_memory_storage = True
                self._templates: Dict[str, Dict] = {}
                self._db_manager = None
    
    def create(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建模板"""
        try:
            record_id = template_data.get('id') or _generate_id()
            template_id = template_data.get('template_id') or f"tmpl_{_generate_id()[:8]}"
            
            if self._use_memory_storage:
                template_data['id'] = record_id
                template_data['template_id'] = template_id
                template_data['created_at'] = datetime.utcnow().isoformat()
                template_data['updated_at'] = datetime.utcnow().isoformat()
                template_data.setdefault('template_type', 'custom')
                template_data.setdefault('is_active', True)
                template_data.setdefault('is_public', False)
                template_data.setdefault('usage_count', 0)
                self._templates[record_id] = template_data
                return template_data
            
            from backend.schemas.training_models import PipelineTemplate
            
            with self._db_manager.get_db_session() as db:
                template = PipelineTemplate(
                    id=record_id,
                    tenant_id=template_data.get('tenant_id'),
                    template_id=template_id,
                    name=template_data.get('name'),
                    description=template_data.get('description'),
                    user_id=template_data.get('user_id'),
                    template_type=template_data.get('template_type', 'custom'),
                    category=template_data.get('category'),
                    steps_template=template_data.get('steps_template', []),
                    default_config=template_data.get('default_config'),
                    required_params=template_data.get('required_params'),
                    version=template_data.get('version', '1.0.0'),
                    is_active=template_data.get('is_active', True),
                    is_public=template_data.get('is_public', False),
                    tags=template_data.get('tags'),
                    metadata_=template_data.get('metadata')
                )
                
                db.add(template)
                db.commit()
                db.refresh(template)
                
                return template.to_dict()
                
        except Exception as e:
            logger.error(f"Failed to create pipeline template: {e}")
            raise
    
    def get_by_id(self, template_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
        """通过模板ID获取"""
        try:
            if self._use_memory_storage:
                for record in self._templates.values():
                    if record.get('template_id') == template_id:
                        # 系统模板对所有租户可见
                        if record.get('template_type') == 'system' or record.get('tenant_id') == tenant_id:
                            return record
                return None
            
            from backend.schemas.training_models import PipelineTemplate
            from sqlalchemy import or_
            
            with self._db_manager.get_db_session() as db:
                template = db.query(PipelineTemplate).filter(
                    PipelineTemplate.template_id == template_id,
                    or_(
                        PipelineTemplate.tenant_id == tenant_id,
                        PipelineTemplate.template_type == 'system'
                    )
                ).first()
                
                return template.to_dict() if template else None
                
        except Exception as e:
            logger.error(f"Failed to get template: {e}")
            return None
    
    def list_templates(self, tenant_id: str, category: Optional[str] = None,
                      include_system: bool = True,
                      limit: int = 50, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """获取模板列表"""
        try:
            if self._use_memory_storage:
                results = []
                for record in self._templates.values():
                    if not record.get('is_active', True):
                        continue
                    
                    # 租户自己的模板或系统模板
                    if record.get('tenant_id') != tenant_id:
                        if not (include_system and record.get('template_type') == 'system'):
                            continue
                    
                    if category and record.get('category') != category:
                        continue
                    
                    results.append(record)
                
                results.sort(key=lambda x: x.get('usage_count', 0), reverse=True)
                total = len(results)
                return results[offset:offset + limit], total
            
            from backend.schemas.training_models import PipelineTemplate
            from sqlalchemy import or_
            
            with self._db_manager.get_db_session() as db:
                query = db.query(PipelineTemplate).filter(
                    PipelineTemplate.is_active == True
                )
                
                # 租户自己的模板或系统模板
                if include_system:
                    query = query.filter(
                        or_(
                            PipelineTemplate.tenant_id == tenant_id,
                            PipelineTemplate.template_type == 'system'
                        )
                    )
                else:
                    query = query.filter(PipelineTemplate.tenant_id == tenant_id)
                
                if category:
                    query = query.filter(PipelineTemplate.category == category)
                
                total = query.count()
                templates = query.order_by(PipelineTemplate.usage_count.desc()).offset(offset).limit(limit).all()
                
                return [t.to_dict() for t in templates], total
                
        except Exception as e:
            logger.error(f"Failed to list templates: {e}")
            return [], 0
    
    def increment_usage(self, template_id: str, tenant_id: str) -> bool:
        """增加模板使用次数"""
        try:
            if self._use_memory_storage:
                for record in self._templates.values():
                    if record.get('template_id') == template_id:
                        record['usage_count'] = record.get('usage_count', 0) + 1
                        return True
                return False
            
            from backend.schemas.training_models import PipelineTemplate
            
            with self._db_manager.get_db_session() as db:
                template = db.query(PipelineTemplate).filter(
                    PipelineTemplate.template_id == template_id
                ).first()
                
                if template:
                    template.usage_count = (template.usage_count or 0) + 1
                    db.commit()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Failed to increment template usage: {e}")
            return False


# ==============================================================================
# 获取仓库实例的辅助函数
# ==============================================================================

# 全局仓库实例缓存
_pipeline_repository: Optional[TrainingPipelineRepository] = None
_execution_repository: Optional[PipelineExecutionRepository] = None
_step_execution_repository: Optional[PipelineStepExecutionRepository] = None
_template_repository: Optional[PipelineTemplateRepository] = None


def get_pipeline_repository(use_memory_storage: bool = False) -> TrainingPipelineRepository:
    """获取流水线仓库实例"""
    global _pipeline_repository
    if _pipeline_repository is None:
        _pipeline_repository = TrainingPipelineRepository(use_memory_storage=use_memory_storage)
    return _pipeline_repository


def get_execution_repository(use_memory_storage: bool = False) -> PipelineExecutionRepository:
    """获取执行记录仓库实例"""
    global _execution_repository
    if _execution_repository is None:
        _execution_repository = PipelineExecutionRepository(use_memory_storage=use_memory_storage)
    return _execution_repository


def get_step_execution_repository(use_memory_storage: bool = False) -> PipelineStepExecutionRepository:
    """获取步骤执行记录仓库实例"""
    global _step_execution_repository
    if _step_execution_repository is None:
        _step_execution_repository = PipelineStepExecutionRepository(use_memory_storage=use_memory_storage)
    return _step_execution_repository


def get_template_repository(use_memory_storage: bool = False) -> PipelineTemplateRepository:
    """获取模板仓库实例"""
    global _template_repository
    if _template_repository is None:
        _template_repository = PipelineTemplateRepository(use_memory_storage=use_memory_storage)
    return _template_repository


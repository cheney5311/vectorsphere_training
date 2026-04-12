"""数据预处理仓库层

实现数据预处理任务、历史记录等的持久化操作。
支持内存存储和数据库持久化两种模式。
"""

import sys
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.modules.dataset.dataset_exceptions import (
    PreprocessingTaskNotFoundError,
    DataPreprocessingError,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 预处理任务仓库
# ============================================================================

class PreprocessingTaskRepository:
    """预处理任务仓库
    
    管理预处理任务的CRUD操作。
    支持内存存储和数据库持久化两种模式。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化预处理任务仓库
        
        Args:
            db_service: 数据库服务实例
            use_memory_storage: 是否使用内存存储
        """
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._tasks: Dict[str, Dict[str, Any]] = {}
        
        if not use_memory_storage:
            self._init_database(db_service)
    
    def _init_database(self, db_service=None):
        """初始化数据库连接"""
        try:
            if db_service:
                self._db_service = db_service
            else:
                from backend.modules.database.manager import get_database_manager
                self._db_service = get_database_manager()
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建预处理任务
        
        Args:
            task_data: 任务数据字典
            
        Returns:
            创建的任务数据
        """
        if not task_data.get('task_id'):
            task_data['task_id'] = str(uuid.uuid4())
        task_data['created_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._tasks[task_data['task_id']] = task_data.copy()
            logger.info(f"Created preprocessing task (memory): {task_data['task_id']}")
            return task_data
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingTask
            
            with self._get_session() as session:
                if session:
                    db_task = PreprocessingTask(
                        id=uuid.UUID(task_data['task_id']),
                        dataset_id=task_data.get('dataset_id', ''),
                        user_id=task_data.get('user_id', ''),
                        tenant_id=task_data.get('tenant_id'),
                        task_type=task_data.get('task_type', 'preprocessing'),
                        task_name=task_data.get('task_name'),
                        description=task_data.get('description'),
                        status=task_data.get('status', 'pending'),
                        priority=task_data.get('priority', 0),
                        config=task_data.get('config', {}),
                        result=task_data.get('result'),
                        error_message=task_data.get('error_message'),
                        snapshot_path=task_data.get('snapshot_path'),
                        original_rows=task_data.get('original_rows', 0),
                        final_rows=task_data.get('final_rows', 0),
                        original_columns=task_data.get('original_columns', 0),
                        final_columns=task_data.get('final_columns', 0),
                        started_at=task_data.get('started_at'),
                        completed_at=task_data.get('completed_at'),
                        metadata_=task_data.get('metadata', {}),
                    )
                    session.add(db_task)
                    session.flush()
                    logger.info(f"Created preprocessing task (db): {task_data['task_id']}")
                    return db_task.to_dict()
        except Exception as e:
            logger.error(f"Failed to create preprocessing task in database: {e}")
            self._tasks[task_data['task_id']] = task_data.copy()
        
        return task_data
    
    def get_by_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            任务数据，不存在则返回None
        """
        if self._use_memory_storage:
            return self._tasks.get(task_id)
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingTask
            
            with self._get_session() as session:
                if session:
                    db_task = session.query(PreprocessingTask).filter(
                        PreprocessingTask.id == uuid.UUID(task_id)
                    ).first()
                    if db_task:
                        return db_task.to_dict()
        except Exception as e:
            logger.error(f"Failed to get preprocessing task from database: {e}")
            return self._tasks.get(task_id)
        
        return None
    
    def get_by_dataset(
        self,
        dataset_id: str,
        status_filter: Optional[List[str]] = None,
        task_type_filter: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取数据集的任务列表
        
        Args:
            dataset_id: 数据集ID
            status_filter: 状态过滤
            task_type_filter: 任务类型过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            任务列表和总数
        """
        if self._use_memory_storage:
            filtered = []
            for task in self._tasks.values():
                if task.get('dataset_id') != dataset_id:
                    continue
                if status_filter and task.get('status') not in status_filter:
                    continue
                if task_type_filter and task.get('task_type') not in task_type_filter:
                    continue
                filtered.append(task)
            
            filtered.sort(key=lambda x: x.get('created_at') or datetime.min, reverse=True)
            total = len(filtered)
            return filtered[offset:offset + limit], total
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingTask
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    query = session.query(PreprocessingTask).filter(
                        PreprocessingTask.dataset_id == dataset_id
                    )
                    if status_filter:
                        query = query.filter(PreprocessingTask.status.in_(status_filter))
                    if task_type_filter:
                        query = query.filter(PreprocessingTask.task_type.in_(task_type_filter))
                    
                    total = query.count()
                    tasks = query.order_by(desc(PreprocessingTask.created_at)).offset(offset).limit(limit).all()
                    return [t.to_dict() for t in tasks], total
        except Exception as e:
            logger.error(f"Failed to get preprocessing tasks from database: {e}")
        
        return [], 0
    
    def get_by_user(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        status_filter: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的任务列表
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            status_filter: 状态过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            任务列表和总数
        """
        if self._use_memory_storage:
            filtered = []
            for task in self._tasks.values():
                if task.get('user_id') != user_id:
                    continue
                if tenant_id is not None and task.get('tenant_id') != tenant_id:
                    continue
                if status_filter and task.get('status') not in status_filter:
                    continue
                filtered.append(task)
            
            filtered.sort(key=lambda x: x.get('created_at') or datetime.min, reverse=True)
            total = len(filtered)
            return filtered[offset:offset + limit], total
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingTask
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    query = session.query(PreprocessingTask).filter(
                        PreprocessingTask.user_id == user_id
                    )
                    if tenant_id:
                        query = query.filter(PreprocessingTask.tenant_id == tenant_id)
                    if status_filter:
                        query = query.filter(PreprocessingTask.status.in_(status_filter))
                    
                    total = query.count()
                    tasks = query.order_by(desc(PreprocessingTask.created_at)).offset(offset).limit(limit).all()
                    return [t.to_dict() for t in tasks], total
        except Exception as e:
            logger.error(f"Failed to get preprocessing tasks from database: {e}")
        
        return [], 0
    
    def update(self, task_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新任务
        
        Args:
            task_id: 任务ID
            update_data: 更新数据
            
        Returns:
            更新后的任务数据
        """
        if self._use_memory_storage:
            if task_id not in self._tasks:
                raise PreprocessingTaskNotFoundError(task_id)
            self._tasks[task_id].update(update_data)
            logger.info(f"Updated preprocessing task (memory): {task_id}")
            return self._tasks[task_id]
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingTask
            
            with self._get_session() as session:
                if session:
                    db_task = session.query(PreprocessingTask).filter(
                        PreprocessingTask.id == uuid.UUID(task_id)
                    ).first()
                    if not db_task:
                        raise PreprocessingTaskNotFoundError(task_id)
                    
                    for key, value in update_data.items():
                        if key == 'metadata':
                            setattr(db_task, 'metadata_', value)
                        elif hasattr(db_task, key):
                            setattr(db_task, key, value)
                    
                    session.flush()
                    logger.info(f"Updated preprocessing task (db): {task_id}")
                    return db_task.to_dict()
        except PreprocessingTaskNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to update preprocessing task in database: {e}")
        
        return None
    
    def update_status(
        self,
        task_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """更新任务状态
        
        Args:
            task_id: 任务ID
            status: 新状态
            result: 任务结果
            error_message: 错误信息
            
        Returns:
            更新后的任务数据
        """
        update_data = {'status': status}
        if result:
            update_data['result'] = result
        if error_message:
            update_data['error_message'] = error_message
        
        if status == "processing":
            update_data['started_at'] = datetime.utcnow()
        elif status in ("completed", "failed", "cancelled"):
            update_data['completed_at'] = datetime.utcnow()
        
        return self.update(task_id, update_data)
    
    def delete(self, task_id: str) -> bool:
        """删除任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory_storage:
            if task_id in self._tasks:
                del self._tasks[task_id]
                logger.info(f"Deleted preprocessing task (memory): {task_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingTask
            
            with self._get_session() as session:
                if session:
                    db_task = session.query(PreprocessingTask).filter(
                        PreprocessingTask.id == uuid.UUID(task_id)
                    ).first()
                    if db_task:
                        session.delete(db_task)
                        logger.info(f"Deleted preprocessing task (db): {task_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete preprocessing task from database: {e}")
        
        return False
    
    def get_pending_tasks(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取待处理的任务
        
        Args:
            limit: 返回数量限制
            
        Returns:
            待处理任务列表
        """
        if self._use_memory_storage:
            pending = [t for t in self._tasks.values() if t.get('status') == "pending"]
            pending.sort(key=lambda x: (x.get('priority', 0), x.get('created_at') or datetime.min), reverse=True)
            return pending[:limit]
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingTask
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    tasks = session.query(PreprocessingTask).filter(
                        PreprocessingTask.status == "pending"
                    ).order_by(
                        desc(PreprocessingTask.priority),
                        PreprocessingTask.created_at
                    ).limit(limit).all()
                    return [t.to_dict() for t in tasks]
        except Exception as e:
            logger.error(f"Failed to get pending tasks from database: {e}")
        
        return []


# ============================================================================
# 预处理历史仓库
# ============================================================================

class PreprocessingHistoryRepository:
    """预处理历史仓库
    
    管理预处理历史记录的CRUD操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化历史仓库"""
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._histories: Dict[str, Dict[str, Any]] = {}
        
        if not use_memory_storage:
            self._init_database(db_service)
    
    def _init_database(self, db_service=None):
        """初始化数据库连接"""
        try:
            if db_service:
                self._db_service = db_service
            else:
                from backend.modules.database.manager import get_database_manager
                self._db_service = get_database_manager()
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, history_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建历史记录
        
        Args:
            history_data: 历史记录数据字典
            
        Returns:
            创建的历史记录数据
        """
        if not history_data.get('history_id'):
            history_data['history_id'] = str(uuid.uuid4())
        history_data['executed_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._histories[history_data['history_id']] = history_data.copy()
            logger.info(f"Created preprocessing history (memory): {history_data['history_id']}")
            return history_data
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingHistory
            
            with self._get_session() as session:
                if session:
                    db_history = PreprocessingHistory(
                        id=uuid.UUID(history_data['history_id']),
                        dataset_id=history_data.get('dataset_id', ''),
                        task_id=uuid.UUID(history_data['task_id']) if history_data.get('task_id') else None,
                        user_id=history_data.get('user_id', ''),
                        tenant_id=history_data.get('tenant_id'),
                        operation_type=history_data.get('operation_type', ''),
                        operation_name=history_data.get('operation_name'),
                        operation_config=history_data.get('operation_config', {}),
                        operation_result=history_data.get('operation_result', {}),
                        rows_before=history_data.get('rows_before', 0),
                        rows_after=history_data.get('rows_after', 0),
                        columns_before=history_data.get('columns_before', 0),
                        columns_after=history_data.get('columns_after', 0),
                        columns_added=history_data.get('columns_added', []),
                        columns_removed=history_data.get('columns_removed', []),
                        columns_modified=history_data.get('columns_modified', []),
                        snapshot_path=history_data.get('snapshot_path'),
                        can_rollback=history_data.get('can_rollback', True),
                        duration_ms=history_data.get('duration_ms', 0),
                    )
                    session.add(db_history)
                    session.flush()
                    logger.info(f"Created preprocessing history (db): {history_data['history_id']}")
                    return db_history.to_dict()
        except Exception as e:
            logger.error(f"Failed to create preprocessing history in database: {e}")
            self._histories[history_data['history_id']] = history_data.copy()
        
        return history_data
    
    def get_by_id(self, history_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取历史记录
        
        Args:
            history_id: 历史记录ID
            
        Returns:
            历史记录数据，不存在则返回None
        """
        if self._use_memory_storage:
            return self._histories.get(history_id)
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingHistory
            
            with self._get_session() as session:
                if session:
                    db_history = session.query(PreprocessingHistory).filter(
                        PreprocessingHistory.id == uuid.UUID(history_id)
                    ).first()
                    if db_history:
                        return db_history.to_dict()
        except Exception as e:
            logger.error(f"Failed to get preprocessing history from database: {e}")
            return self._histories.get(history_id)
        
        return None
    
    def get_by_dataset(
        self,
        dataset_id: str,
        operation_type_filter: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取数据集的历史记录
        
        Args:
            dataset_id: 数据集ID
            operation_type_filter: 操作类型过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            历史记录列表和总数
        """
        if self._use_memory_storage:
            filtered = []
            for history in self._histories.values():
                if history.get('dataset_id') != dataset_id:
                    continue
                if operation_type_filter and history.get('operation_type') not in operation_type_filter:
                    continue
                filtered.append(history)
            
            filtered.sort(key=lambda x: x.get('executed_at') or datetime.min, reverse=True)
            total = len(filtered)
            return filtered[offset:offset + limit], total
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingHistory
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    query = session.query(PreprocessingHistory).filter(
                        PreprocessingHistory.dataset_id == dataset_id
                    )
                    if operation_type_filter:
                        query = query.filter(PreprocessingHistory.operation_type.in_(operation_type_filter))
                    
                    total = query.count()
                    histories = query.order_by(desc(PreprocessingHistory.executed_at)).offset(offset).limit(limit).all()
                    return [h.to_dict() for h in histories], total
        except Exception as e:
            logger.error(f"Failed to get preprocessing histories from database: {e}")
        
        return [], 0
    
    def get_by_task(self, task_id: str) -> List[Dict[str, Any]]:
        """获取任务的历史记录
        
        Args:
            task_id: 任务ID
            
        Returns:
            历史记录列表
        """
        if self._use_memory_storage:
            histories = [h for h in self._histories.values() if h.get('task_id') == task_id]
            histories.sort(key=lambda x: x.get('executed_at') or datetime.min)
            return histories
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingHistory
            
            with self._get_session() as session:
                if session:
                    histories = session.query(PreprocessingHistory).filter(
                        PreprocessingHistory.task_id == uuid.UUID(task_id)
                    ).order_by(PreprocessingHistory.executed_at).all()
                    return [h.to_dict() for h in histories]
        except Exception as e:
            logger.error(f"Failed to get preprocessing histories from database: {e}")
        
        return []
    
    def get_latest_by_dataset(
        self,
        dataset_id: str,
        can_rollback_only: bool = False
    ) -> Optional[Dict[str, Any]]:
        """获取数据集最新的历史记录
        
        Args:
            dataset_id: 数据集ID
            can_rollback_only: 是否只获取可回滚的记录
            
        Returns:
            最新的历史记录数据
        """
        if self._use_memory_storage:
            histories = []
            for history in self._histories.values():
                if history.get('dataset_id') != dataset_id:
                    continue
                if can_rollback_only and not history.get('can_rollback', True):
                    continue
                histories.append(history)
            
            if not histories:
                return None
            
            return max(histories, key=lambda x: x.get('executed_at') or datetime.min)
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingHistory
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    query = session.query(PreprocessingHistory).filter(
                        PreprocessingHistory.dataset_id == dataset_id
                    )
                    if can_rollback_only:
                        query = query.filter(PreprocessingHistory.can_rollback == True)
                    
                    history = query.order_by(desc(PreprocessingHistory.executed_at)).first()
                    if history:
                        return history.to_dict()
        except Exception as e:
            logger.error(f"Failed to get latest preprocessing history from database: {e}")
        
        return None
    
    def mark_as_non_rollbackable(self, history_id: str) -> Optional[Dict[str, Any]]:
        """标记历史记录为不可回滚
        
        Args:
            history_id: 历史记录ID
            
        Returns:
            更新后的历史记录数据
        """
        if self._use_memory_storage:
            if history_id in self._histories:
                self._histories[history_id]['can_rollback'] = False
                return self._histories[history_id]
            return None
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingHistory
            
            with self._get_session() as session:
                if session:
                    db_history = session.query(PreprocessingHistory).filter(
                        PreprocessingHistory.id == uuid.UUID(history_id)
                    ).first()
                    if db_history:
                        db_history.can_rollback = False
                        session.flush()
                        return db_history.to_dict()
        except Exception as e:
            logger.error(f"Failed to mark history as non-rollbackable in database: {e}")
        
        return None
    
    def delete(self, history_id: str) -> bool:
        """删除历史记录
        
        Args:
            history_id: 历史记录ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory_storage:
            if history_id in self._histories:
                del self._histories[history_id]
                logger.info(f"Deleted preprocessing history (memory): {history_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingHistory
            
            with self._get_session() as session:
                if session:
                    db_history = session.query(PreprocessingHistory).filter(
                        PreprocessingHistory.id == uuid.UUID(history_id)
                    ).first()
                    if db_history:
                        session.delete(db_history)
                        logger.info(f"Deleted preprocessing history (db): {history_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete preprocessing history from database: {e}")
        
        return False


# ============================================================================
# 预处理流水线仓库
# ============================================================================

class PreprocessingPipelineRepository:
    """预处理流水线仓库
    
    管理预处理流水线模板的CRUD操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化流水线仓库"""
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._pipelines: Dict[str, Dict[str, Any]] = {}
        
        if not use_memory_storage:
            self._init_database(db_service)
    
    def _init_database(self, db_service=None):
        """初始化数据库连接"""
        try:
            if db_service:
                self._db_service = db_service
            else:
                from backend.modules.database.manager import get_database_manager
                self._db_service = get_database_manager()
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建流水线
        
        Args:
            pipeline_data: 流水线数据字典
            
        Returns:
            创建的流水线数据
        """
        if not pipeline_data.get('pipeline_id'):
            pipeline_data['pipeline_id'] = str(uuid.uuid4())
        pipeline_data['created_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._pipelines[pipeline_data['pipeline_id']] = pipeline_data.copy()
            logger.info(f"Created preprocessing pipeline (memory): {pipeline_data['pipeline_id']}")
            return pipeline_data
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingPipeline
            
            with self._get_session() as session:
                if session:
                    db_pipeline = PreprocessingPipeline(
                        id=uuid.UUID(pipeline_data['pipeline_id']),
                        user_id=pipeline_data.get('user_id', ''),
                        tenant_id=pipeline_data.get('tenant_id'),
                        name=pipeline_data.get('name', ''),
                        description=pipeline_data.get('description'),
                        operations=pipeline_data.get('operations', []),
                        is_template=pipeline_data.get('is_template', False),
                        is_public=pipeline_data.get('is_public', False),
                        usage_count=pipeline_data.get('usage_count', 0),
                    )
                    session.add(db_pipeline)
                    session.flush()
                    logger.info(f"Created preprocessing pipeline (db): {pipeline_data['pipeline_id']}")
                    return db_pipeline.to_dict()
        except Exception as e:
            logger.error(f"Failed to create preprocessing pipeline in database: {e}")
            self._pipelines[pipeline_data['pipeline_id']] = pipeline_data.copy()
        
        return pipeline_data
    
    def get_by_id(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取流水线
        
        Args:
            pipeline_id: 流水线ID
            
        Returns:
            流水线数据
        """
        if self._use_memory_storage:
            return self._pipelines.get(pipeline_id)
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingPipeline
            
            with self._get_session() as session:
                if session:
                    db_pipeline = session.query(PreprocessingPipeline).filter(
                        PreprocessingPipeline.id == uuid.UUID(pipeline_id)
                    ).first()
                    if db_pipeline:
                        return db_pipeline.to_dict()
        except Exception as e:
            logger.error(f"Failed to get preprocessing pipeline from database: {e}")
            return self._pipelines.get(pipeline_id)
        
        return None
    
    def get_by_user(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        is_template: Optional[bool] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的流水线列表
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID
            is_template: 是否为模板
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            流水线列表和总数
        """
        if self._use_memory_storage:
            filtered = []
            for pipeline in self._pipelines.values():
                # 用户自己的流水线或公开的模板
                if pipeline.get('user_id') != user_id:
                    if not (pipeline.get('is_public') and pipeline.get('is_template')):
                        if tenant_id is None or pipeline.get('tenant_id') != tenant_id:
                            continue
                if is_template is not None and pipeline.get('is_template') != is_template:
                    continue
                filtered.append(pipeline)
            
            filtered.sort(key=lambda x: x.get('created_at') or datetime.min, reverse=True)
            total = len(filtered)
            return filtered[offset:offset + limit], total
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingPipeline
            from sqlalchemy import desc, or_
            
            with self._get_session() as session:
                if session:
                    # 用户自己的流水线或公开的模板
                    query = session.query(PreprocessingPipeline).filter(
                        or_(
                            PreprocessingPipeline.user_id == user_id,
                            PreprocessingPipeline.is_public == True
                        )
                    )
                    if is_template is not None:
                        query = query.filter(PreprocessingPipeline.is_template == is_template)
                    
                    total = query.count()
                    pipelines = query.order_by(desc(PreprocessingPipeline.created_at)).offset(offset).limit(limit).all()
                    return [p.to_dict() for p in pipelines], total
        except Exception as e:
            logger.error(f"Failed to get preprocessing pipelines from database: {e}")
        
        return [], 0
    
    def get_public_templates(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取公开的流水线模板
        
        Args:
            limit: 返回数量限制
            
        Returns:
            公开模板列表
        """
        if self._use_memory_storage:
            templates = [
                p for p in self._pipelines.values()
                if p.get('is_template') and p.get('is_public')
            ]
            templates.sort(key=lambda x: x.get('usage_count', 0), reverse=True)
            return templates[:limit]
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingPipeline
            from sqlalchemy import desc
            
            with self._get_session() as session:
                if session:
                    templates = session.query(PreprocessingPipeline).filter(
                        PreprocessingPipeline.is_template == True,
                        PreprocessingPipeline.is_public == True
                    ).order_by(desc(PreprocessingPipeline.usage_count)).limit(limit).all()
                    return [t.to_dict() for t in templates]
        except Exception as e:
            logger.error(f"Failed to get public templates from database: {e}")
        
        return []
    
    def update(self, pipeline_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新流水线
        
        Args:
            pipeline_id: 流水线ID
            update_data: 更新数据
            
        Returns:
            更新后的流水线数据
        """
        if self._use_memory_storage:
            if pipeline_id in self._pipelines:
                self._pipelines[pipeline_id].update(update_data)
                self._pipelines[pipeline_id]['updated_at'] = datetime.utcnow()
                logger.info(f"Updated preprocessing pipeline (memory): {pipeline_id}")
                return self._pipelines[pipeline_id]
            return None
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingPipeline
            
            with self._get_session() as session:
                if session:
                    db_pipeline = session.query(PreprocessingPipeline).filter(
                        PreprocessingPipeline.id == uuid.UUID(pipeline_id)
                    ).first()
                    if not db_pipeline:
                        return None
                    
                    for key, value in update_data.items():
                        if hasattr(db_pipeline, key):
                            setattr(db_pipeline, key, value)
                    
                    session.flush()
                    logger.info(f"Updated preprocessing pipeline (db): {pipeline_id}")
                    return db_pipeline.to_dict()
        except Exception as e:
            logger.error(f"Failed to update preprocessing pipeline in database: {e}")
        
        return None
    
    def increment_usage(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """增加流水线使用次数
        
        Args:
            pipeline_id: 流水线ID
            
        Returns:
            更新后的流水线数据
        """
        if self._use_memory_storage:
            if pipeline_id in self._pipelines:
                self._pipelines[pipeline_id]['usage_count'] = self._pipelines[pipeline_id].get('usage_count', 0) + 1
                return self._pipelines[pipeline_id]
            return None
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingPipeline
            
            with self._get_session() as session:
                if session:
                    db_pipeline = session.query(PreprocessingPipeline).filter(
                        PreprocessingPipeline.id == uuid.UUID(pipeline_id)
                    ).first()
                    if db_pipeline:
                        db_pipeline.usage_count = (db_pipeline.usage_count or 0) + 1
                        session.flush()
                        return db_pipeline.to_dict()
        except Exception as e:
            logger.error(f"Failed to increment pipeline usage in database: {e}")
        
        return None
    
    def delete(self, pipeline_id: str) -> bool:
        """删除流水线
        
        Args:
            pipeline_id: 流水线ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory_storage:
            if pipeline_id in self._pipelines:
                del self._pipelines[pipeline_id]
                logger.info(f"Deleted preprocessing pipeline (memory): {pipeline_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_preprocessing_db_models import PreprocessingPipeline
            
            with self._get_session() as session:
                if session:
                    db_pipeline = session.query(PreprocessingPipeline).filter(
                        PreprocessingPipeline.id == uuid.UUID(pipeline_id)
                    ).first()
                    if db_pipeline:
                        session.delete(db_pipeline)
                        logger.info(f"Deleted preprocessing pipeline (db): {pipeline_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete preprocessing pipeline from database: {e}")
        
        return False


# ============================================================================
# 特征存储仓库
# ============================================================================

class FeatureStoreRepository:
    """特征存储仓库
    
    管理特征定义的CRUD操作。
    """
    
    def __init__(self, db_service=None, use_memory_storage: bool = False):
        """初始化特征存储仓库"""
        self._use_memory_storage = use_memory_storage
        self._db_service = None
        self._features: Dict[str, Dict[str, Any]] = {}
        
        if not use_memory_storage:
            self._init_database(db_service)
    
    def _init_database(self, db_service=None):
        """初始化数据库连接"""
        try:
            if db_service:
                self._db_service = db_service
            else:
                from backend.modules.database.manager import get_database_manager
                self._db_service = get_database_manager()
        except ImportError as e:
            logger.warning(f"Database manager not available: {e}, using memory storage")
            self._use_memory_storage = True
    
    @contextmanager
    def _get_session(self):
        """获取数据库会话"""
        if self._db_service:
            with self._db_service.get_db_session() as session:
                yield session
        else:
            yield None
    
    def create(self, feature_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建特征
        
        Args:
            feature_data: 特征数据字典
            
        Returns:
            创建的特征数据
        """
        if not feature_data.get('feature_id'):
            feature_data['feature_id'] = str(uuid.uuid4())
        feature_data['created_at'] = datetime.utcnow()
        
        if self._use_memory_storage:
            self._features[feature_data['feature_id']] = feature_data.copy()
            logger.info(f"Created feature (memory): {feature_data['feature_id']}")
            return feature_data
        
        try:
            from backend.schemas.data_preprocessing_db_models import FeatureStore
            
            with self._get_session() as session:
                if session:
                    db_feature = FeatureStore(
                        id=uuid.UUID(feature_data['feature_id']),
                        dataset_id=feature_data.get('dataset_id', ''),
                        user_id=feature_data.get('user_id', ''),
                        tenant_id=feature_data.get('tenant_id'),
                        feature_name=feature_data.get('feature_name', ''),
                        feature_type=feature_data.get('feature_type', ''),
                        description=feature_data.get('description'),
                        expression=feature_data.get('expression'),
                        source_columns=feature_data.get('source_columns', []),
                        transform_config=feature_data.get('transform_config', {}),
                        statistics=feature_data.get('statistics', {}),
                        importance_score=feature_data.get('importance_score'),
                        version=feature_data.get('version', 1),
                    )
                    session.add(db_feature)
                    session.flush()
                    logger.info(f"Created feature (db): {feature_data['feature_id']}")
                    return db_feature.to_dict()
        except Exception as e:
            logger.error(f"Failed to create feature in database: {e}")
            self._features[feature_data['feature_id']] = feature_data.copy()
        
        return feature_data
    
    def get_by_id(self, feature_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取特征
        
        Args:
            feature_id: 特征ID
            
        Returns:
            特征数据
        """
        if self._use_memory_storage:
            return self._features.get(feature_id)
        
        try:
            from backend.schemas.data_preprocessing_db_models import FeatureStore
            
            with self._get_session() as session:
                if session:
                    db_feature = session.query(FeatureStore).filter(
                        FeatureStore.id == uuid.UUID(feature_id)
                    ).first()
                    if db_feature:
                        return db_feature.to_dict()
        except Exception as e:
            logger.error(f"Failed to get feature from database: {e}")
            return self._features.get(feature_id)
        
        return None
    
    def get_by_dataset(
        self,
        dataset_id: str,
        feature_type_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """获取数据集的特征列表
        
        Args:
            dataset_id: 数据集ID
            feature_type_filter: 特征类型过滤
            
        Returns:
            特征列表
        """
        if self._use_memory_storage:
            features = []
            for feature in self._features.values():
                if feature.get('dataset_id') != dataset_id:
                    continue
                if feature_type_filter and feature.get('feature_type') not in feature_type_filter:
                    continue
                features.append(feature)
            return features
        
        try:
            from backend.schemas.data_preprocessing_db_models import FeatureStore
            
            with self._get_session() as session:
                if session:
                    query = session.query(FeatureStore).filter(
                        FeatureStore.dataset_id == dataset_id
                    )
                    if feature_type_filter:
                        query = query.filter(FeatureStore.feature_type.in_(feature_type_filter))
                    
                    features = query.all()
                    return [f.to_dict() for f in features]
        except Exception as e:
            logger.error(f"Failed to get features from database: {e}")
        
        return []
    
    def get_by_name(self, dataset_id: str, feature_name: str) -> Optional[Dict[str, Any]]:
        """根据名称获取特征
        
        Args:
            dataset_id: 数据集ID
            feature_name: 特征名称
            
        Returns:
            特征数据
        """
        if self._use_memory_storage:
            for feature in self._features.values():
                if feature.get('dataset_id') == dataset_id and feature.get('feature_name') == feature_name:
                    return feature
            return None
        
        try:
            from backend.schemas.data_preprocessing_db_models import FeatureStore
            
            with self._get_session() as session:
                if session:
                    db_feature = session.query(FeatureStore).filter(
                        FeatureStore.dataset_id == dataset_id,
                        FeatureStore.feature_name == feature_name
                    ).first()
                    if db_feature:
                        return db_feature.to_dict()
        except Exception as e:
            logger.error(f"Failed to get feature by name from database: {e}")
        
        return None
    
    def update(self, feature_id: str, update_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新特征
        
        Args:
            feature_id: 特征ID
            update_data: 更新数据
            
        Returns:
            更新后的特征数据
        """
        if self._use_memory_storage:
            if feature_id in self._features:
                self._features[feature_id].update(update_data)
                self._features[feature_id]['updated_at'] = datetime.utcnow()
                self._features[feature_id]['version'] = self._features[feature_id].get('version', 0) + 1
                logger.info(f"Updated feature (memory): {feature_id}")
                return self._features[feature_id]
            return None
        
        try:
            from backend.schemas.data_preprocessing_db_models import FeatureStore
            
            with self._get_session() as session:
                if session:
                    db_feature = session.query(FeatureStore).filter(
                        FeatureStore.id == uuid.UUID(feature_id)
                    ).first()
                    if not db_feature:
                        return None
                    
                    for key, value in update_data.items():
                        if hasattr(db_feature, key):
                            setattr(db_feature, key, value)
                    
                    db_feature.version = (db_feature.version or 0) + 1
                    session.flush()
                    logger.info(f"Updated feature (db): {feature_id}")
                    return db_feature.to_dict()
        except Exception as e:
            logger.error(f"Failed to update feature in database: {e}")
        
        return None
    
    def delete(self, feature_id: str) -> bool:
        """删除特征
        
        Args:
            feature_id: 特征ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory_storage:
            if feature_id in self._features:
                del self._features[feature_id]
                logger.info(f"Deleted feature (memory): {feature_id}")
                return True
            return False
        
        try:
            from backend.schemas.data_preprocessing_db_models import FeatureStore
            
            with self._get_session() as session:
                if session:
                    db_feature = session.query(FeatureStore).filter(
                        FeatureStore.id == uuid.UUID(feature_id)
                    ).first()
                    if db_feature:
                        session.delete(db_feature)
                        logger.info(f"Deleted feature (db): {feature_id}")
                        return True
        except Exception as e:
            logger.error(f"Failed to delete feature from database: {e}")
        
        return False


# ============================================================================
# 数据预处理仓库管理器
# ============================================================================

class DataPreprocessingRepositoryManager:
    """数据预处理仓库管理器
    
    聚合管理所有数据预处理相关的仓库。
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, use_memory_storage: bool = False):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, use_memory_storage: bool = False):
        if self._initialized:
            return
        
        self._use_memory_storage = use_memory_storage
        self.task_repo = PreprocessingTaskRepository(use_memory_storage=use_memory_storage)
        self.history_repo = PreprocessingHistoryRepository(use_memory_storage=use_memory_storage)
        self.pipeline_repo = PreprocessingPipelineRepository(use_memory_storage=use_memory_storage)
        self.feature_repo = FeatureStoreRepository(use_memory_storage=use_memory_storage)
        
        self._initialized = True
        storage_mode = "memory" if use_memory_storage else "database"
        logger.info(f"DataPreprocessingRepositoryManager initialized with {storage_mode} storage")
    
    def reset(self):
        """重置所有仓库（用于测试）"""
        self.task_repo = PreprocessingTaskRepository(use_memory_storage=self._use_memory_storage)
        self.history_repo = PreprocessingHistoryRepository(use_memory_storage=self._use_memory_storage)
        self.pipeline_repo = PreprocessingPipelineRepository(use_memory_storage=self._use_memory_storage)
        self.feature_repo = FeatureStoreRepository(use_memory_storage=self._use_memory_storage)
        logger.info("DataPreprocessingRepositoryManager reset")


# 全局仓库管理器实例
_repository_manager: Optional[DataPreprocessingRepositoryManager] = None


def get_preprocessing_repository_manager(use_memory_storage: bool = False) -> DataPreprocessingRepositoryManager:
    """获取数据预处理仓库管理器实例
    
    Args:
        use_memory_storage: 是否使用内存存储
    
    Returns:
        DataPreprocessingRepositoryManager实例
    """
    global _repository_manager
    if _repository_manager is None:
        _repository_manager = DataPreprocessingRepositoryManager(use_memory_storage=use_memory_storage)
    return _repository_manager


# ============================================================================
# 实体类别名导出（向后兼容）
# ============================================================================

# 从数据库模型导入并创建别名
try:
    from backend.schemas.data_preprocessing_db_models import (
        PreprocessingTask as PreprocessingTaskEntity,
        PreprocessingHistory as PreprocessingHistoryEntity,
        PreprocessingPipeline as PreprocessingPipelineEntity,
        FeatureStore as FeatureStoreEntity,
    )
except ImportError as e:
    # 如果数据库模型不可用，创建占位符类
    logger.warning(f"Failed to import data preprocessing db models: {e}")
    
    class PreprocessingTaskEntity:
        """占位符类"""
        pass
    
    class PreprocessingHistoryEntity:
        """占位符类"""
        pass
    
    class PreprocessingPipelineEntity:
        """占位符类"""
        pass
    
    class FeatureStoreEntity:
        """占位符类"""
        pass


# 导出所有公开接口
__all__ = [
    # 仓库类
    'PreprocessingTaskRepository',
    'PreprocessingHistoryRepository',
    'PreprocessingPipelineRepository',
    'FeatureStoreRepository',
    'DataPreprocessingRepositoryManager',
    # 工厂函数
    'get_preprocessing_repository_manager',
    # 实体类别名
    'PreprocessingTaskEntity',
    'PreprocessingHistoryEntity',
    'PreprocessingPipelineEntity',
    'FeatureStoreEntity',
]

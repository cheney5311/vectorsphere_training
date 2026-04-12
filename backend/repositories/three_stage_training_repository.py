"""三阶段训练仓库

提供三阶段训练会话和进度的数据访问层操作。
支持租户级别的数据隔离。
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ThreeStageSessionRepository:
    """三阶段训练会话仓库
    
    支持内存存储和数据库存储两种模式。
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化仓库
        
        Args:
            use_memory_storage: 是否使用内存存储（测试用）
        """
        self._use_memory = use_memory_storage
        self._memory_storage: Dict[str, Dict[str, Any]] = {}
        self._db_session = None
        
        if not use_memory_storage:
            self._init_db_session()
    
    def _init_db_session(self):
        """初始化数据库会话"""
        try:
            from backend.modules.database.manager import get_database_manager
            self._db_manager = get_database_manager()
        except Exception as e:
            logger.warning(f"Failed to initialize database session: {e}")
            self._db_manager = None
    
    def _get_db_session(self):
        """获取数据库会话"""
        if self._db_manager:
            return self._db_manager.get_session()
        return None
    
    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """创建三阶段训练会话
        
        Args:
            data: 会话数据
            
        Returns:
            创建的会话信息
        """
        session_id = data.get('session_id') or f"tss_{uuid.uuid4().hex[:8]}"
        now = datetime.utcnow()
        
        session_data = {
            'session_id': session_id,
            'tenant_id': data.get('tenant_id'),
            'user_id': data.get('user_id'),
            'name': data.get('name', ''),
            'description': data.get('description', ''),
            'model_name': data.get('model_name', ''),
            'status': data.get('status', 'pending'),
            'config': data.get('config', {}),
            'result': data.get('result'),
            'error_message': data.get('error_message'),
            'progress': data.get('progress', 0.0),
            'current_stage': data.get('current_stage'),
            'pretrain_progress': data.get('pretrain_progress', 0.0),
            'finetune_progress': data.get('finetune_progress', 0.0),
            'preference_progress': data.get('preference_progress', 0.0),
            'started_at': data.get('started_at'),
            'completed_at': data.get('completed_at'),
            'created_at': now,
            'updated_at': now
        }
        
        if self._use_memory:
            self._memory_storage[session_id] = session_data
            return session_data
        
        # 数据库存储
        try:
            from backend.schemas.training_models import ThreeStageSession
            
            db_ctx = self._get_db_session()
            if db_ctx:
                with db_ctx as db:
                    session = ThreeStageSession(
                        tenant_id=uuid.UUID(data['tenant_id']) if data.get('tenant_id') else None,
                        user_id=data.get('user_id'),
                        name=data.get('name', ''),
                        description=data.get('description'),
                        model_name=data.get('model_name', ''),
                        status=data.get('status', 'pending'),
                        config=data.get('config', {}),
                        result=data.get('result'),
                        error_message=data.get('error_message'),
                        progress=data.get('progress', 0.0),
                        current_stage=data.get('current_stage'),
                        pretrain_progress=data.get('pretrain_progress', 0.0),
                        finetune_progress=data.get('finetune_progress', 0.0),
                        preference_progress=data.get('preference_progress', 0.0),
                        started_at=data.get('started_at'),
                        completed_at=data.get('completed_at')
                    )
                    db.add(session)
                    db.commit()
                    db.refresh(session)
                    session_data['session_id'] = str(session.id)
                    session_data['id'] = str(session.id)
                    return session_data
        except Exception as e:
            logger.error(f"Failed to create session in database: {e}")
        
        # 回退到内存存储
        self._memory_storage[session_id] = session_data
        return session_data
    
    def get_by_id(self, session_id: str, tenant_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """根据ID获取会话
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID（可选，用于租户隔离）
            
        Returns:
            会话信息或None
        """
        if self._use_memory:
            session = self._memory_storage.get(session_id)
            if session and (not tenant_id or session.get('tenant_id') == tenant_id):
                return session
            return None
        
        try:
            from backend.schemas.training_models import ThreeStageSession
            
            db_ctx = self._get_db_session()
            if db_ctx:
                with db_ctx as db:
                    query = db.query(ThreeStageSession).filter(
                        ThreeStageSession.id == uuid.UUID(session_id)
                    )
                    if tenant_id:
                        query = query.filter(ThreeStageSession.tenant_id == uuid.UUID(tenant_id))
                    
                    session = query.first()
                    if session:
                        return self._session_to_dict(session)
        except Exception as e:
            logger.error(f"Failed to get session from database: {e}")
        
        return self._memory_storage.get(session_id)
    
    def get_by_user(self, user_id: str, tenant_id: Optional[str] = None,
                    status: Optional[str] = None,
                    limit: int = 50, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """获取用户的会话列表
        
        Args:
            user_id: 用户ID
            tenant_id: 租户ID（可选）
            status: 状态过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            (会话列表, 总数)
        """
        if self._use_memory:
            sessions = [s for s in self._memory_storage.values() 
                       if s.get('user_id') == user_id]
            if tenant_id:
                sessions = [s for s in sessions if s.get('tenant_id') == tenant_id]
            if status:
                sessions = [s for s in sessions if s.get('status') == status]
            
            sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(sessions)
            return sessions[offset:offset + limit], total
        
        try:
            from backend.schemas.training_models import ThreeStageSession
            
            db_ctx = self._get_db_session()
            if db_ctx:
                with db_ctx as db:
                    query = db.query(ThreeStageSession).filter(
                        ThreeStageSession.user_id == user_id
                    )
                    if tenant_id:
                        query = query.filter(ThreeStageSession.tenant_id == uuid.UUID(tenant_id))
                    if status:
                        query = query.filter(ThreeStageSession.status == status)
                    
                    total = query.count()
                    sessions = query.order_by(ThreeStageSession.created_at.desc()).offset(offset).limit(limit).all()
                    
                    return [self._session_to_dict(s) for s in sessions], total
        except Exception as e:
            logger.error(f"Failed to get sessions from database: {e}")
        
        return [], 0
    
    def list_by_tenant(self, tenant_id: str, status: Optional[str] = None,
                       model_name: Optional[str] = None,
                       limit: int = 50, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """获取租户的会话列表
        
        Args:
            tenant_id: 租户ID
            status: 状态过滤
            model_name: 模型名称过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            (会话列表, 总数)
        """
        if self._use_memory:
            sessions = [s for s in self._memory_storage.values() 
                       if s.get('tenant_id') == tenant_id]
            if status:
                sessions = [s for s in sessions if s.get('status') == status]
            if model_name:
                sessions = [s for s in sessions if s.get('model_name') == model_name]
            
            sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(sessions)
            return sessions[offset:offset + limit], total
        
        try:
            from backend.schemas.training_models import ThreeStageSession
            
            db_ctx = self._get_db_session()
            if db_ctx:
                with db_ctx as db:
                    query = db.query(ThreeStageSession).filter(
                        ThreeStageSession.tenant_id == uuid.UUID(tenant_id)
                    )
                    if status:
                        query = query.filter(ThreeStageSession.status == status)
                    if model_name:
                        query = query.filter(ThreeStageSession.model_name == model_name)
                    
                    total = query.count()
                    sessions = query.order_by(ThreeStageSession.created_at.desc()).offset(offset).limit(limit).all()
                    
                    return [self._session_to_dict(s) for s in sessions], total
        except Exception as e:
            logger.error(f"Failed to list sessions from database: {e}")
        
        return [], 0
    
    def update(self, session_id: str, tenant_id: Optional[str], 
               updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """更新会话
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID（用于租户隔离）
            updates: 更新内容
            
        Returns:
            更新后的会话信息或None
        """
        updates['updated_at'] = datetime.utcnow()
        
        if self._use_memory:
            if session_id in self._memory_storage:
                session = self._memory_storage[session_id]
                if tenant_id and session.get('tenant_id') != tenant_id:
                    return None
                session.update(updates)
                return session
            return None
        
        try:
            from backend.schemas.training_models import ThreeStageSession
            
            db_ctx = self._get_db_session()
            if db_ctx:
                with db_ctx as db:
                    query = db.query(ThreeStageSession).filter(
                        ThreeStageSession.id == uuid.UUID(session_id)
                    )
                    if tenant_id:
                        query = query.filter(ThreeStageSession.tenant_id == uuid.UUID(tenant_id))
                    
                    session = query.first()
                    if session:
                        for key, value in updates.items():
                            if hasattr(session, key):
                                setattr(session, key, value)
                        db.commit()
                        db.refresh(session)
                        return self._session_to_dict(session)
        except Exception as e:
            logger.error(f"Failed to update session in database: {e}")
        
        return None
    
    def update_status(self, session_id: str, tenant_id: Optional[str],
                      status: str, **kwargs) -> bool:
        """更新会话状态
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            status: 新状态
            **kwargs: 其他更新字段
            
        Returns:
            是否更新成功
        """
        updates = {'status': status, **kwargs}
        result = self.update(session_id, tenant_id, updates)
        return result is not None
    
    def update_progress(self, session_id: str, tenant_id: Optional[str],
                        progress: float, current_stage: Optional[str] = None,
                        stage_progress: Optional[Dict[str, float]] = None) -> bool:
        """更新会话进度
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            progress: 总体进度
            current_stage: 当前阶段
            stage_progress: 阶段进度 {pretrain: 0.5, finetune: 0.0, preference: 0.0}
            
        Returns:
            是否更新成功
        """
        updates = {'progress': progress}
        if current_stage:
            updates['current_stage'] = current_stage
        if stage_progress:
            if 'pretrain' in stage_progress:
                updates['pretrain_progress'] = stage_progress['pretrain']
            if 'finetune' in stage_progress:
                updates['finetune_progress'] = stage_progress['finetune']
            if 'preference' in stage_progress:
                updates['preference_progress'] = stage_progress['preference']
        
        result = self.update(session_id, tenant_id, updates)
        return result is not None
    
    def delete(self, session_id: str, tenant_id: Optional[str]) -> bool:
        """删除会话
        
        Args:
            session_id: 会话ID
            tenant_id: 租户ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory:
            if session_id in self._memory_storage:
                session = self._memory_storage[session_id]
                if tenant_id and session.get('tenant_id') != tenant_id:
                    return False
                del self._memory_storage[session_id]
                return True
            return False
        
        try:
            from backend.schemas.training_models import ThreeStageSession
            
            db_ctx = self._get_db_session()
            if db_ctx:
                with db_ctx as db:
                    query = db.query(ThreeStageSession).filter(
                        ThreeStageSession.id == uuid.UUID(session_id)
                    )
                    if tenant_id:
                        query = query.filter(ThreeStageSession.tenant_id == uuid.UUID(tenant_id))
                    
                    session = query.first()
                    if session:
                        db.delete(session)
                        db.commit()
                        return True
        except Exception as e:
            logger.error(f"Failed to delete session from database: {e}")
        
        return False
    
    def get_statistics(self, tenant_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取统计信息
        
        Args:
            tenant_id: 租户ID
            user_id: 用户ID（可选）
            
        Returns:
            统计信息
        """
        stats = {
            'total': 0,
            'by_status': {},
            'by_model': {},
            'running_count': 0,
            'completed_count': 0,
            'failed_count': 0
        }
        
        if self._use_memory:
            sessions = [s for s in self._memory_storage.values() 
                       if s.get('tenant_id') == tenant_id]
            if user_id:
                sessions = [s for s in sessions if s.get('user_id') == user_id]
            
            stats['total'] = len(sessions)
            for s in sessions:
                status = s.get('status', 'unknown')
                stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
                model = s.get('model_name', 'unknown')
                stats['by_model'][model] = stats['by_model'].get(model, 0) + 1
                
                if status == 'running':
                    stats['running_count'] += 1
                elif status == 'completed':
                    stats['completed_count'] += 1
                elif status in ['failed', 'error']:
                    stats['failed_count'] += 1
            
            return stats
        
        try:
            from backend.schemas.training_models import ThreeStageSession
            from sqlalchemy import func
            
            db_ctx = self._get_db_session()
            if db_ctx:
                with db_ctx as db:
                    query = db.query(ThreeStageSession).filter(
                        ThreeStageSession.tenant_id == uuid.UUID(tenant_id)
                    )
                    if user_id:
                        query = query.filter(ThreeStageSession.user_id == user_id)
                    
                    stats['total'] = query.count()
                    
                    # 按状态统计
                    status_counts = db.query(
                        ThreeStageSession.status,
                        func.count(ThreeStageSession.id)
                    ).filter(
                        ThreeStageSession.tenant_id == uuid.UUID(tenant_id)
                    )
                    if user_id:
                        status_counts = status_counts.filter(ThreeStageSession.user_id == user_id)
                    status_counts = status_counts.group_by(ThreeStageSession.status).all()
                    
                    for status, count in status_counts:
                        stats['by_status'][status] = count
                        if status == 'running':
                            stats['running_count'] = count
                        elif status == 'completed':
                            stats['completed_count'] = count
                        elif status in ['failed', 'error']:
                            stats['failed_count'] = count
                    
                    # 按模型统计
                    model_counts = db.query(
                        ThreeStageSession.model_name,
                        func.count(ThreeStageSession.id)
                    ).filter(
                        ThreeStageSession.tenant_id == uuid.UUID(tenant_id)
                    )
                    if user_id:
                        model_counts = model_counts.filter(ThreeStageSession.user_id == user_id)
                    model_counts = model_counts.group_by(ThreeStageSession.model_name).all()
                    
                    for model, count in model_counts:
                        stats['by_model'][model] = count
        except Exception as e:
            logger.error(f"Failed to get statistics from database: {e}")
        
        return stats
    
    def _session_to_dict(self, session) -> Dict[str, Any]:
        """将数据库模型转换为字典"""
        return {
            'session_id': str(session.id),
            'id': str(session.id),
            'tenant_id': str(session.tenant_id) if session.tenant_id else None,
            'user_id': session.user_id,
            'name': session.name,
            'description': session.description,
            'model_name': session.model_name,
            'status': session.status,
            'config': session.config,
            'result': session.result,
            'error_message': session.error_message,
            'progress': session.progress,
            'current_stage': session.current_stage,
            'pretrain_progress': session.pretrain_progress,
            'finetune_progress': session.finetune_progress,
            'preference_progress': session.preference_progress,
            'started_at': session.started_at.isoformat() if session.started_at else None,
            'completed_at': session.completed_at.isoformat() if session.completed_at else None,
            'created_at': session.created_at.isoformat() if session.created_at else None,
            'updated_at': session.updated_at.isoformat() if session.updated_at else None
        }


class ThreeStageProgressRepository:
    """三阶段训练进度仓库
    
    支持内存存储和数据库存储两种模式。
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """初始化仓库
        
        Args:
            use_memory_storage: 是否使用内存存储（测试用）
        """
        self._use_memory = use_memory_storage
        self._memory_storage: Dict[str, Dict[str, Any]] = {}
        self._db_session = None
        
        if not use_memory_storage:
            self._init_db_session()
    
    def _init_db_session(self):
        """初始化数据库会话"""
        try:
            from backend.modules.database.manager import get_database_manager
            self._db_manager = get_database_manager()
        except Exception as e:
            logger.warning(f"Failed to initialize database session: {e}")
            self._db_manager = None
    
    def _get_db_session(self):
        """获取数据库会话"""
        if self._db_manager:
            return self._db_manager.get_session()
        return None
    
    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """创建进度记录
        
        Args:
            data: 进度数据
            
        Returns:
            创建的进度信息
        """
        progress_id = data.get('progress_id') or f"tsp_{uuid.uuid4().hex[:8]}"
        now = datetime.utcnow()
        
        progress_data = {
            'progress_id': progress_id,
            'session_id': data.get('session_id'),
            'stage': data.get('stage'),
            'epoch': data.get('epoch'),
            'step': data.get('step'),
            'total_steps': data.get('total_steps'),
            'loss': data.get('loss'),
            'accuracy': data.get('accuracy'),
            'learning_rate': data.get('learning_rate'),
            'val_loss': data.get('val_loss'),
            'val_accuracy': data.get('val_accuracy'),
            'metrics': data.get('metrics', {}),
            'created_at': now,
            'updated_at': now
        }
        
        if self._use_memory:
            self._memory_storage[progress_id] = progress_data
            return progress_data
        
        # 数据库存储
        try:
            from backend.schemas.training_models import ThreeStageProgress
            
            db_ctx = self._get_db_session()
            if db_ctx:
                with db_ctx as db:
                    progress = ThreeStageProgress(
                        session_id=uuid.UUID(data['session_id']),
                        stage=data.get('stage'),
                        epoch=data.get('epoch'),
                        step=data.get('step'),
                        total_steps=data.get('total_steps'),
                        loss=data.get('loss'),
                        accuracy=data.get('accuracy'),
                        learning_rate=data.get('learning_rate'),
                        val_loss=data.get('val_loss'),
                        val_accuracy=data.get('val_accuracy'),
                        metrics=data.get('metrics', {})
                    )
                    db.add(progress)
                    db.commit()
                    db.refresh(progress)
                    progress_data['progress_id'] = str(progress.id)
                    return progress_data
        except Exception as e:
            logger.error(f"Failed to create progress in database: {e}")
        
        # 回退到内存存储
        self._memory_storage[progress_id] = progress_data
        return progress_data
    
    def get_by_session(self, session_id: str, stage: Optional[str] = None,
                       limit: int = 100, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
        """获取会话的进度记录
        
        Args:
            session_id: 会话ID
            stage: 阶段过滤
            limit: 返回数量限制
            offset: 偏移量
            
        Returns:
            (进度列表, 总数)
        """
        if self._use_memory:
            records = [p for p in self._memory_storage.values() 
                      if p.get('session_id') == session_id]
            if stage:
                records = [p for p in records if p.get('stage') == stage]
            
            records.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            total = len(records)
            return records[offset:offset + limit], total
        
        try:
            from backend.schemas.training_models import ThreeStageProgress
            
            db_ctx = self._get_db_session()
            if db_ctx:
                with db_ctx as db:
                    query = db.query(ThreeStageProgress).filter(
                        ThreeStageProgress.session_id == uuid.UUID(session_id)
                    )
                    if stage:
                        query = query.filter(ThreeStageProgress.stage == stage)
                    
                    total = query.count()
                    records = query.order_by(ThreeStageProgress.created_at.desc()).offset(offset).limit(limit).all()
                    
                    return [self._progress_to_dict(p) for p in records], total
        except Exception as e:
            logger.error(f"Failed to get progress from database: {e}")
        
        return [], 0
    
    def get_latest(self, session_id: str, stage: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """获取最新的进度记录
        
        Args:
            session_id: 会话ID
            stage: 阶段过滤
            
        Returns:
            最新进度记录或None
        """
        records, _ = self.get_by_session(session_id, stage, limit=1)
        return records[0] if records else None
    
    def get_stage_summary(self, session_id: str) -> Dict[str, Any]:
        """获取各阶段的摘要
        
        Args:
            session_id: 会话ID
            
        Returns:
            阶段摘要
        """
        summary = {}
        
        for stage in ['pretrain', 'finetune', 'preference']:
            latest = self.get_latest(session_id, stage)
            if latest:
                summary[stage] = {
                    'epoch': latest.get('epoch'),
                    'step': latest.get('step'),
                    'total_steps': latest.get('total_steps'),
                    'loss': latest.get('loss'),
                    'accuracy': latest.get('accuracy'),
                    'learning_rate': latest.get('learning_rate'),
                    'last_updated': latest.get('created_at')
                }
            else:
                summary[stage] = None
        
        return summary
    
    def delete_by_session(self, session_id: str) -> int:
        """删除会话的所有进度记录
        
        Args:
            session_id: 会话ID
            
        Returns:
            删除的记录数
        """
        if self._use_memory:
            to_delete = [k for k, v in self._memory_storage.items() 
                        if v.get('session_id') == session_id]
            for key in to_delete:
                del self._memory_storage[key]
            return len(to_delete)
        
        try:
            from backend.schemas.training_models import ThreeStageProgress
            
            db_ctx = self._get_db_session()
            if db_ctx:
                with db_ctx as db:
                    count = db.query(ThreeStageProgress).filter(
                        ThreeStageProgress.session_id == uuid.UUID(session_id)
                    ).delete()
                    db.commit()
                    return count
        except Exception as e:
            logger.error(f"Failed to delete progress from database: {e}")
        
        return 0
    
    def _progress_to_dict(self, progress) -> Dict[str, Any]:
        """将数据库模型转换为字典"""
        return {
            'progress_id': str(progress.id),
            'session_id': str(progress.session_id),
            'stage': progress.stage,
            'epoch': progress.epoch,
            'step': progress.step,
            'total_steps': progress.total_steps,
            'loss': progress.loss,
            'accuracy': progress.accuracy,
            'learning_rate': progress.learning_rate,
            'val_loss': progress.val_loss,
            'val_accuracy': progress.val_accuracy,
            'metrics': progress.metrics,
            'created_at': progress.created_at.isoformat() if progress.created_at else None,
            'updated_at': progress.updated_at.isoformat() if progress.updated_at else None
        }


# ==============================================================================
# 单例模式获取器
# ==============================================================================

_session_repository_instance: Optional[ThreeStageSessionRepository] = None
_progress_repository_instance: Optional[ThreeStageProgressRepository] = None


def get_session_repository(use_memory_storage: bool = False) -> ThreeStageSessionRepository:
    """获取会话仓库实例（单例）"""
    global _session_repository_instance
    if _session_repository_instance is None or use_memory_storage:
        _session_repository_instance = ThreeStageSessionRepository(use_memory_storage)
    return _session_repository_instance


def get_progress_repository(use_memory_storage: bool = False) -> ThreeStageProgressRepository:
    """获取进度仓库实例（单例）"""
    global _progress_repository_instance
    if _progress_repository_instance is None or use_memory_storage:
        _progress_repository_instance = ThreeStageProgressRepository(use_memory_storage)
    return _progress_repository_instance


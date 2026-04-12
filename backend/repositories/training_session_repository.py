"""训练会话仓库

提供训练会话数据访问接口。
"""

import logging
import os
import sys
from typing import Optional, List, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 修复导入错误，使用正确的模块路径
from backend.core.exceptions import ResourceNotFoundError, DatabaseError
from backend.schemas.training_models import TrainingSession
from backend.schemas.enums import TrainingStatus

logger = logging.getLogger(__name__)


class TrainingSessionRepository:
    """训练会话仓库"""
    
    def __init__(self, db_service=None, use_memory_storage=False):
        """初始化训练会话仓库
        
        Args:
            db_service: 数据库服务实例，如果为None则尝试获取默认服务
            use_memory_storage: 是否使用内存存储，默认False使用数据库
        """
        self._use_memory_storage = use_memory_storage
        
        if use_memory_storage:
            # 内存存储模式
            self._sessions: Dict[str, TrainingSession] = {}
            self._db_service = None
        else:
            # 数据库存储模式
            try:
                from backend.modules.database.manager import get_database_manager
                self._db_service = db_service or get_database_manager()
            except ImportError:
                logger.warning("无法导入数据库服务，回退到内存存储模式")
                self._use_memory_storage = True
                self._sessions: Dict[str, TrainingSession] = {}
                self._db_service = None
        
    def create(self, session: Union[TrainingSession, Dict[str, Any]]) -> TrainingSession:
        """创建训练会话
        
        Args:
            session: 训练会话对象或数据字典
            
        Returns:
            创建的训练会话对象
            
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            if self._use_memory_storage:
                # 内存存储模式
                if isinstance(session, dict):
                    # 如果是字典，创建TrainingSession对象
                    session = TrainingSession(**session)
                
                # 使用 TrainingSession.session_id 作为键
                key = getattr(session, "session_id", None)
                if not key:
                    # 兼容早期模型，回退到 id 字段
                    key = getattr(session, "id", None)
                self._sessions[key] = session
                return session
            else:
                # 数据库存储模式
                with self._db_service.get_db_session() as db_session:
                    if isinstance(session, dict):
                        training_session = TrainingSession(**session)
                    else:
                        training_session = session
                    
                    db_session.add(training_session)
                    db_session.commit()
                    # 确保时间戳字段被设置
                    if not hasattr(training_session, 'created_at') or training_session.created_at is None:
                        from datetime import datetime
                        training_session.created_at = datetime.utcnow()
                        training_session.updated_at = datetime.utcnow()
                    # 刷新对象以获取数据库生成的字段
                    db_session.refresh(training_session)
                    # 分离对象，使其可以在会话外使用
                    db_session.expunge(training_session)
                    return training_session
        except Exception as e:
            logger.error(f"创建训练会话失败: {e}")
            raise DatabaseError(f"创建训练会话失败: {str(e)}", operation="create")
            
    def get_by_id(self, session_id: str) -> Optional[TrainingSession]:
        """根据ID获取训练会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            训练会话对象，如果不存在则返回None
            
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            if self._use_memory_storage:
                # 内存存储模式
                return self._sessions.get(session_id)
            else:
                # 数据库存储模式
                with self._db_service.get_db_session() as db_session:
                    session = db_session.query(TrainingSession).filter(
                        TrainingSession.session_id == session_id
                    ).first()
                    if session:
                        # 刷新对象以确保所有关联数据都已加载
                        db_session.refresh(session)
                        # 分离对象，使其可以在会话外使用
                        db_session.expunge(session)
                    return session
        except Exception as e:
            logger.error(f"获取训练会话失败: {e}")
            raise DatabaseError(f"获取训练会话失败: {str(e)}", operation="get_by_id")
            
    def list_by_user(self, user_id: str, limit: int = 100, offset: int = 0) -> List[TrainingSession]:
        """根据用户ID获取训练会话列表
        
        Args:
            user_id: 用户ID
            limit: 限制数量
            offset: 偏移量
            
        Returns:
            训练会话列表
            
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            if self._use_memory_storage:
                # 内存存储模式
                sessions = [s for s in self._sessions.values() if s.user_id == user_id]
                # 按创建时间倒序排列，处理None值
                sessions.sort(key=lambda x: (x.created_at or datetime.min), reverse=True)
                return sessions[offset:offset + limit]
            else:
                # 数据库存储模式
                with self._db_service.get_db_session() as db_session:
                    sessions = db_session.query(TrainingSession).filter(
                        TrainingSession.user_id == user_id
                    ).order_by(TrainingSession.created_at.desc()).offset(offset).limit(limit).all()
                    # 刷新并分离所有对象，使其可以在会话外使用
                    for session in sessions:
                        db_session.refresh(session)
                        db_session.expunge(session)
                    return sessions
        except Exception as e:
            logger.error(f"获取用户训练会话列表失败: {e}")
            raise DatabaseError(message=f"获取用户训练会话列表失败: {str(e)}", operation="list_by_user")
            
    def get_by_user_id(self, user_id: str) -> List[TrainingSession]:
        """根据用户ID获取所有训练会话（兼容方法）
        
        Args:
            user_id: 用户ID
            
        Returns:
            训练会话列表
        """
        return self.list_by_user(user_id, limit=1000, offset=0)
            
    def update(self, session: TrainingSession) -> TrainingSession:
        """扩展：保存 resource_allocation 到数据库/内存记录（若存在该字段）"""
        """更新训练会话
        
        Args:
            session: 训练会话对象
            
        Returns:
            更新后的训练会话对象
            
        Raises:
            ResourceNotFoundError: 会话不存在
            DatabaseError: 数据库操作失败
        """
        try:
            if self._use_memory_storage:
                # 内存存储模式
                session_id = getattr(session, "session_id", None) or getattr(session, "id", None)
                if session_id not in self._sessions:
                    raise ResourceNotFoundError(f"训练会话不存在: {session_id}")
                
                self._sessions[session_id] = session
                return session
            else:
                # 数据库存储模式
                with self._db_service.get_db_session() as db_session:
                    existing_session = db_session.query(TrainingSession).filter(
                        TrainingSession.session_id == session.session_id
                    ).first()
                    
                    if not existing_session:
                        raise ResourceNotFoundError(f"训练会话不存在: {session.session_id}")
                    
                    # 更新字段
                    for key, value in session.__dict__.items():
                        if not key.startswith('_') and hasattr(existing_session, key):
                            setattr(existing_session, key, value)
                    
                    db_session.commit()
                    # 确保时间戳字段被设置
                    if not hasattr(existing_session, 'updated_at') or existing_session.updated_at is None:
                        from datetime import datetime
                        existing_session.updated_at = datetime.utcnow()
                    # 刷新对象以获取最新数据
                    db_session.refresh(existing_session)
                    # 分离对象，使其可以在会话外使用
                    db_session.expunge(existing_session)
                    return existing_session
        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"更新训练会话失败: {e}")
            raise DatabaseError(f"更新训练会话失败: {str(e)}", operation="update")
            
    def update_status(self, session_id: str, status: TrainingStatus) -> Optional[TrainingSession]:
        """更新训练会话状态
        
        Args:
            session_id: 会话ID
            status: 新状态
            
        Returns:
            更新后的训练会话对象，如果不存在则返回None
            
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            if self._use_memory_storage:
                # 内存存储模式
                session = self._sessions.get(session_id)
                if session:
                    session.status = status
                    return session
                return None
            else:
                # 数据库存储模式
                with self._db_service.get_db_session() as db_session:
                    session = db_session.query(TrainingSession).filter(
                        TrainingSession.session_id == session_id
                    ).first()
                    
                    if session:
                        session.status = status
                        db_session.commit()
                        # 确保时间戳字段被设置
                        if not hasattr(session, 'updated_at') or session.updated_at is None:
                            from datetime import datetime
                            session.updated_at = datetime.utcnow()
                        # 刷新对象以获取最新数据
                        db_session.refresh(session)
                        # 分离对象，使其可以在会话外使用
                        db_session.expunge(session)
                        return session
                    return None
        except Exception as e:
            logger.error(f"更新训练会话状态失败: {e}")
            raise DatabaseError(f"更新训练会话状态失败: {str(e)}", operation="update_status")
            
    def count_by_tenant(self, tenant_id: str, status: Optional[TrainingStatus] = None) -> int:
        """统计租户训练会话数量
        
        Args:
            tenant_id: 租户ID
            status: 状态过滤 (可选)
            
        Returns:
            int: 数量
        """
        try:
            if self._use_memory_storage:
                count = 0
                for s in self._sessions.values():
                    if getattr(s, 'tenant_id', None) == tenant_id:
                        if status is None or s.status == status:
                            count += 1
                return count
            else:
                with self._db_service.get_db_session() as db_session:
                    query = db_session.query(TrainingSession).filter(
                        TrainingSession.tenant_id == tenant_id
                    )
                    if status:
                        query = query.filter(TrainingSession.status == status)
                    return query.count()
        except Exception as e:
            logger.error(f"统计租户训练会话失败: {e}")
            return 0

    def suspend_active_sessions(self, tenant_id: str) -> int:
        """暂停租户所有活跃的训练会话
        
        Args:
            tenant_id: 租户ID
            
        Returns:
            int: 暂停的会话数量
        """
        try:
            if self._use_memory_storage:
                count = 0
                for s in self._sessions.values():
                    if getattr(s, 'tenant_id', None) == tenant_id and s.status == TrainingStatus.RUNNING:
                        s.status = TrainingStatus.PAUSED if hasattr(TrainingStatus, 'PAUSED') else 'paused'
                        count += 1
                return count
            else:
                with self._db_service.get_db_session() as db_session:
                    # 查找活跃会话
                    active_sessions = db_session.query(TrainingSession).filter(
                        TrainingSession.tenant_id == tenant_id,
                        TrainingSession.status == TrainingStatus.RUNNING
                    ).all()
                    
                    count = len(active_sessions)
                    paused_status = TrainingStatus.PAUSED if hasattr(TrainingStatus, 'PAUSED') else 'paused'
                    
                    for session in active_sessions:
                        session.status = paused_status
                    
                    db_session.commit()
                    return count
        except Exception as e:
            logger.error(f"暂停租户训练会话失败: {e}")
            raise DatabaseError(f"暂停租户训练会话失败: {str(e)}", operation="suspend_active_sessions")

    def delete(self, session_id: str) -> bool:
        """删除训练会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否删除成功
            
        Raises:
            DatabaseError: 数据库操作失败
        """
        try:
            if self._use_memory_storage:
                # 内存存储模式
                if session_id in self._sessions:
                    del self._sessions[session_id]
                    return True
                return False
            else:
                # 数据库存储模式
                with self._db_service.get_db_session() as db_session:
                    session = db_session.query(TrainingSession).filter(
                        TrainingSession.session_id == session_id
                    ).first()
                    
                    if session:
                        db_session.delete(session)
                        db_session.commit()
                        return True
                    return False
        except Exception as e:
            logger.error(f"删除训练会话失败: {e}")
            raise DatabaseError(f"删除训练会话失败: {str(e)}", operation="delete")


# 全局实例
_training_session_repository = None

def get_training_session_repository(use_memory_storage: bool = False) -> TrainingSessionRepository:
    """获取训练会话仓库实例
    
    Args:
        use_memory_storage: 是否使用内存存储，默认False使用数据库
    
    Returns:
        训练会话仓库实例
    """
    global _training_session_repository
    if _training_session_repository is None:
        _training_session_repository = TrainingSessionRepository(use_memory_storage=use_memory_storage)
    return _training_session_repository
# -*- coding: utf-8 -*-
"""训练历史数据访问层

提供训练历史记录的数据库访问功能。

支持的操作：
- 分页查询训练会话
- 根据ID获取训练会话
- 创建训练会话
- 更新训练会话
- 删除训练会话
- 获取用户训练统计

架构：
Service层 -> Repository层（本模块）-> Database
"""

import logging
import uuid
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# 导入异常类
try:
    from backend.modules.training.exceptions import ValidationError, BusinessLogicError
except ImportError:
    try:
        from backend.core.exceptions import ValidationError, BusinessLogicError
    except ImportError:
        class ValidationError(Exception):
            pass
        class BusinessLogicError(Exception):
            def __init__(self, message: str, operation: str = None):
                self.message = message
                self.operation = operation
                super().__init__(message)

# 导入模型
try:
    from backend.schemas.training_models import TrainingSession
except ImportError:
    TrainingSession = None

# 导入数据库管理器
try:
    from backend.modules.database.manager import get_database_manager
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    def get_database_manager():
        return None


class TrainingHistoryRepository:
    """
    训练历史数据访问层
    
    提供对 TrainingSession 表的 CRUD 操作，支持：
    - 分页和多条件筛选查询
    - 统计聚合查询
    - 事务性写入操作
    """
    
    def __init__(self, use_memory_storage: bool = False):
        """
        初始化仓库
        
        Args:
            use_memory_storage: 是否使用内存存储（用于测试）
        """
        self._use_memory_storage = use_memory_storage
        self._memory_store: Dict[str, Any] = {}
        
        if not use_memory_storage and DB_AVAILABLE:
            self._db_manager = get_database_manager()
        else:
            self._db_manager = None
    
    def get_training_sessions_paginated(
        self, 
        user_id: str, 
        page: int = 1, 
        limit: int = 10, 
        status: Optional[str] = None,
        training_type: Optional[str] = None,
        model_name: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[List[Any], int]:
        """分页获取训练会话
        
        Args:
            user_id: 用户ID
            page: 页码
            limit: 每页数量
            status: 状态过滤
            training_type: 训练类型过滤
            model_name: 模型名称搜索
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            Tuple[List, int]: (训练会话列表, 总数)
        """
        # 限制参数范围
        page = max(page, 1)
        limit = min(max(limit, 1), 50)
        offset = (page - 1) * limit
        
        # 内存存储模式
        if self._use_memory_storage or self._db_manager is None:
            return self._get_from_memory(
                user_id, page, limit, status, training_type, model_name, start_date, end_date
            )
        
        try:
            with self._db_manager.get_db_session() as db:
                # 构建查询
                query = db.query(TrainingSession).filter(
                    TrainingSession.user_id == user_id
                )
                
                # 根据状态过滤
                if status:
                    query = query.filter(TrainingSession.status == status)
                
                # 根据训练类型过滤
                if training_type:
                    query = query.filter(TrainingSession.training_type == training_type)
                
                # 根据模型名称搜索
                if model_name:
                    query = query.filter(TrainingSession.model_id.contains(model_name))
                
                # 根据日期范围过滤
                if start_date:
                    try:
                        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                        query = query.filter(TrainingSession.created_at >= start_dt)
                    except ValueError:
                        pass  # 忽略无效日期格式
                
                if end_date:
                    try:
                        end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
                        query = query.filter(TrainingSession.created_at < end_dt)
                    except ValueError:
                        pass  # 忽略无效日期格式
                
                # 获取总数
                total = query.count()
                
                # 分页查询
                sessions = query.order_by(TrainingSession.created_at.desc()).offset(offset).limit(limit).all()
                
                return sessions, total
                
        except Exception as e:
            logger.error(f"获取训练历史记录失败: {e}")
            raise BusinessLogicError(f"获取训练历史记录失败: {e}")
    
    def _get_from_memory(
        self,
        user_id: str,
        page: int,
        limit: int,
        status: Optional[str],
        training_type: Optional[str],
        model_name: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Tuple[List[Any], int]:
        """从内存存储获取数据"""
        # 过滤用户的会话
        user_sessions = [
            v for k, v in self._memory_store.items() 
            if k.startswith(f"{user_id}:")
        ]
        
        # 应用过滤器
        if status:
            user_sessions = [s for s in user_sessions if s.get('status') == status]
        if training_type:
            user_sessions = [s for s in user_sessions if s.get('training_type') == training_type]
        if model_name:
            user_sessions = [s for s in user_sessions if model_name.lower() in s.get('model_id', '').lower()]
        
        total = len(user_sessions)
        offset = (page - 1) * limit
        paginated = user_sessions[offset:offset + limit]
        
        # 转换为类似对象
        result = [type('MemorySession', (), s)() for s in paginated]
        return result, total
    
    def get_training_session_by_id(self, session_id: str, user_id: str) -> Optional[Any]:
        """根据ID获取训练会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            训练会话，如果不存在则返回None
        """
        if self._use_memory_storage or self._db_manager is None:
            # 内存存储
            key = f"{user_id}:{session_id}"
            if key in self._memory_store:
                return type('MemorySession', (), self._memory_store[key])()
            return None
        
        try:
            with self._db_manager.get_db_session() as db:
                session = db.query(TrainingSession).filter(
                    TrainingSession.session_id == session_id,
                    TrainingSession.user_id == user_id
                ).first()
                
                return session
                
        except Exception as e:
            logger.error(f"获取训练记录详情失败: {e}")
            raise BusinessLogicError(f"获取训练记录详情失败: {e}")
    
    def create_training_session(self, **kwargs) -> Any:
        """创建训练会话
        
        Args:
            **kwargs: 训练会话参数
            
        Returns:
            创建的训练会话
        """
        if self._use_memory_storage or self._db_manager is None:
            # 内存存储
            session_id = kwargs.get('session_id', str(uuid.uuid4()))
            user_id = kwargs.get('user_id', 'anonymous')
            
            key = f"{user_id}:{session_id}"
            kwargs['session_id'] = session_id
            kwargs['created_at'] = datetime.utcnow().isoformat()
            kwargs['updated_at'] = kwargs['created_at']
            
            self._memory_store[key] = kwargs
            
            # 返回类似对象的字典
            return type('MemorySession', (), kwargs)()
        
        try:
            with self._db_manager.get_db_session() as db:
                session = TrainingSession(**kwargs)
                db.add(session)
                db.commit()
                db.refresh(session)
                
                return session
                
        except Exception as e:
            logger.error(f"创建训练会话失败: {e}")
            raise BusinessLogicError(f"创建训练会话失败: {e}")
    
    def delete_training_session(self, session_id: str, user_id: str) -> bool:
        """删除训练会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            
        Returns:
            是否删除成功
        """
        if self._use_memory_storage or self._db_manager is None:
            # 内存存储
            key = f"{user_id}:{session_id}"
            if key in self._memory_store:
                del self._memory_store[key]
                return True
            return False
        
        try:
            with self._db_manager.get_db_session() as db:
                session = db.query(TrainingSession).filter(
                    TrainingSession.session_id == session_id,
                    TrainingSession.user_id == user_id
                ).first()
                
                if not session:
                    return False
                
                db.delete(session)
                db.commit()
                
                return True
                
        except Exception as e:
            logger.error(f"删除训练记录失败: {e}")
            raise BusinessLogicError(f"删除训练记录失败: {e}")
    
    def update_training_session(
        self, 
        session_id: str, 
        user_id: str,
        **update_fields
    ) -> Optional[Any]:
        """更新训练会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            **update_fields: 要更新的字段
            
        Returns:
            更新后的训练会话，如果不存在则返回None
        """
        if self._use_memory_storage or self._db_manager is None:
            # 内存存储
            key = f"{user_id}:{session_id}"
            if key in self._memory_store:
                self._memory_store[key].update(update_fields)
                self._memory_store[key]['updated_at'] = datetime.utcnow().isoformat()
                return self._memory_store[key]
            return None
        
        try:
            with self._db_manager.get_db_session() as db:
                session = db.query(TrainingSession).filter(
                    TrainingSession.session_id == session_id,
                    TrainingSession.user_id == user_id
                ).first()
                
                if not session:
                    return None
                
                # 更新字段
                for field, value in update_fields.items():
                    if hasattr(session, field):
                        setattr(session, field, value)
                
                db.commit()
                db.refresh(session)
                
                return session
                
        except Exception as e:
            logger.error(f"更新训练会话失败: {e}")
            raise BusinessLogicError(f"更新训练会话失败: {e}")
    
    def update_session_status(
        self, 
        session_id: str, 
        user_id: str, 
        status: str,
        error_message: str = None
    ) -> bool:
        """更新训练会话状态
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            status: 新状态
            error_message: 错误信息（可选）
            
        Returns:
            是否更新成功
        """
        update_fields = {'status': status}
        
        if status == 'running':
            update_fields['started_at'] = datetime.utcnow()
        elif status in ['completed', 'failed', 'cancelled']:
            update_fields['completed_at'] = datetime.utcnow()
        
        if error_message:
            update_fields['error_message'] = error_message
        
        result = self.update_training_session(session_id, user_id, **update_fields)
        return result is not None
    
    def update_session_result(
        self, 
        session_id: str, 
        user_id: str, 
        result: Dict[str, Any],
        status: str = 'completed'
    ) -> bool:
        """更新训练会话结果
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            result: 训练结果
            status: 状态（默认completed）
            
        Returns:
            是否更新成功
        """
        update_fields = {
            'result': result,
            'status': status,
            'completed_at': datetime.utcnow()
        }
        
        result_obj = self.update_training_session(session_id, user_id, **update_fields)
        return result_obj is not None
    
    def update_session_progress(
        self, 
        session_id: str, 
        user_id: str, 
        progress: float
    ) -> bool:
        """更新训练进度
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            progress: 进度 (0.0-100.0)
            
        Returns:
            是否更新成功
        """
        result = self.update_training_session(session_id, user_id, progress=progress)
        return result is not None
    
    def get_metrics_by_session(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取会话的训练指标历史
        
        Args:
            session_id: 会话ID
            limit: 限制数量
            
        Returns:
            指标历史列表
        """
        if self._use_memory_storage or self._db_manager is None:
            # 内存存储不支持历史指标查询，返回空
            return []
            
        try:
            # 这里应该从 TrainingProgress 表查询
            from backend.schemas.training_models import TrainingProgress
            from sqlalchemy import desc
            
            with self._db_manager.get_db_session() as db:
                query = db.query(TrainingProgress).filter(
                    TrainingProgress.session_id == session_id
                ).order_by(desc(TrainingProgress.created_at)).limit(limit)
                
                records = query.all()
                metrics_list = []
                
                for r in records:
                    if r.metrics:
                        metric_data = dict(r.metrics)
                        metric_data['epoch'] = r.epoch
                        metric_data['step'] = r.step
                        metric_data['timestamp'] = r.created_at.isoformat() if r.created_at else None
                        metrics_list.append(metric_data)
                        
                # 按时间正序排列
                return sorted(metrics_list, key=lambda x: x.get('timestamp') or '')
                
        except Exception as e:
            # 避免循环导入 logging
            import logging
            logging.getLogger(__name__).error(f"Failed to get metrics history: {e}")
            return []

    def get_user_training_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户训练统计信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            训练统计信息
        """
        if self._use_memory_storage or self._db_manager is None:
            # 内存存储统计
            return self._calculate_memory_statistics(user_id)
        
        try:
            with self._db_manager.get_db_session() as db:
                # 查询用户的所有训练会话
                sessions = db.query(TrainingSession).filter(
                    TrainingSession.user_id == user_id
                ).all()
                
                # 计算统计信息
                total_trainings = len(sessions)
                completed_trainings = len([s for s in sessions if getattr(s, 'status', None) == 'completed'])
                failed_trainings = len([s for s in sessions if getattr(s, 'status', None) == 'failed'])
                cancelled_trainings = len([s for s in sessions if getattr(s, 'status', None) == 'cancelled'])
                
                # 计算平均训练时间
                total_duration = 0
                completed_sessions = [s for s in sessions if getattr(s, 'status', None) == 'completed' and getattr(s, 'started_at', None) is not None and getattr(s, 'completed_at', None) is not None]
                for session in completed_sessions:
                    duration = (session.completed_at - session.started_at).total_seconds()
                    total_duration += duration
                
                average_training_time = (total_duration / max(len(completed_sessions), 1)) / 60  # 转换为分钟
                
                # 计算平均准确率和损失
                total_accuracy = 0
                total_loss = 0
                completed_with_result = [s for s in completed_sessions if getattr(s, 'result', None)]
                for session in completed_with_result:
                    result = session.result or {}
                    total_accuracy += result.get('accuracy', 0)
                    total_loss += result.get('final_loss', 0)
                
                average_accuracy = total_accuracy / max(len(completed_with_result), 1)
                average_loss = total_loss / max(len(completed_with_result), 1)
                
                # 统计最常用的模型和训练类型
                model_counts = {}
                training_type_counts = {}
                for session in sessions:
                    model_id = getattr(session, 'model_id', 'unknown')
                    training_type = getattr(session, 'training_type', 'unknown')
                    
                    model_counts[model_id] = model_counts.get(model_id, 0) + 1
                    training_type_counts[training_type] = training_type_counts.get(training_type, 0) + 1
                
                most_used_model = max(model_counts.items(), key=lambda x: x[1])[0] if model_counts else 'unknown'
                most_used_training_type = max(training_type_counts.items(), key=lambda x: x[1])[0] if training_type_counts else 'unknown'
                
                return {
                    'total_trainings': total_trainings,
                    'completed_trainings': completed_trainings,
                    'failed_trainings': failed_trainings,
                    'cancelled_trainings': cancelled_trainings,
                    'average_training_time': round(average_training_time, 2),
                    'average_accuracy': round(average_accuracy, 4),
                    'average_loss': round(average_loss, 4),
                    'most_used_model': most_used_model,
                    'most_used_training_type': most_used_training_type
                }
                
        except Exception as e:
            logger.error(f"获取训练统计信息失败: {e}")
            raise BusinessLogicError(f"获取训练统计信息失败: {e}")
    
    def _calculate_memory_statistics(self, user_id: str) -> Dict[str, Any]:
        """计算内存存储的统计信息"""
        user_sessions = [
            v for k, v in self._memory_store.items() 
            if k.startswith(f"{user_id}:")
        ]
        
        total = len(user_sessions)
        completed = len([s for s in user_sessions if s.get('status') == 'completed'])
        failed = len([s for s in user_sessions if s.get('status') == 'failed'])
        cancelled = len([s for s in user_sessions if s.get('status') == 'cancelled'])
        
        return {
            'total_trainings': total,
            'completed_trainings': completed,
            'failed_trainings': failed,
            'cancelled_trainings': cancelled,
            'average_training_time': 0.0,
            'average_accuracy': 0.0,
            'average_loss': 0.0,
            'most_used_model': 'unknown',
            'most_used_training_type': 'unknown'
        }


# 全局训练历史仓库实例
_global_training_history_repository: Optional[TrainingHistoryRepository] = None


def get_training_history_repository(use_memory_storage: bool = False) -> TrainingHistoryRepository:
    """获取训练历史仓库实例
    
    Args:
        use_memory_storage: 是否使用内存存储
    
    Returns:
        TrainingHistoryRepository: 训练历史仓库实例
    """
    global _global_training_history_repository
    
    if _global_training_history_repository is None:
        _global_training_history_repository = TrainingHistoryRepository(use_memory_storage)
    
    return _global_training_history_repository
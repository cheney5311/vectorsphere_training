"""数据库操作服务

提供高级数据库操作功能。
"""

import logging
from typing import Type, TypeVar, List, Optional, Dict, Any

from sqlalchemy.exc import SQLAlchemyError

from backend.schemas.base_models import Base
from .manager import get_database_manager

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Base)


class DatabaseService:
    """数据库操作服务"""
    
    def __init__(self):
        self.db_manager = get_database_manager()
    
    def create(self, model: T) -> T:
        """创建记录"""
        with self.db_manager.get_db_session() as session:
            try:
                session.add(model)
                session.flush()  # 获取ID但不提交
                return model
            except SQLAlchemyError as e:
                logger.error(f"创建记录失败: {e}")
                raise
    
    def get_by_id(self, model_class: Type[T], id: str) -> Optional[T]:
        """根据ID获取记录"""
        with self.db_manager.get_db_session() as session:
            try:
                return session.query(model_class).filter(model_class.id == id).first()
            except SQLAlchemyError as e:
                logger.error(f"查询记录失败: {e}")
                raise
    
    def list_all(self, model_class: Type[T], limit: int = 100, offset: int = 0) -> List[T]:
        """获取所有记录"""
        with self.db_manager.get_db_session() as session:
            try:
                return session.query(model_class).offset(offset).limit(limit).all()
            except SQLAlchemyError as e:
                logger.error(f"查询记录列表失败: {e}")
                raise
    
    def update(self, model: T, update_data: Dict[str, Any]) -> T:
        """更新记录"""
        with self.db_manager.get_db_session() as session:
            try:
                for key, value in update_data.items():
                    if hasattr(model, key):
                        setattr(model, key, value)
                session.add(model)
                return model
            except SQLAlchemyError as e:
                logger.error(f"更新记录失败: {e}")
                raise
    
    def delete(self, model: T) -> bool:
        """删除记录"""
        with self.db_manager.get_db_session() as session:
            try:
                session.delete(model)
                return True
            except SQLAlchemyError as e:
                logger.error(f"删除记录失败: {e}")
                raise
    
    def filter_by(self, model_class: Type[T], **kwargs) -> List[T]:
        """根据条件过滤记录"""
        with self.db_manager.get_db_session() as session:
            try:
                query = session.query(model_class)
                for key, value in kwargs.items():
                    if hasattr(model_class, key):
                        query = query.filter(getattr(model_class, key) == value)
                return query.all()
            except SQLAlchemyError as e:
                logger.error(f"过滤记录失败: {e}")
                raise


# 全局数据库服务实例
_global_db_service: Optional[DatabaseService] = None


def get_database_service() -> DatabaseService:
    """获取全局数据库服务实例"""
    global _global_db_service
    if _global_db_service is None:
        _global_db_service = DatabaseService()
    return _global_db_service


def close_database_service():
    """关闭全局数据库服务"""
    global _global_db_service
    if _global_db_service:
        _global_db_service = None
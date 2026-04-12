"""核心数据库模块

提供数据库连接和会话管理功能。
"""

import logging
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from backend.schemas.base_models import Base

logger = logging.getLogger(__name__)

# 数据库引擎
_engine = None
_SessionLocal = None


def init_database(database_url: str = "sqlite:///./test.db"):
    """初始化数据库
    
    Args:
        database_url: 数据库连接URL
    """
    global _engine, _SessionLocal
    
    try:
        _engine = create_engine(
            database_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
            echo=False
        )
        
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
        
        # 创建所有表
        Base.metadata.create_all(bind=_engine)
        
        logger.info(f"数据库初始化成功: {database_url}")
        
    except Exception as e:
        logger.error(f"数据库初始化失败: {e}")
        raise


def get_engine():
    """获取数据库引擎
    
    Returns:
        Engine: SQLAlchemy引擎实例
    """
    global _engine
    if _engine is None:
        init_database()
    return _engine


def get_session_local():
    """获取会话工厂
    
    Returns:
        sessionmaker: SQLAlchemy会话工厂
    """
    global _SessionLocal
    if _SessionLocal is None:
        init_database()
    return _SessionLocal


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """获取数据库会话上下文管理器
    
    Yields:
        Session: SQLAlchemy会话实例
    """
    session_local = get_session_local()
    session = session_local()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"数据库会话错误: {e}")
        raise
    finally:
        session.close()


def create_session() -> Session:
    """创建新的数据库会话
    
    Returns:
        Session: SQLAlchemy会话实例
    """
    session_local = get_session_local()
    return session_local()


def close_database():
    """关闭数据库连接"""
    global _engine, _SessionLocal
    
    if _engine:
        _engine.dispose()
        _engine = None
        
    _SessionLocal = None
    logger.info("数据库连接已关闭")
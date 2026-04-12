"""数据库连接池管理器

优化数据库连接性能和资源利用，支持多种数据库类型。
"""

import logging
import threading
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from backend.modules.performance.performance_errors import DatabasePoolError

logger = logging.getLogger(__name__)


class DatabasePoolManager:
    """数据库连接池管理器"""

    def __init__(self):
        self.engine = None
        self.session_factory = None
        self.pool_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'idle_connections': 0,
            'pool_size': 0,
            'max_overflow': 0,
            'checked_out': 0,
            'checked_in': 0,
            'pool_hits': 0,
            'pool_misses': 0
        }
        self._lock = threading.Lock()
        self._initialized = False

    def init_app(self, app):
        """初始化数据库连接池"""
        try:
            # 数据库配置
            database_url = app.config.get('DATABASE_URL', 'sqlite:///vectorsphere.db')

            # 连接池配置 - 针对SQLite优化
            if database_url.startswith('sqlite'):
                # SQLite特定配置
                pool_config = {
                    'poolclass': pool.StaticPool,
                    'pool_size': 1,  # SQLite建议单连接
                    'max_overflow': 0,
                    'pool_timeout': app.config.get('DB_POOL_TIMEOUT', 30),
                    'pool_recycle': -1,  # SQLite不需要回收连接
                    'pool_pre_ping': False,  # SQLite不需要预检
                    'echo': app.config.get('DB_ECHO', False),
                    'connect_args': {'check_same_thread': False}  # 允许多线程访问
                }
            else:
                # PostgreSQL/MySQL配置
                pool_config = {
                    'poolclass': pool.QueuePool,
                    'pool_size': app.config.get('DB_POOL_SIZE', 10),
                    'max_overflow': app.config.get('DB_MAX_OVERFLOW', 20),
                    'pool_timeout': app.config.get('DB_POOL_TIMEOUT', 30),
                    'pool_recycle': app.config.get('DB_POOL_RECYCLE', 3600),
                    'pool_pre_ping': True,
                    'echo': app.config.get('DB_ECHO', False)
                }

            # 创建引擎
            self.engine = create_engine(database_url, **pool_config)

            # 创建会话工厂
            self.session_factory = sessionmaker(bind=self.engine)

            # 更新统计信息
            with self._lock:
                self.pool_stats.update({
                    'pool_size': pool_config['pool_size'],
                    'max_overflow': pool_config['max_overflow']
                })

            self._initialized = True
            logger.info("Database connection pool initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize database pool: {str(e)}")
            raise DatabasePoolError(f"Failed to initialize database pool: {str(e)}")

    @contextmanager
    def get_session(self):
        """获取数据库会话上下文管理器"""
        if not self._initialized:
            raise DatabasePoolError("Database pool not initialized")

        session = None
        try:
            session = self.session_factory()
            self._update_stats('checked_out')
            yield session
            session.commit()

        except SQLAlchemyError as e:
            if session:
                session.rollback()
            logger.error(f"Database session error: {str(e)}")
            raise DatabasePoolError(f"Database session error: {str(e)}")

        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Unexpected error in database session: {str(e)}")
            raise DatabasePoolError(f"Unexpected error in database session: {str(e)}")

        finally:
            if session:
                session.close()
                self._update_stats('checked_in')

    def get_session_direct(self) -> Session:
        """直接获取数据库会话（需要手动管理）"""
        if not self._initialized:
            raise DatabasePoolError("Database pool not initialized")

        session = self.session_factory()
        self._update_stats('checked_out')
        return session

    def close_session(self, session: Session):
        """关闭数据库会话"""
        try:
            if session:
                session.close()
                self._update_stats('checked_in')
        except Exception as e:
            logger.error(f"Error closing session: {str(e)}")
            raise DatabasePoolError(f"Error closing session: {str(e)}")

    def _update_stats(self, operation: str):
        """更新连接池统计信息"""
        with self._lock:
            if operation == 'checked_out':
                self.pool_stats['checked_out'] += 1
                self.pool_stats['active_connections'] += 1
            elif operation == 'checked_in':
                self.pool_stats['checked_in'] += 1
                self.pool_stats['active_connections'] = max(0, self.pool_stats['active_connections'] - 1)

    def get_pool_status(self) -> Dict[str, Any]:
        """获取连接池状态"""
        if not self._initialized or not self.engine:
            return {'status': 'not_initialized'}

        try:
            pool = self.engine.pool

            with self._lock:
                status = {
                    'status': 'healthy',
                    'pool_size': pool.size(),
                    'checked_out_connections': pool.checkedout(),
                    'overflow_connections': pool.overflow(),
                    'checked_in_connections': pool.checkedin(),
                    'total_connections': pool.size() + pool.overflow(),
                    'statistics': self.pool_stats.copy(),
                    'timestamp': time.time()
                }

            return status

        except Exception as e:
            logger.error(f"Error getting pool status: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

    def health_check(self) -> bool:
        """健康检查"""
        try:
            if not self._initialized or not self.engine:
                return False

            # 测试连接
            with self.get_session() as session:
                session.execute("SELECT 1")

            return True

        except Exception as e:
            logger.error(f"Database health check failed: {str(e)}")
            return False

    def optimize_pool(self):
        """优化连接池配置"""
        try:
            if not self._initialized or not self.engine:
                return

            pool = self.engine.pool
            status = self.get_pool_status()

            # 记录优化建议
            suggestions = []

            # 检查连接池使用率
            if status.get('checked_out_connections', 0) > status.get('pool_size', 0) * 0.8:
                suggestions.append("Consider increasing pool_size")

            # 检查溢出连接
            if status.get('overflow_connections', 0) > 0:
                suggestions.append("Pool overflow detected, consider increasing pool_size")

            # 记录建议
            if suggestions:
                logger.info(f"Pool optimization suggestions: {', '.join(suggestions)}")

            return {
                'status': status,
                'suggestions': suggestions
            }

        except Exception as e:
            logger.error(f"Error optimizing pool: {str(e)}")
            return {'error': str(e)}

    def close_pool(self):
        """关闭连接池"""
        try:
            if self.engine:
                self.engine.dispose()
                logger.info("Database connection pool closed")
        except Exception as e:
            logger.error(f"Error closing database pool: {str(e)}")
        finally:
            self._initialized = False


# 全局连接池管理器实例
_global_db_pool_manager: Optional[DatabasePoolManager] = None


def get_database_pool_manager() -> DatabasePoolManager:
    """获取全局数据库连接池管理器实例

    Returns:
        DatabasePoolManager: 数据库连接池管理器实例
    """
    global _global_db_pool_manager
    if _global_db_pool_manager is None:
        _global_db_pool_manager = DatabasePoolManager()
    return _global_db_pool_manager


def create_database_pool_manager() -> DatabasePoolManager:
    """创建数据库连接池管理器实例

    Returns:
        DatabasePoolManager: 数据库连接池管理器实例
    """
    return DatabasePoolManager()
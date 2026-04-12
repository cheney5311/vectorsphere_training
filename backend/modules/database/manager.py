"""数据库管理器

提供数据库连接管理和会话管理功能。
"""

import logging
from contextlib import contextmanager
from typing import Optional, Any, Dict

from sqlalchemy import create_engine, inspect, text, Column, Index
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.schema import CreateIndex

from backend.schemas.base_models import Base
from .config import get_database_config, DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or get_database_config()
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """初始化数据库引擎"""
        try:
            self.engine = create_engine(
                self.config.connection_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            logger.info("数据库引擎初始化成功")
        except Exception as e:
            logger.error(f"数据库引擎初始化失败: {e}")
            raise
    
    def get_session(self) -> Session:
        """获取数据库会话"""
        if not self.SessionLocal:
            raise RuntimeError("数据库引擎未初始化")
        return self.SessionLocal()
    
    @contextmanager
    def get_db_session(self):
        """获取数据库会话上下文管理器"""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"数据库会话错误: {e}")
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """智能创建所有表 - 支持增量更新"""
        if not self.engine:
            raise RuntimeError("数据库引擎未初始化")
        
        try:
            self._create_tables_incrementally()
            logger.info("数据库表创建/更新成功")
        except Exception as e:
            logger.error(f"数据库表创建/更新失败: {e}")
            raise
    
    def _create_tables_incrementally(self):
        """增量创建表和索引"""
        inspector = inspect(self.engine)
        existing_tables = set(inspector.get_table_names())
        
        # 获取所有需要创建的表
        tables_to_create = []
        tables_to_update = []
        
        for table in Base.metadata.tables.values():
            if table.name not in existing_tables:
                tables_to_create.append(table)
                logger.info(f"发现新表需要创建: {table.name}")
            else:
                tables_to_update.append(table)
                logger.info(f"检查现有表是否需要更新: {table.name}")
        
        # 创建新表
        if tables_to_create:
            logger.info(f"开始创建 {len(tables_to_create)} 个新表...")
            for table in tables_to_create:
                try:
                    table.create(bind=self.engine, checkfirst=True)
                    logger.info(f"✅ 成功创建表: {table.name}")
                except Exception as e:
                    logger.error(f"创建表 {table.name} 失败: {e}")
                    raise
        
        # 更新现有表（添加缺失的列和索引）
        if tables_to_update:
            logger.info(f"开始检查 {len(tables_to_update)} 个现有表的更新...")
            for table in tables_to_update:
                try:
                    self._update_table_schema(table, inspector)
                except Exception as e:
                    logger.error(f"更新表 {table.name} 失败: {e}")
                    # 不抛出异常，继续处理其他表
                    continue
    
    def _update_table_schema(self, table, inspector):
        """更新表结构 - 添加缺失的列和索引"""
        table_name = table.name
        
        # 获取现有列信息
        existing_columns = {col['name']: col for col in inspector.get_columns(table_name)}
        existing_indexes = {idx['name']: idx for idx in inspector.get_indexes(table_name)}
        
        # 检查并添加缺失的列
        columns_added = 0
        for column in table.columns:
            if column.name not in existing_columns:
                try:
                    self._add_column_safely(table_name, column)
                    columns_added += 1
                    logger.info(f"✅ 成功添加列: {table_name}.{column.name}")
                except Exception as e:
                    logger.warning(f"⚠️ 添加列 {table_name}.{column.name} 失败: {e}")
        
        # 检查并添加缺失的索引
        indexes_added = 0
        for index in table.indexes:
            if index.name not in existing_indexes:
                try:
                    self._create_index_safely(index)
                    indexes_added += 1
                    logger.info(f"✅ 成功创建索引: {index.name}")
                except Exception as e:
                    logger.warning(f"⚠️ 创建索引 {index.name} 失败: {e}")
        
        # 处理列级索引（通过 index=True 创建的索引）
        for column in table.columns:
            if column.index:
                # 检查是否存在对应的索引
                expected_index_name = f"ix_{table_name}_{column.name}"
                if expected_index_name not in existing_indexes:
                    try:
                        self._create_column_index_safely(table_name, column)
                        indexes_added += 1
                        logger.info(f"✅ 成功创建列索引: {expected_index_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ 创建列索引 {expected_index_name} 失败: {e}")
        
        if columns_added > 0 or indexes_added > 0:
            logger.info(f"✅ 表 {table_name} 更新完成: 添加了 {columns_added} 列, {indexes_added} 索引")
        else:
            logger.info(f"✅ 表 {table_name} 无需更新")
    
    def _add_column_safely(self, table_name: str, column: Column):
        """安全地添加列"""
        # 构建 ALTER TABLE 语句
        column_type = column.type.compile(self.engine.dialect)
        
        # 处理默认值
        default_clause = ""
        if column.default is not None:
            if hasattr(column.default, 'arg'):
                if isinstance(column.default.arg, str):
                    default_clause = f" DEFAULT '{column.default.arg}'"
                else:
                    default_clause = f" DEFAULT {column.default.arg}"
        
        # 处理 NOT NULL 约束
        nullable_clause = "" if column.nullable else " NOT NULL"
        
        sql = f"ALTER TABLE {table_name} ADD COLUMN {column.name} {column_type}{default_clause}{nullable_clause}"
        
        with self.engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
    
    def _create_index_safely(self, index: Index):
        """安全地创建索引"""
        try:
            with self.engine.connect() as conn:
                # 修复 dialect 访问方式，使用正确的 SQLAlchemy API
                create_index_sql = CreateIndex(index).compile(dialect=self.engine.dialect)
                conn.execute(text(str(create_index_sql)))
                conn.commit()
                logger.info(f"✅ 成功创建索引 {index.name}")
        except ProgrammingError as e:
            if "already exists" in str(e).lower():
                logger.info(f"索引 {index.name} 已存在，跳过创建")
            else:
                logger.error(f"⚠️ 创建索引 {index.name} 失败: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"⚠️ 创建索引 {index.name} 失败: {str(e)}")
            # 不抛出异常，允许应用继续运行
            pass
    
    def _create_column_index_safely(self, table_name: str, column: Column):
        """安全地为列创建索引"""
        index_name = f"ix_{table_name}_{column.name}"
        
        # 检查数据库类型
        if 'postgresql' in str(self.engine.url):
            sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column.name})"
        else:
            # SQLite 不支持 IF NOT EXISTS，需要先检查
            sql = f"CREATE INDEX {index_name} ON {table_name} ({column.name})"
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(sql))
                conn.commit()
        except ProgrammingError as e:
            if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                logger.info(f"索引 {index_name} 已存在，跳过创建")
            else:
                raise
    
    def drop_tables(self):
        """删除所有表"""
        if not self.engine:
            raise RuntimeError("数据库引擎未初始化")
        
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("数据库表删除成功")
        except Exception as e:
            logger.error(f"数据库表删除失败: {e}")
            raise
    
    def execute_raw_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """执行原始SQL"""
        with self.get_db_session() as session:
            try:
                from sqlalchemy import text
                # 使用text()包装SQL表达式
                result = session.execute(text(sql), params or {})
                return result
            except SQLAlchemyError as e:
                logger.error(f"SQL执行失败: {e}")
                raise
    
    def health_check(self) -> bool:
        """检查数据库连接健康状态
        
        Returns:
            bool: 数据库是否健康
        """
        try:
            # 使用text()包装SQL表达式
            from sqlalchemy import text
            session = self.get_session()
            result = session.execute(text('SELECT 1')).scalar()
            session.close()
            return result == 1
        except Exception as e:
            logger.error(f"数据库健康检查失败: {e}")
            return False
    
    def graceful_shutdown(self, timeout: int = 30) -> bool:
        """优雅关闭数据库管理器
        
        Args:
            timeout: 关闭超时时间（秒）
            
        Returns:
            bool: 是否成功关闭
        """
        try:
            print("开始优雅关闭数据库管理器...")
            
            # 1. 等待活跃会话完成
            active_sessions = 0
            if hasattr(self, '_session_registry'):
                active_sessions = len(self._session_registry)
                print(f"等待 {active_sessions} 个活跃会话完成...")
                
                # 等待会话自然结束，最多等待timeout/2时间
                import time
                wait_time = min(timeout // 2, 15)  # 最多等待15秒
                for i in range(wait_time):
                    if hasattr(self, '_session_registry') and len(self._session_registry) == 0:
                        break
                    time.sleep(1)
                
                remaining_sessions = len(self._session_registry) if hasattr(self, '_session_registry') else 0
                if remaining_sessions > 0:
                    print(f"仍有 {remaining_sessions} 个会话未完成，将强制关闭")
            
            # 2. 关闭连接池
            if self.engine:
                try:
                    # 获取连接池信息
                    pool = self.engine.pool
                    if pool:
                        pool_size = pool.size()
                        checked_in = pool.checkedin()
                        checked_out = pool.checkedout()
                        
                        print(f"连接池状态 - 总大小: {pool_size}, 已归还: {checked_in}, 已借出: {checked_out}")
                        
                        # 等待借出的连接归还
                        if checked_out > 0:
                            print(f"等待 {checked_out} 个借出连接归还...")
                            wait_time = min(timeout // 2, 10)  # 最多等待10秒
                            for i in range(wait_time):
                                if pool.checkedout() == 0:
                                    break
                                time.sleep(1)
                            
                            remaining_out = pool.checkedout()
                            if remaining_out > 0:
                                print(f"仍有 {remaining_out} 个连接未归还，将强制关闭")
                    
                    # 关闭引擎和连接池
                    print("正在关闭数据库引擎...")
                    self.engine.dispose()
                    print("数据库引擎已关闭")
                    
                except Exception as e:
                    print(f"关闭数据库引擎失败: {e}")
                    return False
            
            # 3. 清理会话工厂
            if hasattr(self, 'SessionLocal'):
                try:
                    # 关闭会话工厂
                    if hasattr(self.SessionLocal, 'close_all'):
                        self.SessionLocal.close_all()
                    print("会话工厂已清理")
                except Exception as e:
                    print(f"清理会话工厂失败: {e}")
            
            # 4. 清理会话注册表（如果存在）
            if hasattr(self, '_session_registry'):
                try:
                    self._session_registry.clear()
                    print("会话注册表已清理")
                except Exception as e:
                    print(f"清理会话注册表失败: {e}")
            
            print("数据库管理器优雅关闭完成")
            return True
            
        except Exception as e:
            print(f"数据库管理器关闭失败: {e}")
            return False
    
    def force_close_all_connections(self) -> int:
        """强制关闭所有数据库连接（紧急情况使用）
        
        Returns:
            int: 关闭的连接数
        """
        try:
            closed_count = 0
            
            if self.engine and hasattr(self.engine, 'pool'):
                pool = self.engine.pool
                if pool:
                    # 获取连接池中的所有连接
                    total_connections = pool.size()
                    
                    # 强制关闭引擎，这会关闭所有连接
                    self.engine.dispose()
                    closed_count = total_connections
                    
                    print(f"强制关闭了 {closed_count} 个数据库连接")
            
            return closed_count
            
        except Exception as e:
            print(f"强制关闭数据库连接失败: {e}")
            return 0
    
    def get_connection_stats(self) -> dict:
        """获取连接池统计信息"""
        try:
            if not self.engine or not hasattr(self.engine, 'pool'):
                return {'error': '引擎或连接池不存在'}
            
            pool = self.engine.pool
            if not pool:
                return {'error': '连接池不存在'}
            
            return {
                'pool_size': pool.size(),
                'checked_in': pool.checkedin(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'invalid': pool.invalid()
            }
            
        except Exception as e:
            return {'error': f'获取连接统计失败: {e}'}


# 全局数据库管理器实例
_global_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """获取全局数据库管理器实例"""
    global _global_db_manager
    if _global_db_manager is None:
        _global_db_manager = DatabaseManager()
    return _global_db_manager


def close_database_manager():
    """关闭全局数据库管理器"""
    global _global_db_manager
    if _global_db_manager:
        if _global_db_manager.engine:
            _global_db_manager.engine.dispose()
        _global_db_manager = None
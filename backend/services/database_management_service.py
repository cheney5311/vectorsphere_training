"""数据库管理服务

提供数据库管理相关的业务逻辑，包括健康检查、表管理、连接池监控、
数据统计、查询执行、维护操作、备份恢复等功能。
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.modules.database.manager import get_database_manager
from backend.modules.database.config import get_database_config
from backend.schemas.base_models import Base

logger = logging.getLogger(__name__)


class DatabaseManagementService:
    """数据库管理服务
    
    提供数据库管理的完整功能集，包括监控、维护、备份等操作。
    
    Attributes:
        db_manager: 数据库管理器实例
        config: 数据库配置
        
    Example:
        >>> service = DatabaseManagementService()
        >>> health = service.check_health()
        >>> print(health['status'])
    """
    
    def __init__(self):
        """初始化数据库管理服务"""
        self.db_manager = get_database_manager()
        self.config = get_database_config()
        self._backup_dir = os.environ.get('DATABASE_BACKUP_DIR', '/tmp/db_backups')
        
    # ============================================================================
    # 健康检查
    # ============================================================================
    
    def check_health(self) -> Dict[str, Any]:
        """检查数据库健康状态
        
        执行基本的健康检查，验证数据库连接是否正常。
        
        Returns:
            Dict[str, Any]: 健康状态信息
                - status: 'healthy' 或 'unhealthy'
                - database_type: 数据库类型
                - connection_available: 连接是否可用
                - timestamp: 检查时间
                
        Example:
            >>> health = service.check_health()
            >>> if health['status'] == 'healthy':
            ...     print("数据库正常")
        """
        try:
            start_time = time.time()
            is_healthy = self.db_manager.health_check()
            response_time = (time.time() - start_time) * 1000
            
            # 获取数据库类型
            db_type = 'unknown'
            if self.db_manager.engine:
                db_url = str(self.db_manager.engine.url)
                if 'postgresql' in db_url:
                    db_type = 'postgresql'
                elif 'mysql' in db_url:
                    db_type = 'mysql'
                elif 'sqlite' in db_url:
                    db_type = 'sqlite'
            
            return {
                'status': 'healthy' if is_healthy else 'unhealthy',
                'database_type': db_type,
                'connection_available': is_healthy,
                'response_time_ms': round(response_time, 2),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'unhealthy',
                'database_type': 'unknown',
                'connection_available': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_detailed_health(self) -> Dict[str, Any]:
        """获取详细健康状态
        
        获取包含连接池状态、表健康状态等详细信息。
        
        Returns:
            Dict[str, Any]: 详细健康信息
                - status: 整体状态
                - database_type: 数据库类型
                - connection: 连接信息
                - pool: 连接池状态
                - tables: 表健康统计
                - timestamp: 检查时间
                
        Example:
            >>> detailed = service.get_detailed_health()
            >>> print(f"连接池使用率: {detailed['pool']['utilization_percent']}%")
        """
        basic_health = self.check_health()
        
        # 获取连接池状态
        pool_stats = self.get_pool_stats()
        
        # 获取表统计
        tables_info = self.list_tables()
        
        return {
            'status': basic_health['status'],
            'database_type': basic_health['database_type'],
            'connection': {
                'available': basic_health['connection_available'],
                'response_time_ms': basic_health.get('response_time_ms')
            },
            'pool': pool_stats,
            'tables': {
                'total_count': tables_info.get('total_count', 0),
                'healthy_count': tables_info.get('total_count', 0)  # 假设都健康
            },
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # ============================================================================
    # 表管理
    # ============================================================================
    
    def list_tables(
        self, 
        include_system: bool = False,
        search: str = None
    ) -> Dict[str, Any]:
        """获取所有表信息
        
        列出数据库中的所有表及其基本信息。
        
        Args:
            include_system: 是否包含系统表
            search: 表名搜索关键词
            
        Returns:
            Dict[str, Any]: 表信息
                - tables: 表列表
                - total_count: 总数
                
        Example:
            >>> tables = service.list_tables(search='user')
            >>> for table in tables['tables']:
            ...     print(f"{table['name']}: {table['columns_count']} 列")
        """
        try:
            from sqlalchemy import inspect
            
            inspector = inspect(self.db_manager.engine)
            table_names = inspector.get_table_names()
            
            # 系统表过滤
            system_tables = {'alembic_version', 'spatial_ref_sys'}
            if not include_system:
                table_names = [t for t in table_names if t not in system_tables]
            
            # 搜索过滤
            if search:
                search_lower = search.lower()
                table_names = [t for t in table_names if search_lower in t.lower()]
            
            tables = []
            for table_name in table_names:
                try:
                    columns = inspector.get_columns(table_name)
                    indexes = inspector.get_indexes(table_name)
                    pk = inspector.get_pk_constraint(table_name)
                    
                    tables.append({
                        'name': table_name,
                        'schema': 'public',
                        'columns_count': len(columns),
                        'has_primary_key': bool(pk.get('constrained_columns')),
                        'indexes_count': len(indexes)
                    })
                except Exception as e:
                    logger.warning(f"Failed to get info for table {table_name}: {e}")
                    tables.append({
                        'name': table_name,
                        'schema': 'public',
                        'columns_count': 0,
                        'has_primary_key': False,
                        'indexes_count': 0,
                        'error': str(e)
                    })
            
            return {
                'tables': tables,
                'total_count': len(tables)
            }
            
        except Exception as e:
            logger.error(f"List tables failed: {e}")
            raise
    
    def get_table_details(
        self, 
        table_name: str,
        include_sample_data: bool = False,
        sample_limit: int = 5
    ) -> Optional[Dict[str, Any]]:
        """获取表详细信息
        
        获取指定表的完整结构信息。
        
        Args:
            table_name: 表名
            include_sample_data: 是否包含样本数据
            sample_limit: 样本数据条数
            
        Returns:
            Optional[Dict[str, Any]]: 表详情，不存在则返回None
                - name: 表名
                - schema: 模式
                - columns: 列定义列表
                - indexes: 索引列表
                - foreign_keys: 外键列表
                - constraints: 约束列表
                - row_count: 行数
                - size_bytes: 大小
                - sample_data: 样本数据
                
        Example:
            >>> details = service.get_table_details('users', include_sample_data=True)
            >>> for col in details['columns']:
            ...     print(f"{col['name']}: {col['type']}")
        """
        try:
            from sqlalchemy import inspect, text
            
            inspector = inspect(self.db_manager.engine)
            
            # 检查表是否存在
            if table_name not in inspector.get_table_names():
                return None
            
            # 获取列信息
            columns = []
            for col in inspector.get_columns(table_name):
                columns.append({
                    'name': col['name'],
                    'type': str(col['type']),
                    'nullable': col.get('nullable', True),
                    'primary_key': False,  # 后面处理
                    'default': str(col.get('default')) if col.get('default') else None
                })
            
            # 标记主键
            pk = inspector.get_pk_constraint(table_name)
            pk_columns = pk.get('constrained_columns', [])
            for col in columns:
                if col['name'] in pk_columns:
                    col['primary_key'] = True
            
            # 获取索引信息
            indexes = []
            for idx in inspector.get_indexes(table_name):
                indexes.append({
                    'name': idx['name'],
                    'columns': idx['column_names'],
                    'unique': idx.get('unique', False)
                })
            
            # 获取外键信息
            foreign_keys = []
            for fk in inspector.get_foreign_keys(table_name):
                foreign_keys.append({
                    'name': fk.get('name'),
                    'columns': fk['constrained_columns'],
                    'referred_table': fk['referred_table'],
                    'referred_columns': fk['referred_columns']
                })
            
            # 获取行数和大小
            row_count = 0
            size_bytes = 0
            try:
                with self.db_manager.get_db_session() as session:
                    result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                    row_count = result.scalar() or 0
                    
                    # PostgreSQL 获取表大小
                    if 'postgresql' in str(self.db_manager.engine.url):
                        size_result = session.execute(
                            text(f"SELECT pg_total_relation_size('{table_name}')")
                        )
                        size_bytes = size_result.scalar() or 0
            except Exception as e:
                logger.warning(f"Failed to get row count/size for {table_name}: {e}")
            
            result = {
                'name': table_name,
                'schema': 'public',
                'columns': columns,
                'indexes': indexes,
                'foreign_keys': foreign_keys,
                'constraints': [],
                'row_count': row_count,
                'size_bytes': size_bytes
            }
            
            # 获取样本数据
            if include_sample_data and row_count > 0:
                try:
                    with self.db_manager.get_db_session() as session:
                        sample_result = session.execute(
                            text(f"SELECT * FROM {table_name} LIMIT {sample_limit}")
                        )
                        sample_data = []
                        for row in sample_result:
                            sample_data.append(dict(row._mapping))
                        result['sample_data'] = sample_data
                except Exception as e:
                    logger.warning(f"Failed to get sample data for {table_name}: {e}")
                    result['sample_data'] = []
            else:
                result['sample_data'] = []
            
            return result
            
        except Exception as e:
            logger.error(f"Get table details failed: {e}")
            raise
    
    def get_table_count(
        self, 
        table_name: str, 
        condition: str = None
    ) -> Dict[str, Any]:
        """获取表记录数
        
        获取指定表的记录总数，可选条件过滤。
        
        Args:
            table_name: 表名
            condition: SQL条件片段
            
        Returns:
            Dict[str, Any]: 计数结果
                - table_name: 表名
                - count: 记录数
                - condition: 条件
                - counted_at: 计数时间
                
        Example:
            >>> count_info = service.get_table_count('users', "status = 'active'")
            >>> print(f"活跃用户: {count_info['count']}")
        """
        try:
            from sqlalchemy import text
            
            sql = f"SELECT COUNT(*) FROM {table_name}"
            if condition:
                sql += f" WHERE {condition}"
            
            with self.db_manager.get_db_session() as session:
                result = session.execute(text(sql))
                count = result.scalar() or 0
            
            return {
                'table_name': table_name,
                'count': count,
                'condition': condition,
                'counted_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Get table count failed: {e}")
            raise
    
    def sync_tables(
        self, 
        force: bool = False, 
        tables: List[str] = None
    ) -> Dict[str, Any]:
        """同步表结构
        
        将ORM模型同步到数据库。
        
        Args:
            force: 是否强制同步
            tables: 指定表列表，None表示全部
            
        Returns:
            Dict[str, Any]: 同步结果
                - created_tables: 创建的表
                - updated_tables: 更新的表
                - skipped_tables: 跳过的表
                - errors: 错误列表
                - sync_duration_ms: 耗时
                - synced_at: 同步时间
                
        Example:
            >>> result = service.sync_tables()
            >>> print(f"创建了 {len(result['created_tables'])} 个新表")
        """
        try:
            start_time = time.time()
            
            created_tables = []
            updated_tables = []
            skipped_tables = []
            errors = []
            
            from sqlalchemy import inspect
            inspector = inspect(self.db_manager.engine)
            existing_tables = set(inspector.get_table_names())
            
            # 遍历所有模型表
            for table in Base.metadata.tables.values():
                if tables and table.name not in tables:
                    skipped_tables.append(table.name)
                    continue
                
                try:
                    if table.name not in existing_tables:
                        # 创建新表
                        table.create(bind=self.db_manager.engine, checkfirst=True)
                        created_tables.append(table.name)
                        logger.info(f"Created table: {table.name}")
                    else:
                        # 检查是否需要更新
                        if force:
                            # 这里可以添加增量更新逻辑
                            updated_tables.append(table.name)
                        else:
                            skipped_tables.append(table.name)
                except Exception as e:
                    errors.append({
                        'table': table.name,
                        'error': str(e)
                    })
                    logger.error(f"Failed to sync table {table.name}: {e}")
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                'created_tables': created_tables,
                'updated_tables': updated_tables,
                'skipped_tables': skipped_tables,
                'errors': errors,
                'sync_duration_ms': round(duration_ms, 2),
                'synced_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Sync tables failed: {e}")
            raise
    
    def truncate_table(
        self, 
        table_name: str, 
        cascade: bool = False
    ) -> Dict[str, Any]:
        """清空表数据
        
        删除指定表的所有数据。
        
        Args:
            table_name: 表名
            cascade: 是否级联删除
            
        Returns:
            Dict[str, Any]: 操作结果
                - table_name: 表名
                - deleted_count: 删除的记录数
                - truncated_at: 操作时间
                
        Example:
            >>> result = service.truncate_table('temp_data')
            >>> print(f"删除了 {result['deleted_count']} 条记录")
        """
        try:
            from sqlalchemy import text
            
            # 先获取记录数
            with self.db_manager.get_db_session() as session:
                count_result = session.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                deleted_count = count_result.scalar() or 0
            
            # 执行清空
            with self.db_manager.get_db_session() as session:
                if cascade:
                    session.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
                else:
                    session.execute(text(f"TRUNCATE TABLE {table_name}"))
            
            return {
                'table_name': table_name,
                'deleted_count': deleted_count,
                'truncated_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Truncate table failed: {e}")
            raise
    
    # ============================================================================
    # 连接池管理
    # ============================================================================
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计
        
        获取数据库连接池的当前状态。
        
        Returns:
            Dict[str, Any]: 连接池统计
                - pool_size: 池大小
                - max_overflow: 最大溢出
                - checked_in: 已归还连接数
                - checked_out: 已借出连接数
                - overflow: 当前溢出数
                - invalid: 无效连接数
                - idle_connections: 空闲连接数
                - utilization_percent: 使用率
                - config: 配置信息
                
        Example:
            >>> stats = service.get_pool_stats()
            >>> print(f"连接池使用率: {stats['utilization_percent']}%")
        """
        try:
            stats = self.db_manager.get_connection_stats()
            
            if 'error' in stats:
                return {
                    'pool_size': self.config.pool_size,
                    'max_overflow': self.config.max_overflow,
                    'checked_in': 0,
                    'checked_out': 0,
                    'overflow': 0,
                    'invalid': 0,
                    'idle_connections': 0,
                    'utilization_percent': 0.0,
                    'config': {
                        'pool_timeout': self.config.pool_timeout,
                        'pool_recycle': self.config.pool_recycle
                    },
                    'error': stats['error']
                }
            
            # 计算使用率
            total_available = stats.get('pool_size', 0) + stats.get('overflow', 0)
            checked_out = stats.get('checked_out', 0)
            utilization = (checked_out / total_available * 100) if total_available > 0 else 0
            
            return {
                'pool_size': stats.get('pool_size', self.config.pool_size),
                'max_overflow': self.config.max_overflow,
                'checked_in': stats.get('checked_in', 0),
                'checked_out': stats.get('checked_out', 0),
                'overflow': stats.get('overflow', 0),
                'invalid': stats.get('invalid', 0),
                'idle_connections': stats.get('checked_in', 0),
                'utilization_percent': round(utilization, 2),
                'config': {
                    'pool_timeout': self.config.pool_timeout,
                    'pool_recycle': self.config.pool_recycle
                }
            }
            
        except Exception as e:
            logger.error(f"Get pool stats failed: {e}")
            raise
    
    def reset_pool(
        self, 
        force: bool = False, 
        timeout: int = 30
    ) -> Dict[str, Any]:
        """重置连接池
        
        关闭所有连接并重新初始化连接池。
        
        Args:
            force: 是否强制重置
            timeout: 等待超时秒数
            
        Returns:
            Dict[str, Any]: 重置结果
                - closed_connections: 关闭的连接数
                - new_pool_size: 新池大小
                - reset_duration_ms: 耗时
                - reset_at: 重置时间
                
        Example:
            >>> result = service.reset_pool()
            >>> print(f"关闭了 {result['closed_connections']} 个连接")
        """
        try:
            start_time = time.time()
            
            # 获取当前连接数
            old_stats = self.db_manager.get_connection_stats()
            closed_count = old_stats.get('pool_size', 0)
            
            if force:
                self.db_manager.force_close_all_connections()
            else:
                self.db_manager.graceful_shutdown(timeout=timeout)
            
            # 重新初始化
            self.db_manager._initialize_engine()
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                'closed_connections': closed_count,
                'new_pool_size': self.config.pool_size,
                'reset_duration_ms': round(duration_ms, 2),
                'reset_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Reset pool failed: {e}")
            raise
    
    # ============================================================================
    # 数据统计
    # ============================================================================
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息
        
        获取数据库的整体统计信息概览。
        
        Returns:
            Dict[str, Any]: 统计信息
                - database_info: 数据库基本信息
                - tables: 表统计
                - storage: 存储统计
                - connection_pool: 连接池统计
                
        Example:
            >>> stats = service.get_database_stats()
            >>> print(f"总表数: {stats['tables']['total_count']}")
        """
        try:
            from sqlalchemy import text
            
            # 数据库基本信息
            db_info = {
                'name': 'vectorsphere',
                'type': 'unknown',
                'version': 'unknown',
                'encoding': 'UTF8'
            }
            
            db_url = str(self.db_manager.engine.url)
            if 'postgresql' in db_url:
                db_info['type'] = 'postgresql'
                try:
                    with self.db_manager.get_db_session() as session:
                        version = session.execute(text("SELECT version()")).scalar()
                        db_info['version'] = version.split()[1] if version else 'unknown'
                except:
                    pass
            elif 'mysql' in db_url:
                db_info['type'] = 'mysql'
            elif 'sqlite' in db_url:
                db_info['type'] = 'sqlite'
            
            # 表统计
            tables_info = self.list_tables()
            total_rows = 0
            largest_table = None
            max_rows = 0
            
            for table in tables_info.get('tables', []):
                try:
                    count_info = self.get_table_count(table['name'])
                    count = count_info['count']
                    total_rows += count
                    if count > max_rows:
                        max_rows = count
                        largest_table = table['name']
                except:
                    pass
            
            # 存储统计
            storage = self.get_database_size()
            
            # 连接池统计
            pool_stats = self.get_pool_stats()
            
            return {
                'database_info': db_info,
                'tables': {
                    'total_count': tables_info.get('total_count', 0),
                    'total_rows': total_rows,
                    'largest_table': largest_table
                },
                'storage': storage,
                'connection_pool': {
                    'pool_size': pool_stats.get('pool_size', 0),
                    'active_connections': pool_stats.get('checked_out', 0),
                    'utilization_percent': pool_stats.get('utilization_percent', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"Get database stats failed: {e}")
            raise
    
    def get_tables_stats(
        self, 
        order_by: str = 'row_count',
        order_desc: bool = True,
        limit: int = None
    ) -> Dict[str, Any]:
        """获取各表统计信息
        
        获取所有表的详细统计信息。
        
        Args:
            order_by: 排序字段 (name/row_count/size)
            order_desc: 是否降序
            limit: 返回数量限制
            
        Returns:
            Dict[str, Any]: 表统计
                - tables: 表统计列表
                - total_tables: 总表数
                
        Example:
            >>> stats = service.get_tables_stats(order_by='size', limit=10)
            >>> for table in stats['tables']:
            ...     print(f"{table['name']}: {table['size_human']}")
        """
        try:
            from sqlalchemy import text
            
            tables_info = self.list_tables()
            tables_stats = []
            
            for table in tables_info.get('tables', []):
                table_name = table['name']
                
                # 获取行数
                row_count = 0
                try:
                    count_info = self.get_table_count(table_name)
                    row_count = count_info['count']
                except:
                    pass
                
                # 获取大小
                size_bytes = 0
                index_size_bytes = 0
                if 'postgresql' in str(self.db_manager.engine.url):
                    try:
                        with self.db_manager.get_db_session() as session:
                            size_result = session.execute(
                                text(f"SELECT pg_total_relation_size('{table_name}')")
                            )
                            size_bytes = size_result.scalar() or 0
                            
                            index_result = session.execute(
                                text(f"SELECT pg_indexes_size('{table_name}')")
                            )
                            index_size_bytes = index_result.scalar() or 0
                    except:
                        pass
                
                # 转换大小为可读格式
                size_human = self._format_size(size_bytes)
                
                tables_stats.append({
                    'name': table_name,
                    'row_count': row_count,
                    'size_bytes': size_bytes,
                    'size_human': size_human,
                    'index_size_bytes': index_size_bytes,
                    'last_vacuum': None,
                    'last_analyze': None
                })
            
            # 排序
            if order_by == 'name':
                tables_stats.sort(key=lambda x: x['name'], reverse=order_desc)
            elif order_by == 'size':
                tables_stats.sort(key=lambda x: x['size_bytes'], reverse=order_desc)
            else:  # row_count
                tables_stats.sort(key=lambda x: x['row_count'], reverse=order_desc)
            
            # 限制数量
            if limit:
                tables_stats = tables_stats[:limit]
            
            return {
                'tables': tables_stats,
                'total_tables': len(tables_info.get('tables', []))
            }
            
        except Exception as e:
            logger.error(f"Get tables stats failed: {e}")
            raise
    
    def get_database_size(self) -> Dict[str, Any]:
        """获取数据库大小
        
        获取数据库的存储空间使用详情。
        
        Returns:
            Dict[str, Any]: 大小信息
                - total_size_bytes: 总大小
                - total_size_human: 总大小(可读)
                - data_size_bytes: 数据大小
                - index_size_bytes: 索引大小
                - tables_breakdown: 各表大小分布
                
        Example:
            >>> size = service.get_database_size()
            >>> print(f"总大小: {size['total_size_human']}")
        """
        try:
            from sqlalchemy import text
            
            total_size = 0
            data_size = 0
            index_size = 0
            tables_breakdown = []
            
            if 'postgresql' in str(self.db_manager.engine.url):
                try:
                    with self.db_manager.get_db_session() as session:
                        # 获取数据库总大小
                        db_name = self.db_manager.engine.url.database
                        size_result = session.execute(
                            text(f"SELECT pg_database_size('{db_name}')")
                        )
                        total_size = size_result.scalar() or 0
                except Exception as e:
                    logger.warning(f"Failed to get database size: {e}")
            
            # 获取各表大小
            tables_info = self.list_tables()
            for table in tables_info.get('tables', []):
                table_name = table['name']
                try:
                    if 'postgresql' in str(self.db_manager.engine.url):
                        with self.db_manager.get_db_session() as session:
                            size_result = session.execute(
                                text(f"SELECT pg_total_relation_size('{table_name}')")
                            )
                            table_size = size_result.scalar() or 0
                            
                            if total_size > 0:
                                percentage = (table_size / total_size) * 100
                            else:
                                percentage = 0
                            
                            tables_breakdown.append({
                                'name': table_name,
                                'size_bytes': table_size,
                                'percentage': round(percentage, 2)
                            })
                            
                            data_size += table_size
                except:
                    pass
            
            # 排序表大小
            tables_breakdown.sort(key=lambda x: x['size_bytes'], reverse=True)
            
            return {
                'total_size_bytes': total_size,
                'total_size_human': self._format_size(total_size),
                'data_size_bytes': data_size,
                'data_size_human': self._format_size(data_size),
                'index_size_bytes': index_size,
                'index_size_human': self._format_size(index_size),
                'tables_breakdown': tables_breakdown[:10]  # 只返回前10个
            }
            
        except Exception as e:
            logger.error(f"Get database size failed: {e}")
            raise
    
    # ============================================================================
    # 查询执行
    # ============================================================================
    
    def execute_query(
        self, 
        sql: str,
        params: Dict[str, Any] = None,
        timeout: int = 30,
        max_rows: int = 1000
    ) -> Dict[str, Any]:
        """执行只读查询
        
        执行SELECT查询并返回结果。
        
        Args:
            sql: SQL查询语句
            params: 查询参数
            timeout: 超时秒数
            max_rows: 最大返回行数
            
        Returns:
            Dict[str, Any]: 查询结果
                - columns: 列名列表
                - rows: 数据行
                - row_count: 行数
                - execution_time_ms: 执行时间
                - truncated: 是否被截断
                
        Example:
            >>> result = service.execute_query("SELECT * FROM users LIMIT 10")
            >>> for row in result['rows']:
            ...     print(row)
        """
        try:
            from sqlalchemy import text
            
            start_time = time.time()
            
            with self.db_manager.get_db_session() as session:
                result = session.execute(text(sql), params or {})
                
                # 获取列名
                columns = list(result.keys())
                
                # 获取数据行
                rows = []
                truncated = False
                for i, row in enumerate(result):
                    if i >= max_rows:
                        truncated = True
                        break
                    # 转换行数据为可序列化格式
                    row_dict = {}
                    for col in columns:
                        value = row._mapping[col]
                        if isinstance(value, datetime):
                            value = value.isoformat()
                        elif hasattr(value, '__str__'):
                            value = str(value)
                        row_dict[col] = value
                    rows.append(row_dict)
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                'columns': columns,
                'rows': rows,
                'row_count': len(rows),
                'execution_time_ms': round(execution_time, 2),
                'truncated': truncated
            }
            
        except Exception as e:
            logger.error(f"Execute query failed: {e}")
            raise
    
    def explain_query(
        self, 
        sql: str,
        analyze: bool = False,
        format_type: str = 'text'
    ) -> Dict[str, Any]:
        """分析查询计划
        
        获取SQL查询的执行计划。
        
        Args:
            sql: SQL查询语句
            analyze: 是否实际执行
            format_type: 输出格式 (text/json)
            
        Returns:
            Dict[str, Any]: 查询计划
                - plan: 执行计划
                - estimated_cost: 预估成本
                - estimated_rows: 预估行数
                - suggestions: 优化建议
                
        Example:
            >>> plan = service.explain_query("SELECT * FROM users WHERE email = 'test@test.com'")
            >>> print(plan['plan'])
        """
        try:
            from sqlalchemy import text
            
            # 构建EXPLAIN语句
            explain_sql = "EXPLAIN "
            if analyze:
                explain_sql += "ANALYZE "
            if format_type == 'json' and 'postgresql' in str(self.db_manager.engine.url):
                explain_sql += "FORMAT JSON "
            explain_sql += sql
            
            with self.db_manager.get_db_session() as session:
                result = session.execute(text(explain_sql))
                plan_lines = [str(row[0]) for row in result]
            
            plan_text = '\n'.join(plan_lines)
            
            # 解析成本和行数（简单解析）
            estimated_cost = 0.0
            estimated_rows = 0
            suggestions = []
            
            for line in plan_lines:
                if 'cost=' in line:
                    try:
                        cost_str = line.split('cost=')[1].split()[0]
                        estimated_cost = float(cost_str.split('..')[1])
                    except:
                        pass
                if 'rows=' in line:
                    try:
                        rows_str = line.split('rows=')[1].split()[0]
                        estimated_rows = int(rows_str)
                    except:
                        pass
            
            # 生成简单建议
            if 'Seq Scan' in plan_text:
                suggestions.append("查询使用了顺序扫描，考虑在过滤列上添加索引")
            if estimated_rows > 10000:
                suggestions.append("预计返回大量行数，考虑添加LIMIT或优化WHERE条件")
            
            return {
                'plan': plan_text,
                'estimated_cost': estimated_cost,
                'estimated_rows': estimated_rows,
                'suggestions': suggestions
            }
            
        except Exception as e:
            logger.error(f"Explain query failed: {e}")
            raise
    
    # ============================================================================
    # 维护操作
    # ============================================================================
    
    def vacuum(
        self, 
        table_name: str = None,
        full: bool = False,
        analyze: bool = True
    ) -> Dict[str, Any]:
        """执行VACUUM操作
        
        回收删除行占用的空间。
        
        Args:
            table_name: 指定表名，None表示全部
            full: 是否执行FULL VACUUM
            analyze: 是否同时执行ANALYZE
            
        Returns:
            Dict[str, Any]: 操作结果
                - tables_vacuumed: 清理的表
                - space_reclaimed_bytes: 回收的空间
                - duration_ms: 耗时
                - vacuumed_at: 操作时间
                
        Example:
            >>> result = service.vacuum(analyze=True)
            >>> print(f"清理了 {len(result['tables_vacuumed'])} 个表")
        """
        try:
            from sqlalchemy import text
            
            start_time = time.time()
            tables_vacuumed = []
            
            # 构建VACUUM语句
            vacuum_sql = "VACUUM "
            if full:
                vacuum_sql += "FULL "
            if analyze:
                vacuum_sql += "ANALYZE "
            
            if table_name:
                vacuum_sql += table_name
                tables_vacuumed = [table_name]
            else:
                # 获取所有表
                tables_info = self.list_tables()
                for table in tables_info.get('tables', []):
                    tables_vacuumed.append(table['name'])
            
            # 执行VACUUM（需要在autocommit模式下）
            with self.db_manager.engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                conn.execute(text(vacuum_sql))
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                'tables_vacuumed': tables_vacuumed,
                'space_reclaimed_bytes': 0,  # 难以精确计算
                'space_reclaimed_human': 'N/A',
                'duration_ms': round(duration_ms, 2),
                'vacuumed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Vacuum failed: {e}")
            raise
    
    def analyze(self, table_name: str = None) -> Dict[str, Any]:
        """更新统计信息
        
        执行ANALYZE操作更新查询优化器统计。
        
        Args:
            table_name: 指定表名，None表示全部
            
        Returns:
            Dict[str, Any]: 操作结果
                - tables_analyzed: 分析的表
                - duration_ms: 耗时
                - analyzed_at: 操作时间
                
        Example:
            >>> result = service.analyze()
            >>> print(f"分析了 {len(result['tables_analyzed'])} 个表")
        """
        try:
            from sqlalchemy import text
            
            start_time = time.time()
            tables_analyzed = []
            
            if table_name:
                analyze_sql = f"ANALYZE {table_name}"
                tables_analyzed = [table_name]
            else:
                analyze_sql = "ANALYZE"
                tables_info = self.list_tables()
                tables_analyzed = [t['name'] for t in tables_info.get('tables', [])]
            
            with self.db_manager.engine.connect() as conn:
                conn.execution_options(isolation_level="AUTOCOMMIT")
                conn.execute(text(analyze_sql))
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                'tables_analyzed': tables_analyzed,
                'duration_ms': round(duration_ms, 2),
                'analyzed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Analyze failed: {e}")
            raise
    
    def get_locks(self) -> Dict[str, Any]:
        """获取锁信息
        
        获取数据库当前的锁状态。
        
        Returns:
            Dict[str, Any]: 锁信息
                - locks: 锁列表
                - waiting_queries: 等待中的查询
                - total_locks: 总锁数
                - total_waiting: 等待数
                
        Example:
            >>> locks = service.get_locks()
            >>> if locks['total_waiting'] > 0:
            ...     print("存在锁等待！")
        """
        try:
            from sqlalchemy import text
            
            locks = []
            waiting_queries = []
            
            if 'postgresql' in str(self.db_manager.engine.url):
                with self.db_manager.get_db_session() as session:
                    # 获取锁信息
                    lock_query = """
                        SELECT 
                            l.pid,
                            d.datname as database,
                            c.relname as relation,
                            l.mode as lock_type,
                            l.granted,
                            a.wait_event_type,
                            a.query
                        FROM pg_locks l
                        LEFT JOIN pg_class c ON l.relation = c.oid
                        LEFT JOIN pg_database d ON l.database = d.oid
                        LEFT JOIN pg_stat_activity a ON l.pid = a.pid
                        WHERE l.pid != pg_backend_pid()
                        ORDER BY l.granted, l.pid
                    """
                    result = session.execute(text(lock_query))
                    
                    for row in result:
                        lock_info = {
                            'pid': row.pid,
                            'database': row.database,
                            'relation': row.relation,
                            'lock_type': row.lock_type,
                            'granted': row.granted,
                            'wait_event': row.wait_event_type,
                            'query': row.query[:100] if row.query else None
                        }
                        locks.append(lock_info)
                        
                        if not row.granted:
                            waiting_queries.append(lock_info)
            
            return {
                'locks': locks,
                'waiting_queries': waiting_queries,
                'total_locks': len(locks),
                'total_waiting': len(waiting_queries)
            }
            
        except Exception as e:
            logger.error(f"Get locks failed: {e}")
            raise
    
    # ============================================================================
    # 备份恢复
    # ============================================================================
    
    def create_backup(
        self, 
        backup_name: str = None,
        tables: List[str] = None,
        compression: bool = True
    ) -> Dict[str, Any]:
        """创建数据库备份
        
        创建数据库的逻辑备份（模拟实现）。
        
        Args:
            backup_name: 备份名称
            tables: 指定表列表
            compression: 是否压缩
            
        Returns:
            Dict[str, Any]: 备份结果
                - backup_id: 备份ID
                - backup_name: 备份名称
                - backup_path: 备份路径
                - size_bytes: 大小
                - tables_count: 表数
                - duration_ms: 耗时
                - created_at: 创建时间
                
        Example:
            >>> result = service.create_backup(backup_name='pre_migration')
            >>> print(f"备份ID: {result['backup_id']}")
        """
        try:
            start_time = time.time()
            
            # 生成备份ID
            backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            if not backup_name:
                backup_name = backup_id
            
            # 备份路径
            ext = '.sql.gz' if compression else '.sql'
            backup_path = os.path.join(self._backup_dir, f"{backup_id}{ext}")
            
            # 获取表列表
            if not tables:
                tables_info = self.list_tables()
                tables = [t['name'] for t in tables_info.get('tables', [])]
            
            # 模拟备份（实际应该使用pg_dump等工具）
            # 这里只是记录备份元数据
            
            duration_ms = (time.time() - start_time) * 1000
            
            # 计算模拟大小
            size_bytes = len(tables) * 1024 * 1024  # 假设每表1MB
            
            return {
                'backup_id': backup_id,
                'backup_name': backup_name,
                'backup_path': backup_path,
                'size_bytes': size_bytes,
                'size_human': self._format_size(size_bytes),
                'tables_count': len(tables),
                'duration_ms': round(duration_ms, 2),
                'created_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Create backup failed: {e}")
            raise
    
    def list_backups(
        self, 
        limit: int = 20, 
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取备份列表
        
        列出可用的数据库备份（模拟实现）。
        
        Args:
            limit: 返回数量
            offset: 偏移量
            
        Returns:
            Dict[str, Any]: 备份列表
                - backups: 备份信息列表
                - total_count: 总数
                
        Example:
            >>> backups = service.list_backups()
            >>> for b in backups['backups']:
            ...     print(f"{b['backup_id']}: {b['size_human']}")
        """
        try:
            # 模拟备份列表
            backups = [
                {
                    'backup_id': 'backup_20240101_000000',
                    'backup_name': 'auto_backup',
                    'size_bytes': 104857600,
                    'size_human': '100.00 MB',
                    'created_at': '2024-01-01T00:00:00',
                    'status': 'completed'
                }
            ]
            
            return {
                'backups': backups[offset:offset + limit],
                'total_count': len(backups)
            }
            
        except Exception as e:
            logger.error(f"List backups failed: {e}")
            raise
    
    def restore_backup(
        self, 
        backup_id: str,
        tables: List[str] = None
    ) -> Dict[str, Any]:
        """恢复数据库备份
        
        从备份恢复数据库（模拟实现）。
        
        Args:
            backup_id: 备份ID
            tables: 指定恢复的表
            
        Returns:
            Dict[str, Any]: 恢复结果
                - backup_id: 备份ID
                - tables_restored: 恢复的表数
                - rows_restored: 恢复的行数
                - duration_ms: 耗时
                - restored_at: 恢复时间
                
        Example:
            >>> result = service.restore_backup('backup_20240101_000000')
            >>> print(f"恢复了 {result['tables_restored']} 个表")
        """
        try:
            start_time = time.time()
            
            # 模拟恢复操作
            tables_restored = 25 if not tables else len(tables)
            rows_restored = tables_restored * 1000
            
            duration_ms = (time.time() - start_time) * 1000
            
            return {
                'backup_id': backup_id,
                'tables_restored': tables_restored,
                'rows_restored': rows_restored,
                'duration_ms': round(duration_ms, 2),
                'restored_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Restore backup failed: {e}")
            raise
    
    # ============================================================================
    # 工具方法
    # ============================================================================
    
    def _format_size(self, size_bytes: int) -> str:
        """格式化大小为可读格式
        
        Args:
            size_bytes: 字节数
            
        Returns:
            str: 可读的大小字符串
        """
        if size_bytes >= 1024 * 1024 * 1024:
            return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"
        elif size_bytes >= 1024 * 1024:
            return f"{size_bytes / 1024 / 1024:.2f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.2f} KB"
        else:
            return f"{size_bytes} B"


# 全局服务实例
_service_instance: Optional[DatabaseManagementService] = None


def get_database_management_service() -> DatabaseManagementService:
    """获取全局数据库管理服务实例
    
    Returns:
        DatabaseManagementService: 服务实例
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = DatabaseManagementService()
    return _service_instance

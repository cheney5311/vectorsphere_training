"""数据库 Schema 导出工具

根据 SQLAlchemy ORM 模型生成各数据库的 DDL 语句。
支持 SQLite、PostgreSQL、MySQL 三种数据库。
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import create_engine, MetaData, inspect
from sqlalchemy.schema import CreateTable, CreateIndex
from sqlalchemy.dialects import postgresql, mysql, sqlite

logger = logging.getLogger(__name__)


class SchemaExporter:
    """Schema 导出器
    
    从 SQLAlchemy ORM 模型生成各数据库的建表语句。
    
    Attributes:
        base: SQLAlchemy Base 类
        
    Example:
        >>> from backend.schemas.base_models import Base
        >>> exporter = SchemaExporter(Base)
        >>> exporter.export_all('/data/sql')
    """
    
    # 支持的数据库类型和对应的方言
    DIALECTS = {
        'sqlite': sqlite.dialect(),
        'postgresql': postgresql.dialect(),
        'mysql': mysql.dialect()
    }
    
    def __init__(self, base=None):
        """初始化导出器
        
        Args:
            base: SQLAlchemy declarative_base 实例
        """
        if base is None:
            from backend.schemas.base_models import Base
            base = Base
        self.base = base
        self.metadata = base.metadata
        
    def _load_all_models(self):
        """加载所有 ORM 模型
        
        通过导入所有 schema 模块来注册模型到 metadata。
        只导入核心模块，避免重复定义冲突。
        """
        try:
            # 核心基础模块
            import backend.schemas.base_models
            import backend.schemas.auth_models
            import backend.schemas.project_models
            import backend.schemas.dataset
            import backend.schemas.model_models
            
            # 数据管理模块
            try:
                import backend.schemas.data_discovery_db_models
            except Exception as e:
                logger.warning(f"Failed to import data_discovery_db_models: {e}")
            
            try:
                import backend.schemas.data_preprocessing_db_models
            except Exception as e:
                logger.warning(f"Failed to import data_preprocessing_db_models: {e}")
            
            try:
                import backend.schemas.data_quality_db_models
            except Exception as e:
                logger.warning(f"Failed to import data_quality_db_models: {e}")
            
            # 业务模块
            try:
                import backend.schemas.embedding_models
            except Exception as e:
                logger.warning(f"Failed to import embedding_models: {e}")
            
            try:
                import backend.schemas.billing_models
            except Exception as e:
                logger.warning(f"Failed to import billing_models: {e}")
            
            try:
                import backend.schemas.workflow_models
            except Exception as e:
                logger.warning(f"Failed to import workflow_models: {e}")
            
            # 跳过 security_models 因为有表名冲突
            # try:
            #     import backend.schemas.security_models
            # except Exception as e:
            #     logger.warning(f"Failed to import security_models: {e}")
            
            logger.info(f"Loaded {len(self.metadata.tables)} tables")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def get_table_ddl(self, table, dialect_name: str) -> str:
        """获取单个表的 DDL 语句
        
        Args:
            table: SQLAlchemy Table 对象
            dialect_name: 数据库类型 (sqlite/postgresql/mysql)
            
        Returns:
            str: DDL 语句
        """
        dialect = self.DIALECTS.get(dialect_name)
        if not dialect:
            raise ValueError(f"Unsupported dialect: {dialect_name}")
        
        # 手动构建 DDL 以处理 UUID 类型兼容性
        ddl_lines = [f"CREATE TABLE {table.name} ("]
        column_defs = []
        
        for column in table.columns:
            col_def = self._get_column_ddl(column, dialect_name)
            column_defs.append(f"    {col_def}")
        
        ddl_lines.append(",\n".join(column_defs))
        ddl_lines.append(")")
        
        return "\n".join(ddl_lines)
    
    def _get_column_ddl(self, column, dialect_name: str) -> str:
        """获取列定义 DDL
        
        Args:
            column: SQLAlchemy Column 对象
            dialect_name: 数据库类型
            
        Returns:
            str: 列定义语句
        """
        from sqlalchemy.dialects.postgresql import UUID
        from sqlalchemy import String, Text, Integer, Float, Boolean, DateTime, JSON
        
        col_type = column.type
        type_name = type(col_type).__name__
        
        # 类型映射
        if isinstance(col_type, UUID) or type_name == 'UUID':
            if dialect_name == 'sqlite':
                sql_type = 'VARCHAR(36)'
            elif dialect_name == 'mysql':
                sql_type = 'CHAR(36)'
            else:
                sql_type = 'UUID'
        elif type_name == 'String' or isinstance(col_type, String):
            length = getattr(col_type, 'length', None)
            if length:
                sql_type = f'VARCHAR({length})'
            else:
                sql_type = 'VARCHAR(255)'
        elif type_name == 'Text' or isinstance(col_type, Text):
            sql_type = 'TEXT'
        elif type_name == 'Integer' or isinstance(col_type, Integer):
            sql_type = 'INTEGER'
        elif type_name == 'Float' or isinstance(col_type, Float):
            if dialect_name == 'postgresql':
                sql_type = 'DOUBLE PRECISION'
            else:
                sql_type = 'FLOAT'
        elif type_name == 'Boolean' or isinstance(col_type, Boolean):
            if dialect_name == 'sqlite':
                sql_type = 'INTEGER'
            elif dialect_name == 'mysql':
                sql_type = 'TINYINT(1)'
            else:
                sql_type = 'BOOLEAN'
        elif type_name == 'DateTime' or isinstance(col_type, DateTime):
            if dialect_name == 'mysql':
                sql_type = 'DATETIME'
            else:
                sql_type = 'TIMESTAMP'
        elif type_name in ('JSON', 'JSONB') or isinstance(col_type, JSON):
            if dialect_name == 'sqlite':
                sql_type = 'TEXT'
            elif dialect_name == 'mysql':
                sql_type = 'JSON'
            else:
                sql_type = 'JSONB'
        else:
            # 默认类型
            sql_type = str(col_type)
        
        # 构建列定义
        parts = [column.name, sql_type]
        
        # 主键
        if column.primary_key:
            parts.append('PRIMARY KEY')
        
        # NOT NULL
        if not column.nullable and not column.primary_key:
            parts.append('NOT NULL')
        
        # UNIQUE
        if column.unique and not column.primary_key:
            parts.append('UNIQUE')
        
        # 默认值
        if column.default is not None:
            default_val = column.default.arg if hasattr(column.default, 'arg') else column.default
            if callable(default_val):
                pass  # 跳过函数默认值
            elif isinstance(default_val, str):
                parts.append(f"DEFAULT '{default_val}'")
            elif isinstance(default_val, bool):
                if dialect_name == 'sqlite':
                    parts.append(f"DEFAULT {1 if default_val else 0}")
                else:
                    parts.append(f"DEFAULT {default_val}")
            elif default_val is not None:
                parts.append(f"DEFAULT {default_val}")
        
        return " ".join(parts)
    
    def get_index_ddl(self, index, dialect_name: str) -> str:
        """获取索引的 DDL 语句
        
        Args:
            index: SQLAlchemy Index 对象
            dialect_name: 数据库类型
            
        Returns:
            str: CREATE INDEX 语句
        """
        dialect = self.DIALECTS.get(dialect_name)
        if not dialect:
            raise ValueError(f"Unsupported dialect: {dialect_name}")
        
        try:
            create_index = CreateIndex(index)
            ddl = str(create_index.compile(dialect=dialect))
            return ddl
        except Exception as e:
            logger.warning(f"Failed to generate index DDL for {index.name}: {e}")
            return f"-- Failed to generate index: {index.name}"
    
    def generate_ddl(self, dialect_name: str) -> str:
        """生成指定数据库的完整 DDL
        
        Args:
            dialect_name: 数据库类型 (sqlite/postgresql/mysql)
            
        Returns:
            str: 完整的 DDL 语句
        """
        self._load_all_models()
        
        lines = []
        
        # 文件头部注释
        lines.append(f"-- VectorSphere 智能平台数据库表结构")
        lines.append(f"-- 数据库类型: {dialect_name.upper()}")
        lines.append(f"-- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"-- 表数量: {len(self.metadata.tables)}")
        lines.append("")
        lines.append("-- ============================================================")
        lines.append("-- 注意事项:")
        lines.append("-- 1. 请在执行前确保数据库已创建")
        lines.append("-- 2. 如需重建表，请先备份数据")
        lines.append("-- 3. 外键约束可能需要调整执行顺序")
        lines.append("-- ============================================================")
        lines.append("")
        
        # 按表名排序
        sorted_tables = sorted(self.metadata.tables.values(), key=lambda t: t.name)
        
        # 生成每个表的 DDL
        for table in sorted_tables:
            lines.append(f"-- ------------------------------------------------------------")
            lines.append(f"-- Table: {table.name}")
            lines.append(f"-- ------------------------------------------------------------")
            
            try:
                ddl = self.get_table_ddl(table, dialect_name)
                
                # 清理和格式化 DDL
                ddl = self._clean_ddl(ddl, dialect_name)
                
                lines.append(ddl)
                lines.append("")
                
                # 生成索引
                for index in table.indexes:
                    try:
                        index_ddl = self.get_index_ddl(index, dialect_name)
                        if not index_ddl.startswith("--"):
                            index_ddl = self._clean_ddl(index_ddl, dialect_name)
                            lines.append(index_ddl)
                    except Exception as e:
                        lines.append(f"-- Index {index.name} generation failed: {e}")
                
                lines.append("")
                
            except Exception as e:
                lines.append(f"-- Failed to generate DDL for {table.name}: {e}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _clean_ddl(self, ddl: str, dialect_name: str) -> str:
        """清理和格式化 DDL 语句
        
        Args:
            ddl: 原始 DDL
            dialect_name: 数据库类型
            
        Returns:
            str: 清理后的 DDL
        """
        # 移除多余的换行
        ddl = ddl.strip()
        
        # 确保语句以分号结尾
        if not ddl.endswith(";"):
            ddl += ";"
        
        # SQLite 特殊处理
        if dialect_name == 'sqlite':
            # SQLite 不支持 UUID 类型，转换为 VARCHAR
            ddl = ddl.replace("UUID", "VARCHAR(36)")
            # SQLite 不支持 JSONB，使用 TEXT
            ddl = ddl.replace("JSONB", "TEXT")
            ddl = ddl.replace("JSON", "TEXT")
        
        # MySQL 特殊处理
        elif dialect_name == 'mysql':
            # MySQL UUID 处理
            ddl = ddl.replace("UUID", "CHAR(36)")
            # MySQL JSON 类型
            ddl = ddl.replace("JSONB", "JSON")
            # 添加引擎设置
            if "CREATE TABLE" in ddl and "ENGINE=" not in ddl:
                ddl = ddl.rstrip(";") + " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"
        
        # PostgreSQL 特殊处理
        elif dialect_name == 'postgresql':
            # 确保使用正确的 UUID 类型
            pass
        
        return ddl
    
    def export_to_file(self, dialect_name: str, output_path: str) -> str:
        """导出 DDL 到文件
        
        Args:
            dialect_name: 数据库类型
            output_path: 输出文件路径
            
        Returns:
            str: 输出文件路径
        """
        ddl = self.generate_ddl(dialect_name)
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(ddl)
        
        logger.info(f"Exported {dialect_name} DDL to {output_path}")
        return output_path
    
    def export_all(self, output_dir: str) -> Dict[str, str]:
        """导出所有支持的数据库类型的 DDL
        
        Args:
            output_dir: 输出目录
            
        Returns:
            Dict[str, str]: 数据库类型到文件路径的映射
        """
        result = {}
        
        for dialect_name in self.DIALECTS.keys():
            filename = f"schema_{dialect_name}.sql"
            output_path = os.path.join(output_dir, filename)
            
            try:
                self.export_to_file(dialect_name, output_path)
                result[dialect_name] = output_path
                print(f"✓ 导出 {dialect_name.upper()} DDL: {output_path}")
            except Exception as e:
                logger.error(f"Failed to export {dialect_name} DDL: {e}")
                result[dialect_name] = f"Error: {e}"
        
        return result
    
    def get_table_list(self) -> List[str]:
        """获取所有表名列表
        
        Returns:
            List[str]: 表名列表
        """
        self._load_all_models()
        return sorted(self.metadata.tables.keys())


class DatabaseInitializer:
    """数据库初始化器
    
    负责在应用启动时根据配置初始化数据库表结构。
    
    Attributes:
        db_manager: 数据库管理器实例
        
    Example:
        >>> initializer = DatabaseInitializer()
        >>> initializer.initialize()
    """
    
    def __init__(self, db_manager=None):
        """初始化
        
        Args:
            db_manager: 数据库管理器实例，为 None 时自动获取
        """
        self.db_manager = db_manager
        self._sql_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'sql'
        )
    
    def _get_db_manager(self):
        """获取数据库管理器"""
        if self.db_manager is None:
            from backend.modules.database.manager import get_database_manager
            self.db_manager = get_database_manager()
        return self.db_manager
    
    def _get_db_type(self) -> str:
        """获取当前数据库类型
        
        Returns:
            str: 数据库类型 (sqlite/postgresql/mysql)
        """
        db_manager = self._get_db_manager()
        db_url = str(db_manager.engine.url)
        
        if 'postgresql' in db_url:
            return 'postgresql'
        elif 'mysql' in db_url:
            return 'mysql'
        else:
            return 'sqlite'
    
    def _get_existing_tables(self) -> Set[str]:
        """获取数据库中已存在的表
        
        Returns:
            Set[str]: 已存在的表名集合
        """
        db_manager = self._get_db_manager()
        inspector = inspect(db_manager.engine)
        return set(inspector.get_table_names())
    
    def initialize(self, force: bool = False) -> Dict[str, Any]:
        """初始化数据库表
        
        Args:
            force: 是否强制重建（删除后重建）
            
        Returns:
            Dict[str, Any]: 初始化结果
        """
        result = {
            'success': False,
            'db_type': None,
            'tables_created': [],
            'tables_existed': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            db_type = self._get_db_type()
            result['db_type'] = db_type
            
            logger.info(f"Initializing database tables for {db_type}...")
            
            # 使用 SQLAlchemy 元数据创建表
            from backend.schemas.base_models import Base
            
            # 加载所有模型
            exporter = SchemaExporter(Base)
            exporter._load_all_models()
            
            # 获取已存在的表
            existing_tables = self._get_existing_tables()
            result['tables_existed'] = list(existing_tables)
            
            # 创建表
            db_manager = self._get_db_manager()
            
            if force:
                # 强制重建：先删除所有表
                logger.warning("Force mode: dropping all tables...")
                Base.metadata.drop_all(bind=db_manager.engine)
                existing_tables = set()
            
            # 使用 create_all 一次性创建所有表和索引
            # checkfirst=True 会跳过已存在的表
            try:
                Base.metadata.create_all(bind=db_manager.engine, checkfirst=True)
                logger.info("Tables created successfully using create_all")
            except Exception as create_all_error:
                error_msg = str(create_all_error)
                # 如果是索引已存在的错误，这是正常的，记录为 DEBUG
                if 'already exists' in error_msg.lower():
                    logger.debug(f"Index already exists (expected): {error_msg}")
                else:
                    # 如果 create_all 失败，回退到逐表创建
                    logger.warning(f"create_all failed, falling back to table-by-table creation: {error_msg}")
                    self._create_tables_individually(Base, db_manager, result)
            
            # 确保所有索引存在（忽略已存在的错误）
            self._ensure_indexes(Base, db_manager, result)
            
            # 获取新创建的表
            new_tables = self._get_existing_tables()
            created_tables = new_tables - existing_tables
            result['tables_created'] = list(created_tables)
            
            # 只有在没有严重错误时才视为成功
            result['success'] = len(result['errors']) == 0
            logger.info(f"Database initialization completed. Created {len(created_tables)} tables.")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            result['errors'].append(str(e))
        
        return result
    
    def _create_tables_individually(self, Base, db_manager, result: Dict):
        """逐表创建表结构
        
        Args:
            Base: SQLAlchemy Base 类
            db_manager: 数据库管理器
            result: 结果字典（会被修改）
        """
        for table in Base.metadata.sorted_tables:
            try:
                table.create(bind=db_manager.engine, checkfirst=True)
            except Exception as table_error:
                error_msg = str(table_error)
                # 如果是索引/表已存在的错误，忽略（DEBUG级别）
                if 'already exists' in error_msg.lower():
                    logger.debug(f"Object already exists for table {table.name}: {error_msg}")
                else:
                    result['errors'].append(f"Table {table.name}: {error_msg}")
                    logger.error(f"Failed to create table {table.name}: {error_msg}")
    
    def _ensure_indexes(self, Base, db_manager, result: Dict):
        """确保所有索引存在
        
        Args:
            Base: SQLAlchemy Base 类
            db_manager: 数据库管理器
            result: 结果字典
        """
        # 获取已存在的索引
        inspector = inspect(db_manager.engine)
        existing_indexes = set()
        
        for table_name in inspector.get_table_names():
            try:
                for idx in inspector.get_indexes(table_name):
                    existing_indexes.add(idx['name'])
            except Exception:
                pass
        
        # 尝试创建缺失的索引
        for table in Base.metadata.sorted_tables:
            for index in table.indexes:
                if index.name and index.name not in existing_indexes:
                    try:
                        index.create(bind=db_manager.engine)
                        logger.debug(f"Created index: {index.name}")
                    except Exception as idx_error:
                        error_msg = str(idx_error)
                        if 'already exists' in error_msg.lower():
                            # 索引已存在，这是正常的
                            logger.debug(f"Index {index.name} already exists")
                        else:
                            logger.debug(f"Could not create index {index.name}: {error_msg}")
    
    def execute_sql_file(self, sql_file: str) -> Dict[str, Any]:
        """执行 SQL 文件
        
        Args:
            sql_file: SQL 文件路径
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        result = {
            'success': False,
            'statements_executed': 0,
            'errors': []
        }
        
        try:
            if not os.path.exists(sql_file):
                raise FileNotFoundError(f"SQL file not found: {sql_file}")
            
            with open(sql_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 分割 SQL 语句
            statements = [s.strip() for s in sql_content.split(';') if s.strip() and not s.strip().startswith('--')]
            
            db_manager = self._get_db_manager()
            
            from sqlalchemy import text
            
            with db_manager.engine.connect() as conn:
                for stmt in statements:
                    if stmt and not stmt.startswith('--'):
                        try:
                            conn.execute(text(stmt))
                            result['statements_executed'] += 1
                        except Exception as e:
                            result['errors'].append(f"Statement failed: {str(e)[:100]}")
                conn.commit()
            
            result['success'] = len(result['errors']) == 0
            
        except Exception as e:
            result['errors'].append(str(e))
        
        return result
    
    def check_schema_sync(self) -> Dict[str, Any]:
        """检查 Schema 同步状态
        
        Returns:
            Dict[str, Any]: 同步状态信息
        """
        result = {
            'in_sync': False,
            'model_tables': [],
            'db_tables': [],
            'missing_tables': [],
            'extra_tables': []
        }
        
        try:
            # 加载模型
            from backend.schemas.base_models import Base
            exporter = SchemaExporter(Base)
            exporter._load_all_models()
            
            model_tables = set(exporter.metadata.tables.keys())
            db_tables = self._get_existing_tables()
            
            result['model_tables'] = sorted(list(model_tables))
            result['db_tables'] = sorted(list(db_tables))
            result['missing_tables'] = sorted(list(model_tables - db_tables))
            result['extra_tables'] = sorted(list(db_tables - model_tables))
            result['in_sync'] = len(result['missing_tables']) == 0
            
        except Exception as e:
            logger.error(f"Schema sync check failed: {e}")
        
        return result


def export_schemas(output_dir: str = None) -> Dict[str, str]:
    """导出所有数据库类型的 Schema
    
    Args:
        output_dir: 输出目录，默认为 data/sql
        
    Returns:
        Dict[str, str]: 导出结果
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'sql'
        )
    
    from backend.schemas.base_models import Base
    exporter = SchemaExporter(Base)
    return exporter.export_all(output_dir)


def initialize_database(force: bool = False) -> Dict[str, Any]:
    """初始化数据库
    
    Args:
        force: 是否强制重建
        
    Returns:
        Dict[str, Any]: 初始化结果
    """
    initializer = DatabaseInitializer()
    return initializer.initialize(force=force)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='数据库 Schema 管理工具')
    parser.add_argument('action', choices=['export', 'init', 'check'],
                       help='操作: export=导出DDL, init=初始化表, check=检查同步')
    parser.add_argument('--output', '-o', default=None,
                       help='输出目录 (export 操作)')
    parser.add_argument('--force', '-f', action='store_true',
                       help='强制重建 (init 操作)')
    
    args = parser.parse_args()
    
    if args.action == 'export':
        print("正在导出数据库 Schema...")
        result = export_schemas(args.output)
        print("\n导出结果:")
        for db_type, path in result.items():
            print(f"  {db_type}: {path}")
    
    elif args.action == 'init':
        print("正在初始化数据库...")
        result = initialize_database(force=args.force)
        print(f"\n初始化结果:")
        print(f"  成功: {result['success']}")
        print(f"  数据库类型: {result['db_type']}")
        print(f"  创建的表: {len(result['tables_created'])}")
        print(f"  已存在的表: {len(result['tables_existed'])}")
        if result['errors']:
            print(f"  错误: {result['errors']}")
    
    elif args.action == 'check':
        print("正在检查 Schema 同步状态...")
        initializer = DatabaseInitializer()
        result = initializer.check_schema_sync()
        print(f"\n同步状态:")
        print(f"  同步: {'是' if result['in_sync'] else '否'}")
        print(f"  模型表数: {len(result['model_tables'])}")
        print(f"  数据库表数: {len(result['db_tables'])}")
        if result['missing_tables']:
            print(f"  缺失的表: {result['missing_tables']}")
        if result['extra_tables']:
            print(f"  多余的表: {result['extra_tables']}")

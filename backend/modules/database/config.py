"""数据库配置管理

提供数据库配置加载和管理功能。
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str = "localhost"
    port: int = 5432
    database: str = "vectorsphere"
    username: str = "vectorsphere"
    password: str = "vectorsphere"
    url: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    type: str = "postgresql"  # 数据库类型
    
    @classmethod
    def from_env(cls, prefix: str = "DB_") -> "DatabaseConfig":
        """从环境变量创建配置"""
        return cls(
            host=os.environ.get(f"{prefix}HOST", "localhost"),
            port=int(os.environ.get(f"{prefix}PORT", "5432")),
            database=os.environ.get(f"{prefix}NAME", "vectorsphere"),
            username=os.environ.get(f"{prefix}USER", "postgres"),
            password=os.environ.get(f"{prefix}PASSWORD", "password"),
            url=os.environ.get(f"{prefix}URL"),
            pool_size=int(os.environ.get(f"{prefix}POOL_SIZE", "10")),
            max_overflow=int(os.environ.get(f"{prefix}MAX_OVERFLOW", "20")),
            pool_timeout=int(os.environ.get(f"{prefix}POOL_TIMEOUT", "30")),
            pool_recycle=int(os.environ.get(f"{prefix}POOL_RECYCLE", "3600")),
            echo=os.environ.get(f"{prefix}ECHO", "false").lower() == "true",
            type=os.environ.get(f"{prefix}TYPE", "postgresql")
        )
    
    @classmethod
    def from_config_manager(cls) -> "DatabaseConfig":
        """从配置管理器创建配置
        
        优先级：环境变量 > database.yaml > 默认值
        """
        import yaml
        
        # 1. 首先尝试从环境变量加载
        db_type = os.environ.get("DB_TYPE")
        if db_type:
            # 环境变量优先
            return cls.from_env()
        
        # 2. 尝试加载 database.yaml 配置文件
        try:
            # 获取项目根目录
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            config_file = os.path.join(project_root, 'config', 'database.yaml')
            
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
                
                database_config = config_data.get("database", {})
                
                if database_config:
                    db_type = database_config.get("type", "postgresql")
                    if db_type == "sqlite":
                        # SQLite配置
                        db_name = database_config.get("name", "vectorsphere.db")
                        return cls(
                            database=db_name,
                            type="sqlite",
                            url=f"sqlite:///{db_name}"
                        )
                    else:
                        # PostgreSQL或其他数据库配置
                        return cls(
                            host=database_config.get("host", "localhost"),
                            port=int(database_config.get("port", 5432)),
                            database=database_config.get("name", "vectorsphere"),
                            username=database_config.get("username", "postgres"),
                            password=database_config.get("password", "password"),
                            pool_size=int(database_config.get("pool_size", 10)),
                            max_overflow=int(database_config.get("max_overflow", 20)),
                            pool_timeout=int(database_config.get("pool_timeout", 30)),
                            pool_recycle=int(database_config.get("pool_recycle", 3600)),
                            echo=database_config.get("echo", False),
                            type=db_type
                        )
        except Exception as e:
            print(f"加载 database.yaml 失败: {e}")
        
        # 3. 回退到环境变量配置
        return cls.from_env()
    
    @property
    def connection_url(self) -> str:
        """获取连接URL"""
        if self.url:
            return self.url
        if self.type == "sqlite":
            return f"sqlite:///{self.database}"
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


# 全局数据库配置
_global_db_config: Optional[DatabaseConfig] = None


def get_database_config() -> DatabaseConfig:
    """获取全局数据库配置"""
    global _global_db_config
    if _global_db_config is None:
        _global_db_config = DatabaseConfig.from_config_manager()
    return _global_db_config


def set_database_config(config: DatabaseConfig):
    """设置全局数据库配置"""
    global _global_db_config
    _global_db_config = config
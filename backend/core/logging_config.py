"""日志配置模块

提供统一的日志配置和管理功能。
"""

import os
import logging
import logging.handlers
from typing import Optional
from pathlib import Path

# 默认日志格式
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logging(
    level: str = "INFO",
    log_format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    log_file: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> None:
    """设置日志配置
    
    Args:
        level: 日志级别
        log_format: 日志格式
        date_format: 日期格式
        log_file: 日志文件路径
        max_bytes: 日志文件最大字节数
        backup_count: 日志文件备份数量
    """
    # 设置日志级别
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter(log_format, date_format)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 结构化JSON日志（可选，通过环境变量启用）
    try:
        if os.environ.get('LOG_STRUCTURED', 'false').lower() == 'true':
            from pythonjsonlogger import jsonlogger
            json_formatter = jsonlogger.JsonFormatter('(asctime) (name) (levelname) (message)')
            console_handler.setFormatter(json_formatter)
    except Exception as e:
        logging.warning(f"结构化日志初始化失败，回退到标准格式: {e}")
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        try:
            # 确保日志目录存在
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 创建轮转文件处理器
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
        except Exception as e:
            logging.warning(f"无法设置文件日志: {e}")
    
    # 配置第三方库的日志级别
    _configure_third_party_loggers()


def _configure_third_party_loggers() -> None:
    """配置第三方库的日志级别"""
    # SQLAlchemy日志
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.dialects').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.orm').setLevel(logging.WARNING)
    
    # Werkzeug日志
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    
    # Redis日志
    logging.getLogger('redis').setLevel(logging.WARNING)
    
    # Flask-Limiter日志
    logging.getLogger('flask_limiter').setLevel(logging.WARNING)
    
    # 其他常见库的日志
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器实例
    """
    return logging.getLogger(name)
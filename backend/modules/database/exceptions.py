"""数据库异常类

定义数据库操作过程中可能发生的异常。
"""

from typing import Optional


class DatabaseException(Exception):
    """数据库基础异常类"""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class DatabaseConnectionException(DatabaseException):
    """数据库连接异常"""
    
    def __init__(self, message: str = "数据库连接失败"):
        super().__init__(message, "DB_CONNECTION_ERROR")


class DatabaseQueryException(DatabaseException):
    """数据库查询异常"""
    
    def __init__(self, message: str = "数据库查询失败"):
        super().__init__(message, "DB_QUERY_ERROR")


class DatabaseTransactionException(DatabaseException):
    """数据库事务异常"""
    
    def __init__(self, message: str = "数据库事务失败"):
        super().__init__(message, "DB_TRANSACTION_ERROR")


class DatabaseModelNotFoundException(DatabaseException):
    """数据库模型未找到异常"""
    
    def __init__(self, model_name: str, identifier: str):
        message = f"数据库模型 {model_name} 未找到，标识符: {identifier}"
        super().__init__(message, "DB_MODEL_NOT_FOUND")


class DatabaseConstraintException(DatabaseException):
    """数据库约束异常"""
    
    def __init__(self, message: str = "数据库约束违反"):
        super().__init__(message, "DB_CONSTRAINT_ERROR")


class DatabaseConfigurationException(DatabaseException):
    """数据库配置异常"""
    
    def __init__(self, message: str = "数据库配置错误"):
        super().__init__(message, "DB_CONFIG_ERROR")


__all__ = [
    'DatabaseException',
    'DatabaseConnectionException',
    'DatabaseQueryException',
    'DatabaseTransactionException',
    'DatabaseModelNotFoundException',
    'DatabaseConstraintException',
    'DatabaseConfigurationException',
]
"""模型模块异常定义

定义模型模块相关的自定义异常。
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.exceptions import ValidationError, ResourceNotFoundError, BusinessLogicError


class ModelError(BusinessLogicError):
    """模型模块基础异常"""
    pass


class ModelNotFoundError(ModelError, ResourceNotFoundError):
    """模型不存在异常"""
    pass


class ModelValidationError(ModelError, ValidationError):
    """模型验证异常"""
    pass


class ModelBusinessLogicError(ModelError, BusinessLogicError):
    """模型业务逻辑异常"""
    pass
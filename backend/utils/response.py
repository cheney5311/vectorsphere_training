"""API响应工具

提供统一的API响应格式。
"""

from typing import Any, Optional, Dict
from flask import jsonify
import time


def success_response(data: Any = None, message: str = "操作成功", code: int = 200) -> tuple:
    """成功响应
    
    Args:
        data: 响应数据
        message: 响应消息
        code: HTTP状态码
        
    Returns:
        Flask响应元组
    """
    response = {
        "success": True,
        "code": code,
        "message": message,
        "data": data,
        "timestamp": time.time()
    }
    return jsonify(response), code


def error_response(message: str = "操作失败", code: int = 400, error_type: str = "UNKNOWN_ERROR", details: Optional[Dict[str, Any]] = None) -> tuple:
    """错误响应
    
    Args:
        message: 错误消息
        code: HTTP状态码
        error_type: 错误类型
        details: 错误详情
        
    Returns:
        Flask响应元组
    """
    response = {
        "success": False,
        "code": code,
        "message": message,
        "error_type": error_type,
        "details": details,
        "timestamp": time.time()
    }
    
    return jsonify(response), code


def paginated_response(items: list, page: int, limit: int, total: int, message: str = "获取成功") -> tuple:
    """分页响应格式
    
    Args:
        items: 数据项列表
        page: 当前页码
        limit: 每页限制
        total: 总数量
        message: 响应消息
        
    Returns:
        标准化的分页响应
    """
    pagination = {
        "page": page,
        "limit": limit,
        "total": total,
        "pages": (total + limit - 1) // limit,
        "has_next": page * limit < total,
        "has_prev": page > 1
    }
    
    data = {
        "items": items,
        "pagination": pagination
    }
    
    return success_response(data, message)
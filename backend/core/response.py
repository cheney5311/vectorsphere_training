"""API响应格式化模块

提供统一的API响应格式和工具函数。
"""
import logging
from typing import Any, Dict, Optional, List, Union
from flask import jsonify, g
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """响应格式化器"""
    
    @staticmethod
    def success(data: Any = None, 
                message: str = "Success", 
                meta: Optional[Dict[str, Any]] = None,
                status_code: int = 200) -> tuple:
        """
        创建成功响应
        
        Args:
            data: 响应数据
            message: 响应消息
            meta: 元数据信息
            status_code: HTTP状态码
            
        Returns:
            (response_dict, status_code)
        """
        response = {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": getattr(g, 'request_id', None)
        }
        
        if meta:
            response["meta"] = meta
            
        return jsonify(response), status_code
    
    @staticmethod
    def error(error_code: str,
              message: str,
              details: Optional[Dict[str, Any]] = None,
              status_code: int = 400) -> tuple:
        """
        创建错误响应
        
        Args:
            error_code: 错误代码
            message: 错误消息
            details: 错误详情
            status_code: HTTP状态码
            
        Returns:
            (response_dict, status_code)
        """
        response = {
            "success": False,
            "error": {
                "code": error_code,
                "message": message,
                "details": details or {}
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": getattr(g, 'request_id', None)
        }
        
        return jsonify(response), status_code
    
    @staticmethod
    def paginated(data: List[Any],
                  page: int,
                  per_page: int,
                  total: int,
                  message: str = "Success") -> tuple:
        """
        创建分页响应
        
        Args:
            data: 分页数据
            page: 当前页码
            per_page: 每页数量
            total: 总数量
            message: 响应消息
            
        Returns:
            (response_dict, status_code)
        """
        total_pages = (total + per_page - 1) // per_page
        
        meta = {
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }
        
        return ResponseFormatter.success(data, message, meta)
    
    @staticmethod
    def created(data: Any = None, 
                message: str = "Resource created successfully",
                location: Optional[str] = None) -> tuple:
        """
        创建资源创建成功响应
        
        Args:
            data: 创建的资源数据
            message: 响应消息
            location: 资源位置URL
            
        Returns:
            (response_dict, status_code)
        """
        response, _ = ResponseFormatter.success(data, message, status_code=201)
        
        if location:
            response.headers['Location'] = location
            
        return response, 201
    
    @staticmethod
    def no_content(message: str = "Operation completed successfully") -> tuple:
        """
        创建无内容响应
        
        Args:
            message: 响应消息
            
        Returns:
            (response_dict, status_code)
        """
        response = {
            "success": True,
            "message": message,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request_id": getattr(g, 'request_id', None)
        }
        
        return jsonify(response), 204


class APIResponse:
    """API响应构建器"""
    
    def __init__(self):
        self.data = None
        self.message = "Success"
        self.meta = {}
        self.status_code = 200
        self.headers = {}
    
    def set_data(self, data: Any) -> 'APIResponse':
        """设置响应数据"""
        self.data = data
        return self
    
    def set_message(self, message: str) -> 'APIResponse':
        """设置响应消息"""
        self.message = message
        return self
    
    def set_meta(self, meta: Dict[str, Any]) -> 'APIResponse':
        """设置元数据"""
        self.meta.update(meta)
        return self
    
    def set_status_code(self, status_code: int) -> 'APIResponse':
        """设置状态码"""
        self.status_code = status_code
        return self
    
    def add_header(self, key: str, value: str) -> 'APIResponse':
        """添加响应头"""
        self.headers[key] = value
        return self
    
    def build(self) -> tuple:
        """构建响应"""
        response, status_code = ResponseFormatter.success(
            self.data, self.message, self.meta, self.status_code
        )
        
        # 添加自定义头
        for key, value in self.headers.items():
            response.headers[key] = value
            
        return response, status_code


def success_response(data: Any = None, 
                    message: str = "Success",
                    status_code: int = 200) -> tuple:
    """快捷成功响应函数"""
    return ResponseFormatter.success(data, message, status_code=status_code)


def error_response(error_code: str,
                  message: str,
                  details: Optional[Dict[str, Any]] = None,
                  status_code: int = 400) -> tuple:
    """快捷错误响应函数"""
    return ResponseFormatter.error(error_code, message, details, status_code)


def paginated_response(data: List[Any],
                      page: int,
                      per_page: int,
                      total: int) -> tuple:
    """快捷分页响应函数"""
    return ResponseFormatter.paginated(data, page, per_page, total)


def created_response(data: Any = None,
                    message: str = "Resource created successfully",
                    location: Optional[str] = None) -> tuple:
    """快捷创建响应函数"""
    return ResponseFormatter.created(data, message, location)


def no_content_response(message: str = "Operation completed successfully") -> tuple:
    """快捷无内容响应函数"""
    return ResponseFormatter.no_content(message)


class ResponseValidator:
    """响应验证器"""
    
    @staticmethod
    def validate_response_data(data: Any, max_size: int = 10 * 1024 * 1024) -> bool:
        """
        验证响应数据大小
        
        Args:
            data: 响应数据
            max_size: 最大大小（字节）
            
        Returns:
            是否有效
        """
        try:
            serialized = json.dumps(data, default=str)
            size = len(serialized.encode('utf-8'))
            
            if size > max_size:
                logger.warning(f"Response data too large: {size} bytes")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to validate response data: {e}")
            return False
    
    @staticmethod
    def sanitize_response_data(data: Any) -> Any:
        """
        清理响应数据，移除敏感信息
        
        Args:
            data: 原始数据
            
        Returns:
            清理后的数据
        """
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # 移除敏感字段
                if key.lower() in ['password', 'token', 'secret', 'key', 'private']:
                    sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = ResponseValidator.sanitize_response_data(value)
            return sanitized
        
        elif isinstance(data, list):
            return [ResponseValidator.sanitize_response_data(item) for item in data]
        
        else:
            return data


def build_response() -> APIResponse:
    """创建响应构建器"""
    return APIResponse()


# 响应装饰器
def format_response(success_message: str = "Success"):
    """
    响应格式化装饰器
    
    Args:
        success_message: 成功消息
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            try:
                result = f(*args, **kwargs)
                
                # 如果函数已经返回了格式化的响应，直接返回
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                
                # 否则格式化响应
                return success_response(result, success_message)
                
            except Exception as e:
                logger.error(f"Error in {f.__name__}: {e}", exc_info=True)
                return error_response(
                    "INTERNAL_ERROR",
                    "An unexpected error occurred",
                    status_code=500
                )
        
        return wrapper
    return decorator


def validate_response_size(max_size: int = 10 * 1024 * 1024):
    """
    响应大小验证装饰器
    
    Args:
        max_size: 最大响应大小（字节）
    """
    def decorator(f):
        def wrapper(*args, **kwargs):
            response, status_code = f(*args, **kwargs)
            
            # 验证响应大小
            if hasattr(response, 'get_data'):
                data_size = len(response.get_data())
                if data_size > max_size:
                    logger.warning(f"Response too large: {data_size} bytes")
                    return error_response(
                        "RESPONSE_TOO_LARGE",
                        "Response data too large",
                        status_code=413
                    )
            
            return response, status_code
        
        return wrapper
    return decorator
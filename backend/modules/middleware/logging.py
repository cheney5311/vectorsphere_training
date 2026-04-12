"""请求日志中间件模块

实现API请求的日志记录、性能监控和审计功能
"""

import logging
import time
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from flask import Flask, request, g, current_app
from werkzeug.exceptions import HTTPException

from backend.modules.middleware.auth import get_current_user

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware:
    """请求日志中间件"""
    
    def __init__(self, app: Optional[Flask] = None):
        """初始化中间件
        
        Args:
            app: Flask应用实例
        """
        self.app = app
        self.request_logs = []  # 内存中的请求日志（生产环境应使用数据库或消息队列）
        
        if app:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """初始化Flask应用
        
        Args:
            app: Flask应用实例
        """
        self.app = app
        
        # 注册中间件
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        app.teardown_request(self.teardown_request)
    
    def before_request(self):
        """请求前处理"""
        try:
            # 生成请求ID
            request_id = str(uuid.uuid4())
            g.request_id = request_id
            
            # 记录请求开始时间
            g.request_start_time = time.time()
            
            # 记录请求信息
            g.request_info = {
                'request_id': request_id,
                'method': request.method,
                'url': request.url,
                'path': request.path,
                'remote_addr': self._get_client_ip(),
                'user_agent': request.headers.get('User-Agent', ''),
                'content_type': request.headers.get('Content-Type', ''),
                'content_length': request.headers.get('Content-Length', 0),
                'timestamp': datetime.now().isoformat(),
                'headers': dict(request.headers),
                'args': dict(request.args),
                'form': dict(request.form) if request.form else None,
                'json': self._safe_get_json(),
                'user_id': None,
                'tenant_id': None
            }
            
            # 过滤敏感信息
            self._filter_sensitive_data(g.request_info)
            
            # 记录请求开始日志
            if self._should_log_request():
                logger.info(f"请求开始: {request.method} {request.path}", extra={
                    'request_id': request_id,
                    'method': request.method,
                    'path': request.path,
                    'client_ip': g.request_info['remote_addr']
                })
            
        except Exception as e:
            logger.error(f"请求前处理失败: {e}")
    
    def after_request(self, response):
        """请求后处理
        
        Args:
            response: 响应对象
            
        Returns:
            响应对象
        """
        try:
            # 计算请求处理时间
            if hasattr(g, 'request_start_time'):
                duration = time.time() - g.request_start_time
                g.request_duration = duration
            else:
                g.request_duration = 0
            
            # 获取用户信息
            user = get_current_user()
            
            # 更新请求信息
            if hasattr(g, 'request_info'):
                g.request_info.update({
                    'user_id': user.id if user else None,
                    'status_code': response.status_code,
                    'response_size': len(response.get_data()),
                    'duration': g.request_duration,
                    'response_headers': dict(response.headers)
                })
                
                # 记录响应日志
                if self._should_log_request():
                    log_level = self._get_log_level(response.status_code)
                    logger.log(log_level, f"请求完成: {request.method} {request.path} - {response.status_code} ({g.request_duration:.3f}s)", extra={
                        'request_id': g.request_info['request_id'],
                        'method': request.method,
                        'path': request.path,
                        'status_code': response.status_code,
                        'duration': g.request_duration,
                        'user_id': g.request_info['user_id']
                    })
                
                # 保存请求日志
                self._save_request_log(g.request_info)
            
            # 添加响应头
            if hasattr(g, 'request_id'):
                response.headers['X-Request-ID'] = g.request_id
            
            response.headers['X-Response-Time'] = f"{g.request_duration:.3f}s"
            
        except Exception as e:
            logger.error(f"请求后处理失败: {e}")
        
        return response
    
    def teardown_request(self, exception):
        """请求清理处理
        
        Args:
            exception: 异常对象
        """
        try:
            # 记录异常信息
            if exception:
                error_info = {
                    'exception_type': type(exception).__name__,
                    'exception_message': str(exception),
                    'is_http_exception': isinstance(exception, HTTPException)
                }
                
                if hasattr(g, 'request_info'):
                    g.request_info['exception'] = error_info
                
                # 记录异常日志
                if hasattr(g, 'request_id'):
                    logger.error(f"请求异常: {request.method} {request.path}", extra={
                        'request_id': g.request_id,
                        'exception_type': error_info['exception_type'],
                        'exception_message': error_info['exception_message']
                    }, exc_info=True)
            
            # 清理全局变量
            for attr in ['request_id', 'request_start_time', 'request_duration', 'request_info']:
                if hasattr(g, 'attr'):
                    delattr(g, attr)
                    
        except Exception as e:
            logger.error(f"请求清理处理失败: {e}")
    
    def _get_client_ip(self) -> str:
        """获取客户端IP地址
        
        Returns:
            客户端IP地址
        """
        # 检查代理头
        if request.headers.get('X-Forwarded-For'):
            return request.headers.get('X-Forwarded-For').split(',')[0].strip()
        elif request.headers.get('X-Real-IP'):
            return request.headers.get('X-Real-IP')
        elif request.headers.get('X-Forwarded-Host'):
            return request.headers.get('X-Forwarded-Host')
        else:
            return request.remote_addr or 'unknown'
    
    def _safe_get_json(self) -> Optional[Dict[str, Any]]:
        """安全获取JSON数据
        
        Returns:
            JSON数据或None
        """
        try:
            if request.is_json:
                return request.get_json(silent=True)
        except Exception:
            pass
        return None
    
    def _filter_sensitive_data(self, data: Dict[str, Any]):
        """过滤敏感数据
        
        Args:
            data: 数据字典
        """
        sensitive_keys = {
            'password', 'passwd', 'pwd', 'secret', 'token', 'key', 'auth',
            'authorization', 'x-api-key', 'cookie', 'session'
        }
        
        def filter_dict(d):
            if isinstance(d, dict):
                for key, value in d.items():
                    if isinstance(key, str) and key.lower() in sensitive_keys:
                        d[key] = '[FILTERED]'
                    elif isinstance(value, dict):
                        filter_dict(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                filter_dict(item)
        
        # 过滤headers
        if 'headers' in data:
            filter_dict(data['headers'])
        
        # 过滤args
        if 'args' in data:
            filter_dict(data['args'])
        
        # 过滤form
        if 'form' in data and data['form']:
            filter_dict(data['form'])
        
        # 过滤json
        if 'json' in data and data['json']:
            filter_dict(data['json'])
    
    def _should_log_request(self) -> bool:
        """检查是否应该记录请求
        
        Returns:
            是否记录
        """
        # 跳过的路径
        skip_paths = [
            '/health',
            '/metrics',
            '/favicon.ico',
            '/static'
        ]
        
        path = request.path
        
        # 检查跳过路径
        for skip_path in skip_paths:
            if path.startswith(skip_path):
                return False
        
        return True
    
    def _get_log_level(self, status_code: int) -> int:
        """根据状态码获取日志级别
        
        Args:
            status_code: HTTP状态码
            
        Returns:
            日志级别
        """
        if status_code >= 500:
            return logging.ERROR
        elif status_code >= 400:
            return logging.WARNING
        else:
            return logging.INFO
    
    def _save_request_log(self, request_info: Dict[str, Any]):
        """保存请求日志
        
        Args:
            request_info: 请求信息
        """
        try:
            # 在生产环境中，这里应该保存到数据库或发送到日志系统
            # 这里只是简单地保存到内存中
            self.request_logs.append(request_info.copy())
            
            # 限制内存中的日志数量
            if len(self.request_logs) > 1000:
                self.request_logs = self.request_logs[-500:]
            
            # 记录审计日志
            if self._is_audit_worthy(request_info):
                self._log_audit_event(request_info)
                
        except Exception as e:
            logger.error(f"保存请求日志失败: {e}")
    
    def _is_audit_worthy(self, request_info: Dict[str, Any]) -> bool:
        """检查是否值得审计
        
        Args:
            request_info: 请求信息
            
        Returns:
            是否值得审计
        """
        # 审计条件
        audit_conditions = [
            # 认证相关操作
            request_info['path'].startswith('/api/v1/auth/'),
            # 管理员操作
            request_info['path'].startswith('/api/v1/admin/'),
            # 用户管理操作
            request_info['path'].startswith('/api/v1/users/') and request_info['method'] in ['POST', 'PUT', 'DELETE'],
            # 训练操作
            request_info['path'].startswith('/api/v1/training/') and request_info['method'] in ['POST', 'PUT', 'DELETE'],
            # 模型操作
            request_info['path'].startswith('/api/v1/models/') and request_info['method'] in ['POST', 'PUT', 'DELETE'],
            # 数据集操作
            request_info['path'].startswith('/api/v1/datasets/') and request_info['method'] in ['POST', 'PUT', 'DELETE'],
            # 错误响应
            request_info.get('status_code', 200) >= 400,
            # 异常情况
            'exception' in request_info
        ]
        
        return any(audit_conditions)
    
    def _log_audit_event(self, request_info: Dict[str, Any]):
        """记录审计事件
        
        Args:
            request_info: 请求信息
        """
        try:
            audit_data = {
                'event_type': 'api_request',
                'timestamp': request_info['timestamp'],
                'request_id': request_info['request_id'],
                'user_id': request_info.get('user_id'),
                'method': request_info['method'],
                'path': request_info['path'],
                'status_code': request_info.get('status_code'),
                'client_ip': request_info['remote_addr'],
                'user_agent': request_info['user_agent'],
                'duration': request_info.get('duration'),
                'exception': request_info.get('exception')
            }
            
            # 使用专门的审计日志记录器
            audit_logger = logging.getLogger('audit')
            audit_logger.info(f"审计事件: {audit_data['event_type']}", extra=audit_data)
            
        except Exception as e:
            logger.error(f"记录审计事件失败: {e}")
    
    def get_request_logs(self, limit: int = 100, offset: int = 0,
                        user_id: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """获取请求日志
        
        Args:
            limit: 限制数量
            offset: 偏移量
            user_id: 用户ID过滤
            start_time: 开始时间过滤
            end_time: 结束时间过滤
            
        Returns:
            请求日志列表
        """
        try:
            logs = self.request_logs.copy()
            
            # 应用过滤条件
            if user_id:
                logs = [log for log in logs if log.get('user_id') == user_id]
            
            if start_time:
                logs = [log for log in logs 
                       if datetime.fromisoformat(log['timestamp']) >= start_time]
            
            if end_time:
                logs = [log for log in logs 
                       if datetime.fromisoformat(log['timestamp']) <= end_time]
            
            # 排序（最新的在前）
            logs.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # 分页
            return logs[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"获取请求日志失败: {e}")
            return []
    
    def get_request_stats(self, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """获取请求统计信息
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            统计信息
        """
        try:
            logs = self.request_logs.copy()
            
            # 应用时间过滤
            if start_time:
                logs = [log for log in logs 
                       if datetime.fromisoformat(log['timestamp']) >= start_time]
            
            if end_time:
                logs = [log for log in logs 
                       if datetime.fromisoformat(log['timestamp']) <= end_time]
            
            if not logs:
                return {
                    'total_requests': 0,
                    'avg_duration': 0,
                    'status_codes': {},
                    'methods': {},
                    'paths': {},
                    'error_rate': 0
                }
            
            # 计算统计信息
            total_requests = len(logs)
            durations = [log.get('duration', 0) for log in logs]
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # 状态码统计
            status_codes = {}
            for log in logs:
                status = log.get('status_code', 'unknown')
                status_codes[status] = status_codes.get(status, 0) + 1
            
            # 方法统计
            methods = {}
            for log in logs:
                method = log.get('method', 'unknown')
                methods[method] = methods.get(method, 0) + 1
            
            # 路径统计
            paths = {}
            for log in logs:
                path = log.get('path', 'unknown')
                paths[path] = paths.get(path, 0) + 1
            
            # 错误率
            error_count = sum(1 for log in logs if log.get('status_code', 200) >= 400)
            error_rate = error_count / total_requests if total_requests > 0 else 0
            
            return {
                'total_requests': total_requests,
                'avg_duration': avg_duration,
                'status_codes': status_codes,
                'methods': methods,
                'paths': dict(sorted(paths.items(), key=lambda x: x[1], reverse=True)[:10]),  # 前10个路径
                'error_rate': error_rate
            }
            
        except Exception as e:
            logger.error(f"获取请求统计失败: {e}")
            return {}


# 工具函数
def get_request_id() -> Optional[str]:
    """获取当前请求ID
    
    Returns:
        请求ID
    """
    return getattr(g, 'request_id', None)


def get_request_duration() -> float:
    """获取当前请求处理时间
    
    Returns:
        处理时间（秒）
    """
    return getattr(g, 'request_duration', 0)


# 全局中间件实例
_logging_middleware = None


def get_logging_middleware() -> RequestLoggingMiddleware:
    """获取日志中间件实例
    
    Returns:
        日志中间件实例
    """
    global _logging_middleware
    if _logging_middleware is None:
        _logging_middleware = RequestLoggingMiddleware()
    return _logging_middleware


def init_logging_middleware(app: Flask) -> RequestLoggingMiddleware:
    """初始化日志中间件
    
    Args:
        app: Flask应用实例
        
    Returns:
        日志中间件实例
    """
    global _logging_middleware
    _logging_middleware = RequestLoggingMiddleware(app)
    return _logging_middleware
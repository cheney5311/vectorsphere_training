"""
WebSocket事件处理器

提供完整的 WebSocket 事件处理能力，包括：
- 智能体执行与状态管理
- 工作流执行与进度追踪
- 训练任务管理与监控
- 对话消息处理
- 系统状态与告警
- 房间/频道管理
- 心跳与健康检查
"""

import logging
import time
import asyncio
import threading
import uuid
import json
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# 辅助工具和装饰器
# ============================================================================

class RateLimiter:
    """简单的速率限制器"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def is_allowed(self, client_id: str) -> bool:
        """检查客户端是否被允许发送请求"""
        current_time = time.time()
        
        with self._lock:
            # 清理过期请求记录
            self._requests[client_id] = [
                t for t in self._requests[client_id] 
                if current_time - t < self.window_seconds
            ]
            
            # 检查是否超过限制
            if len(self._requests[client_id]) >= self.max_requests:
                return False
            
            # 记录请求
            self._requests[client_id].append(current_time)
            return True
    
    def get_remaining(self, client_id: str) -> int:
        """获取剩余可用请求数"""
        current_time = time.time()
        
        with self._lock:
            valid_requests = [
                t for t in self._requests.get(client_id, [])
                if current_time - t < self.window_seconds
            ]
            return max(0, self.max_requests - len(valid_requests))


@dataclass
class EventContext:
    """事件上下文"""
    event_id: str
    event_name: str
    user_id: Optional[str]
    socket_id: Optional[str]
    timestamp: float
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, event_name: str, data: Dict[str, Any], 
               socket_id: str = None, user_id: str = None) -> 'EventContext':
        return cls(
            event_id=str(uuid.uuid4()),
            event_name=event_name,
            user_id=user_id,
            socket_id=socket_id,
            timestamp=time.time(),
            data=data
        )


def with_error_handling(func: Callable) -> Callable:
    """错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in handler {func.__name__}: {e}")
            return {
                'success': False, 
                'error': str(e),
                'error_code': 'HANDLER_ERROR'
            }
    return wrapper


def with_rate_limit(limiter: RateLimiter):
    """速率限制装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(data, *args, **kwargs):
            from flask import request
            try:
                socket_id = request.sid if request else 'unknown'
            except RuntimeError:
                socket_id = 'unknown'
            
            if not limiter.is_allowed(socket_id):
                return {
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'error_code': 'RATE_LIMIT_EXCEEDED',
                    'retry_after': limiter.window_seconds
                }
            return func(data, *args, **kwargs)
        return wrapper
    return decorator


def with_validation(required_fields: List[str]):
    """参数验证装饰器"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(data, *args, **kwargs):
            if not isinstance(data, dict):
                return {
                    'success': False,
                    'error': 'Invalid data format, expected object',
                    'error_code': 'INVALID_DATA_FORMAT'
                }
            
            missing_fields = [f for f in required_fields if f not in data or data[f] is None]
            if missing_fields:
                return {
                    'success': False,
                    'error': f'Missing required fields: {", ".join(missing_fields)}',
                    'error_code': 'MISSING_FIELDS',
                    'missing_fields': missing_fields
                }
            return func(data, *args, **kwargs)
        return wrapper
    return decorator


# ============================================================================
# WebSocket 事件处理器类
# ============================================================================

class WebSocketHandlers:
    """WebSocket 事件处理器管理类"""
    
    def __init__(self):
        self._registered = False
        self._rate_limiter = RateLimiter(max_requests=100, window_seconds=60)
        self._event_handlers: Dict[str, Callable] = {}
        self._custom_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # 服务实例缓存
        self._agent_manager = None
        self._training_service = None
        self._monitoring_service = None
        self._performance_monitor = None
    
    # ========== 服务获取器 ==========
    
    def _get_agent_manager(self):
        """获取智能体实例管理器"""
        if self._agent_manager is None:
            try:
                from backend.services.agent_instance_manager import agent_instance_manager
                self._agent_manager = agent_instance_manager
            except ImportError:
                logger.warning("AgentInstanceManager not available")
        return self._agent_manager
    
    def _get_training_service(self):
        """获取训练服务"""
        if self._training_service is None:
            try:
                from backend.services.enhanced_training_service import EnhancedTrainingService
                self._training_service = EnhancedTrainingService()
            except ImportError:
                logger.warning("EnhancedTrainingService not available")
        return self._training_service
    
    def _get_monitoring_service(self):
        """获取监控服务"""
        if self._monitoring_service is None:
            try:
                from backend.services.monitoring_service import MonitoringService
                self._monitoring_service = MonitoringService()
            except ImportError:
                logger.warning("MonitoringService not available")
        return self._monitoring_service
    
    def _get_performance_monitor(self):
        """获取性能监控器"""
        if self._performance_monitor is None:
            try:
                from backend.services.performance_monitor import PerformanceMonitor
                self._performance_monitor = PerformanceMonitor
            except ImportError:
                logger.warning("PerformanceMonitor not available")
        return self._performance_monitor
    
    def _get_websocket_manager(self):
        """获取 WebSocket 管理器"""
        from backend.websocket.manager import websocket_manager
        return websocket_manager
    
    def _get_current_user_id(self, data: Dict[str, Any]) -> Optional[str]:
        """从数据或上下文中获取用户ID"""
        # 优先从数据中获取
        if data and 'user_id' in data:
            return data['user_id']
        
        # 尝试从连接信息中获取
        try:
            from flask import request
            ws_manager = self._get_websocket_manager()
            connection = ws_manager.get_connection(request.sid)
            if connection:
                return connection.user_id
        except Exception:
            pass
        
        return None
    
    def _get_current_socket_id(self) -> Optional[str]:
        """获取当前 Socket ID"""
        try:
            from flask import request
            return request.sid
        except Exception:
            return None
    
    # ========== 事件导入器 ==========
    
    def _import_events(self):
        """导入事件类型和创建函数"""
        from backend.websocket.events import (
            EventType, WebSocketEvent,
            create_agent_event, create_workflow_event,
            create_training_event, create_dialogue_event,
            create_system_event
        )
        return {
            'EventType': EventType,
            'WebSocketEvent': WebSocketEvent,
            'create_agent_event': create_agent_event,
            'create_workflow_event': create_workflow_event,
            'create_training_event': create_training_event,
            'create_dialogue_event': create_dialogue_event,
            'create_system_event': create_system_event
        }
    
    # ========== 处理器注册 ==========
    
    def register_handlers(self):
        """注册所有 WebSocket 事件处理器"""
        ws_manager = self._get_websocket_manager()
        
        if not ws_manager.socketio:
            logger.warning("SocketIO not initialized, handlers not registered")
            return False
        
        if self._registered:
            logger.debug("Handlers already registered")
            return True
        
        socketio = ws_manager.socketio
        
        # 注册各类事件处理器
        self._register_agent_handlers(socketio)
        self._register_workflow_handlers(socketio)
        self._register_training_handlers(socketio)
        self._register_dialogue_handlers(socketio)
        self._register_system_handlers(socketio)
        self._register_room_handlers(socketio)
        self._register_subscription_handlers(socketio)
        self._register_utility_handlers(socketio)
        
        self._registered = True
        logger.info("WebSocket event handlers registered successfully")
        return True
    
    def register_custom_handler(self, event_name: str, handler: Callable):
        """注册自定义事件处理器
        
        Args:
            event_name: 事件名称
            handler: 处理器函数
        """
        self._custom_handlers[event_name].append(handler)
        
        # 如果已经注册了 socketio，直接添加处理器
        ws_manager = self._get_websocket_manager()
        if ws_manager.socketio and self._registered:
            @ws_manager.socketio.on(event_name)
            def custom_handler(data):
                return self._handle_custom_event(event_name, data)
            
            logger.debug(f"Custom handler registered for event: {event_name}")
    
    def _handle_custom_event(self, event_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理自定义事件"""
        handlers = self._custom_handlers.get(event_name, [])
        results = []
        
        for handler in handlers:
            try:
                result = handler(data)
                results.append(result)
            except Exception as e:
                logger.error(f"Custom handler error for {event_name}: {e}")
                results.append({'error': str(e)})
        
        return {'success': True, 'results': results}
    
    # ========== 智能体事件处理器 ==========
    
    def _register_agent_handlers(self, socketio):
        """注册智能体相关事件处理器"""
        
        @socketio.on('agent_execute')
        @with_error_handling
        @with_rate_limit(self._rate_limiter)
        @with_validation(['agent_id'])
        def _handle_agent_execute_event(data):  # noqa: F811
            """处理智能体执行请求"""
            return self._handle_agent_execute(data)
        
        @socketio.on('agent_stop')
        @with_error_handling
        @with_validation(['agent_id'])
        def _handle_agent_stop_event(data):  # noqa: F811
            """处理智能体停止请求"""
            return self._handle_agent_stop(data)
        
        @socketio.on('agent_status')
        @with_error_handling
        @with_validation(['agent_id'])
        def _handle_agent_status_event(data):  # noqa: F811
            """处理智能体状态查询"""
            return self._handle_agent_status(data)
        
        @socketio.on('agent_list')
        @with_error_handling
        def _handle_agent_list_event(data):  # noqa: F811
            """处理智能体列表查询"""
            return self._handle_agent_list(data or {})
    
    def _handle_agent_execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理智能体执行"""
        events = self._import_events()
        ws_manager = self._get_websocket_manager()
        
        agent_id = data.get('agent_id')
        input_data = data.get('input_data', {})
        user_id = self._get_current_user_id(data)
        execution_id = str(uuid.uuid4())
            
        # 发送执行开始事件
        start_event = events['create_agent_event'](
            events['EventType'].AGENT_EXECUTION_STARTED,
            agent_id,
            {
                'execution_id': execution_id,
                'input_data': input_data,
                'message': f'智能体 {agent_id} 开始执行'
            },
            user_id
        )
        ws_manager.broadcast_event(start_event, target_users=[user_id] if user_id else None)
        
        # 异步执行智能体任务
        def execute_agent_task():
            try:
                agent_manager = self._get_agent_manager()
                
                if agent_manager:
                    # 使用真实的智能体服务
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            agent_manager.execute_agent_task(agent_id, input_data, {'user_id': user_id})
                        )
                    finally:
                        loop.close()
                else:
                    # 模拟执行
                    time.sleep(1)
                    result = {
                        'status': 'success',
                        'output': f'智能体 {agent_id} 执行完成 (模拟)',
                        'execution_time': 1000
                    }
                
                # 发送执行完成事件
                completion_event = events['create_agent_event'](
                    events['EventType'].AGENT_EXECUTION_COMPLETED,
                    agent_id,
                    {
                        'execution_id': execution_id,
                        'result': result,
                        'message': f'智能体 {agent_id} 执行完成'
                    },
                    user_id
                )
                ws_manager.broadcast_event(completion_event, target_users=[user_id] if user_id else None)
                
            except Exception as e:
                logger.error(f"Agent execution error: {e}")
                error_event = events['create_system_event'](
                    events['EventType'].ERROR,
                    {
                        'execution_id': execution_id,
                        'agent_id': agent_id,
                        'error': str(e),
                        'message': f'智能体 {agent_id} 执行失败'
                    },
                    user_id
                )
                ws_manager.broadcast_event(error_event, target_users=[user_id] if user_id else None)
        
        # 启动后台线程
        thread = threading.Thread(target=execute_agent_task, daemon=True)
        thread.start()
            
        return {
            'success': True, 
            'execution_id': execution_id,
            'message': '智能体执行请求已提交'
        }
    
    def _handle_agent_stop(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理智能体停止"""
        events = self._import_events()
        ws_manager = self._get_websocket_manager()
        
        agent_id = data.get('agent_id')
        user_id = self._get_current_user_id(data)
        
        agent_manager = self._get_agent_manager()
        if agent_manager:
            try:
                # 停止智能体（如果支持）
                if hasattr(agent_manager, 'stop_agent'):
                    agent_manager.stop_agent(agent_id)
                
                # 发送状态变更事件
                status_event = events['create_agent_event'](
                    events['EventType'].AGENT_STATUS_CHANGED,
                    agent_id,
                    {'status': 'stopped', 'message': f'智能体 {agent_id} 已停止'},
                    user_id
                )
                ws_manager.broadcast_event(status_event, target_users=[user_id] if user_id else None)
                
                return {'success': True, 'message': f'智能体 {agent_id} 已停止'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': '智能体管理器不可用'}
    
    def _handle_agent_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理智能体状态查询"""
        agent_id = data.get('agent_id')
        
        agent_manager = self._get_agent_manager()
        if agent_manager:
            status = agent_manager.get_agent_instance_status(agent_id)
            if status:
                return {
                    'success': True,
                    'agent_id': agent_id,
                    'status': status
                }
            return {'success': False, 'error': f'智能体 {agent_id} 不存在'}
        
        return {'success': False, 'error': '智能体管理器不可用'}
    
    def _handle_agent_list(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理智能体列表查询"""
        agent_manager = self._get_agent_manager()
        if agent_manager:
            try:
                agents = []
                if hasattr(agent_manager, 'agents'):
                    for agent_id, agent in agent_manager.agents.items():
                        agents.append({
                            'agent_id': agent_id,
                            'status': getattr(agent, 'status', 'unknown'),
                            'type': getattr(agent, 'role', 'unknown')
                        })
                return {'success': True, 'agents': agents}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': '智能体管理器不可用'}
    
    # ========== 工作流事件处理器 ==========
    
    def _register_workflow_handlers(self, socketio):
        """注册工作流相关事件处理器"""
        
        @socketio.on('workflow_execute')
        @with_error_handling
        @with_rate_limit(self._rate_limiter)
        @with_validation(['workflow_id'])
        def _handle_workflow_execute_event(data):  # noqa: F811
            """处理工作流执行请求"""
            return self._handle_workflow_execute(data)
        
        @socketio.on('workflow_pause')
        @with_error_handling
        @with_validation(['workflow_id'])
        def _handle_workflow_pause_event(data):  # noqa: F811
            """处理工作流暂停请求"""
            return self._handle_workflow_pause(data)
        
        @socketio.on('workflow_resume')
        @with_error_handling
        @with_validation(['workflow_id'])
        def _handle_workflow_resume_event(data):  # noqa: F811
            """处理工作流恢复请求"""
            return self._handle_workflow_resume(data)
        
        @socketio.on('workflow_cancel')
        @with_error_handling
        @with_validation(['workflow_id'])
        def _handle_workflow_cancel_event(data):  # noqa: F811
            """处理工作流取消请求"""
            return self._handle_workflow_cancel(data)
        
        @socketio.on('workflow_status')
        @with_error_handling
        @with_validation(['workflow_id'])
        def _handle_workflow_status_event(data):  # noqa: F811
            """处理工作流状态查询"""
            return self._handle_workflow_status(data)
    
    def _handle_workflow_execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理工作流执行"""
        events = self._import_events()
        ws_manager = self._get_websocket_manager()
        
        workflow_id = data.get('workflow_id')
        input_data = data.get('input_data', {})
        user_id = self._get_current_user_id(data)
        execution_id = str(uuid.uuid4())
            
            # 发送执行开始事件
        start_event = events['create_workflow_event'](
            events['EventType'].WORKFLOW_EXECUTION_STARTED,
                workflow_id,
                {
                'execution_id': execution_id,
                    'input_data': input_data,
                    'message': f'工作流 {workflow_id} 开始执行'
                },
                user_id
            )
        ws_manager.broadcast_event(start_event, target_users=[user_id] if user_id else None)
        
        # 异步执行工作流
        def execute_workflow():
            try:
                # 模拟工作流步骤执行
                total_steps = data.get('total_steps', 4)
                
                for step in range(1, total_steps + 1):
                    progress = int((step / total_steps) * 100)
                    time.sleep(0.5)  # 模拟步骤执行时间
                    
                    # 发送进度事件
                    progress_event = events['create_workflow_event'](
                        events['EventType'].WORKFLOW_EXECUTION_PROGRESS,
                        workflow_id,
                        {
                            'execution_id': execution_id,
                            'current_step': step,
                            'total_steps': total_steps,
                            'progress': progress,
                            'step_name': f'步骤 {step}',
                            'message': f'工作流执行进度: {progress}%'
                        },
                        user_id
                    )
                    ws_manager.broadcast_event(progress_event, target_users=[user_id] if user_id else None)
                
                # 发送完成事件
                completion_event = events['create_workflow_event'](
                    events['EventType'].WORKFLOW_EXECUTION_COMPLETED,
                    workflow_id,
                    {
                        'execution_id': execution_id,
                        'output_data': {'result': 'success', 'steps_completed': total_steps},
                        'execution_time': total_steps * 500,
                        'message': f'工作流 {workflow_id} 执行完成'
                    },
                    user_id
                )
                ws_manager.broadcast_event(completion_event, target_users=[user_id] if user_id else None)
                
            except Exception as e:
                logger.error(f"Workflow execution error: {e}")
                error_event = events['create_system_event'](
                    events['EventType'].ERROR,
                    {
                        'execution_id': execution_id,
                        'workflow_id': workflow_id,
                        'error': str(e),
                        'message': f'工作流 {workflow_id} 执行失败'
                    },
                    user_id
                )
                ws_manager.broadcast_event(error_event, target_users=[user_id] if user_id else None)
        
        # 启动后台线程
        thread = threading.Thread(target=execute_workflow, daemon=True)
        thread.start()
            
        return {
            'success': True,
            'execution_id': execution_id,
            'message': '工作流执行请求已提交'
        }
    
    def _handle_workflow_pause(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理工作流暂停"""
        workflow_id = data.get('workflow_id')
        # TODO: 实现实际的工作流暂停逻辑
        return {'success': True, 'message': f'工作流 {workflow_id} 已暂停'}
    
    def _handle_workflow_resume(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理工作流恢复"""
        workflow_id = data.get('workflow_id')
        # TODO: 实现实际的工作流恢复逻辑
        return {'success': True, 'message': f'工作流 {workflow_id} 已恢复'}
    
    def _handle_workflow_cancel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理工作流取消"""
        workflow_id = data.get('workflow_id')
        # TODO: 实现实际的工作流取消逻辑
        return {'success': True, 'message': f'工作流 {workflow_id} 已取消'}
    
    def _handle_workflow_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理工作流状态查询"""
        workflow_id = data.get('workflow_id')
        # TODO: 实现实际的工作流状态查询逻辑
        return {
            'success': True,
            'workflow_id': workflow_id,
            'status': 'unknown',
            'message': '工作流状态查询完成'
        }
    
    # ========== 训练事件处理器 ==========
    
    def _register_training_handlers(self, socketio):
        """注册训练相关事件处理器"""
        
        @socketio.on('training_start')
        @with_error_handling
        @with_rate_limit(self._rate_limiter)
        @with_validation(['task_id'])
        def _handle_training_start_event(data):  # noqa: F811
            """处理训练开始请求"""
            return self._handle_training_start(data)
        
        @socketio.on('training_stop')
        @with_error_handling
        @with_validation(['task_id'])
        def _handle_training_stop_event(data):  # noqa: F811
            """处理训练停止请求"""
            return self._handle_training_stop(data)
        
        @socketio.on('training_pause')
        @with_error_handling
        @with_validation(['task_id'])
        def _handle_training_pause_event(data):  # noqa: F811
            """处理训练暂停请求"""
            return self._handle_training_pause(data)
        
        @socketio.on('training_resume')
        @with_error_handling
        @with_validation(['task_id'])
        def _handle_training_resume_event(data):  # noqa: F811
            """处理训练恢复请求"""
            return self._handle_training_resume(data)
        
        @socketio.on('training_status')
        @with_error_handling
        @with_validation(['task_id'])
        def _handle_training_status_event(data):  # noqa: F811
            """处理训练状态查询"""
            return self._handle_training_status(data)
        
        @socketio.on('training_metrics')
        @with_error_handling
        @with_validation(['task_id'])
        def _handle_training_metrics_event(data):  # noqa: F811
            """处理训练指标查询"""
            return self._handle_training_metrics(data)
        
        @socketio.on('training_list')
        @with_error_handling
        def _handle_training_list_event(data):  # noqa: F811
            """处理训练任务列表查询"""
            return self._handle_training_list(data or {})
    
    def _handle_training_start(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理训练开始"""
        events = self._import_events()
        ws_manager = self._get_websocket_manager()
        
        task_id = data.get('task_id')
        user_id = self._get_current_user_id(data)
        config = data.get('config', {})
            
        # 发送训练开始事件
        start_event = events['create_training_event'](
            events['EventType'].TRAINING_TASK_STARTED,
            task_id,
            {
                'message': f'训练任务 {task_id} 开始',
                'config': config,
                'total_epochs': config.get('epochs', 10)
            },
            user_id
        )
        ws_manager.broadcast_event(start_event, target_users=[user_id] if user_id else None)
        
        # 进度回调函数
        def progress_callback(progress_data: Dict[str, Any]):
            """训练进度回调"""
            progress_event = events['create_training_event'](
                events['EventType'].TRAINING_TASK_PROGRESS,
                task_id,
                progress_data,
                user_id
            )
            ws_manager.broadcast_event(progress_event, target_users=[user_id] if user_id else None)
        
        # 尝试使用真实训练服务
        training_service = self._get_training_service()
        
        def execute_training():
            try:
                if training_service:
                    # 设置进度回调
                    training_service.set_progress_callback(progress_callback)
                    
                    # 启动训练
                    result = training_service.start_training_job(task_id)
                    
                    if result.get('success'):
                        # 训练完成事件
                        completion_event = events['create_training_event'](
                            events['EventType'].TRAINING_TASK_COMPLETED,
                        task_id,
                        {
                                'message': f'训练任务 {task_id} 完成',
                                'result': result
                        },
                        user_id
                    )
                        ws_manager.broadcast_event(completion_event, target_users=[user_id] if user_id else None)
                else:
                    # 模拟训练进度
                    total_epochs = config.get('epochs', 10)
                    
                    for epoch in range(1, total_epochs + 1):
                        time.sleep(0.5)  # 模拟训练时间
                        
                        progress_data = {
                            'current_epoch': epoch,
                            'total_epochs': total_epochs,
                            'progress': int((epoch / total_epochs) * 100),
                            'loss': 1.0 / (epoch + 1),
                            'accuracy': epoch / (total_epochs + 1),
                            'learning_rate': 0.001 * (0.9 ** epoch),
                            'message': f'Epoch {epoch}/{total_epochs}'
                        }
                        progress_callback(progress_data)
                        
                        # 发送指标更新
                        metrics_event = events['create_training_event'](
                            events['EventType'].TRAINING_METRICS_UPDATED,
                            task_id,
                            {
                                'epoch': epoch,
                                'metrics': {
                                    'loss': progress_data['loss'],
                                    'accuracy': progress_data['accuracy']
                                },
                                'timestamp': time.time()
                            },
                            user_id
                        )
                        ws_manager.broadcast_event(metrics_event, target_users=[user_id] if user_id else None)
                    
                    # 训练完成
                    completion_event = events['create_training_event'](
                        events['EventType'].TRAINING_TASK_COMPLETED,
                        task_id,
                        {
                            'message': f'训练任务 {task_id} 完成',
                            'final_metrics': {
                                'loss': 1.0 / (total_epochs + 1),
                                'accuracy': total_epochs / (total_epochs + 1)
                            }
                        },
                        user_id
                    )
                    ws_manager.broadcast_event(completion_event, target_users=[user_id] if user_id else None)
                
            except Exception as e:
                logger.error(f"Training execution error: {e}")
                error_event = events['create_training_event'](
                    events['EventType'].TRAINING_ERROR,
                    task_id,
                    {
                        'error': str(e),
                        'message': f'训练任务 {task_id} 执行失败'
                    },
                    user_id
                )
                ws_manager.broadcast_event(error_event, target_users=[user_id] if user_id else None)
        
        # 启动后台线程
        thread = threading.Thread(target=execute_training, daemon=True)
        thread.start()
        
        return {'success': True, 'message': '训练任务已启动', 'task_id': task_id}
    
    def _handle_training_stop(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理训练停止"""
        events = self._import_events()
        ws_manager = self._get_websocket_manager()
        
        task_id = data.get('task_id')
        user_id = self._get_current_user_id(data)
        
        training_service = self._get_training_service()
        if training_service:
            try:
                result = training_service.cancel_training_job(task_id)
                
                # 发送停止事件
                stop_event = events['create_training_event'](
                    events['EventType'].TRAINING_COMPLETE,
                    task_id,
                    {
                        'status': 'cancelled',
                        'message': f'训练任务 {task_id} 已停止'
                    },
                    user_id
                )
                ws_manager.broadcast_event(stop_event, target_users=[user_id] if user_id else None)
                
                return {'success': True, 'message': f'训练任务 {task_id} 已停止'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': True, 'message': f'训练任务 {task_id} 已停止 (模拟)'}
    
    def _handle_training_pause(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理训练暂停"""
        task_id = data.get('task_id')
        
        training_service = self._get_training_service()
        if training_service and hasattr(training_service, 'pause_training_job'):
            try:
                result = training_service.pause_training_job(task_id)
                return {'success': True, 'message': f'训练任务 {task_id} 已暂停', 'result': result}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': True, 'message': f'训练任务 {task_id} 已暂停 (模拟)'}
    
    def _handle_training_resume(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理训练恢复"""
        task_id = data.get('task_id')
        
        training_service = self._get_training_service()
        if training_service and hasattr(training_service, 'resume_training_job'):
            try:
                result = training_service.resume_training_job(task_id)
                return {'success': True, 'message': f'训练任务 {task_id} 已恢复', 'result': result}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': True, 'message': f'训练任务 {task_id} 已恢复 (模拟)'}
    
    def _handle_training_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理训练状态查询"""
        task_id = data.get('task_id')
        
        training_service = self._get_training_service()
        if training_service and hasattr(training_service, 'get_training_job_status'):
            try:
                status = training_service.get_training_job_status(task_id)
                return {'success': True, 'task_id': task_id, 'status': status}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {
            'success': True,
            'task_id': task_id,
            'status': 'unknown',
            'message': '训练状态查询完成 (模拟)'
        }
    
    def _handle_training_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理训练指标查询"""
        task_id = data.get('task_id')
        
        monitoring_service = self._get_monitoring_service()
        if monitoring_service:
            try:
                metrics = monitoring_service.get_training_metrics(task_id)
                return {'success': True, 'task_id': task_id, 'metrics': metrics}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {
            'success': True,
            'task_id': task_id,
            'metrics': {'loss': 0.1, 'accuracy': 0.9},
            'message': '训练指标查询完成 (模拟)'
        }
    
    def _handle_training_list(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理训练任务列表查询"""
        user_id = self._get_current_user_id(data)
        status_filter = data.get('status')
        
        training_service = self._get_training_service()
        if training_service and hasattr(training_service, 'list_training_jobs'):
            try:
                jobs = training_service.list_training_jobs(user_id=user_id, status=status_filter)
                return {'success': True, 'jobs': jobs}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {'success': True, 'jobs': [], 'message': '训练任务列表查询完成 (模拟)'}
    
    # ========== 对话事件处理器 ==========
    
    def _register_dialogue_handlers(self, socketio):
        """注册对话相关事件处理器"""
        
        @socketio.on('dialogue_message')
        @with_error_handling
        @with_rate_limit(self._rate_limiter)
        @with_validation(['session_id', 'message'])
        def _handle_dialogue_message_event(data):  # noqa: F811
            """处理对话消息"""
            return self._handle_dialogue_message(data)
        
        @socketio.on('dialogue_start')
        @with_error_handling
        def _handle_dialogue_start_event(data):  # noqa: F811
            """处理对话开始"""
            return self._handle_dialogue_start(data or {})
        
        @socketio.on('dialogue_end')
        @with_error_handling
        @with_validation(['session_id'])
        def _handle_dialogue_end_event(data):  # noqa: F811
            """处理对话结束"""
            return self._handle_dialogue_end(data)
        
        @socketio.on('dialogue_typing')
        @with_error_handling
        @with_validation(['session_id'])
        def _handle_dialogue_typing_event(data):  # noqa: F811
            """处理正在输入状态"""
            return self._handle_dialogue_typing(data)
    
    def _handle_dialogue_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理对话消息"""
        events = self._import_events()
        ws_manager = self._get_websocket_manager()
        
        session_id = data.get('session_id')
        message = data.get('message')
        user_id = self._get_current_user_id(data)
        message_id = str(uuid.uuid4())
            
        # 发送消息接收事件
        received_event = events['create_dialogue_event'](
            events['EventType'].DIALOGUE_MESSAGE_RECEIVED,
            session_id,
            {
                'message_id': message_id,
                'message': message,
                'sender': user_id or 'user',
                'sender_type': 'user',
                'timestamp': time.time()
            },
            user_id
        )
        ws_manager.broadcast_event(received_event)
        
        # 异步生成回复
        def generate_response():
            try:
                # 获取智能体管理器用于生成回复
                agent_manager = self._get_agent_manager()
                
                if agent_manager:
                    # 使用对话管理智能体
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        # 查找或创建对话智能体
                        dialogue_agent_id = f"dialogue_{session_id}"
                        result = loop.run_until_complete(
                            agent_manager.execute_agent_task(
                                dialogue_agent_id,
                                {'message': message},
                                {'session_id': session_id, 'user_id': user_id}
                            )
                        )
                        response_message = result.get('output', {}).get('response', f"收到您的消息: {message}")
                    finally:
                        loop.close()
                else:
                    time.sleep(0.5)
                    response_message = f"收到您的消息: {message}"
                
                # 发送回复事件
                response_event = events['create_dialogue_event'](
                    events['EventType'].DIALOGUE_MESSAGE_RECEIVED,
                    session_id,
                    {
                        'message_id': str(uuid.uuid4()),
                        'message': response_message,
                        'sender': 'assistant',
                        'sender_type': 'assistant',
                        'reply_to': message_id,
                        'timestamp': time.time()
                    },
                    user_id
                )
                ws_manager.broadcast_event(response_event)
                
            except Exception as e:
                logger.error(f"Dialogue response error: {e}")
                error_event = events['create_system_event'](
                    events['EventType'].ERROR,
                    {
                        'session_id': session_id,
                        'error': str(e),
                        'message': '生成回复失败'
                    },
                    user_id
                )
                ws_manager.broadcast_event(error_event)
        
        # 启动后台线程
        thread = threading.Thread(target=generate_response, daemon=True)
        thread.start()
            
        return {'success': True, 'message_id': message_id, 'message': '消息已发送'}
    
    def _handle_dialogue_start(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理对话开始"""
        user_id = self._get_current_user_id(data)
        session_id = data.get('session_id') or str(uuid.uuid4())
        
        return {
            'success': True,
            'session_id': session_id,
            'message': '对话已开始'
        }
    
    def _handle_dialogue_end(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理对话结束"""
        session_id = data.get('session_id')
        return {'success': True, 'session_id': session_id, 'message': '对话已结束'}
    
    def _handle_dialogue_typing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理正在输入状态"""
        events = self._import_events()
        ws_manager = self._get_websocket_manager()
        
        session_id = data.get('session_id')
        user_id = self._get_current_user_id(data)
        is_typing = data.get('is_typing', True)
        
        # 广播输入状态
        typing_event = events['create_dialogue_event'](
            events['EventType'].MESSAGE,
            session_id,
            {
                'type': 'typing',
                'user_id': user_id,
                'is_typing': is_typing,
                'timestamp': time.time()
            },
            user_id
        )
        ws_manager.broadcast_event(typing_event)
        
        return {'success': True}
    
    # ========== 系统事件处理器 ==========
    
    def _register_system_handlers(self, socketio):
        """注册系统相关事件处理器"""
        
        @socketio.on('system_status')
        @with_error_handling
        def _handle_system_status_event(data):  # noqa: F811
            """处理系统状态查询"""
            return self._handle_system_status(data or {})
        
        @socketio.on('system_health')
        @with_error_handling
        def _handle_system_health_event(data):  # noqa: F811
            """处理系统健康检查"""
            return self._handle_system_health(data or {})
        
        @socketio.on('system_metrics')
        @with_error_handling
        def _handle_system_metrics_event(data):  # noqa: F811
            """处理系统指标查询"""
            return self._handle_system_metrics(data or {})
        
        @socketio.on('heartbeat')
        @with_error_handling
        def _handle_heartbeat_event(data):  # noqa: F811
            """处理心跳事件"""
            return self._handle_heartbeat(data or {})
        
        @socketio.on('error')
        @with_error_handling
        def _handle_error_event(data):  # noqa: F811
            """处理错误事件"""
            return self._handle_error(data or {})
    
    def _handle_system_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理系统状态查询"""
        events = self._import_events()
        ws_manager = self._get_websocket_manager()
        user_id = self._get_current_user_id(data)
        
        # 尝试获取真实系统状态
        monitoring_service = self._get_monitoring_service()
        
        if monitoring_service:
            try:
                system_health = monitoring_service.get_system_health()
                system_status = {
                    'cpu_usage': system_health.get('cpu', {}).get('usage', 0),
                    'memory_usage': system_health.get('memory', {}).get('usage', 0),
                    'disk_usage': system_health.get('disk', {}).get('usage', 0),
                    'network_io': system_health.get('network', {}),
                    'active_connections': ws_manager.get_stats().get('current_connections', 0),
                    'uptime': system_health.get('uptime', 0),
                    'timestamp': datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.warning(f"Failed to get real system status: {e}")
                system_status = self._get_mock_system_status()
        else:
            system_status = self._get_mock_system_status()
        
        # 添加 WebSocket 统计
        system_status['websocket_stats'] = ws_manager.get_stats()
            
        # 发送系统状态事件
        status_event = events['create_system_event'](
            events['EventType'].SYSTEM_STATUS,
            system_status,
            user_id
        )
        ws_manager.broadcast_event(status_event, target_users=[user_id] if user_id else None)
        
        return {'success': True, 'status': system_status}
    
    def _get_mock_system_status(self) -> Dict[str, Any]:
        """获取模拟的系统状态"""
        import random
        return {
            'cpu_usage': random.uniform(20, 60),
            'memory_usage': random.uniform(40, 80),
            'disk_usage': random.uniform(30, 50),
            'network_io': {'in': random.randint(1000, 10000), 'out': random.randint(1000, 10000)},
            'active_connections': random.randint(5, 20),
            'uptime': random.randint(3600, 86400),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _handle_system_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理系统健康检查"""
        monitoring_service = self._get_monitoring_service()
        
        if monitoring_service:
            try:
                health = monitoring_service.get_system_health()
                return {'success': True, 'health': health}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {
            'success': True,
            'health': {
                'status': 'healthy',
                'components': {
                    'database': 'healthy',
                    'cache': 'healthy',
                    'queue': 'healthy'
                }
            }
        }
    
    def _handle_system_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理系统指标查询"""
        monitoring_service = self._get_monitoring_service()
        
        if monitoring_service:
            try:
                metrics = monitoring_service.get_system_metrics()
                return {'success': True, 'metrics': metrics}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        return {
            'success': True,
            'metrics': {
                'cpu': [],
                'memory': [],
                'disk': [],
                'network': []
            },
            'message': '系统指标查询完成 (模拟)'
        }
    
    def _handle_heartbeat(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理心跳事件"""
        events = self._import_events()
        ws_manager = self._get_websocket_manager()
        user_id = self._get_current_user_id(data)
        
        # 更新连接的心跳时间
        socket_id = self._get_current_socket_id()
        if socket_id:
            connection = ws_manager.get_connection(socket_id)
            if connection:
                connection.update_heartbeat()
            
        # 发送心跳响应
        heartbeat_event = events['create_system_event'](
            events['EventType'].HEARTBEAT,
            {
                'type': 'pong',
                'timestamp': time.time(),
                'server_time': datetime.utcnow().isoformat()
            },
            user_id
        )
        ws_manager.broadcast_event(heartbeat_event, target_users=[user_id] if user_id else None)
        
        return {'success': True, 'type': 'pong', 'timestamp': time.time()}
    
    def _handle_error(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理错误事件"""
        events = self._import_events()
        ws_manager = self._get_websocket_manager()
        
        error_message = data.get('message', 'Unknown error')
        error_code = data.get('code', 'UNKNOWN')
        user_id = self._get_current_user_id(data)
            
        logger.error(f"Client error: {error_message} (Code: {error_code})")
            
        # 发送错误确认
        error_event = events['create_system_event'](
            events['EventType'].ERROR,
            {
                'received': True,
                    'original_error': error_message,
                'code': error_code,
                'message': '服务器已收到错误报告'
                },
                user_id
            )
        ws_manager.broadcast_event(error_event, target_users=[user_id] if user_id else None)
            
        return {'success': True, 'message': '错误报告已接收'}
            
    # ========== 房间事件处理器 ==========
    
    def _register_room_handlers(self, socketio):
        """注册房间相关事件处理器"""
        
        @socketio.on('room_create')
        @with_error_handling
        @with_validation(['room_name'])
        def handle_room_create(data):
            """处理创建房间"""
            return self._handle_room_create(data)
        
        @socketio.on('room_delete')
        @with_error_handling
        @with_validation(['room_name'])
        def handle_room_delete(data):
            """处理删除房间"""
            return self._handle_room_delete(data)
        
        @socketio.on('room_list')
        @with_error_handling
        def handle_room_list(data):
            """处理房间列表查询"""
            return self._handle_room_list(data or {})
        
        @socketio.on('room_members')
        @with_error_handling
        @with_validation(['room_name'])
        def handle_room_members(data):
            """处理房间成员查询"""
            return self._handle_room_members(data)
        
        @socketio.on('room_message')
        @with_error_handling
        @with_rate_limit(self._rate_limiter)
        @with_validation(['room_name', 'message'])
        def handle_room_message(data):
            """处理房间消息"""
            return self._handle_room_message(data)
    
    def _handle_room_create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理创建房间"""
        ws_manager = self._get_websocket_manager()
        
        room_name = data.get('room_name')
        metadata = data.get('metadata', {})
        
        success = ws_manager.create_room(room_name, metadata)
        
        if success:
            return {'success': True, 'room_name': room_name, 'message': f'房间 {room_name} 已创建'}
        else:
            return {'success': False, 'error': f'房间 {room_name} 已存在'}
    
    def _handle_room_delete(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理删除房间"""
        ws_manager = self._get_websocket_manager()
        
        room_name = data.get('room_name')
        
        success = ws_manager.delete_room(room_name)
        
        if success:
            return {'success': True, 'room_name': room_name, 'message': f'房间 {room_name} 已删除'}
        else:
            return {'success': False, 'error': f'房间 {room_name} 不存在'}
    
    def _handle_room_list(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理房间列表查询"""
        ws_manager = self._get_websocket_manager()
        
        rooms = ws_manager.get_room_list()
        
        return {'success': True, 'rooms': rooms}
    
    def _handle_room_members(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理房间成员查询"""
        ws_manager = self._get_websocket_manager()
        
        room_name = data.get('room_name')
        members = ws_manager.get_room_members(room_name)
        
        return {'success': True, 'room_name': room_name, 'members': members}
    
    def _handle_room_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理房间消息"""
        events = self._import_events()
        ws_manager = self._get_websocket_manager()
        
        room_name = data.get('room_name')
        message = data.get('message')
        user_id = self._get_current_user_id(data)
        
        # 创建消息事件
        message_event = events['WebSocketEvent'](
            event_type=events['EventType'].MESSAGE,
            data={
                'room': room_name,
                'message': message,
                'sender': user_id,
                'timestamp': time.time()
            },
            client_id=self._get_current_socket_id()
        )
        
        # 发送到房间
        ws_manager.emit_to_room(room_name, message_event, exclude_sender=True)
        
        return {'success': True, 'message': '消息已发送到房间'}
    
    # ========== 订阅事件处理器 ==========
    
    def _register_subscription_handlers(self, socketio):
        """注册订阅相关事件处理器"""
        
        @socketio.on('subscribe')
        @with_error_handling
        @with_validation(['topics'])
        def handle_subscribe(data):
            """处理订阅请求"""
            return self._handle_subscribe(data)
        
        @socketio.on('unsubscribe')
        @with_error_handling
        @with_validation(['topics'])
        def handle_unsubscribe(data):
            """处理取消订阅请求"""
            return self._handle_unsubscribe(data)
        
        @socketio.on('subscriptions')
        @with_error_handling
        def handle_subscriptions(data):
            """处理订阅列表查询"""
            return self._handle_subscriptions(data or {})
    
    def _handle_subscribe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理订阅"""
        ws_manager = self._get_websocket_manager()
        
        topics = data.get('topics', [])
        if isinstance(topics, str):
            topics = [topics]
        
        socket_id = self._get_current_socket_id()
        subscribed = []
        
        for topic in topics:
            room_name = f"topic_{topic}"
            if ws_manager._join_room_internal(socket_id, room_name):
                subscribed.append(topic)
        
        return {
            'success': True,
            'subscribed': subscribed,
            'message': f'已订阅 {len(subscribed)} 个主题'
        }
    
    def _handle_unsubscribe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理取消订阅"""
        ws_manager = self._get_websocket_manager()
        
        topics = data.get('topics', [])
        if isinstance(topics, str):
            topics = [topics]
        
        socket_id = self._get_current_socket_id()
        unsubscribed = []
        
        for topic in topics:
            room_name = f"topic_{topic}"
            if ws_manager._leave_room_internal(socket_id, room_name):
                unsubscribed.append(topic)
        
        return {
            'success': True,
            'unsubscribed': unsubscribed,
            'message': f'已取消订阅 {len(unsubscribed)} 个主题'
        }
    
    def _handle_subscriptions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理订阅列表查询"""
        ws_manager = self._get_websocket_manager()
        socket_id = self._get_current_socket_id()
        
        connection = ws_manager.get_connection(socket_id)
        if not connection:
            return {'success': False, 'error': '连接不存在'}
        
        # 过滤出主题订阅（以 topic_ 开头的房间）
        topics = [
            room.replace('topic_', '') 
            for room in connection.subscribed_rooms 
            if room.startswith('topic_')
        ]
        
        return {'success': True, 'topics': topics}
    
    # ========== 工具事件处理器 ==========
    
    def _register_utility_handlers(self, socketio):
        """注册工具相关事件处理器"""
        
        @socketio.on('get_online_users')
        @with_error_handling
        def handle_get_online_users(data):
            """获取在线用户列表"""
            return self._handle_get_online_users(data or {})
        
        @socketio.on('get_connection_info')
        @with_error_handling
        def handle_get_connection_info(data):
            """获取连接信息"""
            return self._handle_get_connection_info(data or {})
        
        @socketio.on('broadcast_message')
        @with_error_handling
        @with_validation(['message'])
        def handle_broadcast_message(data):
            """广播消息"""
            return self._handle_broadcast_message(data)
        
        @socketio.on('send_to_user')
        @with_error_handling
        @with_validation(['target_user_id', 'message'])
        def handle_send_to_user(data):
            """发送消息给指定用户"""
            return self._handle_send_to_user(data)
    
    def _handle_get_online_users(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取在线用户列表"""
        ws_manager = self._get_websocket_manager()
        
        online_users = ws_manager.get_online_users()
        
        return {
            'success': True,
            'online_users': online_users,
            'count': len(online_users)
        }
    
    def _handle_get_connection_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取连接信息"""
        ws_manager = self._get_websocket_manager()
        socket_id = self._get_current_socket_id()
        
        connection = ws_manager.get_connection(socket_id)
        if not connection:
            return {'success': False, 'error': '连接不存在'}
        
        return {
            'success': True,
            'connection': connection.to_dict()
        }
    
    def _handle_broadcast_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """广播消息"""
        events = self._import_events()
        ws_manager = self._get_websocket_manager()
        
        message = data.get('message')
        user_id = self._get_current_user_id(data)
        target_users = data.get('target_users')  # 可选的目标用户列表
        
        broadcast_event = events['WebSocketEvent'](
            event_type=events['EventType'].BROADCAST,
            data={
                'message': message,
                'sender': user_id,
                'timestamp': time.time()
            },
            client_id=self._get_current_socket_id(),
            target_users=target_users
        )
        
        ws_manager.broadcast_event(broadcast_event, target_users=target_users)
        
        return {'success': True, 'message': '消息已广播'}
    
    def _handle_send_to_user(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """发送消息给指定用户"""
        events = self._import_events()
        ws_manager = self._get_websocket_manager()
        
        target_user_id = data.get('target_user_id')
        message = data.get('message')
        sender_id = self._get_current_user_id(data)
        
        # 检查目标用户是否在线
        if not ws_manager.is_user_online(target_user_id):
            return {'success': False, 'error': f'用户 {target_user_id} 不在线'}
        
        # 发送消息
        message_event = events['WebSocketEvent'](
            event_type=events['EventType'].MESSAGE,
            data={
                'message': message,
                'sender': sender_id,
                'type': 'direct_message',
                'timestamp': time.time()
            },
            client_id=self._get_current_socket_id()
        )
        
        ws_manager.emit_to_user(target_user_id, message_event)
        
        return {'success': True, 'message': f'消息已发送给用户 {target_user_id}'}


# ============================================================================
# 全局实例和注册函数
# ============================================================================

# 全局处理器实例
_handlers = WebSocketHandlers()


def register_websocket_handlers():
    """注册 WebSocket 事件处理器（兼容性函数）"""
    return _handlers.register_handlers()


def get_websocket_handlers() -> WebSocketHandlers:
    """获取 WebSocket 处理器实例"""
    return _handlers


def register_custom_handler(event_name: str, handler: Callable):
    """注册自定义事件处理器
    
    Args:
        event_name: 事件名称
        handler: 处理器函数
    """
    _handlers.register_custom_handler(event_name, handler)


# 注意：不再在模块加载时自动注册处理器
# 应该在 WebSocket 初始化后显式调用 register_websocket_handlers()

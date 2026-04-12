"""WebSocket 模块

提供完整的 WebSocket 能力，包括：
- 连接管理和生命周期
- 事件处理和广播
- 房间/频道管理
- 用户认证和授权
- 心跳检测
"""

from .websocket_manager import websocket_manager, get_websocket_manager, WebSocketManager, WebSocketConnection, ConnectionState
from .events import EventType, WebSocketEvent, create_agent_event, create_workflow_event, create_training_event, create_dialogue_event, create_system_event
from .handlers import register_websocket_handlers, get_websocket_handlers, register_custom_handler, WebSocketHandlers

__all__ = [
    # WebSocket 管理器
    'websocket_manager',
    'get_websocket_manager',
    'WebSocketManager',
    'WebSocketConnection',
    'ConnectionState',
    
    # 事件
    'EventType',
    'WebSocketEvent',
    'create_agent_event',
    'create_workflow_event',
    'create_training_event',
    'create_dialogue_event',
    'create_system_event',
    
    # 处理器
    'register_websocket_handlers',
    'get_websocket_handlers',
    'register_custom_handler',
    'WebSocketHandlers',
]

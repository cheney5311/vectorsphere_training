"""WebSocket连接管理器

管理WebSocket连接、房间和事件广播。
"""

import logging
import time
import threading
import uuid
from typing import Dict, Set, Optional, List, Any, Callable
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

try:
    from flask import request
    from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
except ImportError:
    # 处理 Flask-SocketIO 未安装的情况
    SocketIO = None
    emit = None
    join_room = None
    leave_room = None
    disconnect = None
    request = None

from .events import WebSocketEvent, EventType

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """连接状态枚举"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"


@dataclass
class WebSocketConnection:
    """WebSocket连接信息"""
    
    session_id: str
    socket_id: str
    user_id: Optional[str] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    subscribed_rooms: Set[str] = field(default_factory=set)
    state: ConnectionState = ConnectionState.CONNECTING
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'session_id': self.session_id,
            'socket_id': self.socket_id,
            'user_id': self.user_id,
            'username': self.username,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'connected_at': self.connected_at.isoformat() if self.connected_at else None,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None,
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'subscribed_rooms': list(self.subscribed_rooms),
            'state': self.state.value,
            'metadata': self.metadata
        }
    
    def update_activity(self):
        """更新最后活动时间"""
        self.last_activity = datetime.utcnow()
    
    def update_heartbeat(self):
        """更新心跳时间"""
        self.last_heartbeat = datetime.utcnow()
        self.last_activity = datetime.utcnow()


class WebSocketManager:
    """WebSocket连接管理器
    
    提供完整的 WebSocket 连接管理能力，包括：
    - 连接生命周期管理
    - 用户会话管理
    - 房间（频道）管理
    - 事件广播
    - 心跳检测
    - 优雅关闭
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.socketio: Optional[SocketIO] = None
        
        # 连接管理
        self.connections: Dict[str, WebSocketConnection] = {}  # socket_id -> connection
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)  # user_id -> set of socket_ids
        self.session_connections: Dict[str, str] = {}  # session_id -> socket_id
        
        # 房间管理
        self.rooms: Dict[str, Set[str]] = defaultdict(set)  # room_name -> set of socket_ids
        self.room_metadata: Dict[str, Dict[str, Any]] = {}  # room_name -> metadata
        
        # 事件处理
        self.event_handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # 心跳配置
        self.heartbeat_interval = 25  # 秒
        self.heartbeat_timeout = 60  # 秒
        self._heartbeat_thread: Optional[threading.Thread] = None
        
        # 状态
        self._running = False
        self._shutting_down = False
        
        # 统计信息
        self._stats = {
            'total_connections': 0,
            'total_disconnections': 0,
            'total_messages_sent': 0,
            'total_messages_received': 0,
            'started_at': None
        }
        
        self._initialized = True
        logger.info("WebSocketManager initialized")
    
    def init_socketio(self, app, cors_allowed_origins="*", **kwargs):
        """初始化SocketIO
        
        Args:
            app: Flask应用实例
            cors_allowed_origins: CORS允许的来源
            **kwargs: 额外的SocketIO配置参数
        """
        if SocketIO is None:
            logger.warning("Flask-SocketIO not installed, WebSocket functionality will not be available")
            return False
        
        try:
            # 合并默认配置和用户配置
            socketio_config = {
                'cors_allowed_origins': cors_allowed_origins,
                'async_mode': 'threading',
                'logger': False,
                'engineio_logger': False,
                'ping_timeout': self.heartbeat_timeout,
                'ping_interval': self.heartbeat_interval,
                'transports': ['websocket', 'polling'],
                'always_connect': False,
                'manage_session': False
            }
            socketio_config.update(kwargs)
            
            self.socketio = SocketIO(app, **socketio_config)
            
            # 注册内置事件处理器
            self._register_builtin_events()
            
            # 启动心跳检测
            self._start_heartbeat_checker()
            
            self._running = True
            self._stats['started_at'] = datetime.utcnow().isoformat()
            
            logger.info("WebSocket manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SocketIO: {e}")
            raise
    
    def _register_builtin_events(self):
        """注册内置SocketIO事件处理器"""
        if self.socketio is None:
            return
            
        @self.socketio.on('connect')
        def handle_connect(auth=None):
            """处理连接事件"""
            return self._handle_connect(auth)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """处理断开连接事件"""
            return self._handle_disconnect()
        
        @self.socketio.on('join_room')
        def handle_join_room_event(data):
            """处理加入房间事件"""
            return self._handle_join_room(data)
        
        @self.socketio.on('leave_room')
        def handle_leave_room_event(data):
            """处理离开房间事件"""
            return self._handle_leave_room(data)
        
        @self.socketio.on('ping')
        def handle_ping(data=None):
            """处理心跳事件"""
            return self._handle_ping(data)
        
        @self.socketio.on('authenticate')
        def handle_authenticate(data):
            """处理认证事件"""
            return self._handle_authenticate(data)
        
        @self.socketio.on('message')
        def handle_message(data):
            """处理通用消息事件"""
            return self._handle_message(data)
        
        logger.debug("Built-in WebSocket events registered")
    
    def _handle_connect(self, auth: Optional[Dict] = None) -> bool:
        """处理连接事件
        
        Args:
            auth: 认证信息
            
        Returns:
            是否允许连接
        """
        try:
            socket_id = request.sid if request else str(uuid.uuid4())
            session_id = str(uuid.uuid4())
            
            # 获取客户端信息
            ip_address = None
            user_agent = None
            if request:
                ip_address = request.remote_addr or request.environ.get('HTTP_X_FORWARDED_FOR', '')
                user_agent = request.environ.get('HTTP_USER_AGENT', '')
            
            # 创建连接对象
            connection = WebSocketConnection(
                session_id=session_id,
                socket_id=socket_id,
                ip_address=ip_address,
                user_agent=user_agent,
                state=ConnectionState.CONNECTED
            )
            
            # 处理认证信息
            if auth:
                user_id = auth.get('user_id')
                username = auth.get('username')
                token = auth.get('token')
                
                # 验证 token（这里应该调用实际的认证服务）
                if token and self._validate_token(token):
                    connection.user_id = user_id
                    connection.username = username
                    connection.state = ConnectionState.AUTHENTICATED
                    
                    # 将连接添加到用户连接映射
                    if user_id:
                        self.user_connections[user_id].add(socket_id)
                        # 自动加入用户专属房间
                        self._join_room_internal(socket_id, f"user_{user_id}")
            
            # 保存连接
            self.connections[socket_id] = connection
            self.session_connections[session_id] = socket_id
            
            # 更新统计
            self._stats['total_connections'] += 1
            
            # 发送连接成功事件
            emit('connected', {
                'session_id': session_id,
                'socket_id': socket_id,
                'authenticated': connection.state == ConnectionState.AUTHENTICATED,
                'timestamp': datetime.utcnow().isoformat()
            })
            
            logger.info(f"Client connected: socket_id={socket_id}, user_id={connection.user_id}")
            
            # 触发连接事件回调
            self._trigger_event(EventType.CONNECT, {
                'socket_id': socket_id,
                'user_id': connection.user_id
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Connection handling error: {e}")
            return False
    
    def _handle_disconnect(self):
        """处理断开连接事件"""
        try:
            socket_id = request.sid if request else None
            if not socket_id:
                return
            
            connection = self.connections.get(socket_id)
            if not connection:
                return
            
            # 更新连接状态
            connection.state = ConnectionState.DISCONNECTED
            
            # 从所有房间中移除
            for room in list(connection.subscribed_rooms):
                self._leave_room_internal(socket_id, room)
            
            # 从用户连接映射中移除
            if connection.user_id:
                self.user_connections[connection.user_id].discard(socket_id)
                if not self.user_connections[connection.user_id]:
                    del self.user_connections[connection.user_id]
            
            # 从会话映射中移除
            if connection.session_id in self.session_connections:
                del self.session_connections[connection.session_id]
            
            # 从连接映射中移除
            del self.connections[socket_id]
            
            # 更新统计
            self._stats['total_disconnections'] += 1
            
            logger.info(f"Client disconnected: socket_id={socket_id}, user_id={connection.user_id}")
            
            # 触发断开事件回调
            self._trigger_event(EventType.DISCONNECT, {
                'socket_id': socket_id,
                'user_id': connection.user_id
            })
            
        except Exception as e:
            logger.error(f"Disconnect handling error: {e}")
    
    def _handle_join_room(self, data: Dict) -> Dict[str, Any]:
        """处理加入房间事件
        
        Args:
            data: 包含 room_name 的数据
            
        Returns:
            处理结果
        """
        try:
            socket_id = request.sid if request else None
            if not socket_id:
                return {'success': False, 'error': 'Invalid socket connection'}
            
            room_name = data.get('room') or data.get('room_name')
            if not room_name:
                return {'success': False, 'error': 'Room name is required'}
            
            connection = self.connections.get(socket_id)
            if not connection:
                return {'success': False, 'error': 'Connection not found'}
            
            # 更新活动时间
            connection.update_activity()
            
            # 加入房间
            success = self._join_room_internal(socket_id, room_name)
            
            if success:
                # 广播加入房间事件
                self.emit_to_room(room_name, WebSocketEvent(
                    event_type=EventType.MESSAGE,
                    data={
                        'type': 'user_joined',
                        'user_id': connection.user_id,
                        'username': connection.username,
                        'room': room_name,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                ), exclude_sender=True)
                
                return {
                    'success': True,
                    'room': room_name,
                    'members': len(self.rooms.get(room_name, set()))
                }
            else:
                return {'success': False, 'error': 'Failed to join room'}
                
        except Exception as e:
            logger.error(f"Join room error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_leave_room(self, data: Dict) -> Dict[str, Any]:
        """处理离开房间事件
        
        Args:
            data: 包含 room_name 的数据
            
        Returns:
            处理结果
        """
        try:
            socket_id = request.sid if request else None
            if not socket_id:
                return {'success': False, 'error': 'Invalid socket connection'}
            
            room_name = data.get('room') or data.get('room_name')
            if not room_name:
                return {'success': False, 'error': 'Room name is required'}
            
            connection = self.connections.get(socket_id)
            if not connection:
                return {'success': False, 'error': 'Connection not found'}
            
            # 更新活动时间
            connection.update_activity()
            
            # 广播离开房间事件
            self.emit_to_room(room_name, WebSocketEvent(
                event_type=EventType.MESSAGE,
                data={
                    'type': 'user_left',
                    'user_id': connection.user_id,
                    'username': connection.username,
                    'room': room_name,
                    'timestamp': datetime.utcnow().isoformat()
                }
            ), exclude_sender=True)
            
            # 离开房间
            success = self._leave_room_internal(socket_id, room_name)
            
            if success:
                return {'success': True, 'room': room_name}
            else:
                return {'success': False, 'error': 'Failed to leave room'}
                
        except Exception as e:
            logger.error(f"Leave room error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_ping(self, data: Optional[Dict] = None) -> Dict[str, Any]:
        """处理心跳事件
        
        Args:
            data: 可选的心跳数据
            
        Returns:
            心跳响应
        """
        try:
            socket_id = request.sid if request else None
            if not socket_id:
                return {'success': False, 'error': 'Invalid socket connection'}
            
            connection = self.connections.get(socket_id)
            if connection:
                connection.update_heartbeat()
            
            # 发送pong响应
            response = {
                'success': True,
                'type': 'pong',
                'timestamp': time.time(),
                'server_time': datetime.utcnow().isoformat()
            }
            
            emit('pong', response)
            
            return response
            
        except Exception as e:
            logger.error(f"Ping handling error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_authenticate(self, data: Dict) -> Dict[str, Any]:
        """处理认证事件
        
        Args:
            data: 认证数据，包含 token, user_id, username 等
            
        Returns:
            认证结果
        """
        try:
            socket_id = request.sid if request else None
            if not socket_id:
                return {'success': False, 'error': 'Invalid socket connection'}
            
            connection = self.connections.get(socket_id)
            if not connection:
                return {'success': False, 'error': 'Connection not found'}
            
            token = data.get('token')
            user_id = data.get('user_id')
            username = data.get('username')
            
            # 验证 token
            if not token or not self._validate_token(token):
                return {'success': False, 'error': 'Invalid token'}
            
            # 更新连接信息
            old_user_id = connection.user_id
            connection.user_id = user_id
            connection.username = username
            connection.state = ConnectionState.AUTHENTICATED
            connection.update_activity()
            
            # 更新用户连接映射
            if old_user_id:
                self.user_connections[old_user_id].discard(socket_id)
                if not self.user_connections[old_user_id]:
                    del self.user_connections[old_user_id]
            
            if user_id:
                self.user_connections[user_id].add(socket_id)
                # 加入用户专属房间
                self._join_room_internal(socket_id, f"user_{user_id}")
            
            logger.info(f"Client authenticated: socket_id={socket_id}, user_id={user_id}")
            
            return {
                'success': True,
                'user_id': user_id,
                'username': username,
                'authenticated': True
            }
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _handle_message(self, data: Dict) -> Dict[str, Any]:
        """处理通用消息事件
        
        Args:
            data: 消息数据
            
        Returns:
            处理结果
        """
        try:
            socket_id = request.sid if request else None
            if not socket_id:
                return {'success': False, 'error': 'Invalid socket connection'}
            
            connection = self.connections.get(socket_id)
            if connection:
                connection.update_activity()
            
            # 更新统计
            self._stats['total_messages_received'] += 1
            
            # 触发消息事件回调
            self._trigger_event(EventType.MESSAGE, {
                'socket_id': socket_id,
                'user_id': connection.user_id if connection else None,
                'data': data
            })
            
            return {'success': True, 'received': True}
            
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _validate_token(self, token: str) -> bool:
        """验证认证 token
        
        Args:
            token: 认证 token
            
        Returns:
            是否有效
        """
        # TODO: 实现实际的 token 验证逻辑
        # 这里应该调用认证服务验证 token
        if not token:
            return False
        return True
    
    def _join_room_internal(self, socket_id: str, room_name: str) -> bool:
        """内部房间加入方法
        
        Args:
            socket_id: Socket ID
            room_name: 房间名称
            
        Returns:
            是否成功
        """
        try:
            connection = self.connections.get(socket_id)
            if not connection:
                return False
            
            # 使用 Flask-SocketIO 的 join_room
            if join_room:
                join_room(room_name, sid=socket_id)
            
            # 更新连接的房间列表
            connection.subscribed_rooms.add(room_name)
            
            # 更新房间成员
            self.rooms[room_name].add(socket_id)
            
            logger.debug(f"Socket {socket_id} joined room {room_name}")
            return True
            
        except Exception as e:
            logger.error(f"Join room internal error: {e}")
            return False
    
    def _leave_room_internal(self, socket_id: str, room_name: str) -> bool:
        """内部房间离开方法
        
        Args:
            socket_id: Socket ID
            room_name: 房间名称
            
        Returns:
            是否成功
        """
        try:
            connection = self.connections.get(socket_id)
            if connection:
                connection.subscribed_rooms.discard(room_name)
            
            # 使用 Flask-SocketIO 的 leave_room
            if leave_room:
                leave_room(room_name, sid=socket_id)
            
            # 更新房间成员
            if room_name in self.rooms:
                self.rooms[room_name].discard(socket_id)
                # 如果房间为空，删除房间
                if not self.rooms[room_name]:
                    del self.rooms[room_name]
                    if room_name in self.room_metadata:
                        del self.room_metadata[room_name]
            
            logger.debug(f"Socket {socket_id} left room {room_name}")
            return True
            
        except Exception as e:
            logger.error(f"Leave room internal error: {e}")
            return False
    
    def _start_heartbeat_checker(self):
        """启动心跳检测线程"""
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return
        
        def heartbeat_checker():
            while self._running and not self._shutting_down:
                try:
                    self._check_heartbeats()
                except Exception as e:
                    logger.error(f"Heartbeat check error: {e}")
                time.sleep(self.heartbeat_interval)
        
        self._heartbeat_thread = threading.Thread(target=heartbeat_checker, daemon=True)
        self._heartbeat_thread.start()
        logger.debug("Heartbeat checker started")
    
    def _check_heartbeats(self):
        """检查连接心跳"""
        now = datetime.utcnow()
        timeout_threshold = self.heartbeat_timeout
        
        for socket_id, connection in list(self.connections.items()):
            if connection.state == ConnectionState.DISCONNECTED:
                continue
            
            # 计算最后心跳时间
            last_heartbeat = connection.last_heartbeat
            seconds_since_heartbeat = (now - last_heartbeat).total_seconds()
            
            if seconds_since_heartbeat > timeout_threshold:
                logger.warning(f"Connection timeout: socket_id={socket_id}, last_heartbeat={last_heartbeat}")
                # 强制断开连接
                self.disconnect_client(socket_id, reason='heartbeat_timeout')
    
    def _trigger_event(self, event_type: EventType, data: Dict[str, Any]):
        """触发事件回调
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        handlers = self.event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error for {event_type}: {e}")
    
    # ========== 公共 API 方法 ==========
    
    def register_event_handler(self, event_type: EventType, handler: Callable):
        """注册事件处理器
        
        Args:
            event_type: 事件类型
            handler: 处理器函数
        """
        self.event_handlers[event_type].append(handler)
        logger.debug(f"Event handler registered for {event_type}")
    
    def emit_to_user(self, user_id: str, event: WebSocketEvent):
        """向特定用户发送事件
        
        Args:
            user_id: 用户 ID
            event: 事件对象
        """
        if not self.socketio:
            return
        
        try:
            room_name = f"user_{user_id}"
            self.socketio.emit('event', event.to_dict(), room=room_name)
            self._stats['total_messages_sent'] += 1
            logger.debug(f"Event {event.event_type.value} sent to user {user_id}")
        except Exception as e:
            logger.error(f"Failed to send event to user {user_id}: {e}")
    
    def emit_to_room(self, room_name: str, event: WebSocketEvent, exclude_sender: bool = False):
        """向特定房间发送事件
        
        Args:
            room_name: 房间名称
            event: 事件对象
            exclude_sender: 是否排除发送者
        """
        if not self.socketio:
            return
        
        try:
            skip_sid = request.sid if exclude_sender and request else None
            self.socketio.emit('event', event.to_dict(), room=room_name, skip_sid=skip_sid)
            self._stats['total_messages_sent'] += 1
            logger.debug(f"Event {event.event_type.value} sent to room {room_name}")
        except Exception as e:
            logger.error(f"Failed to send event to room {room_name}: {e}")
    
    def emit_to_socket(self, socket_id: str, event: WebSocketEvent):
        """向特定 socket 发送事件
        
        Args:
            socket_id: Socket ID
            event: 事件对象
        """
        if not self.socketio:
            return
        
        try:
            self.socketio.emit('event', event.to_dict(), room=socket_id)
            self._stats['total_messages_sent'] += 1
            logger.debug(f"Event {event.event_type.value} sent to socket {socket_id}")
        except Exception as e:
            logger.error(f"Failed to send event to socket {socket_id}: {e}")
    
    def broadcast_event(self, event: WebSocketEvent, target_users: List[str] = None):
        """广播事件
        
        Args:
            event: 事件对象
            target_users: 目标用户列表（为空则广播给所有人）
        """
        if not self.socketio:
            return
        
        try:
            if target_users:
                for user_id in target_users:
                    self.emit_to_user(user_id, event)
            else:
                self.socketio.emit('event', event.to_dict())
                self._stats['total_messages_sent'] += 1
            
            logger.debug(f"Event {event.event_type.value} broadcasted")
        except Exception as e:
            logger.error(f"Failed to broadcast event: {e}")
    
    def broadcast(self, message: str, event_type: str = 'message'):
        """简单的广播消息方法（兼容性接口）
        
        Args:
            message: 消息内容或事件名称
            event_type: 事件类型或消息数据
        """
        if not self.socketio:
            return
        
        try:
            # 兼容两种调用方式
            if isinstance(event_type, dict):
                # broadcast(event_name, data)
                self.socketio.emit(message, event_type)
            else:
                # broadcast(message)
                self.socketio.emit(event_type, {'message': message})
            
            self._stats['total_messages_sent'] += 1
            logger.debug(f"Broadcast message: {message}")
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
    
    def send_to_user(self, user_id: str, event_name: str, data: Any):
        """向用户发送消息
        
        Args:
            user_id: 用户 ID
            event_name: 事件名称
            data: 消息数据
        """
        if not self.socketio:
            return
        
        try:
            room_name = f"user_{user_id}"
            self.socketio.emit(event_name, data, room=room_name)
            self._stats['total_messages_sent'] += 1
        except Exception as e:
            logger.error(f"Failed to send to user {user_id}: {e}")
    
    def create_room(self, room_name: str, metadata: Dict[str, Any] = None) -> bool:
        """创建房间
        
        Args:
            room_name: 房间名称
            metadata: 房间元数据
            
        Returns:
            是否成功
        """
        if room_name in self.rooms:
            return False
        
        self.rooms[room_name] = set()
        if metadata:
            self.room_metadata[room_name] = metadata
        
        logger.info(f"Room created: {room_name}")
        return True
    
    def delete_room(self, room_name: str) -> bool:
        """删除房间
        
        Args:
            room_name: 房间名称
            
        Returns:
            是否成功
        """
        if room_name not in self.rooms:
            return False
        
        # 将所有成员移出房间
        for socket_id in list(self.rooms[room_name]):
            self._leave_room_internal(socket_id, room_name)
        
        # 删除房间
        if room_name in self.rooms:
            del self.rooms[room_name]
        if room_name in self.room_metadata:
            del self.room_metadata[room_name]
        
        logger.info(f"Room deleted: {room_name}")
        return True
    
    def get_room_members(self, room_name: str) -> List[Dict[str, Any]]:
        """获取房间成员列表
        
        Args:
            room_name: 房间名称
            
        Returns:
            成员列表
        """
        members = []
        socket_ids = self.rooms.get(room_name, set())
        
        for socket_id in socket_ids:
            connection = self.connections.get(socket_id)
            if connection:
                members.append({
                    'socket_id': socket_id,
                    'user_id': connection.user_id,
                    'username': connection.username,
                    'joined_at': connection.connected_at.isoformat()
                })
        
        return members
    
    def disconnect_client(self, socket_id: str, reason: str = None):
        """断开客户端连接
        
        Args:
            socket_id: Socket ID
            reason: 断开原因
        """
        if not self.socketio or disconnect is None:
            return
        
        try:
            connection = self.connections.get(socket_id)
            if connection:
                connection.state = ConnectionState.DISCONNECTING
                
                # 发送断开通知
                self.socketio.emit('disconnecting', {
                    'reason': reason or 'server_disconnect',
                    'timestamp': datetime.utcnow().isoformat()
                }, room=socket_id)
            
            # 断开连接
            disconnect(sid=socket_id)
            
            logger.info(f"Client disconnected by server: socket_id={socket_id}, reason={reason}")
            
        except Exception as e:
            logger.error(f"Failed to disconnect client {socket_id}: {e}")
    
    def get_connection(self, socket_id: str) -> Optional[WebSocketConnection]:
        """获取连接信息
        
        Args:
            socket_id: Socket ID
            
        Returns:
            连接对象
        """
        return self.connections.get(socket_id)
    
    def get_user_connections(self, user_id: str) -> List[WebSocketConnection]:
        """获取用户的所有连接
        
        Args:
            user_id: 用户 ID
            
        Returns:
            连接列表
        """
        connections = []
        socket_ids = self.user_connections.get(user_id, set())
        
        for socket_id in socket_ids:
            connection = self.connections.get(socket_id)
            if connection:
                connections.append(connection)
        
        return connections
    
    def get_online_users(self) -> List[Dict[str, Any]]:
        """获取在线用户列表
        
        Returns:
            在线用户列表
        """
        online_users = []
        
        for user_id, socket_ids in self.user_connections.items():
            if not socket_ids:
                continue
            
                # 获取最新的连接信息
                latest_connection = None
                for socket_id in socket_ids:
                    connection = self.connections.get(socket_id)
                    if connection and (not latest_connection or 
                                     connection.last_activity > latest_connection.last_activity):
                        latest_connection = connection
                
                if latest_connection:
                    online_users.append({
                        'user_id': user_id,
                        'username': latest_connection.username,
                        'connected_at': latest_connection.connected_at.isoformat(),
                        'last_activity': latest_connection.last_activity.isoformat(),
                        'connection_count': len(socket_ids)
                    })
        
        return online_users
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息
        """
        return {
            **self._stats,
            'current_connections': len(self.connections),
            'online_users': len(self.user_connections),
            'active_rooms': len(self.rooms),
            'is_running': self._running
        }
    
    def get_room_list(self) -> List[Dict[str, Any]]:
        """获取房间列表
        
        Returns:
            房间列表
        """
        room_list = []
        for room_name, socket_ids in self.rooms.items():
            room_list.append({
                'name': room_name,
                'member_count': len(socket_ids),
                'metadata': self.room_metadata.get(room_name, {})
            })
        return room_list
    
    def is_user_online(self, user_id: str) -> bool:
        """检查用户是否在线
        
        Args:
            user_id: 用户 ID
            
        Returns:
            是否在线
        """
        return user_id in self.user_connections and len(self.user_connections[user_id]) > 0
    
    def graceful_shutdown(self, timeout: int = 30) -> bool:
        """优雅关闭WebSocket管理器
        
        Args:
            timeout: 关闭超时时间（秒）
            
        Returns:
            是否成功关闭
        """
        if self._shutting_down:
            return True
        
        self._shutting_down = True
        logger.info("Starting graceful shutdown of WebSocket manager...")
        
        try:
            # 1. 向所有连接的客户端发送服务停止通知
            shutdown_message = {
                'type': 'service_shutdown',
                'message': 'Server is shutting down, please save your work',
                'timestamp': datetime.utcnow().isoformat(),
                'countdown': timeout
            }
            
            try:
                self.broadcast('system_notification', shutdown_message)
                logger.info("Shutdown notification sent to all clients")
            except Exception as e:
                logger.warning(f"Failed to send shutdown notification: {e}")
            
            # 2. 等待客户端处理通知
            time.sleep(min(2, timeout // 10))
            
            # 3. 逐个断开用户连接
            disconnected_count = 0
            total_connections = len(self.connections)
            
            for socket_id, connection in list(self.connections.items()):
                try:
                    # 向用户发送个人断开通知
                    disconnect_message = {
                        'type': 'connection_closing',
                        'message': 'Your connection is being closed',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                    
                    if self.socketio:
                        self.socketio.emit('connection_status', disconnect_message, room=socket_id)
                    
                            # 从所有房间中移除
                    for room in list(connection.subscribed_rooms):
                        self._leave_room_internal(socket_id, room)
                            
                            # 断开连接
                    if self.socketio and disconnect:
                        disconnect(sid=socket_id)
                    
                    disconnected_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error disconnecting socket {socket_id}: {e}")
            
            # 4. 清理所有房间
                room_count = len(self.rooms)
                self.rooms.clear()
            self.room_metadata.clear()
            logger.info(f"Cleared {room_count} rooms")
            
            # 5. 清理连接数据
            self.connections.clear()
            self.user_connections.clear()
            self.session_connections.clear()
            logger.info(f"Disconnected {disconnected_count}/{total_connections} connections")
            
            # 6. 停止服务
            self._running = False
            
            # 7. 关闭SocketIO服务器
            if self.socketio:
                try:
                    self.socketio.emit('server_shutdown', {
                        'message': 'Server has been shut down',
                        'timestamp': datetime.utcnow().isoformat()
                    })
                    
                    if hasattr(self.socketio, 'stop'):
                        self.socketio.stop()
                    
                    logger.info("SocketIO server stopped")
                except Exception as e:
                    logger.warning(f"Error stopping SocketIO server: {e}")
            
            logger.info("WebSocket manager graceful shutdown completed")
            return True
            
        except Exception as e:
            logger.error(f"WebSocket manager shutdown failed: {e}")
            return False
        finally:
            self._shutting_down = False
    
    def force_disconnect_all(self) -> int:
        """强制断开所有连接（紧急情况使用）
        
        Returns:
            断开的连接数
        """
        try:
            total_disconnected = 0
            
            # 强制断开所有连接
            for socket_id in list(self.connections.keys()):
                try:
                    if self.socketio and disconnect:
                        disconnect(sid=socket_id)
                        total_disconnected += 1
                except Exception as e:
                    logger.warning(f"Error force disconnecting {socket_id}: {e}")
            
            # 清理所有数据
            self.connections.clear()
            self.user_connections.clear()
            self.session_connections.clear()
            self.rooms.clear()
            self.room_metadata.clear()
            
            logger.info(f"Force disconnected {total_disconnected} connections")
            return total_disconnected
            
        except Exception as e:
            logger.error(f"Force disconnect all failed: {e}")
            return 0


# 全局单例实例
websocket_manager = WebSocketManager()


def get_websocket_manager() -> WebSocketManager:
    """获取WebSocket管理器实例"""
    return websocket_manager

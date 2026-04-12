# -*- coding: utf-8 -*-
"""训练进度实时推送 API

提供训练进度的实时推送能力，支持多种通信方式：
- WebSocket（推荐）: 通过 Flask-SocketIO 实现全双工通信
- SSE (Server-Sent Events): 服务器单向推送，浏览器兼容性好
- 长轮询 (Long Polling): 降级方案，无额外依赖

架构：
API层 (本模块)
    -> Service层 (training_progress_service.py)
        -> Repository层 (training_progress_repository.py)
        -> WebSocket管理器 (websocket/websocket_manager.py)
"""

import time
import threading
import logging
import json
from typing import Dict, Any, Optional, Set, Callable
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict
from flask import Blueprint, jsonify, request, Response, stream_with_context

logger = logging.getLogger(__name__)

# 创建蓝图
progress_ws_bp = Blueprint(
    "training_progress_ws_api",
    __name__,
    url_prefix="/api/v1/training/progress/realtime"
)


# ==================== 订阅管理器 ====================

@dataclass
class ProgressSubscription:
    """进度订阅信息"""
    session_id: str
    user_id: str
    subscribed_at: datetime = field(default_factory=datetime.utcnow)
    last_push_at: Optional[datetime] = None
    push_count: int = 0


class ProgressPushManager:
    """
    进度推送管理器
    
    管理订阅、推送和广播逻辑
    """
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
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
        
        # 订阅管理：session_id -> set of user_ids
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # 用户订阅详情：(session_id, user_id) -> ProgressSubscription
        self._subscription_details: Dict[tuple, ProgressSubscription] = {}
        
        # SSE 客户端队列：(session_id, user_id) -> list of progress updates
        self._sse_queues: Dict[tuple, list] = defaultdict(list)
        
        # 回调函数
        self._push_callbacks: Dict[str, Callable] = {}
        
        # WebSocket 管理器引用
        self._ws_manager = None
        
        # 统计
        self._stats = {
            'total_pushes': 0,
            'total_subscriptions': 0,
            'active_subscriptions': 0
        }
        
        self._initialized = True
        logger.info("ProgressPushManager initialized")
    
    def _get_ws_manager(self):
        """获取 WebSocket 管理器"""
        if self._ws_manager is None:
            try:
                from backend.websocket.websocket_manager import get_websocket_manager
                self._ws_manager = get_websocket_manager()
            except ImportError:
                logger.warning("WebSocket manager not available")
        return self._ws_manager
    
    def subscribe(self, session_id: str, user_id: str) -> bool:
        """
        订阅训练进度
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            
        Returns:
            是否订阅成功
        """
        key = (session_id, user_id)
        
        if key not in self._subscription_details:
            self._subscriptions[session_id].add(user_id)
            self._subscription_details[key] = ProgressSubscription(
                session_id=session_id,
                user_id=user_id
            )
            self._stats['total_subscriptions'] += 1
            self._stats['active_subscriptions'] = len(self._subscription_details)
            
            # 加入 WebSocket 房间
            ws_manager = self._get_ws_manager()
            if ws_manager and ws_manager.socketio:
                room_name = f"training_progress_{session_id}"
                # 注意：实际加入房间需要在 WebSocket 连接上下文中
                logger.info(f"User {user_id} subscribed to progress room: {room_name}")
            
            logger.info(f"User {user_id} subscribed to session {session_id}")
            return True
        
        return False
    
    def unsubscribe(self, session_id: str, user_id: str) -> bool:
        """
        取消订阅
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            
        Returns:
            是否取消成功
        """
        key = (session_id, user_id)
        
        if key in self._subscription_details:
            self._subscriptions[session_id].discard(user_id)
            del self._subscription_details[key]
            
            # 清理 SSE 队列
            if key in self._sse_queues:
                del self._sse_queues[key]
            
            self._stats['active_subscriptions'] = len(self._subscription_details)
            
            logger.info(f"User {user_id} unsubscribed from session {session_id}")
            return True
        
        return False
    
    def push_progress(self, session_id: str, progress_data: Dict[str, Any]) -> int:
        """
        推送进度更新
        
        Args:
            session_id: 训练会话ID
            progress_data: 进度数据
            
        Returns:
            推送到的用户数量
        """
        subscribers = self._subscriptions.get(session_id, set())
        if not subscribers:
            return 0
        
        push_count = 0
        now = datetime.utcnow()
        
        # 构建推送消息
        message = {
            'type': 'progress_update',
            'session_id': session_id,
            'timestamp': now.isoformat(),
            'data': progress_data
        }
        
        # 1. 尝试 WebSocket 推送
        ws_manager = self._get_ws_manager()
        if ws_manager and ws_manager.socketio:
            try:
                room_name = f"training_progress_{session_id}"
                ws_manager.socketio.emit('training_progress', message, room=room_name)
                push_count = len(subscribers)
                logger.debug(f"WebSocket push to room {room_name}: {push_count} subscribers")
            except Exception as e:
                logger.warning(f"WebSocket push failed: {e}")
        
        # 2. 更新 SSE 队列
        for user_id in subscribers:
            key = (session_id, user_id)
            self._sse_queues[key].append(message)
            
            # 限制队列长度
            if len(self._sse_queues[key]) > 100:
                self._sse_queues[key] = self._sse_queues[key][-50:]
            
            # 更新订阅详情
            if key in self._subscription_details:
                sub = self._subscription_details[key]
                sub.last_push_at = now
                sub.push_count += 1
        
        # 3. 调用注册的回调
        for callback_name, callback in self._push_callbacks.items():
            try:
                callback(session_id, message)
            except Exception as e:
                logger.error(f"Push callback {callback_name} error: {e}")
        
        self._stats['total_pushes'] += 1
        return push_count
    
    def get_sse_updates(
        self,
        session_id: str,
        user_id: str,
        timeout: float = 30.0
    ) -> list:
        """
        获取 SSE 更新（阻塞式）
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            timeout: 超时时间（秒）
            
        Returns:
            更新列表
        """
        key = (session_id, user_id)
        
        # 等待新消息
        start_time = time.time()
        while time.time() - start_time < timeout:
            if key in self._sse_queues and self._sse_queues[key]:
                messages = self._sse_queues[key]
                self._sse_queues[key] = []
                return messages
            time.sleep(0.5)
        
        return []
    
    def pop_sse_updates(self, session_id: str, user_id: str) -> list:
        """
        获取并清空 SSE 更新（非阻塞）
        
        Args:
            session_id: 训练会话ID
            user_id: 用户ID
            
        Returns:
            更新列表
        """
        key = (session_id, user_id)
        
        if key in self._sse_queues and self._sse_queues[key]:
            messages = self._sse_queues[key]
            self._sse_queues[key] = []
            return messages
        
        return []
    
    def register_push_callback(self, name: str, callback: Callable):
        """注册推送回调"""
        self._push_callbacks[name] = callback
        logger.info(f"Push callback registered: {name}")
    
    def unregister_push_callback(self, name: str):
        """取消注册推送回调"""
        if name in self._push_callbacks:
            del self._push_callbacks[name]
    
    def get_subscribers(self, session_id: str) -> Set[str]:
        """获取会话的订阅者"""
        return self._subscriptions.get(session_id, set()).copy()
    
    def get_subscription_info(
        self,
        session_id: str,
        user_id: str
    ) -> Optional[ProgressSubscription]:
        """获取订阅详情"""
        return self._subscription_details.get((session_id, user_id))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            'sessions_with_subscribers': len(self._subscriptions),
            'sse_queues_active': len(self._sse_queues)
        }


# 全局推送管理器实例
_push_manager = ProgressPushManager()


def get_push_manager() -> ProgressPushManager:
    """获取推送管理器实例"""
    return _push_manager


# ==================== JWT 认证 ====================

try:
    from flask_jwt_extended import jwt_required, get_jwt_identity
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    
    def jwt_required():
        def decorator(f):
            return f
        return decorator
    
    def get_jwt_identity():
        return 'anonymous_user'


def _get_current_user() -> str:
    """获取当前用户ID"""
    if JWT_AVAILABLE:
        identity = get_jwt_identity()
        if isinstance(identity, dict):
            return identity.get('user_id', identity.get('id', 'anonymous'))
        return str(identity) if identity else 'anonymous'
    return 'anonymous_user'


def _get_tenant_id() -> str:
    """获取当前租户ID"""
    tenant_id = request.headers.get('X-Tenant-ID')
    if tenant_id:
        return tenant_id
    
    if JWT_AVAILABLE:
        identity = get_jwt_identity()
        if isinstance(identity, dict):
            return identity.get('tenant_id', 'default')
    
    return 'default'


# ==================== 订阅管理 API ====================

@progress_ws_bp.route('/subscribe/<session_id>', methods=['POST'])
@jwt_required()
def subscribe_progress(session_id: str):
    """
    订阅训练进度
    
    客户端调用此接口后，可以通过以下方式接收更新：
    1. WebSocket: 连接后加入 `training_progress_{session_id}` 房间
    2. SSE: 调用 /stream/<session_id> 端点
    3. 长轮询: 调用 /poll/<session_id> 端点
    
    Returns:
        {
            "success": true,
            "data": {
                "session_id": "uuid",
                "subscribed": true,
                "channels": {
                    "websocket": "training_progress_uuid",
                    "sse": "/api/v1/training/progress/realtime/stream/uuid",
                    "poll": "/api/v1/training/progress/realtime/poll/uuid"
                }
            },
            "message": "订阅成功"
        }
    """
    user_id = _get_current_user()
    
    push_manager = get_push_manager()
    success = push_manager.subscribe(session_id, user_id)
    
    data = {
        'session_id': session_id,
        'subscribed': success,
        'channels': {
            'websocket': f"training_progress_{session_id}",
            'sse': f"/api/v1/training/progress/realtime/stream/{session_id}",
            'poll': f"/api/v1/training/progress/realtime/poll/{session_id}"
        }
    }
    
    return jsonify({
        'success': True,
        'data': data,
        'message': '订阅成功' if success else '已订阅'
    })


@progress_ws_bp.route('/unsubscribe/<session_id>', methods=['POST'])
@jwt_required()
def unsubscribe_progress(session_id: str):
    """
    取消订阅训练进度
    
    Returns:
        {
            "success": true,
            "data": {"session_id": "uuid", "unsubscribed": true},
            "message": "取消订阅成功"
        }
    """
    user_id = _get_current_user()
    
    push_manager = get_push_manager()
    success = push_manager.unsubscribe(session_id, user_id)
    
    return jsonify({
        'success': True,
        'data': {'session_id': session_id, 'unsubscribed': success},
        'message': '取消订阅成功' if success else '未订阅'
    })


@progress_ws_bp.route('/subscriptions', methods=['GET'])
@jwt_required()
def get_subscriptions():
    """
    获取当前用户的订阅列表
    
    Returns:
        {
            "success": true,
            "data": {
                "subscriptions": [
                    {
                        "session_id": "uuid",
                        "subscribed_at": "2025-01-09T10:00:00Z",
                        "last_push_at": "2025-01-09T10:05:00Z",
                        "push_count": 50
                    }
                ]
            },
            "message": "获取订阅列表成功"
        }
    """
    user_id = _get_current_user()
    push_manager = get_push_manager()
    
    subscriptions = []
    for (sid, uid), sub in push_manager._subscription_details.items():
        if uid == user_id:
            subscriptions.append({
                'session_id': sub.session_id,
                'subscribed_at': sub.subscribed_at.isoformat(),
                'last_push_at': sub.last_push_at.isoformat() if sub.last_push_at else None,
                'push_count': sub.push_count
            })
    
    return jsonify({
        'success': True,
        'data': {'subscriptions': subscriptions},
        'message': '获取订阅列表成功'
    })


# ==================== 进度推送 API ====================

@progress_ws_bp.route('/push/<session_id>', methods=['POST'])
@jwt_required()
def push_progress(session_id: str):
    """
    推送进度更新（供训练模块内部调用）
    
    Request Body:
        {
            "stage": "finetune",
            "epoch": 5,
            "step": 500,
            "total_steps": 1000,
            "loss": 0.15,
            "accuracy": 0.92,
            "learning_rate": 1e-4,
            "metrics": {...}
        }
    
    Returns:
        {
            "success": true,
            "data": {"session_id": "uuid", "pushed_to": 3},
            "message": "推送成功"
        }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': '请求数据不能为空'}), 400
    
    push_manager = get_push_manager()
    
    # 推送到所有订阅者
    pushed_count = push_manager.push_progress(session_id, data)
    
    # 同时持久化到数据库（可选）
    try:
        from backend.services.training_progress_service import get_training_progress_service
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_progress_service()
        service.update_progress(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            progress_data=data
        )
    except Exception as e:
        logger.warning(f"Failed to persist progress: {e}")
    
    return jsonify({
        'success': True,
        'data': {'session_id': session_id, 'pushed_to': pushed_count},
        'message': f'已推送到 {pushed_count} 个订阅者'
    })


# ==================== SSE 流式推送 ====================

@progress_ws_bp.route('/stream/<session_id>', methods=['GET'])
@jwt_required()
def stream_progress(session_id: str):
    """
    SSE (Server-Sent Events) 流式推送
    
    客户端使用 EventSource API 连接：
    ```javascript
    const eventSource = new EventSource('/api/v1/training/progress/realtime/stream/uuid');
    eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);
        console.log('Progress update:', data);
    };
    ```
    
    Returns:
        SSE 事件流
    """
    user_id = _get_current_user()
    push_manager = get_push_manager()
    
    # 确保订阅
    push_manager.subscribe(session_id, user_id)
    
    def generate():
        """生成 SSE 事件流"""
        # 发送初始连接成功消息
        yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"
        
        # 发送初始进度（如果有）
        try:
            from backend.services.training_progress_service import get_training_progress_service
            service = get_training_progress_service()
            initial_progress = service.get_progress(session_id, user_id)
            if initial_progress:
                yield f"data: {json.dumps({'type': 'initial', 'data': initial_progress})}\n\n"
        except Exception as e:
            logger.warning(f"Failed to get initial progress: {e}")
        
        # 持续推送更新
        last_heartbeat = time.time()
        while True:
            try:
                # 获取更新
                updates = push_manager.pop_sse_updates(session_id, user_id)
                
                for update in updates:
                    yield f"data: {json.dumps(update)}\n\n"
                
                # 心跳（每 15 秒）
                if time.time() - last_heartbeat > 15:
                    yield f"data: {json.dumps({'type': 'heartbeat', 'timestamp': time.time()})}\n\n"
                    last_heartbeat = time.time()
                
                # 检查是否仍然订阅
                if not push_manager.get_subscription_info(session_id, user_id):
                    yield f"data: {json.dumps({'type': 'unsubscribed'})}\n\n"
                    break
                
                time.sleep(0.5)
                
            except GeneratorExit:
                # 客户端断开连接
                logger.info(f"SSE client disconnected: session={session_id}, user={user_id}")
                push_manager.unsubscribe(session_id, user_id)
                break
            except Exception as e:
                logger.error(f"SSE stream error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
                break
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'  # 禁用 nginx 缓冲
        }
    )


# ==================== 长轮询 ====================

@progress_ws_bp.route('/poll/<session_id>', methods=['GET'])
@jwt_required()
def poll_progress(session_id: str):
    """
    长轮询获取进度更新
    
    Query Parameters:
        - timeout: 超时时间（秒），默认 30
        - limit: 返回数量限制，默认 50
        - since: 只返回此时间戳之后的更新（毫秒）
    
    Returns:
        {
            "success": true,
            "data": {
                "session_id": "uuid",
                "updates": [
                    {
                        "type": "progress_update",
                        "timestamp": "2025-01-09T10:00:00Z",
                        "data": {...}
                    }
                ],
                "has_more": false
            },
            "message": "获取成功"
        }
    """
    user_id = _get_current_user()
    timeout = request.args.get('timeout', 30, type=float)
    limit = request.args.get('limit', 50, type=int)
    since = request.args.get('since', type=float)  # 毫秒时间戳
    
    push_manager = get_push_manager()
    
    # 确保订阅
    push_manager.subscribe(session_id, user_id)
    
    # 首先尝试从 SSE 队列获取
    updates = push_manager.pop_sse_updates(session_id, user_id)
    
    # 如果没有更新，从数据库获取最新数据
    if not updates:
        try:
            from backend.services.training_progress_service import get_training_progress_service
            from backend.repositories.training_progress_repository import get_training_progress_repository
            
            repo = get_training_progress_repository()
            records, total = repo.get_progress_history(session_id, limit, 0)
            
            for r in records:
                ts = getattr(r, 'created_at', None)
                if ts:
                    ts_ms = ts.timestamp() * 1000
                    # 过滤 since 之后的记录
                    if since and ts_ms <= since:
                        continue
                
                updates.append({
                    'type': 'progress_update',
                    'session_id': session_id,
                    'timestamp': ts.isoformat() if ts else None,
                    'data': {
                        'epoch': getattr(r, 'epoch', None),
                        'step': getattr(r, 'step', None),
                        'total_steps': getattr(r, 'total_steps', None),
                        'loss': getattr(r, 'loss', None),
                        'accuracy': getattr(r, 'accuracy', None),
                        'learning_rate': getattr(r, 'learning_rate', None),
                        'stage': getattr(r, 'stage', None),
                        'metrics': getattr(r, 'metrics', None)
                    }
                })
            
            # 反转为时间正序
            updates.reverse()
            
        except Exception as e:
            logger.warning(f"Failed to get progress from database: {e}")
    
    return jsonify({
        'success': True,
        'data': {
            'session_id': session_id,
            'updates': updates[:limit],
            'has_more': len(updates) > limit
        },
        'message': '获取成功'
    })


# ==================== 批量推送 ====================

@progress_ws_bp.route('/push/batch', methods=['POST'])
@jwt_required()
def push_batch_progress():
    """
    批量推送进度更新（供训练模块高频上报使用）
    
    Request Body:
        {
            "session_id": "uuid",
            "updates": [
                {"epoch": 1, "step": 100, "loss": 0.5, ...},
                {"epoch": 1, "step": 200, "loss": 0.45, ...}
            ]
        }
    
    Returns:
        {
            "success": true,
            "data": {"session_id": "uuid", "pushed_count": 2},
            "message": "批量推送成功"
        }
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': '请求数据不能为空'}), 400
    
    session_id = data.get('session_id')
    updates = data.get('updates', [])
    
    if not session_id:
        return jsonify({'success': False, 'error': 'session_id 不能为空'}), 400
    
    push_manager = get_push_manager()
    pushed_count = 0
    
    for update in updates:
        count = push_manager.push_progress(session_id, update)
        if count > 0:
            pushed_count += 1
    
    return jsonify({
        'success': True,
        'data': {'session_id': session_id, 'pushed_count': pushed_count},
        'message': f'批量推送成功，共 {pushed_count} 条'
    })


# ==================== 统计与监控 ====================

@progress_ws_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_push_stats():
    """
    获取推送统计信息
    
    Returns:
        {
            "success": true,
            "data": {
                "total_pushes": 1000,
                "total_subscriptions": 50,
                "active_subscriptions": 10,
                "sessions_with_subscribers": 5,
                "sse_queues_active": 3
            },
            "message": "获取统计成功"
        }
    """
    push_manager = get_push_manager()
    stats = push_manager.get_stats()
    
    return jsonify({
        'success': True,
        'data': stats,
        'message': '获取统计成功'
    })


@progress_ws_bp.route('/subscribers/<session_id>', methods=['GET'])
@jwt_required()
def get_session_subscribers(session_id: str):
    """
    获取会话的订阅者列表
    
    Returns:
        {
            "success": true,
            "data": {
                "session_id": "uuid",
                "subscribers": ["user1", "user2"],
                "count": 2
            },
            "message": "获取成功"
        }
    """
    push_manager = get_push_manager()
    subscribers = push_manager.get_subscribers(session_id)
    
    return jsonify({
        'success': True,
        'data': {
            'session_id': session_id,
            'subscribers': list(subscribers),
            'count': len(subscribers)
        },
        'message': '获取成功'
    })


# ==================== 健康检查 ====================

@progress_ws_bp.route('/health', methods=['GET'])
def health_check():
    """实时推送服务健康检查"""
    push_manager = get_push_manager()
    stats = push_manager.get_stats()
    
    # 检查 WebSocket 可用性
    ws_available = False
    try:
        ws_manager = push_manager._get_ws_manager()
        ws_available = ws_manager is not None and ws_manager.socketio is not None
    except Exception:
        pass
    
    return jsonify({
        'success': True,
        'data': {
            'status': 'healthy',
            'service': 'training_progress_realtime',
            'websocket_available': ws_available,
            'active_subscriptions': stats.get('active_subscriptions', 0),
            'timestamp': datetime.utcnow().isoformat()
        },
        'message': 'Service is healthy'
    })


# ==================== WebSocket 事件注册 ====================

def register_progress_websocket_handlers():
    """
    注册训练进度 WebSocket 事件处理器
    
    应在 WebSocket 初始化后调用
    """
    try:
        from backend.websocket.websocket_manager import get_websocket_manager
        from backend.websocket.events import EventType, WebSocketEvent
        
        ws_manager = get_websocket_manager()
        
        if not ws_manager.socketio:
            logger.warning("SocketIO not initialized, skipping progress handlers registration")
            return False
        
        socketio = ws_manager.socketio
        
        @socketio.on('subscribe_progress')
        def handle_subscribe_progress(data):
            """处理订阅进度请求"""
            from flask import request
            
            session_id = data.get('session_id')
            if not session_id:
                return {'success': False, 'error': 'session_id required'}
            
            # 获取用户ID
            user_id = data.get('user_id', 'anonymous')
            
            # 加入房间
            room_name = f"training_progress_{session_id}"
            try:
                from flask_socketio import join_room
                join_room(room_name)
            except Exception as e:
                logger.error(f"Failed to join room: {e}")
            
            # 订阅
            push_manager = get_push_manager()
            push_manager.subscribe(session_id, user_id)
            
            logger.info(f"WebSocket client subscribed to progress: {session_id}")
            
            return {
                'success': True,
                'room': room_name,
                'message': f'Subscribed to {session_id}'
            }
        
        @socketio.on('unsubscribe_progress')
        def handle_unsubscribe_progress(data):
            """处理取消订阅请求"""
            session_id = data.get('session_id')
            if not session_id:
                return {'success': False, 'error': 'session_id required'}
            
            user_id = data.get('user_id', 'anonymous')
            
            # 离开房间
            room_name = f"training_progress_{session_id}"
            try:
                from flask_socketio import leave_room
                leave_room(room_name)
            except Exception as e:
                logger.error(f"Failed to leave room: {e}")
            
            # 取消订阅
            push_manager = get_push_manager()
            push_manager.unsubscribe(session_id, user_id)
            
            return {
                'success': True,
                'message': f'Unsubscribed from {session_id}'
            }
        
        @socketio.on('get_progress')
        def handle_get_progress(data):
            """处理获取进度请求"""
            session_id = data.get('session_id')
            user_id = data.get('user_id', 'anonymous')
            
            if not session_id:
                return {'success': False, 'error': 'session_id required'}
            
            try:
                from backend.services.training_progress_service import get_training_progress_service
                service = get_training_progress_service()
                progress = service.get_progress(session_id, user_id)
                
                return {
                    'success': True,
                    'data': progress,
                    'message': 'Progress retrieved'
                }
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        logger.info("Training progress WebSocket handlers registered")
        return True
        
    except ImportError as e:
        logger.warning(f"Failed to register WebSocket handlers: {e}")
        return False
    except Exception as e:
        logger.error(f"Error registering WebSocket handlers: {e}")
        return False


# ==================== 服务集成接口 ====================

def push_training_progress(session_id: str, progress_data: Dict[str, Any]) -> int:
    """
    推送训练进度（供服务层调用）
    
    Args:
        session_id: 训练会话ID
        progress_data: 进度数据
        
    Returns:
        推送到的用户数量
    """
    push_manager = get_push_manager()
    return push_manager.push_progress(session_id, progress_data)


def subscribe_to_progress(session_id: str, user_id: str) -> bool:
    """
    订阅训练进度（供服务层调用）
    
    Args:
        session_id: 训练会话ID
        user_id: 用户ID
        
    Returns:
        是否订阅成功
    """
    push_manager = get_push_manager()
    return push_manager.subscribe(session_id, user_id)


def unsubscribe_from_progress(session_id: str, user_id: str) -> bool:
    """
    取消订阅训练进度（供服务层调用）
    
    Args:
        session_id: 训练会话ID
        user_id: 用户ID
        
    Returns:
        是否取消成功
    """
    push_manager = get_push_manager()
    return push_manager.unsubscribe(session_id, user_id)

"""WebSocket事件定义模块"""
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass


class EventType(Enum):
    """WebSocket事件类型枚举"""
    # 连接相关事件
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    
    # 消息相关事件
    MESSAGE = "message"
    BROADCAST = "broadcast"
    
    # 智能体相关事件
    AGENT_EXECUTION_STARTED = "agent_execution_started"
    AGENT_EXECUTION_COMPLETED = "agent_execution_completed"
    AGENT_STATUS_CHANGED = "agent_status_changed"
    
    # 工作流相关事件
    WORKFLOW_EXECUTION_STARTED = "workflow_execution_started"
    WORKFLOW_EXECUTION_PROGRESS = "workflow_execution_progress"
    WORKFLOW_EXECUTION_COMPLETED = "workflow_execution_completed"
    
    # 训练相关事件
    TRAINING_START = "training_start"
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETE = "training_complete"
    TRAINING_ERROR = "training_error"
    TRAINING_TASK_STARTED = "training_task_started"
    TRAINING_TASK_PROGRESS = "training_task_progress"
    TRAINING_TASK_COMPLETED = "training_task_completed"
    TRAINING_METRICS_UPDATED = "training_metrics_updated"
    
    # 对话相关事件
    DIALOGUE_MESSAGE_RECEIVED = "dialogue_message_received"
    
    # 系统相关事件
    SYSTEM_STATUS = "system_status"
    SYSTEM_STATUS_CHANGED = "system_status_changed"
    SYSTEM_ALERT = "system_alert"
    SYSTEM_NOTIFICATION = "system_notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class WebSocketEvent:
    """WebSocket事件数据类"""
    event_type: EventType
    data: Optional[Dict[str, Any]] = None
    client_id: Optional[str] = None
    timestamp: Optional[float] = None
    target_users: Optional[list] = None  # 目标用户列表
    target_rooms: Optional[list] = None  # 目标房间列表
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "event_type": self.event_type.value,
            "data": self.data,
            "client_id": self.client_id,
            "timestamp": self.timestamp
        }
        if self.target_users:
            result["target_users"] = self.target_users
        if self.target_rooms:
            result["target_rooms"] = self.target_rooms
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSocketEvent':
        """从字典创建事件对象"""
        return cls(
            event_type=EventType(data["event_type"]),
            data=data.get("data"),
            client_id=data.get("client_id"),
            timestamp=data.get("timestamp"),
            target_users=data.get("target_users"),
            target_rooms=data.get("target_rooms")
        )


# 事件创建辅助函数
def create_agent_event(event_type: EventType, agent_id: str, data: Dict[str, Any], user_id: str = None) -> WebSocketEvent:
    """创建智能体相关事件"""
    import time
    
    event_data = {
        'agent_id': agent_id,
        **data
    }
    
    return WebSocketEvent(
        event_type=event_type,
        data=event_data,
        client_id=user_id,
        timestamp=time.time()
    )


def create_workflow_event(event_type: EventType, workflow_id: str, data: Dict[str, Any], user_id: str = None) -> WebSocketEvent:
    """创建工作流相关事件"""
    import time
    
    event_data = {
        'workflow_id': workflow_id,
        **data
    }
    
    return WebSocketEvent(
        event_type=event_type,
        data=event_data,
        client_id=user_id,
        timestamp=time.time()
    )


def create_training_event(event_type: EventType, task_id: str, data: Dict[str, Any], user_id: str = None) -> WebSocketEvent:
    """创建训练相关事件"""
    import time
    
    event_data = {
        'task_id': task_id,
        **data
    }
    
    return WebSocketEvent(
        event_type=event_type,
        data=event_data,
        client_id=user_id,
        timestamp=time.time()
    )


def create_dialogue_event(event_type: EventType, session_id: str, data: Dict[str, Any], user_id: str = None) -> WebSocketEvent:
    """创建对话相关事件"""
    import time
    
    event_data = {
        'session_id': session_id,
        **data
    }
    
    return WebSocketEvent(
        event_type=event_type,
        data=event_data,
        client_id=user_id,
        timestamp=time.time()
    )


def create_system_event(event_type: EventType, data: Dict[str, Any], user_id: str = None) -> WebSocketEvent:
    """创建系统相关事件"""
    import time
    
    return WebSocketEvent(
        event_type=event_type,
        data=data,
        client_id=user_id,
        timestamp=time.time()
    )
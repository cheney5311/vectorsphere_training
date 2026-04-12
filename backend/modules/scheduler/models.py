# -*- coding: utf-8 -*-
"""
Scheduler模块数据模型
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum


class TaskStatus(Enum):
    """任务状态枚举"""
    SCHEDULED = "scheduled"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级枚举"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ScheduledTask:
    """调度任务数据模型"""
    id: str
    config: Dict[str, Any]
    schedule_time: datetime
    status: TaskStatus
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = None
    updated_at: datetime = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'config': self.config,
            'schedule_time': self.schedule_time.isoformat() if self.schedule_time else None,
            'status': self.status.value,
            'priority': self.priority.value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'result': self.result,
            'error_message': self.error_message
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScheduledTask':
        """从字典创建实例"""
        data = data.copy()
        if 'schedule_time' in data and data['schedule_time']:
            data['schedule_time'] = datetime.fromisoformat(data['schedule_time'])
        if 'created_at' in data and data['created_at']:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and data['updated_at']:
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if 'status' in data:
            data['status'] = TaskStatus(data['status'])
        if 'priority' in data:
            data['priority'] = TaskPriority(data['priority'])
        return cls(**data)
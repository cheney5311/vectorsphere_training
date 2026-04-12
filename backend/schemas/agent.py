from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid

from .agent_type import AgentType


@dataclass
class Agent:
    """智能体数据传输对象
    
    用于在服务层和API层之间传递智能体信息。
    """
    # 主键使用 agent_id，兼容仓库实现
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # 兼容旧代码：提供只读 id 属性映射到 agent_id
    @property
    def id(self) -> str:
        return self.agent_id

    user_id: str = ""
    name: str = ""
    description: Optional[str] = None
    version: str = "1.0.0"
    status: str = "active"  # active, inactive, deleted
    config: Dict[str, Any] = field(default_factory=dict)
    agent_type: Optional[AgentType] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    capabilities: List[str] = field(default_factory=list)
    
    # 兼容旧代码的 active 属性
    @property
    def active(self) -> bool:
        return self.status == "active"

    def activate(self) -> None:
        """激活智能体"""
        self.status = "active"

    def deactivate(self) -> None:
        """停用智能体"""
        self.status = "inactive"

    def add_capability(self, capability: str) -> None:
        """添加能力"""
        if capability and capability not in self.capabilities:
            self.capabilities.append(capability)

    def remove_capability(self, capability: str) -> None:
        """移除能力"""
        if capability in self.capabilities:
            self.capabilities.remove(capability)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'agent_id': self.agent_id,
            'id': self.agent_id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'status': self.status,
            'config': self.config or {},
            'agent_type': self.agent_type.value if self.agent_type else None,
            'capabilities': self.capabilities or [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }
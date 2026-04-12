"""智能体数据库模型

定义智能体相关的 SQLAlchemy ORM 模型，支持：
- 智能体基本信息
- 会话管理
- 消息历史
- 长期记忆
- 执行记录
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Boolean, Integer, Float, DateTime, 
    ForeignKey, Index, func, JSON
)
from sqlalchemy.orm import relationship

from backend.schemas.base_models import Base, GUID, UUIDMixin, TimestampMixin, TenantMixin


# ============================================================================
# 智能体实体模型
# ============================================================================

class AgentEntity(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """智能体实体
    
    存储智能体的基本配置和状态信息。
    
    Attributes:
        id: 智能体唯一标识 (UUID)
        user_id: 所属用户ID
        name: 智能体名称
        description: 智能体描述
        agent_type: 智能体类型 (training_assistant, chat, reasoning 等)
        version: 版本号
        status: 状态 (active, inactive, deleted)
        config: 配置信息 (JSON)
        capabilities: 能力列表 (JSON数组)
        system_prompt: 系统提示词
        model_config: 模型配置 (JSON)
        is_public: 是否公开
        execution_count: 执行次数
        success_rate: 成功率
        avg_response_time: 平均响应时间(ms)
    """
    __tablename__ = 'agents'
    
    user_id = Column(String(36), nullable=False, index=True, comment="所属用户ID")
    name = Column(String(200), nullable=False, comment="智能体名称")
    description = Column(Text, comment="智能体描述")
    agent_type = Column(String(50), default='chat', index=True, comment="智能体类型")
    version = Column(String(20), default='1.0.0', comment="版本号")
    status = Column(String(20), default='active', index=True, comment="状态")
    config = Column(JSON, default=dict, comment="配置信息")
    capabilities = Column(JSON, default=list, comment="能力列表")
    system_prompt = Column(Text, comment="系统提示词")
    model_config = Column(JSON, default=dict, comment="模型配置")
    is_public = Column(Boolean, default=False, index=True, comment="是否公开")
    execution_count = Column(Integer, default=0, comment="执行次数")
    success_rate = Column(Float, default=0.0, comment="成功率")
    avg_response_time = Column(Float, default=0.0, comment="平均响应时间(ms)")
    
    # 关系
    sessions = relationship("AgentSessionEntity", back_populates="agent", cascade="all, delete-orphan")
    memories = relationship("AgentMemoryEntity", back_populates="agent", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_agents_user_status', 'user_id', 'status'),
        Index('ix_agents_type_public', 'agent_type', 'is_public'),
    )
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'agent_id': str(self.id) if self.id else None,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'agent_type': self.agent_type,
            'version': self.version,
            'status': self.status,
            'config': self.config or {},
            'capabilities': self.capabilities or [],
            'system_prompt': self.system_prompt,
            'model_config': self.model_config or {},
            'is_public': self.is_public,
            'execution_count': self.execution_count,
            'success_rate': self.success_rate,
            'avg_response_time': self.avg_response_time,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class AgentSessionEntity(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """智能体会话实体
    
    存储智能体与用户之间的会话信息。
    
    Attributes:
        id: 会话唯一标识 (UUID)
        agent_id: 智能体ID
        user_id: 用户ID
        title: 会话标题
        status: 状态 (active, completed, archived)
        context: 会话上下文 (JSON)
        metadata: 元数据 (JSON)
        message_count: 消息数量
        last_message_at: 最后消息时间
        summary: 会话摘要
    """
    __tablename__ = 'agent_sessions'
    
    agent_id = Column(GUID(), ForeignKey('agents.id', ondelete='CASCADE'), nullable=False, index=True, comment="智能体ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    title = Column(String(500), comment="会话标题")
    status = Column(String(20), default='active', index=True, comment="状态")
    context = Column(JSON, default=dict, comment="会话上下文")
    session_metadata = Column(JSON, default=dict, comment="元数据")
    message_count = Column(Integer, default=0, comment="消息数量")
    last_message_at = Column(DateTime, comment="最后消息时间")
    summary = Column(Text, comment="会话摘要")
    
    # 关系
    agent = relationship("AgentEntity", back_populates="sessions")
    messages = relationship("AgentMessageEntity", back_populates="session", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_agent_sessions_agent_user', 'agent_id', 'user_id'),
        Index('ix_agent_sessions_status_last', 'status', 'last_message_at'),
    )
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'session_id': str(self.id) if self.id else None,
            'agent_id': str(self.agent_id) if self.agent_id else None,
            'user_id': self.user_id,
            'title': self.title,
            'status': self.status,
            'context': self.context or {},
            'metadata': self.session_metadata or {},
            'message_count': self.message_count,
            'last_message_at': self.last_message_at.isoformat() if self.last_message_at else None,
            'summary': self.summary,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class AgentMessageEntity(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """智能体消息实体
    
    存储会话中的消息记录。
    
    Attributes:
        id: 消息唯一标识 (UUID)
        session_id: 会话ID
        role: 角色 (user, assistant, system, function)
        content: 消息内容
        content_type: 内容类型 (text, image, code, tool_call)
        tokens_used: 使用的token数
        latency_ms: 响应延迟(ms)
        metadata: 元数据 (JSON)
        tool_calls: 工具调用 (JSON)
        function_call: 函数调用 (JSON)
        parent_message_id: 父消息ID (用于分支对话)
    """
    __tablename__ = 'agent_messages'
    
    session_id = Column(GUID(), ForeignKey('agent_sessions.id', ondelete='CASCADE'), nullable=False, index=True, comment="会话ID")
    role = Column(String(20), nullable=False, index=True, comment="角色")
    content = Column(Text, nullable=False, comment="消息内容")
    content_type = Column(String(20), default='text', comment="内容类型")
    tokens_used = Column(Integer, default=0, comment="使用的token数")
    latency_ms = Column(Integer, default=0, comment="响应延迟(ms)")
    message_metadata = Column(JSON, default=dict, comment="元数据")
    tool_calls = Column(JSON, comment="工具调用")
    function_call = Column(JSON, comment="函数调用")
    parent_message_id = Column(GUID(), comment="父消息ID")
    
    # 关系
    session = relationship("AgentSessionEntity", back_populates="messages")
    
    __table_args__ = (
        Index('ix_agent_messages_session_created', 'session_id', 'created_at'),
        Index('ix_agent_messages_role', 'role'),
    )
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'message_id': str(self.id) if self.id else None,
            'session_id': str(self.session_id) if self.session_id else None,
            'role': self.role,
            'content': self.content,
            'content_type': self.content_type,
            'tokens_used': self.tokens_used,
            'latency_ms': self.latency_ms,
            'metadata': self.message_metadata or {},
            'tool_calls': self.tool_calls,
            'function_call': self.function_call,
            'parent_message_id': str(self.parent_message_id) if self.parent_message_id else None,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class AgentMemoryEntity(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """智能体长期记忆实体
    
    存储智能体的长期记忆，用于跨会话的知识保持。
    
    Attributes:
        id: 记忆唯一标识 (UUID)
        agent_id: 智能体ID
        user_id: 用户ID
        memory_type: 记忆类型 (fact, preference, event, skill)
        content: 记忆内容
        embedding: 向量嵌入 (JSON数组)
        importance: 重要性分数 (0-1)
        access_count: 访问次数
        last_accessed_at: 最后访问时间
        source_session_id: 来源会话ID
        metadata: 元数据 (JSON)
        expires_at: 过期时间
        is_active: 是否激活
    """
    __tablename__ = 'agent_memories'
    
    agent_id = Column(GUID(), ForeignKey('agents.id', ondelete='CASCADE'), nullable=False, index=True, comment="智能体ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    memory_type = Column(String(30), default='fact', index=True, comment="记忆类型")
    content = Column(Text, nullable=False, comment="记忆内容")
    embedding = Column(JSON, comment="向量嵌入")
    importance = Column(Float, default=0.5, index=True, comment="重要性分数")
    access_count = Column(Integer, default=0, comment="访问次数")
    last_accessed_at = Column(DateTime, comment="最后访问时间")
    source_session_id = Column(GUID(), comment="来源会话ID")
    memory_metadata = Column(JSON, default=dict, comment="元数据")
    expires_at = Column(DateTime, comment="过期时间")
    is_active = Column(Boolean, default=True, index=True, comment="是否激活")
    
    # 关系
    agent = relationship("AgentEntity", back_populates="memories")
    
    __table_args__ = (
        Index('ix_agent_memories_agent_user', 'agent_id', 'user_id'),
        Index('ix_agent_memories_type_importance', 'memory_type', 'importance'),
        Index('ix_agent_memories_active_expires', 'is_active', 'expires_at'),
    )
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'memory_id': str(self.id) if self.id else None,
            'agent_id': str(self.agent_id) if self.agent_id else None,
            'user_id': self.user_id,
            'memory_type': self.memory_type,
            'content': self.content,
            'embedding': self.embedding,
            'importance': self.importance,
            'access_count': self.access_count,
            'last_accessed_at': self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            'source_session_id': str(self.source_session_id) if self.source_session_id else None,
            'metadata': self.memory_metadata or {},
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'is_active': self.is_active,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class AgentExecutionEntity(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """智能体执行记录实体
    
    存储智能体的执行历史和性能数据。
    
    Attributes:
        id: 执行唯一标识 (UUID)
        agent_id: 智能体ID
        session_id: 会话ID
        user_id: 用户ID
        execution_type: 执行类型 (chat, tool_call, reasoning)
        input_data: 输入数据 (JSON)
        output_data: 输出数据 (JSON)
        status: 状态 (pending, running, completed, failed)
        error_message: 错误信息
        started_at: 开始时间
        completed_at: 完成时间
        duration_ms: 执行时长(ms)
        tokens_input: 输入token数
        tokens_output: 输出token数
        cost: 执行成本
        graph_state: LangGraph状态 (JSON)
        checkpoints: 检查点列表 (JSON)
    """
    __tablename__ = 'agent_executions'
    
    agent_id = Column(GUID(), ForeignKey('agents.id', ondelete='CASCADE'), nullable=False, index=True, comment="智能体ID")
    session_id = Column(GUID(), index=True, comment="会话ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    execution_type = Column(String(30), default='chat', index=True, comment="执行类型")
    input_data = Column(JSON, comment="输入数据")
    output_data = Column(JSON, comment="输出数据")
    status = Column(String(20), default='pending', index=True, comment="状态")
    error_message = Column(Text, comment="错误信息")
    started_at = Column(DateTime, comment="开始时间")
    completed_at = Column(DateTime, comment="完成时间")
    duration_ms = Column(Integer, default=0, comment="执行时长(ms)")
    tokens_input = Column(Integer, default=0, comment="输入token数")
    tokens_output = Column(Integer, default=0, comment="输出token数")
    cost = Column(Float, default=0.0, comment="执行成本")
    graph_state = Column(JSON, comment="LangGraph状态")
    checkpoints = Column(JSON, comment="检查点列表")
    
    __table_args__ = (
        Index('ix_agent_executions_agent_status', 'agent_id', 'status'),
        Index('ix_agent_executions_user_started', 'user_id', 'started_at'),
        Index('ix_agent_executions_session', 'session_id'),
    )
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'execution_id': str(self.id) if self.id else None,
            'agent_id': str(self.agent_id) if self.agent_id else None,
            'session_id': str(self.session_id) if self.session_id else None,
            'user_id': self.user_id,
            'execution_type': self.execution_type,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'status': self.status,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration_ms': self.duration_ms,
            'tokens_input': self.tokens_input,
            'tokens_output': self.tokens_output,
            'cost': self.cost,
            'graph_state': self.graph_state,
            'checkpoints': self.checkpoints,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class AgentToolEntity(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """智能体工具实体
    
    存储智能体可用的工具定义。
    
    Attributes:
        id: 工具唯一标识 (UUID)
        agent_id: 智能体ID (为NULL表示全局工具)
        name: 工具名称
        description: 工具描述
        tool_type: 工具类型 (builtin, custom, api)
        schema: 工具schema (JSON)
        config: 工具配置 (JSON)
        is_enabled: 是否启用
        execution_count: 执行次数
        avg_latency_ms: 平均延迟(ms)
    """
    __tablename__ = 'agent_tools'
    
    agent_id = Column(GUID(), ForeignKey('agents.id', ondelete='CASCADE'), index=True, comment="智能体ID")
    name = Column(String(100), nullable=False, comment="工具名称")
    description = Column(Text, comment="工具描述")
    tool_type = Column(String(30), default='builtin', index=True, comment="工具类型")
    schema = Column(JSON, comment="工具schema")
    config = Column(JSON, default=dict, comment="工具配置")
    is_enabled = Column(Boolean, default=True, index=True, comment="是否启用")
    execution_count = Column(Integer, default=0, comment="执行次数")
    avg_latency_ms = Column(Float, default=0.0, comment="平均延迟(ms)")
    
    __table_args__ = (
        Index('ix_agent_tools_agent_enabled', 'agent_id', 'is_enabled'),
        Index('ix_agent_tools_type', 'tool_type'),
    )
    
    def to_dict(self):
        """转换为字典"""
        return {
            'id': str(self.id) if self.id else None,
            'tool_id': str(self.id) if self.id else None,
            'agent_id': str(self.agent_id) if self.agent_id else None,
            'name': self.name,
            'description': self.description,
            'tool_type': self.tool_type,
            'schema': self.schema,
            'config': self.config or {},
            'is_enabled': self.is_enabled,
            'execution_count': self.execution_count,
            'avg_latency_ms': self.avg_latency_ms,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

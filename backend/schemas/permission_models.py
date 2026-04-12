"""权限相关数据模型

定义权限、角色及相关关联表的数据模型，包括：
- 权限模型: Permission
- 角色模型: Role
- 用户角色关联: UserRole
- 权限审计日志: PermissionAuditLog
- Agent权限分析: PermissionAgentMemory, PermissionAgentReasoning
- 权限策略: PermissionPolicy
- 资源权限: ResourcePermission
"""

from sqlalchemy import Column, String, Text, Boolean, ForeignKey, Table, Integer, Float, DateTime, JSON, Index
from sqlalchemy.orm import relationship
from datetime import datetime
from backend.schemas.base_models import Base, UUIDMixin, TimestampMixin, GUID

# 角色权限关联表
role_permissions = Table(
    'role_permissions',
    Base.metadata,
    Column('role_id', GUID(), ForeignKey('roles.id'), primary_key=True),
    Column('permission_id', GUID(), ForeignKey('permissions.id'), primary_key=True),
    Column('granted_by', GUID(), comment="授权人ID"),
    Column('granted_at', DateTime, default=datetime.utcnow, comment="授权时间"),
    extend_existing=True
)


class Permission(Base, UUIDMixin, TimestampMixin):
    """权限模型
    
    定义系统权限，每个权限对应特定资源上的特定操作。
    
    Attributes:
        name: 权限名称，唯一标识
        description: 权限描述
        resource: 资源类型（如 users, roles, datasets）
        action: 操作类型（如 create, read, update, delete）
        is_system: 是否为系统内置权限
        is_active: 是否启用
        scope: 权限范围（global, tenant, personal）
        conditions: 权限条件（JSON格式的条件表达式）
        priority: 优先级（数值越大优先级越高）
        risk_level: 风险等级（low, medium, high, critical）
        requires_mfa: 是否需要MFA验证
        audit_level: 审计级别（none, basic, detailed）
    """
    __tablename__ = 'permissions'
    __table_args__ = (
        Index('ix_permissions_resource_action', 'resource', 'action'),
        Index('ix_permissions_risk_level', 'risk_level'),
        {'extend_existing': True}
    )
    
    name = Column(String(100), nullable=False, unique=True, comment="权限名称")
    description = Column(Text, comment="权限描述")
    resource = Column(String(100), nullable=False, index=True, comment="资源类型")
    action = Column(String(50), nullable=False, index=True, comment="操作类型")
    is_system = Column(Boolean, default=False, comment="是否系统内置")
    is_active = Column(Boolean, default=True, comment="是否启用")
    scope = Column(String(20), default='global', comment="权限范围")
    conditions = Column(JSON, comment="权限条件")
    priority = Column(Integer, default=0, comment="优先级")
    risk_level = Column(String(20), default='low', comment="风险等级")
    requires_mfa = Column(Boolean, default=False, comment="是否需要MFA")
    audit_level = Column(String(20), default='basic', comment="审计级别")
    
    # 关联关系
    roles = relationship(
        "Role",
        secondary=role_permissions,
        back_populates="permissions"
    )
    
    def __repr__(self):
        return f"<Permission(id='{self.id}', name='{self.name}', resource='{self.resource}', action='{self.action}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "resource": self.resource,
            "action": self.action,
            "is_system": self.is_system,
            "is_active": self.is_active,
            "scope": self.scope,
            "conditions": self.conditions,
            "priority": self.priority,
            "risk_level": self.risk_level,
            "requires_mfa": self.requires_mfa,
            "audit_level": self.audit_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class Role(Base, UUIDMixin, TimestampMixin):
    """角色模型
    
    定义系统角色，角色是权限的集合。
    
    Attributes:
        name: 角色名称，唯一标识
        display_name: 显示名称
        description: 角色描述
        is_system: 是否为系统内置角色
        is_active: 是否启用
        parent_role_id: 父角色ID（用于角色继承）
        level: 角色级别（数值越大权限越高）
        max_users: 最大用户数限制
        metadata: 元数据
    """
    __tablename__ = 'roles'
    __table_args__ = (
        Index('ix_roles_level', 'level'),
        {'extend_existing': True}
    )
    
    name = Column(String(100), nullable=False, unique=True, comment="角色名称")
    display_name = Column(String(100), comment="显示名称")
    description = Column(Text, comment="角色描述")
    is_system = Column(Boolean, default=False, comment="是否系统内置")
    is_active = Column(Boolean, default=True, comment="是否启用")
    parent_role_id = Column(GUID(), ForeignKey('roles.id'), comment="父角色ID")
    level = Column(Integer, default=0, comment="角色级别")
    max_users = Column(Integer, comment="最大用户数")
    extra_data = Column(JSON, comment="元数据")
    
    # 关联关系
    permissions = relationship(
        "Permission",
        secondary=role_permissions,
        back_populates="roles"
    )
    users = relationship("UserRole", back_populates="role")
    children = relationship("Role", backref="parent", remote_side="Role.id")
    
    def __repr__(self):
        return f"<Role(id='{self.id}', name='{self.name}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "is_system": self.is_system,
            "is_active": self.is_active,
            "parent_role_id": str(self.parent_role_id) if self.parent_role_id else None,
            "level": self.level,
            "max_users": self.max_users,
            "extra_data": self.extra_data,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class UserRole(Base, UUIDMixin, TimestampMixin):
    """用户角色关联模型
    
    管理用户与角色的关联关系。
    
    Attributes:
        user_id: 用户ID
        role_id: 角色ID
        assigned_by: 分配者ID
        expires_at: 过期时间
        is_active: 是否有效
        conditions: 关联条件
        scope: 权限范围
    """
    __tablename__ = 'user_roles'
    __table_args__ = (
        Index('ix_user_roles_user_role', 'user_id', 'role_id', unique=True),
        Index('ix_user_roles_expires', 'expires_at'),
        {'extend_existing': True}
    )
    
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False, index=True, comment="用户ID")
    role_id = Column(GUID(), ForeignKey('roles.id'), nullable=False, index=True, comment="角色ID")
    assigned_by = Column(GUID(), comment="分配者ID")
    assigned_at = Column(DateTime, default=datetime.utcnow, comment="分配时间")
    expires_at = Column(DateTime, comment="过期时间")
    is_active = Column(Boolean, default=True, comment="是否有效")
    conditions = Column(JSON, comment="关联条件")
    scope = Column(String(20), default='global', comment="权限范围")
    
    # 关联关系
    user = relationship("User", back_populates="roles")
    role = relationship("Role", back_populates="users")
    
    def __repr__(self):
        return f"<UserRole(user_id='{self.user_id}', role_id='{self.role_id}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "role_id": str(self.role_id),
            "assigned_by": str(self.assigned_by) if self.assigned_by else None,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "conditions": self.conditions,
            "scope": self.scope
        }
    
    def is_expired(self) -> bool:
        """检查是否已过期"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at


# ============================================================================
# 权限审计模型
# ============================================================================

class PermissionAuditLog(Base, UUIDMixin, TimestampMixin):
    """权限审计日志模型
    
    记录所有权限相关操作，用于审计和合规。
    
    Attributes:
        user_id: 操作用户ID
        target_user_id: 目标用户ID
        action: 操作类型
        resource_type: 资源类型
        resource_id: 资源ID
        permission_id: 权限ID
        role_id: 角色ID
        old_value: 变更前的值
        new_value: 变更后的值
        ip_address: IP地址
        user_agent: 用户代理
        status: 操作状态
        error_message: 错误信息
        agent_analysis: Agent分析结果
    """
    __tablename__ = 'permission_audit_logs'
    __table_args__ = (
        Index('ix_perm_audit_user_time', 'user_id', 'created_at'),
        Index('ix_perm_audit_target_time', 'target_user_id', 'created_at'),
        Index('ix_perm_audit_action_time', 'action', 'created_at'),
    )
    
    user_id = Column(GUID(), ForeignKey('users.id'), index=True, comment="操作用户ID")
    target_user_id = Column(GUID(), index=True, comment="目标用户ID")
    action = Column(String(50), nullable=False, index=True, comment="操作类型")
    resource_type = Column(String(50), comment="资源类型")
    resource_id = Column(String(36), comment="资源ID")
    permission_id = Column(GUID(), ForeignKey('permissions.id'), comment="权限ID")
    role_id = Column(GUID(), ForeignKey('roles.id'), comment="角色ID")
    old_value = Column(JSON, comment="变更前的值")
    new_value = Column(JSON, comment="变更后的值")
    ip_address = Column(String(45), comment="IP地址")
    user_agent = Column(Text, comment="用户代理")
    status = Column(String(20), default='success', comment="操作状态")
    error_message = Column(Text, comment="错误信息")
    agent_analysis = Column(JSON, comment="Agent分析结果")
    
    def __repr__(self):
        return f"<PermissionAuditLog(id='{self.id}', action='{self.action}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "target_user_id": str(self.target_user_id) if self.target_user_id else None,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "permission_id": str(self.permission_id) if self.permission_id else None,
            "role_id": str(self.role_id) if self.role_id else None,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "ip_address": self.ip_address,
            "status": self.status,
            "error_message": self.error_message,
            "agent_analysis": self.agent_analysis,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class PermissionAccessLog(Base, UUIDMixin, TimestampMixin):
    """权限访问日志模型
    
    记录权限验证请求，用于分析和优化。
    
    Attributes:
        user_id: 用户ID
        resource: 请求的资源
        action: 请求的操作
        result: 验证结果
        reason: 原因说明
        ip_address: IP地址
        user_agent: 用户代理
        latency_ms: 延迟毫秒
        cached: 是否使用缓存
    """
    __tablename__ = 'permission_access_logs'
    __table_args__ = (
        Index('ix_perm_access_user_time', 'user_id', 'created_at'),
        Index('ix_perm_access_resource_action', 'resource', 'action'),
        Index('ix_perm_access_result', 'result'),
    )
    
    user_id = Column(GUID(), ForeignKey('users.id'), index=True, comment="用户ID")
    resource = Column(String(100), nullable=False, comment="请求的资源")
    action = Column(String(50), nullable=False, comment="请求的操作")
    result = Column(Boolean, nullable=False, comment="验证结果")
    reason = Column(String(255), comment="原因说明")
    ip_address = Column(String(45), comment="IP地址")
    user_agent = Column(Text, comment="用户代理")
    latency_ms = Column(Float, comment="延迟毫秒")
    cached = Column(Boolean, default=False, comment="是否使用缓存")
    
    def __repr__(self):
        return f"<PermissionAccessLog(user_id='{self.user_id}', resource='{self.resource}', result={self.result})>"


# ============================================================================
# Agent 权限分析模型
# ============================================================================

class PermissionAgentMemory(Base, UUIDMixin, TimestampMixin):
    """权限Agent记忆模型
    
    存储权限分析相关的Agent长期记忆。
    
    Attributes:
        user_id: 关联的用户ID（可选）
        memory_type: 记忆类型
        content: 记忆内容
        embedding: 向量嵌入
        importance: 重要性分数
        access_count: 访问次数
        last_accessed: 最后访问时间
        expires_at: 过期时间
        metadata: 元数据
        source: 记忆来源
        is_active: 是否有效
    """
    __tablename__ = 'permission_agent_memories'
    __table_args__ = (
        Index('ix_perm_mem_user_type', 'user_id', 'memory_type'),
        Index('ix_perm_mem_importance', 'importance'),
        Index('ix_perm_mem_active_type', 'is_active', 'memory_type'),
    )
    
    user_id = Column(GUID(), ForeignKey('users.id'), index=True, comment="用户ID")
    memory_type = Column(String(50), nullable=False, index=True, comment="记忆类型")
    content = Column(Text, nullable=False, comment="记忆内容")
    embedding = Column(JSON, comment="向量嵌入")
    importance = Column(Float, default=0.5, comment="重要性分数")
    access_count = Column(Integer, default=0, comment="访问次数")
    last_accessed = Column(DateTime, comment="最后访问时间")
    expires_at = Column(DateTime, comment="过期时间")
    extra_data = Column(JSON, comment="元数据")
    source = Column(String(100), comment="记忆来源")
    is_active = Column(Boolean, default=True, comment="是否有效")
    
    def __repr__(self):
        return f"<PermissionAgentMemory(id='{self.id}', type='{self.memory_type}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "memory_type": self.memory_type,
            "content": self.content,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.extra_data,
            "source": self.source,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class PermissionAgentReasoning(Base, UUIDMixin, TimestampMixin):
    """权限Agent推理记录模型
    
    记录Agent的权限分析推理过程和决策。
    
    Attributes:
        user_id: 关联的用户ID
        trigger: 触发推理的事件
        context: 推理上下文
        reasoning_steps: 推理步骤
        conclusion: 结论
        confidence: 置信度
        action_suggested: 建议的行动
        action_taken: 实际采取的行动
        outcome: 结果
        feedback: 反馈
        model_used: 使用的模型
        tokens_used: 使用的token数
        latency_ms: 延迟毫秒
    """
    __tablename__ = 'permission_agent_reasoning'
    __table_args__ = (
        Index('ix_perm_reason_user_time', 'user_id', 'created_at'),
        Index('ix_perm_reason_trigger', 'trigger'),
    )
    
    user_id = Column(GUID(), ForeignKey('users.id'), index=True, comment="用户ID")
    trigger = Column(String(100), nullable=False, comment="触发事件")
    context = Column(JSON, comment="推理上下文")
    reasoning_steps = Column(JSON, comment="推理步骤")
    conclusion = Column(Text, comment="结论")
    confidence = Column(Float, default=0.0, comment="置信度")
    action_suggested = Column(String(100), comment="建议的行动")
    action_taken = Column(String(100), comment="实际采取的行动")
    outcome = Column(String(50), comment="结果")
    feedback = Column(JSON, comment="反馈")
    model_used = Column(String(100), comment="使用的模型")
    tokens_used = Column(Integer, default=0, comment="使用的token数")
    latency_ms = Column(Float, default=0, comment="延迟毫秒")
    
    def __repr__(self):
        return f"<PermissionAgentReasoning(id='{self.id}', trigger='{self.trigger}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "trigger": self.trigger,
            "reasoning_steps": self.reasoning_steps,
            "conclusion": self.conclusion,
            "confidence": self.confidence,
            "action_suggested": self.action_suggested,
            "action_taken": self.action_taken,
            "outcome": self.outcome,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# ============================================================================
# 权限策略模型
# ============================================================================

class PermissionPolicy(Base, UUIDMixin, TimestampMixin):
    """权限策略模型
    
    定义细粒度的权限策略规则。
    
    Attributes:
        name: 策略名称
        description: 策略描述
        policy_type: 策略类型 (allow, deny, conditional)
        resource_pattern: 资源模式（支持通配符）
        action_pattern: 操作模式（支持通配符）
        conditions: 策略条件
        priority: 优先级
        is_active: 是否启用
        effect: 效果 (allow, deny)
        scope: 作用范围
        applies_to: 适用对象类型 (user, role, group)
        target_ids: 目标ID列表
    """
    __tablename__ = 'permission_policies'
    __table_args__ = (
        Index('ix_perm_policy_type_active', 'policy_type', 'is_active'),
        Index('ix_perm_policy_priority', 'priority'),
    )
    
    name = Column(String(100), nullable=False, unique=True, comment="策略名称")
    description = Column(Text, comment="策略描述")
    policy_type = Column(String(20), nullable=False, default='allow', comment="策略类型")
    resource_pattern = Column(String(200), nullable=False, comment="资源模式")
    action_pattern = Column(String(100), nullable=False, comment="操作模式")
    conditions = Column(JSON, comment="策略条件")
    priority = Column(Integer, default=0, comment="优先级")
    is_active = Column(Boolean, default=True, comment="是否启用")
    effect = Column(String(10), default='allow', comment="效果")
    scope = Column(String(20), default='global', comment="作用范围")
    applies_to = Column(String(20), default='user', comment="适用对象类型")
    target_ids = Column(JSON, comment="目标ID列表")
    
    def __repr__(self):
        return f"<PermissionPolicy(id='{self.id}', name='{self.name}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "policy_type": self.policy_type,
            "resource_pattern": self.resource_pattern,
            "action_pattern": self.action_pattern,
            "conditions": self.conditions,
            "priority": self.priority,
            "is_active": self.is_active,
            "effect": self.effect,
            "scope": self.scope,
            "applies_to": self.applies_to,
            "target_ids": self.target_ids,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class ResourcePermission(Base, UUIDMixin, TimestampMixin):
    """资源权限模型
    
    为特定资源实例定义权限。
    
    Attributes:
        resource_type: 资源类型
        resource_id: 资源ID
        user_id: 用户ID
        permission_type: 权限类型 (owner, editor, viewer, custom)
        permissions: 具体权限列表
        granted_by: 授权人ID
        expires_at: 过期时间
        is_active: 是否有效
        conditions: 权限条件
    """
    __tablename__ = 'resource_permissions'
    __table_args__ = (
        Index('ix_res_perm_resource', 'resource_type', 'resource_id'),
        Index('ix_res_perm_user', 'user_id'),
        Index('ix_res_perm_type', 'permission_type'),
    )
    
    resource_type = Column(String(50), nullable=False, comment="资源类型")
    resource_id = Column(String(36), nullable=False, comment="资源ID")
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False, index=True, comment="用户ID")
    permission_type = Column(String(20), nullable=False, default='viewer', comment="权限类型")
    permissions = Column(JSON, comment="具体权限列表")
    granted_by = Column(GUID(), comment="授权人ID")
    expires_at = Column(DateTime, comment="过期时间")
    is_active = Column(Boolean, default=True, comment="是否有效")
    conditions = Column(JSON, comment="权限条件")
    
    def __repr__(self):
        return f"<ResourcePermission(resource='{self.resource_type}:{self.resource_id}', user_id='{self.user_id}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "user_id": str(self.user_id),
            "permission_type": self.permission_type,
            "permissions": self.permissions,
            "granted_by": str(self.granted_by) if self.granted_by else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "conditions": self.conditions,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class PermissionRecommendation(Base, UUIDMixin, TimestampMixin):
    """权限建议模型
    
    存储Agent生成的权限配置建议。
    
    Attributes:
        user_id: 目标用户ID
        recommendation_type: 建议类型 (grant, revoke, modify)
        permission_id: 相关权限ID
        role_id: 相关角色ID
        reason: 建议原因
        confidence: 置信度
        risk_assessment: 风险评估
        status: 状态 (pending, accepted, rejected, expired)
        reviewed_by: 审核人ID
        reviewed_at: 审核时间
        review_notes: 审核备注
    """
    __tablename__ = 'permission_recommendations'
    __table_args__ = (
        Index('ix_perm_rec_user_status', 'user_id', 'status'),
        Index('ix_perm_rec_type', 'recommendation_type'),
    )
    
    user_id = Column(GUID(), ForeignKey('users.id'), nullable=False, index=True, comment="目标用户ID")
    recommendation_type = Column(String(20), nullable=False, comment="建议类型")
    permission_id = Column(GUID(), ForeignKey('permissions.id'), comment="相关权限ID")
    role_id = Column(GUID(), ForeignKey('roles.id'), comment="相关角色ID")
    reason = Column(Text, nullable=False, comment="建议原因")
    confidence = Column(Float, default=0.0, comment="置信度")
    risk_assessment = Column(JSON, comment="风险评估")
    status = Column(String(20), default='pending', comment="状态")
    reviewed_by = Column(GUID(), comment="审核人ID")
    reviewed_at = Column(DateTime, comment="审核时间")
    review_notes = Column(Text, comment="审核备注")
    
    def __repr__(self):
        return f"<PermissionRecommendation(id='{self.id}', type='{self.recommendation_type}', status='{self.status}')>"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "recommendation_type": self.recommendation_type,
            "permission_id": str(self.permission_id) if self.permission_id else None,
            "role_id": str(self.role_id) if self.role_id else None,
            "reason": self.reason,
            "confidence": self.confidence,
            "risk_assessment": self.risk_assessment,
            "status": self.status,
            "reviewed_by": str(self.reviewed_by) if self.reviewed_by else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "review_notes": self.review_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

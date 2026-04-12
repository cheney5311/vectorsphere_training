"""数据库模型基类和混入类

定义所有数据库模型的基类和通用混入类。
支持 SQLite、PostgreSQL、MySQL 多数据库兼容。
"""

from sqlalchemy import Column, String, Text, Boolean, Integer, Float, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import uuid


# ============================================================================
# 跨数据库兼容的 UUID 类型
# ============================================================================

class GUID(TypeDecorator):
    """跨数据库兼容的 UUID 类型
    
    在 PostgreSQL 使用原生 UUID 类型
    在 SQLite/MySQL 使用 CHAR(36) 存储
    """
    impl = CHAR
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            if isinstance(value, uuid.UUID):
                return str(value)
            else:
                return str(uuid.UUID(value))
    
    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(value)


# 创建基类
Base = declarative_base()


class UUIDMixin:
    """UUID混入类 - 跨数据库兼容"""
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)

class TimestampMixin:
    """时间戳混入类"""
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)

class TenantMixin:
    """租户混入类"""
    tenant_id = Column(String(36), nullable=False, index=True)

class Tenant(Base, UUIDMixin, TimestampMixin):
    """租户模型"""
    __tablename__ = 'tenants'
    
    name = Column(String(100), unique=True, nullable=False, index=True, comment="租户名称")
    display_name = Column(String(200), comment="显示名称")
    description = Column(Text, comment="描述")
    status = Column(String(20), default='pending', index=True, comment="状态")
    settings = Column(Text, comment="配置信息")
    creator_user_id = Column(String(36), nullable=False, index=True, comment="创建者用户ID")
    
    def __repr__(self):
        return f"<Tenant(id='{self.id}', name='{self.name}', status='{self.status}')>"

class TenantUser(Base, UUIDMixin, TimestampMixin):
    """租户用户关联模型"""
    __tablename__ = 'tenant_users'
    
    tenant_id = Column(String(36), nullable=False, index=True, comment="租户ID")
    user_id = Column(String(36), nullable=False, index=True, comment="用户ID")
    role = Column(String(50), default='member', comment="角色")
    is_active = Column(Boolean, default=True, comment="是否激活")
    
    def __repr__(self):
        return f"<TenantUser(tenant_id='{self.tenant_id}', user_id='{self.user_id}', role='{self.role}')>"


class TenantApiKey(Base, UUIDMixin, TimestampMixin):
    """租户API密钥模型"""
    __tablename__ = 'tenant_api_keys'
    
    tenant_id = Column(String(36), nullable=False, index=True, comment="租户ID")
    name = Column(String(100), nullable=False, comment="密钥名称")
    description = Column(Text, comment="密钥描述")
    key_hash = Column(String(255), nullable=False, comment="密钥哈希")
    key_prefix = Column(String(16), nullable=False, comment="密钥前缀（用于识别）")
    permissions = Column(Text, comment="权限列表（JSON格式）")
    scopes = Column(Text, comment="作用域列表（JSON格式）")
    rate_limit = Column(Integer, default=1000, comment="每小时请求限制")
    created_by = Column(String(36), nullable=False, index=True, comment="创建者用户ID")
    is_active = Column(Boolean, default=True, index=True, comment="是否激活")
    expires_at = Column(DateTime, comment="过期时间")
    last_used_at = Column(DateTime, comment="最后使用时间")
    last_used_ip = Column(String(45), comment="最后使用IP")
    use_count = Column(Integer, default=0, comment="使用次数")
    revoked_at = Column(DateTime, comment="撤销时间")
    revoked_by = Column(String(36), comment="撤销者用户ID")
    revoke_reason = Column(Text, comment="撤销原因")
    
    def __repr__(self):
        return f"<TenantApiKey(id='{self.id}', tenant_id='{self.tenant_id}', name='{self.name}')>"


class TenantInvite(Base, UUIDMixin, TimestampMixin):
    """租户邀请模型"""
    __tablename__ = 'tenant_invites'
    
    tenant_id = Column(String(36), nullable=False, index=True, comment="租户ID")
    email = Column(String(255), nullable=False, index=True, comment="被邀请人邮箱")
    role = Column(String(50), default='member', comment="分配角色")
    invite_code = Column(String(64), unique=True, nullable=False, index=True, comment="邀请码")
    invited_by = Column(String(36), nullable=False, comment="邀请人用户ID")
    status = Column(String(20), default='pending', index=True, comment="状态")
    expires_at = Column(DateTime, nullable=False, comment="过期时间")
    accepted_at = Column(DateTime, comment="接受时间")
    accepted_by = Column(String(36), comment="接受者用户ID")
    message = Column(Text, comment="邀请消息")
    
    def __repr__(self):
        return f"<TenantInvite(id='{self.id}', tenant_id='{self.tenant_id}', email='{self.email}')>"


class TenantAuditLog(Base, UUIDMixin):
    """租户审计日志模型"""
    __tablename__ = 'tenant_audit_logs'
    
    tenant_id = Column(String(36), nullable=False, index=True, comment="租户ID")
    user_id = Column(String(36), nullable=False, index=True, comment="操作用户ID")
    action = Column(String(50), nullable=False, index=True, comment="操作类型")
    resource_type = Column(String(50), nullable=False, index=True, comment="资源类型")
    resource_id = Column(String(36), index=True, comment="资源ID")
    details = Column(Text, comment="详细信息（JSON格式）")
    ip_address = Column(String(45), comment="IP地址")
    user_agent = Column(Text, comment="用户代理")
    timestamp = Column(DateTime, default=func.now(), nullable=False, index=True, comment="时间戳")
    
    def __repr__(self):
        return f"<TenantAuditLog(id='{self.id}', tenant_id='{self.tenant_id}', action='{self.action}')>"


class TenantQuota(Base, UUIDMixin, TimestampMixin):
    """租户配额模型"""
    __tablename__ = 'tenant_quotas'
    
    tenant_id = Column(String(36), unique=True, nullable=False, index=True, comment="租户ID")
    plan = Column(String(50), default='basic', comment="套餐类型")
    max_users = Column(Integer, default=10, comment="最大用户数")
    max_training_sessions = Column(Integer, default=5, comment="最大训练会话数")
    max_concurrent_trainings = Column(Integer, default=2, comment="最大并发训练数")
    max_models = Column(Integer, default=20, comment="最大模型数")
    max_datasets = Column(Integer, default=50, comment="最大数据集数")
    storage_limit_gb = Column(Integer, default=100, comment="存储限制(GB)")
    compute_hours_monthly = Column(Integer, default=100, comment="每月计算时长")
    api_requests_daily = Column(Integer, default=10000, comment="每日API请求限制")
    gpu_hours_monthly = Column(Integer, default=50, comment="每月GPU时长")
    max_api_keys = Column(Integer, default=10, comment="最大API密钥数")
    
    def __repr__(self):
        return f"<TenantQuota(tenant_id='{self.tenant_id}', plan='{self.plan}')>"
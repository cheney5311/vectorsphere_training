"""Schemas 基础模型导出
为统一引用，转发 backend.schemas.base_models 中定义的 Base/UUIDMixin/TimestampMixin。
避免重复定义引发表结构冲突。
"""
from backend.schemas.base_models import Base, UUIDMixin, TimestampMixin, TenantMixin

__all__ = ["Base", "UUIDMixin", "TimestampMixin", "TenantMixin"]
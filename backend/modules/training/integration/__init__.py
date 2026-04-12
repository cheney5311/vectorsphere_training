"""训练集成模块

提供与外部平台的集成功能。
"""

from .tenant_platform_integration import (
    report_training_progress_to_tenant_platform,
    report_training_status_to_tenant_platform
)

__all__ = [
    "report_training_progress_to_tenant_platform",
    "report_training_status_to_tenant_platform"
]
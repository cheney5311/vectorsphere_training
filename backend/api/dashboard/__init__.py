"""仪表盘模块API接口

提供仪表盘相关的API接口。
"""

from .dashboard_api import dashboard_bp
from .dashboard_statistics_api import dashboard_statistics_bp
from .user_training_api import user_training_bp

__all__ = [
    'dashboard_bp',
    'dashboard_statistics_bp',
    'user_training_bp'
]
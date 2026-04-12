"""监控模块API

提供系统性能监控相关的API接口。
"""

# 直接从api.py文件导入符号（避免循环导入）
import sys
import os

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# 直接导入api.py模块
api_module_path = os.path.join(project_root, 'backend', 'modules', 'monitoring', 'api.py')
spec = __import__('importlib.util').util.spec_from_file_location('monitoring_api', api_module_path)
monitoring_api = __import__('importlib.util').util.module_from_spec(spec)
spec.loader.exec_module(monitoring_api)

# 获取符号
monitoring_bp = monitoring_api.monitoring_bp
init_monitoring_api = monitoring_api.init_monitoring_api
cleanup_monitoring_api = monitoring_api.cleanup_monitoring_api

from .performance_api import performance_bp
from .realtime_performance_api import realtime_performance_bp

__all__ = [
    'performance_bp',
    'realtime_performance_bp',
    'monitoring_bp',
    'init_monitoring_api',
    'cleanup_monitoring_api'
]
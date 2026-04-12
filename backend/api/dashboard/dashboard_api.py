"""仪表盘API接口

提供仪表盘相关的API接口，聚合训练、模型、系统资源等多维度数据。

API 端点:
    GET /api/v1/dashboard/overview       - 获取仪表盘概览
    GET /api/v1/dashboard/training-stats - 获取详细训练统计
    GET /api/v1/dashboard/system-metrics - 获取系统指标历史
    GET /api/v1/dashboard/model-stats    - 获取模型统计
    GET /api/v1/dashboard/dataset-stats  - 获取数据集统计
    GET /api/v1/dashboard/user-activity  - 获取用户活动
    GET /api/v1/dashboard/gpu-status     - 获取GPU状态
    GET /api/v1/dashboard/alerts         - 获取告警概要
"""

import sys
import os
import logging
from flask import Blueprint, request, jsonify, g
from flask_jwt_extended import jwt_required, get_jwt_identity
from functools import wraps

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.exceptions import ValidationError
from backend.utils.response import success_response, error_response
from backend.services.dashboard_service import get_dashboard_service, DashboardService
from backend.schemas.dashboard_models import (
    DashboardTimeRange,
    MetricGranularity
)

logger = logging.getLogger(__name__)

# 创建蓝图
dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/api/v1/dashboard')


def get_tenant_id() -> str:
    """获取当前租户ID
    
    从请求头或上下文中获取租户ID。
    
    Returns:
        str: 租户ID，如果未找到则返回 None
    """
    return request.headers.get('X-Tenant-ID') or getattr(g, 'tenant_id', None)


def handle_dashboard_errors(f):
    """仪表盘API错误处理装饰器
    
    统一处理API错误，返回标准化的错误响应。
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValidationError as e:
            logger.warning(f"Validation error in {f.__name__}: {e}")
            return error_response(str(e), 400)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}", exc_info=True)
            return error_response(f"服务器内部错误: {str(e)}", 500)
    return wrapper


@dashboard_bp.route('/overview', methods=['GET'])
@jwt_required()
@handle_dashboard_errors
def get_dashboard_overview():
    """获取仪表盘概览数据
    
    聚合训练、模型、系统资源等多维度数据的概览视图。
    
    Query Parameters:
        time_range (str, optional): 时间范围，可选值:
            - last_hour: 最近1小时
            - last_24_hours: 最近24小时 (默认)
            - last_7_days: 最近7天
            - last_30_days: 最近30天
            - this_month: 本月
            - this_year: 今年
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选，用于多租户)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取仪表盘概览成功",
                "data": {
                    "overview": {
                        "training": {
                            "active_count": 3,
                            "completed_count": 45,
                            "failed_count": 2,
                            "pending_count": 1,
                            "paused_count": 0,
                            "success_rate": 0.9574,
                            "total_training_time_hours": 156.5,
                            "avg_training_time_hours": 3.26
                        },
                        "models": {
                            "total_count": 15,
                            "deployed_count": 5,
                            "draft_count": 8,
                            "archived_count": 2,
                            "avg_accuracy": 0.8745,
                            "best_accuracy": 0.9523,
                            "avg_f1_score": 0.8612,
                            "total_size_gb": 12.5
                        },
                        "system": {
                            "timestamp": "2026-01-16T12:00:00",
                            "cpu_usage": 45.5,
                            "cpu_count": 8,
                            "memory_usage": 62.3,
                            "memory_used_gb": 12.5,
                            "memory_total_gb": 32.0,
                            "disk_usage": 55.2,
                            "disk_used_gb": 256.5,
                            "disk_total_gb": 512.0,
                            "network_sent_mb": 1024.5,
                            "network_recv_mb": 2048.3
                        },
                        "gpu": [
                            {
                                "device_id": 0,
                                "name": "NVIDIA RTX 3090",
                                "utilization": 78.5,
                                "memory_used_gb": 18.2,
                                "memory_total_gb": 24.0,
                                "memory_usage": 75.8,
                                "temperature": 65.0,
                                "power_draw": 280.5,
                                "power_limit": 350.0
                            }
                        ],
                        "user_activity": {
                            "total_training_count": 48,
                            "total_models_created": 15,
                            "total_datasets_used": 8,
                            "last_active_at": "2026-01-16T11:30:00",
                            "most_used_model_type": "classification",
                            "avg_training_time_hours": 3.26
                        },
                        "alerts_count": 2,
                        "last_updated": "2026-01-16T12:00:00"
                    }
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/overview?time_range=last_7_days" \\
             -H "Authorization: Bearer <token>"
    """
    # 获取当前用户ID
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    # 获取时间范围参数
    time_range_str = request.args.get('time_range', 'last_24_hours')
    
    # 解析时间范围
    time_range_map = {
        'last_hour': DashboardTimeRange.LAST_HOUR,
        'last_24_hours': DashboardTimeRange.LAST_24_HOURS,
        'last_7_days': DashboardTimeRange.LAST_7_DAYS,
        'last_30_days': DashboardTimeRange.LAST_30_DAYS,
        'this_month': DashboardTimeRange.THIS_MONTH,
        'this_year': DashboardTimeRange.THIS_YEAR
    }
    time_range = time_range_map.get(time_range_str, DashboardTimeRange.LAST_24_HOURS)
    
    # 获取仪表盘服务
    service = get_dashboard_service()
    
    # 获取概览数据
    overview = service.get_dashboard_overview(
        user_id=user_id,
        tenant_id=tenant_id,
        time_range=time_range
    )
    
    return success_response({
        'overview': overview.to_dict()
    }, "获取仪表盘概览成功")


@dashboard_bp.route('/training-stats', methods=['GET'])
@jwt_required()
@handle_dashboard_errors
def get_training_stats():
    """获取详细训练统计
    
    提供训练任务的详细统计信息，包括概览、时间趋势、分布等。
    
    Query Parameters:
        time_range (str, optional): 时间范围，默认 last_7_days
        granularity (str, optional): 数据粒度，可选值:
            - minute: 分钟级
            - hour: 小时级 (默认)
            - day: 天级
            - week: 周级
            - month: 月级
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取训练统计成功",
                "data": {
                    "stats": {
                        "overview": {
                            "active_count": 3,
                            "completed_count": 45,
                            "failed_count": 2,
                            "pending_count": 1,
                            "paused_count": 0,
                            "success_rate": 0.9574,
                            "total_training_time_hours": 156.5,
                            "avg_training_time_hours": 3.26
                        },
                        "trends": [
                            {
                                "timestamp": "2026-01-10T00:00:00",
                                "date_label": "2026-01-10",
                                "count": 8,
                                "success_count": 7,
                                "failed_count": 1,
                                "avg_duration_hours": 2.5
                            }
                        ],
                        "by_type": {
                            "classification": 25,
                            "regression": 15,
                            "nlp": 8
                        },
                        "by_status": {
                            "running": 3,
                            "completed": 45,
                            "failed": 2,
                            "pending": 1,
                            "paused": 0
                        },
                        "recent_sessions": [
                            {
                                "session_id": "sess_123",
                                "name": "BERT微调",
                                "status": "running",
                                "progress": 75.5,
                                "created_at": "2026-01-16T10:00:00"
                            }
                        ]
                    }
                }
            }
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/training-stats?time_range=last_7_days&granularity=day" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    # 解析参数
    time_range_str = request.args.get('time_range', 'last_7_days')
    granularity_str = request.args.get('granularity', 'day')
    
    time_range_map = {
        'last_hour': DashboardTimeRange.LAST_HOUR,
        'last_24_hours': DashboardTimeRange.LAST_24_HOURS,
        'last_7_days': DashboardTimeRange.LAST_7_DAYS,
        'last_30_days': DashboardTimeRange.LAST_30_DAYS,
        'this_month': DashboardTimeRange.THIS_MONTH,
        'this_year': DashboardTimeRange.THIS_YEAR
    }
    time_range = time_range_map.get(time_range_str, DashboardTimeRange.LAST_7_DAYS)
    
    granularity_map = {
        'minute': MetricGranularity.MINUTE,
        'hour': MetricGranularity.HOUR,
        'day': MetricGranularity.DAY,
        'week': MetricGranularity.WEEK,
        'month': MetricGranularity.MONTH
    }
    granularity = granularity_map.get(granularity_str, MetricGranularity.DAY)
    
    service = get_dashboard_service()
    
    stats = service.get_training_detailed_stats(
        user_id=user_id,
        tenant_id=tenant_id,
        time_range=time_range,
        granularity=granularity
    )
    
    return success_response({
        'stats': stats.to_dict()
    }, "获取训练统计成功")


@dashboard_bp.route('/system-metrics', methods=['GET'])
@jwt_required()
@handle_dashboard_errors
def get_system_metrics():
    """获取系统指标历史数据
    
    获取 CPU、内存、磁盘、网络等系统资源的历史使用情况。
    
    Query Parameters:
        hours (int, optional): 获取多少小时的数据，默认 24，最大 168（一周）
        granularity (str, optional): 数据粒度，默认 hour
    
    Request Headers:
        Authorization: Bearer <token> (必需)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取系统指标成功",
                "data": {
                    "metrics": {
                        "cpu_usage": [
                            {"timestamp": "2026-01-16T00:00:00", "value": 45.5},
                            {"timestamp": "2026-01-16T01:00:00", "value": 52.3}
                        ],
                        "memory_usage": [
                            {"timestamp": "2026-01-16T00:00:00", "value": 62.3},
                            {"timestamp": "2026-01-16T01:00:00", "value": 65.1}
                        ],
                        "disk_usage": [
                            {"timestamp": "2026-01-16T00:00:00", "value": 55.2}
                        ],
                        "network_io": [
                            {
                                "timestamp": "2026-01-16T00:00:00",
                                "sent_mb": 1024.5,
                                "recv_mb": 2048.3
                            }
                        ],
                        "gpu_usage": [
                            {
                                "timestamp": "2026-01-16T00:00:00",
                                "device_id": 0,
                                "utilization": 78.5,
                                "memory_usage": 75.8
                            }
                        ]
                    },
                    "current": {
                        "timestamp": "2026-01-16T12:00:00",
                        "cpu_usage": 45.5,
                        "cpu_count": 8,
                        "memory_usage": 62.3,
                        "memory_used_gb": 12.5,
                        "memory_total_gb": 32.0,
                        "disk_usage": 55.2,
                        "disk_used_gb": 256.5,
                        "disk_total_gb": 512.0,
                        "network_sent_mb": 1024.5,
                        "network_recv_mb": 2048.3
                    }
                }
            }
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/system-metrics?hours=48" \\
             -H "Authorization: Bearer <token>"
    """
    # 获取查询参数
    hours = min(int(request.args.get('hours', 24)), 168)  # 最大一周
    granularity_str = request.args.get('granularity', 'hour')
    
    granularity_map = {
        'minute': MetricGranularity.MINUTE,
        'hour': MetricGranularity.HOUR,
        'day': MetricGranularity.DAY
    }
    granularity = granularity_map.get(granularity_str, MetricGranularity.HOUR)
    
    service = get_dashboard_service()
    
    # 获取历史数据
    metrics = service.get_system_metrics_history(
        hours=hours,
        granularity=granularity
    )
    
    # 获取当前快照
    current = service.get_current_system_snapshot()
    
    return success_response({
        'metrics': metrics.to_dict(),
        'current': current.to_dict()
    }, "获取系统指标成功")


@dashboard_bp.route('/model-stats', methods=['GET'])
@jwt_required()
@handle_dashboard_errors
def get_model_stats():
    """获取模型统计数据
    
    提供模型的统计信息，包括概览、分布和表现最好的模型。
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取模型统计成功",
                "data": {
                    "stats": {
                        "overview": {
                            "total_count": 15,
                            "deployed_count": 5,
                            "draft_count": 8,
                            "archived_count": 2,
                            "avg_accuracy": 0.8745,
                            "best_accuracy": 0.9523,
                            "avg_f1_score": 0.8612,
                            "total_size_gb": 12.5
                        },
                        "distribution": {
                            "by_type": {"classification": 8, "regression": 5, "nlp": 2},
                            "by_framework": {"pytorch": 10, "tensorflow": 5},
                            "by_status": {"deployed": 5, "draft": 8, "archived": 2},
                            "by_category": {"image": 6, "text": 5, "tabular": 4}
                        },
                        "top_models": [
                            {
                                "id": "model_123",
                                "name": "BERT分类模型",
                                "model_type": "classification",
                                "framework": "pytorch",
                                "accuracy": 0.9523,
                                "f1_score": 0.9412,
                                "status": "deployed",
                                "created_at": "2026-01-15T10:00:00"
                            }
                        ]
                    }
                }
            }
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/model-stats" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    service = get_dashboard_service()
    
    stats = service.get_model_stats(
        user_id=user_id,
        tenant_id=tenant_id
    )
    
    return success_response({
        'stats': stats
    }, "获取模型统计成功")


@dashboard_bp.route('/dataset-stats', methods=['GET'])
@jwt_required()
@handle_dashboard_errors
def get_dataset_stats():
    """获取数据集统计
    
    提供数据集的统计信息。
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取数据集统计成功",
                "data": {
                    "stats": {
                        "total_count": 12,
                        "active_count": 8,
                        "archived_count": 3,
                        "processing_count": 1
                    }
                }
            }
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/dataset-stats" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    service = get_dashboard_service()
    
    stats = service.get_dataset_stats(
        user_id=user_id,
        tenant_id=tenant_id
    )
    
    return success_response({
        'stats': stats
    }, "获取数据集统计成功")


@dashboard_bp.route('/user-activity', methods=['GET'])
@jwt_required()
@handle_dashboard_errors
def get_user_activity():
    """获取用户活动概要
    
    获取当前用户的活动统计信息。
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取用户活动成功",
                "data": {
                    "activity": {
                        "total_training_count": 48,
                        "total_models_created": 15,
                        "total_datasets_used": 8,
                        "last_active_at": "2026-01-16T11:30:00",
                        "most_used_model_type": "classification",
                        "avg_training_time_hours": 3.26
                    }
                }
            }
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/user-activity" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    service = get_dashboard_service()
    
    activity = service.get_user_activity(
        user_id=user_id,
        tenant_id=tenant_id
    )
    
    return success_response({
        'activity': activity.to_dict()
    }, "获取用户活动成功")


@dashboard_bp.route('/gpu-status', methods=['GET'])
@jwt_required()
@handle_dashboard_errors
def get_gpu_status():
    """获取GPU状态
    
    获取所有可用GPU的实时状态信息。
    
    Request Headers:
        Authorization: Bearer <token> (必需)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取GPU状态成功",
                "data": {
                    "gpu_count": 2,
                    "gpu_available": true,
                    "gpus": [
                        {
                            "device_id": 0,
                            "name": "NVIDIA RTX 3090",
                            "utilization": 78.5,
                            "memory_used_gb": 18.2,
                            "memory_total_gb": 24.0,
                            "memory_usage": 75.8,
                            "temperature": 65.0,
                            "power_draw": 280.5,
                            "power_limit": 350.0
                        },
                        {
                            "device_id": 1,
                            "name": "NVIDIA RTX 3090",
                            "utilization": 45.2,
                            "memory_used_gb": 12.5,
                            "memory_total_gb": 24.0,
                            "memory_usage": 52.1,
                            "temperature": 55.0,
                            "power_draw": 180.3,
                            "power_limit": 350.0
                        }
                    ]
                }
            }
        
        200 OK (无GPU):
            {
                "success": true,
                "message": "获取GPU状态成功",
                "data": {
                    "gpu_count": 0,
                    "gpu_available": false,
                    "gpus": []
                }
            }
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/gpu-status" \\
             -H "Authorization: Bearer <token>"
    """
    service = get_dashboard_service()
    
    gpus = service.get_current_gpu_snapshot()
    
    return success_response({
        'gpu_count': len(gpus),
        'gpu_available': len(gpus) > 0,
        'gpus': [g.to_dict() for g in gpus]
    }, "获取GPU状态成功")


@dashboard_bp.route('/alerts', methods=['GET'])
@jwt_required()
@handle_dashboard_errors
def get_alerts_summary():
    """获取告警概要
    
    获取当前活跃的告警数量和分类统计。
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取告警概要成功",
                "data": {
                    "alerts": {
                        "active_count": 2,
                        "critical_count": 0,
                        "warning_count": 2
                    }
                }
            }
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/alerts" \\
             -H "Authorization: Bearer <token>"
    """
    tenant_id = get_tenant_id()
    
    service = get_dashboard_service()
    
    alerts = service.get_alerts_summary(tenant_id=tenant_id)
    
    return success_response({
        'alerts': alerts
    }, "获取告警概要成功")


@dashboard_bp.route('/health', methods=['GET'])
def get_dashboard_health():
    """仪表盘健康检查
    
    检查仪表盘服务是否正常运行。
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "仪表盘服务正常",
                "data": {
                    "status": "healthy",
                    "timestamp": "2026-01-16T12:00:00"
                }
            }
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/health"
    """
    return success_response({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    }, "仪表盘服务正常")


# 导入必要的模块
from datetime import datetime

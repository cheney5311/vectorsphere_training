"""仪表盘统计API接口

提供仪表盘统计相关的API接口，包含训练进度趋势、活跃任务、时长统计、
系统资源历史、模型性能分布和系统健康状态等功能。

API 端点:
    GET /api/v1/dashboard/training/progress    - 获取训练进度趋势
    GET /api/v1/dashboard/training/active      - 获取活跃训练任务
    GET /api/v1/dashboard/training/duration    - 获取训练时长统计
    GET /api/v1/dashboard/system/resources/history - 获取系统资源历史
    GET /api/v1/dashboard/models/performance   - 获取模型性能分布
    GET /api/v1/dashboard/system/health        - 获取系统健康状态
"""

import sys
import os
import logging
from flask import Blueprint, request, jsonify, g
from flask_jwt_extended import jwt_required, get_jwt_identity
from functools import wraps
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.exceptions import ValidationError
from backend.utils.response import success_response, error_response
from backend.services.dashboard_service import get_dashboard_service

logger = logging.getLogger(__name__)

# 创建蓝图
dashboard_statistics_bp = Blueprint('dashboard_statistics', __name__, url_prefix='/api/v1/dashboard')


def get_tenant_id() -> str:
    """获取当前租户ID
    
    从请求头或上下文中获取租户ID。
    
    Returns:
        str: 租户ID，如果未找到则返回 None
    """
    return request.headers.get('X-Tenant-ID') or getattr(g, 'tenant_id', None)


def handle_stats_errors(f):
    """统计API错误处理装饰器
    
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


@dashboard_statistics_bp.route('/training/progress', methods=['GET'])
@jwt_required()
@handle_stats_errors
def get_training_progress():
    """获取训练进度趋势数据
    
    按日期统计训练任务的完成、运行中、失败数量趋势。
    
    Query Parameters:
        days (int, optional): 获取多少天的数据，默认 7，最大 90
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取训练进度趋势成功",
                "data": {
                    "trend": [
                        {
                            "date": "2026-01-10",
                            "completed": 8,
                            "running": 2,
                            "failed": 1,
                            "pending": 0,
                            "total": 11
                        },
                        {
                            "date": "2026-01-11",
                            "completed": 12,
                            "running": 3,
                            "failed": 0,
                            "pending": 1,
                            "total": 16
                        }
                    ],
                    "period": "7 days",
                    "summary": {
                        "total_completed": 45,
                        "total_failed": 3,
                        "total_running": 5,
                        "avg_daily_completed": 6.43,
                        "success_rate": 0.9375
                    }
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/training/progress?days=14" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    # 获取并验证参数
    days = min(int(request.args.get('days', 7)), 90)
    
    service = get_dashboard_service()
    
    result = service.get_training_progress_trend(
        user_id=user_id,
        tenant_id=tenant_id,
        days=days
    )
    
    return success_response(result, "获取训练进度趋势成功")


@dashboard_statistics_bp.route('/training/active', methods=['GET'])
@jwt_required()
@handle_stats_errors
def get_active_training():
    """获取活跃训练任务
    
    获取当前正在运行、等待中或暂停的训练任务列表。
    
    Query Parameters:
        limit (int, optional): 返回数量限制，默认 20，最大 100
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取活跃训练任务成功",
                "data": {
                    "active_tasks": [
                        {
                            "id": "sess_abc123",
                            "name": "BERT文本分类训练",
                            "model_type": "classification",
                            "progress": 75,
                            "accuracy": 92.3,
                            "remaining_time": "2小时15分钟",
                            "status": "running",
                            "start_time": "2026-01-16T10:30:00",
                            "estimated_end": "2026-01-16T16:45:00",
                            "current_epoch": 8,
                            "total_epochs": 10
                        },
                        {
                            "id": "sess_def456",
                            "name": "ResNet图像识别",
                            "model_type": "image_classification",
                            "progress": 45,
                            "accuracy": 88.7,
                            "remaining_time": "4小时30分钟",
                            "status": "running",
                            "start_time": "2026-01-16T08:00:00",
                            "estimated_end": "2026-01-16T18:30:00",
                            "current_epoch": 5,
                            "total_epochs": 12
                        }
                    ],
                    "total_count": 2,
                    "running_count": 2,
                    "pending_count": 0,
                    "paused_count": 0
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/training/active?limit=10" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    # 获取并验证参数
    limit = min(int(request.args.get('limit', 20)), 100)
    
    service = get_dashboard_service()
    
    result = service.get_active_training_tasks(
        user_id=user_id,
        tenant_id=tenant_id,
        limit=limit
    )
    
    return success_response(result, "获取活跃训练任务成功")


@dashboard_statistics_bp.route('/training/duration', methods=['GET'])
@jwt_required()
@handle_stats_errors
def get_training_duration():
    """获取训练时长统计
    
    统计训练任务的平均、最小、最大时长及时长分布。
    
    Query Parameters:
        days (int, optional): 统计天数，默认 30，最大 365
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取训练时长统计成功",
                "data": {
                    "avg_duration": 2.5,
                    "min_duration": 0.5,
                    "max_duration": 8.2,
                    "total_count": 48,
                    "duration_distribution": [
                        {"range": "0-1小时", "count": 5},
                        {"range": "1-3小时", "count": 18},
                        {"range": "3-6小时", "count": 15},
                        {"range": "6-12小时", "count": 8},
                        {"range": "12+小时", "count": 2}
                    ]
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Note:
        - 时长单位为小时
        - 仅统计已完成的训练任务
        
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/training/duration?days=60" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    # 获取并验证参数
    days = min(int(request.args.get('days', 30)), 365)
    
    service = get_dashboard_service()
    
    result = service.get_training_duration_stats(
        user_id=user_id,
        tenant_id=tenant_id,
        days=days
    )
    
    return success_response(result, "获取训练时长统计成功")


@dashboard_statistics_bp.route('/system/resources/history', methods=['GET'])
@jwt_required()
@handle_stats_errors
def get_system_resources_history():
    """获取系统资源历史数据
    
    获取 CPU、内存、GPU、磁盘等系统资源的历史使用情况。
    
    Query Parameters:
        hours (int, optional): 获取多少小时的数据，默认 24，最大 168（一周）
        interval (int, optional): 数据点间隔（分钟），默认 60
    
    Request Headers:
        Authorization: Bearer <token> (必需)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取系统资源历史数据成功",
                "data": {
                    "history": [
                        {
                            "timestamp": "2026-01-15T12:00:00",
                            "cpu_percent": 45.5,
                            "memory_percent": 62.3,
                            "gpu_percent": 78.5,
                            "disk_percent": 55.2
                        },
                        {
                            "timestamp": "2026-01-15T13:00:00",
                            "cpu_percent": 52.3,
                            "memory_percent": 65.1,
                            "gpu_percent": 82.1,
                            "disk_percent": 55.3
                        }
                    ],
                    "period_hours": 24,
                    "interval_minutes": 60,
                    "data_points": 24
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Note:
        由于没有持久化的监控数据，此API返回基于当前值的模拟历史数据。
        生产环境建议接入 Prometheus/InfluxDB 等时序数据库。
        
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/system/resources/history?hours=48&interval=30" \\
             -H "Authorization: Bearer <token>"
    """
    # 获取并验证参数
    hours = min(int(request.args.get('hours', 24)), 168)
    interval = max(min(int(request.args.get('interval', 60)), 120), 5)  # 5-120分钟
    
    service = get_dashboard_service()
    
    history = service.get_resource_usage_history(
        hours=hours,
        interval_minutes=interval
    )
    
    return success_response({
        'history': history,
        'period_hours': hours,
        'interval_minutes': interval,
        'data_points': len(history)
    }, "获取系统资源历史数据成功")


@dashboard_statistics_bp.route('/models/performance', methods=['GET'])
@jwt_required()
@handle_stats_errors
def get_model_performance():
    """获取模型性能分布
    
    统计模型准确率分布和按模型类型的性能统计。
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取模型性能分布成功",
                "data": {
                    "accuracy_distribution": [
                        {"range": "90-100%", "count": 5},
                        {"range": "80-90%", "count": 12},
                        {"range": "70-80%", "count": 3},
                        {"range": "60-70%", "count": 1},
                        {"range": "<60%", "count": 0}
                    ],
                    "model_types": [
                        {
                            "type": "classification",
                            "count": 8,
                            "avg_accuracy": 91.2
                        },
                        {
                            "type": "regression",
                            "count": 5,
                            "avg_accuracy": 87.8
                        },
                        {
                            "type": "nlp",
                            "count": 3,
                            "avg_accuracy": 89.5
                        }
                    ],
                    "total_models": 21
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/models/performance" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    service = get_dashboard_service()
    
    result = service.get_model_performance_distribution(
        user_id=user_id,
        tenant_id=tenant_id
    )
    
    return success_response(result, "获取模型性能分布成功")


@dashboard_statistics_bp.route('/system/health', methods=['GET'])
@jwt_required()
@handle_stats_errors
def get_system_health():
    """获取系统健康状态
    
    检查各服务组件的运行状态，包括数据库、API服务器、调度器等。
    
    Request Headers:
        Authorization: Bearer <token> (必需)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取系统健康状态成功",
                "data": {
                    "overall_status": "healthy",
                    "services": {
                        "database": "running",
                        "api_server": "running",
                        "scheduler": "running",
                        "model_manager": "running"
                    },
                    "uptime": "5天 12小时 30分钟",
                    "uptime_seconds": 475800,
                    "last_check": "2026-01-16T14:30:00"
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Note:
        - overall_status 可能的值: healthy, degraded, unhealthy, unknown
        - 各服务状态可能的值: running, error, unknown
        
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/system/health" \\
             -H "Authorization: Bearer <token>"
    """
    service = get_dashboard_service()
    
    result = service.get_system_health_status()
    
    return success_response(result, "获取系统健康状态成功")


@dashboard_statistics_bp.route('/training/summary', methods=['GET'])
@jwt_required()
@handle_stats_errors
def get_training_summary():
    """获取训练汇总统计
    
    获取训练任务的综合汇总统计信息。
    
    Query Parameters:
        days (int, optional): 统计天数，默认 30
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取训练汇总统计成功",
                "data": {
                    "overview": {
                        "total_sessions": 156,
                        "completed_sessions": 142,
                        "failed_sessions": 8,
                        "running_sessions": 6,
                        "success_rate": 0.9467
                    },
                    "time_stats": {
                        "total_training_hours": 425.5,
                        "avg_training_hours": 2.73,
                        "min_training_hours": 0.25,
                        "max_training_hours": 12.5
                    },
                    "performance_stats": {
                        "avg_accuracy": 0.8745,
                        "best_accuracy": 0.9823,
                        "avg_loss": 0.1256
                    },
                    "resource_utilization": {
                        "avg_cpu_usage": 45.5,
                        "avg_memory_usage": 62.3,
                        "avg_gpu_usage": 78.5
                    }
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/training/summary?days=60" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    # 获取并验证参数
    days = min(int(request.args.get('days', 30)), 365)
    
    service = get_dashboard_service()
    
    # 获取各项统计
    from backend.schemas.dashboard_models import DashboardTimeRange
    
    # 使用已有方法获取数据
    overview = service.get_dashboard_overview(
        user_id=user_id,
        tenant_id=tenant_id,
        time_range=DashboardTimeRange.LAST_30_DAYS
    )
    
    duration_stats = service.get_training_duration_stats(
        user_id=user_id,
        tenant_id=tenant_id,
        days=days
    )
    
    current_system = service.get_current_system_snapshot()
    
    result = {
        'overview': {
            'total_sessions': overview.training.active_count + overview.training.completed_count + overview.training.failed_count,
            'completed_sessions': overview.training.completed_count,
            'failed_sessions': overview.training.failed_count,
            'running_sessions': overview.training.active_count,
            'pending_sessions': overview.training.pending_count,
            'success_rate': overview.training.success_rate
        },
        'time_stats': {
            'total_training_hours': overview.training.total_training_time_hours,
            'avg_training_hours': duration_stats.get('avg_duration', 0),
            'min_training_hours': duration_stats.get('min_duration', 0),
            'max_training_hours': duration_stats.get('max_duration', 0)
        },
        'performance_stats': {
            'avg_accuracy': overview.models.avg_accuracy,
            'best_accuracy': overview.models.best_accuracy,
            'avg_f1_score': overview.models.avg_f1_score
        },
        'resource_utilization': {
            'avg_cpu_usage': current_system.cpu_usage,
            'avg_memory_usage': current_system.memory_usage,
            'disk_usage': current_system.disk_usage
        },
        'period_days': days
    }
    
    return success_response(result, "获取训练汇总统计成功")


@dashboard_statistics_bp.route('/comparison', methods=['GET'])
@jwt_required()
@handle_stats_errors
def get_period_comparison():
    """获取周期对比统计
    
    对比当前周期与上一周期的训练统计数据。
    
    Query Parameters:
        period (str, optional): 对比周期，可选值: day, week, month，默认 week
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取周期对比统计成功",
                "data": {
                    "period": "week",
                    "current": {
                        "start_date": "2026-01-09",
                        "end_date": "2026-01-16",
                        "total_tasks": 45,
                        "completed_tasks": 42,
                        "failed_tasks": 3,
                        "success_rate": 0.9333
                    },
                    "previous": {
                        "start_date": "2026-01-02",
                        "end_date": "2026-01-09",
                        "total_tasks": 38,
                        "completed_tasks": 35,
                        "failed_tasks": 3,
                        "success_rate": 0.9211
                    },
                    "comparison": {
                        "total_tasks_change": 7,
                        "total_tasks_change_percent": 18.42,
                        "completed_tasks_change": 7,
                        "completed_tasks_change_percent": 20.0,
                        "success_rate_change": 0.0122
                    }
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/dashboard/comparison?period=month" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    # 获取并验证参数
    period = request.args.get('period', 'week')
    if period not in ['day', 'week', 'month']:
        period = 'week'
    
    # 计算时间范围
    from datetime import timedelta
    
    now = datetime.utcnow()
    
    period_days = {
        'day': 1,
        'week': 7,
        'month': 30
    }
    days = period_days.get(period, 7)
    
    service = get_dashboard_service()
    
    # 获取当前周期数据
    current_trend = service.get_training_progress_trend(
        user_id=user_id,
        tenant_id=tenant_id,
        days=days
    )
    
    # 获取上一周期数据（通过获取两倍天数然后取前半部分）
    # 由于仓库方法不支持自定义日期范围，这里使用简化的计算方式
    from backend.repositories.dashboard_repository import get_dashboard_repository
    repo = get_dashboard_repository()
    
    current_end = now
    current_start = now - timedelta(days=days)
    previous_end = current_start
    previous_start = current_start - timedelta(days=days)
    
    # 当前周期统计
    current_overview = repo.get_training_overview(
        user_id=user_id,
        tenant_id=tenant_id,
        start_date=current_start,
        end_date=current_end
    )
    
    # 上一周期统计
    previous_overview = repo.get_training_overview(
        user_id=user_id,
        tenant_id=tenant_id,
        start_date=previous_start,
        end_date=previous_end
    )
    
    # 计算变化
    current_total = current_overview.get('total_count', 0)
    previous_total = previous_overview.get('total_count', 0)
    current_completed = current_overview.get('completed_count', 0)
    previous_completed = previous_overview.get('completed_count', 0)
    current_success_rate = current_overview.get('success_rate', 0)
    previous_success_rate = previous_overview.get('success_rate', 0)
    
    result = {
        'period': period,
        'current': {
            'start_date': current_start.strftime('%Y-%m-%d'),
            'end_date': current_end.strftime('%Y-%m-%d'),
            'total_tasks': current_total,
            'completed_tasks': current_completed,
            'failed_tasks': current_overview.get('failed_count', 0),
            'success_rate': round(current_success_rate, 4)
        },
        'previous': {
            'start_date': previous_start.strftime('%Y-%m-%d'),
            'end_date': previous_end.strftime('%Y-%m-%d'),
            'total_tasks': previous_total,
            'completed_tasks': previous_completed,
            'failed_tasks': previous_overview.get('failed_count', 0),
            'success_rate': round(previous_success_rate, 4)
        },
        'comparison': {
            'total_tasks_change': current_total - previous_total,
            'total_tasks_change_percent': round((current_total - previous_total) / previous_total * 100, 2) if previous_total > 0 else 0,
            'completed_tasks_change': current_completed - previous_completed,
            'completed_tasks_change_percent': round((current_completed - previous_completed) / previous_completed * 100, 2) if previous_completed > 0 else 0,
            'success_rate_change': round(current_success_rate - previous_success_rate, 4)
        }
    }
    
    return success_response(result, "获取周期对比统计成功")

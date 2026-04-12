"""用户训练信息API接口

提供用户训练信息相关的API接口，支持前端仪表盘显示。

API 端点:
    GET /api/v1/user/training/overview          - 获取用户训练概览
    GET /api/v1/user/training/recent-sessions   - 获取最近训练会话
    GET /api/v1/user/training/sessions          - 获取训练会话列表（分页）
    GET /api/v1/user/training/sessions/<id>     - 获取会话详情
    GET /api/v1/user/training/active            - 获取活跃训练会话
    GET /api/v1/user/training/statistics        - 获取训练统计
    GET /api/v1/user/training/trend             - 获取训练趋势
    GET /api/v1/user/training/duration-stats    - 获取时长统计
    GET /api/v1/user/training/model-ranking     - 获取模型性能排行
"""

import sys
import os
import logging
from flask import Blueprint, request, jsonify, g
from flask_jwt_extended import jwt_required, get_jwt_identity
from functools import wraps
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError
from backend.utils.response import success_response, error_response
from backend.services.user_training_service import get_user_training_service

logger = logging.getLogger(__name__)

# 创建蓝图
user_training_bp = Blueprint('user_training', __name__, url_prefix='/api/v1/user/training')


# =============================================================================
# 辅助函数和装饰器
# =============================================================================

def get_tenant_id() -> str:
    """获取当前租户ID
    
    从请求头或上下文中获取租户ID。
    
    Returns:
        str: 租户ID，如果未找到则返回 None
    """
    return request.headers.get('X-Tenant-ID') or getattr(g, 'tenant_id', None)


def handle_errors(f):
    """错误处理装饰器
    
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


# =============================================================================
# 概览相关 API
# =============================================================================

@user_training_bp.route('/overview', methods=['GET'])
@jwt_required()
@handle_errors
def get_user_training_overview():
    """获取用户训练概览信息
    
    汇总当前用户的训练活动数据，包括活跃会话、完成会话、模型数量和平均准确率等。
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取用户训练概览成功",
                "data": {
                    "activeSessions": 2,
                    "completedSessions": 45,
                    "totalModels": 12,
                    "avgAccuracy": 87.5,
                    "successRate": 92.3,
                    "totalTrainingHours": 156.5
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/user/training/overview" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    service = get_user_training_service()
    overview = service.get_user_overview(user_id=user_id, tenant_id=tenant_id)
    
    return success_response(overview.to_dict(), "获取用户训练概览成功")


# =============================================================================
# 会话相关 API
# =============================================================================

@user_training_bp.route('/recent-sessions', methods=['GET'])
@jwt_required()
@handle_errors
def get_recent_training_sessions():
    """获取用户最近的训练会话
    
    获取当前用户最近的训练会话列表，按创建时间倒序排列。
    
    Query Parameters:
        limit (int, optional): 返回数量，默认5，最大20
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取最近训练会话成功",
                "data": {
                    "recentSessions": [
                        {
                            "id": "sess_abc123",
                            "name": "BERT文本分类训练",
                            "modelType": "classification",
                            "status": "completed",
                            "progress": 100,
                            "accuracy": 0.9234,
                            "loss": 0.0856,
                            "startTime": "2026-01-15T10:30:00",
                            "endTime": "2026-01-15T14:45:00",
                            "durationMinutes": 255.0,
                            "currentEpoch": 10,
                            "totalEpochs": 10
                        },
                        {
                            "id": "sess_def456",
                            "name": "ResNet图像识别",
                            "modelType": "image_classification",
                            "status": "running",
                            "progress": 65,
                            "accuracy": 0.8756,
                            "loss": 0.1234,
                            "startTime": "2026-01-16T08:00:00",
                            "endTime": null,
                            "durationMinutes": 180.5,
                            "currentEpoch": 7,
                            "totalEpochs": 10
                        }
                    ],
                    "count": 2
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/user/training/recent-sessions?limit=10" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    limit = min(int(request.args.get('limit', 5)), 20)
    
    service = get_user_training_service()
    sessions = service.get_recent_sessions(
        user_id=user_id,
        limit=limit,
        tenant_id=tenant_id
    )
    
    return success_response({
        'recentSessions': [s.to_dict() for s in sessions],
        'count': len(sessions)
    }, "获取最近训练会话成功")


@user_training_bp.route('/sessions', methods=['GET'])
@jwt_required()
@handle_errors
def get_user_training_sessions():
    """获取用户所有训练会话（支持分页和筛选）
    
    获取当前用户的所有训练会话，支持分页、状态筛选、日期范围筛选和排序。
    
    Query Parameters:
        page (int, optional): 页码，从1开始，默认1
        limit (int, optional): 每页数量，默认10，最大50
        status (str, optional): 状态筛选，可选值: pending, running, training, completed, failed, cancelled
        model_type (str, optional): 模型类型筛选
        start_date (str, optional): 开始日期筛选 (ISO格式: YYYY-MM-DD)
        end_date (str, optional): 结束日期筛选 (ISO格式: YYYY-MM-DD)
        sort_by (str, optional): 排序字段，默认 'created_at'，可选: created_at, started_at, progress, status
        sort_order (str, optional): 排序方向，默认 'desc'，可选: asc, desc
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取训练会话列表成功",
                "data": {
                    "sessions": [
                        {
                            "id": "sess_abc123",
                            "name": "BERT文本分类训练",
                            "modelType": "classification",
                            "status": "completed",
                            "progress": 100,
                            "accuracy": 0.9234,
                            "loss": 0.0856,
                            "startTime": "2026-01-15T10:30:00",
                            "endTime": "2026-01-15T14:45:00",
                            "durationMinutes": 255.0
                        }
                    ],
                    "total": 45,
                    "page": 1,
                    "limit": 10,
                    "totalPages": 5
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/user/training/sessions?page=1&limit=10&status=completed" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    # 获取查询参数
    page = max(int(request.args.get('page', 1)), 1)
    limit = min(max(int(request.args.get('limit', 10)), 1), 50)
    status = request.args.get('status')
    model_type = request.args.get('model_type')
    sort_by = request.args.get('sort_by', 'created_at')
    sort_order = request.args.get('sort_order', 'desc')
    
    # 日期筛选
    start_date = None
    end_date = None
    if request.args.get('start_date'):
        try:
            start_date = datetime.fromisoformat(request.args.get('start_date'))
        except ValueError:
            pass
    if request.args.get('end_date'):
        try:
            end_date = datetime.fromisoformat(request.args.get('end_date'))
        except ValueError:
            pass
    
    service = get_user_training_service()
    sessions, total = service.get_user_sessions(
        user_id=user_id,
        page=page,
        limit=limit,
        status=status,
        model_type=model_type,
        start_date=start_date,
        end_date=end_date,
        sort_by=sort_by,
        sort_order=sort_order,
        tenant_id=tenant_id
    )
    
    total_pages = (total + limit - 1) // limit
    
    return success_response({
        'sessions': [s.to_dict() for s in sessions],
        'total': total,
        'page': page,
        'limit': limit,
        'totalPages': total_pages
    }, "获取训练会话列表成功")


@user_training_bp.route('/sessions/<session_id>', methods=['GET'])
@jwt_required()
@handle_errors
def get_training_session_detail(session_id: str):
    """获取训练会话详情
    
    获取指定训练会话的详细信息，包括配置、结果和进度历史。
    
    Path Parameters:
        session_id (str): 会话ID
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取会话详情成功",
                "data": {
                    "session_id": "sess_abc123",
                    "user_id": "user_123",
                    "model_id": "model_456",
                    "dataset_id": "dataset_789",
                    "training_type": "classification",
                    "status": "completed",
                    "progress": 100,
                    "config": {
                        "session_name": "BERT文本分类训练",
                        "epochs": 10,
                        "batch_size": 32,
                        "learning_rate": 0.001
                    },
                    "result": {
                        "final_accuracy": 0.9234,
                        "final_loss": 0.0856
                    },
                    "created_at": "2026-01-15T10:30:00",
                    "started_at": "2026-01-15T10:31:00",
                    "completed_at": "2026-01-15T14:45:00",
                    "progress_history": [
                        {
                            "epoch": 1,
                            "step": 100,
                            "loss": 0.5,
                            "accuracy": 0.7,
                            "timestamp": "2026-01-15T10:45:00"
                        }
                    ]
                }
            }
        
        401 Unauthorized: 未授权访问
        404 Not Found: 会话不存在或无权限访问
        500 Internal Server Error: 服务器内部错误
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/user/training/sessions/sess_abc123" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    service = get_user_training_service()
    detail = service.get_session_detail(
        session_id=session_id,
        user_id=user_id,
        tenant_id=tenant_id
    )
    
    if not detail:
        return error_response("会话不存在或无权限访问", 404)
    
    return success_response(detail, "获取会话详情成功")


@user_training_bp.route('/active', methods=['GET'])
@jwt_required()
@handle_errors
def get_active_training_sessions():
    """获取用户活跃训练会话
    
    获取当前用户正在运行或等待中的训练会话。
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取活跃训练会话成功",
                "data": {
                    "activeSessions": [
                        {
                            "id": "sess_abc123",
                            "name": "BERT训练",
                            "modelType": "classification",
                            "status": "running",
                            "progress": 65,
                            "accuracy": 0.8756,
                            "startTime": "2026-01-16T08:00:00",
                            "durationMinutes": 180.5
                        }
                    ],
                    "count": 1,
                    "runningCount": 1,
                    "pendingCount": 0
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/user/training/active" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    service = get_user_training_service()
    sessions = service.get_active_sessions(
        user_id=user_id,
        tenant_id=tenant_id
    )
    
    running_count = sum(1 for s in sessions if s.status in ['running', 'training'])
    pending_count = sum(1 for s in sessions if s.status == 'pending')
    
    return success_response({
        'activeSessions': [s.to_dict() for s in sessions],
        'count': len(sessions),
        'runningCount': running_count,
        'pendingCount': pending_count
    }, "获取活跃训练会话成功")


# =============================================================================
# 统计相关 API
# =============================================================================

@user_training_bp.route('/statistics', methods=['GET'])
@jwt_required()
@handle_errors
def get_user_training_statistics():
    """获取用户训练统计信息
    
    汇总当前用户在指定时间范围内的训练统计数据。
    
    Query Parameters:
        days (int, optional): 统计天数，默认30，最大365
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取训练统计成功",
                "data": {
                    "totalTasks": 50,
                    "completedTasks": 45,
                    "runningTasks": 2,
                    "pendingTasks": 1,
                    "failedTasks": 2,
                    "cancelledTasks": 0,
                    "successRate": 90.0,
                    "avgTrainingTime": 120.5,
                    "totalTrainingTime": 156.5,
                    "avgAccuracy": 87.5,
                    "bestAccuracy": 95.2,
                    "avgLoss": 0.125,
                    "bestLoss": 0.056
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Note:
        - avgTrainingTime 单位为分钟
        - totalTrainingTime 单位为小时
        - accuracy 为百分比形式
        
    Example:
        curl -X GET "http://localhost:5000/api/v1/user/training/statistics?days=60" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    days = min(int(request.args.get('days', 30)), 365)
    
    service = get_user_training_service()
    stats = service.get_user_statistics(
        user_id=user_id,
        days=days,
        tenant_id=tenant_id
    )
    
    return success_response(stats.to_dict(), "获取训练统计成功")


@user_training_bp.route('/trend', methods=['GET'])
@jwt_required()
@handle_errors
def get_training_trend():
    """获取训练趋势数据
    
    按日期统计训练任务的数量变化趋势。
    
    Query Parameters:
        days (int, optional): 统计天数，默认7，最大90
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取训练趋势成功",
                "data": {
                    "trend": [
                        {
                            "date": "2026-01-10",
                            "completed": 5,
                            "running": 1,
                            "failed": 0,
                            "total": 6,
                            "avg_accuracy": 0.0
                        },
                        {
                            "date": "2026-01-11",
                            "completed": 8,
                            "running": 2,
                            "failed": 1,
                            "total": 11,
                            "avg_accuracy": 0.0
                        }
                    ],
                    "period": "7 days",
                    "summary": {
                        "totalCompleted": 35,
                        "totalFailed": 3,
                        "avgDailyCompleted": 5.0
                    }
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/user/training/trend?days=14" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    days = min(int(request.args.get('days', 7)), 90)
    
    service = get_user_training_service()
    trend_data = service.get_training_trend(
        user_id=user_id,
        days=days,
        tenant_id=tenant_id
    )
    
    # 计算汇总
    total_completed = sum(t.completed for t in trend_data)
    total_failed = sum(t.failed for t in trend_data)
    
    return success_response({
        'trend': [t.to_dict() for t in trend_data],
        'period': f'{days} days',
        'summary': {
            'totalCompleted': total_completed,
            'totalFailed': total_failed,
            'avgDailyCompleted': round(total_completed / days, 2) if days > 0 else 0
        }
    }, "获取训练趋势成功")


@user_training_bp.route('/duration-stats', methods=['GET'])
@jwt_required()
@handle_errors
def get_training_duration_stats():
    """获取训练时长统计
    
    统计用户训练任务的时长分布。
    
    Query Parameters:
        days (int, optional): 统计天数，默认30，最大365
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取时长统计成功",
                "data": {
                    "avgDuration": 45.5,
                    "minDuration": 10.2,
                    "maxDuration": 180.5,
                    "totalDuration": 156.5,
                    "totalCount": 45,
                    "distribution": [
                        {"range": "0-30分钟", "count": 10},
                        {"range": "30-60分钟", "count": 15},
                        {"range": "1-2小时", "count": 12},
                        {"range": "2-4小时", "count": 6},
                        {"range": "4+小时", "count": 2}
                    ]
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Note:
        - avgDuration, minDuration, maxDuration 单位为分钟
        - totalDuration 单位为小时
        
    Example:
        curl -X GET "http://localhost:5000/api/v1/user/training/duration-stats?days=60" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    days = min(int(request.args.get('days', 30)), 365)
    
    service = get_user_training_service()
    stats = service.get_training_duration_stats(
        user_id=user_id,
        days=days,
        tenant_id=tenant_id
    )
    
    return success_response(stats, "获取时长统计成功")


@user_training_bp.route('/model-ranking', methods=['GET'])
@jwt_required()
@handle_errors
def get_model_performance_ranking():
    """获取模型性能排行
    
    按最佳准确率排序获取用户的模型性能排行。
    
    Query Parameters:
        limit (int, optional): 返回数量，默认10，最大20
    
    Request Headers:
        Authorization: Bearer <token> (必需)
        X-Tenant-ID: <tenant_id> (可选)
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "获取模型排行成功",
                "data": {
                    "ranking": [
                        {
                            "modelId": "model_123",
                            "modelName": "BERT-base",
                            "modelType": "classification",
                            "bestAccuracy": 0.9523,
                            "bestLoss": 0.0456,
                            "trainingCount": 5,
                            "lastTrained": "2026-01-15T14:45:00"
                        },
                        {
                            "modelId": "model_456",
                            "modelName": "ResNet50",
                            "modelType": "image_classification",
                            "bestAccuracy": 0.9234,
                            "bestLoss": 0.0856,
                            "trainingCount": 3,
                            "lastTrained": "2026-01-14T16:30:00"
                        }
                    ],
                    "count": 2
                }
            }
        
        401 Unauthorized: 未授权访问
        500 Internal Server Error: 服务器内部错误
    
    Example:
        curl -X GET "http://localhost:5000/api/v1/user/training/model-ranking?limit=5" \\
             -H "Authorization: Bearer <token>"
    """
    user_id = get_jwt_identity()
    tenant_id = get_tenant_id()
    
    limit = min(int(request.args.get('limit', 10)), 20)
    
    service = get_user_training_service()
    ranking = service.get_model_performance_ranking(
        user_id=user_id,
        limit=limit,
        tenant_id=tenant_id
    )
    
    return success_response({
        'ranking': [m.to_dict() for m in ranking],
        'count': len(ranking)
    }, "获取模型排行成功")


# =============================================================================
# 健康检查
# =============================================================================

@user_training_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查端点
    
    Returns:
        200 OK:
            {
                "success": true,
                "message": "服务正常",
                "data": {
                    "status": "healthy",
                    "timestamp": "2026-01-16T14:30:00"
                }
            }
    """
    return success_response({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat()
    }, "服务正常")

# -*- coding: utf-8 -*-
"""训练进度API接口

提供训练进度管理的完整 RESTful API。

API端点：
- GET    /api/v1/training/progress/<session_id>           - 获取训练进度
- POST   /api/v1/training/progress/<session_id>           - 更新训练进度
- GET    /api/v1/training/progress/<session_id>/history   - 获取进度历史
- GET    /api/v1/training/progress/<session_id>/logs      - 获取训练日志
- GET    /api/v1/training/progress/<session_id>/metrics   - 获取训练指标
- GET    /api/v1/training/progress/<session_id>/events    - 获取训练事件
- GET    /api/v1/training/progress/<session_id>/checkpoints - 获取检查点
- GET    /api/v1/training/progress/<session_id>/resources - 获取资源使用
- GET    /api/v1/training/progress/<session_id>/summary   - 获取进度摘要

架构：
API层 -> Service层 (training_progress_service.py) -> Repository层 -> Database
              ↘-> Training模块 (backend/modules/training)
"""

import logging
from datetime import datetime
from flask import Blueprint, request

logger = logging.getLogger(__name__)

# 响应工具
try:
    from backend.utils.response import success_response, error_response, paginated_response
except ImportError:
    from flask import jsonify
    
    def success_response(data, message="Success", status_code=200):
        return jsonify({'success': True, 'data': data, 'message': message}), status_code
    
    def error_response(message, status_code=400):
        return jsonify({'success': False, 'error': message}), status_code
    
    def paginated_response(items, page, limit, total, message="Success"):
        return jsonify({
            'success': True,
            'data': items,
            'page': page,
            'limit': limit,
            'total': total,
            'message': message
        }), 200

# JWT认证
try:
    from flask_jwt_extended import jwt_required, get_jwt_identity
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    def jwt_required():
        def decorator(f):
            return f
        return decorator
    def get_jwt_identity():
        return 'anonymous_user'

# 服务层导入
from backend.services.training_progress_service import get_training_progress_service


# 创建蓝图
training_progress_bp = Blueprint('training_progress', __name__, url_prefix='/api/v1/training/progress')


def _get_current_user() -> str:
    """获取当前用户ID"""
    if JWT_AVAILABLE:
        identity = get_jwt_identity()
        if isinstance(identity, dict):
            return identity.get('user_id', identity.get('id', 'anonymous'))
        return str(identity) if identity else 'anonymous'
    return 'anonymous_user'


def _get_tenant_id() -> str:
    """获取当前租户ID"""
    tenant_id = request.headers.get('X-Tenant-ID')
    if tenant_id:
        return tenant_id
    
    if JWT_AVAILABLE:
        identity = get_jwt_identity()
        if isinstance(identity, dict):
            return identity.get('tenant_id', 'default')
    
    return 'default'


# ==================== 进度查询 ====================

@training_progress_bp.route('/<session_id>', methods=['GET'])
@jwt_required()
def get_training_progress(session_id: str):
    """
    获取训练进度
    
    Path Parameters:
        - session_id: 训练会话ID
    
    Returns:
        {
            "success": true,
            "data": {
                "session_id": "uuid",
                "current_epoch": 5,
                "total_epochs": 10,
                "current_step": 500,
                "total_steps": 1000,
                "current_stage": "finetune",
                "total_stages": 3,
                "loss": 0.15,
                "accuracy": 0.92,
                "learning_rate": 1e-4,
                "status": "running",
                "progress_percentage": 50.0,
                "stage_progress_percentage": 66.7,
                "updated_at": "2025-01-09T10:00:00Z"
            },
            "message": "获取训练进度成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_progress_service()
        progress = service.get_progress(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        if not progress:
            return error_response("训练会话不存在", 404)
        
        return success_response(progress, "获取训练进度成功")
        
    except Exception as e:
        logger.error(f"Error getting progress for session {session_id}: {e}")
        return error_response(f"获取训练进度失败: {str(e)}", 500)


@training_progress_bp.route('/<session_id>', methods=['POST'])
@jwt_required()
def update_training_progress(session_id: str):
    """
    更新训练进度
    
    用于训练模块上报进度信息。
    
    Request Body:
        {
            "stage": "finetune",
            "epoch": 5,
            "step": 500,
            "total_steps": 1000,
            "total_epochs": 10,
            "loss": 0.15,
            "accuracy": 0.92,
            "learning_rate": 1e-4,
            "metrics": {...},
            "gpu_utilization": 0.85,
            "gpu_memory_used": 12.5,
            "gpu_memory_total": 16.0,
            "cpu_utilization": 0.45
        }
    
    Returns:
        {
            "success": true,
            "data": {"session_id": "uuid"},
            "message": "进度更新成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        service = get_training_progress_service()
        result = service.update_progress(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            progress_data=data
        )
        
        if not result.get('success'):
            return error_response(result.get('message', '更新失败'), 400)
        
        return success_response(result, "进度更新成功")
        
    except Exception as e:
        logger.error(f"Error updating progress for session {session_id}: {e}")
        return error_response(f"更新训练进度失败: {str(e)}", 500)


@training_progress_bp.route('/<session_id>/history', methods=['GET'])
@jwt_required()
def get_progress_history(session_id: str):
    """
    获取进度历史记录
    
    Query Parameters:
        - limit: 返回数量 (默认: 100)
        - offset: 偏移量 (默认: 0)
    
    Returns:
        {
            "success": true,
            "data": {
                "history": [...],
                "total": 500
            },
            "message": "获取进度历史成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = get_training_progress_service()
        result = service.get_progress_history(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            limit=limit,
            offset=offset
        )
        
        return success_response(result, "获取进度历史成功")
        
    except Exception as e:
        logger.error(f"Error getting progress history: {e}")
        return error_response(f"获取进度历史失败: {str(e)}", 500)


# ==================== 日志管理 ====================

@training_progress_bp.route('/<session_id>/logs', methods=['GET'])
@jwt_required()
def get_training_logs(session_id: str):
    """
    获取训练日志
    
    Query Parameters:
        - limit: 日志行数限制 (默认: 100)
        - level: 日志级别过滤 (DEBUG/INFO/WARNING/ERROR)
    
    Returns:
        {
            "success": true,
            "data": {
                "session_id": "uuid",
                "logs": [
                    {
                        "timestamp": "2025-01-09T10:00:00Z",
                        "level": "INFO",
                        "message": "Epoch 5/10 completed"
                    }
                ],
                "total": 100
            },
            "message": "获取训练日志成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        limit = request.args.get('limit', 100, type=int)
        level = request.args.get('level')
        
        service = get_training_progress_service()
        result = service.get_logs(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            limit=limit,
            level=level
        )
        
        return success_response(result, "获取训练日志成功")
        
    except Exception as e:
        logger.error(f"Error getting logs for session {session_id}: {e}")
        return error_response(f"获取训练日志失败: {str(e)}", 500)


# ==================== 指标管理 ====================

@training_progress_bp.route('/<session_id>/metrics', methods=['GET'])
@jwt_required()
def get_training_metrics(session_id: str):
    """
    获取训练指标
    
    Query Parameters:
        - limit: 指标数量限制 (默认: 100)
        - metrics: 指定指标名称（逗号分隔）
    
    Returns:
        {
            "success": true,
            "data": {
                "session_id": "uuid",
                "metrics": [
                    {
                        "timestamp": "2025-01-09T10:00:00Z",
                        "epoch": 5,
                        "step": 500,
                        "loss": 0.15,
                        "accuracy": 0.92,
                        "learning_rate": 1e-4,
                        "gpu_utilization": 0.85,
                        "gpu_memory_used": 12.5,
                        "cpu_utilization": 0.45,
                        ...
                    }
                ],
                "total": 100
            },
            "message": "获取训练指标成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        limit = request.args.get('limit', 100, type=int)
        metrics_param = request.args.get('metrics')
        metric_names = metrics_param.split(',') if metrics_param else None
        
        service = get_training_progress_service()
        result = service.get_metrics(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            limit=limit,
            metric_names=metric_names
        )
        
        return success_response(result, "获取训练指标成功")
        
    except Exception as e:
        logger.error(f"Error getting metrics for session {session_id}: {e}")
        return error_response(f"获取训练指标失败: {str(e)}", 500)


@training_progress_bp.route('/<session_id>/metrics/summary', methods=['GET'])
@jwt_required()
def get_metrics_summary(session_id: str):
    """
    获取指标摘要统计
    
    Returns:
        {
            "success": true,
            "data": {
                "loss": {"min": 0.1, "max": 1.0, "avg": 0.3},
                "accuracy": {"min": 0.5, "max": 0.95, "avg": 0.8},
                "epochs_completed": 5,
                "steps_completed": 500,
                "avg_gpu_utilization": 0.82,
                "avg_cpu_utilization": 0.45
            },
            "message": "获取指标摘要成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_progress_service()
        result = service.get_metric_summary(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        return success_response(result, "获取指标摘要成功")
        
    except Exception as e:
        logger.error(f"Error getting metric summary: {e}")
        return error_response(f"获取指标摘要失败: {str(e)}", 500)


# ==================== 事件管理 ====================

@training_progress_bp.route('/<session_id>/events', methods=['GET'])
@jwt_required()
def get_training_events(session_id: str):
    """
    获取训练事件
    
    Query Parameters:
        - event_types: 事件类型过滤（逗号分隔）
        - limit: 事件数量限制 (默认: 100)
    
    Returns:
        {
            "success": true,
            "data": {
                "session_id": "uuid",
                "events": [
                    {
                        "timestamp": "2025-01-09T10:00:00Z",
                        "event_type": "epoch_completed",
                        "description": "完成第 5 轮训练",
                        "details": {
                            "epoch": 5,
                            "loss": 0.15,
                            "accuracy": 0.92
                        }
                    }
                ]
            },
            "message": "获取训练事件成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        event_types_param = request.args.get('event_types')
        event_types = event_types_param.split(',') if event_types_param else None
        limit = request.args.get('limit', 100, type=int)
        
        service = get_training_progress_service()
        result = service.get_events(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            event_types=event_types,
            limit=limit
        )
        
        return success_response(result, "获取训练事件成功")
        
    except Exception as e:
        logger.error(f"Error getting events for session {session_id}: {e}")
        return error_response(f"获取训练事件失败: {str(e)}", 500)


# ==================== 检查点管理 ====================

@training_progress_bp.route('/<session_id>/checkpoints', methods=['GET'])
@jwt_required()
def get_checkpoints(session_id: str):
    """
    获取检查点列表
    
    Returns:
        {
            "success": true,
            "data": {
                "session_id": "uuid",
                "checkpoints": [
                    {
                        "epoch": 5,
                        "step": 500,
                        "path": "/checkpoints/session_xxx/epoch_5",
                        "created_at": "2025-01-09T10:00:00Z",
                        "metrics": {
                            "loss": 0.15,
                            "accuracy": 0.92
                        }
                    }
                ]
            },
            "message": "获取检查点列表成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_progress_service()
        result = service.get_checkpoints(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        return success_response(result, "获取检查点列表成功")
        
    except Exception as e:
        logger.error(f"Error getting checkpoints for session {session_id}: {e}")
        return error_response(f"获取检查点失败: {str(e)}", 500)


# ==================== 资源监控 ====================

@training_progress_bp.route('/<session_id>/resources', methods=['GET'])
@jwt_required()
def get_resource_usage(session_id: str):
    """
    获取资源使用情况
    
    Returns:
        {
            "success": true,
            "data": {
                "gpu": {
                    "utilization": 0.85,
                    "memory_used": 12.5,
                    "memory_total": 16.0,
                    "temperature": 72.0,
                    "power_draw": 180.0
                },
                "cpu": {
                    "utilization": 0.45,
                    "memory_used": 32.0,
                    "memory_total": 64.0,
                    "temperature": 55.0
                },
                "disk": {
                    "read_speed": 150.0,
                    "write_speed": 100.0,
                    "utilization": 0.30
                },
                "network": {
                    "download_speed": 50.0,
                    "upload_speed": 20.0,
                    "latency": 5.0
                },
                "training": {
                    "samples_per_second": 256.0,
                    "tokens_per_second": 8192.0,
                    "batch_size": 32,
                    "gradient_norm": 0.5
                },
                "timestamp": "2025-01-09T10:00:00Z"
            },
            "message": "获取资源使用情况成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_progress_service()
        result = service.get_resource_usage(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        return success_response(result, "获取资源使用情况成功")
        
    except Exception as e:
        logger.error(f"Error getting resource usage: {e}")
        return error_response(f"获取资源使用情况失败: {str(e)}", 500)


# ==================== 进度摘要 ====================

@training_progress_bp.route('/<session_id>/summary', methods=['GET'])
@jwt_required()
def get_progress_summary(session_id: str):
    """
    获取进度摘要（整合多个信息）
    
    Returns:
        {
            "success": true,
            "data": {
                "progress": {...},
                "metrics_summary": {...},
                "resource_usage": {...},
                "checkpoints_count": 5,
                "events_count": 20
            },
            "message": "获取进度摘要成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_progress_service()
        
        # 获取各项数据
        progress = service.get_progress(session_id, user_id, tenant_id) or {}
        metrics_summary = service.get_metric_summary(session_id, user_id, tenant_id) or {}
        resource_usage = service.get_resource_usage(session_id, user_id, tenant_id) or {}
        checkpoints = service.get_checkpoints(session_id, user_id, tenant_id) or {}
        events = service.get_events(session_id, user_id, tenant_id) or {}
        
        summary = {
            'progress': progress,
            'metrics_summary': metrics_summary,
            'resource_usage': resource_usage,
            'checkpoints_count': len(checkpoints.get('checkpoints', [])),
            'events_count': len(events.get('events', []))
        }
        
        return success_response(summary, "获取进度摘要成功")
        
    except Exception as e:
        logger.error(f"Error getting progress summary: {e}")
        return error_response(f"获取进度摘要失败: {str(e)}", 500)


# ==================== 实时进度（WebSocket 支持） ====================

@training_progress_bp.route('/<session_id>/realtime', methods=['GET'])
@jwt_required()
def get_realtime_progress(session_id: str):
    """
    获取实时进度（用于轮询或 WebSocket 初始化）
    
    优先从进度管理器获取实时数据，否则从数据库获取。
    
    Returns:
        与 get_training_progress 相同的格式
    """
    try:
        user_id = _get_current_user()
        
        service = get_training_progress_service()
        progress = service.get_realtime_progress(
            session_id=session_id,
            user_id=user_id
        )
        
        if not progress:
            return error_response("训练会话不存在", 404)
        
        return success_response(progress, "获取实时进度成功")
        
    except Exception as e:
        logger.error(f"Error getting realtime progress: {e}")
        return error_response(f"获取实时进度失败: {str(e)}", 500)


# ==================== 健康检查 ====================

@training_progress_bp.route('/health', methods=['GET'])
def health_check():
    """API健康检查"""
    return success_response({
        'status': 'healthy',
        'service': 'training_progress',
        'timestamp': datetime.utcnow().isoformat()
    }, "Service is healthy")

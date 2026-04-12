# -*- coding: utf-8 -*-
"""训练历史API接口

提供训练历史记录相关的API接口，支持前端训练历史页面显示。

API端点：
- GET  /api/v1/training/history          - 获取训练历史列表（分页、过滤）
- GET  /api/v1/training/history/<id>     - 获取训练详情
- POST /api/v1/training/history/<id>/download - 获取模型下载链接
- POST /api/v1/training/history/<id>/restart  - 重新开始训练
- DELETE /api/v1/training/history/<id>   - 删除训练记录
- GET  /api/v1/training/history/statistics - 获取训练统计
- GET  /api/v1/training/history/<id>/logs - 获取训练日志
- POST /api/v1/training/history/compare   - 比较多个训练会话

架构：
API层 -> Service层 -> Repository层 -> Database
              ↘-> Training模块 (重新训练时)
"""

import logging
from flask import Blueprint, request

logger = logging.getLogger(__name__)

# 异常类
try:
    from backend.modules.training.exceptions import ValidationError, BusinessLogicError
except ImportError:
    try:
        from backend.core.exceptions import ValidationError, BusinessLogicError
    except ImportError:
        class ValidationError(Exception):
            pass
        class BusinessLogicError(Exception):
            pass

# 响应工具
try:
    from backend.utils.response import success_response, error_response, paginated_response
except ImportError:
    # Fallback响应函数
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

# 服务层
from backend.services.training_history_service import get_training_history_service


# 创建蓝图
training_history_bp = Blueprint('training_history', __name__, url_prefix='/api/v1/training/history')


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
    # 优先从请求头获取
    tenant_id = request.headers.get('X-Tenant-ID')
    if tenant_id:
        return tenant_id
    
    # 尝试从JWT获取
    if JWT_AVAILABLE:
        identity = get_jwt_identity()
        if isinstance(identity, dict):
            return identity.get('tenant_id', 'default')
    
    return 'default'


# ==================== API端点 ====================

@training_history_bp.route('', methods=['GET'])
@jwt_required()
def get_training_history():
    """
    获取训练历史记录（支持分页和筛选）
    
    Query Parameters:
        - page: 页码 (默认: 1)
        - limit: 每页数量 (默认: 10, 最大: 50)
        - status: 状态过滤 (completed, failed, cancelled, running, pending)
        - training_type: 训练类型过滤 (PT, SFT, DPO, standard, distributed, multimodal, distillation)
        - model_name: 模型名称搜索（模糊匹配）
        - start_date: 开始日期 (YYYY-MM-DD)
        - end_date: 结束日期 (YYYY-MM-DD)
    
    Returns:
        {
            "success": true,
            "data": [
                {
                    "id": "uuid",
                    "name": "Training Task Name",
                    "status": "completed|failed|cancelled|running|pending",
                    "trainingType": "PT|SFT|DPO|standard|...",
                    "modelName": "model_name",
                    "startTime": "2025-01-09T10:00:00Z",
                    "endTime": "2025-01-09T12:00:00Z",
                    "duration": 7200,
                    "finalLoss": 0.123,
                    "accuracy": 0.95,
                    "epochs": 10,
                    "datasetSize": 10000,
                    "outputPath": "/path/to/output",
                    "configPath": "/path/to/config.json",
                    "logPath": "/path/to/training.log"
                }
            ],
            "total": 100,
            "page": 1,
            "limit": 10,
            "message": "获取训练历史记录成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        # 获取查询参数
        page = request.args.get('page', 1, type=int)
        limit = request.args.get('limit', 10, type=int)
        status = request.args.get('status', None)
        training_type = request.args.get('training_type', None)
        model_name = request.args.get('model_name', None)
        start_date = request.args.get('start_date', None)
        end_date = request.args.get('end_date', None)
        
        # 参数验证
        if status and status not in ['completed', 'failed', 'cancelled', 'running', 'pending']:
            return error_response(f"无效的状态值: {status}", 400)
        
        # 调用服务
        service = get_training_history_service()
        result = service.get_training_history(
            user_id=user_id,
            page=page,
            limit=limit,
            status=status,
            training_type=training_type,
            model_name=model_name,
            start_date=start_date,
            end_date=end_date,
            _tenant_id=tenant_id
        )
        
        return paginated_response(
            items=result['sessions'],
            page=result['page'],
            limit=result['limit'],
            total=result['total'],
            message="获取训练历史记录成功"
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in get_training_history: {e}")
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        logger.error(f"Business logic error in get_training_history: {e}")
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"Unexpected error in get_training_history: {e}")
        return error_response(f"获取训练历史记录失败: {str(e)}", 500)


@training_history_bp.route('/<session_id>', methods=['GET'])
@jwt_required()
def get_training_history_detail(session_id: str):
    """
    获取训练历史记录详情
    
    Path Parameters:
        - session_id: 训练会话ID
    
    Returns:
        {
            "success": true,
            "data": {
                "id": "uuid",
                "name": "Training Task Name",
                "status": "completed",
                "trainingType": "SFT",
                "modelName": "llama-7b",
                "startTime": "2025-01-09T10:00:00Z",
                "endTime": "2025-01-09T12:00:00Z",
                "duration": 7200,
                "finalLoss": 0.123,
                "accuracy": 0.95,
                "epochs": 10,
                "datasetSize": 10000,
                "outputPath": "/path/to/output",
                "configPath": "/path/to/config.json",
                "logPath": "/path/to/training.log",
                "config": {
                    "learning_rate": 1e-4,
                    "batch_size": 32,
                    ...
                },
                "result": {
                    "final_loss": 0.123,
                    "best_epoch": 8,
                    ...
                },
                "errorMessage": null,
                "progress": 100.0,
                "datasetId": "dataset_uuid"
            },
            "message": "获取训练记录详情成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_history_service()
        session_data = service.get_training_history_detail(
            session_id=session_id,
            user_id=user_id,
            _tenant_id=tenant_id
        )
        
        if not session_data:
            return error_response("训练记录不存在", 404)
        
        return success_response(session_data, "获取训练记录详情成功")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"Unexpected error in get_training_history_detail: {e}")
        return error_response(f"获取训练记录详情失败: {str(e)}", 500)


@training_history_bp.route('/<session_id>/download', methods=['POST'])
@jwt_required()
def download_training_model(session_id: str):
    """
    获取训练模型下载链接
    
    仅支持已完成的训练任务。
    
    Path Parameters:
        - session_id: 训练会话ID
    
    Returns:
        {
            "success": true,
            "data": {
                "downloadUrl": "/api/v1/training/models/download?session_id=xxx"
            },
            "message": "获取下载链接成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_history_service()
        download_url = service.download_training_model(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        if not download_url:
            return error_response("训练记录不存在", 404)
        
        return success_response({
            'downloadUrl': download_url
        }, "获取下载链接成功")
        
    except BusinessLogicError as e:
        # 区分不同的业务错误
        error_msg = str(e)
        if "只有已完成" in error_msg:
            return error_response(error_msg, 400)
        elif "模型路径不存在" in error_msg:
            return error_response(error_msg, 404)
        return error_response(error_msg, 500)
    except Exception as e:
        logger.error(f"Unexpected error in download_training_model: {e}")
        return error_response(f"获取下载链接失败: {str(e)}", 500)


@training_history_bp.route('/<session_id>/restart', methods=['POST'])
@jwt_required()
def restart_training(session_id: str):
    """
    重新开始训练
    
    基于历史训练配置创建新的训练任务。可以通过请求体覆盖部分配置。
    
    Path Parameters:
        - session_id: 原训练会话ID
    
    Request Body (optional):
        {
            "epochs": 20,
            "learning_rate": 1e-5,
            ...
        }
    
    Returns:
        {
            "success": true,
            "data": {
                "newSessionId": "new-uuid"
            },
            "message": "重新开始训练成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        # 获取配置覆盖
        config_overrides = None
        if request.is_json:
            config_overrides = request.get_json(silent=True)
        
        service = get_training_history_service()
        new_session_id = service.restart_training(
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            config_overrides=config_overrides
        )
        
        return success_response({
            'newSessionId': new_session_id
        }, "重新开始训练成功", 201)
        
    except ValidationError as e:
        return error_response(str(e), 404)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"Unexpected error in restart_training: {e}")
        return error_response(f"重新开始训练失败: {str(e)}", 500)


@training_history_bp.route('/<session_id>', methods=['DELETE'])
@jwt_required()
def delete_training_record(session_id: str):
    """
    删除训练记录
    
    Path Parameters:
        - session_id: 训练会话ID
    
    Returns:
        {
            "success": true,
            "data": {
                "deleted": true
            },
            "message": "删除训练记录成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_history_service()
        success = service.delete_training_record(
            session_id=session_id,
            user_id=user_id,
            _tenant_id=tenant_id
        )
        
        if not success:
            return error_response("训练记录不存在", 404)
        
        return success_response({
            'deleted': True
        }, "删除训练记录成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"Unexpected error in delete_training_record: {e}")
        return error_response(f"删除训练记录失败: {str(e)}", 500)


@training_history_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_training_history_statistics():
    """
    获取训练历史统计信息
    
    Returns:
        {
            "success": true,
            "data": {
                "totalTrainings": 150,
                "completedTrainings": 120,
                "failedTrainings": 20,
                "cancelledTrainings": 10,
                "averageTrainingTime": 45.5,
                "averageAccuracy": 0.92,
                "averageLoss": 0.15,
                "mostUsedModel": "llama-7b",
                "mostUsedTrainingType": "SFT"
            },
            "message": "获取训练统计信息成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_history_service()
        statistics_data = service.get_training_statistics(
            user_id=user_id,
            _tenant_id=tenant_id
        )
        
        return success_response(statistics_data, "获取训练统计信息成功")
        
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"Unexpected error in get_training_history_statistics: {e}")
        return error_response(f"获取训练统计信息失败: {str(e)}", 500)


@training_history_bp.route('/<session_id>/logs', methods=['GET'])
@jwt_required()
def get_training_logs(session_id: str):
    """
    获取训练日志
    
    Path Parameters:
        - session_id: 训练会话ID
    
    Query Parameters:
        - lines: 返回日志行数 (默认: 100, 最大: 1000)
    
    Returns:
        {
            "success": true,
            "data": {
                "logs": ["line1", "line2", ...],
                "totalLines": 100
            },
            "message": "获取训练日志成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        lines = request.args.get('lines', 100, type=int)
        lines = min(max(lines, 1), 1000)  # 限制范围
        
        service = get_training_history_service()
        log_lines = service.get_training_logs(
            session_id=session_id,
            user_id=user_id,
            lines=lines,
            tenant_id=tenant_id
        )
        
        return success_response({
            'logs': log_lines,
            'totalLines': len(log_lines)
        }, "获取训练日志成功")
        
    except Exception as e:
        logger.error(f"Unexpected error in get_training_logs: {e}")
        return error_response(f"获取训练日志失败: {str(e)}", 500)


@training_history_bp.route('/compare', methods=['POST'])
@jwt_required()
def compare_training_sessions():
    """
    比较多个训练会话
    
    Request Body:
        {
            "sessionIds": ["uuid1", "uuid2", "uuid3"]
        }
    
    Returns:
        {
            "success": true,
            "data": {
                "sessions": [...],
                "comparison": {
                    "loss": [0.1, 0.15, 0.12],
                    "accuracy": [0.95, 0.92, 0.94],
                    "duration": [7200, 8000, 7500],
                    "epochs": [10, 12, 11]
                }
            },
            "message": "比较训练会话成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        data = request.get_json()
        if not data:
            return error_response("请求体不能为空", 400)
        
        session_ids = data.get('sessionIds', [])
        if not session_ids or not isinstance(session_ids, list):
            return error_response("sessionIds必须是非空数组", 400)
        
        if len(session_ids) < 2:
            return error_response("至少需要2个会话进行比较", 400)
        
        service = get_training_history_service()
        result = service.compare_training_sessions(
            session_ids=session_ids,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        return success_response(result, "比较训练会话成功")
        
    except Exception as e:
        logger.error(f"Unexpected error in compare_training_sessions: {e}")
        return error_response(f"比较训练会话失败: {str(e)}", 500)


# ==================== 健康检查 ====================

@training_history_bp.route('/health', methods=['GET'])
def health_check():
    """API健康检查"""
    return success_response({
        'status': 'healthy',
        'service': 'training_history'
    }, "Service is healthy")

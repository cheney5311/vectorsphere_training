# -*- coding: utf-8 -*-
"""训练任务API接口

提供训练任务管理的完整 RESTful API。

API端点：
- POST   /api/v1/training/jobs           - 创建训练任务
- GET    /api/v1/training/jobs           - 获取任务列表
- GET    /api/v1/training/jobs/<id>      - 获取任务详情
- POST   /api/v1/training/jobs/<id>/start  - 开始任务
- POST   /api/v1/training/jobs/<id>/pause  - 暂停任务
- POST   /api/v1/training/jobs/<id>/resume - 恢复任务
- POST   /api/v1/training/jobs/<id>/cancel - 取消任务
- DELETE /api/v1/training/jobs/<id>      - 删除任务
- GET    /api/v1/training/jobs/<id>/logs - 获取任务日志
- GET    /api/v1/training/jobs/<id>/metrics - 获取任务指标
- GET    /api/v1/training/statistics     - 获取统计信息

架构：
API层 -> Service层 -> Repository层 -> Database
              ↘-> Training模块 (backend/modules/training)
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

# 异常类
from backend.modules.training.exceptions import (
    ValidationError, BusinessLogicError, TrainingError,
    TrainingJobNotFoundException
)

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
from backend.services.training_jobs_service import get_training_jobs_service


# 创建蓝图
training_jobs_bp = Blueprint('training_jobs', __name__, url_prefix='/api/v1/training')


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


# ==================== 训练任务 CRUD ====================

@training_jobs_bp.route('/jobs', methods=['POST'])
@jwt_required()
def create_training_job():
    """
    创建训练任务
    
    Request Body:
        {
            "model_name": "string",
            "scenario_type": "string",
            "name": "string",
            "description": "string",
            "training_method": "string",
            "output_dir": "string",
            "base_model_path": "string",
            "use_distributed": "boolean",
            "enable_wandb": "boolean",
            "device": "string",
            "pretrain": {
                "enabled": "boolean",
                "data_path": "string",
                "num_epochs": "integer",
                "batch_size": "integer",
                "learning_rate": "number"
            },
            "finetune": {
                "enabled": "boolean",
                "data_path": "string",
                "num_epochs": "integer",
                "batch_size": "integer",
                "learning_rate": "number"
            },
            "preference": {
                "enabled": "boolean",
                "data_path": "string",
                "num_epochs": "integer",
                "batch_size": "integer",
                "learning_rate": "number"
            },
            "schedule": {
                "type": "string",
                "start_time": "string",
                "priority": "string",
                "max_concurrent_jobs": "integer"
            }
        }
    
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "uuid",
                "name": "任务名称",
                "status": "pending",
                "created_at": "2025-01-09T10:00:00Z"
            },
            "message": "训练任务创建成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        # 验证必需字段
        required_fields = ['model_name', 'scenario_type']
        for field in required_fields:
            if field not in data:
                return error_response(f"缺少必需字段: {field}", 400)
        
        # 验证场景类型
        valid_scenarios = ['standard', 'distributed', 'multimodal', 'distillation', 
                          'three_stage', 'industry', 'scenario']
        if data.get('scenario_type') not in valid_scenarios:
            logger.warning(f"Unknown scenario_type: {data.get('scenario_type')}, using 'standard'")
        
        # 创建任务
        service = get_training_jobs_service()
        job_info = service.create_job(
            user_id=user_id,
            tenant_id=tenant_id,
            job_config=data
        )
        
        # 兼容旧格式返回
        response_data = job_info.copy() if isinstance(job_info, dict) else {'job_id': str(job_info)}
        
        if 'job_id' not in response_data and 'id' in response_data:
            response_data['job_id'] = response_data['id']
        
        # 直接返回数据（兼容测试脚本）
        return jsonify(response_data), 201
        
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        logger.error(f"Business logic error: {e}")
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"Unexpected error creating job: {e}")
        return error_response(f"创建训练任务失败: {str(e)}", 500)


@training_jobs_bp.route('/jobs', methods=['GET'])
@jwt_required()
def list_training_jobs():
    """
    获取训练任务列表
    
    Query Parameters:
        - status: 状态过滤 (pending, running, completed, failed, cancelled, paused)
        - scenario_type: 场景类型过滤
        - page: 页码 (默认: 1)
        - per_page: 每页数量 (默认: 20, 最大: 100)
        - sort_by: 排序字段 (created_at, updated_at, status)
        - sort_order: 排序方向 (asc, desc)
    
    Returns:
        {
            "success": true,
            "data": [...],
            "page": 1,
            "limit": 20,
            "total": 100,
            "message": "获取训练任务列表成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        # 获取查询参数
        status = request.args.get('status')
        scenario_type = request.args.get('scenario_type')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        sort_by = request.args.get('sort_by', 'created_at')
        sort_order = request.args.get('sort_order', 'desc')
        
        # 限制分页参数
        page = max(page, 1)
        per_page = min(max(per_page, 1), 100)
        
        service = get_training_jobs_service()
        result = service.list_jobs(
            user_id=user_id,
            tenant_id=tenant_id,
            status=status,
            scenario_type=scenario_type,
            page=page,
            per_page=per_page,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        return paginated_response(
            items=result.get('jobs', []),
            page=result.get('page', page),
            limit=result.get('per_page', per_page),
            total=result.get('total', 0),
            message="获取训练任务列表成功"
        )
        
    except Exception as e:
        logger.error(f"Error listing jobs: {e}")
        return error_response(f"获取训练任务列表失败: {str(e)}", 500)


@training_jobs_bp.route('/jobs/<job_id>', methods=['GET'])
@jwt_required()
def get_training_job(job_id: str):
    """
    获取训练任务详情
    
    Path Parameters:
        - job_id: 任务ID
    
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "uuid",
                "name": "任务名称",
                "status": "running",
                "progress": 45.5,
                "current_epoch": 5,
                "total_epochs": 10,
                "metrics": {...},
                "config": {...},
                "result": {...},
                "created_at": "...",
                "started_at": "...",
                "error_message": null
            },
            "message": "获取训练任务详情成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_jobs_service()
        job_detail = service.get_job_detail(
            job_id=job_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        if not job_detail:
            return error_response("训练任务不存在", 404)
        
        return success_response(job_detail, "获取训练任务详情成功")
        
    except TrainingJobNotFoundException:
        return error_response("训练任务不存在", 404)
    except Exception as e:
        logger.error(f"Error getting job {job_id}: {e}")
        return error_response(f"获取训练任务详情失败: {str(e)}", 500)


@training_jobs_bp.route('/jobs/<job_id>', methods=['DELETE'])
@jwt_required()
def delete_training_job(job_id: str):
    """
    删除训练任务
    
    只能删除已完成、失败或取消的任务。运行中的任务需要先取消。
    
    Returns:
        {
            "success": true,
            "data": {"deleted": true},
            "message": "训练任务删除成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_jobs_service()
        result = service.delete_job(
            job_id=job_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        if not result.get('success'):
            return error_response(result.get('message', '删除失败'), 400)
        
        return success_response({'deleted': True}, "训练任务删除成功")
        
    except TrainingJobNotFoundException:
        return error_response("训练任务不存在", 404)
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error deleting job {job_id}: {e}")
        return error_response(f"删除训练任务失败: {str(e)}", 500)


# ==================== 任务控制 ====================

@training_jobs_bp.route('/jobs/<job_id>/start', methods=['POST'])
@jwt_required()
def start_training_job(job_id: str):
    """
    开始训练任务
    
    将待处理的任务提交到训练队列开始执行。
    
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "uuid",
                "status": "running",
                "started_at": "..."
            },
            "message": "训练任务已开始"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_jobs_service()
        result = service.start_job(
            job_id=job_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        if not result.get('success'):
            return error_response(result.get('message', '无法启动任务'), 400)
        
        return success_response(result, "训练任务已开始")
        
    except TrainingJobNotFoundException:
        return error_response("训练任务不存在", 404)
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error starting job {job_id}: {e}")
        return error_response(f"开始训练任务失败: {str(e)}", 500)


@training_jobs_bp.route('/jobs/<job_id>/pause', methods=['POST'])
@jwt_required()
def pause_training_job(job_id: str):
    """
    暂停训练任务
    
    保存当前进度和检查点，暂停训练执行。
    
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "uuid",
                "status": "paused",
                "checkpoint_path": "..."
            },
            "message": "训练任务已暂停"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_jobs_service()
        result = service.pause_job(
            job_id=job_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        if not result.get('success'):
            return error_response(result.get('message', '无法暂停任务'), 400)
        
        return success_response(result, "训练任务已暂停")
        
    except TrainingJobNotFoundException:
        return error_response("训练任务不存在", 404)
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error pausing job {job_id}: {e}")
        return error_response(f"暂停训练任务失败: {str(e)}", 500)


@training_jobs_bp.route('/jobs/<job_id>/resume', methods=['POST'])
@jwt_required()
def resume_training_job(job_id: str):
    """
    恢复训练任务
    
    从检查点恢复暂停的训练任务。
    
    Request Body (optional):
        {
            "checkpoint_path": "指定检查点路径"
        }
    
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "uuid",
                "status": "running",
                "resumed_from": "checkpoint_path"
            },
            "message": "训练任务已恢复"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        # 获取可选的检查点路径
        data = request.get_json(silent=True) or {}
        checkpoint_path = data.get('checkpoint_path')
        
        service = get_training_jobs_service()
        result = service.resume_job(
            job_id=job_id,
            user_id=user_id,
            tenant_id=tenant_id,
            checkpoint_path=checkpoint_path
        )
        
        if not result.get('success'):
            return error_response(result.get('message', '无法恢复任务'), 400)
        
        return success_response(result, "训练任务已恢复")
        
    except TrainingJobNotFoundException:
        return error_response("训练任务不存在", 404)
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error resuming job {job_id}: {e}")
        return error_response(f"恢复训练任务失败: {str(e)}", 500)


@training_jobs_bp.route('/jobs/<job_id>/cancel', methods=['POST'])
@jwt_required()
def cancel_training_job(job_id: str):
    """
    取消训练任务
    
    停止训练并释放资源。取消后的任务不能恢复。
    
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "uuid",
                "status": "cancelled"
            },
            "message": "训练任务已取消"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_jobs_service()
        result = service.cancel_job(
            job_id=job_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        if not result.get('success'):
            return error_response(result.get('message', '无法取消任务'), 400)
        
        return success_response(result, "训练任务已取消")
        
    except TrainingJobNotFoundException:
        return error_response("训练任务不存在", 404)
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error cancelling job {job_id}: {e}")
        return error_response(f"取消训练任务失败: {str(e)}", 500)


@training_jobs_bp.route('/jobs/<job_id>/restart', methods=['POST'])
@jwt_required()
def restart_training_job(job_id: str):
    """
    重启训练任务
    
    从头开始重新训练，或从指定检查点重新开始。
    
    Request Body (optional):
        {
            "from_checkpoint": false,
            "config_overrides": {...}
        }
    
    Returns:
        {
            "success": true,
            "data": {
                "new_job_id": "uuid",
                "status": "pending"
            },
            "message": "训练任务重启成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        data = request.get_json(silent=True) or {}
        from_checkpoint = data.get('from_checkpoint', False)
        config_overrides = data.get('config_overrides')
        
        service = get_training_jobs_service()
        result = service.restart_job(
            job_id=job_id,
            user_id=user_id,
            tenant_id=tenant_id,
            from_checkpoint=from_checkpoint,
            config_overrides=config_overrides
        )
        
        if not result.get('success'):
            return error_response(result.get('message', '无法重启任务'), 400)
        
        return success_response(result, "训练任务重启成功", 201)
        
    except TrainingJobNotFoundException:
        return error_response("训练任务不存在", 404)
    except BusinessLogicError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Error restarting job {job_id}: {e}")
        return error_response(f"重启训练任务失败: {str(e)}", 500)


# ==================== 日志和指标 ====================

@training_jobs_bp.route('/jobs/<job_id>/logs', methods=['GET'])
@jwt_required()
def get_job_logs(job_id: str):
    """
    获取训练任务日志
    
    Query Parameters:
        - log_type: 日志类型 (status_change, progress, error, checkpoint, metric)
        - log_level: 日志级别 (debug, info, warning, error)
        - limit: 返回数量 (默认: 100)
        - offset: 偏移量
    
    Returns:
        {
            "success": true,
            "data": {
                "logs": [...],
                "total": 500
            },
            "message": "获取日志成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        log_type = request.args.get('log_type')
        log_level = request.args.get('log_level')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = get_training_jobs_service()
        result = service.get_job_logs(
            job_id=job_id,
            user_id=user_id,
            tenant_id=tenant_id,
            log_type=log_type,
            log_level=log_level,
            limit=limit,
            offset=offset
        )
        
        return success_response(result, "获取日志成功")
        
    except TrainingJobNotFoundException:
        return error_response("训练任务不存在", 404)
    except Exception as e:
        logger.error(f"Error getting logs for job {job_id}: {e}")
        return error_response(f"获取日志失败: {str(e)}", 500)


@training_jobs_bp.route('/jobs/<job_id>/metrics', methods=['GET'])
@jwt_required()
def get_job_metrics(job_id: str):
    """
    获取训练任务指标
    
    Query Parameters:
        - metric_names: 指标名称（逗号分隔）
        - start_epoch: 开始轮次
        - end_epoch: 结束轮次
    
    Returns:
        {
            "success": true,
            "data": {
                "current_metrics": {
                    "loss": 0.15,
                    "accuracy": 0.92
                },
                "history": [
                    {"epoch": 1, "loss": 0.5, "accuracy": 0.7},
                    ...
                ]
            },
            "message": "获取指标成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        metric_names = request.args.get('metric_names')
        start_epoch = request.args.get('start_epoch', type=int)
        end_epoch = request.args.get('end_epoch', type=int)
        
        service = get_training_jobs_service()
        result = service.get_job_metrics(
            job_id=job_id,
            user_id=user_id,
            tenant_id=tenant_id,
            metric_names=metric_names.split(',') if metric_names else None,
            start_epoch=start_epoch,
            end_epoch=end_epoch
        )
        
        return success_response(result, "获取指标成功")
        
    except TrainingJobNotFoundException:
        return error_response("训练任务不存在", 404)
    except Exception as e:
        logger.error(f"Error getting metrics for job {job_id}: {e}")
        return error_response(f"获取指标失败: {str(e)}", 500)


@training_jobs_bp.route('/jobs/<job_id>/checkpoints', methods=['GET'])
@jwt_required()
def get_job_checkpoints(job_id: str):
    """
    获取训练任务检查点列表
    
    Returns:
        {
            "success": true,
            "data": {
                "checkpoints": [
                    {
                        "path": "...",
                        "epoch": 5,
                        "step": 1000,
                        "created_at": "...",
                        "metrics": {...}
                    }
                ]
            },
            "message": "获取检查点列表成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_jobs_service()
        result = service.get_job_checkpoints(
            job_id=job_id,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        return success_response(result, "获取检查点列表成功")
        
    except TrainingJobNotFoundException:
        return error_response("训练任务不存在", 404)
    except Exception as e:
        logger.error(f"Error getting checkpoints for job {job_id}: {e}")
        return error_response(f"获取检查点失败: {str(e)}", 500)


# ==================== 统计信息 ====================

@training_jobs_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_training_statistics():
    """
    获取训练统计信息
    
    Returns:
        {
            "success": true,
            "data": {
                "total_jobs": 150,
                "running_jobs": 5,
                "pending_jobs": 10,
                "completed_jobs": 120,
                "failed_jobs": 10,
                "cancelled_jobs": 5,
                "average_duration": 3600,
                "success_rate": 0.92,
                "resource_usage": {
                    "gpu_utilization": 0.75,
                    "memory_utilization": 0.60
                },
                "recent_jobs": [...]
            },
            "message": "获取训练统计信息成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_jobs_service()
        stats = service.get_statistics(
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        return success_response(stats, "获取训练统计信息成功")
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return error_response(f"获取训练统计信息失败: {str(e)}", 500)


@training_jobs_bp.route('/statistics/by-scenario', methods=['GET'])
@jwt_required()
def get_statistics_by_scenario():
    """
    按场景类型获取训练统计
    
    Returns:
        {
            "success": true,
            "data": {
                "standard": {"total": 50, "success_rate": 0.95, ...},
                "distributed": {"total": 30, "success_rate": 0.90, ...},
                ...
            },
            "message": "获取场景统计成功"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        service = get_training_jobs_service()
        stats = service.get_statistics_by_scenario(
            _user_id=user_id,
            _tenant_id=tenant_id
        )
        
        return success_response(stats, "获取场景统计成功")
        
    except Exception as e:
        logger.error(f"Error getting scenario statistics: {e}")
        return error_response(f"获取场景统计失败: {str(e)}", 500)


# ==================== 批量操作 ====================

@training_jobs_bp.route('/jobs/batch/cancel', methods=['POST'])
@jwt_required()
def batch_cancel_jobs():
    """
    批量取消训练任务
    
    Request Body:
        {
            "job_ids": ["id1", "id2", ...]
        }
    
    Returns:
        {
            "success": true,
            "data": {
                "cancelled": ["id1", "id2"],
                "failed": [{"id": "id3", "reason": "..."}]
            },
            "message": "批量取消完成"
        }
    """
    try:
        user_id = _get_current_user()
        tenant_id = _get_tenant_id()
        
        data = request.get_json()
        if not data or not data.get('job_ids'):
            return error_response("请提供要取消的任务ID列表", 400)
        
        job_ids = data['job_ids']
        if not isinstance(job_ids, list):
            return error_response("job_ids必须是数组", 400)
        
        service = get_training_jobs_service()
        result = service.batch_cancel_jobs(
            job_ids=job_ids,
            user_id=user_id,
            tenant_id=tenant_id
        )
        
        return success_response(result, "批量取消完成")
        
    except Exception as e:
        logger.error(f"Error batch cancelling jobs: {e}")
        return error_response(f"批量取消失败: {str(e)}", 500)


# ==================== 健康检查 ====================

@training_jobs_bp.route('/health', methods=['GET'])
def health_check():
    """API健康检查"""
    return success_response({
        'status': 'healthy',
        'service': 'training_jobs',
        'timestamp': datetime.utcnow().isoformat()
    }, "Service is healthy")

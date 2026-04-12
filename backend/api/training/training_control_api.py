"""训练任务控制API接口

提供训练任务控制相关的API接口，支持前端对训练任务的控制操作。
包括创建、启动、暂停、恢复、取消、重启等功能。

功能特性：
- 完整的训练任务生命周期管理
- 租户隔离的数据访问
- 实时进度跟踪
- 详细的操作日志
"""

import sys
import os
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError, BusinessLogicError, ResourceNotFoundError
from backend.utils.response import success_response, error_response
from backend.services.training_control_service import (
    get_training_control_service,
    TrainingJobConfig
)

# 创建蓝图
training_control_bp = Blueprint('training_control', __name__, url_prefix='/api/v1/training/control')


def _get_tenant_id() -> str:
    """获取当前租户ID
    
    Returns:
        租户ID
    """
    # 从请求头或JWT中获取租户ID
    tenant_id = request.headers.get('X-Tenant-ID')
    if not tenant_id:
        # 尝试从JWT中获取
        try:
            identity = get_jwt_identity()
            if isinstance(identity, dict):
                tenant_id = identity.get('tenant_id')
            else:
                tenant_id = 'default'
        except Exception:
            tenant_id = 'default'
    return tenant_id


def _get_user_id() -> str:
    """获取当前用户ID
    
    Returns:
        用户ID
    """
    try:
        identity = get_jwt_identity()
        if isinstance(identity, dict):
            return identity.get('user_id', identity.get('sub', str(identity)))
        return str(identity)
    except Exception:
        return 'anonymous'


# ==================== 任务创建 ====================

@training_control_bp.route('/jobs', methods=['POST'])
@jwt_required()
def create_training_job():
    """创建训练任务
    
    请求体:
        {
            "name": "任务名称",
            "description": "任务描述",
            "scenario_type": "训练场景类型",
            "training_mode": "训练模式(standard/distributed/multimodal等)",
            "model_id": "模型ID",
            "model_name": "模型名称",
            "dataset_id": "数据集ID",
            "priority": 5,
            "config": {...},
            "resource_config": {...},
            "tags": ["tag1", "tag2"]
        }
        
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "任务ID",
                "status": "pending",
                "job": {...}
            },
            "message": "训练任务创建成功"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        data = request.get_json() or {}
        
        # 验证必需字段
        if not data.get('name'):
            return error_response("任务名称不能为空", 400)
        if not data.get('scenario_type'):
            return error_response("训练场景类型不能为空", 400)
        
        # 构建配置
        config = TrainingJobConfig(
            name=data.get('name'),
            scenario_type=data.get('scenario_type'),
            description=data.get('description', ''),
            training_mode=data.get('training_mode', 'standard'),
            model_id=data.get('model_id'),
            model_name=data.get('model_name'),
            dataset_id=data.get('dataset_id'),
            priority=data.get('priority', 5),
            config=data.get('config', {}),
            resource_config=data.get('resource_config', {}),
            tags=data.get('tags', [])
        )
        
        # 创建任务
        service = get_training_control_service()
        result = service.create_job(tenant_id, user_id, config)
        
        return success_response(result, "训练任务创建成功")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"创建训练任务失败: {str(e)}", 500)


# ==================== 任务控制 ====================

@training_control_bp.route('/jobs/<job_id>/start', methods=['POST'])
@jwt_required()
def start_training_job(job_id: str):
    """开始训练任务
    
    Args:
        job_id: 训练任务ID
        
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "任务ID",
                "status": "running",
                "message": "训练任务已开始"
            },
            "message": "训练任务已开始"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        service = get_training_control_service()
        result = service.start_job(job_id, tenant_id, user_id)
        
        return success_response(result, "训练任务已开始")
        
    except ResourceNotFoundError as e:
        return error_response(str(e), 404)
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"开始训练任务失败: {str(e)}", 500)


@training_control_bp.route('/jobs/<job_id>/pause', methods=['POST'])
@jwt_required()
def pause_training_job(job_id: str):
    """暂停训练任务
    
    Args:
        job_id: 训练任务ID
        
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "任务ID",
                "status": "paused",
                "message": "训练任务已暂停"
            },
            "message": "训练任务已暂停"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        service = get_training_control_service()
        result = service.pause_job(job_id, tenant_id, user_id)
        
        return success_response(result, "训练任务已暂停")
        
    except ResourceNotFoundError as e:
        return error_response(str(e), 404)
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"暂停训练任务失败: {str(e)}", 500)


@training_control_bp.route('/jobs/<job_id>/resume', methods=['POST'])
@jwt_required()
def resume_training_job(job_id: str):
    """恢复训练任务
    
    Args:
        job_id: 训练任务ID
        
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "任务ID",
                "status": "running",
                "message": "训练任务已恢复"
            },
            "message": "训练任务已恢复"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        service = get_training_control_service()
        result = service.resume_job(job_id, tenant_id, user_id)
        
        return success_response(result, "训练任务已恢复")
        
    except ResourceNotFoundError as e:
        return error_response(str(e), 404)
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"恢复训练任务失败: {str(e)}", 500)


@training_control_bp.route('/jobs/<job_id>/cancel', methods=['POST'])
@jwt_required()
def cancel_training_job(job_id: str):
    """取消训练任务
    
    Args:
        job_id: 训练任务ID
        
    请求体（可选）:
        {
            "reason": "取消原因"
        }
        
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "任务ID",
                "status": "cancelled",
                "message": "训练任务已取消"
            },
            "message": "训练任务已取消"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        data = request.get_json() or {}
        reason = data.get('reason')
        
        service = get_training_control_service()
        result = service.cancel_job(job_id, tenant_id, user_id, reason=reason)
        
        return success_response(result, "训练任务已取消")
        
    except ResourceNotFoundError as e:
        return error_response(str(e), 404)
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"取消训练任务失败: {str(e)}", 500)


@training_control_bp.route('/jobs/<job_id>/restart', methods=['POST'])
@jwt_required()
def restart_training_job(job_id: str):
    """重新开始训练任务
    
    Args:
        job_id: 训练任务ID
        
    请求体（可选）:
        {
            "from_checkpoint": true  // 是否从检查点恢复，默认true
        }
        
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "任务ID",
                "status": "running",
                "from_checkpoint": true,
                "message": "训练任务已重新开始"
            },
            "message": "训练任务已重新开始"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        data = request.get_json() or {}
        from_checkpoint = data.get('from_checkpoint', True)
        
        service = get_training_control_service()
        result = service.restart_job(job_id, tenant_id, user_id, 
                                     from_checkpoint=from_checkpoint)
        
        return success_response(result, "训练任务已重新开始")
        
    except ResourceNotFoundError as e:
        return error_response(str(e), 404)
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        return error_response(f"重新开始训练任务失败: {str(e)}", 500)


# ==================== 状态和进度查询 ====================

@training_control_bp.route('/jobs/<job_id>/status', methods=['GET'])
@jwt_required()
def get_training_job_status(job_id: str):
    """获取训练任务状态
    
    Args:
        job_id: 训练任务ID
        
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "任务ID",
                "name": "任务名称",
                "status": "任务状态",
                "progress": 进度百分比,
                "current_epoch": 当前轮次,
                "total_epochs": 总轮次,
                "current_step": 当前步骤,
                "total_steps": 总步骤,
                "metrics": {...},
                "started_at": "开始时间",
                "duration_seconds": 持续时间秒,
                "error_message": "错误信息"
            },
            "message": "获取任务状态成功"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = get_training_control_service()
        status = service.get_job_status(job_id, tenant_id)
        
        if not status:
            return error_response("训练任务不存在", 404)
        
        return success_response(status, "获取任务状态成功")
        
    except Exception as e:
        return error_response(f"获取任务状态失败: {str(e)}", 500)


@training_control_bp.route('/jobs/<job_id>/progress', methods=['GET'])
@jwt_required()
def get_training_job_progress(job_id: str):
    """获取训练任务进度
    
    Args:
        job_id: 训练任务ID
        
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "任务ID",
                "progress": 进度百分比,
                "current_epoch": 当前轮次,
                "total_epochs": 总轮次,
                "current_step": 当前步骤,
                "total_steps": 总步骤,
                "eta": 预计剩余时间秒,
                "metrics": {
                    "loss": 损失值,
                    "accuracy": 准确率,
                    "learning_rate": 学习率
                },
                "best_metrics": {...},
                "checkpoint_path": "检查点路径",
                "checkpoint_epoch": 检查点轮次
            },
            "message": "获取任务进度成功"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = get_training_control_service()
        progress = service.get_job_progress(job_id, tenant_id)
        
        if not progress:
            return error_response("训练任务不存在", 404)
        
        return success_response(progress, "获取任务进度成功")
        
    except Exception as e:
        return error_response(f"获取任务进度失败: {str(e)}", 500)


@training_control_bp.route('/jobs/<job_id>', methods=['GET'])
@jwt_required()
def get_training_job_info(job_id: str):
    """获取训练任务详细信息
    
    Args:
        job_id: 训练任务ID
        
    Returns:
        {
            "success": true,
            "data": {
                "job_id": "任务ID",
                "name": "任务名称",
                "description": "任务描述",
                "scenario_type": "场景类型",
                "training_mode": "训练模式",
                "status": "状态",
                "progress": 进度,
                "config": {...},
                "result": {...},
                ...
            },
            "message": "获取任务信息成功"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = get_training_control_service()
        job_info = service.get_job_info(job_id, tenant_id)
        
        if not job_info:
            return error_response("训练任务不存在", 404)
        
        return success_response(job_info, "获取任务信息成功")
        
    except Exception as e:
        return error_response(f"获取任务信息失败: {str(e)}", 500)


# ==================== 任务列表和统计 ====================

@training_control_bp.route('/jobs', methods=['GET'])
@jwt_required()
def list_training_jobs():
    """获取训练任务列表
    
    查询参数:
        - status: 状态过滤（pending/running/completed/failed/paused/cancelled）
        - scenario_type: 场景类型过滤
        - user_id: 用户ID过滤（仅管理员）
        - limit: 限制数量（默认100）
        - offset: 偏移量（默认0）
        
    Returns:
        {
            "success": true,
            "data": {
                "jobs": [...],
                "total": 总数,
                "limit": 限制,
                "offset": 偏移
            },
            "message": "获取任务列表成功"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        # 获取查询参数
        status = request.args.get('status')
        scenario_type = request.args.get('scenario_type')
        filter_user_id = request.args.get('user_id')
        limit = min(int(request.args.get('limit', 100)), 1000)
        offset = int(request.args.get('offset', 0))
        
        # 普通用户只能查看自己的任务
        if not filter_user_id:
            filter_user_id = user_id
        
        service = get_training_control_service()
        jobs = service.list_jobs(
            tenant_id=tenant_id,
            user_id=filter_user_id,
            status=status,
            scenario_type=scenario_type,
            limit=limit,
            offset=offset
        )
        
        return success_response({
            'jobs': jobs,
            'total': len(jobs),
            'limit': limit,
            'offset': offset
        }, "获取任务列表成功")
        
    except Exception as e:
        return error_response(f"获取任务列表失败: {str(e)}", 500)


@training_control_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_training_statistics():
    """获取训练统计信息
    
    查询参数:
        - user_id: 用户ID过滤（可选）
        
    Returns:
        {
            "success": true,
            "data": {
                "total_jobs": 总任务数,
                "pending_jobs": 待处理数,
                "running_jobs": 运行中数,
                "completed_jobs": 已完成数,
                "failed_jobs": 失败数,
                "paused_jobs": 暂停数,
                "cancelled_jobs": 取消数,
                "scheduler_running": 调度器是否运行,
                "max_concurrent_jobs": 最大并发数,
                "queue_size": 队列大小
            },
            "message": "获取训练统计信息成功"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = request.args.get('user_id')
        
        service = get_training_control_service()
        stats = service.get_statistics(tenant_id, user_id)
        
        return success_response(stats, "获取训练统计信息成功")
        
    except Exception as e:
        return error_response(f"获取训练统计信息失败: {str(e)}", 500)


# ==================== 日志查询 ====================

@training_control_bp.route('/jobs/<job_id>/logs', methods=['GET'])
@jwt_required()
def get_training_job_logs(job_id: str):
    """获取训练任务日志
    
    Args:
        job_id: 训练任务ID
        
    查询参数:
        - log_type: 日志类型过滤（status_change/progress/error/checkpoint/metric）
        - limit: 限制数量（默认100）
        
    Returns:
        {
            "success": true,
            "data": {
                "logs": [
                    {
                        "id": "日志ID",
                        "job_id": "任务ID",
                        "log_type": "日志类型",
                        "log_level": "日志级别",
                        "message": "日志消息",
                        "from_status": "变更前状态",
                        "to_status": "变更后状态",
                        "epoch": 轮次,
                        "step": 步骤,
                        "metrics": {...},
                        "created_at": "创建时间"
                    },
                    ...
                ]
            },
            "message": "获取任务日志成功"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        
        log_type = request.args.get('log_type')
        limit = min(int(request.args.get('limit', 100)), 1000)
        
        service = get_training_control_service()
        logs = service.get_job_logs(job_id, tenant_id, log_type=log_type, limit=limit)
        
        return success_response({'logs': logs}, "获取任务日志成功")
        
    except Exception as e:
        return error_response(f"获取任务日志失败: {str(e)}", 500)


# ==================== 批量操作 ====================

@training_control_bp.route('/jobs/batch/cancel', methods=['POST'])
@jwt_required()
def batch_cancel_jobs():
    """批量取消训练任务
    
    请求体:
        {
            "job_ids": ["job_id_1", "job_id_2", ...],
            "reason": "取消原因"
        }
        
    Returns:
        {
            "success": true,
            "data": {
                "cancelled": ["job_id_1", ...],
                "failed": [{"job_id": "job_id_2", "error": "错误信息"}, ...]
            },
            "message": "批量取消完成"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        data = request.get_json() or {}
        job_ids = data.get('job_ids', [])
        reason = data.get('reason', 'Batch cancellation')
        
        if not job_ids:
            return error_response("任务ID列表不能为空", 400)
        
        service = get_training_control_service()
        
        cancelled = []
        failed = []
        
        for job_id in job_ids:
            try:
                result = service.cancel_job(job_id, tenant_id, user_id, reason=reason)
                if result.get('success'):
                    cancelled.append(job_id)
                else:
                    failed.append({'job_id': job_id, 'error': result.get('message', 'Unknown error')})
            except Exception as e:
                failed.append({'job_id': job_id, 'error': str(e)})
        
        return success_response({
            'cancelled': cancelled,
            'failed': failed
        }, f"批量取消完成: 成功 {len(cancelled)}, 失败 {len(failed)}")
        
    except Exception as e:
        return error_response(f"批量取消失败: {str(e)}", 500)


@training_control_bp.route('/jobs/batch/start', methods=['POST'])
@jwt_required()
def batch_start_jobs():
    """批量启动训练任务
    
    请求体:
        {
            "job_ids": ["job_id_1", "job_id_2", ...]
        }
        
    Returns:
        {
            "success": true,
            "data": {
                "started": ["job_id_1", ...],
                "failed": [{"job_id": "job_id_2", "error": "错误信息"}, ...]
            },
            "message": "批量启动完成"
        }
    """
    try:
        tenant_id = _get_tenant_id()
        user_id = _get_user_id()
        
        data = request.get_json() or {}
        job_ids = data.get('job_ids', [])
        
        if not job_ids:
            return error_response("任务ID列表不能为空", 400)
        
        service = get_training_control_service()
        
        started = []
        failed = []
        
        for job_id in job_ids:
            try:
                result = service.start_job(job_id, tenant_id, user_id)
                if result.get('success'):
                    started.append(job_id)
                else:
                    failed.append({'job_id': job_id, 'error': result.get('message', 'Unknown error')})
            except Exception as e:
                failed.append({'job_id': job_id, 'error': str(e)})
        
        return success_response({
            'started': started,
            'failed': failed
        }, f"批量启动完成: 成功 {len(started)}, 失败 {len(failed)}")
        
    except Exception as e:
        return error_response(f"批量启动失败: {str(e)}", 500)


# ==================== 健康检查 ====================

@training_control_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查
    
    Returns:
        {
            "success": true,
            "data": {
                "status": "healthy",
                "service": "training_control",
                "scheduler_running": true
            },
            "message": "服务正常"
        }
    """
    try:
        service = get_training_control_service()
        stats = service.get_statistics('health_check')
        
        return success_response({
            'status': 'healthy',
            'service': 'training_control',
            'scheduler_running': stats.get('scheduler_running', True),
            'running_jobs': stats.get('running_jobs', 0)
        }, "服务正常")
        
    except Exception as e:
        return error_response(f"健康检查失败: {str(e)}", 500)

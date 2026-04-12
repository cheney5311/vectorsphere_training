"""工作流执行API接口

提供工作流执行相关的API接口，支持租户维度的数据隔离和持久化。
包括执行管理、步骤跟踪、日志查询、重试机制等功能。
"""

import sys
import os
import logging
from flask import Blueprint, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from backend.core.exceptions import ValidationError
except ImportError:
    from backend.core.exceptions import ValidationError

from backend.utils.response import success_response, error_response
from backend.utils.validation import validate_json, validate_required_fields

logger = logging.getLogger(__name__)

# 创建蓝图
workflow_execution_bp = Blueprint('workflow_execution', __name__, url_prefix='/api/v1/workflow-executions')


def _get_workflow_service():
    """获取工作流服务（延迟加载）"""
    try:
        from backend.services.workflow_service import get_workflow_service
        return get_workflow_service(use_memory_storage=True)  # 可改为 False 使用数据库
    except ImportError as e:
        logger.warning(f"Failed to import WorkflowService: {e}")
        return None


def _get_tenant_id(user_id: str) -> str:
    """获取用户的租户ID
    
    实际应用中应从用户会话或JWT中获取租户ID
    这里简化为从request header获取或使用默认值
    """
    tenant_id = request.headers.get('X-Tenant-ID')
    if not tenant_id:
        # 默认租户ID（实际应用中应该从用户信息获取）
        tenant_id = f"tenant_{user_id}"
    return tenant_id


# ============================================================================
# 执行管理接口
# ============================================================================

@workflow_execution_bp.route('', methods=['GET'])
@jwt_required()
def list_executions():
    """获取执行记录列表
    
    Query Parameters:
        - workflow_id: 工作流ID过滤
        - status: 状态过滤 (pending, running, completed, failed, cancelled)
        - triggered_by: 触发者过滤
        - start_date: 开始日期过滤 (YYYY-MM-DD)
        - end_date: 结束日期过滤 (YYYY-MM-DD)
        - page: 页码 (默认: 1)
        - limit: 每页数量 (默认: 20)
    
    Returns:
        {
            "executions": [...],
            "pagination": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        workflow_id = request.args.get('workflow_id')
        status = request.args.get('status')
        triggered_by = request.args.get('triggered_by')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        result = service.list_executions(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            status=status,
            triggered_by=triggered_by,
            page=page,
            page_size=limit
        )
        
        return success_response({
            "executions": result.get('items', []),
            "pagination": {
                "page": result.get('page', page),
                "limit": result.get('page_size', limit),
                "total": result.get('total', 0),
                "pages": result.get('total_pages', 0)
            }
        }, "获取执行记录列表成功")
        
    except Exception as e:
        logger.error(f"Failed to list executions: {e}")
        return error_response(f"获取执行记录列表失败: {str(e)}", 500)


@workflow_execution_bp.route('/<execution_id>', methods=['GET'])
@jwt_required()
def get_execution(execution_id):
    """获取执行记录详情
    
    Returns:
        {
            "execution": {
                "id": "string",
                "workflow_id": "string",
                "workflow_name": "string",
                "status": "string",
                "progress": "float",
                "current_step": "string",
                "steps": [...],
                "input_data": {},
                "output_data": {},
                "error_message": "string",
                "started_at": "string",
                "completed_at": "string",
                "duration_seconds": "float"
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        execution = service.get_execution(execution_id, tenant_id)
        if not execution:
            return error_response("执行记录不存在", 404)
        
        return success_response({
            "execution": execution
        }, "获取执行记录详情成功")
        
    except Exception as e:
        logger.error(f"Failed to get execution: {e}")
        return error_response(f"获取执行记录详情失败: {str(e)}", 500)


@workflow_execution_bp.route('/<execution_id>/cancel', methods=['POST'])
@jwt_required()
def cancel_execution(execution_id):
    """取消执行
    
    Request Body:
        {
            "reason": "string" (optional)
        }
    
    Returns:
        {
            "execution": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = validate_json(request) if request.is_json else {}
        reason = data.get('reason', '用户取消执行')
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        execution = service.cancel_execution(execution_id, tenant_id, user_id, reason)
        if not execution:
            return error_response("执行记录不存在", 404)
        
        return success_response({
            "execution": execution
        }, "执行已取消")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to cancel execution: {e}")
        return error_response(f"取消执行失败: {str(e)}", 500)


@workflow_execution_bp.route('/<execution_id>/retry', methods=['POST'])
@jwt_required()
def retry_execution(execution_id):
    """重试失败的执行
    
    Request Body:
        {
            "input_data": {} (optional, 使用原有输入数据)
        }
    
    Returns:
        {
            "execution": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = validate_json(request) if request.is_json else {}
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        # 获取原执行记录
        original = service.get_execution(execution_id, tenant_id)
        if not original:
            return error_response("执行记录不存在", 404)
        
        # 检查状态是否可重试
        if original.get('status') not in ('failed', 'cancelled', 'timeout'):
            return error_response("只能重试失败、取消或超时的执行", 400)
        
        # 使用新输入数据或原有输入数据
        input_data = data.get('input_data') or original.get('input_data', {})
        
        # 创建新的执行
        new_execution = service.execute_workflow(
            workflow_id=original.get('workflow_id'),
            tenant_id=tenant_id,
            user_id=user_id,
            input_data=input_data,
            trigger_type='retry'
        )
        
        # 更新新执行记录，记录重试来源
        service.execution_repo.update(new_execution.get('id'), {
            'parent_execution_id': execution_id,
            'retry_count': original.get('retry_count', 0) + 1
        })
        
        return success_response({
            "execution": new_execution,
            "original_execution_id": execution_id
        }, "重试执行已启动", 202)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to retry execution: {e}")
        return error_response(f"重试执行失败: {str(e)}", 500)


@workflow_execution_bp.route('/<execution_id>/pause', methods=['POST'])
@jwt_required()
def pause_execution(execution_id):
    """暂停执行
    
    Returns:
        {
            "execution": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        execution = service.get_execution(execution_id, tenant_id)
        if not execution:
            return error_response("执行记录不存在", 404)
        
        # 检查状态是否可暂停
        if execution.get('status') != 'running':
            return error_response("只能暂停运行中的执行", 400)
        
        # 更新状态为暂停
        updated = service.execution_repo.update(execution_id, {'status': 'paused'})
        
        # 记录日志
        service._log(execution_id, execution.get('workflow_id'), 'info', 
                    f'Execution paused by user {user_id}')
        
        return success_response({
            "execution": service._execution_to_dict(updated)
        }, "执行已暂停")
        
    except Exception as e:
        logger.error(f"Failed to pause execution: {e}")
        return error_response(f"暂停执行失败: {str(e)}", 500)


@workflow_execution_bp.route('/<execution_id>/resume', methods=['POST'])
@jwt_required()
def resume_execution(execution_id):
    """恢复暂停的执行
    
    Returns:
        {
            "execution": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        execution = service.get_execution(execution_id, tenant_id)
        if not execution:
            return error_response("执行记录不存在", 404)
        
        # 检查状态是否可恢复
        if execution.get('status') != 'paused':
            return error_response("只能恢复暂停的执行", 400)
        
        # 更新状态为运行中
        updated = service.execution_repo.update(execution_id, {'status': 'running'})
        
        # 记录日志
        service._log(execution_id, execution.get('workflow_id'), 'info', 
                    f'Execution resumed by user {user_id}')
        
        return success_response({
            "execution": service._execution_to_dict(updated)
        }, "执行已恢复")
        
    except Exception as e:
        logger.error(f"Failed to resume execution: {e}")
        return error_response(f"恢复执行失败: {str(e)}", 500)


# ============================================================================
# 步骤管理接口
# ============================================================================

@workflow_execution_bp.route('/<execution_id>/steps', methods=['GET'])
@jwt_required()
def get_execution_steps(execution_id):
    """获取执行的步骤列表
    
    Returns:
        {
            "steps": [
                {
                    "id": "string",
                    "step_name": "string",
                    "step_type": "string",
                    "step_index": "integer",
                "status": "string",
                    "progress": "float",
                "started_at": "string",
                "completed_at": "string",
                    "duration_seconds": "float",
                    "error_message": "string"
            }
            ]
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        # 验证执行记录存在且属于租户
        execution = service.get_execution(execution_id, tenant_id)
        if not execution:
            return error_response("执行记录不存在", 404)
        
        # 获取步骤列表
        steps = service.step_repo.get_by_execution(execution_id)
        
        return success_response({
            "steps": [service._step_to_dict(s) for s in steps]
        }, "获取步骤列表成功")
        
    except Exception as e:
        logger.error(f"Failed to get execution steps: {e}")
        return error_response(f"获取步骤列表失败: {str(e)}", 500)


@workflow_execution_bp.route('/<execution_id>/steps/<step_id>', methods=['GET'])
@jwt_required()
def get_step_detail(execution_id, step_id):
    """获取步骤详情
    
    Returns:
        {
            "step": {
                "id": "string",
                "step_name": "string",
                "step_type": "string",
                "status": "string",
                "input_data": {},
                "output_data": {},
                "config": {},
                "error_message": "string",
                "error_details": {}
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        # 验证执行记录存在且属于租户
        execution = service.get_execution(execution_id, tenant_id)
        if not execution:
            return error_response("执行记录不存在", 404)
        
        # 获取步骤详情
        steps = service.step_repo.get_by_execution(execution_id)
        step = None
        for s in steps:
            s_id = s.get('id') if isinstance(s, dict) else s.id
            if str(s_id) == step_id:
                step = s
                break
        
        if not step:
            return error_response("步骤不存在", 404)
        
        # 获取步骤日志
        logs = service.log_repo.get_by_step(step_id)
        step_dict = service._step_to_dict(step)
        step_dict['logs'] = [service._log_to_dict(l) for l in logs]
        
        return success_response({
            "step": step_dict
        }, "获取步骤详情成功")
        
    except Exception as e:
        logger.error(f"Failed to get step detail: {e}")
        return error_response(f"获取步骤详情失败: {str(e)}", 500)


# ============================================================================
# 日志接口
# ============================================================================

@workflow_execution_bp.route('/<execution_id>/logs', methods=['GET'])
@jwt_required()
def get_execution_logs(execution_id):
    """获取执行日志
    
    Query Parameters:
        - level: 日志级别过滤 (debug, info, warning, error, critical)
        - step_id: 步骤ID过滤
        - page: 页码 (默认: 1)
        - limit: 每页数量 (默认: 100)
    
    Returns:
        {
            "logs": [...],
            "pagination": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        level = request.args.get('level')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 100))
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        result = service.get_execution_logs(
            execution_id=execution_id,
            tenant_id=tenant_id,
            level=level,
            page=page,
            page_size=limit
        )
        
        return success_response({
            "logs": result.get('items', []),
            "pagination": {
                "page": result.get('page', page),
                "limit": result.get('page_size', limit),
                "total": result.get('total', 0),
                "pages": result.get('total_pages', 0)
            }
        }, "获取执行日志成功")
        
    except Exception as e:
        logger.error(f"Failed to get execution logs: {e}")
        return error_response(f"获取执行日志失败: {str(e)}", 500)


@workflow_execution_bp.route('/<execution_id>/logs', methods=['POST'])
@jwt_required()
def add_execution_log(execution_id):
    """添加执行日志（供内部或外部系统调用）
    
    Request Body:
        {
            "level": "string" (info, warning, error, etc.),
            "message": "string",
            "step_id": "string" (optional),
            "details": {} (optional)
        }
    
    Returns:
        {
            "log": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = validate_json(request)
        validate_required_fields(data, ['level', 'message'])
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        # 验证执行记录存在且属于租户
        execution = service.get_execution(execution_id, tenant_id)
        if not execution:
            return error_response("执行记录不存在", 404)
        
        # 创建日志
        log = service.log_repo.create({
            'execution_id': execution_id,
            'workflow_id': execution.get('workflow_id'),
            'step_id': data.get('step_id'),
            'level': data['level'],
            'message': data['message'],
            'details': data.get('details'),
            'source': 'api'
        })
        
        return success_response({
            "log": service._log_to_dict(log)
        }, "日志添加成功", 201)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to add execution log: {e}")
        return error_response(f"添加日志失败: {str(e)}", 500)


# ============================================================================
# 进度更新接口（供执行器调用）
# ============================================================================

@workflow_execution_bp.route('/<execution_id>/progress', methods=['PUT'])
@jwt_required()
def update_execution_progress(execution_id):
    """更新执行进度
    
    Request Body:
        {
            "progress": "float" (0-100),
            "current_step": "string" (optional),
            "current_step_index": "integer" (optional),
            "context_data": {} (optional)
        }
    
    Returns:
        {
            "execution": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = validate_json(request)
        validate_required_fields(data, ['progress'])
        
        progress = float(data['progress'])
        if progress < 0 or progress > 100:
            return error_response("进度必须在0-100之间", 400)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        # 验证执行记录存在且属于租户
        execution = service.get_execution(execution_id, tenant_id)
        if not execution:
            return error_response("执行记录不存在", 404)
        
        # 检查状态是否允许更新进度
        if execution.get('status') not in ('running', 'paused'):
            return error_response("只能更新运行中或暂停的执行进度", 400)
        
        # 更新进度
        update_data = {'progress': progress}
        if data.get('current_step'):
            update_data['current_step'] = data['current_step']
        if data.get('current_step_index') is not None:
            update_data['current_step_index'] = data['current_step_index']
        if data.get('context_data'):
            update_data['context_data'] = data['context_data']
        
        updated = service.execution_repo.update(execution_id, update_data)
        
        return success_response({
            "execution": service._execution_to_dict(updated)
        }, "进度更新成功")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to update execution progress: {e}")
        return error_response(f"更新进度失败: {str(e)}", 500)


@workflow_execution_bp.route('/<execution_id>/steps/<step_id>/status', methods=['PUT'])
@jwt_required()
def update_step_status(execution_id, step_id):
    """更新步骤状态
    
    Request Body:
        {
            "status": "string" (pending, running, completed, failed, skipped),
            "progress": "float" (optional),
            "output_data": {} (optional),
            "error_message": "string" (optional),
            "error_details": {} (optional)
        }
    
    Returns:
        {
            "step": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = validate_json(request)
        validate_required_fields(data, ['status'])
        
        valid_statuses = ['pending', 'running', 'completed', 'failed', 'skipped']
        if data['status'] not in valid_statuses:
            return error_response(f"无效状态，必须是: {', '.join(valid_statuses)}", 400)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        # 验证执行记录存在且属于租户
        execution = service.get_execution(execution_id, tenant_id)
        if not execution:
            return error_response("执行记录不存在", 404)
        
        # 更新步骤状态
        updated = service.step_repo.update_status(
            step_id=step_id,
            status=data['status'],
            error_message=data.get('error_message'),
            output_data=data.get('output_data')
        )
        
        if not updated:
            return error_response("步骤不存在", 404)
        
        # 如果更新了progress，也更新
        if data.get('progress') is not None:
            service.step_repo.update(step_id, {'progress': data['progress']})
        
        # 记录日志
        log_level = 'error' if data['status'] == 'failed' else 'info'
        log_msg = f"Step status changed to {data['status']}"
        if data.get('error_message'):
            log_msg += f": {data['error_message']}"
        service._log(execution_id, execution.get('workflow_id'), log_level, log_msg, step_id)
        
        return success_response({
            "step": service._step_to_dict(service.step_repo.get_by_execution(execution_id))
        }, "步骤状态更新成功")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to update step status: {e}")
        return error_response(f"更新步骤状态失败: {str(e)}", 500)


@workflow_execution_bp.route('/<execution_id>/complete', methods=['POST'])
@jwt_required()
def complete_execution(execution_id):
    """完成执行
    
    Request Body:
        {
            "success": "boolean",
            "output_data": {} (optional),
            "error_message": "string" (optional, 当 success=false 时)
        }
    
    Returns:
        {
            "execution": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = validate_json(request)
        validate_required_fields(data, ['success'])
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        # 验证执行记录存在且属于租户
        execution = service.get_execution(execution_id, tenant_id)
        if not execution:
            return error_response("执行记录不存在", 404)
        
        # 检查状态是否允许完成
        if execution.get('status') not in ('running', 'paused'):
            return error_response("只能完成运行中或暂停的执行", 400)
        
        # 完成执行
        updated = service.complete_execution(
            execution_id=execution_id,
            success=data['success'],
            output_data=data.get('output_data'),
            error_message=data.get('error_message')
        )
        
        return success_response({
            "execution": updated
        }, "执行完成" if data['success'] else "执行失败")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to complete execution: {e}")
        return error_response(f"完成执行失败: {str(e)}", 500)


# ============================================================================
# 统计接口
# ============================================================================

@workflow_execution_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_execution_statistics():
    """获取执行统计信息
    
    Query Parameters:
        - workflow_id: 工作流ID过滤 (optional)
        - period: 统计周期 (day, week, month, all), 默认 all
    
    Returns:
        {
            "statistics": {
                "total_executions": "integer",
                "by_status": {
                "pending": "integer",
                "running": "integer",
                "completed": "integer",
                "failed": "integer",
                    "cancelled": "integer"
                },
                "success_rate": "float",
                "avg_duration_seconds": "float",
                "executions_today": "integer",
                "executions_this_week": "integer"
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        workflow_id = request.args.get('workflow_id')
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        # 获取执行统计
        stats = service.execution_repo.get_execution_statistics(tenant_id, workflow_id)
        
        # 获取运行中的执行数
        running = service.get_running_executions(tenant_id)
        if workflow_id:
            running = [e for e in running if e.get('workflow_id') == workflow_id]
        
        statistics = {
            'total_executions': stats.get('total_executions', 0),
            'by_status': stats.get('by_status', {}),
            'success_rate': stats.get('success_rate', 0),
            'avg_duration_seconds': stats.get('avg_duration_seconds', 0),
            'running_count': len(running)
        }
        
        return success_response({
            "statistics": statistics
        }, "获取执行统计成功")
        
    except Exception as e:
        logger.error(f"Failed to get execution statistics: {e}")
        return error_response(f"获取执行统计失败: {str(e)}", 500)


@workflow_execution_bp.route('/running', methods=['GET'])
@jwt_required()
def get_running_executions():
    """获取正在运行的执行
    
    Returns:
        {
            "executions": [...]
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        executions = service.get_running_executions(tenant_id)
        
        return success_response({
            "executions": executions
        }, "获取运行中的执行成功")
        
    except Exception as e:
        logger.error(f"Failed to get running executions: {e}")
        return error_response(f"获取运行中的执行失败: {str(e)}", 500)


@workflow_execution_bp.route('/recent', methods=['GET'])
@jwt_required()
def get_recent_executions():
    """获取最近的执行记录
    
    Query Parameters:
        - limit: 返回数量 (默认: 10, 最大: 50)
    
    Returns:
        {
            "executions": [...]
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        limit = min(int(request.args.get('limit', 10)), 50)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        result = service.list_executions(
            tenant_id=tenant_id,
            page=1,
            page_size=limit
        )
        
        return success_response({
            "executions": result.get('items', [])
        }, "获取最近执行记录成功")
        
    except Exception as e:
        logger.error(f"Failed to get recent executions: {e}")
        return error_response(f"获取最近执行记录失败: {str(e)}", 500)


# ============================================================================
# 批量操作接口
# ============================================================================

@workflow_execution_bp.route('/batch/cancel', methods=['POST'])
@jwt_required()
def batch_cancel_executions():
    """批量取消执行
    
    Request Body:
        {
            "execution_ids": ["string"],
            "reason": "string" (optional)
        }
    
    Returns:
        {
            "cancelled": ["string"],
            "failed": [
                {"id": "string", "error": "string"}
            ]
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = validate_json(request)
        validate_required_fields(data, ['execution_ids'])
        
        execution_ids = data['execution_ids']
        reason = data.get('reason', '批量取消')
        
        if not execution_ids or len(execution_ids) > 100:
            return error_response("执行ID列表不能为空且不能超过100个", 400)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        cancelled = []
        failed = []
        
        for exe_id in execution_ids:
            try:
                result = service.cancel_execution(exe_id, tenant_id, user_id, reason)
                if result:
                    cancelled.append(exe_id)
                else:
                    failed.append({'id': exe_id, 'error': '执行不存在'})
            except Exception as e:
                failed.append({'id': exe_id, 'error': str(e)})
        
        return success_response({
            "cancelled": cancelled,
            "failed": failed
        }, f"批量取消完成: 成功 {len(cancelled)}, 失败 {len(failed)}")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to batch cancel executions: {e}")
        return error_response(f"批量取消失败: {str(e)}", 500)


@workflow_execution_bp.route('/cleanup', methods=['POST'])
@jwt_required()
def cleanup_old_executions():
    """清理旧的执行记录和日志
    
    Request Body:
        {
            "days": "integer" (清理多少天前的记录，最小30天),
            "status": ["string"] (可选，只清理特定状态的记录)
        }
    
    Returns:
        {
            "deleted_count": "integer"
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = validate_json(request)
        validate_required_fields(data, ['days'])
        
        days = int(data['days'])
        if days < 30:
            return error_response("为安全起见，只能清理30天以前的记录", 400)
        
        statuses = data.get('status', ['completed', 'failed', 'cancelled'])
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        # 注意：实际清理逻辑需要在 Repository 中实现
        # 这里只是演示接口设计
        
        return success_response({
            "message": f"清理任务已提交，将清理 {days} 天前状态为 {statuses} 的记录",
            "status": "pending"
        }, "清理任务已提交")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to cleanup executions: {e}")
        return error_response(f"清理执行记录失败: {str(e)}", 500)

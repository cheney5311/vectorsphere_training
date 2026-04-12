"""工作流管理API接口

提供工作流管理相关的API接口，支持租户维度的数据隔离和持久化。
"""

import sys
import os
import logging
from flask import Blueprint, request
from flask_jwt_extended import jwt_required, get_jwt_identity

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from backend.core.exceptions import ValidationError
except ImportError:
    from backend.core.exceptions import ValidationError

from backend.utils.response import success_response, error_response
from backend.utils.validation import validate_json, validate_required_fields, validate_string_length

logger = logging.getLogger(__name__)

# 创建蓝图
workflow_bp = Blueprint('workflow', __name__, url_prefix='/api/v1/workflows')


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
# 工作流 CRUD 接口
# ============================================================================

@workflow_bp.route('', methods=['GET'])
@jwt_required()
def get_workflows():
    """获取工作流列表
    
    Query Parameters:
        - type: 工作流类型过滤
        - status: 状态过滤
        - search: 搜索关键字
        - page: 页码 (默认: 1)
        - limit: 每页数量 (默认: 20)
    
    Returns:
        {
            "workflows": [...],
            "pagination": {
                "page": "integer",
                "limit": "integer",
                "total": "integer",
                "pages": "integer"
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        workflow_type = request.args.get('type')
        status = request.args.get('status')
        search = request.args.get('search')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        result = service.list_workflows(
            tenant_id=tenant_id,
            workflow_type=workflow_type,
            status=status,
            search=search,
            page=page,
            page_size=limit
        )
        
        return success_response({
            "workflows": result.get('items', []),
            "pagination": {
                "page": result.get('page', page),
                "limit": result.get('page_size', limit),
                "total": result.get('total', 0),
                "pages": result.get('total_pages', 0)
            }
        }, "获取工作流列表成功")
        
    except Exception as e:
        logger.error(f"Failed to get workflows: {e}")
        return error_response(f"获取工作流列表失败: {str(e)}", 500)


@workflow_bp.route('/<workflow_id>', methods=['GET'])
@jwt_required()
def get_workflow(workflow_id):
    """获取工作流详情
    
    Returns:
        {
            "workflow": {
                "workflow_id": "string",
                "name": "string",
                "description": "string",
                "workflow_type": "string",
                "status": "string",
                "config": {},
                "steps_config": [],
                ...
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        workflow = service.get_workflow(workflow_id, tenant_id)
        if not workflow:
            return error_response("工作流不存在", 404)
        
        return success_response({
            "workflow": workflow
        }, "获取工作流详情成功")
        
    except Exception as e:
        logger.error(f"Failed to get workflow: {e}")
        return error_response(f"获取工作流详情失败: {str(e)}", 500)


@workflow_bp.route('', methods=['POST'])
@jwt_required()
def create_workflow():
    """创建工作流
    
    Request Body:
        {
            "name": "string",
            "description": "string",
            "workflow_type": "string",
            "config": {},
            "steps_config": [],
            "template_id": "string" (optional)
        }
    
    Returns:
        {
            "workflow": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = validate_json(request)
        validate_required_fields(data, ['name', 'workflow_type'])
        validate_string_length(data['name'], 'name', 1, 200)
        
        if data.get('description'):
            validate_string_length(data['description'], 'description', 0, 1000)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        workflow = service.create_workflow(tenant_id, user_id, data)
        
        return success_response({
            "workflow": workflow
        }, "工作流创建成功", 201)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        return error_response(f"创建工作流失败: {str(e)}", 500)


@workflow_bp.route('/<workflow_id>', methods=['PUT'])
@jwt_required()
def update_workflow(workflow_id):
    """更新工作流
    
    Request Body:
        {
            "name": "string",
            "description": "string",
            "status": "string",
            "config": {},
            "steps_config": []
        }
    
    Returns:
        {
            "workflow": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = validate_json(request)
        
        if data.get('name'):
            validate_string_length(data['name'], 'name', 1, 200)
        if data.get('description'):
            validate_string_length(data['description'], 'description', 0, 1000)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        workflow = service.update_workflow(workflow_id, tenant_id, user_id, data)
        if not workflow:
            return error_response("工作流不存在", 404)
        
        return success_response({
            "workflow": workflow
        }, "工作流更新成功")
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to update workflow: {e}")
        return error_response(f"更新工作流失败: {str(e)}", 500)


@workflow_bp.route('/<workflow_id>', methods=['DELETE'])
@jwt_required()
def delete_workflow(workflow_id):
    """删除工作流
    
    Returns:
        {
            "message": "string"
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        success = service.delete_workflow(workflow_id, tenant_id)
        if not success:
            return error_response("工作流不存在或无法删除", 404)
        
        return success_response(None, "工作流删除成功")
        
    except Exception as e:
        logger.error(f"Failed to delete workflow: {e}")
        return error_response(f"删除工作流失败: {str(e)}", 500)


# ============================================================================
# 工作流执行接口
# ============================================================================

@workflow_bp.route('/<workflow_id>/execute', methods=['POST'])
@jwt_required()
def execute_workflow(workflow_id):
    """执行工作流
    
    Request Body:
        {
            "input_data": {}
        }
    
    Returns:
        {
            "execution": {
                "execution_id": "string",
                "workflow_id": "string",
                "status": "string",
                "started_at": "string",
                ...
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = validate_json(request) if request.is_json else {}
        input_data = data.get('input_data', {})
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        execution = service.execute_workflow(
            workflow_id=workflow_id,
            tenant_id=tenant_id,
            user_id=user_id,
            input_data=input_data,
            trigger_type='manual'
        )
        
        return success_response({
            "execution": execution
        }, "工作流执行已启动", 202)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to execute workflow: {e}")
        return error_response(f"执行工作流失败: {str(e)}", 500)


@workflow_bp.route('/executions', methods=['GET'])
@jwt_required()
def get_workflow_executions():
    """获取工作流执行记录
    
    Query Parameters:
        - workflow_id: 工作流ID过滤
        - status: 状态过滤
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
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        result = service.list_executions(
            tenant_id=tenant_id,
            workflow_id=workflow_id,
            status=status,
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
        }, "获取工作流执行记录成功")
        
    except Exception as e:
        logger.error(f"Failed to get workflow executions: {e}")
        return error_response(f"获取工作流执行记录失败: {str(e)}", 500)


@workflow_bp.route('/executions/<execution_id>', methods=['GET'])
@jwt_required()
def get_execution_status(execution_id):
    """获取工作流执行状态
    
    Returns:
        {
            "execution": {
                "execution_id": "string",
                "workflow_id": "string",
                "status": "string",
                "progress": "float",
                "steps": [...],
                ...
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
        }, "获取执行状态成功")
        
    except Exception as e:
        logger.error(f"Failed to get execution status: {e}")
        return error_response(f"获取执行状态失败: {str(e)}", 500)


@workflow_bp.route('/executions/<execution_id>/cancel', methods=['POST'])
@jwt_required()
def cancel_execution(execution_id):
    """取消工作流执行
    
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
        reason = data.get('reason')
        
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


@workflow_bp.route('/executions/<execution_id>/logs', methods=['GET'])
@jwt_required()
def get_execution_logs(execution_id):
    """获取执行日志
    
    Query Parameters:
        - level: 日志级别过滤
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


# ============================================================================
# 模板接口
# ============================================================================

@workflow_bp.route('/templates', methods=['GET'])
@jwt_required()
def list_templates():
    """列出所有配置模板
    
    Query Parameters:
        - type: 工作流类型过滤
        - category: 分类过滤
        - page: 页码 (默认: 1)
        - limit: 每页数量 (默认: 20)
    
    Returns:
        {
            "templates": [...],
            "pagination": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        workflow_type = request.args.get('type')
        category = request.args.get('category')
        page = int(request.args.get('page', 1))
        limit = int(request.args.get('limit', 20))
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        result = service.list_templates(
            tenant_id=tenant_id,
            workflow_type=workflow_type,
            category=category,
            page=page,
            page_size=limit
        )
        
        return success_response({
            "templates": result.get('items', []),
            "pagination": {
                "page": result.get('page', page),
                "limit": result.get('page_size', limit),
                "total": result.get('total', 0),
                "pages": result.get('total_pages', 0)
            }
        }, "获取模板列表成功")
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        return error_response(f"获取模板列表失败: {str(e)}", 500)


@workflow_bp.route('/templates/<template_id>', methods=['GET'])
@jwt_required()
def get_template(template_id):
    """获取配置模板
    
    Returns:
        {
            "template": {
                "template_id": "string",
                "name": "string",
                "description": "string",
                "workflow_type": "string",
                "config": {},
                "steps_config": [],
                "default_params": {}
            }
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        template = service.get_template(template_id, tenant_id)
        if not template:
            return error_response("模板不存在", 404)
        
        return success_response({
            "template": template
        }, "获取模板成功")
        
    except Exception as e:
        logger.error(f"Failed to get template: {e}")
        return error_response(f"获取模板失败: {str(e)}", 500)


@workflow_bp.route('/templates', methods=['POST'])
@jwt_required()
def create_template():
    """创建工作流模板
    
    Request Body:
        {
            "name": "string",
            "description": "string",
            "workflow_type": "string",
            "config": {},
            "steps_config": [],
            "default_params": {},
            "is_public": boolean,
            "category": "string"
        }
    
    Returns:
        {
            "template": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = validate_json(request)
        validate_required_fields(data, ['name', 'workflow_type'])
        validate_string_length(data['name'], 'name', 1, 200)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        template = service.create_template(tenant_id, user_id, data)
        
        return success_response({
            "template": template
        }, "模板创建成功", 201)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to create template: {e}")
        return error_response(f"创建模板失败: {str(e)}", 500)


@workflow_bp.route('/templates/<template_id>/create-workflow', methods=['POST'])
@jwt_required()
def create_workflow_from_template(template_id):
    """从模板创建工作流
    
    Request Body:
        {
            "name": "string",
            "description": "string" (optional),
            "config": {} (optional, 覆盖默认配置)
        }
    
    Returns:
        {
            "workflow": {...}
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        data = validate_json(request)
        validate_required_fields(data, ['name'])
        validate_string_length(data['name'], 'name', 1, 200)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        workflow = service.create_workflow_from_template(
            template_id=template_id,
            tenant_id=tenant_id,
            user_id=user_id,
            workflow_name=data['name'],
            workflow_description=data.get('description'),
            override_config=data.get('config')
        )
        
        return success_response({
            "workflow": workflow
        }, "从模板创建工作流成功", 201)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"Failed to create workflow from template: {e}")
        return error_response(f"从模板创建工作流失败: {str(e)}", 500)


# ============================================================================
# 统计接口
# ============================================================================

@workflow_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_workflow_statistics():
    """获取工作流统计信息
    
    Returns:
        {
            "workflow_stats": {
                "total_workflows": "integer",
                "by_type": {},
                "by_status": {},
                ...
            },
            "execution_stats": {
                "total_executions": "integer",
                "by_status": {},
                "avg_duration_seconds": "float",
                "success_rate": "float"
            },
            "templates_available": "integer"
        }
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id(user_id)
        
        service = _get_workflow_service()
        if not service:
            return error_response("Workflow service not available", 503)
        
        statistics = service.get_workflow_statistics(tenant_id)
        
        return success_response(statistics, "获取工作流统计成功")
        
    except Exception as e:
        logger.error(f"Failed to get workflow statistics: {e}")
        return error_response(f"获取工作流统计失败: {str(e)}", 500)


@workflow_bp.route('/running', methods=['GET'])
@jwt_required()
def get_running_workflows():
    """获取正在运行的工作流执行
    
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
        }, "获取运行中的工作流成功")
        
    except Exception as e:
        logger.error(f"Failed to get running workflows: {e}")
        return error_response(f"获取运行中的工作流失败: {str(e)}", 500)

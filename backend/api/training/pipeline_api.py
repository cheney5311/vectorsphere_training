"""训练流水线 API

提供创建/启动/暂停/恢复/回滚/查询等流水线管理端点。
支持租户级别的数据隔离和数据库持久化存储。
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from typing import Dict, Any, Optional
import logging

from backend.services.pipeline_service import get_pipeline_service

pipeline_bp = Blueprint("training_pipeline_api", __name__, url_prefix="/api/v1/training/pipeline")
logger = logging.getLogger(__name__)


def _get_tenant_id() -> Optional[str]:
    """获取当前租户ID"""
    # 优先从请求头获取
    tenant_id = request.headers.get('X-Tenant-ID')
    if tenant_id:
        return tenant_id
    
    # 从查询参数获取
    tenant_id = request.args.get('tenant_id')
    if tenant_id:
        return tenant_id
    
    # 尝试从JWT获取
    try:
        from flask_jwt_extended import get_jwt
        claims = get_jwt()
        return claims.get('tenant_id')
    except Exception:
            pass
    
    return None


def _get_service():
    """获取流水线服务实例"""
    return get_pipeline_service()


# ==============================================================================
# 流水线管理端点
# ==============================================================================

@pipeline_bp.route("/create", methods=["POST"])
@jwt_required()
def create_pipeline():
    """
    创建流水线
    
    Request Body:
        name: 流水线名称
        steps: 步骤配置列表 [{name, type, params, on_fail}]
        description: 描述 (可选)
        model_name: 模型名称 (可选)
        model_id: 模型ID (可选)
        dataset_id: 数据集ID (可选)
        global_config: 全局配置 (可选)
        enable_rollback: 是否启用回滚 (默认true)
        tags: 标签列表 (可选)
        
    Returns:
        创建的流水线信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        data = request.get_json(force=True) or {}
        
        # 验证必需字段
        name = data.get('name')
        steps = data.get('steps', [])
        
        if not name:
            return jsonify({'success': False, 'error': 'Pipeline name is required'}), 400
        
        if not steps:
            return jsonify({'success': False, 'error': 'At least one step is required'}), 400
        
        service = _get_service()
        result = service.create_pipeline(
            name=name,
            steps_config=steps,
            tenant_id=tenant_id,
            user_id=user_id,
            description=data.get('description'),
            model_name=data.get('model_name'),
            model_id=data.get('model_id'),
            dataset_id=data.get('dataset_id'),
            global_config=data.get('global_config'),
            enable_rollback=data.get('enable_rollback', True),
            tags=data.get('tags')
        )
        
        logger.info(f"User {user_id} created pipeline: {result.get('pipeline_id')}")
        
        return jsonify({
            'success': True,
            'message': 'Pipeline created',
            'data': result
        }), 201
        
    except Exception as e:
        logger.error(f"Failed to create pipeline: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/pipelines", methods=["GET"])
@jwt_required()
def list_pipelines():
    """
    获取流水线列表
    
    Query Parameters:
        status: 状态过滤
        model_id: 模型ID过滤
        limit: 返回数量限制 (默认50)
        offset: 偏移量 (默认0)
        
    Returns:
        流水线列表
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        status = request.args.get('status')
        model_id = request.args.get('model_id')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = _get_service()
        result = service.list_pipelines(
            tenant_id=tenant_id,
            status=status,
            model_id=model_id,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list pipelines: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/pipelines/<pipeline_id>", methods=["GET"])
@jwt_required()
def get_pipeline(pipeline_id: str):
    """
    获取流水线详情
    
    Args:
        pipeline_id: 流水线ID
        
    Returns:
        流水线详情
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        pipeline = service.get_pipeline(pipeline_id, tenant_id)
        
        if not pipeline:
            # 尝试通过名称查找
            pipeline = service.get_pipeline_by_name(pipeline_id, tenant_id)
        
        if pipeline:
            return jsonify({
                'success': True,
                'data': pipeline
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Pipeline not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Failed to get pipeline: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/pipelines/<pipeline_id>", methods=["PUT"])
@jwt_required()
def update_pipeline(pipeline_id: str):
    """
    更新流水线
    
    Args:
        pipeline_id: 流水线ID
        
    Request Body:
        可更新的字段: name, description, steps, global_config, enable_rollback, tags
        
    Returns:
        更新后的流水线信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        data = request.get_json(force=True) or {}
        
        # 构建更新内容
        updates = {}
        for key in ['name', 'description', 'global_config', 'enable_rollback', 'tags', 'model_name', 'model_id', 'dataset_id']:
            if key in data:
                updates[key] = data[key]
        
        if 'steps' in data:
            updates['steps_config'] = data['steps']
        
        if not updates:
            return jsonify({'success': False, 'error': 'No updates provided'}), 400
        
        service = _get_service()
        result = service.update_pipeline(pipeline_id, tenant_id, updates)
        
        if result:
            logger.info(f"User {user_id} updated pipeline: {pipeline_id}")
            return jsonify({
                'success': True,
                'message': 'Pipeline updated',
                'data': result
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Pipeline not found or update failed'
            }), 404
        
    except Exception as e:
        logger.error(f"Failed to update pipeline: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/pipelines/<pipeline_id>", methods=["DELETE"])
@jwt_required()
def delete_pipeline(pipeline_id: str):
    """
    删除流水线
    
    Args:
        pipeline_id: 流水线ID
        
    Returns:
        操作结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        success = service.delete_pipeline(pipeline_id, tenant_id)
        
        if success:
            logger.info(f"User {user_id} deleted pipeline: {pipeline_id}")
            return jsonify({
                'success': True,
                'message': 'Pipeline deleted'
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Pipeline not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Failed to delete pipeline: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/status/<name>", methods=["GET"])
@jwt_required()
def status_pipeline(name: str):
    """
    获取流水线状态（兼容旧API）
    
    Args:
        name: 流水线名称或ID
        
    Returns:
        流水线状态
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        pipeline = service.get_pipeline(name, tenant_id)
        
        if not pipeline:
            pipeline = service.get_pipeline_by_name(name, tenant_id)
        
        if pipeline:
            # 获取最近的执行记录
            executions = service.list_executions(
                tenant_id=tenant_id,
                pipeline_id=pipeline.get('pipeline_id'),
                limit=1
            )
            
            last_result = None
            if executions.get('executions'):
                last_execution = executions['executions'][0]
                last_result = last_execution.get('result')
            
            return jsonify({
                'success': True,
                'name': pipeline.get('name'),
                'pipeline_id': pipeline.get('pipeline_id'),
                'status': pipeline.get('status'),
                'last_result': last_result
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Pipeline not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/statistics", methods=["GET"])
@jwt_required()
def get_pipeline_statistics():
    """
    获取流水线统计信息
    
    Returns:
        统计信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        stats = service.get_pipeline_statistics(tenant_id, user_id)
        
        return jsonify({
            'success': True,
            'data': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get pipeline statistics: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==============================================================================
# 流水线执行端点
# ==============================================================================

@pipeline_bp.route("/start", methods=["POST"])
@jwt_required()
def start_pipeline():
    """
    启动流水线
    
    Request Body:
        name: 流水线名称或ID
        session_id: 自定义会话ID (可选)
        runtime_config: 运行时配置 (可选)
        
    Returns:
        执行信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        data = request.get_json(force=True) or {}
        name = data.get('name')
        session_id = data.get('session_id')
        runtime_config = data.get('runtime_config')
        
        if not name:
            return jsonify({'success': False, 'error': 'Pipeline name is required'}), 400
        
        service = _get_service()
        result = service.start_pipeline(
            pipeline_id=name,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            runtime_config=runtime_config
        )
        
        if result.get('success'):
            logger.info(f"User {user_id} started pipeline: {name}, execution_id={result.get('execution_id')}")
            return jsonify({
                'success': True,
                'message': 'Pipeline started',
                'name': name,
                'status': 'running',
                **result
            }), 200
        else:
            return jsonify(result), 400 if 'not found' in result.get('error', '').lower() else 409
        
    except Exception as e:
        logger.error(f"Failed to start pipeline: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/pause", methods=["POST"])
@jwt_required()
def pause_pipeline():
    """
    暂停流水线执行
    
    Request Body:
        session_id: 会话ID
        
    Returns:
        操作结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        data = request.get_json(force=True) or {}
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'session_id is required'}), 400
        
        service = _get_service()
        result = service.pause_execution(session_id, tenant_id)
        
        if result.get('success'):
            logger.info(f"User {user_id} paused pipeline execution: {session_id}")
            return jsonify(result), 200
        else:
            return jsonify(result), 500
        
    except Exception as e:
        logger.error(f"Failed to pause pipeline: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/resume", methods=["POST"])
@jwt_required()
def resume_pipeline():
    """
    恢复流水线执行
    
    Request Body:
        session_id: 会话ID
        
    Returns:
        操作结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        data = request.get_json(force=True) or {}
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'success': False, 'error': 'session_id is required'}), 400
        
        service = _get_service()
        result = service.resume_execution(session_id, tenant_id)
        
        if result.get('success'):
            logger.info(f"User {user_id} resumed pipeline execution: {session_id}")
            return jsonify(result), 200
        else:
            return jsonify(result), 500
        
    except Exception as e:
        logger.error(f"Failed to resume pipeline: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/rollback", methods=["POST"])
@jwt_required()
def rollback_pipeline():
    """
    回滚流水线
    
    Request Body:
        name: 流水线名称或ID
        session_id: 会话ID (可选)
        
    Returns:
        操作结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        data = request.get_json(force=True) or {}
        name = data.get('name')
        session_id = data.get('session_id')
        
        if not name:
            return jsonify({'success': False, 'error': 'Pipeline name is required'}), 400
        
        service = _get_service()
        result = service.rollback_execution(name, tenant_id, session_id)
        
        if result.get('success'):
            logger.info(f"User {user_id} rolled back pipeline: {name}")
            return jsonify({
                'message': 'rolled_back',
                'name': name,
                **result
            }), 200
        else:
            return jsonify(result), 500
        
    except Exception as e:
        logger.error(f"Failed to rollback pipeline: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/executions", methods=["GET"])
@jwt_required()
def list_executions():
    """
    获取执行记录列表
    
    Query Parameters:
        pipeline_id: 流水线ID过滤
        status: 状态过滤
        limit: 返回数量限制 (默认50)
        offset: 偏移量 (默认0)
        
    Returns:
        执行记录列表
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        pipeline_id = request.args.get('pipeline_id')
        status = request.args.get('status')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = _get_service()
        result = service.list_executions(
            tenant_id=tenant_id,
            pipeline_id=pipeline_id,
            status=status,
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list executions: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/executions/<execution_id>", methods=["GET"])
@jwt_required()
def get_execution_status(execution_id: str):
    """
    获取执行状态详情
    
    Args:
        execution_id: 执行ID
        
    Returns:
        执行状态和步骤详情
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        execution = service.get_execution_status(execution_id, tenant_id)
        
        if execution:
            return jsonify({
                'success': True,
                'data': execution
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Execution not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Failed to get execution status: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==============================================================================
# 模板管理端点
# ==============================================================================

@pipeline_bp.route("/templates", methods=["GET"])
@jwt_required()
def list_templates():
    """
    获取流水线模板列表
    
    Query Parameters:
        category: 分类过滤
        include_system: 是否包含系统模板 (默认true)
        limit: 返回数量限制 (默认50)
        offset: 偏移量 (默认0)
        
    Returns:
        模板列表
    """
    try:
        tenant_id = _get_tenant_id()
        
        category = request.args.get('category')
        include_system = request.args.get('include_system', 'true').lower() == 'true'
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = _get_service()
        result = service.list_templates(
            tenant_id=tenant_id,
            category=category,
            include_system=include_system,
            limit=limit,
            offset=offset
        )
        
        return jsonify({
            'success': True,
            'data': result
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to list templates: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/templates", methods=["POST"])
@jwt_required()
def create_template():
    """
    创建流水线模板
    
    Request Body:
        name: 模板名称
        steps_template: 步骤模板配置
        description: 描述 (可选)
        category: 分类 (可选)
        default_config: 默认配置 (可选)
        required_params: 必需参数 (可选)
        tags: 标签 (可选)
        is_public: 是否公开 (默认false)
        
    Returns:
        创建的模板信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        data = request.get_json(force=True) or {}
        
        name = data.get('name')
        steps_template = data.get('steps_template', [])
        
        if not name:
            return jsonify({'success': False, 'error': 'Template name is required'}), 400
        
        if not steps_template:
            return jsonify({'success': False, 'error': 'At least one step template is required'}), 400
        
        service = _get_service()
        result = service.create_template(
            name=name,
            steps_template=steps_template,
            tenant_id=tenant_id,
            user_id=user_id,
            description=data.get('description'),
            category=data.get('category'),
            default_config=data.get('default_config'),
            required_params=data.get('required_params'),
            tags=data.get('tags'),
            is_public=data.get('is_public', False)
        )
        
        logger.info(f"User {user_id} created template: {result.get('template_id')}")
        
        return jsonify({
            'success': True,
            'message': 'Template created',
            'data': result
        }), 201
        
    except Exception as e:
        logger.error(f"Failed to create template: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/templates/<template_id>", methods=["GET"])
@jwt_required()
def get_template(template_id: str):
    """
    获取模板详情
    
    Args:
        template_id: 模板ID
        
    Returns:
        模板详情
    """
    try:
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        template = service.get_template(template_id, tenant_id)
        
        if template:
            return jsonify({
                'success': True,
                'data': template
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': 'Template not found'
            }), 404
        
    except Exception as e:
        logger.error(f"Failed to get template: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@pipeline_bp.route("/templates/<template_id>/create-pipeline", methods=["POST"])
@jwt_required()
def create_from_template(template_id: str):
    """
    从模板创建流水线
    
    Args:
        template_id: 模板ID
        
    Request Body:
        name: 流水线名称
        params: 模板参数 (可选)
        其他流水线配置...
        
    Returns:
        创建的流水线信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        data = request.get_json(force=True) or {}
        
        name = data.get('name')
        if not name:
            return jsonify({'success': False, 'error': 'Pipeline name is required'}), 400
        
        service = _get_service()
        result = service.create_pipeline_from_template(
            template_id=template_id,
            name=name,
            tenant_id=tenant_id,
            user_id=user_id,
            params=data.get('params'),
            description=data.get('description'),
            model_name=data.get('model_name'),
            model_id=data.get('model_id'),
            dataset_id=data.get('dataset_id'),
            tags=data.get('tags')
        )
        
        if result.get('success') == False:
            return jsonify(result), 404
        
        logger.info(f"User {user_id} created pipeline from template {template_id}: {result.get('pipeline_id')}")
        
        return jsonify({
            'success': True,
            'message': 'Pipeline created from template',
            'data': result
        }), 201
        
    except Exception as e:
        logger.error(f"Failed to create pipeline from template: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ==============================================================================
# 步骤类型端点
# ==============================================================================

@pipeline_bp.route("/step-types", methods=["GET"])
@jwt_required()
def get_step_types():
    """
    获取支持的步骤类型列表
    
    Returns:
        步骤类型列表
    """
    try:
        step_types = [
            {
                'type': 'pretrain',
                'name': 'Pre-training',
                'description': 'Model pre-training stage'
            },
            {
                'type': 'finetune',
                'name': 'Fine-tuning',
                'description': 'Model fine-tuning stage'
            },
            {
                'type': 'sft',
                'name': 'Supervised Fine-Tuning',
                'description': 'Supervised fine-tuning (SFT) stage'
            },
            {
                'type': 'preference_optim',
                'name': 'Preference Optimization',
                'description': 'DPO/RLHF preference optimization stage'
            },
            {
                'type': 'evaluation',
                'name': 'Evaluation',
                'description': 'Model evaluation stage'
            },
            {
                'type': 'validation',
                'name': 'Validation',
                'description': 'Model validation stage'
            },
            {
                'type': 'data_processing',
                'name': 'Data Processing',
                'description': 'Data preprocessing stage'
            },
            {
                'type': 'model_export',
                'name': 'Model Export',
                'description': 'Model export/conversion stage'
            },
            {
                'type': 'deployment',
                'name': 'Deployment',
                'description': 'Model deployment stage'
            },
            {
                'type': 'checkpoint',
                'name': 'Checkpoint',
                'description': 'Checkpoint saving stage'
            },
            {
                'type': 'custom',
                'name': 'Custom',
                'description': 'Custom step type'
            }
        ]
        
        failure_policies = [
            {'policy': 'continue', 'description': 'Continue to next step on failure'},
            {'policy': 'stop', 'description': 'Stop pipeline execution on failure'},
            {'policy': 'rollback', 'description': 'Rollback to previous step on failure'},
            {'policy': 'retry', 'description': 'Retry current step on failure'}
        ]
        
        return jsonify({
            'success': True,
            'data': {
                'step_types': step_types,
                'failure_policies': failure_policies
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Failed to get step types: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

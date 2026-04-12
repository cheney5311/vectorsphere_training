"""三阶段训练API接口

提供三阶段训练相关的REST API接口。
支持租户级别的数据隔离和服务层集成。
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
from typing import Optional

from backend.core.exceptions import ValidationError, BusinessLogicError
from backend.core.validation import validate_json_schema
from backend.utils.response import success_response, error_response
from backend.services.three_stage_training_service import get_three_stage_training_service

logger = logging.getLogger(__name__)

# 创建蓝图
three_stage_training_bp = Blueprint('three_stage_training', __name__, url_prefix='/api/v1/training/three-stage')


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
    """获取服务实例"""
    return get_three_stage_training_service()


# ==============================================================================
# 会话管理端点
# ==============================================================================

@three_stage_training_bp.route('/sessions', methods=['POST'])
@jwt_required()
def create_session():
    """
    创建三阶段训练会话
    
    Request Body:
        name: 会话名称 (可选，默认使用model_name)
        model_name: 模型名称 (必需)
        description: 描述 (可选)
        config: 训练配置
            - base_model_path: 基础模型路径
            - output_dir: 输出目录
            - pass_model_between_stages: 是否在阶段间传递模型
            - stages: 阶段配置
                - pretrain/pt: {enabled, epochs, batch_size, learning_rate, data_path}
                - finetune/sft: {enabled, epochs, batch_size, learning_rate, data_path}
                - preference/dpo: {enabled, epochs, batch_size, learning_rate, beta, data_path}
        
    Returns:
        创建的会话信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        data = request.get_json(force=True) or {}
        
        # 验证必需字段
        model_name = data.get('model_name')
        if not model_name:
            return error_response("缺少必需字段: model_name", 400)
        
        # 构建配置
        config = _build_config(data)
        
        service = _get_service()
        result = service.create_session(
            name=data.get('name', f"三阶段训练: {model_name}"),
            model_name=model_name,
            config=config,
            tenant_id=tenant_id,
            user_id=user_id,
            description=data.get('description')
        )
        
        logger.info(f"User {user_id} created three-stage session: {result.get('session_id')}")
        
        return success_response(result, "创建训练会话成功", 201)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"创建三阶段训练会话失败: {str(e)}")
        return error_response("创建训练会话失败", 500)


@three_stage_training_bp.route('/start', methods=['POST'])
@jwt_required()
def start_three_stage_training():
    """
    启动三阶段训练（创建并启动）
    
    Request Body:
        model_name: 模型名称 (必需)
        name: 会话名称 (可选)
        description: 描述 (可选)
        base_model: 基础模型路径 (可选)
        output_dir: 输出目录 (可选)
        pass_model_between_stages: 是否在阶段间传递模型 (可选)
        stages: 阶段配置 (可选)
        
        # 旧格式兼容
        pt_enabled, pt_epochs, pt_batch_size, pt_learning_rate, pt_data_path
        sft_enabled, sft_epochs, sft_batch_size, sft_learning_rate, sft_data_path
        dpo_enabled, dpo_epochs, dpo_batch_size, dpo_learning_rate, dpo_beta, dpo_data_path
        
    Returns:
        创建的会话信息和启动结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        data = request.get_json(force=True) or {}
        
        # 验证必需字段
        model_name = data.get('model_name')
        if not model_name:
            return error_response("缺少必需字段: model_name", 400)
        
        # 构建配置
        config = _build_config(data)
        
        service = _get_service()
        result = service.create_and_start(
            name=data.get('name', f"三阶段训练: {model_name}"),
            model_name=model_name,
            config=config,
            tenant_id=tenant_id,
            user_id=user_id,
            description=data.get('description')
        )
        
        logger.info(f"User {user_id} started three-stage training: {result.get('session_id')}")
        
        return success_response({
            'session_id': result.get('session_id'),
            'model_name': model_name,
            'status': result.get('status', 'running'),
            'config': config,
            'created_at': result.get('created_at'),
            'started_at': result.get('start_result', {}).get('started_at')
        }, "三阶段训练已启动", 201)
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except BusinessLogicError as e:
        return error_response(str(e), 500)
    except Exception as e:
        logger.error(f"启动三阶段训练失败: {str(e)}")
        return error_response("启动三阶段训练失败", 500)


@three_stage_training_bp.route('/sessions', methods=['GET'])
@jwt_required()
def list_sessions():
    """
    获取三阶段训练会话列表
    
    Query Parameters:
        status: 状态过滤
        model_name: 模型名称过滤
        limit: 返回数量限制 (默认50)
        offset: 偏移量 (默认0)
        
    Returns:
        会话列表
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        status = request.args.get('status')
        model_name = request.args.get('model_name')
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = _get_service()
        result = service.list_sessions(
            tenant_id=tenant_id,
            user_id=user_id,
            status=status,
            model_name=model_name,
            limit=limit,
            offset=offset
        )
        
        return success_response(result, "获取会话列表成功")
        
    except Exception as e:
        logger.error(f"获取三阶段训练会话列表失败: {str(e)}")
        return error_response("获取会话列表失败", 500)


@three_stage_training_bp.route('/status', methods=['GET'])
@jwt_required()
def get_all_three_stage_training_status():
    """
    获取当前用户的所有三阶段训练状态
    
    Returns:
        所有会话状态和统计信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        
        # 获取会话列表
        sessions_result = service.list_sessions(
            tenant_id=tenant_id,
            user_id=user_id,
            limit=100
        )
        
        # 获取统计信息
        stats = service.get_statistics(tenant_id, user_id)
        
        result = {
            'sessions': sessions_result.get('sessions', []),
            'total_count': stats.get('total', 0),
            'running_count': stats.get('running_count', 0),
            'completed_count': stats.get('completed_count', 0),
            'failed_count': stats.get('failed_count', 0)
        }
        
        return success_response(result, "获取训练状态成功")
        
    except Exception as e:
        logger.error(f"获取三阶段训练状态失败: {str(e)}")
        return error_response("获取训练状态失败", 500)


@three_stage_training_bp.route('/<session_id>', methods=['GET'])
@jwt_required()
def get_three_stage_training_status(session_id: str):
    """
    获取特定三阶段训练会话状态
    
    Args:
        session_id: 会话ID
        
    Returns:
        会话详情
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        session = service.get_session(session_id, tenant_id)
        
        if not session:
            return error_response("训练任务不存在", 404)
        
        # 验证用户权限
        if session.get('user_id') != user_id:
            return error_response("无权访问此训练任务", 403)
        
        return success_response(session, "获取训练状态成功")
        
    except Exception as e:
        logger.error(f"获取三阶段训练状态失败: {str(e)}")
        return error_response("获取训练状态失败", 500)


@three_stage_training_bp.route('/<session_id>', methods=['PUT'])
@jwt_required()
def update_session(session_id: str):
    """
    更新三阶段训练会话
    
    Args:
        session_id: 会话ID
        
    Request Body:
        name: 会话名称
        description: 描述
        
    Returns:
        更新后的会话信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        data = request.get_json(force=True) or {}
        
        service = _get_service()
        
        # 验证会话存在
        session = service.get_session(session_id, tenant_id)
        if not session:
            return error_response("训练任务不存在", 404)
        
        # 验证用户权限
        if session.get('user_id') != user_id:
            return error_response("无权修改此训练任务", 403)
        
        # 构建更新内容
        updates = {}
        if 'name' in data:
            updates['name'] = data['name']
        if 'description' in data:
            updates['description'] = data['description']
        
        if not updates:
            return error_response("无更新内容", 400)
        
        result = service.update_session(session_id, tenant_id, updates)
        
        if result:
            return success_response(result, "更新成功")
        else:
            return error_response("更新失败", 500)
        
    except Exception as e:
        logger.error(f"更新三阶段训练会话失败: {str(e)}")
        return error_response("更新失败", 500)


@three_stage_training_bp.route('/<session_id>', methods=['DELETE'])
@jwt_required()
def delete_session(session_id: str):
    """
    删除三阶段训练会话
    
    Args:
        session_id: 会话ID
        
    Returns:
        操作结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        
        # 验证会话存在
        session = service.get_session(session_id, tenant_id)
        if not session:
            return error_response("训练任务不存在", 404)
        
        # 验证用户权限
        if session.get('user_id') != user_id:
            return error_response("无权删除此训练任务", 403)
        
        # 不能删除运行中的任务
        if session.get('status') == 'running':
            return error_response("不能删除运行中的训练任务", 400)
        
        success = service.delete_session(session_id, tenant_id)
        
        if success:
            logger.info(f"User {user_id} deleted session: {session_id}")
            return success_response(None, "删除成功")
        else:
            return error_response("删除失败", 500)
        
    except Exception as e:
        logger.error(f"删除三阶段训练会话失败: {str(e)}")
        return error_response("删除失败", 500)


# ==============================================================================
# 训练控制端点
# ==============================================================================

@three_stage_training_bp.route('/<session_id>/start', methods=['POST'])
@jwt_required()
def start_session(session_id: str):
    """
    启动已创建的训练会话
    
    Args:
        session_id: 会话ID
        
    Returns:
        操作结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        
        # 验证会话存在
        session = service.get_session(session_id, tenant_id)
        if not session:
            return error_response("训练任务不存在", 404)
        
        # 验证用户权限
        if session.get('user_id') != user_id:
            return error_response("无权操作此训练任务", 403)
        
        result = service.start_training(session_id, tenant_id, user_id)
        
        if result.get('success'):
            logger.info(f"User {user_id} started training: {session_id}")
            return success_response(result, "训练已启动")
        else:
            return error_response(result.get('error', '启动失败'), 400)
        
    except Exception as e:
        logger.error(f"启动三阶段训练失败: {str(e)}")
        return error_response("启动训练失败", 500)


@three_stage_training_bp.route('/<session_id>/stop', methods=['POST'])
@jwt_required()
def stop_three_stage_training(session_id: str):
    """
    停止三阶段训练
    
    Args:
        session_id: 会话ID
        
    Returns:
        操作结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        
        # 验证会话存在
        session = service.get_session(session_id, tenant_id)
        if not session:
            return error_response("训练任务不存在", 404)
        
        # 验证用户权限
        if session.get('user_id') != user_id:
            return error_response("无权操作此训练任务", 403)
        
        result = service.stop_training(session_id, tenant_id)
        
        if result.get('success'):
            logger.info(f"User {user_id} stopped training: {session_id}")
            return success_response(result, "训练任务已停止")
        else:
            return error_response(result.get('error', '停止失败'), 400)
        
    except Exception as e:
        logger.error(f"停止三阶段训练失败: {str(e)}")
        return error_response("停止训练失败", 500)


@three_stage_training_bp.route('/<session_id>/pause', methods=['POST'])
@jwt_required()
def pause_three_stage_training(session_id: str):
    """
    暂停三阶段训练
    
    Args:
        session_id: 会话ID
        
    Returns:
        操作结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        
        # 验证会话存在
        session = service.get_session(session_id, tenant_id)
        if not session:
            return error_response("训练任务不存在", 404)
        
        # 验证用户权限
        if session.get('user_id') != user_id:
            return error_response("无权操作此训练任务", 403)
        
        result = service.pause_training(session_id, tenant_id)
        
        if result.get('success'):
            logger.info(f"User {user_id} paused training: {session_id}")
            return success_response(result, "训练任务已暂停")
        else:
            return error_response(result.get('error', '暂停失败'), 400)
        
    except Exception as e:
        logger.error(f"暂停三阶段训练失败: {str(e)}")
        return error_response("暂停训练失败", 500)


@three_stage_training_bp.route('/<session_id>/resume', methods=['POST'])
@jwt_required()
def resume_three_stage_training(session_id: str):
    """
    恢复三阶段训练
    
    Args:
        session_id: 会话ID
        
    Returns:
        操作结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        
        # 验证会话存在
        session = service.get_session(session_id, tenant_id)
        if not session:
            return error_response("训练任务不存在", 404)
        
        # 验证用户权限
        if session.get('user_id') != user_id:
            return error_response("无权操作此训练任务", 403)
        
        result = service.resume_training(session_id, tenant_id)
        
        if result.get('success'):
            logger.info(f"User {user_id} resumed training: {session_id}")
            return success_response(result, "训练任务已恢复")
        else:
            return error_response(result.get('error', '恢复失败'), 400)
        
    except Exception as e:
        logger.error(f"恢复三阶段训练失败: {str(e)}")
        return error_response("恢复训练失败", 500)


# ==============================================================================
# 进度和报告端点
# ==============================================================================

@three_stage_training_bp.route('/<session_id>/progress', methods=['GET'])
@jwt_required()
def get_progress(session_id: str):
    """
    获取训练进度
    
    Args:
        session_id: 会话ID
        
    Returns:
        进度信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        
        # 验证会话存在
        session = service.get_session(session_id, tenant_id)
        if not session:
            return error_response("训练任务不存在", 404)
        
        # 验证用户权限
        if session.get('user_id') != user_id:
            return error_response("无权访问此训练任务", 403)
        
        result = service.get_progress(session_id, tenant_id)
        
        if 'error' in result:
            return error_response(result['error'], 404)
        
        return success_response(result, "获取进度成功")
        
    except Exception as e:
        logger.error(f"获取三阶段训练进度失败: {str(e)}")
        return error_response("获取进度失败", 500)


@three_stage_training_bp.route('/<session_id>/progress/history', methods=['GET'])
@jwt_required()
def get_progress_history(session_id: str):
    """
    获取训练进度历史
    
    Args:
        session_id: 会话ID
        
    Query Parameters:
        stage: 阶段过滤 (pretrain, finetune, preference)
        limit: 返回数量限制 (默认100)
        
    Returns:
        进度历史记录
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        stage = request.args.get('stage')
        limit = request.args.get('limit', 100, type=int)
        
        service = _get_service()
        
        # 验证会话存在
        session = service.get_session(session_id, tenant_id)
        if not session:
            return error_response("训练任务不存在", 404)
        
        # 验证用户权限
        if session.get('user_id') != user_id:
            return error_response("无权访问此训练任务", 403)
        
        result = service.get_progress_history(session_id, tenant_id, stage, limit)
        
        if 'error' in result:
            return error_response(result['error'], 404)
        
        return success_response(result, "获取进度历史成功")
        
    except Exception as e:
        logger.error(f"获取三阶段训练进度历史失败: {str(e)}")
        return error_response("获取进度历史失败", 500)


@three_stage_training_bp.route('/history', methods=['GET'])
@jwt_required()
def get_three_stage_training_history():
    """
    获取三阶段训练历史
    
    Query Parameters:
        limit: 返回数量限制 (默认50)
        offset: 偏移量 (默认0)
        
    Returns:
        训练历史
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = _get_service()
        result = service.list_sessions(
            tenant_id=tenant_id,
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        return success_response(result.get('sessions', []), "获取训练历史成功")
        
    except Exception as e:
        logger.error(f"获取三阶段训练历史失败: {str(e)}")
        return error_response("获取训练历史失败", 500)


@three_stage_training_bp.route('/report', methods=['GET'])
@jwt_required()
def get_three_stage_training_reports():
    """
    获取用户的三阶段训练报告列表
    
    Query Parameters:
        limit: 返回数量限制 (默认50)
        offset: 偏移量 (默认0)
        
    Returns:
        报告列表
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        service = _get_service()
        sessions_result = service.list_sessions(
            tenant_id=tenant_id,
            user_id=user_id,
            limit=limit,
            offset=offset
        )
        
        # 获取每个会话的报告
        reports = []
        for session in sessions_result.get('sessions', []):
            session_id = session.get('session_id')
            if session_id:
                report = service.get_report(session_id, tenant_id)
                if 'error' not in report:
                    reports.append(report)
        
        return success_response(reports, "获取训练报告成功")
        
    except Exception as e:
        logger.error(f"获取三阶段训练报告失败: {str(e)}")
        return error_response("获取训练报告失败", 500)


@three_stage_training_bp.route('/<session_id>/report', methods=['GET'])
@jwt_required()
def get_three_stage_training_report(session_id: str):
    """
    获取特定会话的三阶段训练报告
    
    Args:
        session_id: 会话ID
        
    Returns:
        训练报告
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        
        # 验证会话存在
        session = service.get_session(session_id, tenant_id)
        if not session:
            return error_response("训练任务不存在", 404)
        
        # 验证用户权限
        if session.get('user_id') != user_id:
            return error_response("无权访问此训练任务", 403)
        
        report = service.get_report(session_id, tenant_id)
        
        if 'error' in report:
            return error_response(report['error'], 404)
        
        return success_response(report, "获取训练报告成功")
        
    except Exception as e:
        logger.error(f"获取三阶段训练报告失败: {str(e)}")
        return error_response("获取训练报告失败", 500)


# ==============================================================================
# 统计端点
# ==============================================================================

@three_stage_training_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_statistics():
    """
    获取三阶段训练统计信息
    
    Returns:
        统计信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = _get_tenant_id()
        
        service = _get_service()
        stats = service.get_statistics(tenant_id, user_id)
        
        return success_response(stats, "获取统计信息成功")
        
    except Exception as e:
        logger.error(f"获取三阶段训练统计信息失败: {str(e)}")
        return error_response("获取统计信息失败", 500)


# ==============================================================================
# 辅助函数
# ==============================================================================

def _build_config(data: dict) -> dict:
    """构建训练配置
    
    支持新格式（stages对象）和旧格式（pt_enabled等字段）
    
    Args:
        data: 请求数据
        
    Returns:
        规范化的配置
    """
    config = {
        'model_name': data.get('model_name', ''),
        'base_model_path': data.get('base_model', data.get('base_model_path', 'gpt2')),
        'output_dir': data.get('output_dir', f"./output/{data.get('model_name', 'model')}"),
        'pass_model_between_stages': data.get('pass_model_between_stages', True),
        'stages': {}
    }
    
    # 支持两种数据格式：新格式（stages对象）和旧格式（pt_enabled等）
    if 'stages' in data:
        # 新格式：使用stages对象
        stages_data = data['stages']
        
        # 预训练阶段
        pretrain_config = stages_data.get('pretrain') or stages_data.get('pt', {})
        if pretrain_config.get('enabled', False):
            config['stages']['pt'] = {
                'enabled': True,
                'epochs': pretrain_config.get('epochs', 3),
                'batch_size': pretrain_config.get('batch_size', 8),
                'learning_rate': pretrain_config.get('learning_rate', 2e-5),
                'data_path': pretrain_config.get('data_path', './data/pt')
            }
        
        # 微调阶段
        finetune_config = stages_data.get('finetune') or stages_data.get('sft', {})
        if finetune_config.get('enabled', False):
            config['stages']['sft'] = {
                'enabled': True,
                'epochs': finetune_config.get('epochs', 3),
                'batch_size': finetune_config.get('batch_size', 8),
                'learning_rate': finetune_config.get('learning_rate', 2e-5),
                'data_path': finetune_config.get('data_path', './data/sft')
            }
        
        # 偏好学习阶段（支持distillation别名）
        preference_config = stages_data.get('preference') or stages_data.get('distillation') or stages_data.get('dpo', {})
        if preference_config.get('enabled', False):
            config['stages']['dpo'] = {
                'enabled': True,
                'epochs': preference_config.get('epochs', 3),
                'batch_size': preference_config.get('batch_size', 8),
                'learning_rate': preference_config.get('learning_rate', 2e-5),
                'beta': preference_config.get('beta', 0.1),
                'data_path': preference_config.get('data_path', './data/dpo')
            }
    else:
        # 旧格式：使用pt_enabled等字段
        if data.get('pt_enabled', False):
            config['stages']['pt'] = {
                'enabled': True,
                'epochs': data.get('pt_epochs', 3),
                'batch_size': data.get('pt_batch_size', 8),
                'learning_rate': data.get('pt_learning_rate', 2e-5),
                'data_path': data.get('pt_data_path', './data/pt')
            }
        
        if data.get('sft_enabled', False):
            config['stages']['sft'] = {
                'enabled': True,
                'epochs': data.get('sft_epochs', 3),
                'batch_size': data.get('sft_batch_size', 8),
                'learning_rate': data.get('sft_learning_rate', 2e-5),
                'data_path': data.get('sft_data_path', './data/sft')
            }
        
        if data.get('dpo_enabled', False):
            config['stages']['dpo'] = {
                'enabled': True,
                'epochs': data.get('dpo_epochs', 3),
                'batch_size': data.get('dpo_batch_size', 8),
                'learning_rate': data.get('dpo_learning_rate', 2e-5),
                'beta': data.get('dpo_beta', 0.1),
                'data_path': data.get('dpo_data_path', './data/dpo')
            }
    
    return config

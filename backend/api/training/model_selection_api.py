"""模型选择API

提供模型选择和配置相关的API接口，支持租户隔离和持久化。
"""

import sys
import os
from typing import Dict, Any, List
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError, BusinessLogicError
from backend.utils.response import success_response, error_response
from backend.services.model_selection_service import (
    get_model_selection_service,
    ModelRecommendation,
    ModelConfiguration
)

# 创建蓝图
model_selection_bp = Blueprint('model_selection', __name__, url_prefix='/api/v1/training')
logger = logging.getLogger(__name__)

# 初始化服务（使用内存存储以便开发测试）
model_selection_service = get_model_selection_service(use_memory_storage=True)


@model_selection_bp.route('/models/recommend', methods=['POST'])
@jwt_required()
def recommend_models():
    """推荐模型
    
    Request Body:
        task_type: 任务类型 (必填)
        requirements: 硬件和环境要求 (可选)
        performance_requirements: 性能要求 (可选)
        
    Returns:
        推荐的模型列表
    """
    try:
        # 获取当前用户ID和租户ID
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        # 获取请求数据
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
            
        # 提取参数
        task_type = data.get('task_type')
        requirements = data.get('requirements')
        performance_requirements = data.get('performance_requirements')
        
        if not task_type:
            return error_response("任务类型不能为空", 400)
        
        logger.info(f"用户 {user_id} 请求推荐模型, task_type={task_type}, tenant_id={tenant_id}")
            
        # 推荐模型
        recommendations = model_selection_service.recommend_models(
            task_type=task_type,
            requirements=requirements,
            performance_requirements=performance_requirements,
            tenant_id=tenant_id,
            user_id=user_id
        )
        
        # 转换为可序列化的格式
        recommendations_data = []
        for rec in recommendations:
            rec_data = {
                'model_name': rec.model_name,
                'framework': rec.framework.value,
                'model_type': rec.model_type.value,
                'description': rec.description,
                'confidence': rec.confidence,
                'recommended_for': rec.recommended_for,
                'performance_metrics': rec.performance_metrics
            }
            recommendations_data.append(rec_data)
            
        return success_response(
            data={
                'recommendations': recommendations_data,
                'total': len(recommendations_data),
                'task_type': task_type
            },
            message="模型推荐完成"
        )
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"模型推荐失败: {str(e)}")
        return error_response(f"模型推荐失败: {str(e)}", 500)


@model_selection_bp.route('/models/<model_name>/config', methods=['POST'])
@jwt_required()
def get_model_configuration(model_name: str):
    """获取模型配置
    
    Args:
        model_name: 模型名称
        
    Request Body:
        task_type: 任务类型 (必填)
        dataset_info: 数据集信息 (可选)
        save_configuration: 是否保存配置 (可选, 默认true)
        
    Returns:
        模型配置
    """
    try:
        # 获取当前用户ID和租户ID
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        # 获取请求数据
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
            
        # 提取参数
        task_type = data.get('task_type')
        dataset_info = data.get('dataset_info')
        save_configuration = data.get('save_configuration', True)
        
        if not task_type:
            return error_response("任务类型不能为空", 400)
        
        logger.info(f"用户 {user_id} 请求获取模型配置, model={model_name}, task_type={task_type}")
            
        # 获取模型配置
        config = model_selection_service.get_model_configuration(
            model_name=model_name,
            task_type=task_type,
            dataset_info=dataset_info,
            tenant_id=tenant_id,
            user_id=user_id,
            save_configuration=save_configuration
        )
        
        # 转换为可序列化的格式
        config_data = {
            'model_name': config.model_name,
            'framework': config.framework.value,
            'model_type': config.model_type.value,
            'hyperparameters': config.hyperparameters,
            'training_config': config.training_config,
            'hardware_config': config.hardware_config
        }
            
        return success_response(
            data=config_data,
            message="模型配置获取成功"
        )
        
    except ValidationError as e:
        return error_response(str(e), 400)
    except Exception as e:
        logger.error(f"获取模型配置失败: {str(e)}")
        return error_response(f"获取模型配置失败: {str(e)}", 500)


@model_selection_bp.route('/models/search', methods=['GET'])
@jwt_required()
def search_models():
    """搜索模型
    
    Query Parameters:
        q: 搜索关键词 (必填)
        limit: 返回数量限制 (可选, 默认10)
        
    Returns:
        搜索结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        # 获取查询参数
        query = request.args.get('q', '')
        limit = request.args.get('limit', 10, type=int)
        
        if not query:
            return error_response("查询参数不能为空", 400)
        
        logger.info(f"用户 {user_id} 搜索模型, query={query}")
            
        # 搜索模型
        results = model_selection_service.search_models(query, limit)
        
        return success_response(
            data={
                'results': results,
                'total': len(results),
                'query': query
            },
            message="模型搜索完成"
        )
        
    except Exception as e:
        logger.error(f"模型搜索失败: {str(e)}")
        return error_response(f"模型搜索失败: {str(e)}", 500)


@model_selection_bp.route('/models/task-types', methods=['GET'])
@jwt_required()
def get_task_types():
    """获取支持的任务类型
    
    Returns:
        任务类型列表
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        task_types = model_selection_service.get_task_types(tenant_id)
        
        return success_response(
            data={
                'task_types': task_types,
                'total': len(task_types)
            },
            message="获取任务类型成功"
        )
        
    except Exception as e:
        logger.error(f"获取任务类型失败: {str(e)}")
        return error_response(f"获取任务类型失败: {str(e)}", 500)


@model_selection_bp.route('/models/by-task/<task_type>', methods=['GET'])
@jwt_required()
def get_models_for_task(task_type: str):
    """获取任务类型的可用模型
    
    Args:
        task_type: 任务类型
        
    Query Parameters:
        framework: 框架过滤 (可选)
        limit: 返回数量限制 (可选, 默认50)
        
    Returns:
        模型列表
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        framework = request.args.get('framework')
        limit = request.args.get('limit', 50, type=int)
        
        models = model_selection_service.get_models_for_task(
            task_type=task_type,
            tenant_id=tenant_id,
            framework=framework,
            limit=limit
        )
        
        return success_response(
            data={
                'models': models,
                'total': len(models),
                'task_type': task_type
            },
            message="获取模型列表成功"
        )
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {str(e)}")
        return error_response(f"获取模型列表失败: {str(e)}", 500)


@model_selection_bp.route('/recommendations/history', methods=['GET'])
@jwt_required()
def get_recommendation_history():
    """获取推荐历史
    
    Query Parameters:
        task_type: 任务类型过滤 (可选)
        limit: 返回数量限制 (可选, 默认20)
        
    Returns:
        推荐历史列表
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        task_type = request.args.get('task_type')
        limit = request.args.get('limit', 20, type=int)
        
        history = model_selection_service.get_recommendation_history(
            user_id=user_id,
            tenant_id=tenant_id,
            task_type=task_type,
            limit=limit
        )
        
        return success_response(
            data={
                'history': history,
                'total': len(history)
            },
            message="获取推荐历史成功"
        )
        
    except Exception as e:
        logger.error(f"获取推荐历史失败: {str(e)}")
        return error_response(f"获取推荐历史失败: {str(e)}", 500)


@model_selection_bp.route('/recommendations/<recommendation_id>/feedback', methods=['POST'])
@jwt_required()
def submit_recommendation_feedback(recommendation_id: str):
    """提交推荐反馈
    
    Args:
        recommendation_id: 推荐ID
        
    Request Body:
        selected_model: 用户选择的模型 (可选)
        feedback_score: 反馈评分 1-5 (可选)
        feedback_comment: 反馈评论 (可选)
        is_helpful: 推荐是否有帮助 (可选)
        
    Returns:
        操作结果
    """
    try:
        user_id = get_jwt_identity()
        
        data = request.get_json() or {}
        
        selected_model = data.get('selected_model')
        feedback_score = data.get('feedback_score')
        feedback_comment = data.get('feedback_comment')
        is_helpful = data.get('is_helpful')
        
        # 验证评分范围
        if feedback_score is not None and (feedback_score < 1 or feedback_score > 5):
            return error_response("反馈评分必须在1-5之间", 400)
        
        logger.info(f"用户 {user_id} 提交推荐反馈, recommendation_id={recommendation_id}")
        
        success = model_selection_service.submit_recommendation_feedback(
            recommendation_id=recommendation_id,
            selected_model=selected_model,
            feedback_score=feedback_score,
            feedback_comment=feedback_comment,
            is_helpful=is_helpful
        )
        
        if success:
            return success_response(
                data={'recommendation_id': recommendation_id},
                message="反馈提交成功"
            )
        else:
            return error_response("反馈提交失败", 400)
            
    except Exception as e:
        logger.error(f"提交反馈失败: {str(e)}")
        return error_response(f"提交反馈失败: {str(e)}", 500)


@model_selection_bp.route('/recommendations/statistics', methods=['GET'])
@jwt_required()
def get_recommendation_statistics():
    """获取推荐统计信息
    
    Query Parameters:
        task_type: 任务类型过滤 (可选)
        
    Returns:
        统计信息
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        if not tenant_id:
            return error_response("获取统计信息需要提供租户ID", 400)
        
        task_type = request.args.get('task_type')
        
        stats = model_selection_service.get_recommendation_statistics(
            tenant_id=tenant_id,
            user_id=user_id,
            task_type=task_type
        )
        
        return success_response(
            data=stats,
            message="获取统计信息成功"
        )
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {str(e)}")
        return error_response(f"获取统计信息失败: {str(e)}", 500)


@model_selection_bp.route('/configurations/history', methods=['GET'])
@jwt_required()
def get_configuration_history():
    """获取配置历史
    
    Query Parameters:
        model_name: 模型名称过滤 (可选)
        task_type: 任务类型过滤 (可选)
        limit: 返回数量限制 (可选, 默认20)
        
    Returns:
        配置历史列表
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        model_name = request.args.get('model_name')
        task_type = request.args.get('task_type')
        limit = request.args.get('limit', 20, type=int)
        
        history = model_selection_service.get_configuration_history(
            user_id=user_id,
            tenant_id=tenant_id,
            model_name=model_name,
            task_type=task_type,
            limit=limit
        )
        
        return success_response(
            data={
                'history': history,
                'total': len(history)
            },
            message="获取配置历史成功"
        )
        
    except Exception as e:
        logger.error(f"获取配置历史失败: {str(e)}")
        return error_response(f"获取配置历史失败: {str(e)}", 500)


@model_selection_bp.route('/configurations', methods=['POST'])
@jwt_required()
def save_custom_configuration():
    """保存自定义配置
    
    Request Body:
        model_name: 模型名称 (必填)
        task_type: 任务类型 (必填)
        hyperparameters: 超参数 (必填)
        training_config: 训练配置 (必填)
        hardware_config: 硬件配置 (必填)
        is_default: 是否设为默认 (可选, 默认false)
        tags: 标签 (可选)
        
    Returns:
        保存的配置记录
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        if not tenant_id:
            return error_response("保存配置需要提供租户ID", 400)
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        model_name = data.get('model_name')
        task_type = data.get('task_type')
        hyperparameters = data.get('hyperparameters')
        training_config = data.get('training_config')
        hardware_config = data.get('hardware_config')
        
        if not all([model_name, task_type, hyperparameters, training_config, hardware_config]):
            return error_response("缺少必要参数", 400)
        
        is_default = data.get('is_default', False)
        tags = data.get('tags')
        
        logger.info(f"用户 {user_id} 保存自定义配置, model={model_name}, task={task_type}")
        
        configuration = model_selection_service.save_custom_configuration(
            model_name=model_name,
            task_type=task_type,
            hyperparameters=hyperparameters,
            training_config=training_config,
            hardware_config=hardware_config,
            tenant_id=tenant_id,
            user_id=user_id,
            is_default=is_default,
            tags=tags
        )
        
        if configuration:
            return success_response(
                data=configuration,
                message="配置保存成功"
            )
        else:
            return error_response("配置保存失败", 500)
            
    except Exception as e:
        logger.error(f"保存配置失败: {str(e)}")
        return error_response(f"保存配置失败: {str(e)}", 500)


@model_selection_bp.route('/configurations/<configuration_id>', methods=['DELETE'])
@jwt_required()
def delete_configuration(configuration_id: str):
    """删除配置记录
    
    Args:
        configuration_id: 配置ID
        
    Returns:
        操作结果
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        logger.info(f"用户 {user_id} 删除配置, configuration_id={configuration_id}")
        
        success = model_selection_service.delete_configuration(
            configuration_id=configuration_id,
            tenant_id=tenant_id
        )
        
        if success:
            return success_response(
                data={'configuration_id': configuration_id},
                message="配置删除成功"
            )
        else:
            return error_response("配置删除失败或配置不存在", 404)
            
    except Exception as e:
        logger.error(f"删除配置失败: {str(e)}")
        return error_response(f"删除配置失败: {str(e)}", 500)


@model_selection_bp.route('/catalog/models', methods=['POST'])
@jwt_required()
def add_model_to_catalog():
    """添加模型到目录
    
    Request Body:
        model_name: 模型名称 (必填)
        task_type: 任务类型 (必填)
        framework: 框架 (必填)
        model_type: 模型类型 (必填)
        description: 描述 (必填)
        performance_metrics: 性能指标 (必填)
        hardware_requirements: 硬件要求 (必填)
        default_hyperparameters: 默认超参数 (可选)
        tags: 标签 (可选)
        
    Returns:
        创建的目录条目
    """
    try:
        user_id = get_jwt_identity()
        tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
        
        if not tenant_id:
            return error_response("添加模型需要提供租户ID", 400)
        
        data = request.get_json()
        if not data:
            return error_response("请求数据不能为空", 400)
        
        model_name = data.get('model_name')
        task_type = data.get('task_type')
        framework = data.get('framework')
        model_type = data.get('model_type')
        description = data.get('description')
        performance_metrics = data.get('performance_metrics')
        hardware_requirements = data.get('hardware_requirements')
        
        if not all([model_name, task_type, framework, model_type, description, 
                   performance_metrics, hardware_requirements]):
            return error_response("缺少必要参数", 400)
        
        default_hyperparameters = data.get('default_hyperparameters')
        tags = data.get('tags')
        
        logger.info(f"用户 {user_id} 添加模型到目录, model={model_name}, task={task_type}")
        
        entry = model_selection_service.add_model_to_catalog(
            model_name=model_name,
            task_type=task_type,
            framework=framework,
            model_type=model_type,
            description=description,
            performance_metrics=performance_metrics,
            hardware_requirements=hardware_requirements,
            tenant_id=tenant_id,
            default_hyperparameters=default_hyperparameters,
            tags=tags
        )
        
        if entry:
            return success_response(
                data=entry,
                message="模型添加成功"
            )
        else:
            return error_response("模型添加失败", 500)
            
    except Exception as e:
        logger.error(f"添加模型失败: {str(e)}")
        return error_response(f"添加模型失败: {str(e)}", 500)

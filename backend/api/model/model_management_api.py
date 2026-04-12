#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型管理API接口

提供模型管理相关的高级API接口：
- 获取用户模型列表
- 创建/删除用户模型
- 获取模型性能指标
- 模型比较
- 模型导出/导入
- 模型克隆
- 获取训练历史
- 模型验证

架构调用关系：
API层 (本模块)
    -> Service层 (model_management_service.py)
        -> Repository层 (model_management_repository.py)
        -> ModelService, TrainingHistoryService
"""

from flask import Blueprint, request
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# 创建蓝图
model_management_bp = Blueprint('model_management', __name__, url_prefix='/api/v1/model-management')

# 延迟初始化的服务实例
_management_service = None


def _get_management_service():
    """获取模型管理服务实例（延迟初始化）"""
    global _management_service
    if _management_service is None:
        try:
            from backend.services.model_management_service import get_management_service
            _management_service = get_management_service()
        except Exception as e:
            logger.error(f"Failed to initialize ModelManagementService: {e}")
            raise
    return _management_service


def _success_response(data: Any = None, message: str = "操作成功", code: int = 200) -> tuple:
    """统一成功响应格式"""
    return {
        'success': True,
        'message': message,
        'data': data
    }, code


def _error_response(message: str, code: int = 400, errors: Any = None) -> tuple:
    """统一错误响应格式"""
    response = {
        'success': False,
        'message': message,
        'data': None
    }
    if errors:
        response['errors'] = errors
    return response, code


# ==========================================================================
# 用户模型管理
# ==========================================================================

@model_management_bp.route('/user', methods=['GET'])
@jwt_required()
def get_user_models():
    """获取用户模型列表
    
    Query Parameters:
        limit: 返回数量限制 (默认50)
        offset: 偏移量 (默认0)
        status: 状态过滤
        model_type: 类型过滤
        search: 搜索关键词
    
    Returns:
        用户模型列表
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        # 获取查询参数
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        status = request.args.get('status')
        model_type = request.args.get('model_type')
        search = request.args.get('search')
        
        # 限制参数范围
        limit = min(max(limit, 1), 100)
        offset = max(offset, 0)
        
        # 获取用户模型列表
        if service._model_service:
            models = service._model_service.list_models(
                user_id=current_user_id,
                limit=limit,
                offset=offset,
                status=status,
                model_type=model_type,
                search=search
            )
            
            # 转换为字典列表
            models_data = []
            for model in models:
                if hasattr(model, 'to_dict'):
                    models_data.append(model.to_dict())
                else:
                    models_data.append({
                        'id': getattr(model, 'id', None),
                        'name': getattr(model, 'name', None),
                        'status': getattr(model, 'status', None),
                    })
            
            return _success_response({
                'models': models_data,
                'limit': limit,
                'offset': offset,
                'count': len(models_data)
            }, "获取用户模型列表成功")
        
        return _success_response({
            'models': [],
            'limit': limit,
            'offset': offset,
            'count': 0
        }, "获取用户模型列表成功")
        
    except Exception as e:
        logger.error(f"获取用户模型列表失败: {str(e)}")
        return _error_response(f"获取用户模型列表失败: {str(e)}", 500)


@model_management_bp.route('/user', methods=['POST'])
@jwt_required()
def create_user_model():
    """创建用户模型
    
    Request Body:
        name: 模型名称 (必需)
        description: 模型描述
        modelType: 模型类型 (默认llm)
        baseModel: 基础模型
        config: 配置信息
        tags: 标签列表
    
    Returns:
        创建的模型信息
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        # 验证请求数据
        data = request.get_json()
        if not data:
            return _error_response("请求数据不能为空", 400)
        
        # 提取参数
        name = data.get('name')
        if not name:
            return _error_response("模型名称不能为空", 400)
            
        description = data.get('description')
        model_type = data.get('modelType', 'llm')
        framework = data.get('framework', 'pytorch')
        config = data.get('config', {})
        
        # 创建模型
        if service._model_service:
            model = service._model_service.create_model(
                user_id=current_user_id,
                name=name,
                description=description,
                model_type=model_type,
                architecture='transformer',
                framework=framework,
                storage_path='',
                config=config
            )
            
            model_dict = model.to_dict() if hasattr(model, 'to_dict') else {
                'id': model.id,
                'name': model.name,
            }
            
            return _success_response(model_dict, "模型创建成功", 201)
        
        return _error_response("模型服务不可用", 503)
        
    except Exception as e:
        logger.error(f"创建用户模型失败: {str(e)}")
        return _error_response(f"创建用户模型失败: {str(e)}", 500)


@model_management_bp.route('/user/<model_id>', methods=['DELETE'])
@jwt_required()
def delete_user_model(model_id: str):
    """删除用户模型
    
    Args:
        model_id: 模型ID
    
    Returns:
        删除结果
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        if service._model_service:
            # 获取模型验证权限
            model = service._model_service.get_model(model_id)
            
            if not model:
                return _error_response("模型不存在", 404)
                
            # 检查权限
            if getattr(model, 'user_id', None) != current_user_id:
                return _error_response("无权限删除该模型", 403)
                
            # 删除模型
            success = service._model_service.delete_model(model_id, user_id=current_user_id)
            
            if success:
                return _success_response(None, "模型删除成功", 200)
            else:
                return _error_response("删除模型失败", 500)
        
        return _error_response("模型服务不可用", 503)
        
    except Exception as e:
        logger.error(f"删除用户模型失败: {str(e)}")
        return _error_response(f"删除用户模型失败: {str(e)}", 500)


# ==========================================================================
# 模型性能
# ==========================================================================

@model_management_bp.route('/<model_id>/performance', methods=['GET'])
@jwt_required()
def get_model_performance(model_id: str):
    """获取模型性能指标
    
    Args:
        model_id: 模型ID
    
    Query Parameters:
        include_history: 是否包含历史记录
    
    Returns:
        模型性能指标数据
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        include_history = request.args.get('include_history', 'false').lower() == 'true'
        
        performance_data = service.get_model_performance(
            model_id=model_id,
            user_id=current_user_id,
            include_history=include_history
        )
        
        return _success_response(performance_data, "获取模型性能指标成功")
        
    except ValueError as e:
        return _error_response(str(e), 400)
    except Exception as e:
        logger.error(f"获取模型性能指标失败: {str(e)}")
        return _error_response(f"获取模型性能指标失败: {str(e)}", 500)


@model_management_bp.route('/<model_id>/performance', methods=['POST'])
@jwt_required()
def record_model_performance(model_id: str):
    """记录模型性能指标
    
    Args:
        model_id: 模型ID
    
    Request Body:
        accuracy: 准确率
        precision: 精确率
        recall: 召回率
        f1_score: F1分数
        loss: 损失值
        ... 其他指标
    
    Returns:
        创建的性能记录
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        data = request.get_json()
        if not data:
            return _error_response("请求数据不能为空", 400)
        
        record = service.record_model_performance(
            model_id=model_id,
            user_id=current_user_id,
            metrics=data
        )
        
        return _success_response(record, "性能指标记录成功", 201)
        
    except ValueError as e:
        return _error_response(str(e), 400)
    except Exception as e:
        logger.error(f"记录模型性能指标失败: {str(e)}")
        return _error_response(f"记录模型性能指标失败: {str(e)}", 500)


# ==========================================================================
# 模型比较
# ==========================================================================

@model_management_bp.route('/compare', methods=['POST'])
@jwt_required()
def compare_models():
    """比较多个模型
    
    Request Body:
        model_ids: 模型ID列表 (必需，至少2个)
        metrics_to_compare: 要比较的指标列表
        comparison_config: 比较配置
    
    Returns:
        比较结果，包含各模型指标和胜出者
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        # 验证请求数据
        data = request.get_json()
        if not data:
            return _error_response("请求数据不能为空", 400)
            
        model_ids = data.get('model_ids', [])
        if not model_ids or not isinstance(model_ids, list):
            return _error_response("必须提供模型ID列表", 400)
        
        if len(model_ids) < 2:
            return _error_response("至少需要2个模型进行比较", 400)
        
        if len(model_ids) > 10:
            return _error_response("最多比较10个模型", 400)
        
        metrics_to_compare = data.get('metrics_to_compare')
        comparison_config = data.get('comparison_config', {})
        
        comparison_result = service.compare_models(
            model_ids=model_ids,
            user_id=current_user_id,
            metrics_to_compare=metrics_to_compare,
            comparison_config=comparison_config
        )
        
        return _success_response(comparison_result, "模型比较完成")
        
    except ValueError as e:
        return _error_response(str(e), 400)
    except Exception as e:
        logger.error(f"模型比较失败: {str(e)}")
        return _error_response(f"模型比较失败: {str(e)}", 500)


# ==========================================================================
# 模型导出
# ==========================================================================

@model_management_bp.route('/<model_id>/export', methods=['GET'])
@jwt_required()
def export_model(model_id: str):
    """导出模型
    
    Args:
        model_id: 模型ID
    
    Query Parameters:
        format: 导出格式 (默认pytorch)
    
    Returns:
        导出信息，包含导出路径
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        # 获取导出格式
        export_format = request.args.get('format', 'pytorch')
        
        export_info = service.export_model(
            model_id=model_id,
            user_id=current_user_id,
            export_format=export_format
        )
        
        return _success_response(export_info, "模型导出准备完成")
        
    except ValueError as e:
        return _error_response(str(e), 400)
    except Exception as e:
        logger.error(f"导出模型失败: {str(e)}")
        return _error_response(f"导出模型失败: {str(e)}", 500)


@model_management_bp.route('/export/formats', methods=['GET'])
@jwt_required()
def get_export_formats():
    """获取支持的导出格式
    
    Returns:
        支持的导出格式列表
    """
    try:
        service = _get_management_service()
        formats = service.get_supported_export_formats()
        
        return _success_response({
            'formats': formats,
            'default': 'onnx'
        }, "获取导出格式成功")
        
    except Exception as e:
        logger.error(f"获取导出格式失败: {str(e)}")
        return _error_response(f"获取导出格式失败: {str(e)}", 500)


# ==========================================================================
# 模型导入
# ==========================================================================

@model_management_bp.route('/import', methods=['POST'])
@jwt_required()
def import_model():
    """导入模型
    
    Request Body:
        model_name: 模型名称 (必需)
        import_source: 导入来源 (local/url/huggingface/s3/gcs)
        source_path: 来源路径 (本地/云存储)
        source_url: 来源URL (URL/HuggingFace)
        model_type: 模型类型
        model_framework: 模型框架
        import_config: 导入配置
    
    Returns:
        导入结果信息
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        data = request.get_json()
        if not data:
            return _error_response("请求数据不能为空", 400)
        
        model_name = data.get('model_name')
        if not model_name:
            return _error_response("模型名称不能为空", 400)
        
        import_source = data.get('import_source', 'local')
        source_path = data.get('source_path')
        source_url = data.get('source_url')
        model_type = data.get('model_type')
        model_framework = data.get('model_framework')
        import_config = data.get('import_config', {})
        
        import_result = service.import_model(
            user_id=current_user_id,
            model_name=model_name,
            import_source=import_source,
            source_path=source_path,
            source_url=source_url,
            model_type=model_type,
            model_framework=model_framework,
            import_config=import_config
        )
        
        return _success_response(import_result, "模型导入完成", 201)
        
    except ValueError as e:
        return _error_response(str(e), 400)
    except Exception as e:
        logger.error(f"导入模型失败: {str(e)}")
        return _error_response(f"导入模型失败: {str(e)}", 500)


@model_management_bp.route('/import/<import_id>/status', methods=['GET'])
@jwt_required()
def get_import_status(import_id: str):
    """获取导入状态
    
    Args:
        import_id: 导入任务ID
    
    Returns:
        导入状态信息
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        status = service.get_import_status(import_id, current_user_id)
        
        if not status:
            return _error_response("导入任务不存在或无权限", 404)
        
        return _success_response(status, "获取导入状态成功")
        
    except Exception as e:
        logger.error(f"获取导入状态失败: {str(e)}")
        return _error_response(f"获取导入状态失败: {str(e)}", 500)


@model_management_bp.route('/import/formats', methods=['GET'])
@jwt_required()
def get_import_formats():
    """获取支持的导入格式
    
    Returns:
        支持的导入格式列表
    """
    try:
        service = _get_management_service()
        formats = service.get_supported_import_formats()
        
        return _success_response({
            'formats': formats,
            'default': 'pytorch'
        }, "获取导入格式成功")
        
    except Exception as e:
        logger.error(f"获取导入格式失败: {str(e)}")
        return _error_response(f"获取导入格式失败: {str(e)}", 500)


# ==========================================================================
# 模型克隆
# ==========================================================================

@model_management_bp.route('/<model_id>/clone', methods=['POST'])
@jwt_required()
def clone_model(model_id: str):
    """克隆模型
    
    Args:
        model_id: 源模型ID
    
    Request Body:
        name: 新模型名称 (必需)
        include_versions: 是否包含版本历史
    
    Returns:
        克隆的模型信息
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        # 验证请求数据
        data = request.get_json()
        if not data:
            return _error_response("请求数据不能为空", 400)
            
        new_name = data.get('name')
        if not new_name:
            return _error_response("新模型名称不能为空", 400)
        
        include_versions = data.get('include_versions', False)
        
        cloned_model = service.clone_model(
            model_id=model_id,
            user_id=current_user_id,
            new_name=new_name,
            include_versions=include_versions
        )
        
        return _success_response(cloned_model, "模型克隆成功", 201)
        
    except ValueError as e:
        return _error_response(str(e), 400)
    except Exception as e:
        logger.error(f"克隆模型失败: {str(e)}")
        return _error_response(f"克隆模型失败: {str(e)}", 500)


# ==========================================================================
# 训练历史
# ==========================================================================

@model_management_bp.route('/<model_id>/training-history', methods=['GET'])
@jwt_required()
def get_model_training_history(model_id: str):
    """获取模型训练历史
    
    Args:
        model_id: 模型ID
    
    Query Parameters:
        limit: 返回数量限制
    
    Returns:
        训练历史记录列表
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        limit = request.args.get('limit', 50, type=int)
        limit = min(max(limit, 1), 100)
        
        training_history = service.get_model_training_history(
            model_id=model_id,
            user_id=current_user_id,
            limit=limit
        )
        
        return _success_response(training_history, "获取模型训练历史成功")
        
    except ValueError as e:
        return _error_response(str(e), 400)
    except Exception as e:
        logger.error(f"获取模型训练历史失败: {str(e)}")
        return _error_response(f"获取模型训练历史失败: {str(e)}", 500)


# ==========================================================================
# 模型验证
# ==========================================================================

@model_management_bp.route('/<model_id>/validate', methods=['POST'])
@jwt_required()
def validate_model(model_id: str):
    """验证模型
    
    Args:
        model_id: 模型ID
    
    Request Body:
        test_data: 测试数据 (必需)
        validation_config: 验证配置
    
    Returns:
        验证结果
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        # 验证请求数据
        data = request.get_json()
        if not data:
            return _error_response("请求数据不能为空", 400)
            
        test_data = data.get('test_data')
        if not test_data:
            return _error_response("测试数据不能为空", 400)
        
        validation_config = data.get('validation_config', {})
        
        validation_result = service.validate_model(
            model_id=model_id,
            user_id=current_user_id,
            test_data=test_data,
            validation_config=validation_config
        )
        
        return _success_response(validation_result, "模型验证完成")
        
    except ValueError as e:
        return _error_response(str(e), 400)
    except Exception as e:
        logger.error(f"验证模型失败: {str(e)}")
        return _error_response(f"验证模型失败: {str(e)}", 500)


@model_management_bp.route('/<model_id>/validation-history', methods=['GET'])
@jwt_required()
def get_validation_history(model_id: str):
    """获取模型验证历史
    
    Args:
        model_id: 模型ID
    
    Query Parameters:
        limit: 返回数量限制
    
    Returns:
        验证历史记录列表
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        limit = request.args.get('limit', 50, type=int)
        limit = min(max(limit, 1), 100)
        
        validation_history = service.get_validation_history(
            model_id=model_id,
            user_id=current_user_id,
            limit=limit
        )
        
        return _success_response(validation_history, "获取验证历史成功")
        
    except ValueError as e:
        return _error_response(str(e), 400)
    except Exception as e:
        logger.error(f"获取验证历史失败: {str(e)}")
        return _error_response(f"获取验证历史失败: {str(e)}", 500)


# ==========================================================================
# 用户摘要
# ==========================================================================

@model_management_bp.route('/summary', methods=['GET'])
@jwt_required()
def get_user_model_summary():
    """获取用户模型摘要
    
    Returns:
        模型统计摘要，包括按状态/类型/框架的分布
    """
    try:
        current_user_id = get_jwt_identity()
        service = _get_management_service()
        
        summary = service.get_user_model_summary(current_user_id)
        
        return _success_response(summary, "获取模型摘要成功")
        
    except Exception as e:
        logger.error(f"获取模型摘要失败: {str(e)}")
        return _error_response(f"获取模型摘要失败: {str(e)}", 500)


# ==========================================================================
# 模块初始化和清理
# ==========================================================================

def init_management_api():
    """初始化模型管理 API"""
    global _management_service
    try:
        from backend.services.model_management_service import get_management_service
        _management_service = get_management_service()
        logger.info("ModelManagement API initialized successfully")
    except Exception as e:
        logger.warning(f"ModelManagement API initialization warning: {e}")


def cleanup_management_api():
    """清理模型管理 API 资源"""
    global _management_service
    _management_service = None
    logger.info("ModelManagement API resources cleaned up")


# ==================== 导出 ====================

__all__ = [
    'model_management_bp',
    'init_management_api',
    'cleanup_management_api',
]

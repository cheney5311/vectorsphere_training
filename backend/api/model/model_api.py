#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型管理 API

提供完整的模型管理 REST API 接口：
- 模型CRUD操作
- 版本管理
- 模型部署
- 模型导出
- 模型统计
"""

import sys
import os
import logging
from functools import wraps
from datetime import datetime

from flask import Blueprint, request, jsonify, g
from flask_jwt_extended import jwt_required, get_jwt_identity

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError
from backend.utils.response import success_response, error_response
from backend.services.model_service import ModelService
from backend.repositories.model_repository import ModelRepository
from backend.modules.model.exceptions.model_exceptions import ModelNotFoundError, ModelValidationError

logger = logging.getLogger(__name__)

# 创建蓝图
model_bp = Blueprint('model', __name__, url_prefix='/api/v1/models')

# 初始化服务（懒加载）
_model_service = None


def get_model_service() -> ModelService:
    """获取模型服务实例"""
    global _model_service
    if _model_service is None:
        model_repository = ModelRepository()
        _model_service = ModelService(model_repository)
    return _model_service


def get_tenant_id():
    """获取当前租户ID"""
    return request.headers.get('X-Tenant-ID') or getattr(g, 'tenant_id', None)


def handle_api_errors(f):
    """API错误处理装饰器"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ModelNotFoundError as e:
            return error_response(str(e), 404)
        except ModelValidationError as e:
            return error_response(str(e), 400)
        except ValidationError as e:
            return error_response(str(e), 400)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}", exc_info=True)
            return error_response(f"服务器内部错误: {str(e)}", 500)
    return wrapper


# ==================== 模型CRUD接口 ====================

@model_bp.route('', methods=['POST'])
@jwt_required()
@handle_api_errors
def create_model():
    """创建模型
    
    Request Body:
        name: 模型名称 (必需)
        description: 模型描述
        version: 版本号 (默认1.0.0)
        type: 模型类型 (classification/regression/clustering等)
        architecture: 架构类型
        framework: 框架类型 (pytorch/tensorflow等)
        storage_path: 存储路径
        config: 配置信息
        tags: 标签列表
        category: 分类
    
    Returns:
        创建的模型信息
    """
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    if not data:
        return error_response("请求数据不能为空", 400)
    
    name = data.get('name')
    if not name:
        return error_response("模型名称不能为空", 400)
    
    service = get_model_service()
    model = service.create_model(
        user_id=user_id,
        name=name,
        description=data.get('description'),
        version=data.get('version', '1.0.0'),
        model_type=data.get('type', 'classification'),
        architecture=data.get('architecture', 'transformer'),
        framework=data.get('framework', 'pytorch'),
        storage_path=data.get('storage_path', ''),
        config=data.get('config', {}),
        training_session_id=data.get('training_session_id'),
        dataset_id=data.get('dataset_id'),
        tags=data.get('tags'),
        category=data.get('category'),
        tenant_id=get_tenant_id()
    )
    
    result = model.to_dict() if hasattr(model, 'to_dict') else {
        'id': model.id, 'name': model.name, 'version': model.version
    }
    
    return success_response(result, "模型创建成功", 201)


@model_bp.route('/<model_id>', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_model(model_id):
    """获取模型详情
    
    Path Parameters:
        model_id: 模型ID
    
    Query Parameters:
        include_versions: 是否包含版本信息 (默认false)
        include_events: 是否包含事件信息 (默认false)
    
    Returns:
        模型详细信息
    """
    user_id = get_jwt_identity()
    include_versions = request.args.get('include_versions', 'false').lower() == 'true'
    include_events = request.args.get('include_events', 'false').lower() == 'true'
    
    service = get_model_service()
    
    if include_versions or include_events:
        model_detail = service.get_model_detail(model_id)
        
        # 权限检查
        if model_detail.get('user_id') != user_id:
            return error_response("无权访问此模型", 403)
        
        if not include_versions and 'versions' in model_detail:
            del model_detail['versions']
        if not include_events and 'recent_events' in model_detail:
            del model_detail['recent_events']
        
        return success_response(model_detail)
    else:
        model = service.get_model(model_id)
        
        if not model:
            return error_response("模型不存在", 404)
        
        if model.user_id != user_id:
            return error_response("无权访问此模型", 403)
        
        result = model.to_dict() if hasattr(model, 'to_dict') else {
            'id': model.id,
            'name': model.name,
            'description': model.description,
            'version': model.version,
            'model_type': getattr(model, 'model_type', None),
            'framework': getattr(model, 'framework', None),
            'status': getattr(model, 'status', None),
            'config': model.config,
            'user_id': model.user_id,
            'created_at': model.created_at.isoformat() if model.created_at else None,
            'updated_at': model.updated_at.isoformat() if model.updated_at else None,
        }
        
        return success_response(result)


@model_bp.route('', methods=['GET'])
@jwt_required()
@handle_api_errors
def list_models():
    """获取模型列表
    
    Query Parameters:
        limit: 返回数量 (默认50, 最大100)
        offset: 偏移量 (默认0)
        status: 状态过滤
        type: 类型过滤
        framework: 框架过滤
        search: 搜索关键词
    
    Returns:
        模型列表
    """
    user_id = get_jwt_identity()
    
    limit = request.args.get('limit', 50, type=int)
    offset = request.args.get('offset', 0, type=int)
    status = request.args.get('status')
    model_type = request.args.get('type')
    framework = request.args.get('framework')
    search = request.args.get('search')
    
    service = get_model_service()
    models = service.list_models(
        user_id=user_id,
        limit=limit,
        offset=offset,
        status=status,
        model_type=model_type,
        framework=framework,
        search=search
    )
    
    models_data = []
    for model in models:
        if hasattr(model, 'to_dict'):
            models_data.append(model.to_dict())
        else:
            models_data.append({
                'id': model.id,
                'name': model.name,
                'version': model.version,
                'status': getattr(model, 'status', None),
                'model_type': getattr(model, 'model_type', None),
                'framework': getattr(model, 'framework', None),
                'created_at': model.created_at.isoformat() if model.created_at else None,
            })
    
    return success_response({
        'models': models_data,
        'limit': limit,
        'offset': offset,
        'count': len(models_data)
    })


@model_bp.route('/paginated', methods=['GET'])
@jwt_required()
@handle_api_errors
def list_models_paginated():
    """分页获取模型列表
    
    Query Parameters:
        page: 页码 (默认1)
        page_size: 每页数量 (默认20, 最大100)
        sort_by: 排序字段 (默认created_at)
        sort_order: 排序方向 (asc/desc, 默认desc)
        status: 状态过滤
        type: 类型过滤
    
    Returns:
        分页模型列表
    """
    user_id = get_jwt_identity()
    
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 20, type=int)
    sort_by = request.args.get('sort_by', 'created_at')
    sort_order = request.args.get('sort_order', 'desc')
    status = request.args.get('status')
    model_type = request.args.get('type')
    
    service = get_model_service()
    result = service.list_models_paginated(
        user_id=user_id,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_order=sort_order,
        status=status,
        model_type=model_type
    )
    
    return success_response(result)


@model_bp.route('/<model_id>', methods=['PUT'])
@jwt_required()
@handle_api_errors
def update_model(model_id):
    """更新模型
    
    Path Parameters:
        model_id: 模型ID
    
    Request Body:
        name: 模型名称
        description: 模型描述
        config: 配置信息
        tags: 标签列表
        category: 分类
    
    Returns:
        更新后的模型信息
    """
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    if not data:
        return error_response("请求数据不能为空", 400)
    
    service = get_model_service()
    
    # 权限检查
    model = service.get_model(model_id)
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    updated_model = service.update_model(
        model_id=model_id,
        name=data.get('name'),
        description=data.get('description'),
        config=data.get('config'),
        tags=data.get('tags'),
        category=data.get('category')
    )
    
    result = updated_model.to_dict() if hasattr(updated_model, 'to_dict') else {
        'id': updated_model.id, 'name': updated_model.name
    }
    
    return success_response(result, "模型更新成功")


@model_bp.route('/<model_id>', methods=['DELETE'])
@jwt_required()
@handle_api_errors
def delete_model(model_id):
    """删除模型
    
    Path Parameters:
        model_id: 模型ID
    
    Returns:
        删除结果
    """
    user_id = get_jwt_identity()
    
    service = get_model_service()
    
    # 权限检查
    model = service.get_model(model_id)
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    success = service.delete_model(model_id, user_id=user_id)
    
    if success:
        return success_response(None, "模型删除成功", 204)
    else:
        return error_response("删除模型失败", 500)


# ==================== 模型状态管理接口 ====================

@model_bp.route('/<model_id>/process', methods=['POST'])
@jwt_required()
@handle_api_errors
def process_model(model_id):
    """处理模型（开始训练/处理）
    
    Path Parameters:
        model_id: 模型ID
    
    Returns:
        处理后的模型信息
    """
    user_id = get_jwt_identity()
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    processed_model = service.process_model(model_id)
    
    result = processed_model.to_dict() if hasattr(processed_model, 'to_dict') else {
        'id': processed_model.id, 'status': getattr(processed_model, 'status', None)
    }
    
    return success_response(result, "模型处理已开始")


@model_bp.route('/<model_id>/validate', methods=['POST'])
@jwt_required()
@handle_api_errors
def validate_model(model_id):
    """验证模型
    
    Path Parameters:
        model_id: 模型ID
    
    Request Body:
        metrics: 验证指标
    
    Returns:
        验证后的模型信息
    """
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    metrics = data.get('metrics', {})
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    validated_model = service.validate_model(model_id, metrics)
    
    result = validated_model.to_dict() if hasattr(validated_model, 'to_dict') else {
        'id': validated_model.id, 'status': getattr(validated_model, 'status', None)
    }
    
    return success_response(result, "模型验证完成")


@model_bp.route('/<model_id>/ready', methods=['POST'])
@jwt_required()
@handle_api_errors
def mark_model_ready(model_id):
    """标记模型为就绪状态
    
    Path Parameters:
        model_id: 模型ID
    
    Returns:
        更新后的模型信息
    """
    user_id = get_jwt_identity()
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    ready_model = service.mark_model_ready(model_id)
    
    result = ready_model.to_dict() if hasattr(ready_model, 'to_dict') else {
        'id': ready_model.id, 'status': getattr(ready_model, 'status', None)
    }
    
    return success_response(result, "模型已标记为就绪")


@model_bp.route('/<model_id>/deploy', methods=['POST'])
@jwt_required()
@handle_api_errors
def deploy_model(model_id):
    """部署模型
    
    Path Parameters:
        model_id: 模型ID
    
    Request Body:
        config: 部署配置
            - replicas: 副本数
            - resources: 资源配置
            - auto_scale: 自动扩缩容配置
    
    Returns:
        部署后的模型信息
    """
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    deployment_config = data.get('config', {})
    
    deployed_model = service.deploy_model(
        model_id=model_id,
        deployment_config=deployment_config,
        user_id=user_id
    )
    
    result = deployed_model.to_dict() if hasattr(deployed_model, 'to_dict') else {
        'id': deployed_model.id,
        'status': getattr(deployed_model, 'status', None),
        'deployment_endpoint': deployed_model.config.get('deployment_endpoint') if deployed_model.config else None
    }
    
    return success_response(result, "模型部署成功")


@model_bp.route('/<model_id>/undeploy', methods=['POST'])
@jwt_required()
@handle_api_errors
def undeploy_model(model_id):
    """取消部署模型
    
    Path Parameters:
        model_id: 模型ID
    
    Returns:
        更新后的模型信息
    """
    user_id = get_jwt_identity()
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    updated_model = service.undeploy_model(model_id, user_id=user_id)
    
    result = updated_model.to_dict() if hasattr(updated_model, 'to_dict') else {
        'id': updated_model.id, 'status': getattr(updated_model, 'status', None)
    }
    
    return success_response(result, "模型已取消部署")


@model_bp.route('/<model_id>/archive', methods=['POST'])
@jwt_required()
@handle_api_errors
def archive_model(model_id):
    """归档模型
    
    Path Parameters:
        model_id: 模型ID
    
    Returns:
        更新后的模型信息
    """
    user_id = get_jwt_identity()
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    archived_model = service.archive_model(model_id, user_id=user_id)
    
    result = archived_model.to_dict() if hasattr(archived_model, 'to_dict') else {
        'id': archived_model.id, 'status': getattr(archived_model, 'status', None)
    }
    
    return success_response(result, "模型已归档")


@model_bp.route('/<model_id>/restore', methods=['POST'])
@jwt_required()
@handle_api_errors
def restore_model(model_id):
    """恢复归档的模型
    
    Path Parameters:
        model_id: 模型ID
    
    Returns:
        更新后的模型信息
    """
    user_id = get_jwt_identity()
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    restored_model = service.restore_model(model_id, user_id=user_id)
    
    result = restored_model.to_dict() if hasattr(restored_model, 'to_dict') else {
        'id': restored_model.id, 'status': getattr(restored_model, 'status', None)
    }
    
    return success_response(result, "模型已恢复")


# ==================== 版本管理接口 ====================

@model_bp.route('/<model_id>/versions', methods=['GET'])
@jwt_required()
@handle_api_errors
def list_versions(model_id):
    """获取模型版本列表
    
    Path Parameters:
        model_id: 模型ID
    
    Returns:
        版本列表
    """
    user_id = get_jwt_identity()
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    versions = service.list_versions(model_id)
    
    return success_response({
        'versions': versions,
        'count': len(versions)
    })


@model_bp.route('/<model_id>/versions', methods=['POST'])
@jwt_required()
@handle_api_errors
def create_version(model_id):
    """创建新版本
    
    Path Parameters:
        model_id: 模型ID
    
    Request Body:
        version: 版本号 (必需)
        description: 版本描述
        changelog: 变更日志
        storage_path: 存储路径
    
    Returns:
        创建的版本信息
    """
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    version = data.get('version')
    if not version:
        return error_response("版本号不能为空", 400)
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    version_data = service.create_version(
        model_id=model_id,
        version=version,
        description=data.get('description'),
        user_id=user_id,
        storage_path=data.get('storage_path'),
        changelog=data.get('changelog')
    )
    
    return success_response(version_data, "版本创建成功", 201)


@model_bp.route('/<model_id>/versions/<version>', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_version(model_id, version):
    """获取特定版本
    
    Path Parameters:
        model_id: 模型ID
        version: 版本号
    
    Returns:
        版本信息
    """
    user_id = get_jwt_identity()
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    version_data = service.get_version(model_id, version)
    
    if not version_data:
        return error_response(f"版本 {version} 不存在", 404)
    
    return success_response(version_data)


@model_bp.route('/<model_id>/versions/<version>/activate', methods=['POST'])
@jwt_required()
@handle_api_errors
def activate_version(model_id, version):
    """激活指定版本
    
    Path Parameters:
        model_id: 模型ID
        version: 版本号
    
    Returns:
        更新后的版本信息
    """
    user_id = get_jwt_identity()
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    version_data = service.activate_version(model_id, version, user_id=user_id)
    
    return success_response(version_data, f"版本 {version} 已激活")


# ==================== 指标管理接口 ====================

@model_bp.route('/<model_id>/metrics', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_metrics(model_id):
    """获取模型指标
    
    Path Parameters:
        model_id: 模型ID
    
    Returns:
        指标数据
    """
    user_id = get_jwt_identity()
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    metrics = service.get_metrics(model_id)
    
    return success_response({'metrics': metrics})


@model_bp.route('/<model_id>/metrics', methods=['PUT'])
@jwt_required()
@handle_api_errors
def update_metrics(model_id):
    """更新模型指标
    
    Path Parameters:
        model_id: 模型ID
    
    Request Body:
        metrics: 指标数据
    
    Returns:
        更新后的指标
    """
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    metrics = data.get('metrics', {})
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    result = service.update_metrics(model_id, metrics, user_id=user_id)
    
    return success_response(result, "指标更新成功")


# ==================== 导出接口 ====================

@model_bp.route('/<model_id>/export', methods=['POST'])
@jwt_required()
@handle_api_errors
def export_model(model_id):
    """导出模型
    
    Path Parameters:
        model_id: 模型ID
    
    Request Body:
        format: 导出格式 (onnx/torchscript/tensorflow/tensorrt/safetensors)
        config: 导出配置
    
    Returns:
        导出任务信息
    """
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    export_format = data.get('format', 'onnx')
    export_config = data.get('config', {})
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    export_task = service.export_model(
        model_id=model_id,
        export_format=export_format,
        export_config=export_config,
        user_id=user_id
    )
    
    return success_response(export_task, "导出任务已创建", 202)


@model_bp.route('/<model_id>/exports', methods=['GET'])
@jwt_required()
@handle_api_errors
def list_exports(model_id):
    """获取模型导出历史
    
    Path Parameters:
        model_id: 模型ID
    
    Returns:
        导出记录列表
    """
    user_id = get_jwt_identity()
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    exports = service.list_exports(model_id)
    
    return success_response({
        'exports': exports,
        'count': len(exports)
    })


@model_bp.route('/exports/<export_id>', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_export_status(export_id):
    """获取导出任务状态
    
    Path Parameters:
        export_id: 导出任务ID
    
    Returns:
        导出任务信息
    """
    service = get_model_service()
    export_task = service.get_export_status(export_id)
    
    if not export_task:
        return error_response("导出任务不存在", 404)
    
    return success_response(export_task)


# ==================== 事件接口 ====================

@model_bp.route('/<model_id>/events', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_events(model_id):
    """获取模型事件
    
    Path Parameters:
        model_id: 模型ID
    
    Query Parameters:
        event_type: 事件类型过滤
        limit: 返回数量 (默认100)
        offset: 偏移量 (默认0)
    
    Returns:
        事件列表
    """
    user_id = get_jwt_identity()
    
    event_type = request.args.get('event_type')
    limit = request.args.get('limit', 100, type=int)
    offset = request.args.get('offset', 0, type=int)
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    result = service.get_events(
        model_id=model_id,
        event_type=event_type,
        limit=limit,
        offset=offset
    )
    
    return success_response(result)


# ==================== 统计接口 ====================

@model_bp.route('/summary', methods=['GET'])
@jwt_required()
@handle_api_errors
def get_summary():
    """获取模型统计摘要
    
    Returns:
        模型统计信息
    """
    user_id = get_jwt_identity()
    
    service = get_model_service()
    summary = service.get_summary(user_id)
    
    return success_response(summary)


# ==================== 批量操作接口 ====================

@model_bp.route('/<model_id>/clone', methods=['POST'])
@jwt_required()
@handle_api_errors
def clone_model(model_id):
    """克隆模型
    
    Path Parameters:
        model_id: 源模型ID
    
    Request Body:
        name: 新模型名称 (必需)
        include_versions: 是否包含版本历史 (默认false)
    
    Returns:
        克隆的模型信息
    """
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    new_name = data.get('name')
    if not new_name:
        return error_response("新模型名称不能为空", 400)
    
    include_versions = data.get('include_versions', False)
    
    service = get_model_service()
    model = service.get_model(model_id)
    
    if not model:
        return error_response("模型不存在", 404)
    if model.user_id != user_id:
        return error_response("无权访问此模型", 403)
    
    cloned_model = service.clone_model(
        model_id=model_id,
        new_name=new_name,
        user_id=user_id,
        include_versions=include_versions
    )
    
    result = cloned_model.to_dict() if hasattr(cloned_model, 'to_dict') else {
        'id': cloned_model.id, 'name': cloned_model.name
    }
    
    return success_response(result, "模型克隆成功", 201)


@model_bp.route('/compare', methods=['POST'])
@jwt_required()
@handle_api_errors
def compare_models():
    """比较多个模型
    
    Request Body:
        model_ids: 模型ID列表 (2-10个)
    
    Returns:
        比较结果
    """
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    model_ids = data.get('model_ids', [])
    
    if not model_ids or len(model_ids) < 2:
        return error_response("至少需要2个模型进行比较", 400)
    
    service = get_model_service()
    
    # 验证权限
    for mid in model_ids:
        model = service.get_model(mid)
        if not model:
            return error_response(f"模型 {mid} 不存在", 404)
        if model.user_id != user_id:
            return error_response(f"无权访问模型 {mid}", 403)
    
    result = service.compare_models(model_ids)
    
    return success_response(result)


@model_bp.route('/batch/delete', methods=['POST'])
@jwt_required()
@handle_api_errors
def batch_delete():
    """批量删除模型
    
    Request Body:
        model_ids: 模型ID列表
    
    Returns:
        删除结果
    """
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    model_ids = data.get('model_ids', [])
    
    if not model_ids:
        return error_response("模型ID列表不能为空", 400)
    
    service = get_model_service()
    results = []
    
    for model_id in model_ids:
        try:
            model = service.get_model(model_id)
            if not model:
                results.append({'model_id': model_id, 'success': False, 'error': '模型不存在'})
                continue
            if model.user_id != user_id:
                results.append({'model_id': model_id, 'success': False, 'error': '无权限'})
                continue
            
            success = service.delete_model(model_id, user_id=user_id)
            results.append({'model_id': model_id, 'success': success})
        except Exception as e:
            results.append({'model_id': model_id, 'success': False, 'error': str(e)})
    
    success_count = sum(1 for r in results if r['success'])
    
    return success_response({
        'results': results,
        'total': len(model_ids),
        'success': success_count,
        'failed': len(model_ids) - success_count
    })


@model_bp.route('/batch/archive', methods=['POST'])
@jwt_required()
@handle_api_errors
def batch_archive():
    """批量归档模型
    
    Request Body:
        model_ids: 模型ID列表
    
    Returns:
        归档结果
    """
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    model_ids = data.get('model_ids', [])
    
    if not model_ids:
        return error_response("模型ID列表不能为空", 400)
    
    service = get_model_service()
    results = []
    
    for model_id in model_ids:
        try:
            model = service.get_model(model_id)
            if not model:
                results.append({'model_id': model_id, 'success': False, 'error': '模型不存在'})
                continue
            if model.user_id != user_id:
                results.append({'model_id': model_id, 'success': False, 'error': '无权限'})
                continue
            
            service.archive_model(model_id, user_id=user_id)
            results.append({'model_id': model_id, 'success': True})
        except Exception as e:
            results.append({'model_id': model_id, 'success': False, 'error': str(e)})
    
    success_count = sum(1 for r in results if r['success'])
    
    return success_response({
        'results': results,
        'total': len(model_ids),
        'success': success_count,
        'failed': len(model_ids) - success_count
    })


# ==================== 服务状态接口 ====================

@model_bp.route('/service/health', methods=['GET'])
@handle_api_errors
def service_health():
    """服务健康检查
    
    Returns:
        服务健康状态
    """
    try:
        service = get_model_service()
        
        return success_response({
            'status': 'healthy',
            'service': 'model_service',
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return error_response(f'服务不健康: {str(e)}', 503)


# ==================== 初始化和清理 ====================

def init_model_api(app=None):
    """初始化模型API
    
    Args:
        app: Flask应用实例
    """
    if app:
        app.register_blueprint(model_bp)
    
    logger.info("Model API initialized")
    return model_bp


def cleanup_model_api():
    """清理模型API资源"""
    global _model_service
    _model_service = None
    logger.info("Model API cleanup completed")

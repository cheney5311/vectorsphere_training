"""数据集管理API接口

提供数据集管理相关的高级API接口，包括数据集的上传、列表、搜索、批量操作、
导入导出、迁移、归档等管理功能。

所有接口都需要JWT认证。

API端点:
    基本管理:
        - GET /: 获取数据集列表（支持过滤、搜索、排序）
        - POST /: 上传/创建数据集
        - GET /<dataset_id>: 获取数据集详情
        - PUT /<dataset_id>: 更新数据集
        - DELETE /<dataset_id>: 删除数据集
        - GET /<dataset_id>/download: 下载数据集
    
    批量操作:
        - POST /batch/delete: 批量删除数据集
        - POST /batch/archive: 批量归档数据集
        - POST /batch/restore: 批量恢复数据集
        - POST /batch/export: 批量导出数据集
        - PUT /batch/tags: 批量更新标签
    
    高级管理:
        - POST /import: 从外部源导入数据集
        - POST /<dataset_id>/export: 导出数据集
        - POST /<dataset_id>/duplicate: 复制数据集
        - POST /<dataset_id>/archive: 归档数据集
        - POST /<dataset_id>/restore: 恢复数据集
        - POST /<dataset_id>/transfer: 转移数据集所有权
        - POST /<dataset_id>/merge: 合并数据集
    
    搜索和统计:
        - GET /search: 高级搜索
        - GET /statistics: 获取用户数据集统计
        - GET /recent: 获取最近访问的数据集
        - GET /popular: 获取热门数据集（公开）
    
    健康检查:
        - GET /health: 服务健康检查
"""

import sys
import os
from flask import Blueprint, request, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from backend.core.exceptions import ValidationError, BusinessLogicError
from backend.utils.response import success_response, error_response
from backend.services.dataset_service import DatasetService
from backend.repositories.dataset_repository import DatasetRepository
from backend.modules.dataset.dataset_exceptions import (
    DatasetNotFoundError,
    DatasetValidationError,
    DatasetBusinessLogicError
)

logger = logging.getLogger(__name__)

# 创建蓝图
dataset_management_bp = Blueprint('dataset_management', __name__, url_prefix='/api/v1/datasets')

# 初始化服务
dataset_repository = DatasetRepository()
dataset_service = DatasetService(dataset_repository)


# ============================================================================
# 基本管理API
# ============================================================================

@dataset_management_bp.route('', methods=['GET'])
@jwt_required()
def list_datasets():
    """获取数据集列表
    
    获取当前用户的数据集列表，支持过滤、搜索、排序和分页。
    需要JWT认证。
    
    查询参数:
        type (str): 数据集类型过滤 (可选)
            - 可选值: text, image, audio, video, tabular, mixed
            - 示例: type=text
        status (str): 状态过滤 (可选)
            - 可选值: pending, uploading, processing, ready, error, archived
            - 示例: status=ready
        search (str): 搜索关键词 (可选)
            - 在名称和描述中搜索
            - 示例: search=训练数据
        tags (str): 标签过滤，逗号分隔 (可选)
            - 示例: tags=NLP,v1.0
        limit (int): 返回数量限制 (可选)
            - 范围: 1-100
            - 默认: 50
        offset (int): 偏移量 (可选)
            - 默认: 0
        order_by (str): 排序字段 (可选)
            - 可选值: created_at, updated_at, name, size
            - 默认: created_at
        order_desc (bool): 是否降序 (可选)
            - 默认: true
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取数据集列表成功",
                "data": {
                    "datasets": [
                        {
                            "dataset_id": "string",
                            "name": "string",
                            "description": "string",
                            "dataset_type": "string",
                            "status": "string",
                            "size": "integer",
                            "record_count": "integer",
                            "created_at": "string",
                            "updated_at": "string"
                        }
                    ],
                    "total_count": "integer",
                    "filtered_count": "integer",
                    "limit": "integer",
                    "offset": "integer",
                    "has_more": "boolean"
                }
            }
        失败:
            - 401: 未授权
            - 500: 服务器内部错误
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/datasets?type=text&status=ready&limit=20" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        current_user_id = get_jwt_identity()
        
        # 获取查询参数
        dataset_type = request.args.get('type')
        status = request.args.get('status')
        limit = min(int(request.args.get('limit', 50)), 100)
        offset = max(int(request.args.get('offset', 0)), 0)
        search = request.args.get('search')
        tags_param = request.args.get('tags')
        order_by = request.args.get('order_by', 'created_at')
        order_desc = request.args.get('order_desc', 'true').lower() == 'true'
        
        # 解析标签
        tag_filter = tags_param.split(',') if tags_param else None
        
        # 获取数据集列表（从服务层获取已过滤的数据）
        result = dataset_service.search_datasets(
            user_id=current_user_id,
            dataset_type=dataset_type,
            status=status,
            search=search,
            tag_filter=tag_filter,
            limit=limit,
            offset=offset,
            order_by=order_by,
            order_desc=order_desc
        )
        
        return success_response(
            data=result,
            message="获取数据集列表成功"
        )
        
    except ValueError as e:
        return error_response(
            message=f"查询参数无效: {str(e)}",
            code=400,
            error_type="VALIDATION_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        return error_response(
            message=f"获取数据集列表失败: {str(e)}",
            code=500,
            error_type="LIST_DATASETS_ERROR"
        ), 500


@dataset_management_bp.route('', methods=['POST'])
@jwt_required()
def upload_dataset():
    """上传/创建数据集
    
    上传新的数据集文件或创建数据集元数据。
    支持文件上传和JSON数据直接创建两种方式。
    需要JWT认证。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): multipart/form-data 或 application/json
            - 必填
    
    请求体 - 文件上传方式 (multipart/form-data):
        file (File): 数据集文件 (必填)
            - 支持格式: .json, .jsonl, .csv, .txt, .parquet
            - 最大大小: 100MB
        name (str): 数据集名称 (可选，默认为文件名)
            - 长度: 1-200字符
        description (str): 数据集描述 (可选)
        dataset_type (str): 数据集类型 (可选，默认'text')
            - 可选值: text, image, audio, video, tabular, mixed
        tags (str): 标签JSON数组字符串 (可选)
            - 示例: '["NLP", "训练数据"]'
        config (str): 配置JSON字符串 (可选)
            - 示例: '{"max_samples": 10000}'
    
    请求体 - JSON方式 (application/json):
        {
            "name": "string",            // 数据集名称 (必填)
            "description": "string",     // 描述 (可选)
            "dataset_type": "string",    // 类型 (可选，默认'text')
            "format": "string",          // 格式 (可选，默认'json')
            "storage_path": "string",    // 存储路径 (可选)
            "config": {},                // 配置 (可选)
            "tags": ["string"]           // 标签列表 (可选)
        }
    
    响应体 (JSON):
        成功 (201):
            {
                "code": 201,
                "message": "数据集创建成功",
                "data": {
                    "dataset_id": "string",
                    "name": "string",
                    "status": "pending",
                    "created_at": "string"
                    // ...其他字段
                }
            }
        失败:
            - 400: 请求数据无效
            - 401: 未授权
            - 413: 文件过大
            - 415: 不支持的文件类型
            - 500: 服务器内部错误
    
    示例 - 文件上传:
        curl -X POST "http://localhost:5000/api/v1/datasets" \\
             -H "Authorization: Bearer <token>" \\
             -F "file=@data.json" \\
             -F "name=训练数据集" \\
             -F "dataset_type=text"
    
    示例 - JSON创建:
        curl -X POST "http://localhost:5000/api/v1/datasets" \\
             -H "Authorization: Bearer <token>" \\
             -H "Content-Type: application/json" \\
             -d '{"name": "训练数据集", "dataset_type": "text"}'
    """
    try:
        current_user_id = get_jwt_identity()
        
        # 判断请求类型
        content_type = request.content_type or ''
        
        if 'multipart/form-data' in content_type:
            # 文件上传方式
            return _handle_file_upload(current_user_id)
        else:
            # JSON方式
            return _handle_json_create(current_user_id)
            
    except DatasetValidationError as e:
        return error_response(
            message=str(e),
            code=400,
            error_type="VALIDATION_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        return error_response(
            message=f"上传数据集失败: {str(e)}",
            code=500,
            error_type="UPLOAD_ERROR"
        ), 500


def _handle_file_upload(user_id: str):
    """处理文件上传"""
    if 'file' not in request.files:
        return error_response(
            message="缺少文件",
            code=400,
            error_type="MISSING_FILE"
        ), 400
        
    file = request.files['file']
    if file.filename == '':
        return error_response(
            message="文件名不能为空",
            code=400,
            error_type="EMPTY_FILENAME"
        ), 400
    
    # 检查文件类型
    allowed_extensions = {'.json', '.jsonl', '.csv', '.txt', '.parquet'}
    file_ext = os.path.splitext(file.filename.lower())[1] if file.filename else ''
    if file_ext not in allowed_extensions:
        return error_response(
            message=f"不支持的文件类型: {file_ext}，支持: {', '.join(allowed_extensions)}",
            code=415,
            error_type="UNSUPPORTED_FILE_TYPE"
        ), 415
        
        # 获取表单数据
        name = request.form.get('name') or file.filename or 'unnamed_dataset'
        description = request.form.get('description', '')
        dataset_type = request.form.get('dataset_type', 'text')
        tags = request.form.get('tags', '[]')
        config = request.form.get('config', '{}')
        
    # 确定格式
    format_map = {
        '.json': 'json',
        '.jsonl': 'jsonl',
        '.csv': 'csv',
        '.txt': 'text',
        '.parquet': 'parquet'
    }
    file_format = format_map.get(file_ext, 'json')
    
    # 保存文件（实际应用中应该保存到存储服务）
    # storage_path = save_uploaded_file(file, user_id)
    storage_path = f"./data/datasets/{user_id}/{name}"
    
    # 创建数据集
    dataset = dataset_service.create_dataset(
        user_id=user_id,
        name=name,
        description=description,
        dataset_type=dataset_type,
        format=file_format,
        storage_path=storage_path,
        config={**config, 'tags': tags, 'original_filename': file.filename}
    )
    
    # 添加标签
    for tag in tags:
        try:
            dataset_service.add_dataset_tag(dataset.dataset_id, tag, user_id)
        except Exception as e:
            logger.warning(f"Failed to add tag '{tag}': {e}")
    
    logger.info(f"User {user_id} uploaded dataset {dataset.dataset_id}: {name}")
    
    return success_response(
        data=dataset.to_dict(),
        message="数据集上传成功"
    ), 201


def _handle_json_create(user_id: str):
    """处理JSON创建"""
    data = request.get_json()
    
    if not data:
        return error_response(
            message="请求数据不能为空",
            code=400,
            error_type="EMPTY_REQUEST"
        ), 400
    
    name = data.get('name')
    if not name:
        return error_response(
            message="数据集名称不能为空",
            code=400,
            error_type="MISSING_NAME"
        ), 400
    
    # 创建数据集
    dataset = dataset_service.create_dataset(
        user_id=user_id,
        name=name,
        description=data.get('description'),
        dataset_type=data.get('dataset_type', 'text'),
        format=data.get('format', 'json'),
        storage_path=data.get('storage_path', ''),
        config=data.get('config', {})
    )
    
    # 添加标签
    tags = data.get('tags', [])
    for tag in tags:
        try:
            dataset_service.add_dataset_tag(dataset.dataset_id, tag, user_id)
        except Exception as e:
            logger.warning(f"Failed to add tag '{tag}': {e}")
    
    logger.info(f"User {user_id} created dataset {dataset.dataset_id}: {name}")
    
    return success_response(
        data=dataset.to_dict(),
        message="数据集创建成功"
    ), 201


@dataset_management_bp.route('/<dataset_id>', methods=['GET'])
@jwt_required()
def get_dataset(dataset_id: str):
    """获取数据集详情
    
    获取指定数据集的详细信息，包括基本信息、统计信息和标签。
    需要JWT认证，且只能获取当前用户拥有的数据集。
    
    路径参数:
        dataset_id (str): 数据集唯一标识符 (UUID格式)
            - 必填
            - 格式: "550e8400-e29b-41d4-a716-446655440000"
    
    查询参数:
        include_stats (bool): 是否包含统计信息 (可选)
            - 默认: false
        include_tags (bool): 是否包含标签 (可选)
            - 默认: true
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取数据集详情成功",
                "data": {
                    "dataset": {
                        "dataset_id": "string",
                        "user_id": "string",
                        "name": "string",
                        "description": "string",
                        "dataset_type": "string",
                        "format": "string",
                        "status": "string",
                        "size": "integer",
                        "record_count": "integer",
                        "storage_path": "string",
                        "version": "string",
                        "ready": "boolean",
                        "validated": "boolean",
                        "created_at": "string",
                        "updated_at": "string",
                        "config": {}
                    },
                    "tags": ["string"],          // 如果include_tags=true
                    "statistics": {}             // 如果include_stats=true
                }
            }
        失败:
            - 400: 参数无效
            - 401: 未授权
            - 403: 无权访问
            - 404: 数据集不存在
            - 500: 服务器内部错误
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/datasets/xxx?include_stats=true" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        current_user_id = get_jwt_identity()
        include_stats = request.args.get('include_stats', 'false').lower() == 'true'
        include_tags = request.args.get('include_tags', 'true').lower() == 'true'
        
        # 获取数据集
        dataset = dataset_service.get_dataset(dataset_id)
        
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
            
        # 检查权限
        if dataset.user_id != current_user_id:
            return error_response(
                message="无权限访问该数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        response_data = {
            "dataset": dataset.to_dict()
        }
        
        # 获取标签
        if include_tags:
            response_data["tags"] = dataset_service.get_dataset_tags(dataset_id)
        
        # 获取统计信息
        if include_stats:
            stats = dataset_service.get_detailed_statistics(dataset_id)
            response_data["statistics"] = stats
        
        # 记录访问
        dataset_service.log_access(dataset_id, current_user_id, "view")
        
        return success_response(
            data=response_data,
            message="获取数据集详情成功"
        )
        
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error getting dataset {dataset_id}: {e}")
        return error_response(
            message=f"获取数据集详情失败: {str(e)}",
            code=500,
            error_type="GET_DATASET_ERROR"
        ), 500


@dataset_management_bp.route('/<dataset_id>', methods=['PUT'])
@jwt_required()
def update_dataset(dataset_id: str):
    """更新数据集
    
    更新指定数据集的信息。
    需要JWT认证，且只能更新当前用户拥有的数据集。
    
    路径参数:
        dataset_id (str): 数据集唯一标识符 (UUID格式)
            - 必填
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 必填
    
    请求体 (JSON):
        {
            "name": "string",           // 数据集名称 (可选)
            "description": "string",    // 描述 (可选)
            "config": {},               // 配置信息 (可选)
            "tags": ["string"]          // 标签列表 (可选，将覆盖现有标签)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "数据集更新成功",
                "data": {
                    "dataset": {
                        "dataset_id": "string",
                        "name": "string",
                        "updated_at": "string"
                        // ...其他字段
                    }
                }
            }
        失败:
            - 400: 请求数据无效
            - 401: 未授权
            - 403: 无权访问
            - 404: 数据集不存在
            - 500: 服务器内部错误
    
    示例:
        curl -X PUT "http://localhost:5000/api/v1/datasets/xxx" \\
             -H "Authorization: Bearer <token>" \\
             -H "Content-Type: application/json" \\
             -d '{"name": "新名称", "description": "新描述"}'
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data:
            return error_response(
                message="请求数据不能为空",
                code=400,
                error_type="EMPTY_REQUEST"
            ), 400
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
            
        if dataset.user_id != current_user_id:
            return error_response(
                message="无权限修改该数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 更新数据集
        updated_dataset = dataset_service.update_dataset(
            dataset_id=dataset_id,
            name=data.get('name'),
            description=data.get('description'),
            config=data.get('config')
        )
        
        # 更新标签
        if 'tags' in data:
            tags = data['tags']
            if isinstance(tags, list):
                dataset_service.clear_dataset_tags(dataset_id)
                for tag in tags:
                    dataset_service.add_dataset_tag(dataset_id, tag, current_user_id)
        
        # 记录访问
        dataset_service.log_access(dataset_id, current_user_id, "update")
        
        return success_response(
            data={"dataset": updated_dataset.to_dict()},
            message="数据集更新成功"
        )
        
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except DatasetValidationError as e:
        return error_response(
            message=str(e),
            code=400,
            error_type="VALIDATION_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error updating dataset {dataset_id}: {e}")
        return error_response(
            message=f"更新数据集失败: {str(e)}",
            code=500,
            error_type="UPDATE_ERROR"
        ), 500


@dataset_management_bp.route('/<dataset_id>', methods=['DELETE'])
@jwt_required()
def delete_dataset(dataset_id: str):
    """删除数据集
    
    永久删除指定的数据集。此操作不可逆。
    需要JWT认证，且只能删除当前用户拥有的数据集。
    
    路径参数:
        dataset_id (str): 数据集唯一标识符 (UUID格式)
            - 必填
    
    查询参数:
        force (bool): 是否强制删除 (可选)
            - true: 强制删除，包括正在处理的数据集
            - false: 只删除非活动状态的数据集
            - 默认: false
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "数据集删除成功",
                "data": {
                    "deleted": true,
                    "dataset_id": "string"
                }
            }
        失败:
            - 400: 数据集正在使用中
            - 401: 未授权
            - 403: 无权删除
            - 404: 数据集不存在
            - 500: 服务器内部错误
    
    示例:
        curl -X DELETE "http://localhost:5000/api/v1/datasets/xxx?force=true" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        current_user_id = get_jwt_identity()
        force = request.args.get('force', 'false').lower() == 'true'
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
            
        if dataset.user_id != current_user_id:
            return error_response(
                message="无权限删除该数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 检查是否正在使用
        if not force and dataset.status == 'processing':
            return error_response(
                message="数据集正在处理中，无法删除。使用force=true强制删除",
                code=400,
                error_type="DATASET_IN_USE"
            ), 400
            
        # 删除数据集
        success = dataset_service.delete_dataset(dataset_id)
        
        if success:
            logger.info(f"User {current_user_id} deleted dataset {dataset_id}")
            return success_response(
                data={"deleted": True, "dataset_id": dataset_id},
                message="数据集删除成功"
            )
        else:
            return error_response(
                message="删除数据集失败",
                code=500,
                error_type="DELETE_ERROR"
            ), 500
            
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_id}: {e}")
        return error_response(
            message=f"删除数据集失败: {str(e)}",
            code=500,
            error_type="DELETE_ERROR"
        ), 500


@dataset_management_bp.route('/<dataset_id>/download', methods=['GET'])
@jwt_required()
def download_dataset(dataset_id: str):
    """下载数据集
    
    获取数据集文件的下载链接。
    需要JWT认证，且只能下载当前用户拥有的数据集。
    
    路径参数:
        dataset_id (str): 数据集唯一标识符 (UUID格式)
            - 必填
    
    查询参数:
        format (str): 下载格式 (可选)
            - 可选值: original, json, csv, parquet
            - 默认: original
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取下载链接成功",
                "data": {
                    "download_url": "string",
                    "file_name": "string",
                    "file_size": "integer",
                    "format": "string",
                    "expires_at": "string"
                }
            }
        失败:
            - 401: 未授权
            - 403: 无权下载
            - 404: 数据集不存在
            - 500: 服务器内部错误
    
    示例:
        curl -X GET "http://localhost:5000/api/v1/datasets/xxx/download?format=csv" \\
             -H "Authorization: Bearer <token>"
    """
    try:
        current_user_id = get_jwt_identity()
        download_format = request.args.get('format', 'original')
        
        # 获取数据集验证权限
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
            
        if dataset.user_id != current_user_id:
            return error_response(
                message="无权限下载该数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 生成下载URL
        download_info = dataset_service.generate_download_url(
            dataset_id=dataset_id,
            user_id=current_user_id,
            format=download_format
        )
        
        # 记录访问
        dataset_service.log_access(dataset_id, current_user_id, "download")
        
        return success_response(
            data=download_info,
            message="获取下载链接成功"
        )
        
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error downloading dataset {dataset_id}: {e}")
        return error_response(
            message=f"下载数据集失败: {str(e)}",
            code=500,
            error_type="DOWNLOAD_ERROR"
        ), 500


# ============================================================================
# 批量操作API
# ============================================================================

@dataset_management_bp.route('/batch/delete', methods=['POST'])
@jwt_required()
def batch_delete_datasets():
    """批量删除数据集
    
    批量删除多个数据集。只会删除当前用户拥有的数据集。
    需要JWT认证。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 必填
    
    请求体 (JSON):
        {
            "dataset_ids": ["string"],   // 要删除的数据集ID列表 (必填)
            "force": "boolean"           // 是否强制删除 (可选，默认false)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "批量删除完成",
                "data": {
                    "total": "integer",
                    "success_count": "integer",
                    "failed_count": "integer",
                    "results": [
                        {
                            "dataset_id": "string",
                            "success": "boolean",
                            "error": "string"      // 仅失败时有
                        }
                    ]
                }
            }
        失败:
            - 400: 请求数据无效
            - 401: 未授权
            - 500: 服务器内部错误
    
    示例:
        curl -X POST "http://localhost:5000/api/v1/datasets/batch/delete" \\
             -H "Authorization: Bearer <token>" \\
             -H "Content-Type: application/json" \\
             -d '{"dataset_ids": ["id1", "id2", "id3"]}'
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or 'dataset_ids' not in data:
            return error_response(
                message="缺少dataset_ids参数",
                code=400,
                error_type="MISSING_PARAMETER"
            ), 400
        
        dataset_ids = data['dataset_ids']
        force = data.get('force', False)
        
        if not isinstance(dataset_ids, list) or len(dataset_ids) == 0:
            return error_response(
                message="dataset_ids必须是非空数组",
                code=400,
                error_type="INVALID_PARAMETER"
            ), 400
        
        results = []
        success_count = 0
        
        for dataset_id in dataset_ids:
            try:
                dataset = dataset_service.get_dataset(dataset_id)
                
                if not dataset:
                    results.append({
                        "dataset_id": dataset_id,
                        "success": False,
                        "error": "数据集不存在"
                    })
                    continue
                
                if dataset.user_id != current_user_id:
                    results.append({
                        "dataset_id": dataset_id,
                        "success": False,
                        "error": "无权限删除"
                    })
                    continue
                
                if not force and dataset.status == 'processing':
                    results.append({
                        "dataset_id": dataset_id,
                        "success": False,
                        "error": "数据集正在处理中"
                    })
                    continue
                
                success = dataset_service.delete_dataset(dataset_id)
                results.append({
                    "dataset_id": dataset_id,
                    "success": success,
                    "error": None if success else "删除失败"
                })
                if success:
                    success_count += 1
                    
            except Exception as e:
                results.append({
                    "dataset_id": dataset_id,
                    "success": False,
                    "error": str(e)
                })
        
        return success_response(
            data={
                "total": len(dataset_ids),
                "success_count": success_count,
                "failed_count": len(dataset_ids) - success_count,
                "results": results
            },
            message=f"批量删除完成: {success_count} 成功, {len(dataset_ids) - success_count} 失败"
        )
        
    except Exception as e:
        logger.error(f"Error batch deleting datasets: {e}")
        return error_response(
            message=f"批量删除失败: {str(e)}",
            code=500,
            error_type="BATCH_DELETE_ERROR"
        ), 500


@dataset_management_bp.route('/batch/archive', methods=['POST'])
@jwt_required()
def batch_archive_datasets():
    """批量归档数据集
    
    批量归档多个数据集。归档后的数据集不会被删除，但会被标记为archived状态。
    需要JWT认证。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 必填
    
    请求体 (JSON):
        {
            "dataset_ids": ["string"]    // 要归档的数据集ID列表 (必填)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "批量归档完成",
                "data": {
                    "total": "integer",
                    "success_count": "integer",
                    "failed_count": "integer",
                    "results": [
                        {
                            "dataset_id": "string",
                            "success": "boolean",
                            "error": "string"
                        }
                    ]
                }
            }
        失败:
            - 400: 请求数据无效
            - 401: 未授权
            - 500: 服务器内部错误
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or 'dataset_ids' not in data:
            return error_response(
                message="缺少dataset_ids参数",
                code=400,
                error_type="MISSING_PARAMETER"
            ), 400
        
        dataset_ids = data['dataset_ids']
        results = []
        success_count = 0
        
        for dataset_id in dataset_ids:
            try:
                dataset = dataset_service.get_dataset(dataset_id)
                
                if not dataset:
                    results.append({
                        "dataset_id": dataset_id,
                        "success": False,
                        "error": "数据集不存在"
                    })
                    continue
                
                if dataset.user_id != current_user_id:
                    results.append({
                        "dataset_id": dataset_id,
                        "success": False,
                        "error": "无权限操作"
                    })
                    continue
                
                # 归档数据集
                dataset_service.archive_dataset(dataset_id, current_user_id)
                results.append({
                    "dataset_id": dataset_id,
                    "success": True
                })
                success_count += 1
                
            except Exception as e:
                results.append({
                    "dataset_id": dataset_id,
                    "success": False,
                    "error": str(e)
                })
        
        return success_response(
            data={
                "total": len(dataset_ids),
                "success_count": success_count,
                "failed_count": len(dataset_ids) - success_count,
                "results": results
            },
            message=f"批量归档完成: {success_count} 成功"
        )
        
    except Exception as e:
        logger.error(f"Error batch archiving datasets: {e}")
        return error_response(
            message=f"批量归档失败: {str(e)}",
            code=500,
            error_type="BATCH_ARCHIVE_ERROR"
        ), 500


@dataset_management_bp.route('/batch/restore', methods=['POST'])
@jwt_required()
def batch_restore_datasets():
    """批量恢复数据集
    
    批量恢复已归档的数据集。
    需要JWT认证。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 必填
    
    请求体 (JSON):
        {
            "dataset_ids": ["string"]    // 要恢复的数据集ID列表 (必填)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "批量恢复完成",
                "data": {
                    "total": "integer",
                    "success_count": "integer",
                    "failed_count": "integer",
                    "results": [...]
                }
            }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or 'dataset_ids' not in data:
            return error_response(
                message="缺少dataset_ids参数",
                code=400,
                error_type="MISSING_PARAMETER"
            ), 400
        
        dataset_ids = data['dataset_ids']
        results = []
        success_count = 0
        
        for dataset_id in dataset_ids:
            try:
                dataset = dataset_service.get_dataset(dataset_id)
                
                if not dataset:
                    results.append({
                        "dataset_id": dataset_id,
                        "success": False,
                        "error": "数据集不存在"
                    })
                    continue
                
                if dataset.user_id != current_user_id:
                    results.append({
                        "dataset_id": dataset_id,
                        "success": False,
                        "error": "无权限操作"
                    })
                    continue
                
                if dataset.status != 'archived':
                    results.append({
                        "dataset_id": dataset_id,
                        "success": False,
                        "error": "数据集未归档"
                    })
                    continue
                
                # 恢复数据集
                dataset_service.restore_dataset(dataset_id, current_user_id)
                results.append({
                    "dataset_id": dataset_id,
                    "success": True
                })
                success_count += 1
                
            except Exception as e:
                results.append({
                    "dataset_id": dataset_id,
                    "success": False,
                    "error": str(e)
                })
        
        return success_response(
            data={
                "total": len(dataset_ids),
                "success_count": success_count,
                "failed_count": len(dataset_ids) - success_count,
                "results": results
            },
            message=f"批量恢复完成: {success_count} 成功"
        )
        
    except Exception as e:
        logger.error(f"Error batch restoring datasets: {e}")
        return error_response(
            message=f"批量恢复失败: {str(e)}",
            code=500,
            error_type="BATCH_RESTORE_ERROR"
        ), 500


@dataset_management_bp.route('/batch/tags', methods=['PUT'])
@jwt_required()
def batch_update_tags():
    """批量更新标签
    
    为多个数据集批量添加或删除标签。
    需要JWT认证。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 必填
    
    请求体 (JSON):
        {
            "dataset_ids": ["string"],   // 数据集ID列表 (必填)
            "add_tags": ["string"],       // 要添加的标签 (可选)
            "remove_tags": ["string"]     // 要删除的标签 (可选)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "批量更新标签完成",
                "data": {
                    "total": "integer",
                    "success_count": "integer",
                    "results": [...]
                }
            }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or 'dataset_ids' not in data:
            return error_response(
                message="缺少dataset_ids参数",
                code=400,
                error_type="MISSING_PARAMETER"
            ), 400
        
        dataset_ids = data['dataset_ids']
        add_tags = data.get('add_tags', [])
        remove_tags = data.get('remove_tags', [])
        
        if not add_tags and not remove_tags:
            return error_response(
                message="必须指定add_tags或remove_tags",
                code=400,
                error_type="MISSING_PARAMETER"
            ), 400
        
        results = []
        success_count = 0
        
        for dataset_id in dataset_ids:
            try:
                dataset = dataset_service.get_dataset(dataset_id)
                
                if not dataset or dataset.user_id != current_user_id:
                    results.append({
                        "dataset_id": dataset_id,
                        "success": False,
                        "error": "数据集不存在或无权限"
                    })
                    continue
                
                # 添加标签
                for tag in add_tags:
                    dataset_service.add_dataset_tag(dataset_id, tag, current_user_id)
                
                # 删除标签
                for tag in remove_tags:
                    dataset_service.remove_dataset_tag(dataset_id, tag)
                
                results.append({
                    "dataset_id": dataset_id,
                    "success": True
                })
                success_count += 1
                
            except Exception as e:
                results.append({
                    "dataset_id": dataset_id,
                    "success": False,
                    "error": str(e)
                })
        
        return success_response(
            data={
                "total": len(dataset_ids),
                "success_count": success_count,
                "failed_count": len(dataset_ids) - success_count,
                "results": results
            },
            message=f"批量更新标签完成: {success_count} 成功"
        )
        
    except Exception as e:
        logger.error(f"Error batch updating tags: {e}")
        return error_response(
            message=f"批量更新标签失败: {str(e)}",
            code=500,
            error_type="BATCH_TAGS_ERROR"
        ), 500


# ============================================================================
# 高级管理API
# ============================================================================

@dataset_management_bp.route('/<dataset_id>/duplicate', methods=['POST'])
@jwt_required()
def duplicate_dataset(dataset_id: str):
    """复制数据集
    
    创建指定数据集的副本。
    需要JWT认证，且只能复制当前用户拥有的数据集。
    
    路径参数:
        dataset_id (str): 源数据集ID
            - 必填
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 可选
    
    请求体 (JSON):
        {
            "new_name": "string",        // 新数据集名称 (可选，默认加_copy后缀)
            "include_tags": "boolean"    // 是否复制标签 (可选，默认true)
        }
    
    响应体 (JSON):
        成功 (201):
            {
                "code": 201,
                "message": "数据集复制成功",
                "data": {
                    "source_dataset_id": "string",
                    "new_dataset": {
                        "dataset_id": "string",
                        "name": "string"
                        // ...其他字段
                    }
                }
            }
        失败:
            - 401: 未授权
            - 403: 无权访问
            - 404: 源数据集不存在
            - 500: 服务器内部错误
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json() or {}
        
        new_name = data.get('new_name')
        include_tags = data.get('include_tags', True)
        
        # 验证源数据集
        source_dataset = dataset_service.get_dataset(dataset_id)
        if not source_dataset:
            return error_response(
                message=f"源数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if source_dataset.user_id != current_user_id:
            return error_response(
                message="无权限复制该数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 复制数据集
        new_dataset = dataset_service.clone_dataset(
            source_dataset_id=dataset_id,
            user_id=current_user_id,
            new_name=new_name
        )
        
        # 复制标签
        if include_tags:
            source_tags = dataset_service.get_dataset_tags(dataset_id)
            for tag in source_tags:
                dataset_service.add_dataset_tag(new_dataset.dataset_id, tag, current_user_id)
        
        logger.info(f"User {current_user_id} duplicated dataset {dataset_id} to {new_dataset.dataset_id}")
        
        return success_response(
            data={
                "source_dataset_id": dataset_id,
                "new_dataset": new_dataset.to_dict()
            },
            message="数据集复制成功"
        ), 201
        
    except DatasetNotFoundError as e:
        return error_response(
            message=str(e),
            code=404,
            error_type="DATASET_NOT_FOUND"
        ), 404
    except Exception as e:
        logger.error(f"Error duplicating dataset {dataset_id}: {e}")
        return error_response(
            message=f"复制数据集失败: {str(e)}",
            code=500,
            error_type="DUPLICATE_ERROR"
        ), 500


@dataset_management_bp.route('/<dataset_id>/archive', methods=['POST'])
@jwt_required()
def archive_dataset(dataset_id: str):
    """归档数据集
    
    将数据集标记为归档状态。归档后的数据集不会被删除，但会从默认列表中隐藏。
    需要JWT认证，且只能归档当前用户拥有的数据集。
    
    路径参数:
        dataset_id (str): 数据集ID
            - 必填
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "数据集归档成功",
                "data": {
                    "dataset_id": "string",
                    "status": "archived",
                    "archived_at": "string"
                }
            }
    """
    try:
        current_user_id = get_jwt_identity()
        
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != current_user_id:
            return error_response(
                message="无权限归档该数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        archived_dataset = dataset_service.archive_dataset(dataset_id, current_user_id)
        
        return success_response(
            data={
                "dataset_id": dataset_id,
                "status": "archived",
                "archived_at": datetime.utcnow().isoformat()
            },
            message="数据集归档成功"
        )
        
    except Exception as e:
        logger.error(f"Error archiving dataset {dataset_id}: {e}")
        return error_response(
            message=f"归档数据集失败: {str(e)}",
            code=500,
            error_type="ARCHIVE_ERROR"
        ), 500


@dataset_management_bp.route('/<dataset_id>/restore', methods=['POST'])
@jwt_required()
def restore_dataset(dataset_id: str):
    """恢复归档的数据集
    
    将已归档的数据集恢复为正常状态。
    需要JWT认证，且只能恢复当前用户拥有的数据集。
    
    路径参数:
        dataset_id (str): 数据集ID
            - 必填
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "数据集恢复成功",
                "data": {
                    "dataset_id": "string",
                    "status": "ready",
                    "restored_at": "string"
                }
            }
    """
    try:
        current_user_id = get_jwt_identity()
        
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != current_user_id:
            return error_response(
                message="无权限恢复该数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        if dataset.status != 'archived':
            return error_response(
                message="数据集未归档，无需恢复",
                code=400,
                error_type="NOT_ARCHIVED"
            ), 400
        
        restored_dataset = dataset_service.restore_dataset(dataset_id, current_user_id)
        
        return success_response(
            data={
                "dataset_id": dataset_id,
                "status": "ready",
                "restored_at": datetime.utcnow().isoformat()
            },
            message="数据集恢复成功"
        )
        
    except Exception as e:
        logger.error(f"Error restoring dataset {dataset_id}: {e}")
        return error_response(
            message=f"恢复数据集失败: {str(e)}",
            code=500,
            error_type="RESTORE_ERROR"
        ), 500


@dataset_management_bp.route('/<dataset_id>/transfer', methods=['POST'])
@jwt_required()
def transfer_dataset(dataset_id: str):
    """转移数据集所有权
    
    将数据集的所有权转移给另一个用户。
    转移后，当前用户将无法访问该数据集。
    需要JWT认证，且只能转移当前用户拥有的数据集。
    
    路径参数:
        dataset_id (str): 数据集ID
            - 必填
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 必填
    
    请求体 (JSON):
        {
            "target_user_id": "string"   // 目标用户ID (必填)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "数据集转移成功",
                "data": {
                    "dataset_id": "string",
                    "from_user_id": "string",
                    "to_user_id": "string",
                    "transferred_at": "string"
                }
            }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or 'target_user_id' not in data:
            return error_response(
                message="缺少target_user_id参数",
                code=400,
                error_type="MISSING_PARAMETER"
            ), 400
        
        target_user_id = data['target_user_id']
        
        if target_user_id == current_user_id:
            return error_response(
                message="不能转移给自己",
                code=400,
                error_type="INVALID_TRANSFER"
            ), 400
        
        dataset = dataset_service.get_dataset(dataset_id)
        if not dataset:
            return error_response(
                message=f"数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if dataset.user_id != current_user_id:
            return error_response(
                message="无权限转移该数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 执行转移
        transferred = dataset_service.transfer_dataset(
            dataset_id=dataset_id,
            from_user_id=current_user_id,
            to_user_id=target_user_id
        )
        
        logger.info(f"Dataset {dataset_id} transferred from {current_user_id} to {target_user_id}")
        
        return success_response(
            data={
                "dataset_id": dataset_id,
                "from_user_id": current_user_id,
                "to_user_id": target_user_id,
                "transferred_at": datetime.utcnow().isoformat()
            },
            message="数据集转移成功"
        )
        
    except Exception as e:
        logger.error(f"Error transferring dataset {dataset_id}: {e}")
        return error_response(
            message=f"转移数据集失败: {str(e)}",
            code=500,
            error_type="TRANSFER_ERROR"
        ), 500


@dataset_management_bp.route('/<dataset_id>/merge', methods=['POST'])
@jwt_required()
def merge_datasets(dataset_id: str):
    """合并数据集
    
    将多个数据集合并到指定的目标数据集中。
    只能合并相同类型和格式的数据集。
    需要JWT认证，且只能合并当前用户拥有的数据集。
    
    路径参数:
        dataset_id (str): 目标数据集ID
            - 必填
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
        Content-Type (str): application/json
            - 必填
    
    请求体 (JSON):
        {
            "source_dataset_ids": ["string"],  // 源数据集ID列表 (必填)
            "delete_sources": "boolean"         // 合并后是否删除源数据集 (可选，默认false)
        }
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "数据集合并成功",
                "data": {
                    "target_dataset_id": "string",
                    "merged_count": "integer",
                    "total_records": "integer",
                    "merged_at": "string"
                }
            }
    """
    try:
        current_user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or 'source_dataset_ids' not in data:
            return error_response(
                message="缺少source_dataset_ids参数",
                code=400,
                error_type="MISSING_PARAMETER"
            ), 400
        
        source_dataset_ids = data['source_dataset_ids']
        delete_sources = data.get('delete_sources', False)
        
        if not isinstance(source_dataset_ids, list) or len(source_dataset_ids) == 0:
            return error_response(
                message="source_dataset_ids必须是非空数组",
                code=400,
                error_type="INVALID_PARAMETER"
            ), 400
        
        # 验证目标数据集
        target_dataset = dataset_service.get_dataset(dataset_id)
        if not target_dataset:
            return error_response(
                message=f"目标数据集 {dataset_id} 不存在",
                code=404,
                error_type="DATASET_NOT_FOUND"
            ), 404
        
        if target_dataset.user_id != current_user_id:
            return error_response(
                message="无权限操作目标数据集",
                code=403,
                error_type="FORBIDDEN"
            ), 403
        
        # 执行合并
        merge_result = dataset_service.merge_datasets(
            target_dataset_id=dataset_id,
            source_dataset_ids=source_dataset_ids,
            user_id=current_user_id,
            delete_sources=delete_sources
        )
        
        return success_response(
            data=merge_result,
            message="数据集合并成功"
        )
        
    except DatasetValidationError as e:
        return error_response(
            message=str(e),
            code=400,
            error_type="VALIDATION_ERROR"
        ), 400
    except Exception as e:
        logger.error(f"Error merging datasets to {dataset_id}: {e}")
        return error_response(
            message=f"合并数据集失败: {str(e)}",
            code=500,
            error_type="MERGE_ERROR"
        ), 500


# ============================================================================
# 搜索和统计API
# ============================================================================

@dataset_management_bp.route('/search', methods=['GET'])
@jwt_required()
def search_datasets():
    """高级搜索数据集
    
    使用高级搜索条件搜索数据集。
    需要JWT认证。
    
    查询参数:
        q (str): 搜索关键词 (可选)
            - 在名称、描述中搜索
        type (str): 数据集类型 (可选)
        status (str): 状态 (可选)
        tags (str): 标签，逗号分隔 (可选)
        min_size (int): 最小大小(字节) (可选)
        max_size (int): 最大大小(字节) (可选)
        min_records (int): 最小记录数 (可选)
        max_records (int): 最大记录数 (可选)
        created_after (str): 创建时间起始 (ISO8601格式) (可选)
        created_before (str): 创建时间结束 (ISO8601格式) (可选)
        limit (int): 返回数量 (可选，默认50)
        offset (int): 偏移量 (可选，默认0)
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "搜索完成",
                "data": {
                    "datasets": [...],
                    "total_count": "integer",
                    "query": {}
                }
            }
    """
    try:
        current_user_id = get_jwt_identity()
        
        # 收集搜索参数
        search_params = {
            'q': request.args.get('q'),
            'type': request.args.get('type'),
            'status': request.args.get('status'),
            'tags': request.args.get('tags', '').split(',') if request.args.get('tags') else None,
            'min_size': request.args.get('min_size', type=int),
            'max_size': request.args.get('max_size', type=int),
            'min_records': request.args.get('min_records', type=int),
            'max_records': request.args.get('max_records', type=int),
            'created_after': request.args.get('created_after'),
            'created_before': request.args.get('created_before'),
        }
        
        limit = min(int(request.args.get('limit', 50)), 100)
        offset = max(int(request.args.get('offset', 0)), 0)
        
        # 执行搜索
        result = dataset_service.advanced_search(
            user_id=current_user_id,
            search_params=search_params,
            limit=limit,
            offset=offset
        )
        
        return success_response(
            data=result,
            message="搜索完成"
        )
        
    except Exception as e:
        logger.error(f"Error searching datasets: {e}")
        return error_response(
            message=f"搜索失败: {str(e)}",
            code=500,
            error_type="SEARCH_ERROR"
        ), 500


@dataset_management_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_user_statistics():
    """获取用户数据集统计
    
    获取当前用户的数据集统计信息概览。
    需要JWT认证。
    
    请求头:
        Authorization (str): Bearer <access_token>
            - 必填
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取统计成功",
                "data": {
                    "total_datasets": "integer",
                    "total_size_bytes": "integer",
                    "total_size_human": "string",
                    "total_records": "integer",
                    "datasets_by_type": {
                        "text": "integer",
                        "image": "integer"
                    },
                    "datasets_by_status": {
                        "ready": "integer",
                        "processing": "integer"
                    },
                    "recent_uploads": "integer",    // 最近7天
                    "storage_usage_percent": "float"
                }
            }
    """
    try:
        current_user_id = get_jwt_identity()
        
        stats = dataset_service.get_user_statistics(current_user_id)
        
        return success_response(
            data=stats,
            message="获取统计成功"
        )
        
    except Exception as e:
        logger.error(f"Error getting user statistics: {e}")
        return error_response(
            message=f"获取统计失败: {str(e)}",
            code=500,
            error_type="STATISTICS_ERROR"
        ), 500


@dataset_management_bp.route('/recent', methods=['GET'])
@jwt_required()
def get_recent_datasets():
    """获取最近访问的数据集
    
    获取当前用户最近访问的数据集列表。
    需要JWT认证。
    
    查询参数:
        limit (int): 返回数量 (可选，默认10，最大20)
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "获取成功",
                "data": {
                    "datasets": [
                        {
                            "dataset_id": "string",
                            "name": "string",
                            "last_accessed": "string"
                        }
                    ]
                }
            }
    """
    try:
        current_user_id = get_jwt_identity()
        limit = min(int(request.args.get('limit', 10)), 20)
        
        recent = dataset_service.get_recent_datasets(current_user_id, limit)
        
        return success_response(
            data={"datasets": recent},
            message="获取成功"
        )
        
    except Exception as e:
        logger.error(f"Error getting recent datasets: {e}")
        return error_response(
            message=f"获取最近数据集失败: {str(e)}",
            code=500,
            error_type="RECENT_ERROR"
        ), 500


# ============================================================================
# 健康检查API
# ============================================================================

@dataset_management_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查
    
    检查数据集管理服务的运行状态。
    不需要认证。
    
    响应体 (JSON):
        成功 (200):
            {
                "code": 200,
                "message": "服务运行正常",
                "data": {
                    "status": "healthy",
                    "service": "dataset_management",
                    "version": "1.0.0",
                    "timestamp": "string"
                }
            }
    """
    return success_response(
        data={
            "status": "healthy",
            "service": "dataset_management",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat()
        },
        message="服务运行正常"
    )
